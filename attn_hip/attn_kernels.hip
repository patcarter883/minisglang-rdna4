// Native flash-attention PREFILL for gfx1201 (RDNA4) — rocwmma, Triton-free.
//
// Provenance / why this shape:
//   The ALGORITHM (tiled flash-attention with a shared-memory-staged online
//   softmax that decouples the softmax+rescale math from the WMMA fragment
//   layout) is the structure proven in Atlas's AMD port
//   (atlas-ref/kernels/strix-hip/common/inferspark_prefill.cu, AGPL — read for
//   structure, NOT copied). The WMMA itself is DELIBERATELY different:
//
//   Atlas targets gfx1151 (RDNA3.5) with raw `__builtin_amdgcn_wmma_*_w32`
//   intrinsics and a hand-derived C-fragment map `row = 2*e + (lane>>4)`.
//   That map is WRONG on gfx1201 (RDNA4 / gfx12): our own W4A8 kernel
//   (w4a8_fp8_wmma/gemm_tiled.h) uses the `_w32_gfx12` builtins whose
//   accumulator is `row = (lane>>4)*8 + e` — a different layout. Porting the
//   raw intrinsics verbatim is the documented "naive port = garbage" trap.
//
//   So we use rocwmma (rocwmma::load/store/mma_sync), which encapsulates the
//   gfx12 fragment layout. EVERY result that needs per-row reasoning (the QK^T
//   scores, the PV output accumulator) is store_matrix_sync'd to fp32 smem and
//   handled there with explicit (row,col) indices. No fragment-layout
//   assumption survives into our code -> portable + provably indexable.
//
// SCOPE (v0, correctness-first): dense (non-paged) bf16 prefill, causal +
//   optional sliding window, GQA. Q/K/V are contiguous [seq, heads, head_dim].
//   One (q_head, q_block) per CTA. NOT YET: paging, fp8-KV, cp.async-style
//   double buffering, multi-warp-per-mtile widening. Those are perf passes on
//   top of a kernel that must first match torch SDPA bit-for-bit-ish (fp32 acc).
//
// STATUS: compiles-pending / numerics-pending GPU. Validate with
//   attn_hip_parity.py under a 1-card lease before trusting.

#undef __HIP_NO_HALF_CONVERSIONS__
#undef __HIP_NO_HALF_OPERATORS__
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <rocwmma/rocwmma.hpp>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

namespace {

using bf16_t = rocwmma::bfloat16_t;

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;
// Tile sizes are head_dim-dependent. head_dim 64/128 use BR=BC=32 (2 warps), unchanged. head_dim 256
// (Qwen3.5/3.6 full-attn) would blow the 64 KB gfx1201 LDS at BR=BC=32 — the two fp32 [BR][HEAD_DIM]
// smem buffers (smem_O + smem_PV) alone are 64 KB before Q/K/V — so it drops to BR=BC=16 (1 warp,
// ~58 KB), trading occupancy for fit. Same algorithm; all tile counts derive from these.
template <int HEAD_DIM> constexpr int kBR()      { return HEAD_DIM >= 256 ? 16 : 32; }
template <int HEAD_DIM> constexpr int kBC()      { return HEAD_DIM >= 256 ? 16 : 32; }
template <int HEAD_DIM> constexpr int kThreads() { return (kBR<HEAD_DIM>() / WMMA_M) * 32; }

// QK^T fragments: S[m][n] = sum_k Q[m][k] * K[n][k].
//   A = Q  (M x K, row-major)                 : smem_Q[m][k]
//   B = K^T (K x N, col-major over smem_K[n][k]) : a column of B (fixed n,
//       varying k) is the contiguous k-run in row-major smem_K -> col_major.
using FragQ  = rocwmma::fragment<rocwmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, bf16_t, rocwmma::row_major>;
using FragKt = rocwmma::fragment<rocwmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, bf16_t, rocwmma::col_major>;
// PV fragments: O[m][d] = sum_n P[m][n] * V[n][d].
//   A = P (M x N, row-major) : smem_P[m][n]
//   B = V (N x D, row-major) : smem_V[n][d]
using FragP = rocwmma::fragment<rocwmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, bf16_t, rocwmma::row_major>;
using FragV = rocwmma::fragment<rocwmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, bf16_t, rocwmma::row_major>;
using FragAcc = rocwmma::fragment<rocwmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>;

template <int HEAD_DIM>
__global__ __launch_bounds__(kThreads<HEAD_DIM>()) void flash_prefill_kernel(
    const __hip_bfloat16* __restrict__ Q,   // [seq, num_q_heads, HEAD_DIM]
    const __hip_bfloat16* __restrict__ K,   // [seq, num_kv_heads, HEAD_DIM]
    const __hip_bfloat16* __restrict__ V,   // [seq, num_kv_heads, HEAD_DIM]
    __hip_bfloat16* __restrict__ O,         // [seq, num_q_heads, HEAD_DIM]
    const unsigned int seq_len,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const float inv_sqrt_d,
    const unsigned int causal,
    const unsigned int sliding_window,       // 0 = unlimited
    const float* __restrict__ mask_bias) {   // [seq, seq] additive bias, or nullptr
    // mask_bias (TiDAR structured mask etc.): square [seq_len, seq_len] row-major additive bias on
    // the QK^T scores; 0 = allowed, -inf = denied. Composes with causal/SWA (intersection). nullptr
    // => unchanged behaviour. The additive form is validated bit-for-bit against an SDPA reference
    // (zaya/tidar/gpu_validate.py Part B). For TiDAR pass causal=0 and let the bias define the mask.
    constexpr int KSTEPS = HEAD_DIM / WMMA_K;     // QK^T contraction over head_dim
    constexpr int D_TILES = HEAD_DIM / WMMA_N;    // PV output d-tiles
    constexpr int PAD = 8;                        // 16B-aligned smem rows
    constexpr int LDK = HEAD_DIM + PAD;
    // head_dim-dependent tiling (see kBR/kBC/kThreads): 32 for head_dim 64/128, 16 for 256.
    constexpr int BR = kBR<HEAD_DIM>();
    constexpr int BC = kBC<HEAD_DIM>();
    constexpr int M_TILES = BR / WMMA_M;
    constexpr int N_TILES = BC / WMMA_N;
    constexpr int THREADS = kThreads<HEAD_DIM>();

    const unsigned int q_head  = blockIdx.x;
    const unsigned int q_block = blockIdx.y;
    if (q_head >= num_q_heads) return;
    const unsigned int q_start = q_block * BR;
    if (q_start >= seq_len) return;

    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid >> 5;
    const unsigned int q_end = min(q_start + BR, seq_len);
    const unsigned int q_len = q_end - q_start;
    const unsigned int gqa = num_q_heads / num_kv_heads;
    const unsigned int kv_head = q_head / gqa;
    const unsigned int q_stride = num_q_heads * HEAD_DIM;
    const unsigned int kv_stride = num_kv_heads * HEAD_DIM;

    __shared__ bf16_t smem_Q[BR][LDK];
    __shared__ bf16_t smem_K[BC][LDK];
    __shared__ bf16_t smem_V[BC][LDK];
    __shared__ bf16_t smem_P[BR][BC];
    __shared__ float  smem_S[BR][BC];        // raw QK^T scores (then scaled+masked)
    __shared__ float  smem_PV[BR][HEAD_DIM]; // one KV-block's P@V, staged out of acc
    __shared__ float  smem_O[BR][HEAD_DIM];  // running output accumulator (fp32)
    __shared__ float  smem_m[BR];            // online max
    __shared__ float  smem_l[BR];            // online denom
    __shared__ float  smem_resc[BR];         // exp(m_old - m_new) this block

    // init running state + O accumulator
    for (unsigned int r = tid; r < BR; r += THREADS) { smem_m[r] = -1e30f; smem_l[r] = 0.0f; }
    for (unsigned int i = tid; i < BR * HEAD_DIM; i += THREADS)
        ((float*)smem_O)[i] = 0.0f;

    // ---- load Q tile (vec8 bf16 = 16B copies) ----
    constexpr int VEC = 8;
    constexpr int CH_PER_ROW = HEAD_DIM / VEC;
    for (unsigned int idx = tid; idx < BR * CH_PER_ROW; idx += THREADS) {
        unsigned int row = idx / CH_PER_ROW, col = (idx % CH_PER_ROW) * VEC;
        unsigned int gr = q_start + row;
        if (gr < seq_len)
            *(uint4*)&smem_Q[row][col] = *(const uint4*)&Q[gr * q_stride + q_head * HEAD_DIM + col];
        else
            *(uint4*)&smem_Q[row][col] = make_uint4(0, 0, 0, 0);
    }
    __syncthreads();

    unsigned int num_kv_blocks = (seq_len + BC - 1) / BC;
    if (causal) num_kv_blocks = min(num_kv_blocks, (q_end - 1) / BC + 1);

    for (unsigned int kb = 0; kb < num_kv_blocks; kb++) {
        const unsigned int kv_start = kb * BC;
        const unsigned int kv_end = min(kv_start + BC, seq_len);
        const unsigned int kv_len = kv_end - kv_start;

        for (unsigned int idx = tid; idx < BC * CH_PER_ROW; idx += THREADS) {
            unsigned int row = idx / CH_PER_ROW, col = (idx % CH_PER_ROW) * VEC;
            unsigned int gr = kv_start + row;
            if (gr < seq_len) {
                *(uint4*)&smem_K[row][col] = *(const uint4*)&K[gr * kv_stride + kv_head * HEAD_DIM + col];
                *(uint4*)&smem_V[row][col] = *(const uint4*)&V[gr * kv_stride + kv_head * HEAD_DIM + col];
            } else {
                *(uint4*)&smem_K[row][col] = make_uint4(0, 0, 0, 0);
                *(uint4*)&smem_V[row][col] = make_uint4(0, 0, 0, 0);
            }
        }
        __syncthreads();

        // ---- QK^T : warp `warp_id` owns query m-tile `warp_id` ----
        {
            const int mt = warp_id;                 // 0..M_TILES-1
            for (int nt = 0; nt < N_TILES; nt++) {
                FragAcc acc; rocwmma::fill_fragment(acc, 0.0f);
                for (int ks = 0; ks < KSTEPS; ks++) {
                    FragQ  qf; FragKt kf;
                    rocwmma::load_matrix_sync(qf, &smem_Q[mt * WMMA_M][ks * WMMA_K], LDK);
                    rocwmma::load_matrix_sync(kf, &smem_K[nt * WMMA_N][ks * WMMA_K], LDK);
                    rocwmma::mma_sync(acc, qf, kf, acc);
                }
                rocwmma::store_matrix_sync(&smem_S[mt * WMMA_M][nt * WMMA_N], acc, BC,
                                           rocwmma::mem_row_major);
            }
        }
        __syncthreads();

        // ---- online softmax in smem: one thread per query row ----
        if (tid < BR) {
            const unsigned int r = tid;
            const unsigned int qr = q_start + r;
            const bool row_valid = (r < q_len);
            float rmax = -1e30f;
            for (unsigned int c = 0; c < BC; c++) {
                float s = smem_S[r][c] * inv_sqrt_d;
                const unsigned int kpos = kv_start + c;
                bool masked = (c >= kv_len) || !row_valid;
                if (causal && kpos > qr) masked = true;
                if (causal && sliding_window > 0 && kpos <= qr && (qr - kpos) >= sliding_window)
                    masked = true;
                // Structured additive mask (composes with causal/SWA): add the [qr,kpos] bias; a
                // -inf entry denies the cell. Skip when already masked (s pinned to -1e30f below).
                if (mask_bias != nullptr && !masked) s += mask_bias[qr * seq_len + kpos];
                if (masked) s = -1e30f;
                smem_S[r][c] = s;
                rmax = fmaxf(rmax, s);
            }
            const float m_old = smem_m[r];
            const float m_new = fmaxf(m_old, rmax);
            const float resc = __expf(m_old - m_new);
            float sum = 0.0f;
            for (unsigned int c = 0; c < BC; c++) {
                // Round p to bf16 BEFORE accumulating the denominator so the running
                // softmax denom `l` matches the EXACT P fed to the bf16 P@V WMMA below.
                // Accumulating the un-rounded fp32 p here breaks partition-of-unity by the
                // bf16 rounding of P — a ~bf16-eps bias that is invisible when P is spread
                // over many keys (non-causal) but visible when causal masking concentrates
                // P on a few near-1.0 weights (the causal-only parity failure).
                bf16_t pb = (bf16_t)__expf(smem_S[r][c] - m_new);
                smem_P[r][c] = pb;
                sum += (float)pb;
            }
            smem_m[r] = m_new;
            smem_l[r] = smem_l[r] * resc + sum;
            smem_resc[r] = resc;
        }
        __syncthreads();

        // ---- PV : O_block = P @ V  (warp `warp_id` owns m-tile, loops d-tiles) ----
        {
            const int mt = warp_id;
            for (int dt = 0; dt < D_TILES; dt++) {
                FragAcc acc; rocwmma::fill_fragment(acc, 0.0f);
                for (int nt = 0; nt < N_TILES; nt++) {
                    FragP pf; FragV vf;
                    rocwmma::load_matrix_sync(pf, &smem_P[mt * WMMA_M][nt * WMMA_N], BC);
                    rocwmma::load_matrix_sync(vf, &smem_V[nt * WMMA_N][dt * WMMA_N], LDK);
                    rocwmma::mma_sync(acc, pf, vf, acc);
                }
                rocwmma::store_matrix_sync(&smem_PV[mt * WMMA_M][dt * WMMA_N], acc, HEAD_DIM,
                                           rocwmma::mem_row_major);
            }
        }
        __syncthreads();

        // ---- fold: O = O * resc + PV  (one thread per (row, d)) ----
        for (unsigned int i = tid; i < BR * HEAD_DIM; i += THREADS) {
            unsigned int r = i / HEAD_DIM, d = i % HEAD_DIM;
            smem_O[r][d] = smem_O[r][d] * smem_resc[r] + smem_PV[r][d];
        }
        __syncthreads();
    }

    // ---- normalize by l and store ----
    for (unsigned int i = tid; i < BR * HEAD_DIM; i += THREADS) {
        unsigned int r = i / HEAD_DIM, d = i % HEAD_DIM;
        unsigned int gr = q_start + r;
        if (r < q_len && gr < seq_len) {
            float l = smem_l[r];
            float inv_l = (l > 0.0f) ? 1.0f / l : 0.0f;
            O[gr * q_stride + q_head * HEAD_DIM + d] = (__hip_bfloat16)(smem_O[r][d] * inv_l);
        }
    }
}

}  // namespace

// ----------------------------------------------------------------------------
// Launcher (called from bindings.cpp). Dense bf16 prefill, single sequence.
//   q: [seq, num_q_heads, head_dim]  k/v: [seq, num_kv_heads, head_dim]
// ----------------------------------------------------------------------------
void launch_flash_prefill(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                          at::Tensor& out, double scale, int64_t causal,
                          int64_t sliding_window, const float* mask_bias) {
    const unsigned int seq_len = q.size(0);
    const unsigned int num_q_heads = q.size(1);
    const unsigned int head_dim = q.size(2);
    const unsigned int num_kv_heads = k.size(1);
    const float inv_sqrt_d = (scale > 0) ? (float)scale : 1.0f / sqrtf((float)head_dim);

    auto stream = at::cuda::getCurrentCUDAStream();

    const auto* qp = reinterpret_cast<const __hip_bfloat16*>(q.data_ptr());
    const auto* kp = reinterpret_cast<const __hip_bfloat16*>(k.data_ptr());
    const auto* vp = reinterpret_cast<const __hip_bfloat16*>(v.data_ptr());
    auto* op = reinterpret_cast<__hip_bfloat16*>(out.data_ptr());

    // grid/block are head_dim-dependent (BR / THREADS vary), so build them per instantiation.
#define LAUNCH(HD)                                                                      \
    do {                                                                                \
        dim3 grid(num_q_heads, (seq_len + kBR<HD>() - 1) / kBR<HD>(), 1);               \
        dim3 block(kThreads<HD>(), 1, 1);                                               \
        flash_prefill_kernel<HD><<<grid, block, 0, stream>>>(                           \
            qp, kp, vp, op, seq_len, num_q_heads, num_kv_heads, inv_sqrt_d,             \
            (unsigned)causal, (unsigned)sliding_window, mask_bias);                     \
    } while (0); break

    switch (head_dim) {
        case 64:  LAUNCH(64);
        case 128: LAUNCH(128);
        // head_dim 256 (Qwen3.5/3.6 full-attn): BR=BC=16 keeps the two fp32 [BR][HEAD_DIM] smem
        // buffers at 16 KB each (~58 KB total) so it fits the 64 KB gfx1201 LDS. Same algorithm.
        case 256: LAUNCH(256);
        default:
            TORCH_CHECK(false, "attn_hip: unsupported head_dim ", head_dim,
                        " (supports 64/128/256; must be a multiple of 16)");
    }
#undef LAUNCH
}
