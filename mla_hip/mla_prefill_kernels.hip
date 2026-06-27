// Native HIP MLA (Multi-head Latent Attention) PREFILL for gfx1201 (RDNA4) — rocwmma, Triton-free.
//
// The DeepSeek/GLM MLA prefill long-pole. Unlike decode (absorbed, shared latent), prefill runs the
// MATERIALIZED form: the model layer up-projects the latent KV to per-head K (via W_UK) and V (via
// W_UV) and concatenates the decoupled RoPE, so this kernel sees a regular varlen MHA — EXCEPT the
// QK^T contract dim and the PV output dim differ:
//   q = [q_nope (128) ‖ q_rope (64)]   -> qk_head_dim = 192
//   k = [k_nope (128) ‖ k_rope (64)]   -> qk_head_dim = 192   (k_rope broadcast per head, materialized)
//   v =  v_nope (128)                  -> v_head_dim  = 128
//   score = (q · k) * scale            // dot over 192 dims
//   out   = Σ_j softmax_j(score) * v_j // over 128 dims
// num_kv_heads == num_q_heads (each head carries its own materialized K/V), so no GQA folding here.
//
// Reuses attn_prefill_paged's VALIDATED gfx12 WMMA core + smem-staged online softmax, with two changes:
//   1. K/V are PACKED varlen (cu_seqlens_k), not paged — prefill K/V are freshly materialized.
//   2. ASYMMETRIC dims: QK^T contracts over QK_DIM=192 (KSTEPS=12); PV outputs V_DIM=128 (D_TILES=8).
// Causal carries the standard prefix offset (query global pos = prefix_len + local; prefix_len =
// k_len - q_len) so cold prefill (q_len==k_len) AND chunked-extend (prefix>0) both work.
//
// LDS BUDGET (RDNA4 = 64 KB/workgroup): symmetric BR=BC=32 with qk=192/v=128 overflows (~73.6 KB), so
// v0 uses BR=32, BC=16 (~59.8 KB). M_TILES=2 (2 warps), N_TILES=1.
//
// SCOPE (v0, correctness-first): bf16, causal + optional sliding window, qk192/v128. NOT YET: fp8,
// other (qk,v) ratios (add a template instantiation). Validate with mla_hip_parity.py.

#undef __HIP_NO_HALF_CONVERSIONS__
#undef __HIP_NO_HALF_OPERATORS__
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <rocwmma/rocwmma.hpp>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

namespace {

using bf16_t = rocwmma::bfloat16_t;

constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;
constexpr int BR = 32, BC = 16;                 // query rows / key cols per tile (BC=16 to fit LDS)
constexpr int M_TILES = BR / WMMA_M;            // 2
constexpr int N_TILES = BC / WMMA_N;            // 1
constexpr int WARPS = M_TILES;                  // 2
constexpr int THREADS = WARPS * 32;             // 64

using FragQ  = rocwmma::fragment<rocwmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, bf16_t, rocwmma::row_major>;
using FragKt = rocwmma::fragment<rocwmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, bf16_t, rocwmma::col_major>;
using FragP  = rocwmma::fragment<rocwmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, bf16_t, rocwmma::row_major>;
using FragV  = rocwmma::fragment<rocwmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, bf16_t, rocwmma::row_major>;
using FragAcc = rocwmma::fragment<rocwmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>;

template <int QK_DIM, int V_DIM>
__global__ __launch_bounds__(THREADS) void mla_prefill_kernel(
    const __hip_bfloat16* __restrict__ Q,        // [total_q_tokens, num_heads, QK_DIM]  (packed varlen)
    const __hip_bfloat16* __restrict__ K,        // [total_kv_tokens, num_heads, QK_DIM] (packed varlen)
    const __hip_bfloat16* __restrict__ V,        // [total_kv_tokens, num_heads, V_DIM]  (packed varlen)
    const int* __restrict__ cu_seqlens_q,        // [num_seqs+1] : query offsets in packed Q
    const int* __restrict__ cu_seqlens_k,        // [num_seqs+1] : key offsets in packed K/V
    __hip_bfloat16* __restrict__ O,              // [total_q_tokens, num_heads, V_DIM]
    const unsigned int num_heads,
    const float inv_sqrt_d,
    const unsigned int causal,
    const unsigned int sliding_window) {
    constexpr int KSTEPS = QK_DIM / WMMA_K;      // 12
    constexpr int D_TILES = V_DIM / WMMA_N;      // 8
    constexpr int PAD = 8;
    constexpr int LDQ = QK_DIM + PAD;            // 200
    constexpr int LDV = V_DIM + PAD;             // 136
    constexpr int VEC = 8;                       // 8 bf16 = 16 B (uint4)
    constexpr int QK_CH = QK_DIM / VEC;          // 24
    constexpr int V_CH = V_DIM / VEC;            // 16

    const unsigned int head = blockIdx.x;
    const unsigned int q_block = blockIdx.y;
    const unsigned int seq = blockIdx.z;
    if (head >= num_heads) return;

    const unsigned int q_base = cu_seqlens_q[seq];
    const unsigned int q_len = cu_seqlens_q[seq + 1] - q_base;
    const unsigned int q_start = q_block * BR;
    if (q_start >= q_len) return;
    const unsigned int q_rows = min(BR, q_len - q_start);

    const unsigned int k_base = cu_seqlens_k[seq];
    const unsigned int k_len = cu_seqlens_k[seq + 1] - k_base;
    const unsigned int prefix_len = k_len - q_len;   // cached/earlier-chunk prefix before the new tokens

    const unsigned int tid = threadIdx.x, warp_id = tid >> 5;
    const unsigned int q_stride = num_heads * QK_DIM;
    const unsigned int k_stride = num_heads * QK_DIM;
    const unsigned int v_stride = num_heads * V_DIM;

    __shared__ bf16_t smem_Q[BR][LDQ];
    __shared__ bf16_t smem_K[BC][LDQ];
    __shared__ bf16_t smem_V[BC][LDV];
    __shared__ bf16_t smem_P[BR][BC];
    __shared__ float  smem_S[BR][BC];
    __shared__ float  smem_PV[BR][V_DIM];
    __shared__ float  smem_O[BR][V_DIM];
    __shared__ float  smem_m[BR];
    __shared__ float  smem_l[BR];
    __shared__ float  smem_resc[BR];

    for (unsigned int r = tid; r < BR; r += THREADS) { smem_m[r] = -1e30f; smem_l[r] = 0.0f; }
    for (unsigned int i = tid; i < BR * V_DIM; i += THREADS) ((float*)smem_O)[i] = 0.0f;

    // ---- load Q tile (packed varlen) ----
    for (unsigned int idx = tid; idx < BR * QK_CH; idx += THREADS) {
        unsigned int row = idx / QK_CH, col = (idx % QK_CH) * VEC;
        if (row < q_rows) {
            unsigned int gq = q_base + q_start + row;
            *(uint4*)&smem_Q[row][col] = *(const uint4*)&Q[gq * q_stride + head * QK_DIM + col];
        } else {
            *(uint4*)&smem_Q[row][col] = make_uint4(0, 0, 0, 0);
        }
    }
    __syncthreads();

    // causal cap: last query in this tile has global pos prefix_len + q_start + q_rows - 1.
    unsigned int kv_limit = k_len;
    if (causal) kv_limit = min(k_len, prefix_len + q_start + q_rows);
    const unsigned int num_kv_blocks = (kv_limit + BC - 1) / BC;

    for (unsigned int kb = 0; kb < num_kv_blocks; kb++) {
        const unsigned int kv_start = kb * BC;
        const unsigned int kv_len = min(BC, k_len - kv_start);

        // ---- load K tile (packed varlen) ----
        for (unsigned int idx = tid; idx < BC * QK_CH; idx += THREADS) {
            unsigned int row = idx / QK_CH, col = (idx % QK_CH) * VEC;
            unsigned int j = kv_start + row;
            if (j < k_len) {
                unsigned int gk = k_base + j;
                *(uint4*)&smem_K[row][col] = *(const uint4*)&K[gk * k_stride + head * QK_DIM + col];
            } else {
                *(uint4*)&smem_K[row][col] = make_uint4(0, 0, 0, 0);
            }
        }
        // ---- load V tile (packed varlen) ----
        for (unsigned int idx = tid; idx < BC * V_CH; idx += THREADS) {
            unsigned int row = idx / V_CH, col = (idx % V_CH) * VEC;
            unsigned int j = kv_start + row;
            if (j < k_len) {
                unsigned int gk = k_base + j;
                *(uint4*)&smem_V[row][col] = *(const uint4*)&V[gk * v_stride + head * V_DIM + col];
            } else {
                *(uint4*)&smem_V[row][col] = make_uint4(0, 0, 0, 0);
            }
        }
        __syncthreads();

        // ---- QK^T (warp owns query m-tile; contract over QK_DIM) ----
        {
            const int mt = warp_id;
            for (int nt = 0; nt < N_TILES; nt++) {
                FragAcc acc; rocwmma::fill_fragment(acc, 0.0f);
                for (int ks = 0; ks < KSTEPS; ks++) {
                    FragQ qf; FragKt kf;
                    rocwmma::load_matrix_sync(qf, &smem_Q[mt * WMMA_M][ks * WMMA_K], LDQ);
                    rocwmma::load_matrix_sync(kf, &smem_K[nt * WMMA_N][ks * WMMA_K], LDQ);
                    rocwmma::mma_sync(acc, qf, kf, acc);
                }
                rocwmma::store_matrix_sync(&smem_S[mt * WMMA_M][nt * WMMA_N], acc, BC,
                                           rocwmma::mem_row_major);
            }
        }
        __syncthreads();

        // ---- online softmax in smem, one thread per query row, prefix-offset causal ----
        if (tid < BR) {
            const unsigned int r = tid;
            const bool row_valid = (r < q_rows);
            const unsigned int qpos = prefix_len + q_start + r;   // query GLOBAL position
            float rmax = -1e30f;
            for (unsigned int c = 0; c < BC; c++) {
                float s = smem_S[r][c] * inv_sqrt_d;
                const unsigned int kpos = kv_start + c;           // key GLOBAL position
                bool masked = (c >= kv_len) || !row_valid;
                if (causal && kpos > qpos) masked = true;
                if (causal && sliding_window > 0 && kpos <= qpos && (qpos - kpos) >= sliding_window)
                    masked = true;
                if (masked) s = -1e30f;
                smem_S[r][c] = s;
                rmax = fmaxf(rmax, s);
            }
            const float m_old = smem_m[r];
            const float m_new = fmaxf(m_old, rmax);
            const float resc = __expf(m_old - m_new);
            float sum = 0.0f;
            for (unsigned int c = 0; c < BC; c++) {
                bf16_t pb = (bf16_t)__expf(smem_S[r][c] - m_new);  // round P->bf16 before summing l
                smem_P[r][c] = pb;
                sum += (float)pb;
            }
            smem_m[r] = m_new;
            smem_l[r] = smem_l[r] * resc + sum;
            smem_resc[r] = resc;
        }
        __syncthreads();

        // ---- PV (contract over BC keys; output V_DIM) ----
        {
            const int mt = warp_id;
            for (int dt = 0; dt < D_TILES; dt++) {
                FragAcc acc; rocwmma::fill_fragment(acc, 0.0f);
                for (int nt = 0; nt < N_TILES; nt++) {
                    FragP pf; FragV vf;
                    rocwmma::load_matrix_sync(pf, &smem_P[mt * WMMA_M][nt * WMMA_N], BC);
                    rocwmma::load_matrix_sync(vf, &smem_V[nt * WMMA_N][dt * WMMA_N], LDV);
                    rocwmma::mma_sync(acc, pf, vf, acc);
                }
                rocwmma::store_matrix_sync(&smem_PV[mt * WMMA_M][dt * WMMA_N], acc, V_DIM,
                                           rocwmma::mem_row_major);
            }
        }
        __syncthreads();

        for (unsigned int i = tid; i < BR * V_DIM; i += THREADS) {
            unsigned int r = i / V_DIM, d = i % V_DIM;
            smem_O[r][d] = smem_O[r][d] * smem_resc[r] + smem_PV[r][d];
        }
        __syncthreads();
    }

    // ---- normalize + store (packed) ----
    for (unsigned int i = tid; i < BR * V_DIM; i += THREADS) {
        unsigned int r = i / V_DIM, d = i % V_DIM;
        if (r < q_rows) {
            unsigned int gq = q_base + q_start + r;
            float l = smem_l[r];
            float inv_l = (l > 0.0f) ? 1.0f / l : 0.0f;
            O[gq * v_stride + head * V_DIM + d] = (__hip_bfloat16)(smem_O[r][d] * inv_l);
        }
    }
}

}  // namespace

// q:[total_q,H,QK_DIM]  k:[total_kv,H,QK_DIM]  v:[total_kv,H,V_DIM]
// cu_seqlens_q:[S+1] int32  cu_seqlens_k:[S+1] int32  ->  out:[total_q,H,V_DIM]
void launch_mla_prefill(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                        const at::Tensor& cu_seqlens_q, const at::Tensor& cu_seqlens_k,
                        at::Tensor& out, double scale, int64_t causal, int64_t sliding_window,
                        int64_t max_seqlen_q) {
    const unsigned int num_heads = q.size(1);
    const unsigned int qk_dim = q.size(2);
    const unsigned int v_dim = v.size(2);
    const unsigned int num_seqs = cu_seqlens_q.size(0) - 1;
    const float inv = (scale > 0) ? (float)scale : 1.0f / sqrtf((float)qk_dim);

    dim3 grid(num_heads, (max_seqlen_q + BR - 1) / BR, num_seqs);
    dim3 block(THREADS, 1, 1);
    auto stream = at::cuda::getCurrentCUDAStream();

    const auto* qp = reinterpret_cast<const __hip_bfloat16*>(q.data_ptr());
    const auto* kp = reinterpret_cast<const __hip_bfloat16*>(k.data_ptr());
    const auto* vp = reinterpret_cast<const __hip_bfloat16*>(v.data_ptr());
    const int* cqp = cu_seqlens_q.data_ptr<int>();
    const int* ckp = cu_seqlens_k.data_ptr<int>();
    auto* op = reinterpret_cast<__hip_bfloat16*>(out.data_ptr());

#define LAUNCH(QK, VD)                                                                       \
    mla_prefill_kernel<QK, VD><<<grid, block, 0, stream>>>(                                  \
        qp, kp, vp, cqp, ckp, op, num_heads, inv, (unsigned)causal, (unsigned)sliding_window)

    // DeepSeek-V3 / GLM materialized MLA: qk_head_dim=192 (128 nope + 64 rope), v_head_dim=128.
    if (qk_dim == 192 && v_dim == 128) { LAUNCH(192, 128); }
    else {
        TORCH_CHECK(false, "mla_hip prefill: unsupported (qk_head_dim=", qk_dim, ", v_head_dim=", v_dim,
                    ") — add a template instantiation (QK_DIM%16==0, V_DIM%16==0)");
    }
#undef LAUNCH
}
