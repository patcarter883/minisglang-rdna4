// Native RXF ("Rotated eXtra Fast") W4A8 HIP kernels for gfx1201 (RDNA4), Triton-free.
//
// Migrated from the Triton RXF kernels in vllm-gfx1201/paroquant_rotation/rxf_kernels.py and
// built out to mirror the PRODUCTION fp8 W4A8 kernels (w4a8_fp8_wmma/{gemm_tiled.h,
// moe_gemm_tiled.h, moe_kernel.hip}). RXF is W4(NL-codebook)-A8(int8) with a fixed block-diagonal
// Hadamard (FWHT) rotation applied to the activation at runtime and to the weights offline, so the
// rotation cancels in the dot ( (H x) . (H w)^T == x . w^T ) while spreading activation outliers
// across each size-S group to tighten the 4-bit weight scale.
//
// THE fp8 LESSON THAT HOLDS FOR int8: the gfx12 WMMA fragment layout is IDENTICAL between the
// fp8 (f32_16x16x16_fp8) and int8 (i32_16x16x16_iu8) builtins — same operand packing (v2i = 8
// elems/lane, lane&15 = row/col, (lane>>4)*8 = k-half) and same accumulator layout
// (row = (lane>>4)*8 + e, col = lane&15). The fp8 kernel is validated against this layout, so we
// reuse it verbatim with the int8 builtin: register-resident int32 accumulation (no smem
// round-trip), BMxBN multi-warp K-pipelined tiles, group-32 weight scale folded between groups,
// per-token act scale in the epilogue. Only fp8->int8 staging (NL unpack vs e4m3 decode) and the
// accumulator dtype (int32 vs float) differ.
//
// Numerics matched bit-for-bit to the Triton reference (rxf_hip_parity.py / tools/rxf_parity.py):
//   FWHT  : in-place decimation butterfly, partner at distance h for h in {1,2,...,S/2};
//           normalize by 0.1767766953 for S==32 (== 1/sqrt(32)).
//   quant : per-token symmetric int8; scale = absmax/127; q = round(x*127/absmax) in [-127,127].
//   NL    : weight_packed uint8 [N,K/2], low nibble = even channel, high = odd; index -> nl[idx]
//           int8 codebook; per-group fp16 weight_scale [N,K/32]; per-token act scale once at end.

#undef __HIP_NO_HALF_CONVERSIONS__
#undef __HIP_NO_HALF_OPERATORS__
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

namespace {

using bf16 = __hip_bfloat16;

typedef int v2i __attribute__((ext_vector_type(2)));  // WMMA A/B operand: 8 int8 / lane
typedef int v8i __attribute__((ext_vector_type(8)));  // WMMA C accumulator: 8 int32 / lane

constexpr int WMMA_DIM = 16;
constexpr int GROUP = 32;      // RXF group == 2 WMMA K-steps (no K=32 int8 WMMA on gfx12)
constexpr int LDS_PAD = 8;     // bank-conflict pad on the LDS K stride (== fp8 kernel's pad)

// int8 WMMA: C += A . B, signed operands, int32 accumulate. Same gfx12 fragment layout as the
// validated fp8 path's f32_16x16x16_fp8 builtin (only the element/acc dtype differs).
struct MMA {
    using AFrag = v2i;
    using BFrag = v2i;
    using CFrag = v8i;
    static __device__ __forceinline__ CFrag zero() { return CFrag{0, 0, 0, 0, 0, 0, 0, 0}; }
    static __device__ __forceinline__ CFrag mma(AFrag a, BFrag b, CFrag c) {
        return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a, true, b, c, false);
    }
};

// ----------------------------------------------------------------------------------------
// 1. rotate_quant_int8 : block-diagonal FWHT-S over each size-S group of a row, then a
//    per-token symmetric int8 quant. One block per row; each thread owns whole S-groups
//    (S floats in registers -> no intra-group sync), rotated values staged in dynamic smem
//    for the row absmax reduction + quantize passes.
// ----------------------------------------------------------------------------------------

__device__ __forceinline__ float block_reduce_max(float v, float* shared /*[32]*/) {
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor(v, o));
    if (lane == 0) shared[wid] = v;
    __syncthreads();
    const int nwarps = (blockDim.x + 31) >> 5;
    v = (threadIdx.x < nwarps) ? shared[lane] : 0.0f;
    if (wid == 0) {
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor(v, o));
        if (lane == 0) shared[0] = v;
    }
    __syncthreads();
    return shared[0];
}

template <typename scalar_t, int SPAN>
__global__ void rotate_quant_kernel(
    const scalar_t* __restrict__ x,   // [M, K]
    int8_t* __restrict__ q,           // [M, K]
    float* __restrict__ scale,        // [M]
    int M, int K, float norm) {
    extern __shared__ float srow[];   // [K] rotated row, fp32
    __shared__ float red[32];
    const int row = blockIdx.x;
    if (row >= M) return;
    const scalar_t* xr = x + (long)row * K;
    const int nG = K / SPAN;

    for (int g = threadIdx.x; g < nG; g += blockDim.x) {
        float buf[SPAN];
        const int base = g * SPAN;
        #pragma unroll
        for (int i = 0; i < SPAN; ++i) buf[i] = (float)xr[base + i];
        #pragma unroll
        for (int h = 1; h < SPAN; h <<= 1) {
            #pragma unroll
            for (int i = 0; i < SPAN; ++i) {
                if ((i & h) == 0) {
                    const float a = buf[i], b = buf[i | h];
                    buf[i] = a + b;
                    buf[i | h] = a - b;
                }
            }
        }
        #pragma unroll
        for (int i = 0; i < SPAN; ++i) srow[base + i] = buf[i] * norm;
    }
    __syncthreads();

    float local = 0.0f;
    for (int i = threadIdx.x; i < K; i += blockDim.x) local = fmaxf(local, fabsf(srow[i]));
    const float absmax = fmaxf(block_reduce_max(local, red), 1e-12f);
    const float inv = 127.0f / absmax;
    if (threadIdx.x == 0) scale[row] = absmax / 127.0f;

    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        float v = roundf(srow[i] * inv);
        v = fminf(fmaxf(v, -127.0f), 127.0f);
        q[(long)row * K + i] = (int8_t)v;
    }
}

// ----------------------------------------------------------------------------------------
// 2. dense linear (BMxBN multi-warp WMMA, mirror of gemm_tiled_kernel).
//    Consumes pre-rotated/quantized int8 q + per-token a_scale (the rotation lives in
//    rotate_quant_int8, since a cross-channel FWHT can't be folded into per-element staging).
//    NWARPS = BM/16 warps, each owns NFRAG = BN/16 N-fragments; per group the int32 partial is
//    folded into a per-fragment fp32 `running` accumulator scaled by the group's fp16 weight
//    scale; per-token act scale is applied in the epilogue (== fp8 structure).
// ----------------------------------------------------------------------------------------

// WARPS_N warps split the BN columns (each owns NFRAG/WARPS_N frags), on top of BM/16 M-warps that
// split the rows -> (BM/16)*WARPS_N warps fill the WGP with NFRAG_W (not NFRAG) accumulators per
// thread (less register pressure -> more blocks/CU). Same WARPS_N occupancy idea as the MoE kernel.
template <int BM, int BN, int BK_TILE, int WARPS_N>
__global__ void __launch_bounds__((BM / 16) * WARPS_N * 32) linear_tiled_kernel(
    const int8_t* __restrict__ q,        // [M, K]
    const float* __restrict__ a_scale,   // [M]
    const uint8_t* __restrict__ wpk,     // [N, K/2]
    const __half* __restrict__ wsc,      // [N, K/32]
    const int8_t* __restrict__ nl,       // [16]
    const float* __restrict__ bias,      // [N] or nullptr
    bf16* __restrict__ out,              // [M, N]
    int M, int N, int K) {
    constexpr int NWARPS_M = BM / 16;
    constexpr int NFRAG = BN / WMMA_DIM;
    constexpr int NFRAG_W = NFRAG / WARPS_N;
    constexpr int GPC = BK_TILE / GROUP;       // scale-groups staged per sync (decoupled from sync)
    constexpr int LDSBK = BK_TILE + LDS_PAD;
    constexpr int THREADS = NWARPS_M * WARPS_N * 32;

    const int block_m = blockIdx.y * BM;
    const int block_n = blockIdx.x * BN;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int lrow = lane & 15;
    const int khalf = (lane >> 4) * 8;
    const int frag_col = lane & 15;
    const int frag_row0 = (lane >> 4) * 8;
    const int warp_m = warp_id / WARPS_N;        // which 16-row M-subtile
    const int warp_n = warp_id % WARPS_N;        // which BN/WARPS_N column slab
    const int nfrag0 = warp_n * NFRAG_W;         // first global N-frag for this warp

    const int num_k_groups = K / GROUP;
    const int kpr = K / 2;               // packed bytes per weight row
    const int num_chunks = K / BK_TILE;

    __shared__ int8_t sNL[16];
    __shared__ int8_t A_tile[BM * LDSBK];
    __shared__ int8_t B_tile[BN * LDSBK];
    __shared__ float act_scale_s[BM];
    if (tid < 16) sNL[tid] = nl[tid];
    for (int r = tid; r < BM; r += THREADS) {
        const int am = block_m + r;
        act_scale_s[r] = (am < M) ? a_scale[am] : 0.0f;
    }

    float running[NFRAG_W][8];
    #pragma unroll
    for (int f = 0; f < NFRAG_W; ++f)
        #pragma unroll
        for (int e = 0; e < 8; ++e) running[f][e] = 0.0f;

    for (int c = 0; c < num_chunks; ++c) {
        const int k0 = c * BK_TILE;
        // stage a BK_TILE chunk of activations + unpacked weights once per sync.
        for (int idx = tid * 4; idx < BM * BK_TILE; idx += THREADS * 4) {
            const int r = idx / BK_TILE, k = idx % BK_TILE, am = block_m + r;
            unsigned int v = 0u;
            if (am < M) v = *reinterpret_cast<const unsigned int*>(&q[(long)am * K + k0 + k]);
            *reinterpret_cast<unsigned int*>(&A_tile[r * LDSBK + k]) = v;
        }
        // vectorized weight staging: one uint32 (8 nibbles) per iter -> 8 NL codes -> 8 int8
        // (2 words), matching the fp8 kernel's int32-granular staging. Columns an>=N write
        // sNL[0] but are masked by wsc_v==0 below, so the garbage never contributes.
        const uint32_t* wpk32 = reinterpret_cast<const uint32_t*>(wpk);
        const long wpr32 = (long)kpr / 4;       // K/8 words per weight row
        const int njobs = BN * (BK_TILE / 8);
        for (int j = tid; j < njobs; j += THREADS) {
            const int n = j / (BK_TILE / 8), wj = j % (BK_TILE / 8);
            const int an = block_n + n;
            uint32_t word = (an < N) ? wpk32[(long)an * wpr32 + (k0 >> 3) + wj] : 0u;
            uint32_t lo = 0, hi = 0;
            #pragma unroll
            for (int t = 0; t < 4; ++t) lo |= ((uint32_t)(uint8_t)sNL[(word >> (t * 4)) & 0xF]) << (t * 8);
            #pragma unroll
            for (int t = 0; t < 4; ++t) hi |= ((uint32_t)(uint8_t)sNL[(word >> ((t + 4) * 4)) & 0xF]) << (t * 8);
            uint32_t* dst = reinterpret_cast<uint32_t*>(&B_tile[n * LDSBK + wj * 8]);
            dst[0] = lo;
            dst[1] = hi;
        }
        __syncthreads();

        // GPC scale-groups within the chunk; each accumulates int32 over its 2 K-steps then folds
        // into `running` with its own fp16 weight scale.
        #pragma unroll
        for (int gi = 0; gi < GPC; ++gi) {
            const int g = c * GPC + gi;
            typename MMA::CFrag acc[NFRAG_W];
            #pragma unroll
            for (int f = 0; f < NFRAG_W; ++f) acc[f] = MMA::zero();
            const int a_base = (warp_m * WMMA_DIM + lrow) * LDSBK + khalf + gi * GROUP;
            const int b_base = lrow * LDSBK + khalf + gi * GROUP;
            #pragma unroll
            for (int kt = 0; kt < GROUP / WMMA_DIM; ++kt) {
                const int ko = kt * WMMA_DIM;
                typename MMA::AFrag a_cur =
                    *reinterpret_cast<const typename MMA::AFrag*>(&A_tile[a_base + ko]);
                #pragma unroll
                for (int f = 0; f < NFRAG_W; ++f) {
                    const int gf = nfrag0 + f;
                    typename MMA::BFrag b_cur = *reinterpret_cast<const typename MMA::BFrag*>(
                        &B_tile[b_base + gf * WMMA_DIM * LDSBK + ko]);
                    acc[f] = MMA::mma(a_cur, b_cur, acc[f]);
                }
            }
            #pragma unroll
            for (int f = 0; f < NFRAG_W; ++f) {
                const int abs_n = block_n + (nfrag0 + f) * WMMA_DIM + frag_col;
                const float wsc_v = (abs_n < N) ? __half2float(wsc[(long)abs_n * num_k_groups + g]) : 0.0f;
                #pragma unroll
                for (int e = 0; e < 8; ++e) running[f][e] += (float)acc[f][e] * wsc_v;
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int e = 0; e < 8; ++e) {
        const int lrow_m = warp_m * WMMA_DIM + frag_row0 + e;
        const int abs_m = block_m + lrow_m;
        if (abs_m >= M) continue;
        const float asc = act_scale_s[lrow_m];
        #pragma unroll
        for (int f = 0; f < NFRAG_W; ++f) {
            const int abs_n = block_n + (nfrag0 + f) * WMMA_DIM + frag_col;
            if (abs_n < N) {
                float val = running[f][e] * asc;
                if (bias) val += bias[abs_n];
                out[(long)abs_m * N + abs_n] = (bf16)val;
            }
        }
    }
}

// Decode (M<=2) scalar GEMV: int8 . NL-int4 dot, per-token act scale.
__global__ void linear_gemv_kernel(
    const int8_t* __restrict__ q, const float* __restrict__ a_scale,
    const uint8_t* __restrict__ wpk, const __half* __restrict__ wsc,
    const int8_t* __restrict__ nl, const float* __restrict__ bias,
    bf16* __restrict__ out, int M, int N, int K) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y;
    __shared__ int8_t sNL[16];
    if (threadIdx.x < 16) sNL[threadIdx.x] = nl[threadIdx.x];
    __syncthreads();
    if (n >= N || m >= M) return;
    const uint8_t* wn = wpk + (long)n * (K / 2);
    const __half* wsn = wsc + (long)n * (K / GROUP);
    const int8_t* qm = q + (long)m * K;
    float acc = 0.0f;
    for (int g = 0; g < K / GROUP; ++g) {
        const int kb = g * GROUP;
        int part = 0;
        #pragma unroll
        for (int kk = 0; kk < GROUP; ++kk) {
            const uint8_t byte = wn[(kb + kk) >> 1];
            const int nib = (kk & 1) ? (byte >> 4) : (byte & 0x0F);
            part += (int)qm[kb + kk] * (int)sNL[nib];
        }
        acc += (float)part * __half2float(wsn[g]);
    }
    acc *= a_scale[m];
    if (bias) acc += bias[n];
    out[(long)m * N + n] = (bf16)acc;
}

// ----------------------------------------------------------------------------------------
// 3. grouped MoE (BN-tiled WMMA, mirror of moe_gemm_tiled_kernel v5). SCATTER=false stores
//    (P,N) bf16; SCATTER=true fuses gemm2's topk-weighted atomic scatter into an (M,K) fp32
//    accumulator (== mmq_fp8_moe_gemm_scatter). block_m tokens/expert -> n_warps = block_m/16.
// ----------------------------------------------------------------------------------------

// WARPS_N warps split the BN columns (each owns NFRAG_W = NFRAG/WARPS_N fragments); n_warps_m =
// block_m/16 warps split the token-rows. So block_m stays 16 (minimal padding) while the block runs
// n_warps_m*WARPS_N warps -> full WGP occupancy and NFRAG_W (not NFRAG) accumulators per thread
// (4x less register pressure than the 1-warp-does-all-N design). All warps cooperate on staging.
template <int BN, int BK_TILE, int WARPS_N, bool SCATTER>
__global__ void __launch_bounds__(256) moe_gemm_tiled_kernel(
    const int8_t* __restrict__ q,        // [T, K] (gemm1) / [P, K] (gemm2 ident)
    const float* __restrict__ a_scale,   // [T] / [P]
    const uint8_t* __restrict__ wpk,     // [E, N, K/2]
    const __half* __restrict__ wsc,      // [E, N, K/32]
    const int8_t* __restrict__ nl,       // [16]
    const int* __restrict__ sorted_ids,  // [P]
    const int* __restrict__ expert_ids,  // [P/block_m]
    const int* __restrict__ ntp,         // [1]
    bf16* __restrict__ out,              // [P, N] (non-scatter)
    float* __restrict__ out_scatter,     // [M, N] fp32 (scatter)
    const float* __restrict__ topk_weights,  // [M*top_k] (scatter)
    int N, int K, int top_k, int block_m, int num_valid_tokens) {
    constexpr int NFRAG = BN / WMMA_DIM;
    constexpr int NFRAG_W = NFRAG / WARPS_N;     // N-fragments owned by each N-warp
    constexpr int GPC = BK_TILE / GROUP;
    constexpr int LDSBK = BK_TILE + LDS_PAD;

    const int block_idx = blockIdx.y;
    const int row0 = block_idx * block_m;
    if (row0 >= ntp[0]) return;

    const int block_n = blockIdx.x * BN;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int lrow = lane & 15;
    const int khalf = (lane >> 4) * 8;
    const int frag_col = lane & 15;
    const int frag_row0 = (lane >> 4) * 8;
    const int n_warps_m = block_m / WMMA_DIM;
    const int warp_m = warp_id / WARPS_N;        // which 16-row token subtile
    const int warp_n = warp_id % WARPS_N;        // which BN/WARPS_N column slab
    if (warp_m >= n_warps_m) return;
    const int nfrag0 = warp_n * NFRAG_W;         // first global N-frag for this warp

    const int e = expert_ids[block_idx];
    const int num_groups = K / GROUP;
    const long ppr = K / 2;
    const int num_chunks = K / BK_TILE;
    const uint8_t* wq_e = wpk + (long)e * N * ppr;
    const __half* ws_e = wsc + (long)e * N * num_groups;

    extern __shared__ int8_t smem[];
    int8_t* A_tile = smem;                       // block_m rows
    int8_t* B_tile = smem + block_m * LDSBK;     // BN cols
    __shared__ int8_t sNL[16];
    if (tid < 16) sNL[tid] = nl[tid];

    float running[NFRAG_W][8];
    #pragma unroll
    for (int f = 0; f < NFRAG_W; ++f)
        #pragma unroll
        for (int ee = 0; ee < 8; ++ee) running[f][ee] = 0.0f;

    const int total_threads = n_warps_m * WARPS_N * 32;
    for (int c = 0; c < num_chunks; ++c) {
        const int k0 = c * BK_TILE;
        for (int idx = tid * 4; idx < block_m * BK_TILE; idx += total_threads * 4) {
            const int r = idx / BK_TILE, k = idx % BK_TILE;
            unsigned int v = 0u;
            const int offs_token = sorted_ids[row0 + r];
            if (offs_token < num_valid_tokens) {
                const int src = SCATTER ? (row0 + r) : (offs_token / top_k);
                v = *reinterpret_cast<const unsigned int*>(&q[(long)src * K + k0 + k]);
            }
            *reinterpret_cast<unsigned int*>(&A_tile[r * LDSBK + k]) = v;
        }
        const uint32_t* wq32 = reinterpret_cast<const uint32_t*>(wq_e);
        const long wpr32 = ppr / 4;             // K/8 words per weight row
        const int njobs = BN * (BK_TILE / 8);
        for (int j = tid; j < njobs; j += total_threads) {
            const int n = j / (BK_TILE / 8), wj = j % (BK_TILE / 8);
            const int an = block_n + n;
            uint32_t word = (an < N) ? wq32[(long)an * wpr32 + (k0 >> 3) + wj] : 0u;
            uint32_t lo = 0, hi = 0;
            #pragma unroll
            for (int t = 0; t < 4; ++t) lo |= ((uint32_t)(uint8_t)sNL[(word >> (t * 4)) & 0xF]) << (t * 8);
            #pragma unroll
            for (int t = 0; t < 4; ++t) hi |= ((uint32_t)(uint8_t)sNL[(word >> ((t + 4) * 4)) & 0xF]) << (t * 8);
            uint32_t* dst = reinterpret_cast<uint32_t*>(&B_tile[n * LDSBK + wj * 8]);
            dst[0] = lo;
            dst[1] = hi;
        }
        __syncthreads();

        #pragma unroll
        for (int gi = 0; gi < GPC; ++gi) {
            const int g = c * GPC + gi;
            typename MMA::CFrag acc[NFRAG_W];
            #pragma unroll
            for (int f = 0; f < NFRAG_W; ++f) acc[f] = MMA::zero();
            const int a_base = (warp_m * WMMA_DIM + lrow) * LDSBK + khalf + gi * GROUP;
            const int b_base = lrow * LDSBK + khalf + gi * GROUP;
            #pragma unroll
            for (int kt = 0; kt < GROUP / WMMA_DIM; ++kt) {
                const int ko = kt * WMMA_DIM;
                typename MMA::AFrag a_cur =
                    *reinterpret_cast<const typename MMA::AFrag*>(&A_tile[a_base + ko]);
                #pragma unroll
                for (int f = 0; f < NFRAG_W; ++f) {
                    const int gf = nfrag0 + f;
                    typename MMA::BFrag b_cur = *reinterpret_cast<const typename MMA::BFrag*>(
                        &B_tile[b_base + gf * WMMA_DIM * LDSBK + ko]);
                    acc[f] = MMA::mma(a_cur, b_cur, acc[f]);
                }
            }
            #pragma unroll
            for (int f = 0; f < NFRAG_W; ++f) {
                const int abs_n = block_n + (nfrag0 + f) * WMMA_DIM + frag_col;
                const float wsc_v = (abs_n < N) ? __half2float(ws_e[(long)abs_n * num_groups + g]) : 0.0f;
                #pragma unroll
                for (int ee = 0; ee < 8; ++ee) running[f][ee] += (float)acc[f][ee] * wsc_v;
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int ee = 0; ee < 8; ++ee) {
        const int r = warp_m * WMMA_DIM + frag_row0 + ee;
        if (r >= block_m) continue;
        const int row_pad = row0 + r;
        const int offs_token = sorted_ids[row_pad];
        if (offs_token >= num_valid_tokens) continue;
        const int src = SCATTER ? row_pad : (offs_token / top_k);
        const float asc = a_scale[src];
        if constexpr (SCATTER) {
            const int token = offs_token / top_k;
            const float w = topk_weights[offs_token];
            #pragma unroll
            for (int f = 0; f < NFRAG_W; ++f) {
                const int abs_n = block_n + (nfrag0 + f) * WMMA_DIM + frag_col;
                if (abs_n < N)
                    atomicAdd(&out_scatter[(long)token * N + abs_n], w * running[f][ee] * asc);
            }
        } else {
            #pragma unroll
            for (int f = 0; f < NFRAG_W; ++f) {
                const int abs_n = block_n + (nfrag0 + f) * WMMA_DIM + frag_col;
                if (abs_n < N) out[(long)row_pad * N + abs_n] = (bf16)(running[f][ee] * asc);
            }
        }
    }
}

// grouped MoE GEMV for gemm1 at decode (M<=2), mirroring the fp8 per-GEMM kernel selection.
// One warp per output column; the activation row is staged in LDS once and shared across all
// columns. With GROUP==32==warp width, lane l owns k=g*32+l of group g, so a single warp-reduce
// is the exact group dot -> per-group fp16 scale -> fp32 acc. Padding rows early-exit the whole
// block (one block per padded row), so the ~16x WMMA-on-padding waste at decode is gone.
__global__ void moe_gemv_kernel(
    const int8_t* __restrict__ q,        // [T, K]
    const float* __restrict__ a_scale,   // [T]
    const uint8_t* __restrict__ wpk,     // [E, N, K/2]
    const __half* __restrict__ wsc,      // [E, N, K/32]
    const int8_t* __restrict__ nl,       // [16]
    const int* __restrict__ sorted_ids,  // [P]
    const int* __restrict__ expert_ids,  // [P/block_m]
    const int* __restrict__ ntp,         // [1]
    bf16* __restrict__ out,              // [P, N]
    int N, int K, int top_k, int block_m, int num_valid_tokens, int COLS) {
    const int row_pad = blockIdx.y;
    if (row_pad >= ntp[0]) return;
    const int offs = sorted_ids[row_pad];
    if (offs >= num_valid_tokens) return;       // padding row -> whole block exits
    const int src = offs / top_k;
    const int e = expert_ids[row_pad / block_m];
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

    extern __shared__ int8_t qs[];              // [K] staged activation row
    __shared__ int8_t sNL[16];
    if (threadIdx.x < 16) sNL[threadIdx.x] = nl[threadIdx.x];
    for (int i = threadIdx.x; i < K; i += blockDim.x) qs[i] = q[(long)src * K + i];
    __syncthreads();

    const int col = blockIdx.x * COLS + warp;
    if (col >= N) return;
    const uint8_t* wq = wpk + ((long)e * N + col) * (K / 2);
    const __half* wsn = wsc + ((long)e * N + col) * (K / GROUP);
    float acc = 0.0f;
    for (int g = 0; g < K / GROUP; ++g) {
        const int kk = g * GROUP + lane;
        const uint8_t byte = wq[kk >> 1];
        const int nib = (lane & 1) ? (byte >> 4) : (byte & 0x0F);
        int prod = (int)qs[kk] * (int)sNL[nib];
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) prod += __shfl_xor(prod, o);
        acc += (float)prod * __half2float(wsn[g]);
    }
    acc *= a_scale[src];
    if (lane == 0) out[(long)row_pad * N + col] = (bf16)acc;
}

// gemm2 epilogue (non-scatter prefill): gather (P,N) by sorted_ids, topk-weight, reduce -> (M,N).
__global__ void moe_gather_reduce_kernel(
    const bf16* __restrict__ out2,       // [P, N]
    const int* __restrict__ sorted_ids,  // [P]
    const float* __restrict__ topk_weights,  // [M*top_k]
    const int* __restrict__ ntp,         // [1]
    float* __restrict__ out,             // [M, N] fp32, pre-zeroed
    int P, int N, int top_k, int num_valid_tokens) {
    const int r = blockIdx.y;
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N || r >= ntp[0]) return;
    const int offs = sorted_ids[r];
    if (offs >= num_valid_tokens) return;
    const int token = offs / top_k;
    const float w = topk_weights[offs];
    atomicAdd(&out[(long)token * N + n], w * (float)out2[(long)r * N + n]);
}

}  // namespace

// ============================== launchers ==============================

void launch_rotate_quant(const at::Tensor& x, at::Tensor& q, at::Tensor& scale, int64_t span) {
    const int K = x.size(-1);
    const int M = x.numel() / K;
    TORCH_CHECK(span == 32, "rxf_hip: only Hadamard span 32 is supported (this migration)");
    TORCH_CHECK(K % span == 0, "rxf_hip: K must be a multiple of span");
    const float norm = 0.1767766953f;  // == 1/sqrt(32), matched to the Triton reference literal
    const int threads = 256;
    const size_t shmem = (size_t)K * sizeof(float);
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "rxf_rotate_quant", [&] {
        rotate_quant_kernel<scalar_t, 32><<<M, threads, shmem, stream>>>(
            x.data_ptr<scalar_t>(), q.data_ptr<int8_t>(), scale.data_ptr<float>(), M, K, norm);
    });
}

void launch_linear(const at::Tensor& q, const at::Tensor& a_scale, const at::Tensor& wpk,
                   const at::Tensor& wsc, const at::Tensor& nl, const at::Tensor& bias,
                   at::Tensor& out, bool has_bias) {
    const int M = q.size(0), K = q.size(1), N = wpk.size(0);
    auto stream = at::cuda::getCurrentCUDAStream();
    const float* bias_ptr = has_bias ? bias.data_ptr<float>() : nullptr;
    const __half* wsc_ptr = reinterpret_cast<const __half*>(wsc.data_ptr());
    bf16* out_ptr = reinterpret_cast<bf16*>(out.data_ptr());
    if (M <= 2) {
        constexpr int TPB = 256;
        dim3 grid((N + TPB - 1) / TPB, M);
        linear_gemv_kernel<<<grid, TPB, 0, stream>>>(
            q.data_ptr<int8_t>(), a_scale.data_ptr<float>(), wpk.data_ptr<uint8_t>(), wsc_ptr,
            nl.data_ptr<int8_t>(), bias_ptr, out_ptr, M, N, K);
    } else {
        constexpr int BM = 64, BN = 128, WARPS_N = 2;  // 4 M-warps x 2 N-warps = 8 warps (256 thr)
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
        const int threads = (BM / 16) * WARPS_N * 32;
        // BK_TILE decouples the LDS staging chunk from the 32-wide scale group (== fp8's
        // group_size staging): 128 when K divides it, else fall back to 32.
        if (K % 128 == 0)
            linear_tiled_kernel<BM, BN, 128, WARPS_N><<<grid, threads, 0, stream>>>(
                q.data_ptr<int8_t>(), a_scale.data_ptr<float>(), wpk.data_ptr<uint8_t>(), wsc_ptr,
                nl.data_ptr<int8_t>(), bias_ptr, out_ptr, M, N, K);
        else
            linear_tiled_kernel<BM, BN, 32, WARPS_N><<<grid, threads, 0, stream>>>(
                q.data_ptr<int8_t>(), a_scale.data_ptr<float>(), wpk.data_ptr<uint8_t>(), wsc_ptr,
                nl.data_ptr<int8_t>(), bias_ptr, out_ptr, M, N, K);
    }
}

static void launch_moe_common(const at::Tensor& q, const at::Tensor& a_scale,
                              const at::Tensor& wpk, const at::Tensor& wsc, const at::Tensor& nl,
                              const at::Tensor& sorted_ids, const at::Tensor& expert_ids,
                              const at::Tensor& ntp, bf16* out, float* out_scatter,
                              const float* topk_w, int N, int K, int top_k, int block_m,
                              int num_valid, bool scatter) {
    constexpr int BN = 128;
    const int P = sorted_ids.size(0);
    // Decode (scatter, few real rows) is occupancy-bound -> keep LDS small (BK_TILE=32). Prefill
    // (non-scatter, many token-blocks) has occupancy to spare -> the wide 128 chunk cuts syncs.
    const int BK_TILE = (scatter || K % 128 != 0) ? 32 : 128;
    // WARPS_N warps split BN to fill the WGP at block_m=16 (NFRAG=BN/16=8 -> NFRAG_W=2 each).
    constexpr int WARPS_N = 4;
    const size_t shmem = (size_t)(block_m + BN) * (BK_TILE + LDS_PAD) * sizeof(int8_t);
    dim3 grid((N + BN - 1) / BN, (P + block_m - 1) / block_m);
    const int threads = (block_m / WMMA_DIM) * WARPS_N * 32;
    auto stream = at::cuda::getCurrentCUDAStream();
    auto* qp = q.data_ptr<int8_t>();
    auto* asp = a_scale.data_ptr<float>();
    auto* wp = wpk.data_ptr<uint8_t>();
    auto* wsp = reinterpret_cast<const __half*>(wsc.data_ptr());
    auto* nlp = nl.data_ptr<int8_t>();
    auto* sip = sorted_ids.data_ptr<int>();
    auto* eip = expert_ids.data_ptr<int>();
    auto* ntpp = ntp.data_ptr<int>();
    #define RXF_MOE_LAUNCH(BKT, SCAT)                                                          \
        moe_gemm_tiled_kernel<BN, BKT, WARPS_N, SCAT><<<grid, threads, shmem, stream>>>(      \
            qp, asp, wp, wsp, nlp, sip, eip, ntpp, out, out_scatter, topk_w,                  \
            N, K, top_k, block_m, num_valid)
    if (scatter) {
        if (BK_TILE == 128) RXF_MOE_LAUNCH(128, true); else RXF_MOE_LAUNCH(32, true);
    } else {
        if (BK_TILE == 128) RXF_MOE_LAUNCH(128, false); else RXF_MOE_LAUNCH(32, false);
    }
    #undef RXF_MOE_LAUNCH
}

void launch_moe_gemm(const at::Tensor& q, const at::Tensor& a_scale, const at::Tensor& wpk,
                     const at::Tensor& wsc, const at::Tensor& nl, const at::Tensor& sorted_ids,
                     const at::Tensor& expert_ids, const at::Tensor& ntp, at::Tensor& out,
                     int64_t top_k, int64_t block_m, int64_t num_valid_tokens) {
    launch_moe_common(q, a_scale, wpk, wsc, nl, sorted_ids, expert_ids, ntp,
                      reinterpret_cast<bf16*>(out.data_ptr()), nullptr, nullptr,
                      wpk.size(1), wpk.size(2) * 2, (int)top_k, (int)block_m,
                      (int)num_valid_tokens, false);
}

void launch_moe_gemv(const at::Tensor& q, const at::Tensor& a_scale, const at::Tensor& wpk,
                     const at::Tensor& wsc, const at::Tensor& nl, const at::Tensor& sorted_ids,
                     const at::Tensor& expert_ids, const at::Tensor& ntp, at::Tensor& out,
                     int64_t top_k, int64_t block_m, int64_t num_valid_tokens) {
    const int N = wpk.size(1), K = wpk.size(2) * 2, P = sorted_ids.size(0);
    constexpr int COLS = 8;                     // warps (=columns) per block
    const size_t shmem = (size_t)K * sizeof(int8_t);
    dim3 grid((N + COLS - 1) / COLS, P);
    auto stream = at::cuda::getCurrentCUDAStream();
    moe_gemv_kernel<<<grid, COLS * 32, shmem, stream>>>(
        q.data_ptr<int8_t>(), a_scale.data_ptr<float>(), wpk.data_ptr<uint8_t>(),
        reinterpret_cast<const __half*>(wsc.data_ptr()), nl.data_ptr<int8_t>(),
        sorted_ids.data_ptr<int>(), expert_ids.data_ptr<int>(), ntp.data_ptr<int>(),
        reinterpret_cast<bf16*>(out.data_ptr()), N, K, (int)top_k, (int)block_m,
        (int)num_valid_tokens, COLS);
}

void launch_moe_gemm_scatter(const at::Tensor& q, const at::Tensor& a_scale, const at::Tensor& wpk,
                             const at::Tensor& wsc, const at::Tensor& nl,
                             const at::Tensor& sorted_ids, const at::Tensor& expert_ids,
                             const at::Tensor& ntp, const at::Tensor& topk_weights,
                             at::Tensor& out_scatter, int64_t top_k, int64_t block_m,
                             int64_t num_valid_tokens) {
    launch_moe_common(q, a_scale, wpk, wsc, nl, sorted_ids, expert_ids, ntp, nullptr,
                      out_scatter.data_ptr<float>(), topk_weights.data_ptr<float>(),
                      wpk.size(1), wpk.size(2) * 2, (int)top_k, (int)block_m,
                      (int)num_valid_tokens, true);
}

void launch_moe_gather_reduce(const at::Tensor& out2, const at::Tensor& sorted_ids,
                              const at::Tensor& topk_weights, const at::Tensor& ntp,
                              at::Tensor& out, int64_t top_k, int64_t num_valid_tokens) {
    const int P = out2.size(0), N = out2.size(1);
    constexpr int TPB = 256;
    dim3 grid((N + TPB - 1) / TPB, P);
    auto stream = at::cuda::getCurrentCUDAStream();
    moe_gather_reduce_kernel<<<grid, TPB, 0, stream>>>(
        reinterpret_cast<const bf16*>(out2.data_ptr()), sorted_ids.data_ptr<int>(),
        topk_weights.data_ptr<float>(), ntp.data_ptr<int>(), out.data_ptr<float>(),
        P, N, (int)top_k, (int)num_valid_tokens);
}
