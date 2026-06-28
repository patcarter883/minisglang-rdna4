// Grouped (MoE) W8A8-FP8 GEMM for AMD RDNA4 (gfx1201).
//
// The MoE analogue of w4a8_fp8_wmma_kernel.hip, simplified to W8A8 = fp8 (e4m3)
// weights with a PER-OUTPUT-CHANNEL fp32 scale + fp8 activations. Instead of one
// (M,K)x(K,N) GEMM, it computes a *grouped* GEMM over routed tokens, matching
// vLLM's fused_moe contract (moe_align_block_size + a stacked per-expert weight
// tensor). W8A8 is a strict simplification of W4A8: identical WMMA core, identical
// activation quant, identical scatter/gather/reduce plumbing. The only deltas are
// (1) the weight HBM operand is a contiguous e4m3 byte tensor (no int4 unpack /
// zeros / group scale), and (2) the weight scale is per-output-channel, applied
// once in the epilogue rather than folded per K-group in the K-loop.
//
// Contract (mirrors vllm/model_executor/layers/fused_moe/fused_moe.py):
//   x          (T, K)            fp16   token activations (T = num tokens)
//   w_fp8      (E, N, K)         uint8  stacked per-expert e4m3 weights (row-major)
//   scales     (E, N)            fp32   per-expert per-output-channel weight scales
//   sorted_token_ids (P,)        int32  padded, expert-sorted token slots
//   expert_ids (P/block_m,)      int32  expert for each block of block_m rows
//   num_tokens_post_padded (1,)  int32  valid padded length
//   top_k                        int    activation row = offs_token / top_k
//   block_m                      int    moe_align block size (mult. of 16)
//   out        (P, N)            fp16   one row per (token, expert-slot)
//
// Each block of `block_m` sorted rows belongs to ONE expert (expert_ids[block]),
// so a WMMA tile of block_m rows reads a single expert's weight slab — that's why
// the tile's M dimension is block_m (16/32/64), not the dense kernel's BM=256.
// Padding rows (offs_token >= num_valid) are guarded: their A is staged as fp8
// zero and their output is not written.
//
// FUSED SCATTER (template bool SCATTER): the gemm2 epilogue can write its result
// directly to the final (M,N) fp32 `output` via an indirect atomicAdd instead of
// a contiguous (P,N) buffer + a torch scatter-reduce. In scatter mode the
// activation gather is IDENTITY (src = padded row), and the epilogue does
//   atomicAdd(&output[(sorted_ids[r]/top_k)*N + n], topk_weights[sorted_ids[r]]
//             * result * act_scale[r])
// A token's top_k expert partials live in top_k DIFFERENT blocks (workgroups),
// so the cross-block accumulation needs atomics; within a block all rows are
// distinct tokens (a token is never routed to one expert twice) so there is no
// intra-block collision. Contention is top_k-wide -> atomics are not the
// bottleneck. fp32 accumulation (matches Marlin use_fp32_reduce). This removes
// the torch epilogue (where/arange/index_add) and makes the op HIP-graph-safe.

#undef __HIP_NO_HALF_CONVERSIONS__
#undef __HIP_NO_HALF_OPERATORS__

#include <cstdlib>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "moe_gemm_tiled.h"   // consolidated TileConfig WMMA grouped GEMM (the live default)
#include "kernel_names.h"     // w4a8::MoeKernel (descriptive ids; opaque int over the ABI)

namespace w4a8_fp8_moe {

constexpr float E4M3_MAX = 448.0f;
constexpr int WMMA_DIM = 16;
constexpr int MAX_GROUP_SIZE = 128;

__device__ __forceinline__ unsigned char f32_to_e4m3(float v) {
    int packed = __builtin_amdgcn_cvt_pk_fp8_f32(v, v, 0, false);
    return static_cast<unsigned char>(packed & 0xFF);
}
__device__ __forceinline__ float e4m3_to_f32(unsigned char b) {
    return __builtin_amdgcn_cvt_f32_fp8(static_cast<int>(b), 0);
}

// ---- per-token activation fp8 quant (one block per token row, T rows) ----
constexpr int ACTQUANT_BLOCK = 256;
__global__ void moe_compute_act_fp8_kernel(
    const __half* __restrict__ x, unsigned char* __restrict__ x_fp8,
    float* __restrict__ act_scales, int T, int K) {
    const int row = blockIdx.x;
    if (row >= T) return;
    const int tid = threadIdx.x;
    const __half* xr = x + row * K;
    unsigned char* outr = x_fp8 + row * K;
    float local_max = 0.0f;
    for (int k = tid; k < K; k += ACTQUANT_BLOCK) {
        float v = fabsf(__half2float(xr[k]));
        if (v > local_max) local_max = v;
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        float other = __shfl_xor(local_max, o);
        if (other > local_max) local_max = other;
    }
    __shared__ float wm[ACTQUANT_BLOCK / 32];
    const int lane = tid & 31, wid = tid >> 5;
    if (lane == 0) wm[wid] = local_max;
    __syncthreads();
    if (wid == 0) {
        float v = (lane < ACTQUANT_BLOCK / 32) ? wm[lane] : 0.0f;
        #pragma unroll
        for (int o = 4; o > 0; o >>= 1) {
            float other = __shfl_xor(v, o);
            if (other > v) v = other;
        }
        if (lane == 0) {
            float s = v / E4M3_MAX;
            if (s < 1e-8f) s = 1e-8f;
            wm[0] = s;
            act_scales[row] = s;
        }
    }
    __syncthreads();
    const float inv = 1.0f / wm[0];
    for (int k = tid; k < K; k += ACTQUANT_BLOCK)
        outr[k] = f32_to_e4m3(__half2float(xr[k]) * inv);
}

// ============================================================
// grouped v0: scalar golden. One thread per (padded_row, col).
// ============================================================
constexpr int MV0_BM = 16, MV0_BN = 16;
template<bool SCATTER>
__global__ void moe_gemm_v0_kernel(
    const unsigned char* __restrict__ x_fp8,    // (T, K)
    const float*         __restrict__ act_scales,// (T,)
    const unsigned char* __restrict__ w_fp8,    // (E, N, K) e4m3
    const float*         __restrict__ w_scales, // (E, N) per-output-channel
    const int*           __restrict__ sorted_token_ids, // (P,)
    const int*           __restrict__ expert_ids,       // (P/block_m,)
    const int*           __restrict__ num_tokens_post_padded,
    __half*              __restrict__ out,       // (P, N) non-scatter
    int T, int N, int K, int group_size, int top_k, int block_m,
    int num_valid_tokens,
    float*               __restrict__ output_scatter,  // (M, N) fp32 scatter
    const float*         __restrict__ topk_weights,    // (M*top_k,) scatter
    int out_top_k) {

    const int row_pad = blockIdx.y * MV0_BM + threadIdx.y;
    const int col = blockIdx.x * MV0_BN + threadIdx.x;
    if (col >= N) return;
    const int P_valid = num_tokens_post_padded[0];
    if (row_pad >= P_valid) return;

    const int offs_token = sorted_token_ids[row_pad];
    if (offs_token >= num_valid_tokens) return;  // padding slot
    const int src = SCATTER ? row_pad : (offs_token / top_k);  // activation row
    const int e = expert_ids[row_pad / block_m];

    const unsigned char* wq_e = w_fp8 + (long)e * N * K;     // e4m3 weights for this expert
    const float* ws_e = w_scales + (long)e * N;             // per-output-channel scale
    const float a_scale = act_scales[src];

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        const float wf = e4m3_to_f32(wq_e[(long)col * K + k]);
        const float af = e4m3_to_f32(x_fp8[(long)src * K + k]);
        acc += wf * af;
    }
    acc *= ws_e[col];
    if constexpr (SCATTER) {
        const int token = offs_token / out_top_k;
        const float w = topk_weights[offs_token];
        atomicAdd(&output_scatter[(long)token * N + col], w * acc * a_scale);
    } else {
        out[(long)row_pad * N + col] = __float2half(acc * a_scale);
    }
}

// ============================================================
// Shared tile constants + vector aliases for the grouped MoE kernels. The served
// WMMA path is moe_gemm_tiled.h (the TileConfig v5/v6 twins); the standalone
// moe_gemm_v5/v6_kernel that used to live here were W4A8-only research kernels and
// have been dropped for W8A8 (the tiled path is the only WMMA path). The aliases /
// MV5_* constants below are still used by the fused-silu and GEMV kernels and the
// launchers, so they remain.
// ============================================================
using v2i_t = int __attribute__((ext_vector_type(2)));
using v8f_t = float __attribute__((ext_vector_type(8)));
using v2f_t = float __attribute__((ext_vector_type(2)));
using v4i_t = int __attribute__((ext_vector_type(4)));   // 16 e4m3 / lane (GEMV b128 read)
constexpr int MV5_BN = 64;
constexpr int MV5_NFRAG = MV5_BN / 16;     // N-fragments per warp
constexpr int MV5_MAX_WARPS = 8;           // block_m up to 128 (8 warps)
constexpr int MV5_MAX_BM = 128;
constexpr int LDS_PAD = 8;                 // bank-conflict pad on the K stride

#if 0
// (retired W4A8-only standalone WMMA kernels — superseded by moe_gemm_tiled.h)
template<bool SCATTER, int BN>
__global__ void __launch_bounds__(MV5_MAX_WARPS * 32) moe_gemm_v5_kernel(
    const unsigned char* __restrict__ x_fp8,
    const float*         __restrict__ act_scales,
    const int*           __restrict__ w_packed,
    const __half*        __restrict__ w_scales,
    const int*           __restrict__ w_zeros,
    const int*           __restrict__ sorted_token_ids,
    const int*           __restrict__ expert_ids,
    const int*           __restrict__ num_tokens_post_padded,
    __half*              __restrict__ out,       // (P, N) non-scatter
    int T, int N, int K, int group_size, int top_k, int block_m,
    int num_valid_tokens,
    float*               __restrict__ output_scatter,  // (M, N) fp32 scatter
    const float*         __restrict__ topk_weights,    // (M*top_k,) scatter
    int out_top_k) {

    constexpr int NFRAG = BN / WMMA_DIM;       // independent accumulators per warp
    const int block_idx = blockIdx.y;          // one moe_align block (one expert)
    const int row0 = block_idx * block_m;      // first padded row
    const int P_valid = num_tokens_post_padded[0];
    if (row0 >= P_valid) return;

    const int block_n = blockIdx.x * BN;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;              // which 16-row M-subtile
    const int lane = tid & 31;
    const int lrow = lane & 15;
    const int khalf = (lane >> 4) * 8;
    const int frag_col = lane & 15;
    const int frag_row0 = (lane >> 4) * 8;
    const int n_warps = block_m / WMMA_DIM;    // active warps
    if (warp_id >= n_warps) return;

    const int e = expert_ids[block_idx];
    const int num_groups = K / group_size;
    const int ppr = K / PACK_FACTOR;
    const int* wq_e = w_packed + (long)e * N * ppr;
    const __half* ws_e = w_scales + (long)e * N * num_groups;
    const int* wz_e = (w_zeros != nullptr)
                      ? w_zeros + (long)e * (N / 8) * num_groups : nullptr;

    const int BK = group_size;
    const int LDSBK = BK + LDS_PAD;
    // Dynamic LDS sized for the ACTUAL (block_m, group_size) at launch: A_tile
    // (block_m rows) then B_tile (MV5_BN cols). Avoids over-allocating for the
    // MAX_GROUP_SIZE/MAX_BM worst case, which crushes occupancy for g=32 tiles.
    extern __shared__ unsigned char smem[];
    unsigned char* A_tile = smem;
    unsigned char* B_tile = smem + block_m * LDSBK;

    float running[NFRAG][8];
    #pragma unroll
    for (int f = 0; f < NFRAG; ++f)
        #pragma unroll
        for (int ee = 0; ee < 8; ++ee) running[f][ee] = 0.0f;

    const int total_threads = n_warps * 32;
    for (int g = 0; g < num_groups; ++g) {
        const int k0 = g * BK;

        // ---- stage activations: A_tile[r][k] = x_fp8[src(r)][k0+k], guarded ----
        // uint32 (4 fp8) coalesced loads over block_m rows.
        for (int idx = tid * 4; idx < block_m * BK; idx += total_threads * 4) {
            const int r = idx / BK, k = idx % BK;
            unsigned int v = 0u;
            const int offs_token = sorted_token_ids[row0 + r];
            if (offs_token < num_valid_tokens) {
                const int src = SCATTER ? (row0 + r) : (offs_token / top_k);
                v = *reinterpret_cast<const unsigned int*>(&x_fp8[(long)src * K + k0 + k]);
            }
            *reinterpret_cast<unsigned int*>(&A_tile[r * LDSBK + k]) = v;
        }
        // ---- stage weights: expand int4 -> fp8 into B_tile[n][k] ----
        const int njobs = BN * (BK / PACK_FACTOR);
        for (int j = tid; j < njobs; j += total_threads) {
            const int n = j / (BK / PACK_FACTOR), k8 = j % (BK / PACK_FACTOR);
            const int an = block_n + n;
            int word = 0, zp = 8;
            if (an < N) {
                word = wq_e[an * ppr + (k0 / PACK_FACTOR) + k8];
                if (wz_e != nullptr) {
                    const int pk = wz_e[(an / 8) * num_groups + g];
                    zp = (pk >> ((an % 8) * 4)) & 0xF;
                }
            }
            unsigned char* dst = &B_tile[n * LDSBK + k8 * PACK_FACTOR];
            #pragma unroll
            for (int jj = 0; jj < PACK_FACTOR; ++jj)
                dst[jj] = (an < N)
                    ? int4_signed_to_e4m3(((word >> (jj * 4)) & 0xF) - zp) : 0;
        }
        __syncthreads();

        v8f_t acc[NFRAG];
        #pragma unroll
        for (int f = 0; f < NFRAG; ++f) acc[f] = v8f_t{0,0,0,0,0,0,0,0};

        const int k_sub = BK / WMMA_DIM;
        const int a_base = (warp_id * WMMA_DIM + lrow) * LDSBK + khalf;
        const int b_base = lrow * LDSBK + khalf;
        for (int kt = 0; kt < k_sub; ++kt) {
            const int ko = kt * WMMA_DIM;
            v2i_t a_cur = *reinterpret_cast<const v2i_t*>(&A_tile[a_base + ko]);
            #pragma unroll
            for (int f = 0; f < NFRAG; ++f) {
                v2i_t b_cur = *reinterpret_cast<const v2i_t*>(
                    &B_tile[b_base + f * WMMA_DIM * LDSBK + ko]);
                acc[f] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(
                    a_cur, b_cur, acc[f]);
            }
        }
        #pragma unroll
        for (int f = 0; f < NFRAG; ++f) {
            const int abs_n = block_n + f * WMMA_DIM + frag_col;
            const float wsc = (abs_n < N)
                ? __half2float(ws_e[abs_n * num_groups + g]) : 0.0f;
            #pragma unroll
            for (int ee = 0; ee < 8; ++ee) running[f][ee] += acc[f][ee] * wsc;
        }
        __syncthreads();
    }

    // ---- epilogue: per-row act scale, then write (non-scatter) or indirect
    //      atomicAdd scatter to the final (M,N) output (guarded at padding). ----
    #pragma unroll
    for (int ee = 0; ee < 8; ++ee) {
        const int r = warp_id * WMMA_DIM + frag_row0 + ee;
        if (r >= block_m) continue;
        const int row_pad = row0 + r;
        const int offs_token = sorted_token_ids[row_pad];
        if (offs_token >= num_valid_tokens) continue;
        const int src = SCATTER ? row_pad : (offs_token / top_k);
        const float asc = act_scales[src];
        if constexpr (SCATTER) {
            const int token = offs_token / out_top_k;
            const float w = topk_weights[offs_token];
            #pragma unroll
            for (int f = 0; f < NFRAG; ++f) {
                const int abs_n = block_n + f * WMMA_DIM + frag_col;
                if (abs_n < N)
                    atomicAdd(&output_scatter[(long)token * N + abs_n],
                              w * running[f][ee] * asc);
            }
        } else {
            #pragma unroll
            for (int f = 0; f < NFRAG; ++f) {
                const int abs_n = block_n + f * WMMA_DIM + frag_col;
                if (abs_n < N)
                    out[(long)row_pad * N + abs_n] = __float2half(running[f][ee] * asc);
            }
        }
    }
}

// ============================================================
// grouped v6: the dense-v10 lesson applied to MoE -- get the (warp-private)
// gathered activations OUT of LDS via a warp-shuffle transpose, so LDS holds B
// (weights) only. Each warp owns 16 padded rows; lane L gathers row
// (warp*16 + L>>1) (token = sorted_token_ids/top_k) coalesced-per-row, then
// __shfl transposes to the WMMA layout (src = 2*(lane&15)+(lane>>4)). Removes the
// A staging + A LDS. (MoE tiles are already small, so unlike dense this is not an
// occupancy play -- it just deletes the A round-trip; measure to see if it pays.)
template<bool SCATTER, int BN>
__global__ void __launch_bounds__(MV5_MAX_WARPS * 32) moe_gemm_v6_kernel(
    const unsigned char* __restrict__ x_fp8,
    const float*         __restrict__ act_scales,
    const int*           __restrict__ w_packed,
    const __half*        __restrict__ w_scales,
    const int*           __restrict__ w_zeros,
    const int*           __restrict__ sorted_token_ids,
    const int*           __restrict__ expert_ids,
    const int*           __restrict__ num_tokens_post_padded,
    __half*              __restrict__ out,
    int T, int N, int K, int group_size, int top_k, int block_m,
    int num_valid_tokens,
    float*               __restrict__ output_scatter,
    const float*         __restrict__ topk_weights,
    int out_top_k, int gtile) {

    constexpr int NFRAG = BN / WMMA_DIM;
    const int block_idx = blockIdx.y;
    const int row0 = block_idx * block_m;
    const int P_valid = num_tokens_post_padded[0];
    if (row0 >= P_valid) return;

    const int block_n = blockIdx.x * BN;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int lrow = lane & 15;
    const int khalf = (lane >> 4) * 8;
    const int frag_col = lane & 15;
    const int frag_row0 = (lane >> 4) * 8;
    const int n_warps = block_m / WMMA_DIM;
    if (warp_id >= n_warps) return;

    const int e = expert_ids[block_idx];
    const int num_groups = K / group_size;
    const int ppr = K / PACK_FACTOR;
    const int* wq_e = w_packed + (long)e * N * ppr;
    const __half* ws_e = w_scales + (long)e * N * num_groups;
    const int* wz_e = (w_zeros != nullptr)
                      ? w_zeros + (long)e * (N / 8) * num_groups : nullptr;

    const int BK = group_size;
    const int LDSBK = BK + LDS_PAD;
    extern __shared__ unsigned char smem[];
    unsigned char* B_tile = smem;                 // B only -- no A in LDS

    // hoisted per-lane activation gather (constant across the K loop).
    const int a_my_row = row0 + warp_id * WMMA_DIM + (lane >> 1);
    const int a_offs = sorted_token_ids[a_my_row];
    const bool a_valid = a_offs < num_valid_tokens;
    const int a_src = SCATTER ? a_my_row : (a_offs / top_k);
    const int a_kh = (lane & 1) * 8;
    const int a_src_lane = 2 * (lane & 15) + (lane >> 4);

    float running[NFRAG][8];
    #pragma unroll
    for (int f = 0; f < NFRAG; ++f)
        #pragma unroll
        for (int ee = 0; ee < 8; ++ee) running[f][ee] = 0.0f;

    const int total_threads = n_warps * 32;
    const int k_sub = BK / WMMA_DIM;
    const int b_base = lrow * LDSBK + khalf;

    // gtile groups staged per __syncthreads pair (gtile=1 == original v6): cuts the
    // per-group sync count num_groups -> num_groups/gtile, the WMMA-path analogue of
    // the GEMV's BK=K win (g=32, K=2304 -> 72 groups -> 72 syncs is the M=128 tax).
    const int njobs = BN * (BK / PACK_FACTOR);
    for (int gt = 0; gt < num_groups; gt += gtile) {
        const int gend = (gt + gtile < num_groups) ? (gt + gtile) : num_groups;
        for (int gi = gt; gi < gend; ++gi) {
            const int k0 = gi * BK;
            unsigned char* Bg = B_tile + (long)(gi - gt) * BN * LDSBK;
            for (int j = tid; j < njobs; j += total_threads) {
                const int n = j / (BK / PACK_FACTOR), k8 = j % (BK / PACK_FACTOR);
                const int an = block_n + n;
                int word = 0, zp = 8;
                if (an < N) {
                    word = wq_e[an * ppr + (k0 / PACK_FACTOR) + k8];
                    if (wz_e != nullptr) {
                        const int pk = wz_e[(an / 8) * num_groups + gi];
                        zp = (pk >> ((an % 8) * 4)) & 0xF;
                    }
                }
                unsigned char* dst = &Bg[n * LDSBK + k8 * PACK_FACTOR];
                #pragma unroll
                for (int jj = 0; jj < PACK_FACTOR; ++jj)
                    dst[jj] = (an < N)
                        ? int4_signed_to_e4m3(((word >> (jj * 4)) & 0xF) - zp) : 0;
            }
        }
        __syncthreads();

        for (int gi = gt; gi < gend; ++gi) {
            const int k0 = gi * BK;
            unsigned char* Bg = B_tile + (long)(gi - gt) * BN * LDSBK;
            v8f_t acc[NFRAG];
            #pragma unroll
            for (int f = 0; f < NFRAG; ++f) acc[f] = v8f_t{0,0,0,0,0,0,0,0};

            for (int kt = 0; kt < k_sub; ++kt) {
                const int gk = k0 + kt * WMMA_DIM + a_kh;
                v2i_t av = v2i_t{0, 0};
                if (a_valid)
                    av = *reinterpret_cast<const v2i_t*>(&x_fp8[(long)a_src * K + gk]);
                v2i_t a_cur;
                a_cur[0] = __shfl(av[0], a_src_lane);
                a_cur[1] = __shfl(av[1], a_src_lane);
                #pragma unroll
                for (int f = 0; f < NFRAG; ++f) {
                    v2i_t b_cur = *reinterpret_cast<const v2i_t*>(
                        &Bg[b_base + f * WMMA_DIM * LDSBK + kt * WMMA_DIM]);
                    acc[f] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(
                        a_cur, b_cur, acc[f]);
                }
            }
            #pragma unroll
            for (int f = 0; f < NFRAG; ++f) {
                const int abs_n = block_n + f * WMMA_DIM + frag_col;
                const float wsc = (abs_n < N)
                    ? __half2float(ws_e[abs_n * num_groups + gi]) : 0.0f;
                #pragma unroll
                for (int ee = 0; ee < 8; ++ee) running[f][ee] += acc[f][ee] * wsc;
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int ee = 0; ee < 8; ++ee) {
        const int r = warp_id * WMMA_DIM + frag_row0 + ee;
        if (r >= block_m) continue;
        const int row_pad = row0 + r;
        const int offs_token = sorted_token_ids[row_pad];
        if (offs_token >= num_valid_tokens) continue;
        const int src = SCATTER ? row_pad : (offs_token / top_k);
        const float asc = act_scales[src];
        if constexpr (SCATTER) {
            const int token = offs_token / out_top_k;
            const float w = topk_weights[offs_token];
            #pragma unroll
            for (int f = 0; f < NFRAG; ++f) {
                const int abs_n = block_n + f * WMMA_DIM + frag_col;
                if (abs_n < N)
                    atomicAdd(&output_scatter[(long)token * N + abs_n],
                              w * running[f][ee] * asc);
            }
        } else {
            #pragma unroll
            for (int f = 0; f < NFRAG; ++f) {
                const int abs_n = block_n + f * WMMA_DIM + frag_col;
                if (abs_n < N)
                    out[(long)row_pad * N + abs_n] = __float2half(running[f][ee] * asc);
            }
        }
    }
}
#endif  // retired W4A8-only standalone WMMA kernels

// ============================================================
// gemm1 + silu_and_mul FUSED epilogue (gemm1-only, non-scatter). The gated MoE's
// gemm1 produces (P, 2*inter) [gate | up], which the stock apply then runs through
// a SEPARATE silu_and_mul launch -> (P, inter) buf2 -> gemm2. That stages out1
// (P,2*inter) AND buf2 (P,inter) through HBM. These kernels fuse the activation
// into gemm1's epilogue: ONE block computes BOTH the gate N-tile [block_n,
// block_n+BN) and the matching up N-tile [block_n+inter, ...) for the same expert,
// then writes silu(gate)*up directly to (P, inter) -- dropping buf2 and the silu
// launch, and halving gemm1's output HBM write.
//
// BIT-EXACT to the unfused gemm1 + torch.ops._C.silu_and_mul path: each half-tile
// accumulator is rounded to fp16 EXACTLY as the unfused gemm1 store would be
// (__float2half(acc*asc)), then silu is computed in fp32 and cast to fp16
// (vllm::silu_kernel<half>: (half)((float)x/(1+expf(-x)))), and the final product
// is a HALF*HALF multiply (vllm act_first compute: ACT_FN(x)*y) -- matching
// activation_kernels.cu silu_and_mul and test_moe_experts.fp8_ref_moe.
__device__ __forceinline__ __half moe_silu_and_mul_h(float gate_f, float up_f) {
    const __half gate_h = __float2half(gate_f);  // == unfused gemm1's gate store
    const __half up_h = __float2half(up_f);      // == unfused gemm1's up store
    const float g = __half2float(gate_h);
    const __half silu_h = __float2half(g / (1.0f + expf(-g)));  // silu_kernel<half>
    return __hmul(silu_h, up_h);                 // half*half (act_first compute)
}

// v5-based fused gemm1+silu (A+B in LDS). block_n spans the GATE columns; the
// expert's up weights live at row (inter + abs_n), so we stage TWO B-slabs/group
// and keep TWO accumulator sets. group_size BK; A_tile then gate B_tile then up
// B_tile in dynamic LDS.
template<int BN>
__global__ void __launch_bounds__(MV5_MAX_WARPS * 32) moe_gemm1_silu_v5_kernel(
    const unsigned char* __restrict__ x_fp8,
    const float*         __restrict__ act_scales,
    const unsigned char* __restrict__ w_fp8,     // (E, 2*inter, K) e4m3
    const float*         __restrict__ w_scales,  // (E, 2*inter) per-output-channel
    const int*           __restrict__ sorted_token_ids,
    const int*           __restrict__ expert_ids,
    const int*           __restrict__ num_tokens_post_padded,
    __half*              __restrict__ out,       // (P, inter)
    int T, int N, int K, int group_size, int top_k, int block_m,
    int num_valid_tokens, int inter) {

    constexpr int NFRAG = BN / WMMA_DIM;
    const int block_idx = blockIdx.y;
    const int row0 = block_idx * block_m;
    const int P_valid = num_tokens_post_padded[0];
    if (row0 >= P_valid) return;

    const int block_n = blockIdx.x * BN;         // gate column base in [0, inter)
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int lrow = lane & 15;
    const int khalf = (lane >> 4) * 8;
    const int frag_col = lane & 15;
    const int frag_row0 = (lane >> 4) * 8;
    const int n_warps = block_m / WMMA_DIM;
    if (warp_id >= n_warps) return;

    const int e = expert_ids[block_idx];
    const int num_groups = K / group_size;
    const unsigned char* wq_e = w_fp8 + (long)e * N * K;     // e4m3 weights for this expert
    const float* ws_e = w_scales + (long)e * N;             // per-output-channel scale

    const int BK = group_size;
    const int LDSBK = BK + LDS_PAD;
    extern __shared__ unsigned char smem[];
    unsigned char* A_tile = smem;                     // block_m rows
    unsigned char* Bg_tile = A_tile + block_m * LDSBK;     // gate B
    unsigned char* Bu_tile = Bg_tile + BN * LDSBK;         // up B

    float run_g[NFRAG][8], run_u[NFRAG][8];
    #pragma unroll
    for (int f = 0; f < NFRAG; ++f)
        #pragma unroll
        for (int ee = 0; ee < 8; ++ee) { run_g[f][ee] = 0.0f; run_u[f][ee] = 0.0f; }

    const int total_threads = n_warps * 32;
    const int njobs = BN * (BK / 4);
    for (int g = 0; g < num_groups; ++g) {
        const int k0 = g * BK;
        // stage activations (shared by both halves).
        for (int idx = tid * 4; idx < block_m * BK; idx += total_threads * 4) {
            const int r = idx / BK, k = idx % BK;
            unsigned int v = 0u;
            const int offs_token = sorted_token_ids[row0 + r];
            if (offs_token < num_valid_tokens) {
                const int src = offs_token / top_k;
                v = *reinterpret_cast<const unsigned int*>(&x_fp8[(long)src * K + k0 + k]);
            }
            *reinterpret_cast<unsigned int*>(&A_tile[r * LDSBK + k]) = v;
        }
        // stage BOTH weight slabs: gate row (block_n+n), up row (inter+block_n+n);
        // contiguous fp8 byte copy (uint = 4 e4m3), no nibble-unpack / zeros.
        for (int j = tid; j < njobs; j += total_threads) {
            const int n = j / (BK / 4), k4 = j % (BK / 4);
            const int ang = block_n + n;            // gate output row
            const int anu = inter + block_n + n;    // up output row
            unsigned int vg = 0u, vu = 0u;
            if (ang < inter) {
                vg = *reinterpret_cast<const unsigned int*>(&wq_e[(long)ang * K + k0 + k4 * 4]);
                vu = *reinterpret_cast<const unsigned int*>(&wq_e[(long)anu * K + k0 + k4 * 4]);
            }
            *reinterpret_cast<unsigned int*>(&Bg_tile[n * LDSBK + k4 * 4]) = vg;
            *reinterpret_cast<unsigned int*>(&Bu_tile[n * LDSBK + k4 * 4]) = vu;
        }
        __syncthreads();

        v8f_t accg[NFRAG], accu[NFRAG];
        #pragma unroll
        for (int f = 0; f < NFRAG; ++f) {
            accg[f] = v8f_t{0,0,0,0,0,0,0,0}; accu[f] = v8f_t{0,0,0,0,0,0,0,0};
        }
        const int k_sub = BK / WMMA_DIM;
        const int a_base = (warp_id * WMMA_DIM + lrow) * LDSBK + khalf;
        const int b_base = lrow * LDSBK + khalf;
        for (int kt = 0; kt < k_sub; ++kt) {
            const int ko = kt * WMMA_DIM;
            v2i_t a_cur = *reinterpret_cast<const v2i_t*>(&A_tile[a_base + ko]);
            #pragma unroll
            for (int f = 0; f < NFRAG; ++f) {
                v2i_t bg = *reinterpret_cast<const v2i_t*>(
                    &Bg_tile[b_base + f * WMMA_DIM * LDSBK + ko]);
                v2i_t bu = *reinterpret_cast<const v2i_t*>(
                    &Bu_tile[b_base + f * WMMA_DIM * LDSBK + ko]);
                accg[f] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a_cur, bg, accg[f]);
                accu[f] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a_cur, bu, accu[f]);
            }
        }
        // W8A8: accumulate raw acc; per-N gate/up weight scales applied in the epilogue.
        #pragma unroll
        for (int f = 0; f < NFRAG; ++f)
            #pragma unroll
            for (int ee = 0; ee < 8; ++ee) {
                run_g[f][ee] += accg[f][ee];
                run_u[f][ee] += accu[f][ee];
            }
        __syncthreads();
    }

    #pragma unroll
    for (int ee = 0; ee < 8; ++ee) {
        const int r = warp_id * WMMA_DIM + frag_row0 + ee;
        if (r >= block_m) continue;
        const int row_pad = row0 + r;
        const int offs_token = sorted_token_ids[row_pad];
        if (offs_token >= num_valid_tokens) continue;
        const float asc = act_scales[offs_token / top_k];
        #pragma unroll
        for (int f = 0; f < NFRAG; ++f) {
            const int abs_n = block_n + f * WMMA_DIM + frag_col;
            if (abs_n < inter) {
                const float wsc_g_n = ws_e[abs_n];
                const float wsc_u_n = ws_e[inter + abs_n];
                out[(long)row_pad * inter + abs_n] =
                    moe_silu_and_mul_h(run_g[f][ee] * asc * wsc_g_n,
                                       run_u[f][ee] * asc * wsc_u_n);
            }
        }
    }
}

// v6-based fused gemm1+silu (A out of LDS via warp-shuffle, B-only LDS). Served
// default (version 6). Same two-slab gate/up structure as the v5 fused kernel; B
// LDS holds gtile groups x (gate + up) slabs.
template<int BN>
__global__ void __launch_bounds__(MV5_MAX_WARPS * 32) moe_gemm1_silu_v6_kernel(
    const unsigned char* __restrict__ x_fp8,
    const float*         __restrict__ act_scales,
    const unsigned char* __restrict__ w_fp8,     // (E, 2*inter, K) e4m3
    const float*         __restrict__ w_scales,  // (E, 2*inter) per-output-channel
    const int*           __restrict__ sorted_token_ids,
    const int*           __restrict__ expert_ids,
    const int*           __restrict__ num_tokens_post_padded,
    __half*              __restrict__ out,       // (P, inter)
    int T, int N, int K, int group_size, int top_k, int block_m,
    int num_valid_tokens, int gtile, int inter) {

    constexpr int NFRAG = BN / WMMA_DIM;
    const int block_idx = blockIdx.y;
    const int row0 = block_idx * block_m;
    const int P_valid = num_tokens_post_padded[0];
    if (row0 >= P_valid) return;

    const int block_n = blockIdx.x * BN;         // gate column base in [0, inter)
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int lrow = lane & 15;
    const int khalf = (lane >> 4) * 8;
    const int frag_col = lane & 15;
    const int frag_row0 = (lane >> 4) * 8;
    const int n_warps = block_m / WMMA_DIM;
    if (warp_id >= n_warps) return;

    const int e = expert_ids[block_idx];
    const int num_groups = K / group_size;
    const unsigned char* wq_e = w_fp8 + (long)e * N * K;     // e4m3 weights for this expert
    const float* ws_e = w_scales + (long)e * N;             // per-output-channel scale

    const int BK = group_size;
    const int LDSBK = BK + LDS_PAD;
    extern __shared__ unsigned char smem[];
    unsigned char* B_tile = smem;                 // gtile x (gate|up) slabs, B only

    // hoisted per-lane activation gather (constant across the K loop), as in v6.
    const int a_my_row = row0 + warp_id * WMMA_DIM + (lane >> 1);
    const int a_offs = sorted_token_ids[a_my_row];
    const bool a_valid = a_offs < num_valid_tokens;
    const int a_src = a_offs / top_k;
    const int a_kh = (lane & 1) * 8;
    const int a_src_lane = 2 * (lane & 15) + (lane >> 4);

    float run_g[NFRAG][8], run_u[NFRAG][8];
    #pragma unroll
    for (int f = 0; f < NFRAG; ++f)
        #pragma unroll
        for (int ee = 0; ee < 8; ++ee) { run_g[f][ee] = 0.0f; run_u[f][ee] = 0.0f; }

    const int total_threads = n_warps * 32;
    const int k_sub = BK / WMMA_DIM;
    const int b_base = lrow * LDSBK + khalf;
    const int slab = BN * LDSBK;                   // bytes for one (BN, BK) slab
    const int njobs = BN * (BK / 4);
    for (int gt = 0; gt < num_groups; gt += gtile) {
        const int gend = (gt + gtile < num_groups) ? (gt + gtile) : num_groups;
        for (int gi = gt; gi < gend; ++gi) {
            const int k0 = gi * BK;
            // two slabs per group: gate then up. contiguous fp8 byte copy (uint = 4 e4m3).
            unsigned char* Bg = B_tile + (long)(gi - gt) * 2 * slab;
            unsigned char* Bu = Bg + slab;
            for (int j = tid; j < njobs; j += total_threads) {
                const int n = j / (BK / 4), k4 = j % (BK / 4);
                const int ang = block_n + n;
                const int anu = inter + block_n + n;
                unsigned int vg = 0u, vu = 0u;
                if (ang < inter) {
                    vg = *reinterpret_cast<const unsigned int*>(&wq_e[(long)ang * K + k0 + k4 * 4]);
                    vu = *reinterpret_cast<const unsigned int*>(&wq_e[(long)anu * K + k0 + k4 * 4]);
                }
                *reinterpret_cast<unsigned int*>(&Bg[n * LDSBK + k4 * 4]) = vg;
                *reinterpret_cast<unsigned int*>(&Bu[n * LDSBK + k4 * 4]) = vu;
            }
        }
        __syncthreads();

        for (int gi = gt; gi < gend; ++gi) {
            const int k0 = gi * BK;
            unsigned char* Bg = B_tile + (long)(gi - gt) * 2 * slab;
            unsigned char* Bu = Bg + slab;
            v8f_t accg[NFRAG], accu[NFRAG];
            #pragma unroll
            for (int f = 0; f < NFRAG; ++f) {
                accg[f] = v8f_t{0,0,0,0,0,0,0,0}; accu[f] = v8f_t{0,0,0,0,0,0,0,0};
            }
            for (int kt = 0; kt < k_sub; ++kt) {
                const int gk = k0 + kt * WMMA_DIM + a_kh;
                v2i_t av = v2i_t{0, 0};
                if (a_valid)
                    av = *reinterpret_cast<const v2i_t*>(&x_fp8[(long)a_src * K + gk]);
                v2i_t a_cur;
                a_cur[0] = __shfl(av[0], a_src_lane);
                a_cur[1] = __shfl(av[1], a_src_lane);
                #pragma unroll
                for (int f = 0; f < NFRAG; ++f) {
                    v2i_t bg = *reinterpret_cast<const v2i_t*>(
                        &Bg[b_base + f * WMMA_DIM * LDSBK + kt * WMMA_DIM]);
                    v2i_t bu = *reinterpret_cast<const v2i_t*>(
                        &Bu[b_base + f * WMMA_DIM * LDSBK + kt * WMMA_DIM]);
                    accg[f] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a_cur, bg, accg[f]);
                    accu[f] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a_cur, bu, accu[f]);
                }
            }
            // W8A8: accumulate raw acc; per-N gate/up weight scales applied in the epilogue.
            #pragma unroll
            for (int f = 0; f < NFRAG; ++f)
                #pragma unroll
                for (int ee = 0; ee < 8; ++ee) {
                    run_g[f][ee] += accg[f][ee];
                    run_u[f][ee] += accu[f][ee];
                }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int ee = 0; ee < 8; ++ee) {
        const int r = warp_id * WMMA_DIM + frag_row0 + ee;
        if (r >= block_m) continue;
        const int row_pad = row0 + r;
        const int offs_token = sorted_token_ids[row_pad];
        if (offs_token >= num_valid_tokens) continue;
        const float asc = act_scales[offs_token / top_k];
        #pragma unroll
        for (int f = 0; f < NFRAG; ++f) {
            const int abs_n = block_n + f * WMMA_DIM + frag_col;
            if (abs_n < inter) {
                const float wsc_g_n = ws_e[abs_n];
                const float wsc_u_n = ws_e[inter + abs_n];
                out[(long)row_pad * inter + abs_n] =
                    moe_silu_and_mul_h(run_g[f][ee] * asc * wsc_g_n,
                                       run_u[f][ee] * asc * wsc_u_n);
            }
        }
    }
}

// ============================================================
// grouped v7: DECODE-specialised grouped GEMV -- the MoE analogue of dense v11.
// One warp per output column n of ONE expert; the 32 lanes stream that column's
// K/8 int32 weight row coalesced (b128 = v4i = 32 fp8/lane), expand int4->fp8
// in-register, and dot against the block's GATHERED activation rows (staged once
// into LDS, K-tiled). The per-group weight scale is folded in inline as K
// streams, so there are NO per-group barriers and NO B-LDS -- it just hits weight
// bandwidth, which is the decode bottleneck (g=32 => 72 K-groups, each a
// __syncthreads in the v5/v6 WMMA path). One weight read is reused across all
// block_m rows. Requires K % 32 == 0, group_size % 32 == 0, BK % group_size == 0.
// MMAX = compile-time block_m cap (rows/block); NWARPS = output cols/block.
// ============================================================
template<bool SCATTER, int NWARPS, int MMAX, int COLS>
__global__ void __launch_bounds__(NWARPS * 32) moe_gemv_v7_kernel(
    const unsigned char* __restrict__ x_fp8,
    const float*         __restrict__ act_scales,
    const unsigned char* __restrict__ w_fp8,     // (E, N, K) e4m3
    const float*         __restrict__ w_scales,  // (E, N) per-output-channel
    const int*           __restrict__ sorted_token_ids,
    const int*           __restrict__ expert_ids,
    const int*           __restrict__ num_tokens_post_padded,
    __half*              __restrict__ out,
    int T, int N, int K, int group_size, int top_k, int block_m,
    int num_valid_tokens,
    float*               __restrict__ output_scatter,
    const float*         __restrict__ topk_weights,
    int out_top_k, int BK) {

    const int block_idx = blockIdx.y;
    const int row0 = block_idx * block_m;
    const int P_valid = num_tokens_post_padded[0];
    if (row0 >= P_valid) return;

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    // Each warp owns COLS consecutive output columns. With short K (gemm2 K=896 =>
    // ~1 b128 read/column) a 1-column warp can't hide the weight-read latency;
    // COLS>1 issues COLS reads back-to-back so the scheduler overlaps them -- the
    // gemm2 fix (gemm1's K=2304 already pipelines, so it runs COLS=1).
    const int col_base = (blockIdx.x * NWARPS + warp_id) * COLS;

    const int e = expert_ids[block_idx];
    const unsigned char* wq_e = w_fp8 + (long)e * N * K;     // e4m3 weights for this expert
    const float* ws_e = w_scales + (long)e * N;             // per-output-channel scale

    // Compact the block's REAL rows once (decode: usually 1-2 of block_m).
    __shared__ int s_src[MMAX], s_pad[MMAX], s_offs[MMAX], s_mreal;
    if (tid == 0) {
        int c = 0;
        for (int r = 0; r < block_m; ++r) {
            const int offs = sorted_token_ids[row0 + r];
            if (offs < num_valid_tokens) {
                s_offs[c] = offs;
                s_pad[c] = row0 + r;
                s_src[c] = SCATTER ? (row0 + r) : (offs / top_k);
                ++c;
            }
        }
        s_mreal = c;
    }
    __syncthreads();
    const int M_real = s_mreal;

    // gathered activation slice (M_real, BK) fp8, K-tiled into LDS.
    extern __shared__ __attribute__((aligned(16))) unsigned char v7_a[];

    float acc[COLS][MMAX];
    #pragma unroll
    for (int c = 0; c < COLS; ++c)
        #pragma unroll
        for (int m = 0; m < MMAX; ++m) acc[c][m] = 0.0f;

    for (int kc = 0; kc < K; kc += BK) {
        const int ck = (K - kc < BK) ? (K - kc) : BK;
        // gather-stage only the real rows (shared across all COLS columns).
        const int tile = M_real * ck;
        for (int i = tid; i < tile; i += NWARPS * 32) {
            const int r = i / ck, kk = i - r * ck;
            v7_a[i] = x_fp8[(long)s_src[r] * K + kc + kk];
        }
        __syncthreads();
        // W8A8: weights are e4m3 bytes (E,N,K). Each lane streams 32 weight bytes
        // (8 int32 = 2x v4i_t) per step, matching the 32-element activation slice the
        // dot loops below assume. No nibble-unpack, no zeros, no per-group scale fold
        // — the per-output-channel scale is applied once in the epilogue.
        for (int base = lane * 32; base < ck; base += 32 * 32) {
            const int k0 = base;                    // first k of these 32, within chunk
            // issue all COLS weight reads first (overlap latency).
            v4i_t wv0[COLS], wv1[COLS];
            #pragma unroll
            for (int c = 0; c < COLS; ++c) {
                const int nc = col_base + c;
                if (nc < N) {
                    const unsigned char* wp = &wq_e[(long)nc * K + kc + k0];
                    wv0[c] = *reinterpret_cast<const v4i_t*>(wp);       // weights 0..15
                    wv1[c] = *reinterpret_cast<const v4i_t*>(wp + 16);  // weights 16..31
                } else {
                    wv0[c] = v4i_t{0, 0, 0, 0}; wv1[c] = v4i_t{0, 0, 0, 0};
                }
            }
            // Activations stay converted per (c,r) (no af[] hoist) to keep registers
            // low; weights converted 2-at-a-time via cvt_pk_f32_fp8 like the act side.
            #pragma unroll
            for (int c = 0; c < COLS; ++c) {
                if (col_base + c >= N) continue;
                const unsigned int* wint0 = reinterpret_cast<const unsigned int*>(&wv0[c]);
                const unsigned int* wint1 = reinterpret_cast<const unsigned int*>(&wv1[c]);
                for (int r = 0; r < M_real; ++r) {
                    const unsigned int* ar4 =
                        reinterpret_cast<const unsigned int*>(v7_a + r * ck + k0);
                    float p = 0.0f;
                    #pragma unroll
                    for (int q = 0; q < 8; ++q) {     // 8 act ints = 32 fp8 = 32 weights
                        const unsigned int a4 = ar4[q];
                        const unsigned int w4 = (q < 4) ? wint0[q] : wint1[q - 4];
                        const v2f_t alo = __builtin_amdgcn_cvt_pk_f32_fp8(a4, false);
                        const v2f_t ahi = __builtin_amdgcn_cvt_pk_f32_fp8(a4, true);
                        const v2f_t wlo = __builtin_amdgcn_cvt_pk_f32_fp8(w4, false);
                        const v2f_t whi = __builtin_amdgcn_cvt_pk_f32_fp8(w4, true);
                        p += alo[0] * wlo[0] + alo[1] * wlo[1]
                           + ahi[0] * whi[0] + ahi[1] * whi[1];
                    }
                    acc[c][r] += p;
                }
            }
        }
        __syncthreads();
    }

    // warp-reduce + epilogue, per owned column.
    #pragma unroll
    for (int c = 0; c < COLS; ++c) {
        for (int off = 16; off > 0; off >>= 1)
            for (int r = 0; r < M_real; ++r)
                acc[c][r] += __shfl_xor(acc[c][r], off);
        const int nc = col_base + c;
        if (lane != 0 || nc >= N) continue;
        const float wsc = ws_e[nc];               // per-output-channel weight scale
        for (int r = 0; r < M_real; ++r) {
            const float asc = act_scales[s_src[r]];
            if constexpr (SCATTER) {
                const int token = s_offs[r] / out_top_k;
                const float w = topk_weights[s_offs[r]];
                atomicAdd(&output_scatter[(long)token * N + nc], w * acc[c][r] * asc * wsc);
            } else {
                out[(long)s_pad[r] * N + nc] = __float2half(acc[c][r] * asc * wsc);
            }
        }
    }
}

// shared body for both launchers: act-fp8-quant then the grouped GEMM. SCATTER
// selects the epilogue. For non-scatter, output_scatter/topk_weights are null.
// TileConfig for the consolidated WMMA grouped GEMM: the named `Wmma` kernel's default
// config (BN=64, GTILE=4, A-shuffle = the prior served v6) overlaid with the dev tuning
// env knobs. A_IN_LDS=1 selects the former v5 (A-in-LDS). The GTILE/LDS clamps stay at
// the launch site (they depend on K/group_size/BN).
static inline w4a8_tile::TileConfig make_moe_tile_config(int /*K*/, int /*group_size*/) {
    w4a8_tile::TileConfig cfg;
    cfg.BN = MV5_BN;                 // 64
    if (const char* e = std::getenv("VLLM_W4A8_MOE_BN")) cfg.BN = std::atoi(e);
    cfg.GTILE = 4;
    if (const char* e = std::getenv("VLLM_W4A8_MOE_GTILE")) cfg.GTILE = std::atoi(e);
    cfg.A_IN_LDS = false;            // default = A-shuffle / B-only LDS (the prior v6)
    if (const char* e = std::getenv("VLLM_W4A8_MOE_A_IN_LDS")) cfg.A_IN_LDS = std::atoi(e) != 0;
    return cfg;
}

template<bool SCATTER>
static void run_moe_gemm(
    const at::Tensor& x, const at::Tensor& w_fp8, const at::Tensor& scales,
    const at::Tensor& sorted_token_ids,
    const at::Tensor& expert_ids, const at::Tensor& num_tokens_post_padded,
    __half* out, int N, int num_valid_tokens, int top_k, int block_m,
    int version, float* output_scatter, const float* topk_weights,
    int out_top_k) {
    const int T = x.size(0);
    const int K = x.size(1);
    // W8A8: scales are per-output-channel (E,N), no K-grouping. group_size is just the
    // K-staging tile depth for the WMMA path; pick the largest <=MAX_GROUP_SIZE divisor.
    int group_size = (K < MAX_GROUP_SIZE) ? K : MAX_GROUP_SIZE;
    while (group_size > WMMA_DIM && K % group_size != 0) group_size -= WMMA_DIM;
    const int P = sorted_token_ids.size(0);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    auto x_fp8 = at::empty({T, K}, x.options().dtype(at::kByte));
    auto act_scales = at::empty({T}, x.options().dtype(at::kFloat));
    moe_compute_act_fp8_kernel<<<dim3(T), dim3(ACTQUANT_BLOCK), 0, stream>>>(
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        x_fp8.data_ptr<unsigned char>(), act_scales.data_ptr<float>(), T, K);

    const float* wsc = scales.data_ptr<float>();
    const int* sti = sorted_token_ids.data_ptr<int>();
    const int* eid = expert_ids.data_ptr<int>();
    const int* ntp = num_tokens_post_padded.data_ptr<int>();
    const unsigned char* xf = x_fp8.data_ptr<unsigned char>();
    const float* asc = act_scales.data_ptr<float>();
    const unsigned char* wq = w_fp8.data_ptr<unsigned char>();

    const w4a8::MoeKernel mk = static_cast<w4a8::MoeKernel>(version);
    if (mk == w4a8::MoeKernel::Wmma) {
        // Consolidated WMMA grouped GEMM behind the WmmaFp8 MMA policy + TileConfig
        // (tile_config.h / moe_gemm_tiled.h) — the ONLY WMMA path now. The former v5
        // (A-in-LDS) vs v6 (A-shuffle, B-only LDS) split is cfg.A_IN_LDS, not a version
        // number; default = A-shuffle (the prior served v6). BIT-EXACT to the retired
        // standalone moe_gemm_v5/v6_kernel (test_tiled_bitexact.py: 56/56 across
        // v5/v6 x sym/asym x BN=64/128 incl. the real 35B gemm1/gemm2 shapes).
        // BN tile width: 64 (4 accs/warp) or 128 (8 accs/warp ILP); VLLM_W4A8_MOE_BN.
        // gt6 = groups per __syncthreads (v6 multi-group staging); capped so B LDS <=40KB.
        const w4a8_tile::TileConfig cfg = make_moe_tile_config(K, group_size);
        int bn = cfg.BN;
        int gt6 = cfg.GTILE;
        const int ng = K / group_size;
        if (gt6 < 1) gt6 = 1; if (gt6 > ng) gt6 = ng;
        const size_t per_grp = (size_t)bn * (group_size + LDS_PAD);
        while (gt6 > 1 && per_grp * gt6 > 40960) gt6--;
        // WARPS_N: split BN across extra warps so block_m stays small (minimal padding) while the
        // block fills the 8-warp WGP budget -> better occupancy + less register pressure. Pick the
        // largest WARPS_N in {1,2,4} keeping (block_m/16)*WARPS_N <= MAX_WARPS and WARPS_N <= NFRAG.
        const int nwm = block_m / WMMA_DIM;
        const int nfrag = bn / WMMA_DIM;
        int warps_n = w4a8_tile::MAX_WARPS / nwm;
        if (warps_n > nfrag) warps_n = nfrag;
        warps_n = (warps_n >= 4) ? 4 : (warps_n >= 2 ? 2 : 1);
        dim3 block(nwm * warps_n * 32);
        dim3 grid((N + bn - 1) / bn, P / block_m);
        const size_t smem5 = (size_t)(block_m + bn) * (group_size + LDS_PAD);
        const size_t smem6 = (size_t)bn * (group_size + LDS_PAD) * gt6;  // B only, gtile groups
        using w4a8_tile::WmmaFp8;
        if (!cfg.A_IN_LDS) {   // default: A via warp-shuffle gather, B-only LDS (former v6)
            #define TV6(BN_T, WN_T) w4a8_tile::moe_gemm_tiled_v6_kernel<WmmaFp8, SCATTER, BN_T, WN_T><<<grid, block, smem6, stream>>>( \
                xf, asc, wq, wsc, sti, eid, ntp, out, T, N, K, group_size, \
                top_k, block_m, num_valid_tokens, output_scatter, topk_weights, out_top_k, gt6)
            #define TV6_WN(WN_T) do { if (bn == 128) TV6(128, WN_T); else TV6(MV5_BN, WN_T); } while (0)
            if (warps_n == 4) TV6_WN(4); else if (warps_n == 2) TV6_WN(2); else TV6_WN(1);
            #undef TV6_WN
            #undef TV6
        } else {               // A staged in LDS (former v5)
            #define TV5(BN_T, WN_T) w4a8_tile::moe_gemm_tiled_kernel<WmmaFp8, SCATTER, BN_T, WN_T><<<grid, block, smem5, stream>>>( \
                xf, asc, wq, wsc, sti, eid, ntp, out, T, N, K, group_size, \
                top_k, block_m, num_valid_tokens, output_scatter, topk_weights, out_top_k)
            #define TV5_WN(WN_T) do { if (bn == 128) TV5(128, WN_T); else TV5(MV5_BN, WN_T); } while (0)
            if (warps_n == 4) TV5_WN(4); else if (warps_n == 2) TV5_WN(2); else TV5_WN(1);
            #undef TV5_WN
            #undef TV5
        }
    } else if (mk == w4a8::MoeKernel::Gemv) {
        // grouped GEMV (decode): NWARPS cols/block, K-tiled LDS activations.
        // Each expert block has at most min(block_m, T) real rows (a token routes
        // to an expert at most once) -> size LDS by that, not the padded block_m,
        // so occupancy isn't throttled by padding at decode.
        // short-K (gemm2) likes fewer warps + COLS pipelining; long-K (gemm1) likes
        // more warps + COLS=1 (K already pipelines). Auto-pick by K; env overrides.
        int nw = (K <= 1024) ? 8 : 32;
        if (const char* e = std::getenv("VLLM_W4A8_MOE_GEMV_NWARPS")) nw = std::atoi(e);
        if (nw != 4 && nw != 8 && nw != 16) nw = 32;   // valid template instantiations
        const int mreal_cap = (block_m < T) ? block_m : T;
        int bk = 0;
        if (const char* e = std::getenv("VLLM_W4A8_MOE_GEMV_BK")) bk = std::atoi(e);
        if (bk <= 0) {  // FULL K (1 chunk -> no per-chunk __syncthreads, the big win
            // across all M) unless mreal_cap*K would exceed ~48KB LDS, then chunk.
            bk = (49152 / mreal_cap / group_size) * group_size;
            if (bk < group_size) bk = group_size;
        }
        if (bk > K) bk = K;
        // COLS columns/warp: short K (gemm2) can't pipeline 1 read/warp -> issue
        // COLS reads to hide latency. gemm1 (large K) already pipelines -> COLS=1.
        // Taper COLS as mreal_cap grows: more real rows give their own latency
        // hiding AND raise the acc[COLS][MMAX] register cost (MMAX=mreal_cap), so
        // big COLS at large M just kills occupancy. (decode T<=8 -> 4; T<=16 -> 2.)
        int cols = 1;
        if (K <= 1024) cols = (mreal_cap <= 8) ? 4 : (mreal_cap <= 16 ? 2 : 1);
        if (const char* e = std::getenv("VLLM_W4A8_MOE_GEMV_COLS")) cols = std::atoi(e);
        if (cols != 1 && cols != 2 && cols != 4 && cols != 8) cols = 1;
        const int per_block = nw * cols;
        dim3 grid((N + per_block - 1) / per_block, P / block_m);
        const size_t smem7 = (size_t)mreal_cap * bk;
        #define GEMV7(NW, MM, CO) moe_gemv_v7_kernel<SCATTER, NW, MM, CO> \
            <<<grid, dim3(NW * 32), smem7, stream>>>( \
            xf, asc, wq, wsc, sti, eid, ntp, out, T, N, K, group_size, \
            top_k, block_m, num_valid_tokens, output_scatter, topk_weights, \
            out_top_k, bk)
        #define GEMV7_CO(NW, MM) do { \
            if (cols == 8) GEMV7(NW, MM, 8); \
            else if (cols == 4) GEMV7(NW, MM, 4); \
            else if (cols == 2) GEMV7(NW, MM, 2); \
            else GEMV7(NW, MM, 1); } while (0)
        /* MMAX = a compile-time cap on real rows/block; M_real <= mreal_cap <= MMAX.
           Tighter MMAX (decode T<=8 -> 8) halves the acc[COLS][MMAX] register
           footprint, lifting occupancy so COLS pipelining actually hides latency. */
        #define GEMV7_MM(NW) do { \
            if (mreal_cap <= 8) GEMV7_CO(NW, 8); \
            else if (mreal_cap <= 16) GEMV7_CO(NW, 16); \
            else if (mreal_cap <= 32) GEMV7_CO(NW, 32); \
            else GEMV7_CO(NW, 64); } while (0)
        if (nw == 4) GEMV7_MM(4);
        else if (nw == 16) GEMV7_MM(16);
        else if (nw == 32) GEMV7_MM(32);
        else GEMV7_MM(8);
        #undef GEMV7_MM
        #undef GEMV7_CO
        #undef GEMV7
    } else {
        dim3 block(MV0_BN, MV0_BM);
        dim3 grid((N + MV0_BN - 1) / MV0_BN, (P + MV0_BM - 1) / MV0_BM);
        moe_gemm_v0_kernel<SCATTER><<<grid, block, 0, stream>>>(
            xf, asc, wq, wsc, sti, eid, ntp, out, T, N, K, group_size,
            top_k, block_m, num_valid_tokens, output_scatter, topk_weights,
            out_top_k);
    }
}

// ============================================================
// Contention-free MoE reduce: replaces the gemm2 atomic-scatter epilogue at
// scale. gemm2 runs NON-scatter -> (P,N); then (a) build the slot->padded-row
// inverse map (one thread per padded row), (b) per (token,n) gather the token's
// top_k rows and weighted-sum. No atomics, coalesced over N. Microbench: the
// atomic scatter cost +3.7ms at T=2048 (top_k-contended global atomics); this
// removes it. (See profiling/kernel-wip/VALIDATION.md.)
// ============================================================
__global__ void moe_build_inverse_map_kernel(
    const int* __restrict__ sorted_token_ids,       // (P,)
    const int* __restrict__ num_tokens_post_padded,  // (1,)
    int*       __restrict__ slot_to_row,             // (M*top_k,) pre-filled -1
    int P, int M_tk) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= P || r >= num_tokens_post_padded[0]) return;
    const int slot = sorted_token_ids[r];
    if (slot < M_tk) slot_to_row[slot] = r;          // valid (token,slot) -> its row
}

__global__ void moe_gather_reduce_kernel(
    const __half* __restrict__ out2,         // (P, N) gemm2 result (non-scatter)
    const int*    __restrict__ slot_to_row,  // (M*top_k,), -1 = missing
    const float*  __restrict__ topk_weights, // (M*top_k,)
    float*        __restrict__ out,          // (M, N) fp32
    int M, int N, int top_k) {
    const int t = blockIdx.y;
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    const int base = t * top_k;
    float acc = 0.0f;
    for (int k = 0; k < top_k; ++k) {
        const int r = slot_to_row[base + k];
        if (r >= 0)
            acc += topk_weights[base + k] * __half2float(out2[(long)r * N + n]);
    }
    out[(long)t * N + n] = acc;
}

}  // namespace w4a8_fp8_moe

// run body for the fused gemm1+silu: act-fp8-quant then the gate/up two-slab GEMM
// with the silu_and_mul epilogue, writing (P, inter). gemm1-only / non-scatter.
static void run_moe_gemm1_silu(
    const at::Tensor& x, const at::Tensor& w_fp8, const at::Tensor& scales,
    const at::Tensor& sorted_token_ids,
    const at::Tensor& expert_ids, const at::Tensor& num_tokens_post_padded,
    __half* out, int N, int inter, int num_valid_tokens, int top_k, int block_m,
    int version) {
    using namespace w4a8_fp8_moe;
    const int T = x.size(0);
    const int K = x.size(1);
    // W8A8: per-output-channel scales (E,2*inter); group_size is just the K-staging tile.
    int group_size = (K < MAX_GROUP_SIZE) ? K : MAX_GROUP_SIZE;
    while (group_size > WMMA_DIM && K % group_size != 0) group_size -= WMMA_DIM;
    const int P = sorted_token_ids.size(0);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    auto x_fp8 = at::empty({T, K}, x.options().dtype(at::kByte));
    auto act_scales = at::empty({T}, x.options().dtype(at::kFloat));
    moe_compute_act_fp8_kernel<<<dim3(T), dim3(ACTQUANT_BLOCK), 0, stream>>>(
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        x_fp8.data_ptr<unsigned char>(), act_scales.data_ptr<float>(), T, K);

    const float* wsc = scales.data_ptr<float>();
    const int* sti = sorted_token_ids.data_ptr<int>();
    const int* eid = expert_ids.data_ptr<int>();
    const int* ntp = num_tokens_post_padded.data_ptr<int>();
    const unsigned char* xf = x_fp8.data_ptr<unsigned char>();
    const float* asc = act_scales.data_ptr<float>();
    const unsigned char* wq = w_fp8.data_ptr<unsigned char>();

    // Same TileConfig as the unfused WMMA path; the grid spans `inter` (gate
    // columns), each block also pulling the matching up tile. NB: the fused-silu
    // kernels are the hand-written moe_gemm1_silu_v5/v6 (no tiled equivalent) —
    // A-residence still selects between them, now via cfg.A_IN_LDS not a version.
    const w4a8_tile::TileConfig cfg = make_moe_tile_config(K, group_size);
    int bn = cfg.BN;
    int gt6 = cfg.GTILE;
    const int ng = K / group_size;
    if (gt6 < 1) gt6 = 1; if (gt6 > ng) gt6 = ng;
    // two slabs/group (gate + up) -> half the 40KB B budget per group-pair.
    const size_t per_grp = (size_t)2 * bn * (group_size + LDS_PAD);
    while (gt6 > 1 && per_grp * gt6 > 40960) gt6--;
    dim3 block((block_m / WMMA_DIM) * 32);
    dim3 grid((inter + bn - 1) / bn, P / block_m);
    if (!cfg.A_IN_LDS) {  // former v6: A-shuffle, B-only LDS, gtile groups
        const size_t smem6 = per_grp * gt6;  // 2 slabs (gate|up) x gtile groups
        #define G1SILU_V6(BN_T) moe_gemm1_silu_v6_kernel<BN_T><<<grid, block, smem6, stream>>>( \
            xf, asc, wq, wsc, sti, eid, ntp, out, T, N, K, group_size, \
            top_k, block_m, num_valid_tokens, gt6, inter)
        if (bn == 128) G1SILU_V6(128); else G1SILU_V6(MV5_BN);
        #undef G1SILU_V6
    } else {  // former v5: A+B LDS, gate + up B slabs + A tile
        const size_t smem5 = (size_t)(block_m + 2 * bn) * (group_size + LDS_PAD);
        #define G1SILU_V5(BN_T) moe_gemm1_silu_v5_kernel<BN_T><<<grid, block, smem5, stream>>>( \
            xf, asc, wq, wsc, sti, eid, ntp, out, T, N, K, group_size, \
            top_k, block_m, num_valid_tokens, inter)
        if (bn == 128) G1SILU_V5(128); else G1SILU_V5(MV5_BN);
        #undef G1SILU_V5
    }
}

// =================== Host launchers ===================
// Fused gemm1 + silu_and_mul: runs the gated gemm1 (w13, output 2*inter) and the
// silu_and_mul activation in ONE kernel, writing the post-activation (P, inter)
// directly. Drops the separate silu launch + the (P,2*inter) out1 / (P,inter) buf2
// HBM round-trip. v5/v6 only (the WMMA tiles); bit-exact to gemm1 + _C.silu_and_mul.
void launch_mmq_w8a8_moe_gemm1_silu_gfx1201(
    const at::Tensor& x,                 // (T, K) fp16
    const at::Tensor& w_fp8,             // (E, 2*inter, K) e4m3 (w13)
    const at::Tensor& scales,            // (E, 2*inter) f32 per-output-channel
    const at::Tensor& sorted_token_ids,  // (P,) int32
    const at::Tensor& expert_ids,        // (P/block_m,) int32
    const at::Tensor& num_tokens_post_padded,  // (1,) int32
    at::Tensor& out,                     // (P, inter) fp16
    int64_t top_k, int64_t block_m, int64_t kernel) {

    using namespace w4a8_fp8_moe;
    const int N = w_fp8.size(1);              // 2*inter (gate|up)
    const int inter = N / 2;
    const int num_valid_tokens = x.size(0) * (int)top_k;
    const int K = x.size(1);
    TORCH_CHECK(K % WMMA_DIM == 0, "K must be a multiple of 16; got ", K);
    TORCH_CHECK(static_cast<w4a8::MoeKernel>(kernel) == w4a8::MoeKernel::Wmma,
                "fused gemm1+silu needs the 'wmma' kernel; got id ", kernel);
    TORCH_CHECK(block_m % WMMA_DIM == 0 && block_m <= 128,
                "fused gemm1+silu block_m must be mult of 16 and <=128; got ", block_m);
    run_moe_gemm1_silu(
        x, w_fp8, scales, sorted_token_ids, expert_ids,
        num_tokens_post_padded,
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()), N, inter,
        num_valid_tokens, (int)top_k, (int)block_m, (int)kernel);
}

void launch_mmq_w8a8_moe_gemm_gfx1201(
    const at::Tensor& x,                 // (T, K) fp16
    const at::Tensor& w_fp8,             // (E, N, K) e4m3
    const at::Tensor& scales,            // (E, N) f32 per-output-channel
    const at::Tensor& sorted_token_ids,  // (P,) int32
    const at::Tensor& expert_ids,        // (P/block_m,) int32
    const at::Tensor& num_tokens_post_padded,  // (1,) int32
    at::Tensor& out,                     // (P, N) fp16
    int64_t top_k, int64_t block_m, int64_t kernel) {

    using namespace w4a8_fp8_moe;
    const int N = w_fp8.size(1);
    const int num_valid_tokens = x.size(0) * (int)top_k;
    const int K = x.size(1);
    TORCH_CHECK(K % WMMA_DIM == 0, "K must be a multiple of 16; got ", K);
    const w4a8::MoeKernel mk = static_cast<w4a8::MoeKernel>(kernel);
    if (mk == w4a8::MoeKernel::Wmma)
        TORCH_CHECK(block_m % WMMA_DIM == 0 && block_m <= 128,
                    "wmma block_m must be mult of 16 and <=128; got ", block_m);
    if (mk == w4a8::MoeKernel::Gemv)
        TORCH_CHECK(block_m % 8 == 0 && block_m <= 64 && K % 32 == 0,
                    "gemv needs block_m mult of 8 <=64, K%32==0");
    run_moe_gemm</*SCATTER=*/false>(
        x, w_fp8, scales, sorted_token_ids, expert_ids,
        num_tokens_post_padded,
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()), N,
        num_valid_tokens, (int)top_k, (int)block_m, (int)kernel,
        /*output_scatter=*/nullptr, /*topk_weights=*/nullptr, /*out_top_k=*/0);
}

// Fused gemm2 + topk-weight + indirect atomic scatter into the final (M,N) fp32
// output. `x` = the (P, inter) post-activation buffer (identity-gathered),
// `sorted_token_ids` = the gemm1 (token,slot) ids (for token = id/top_k + the
// per-slot topk weight), `output` = pre-zeroed (M, N) fp32 accumulator.
void launch_mmq_w8a8_moe_gemm_scatter_gfx1201(
    const at::Tensor& x,                 // (P, K=inter) fp16
    const at::Tensor& w_fp8,             // (E, N=hidden, K) e4m3
    const at::Tensor& scales,            // (E, N) f32 per-output-channel
    const at::Tensor& sorted_token_ids,  // (P,) int32 (gemm1 ids)
    const at::Tensor& expert_ids,        // (P/block_m,) int32
    const at::Tensor& num_tokens_post_padded,  // (1,) int32
    const at::Tensor& topk_weights,      // (M*top_k,) fp32
    at::Tensor& output,                  // (M, N) fp32, pre-zeroed
    int64_t top_k, int64_t block_m, int64_t kernel) {

    using namespace w4a8_fp8_moe;
    const int N = w_fp8.size(1);
    const int M = output.size(0);
    const int num_valid_tokens = M * (int)top_k;
    const int K = x.size(1);
    TORCH_CHECK(K % WMMA_DIM == 0, "K must be a multiple of 16; got ", K);
    const w4a8::MoeKernel mk = static_cast<w4a8::MoeKernel>(kernel);
    if (mk == w4a8::MoeKernel::Wmma)
        TORCH_CHECK(block_m % WMMA_DIM == 0 && block_m <= 128,
                    "wmma block_m must be mult of 16 and <=128; got ", block_m);
    if (mk == w4a8::MoeKernel::Gemv)
        TORCH_CHECK(block_m % 8 == 0 && block_m <= 64 && K % 32 == 0,
                    "gemv needs block_m mult of 8 <=64, K%32==0");
    run_moe_gemm</*SCATTER=*/true>(
        x, w_fp8, scales, sorted_token_ids, expert_ids,
        num_tokens_post_padded, /*out=*/nullptr, N, num_valid_tokens,
        (int)top_k, (int)block_m, (int)kernel,
        output.data_ptr<float>(), topk_weights.data_ptr<float>(), (int)top_k);
}

// Contention-free MoE reduce (replaces the gemm2 atomic scatter at scale).
// out2 = gemm2 NON-scatter result (P,N) fp16; out = (M,N) fp32 (caller-allocated).
void launch_mmq_w8a8_moe_gather_reduce_gfx1201(
    const at::Tensor& out2,                     // (P, N) fp16
    const at::Tensor& sorted_token_ids,         // (P,) int32
    const at::Tensor& topk_weights,             // (M*top_k,) fp32
    const at::Tensor& num_tokens_post_padded,   // (1,) int32
    at::Tensor& out,                            // (M, N) fp32
    int64_t top_k) {
    using namespace w4a8_fp8_moe;
    const int P = out2.size(0);
    const int N = out2.size(1);
    const int M = out.size(0);
    const int M_tk = M * (int)top_k;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    auto slot_to_row = at::full({M_tk}, -1, out2.options().dtype(at::kInt));
    moe_build_inverse_map_kernel<<<dim3((P + 255) / 256), dim3(256), 0, stream>>>(
        sorted_token_ids.data_ptr<int>(),
        num_tokens_post_padded.data_ptr<int>(),
        slot_to_row.data_ptr<int>(), P, M_tk);

    constexpr int TPB = 256;
    dim3 grid((N + TPB - 1) / TPB, M);
    moe_gather_reduce_kernel<<<grid, dim3(TPB), 0, stream>>>(
        reinterpret_cast<const __half*>(out2.data_ptr<at::Half>()),
        slot_to_row.data_ptr<int>(),
        topk_weights.data_ptr<float>(),
        out.data_ptr<float>(), M, N, (int)top_k);
}
