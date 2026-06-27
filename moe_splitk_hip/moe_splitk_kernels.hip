// moe_splitk_kernels.hip — minisgl-local SPLIT-K variant of the W4A8 fp8 grouped gemm2 (down-proj)
// SCATTER kernel, forked from vllm-gfx1201 w4a8_fp8_wmma (moe_gemm_tiled.h v5 `moe_gemm_tiled_kernel`
// + moe_kernel.hip `moe_compute_act_fp8_kernel`). Task A (#17).
//
// WHY a fork: at decode (M<=2) gemm2 is occupancy-starved — grid = (N/BN, P/block_m) is only a few
// hundred blocks, reaching ~15% of peak weight-read BW. Split-K carves the `num_groups` (K=inter)
// contraction across a new grid.z so block count *= split_k -> more concurrent waves -> more BW.
// The decode SCATTER epilogue already uses atomicAdd, and w/asc are constants independent of K, so
// each z-block atomic-adds w*running_partial*asc and the partials sum to the exact result with NO
// extra reduction kernel. (SCATTER-only: non-scatter would overwrite across z-blocks.)
//
// Lives in minisgl (NOT the shared w4a8_fp8_wmma lib) to avoid image/blast-radius — selected by
// quant/kernels.py:w4a8_moe at decode under MINISGL_MOE_SPLITK. Parity vs the vendored scatter
// kernel: tools/moe_parity.py. Bit-exact base is moe_gemm_tiled_kernel (v5, A-in-LDS).

#undef __HIP_NO_HALF_CONVERSIONS__
#undef __HIP_NO_HALF_OPERATORS__
#include <algorithm>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include "tile_config.h"

namespace moe_splitk {

using namespace w4a8_tile;
constexpr int ACTQUANT_BLOCK = 256;

// Per-row symmetric fp8(e4m3) activation quant: x[T,K] fp16 -> x_fp8[T,K] + act_scales[T].
// Verbatim from w4a8_fp8_wmma/moe_kernel.hip (the gemm2 act-quant the scatter path runs first).
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

// SPLIT-K v5 grouped GEMM (SCATTER). Identical staging/accumulation/epilogue to
// w4a8_tile::moe_gemm_tiled_kernel EXCEPT the group loop runs only this z-block's slice
// [g_lo, g_hi) of num_groups, and the scatter atomicAdd composes the per-z partials.
template<class MMA, bool SCATTER, int BN, int WARPS_N = 1>
__global__ void __launch_bounds__(MAX_THREADS) moe_gemm_splitk_kernel(
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
    int out_top_k, bool weight_is_e2m1, int split_k) {

    static_assert(BN % WMMA_DIM == 0, "BN must be a multiple of WMMA_DIM");
    constexpr int NFRAG = BN / WMMA_DIM;
    constexpr int NFRAG_W = NFRAG / WARPS_N;

    const int block_idx = blockIdx.y;               // one moe_align block (one expert)
    const int row0 = block_idx * block_m;
    const int P_valid = num_tokens_post_padded[0];
    if (row0 >= P_valid) return;

    const int num_groups = K / group_size;
    const int gpz = (num_groups + split_k - 1) / split_k;   // groups per z-split
    const int g_lo = blockIdx.z * gpz;
    if (g_lo >= num_groups) return;
    const int g_hi = (g_lo + gpz < num_groups) ? (g_lo + gpz) : num_groups;

    const int block_n = blockIdx.x * BN;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int lrow = lane & 15;
    const int khalf = (lane >> 4) * 8;
    const int frag_col = lane & 15;
    const int frag_row0 = (lane >> 4) * 8;
    const int n_warps_m = block_m / WMMA_DIM;
    const int warp_m = warp_id / WARPS_N;
    const int warp_n = warp_id % WARPS_N;
    if (warp_m >= n_warps_m) return;
    const int nfrag0 = warp_n * NFRAG_W;

    const int e = expert_ids[block_idx];
    const int ppr = K / PACK_FACTOR;
    const int* wq_e = w_packed + (long)e * N * ppr;
    const __half* ws_e = w_scales + (long)e * N * num_groups;
    const int* wz_e = (w_zeros != nullptr)
                      ? w_zeros + (long)e * (N / 8) * num_groups : nullptr;

    const int BK = group_size;
    const int LDSBK = BK + LDS_PAD;
    extern __shared__ unsigned char smem[];
    unsigned char* A_tile = smem;                   // block_m rows
    unsigned char* B_tile = smem + block_m * LDSBK; // BN cols

    float running[NFRAG_W][8];
    #pragma unroll
    for (int f = 0; f < NFRAG_W; ++f)
        #pragma unroll
        for (int ee = 0; ee < 8; ++ee) running[f][ee] = 0.0f;

    const int total_threads = n_warps_m * WARPS_N * 32;
    for (int g = g_lo; g < g_hi; ++g) {
        const int k0 = g * BK;

        // ---- stage activations: A_tile[r][k] = x_fp8[src(r)][k0+k], guarded ----
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
                    ? decode_w4_to_e4m3((word >> (jj * 4)) & 0xF, zp, weight_is_e2m1) : 0;
        }
        __syncthreads();

        typename MMA::CFrag acc[NFRAG_W];
        #pragma unroll
        for (int f = 0; f < NFRAG_W; ++f) acc[f] = MMA::zero();

        const int k_sub = BK / WMMA_DIM;
        const int a_base = (warp_m * WMMA_DIM + lrow) * LDSBK + khalf;
        const int b_base = lrow * LDSBK + khalf;
        for (int kt = 0; kt < k_sub; ++kt) {
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
            const float wsc = (abs_n < N)
                ? __half2float(ws_e[abs_n * num_groups + g]) : 0.0f;
            #pragma unroll
            for (int ee = 0; ee < 8; ++ee) running[f][ee] += acc[f][ee] * wsc;
        }
        __syncthreads();
    }

    // ---- epilogue: per-row act scale, atomic-scatter the partial (SCATTER only) ----
    #pragma unroll
    for (int ee = 0; ee < 8; ++ee) {
        const int r = warp_m * WMMA_DIM + frag_row0 + ee;
        if (r >= block_m) continue;
        const int row_pad = row0 + r;
        const int offs_token = sorted_token_ids[row_pad];
        if (offs_token >= num_valid_tokens) continue;
        const int src = SCATTER ? row_pad : (offs_token / top_k);
        const float asc = act_scales[src];
        if constexpr (SCATTER) {
            const int token = offs_token / out_top_k;
            const float w = topk_weights[offs_token];
            // All split-K z-blocks atomic-add into the same (token,n) (single buffer). The added
            // split_k-way atomic contention is cheaper than the two-pass acc_z alloc+zero+reduce
            // overhead (measured: single-buffer S=2 1.16x vs two-pass 1.08x at M=1). The inherent
            // top_k-way expert reduction was already atomic in the baseline.
            #pragma unroll
            for (int f = 0; f < NFRAG_W; ++f) {
                const int abs_n = block_n + (nfrag0 + f) * WMMA_DIM + frag_col;
                if (abs_n < N)
                    atomicAdd(&output_scatter[(long)token * N + abs_n],
                              w * running[f][ee] * asc);
            }
        } else {
            #pragma unroll
            for (int f = 0; f < NFRAG_W; ++f) {
                const int abs_n = block_n + (nfrag0 + f) * WMMA_DIM + frag_col;
                if (abs_n < N)
                    out[(long)row_pad * N + abs_n] = __float2half(running[f][ee] * asc);
            }
        }
    }
}

}  // namespace moe_splitk

// ---------------------------------------------------------------------------- launcher
// SCATTER gemm2 with split-K. Mirrors launch_mmq_fp8_moe_gemm_scatter_gfx1201 (Wmma/A-in-LDS path)
// + split_k grid.z. weight_is_e2m1=false (AWQ int4; minisgl's w4a8_moe path).
void launch_moe_gemm_splitk_scatter(
    const at::Tensor& x,                 // (P, K=inter) fp16
    const at::Tensor& w_packed,          // (E, N=hidden, K/8) int32
    const at::Tensor& scales,            // (E, N, K/group) fp16
    const at::Tensor& w_zeros,           // (E, N/8, K/group) int32 or empty
    const at::Tensor& sorted_token_ids,  // (P,) int32
    const at::Tensor& expert_ids,        // (P/block_m,) int32
    const at::Tensor& num_tokens_post_padded,  // (1,) int32
    const at::Tensor& topk_weights,      // (M*top_k,) fp32
    at::Tensor& output,                  // (M, N) fp32, pre-zeroed
    int64_t top_k, int64_t block_m, int64_t split_k) {
    using namespace w4a8_tile;
    using namespace moe_splitk;

    const int T = x.size(0);
    const int K = x.size(1);
    const int N = w_packed.size(1);
    const int M = output.size(0);
    const int num_valid_tokens = M * (int)top_k;
    const int group_size = K / scales.size(2);
    const int P = sorted_token_ids.size(0);
    TORCH_CHECK(group_size % WMMA_DIM == 0 && group_size <= MAX_GROUP_SIZE,
                "group_size must be mult of 16 and <=128; got ", group_size);
    TORCH_CHECK(block_m % WMMA_DIM == 0 && block_m <= 128,
                "block_m must be mult of 16 and <=128; got ", block_m);
    const int num_groups = K / group_size;
    int sk = (int)split_k;
    if (sk < 1) sk = 1;
    if (sk > num_groups) sk = num_groups;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    auto x_fp8 = at::empty({T, K}, x.options().dtype(at::kByte));
    auto act_scales = at::empty({T}, x.options().dtype(at::kFloat));
    moe_compute_act_fp8_kernel<<<dim3(T), dim3(ACTQUANT_BLOCK), 0, stream>>>(
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        x_fp8.data_ptr<unsigned char>(), act_scales.data_ptr<float>(), T, K);

    const int* wz = (w_zeros.defined() && w_zeros.numel() > 0)
                    ? w_zeros.data_ptr<int>() : nullptr;
    const __half* wsc = reinterpret_cast<const __half*>(scales.data_ptr<at::Half>());

    constexpr int BN = 64;       // == w4a8_fp8_wmma MV5_BN (4 N-accumulators/warp)
    const int nwm = block_m / WMMA_DIM;
    const int nfrag = BN / WMMA_DIM;
    int warps_n = MAX_WARPS / nwm;
    if (warps_n > nfrag) warps_n = nfrag;
    warps_n = (warps_n >= 4) ? 4 : (warps_n >= 2 ? 2 : 1);
    dim3 block(nwm * warps_n * 32);
    dim3 grid((N + BN - 1) / BN, P / block_m, sk);
    const size_t smem = (size_t)(block_m + BN) * (group_size + LDS_PAD);

    // Single (pre-zeroed) output buffer: all split-K z-blocks atomic-add into it (cheaper than a
    // two-pass per-z buffer + reduce, measured). `output` is pre-zeroed by the caller.
    #define LAUNCH(WN) moe_gemm_splitk_kernel<WmmaFp8, /*SCATTER=*/true, BN, WN> \
        <<<grid, block, smem, stream>>>( \
        x_fp8.data_ptr<unsigned char>(), act_scales.data_ptr<float>(), \
        w_packed.data_ptr<int>(), wsc, wz, sorted_token_ids.data_ptr<int>(), \
        expert_ids.data_ptr<int>(), num_tokens_post_padded.data_ptr<int>(), \
        /*out=*/nullptr, T, N, K, group_size, (int)top_k, (int)block_m, num_valid_tokens, \
        output.data_ptr<float>(), topk_weights.data_ptr<float>(), (int)top_k, \
        /*weight_is_e2m1=*/false, sk)
    if (warps_n == 4) LAUNCH(4); else if (warps_n == 2) LAUNCH(2); else LAUNCH(1);
    #undef LAUNCH
}
