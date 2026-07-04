// moe_w8a16_kernels.hip — W8A16 grouped/routed MoE expert GEMM for gfx1201 (RDNA4).
//
// fp8 (e4m3) WEIGHTS × bf16 ACTIVATIONS. The serving win for the fused-TiDAR path: OLDMOE=1 keeps
// bf16-activation correctness but dequants the WHOLE (E,N,K) fp8 stack to bf16 per forward and runs a
// Triton bf16 MoE over ALL experts (~284ms/step). This kernel keeps the bf16-act correctness while
// (a) touching ONLY the routed/activated experts (moe_align grouped) and (b) dequanting the fp8 weight
// tile to bf16 IN-REGISTER during B-staging (no full-stack materialization) — so it runs at the native
// grouped-WMMA speed (~55ms class) WITHOUT the fp8-activation precision loss of the W8A8 path.
//
// Structurally identical to moe_bf16_kernels.hip (same flash_wmma rocWMMA bf16 spine, same moe_align
// sorted-padded contract, same gemm1(store)/gemm2(scatter) epilogues). ONLY difference: the weight is
// e4m3 bytes + a per-output-channel f32 scale; B-staging converts each e4m3 byte to bf16 (EXACT —
// e4m3's 4-exp/3-mant fits inside bf16's 8-exp/7-mant) UNSCALED, and the per-N weight scale is applied
// once in the epilogue (it factors out of the K-sum), mirroring w8a8_fp8_wmma/moe_gemm_tiled.h.
#undef __HIP_NO_HALF_CONVERSIONS__
#undef __HIP_NO_HALF_OPERATORS__
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <ATen/hip/HIPContext.h>
#include <torch/extension.h>
#include "flash_wmma.h"

namespace moe_w8a16 {

using bf16_t = flash_wmma::bf16_t;         // rocwmma::bfloat16_t
constexpr int WMMA_DIM = 16;               // 16x16x16 tile
constexpr int BK = 32;                     // K-chunk staged per __syncthreads (mult of 16)

// e4m3 -> f32 (lowers to v_cvt_f32_fp8 on gfx12); exact widening.
__device__ __forceinline__ float e4m3_to_f32(unsigned char b) {
    return __builtin_amdgcn_cvt_f32_fp8(static_cast<int>(b), 0);
}

// One block computes a (block_m x BN) tile for ONE expert. block_m/16 warps (M-subtiles), each warp
// owns all NFRAG = BN/16 N-accumulators. Grid = (ceil(OUT/BN), P/block_m).
template <int BN, bool SCATTER>
__global__ void __launch_bounds__(256) moe_w8a16_gemm_kernel(
    const bf16_t*        __restrict__ A,        // (T, IN)   token-major bf16 activations
    const unsigned char* __restrict__ w_fp8,    // (E, OUT, IN)  e4m3 weights
    const float*         __restrict__ w_scales, // (E, OUT)      per-output-channel f32 scale
    const int*           __restrict__ sorted_token_ids,// (P,)
    const int*           __restrict__ expert_ids,      // (P/block_m,)
    const int*           __restrict__ num_tokens_post_padded,
    const float*         __restrict__ topk_weights,    // (M*top_k,) or null
    bf16_t*              __restrict__ out,             // (P, OUT)  non-scatter
    float*               __restrict__ out_scatter,     // (M, OUT)  scatter accum (fp32)
    int T, int OUT, int IN, int top_k, int block_m, int num_valid_tokens,
    int mul_weight, int out_top_k) {

  static_assert(BN % WMMA_DIM == 0, "BN must be a multiple of 16");
  constexpr int NFRAG = BN / WMMA_DIM;

  const int block_idx = blockIdx.y;
  const int row0 = block_idx * block_m;
  if (row0 >= num_tokens_post_padded[0]) return;

  const int block_n = blockIdx.x * BN;
  const int tid = threadIdx.x;
  const int warp_m = tid >> 5;                  // one warp per 16-row M-subtile
  const int n_warps_m = block_m / WMMA_DIM;
  const int total_threads = blockDim.x;         // = 256 (all threads cooperate on staging/epilogue)
  const int e = expert_ids[block_idx];
  const unsigned char* we = w_fp8 + (long)e * OUT * IN;   // e4m3 weights for this expert
  const float* wse = w_scales + (long)e * OUT;            // per-output-channel scale

  // LDS: bf16 A/B chunk tiles during the K loop, reused as fp32 output tile in the epilogue.
  extern __shared__ __align__(16) unsigned char smem_raw[];
  bf16_t* A_tile = reinterpret_cast<bf16_t*>(smem_raw);        // block_m * BK
  bf16_t* B_tile = A_tile + block_m * BK;                       // BN * BK

  flash_wmma::FragAcc acc[NFRAG];
  const bool active = (warp_m < n_warps_m);
  if (active) {
    #pragma unroll
    for (int f = 0; f < NFRAG; ++f) rocwmma::fill_fragment(acc[f], 0.0f);
  }

  for (int k0 = 0; k0 < IN; k0 += BK) {
    const int bk = min(BK, IN - k0);
    // stage A_tile[r][k] = A[src(r)][k0+k]  (bf16 activation, 0-pad guard rows / tail-K)
    for (int idx = tid; idx < block_m * BK; idx += total_threads) {
      const int r = idx / BK, k = idx % BK;
      bf16_t v = bf16_t(0.0f);
      if (k < bk) {
        const int offs = sorted_token_ids[row0 + r];
        if (offs < num_valid_tokens) {
          const int src = SCATTER ? (row0 + r) : (offs / top_k);
          v = A[(long)src * IN + k0 + k];
        }
      }
      A_tile[r * BK + k] = v;
    }
    // stage B_tile[n][k] = dequant(we[block_n+n][k0+k]) : e4m3 byte -> bf16 (EXACT, unscaled)
    for (int idx = tid; idx < BN * BK; idx += total_threads) {
      const int n = idx / BK, k = idx % BK;
      const int an = block_n + n;
      bf16_t v = bf16_t(0.0f);
      if (k < bk && an < OUT) v = bf16_t(e4m3_to_f32(we[(long)an * IN + k0 + k]));
      B_tile[n * BK + k] = v;
    }
    __syncthreads();

    if (active) {
      const int ksteps = bk / WMMA_DIM;         // bk is a multiple of 16 for MoE shapes
      for (int kt = 0; kt < ksteps; ++kt) {
        flash_wmma::FragQ af;                    // matrix_a, row_major (A rows)
        rocwmma::load_matrix_sync(af, A_tile + (warp_m * WMMA_DIM) * BK + kt * WMMA_DIM, BK);
        #pragma unroll
        for (int f = 0; f < NFRAG; ++f) {
          flash_wmma::FragKt bf;                 // matrix_b, col_major (w read as w^T)
          rocwmma::load_matrix_sync(bf, B_tile + (f * WMMA_DIM) * BK + kt * WMMA_DIM, BK);
          rocwmma::mma_sync(acc[f], af, bf, acc[f]);
        }
      }
    }
    __syncthreads();
  }

  // epilogue: dump fp32 accumulators to LDS, apply per-N weight scale, map sorted-padded rows -> out.
  float* sOut = reinterpret_cast<float*>(smem_raw);            // block_m * BN floats (>= A+B bf16 bytes)
  if (active) {
    #pragma unroll
    for (int f = 0; f < NFRAG; ++f)
      rocwmma::store_matrix_sync(sOut + (warp_m * WMMA_DIM) * BN + f * WMMA_DIM, acc[f], BN,
                                 rocwmma::mem_row_major);
  }
  __syncthreads();

  for (int idx = tid; idx < block_m * BN; idx += total_threads) {
    const int r = idx / BN, n = idx % BN;
    const int an = block_n + n;
    if (an >= OUT) continue;
    const int row_pad = row0 + r;
    const int offs = sorted_token_ids[row_pad];
    if (offs >= num_valid_tokens) continue;
    float val = sOut[r * BN + n] * wse[an];      // W8A16: apply per-output-channel weight scale here
    if constexpr (SCATTER) {
      const int token = offs / out_top_k;
      const float wgt = (topk_weights != nullptr) ? topk_weights[offs] : 1.0f;
      atomicAdd(&out_scatter[(long)token * OUT + an], wgt * val);
    } else {
      if (mul_weight && topk_weights != nullptr) val *= topk_weights[offs];
      out[(long)row_pad * OUT + an] = bf16_t(val);
    }
  }
}

// ---- launchers -----------------------------------------------------------------------------------
static inline size_t smem_bytes(int block_m, int BN) {
  size_t ab = (size_t)(block_m + BN) * BK * sizeof(bf16_t);   // A_tile + B_tile
  size_t oc = (size_t)block_m * BN * sizeof(float);           // fp32 epilogue tile
  return ab > oc ? ab : oc;
}

template <int BN>
static void launch_bn(const at::Tensor& A, const at::Tensor& w_fp8, const at::Tensor& w_scales,
                      const at::Tensor& sorted_token_ids, const at::Tensor& expert_ids,
                      const at::Tensor& num_tokens_post_padded,
                      const c10::optional<at::Tensor>& topk_weights,
                      at::Tensor& out, at::Tensor& out_scatter,
                      int OUT, int IN, int top_k, int block_m, int num_valid_tokens,
                      bool scatter, int mul_weight, int out_top_k) {
  const int T = A.size(0);
  const int P = sorted_token_ids.size(0);
  dim3 grid((OUT + BN - 1) / BN, P / block_m);
  dim3 block(256);
  size_t shmem = smem_bytes(block_m, BN);
  auto stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
  const bf16_t* Ap = reinterpret_cast<const bf16_t*>(A.data_ptr<at::BFloat16>());
  const unsigned char* wp = reinterpret_cast<const unsigned char*>(w_fp8.data_ptr<uint8_t>());
  const float* wsp = w_scales.data_ptr<float>();
  const float* twp = topk_weights.has_value() ? topk_weights->data_ptr<float>() : nullptr;
  bf16_t* op = out.numel() ? reinterpret_cast<bf16_t*>(out.data_ptr<at::BFloat16>()) : nullptr;
  float* osp = out_scatter.numel() ? out_scatter.data_ptr<float>() : nullptr;
  if (scatter) {
    hipLaunchKernelGGL((moe_w8a16_gemm_kernel<BN, true>), grid, block, shmem, stream,
        Ap, wp, wsp, sorted_token_ids.data_ptr<int>(), expert_ids.data_ptr<int>(),
        num_tokens_post_padded.data_ptr<int>(), twp, op, osp,
        T, OUT, IN, top_k, block_m, num_valid_tokens, mul_weight, out_top_k);
  } else {
    hipLaunchKernelGGL((moe_w8a16_gemm_kernel<BN, false>), grid, block, shmem, stream,
        Ap, wp, wsp, sorted_token_ids.data_ptr<int>(), expert_ids.data_ptr<int>(),
        num_tokens_post_padded.data_ptr<int>(), twp, op, osp,
        T, OUT, IN, top_k, block_m, num_valid_tokens, mul_weight, out_top_k);
  }
}

void launch_moe_w8a16_gemm(const at::Tensor& A, const at::Tensor& w_fp8, const at::Tensor& w_scales,
                           const at::Tensor& sorted_token_ids, const at::Tensor& expert_ids,
                           const at::Tensor& num_tokens_post_padded,
                           const c10::optional<at::Tensor>& topk_weights,
                           at::Tensor& out, at::Tensor& out_scatter,
                           int64_t OUT, int64_t IN, int64_t top_k, int64_t block_m,
                           int64_t num_valid_tokens, int64_t BN, bool scatter,
                           int64_t mul_weight, int64_t out_top_k) {
  switch (BN) {
    case 32:  launch_bn<32>(A, w_fp8, w_scales, sorted_token_ids, expert_ids, num_tokens_post_padded,
                            topk_weights, out, out_scatter, OUT, IN, top_k, block_m, num_valid_tokens,
                            scatter, mul_weight, out_top_k); break;
    case 64:  launch_bn<64>(A, w_fp8, w_scales, sorted_token_ids, expert_ids, num_tokens_post_padded,
                            topk_weights, out, out_scatter, OUT, IN, top_k, block_m, num_valid_tokens,
                            scatter, mul_weight, out_top_k); break;
    case 128: launch_bn<128>(A, w_fp8, w_scales, sorted_token_ids, expert_ids, num_tokens_post_padded,
                             topk_weights, out, out_scatter, OUT, IN, top_k, block_m, num_valid_tokens,
                             scatter, mul_weight, out_top_k); break;
    default:  TORCH_CHECK(false, "moe_w8a16: unsupported BN ", BN, " (want 32/64/128)");
  }
}

}  // namespace moe_w8a16
