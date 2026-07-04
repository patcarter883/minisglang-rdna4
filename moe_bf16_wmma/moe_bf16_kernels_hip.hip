// moe_bf16_kernels.hip — UNQUANTIZED (bf16/fp16) grouped/routed MoE expert GEMM for gfx1201 (RDNA4).
//
// Closes the one remaining Triton dependency on the full-HIP MoE path: minisgl's
// fused_moe_kernel_triton / moe_sum_reduce_triton (the bf16/fp16 fallback used when experts are NOT
// quantized — the W4A8/fp8 paths already have HIP WMMA kernels).
//
// The GEMM is C[M,OUT] = A[M,IN] @ w[e][OUT,IN]^T, grouped by expert via the moe_align sorted layout
// (sorted_token_ids + expert_ids), exactly the w4a8 moe_gemm_tiled contract but unquantized (no fp8
// quant, no int4 decode). BOTH MoE GEMMs fit the single [E,OUT,IN] weight contract:
//   gemm1: w1 [E, 2N, K]  -> OUT=2N, IN=K   (SCATTER=false: direct store to the sorted-padded rows)
//   gemm2: w2 [E, K,  N]  -> OUT=K,  IN=N   (SCATTER=true : atomicAdd to [M,K] with topk weight)
// The kernel is templated on the activation/weight element type T (rocwmma::bfloat16_t OR
// rocwmma::float16_t): fp32 accumulate keeps both dtypes numerically faithful.
//
// GRAPH-CAPTURE SAFETY: the launchers ALLOCATE NOTHING and there is no host-sync/.item() on the op
// path. The output tensors (out / out_scatter) are PASSED IN pre-allocated by the caller — see the
// *_out bindings. The scatter epilogue atomic-adds into a caller-zeroed fp32 accumulator, so the only
// requirement inside a captured graph is a `.zero_()` on that static buffer (capturable), never a
// malloc. The inner matmul reuses the PROVEN flash_wmma rocWMMA spine convention (A = matrix_a
// row_major, w read as w^T = matrix_b col_major, fp32 accumulate) — byte-identical WMMA to the
// attention family.
//
// torch's CUDAExtension puts -D__HIP_NO_HALF_CONVERSIONS__=1 on the command line, which breaks
// rocwmma's fp16 vector constructors (pulled in by flash_wmma.h). Undo it before ANY HIP/rocwmma
// header, matching mla_hip/mla_prefill_kernels.hip. torch's hipify rewrites the cuda-named stream API.
#undef __HIP_NO_HALF_CONVERSIONS__
#undef __HIP_NO_HALF_OPERATORS__
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <ATen/hip/HIPContext.h>
#include <torch/extension.h>
#include "flash_wmma.h"

namespace moe_bf16 {

constexpr int WMMA_DIM = 16;               // 16x16x16 tile
constexpr int BK = 32;                     // K-chunk staged per __syncthreads (mult of 16)

// Generic (bf16 OR fp16) rocWMMA fragment aliases — mirror flash_wmma's bf16 spine but parametric on T.
template <typename T>
using FragA = rocwmma::fragment<rocwmma::matrix_a, WMMA_DIM, WMMA_DIM, WMMA_DIM, T, rocwmma::row_major>;
template <typename T>
using FragBt = rocwmma::fragment<rocwmma::matrix_b, WMMA_DIM, WMMA_DIM, WMMA_DIM, T, rocwmma::col_major>;
using FragAcc = flash_wmma::FragAcc;       // rocwmma::accumulator, float

// One block computes a (block_m x BN) tile for ONE expert. block_m/16 warps (M-subtiles), each warp
// owns all NFRAG = BN/16 N-accumulators. Grid = (ceil(OUT/BN), P/block_m).
template <int BN, bool SCATTER, typename T>
__global__ void __launch_bounds__(256) moe_bf16_gemm_kernel(
    const T*   __restrict__ A,                  // (T, IN)   token-major activations
    const T*   __restrict__ w,                  // (E, OUT, IN)
    const int* __restrict__ sorted_token_ids,   // (P,)
    const int* __restrict__ expert_ids,         // (P/block_m,)
    const int* __restrict__ num_tokens_post_padded,
    const float* __restrict__ topk_weights,     // (M*top_k,) or null
    T*         __restrict__ out,                // (P, OUT)  non-scatter
    float*     __restrict__ out_scatter,        // (M, OUT)  scatter accum (fp32)
    int Tn, int OUT, int IN, int top_k, int block_m, int num_valid_tokens,
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
  // ALL block threads cooperate on the LDS staging + epilogue loops (stride = blockDim.x) so every
  // index is touched exactly once. Using only n_warps_m*32 here would let the inactive warps
  // re-process indices the active threads already did — a DOUBLE atomicAdd in the scatter epilogue.
  const int total_threads = blockDim.x;         // = 256
  const int e = expert_ids[block_idx];
  const T* we = w + (long)e * OUT * IN;

  // LDS: T A/B chunk tiles during the K loop, reused as fp32 output tile in the epilogue.
  extern __shared__ __align__(16) unsigned char smem_raw[];
  T* A_tile = reinterpret_cast<T*>(smem_raw);                  // block_m * BK
  T* B_tile = A_tile + block_m * BK;                           // BN * BK

  FragAcc acc[NFRAG];
  const bool active = (warp_m < n_warps_m);
  if (active) {
    #pragma unroll
    for (int f = 0; f < NFRAG; ++f) rocwmma::fill_fragment(acc[f], 0.0f);
  }

  for (int k0 = 0; k0 < IN; k0 += BK) {
    const int bk = min(BK, IN - k0);
    // stage A_tile[r][k] = A[src(r)][k0+k]  (0-pad guard rows / tail-K)
    for (int idx = tid; idx < block_m * BK; idx += total_threads) {
      const int r = idx / BK, k = idx % BK;
      T v = T(0.0f);
      if (k < bk) {
        const int offs = sorted_token_ids[row0 + r];
        if (offs < num_valid_tokens) {
          const int src = SCATTER ? (row0 + r) : (offs / top_k);
          v = A[(long)src * IN + k0 + k];
        }
      }
      A_tile[r * BK + k] = v;
    }
    // stage B_tile[n][k] = we[block_n+n][k0+k]
    for (int idx = tid; idx < BN * BK; idx += total_threads) {
      const int n = idx / BK, k = idx % BK;
      const int an = block_n + n;
      T v = T(0.0f);
      if (k < bk && an < OUT) v = we[(long)an * IN + k0 + k];
      B_tile[n * BK + k] = v;
    }
    __syncthreads();

    if (active) {
      const int ksteps = bk / WMMA_DIM;         // bk is a multiple of 16 for MoE shapes
      for (int kt = 0; kt < ksteps; ++kt) {
        FragA<T> af;                             // matrix_a, row_major (A rows)
        rocwmma::load_matrix_sync(af, A_tile + (warp_m * WMMA_DIM) * BK + kt * WMMA_DIM, BK);
        #pragma unroll
        for (int f = 0; f < NFRAG; ++f) {
          FragBt<T> bf;                          // matrix_b, col_major (w read as w^T)
          rocwmma::load_matrix_sync(bf, B_tile + (f * WMMA_DIM) * BK + kt * WMMA_DIM, BK);
          rocwmma::mma_sync(acc[f], af, bf, acc[f]);
        }
      }
    }
    __syncthreads();
  }

  // epilogue: dump fp32 accumulators to LDS, then map sorted-padded rows -> output.
  float* sOut = reinterpret_cast<float*>(smem_raw);            // block_m * BN floats (>= A+B bytes)
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
    float val = sOut[r * BN + n];
    if constexpr (SCATTER) {
      const int token = offs / out_top_k;
      const float wgt = (topk_weights != nullptr) ? topk_weights[offs] : 1.0f;
      atomicAdd(&out_scatter[(long)token * OUT + an], wgt * val);
    } else {
      if (mul_weight && topk_weights != nullptr) val *= topk_weights[offs];
      out[(long)row_pad * OUT + an] = T(val);
    }
  }
}

// ---- launchers -----------------------------------------------------------------------------------
static inline size_t smem_bytes(int block_m, int BN) {
  size_t ab = (size_t)(block_m + BN) * BK * sizeof(uint16_t);  // A_tile + B_tile (bf16/fp16 = 2 bytes)
  size_t oc = (size_t)block_m * BN * sizeof(float);            // fp32 epilogue tile
  return ab > oc ? ab : oc;
}

template <int BN, typename T>
static void launch_bn(const at::Tensor& A, const at::Tensor& w,
                      const at::Tensor& sorted_token_ids, const at::Tensor& expert_ids,
                      const at::Tensor& num_tokens_post_padded,
                      const c10::optional<at::Tensor>& topk_weights,
                      at::Tensor& out, at::Tensor& out_scatter,
                      int OUT, int IN, int top_k, int block_m, int num_valid_tokens,
                      bool scatter, int mul_weight, int out_top_k) {
  const int Tn = A.size(0);
  const int P = sorted_token_ids.size(0);
  dim3 grid((OUT + BN - 1) / BN, P / block_m);
  dim3 block(256);
  size_t shmem = smem_bytes(block_m, BN);
  auto stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();  // hipified by torch cpp_extension
  const T* Ap = reinterpret_cast<const T*>(A.data_ptr());
  const T* wp = reinterpret_cast<const T*>(w.data_ptr());
  const float* twp = topk_weights.has_value() ? topk_weights->data_ptr<float>() : nullptr;
  // Pointer selection keys off `scatter`, not tensor.numel(): the *_out bindings pass the SAME tensor
  // for the unused slot (no dummy allocation), and the unused slot is never dereferenced.
  T* op = scatter ? nullptr : reinterpret_cast<T*>(out.data_ptr());
  float* osp = scatter ? out_scatter.data_ptr<float>() : nullptr;
  if (scatter) {
    hipLaunchKernelGGL((moe_bf16_gemm_kernel<BN, true, T>), grid, block, shmem, stream,
        Ap, wp, sorted_token_ids.data_ptr<int>(), expert_ids.data_ptr<int>(),
        num_tokens_post_padded.data_ptr<int>(), twp, op, osp,
        Tn, OUT, IN, top_k, block_m, num_valid_tokens, mul_weight, out_top_k);
  } else {
    hipLaunchKernelGGL((moe_bf16_gemm_kernel<BN, false, T>), grid, block, shmem, stream,
        Ap, wp, sorted_token_ids.data_ptr<int>(), expert_ids.data_ptr<int>(),
        num_tokens_post_padded.data_ptr<int>(), twp, op, osp,
        Tn, OUT, IN, top_k, block_m, num_valid_tokens, mul_weight, out_top_k);
  }
}

template <int BN>
static void launch_bn_dtype(const at::Tensor& A, const at::Tensor& w,
                            const at::Tensor& sorted_token_ids, const at::Tensor& expert_ids,
                            const at::Tensor& num_tokens_post_padded,
                            const c10::optional<at::Tensor>& topk_weights,
                            at::Tensor& out, at::Tensor& out_scatter,
                            int OUT, int IN, int top_k, int block_m, int num_valid_tokens,
                            bool scatter, int mul_weight, int out_top_k) {
  if (A.scalar_type() == at::kBFloat16) {
    launch_bn<BN, rocwmma::bfloat16_t>(A, w, sorted_token_ids, expert_ids, num_tokens_post_padded,
        topk_weights, out, out_scatter, OUT, IN, top_k, block_m, num_valid_tokens, scatter,
        mul_weight, out_top_k);
  } else if (A.scalar_type() == at::kHalf) {
    launch_bn<BN, rocwmma::float16_t>(A, w, sorted_token_ids, expert_ids, num_tokens_post_padded,
        topk_weights, out, out_scatter, OUT, IN, top_k, block_m, num_valid_tokens, scatter,
        mul_weight, out_top_k);
  } else {
    TORCH_CHECK(false, "moe_bf16: only bf16/fp16 activations supported, got ", A.scalar_type());
  }
}

void launch_moe_bf16_gemm(const at::Tensor& A, const at::Tensor& w,
                          const at::Tensor& sorted_token_ids, const at::Tensor& expert_ids,
                          const at::Tensor& num_tokens_post_padded,
                          const c10::optional<at::Tensor>& topk_weights,
                          at::Tensor& out, at::Tensor& out_scatter,
                          int64_t OUT, int64_t IN, int64_t top_k, int64_t block_m,
                          int64_t num_valid_tokens, int64_t BN, bool scatter,
                          int64_t mul_weight, int64_t out_top_k) {
  switch (BN) {
    case 32:  launch_bn_dtype<32>(A, w, sorted_token_ids, expert_ids, num_tokens_post_padded,
                  topk_weights, out, out_scatter, OUT, IN, top_k, block_m, num_valid_tokens, scatter,
                  mul_weight, out_top_k); break;
    case 64:  launch_bn_dtype<64>(A, w, sorted_token_ids, expert_ids, num_tokens_post_padded,
                  topk_weights, out, out_scatter, OUT, IN, top_k, block_m, num_valid_tokens, scatter,
                  mul_weight, out_top_k); break;
    case 128: launch_bn_dtype<128>(A, w, sorted_token_ids, expert_ids, num_tokens_post_padded,
                  topk_weights, out, out_scatter, OUT, IN, top_k, block_m, num_valid_tokens, scatter,
                  mul_weight, out_top_k); break;
    default:  TORCH_CHECK(false, "moe_bf16: unsupported BN ", BN, " (want 32/64/128)");
  }
}

}  // namespace moe_bf16
