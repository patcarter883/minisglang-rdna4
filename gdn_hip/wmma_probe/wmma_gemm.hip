// rocWMMA GEMM primitive probe for gfx1201 (RDNA4) — the de-risking foundation for the WMMA chunked
// gated-delta-rule prefill (gdn_hip #22). Proves rocWMMA compiles + runs correctly in our
// torch-extension/hipcc setup before the full chunked kernel is built on top.
//
// D[M,N] = A[M,K] @ B[K,N], fp16 inputs / fp32 accumulate, 16x16x16 tiles, one wave per 16x16 tile.
// (RDNA4 is wave32; rocWMMA handles the fragment/lane layout.) Pattern from the migrated W4A8
// kernel's wmma_peak_fp16.hip (vllm-gfx1201/w4a8_fp8_wmma/), incl. the half-op undef gotcha:
// the torch cpp_extension build passes -D__HIP_NO_HALF_*=1 which breaks float16_t ops, so undo it.
#undef __HIP_NO_HALF_CONVERSIONS__
#undef __HIP_NO_HALF_OPERATORS__
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

using rocwmma::accumulator;
using rocwmma::col_major;
using rocwmma::row_major;
using rocwmma::matrix_a;
using rocwmma::matrix_b;

static constexpr int WM = 16, WN = 16, WK = 16;

__global__ void wmma_gemm_kernel(const rocwmma::float16_t* __restrict__ A,
                                 const rocwmma::float16_t* __restrict__ B,
                                 float* __restrict__ D, int M, int N, int Kd,
                                 int lda, int ldb, int ldd) {
  // one wave per 16x16 output tile; grid.x over N tiles, grid.y over M tiles
  const int tileN = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;  // wave index along N
  const int tileM = blockIdx.y * blockDim.y + threadIdx.y;               // tile index along M
  const int row = tileM * WM, col = tileN * WN;
  if (row >= M || col >= N) return;

  rocwmma::fragment<accumulator, WM, WN, WK, float> acc;
  rocwmma::fill_fragment(acc, 0.0f);
  for (int i = 0; i < Kd; i += WK) {
    rocwmma::fragment<matrix_a, WM, WN, WK, rocwmma::float16_t, row_major> aFrag;
    rocwmma::fragment<matrix_b, WM, WN, WK, rocwmma::float16_t, row_major> bFrag;
    rocwmma::load_matrix_sync(aFrag, A + row * lda + i, lda);   // A row-major MxK
    rocwmma::load_matrix_sync(bFrag, B + i * ldb + col, ldb);   // B row-major KxN
    rocwmma::mma_sync(acc, aFrag, bFrag, acc);
  }
  rocwmma::store_matrix_sync(D + row * ldd + col, acc, ldd, rocwmma::mem_row_major);
}

void launch_wmma_gemm(const at::Tensor& A, const at::Tensor& B, at::Tensor& D) {
  const int M = A.size(0), Kd = A.size(1), N = B.size(1);
  dim3 block(32, 4);                         // 1 wave (32) along N, 4 tiles along M
  dim3 grid((N + WN - 1) / WN, (M + WM * 4 - 1) / (WM * 4));
  auto stream = at::cuda::getCurrentCUDAStream();
  hipLaunchKernelGGL(wmma_gemm_kernel, grid, block, 0, stream,
      reinterpret_cast<const rocwmma::float16_t*>(A.data_ptr<at::Half>()),
      reinterpret_cast<const rocwmma::float16_t*>(B.data_ptr<at::Half>()),
      D.data_ptr<float>(), M, N, Kd, Kd, N, N);
}
