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

// D[M,N] = A[M,K] @ B[N,K]^T  (the chunked KK / KQ / K·S0 shape). matrix_b col_major over B[N,K]
// (col_major with ld=K reads B^T) — element (k,n) at n*K+k == B[n][k].
__global__ void wmma_gemm_nt_kernel(const rocwmma::float16_t* __restrict__ A,
                                    const rocwmma::float16_t* __restrict__ B,
                                    float* __restrict__ D, int M, int N, int Kd) {
  const int tileN = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
  const int tileM = blockIdx.y * blockDim.y + threadIdx.y;
  const int row = tileM * WM, col = tileN * WN;
  if (row >= M || col >= N) return;
  rocwmma::fragment<accumulator, WM, WN, WK, float> acc;
  rocwmma::fill_fragment(acc, 0.0f);
  for (int i = 0; i < Kd; i += WK) {
    rocwmma::fragment<matrix_a, WM, WN, WK, rocwmma::float16_t, row_major> aFrag;
    rocwmma::fragment<matrix_b, WM, WN, WK, rocwmma::float16_t, col_major> bFrag;
    rocwmma::load_matrix_sync(aFrag, A + row * Kd + i, Kd);   // A[M,K] row-major
    rocwmma::load_matrix_sync(bFrag, B + col * Kd + i, Kd);   // B[N,K]; col_major frag => B^T
    rocwmma::mma_sync(acc, aFrag, bFrag, acc);
  }
  rocwmma::store_matrix_sync(D + row * N + col, acc, N, rocwmma::mem_row_major);
}

// D[M,N] = A[K,M]^T @ B[K,N]  (the U^T·K state-carry shape). matrix_a col_major over A[K,M].
__global__ void wmma_gemm_tn_kernel(const rocwmma::float16_t* __restrict__ A,
                                    const rocwmma::float16_t* __restrict__ B,
                                    float* __restrict__ D, int M, int N, int Kd) {
  const int tileN = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
  const int tileM = blockIdx.y * blockDim.y + threadIdx.y;
  const int row = tileM * WM, col = tileN * WN;
  if (row >= M || col >= N) return;
  rocwmma::fragment<accumulator, WM, WN, WK, float> acc;
  rocwmma::fill_fragment(acc, 0.0f);
  for (int i = 0; i < Kd; i += WK) {
    rocwmma::fragment<matrix_a, WM, WN, WK, rocwmma::float16_t, col_major> aFrag;
    rocwmma::fragment<matrix_b, WM, WN, WK, rocwmma::float16_t, row_major> bFrag;
    rocwmma::load_matrix_sync(aFrag, A + i * M + row, M);     // A[K,M]; col_major frag => A^T
    rocwmma::load_matrix_sync(bFrag, B + i * N + col, N);     // B[K,N] row-major
    rocwmma::mma_sync(acc, aFrag, bFrag, acc);
  }
  rocwmma::store_matrix_sync(D + row * N + col, acc, N, rocwmma::mem_row_major);
}

// Unit-lower-triangular solve U = (I + L)^-1 B, where L is strictly-lower C x C (diagonal implied 1,
// not read) and B is C x N. This is the ONLY non-matmul piece of the chunked gated-delta-rule prefill
// (the (I + tril(beta*M,-1))^-1 step). Per-column forward substitution: thread n owns column n and
// solves U[:,n] sequentially over rows j=0..C-1 holding its column in registers; L lives in LDS,
// shared across all columns. grid = (1), block = N threads. C <= GDN_SOLVE_MAXC.
#ifndef GDN_SOLVE_MAXC
#define GDN_SOLVE_MAXC 16
#endif
__global__ void tri_solve_kernel(const float* __restrict__ L, const float* __restrict__ B,
                                 float* __restrict__ U, int C, int N) {
  const int n = threadIdx.x;
  __shared__ float Ls[GDN_SOLVE_MAXC * GDN_SOLVE_MAXC];
  for (int idx = threadIdx.x; idx < C * C; idx += blockDim.x) Ls[idx] = L[idx];
  __syncthreads();
  if (n >= N) return;
  float u[GDN_SOLVE_MAXC];
  for (int j = 0; j < C; ++j) {
    float s = B[j * N + n];
    for (int i = 0; i < j; ++i) s -= Ls[j * C + i] * u[i];
    u[j] = s;
    U[j * N + n] = s;
  }
}

void launch_tri_solve(const at::Tensor& L, const at::Tensor& B, at::Tensor& U) {
  const int C = L.size(0), N = B.size(1);
  auto s = at::cuda::getCurrentCUDAStream();
  hipLaunchKernelGGL(tri_solve_kernel, dim3(1), dim3(N), 0, s,
      L.data_ptr<float>(), B.data_ptr<float>(), U.data_ptr<float>(), C, N);
}

static dim3 _grid(int M, int N) { return dim3((N + WN - 1) / WN, (M + WM * 4 - 1) / (WM * 4)); }

void launch_wmma_gemm(const at::Tensor& A, const at::Tensor& B, at::Tensor& D) {
  const int M = A.size(0), Kd = A.size(1), N = B.size(1);
  auto s = at::cuda::getCurrentCUDAStream();
  hipLaunchKernelGGL(wmma_gemm_kernel, _grid(M, N), dim3(32, 4), 0, s,
      reinterpret_cast<const rocwmma::float16_t*>(A.data_ptr<at::Half>()),
      reinterpret_cast<const rocwmma::float16_t*>(B.data_ptr<at::Half>()),
      D.data_ptr<float>(), M, N, Kd, Kd, N, N);
}

void launch_wmma_gemm_nt(const at::Tensor& A, const at::Tensor& B, at::Tensor& D) {
  const int M = A.size(0), Kd = A.size(1), N = B.size(0);  // B is [N,K]
  auto s = at::cuda::getCurrentCUDAStream();
  hipLaunchKernelGGL(wmma_gemm_nt_kernel, _grid(M, N), dim3(32, 4), 0, s,
      reinterpret_cast<const rocwmma::float16_t*>(A.data_ptr<at::Half>()),
      reinterpret_cast<const rocwmma::float16_t*>(B.data_ptr<at::Half>()),
      D.data_ptr<float>(), M, N, Kd);
}

void launch_wmma_gemm_tn(const at::Tensor& A, const at::Tensor& B, at::Tensor& D) {
  const int Kd = A.size(0), M = A.size(1), N = B.size(1);  // A is [K,M]
  auto s = at::cuda::getCurrentCUDAStream();
  hipLaunchKernelGGL(wmma_gemm_tn_kernel, _grid(M, N), dim3(32, 4), 0, s,
      reinterpret_cast<const rocwmma::float16_t*>(A.data_ptr<at::Half>()),
      reinterpret_cast<const rocwmma::float16_t*>(B.data_ptr<at::Half>()),
      D.data_ptr<float>(), M, N, Kd);
}
