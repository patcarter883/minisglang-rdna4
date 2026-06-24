// Torch binding for the rocWMMA GEMM probe (gfx1201). torch.ops.wmma_probe.gemm(A, B) -> D.
#include <torch/extension.h>
#include <torch/library.h>
#include <ATen/cuda/CUDAContext.h>

void launch_wmma_gemm(const at::Tensor& A, const at::Tensor& B, at::Tensor& D);
void launch_wmma_gemm_nt(const at::Tensor& A, const at::Tensor& B, at::Tensor& D);
void launch_wmma_gemm_tn(const at::Tensor& A, const at::Tensor& B, at::Tensor& D);
void launch_tri_solve(const at::Tensor& L, const at::Tensor& B, at::Tensor& U);

namespace {
at::Tensor gemm(const at::Tensor& A, const at::Tensor& B) {  // A[M,K] @ B[K,N]
  auto D = at::empty({A.size(0), B.size(1)}, A.options().dtype(at::kFloat));
  launch_wmma_gemm(A, B, D);
  return D;
}
at::Tensor gemm_nt(const at::Tensor& A, const at::Tensor& B) {  // A[M,K] @ B[N,K]^T
  auto D = at::empty({A.size(0), B.size(0)}, A.options().dtype(at::kFloat));
  launch_wmma_gemm_nt(A, B, D);
  return D;
}
at::Tensor gemm_tn(const at::Tensor& A, const at::Tensor& B) {  // A[K,M]^T @ B[K,N]
  auto D = at::empty({A.size(1), B.size(1)}, A.options().dtype(at::kFloat));
  launch_wmma_gemm_tn(A, B, D);
  return D;
}
at::Tensor tri_solve(const at::Tensor& L, const at::Tensor& B) {  // (I+L)^-1 @ B, L unit-lower CxC
  auto U = at::empty_like(B);
  launch_tri_solve(L, B, U);
  return U;
}
}  // namespace

TORCH_LIBRARY(wmma_probe, m) {
  m.def("gemm(Tensor A, Tensor B) -> Tensor");
  m.def("gemm_nt(Tensor A, Tensor B) -> Tensor");
  m.def("gemm_tn(Tensor A, Tensor B) -> Tensor");
  m.def("tri_solve(Tensor L, Tensor B) -> Tensor");
}
TORCH_LIBRARY_IMPL(wmma_probe, CUDA, m) {
  m.impl("gemm", gemm);
  m.impl("gemm_nt", gemm_nt);
  m.impl("gemm_tn", gemm_tn);
  m.impl("tri_solve", tri_solve);
}
