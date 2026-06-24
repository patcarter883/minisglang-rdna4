// Torch binding for the rocWMMA GEMM probe (gfx1201). torch.ops.wmma_probe.gemm(A, B) -> D.
#include <torch/extension.h>
#include <torch/library.h>
#include <ATen/cuda/CUDAContext.h>

void launch_wmma_gemm(const at::Tensor& A, const at::Tensor& B, at::Tensor& D);

namespace {
at::Tensor gemm(const at::Tensor& A, const at::Tensor& B) {
  auto D = at::empty({A.size(0), B.size(1)}, A.options().dtype(at::kFloat));
  launch_wmma_gemm(A, B, D);
  return D;
}
}  // namespace

TORCH_LIBRARY(wmma_probe, m) {
  m.def("gemm(Tensor A, Tensor B) -> Tensor");
}
TORCH_LIBRARY_IMPL(wmma_probe, CUDA, m) {
  m.impl("gemm", gemm);
}
