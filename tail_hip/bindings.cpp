// Torch bindings for the native "tail" elementwise HIP kernels on gfx1201.
// Ops under tail_hip:: (torch.ops.tail_hip.*), framework-agnostic (minisgl OR vLLM), opaque to
// torch.compile via the fakes in op.py. Mirrors the gdn_hip / attn_decode TORCH_LIBRARY pattern.

#include <torch/extension.h>
#include <torch/library.h>
#include <ATen/cuda/CUDAContext.h>

void launch_rms_norm(const at::Tensor&, const at::Tensor&, at::Tensor&, double, int64_t);
void launch_rms_norm_add(const at::Tensor&, at::Tensor&, const at::Tensor&, at::Tensor&, double,
                         int64_t);
void launch_silu_mul(const at::Tensor&, at::Tensor&);
void launch_rope(const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&, int64_t,
                 int64_t);

namespace {

at::Tensor rms_norm(const at::Tensor& x, const at::Tensor& w, double eps, int64_t plus_one) {
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && w.scalar_type() == at::kBFloat16, "bf16 only");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  auto out = at::empty_like(x);
  launch_rms_norm(x, w, out, eps, plus_one);
  return out;
}

at::Tensor rms_norm_add(const at::Tensor& x, at::Tensor& residual, const at::Tensor& w, double eps,
                        int64_t plus_one) {
  TORCH_CHECK(x.scalar_type() == at::kBFloat16, "bf16 only");
  TORCH_CHECK(x.is_contiguous() && residual.is_contiguous(), "contiguous only");
  auto out = at::empty_like(x);
  launch_rms_norm_add(x, residual, w, out, eps, plus_one);
  return out;
}

at::Tensor silu_and_mul(const at::Tensor& x) {
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kHalf ||
              x.scalar_type() == at::kFloat, "silu_and_mul: bf16/fp16/fp32 only");
  TORCH_CHECK(x.is_contiguous() && x.size(-1) % 2 == 0, "x must be contiguous, last dim even");
  auto shape = x.sizes().vec();
  shape.back() /= 2;
  auto out = at::empty(shape, x.options());
  launch_silu_mul(x, out);
  return out;
}

at::Tensor rope(const at::Tensor& x, const at::Tensor& pos, const at::Tensor& cache,
                int64_t head_size, int64_t rotary_dim) {
  TORCH_CHECK(x.scalar_type() == at::kBFloat16, "x bf16 only");
  TORCH_CHECK(pos.scalar_type() == at::kInt, "positions must be int32");
  TORCH_CHECK(cache.scalar_type() == at::kFloat, "cos_sin_cache must be fp32");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  auto out = at::empty_like(x);
  launch_rope(x, pos, cache, out, head_size, rotary_dim);
  return out;
}

}  // namespace

TORCH_LIBRARY(tail_hip, m) {
  m.def("rms_norm(Tensor x, Tensor w, float eps, int plus_one) -> Tensor");
  m.def("rms_norm_add(Tensor x, Tensor(a!) residual, Tensor w, float eps, int plus_one) -> Tensor");
  m.def("silu_and_mul(Tensor x) -> Tensor");
  m.def("rope(Tensor x, Tensor pos, Tensor cache, int head_size, int rotary_dim) -> Tensor");
}

TORCH_LIBRARY_IMPL(tail_hip, CUDA, m) {
  m.impl("rms_norm", rms_norm);
  m.impl("rms_norm_add", rms_norm_add);
  m.impl("silu_and_mul", silu_and_mul);
  m.impl("rope", rope);
}
