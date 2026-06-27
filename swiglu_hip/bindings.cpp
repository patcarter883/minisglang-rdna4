// Torch bindings for the native fused SwiGLU HIP kernel on gfx1201 (Task B #18).
// torch.ops.swiglu_hip.fused_swiglu, framework-agnostic, opaque to torch.compile via op.py fakes.
// Mirrors the tail_hip / gdn_hip / attn_decode TORCH_LIBRARY pattern.

#include <torch/extension.h>
#include <torch/library.h>
#include <ATen/cuda/CUDAContext.h>

void launch_fused_swiglu(const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&,
                         at::Tensor&);

namespace {

// out = down( silu(x @ Wg^T) * (x @ Wu^T) )
//   x:[M,K]  w_gate_up:[2*inter,K]  w_down:[K,inter]  ->  out:[M,K]
at::Tensor fused_swiglu(const at::Tensor& x, const at::Tensor& w_gate_up,
                        const at::Tensor& w_down) {
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kHalf,
              "fused_swiglu: bf16/fp16 only");
  TORCH_CHECK(x.scalar_type() == w_gate_up.scalar_type() &&
              x.scalar_type() == w_down.scalar_type(), "x/w dtypes must match");
  TORCH_CHECK(x.is_contiguous() && w_gate_up.is_contiguous() && w_down.is_contiguous(),
              "x/w must be contiguous");
  const int64_t K = x.size(-1);
  const int64_t inter = w_down.size(-1);
  TORCH_CHECK(w_down.size(0) == K, "w_down rows must equal hidden K");
  TORCH_CHECK(w_gate_up.size(1) == K, "w_gate_up cols must equal hidden K");
  TORCH_CHECK(w_gate_up.size(0) == 2 * inter, "w_gate_up rows must equal 2*inter");
  const int64_t M = x.numel() / K;
  auto h = at::empty({M, inter}, x.options());
  auto out = at::empty({M, K}, x.options());
  launch_fused_swiglu(x, w_gate_up, w_down, h, out);
  return out.reshape(x.sizes());
}

}  // namespace

TORCH_LIBRARY(swiglu_hip, m) {
  m.def("fused_swiglu(Tensor x, Tensor w_gate_up, Tensor w_down) -> Tensor");
}

TORCH_LIBRARY_IMPL(swiglu_hip, CUDA, m) {
  m.impl("fused_swiglu", fused_swiglu);
}
