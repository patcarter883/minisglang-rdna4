// Torch bindings for the native flash-attention HIP kernels on gfx1201.
//
// Framework-agnostic ops under attn_hip:: (callable as torch.ops.attn_hip.* from minisgl OR
// vLLM). Mirrors the gdn_hip / zaya_cca TORCH_LIBRARY pattern: opaque custom op + a registered
// fake/meta (in op.py) so torch.compile/Inductor steps over it without graph-breaking.
//
// v0 is bf16 dense prefill (no paging, no fp8-KV). See attn_kernels.hip for scope + provenance.

#include <torch/extension.h>
#include <torch/library.h>
#include <ATen/cuda/CUDAContext.h>

// ---- launcher defined in attn_kernels.hip ----
void launch_flash_prefill(const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&,
                          double, int64_t, int64_t, const float*);

namespace {

at::Tensor flash_prefill(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                         double scale, int64_t causal, int64_t sliding_window,
                         const std::optional<at::Tensor>& mask_bias) {
  TORCH_CHECK(q.dim() == 3 && k.dim() == 3 && v.dim() == 3,
              "attn_hip: q/k/v must be [seq, heads, head_dim]");
  TORCH_CHECK(q.scalar_type() == at::kBFloat16, "attn_hip v0 is bf16-only");
  TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
              "attn_hip v0 expects contiguous q/k/v");
  const float* mask_ptr = nullptr;
  if (mask_bias.has_value()) {
    const auto& mb = *mask_bias;
    const auto seq = q.size(0);
    TORCH_CHECK(mb.dim() == 2 && mb.size(0) == seq && mb.size(1) == seq,
                "attn_hip: mask_bias must be a square [seq, seq] matrix");
    TORCH_CHECK(mb.scalar_type() == at::kFloat && mb.is_contiguous(),
                "attn_hip: mask_bias must be contiguous float32");
    mask_ptr = mb.data_ptr<float>();
  }
  auto out = at::empty_like(q);
  launch_flash_prefill(q, k, v, out, scale, causal, sliding_window, mask_ptr);
  return out;
}

}  // namespace

TORCH_LIBRARY(attn_hip, m) {
  m.def("flash_prefill(Tensor q, Tensor k, Tensor v, float scale, int causal, "
        "int sliding_window, Tensor? mask_bias=None) -> Tensor");
}

TORCH_LIBRARY_IMPL(attn_hip, CUDA, m) {
  m.impl("flash_prefill", flash_prefill);
}
