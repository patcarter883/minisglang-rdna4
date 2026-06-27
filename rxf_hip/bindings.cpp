// Torch bindings for the native RXF W4A8 HIP kernels on gfx1201.
// Ops under rxf_hip:: (torch.ops.rxf_hip.*), framework-agnostic, opaque to torch.compile
// via the fakes in op.py. Mirrors the gdn_hip / attn_hip / tail_hip binding pattern.

#include <torch/extension.h>
#include <torch/library.h>
#include <ATen/cuda/CUDAContext.h>

// Launchers from rxf_kernels.hip
void launch_rotate_quant(const at::Tensor&, at::Tensor&, at::Tensor&, int64_t);
void launch_linear(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                   const at::Tensor&, const at::Tensor&, at::Tensor&, bool);
void launch_moe_gemm(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                     const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                     at::Tensor&, int64_t, int64_t, int64_t);

namespace {

std::tuple<at::Tensor, at::Tensor> rotate_quant_int8(const at::Tensor& x, int64_t span) {
    TORCH_CHECK(x.is_contiguous(), "rxf_hip::rotate_quant_int8: x must be contiguous");
    auto q = at::empty(x.sizes(), x.options().dtype(at::kChar));
    auto shape = x.sizes().vec();
    shape.pop_back();
    auto scale = at::empty(shape, x.options().dtype(at::kFloat));
    auto x2d = x.reshape({-1, x.size(-1)});
    auto q2d = q.reshape({-1, x.size(-1)});
    auto scale1d = scale.reshape({-1});
    launch_rotate_quant(x2d, q2d, scale1d, span);
    return {q, scale};
}

at::Tensor linear(const at::Tensor& q, const at::Tensor& a_scale, const at::Tensor& w_packed,
                  const at::Tensor& w_scale, const at::Tensor& nl,
                  const c10::optional<at::Tensor>& bias) {
    TORCH_CHECK(q.scalar_type() == at::kChar, "rxf_hip::linear: q must be int8");
    TORCH_CHECK(w_packed.scalar_type() == at::kByte, "rxf_hip::linear: w_packed must be uint8");
    TORCH_CHECK(w_scale.scalar_type() == at::kHalf, "rxf_hip::linear: w_scale must be fp16");
    TORCH_CHECK(nl.scalar_type() == at::kChar && nl.numel() == 16, "rxf_hip::linear: nl int8[16]");
    TORCH_CHECK(q.is_contiguous() && w_packed.is_contiguous(), "rxf_hip::linear: contiguous only");
    const int64_t M = q.size(0), N = w_packed.size(0);
    auto out = at::empty({M, N}, q.options().dtype(at::kBFloat16));
    const bool has_bias = bias.has_value();
    auto bias_t = has_bias ? bias.value().to(at::kFloat).contiguous() : at::empty({0}, q.options().dtype(at::kFloat));
    launch_linear(q, a_scale, w_packed, w_scale, nl, bias_t, out, has_bias);
    return out;
}

at::Tensor moe_gemm(const at::Tensor& q, const at::Tensor& a_scale, const at::Tensor& w_packed,
                    const at::Tensor& w_scale, const at::Tensor& nl, const at::Tensor& sorted_ids,
                    const at::Tensor& expert_ids, const at::Tensor& num_tokens_post_padded,
                    int64_t top_k, int64_t block_m, int64_t num_valid_tokens) {
    TORCH_CHECK(q.scalar_type() == at::kChar, "rxf_hip::moe_gemm: q must be int8");
    TORCH_CHECK(w_packed.dim() == 3, "rxf_hip::moe_gemm: w_packed is [E, N, K/2]");
    const int64_t P = sorted_ids.size(0), N = w_packed.size(1);
    auto out = at::empty({P, N}, q.options().dtype(at::kBFloat16));
    launch_moe_gemm(q, a_scale, w_packed, w_scale, nl, sorted_ids, expert_ids,
                    num_tokens_post_padded, out, top_k, block_m, num_valid_tokens);
    return out;
}

}  // namespace

TORCH_LIBRARY(rxf_hip, m) {
    m.def("rotate_quant_int8(Tensor x, int span) -> (Tensor, Tensor)");
    m.def("linear(Tensor q, Tensor a_scale, Tensor w_packed, Tensor w_scale, Tensor nl, Tensor? bias) -> Tensor");
    m.def("moe_gemm(Tensor q, Tensor a_scale, Tensor w_packed, Tensor w_scale, Tensor nl, "
          "Tensor sorted_ids, Tensor expert_ids, Tensor num_tokens_post_padded, "
          "int top_k, int block_m, int num_valid_tokens) -> Tensor");
}

TORCH_LIBRARY_IMPL(rxf_hip, CUDA, m) {
    m.impl("rotate_quant_int8", rotate_quant_int8);
    m.impl("linear", linear);
    m.impl("moe_gemm", moe_gemm);
}
