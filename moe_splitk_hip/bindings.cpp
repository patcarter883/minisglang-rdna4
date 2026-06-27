// Torch bindings for the minisgl-local split-K W4A8 gemm2 SCATTER kernel (Task A #17).
// torch.ops.moe_splitk_hip.moe_gemm_splitk_scatter — writes the pre-zeroed fp32 (M,N) output in
// place (atomic scatter), mirroring w4a8_fp8_wmma.mmq_fp8_moe_gemm_scatter + a split_k grid.z.
// Mirrors the gdn_hip / tail_hip TORCH_LIBRARY pattern.

#include <torch/extension.h>
#include <torch/library.h>
#include <ATen/cuda/CUDAContext.h>

void launch_moe_gemm_splitk_scatter(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&, int64_t, int64_t, int64_t);

namespace {

void moe_gemm_splitk_scatter(
    const at::Tensor& x, const at::Tensor& w_packed, const at::Tensor& scales,
    const c10::optional<at::Tensor>& w_zeros, const at::Tensor& sorted_token_ids,
    const at::Tensor& expert_ids, const at::Tensor& num_tokens_post_padded,
    const at::Tensor& topk_weights, at::Tensor& output,
    int64_t top_k, int64_t block_m, int64_t split_k) {
    TORCH_CHECK(x.scalar_type() == at::kHalf, "x (post-act buf) must be fp16");
    TORCH_CHECK(output.scalar_type() == at::kFloat, "output accumulator must be fp32");
    TORCH_CHECK(x.is_contiguous() && output.is_contiguous(), "x/output must be contiguous");
    const at::Tensor wz = w_zeros.has_value() ? *w_zeros : at::Tensor();
    launch_moe_gemm_splitk_scatter(x, w_packed, scales, wz, sorted_token_ids, expert_ids,
                                   num_tokens_post_padded, topk_weights, output,
                                   top_k, block_m, split_k);
}

}  // namespace

TORCH_LIBRARY(moe_splitk_hip, m) {
  m.def("moe_gemm_splitk_scatter(Tensor x, Tensor w_packed, Tensor scales, Tensor? w_zeros, "
        "Tensor sorted_token_ids, Tensor expert_ids, Tensor num_tokens_post_padded, "
        "Tensor topk_weights, Tensor(a!) output, int top_k, int block_m, int split_k) -> ()");
}

TORCH_LIBRARY_IMPL(moe_splitk_hip, CUDA, m) {
  m.impl("moe_gemm_splitk_scatter", moe_gemm_splitk_scatter);
}
