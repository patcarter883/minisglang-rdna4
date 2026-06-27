// Torch bindings for the native HIP moe_align_block_size op on gfx1201.
// torch.ops.moe_hip.moe_align — drop-in for vLLM moe_align_block_size(topk_ids, block_size,
// num_experts, None, pad_sorted_ids=True), returning (sorted_ids, expert_ids, num_tokens_post_pad).

#include <algorithm>
#include <tuple>
#include <torch/extension.h>
#include <torch/library.h>
#include <ATen/cuda/CUDAContext.h>

void launch_moe_align(const at::Tensor&, int64_t, int64_t, at::Tensor&, at::Tensor&, at::Tensor&);

namespace {

// P = max_num_tokens_padded — computed IDENTICALLY to vLLM (pad_sorted_ids=True) so downstream
// shapes (P-row GEMM intermediates, ident=arange(P)) match the reference exactly.
static long padded_len(long numel, long num_experts, long block_size) {
    long P = numel + num_experts * (block_size - 1);
    P = ((P + block_size - 1) / block_size) * block_size;                 // round_up(_, block_size)
    if (numel < num_experts) P = std::min(numel * block_size, P);         // tiny-numel clamp
    return P;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> moe_align(const at::Tensor& topk_ids,
                                                         int64_t num_experts, int64_t block_size) {
    TORCH_CHECK(topk_ids.scalar_type() == at::kInt, "topk_ids must be int32");
    TORCH_CHECK(topk_ids.is_contiguous(), "topk_ids must be contiguous");
    TORCH_CHECK(num_experts > 0 && num_experts <= 4096, "num_experts in (0, 4096]");
    TORCH_CHECK(block_size > 0, "block_size > 0");
    const long numel = topk_ids.numel();
    const long P = padded_len(numel, num_experts, block_size);
    const long num_blocks = (P + block_size - 1) / block_size;
    auto opt = topk_ids.options();
    auto sorted_ids = at::empty({P}, opt);
    auto expert_ids = at::empty({num_blocks}, opt);
    auto ntp = at::empty({1}, opt);
    launch_moe_align(topk_ids, num_experts, block_size, sorted_ids, expert_ids, ntp);
    return std::make_tuple(sorted_ids, expert_ids, ntp);
}

}  // namespace

TORCH_LIBRARY(moe_hip, m) {
    m.def("moe_align(Tensor topk_ids, int num_experts, int block_size) -> (Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(moe_hip, CUDA, m) {
    m.impl("moe_align", moe_align);
}
