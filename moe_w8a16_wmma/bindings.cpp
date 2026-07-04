// bindings.cpp — torch.ops.moe_w8a16.* for the fp8-weight × bf16-act grouped MoE WMMA GEMM (gfx1201).
#include <ATen/ATen.h>
#include <torch/library.h>

namespace moe_w8a16 {

void launch_moe_w8a16_gemm(const at::Tensor& A, const at::Tensor& w_fp8, const at::Tensor& w_scales,
                           const at::Tensor& sorted_token_ids, const at::Tensor& expert_ids,
                           const at::Tensor& num_tokens_post_padded,
                           const c10::optional<at::Tensor>& topk_weights,
                           at::Tensor& out, at::Tensor& out_scatter,
                           int64_t OUT, int64_t IN, int64_t top_k, int64_t block_m,
                           int64_t num_valid_tokens, int64_t BN, bool scatter,
                           int64_t mul_weight, int64_t out_top_k);

// gemm1-style: C[P,OUT] bf16 = A[src,IN] @ dequant(w_fp8[e,OUT,IN])^T, sorted-padded rows.
at::Tensor moe_w8a16_gemm(const at::Tensor& A, const at::Tensor& w_fp8, const at::Tensor& w_scales,
                          const at::Tensor& sorted_token_ids, const at::Tensor& expert_ids,
                          const at::Tensor& num_tokens_post_padded,
                          const c10::optional<at::Tensor>& topk_weights,
                          int64_t top_k, int64_t block_m, int64_t num_valid_tokens,
                          int64_t BN, int64_t mul_weight) {
  const int64_t OUT = w_fp8.size(1), IN = w_fp8.size(2);
  const int64_t P = sorted_token_ids.size(0);
  auto out = at::empty({P, OUT}, A.options());
  auto dummy = at::empty({0}, A.options().dtype(at::kFloat));
  launch_moe_w8a16_gemm(A, w_fp8, w_scales, sorted_token_ids, expert_ids, num_tokens_post_padded,
                        topk_weights, out, dummy, OUT, IN, top_k, block_m, num_valid_tokens, BN,
                        /*scatter=*/false, mul_weight, 0);
  return out;
}

// gemm2-style: C[M,OUT] fp32 = sum over top_k rows of (topk_w * A[row_pad,IN] @ dequant(w_fp8)^T).
at::Tensor moe_w8a16_gemm_scatter(const at::Tensor& A, const at::Tensor& w_fp8,
                                  const at::Tensor& w_scales,
                                  const at::Tensor& sorted_token_ids, const at::Tensor& expert_ids,
                                  const at::Tensor& num_tokens_post_padded,
                                  const at::Tensor& topk_weights, int64_t M, int64_t top_k,
                                  int64_t block_m, int64_t num_valid_tokens, int64_t BN,
                                  int64_t out_top_k) {
  const int64_t OUT = w_fp8.size(1), IN = w_fp8.size(2);
  auto out = at::zeros({M, OUT}, A.options().dtype(at::kFloat));
  auto dummy = at::empty({0}, A.options());
  c10::optional<at::Tensor> tw = topk_weights;
  launch_moe_w8a16_gemm(A, w_fp8, w_scales, sorted_token_ids, expert_ids, num_tokens_post_padded, tw,
                        dummy, out, OUT, IN, top_k, block_m, num_valid_tokens, BN,
                        /*scatter=*/true, 0, out_top_k);
  return out;
}

TORCH_LIBRARY(moe_w8a16, m) {
  m.def("moe_w8a16_gemm(Tensor A, Tensor w_fp8, Tensor w_scales, Tensor sorted_token_ids, "
        "Tensor expert_ids, Tensor num_tokens_post_padded, Tensor? topk_weights, int top_k, "
        "int block_m, int num_valid_tokens, int BN, int mul_weight) -> Tensor");
  m.def("moe_w8a16_gemm_scatter(Tensor A, Tensor w_fp8, Tensor w_scales, Tensor sorted_token_ids, "
        "Tensor expert_ids, Tensor num_tokens_post_padded, Tensor topk_weights, int M, int top_k, "
        "int block_m, int num_valid_tokens, int BN, int out_top_k) -> Tensor");
}
TORCH_LIBRARY_IMPL(moe_w8a16, CUDA, m) {
  m.impl("moe_w8a16_gemm", moe_w8a16_gemm);
  m.impl("moe_w8a16_gemm_scatter", moe_w8a16_gemm_scatter);
}

}  // namespace moe_w8a16
