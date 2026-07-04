// bindings.cpp — torch.ops.moe_bf16.* for the unquantized (bf16/fp16) grouped MoE WMMA GEMM (gfx1201).
// Opaque custom ops (TORCH_LIBRARY), same pattern as gdn_hip / mla_hip.
//
// Two op families:
//   * ALLOCATING  (moe_bf16_gemm / moe_bf16_gemm_scatter): allocate their output, return it. Handy for
//     the parity harness / eager use, but NOT CUDA-graph-capturable (the at::empty/at::zeros is a
//     malloc during capture).
//   * *_out (moe_bf16_gemm_out / moe_bf16_gemm_scatter_out): CALLER passes the pre-allocated output;
//     the op only writes/atomic-adds into it — NO allocation, NO host-sync. These are the graph-safe
//     entry points wired into fused_experts_impl. For the scatter, the caller must `.zero_()` the fp32
//     accumulator before the call (a capturable op on a static buffer).
#include <ATen/ATen.h>
#include <torch/library.h>

namespace moe_bf16 {

void launch_moe_bf16_gemm(const at::Tensor& A, const at::Tensor& w,
                          const at::Tensor& sorted_token_ids, const at::Tensor& expert_ids,
                          const at::Tensor& num_tokens_post_padded,
                          const c10::optional<at::Tensor>& topk_weights,
                          at::Tensor& out, at::Tensor& out_scatter,
                          int64_t OUT, int64_t IN, int64_t top_k, int64_t block_m,
                          int64_t num_valid_tokens, int64_t BN, bool scatter,
                          int64_t mul_weight, int64_t out_top_k);

// ---- allocating variants (NOT graph-capturable) --------------------------------------------------

// gemm1-style: C[P,OUT] = A[src,IN] @ w[e,OUT,IN]^T, sorted-padded rows. src = offs/top_k.
at::Tensor moe_bf16_gemm(const at::Tensor& A, const at::Tensor& w,
                         const at::Tensor& sorted_token_ids, const at::Tensor& expert_ids,
                         const at::Tensor& num_tokens_post_padded,
                         const c10::optional<at::Tensor>& topk_weights,
                         int64_t top_k, int64_t block_m, int64_t num_valid_tokens,
                         int64_t BN, int64_t mul_weight) {
  const int64_t OUT = w.size(1), IN = w.size(2);
  const int64_t P = sorted_token_ids.size(0);
  auto out = at::empty({P, OUT}, A.options());
  launch_moe_bf16_gemm(A, w, sorted_token_ids, expert_ids, num_tokens_post_padded, topk_weights,
                       out, out, OUT, IN, top_k, block_m, num_valid_tokens, BN,
                       /*scatter=*/false, mul_weight, 0);
  return out;
}

// gemm2-style: C[M,OUT] fp32 = sum over the top_k rows of (topk_w * A[row_pad,IN] @ w[e,OUT,IN]^T).
at::Tensor moe_bf16_gemm_scatter(const at::Tensor& A, const at::Tensor& w,
                                 const at::Tensor& sorted_token_ids, const at::Tensor& expert_ids,
                                 const at::Tensor& num_tokens_post_padded,
                                 const at::Tensor& topk_weights, int64_t M, int64_t top_k,
                                 int64_t block_m, int64_t num_valid_tokens, int64_t BN,
                                 int64_t out_top_k) {
  const int64_t OUT = w.size(1), IN = w.size(2);
  auto out = at::zeros({M, OUT}, A.options().dtype(at::kFloat));
  c10::optional<at::Tensor> tw = topk_weights;
  launch_moe_bf16_gemm(A, w, sorted_token_ids, expert_ids, num_tokens_post_padded, tw,
                       out, out, OUT, IN, top_k, block_m, num_valid_tokens, BN,
                       /*scatter=*/true, 0, out_top_k);
  return out;
}

// ---- *_out variants (graph-capturable; caller owns the output) -----------------------------------

// out: pre-allocated [P, OUT] in A.dtype. The op overwrites every valid sorted-padded row.
void moe_bf16_gemm_out(const at::Tensor& A, const at::Tensor& w,
                       const at::Tensor& sorted_token_ids, const at::Tensor& expert_ids,
                       const at::Tensor& num_tokens_post_padded,
                       const c10::optional<at::Tensor>& topk_weights, at::Tensor& out,
                       int64_t top_k, int64_t block_m, int64_t num_valid_tokens,
                       int64_t BN, int64_t mul_weight) {
  const int64_t OUT = w.size(1), IN = w.size(2);
  // `out` is passed for BOTH slots; the scatter=false path only ever touches the first (bf16/fp16).
  launch_moe_bf16_gemm(A, w, sorted_token_ids, expert_ids, num_tokens_post_padded, topk_weights,
                       const_cast<at::Tensor&>(out), const_cast<at::Tensor&>(out), OUT, IN, top_k,
                       block_m, num_valid_tokens, BN, /*scatter=*/false, mul_weight, 0);
}

// out: pre-allocated [M, OUT] fp32, ZEROED BY THE CALLER (the op only atomic-adds into it).
void moe_bf16_gemm_scatter_out(const at::Tensor& A, const at::Tensor& w,
                               const at::Tensor& sorted_token_ids, const at::Tensor& expert_ids,
                               const at::Tensor& num_tokens_post_padded,
                               const at::Tensor& topk_weights, at::Tensor& out,
                               int64_t top_k, int64_t block_m, int64_t num_valid_tokens,
                               int64_t BN, int64_t out_top_k) {
  const int64_t OUT = w.size(1), IN = w.size(2);
  c10::optional<at::Tensor> tw = topk_weights;
  // `out` (fp32) is passed for BOTH slots; the scatter=true path only touches the second (fp32).
  launch_moe_bf16_gemm(A, w, sorted_token_ids, expert_ids, num_tokens_post_padded, tw,
                       const_cast<at::Tensor&>(out), const_cast<at::Tensor&>(out), OUT, IN, top_k,
                       block_m, num_valid_tokens, BN, /*scatter=*/true, 0, out_top_k);
}

TORCH_LIBRARY(moe_bf16, m) {
  m.def("moe_bf16_gemm(Tensor A, Tensor w, Tensor sorted_token_ids, Tensor expert_ids, "
        "Tensor num_tokens_post_padded, Tensor? topk_weights, int top_k, int block_m, "
        "int num_valid_tokens, int BN, int mul_weight) -> Tensor");
  m.def("moe_bf16_gemm_scatter(Tensor A, Tensor w, Tensor sorted_token_ids, Tensor expert_ids, "
        "Tensor num_tokens_post_padded, Tensor topk_weights, int M, int top_k, int block_m, "
        "int num_valid_tokens, int BN, int out_top_k) -> Tensor");
  m.def("moe_bf16_gemm_out(Tensor A, Tensor w, Tensor sorted_token_ids, Tensor expert_ids, "
        "Tensor num_tokens_post_padded, Tensor? topk_weights, Tensor(a!) out, int top_k, "
        "int block_m, int num_valid_tokens, int BN, int mul_weight) -> ()");
  m.def("moe_bf16_gemm_scatter_out(Tensor A, Tensor w, Tensor sorted_token_ids, Tensor expert_ids, "
        "Tensor num_tokens_post_padded, Tensor topk_weights, Tensor(a!) out, int top_k, "
        "int block_m, int num_valid_tokens, int BN, int out_top_k) -> ()");
}
TORCH_LIBRARY_IMPL(moe_bf16, CUDA, m) {
  m.impl("moe_bf16_gemm", moe_bf16_gemm);
  m.impl("moe_bf16_gemm_scatter", moe_bf16_gemm_scatter);
  m.impl("moe_bf16_gemm_out", moe_bf16_gemm_out);
  m.impl("moe_bf16_gemm_scatter_out", moe_bf16_gemm_scatter_out);
}

}  // namespace moe_bf16
