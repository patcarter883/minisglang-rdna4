// torch bindings for the W8A8-FP8 grouped-MoE MMQ HIP custom op (gfx1201 / RDNA4).
//
// W8A8 = fp8 (e4m3) weights with a PER-OUTPUT-CHANNEL fp32 scale + fp8 activations.
// A strict simplification of the W4A8 package (w4a8_fp8_wmma): identical WMMA core,
// activation quant, scatter/gather/reduce plumbing; the weight operand is a plain
// e4m3 byte tensor (no int4 unpack / zeros / group scale) and the weight scale is
// per-output-channel, applied once in the epilogue.
//
// Exposes (library w8a8_fp8_wmma):
//   mmq_w8a8_moe_gemm(x, w_fp8, scales, sorted_token_ids, expert_ids,
//                     num_tokens_post_padded, top_k, block_m, kernel) -> out
//   mmq_w8a8_moe_gemm1_silu(... same args ...) -> out                 # (P, inter)
//   mmq_w8a8_moe_gemm_scatter(x, w_fp8, scales, ..., topk_weights, output, ...) -> ()
//   mmq_w8a8_moe_gather_reduce(out2, sorted_token_ids, topk_weights,
//                              num_tokens_post_padded, top_k) -> out   # (M, N)
//
//   kernel: an opaque MoeKernel id (kernel_names.h): 0=scalar golden, 6=wmma
//   (prefill/gemm2), 7=gemv (decode gemm1). Descriptive names live above the ABI
//   in __init__.py. NO PYBIND11_MODULE (loaded via torch.ops.load_library).
//
// Tensor contracts:
//   x         : (T, K)        fp16,  CUDA, contiguous   (gemm2: (P, inter))
//   w_fp8     : (E, N, K)     uint8, CUDA, contiguous   (e4m3, row-major)
//   scales    : (E, N)        fp32,  CUDA, contiguous   (per-output-channel)
//   routing   : int32, CUDA;  output (scatter): (M, N) fp32, pre-zeroed

#include <torch/extension.h>
#include <torch/library.h>
#include <ATen/cuda/CUDAContext.h>
#include "kernel_names.h"   // w4a8::MoeKernel + moe_kernel_valid() (descriptive ids; opaque int ABI)

void launch_mmq_w8a8_moe_gemm_gfx1201(
    const at::Tensor& x,
    const at::Tensor& w_fp8,
    const at::Tensor& scales,
    const at::Tensor& sorted_token_ids,
    const at::Tensor& expert_ids,
    const at::Tensor& num_tokens_post_padded,
    at::Tensor& out,
    int64_t top_k,
    int64_t block_m,
    int64_t kernel);

void launch_mmq_w8a8_moe_gemm1_silu_gfx1201(
    const at::Tensor& x,
    const at::Tensor& w_fp8,
    const at::Tensor& scales,
    const at::Tensor& sorted_token_ids,
    const at::Tensor& expert_ids,
    const at::Tensor& num_tokens_post_padded,
    at::Tensor& out,
    int64_t top_k,
    int64_t block_m,
    int64_t kernel);

void launch_mmq_w8a8_moe_gemm_scatter_gfx1201(
    const at::Tensor& x,
    const at::Tensor& w_fp8,
    const at::Tensor& scales,
    const at::Tensor& sorted_token_ids,
    const at::Tensor& expert_ids,
    const at::Tensor& num_tokens_post_padded,
    const at::Tensor& topk_weights,
    at::Tensor& output,
    int64_t top_k,
    int64_t block_m,
    int64_t kernel);

void launch_mmq_w8a8_moe_gather_reduce_gfx1201(
    const at::Tensor& out2,
    const at::Tensor& sorted_token_ids,
    const at::Tensor& topk_weights,
    const at::Tensor& num_tokens_post_padded,
    at::Tensor& out,
    int64_t top_k);

namespace {

// Shared validation for the W8A8 weight operand: (E, N, K) e4m3 + (E, N) f32 scale.
static inline void check_w8a8_weights(
    const at::Tensor& x, const at::Tensor& w_fp8, const at::Tensor& scales) {
    TORCH_CHECK(x.is_cuda() && w_fp8.is_cuda() && scales.is_cuda(),
                "x, w_fp8, scales must be CUDA");
    TORCH_CHECK(x.scalar_type() == at::kHalf, "x must be fp16");
    TORCH_CHECK(w_fp8.scalar_type() == at::kByte, "w_fp8 must be uint8 (e4m3)");
    TORCH_CHECK(scales.scalar_type() == at::kFloat, "scales must be fp32");
    TORCH_CHECK(x.dim() == 2, "x must be 2D (T, K)");
    TORCH_CHECK(w_fp8.dim() == 3 && scales.dim() == 2,
                "w_fp8 (E,N,K) must be 3D and scales (E,N) must be 2D");
    TORCH_CHECK(x.is_contiguous() && w_fp8.is_contiguous() && scales.is_contiguous(),
                "x, w_fp8, scales must be contiguous");
    TORCH_CHECK(w_fp8.size(2) == x.size(1),
                "w_fp8 last dim must be K=", x.size(1), "; got ", w_fp8.size(2));
    TORCH_CHECK(scales.size(0) == w_fp8.size(0) && scales.size(1) == w_fp8.size(1),
                "scales must be (E, N) matching w_fp8");
}

at::Tensor mmq_w8a8_moe_gemm_forward(
    const at::Tensor& x,
    const at::Tensor& w_fp8,
    const at::Tensor& scales,
    const at::Tensor& sorted_token_ids,
    const at::Tensor& expert_ids,
    const at::Tensor& num_tokens_post_padded,
    int64_t top_k,
    int64_t block_m,
    int64_t kernel) {

    check_w8a8_weights(x, w_fp8, scales);
    TORCH_CHECK(sorted_token_ids.scalar_type() == at::kInt &&
                expert_ids.scalar_type() == at::kInt &&
                num_tokens_post_padded.scalar_type() == at::kInt,
                "routing tensors must be int32");

    const int64_t N = w_fp8.size(1);
    const int64_t P = sorted_token_ids.size(0);
    TORCH_CHECK(P % block_m == 0, "P=", P, " not divisible by block_m=", block_m);
    TORCH_CHECK(expert_ids.size(0) == P / block_m,
                "expert_ids must have P/block_m=", P / block_m, " entries");
    TORCH_CHECK(w4a8::moe_kernel_valid(kernel),
                "moe kernel id must be 0 (scalar) / 6 (wmma) / 7 (gemv); got ", kernel);

    auto out = at::empty({P, N}, x.options());
    launch_mmq_w8a8_moe_gemm_gfx1201(
        x, w_fp8, scales, sorted_token_ids, expert_ids,
        num_tokens_post_padded, out, top_k, block_m, kernel);
    return out;
}

// Fused gemm1 + silu_and_mul. Runs the gated gemm1 (w13, N=2*inter = [gate|up])
// and the silu_and_mul activation in ONE kernel, returning the post-activation
// (P, inter) directly. Bit-exact to mmq_fp8_moe_gemm(w13) + _C.silu_and_mul.
at::Tensor mmq_w8a8_moe_gemm1_silu_forward(
    const at::Tensor& x,
    const at::Tensor& w_fp8,
    const at::Tensor& scales,
    const at::Tensor& sorted_token_ids,
    const at::Tensor& expert_ids,
    const at::Tensor& num_tokens_post_padded,
    int64_t top_k,
    int64_t block_m,
    int64_t kernel) {

    check_w8a8_weights(x, w_fp8, scales);
    TORCH_CHECK(sorted_token_ids.scalar_type() == at::kInt &&
                expert_ids.scalar_type() == at::kInt &&
                num_tokens_post_padded.scalar_type() == at::kInt,
                "routing tensors must be int32");
    TORCH_CHECK(static_cast<w4a8::MoeKernel>(kernel) == w4a8::MoeKernel::Wmma,
                "fused gemm1+silu needs the 'wmma' kernel; got id ", kernel);

    const int64_t N = w_fp8.size(1);             // 2*inter
    const int64_t P = sorted_token_ids.size(0);
    TORCH_CHECK(N % 2 == 0, "w13 N must be 2*inter (even); got ", N);
    const int64_t inter = N / 2;
    TORCH_CHECK(P % block_m == 0, "P=", P, " not divisible by block_m=", block_m);
    TORCH_CHECK(expert_ids.size(0) == P / block_m,
                "expert_ids must have P/block_m=", P / block_m, " entries");

    auto out = at::empty({P, inter}, x.options());
    launch_mmq_w8a8_moe_gemm1_silu_gfx1201(
        x, w_fp8, scales, sorted_token_ids, expert_ids,
        num_tokens_post_padded, out, top_k, block_m, kernel);
    return out;
}

// Fused gemm2 + topk-weight + indirect atomic scatter. `x` is the (P, inter)
// post-activation buffer; the result is accumulated IN PLACE into the caller's
// pre-zeroed (M, N) fp32 `output`. `sorted_token_ids` are the gemm1 (token,slot) ids.
void mmq_w8a8_moe_gemm_scatter_forward(
    const at::Tensor& x,
    const at::Tensor& w_fp8,
    const at::Tensor& scales,
    const at::Tensor& sorted_token_ids,
    const at::Tensor& expert_ids,
    const at::Tensor& num_tokens_post_padded,
    const at::Tensor& topk_weights,
    at::Tensor& output,
    int64_t top_k,
    int64_t block_m,
    int64_t kernel) {

    check_w8a8_weights(x, w_fp8, scales);
    TORCH_CHECK(sorted_token_ids.scalar_type() == at::kInt &&
                expert_ids.scalar_type() == at::kInt &&
                num_tokens_post_padded.scalar_type() == at::kInt,
                "routing tensors must be int32");
    TORCH_CHECK(topk_weights.scalar_type() == at::kFloat &&
                output.scalar_type() == at::kFloat,
                "topk_weights and output must be fp32");
    TORCH_CHECK(topk_weights.is_contiguous() && output.is_contiguous(),
                "topk_weights and output must be contiguous");
    TORCH_CHECK(output.dim() == 2 && output.size(1) == w_fp8.size(1),
                "output must be (M, N) with N=", w_fp8.size(1));

    const int64_t P = sorted_token_ids.size(0);
    const int64_t M = output.size(0);
    TORCH_CHECK(P % block_m == 0, "P=", P, " not divisible by block_m=", block_m);
    TORCH_CHECK(expert_ids.size(0) == P / block_m,
                "expert_ids must have P/block_m=", P / block_m, " entries");
    TORCH_CHECK(topk_weights.numel() == M * top_k,
                "topk_weights must have M*top_k=", M * top_k, " entries");
    TORCH_CHECK(w4a8::moe_kernel_valid(kernel),
                "moe kernel id must be 0 (scalar) / 6 (wmma) / 7 (gemv); got ", kernel);

    launch_mmq_w8a8_moe_gemm_scatter_gfx1201(
        x, w_fp8, scales, sorted_token_ids, expert_ids,
        num_tokens_post_padded, topk_weights, output, top_k, block_m, kernel);
}

// Contention-free MoE reduce: gemm2 NON-scatter (P,N) -> weighted gather-reduce
// to (M,N) fp32. Verbatim from w4a8 (no weights/scales involved).
at::Tensor mmq_w8a8_moe_gather_reduce_forward(
    const at::Tensor& out2,                    // (P, N) fp16
    const at::Tensor& sorted_token_ids,        // (P,) int32
    const at::Tensor& topk_weights,            // (M*top_k,) fp32
    const at::Tensor& num_tokens_post_padded,  // (1,) int32
    int64_t top_k) {

    TORCH_CHECK(out2.is_cuda() && out2.scalar_type() == at::kHalf && out2.dim() == 2,
                "out2 must be (P,N) fp16 CUDA");
    TORCH_CHECK(sorted_token_ids.scalar_type() == at::kInt &&
                num_tokens_post_padded.scalar_type() == at::kInt,
                "sorted_token_ids / num_tokens_post_padded must be int32");
    TORCH_CHECK(topk_weights.scalar_type() == at::kFloat, "topk_weights must be fp32");
    TORCH_CHECK(out2.is_contiguous() && topk_weights.is_contiguous(),
                "out2 and topk_weights must be contiguous");
    TORCH_CHECK(topk_weights.numel() % top_k == 0, "topk_weights.numel() must be M*top_k");

    const int64_t N = out2.size(1);
    const int64_t M = topk_weights.numel() / top_k;
    auto out = at::zeros({M, N}, out2.options().dtype(at::kFloat));
    launch_mmq_w8a8_moe_gather_reduce_gfx1201(out2, sorted_token_ids, topk_weights,
                                              num_tokens_post_padded, out, top_k);
    return out;
}

}  // namespace

// pt2_compliant_tag: marks every op as safe for torch.compile / Dynamo full-graph
// capture (vLLM's aot_compile is fullgraph-strict and otherwise raises "unsupported
// operator: w8a8_fp8_wmma.*" the moment the kernel engages). The matching fake/meta
// kernels (output-shape inference) live in op.py.
TORCH_LIBRARY(w8a8_fp8_wmma, m) {
    m.def("mmq_w8a8_moe_gemm(Tensor x, Tensor w_fp8, Tensor scales, "
          "Tensor sorted_token_ids, Tensor expert_ids, Tensor num_tokens_post_padded, "
          "int top_k, int block_m, int kernel) -> Tensor",
          {at::Tag::pt2_compliant_tag});
    m.def("mmq_w8a8_moe_gemm1_silu(Tensor x, Tensor w_fp8, Tensor scales, "
          "Tensor sorted_token_ids, Tensor expert_ids, Tensor num_tokens_post_padded, "
          "int top_k, int block_m, int kernel) -> Tensor",
          {at::Tag::pt2_compliant_tag});
    m.def("mmq_w8a8_moe_gemm_scatter(Tensor x, Tensor w_fp8, Tensor scales, "
          "Tensor sorted_token_ids, Tensor expert_ids, "
          "Tensor num_tokens_post_padded, Tensor topk_weights, Tensor(a!) output, "
          "int top_k, int block_m, int kernel) -> ()",
          {at::Tag::pt2_compliant_tag});
    m.def("mmq_w8a8_moe_gather_reduce(Tensor out2, Tensor sorted_token_ids, "
          "Tensor topk_weights, Tensor num_tokens_post_padded, int top_k) -> Tensor",
          {at::Tag::pt2_compliant_tag});
}

TORCH_LIBRARY_IMPL(w8a8_fp8_wmma, CUDA, m) {
    m.impl("mmq_w8a8_moe_gemm", &mmq_w8a8_moe_gemm_forward);
    m.impl("mmq_w8a8_moe_gemm1_silu", &mmq_w8a8_moe_gemm1_silu_forward);
    m.impl("mmq_w8a8_moe_gemm_scatter", &mmq_w8a8_moe_gemm_scatter_forward);
    m.impl("mmq_w8a8_moe_gather_reduce", &mmq_w8a8_moe_gather_reduce_forward);
}
