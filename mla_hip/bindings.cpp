// Torch bindings for the native HIP MLA (multi-head latent attention) DECODE kernel on gfx1201.
// torch.ops.mla_hip.mla_decode — framework-agnostic (vLLM mla backend / minisgl), opaque to
// torch.compile via the fake in op.py. Mirrors the attn_decode TORCH_LIBRARY pattern.

#include <torch/extension.h>
#include <torch/library.h>
#include <ATen/cuda/CUDAContext.h>

void launch_mla_decode(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                       at::Tensor&, double, int64_t, int64_t);
void launch_mla_decode_fp8(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                           at::Tensor&, double, double, double, int64_t, int64_t);
void launch_mla_prefill(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                        const at::Tensor&, at::Tensor&, double, int64_t, int64_t, int64_t);
void launch_mla_verify(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                       const at::Tensor&, at::Tensor&, double, int64_t, int64_t);
void launch_mla_verify_fp8(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                           const at::Tensor&, at::Tensor&, double, double, double, int64_t, int64_t);

namespace {

at::Tensor mla_decode(const at::Tensor& q, const at::Tensor& latent_cache,
                      const at::Tensor& block_table, const at::Tensor& context_lens, double scale,
                      int64_t sliding_window, int64_t kv_block_stride) {
  TORCH_CHECK(q.dim() == 3, "q must be [B, num_heads, kv_lora_rank + qk_rope_head_dim]");
  TORCH_CHECK(latent_cache.dim() == 3, "latent_cache must be [num_blocks, block_size, kv_lora_rank + qk_rope]");
  TORCH_CHECK(q.scalar_type() == at::kBFloat16 && latent_cache.scalar_type() == at::kBFloat16, "bf16-only v0");
  TORCH_CHECK(block_table.scalar_type() == at::kInt && context_lens.scalar_type() == at::kInt,
              "block_table/context_lens must be int32");
  TORCH_CHECK(q.is_contiguous(), "q must be contiguous");  // latent_cache may be strided (kv_block_stride)
  TORCH_CHECK(q.size(2) == latent_cache.size(2), "q and latent_cache last dim (LATENT+ROPE) must match");
  const int64_t qk = q.size(2);
  const int64_t latent = qk - 64;                          // v0: qk_rope_head_dim = 64
  auto out = at::empty({q.size(0), q.size(1), latent}, q.options());
  launch_mla_decode(q, latent_cache, block_table, context_lens, out, scale, sliding_window, kv_block_stride);
  return out;
}

// fp8 latent-cache decode: latent_cache is OCP e4m3 (float8_e4m3fn), one byte/element; q stays bf16.
// k_descale folds into the score, v_descale into the output (per-tensor; both == cache_descale for a
// single-scale latent cache).
at::Tensor mla_decode_fp8(const at::Tensor& q, const at::Tensor& latent_cache,
                          const at::Tensor& block_table, const at::Tensor& context_lens, double scale,
                          double k_descale, double v_descale, int64_t sliding_window,
                          int64_t kv_block_stride) {
  TORCH_CHECK(q.dim() == 3, "q must be [B, num_heads, kv_lora_rank + qk_rope_head_dim]");
  TORCH_CHECK(latent_cache.dim() == 3, "latent_cache must be [num_blocks, block_size, kv_lora_rank + qk_rope]");
  TORCH_CHECK(q.scalar_type() == at::kBFloat16, "q must be bf16");
  TORCH_CHECK(latent_cache.scalar_type() == at::kFloat8_e4m3fn, "latent_cache must be float8_e4m3fn");
  TORCH_CHECK(block_table.scalar_type() == at::kInt && context_lens.scalar_type() == at::kInt,
              "block_table/context_lens must be int32");
  TORCH_CHECK(q.is_contiguous(), "q must be contiguous");  // latent_cache may be strided (kv_block_stride)
  TORCH_CHECK(q.size(2) == latent_cache.size(2), "q and latent_cache last dim (LATENT+ROPE) must match");
  const int64_t qk = q.size(2);
  const int64_t latent = qk - 64;                          // v0: qk_rope_head_dim = 64
  auto out = at::empty({q.size(0), q.size(1), latent}, q.options());
  launch_mla_decode_fp8(q, latent_cache, block_table, context_lens, out, scale, k_descale, v_descale,
                        sliding_window, kv_block_stride);
  return out;
}

// MLA PREFILL (materialized form): dense varlen MHA with asymmetric qk_head_dim (q/k) and v_head_dim.
// q:[total_q,H,qk_head_dim] k:[total_kv,H,qk_head_dim] v:[total_kv,H,v_head_dim], packed varlen via
// cu_seqlens_q/cu_seqlens_k. Causal carries a prefix offset (prefix_len = k_len - q_len) so cold
// prefill (q_len==k_len) and chunked-extend (prefix>0) both work. -> out:[total_q,H,v_head_dim].
at::Tensor mla_prefill(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                       const at::Tensor& cu_seqlens_q, const at::Tensor& cu_seqlens_k, double scale,
                       int64_t causal, int64_t sliding_window, int64_t max_seqlen_q) {
  TORCH_CHECK(q.dim() == 3 && k.dim() == 3 && v.dim() == 3, "q/k/v must be [total_tokens, num_heads, dim]");
  TORCH_CHECK(q.scalar_type() == at::kBFloat16 && k.scalar_type() == at::kBFloat16 &&
              v.scalar_type() == at::kBFloat16, "bf16-only v0");
  TORCH_CHECK(cu_seqlens_q.scalar_type() == at::kInt && cu_seqlens_k.scalar_type() == at::kInt,
              "cu_seqlens_q/cu_seqlens_k must be int32");
  TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(), "q/k/v must be contiguous");
  TORCH_CHECK(q.size(1) == k.size(1) && k.size(1) == v.size(1), "num_heads must match across q/k/v");
  TORCH_CHECK(q.size(2) == k.size(2), "q and k qk_head_dim must match");
  TORCH_CHECK(k.size(0) == v.size(0), "k and v must share total_kv_tokens");
  TORCH_CHECK(cu_seqlens_q.size(0) == cu_seqlens_k.size(0), "cu_seqlens_q/k must have S+1 entries");
  auto out = at::empty({q.size(0), q.size(1), v.size(2)}, q.options());
  launch_mla_prefill(q, k, v, cu_seqlens_q, cu_seqlens_k, out, scale, causal, sliding_window,
                     max_seqlen_q);
  return out;
}

// MLA multi-query VERIFY (speculative decoding): absorbed multi-query attention over the paged
// latent. q packs total_q query tokens (confirmed + drafts across sequences); q_seq_idx maps each
// query row to its sequence (block_table row) and q_kbound is its per-query causal context length
// (cached_len + within-seq-query-index + 1). -> out:[total_q, num_heads, kv_lora_rank].
at::Tensor mla_verify(const at::Tensor& q, const at::Tensor& latent_cache,
                      const at::Tensor& block_table, const at::Tensor& q_seq_idx,
                      const at::Tensor& q_kbound, double scale, int64_t sliding_window,
                      int64_t kv_block_stride) {
  TORCH_CHECK(q.dim() == 3, "q must be [total_q, num_heads, kv_lora_rank + qk_rope_head_dim]");
  TORCH_CHECK(latent_cache.dim() == 3, "latent_cache must be [num_blocks, block_size, kv_lora_rank + qk_rope]");
  TORCH_CHECK(q.scalar_type() == at::kBFloat16 && latent_cache.scalar_type() == at::kBFloat16, "bf16-only v0");
  TORCH_CHECK(block_table.scalar_type() == at::kInt && q_seq_idx.scalar_type() == at::kInt &&
              q_kbound.scalar_type() == at::kInt, "block_table/q_seq_idx/q_kbound must be int32");
  TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
  TORCH_CHECK(q.size(2) == latent_cache.size(2), "q and latent_cache last dim (LATENT+ROPE) must match");
  TORCH_CHECK(q_seq_idx.size(0) == q.size(0) && q_kbound.size(0) == q.size(0),
              "q_seq_idx/q_kbound must have total_q entries");
  const int64_t qk = q.size(2);
  const int64_t latent = qk - 64;                          // v0: qk_rope_head_dim = 64
  auto out = at::empty({q.size(0), q.size(1), latent}, q.options());
  launch_mla_verify(q, latent_cache, block_table, q_seq_idx, q_kbound, out, scale, sliding_window,
                    kv_block_stride);
  return out;
}

at::Tensor mla_verify_fp8(const at::Tensor& q, const at::Tensor& latent_cache,
                          const at::Tensor& block_table, const at::Tensor& q_seq_idx,
                          const at::Tensor& q_kbound, double scale, double k_descale,
                          double v_descale, int64_t sliding_window, int64_t kv_block_stride) {
  TORCH_CHECK(q.dim() == 3, "q must be [total_q, num_heads, kv_lora_rank + qk_rope_head_dim]");
  TORCH_CHECK(latent_cache.dim() == 3, "latent_cache must be [num_blocks, block_size, kv_lora_rank + qk_rope]");
  TORCH_CHECK(q.scalar_type() == at::kBFloat16, "q must be bf16");
  TORCH_CHECK(latent_cache.scalar_type() == at::kFloat8_e4m3fn, "latent_cache must be float8_e4m3fn");
  TORCH_CHECK(block_table.scalar_type() == at::kInt && q_seq_idx.scalar_type() == at::kInt &&
              q_kbound.scalar_type() == at::kInt, "block_table/q_seq_idx/q_kbound must be int32");
  TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
  TORCH_CHECK(q.size(2) == latent_cache.size(2), "q and latent_cache last dim (LATENT+ROPE) must match");
  TORCH_CHECK(q_seq_idx.size(0) == q.size(0) && q_kbound.size(0) == q.size(0),
              "q_seq_idx/q_kbound must have total_q entries");
  const int64_t qk = q.size(2);
  const int64_t latent = qk - 64;                          // v0: qk_rope_head_dim = 64
  auto out = at::empty({q.size(0), q.size(1), latent}, q.options());
  launch_mla_verify_fp8(q, latent_cache, block_table, q_seq_idx, q_kbound, out, scale, k_descale,
                        v_descale, sliding_window, kv_block_stride);
  return out;
}

}  // namespace

TORCH_LIBRARY(mla_hip, m) {
  m.def("mla_decode(Tensor q, Tensor latent_cache, Tensor block_table, Tensor context_lens, "
        "float scale, int sliding_window, int kv_block_stride=0) -> Tensor");
  m.def("mla_decode_fp8(Tensor q, Tensor latent_cache, Tensor block_table, Tensor context_lens, "
        "float scale, float k_descale, float v_descale, int sliding_window, int kv_block_stride=0) "
        "-> Tensor");
  m.def("mla_prefill(Tensor q, Tensor k, Tensor v, Tensor cu_seqlens_q, Tensor cu_seqlens_k, "
        "float scale, int causal, int sliding_window, int max_seqlen_q) -> Tensor");
  m.def("mla_verify(Tensor q, Tensor latent_cache, Tensor block_table, Tensor q_seq_idx, "
        "Tensor q_kbound, float scale, int sliding_window, int kv_block_stride=0) -> Tensor");
  m.def("mla_verify_fp8(Tensor q, Tensor latent_cache, Tensor block_table, Tensor q_seq_idx, "
        "Tensor q_kbound, float scale, float k_descale, float v_descale, int sliding_window, "
        "int kv_block_stride=0) -> Tensor");
}

TORCH_LIBRARY_IMPL(mla_hip, CUDA, m) {
  m.impl("mla_decode", mla_decode);
  m.impl("mla_decode_fp8", mla_decode_fp8);
  m.impl("mla_prefill", mla_prefill);
  m.impl("mla_verify", mla_verify);
  m.impl("mla_verify_fp8", mla_verify_fp8);
}
