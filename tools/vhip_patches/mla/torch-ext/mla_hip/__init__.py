"""mla_hip — native HIP MLA (multi-head latent attention) kernels on gfx1201 (RDNA4).

The kernel-builder / `kernels`-hub packaging of the kernel formerly vendored at
minisgl-rdna4/mla_hip. DeepSeek-style multi-head latent attention:

  * mla_decode / mla_decode_fp8 — ABSORBED decode over the shared paged LATENT cache. The W_UK/W_UV
    absorption happens in the model layer (outside the kernel).
        q:[B, num_heads, kv_lora_rank+qk_rope]  latent_cache:[num_blocks, block_size, kv_lora_rank+qk_rope]
        block_table:[B, max_blocks] int32  context_lens:[B] int32  -> [B, num_heads, kv_lora_rank]
  * mla_prefill — materialized varlen MHA with asymmetric qk_head_dim (q/k) and v_head_dim.
  * mla_verify / mla_verify_fp8 — absorbed multi-query decode for speculative verification.

Ops are exposed both as callables here and (for torch.compile) as opaque custom ops with registered
fake/meta impls. Load with `kernels.get_kernel("<repo-id>")` or import the built package directly.
"""
import torch

from ._ops import add_op_namespace_prefix, ops

# ---- fake/meta impls: keep torch.compile/Inductor from graph-breaking on the opaque custom ops ----


@torch.library.register_fake(add_op_namespace_prefix("mla_decode"))
def _mla_decode_fake(q, latent_cache, block_table, context_lens, scale, sliding_window,
                     kv_block_stride=0):
    return q.new_empty((q.shape[0], q.shape[1], q.shape[2] - 64))


@torch.library.register_fake(add_op_namespace_prefix("mla_decode_fp8"))
def _mla_decode_fp8_fake(q, latent_cache, block_table, context_lens, scale, k_descale, v_descale,
                         sliding_window, kv_block_stride=0):
    return q.new_empty((q.shape[0], q.shape[1], q.shape[2] - 64))


@torch.library.register_fake(add_op_namespace_prefix("mla_prefill"))
def _mla_prefill_fake(q, k, v, cu_seqlens_q, cu_seqlens_k, scale, causal, sliding_window,
                      max_seqlen_q):
    # materialized MLA: out has v_head_dim (v.shape[2]), not qk_head_dim
    return q.new_empty((q.shape[0], q.shape[1], v.shape[2]))


@torch.library.register_fake(add_op_namespace_prefix("mla_prefill_lse"))
def _mla_prefill_lse_fake(q, k, v, cu_seqlens_q, cu_seqlens_k, scale, causal, sliding_window,
                          max_seqlen_q):
    # same core as mla_prefill, plus the natural-log LSE in FlashAttention's [num_heads, total_q]
    # layout (what merge_attn_states wants for chunked-context prefill).
    return [
        q.new_empty((q.shape[0], q.shape[1], v.shape[2])),
        q.new_empty((q.shape[1], q.shape[0]), dtype=torch.float32),
    ]


@torch.library.register_fake(add_op_namespace_prefix("mla_verify"))
def _mla_verify_fake(q, latent_cache, block_table, q_seq_idx, q_kbound, scale, sliding_window,
                     kv_block_stride=0):
    return q.new_empty((q.shape[0], q.shape[1], q.shape[2] - 64))


@torch.library.register_fake(add_op_namespace_prefix("mla_verify_fp8"))
def _mla_verify_fp8_fake(q, latent_cache, block_table, q_seq_idx, q_kbound, scale, k_descale,
                         v_descale, sliding_window, kv_block_stride=0):
    return q.new_empty((q.shape[0], q.shape[1], q.shape[2] - 64))


# ---- op wrappers ----
mla_decode = ops.mla_decode
mla_decode_fp8 = ops.mla_decode_fp8
mla_prefill = ops.mla_prefill
mla_prefill_lse = ops.mla_prefill_lse
mla_verify = ops.mla_verify
mla_verify_fp8 = ops.mla_verify_fp8

__all__ = ["mla_decode", "mla_decode_fp8", "mla_prefill", "mla_prefill_lse", "mla_verify",
           "mla_verify_fp8"]
