"""Python entry for the native HIP MLA decode op (gfx1201).

torch.ops.mla_hip.mla_decode: DeepSeek-style multi-head latent attention, ABSORBED decode form.
The W_UK/W_UV absorption happens in the model layer (outside the kernel); the kernel attends over the
shared paged LATENT cache. For GLM-4.7-Flash etc. (kv_lora_rank=512, qk_rope_head_dim=64).
    q:[B, num_heads, kv_lora_rank+qk_rope]  latent_cache:[num_blocks, block_size, kv_lora_rank+qk_rope]
    block_table:[B, max_blocks] int32  context_lens:[B] int32  -> [B, num_heads, kv_lora_rank]
"""
import glob
import os

import torch

_so = glob.glob(os.path.join(os.path.dirname(__file__), "mla_hip_C*.so"))
if not _so:
    raise ImportError("mla_hip_C not built; run `GPU_ARCHS=gfx1201 python setup.py build_ext --inplace`")
torch.ops.load_library(_so[0])


@torch.library.register_fake("mla_hip::mla_decode")
def _fake(q, latent_cache, block_table, context_lens, scale, sliding_window, kv_block_stride=0):
    return q.new_empty((q.shape[0], q.shape[1], q.shape[2] - 64))


@torch.library.register_fake("mla_hip::mla_decode_fp8")
def _fake_fp8(q, latent_cache, block_table, context_lens, scale, k_descale, v_descale,
              sliding_window, kv_block_stride=0):
    return q.new_empty((q.shape[0], q.shape[1], q.shape[2] - 64))


@torch.library.register_fake("mla_hip::mla_prefill")
def _fake_prefill(q, k, v, cu_seqlens_q, cu_seqlens_k, scale, causal, sliding_window, max_seqlen_q):
    # materialized MLA: out has v_head_dim (v.shape[2]), not qk_head_dim
    return q.new_empty((q.shape[0], q.shape[1], v.shape[2]))


@torch.library.register_fake("mla_hip::mla_verify")
def _fake_verify(q, latent_cache, block_table, q_seq_idx, q_kbound, scale, sliding_window,
                 kv_block_stride=0):
    return q.new_empty((q.shape[0], q.shape[1], q.shape[2] - 64))


@torch.library.register_fake("mla_hip::mla_verify_fp8")
def _fake_verify_fp8(q, latent_cache, block_table, q_seq_idx, q_kbound, scale, k_descale, v_descale,
                     sliding_window, kv_block_stride=0):
    return q.new_empty((q.shape[0], q.shape[1], q.shape[2] - 64))


mla_decode = torch.ops.mla_hip.mla_decode
mla_decode_fp8 = torch.ops.mla_hip.mla_decode_fp8
mla_prefill = torch.ops.mla_hip.mla_prefill
mla_verify = torch.ops.mla_hip.mla_verify
mla_verify_fp8 = torch.ops.mla_hip.mla_verify_fp8

__all__ = ["mla_decode", "mla_decode_fp8", "mla_prefill", "mla_verify", "mla_verify_fp8"]
