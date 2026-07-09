from __future__ import annotations

import glob
import re
from typing import Dict, Iterator, Tuple

import safetensors
import torch
from minisgl.distributed import get_dp_info, get_tp_info, is_ep_enabled
from minisgl.utils import cached_load_hf_config, div_ceil, download_hf_weight
from tqdm import tqdm

_SPLIT_DIM_0 = [".q_proj", ".k_proj", ".v_proj", ".gate_proj", ".up_proj", ".q_b_proj", ".kv_b_proj"]
_SPLIT_DIM_1 = [".o_proj", ".down_proj"]
# AWQ/GPTQ packed siblings store the OUTPUT features on axis 1 (qweight [K, N//8], scales/qzeros
# [G, N(//8)]), the opposite of a bf16 weight [N, K]. So a column-parallel (output-sharded) AWQ
# tensor shards axis 1, and a row-parallel (input-sharded) one shards axis 0 — both flipped vs bf16.
_AWQ_SUFFIXES = (".qweight", ".qzeros", ".scales")

# Merge groups: individual projections -> fused projection
_MERGE_GROUPS = {
    ".q_proj": (".qkv_proj", ("q", "k", "v")),
    ".k_proj": (".qkv_proj", ("q", "k", "v")),
    ".v_proj": (".qkv_proj", ("q", "k", "v")),
    ".gate_proj": (".gate_up_proj", ("gate", "up")),
    ".up_proj": (".gate_up_proj", ("gate", "up")),
}
_SLOT_NAMES = {
    ".q_proj": "q",
    ".k_proj": "k",
    ".v_proj": "v",
    ".gate_proj": "gate",
    ".up_proj": "up",
}
_EXPERT_PATTERN = re.compile(r"^(?P<prefix>.+\.experts)\.(?P<idx>\d+)\.(?P<name>.+)$")
_LAYER_IDX_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def _is_beyond_decoder(name: str, num_layers: int) -> bool:
    """True for a `...layers.<n>....` tensor with n >= num_layers — i.e. an appended MTP /
    next-token-prediction head (GLM-4.x / DeepSeek). We serve the decoder only, no speculative
    decode, so those are skipped. Safe for every model: no standard checkpoint has real decoder
    layers past num_hidden_layers."""
    m = _LAYER_IDX_PATTERN.search(name)
    return m is not None and int(m.group(1)) >= num_layers


def _remap_glm_mtp(name: str, num_layers: int) -> str | None:
    """Remap a GLM-4.x MTP checkpoint key `(model.)layers.<num_layers>.X` -> the model-native
    `mtp.X` so GLMMTPHead receives it. The MTP layer is structurally a GLMDecoderLayer (self_attn
    MLA + mlp MoE) PLUS embed_tokens / enorm / hnorm / eh_proj / shared_head.{norm,head}; every
    sub-key maps 1:1 under the `mtp.` root, and the standard merge/shard/expert-stack pipeline then
    handles the MLA q/kv splits, the AWQ routed-expert stacking, and (TP>1) head/vocab sharding."""
    body = name.removeprefix("language_model.").removeprefix("model.")
    prefix = f"layers.{num_layers}."
    assert body.startswith(prefix), f"unexpected MTP key layout: {name!r}"
    return "mtp." + body[len(prefix) :]


def _shard_tensor(key: str, value: torch.Tensor, r: int, n: int, num_kv_heads: int):
    """Extract rank r's shard from a single tensor. Returns a contiguous copy. (No-op at n==1.)
    AWQ/GPTQ packed tensors flip the shard axis vs a bf16 weight (see _AWQ_SUFFIXES)."""
    # GLM's always-on shared expert is REPLICATED, not TP-sharded (its down_proj K=moe_intermediate
    # must stay a multiple of 512 for the W4A8 dense kernel — see GLMSharedExpert). Keep it whole.
    if ".shared_experts." in key:
        return value
    is_awq = key.endswith(_AWQ_SUFFIXES)
    if any(key.count(sub) for sub in _SPLIT_DIM_0):  # column-parallel (output-sharded)
        if not is_awq:
            is_kv_proj = any(key.count(sub) for sub in (".k_proj", ".v_proj"))
            if is_kv_proj and num_kv_heads is not None and num_kv_heads < n:
                head_dim = value.shape[0] // num_kv_heads
                head_idx = r * num_kv_heads // n
                return value[head_idx * head_dim : (head_idx + 1) * head_dim].clone()
        return value.chunk(n, dim=1 if is_awq else 0)[r].clone()
    elif any(key.count(sub) for sub in _SPLIT_DIM_1):  # row-parallel (input-sharded)
        return value.chunk(n, dim=0 if is_awq else 1)[r].clone()
    elif key.count("lm_head") or key.count("embed_tokens") or key.count("shared_head.head"):
        num_embeddings = value.shape[0]
        num_embeddings_per_partition = div_ceil(num_embeddings, n)
        vocab_start_idx = r * num_embeddings_per_partition
        vocab_end_idx = min((r + 1) * num_embeddings_per_partition, num_embeddings)
        return value[vocab_start_idx:vocab_end_idx, :].clone()
    else:
        return value


def _get_merge_info(key: str):
    """If key belongs to a merge group, return (merged_key, slot, all_slots). Else None."""
    for suffix, (fused_suffix, slots) in _MERGE_GROUPS.items():
        if key.count(suffix):
            return key.replace(suffix, fused_suffix), _SLOT_NAMES[suffix], slots
    return None


def _get_expert_stack_info(key: str) -> tuple[str, int] | None:
    """Map an expert-scoped checkpoint key to the packed runtime key."""
    match = _EXPERT_PATTERN.match(key)
    if match is None:
        return None

    packed_name = match.group("name")
    if packed_name.endswith(".weight"):
        packed_name = packed_name.removesuffix(".weight")
    return f"{match.group('prefix')}.{packed_name}", int(match.group("idx"))


# ---- Qwen3.5 GDN-hybrid weight-name remap (Phase 3d-3) ----
# The checkpoint is a multimodal wrapper: `model.language_model.*` (the text decoder we serve),
# `model.visual.*` (vision tower) and `mtp.*` (the multi-token-prediction speculative head). We
# serve text only with no MTP, so vision + mtp are skipped. The GDN `linear_attn` ships the
# fused projections SPLIT — concat them into what `QwenGatedDeltaNet` wants: in_proj_qkv+in_proj_z
# -> in_proj_qkvz, in_proj_b+in_proj_a -> in_proj_ba; conv1d.weight -> conv1d_weight (the module
# stores it as a flat Parameter, not an nn.Conv1d). Dense MLP gate/up -> gate_up (as elsewhere).
# Full-attn q/k/v stay SEPARATE: q_proj carries the per-head output gate (emits 2x), so it cannot
# be fused into a single qkv_proj the way the dense Qwen3 path does.
# mtp.* is handled separately (loaded for the mtp proposer, else skipped) before this check.
_QWEN35_SKIP_PREFIXES = ("model.visual.", "visual.")
_QWEN35_LM_PREFIX = "model.language_model."
# checkpoint suffix -> renamed native suffix (rename only, no concat)
_QWEN35_RENAME = {".linear_attn.conv1d.weight": ".linear_attn.conv1d_weight"}
# checkpoint suffix -> (merged native suffix, ordered source suffixes, cat_dim). All are
# nn.Linear weights (out, in), so the fused tensor concatenates along the output dim (0).
_QKVZ = (".linear_attn.in_proj_qkv.weight", ".linear_attn.in_proj_z.weight")
_BA = (".linear_attn.in_proj_b.weight", ".linear_attn.in_proj_a.weight")
_GATE_UP = (".mlp.gate_proj.weight", ".mlp.up_proj.weight")
_QWEN35_CONCAT = {
    _QKVZ[0]: (".linear_attn.in_proj_qkvz.weight", _QKVZ, 0),
    _QKVZ[1]: (".linear_attn.in_proj_qkvz.weight", _QKVZ, 0),
    _BA[0]: (".linear_attn.in_proj_ba.weight", _BA, 0),
    _BA[1]: (".linear_attn.in_proj_ba.weight", _BA, 0),
    _GATE_UP[0]: (".mlp.gate_up_proj.weight", _GATE_UP, 0),
    _GATE_UP[1]: (".mlp.gate_up_proj.weight", _GATE_UP, 0),
}
# QUANTIZED GDN in_proj (compressed-tensors 27B): in_proj_qkv + in_proj_z are quantized, so their
# weight_packed / weight_scale / weight_zero_point concat into in_proj_qkvz.<field> along the OUTPUT
# dim (0) — output channels are independent under group-wise W4A16, and the zero_point is int4-packed
# 8-per-int32 ALONG N (each part's N is a multiple of 8: 10240 & 6144), so the packed-row concat is
# exact. Same ordered [qkv, z] members as the bf16 _QKVZ; in_proj_a/b stay bf16 (.weight, above).
for _fld in ("weight_packed", "weight_scale", "weight_zero_point"):
    _qz = (f".linear_attn.in_proj_qkv.{_fld}", f".linear_attn.in_proj_z.{_fld}")
    _merged = f".linear_attn.in_proj_qkvz.{_fld}"
    _QWEN35_CONCAT[_qz[0]] = (_merged, _qz, 0)
    _QWEN35_CONCAT[_qz[1]] = (_merged, _qz, 0)
# QUANTIZED GDN in_proj_ba (MXFP4 35B): unlike AWQ/CT-int4 (where the tiny per-v-head b/a gates stay
# bf16, in the ignore list and concatenated via `.weight` above), the MXFP4 checkpoint ALSO quantizes
# in_proj_b + in_proj_a — so their weight_packed / weight_scale concat into in_proj_ba.<field> along
# the OUTPUT dim (0), same [b, a] order as the bf16 _BA members. (No weight_zero_point: MXFP4 is
# symmetric; that key simply never appears, so its map entry is inert.)
for _fld in ("weight_packed", "weight_scale", "weight_zero_point"):
    _ba = (f".linear_attn.in_proj_b.{_fld}", f".linear_attn.in_proj_a.{_fld}")
    _merged = f".linear_attn.in_proj_ba.{_fld}"
    _QWEN35_CONCAT[_ba[0]] = (_merged, _ba, 0)
    _QWEN35_CONCAT[_ba[1]] = (_merged, _ba, 0)


def _gate_up_merge(key: str):
    """gate_proj/up_proj -> gate_up_proj for the qwen3_5_moe shared + routed experts. Unlike the
    dense q/k/v path, qwen3_5 keeps full-attn q/k/v SEPARATE (q_proj carries the output gate), so
    this handles ONLY gate/up. Returns (merged_key, slot) or None."""
    for sub, slot in ((".gate_proj.", "gate"), (".up_proj.", "up")):
        if sub in key:
            return key.replace(sub, ".gate_up_proj."), slot
    return None


# Qwen3.5 MTP (mtp.* namespace) -> model-native `mtp.*` remap (single full-attention layer +
# fc fuser + pre-norms; reuses the target embed + tied lm_head, so no dedicated embed/head key).
# `mtp.layers.0.X` collapses to `mtp.X`; the gate/up of the dense MLP still merges to gate_up.
_QWEN35_MTP_LAYER0 = "mtp.layers.0."


def _qwen3_5_mtp_remap(ckpt_key: str):
    """Map an `mtp.*` checkpoint key to the model-native `mtp.*` key plan (or None to skip)."""
    if ckpt_key.startswith(_QWEN35_MTP_LAYER0):
        native = "mtp." + ckpt_key[len(_QWEN35_MTP_LAYER0) :]
    else:
        native = ckpt_key  # mtp.fc / mtp.norm / mtp.pre_fc_norm_* — already native
    # dense MLP gate/up -> gate_up (same concat as the backbone dense path).
    for suffix, slot in ((".mlp.gate_proj.weight", "gate"), (".mlp.up_proj.weight", "up")):
        if native.endswith(suffix):
            merged = native[: -len(suffix)] + ".mlp.gate_up_proj.weight"
            members = (".mlp.gate_proj.weight", ".mlp.up_proj.weight")
            return ("concat", merged, members.index(suffix), 2, 0)
    return ("direct", native)


def qwen3_5_remap(ckpt_key: str, load_mtp: bool = False):
    """Map a Qwen3.5 GDN-hybrid checkpoint key to a minisgl-native key plan. Pure (no tensors),
    so it is CPU-testable against the checkpoint header vs. the model's `state_dict()`.

    Returns one of:
      ``None``                                           -> skip this checkpoint tensor
      ``("direct", native_key)``                         -> rename only
      ``("concat", merged_key, slot, n_slots, cat_dim)`` -> one member of an ordered concat group
    """
    if ckpt_key.endswith(".weight_shape"):
        return None  # compressed-tensors metadata (the original [N,K]); not a model param
    if ckpt_key.startswith("mtp."):
        return _qwen3_5_mtp_remap(ckpt_key) if load_mtp else None
    if ckpt_key.startswith(_QWEN35_SKIP_PREFIXES):
        return None
    if ckpt_key == "lm_head.weight":
        return ("direct", "lm_head.weight")  # untied (qwen3_5_moe); top-level, no LM prefix
    if not ckpt_key.startswith(_QWEN35_LM_PREFIX):
        raise ValueError(f"unexpected Qwen3.5 checkpoint key (not under {_QWEN35_LM_PREFIX!r}): {ckpt_key}")
    native = "model." + ckpt_key[len(_QWEN35_LM_PREFIX) :]
    for suffix, renamed in _QWEN35_RENAME.items():
        if native.endswith(suffix):
            return ("direct", native[: -len(suffix)] + renamed)
    for suffix, (merged_suffix, members, cat_dim) in _QWEN35_CONCAT.items():
        if native.endswith(suffix):
            return ("concat", native[: -len(suffix)] + merged_suffix, members.index(suffix), len(members), cat_dim)
    return ("direct", native)


def _shard_blocks_dim0(t: torch.Tensor, sizes: list[int], r: int, n: int) -> torch.Tensor:
    """Head-aligned column shard: split `t` along dim 0 into the given sub-blocks (q/k/v, or
    gate/up), take rank r's even chunk of EACH, and re-concat. A naive `t.chunk(n, 0)[r]` would
    corrupt a concatenated projection (it would hand rank 0 all of q and rank 1 all of v); this
    keeps every sub-block's heads partitioned consistently. Each block size must divide n."""
    parts = torch.split(t, sizes, dim=0)
    return torch.cat([p.chunk(n, dim=0)[r].contiguous() for p in parts], dim=0)


def _shard_qwen3_5(name: str, t: torch.Tensor, r: int, n: int, config) -> torch.Tensor:
    """Extract rank r's TP shard of a Qwen3.5 GDN-hybrid CHECKPOINT tensor (Phase 4-1).

    Keyed on the checkpoint suffix and applied at READ time — BEFORE the loader's GDN in_proj
    concat / MoE gate-up merge / per-expert stack — so those compose pre-sharded parts into the
    rank-local fused buffers the TP-aware model declares. Mirrors the dense `_shard_tensor` rules
    (q/k/v + gate/up split output dim 0; o/down split input dim 1; embed/lm_head vocab-parallel)
    and adds the GDN head-parallel splits (qkvz/ba/conv1d are concats -> head-aligned per sub-block;
    A_log/dt_bias per v-head; out_proj row-parallel) and the AWQ routed-expert splits (gate/up on
    the packed N=dim1, down on K=dim0). Everything not matched (norms, q/k/gdn norm, router gate,
    shared_expert_gate) is replicated. n==1 is the identity."""
    if n == 1:
        return t
    key_dim = config.linear_key_head_dim * config.linear_num_key_heads
    value_dim = config.linear_value_head_dim * config.linear_num_value_heads

    # ---- GDN linear-attention (head-parallel) ----
    if name.endswith((".linear_attn.in_proj_qkv.weight", ".linear_attn.conv1d.weight")):
        # qkv = [q(key_dim) | k(key_dim) | v(value_dim)]; conv1d (conv_dim,1,kernel) same order.
        return _shard_blocks_dim0(t, [key_dim, key_dim, value_dim], r, n)
    if name.endswith(
        (".linear_attn.in_proj_z.weight", ".linear_attn.in_proj_b.weight",
         ".linear_attn.in_proj_a.weight", ".linear_attn.A_log", ".linear_attn.dt_bias")
    ):
        return t.chunk(n, dim=0)[r].clone()  # z (value_dim) / b,a,A_log,dt_bias (per v-head)
    if name.endswith(".linear_attn.out_proj.weight"):
        return t.chunk(n, dim=1)[r].clone()  # row-parallel (input value_dim/n) + all-reduce

    # QUANTIZED GDN in_proj (compressed-tensors 27B): the same head-parallel splits as the bf16
    # rules, applied per checkpoint field. in_proj_qkv is the [q|k|v] head-block col split (output N,
    # dim 0); in_proj_z is a plain col split (z, value_dim); out_proj is row-parallel (input, dim 1).
    # weight_zero_point is int4-packed 8-per-int32 ALONG N, so its qkv head-blocks are in PACKED units
    # (key_dim//pf, value_dim//pf) — each block is a multiple of pf and divisible by n, keeping whole
    # heads on a rank. Applied at READ, BEFORE the qkvz concat above, so the concat composes the
    # pre-sharded parts. (n==1 already returned; nothing runs at TP=1.)
    if config.quant is not None:
        pf = 32 // config.quant.bits
        if name.endswith(
            (".linear_attn.in_proj_qkv.weight_packed", ".linear_attn.in_proj_qkv.weight_scale")
        ):
            return _shard_blocks_dim0(t, [key_dim, key_dim, value_dim], r, n)
        if name.endswith(".linear_attn.in_proj_qkv.weight_zero_point"):
            return _shard_blocks_dim0(t, [key_dim // pf, key_dim // pf, value_dim // pf], r, n)
        if name.endswith(
            (".linear_attn.in_proj_z.weight_packed", ".linear_attn.in_proj_z.weight_scale",
             ".linear_attn.in_proj_z.weight_zero_point")
        ):
            return t.chunk(n, dim=0)[r].clone()  # z: col-parallel (output value_dim)
        if name.endswith(
            (".linear_attn.in_proj_b.weight_packed", ".linear_attn.in_proj_b.weight_scale",
             ".linear_attn.in_proj_b.weight_zero_point",
             ".linear_attn.in_proj_a.weight_packed", ".linear_attn.in_proj_a.weight_scale",
             ".linear_attn.in_proj_a.weight_zero_point")
        ):
            return t.chunk(n, dim=0)[r].clone()  # b/a: col-parallel (output per v-head), MXFP4 35B
        if name.endswith(
            (".linear_attn.out_proj.weight_packed", ".linear_attn.out_proj.weight_scale",
             ".linear_attn.out_proj.weight_zero_point")
        ):
            return t.chunk(n, dim=1)[r].clone()  # out_proj: row-parallel (input value_dim/group)
    # .linear_attn.norm.weight (head_v_dim) is per-head -> replicate (falls through)

    # ---- full attention (head-parallel; same rules as the dense path) ----
    if name.endswith(
        (".self_attn.q_proj.weight", ".self_attn.k_proj.weight", ".self_attn.v_proj.weight")
    ):
        return t.chunk(n, dim=0)[r].clone()  # q carries q+gate per head; heads are contiguous
    if name.endswith(".self_attn.o_proj.weight"):
        return t.chunk(n, dim=1)[r].clone()
    # q_norm/k_norm (head_dim) -> replicate

    # ---- routed experts (AWQ): gate/up split packed N (dim 1); down split K (dim 0) ----
    if ".mlp.experts." in name:
        if name.endswith(
            (".gate_proj.qweight", ".gate_proj.scales", ".gate_proj.qzeros",
             ".up_proj.qweight", ".up_proj.scales", ".up_proj.qzeros")
        ):
            return t.chunk(n, dim=1)[r].clone()
        if name.endswith(
            (".down_proj.qweight", ".down_proj.scales", ".down_proj.qzeros")
        ):
            return t.chunk(n, dim=0)[r].clone()
        # compressed-tensors W4A16: weight_packed [N, K//pf] / weight_scale [N, K//g] are N-major
        # (output on dim 0). gate/up split the output N (dim 0); down splits the input K (dim 1).
        if name.endswith(
            (".gate_proj.weight_packed", ".gate_proj.weight_scale",
             ".up_proj.weight_packed", ".up_proj.weight_scale")
        ):
            return t.chunk(n, dim=0)[r].clone()
        if name.endswith((".down_proj.weight_packed", ".down_proj.weight_scale")):
            return t.chunk(n, dim=1)[r].clone()

    # ---- dense MLP (4B) + shared expert (35B): col gate/up (dim 0), row down (dim 1) ----
    if name.endswith((".gate_proj.weight", ".up_proj.weight")):
        return t.chunk(n, dim=0)[r].clone()
    if name.endswith(".down_proj.weight"):
        return t.chunk(n, dim=1)[r].clone()

    # ---- FULLY-dense-quantized (compressed-tensors) linears: the dense 27B quantizes q/k/v/o AND
    # the MLP, so their weight_packed [N, K//pf] / weight_scale [N, K//g] (N-major, output on dim 0)
    # need the SAME TP splits as the bf16 .weight rules above — column-parallel q/k/v + gate/up split
    # the output N (dim 0); row-parallel o + down split the input K (packed/group dim 1). Without
    # this the quantized dense weights fall through to REPLICATE and every rank loads the whole model
    # (~13.5 GB int4) -> OOM at load. (`.mlp.experts.*` is handled + returned above, so these
    # unqualified suffixes only match the dense linears.) q_proj carries q+gate per head; output-dim
    # chunk keeps whole heads on a rank, as for the bf16 path. ----
    # weight_zero_point (asymmetric CT, [N//pf, G]) follows the SAME axis: for a column-parallel
    # linear it is packed along the OUTPUT N, so it splits dim 0 with weight_packed/scale; for a
    # row-parallel linear the output N (packed dim 0) is replicated and the input group dim 1 splits.
    if name.endswith(
        (".q_proj.weight_packed", ".q_proj.weight_scale", ".q_proj.weight_zero_point",
         ".k_proj.weight_packed", ".k_proj.weight_scale", ".k_proj.weight_zero_point",
         ".v_proj.weight_packed", ".v_proj.weight_scale", ".v_proj.weight_zero_point",
         ".gate_proj.weight_packed", ".gate_proj.weight_scale", ".gate_proj.weight_zero_point",
         ".up_proj.weight_packed", ".up_proj.weight_scale", ".up_proj.weight_zero_point")
    ):
        return t.chunk(n, dim=0)[r].clone()
    if name.endswith(
        (".o_proj.weight_packed", ".o_proj.weight_scale", ".o_proj.weight_zero_point",
         ".down_proj.weight_packed", ".down_proj.weight_scale", ".down_proj.weight_zero_point")
    ):
        return t.chunk(n, dim=1)[r].clone()

    # ---- vocab-parallel embedding + untied lm_head ----
    if name.endswith("embed_tokens.weight") or name == "lm_head.weight":
        num = t.shape[0]
        per = div_ceil(num, n)
        return t[r * per : min((r + 1) * per, num)].clone()

    # norms, router gate (.mlp.gate.weight), shared_expert_gate -> replicated
    return t


def _ep_expert_shard(config) -> Tuple[bool, int, int]:
    """EP expert-shard params for the QUANTIZED MoE loaders, mirroring MoELayer's gate exactly (moe.py:
    `is_ep_enabled() and (fp8 or W4A8/W4A16 quant)`). Returns (should_shard, ep_local, ep_offset): when
    sharding, each replica keeps only experts [offset : offset+local) and stacks them at local ids
    0..local-1; off => (False, num_experts, 0) i.e. the full replicated stack (unchanged). Only the
    EP-eligible quant formats shard — W4A8 (GPTQ/AWQ) and W4A16 (compressed-tensors), which share the
    w4a8_moe op layout; RXF and unquantized experts are replicated (must match the MoELayer decision or
    the loaded stack won't fit the buffer). fp8/ZAYA has its own sharded loader (_load_zaya_weight)."""
    q = getattr(config, "quant", None)
    ep_quant = q is not None and (q.is_gptq or q.is_awq or q.is_compressed_tensors)
    if is_ep_enabled() and ep_quant:
        dp_info = get_dp_info()
        assert config.num_experts % dp_info.dp_size == 0, (
            f"EP needs num_experts ({config.num_experts}) divisible by dp_size ({dp_info.dp_size})"
        )
        ep_local = config.num_experts // dp_info.dp_size
        return True, ep_local, dp_info.dp_rank * ep_local
    return False, config.num_experts, 0


def _load_qwen3_5_weight(
    model_folder: str, device: torch.device, config
) -> Iterator[Tuple[str, torch.Tensor]]:
    """Streaming loader for the Qwen3.5 GDN-hybrid checkpoint (TP=1; dense 4B or MoE 35B).
    Applies `qwen3_5_remap` (LM-prefix strip, GDN in_proj concat, conv1d rename, vision/MTP skip,
    untied lm_head), then for MoE checkpoints merges gate/up -> gate_up and stacks the per-expert
    tensors over E (reusing the generic `_get_expert_stack_info`). dtype coercion
    (A_log/dt_bias -> fp32) is the engine's job, exactly as for the dense path."""
    tp_info = get_tp_info()
    files = glob.glob(f"{model_folder}/*.safetensors")
    files = [f for f in files if not f.endswith("consolidated.safetensors")] or files
    concat_buf: Dict[str, Dict[int, torch.Tensor]] = {}  # GDN/dense in_proj concat (qwen3_5_remap)
    merge_buf: Dict[str, Dict[str, torch.Tensor]] = {}   # MoE gate/up -> gate_up
    expert_buf: Dict[str, Dict[int, torch.Tensor]] = {}  # MoE per-expert -> stacked over E
    # EP: this replica keeps only its expert shard; skip the rest and stack at local ids. Off => full.
    _ep_shard, _ep_local, _ep_offset = _ep_expert_shard(config)

    def emit(native_key: str, tensor: torch.Tensor) -> Iterator[Tuple[str, torch.Tensor]]:
        # MoE gate/up merge (shared expert: dense .weight -> dim 0; routed experts: AWQ
        # .qweight/.qzeros/.scales -> dim 1). q/k/v are NOT merged (qwen3_5 keeps them separate).
        if (mm := _gate_up_merge(native_key)) is not None:
            merged_key, slot = mm
            merge_buf.setdefault(merged_key, {})[slot] = tensor
            if len(merge_buf[merged_key]) != 2:
                return
            parts = [merge_buf[merged_key][s] for s in ("gate", "up")]
            del merge_buf[merged_key]
            cat_dim = 1 if merged_key.endswith((".qweight", ".qzeros", ".scales")) else 0
            native_key, tensor = merged_key, torch.cat(parts, dim=cat_dim)
        # MoE expert stacking (experts.<e>.<name> -> experts.<name>, stacked over E). Under EP, keep
        # only the local expert shard and re-index to 0..ep_local-1 so the stacked buffer matches the
        # MoELayer's local_num_experts sizing.
        if config.is_moe and (einfo := _get_expert_stack_info(native_key)) is not None:
            packed_key, idx = einfo
            if _ep_shard and not (_ep_offset <= idx < _ep_offset + _ep_local):
                return  # not this replica's expert
            local_idx = idx - _ep_offset
            slots = expert_buf.setdefault(packed_key, {})
            slots[local_idx] = tensor
            if len(slots) != _ep_local:
                return
            experts = [slots[i] for i in range(_ep_local)]
            del expert_buf[packed_key]
            yield packed_key, torch.stack(experts, dim=0)
        else:
            yield native_key, tensor

    for file in tqdm(files, desc="Loading weights", disable=not tp_info.is_primary()):
        with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
            for name in f.keys():
                plan = qwen3_5_remap(name, load_mtp=config.mtp_num_hidden_layers > 0)
                if plan is None:
                    continue
                # Shard at READ (on the checkpoint name), so the GDN concat / gate-up merge /
                # expert stack below all compose rank-local parts (Phase 4-1; no-op at TP=1).
                raw = _shard_qwen3_5(name, f.get_tensor(name), tp_info.rank, tp_info.size, config)
                if plan[0] == "direct":
                    yield from emit(plan[1], raw)
                    continue
                _, merged, slot, n_slots, cat_dim = plan
                concat_buf.setdefault(merged, {})[slot] = raw
                if len(concat_buf[merged]) != n_slots:
                    continue
                parts = [concat_buf[merged][i] for i in range(n_slots)]
                del concat_buf[merged]
                yield from emit(merged, torch.cat(parts, dim=cat_dim))
    assert not concat_buf, f"incomplete concat groups in checkpoint: {list(concat_buf.keys())}"
    assert not merge_buf, f"incomplete gate/up merges in checkpoint: {list(merge_buf.keys())}"
    assert not expert_buf, f"incomplete expert stacks in checkpoint: {list(expert_buf.keys())}"


# ---- ZAYA1-8B CCA-hybrid weight-name remap (Step 3) ----
# The checkpoint is single-prefix `model.*` (no LM/vision wrapper). Three remap families:
#  1. Plain rename: top-level `model.res_scale.*` -> `model.res_scale_final.*` (the final merge);
#     CCA `self_attn.qkv.*` dotted submodule names (`conv_qk.0.weight`, `linear_q.weight`, ...) ->
#     the CCAConv nn.Module's FLAT param names (`conv_qk_0_weight`, `linear_q`, ...); router
#     `zaya_block.router.<sub>.weight|bias` -> the ZayaRouter nn.Module's flat names
#     (`down_proj.weight` -> `down_proj_weight`, `router_mlp.0.bias` -> `router_mlp_0_bias`, ...).
#  2. fp8 experts: `zaya_block.experts.local_experts.{e}.linear_fc1.{weight,weight_scale}` STAY fp8
#     -- stack the raw F8_E4M3 weight + F32 per-channel scale over e ->
#     `zaya_block.experts.gate_up_proj.{weight,weight_scale}` (linear_fc1 is ALREADY the merged
#     [gate|up] w13 [4096,2048]; gate is the first half, the silu_and_mul convention, so NO
#     re-split). `linear_fc2` -> `experts.down_proj.{weight,weight_scale}` (w2). Dequant is deferred
#     to compute (_GroupedFP8Experts); dequant-to-bf16 at load is ~16 GB and OOMs the 16 GB card.
#  3. Direct: embed_tokens, final_norm, input_norm, layer-level res_scale, o_proj.
# Router + CCA + res_scale are bf16/fp32 (quant ignore list); the experts keep their checkpoint fp8
# dtype (~8 GB). TP=1 (no shard) for v0.

# CCAConv submodule (`self_attn.qkv.*`) checkpoint suffix -> CCAConv flat param name.
_ZAYA_CCA_RENAME = {
    "linear_q.weight": "linear_q",
    "linear_k.weight": "linear_k",
    "val_proj1.weight": "val_proj1",
    "val_proj2.weight": "val_proj2",
    "conv_qk.0.weight": "conv_qk_0_weight",
    "conv_qk.0.bias": "conv_qk_0_bias",
    "conv_qk.1.weight": "conv_qk_1_weight",
    "conv_qk.1.bias": "conv_qk_1_bias",
    "temp": "temp",
}
# Router (`zaya_block.router.*`) checkpoint suffix -> ZayaRouter flat param name.
_ZAYA_ROUTER_RENAME = {
    "down_proj.weight": "down_proj_weight",
    "down_proj.bias": "down_proj_bias",
    "rmsnorm_eda.weight": "rmsnorm_eda_weight",
    "router_states_scale": "router_states_scale",
    "router_mlp.0.weight": "router_mlp_0_weight",
    "router_mlp.0.bias": "router_mlp_0_bias",
    "router_mlp.2.weight": "router_mlp_2_weight",
    "router_mlp.2.bias": "router_mlp_2_bias",
    "router_mlp.4.weight": "router_mlp_4_weight",
    "balancing_biases": "balancing_biases",
}
# Expert key (both fields): local_experts.{e}.linear_fc{1,2}.{weight,weight_scale}.
_ZAYA_EXPERT_PATTERN = re.compile(
    r"^(?P<prefix>model\.layers\.\d+\.zaya_block\.experts)\.local_experts\."
    r"(?P<idx>\d+)\.(?P<fc>linear_fc1|linear_fc2)\.(?P<field>weight|weight_scale)$"
)


def _zaya_remap(ckpt_key: str) -> str | None:
    """Map a non-expert ZAYA checkpoint key to its native module key (None to skip).

    Expert tensors (`local_experts.{e}.linear_fc*.{weight,weight_scale}`) are handled separately
    (stacked over E, kept fp8), so they return None here."""
    if _ZAYA_EXPERT_PATTERN.match(ckpt_key) is not None:
        return None  # expert weight/scale -> handled by the fp8 stacking path
    # Top-level final merge: model.res_scale.* -> model.res_scale_final.*
    if ckpt_key.startswith("model.res_scale."):
        return "model.res_scale_final." + ckpt_key[len("model.res_scale.") :]
    # CCA conv front-end: ...self_attn.qkv.<sub> -> ...self_attn.qkv.<flat>
    m = re.match(r"^(model\.layers\.\d+\.self_attn\.qkv)\.(.+)$", ckpt_key)
    if m is not None:
        sub = _ZAYA_CCA_RENAME.get(m.group(2))
        assert sub is not None, f"unmapped CCA qkv key suffix: {m.group(2)!r} ({ckpt_key})"
        return f"{m.group(1)}.{sub}"
    # Router: ...zaya_block.router.<sub> -> ...zaya_block.router.<flat>
    m = re.match(r"^(model\.layers\.\d+\.zaya_block\.router)\.(.+)$", ckpt_key)
    if m is not None:
        sub = _ZAYA_ROUTER_RENAME.get(m.group(2))
        assert sub is not None, f"unmapped router key suffix: {m.group(2)!r} ({ckpt_key})"
        return f"{m.group(1)}.{sub}"
    # Direct: embed_tokens, final_norm, input_norm, layer res_scale, o_proj (LinearOProj.weight).
    return ckpt_key


def _load_zaya_weight(
    model_folder: str, device: torch.device, config
) -> Iterator[Tuple[str, torch.Tensor]]:
    """Streaming loader for the ZAYA1-8B CCA-hybrid fp8 checkpoint (TP=1).

    Plain keys are renamed by `_zaya_remap`. The fp8 routed experts STAY fp8: the raw F8_E4M3
    `weight` and per-output-channel F32 `weight_scale` are stacked over the 16 experts into
    `...experts.{gate_up_proj,down_proj}.{weight,weight_scale}` and consumed by `_GroupedFP8Experts`
    (dequant deferred to compute). Dequantizing to bf16 at load is ~16 GB and OOMs the 16 GB card;
    fp8 storage is ~8 GB. `tie_word_embeddings` -> no separate `lm_head.weight`."""
    tp_info = get_tp_info()
    files = glob.glob(f"{model_folder}/*.safetensors")
    files = [f for f in files if not f.endswith("consolidated.safetensors")] or files

    # Expert parallelism: each replica loads ONLY its expert shard [offset : offset+local). EP off
    # (single replica or DP-only) loads the full E. is_ep_enabled() already gates on dp_size>1.
    dp_info = get_dp_info()
    if is_ep_enabled():
        assert config.num_experts % dp_info.dp_size == 0, (
            f"EP needs num_experts ({config.num_experts}) divisible by dp_size ({dp_info.dp_size})"
        )
        ep_local = config.num_experts // dp_info.dp_size
        ep_offset = dp_info.dp_rank * ep_local
    else:
        ep_local, ep_offset = config.num_experts, 0

    # native gemm key -> {"weight": {e: w_fp8 [N,K]}, "weight_scale": {e: scale [N,1]}}; the two
    # stacks (weight, weight_scale) flush independently once all (local) experts of that field arrive.
    # Indices are stored as LOCAL ids (global gid - ep_offset) so the stack is over 0..ep_local-1.
    expert_buf: Dict[str, Dict[str, Dict[int, torch.Tensor]]] = {}
    _FC_TO_NATIVE = {"linear_fc1": "gate_up_proj", "linear_fc2": "down_proj"}

    def _store_expert(ckpt_name: str, tensor: torch.Tensor) -> Iterator[Tuple[str, torch.Tensor]]:
        m = _ZAYA_EXPERT_PATTERN.match(ckpt_name)
        assert m is not None, f"unexpected expert key: {ckpt_name}"
        gid = int(m.group("idx"))
        # EP: skip experts this replica does not own (load only the local shard).
        if not (ep_offset <= gid < ep_offset + ep_local):
            return
        local_id = gid - ep_offset
        native_key = f"{m.group('prefix')}.{_FC_TO_NATIVE[m.group('fc')]}"
        field = m.group("field")
        fields = expert_buf.setdefault(native_key, {"weight": {}, "weight_scale": {}})
        slots = fields[field]
        slots[local_id] = tensor
        if len(slots) != ep_local:
            return
        # weight: [N,K] fp8 -> stack [local,N,K] fp8. weight_scale: [N,1] f32 -> [local,N,1] f32.
        stacked = torch.stack([slots[e] for e in range(ep_local)], dim=0).contiguous()
        del fields[field]
        if not fields.get("weight") and not fields.get("weight_scale"):
            del expert_buf[native_key]
        yield f"{native_key}.{field}", stacked

    for file in tqdm(files, desc="Loading weights", disable=not tp_info.is_primary()):
        with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
            for ckpt_name in f.keys():
                if _ZAYA_EXPERT_PATTERN.match(ckpt_name) is not None:
                    yield from _store_expert(ckpt_name, f.get_tensor(ckpt_name))
                    continue
                native = _zaya_remap(ckpt_name)
                if native is None:
                    continue
                yield native, f.get_tensor(ckpt_name)

    assert not expert_buf, f"incomplete Zaya expert stacks: {list(expert_buf.keys())}"


def load_weight(model_path: str, device: torch.device) -> Iterator[Tuple[str, torch.Tensor]]:
    """Streaming weight loader. Yields (name, tensor) pairs already sharded, merged,
    and on device. Peak CPU memory: one full tensor + a small merge buffer."""
    from .config import ModelConfig

    model_folder = download_hf_weight(model_path)
    config = ModelConfig.from_hf(cached_load_hf_config(model_path))
    if config.is_gdn_hybrid:
        yield from _load_qwen3_5_weight(model_folder, device, config)
        return
    if config.is_cca_hybrid:
        yield from _load_zaya_weight(model_folder, device, config)
        return
    files = glob.glob(f"{model_folder}/*.safetensors")
    files = [f for f in files if not f.endswith("consolidated.safetensors")] or files
    tp_info = get_tp_info()

    # Buffer for merge groups: merged_key -> {slot: tensor}
    merge_buf: Dict[str, Dict[str, torch.Tensor]] = {}
    expert_buf: Dict[str, Dict[int, torch.Tensor]] = {}
    # EP: keep only this replica's expert shard, re-indexed to 0..ep_local-1. Off => full stack.
    _ep_shard, _ep_local, _ep_offset = _ep_expert_shard(config)
    for file in tqdm(files, desc="Loading weights", disable=not tp_info.is_primary()):
        with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
            for ckpt_name in f.keys():
                name = ckpt_name
                # Strip multimodal wrapper prefix, skip vision/projector weights
                if name.startswith(("vision_tower.", "multi_modal_projector.")):
                    continue
                # Appended MTP / next-token-prediction layers (GLM-4.x / DeepSeek): layers.<n> with
                # n >= num_layers is the MTP head. Skip it UNLESS the model loads one (mtp proposer),
                # in which case remap `(model.)layers.<num_layers>.X` -> `mtp.X` so the model's
                # GLMMTPHead receives it (and the merge/shard/expert-stack pipeline below applies).
                if _is_beyond_decoder(name, config.num_layers):
                    if config.num_nextn_predict_layers <= 0:
                        continue
                    name = _remap_glm_mtp(name, config.num_layers)
                    if name is None:
                        continue
                # GPTQ act-order indices: with desc_act=False the group map is the trivial
                # arange(K)//group_size, already implied by the op's grouped layout, so g_idx is
                # never materialized. (desc_act=True is rejected later in process_weights_after_load.)
                if name.endswith(".g_idx"):
                    continue
                raw = f.get_tensor(ckpt_name)
                name = name.removeprefix("language_model.")
                # AutoGPTQ emits a bias for EVERY linear, all-zero where the original layer had
                # bias=False (here: o_proj, all experts, the shared expert). The model declares no
                # bias buffer for those, and adding a zero bias is a no-op, so drop all-zero biases.
                # The genuine q/k/v biases are non-zero and flow on to merge as usual.
                if name.endswith(".bias") and not bool(raw.any()):
                    del raw
                    continue
                tensor = _shard_tensor(name, raw, tp_info.rank, tp_info.size, config.num_kv_heads)
                del raw

                if (info := _get_merge_info(name)) is None:
                    out = (name, tensor)
                else:
                    merged_key, slot, all_slots = info
                    merge_buf.setdefault(merged_key, {})[slot] = tensor
                    if not all(s in merge_buf[merged_key] for s in all_slots):
                        continue
                    parts = [merge_buf[merged_key][s] for s in all_slots]
                    del merge_buf[merged_key]
                    # AWQ quant siblings ((K,N//8)/(G,N)/(G,N//8)) merge along the
                    # output dim=1; dense .weight (N,K) and .bias merge along dim=0.
                    cat_dim = 1 if merged_key.endswith((".qweight", ".qzeros", ".scales")) else 0
                    out = (merged_key, torch.cat(parts, dim=cat_dim))

                if config.is_moe and (expert_info := _get_expert_stack_info(out[0])) is not None:
                    packed_key, expert_idx = expert_info
                    if _ep_shard and not (_ep_offset <= expert_idx < _ep_offset + _ep_local):
                        continue  # not this replica's expert
                    slots = expert_buf.setdefault(packed_key, {})
                    slots[expert_idx - _ep_offset] = out[1]
                    if len(slots) != _ep_local:
                        continue
                    experts = [slots[idx] for idx in range(_ep_local)]
                    del expert_buf[packed_key]
                    yield packed_key, torch.stack(experts, dim=0)
                else:  # Normal dense model
                    yield out[0], out[1]

    assert not merge_buf, f"Incomplete merge groups in checkpoint: {list(merge_buf.keys())}"
    assert not expert_buf, f"Incomplete expert tensors in checkpoint: {list(expert_buf.keys())}"
