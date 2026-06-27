from __future__ import annotations

import glob
import re
from typing import Dict, Iterator, Tuple

import safetensors
import torch
from minisgl.distributed import get_tp_info
from minisgl.utils import cached_load_hf_config, div_ceil, download_hf_weight
from tqdm import tqdm

_SPLIT_DIM_0 = [".q_proj", ".k_proj", ".v_proj", ".gate_proj", ".up_proj"]
_SPLIT_DIM_1 = [".o_proj", ".down_proj"]

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


def _shard_tensor(key: str, value: torch.Tensor, r: int, n: int, num_kv_heads: int):
    """Extract rank r's shard from a single tensor. Returns a contiguous copy."""
    if any(key.count(sub) for sub in _SPLIT_DIM_0):
        is_kv_proj = any(key.count(sub) for sub in (".k_proj", ".v_proj"))
        if is_kv_proj and num_kv_heads is not None and num_kv_heads < n:
            head_dim = value.shape[0] // num_kv_heads
            head_idx = r * num_kv_heads // n
            return value[head_idx * head_dim : (head_idx + 1) * head_dim].clone()
        return value.chunk(n, dim=0)[r].clone()
    elif any(key.count(sub) for sub in _SPLIT_DIM_1):
        return value.chunk(n, dim=1)[r].clone()
    elif key.count("lm_head") or key.count("embed_tokens"):
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
_QWEN35_SKIP_PREFIXES = ("model.visual.", "visual.", "mtp.")
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


def _gate_up_merge(key: str):
    """gate_proj/up_proj -> gate_up_proj for the qwen3_5_moe shared + routed experts. Unlike the
    dense q/k/v path, qwen3_5 keeps full-attn q/k/v SEPARATE (q_proj carries the output gate), so
    this handles ONLY gate/up. Returns (merged_key, slot) or None."""
    for sub, slot in ((".gate_proj.", "gate"), (".up_proj.", "up")):
        if sub in key:
            return key.replace(sub, ".gate_up_proj."), slot
    return None


def qwen3_5_remap(ckpt_key: str):
    """Map a Qwen3.5 GDN-hybrid checkpoint key to a minisgl-native key plan. Pure (no tensors),
    so it is CPU-testable against the checkpoint header vs. the model's `state_dict()`.

    Returns one of:
      ``None``                                           -> skip this checkpoint tensor
      ``("direct", native_key)``                         -> rename only
      ``("concat", merged_key, slot, n_slots, cat_dim)`` -> one member of an ordered concat group
    """
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

    # ---- dense MLP (4B) + shared expert (35B): col gate/up (dim 0), row down (dim 1) ----
    if name.endswith((".gate_proj.weight", ".up_proj.weight")):
        return t.chunk(n, dim=0)[r].clone()
    if name.endswith(".down_proj.weight"):
        return t.chunk(n, dim=1)[r].clone()

    # ---- vocab-parallel embedding + untied lm_head ----
    if name.endswith("embed_tokens.weight") or name == "lm_head.weight":
        num = t.shape[0]
        per = div_ceil(num, n)
        return t[r * per : min((r + 1) * per, num)].clone()

    # norms, router gate (.mlp.gate.weight), shared_expert_gate -> replicated
    return t


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
        # MoE expert stacking (experts.<e>.<name> -> experts.<name>, stacked over E).
        if config.is_moe and (einfo := _get_expert_stack_info(native_key)) is not None:
            packed_key, idx = einfo
            slots = expert_buf.setdefault(packed_key, {})
            slots[idx] = tensor
            if len(slots) != config.num_experts:
                return
            experts = [slots[i] for i in range(config.num_experts)]
            del expert_buf[packed_key]
            yield packed_key, torch.stack(experts, dim=0)
        else:
            yield native_key, tensor

    for file in tqdm(files, desc="Loading weights", disable=not tp_info.is_primary()):
        with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
            for name in f.keys():
                plan = qwen3_5_remap(name)
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


def load_weight(model_path: str, device: torch.device) -> Iterator[Tuple[str, torch.Tensor]]:
    """Streaming weight loader. Yields (name, tensor) pairs already sharded, merged,
    and on device. Peak CPU memory: one full tensor + a small merge buffer."""
    from .config import ModelConfig

    model_folder = download_hf_weight(model_path)
    config = ModelConfig.from_hf(cached_load_hf_config(model_path))
    if config.is_gdn_hybrid:
        yield from _load_qwen3_5_weight(model_folder, device, config)
        return
    files = glob.glob(f"{model_folder}/*.safetensors")
    files = [f for f in files if not f.endswith("consolidated.safetensors")] or files
    tp_info = get_tp_info()

    # Buffer for merge groups: merged_key -> {slot: tensor}
    merge_buf: Dict[str, Dict[str, torch.Tensor]] = {}
    expert_buf: Dict[str, Dict[int, torch.Tensor]] = {}
    for file in tqdm(files, desc="Loading weights", disable=not tp_info.is_primary()):
        with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
            for name in f.keys():
                # Strip multimodal wrapper prefix, skip vision/projector weights
                if name.startswith(("vision_tower.", "multi_modal_projector.")):
                    continue
                # Skip appended MTP / next-token-prediction layers (GLM-4.x / DeepSeek): we serve
                # the decoder only. layers.<n> with n >= num_layers is the MTP head.
                if _is_beyond_decoder(name, config.num_layers):
                    continue
                # GPTQ act-order indices: with desc_act=False the group map is the trivial
                # arange(K)//group_size, already implied by the op's grouped layout, so g_idx is
                # never materialized. (desc_act=True is rejected later in process_weights_after_load.)
                if name.endswith(".g_idx"):
                    continue
                raw = f.get_tensor(name)
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
                    slots = expert_buf.setdefault(packed_key, {})
                    slots[expert_idx] = out[1]
                    if len(slots) != config.num_experts:
                        continue
                    experts = [slots[idx] for idx in range(config.num_experts)]
                    del expert_buf[packed_key]
                    yield packed_key, torch.stack(experts, dim=0)
                else:  # Normal dense model
                    yield out[0], out[1]

    assert not merge_buf, f"Incomplete merge groups in checkpoint: {list(merge_buf.keys())}"
    assert not expert_buf, f"Incomplete expert tensors in checkpoint: {list(expert_buf.keys())}"
