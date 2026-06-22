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


def _load_qwen3_5_weight(
    model_folder: str, device: torch.device
) -> Iterator[Tuple[str, torch.Tensor]]:
    """Streaming loader for the Qwen3.5 GDN-hybrid checkpoint (TP=1). Applies `qwen3_5_remap`
    and buffers the ordered concat groups until complete. dtype coercion (A_log/dt_bias -> fp32)
    is the engine's job, exactly as for the dense path."""
    tp_info = get_tp_info()
    if tp_info.size != 1:
        raise NotImplementedError("Qwen3.5 GDN-hybrid weight loading is TP=1 only (TP is Phase 4)")
    files = glob.glob(f"{model_folder}/*.safetensors")
    files = [f for f in files if not f.endswith("consolidated.safetensors")] or files
    concat_buf: Dict[str, Dict[int, torch.Tensor]] = {}  # merged_key -> {slot: tensor}
    for file in tqdm(files, desc="Loading weights", disable=not tp_info.is_primary()):
        with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
            for name in f.keys():
                plan = qwen3_5_remap(name)
                if plan is None:
                    continue
                if plan[0] == "direct":
                    yield plan[1], f.get_tensor(name)
                    continue
                _, merged, slot, n_slots, cat_dim = plan
                concat_buf.setdefault(merged, {})[slot] = f.get_tensor(name)
                if len(concat_buf[merged]) != n_slots:
                    continue
                parts = [concat_buf[merged][i] for i in range(n_slots)]
                del concat_buf[merged]
                yield merged, torch.cat(parts, dim=cat_dim)
    assert not concat_buf, f"incomplete concat groups in checkpoint: {list(concat_buf.keys())}"


def load_weight(model_path: str, device: torch.device) -> Iterator[Tuple[str, torch.Tensor]]:
    """Streaming weight loader. Yields (name, tensor) pairs already sharded, merged,
    and on device. Peak CPU memory: one full tensor + a small merge buffer."""
    from .config import ModelConfig

    model_folder = download_hf_weight(model_path)
    config = ModelConfig.from_hf(cached_load_hf_config(model_path))
    if config.is_gdn_hybrid:
        yield from _load_qwen3_5_weight(model_folder, device)
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
