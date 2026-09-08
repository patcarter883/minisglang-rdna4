from __future__ import annotations

import glob
import json
import os
import re
import time
from typing import (
    Any,
    Callable,
    Collection,
    Dict,
    FrozenSet,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
)

import safetensors
import torch
from minisgl.distributed import get_dp_info, get_ep_rank, get_ep_size, get_tp_info, is_ep_enabled
from minisgl.quant import nvfp4
from minisgl.utils import cached_load_hf_config, div_ceil, download_hf_weight, init_logger
from minisgl.weights import ckpt_read
from tqdm import tqdm

logger = init_logger(__name__)


def cast_checkpoint_tensor(key: str, v: torch.Tensor, model_dtype: torch.dtype) -> torch.Tensor:
    """The dtype normalization every `load_weight` consumer MUST apply. Part of loading, not of the
    engine.

    A checkpoint's stored dtype is not the dtype the layer runs in, and the rules are per-tensor-KIND
    rather than global: packed quant blobs and scales keep their encoding, the GDN gating params and
    the fp32 residual affines are upcast, everything else follows the model dtype.

    THIS LIVED AS A CLOSURE INSIDE `Engine._load_weight_state_dict`, which made it invisible to every
    other caller of `load_weight` — and that is not hypothetical. The `qwen4_exp` bring-up harness
    fed the loader's raw output straight into `load_state_dict` and died in `gdn_prefill_wmma` with
    "expected scalar type Float but found BFloat16", because `RadixArk/Qwen3.8-Flash-Next-NVFP4`
    ships BOTH `A_log` and `dt_bias` as bf16 while the kernels and the module's `nn.Parameter` are
    fp32. `GDNLinearAttn` loads its wrapped module with `assign=True`, so a checkpoint dtype survives
    into the parameter and nothing downstream would have caught it.

    `model_dtype` is passed rather than read from a global because it is the ENGINE's activation
    dtype, which a bring-up harness or an offline repack picks for itself.
    """
    if not v.is_floating_point() or key.endswith(".scales"):
        return v
    # ZAYA experts stay fp8: the F8_E4M3 weight must NOT be upcast (that re-inflates ~8 GB fp8 ->
    # ~16 GB bf16 and OOMs the 16 GB card), and its per-channel fp32 weight_scale must keep fp32.
    # Dequant is deferred to compute (_GroupedFP8Experts).
    if v.dtype == torch.float8_e4m3fn:
        return v
    if key.endswith(".weight_scale"):
        # fp8 (ZAYA) per-channel scales are fp32 and MUST stay fp32; compressed-tensors int4 scales
        # ship fp16 OR bf16 -> normalize to fp16 (the op's + the linear buffer's declared scale
        # dtype) so a bf16-scale head and an fp16-scale backbone both load against the same float16
        # buffer.
        return v if v.dtype == torch.float32 else v.to(torch.float16)
    # GDN gating params stay fp32. The kernels and the model's nn.Parameter require fp32 whatever the
    # checkpoint stores, so the upcast is unconditional: A_log ships fp32 in the Qwen3.5 line and
    # BF16 in Qwen3.8-Flash-Next, dt_bias ships bf16 in both.
    if key.endswith((".A_log", ".dt_bias")):
        return v.to(torch.float32)
    # ZAYA router balancing_biases is an fp32 buffer (added to the fp32 router softmax; ships bf16,
    # upcast to keep the buffer fp32 and the choice numerics faithful). The ResidualScaling affines
    # run in fp32 on the fp32 residual stream (scale_residual_merge ships bf16 -> upcast so the merge
    # stays fp32). CCA conv/temp can stay bf16 (post_load upcasts them to fp32 once for the
    # kernel-weight cache).
    if key.endswith(
        (
            ".balancing_biases",
            ".hidden_states_scale",
            ".hidden_states_bias",
            ".residual_scale",
            ".residual_bias",
        )
    ):
        return v.to(torch.float32)
    return v.to(model_dtype)


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


def checkpoint_tensor_names(model_path: str) -> "FrozenSet[str]":
    """Every tensor name in a checkpoint, WITHOUT reading a single tensor.

    Prefers the safetensors index (`*.index.json` -> `weight_map`, one small JSON); falls back to the
    per-file safetensors HEADERS, which `safe_open` reads lazily, so even a sharded 100 GB checkpoint
    costs a few KB. Both the model builder and the streaming loader consult this (through
    `ModelConfig.from_hf`) and must agree, so it is deliberately cheap enough to call twice."""
    folder = download_hf_weight(model_path)
    names: "set[str]" = set()
    for index_file in glob.glob(f"{folder}/*.index.json"):
        try:
            with open(index_file) as fh:
                names.update(json.load(fh).get("weight_map", {}).keys())
        except (OSError, ValueError):
            continue  # unreadable/malformed index -> fall through to the headers
        if names:
            return frozenset(names)
    for file in glob.glob(f"{folder}/*.safetensors"):
        with safetensors.safe_open(file, framework="pt", device="cpu") as f:
            names.update(f.keys())
    return frozenset(names)


def checkpoint_ships_mtp(names: "Collection[str]", num_layers: int) -> bool:
    """Does this checkpoint ACTUALLY ship a speculative (MTP / next-n) head?

    Answered from the TENSORS, never from a config field — a config can claim a head the weights do
    not back. Measured 2026-08-04: `cyankiwi/Agents-A1-AWQ-INT4` sets `mtp_num_hidden_layers: 1` in
    its own config.json and ships ZERO `mtp.*` tensors (the quantizer dropped the head and left the
    field), so trusting the field built a 22-buffer head that `load_state_dict` could not fill.

    Uses the loader's own two namespaces, so this cannot drift from what load actually maps:
      * Qwen3.5 / Qwen3.6  — a dedicated `mtp.*` namespace.
      * GLM-4.x / DeepSeek — appended `(model.)layers.<n>`, n >= num_layers (`_is_beyond_decoder`,
        the same predicate the loader skips/remaps them with).
    """
    return any(n.startswith("mtp.") or _is_beyond_decoder(n, num_layers) for n in names)


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


class _ExpertStacker:
    """Accumulate an MoE layer's per-expert tensors INTO a preallocated ``[E, ...]`` output.

    WHY THIS IS NOT ``torch.stack`` OVER A DICT, which is what both loaders used to do.

    ``torch.stack`` must have every source alive at the moment it allocates its result, so buffering
    all ``E`` experts and stacking at the end holds the layer's expert row TWICE on the card at once.
    On this checkpoint the largest row is 400 MiB, so that one line peaked at 800 MiB of VRAM for a
    400 MiB tensor — and it did so INSIDE ``ChunkedWeightLoader.apply_chunk``, whose whole contract
    is "the live set is one chunk, by construction". It was not: the peak was one chunk plus a
    doubled expert row, and the gap is invisible in the ledger because both allocations retire
    before the next sample.

    That gap is a CAPACITY bug, not an efficiency nit. The 48-layer TP=2 offload serve is bounded by
    host RAM, the only lever against that is moving layers to the device tier, and the device tier
    was pinned at 11 layers because 12 OOM'd here — asking for 400 MiB with 224 MiB free while torch
    held 486 MiB reserved-but-unallocated. Removing the double-buffer frees ~400 MiB of device peak,
    which is what lets the 12th layer be device-resident and drops the node's pinned host arena from
    54.20 GiB to 52.73 GiB.

    BIT-IDENTICAL to the old form: ``out[i].copy_(expert_i)`` for ``i in 0..E-1`` writes exactly the
    rows ``torch.stack(experts, dim=0)`` writes, in the same dim-0 order, with no arithmetic. The
    output is allocated on the FIRST expert's device/dtype, so a source that disagrees on dtype,
    shape or device raises in ``copy_`` here instead of silently broadcasting.

    Held per packed key because experts arrive interleaved across shard files; the completeness
    asserts at the end of each loader read :attr:`pending`.
    """

    __slots__ = ("_out", "_filled", "_width")

    def __init__(self) -> None:
        self._out: Dict[str, torch.Tensor] = {}
        self._filled: Dict[str, set] = {}
        self._width: Dict[str, int] = {}

    def add(
        self, packed_key: str, idx: int, tensor: torch.Tensor, num_experts: int
    ) -> "torch.Tensor | None":
        """Place expert ``idx``. Returns the finished ``[E, ...]`` tensor once all ``E`` landed."""
        out = self._out.get(packed_key)
        if out is None:
            out = torch.empty(
                (num_experts, *tensor.shape), dtype=tensor.dtype, device=tensor.device
            )
            self._out[packed_key] = out
            self._filled[packed_key] = set()
            self._width[packed_key] = num_experts
        elif self._width[packed_key] != num_experts:
            # The expert count for one packed key changed mid-stream (an EP-shard/replication
            # decision applied inconsistently). Say so: the alternative is a partially-filled
            # output that loads without complaint.
            raise AssertionError(
                f"{packed_key}: expert count changed mid-stack "
                f"({self._width[packed_key]} -> {num_experts})"
            )
        filled = self._filled[packed_key]
        if idx in filled:
            raise AssertionError(
                f"{packed_key}: expert {idx} delivered twice; the second write would silently win"
            )
        if not (0 <= idx < num_experts):
            raise AssertionError(
                f"{packed_key}: expert index {idx} outside [0, {num_experts}) — an EP shard offset "
                f"was applied twice or not at all"
            )
        # EXACT shape/dtype, because `Tensor.copy_` BROADCASTS and `torch.stack` does not. Without
        # this, a checkpoint whose experts disagree on shape — a mis-sharded rank, a mixed-width MoE
        # — would have RAISED under `torch.stack` and would now silently fan one expert's row across
        # the slot. That is the one way this rewrite could be less safe than the code it replaces,
        # so it is checked rather than argued.
        if tuple(tensor.shape) != tuple(out.shape[1:]) or tensor.dtype != out.dtype:
            raise AssertionError(
                f"{packed_key}: expert {idx} is {tuple(tensor.shape)}/{tensor.dtype}, but the stack "
                f"was opened as {tuple(out.shape[1:])}/{out.dtype} by its first expert"
            )
        out[idx].copy_(tensor)
        filled.add(idx)
        if len(filled) != num_experts:
            return None
        del self._filled[packed_key], self._width[packed_key]
        return self._out.pop(packed_key)

    @property
    def pending(self) -> "List[str]":
        return sorted(self._out)

    def __bool__(self) -> bool:
        return bool(self._out)

    def __len__(self) -> int:
        return len(self._out)


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
    if ckpt_key.endswith((".k_scale", ".v_scale")):
        # fp8 KV-cache scales (quantization_config.kv_cache_scheme, e.g. Qwen3.8-27B-NVFP4 ships one
        # pair per FULL-attention layer). NOT model params: the cache reads them straight out of the
        # checkpoint files via kvcache/fp8_scales.py, and only when the engine is actually running an
        # fp8 KV cache. Skipped here so they do not land as unexpected keys in load_state_dict.
        return None
    if ckpt_key.startswith("mtp."):
        return _qwen3_5_mtp_remap(ckpt_key) if load_mtp else None
    if ckpt_key.startswith(_QWEN35_SKIP_PREFIXES):
        return None
    if ckpt_key == "lm_head.weight":
        return ("direct", "lm_head.weight")  # untied (qwen3_5_moe); top-level, no LM prefix
    if ckpt_key == "lm_head.weight_scale":
        # fp8 lm_head (mixed-precision checkpoints put lm_head in the fp8 W8A8 group). The loader
        # folds this channel scale INTO `lm_head.weight` before the remap runs, so it is never a
        # param of its own. Skipped rather than raised — it is a legitimate key, not an unknown one.
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

    # fp8 W8A8 (compressed-tensors `float-quantized`, strategy:channel) ships a PER-OUTPUT-CHANNEL
    # weight_scale of shape (N,1). On a ROW-parallel linear (o_proj / down_proj / GDN out_proj) the
    # OUTPUT dim is not sharded — only the input is — so that scale must REPLICATE. It shares the
    # `.weight_scale` suffix with the NVFP4/int4 per-GROUP scale (N, K//g), which DOES split on dim 1,
    # so the two are told apart by the singleton group axis. Without this the row-parallel rules below
    # chunk a size-1 dim and rank 1 gets an EMPTY (N,0) scale.
    if (name.endswith((".o_proj.weight_scale", ".down_proj.weight_scale", ".out_proj.weight_scale"))
            and t.dim() == 2 and t.shape[1] == 1):
        return t

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
        # Under EP the experts are sharded by EXPERT INDEX (the per-expert stack + _ep_expert_shard),
        # NOT tensor-sharded — each rank holds its expert subset at FULL intermediate. So skip the TP
        # gate/up/down intermediate split here (DP+EP has tp=1 so this was a no-op; EP-over-TP has tp>1
        # and MUST keep full intermediate to match the MoELayer's EP buffer sizing). EXCEPT the MTP draft
        # head, which is built REPLICATED (force_no_ep) → plain-TP tensor-split, NOT EP — so fall through.
        if is_ep_enabled() and not name.startswith("mtp"):
            return t
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
        # weight_zero_point (ASYMMETRIC CT, [N//pf, K//g]) follows the SAME axes as its weight — it is
        # int4-packed 8-per-int32 along the OUTPUT N, so a column-parallel gate/up splits its dim 0 in
        # PACKED units (needs N/n % pf == 0, true for these checkpoints) and a row-parallel down keeps
        # the whole packed N and splits the input group dim 1. Identical to the dense CT rules below.
        # Without this the expert zero-points would REPLICATE while their weights shard, and the load
        # would fail on shape (loudly — never a silent half-sharded dequant).
        # NVFP4 split arm: `.weight_global` is the per-OUTPUT-CHANNEL f32 global MULTIPLIER, an (N,)
        # vector. down_proj is row-parallel — it splits the INPUT K, so its output N is full width on
        # every rank and its global REPLICATES. That is the one leaf whose axis differs from its own
        # weight's, so it is matched FIRST and explicitly; letting it reach the `.down_proj.*` rule
        # below would chunk a 1-D tensor on dim 1 and raise, and letting it fall all the way through
        # to "replicate" would be right for the wrong reason and would break the moment somebody
        # added a `.down_proj.weight_global`-matching suffix.
        if name.endswith(".down_proj.weight_global"):
            return t
        if name.endswith(
            (".gate_proj.weight_packed", ".gate_proj.weight_scale", ".gate_proj.weight_zero_point",
             ".gate_proj.weight_global",
             ".up_proj.weight_packed", ".up_proj.weight_scale", ".up_proj.weight_zero_point",
             ".up_proj.weight_global")
        ):
            return t.chunk(n, dim=0)[r].clone()
        if name.endswith((".down_proj.weight_packed", ".down_proj.weight_scale",
                          ".down_proj.weight_zero_point")):
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
    # (a fp8 lm_head's channel scale is per-VOCAB-row, so it rides the same split as its weight)
    if name.endswith("embed_tokens.weight") or name in ("lm_head.weight", "lm_head.weight_scale"):
        num = t.shape[0]
        per = div_ceil(num, n)
        return t[r * per : min((r + 1) * per, num)].clone()

    # norms, router gate (.mlp.gate.weight), shared_expert_gate -> replicated
    return t


# Qwen3.8-Flash-Next submodules that are REPLICATED ON EVERY RANK, by construction of the model:
# each one is built from `LinearReplicated` / a plain `RMSNorm` / a bare buffer, so its parameter is
# full-width on every rank and a shard would leave rank r holding a slice the layer will never index.
#
# This set is spelled out rather than left to `_shard_qwen4_exp`'s fall-through, even though the
# fall-through IS "replicate" and would produce the same tensors today. The reason is the direction
# each mistake fails in. A missing shard rule for something that SHOULD shard fails loudly at
# `load_state_dict` on a shape mismatch; a rule that shards something that should NOT is what this
# list prevents, and that one is silent when the shapes happen to divide — `input_mix_weight_up` is
# [10240, 320] and `key_proj` is [10240, 2560], both evenly splittable at TP=2 by any rule that grew
# a matching suffix later. Naming them here means a future rule that would capture one of these has
# to delete an entry that says why it exists.
#
#   * `*_hyper_connection.*` / `hyper_connection_mixer.*` — the hyper-connection MIXES across the
#     hc_count residual streams of the WHOLE hidden state. A column split would give each rank a
#     partial mix with no all-reduce to complete it: right shapes, plausible text, wrong model. The
#     checkpoint's quant `ignore` list carries `*hyper_connection*` so these stay bf16, and "not
#     quantized" is not "not sharded" — the two are easy to conflate and only this list separates them.
#   * `self_attn.indexer.*` — `QSAIndexer` is parameters-only (bring-up plan T5); `index_qk_proj` is
#     `LinearReplicated` and both layernorms are per-head-dim.
#   * `.ple.*` — the PLE block's `key_proj`/`value_proj` are `LinearReplicated`, its conv1d is a flat
#     depthwise buffer over the wide stream, and its three `GroupedRMSNorm`s are grouped by
#     hidden_size. The block consumes and produces the WIDE stream, which is not sharded.
#   * `.mlp.gate.weight` (router logits over all 512 experts) and `.mlp.shared_expert_gate.weight`
#     (a single sigmoid row) — `LinearReplicated`. The router MUST be replicated: every rank routes
#     the same tokens to the same experts, and a rank that disagreed would compute a different
#     top-k and the all-reduce would sum two different models' outputs.
_QWEN4EXP_REPLICATED_SUBSTRINGS = (
    "_hyper_connection.",
    "hyper_connection_mixer.",
    ".self_attn.indexer.",
    ".ple.",
)
_QWEN4EXP_REPLICATED_SUFFIXES = (
    ".mlp.gate.weight",
    ".mlp.shared_expert_gate.weight",
)

# The routed experts, in modelopt's NVFP4 spelling. `qwen4_exp_remap` renames these leaves to the
# repo-native `.weight_packed` / `.weight_scale` AFTER this function runs, so the rules here are
# keyed on the CHECKPOINT spelling: a bare `.weight` (the packed E2M1 blob) and `.weight_scale` (the
# e4m3 per-group block scale, emitted verbatim by `nvfp4.nvfp4_leaf_scales` before the shard runs).
_Q4_EXPERT_COL = (
    ".gate_proj.weight", ".gate_proj.weight_scale",
    ".up_proj.weight", ".up_proj.weight_scale",
    # `.weight_global` is the NVFP4 per-OUTPUT-CHANNEL global, an (N,) f32 vector. gate/up are
    # column-parallel, so their output N splits — dim 0 of a 1-D vector, which is the same rule and
    # the same axis as their weight and block scale. Named explicitly rather than left to the
    # fall-through, per this file's rule that a shard decision is stated, not inherited.
    ".gate_proj.weight_global", ".up_proj.weight_global",
)
_Q4_EXPERT_ROW = (".down_proj.weight", ".down_proj.weight_scale")
# down_proj is ROW-parallel: it splits the INPUT K, so its OUTPUT N is full width on every rank and
# its per-output-channel global is REPLICATED, not split. This is the one place where the global's
# axis differs from its weight's, which is exactly why it is a separate tuple with its own comment
# instead of an entry in `_Q4_EXPERT_ROW` — chunking it on dim 1 would fail (it is 1-D) and chunking
# it on dim 0 would hand each rank a quarter of the output channels it actually computes.
_Q4_EXPERT_REPLICATED = (".down_proj.weight_global",)


def _shard_qwen4_exp(name: str, t: torch.Tensor, r: int, n: int, config) -> torch.Tensor:
    """Extract rank r's TP shard of a Qwen3.8-Flash-Next (`qwen4_exp`) CHECKPOINT tensor.

    Applied at READ time on the checkpoint name, BEFORE the loader's GDN `in_proj_qkv`+`in_proj_z`
    concat, its gate/up merge and its per-expert stack — so all three compose rank-local parts into
    the rank-local fused buffers the TP-aware model declares. `n == 1` is the identity, so the TP=1
    path is byte-for-byte what it was before this function existed.

    MOST OF THIS MODEL SHARDS EXACTLY LIKE QWEN3.5 AND IS DELEGATED, not copied: the GDN
    head-parallel splits (qkv/conv1d by [key|key|value] head block, z/b/a/A_log/dt_bias per v-head,
    out_proj row-parallel), the gated-attention splits (q_proj carries q interleaved with its
    per-head sigmoid gate, so an output-dim chunk keeps whole heads on a rank; o_proj row-parallel),
    the shared expert (col gate/up, row down) and the vocab-parallel embedding/lm_head are all
    `_shard_qwen3_5`'s rules and stay there. Two things are qwen4_exp's own and are handled here
    first:

      1. **The replicated submodules** — hyper-connections, the QSA indexer, the PLE block, both
         gates. See `_QWEN4EXP_REPLICATED_SUBSTRINGS` for why they are enumerated rather than left
         to fall through.
      2. **The routed experts in modelopt's NVFP4 spelling.** This is the rule that does not exist
         anywhere else. The checkpoint spells the packed blob `.weight` (U8 [N, K/2]) and its
         per-group scale `.weight_scale` (F8_E4M3 [N, K/16]) — where compressed-tensors would say
         `.weight_packed` / `.weight_scale`. NVFP4 packs along the INPUT K, so:
            gate/up  (column-parallel, output N)  -> split dim 0: [640,1280]->[320,1280],
                                                    scale [640,160]->[320,160]
            down     (row-parallel,    input  K)  -> split dim 1: [2560,320]->[2560,160],
                                                    scale [2560,40]->[2560,20]
         The down_proj scale split is the one worth checking by hand: K=640 elements is 320 packed
         bytes and 40 scale groups, so at TP=2 a rank gets 320 elements = 160 bytes = 20 groups, and
         the three stay consistent only because 640/n stays a multiple of the group size 16. Below
         that the packed byte axis and the group axis would round differently and the dequant would
         read another rank's groups — so it is asserted, not assumed.

    A `.weight_scale` that fell through to `_shard_qwen3_5` would REPLICATE (that function has no
    NVFP4-scale rule; its `.weight_scale` cases are compressed-tensors-shaped), which is why these
    are matched here and not left to the delegate.
    """
    if n == 1:
        return t
    if any(s in name for s in _QWEN4EXP_REPLICATED_SUBSTRINGS):
        return t
    if name.endswith(_QWEN4EXP_REPLICATED_SUFFIXES):
        return t

    if ".mlp.experts." in name:
        # Under EP the experts are sharded by expert INDEX at full intermediate width, exactly as
        # `_shard_qwen3_5` documents — no tensor split here. (EP is off for this model today; the
        # branch is kept so enabling it later does not silently double-shard.)
        if is_ep_enabled():
            return t
        if name.endswith(_Q4_EXPERT_REPLICATED):
            return t
        if name.endswith(_Q4_EXPERT_COL):
            return t.chunk(n, dim=0)[r].clone()
        if name.endswith(_Q4_EXPERT_ROW):
            gs = config.quant.group_size if config.quant is not None else 16
            if name.endswith(".weight_scale") and t.shape[1] % n:
                raise ValueError(
                    f"{name}: NVFP4 down_proj scale has {t.shape[1]} groups, which tp={n} does not "
                    f"divide. The packed-byte axis and the group axis would round differently and "
                    f"each rank would dequantize its columns with another rank's group scales — "
                    f"silent, and fluent. moe_intermediate_size/{n} must stay a multiple of {gs}."
                )
            return t.chunk(n, dim=1)[r].clone()

    return _shard_qwen3_5(name, t, r, n, config)


def _ep_expert_shard(config) -> Tuple[bool, int, int]:
    """EP expert-shard params for the QUANTIZED MoE loaders, mirroring MoELayer's gate exactly (moe.py:
    `is_ep_enabled() and (fp8 or W4A8/W4A16 quant)`). Returns (should_shard, ep_local, ep_offset): when
    sharding, each replica keeps only experts [offset : offset+local) and stacks them at local ids
    0..local-1; off => (False, num_experts, 0) i.e. the full replicated stack (unchanged). Only the
    EP-eligible quant formats shard — W4A8 (GPTQ/AWQ) and W4A16 (compressed-tensors), which share the
    w4a8_moe op layout; unquantized experts are replicated (must match the MoELayer decision or
    the loaded stack won't fit the buffer). fp8/ZAYA has its own sharded loader (_load_zaya_weight)."""
    q = getattr(config, "quant", None)
    ep_quant = q is not None and (q.is_gptq or q.is_awq or q.is_compressed_tensors)
    if is_ep_enabled() and ep_quant:
        ep_size, ep_rank = get_ep_size(), get_ep_rank()   # TP group (EP-over-TP) or DP replicas
        assert config.num_experts % ep_size == 0, (
            f"EP needs num_experts ({config.num_experts}) divisible by ep_size ({ep_size})"
        )
        ep_local = config.num_experts // ep_size
        return True, ep_local, ep_rank * ep_local
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
    expert_buf = _ExpertStacker()  # MoE per-expert -> filled into a preallocated [E, ...]
    # NVFP4: pair each proj's e4m3 block scale + per-tensor global to fold them into one fp16 per-group
    # scale at the LEAF (before remap/concat/merge/stack) — see below.
    # NVFP4 is identified STRUCTURALLY, PER MODULE: a proj is NVFP4 iff the checkpoint ships it a
    # `.weight_global_scale`. A MIXED-PRECISION checkpoint (Qwen3.8-27B-NVFP4) carries NVFP4 and fp8
    # W8A8 side by side and BOTH spell their scale `.weight_scale`, so the config-wide `quant.is_nvfp4`
    # flag this replaces cannot separate them — it would buffer every fp8 CHANNEL scale waiting for a
    # global that never arrives, then trip the completeness assert below. Keying on the global's
    # presence is exact for single-format and mixed checkpoints alike.
    nvfp4_fold_buf: Dict[str, Dict[str, torch.Tensor]] = {}
    lm_head_buf: Dict[str, torch.Tensor] = {}  # fp8 lm_head weight + channel scale -> dequantized
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
            # The MTP draft head is REPLICATED (force_no_ep) — keep ALL its experts local, even when the
            # backbone (config quant) is EP-sharded. Matches the bf16 MTP MoELayer's full local_num_experts
            # (else a load-time count mismatch: model 256 vs loader-sharded 128).
            mtp_head = packed_key.startswith("mtp")
            e_shard = _ep_shard and not mtp_head
            e_local = config.num_experts if mtp_head else _ep_local
            e_offset = 0 if mtp_head else _ep_offset
            if e_shard and not (e_offset <= idx < e_offset + e_local):
                return  # not this replica's expert
            local_idx = idx - e_offset
            stacked = expert_buf.add(packed_key, local_idx, tensor, e_local)
            if stacked is None:
                return
            yield packed_key, stacked
        else:
            yield native_key, tensor

    def leaf(f, name: str, override: "torch.Tensor | None"):
        """ONE resolved leaf -> zero or more (native_key, tensor) pairs.

        Extracted from the read loop because ONE checkpoint key can now produce TWO leaves: an NVFP4
        routed expert emits its e4m3 block scale AND its per-output-channel global, and both must
        take the IDENTICAL remap -> TP shard -> GDN concat -> gate|up merge -> expert stack path.
        Duplicating that path for the second leaf is how the global would end up sharded on the wrong
        axis or skipping a merge — silently, since every shape still works out.

        `override` is a tensor the caller already materialized (an NVFP4 leaf scale); None means
        "read `name` from this shard".
        """
        # fp8 lm_head -> DEQUANTIZE to the compute dtype at load: weight (V,H) e4m3 times its
        # per-output-channel scale (V,1). Both split on the VOCAB dim, so each rank folds only
        # ITS OWN rows and the full bf16 head is never materialised.
        #
        # Serving it AS fp8 would be smaller and cheaper to stream, and is tempting on a 248k
        # vocab — but it is WRONG here, and not merely because ParallelLMHead has no quant
        # plumbing. The LM head must stay M-INVARIANT: quant/kernels.py dispatches between
        # kernel arms as a function of M and those arms disagree numerically (dense decode_gemv
        # vs wmma_tiled by up to ~1.95e-3), so a quantized head is only M-invariant WITHIN an
        # arm band. Spec-decode verify runs M=K+1 while decode runs M=1 — straddling bands —
        # and verify logits that do not match decode logits silently destroy draft acceptance
        # (see layers/minv.py, which names spec-decode VERIFY as a protected pathway). bf16
        # keeps `_lm_head_linear` on dense_bf16_gemv, which is M-invariant by construction.
        # This matches the repo-wide stance that lm_head is never quantized (create_linear_
        # method's `quantized=False`; qwen3_5_moe and glm4_moe_lite both keep it full-precision).
        #
        # Handled before the remap because `lm_head.weight_scale` is not a key qwen3_5_remap
        # knows. A bf16 lm_head ships no scale, so _fp8_lm_head is False and nothing changes.
        if _fp8_lm_head and name.startswith("lm_head."):
            field = name.rsplit(".", 1)[1]
            if field not in ("weight", "weight_scale"):
                return
            lm_head_buf[field] = _shard_qwen3_5(
                name, f.get_tensor(name), tp_info.rank, tp_info.size, config
            )
            if len(lm_head_buf) < 2:
                return
            w = lm_head_buf.pop("weight").to(device)
            sc = lm_head_buf.pop("weight_scale").to(device).to(torch.float32)
            out = torch.empty(w.shape, dtype=torch.get_default_dtype(), device=device)
            # CHUNKED over vocab rows. A whole-tensor `w.to(f32) * sc` would allocate two
            # ~2.5 GiB fp32 temporaries for a 248k-row head; the caching allocator does not
            # return those to the driver, so they inflate the engine's `model_memory =
            # free_before - free_after` measurement and silently starve the KV pool (it sized
            # to ZERO pages). One chunk is ~160 MiB and is reused every iteration.
            for i in range(0, w.shape[0], 8192):
                blk = slice(i, i + 8192)
                out[blk] = (w[blk].to(torch.float32) * sc[blk]).to(out.dtype)
            del w, sc
            yield from emit("lm_head.weight", out)
            return
        plan = qwen3_5_remap(name, load_mtp=config.mtp_num_hidden_layers > 0)
        if plan is None:
            return
        # Shard at READ (on the checkpoint name), so the GDN concat / gate-up merge /
        # expert stack below all compose rank-local parts (Phase 4-1; no-op at TP=1).
        tens = override if override is not None else f.get_tensor(name)
        # .to(device) AFTER the shard: everything downstream (GDN concat, gate/up merge,
        # expert stack) then composes rank-local tensors already in VRAM, as before.
        raw = _shard_qwen3_5(name, tens, tp_info.rank, tp_info.size, config).to(device)
        if plan[0] == "direct":
            yield from emit(plan[1], raw)
            return
        _, merged, slot, n_slots, cat_dim = plan
        concat_buf.setdefault(merged, {})[slot] = raw
        if len(concat_buf[merged]) != n_slots:
            return
        parts = [concat_buf[merged][i] for i in range(n_slots)]
        del concat_buf[merged]
        yield from emit(merged, torch.cat(parts, dim=cat_dim))

    # NVFP4 bases and the fp8 lm_head flag are resolved across ALL shard files BEFORE streaming.
    # They used to be recomputed per file, which silently required a proj's weight_scale and
    # weight_global_scale to be CO-LOCATED in one shard: in the shard holding only weight_scale the
    # base was absent from the per-file set, so that tensor fell through to the generic remap and
    # was dropped while its partner waited in the fold buffer forever — every quantized proj
    # "incomplete" at the end of the load. The layout is repack luck, not a convention:
    # sakamakismile/Qwen3.8-27B-MTP-NVFP4 co-locates all 496 pairs (which is how the per-file set
    # passed validation), cyankiwi/Qwen3.6-27B-AWQ-BF16-NVFP4 splits all 256 across its 6 shards
    # (which is how it failed to boot). Metadata-only pass — safe_open reads headers, no tensors.
    nvfp4_bases: set = set()
    _fp8_lm_head = False
    for file in files:
        with safetensors.safe_open(file, framework="pt", device="cpu") as f:
            for k in f.keys():
                if k.endswith(".weight_global_scale"):
                    nvfp4_bases.add(k.rsplit(".", 1)[0])
                elif k == "lm_head.weight_scale":
                    _fp8_lm_head = True

    for file in tqdm(files, desc="Loading weights", disable=not tp_info.is_primary()):
        # Read on the HOST and move only the RANK-LOCAL shard to the GPU. Reading straight to device
        # materialised every FULL tensor in VRAM, sliced this rank's shard out of it and freed the
        # rest — so at TP=2 the peak was ~2x the resident weights and, far worse, it left the heap
        # badly FRAGMENTED. empty_cache() cannot recover that (it only returns segments with NO live
        # block), so the engine's free-memory delta billed 14.40 GiB for 11.67 GiB of real tensors
        # and the KV pool sized NEGATIVE. Slicing host-side keeps GPU peak == GPU resident.
        with safetensors.safe_open(file, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            for ckpt_name in keys:
                # NVFP4 SCALE RESOLUTION AT THE LEAF — before remap / GDN in_proj concat / gate-up
                # merge / expert stack, so every downstream fusion (each combining differently-scaled
                # matrices) composes without a special case. ROUTED EXPERTS keep the checkpoint's TWO
                # levels and emit TWO leaves (`.weight_scale` e4m3 byte-verbatim + `.weight_global`,
                # the per-output-channel f32 MULTIPLIER); everything else still folds to one fp16
                # per-group scale, because the DENSE e2m1 kernel hardcodes a `const __half*` scale
                # pointer. `nvfp4.nvfp4_leaf_scales` owns that decision — see `nvfp4_leaf_splits`.
                # weight_packed passes through unchanged (4-bit); input_global_scale (FP4 act calib)
                # is dropped — the e2m1 kernel quantizes acts to fp8.
                leaves: "list[Tuple[str, torch.Tensor | None]]" = [(ckpt_name, None)]
                if ckpt_name.rsplit(".", 1)[0] in nvfp4_bases:
                    if ckpt_name.endswith(".input_global_scale"):
                        continue
                    if ckpt_name.endswith((".weight_scale", ".weight_global_scale")):
                        base, field = ckpt_name.rsplit(".", 1)
                        buf = nvfp4_fold_buf.setdefault(base, {})
                        buf[field] = f.get_tensor(ckpt_name)
                        if len(buf) < 2:
                            continue
                        del nvfp4_fold_buf[base]
                        # `device=` is consumed by the FOLD arm only: the fold is fp32 arithmetic on
                        # an e4m3 input and torch has no CPU float8 math. The split arm is a dtype
                        # passthrough and stays host-side, like every other tensor read here.
                        leaves = list(
                            nvfp4.nvfp4_leaf_scales(
                                base,
                                buf["weight_scale"],
                                buf["weight_global_scale"],
                                global_field="weight_global_scale",
                                device=device,
                            )
                        )
                for name, override in leaves:
                    yield from leaf(f, name, override)

    assert not concat_buf, f"incomplete concat groups in checkpoint: {list(concat_buf.keys())}"
    assert not merge_buf, f"incomplete gate/up merges in checkpoint: {list(merge_buf.keys())}"
    assert not expert_buf, f"incomplete expert stacks in checkpoint: {expert_buf.pending}"
    assert not nvfp4_fold_buf, (
        f"incomplete NVFP4 scale/global pairs (a proj missing its weight_scale or weight_global_scale): "
        f"{list(nvfp4_fold_buf.keys())}"
    )
    assert not lm_head_buf, (
        f"fp8 lm_head missing its counterpart tensor (have {list(lm_head_buf.keys())}; "
        f"need both weight and weight_scale)"
    )


# ---- Qwen3.8-Flash-Next (`qwen4_exp`) weight-name remap — bring-up tranche 1a ----
#
# The checkpoint (`RadixArk/Qwen3.8-Flash-Next-NVFP4`, 296,475 tensors over 206 files) is a
# multimodal wrapper laid out like Qwen3.5's — `model.language_model.*` for the text decoder,
# `model.visual.*` for the vision tower, top-level `mtp.*` for the speculative head — but it is NOT
# the Qwen3.5 key set, so it gets its own remap:
#
#   * hyper-connections: `layers.<L>.{attn,mlp}_hyper_connection.*` (4 tensors each) and a top-level
#     `hyper_connection_mixer.*` (3 tensors, no block_inject_weight). The mixer REPLACES the final
#     norm — this checkpoint ships no `model.language_model.norm.weight` at all.
#   * PLE: `layers.1.ple.*`, including a 51.2 GB n-gram table split into 128 F8_E4M3 shards. That
#     table is NOT a model parameter: it is NVMe-resident and served by `weights/row_table.py`, so
#     every `ple_embedding.ngram_embedding.*` key (and the two head-metadata tensors `NgramHeads`
#     reads directly from the files) is SKIPPED here rather than loaded.
#   * QSA indexer: `layers.<L>.self_attn.indexer.*` on the 12 full-attention layers.
#   * routed experts are modelopt-NVFP4 and spell their leaves `.weight` / `.weight_scale` /
#     `.weight_scale_2` / `.input_scale`, where this repo's NVFP4 path expects `.weight_packed` /
#     `.weight_scale` / `.weight_global_scale`. The rename happens HERE, i.e. before any loader's
#     `nvfp4_bases` pre-pass (which keys on `.weight_global_scale`) — do it later and the two-level
#     scale fold silently no-ops.
#
# Everything shared with Qwen3.5 is reused unchanged: the GDN `in_proj_qkv`+`in_proj_z` ->
# `in_proj_qkvz` / `in_proj_b`+`in_proj_a` -> `in_proj_ba` concats, the `conv1d.weight` ->
# `conv1d_weight` flattening, and the MoE gate/up merge + expert stack (`_gate_up_merge`,
# `_get_expert_stack_info`) which the caller applies to this function's output.
#
# NOTHING IS DROPPED SILENTLY. Every checkpoint key resolves to a plan or to an explicit
# ("skip", reason) — there is no `None` return — and an unrecognised `model.language_model.*` key
# RAISES. `qwen4_exp_ignored_summary` turns the skips into a per-reason count a loader can log.

_QWEN4EXP_LM_PREFIX = "model.language_model."
_QWEN4EXP_SKIP_PREFIXES = ("model.visual.", "visual.")

# skip reason -> what it means, for the loader's ignore ledger.
QWEN4EXP_SKIP_REASONS: Dict[str, str] = {
    "vision": "model.visual.* — vision tower; this engine serves the text decoder only",
    "mtp-head": "mtp.* — the MTP speculative head is not implemented (bring-up plan T8.1)",
    "ple-ngram-table": (
        "ple_embedding.ngram_embedding.* + ngram head metadata — the 51.2 GB n-gram table stays "
        "NVMe-resident and is served by weights/row_table.py, never streamed into VRAM"
    ),
    "act-calibration": (
        "*.input_scale / *.input_global_scale — FP4 activation calibration. gfx1201 has no FP4 "
        "math, so the e2m1 kernel quantizes activations to fp8 and this calibration is unused "
        "(the served scheme is W4A8, not the declared W4A4)"
    ),
    "quant-metadata": "*.weight_shape — compressed-tensors bookkeeping, not a model parameter",
    "fp8-kv-scale": "*.k_scale / *.v_scale — read from the files by kvcache/fp8_scales.py",
    "beyond-decoder": (
        "layers.<n> with n >= num_hidden_layers — the model-bf16-* shards carry every layer the "
        "CHECKPOINT has, and a config that serves fewer (a layer-subset bring-up config) declares "
        "fewer. Same rule `load_weight` applies for every other family"
    ),
}

#: The reason key above, as a constant, because it is recorded in one place and rendered in another.
_Q4_BEYOND_DECODER = "beyond-decoder"

# rename-only leaves (checkpoint suffix -> native suffix). Both conv1ds are stored FLAT by the
# model (a plain Parameter, not an nn.Conv1d), matching the Qwen3.5 GDN convention.
_QWEN4EXP_RENAME = {
    ".linear_attn.conv1d.weight": ".linear_attn.conv1d_weight",
    ".ple.conv1d.weight": ".ple.conv1d_weight",
}
_Q4_QKVZ = (".linear_attn.in_proj_qkv.weight", ".linear_attn.in_proj_z.weight")
_Q4_BA = (".linear_attn.in_proj_b.weight", ".linear_attn.in_proj_a.weight")
_QWEN4EXP_CONCAT = {
    _Q4_QKVZ[0]: (".linear_attn.in_proj_qkvz.weight", _Q4_QKVZ, 0),
    _Q4_QKVZ[1]: (".linear_attn.in_proj_qkvz.weight", _Q4_QKVZ, 0),
    _Q4_BA[0]: (".linear_attn.in_proj_ba.weight", _Q4_BA, 0),
    _Q4_BA[1]: (".linear_attn.in_proj_ba.weight", _Q4_BA, 0),
}

# modelopt NVFP4 leaf spellings -> this repo's compressed-tensors NVFP4 spellings.
_MODELOPT_GLOBAL_SCALE = ".weight_scale_2"
_MODELOPT_ACT_CALIB = (".input_scale", ".input_global_scale")

# Every native key this remap is allowed to produce. A checkpoint key that lands anywhere else is a
# key we have not thought about, and it RAISES instead of being passed through — a pass-through
# would reach `load_state_dict` as an "unexpected key", which is loud, but a name that happens to
# collide with a real parameter would not be.
_QWEN4EXP_NATIVE_OK = tuple(
    re.compile(p)
    for p in (
        r"^lm_head\.weight$",
        r"^model\.embed_tokens\.weight$",
        r"^model\.hyper_connection_mixer\."
        r"(hc_norm|input_mix_weight_down|input_mix_weight_up)\.weight$",
        r"^model\.layers\.\d+\.(attn|mlp)_hyper_connection\."
        r"(hc_norm|input_mix_weight_down|input_mix_weight_up|block_inject_weight)\.weight$",
        r"^model\.layers\.\d+\.linear_attn\.(in_proj_qkvz|in_proj_ba|out_proj|norm)\.weight$",
        r"^model\.layers\.\d+\.linear_attn\.(conv1d_weight|A_log|dt_bias)$",
        r"^model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj|q_norm|k_norm)\.weight$",
        r"^model\.layers\.\d+\.self_attn\.indexer\."
        r"(index_qk_proj|q_layernorm|k_layernorm)\.weight$",
        r"^model\.layers\.\d+\.mlp\.(gate|shared_expert_gate)\.weight$",
        r"^model\.layers\.\d+\.mlp\.shared_expert\.(gate_proj|up_proj|down_proj)\.weight$",
        # `weight_global` is the NVFP4 split arm's second leaf: the per-output-channel f32 global
        # MULTIPLIER (see quant/nvfp4.py). `weight_global_scale` is the OLD repo-native spelling of
        # the raw per-TENSOR global, which the fold arm consumed and dropped; both are listed because
        # a mixed checkpoint can still hit the fold path on a non-expert module.
        r"^model\.layers\.\d+\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)\."
        r"(weight|weight_packed|weight_scale|weight_global|weight_global_scale)$",
        r"^model\.layers\.\d+\.ple\."
        r"(key_proj|value_proj|norm_key|norm_query|norm_conv)\.weight$",
        r"^model\.layers\.\d+\.ple\.conv1d_weight$",
        r"^model\.layers\.\d+\.ple\.ple_embedding\.layer_multipliers$",
    )
)


def _q4_check_native(ckpt_key: str, native: str) -> str:
    for pat in _QWEN4EXP_NATIVE_OK:
        if pat.match(native):
            return native
    raise ValueError(
        f"unrecognised qwen4_exp checkpoint key {ckpt_key!r} (would map to {native!r}). Add it to "
        f"_QWEN4EXP_NATIVE_OK (and to the model) or give it an explicit skip reason — a silent "
        f"pass-through is how a tensor goes missing."
    )


def qwen4_exp_nvfp4_modules(ckpt_names: "Collection[str]") -> FrozenSet[str]:
    """Module base names the checkpoint actually ships NVFP4-quantized, keyed STRUCTURALLY on the
    presence of a two-level global scale (`.weight_scale_2`) rather than on the config's ignore list.

    This is the same "the tensors are the fact, the config is a claim" rule
    `_load_qwen3_5_weight`'s `nvfp4_bases` pre-pass already uses, and it is what tells
    `qwen4_exp_remap` that a bare `.weight` on this module is a packed E2M1 blob rather than a bf16
    matrix. In `RadixArk/Qwen3.8-Flash-Next-NVFP4` that is exactly the 73,728 routed-expert
    projections; the shared expert, both gates, the GDN, self_attn, the hyper-connections, the PLE
    block and lm_head are all bf16.

    Takes CHECKPOINT names and returns NATIVE (de-wrapped, `model.language_model.` -> `model.`)
    module names — the namespace `qwen4_exp_remap` compares in, and the same namespace
    `QuantConfig.ckpt_quantized` is keyed in (see `ModelConfig.from_hf`'s `_native`). Returning the
    wrapped spelling instead is a silent no-op: every membership test misses and the whole MoE quietly
    builds full precision."""
    return frozenset(
        n[: -len(_MODELOPT_GLOBAL_SCALE)].replace(_QWEN4EXP_LM_PREFIX, "model.", 1)
        for n in ckpt_names
        if n.endswith(_MODELOPT_GLOBAL_SCALE)
    )


def qwen4_exp_remap(ckpt_key: str, *, nvfp4_modules: "Collection[str]" = ()):
    """Map a Qwen3.8-Flash-Next checkpoint key to a minisgl-native key plan. Pure (no tensors), so
    it is CPU-testable against the checkpoint index vs. the model's `state_dict()`.

    `nvfp4_modules`: module base names known to be NVFP4 (see `qwen4_exp_nvfp4_modules`). A bare
    `.weight` on one of those is the packed E2M1 blob and is renamed to `.weight_packed`; on any
    other module `.weight` is an ordinary bf16 matrix and is left alone. This has to be told to the
    function because the two are indistinguishable from the key string, and guessing either way is a
    silent-wrong: guess "packed" and a bf16 matrix lands in a uint8 buffer, guess "bf16" and the MoE
    silently builds full precision.

    Returns one of:
      ``("skip",   reason)``                          -> not a model parameter; `reason` is a key of
                                                         QWEN4EXP_SKIP_REASONS
      ``("direct", native_key)``                      -> rename only
      ``("concat", merged_key, slot, n_slots, dim)``  -> one member of an ordered concat group

    Never returns None, and raises on an unrecognised `model.language_model.*` key.
    """
    # Namespace first, so a skipped tensor is attributed to the reason a reader would expect
    # (`mtp.<...>.input_scale` is skipped because the MTP head is not built, not because of FP4
    # activation calibration).
    if ckpt_key.startswith("mtp."):
        # The MTP head is a different animal here (fused expert tensors, its own hyper-connections,
        # a 10240-wide pre-mixer seed) and is not implemented — ModelConfig.from_hf refuses
        # --spec-algorithm mtp for this architecture rather than letting it half-build.
        return ("skip", "mtp-head")
    if ckpt_key.startswith(_QWEN4EXP_SKIP_PREFIXES):
        return ("skip", "vision")
    if ckpt_key.endswith(".weight_shape"):
        return ("skip", "quant-metadata")
    if ckpt_key.endswith((".k_scale", ".v_scale")):
        return ("skip", "fp8-kv-scale")
    if ckpt_key.endswith(_MODELOPT_ACT_CALIB):
        return ("skip", "act-calibration")
    if ckpt_key == "lm_head.weight":
        return ("direct", "lm_head.weight")  # untied; top-level, no LM prefix
    if not ckpt_key.startswith(_QWEN4EXP_LM_PREFIX):
        raise ValueError(
            f"unexpected qwen4_exp checkpoint key (not under {_QWEN4EXP_LM_PREFIX!r}, "
            f"'mtp.' or 'lm_head.'): {ckpt_key}"
        )
    native = "model." + ckpt_key[len(_QWEN4EXP_LM_PREFIX) :]

    # The n-gram table and its head metadata are read straight off disk by weights/row_table.py.
    # `layer_multipliers` is NOT part of that — it is the hash multiplier and stays a model buffer.
    if ".ple.ple_embedding." in native and not native.endswith(".layer_multipliers"):
        return ("skip", "ple-ngram-table")

    for suffix, renamed in _QWEN4EXP_RENAME.items():
        if native.endswith(suffix):
            return ("direct", _q4_check_native(ckpt_key, native[: -len(suffix)] + renamed))
    for suffix, (merged_suffix, members, cat_dim) in _QWEN4EXP_CONCAT.items():
        if native.endswith(suffix):
            merged = _q4_check_native(ckpt_key, native[: -len(suffix)] + merged_suffix)
            return ("concat", merged, members.index(suffix), len(members), cat_dim)

    # modelopt NVFP4 leaf spellings -> the repo's.
    base, _, field = native.rpartition(".")
    if field == "weight_scale_2":
        native = base + ".weight_global_scale"
    elif field == "weight" and base in nvfp4_modules:
        native = base + ".weight_packed"
    return ("direct", _q4_check_native(ckpt_key, native))


def qwen4_exp_ignored_summary(
    ckpt_names: "Collection[str]", *, nvfp4_modules: "Collection[str]" = ()
) -> "Dict[str, int]":
    """Per-reason count of the checkpoint tensors `qwen4_exp_remap` deliberately does not load.

    A loader should log this AFTER streaming, so "we ignored 296,344 tensors" is a stated fact with
    a stated reason rather than an unnoticed hole. (Of this checkpoint's 296,475 tensors, the vast
    majority are the 128 n-gram shards' siblings and the vision tower.)"""
    counts: Dict[str, int] = {}
    for name in ckpt_names:
        plan = qwen4_exp_remap(name, nvfp4_modules=nvfp4_modules)
        if plan[0] == "skip":
            counts[plan[1]] = counts.get(plan[1], 0) + 1
    return counts


def log_qwen4_exp_ignored(counts: "Dict[str, int]", log) -> None:
    """Emit the ignore ledger. `log` is a callable (e.g. `logger.info_rank0`)."""
    if not counts:
        return
    total = sum(counts.values())
    log(f"qwen4_exp loader ignored {total} checkpoint tensors, by reason:")
    for reason, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        log(f"  {n:>7}  {reason}: {QWEN4EXP_SKIP_REASONS.get(reason, '?')}")


#: The 51.2 GB NVMe-resident n-gram table shards. They are named in the checkpoint index but are NOT
#: model parameters (`weights/row_table.py` mmaps them directly), which is why they are normally kept
#: in a separate directory. Skipped by FILE name as well as by key, so a deployment that does
#: colocate them does not spend 51 GB of read bandwidth proving they are skipped.
_QWEN4EXP_PLE_FILE_MARK = "model-plefp8-"

#: `layer-<LLLLL>-experts-<lo>-<hi>.safetensors` — the per-layer, per-expert-range routed-expert
#: shards. This naming IS the chunk boundary Stage B streams on, so the pattern lives next to the
#: loader that consumes it rather than in the driver.
_QWEN4EXP_EXPERT_FILE = re.compile(r"^layer-(?P<layer>\d+)-experts-(?P<lo>\d+)-(?P<hi>\d+)\.safetensors$")


def qwen4_exp_shard_files(model_folder: str) -> "list[str]":
    """Every non-PLE safetensors shard of a qwen4_exp checkpoint, sorted. One definition, because the
    n-gram-table exclusion is a correctness rule (51.2 GB of read bandwidth spent proving those keys
    are skipped) and both the whole-checkpoint loader and the chunked one must apply it."""
    files = sorted(
        f
        for f in glob.glob(f"{model_folder}/*.safetensors")
        if _QWEN4EXP_PLE_FILE_MARK not in os.path.basename(f)
    )
    if not files:
        raise FileNotFoundError(f"no non-PLE *.safetensors under {model_folder}")
    return files


def qwen4_exp_expert_file_layer(path: str) -> "int | None":
    """The decoder-layer index a routed-expert shard belongs to, or None for a non-expert shard."""
    m = _QWEN4EXP_EXPERT_FILE.match(os.path.basename(path))
    return None if m is None else int(m.group("layer"))


def qwen4_exp_chunk_files(model_folder: str, num_layers: int) -> "tuple[list[str], dict[int, list[str]]]":
    """Split the checkpoint into `(body_files, {layer_index: expert_files})` — the STAGE B chunks.

    The 84 GB checkpoint cannot be materialized as one `state_dict` (70.31 GiB of it is routed
    experts, on a 15.92 GiB card), and this layout is what makes chunking natural rather than
    surgical: every routed expert of layer L, and nothing else, lives in exactly four
    `layer-LLLLL-experts-*.safetensors` files, and the entire non-expert body lives in the
    `model-bf16-*` shards. So a chunk is a FILE SET, not a key filter — no key has to be classified,
    and a file that matches neither pattern lands in the body chunk where the ordinary remap decides
    its fate. That is deliberately the fail-safe direction: an unrecognised shard is loaded and
    accounted for, never silently dropped.

    Raises when a layer in `range(num_layers)` has no expert shards at all. A missing layer would
    otherwise surface as `_load_qwen4_exp_weight`'s "incomplete expert stacks" assert at the END of a
    multi-minute load, or — far worse, if the model happens to have fewer layers than the checkpoint
    — as a silently truncated model.
    """
    body: "list[str]" = []
    per_layer: "dict[int, list[str]]" = {}
    for path in qwen4_exp_shard_files(model_folder):
        lid = qwen4_exp_expert_file_layer(path)
        if lid is None:
            body.append(path)
        else:
            per_layer.setdefault(lid, []).append(path)
    for files in per_layer.values():
        files.sort()
    missing = [i for i in range(num_layers) if not per_layer.get(i)]
    if missing:
        raise FileNotFoundError(
            f"qwen4_exp chunked load: {model_folder} has no `layer-LLLLL-experts-*.safetensors` "
            f"shards for decoder layers {missing[:8]}{'...' if len(missing) > 8 else ''} "
            f"(model declares {num_layers} layers)."
        )
    return body, per_layer


def qwen4_exp_nvfp4_prepass(files: "Sequence[str]") -> "tuple[set[str], FrozenSet[str]]":
    """`(nvfp4_ckpt_bases, nvfp4_modules)` read from the shard HEADERS only — no tensor is read.

    Hoisted out of `_load_qwen4_exp_weight` because the chunked loader must run it ONCE over the
    whole checkpoint and then reuse the answer for every chunk. Re-deriving it per chunk would be
    both wasteful (196 header opens x 49 chunks) and WRONG in the general case: the rule is
    structural ("this module ships a `.weight_scale_2`"), and a chunk that happens to contain a
    module's `weight_scale` but not its global would classify that module as non-NVFP4 and pass its
    packed E2M1 blob through as a bf16 matrix. On this checkpoint the pair is always co-located, but
    that is a property of one file layout, not of the format.
    """
    from minisgl.weights.boot_timeline import timeline as _bt_timeline

    _tl = _bt_timeline()
    _t0 = time.perf_counter()
    ckpt_names: "set[str]" = set()
    for file in files:
        _tl.note_file(file, "header")
        with safetensors.safe_open(file, framework="pt", device="cpu") as f:
            ckpt_names.update(f.keys())
    _tl.tick("ckpt.nvfp4_prepass", time.perf_counter() - _t0)
    bases = {
        n[: -len(_MODELOPT_GLOBAL_SCALE)] for n in ckpt_names if n.endswith(_MODELOPT_GLOBAL_SCALE)
    }
    return bases, qwen4_exp_nvfp4_modules(ckpt_names)


def _load_qwen4_exp_weight(
    model_folder: str,
    device: torch.device,
    config,
    *,
    files: "Sequence[str] | None" = None,
    nvfp4_sets: "tuple[set[str], FrozenSet[str]] | None" = None,
) -> Iterator[Tuple[str, torch.Tensor]]:
    """Streaming loader for `RadixArk/Qwen3.8-Flash-Next-NVFP4` (`qwen4_exp`). TP=1 only.

    Same skeleton as `_load_qwen3_5_weight`, deliberately: the two share the GDN in_proj concat, the
    MoE gate/up merge, the per-expert stack over E, and the NVFP4 leaf fold. Everything that differs
    lives in `qwen4_exp_remap` (hyper-connections, PLE, the QSA indexer, modelopt's spelling of
    NVFP4) except three things worth naming here:

      1. **The fold keys on `.weight_scale_2`, the CHECKPOINT's spelling** — not on the repo-native
         `.weight_global_scale` the qwen3_5 loader searches for, which here would match nothing and
         silently leave every routed expert scaled by its block scale alone (a ~2^k error per group,
         i.e. fluent garbage). It happens at the LEAF, before the merge and the stack, so the
         per-tensor global is absorbed into the per-group scale and no scalar survives a concat.
      2. **The n-gram table is never read**, by key AND by file name.
      3. **The ignore ledger is emitted** per reason after the stream: ~296k tensors in, ~1.1k
         parameters out, and the difference is a stated fact with a stated reason rather than a hole.

    TP > 1 shards through `_shard_qwen4_exp`, applied at READ on the checkpoint name so the GDN
    concat, the gate/up merge and the per-expert stack below all compose rank-local parts. It was
    TP=1-only until 2026-09-04 and the refusal was correct while it stood: the fall-through
    behaviour of the generic sharders is "replicate", which loads the whole 84 GB on every rank and
    then fails somewhere else entirely.
    """
    tp_info = get_tp_info()
    # `files=None` is the whole checkpoint (the one-shot path). Stage B passes ONE CHUNK's shards and
    # the checkpoint-wide NVFP4 pre-pass result, so the fold decision is identical to the one-shot
    # load's — see `qwen4_exp_nvfp4_prepass` for why the pre-pass may not be re-derived per chunk.
    files = qwen4_exp_shard_files(model_folder) if files is None else sorted(files)
    if not files:
        raise FileNotFoundError(f"no shards to load under {model_folder}")
    num_layers = int(config.num_layers)

    # Which modules does the checkpoint ACTUALLY ship NVFP4? Keyed STRUCTURALLY on the presence of a
    # two-level global scale, across ALL shards, before streaming — the same rule and the same reason
    # as the qwen3_5 loader's `nvfp4_bases` pre-pass: a module's `weight_scale` and its global
    # routinely land in DIFFERENT shards, so a per-file set drops one half of the pair and the fold
    # never completes. Two spellings of the same set are needed: CHECKPOINT bases to drive the fold,
    # NATIVE bases (`model.language_model.` -> `model.`) to tell `qwen4_exp_remap` that a bare
    # `.weight` on that module is a packed E2M1 blob and not a bf16 matrix.
    nvfp4_ckpt_bases, nvfp4_modules = (
        qwen4_exp_nvfp4_prepass(files) if nvfp4_sets is None else nvfp4_sets
    )
    logger.info_rank0(
        f"qwen4_exp: {len(files)} shards, {len(nvfp4_modules)} NVFP4 "
        f"modules (structural: they ship a '{_MODELOPT_GLOBAL_SCALE}')"
    )

    concat_buf: Dict[str, Dict[int, torch.Tensor]] = {}  # GDN in_proj qkv+z / b+a
    merge_buf: Dict[str, Dict[str, torch.Tensor]] = {}  # MoE gate/up -> gate_up
    expert_buf = _ExpertStacker()  # per-expert -> filled into a preallocated [E, ...]
    fold_buf: Dict[str, Dict[str, torch.Tensor]] = {}  # NVFP4 weight_scale + weight_scale_2
    skips: Dict[str, int] = {}

    def emit(native_key: str, tensor: torch.Tensor) -> Iterator[Tuple[str, torch.Tensor]]:
        from minisgl.weights.boot_timeline import tick as _tk

        if (mm := _gate_up_merge(native_key)) is not None:
            merged_key, slot = mm
            merge_buf.setdefault(merged_key, {})[slot] = tensor
            if len(merge_buf[merged_key]) != 2:
                return
            parts = [merge_buf[merged_key][s] for s in ("gate", "up")]
            del merge_buf[merged_key]
            # NVFP4 packs along the INPUT K (weight_packed [N, K//2], folded weight_scale [N, K//16])
            # and the gate/up merge concatenates the OUTPUT N — so dim 0 for every leaf here, unlike
            # AWQ's K-major qweight/qzeros/scales, which merge on dim 1.
            _t = time.perf_counter()
            native_key, tensor = merged_key, torch.cat(parts, dim=0)
            _tk("ckpt.gate_up_cat", time.perf_counter() - _t)
        if (einfo := _get_expert_stack_info(native_key)) is not None:
            packed_key, idx = einfo
            _t = time.perf_counter()
            stacked = expert_buf.add(packed_key, idx, tensor, config.num_experts)
            _tk("ckpt.expert_stack_add", time.perf_counter() - _t)
            if stacked is None:
                return
            yield packed_key, stacked
        else:
            yield native_key, tensor

    # BOOT ATTRIBUTION. These are bare dict adds against a `perf_counter()` delta (see
    # `weights/boot_timeline.py`): this loop runs ~2.2e5 times on a 48-layer boot, so a context
    # manager or a formatted log per tensor would itself be a measurable term. The buckets partition
    # the per-tensor pipeline — header read, tensor read, NVFP4 scale fold, remap, TP shard, H2D,
    # concat/merge, expert stack — which is the split that says where 105-174 MB/s goes against a
    # 4.9 GB/s drive.
    from minisgl.weights.boot_timeline import count as _bt_count
    from minisgl.weights.boot_timeline import tick as _bt_tick
    from minisgl.weights.boot_timeline import timeline as _bt_timeline

    # THE READ. `ckpt_read.safe_open` is a routing POLICY, not a second reader: expert shards
    # (<= WHOLE_FILE_CAP — 64.8 GiB of this checkpoint's 72.6) are read with one O_DIRECT `preadv`
    # into ONE process-wide REUSED buffer, and the four 3.4-10.0 GiB `model-bf16-*` body shards stay
    # on safetensors' mmap, byte-for-byte the reader they always had.
    #
    # MEASURED [BOOT-2026-09-05], 48-layer TP=2, FOUR boots (n=2 per arm), two worktrees, strictly
    # sequential, one job on the box (docs/measurements/BOOT_DEFECT.md): boot 506.6 -> 300.1 s
    # (1.69x), weight_load 355.5 -> 140.6 s, ckpt.h2d 243.5 -> 33.2 s, ckpt.shard 42.2 -> 8.1 s,
    # major faults 56-60 M -> 24 M. Identical weights, proven byte-exactly: blake2b over 1990
    # tensors / 70.44 GiB per rank read through the device pointer, same digest on all four boots.
    #
    # IT IS THE READ. The single-leg r3 verdict claimed otherwise ("the two largest wins are not
    # reads: post_load 106.1 -> 7.0, graph_capture 84.1 -> 33.6") and n=2 refutes it: post_load is
    # 6.67 and 6.72 s on BOTH before legs, and graph_capture does not improve (68.79/53.62 before vs
    # 68.65/70.45 after). What actually happens is that `ckpt.h2d` is a COPY whose mmap'd source
    # pages used to be faulted in DURING the copy, so the read was billed to it; O_DIRECT moves the
    # same work into `ckpt.safe_open` (0.5 -> 19.0 s) and the faults disappear.
    #
    # On this pool, one cold 337.7 MiB shard per leg: mmap 4 KiB walk 621.7-627.0 MiB/s costing
    # +0.33 GiB of page cache AND +0.33 GiB of ARC, O_DIRECT into the reused buffer 5122.9 MiB/s
    # costing NOTHING.
    #
    # REAL COSTS, not netted out: `ckpt.nvfp4_prepass` 2.2 -> 21.1 s and `ct_sign_verify` 0.1 ->
    # 8.4 s. Both still read through mmap and were cheap only while something else left the ARC warm.
    #
    # WHAT ROUND 2'S 3.5x LOSS TAUGHT (docs/measurements/BOOT_TIMELINE_2026-09-06/r2/): a FRESH
    # buffer per shard is NOT this change. It made the read 5x faster and the boot 3.5x slower,
    # because 64.8 GiB of anonymous churn is reclaimed by COMPRESSION where file-backed pages are
    # reclaimed by being dropped. REUSE is what makes O_DIRECT viable — and reuse is a correctness
    # hazard, because `get_tensor` hands out zero-copy views of the shared buffer, so a view that
    # outlives its shard would read the NEXT shard's bytes under this shard's name: an expert
    # dequantized against another expert's scale, which is plausible text and no crash. The two
    # halves of the contract are marked below, and `_SharedReadBuffer.acquire` proves the first one
    # mechanically at every refill rather than trusting this comment.
    _bt_tl = _bt_timeline()
    for file in tqdm(files, desc="Loading weights", disable=not tp_info.is_primary()):
        # CONTRACT, HALF 1 — no host view of the PREVIOUS shard may be alive when this one is read.
        # These are the loop's only host-side tensor locals (`raw`, `_cat`, and everything in
        # `concat_buf`/`merge_buf`/`expert_buf` is already on the device), and a generator frame
        # keeps its locals bound across the `for file` boundary, so without this the last tensor of
        # the previous shard is still exported when the next `acquire()` runs.
        _sc = tens = sharded = leaves = override = None
        _t = time.perf_counter()
        _fh = ckpt_read.safe_open(file, framework="pt", device="cpu")
        _bt_tick("ckpt.safe_open", time.perf_counter() - _t)
        _bt_count("ckpt.file_bytes", os.path.getsize(file))
        _bt_count("ckpt.files_opened")
        _bt_tl.note_file(file, "tensors")
        with _fh as f:
            for ckpt_name in list(f.keys()):
                # ONE checkpoint key can produce TWO leaves. An NVFP4 routed expert now keeps its
                # scale in the checkpoint's own TWO levels — the e4m3 block scale byte-verbatim plus
                # a per-output-channel f32 global — so the pair `(weight_scale, weight_scale_2)`
                # resolves to `(.weight_scale, .weight_global)` and BOTH ride the identical
                # remap -> shard -> GDN concat -> gate|up merge -> expert stack path below. Nothing
                # downstream special-cases the global; that is what the N-vector shape buys.
                #
                # `global_field` is modelopt's `weight_scale_2`, the RECIPROCAL of
                # compressed-tensors' `weight_global_scale`: a MULTIPLIER, not a divisor. The fold
                # call this replaced originally passed no convention and inherited the
                # compressed-tensors divide, which produced per-group scales up to 4.2e6, overflowed
                # the kernel's fp16 scale to inf, and made every logit NaN from the first MoE block
                # onward. `nvfp4_global_multiplier` now normalises direction ONCE, host-side, so the
                # kernel only ever multiplies.
                leaves: "list[Tuple[str, torch.Tensor | None]]" = [(ckpt_name, None)]
                base, _, field = ckpt_name.rpartition(".")
                if base in nvfp4_ckpt_bases and field in ("weight_scale", "weight_scale_2"):
                    buf = fold_buf.setdefault(base, {})
                    _t = time.perf_counter()
                    _sc = f.get_tensor(ckpt_name)
                    _bt_tick("ckpt.get_tensor", time.perf_counter() - _t)
                    _bt_count("ckpt.get_tensor_calls")
                    _bt_count("ckpt.get_tensor_bytes", _sc.numel() * _sc.element_size())
                    # CONTRACT, HALF 2 — `fold_buf` is the ONE place a `get_tensor` result is
                    # retained past the call that produced it: it parks a `weight_scale` until its
                    # `weight_scale_2` arrives. On this checkpoint the pair is always co-located in
                    # one shard, but that is a property of one repack (see `qwen4_exp_nvfp4_prepass`,
                    # which exists because another Qwen NVFP4 repack splits every pair across
                    # shards), so the retention is made SAFE rather than assumed-not-to-happen: the
                    # clone costs ~1/8 of the packed bytes and buys a fold that cannot read the next
                    # shard's buffer.
                    buf[field] = _sc.clone()
                    _sc = None
                    if len(buf) < 2:
                        continue
                    del fold_buf[base]
                    _t = time.perf_counter()
                    leaves = list(
                        nvfp4.nvfp4_leaf_scales(
                            base,
                            buf["weight_scale"],
                            buf["weight_scale_2"],
                            global_field="weight_scale_2",
                            device=device,
                        )
                    )
                    _bt_tick("ckpt.nvfp4_leaf_scales", time.perf_counter() - _t)
                for name, override in leaves:
                    _t = time.perf_counter()
                    plan = qwen4_exp_remap(name, nvfp4_modules=nvfp4_modules)
                    _bt_tick("ckpt.remap", time.perf_counter() - _t)
                    if plan[0] == "skip":
                        skips[plan[1]] = skips.get(plan[1], 0) + 1
                        continue
                    # `layers.<n>` with n >= num_layers is not this model's. The `model-bf16-*`
                    # shards carry every layer the CHECKPOINT has, so a config that serves fewer — a
                    # layer subset, the only way this checkpoint is bootable on one card — is handed
                    # ~44 layers' worth of body tensors it has no home for, and `load_state_dict`
                    # refuses with "Unexpected keys". Every other family gets this filter from
                    # `load_weight`; the qwen4_exp remap did not carry it, so it lived in
                    # `qwen4_exp_chunked_source`'s `stream` wrapper — i.e. the CHUNKED path could
                    # load a subset and the ONE-SHOT path could not, for the same checkpoint and the
                    # same config. It belongs here, once, on the code path both share.
                    #
                    # AFTER the remap's own skip branch, so the ignore ledger still attributes a
                    # vision or MTP tensor to the reason a reader would expect; BEFORE
                    # `f.get_tensor`, so the dropped layers are never read and never touch the device.
                    if _is_beyond_decoder(name, num_layers):
                        skips[_Q4_BEYOND_DECODER] = skips.get(_Q4_BEYOND_DECODER, 0) + 1
                        continue
                    # Shard at READ (on the CHECKPOINT name), so the GDN in_proj concat, the gate/up
                    # merge and the per-expert stack below all compose rank-local parts.
                    # `.to(device)` comes AFTER the shard: everything downstream then composes
                    # rank-local tensors already in VRAM, and a rank never materializes the
                    # full-width tensor on its card. An `override` (an NVFP4 leaf scale) is already
                    # full-width — both levels are read whole and the shard applies to the result,
                    # so no scale is folded/split twice and none from a partial.
                    if override is not None:
                        tens = override
                    else:
                        _t = time.perf_counter()
                        tens = f.get_tensor(name)
                        _bt_tick("ckpt.get_tensor", time.perf_counter() - _t)
                        _bt_count("ckpt.get_tensor_calls")
                        _bt_count("ckpt.get_tensor_bytes", tens.numel() * tens.element_size())
                    _t = time.perf_counter()
                    sharded = _shard_qwen4_exp(name, tens, tp_info.rank, tp_info.size, config)
                    _t2 = time.perf_counter()
                    _bt_tick("ckpt.shard", _t2 - _t)
                    # THE H2D. One `.to(device)` per LEAF — ~2.2e5 of them on a 48-layer boot, each a
                    # pageable copy of a few hundred KiB. Timed on its own because "the checkpoint is
                    # read slowly" and "the checkpoint is copied to the card in 200k pieces" are
                    # different defects with different fixes, and the byte counter next to it prices
                    # the transfer against PCIe.
                    raw = sharded.to(device)
                    _bt_tick("ckpt.h2d", time.perf_counter() - _t2)
                    _bt_count("ckpt.h2d_calls")
                    _bt_count("ckpt.h2d_bytes", raw.numel() * raw.element_size())
                    if plan[0] == "direct":
                        yield from emit(plan[1], raw)
                        continue
                    _, merged, slot, n_slots, cat_dim = plan
                    concat_buf.setdefault(merged, {})[slot] = raw
                    if len(concat_buf[merged]) != n_slots:
                        continue
                    parts = [concat_buf[merged][i] for i in range(n_slots)]
                    del concat_buf[merged]
                    _t = time.perf_counter()
                    _cat = torch.cat(parts, dim=cat_dim)
                    _bt_tick("ckpt.gdn_concat", time.perf_counter() - _t)
                    yield from emit(merged, _cat)

    # Only shards that were actually opened contribute to this ledger; when the n-gram table lives in
    # its own directory (the normal deployment) its 129 keys are never seen at all, which is why the
    # ledger is printed rather than asserted against a fixed count.
    log_qwen4_exp_ignored(skips, logger.info_rank0)
    assert not concat_buf, f"incomplete GDN in_proj concat groups: {sorted(concat_buf)}"
    assert not merge_buf, f"incomplete gate/up merges: {sorted(merge_buf)}"
    assert not expert_buf, (
        f"incomplete expert stacks ({len(expert_buf)} groups, e.g. {expert_buf.pending[:3]}): a "
        f"layer's expert shards are missing from {model_folder}"
    )
    assert not fold_buf, (
        f"incomplete NVFP4 scale pairs (a module with a '{_MODELOPT_GLOBAL_SCALE}' but no "
        f"weight_scale, or the reverse): {sorted(fold_buf)[:3]}"
    )


class Qwen4ExpExpertRowSource:
    """`weights.stream_tier.ExpertRowSource` for qwen4_exp's per-expert NVFP4 shards.

    The checkpoint stores each expert as its own set of leaves inside one of four
    `layer-{L:05d}-experts-{lo:04d}-{lo+127:04d}.safetensors` files, so reading 10 of 512 experts is
    a `safe_open` plus six `get_tensor` calls, not a 1.465 GiB stack. That per-expert granularity is
    the entire reason the stream tier is viable on this checkpoint.

    THE LEAF ARITHMETIC IS `_load_qwen4_exp_weight`'S, NOT A SECOND COPY OF IT — same order, same
    convention: fold the modelopt `weight_scale_2` global into the e4m3 block scale AT THE LEAF, then
    merge gate/up on dim 0 (NVFP4 packs along the input axis K, so the gate/up concat is the OUTPUT
    dim for every leaf), then hand the pair to `nvfp4.convert_nvfp4_moe` — which is
    `_GroupedNvFp4Experts.post_load`'s own conversion, run over `len(ids)` experts instead of E.
    `convert_nvfp4_moe` is per-expert internally, so a subset stack is bit-identical to the same
    experts inside a full one. Getting the fold direction or the merge axis wrong here is silent
    (right shapes, plausible magnitudes), which is why nothing about either is re-derived.
    """

    def __init__(self, model_path: str, device: torch.device) -> None:
        self.model_folder = download_hf_weight(model_path)
        self.device = device
        self.bytes_read = 0
        self._handles: "Dict[Tuple[int, int], Any]" = {}

    def _handle(self, layer: int, lo: int):
        key = (layer, lo)
        h = self._handles.get(key)
        if h is None:
            import safetensors

            path = (
                f"{self.model_folder}/layer-{layer:05d}-experts-{lo:04d}-{lo + 127:04d}.safetensors"
            )
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"layer {layer}, experts {lo}..{lo + 127}: missing {path}. The layer->shard "
                    f"mapping is the one place a streamed run goes silently wrong, so it is "
                    f"asserted rather than skipped."
                )
            h = safetensors.safe_open(path, framework="pt", device="cpu")
            h.__enter__()
            self._handles[key] = h
        return h

    def _leaf(self, layer: int, eid: int, projs: "Tuple[str, ...]"):
        """(packed, block_scale, global_vec) for one expert, gate|up already concatenated on dim 0.

        The gate|up concat on the GLOBAL is the same `torch.cat(dim=0)` the one-shot loader's
        `emit` does, on the same axis, for the same reason — the global is a per-output-channel
        vector, so merging two differently-scaled matrices just concatenates their channel ranges.
        """
        from minisgl.quant import nvfp4

        f = self._handle(layer, (eid // 128) * 128)
        pre = f"model.language_model.layers.{layer}.mlp.experts.{eid}."
        packed, scales, globals_ = [], [], []
        for p in projs:
            w = f.get_tensor(pre + p + ".weight")
            s = f.get_tensor(pre + p + ".weight_scale")
            # `_MODELOPT_GLOBAL_SCALE` already carries its leading dot (it is a SUFFIX matched
            # against full tensor names elsewhere in this file), while the nvfp4 convention table is
            # keyed on the bare field name. Two spellings of one thing, so both are derived from the
            # constant rather than written out.
            g = f.get_tensor(pre + p + _MODELOPT_GLOBAL_SCALE)
            self.bytes_read += w.numel() * w.element_size() + s.numel() * s.element_size() + 4
            block, gvec = nvfp4.split_nvfp4_scale(
                s, g, global_field=_MODELOPT_GLOBAL_SCALE.lstrip(".")
            )
            packed.append(w.to(self.device, non_blocking=True))
            scales.append(block.to(self.device, non_blocking=True))
            globals_.append(gvec.to(self.device, non_blocking=True))
        if len(projs) == 1:
            return packed[0], scales[0], globals_[0]
        return (
            torch.cat(packed, dim=0),
            torch.cat(scales, dim=0),
            torch.cat(globals_, dim=0),
        )

    def gather(self, layer: int, expert_ids) -> "Dict[str, Dict[str, torch.Tensor]]":
        from minisgl.quant import nvfp4

        p13, s13, g13, p2, s2, g2 = [], [], [], [], [], []
        for e in expert_ids:
            a, b, c = self._leaf(layer, int(e), ("gate_proj", "up_proj"))
            d, x, y = self._leaf(layer, int(e), ("down_proj",))
            p13.append(a)
            s13.append(b)
            g13.append(c)
            p2.append(d)
            s2.append(x)
            g2.append(y)
        out: "Dict[str, Dict[str, torch.Tensor]]" = {}
        for attr, packed, scale, glob in (
            ("gate_up_proj", p13, s13, g13),
            ("down_proj", p2, s2, g2),
        ):
            conv = nvfp4.convert_nvfp4_moe(torch.stack(packed), torch.stack(scale))
            out[attr] = {
                "_w_op": conv["w_packed"],
                # GROUP-MAJOR, exactly as `_GroupedNvFp4Experts.post_load` leaves it.
                "_scales_op": conv["scales"].transpose(1, 2).contiguous(),
                # (E, N) f32 global bitcast to int32 — the `w_zeros` pointer slot, exactly as
                # `post_load` leaves it. Bit-for-bit the same transform, not a second recipe.
                "_global_op": torch.stack(glob).contiguous().view(torch.int32),
            }
        return out

    def close(self) -> None:
        for h in self._handles.values():
            try:
                h.__exit__(None, None, None)
            except Exception:
                pass
        self._handles.clear()


def expert_row_source(model_path: str, device: torch.device, spec_algorithm: str = "mtp"):
    """`weights.stream_tier.ExpertRowSource` for this family, or None if it has none.

    The third dispatch twin of `load_weight` / `chunked_weight_source`. A family whose checkpoint has
    no per-expert granularity returns None and the stream tier is simply unavailable — a capacity
    limit, not a degradation: every other model this repo serves fits {device, pinned host}.
    """
    from .config import ModelConfig

    config = ModelConfig.from_hf(
        cached_load_hf_config(model_path),
        spec_algorithm=spec_algorithm,
        ckpt_tensor_names=checkpoint_tensor_names(model_path),
    )
    if config.is_qwen4_exp:
        return Qwen4ExpExpertRowSource(model_path, device)
    return None


def qwen4_exp_chunked_source(
    model_path: str, device: torch.device, config
) -> "Tuple[List, Callable]":
    """`(chunks, stream)` for `weights.stage_b.ChunkedWeightLoader` — the STAGE B entry point.

    One `LoadChunk` for the whole non-expert body, then one per decoder layer carrying that layer's
    512 routed experts and finalizing `model.layers.{L}.mlp.experts`. Body first, deliberately: the
    body is resident for the whole serve whatever the plan says, so loading it first means every
    per-layer chunk is measured against the real steady-state device occupancy rather than against an
    empty card — a chunk that only fits because the body has not arrived yet is a boot that fails at
    layer 47.

    `stream` re-enters `_load_qwen4_exp_weight` with that chunk's shards and the checkpoint-wide
    NVFP4 pre-pass, so the fold, the gate/up merge, the per-expert stack over E and the ignore ledger
    are the SAME code the one-shot loader runs — Stage B changes when tensors are read, never how
    they are interpreted. In particular the expert stack is still asserted complete per layer: four
    shards x 128 experts, and a missing shard is `_load_qwen4_exp_weight`'s own "incomplete expert
    stacks" assert, raised at that layer instead of at the end of an 84 GB read.
    """
    from minisgl.weights.stage_b import LoadChunk

    model_folder = download_hf_weight(model_path)
    num_layers = int(config.num_layers)
    body_files, per_layer = qwen4_exp_chunk_files(model_folder, num_layers)
    # ONE pre-pass over the whole checkpoint's headers, reused by every chunk. See
    # `qwen4_exp_nvfp4_prepass` for why it may not be re-derived per chunk.
    nvfp4_sets = qwen4_exp_nvfp4_prepass(body_files + [f for fs in per_layer.values() for f in fs])

    chunks = [LoadChunk(name="body", files=tuple(body_files))]
    for lid in range(num_layers):
        chunks.append(
            LoadChunk(
                name=f"layer-{lid:05d}-experts",
                files=tuple(per_layer[lid]),
                finalize_paths=(f"model.layers.{lid}.mlp.experts",),
            )
        )

    def stream(chunk) -> Iterator[Tuple[str, torch.Tensor]]:
        # No key filtering here. `_load_qwen4_exp_weight` drops `layers.<n>` with n >= num_layers
        # itself, before it reads the tensor — this wrapper used to do it after, which meant the
        # CHUNKED path could load a layer-subset config and the ONE-SHOT path (the engine's) could
        # not. `ChunkedWeightLoader` stays strict: an unhomed key is still an error, because for a
        # key that is not beyond the decoder it means the remap dropped something real.
        yield from _load_qwen4_exp_weight(
            model_folder, device, config, files=chunk.files, nvfp4_sets=nvfp4_sets
        )

    return chunks, stream


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
    r"(?P<idx>\d+)\.(?P<fc>linear_fc1|linear_fc2)\.(?P<field>weight|weight_scale|weight_packed)$"
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


# HF-format ZAYA (Zyphra's transformers release): 40 FUSED layers (attn+MoE per layer) with HF names,
# vs our Megatron layout of 80 ALTERNATING layers. Map HF key -> Megatron-checkpoint-style key (then
# _zaya_remap finishes the CCA/router rename). L -> 2L (attn) / 2L+1 (MoE). Experts + expert-scales
# return None (handled by the stacked EP path in _load_zaya_weight). Full derivation + shape checks:
# docs/zaya-port/HF_FORMAT_LOADER.md.
_HF_ATTN_QKV = {
    "q_proj.weight": "linear_q.weight", "k_proj.weight": "linear_k.weight",
    "v_proj_current.weight": "val_proj1.weight", "v_proj_delayed.weight": "val_proj2.weight",
    "conv_qk_depthwise.weight": "conv_qk.0.weight", "conv_qk_depthwise.bias": "conv_qk.0.bias",
    "conv_qk_grouped.weight": "conv_qk.1.weight", "conv_qk_grouped.bias": "conv_qk.1.bias",
}
_HF_ROUTER = {
    "down_proj.weight": "down_proj.weight", "down_proj.bias": "down_proj.bias",
    "router_mlp.norm.weight": "rmsnorm_eda.weight",
    "router_mlp.fc1.weight": "router_mlp.0.weight", "router_mlp.fc1.bias": "router_mlp.0.bias",
    "router_mlp.fc2.weight": "router_mlp.2.weight", "router_mlp.fc2.bias": "router_mlp.2.bias",
    "router_mlp.out_proj.weight": "router_mlp.4.weight",
    "router_states_scale": "router_states_scale", "balancing_biases": "balancing_biases",
}


def _zaya_remap_hf(key: str, n_blocks: int) -> str | None:
    """HF-format ZAYA key -> Megatron-checkpoint-style key (None => skip; experts handled elsewhere)."""
    if key in ("model.embed_tokens.weight",):
        return key
    if key == "model.norm.weight":
        return "model.final_norm.weight"
    if key.startswith("model.input_hidden_states_"):  # first attn layer's hidden-only res_scale
        return f"model.layers.0.res_scale.hidden_states_{key[len('model.input_hidden_states_'):]}"
    m = re.match(r"^model\.layers\.(\d+)\.(.+)$", key)
    if m is None:
        raise AssertionError(f"unmapped HF ZAYA top-level key: {key}")
    L, sub = int(m.group(1)), m.group(2)
    attn, moe = 2 * L, 2 * L + 1
    # residual scales: index by CONSUMING minisgl layer (post_attention -> the MoE layer entered after
    # attn; post_mlp -> the NEXT block's attn layer, or the top-level final merge for the last block).
    if sub.startswith("post_attention_residual_scale."):
        return f"model.layers.{moe}.res_scale.{sub.split('.', 1)[1]}"
    if sub.startswith("post_mlp_residual_scale."):
        f = sub.split(".", 1)[1]
        return f"model.res_scale.{f}" if L == n_blocks - 1 else f"model.layers.{2*L+2}.res_scale.{f}"
    if sub == "input_layernorm.weight":
        return f"model.layers.{attn}.input_norm.weight"
    if sub == "post_attention_layernorm.weight":
        return f"model.layers.{moe}.input_norm.weight"
    if sub.startswith("self_attn."):
        a = sub[len("self_attn."):]
        if a == "o_proj.weight":
            return f"model.layers.{attn}.self_attn.o_proj.weight"
        if a == "qk_norm.temp":
            return f"model.layers.{attn}.self_attn.qkv.temp"
        if a.startswith("qkv_proj."):
            nn = _HF_ATTN_QKV.get(a[len("qkv_proj."):])
            assert nn is not None, f"unmapped HF attn qkv key: {a} ({key})"
            return f"model.layers.{attn}.self_attn.qkv.{nn}"
        raise AssertionError(f"unmapped HF self_attn key: {a} ({key})")
    if sub.startswith("mlp.gate."):
        nn = _HF_ROUTER.get(sub[len("mlp.gate."):])
        assert nn is not None, f"unmapped HF router key: {sub} ({key})"
        return f"model.layers.{moe}.zaya_block.router.{nn}"
    if sub.startswith("mlp.experts."):
        return None  # stacked experts -> EP path in _load_zaya_weight
    raise AssertionError(f"unmapped HF ZAYA layer key: {sub} ({key})")


_HF_EXPERT_RE = re.compile(
    r"^model\.layers\.(?P<L>\d+)\.mlp\.experts\.(?P<proj>gate_up_proj|down_proj)(?P<scale>\.weight_scale)?$"
)


def _shard_zaya(name: str, t: torch.Tensor, r: int, n: int, config) -> torch.Tensor:
    """Extract rank r's TP head-shard of a ZAYA CCA-hybrid NATIVE (post-`_zaya_remap`) tensor.

    CCA is head-parallel (analog of `_shard_qwen3_5`'s GDN splits). Applied AFTER the remap, so it
    keys on the native flat param names. n==1 is the identity (TP=1, unchanged). The fp8 experts are
    EP-sharded by index in the loader (NOT tensor-sharded), so they never reach here.

      - qkv.linear_q / linear_k: column-parallel (output = q|k head latents), split dim 0 into the
        rank's contiguous head group. gqa is preserved (q heads 0..3 -> kv head 0 stays on rank 0).
      - qkv.conv_qk_0/1_{weight,bias}: the packed conv over [q(latent_q) | k(latent_k)] channels.
        Block-shard dim 0 by [latent_q, latent_k] so each rank keeps its q-head AND k-head channels
        (a naive chunk would hand rank 0 all q and rank 1 all k — corrupting the head grouping).
      - qkv.temp: per-KV-head temperature, split dim 0.
      - qkv.val_proj1/val_proj2: REPLICATED (per-head [hd, hidden]; the model selects this rank's KV
        head from the [val_proj1|val_proj2] value pair — see ZayaCCAAttn.forward).
      - self_attn.o_proj.weight: row-parallel (input = q-head latent), split dim 1 + all_reduce.
      - embed_tokens.weight: vocab-parallel (tied lm_head shares it), split dim 0.
      - res_scale / input_norm / final_norm / zaya_block.router.* : hidden-wide or route-side ->
        REPLICATED (fall through).
    """
    if n == 1:
        return t
    hd = config.cca_head_dim
    latent_q = config.cca_num_q_heads * hd  # full
    latent_k = config.cca_num_k_heads * hd  # full
    if name.endswith((".self_attn.qkv.linear_q", ".self_attn.qkv.linear_k")):
        return t.chunk(n, dim=0)[r].clone()  # col-parallel: contiguous head group
    if name.endswith(
        (".self_attn.qkv.conv_qk_0_weight", ".self_attn.qkv.conv_qk_0_bias",
         ".self_attn.qkv.conv_qk_1_weight", ".self_attn.qkv.conv_qk_1_bias")
    ):
        return _shard_blocks_dim0(t, [latent_q, latent_k], r, n)  # [q|k] channel block shard
    if name.endswith(".self_attn.qkv.temp"):
        return t.chunk(n, dim=0)[r].clone()  # per-KV-head temperature
    if name.endswith(".self_attn.o_proj.weight"):
        return t.chunk(n, dim=1)[r].clone()  # row-parallel (input head latent) + all_reduce
    if name.endswith("embed_tokens.weight"):
        num = t.shape[0]
        per = div_ceil(num, n)
        return t[r * per : min((r + 1) * per, num)].clone()  # vocab-parallel (tied lm_head)
    # val_proj1/val_proj2, res_scale, input_norm, final_norm, router.* -> replicated
    return t


def _load_zaya_weight(
    model_folder: str, device: torch.device, config
) -> Iterator[Tuple[str, torch.Tensor]]:
    """Streaming loader for the ZAYA1-8B CCA-hybrid fp8 checkpoint (TP head-parallel; EP-over-TP).

    Plain keys are renamed by `_zaya_remap`. The fp8 routed experts STAY fp8: the raw F8_E4M3
    `weight` and per-output-channel F32 `weight_scale` are stacked over the 16 experts into
    `...experts.{gate_up_proj,down_proj}.{weight,weight_scale}` and consumed by `_GroupedFP8Experts`
    (dequant deferred to compute). Dequantizing to bf16 at load is ~16 GB and OOMs the 16 GB card;
    fp8 storage is ~8 GB. `tie_word_embeddings` -> no separate `lm_head.weight`."""
    tp_info = get_tp_info()
    files = glob.glob(f"{model_folder}/*.safetensors")
    files = [f for f in files if not f.endswith("consolidated.safetensors")] or files

    # Expert parallelism: each rank loads ONLY its expert shard [offset : offset+local). EP off loads
    # the full E. The shard group is the DP replicas (DP+EP) or the TP ranks (EP-over-TP) — abstracted.
    if is_ep_enabled():
        ep_size, ep_rank = get_ep_size(), get_ep_rank()
        assert config.num_experts % ep_size == 0, (
            f"EP needs num_experts ({config.num_experts}) divisible by ep_size ({ep_size})"
        )
        ep_local = config.num_experts // ep_size
        ep_offset = ep_rank * ep_local
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
        # Field-agnostic: fp8 experts ship {weight, weight_scale}; 4-bit experts ship {weight_packed,
        # weight_scale}. Accumulate whatever fields the checkpoint has, stack each independently over E.
        fields = expert_buf.setdefault(native_key, {})
        slots = fields.setdefault(field, {})
        slots[local_id] = tensor
        if len(slots) != ep_local:
            return
        # weight/weight_packed: [N,K(/2)] -> stack [local,N,K(/2)]; weight_scale -> [local,...].
        stacked = torch.stack([slots[e] for e in range(ep_local)], dim=0).contiguous()
        del fields[field]
        if not fields:
            del expert_buf[native_key]
        yield f"{native_key}.{field}", stacked

    # HF-format detection: Zyphra's transformers release fuses attn+MoE per layer with HF names
    # (input_layernorm / mlp.experts.gate_up_proj); our Megatron export uses input_norm / local_experts.
    is_hf = False
    if files:
        with safetensors.safe_open(files[0], framework="pt", device="cpu") as _f0:
            is_hf = any(k.endswith("input_layernorm.weight") or ".mlp.experts.gate_up_proj" in k
                        for k in _f0.keys())
    n_blocks = config.num_layers // 2  # HF fused-block count (minisgl models each block as 2 layers)
    # HF names the bare expert tensor `gate_up_proj`/`down_proj`; the model's param is `.weight` (fp8
    # W8A8) or `.weight_packed` (4-bit mxfp4-pack). Pick by declared quant width (the Megatron path
    # reads the field straight off the per-expert key, so this is HF-only).
    _wfield = "weight_packed" if (config.quant is not None
                                  and getattr(config.quant, "bits", 8) == 4) else "weight"

    for file in tqdm(files, desc="Loading weights", disable=not tp_info.is_primary()):
        with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
            for ckpt_name in f.keys():
                if is_hf:
                    # HF ships experts ALREADY stacked [E,...] -> native, EP-sliced (no accumulate).
                    if (em := _HF_EXPERT_RE.match(ckpt_name)) is not None:
                        field = "weight_scale" if em.group("scale") else _wfield
                        native = (f"model.layers.{2 * int(em.group('L')) + 1}.zaya_block.experts."
                                  f"{em.group('proj')}.{field}")
                        t = f.get_tensor(ckpt_name)
                        yield native, t[ep_offset:ep_offset + ep_local].contiguous()
                        continue
                    mega = _zaya_remap_hf(ckpt_name, n_blocks)
                    if mega is None:
                        continue
                    native = _zaya_remap(mega)
                else:
                    if _ZAYA_EXPERT_PATTERN.match(ckpt_name) is not None:
                        yield from _store_expert(ckpt_name, f.get_tensor(ckpt_name))
                        continue
                    native = _zaya_remap(ckpt_name)
                if native is None:
                    continue
                # Head-shard the CCA/o_proj/embed tensors for this rank (no-op at TP=1); experts are
                # EP-sharded above and never reach here.
                raw = _shard_zaya(native, f.get_tensor(ckpt_name), tp_info.rank, tp_info.size, config)
                yield native, raw

    assert not expert_buf, f"incomplete Zaya expert stacks: {list(expert_buf.keys())}"


# ---- Gemma4 / DiffusionGemma (split-head_dim SWA hybrid, parallel dense+MoE FFN) weight loader ----
# The two checkpoints are the SAME backbone under different namespaces: `model.language_model.*`
# (gemma4) and `model.decoder.*` (diffusion_gemma). Everything else they ship is vision (a 27-layer
# tower plus its projector) which this text-only engine skips.
#
# Four remaps and one skip carry the whole thing:
#   1. namespace  -> `model.*`;
#   2. `router.proj.weight` -> `router.weight` (the model flattens the router's single Linear, as
#      the Laguna loader does for its gate);
#   3. `.weight_shape` is DROPPED — compressed-tensors ships an int64 [2] logical-shape tensor beside
#      every packed weight, and the model declares no buffer for it. It is not a weight;
#   4. the 128 per-expert gate/up/down merge + stack through the generic `_gate_up_merge` /
#      `_get_expert_stack_info` into the MoELayer containers.
#
# Sharding differs from every other model here in ONE respect that matters: the k/v head count is
# PER LAYER (8 on the 25 sliding layers, 2 on the 5 full ones), so a single `config.num_kv_heads`
# cannot drive the k/v split — `_shard_gemma4` recovers the layer type from the key instead.
#
# DiffusionGemma's `model.encoder.*` is NOT purely vision. Its text encoder is the SAME 30-layer
# stack as the decoder, tied parameter for parameter — except `layer_scalar`, which is an nn.Buffer,
# and HF's tying machinery ties Parameters only. So the checkpoint ships 30 loose encoder
# `layer_scalar` tensors and nothing else textual. They are numerically equal to the decoder's here,
# which is what lets ONE instantiated stack serve both roles — but that is a property of the export,
# not of the architecture, so it is ASSERTED at load rather than assumed. A future export that
# diverged would otherwise run the encoder pass with the decoder's scalars: no error, just a
# quietly worse model.
_GEMMA4_SKIP_PREFIXES = (
    "model.vision_tower.",
    "model.embed_vision.",
    "model.encoder.vision_tower.",
    "model.encoder.embed_vision.",
)
_GEMMA4_NAMESPACES = ("model.language_model.", "model.decoder.")
_GEMMA4_ENCODER_TEXT = "model.encoder.language_model."


def _gemma4_tie_check_key(name: str) -> str:
    """The model-native decoder key an encoder-side TEXT tensor must equal.

    Only `layer_scalar` may reach here. Anything else in the encoder text namespace means the export
    stopped tying a parameter this engine serves from a single stack, so it is refused loudly."""
    suffix = name.removeprefix(_GEMMA4_ENCODER_TEXT)
    if not suffix.endswith(".layer_scalar"):
        raise ValueError(
            f"DiffusionGemma loader: untied encoder text tensor {name!r}. The port serves the "
            f"encoder and decoder roles from ONE instantiated stack, which holds only while every "
            f"encoder text parameter is tied to its decoder twin; the checkpoint's one legitimate "
            f"exception is `layer_scalar` (a buffer, which HF's tying machinery cannot tie)."
        )
    return "model." + suffix


def _gemma4_remap(name: str) -> str | None:
    """Checkpoint key -> model-native key, or None to skip."""
    if name.endswith(".weight_shape"):
        return None
    if name.startswith(_GEMMA4_SKIP_PREFIXES):
        return None
    if name.startswith(_GEMMA4_ENCODER_TEXT):
        return None  # checked against its decoder twin by the caller, never emitted
    for namespace in _GEMMA4_NAMESPACES:
        if name.startswith(namespace):
            name = "model." + name.removeprefix(namespace)
            break
    else:
        # A key in neither namespace and not vision: refuse rather than silently drop it. A dropped
        # weight leaves the model holding uninitialized meta memory, which reads as garbage output
        # and not as an error.
        if not name.startswith("model."):
            raise ValueError(
                f"Gemma4 loader: unrecognized checkpoint key {name!r} — it is in neither the "
                f"`model.language_model.*` (gemma4) nor the `model.decoder.*` (diffusion_gemma) "
                f"namespace, and is not a vision tensor. Refusing to drop it silently."
            )
    if name.endswith(".router.proj.weight"):
        return name.replace(".router.proj.weight", ".router.weight")
    return name


def _gemma4_layer_is_sliding(name: str, config) -> bool | None:
    """Which attention schedule slot does this key belong to? None when the key is not layer-scoped."""
    match = _LAYER_IDX_PATTERN.search(name)
    if match is None or config.layer_types is None:
        return None
    return config.layer_types[int(match.group(1))] == "sliding_attention"


def _shard_gemma4(name: str, t: torch.Tensor, r: int, n: int, config) -> torch.Tensor:
    """Rank-r plain-TP shard, applied BEFORE the gate/up merge and the expert stack.

    compressed-tensors packs int4 along the INPUT dim (weight_packed is (out, in//8), weight_scale is
    (out, in//group)), so — unlike the AWQ suffixes — an output-parallel split is dim 0 and an
    input-parallel split is dim 1 for the packed tensor exactly as for a plain `.weight`."""
    if n == 1:
        return t
    # Replicated: every norm, the per-layer residual scalar, and all three router tensors (the
    # router must produce identical logits on every rank or the ranks route to different experts).
    if (
        name.endswith("_layernorm.weight")
        or name.endswith("_norm.weight")
        or name == "model.norm.weight"
        or name.endswith(".layer_scalar")
        or ".router." in name
    ):
        return t
    if name.endswith("embed_tokens.weight"):
        num_emb = t.shape[0]
        per = div_ceil(num_emb, n)
        return t[r * per : min((r + 1) * per, num_emb), :].clone()
    # q_proj and the dense/expert gate+up are output-parallel; o_proj and the down projections are
    # input-parallel. Both hold for the packed and the scale tensor.
    if ".q_proj." in name or ".gate_proj." in name or ".up_proj." in name:
        return t.chunk(n, dim=0)[r].clone()
    if ".o_proj." in name or ".down_proj." in name:
        return t.chunk(n, dim=1)[r].clone()
    if ".k_proj." in name or ".v_proj." in name:
        # PER-LAYER kv head count: 8 on a sliding layer, 2 on a full one. AttentionLayer sizes the
        # layer with div_even(nkv, tp, allow_replicate=True), so when the heads do not divide the TP
        # size it REPLICATES — and the loader must make the same call or the buffer will not fit.
        is_sliding = _gemma4_layer_is_sliding(name, config)
        nkv = (config.swa_num_kv_heads or config.num_kv_heads) if is_sliding else config.num_kv_heads
        if nkv % n:
            return t  # replicated, mirroring div_even(..., allow_replicate=True)
        return t.chunk(n, dim=0)[r].clone()
    return t


def _load_gemma4_weight(
    model_folder: str, device: torch.device, config
) -> Iterator[Tuple[str, torch.Tensor]]:
    """Streaming loader for the Gemma4 / DiffusionGemma backbone (see the family note above)."""
    tp_info = get_tp_info()
    files = glob.glob(f"{model_folder}/*.safetensors")
    files = [f for f in files if not f.endswith("consolidated.safetensors")] or files
    merge_buf: Dict[str, Dict[str, torch.Tensor]] = {}
    expert_buf = _ExpertStacker()
    _ep_shard, _ep_local, _ep_offset = _ep_expert_shard(config)

    def emit(native_key: str, tensor: torch.Tensor) -> Iterator[Tuple[str, torch.Tensor]]:
        if (mm := _gate_up_merge(native_key)) is not None:
            merged_key, slot = mm
            merge_buf.setdefault(merged_key, {})[slot] = tensor
            if len(merge_buf[merged_key]) != 2:
                return
            parts = [merge_buf[merged_key][s] for s in ("gate", "up")]
            del merge_buf[merged_key]
            # compressed-tensors packs along the INPUT dim, so gate|up concatenate on the OUTPUT dim
            # (0) for the packed weight and its scale alike — the AWQ dim-1 flip does not apply.
            native_key, tensor = merged_key, torch.cat(parts, dim=0)
        if (einfo := _get_expert_stack_info(native_key)) is not None:
            packed_key, idx = einfo
            if _ep_shard and not (_ep_offset <= idx < _ep_offset + _ep_local):
                return
            stacked = expert_buf.add(packed_key, idx - _ep_offset, tensor, _ep_local)
            if stacked is None:
                return
            yield packed_key, stacked
        else:
            yield native_key, tensor

    # The encoder/decoder tie check (see the family note above). Both sides are collected because
    # the two namespaces land in different shards and neither ordering is guaranteed; 30 scalars is
    # 30 floats, so buffering them costs nothing.
    tie_encoder: Dict[str, torch.Tensor] = {}
    tie_decoder: Dict[str, torch.Tensor] = {}

    for file in tqdm(files, desc="Loading weights", disable=not tp_info.is_primary()):
        with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
            for ckpt_name in f.keys():
                if ckpt_name.startswith(_GEMMA4_ENCODER_TEXT):
                    tie_encoder[_gemma4_tie_check_key(ckpt_name)] = f.get_tensor(ckpt_name)
                    continue
                native = _gemma4_remap(ckpt_name)
                if native is None:
                    continue
                raw = _shard_gemma4(
                    native, f.get_tensor(ckpt_name), tp_info.rank, tp_info.size, config
                )
                if native.endswith(".layer_scalar"):
                    tie_decoder[native] = raw
                yield from emit(native, raw)

    assert not merge_buf, f"incomplete gate/up merges in checkpoint: {list(merge_buf.keys())}"
    assert not expert_buf, f"incomplete expert stacks in checkpoint: {expert_buf.pending}"
    for key, enc in tie_encoder.items():
        dec = tie_decoder.get(key)
        # Bit-equality, not a tolerance: these are the same trained scalar exported twice, so any
        # difference at all means the two roles no longer share one stack.
        if dec is None or not torch.equal(enc.reshape(-1).to(dec.dtype), dec.reshape(-1)):
            raise ValueError(
                f"DiffusionGemma loader: encoder/decoder {key!r} are not tied — encoder="
                f"{enc.reshape(-1)[:4].tolist()} decoder="
                f"{None if dec is None else dec.reshape(-1)[:4].tolist()}. This engine runs both "
                f"roles through ONE instantiated stack, so a divergence here needs two scalar "
                f"vectors selected by execution mode, not a tolerance."
            )


# ---- poolside/Laguna-XS-2.1 (SWA-hybrid gated-attention NVFP4 MoE) weight loader ----
# Standard `model.layers.N.*` naming, so the remap is nearly identity. Two families need a touch:
#   1. the router balancing bias ships as `mlp.experts.e_score_correction_bias` (co-located with the
#      experts in the checkpoint) but the model holds it on the router -> rename to `mlp.gate.*`;
#   2. fp8-KV `self_attn.{k,v}_scale` are dropped for the bf16-KV v1 serve.
# gate/up merge (dense L0 + shared expert + routed experts) and per-expert stacking are the generic
# `_gate_up_merge` / `_get_expert_stack_info`; the NVFP4 two-level scale is folded to one fp16 per-group
# scale at the leaf (before any merge) exactly as in `_load_qwen3_5_weight`. Plain TP=2 (no EP):
# attention q/k/v/g/o, dense L0, embed/lm_head AND the routed experts are TP-sharded (the experts split
# their moe_intermediate FFN across ranks — gate/up on output dim 0, down on input dim 1 — so each card
# holds ~half the 31B of expert weight); the router gate, correction bias and the always-on shared
# expert stay whole. NVFP4 packed/scale tensors split like a bf16 weight (not `_AWQ_SUFFIXES`).


def _laguna_remap(name: str) -> tuple[str] | None:
    """Map a Laguna checkpoint key to its native key (or None to skip). Returns a 1-tuple `(native,)`
    (the outer loop reads plan[1]) so the shape mirrors the qwen3_5 `("direct", native)` usage."""
    # bf16-KV v1: drop the checkpoint's fp8-KV per-tensor scales (attention runs bf16 KV).
    if name.endswith((".self_attn.k_scale", ".self_attn.v_scale")):
        return None
    # Router balancing bias lives on the experts in the checkpoint; the model holds it on the router.
    if name.endswith(".mlp.experts.e_score_correction_bias"):
        return (name.replace(".mlp.experts.e_score_correction_bias", ".mlp.gate.e_score_correction_bias"),)
    return (name,)


def _shard_laguna(name: str, t: torch.Tensor, r: int, n: int, config) -> torch.Tensor:
    """Rank-r plain-TP shard of a Laguna checkpoint tensor (applied BEFORE gate/up merge + expert
    stack). Mirrors the dense `_shard_tensor` rules AND intermediate-splits the routed experts, exactly
    as GLM/qwen do at TP=2 (no EP): each rank holds ~half the 31B of expert weight.

      * Column-parallel (output dim 0): attention q/k/v/g; dense-L0 gate/up; ROUTED-EXPERT gate/up.
        For a routed-expert NVFP4 pair this splits N=moe_intermediate across ranks (weight_packed
        (N,K//2) & weight_scale (N,K//16) both split dim 0), giving each rank moe_intermediate/n rows.
      * Row-parallel (input dim 1): attention o_proj; dense-L0 down; ROUTED-EXPERT down (splits
        K=moe_intermediate: weight_packed (N,K//2) & weight_scale (N,K//16) both split dim 1).
      * Vocab-parallel (dim 0): embed / untied lm_head.
      * REPLICATED (whole): router gate + `e_score_correction_bias`; the always-on shared expert
        (its down K=shared_inter must stay whole for the e2m1 kernel); every norm.
    n==1 is the identity. NVFP4 packed/scale tensors are NOT `_AWQ_SUFFIXES`, so — like a bf16 weight —
    output-parallel splits dim 0 and input-parallel splits dim 1 (no AWQ axis flip)."""
    if n == 1:
        return t
    # Replicated / whole (no TP split): shared expert, router gate + bias, all norms.
    if (
        ".shared_expert." in name
        or name.endswith(".mlp.gate.weight")
        or name.endswith(".e_score_correction_bias")
        or name.endswith("_norm.weight")
        or name.endswith("layernorm.weight")
        or name == "model.norm.weight"
    ):
        return t
    # NVFP4 split arm: `.down_proj.weight_global` is the per-OUTPUT-CHANNEL f32 global, an (N,)
    # vector, and down_proj is ROW-parallel — it splits the input K, so its output N is full width on
    # every rank and the global REPLICATES. Matched before the `.down_proj.` rule below, which would
    # chunk a 1-D tensor on dim 1 and raise. (gate/up's global DOES split, on dim 0, which is what the
    # column-parallel rule already does.)
    if name.endswith(".down_proj.weight_global"):
        return t
    # Column-parallel (output dim 0): attention q/k/v/g; dense-L0 + routed-expert gate/up.
    if name.endswith((".q_proj.weight", ".k_proj.weight", ".v_proj.weight", ".g_proj.weight")) or (
        ".gate_proj." in name or ".up_proj." in name
    ):
        return t.chunk(n, dim=0)[r].clone()
    # Row-parallel (input dim 1): attention o_proj; dense-L0 + routed-expert down.
    if name.endswith(".o_proj.weight") or ".down_proj." in name:
        return t.chunk(n, dim=1)[r].clone()
    # Vocab-parallel: embed / untied lm_head.
    if name.endswith("embed_tokens.weight") or name == "lm_head.weight":
        num_emb = t.shape[0]
        per = div_ceil(num_emb, n)
        return t[r * per : min((r + 1) * per, num_emb), :].clone()
    return t  # anything else (unreached) replicated


def _load_laguna_weight(
    model_folder: str, device: torch.device, config
) -> Iterator[Tuple[str, torch.Tensor]]:
    """Streaming loader for poolside/Laguna-XS-2.1-NVFP4 (see the family note above)."""
    tp_info = get_tp_info()
    files = glob.glob(f"{model_folder}/*.safetensors")
    files = [f for f in files if not f.endswith("consolidated.safetensors")] or files
    merge_buf: Dict[str, Dict[str, torch.Tensor]] = {}  # gate/up -> gate_up
    expert_buf = _ExpertStacker()  # per-expert -> filled into a preallocated [E, ...]
    _is_nvfp4 = config.quant is not None and config.quant.is_nvfp4
    nvfp4_fold_buf: Dict[str, Dict[str, torch.Tensor]] = {}
    _ep_shard, _ep_local, _ep_offset = _ep_expert_shard(config)

    def emit(native_key: str, tensor: torch.Tensor) -> Iterator[Tuple[str, torch.Tensor]]:
        if (mm := _gate_up_merge(native_key)) is not None:
            merged_key, slot = mm
            merge_buf.setdefault(merged_key, {})[slot] = tensor
            if len(merge_buf[merged_key]) != 2:
                return
            parts = [merge_buf[merged_key][s] for s in ("gate", "up")]
            del merge_buf[merged_key]
            cat_dim = 1 if merged_key.endswith((".qweight", ".qzeros", ".scales")) else 0
            native_key, tensor = merged_key, torch.cat(parts, dim=cat_dim)
        if config.is_moe and (einfo := _get_expert_stack_info(native_key)) is not None:
            packed_key, idx = einfo
            if _ep_shard and not (_ep_offset <= idx < _ep_offset + _ep_local):
                return  # not this replica's expert
            stacked = expert_buf.add(packed_key, idx - _ep_offset, tensor, _ep_local)
            if stacked is None:
                return
            yield packed_key, stacked
        else:
            yield native_key, tensor

    for file in tqdm(files, desc="Loading weights", disable=not tp_info.is_primary()):
        with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
            for ckpt_name in f.keys():
                # NVFP4 scale resolution at the LEAF (before rename / gate-up merge / expert stack).
                # Routed experts keep BOTH levels and emit TWO leaves; dense linears still fold to one
                # fp16 per-group scale — `nvfp4.nvfp4_leaf_scales` owns that split, once, for every
                # loader. weight_packed passes through 4-bit; input_global_scale (FP4 act calib) is
                # dropped (the e2m1 kernel quantizes acts to fp8).
                leaves: "list[Tuple[str, torch.Tensor | None]]" = [(ckpt_name, None)]
                if _is_nvfp4:
                    if ckpt_name.endswith(".input_global_scale"):
                        continue
                    if ckpt_name.endswith((".weight_scale", ".weight_global_scale")):
                        base, field = ckpt_name.rsplit(".", 1)
                        buf = nvfp4_fold_buf.setdefault(base, {})
                        buf[field] = f.get_tensor(ckpt_name)
                        if len(buf) < 2:
                            continue
                        del nvfp4_fold_buf[base]
                        leaves = list(
                            nvfp4.nvfp4_leaf_scales(
                                base,
                                buf["weight_scale"],
                                buf["weight_global_scale"],
                                global_field="weight_global_scale",
                            )
                        )
                for name, override in leaves:
                    plan = _laguna_remap(name)
                    if plan is None:
                        continue
                    native = plan[0]
                    tens = override if override is not None else f.get_tensor(name)
                    raw = _shard_laguna(native, tens, tp_info.rank, tp_info.size, config)
                    yield from emit(native, raw)
    assert not merge_buf, f"incomplete gate/up merges in checkpoint: {list(merge_buf.keys())}"
    assert not expert_buf, f"incomplete expert stacks in checkpoint: {expert_buf.pending}"
    assert not nvfp4_fold_buf, (
        f"incomplete NVFP4 scale/global pairs: {list(nvfp4_fold_buf.keys())}"
    )


_MUSE_SKIP_PREFIXES = (
    "model.vision_tower.",
    "model.vision_adapter.",
    "model.vision_projection",
)


def _muse_glimmer_remap(name: str) -> "str | None":
    """Muse-Glimmer checkpoint key -> native key, or None to skip.

    Only two rules: drop the vision stack (text-only engine), and collapse the transformers-v5
    `model.language_model.` namespace to the `model.` the decoder is built in. `lm_head.weight` is
    already top-level. An unrecognised namespace RAISES rather than being silently dropped — a
    quietly ignored key is how a mis-shaped port loads cleanly and serves garbage."""
    if name.startswith(_MUSE_SKIP_PREFIXES):
        return None
    if name.startswith("model.language_model."):
        return "model." + name[len("model.language_model.") :]
    if name == "lm_head.weight":
        return name
    raise ValueError(f"unexpected Muse-Glimmer checkpoint key: {name}")


def _muse_gate_up_merge(key: str):
    """`mlp.gate_proj`/`mlp.up_proj` -> `mlp.gate_up_proj`. Returns (merged_key, slot) or None.

    Anchored on `.mlp.` ON PURPOSE. Muse-Glimmer has a SECOND, unrelated `gate_proj` — the attention
    output gate at `self_attn.gate_proj` — and the generic `_gate_up_merge` would sweep it into a
    `self_attn.gate_up_proj` merge that can never complete (there is no `self_attn.up_proj`)."""
    for sub, slot in ((".mlp.gate_proj.", "gate"), (".mlp.up_proj.", "up")):
        if sub in key:
            return key.replace(sub, ".mlp.gate_up_proj."), slot
    return None


def _shard_muse_glimmer(name: str, t: torch.Tensor, r: int, n: int) -> torch.Tensor:
    """Rank-r plain-TP shard of a Muse-Glimmer tensor, applied BEFORE the gate/up merge.

    Matching is on the MODULE infix (`.q_proj.`), not on a `.weight` suffix: every linear here is
    NVFP4, so the leaves are `.weight_packed` / `.weight_scale`, and a suffix match would silently
    replicate all of them. NVFP4 tensors are not `_AWQ_SUFFIXES`, so — like a bf16 weight —
    output-parallel splits dim 0 and input-parallel splits dim 1 with no AWQ axis flip: for a
    (N, K//2) packed weight and its (N, K//16) group scale, both axes stay in step.

      * Column-parallel (dim 0): q/k/v/gate (attention) and mlp gate/up.
      * Row-parallel (dim 1): o_proj, mlp down.
      * Vocab-parallel (dim 0): embed_tokens, untied lm_head.
      * Replicated: every norm.
    n == 1 is the identity."""
    if n == 1:
        return t
    if name.endswith("layernorm.weight") or name == "model.norm.weight":
        return t
    # `.gate_proj.` deliberately covers BOTH the attention gate and the MLP gate: both are
    # column-parallel over dim 0, so one rule is right for both. Only the MERGE has to tell them
    # apart, which `_muse_gate_up_merge` does.
    if any(s in name for s in (".q_proj.", ".k_proj.", ".v_proj.", ".gate_proj.", ".up_proj.")):
        return t.chunk(n, dim=0)[r].clone()
    if ".o_proj." in name or ".down_proj." in name:
        return t.chunk(n, dim=1)[r].clone()
    if name.endswith("embed_tokens.weight") or name == "lm_head.weight":
        num_emb = t.shape[0]
        per = div_ceil(num_emb, n)
        return t[r * per : min((r + 1) * per, num_emb), :].clone()
    return t


def _load_muse_glimmer_weight(
    model_folder: str, device: torch.device, config
) -> Iterator[Tuple[str, torch.Tensor]]:
    """Streaming loader for Muse-Glimmer (see docs/MUSE_GLIMMER_PORT.md).

    Dense, so there is no expert stack — the only buffering is the NVFP4 scale/global pair and the
    MLP gate/up merge, both asserted empty at the end."""
    tp_info = get_tp_info()
    files = glob.glob(f"{model_folder}/*.safetensors")
    files = [f for f in files if not f.endswith("consolidated.safetensors")] or files
    merge_buf: Dict[str, Dict[str, torch.Tensor]] = {}
    _is_nvfp4 = config.quant is not None and config.quant.is_nvfp4
    nvfp4_fold_buf: Dict[str, Dict[str, torch.Tensor]] = {}

    for file in tqdm(files, desc="Loading weights", disable=not tp_info.is_primary()):
        with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
            for name in f.keys():
                if name.startswith(_MUSE_SKIP_PREFIXES):
                    continue  # vision stack: unquantized, so it never enters the fold buffer
                # NVFP4 scale resolution at the LEAF — before the rename and before the gate/up
                # merge — so no per-tensor scalar ever has to survive a concat. weight_packed passes
                # through still 4-bit; input_global_scale (the FP4 activation calibration) is
                # dropped, because the e2m1 kernel quantizes activations to fp8 dynamically.
                #
                # Muse-Glimmer is DENSE, so `nvfp4_leaf_scales` always takes its FOLD arm here and
                # this loader can only ever see one leaf per pair — asserted rather than assumed, so
                # that adding an `.experts.` module to this family (which would start splitting) is a
                # loud failure here instead of a global vector silently dropped on the floor.
                override = None
                if _is_nvfp4:
                    if name.endswith(".input_global_scale"):
                        continue
                    if name.endswith((".weight_scale", ".weight_global_scale")):
                        base, field = name.rsplit(".", 1)
                        buf = nvfp4_fold_buf.setdefault(base, {})
                        buf[field] = f.get_tensor(name)
                        if len(buf) < 2:
                            continue
                        del nvfp4_fold_buf[base]
                        leaves = nvfp4.nvfp4_leaf_scales(
                            base,
                            buf["weight_scale"],
                            buf["weight_global_scale"],
                            global_field="weight_global_scale",
                        )
                        assert len(leaves) == 1, (
                            f"{base}: Muse-Glimmer is dense, but nvfp4_leaf_scales returned "
                            f"{len(leaves)} leaves ({[n for n, _ in leaves]}). This loader has no "
                            f"expert stack to carry a per-output-channel global through."
                        )
                        name, override = leaves[0]
                native = _muse_glimmer_remap(name)
                if native is None:
                    continue
                tens = override if override is not None else f.get_tensor(name)
                tens = _shard_muse_glimmer(native, tens, tp_info.rank, tp_info.size)
                if (mm := _muse_gate_up_merge(native)) is not None:
                    merged_key, slot = mm
                    merge_buf.setdefault(merged_key, {})[slot] = tens
                    if len(merge_buf[merged_key]) != 2:
                        continue
                    parts = [merge_buf[merged_key][s] for s in ("gate", "up")]
                    del merge_buf[merged_key]
                    # Both the packed weight (N, K//2) and its group scale (N, K//16) concat on
                    # dim 0: the gate/up merge stacks OUTPUT rows, the axis neither tensor packs
                    # along, so the scale stays row-aligned with the weight it describes.
                    yield merged_key, torch.cat(parts, dim=0)
                else:
                    yield native, tens
    assert not merge_buf, f"incomplete gate/up merges in checkpoint: {list(merge_buf.keys())}"
    assert not nvfp4_fold_buf, f"incomplete NVFP4 scale/global pairs: {list(nvfp4_fold_buf.keys())}"


def chunked_weight_source(
    model_path: str, device: torch.device, spec_algorithm: str = "mtp"
) -> "Optional[Tuple[List, Callable]]":
    """`(chunks, stream)` for `weights.stage_b.ChunkedWeightLoader`, or None if this family has none.

    THE DISPATCH TWIN OF `load_weight`, and deliberately shaped like it. `load_weight` answers "give
    me every tensor"; this answers "give them to me in units small enough to place as they arrive".
    A family that has not implemented a chunk enumeration returns None and the caller falls back to
    the one-shot load — which is the correct default, not a degraded one: chunking only buys anything
    for a checkpoint that does not fit at once, and every other model this repo serves does.

    It re-derives `ModelConfig` from the same three inputs `load_weight` does rather than taking the
    engine's, because the chunk enumeration is keyed on `num_layers` and a config built from a
    different `spec_algorithm` has a different decoder depth. Two derivations of the layer count that
    can disagree is exactly how a chunk list ends up naming a layer the model does not have.
    """
    from .config import ModelConfig

    config = ModelConfig.from_hf(
        cached_load_hf_config(model_path),
        spec_algorithm=spec_algorithm,
        ckpt_tensor_names=checkpoint_tensor_names(model_path),
    )
    if config.is_qwen4_exp:
        return qwen4_exp_chunked_source(model_path, device, config)
    return None


def load_weight(
    model_path: str, device: torch.device, spec_algorithm: str = "mtp"
) -> Iterator[Tuple[str, torch.Tensor]]:
    """Streaming weight loader. Yields (name, tensor) pairs already sharded, merged,
    and on device. Peak CPU memory: one full tensor + a small merge buffer.

    `spec_algorithm` must match the value used to build the model so load_mtp agrees (the MTP
    head is only built under --spec-algorithm mtp; see ModelConfig.from_hf)."""
    from .config import ModelConfig

    model_folder = download_hf_weight(model_path)
    config = ModelConfig.from_hf(
        cached_load_hf_config(model_path),
        spec_algorithm=spec_algorithm,
        # Same cross-check the model builder does, from the same source, so load_mtp cannot disagree
        # with what was built (that disagreement is exactly how an unfillable head reaches the loader).
        ckpt_tensor_names=checkpoint_tensor_names(model_path),
    )
    # MUST precede the is_gdn_hybrid branch: qwen4_exp IS a GDN hybrid (36 of its 48 layers are
    # linear-attention), so without this it silently routes into the Qwen3.5 loader — whose remap
    # happily passes `hyper_connection_mixer.*` / `ple.*` / `indexer.*` through as "direct" and then
    # dies in load_state_dict with a 296k-key unexpected-keys dump that names no cause. Its streaming
    # loader is bring-up plan T1.3; the NAME mapping it will use (`qwen4_exp_remap`) already exists
    # above and is tested against the full checkpoint index.
    if config.is_qwen4_exp:
        yield from _load_qwen4_exp_weight(model_folder, device, config)
        return
    if config.is_gdn_hybrid:
        yield from _load_qwen3_5_weight(model_folder, device, config)
        return
    if config.is_cca_hybrid:
        yield from _load_zaya_weight(model_folder, device, config)
        return
    # MUST precede the is_swa_hybrid branch: Gemma4 is also a SWA hybrid, so it would otherwise be
    # routed into the Laguna loader, whose remap and TP sharding are Laguna-specific.
    if config.is_gemma4:
        yield from _load_gemma4_weight(model_folder, device, config)
        return
    # MUST precede the is_swa_hybrid branch, for the same reason Gemma4 does: Muse-Glimmer is also
    # a SWA hybrid, so it would otherwise be routed into the Laguna loader, whose remap (no
    # `model.language_model.` namespace, no vision skip) and TP sharding (suffix-matched on
    # `.weight`, which no NVFP4 leaf ends in) are both wrong for it.
    if config.is_muse_glimmer:
        yield from _load_muse_glimmer_weight(model_folder, device, config)
        return
    if config.is_swa_hybrid:
        yield from _load_laguna_weight(model_folder, device, config)
        return
    files = glob.glob(f"{model_folder}/*.safetensors")
    files = [f for f in files if not f.endswith("consolidated.safetensors")] or files
    tp_info = get_tp_info()

    # Buffer for merge groups: merged_key -> {slot: tensor}
    merge_buf: Dict[str, Dict[str, torch.Tensor]] = {}
    expert_buf = _ExpertStacker()
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
                    stacked = expert_buf.add(
                        packed_key, expert_idx - _ep_offset, out[1], _ep_local
                    )
                    if stacked is None:
                        continue
                    yield packed_key, stacked
                else:  # Normal dense model
                    yield out[0], out[1]

    assert not merge_buf, f"Incomplete merge groups in checkpoint: {list(merge_buf.keys())}"
    assert not expert_buf, f"Incomplete expert tensors in checkpoint: {expert_buf.pending}"
