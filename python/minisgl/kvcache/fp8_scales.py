"""fp8-KV scale resolution: where a served engine's `k_scale`/`v_scale` actually come from.

An fp8 (e4m3) KV cache stores `k / scale` and reads back `stored * scale`. The scale is therefore
part of the CACHE'S CONTRACT, not a tuning knob: one descale has to undo every store that was ever
written under it. Two consequences drive the whole design of this module.

1. **The scale must be final before the first KV that will be read is written.** Calibrating
   mid-serve and mutating the table under a live cache silently corrupts everything already stored
   (measured: a captured graph DOES pick up a later in-place descale write — `max|Δ|=4.85e-01` on
   the attention output — and restoring the old value restores it bit-exactly, so the mutation is
   real and the corruption is real). Every path here therefore resolves and installs at ENGINE BOOT,
   between KV-pool construction and graph capture, and never touches the table again.
2. **Inventing a scale is worse than reporting that you have none.** So the order is: use whatever
   the checkpoint/sidecar actually measured, and if there is nothing, WARN and fall back to the
   defined identity (1.0) rather than guessing.

RESOLUTION ORDER (first hit wins):

  a. `MINISGL_KV_FP8_SCALES=<file|dir>` — an explicit sidecar produced by `tools/kv_fp8_calibrate.py`
     from representative text. This is the only source that yields genuine PER-HEAD scales, because
     per-head amax is a property of the ACTIVATIONS, not of the checkpoint.
  b. `<model_dir>/kv_scales.safetensors` — the same sidecar, discovered next to the weights.
  c. The CHECKPOINT'S OWN per-tensor scales: compressed-tensors ships `quantization_config.
     kv_cache_scheme` plus `model.layers.N.self_attn.{k,v}_scale`. Broadcast to every head. This is
     general serving infrastructure — the trigger is the declared `quant_config` + the tensor names,
     never a model name.
  d. Nothing → warn (naming both fixes) and keep the identity scale.

WHAT THE SCALE TENSOR MEANS. `k_scale` is the DEQUANT (descale) factor, the compressed-tensors /
vLLM convention: `stored = k / k_scale`, `k ≈ stored * k_scale`, so a symmetric max-calibrated
scale is `amax / 448` for OCP e4m3. That maps 1:1 onto `MHAKVCache.k_descale`, and the store side
gets `1 / k_scale`. (No `* 2` fixup: this engine's cache is `float8_e4m3fn` (OCP, max 448), not the
`e4m3fnuz` variant some ROCm stacks quantize for.)
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Tuple

import torch
from minisgl.distributed import get_tp_info
from minisgl.utils import cached_load_hf_config, download_hf_weight, init_logger

if TYPE_CHECKING:
    from minisgl.models import ModelConfig

    from .mha_pool import MHAKVCache

logger = init_logger(__name__)

# Sidecar filename looked for next to the weights (written by tools/kv_fp8_calibrate.py).
SIDECAR_NAME = "kv_scales.safetensors"

# `<anything>.<layer index>.<anything>.k_scale` — the compressed-tensors naming, matched
# structurally rather than by model. The LAST numeric path component before the suffix is the layer
# index (`model.layers.7.self_attn.k_scale` -> 7).
_SCALE_KEY = re.compile(r"(?:^|\.)(\d+)\.(?:[^.]+\.)*(k|v)_scale$")

FP8_MAX = 448.0  # e4m3 (OCP) max representable magnitude


@dataclass
class KVScaleSet:
    """Per-global-layer `(k_scale, v_scale)` CPU fp32 rows, each numel 1 (per-tensor) or
    num_kv_heads (per-head), plus where they came from (for the boot log)."""

    scales: Dict[int, Tuple[torch.Tensor, torch.Tensor]]
    source: str
    per_head: bool


def _read_scale_file(path: str) -> Dict[int, Dict[str, torch.Tensor]]:
    """Pull every `*.{k,v}_scale` out of one safetensors file, keyed by layer index."""
    from safetensors import safe_open

    out: Dict[int, Dict[str, torch.Tensor]] = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            m = _SCALE_KEY.search(key)
            if m is None:
                continue
            out.setdefault(int(m.group(1)), {})[m.group(2)] = (
                f.get_tensor(key).reshape(-1).float()
            )
    return out


def _collect(paths: list[str], source: str) -> KVScaleSet | None:
    """Merge scale tensors across shards; require both k and v for every layer that has either."""
    merged: Dict[int, Dict[str, torch.Tensor]] = {}
    for p in paths:
        for layer, kv in _read_scale_file(p).items():
            merged.setdefault(layer, {}).update(kv)
    if not merged:
        return None
    scales: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
    per_head = False
    for layer, kv in merged.items():
        if "k" not in kv or "v" not in kv:
            logger.warning_rank0(
                f"fp8-KV scales ({source}): layer {layer} has only "
                f"{sorted(kv)}_scale — skipping the layer rather than half-scaling it"
            )
            continue
        k, v = kv["k"], kv["v"]
        if k.numel() != v.numel():
            logger.warning_rank0(
                f"fp8-KV scales ({source}): layer {layer} k_scale numel {k.numel()} != "
                f"v_scale numel {v.numel()} — skipping the layer"
            )
            continue
        per_head |= k.numel() > 1
        scales[layer] = (k, v)
    if not scales:
        return None
    return KVScaleSet(scales=scales, source=source, per_head=per_head)


def _sidecar(model_path: str) -> KVScaleSet | None:
    explicit = os.environ.get("MINISGL_KV_FP8_SCALES")
    if explicit:
        path = os.path.join(explicit, SIDECAR_NAME) if os.path.isdir(explicit) else explicit
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"MINISGL_KV_FP8_SCALES={explicit!r} does not resolve to a scale file ({path}). "
                f"Produce one with tools/kv_fp8_calibrate.py, or unset the variable."
            )
        return _collect([path], f"sidecar {path}")
    folder = download_hf_weight(model_path)
    path = os.path.join(folder, SIDECAR_NAME)
    if os.path.isfile(path):
        return _collect([path], f"sidecar {path}")
    return None


def _from_checkpoint(model_path: str) -> KVScaleSet | None:
    """The checkpoint's own kv-cache scales, gated on the DECLARED `kv_cache_scheme`.

    The gate matters: a `k_scale` calibrated for int8 KV is `amax/127`, not `amax/448`, so applying
    it as an e4m3 descale would quietly store everything at 28% of range. So the scheme has to say
    8-bit FLOAT before we believe the tensors; anything else is reported and skipped."""
    hf = cached_load_hf_config(model_path)
    qc = getattr(hf, "quantization_config", None) or {}
    if not isinstance(qc, dict):
        qc = getattr(qc, "to_dict", lambda: {})()
    scheme = qc.get("kv_cache_scheme")
    folder = download_hf_weight(model_path)
    files = sorted(glob.glob(os.path.join(folder, "*.safetensors")))
    if not scheme:
        return None
    bits, kind = scheme.get("num_bits"), (scheme.get("type") or "").lower()
    if bits != 8 or kind != "float":
        logger.warning_rank0(
            f"fp8-KV: checkpoint declares kv_cache_scheme num_bits={bits} type={kind!r}, which is "
            f"NOT the e4m3 this cache stores — its k_scale/v_scale are calibrated for a different "
            f"grid, so they are IGNORED (using them would misplace every stored value)."
        )
        return None
    if scheme.get("dynamic"):
        logger.warning_rank0(
            "fp8-KV: checkpoint declares a DYNAMIC kv_cache_scheme; this cache needs one static "
            "scale per (layer, head) because a single descale must undo every stored token. "
            "Ignoring."
        )
        return None
    return _collect(files, f"checkpoint kv_cache_scheme (strategy={scheme.get('strategy')})")


def resolve_kv_fp8_scales(model_path: str) -> KVScaleSet | None:
    """Sidecar first (real per-head data), then the checkpoint's per-tensor scales, else None."""
    return _sidecar(model_path) or _from_checkpoint(model_path)


def _shard_row(row: torch.Tensor, local_heads: int, global_heads: int) -> torch.Tensor:
    """TP-shard one per-head scale row. A per-tensor row (numel 1) broadcasts unchanged; a per-head
    row is sliced exactly like the KV heads themselves (`div_even(..., allow_replicate=True)`, so
    local == global means the heads are REPLICATED on every rank and the row is not split)."""
    if row.numel() == 1 or local_heads == global_heads:
        return row
    tp = get_tp_info()
    assert row.numel() == global_heads, (
        f"per-head kv scale row has numel {row.numel()}, expected {global_heads}"
    )
    return row[tp.rank * local_heads : (tp.rank + 1) * local_heads].contiguous()


def install_kv_fp8_scales(
    model_path: str,
    model_config: "ModelConfig",
    kv_cache: "MHAKVCache | None",
    swa_kv_cache: "MHAKVCache | None" = None,
) -> None:
    """Resolve and freeze the fp8-KV scales for every pool, at boot, before graph capture.

    No-op for a non-fp8 cache. Maps GLOBAL checkpoint layer indices onto each pool's COMPACT layer
    index: a SWA-hybrid model splits its layers across two pools (full-attention layers into the
    main pool, sliding layers into the ring), and `ModelConfig.full_attn_layer_ids` /
    `swa_layer_ids` are exactly the compaction the model itself uses (see `laguna_layer_plan`)."""
    pools = [(kv_cache, None), (swa_kv_cache, "swa")]
    pools = [
        (p, tag)
        for p, tag in pools
        # MLA pools are excluded by construction, not by omission: an MLA cache is ONE latent vector
        # per token with no head axis at all, so a per-head (or even a per-query-head) descale has
        # nothing to index — see the note in the mla_attend kernel.
        if p is not None and getattr(p, "kv_is_fp8", False) and hasattr(p, "set_fp8_kv_scales")
    ]
    if not pools:
        return

    scaleset = resolve_kv_fp8_scales(model_path)
    if scaleset is None:
        logger.warning_rank0(
            "fp8-KV is ON but NO calibrated scales were found — every k_scale/v_scale stays 1.0, "
            "i.e. K/V are cast straight to e4m3 with no range fitting. That is a DEFINED fallback, "
            "not a good one: an un-scaled store flushes everything below 2^-9 to zero, which on a "
            "typical V tensor (amax ~0.5) is a few percent of RMS. Fix by either (a) serving a "
            "checkpoint that carries quantization_config.kv_cache_scheme + self_attn.{k,v}_scale, "
            "or (b) running tools/kv_fp8_calibrate.py to write kv_scales.safetensors next to the "
            "weights (that path also gives PER-HEAD scales, which no checkpoint does)."
        )
        return

    # GLOBAL -> COMPACT pool index. `full_attn_layer_ids` is the identity [0..num_layers) for a
    # plain model and the exact compaction the hybrids use (GDN skips its linear layers, SWA sends
    # its sliding layers to the ring pool instead) — the same properties the model itself indexes
    # with, so this cannot drift from `laguna_layer_plan` / the GDN layer plan.
    layer_map = {
        None: model_config.full_attn_layer_ids,
        "swa": model_config.swa_layer_ids,
    }

    global_heads = model_config.num_kv_heads
    for pool, tag in pools:
        ids = layer_map[tag]
        assert len(ids) == pool.num_layers, (
            f"fp8-KV scale install: pool has {pool.num_layers} layers but the config maps "
            f"{len(ids)} global layers onto it"
        )
        missing = [gl for gl in ids if gl not in scaleset.scales]
        for compact, gl in enumerate(ids):
            got = scaleset.scales.get(gl)
            if got is None:
                continue
            k, v = got
            pool.set_fp8_kv_scales(
                compact,
                _shard_row(k, pool.num_kv_heads, global_heads),
                _shard_row(v, pool.num_kv_heads, global_heads),
            )
        name = "SWA ring" if tag == "swa" else "main"
        if missing:
            logger.warning_rank0(
                f"fp8-KV {name} pool: no scale for global layers {missing} — those layers keep "
                f"the identity scale 1.0 (uncalibrated)."
            )
        kd = pool.k_descale
        logger.info_rank0(
            f"fp8-KV {name} pool: installed "
            f"{'PER-HEAD' if scaleset.per_head else 'per-tensor (broadcast to heads)'} scales from "
            f"{scaleset.source} — {pool.num_layers} layers x {pool.num_kv_heads} heads, "
            f"k_descale range [{kd.min().item():.4g}, {kd.max().item():.4g}]"
        )
