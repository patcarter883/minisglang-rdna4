"""Phase 3d-3 — CPU test for the Qwen3.5 GDN-hybrid weight-name mapping.

Proves that `qwen3_5_remap` (python/minisgl/models/weight.py) turns the REAL checkpoint key set
into EXACTLY the minisgl-native `state_dict()` the model exposes — no missing, no unexpected, with
matching shapes and dtypes — WITHOUT a GPU and WITHOUT materializing the 8 GB of weights:

  * checkpoint side: parse each safetensors file's header (8-byte length + JSON) for key/dtype/shape
    only (no tensor reads), then run `qwen3_5_remap` over every key, folding concat groups by
    summing the fused dim and applying the engine's dtype cast rule (A_log/dt_bias -> fp32, else bf16);
  * model side: build the 4B on meta exactly as the engine does (from the real HF config) and read
    its `state_dict()` keys/shapes/dtypes;
  * assert the two key sets are identical and every shape + dtype matches.

CPU-only; run in the combined ROCm image (healthy triton needed to import the GDN kernels):
    PYTHONPATH=/engine/python python /engine/tools/qwen3_5_weight_map_test.py
"""

from __future__ import annotations

import glob
import json
import struct
from dataclasses import replace
from typing import Dict, Tuple

import torch
from minisgl.distributed import set_tp_info
from minisgl.layers import set_rope_device
from minisgl.models import create_model
from minisgl.models.config import ModelConfig
from minisgl.models.weight import qwen3_5_remap
from minisgl.utils import cached_load_hf_config, download_hf_weight, torch_dtype

MODEL_PATH = "Qwen/Qwen3.5-4B"

# safetensors dtype string -> torch dtype (only the ones this checkpoint uses + common siblings)
_ST_DTYPE = {
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F16": torch.float16,
    "F64": torch.float64,
    "I64": torch.int64,
    "I32": torch.int32,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}


def _read_header(path: str) -> Dict[str, Tuple[torch.dtype, Tuple[int, ...]]]:
    """Parse a safetensors header (no tensor data) -> {key: (dtype, shape)}."""
    out: Dict[str, Tuple[torch.dtype, Tuple[int, ...]]] = {}
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
    for key, meta in header.items():
        if key == "__metadata__":
            continue
        out[key] = (_ST_DTYPE[meta["dtype"]], tuple(meta["shape"]))
    return out


def _cast_dtype(key: str, dtype: torch.dtype, model_dtype: torch.dtype) -> torch.dtype:
    """Mirror Engine._load_weight_state_dict._cast for dtype prediction."""
    if not dtype.is_floating_point or key.endswith(".scales"):
        return dtype
    if key.endswith((".A_log", ".dt_bias")):
        return torch.float32
    return model_dtype


def _predict_native(
    ckpt: Dict[str, Tuple[torch.dtype, Tuple[int, ...]]], model_dtype: torch.dtype
) -> Tuple[Dict[str, Tuple[torch.dtype, Tuple[int, ...]]], int]:
    """Run qwen3_5_remap over the checkpoint header -> predicted native {key: (dtype, shape)}.
    Concat groups fold by summing the fused dim. Returns (predicted, num_skipped)."""
    predicted: Dict[str, Tuple[torch.dtype, Tuple[int, ...]]] = {}
    # merged_key -> {slot: (dtype, shape)} plus its (n_slots, cat_dim)
    groups: Dict[str, Dict[int, Tuple[torch.dtype, Tuple[int, ...]]]] = {}
    group_meta: Dict[str, Tuple[int, int]] = {}
    skipped = 0
    for key, (dtype, shape) in ckpt.items():
        plan = qwen3_5_remap(key)
        if plan is None:
            skipped += 1
            continue
        if plan[0] == "direct":
            native = plan[1]
            assert native not in predicted, f"duplicate native key {native}"
            predicted[native] = (_cast_dtype(native, dtype, model_dtype), shape)
            continue
        _, merged, slot, n_slots, cat_dim = plan
        groups.setdefault(merged, {})[slot] = (dtype, shape)
        group_meta[merged] = (n_slots, cat_dim)
    for merged, slots in groups.items():
        n_slots, cat_dim = group_meta[merged]
        assert len(slots) == n_slots, f"incomplete concat group {merged}: {sorted(slots)}"
        dtypes = {slots[i][0] for i in range(n_slots)}
        assert len(dtypes) == 1, f"mixed dtypes in concat group {merged}: {dtypes}"
        base = list(slots[0][1])
        base[cat_dim] = sum(slots[i][1][cat_dim] for i in range(n_slots))
        predicted[merged] = (_cast_dtype(merged, slots[0][0], model_dtype), tuple(base))
    return predicted, skipped


def main() -> None:
    folder = download_hf_weight(MODEL_PATH)
    cfg = ModelConfig.from_hf(cached_load_hf_config(MODEL_PATH))
    assert cfg.is_gdn_hybrid, "config did not parse as a GDN hybrid"
    model_dtype = torch.bfloat16

    # --- checkpoint side: headers only, no tensor reads ---
    ckpt: Dict[str, Tuple[torch.dtype, Tuple[int, ...]]] = {}
    for f in glob.glob(f"{folder}/*.safetensors"):
        ckpt.update(_read_header(f))
    print(f"checkpoint keys: {len(ckpt)}")
    predicted, skipped = _predict_native(ckpt, model_dtype)
    print(f"  skipped (visual/mtp): {skipped}")
    print(f"  predicted native keys: {len(predicted)}")

    # --- model side: build on meta exactly as the engine does ---
    # NOTE: rope is non-parametric (cos/sin cache, never in state_dict), so it cannot affect the
    # weight-key layout under test. We null out rotary scaling here ONLY to dodge an unrelated rope
    # bug: from_hf maps Qwen3.5's `rope_parameters` (rope_type "default" + an mrope_section LIST)
    # into RotaryConfig.scaling, and AttentionLayer feeds tuple(scaling.items()) to the lru-cached
    # _get_rope -> "unhashable type: 'list'". That belongs to the rope/3d-4 sub-phase, not 3d-3.
    cfg = replace(cfg, rotary_config=replace(cfg.rotary_config, scaling=None))
    set_tp_info(rank=0, size=1)
    set_rope_device(torch.device("cpu"))
    with torch.device("meta"), torch_dtype(model_dtype):
        model = create_model(cfg)
    native = {k: (v.dtype, tuple(v.shape)) for k, v in model.state_dict().items()}
    print(f"model state_dict keys: {len(native)}")

    # --- compare key sets ---
    pred_keys, model_keys = set(predicted), set(native)
    missing = model_keys - pred_keys  # model wants, mapping didn't produce
    unexpected = pred_keys - model_keys  # mapping produced, model doesn't want
    assert not missing, f"{len(missing)} keys the model needs but the mapping does not produce:\n  " + \
        "\n  ".join(sorted(missing)[:20])
    assert not unexpected, f"{len(unexpected)} keys the mapping produces but the model does not want:\n  " + \
        "\n  ".join(sorted(unexpected)[:20])
    print(f"[OK] key sets identical ({len(native)} keys).")

    # --- compare shapes + dtypes ---
    shape_bad = {k: (predicted[k][1], native[k][1]) for k in native if predicted[k][1] != native[k][1]}
    dtype_bad = {k: (predicted[k][0], native[k][0]) for k in native if predicted[k][0] != native[k][0]}
    assert not shape_bad, "shape mismatches (predicted vs model):\n  " + \
        "\n  ".join(f"{k}: {p} != {m}" for k, (p, m) in list(shape_bad.items())[:20])
    assert not dtype_bad, "dtype mismatches (predicted vs model):\n  " + \
        "\n  ".join(f"{k}: {p} != {m}" for k, (p, m) in list(dtype_bad.items())[:20])
    print(f"[OK] all {len(native)} shapes + dtypes match.")

    # spot-check the load-bearing GDN concats + the fp32 gating params
    g = "model.layers.0.linear_attn."
    assert predicted[g + "in_proj_qkvz.weight"][1] == (2 * 2048 + 2 * 4096, 2560)
    assert predicted[g + "in_proj_ba.weight"][1] == (64, 2560)
    assert predicted[g + "conv1d_weight"][1] == (8192, 1, 4)
    assert predicted[g + "A_log"][0] == torch.float32
    assert predicted[g + "dt_bias"][0] == torch.float32
    assert predicted["model.layers.0.mlp.gate_up_proj.weight"][1] == (2 * 9216, 2560)
    # full-attn q/k/v stay separate (NOT fused into qkv_proj)
    assert "model.layers.3.self_attn.q_proj.weight" in predicted
    assert "model.layers.3.self_attn.qkv_proj.weight" not in predicted
    print("[OK] GDN concat shapes, fp32 gating dtypes, and split-QKV spot-checks pass.")
    print("\nPASS — Qwen3.5 weight-key mapping is exact on CPU.")


if __name__ == "__main__":
    main()
