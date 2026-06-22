"""Phase 3M-3 — full checkpoint<->model weight-map coverage for Qwen3.6-35B-A3B (qwen3_5_moe AWQ).

Replays the qwen3_5 GDN-hybrid loader's exact name/shape transforms over the REAL 95427-key
canonical (refs/main, AWQ) checkpoint header — reusing the loader's own helpers (`qwen3_5_remap`,
`_gate_up_merge`, `_get_expert_stack_info`) so the test can't drift from it:

  * skip vision (`model.visual.*`) + MTP (`mtp.*`); strip the `model.language_model.` LM prefix;
  * GDN in_proj concat (in_proj_qkv+z -> qkvz, in_proj_b+a -> ba, dim 0) + conv1d rename;
  * untied top-level lm_head;
  * MoE gate/up -> gate_up merge (shared expert dense .weight dim 0; routed experts AWQ
    .qweight/.qzeros/.scales dim 1) + per-expert stacking over E;

then asserts the produced {native_key: shape} set is EXACTLY model.state_dict()'s keys+shapes —
every served checkpoint tensor lands on a declared buffer, every buffer is filled, nothing leftover
(vision + MTP are intentionally dropped). dtype is not compared (F16 checkpoint loads into the
bf16/f16 compute + AWQ buffers via the engine's cast).

CPU-only; run in the combined ROCm image (no GPU lease):
    PYTHONPATH=/engine/python python /engine/tools/qwen3_5_moe_weight_map_test.py
"""
from __future__ import annotations

import glob
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig

from minisgl.distributed import set_tp_info
from minisgl.layers import set_rope_device
from minisgl.models import create_model
from minisgl.models.config import ModelConfig
from minisgl.models.weight import _gate_up_merge, _get_expert_stack_info, qwen3_5_remap
from minisgl.utils import torch_dtype

MODEL = "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit"
REPO = Path.home() / ".cache/huggingface/hub/models--cyankiwi--Qwen3.6-35B-A3B-AWQ-4bit"
_CAT_DIM1 = (".qweight", ".qzeros", ".scales")


def _canonical_shapes() -> dict[str, tuple[int, ...]]:
    snap = (REPO / "refs/main").read_text().strip()
    handles = [safe_open(f, framework="pt")
               for f in glob.glob(str(REPO / "snapshots" / snap / "*.safetensors"))]
    return {k: tuple(h.get_slice(k).get_shape()) for h in handles for k in h.keys()}


def simulate_loader(cfg: ModelConfig, shape_of: dict[str, tuple[int, ...]]) -> dict:
    """Replay _load_qwen3_5_weight's name/shape transforms over the header (TP=1)."""
    produced: dict[str, tuple[int, ...]] = {}
    concat_buf: dict[str, dict[int, tuple[int, ...]]] = {}
    merge_buf: dict[str, dict[str, tuple[int, ...]]] = {}
    expert_buf: dict[str, dict[int, tuple[int, ...]]] = {}

    def cat(shapes, dim):
        out = list(shapes[0])
        out[dim] = sum(s[dim] for s in shapes)
        return tuple(out)

    def emit(native: str, shape: tuple[int, ...]):
        if (mm := _gate_up_merge(native)) is not None:
            merged_key, slot = mm
            merge_buf.setdefault(merged_key, {})[slot] = shape
            if len(merge_buf[merged_key]) != 2:
                return
            parts = [merge_buf[merged_key][s] for s in ("gate", "up")]
            del merge_buf[merged_key]
            cat_dim = 1 if merged_key.endswith(_CAT_DIM1) else 0
            native, shape = merged_key, cat(parts, cat_dim)
        if cfg.is_moe and (einfo := _get_expert_stack_info(native)) is not None:
            packed_key, idx = einfo
            slots = expert_buf.setdefault(packed_key, {})
            slots[idx] = shape
            if len(slots) != cfg.num_experts:
                return
            del expert_buf[packed_key]
            produced[packed_key] = (cfg.num_experts, *shape)
        else:
            produced[native] = shape

    for name in sorted(shape_of):
        plan = qwen3_5_remap(name)
        if plan is None:
            continue
        if plan[0] == "direct":
            emit(plan[1], shape_of[name])
            continue
        _, merged, slot, n_slots, cat_dim = plan
        concat_buf.setdefault(merged, {})[slot] = shape_of[name]
        if len(concat_buf[merged]) != n_slots:
            continue
        parts = [concat_buf[merged][i] for i in range(n_slots)]
        del concat_buf[merged]
        emit(merged, cat(parts, cat_dim))

    assert not concat_buf, f"incomplete GDN concats: {list(concat_buf)}"
    assert not merge_buf, f"incomplete gate/up merges: {list(merge_buf)}"
    assert not expert_buf, f"incomplete expert stacks: {list(expert_buf)}"
    return produced


def main() -> None:
    cfg = ModelConfig.from_hf(AutoConfig.from_pretrained(MODEL))
    assert cfg.is_gdn_hybrid and cfg.is_moe and cfg.quant.is_awq, cfg

    set_tp_info(rank=0, size=1)
    set_rope_device(torch.device("cpu"))
    with torch.device("meta"), torch_dtype(torch.bfloat16):
        model = create_model(cfg)
    want = {k: tuple(v.shape) for k, v in model.state_dict().items()}

    shape_of = _canonical_shapes()
    nck = len(shape_of)
    nskip = sum(1 for k in shape_of if k.startswith(("model.visual.", "mtp.")))
    print(f"checkpoint {nck} tensors ({nskip} vision/MTP skipped); model declares {len(want)} buffers")
    got = simulate_loader(cfg, shape_of)

    missing = {k: want[k] for k in want.keys() - got.keys()}
    extra = {k: got[k] for k in got.keys() - want.keys()}
    mismatch = {k: (got[k], want[k]) for k in want.keys() & got.keys() if got[k] != want[k]}
    for label, d in (("MISSING (unfilled buffers)", missing),
                     ("EXTRA (unconsumed served keys)", extra),
                     ("SHAPE MISMATCH", mismatch)):
        if d:
            print(f"  {label}: {len(d)}")
            for k in sorted(d)[:10]:
                print(f"    {k}: {d[k]}")
    assert not missing and not extra and not mismatch, "weight map is not a bijection"

    nexp = sum(1 for k in got if ".experts." in k and k.endswith(".qweight"))
    print(f"[OK] served checkpoint <-> {len(want)} model buffers: exact bijection; shapes match "
          f"(incl. {nexp} stacked AWQ expert tensors, GDN in_proj concats, untied lm_head; "
          f"{nskip} vision/MTP tensors correctly skipped).")


if __name__ == "__main__":
    main()
