"""Phase 2M-3 — full checkpoint<->model weight-map coverage for Qwen1.5-MoE-A2.7B-GPTQ-Int4.

Ties the REAL 22539-key GPTQ checkpoint to the REAL model graph with NO GPU and NO weight
materialization beyond tiny bias tensors. It replays the streaming loader's exact key transforms
(reusing the loader's own helpers so the test can't drift from it):

  * skip vision/projector + `.g_idx` (desc_act=False) + all-zero `.bias` placeholders;
  * merge q/k/v -> qkv_proj and gate/up -> gate_up_proj (cat dim 1 for .qweight/.qzeros/.scales,
    dim 0 for .weight/.bias);
  * stack the 60 per-expert tensors into the grouped (E, ...) runtime buffers,

then asserts the produced {native_key: shape} set is EXACTLY model.state_dict()'s keys+shapes —
every checkpoint tensor lands on a declared buffer, every buffer is filled, nothing is left over.
(dtype is intentionally not compared: F16 checkpoint scales/gates load into the bf16/f16 compute
buffers via the engine's cast.)

CPU-only; run in the combined ROCm image (no GPU lease):
    PYTHONPATH=/engine/python python /engine/tools/qwen2_moe_weight_map_test.py
"""
from __future__ import annotations

import glob

import torch
from safetensors import safe_open
from transformers import AutoConfig

from minisgl.distributed import set_tp_info
from minisgl.layers import set_rope_device
from minisgl.models import create_model
from minisgl.models.config import ModelConfig
from minisgl.models.weight import _get_expert_stack_info, _get_merge_info
from minisgl.utils import torch_dtype

MODEL = "Qwen/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4"
MODEL_GLOB = (
    "/root/.cache/huggingface/hub/models--Qwen--Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4/"
    "snapshots/*/*.safetensors"
)
_CAT_DIM1 = (".qweight", ".qzeros", ".scales")


def simulate_loader(cfg: ModelConfig) -> dict[str, tuple[int, ...]]:
    """Replay load_weight's name/shape transforms over the checkpoint header (TP=1)."""
    handles = [safe_open(f, framework="pt") for f in glob.glob(MODEL_GLOB)]
    shape_of = {k: tuple(h.get_slice(k).get_shape()) for h in handles for k in h.keys()}
    get = {k: h for h in handles for k in h.keys()}

    produced: dict[str, tuple[int, ...]] = {}
    merge_buf: dict[str, dict[str, tuple[int, ...]]] = {}
    expert_buf: dict[str, dict[int, tuple[int, ...]]] = {}

    for name in sorted(shape_of):  # deterministic; order doesn't affect the merge/stack result
        if name.startswith(("vision_tower.", "multi_modal_projector.")):
            continue
        if name.endswith(".g_idx"):
            continue
        if name.endswith(".bias") and not bool(get[name].get_tensor(name).any()):
            continue
        shape = shape_of[name]  # TP=1 -> sharding is identity

        if (info := _get_merge_info(name)) is None:
            out_key, out_shape = name, shape
        else:
            merged_key, slot, all_slots = info
            merge_buf.setdefault(merged_key, {})[slot] = shape
            if not all(s in merge_buf[merged_key] for s in all_slots):
                continue
            parts = [merge_buf[merged_key][s] for s in all_slots]
            del merge_buf[merged_key]
            cat_dim = 1 if merged_key.endswith(_CAT_DIM1) else 0
            merged_shape = list(parts[0])
            merged_shape[cat_dim] = sum(p[cat_dim] for p in parts)
            out_key, out_shape = merged_key, tuple(merged_shape)

        if cfg.is_moe and (einfo := _get_expert_stack_info(out_key)) is not None:
            packed_key, idx = einfo
            slots = expert_buf.setdefault(packed_key, {})
            slots[idx] = out_shape
            if len(slots) != cfg.num_experts:
                continue
            del expert_buf[packed_key]
            produced[packed_key] = (cfg.num_experts, *out_shape)
        else:
            produced[out_key] = out_shape

    assert not merge_buf, f"incomplete merges: {list(merge_buf)}"
    assert not expert_buf, f"incomplete expert stacks: {list(expert_buf)}"
    return produced


def main() -> None:
    cfg = ModelConfig.from_hf(AutoConfig.from_pretrained(MODEL))
    assert cfg.is_moe and cfg.quant is not None and cfg.quant.is_gptq, cfg

    set_tp_info(rank=0, size=1)
    set_rope_device(torch.device("cpu"))
    with torch.device("meta"), torch_dtype(torch.bfloat16):
        model = create_model(cfg)
    want = {k: tuple(v.shape) for k, v in model.state_dict().items()}

    print(f"checkpoint -> simulating loader; model declares {len(want)} buffers")
    got = simulate_loader(cfg)

    missing = {k: want[k] for k in want.keys() - got.keys()}   # buffers no ckpt tensor fills
    extra = {k: got[k] for k in got.keys() - want.keys()}      # produced keys with no buffer
    mismatch = {k: (got[k], want[k]) for k in want.keys() & got.keys() if got[k] != want[k]}
    for label, d in (("MISSING (unfilled buffers)", missing),
                     ("EXTRA (unconsumed ckpt keys)", extra),
                     ("SHAPE MISMATCH", mismatch)):
        if d:
            print(f"  {label}: {len(d)}")
            for k in sorted(d)[:8]:
                print(f"    {k}: {d[k]}")
    assert not missing and not extra and not mismatch, "weight map is not a bijection"

    nexp = sum(1 for k in got if ".experts." in k and k.endswith(".qweight"))
    print(f"[OK] {len(got)} checkpoint groups <-> {len(want)} model buffers, exact bijection; "
          f"shapes match (incl. {nexp} stacked grouped-GPTQ expert tensors; g_idx + zero-bias "
          f"placeholders correctly skipped).")


if __name__ == "__main__":
    main()
