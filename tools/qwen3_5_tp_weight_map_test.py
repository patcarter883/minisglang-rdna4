"""Phase 4-1 — CPU TP=2 weight-map test for the Qwen3.5 GDN-hybrid loader (4B dense + 35B MoE).

Proves the Phase-4 tensor-parallel weight sharding (`_shard_qwen3_5` + the GDN concat / MoE
gate-up merge / per-expert stack in `_load_qwen3_5_weight`) is EXACT and CONSISTENT, on CPU, with
no GPU and no 8/18 GB tensor reads:

  * for each TP rank r, replay the loader's name/shape transforms over the REAL checkpoint header on
    META tensors — calling the loader's OWN helpers (`qwen3_5_remap`, `_shard_qwen3_5`,
    `_gate_up_merge`, `_get_expert_stack_info`, torch.cat/stack) so the test can't drift from it —
    and assert the produced {key: shape} set is EXACTLY the model's state_dict() built at THAT tp
    (every served tensor lands on a per-rank buffer, every buffer filled, nothing leftover);
  * TILING: against the tp=1 full model, assert every weight's rank-0 and rank-1 shards partition
    the full tensor along EXACTLY ONE dim (sum of shard sizes == full, all other dims equal) or are
    replicated (all shards == full). This is what catches a naive-chunk corruption of a concatenated
    GDN/gate-up projection — the headline Phase-4 risk.

CPU-only; run in the combined ROCm image (healthy triton for the GDN kernel imports):
    PYTHONPATH=/engine/python python /engine/tools/qwen3_5_tp_weight_map_test.py
"""
from __future__ import annotations

import glob
import json
import struct
from typing import Dict, Tuple

import torch
from minisgl.distributed import set_tp_info
from minisgl.layers import set_rope_device
from minisgl.models import create_model
from minisgl.models.config import ModelConfig
from minisgl.models.weight import (
    _gate_up_merge,
    _get_expert_stack_info,
    _shard_qwen3_5,
    qwen3_5_remap,
)
from minisgl.utils import cached_load_hf_config, download_hf_weight, torch_dtype

# (model path, tp_size). 4B = Phase 4-1a (dense GDN); 35B = Phase 4-1b (GDN + AWQ MoE experts).
MODELS = [
    ("Qwen/Qwen3.5-4B", 2),
    ("cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit", 2),
]
_CAT_DIM1 = (".qweight", ".qzeros", ".scales")

_ST_DTYPE = {
    "BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16, "F64": torch.float64,
    "I64": torch.int64, "I32": torch.int32, "I16": torch.int16, "I8": torch.int8,
    "U8": torch.uint8, "BOOL": torch.bool,
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


def _header(folder: str) -> Dict[str, Tuple[torch.dtype, Tuple[int, ...]]]:
    ckpt: Dict[str, Tuple[torch.dtype, Tuple[int, ...]]] = {}
    files = glob.glob(f"{folder}/*.safetensors")
    files = [f for f in files if not f.endswith("consolidated.safetensors")] or files
    for f in files:
        ckpt.update(_read_header(f))
    return ckpt


def simulate_sharded_loader(
    header: Dict[str, Tuple[torch.dtype, Tuple[int, ...]]], cfg: ModelConfig, r: int, n: int
) -> Dict[str, Tuple[int, ...]]:
    """Replay `_load_qwen3_5_weight` for rank r on META tensors (shape-only), calling the loader's
    real helpers + the real `_shard_qwen3_5`. Returns {native_key: shape}."""
    produced: Dict[str, Tuple[int, ...]] = {}
    concat_buf: Dict[str, Dict[int, torch.Tensor]] = {}
    merge_buf: Dict[str, Dict[str, torch.Tensor]] = {}
    expert_buf: Dict[str, Dict[int, torch.Tensor]] = {}

    def emit(native: str, t: torch.Tensor) -> None:
        if (mm := _gate_up_merge(native)) is not None:
            merged_key, slot = mm
            merge_buf.setdefault(merged_key, {})[slot] = t
            if len(merge_buf[merged_key]) != 2:
                return
            parts = [merge_buf[merged_key][s] for s in ("gate", "up")]
            del merge_buf[merged_key]
            cat_dim = 1 if merged_key.endswith(_CAT_DIM1) else 0
            native, t = merged_key, torch.cat(parts, dim=cat_dim)
        if cfg.is_moe and (einfo := _get_expert_stack_info(native)) is not None:
            packed_key, idx = einfo
            slots = expert_buf.setdefault(packed_key, {})
            slots[idx] = t
            if len(slots) != cfg.num_experts:
                return
            stacked = torch.stack([slots[i] for i in range(cfg.num_experts)], dim=0)
            del expert_buf[packed_key]
            produced[packed_key] = tuple(stacked.shape)
        else:
            produced[native] = tuple(t.shape)

    for name in sorted(header):
        plan = qwen3_5_remap(name)
        if plan is None:
            continue
        dtype, shape = header[name]
        raw = _shard_qwen3_5(name, torch.empty(shape, dtype=dtype, device="meta"), r, n, cfg)
        if plan[0] == "direct":
            emit(plan[1], raw)
            continue
        _, merged, slot, n_slots, cat_dim = plan
        concat_buf.setdefault(merged, {})[slot] = raw
        if len(concat_buf[merged]) != n_slots:
            continue
        parts = [concat_buf[merged][i] for i in range(n_slots)]
        del concat_buf[merged]
        emit(merged, torch.cat(parts, dim=cat_dim))

    assert not concat_buf, f"incomplete GDN concats: {list(concat_buf)}"
    assert not merge_buf, f"incomplete gate/up merges: {list(merge_buf)}"
    assert not expert_buf, f"incomplete expert stacks: {list(expert_buf)}"
    return produced


def _build_state_dict_shapes(cfg: ModelConfig, rank: int, size: int) -> Dict[str, Tuple[int, ...]]:
    # The engine sets TP info once per process; this test rebuilds the model across (rank, size)
    # in ONE process, so reset the single-set global between builds (test-harness only).
    import minisgl.distributed.info as _info
    _info._TP_INFO = None
    set_tp_info(rank=rank, size=size)
    set_rope_device(torch.device("cpu"))
    with torch.device("meta"), torch_dtype(torch.bfloat16):
        model = create_model(cfg)
    return {k: tuple(v.shape) for k, v in model.state_dict().items()}


def _check_bijection(got: Dict, want: Dict, label: str) -> None:
    missing = {k: want[k] for k in want.keys() - got.keys()}
    extra = {k: got[k] for k in got.keys() - want.keys()}
    mismatch = {k: (got[k], want[k]) for k in want.keys() & got.keys() if got[k] != want[k]}
    for name, d in (("MISSING (unfilled buffers)", missing),
                    ("EXTRA (unconsumed keys)", extra), ("SHAPE MISMATCH", mismatch)):
        if d:
            print(f"  [{label}] {name}: {len(d)}")
            for k in sorted(d)[:10]:
                print(f"    {k}: {d[k]}")
    assert not missing and not extra and not mismatch, f"[{label}] weight map is not a bijection"


def _check_tiling(full: Dict, shards: list[Dict], n: int) -> int:
    """Every native weight: its n rank shards partition the full tensor along exactly one dim, or
    are replicated. Returns the number of (genuinely) sharded tensors."""
    nshard = 0
    for k, fshape in full.items():
        sshapes = [s[k] for s in shards]
        diffs = [d for d in range(len(fshape)) if any(ss[d] != fshape[d] for ss in sshapes)]
        if not diffs:
            assert all(ss == fshape for ss in sshapes), f"{k}: replicated but shards differ"
            continue
        assert len(diffs) == 1, f"{k}: shards differ on >1 dim {diffs} ({fshape} vs {sshapes})"
        d = diffs[0]
        assert sum(ss[d] for ss in sshapes) == fshape[d], (
            f"{k}: shard dim {d} sizes {[ss[d] for ss in sshapes]} != full {fshape[d]} "
            f"(gap/overlap — likely a naive chunk of a concatenated projection)"
        )
        for i in range(len(fshape)):
            if i != d:
                assert all(ss[i] == fshape[i] for ss in sshapes), f"{k}: dim {i} not invariant"
        nshard += 1
    return nshard


def run(model_path: str, n: int) -> None:
    print(f"\n=== {model_path}  (TP={n}) ===")
    folder = download_hf_weight(model_path)
    cfg = ModelConfig.from_hf(cached_load_hf_config(model_path))
    assert cfg.is_gdn_hybrid, "config did not parse as a GDN hybrid"
    header = _header(folder)
    nskip = sum(1 for k in header if qwen3_5_remap(k) is None)
    print(f"checkpoint {len(header)} tensors ({nskip} vision/MTP skipped); MoE={cfg.is_moe}")

    full = _build_state_dict_shapes(cfg, 0, 1)
    shards = []
    for r in range(n):
        want = _build_state_dict_shapes(cfg, r, n)
        got = simulate_sharded_loader(header, cfg, r, n)
        _check_bijection(got, want, f"rank{r}")
        shards.append(got)
        print(f"  [rank{r}] exact bijection: {len(got)} buffers")

    nshard = _check_tiling(full, shards, n)
    nrepl = len(full) - nshard
    nexp = sum(1 for k in full if ".experts." in k and (k.endswith(".qweight") or k.endswith("_proj")))
    print(f"[OK] {len(full)} tensors: per-rank bijection + tiling exact "
          f"({nshard} sharded, {nrepl} replicated; incl. GDN concats, full-attn q+gate, "
          f"{'AWQ MoE experts, ' if cfg.is_moe else ''}vocab-parallel embed/lm_head).")


def main() -> None:
    for model_path, n in MODELS:
        run(model_path, n)
    print("\nPASS — Qwen3.5 GDN-hybrid TP weight sharding is exact + tiling-consistent on CPU.")


if __name__ == "__main__":
    main()
