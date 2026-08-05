"""Per-model table of the prefix-cache plan and every VRAM reservation that depends on it.

CPU-only (no GPU, no weights loaded — ModelConfig is built from the cached HF config). Proves the ONE
thing the engine/scheduler split used to get wrong: that the engine's snapshot-store RESERVATION and
the scheduler's cache-type DECISION are the same fact. Both columns below come from
engine/config.py::resolve_prefix_cache, so "AGREE" is structural, not coincidental — the table exists
to show the per-model ANSWERS are right, and that the GDN/CCA/SWA byte counts did not move (the
`wasGiB` column re-computes the PRE-FIX engine formula alongside the new one).

    PYTHONPATH=<worktree>/python python tools/prefix_cache_plan_table.py [--max-running N]
                                                                        [--cache-type naive] [--only X]
    MINISGL_SWA_RADIX=0 python tools/prefix_cache_plan_table.py     # the off switch
"""
from __future__ import annotations

import argparse
import os

import torch
from minisgl.distributed import DistributedInfo
from minisgl.engine import resolve_prefix_cache
from minisgl.engine.engine import Engine
from minisgl.scheduler.config import SchedulerConfig

MODELS = [
    ("gemma-4-26B (SWA, no recurrent)", "cyankiwi/gemma-4-26B-A4B-it-qat-AWQ-INT4", 2),
    ("Laguna-XS-2.1 (SWA)", "poolside/Laguna-XS-2.1-NVFP4", 2),
    ("Qwen3.6-35B (GDN hybrid)", "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit", 2),
    ("Qwen3.5-4B (GDN hybrid)", "cyankiwi/Qwen3.5-4B-AWQ-BF16-INT4", 1),
    ("ZAYA1-8B (CCA hybrid)", "/models/ZAYA1-8B-fp8", 1),
    ("GLM-4.7-Flash (MLA, dense-attn)", "QuantTrio/GLM-4.7-Flash-AWQ", 2),
]


class _Probe(Engine):
    """Engine byte-accounting methods without building an engine (no GPU, no weights)."""

    def __init__(self, kv_dtype: torch.dtype) -> None:  # noqa: D107 — deliberately not super().__init__
        self.kv_dtype = kv_dtype


def baseline_snap_bytes(probe, cfg) -> int:
    """The PRE-FIX engine reservation, reproduced verbatim (engine.py@2feed6b1 _rec_snapshot_store_bytes).

    Kept here so "no change for GDN/CCA" is a measured equality, not an argument about a diff. Only the
    GATE moved; the sizing arithmetic below is byte-for-byte the code that is still in engine.py.
    """
    if not getattr(cfg, "gdn_radix", True):
        return 0
    per_slot = probe._recurrent_state_bytes(cfg, replay_ring=False) // max(1, cfg.max_running_req + 2)
    if per_slot <= 0:
        return 0
    live = probe._rec_snap_live_snapshots(cfg)
    env_gib = os.environ.get("MINISGL_GDN_RADIX_SNAP_BUDGET_GIB")
    budget = int(float(env_gib) * (1 << 30)) if env_gib else 0
    return max(budget, live * per_slot)


def row(name: str, path: str, tp: int, max_running: int, kv_dtype: torch.dtype,
        cache_type: str = "radix"):
    cfg = SchedulerConfig(
        model_path=path,
        tp_info=DistributedInfo(rank=0, size=tp),
        dtype=torch.bfloat16,
        max_running_req=max_running,
        page_size=16,
        cache_type=cache_type,
    )
    mc = cfg.model_config
    plan = resolve_prefix_cache(cfg)
    probe = _Probe(kv_dtype)
    return {
        "model": name,
        "recurrent": bool(mc.is_gdn_hybrid or mc.is_cca_hybrid),
        "swa": bool(mc.is_swa_hybrid),
        "gdn_radix": bool(cfg.gdn_radix),
        "SWA_RADIX": os.environ.get("MINISGL_SWA_RADIX", "(unset)"),
        "cache_type": plan.cache_type,
        "snap_kind": plan.snapshot_kind or "-",
        "snap_bytes": probe._rec_snapshot_store_bytes(cfg),
        "old_snap_bytes": baseline_snap_bytes(probe, cfg),
        # What the SCHEDULER will cap the LRU at, and the per-entry size that cap multiplies. Printed
        # here so the reconciliation (cap x per == engine reservation) is checkable on CPU, before any
        # boot; the boot log then prints the same two numbers from the live tensors.
        "cap": probe._rec_snap_live_snapshots(cfg) if plan.snapshot_kind else 0,
        "state_bytes": probe._recurrent_state_bytes(cfg),
        "reason": plan.reason,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-running", type=int, default=6)
    ap.add_argument("--only", default="")
    ap.add_argument("--cache-type", default="radix")
    a = ap.parse_args()

    hdr = (f"{'model':34} {'rec':>5} {'swa':>5} {'gdnrx':>6} {'SWA_RADIX':>9} "
           f"{'cache_type':>16} {'snap':>10} {'cap':>4} {'perMiB':>8} "
           f"{'snapGiB':>8} {'wasGiB':>8} {'stateGiB':>9}  agree?")
    print(f"[max_running={a.max_running} cache_type={a.cache_type!r} "
          f"MINISGL_SWA_RADIX={os.environ.get('MINISGL_SWA_RADIX', '(unset)')}]")
    print(hdr)
    print("-" * len(hdr))
    for name, path, tp in MODELS:
        if a.only and a.only not in path:
            continue
        try:
            r = row(name, path, tp, a.max_running, torch.bfloat16, a.cache_type)
        except Exception as e:  # noqa: BLE001 — an uncached model must not abort the table
            print(f"{name:34} SKIP ({type(e).__name__}: {str(e)[:60]})")
            continue
        # THE invariant: a snapshot store is reserved iff the scheduler will build a cache that fills
        # it. cache_type 'recurrent_radix' <=> snap_bytes > 0.
        will_store = r["cache_type"] == "recurrent_radix"
        agree = will_store == (r["snap_bytes"] > 0)
        delta = "same" if r["snap_bytes"] == r["old_snap_bytes"] else "CHANGED"
        per = r["snap_bytes"] / r["cap"] / (1 << 20) if r["cap"] else 0.0
        print(
            f"{r['model']:34} {str(r['recurrent']):>5} {str(r['swa']):>5} {str(r['gdn_radix']):>6} "
            f"{r['SWA_RADIX']:>9} {r['cache_type']:>16} {r['snap_kind']:>10} "
            f"{r['cap']:>4} {per:8.1f} "
            f"{r['snap_bytes']/(1<<30):8.3f} {r['old_snap_bytes']/(1<<30):8.3f} "
            f"{r['state_bytes']/(1<<30):9.3f}  "
            f"{'AGREE' if agree else '*** DISAGREE ***'} {delta:8} [{r['reason']}]"
        )


if __name__ == "__main__":
    main()
