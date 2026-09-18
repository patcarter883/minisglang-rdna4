"""The in-flight ceiling must be bounded by the POOL, not only by PCIe bytes.

WHY THIS EXISTS. `MINISGL_EXPERT_CACHE_MAX_INFLIGHT` was tuned to 512 at the validated operating
point (`--expert-cache-gb 2.5` -> 1923 slots), where it is 27% of the pool and healthy. On
2026-09-18 a spec run needed VRAM, the cache was moved to 1.0 GiB -- 769 slots -- and the 512
carried over untouched. An in-flight copy HOLDS A SLOT until it publishes, so 512 of 769 slots were
permanently in flight: the manager could free nothing, `promotions` and `evictions` stayed at 0,
`throttled` climbed past 15k, and the serve emitted 99 tokens and then none for eleven minutes while
`running_requests` sat at 1. Nothing raised -- the BYTE ceiling the pre-existing guard watches was
still being respected, so the launch looked entirely normal.

What is pinned here:
  T1  the validated point is NOT moved by the guard (a fix that shifts the shipped operating point
      is a regression, however well-intentioned) -- this is the one that matters;
  T2  the config that wedged is clamped instead;
  T3  the clamp cannot itself produce a degenerate ceiling on a small cache;
  T4  the guard only ever lowers.

CPU-only; no GPU and no kernel package. Run inside the serve image (the host lacks libmpi):

    PYTHONPATH=python python3 tests/expert_cache_inflight_slot_cap_test.py
"""

from __future__ import annotations

import os
import sys

FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'OK  ' if ok else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    if not ok:
        FAILED.append(name)


try:
    import torch
except Exception as e:  # noqa: BLE001
    print(f"SKIPPED: torch unavailable ({type(e).__name__}: {e})")
    sys.exit(0)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
from minisgl.weights.expert_cache import ExpertResidencyCache  # noqa: E402

ROW = 1024                      # bytes per expert row; only the slot ARITHMETIC matters here
CPU = torch.device("cpu")


def cache(slots: int, inflight: str) -> ExpertResidencyCache:
    os.environ["MINISGL_EXPERT_CACHE_MAX_INFLIGHT"] = inflight
    return ExpertResidencyCache(num_experts=slots * 2, expert_bytes=ROW,
                                budget_bytes=slots * ROW, device=CPU)


print("=" * 88)
print("Expert cache: the in-flight ceiling is bounded by the slot pool, not just by PCIe bytes")
print("=" * 88)

c = cache(1923, "512")          # the VALIDATED point: 2.5 GiB
check("T1 validated point (1923 slots) keeps MAX_INFLIGHT=512 verbatim",
      c.slots == 1923 and c._max_inflight == 512,
      f"slots={c.slots} max_inflight={c._max_inflight} (512 here is measured-good; clamping it "
      f"would move the shipped operating point)")

c = cache(769, "512")           # the config that WEDGED: 1.0 GiB
check("T2 the wedging config (769 slots) is clamped to slots//3",
      c.slots == 769 and c._max_inflight == 256,
      f"slots={c.slots} max_inflight={c._max_inflight}, want 256")
check("T2b clamped ceiling leaves the pool able to drain",
      c._max_inflight < c.slots // 2,
      f"{c._max_inflight} vs slots//2={c.slots // 2}")

c = cache(9, "512")
check("T3 tiny cache degrades to the floor of 8 rather than a degenerate ceiling",
      c._max_inflight == 8, f"max_inflight={c._max_inflight}")

c = cache(1923, "64")
check("T4 a ceiling already under the bound passes through untouched",
      c._max_inflight == 64, f"max_inflight={c._max_inflight}")

print()
if FAILED:
    print(f"FAILED ({len(FAILED)}): " + "; ".join(FAILED))
    sys.exit(1)
print("ALL CHECKS PASS")
