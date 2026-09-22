#!/usr/bin/env python3
"""The free-slot target must track a STEP's admission demand, not a fixed constant.

WHAT THIS GATES. `_low_water` used to be both the free pool's DEPTH and, through
`_refill_batch`, the per-step install CEILING. Every admission holds its slot until the scheduler
publishes, so a 25-deep pool installed at most 25 experts per publish cycle no matter how many
were missed. MEASURED 2026-09-22 on qwen4exp MXFP4 at 2056 slots: ~290 misses/step against 25
installs/step, h=0.3026 live where this repo's own SLRU simulation gives 0.656 at the IDENTICAL
slot count, with `free=0` and `inflight=25` in every summary line and 0.08% churn over 1.57M
references. Promotions and evictions DID advance, which is why this read as "the cache does not
help" rather than as a defect — the fourth recurrence of a replacement freeze on this cache.

WHY IT IS A HOST TEST. `apply_pending` returns early on a non-CUDA device, so this policy was
unreachable without a GPU, and every one of this cache's failures has been a quiet one. The three
existing expert-cache tests all SKIP GREEN with rc=0 on a host without torch — presence is not
coverage. `_next_free_target` is pure and static precisely so this gate cannot become a skip.

    python3 tests/expert_cache_free_target_test.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))


def _target():
    """Import the pure helper WITHOUT importing torch (the module imports it at top level)."""
    import ast
    import textwrap

    src = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "python", "minisgl", "weights", "expert_cache.py")
    tree = ast.parse(open(src).read())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_next_free_target":
            # strip decorators; compile the function alone so no torch import is needed
            node.decorator_list = []
            mod = ast.Module(body=[node], type_ignores=[])
            ast.fix_missing_locations(mod)
            ns: dict = {"Tuple": tuple}
            exec(compile(mod, src, "exec"), ns)
            return ns["_next_free_target"]
    raise AssertionError("_next_free_target not found — was it renamed? This gate is now blind.")


def main() -> int:
    nxt = _target()
    LOW, CAP = 25, 257           # the live config: low_water 25, slots//8 of 2056
    fails = []

    def check(cond, msg):
        if not cond:
            fails.append(msg)

    # 1. FLOOR. With no demand the pool must sit exactly at the validated steady state, never
    #    below it — low_water=25 measured h=0.6314 against 1024's 0.5961.
    ewma, tgt = 0.0, LOW
    for _ in range(50):
        ewma, tgt = nxt(ewma, 0, LOW, CAP)
    check(tgt == LOW, f"idle target should rest at low_water {LOW}, got {tgt}")

    # 2. THE ACTUAL DEFECT. Sustained demand of ~290/step must raise the pool above 25. This is
    #    the assertion that fails against the old code, where the target WAS the constant 25.
    ewma, tgt = 0.0, LOW
    for _ in range(40):
        ewma, tgt = nxt(ewma, 290, LOW, CAP)
    check(tgt > LOW * 4, f"under 290/step demand the target must rise well above {LOW}, got {tgt}")
    check(tgt >= 250, f"target should approach observed demand (~290, capped {CAP}), got {tgt}")

    # 3. CEILING. A pathological step must not reclaim the residency out from under the cache.
    ewma, tgt = 0.0, LOW
    for _ in range(200):
        ewma, tgt = nxt(ewma, 100000, LOW, CAP)
    check(tgt == CAP, f"target must clamp at cap {CAP}, got {tgt}")

    # 4. SMOOTHING. One wide prefill step must not resize the decode pool. A single 5000-wide
    #    step from idle may lift the target, but nowhere near the spike.
    ewma, tgt = nxt(0.0, 5000, LOW, CAP)
    check(tgt < 1200, f"a single spike must not jump the target to the spike, got {tgt}")

    # 5. DECAY. When load stops the pool must come back down to the floor, or the cache
    #    permanently gives up residency it measured as worth holding.
    ewma, tgt = 0.0, LOW
    for _ in range(40):
        ewma, tgt = nxt(ewma, 290, LOW, CAP)
    high = tgt
    for _ in range(60):
        ewma, tgt = nxt(ewma, 0, LOW, CAP)
    check(tgt == LOW, f"target must decay back to {LOW} after load stops (peaked {high}), got {tgt}")

    # 6. MONOTONE IN DEMAND — a higher sustained rate must never ask for a shallower pool.
    def settle(rate):
        e, t = 0.0, LOW
        for _ in range(60):
            e, t = nxt(e, rate, LOW, CAP)
        return t
    seq = [settle(r) for r in (0, 10, 50, 120, 240)]
    check(all(b >= a for a, b in zip(seq, seq[1:])),
          f"target must be monotone non-decreasing in demand, got {seq}")

    for f in fails:
        print(f"  FAIL: {f}")
    print(f"{'FAILED' if fails else 'PASSED'}: expert-cache free-target "
          f"({6 - len(fails)}/6 checks)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
