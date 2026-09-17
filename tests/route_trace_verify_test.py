"""A speculative VERIFY's routings must reach the expert cache's observer.

    python tests/route_trace_verify_test.py      # needs a GPU (the ring is a device tensor)

WHY THIS EXISTS. Under MTP, EVERY target forward is a verify with M = bs*(K+1) rows. `RouteTrace`'s
sync-free ring path was gated on `M == 1`, so under spec it never fired once: no record reached
`drain()`, `drain()` never called the observer, and the expert cache sat at fill=0.000 holding its
entire budget (2.5 GiB on the shipped qwen4exp arm) for literally zero hits. Nothing failed, nothing
logged; "spec with the expert cache on" simply measured an inert cache.

Two independent gates had to be wrong at once, which is why it survived:
  1. the verify sites passed `trace=False`, so `_cur_kind` kept the previous forward's value -- a
     PREFILL on a spec serve -- and `record` early-returned before the ring;
  2. even had it recorded, M>1 goes to the HOST path, whose records land in `self.oversize`, which
     `drain()` writes to the trace file but NEVER forwards to the observer.

So this test asserts the end-to-end property rather than either half: feed a verify-shaped batch,
and the observer must see the UNION of all its rows.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))

from minisgl.weights import route_trace as RT  # noqa: E402

FAILS = []


def report(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}  {detail}")
    if not ok:
        FAILS.append(name)


def make(top_k=4, rows=8, layers=2, experts=64, ring_steps=16):
    return RT.RouteTracer(
        None, model_slug="t", num_layers=layers, num_experts=experts, top_k=top_k,
        tp_rank=0, dp_rank=0, expert_bytes=1, ring_steps=ring_steps, ring_rows=rows,
        # drain_every <= ring_steps is enforced by the tracer (a larger value would wrap the
        # ring before it is read); this test calls drain() explicitly after each forward anyway.
        drain_every=ring_steps, max_steps=1_000_000, record_prefill=False,
        blockmap_checks=0, device=torch.device("cuda"),
    )


def feed(tr, ids_2d, *, is_verify, lid=0):
    """One forward with one MoE layer, shaped (M, top_k)."""
    seen = []
    tr.set_observer(lambda l, ids: seen.append((l, tuple(ids))))
    tr.begin_forward(False, 1, is_verify=is_verify)
    RT._CUR_LID = lid
    RT._CUR_CHUNK.clear()
    tr.record(torch.tensor(ids_2d, dtype=torch.int32, device="cuda"), num_tokens=len(ids_2d))
    tr.drain()
    return seen


print("== a verify's rows must reach the observer, as their UNION ==")
tr = make()
# 3 rows, disjoint experts: the union is all nine. A row-0-only read would report just {1,2,3}.
seen = feed(tr, [[1, 2, 3, 1], [4, 5, 6, 4], [7, 8, 9, 7]], is_verify=True)
report("the observer was called at all", len(seen) == 1,
       f"{len(seen)} call(s) -- 0 means the ring rejected the verify, the original bug")
if seen:
    got = set(seen[0][1])
    report("it saw the UNION of every row, not row 0", got == {1, 2, 3, 4, 5, 6, 7, 8, 9},
           f"got {sorted(got)}")
report("no verify row fell to the host path", tr.rows_dropped == 0,
       f"rows_dropped={tr.rows_dropped}")

print("== a plain decode still works (M == 1), unchanged ==")
tr = make()
seen = feed(tr, [[5, 6, 7, 5]], is_verify=False)
report("decode still reaches the observer", len(seen) == 1 and set(seen[0][1]) == {5, 6, 7},
       str(seen))

print("== a prefill must still NOT reach the observer ==")
tr = make()
seen = feed(tr, [[1, 2, 3, 4]] * 3, is_verify=False)
tr.begin_forward(True, 1)          # prefill
RT._CUR_LID = 0
RT._CUR_CHUNK.clear()
before = len(seen)
tr.record(torch.tensor([[11, 12, 13, 14]], dtype=torch.int32, device="cuda"), num_tokens=1)
tr.drain()
report("prefill routings are still withheld from the policy", len(seen) == before,
       "a prefill touches nearly every expert; feeding it would look like one enormous sweep")

print("== a verify too wide for the ring is COUNTED, not silently truncated ==")
tr = make(rows=2)                   # ring holds 2 rows; feed 4
feed(tr, [[1, 2, 3, 4]] * 4, is_verify=True)
report("an oversized verify increments rows_dropped", tr.rows_dropped >= 1,
       f"rows_dropped={tr.rows_dropped} -- a partial union makes the cache look better than it is")

print("== stale ids from a wider previous step must not leak into a narrower one ==")
tr = make(rows=4)
feed(tr, [[20, 21, 22, 23], [24, 25, 26, 27]], is_verify=True)   # wide step
seen = feed(tr, [[1, 2, 3, 4]], is_verify=True)                   # narrow step, SAME slot width
if seen:
    got = set(seen[-1][1])
    report("the narrow step reports only its own ids", got == {1, 2, 3, 4}, f"got {sorted(got)}")

print(f"\n{'FAILED: ' + ', '.join(FAILS) if FAILS else 'ALL PASS'}")
sys.exit(1 if FAILS else 0)
