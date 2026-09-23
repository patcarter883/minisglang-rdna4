"""A CAPTURED BUCKET'S PADDED ROWS MUST NEVER REACH THE EXPERT CACHE'S OBSERVER.

    python3 tests/route_trace_capture_padding_test.py      # CPU-only; no GPU, no lease
    pytest -c /dev/null tests/route_trace_capture_padding_test.py

WHY THIS EXISTS. HIP/CUDA graph capture is BUCKETED: `GraphRunner.pad_batch` replays the smallest
captured bucket >= batch.size, appending `[dummy_req] * k` to make up the width, so a bs=4 graph
serving ONE request pushes three padded rows through routing. Those rows' hidden states are whatever
was left in the static capture buffer; the experts they land on were referenced by NO request. The
route ring is the expert cache's ONLY input (`RouteTracer.set_observer` -> `ExpertCache.observe`), so
admitting those experts would install slabs nothing reads and evict slabs something does — a cache
that is worse than inert while still holding its whole 2.5 GiB budget. `tools/serve.sh`'s
`GRAPH_BS=0` rationale names this as one of the two reasons capture was left off on the qwen4exp arm.

`record()` CANNOT fix this: under capture it sees only the bucket width (the Python runs once, at
capture; every later replay just re-executes the baked device write). The mask therefore lives on the
host at the step boundary, in `RouteTracer.harvest()`, against the REAL row count that
`Scheduler._step_boundary` plumbs through `begin_forward(num_rows=...)`. This file gates that.

WHAT IT ASSERTS, and why each half matters:
  * the observer sees EXACTLY the real rows' experts — equality, not "no poison". A mask that
    dropped everything, or kept the wrong END of the row, or was off by one row, also emits no
    poison; only equality separates "masked correctly" from "masked to nothing".
  * poison ids are chosen at the TOP of the expert range, including `num_experts - 1`, so they
    survive `drain()`'s `0 <= e < num_experts` filter. An out-of-range poison would be silently
    dropped by that filter and the test would pass with the mask removed.
  * per-layer ids differ, so a layer mix-up in the [num_layers, ring_width] stage is visible.

FALSIFICATION IS PART OF THE TEST, not a side note. `_variant()` re-execs the real
`route_trace.py` source with the mask surgically deleted (the anchor lines are asserted to exist, so
a source edit that moves them fails loudly instead of quietly stopping the falsification) and the
same scenario is re-run. The finding this produced: the truncation is enforced in TWO places, both
fed from the same row count — `harvest`'s `n = min(rows*top_k, ring_width)` and `drain`'s
`ncols = min(ntok*top_k, ring_width)` — so removing EITHER alone still blocks the poison. The gate is
only falsifiable by removing BOTH, which is what `both` does. That redundancy is a strength, but it
means a reviewer must not read `drain`'s bound as the pure drain-cost optimisation its comment
describes: it is also half of this correctness property.

NO GPU. Everything under test is host logic over tensors, so the tracer is built on
`device=cpu`. One seam needs care: `torch.cuda.is_current_stream_capturing()` RAISES
`AcceleratorError` on a host with no ROCm device, so `_capturing()` patches it — False for the eager
path, True to drive the branch a graph bakes.
"""
import os
import sys
import types
from contextlib import contextmanager

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "python"))

from minisgl.weights import route_trace as RT  # noqa: E402

RT_SRC = os.path.join(_HERE, "..", "python", "minisgl", "weights", "route_trace.py")

# ---- shape: the production one where it matters (top_k, num_experts), small elsewhere -----------
TOP_K = 10          # qwen4exp
E = 512             # qwen4exp
LAYERS = 4          # enough to catch a lid mix-up; the real arm has 48
DEV = torch.device("cpu")

# Real-request routing: all ids < 128. Per-layer offset so a lid mix-up shows up.
REAL_BASE = [3, 17, 29, 41, 53, 67, 79, 91, 103, 115]
# Padded-row routing: all ids >= 400, and row 1 starts at num_experts-1 so the poison is INSIDE the
# range drain() accepts. An out-of-range poison would be filtered for free and prove nothing.
POISON_BASE = [
    [511, 510, 509, 508, 507, 506, 505, 504, 503, 502],
    [499, 498, 497, 496, 495, 494, 493, 492, 491, 490],
    [489, 488, 487, 486, 485, 484, 483, 482, 481, 480],
    [479, 478, 477, 476, 475, 474, 473, 472, 471, 470],
    [469, 468, 467, 466, 465, 464, 463, 462, 461, 460],
    [459, 458, 457, 456, 455, 454, 453, 452, 451, 450],
    [449, 448, 447, 446, 445, 444, 443, 442, 441, 440],
]
POISON_FLOOR = 400          # nothing real is at or above this


def real_ids(lid: int, row: int) -> "list[int]":
    """Row `row` of layer `lid`, as a real request would route. Distinct per (lid, row)."""
    return [(b + 7 * lid + row) % 128 for b in REAL_BASE]


def poison_ids(lid: int, pad_row: int) -> "list[int]":
    return [p - lid for p in POISON_BASE[pad_row % len(POISON_BASE)]]


FAILS = []


def report(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}  {detail}")
    if not ok:
        FAILS.append(name)


# ---- the capture seam -------------------------------------------------------------------------
@contextmanager
def _capturing(flag: bool):
    """`torch.cuda.is_current_stream_capturing()` raises with no ROCm device, so it is patched.

    True drives the branch a graph BAKES (`_record_captured`); False is the eager path."""
    orig = torch.cuda.is_current_stream_capturing
    torch.cuda.is_current_stream_capturing = lambda: flag
    try:
        yield
    finally:
        torch.cuda.is_current_stream_capturing = orig


# ---- variant loader: the falsification ---------------------------------------------------------
_HARVEST_MASK = "            n = min(rows * self.top_k, self.ring_width)"
_DRAIN_BOUND = "                    ncols = min(max(int(ntok), 0) * self.top_k, self.ring_width)"
_MUTATIONS = {
    "harvest": [(_HARVEST_MASK, "            n = self.ring_width")],
    "drain": [(_DRAIN_BOUND, "                    ncols = self.ring_width")],
    "both": [(_HARVEST_MASK, "            n = self.ring_width"),
             (_DRAIN_BOUND, "                    ncols = self.ring_width")],
    # Not "removed" but WRONG BY ONE ROW, at both sites. This is the mutation a careless edit would
    # actually produce, and it proves the gate is sensitive to a one-row error rather than only to
    # total removal of the mask.
    "offbyone": [(_HARVEST_MASK, "            n = min((rows + 1) * self.top_k, self.ring_width)"),
                 (_DRAIN_BOUND,
                  "                    ncols = min((max(int(ntok), 0) + 1) * self.top_k, self.ring_width)")],
}


def _variant(which: "str | None"):
    """The real module, or a copy of its SOURCE with the named mask(s) deleted.

    Mutating the shipped source text (not a hand-written stand-in) is what makes this a
    falsification rather than a second implementation that could drift. Each anchor is asserted to
    appear exactly once, so an edit that moves it fails here instead of silently disarming the gate.
    """
    if which is None:
        return RT
    with open(RT_SRC) as fh:
        src = fh.read()
    for anchor, repl in _MUTATIONS[which]:
        if src.count(anchor) != 1:
            raise AssertionError(
                f"falsification anchor for {which!r} appears {src.count(anchor)} times, expected 1:\n"
                f"  {anchor!r}\nThe mask moved; this gate can no longer be made to fail and must be "
                f"re-anchored before it is trusted."
            )
        src = src.replace(anchor, repl)
    mod = types.ModuleType(f"route_trace_unmasked_{which}")
    mod.__file__ = RT_SRC
    exec(compile(src, f"{RT_SRC}[unmasked:{which}]", "exec"), mod.__dict__)
    return mod


def make(mod, *, ring_rows, ring_steps=8, layers=LAYERS):
    return mod.RouteTracer(
        None, model_slug="t", num_layers=layers, num_experts=E, top_k=TOP_K,
        tp_rank=0, dp_rank=0, expert_bytes=1, ring_steps=ring_steps, ring_rows=ring_rows,
        drain_every=ring_steps, max_steps=1_000_000, record_prefill=False,
        blockmap_checks=0, device=DEV,
    )


def bucket_rows(lid: int, real_rows: int, bucket: int) -> torch.Tensor:
    """The (bucket, top_k) topk_ids a REPLAY produces: real rows first, then padded rows.

    Real-first is the invariant the host mask rests on and it comes from `GraphRunner.pad_batch`:
    `batch.padded_reqs = batch.reqs + [self.dummy_req] * (padded_size - batch.size)`, and
    `DecodeCaptureBuffer.copy_from` fills the static buffers in that order.
    """
    rows = [real_ids(lid, r) for r in range(real_rows)]
    rows += [poison_ids(lid, p) for p in range(bucket - real_rows)]
    return torch.tensor(rows, dtype=torch.int32, device=DEV)


def replay_step(mod, tr, *, real_rows, bucket, uid=7, is_verify=False, layers=LAYERS):
    """One CAPTURED decode step as the engine runs it.

    `record()` is called with the BUCKET-wide tensor under `is_current_stream_capturing() == True`:
    that call is exactly the device write the graph records, and every replay re-executes it with the
    same bucket width and no Python. The host side (`begin_forward(num_rows=real)` then the next
    boundary's `harvest`) is what knows the step's real width.
    """
    tr.begin_forward(False, uid, is_verify=is_verify, num_rows=real_rows)
    with _capturing(True):
        for lid in range(layers):
            mod._CUR_LID = lid
            tr.record(bucket_rows(lid, real_rows, bucket), num_tokens=bucket)
    mod._CUR_LID = None


def eager_step(mod, tr, *, real_rows, uid=7, is_verify=False, layers=LAYERS):
    """One EAGER decode step: `record` under no capture, with only the real rows."""
    tr.begin_forward(False, uid, is_verify=is_verify, num_rows=real_rows)
    with _capturing(False):
        mod._CUR_CHUNK.clear()
        for lid in range(layers):
            mod._CUR_LID = lid
            tr.record(bucket_rows(lid, real_rows, real_rows), num_tokens=real_rows)
    mod._CUR_LID = None


def observe(mod, tr):
    """Drain and return {lid: set(ids)} as the OBSERVER saw it (not as the file recorded it)."""
    seen = {}

    def obs(lid, ids):
        seen.setdefault(lid, set()).update(ids)

    tr.set_observer(obs)
    with _capturing(False):
        tr.drain()
    return seen


# =================================================================================================
def scenario_bucket_gt_real(mod, *, bucket, real_rows, ring_rows=None, label=""):
    """Returns (seen, tracer). One captured step whose bucket is wider than the live batch."""
    tr = make(mod, ring_rows=ring_rows if ring_rows is not None else bucket)
    replay_step(mod, tr, real_rows=real_rows, bucket=bucket)
    return observe(mod, tr), tr


def check_clean(seen, *, real_rows, label, layers=LAYERS):
    """Every layer observed, EXACTLY its real rows' experts, no id from the padded band."""
    report(f"{label}: every layer reached the observer",
           sorted(seen) == list(range(layers)),
           f"got layers {sorted(seen)} -- a missing layer means the cache never learns it")
    poison_seen = {lid: sorted(i for i in ids if i >= POISON_FLOOR) for lid, ids in seen.items()}
    leaked = {lid: p for lid, p in poison_seen.items() if p}
    report(f"{label}: NO padded-row expert reached the observer", not leaked,
           f"leaked {leaked}" if leaked else "(the property under test)")
    bad = {}
    for lid in range(layers):
        want = set()
        for r in range(real_rows):
            want |= set(real_ids(lid, r))
        got = seen.get(lid, set())
        if got != want:
            bad[lid] = (sorted(got - want), sorted(want - got))
    report(f"{label}: the observer saw EXACTLY the real rows' experts", not bad,
           f"(extra, missing) per layer: {bad}" if bad
           else "equality, so a mask that dropped everything would fail here too")


print("== (1) PRIMARY GATE: bucket 4 replayed for ONE request ==")
seen, tr = scenario_bucket_gt_real(RT, bucket=4, real_rows=1)
check_clean(seen, real_rows=1, label="bucket 4 / real 1")
report("bucket 4 / real 1: the row count was plumbed (rows_unknown == 0)", tr.rows_unknown == 0,
       f"rows_unknown={tr.rows_unknown} -- nonzero means harvested UNMASKED")
report("bucket 4 / real 1: nothing fell to the host path", tr.rows_dropped == 0,
       f"rows_dropped={tr.rows_dropped}")
report("bucket 4 / real 1: no captured MoE call lost its layer id",
       tr.capture_unwrapped == 0, f"capture_unwrapped={tr.capture_unwrapped}")

print("== (2) the shipped arm's shape: max_running_req 2 -> bucket 2, one live request ==")
seen, tr = scenario_bucket_gt_real(RT, bucket=2, real_rows=1)
check_clean(seen, real_rows=1, label="bucket 2 / real 1")

print("== (3) a wide bucket with several real rows: bucket 8, real 3 ==")
seen, tr = scenario_bucket_gt_real(RT, bucket=8, real_rows=3)
check_clean(seen, real_rows=3, label="bucket 8 / real 3")

print("== (4) the ring may be WIDER than the bucket (ring_rows from max_running_req) ==")
# _route_trace_ring_rows takes max(max_running_req, cuda_graph_max_bs), so ring_width routinely
# exceeds the replayed bucket. The tail past the bucket is stage -1, which must not become a record.
seen, tr = scenario_bucket_gt_real(RT, bucket=4, real_rows=1, ring_rows=16)
check_clean(seen, real_rows=1, label="ring 16 / bucket 4 / real 1")

print("== (5) an unpadded replay (bucket == real) must still be reported in FULL ==")
seen, tr = scenario_bucket_gt_real(RT, bucket=4, real_rows=4)
check_clean(seen, real_rows=4, label="bucket 4 / real 4")

print("== (6) a WIDE step then a NARROW one in the same drain window ==")
# Slot reuse and stage reuse: the narrow step must not inherit the wide one's rows, and the wide
# step must not be truncated to the narrow one's width.
tr = make(RT, ring_rows=8)
replay_step(RT, tr, real_rows=3, bucket=8, uid=11)
replay_step(RT, tr, real_rows=1, bucket=8, uid=12)      # boundary harvests step 1
per_step = {}


def obs_step(lid, ids):
    per_step.setdefault(lid, []).append(sorted(ids))


tr.set_observer(obs_step)
with _capturing(False):
    tr.drain()
ok = True
detail = []
for lid in range(LAYERS):
    got = per_step.get(lid, [])
    want_wide = sorted(set(real_ids(lid, 0)) | set(real_ids(lid, 1)) | set(real_ids(lid, 2)))
    want_narrow = sorted(set(real_ids(lid, 0)))
    if got != [want_wide, want_narrow]:
        ok = False
        detail.append((lid, got, [want_wide, want_narrow]))
report("wide-then-narrow: each step reported its OWN rows, in order", ok,
       f"first mismatch {detail[:1]}" if detail else "")

print("== (7) an EAGER step between two captured ones stays exact (no stale stage) ==")
tr = make(RT, ring_rows=8)
order = []
tr.set_observer(lambda lid, ids: order.append((lid, sorted(ids))))
replay_step(RT, tr, real_rows=2, bucket=8, uid=21)
eager_step(RT, tr, real_rows=1, uid=22)
replay_step(RT, tr, real_rows=1, bucket=8, uid=23)
with _capturing(False):
    tr.drain()
got_l0 = [ids for lid, ids in order if lid == 0]
want_l0 = [
    sorted(set(real_ids(0, 0)) | set(real_ids(0, 1))),
    sorted(set(real_ids(0, 0))),
    sorted(set(real_ids(0, 0))),
]
report("captured / eager / captured all exact on layer 0", got_l0 == want_l0,
       f"got {got_l0}\n        want {want_l0}")
report("the eager step did not trip rows_mismatch", tr.rows_mismatch == 0,
       f"rows_mismatch={tr.rows_mismatch}")

print("== (8) the KNOWN HOLE, asserted so it cannot widen unnoticed: num_rows=None ==")
# `begin_forward(num_rows=None)` is honest-but-unmasked by design. It must (a) actually leak, and
# (b) be counted -- because that counter is the only thing standing between a caller that forgets to
# plumb the row count and a silently polluted cache.
tr = make(RT, ring_rows=4)
tr.begin_forward(False, 31, num_rows=None)
with _capturing(True):
    for lid in range(LAYERS):
        RT._CUR_LID = lid
        tr.record(bucket_rows(lid, 1, 4), num_tokens=4)
RT._CUR_LID = None
seen = observe(RT, tr)
leaked = any(i >= POISON_FLOOR for ids in seen.values() for i in ids)
report("num_rows=None leaks padding (documented) AND is counted",
       leaked and tr.rows_unknown == 1,
       f"leaked={leaked} rows_unknown={tr.rows_unknown} -- if this ever stops leaking, the "
       f"counter is no longer load-bearing and this test's premise changed")

print("== (9) the ONE production caller must plumb num_rows ==")
# (8) is only a hole if a caller can hit it. `Scheduler._step_boundary` is the single chokepoint all
# five forward/verify sites pass through, so this is a static check of that one call.
sched = open(os.path.join(_HERE, "..", "python", "minisgl", "scheduler", "scheduler.py")).read()
calls = sched.count("_route_trace.begin_forward(")
plumbed = sched.count("num_rows=sum(r.extend_len for r in batch.reqs)")
report("scheduler has exactly one begin_forward call and it passes num_rows",
       calls == 1 and plumbed == 1, f"calls={calls} with-num_rows={plumbed}")

print("== (10) a bucket too wide for the ring STOPS THE BOOT, it does not truncate ==")
# A truncated baked write would feed a partial union forever. It must raise at capture time.
tr = make(RT, ring_rows=2)
raised = None
try:
    with _capturing(True):
        RT._CUR_LID = 0
        tr.record(bucket_rows(0, 1, 8), num_tokens=8)
except RT.RouteTraceError as e:
    raised = str(e)
finally:
    RT._CUR_LID = None
report("an over-wide captured bucket raises RouteTraceError", raised is not None,
       (raised or "NO RAISE -- the graph would bake a truncated write")[:90])

print("== (11) an ALL-PADDING replay (zero real rows) must observe nothing at all ==")
# `_step_boundary` contemplates an empty batch (`batch.reqs[0].uid if batch.reqs else 0`) and
# `can_use_cuda_graph` would replay the bs=1 bucket for it, i.e. a step that is 100% padding. n == 0
# has to mean "no record", not "the whole width".
tr = make(RT, ring_rows=2)
replay_step(RT, tr, real_rows=0, bucket=2, uid=41)
seen = observe(RT, tr)
report("a zero-real-row captured step feeds the observer nothing", seen == {},
       f"observed {seen} -- every id there is a dummy request's routing")

print("== (12) the ring is sized to cover the widest captured bucket ==")
try:
    from minisgl.engine.engine import _route_trace_ring_rows
    from minisgl.engine.graph import _determine_cuda_graph_bs

    class _Cfg:
        def __init__(self, mrr, gmax):
            self.max_running_req = mrr
            self.cuda_graph_max_bs = gmax
            self.spec_config = None
            self.spec_num_draft = 0

    bad = []
    for mrr, gmax in ((2, None), (2, 2), (2, 0), (4, None), (8, 32), (256, None), (12, 12)):
        rows = _route_trace_ring_rows(_Cfg(mrr, gmax))
        buckets = _determine_cuda_graph_bs(None, gmax, 16 << 30, mrr)
        if buckets and max(buckets) > rows:
            bad.append((mrr, gmax, rows, max(buckets)))
    report("ring_rows >= the widest capture bucket for every config tried", not bad,
           f"under-wide: {bad}" if bad
           else "so (10) can only fire on a sizing bug, never on a normal serve")
except Exception as e:      # noqa: BLE001 - an import failure must not read as a pass
    report("ring_rows >= the widest capture bucket", False, f"could not check: {type(e).__name__}: {e}")

# =================================================================================================
print("\n== FALSIFICATION: re-run the primary gate against the source with the mask(s) removed ==")
for which in ("harvest", "drain", "both", "offbyone"):
    mod = _variant(which)
    seen, tr = scenario_bucket_gt_real(mod, bucket=4, real_rows=1)
    leaked = sorted({i for ids in seen.values() for i in ids if i >= POISON_FLOOR})
    print(f"  [{which:>8}] padded-row experts reaching the observer: "
          f"{leaked[:8]}{'...' if len(leaked) > 8 else ''} ({len(leaked)} ids)")
    if which == "both":
        report("removing BOTH masks makes the gate FAIL (poison arrives)", bool(leaked),
               "a gate that cannot fail is not a gate")
    elif which == "offbyone":
        report("a ONE-ROW-too-wide mask also makes the gate FAIL", bool(leaked),
               "so the gate is sensitive to an off-by-one, not just to deletion")
    else:
        # Not a defect -- a redundancy. Recorded because a reviewer reading `drain`'s comment as a
        # pure cost optimisation would not know it is also half of this correctness property.
        print(f"           -> removing only {which} does NOT leak: the mask is enforced twice, "
              f"both fed from the same row count.")

print(f"\n{'FAILED: ' + ', '.join(FAILS) if FAILS else 'ALL PASS'}")


def test_captured_padding_cannot_reach_observer():
    assert not FAILS, FAILS


if __name__ == "__main__":
    sys.exit(1 if FAILS else 0)
