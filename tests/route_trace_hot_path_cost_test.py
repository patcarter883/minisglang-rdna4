"""HOT-PATH BUDGET for `RouteTracer.record`, plus the two cost regressions the capture fix can hide.

    # CPU only. No GPU, no lease, nothing touches /dev/kfd.
    docker run --rm --entrypoint bash -v /home/pat/code/minisgl-rdna4-captsafe:/engine:ro \
      minisgl-gate:pytest -lc 'PYTHONPATH=/engine/python:/opt/kernels python3 -m pytest -q \
        -c /dev/null -o python_files="*_test.py" /engine/tests/route_trace_hot_path_cost_test.py'

`record` is called ONCE PER MoE LAYER PER FORWARD -- 48x per decode step on the shipped qwen4exp arm
-- so its op count is decode-path launch overhead. A parity test proves values, not cost, and the
capture-safety fix rewrote exactly this function: the ring write became a stage write, the per-row
stale-tail clear moved out to a once-per-step `harvest`, the per-layer `meta[slot][lid]` dict store
went away, and a row-count compare came in.

WHAT IS ASSERTED, AND WHY IT IS AN OP COUNT RATHER THAN A TIME. On the card each of these aten calls
is a dispatch and a kernel launch; on a CPU fixture they are microseconds of interpreter work. The
COUNT is the part that transfers -- one fewer op per layer is one fewer launch per layer on any
device -- so the count is what is gated here. The wall-clock A/B against the pre-fix module lives in
`tests/route_trace_hot_path_bench.py`, which needs a pre-fix checkout and so cannot be a unit test.

MEASURED 2026-09-23 in minisgl-gate:pytest (torch 2.15.0.dev20260827+rocm7.2, CPU, qwen4exp shapes:
48 layers / 512 experts / top_k 10 / ring_rows 2):

    record(), ring path, M=1   5 aten ops   view, slice, slice, select, copy_
    record(), ring path, M=2   4 aten ops   (the source `[:n]` slice is elided at full width)
    harvest(), per STEP        8 aten ops   select+slice+copy_, slice+select+fill_, fill_
    ------------------------------------------------------------------------------------
    per 48-layer decode step   248 ops      vs 240 pre-fix (ring_rows=1) and 480 width-matched

TIMED, same image, 720,000 record() calls per arm, arms alternated in ONE process with a control
arm (`tests/route_trace_hot_path_bench.py`): new 4214.3 ns/call vs pre-fix 4315.6, i.e. -2.35% on
mins against a 0.33% control delta -- FASTER, repeatable across three runs. `harvest()` is 44 ns per
step, +0.9 ns amortized per record(), 0.00007% of a 60 ms decode step. NO REGRESSION.

The +8 is the harvest and it is ONCE PER STEP; the pre-fix 240 is not a target to beat, because at
ring_rows=1 the pre-fix code could only take the ring path for a ONE-row forward at all -- see
`test_two_row_decode_stays_on_the_ring_path`, which is defect (A): a 2-request step went to the host
path, 48 blocking `.tolist()` per step, and reached the expert cache not at all.

FALSIFY THESE BUDGETS by putting either removed op back: restore
`self.ids_ring[slot, lid, n:] = -1` inside `record` (the tail clear harvest now does once per step)
and the M=1 budget goes 5 -> 10; restore `self.meta[self.slot][lid] = (...)` and `test_no_per_layer_
host_dict_store` fails on the attribute. Both were checked that way while writing this.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))

from minisgl.weights import route_trace as RT  # noqa: E402

# The shipped qwen4exp arm.
LAYERS, EXPERTS, TOP_K, RING_ROWS = 48, 512, 10, 2
CPU = torch.device("cpu")


class _Flag:
    value = False


@pytest.fixture(autouse=True)
def _stub_capture_probe(monkeypatch):
    """`is_current_stream_capturing()` raises AcceleratorError with no ROCm device.

    Both the pre-fix and post-fix `record` call it exactly ONCE on either path, so stubbing it
    changes which branch is taken and nothing about the cost being counted.
    """
    _Flag.value = False
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: _Flag.value)
    RT._CUR_LID = None
    RT._CUR_CHUNK.clear()
    yield
    RT._CUR_LID = None
    RT._CUR_CHUNK.clear()


class Census(TorchDispatchMode):
    def __init__(self) -> None:
        self.hist: dict[str, int] = {}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.hist[str(func)] = self.hist.get(str(func), 0) + 1
        return func(*args, **(kwargs or {}))

    @property
    def total(self) -> int:
        return sum(self.hist.values())

    @property
    def syncs(self) -> list:
        return [k for k in self.hist if "_local_scalar_dense" in k or k.endswith("item")]


def tracer(ring_rows=RING_ROWS, ring_steps=16):
    return RT.RouteTracer(
        None, model_slug="cost", num_layers=LAYERS, num_experts=EXPERTS, top_k=TOP_K,
        tp_rank=0, dp_rank=0, expert_bytes=1, ring_steps=ring_steps, ring_rows=ring_rows,
        drain_every=ring_steps, max_steps=1 << 40, record_prefill=False, blockmap_checks=0,
        device=CPU,
    )


def one_record(tr, rows):
    route = torch.randint(0, EXPERTS, (rows, TOP_K), dtype=torch.int32, device=CPU)
    tr.begin_forward(False, 1, num_rows=rows)
    RT._CUR_CHUNK.clear()
    RT._CUR_LID = 0
    c = Census()
    with c:
        tr.record(route, rows)
    RT._CUR_LID = None
    return c


# ---- the budget -------------------------------------------------------------------------------
@pytest.mark.parametrize("rows,budget", [(1, 5), (2, 4)])
def test_record_ring_path_op_budget(rows, budget):
    tr = tracer()
    c = one_record(tr, rows)
    assert c.total <= budget, (
        f"record() issues {c.total} aten ops for M={rows} (budget {budget}); each one is a kernel "
        f"launch on the card, 48 times per decode step. {c.hist}"
    )
    assert not tr.oversize, "this must be the RING path; a host-path record would be a sync per layer"


def test_record_never_syncs_on_the_decode_path():
    """The forbidden implementation: a `.item()`/`.tolist()` per MoE layer.

    This repo measured that exact shape at 36% of scheduler-rank samples once (ffa1d8c6, QSA's
    `int(lens.sum().item())`, 131 -> 85 ms/token when deleted). 48 MoE layers of it is ~+180 ms on a
    60 ms step.
    """
    for rows in (1, 2):
        c = one_record(tracer(), rows)
        assert not c.syncs, f"record() synced on the decode path at M={rows}: {c.hist}"


def test_harvest_is_once_per_step_and_small():
    tr = tracer()
    one_record(tr, 1)
    tr._harvested = False
    c = Census()
    with c:
        tr.harvest()
    assert c.total <= 8, f"harvest() grew to {c.total} ops: {c.hist}"
    assert not c.syncs, f"harvest() must not sync: {c.hist}"
    # And it must be ONCE per step, not once per layer: idempotent until the next begin_forward.
    again = Census()
    with again:
        tr.harvest()
        tr.harvest()
    assert again.total == 0, f"harvest() re-ran for the same step: {again.hist}"


def test_whole_decode_step_op_budget():
    """48 records + one harvest, the way a decode step actually runs it."""
    tr = tracer()
    route = torch.randint(0, EXPERTS, (1, TOP_K), dtype=torch.int32, device=CPU)
    tr.begin_forward(False, 1, num_rows=1)
    RT._CUR_CHUNK.clear()
    c = Census()
    with c:
        for lid in range(LAYERS):
            RT._CUR_LID = lid
            tr.record(route, 1)
        RT._CUR_LID = None
        tr._harvested = False
        tr.harvest()
    assert c.total <= 5 * LAYERS + 8, f"a decode step now costs {c.total} aten ops: {c.hist}"
    assert not c.syncs, c.hist


def test_no_per_layer_host_dict_store():
    """The pre-fix code wrote `meta[slot][lid]` per layer to hold 48 copies of one tuple.

    Asserting the STRUCTURE, not the timing: one tuple per STEP, and no per-layer dict at all.
    """
    tr = tracer()
    assert not hasattr(tr, "meta"), "the per-(slot, lid) dict is back; that is 48 setitems per step"
    one_record(tr, 1)
    tr.harvest()
    assert isinstance(tr.step_meta[tr.slot], tuple) and len(tr.step_meta[tr.slot]) == 4


def test_two_row_decode_stays_on_the_ring_path():
    """Defect (A) as a COST statement, which is how it actually hurt.

    ring_rows=1 was the pre-fix engine value on a non-spec serve. M=2 fails `M <= ring_rows`, so a
    two-request decode step took the HOST path: `topk_ids.tolist()` per MoE layer, a blocking D2H on
    the card, 48 per step -- and those records land in `oversize`, which `drain()` writes to the
    trace file and never forwards to the observer.
    """
    wide = tracer(ring_rows=2)
    c = one_record(wide, 2)
    assert not wide.oversize and not c.syncs, f"M=2 must stay on the ring: {c.hist}"

    starved = tracer(ring_rows=1)
    c_old = one_record(starved, 2)
    assert len(starved.oversize) == 1, "ring_rows=1 must be shown to take the host path"
    # A TorchDispatchMode census is BLIND to `.tolist()` on a cpu tensor -- it reads the storage
    # directly rather than dispatching an aten op, which is why `host={}` here. On the card it is a
    # blocking D2H. So the cost half of this is asserted in the bench script, not here; what IS
    # asserted here is the part a CPU fixture can settle and that no timer would catch: a host-path
    # record lands in `oversize`, and `drain()` writes `oversize` to the trace FILE and never
    # forwards it to the observer, so pre-fix a 2-request decode step fed the expert cache NOTHING.
    assert c_old.hist == {}, f"expected the cpu census to be blind to .tolist(); got {c_old.hist}"
    seen = []
    starved.set_observer(lambda lid, ids: seen.append(lid))
    starved.close()
    assert seen == [], "a host-path record must be shown to never reach the observer"


# ---- the cost of being wrong about ring_rows --------------------------------------------------
def test_wide_ring_does_not_walk_the_full_row_on_every_drain():
    """Fallout of (A): `ring_rows` now scales with max_running_req, so the drain's dedupe had to
    stop walking `top_k * ring_rows` columns in Python.

    At max_running_req 256 and top_k 10 the unbounded scan is 64 steps x 48 layers x 2560 = 7.8M
    interpreter iterations PER DRAIN on the scheduler thread, to extract a few hundred ids. The gate
    is behavioural, not a timer: the drain must look at only the step's REAL columns, which is
    observable because a poisoned tail is NOT reported.
    """
    tr = tracer(ring_rows=64, ring_steps=4)
    seen = []
    tr.set_observer(lambda lid, ids: seen.append((lid, tuple(ids))))
    route = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 1]], dtype=torch.int32, device=CPU)
    tr.begin_forward(False, 1, num_rows=1)
    RT._CUR_CHUNK.clear()
    RT._CUR_LID = 0
    tr.record(route, 1)
    RT._CUR_LID = None
    tr.harvest()
    # Poison the columns past the step's real width. A drain that walks the whole row reports them.
    tr.ids_ring[tr.slot, 0, TOP_K:] = 99
    tr.drain()
    assert seen and set(seen[0][1]) == {1, 2, 3, 4, 5, 6, 7, 8, 9}, (
        f"the drain read past the step's real {TOP_K} columns: {seen}"
    )


def test_wide_ring_cannot_crash_the_boot_on_the_budget_path():
    """The byte-budget shrink in `maybe_install` is now REACHABLE, and it must not be a NameError.

    `ring_rows` used to be 1 on a non-spec serve, so `per_step * ring > budget` could not fire there.
    It now scales with `max_running_req`, whose DEFAULT is 256: at 48 layers / top_k 10 / ring 1024
    that is 503 MB against a 64 MiB budget, so the shrink branch fires on any default-concurrency
    serve with the expert cache attached. The branch logs through `_logger`, which this module never
    imports or defines.
    """
    # The arithmetic that makes the branch reachable, so the failure carries its own numbers.
    per_step = LAYERS * TOP_K * 256 * 4          # max_running_req default
    budget = 64 << 20                            # MINISGL_MOE_ROUTE_TRACE_MAX_MB default
    assert per_step * 1024 > budget, "the shrink branch would not fire; re-derive this test"
    rt_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python", "minisgl",
                               "weights", "route_trace.py")).read()
    assert "_logger" in rt_src, "maybe_install no longer logs; this gate is stale, delete it"
    assert hasattr(RT, "_logger"), (
        "route_trace.py calls `_logger.info_rank0(...)` in maybe_install's ring-shrink branch but "
        "never defines or imports `_logger`. Pre-fix that branch was unreachable on a non-spec "
        "serve (ring_rows=1); widening ring_rows to max_running_req (default 256) makes it fire at "
        "boot, where it raises NameError instead of logging. Import the logger (or drop the log)."
    )
