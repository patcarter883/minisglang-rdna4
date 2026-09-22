"""GATE: the route ring must survive HIP graph capture, and must be WIDE ENOUGH to see a
concurrent decode. Three defects, one file, CPU-only.

    # in the image, NO GPU, NO lease:
    docker run --rm --entrypoint bash -v <worktree>:/engine:ro minisgl-rdna4:lean \
      -lc 'cd /tmp; PYTHONPATH=/engine/python:/opt/kernels \
           python3 /engine/tests/route_trace_capture_gate_test.py'
    # or:  python3 -m pytest -c /dev/null -q /engine/tests/route_trace_capture_gate_test.py

WHY THIS FILE EXISTS. `route_trace`'s ring is the expert cache's ONLY input (`set_observer`). Three
separate things can make it feed the policy a union that is narrower or dirtier than the truth, and
NONE of them fails anything — they all show up as a quietly optimistic or quietly frozen hit rate:

  (A) RING TOO NARROW. `record` takes the sync-free device path only when `M <= ring_rows`; a wider
      forward falls to the HOST path, whose records `drain()` writes to the fixture but NEVER
      forwards to the observer. With `ring_rows == 1` (the old engine value whenever spec was off)
      every 2-request decode step was invisible to the cache.
  (B) CAPTURE. The old `record` early-returned under `is_current_stream_capturing()`, so on a
      captured serve the ring recorded nothing and the cache went inert while still holding its
      whole budget. A naive fix is worse: the ring's step index is host Python, so a graph would
      bake ONE slot and every replay would rewrite it.
  (C) BUCKET PADDING. A bs=2 graph replayed for one request pushes a padded row through routing.
      Those experts were referenced by nothing and must not reach the policy.

WHAT A GATE HAS TO DO THAT A PARITY TEST DOES NOT. Every assertion below is written so that it FAILS
against the unfixed code for a BEHAVIOURAL reason (the observer saw nothing / saw the padded row),
not because an attribute is missing. G4/G4B are a matched pair: G4 says the padded row is masked,
G4B mis-plumbs the row count on purpose and REQUIRES the leak to appear — so a G4 that passed because
the fixture could not express the bug would be caught by G4B failing.

TWO CPU ACCOMMODATIONS, both unavoidable and both loud:
  * `torch.cuda.is_current_stream_capturing()` RAISES `AcceleratorError` with no ROCm device, so the
    capture branch can only be reached by monkeypatching it (`capturing(True)`). That is the seam,
    not a cheat: the production code calls exactly this predicate.
  * `torch.empty(..., pin_memory=True)` raises with no GPU. The fixed tracer already conditions
    pinning on the device type; the UNFIXED one does not, so `_strip_pin` patches it away during
    construction only, and prints whether the patch was needed. Without that, every gate below would
    fail on the old tree with `RuntimeError: No CUDA GPUs are available` — an environment failure
    that proves nothing.
"""
import contextlib
import os
import sys
import traceback

import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(_REPO, "python") not in sys.path:
    sys.path.insert(0, os.path.join(_REPO, "python"))

from minisgl.weights import route_trace as RT   # noqa: E402

CPU = torch.device("cpu")
NOTES = []


# ---- harness ---------------------------------------------------------------------------------
# THE CAPTURE SEAM. `torch.cuda.is_current_stream_capturing()` RAISES on a CPU host, and BOTH
# `record` and `drain` call it, so it is replaced process-wide for the whole file by a flag this
# harness owns. On a GPU host the predicate answers False everywhere except inside a real capture
# region, which is exactly what this models: False by default, True only around the writes that a
# graph would bake.
_CAPTURING = {"v": False}
torch.cuda.is_current_stream_capturing = lambda: _CAPTURING["v"]


@contextlib.contextmanager
def capturing(flag: bool):
    """Drive `record`'s capture branch. This predicate IS the seam the shipped code keys on."""
    prev = _CAPTURING["v"]
    _CAPTURING["v"] = bool(flag)
    try:
        yield
    finally:
        _CAPTURING["v"] = prev


@contextlib.contextmanager
def _strip_pin():
    """`pin_memory=True` needs a GPU. The fixed tracer already guards it; the unfixed one does not,
    so strip the kwarg during construction so BOTH trees reach the behavioural assertions."""
    orig = torch.empty
    hit = {"n": 0}

    def patched(*a, **kw):
        if kw.pop("pin_memory", False):
            hit["n"] += 1
        return orig(*a, **kw)

    torch.empty = patched
    try:
        yield hit
    finally:
        torch.empty = orig


def make(*, top_k=2, ring_rows=2, layers=2, experts=128, ring_steps=4, drain_every=4,
         max_steps=1 << 40):
    with _strip_pin() as hit:
        tr = RT.RouteTracer(
            None, model_slug="gate", num_layers=layers, num_experts=experts, top_k=top_k,
            tp_rank=0, dp_rank=0, expert_bytes=1, ring_steps=ring_steps, ring_rows=ring_rows,
            drain_every=drain_every, max_steps=max_steps, record_prefill=False,
            blockmap_checks=0, device=CPU,
        )
    if hit["n"]:
        NOTES.append("tracer asked for pinned host memory on a cpu device (unfixed tree): "
                     "pin_memory stripped by the harness")
    tr.seen = []
    tr.set_observer(lambda lid, ids: tr.seen.append((lid, tuple(ids))))
    return tr


NO_NUM_ROWS = {"hit": False}


def bf(tr, *, prefill=False, verify=False, uid=1, rows=None):
    """`begin_forward`, tolerating a tracer that has no `num_rows` parameter.

    On the unfixed tree there is NO WAY to tell the tracer the real row count — that absence IS
    defect (C), so the shim records it and lets the padding gate fail behaviourally instead of
    dying on a TypeError."""
    try:
        tr.begin_forward(prefill, uid, is_verify=verify, num_rows=rows)
    except TypeError:
        NO_NUM_ROWS["hit"] = True
        tr.begin_forward(prefill, uid, is_verify=verify)


def moe(tr, lid, rows_2d, *, captured=False):
    """One MoE layer's `record`. `captured=True` is the write a graph would BAKE."""
    t = torch.tensor(rows_2d, dtype=torch.int32, device=CPU)
    RT._CUR_LID = lid
    with capturing(captured):
        tr.record(t, t.shape[0])
    RT._CUR_LID = None


FAILS = []
NAMES = []


def check(name, ok, detail=""):
    NAMES.append(name)
    # detail only on FAIL: a failure message printed beside PASS reads as a failure.
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"\n        -> {detail}" if detail and not ok else ""))
    if not ok:
        FAILS.append(name)


def gate(fn):
    """Run one gate; an exception is a FAIL with its traceback, never a skip."""
    print(f"\n== {fn.__name__}: {fn.__doc__.strip().splitlines()[0]}")
    try:
        fn()
    except BaseException as e:                                        # noqa: BLE001
        check(f"{fn.__name__} raised", False, f"{type(e).__name__}: {e}")
        traceback.print_exc()
    return fn


# =============================================================================================
# (A) the ring must be wide enough for the widest forward the engine can produce
# =============================================================================================
def _ring_rows_for(**kw):
    """The engine's own ring width for a config. Falls back to the OLD inline expression so the
    unfixed tree reaches a BEHAVIOURAL failure (G2) rather than an AttributeError."""
    from minisgl.distributed.info import DistributedInfo
    from minisgl.engine.config import EngineConfig
    import minisgl.engine.engine as EE

    cfg = EngineConfig(model_path="/nonexistent-gate-model", tp_info=DistributedInfo(0, 1),
                       dtype=torch.bfloat16, **kw)
    fn = getattr(EE, "_route_trace_ring_rows", None)
    if fn is None:
        return (cfg.max_running_req * (1 + cfg.spec_num_draft)
                if cfg.spec_config is not None else 1), False
    return int(fn(cfg)), True


@gate
def G1_engine_ring_rows_covers_the_widest_forward():
    """the engine's ring_rows must cover max_running_req, cuda_graph_max_bs and the spec K+1."""
    n, have = _ring_rows_for(max_running_req=2, cuda_graph_max_bs=0)
    check("engine derives ring_rows from the config (not a hardcoded 1)", have,
          "minisgl.engine.engine._route_trace_ring_rows is missing — defect (A) is unfixed")
    check("shipped qwen4exp arm (mrr=2, capture off) sizes the ring for 2 rows", n >= 2,
          f"ring_rows={n}; at 1 every 2-request decode step falls to the host path and the "
          f"expert cache never sees it")
    n, _ = _ring_rows_for(max_running_req=2, cuda_graph_max_bs=2)
    check("capture on at bs=2 still covers 2 rows", n >= 2, f"ring_rows={n}")
    n, _ = _ring_rows_for(max_running_req=4, cuda_graph_max_bs=None)
    check("cuda_graph_max_bs=None ('auto') covers max_running_req", n >= 4, f"ring_rows={n}")
    n, _ = _ring_rows_for(max_running_req=8, cuda_graph_max_bs=32)
    check("a capture bucket WIDER than max_running_req is covered", n >= 32,
          f"ring_rows={n}; a captured bucket the ring cannot hold bakes a TRUNCATED write")
    n, _ = _ring_rows_for(max_running_req=8, cuda_graph_max_bs=32, spec_algorithm="ngram",
                          spec_num_draft=3)
    check("spec still multiplies by (1 + num_draft)", n >= 32 * 4,
          f"ring_rows={n}; a verify carries K+1 rows per request")


@gate
def G2_a_two_row_decode_must_reach_the_observer():
    """with the ENGINE's own ring width, a 2-request decode's union must reach the policy."""
    rows, _ = _ring_rows_for(max_running_req=2, cuda_graph_max_bs=0)
    tr = make(ring_rows=rows, top_k=2)
    bf(tr, rows=2)
    moe(tr, 0, [[7, 8], [9, 10]])           # two concurrent requests, disjoint experts
    tr.drain()
    got = set(tr.seen[0][1]) if tr.seen else set()
    check("the 2-request decode step was observed at all", len(tr.seen) == 1,
          f"ring_rows={rows}, observer calls={len(tr.seen)} — 0 means the step fell to the HOST "
          f"path, which drain() writes to the fixture but never forwards to the observer")
    check("the observer saw BOTH rows' experts", got == {7, 8, 9, 10}, f"got {sorted(got)}")
    check("a decode that fell off the ring is not even counted", tr.rows_dropped == 0,
          f"rows_dropped={tr.rows_dropped} (only VERIFY increments it — a starved DECODE is silent)")


@gate
def G3_mechanism_ring_rows_1_really_does_starve():
    """positive control for (A): at ring_rows=1 the 2-row step IS invisible (passes on both trees)."""
    tr = make(ring_rows=1, top_k=2)
    bf(tr, rows=2)
    moe(tr, 0, [[7, 8], [9, 10]])
    tr.drain()
    check("ring_rows=1 starves a 2-row decode (so G2 is not vacuous)", len(tr.seen) == 0,
          f"observer calls={len(tr.seen)}")


# =============================================================================================
# (B) capture safety
# =============================================================================================
@gate
def G4_a_captured_step_reaches_the_observer():
    """a decode recorded under capture must still be observed, on every layer."""
    tr = make(ring_rows=2, top_k=2, layers=2)
    bf(tr, rows=1)                                    # ONE real request
    moe(tr, 0, [[1, 2], [90, 91]], captured=True)     # bs=2 bucket: row 1 is PADDING
    moe(tr, 1, [[3, 4], [92, 93]], captured=True)
    bf(tr, rows=1)                                    # step boundary -> harvest
    tr.drain()
    per_lid = {lid: set(ids) for lid, ids in tr.seen}
    check("the captured step reached the observer", len(tr.seen) == 2,
          f"observer calls={len(tr.seen)} for 2 layers — 0 means record() still early-returns "
          f"under capture, so a captured serve feeds the expert cache NOTHING")
    check("layer 0's real row was observed", per_lid.get(0) == {1, 2}, f"got {per_lid.get(0)}")
    check("layer 1's real row was observed", per_lid.get(1) == {3, 4}, f"got {per_lid.get(1)}")


@gate
def G5_the_captured_write_touches_no_per_step_host_state():
    """a graph can only bake device ops: the captured branch must not read/write slot or meta."""
    tr = make(ring_rows=2, top_k=2, layers=2)
    bf(tr, rows=1)
    ring_before = tr.ids_ring.clone()
    slot_before, meta_before = tr.slot, list(tr.step_meta)
    chunk_before = dict(RT._CUR_CHUNK)
    stage_before = tr.stage.clone() if hasattr(tr, "stage") else None
    moe(tr, 0, [[5, 6], [70, 71]], captured=True)
    check("the ring is NOT written under capture (its step index would be baked)",
          torch.equal(tr.ids_ring, ring_before),
          "a graph that writes ids_ring[self.slot] rewrites ONE frozen slot on every replay")
    check("no per-step host state moved under capture",
          tr.slot == slot_before and list(tr.step_meta) == meta_before
          and dict(RT._CUR_CHUNK) == chunk_before,
          f"slot {slot_before}->{tr.slot}, meta changed={list(tr.step_meta) != meta_before}, "
          f"chunk changed={dict(RT._CUR_CHUNK) != chunk_before}")
    check("a per-step STAGE buffer exists and DID take the write", stage_before is not None
          and not torch.equal(tr.stage, stage_before),
          "no stage, or the captured record was a no-op — either way nothing reaches the policy")


@gate
def G6_replaying_one_baked_write_lands_in_a_DIFFERENT_slot_each_step():
    """the emulated graph: the SAME write, re-run per step, must be attributed to its own step."""
    tr = make(ring_rows=2, top_k=2, layers=1, ring_steps=4, drain_every=1)
    static = torch.zeros((2, 2), dtype=torch.int32, device=CPU)   # the graph's static input buffer

    def replay():
        """Exactly what a replay does: re-run the recorded device op, NO Python bookkeeping."""
        RT._CUR_LID = 0
        with capturing(True):
            tr.record(static, static.shape[0])
        RT._CUR_LID = None

    with capturing(True):                    # "capture" once
        RT._CUR_LID = 0
        tr.record(static, static.shape[0])
        RT._CUR_LID = None

    expect = []
    for step in range(11):                   # 11 steps over a 4-slot ring => wraps twice
        real = 20 + 2 * step
        static.copy_(torch.tensor([[real, real + 1], [111, 112]], dtype=torch.int32))
        bf(tr, rows=1, uid=step + 1)
        replay()
        expect.append({real, real + 1})
    bf(tr, rows=1, uid=99)                   # harvest the last one
    tr.drain()
    got = [set(ids) for _, ids in tr.seen]
    check("every step was observed exactly once", len(got) == 11,
          f"{len(got)} of 11 — a baked slot index makes replays overwrite one entry")
    check("each step was observed with ITS OWN experts", got == expect,
          f"got {got[:4]}... expected {expect[:4]}...")


@gate
def G7_an_over_wide_captured_bucket_must_RAISE():
    """baking a truncated write would feed a partial union forever: stop the boot instead."""
    tr = make(ring_rows=2, top_k=2)
    bf(tr, rows=1)
    err = None
    try:
        moe(tr, 0, [[1, 2]] * 4, captured=True)       # 4 rows into a 2-row ring
    except BaseException as e:                        # noqa: BLE001
        err = e
    check("a bucket wider than the ring raises RouteTraceError", isinstance(err, RT.RouteTraceError),
          f"got {type(err).__name__ if err else 'no exception'} — silence here bakes a truncated "
          f"write and every replay under-reports the union")


# =============================================================================================
# (C) a captured bucket's PADDED rows must not reach the policy
# =============================================================================================
@gate
def G8_padded_rows_are_masked_at_the_harvest():
    """bs=2 bucket, ONE real request: the padded row's experts must not reach the policy."""
    tr = make(ring_rows=2, top_k=2, layers=1)
    bf(tr, rows=1)
    moe(tr, 0, [[1, 2], [90, 91]], captured=True)
    bf(tr, rows=1)
    tr.drain()
    got = set(tr.seen[-1][1]) if tr.seen else set()
    check("the real row is reported", {1, 2} <= got, f"got {sorted(got)}")
    check("the PADDED row's experts are NOT reported", not ({90, 91} & got),
          f"got {sorted(got)} — 90/91 were routed by a cudagraph dummy req, referenced by nothing; "
          f"feeding them pollutes the cache with experts no request wanted")
    check("nothing but the real row is reported", got == {1, 2}, f"got {sorted(got)}")


@gate
def G8B_falsification_mis_plumbing_the_row_count_DOES_leak():
    """positive control for G8: told the BUCKET width, the padding must leak (else G8 is vacuous)."""
    tr = make(ring_rows=2, top_k=2, layers=1)
    bf(tr, rows=2)                                    # WRONG on purpose: 2 is the bucket, not the batch
    moe(tr, 0, [[1, 2], [90, 91]], captured=True)
    bf(tr, rows=2)
    tr.drain()
    got = set(tr.seen[-1][1]) if tr.seen else set()
    check("with the row count mis-plumbed the padding LEAKS", {90, 91} <= got,
          f"got {sorted(got)} — if this passes, G8's mask is really masking; if it FAILS, G8 is "
          f"measuring something else and cannot see defect (C) at all")


@gate
def G9_an_unknown_row_count_is_counted_not_assumed():
    """no row count => harvested unmasked AND counted, never silently assumed to be 1."""
    tr = make(ring_rows=2, top_k=2, layers=1)
    bf(tr, rows=None)
    moe(tr, 0, [[1, 2], [90, 91]], captured=True)
    bf(tr, rows=None)
    tr.drain()
    check("rows_unknown counts a step with no plumbed row count", getattr(tr, "rows_unknown", 0) >= 1,
          f"rows_unknown={getattr(tr, 'rows_unknown', 'ABSENT')} — the one way padded routing can "
          f"still reach the policy has to be visible at close()")
    for k in ("route_trace_rows_unknown", "route_trace_rows_mismatch",
              "route_trace_capture_unwrapped", "route_trace_rows_dropped"):
        check(f"stats() exports {k}", k in tr.stats(), str(sorted(tr.stats())))


# =============================================================================================
# regression guards — these must pass on BOTH trees
# =============================================================================================
@gate
def G10_prefill_is_still_withheld_from_the_policy():
    """a prefill touches most experts; feeding it would look like one enormous sweep."""
    tr = make(ring_rows=2, top_k=2, layers=1)
    bf(tr, prefill=True, rows=8)
    moe(tr, 0, [[11, 12]] * 8)
    bf(tr, rows=1)
    tr.drain()
    check("no prefill record reached the observer", len(tr.seen) == 0, f"seen={tr.seen}")


@gate
def G11_eager_decode_and_verify_are_unchanged():
    """the eager paths this change was not supposed to touch."""
    tr = make(ring_rows=4, top_k=2, layers=1)
    bf(tr, rows=1)
    moe(tr, 0, [[5, 6]])
    tr.drain()
    check("eager 1-row decode still observed", len(tr.seen) == 1 and set(tr.seen[0][1]) == {5, 6},
          str(tr.seen))
    tr = make(ring_rows=4, top_k=2, layers=1)
    bf(tr, verify=True, rows=3)
    moe(tr, 0, [[1, 2], [3, 4], [5, 6]])
    tr.drain()
    check("eager verify reports the UNION of its rows",
          len(tr.seen) == 1 and set(tr.seen[0][1]) == {1, 2, 3, 4, 5, 6}, str(tr.seen))
    check("no verify row fell to the host path", tr.rows_dropped == 0,
          f"rows_dropped={tr.rows_dropped}")


@gate
def G12_close_harvests_the_last_step():
    """the only step of a one-step run must not be dropped."""
    tr = make(ring_rows=2, top_k=2, layers=1, ring_steps=8, drain_every=8)
    bf(tr, rows=1)
    moe(tr, 0, [[41, 42]], captured=True)
    tr.close()                                   # no following begin_forward
    check("close() harvested the final captured step",
          any(set(ids) == {41, 42} for _, ids in tr.seen), str(tr.seen))


@gate
def G13_a_narrower_step_does_not_inherit_a_wider_one():
    """stage/ring reuse must not leak the previous step's ids into this one."""
    tr = make(ring_rows=4, top_k=2, layers=1, ring_steps=1, drain_every=1)
    bf(tr, rows=3)
    moe(tr, 0, [[20, 21], [22, 23], [24, 25]], captured=True)
    bf(tr, rows=1)                               # same slot (ring_steps=1)
    moe(tr, 0, [[1, 2]], captured=True)
    bf(tr, rows=1)
    tr.drain()
    last = set(tr.seen[-1][1]) if tr.seen else set()
    check("the narrow step reports only its own ids", last == {1, 2}, f"got {sorted(last)}")


@gate
def G15_capture_warmup_routings_must_not_survive_into_the_first_real_step():
    """the stage is written at CAPTURE with warmup routings and nothing resets it before step 0.

    `harvest`'s comment claims "the stage is all -1 by induction, since only a decode/verify step
    writes it and that step's own harvest reset it". CAPTURE BREAKS THAT INDUCTION: it writes the
    stage with the warmup forward's routings while `slot == -1`, the first `begin_forward`'s harvest
    early-returns on `slot < 0` without resetting, and a PREFILL step's harvest early-returns without
    resetting either. The garbage is then read by the first decode step whose plumbed row count
    EXCEEDS the rows its MoE call carried -- the tp_overlap ROW-SPLIT shape (>=256 rows, chunk 0 is
    the only chunk the ring takes), which `max_running_req`'s default of 256 makes reachable.

    ONE STEP PER BOOT, and `rows_mismatch` does flag it, so this is small -- but it is a NEW pollution
    path (there was no stage before capture support) and the fix is one line: reset the stage
    unconditionally in `harvest`, above the kind check, instead of only on the decode/verify path."""
    # experts=1024 so the warmup ids below are IN RANGE: drain filters `0 <= e < num_experts`, and
    # with the default 128 this gate would pass for the wrong reason.
    tr = make(ring_rows=4, top_k=2, layers=1, experts=1024)
    moe(tr, 0, [[500, 501], [502, 503], [504, 505], [506, 507]], captured=True)   # engine init
    bf(tr, prefill=True, rows=8)                      # first real forward is a prefill
    moe(tr, 0, [[11, 12]] * 8)
    bf(tr, rows=4)                                    # row-split decode: plumbed 4, MoE carries 2
    moe(tr, 0, [[1, 2], [3, 4]], captured=True)
    bf(tr, rows=1)
    tr.drain()
    got = set(tr.seen[-1][1]) if tr.seen else set()
    warm = {500, 501, 502, 503, 504, 505, 506, 507}
    check("no capture-warmup expert reaches the policy", not (warm & got),
          f"leaked {sorted(warm & got)} (observed {sorted(got)}) — those experts were routed by the "
          f"capture warmup's dummy hidden states, not by any request")


@gate
def G16_widening_ring_rows_must_not_walk_into_an_undefined_logger():
    """the ring-shrink branch widening (A) newly reaches calls `_logger`, which nothing defines.

    `maybe_install` shrinks `ring_steps` when `num_layers * top_k * ring_rows * 4 * ring` exceeds
    MINISGL_MOE_ROUTE_TRACE_MAX_MB, and logs it with `_logger.info_rank0(...)`. `route_trace.py`
    never imports or defines `_logger`. BEFORE this change that branch was dead on any non-spec
    serve (ring_rows was 1, so 1.9 MiB against a 64 MiB budget); sizing ring_rows from
    max_running_req makes it FIRE, and it raises `NameError` at boot instead of logging.

    REACHABLE AT max_running_req >= 35 on the qwen4exp shape (48 layers, top_k 10): 1920*rows*1024
    passes 64 MiB at rows 35. The shipped offload arm sets --max-running-requests 2 and is safe, but
    `MINISGL_MOE_ROUTE_TRACE=<dir>` on an otherwise-default serve (mrr 256) is not -- and that is the
    fixture-capture path the whole expert-cache line of work depends on.

    FIX: `from minisgl.utils import init_logger` + `_logger = init_logger(__name__)`, the same two
    lines weights/prefill_stage.py:50-54 already uses."""
    check("route_trace defines the `_logger` its shrink branch calls", hasattr(RT, "_logger"),
          "route_trace.py:~813 calls _logger.info_rank0(...) and the module never defines it -> "
          "NameError at boot on any config whose ring exceeds the byte budget")
    n, _ = _ring_rows_for(max_running_req=256, cuda_graph_max_bs=None)
    per_step = 48 * 10 * n * 4
    check("...and that branch is now REACHABLE on a default-max_running_req serve",
          per_step * 1024 > (64 << 20),
          f"ring_rows={n}, {per_step} B/step x 1024 steps = {per_step * 1024 / (1 << 20):.1f} MiB "
          f"vs the 64 MiB default budget — this check documents WHY the missing _logger now matters; "
          f"it is expected to be False on the unfixed tree, where ring_rows is 1")


# =============================================================================================
# the scheduler's half of (C): the row count it plumbs must be the REAL query rows
# =============================================================================================
@gate
def G14_step_boundary_plumbs_the_real_query_row_count():
    """_step_boundary must tell the tracer sum(extend_len), the one value right for all 4 shapes."""
    from minisgl.scheduler.scheduler import Scheduler

    class R:
        def __init__(self, n, uid=1):
            self.extend_len, self.uid = n, uid

    class B:
        def __init__(self, reqs):
            self.reqs = reqs

    tr = make(ring_rows=64, top_k=2, layers=1)
    RT._TRACER = tr
    try:
        for label, reqs, want in (
            ("plain decode, 2 requests", [R(1), R(1)], 2),
            ("spec verify, 2 requests x K+1=4", [R(4), R(4)], 8),
            ("DDTree verify, per-req widths", [R(3), R(5)], 8),
            ("empty (DP dummy) batch", [], 0),
        ):
            Scheduler._step_boundary(object(), B(reqs), False)
            check(f"{label} -> num_rows={want}", tr._cur_rows == want,
                  f"tracer saw {tr._cur_rows!r}; a wrong count either truncates a real row's "
                  f"union or admits a padded row's experts, and neither shows in a hit rate")
    finally:
        RT._TRACER = None


# ---- report ----------------------------------------------------------------------------------
def test_route_trace_capture_gate():
    """pytest entry point: the whole gate, as one test."""
    assert not FAILS, f"{len(FAILS)} of {len(NAMES)} checks failed: {FAILS}"


print(f"\n{'=' * 92}")
for n in dict.fromkeys(NOTES):          # dedupe, keep order
    print(f"NOTE: {n}")
if NO_NUM_ROWS["hit"]:
    print("NOTE: this tracer's begin_forward has NO num_rows parameter — the real row count cannot "
          "be plumbed at all, which is defect (C) unfixed.")
print(f"{len(NAMES) - len(FAILS)}/{len(NAMES)} checks passed")
print("ALL PASS" if not FAILS else "FAILED: " + ", ".join(FAILS))
print(f"{'=' * 92}")

if __name__ == "__main__":
    sys.exit(1 if FAILS else 0)
