"""Stage A under HIP graph capture and at TP>=2 — the two lenses `test_weight_bake.py` does not cover.

GPU-FREE and torch-free. Two of these are structural (they read `engine.py`'s AST) because the
defects they guard are defects of ORDER inside `Engine.__init__`, and order is exactly what a unit
test over a fake driver cannot see: the original post-capture gate was wired correctly in every
respect except that it ran 300 lines before the first graph was captured, which no behavioural test
of `StageASession` could ever have caught.

THE THREE THINGS BEING GUARDED

1. The arena gate must run AFTER graph capture. `ArenaMemPool.assert_clean()` carries a dedicated
   `alloc_during_capture` failure branch, and that counter can only move while a HIP capture is in
   flight. Read only at `seal()` it is provably zero.
2. Every step of the Stage-A window must succeed or fail SYMMETRICALLY across TP ranks. Pinning host
   pages (P3b: a 62 GiB two-rank ceiling against a 68.8 GiB target) and measuring free VRAM are
   per-rank and racy; a one-sided abort leaves the peer in an `all_reduce` on a gloo group built with
   a seven-day timeout.
3. `num_pages` must be cross-rank reduced. Every other input to the KV budget is a config constant or
   already `all_reduce`d; `memory_allocated()`/`memory_reserved()` and the weight-arena correction
   derived from them are not.
"""

from __future__ import annotations

import ast
import pathlib

import pytest
from minisgl.weights import bake as bake_mod
from minisgl.weights.bake import StageAPhase, StageASession

GiB = 1 << 30
_PY = pathlib.Path(__file__).resolve().parents[2] / "python" / "minisgl"
ENGINE_PY = _PY / "engine" / "engine.py"
SCHEDULER_PY = _PY / "scheduler" / "scheduler.py"


def _method(path, cls_name, fn_name):
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls_name)
    return next(
        n
        for n in cls.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == fn_name
    )


def _call_line(fn, fragment):
    return next(
        (n.lineno for n in ast.walk(fn) if isinstance(n, ast.Call) and fragment in ast.dump(n)),
        None,
    )


def _call_lines(fn, fragment):
    return sorted(
        n.lineno for n in ast.walk(fn) if isinstance(n, ast.Call) and fragment in ast.dump(n)
    )


class _Driver:
    """A `StageADriver` that can be told to fail one step, and to move the arena during capture."""

    def __init__(self, *, host=30 * GiB, device=6 * GiB, fail_on=None):
        self._host = host
        self._device = device
        self._fail_on = fail_on
        self.activity = (8, host)
        self.clean_calls = 0
        self.frozen = False

    def plan_bytes(self):
        return (self._host, self._device, self._host + self._device)

    def _maybe_fail(self, step):
        if self._fail_on == step:
            raise RuntimeError(f"driver failed at {step}")

    def attach_host_arena(self):
        self._maybe_fail("attach")

    def bind(self, model):
        self._maybe_fail("bind")
        return "bound"

    def moved_bytes(self):
        return self._host

    def arena_torch_bytes(self):
        return 0

    def arena_activity(self):
        return self.activity

    def assert_arena_clean(self):
        self.clean_calls += 1
        self._maybe_fail("clean")

    def freeze(self):
        self.frozen = True

    def describe(self):
        return "fake plan"


def _probe(host=30 * GiB):
    free, alloc, res = 40 * GiB, 12 * GiB, 13 * GiB
    seq = iter(
        [
            (free, alloc, res),
            (free, alloc, res),
            (free, alloc, res),
            (free, alloc - host, res),
        ]
    )
    return lambda: next(seq)


def _sealed(driver=None, **kw):
    d = driver or _Driver()
    s = StageASession.begin(object(), driver=d, probe=_probe(), **kw)
    s.attach()
    s.note_loaded()
    s.bind(object())
    s.seal()
    return s, d


@pytest.fixture(autouse=True)
def _clear_published_slack():
    bake_mod._reset_for_tests()
    yield
    bake_mod._reset_for_tests()


# =================================================================================================
# 1. The post-capture gate
# =================================================================================================


class TestPostCaptureGate:
    def test_a_quiet_capture_passes_and_re_runs_the_clean_check(self):
        s, d = _sealed()
        before = d.clean_calls
        s.verify_after_capture()
        # The whole defect was that `assert_clean()` only ever ran where its capture branch was
        # unreachable. It has to run a SECOND time, on the far side of capture.
        assert d.clean_calls == before + 1

    def test_an_arena_allocation_during_capture_is_refused(self):
        s, d = _sealed()
        # torch called the C ABI alloc callback while a graph was capturing: the bump allocator
        # handed out a pointer that is now baked into a graph and can never be freed.
        d.activity = (d.activity[0] + 1, d.activity[1] + (4 << 20))
        with pytest.raises(RuntimeError, match="ALLOCATED FROM during HIP graph capture"):
            s.verify_after_capture()

    def test_growth_is_caught_even_when_the_pool_never_saw_a_capture(self):
        # `ArenaMemPool._capturing()` answers False on a torch without
        # `is_current_stream_capturing`, so `alloc_during_capture` stays 0 and `assert_clean()`
        # passes. The activity comparison is arithmetic, not a query, so it still fires.
        s, d = _sealed()
        d.activity = (d.activity[0], d.activity[1] + 1)
        with pytest.raises(RuntimeError, match="ALLOCATED FROM during HIP graph capture"):
            s.verify_after_capture()
        assert d.clean_calls == 1  # the growth check ran FIRST; assert_clean would have passed

    def test_a_fallback_that_only_appears_after_capture_is_refused(self):
        s, d = _sealed()
        d._fail_on = "clean"
        with pytest.raises(RuntimeError, match="failed at clean"):
            s.verify_after_capture()

    def test_it_cannot_run_before_the_seal(self):
        d = _Driver()
        s = StageASession.begin(object(), driver=d, probe=_probe())
        s.attach()
        s.note_loaded()
        s.bind(object())
        with pytest.raises(RuntimeError, match="before seal"):
            s.verify_after_capture()

    def test_a_disabled_session_is_a_no_op(self):
        s = StageASession.disabled()
        s.attach()
        s.note_loaded()
        s.bind(object())
        s.seal()
        s.verify_after_capture()  # must not raise, must not need a driver
        assert s.phase is StageAPhase.SEALED

    def test_it_does_not_advance_the_phase(self):
        # It is a gate, not a fifth step: calling it twice (or not at all) must be legal.
        s, _ = _sealed()
        s.verify_after_capture()
        s.verify_after_capture()
        assert s.phase is StageAPhase.SEALED


class TestEngineWiresTheGateAfterCapture:
    """The gate is worthless in the right place and worthless in the wrong one — check the place."""

    def test_engine_gates_after_its_own_capture_sites(self):
        fn = _method(ENGINE_PY, "Engine", "__init__")
        gate = _call_line(fn, "'verify_weight_arena_after_capture'")
        runner = _call_line(fn, "'GraphRunner'")
        canvas = _call_line(fn, "'_capture_canvas_graphs'")
        assert gate is not None, "Engine.__init__ must re-gate the arena after graph capture"
        assert runner is not None and canvas is not None
        # If this fails, the gate has drifted back above a capture site and the
        # `alloc_during_capture` branch it exists to reach is unreachable again.
        assert gate > runner and gate > canvas

    def test_the_scheduler_gates_after_the_SPEC_capture_families(self):
        """THE ONE THAT MATTERS. Three of the five families are captured after `Engine.__init__`
        returns (they need the proposer), so an engine-only gate covers decode and canvas and leaves
        spec-verify, propose, fused-TiDAR and DDTree completely ungated."""
        fn = _method(SCHEDULER_PY, "Scheduler", "__init__")
        gate = _call_line(fn, "'verify_weight_arena_after_capture'")
        assert gate is not None, (
            "Scheduler.__init__ must re-gate the weight arena after it captures the spec graphs; "
            "Engine.__init__'s gate runs before they exist."
        )
        captures = [
            ln
            for frag in (
                "'capture_spec_verify_graphs'",
                "'capture_spec_propose_graphs'",
                "'capture_spec_fused_verify_graphs'",
                "'capture_spec_ddtree_verify_graphs'",
            )
            for ln in _call_lines(fn, frag)
        ]
        assert len(captures) == 4, "a capture family was added or renamed — re-anchor the gate"
        assert gate > max(captures)

    def test_the_scheduler_gate_is_not_under_the_spec_branch(self):
        # It must run on every serve, not only on spec ones, or it rots into a branch nobody
        # exercises — the same argument that keeps the whole disabled session on the boot path.
        fn = _method(SCHEDULER_PY, "Scheduler", "__init__")
        nested = [
            n.lineno
            for stmt in fn.body
            if isinstance(stmt, ast.If)
            for n in ast.walk(stmt)
            if isinstance(n, ast.Call) and "'verify_weight_arena_after_capture'" in ast.dump(n)
        ]
        assert not nested

    def test_seal_still_runs_before_the_kv_pool_is_sized(self):
        fn = _method(ENGINE_PY, "Engine", "__init__")
        seal = _call_line(fn, "'seal'")
        sizing = _call_line(fn, "'_determine_num_pages'")
        assert seal is not None and sizing is not None and seal < sizing


# =================================================================================================
# 2. Cross-rank symmetry
# =================================================================================================


def _gather(peer_statuses):
    """A `gather` double: this rank's status plus the peers'. `g.peers` is live, so a test can make
    the peer fail at one specific step (which is the realistic shape — the loser of the pinned-host
    race fails at attach, and only there)."""

    calls = []

    def g(mine):
        calls.append(mine)
        return [mine, *g.peers]

    g.peers = list(peer_statuses)
    g.calls = calls
    return g


class TestRankSymmetry:
    def test_a_healthy_pair_does_not_raise(self):
        g = _gather([""])
        s, _ = _sealed(gather=g)
        s.verify_after_capture()
        # attach, bind, seal, post_capture — every step is barriered, not just the risky-looking one.
        assert len(g.calls) == 4

    def test_a_peer_failing_to_pin_raises_here_too(self):
        # THE HANG. Rank 1 loses the race for pinned host pages (P3b: 62 GiB across two ranks) and
        # dies; without this, rank 0 loads the whole checkpoint and then blocks in
        # _sync_get_memory()'s all_reduce on a gloo group whose timeout is SEVEN DAYS.
        g = _gather(["HostCapacityError: cannot pin 34.0 GiB"])
        d = _Driver()
        s = StageASession.begin(object(), driver=d, probe=_probe(), gather=g)
        with pytest.raises(RuntimeError, match="failed on 1 of 2 rank"):
            s.attach()

    def test_the_lockstep_message_names_the_failing_rank_and_its_cause(self):
        g = _gather(["HostCapacityError: cannot pin 34.0 GiB"])
        d = _Driver()
        s = StageASession.begin(object(), driver=d, probe=_probe(), gather=g)
        with pytest.raises(RuntimeError) as ei:
            s.attach()
        msg = str(ei.value)
        assert "rank1" in msg and "cannot pin 34.0 GiB" in msg

    def test_the_failing_rank_still_enters_the_gather(self):
        # If the loser raised without gathering, the barrier itself would be the desync — the exact
        # trap `WeightPlanResolution.assert_rank_agreement` documents.
        g = _gather([""])
        d = _Driver(fail_on="attach")
        s = StageASession.begin(object(), driver=d, probe=_probe(), gather=g)
        with pytest.raises(RuntimeError, match="driver failed at attach"):
            s.attach()
        assert g.calls and "driver failed at attach" in g.calls[0]

    def test_a_local_failure_reports_the_original_cause_not_the_barrier(self):
        g = _gather([""])
        d = _Driver(fail_on="bind")
        s = StageASession.begin(object(), driver=d, probe=_probe(), gather=g)
        s.attach()
        s.note_loaded()
        with pytest.raises(RuntimeError, match="driver failed at bind"):
            s.bind(object())

    def test_the_seal_gate_is_barriered_too(self):
        # seal()'s three gates are per-rank MEASUREMENTS off two different physical cards, so they
        # can genuinely disagree even when the plan does not.
        g = _gather([""])
        d = _Driver()
        s = StageASession.begin(object(), driver=d, probe=_probe(), gather=g)
        s.attach()
        s.note_loaded()
        s.bind(object())
        g.peers = ["RuntimeError: host arena costs 0 device bytes"]
        with pytest.raises(RuntimeError, match="failed on 1 of 2 rank"):
            s.seal()

    def test_a_disabled_session_still_participates(self):
        # "one rank thinks offload is off" is a divergence the barrier must survive, not deadlock on,
        # so participation cannot be gated on `enabled`.
        g = _gather([""])
        s = StageASession(gather=g, phase=StageAPhase.PLANNED)
        s.attach()
        s.note_loaded()
        s.bind(object())
        s.seal()
        s.verify_after_capture()
        assert len(g.calls) == 4

    def test_no_group_and_no_gather_means_no_collective(self):
        # tp_size == 1 must be byte-identical: nothing to gather over, nothing gathered.
        s, d = _sealed()
        s.verify_after_capture()
        assert s.agreement_group is None and s.gather is None
        assert d.frozen

    def test_a_local_failure_with_no_peers_is_re_raised_unchanged(self):
        d = _Driver(fail_on="attach")
        s = StageASession.begin(object(), driver=d, probe=_probe())
        with pytest.raises(RuntimeError, match="driver failed at attach"):
            s.attach()


# =================================================================================================
# 3. num_pages must be cross-rank reduced
# =================================================================================================


class TestNumPagesIsReduced:
    @staticmethod
    def _fn(name):
        return _method(ENGINE_PY, "Engine", name)

    def test_the_reducer_exists_and_is_a_min_all_reduce(self):
        src = ast.dump(self._fn("_tp_min_num_pages"))
        assert "ReduceOp" in src and "MIN" in src and "tp_cpu_group" in src

    def test_determine_num_pages_reduces_before_it_asserts(self):
        fn = self._fn("_determine_num_pages")
        call_line = _call_line(fn, "'_tp_min_num_pages'")
        assert call_line is not None, (
            "_determine_num_pages must MIN-reduce num_pages across the TP group: every other term "
            "is config or already all_reduce'd, but memory_allocated/memory_reserved and the "
            "weight-arena correction are live per-rank readings, and a one-page disagreement means "
            "the peer writes past the end of this rank's KV pool."
        )
        assert_line = next(n.lineno for n in ast.walk(fn) if isinstance(n, ast.Assert))
        assert call_line < assert_line

    def test_the_reduction_is_not_nested_inside_the_auto_sizing_branch(self):
        # It has to run on the `--num-pages` override path too: a collective only some ranks enter
        # is a worse failure than the divergence it fixes.
        fn = self._fn("_determine_num_pages")
        top_level = [
            n
            for stmt in fn.body
            for n in ast.walk(stmt)
            if isinstance(n, ast.Call) and "_tp_min_num_pages" in ast.dump(n)
        ]
        nested = [
            n
            for stmt in fn.body
            if isinstance(stmt, ast.If)
            for n in ast.walk(stmt)
            if isinstance(n, ast.Call) and "_tp_min_num_pages" in ast.dump(n)
        ]
        assert top_level and not nested


# =================================================================================================
# 4. Nothing on the decode path
# =================================================================================================


def test_the_prefill_guard_reads_a_plain_module_global():
    """`weight_arena_torch_slack_bytes` is called inside `Scheduler._prefill_budget_now`, i.e. per
    scheduling step. It must be a constant read — no device query, no sync, no allocator call — or
    the offload feature buys a host round-trip on the hot path."""
    tree = ast.parse(pathlib.Path(bake_mod.__file__).read_text())
    fn = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "weight_arena_torch_slack_bytes"
    )
    stmts = [s for s in fn.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
    assert len(stmts) == 1, "the guard's read must stay a single statement"
    ret = stmts[0]
    assert isinstance(ret, ast.Return) and isinstance(ret.value, ast.Name)
    assert ret.value.id == "_TORCH_SLACK"
