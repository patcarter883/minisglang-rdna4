"""The CPU tier's ASYNC HANDOFF: every ordering guarantee, exercised without a GPU.

The handoff is where a CPU-compute MoE tier gets to be wrong QUIETLY. Nothing here crashes when it
breaks: a partial joined out of order, a result that arrived after the buffer was recycled, or a
worker exception swallowed into a zero partial all produce fluent, plausible, wrong text. So the
rules are a state machine with checkable edges (`cpu_tier.CpuHandoff` / `HandoffLedger`) rather than
a `Future`, and this file is the proof that each rule actually fires.

NO GPU, NO TORCH. `cpu_tier` is torch-free by construction and `cpu_worker`'s dispatcher never
touches a device tensor or calls a HIP API — the payload is opaque. That is deliberate: it means
the ordering machinery is exercised here EXACTLY as the engine drives it.
"""

from __future__ import annotations

import threading
import time

import pytest
from minisgl.weights.cpu_tier import (
    CpuHandoff,
    CpuTierError,
    HandoffError,
    HandoffLedger,
    HandoffState,
)
from minisgl.weights.cpu_worker import CpuMoEWorker, split_route_for_cpu


class _EchoBackend:
    """Returns the payload, slowly enough that a join really has to wait."""

    name = "echo"

    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay
        self.seen: list[tuple] = []

    def compute(self, x, ids, weights):
        if self.delay:
            time.sleep(self.delay)
        self.seen.append((x, tuple(ids), tuple(weights)))
        return ("partial", x)


class _AngryBackend:
    name = "angry"

    def compute(self, x, ids, weights):
        raise ZeroDivisionError("expert 7 has no weights")


# ════════════════════════════════════════════════════════════════════════════════════════════════
class TestHandoffStateMachine:
    def test_the_happy_path_visits_every_state_in_order(self):
        h = CpuHandoff(0, layer_index=3, step=1, payload="x")
        assert h.state is HandoffState.NEW
        h.seal_input()
        h.start()
        h.finish("y")
        assert h.join() == "y"
        assert h.history == (
            HandoffState.NEW, HandoffState.SEALED, HandoffState.RUNNING,
            HandoffState.DONE, HandoffState.JOINED,
        )

    def test_SEALED_is_a_real_edge_a_worker_cannot_skip(self):
        """The one that distinguishes "read a recycled staging buffer" from correct behaviour.

        `seal_input` is the point the device->host copy has been OBSERVED complete. A worker that
        started on an unsealed handoff would be reading a buffer whose DMA is still in flight —
        silent wrong numbers, no crash. It is an illegal transition, not a comment.
        """
        h = CpuHandoff(0, 3, 1)
        with pytest.raises(HandoffError, match="new -> running"):
            h.start()

    def test_input_cannot_be_resealed_once_the_worker_is_running(self):
        h = CpuHandoff(0, 3, 1).seal_input("a")
        h.start()
        with pytest.raises(HandoffError, match="running -> sealed"):
            h.seal_input("b")

    def test_a_join_before_the_work_finished_RAISES_it_does_not_return_early(self):
        h = CpuHandoff(0, 3, 1).seal_input()
        h.start()
        with pytest.raises(HandoffError, match="only `done`"):
            h.join()

    def test_double_join_is_refused(self):
        """Exactly-once. A second join of the same partial would double-count it in the residual."""
        h = CpuHandoff(0, 3, 1).seal_input()
        h.start()
        h.finish(1.0)
        assert h.join() == 1.0
        with pytest.raises(HandoffError, match="joined in state joined"):
            h.join()

    def test_a_failure_RAISES_at_join_and_never_becomes_a_zero_partial(self):
        h = CpuHandoff(0, 3, 1).seal_input()
        h.start()
        h.fail(ValueError("boom"))
        with pytest.raises(HandoffError, match="failed: ValueError"):
            h.join()

    def test_a_finished_handoff_cannot_then_fail(self):
        h = CpuHandoff(0, 3, 1).seal_input()
        h.start()
        h.finish(1.0)
        with pytest.raises(HandoffError, match="done -> failed"):
            h.fail(ValueError("late"))


# ════════════════════════════════════════════════════════════════════════════════════════════════
class TestLedgerOrdering:
    """G1-G4 from `HandoffLedger`'s docstring, one test each and then some."""

    def test_G1_joins_are_FIFO(self):
        led = HandoffLedger(max_inflight=3)
        led.begin_step(1)
        a = led.submit(0)
        b = led.submit(1)
        for h in (a, b):
            h.seal_input().start().finish(h.layer_index)
        with pytest.raises(HandoffError, match="out-of-order join"):
            led.join(b)
        assert led.join(a) == 0
        assert led.join(b) == 1

    def test_G2_in_flight_is_bounded_and_BLOCK_mode_bounds_it_to_one(self):
        """max_inflight=1 makes "serial, no overlap" a CHECKED property, not a comment."""
        led = HandoffLedger(max_inflight=1)
        led.begin_step(1)
        led.submit(0)
        with pytest.raises(HandoffError, match="in-flight bound exceeded"):
            led.submit(1)

    def test_G2_a_higher_bound_is_a_SCHEDULE_claim(self):
        led = HandoffLedger(max_inflight=2)
        led.begin_step(1)
        a, b = led.submit(0), led.submit(1)
        assert led.inflight == 2
        for h in (a, b):
            h.seal_input().start().finish(None)
        led.join(a)
        led.join(b)
        assert led.inflight == 0

    def test_G3_a_barrier_refuses_to_pass_with_work_in_flight(self):
        """A captured segment replays device nodes with no host involvement. Nothing may straddle it."""
        led = HandoffLedger()
        led.begin_step(1)
        led.submit(4)
        with pytest.raises(HandoffError, match="still in flight at graph-segment boundary"):
            led.barrier()

    def test_G3_a_drained_barrier_passes(self):
        led = HandoffLedger()
        led.begin_step(1)
        h = led.submit(4).seal_input()
        h.start().finish(0)
        led.join(h)
        led.barrier()  # no raise

    def test_G4_a_handoff_may_not_cross_a_decode_step(self):
        """Otherwise the PREVIOUS token's expert partial lands in THIS token's residual."""
        led = HandoffLedger()
        led.begin_step(1)
        led.submit(0)
        with pytest.raises(HandoffError, match="still in flight"):
            led.begin_step(2)

    def test_G4_a_stale_handoff_is_refused_even_if_the_ledger_moved_on(self):
        led = HandoffLedger()
        led.begin_step(1)
        h = led.submit(0).seal_input()
        h.start().finish(0)
        led.join(h)
        led.begin_step(2)
        h2 = led.submit(0).seal_input()
        h2.start().finish(0)
        # Forge a stale step on the in-flight handoff and confirm the ledger notices.
        h2.step = 1
        with pytest.raises(HandoffError, match="belongs to step 1"):
            led.join(h2)

    def test_a_failed_handoff_still_leaves_the_ledger_consistent(self):
        """The FIFO cursor must advance on the failure path or the next step deadlocks on a barrier."""
        led = HandoffLedger()
        led.begin_step(1)
        h = led.submit(0).seal_input()
        h.start().fail(RuntimeError("x"))
        with pytest.raises(HandoffError):
            led.join(h)
        assert led.inflight == 0
        led.barrier()  # the step can still be closed


# ════════════════════════════════════════════════════════════════════════════════════════════════
class TestWorkerThread:
    """The real dispatcher thread, with the real ledger, on an opaque payload."""

    def test_submit_returns_immediately_and_join_waits(self):
        be = _EchoBackend(delay=0.15)
        w = CpuMoEWorker(be).start()
        try:
            w.begin_step(1)
            t0 = time.perf_counter()
            h = w.submit(0, "x", [1, 2], [0.5, 0.5])
            submit_s = time.perf_counter() - t0
            assert submit_s < 0.05, "submit must not block on the compute"
            out = w.join(h)
            total_s = time.perf_counter() - t0
            assert out == ("partial", "x")
            assert total_s >= 0.15
        finally:
            w.stop()

    def test_the_gpu_really_can_do_work_between_submit_and_join(self):
        """The overlap property, demonstrated rather than asserted: a 150 ms CPU layer costs ~0 ms
        of the submitting thread's time when there is 150 ms of other work to do."""
        be = _EchoBackend(delay=0.15)
        w = CpuMoEWorker(be).start()
        try:
            w.begin_step(1)
            t0 = time.perf_counter()
            h = w.submit(0, "x", [1], [1.0])
            time.sleep(0.15)  # stand-in for the GPU-side partial
            w.join(h)
            assert time.perf_counter() - t0 < 0.28  # ~0.15, not ~0.30
        finally:
            w.stop()

    def test_a_backend_exception_surfaces_at_join_and_is_not_a_zero_partial(self):
        w = CpuMoEWorker(_AngryBackend()).start()
        try:
            w.begin_step(1)
            h = w.submit(0, "x", [1], [1.0])
            with pytest.raises(HandoffError, match="ZeroDivisionError"):
                w.join(h)
            assert w.ledger.inflight == 0
        finally:
            w.stop()

    def test_join_timeout_is_reported_as_a_hang_not_papered_over(self):
        be = _EchoBackend(delay=1.0)
        w = CpuMoEWorker(be, join_timeout_s=0.05).start()
        try:
            w.begin_step(1)
            h = w.submit(0, "x", [1], [1.0])
            with pytest.raises(HandoffError, match="did not complete within"):
                w.join(h)
        finally:
            w._ledger._inflight.clear()  # noqa: SLF001 - the handoff is genuinely stuck
            w.stop()

    def test_many_layers_in_a_step_stay_FIFO_and_serial_under_the_default_bound(self):
        be = _EchoBackend()
        w = CpuMoEWorker(be).start()
        try:
            w.begin_step(1)
            for layer in range(21):
                h = w.submit(layer, f"x{layer}", [layer], [1.0])
                assert w.join(h) == ("partial", f"x{layer}")
            w.barrier("end of step")
            assert w.layers_computed == 21
            assert [s[0] for s in be.seen] == [f"x{i}" for i in range(21)]
        finally:
            w.stop()

    def test_submitting_before_start_is_refused(self):
        w = CpuMoEWorker(_EchoBackend())
        with pytest.raises(CpuTierError, match="not started"):
            w.submit(0, "x", [1], [1.0])

    def test_stop_refuses_while_work_is_in_flight(self):
        be = _EchoBackend(delay=0.2)
        w = CpuMoEWorker(be).start()
        try:
            w.begin_step(1)
            h = w.submit(0, "x", [1], [1.0])
            with pytest.raises(HandoffError, match="still in flight at worker stop"):
                w.stop()
            w.join(h)
        finally:
            w.stop()

    def test_concurrent_submits_from_two_threads_are_still_bounded(self):
        """The ledger is the single point of truth even if a caller gets creative."""
        be = _EchoBackend(delay=0.1)
        w = CpuMoEWorker(be).start()
        errors: list[BaseException] = []
        try:
            w.begin_step(1)
            h = w.submit(0, "x", [1], [1.0])

            def _second():
                try:
                    w.submit(1, "y", [2], [1.0])
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)

            t = threading.Thread(target=_second)
            t.start()
            t.join()
            assert errors and isinstance(errors[0], HandoffError)
            w.join(h)
        finally:
            w.stop()


# ════════════════════════════════════════════════════════════════════════════════════════════════
class TestSplitRoute:
    """SPLIT mode's route partition — the reason SPLIT needs NO kernel change on either side."""

    def test_cpu_slots_are_zero_weighted_and_id_aliased_to_a_kept_expert(self):
        ids = [3, 9, 4, 11, 7]
        w = [0.3, 0.2, 0.2, 0.2, 0.1]
        gi, gw, ci, cw = split_route_for_cpu(ids, w, lambda e: e in (9, 11))
        assert ci == [9, 11] and cw == [0.2, 0.2]
        assert gw == [0.3, 0.0, 0.2, 0.0, 0.1]
        # aliased to the first KEPT expert, so the CPU slots add nothing to the expert UNION the
        # grouped kernel's cost is a function of. And never -1: `moe_align` does an unbounded
        # `atomicAdd(&cnt[topk_ids[t]], 1)`, so a negative id is an OOB shared-memory write.
        assert gi == [3, 3, 4, 3, 7]
        assert all(i >= 0 for i in gi)

    def test_the_two_halves_partition_the_route_exactly(self):
        ids = list(range(10))
        w = [0.1] * 10
        gi, gw, ci, cw = split_route_for_cpu(ids, w, lambda e: e % 3 == 0)
        assert len(gi) == len(gw) == 10  # the GPU route keeps FULL LENGTH (static shapes)
        assert sum(cw) + sum(gw) == pytest.approx(sum(w))
        assert ci == [0, 3, 6, 9]

    def test_an_all_cpu_row_leaves_slot_zero_alone_and_still_contributes_nothing(self):
        gi, gw, ci, cw = split_route_for_cpu([2, 5], [0.6, 0.4], lambda e: True)
        assert gw == [0.0, 0.0]
        assert gi == [2, 2]
        assert ci == [2, 5] and cw == [0.6, 0.4]

    def test_no_cpu_experts_is_the_identity(self):
        gi, gw, ci, cw = split_route_for_cpu([1, 2], [0.5, 0.5], lambda e: False)
        assert (gi, gw, ci, cw) == ([1, 2], [0.5, 0.5], [], [])

    def test_a_truncated_route_pair_is_refused(self):
        with pytest.raises(CpuTierError, match="route mismatch"):
            split_route_for_cpu([1, 2, 3], [0.5, 0.5], lambda e: False)
