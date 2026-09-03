"""Stage A of the weight-offload load path: the ordered window and the seal.

GPU-FREE. The session is driven with a fake `StageADriver`, which is the whole point of that
Protocol: the ORDER of the window and the byte LEDGER are testable without pinning a page, and the
byte-moving itself is `moe_interpose.MoEWeightSeam`'s (covered by its own tests) rather than a second
implementation here.

What is NOT covered without a card: that an arena row's `data_ptr()` really is a
`hipHostGetDevicePointer` address, and that the copy is device-issued. Those are P5b/P2prime's
ground, and they are re-checked at boot by the accounting gate — `host arena costs 0 device bytes` is
exactly the Phase-0 trap detector, and it runs on every offloading serve.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from minisgl.weights import bake as bake_mod  # noqa: E402
from minisgl.weights.accounting import WeightArenaAccounting  # noqa: E402
from minisgl.weights.bake import (  # noqa: E402
    StageAPhase,
    StageASession,
    weight_arena_torch_slack_bytes,
)

GiB = 1 << 30


class _FakeDriver:
    """A `StageADriver` that moves no bytes but reports a plan's worth of them."""

    def __init__(self, *, host=30 * GiB, device=6 * GiB, arena_torch=0, moved=None):
        self._host = host
        self._device = device
        self._arena_torch = arena_torch
        self._moved = host if moved is None else moved
        self.attached = False
        self.bound = False
        self.frozen = False

    def plan_bytes(self):
        return (self._host, self._device, self._host + self._device)

    def attach_host_arena(self):
        self.attached = True

    def bind(self, model):
        self.bound = True
        return "bound"

    def moved_bytes(self):
        return self._moved

    def arena_torch_bytes(self):
        return self._arena_torch

    def freeze(self):
        self.frozen = True

    def describe(self):
        return "fake plan"


def _probe(driver, *, attach_free_drop=0, arena_counts_in_torch=False, released=None):
    """A `MemProbe` walking a healthy card across the four sample points."""
    host = driver.moved_bytes()
    released = host if released is None else released
    free, alloc, res = 40 * GiB, 12 * GiB, 13 * GiB
    growth = host if arena_counts_in_torch else 0
    seq = [
        (free, alloc, res),
        (free - attach_free_drop, alloc, res),
        (free - attach_free_drop, alloc, res),
        (free - attach_free_drop, alloc + growth - released, res + growth),
    ]
    it = iter(seq)
    return lambda: next(it)


@pytest.fixture(autouse=True)
def _clear_published_slack():
    bake_mod._reset_for_tests()
    yield
    bake_mod._reset_for_tests()


class TestDisabledSession:
    """A serve whose model fits must run the identical call sequence and pay nothing for it."""

    def test_full_sequence_is_a_no_op(self):
        s = StageASession.disabled()
        assert not s.enabled
        s.attach()
        s.note_loaded()
        assert s.bind(object()) is None
        s.seal()
        assert s.model_memory_correction() == (0, 0)
        assert s.kv_annotation() == ""
        assert weight_arena_torch_slack_bytes() == 0

    def test_an_all_device_plan_is_also_disabled(self):
        # plan §6.2: the flag CLAMPS an automatic decision; "nothing host-resident" is a legitimate
        # resolved plan, not an error, and it must cost nothing.
        s = StageASession.begin(object(), driver=_FakeDriver(host=0, device=6 * GiB))
        assert not s.enabled

    def test_ledger_is_the_accounting_object(self):
        assert isinstance(StageASession.disabled().accounting, WeightArenaAccounting)


class TestPhaseOrdering:
    def test_skipping_attach_raises(self):
        s = StageASession.disabled()
        with pytest.raises(RuntimeError, match="expected phase"):
            s.note_loaded()

    def test_binding_before_load_raises(self):
        s = StageASession.disabled()
        s.attach()
        with pytest.raises(RuntimeError, match="expected phase"):
            s.bind(object())

    def test_attaching_after_the_seal_raises(self):
        d = _FakeDriver()
        s = StageASession.begin(object(), driver=d, probe=_probe(d))
        s.attach()
        s.note_loaded()
        s.bind(object())
        s.seal()
        with pytest.raises(RuntimeError, match="expected phase"):
            s.attach()

    def test_cannot_seal_twice(self):
        s = StageASession.disabled()
        s.attach()
        s.note_loaded()
        s.bind(object())
        s.seal()
        with pytest.raises(RuntimeError, match="expected phase"):
            s.seal()

    def test_correction_is_zero_before_the_seal(self):
        # Reading it early would apply a correction derived from an incomplete ledger.
        d = _FakeDriver(arena_torch=30 * GiB)
        s = StageASession.begin(object(), driver=d, probe=_probe(d, arena_counts_in_torch=True))
        s.attach()
        s.note_loaded()
        s.bind(object())
        assert s.model_memory_correction() == (0, 0)
        s.seal()
        assert s.model_memory_correction() != (0, 0)


class TestHealthyWindow:
    def _run(self, **probe_kw):
        d = _FakeDriver()
        s = StageASession.begin(object(), driver=d, probe=_probe(d, **probe_kw))
        s.attach()
        s.note_loaded()
        s.bind(object())
        s.seal()
        return s, d

    def test_every_step_reaches_the_driver(self):
        s, d = self._run()
        assert d.attached and d.bound and d.frozen
        assert s.phase is StageAPhase.SEALED

    def test_no_correction_when_the_arena_bypasses_the_torch_allocator(self):
        s, _ = self._run()
        assert s.model_memory_correction() == (0, 0)

    def test_pool_served_arena_produces_the_correction(self):
        d = _FakeDriver(arena_torch=30 * GiB)
        s = StageASession.begin(object(), driver=d, probe=_probe(d, arena_counts_in_torch=True))
        s.attach()
        s.note_loaded()
        s.bind(object())
        s.seal()
        # Without this, 30 GiB of HOST RAM is billed as device memory, available_memory goes hard
        # negative, and the engine refuses to boot with a misleading cause (plan §5.3).
        assert s.model_memory_correction() == (30 * GiB, 30 * GiB)

    def test_slack_is_published_for_the_prefill_guard(self):
        d = _FakeDriver(arena_torch=30 * GiB)
        # Arena reserved 2 GiB more than it allocated: real for a chunked bump allocator, and NOT
        # reusable cache, so _prefill_budget_now must not count it as free VRAM.
        free, alloc, res = 40 * GiB, 12 * GiB, 13 * GiB
        seq = iter(
            [
                (free, alloc, res),
                (free, alloc, res),
                (free, alloc, res),
                (free, alloc, res + 32 * GiB),
            ]
        )
        s = StageASession.begin(object(), driver=d, probe=lambda: next(seq))
        s.attach()
        s.note_loaded()
        s.bind(object())
        s.seal()
        assert weight_arena_torch_slack_bytes() == 2 * GiB


class TestSealGate:
    def test_host_arena_that_took_vram_fails_the_seal(self):
        # THE Phase-0 trap: hipMemCreate(location=Host) silently returned VRAM with the property
        # echoed back verbatim. Nothing is asked of the driver; the card is asked how much is left.
        d = _FakeDriver()
        s = StageASession.begin(object(), driver=d, probe=_probe(d, attach_free_drop=30 * GiB))
        s.attach()
        s.note_loaded()
        s.bind(object())
        with pytest.raises(RuntimeError, match="host arena costs 0 device bytes"):
            s.seal()

    def test_moving_fewer_bytes_than_planned_fails_the_seal(self):
        d = _FakeDriver(moved=20 * GiB)
        s = StageASession.begin(object(), driver=d, probe=_probe(d))
        s.attach()
        s.note_loaded()
        s.bind(object())
        with pytest.raises(RuntimeError, match="bake moved plan.host_resident_bytes"):
            s.seal()

    def test_undropped_originals_fail_the_seal_even_when_pool_served(self):
        # The missed-alias bug: the copy succeeds, the read-back passes, and VRAM never comes back
        # down. When the arena is pool-served its rows and the dropped originals cancel exactly, so
        # this is only visible because the ledger nets out arena_torch_bytes.
        d = _FakeDriver(arena_torch=30 * GiB)
        s = StageASession.begin(
            object(), driver=d, probe=_probe(d, arena_counts_in_torch=True, released=0)
        )
        s.attach()
        s.note_loaded()
        s.bind(object())
        with pytest.raises(RuntimeError, match="torch released the originals"):
            s.seal()

    def test_failure_message_carries_the_whole_sample_table(self):
        d = _FakeDriver()
        s = StageASession.begin(object(), driver=d, probe=_probe(d, attach_free_drop=30 * GiB))
        s.attach()
        s.note_loaded()
        s.bind(object())
        with pytest.raises(RuntimeError) as ei:
            s.seal()
        msg = str(ei.value)
        assert "samples:" in msg and "pre_attach" in msg and "post_bake" in msg

    def test_the_driver_is_frozen_before_the_gate_runs(self):
        # R1 must close even on a failing boot: a half-built arena that is still mappable is worse
        # than one that is not.
        d = _FakeDriver()
        s = StageASession.begin(object(), driver=d, probe=_probe(d, attach_free_drop=30 * GiB))
        s.attach()
        s.note_loaded()
        s.bind(object())
        with pytest.raises(RuntimeError):
            s.seal()
        assert d.frozen


class TestAnnotation:
    def test_delegates_to_the_plan_summary_line(self):
        # One source of truth for the numbers: the KV-sizing line and the [serve] banner must not be
        # able to quote different figures for the same plan.
        class WithSummary(_FakeDriver):
            def summary_line(self):
                return "weight-arena=6.00 GiB (dev tier, inside model) host=30.00 GiBx2"

        d = WithSummary()
        s = StageASession.begin(object(), driver=d, probe=_probe(d))
        s.attach()
        s.note_loaded()
        s.bind(object())
        s.seal()
        assert s.kv_annotation() == " " + d.summary_line()

    def test_driver_without_a_summary_line_contributes_nothing(self):
        d = _FakeDriver()
        s = StageASession.begin(object(), driver=d, probe=_probe(d))
        s.attach()
        s.note_loaded()
        s.bind(object())
        s.seal()
        assert s.kv_annotation() == ""


class TestLogging:
    def test_an_enabled_session_logs_the_plan_the_attach_and_the_ledger(self):
        lines = []
        d = _FakeDriver()
        s = StageASession.begin(object(), driver=d, probe=_probe(d), log=lines.append)
        s.attach()
        s.note_loaded()
        s.bind(object())
        s.seal()
        joined = "\n".join(lines)
        assert "fake plan" in joined
        assert "host arena attached" in joined
        assert "weight-arena accounting" in joined

    def test_a_disabled_session_logs_nothing(self):
        lines = []
        s = StageASession.disabled(log=lines.append)
        s.attach()
        s.note_loaded()
        s.bind(object())
        s.seal()
        assert lines == []
