"""Boot-order and KV-budget regressions for the weight-offload arena.

GPU-FREE and torch-FREE. Everything here is about the ARITHMETIC and the ORDERING that decide how
big the KV pool comes out, which is the part that fails silently: a mis-sized pool either wastes
~100k tokens or OOMs on the first forward, and neither says "weight offload" anywhere.

Each class below pins one defect found reviewing the boot path.
"""

from __future__ import annotations

import pytest
from minisgl.weights.accounting import corrected_model_memory
from minisgl.weights.bake import StageASession

GiB = 1 << 30


def _uncorrected(allocated: int, reserved: int, device_used: int) -> int:
    """`Engine._determine_num_pages`'s model term exactly as it reads with no offload."""
    return allocated + max(0, device_used - reserved)


class _Driver:
    """Minimal `StageADriver`: reports a plan, moves nothing, records the calls."""

    def __init__(self, *, host=30 * GiB, device=6 * GiB, arena_torch=0):
        self._host, self._device, self._arena_torch = host, device, arena_torch
        self.frozen = False

    def plan_bytes(self):
        return (self._host, self._device, self._host + self._device)

    def attach_host_arena(self):
        pass

    def bind(self, model):
        return "bound"

    def moved_bytes(self):
        return self._host

    def arena_torch_bytes(self):
        return self._arena_torch

    def freeze(self):
        self.frozen = True

    def describe(self):
        return "fake plan"


def _run(session: StageASession) -> None:
    session.attach()
    session.note_loaded()
    session.bind(object())
    session.seal()


# =================================================================================================
# The model term: a seal-time delta applied to a LATER reading
# =================================================================================================


class TestCorrectedModelMemory:
    """`_determine_num_pages` reads `memory_allocated`/`memory_reserved` AFTER
    `_sync_get_memory()` has called `empty_cache()`, but the correction is a delta sampled back at
    `seal()`. The two readings are not the same reading, and the gap between them is roughly the
    arena."""

    def test_no_offload_is_byte_identical(self):
        # The whole feature's cost on a serve that does not offload must be exactly zero.
        for allocated, reserved, used in (
            (12 * GiB, 13 * GiB, 14 * GiB),
            (9 * GiB, 9 * GiB, 9 * GiB),
            (0, 0, 0),
        ):
            model, ca, cr = corrected_model_memory(
                allocated=allocated,
                reserved=reserved,
                device_used=used,
                alloc_correction=0,
                reserved_correction=0,
            )
            assert (ca, cr) == (0, 0)
            assert model == _uncorrected(allocated, reserved, used)

    def test_healthy_pool_served_arena_removes_exactly_the_arena(self):
        arena = 30 * GiB
        model, ca, cr = corrected_model_memory(
            allocated=12 * GiB + arena,
            reserved=13 * GiB + arena,
            device_used=14 * GiB,
            alloc_correction=arena,
            reserved_correction=arena,
        )
        assert (ca, cr) == (arena, arena)
        assert model == _uncorrected(12 * GiB, 13 * GiB, 14 * GiB)

    def test_empty_cache_shrinking_reserved_cannot_inflate_the_model_term(self):
        """REGRESSION. `empty_cache()` between the seal sample and this read releases the segments
        the bake's dropped originals left behind, so `reserved` can fall below the correction. The
        unclamped form then computes `device_used - (negative)` and bills the model up to a whole
        extra arena, which drives `available_memory` hard negative and refuses the boot."""
        arena = 30 * GiB
        # reserved has already fallen back to the true device figure; the seal-time delta has not.
        model, ca, cr = corrected_model_memory(
            allocated=1 * GiB,
            reserved=2 * GiB,
            device_used=3 * GiB,
            alloc_correction=arena,
            reserved_correction=arena,
        )
        assert cr == 2 * GiB, "the reserved correction must be clamped to what is actually reserved"
        assert model >= 0
        # The unclamped arithmetic would have produced this, which is ~30 GiB of pure fiction.
        unclamped = max(0, 1 * GiB - arena) + max(0, 3 * GiB - (2 * GiB - arena))
        assert model < unclamped
        assert model <= 3 * GiB

    def test_model_memory_is_never_negative(self):
        """The mirror failure, and the dangerous one: it does not abort the boot, it over-states
        `available_memory` and sizes a KV pool against VRAM that is already gone."""
        model, ca, _ = corrected_model_memory(
            allocated=4 * GiB,
            reserved=40 * GiB,
            device_used=4 * GiB,
            alloc_correction=30 * GiB,
            reserved_correction=0,
        )
        assert ca == 4 * GiB
        assert model == 0

    def test_applied_corrections_are_what_the_log_must_print(self):
        """The returned figures are the APPLIED ones, not the requested ones — printing the request
        would tell an operator bytes were removed that were not."""
        _, ca, cr = corrected_model_memory(
            allocated=5 * GiB,
            reserved=6 * GiB,
            device_used=0,
            alloc_correction=100 * GiB,
            reserved_correction=100 * GiB,
        )
        assert (ca, cr) == (5 * GiB, 6 * GiB)

    def test_negative_corrections_are_ignored_not_added(self):
        model, ca, cr = corrected_model_memory(
            allocated=5 * GiB,
            reserved=6 * GiB,
            device_used=7 * GiB,
            alloc_correction=-9 * GiB,
            reserved_correction=-9 * GiB,
        )
        assert (ca, cr) == (0, 0)
        assert model == _uncorrected(5 * GiB, 6 * GiB, 7 * GiB)


# =================================================================================================
# The seal gate must have something to gate on
# =================================================================================================


class TestLedgerCompleteness:
    """`WeightArenaAccounting._delta` returns 0 for a missing sample, so on an enabled session an
    unpopulated ledger makes every check pass VACUOUSLY — including `host arena costs 0 device
    bytes`, which is the Phase-0 trap detector and the single highest-value assertion here."""

    def test_enabled_session_without_a_probe_refuses_to_seal(self):
        s = StageASession.begin(object(), driver=_Driver())
        assert s.enabled
        with pytest.raises(RuntimeError, match="ledger is incomplete"):
            _run(s)

    def test_the_refusal_names_the_missing_samples(self):
        s = StageASession.begin(object(), driver=_Driver())
        with pytest.raises(RuntimeError) as exc:
            _run(s)
        for tag in ("pre_attach", "post_attach", "pre_bake", "post_bake"):
            assert tag in str(exc.value)

    def test_a_disabled_session_still_seals_with_no_samples(self):
        # The inert path must keep working: it takes no samples by design.
        s = StageASession.disabled()
        _run(s)
        assert s.model_memory_correction() == (0, 0)

    def test_a_complete_ledger_seals(self):
        d = _Driver()
        free, alloc, res = 40 * GiB, 12 * GiB, 13 * GiB
        seq = iter(
            [
                (free, alloc, res),
                (free, alloc, res),
                (free, alloc, res),
                (free, alloc - d.moved_bytes(), res),
            ]
        )
        s = StageASession.begin(object(), driver=d, probe=lambda: next(seq))
        _run(s)
        assert d.frozen


# =================================================================================================
# The KV-budget failure message
# =================================================================================================


class TestKvBudgetFailureHint:
    """The device tier is billed inside `model_memory` and has no reservation of its own (plan
    §5.3, "no sixth subtrahend"). So a tier that leaves nothing for KV does not fail as "the tier is
    too large" — it fails as a zero-page pool blamed on recurrent state, the draft model or graph
    capture, and every remedy that assert names makes it worse."""

    def _sealed(self, **kw) -> StageASession:
        d = _Driver(**kw)
        free, alloc, res = 40 * GiB, 12 * GiB, 13 * GiB
        seq = iter([(free, alloc, res)] * 3 + [(free, alloc - d.moved_bytes(), res)])
        s = StageASession.begin(object(), driver=d, probe=lambda: next(seq))
        _run(s)
        return s

    def test_no_hint_when_nothing_is_offloaded(self):
        # Byte-identical assert text on every serve that does not offload.
        assert StageASession.disabled().kv_budget_failure_hint() == ""
        assert StageASession.begin(object(), driver=_Driver(host=0)).kv_budget_failure_hint() == ""

    def test_hint_names_the_lever_that_actually_moves(self):
        hint = self._sealed().kv_budget_failure_hint()
        assert "--weight-offload-device-gb" in hint
        # ...and says which way to turn it. "LOWER" is load-bearing: the intuitive direction for an
        # out-of-memory message is to grant MORE device, which makes the pool smaller still.
        assert "LOWER" in hint

    def test_hint_carries_both_tier_sizes(self):
        hint = self._sealed(host=30 * GiB, device=6 * GiB).kv_budget_failure_hint()
        assert "6.00 GiB" in hint and "30.00 GiB" in hint

    def test_hint_warns_off_memory_ratio(self):
        # --memory-ratio shrinks the same budget the tier already consumed.
        assert "--memory-ratio" in self._sealed().kv_budget_failure_hint()

    def test_hint_is_available_before_the_seal_too(self):
        """The assert it feeds can fire on a path where the plan resolved but the ledger did not, so
        it must never itself raise."""
        s = StageASession.begin(object(), driver=_Driver())
        assert "--weight-offload-device-gb" in s.kv_budget_failure_hint()
