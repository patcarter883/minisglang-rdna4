"""VRAM accounting for the weight-offload arena — pure arithmetic, no GPU and no torch.

`minisgl.weights.accounting` is deliberately torch-free, so this whole file runs anywhere. Every
case below is one of the ways the KV pool could be silently mis-sized.
"""

from __future__ import annotations

import pytest
from minisgl.weights.accounting import (
    DEFAULT_TOL_BYTES,
    AccountingReport,
    WeightArenaAccounting,
    gib,
    kv_sizing_annotation,
    step_floor_ms,
)

GiB = 1 << 30


def _acct(
    *,
    host: int,
    device: int,
    offloadable: int,
    copied: int,
    arena_torch: int = 0,
) -> WeightArenaAccounting:
    return WeightArenaAccounting(
        host_bytes=host,
        device_bytes=device,
        offloadable_bytes=offloadable,
        copied_bytes=copied,
        arena_torch_bytes=arena_torch,
    )


def _run(
    acct: WeightArenaAccounting,
    *,
    attach_free_drop: int = 0,
    arena_alloc_growth: int = 0,
    arena_reserved_growth: int = 0,
    released: int | None = None,
) -> AccountingReport:
    """Drive a full window with synthetic samples.

    `released` defaults to `copied_bytes` (the healthy case: the bake dropped exactly what it moved).
    """
    released = acct.copied_bytes if released is None else released
    free, alloc, res = 40 * GiB, 12 * GiB, 13 * GiB
    acct.sample("pre_attach", free=free, allocated=alloc, reserved=res)
    free -= attach_free_drop
    acct.sample("post_attach", free=free, allocated=alloc, reserved=res)
    acct.sample("pre_bake", free=free, allocated=alloc, reserved=res)
    acct.sample(
        "post_bake",
        free=free,
        allocated=alloc + arena_alloc_growth - released,
        reserved=res + arena_reserved_growth,
    )
    return acct.report()


class TestSampleOrdering:
    """Every derived number is a difference between two named samples, so order is correctness."""

    def test_out_of_order_sample_raises(self):
        a = WeightArenaAccounting()
        a.sample("pre_bake", free=1, allocated=2, reserved=3)
        with pytest.raises(RuntimeError, match="ordered"):
            a.sample("pre_attach", free=1, allocated=2, reserved=3)

    def test_duplicate_sample_raises(self):
        a = WeightArenaAccounting()
        a.sample("pre_attach", free=1, allocated=2, reserved=3)
        with pytest.raises(RuntimeError, match="twice"):
            a.sample("pre_attach", free=1, allocated=2, reserved=3)

    def test_unknown_tag_raises(self):
        with pytest.raises(ValueError, match="unknown accounting sample"):
            WeightArenaAccounting().sample("after_the_fact", free=1, allocated=2, reserved=3)

    def test_skipping_a_sample_is_allowed_and_deltas_degrade_to_zero(self):
        # A disabled session takes no samples at all; the ledger must stay usable rather than raise.
        a = WeightArenaAccounting()
        assert a.host_arena_device_cost == 0
        assert a.alloc_delta_bake == 0
        assert a.model_memory_correction() == (0, 0)

    def test_order_is_a_classvar_not_a_field(self):
        # A bare annotated tuple inside a @dataclass becomes a constructor argument; if ORDER ever
        # regressed to a field, this would silently accept a window reordering.
        with pytest.raises(TypeError):
            WeightArenaAccounting(0, 0, 0, 0, 0, 0, {}, [], "extra")


class TestHealthyWindow:
    def test_all_checks_pass_when_the_arena_is_invisible_to_torch(self):
        a = _acct(host=30 * GiB, device=6 * GiB, offloadable=36 * GiB, copied=30 * GiB)
        rep = _run(a)
        assert rep.ok, rep.render()
        assert a.model_memory_correction() == (0, 0)
        assert a.torch_slack_bytes == 0

    def test_all_checks_pass_when_the_arena_is_pool_served(self):
        # P5b's MemPool route: the arena's rows ARE torch allocations, so `allocated` grows by the
        # copied bytes and then falls by the dropped originals — a net delta of zero that must not be
        # mistaken for "nothing was released".
        copied = 30 * GiB
        a = _acct(
            host=copied, device=6 * GiB, offloadable=36 * GiB, copied=copied, arena_torch=copied
        )
        rep = _run(a, arena_alloc_growth=copied, arena_reserved_growth=copied)
        assert rep.ok, rep.render()
        assert a.originals_released == copied
        assert a.alloc_delta_bake == 0  # the trap: a naive check would read this as a leak

    def test_pool_served_arena_produces_the_model_memory_correction(self):
        copied = 30 * GiB
        a = _acct(
            host=copied, device=6 * GiB, offloadable=36 * GiB, copied=copied, arena_torch=copied
        )
        _run(a, arena_alloc_growth=copied, arena_reserved_growth=copied)
        alloc_corr, res_corr = a.model_memory_correction()
        # Without these, ~30 GiB of HOST RAM is billed as device memory and available_memory goes
        # hard negative — the boot refusal plan §5.3 exists to prevent.
        assert alloc_corr == copied
        assert res_corr == copied
        assert a.arena_visible_to_torch


class TestHostArenaTookVram:
    """The single highest-value assertion: Phase 0 caught the driver calling VRAM 'host'."""

    def test_free_vram_drop_across_attach_fails_the_gate(self):
        a = _acct(host=30 * GiB, device=6 * GiB, offloadable=36 * GiB, copied=30 * GiB)
        rep = _run(a, attach_free_drop=30 * GiB)
        assert not rep.ok
        names = [c.name for c in rep.failures]
        assert "host arena costs 0 device bytes" in names

    def test_tolerance_absorbs_page_rounding(self):
        a = _acct(host=30 * GiB, device=6 * GiB, offloadable=36 * GiB, copied=30 * GiB)
        rep = _run(a, attach_free_drop=DEFAULT_TOL_BYTES // 2)
        assert rep.ok, rep.render()


class TestPlanDisagreement:
    def test_copying_fewer_bytes_than_the_plan_priced_fails(self):
        a = _acct(host=30 * GiB, device=6 * GiB, offloadable=36 * GiB, copied=20 * GiB)
        rep = _run(a)
        names = [c.name for c in rep.failures]
        assert "bake moved plan.host_resident_bytes" in names
        # ...and the device tier is then also wrong, because it is (offloadable - copied). Both fire,
        # which is the point: the two numbers are one arithmetic identity.
        assert "device tier == plan.device_resident_bytes" in names

    def test_device_tier_is_counted_from_granules_not_from_the_allocator(self):
        a = _acct(host=30 * GiB, device=6 * GiB, offloadable=36 * GiB, copied=30 * GiB)
        _run(a, arena_alloc_growth=99 * GiB, arena_reserved_growth=99 * GiB)
        # Nonsense allocator movement does not move the device-tier number at all.
        assert a.device_tier_bytes == 6 * GiB

    def test_undropped_originals_fail_even_when_the_arena_is_pool_served(self):
        copied = 30 * GiB
        a = _acct(
            host=copied, device=6 * GiB, offloadable=36 * GiB, copied=copied, arena_torch=copied
        )
        rep = _run(a, arena_alloc_growth=copied, arena_reserved_growth=copied, released=0)
        assert not rep.ok
        assert "torch released the originals" in [c.name for c in rep.failures]

    def test_a_missed_alias_is_caught_by_the_release_check_not_by_the_copy(self):
        """The missed-alias bug, end to end in ledger terms.

        `_GroupedFP8Experts.post_load` leaves `_w_op` and `weight` on one storage. If the rebind
        moves only one of them, the copy succeeds, the granule read-back passes, and VRAM simply
        never comes back down. The ONLY signal is the allocator delta — and only because the ledger
        nets out `arena_torch_bytes`, without which the arena's rows and the drop cancel exactly.
        """
        copied = 30 * GiB
        a = _acct(
            host=copied, device=6 * GiB, offloadable=36 * GiB, copied=copied, arena_torch=copied
        )
        rep = _run(a, arena_alloc_growth=copied, arena_reserved_growth=copied, released=copied // 2)
        assert not rep.ok
        assert "torch released the originals" in [c.name for c in rep.failures]


class TestCorrectionClamps:
    def test_alloc_correction_never_exceeds_the_bytes_actually_copied(self):
        a = _acct(host=8 * GiB, device=0, offloadable=8 * GiB, copied=8 * GiB, arena_torch=64 * GiB)
        _run(a, arena_alloc_growth=64 * GiB)
        # An unclamped correction here would subtract 64 GiB from model_memory and hand the KV pool
        # memory that does not exist.
        assert a.alloc_correction == 8 * GiB

    def test_corrections_are_never_negative(self):
        a = _acct(host=8 * GiB, device=0, offloadable=8 * GiB, copied=8 * GiB)
        _run(a, arena_reserved_growth=-(4 * GiB))
        assert a.reserved_correction == 0
        assert a.alloc_correction == 0

    def test_torch_slack_is_zero_when_the_arena_is_fully_allocated(self):
        copied = 30 * GiB
        a = _acct(host=copied, device=0, offloadable=copied, copied=copied, arena_torch=copied)
        _run(a, arena_alloc_growth=copied, arena_reserved_growth=copied)
        assert a.torch_slack_bytes == 0

    def test_torch_slack_counts_reserved_but_unallocated_arena_segments(self):
        copied = 30 * GiB
        slack = 2 * GiB
        a = _acct(host=copied, device=0, offloadable=copied, copied=copied, arena_torch=copied)
        _run(a, arena_alloc_growth=copied, arena_reserved_growth=copied + slack)
        # Left uncorrected, _prefill_budget_now would believe it had 2 GiB of reusable cache that is
        # actually pinned host RAM, and size a prefill step it cannot afford.
        assert a.torch_slack_bytes == slack


class TestAnnotation:
    def test_empty_when_nothing_is_offloaded(self):
        assert kv_sizing_annotation(device_bytes=0, host_bytes=0, tp_size=2) == ""

    def test_step_floor_omitted_without_a_measured_bandwidth(self):
        s = kv_sizing_annotation(
            device_bytes=6 * GiB, host_bytes=30 * GiB, tp_size=2, host_bytes_per_forward=10**9
        )
        assert "weight-arena=" in s and "step floor" not in s

    def test_step_floor_uses_decimal_gb_per_second(self):
        # 1.0 GB (1e9) at 14.48 GB/s = 69.06 ms. Quoting GiB here would be a silent 7% error.
        assert step_floor_ms(10**9, 14.48) == pytest.approx(69.06, rel=1e-3)

    def test_annotation_reports_both_tiers_and_the_floor(self):
        s = kv_sizing_annotation(
            device_bytes=6 * GiB,
            host_bytes=30 * GiB,
            tp_size=2,
            host_bytes_per_forward=665 * 10**6,
            host_read_gb_s=14.48,
        )
        assert "weight-arena=6.00 GiB (dev tier, inside model)" in s
        assert "host=30.00 GiB" in s
        assert "step floor 45.9 ms" in s  # Phase 0's card-1-gated 45.9 ms per token
        assert "forwards/s" in s

    def test_zero_bandwidth_or_zero_bytes_never_divides_by_zero(self):
        assert step_floor_ms(0, 14.48) == 0.0
        assert step_floor_ms(10**9, 0.0) == 0.0


def test_gib_matches_mem_GB_format():
    # engine.graph.mem_GB is `f"{size / 1024**3:.2f} GiB"`; the boot log mixes the two on one line.
    assert gib(GiB) == "1.00 GiB"
    assert gib(0) == "0.00 GiB"
