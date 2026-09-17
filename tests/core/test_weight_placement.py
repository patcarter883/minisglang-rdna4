"""Layer-granular placement arithmetic — the decision that must be identical on every TP rank.

NO GPU, NO TORCH. `minisgl.weights.placement`, `.prior` and `.stacks` are deliberately torch-free
(`stacks` imports torch only inside `TorchStackAllocator` / `as_tensor`), so this whole file runs on
a machine where `import torch` fails outright. That is the point: the part of weight offload that
can be silently wrong by a factor of two is the byte arithmetic, and the byte arithmetic is the part
that is tested here rather than on a real card.
"""

from __future__ import annotations

import pytest
from minisgl.weights.placement import (
    LayerPlacement,
    LayerWeights,
    OffloadPlan,
    PlacementError,
    distinct_experts,
    ep_local_top_k,
    format_sweep,
    plan_layer_granular,
    project_plan,
    sweep_device_fraction,
)
from minisgl.weights.prior import CARD1_GEN5_PRIOR, PHASE0_PRIOR
from minisgl.weights.stacks import ExpertStackTable, StackKind

MiB = 1 << 20
GiB = 1 << 30

# P2prime's measured synthetic granule, so the numbers below are anchored to a real measurement
# rather than a round number: E=512, top_k=10, granule 2.338 MiB (w13+w2, one expert).
P2PRIME_GRANULE = int(2.338 * MiB)


def _layer(i: int, *, n: int = 512, k: int = 10, granule: int = P2PRIME_GRANULE, priority: int = 0):
    return LayerWeights(
        path=f"model.layers[{i}].mlp.experts",
        num_experts=n,
        top_k=k,
        granule_bytes=granule,
        resident_bytes=granule * n,
        priority=priority,
    )


def _layers(count: int = 48, **kw):
    return [_layer(i, **kw) for i in range(count)]


class TestDistinctExperts:
    """`E*(1-(1-k/E)^M)` — plan §1(b). This is why offload is a CAPACITY feature, not a throughput
    one, and the test states that consequence numerically so it cannot quietly stop being true."""

    def test_batch_one_is_exactly_top_k(self):
        assert distinct_experts(512, 10, 1) == pytest.approx(10.0)
        assert distinct_experts(256, 8, 1) == pytest.approx(8.0)

    def test_matches_the_plan_table(self):
        # Plan §1(b), E=512 top_k=10: M=1 -> 9.9, M=6 -> 57, M=16 -> 139, M=32 -> 238.
        assert distinct_experts(512, 10, 6) == pytest.approx(57.0, abs=0.6)
        assert distinct_experts(512, 10, 16) == pytest.approx(139.0, abs=1.0)
        assert distinct_experts(512, 10, 32) == pytest.approx(238.0, abs=2.0)

    def test_bytes_per_token_barely_fall_with_batch(self):
        """Batching 1->6 buys ~5%. If this ever changes, offload became a throughput feature and
        the whole sales pitch (and gate K0) has to be rewritten."""
        per_token_1 = distinct_experts(512, 10, 1) / 1
        per_token_6 = distinct_experts(512, 10, 6) / 6
        assert per_token_6 / per_token_1 == pytest.approx(0.95, abs=0.02)

    def test_saturates_at_num_experts(self):
        assert distinct_experts(8, 8, 1000) == pytest.approx(8.0)

    @pytest.mark.parametrize("args", [(0, 1, 1), (8, 0, 1), (8, 1, 0), (-1, 1, 1)])
    def test_rejects_nonpositive(self, args):
        with pytest.raises(ValueError):
            distinct_experts(*args)


class TestLayerWeightsInvariants:
    def test_traffic_and_capacity_are_different_numbers(self):
        lw = _layer(0)
        assert lw.granule_bytes == P2PRIME_GRANULE
        assert lw.resident_bytes == P2PRIME_GRANULE * 512
        assert lw.active_bytes(batch=1) == pytest.approx(10 * P2PRIME_GRANULE, rel=1e-6)

    def test_capacity_below_traffic_times_experts_is_refused(self):
        """The signature of a spec derived against the GLOBAL expert count on an EP rank."""
        with pytest.raises(PlacementError, match="smaller than"):
            LayerWeights(
                path="l0", num_experts=512, top_k=10,
                granule_bytes=P2PRIME_GRANULE, resident_bytes=P2PRIME_GRANULE * 256,
            )

    @pytest.mark.parametrize("top_k", [0, -1, 513])
    def test_top_k_must_be_in_range(self, top_k):
        with pytest.raises(PlacementError):
            LayerWeights(
                path="l0", num_experts=512, top_k=top_k,
                granule_bytes=P2PRIME_GRANULE, resident_bytes=P2PRIME_GRANULE * 512,
            )


class TestPlanLayerGranular:
    def test_zero_budget_is_all_host(self):
        plan = plan_layer_granular(_layers(8), device_budget_bytes=0)
        assert plan.num_host_layers == 8
        assert plan.num_device_layers == 0
        assert plan.device_resident_bytes == 0
        assert not plan.is_empty

    def test_budget_above_total_is_a_no_op_plan(self):
        layers = _layers(8)
        total = sum(lw.resident_bytes for lw in layers)
        plan = plan_layer_granular(layers, device_budget_bytes=total)
        assert plan.is_empty
        assert plan.num_device_layers == 8
        assert plan.host_resident_bytes == 0
        assert plan.unused_device_bytes == 0

    def test_layers_are_never_split(self):
        """Round DOWN. A partially-placed layer would need the per-expert selector P2prime priced
        at 1.013x at the reachable operating point, so the residual is left unused and REPORTED."""
        layers = _layers(10)
        one = layers[0].resident_bytes
        plan = plan_layer_granular(layers, device_budget_bytes=int(one * 3.7))
        assert plan.num_device_layers == 3
        assert plan.unused_device_bytes == pytest.approx(one * 0.7, rel=1e-6)
        for p in plan.placements:
            assert p.table().is_uniform

    def test_every_layer_is_placed_exactly_once(self):
        layers = _layers(48)
        plan = plan_layer_granular(layers, device_budget_bytes=layers[0].resident_bytes * 5)
        assert len(plan.placements) == 48
        assert {p.path for p in plan.placements} == {lw.path for lw in layers}
        assert plan.device_resident_bytes + plan.host_resident_bytes == plan.total_resident_bytes

    def test_declaration_order_wins_without_a_prior(self):
        layers = _layers(6)
        plan = plan_layer_granular(layers, device_budget_bytes=layers[0].resident_bytes * 2)
        on_dev = [p.path for p in plan.placements if p.kind is StackKind.DEVICE]
        assert on_dev == [layers[0].path, layers[1].path]

    def test_integer_priority_reorders_deterministically(self):
        layers = [_layer(i, priority=(10 if i == 5 else 0)) for i in range(6)]
        plan = plan_layer_granular(layers, device_budget_bytes=layers[0].resident_bytes)
        on_dev = [p.path for p in plan.placements if p.kind is StackKind.DEVICE]
        assert on_dev == [layers[5].path]

    def test_uneven_layers_are_packed_not_abandoned(self):
        """A big layer that does not fit must not stop the walk — the outcome is still a pure
        function of the ordering, so both ranks pack identically."""
        small = _layer(0, granule=1 * MiB, n=4, k=2)
        big = _layer(1, granule=100 * MiB, n=4, k=2)
        small2 = _layer(2, granule=1 * MiB, n=4, k=2)
        plan = plan_layer_granular([big, small, small2], device_budget_bytes=8 * 4 * MiB)
        kinds = {p.path: p.kind for p in plan.placements}
        assert kinds[big.path] is StackKind.HOST
        assert kinds[small.path] is StackKind.DEVICE
        assert kinds[small2.path] is StackKind.DEVICE

    def test_duplicate_paths_are_refused(self):
        """A construction counter renumbers every layer after the MTP draft head builds its own
        MoELayer; a colliding path would place the wrong layers on the wrong stack."""
        dup = [_layer(0), _layer(0)]
        with pytest.raises(PlacementError, match="duplicate layer path"):
            plan_layer_granular(dup, device_budget_bytes=0)

    def test_negative_budget_is_refused(self):
        with pytest.raises(PlacementError):
            plan_layer_granular(_layers(2), device_budget_bytes=-1)


class TestRankAgreement:
    """Two ranks must derive the identical plan or they place granules at different offsets."""

    def test_digest_is_stable_across_repeated_derivation(self):
        layers = _layers(48)
        a = plan_layer_granular(layers, device_budget_bytes=7 * GiB)
        b = plan_layer_granular(list(layers), device_budget_bytes=7 * GiB)
        assert a.digest() == b.digest()

    def test_digest_changes_with_the_budget(self):
        layers = _layers(48)
        a = plan_layer_granular(layers, device_budget_bytes=7 * GiB)
        b = plan_layer_granular(layers, device_budget_bytes=8 * GiB)
        assert a.digest() != b.digest()

    def test_digest_changes_when_one_rank_repacked(self):
        """`_w_rep` on one rank and `_w_op` on the other is a real half-applied-knob state; it
        changes the byte budget, and the digest must not paper over it."""
        a = plan_layer_granular(_layers(4), device_budget_bytes=0)
        b = plan_layer_granular(_layers(4, granule=P2PRIME_GRANULE + 4096), device_budget_bytes=0)
        assert a.digest() != b.digest()

    def test_kind_of_round_trips(self):
        layers = _layers(4)
        plan = plan_layer_granular(layers, device_budget_bytes=layers[0].resident_bytes)
        assert plan.kind_of(layers[0].path) is StackKind.DEVICE
        assert plan.kind_of(layers[3].path) is StackKind.HOST
        with pytest.raises(KeyError):
            plan.kind_of("nope")


class TestProjection:
    """The LINEAR miss-cost model P2prime measured: additive, no cliff, slow rank gates."""

    def test_all_host_at_tp2_uses_the_card1_bandwidth(self):
        layers = _layers(48)
        plan = plan_layer_granular(layers, device_budget_bytes=0)
        proj = project_plan(plan, PHASE0_PRIOR, num_ranks=2)
        assert proj.host_gbps == pytest.approx(14.48)
        # step = compute floor + host bytes / 14.48 GB/s, nothing else.
        expected = PHASE0_PRIOR.compute_floor_ms + plan.host_active_bytes(1) / (14.48e9) * 1000
        assert proj.step_ms == pytest.approx(expected, rel=1e-9)

    def test_single_rank_uses_card0(self):
        plan = plan_layer_granular(_layers(48), device_budget_bytes=0)
        assert project_plan(plan, PHASE0_PRIOR, num_ranks=1).host_gbps == pytest.approx(28.93)

    def test_device_placement_is_monotonically_faster(self):
        layers = _layers(48)
        total = sum(lw.resident_bytes for lw in layers)
        toks = [
            project_plan(
                plan_layer_granular(layers, device_budget_bytes=int(total * f)),
                PHASE0_PRIOR,
                num_ranks=2,
            ).tok_s
            for f in (0.0, 0.10, 0.20, 0.30)
        ]
        assert toks == sorted(toks)

    def test_relative_gains_track_the_phase0_table(self):
        """Plan §7 (as corrected by Phase 0) publishes 1.09x / 1.20x / 1.27x / 1.34x at
        f = 0.10 / 0.20 / 0.25 / 0.30 against the all-host baseline. The ratios are a property of
        the LINEAR model and the byte split, so they must reproduce independently of the absolute
        model size used here."""
        layers = _layers(48)
        total = sum(lw.resident_bytes for lw in layers)
        base = project_plan(
            plan_layer_granular(layers, device_budget_bytes=0), PHASE0_PRIOR, num_ranks=2
        ).tok_s
        for f, want in ((0.10, 1.09), (0.20, 1.20), (0.25, 1.27), (0.30, 1.34)):
            got = project_plan(
                plan_layer_granular(layers, device_budget_bytes=int(total * f)),
                PHASE0_PRIOR,
                num_ranks=2,
            ).tok_s
            assert got / base == pytest.approx(want, abs=0.03), f"f={f}"

    def test_loaded_arm_is_slower(self):
        plan = plan_layer_granular(_layers(48), device_budget_bytes=0)
        idle = project_plan(plan, PHASE0_PRIOR, num_ranks=2)
        loaded = project_plan(plan, PHASE0_PRIOR, num_ranks=2, loaded=True)
        assert loaded.tok_s < idle.tok_s

    def test_card1_gen5_whatif_is_a_separate_prior(self):
        """K7: the BIOS fix is worth ~1.75x, but it is HYPOTHETICAL and must never be the default."""
        plan = plan_layer_granular(_layers(48), device_budget_bytes=0)
        assert PHASE0_PRIOR.slow_host_gbps(2) == pytest.approx(14.48)
        assert CARD1_GEN5_PRIOR.slow_host_gbps(2) == pytest.approx(28.93)
        fixed = project_plan(plan, CARD1_GEN5_PRIOR, num_ranks=2)
        idle = project_plan(plan, PHASE0_PRIOR, num_ranks=2)
        assert fixed.tok_s > idle.tok_s

    def test_a_nonlinear_prior_refuses_to_project(self):
        """If a future box measures a CLIFF, this additive model is wrong and must say so loudly
        rather than stay quietly optimistic."""
        cliffed = PHASE0_PRIOR.with_overrides(miss_cost_is_linear=False, name="hypothetical-cliff")
        plan = plan_layer_granular(_layers(4), device_budget_bytes=0)
        with pytest.raises(PlacementError, match="NOT linear"):
            project_plan(plan, cliffed, num_ranks=2)

    def test_device_bytes_are_charged_at_hbm_speed(self):
        layers = _layers(4)
        total = sum(lw.resident_bytes for lw in layers)
        plan = plan_layer_granular(layers, device_budget_bytes=total)
        proj = project_plan(plan, PHASE0_PRIOR, num_ranks=2)
        assert proj.host_ms == pytest.approx(0.0)
        assert proj.device_ms > 0.0
        assert proj.device_ms < 1.0  # 692 GB/s: the device tier is ~free next to PCIe


class TestSweep:
    def test_sweep_covers_the_prior_grid_and_formats(self):
        rows = sweep_device_fraction(_layers(48), PHASE0_PRIOR, num_ranks=2)
        assert [r.f_requested for r in rows] == list(PHASE0_PRIOR.f_grid)
        text = format_sweep(rows, PHASE0_PRIOR)
        assert "PROJECTED not measured" in text
        # A2.4 is a two-axis frontier: tok/s AND the VRAM it surrenders. Both must be printed.
        assert "dev GiB" in text and "tok/s" in text
        assert "K4 hard-kill" in text

    def test_sweep_host_bytes_shrink_as_f_grows(self):
        rows = sweep_device_fraction(_layers(48), PHASE0_PRIOR, num_ranks=2)
        assert [r.host_gib for r in rows] == sorted((r.host_gib for r in rows), reverse=True)


class TestPlanDescription:
    def test_describe_reports_the_unused_budget(self):
        layers = _layers(10)
        plan = plan_layer_granular(layers, device_budget_bytes=int(layers[0].resident_bytes * 3.5))
        text = plan.describe()
        assert "unused device budget" in text
        assert plan.digest() in text

    def test_empty_plan_still_reports(self):
        layers = _layers(2)
        plan = plan_layer_granular(layers, device_budget_bytes=10 * GiB)
        assert isinstance(plan, OffloadPlan)
        assert plan.is_empty
        assert "0 on HOST" in plan.describe()


class TestExpertStackTable:
    """The residency ledger. Torch-free except `as_tensor`, which is not exercised here."""

    def test_uniform_tables(self):
        t = ExpertStackTable.uniform(512, StackKind.HOST)
        assert len(t) == 512
        assert t.is_uniform and t.uniform_kind is StackKind.HOST
        assert t.count(StackKind.HOST) == 512 and t.count(StackKind.DEVICE) == 0

    def test_uniform_tables_are_canonical_so_as_tensor_is_address_stable(self):
        """REGRESSION, graph capture. `as_tensor` caches its device tensor PER INSTANCE and
        documents that as the reason a captured graph does not bake a dead pointer — but every
        caller minted a fresh instance on every ask (`LayerPlacement.table()` returns
        `ExpertStackTable.uniform(...)`, `MoEWeightSeam.bind` rebuilds its ledger), so the cache
        could never hit and each capture would allocate, bake and drop a new selector. The uniform
        constructor is therefore canonical.
        """
        a = ExpertStackTable.uniform(64, StackKind.HOST)
        assert ExpertStackTable.uniform(64, StackKind.HOST) is a
        assert ExpertStackTable.uniform(64, StackKind.DEVICE) is not a
        assert ExpertStackTable.uniform(32, StackKind.HOST) is not a
        # ...and the transforms are NOT cached, so a per-layer mixed ledger is never aliased.
        m1 = a.with_kind([0], StackKind.DEVICE)
        m2 = a.with_kind([0], StackKind.DEVICE)
        assert m1 is not m2 and m1 == m2
        assert a.is_uniform and a.uniform_kind is StackKind.HOST  # unchanged by the transforms

    def test_two_asks_for_the_same_placement_ledger_share_one_object(self):
        p = LayerPlacement(
            path="model.layers.0.mlp.experts",
            kind=StackKind.HOST,
            resident_bytes=1 << 20,
            granule_bytes=1 << 12,
            num_experts=16,
            top_k=2,
        )
        assert p.table() is p.table()

    def test_device_is_zero_so_a_zeroed_selector_means_no_offload(self):
        assert int(StackKind.DEVICE) == 0
        assert int(StackKind.HOST) == 1

    def test_mixed_table_refuses_to_pretend_it_is_uniform(self):
        t = ExpertStackTable.uniform(8, StackKind.DEVICE).with_kind([3], StackKind.HOST)
        assert not t.is_uniform
        with pytest.raises(ValueError, match="MIXED"):
            _ = t.uniform_kind

    def test_local_view_mirrors_the_ep_global_to_local_remap(self):
        """`MoELayer._ep_dispatch` does `local_ids = where(is_local, g_ids - lo, 0)` and the
        containers are already sized to the local shard, so a global table must be renumbered the
        same way and at the same point."""
        glob = ExpertStackTable.uniform(8, StackKind.DEVICE).with_kind([4, 5], StackKind.HOST)
        rank1 = glob.local_view(4, 4)
        assert len(rank1) == 4
        assert rank1[0] is StackKind.HOST and rank1[1] is StackKind.HOST
        assert rank1[2] is StackKind.DEVICE
        rank0 = glob.local_view(0, 4)
        assert rank0.is_uniform and rank0.uniform_kind is StackKind.DEVICE

    @pytest.mark.parametrize("args", [(-1, 4), (0, 0), (6, 4)])
    def test_local_view_range_is_checked(self, args):
        with pytest.raises(ValueError):
            ExpertStackTable.uniform(8, StackKind.DEVICE).local_view(*args)

    def test_permuted_is_the_a12_mirror(self):
        """A1.2 populates the arena under a fixed permutation of expert rows with the route
        remapped to match; the ledger must follow the rows."""
        t = ExpertStackTable.uniform(4, StackKind.DEVICE).with_kind([0], StackKind.HOST)
        perm = [3, 2, 1, 0]  # row `new` holds what was expert `perm[new]`
        p = t.permuted(perm)
        assert p[3] is StackKind.HOST
        assert p[0] is StackKind.DEVICE
        assert p.permuted([3, 2, 1, 0]) == t

    def test_permuted_rejects_a_non_permutation(self):
        t = ExpertStackTable.uniform(4, StackKind.DEVICE)
        with pytest.raises(ValueError):
            t.permuted([0, 0, 1, 2])

    def test_table_is_immutable(self):
        t = ExpertStackTable.uniform(4, StackKind.DEVICE)
        t2 = t.with_kind([1], StackKind.HOST)
        assert t.is_uniform  # unchanged: residency is frozen at boot, never mutated in a forward
        assert not t2.is_uniform

    def test_rejects_bad_ids_and_empty(self):
        # 2 is `StackKind.CPU` since the CPU-compute tier landed, and is now LEGAL in the ledger; 3
        # is still not a StackKind. See `test_cpu_tier_placement.py` for the CPU tier's own rules —
        # above all that a table containing CPU may not be materialised as a DEVICE SELECTOR, which
        # is the invariant that used to be enforced by this constructor rejecting the value.
        with pytest.raises(ValueError):
            ExpertStackTable([0, 3])
        with pytest.raises(ValueError):
            ExpertStackTable([])


class TestEPLocalTopK:
    """`num_experts` is EP-LOCAL, `MoELayer.top_k` is GLOBAL — pairing them over-counts traffic.

    REGRESSION for a real divergence between the two entry points that build `LayerWeights` for the
    same model: `plan.size_planned_layers_from_model` corrected `top_k` for EP inline while
    `moe_interpose.MoEWeightSeam.layer_weights` passed the global `top_k` straight through. Both now
    call `ep_local_top_k`, which is the only implementation.
    """

    def test_ep1_is_the_identity(self):
        assert ep_local_top_k(10, 1, 512) == 10

    def test_ceil_not_mean(self):
        """The step waits for the SLOWEST rank, so round UP: a rank can draw 5 of an odd 9."""
        assert ep_local_top_k(9, 2, 256) == 5
        assert ep_local_top_k(10, 2, 256) == 5
        assert ep_local_top_k(10, 4, 128) == 3

    def test_more_ranks_than_routed_slots_still_reads_one_expert(self):
        """E=512, k=8, ep=16: a rank owns 32 experts and usually draws 0 or 1. Never 0 —
        `LayerWeights` refuses `top_k=0`, and a zero would price the layer's traffic at nothing."""
        assert ep_local_top_k(8, 16, 32) == 1

    def test_clamped_to_the_local_shard(self):
        assert ep_local_top_k(10, 1, 4) == 4

    def test_rejects_nonpositive(self):
        with pytest.raises(PlacementError):
            ep_local_top_k(0, 1, 8)
        with pytest.raises(PlacementError):
            ep_local_top_k(4, 1, 0)

    def test_the_global_top_k_over_counts_host_traffic_by_ep_size(self):
        """The consequence, in bytes: this is what the seam used to report on an EP=2 rank.

        At M=1, `distinct_experts(E_local, k) == k`, so pairing the local expert count with the
        global `top_k` prices exactly `ep_size` times the granules the rank actually reads. Every
        step-time and tok/s figure derived from it is then wrong by that factor, silently.
        """
        wrong = LayerWeights(
            path="m.0", num_experts=256, top_k=10,  # global k against a local shard
            granule_bytes=P2PRIME_GRANULE, resident_bytes=P2PRIME_GRANULE * 256,
        )
        right = LayerWeights(
            path="m.0", num_experts=256, top_k=ep_local_top_k(10, 2, 256),
            granule_bytes=P2PRIME_GRANULE, resident_bytes=P2PRIME_GRANULE * 256,
        )
        assert wrong.active_bytes(1) == 2 * right.active_bytes(1)

    def test_the_two_planners_now_hash_to_the_same_digest(self):
        """`OffloadPlan.digest()` hashes `top_k`, so the two entry points disagreeing about it also
        reported a spurious cross-rank plan divergence — the one thing the digest exists to detect."""
        k_local = ep_local_top_k(10, 2, 256)
        common = {
            "granule_bytes": P2PRIME_GRANULE,
            "resident_bytes": P2PRIME_GRANULE * 256,
            "num_experts": 256,
        }
        a = plan_layer_granular(
            [LayerWeights(path="m.0", top_k=k_local, **common)], device_budget_bytes=0
        )
        b = plan_layer_granular(
            [LayerWeights(path="m.0", top_k=10, **common)], device_budget_bytes=0
        )
        assert a.digest() != b.digest(), "the digest must be sensitive to top_k at all"
        c = plan_layer_granular(
            [LayerWeights(path="m.0", top_k=ep_local_top_k(10, 2, 256), **common)],
            device_budget_bytes=0,
        )
        assert a.digest() == c.digest()


# =================================================================================================
# REGRESSION (memory-accounting/boot lens, 2026-09-03): the arena's PACKING bound.
#
# The host arena is a set of fixed `hipHostMalloc` chunks and a region may never straddle one, so
# `chunk_plan.BumpAllocator` (forward-only next-fit) abandons a chunk's tail the instant the next row
# does not fit. Reserving `ceil(payload / chunk)` chunks therefore assumes perfect packing, which
# next-fit does not do: on this feature's shape a ~1 GiB fused w13 row in a 2 GiB chunk means one
# layer per chunk and ~25-30% of every chunk abandoned. The overflow rows then come back through
# `ArenaMemPool`'s `hipMalloc` fallback as VRAM on a 16 GB card, budgeted as host RAM — refused by
# `seal()`, but only after the arena is pinned and the checkpoint is loaded, i.e. exactly the late
# failure the reserve/attach split exists to prevent.
#
# `LayerWeights.max_row_bytes` / `OffloadPlan.max_host_row_bytes` are what carry that bound from the
# sizing path to `PinnedWeightArena.reserve(extra_max_region_bytes=)`.
# =================================================================================================


class TestArenaPackingBound:
    def test_row_bound_falls_back_to_the_whole_layer_and_is_never_zero(self):
        """A caller that derived nothing must get a SOUND bound, not the old optimistic `0`.

        `0` means "unknown", and every consumer of `0` goes straight back to `ceil(payload/chunk)`.
        A whole layer is a proven upper bound on any one of its rows, so that is the fallback."""
        lw = _layer(0)
        assert lw.max_row_bytes == 0
        assert lw.row_bound == lw.resident_bytes
        plan = plan_layer_granular([lw], device_budget_bytes=0)
        assert plan.max_host_row_bytes == lw.resident_bytes

    def test_the_bound_survives_into_the_placement_and_the_plan(self):
        rows = 7 * MiB
        lw = LayerWeights(
            path="m.0", num_experts=8, top_k=2,
            granule_bytes=1 * MiB, resident_bytes=16 * MiB, max_row_bytes=rows,
        )
        plan = plan_layer_granular([lw], device_budget_bytes=0)
        assert plan.placements[0].max_row_bytes == rows
        assert plan.placements[0].row_bound == rows
        assert plan.max_host_row_bytes == rows

    def test_only_HOST_layers_contribute_to_the_bound(self):
        """A device-resident layer never touches the arena. Charging its rows against the chunk
        bound reserves host RAM for weights staying in VRAM — and the greedy fill puts the layers it
        can afford on the DEVICE, so including them inflates the bound by exactly what is not there.
        """
        big = LayerWeights(
            path="m.0", num_experts=8, top_k=2,
            granule_bytes=1 * MiB, resident_bytes=64 * MiB, max_row_bytes=40 * MiB,
        )
        small = LayerWeights(
            path="m.1", num_experts=8, top_k=2,
            granule_bytes=1 * MiB, resident_bytes=16 * MiB, max_row_bytes=4 * MiB,
        )
        plan = plan_layer_granular([big, small], device_budget_bytes=64 * MiB)
        assert plan.kind_of("m.0") is StackKind.DEVICE
        assert plan.kind_of("m.1") is StackKind.HOST
        assert plan.max_host_row_bytes == 4 * MiB
        # All-device: no arena is built at all, so there is no bound to report.
        assert plan_layer_granular(
            [big, small], device_budget_bytes=1 * GiB
        ).max_host_row_bytes == 0

    def test_a_row_larger_than_its_layer_is_refused(self):
        """A bound bigger than the layer means it was derived against a different container set, and
        the arena would then be reserved against a packing bound for weights this rank is not
        holding — an over-reservation that refuses boots which would have fitted."""
        with pytest.raises(PlacementError, match="max_row_bytes"):
            LayerWeights(
                path="m.0", num_experts=8, top_k=2,
                granule_bytes=1 * MiB, resident_bytes=16 * MiB, max_row_bytes=17 * MiB,
            )
        with pytest.raises(PlacementError, match="max_row_bytes"):
            LayerWeights(
                path="m.0", num_experts=8, top_k=2,
                granule_bytes=1 * MiB, resident_bytes=16 * MiB, max_row_bytes=-1,
            )

    def test_from_specs_takes_the_largest_COMPONENT_not_the_container(self):
        """The row the arena is asked for is one COMPONENT's whole stacked slab (E x the per-expert
        slice — the kernels index `base + e*row_bytes`, so the stack must be contiguous), plus one
        row per replicated tensor. Not the container, which is ~2x too loose on every quantized
        checkpoint and over-reserves host RAM the box may not have."""

        class _C:
            def __init__(self, name, nbytes):
                self.name, self.nbytes = name, nbytes

        class _Spec:
            num_experts = 8
            num_granules = 8

            def __init__(self, comps, repl=()):
                self.components, self.replicated = comps, repl

            @property
            def granule_bytes(self):
                return sum(c.nbytes for c in self.components)

            @property
            def total_bytes(self):
                return self.granule_bytes * 8 + sum(r.nbytes for r in self.replicated)

            def fingerprint(self):
                return "fp"

        w13 = _Spec([_C("weight", 1 * MiB), _C("weight_scale", 64 * 1024)])
        w2 = _Spec([_C("weight", 512 * 1024)], repl=(_C("_zeros_op", 3 * MiB),))
        lw = LayerWeights.from_specs("m.0", num_experts=8, top_k=2, w13=w13, w2=w2)
        # Largest row = w13.weight stacked over 8 experts = 8 MiB; NOT w13's container total
        # (8.5 MiB) and NOT the layer (12.5 MiB). The replicated 3 MiB row is a candidate too.
        assert lw.max_row_bytes == 8 * MiB
        assert lw.resident_bytes == w13.total_bytes + w2.total_bytes
        assert lw.max_row_bytes < lw.resident_bytes

    def test_from_specs_falls_back_to_total_bytes_for_a_spec_with_no_components(self):
        """A descriptor this module cannot introspect must still yield a SOUND (over-)estimate. The
        packing bound may be loose; it may never be optimistic."""

        class _Opaque:
            num_experts = 4
            granule_bytes = 1 * MiB
            total_bytes = 4 * MiB
            components = None

            def fingerprint(self):
                return "fp"

        lw = LayerWeights.from_specs("m.0", num_experts=4, top_k=1, w13=_Opaque(), w2=_Opaque())
        assert lw.max_row_bytes == 4 * MiB
