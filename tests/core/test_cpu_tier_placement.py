"""The CPU-COMPUTE tier: planner assignment and cost accounting.

NO GPU, NO TORCH. `minisgl.weights.cpu_tier`, `.placement`, `.prior` and `.stacks` are all in the
torch-free planning layer, so this whole file runs on a box where `import torch` fails outright —
which is the state of this box's host, and which is the point: the byte arithmetic is the part that
can be silently wrong by a factor of two, and it is tested here rather than on a leased card.

The numbers this file pins are the SHAPE OF THE REAL MODEL, not round numbers:
Qwen3.8-Flash-Next-NVFP4, 48 layers, 512 experts, top-10, hidden 2560, moe_intermediate 640, i.e.
4,915,200 weights/expert = 3,072,000 B in the device (fp16-folded scale) layout and 2,764,800 B in
the CPU-native (e4m3 scale byte) layout.
"""

from __future__ import annotations

import pytest
from minisgl.weights.cpu_tier import (
    ACT_FP32,
    ACT_VNNI_INT8,
    CORE_BUDGET,
    CPU_TIER_PRIOR,
    CoreBudget,
    CpuTierError,
    CpuTierMode,
    assign_cpu_block,
    cpu_layer_ms,
    ddr_share,
    graph_segments,
    project_cpu_tier,
    split_speedup,
)
from minisgl.weights.placement import (
    LayerWeights,
    OffloadPlan,
    plan_layer_granular,
    plan_three_tier,
)
from minisgl.weights.prior import PHASE0_PRIOR
from minisgl.weights.stacks import ExpertStackTable, StackKind

GiB = 1 << 30
MiB = 1 << 20

# ── the real checkpoint's expert shape ───────────────────────────────────────────────────────────
E_GLOBAL = 512
TOP_K = 10
NUM_MOE_LAYERS = 48
B_EXPERT_DEVICE = 3_072_000  # E2M1 codes 2,457,600 + fp16 group-16 scales 614,400
B_EXPERT_CPU = 2_764_800  # E2M1 codes 2,457,600 + e4m3 group-16 scales 307,200
CPU_LAYOUT_FRACTION = B_EXPERT_CPU / B_EXPERT_DEVICE  # exactly 0.9


def _layer(i: int, *, e: int = E_GLOBAL, k: int = TOP_K, granule: int = B_EXPERT_DEVICE):
    return LayerWeights(
        path=f"model.layers[{i}].mlp.experts",
        num_experts=e,
        top_k=k,
        granule_bytes=granule,
        resident_bytes=granule * e,
    )


def _model(n: int = NUM_MOE_LAYERS, **kw):
    return [_layer(i, **kw) for i in range(n)]


def _ep_shard(n: int = NUM_MOE_LAYERS):
    """The TP=2 + EP view: each rank's containers are already sized to its 256-expert shard."""
    return [_layer(i, e=E_GLOBAL // 2, k=(TOP_K + 1) // 2) for i in range(n)]


def _tp_shard(n: int = NUM_MOE_LAYERS):
    """The TP=2, EP-OFF view — the ONLY sharding the CPU tier currently supports.

    `MoELayer.forward` refuses a CPU-tier layer when `enable_ep` is set: `_ep_dispatch` re-orders
    rows across ranks, so a CPU partial computed from pre-gather rows would be added to the wrong
    tokens. Without EP every rank still routes the full top-10 but holds only half of each
    expert's intermediate, so the per-rank granule is half the device figure and the two ranks
    between them read the WHOLE expert per token — over one DDR bus, which is why
    `project_cpu_tier` scales `cpu_bytes_per_rank` by `num_ranks`.
    """
    return [_layer(i, e=E_GLOBAL, k=TOP_K, granule=B_EXPERT_DEVICE // 2) for i in range(n)]


# ════════════════════════════════════════════════════════════════════════════════════════════════
class TestStackKindThirdTier:
    """`StackKind.CPU` is a THIRD tier, and it is not "HOST but different"."""

    def test_cpu_is_a_distinct_value_and_not_pinned(self):
        assert int(StackKind.CPU) == 2
        assert StackKind.HOST.uses_pinned_arena
        assert not StackKind.CPU.uses_pinned_arena
        assert not StackKind.DEVICE.uses_pinned_arena

    def test_gpu_does_not_read_cpu_tier_weights(self):
        assert StackKind.DEVICE.gpu_reads_weights
        assert StackKind.HOST.gpu_reads_weights  # over PCIe, but it reads them
        assert not StackKind.CPU.gpu_reads_weights

    def test_ledger_accepts_cpu_but_a_device_selector_refuses_it(self):
        """The ledger may record CPU; the on-device `c_*_tab` may never be built from it.

        The device table has exactly two entries. An expert tagged 2 would index past the end of it
        and be dequantized against whatever follows in constant memory — plausible numbers, no
        crash, which is the failure class this whole subsystem's read-back verification exists for.
        """
        t = ExpertStackTable.uniform(8, StackKind.CPU)
        assert t.uniform_kind is StackKind.CPU
        with pytest.raises(ValueError, match="StackKind.CPU"):
            t.as_tensor()  # would import torch, but must refuse BEFORE it gets there

    def test_three_is_still_not_a_stack_kind(self):
        with pytest.raises(ValueError):
            ExpertStackTable([0, 3])


# ════════════════════════════════════════════════════════════════════════════════════════════════
class TestAssignment:
    def test_block_is_taken_from_the_deep_end_and_is_contiguous(self):
        a = assign_cpu_block(range(48), 21)
        assert a.indices == tuple(range(27, 48))
        assert a.contiguous
        assert a.count == 21

    def test_from_start_also_contiguous(self):
        a = assign_cpu_block(range(48), 4, from_end=False)
        assert a.indices == (0, 1, 2, 3)
        assert a.contiguous

    def test_zero_and_off_are_no_ops(self):
        assert assign_cpu_block(range(48), 0).indices == ()
        assert assign_cpu_block(range(48), 0).mode is CpuTierMode.OFF
        assert assign_cpu_block(range(48), 10, mode=CpuTierMode.OFF).indices == ()

    def test_refuses_more_layers_than_exist(self):
        with pytest.raises(CpuTierError, match="only 5 are eligible"):
            assign_cpu_block(range(5), 6)

    def test_refuses_an_unordered_eligible_set(self):
        """Rank-identical placement: the block is a SLICE, so the input order is the decision."""
        with pytest.raises(CpuTierError, match="declaration order"):
            assign_cpu_block([3, 1, 2], 2)

    def test_refuses_duplicates(self):
        with pytest.raises(CpuTierError, match="duplicates"):
            assign_cpu_block([1, 1, 2], 1)

    def test_a_gappy_eligible_set_reports_itself_as_NOT_contiguous(self):
        """The MTP head and refusing containers make holes; the assignment must SAY so, not hide it."""
        a = assign_cpu_block([0, 1, 2, 5, 6], 4)
        assert a.indices == (1, 2, 5, 6)
        assert not a.contiguous


class TestGraphSegments:
    """A CPU layer cuts the captured region INSIDE itself. K CPU layers = K+1 device segments.

    This class was rewritten. The previous version counted maximal runs of GPU *layers* and
    concluded that a contiguous block was nearly free (a 21-layer tail block "cost 1 segment").
    That is wrong in the optimistic direction: a CPU-tier layer's attention, norms, router and
    residual adds are all still GPU work, so the host call lands in the MIDDLE of that layer's
    device work. Contiguity buys nothing here — see `assign_cpu_block` for what it does buy.
    """

    def test_no_cpu_layers_is_one_unbroken_capture(self):
        assert graph_segments(48, (), CpuTierMode.BLOCK) == 1
        assert graph_segments(48, (), CpuTierMode.OFF) == 1

    def test_a_contiguous_tail_block_costs_K_PLUS_ONE_not_one(self):
        assert graph_segments(48, range(27, 48), CpuTierMode.BLOCK) == 22

    def test_position_does_not_matter(self):
        """Tail, middle and head all cost the same: there is device work on both sides regardless."""
        assert graph_segments(48, range(20, 41), CpuTierMode.BLOCK) == 22
        assert graph_segments(48, range(0, 21), CpuTierMode.BLOCK) == 22

    def test_scattering_costs_EXACTLY_THE_SAME_as_a_contiguous_block(self):
        """The number that removes contiguity's capture justification.

        21 layers every other one, versus 21 in a row: identical segment counts. The old test
        asserted 21 vs 1 and that comparison was the stated reason the assignment must be
        contiguous and must be made before the device fill. The ordering rule survives — it is
        needed for rank-determinism — but this was never its reason.
        """
        scattered = tuple(range(0, 42, 2))
        assert len(scattered) == 21
        assert graph_segments(48, scattered, CpuTierMode.BLOCK) == 22
        assert graph_segments(48, range(27, 48), CpuTierMode.BLOCK) == 22

    def test_split_mode_is_not_expressible_as_segments_at_all(self):
        """SPLIT needs the CPU INSIDE a layer's expert reduction. That is a hole, not a cut."""
        assert graph_segments(48, range(27, 48), CpuTierMode.SPLIT) == 0

    def test_all_layers_on_cpu_still_leaves_device_work_between_them(self):
        """Embedding + attention before, final norm + lm_head after. Neither end collapses."""
        assert graph_segments(4, range(4), CpuTierMode.BLOCK) == 5

    def test_no_mode_carrying_a_cpu_layer_claims_to_be_capturable(self):
        """`engine/graph.py` captures the WHOLE forward into ONE CUDAGraph per bs bucket."""
        assert CpuTierMode.OFF.is_capturable
        assert not CpuTierMode.BLOCK.is_capturable
        assert not CpuTierMode.SPLIT.is_capturable


# ════════════════════════════════════════════════════════════════════════════════════════════════
class TestCoreBudget:
    """The constraint that decided the whole design. Denominated in PHYSICAL cores."""

    def test_the_budget_is_physical_cores_minus_the_engine_and_the_os(self):
        b = CORE_BUDGET
        assert b.physical_cores == 8 and b.threads_per_core == 2
        assert b.engine_cores == pytest.approx(1.88)
        assert b.usable_physical == pytest.approx(5.62)
        assert b.max_threads == 5  # floor: a fractional core is not a core

    def test_the_vnni_operating_point_fits_at_tp2_and_the_next_one_up_does_not(self):
        """2 threads/rank x 2 ranks = 4 <= 5. 3/rank = 6 > 5, and that must RAISE."""
        CORE_BUDGET.assert_fits(2 * 2)
        with pytest.raises(CpuTierError, match="only 5 are free"):
            CORE_BUDGET.assert_fits(3 * 2)

    def test_the_fp32_core_never_fitted_and_that_is_why_the_tier_was_useless(self):
        """fp32 needs 6 threads to reach ~46 GB/s; the box has 5 free. VNNI needs 2-3."""
        with pytest.raises(CpuTierError):
            CORE_BUDGET.assert_fits(ACT_FP32.threads_for(44.0, max_threads=16))
        assert ACT_VNNI_INT8.threads_for(44.0, max_threads=5) == 2
        CORE_BUDGET.assert_fits(2)

    def test_starvation_is_a_refusal_and_the_message_says_why(self):
        with pytest.raises(CpuTierError, match="6.0 ms/layer"):
            CORE_BUDGET.assert_fits(8)

    def test_cores_are_taken_from_the_top_leaving_core_0_to_the_engine(self):
        """Core 0 is the only one that boosts; the VNNI kernel is clock-insensitive, the engine is not."""
        assert CORE_BUDGET.core_ids(4) == (4, 5, 6, 7)
        assert 0 not in CORE_BUDGET.core_ids(5)

    def test_smt_siblings_are_named_so_they_can_be_excluded(self):
        """Pinning onto the sibling of a busy core measured ~50% slower."""
        assert CORE_BUDGET.smt_sibling_ids((6, 7)) == (14, 15)

    def test_a_bigger_box_relaxes_it_arithmetically(self):
        big = CoreBudget(physical_cores=16, engine_cores=1.88, os_cores=0.5)
        assert big.max_threads == 13
        big.assert_fits(6)


# ════════════════════════════════════════════════════════════════════════════════════════════════
class TestZeroSumDDR:
    """CPU compute does not ADD bandwidth. It removes the PCIe cap on the share it takes."""

    def test_demands_under_the_wall_do_not_contend_at_all(self):
        """Measured: half a TFLOP of AVX-512 on the neighbouring cores cost the MoE 2%."""
        assert ddr_share(20.0, 12.36) == (20.0, 12.36)

    def test_demands_over_the_wall_are_scaled_and_derated(self):
        cpu, pcie = ddr_share(53.5, 12.36)
        assert cpu + pcie < CPU_TIER_PRIOR.ddr_wall_gbps
        assert cpu < 53.5 and pcie < 12.36
        # proportional share x the single measured contention derate
        assert cpu == pytest.approx(53.5 * 57.0 / 65.86 * 0.815, rel=1e-3)

    def test_the_derate_is_ONE_calibration_point_and_says_so(self):
        assert CPU_TIER_PRIOR.ddr_contention_points == 1
        assert "deliberate worst case" in CPU_TIER_PRIOR.provenance

    def test_split_is_worth_six_percent_at_the_operating_point(self):
        """The measured reason SPLIT is not the default.

        At 4 node threads the CPU tier alone already takes 53.5 of the ~57 GB/s wall, so the GPU
        half of a split has almost nothing left to contribute.
        """
        assert split_speedup(53.50, 12.36) == pytest.approx(1.065, abs=0.005)

    def test_split_matters_more_when_the_cpu_is_NOT_saturating_the_bus(self):
        """Not a blanket dismissal: at 2 node threads (44.64 GB/s) a split is worth 28%."""
        assert split_speedup(44.64, 12.36) == pytest.approx(1.277, abs=0.005)

    def test_split_can_never_exceed_the_wall(self):
        assert split_speedup(10.0, 1000.0) == pytest.approx(5.7, abs=1e-6)


# ════════════════════════════════════════════════════════════════════════════════════════════════
class TestPlannerAssignment:
    def test_cpu_layers_are_placed_and_are_neither_device_nor_host(self):
        plan = plan_three_tier(_model(), device_budget_bytes=0, num_cpu_layers=21)
        kinds = [p.kind for p in plan.placements]
        assert kinds.count(StackKind.CPU) == 21
        assert kinds.count(StackKind.HOST) == 27
        assert kinds.count(StackKind.DEVICE) == 0
        assert plan.num_cpu_layers == 21
        assert plan.cpu_layer_indices == tuple(range(27, 48))
        assert plan.cpu_block_is_contiguous

    def test_cpu_is_decided_BEFORE_the_device_fill_so_the_block_stays_contiguous(self):
        """The ordering that matters. A device-first planner would scatter the CPU set.

        Here the budget takes 4 layers. If the greedy fill ran first it would take layers 0-3 and
        the CPU block would still be contiguous by luck — so the test uses a PRIORITY that pulls
        the device fill into the middle of what would otherwise be the CPU block, and asserts the
        block is unmoved.
        """
        layers = _model()
        layers = [
            LayerWeights(
                path=lw.path,
                num_experts=lw.num_experts,
                top_k=lw.top_k,
                granule_bytes=lw.granule_bytes,
                resident_bytes=lw.resident_bytes,
                priority=(100 if 30 <= i <= 35 else 0),
            )
            for i, lw in enumerate(layers)
        ]
        budget = 4 * layers[0].resident_bytes
        plan = plan_three_tier(layers, device_budget_bytes=budget, num_cpu_layers=21)
        assert plan.cpu_layer_indices == tuple(range(27, 48))
        assert plan.cpu_block_is_contiguous
        # The high-priority layers 30-35 are INSIDE the CPU block, so the device fill could not
        # take them and fell back to declaration order over the remaining 0..26.
        dev = [i for i, p in enumerate(plan.placements) if p.kind is StackKind.DEVICE]
        assert dev == [0, 1, 2, 3]
        # The segment count is a function of HOW MANY CPU layers there are, not of where they sit
        # — which is exactly why this ordering rule is justified by rank-determinism and not by
        # capture. See TestGraphSegments.
        assert graph_segments(len(layers), plan.cpu_layer_indices, CpuTierMode.BLOCK) == 22

    def test_device_budget_is_not_spent_on_cpu_layers(self):
        """A CPU layer needs no device bytes; giving it budget would spend the budget twice."""
        layers = _model(8)
        per = layers[0].resident_bytes
        plan = plan_three_tier(layers, device_budget_bytes=8 * per, num_cpu_layers=3)
        assert plan.num_cpu_layers == 3
        assert plan.num_device_layers == 5  # all five NON-cpu layers fit
        assert plan.device_resident_bytes == 5 * per
        # ... and the leftover budget is REPORTED, not silently consumed.
        assert plan.unused_device_bytes == 3 * per

    def test_cpu_eligible_excludes_layers_by_policy(self):
        """The MTP draft head is excluded from offload by policy; it must be excluded here too."""
        plan = plan_three_tier(
            _model(10), device_budget_bytes=0, num_cpu_layers=3, cpu_eligible=range(8)
        )
        assert plan.cpu_layer_indices == (5, 6, 7)
        assert plan.placements[8].kind is StackKind.HOST
        assert plan.placements[9].kind is StackKind.HOST

    def test_zero_cpu_layers_reproduces_the_shipped_two_tier_plan_EXACTLY(self):
        """Including the digest — the CPU tier must be invisible when it is not used."""
        layers = _model()
        budget = 6 * layers[0].resident_bytes
        old = plan_layer_granular(layers, device_budget_bytes=budget)
        new = plan_three_tier(layers, device_budget_bytes=budget, num_cpu_layers=0)
        assert new.digest() == old.digest()
        assert [p.kind for p in new.placements] == [p.kind for p in old.placements]

    def test_digest_separates_repacked_from_verbatim_cpu_placement(self):
        """Two ranks that disagree about whether a repack ran hold different byte counts."""
        layers = _model(8)
        a = plan_three_tier(layers, device_budget_bytes=0, num_cpu_layers=3)
        b = plan_three_tier(
            layers, device_budget_bytes=0, num_cpu_layers=3,
            cpu_layout_fraction=CPU_LAYOUT_FRACTION,
        )
        assert a.digest() != b.digest()

    def test_a_cpu_only_plan_is_not_is_empty(self):
        """`is_empty` means "the offload path moves nothing". A CPU tier moves plenty."""
        plan = plan_three_tier(_model(4), device_budget_bytes=0, num_cpu_layers=4)
        assert plan.num_host_layers == 0
        assert not plan.is_empty

    def test_refuses_a_negative_layout_fraction(self):
        with pytest.raises(Exception):
            LayerWeights(
                path="x", num_experts=4, top_k=1, granule_bytes=8, resident_bytes=32,
                cpu_layout_fraction=0.0,
            )


# ════════════════════════════════════════════════════════════════════════════════════════════════
class TestCostAccounting:
    """The four byte questions a CPU layer answers DIFFERENTLY from a host layer."""

    def test_cpu_layers_consume_zero_device_bytes(self):
        plan = plan_three_tier(_model(), device_budget_bytes=0, num_cpu_layers=21)
        assert plan.device_resident_bytes == 0
        assert plan.device_active_bytes(1) == 0

    def test_cpu_layers_consume_zero_PINNED_arena_bytes(self):
        """The headline capacity property. `host_resident_bytes` must not see them at all."""
        layers = _model()
        base = plan_three_tier(layers, device_budget_bytes=0, num_cpu_layers=0)
        with_cpu = plan_three_tier(layers, device_budget_bytes=0, num_cpu_layers=21)
        per = layers[0].resident_bytes
        assert base.host_resident_bytes == 48 * per
        assert with_cpu.host_resident_bytes == 27 * per
        assert with_cpu.pinned_arena_bytes_saved_vs_host == 21 * per

    def test_the_pinned_arena_RESERVATION_shrinks_too_not_just_the_payload(self):
        """`host_row_requests` / `max_host_row_bytes` are what the arena actually pins."""
        rows = (("w13.weight", B_EXPERT_DEVICE * E_GLOBAL * 2 // 3),
                ("w2.weight", B_EXPERT_DEVICE * E_GLOBAL - B_EXPERT_DEVICE * E_GLOBAL * 2 // 3))
        layers = [
            LayerWeights(
                path=f"L{i}", num_experts=E_GLOBAL, top_k=TOP_K,
                granule_bytes=B_EXPERT_DEVICE, resident_bytes=B_EXPERT_DEVICE * E_GLOBAL,
                rows=rows,
            )
            for i in range(8)
        ]
        plan = plan_three_tier(layers, device_budget_bytes=0, num_cpu_layers=5)
        names = [r.name for r in plan.host_row_requests()]
        assert len(names) == 3 * 2  # only the 3 HOST layers ask the arena for anything
        assert all(n.startswith(("L0.", "L1.", "L2.")) for n in names)
        assert plan.host_rows_known

    def test_cpu_resident_bytes_use_the_CPU_layout_and_host_bytes_use_the_device_layout(self):
        layers = _model()
        per = layers[0].resident_bytes
        plan = plan_three_tier(
            layers, device_budget_bytes=0, num_cpu_layers=21,
            cpu_layout_fraction=CPU_LAYOUT_FRACTION,
        )
        assert plan.cpu_resident_bytes == pytest.approx(21 * per * 0.9, rel=1e-9)
        assert plan.cpu_resident_bytes_device_layout == 21 * per
        # ... and the pinned-arena relief is priced in the DEVICE layout, because that is what those
        # layers WOULD have pinned.
        assert plan.pinned_arena_bytes_saved_vs_host == 21 * per

    def test_verbatim_placement_may_not_claim_the_shrink(self):
        """`repacked=False` -> 1.0, unconditionally. The 0.9 is a property of a repack that RAN."""
        assert CPU_TIER_PRIOR.layout_fraction_for("nvfp4_e4m3_g16", repacked=False) == 1.0
        assert CPU_TIER_PRIOR.layout_fraction_for("nvfp4_e4m3_g16", repacked=True) == 0.9
        assert CPU_TIER_PRIOR.layout_fraction_for("some_unmeasured_format", repacked=True) == 1.0
        plan = plan_three_tier(_model(4), device_budget_bytes=0, num_cpu_layers=2)
        assert plan.cpu_resident_bytes == plan.cpu_resident_bytes_device_layout

    def test_total_host_bytes_counts_BOTH_pinned_and_pageable(self):
        """MemAvailable does not care whether a page is pinned; only the hipHostMalloc ceiling does."""
        layers = _model()
        per = layers[0].resident_bytes
        plan = plan_three_tier(
            layers, device_budget_bytes=0, num_cpu_layers=21,
            cpu_layout_fraction=CPU_LAYOUT_FRACTION,
        )
        assert plan.total_host_bytes == plan.host_resident_bytes + plan.cpu_resident_bytes
        assert plan.total_host_bytes == pytest.approx(27 * per + 21 * per * 0.9, rel=1e-9)

    def test_cpu_ACTIVE_bytes_are_ddr_traffic_in_the_cpu_layout(self):
        plan = plan_three_tier(
            _model(), device_budget_bytes=0, num_cpu_layers=21,
            cpu_layout_fraction=CPU_LAYOUT_FRACTION,
        )
        # top-10 distinct experts at batch 1, per layer, in the e4m3 layout.
        assert plan.cpu_active_bytes(1) == pytest.approx(21 * TOP_K * B_EXPERT_CPU, rel=1e-6)
        # The C harness's headline unit: 27.648 MB per layer-token.
        assert plan.cpu_active_bytes(1) / 21 == pytest.approx(27_648_000, rel=1e-6)

    def test_host_active_bytes_lose_exactly_the_cpu_layers(self):
        layers = _model()
        base = plan_three_tier(layers, device_budget_bytes=0, num_cpu_layers=0)
        with_cpu = plan_three_tier(layers, device_budget_bytes=0, num_cpu_layers=21)
        assert base.host_active_bytes(1) == 48 * TOP_K * B_EXPERT_DEVICE
        assert with_cpu.host_active_bytes(1) == 27 * TOP_K * B_EXPERT_DEVICE


# ════════════════════════════════════════════════════════════════════════════════════════════════
class TestTheCapacityWin:
    """The number this feature is being sold on, derived rather than asserted.

    TP=2 with EP: each rank's containers hold 256 of the 512 experts, so a layer is 0.7325 GiB/rank.
    37 host layers = 27.1 GiB/rank = 54.2 GiB node-wide, against `usable_host_arena_bytes()` =
    62 GiB x 0.90 = 55.8 GiB. That 97%-of-ceiling near-miss is the shipped plan, and P3b reached the
    62 GiB only by swapping 114,813 pages.
    """

    def test_the_shipped_plan_is_at_97_percent_of_the_pinned_ceiling(self):
        layers = _ep_shard()
        plan = plan_three_tier(
            layers, device_budget_bytes=11 * layers[0].resident_bytes, num_cpu_layers=0
        )
        assert plan.num_host_layers == 37
        node = plan.host_resident_bytes * 2
        ceiling = PHASE0_PRIOR.usable_host_arena_bytes()
        assert node / ceiling == pytest.approx(0.97, abs=0.02)

    def test_moving_21_layers_to_the_cpu_tier_relieves_it_to_42_percent(self):
        layers = _ep_shard()
        plan = plan_three_tier(
            layers,
            device_budget_bytes=11 * layers[0].resident_bytes,
            num_cpu_layers=21,
            cpu_layout_fraction=CPU_LAYOUT_FRACTION,
        )
        assert (plan.num_device_layers, plan.num_host_layers, plan.num_cpu_layers) == (11, 16, 21)
        node = plan.host_resident_bytes * 2
        ceiling = PHASE0_PRIOR.usable_host_arena_bytes()
        assert node / ceiling == pytest.approx(0.42, abs=0.02)

    def test_the_win_in_gib_node_wide(self):
        """PINNED relief + LAYOUT shrink, reported separately because they are different ceilings."""
        layers = _ep_shard()
        budget = 11 * layers[0].resident_bytes
        base = plan_three_tier(layers, device_budget_bytes=budget, num_cpu_layers=0)
        new = plan_three_tier(
            layers, device_budget_bytes=budget, num_cpu_layers=21,
            cpu_layout_fraction=CPU_LAYOUT_FRACTION,
        )
        pinned_relief_node = (base.host_resident_bytes - new.host_resident_bytes) * 2 / GiB
        layout_shrink_node = (
            new.cpu_resident_bytes_device_layout - new.cpu_resident_bytes
        ) * 2 / GiB
        total_host_saved_node = (base.host_resident_bytes - new.total_host_bytes) * 2 / GiB
        assert pinned_relief_node == pytest.approx(30.77, abs=0.05)
        assert layout_shrink_node == pytest.approx(3.08, abs=0.05)
        assert total_host_saved_node == pytest.approx(3.08, abs=0.05)
        # The headline: 30.8 GiB of PINNED arena node-wide that no longer has to be pinned, of
        # which 3.1 GiB stops being resident at all. Both are real and they are not the same claim.

    def test_device_bytes_are_UNCHANGED_the_win_is_not_vram(self):
        layers = _ep_shard()
        budget = 11 * layers[0].resident_bytes
        base = plan_three_tier(layers, device_budget_bytes=budget, num_cpu_layers=0)
        new = plan_three_tier(layers, device_budget_bytes=budget, num_cpu_layers=21)
        assert new.device_resident_bytes == base.device_resident_bytes
        assert new.num_device_layers == base.num_device_layers


# ════════════════════════════════════════════════════════════════════════════════════════════════
class TestPrior:
    def test_the_shipped_policy_is_the_VNNI_one_and_the_ORACLE_is_fp32(self):
        assert CPU_TIER_PRIOR.policy is ACT_VNNI_INT8
        assert CPU_TIER_PRIOR.oracle is ACT_FP32
        assert "RESULTS_VNNI_2026-09-04" in CPU_TIER_PRIOR.provenance

    def test_the_shipped_bandwidths_are_the_measured_ones(self):
        assert ACT_VNNI_INT8.gbps_by_threads[1] == 27.38
        assert ACT_VNNI_INT8.gbps_by_threads[2] == 44.64
        assert ACT_VNNI_INT8.gbps_by_threads[3] == 54.00
        assert ACT_FP32.gbps_by_threads[6] == 46.39
        assert ACT_FP32.gbps_by_threads[16] == 55.29

    def test_the_curves_are_NODE_wide_aggregates_and_both_end_at_the_same_wall(self):
        """VNNI saturates at 3 threads, fp32 at 16. Same ~56 GB/s. Occupancy win, not bandwidth."""
        assert ACT_VNNI_INT8.saturates_at_threads == 3
        assert ACT_FP32.saturates_at_threads == 16
        assert max(ACT_VNNI_INT8.gbps_by_threads.values()) == pytest.approx(
            max(ACT_FP32.gbps_by_threads.values()), abs=1.5
        )
        assert CPU_TIER_PRIOR.ddr_wall_gbps == pytest.approx(57.0)

    def test_the_default_is_2_threads_PER_RANK_not_6(self):
        """The whole reason the tier became viable. 2/rank x 2 ranks = 4 cores, and 4 fits."""
        assert CPU_TIER_PRIOR.default_threads_per_rank == 2
        CPU_TIER_PRIOR.cores.assert_fits(CPU_TIER_PRIOR.default_threads_per_rank * 2)

    def test_the_accuracy_cost_travels_with_the_speed(self):
        """A policy cannot be quoted for throughput without carrying its measured error."""
        assert ACT_VNNI_INT8.rel_rms == pytest.approx(8.279e-03)
        assert ACT_FP32.rel_rms == pytest.approx(2.416e-07)
        assert ACT_VNNI_INT8.rel_rms / ACT_FP32.rel_rms > 3e4
        # ...and the comparison that actually decides it is in the provenance, not in a report.
        assert "4.094e-02" in ACT_VNNI_INT8.provenance

    def test_unmeasured_thread_counts_resolve_DOWNWARD(self):
        """Both curves are non-monotonic past the knee; interpolating up would invent throughput."""
        assert ACT_VNNI_INT8.gbps(5) == ACT_VNNI_INT8.gbps(4)
        assert ACT_FP32.gbps(7) == ACT_FP32.gbps(6)
        assert ACT_VNNI_INT8.gbps(64) == ACT_VNNI_INT8.gbps(16)
        with pytest.raises(CpuTierError, match="threads must be >= 1"):
            CPU_TIER_PRIOR.policy.gbps(0)

    def test_threads_for_refuses_a_target_the_core_cannot_reach_in_budget(self):
        with pytest.raises(CpuTierError, match="cannot reach"):
            ACT_FP32.threads_for(50.0, max_threads=5)

    def test_layout_bytes_are_exact_arithmetic(self):
        p = CPU_TIER_PRIOR
        weights = 2 * 640 * 2560 + 2560 * 640
        assert weights == 4_915_200
        assert p.bytes_per_expert_cpu_native == weights // 2 + weights // 16
        assert p.bytes_per_expert_gpu_fp16_scale == weights // 2 + 2 * (weights // 16)
        assert p.bytes_per_expert_cpu_native / p.bytes_per_expert_gpu_fp16_scale == 0.9

    def test_the_vnni_wload_also_claims_the_shrink_but_only_when_repacked(self):
        p = CPU_TIER_PRIOR
        assert p.layout_fraction_for("vnni_nvfp4_e4m3_g16", repacked=True) == 0.9
        assert p.layout_fraction_for("vnni_nvfp4_e4m3_g16", repacked=False) == 1.0
        assert p.layout_fraction_for("vnni_nvfp4_fp16_g16", repacked=True) == 1.0
        assert p.layout_fraction_for("some_new_format", repacked=True) == 1.0

    def test_the_handoff_is_declared_UNMEASURED(self):
        """Nothing may quietly inherit an optimistic activation round-trip."""
        assert CPU_TIER_PRIOR.handoff_measured is False
        assert CPU_TIER_PRIOR.handoff_us_bracket == (10.0, 40.0)

    def test_the_projection_fidelity_is_carried_and_is_ONE_point(self):
        """11.85 measured / 18.6 projected on the shipped two-tier plan."""
        assert CPU_TIER_PRIOR.projection_fidelity == pytest.approx(0.637)
        assert CPU_TIER_PRIOR.projection_fidelity_points == 1


class TestProjection:
    """The step model. Every number here is PROJECTED and the object says so."""

    def _plan(self, ncpu: int):
        layers = _tp_shard()
        return plan_three_tier(
            layers,
            device_budget_bytes=11 * layers[0].resident_bytes,
            num_cpu_layers=ncpu,
            cpu_layout_fraction=CPU_LAYOUT_FRACTION,
        )

    def _project(self, plan, mode, *, handoff=40.0, threads=2):
        return project_cpu_tier(
            host_bytes_per_rank=plan.host_active_bytes(1),
            device_bytes_per_rank=plan.device_active_bytes(1),
            cpu_bytes_per_rank=plan.cpu_active_bytes(1),
            num_cpu_layers=plan.num_cpu_layers,
            mode=mode,
            handoff_us_per_layer=handoff,
            threads_per_rank=threads,
            num_ranks=2,
            num_layers=NUM_MOE_LAYERS,
        )

    def test_a_cpu_layer_is_cheaper_than_a_streamed_layer_even_FULLY_SERIAL(self):
        """The claim BLOCK mode rests on, and it needs no overlap to be true.

        Streamed: one rank's half of 10 experts over its own PCIe link, both ranks concurrent.
        CPU: BOTH ranks' halves — i.e. the whole layer — over ONE DDR bus at 4 node threads.
        """
        streamed_ms = (
            TOP_K * (B_EXPERT_DEVICE // 2)
            / (PHASE0_PRIOR.slow_host_gbps(2, loaded=True) * 1e9) * 1e3
        )
        cpu_ms = cpu_layer_ms(TOP_K * B_EXPERT_CPU, threads=4)
        assert streamed_ms == pytest.approx(1.243, abs=0.01)
        assert cpu_ms == pytest.approx(0.517, abs=0.01)
        assert streamed_ms / cpu_ms > 2.4

    def test_the_core_budget_is_checked_BEFORE_any_throughput_is_claimed(self):
        with pytest.raises(CpuTierError, match="only 5 are free"):
            self._project(self._plan(21), CpuTierMode.BLOCK, threads=3)

    def test_block_mode_adds_the_cpu_term_and_overlaps_nothing(self):
        p = self._project(self._plan(21), CpuTierMode.BLOCK)
        assert p.overlapped_ms == 0.0
        assert not p.ddr_contended  # serial phases do not share the bus
        assert p.step_ms == pytest.approx(
            p.compute_ms + p.device_ms + p.host_ms + p.cpu_ms + p.handoff_ms, rel=1e-9
        )
        assert p.handoff_measured is False
        assert p.total_threads == 4

    def test_the_cpu_term_is_NODE_traffic_not_per_rank(self):
        """Both ranks' halves cross one memory controller; dividing per-rank bytes by a node
        bandwidth would under-count the traffic by exactly `num_ranks`."""
        plan = self._plan(21)
        p = self._project(plan, CpuTierMode.BLOCK)
        expected = plan.cpu_active_bytes(1) * 2 / (53.50 * 1e9) * 1e3
        assert p.cpu_ms == pytest.approx(expected, rel=1e-9)

    def test_block_mode_beats_the_all_streamed_plan(self):
        a = self._project(self._plan(0), CpuTierMode.OFF, handoff=0.0)
        b = self._project(self._plan(21), CpuTierMode.BLOCK)
        assert b.tok_s / a.tok_s == pytest.approx(1.37, abs=0.03)

    def test_moving_EVERY_host_layer_is_better_still_and_empties_the_arena(self):
        p = self._project(self._plan(37), CpuTierMode.BLOCK)
        assert p.host_ms == 0.0
        assert p.tok_s == pytest.approx(35.3, abs=0.5)

    def test_the_raw_projection_is_paired_with_the_measured_fidelity(self):
        """The model over-predicted the ONE plan anybody has run by 1.57x. Both numbers ship."""
        base = self._project(self._plan(0), CpuTierMode.OFF, handoff=0.0)
        assert base.tok_s == pytest.approx(18.6, abs=0.2)
        assert base.calibrated_tok_s == pytest.approx(11.85, abs=0.1)  # the MEASURED serve
        best = self._project(self._plan(37), CpuTierMode.BLOCK)
        assert best.calibrated_tok_s == pytest.approx(22.5, abs=0.3)
        assert best.calibrated_tok_s < 23.3, (
            "at the measured model fidelity the CPU tier lands just SHORT of the 23.3 tok/s "
            "target; a report that quotes the raw 35.3 is quoting a number this same model got "
            "wrong by 1.57x on the only case that has ever been run"
        )

    def test_the_projection_names_its_capture_cost(self):
        p = self._project(self._plan(37), CpuTierMode.BLOCK)
        assert p.graph_segments == 38
        assert "38 capture segment(s)" in p.describe()

    def test_split_mode_puts_both_halves_on_the_SAME_bus(self):
        """The correction. The old model added a PCIe term and a DDR term as independent."""
        p = self._project(self._plan(21), CpuTierMode.SPLIT)
        assert p.ddr_contended
        assert p.cpu_gbps < ACT_VNNI_INT8.gbps(4)
        assert p.host_gbps < PHASE0_PRIOR.slow_host_gbps(2, loaded=True)

    def test_split_mode_hides_the_cpu_term_under_the_gpu_partial_but_never_more(self):
        p = project_cpu_tier(
            host_bytes_per_rank=10_000_000,
            device_bytes_per_rank=0,
            cpu_bytes_per_rank=1_000,  # trivially small: cpu finishes long before the GPU
            num_cpu_layers=1, mode=CpuTierMode.SPLIT,
            handoff_us_per_layer=10.0, threads_per_rank=2, num_ranks=2,
        )
        assert p.overlapped_ms == pytest.approx(p.cpu_ms + p.handoff_ms, rel=1e-9)
        assert p.step_ms == pytest.approx(p.compute_ms + p.device_ms + p.host_ms, rel=1e-9)

    def test_split_mode_never_saves_more_than_the_gpu_work_it_hides_behind(self):
        p = project_cpu_tier(
            host_bytes_per_rank=1_000,  # trivial GPU partial
            device_bytes_per_rank=0,
            cpu_bytes_per_rank=100_000_000,  # huge CPU partial
            num_cpu_layers=1, mode=CpuTierMode.SPLIT,
            handoff_us_per_layer=40.0, threads_per_rank=2, num_ranks=2,
        )
        assert p.overlapped_ms == pytest.approx(p.host_ms, rel=1e-9)
        assert p.step_ms > p.compute_ms  # the CPU is now the critical path and it SHOWS

    def test_the_handoff_value_is_carried_not_laundered(self):
        p = project_cpu_tier(
            host_bytes_per_rank=0, device_bytes_per_rank=0, cpu_bytes_per_rank=1_000,
            num_cpu_layers=21, mode=CpuTierMode.BLOCK, handoff_us_per_layer=40.0,
            threads_per_rank=2, num_ranks=2,
        )
        assert p.handoff_us_used == 40.0
        assert "UNMEASURED" in p.describe()
        assert p.handoff_ms == pytest.approx(21 * 0.040, rel=1e-9)

    def test_the_projection_names_the_activation_error_it_priced(self):
        p = project_cpu_tier(
            host_bytes_per_rank=0, device_bytes_per_rank=0, cpu_bytes_per_rank=1_000,
            num_cpu_layers=1, mode=CpuTierMode.BLOCK, handoff_us_per_layer=10.0,
            threads_per_rank=2, num_ranks=2,
        )
        assert p.policy == "vnni_int8"
        assert p.rel_rms == pytest.approx(8.279e-03)
        assert "8.28e-03" in p.describe()

    def test_the_fp32_oracle_can_still_be_costed_but_does_not_fit(self):
        """Kept reachable on purpose: it is the correctness oracle, not a serving option."""
        with pytest.raises(CpuTierError, match="only 5 are free"):
            project_cpu_tier(
                host_bytes_per_rank=0, device_bytes_per_rank=0, cpu_bytes_per_rank=10_000_000,
                num_cpu_layers=1, mode=CpuTierMode.BLOCK, handoff_us_per_layer=10.0,
                threads_per_rank=3, num_ranks=2, policy=ACT_FP32,
            )

    def test_handoff_is_required_and_a_negative_one_is_refused(self):
        with pytest.raises(TypeError):
            project_cpu_tier(  # type: ignore[call-arg]
                host_bytes_per_rank=0, device_bytes_per_rank=0, cpu_bytes_per_rank=0,
                num_cpu_layers=0, mode=CpuTierMode.OFF,
            )
        with pytest.raises(CpuTierError):
            project_cpu_tier(
                host_bytes_per_rank=0, device_bytes_per_rank=0, cpu_bytes_per_rank=0,
                num_cpu_layers=0, mode=CpuTierMode.OFF, handoff_us_per_layer=-1.0,
            )

    def test_a_37_layer_block_at_the_operating_point_matches_the_hand_arithmetic(self):
        """Anchors the projection to `RESULTS_VNNI_2026-09-04.txt` section 6 so it cannot drift."""
        # 37 layers x 10 experts x 2,764,800 B node-wide at 53.50 GB/s (4 physical cores).
        expected_ms = 37 * TOP_K * B_EXPERT_CPU / (53.50 * 1e9) * 1e3
        got = cpu_layer_ms(37 * TOP_K * B_EXPERT_CPU, threads=4)
        assert got == pytest.approx(expected_ms, rel=1e-9)
        assert got == pytest.approx(19.12, abs=0.05)
        # ...and RESULTS_VNNI's own 2-core line, which is the per-RANK view of the same point.
        assert cpu_layer_ms(TOP_K * B_EXPERT_CPU, threads=2) == pytest.approx(0.6193, abs=0.005)
