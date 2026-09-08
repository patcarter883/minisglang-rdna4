"""`resolve_weight_plan` with the CPU-COMPUTE tier: the gate, the sweep, and the capacity win.

NO GPU. `plan.py` is defensively getattr-based, so a `SimpleNamespace` config is enough to drive the
whole resolver — which is the point: the capacity arithmetic and the four refusals are the parts
that can be silently wrong, and they are tested here rather than on a leased card.

The shape is the real checkpoint's: Qwen3.8-Flash-Next-NVFP4, 48 MoE layers, 512 experts, top-10,
hidden 2560, moe_intermediate 640.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

# MUST precede the minisgl imports; installs a torch stub only when real torch is unimportable.
from _offload_torch_stub import TORCH_IS_REAL  # noqa: F401,I001
from minisgl.weights.cpu_tier import ACT_VNNI_INT8, CPU_TIER_PRIOR  # noqa: E402
from minisgl.weights.placement import PlacementError  # noqa: E402
from minisgl.weights.plan import (  # noqa: E402
    cpu_tier_gate,
    cpu_tier_sweep,
    resolve_weight_plan,
)
from minisgl.weights.stacks import StackKind  # noqa: E402

GiB = 1 << 30


class _Nvfp4Quant:
    """The target checkpoint's expert format: NVFP4, E2M1 codes + group-16 scales."""

    method = "compressed-tensors"
    bits = 4
    group_size = 16
    sym = True
    weight_type = "float"
    ct_groups = ()
    is_fp8_w8a8 = False
    is_nvfp4 = True
    weight_is_e2m1 = True
    is_int4 = False
    is_gptq = False
    is_awq = False
    is_compressed_tensors = True


def _model_config(**kw):
    base = dict(
        num_layers=48,
        num_experts=512,
        num_experts_per_tok=10,
        hidden_size=2560,
        moe_intermediate_size=640,
        is_moe=True,
        is_cca_hybrid=False,
        first_k_dense_replace=0,
        num_nextn_predict_layers=0,
        mtp_num_hidden_layers=0,
        quant=_Nvfp4Quant(),
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _config(*, tp=2, enable_ep=False, mc=None, **kw):
    base = dict(
        model_config=mc if mc is not None else _model_config(),
        tp_info=SimpleNamespace(size=tp, rank=0),
        dp_info=SimpleNamespace(dp_size=1, dp_rank=0),
        enable_ep=enable_ep,
        dtype=SimpleNamespace(itemsize=2),
        model_path="/models/qwen38-flash-next-nvfp4",
        weight_offload_device_gb=0.0,
        weight_offload_gb=0.0,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _resolve(**kw):
    kw.setdefault("prefer_meta", False)
    kw.setdefault("config", _config())
    cfg = kw.pop("config")
    return resolve_weight_plan(cfg, **kw)


# ════════════════════════════════════════════════════════════════════════════════════════════════
class TestTheGate:
    """Four refusals, each a real failure mode."""

    def test_nvfp4_is_allowed_and_names_the_VNNI_wload(self):
        g = cpu_tier_gate(
            {"schemes": ["nvfp4"], "expert_parallel": False, "ep_size": 1},
            local_ranks=2, n_offloadable=48, repacked=True,
        )
        assert g.allowed
        assert g.wload == "vnni_nvfp4_e4m3_g16"
        assert g.layout_fraction == 0.9
        assert (g.threads_per_rank, g.total_threads) == (2, 4)

    def test_a_format_with_no_cpu_core_is_refused_by_name(self):
        g = cpu_tier_gate(
            {"schemes": ["gptq"], "expert_parallel": False, "ep_size": 1},
            local_ranks=1, n_offloadable=48,
        )
        assert not g.allowed
        assert "no CPU expert core reads gptq" in g.reason
        assert "WLoad policy" in g.reason  # points at the table row, not at a new kernel

    def test_a_mixed_checkpoint_is_refused_if_ANY_scheme_lacks_a_core(self):
        g = cpu_tier_gate(
            {"schemes": ["nvfp4", "fp8"], "expert_parallel": False, "ep_size": 1},
            local_ranks=1, n_offloadable=48,
        )
        assert not g.allowed and "fp8" in g.reason

    def test_expert_parallel_is_refused_and_the_reason_is_the_row_reorder(self):
        g = cpu_tier_gate(
            {"schemes": ["nvfp4"], "expert_parallel": True, "ep_size": 2},
            local_ranks=2, n_offloadable=48,
        )
        assert not g.allowed
        assert "WRONG TOKENS" in g.reason
        assert "--enable-ep" in g.reason

    def test_the_core_budget_refuses_three_threads_per_rank_at_tp2(self):
        g = cpu_tier_gate(
            {"schemes": ["nvfp4"], "expert_parallel": False, "ep_size": 1},
            local_ranks=2, n_offloadable=48, threads_per_rank=3,
        )
        assert not g.allowed and "only 5 are free" in g.reason

    def test_verbatim_placement_does_not_get_the_layout_shrink(self):
        g = cpu_tier_gate(
            {"schemes": ["nvfp4"], "expert_parallel": False, "ep_size": 1},
            local_ranks=2, n_offloadable=48, repacked=False,
        )
        assert g.allowed and g.layout_fraction == 1.0

    def test_no_layers_is_a_refusal_not_a_crash(self):
        g = cpu_tier_gate({"schemes": [], "expert_parallel": False}, local_ranks=1, n_offloadable=0)
        assert not g.allowed and "no offloadable" in g.reason


# ════════════════════════════════════════════════════════════════════════════════════════════════
class TestTheResolver:
    def test_the_default_is_a_two_tier_plan_and_the_lever_is_still_reported(self):
        """The tier is opt-in — there is no native `.so` yet — but the sweep runs anyway."""
        r = _resolve()
        assert r.plan.num_cpu_layers == 0
        assert r.cpu_gate is not None and r.cpu_gate.allowed
        assert r.cpu_sweep and len(r.cpu_sweep) >= 4
        assert "cpu-compute tier ALLOWED" in r.cpu_sweep_text

    def test_asking_for_cpu_layers_places_them_and_nothing_else_moves_to_pinned(self):
        r = _resolve(num_cpu_layers=21, cpu_repacked=True)
        assert r.plan.num_cpu_layers == 21
        kinds = [p.kind for p in r.plan.placements]
        assert kinds[-21:] == [StackKind.CPU] * 21
        assert StackKind.HOST in kinds  # the rest still stream
        assert r.plan.cpu_layer_indices == tuple(range(27, 48))

    def test_a_refused_request_RAISES_it_is_never_downgraded_to_zero(self):
        """A silent downgrade leaves every capacity number describing a residency that is not there."""
        with pytest.raises(PlacementError, match="expert parallel is active"):
            _resolve(config=_config(enable_ep=True), num_cpu_layers=21)

    def test_asking_for_more_layers_than_exist_raises(self):
        with pytest.raises(PlacementError, match="exceeds the 48 offloadable"):
            _resolve(num_cpu_layers=49)

    def test_the_HEADLINE_projection_becomes_the_three_tier_one(self):
        """`project_plan` sums HOST + DEVICE only. On a CPU plan that drops the biggest term.

        Left alone it reported 133 tok/s for an all-CPU plan — every byte had left both of the
        tiers it knows about — and the K4 gate, the A1.7 threshold, `summary_line` and the engine
        banner all read that number.
        """
        r = _resolve(num_cpu_layers=48, cpu_repacked=True)
        assert r.projection.tok_s == pytest.approx(r.cpu_projection.tok_s, rel=1e-9)
        assert r.projection.step_ms > 30.0  # NOT the 7.5 ms compute floor alone
        assert r.acceptance_threshold_tok_s == pytest.approx(r.cpu_projection.tok_s * 0.75, rel=1e-6)
        # ...and the banner one line up says the same thing the three-tier projection does.
        assert f"{r.cpu_projection.tok_s:.1f} tok/s projected" in r.summary_line()
        # The CPU term is folded into `host_ms` because `Projection` has no field for it; the
        # composite must still add up, or the banner's arithmetic stops reconciling.
        assert r.projection.step_ms == pytest.approx(
            r.projection.compute_ms + r.projection.device_ms + r.projection.host_ms, rel=1e-9
        )

    def test_the_two_tier_projection_is_untouched_when_no_layer_is_on_the_cpu(self):
        r = _resolve()
        assert r.cpu_projection is None
        assert r.projection.tok_s > 0

    def test_the_projection_and_the_capture_cost_are_both_carried(self):
        r = _resolve(num_cpu_layers=21, cpu_repacked=True)
        assert r.cpu_projection is not None
        assert r.cpu_projection.graph_segments == 22
        assert r.cpu_projection.calibrated_tok_s < r.cpu_projection.tok_s
        assert any("CUDAGraph capture cannot express" in w for w in r.warnings)
        assert any("UNMEASURED" in w for w in r.warnings)

    def test_the_boot_banner_names_the_capacity_win_and_the_accuracy_cost(self):
        r = _resolve(num_cpu_layers=21, cpu_repacked=True)
        text = "\n".join(r.render_lines())
        assert "cpu-tier capacity:" in text
        assert "of pinned arena RELIEVED" in text
        assert "rel_rms" in text

    def test_as_dict_is_a_superset_of_the_two_tier_shape(self):
        two = _resolve().as_dict()
        three = _resolve(num_cpu_layers=21, cpu_repacked=True).as_dict()
        assert set(two) == set(three)
        assert two["num_cpu_layers"] == 0 and two["cpu_projected_tok_s"] is None
        assert three["num_cpu_layers"] == 21
        assert three["cpu_tier_wload"] == "vnni_nvfp4_e4m3_g16"


# ════════════════════════════════════════════════════════════════════════════════════════════════
class TestTheCapacityWin:
    """The number that may matter more than the throughput. All arithmetic, all node-wide."""

    def _win(self, k):
        base = _resolve()
        new = _resolve(num_cpu_layers=k, cpu_repacked=True)
        return base, new

    def test_the_two_tier_plan_at_a_zero_device_tier_does_not_FIT_the_pinned_ceiling(self):
        """Which is the whole reason the capacity win matters as much as the throughput one."""
        base = _resolve()
        assert not base.feasible
        assert base.host_reservation_bytes_per_node > base.host_ceiling_bytes

    def test_the_cpu_tier_makes_that_same_plan_feasible_with_NO_device_tier_at_all(self):
        """The lever `required_device_bytes` could not offer: KV is not surrendered for it."""
        new = _resolve(num_cpu_layers=48, cpu_repacked=True)
        assert new.feasible
        assert new.device_budget_bytes == 0
        assert new.host_reservation_bytes_per_node == 0

    def test_moving_every_streamed_layer_to_the_cpu_EMPTIES_the_pinned_arena(self):
        base, new = self._win(48)
        assert new.plan.num_host_layers == 0
        assert new.host_bytes_per_node == 0
        assert new.host_reservation_bytes_per_node == 0
        # ...and the win is exactly what the two-tier plan was holding.
        assert new.pinned_bytes_saved_per_node == base.host_bytes_per_node

    def test_the_win_in_gib_is_the_whole_two_tier_arena(self):
        base, new = self._win(48)
        win_gib = new.pinned_bytes_saved_per_node / GiB
        assert win_gib == pytest.approx(base.host_bytes_per_node / GiB, rel=1e-9)
        assert win_gib > 40.0  # target shape: tens of GiB, not a rounding error

    def test_the_repack_removes_a_further_10_percent_of_HOST_RAM_outright(self):
        """Pinned-vs-pageable is a ceiling question; this is a bytes question, and they differ."""
        base, new = self._win(48)
        assert new.total_host_bytes_per_node < base.host_bytes_per_node
        assert new.total_host_bytes_per_node / base.host_bytes_per_node == pytest.approx(
            0.9, rel=1e-3
        )

    def test_a_verbatim_bake_gets_the_pinned_win_but_NOT_the_byte_win(self):
        base = _resolve()
        new = _resolve(num_cpu_layers=48, cpu_repacked=False)
        assert new.pinned_bytes_saved_per_node == base.host_bytes_per_node
        assert new.total_host_bytes_per_node == base.host_bytes_per_node

    def test_device_bytes_are_UNCHANGED_the_win_is_not_vram(self):
        base, new = self._win(21)
        assert new.device_bytes == base.device_bytes

    def test_the_sweep_prices_capacity_and_throughput_side_by_side(self):
        r = _resolve()
        rows = {row["cpu_layers"]: row for row in r.cpu_sweep}
        assert rows[0]["pinned_saved_node_bytes"] == 0
        biggest = max(rows)
        assert rows[biggest]["pinned_node_bytes"] == 0
        assert rows[biggest]["pinned_saved_node_bytes"] > 0
        # The RESERVATION shrinks by MORE than the payload does: pinning is charged in whole 2 GiB
        # chunks, so the last partial chunk of every rank disappears too.
        assert rows[0]["pinned_node_bytes"] > rows[biggest]["pinned_saved_node_bytes"]
        # throughput improves monotonically in K on this box, because a CPU layer at 53.5 GB/s is
        # always cheaper than the same layer streamed at 12.36
        toks = [rows[k]["tok_s"] for k in sorted(rows)]
        assert toks == sorted(toks)

    def test_the_sweep_uses_the_PESSIMISTIC_end_of_the_unmeasured_handoff(self):
        rows = cpu_tier_sweep(
            _resolve().layers,
            device_budget_bytes=0,
            gate=cpu_tier_gate(
                {"schemes": ["nvfp4"], "expert_parallel": False, "ep_size": 1},
                local_ranks=2, n_offloadable=48, repacked=True,
            ),
            local_ranks=2,
        )
        optimistic = cpu_tier_sweep(
            _resolve().layers,
            device_budget_bytes=0,
            gate=cpu_tier_gate(
                {"schemes": ["nvfp4"], "expert_parallel": False, "ep_size": 1},
                local_ranks=2, n_offloadable=48, repacked=True,
            ),
            local_ranks=2,
            handoff_us=CPU_TIER_PRIOR.handoff_us_bracket[0],
        )
        assert rows[-1]["step_ms"] > optimistic[-1]["step_ms"]

    def test_the_sweep_row_carries_the_activation_error_of_the_policy_it_priced(self):
        r = _resolve()
        for row in r.cpu_sweep:
            if row["cpu_layers"]:
                assert row["rel_rms"] == pytest.approx(ACT_VNNI_INT8.rel_rms)
