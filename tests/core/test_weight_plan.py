"""Layer-granular placement resolution -- `minisgl.weights.plan`. No GPU required.

This suite guards the arithmetic that decides whether the serve boots. Three groups:

  * STRUCTURE -- which layers own a MoELayer, and how TP/EP shards them. A factor-of-tp slip here
    turns an infeasible plan feasible and the failure surfaces as an unrecoverable page fault 40 GiB
    into the load.
  * CAPACITY -- the inequality against P3b's measured 62 GiB two-rank pinned ceiling, and the
    minimum device tier that satisfies it.
  * PROJECTION -- locked against the published Phase 0 3.4 table, so a change to the model is a
    failing test rather than a quietly different number on a banner.

GPU-REQUIRING TESTS: none in this file. The one thing that genuinely needs a device -- that the
resolved plan survives full-forward HIP graph capture at the served TP -- is NOT covered here and is
called out in the module's not-done list. Do not read a green run here as discharging that.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

# MUST precede the minisgl imports: installs a torch stub only when the real torch is unimportable,
# so this arithmetic is testable on a host with no working torch. Inert in the container. See
# `_offload_torch_stub`'s docstring.
from _offload_torch_stub import TORCH_IS_REAL  # noqa: I001
from minisgl.weights.host_capacity import P3B_PINNED_CEILING_BYTES
from minisgl.weights.placement import (
    LayerWeights,
    PlacementError,
    plan_layer_granular,
    project_plan,
    sweep_device_fraction,
)
from minisgl.weights.plan import (
    arena_reservation_bytes,
    build_planned_layers,
    host_arena_ceiling_bytes,
    local_arena_count,
    moe_layer_indices,
    moe_layer_shapes,
    plan_arena_reservation_bytes,
    required_device_bytes,
    resolve_expert_parallel,
    resolve_weight_plan,
)
from minisgl.weights.prior import GB, PHASE0_PRIOR, GiB, project_step

# =================================================================================================
# Fixtures: config stubs. `plan.py` is defensively getattr-based precisely so these work.
# =================================================================================================


class FakeQuant:
    def __init__(self, **kw):
        self.method = kw.pop("method", "compressed-tensors")
        self.bits = kw.pop("bits", 4)
        self.group_size = kw.pop("group_size", 32)
        self.sym = kw.pop("sym", True)
        self.weight_type = kw.pop("weight_type", "int")
        self.ct_groups = kw.pop("ct_groups", ())
        self.is_fp8_w8a8 = kw.pop("is_fp8_w8a8", False)
        self.is_nvfp4 = kw.pop("is_nvfp4", False)
        self.weight_is_e2m1 = kw.pop("weight_is_e2m1", False)
        self.is_int4 = kw.pop("is_int4", True)
        self.is_gptq = kw.pop("is_gptq", False)
        self.is_awq = kw.pop("is_awq", False)
        self.is_compressed_tensors = kw.pop("is_compressed_tensors", True)
        assert not kw, kw


def make_model_config(**kw):
    base = {
        "num_layers": 48,
        "num_experts": 512,
        "num_experts_per_tok": 10,
        "hidden_size": 2048,
        "moe_intermediate_size": 768,
        "is_moe": True,
        "is_cca_hybrid": False,
        "first_k_dense_replace": 0,
        "num_nextn_predict_layers": 0,
        "mtp_num_hidden_layers": 0,
        "quant": FakeQuant(),
    }
    base.update(kw)
    return SimpleNamespace(**base)


def make_config(model_config=None, *, tp=2, rank=0, dp=1, enable_ep=False, **kw):
    base = {
        "model_config": model_config if model_config is not None else make_model_config(),
        "tp_info": SimpleNamespace(size=tp, rank=rank),
        "dp_info": SimpleNamespace(dp_size=dp, dp_rank=0),
        "enable_ep": enable_ep,
        "dtype": SimpleNamespace(itemsize=2),
        "model_path": "/models/target-shaped",
        "weight_offload_device_gb": 0.0,
        "weight_offload_gb": 0.0,
    }
    base.update(kw)
    return SimpleNamespace(**base)


def resolve(config=None, **kw):
    kw.setdefault("prefer_meta", False)
    return resolve_weight_plan(config if config is not None else make_config(), **kw)


# =================================================================================================
# STRUCTURE -- which layers, and how they shard
# =================================================================================================


def test_dense_model_has_no_offloadable_layers():
    mc = make_model_config(is_moe=False, num_experts=0)
    assert moe_layer_indices(mc) == ()
    r = resolve(make_config(mc))
    assert not r.enabled and r.feasible
    assert "no offloadable expert stacks" in r.reason


def test_first_k_dense_replace_skips_the_leading_dense_layers():
    """`glm4_moe_lite.py:339` / `laguna.py:281`: layer_id < first_k_dense_replace is a dense MLP."""
    mc = make_model_config(num_layers=10, first_k_dense_replace=3)
    assert moe_layer_indices(mc) == (3, 4, 5, 6, 7, 8, 9)


def test_zaya_puts_moe_on_odd_layers_only():
    """ZAYA interleaves: even = CCA attention, odd = EDA/MOD MoE (`zaya.py:718-722`)."""
    mc = make_model_config(num_layers=8, is_cca_hybrid=True)
    assert moe_layer_indices(mc) == (1, 3, 5, 7)


def test_gemma4_style_every_layer_is_moe():
    mc = make_model_config(num_layers=6, first_k_dense_replace=0)
    assert moe_layer_indices(mc) == (0, 1, 2, 3, 4, 5)


def test_mtp_layers_are_enumerated_only_on_request_and_never_offloaded():
    """The MTP draft head is re-read on every draft step; streaming it would defeat spec decode."""
    mc = make_model_config(num_layers=4, mtp_num_hidden_layers=1)
    assert moe_layer_indices(mc) == (0, 1, 2, 3)
    assert moe_layer_indices(mc, include_mtp=True) == (0, 1, 2, 3, 4)
    shapes = moe_layer_shapes(make_config(mc), layer_indices=(0, 1, 2, 3, 4))
    mtp = [s for s in shapes if s.label == "moe_mtp"]
    assert len(mtp) == 1 and not mtp[0].offloadable
    layers, diag = build_planned_layers(
        make_config(mc), layer_indices=(0, 1, 2, 3, 4), prefer_meta=False
    )
    assert len(layers) == 4
    assert any("mtp" in s for s in diag["skipped"])


def test_layer_paths_are_structural_not_a_construction_counter():
    """A counter would renumber every layer after the MTP head builds its own MoELayer, silently
    placing the wrong layers on the wrong stack. `plan_layer_granular` refuses duplicates."""
    shapes = moe_layer_shapes(make_config(make_model_config(num_layers=3, mtp_num_hidden_layers=1)),
                              layer_indices=(0, 1, 2, 3))
    assert [s.path for s in shapes] == [
        "model.layers.0.mlp.experts",
        "model.layers.1.mlp.experts",
        "model.layers.2.mlp.experts",
        "mtp.layers.0.mlp.experts",
    ]


@pytest.mark.parametrize(
    "tp,dp,enable_ep,expected",
    [
        (2, 1, False, (False, 1)),   # plain TP
        (1, 1, False, (False, 1)),
        (2, 1, True, (True, 2)),     # EP-over-TP (engine.py:172)
        (1, 1, True, (False, 1)),    # EP requested but inert
        (2, 4, True, (True, 4)),     # DP+EP: experts shard across DP replicas
    ],
)
def test_expert_parallel_resolution_mirrors_the_engine(tp, dp, enable_ep, expected):
    assert resolve_expert_parallel(make_config(tp=tp, dp=dp, enable_ep=enable_ep)) == expected


def test_plain_tp_splits_the_intermediate_and_keeps_every_expert():
    s = moe_layer_shapes(make_config(tp=2))[0]
    assert s.num_local_experts == 512
    assert s.intermediate_size_per_partition == 768 // 2
    assert s.top_k_local == 10


def test_ep_over_tp_shards_experts_and_keeps_the_full_intermediate():
    """EP-over-TP keeps whole experts, so the intermediate is NOT split -- the opposite of TP."""
    s = moe_layer_shapes(make_config(tp=2, enable_ep=True))[0]
    assert s.num_local_experts == 512 // 2
    assert s.intermediate_size_per_partition == 768
    # ceil(10/2): a step waits for the SLOWEST rank, so round up, never average down.
    assert s.top_k_local == 5


def test_ep_top_k_local_rounds_up():
    mc = make_model_config(num_experts_per_tok=7)
    assert moe_layer_shapes(make_config(mc, tp=2, enable_ep=True))[0].top_k_local == 4


def test_indivisible_intermediate_raises_like_div_even():
    mc = make_model_config(moe_intermediate_size=769)
    with pytest.raises(PlacementError, match="divisible"):
        moe_layer_shapes(make_config(mc, tp=2))


def test_indivisible_expert_count_under_ep_raises():
    mc = make_model_config(num_experts=513)
    with pytest.raises(PlacementError, match="divisible"):
        moe_layer_shapes(make_config(mc, tp=2, enable_ep=True))


@pytest.mark.parametrize("tp,dp,expected", [(1, 1, 1), (2, 1, 2), (2, 4, 8)])
def test_local_arena_count_charges_every_rank_on_the_node(tp, dp, expected):
    """Host RAM is a node resource. A per-rank check passes twice and the box still dies."""
    assert local_arena_count(make_config(tp=tp, dp=dp)) == expected


def test_the_resolver_emits_the_placement_modules_own_layer_type():
    """There is no parallel layer type. The config path and the post-load granule path must feed
    ONE planner, or they can quietly disagree about which layer went where -- and the two ranks that
    took different paths would then serve different weights with no error anywhere."""
    layers, _ = build_planned_layers(make_config(), prefer_meta=False)
    assert layers and all(isinstance(lw, LayerWeights) for lw in layers)
    plan = plan_layer_granular(layers, device_budget_bytes=layers[0].resident_bytes)
    assert plan.num_device_layers == 1


def test_estimated_layers_carry_an_estimate_marked_fingerprint():
    """`fingerprint` says WHICH weights these bytes describe. A config estimate and a post-load
    measurement must not be mistakable for one another."""
    layers, _ = build_planned_layers(make_config(), prefer_meta=False)
    assert all(lw.fingerprint.startswith("est:") for lw in layers)
    assert len({lw.fingerprint for lw in layers}) == 1  # uniform stack -> one shape


def test_layer_weights_rejects_a_sharding_slip():
    """granule x experts above the resident size means the two were computed against different
    shardings -- the classic global-vs-EP-local slip, caught at construction."""
    with pytest.raises(PlacementError, match="wrong expert count|smaller than"):
        LayerWeights(path="p", num_experts=512, top_k=10, granule_bytes=1024,
                     resident_bytes=1024)


# =================================================================================================
# CAPACITY -- the inequality the whole feature turns on
# =================================================================================================


def test_ceiling_is_the_p3b_table_with_the_headroom_derate():
    for ranks, raw in P3B_PINNED_CEILING_BYTES.items():
        assert host_arena_ceiling_bytes(ranks) == int(
            raw * PHASE0_PRIOR.host_arena_headroom_fraction
        )


def test_ceiling_does_not_grow_with_more_ranks():
    """62 GiB is a node MemAvailable floor, not a per-rank allowance. Adding ranks adds no RAM."""
    assert host_arena_ceiling_bytes(4) == host_arena_ceiling_bytes(2)
    assert host_arena_ceiling_bytes(8) == host_arena_ceiling_bytes(2)


def test_prior_and_host_capacity_agree_on_the_measured_ceiling():
    """Two modules carry P3b's 62 GiB. If they ever diverge, one of them is deciding boots against
    a number nobody measured."""
    assert PHASE0_PRIOR.host_arena_ceiling_bytes == P3B_PINNED_CEILING_BYTES[2]


def test_ceiling_rejects_zero_ranks():
    with pytest.raises(ValueError, match="local_ranks"):
        host_arena_ceiling_bytes(0)


def _uniform(n, resident, granule, experts=512, top_k=10, max_row=None):
    """`max_row` is the PACKING bound (`LayerWeights.max_row_bytes`), and it is explicit here on
    purpose. Left unset it is 0, which means "not derived" and makes every consumer fall back to the
    whole layer — a sound but very loose bound that puts one layer per arena chunk. Tests that are
    about the greedy minimum, not about packing, pass a realistic component-sized row so the chunk
    arithmetic stays near-perfect; the packing tests below pass a deliberately large one."""
    return [
        LayerWeights(
            path=f"model.layers.{i}.mlp.experts", num_experts=experts, top_k=top_k,
            granule_bytes=granule, resident_bytes=resident,
            max_row_bytes=(resident // 16 if max_row is None else max_row),
        )
        for i in range(n)
    ]


def test_required_device_bytes_is_zero_when_all_host_already_fits():
    layers = _uniform(4, 1 * GiB, 2 * (1 << 20))
    assert required_device_bytes(layers, ceiling_total_bytes=100 * GiB, local_ranks=2) == 0


def test_required_device_bytes_is_the_minimum_that_makes_it_fit():
    """10 layers x 1 GiB/rank x 2 ranks against a 13 GiB ceiling.

    The answer is 5 layers on device, not the 4 a payload-only inequality gives, and the extra layer
    is the PACKING charge: the arena reserves whole 2 GiB chunks and a row may never straddle one,
    so with rows of up to `m` bytes only `chunk - m` per chunk is guaranteed placeable
    (`chunk_plan.headroom_chunks`). 6 host layers = 6 GiB needs ceil(6144 / 1984 MiB) = 4 chunks =
    8 GiB/rank = 16 GiB node, over the ceiling; 5 layers needs 3 chunks = 12 GiB node and fits.
    The old 4 GiB answer was a tier the operator could grant and still abort mid-pin."""
    layers = _uniform(10, 1 * GiB, 2 * (1 << 20))
    need = required_device_bytes(layers, ceiling_total_bytes=13 * GiB, local_ranks=2)
    assert need == 5 * GiB
    # And it is minimal: one layer less of device tier must NOT clear the ceiling.
    from minisgl.weights.plan import arena_reservation_bytes as _res
    row = layers[0].row_bound
    assert _res(10 * GiB - need, 2 * GiB, row) * 2 <= 13 * GiB
    assert _res(10 * GiB - (need - 1 * GiB), 2 * GiB, row) * 2 > 13 * GiB


def test_required_device_bytes_saturates_at_the_whole_stack():
    layers = _uniform(3, 1 * GiB, 2 * (1 << 20))
    assert required_device_bytes(layers, ceiling_total_bytes=0, local_ranks=1) == 3 * GiB


def test_required_device_bytes_walks_the_planner_order():
    """It must return the budget that produces the plan the operator would actually get, so it
    honours the same `(-priority, index)` order `plan_layer_granular` uses."""
    layers = _uniform(4, 1 * GiB, 2 * (1 << 20))
    layers[3] = LayerWeights(
        path="model.layers.3.mlp.experts", num_experts=512, top_k=10,
        granule_bytes=2 * (1 << 20), resident_bytes=1 * GiB, priority=5,
    )
    need = required_device_bytes(layers, ceiling_total_bytes=3 * GiB, local_ranks=1)
    plan = plan_layer_granular(layers, device_budget_bytes=need)
    assert plan.host_resident_bytes <= 3 * GiB
    assert plan.kind_of("model.layers.3.mlp.experts").name == "DEVICE"


def test_target_shaped_checkpoint_does_not_fit_all_host():
    """Phase 0 3.4's headline: the target checkpoint's all-host arena exceeds the pinned ceiling, so
    the device tier is a CAPACITY PREREQUISITE, not a speedup. This fixture is target-SHAPED
    (48 layers, E=512, top_k=10, CT-int4 g32, TP=2), not the checkpoint itself."""
    r = resolve()
    assert not r.feasible
    assert r.required_device_bytes > 0
    assert r.shortfall_bytes == r.required_device_bytes
    assert "INFEASIBLE" in r.reason
    with pytest.raises(PlacementError, match="cannot fit this model"):
        r.raise_if_infeasible()
    msg = r.failure_message()
    assert "device tier required" in msg and "200k KV tokens" in msg


def test_target_shaped_arena_charges_the_zeros_post_load_synthesises():
    """The arena holds what exists AFTER `post_load()`, and the target scheme is CT-int4 SYMMETRIC.

    `_GroupedCompressedTensorsExperts.post_load`'s `zp is None` arm allocates a REAL
    `torch.empty((E, G, N/pf), int32)` of 0x88 that `__init__` never declared, and
    `moe_interpose._plan_items` copies it into the arena like every other tensor. Sizing it as
    absent -- which BOTH byte models did, the meta one included, because meta builds `__init__`
    shapes and can never run `post_load` -- under-reserves the arena by ~3.1%: on this 48-layer
    fixture 864 MiB/rank, 13x the accounting gate's 64 MiB tolerance. The arena then runs dry
    mid-bake, `ArenaMemPool._alloc` falls back to `hipMalloc`, and weights budgeted as host-resident
    land in VRAM; and before that, `required_device_bytes` tells the operator to grant a tier that
    is too small.

    Expected values are explicit products from the fixture, not re-derived from the code under test:
    E=512, H=2048, I=768, tp=2 with no EP -> I_part=384, so w13 is (512, 768, 2048) and w2 is
    (512, 2048, 384); pf=8, g=32.
    """
    r = resolve()
    e, h, i_part = 512, 2048, 384
    w13_n, w13_k = 2 * i_part, h
    w2_n, w2_k = h, i_part
    checkpoint = (
        e * w13_n * (w13_k // 8) * 4 + e * w13_n * (w13_k // 32) * 2
        + e * w2_n * (w2_k // 8) * 4 + e * w2_n * (w2_k // 32) * 2
    )
    zeros = e * (w13_k // 32) * (w13_n // 8) * 4 + e * (w2_k // 32) * (w2_n // 8) * 4
    assert (checkpoint, zeros) == (679_477_248, 18_874_368)

    assert r.layers[0].resident_bytes == checkpoint + zeros
    # ... and the zeros are NOT in the granule: E bitwise-identical rows, so the walker calls them
    # replicated and a routed expert never pays for them. Charging them there would inflate the
    # projected step time instead of the arena.
    assert r.layers[0].granule_bytes == checkpoint // e
    assert r.layers[0].granule_bytes * e < r.layers[0].resident_bytes

    # The capacity consequence, which is the whole point of getting this right.
    assert r.host_bytes_per_node == 48 * (checkpoint + zeros) * 2
    assert not r.feasible


def test_granting_the_required_device_tier_makes_it_feasible():
    """The number the failure message tells the operator to grant must actually work. If it did not,
    the message would send them round a loop."""
    infeasible = resolve()
    ok = resolve(device_budget_bytes=infeasible.required_device_bytes)
    ok.raise_if_infeasible()
    assert ok.feasible and ok.enabled
    assert ok.host_bytes_per_node <= ok.host_ceiling_bytes
    # One layer less must NOT fit, or the "required" number is not minimal.
    layer_bytes = infeasible.layers[0].resident_bytes
    tight = resolve(device_budget_bytes=max(0, infeasible.required_device_bytes - layer_bytes))
    assert not tight.feasible


def test_whole_stack_fitting_the_device_budget_is_a_no_op_plan():
    """The §6.2 contract: the plan is DERIVED, so when everything fits the path is exercised and
    costs nothing -- it is never an is-my-feature-enabled flag."""
    r = resolve(device_budget_bytes=1000 * GB)
    assert r.feasible and not r.enabled
    assert r.plan.is_empty
    assert r.host_bytes_per_rank == 0
    assert r.device_bytes == r.plan.total_resident_bytes
    assert "no-op" in r.reason
    assert r.summary_line().startswith("weight-offload: OFF")
    # Leftover budget in the no-op plan is VRAM the operator offered and the model did not need,
    # not a layer-quantisation residual. Warning about it would train people to ignore that line.
    assert not any("device budget is unused" in w for w in r.warnings)


def test_weight_offload_gb_sets_the_host_budget_and_never_enables_anything():
    """The flag sets the per-rank pinned-host budget. Tightening it can only make a plan harder to
    satisfy, never easier -- so the required device tier must be non-decreasing as it shrinks."""
    loose = resolve(device_budget_bytes=8 * GiB)
    tight = resolve(make_config(weight_offload_gb=4.0), device_budget_bytes=8 * GiB)
    assert tight.host_ceiling_bytes < loose.host_ceiling_bytes
    assert tight.required_device_bytes >= loose.required_device_bytes
    # And it never turns the feature ON: a plan that fits VRAM stays empty whatever the flag says.
    assert not resolve(make_config(weight_offload_gb=999.0),
                       device_budget_bytes=1000 * GiB).enabled


def test_weight_offload_gb_can_RAISE_the_budget_above_the_baked_p3b_table():
    """REGRESSION. `host_capacity.py` says of `P3B_PINNED_CEILING_BYTES`: "Advisory only -- never a
    gate, because it is a property of that box on that day, while MemAvailable is the live truth."
    This resolver turns it into THE boot gate, so it MUST be overridable upward. When the knob could
    only `min()`, a box with more RAM than the one P3b ran on aborted with "add host RAM" while tens
    of GiB sat free and the operator had no way to say otherwise. The live `check_capacity()` against
    real MemAvailable is still what stands between this number and a pinned page."""
    raised = resolve(make_config(weight_offload_gb=80.0))
    assert raised.host_ceiling_bytes > host_arena_ceiling_bytes(2)
    assert raised.feasible  # the all-host arena now fits a 2 x 80 GiB node budget
    # ...but never silently: an operator assertion above anything demonstrated has to say so.
    assert any("demonstrated" in w for w in raised.warnings)


def test_device_budget_falls_back_to_the_config_field_in_GiB_not_decimal_GB():
    """REGRESSION. Every other memory figure on this boot path is binary -- P3b's 34/62 GiB table,
    the 12 GiB floor, the 2 GiB chunk, `engine.py`'s own `MINISGL_GRAPH_RESERVE_MARGIN_GB`
    (`int(float(margin) * (1 << 30))`) -- and `_gb()` RENDERS in GiB. Parsing the operator's number
    as decimal 1e9 under-granted the tier by 7.4% and printed it back as if it had been honoured."""
    cfg = make_config(weight_offload_device_gb=6.0)
    assert resolve(cfg).device_budget_bytes == 6 * GiB
    assert resolve(cfg).device_budget_bytes != 6 * GB


# -------------------------------------------------------------------------------------------------
# REGRESSION: the plan must charge what the ARENA PINS, not what the weights weigh.
# -------------------------------------------------------------------------------------------------


def test_capacity_is_charged_in_whole_arena_chunks_not_payload_bytes():
    """`PinnedWeightArena.attach` hipHostMallocs and first-touches every chunk at FULL chunk_bytes,
    so the pinned footprint is the payload rounded UP -- up to one chunk per rank of host RAM the
    old inequality ignored. The unit assertions below pin `arena_reservation_bytes`, which remains
    the reservation on any plan whose rows cannot be enumerated."""
    chunk = 2 * GiB
    assert arena_reservation_bytes(0, chunk) == 0
    assert arena_reservation_bytes(1, chunk) == chunk
    assert arena_reservation_bytes(2 * GiB, chunk) == 2 * GiB
    assert arena_reservation_bytes(2 * GiB + 1, chunk) == 4 * GiB
    # And the resolution reports it, per rank and per node, so a boot log can explain a mid-pin abort.
    r = resolve(device_budget_bytes=8 * GiB)
    assert r.host_reservation_bytes_per_rank == plan_arena_reservation_bytes(
        r.plan, r.arena_chunk_bytes
    )
    assert r.host_reservation_bytes_per_rank >= r.host_bytes_per_rank
    assert r.host_reservation_bytes_per_node == r.host_reservation_bytes_per_rank * r.local_ranks
    # ...and on this fixture the rows ARE enumerated, so the charge is the EXACT packing and is
    # strictly cheaper than the worst-case bound. That gap is the M1-B headline: it is device tier.
    assert r.plan.host_rows_known
    assert r.host_reservation_bytes_per_rank < arena_reservation_bytes(
        r.host_bytes_per_rank, r.arena_chunk_bytes, r.plan.max_host_row_bytes
    )


def test_the_plan_charge_equals_what_chunk_plan_will_actually_reserve():
    """Cross-check against the real allocator planner rather than re-deriving the rounding here: the
    two must not be able to drift, because a drift is invisible until a boot dies mid-pin.

    `plan_regions` is fed the SAME `RegionRequest` list `StageARuntime.attach_host_arena` hands to
    `reserve()`, so this compares the resolver against literally the code that reserves -- a
    stronger statement than the old form, which compared it against the anonymous-headroom bound.
    """
    from minisgl.weights.chunk_plan import plan_regions

    r = resolve(device_budget_bytes=8 * GiB)
    rows = r.plan.host_row_requests()
    assert rows, "the fixture must enumerate its rows, or this proves nothing"
    real = plan_regions(rows, r.arena_chunk_bytes)
    assert real.reserved_bytes == r.host_reservation_bytes_per_rank
    # Every reserved region is a FORECAST of an anonymous MemPool carve, never a named one.
    assert all(p.forecast for p in real.placements)
    # And the plan digest is no longer blind: it hashes every row's name/chunk/offset/size, so two
    # ranks that laid the arena out differently cannot print the same string.
    assert real.digest() != plan_regions(rows[:-1], r.arena_chunk_bytes).digest()


def test_required_device_tier_produces_a_plan_whose_PINNED_arena_fits():
    """REGRESSION, and this is the bug that mattered. On the target-shaped fixture the old
    `required_device_bytes` returned 2.53 GiB/rank; granting exactly that produced a plan the
    resolver called FEASIBLE at 27.84 GiB/rank while the arena went on to pin 28.00 GiB/rank =
    56.00 GiB node against a 55.80 GiB usable budget. The operator followed the resolver's own
    "raise the device tier to >= X" advice and still got an abort mid-pin, with the box in swap --
    exactly the failure mode `host_capacity.py` exists to prevent, produced by the planner."""
    from minisgl.weights.chunk_plan import plan_regions

    infeasible = resolve()
    assert not infeasible.feasible
    ok = resolve(device_budget_bytes=infeasible.required_device_bytes)
    ok.raise_if_infeasible()
    # The pinned figure, not the payload figure, is what has to clear the budget.
    assert ok.host_reservation_bytes_per_node <= ok.host_ceiling_bytes
    real = plan_regions(ok.plan.host_row_requests(), ok.arena_chunk_bytes)
    assert real.reserved_bytes * ok.local_ranks <= ok.host_ceiling_bytes
    # Still minimal: one layer less of device tier must NOT fit.
    layer_bytes = infeasible.layers[0].resident_bytes
    tight = resolve(
        device_budget_bytes=max(0, infeasible.required_device_bytes - layer_bytes)
    )
    assert not tight.feasible


def test_required_device_bytes_charges_the_rounding_too():
    """8 layers x 1.5 GiB = 12 GiB on one rank against a 12 GiB budget.

    The payload fits the budget EXACTLY, and the arena still does not: it pins whole 2 GiB chunks,
    and a row may never straddle one, so `headroom_chunks(12 GiB, 2 GiB, 64 MiB)` is 7 chunks =
    14 GiB. A payload-only inequality returns 0 here and lies twice over -- once for the chunk
    rounding, once for the abandoned next-fit tails. Whatever tier this returns must produce a
    RESERVATION inside the budget, which is the assertion that actually matters."""
    layers = _uniform(8, 3 * GiB // 2, 2 * (1 << 20), max_row=64 * (1 << 20))
    row = layers[0].max_row_bytes
    need = required_device_bytes(layers, 12 * GiB, 1, 2 * GiB)
    assert need > 0
    assert arena_reservation_bytes(12 * GiB - need, 2 * GiB, row) <= 12 * GiB
    # ... and it is minimal: one layer less of tier does not fit.
    assert arena_reservation_bytes(
        12 * GiB - (need - layers[0].resident_bytes), 2 * GiB, row
    ) > 12 * GiB


def test_the_failure_message_names_the_pinned_figure_and_the_chunk_size():
    """An operator diagnosing a capacity abort has to be able to see WHY the number is bigger than
    the weights -- otherwise the message and the arena's own error look like they disagree."""
    msg = resolve().failure_message()
    assert "pinned host arena" in msg and "chunks" in msg


# -------------------------------------------------------------------------------------------------
# REGRESSION: the bandwidth gate counts CARDS on the node, not TP ranks.
# -------------------------------------------------------------------------------------------------


def test_projection_bandwidth_is_gated_by_every_card_on_the_node_not_just_tp():
    """REGRESSION. `prior.slow_host_gbps(n)` returns the slowest of the first n cards because P4
    measured the links as independent (efficiency 0.999) and the step waits for the last rank. Under
    dp=2/tp=1 there are two ranks on two cards -- `local_arena_count` already charges both arenas --
    but the projection passed `tp_size=1`, so it quoted card 0's 28.93 GB/s while rank 1 runs on the
    Gen4 card at 14.48. That is a 1.90x over-projection of the mechanism ceiling, and BOTH gates read
    it: K4 could pass a plan that should be killed, and A1.7 (0.75x this) would set an acceptance
    threshold no serve can reach -- which reads as the mechanism failing, not the arithmetic."""
    dp = resolve(make_config(tp=1, dp=2))
    assert dp.local_ranks == 2
    assert dp.projection.host_gbps == pytest.approx(PHASE0_PRIOR.host_read_gbps[1])
    assert dp.projection.host_gbps != pytest.approx(PHASE0_PRIOR.host_read_gbps[0])
    # Single rank, single card: the fast card is correct and must not regress to the slow one.
    solo = resolve(make_config(tp=1, dp=1))
    assert solo.local_ranks == 1
    assert solo.projection.host_gbps == pytest.approx(PHASE0_PRIOR.host_read_gbps[0])
    # The all-host projection and the f-sweep read the same gate, or the banner contradicts itself.
    assert dp.all_host_projection.host_gbps == dp.projection.host_gbps
    assert f"{PHASE0_PRIOR.host_read_gbps[1]}" in dp.sweep_text or "14.48" in dp.sweep_text


def test_every_sharding_mode_projects_against_the_slow_card_once_two_cards_are_in_use():
    for tp, dp, ep in ((2, 1, False), (2, 1, True), (1, 2, False)):
        r = resolve(make_config(tp=tp, dp=dp, enable_ep=ep))
        assert r.local_ranks == 2
        assert r.projection.host_gbps == pytest.approx(
            PHASE0_PRIOR.slow_host_gbps(r.local_ranks)
        ), f"tp={tp} dp={dp} ep={ep}"


def test_near_ceiling_plans_warn():
    """P3b reached 62 GiB on an IDLE box while swapping 114,813 pages. A plan that lands just under
    it is not a comfortable fit and must say so."""
    r = resolve()
    r2 = resolve(device_budget_bytes=r.required_device_bytes)
    assert any("within 5%" in w or "swapping" in w for w in r2.warnings) or (
        r2.host_bytes_per_node <= r2.host_ceiling_bytes * 0.95
    )


# =================================================================================================
# PROJECTION -- locked to the published Phase 0 3.4 table
# =================================================================================================

# Phase 0 3.4 / 7's inputs, verbatim: 34.4 GB expert bytes per rank, 0.665 GB active per token per
# rank, card-1-gated 14.48 GB/s host, 692 GB/s device, 7.5 ms compute, TP=2.
_REPORT_ACTIVE_PER_RANK = 0.665 * GB
_REPORT_ROWS = {  # f -> (step_ms, tok_s) as published
    0.00: (53.4, 18.7),
    0.10: (48.9, 20.4),
    0.20: (44.4, 22.5),
    0.25: (42.2, 23.7),
    0.30: (39.9, 25.0),
}


@pytest.mark.parametrize("f,expected", sorted(_REPORT_ROWS.items()))
def test_projection_reproduces_the_phase0_table(f, expected):
    """The step model is `compute + host_bytes/host_BW + device_bytes/device_BW`, additive because
    P2' measured the miss-cost curve LINEAR (cliff_index 0.090/0.094 vs a pure-linear 0.100). If
    this drifts, either the model changed or the prior did -- both must be deliberate."""
    step_ms, tok_s = expected
    got = project_step(
        host_bytes_per_rank=int(_REPORT_ACTIVE_PER_RANK * (1 - f)),
        device_bytes_per_rank=int(_REPORT_ACTIVE_PER_RANK * f),
        prior=PHASE0_PRIOR,
        num_ranks=2,
    )
    assert got.step_ms == pytest.approx(step_ms, abs=0.05)
    assert got.tok_s == pytest.approx(tok_s, abs=0.05)


def test_all_host_projection_clears_the_k4_kill_gate_by_the_reported_margin():
    """Phase 0 3.3: T1 all-host is ~18.7 tok/s against a K4 hard-kill of 3.178 -- clears by ~5.9x."""
    got = project_step(
        host_bytes_per_rank=int(_REPORT_ACTIVE_PER_RANK), device_bytes_per_rank=0,
        prior=PHASE0_PRIOR, num_ranks=2,
    )
    assert got.tok_s / PHASE0_PRIOR.kill_tok_s == pytest.approx(5.9, abs=0.1)


def test_tp2_ceiling_is_gated_by_the_slow_card_not_the_average():
    """Card 1's root port trained Gen4 x8 against card 0's Gen5 x8, so its host read is exactly
    half. The links are independent (P4, efficiency 0.999), so the SLOW rank sets the step. Averaging
    the pair would over-project TP=2 by ~1.5x."""
    assert PHASE0_PRIOR.slow_host_gbps(2) == 14.48
    assert PHASE0_PRIOR.slow_host_gbps(1) == 28.93
    assert PHASE0_PRIOR.host_read_gbps[0] / PHASE0_PRIOR.host_read_gbps[1] == pytest.approx(2.0,
                                                                                            abs=0.01)


def test_loaded_bracket_is_slower_and_symmetric():
    """Under synthetic host DDR load BOTH cards collapse to ~12.4 GB/s (P4 copy-engine PROXY)."""
    assert PHASE0_PRIOR.slow_host_gbps(2, loaded=True) < PHASE0_PRIOR.slow_host_gbps(2)
    r = resolve(device_budget_bytes=8 * GB, project_loaded=True)
    fast = resolve(device_budget_bytes=8 * GB)
    assert r.projection.tok_s < fast.projection.tok_s


def test_sweep_covers_the_published_grid_and_is_monotone_in_f():
    layers = _uniform(48, 700 * (1 << 20), 1_400_000)
    rows = sweep_device_fraction(layers, PHASE0_PRIOR, num_ranks=2)
    assert [r.f_requested for r in rows] == list(PHASE0_PRIOR.f_grid)
    # More device budget can never mean fewer device bytes or a slower step.
    assert all(
        a.plan.device_resident_bytes <= b.plan.device_resident_bytes
        for a, b in zip(rows, rows[1:])
    )
    assert all(a.projection.tok_s <= b.projection.tok_s + 1e-9 for a, b in zip(rows, rows[1:]))


def test_sweep_quantises_to_whole_layers():
    """Layers are indivisible, so a requested f is realised as floor(f*n) layers. The residual is
    reported (`unused_device_bytes`), never hidden -- it is exactly what a per-expert split would
    capture, and P2' priced that at ~1%."""
    layers = _uniform(48, 700 * (1 << 20), 1_400_000)
    rows = {r.f_requested: r for r in sweep_device_fraction(layers, PHASE0_PRIOR, num_ranks=2)}
    assert rows[0.25].plan.num_device_layers == 12  # 0.25*48 exactly
    assert rows[0.10].plan.num_device_layers == 4   # 5 layers would be 0.104 > 0.10
    assert rows[0.10].plan.unused_device_bytes > 0


def test_f_zero_is_all_host_and_f_one_is_all_device():
    layers = _uniform(6, 1 * GiB, 2 * (1 << 20))
    assert plan_layer_granular(layers, device_budget_bytes=0).num_device_layers == 0
    full = plan_layer_granular(layers, device_budget_bytes=6 * GiB)
    assert full.num_host_layers == 0 and full.is_empty


def test_projection_refuses_to_run_on_a_non_linear_prior():
    """If a future box measures a CLIFF, this additive model is wrong and must fail loudly rather
    than quietly over-project."""
    cliffed = PHASE0_PRIOR.with_overrides(name="cliffed", miss_cost_is_linear=False)
    plan = plan_layer_granular(_uniform(4, 1 * GiB, 2 * (1 << 20)), device_budget_bytes=0)
    with pytest.raises(PlacementError, match="NOT linear"):
        project_plan(plan, cliffed, num_ranks=2)


# =================================================================================================
# PURITY -- the invariant that makes a rank-divergent plan unreachable
# =================================================================================================


def test_two_ranks_of_the_same_config_resolve_the_identical_plan():
    """The whole point. Rank 0 and rank 1 run this independently with no collective; if they could
    disagree, one would stream a layer the other holds resident and the model would emit plausible
    wrong text with no error anywhere."""
    a = resolve(make_config(rank=0), device_budget_bytes=8 * GB)
    b = resolve(make_config(rank=1), device_budget_bytes=8 * GB)
    assert a.plan.digest() == b.plan.digest()
    assert a.device_bytes == b.device_bytes
    assert a.as_dict()["device_layers"] == b.as_dict()["device_layers"]


def test_the_plan_does_not_read_the_environment():
    """No env gate at merge, and more importantly: an env difference between the two rank processes
    must not be able to move placement."""
    before = resolve(device_budget_bytes=8 * GB).plan.digest()
    os.environ["MINISGL_WEIGHT_OFFLOAD_POISON"] = "1"
    os.environ["MINISGL_MOE_G2FUSE"] = "0"
    try:
        after = resolve(device_budget_bytes=8 * GB).plan.digest()
    finally:
        os.environ.pop("MINISGL_WEIGHT_OFFLOAD_POISON", None)
        os.environ.pop("MINISGL_MOE_G2FUSE", None)
    assert before == after


def test_resolution_is_reproducible_across_calls():
    digests = {resolve(device_budget_bytes=8 * GB).plan.digest() for _ in range(5)}
    assert len(digests) == 1


def test_digest_changes_when_the_decision_changes():
    """A digest that could not distinguish two plans would make the cross-rank check vacuous."""
    a = resolve(device_budget_bytes=8 * GB)
    b = resolve(device_budget_bytes=16 * GB)
    assert a.plan.digest() != b.plan.digest()


# =================================================================================================
# VRAM ACCOUNTING + LOGGING
# =================================================================================================


def test_device_accounting_accepts_an_exact_match_and_small_slack():
    r = resolve(device_budget_bytes=8 * GB)
    r.assert_device_accounting(r.device_bytes)
    r.assert_device_accounting(r.device_bytes + (16 << 20))


def test_device_accounting_rejects_a_real_divergence():
    """If the plan and the allocator disagree, the KV pool is about to be sized against a fiction.
    That must be loud -- Phase 0 produced four cases of this stack reporting success over wrong
    state, so the check is on the DATA and never on a return code."""
    r = resolve(device_budget_bytes=8 * GB)
    with pytest.raises(AssertionError, match="VRAM accounting mismatch"):
        r.assert_device_accounting(r.device_bytes + 2 * GiB)
    with pytest.raises(AssertionError, match="VRAM accounting mismatch"):
        r.assert_device_accounting(0)


def test_no_device_layers_means_zero_device_bytes_to_bill():
    r = resolve(device_budget_bytes=0)
    assert r.device_bytes == 0
    r.assert_device_accounting(0)


def test_render_lines_carry_the_decision_the_capacity_and_the_gates():
    r = resolve(device_budget_bytes=8 * GB)
    text = "\n".join(r.render_lines())
    assert "weight-arena=" in text
    assert "capacity:" in text
    assert "device-fraction sweep" in text
    assert "K4" in text and "A1.7" in text
    assert all(line.startswith("[weight-offload]") for line in r.render_lines())


def test_as_dict_is_json_serialisable_and_carries_the_split():
    import json

    r = resolve(device_budget_bytes=8 * GB)
    blob = json.loads(json.dumps(r.as_dict()))
    assert blob["digest"] == r.plan.digest()
    assert len(blob["device_layers"]) == r.plan.num_device_layers
    assert len(blob["host_layers"]) == r.plan.num_host_layers
    assert set(blob["device_layers"]) & set(blob["host_layers"]) == set()


def test_analytic_only_sizing_is_flagged_for_reconciliation():
    """A plan built from the analytic estimate is still deterministic, but it must not be allocated
    against until the granule walker has re-checked it post-load."""
    r = resolve(device_budget_bytes=8 * GB)
    if not TORCH_IS_REAL:
        assert r.byte_source == "analytic"
        assert any("ANALYTIC model only" in w for w in r.warnings)


def test_zaya_style_fp8_experts_size_as_fp8_not_int4():
    """`zaya.py:620` passes fp8_experts from the declared scheme. Sizing an 8 GB fp8 expert stack as
    a 4 GB int4 one would halve the arena and the load would die at first touch."""
    mc = make_model_config(
        num_layers=8, is_cca_hybrid=True, num_experts=16, num_experts_per_tok=1,
        moe_intermediate_size=2048,
        quant=FakeQuant(bits=8, is_fp8_w8a8=True, is_int4=False, weight_type="float"),
    )
    layers, diag = build_planned_layers(make_config(mc, tp=1), prefer_meta=False)
    assert len(layers) == 4  # odd layers only
    assert diag["schemes"] == ["fp8"]
    # fp8 is 1 byte/weight; sizing it as int4 would halve the arena and the load would die at first
    # touch, so assert the magnitude, not just the label.
    assert layers[0].resident_bytes > 16 * 3 * 2048 * 2048


# =================================================================================================
# GENERALITY AND REPO RULES -- what a NEW model family or quant format must implement here.
# The answer must be NOTHING: `create_moe_quant_method`'s own contract is "purely config-driven, no
# model-name branch", `KERNEL_CORE_POLICY.md` makes a new weight format a WLoad policy on the shared
# core, and `MoELayer` is the ONE layer seven families share.
# =================================================================================================


def test_the_fp8_expert_signal_is_config_driven_not_family_gated():
    """`_fp8_experts_signal` used to read `is_cca_hybrid AND quant.is_fp8_w8a8` -- a model-family
    branch. `zaya.py:620` passes the flag FROM the declared quant, so any family may declare fp8
    experts; the family conjunct only encoded 'fp8 experts means ZAYA'."""
    from minisgl.weights.plan import _fp8_experts_signal

    fp8 = FakeQuant(is_fp8_w8a8=True, is_int4=False, is_compressed_tensors=True)
    assert _fp8_experts_signal(make_model_config(quant=fp8, is_cca_hybrid=False)) is True
    assert _fp8_experts_signal(make_model_config(quant=fp8, is_cca_hybrid=True)) is True
    assert _fp8_experts_signal(make_model_config(quant=FakeQuant())) is False


def test_a_non_zaya_fp8_checkpoint_is_sized_fp8():
    """The bytes have to follow the DECLARATION, not the family. fp8 is 1 byte/elem + an fp32
    per-output-channel scale; int4 is ~0.5 + a group scale, so getting this backwards is ~2x."""
    fp8_cfg = make_config(
        make_model_config(
            num_layers=4, num_experts=8, num_experts_per_tok=2,
            quant=FakeQuant(is_fp8_w8a8=True, is_int4=False), is_cca_hybrid=False,
        )
    )
    layers, diag = build_planned_layers(fp8_cfg, prefer_meta=False)
    assert diag["schemes"] == ["fp8"]
    assert layers


class _ForModuleQuant(FakeQuant):
    """A mixed-precision compressed-tensors config whose `for_module` answers a DIFFERENT scheme for
    the expert module names the old probe guessed. Shaped after Qwen3.8-27B-NVFP4, whose group_0
    target `layers.(56|..|63).mlp.(gate|up|down)_proj` matches an experts path by substring."""

    def __init__(self, **kw):
        super().__init__(ct_groups=(("re:.*mlp.*", "OTHER-SCHEME"),), **kw)
        self.probed = []

    def for_module(self, name):
        self.probed.append(name)
        return FakeQuant(is_fp8_w8a8=True, is_int4=False)


def test_expert_quant_is_the_headline_never_a_per_layer_for_module_probe():
    """No builder refines the EXPERT quant per layer: every family threads `expert_quant =
    config.quant` straight to `MoELayer`, and `create_moe_quant_method` then builds ONE container
    class for the whole stack. `for_module` is for DENSE linears (`qwen3_5.py:125/293`). Probing it
    here sized a mixed checkpoint's layers against a scheme the layer does not build."""
    q = _ForModuleQuant()
    cfg = make_config(make_model_config(num_layers=4, num_experts=8, num_experts_per_tok=2, quant=q))
    layers, diag = build_planned_layers(cfg, prefer_meta=False)
    assert diag["schemes"] == ["ct_int4"], "sized against for_module's answer, not the headline"
    assert q.probed == [], "for_module must not be consulted for routed experts"
    assert len(layers) == 4


def test_the_mtp_head_is_replicated_not_ep_sharded():
    """`qwen3_5_moe.py:200` builds the draft head `force_no_ep=True`, which is the third conjunct of
    `MoELayer.__init__:967` and is PER LAYER. Reported correctly even though the head is excluded
    from offload: `OFFLOAD_MTP_HEAD` is a policy constant somebody may flip after a measurement."""
    mc = make_model_config(num_layers=4, num_experts=8, num_experts_per_tok=2,
                           mtp_num_hidden_layers=1)
    cfg = make_config(mc, tp=2, enable_ep=True)
    shapes = moe_layer_shapes(cfg, layer_indices=moe_layer_indices(mc, include_mtp=True))
    body = [s for s in shapes if s.label == "moe"]
    head = [s for s in shapes if s.label == "moe_mtp"]
    assert len(head) == 1
    assert body[0].expert_parallel and body[0].num_local_experts == 4
    assert not head[0].expert_parallel and head[0].ep_size == 1
    assert head[0].num_local_experts == 8, "the draft head holds every expert"
    assert head[0].top_k_local == 2, "and routes the full top_k, not ceil(k/ep)"
    assert head[0].intermediate_size_per_partition == body[0].intermediate_size_per_partition // 2
    assert not head[0].offloadable


def test_an_unnameable_quant_format_does_not_take_the_boot_down(monkeypatch):
    """`resolve_weight_plan` sits on the boot path of EVERY serve. A weight format added to
    `layers/moe.py` -- the sanctioned way, per KERNEL_CORE_POLICY -- must not make it raise: the meta
    model builds the real container and sizes it without a transcription here."""
    import minisgl.weights.sizing as sizing

    class _Spec:
        total_bytes = 1 << 20
        granule_bytes = 1 << 12

    monkeypatch.setattr(sizing, "meta_gemm_spec", lambda *a, **k: _Spec())
    novel = FakeQuant(method="some-new-format", is_int4=False, is_compressed_tensors=False)
    cfg = make_config(make_model_config(num_layers=4, num_experts=256, num_experts_per_tok=8,
                                        quant=novel))
    r = resolve_weight_plan(cfg, prefer_meta=True, device_budget_bytes=0)
    assert r.byte_source == "meta"
    assert len(r.layers) == 4
    assert r.diagnostics["schemes"] == ["unknown"]


def test_mtp_paths_are_matched_by_segment_not_substring():
    """`is_mtp_path` gates the ONE exclusion the observed walk applies. Substring matching would
    silently drop a decoder layer whose module happens to be spelled `mtp_adapter` -- a whole layer
    quietly not offloaded, with the arena sized as if it were."""
    from minisgl.weights.plan import is_mtp_path

    assert is_mtp_path("mtp.layers.0.mlp.experts")
    assert is_mtp_path("model.mtp.layers.0.mlp.experts")
    assert not is_mtp_path("model.layers.3.mlp.experts")
    assert not is_mtp_path("model.layers.3.mtp_adapter.experts")


def test_build_planned_layers_prefers_the_observed_model_over_the_config_transcription():
    """When a model object exists NOTHING is transcribed: the layer set, the EP sharding and the
    byte counts are read off it. This is the escape from `models/utils.py:145`'s `MoEMLP`, which
    builds `MoELayer(...)` with no `quant=` at all -- so its experts are bf16 whatever
    `config.quant` declares, and the config path under-sizes that arena ~4x with no config signal
    that anything is wrong."""
    sentinel = ((), {"byte_source": "observed", "skipped": (), "schemes": [],
                     "n_offloadable_layers": 0, "sizing_agreement_min": None,
                     "sizing_agreement_max": None, "compute_dtype_bytes": None})
    import minisgl.weights.plan as plan_mod

    called = []

    def fake_observed(model, **kw):
        called.append(model)
        return sentinel

    orig = plan_mod.observed_planned_layers
    plan_mod.observed_planned_layers = fake_observed
    try:
        got = build_planned_layers(make_config(), model="A-MODEL", prefer_meta=False)
    finally:
        plan_mod.observed_planned_layers = orig
    assert called == ["A-MODEL"]
    assert got[1]["byte_source"] == "observed"


def test_a_dense_checkpoint_says_dense_is_not_planned_rather_than_nothing_to_do():
    """`MoE and dense land together` is not satisfied yet. The failure mode of the old message was
    an operator reading `weight-offload: OFF (no offloadable expert stacks)` on a dense 70B as
    'there is nothing here to offload', when the real statement is 'this planner only enumerates
    MoE layers'. The mechanism itself is already dense-capable -- granule.GranuleSpec takes
    num_experts=None -- so the gap is enumeration and it has to be visible."""
    r = resolve(make_config(make_model_config(is_moe=False, num_experts=0)))
    assert not r.enabled and r.feasible
    assert any("DENSE" in w and "enumeration" in w for w in r.warnings), r.warnings
    assert any("dense" in line.lower() for line in r.render_lines())


def test_a_moe_checkpoint_does_not_get_the_dense_warning():
    r = resolve()
    assert not any("DENSE" in w for w in r.warnings)


# =================================================================================================
# REGRESSION (memory-accounting/boot lens, 2026-09-03): the reservation must be a GUARANTEE.
#
# `arena_reservation_bytes` charged `round_up(payload, chunk)` — i.e. it assumed next-fit packs
# perfectly — while `StageARuntime.attach_host_arena` called `reserve()` with no
# `extra_max_region_bytes`, so `chunk_plan.headroom_chunks` fell back to the same assumption. Rows
# may not straddle a chunk and the bump allocator is forward-only, so an abandoned tail is gone: on
# the target shape (a ~1 GiB fused w13 row in a 2 GiB chunk) a 55 GiB payload reserved as 28 chunks
# holds ~40 GiB, and ~15 GiB of rows fall into `ArenaMemPool`'s `hipMalloc` fallback — VRAM on a
# 16 GB card, booked as host RAM. `seal()` refuses that, but only after the arena is pinned and the
# checkpoint is read, which is the exact late failure `host_capacity.py` exists to prevent.
# =================================================================================================


def test_arena_reservation_charges_the_next_fit_waste_not_just_the_rounding():
    chunk = 2 * GiB
    payload = 12 * GiB
    row = 1 * GiB
    # Perfect-packing arithmetic: 6 chunks. It is not achievable with 1 GiB rows.
    assert arena_reservation_bytes(payload, chunk) == 6 * chunk
    # Guaranteed arithmetic: only `chunk - row` per chunk is placeable.
    assert arena_reservation_bytes(payload, chunk, row) == 12 * chunk
    assert arena_reservation_bytes(payload, chunk, row) > arena_reservation_bytes(payload, chunk)
    # A bound of 0 means "unknown" and must be byte-identical to the old behaviour, so a caller that
    # cannot derive one is not silently given a different number.
    assert arena_reservation_bytes(payload, chunk, 0) == arena_reservation_bytes(payload, chunk)


def test_arena_reservation_grows_the_chunk_when_a_row_exceeds_it():
    """`headroom_chunks` guarantees `chunk - m` placeable bytes, which at `chunk <= m` is zero (it
    raises). `PinnedWeightArena.reserve` handles that by GROWING the chunk via `suggest_chunk_bytes`
    — so the planner has to charge the grown chunk, or it prices chunks the arena will not pin."""
    from minisgl.weights.plan import effective_chunk_bytes

    small, row = 512 * (1 << 20), 900 * (1 << 20)
    grown = effective_chunk_bytes(small, row)
    assert grown > row, "the chunk must be STRICTLY larger than any row it must hold"
    res = arena_reservation_bytes(8 * GiB, small, row)
    assert res % grown == 0 and res >= 8 * GiB
    assert effective_chunk_bytes(small, 0) == small


def test_reserve_with_a_row_bound_larger_than_the_chunk_does_not_raise():
    """REGRESSION. `reserve()` probed `suggest_chunk_bytes` with the bound ITSELF, which grows the
    chunk to exactly `round_up(bound)` — and `headroom_chunks` then sees `max_region >= chunk` and
    raises `RegionTooLargeError`. So a caller that did the right thing (passed its row bound) got an
    exception instead of a bigger chunk. The probe now asks for one granule more."""
    from minisgl.weights.pinned_arena import PinnedWeightArena

    row = 900 * (1 << 20)
    arena = PinnedWeightArena(
        device_index=0, chunk_bytes=512 * (1 << 20), local_ranks=1, floor_bytes=0, label="t"
    )
    plan = arena.reserve([], extra_bytes=4 * GiB, extra_max_region_bytes=row, check=False)
    assert plan.chunk_bytes > row
    assert plan.n_chunks * plan.chunk_bytes >= 4 * GiB


def test_the_resolver_and_the_arena_charge_the_same_bound():
    """One property, one definition. If what `StageARuntime.attach_host_arena` hands `reserve()`
    and what the resolver charged disagree, the resolver prints FEASIBLE plus a "raise the device
    tier to >= X" figure and then `reserve()`'s capacity gate refuses the boot anyway — handing the
    operator a tier that still does not work."""
    from minisgl.weights.bake import StageARuntime
    from minisgl.weights.chunk_plan import plan_regions

    r = resolve(device_budget_bytes=8 * GiB)
    stub = SimpleNamespace(plan=r.plan)
    # The FALLBACK bound (used when the rows cannot be enumerated) is still one definition...
    bound = StageARuntime.host_row_bound_bytes(stub)
    assert bound == r.plan.max_host_row_bytes
    # ...and so is the enumeration the driver actually passes to reserve().
    rows = StageARuntime.host_row_requests(stub)
    assert rows == r.plan.host_row_requests()
    assert plan_regions(rows, r.arena_chunk_bytes).reserved_bytes == (
        r.host_reservation_bytes_per_rank
    )
