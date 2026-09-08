"""M1-B: the pinned arena is reserved by ENUMERATING its rows, not by bounding them. No GPU, no torch.

THE DEFECT THIS FILE GUARDS. `StageARuntime.attach_host_arena` reserved the whole host tier as
anonymous headroom -- `arena.reserve([], extra_bytes=host_resident_bytes,
extra_max_region_bytes=max_host_row_bytes)`. That is a GUARANTEE derived from one number: a region
may never straddle a chunk and the bump allocator is forward-only, so with rows of up to `m` bytes
`chunk_plan.headroom_chunks` can only promise `chunk - m` placeable bytes per chunk. It has to price
the worst row landing at the worst offset in EVERY chunk because it has nothing else to go on.

On the target-shaped checkpoint (48 layers, E=512, top_k=10, hidden 2048, inter 768,
compressed-tensors int4 g32 symmetric, TP=2, no EP) the rows are 384/48/12 MiB for w13 and
192/24/6 MiB for w2 -- 666 MiB per layer, 31.22 GiB per rank. The bound charges 20 x 2 GiB chunks
(40.00 GiB/rank, 80.00 GiB/node) where the real next-fit allocator fits three whole layers per chunk
and uses 16 (32.00 GiB/rank, 64.00 GiB/node). Against the 55.80 GiB usable node budget that +28% is
paid in DEVICE TIER on a 16 GiB card: 11.06 GiB/rank required, versus 5.85 GiB/rank once the rows
are enumerated -- 5.2 GiB/rank of VRAM, ~1M KV tokens, recovered from an arithmetic assumption.

`sizing.meta_gemm_spec` builds the real container under `torch.device("meta")` before
`load_state_dict`, so the row sizes were always knowable at `reserve()` time. Two things that were
vacuous gain teeth as a side effect and are asserted here: `ChunkPlan.digest()` (which collapsed to
a hash of the chunk COUNT, so two ranks with different layouts printed the same string) and
`PinnedWeightArena.verify_matches_plan()` (which iterated an empty planned table).

The numbers below are COMPUTED by the code under test, not transcribed from a document. They are
locked so that a change to the byte model, the packing, or the ceiling is a failing test rather than
a quietly different device tier on a boot banner.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

# MUST precede the minisgl imports -- see `_offload_torch_stub`. Inert in the container.
from _offload_torch_stub import TORCH_IS_REAL  # noqa: F401,I001
from minisgl.weights.chunk_plan import (
    GIB,
    MIB,
    BumpAllocator,
    RegionRequest,
    plan_regions,
    torch_allocation_bytes,
)
from minisgl.weights.placement import (
    LayerWeights,
    PlacementError,
    plan_layer_granular,
)
from minisgl.weights.plan import (
    arena_reservation_bytes,
    exact_arena_reservation_bytes,
    plan_arena_reservation_bytes,
    required_device_bytes,
    resolve_weight_plan,
)
from minisgl.weights.sizing import expert_stack_bytes
from minisgl.weights.stacks import StackKind
from test_weight_plan import FakeQuant, make_config  # noqa: I001

CHUNK = 2 * GIB


def resolve(config=None, **kw):
    kw.setdefault("prefer_meta", False)
    return resolve_weight_plan(config if config is not None else make_config(), **kw)


# =================================================================================================
# torch's caching allocator does not ask for the tensor -- it asks for a SEGMENT
# =================================================================================================


def test_torch_allocation_bytes_rounds_to_the_2_MiB_segment_granule():
    """A reservation enumerated in TENSOR bytes is short by torch's own rounding, and the shortfall
    lands in `hipMalloc` VRAM. `kRoundLarge` is 2 MiB and the smallest segment torch ever asks a
    backing allocator for is `kSmallBuffer`, also 2 MiB."""
    assert torch_allocation_bytes(0) == 0
    assert torch_allocation_bytes(1) == 2 * MIB
    assert torch_allocation_bytes(2 * MIB) == 2 * MIB
    assert torch_allocation_bytes(2 * MIB + 1) == 4 * MIB
    assert torch_allocation_bytes(384 * MIB) == 384 * MIB
    # Never smaller than the row: an optimistic model under-reserves, which is the silent direction.
    for n in (1, 3, 5 * MIB - 7, 100 * MIB + 1):
        assert torch_allocation_bytes(n) >= n


def test_the_bump_footprint_counts_abandoned_tails():
    """`consumed_bytes` is not comparable across two runs of a forward-only allocator: an abandoned
    tail is arena that is gone though no cursor counts it. `footprint_bytes` is what
    `verify_matches_plan` compares carve-against-plan on, so it has to include them."""
    b = BumpAllocator(2 * MIB, n_chunks=4)
    b.allocate(MIB + MIB // 2, name="big")          # 1.5 MiB, leaves a 0.5 MiB tail
    assert b.footprint_bytes == MIB + MIB // 2
    b.allocate(MIB, name="next")                    # does not fit the tail -> chunk 1
    assert b.abandoned_bytes == MIB // 2
    assert b.footprint_bytes == 2 * MIB + MIB       # chunk 0 whole + 1 MiB of chunk 1


def test_forecast_survives_the_plan_and_is_separable_from_a_named_region():
    plan = plan_regions(
        [RegionRequest("named", 4096), RegionRequest("row", 8192, forecast=True)], 2 * MIB
    )
    assert [p.forecast for p in plan.placements] == [False, True]
    assert [p.name for p in plan.named_placements] == ["named"]
    assert [p.name for p in plan.forecast_placements] == ["row"]
    assert plan.forecast_bytes == 8192
    assert plan.footprint_bytes == plan.consumed_bytes


# =================================================================================================
# the rows themselves: enumerated, ordered, and forced to agree with the capacity number
# =================================================================================================


def _layer(rows, **kw):
    resident = sum(nb for _, nb in rows)
    base = {
        "path": "model.layers.0.mlp.experts",
        "num_experts": 8,
        "top_k": 2,
        "granule_bytes": resident // 8,
        "resident_bytes": resident,
        "rows": tuple(rows),
    }
    base.update(kw)
    return LayerWeights(**base)


def test_rows_must_account_for_every_resident_byte():
    """The capacity budget is spent in `resident_bytes` and the arena is reserved from the rows. If
    they disagree the plan charges one weight set and pins another -- a dropped scale shows up here
    as a byte count at boot instead of as plausible text at inference."""
    with pytest.raises(PlacementError, match="enumerated arena rows sum to"):
        _layer([("w13.weight", 1024)], resident_bytes=2048)
    with pytest.raises(PlacementError, match="duplicate arena row"):
        _layer([("w13.weight", 1024), ("w13.weight", 1024)])
    with pytest.raises(PlacementError, match="must be > 0 bytes"):
        _layer([("w13.weight", 1024), ("w13.scale", 0)], resident_bytes=1024)


def test_a_packing_bound_smaller_than_a_real_row_is_refused():
    """An optimistic bound under-reserves and pushes the overflow into VRAM."""
    with pytest.raises(PlacementError, match="smaller than the largest enumerated row"):
        _layer([("w13.weight", 4096), ("w2.weight", 1024)], max_row_bytes=2048)


def test_row_bound_falls_out_of_the_rows_when_no_bound_was_declared():
    lw = _layer([("w13.weight", 4096), ("w2.weight", 1024)])
    assert lw.row_bound == 4096
    assert LayerWeights(
        path="p", num_experts=2, top_k=1, granule_bytes=8, resident_bytes=64
    ).row_bound == 64          # nothing declared at all -> the whole layer, sound but loose


def test_host_row_requests_covers_host_layers_only_and_keeps_carve_order():
    rows = [("w13.weight", 4 * MIB), ("w13.scale", 2 * MIB), ("w2.weight", 2 * MIB)]
    layers = [
        _layer(rows, path=f"model.layers.{i}.mlp.experts") for i in range(3)
    ]
    plan = plan_layer_granular(layers, device_budget_bytes=8 * MIB)  # exactly one layer on device
    assert plan.num_device_layers == 1 and plan.host_rows_known
    reqs = plan.host_row_requests()
    # ONE segment, not six requests. Every row here (4, 2, 2 MiB x two host layers) is inside torch's
    # `kLargeBuffer` band, so the FIRST one opens a single 20 MiB segment and the remaining five are
    # served from its split remainder without ever reaching the arena's allocator. The request list
    # is what the arena will really be asked for, so it names the row that OPENS each segment and
    # carries the segment's size -- see `chunk_plan.torch_charged_rows`.
    assert [(r.name, r.nbytes) for r in reqs] == [
        ("model.layers.1.mlp.experts.w13.weight", 20 * MIB)
    ]
    assert all(r.forecast for r in reqs)
    # CARVE ORDER is still the invariant: whatever survives is a subsequence of the host rows, in
    # order, and never reordered across layers.
    host_order = [f"model.layers.{i}.mlp.experts.{n}" for i in (1, 2) for n, _ in rows]
    got = [r.name for r in reqs]
    assert got == [n for n in host_order if n in set(got)]
    # A device layer never touches the arena, so charging its rows would reserve host RAM for
    # weights staying in VRAM -- and the greedy fill puts the LARGEST layers on the device.
    assert not any(".layers.0." in r.name for r in reqs)


def test_a_partially_enumerated_plan_reserves_by_the_bound_instead():
    """All or nothing. A half-enumerated plan would reserve exactly for the layers it can see and
    nothing at all for the rest -- short, in the silent direction."""
    rows = [("w13.weight", 4 * MIB), ("w2.weight", 4 * MIB)]
    layers = [
        _layer(rows, path="model.layers.0.mlp.experts"),
        LayerWeights(path="model.layers.1.mlp.experts", num_experts=8, top_k=2,
                     granule_bytes=MIB, resident_bytes=8 * MIB),
    ]
    plan = plan_layer_granular(layers, device_budget_bytes=0)
    assert not plan.host_rows_known
    assert plan.host_row_requests() == ()
    # ...and the reservation then falls back to the guarantee, not to perfect packing.
    assert plan_arena_reservation_bytes(plan, CHUNK) == arena_reservation_bytes(
        plan.host_resident_bytes, CHUNK, plan.max_host_row_bytes
    )


# =================================================================================================
# sizing: the rows come out of the byte model that already exists
# =================================================================================================


def _sized(quant, **kw):
    base = {
        "num_local_experts": 512,
        "hidden_size": 2048,
        "intermediate_size_per_partition": 384,
        "prefer_meta": False,
    }
    base.update(kw)
    return expert_stack_bytes(quant=quant, **base)


def test_the_analytic_rows_sum_to_the_container_total():
    s = _sized(FakeQuant())
    assert s.rows
    assert sum(nb for _, nb in s.rows) == s.total
    assert [n for n, _ in s.rows] == [
        "w13.weight", "w13.scale", "w13.post_load", "w2.weight", "w2.scale", "w2.post_load"
    ]
    assert [nb // MIB for _, nb in s.rows] == [384, 48, 12, 192, 24, 6]


def test_the_compressed_tensors_symmetric_zeros_are_a_row_of_their_own():
    """`post_load` allocates a REAL `torch.empty` the container `__init__` never made, and
    `_plan_items` copies it into the arena like any other tensor -- so it is carved as its own row,
    not folded into a neighbour."""
    s = _sized(FakeQuant(sym=True))
    assert dict(s.rows)["w13.post_load"] == 512 * (2048 // 32) * ((2 * 384) // 8) * 4


def test_the_mxfp4_scale_row_IS_THE_E8M0_BYTE_and_gains_no_sibling():
    """One u8 row, no widening and no post_load sibling.

    The E8M0 byte now reaches the kernel natively, so `post_load` leaves the buffer alone. Two
    things are pinned here and they pull in opposite directions: the row must be charged at ONE
    byte per group (charging two over-reserves and silently shrinks the KV pool), and the widening
    must not reappear as a SEPARATE row either — two half-size rows pack into a chunk tail that one
    full-size row does not, so a sibling row would make the reservation optimistic in exactly the
    place the never-straddle rule bites."""
    mx = FakeQuant(is_int4=False, is_compressed_tensors=False, weight_is_e2m1=True, group_size=32)
    s = _sized(mx)
    names = [n for n, _ in s.rows]
    assert "w13.post_load" not in names
    E, N, K = 512, 2 * 384, 2048
    assert dict(s.rows)["w13.scale"] == E * N * (K // 32)   # 1 B/group, native E8M0
    assert sum(nb for _, nb in s.rows) == s.total


def test_a_scheme_neither_model_can_break_down_yields_no_rows_and_not_a_wrong_one():
    """A format added to `layers/moe.py` since must not take the boot down, and must not be
    reserved against a row list that does not describe it."""
    unknown = FakeQuant(is_int4=False, is_compressed_tensors=False)
    with pytest.raises(ValueError):
        _sized(unknown)   # both byte models refuse: torch-free host, no analytic arm


# =================================================================================================
# the META model's specs are `__init__` shapes, and two containers grow at post_load()
# =================================================================================================


class _FakeComponent:
    def __init__(self, name, nbytes):
        self.name, self.nbytes = name, nbytes


class _FakeSpec:
    """Duck-typed `granule.GranuleSpec`: the four attributes `placement` reads, nothing else."""

    def __init__(self, comps, replicated=(), n=8, meta=True):
        self.components = tuple(_FakeComponent(a, b) for a, b in comps)
        self.replicated = tuple(_FakeComponent(a, b) for a, b in replicated)
        self.num_granules = n
        self.num_experts = n
        self.meta = meta

    @property
    def granule_bytes(self):
        return sum(c.nbytes for c in self.components)

    @property
    def total_bytes(self):
        return self.granule_bytes * self.num_granules + sum(r.nbytes for r in self.replicated)

    def fingerprint(self):
        return "fp"


def _pair():
    return (
        _FakeSpec([("weight", 4 * MIB), ("scale", MIB)]),
        _FakeSpec([("weight", 2 * MIB), ("scale", MIB // 2)]),
    )


def test_from_specs_without_a_correction_is_unchanged():
    """Off LIVE post-load containers the tensors are already there; adding the delta again would
    double-charge the arena. The correction is opt-in for exactly that reason."""
    w13, w2 = _pair()
    lw = LayerWeights.from_specs("p", num_experts=8, top_k=2, w13=w13, w2=w2)
    assert lw.resident_bytes == w13.total_bytes + w2.total_bytes
    assert sum(nb for _, nb in lw.rows) == lw.resident_bytes


def test_a_new_row_correction_is_charged_to_capacity_but_not_to_traffic():
    """Compressed-tensors SYMMETRIC: `post_load` allocates a real `_zeros_op` of E identical rows.
    Resident (the arena must hold it), NOT granule (every expert reads the same row)."""
    from minisgl.weights.placement import PostLoadCorrection

    w13, w2 = _pair()
    base = LayerWeights.from_specs("p", num_experts=8, top_k=2, w13=w13, w2=w2)
    lw = LayerWeights.from_specs(
        "p", num_experts=8, top_k=2, w13=w13, w2=w2,
        w13_post_load=PostLoadCorrection(resident=MIB, granule=0),
    )
    assert lw.resident_bytes == base.resident_bytes + MIB
    assert lw.granule_bytes == base.granule_bytes
    assert ("w13.post_load", MIB) in lw.rows
    assert sum(nb for _, nb in lw.rows) == lw.resident_bytes


def test_a_widening_correction_grows_the_row_it_names_instead_of_adding_one():
    """MXFP4's E8M0 u8 scale becomes fp16 in the SAME buffer. Two half-size rows pack into a chunk
    tail that one full-size row does not, so this distinction is packing, not bookkeeping."""
    from minisgl.weights.placement import PostLoadCorrection

    w13, w2 = _pair()
    scale_row = MIB * 8            # `scale` component, x8 granules
    lw = LayerWeights.from_specs(
        "p", num_experts=8, top_k=2, w13=w13, w2=w2,
        w13_post_load=PostLoadCorrection(resident=scale_row, granule=MIB, widen_row_bytes=scale_row),
    )
    assert dict(lw.rows)["w13.scale"] == 2 * scale_row
    assert "w13.post_load" not in dict(lw.rows)
    assert sum(nb for _, nb in lw.rows) == lw.resident_bytes


def test_the_observed_path_asks_for_a_correction_only_on_meta():
    """Source-level; the runtime version needs a built model and therefore real torch.

    `engine.py` builds on meta and resolves there — that is what makes the capacity abort land
    before `load_state_dict` — so the observed path reads `__init__` shapes. Re-run after
    `post_load()` with `allow_meta=False` and the same call must NOT correct, or the arena is
    charged twice for the same tensor.
    """
    import ast
    import inspect

    from minisgl.weights.plan import observed_planned_layers

    tree = ast.parse(inspect.getsource(observed_planned_layers))
    src = ast.dump(tree)
    assert "_meta_corrections" in src
    assert "is_meta" in src, "the correction must be gated on the spec being meta-derived"


# =================================================================================================
# THE HEADLINE: what the target-shaped checkpoint now needs
# =================================================================================================


def test_exact_packing_beats_the_bound_on_the_target_shape():
    r = resolve()
    rows = r.plan.host_row_requests()
    # NOT `48 * 6` — a request is a torch SEGMENT, and segments are not 1:1 with rows. The target
    # shape's `w2.post_load` row is 6 MiB, i.e. inside torch's `kLargeBuffer` band, so it opens a
    # 20 MiB segment whose remainder then serves the 12 MiB `w13.post_load` of the NEXT layer with no
    # arena callback at all. 241 segments for 288 rows, and the composition is exact:
    #   384 x48, 192 x48, 48 x48, 24 x48   the four slabs above kMinLargeAlloc, one segment each
    #   20  x48                            one kLargeBuffer per layer, opened by w2.post_load
    #   12  x1                             layer 0's w13.post_load, before any remainder exists
    # See `chunk_plan.torch_charged_rows`, whose model is checked against a live
    # MINISGL_ARENA_TRACE_ALLOCS=1 callback trace.
    from collections import Counter

    assert len(rows) == 241
    assert Counter(rq.nbytes // MIB for rq in rows) == {384: 48, 192: 48, 48: 48, 24: 48, 20: 48,
                                                        12: 1}

    payload = sum(nb for _, nb in r.layers[0].rows) * 48
    assert payload == r.host_bytes_per_rank
    assert r.layers[0].resident_bytes == 666 * MIB

    exact = exact_arena_reservation_bytes(rows, CHUNK)
    bound = arena_reservation_bytes(payload, CHUNK, r.plan.max_host_row_bytes)
    assert exact == 16 * CHUNK          # three whole layers per chunk, 50 MiB abandoned each
    assert bound == 20 * CHUNK          # ceil(payload / (chunk - 444 MiB))
    assert bound / exact == pytest.approx(1.25, abs=0.01)
    # And the exact figure is what the resolution now charges, per rank and per node.
    assert r.host_reservation_bytes_per_rank == exact
    assert r.host_reservation_bytes_per_node == 2 * exact


def test_the_required_device_tier_the_target_shape_now_needs():
    """LOCKED. This is the number M1-B exists to move, and it is COMPUTED here, not transcribed.

    Before: 11.06 GiB/rank (17 of 48 layers) on a 16 GiB card, with the tier billed inside
    `model_memory`, leaving ~3.3 GiB at `--memory-ratio 0.9` for the dense weights, the KV pool, the
    recurrent state, the draft model and the graph buffers -- very likely unbootable.
    After: 5.85 GiB/rank (9 layers), 39 host layers in 13 chunks = 52.00 GiB/node against the
    55.80 GiB usable ceiling.
    """
    infeasible = resolve()
    assert not infeasible.feasible           # all-host is 64.00 GiB/node, over the ceiling
    assert infeasible.host_reservation_bytes_per_node == 64 * GIB
    assert infeasible.host_ceiling_bytes == pytest.approx(55.8 * GIB, rel=1e-6)

    need = infeasible.required_device_bytes
    layer = infeasible.layers[0].resident_bytes
    assert need == 9 * layer
    assert need / GIB == pytest.approx(5.8535, abs=1e-3)

    ok = resolve(device_budget_bytes=need)
    ok.raise_if_infeasible()
    assert ok.plan.num_device_layers == 9 and ok.plan.num_host_layers == 39
    assert ok.host_reservation_bytes_per_rank == 13 * CHUNK
    assert ok.host_reservation_bytes_per_node == 52 * GIB <= ok.host_ceiling_bytes

    # MINIMAL: one layer less of tier must not fit, or the resolver is over-charging the operator.
    assert not resolve(device_budget_bytes=need - layer).feasible


def test_the_old_bound_would_still_demand_seventeen_layers():
    """The counterfactual, so the win is attributable to the packing and not to a byte-model drift.

    Same layers, same ceiling, same chunk -- only the reservation arithmetic differs.
    """
    r = resolve()
    layer = r.layers[0].resident_bytes
    bound_only = tuple(
        LayerWeights(
            path=lw.path, num_experts=lw.num_experts, top_k=lw.top_k,
            granule_bytes=lw.granule_bytes, resident_bytes=lw.resident_bytes,
            max_row_bytes=lw.max_row_bytes,          # rows deliberately dropped
        )
        for lw in r.layers
    )
    old = required_device_bytes(bound_only, r.host_ceiling_bytes, r.local_ranks, CHUNK)
    assert old == 17 * layer
    assert old / GIB == pytest.approx(11.0566, abs=1e-3)
    assert old - r.required_device_bytes == 8 * layer      # 5.20 GiB/rank of VRAM recovered
    assert (old - r.required_device_bytes) / GIB == pytest.approx(5.2031, abs=1e-3)


def test_required_device_bytes_orders_the_rows_by_declaration_not_by_priority():
    """The bake binds seams in declaration order and the arena's allocator is order-sensitive, so a
    reservation laid out in the greedy walk's priority order is not the one the bake consumes."""
    rows = [("w13.weight", 700 * MIB), ("w2.weight", 300 * MIB)]
    layers = [
        _layer(rows, path=f"model.layers.{i}.mlp.experts", priority=(10 if i == 5 else 0))
        for i in range(8)
    ]
    # 8 layers x 1000 MiB, two per 2 GiB chunk (2000 MiB, 48 MiB abandoned): 4 chunks all-host.
    assert exact_arena_reservation_bytes(
        plan_layer_granular(layers, device_budget_bytes=0).host_row_requests(), CHUNK
    ) == 4 * CHUNK
    need = required_device_bytes(layers, 3 * CHUNK, 1, CHUNK)
    plan = plan_layer_granular(layers, device_budget_bytes=need)
    assert plan.kind_of("model.layers.5.mlp.experts") is StackKind.DEVICE
    # Whatever tier it returns must produce a reservation inside the ceiling -- computed the way
    # `attach_host_arena` will compute it.
    assert exact_arena_reservation_bytes(plan.host_row_requests(), CHUNK) <= 3 * CHUNK


def test_the_resolver_charges_what_the_arena_will_pin_even_when_a_row_forces_a_bigger_chunk():
    """`reserve()` GROWS the chunk when a single region does not fit one (a region may never
    straddle two independent `hipHostMalloc` mappings). The planner must charge the grown chunk, or
    it prices chunks the arena will not pin -- and it must grow it the same way, via
    `suggest_chunk_bytes`, which is a pure function of the request list so every rank agrees."""
    from minisgl.weights.chunk_plan import RegionTooLargeError

    rows = [("w13.weight", 3 * GIB), ("w2.weight", GIB)]
    plan = plan_layer_granular(
        [_layer(rows, path="model.layers.0.mlp.experts")], device_budget_bytes=0
    )
    reqs = plan.host_row_requests()
    with pytest.raises(RegionTooLargeError):
        plan_regions(reqs, CHUNK)          # the row does not fit the default chunk at all
    reserved = exact_arena_reservation_bytes(reqs, CHUNK)
    assert reserved == 2 * (3 * GIB)       # chunk grown to 3 GiB; 3 GiB + 1 GiB needs two of them


# =================================================================================================
# the shipping path actually uses it
# =================================================================================================


def test_attach_host_arena_reserves_the_enumerated_rows():
    """Source-level: the runtime version builds an `ArenaMemPool`, which needs a card.

    Reverting the first positional argument to `[]` restores the +28% reservation exactly -- it
    imports, boots, and costs the operator ~5 GiB/rank of device tier with nothing in the log.
    """
    import ast
    from pathlib import Path

    src = (
        Path(__file__).resolve().parents[2] / "python" / "minisgl" / "weights" / "bake.py"
    ).read_text()
    # Selected by "the definition that CONTAINS a reserve() call": `StageADriver` declares a
    # same-named Protocol stub earlier in the file and `ast.walk` reaches it first.
    call = None
    for fn in ast.walk(ast.parse(src)):
        if not (isinstance(fn, ast.FunctionDef) and fn.name == "attach_host_arena"):
            continue
        for n in ast.walk(fn):
            if (
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr == "reserve"
            ):
                call = n
    assert call is not None, "StageARuntime.attach_host_arena no longer calls arena.reserve()"
    assert call.args and isinstance(call.args[0], ast.Name) and call.args[0].id == "rows", (
        "attach_host_arena must reserve the ENUMERATED rows; an empty list is the anonymous-headroom "
        "reservation this milestone removed"
    )
    # ...and the bounded-headroom fallback is still wired for a plan that cannot enumerate.
    assert {k.arg for k in call.keywords} == {"extra_bytes", "extra_max_region_bytes"}


def test_seal_verifies_the_arena_layout():
    """`verify_matches_plan()` had no production caller, which is why nobody noticed it was
    checking nothing. It runs inside `seal()`, before the KV pool is sized."""
    from minisgl.weights.accounting import WeightArenaAccounting
    from minisgl.weights.bake import StageAPhase, StageASession

    seen = {}

    class _Driver:
        def assert_arena_clean(self):
            seen["clean"] = True

        def verify_arena_layout(self):
            seen["layout"] = True
            return SimpleNamespace(describe=lambda: "checked")

        def freeze(self):
            seen["freeze"] = True

    s = StageASession(
        driver=_Driver(),
        accounting=WeightArenaAccounting(host_bytes=1, device_bytes=0, offloadable_bytes=1),
        phase=StageAPhase.BOUND,
    )
    s.accounting.copied_bytes = 1
    s.accounting.observed_device_bytes = 0
    try:
        s.seal()
    except Exception:
        pass  # the ledger gates are not what this test is about
    assert seen.get("layout"), "seal() must verify the carved layout, not only the fallback count"
