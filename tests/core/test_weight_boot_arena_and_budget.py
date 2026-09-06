"""Boot-path regressions found through the MEMORY-ACCOUNTING-AND-BOOT lens. GPU-free, torch-free.

Three defects, all in `weights/bake.StageARuntime` / its engine call site, all of which present as
something other than themselves:

1. **The host arena was reserved with UNBOUNDED headroom.** `StageARuntime.attach_host_arena` called
   `arena.reserve([], extra_bytes=host_resident_bytes)` with no `extra_max_region_bytes`.
   `chunk_plan.headroom_chunks` says outright that this is wrong and wrong in the direction that
   costs a boot — the bump allocator may never straddle a chunk, so next-fit abandons a tail per
   chunk and the last rows do not fit. They do not fail loudly: `allocate_raw` returns None,
   `ArenaMemPool` falls back to `hipMalloc`, and weights the capacity plan booked against HOST RAM
   land in VRAM — which `model_memory_correction()` then subtracts from the model term as if it were
   host memory, so the KV pool is oversized by twice the shortfall.
2. **`arena_torch_bytes()` was not a measurement of torch.** It returned
   `TorchStackAllocator.bytes_used(HOST)` — the bytes this module handed out, which equals
   `copied_bytes` by construction whatever torch did with them. Its documented contract ("0 if the
   rows bypass it") could therefore never be met, and every documented way the `MemPool` stops
   routing (wrong device at TP=2, a torch that changes `use_mem_pool`) produced a full-size
   correction against a `memory_allocated()` that never contained the bytes.
3. **The derived device tier is the WHOLE KV budget.** With no `--weight-offload-device-gb`,
   `Engine._weight_offload_device_budget` returns `total_memory * memory_ratio`. The device tier is
   billed inside `model` in `_determine_num_pages` (no sixth subtrahend, by design), so a non-empty
   plan under that budget makes `available_memory` negative before state/draft/graph/snap are even
   counted. `assert num_pages > 1` cannot not fire — but only after the arena has pinned tens of GiB
   of unevictable host RAM and the whole checkpoint has been read.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
from minisgl.weights import bake as bake_mod
from minisgl.weights.accounting import WeightArenaAccounting
from minisgl.weights.bake import (
    StageARuntime,
    UnconfiguredDeviceTierError,
    _refuse_derived_budget_that_leaves_no_kv,
    _resolve_driver,
)
from minisgl.weights.chunk_plan import BumpAllocator, plan_regions
from minisgl.weights.placement import LayerPlacement, OffloadPlan
from minisgl.weights.plan import resolve_weight_plan
from minisgl.weights.stacks import StackKind


def _needs_arena_settings() -> None:
    """`_resolve_driver` reads `weights/config.resolve_arena_settings()`, which reaches
    `minisgl.kvcache._envutil` and from there the serve image's torch/MPI stack. Everything else in
    this file is pure integers and runs on a bare host; these three cases need the image."""
    try:
        from minisgl.weights.config import resolve_arena_settings

        resolve_arena_settings()
    except BaseException as exc:  # noqa: BLE001 - a missing .so raises OSError, not ImportError
        pytest.skip(f"needs the serve image (arena settings pull in minisgl.kvcache): {exc}")

GiB = 1 << 30
MiB = 1 << 20
REPO = Path(__file__).resolve().parents[2]
ENGINE_PY = REPO / "python" / "minisgl" / "engine" / "engine.py"

CARD_TOTAL_BYTES = 16 * GiB  # gfx1201, both cards
MEMORY_RATIO = 0.9


# =================================================================================================
# DEFECT 1 — the arena reservation must bound a single row
# =================================================================================================


def _next_fit_all(rows, *, chunk_bytes: int, n_chunks: int) -> bool:
    """Place `rows` into `n_chunks` with the SAME allocator the live arena carves with.

    Not a model of the arena: `BumpAllocator` is the one piece of arithmetic shared by
    `plan_regions()` and `PinnedWeightArena.allocate_raw`, so this is literally the code that decides
    whether a row lands in the arena or in VRAM.
    """
    b = BumpAllocator(chunk_bytes, n_chunks=n_chunks)
    for i, n in enumerate(rows):
        if b.try_allocate(n, name=f"row{i}") is None:
            return False
    return True


def test_unbounded_headroom_under_reserves_and_the_last_rows_fall_out():
    """The defect, in the arithmetic that runs at boot.

    34 GiB of 400 MiB rows into 2 GiB chunks: 5 rows fit per chunk (2000 of 2048 MiB), so the
    `ceil(bytes/chunk)` reservation is short by the abandoned 48 MiB tails and the tail rows have
    nowhere to go. In the arena those rows become `hipMalloc` VRAM, silently, while the capacity plan
    still says they are host-resident.
    """
    chunk, row = 2 * GiB, 400 * MiB
    payload = 87 * row  # ~34 GiB, P3b's measured single-rank ceiling
    rows = [row] * 87

    unbounded = plan_regions([], chunk, extra_bytes=payload)
    assert not _next_fit_all(rows, chunk_bytes=chunk, n_chunks=unbounded.n_chunks), (
        "if the unbounded reservation happened to fit, this fixture no longer reproduces the defect"
    )

    bounded = plan_regions([], chunk, extra_bytes=payload, extra_max_region_bytes=row)
    assert _next_fit_all(rows, chunk_bytes=chunk, n_chunks=bounded.n_chunks)
    assert bounded.n_chunks > unbounded.n_chunks
    # The bound is a GUARANTEE, not an estimate, so it over-reserves. That direction is charged
    # against the live MemAvailable gate inside reserve() and aborts in milliseconds; the other
    # direction is silent VRAM.
    assert bounded.reserved_bytes > payload


def test_the_unbounded_reservation_is_the_one_that_advises_against_itself():
    """`ChunkPlan.headroom_advisory()` names the missing argument. It fires only without the bound."""
    chunk, row = 2 * GiB, 400 * MiB
    payload = 87 * row
    assert plan_regions([], chunk, extra_bytes=payload).headroom_advisory()
    assert (
        plan_regions([], chunk, extra_bytes=payload, extra_max_region_bytes=row).headroom_advisory()
        is None
    )


def _placement(kind, resident, *, max_row=0):
    return LayerPlacement(
        path="m.layers.0.mlp",
        kind=kind,
        resident_bytes=resident,
        granule_bytes=max(1, resident // 64),
        num_experts=64,
        top_k=6,
        max_row_bytes=max_row,
    )


def _runtime_with_plan(placements):
    """A `StageARuntime` with only its plan wired — enough for the pure planning helpers.

    `object.__new__` rather than the constructor: `__init__` builds a real `PinnedWeightArena` and
    reads the arena env, neither of which this assertion is about.
    """
    r = object.__new__(StageARuntime)
    r.resolution = SimpleNamespace(
        plan=OffloadPlan(
            placements=tuple(placements),
            device_budget_bytes=0,
            total_resident_bytes=sum(p.resident_bytes for p in placements),
        )
    )
    return r


def test_host_row_bound_is_the_largest_host_row_and_ignores_device_layers():
    """Device placements must not raise the bound: nothing is ever carved out of the arena for them
    (`MoEWeightSeam.bind(DEVICE)` moves no bytes), and at the operating points capacity forces the
    device tier holds the LARGEST layers by construction — so including them would inflate the
    reservation by exactly the layers that are not there, against a measured 62 GiB two-rank ceiling.
    """
    r = _runtime_with_plan(
        [
            _placement(StackKind.HOST, 400 * MiB, max_row=200 * MiB),
            _placement(StackKind.HOST, 700 * MiB, max_row=350 * MiB),
            _placement(StackKind.DEVICE, 9 * GiB, max_row=4 * GiB),
        ]
    )
    assert r.host_row_bound_bytes() == 350 * MiB


def test_host_row_bound_falls_back_to_the_whole_layer_when_no_row_size_is_known():
    """Sound but loose: `resident_bytes` bounds every row because a layer's container tensors sum to
    it. Loose is the safe direction here — it over-reserves, and over-reserving is refused by the
    live MemAvailable gate in milliseconds rather than landing weights in VRAM."""
    r = _runtime_with_plan([_placement(StackKind.HOST, 700 * MiB)])
    assert r.host_row_bound_bytes() == 700 * MiB


def test_host_row_bound_is_zero_for_an_all_device_plan():
    r = _runtime_with_plan([_placement(StackKind.DEVICE, 9 * GiB)])
    assert r.host_row_bound_bytes() == 0
    # 0 means "unbounded", which is correct here: with no host rows there is no headroom to bound,
    # and `headroom_chunks(0, ...)` reserves nothing at all.
    assert plan_regions([], 2 * GiB, extra_bytes=0, extra_max_region_bytes=0).n_chunks == 0


def _reserve_call_keywords() -> set:
    """Keywords of the `arena.reserve(...)` call inside the REAL `attach_host_arena`.

    Selected by "the definition that contains a reserve() call", not by name: `StageADriver` declares
    a same-named Protocol stub earlier in the file, and `ast.walk` reaches it first.
    """
    src = (REPO / "python" / "minisgl" / "weights" / "bake.py").read_text()
    for fn in ast.walk(ast.parse(src)):
        if not (isinstance(fn, ast.FunctionDef) and fn.name == "attach_host_arena"):
            continue
        for n in ast.walk(fn):
            if (
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr == "reserve"
            ):
                return {k.arg for k in n.keywords}
    raise AssertionError("StageARuntime.attach_host_arena no longer calls arena.reserve()")


def test_attach_host_arena_bounds_its_headroom():
    """Source-level, because the runtime version needs a card: `attach_host_arena` builds an
    `ArenaMemPool`, which needs a live `torch.cuda.MemPool` over the custom allocator.

    Dropping the keyword is a one-line edit that imports, boots, and silently puts the tail of the
    host tier in VRAM.
    """
    kwargs = _reserve_call_keywords()
    assert "extra_bytes" in kwargs
    assert "extra_max_region_bytes" in kwargs, (
        "reserve() without a bound on a single row falls back to ceil(bytes/chunk), which "
        "under-reserves by the abandoned next-fit tails; the rows that fall out become hipMalloc "
        "VRAM with the capacity plan still calling them host-resident"
    )


# =================================================================================================
# DEFECT 2 — `arena_torch_bytes` must measure torch, not the caller
# =================================================================================================


class _Allocator:
    """A `TorchStackAllocator` stand-in that has handed out the whole host tier."""

    def __init__(self, host_bytes: int) -> None:
        self._host = host_bytes

    def bytes_used(self, kind) -> int:
        return self._host if kind is StackKind.HOST else 0


def _runtime_with_pool(*, served_bytes, allocator_host_bytes):
    r = object.__new__(StageARuntime)
    r.pool = None if served_bytes is None else SimpleNamespace(served_bytes=served_bytes)
    r.allocator = _Allocator(allocator_host_bytes)
    return r


def test_arena_torch_bytes_reports_what_the_pool_served_not_what_was_handed_out():
    """The two disagree exactly when the `MemPool` stopped routing — the case the term exists for."""
    r = _runtime_with_pool(served_bytes=30 * GiB, allocator_host_bytes=30 * GiB)
    assert r.arena_torch_bytes() == 30 * GiB


def test_arena_torch_bytes_is_zero_when_the_pool_never_routed():
    """`torch.cuda.use_mem_pool` installed on the wrong device at TP=2 is the documented shape: the
    alloc callback never fires, `served_bytes` stays 0, and every allocation went to VRAM through the
    ordinary caching allocator. The stack allocator still counted the full host tier, so the OLD
    implementation returned 30 GiB here and the correction subtracted 30 GiB from a
    `memory_allocated()` that never contained a byte of it."""
    r = _runtime_with_pool(served_bytes=0, allocator_host_bytes=30 * GiB)
    assert r.arena_torch_bytes() == 0


def test_arena_torch_bytes_is_zero_before_the_pool_exists():
    r = _runtime_with_pool(served_bytes=None, allocator_host_bytes=30 * GiB)
    assert r.arena_torch_bytes() == 0


def _ledger(*, arena_torch_bytes, copied=30 * GiB):
    """A four-sample window in which the arena is INVISIBLE to torch: nothing was added to
    `allocated`, and the drop returned the copied bytes."""
    a = WeightArenaAccounting(
        host_bytes=copied,
        device_bytes=6 * GiB,
        offloadable_bytes=copied + 6 * GiB,
        copied_bytes=copied,
        arena_torch_bytes=arena_torch_bytes,
        observed_device_bytes=6 * GiB,
    )
    base = 40 * GiB
    a.sample("pre_attach", free=100 * GiB, allocated=base, reserved=base)
    a.sample("post_attach", free=100 * GiB, allocated=base, reserved=base)
    a.sample("pre_bake", free=100 * GiB, allocated=base, reserved=base)
    a.sample("post_bake", free=100 * GiB, allocated=base - copied, reserved=base)
    return a


def test_a_bypassed_pool_costs_nothing_once_arena_torch_bytes_is_measured():
    """With the term measured (0), the window reads exactly as it should: the originals came back,
    nothing needs correcting, and every serve that does not route through the pool is untouched."""
    a = _ledger(arena_torch_bytes=0)
    assert a.model_memory_correction() == (0, 0)
    assert a.report().ok


def test_the_old_handed_out_total_turned_a_bypass_into_a_wrong_correction():
    """What the previous implementation fed the ledger over the SAME window.

    `originals_released` becomes `copied - (-copied)` = twice the tier, so the boot is refused — with
    a message about a surviving alias, which is not what happened. And had it passed (the double
    fault where the originals are also undropped, which cancels exactly), the correction would be a
    full-size subtraction from a reading that never contained the bytes: `model_memory` under-bills
    the model, `available_memory` is over-stated, and the KV pool OOMs on the first forward.
    """
    a = _ledger(arena_torch_bytes=30 * GiB)
    assert a.originals_released == 60 * GiB
    assert not a.report().ok
    assert a.model_memory_correction()[0] == 30 * GiB


# =================================================================================================
# DEFECT 3 — a DERIVED device tier that is the whole KV budget must refuse before it pins anything
# =================================================================================================


def make_config(num_layers=24, **kw):
    mc = SimpleNamespace(
        num_layers=num_layers,
        num_experts=64,
        num_experts_per_tok=6,
        hidden_size=1024,
        moe_intermediate_size=512,
        is_moe=True,
        is_cca_hybrid=False,
        first_k_dense_replace=0,
        num_nextn_predict_layers=0,
        mtp_num_hidden_layers=0,
        quant=None,
    )
    base = {
        "model_config": mc,
        "tp_info": SimpleNamespace(size=1, rank=0),
        "dp_info": SimpleNamespace(dp_size=1, dp_rank=0),
        "enable_ep": False,
        "dtype": SimpleNamespace(itemsize=2),
        "device_index": 0,
        "weight_offload_device_gb": 0.0,
        "weight_offload_gb": 0.0,
    }
    base.update(kw)
    return SimpleNamespace(**base)


DERIVED_BUDGET = int(CARD_TOTAL_BYTES * MEMORY_RATIO)


def test_the_derived_tier_leaves_the_kv_pool_nothing():
    """The arithmetic behind the refusal, so it is not an opinion.

    `available = memory_ratio * old_free - model - state - draft - graph - snap`, and the tier is
    inside `model`. The greedy fill takes the tier up to the budget, so what is left of the ENTIRE
    budget for the KV pool, the dense weights, the recurrent state, the draft model and the graph
    buffers is under one layer's worth.
    """
    r = resolve_weight_plan(
        make_config(num_layers=96), device_budget_bytes=DERIVED_BUDGET, prefer_meta=False
    )
    assert r.enabled, "the fixture must not fit the card, or this proves nothing"
    per_layer = max(p.resident_bytes for p in r.plan.placements)
    assert DERIVED_BUDGET - r.device_bytes < per_layer


def test_a_derived_budget_that_produced_a_plan_is_refused():
    with pytest.raises(UnconfiguredDeviceTierError, match="DERIVED"):
        _refuse_derived_budget_that_leaves_no_kv(SimpleNamespace(summary_line=lambda: "x"), True)


def test_an_operator_chosen_budget_is_never_refused():
    """The knobs clamp an automatic decision (plan §6.2); they are not on/off switches, and a tier
    the operator named is a decision this module has no standing to overrule."""
    _refuse_derived_budget_that_leaves_no_kv(SimpleNamespace(summary_line=lambda: "x"), False)


def test_resolve_driver_refuses_a_derived_budget_on_a_stack_that_does_not_fit():
    _needs_arena_settings()
    with pytest.raises(UnconfiguredDeviceTierError) as exc:
        _resolve_driver(
            make_config(num_layers=96), DERIVED_BUDGET, budget_is_derived=True
        )
    # The remedy has to be in the message: the assert it replaces names --memory-ratio, which makes
    # the same budget SMALLER and the failure worse.
    assert "--weight-offload-device-gb" in str(exc.value)


def test_a_model_that_fits_never_reaches_the_refusal():
    """The refusal must not become an on/off switch. A fitting model resolves to an EMPTY plan, the
    driver is None, Stage A costs nothing, and the path still runs on every serve so it cannot rot."""
    _needs_arena_settings()
    assert (
        _resolve_driver(make_config(num_layers=24), DERIVED_BUDGET, budget_is_derived=True) is None
    )


def test_an_operator_chosen_budget_still_builds_a_driver_on_a_stack_that_does_not_fit(monkeypatch):
    """The whole point of the flag: naming a tier makes the same plan servable.

    `StageARuntime` is stubbed because constructing the real one reads the arena env and builds a
    `PinnedWeightArena`; what is under test is that `_resolve_driver` reaches it at all.
    """
    _needs_arena_settings()
    built = {}

    class _Stub:
        def __init__(self, resolution, **kw):
            built["resolution"] = resolution

    monkeypatch.setattr(bake_mod, "StageARuntime", _Stub)
    out = _resolve_driver(make_config(num_layers=96), 8 * GiB, budget_is_derived=False)
    assert isinstance(out, _Stub)
    assert built["resolution"].enabled


def test_engine_tells_the_session_whether_the_budget_was_derived():
    """Source-level: the runtime version needs a card and a checkpoint.

    Dropping the keyword restores the old behaviour exactly — the boot still fails, but only after
    the arena has pinned tens of GiB of unevictable host RAM and the checkpoint has been loaded and
    repacked, and the assert it fails on names four causes that do not include the tier.
    """
    tree = ast.parse(ENGINE_PY.read_text())
    call = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "begin"
        and isinstance(n.func.value, ast.Name)
        and n.func.value.id == "StageASession"
    )
    assert "budget_is_derived" in {k.arg for k in call.keywords}


def test_the_merge_gate_is_told_how_many_bytes_it_should_have_served():
    """`ArenaMemPool.assert_clean`'s zero checks are the ones a fallback count cannot see.

    A pool that stopped routing entirely reports `torch_fallbacks == 0` over a serve that put the
    whole host tier in VRAM — a clean-looking ledger. The expectation must be the COPIED bytes and
    not `plan.host_resident_bytes`: `served_bytes` counts the carved (aligned-up) region so it is
    `>= copied` exactly, while the plan figure only matches within the ledger's 64 MiB tolerance, and
    `assert_clean`'s comparison is a strict `<`.
    """
    seen = {}

    class _Pool:
        def assert_clean(self, *, expect_served_bytes=0):
            seen["expect"] = expect_served_bytes

    r = object.__new__(StageARuntime)
    r.pool = _Pool()
    r.outcome = SimpleNamespace(moved_bytes=30 * GiB)
    r.assert_arena_clean()
    assert seen["expect"] == 30 * GiB
