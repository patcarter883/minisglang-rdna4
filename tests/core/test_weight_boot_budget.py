"""Boot-order and VRAM-accounting regressions for the weight-offload Stage-A window.

Every test here pins a defect that was live in the tree, and each one is GPU-free by construction:
the arithmetic that decides the KV pool size is integer arithmetic, and the two call-site rules
(`Engine.__init__` must name a device budget; the KV budget must keep exactly five subtrahends) are
properties of the source, so they are asserted against the source rather than against a serve that
would need a card, a checkpoint and ten minutes to reproduce.

THE FOUR DEFECTS, AND WHY EACH IS A BOOT FAILURE RATHER THAN A PERFORMANCE ONE

1. `Engine.__init__` called `StageASession.begin(config, ...)` with no device budget. The resolver
   falls back to `config.weight_offload_device_gb`, a field that did not exist on `EngineConfig`, so
   `getattr(..., 0.0)` returned 0.0 and the plan read it as "the expert tier may occupy ZERO bytes of
   VRAM" — an ALL-HOST plan for every MoE model, including a 4.5 GiB expert stack on a 16 GB card.
   That is a ~10x decode regression on a serve nobody configured, with no flag to turn it off, and
   the boot log calls it a plan rather than an error.
2. Plan §5.3 requires a boot assertion tying the device tier to what the allocator really did, as the
   price of having NO sixth subtrahend in the KV budget. `WeightPlanResolution.assert_device_
   accounting` existed and was never called, and the ledger's stand-in check reduced algebraically to
   another check (see `test_device_tier_check_is_not_a_restatement_of_the_copied_check`).
3. `ArenaMemPool.assert_clean()` — its own docstring calls it the merge gate — was never called. A
   `hipMalloc` fallback is a DOUBLE error against the KV budget: the bytes are really in VRAM, and
   `model_memory_correction()` then removes those same bytes from the model term as host RAM.
4. The KV budget must keep exactly five subtrahends. A sixth (reserving for the device tier, which is
   already inside `device_used`) double-subtracts and, at a 16 GB tier, sizes the pool negative.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
from minisgl.weights.accounting import WeightArenaAccounting
from minisgl.weights.bake import StageASession, _reset_for_tests
from minisgl.weights.plan import resolve_weight_plan

GiB = 1 << 30
GB = 1_000_000_000
REPO = Path(__file__).resolve().parents[2]
ENGINE_PY = REPO / "python" / "minisgl" / "engine" / "engine.py"


# =================================================================================================
# Fixtures: a small MoE that fits a 16 GB card several times over
# =================================================================================================


def make_config(**kw):
    """A 4.5 GiB bf16 expert stack — comfortably resident on either card on this box."""
    mc = SimpleNamespace(
        num_layers=24,
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


CARD_TOTAL_BYTES = 16 * GiB  # gfx1201, both cards
MEMORY_RATIO = 0.9


# =================================================================================================
# DEFECT 1 — the device budget
# =================================================================================================


def test_zero_device_budget_puts_a_fitting_moe_entirely_on_host():
    """The failure mode, pinned so nobody re-derives it as 'a safe default'.

    This is what the engine was doing on every MoE serve: a stack that fits the card 3x over is
    planned 100% host-resident, `feasible` is True, and the only signal is a log line that reads like
    a plan rather than a mistake.
    """
    r = resolve_weight_plan(make_config(), device_budget_bytes=0, prefer_meta=False)
    assert r.enabled
    assert r.plan.num_device_layers == 0
    assert r.host_bytes_per_rank > 4 * GiB
    assert r.host_bytes_per_rank < CARD_TOTAL_BYTES, (
        "the fixture must be a stack that FITS the card, or this test proves nothing about the "
        "default being wrong"
    )


def test_card_sized_device_budget_makes_a_fitting_moe_a_no_op():
    """What the engine supplies now: `total_memory * memory_ratio`, a stable hardware constant."""
    r = resolve_weight_plan(
        make_config(),
        device_budget_bytes=int(CARD_TOTAL_BYTES * MEMORY_RATIO),
        prefer_meta=False,
    )
    assert not r.enabled, r.reason
    assert r.plan.is_empty
    assert r.host_bytes_per_rank == 0
    assert r.plan.num_device_layers == 24


def test_a_stack_that_does_not_fit_still_spills_to_host():
    """Over-granting the budget must not disable the feature — only make it inert when unneeded."""
    r = resolve_weight_plan(
        make_config(),
        device_budget_bytes=1 * GiB,
        prefer_meta=False,
    )
    assert r.enabled and r.plan.num_host_layers > 0
    assert r.plan.num_device_layers > 0, "1 GiB should still hold several 192 MiB layers"


def test_engine_config_declares_the_fields_the_resolver_reads_by_name():
    """`resolve_weight_plan` reads these through a defensive `getattr`, so their ABSENCE is silent.

    That silence is exactly how a missing field became an all-host plan. If someone deletes them the
    resolver keeps working and starts offloading everything again, so the contract is asserted here.
    """
    from dataclasses import fields

    try:
        from minisgl.engine.config import EngineConfig
    except Exception as exc:  # noqa: BLE001 — torch is unimportable outside the serve image
        pytest.skip(f"minisgl.engine.config needs a working torch: {exc}")

    names = {f.name: f for f in fields(EngineConfig)}
    assert "weight_offload_device_gb" in names
    assert "weight_offload_gb" in names
    assert names["weight_offload_device_gb"].default == 0.0
    assert names["weight_offload_gb"].default == 0.0


def test_resolver_refuses_to_be_reached_without_a_device_budget():
    """`_resolve_driver` must not silently accept "no budget" and plan all-host.

    The engine names the number; anything else is a bug, and a bug that is invisible at runtime
    unless it raises here."""
    from minisgl.weights.bake import _resolve_driver

    with pytest.raises(ValueError, match="without a device budget"):
        _resolve_driver(make_config(), None)


def test_engine_names_the_device_budget_at_the_begin_call_site():
    """Source-level, because the runtime version of this test needs a card and a checkpoint.

    `StageASession.begin` has `device_budget_bytes=None` as a signature default (tests inject a
    driver instead), so dropping the keyword at the engine call site is a one-character edit that
    type-checks, imports, boots, and streams the whole model over PCIe.
    """
    tree = ast.parse(ENGINE_PY.read_text())
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "begin"
        and isinstance(n.func.value, ast.Name)
        and n.func.value.id == "StageASession"
    ]
    assert len(calls) == 1, f"expected exactly one StageASession.begin call site, got {len(calls)}"
    kwargs = {k.arg for k in calls[0].keywords}
    assert "device_budget_bytes" in kwargs, (
        "Engine.__init__ must pass an explicit device budget; without it the resolver plans every "
        "MoE layer host-resident"
    )


# =================================================================================================
# DEFECT 4 — no sixth subtrahend, and nothing maps after the budget
# =================================================================================================


def _available_memory_assignment() -> ast.AST:
    tree = ast.parse(ENGINE_PY.read_text())
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "available_memory"
        ):
            return node.value
    raise AssertionError("Engine._determine_num_pages no longer assigns available_memory")


def test_kv_budget_keeps_exactly_five_subtrahends():
    """Plan §5.3: the device tier is inside `device_used`, so reserving for it again is a DOUBLE
    subtraction that at a 16 GB tier sizes the pool negative and trips 'Not enough memory for KV
    cache' with a cause that names five other things."""
    expr = _available_memory_assignment()
    subs = [
        n
        for n in ast.walk(expr)
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Sub)
    ]
    assert len(subs) == 5, (
        f"expected model/state/draft/graph/snap and nothing else, found {len(subs)} subtractions. "
        "A weight-offload term here is the forbidden sixth subtrahend."
    )
    names = {n.id for n in ast.walk(expr) if isinstance(n, ast.Name)} - {"int"}
    assert names == {
        "config",
        "old_free_memory",
        "model_memory",
        "state_memory",
        "draft_memory",
        "graph_memory",
        "snap_memory",
    }, names


def test_the_bake_is_sealed_before_the_kv_pool_is_sized():
    """Order is the whole VRAM-accounting design: bind inside the measured window, seal, then size.

    A mapping after the seal is invisible both to this sizing and to `Scheduler._prefill_budget_now`,
    and the operator is told to lower --memory-ratio for a problem that has nothing to do with it.
    """
    src = ENGINE_PY.read_text()
    order = [
        src.index("self._woff.attach()"),
        src.index("self.model.load_state_dict("),
        src.index("self.model.post_load()"),
        src.index("self._woff.note_loaded()"),
        src.index("self._woff.bind(self.model)"),
        src.index("self._woff.seal()"),
        src.index("self.num_pages = self._determine_num_pages("),
    ]
    assert order == sorted(order), (
        "the Stage-A window is attach -> load -> post_load -> bind -> seal -> size, and every "
        "boundary is load-bearing for either capacity-failure latency or KV accounting"
    )


# =================================================================================================
# DEFECT 2 — the device-tier check must be a measurement
# =================================================================================================


def test_device_tier_check_is_not_a_restatement_of_the_copied_check():
    """Without a measurement, `offloadable - copied` == `host - copied`, i.e. check 2 again.

    The plan guarantees `total = host + device`, so the derived form cannot disagree with check 2 for
    ANY input — which is why a plan whose byte model diverges from post_load's real output (the known
    MXFP4 E8M0 -> fp16 scale widening) sailed through both.
    """
    a = WeightArenaAccounting(
        host_bytes=30 * GiB, device_bytes=6 * GiB, offloadable_bytes=36 * GiB, copied_bytes=30 * GiB
    )
    assert not a.device_tier_is_measured
    derived = a.report()
    by_name = {c.name: c for c in derived.checks}
    # Perturb the copied bytes: BOTH checks move together, in lockstep, forever.
    a.copied_bytes = 26 * GiB
    moved = {c.name: c for c in a.report().checks}
    assert by_name["bake moved plan.host_resident_bytes"].ok
    assert by_name["device tier == plan.device_resident_bytes"].ok
    assert not moved["bake moved plan.host_resident_bytes"].ok
    assert not moved["device tier == plan.device_resident_bytes"].ok


def test_measured_device_tier_catches_a_plan_that_post_load_disagrees_with():
    """The failure the derived form is blind to: the bake copied EXACTLY what the plan priced, and
    the device tier is still not the size the plan says it is."""
    a = WeightArenaAccounting(
        host_bytes=30 * GiB,
        device_bytes=6 * GiB,
        offloadable_bytes=36 * GiB,
        copied_bytes=30 * GiB,
        observed_device_bytes=9 * GiB,  # post_load produced 3 GiB more than the byte model
    )
    assert a.device_tier_is_measured
    rep = a.report()
    by_name = {c.name: c for c in rep.checks}
    assert by_name["bake moved plan.host_resident_bytes"].ok, (
        "the copied-bytes check must still PASS, or this test would not isolate the device tier"
    )
    assert not by_name["device tier == plan.device_resident_bytes"].ok
    assert not rep.ok
    assert "MEASURED" in by_name["device tier == plan.device_resident_bytes"].detail


def test_reserved_correction_covers_the_arena_segment_pad_not_just_the_payload():
    """The pad is HOST memory; leaving it in `reserved` under-bills the HIP context.

    `model_memory`'s non-torch term is `max(0, device_used - (reserved - correction))`. Capping the
    correction at `copied` leaves the arena's segment rounding inside `reserved`, so a host-backed
    pad is subtracted from a device-memory total and the KV pool is handed bytes that do not exist.
    """
    pad = 512 << 20
    a = WeightArenaAccounting(
        host_bytes=30 * GiB,
        device_bytes=6 * GiB,
        offloadable_bytes=36 * GiB,
        copied_bytes=30 * GiB,
        arena_torch_bytes=30 * GiB,
    )
    # Arena rows added and originals dropped cancel in `allocated`; `reserved` keeps rows + pad.
    a.sample("pre_attach", free=100 * GiB, allocated=40 * GiB, reserved=40 * GiB)
    a.sample("post_attach", free=100 * GiB, allocated=40 * GiB, reserved=40 * GiB)
    a.sample("pre_bake", free=100 * GiB, allocated=40 * GiB, reserved=40 * GiB)
    a.sample(
        "post_bake",
        free=100 * GiB + 30 * GiB,
        allocated=40 * GiB,
        reserved=40 * GiB + 30 * GiB + pad,
    )
    assert a.torch_slack_bytes == pad
    assert a.reserved_correction == 30 * GiB + pad
    # And the ledger still agrees with itself: the whole arena footprint is accounted, nothing more.
    assert a.reserved_correction <= a.reserved_delta_bake


def test_reserved_correction_is_zero_when_the_arena_is_invisible_to_torch():
    """Every serve that does not offload must be byte-identical."""
    a = WeightArenaAccounting(host_bytes=0, device_bytes=0, offloadable_bytes=0, copied_bytes=0)
    a.sample("pre_attach", free=100 * GiB, allocated=40 * GiB, reserved=41 * GiB)
    a.sample("post_attach", free=100 * GiB, allocated=40 * GiB, reserved=41 * GiB)
    a.sample("pre_bake", free=100 * GiB, allocated=40 * GiB, reserved=41 * GiB)
    a.sample("post_bake", free=100 * GiB, allocated=40 * GiB, reserved=41 * GiB)
    assert a.model_memory_correction() == (0, 0)


def test_measured_device_tier_agreeing_keeps_the_report_green():
    a = WeightArenaAccounting(
        host_bytes=30 * GiB,
        device_bytes=6 * GiB,
        offloadable_bytes=36 * GiB,
        copied_bytes=30 * GiB,
        observed_device_bytes=6 * GiB,
    )
    by_name = {c.name: c for c in a.report().checks}
    assert by_name["device tier == plan.device_resident_bytes"].ok


# =================================================================================================
# DEFECTS 2 and 3 — the gates actually run at seal()
# =================================================================================================


class _Driver:
    """A `StageADriver` that records which gates the session invoked."""

    def __init__(self, *, host=30 * GiB, device=6 * GiB, observed=None, clean=True, resolution=None):
        self._host, self._device = host, device
        self._observed = device if observed is None else observed
        self._clean = clean
        self.resolution = resolution
        self.clean_calls = 0
        self.frozen = False

    def plan_bytes(self):
        return (self._host, self._device, self._host + self._device)

    def attach_host_arena(self):
        pass

    def bind(self, model):
        return SimpleNamespace(describe=lambda: "bound")

    def moved_bytes(self):
        return self._host

    def arena_torch_bytes(self):
        return 0

    def observed_device_bytes(self):
        return self._observed

    def assert_arena_clean(self):
        self.clean_calls += 1
        if not self._clean:
            raise RuntimeError(
                "WEIGHT OFFLOAD: torch MemPool did not stay inside the arena: fallbacks=3"
            )

    def freeze(self):
        self.frozen = True

    def describe(self):
        return "test driver"


def _probe(driver):
    """A healthy four-sample window: the arena costs no VRAM, and the drop returns the copied bytes.

    The ledger's checks are all DIFFERENCES between two samples, so a probe that returns a constant
    would make every one of them read zero and pass vacuously — which is the thing
    `StageASession._require_complete_ledger` exists to forbid.
    """
    resident = 40 * GiB
    moved = driver.moved_bytes()
    seq = iter(
        [
            (100 * GiB, resident, resident),  # pre_attach
            (100 * GiB, resident, resident),  # post_attach: free MUST NOT move
            (100 * GiB, resident, resident),  # pre_bake
            (100 * GiB + moved, resident - moved, resident),  # post_bake: originals dropped
        ]
    )
    return lambda: next(seq)


def _run(driver):
    _reset_for_tests()
    s = StageASession.begin(object(), driver=driver, probe=_probe(driver))
    s.attach()
    s.note_loaded()
    s.bind(object())
    s.seal()
    return s


def test_seal_asks_the_pool_whether_any_row_landed_in_vram():
    d = _Driver()
    _run(d)
    assert d.clean_calls == 1, "assert_clean is the merge gate; it has to be on the boot path"


def test_a_hipmalloc_fallback_aborts_the_boot_before_the_pool_is_sized():
    """Bytes budgeted as host-resident sitting in VRAM is a double error against the KV budget: the
    VRAM is gone AND `model_memory_correction` removes those same bytes from the model term."""
    d = _Driver(clean=False)
    with pytest.raises(RuntimeError, match="did not stay inside the arena"):
        _run(d)
    assert not d.frozen, "the arena must not be sealed once the boot is being aborted"


def test_seal_runs_the_plan_mandated_device_accounting_assertion():
    """Plan §5.3's `|observed - plan.device_bytes| < tol`. Without it, 'no sixth subtrahend' rests on
    an unchecked promise: the device tier is billed ONLY through `model_memory`, so nothing else in
    the engine would ever notice the plan being wrong about its size."""
    seen = {}

    class _Res:
        def assert_device_accounting(self, observed, *, tolerance_bytes):
            seen["observed"] = observed
            seen["tol"] = tolerance_bytes
            if abs(observed - 6 * GiB) > tolerance_bytes:
                raise AssertionError("weight-offload VRAM accounting mismatch")

    _run(_Driver(resolution=_Res()))
    assert seen["observed"] == 6 * GiB
    assert seen["tol"] > 0

    with pytest.raises(AssertionError, match="VRAM accounting mismatch"):
        _run(_Driver(observed=9 * GiB, resolution=_Res()))


def test_a_driver_without_the_new_hooks_still_seals():
    """The Protocol grew two methods; a session must not require them (tests inject minimal doubles,
    and a driver that genuinely cannot measure should degrade rather than refuse to boot)."""

    class _Old(_Driver):
        observed_device_bytes = None
        assert_arena_clean = None

    d = _Old()
    d.observed_device_bytes = None
    d.assert_arena_clean = None
    s = _run(d)
    assert s.phase.name == "SEALED"
    assert not s.accounting.device_tier_is_measured
