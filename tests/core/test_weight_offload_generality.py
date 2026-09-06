"""Generality and repo-rule regressions for the Stage-A landing (M1-A).

Three separate ways the load-time bake could stop being general, each pinned here:

  1. THE PLAN MUST BE RESOLVED OFF THE BUILT MODEL. `plan.observed_planned_layers` reads the layer
     set, this rank's EP/TP sharding and the per-expert byte count off live objects through the one
     format-agnostic granule walker, so a new model family or quant format implements NOTHING. The
     config-only fallback (`plan.build_planned_layers`) TRANSCRIBES seven builder files and nine
     container `__init__`s, and it is already wrong for `models/utils.py`'s `MoEMLP`, which builds
     its `MoELayer` with no `quant=` at all: bf16 experts sized as int4, arena under-reserved ~4x,
     in the direction that makes an infeasible plan look FEASIBLE. `Engine.__init__` therefore has
     to hand `StageASession.begin` the meta-built model, and `begin` has to forward it.
  2. THE OPERATOR MUST BE TOLD WHEN THE KNOBS DID NOTHING. An empty plan is the correct outcome for
     a model that fits — but it is also what a DENSE model produces, because this build binds
     `MoELayer` only. Silently accepting `--weight-offload-gb` and then not offloading anything
     turns the repo's outstanding "MoE and dense land together" gap into an unexplained OOM.
  3. GiB IS NOT GB. `weight_offload_device_gb` has two readers — `Engine` and
     `plan.resolve_weight_plan`'s own fallback — and they must agree, or the granted tier is 7.4%
     smaller than every log line says it is.

GPU-FREE and, for the parts that can be, torch-free.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from minisgl.weights import bake as bake_mod  # noqa: E402
from minisgl.weights import plan as plan_mod  # noqa: E402

GiB = 1 << 30


class _FakeResolution:
    """The minimum `WeightPlanResolution` surface `_resolve_driver` touches."""

    def __init__(self, *, enabled=False, diagnostics=None):
        self.enabled = enabled
        self.diagnostics = diagnostics if diagnostics is not None else {}
        self.local_ranks = 1
        self.agreement_calls = 0

    def assert_rank_agreement(self, group=None, *, gather=None):
        self.agreement_calls += 1
        return ()


class _Config:
    device_index = 0
    tp_info = None
    weight_offload_device_gb = 0.0
    weight_offload_gb = 0.0


# =================================================================================================
# 1. The model reaches the resolver
# =================================================================================================


class TestPlanIsResolvedOffTheBuiltModel:
    def test_begin_forwards_the_model_to_resolve_weight_plan(self, monkeypatch):
        seen = {}

        def _fake_resolve(config, **kw):
            seen.update(kw)
            return _FakeResolution(enabled=False)

        monkeypatch.setattr(plan_mod, "resolve_weight_plan", _fake_resolve)
        sentinel = object()
        bake_mod.StageASession.begin(_Config(), device_budget_bytes=8 * GiB, model=sentinel)
        assert "model" in seen, (
            "StageASession.begin did not forward `model=` to resolve_weight_plan. Without it the "
            "resolver falls back to build_planned_layers, which transcribes per-family and "
            "per-format tables — so a new model family or quant format would have to edit them."
        )
        assert seen["model"] is sentinel

    def test_engine_passes_the_meta_built_model_and_does_so_after_the_build(self):
        """Source-level, because constructing an Engine needs a card and a checkpoint.

        Two properties, and the ORDER is half of the point: `begin(...)` must receive
        `model=self.model`, and it must be called AFTER `create_model` rather than before. Resolving
        before the meta build is what forced the config-transcription path in the first place; a
        meta build allocates zero device bytes, so nothing about the fail-fast ordering is lost by
        moving it, and the capacity abort still lands at `attach()` before `load_state_dict`.
        """
        src = Path(plan_mod.__file__).parents[1] / "engine" / "engine.py"
        tree = ast.parse(src.read_text())
        init = next(
            n
            for cls in tree.body
            if isinstance(cls, ast.ClassDef) and cls.name == "Engine"
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "__init__"
        )
        begin_call = None
        create_model_line = None
        for node in ast.walk(init):
            if isinstance(node, ast.Call):
                fn = node.func
                if isinstance(fn, ast.Attribute) and fn.attr == "begin":
                    begin_call = node
                if isinstance(fn, ast.Name) and fn.id == "create_model":
                    create_model_line = node.lineno
        assert begin_call is not None, "Engine.__init__ no longer calls StageASession.begin"
        kwargs = {k.arg for k in begin_call.keywords}
        assert "model" in kwargs, (
            "Engine.__init__ calls StageASession.begin without model=. The plan then comes from "
            "config transcription, which is already wrong for models/utils.py's unquantized MoEMLP."
        )
        assert create_model_line is not None
        assert begin_call.lineno > create_model_line, (
            "StageASession.begin must run AFTER the meta build so it can be given the model; the "
            "meta build allocates no device memory, so nothing is lost by ordering it that way."
        )


# =================================================================================================
# 2. An ignored operator request is reported, not swallowed
# =================================================================================================


class TestEmptyPlanWithAnExplicitRequest:
    def _lines(self, config, diagnostics):
        out = []
        bake_mod._warn_if_offload_was_requested(
            config, _FakeResolution(enabled=False, diagnostics=diagnostics), out.append
        )
        return out

    def test_silent_when_the_operator_asked_for_nothing(self):
        assert self._lines(_Config(), {"n_offloadable_layers": 0}) == []

    def test_dense_model_request_names_the_MoE_only_limitation(self):
        cfg = _Config()
        cfg.weight_offload_gb = 40.0
        (line,) = self._lines(cfg, {"n_offloadable_layers": 0})
        assert "EMPTY" in line
        # The operator has to be able to tell "your model fits" from "this build cannot offload
        # your model at all", because only the second is a reason to stop tuning the flag.
        assert "MoELayer" in line and "DENSE" in line

    def test_fitting_model_request_says_so_differently(self):
        cfg = _Config()
        cfg.weight_offload_device_gb = 12.0
        (line,) = self._lines(cfg, {"n_offloadable_layers": 48})
        assert "fits the device budget" in line
        assert "DENSE" not in line

    def test_skips_are_reported_because_a_refusal_is_the_likeliest_cause(self):
        cfg = _Config()
        cfg.weight_offload_gb = 40.0
        (line,) = self._lines(
            cfg,
            {
                "n_offloadable_layers": 0,
                "skipped": ("model.layers.3 (container refuses host residency: OLDMOE)",),
            },
        )
        assert "OLDMOE" in line

    def test_it_is_a_log_line_and_never_a_raise(self):
        """A knob that had no effect must not become an outage: both flags are documented as clamps
        on an automatic decision, not as enable switches."""
        cfg = _Config()
        cfg.weight_offload_gb = 40.0
        bake_mod._warn_if_offload_was_requested(
            cfg, _FakeResolution(enabled=False, diagnostics={}), lambda _m: None
        )

    def test_resolver_emits_it_on_the_empty_plan_path(self, monkeypatch):
        monkeypatch.setattr(
            plan_mod,
            "resolve_weight_plan",
            lambda config, **kw: _FakeResolution(
                enabled=False, diagnostics={"n_offloadable_layers": 0}
            ),
        )
        cfg = _Config()
        cfg.weight_offload_gb = 40.0
        out = []
        drv = bake_mod._resolve_driver(cfg, 8 * GiB, None, log=out.append)
        assert drv is None
        assert out and "EMPTY" in out[0]


# =================================================================================================
# 3. One unit for one field
# =================================================================================================


class TestDeviceBudgetUnit:
    def test_plan_reads_the_field_as_GiB(self):
        assert plan_mod.GIB_PER_UNIT == 1 << 30

    def test_engine_converts_with_the_same_constant(self):
        """`weight_offload_device_gb` is read by BOTH `Engine._weight_offload_device_budget` and
        `resolve_weight_plan`'s own fallback, and `WeightPlanResolution.summary_line()` prints the
        granted tier back in GiB. Converting with 1e9 in one of the two under-grants the tier by
        7.4% while every log line echoes the request as honoured — enough to push a layer across
        the greedy fill boundary into host residency, silently.

        Read from SOURCE rather than by importing `Engine`: `minisgl.engine.engine` pulls in
        `layers/_tail_hip`, which hard-requires a built `tail_hip` .so matching the running image.
        Making a units regression depend on that would mean this guard is skipped exactly on the
        boxes where the kernels are mid-rebuild."""
        engine_src = Path(plan_mod.__file__).parents[1] / "engine" / "engine.py"
        tree = ast.parse(engine_src.read_text())
        fn = next(
            n
            for cls in tree.body
            if isinstance(cls, ast.ClassDef) and cls.name == "Engine"
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "_weight_offload_device_budget"
        )
        src = ast.unparse(fn)
        assert "GIB_PER_UNIT" in src, (
            "the engine must convert --weight-offload-device-gb with plan.GIB_PER_UNIT, the same "
            "constant resolve_weight_plan's fallback uses"
        )
        assert "1_000_000_000" not in src and "1e9" not in src
