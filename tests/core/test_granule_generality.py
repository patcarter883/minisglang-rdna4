"""Generality guards: one mechanism for every format, and for dense as well as MoE.

The claim this module makes is that a NEW quant format or a NEW model family implements NOTHING here
— it declares its granule axis and the derivation does the rest. These tests are the adversarial
reading of that claim, and each one pins a defect that was live on 2026-09-02:

  * `spec_for_container` DEFAULTED an unanswered granule axis to the dense reading. A stacked
    container that never declared `_num_experts` (a new format, a third-party container, or the bare
    unquantized MoE tensor, whose docstring explicitly advertised this entry point) produced a
    perfectly self-consistent spec in which the whole 512-expert stack was ONE granule. Nothing
    raised; `expert_slice(c, 7)` just returned the entire stack.
  * dense hand-rolled a LOOK-ALIKE pair of methods instead of sharing `ExpertContainer`, so the
    surface a residency consumer writes against (`expert_slice`, `offload_refusal`) existed on a MoE
    container and raised `AttributeError` on a linear — "MoE and dense land together" satisfied on
    paper only.
  * `_GroupedFP8Experts.offload_refusal` re-read `os.environ` at placement time rather than
    reporting what `post_load` actually built, so a container built under a whole-stack knob and
    asked after it was cleared answered "offloadable".

GPU-free; tiny CPU shapes; every container goes through its REAL `post_load`.

    docker run --rm -v <worktree>:/engine --entrypoint bash minisgl-rdna4:lean-pytest \\
      -lc 'cd /engine && PYTHONPATH=/engine/python MINISGL_TAIL_HIP=0 python -m pytest \\
           tests/core/test_granule_generality.py -q'
"""

from __future__ import annotations

import inspect

import pytest
import torch
from minisgl.layers import moe as moe_mod
from minisgl.layers.base import BaseOP
from minisgl.layers.moe import (
    _GroupedAWQExperts,
    _GroupedCompressedTensorsExperts,
    _GroupedFP8Experts,
    _GroupedGPTQExperts,
    _GroupedMxFp4Experts,
    _GroupedNvFp4Experts,
    create_moe_quant_method,
)
from minisgl.quant.config import QuantConfig
from minisgl.weights.granule import (
    UNSET,
    ExpertContainer,
    GranuleError,
    declared_granule_count,
    derive_granule_spec,
    per_expert_tensors,
    spec_for_container,
)

E, N, K = 4, 32, 128


def _lds(monkeypatch):
    """Force every format onto its non-register-direct arm (pure-torch post_load, no fp8_wmma)."""
    monkeypatch.setattr(moe_mod.kernels, "MOE_W4A16", "0", raising=False)
    monkeypatch.setattr(moe_mod.kernels, "MOE_MXFP4_REGDIRECT", False, raising=False)
    monkeypatch.setattr(moe_mod.kernels, "MOE_W8A8_REGDIRECT", False, raising=False)


def _q(method: str, **kw) -> QuantConfig:
    base = {"method": method, "bits": 4, "group_size": 32, "sym": True}
    base.update(kw)
    return QuantConfig(**base)  # type: ignore[arg-type]


# Every shipped per-expert container class, constructed (NOT loaded — the declaration must be made in
# __init__, before any checkpoint exists, because `sizing.meta_gemm_spec` derives from a meta-built
# container that has never seen post_load).
def _all_containers():
    return {
        "gptq": _GroupedGPTQExperts(E, N, K, _q("gptq", sym=False)),
        "awq": _GroupedAWQExperts(E, N, K, _q("awq", sym=False)),
        "ct_int4_sym": _GroupedCompressedTensorsExperts(E, N, K, _q("compressed-tensors", sym=True)),
        "ct_int4_asym": _GroupedCompressedTensorsExperts(
            E, N, K, _q("compressed-tensors", sym=False)
        ),
        "mxfp4": _GroupedMxFp4Experts(E, N, K, _q("compressed-tensors", weight_type="float")),
        "nvfp4": _GroupedNvFp4Experts(
            E, N, K, _q("compressed-tensors", group_size=16, weight_type="float")
        ),
        "fp8": _GroupedFP8Experts(E, N, K),
    }


# =====================================================================================
# The granule axis is a DECLARATION, never a default
# =====================================================================================


class _NewFormatExperts(ExpertContainer, BaseOP):
    """A format nobody has written granule code for: novel buffer names, novel dtypes, a codebook.

    It declares ONE thing (`self._num_experts`) and inherits everything else. If this test needs an
    edit to `granule.py` to pass, the "a new format implements nothing" claim is false.
    """

    def __init__(self, num_experts: int, declare: bool = True):
        self.codebook = torch.arange(num_experts * 16, dtype=torch.uint8).reshape(num_experts, 16)
        self.wq = torch.arange(num_experts * 8 * 4, dtype=torch.int32).reshape(num_experts, 8, 4)
        self.exponent = torch.arange(num_experts * 8, dtype=torch.uint8).reshape(num_experts, 8)
        if declare:
            self._num_experts = num_experts

    def forward(self, *a, **k):  # pragma: no cover - storage container
        raise RuntimeError


def test_a_brand_new_format_needs_no_edit_to_the_granule_module():
    """The whole thesis: declare E, get a complete, correctly-addressed granule."""
    c = _NewFormatExperts(E)
    spec = spec_for_container(c)
    assert spec.num_experts == E
    assert {x.name for x in spec.components} == {"codebook", "wq", "exponent"}
    # Every component's per-expert row really is `t[e]` of its OWN stack (component-major).
    for e in range(E):
        views = c.expert_slice(e)
        for name, v in views.items():
            assert v.data_ptr() == getattr(c, name)[e].data_ptr()
    assert spec.granule_bytes == 16 + 8 * 4 * 4 + 8


def test_undeclared_granule_axis_raises_instead_of_silently_reading_dense():
    """THE fail-open. A stacked container that declares nothing used to come back as ONE granule.

    Non-vacuous by construction: the same container WITH an explicit count gives a materially
    different spec (E granules, 1/E the granule bytes), so the dense reading was never a harmless
    relabelling — it was a wrong answer that no downstream check could distinguish from a right one.
    """
    c = _NewFormatExperts(E, declare=False)
    with pytest.raises(GranuleError) as ei:
        spec_for_container(c)
    msg = str(ei.value)
    assert "_num_experts" in msg and "_granule_dense" in msg

    explicit = derive_granule_spec(c, E)
    dense = derive_granule_spec(c, None)  # what the old default silently produced
    assert explicit.num_granules == E and dense.num_granules == 1
    assert dense.granule_bytes == E * explicit.granule_bytes
    # and the dense reading hands back the WHOLE stack for what a caller thinks is one expert
    assert dense.expert_slice(c, 0)["wq"].shape == (E, 8, 4)
    assert explicit.expert_slice(c, 0)["wq"].shape == (8, 4)


def test_zero_expert_count_raises_from_the_FREE_FUNCTION_too():
    """The mixin already raised on `_num_experts == 0`; `spec_for_container` waved it through.

    Two accessor surfaces that disagree about what is safe is worse than one that is wrong, because
    the safe one is the one the tests use.
    """
    c = _NewFormatExperts(E)
    c._num_experts = 0
    with pytest.raises(GranuleError, match="expert count is unset"):
        c.granule_spec()
    with pytest.raises(GranuleError, match="expert count is unset"):
        spec_for_container(c)
    with pytest.raises(GranuleError, match="expert count is unset"):
        per_expert_tensors(c)


def test_bare_stacked_tensor_must_be_given_its_count():
    """The unquantized MoE container is a bare `torch.empty(E, out, in)` — there is nowhere to
    declare, so `per_expert_tensors(t)` cannot guess and must say so."""
    t = torch.arange(E * 8 * 4, dtype=torch.float32).reshape(E, 8, 4)
    with pytest.raises(GranuleError, match="explicitly"):
        per_expert_tensors(t)
    stacked = per_expert_tensors(t, E)
    assert list(stacked) == ["weight"] and stacked["weight"].shape == (E, 8, 4)
    assert derive_granule_spec(t, E).num_experts == E


def test_declared_granule_count_rejects_bool_and_nonpositive():
    """`True` is an `int` in Python. A container that set `_num_experts = True` (or 0, or -1) is a
    declaration bug, and inferring `E == 1` from it would classify every dim-1 buffer as an expert."""

    class _C:
        pass

    for bad in (True, 0, -1, None, "4"):
        c = _C()
        c._num_experts = bad  # type: ignore[attr-defined]
        with pytest.raises(GranuleError):
            declared_granule_count(c)


def test_every_shipped_container_declares_its_axis_at_construction():
    """A new container class that forgets `self._num_experts = num_experts` is caught HERE rather
    than at the first offload run. Constructed only — the declaration must precede post_load,
    because `sizing.meta_gemm_spec` derives from a never-loaded meta-device container."""
    for name, c in _all_containers().items():
        assert isinstance(c, ExpertContainer), f"{name}: not an ExpertContainer"
        assert declared_granule_count(c) == E, f"{name}: undeclared granule axis"


def test_every_method_built_container_declares_its_axis(monkeypatch):
    """Same guard, reached the way the engine reaches it — through `create_moe_quant_method`, so a
    format added to the selector but not to `_all_containers()` above is still covered. The
    unquantized method returns a BARE tensor, which legitimately cannot declare."""
    _lds(monkeypatch)
    quants = [
        None,
        _q("gptq", sym=False),
        _q("awq", sym=False),
        _q("compressed-tensors", sym=True),
        _q("compressed-tensors", weight_type="float"),
    ]
    for quant in quants:
        for fp8 in (False, True):
            if fp8 and quant is not None:
                continue
            method = create_moe_quant_method(quant, fp8_experts=fp8)
            c = method.create_experts(E, N, K)
            if isinstance(c, torch.Tensor):
                assert derive_granule_spec(c, E).num_experts == E
                continue
            assert declared_granule_count(c) == E, f"{type(c).__name__}: undeclared granule axis"


# =====================================================================================
# Dense and MoE share ONE surface (repo rule: MoE and dense land together)
# =====================================================================================

_SURFACE = ("granule_spec", "per_expert_tensors", "expert_slice", "offload_refusal")


def _dense_linear(method):
    from minisgl.layers.linear import LinearReplicated

    lin = LinearReplicated(input_size=256, output_size=64, has_bias=True, quant_method=method)
    g = torch.Generator().manual_seed(3)
    for _, v in list(vars(lin).items()):
        if isinstance(v, torch.Tensor):
            if v.dtype in (torch.int32, torch.uint8):
                v.copy_(torch.randint(0, 127, v.shape, generator=g, dtype=torch.int64).to(v.dtype))
            else:
                v.copy_(torch.randn(v.shape, generator=g).to(v.dtype))
    lin.post_load()
    return lin


def test_dense_linear_presents_the_same_granule_surface_as_a_moe_container():
    """A consumer written against `ExpertContainer` must not discover, months later, that the dense
    half of the model answers `AttributeError`. Both arms are the SAME class, with the same
    signatures — not two look-alikes."""
    from minisgl.quant.method import UnquantizedLinearMethod

    lin = _dense_linear(UnquantizedLinearMethod())
    moe = _NewFormatExperts(E)
    assert isinstance(lin, ExpertContainer)
    for name in _SURFACE:
        assert hasattr(lin, name), f"dense linear is missing {name}()"
        assert hasattr(moe, name)
        # Same implementation, not a look-alike: identical underlying function object.
        assert getattr(type(lin), name) is getattr(type(moe), name), (
            f"{name} has diverged between the dense and MoE arms"
        )
    # The signatures accept the same call shapes, so generic code needs no isinstance branch.
    for name in ("granule_spec", "per_expert_tensors"):
        params = list(inspect.signature(getattr(lin, name)).parameters)
        assert params[0] == "num_experts"


def test_dense_linear_answers_offload_refusal_and_expert_slice():
    from minisgl.quant.method import Fp8W8A8LinearMethod

    q = QuantConfig(method="compressed-tensors", bits=8, group_size=0, sym=True, weight_type="float")
    lin = _dense_linear(Fp8W8A8LinearMethod(q))
    assert lin.offload_refusal() is None
    assert declared_granule_count(lin) is None  # declared dense, not "undeclared"
    spec = lin.granule_spec()
    assert spec.num_experts is None and spec.num_granules == 1
    names = set(spec.components and [c.name for c in spec.components])
    assert {"_w_op", "_scales_op", "bias"} == names
    # The bias travels: a moved weight with a left-behind bias is the same silent bug as a left
    # behind scale.
    granule = lin.expert_slice(0)
    assert set(granule) == names
    assert granule["bias"].data_ptr() == lin.bias.data_ptr()


def test_dense_expert_slice_rejects_a_nonzero_granule_index():
    """A caller asking a dense container for granule 3 believes it is stacked. Handing the whole
    container back is how a mis-declared axis stays invisible."""
    from minisgl.quant.method import UnquantizedLinearMethod

    lin = _dense_linear(UnquantizedLinearMethod())
    with pytest.raises(IndexError, match="DENSE"):
        lin.expert_slice(3)


def test_dense_and_moe_use_the_same_derivation_for_the_same_bytes():
    """Non-vacuity for the shared mechanism: a dense container and a 1-expert MoE container holding
    the SAME tensors must agree on the granule bytes, because it is literally one code path."""
    c = _NewFormatExperts(E)
    stacked = derive_granule_spec(c, E)
    whole = derive_granule_spec(c, None)
    assert whole.total_bytes == stacked.total_bytes


# =====================================================================================
# Refusals are recorded, not re-read from the environment
# =====================================================================================


def _load_fp8(monkeypatch):
    c = _GroupedFP8Experts(E, N, K)
    g = torch.Generator().manual_seed(5)
    c.weight.copy_(torch.randn(c.weight.shape, generator=g).to(torch.float8_e4m3fn))
    c.weight_scale.copy_(torch.rand(c.weight_scale.shape, generator=g))
    c.post_load()
    return c


def test_offload_refusal_reports_what_post_load_BUILT_not_the_current_env(monkeypatch):
    """The silent direction: built under the whole-stack knob, asked after it was cleared.

    `post_load` snapshots the knob to decide whether to repack, so the container's buffers are the
    whole-stack ones for the rest of its life. Re-reading `os.environ` at placement time answered
    "offloadable", the layer went host-resident, and the forward then streamed the entire (E,N,K)
    stack over PCIe every step — which presents as "the offload mechanism does not work".
    """
    _lds(monkeypatch)
    monkeypatch.setenv("MINISGL_ZAYA_OLDMOE", "1")
    c = _load_fp8(monkeypatch)
    assert c.offload_refusal() is not None
    assert isinstance(getattr(c, "weight", None), torch.Tensor)  # the whole-stack buffers are here

    monkeypatch.delenv("MINISGL_ZAYA_OLDMOE")
    why = c.offload_refusal()
    assert why is not None and "MINISGL_ZAYA_OLDMOE" in why
    with pytest.raises(GranuleError, match="MINISGL_ZAYA_OLDMOE"):
        spec_for_container(c, E, for_offload=True)


def test_offload_refusal_is_not_invented_by_an_env_set_after_the_build(monkeypatch):
    """The other direction. A container built on the fast path keeps its fast-path buffers; a knob
    flipped afterwards changes nothing about it, and refusing would strand a perfectly offloadable
    container (and send someone hunting a phantom)."""
    _lds(monkeypatch)
    monkeypatch.delenv("MINISGL_ZAYA_OLDMOE", raising=False)
    monkeypatch.delenv("MINISGL_ZAYA_W8A16", raising=False)
    c = _load_fp8(monkeypatch)
    assert c.offload_refusal() is None
    monkeypatch.setenv("MINISGL_ZAYA_W8A16", "1")
    assert c.offload_refusal() is None
    spec_for_container(c, E, for_offload=True)  # must not raise


def test_w8a16_knob_is_also_recorded(monkeypatch):
    _lds(monkeypatch)
    monkeypatch.setenv("MINISGL_ZAYA_W8A16", "1")
    c = _load_fp8(monkeypatch)
    monkeypatch.delenv("MINISGL_ZAYA_W8A16")
    why = c.offload_refusal()
    assert why is not None and "MINISGL_ZAYA_W8A16" in why


def test_a_container_that_never_declares_a_refusal_is_offloadable():
    """Non-vacuity: `offload_refusal` defaults to None on the mixin, so the refusal above is a real
    decision and not "everything refuses"."""
    c = _NewFormatExperts(E)
    assert c.offload_refusal() is None
    spec_for_container(c, for_offload=True)


# =====================================================================================
# The sentinel itself
# =====================================================================================


def test_unset_is_distinguishable_from_the_dense_answer():
    """`None` is a legal answer (dense). If the sentinel were `None`, "unanswered" and "dense" would
    be the same value and the fail-closed rule above could not exist."""
    assert UNSET is not None
    c = _NewFormatExperts(E, declare=False)
    assert spec_for_container(c, None).num_experts is None  # explicit dense: accepted
    with pytest.raises(GranuleError):
        spec_for_container(c)  # unanswered: refused
