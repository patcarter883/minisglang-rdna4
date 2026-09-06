"""Granule descriptor: what "one expert" is, for EVERY expert-weight format.

GPU-FREE by construction. Every container here is built on CPU with tiny shapes and run through its
REAL `post_load`, so the descriptor is checked against the buffers the engine actually serves rather
than against a hand-written table (which is the thing this module exists to avoid). The register-direct
arms (`_w_rep`) need the compiled `fp8_wmma` repack and therefore a device: `test_regdirect_shapes`
below synthesizes their post-load buffer set (the descriptor logic is identical either way), and
`test_granule_regdirect_gpu.py` runs the REAL repacks under `@pytest.mark.gpu`.

Nothing in this file touches a device.

    docker run --rm --network none -v <worktree>:/engine:ro --entrypoint bash minisgl-rdna4:lean \\
      -lc 'PYTHONPATH=/engine/python MINISGL_TAIL_HIP=0 pytest /engine/tests/core -k granule'

`MINISGL_TAIL_HIP=0` is only needed on an image whose baked `tail_hip` predates
`layers/_tail_hip.py`'s current re-export list — importing `minisgl.layers` hard-fails there, and
these tests import `layers.moe`. It is not a property of the granule code.
"""

from __future__ import annotations

import pytest
import torch
from minisgl.layers import moe as moe_mod
from minisgl.layers.moe import (
    _GroupedAWQExperts,
    _GroupedCompressedTensorsExperts,
    _GroupedFP8Experts,
    _GroupedGPTQExperts,
    _GroupedMxFp4Experts,
    _GroupedNvFp4Experts,
    _GroupedRXFExperts,
)
from minisgl.quant.config import QuantConfig
from minisgl.weights.granule import (
    ExpertContainer,
    GranuleError,
    derive_granule_spec,
    per_expert_tensors,
    plan_component_major,
    spec_for_container,
)

E, N, K = 4, 32, 128


def _fill(t: torch.Tensor, seed: int, kind: str = "random") -> torch.Tensor:
    """Deterministic per-expert-DISTINCT contents, so an expert-invariance false positive is caught.

    `kind` encodes the two checkpoint invariants the real converters assert on, because filling those
    buffers with uniform noise makes the container refuse to load rather than exercise the walk:
      * `nibble14` — GPTQ `qzeros` stores `zero_point - 1`, so every nibble must be <= 14
        (`kernels.gptq_to_op_layout` asserts `zero+1 <= 15`).
      * `e8m0` — MXFP4's `weight_scale` is a raw E8M0 exponent; values far from 127 overflow the fp16
        group-scale store and make `post_load` log (which needs TP info that a unit test has not set).
    """
    g = torch.Generator().manual_seed(seed)
    if kind == "nibble14":
        nib = torch.randint(0, 15, t.shape + (8,), generator=g, dtype=torch.int64)
        packed = torch.zeros(t.shape, dtype=torch.int64)
        for j in range(8):
            packed |= nib[..., j] << (4 * j)
        t.copy_((packed - (1 << 32) * (packed >> 31)).to(torch.int32))
    elif kind == "e8m0":
        t.copy_(torch.randint(120, 132, t.shape, generator=g, dtype=torch.int64).to(t.dtype))
    elif t.dtype in (torch.int32, torch.uint8):
        hi = 2**31 - 1 if t.dtype == torch.int32 else 255
        t.copy_(torch.randint(0, hi, t.shape, generator=g, dtype=torch.int64).to(t.dtype))
    elif t.dtype == torch.float8_e4m3fn:
        t.copy_(torch.randn(t.shape, generator=g).to(torch.float8_e4m3fn))
    else:
        t.copy_(torch.randn(t.shape, generator=g).to(t.dtype))
    return t


def _fill_kind(container, name: str) -> str:
    if isinstance(container, _GroupedGPTQExperts) and name == "qzeros":
        return "nibble14"
    if isinstance(container, _GroupedMxFp4Experts) and name == "weight_scale":
        return "e8m0"
    return "random"


def _load(container, seed: int = 0):
    """Fill every declared checkpoint buffer, then run the container's real post_load."""
    for i, (name, v) in enumerate(list(vars(container).items())):
        if isinstance(v, torch.Tensor):
            _fill(v, seed + i, _fill_kind(container, name))
    container.post_load()
    return container


def _lds_knobs(monkeypatch):
    """Force every format onto its LDS (non-register-direct) arm — the arms whose post_load is pure
    torch. The regdirect arms need the compiled fp8_wmma repack; see test_regdirect_shapes."""
    monkeypatch.setattr(moe_mod.kernels, "MOE_W4A16", "0", raising=False)
    monkeypatch.setattr(moe_mod.kernels, "RXF_REGDIRECT", False, raising=False)
    monkeypatch.setattr(moe_mod.kernels, "MOE_MXFP4_REGDIRECT", False, raising=False)
    monkeypatch.setattr(moe_mod.kernels, "MOE_W8A8_REGDIRECT", False, raising=False)


def _q(method: str, **kw) -> QuantConfig:
    base = {"method": method, "bits": 4, "group_size": 32, "sym": True}
    base.update(kw)
    return QuantConfig(**base)  # type: ignore[arg-type]


# All eight formats, as (id, factory). The factory returns a POST-LOAD container.
def _containers(monkeypatch):
    _lds_knobs(monkeypatch)
    return {
        "gptq": lambda: _load(_GroupedGPTQExperts(E, N, K, _q("gptq", group_size=32, sym=False))),
        "awq": lambda: _load(_GroupedAWQExperts(E, N, K, _q("awq", group_size=32, sym=False))),
        "rxf": lambda: _load(_GroupedRXFExperts(E, N, K, _q("rxf", group_size=32))),
        "ct_int4_sym": lambda: _load(
            _GroupedCompressedTensorsExperts(E, N, K, _q("compressed-tensors", sym=True))
        ),
        "ct_int4_asym": lambda: _load(
            _GroupedCompressedTensorsExperts(E, N, K, _q("compressed-tensors", sym=False))
        ),
        "mxfp4": lambda: _load(
            _GroupedMxFp4Experts(E, N, K, _q("compressed-tensors", weight_type="float"))
        ),
        "nvfp4": lambda: _load(
            _GroupedNvFp4Experts(E, N, K, _q("compressed-tensors", group_size=16, weight_type="float"))
        ),
        "fp8": lambda: _load(_GroupedFP8Experts(E, N, K)),
        # The unquantized MoE container is a BARE stacked tensor — `_UnquantizedMoEMethod.
        # create_experts` returns `torch.empty(E, out, in)`, there is no object to hang attributes on.
        "unquantized": lambda: _fill(torch.empty(E, N, K), 99),
    }


ALL_FORMATS = [
    "gptq", "awq", "rxf", "ct_int4_sym", "ct_int4_asym", "mxfp4", "nvfp4", "fp8", "unquantized",
]


# =====================================================================================
# The core contract, asserted for every format
# =====================================================================================


@pytest.mark.parametrize("fmt", ALL_FORMATS)
def test_every_format_yields_a_granule(fmt, monkeypatch):
    c = _containers(monkeypatch)[fmt]()
    spec = spec_for_container(c, E)

    assert spec.components, f"{fmt}: no per-expert components found — the walk missed the weights"
    assert spec.num_experts == E
    # DIM 0 IS E. This is the whole classification rule; the kernels' `base + e*row` arithmetic is
    # only true because of it.
    for comp in spec.components:
        assert comp.stacked_shape[0] == E, f"{fmt}.{comp.name}: dim 0 is not E"
        assert tuple(comp.shape) == tuple(comp.stacked_shape[1:])
    assert spec.granule_bytes == sum(c_.nbytes for c_ in spec.components)
    assert spec.stacked_bytes == spec.granule_bytes * E


@pytest.mark.parametrize("fmt", ALL_FORMATS)
def test_scales_and_zeros_travel_with_the_weight(fmt, monkeypatch):
    """The silent-corruption guard: every SURVIVING per-expert buffer is in the granule.

    Derived from the live container, so this asserts equality with reality rather than with a
    remembered list: whatever `post_load` left behind that carries an expert axis must be a
    component (or be PROVEN expert-invariant and recorded as such)."""
    c = _containers(monkeypatch)[fmt]()
    spec = spec_for_container(c, E)
    named = {n for comp in spec.components for n in comp.names}
    replicated = {n for r in spec.replicated for n in (r.name,) + r.aliases}

    live = {
        n: t for n, t in vars(c).items() if isinstance(t, torch.Tensor)
    } if not isinstance(c, torch.Tensor) else {"weight": c}
    for name, t in live.items():
        assert t.shape[0] == E, f"{fmt}.{name}: unexpected non-expert tensor"
        assert name in named or name in replicated, (
            f"{fmt}.{name} survived post_load with an expert axis but is in neither the granule nor "
            f"the replicated set — it would be left behind when the expert moves"
        )


@pytest.mark.parametrize("fmt", ALL_FORMATS)
def test_expert_slice_is_component_major(fmt, monkeypatch):
    """`expert_slice(e)[c]` must be the view at `base(c) + e*row_bytes(c)` of c's OWN stack.

    If placement were frame-major the stride would be `frame_bytes` and this would fail for every
    component but the first."""
    c = _containers(monkeypatch)[fmt]()
    spec = spec_for_container(c, E)
    stacks = spec.stacked_tensors(c)
    for e in range(E):
        views = spec.expert_slice(c, e)
        for comp in spec.components:
            base = stacks[comp.name]
            got, want = views[comp.name], base[e]
            assert got.data_ptr() == want.data_ptr()
            assert got.data_ptr() == base.data_ptr() + e * comp.row_bytes
            assert torch.equal(got.reshape(-1).view(torch.uint8), want.reshape(-1).view(torch.uint8))


@pytest.mark.parametrize("fmt", ALL_FORMATS)
def test_per_expert_tensors_accessor(fmt, monkeypatch):
    """Both surfaces agree: the method on the base class and the free function."""
    c = _containers(monkeypatch)[fmt]()
    free = per_expert_tensors(c, E)
    assert set(free) == {comp.name for comp in spec_for_container(c, E).components}
    if isinstance(c, ExpertContainer):
        # The container records its own expert count, so the accessor needs no argument.
        assert c._num_experts == E
        meth = c.per_expert_tensors()
        assert set(meth) == set(free)
        assert all(meth[k].data_ptr() == free[k].data_ptr() for k in free)


@pytest.mark.parametrize("fmt", ALL_FORMATS)
def test_fingerprint_is_stable_and_discriminating(fmt, monkeypatch):
    c = _containers(monkeypatch)[fmt]()
    s1 = spec_for_container(c, E)
    s2 = spec_for_container(c, E)
    assert s1.fingerprint() == s2.fingerprint()
    # Dropping a component (the exact desync this guards) must change the fingerprint.
    if len(s1.components) > 1:
        from dataclasses import replace

        s3 = replace(s1, components=s1.components[:-1])
        assert s3.fingerprint() != s1.fingerprint()


# =====================================================================================
# Format-specific facts the descriptor must get right
# =====================================================================================


def test_symmetric_ct_zeros_are_declared_and_verified_expert_invariant(monkeypatch):
    """Symmetric compressed-tensors `_zeros_op` is E identical copies of 0x88 — ~3 % of w13 that
    must NOT be paid per expert.

    DECLARED by the post_load branch that filled it (so the exemption is a function of `quant.sym`,
    which every TP rank shares) and then VERIFIED bitwise by the walker (so a stale declaration
    raises rather than dropping a real buffer). Both halves are asserted."""
    c = _containers(monkeypatch)["ct_int4_sym"]()
    assert "_zeros_op" in getattr(c, "_residency_shared", ()), (
        "post_load must DECLARE the constant zeros; a content-only exemption is rank-divergent"
    )
    spec = spec_for_container(c, E)
    rep = {r.name: r for r in spec.replicated}
    assert "_zeros_op" in rep, f"symmetric CT zeros not exempted; replicated={list(rep)}"
    assert rep["_zeros_op"].reason == "expert-invariant"
    assert "_zeros_op" not in {comp.name for comp in spec.components}


def test_asymmetric_ct_zeros_stay_in_the_granule(monkeypatch):
    """The non-vacuous arm: a real per-group zero-point is per-expert and MUST travel."""
    c = _containers(monkeypatch)["ct_int4_asym"]()
    spec = spec_for_container(c, E)
    assert "_zeros_op" in {comp.name for comp in spec.components}
    assert not [r for r in spec.replicated if r.name == "_zeros_op"]


def test_invariance_detection_needs_at_least_two_rows():
    """E==1 cannot PROVE invariance. A false positive here drops a real per-expert scale."""
    t = torch.zeros(1, 8, 8, dtype=torch.int32)
    spec = derive_granule_spec(t, 1)
    assert [c.name for c in spec.components] == ["weight"]
    assert not spec.replicated


def test_fp8_uint8_view_is_one_component_not_two(monkeypatch):
    """`_w_op = weight.contiguous().view(torch.uint8)` is the SAME BYTES under another dtype.

    Keying dedupe on (shape, stride, dtype) would call these two components and double the granule;
    a later copy-based rebind would then de-alias them so `dequant()` and the kernel read different
    memory. Byte-range keying collapses them into one component with two names."""
    monkeypatch.setenv("MINISGL_ZAYA_OLDMOE", "1")  # keeps `weight` alive alongside `_w_op`
    _lds_knobs(monkeypatch)
    c = _load(_GroupedFP8Experts(E, N, K))
    assert isinstance(getattr(c, "weight", None), torch.Tensor), "OLDMOE should keep `weight`"
    spec = spec_for_container(c, E)
    names = {comp.name: comp for comp in spec.components}
    all_names = {n for comp in spec.components for n in comp.names}
    assert {"weight", "_w_op"} <= all_names
    assert len(names) == len(spec.components)
    # One component covering those bytes, not two.
    weight_bytes = E * N * K
    hit = [comp for comp in spec.components if "weight" in comp.names and "_w_op" in comp.names]
    assert len(hit) == 1, f"weight/_w_op were not merged: {[c_.names for c_ in spec.components]}"
    assert hit[0].nbytes * E == weight_bytes


def test_fp8_container_refuses_offload_under_whole_stack_knobs(monkeypatch):
    monkeypatch.setenv("MINISGL_ZAYA_OLDMOE", "1")
    _lds_knobs(monkeypatch)
    c = _load(_GroupedFP8Experts(E, N, K))
    assert c.offload_refusal() is not None
    spec_for_container(c, E)  # descriptor still derivable — only OFFLOAD is refused
    with pytest.raises(GranuleError, match="MINISGL_ZAYA_OLDMOE"):
        spec_for_container(c, E, for_offload=True)


def test_regdirect_shapes(monkeypatch):
    """The `_w_rep` register-direct arms, synthesized (their repack needs compiled fp8_wmma).

    Covers the knob matrix the LDS tests cannot reach: after a regdirect post_load the container
    holds `_w_rep` INSTEAD of `_w_op`, with scales/zeros unchanged. The descriptor must follow the
    live buffers, not a remembered name."""
    _lds_knobs(monkeypatch)
    c = _GroupedCompressedTensorsExperts(E, N, K, _q("compressed-tensors", sym=False))
    for name, v in list(vars(c).items()):
        if isinstance(v, torch.Tensor):
            _fill(v, 7)
    c.post_load()
    # Emulate the MOE_W4A16 tail: drop _w_op, add a _w_rep of the register-direct shape.
    del c._w_op
    c._w_rep = _fill(torch.empty(E, N, K // 8, dtype=torch.int32), 11)
    spec = spec_for_container(c, E)
    names = [comp.name for comp in spec.components]
    assert "_w_rep" in names and "_w_op" not in names
    assert "_scales_op" in names and "_zeros_op" in names


# =====================================================================================
# Fail-closed
# =====================================================================================


def _distinct(*shape) -> torch.Tensor:
    """A tensor whose dim-0 rows differ — an all-zeros tensor is legitimately expert-invariant and
    would be excluded from the granule, which is correct behaviour but useless as a fixture."""
    n = 1
    for d in shape:
        n *= d
    return torch.arange(n, dtype=torch.float32).reshape(shape)


class _Bogus(ExpertContainer):
    def __init__(self):
        self.w = _distinct(E, 8, 8)
        self._num_experts = E

    def forward(self, *a, **k):  # pragma: no cover
        raise RuntimeError


def test_unknown_axis_tensor_raises_with_the_name_and_the_fix():
    c = _Bogus()
    c.mystery_scale = _distinct(E + 1, 3)
    with pytest.raises(GranuleError) as ei:
        c.granule_spec()
    msg = str(ei.value)
    assert "mystery_scale" in msg and "_residency_shared" in msg


def test_declared_shared_is_accepted_and_excluded():
    c = _Bogus()
    c.mystery_scale = _distinct(E + 1, 3)
    c._residency_shared = ("mystery_scale",)
    spec = c.granule_spec()
    assert [comp.name for comp in spec.components] == ["w"]
    rep = {r.name: r.reason for r in spec.replicated}
    assert rep == {"mystery_scale": "declared-shared"}


def test_non_contiguous_expert_axis_raises():
    """A strided per-expert tensor makes `base + e*(numel//E)*itemsize` simply false."""
    c = _Bogus()
    c.w = _distinct(8, E, 8).transpose(0, 1)  # shape[0]==E, NOT contiguous
    with pytest.raises(GranuleError, match="contiguous"):
        c.granule_spec()


def test_partial_storage_overlap_raises():
    c = _Bogus()
    base = _distinct(E, 16)
    c.w = base
    c.half = base.reshape(-1)[: E * 8].reshape(E, 8)  # overlaps `w` partially
    with pytest.raises(GranuleError, match="PARTIALLY overlap"):
        c.granule_spec()


def test_meta_tensors_raise_unless_allowed():
    """The model is built on the meta device; every meta tensor reports data_ptr()==0, so dedupe and
    invariance are both meaningless there."""
    with torch.device("meta"):
        c = _Bogus()
        c.w = torch.zeros(E, 8, 8)  # meta: contents are irrelevant
    with pytest.raises(GranuleError, match="META"):
        c.granule_spec()
    spec = c.granule_spec(allow_meta=True)
    assert spec.meta is True and [x.name for x in spec.components] == ["w"]


def test_missing_expert_count_raises():
    c = _Bogus()
    c._num_experts = 0
    with pytest.raises(GranuleError, match="expert count is unset"):
        c.granule_spec()


def test_list_and_dict_attributes_round_trip():
    """A tensor found inside a list/dict attribute must be both FOUND (fail-closed is only closed
    over what the walk reaches) and RESOLVABLE — a name the walk can emit but `expert_slice` cannot
    look up would raise on a container the descriptor claimed to describe."""
    c = _Bogus()
    c.parts = [_distinct(E, 3), _distinct(E, 5)]
    c.named = {"lo": _distinct(E, 7)}
    spec = c.granule_spec()
    names = [comp.name for comp in spec.components]
    assert names == ["w", "parts[0]", "parts[1]", "named[lo]"]
    views = spec.expert_slice(c, 2)
    assert torch.equal(views["parts[1]"], c.parts[1][2])
    assert torch.equal(views["named[lo]"], c.named["lo"][2])


def test_nested_module_tensors_are_walked():
    """`nn.Module` subtrees are invisible to a plain BaseOP `__dict__` walk; on Qwen3.5 those are
    three of every four layers' dense weights."""
    c = _Bogus()
    sub = torch.nn.Module()
    sub.register_buffer("extra", _distinct(E, 4))
    c.sub = sub
    spec = c.granule_spec()
    assert "sub.extra" in [comp.name for comp in spec.components]


# =====================================================================================
# Dense: the SAME mechanism with granule_axis = None
# =====================================================================================


def test_dense_container_is_one_granule():
    class _Dense:
        def __init__(self):
            self._w_packed_op = torch.zeros(64, 16, dtype=torch.int32)
            self._scales_op = torch.zeros(4, 64, dtype=torch.float16)
            self._zeros_op = torch.zeros(4, 8, dtype=torch.int32)
            self.bias = torch.zeros(64)
            self._tp_size = 2  # a non-tensor attribute must not disturb the walk

    d = _Dense()
    spec = derive_granule_spec(d, None)
    assert spec.num_experts is None and spec.num_granules == 1
    assert [c.name for c in spec.components] == [
        "_w_packed_op", "_scales_op", "_zeros_op", "bias"
    ]
    assert spec.granule_bytes == spec.stacked_bytes == spec.total_bytes
    assert spec.granule_bytes == 64 * 16 * 4 + 4 * 64 * 2 + 4 * 8 * 4 + 64 * 4
    plan = plan_component_major(spec)
    assert plan.num_experts is None
    assert plan.components[0].num_rows == 1


def _dense_linear(method):
    """A REAL `LinearReplicated` (no TP info needed — it passes its sizes straight through), loaded
    and post_load'ed through the real quant method."""
    from minisgl.layers.linear import LinearReplicated

    lin = LinearReplicated(input_size=256, output_size=64, has_bias=True, quant_method=method)
    for i, (name, v) in enumerate(list(vars(lin).items())):
        if isinstance(v, torch.Tensor):
            _fill(v, 40 + i)
    lin.post_load()
    return lin


def test_dense_unquantized_linear_granule():
    from minisgl.quant.method import UnquantizedLinearMethod

    lin = _dense_linear(UnquantizedLinearMethod())
    spec = lin.granule_spec()
    names = [c.name for c in spec.components]
    assert "weight" in names
    # THE BIAS TRAVELS. A moved weight with a left-behind bias is the same family of silent bug as a
    # left-behind scale, so it must be in the granule, not quietly skipped as "not a weight".
    assert "bias" in names
    assert set(lin.per_expert_tensors()) == set(names)


def test_dense_compressed_tensors_linear_granule():
    from minisgl.quant.method import W4A8LinearMethod

    q = QuantConfig(method="compressed-tensors", bits=4, group_size=32, sym=False)
    lin = _dense_linear(W4A8LinearMethod(q))
    names = [c.name for c in lin.granule_spec().components]
    # post_load DELETED weight_packed/weight_scale/weight_zero_point and built the op-layout triple.
    # A state_dict-driven walk would find nothing here; the underscore-inclusive walk finds all three.
    assert {"_w_packed_op", "_scales_op", "_zeros_op", "bias"} == set(names)


def test_dense_fp8_linear_granule():
    from minisgl.quant.method import Fp8W8A8LinearMethod

    q = QuantConfig(method="compressed-tensors", bits=8, group_size=0, sym=True, weight_type="float")
    lin = _dense_linear(Fp8W8A8LinearMethod(q))
    names = [c.name for c in lin.granule_spec().components]
    assert {"_w_op", "_scales_op", "bias"} == set(names)
