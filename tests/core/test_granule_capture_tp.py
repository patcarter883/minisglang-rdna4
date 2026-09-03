"""Granule descriptor under the two constraints that outrank byte accounting: GRAPH CAPTURE and TP.

Each test here pins a defect that was live in `weights/granule.py` before 2026-09-03. They are all
of the same family — nothing crashes, nothing warns, the numbers are merely wrong:

  * a spec derived under graph capture launches kernels and SYNCHRONIZES the host (`torch.equal` in
    the invariance check) and hands back views whose pointers a captured graph would bake forever;
  * a `_residency_shared` declaration was trusted without verification, so a stale one drops a real
    per-expert buffer and every expert dequantizes against expert 0's;
  * `GranuleSpec` says nothing about ADDRESSES, so a rebind after capture leaves replay reading the
    previous allocation with the right shapes.

GPU-FREE. Capture is simulated by monkeypatching the two `torch.cuda` predicates the guard reads,
which is the whole of its logic — there is no device state involved in "am I capturing".

    docker run --rm --network none -v <worktree>:/engine:ro --entrypoint bash minisgl-rdna4:lean \\
      -lc 'PYTHONPATH=/engine/python MINISGL_TAIL_HIP=0 pytest /engine/tests/core -k granule'
"""

from __future__ import annotations

import pytest
import torch
from minisgl.layers import moe as moe_mod
from minisgl.layers.moe import _GroupedCompressedTensorsExperts, _GroupedFP8Experts
from minisgl.quant.config import QuantConfig
from minisgl.weights import granule as gmod
from minisgl.weights.granule import (
    ExpertContainer,
    GranuleError,
    assert_bindings_unchanged,
    assert_spec_still_holds,
    spec_for_container,
)

E, N, K = 4, 32, 128


def _lds(monkeypatch):
    monkeypatch.setattr(moe_mod.kernels, "MOE_W4A16", "0", raising=False)
    monkeypatch.setattr(moe_mod.kernels, "MOE_W8A8_REGDIRECT", False, raising=False)


def _q(sym: bool) -> QuantConfig:
    return QuantConfig(method="compressed-tensors", bits=4, group_size=32, sym=sym)


def _ct(monkeypatch, sym: bool = False):
    _lds(monkeypatch)
    c = _GroupedCompressedTensorsExperts(E, N, K, _q(sym))
    g = torch.Generator().manual_seed(3)
    for _, v in list(vars(c).items()):
        if isinstance(v, torch.Tensor):
            v.copy_(
                torch.randint(0, 2**31 - 1, v.shape, generator=g, dtype=torch.int64).to(v.dtype)
                if v.dtype in (torch.int32, torch.uint8)
                else torch.randn(v.shape, generator=g).to(v.dtype)
            )
    c.post_load()
    return c


def _fp8(monkeypatch, oldmoe: bool = False):
    """`post_load` keeps `weight` alongside its uint8 alias `_w_op` only under OLDMOE; the default
    arm deletes the checkpoint copy. The alias case is the one worth testing."""
    _lds(monkeypatch)
    monkeypatch.setenv("MINISGL_ZAYA_OLDMOE", "1" if oldmoe else "0")
    c = _GroupedFP8Experts(E, N, K)
    c.weight.copy_(torch.randn(E, N, K).to(torch.float8_e4m3fn))
    c.weight_scale.copy_(torch.rand(E, N, 1) + 0.5)
    c.post_load()
    return c


@pytest.fixture()
def tp1():
    """tp_size=1, EP off — the smallest state `MoELayer.__init__` needs. `set_tp_info` refuses a
    second call, so this is process-idempotent by design."""
    from minisgl.distributed import info as dinfo

    if dinfo.try_get_tp_info() is None:
        dinfo.set_tp_info(0, 1)
    return dinfo.get_tp_info()


def _fill_and_post_load(layer):
    for c in layer.expert_containers().values():
        g = torch.Generator().manual_seed(11)
        for _, v in list(vars(c).items()):
            if isinstance(v, torch.Tensor):
                v.copy_(
                    torch.randint(0, 2**31 - 1, v.shape, generator=g, dtype=torch.int64).to(v.dtype)
                    if v.dtype in (torch.int32, torch.uint8)
                    else torch.randn(v.shape, generator=g).to(v.dtype)
                )
        c.post_load()
    return layer


def _pretend_capturing(monkeypatch, on: bool = True):
    """Make `_is_capturing()` answer `on` without a device. Both predicates are patched because the
    guard short-circuits on `is_initialized` precisely so a CPU host never touches the driver."""
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: on, raising=False)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: on, raising=False)


# =====================================================================================
# Graph capture: descriptor work must never happen inside one
# =====================================================================================


def test_derivation_refuses_under_graph_capture(monkeypatch):
    """`torch.equal` on device tensors launches a kernel AND syncs the host — illegal mid-capture."""
    c = _ct(monkeypatch, sym=True)
    _pretend_capturing(monkeypatch)
    with pytest.raises(GranuleError, match="CAPTURING"):
        spec_for_container(c, E)


def test_expert_slice_and_stacked_tensors_refuse_under_capture(monkeypatch):
    """The views are pointers valid NOW; a captured graph would bake them past the next rebind."""
    c = _ct(monkeypatch, sym=True)
    spec = spec_for_container(c, E)  # derived legitimately, before capture
    _pretend_capturing(monkeypatch)
    with pytest.raises(GranuleError, match="CAPTURING"):
        spec.expert_slice(c, 0)
    with pytest.raises(GranuleError, match="CAPTURING"):
        spec.stacked_tensors(c)


def test_moelayer_accessors_refuse_under_capture(monkeypatch, tp1):
    """Same guard reached through the real MoE seam, not just the free functions."""
    _lds(monkeypatch)
    layer = _fill_and_post_load(
        moe_mod.MoELayer(
            num_experts=E, top_k=2, hidden_size=K, intermediate_size=N, quant=_q(True)
        )
    )
    layer.granule_specs()  # fine outside capture
    _pretend_capturing(monkeypatch)
    with pytest.raises(GranuleError, match="CAPTURING"):
        layer.granule_specs()
    with pytest.raises(GranuleError, match="CAPTURING"):
        layer.co_demanded_granule_bytes()


def test_guard_is_free_and_silent_on_a_host_with_no_cuda_context(monkeypatch):
    """`is_initialized()` is a Python flag; a process that never made a context cannot be capturing,
    and asking must not drag the driver in. If this regressed, every CPU unit test would either
    initialise HIP or raise."""
    calls = []
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False, raising=False)
    monkeypatch.setattr(
        torch.cuda,
        "is_current_stream_capturing",
        lambda: calls.append(1) or True,  # pragma: no cover - must never run
        raising=False,
    )
    assert gmod._is_capturing() is False
    assert calls == []


def test_granule_module_creates_no_side_stream():
    """A side-stream op cannot be recorded into a captured graph (`host_arena.py:25`). This module
    must therefore own no stream at all — the check is structural, so it cannot rot."""
    src = open(gmod.__file__).read()
    for banned in ("torch.cuda.Stream(", "with torch.cuda.stream(", "wait_stream", "record_stream"):
        assert banned not in src, f"granule.py must not touch streams; found {banned!r}"


# =====================================================================================
# TP determinism: declarations, not contents
# =====================================================================================


def test_stale_declaration_raises_instead_of_dropping_a_real_buffer(monkeypatch):
    """REGRESSION. A declared name used to be exempted with NO check.

    That is the module's own worst case wearing the module's own uniform: a symmetric-CT declaration
    left in place across a checkpoint change would drop REAL per-group zero-points from every
    granule, every expert would dequantize against expert 0's zeros, and the output is plausible
    text with no error. The declaration is now verified against the live rows."""
    c = _ct(monkeypatch, sym=False)  # asymmetric: _zeros_op is genuinely per-expert
    assert "_zeros_op" in {comp.name for comp in spec_for_container(c, E).components}
    c._residency_shared = ("_zeros_op",)  # the stale declaration
    with pytest.raises(GranuleError, match="NOT bitwise identical"):
        spec_for_container(c, E)


def test_declaration_check_reads_the_LAST_row_not_just_the_first(monkeypatch):
    """The verification is chunked to bound its temporary; an early exit over the first chunk only
    would accept a stack that diverges at expert E-1 — the silent case again, just later in the
    stack. Force a one-row chunk so the loop is genuinely exercised."""
    monkeypatch.setattr(gmod, "_ROWCMP_CHUNK_BYTES", 1)
    c = _ct(monkeypatch, sym=True)
    c._zeros_op[E - 1, 0, 0] += 1  # only the LAST expert differs
    with pytest.raises(GranuleError, match="NOT bitwise identical"):
        spec_for_container(c, E)


def test_declaration_check_bounds_its_temporary(monkeypatch):
    """A single whole-stack `torch.equal` against a broadcast row 0 materializes an elementwise
    temporary the SIZE OF THE STACK — on the device, at boot, when VRAM is tightest and the
    allocator's reserved pool is what OOMs. Chunking is what bounds it, so assert the chunking
    happens rather than only that the answer is right.

    Isolated to a container with ONE declared tensor, so nothing else's cheap 2-row reject can be
    mistaken for the bounded scan."""

    class _Big(ExpertContainer):
        _residency_shared = ("big",)

        def __init__(self):
            self.big = torch.zeros(16, 64, dtype=torch.uint8)  # invariant: 15 rows to verify
            self._num_experts = 16

        def forward(self, *a, **k):  # pragma: no cover
            raise RuntimeError

    seen = []
    real = torch.equal
    monkeypatch.setattr(gmod, "_ROWCMP_CHUNK_BYTES", 64)  # exactly one row per chunk
    monkeypatch.setattr(torch, "equal", lambda a, b: (seen.append(a.numel()), real(a, b))[1])

    spec = _Big().granule_spec()
    assert [r.name for r in spec.replicated] == ["big"], "the declaration was not verified at all"
    assert max(seen) <= 64, f"compared {max(seen)} bytes at once against a 64-byte cap"
    assert len(seen) >= 8, "the whole stack was compared in one shot, not in bounded chunks"


def test_declared_shared_without_an_expert_axis_is_still_accepted(monkeypatch):
    """Non-vacuity for the verification: a genuinely shared buffer carries no expert axis, so there
    is nothing to contradict and it must NOT start raising."""

    class _C(ExpertContainer):
        _residency_shared = ("router_bias",)

        def __init__(self):
            self.w = torch.arange(E * 8, dtype=torch.float32).reshape(E, 8)
            self.router_bias = torch.arange(7, dtype=torch.float32)  # no expert axis
            self._num_experts = E

        def forward(self, *a, **k):  # pragma: no cover
            raise RuntimeError

    spec = _C().granule_spec()
    assert [c.name for c in spec.components] == ["w"]
    assert [(r.name, r.reason) for r in spec.replicated] == [("router_bias", "declared-shared")]


def test_two_ranks_with_different_shards_agree_on_the_plan(monkeypatch):
    """The end-to-end TP property: identical config, different bytes, identical placement.

    `plan_component_major` is what a rank turns into offsets, so agreeing on the fingerprint but not
    on the offsets would still hang the collectives."""
    from minisgl.weights.granule import plan_component_major

    rank0 = _ct(monkeypatch, sym=False)
    rank1 = _ct(monkeypatch, sym=False)
    rank1._w_op.mul_(0)  # rank 1's shard is degenerate: every expert row identical
    rank1._zeros_op[:] = rank1._zeros_op[0]
    rank1._scales_op[:] = rank1._scales_op[0]

    p0 = plan_component_major(spec_for_container(rank0, E))
    p1 = plan_component_major(spec_for_container(rank1, E))
    assert [(c.name, c.offset, c.row_bytes, c.num_rows) for c in p0.components] == [
        (c.name, c.offset, c.row_bytes, c.num_rows) for c in p1.components
    ]
    assert p0.nbytes == p1.nbytes


# =====================================================================================
# Addresses: what a captured graph actually bakes
# =====================================================================================


def test_binding_fingerprint_moves_when_a_component_is_rebound(monkeypatch):
    """`fingerprint()` cannot see a rebind — same names, same dtypes, same shapes, different memory.
    A graph captured before the rebind replays against the OLD allocation."""
    c = _ct(monkeypatch, sym=True)
    spec = spec_for_container(c, E)
    before = spec.binding_fingerprint(c)
    assert spec.binding_fingerprint(c) == before  # stable when nothing moved
    assert_bindings_unchanged(spec, c, before)

    c._w_op = c._w_op.clone()  # exactly what an arena bind does
    assert spec.fingerprint() == spec_for_container(c, E).fingerprint()  # manifest is blind to it
    assert spec.binding_fingerprint(c) != before
    with pytest.raises(GranuleError, match="MOVED"):
        assert_bindings_unchanged(spec, c, before, where="post-capture")


def test_binding_fingerprint_catches_a_DE_ALIASING_rebind(monkeypatch):
    """`_GroupedFP8Experts` keeps `weight` and `_w_op` as ONE component with two names. A copy-based
    rebind that gave each its own buffer leaves `dequant()` and the kernel reading different memory
    — every name still resolves to a right-shaped tensor, so only an address-level or manifest-level
    check sees it. Both are asserted."""
    c = _fp8(monkeypatch, oldmoe=True)  # keeps `weight`; `_w_op` is a uint8 view of the same bytes
    spec = spec_for_container(c, E)
    comp = {n: cc for cc in spec.components for n in cc.names}
    assert comp["weight"] is comp["_w_op"], "the alias merge is the precondition for this test"
    before = spec.binding_fingerprint(c)

    c.weight = c.weight.clone()  # de-aliased: same shape, same dtype, different storage
    assert spec.binding_fingerprint(c) != before
    with pytest.raises(GranuleError, match="MOVED"):
        assert_bindings_unchanged(spec, c, before)
    with pytest.raises(GranuleError, match="DE-ALIAS"):
        assert_spec_still_holds(c, spec)


def test_assert_spec_still_holds_accepts_the_unchanged_container(monkeypatch):
    """Non-vacuity: the detector above has to accept the legitimate case too."""
    c = _ct(monkeypatch, sym=True)
    spec = spec_for_container(c, E)
    assert_spec_still_holds(c, spec, where="unchanged")


# =====================================================================================
# The layer-level offload refusal a placement pass has to consult
# =====================================================================================


def test_layer_offload_refusal_names_the_knob(monkeypatch, tp1):
    """`granule_specs()` deliberately does NOT consult the refusal (a spec is also derived for pure
    sizing), so the layer has to surface it or a planner places a layer whose forward streams the
    whole stack every step.

    The knob is read at `post_load`, not at placement time, so both legs build their own layer."""
    _lds(monkeypatch)

    def _layer():
        return _fill_and_post_load(
            moe_mod.MoELayer(
                num_experts=E, top_k=2, hidden_size=K, intermediate_size=N, fp8_experts=True
            )
        )

    monkeypatch.delenv("MINISGL_ZAYA_OLDMOE", raising=False)
    assert _layer().offload_refusal() is None

    monkeypatch.setenv("MINISGL_ZAYA_OLDMOE", "1")
    why = _layer().offload_refusal()
    assert why is not None and "MINISGL_ZAYA_OLDMOE" in why and "gate_up_proj" in why
