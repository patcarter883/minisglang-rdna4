"""Two silent-wrong-numbers holes in the granule derivation, and the guards that close them.

Both are the same failure shape as the dropped scale the module exists to prevent: right shapes,
right dtypes, no kernel fault, plausible text, and a descriptor that reported the container as
FULLY described while it was not.

  1. STRIDED ALIASES.  `_byte_key` is `(storage, storage_offset, numel*itemsize)`. That last term is
     a tensor's real byte span only when it is contiguous, so `t` and `t.transpose(1, 2)` key
     IDENTICALLY — same storage, same offset, same numel. Before 2026-09-03 contiguity was only
     checked inside the per-expert branch, i.e. AFTER the dedupe had already merged the strided view
     in as an *alias* of the contiguous one (and not at all on the dense arm). `moe_interpose`
     then rebinds every alias of a component to the one arena row, `_reinterpret` sees matching
     dtype+shape and hands it over unchanged, and the transpose is simply gone. The same understated
     span also lets the partial-overlap scan call two genuinely overlapping tensors disjoint.

  2. DECODE POLICY.  Some `post_load`s decide how the bytes are READ. Compressed-tensors int4 is the
     shipped case: `ct_packed_sign_convention` reads the nibble histogram and either XORs the whole
     stack to uint4b8 or passes it through, and transforms `_zeros_op` to match. That decision is not
     a tensor, so the walk cannot see it — yet w13 and w2 of one layer resolving it differently means
     one GEMM decodes `q+8` while the other decodes two's-complement: every weight of that GEMM off
     by 8 quanta, nothing to trip on. `_granule_policy` declares the DECISION path, the spec carries
     it, `fingerprint()` hashes it and `assert_granule_pair_consistent` compares it.

Each guard is also shown to be NON-VACUOUS: the strided pair really does collide under the old key
(so the guard is not rejecting something that was already distinguishable), every shipped format is
really contiguous (so the guard is not silently refusing production containers), and a pair with the
SAME decode decision but different sample diagnostics really is accepted (so the policy comparison
is not just "raise on any two containers").

GPU-free. Run with the rest:

    docker run --rm -v <worktree>:/engine --entrypoint bash minisgl-rdna4:lean-pytest \\
      -lc 'PYTHONPATH=/engine/python MINISGL_TAIL_HIP=0 pytest /engine/tests/core -k granule'
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
)
from minisgl.quant.config import QuantConfig
from minisgl.weights.granule import (
    ExpertContainer,
    GranuleError,
    _byte_key,
    assert_granule_pair_consistent,
    derive_granule_spec,
    spec_for_container,
)

E, N, K = 4, 32, 128
HIDDEN, INTER = 128, 64


class _Bare(ExpertContainer):
    """Minimal per-expert container: declares its granule axis and nothing else."""

    def __init__(self):
        self.w = torch.arange(E * 8 * 8, dtype=torch.float32).reshape(E, 8, 8).contiguous()
        self._num_experts = E

    def forward(self, *a, **k):  # pragma: no cover - storage container
        raise RuntimeError


# =====================================================================================
# 1. Strided aliases
# =====================================================================================


def test_a_transposed_view_collides_with_its_base_under_the_byte_key():
    """NON-VACUITY for the contiguity guard: prove the two really are indistinguishable to the key.

    If they keyed differently there would be nothing to guard against — the guard would just be
    refusing something the dedupe already separated. They do not: same storage, same offset, same
    numel*itemsize. And they are NOT the same bytes in row-major order, which is what makes merging
    them a wrong-numbers bug rather than a harmless one.
    """
    t = torch.arange(E * 4 * 5, dtype=torch.float32).reshape(E, 4, 5).contiguous()
    tt = t.transpose(1, 2)
    assert _byte_key(t) == _byte_key(tt), "the guard would be vacuous if these keyed differently"
    assert not torch.equal(t.reshape(-1), tt.reshape(-1)), (
        "a transpose that happens to be row-identical would make this case harmless"
    )


def test_strided_alias_is_refused_naming_the_attribute():
    """The per-expert arm. Before the fix this merged `_w_t` in as an alias of `w` and returned a
    spec that claimed to describe the container."""
    c = _Bare()
    c._w_t = c.w.transpose(1, 2)  # same storage, offset and numel; a DIFFERENT reading of them
    with pytest.raises(GranuleError) as ei:
        c.granule_spec()
    msg = str(ei.value)
    assert "contiguous" in msg
    assert "_w_t" in msg or "w" in msg


def test_dense_arm_refuses_a_non_contiguous_tensor():
    """The dense arm had NO contiguity check at all, so the hole was open there unconditionally —
    and `_LinearTPImpl` is the dense arm's only production user."""

    class _Dense(ExpertContainer):
        _granule_dense = True

        def __init__(self):
            self.weight = torch.zeros(8, 16).transpose(0, 1)  # (16, 8), strided

        def forward(self, *a, **k):  # pragma: no cover
            raise RuntimeError

    with pytest.raises(GranuleError, match="contiguous"):
        _Dense().granule_spec()
    # ...and the SAME container is accepted once post_load has made it contiguous, so the guard is
    # about the stride and not about the dense arm.
    d = _Dense()
    d.weight = d.weight.contiguous()
    assert [c.name for c in d.granule_spec().components] == ["weight"]


def test_understated_span_would_have_hidden_a_real_partial_overlap():
    """The second consequence of the same understated span: `a`'s real byte reach is 3x what
    `numel*itemsize` reports, so the overlap scan compared the wrong intervals."""
    base = torch.arange(4 * 12, dtype=torch.float32).reshape(4, 12).contiguous()
    a = base[:, :4]  # strided: numel*itemsize = 64 B, real reach = 4*12*4 - 8*4 = 160 B
    b = base.reshape(-1)[24:36].reshape(3, 4)  # contiguous, [96, 144) B — inside a's real reach

    a_span = a.numel() * a.element_size()
    b_off = b.storage_offset() * b.element_size()
    assert b_off >= a_span, (
        "non-vacuity: under the understated span these look DISJOINT, so the overlap scan "
        "would have waved them through"
    )
    assert b_off < (a.shape[0] - 1) * a.stride(0) * a.element_size() + a_span, (
        "...while `b` genuinely lands inside the rows `a` reads"
    )

    # DENSE on purpose: the per-expert arm's own (late) contiguity check would have fired here for
    # an unrelated reason, so it would not have proved the overlap scan was blind. The dense arm had
    # no contiguity check at all, so before the fix this container derived CLEANLY — two components
    # whose slabs both claim bytes the other owns.
    class _Overlap(ExpertContainer):
        _granule_dense = True

        def __init__(self):
            self.a = a
            self.b = b

        def forward(self, *x, **k):  # pragma: no cover
            raise RuntimeError

    with pytest.raises(GranuleError, match="contiguous"):
        _Overlap().granule_spec()


# =====================================================================================
# ...and the guard must not refuse anything the engine actually serves
# =====================================================================================


def _q(method: str, **kw) -> QuantConfig:
    base = {"method": method, "bits": 4, "group_size": 32, "sym": True}
    base.update(kw)
    return QuantConfig(**base)  # type: ignore[arg-type]


def _fill(t: torch.Tensor, seed: int, kind: str = "random") -> torch.Tensor:
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


def _lds(monkeypatch):
    monkeypatch.setattr(moe_mod.kernels, "MOE_W4A16", "0", raising=False)
    monkeypatch.setattr(moe_mod.kernels, "MOE_MXFP4_REGDIRECT", False, raising=False)
    monkeypatch.setattr(moe_mod.kernels, "MOE_W8A8_REGDIRECT", False, raising=False)


def _load(container):
    for i, (name, v) in enumerate(list(vars(container).items())):
        if isinstance(v, torch.Tensor):
            kind = "random"
            if isinstance(container, _GroupedGPTQExperts) and name == "qzeros":
                kind = "nibble14"
            elif isinstance(container, _GroupedMxFp4Experts) and name == "weight_scale":
                kind = "e8m0"
            _fill(v, 7 + i, kind)
    container.post_load()
    return container


_FORMATS = {
    "gptq": lambda: _GroupedGPTQExperts(E, N, K, _q("gptq", sym=False)),
    "awq": lambda: _GroupedAWQExperts(E, N, K, _q("awq", sym=False)),
    "ct_sym": lambda: _GroupedCompressedTensorsExperts(E, N, K, _q("compressed-tensors", sym=True)),
    "ct_asym": lambda: _GroupedCompressedTensorsExperts(E, N, K, _q("compressed-tensors", sym=False)),
    "mxfp4": lambda: _GroupedMxFp4Experts(E, N, K, _q("compressed-tensors", weight_type="float")),
    "nvfp4": lambda: _GroupedNvFp4Experts(
        E, N, K, _q("compressed-tensors", group_size=16, weight_type="float")
    ),
    "fp8": lambda: _GroupedFP8Experts(E, N, K),
}


@pytest.mark.parametrize("fmt", sorted(_FORMATS))
def test_every_shipped_format_survives_the_contiguity_gate(fmt, monkeypatch):
    """The guard is fail-closed, so it MUST be shown not to close on production containers. Every
    format is run through its real `post_load` and then derived; a format that grows a strided
    buffer will fail here at boot rather than corrupt an alias at bind."""
    _lds(monkeypatch)
    c = _load(_FORMATS[fmt]())
    spec = spec_for_container(c, E)
    assert spec.components, f"{fmt}: empty granule"
    for name, t in spec.stacked_tensors(c).items():
        assert t.is_contiguous(), f"{fmt}.{name} is strided"


# =====================================================================================
# 2. Decode policy
# =====================================================================================


def _ct_pair(monkeypatch, sym: bool = False):
    """A real w13/w2 pair with `MoELayer.__init__`'s shapes, each through its real post_load."""
    _lds(monkeypatch)
    out = []
    for n_out, n_in, seed in ((2 * INTER, HIDDEN, 11), (HIDDEN, INTER, 12)):
        c = _GroupedCompressedTensorsExperts(E, n_out, n_in, _q("compressed-tensors", sym=sym))
        for i, (_, v) in enumerate(list(vars(c).items())):
            if isinstance(v, torch.Tensor):
                _fill(v, seed + i)
        c.post_load()
        out.append(c)
    return out


def test_ct_decode_decision_is_recorded_on_the_spec(monkeypatch):
    w13, _ = _ct_pair(monkeypatch)
    spec = spec_for_container(w13, E)
    assert dict(spec.policy)["_ct_sign.uint4b8"] in ("True", "False")


def test_policy_records_the_DECISION_not_the_sample_diagnostics(monkeypatch):
    """NON-VACUITY, and a TP requirement. `CtSignConvention` also carries `margin`/`sampled_words`/
    `stride`, which are properties of THIS stack's sample: w13 and w2 have different shapes, and two
    TP ranks hold different shards, so hashing them would make `fingerprint()` content-dependent —
    the exact divergence the fingerprint exists to detect. Only `uint4b8` may be recorded."""
    w13, w2 = _ct_pair(monkeypatch)
    assert w13._ct_sign.uint4b8 == w2._ct_sign.uint4b8
    assert (w13._ct_sign.margin, w13._ct_sign.sampled_words) != (
        w2._ct_sign.margin,
        w2._ct_sign.sampled_words,
    ), "the two stacks must genuinely differ in their diagnostics or this proves nothing"
    a, b = spec_for_container(w13, E), spec_for_container(w2, E)
    assert a.policy == b.policy
    assert_granule_pair_consistent(a, b, where="ct-pair")  # the legitimate case is ACCEPTED


def test_disagreeing_decode_decision_across_the_pair_is_refused(monkeypatch):
    """w13 XORed to uint4b8 and w2 left two's-complement (or the reverse) is every weight of one
    GEMM off by 8 quanta, with right shapes and no error. It must be a boot failure."""
    w13, w2 = _ct_pair(monkeypatch)
    a = spec_for_container(w13, E)
    w2._ct_sign = w2._ct_sign._replace(uint4b8=not w2._ct_sign.uint4b8)
    b = spec_for_container(w2, E)
    with pytest.raises(GranuleError, match="DECODE POLICY"):
        assert_granule_pair_consistent(a, b, where="ct-pair")


def test_fingerprint_moves_with_the_decode_decision(monkeypatch):
    """Two TP ranks that resolved the nibble histogram differently produce identical component sets,
    identical shapes and identical byte totals. The fingerprint is the only thing that can move."""
    w13, _ = _ct_pair(monkeypatch)
    before = spec_for_container(w13, E)
    w13._ct_sign = w13._ct_sign._replace(uint4b8=not w13._ct_sign.uint4b8)
    after = spec_for_container(w13, E)
    assert [c.name for c in before.components] == [c.name for c in after.components]
    assert before.stacked_bytes == after.stacked_bytes
    assert before.fingerprint() != after.fingerprint()


def test_a_format_with_no_decode_decision_records_an_empty_policy(monkeypatch):
    """`_granule_policy` is empty for every format that makes no such decision, so the field costs
    nothing and cannot drift into a per-format table."""
    _lds(monkeypatch)
    spec = spec_for_container(_load(_GroupedFP8Experts(E, N, K)), E)
    assert spec.policy == ()


def test_missing_policy_attribute_is_recorded_rather_than_raising(monkeypatch):
    """`_granule_policy` is read on a META container too, before any `post_load` has set `_ct_sign`.
    A missing path must record `None`, not explode — but two containers where one HAS run post_load
    and the other has not must still come out different."""
    _lds(monkeypatch)
    fresh = _GroupedCompressedTensorsExperts(E, N, K, _q("compressed-tensors", sym=True))
    a = derive_granule_spec(fresh, E, detect_invariant=False)
    assert dict(a.policy)["_ct_sign.uint4b8"] == "None"
    loaded = spec_for_container(_load(_GroupedCompressedTensorsExperts(E, N, K,
                                                                      _q("compressed-tensors", sym=True))), E)
    assert a.policy != loaded.policy


def test_dense_linear_declares_the_same_policy_path():
    """The dense compressed-tensors linear makes the identical decision (`W4A8LinearMethod` ->
    `layer._ct_sign`). Declaring it there is what stops the dense arm from being the one place a
    convention flip is invisible — "MoE and dense land together"."""
    from minisgl.layers.linear import _LinearTPImpl

    assert "_ct_sign.uint4b8" in _LinearTPImpl._granule_policy


def test_moe_layer_pair_check_covers_the_policy(tp1_info, monkeypatch):
    """The comparison has to run at the real seam, not only when a test calls it directly:
    `MoELayer.granule_specs` is what `moe_interpose.attach_seams` prefers."""
    _lds(monkeypatch)
    layer = moe_mod.MoELayer(
        num_experts=E, top_k=2, hidden_size=HIDDEN, intermediate_size=INTER,
        quant=_q("compressed-tensors", sym=False),
    )
    for c in layer.expert_containers().values():
        for i, (_, v) in enumerate(list(vars(c).items())):
            if isinstance(v, torch.Tensor):
                _fill(v, 21 + i)
        c.post_load()
    layer.granule_specs()  # the legitimate case passes
    layer.down_proj._ct_sign = layer.down_proj._ct_sign._replace(
        uint4b8=not layer.down_proj._ct_sign.uint4b8
    )
    with pytest.raises(GranuleError, match="DECODE POLICY"):
        layer.granule_specs()


@pytest.fixture()
def tp1_info():
    """tp_size=1, EP off — `MoELayer.__init__`'s minimum. `set_tp_info` refuses a second call, so
    this is process-idempotent."""
    from minisgl.distributed import info as dinfo

    if dinfo.try_get_tp_info() is None:
        dinfo.set_tp_info(0, 1)
    return dinfo.get_tp_info()


# =====================================================================================
# 3. One expert's bytes, traced end to end
# =====================================================================================


def test_one_experts_bytes_end_to_end_through_component_major(monkeypatch):
    """THE trace: a real asymmetric compressed-tensors container, packed by `plan_component_major`,
    read back with the kernels' own arithmetic, expert by expert.

    The question this answers is the whole point of the module — *can expert e ever be computed
    against expert f's scale or zero-point?* The kernels do not consult a descriptor: they compute
    `wq_e = w + e*N*K`, `ws_e = scales + e*G*N`, `zp_e = zeros + e*G*(N/8)`, each from the base
    pointer of its OWN component. So the proof has to be arithmetic on bytes, not on tensors: pack
    the arena exactly as the plan says, then for every e and every component recompute the kernel's
    address and byte-compare against `t[e]` of the live stack. Anything that pairs e's weight with
    f's scale shows up as a mismatch on the scale, never on the weight.
    """
    _lds(monkeypatch)
    c = _load(_GroupedCompressedTensorsExperts(E, N, K, _q("compressed-tensors", sym=False)))
    spec = spec_for_container(c, E)
    stacked = spec.stacked_tensors(c)
    names = [x.name for x in spec.components]
    # Non-vacuity: the trace is only interesting if the weight AND its scale AND its zeros are all
    # in the granule, with genuinely different row sizes (equal rows could hide an offset error).
    assert {"_w_op", "_scales_op", "_zeros_op"} <= set(names), names
    rows = {x.name: x.row_bytes for x in spec.components}
    assert len(set(rows.values())) == len(rows), f"equal row sizes would make this vacuous: {rows}"

    from minisgl.weights.granule import plan_component_major

    plan = plan_component_major(spec)
    arena = torch.zeros(plan.nbytes, dtype=torch.uint8)
    for cp in plan.components:
        flat = stacked[cp.name].reshape(-1).view(torch.uint8)
        arena[cp.offset : cp.offset + cp.nbytes] = flat

    for e in range(E):
        for cp in plan.components:
            off = plan.row_offset(cp.name, e)
            got = arena[off : off + cp.row_bytes]
            want = stacked[cp.name][e].reshape(-1).view(torch.uint8)
            assert torch.equal(got, want), (
                f"expert {e} component {cp.name}: component-major read at {off} does not match "
                f"t[{e}] — this is expert {e} being paired with another expert's bytes"
            )
        # And the pairing is exclusive: expert e's scale row must NOT equal any other expert's.
        for f in range(E):
            if f == e:
                continue
            assert not torch.equal(
                stacked["_scales_op"][e].reshape(-1), stacked["_scales_op"][f].reshape(-1)
            ), "the scales are row-identical, so a cross-expert pairing would be undetectable here"


def test_a_frame_major_slab_reads_the_wrong_bytes_for_the_same_arithmetic(monkeypatch):
    """The counterfactual that makes the packing choice load-bearing rather than a style preference.

    `GranuleSpec.layout` is a real `FrameLayout` — the manifest — and it is frame-major: expert 0's
    weight, then expert 0's scales, then expert 1's weight... If that geometry were ever used as the
    PACKING, every component's true row stride would become `frame_bytes`, while the kernel would
    still compute `base(c) + e*row_bytes(c)`. Nothing faults: the read lands inside the slab, is
    correctly aligned, and has the right length. It is just a different expert's — or a different
    component's — bytes. Asserted, so a future consumer cannot quietly pack frame-major.
    """
    _lds(monkeypatch)
    c = _load(_GroupedCompressedTensorsExperts(E, N, K, _q("compressed-tensors", sym=False)))
    spec = spec_for_container(c, E)
    stacked = spec.stacked_tensors(c)
    layout = spec.layout  # frame-major geometry for ONE expert's frame
    frame_bytes = layout.nbytes

    slab = torch.zeros(frame_bytes * E, dtype=torch.uint8)
    for e in range(E):
        for comp, off in zip(layout.components, layout.offsets):
            src = stacked[comp.name][e].reshape(-1).view(torch.uint8)
            base = e * frame_bytes + off
            slab[base : base + comp.nbytes] = src

    # The kernel's arithmetic, applied to the frame-major slab: component base = its offset in
    # frame 0, row stride = its own row bytes.
    wrong = 0
    for comp, off in zip(layout.components, layout.offsets):
        for e in range(1, E):  # e == 0 coincides by construction, which is exactly the trap
            got = slab[off + e * comp.nbytes : off + e * comp.nbytes + comp.nbytes]
            want = stacked[comp.name][e].reshape(-1).view(torch.uint8)
            assert got.numel() == want.numel(), "a wrong-bytes read is still WELL-FORMED"
            if not torch.equal(got, want):
                wrong += 1
    assert wrong > 0, (
        "frame-major happened to coincide with component-major for this container, so this guard "
        "would be vacuous — pick components with unequal row sizes"
    )
