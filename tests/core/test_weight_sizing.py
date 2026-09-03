"""Byte model for MoE expert stacks -- `minisgl.weights.sizing`. No GPU, no torch needed.

These numbers decide whether the pinned host arena fits P3b's measured 62 GiB two-rank ceiling, so
an error here is a serve that either refuses to boot or dies mid-load. Every expected value below is
computed by hand from the container `__init__` in `layers/moe.py` and written out as an explicit
product, not as a re-run of the function under test.
"""

from __future__ import annotations

import pytest

# MUST precede the minisgl import: installs a torch stub only when the real torch is unimportable,
# so this arithmetic is testable on a host with no working torch. Inert in the container.
from _offload_torch_stub import TORCH_IS_REAL  # noqa: I001
from minisgl.weights.sizing import (
    SCHEME_AWQ,
    SCHEME_CT_INT4,
    SCHEME_FP8,
    SCHEME_GPTQ,
    SCHEME_MXFP4,
    SCHEME_NVFP4,
    SCHEME_RXF,
    SCHEME_SUPPORTS_EP,
    SCHEME_UNKNOWN,
    SCHEME_UNQUANTIZED,
    ExpertScheme,
    analytic_gemm_bytes,
    expert_stack_bytes,
    post_load_delta_bytes,
    scheme_from_quant,
)


class FakeQuant:
    """A QuantConfig-shaped stub. `sizing` is duck-typed precisely so this works."""

    def __init__(self, **kw):
        self.method = kw.pop("method", "compressed-tensors")
        self.bits = kw.pop("bits", 4)
        self.group_size = kw.pop("group_size", 32)
        self.sym = kw.pop("sym", True)
        self.weight_type = kw.pop("weight_type", "int")
        self.ct_groups = kw.pop("ct_groups", ())
        self.is_fp8_w8a8 = kw.pop("is_fp8_w8a8", False)
        self.is_nvfp4 = kw.pop("is_nvfp4", False)
        self.is_rxf = kw.pop("is_rxf", False)
        self.weight_is_e2m1 = kw.pop("weight_is_e2m1", False)
        self.is_int4 = kw.pop("is_int4", True)
        self.is_gptq = kw.pop("is_gptq", False)
        self.is_awq = kw.pop("is_awq", False)
        self.is_compressed_tensors = kw.pop("is_compressed_tensors", True)
        assert not kw, kw


# =================================================================================================
# Scheme dispatch -- must mirror create_moe_quant_method's ORDER, not just its cases
# =================================================================================================


def test_none_quant_is_unquantized():
    s = scheme_from_quant(None, compute_dtype_bytes=2)
    assert s.kind == SCHEME_UNQUANTIZED
    assert s.elem_bytes == 2


def test_fp8_wins_over_every_other_predicate():
    """`create_moe_quant_method` tests fp8 FIRST. A checkpoint that is both fp8-declared and
    int4-shaped must size as fp8, or ZAYA's 8 GB expert stack is sized as 4 GB and the arena is
    half what it needs."""
    q = FakeQuant(is_fp8_w8a8=True, is_int4=True, is_nvfp4=True, is_rxf=True)
    assert scheme_from_quant(q).kind == SCHEME_FP8


def test_fp8_experts_flag_alone_selects_fp8():
    """ZAYA passes `fp8_experts=config.quant.is_fp8_w8a8`; the flag alone must route."""
    assert scheme_from_quant(None, fp8_experts=True).kind == SCHEME_FP8


def test_nvfp4_wins_over_e2m1():
    """NVFP4 and MXFP4 share E2M1 weight codes and differ only in the scale (group 16 fp16 vs
    group 32 e8m0). Dispatching to MXFP4 for an NVFP4 checkpoint under-counts scale bytes 4x."""
    q = FakeQuant(is_nvfp4=True, weight_is_e2m1=True, group_size=16, is_int4=False)
    assert scheme_from_quant(q).kind == SCHEME_NVFP4
    assert scheme_from_quant(q).group_size == 16


def test_rxf_wins_over_e2m1():
    q = FakeQuant(is_rxf=True, weight_is_e2m1=True, is_int4=False)
    assert scheme_from_quant(q).kind == SCHEME_RXF


def test_int4_methods_split_by_checkpoint_layout():
    assert scheme_from_quant(FakeQuant(is_gptq=True, is_compressed_tensors=False,
                                       group_size=128)).kind == SCHEME_GPTQ
    assert scheme_from_quant(FakeQuant(is_awq=True, is_compressed_tensors=False,
                                       group_size=128)).kind == SCHEME_AWQ
    assert scheme_from_quant(FakeQuant()).kind == SCHEME_CT_INT4


def test_unsupported_scheme_raises_like_the_dispatcher():
    q = FakeQuant(is_int4=False, is_compressed_tensors=False, bits=3, method="mystery")
    with pytest.raises(ValueError, match="unsupported declared MoE quant scheme"):
        scheme_from_quant(q)


def test_int4_with_unknown_method_raises():
    q = FakeQuant(is_compressed_tensors=False)
    with pytest.raises(ValueError, match="unrecognised method"):
        scheme_from_quant(q)


# =================================================================================================
# Analytic byte model -- hand-computed against layers/moe.py container __init__s
# =================================================================================================

E, N, K = 8, 256, 512


def test_unquantized_is_plain_stacked_tensor():
    s = ExpertScheme(SCHEME_UNQUANTIZED, elem_bytes=2)
    got = analytic_gemm_bytes(s, E, N, K)
    assert got.total == E * N * K * 2
    assert got.scale == 0 and got.zero == 0


def test_fp8_is_one_byte_per_weight_plus_a_per_channel_f32_scale():
    """`_GroupedFP8Experts`: weight (E,N,K) f8_e4m3 + weight_scale (E,N,1) f32."""
    got = analytic_gemm_bytes(ExpertScheme(SCHEME_FP8, bits=8), E, N, K)
    assert got.weight == E * N * K
    assert got.scale == E * N * 4
    assert got.zero == 0


def test_gptq_matches_its_three_checkpoint_tensors():
    """qweight (E,K/8,N) i32 + scales (E,K/g,N) f16 + qzeros (E,K/g,N/8) i32."""
    g = 128
    got = analytic_gemm_bytes(ExpertScheme(SCHEME_GPTQ, bits=4, group_size=g), E, N, K)
    assert got.weight == E * (K // 8) * N * 4
    assert got.scale == E * (K // g) * N * 2
    assert got.zero == E * (K // g) * (N // 8) * 4


def test_awq_totals_equal_gptq_totals():
    """AWQ packs along N and GPTQ along K, but the three tensors hold the same byte count -- which
    is why `post_load`'s repack to the shared op layout is byte-invariant."""
    g = 128
    gptq = analytic_gemm_bytes(ExpertScheme(SCHEME_GPTQ, bits=4, group_size=g), E, N, K)
    awq = analytic_gemm_bytes(ExpertScheme(SCHEME_AWQ, bits=4, group_size=g), E, N, K)
    assert awq.total == gptq.total


def test_compressed_tensors_zeros_are_present_only_when_asymmetric_in_the_CHECKPOINT():
    """A symmetric CT checkpoint omits `weight_zero_point`; an asymmetric one ships (E,N/8,K/g).

    But that is a statement about the CHECKPOINT, not about what is RESIDENT. `post_load`'s
    `zp is None` arm allocates a real `torch.empty((E, G, N/pf), int32)` of 0x88, which has exactly
    the same element count as the asymmetric checkpoint's tensor -- so once loaded the two schemes
    hold the SAME bytes. The old assertion here (`asym.total > sym.total`) encoded the under-count
    that under-reserved the arena.
    """
    g = 32
    sym = analytic_gemm_bytes(ExpertScheme(SCHEME_CT_INT4, bits=4, group_size=g, sym=True),
                              E, N, K)
    asym = analytic_gemm_bytes(ExpertScheme(SCHEME_CT_INT4, bits=4, group_size=g, sym=False),
                               E, N, K)
    assert sym.zero == 0
    assert asym.zero == E * (N // 8) * (K // g) * 4
    assert asym.checkpoint_total > sym.checkpoint_total
    assert sym.post_load_resident == E * (K // g) * (N // 8) * 4
    assert asym.post_load_resident == 0
    assert sym.total == asym.total


def test_symmetric_ct_zeros_are_resident_but_not_part_of_the_granule():
    """`_zeros_op` is E bitwise-identical rows, so the granule walker classifies it replicated.

    That makes it a CAPACITY byte and not a BANDWIDTH byte -- exactly the distinction
    `ExpertStackBytes` calls a ~50x error when conflated. Charging it to the granule would inflate
    the projected step time; charging it to neither is the bug that under-reserves the arena.
    """
    d = post_load_delta_bytes(
        ExpertScheme(SCHEME_CT_INT4, bits=4, group_size=32, sym=True), E, N, K
    )
    assert d.resident == E * (K // 32) * (N // 8) * 4
    assert d.granule == 0


def test_mxfp4_scale_is_one_byte_e8m0_on_disk_and_two_bytes_resident():
    """The whole reason NVFP4 and MXFP4 must not be conflated, in one assertion -- plus the fact
    that MXFP4's E8M0 uint8 scale is widened to fp16 by `post_load` on BOTH arms
    (`convert_mxfp4_moe` -> `_scales_op`, `mxfp4_to_w_rep_moe` -> `_scales_rd`), so the resident
    scale is 2 bytes even though the checkpoint's is 1."""
    mx = analytic_gemm_bytes(ExpertScheme(SCHEME_MXFP4, bits=4, group_size=32), E, N, K)
    nv = analytic_gemm_bytes(ExpertScheme(SCHEME_NVFP4, bits=4, group_size=16), E, N, K)
    assert mx.weight == nv.weight == E * N * (K // 2)
    assert mx.scale == E * N * (K // 32) * 1
    assert nv.scale == E * N * (K // 16) * 2
    assert nv.scale == 4 * mx.scale
    # post_load widens E8M0 -> fp16: +1 byte per group scale, per expert, and it IS per-expert so
    # it is charged to the granule as well as to residency.
    assert mx.post_load_resident == E * N * (K // 32)
    assert mx.post_load_granule == N * (K // 32)
    assert mx.total == mx.checkpoint_total + E * N * (K // 32)
    assert nv.post_load_resident == 0  # NVFP4's folded scale is already fp16 on disk


@pytest.mark.parametrize(
    "scheme",
    [
        ExpertScheme(SCHEME_UNQUANTIZED, bits=16),
        ExpertScheme(SCHEME_FP8, bits=8),
        ExpertScheme(SCHEME_RXF, bits=4, group_size=32),
        ExpertScheme(SCHEME_NVFP4, bits=4, group_size=16),
        ExpertScheme(SCHEME_GPTQ, bits=4, group_size=128),
        ExpertScheme(SCHEME_AWQ, bits=4, group_size=128),
        ExpertScheme(SCHEME_CT_INT4, bits=4, group_size=32, sym=False),
    ],
    ids=lambda s: f"{s.kind}{'-asym' if not s.sym else ''}",
)
def test_every_other_container_is_byte_invariant_across_post_load(scheme):
    """The seven containers whose `post_load` is a permutation or an equal-size repack.

    Pinned as a table so that adding a format with a non-invariant `post_load` and forgetting
    `post_load_delta_bytes` fails HERE, at the byte model, rather than at 3 a.m. as an arena
    exhaustion that falls back to `hipMalloc` and puts host-budgeted weights in VRAM.
    """
    gb = analytic_gemm_bytes(scheme, E, N, K)
    assert gb.post_load_resident == 0 and gb.post_load_granule == 0
    assert gb.total == gb.checkpoint_total


def test_rxf_is_span32_by_construction():
    got = analytic_gemm_bytes(ExpertScheme(SCHEME_RXF, bits=4, group_size=32), E, N, K)
    assert got.weight == E * N * (K // 2)
    assert got.scale == E * N * (K // 32) * 2


def test_indivisible_shape_raises_rather_than_rounding():
    """The container asserts the same divisibility. Rounding here would under-count the arena."""
    with pytest.raises(ValueError, match="divisible"):
        analytic_gemm_bytes(ExpertScheme(SCHEME_CT_INT4, bits=4, group_size=32), E, N, K + 1)


def test_zero_dimensions_raise():
    with pytest.raises(ValueError, match="positive"):
        analytic_gemm_bytes(ExpertScheme(SCHEME_UNQUANTIZED), E, 0, K)


def test_group_size_zero_is_rejected_at_construction():
    with pytest.raises(ValueError, match="group_size"):
        ExpertScheme(SCHEME_CT_INT4, bits=4, group_size=0)


# =================================================================================================
# The layer-level entry point
# =================================================================================================


def test_expert_stack_bytes_uses_moelayer_shapes():
    """w13 is (E, 2*I_part, H) and w2 is (E, H, I_part) -- MoELayer.__init__'s two create_experts
    calls. Swapping them would keep the total the same for a symmetric format and change it for
    every asymmetric one, so it is asserted per-GEMM, not on the total."""
    q = FakeQuant(group_size=32, sym=True)
    hidden, inter = 2048, 384
    got = expert_stack_bytes(
        quant=q, num_local_experts=512, hidden_size=hidden,
        intermediate_size_per_partition=inter, prefer_meta=False,
    )
    s = scheme_from_quant(q)
    assert got.w13 == analytic_gemm_bytes(s, 512, 2 * inter, hidden).total
    assert got.w2 == analytic_gemm_bytes(s, 512, hidden, inter).total
    assert got.source == "analytic"
    assert got.total == got.w13 + got.w2


def test_granule_without_a_walker_is_the_CHECKPOINT_flat_share():
    """With no meta walk there is no proof of expert-invariance, so the granule is the flat share of
    the CHECKPOINT tensors -- every one of which carries E on dim 0 by construction -- plus only the
    per-expert part of the post_load delta.

    Deliberately NOT `total / E`: `total` now includes CT-symmetric's `_zeros_op`, which is E
    bitwise-identical rows and is read once per layer, not once per routed expert. Dividing the
    resident total would bill a capacity byte as a bandwidth byte and inflate the projected step
    time -- the same conflation as under-reserving the arena, pointing the other way. For every
    byte-invariant scheme the two are identical.
    """
    e, hidden, inter = 512, 2048, 384
    got = expert_stack_bytes(
        quant=FakeQuant(), num_local_experts=e, hidden_size=hidden,
        intermediate_size_per_partition=inter, prefer_meta=False,
    )
    s = scheme_from_quant(FakeQuant())
    w13 = analytic_gemm_bytes(s, e, 2 * inter, hidden)
    w2 = analytic_gemm_bytes(s, e, hidden, inter)
    assert got.granule_bytes == (w13.checkpoint_total + w2.checkpoint_total) // e
    assert got.granule_bytes * e < got.total  # the replicated zeros are resident but not routed

    # An invariant scheme is unchanged: the flat share of `total` and of `checkpoint_total` agree.
    fp8 = expert_stack_bytes(
        quant=None, fp8_experts=True, num_local_experts=e, hidden_size=hidden,
        intermediate_size_per_partition=inter, prefer_meta=False,
    )
    assert fp8.granule_bytes == fp8.total // e


def test_prefer_meta_degrades_to_analytic_when_torch_is_unusable():
    """The planner must still produce a plan on a host where the layer stack will not import; it
    just has to say so, which `source` does."""
    got = expert_stack_bytes(
        quant=FakeQuant(), num_local_experts=64, hidden_size=1024,
        intermediate_size_per_partition=256, prefer_meta=True,
    )
    assert got.source in ("meta", "analytic")
    if not TORCH_IS_REAL:
        assert got.source == "analytic"


def test_zero_experts_rejected():
    with pytest.raises(ValueError, match="num_local_experts"):
        expert_stack_bytes(quant=None, num_local_experts=0, hidden_size=16,
                           intermediate_size_per_partition=16, prefer_meta=False)


def test_tp_sharding_halves_the_stack():
    """The single most consequential input to the capacity arithmetic: I_part = I // tp."""
    kw = {"quant": FakeQuant(), "num_local_experts": 512, "hidden_size": 2048,
          "prefer_meta": False}
    full = expert_stack_bytes(intermediate_size_per_partition=768, **kw)
    half = expert_stack_bytes(intermediate_size_per_partition=384, **kw)
    assert full.total == pytest.approx(2 * half.total, rel=1e-9)


# =================================================================================================
# GENERALITY -- what a NEW weight format has to implement here. The answer must be NOTHING.
#
# `KERNEL_CORE_POLICY.md` makes a new weight format a WLoad policy on the existing shared core, so
# adding one is an edit to `layers/moe.py` and nowhere else. `resolve_weight_plan` runs on the boot
# path of EVERY serve (offload requested or not), so anything that raises out of the scheme
# transcription below takes the whole serve down for a checkpoint the model layer builds fine.
# =================================================================================================


class UnknownQuant:
    """A quant config for a format added AFTER this file was written: none of the predicates
    `scheme_from_quant` transcribes match, but `create_moe_quant_method` would route it."""

    method = "some-new-format"
    bits = 6
    group_size = 64
    sym = True
    weight_type = "float"
    ct_groups = ()
    is_fp8_w8a8 = False
    is_nvfp4 = False
    is_rxf = False
    weight_is_e2m1 = False
    is_int4 = False
    is_gptq = False
    is_awq = False
    is_compressed_tensors = False


class _FakeSpec:
    """Stands in for a `granule.GranuleSpec` from the meta walk."""

    def __init__(self, total, granule):
        self.total_bytes = total
        self.granule_bytes = granule


def test_an_unrecognised_scheme_is_strict_only():
    """Strict resolution still raises -- that contract is what the dispatch-order tests above pin."""
    with pytest.raises(ValueError, match="unsupported declared MoE quant scheme"):
        scheme_from_quant(UnknownQuant())


def test_an_unrecognised_scheme_degrades_instead_of_raising():
    got = scheme_from_quant(UnknownQuant(), strict=False)
    assert got.kind == SCHEME_UNKNOWN


def test_a_bad_group_size_also_degrades_rather_than_escaping_post_init():
    """`ExpertScheme.__post_init__` refuses group_size<=0 for the grouped kinds. Non-strict must
    swallow THAT too -- otherwise the degradation has a second door the planner falls through."""
    q = FakeQuant(group_size=0)
    with pytest.raises(ValueError, match="positive group_size"):
        scheme_from_quant(q)
    assert scheme_from_quant(q, strict=False).kind == SCHEME_UNKNOWN


def test_the_unknown_arm_raises_valueerror_not_zerodivision():
    """`pf = 32 // scheme.bits` on a bits=0 scheme is a ZeroDivisionError, which
    `expert_stack_bytes` does not catch -- so the unknown scheme must be refused BEFORE it."""
    s = ExpertScheme(SCHEME_UNKNOWN, bits=0, group_size=0)
    with pytest.raises(ValueError, match="no analytic byte model"):
        analytic_gemm_bytes(s, 8, 256, 128)


def test_an_unknown_format_is_sized_by_the_meta_model_not_refused(monkeypatch):
    """THE POINT OF THIS SECTION. A format `layers/moe.py` can build but this file cannot name must
    still yield a plan: `meta_gemm_spec` builds the real container, so it answers on its own."""
    import minisgl.weights.sizing as sizing

    monkeypatch.setattr(sizing, "meta_gemm_spec", lambda *a, **k: _FakeSpec(4096, 64))
    got = sizing.expert_stack_bytes(
        quant=UnknownQuant(), num_local_experts=64, hidden_size=1024,
        intermediate_size_per_partition=256, prefer_meta=True,
    )
    assert got.source == "meta"
    assert got.scheme == SCHEME_UNKNOWN
    assert got.total == 8192  # two GEMMs
    assert got.granule_bytes == 128
    assert got.agreement is None  # no analytic arm ran, so there is nothing to cross-check


def test_when_both_models_refuse_the_error_names_both(monkeypatch):
    """The remaining hard failure must not send a reader off to fix the transcription when the real
    cause is that the meta model could not run."""
    import minisgl.weights.sizing as sizing

    monkeypatch.setattr(sizing, "meta_gemm_spec", lambda *a, **k: None)
    with pytest.raises(ValueError, match="BOTH byte models refused"):
        sizing.expert_stack_bytes(
            quant=UnknownQuant(), num_local_experts=64, hidden_size=1024,
            intermediate_size_per_partition=256, prefer_meta=True,
        )


def test_an_unknown_scheme_never_claims_ep_support():
    """`SCHEME_SUPPORTS_EP` has no `unknown` key on purpose: `MoEQuantMethod.supports_ep` defaults to
    False, so a method that does not override it does NOT shard -- and a false 'EP supported' halves
    the local expert count, which is the error that turns an infeasible plan feasible."""
    from minisgl.weights.sizing import SCHEME_UNKNOWN as _UNK
    from minisgl.weights.sizing import scheme_supports_ep

    assert _UNK not in SCHEME_SUPPORTS_EP
    assert scheme_supports_ep(UnknownQuant()) is False
