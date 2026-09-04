"""How many bytes one MoE expert-stack container occupies, derived from config alone.

The placement planner has to answer "does the host arena fit" BEFORE `load_state_dict` runs --
engine.py opens shards with `device=str(device)` and `post_load` repacks on GPU, so a 68 GB
checkpoint OOMs long before anything could be measured. A capacity failure must abort in seconds,
not after a seven-minute load.

Two independent byte models are provided and they check each other:

  * `analytic_gemm_bytes` -- closed-form, stdlib only, one arm per expert-container class in
    `layers/moe.py`. Runs anywhere (no torch), which is what makes the planner unit-testable.
  * `meta_gemm_bytes` -- builds the REAL container through `create_moe_quant_method(...)
    .create_experts(...)` under `torch.device("meta")` and sums tensor bytes. Zero allocation, zero
    GPU. This is the authoritative shape, because it is literally the code that will allocate.

`expert_stack_bytes` prefers the meta model, falls back to the analytic one, and records which it
used plus the disagreement ratio when both are available. A drift between them means a container
changed and this file did not -- exactly the "fix lands on one sibling only" failure this repo has
been burned by, made visible instead of silent.

BOTH MODEL `__init__` SHAPES, AND THAT IS NOT WHAT THE ARENA HOLDS. The arena is populated after
`post_load()`, so every number here is corrected by `post_load_delta_bytes` before it is published.
Most containers' `post_load` is a permutation or an equal-size repack (`(E,K/pf,N) i32 ->
(E,N,K/pf) i32`), but TWO of the nine shipped ones are not, and both were sized as invariant until
2026-09-03: `_GroupedCompressedTensorsExperts` SYMMETRIC materialises a real `_zeros_op` that
`__init__` never allocated (+3.1%), and `_GroupedMxFp4Experts` widens its E8M0 uint8 scale to fp16
(+6.25% at g=32). Under-counting either one under-reserves the arena, whereupon
`ArenaMemPool._alloc` falls back to `hipMalloc` and host-budgeted weights land silently in VRAM --
and, earlier and worse, `required_device_bytes()` reports a device tier that is too small while
`feasible` says yes. See `PostLoadDelta`.

Even so, a number from this module is an ESTIMATE: the register-direct arms (`_w_rep`,
`_scales_rd`) go through kernels this file cannot see, and a padded repack would change the total.
It must be re-checked against `granule.derive_granule_spec` on the LIVE containers after
`post_load()` before anything is allocated against it. Never treat a number from here as measured.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Optional, Tuple

# ---------------------------------------------------------------------------------------------
# Scheme identification. Mirrors `layers/moe.py::create_moe_quant_method` ORDER EXACTLY -- the
# dispatch there is `fp8 -> unquantized -> nvfp4 -> rxf -> e2m1 -> int4 -> raise`, and getting the
# order wrong silently sizes an NVFP4 checkpoint as if it were MXFP4 (different group size, 2x the
# scale bytes). Keyed on the same QuantConfig predicates, never on a model name.
# ---------------------------------------------------------------------------------------------

SCHEME_UNQUANTIZED = "unquantized"
SCHEME_FP8 = "fp8"
SCHEME_NVFP4 = "nvfp4"
SCHEME_RXF = "rxf"
SCHEME_MXFP4 = "mxfp4"
SCHEME_GPTQ = "gptq"
SCHEME_AWQ = "awq"
SCHEME_CT_INT4 = "ct_int4"
# A scheme `layers/moe.py` can build but THIS FILE has no closed form for -- i.e. a weight format
# added since, which under `KERNEL_CORE_POLICY.md` is a WLoad policy on the existing core and must
# not require anything of downstream infrastructure. The analytic model is a hand transcription of
# nine container `__init__`s and it WILL lag; the meta model builds the real container through
# `create_moe_quant_method` and cannot. So an unidentified scheme degrades to "meta only" instead of
# raising, because raising out of `scheme_from_quant` aborts `resolve_weight_plan` -- which sits on
# the boot path of EVERY serve, offload or not -- for a checkpoint that would otherwise serve fine.
SCHEME_UNKNOWN = "unknown"

_INT4_SCHEMES = (SCHEME_GPTQ, SCHEME_AWQ, SCHEME_CT_INT4)

# Mirrors the `supports_ep` CLASS ATTRIBUTE on each `MoEQuantMethod` in `layers/moe.py`, because
# `MoELayer.__init__` (moe.py:1004) is
#     is_ep_enabled() AND self._moe_method.supports_ep AND not force_no_ep
# -- the quant method holds a VETO over `--enable-ep`, and a planner that reads only the engine-level
# toggle shards experts the engine will not shard.
#
# TWO of the six methods veto, and both are easy to miss because neither says so in its name:
#   * `_RXFMoEMethod`         `supports_ep = False` (moe.py:805) -- "no precomputed-topk shard route".
#   * `_UnquantizedMoEMethod` (moe.py:585) never sets it at all, so it INHERITS the base-class
#     default `supports_ep: bool = False` (moe.py:528). Silence, not a decision -- which is exactly
#     why it is spelled out here rather than assumed.
# The live class attribute is preferred whenever `layers.moe` is importable; this table is the
# torch-free fallback and the thing `test_weight_plan_tp` pins the live classes against.
SCHEME_SUPPORTS_EP: Dict[str, bool] = {
    SCHEME_FP8: True,  # _FP8MoEMethod   (moe.py:846)
    SCHEME_UNQUANTIZED: False,  # _UnquantizedMoEMethod inherits the base False (moe.py:528/585)
    SCHEME_NVFP4: True,  # _NvFp4MoEMethod (moe.py:770)
    SCHEME_RXF: False,  # _RXFMoEMethod   (moe.py:805) -- the explicit veto
    SCHEME_MXFP4: True,  # _MxFp4MoEMethod (moe.py:707)
    SCHEME_GPTQ: True,  # _W4A8MoEMethod  (moe.py:617)
    SCHEME_AWQ: True,
    SCHEME_CT_INT4: True,
}

# Which `tools/cpu_moe/wload.hpp` policy — if any — can read this scheme's bytes on the CPU tier.
# A scheme absent from this table has NO CPU expert core, and `cpu_wload_policy` returning None is
# what stops `plan.resolve_weight_plan` from placing its layers on `StackKind.CPU`.
#
# THIS IS A SIZING FACT AND THAT IS WHY IT LIVES HERE. The CPU tier's resident bytes are NOT the
# device tier's: the AVX-512 cores read the checkpoint's own e4m3 group-scale BYTE where the GPU
# path holds an fp16-folded scale, so a CPU-placed NVFP4 layer is 0.9x the bytes (2,764,800 vs
# 3,072,000 per expert on the target shape). The ratio is a property of the SCALE ENCODING, which
# is exactly the quantity this module models, and putting it anywhere else would let the capacity
# arithmetic and the byte model drift apart.
#
# The value is the WLoad policy NAME, deliberately, not a bytes-per-weight number:
# `cpu_tier.CpuTierPrior.layout_fraction_for(policy, repacked=...)` owns the ratio and refuses to
# grant it unless a repacker actually ran. Adding a weight format here is the WLoad-policy edit
# `KERNEL_CORE_POLICY.md` describes — a table row, never a new kernel and never a new tier.
_CPU_WLOAD_BY_SCHEME: Dict[str, str] = {
    SCHEME_NVFP4: "vnni_nvfp4_e4m3_g16",
    SCHEME_MXFP4: "mxfp4_e8m0_g32",
}

# THE SECOND SPELLING OF THE SAME FACT, and it is not a duplicate table -- it answers a DIFFERENT
# question, from a different namespace, with a different (and more truthful) answer.
#
# WHY IT HAS TO EXIST. `plan.cpu_tier_gate` reads `diagnostics["schemes"]`, and the two planner
# paths fill that key from two namespaces:
#   * `build_planned_layers` (the CONFIG fallback) puts SCHEME KINDS in it -- "nvfp4", "fp8".
#   * `observed_planned_layers` (the MODEL path, which is what every real serve takes) puts
#     `GranuleSpec.kind` in it, and that is `type(container).__name__` -- "_GroupedNvFp4Experts".
# Keying the gate on scheme kinds alone therefore refused the CPU tier on 100% of real serves while
# passing every unit test, because the tests drive the config path. That was a live defect: the
# refusal message even named the container class as if it were a scheme.
#
# WHY THE VALUES DIFFER FROM THE TABLE ABOVE. A scheme names what the CHECKPOINT holds; a container
# names what is RESIDENT after `post_load`. For NVFP4 those are different layouts:
# `_GroupedNvFp4Experts.post_load` folds the e4m3 block scale and the per-tensor global into ONE
# fp16 per-group scale and deletes the checkpoint copies (layers/moe.py:405-412). So the bytes a
# CPU tier can actually read today are policy E (`vnni_nvfp4_fp16_g16`, 0.625 B/weight), not
# policy D (`vnni_nvfp4_e4m3_g16`, 0.5625 B/weight and ~1800x more accurate). D is what a plan gets
# once a bake-time repacker re-reads the raw scales -- which is exactly what
# `resolve_weight_plan(cpu_repacked=True)` is the gate for, and which nothing implements. Reporting
# D here would claim a 10% capacity win the resident bytes do not have.
_CPU_WLOAD_BY_CONTAINER: Dict[str, str] = {
    "_GroupedNvFp4Experts": "vnni_nvfp4_fp16_g16",
}


def cpu_wload_policy_for_kind(kind: Any) -> Optional[str]:
    """The CPU core's WLoad policy for either spelling of a layer's format, or None.

    Accepts a scheme kind ("nvfp4"), an `ExpertScheme`, or a container class name
    ("_GroupedNvFp4Experts"), because `diagnostics["schemes"]` carries the first from the config
    path and the last from the model path. Returning None is still a REFUSAL the planner honours.
    """
    k = getattr(kind, "kind", kind)
    if not isinstance(k, str):
        return None
    return _CPU_WLOAD_BY_CONTAINER.get(k) or _CPU_WLOAD_BY_SCHEME.get(k)


def cpu_wload_policy(scheme: Any) -> Optional[str]:
    """The CPU expert core's WLoad policy for `scheme`, or None if there is no CPU core for it.

    `scheme` may be an `ExpertScheme` or a bare scheme-kind string. Returning None is a REFUSAL
    the planner must honour: a CPU-tier layer whose bytes no CPU core can decode would bake
    perfectly-valid tensors into pageable memory and then have nothing able to read them, which
    surfaces as a dead layer at the first token rather than as a boot error.

    NVFP4 maps to the VNNI (int8-activation) policy because that is the one that fits the core
    budget — see `weights/cpu_tier.py`. The fp32 policy `nvfp4_e4m3_g16` reads the same bytes and
    remains the correctness ORACLE; it is reachable by passing `policy=ACT_FP32` to
    `cpu_tier.project_cpu_tier`, and it is not a serving option on an 8-core box because it needs
    six of them.
    """
    kind = getattr(scheme, "kind", scheme)
    if not isinstance(kind, str):
        return None
    return _CPU_WLOAD_BY_SCHEME.get(kind)


def scheme_supports_ep(quant: Any, *, fp8_experts: bool = False) -> bool:
    """Would `MoELayer` actually EP-shard experts built from this quant config?

    Answers the `self._moe_method.supports_ep` conjunct of `MoELayer.__init__` (moe.py:1004).
    Prefers the LIVE class attribute (it cannot drift) and falls back to `SCHEME_SUPPORTS_EP` on a
    torch-free host.

    Returns **False** when the scheme cannot be identified at all. That is the conservative
    direction for a capacity planner: a false "EP supported" divides the local expert count by
    `ep_size` and under-counts per-step traffic by the same factor, which is the error that makes an
    infeasible plan look feasible. A false "not supported" only over-counts.

    Every `MoEQuantMethod.__init__` in `layers/moe.py` only stores the config, so this allocates
    nothing -- but it is built under `torch.device("meta")` anyway, so the "safe to call at boot,
    before any device is selected" property is structural rather than a transcription that a future
    `__init__` could quietly falsify.
    """
    try:
        import torch
        from minisgl.layers.moe import create_moe_quant_method

        with torch.device("meta"):
            return bool(create_moe_quant_method(quant, fp8_experts=fp8_experts).supports_ep)
    except Exception:  # pragma: no cover - torch-free host, or an unroutable scheme
        pass
    try:
        kind = scheme_from_quant(quant, fp8_experts=fp8_experts, strict=False).kind
    except Exception:
        return False
    # SCHEME_UNKNOWN is deliberately absent from the table, so a format added since this file was
    # written falls to the conservative `False` rather than inheriting some neighbour's answer.
    return bool(SCHEME_SUPPORTS_EP.get(kind, False))


@dataclass(frozen=True)
class ExpertScheme:
    """The storage facts that decide bytes/element. Deliberately NOT a QuantConfig: the planner
    must be hashable, JSON-serialisable and torch-free so its fingerprint is stable across ranks."""

    kind: str
    bits: int = 16
    group_size: int = 0
    sym: bool = True
    elem_bytes: int = 2  # compute dtype itemsize; only the unquantized arm reads it

    def __post_init__(self) -> None:
        if self.kind in _INT4_SCHEMES or self.kind in (SCHEME_MXFP4, SCHEME_NVFP4):
            if self.group_size <= 0:
                raise ValueError(f"{self.kind} needs a positive group_size, got {self.group_size}")


def scheme_from_quant(
    quant: Any, *, fp8_experts: bool = False, compute_dtype_bytes: int = 2, strict: bool = True
) -> ExpertScheme:
    """Map a QuantConfig (or None) to an ExpertScheme, in `create_moe_quant_method` order.

    Duck-typed on purpose: `QuantConfig` lives behind a transformers import, and the planner must be
    importable (and testable) without it. Any object exposing the same predicates works.

    `strict=False` returns `ExpertScheme(SCHEME_UNKNOWN)` instead of raising on a scheme this file
    does not recognise. That is what the planner passes, and the reason is a repo rule rather than
    politeness: a new weight format is a WLoad policy on the shared core
    (`rdna4-hip-kernels/KERNEL_CORE_POLICY.md`), so adding one must not require an edit here. This
    function is a hand transcription of `create_moe_quant_method`'s predicates and it lags by
    construction; `meta_gemm_spec` builds the actual container and does not. Raising here would take
    the whole serve down -- `resolve_weight_plan` runs on every boot, offload or not -- for a
    checkpoint whose experts `layers/moe.py` allocates perfectly well.
    """
    if strict:
        return _scheme_from_quant(quant, fp8_experts, compute_dtype_bytes)
    try:
        return _scheme_from_quant(quant, fp8_experts, compute_dtype_bytes)
    except (ValueError, TypeError, AttributeError):
        # Catches BOTH refusals: the two `raise ValueError`s below, and `ExpertScheme.__post_init__`
        # rejecting a group_size this file assumes is positive. Either way the answer is the same --
        # this file cannot describe the format, so let the meta model try.
        return ExpertScheme(SCHEME_UNKNOWN, bits=0, group_size=0, sym=True,
                            elem_bytes=compute_dtype_bytes)


def _scheme_from_quant(quant: Any, fp8_experts: bool, compute_dtype_bytes: int) -> ExpertScheme:
    if fp8_experts or (quant is not None and bool(getattr(quant, "is_fp8_w8a8", False))):
        return ExpertScheme(SCHEME_FP8, bits=8, elem_bytes=compute_dtype_bytes)
    if quant is None:
        return ExpertScheme(SCHEME_UNQUANTIZED, bits=8 * compute_dtype_bytes,
                            elem_bytes=compute_dtype_bytes)
    g = int(getattr(quant, "group_size", 0) or 0)
    bits = int(getattr(quant, "bits", 4) or 4)
    sym = bool(getattr(quant, "sym", True))
    if bool(getattr(quant, "is_nvfp4", False)):
        # The loader folds NVFP4's e4m3 block scale + fp32 global into ONE fp16 per-group scale at
        # the leaf, so by load time the container is (E,N,K/2) u8 + (E,N,K/16) fp16.
        return ExpertScheme(SCHEME_NVFP4, bits=4, group_size=g or 16, sym=True,
                            elem_bytes=compute_dtype_bytes)
    if bool(getattr(quant, "is_rxf", False)):
        return ExpertScheme(SCHEME_RXF, bits=4, group_size=32, sym=True,
                            elem_bytes=compute_dtype_bytes)
    if bool(getattr(quant, "weight_is_e2m1", False)):
        return ExpertScheme(SCHEME_MXFP4, bits=4, group_size=g or 32, sym=True,
                            elem_bytes=compute_dtype_bytes)
    if bool(getattr(quant, "is_int4", False)):
        if bool(getattr(quant, "is_gptq", False)):
            kind = SCHEME_GPTQ
        elif bool(getattr(quant, "is_awq", False)):
            kind = SCHEME_AWQ
        elif bool(getattr(quant, "is_compressed_tensors", False)):
            kind = SCHEME_CT_INT4
        else:
            raise ValueError(
                f"int4 MoE with an unrecognised method {getattr(quant, 'method', '?')!r}; "
                "layers/moe.py::_W4A8MoEMethod would refuse it too"
            )
        return ExpertScheme(kind, bits=bits, group_size=g, sym=sym,
                            elem_bytes=compute_dtype_bytes)
    raise ValueError(
        f"unsupported declared MoE quant scheme (method={getattr(quant, 'method', '?')!r}, "
        f"bits={bits}, weight_type={getattr(quant, 'weight_type', '?')!r}); "
        "create_moe_quant_method raises on this too"
    )


# ---------------------------------------------------------------------------------------------
# Analytic byte model. One arm per container in layers/moe.py, transcribed from its __init__.
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PostLoadDelta:
    """Bytes `post_load()` ADDS to (or removes from) a container, relative to its `__init__` shapes.

    THIS IS NOT AN ACCOUNTING NICETY. The arena is reserved from the plan's `host_resident_bytes`
    and then populated with the tensors that exist AFTER `post_load()`, so any tensor `post_load`
    synthesises out of nothing is a byte the arena was never asked to hold. The bump allocator then
    runs dry, `ArenaMemPool._alloc` falls back to `hipMalloc`, and weights budgeted as host-resident
    land silently in VRAM. Worse and earlier: `required_device_bytes()` and `feasible` are computed
    from the same under-count, so an infeasible plan reports FEASIBLE and the operator is told to
    grant a device tier that is too small.

    Two of the nine shipped containers are NOT byte-invariant across `post_load` (traced against
    `layers/moe.py`, 2026-09-03):

      * `_GroupedCompressedTensorsExperts`, SYMMETRIC. `__init__` allocates no zero-points; the
        `zp is None` arm of `post_load` allocates a REAL `torch.empty((E, G, N//pf), int32)` and
        fills it with 0x88. It is materialized, not broadcast, and `moe_interpose._plan_items`
        copies it into the arena like any other tensor. On the target-shaped stack that is +3.1%
        (~864 MiB/rank over 48 layers) — 13x the accounting gate's 64 MiB tolerance.
      * `_GroupedMxFp4Experts`, BOTH arms. `weight_scale` is uint8 E8M0 on disk and becomes an fp16
        `_scales_op` (LDS path) / `_scales_rd` (regdirect path). That is +1 byte per group scale,
        i.e. +E*N*K/g — 6.25% of the container at g=32.

    `resident` is the capacity term; `granule` is the per-expert bandwidth term and is 0 whenever
    the added bytes are expert-INVARIANT (the CT zeros are E identical rows, which the granule
    walker classifies as replicated and excludes from the route). Conflating the two is the ~50x
    error `ExpertStackBytes` warns about, so they are separate fields rather than one number.

    Every other container's `post_load` is a permutation or a same-size repack: GPTQ/AWQ
    `(E,K/pf,N)->(E,N,K/pf)`, CT-asymmetric `(E,N/pf,G)->(E,G,N/pf)`, NVFP4 `u8 (E,N,K/2)` ->
    `i32 (E,N,K/8)`, fp8 `view(uint8)` + `squeeze(-1)`, RXF `transpose(1,2)`. Those are 0 here, and
    the `_w_rep`/`_scales_rd` register-direct repacks go through kernels this file cannot see — if
    one of those ever pads, only the post-`post_load()` reconciliation can catch it.
    """

    resident: int = 0
    granule: int = 0
    note: str = ""


def post_load_delta_bytes(
    scheme: ExpertScheme, num_experts: int, out_features: int, in_features: int
) -> PostLoadDelta:
    """Bytes `post_load()` adds to one container beyond its `__init__` shapes. See `PostLoadDelta`."""
    E, N, K = int(num_experts), int(out_features), int(in_features)
    g = scheme.group_size
    if scheme.kind == SCHEME_CT_INT4 and scheme.sym:
        pf = 32 // scheme.bits
        _require_div(K, g, "CT int4 K")
        _require_div(N, pf, "CT int4 N")
        # `torch.empty((E, G, N//pf), int32)` filled with 0x88 — E bitwise-identical rows, so it is
        # resident but NOT part of the granule.
        return PostLoadDelta(
            resident=E * (K // g) * (N // pf) * 4,
            granule=0,
            note=f"_zeros_op E*K/{g}*N/{pf}*4 synthesised in post_load (expert-invariant)",
        )
    if scheme.kind == SCHEME_MXFP4:
        _require_div(K, g, "MXFP4 K")
        # E8M0 uint8 -> fp16 group scale: +1 byte per group, per expert, on BOTH the LDS
        # (`_scales_op`) and the register-direct (`_scales_rd`) arms.
        return PostLoadDelta(
            resident=E * N * (K // g),
            granule=N * (K // g),
            note=f"_scales_op E8M0 u8 -> fp16, +E*N*K/{g}*1",
        )
    return PostLoadDelta()


@dataclass(frozen=True)
class GemmBytes:
    """Bytes for ONE expert GEMM container (w13 or w2), stacked over `num_experts`.

    `total` is the RESIDENT figure — what the arena must hold after `post_load()`, which is the only
    number a capacity gate may use. `checkpoint_total` is the pre-`post_load` figure, kept separate
    so the meta model (which builds `__init__` shapes and never runs `post_load`) can be compared
    against this one like for like.
    """

    weight: int
    scale: int
    zero: int
    num_experts: int
    scheme: str
    formula: str
    post_load_resident: int = 0
    post_load_granule: int = 0
    post_load_note: str = ""

    @property
    def checkpoint_total(self) -> int:
        """Bytes the container holds as `__init__` declares it, before `post_load()` runs."""
        return self.weight + self.scale + self.zero

    @property
    def total(self) -> int:
        return self.checkpoint_total + self.post_load_resident

    @property
    def per_expert(self) -> float:
        return self.total / self.num_experts if self.num_experts else 0.0


def analytic_gemm_bytes(
    scheme: ExpertScheme, num_experts: int, out_features: int, in_features: int
) -> GemmBytes:
    """Closed-form RESIDENT container size. N = out_features, K = in_features, PER EXPERT.

    Transcribed one-to-one from the container `__init__`s in `layers/moe.py`; the comment on each
    arm names the tensors so a container change can be diffed against this file by eye. A scale or
    zero-point tensor left out here is a granule that travels without its scale later -- silent
    wrong numbers, no crash -- so nothing is rounded away as "negligible".

    `.checkpoint_total` is the `__init__` figure; `.total` adds `post_load_delta_bytes`, because the
    arena holds what exists AFTER `post_load()` and that is the only number a capacity gate may use.
    """
    gb = _checkpoint_gemm_bytes(scheme, num_experts, out_features, in_features)
    d = post_load_delta_bytes(scheme, num_experts, out_features, in_features)
    if not d.resident and not d.granule:
        return gb
    return replace(
        gb,
        post_load_resident=d.resident,
        post_load_granule=d.granule,
        post_load_note=d.note,
        formula=f"{gb.formula} [+post_load {d.note}]",
    )


def _checkpoint_gemm_bytes(
    scheme: ExpertScheme, num_experts: int, out_features: int, in_features: int
) -> GemmBytes:
    """The `__init__`-declared container size, before `post_load()` runs. One arm per container."""
    E, N, K = int(num_experts), int(out_features), int(in_features)
    if E <= 0 or N <= 0 or K <= 0:
        raise ValueError(f"expert stack needs positive E/N/K, got E={E} N={N} K={K}")
    kind = scheme.kind
    g = scheme.group_size

    if kind == SCHEME_UNKNOWN or scheme.bits <= 0:
        # Refuse EARLY and as a ValueError. Falling through to `pf = 32 // scheme.bits` below would
        # raise ZeroDivisionError, which `expert_stack_bytes` does not catch -- so an unrecognised
        # format would still take the boot down through a second door after
        # `scheme_from_quant(strict=False)` closed the first.
        raise ValueError(
            f"no analytic byte model for scheme {kind!r} (bits={scheme.bits}). This file is a hand "
            f"transcription of the container __init__s in layers/moe.py; a format added since is "
            f"expected to land here eventually, but the meta model sizes it correctly meanwhile."
        )
    if kind == SCHEME_UNQUANTIZED:
        # torch.empty(E, N, K) in the compute dtype. No scales.
        return GemmBytes(E * N * K * scheme.elem_bytes, 0, 0, E, kind,
                         f"E*N*K*{scheme.elem_bytes}")
    if kind == SCHEME_FP8:
        # weight (E,N,K) f8_e4m3 + weight_scale (E,N,1) f32 -- per-OUTPUT-CHANNEL, not per group.
        return GemmBytes(E * N * K, E * N * 4, 0, E, kind, "E*N*K*1 + E*N*4")
    if kind == SCHEME_RXF:
        # weight_packed (E,N,K/2) u8 + weight_scale (E,N,K/32) f16. Span-32 by construction.
        _require_div(K, 32, "RXF K")
        return GemmBytes(E * N * (K // 2), E * N * (K // 32) * 2, 0, E, kind,
                         "E*N*K/2 + E*N*K/32*2")
    if kind == SCHEME_MXFP4:
        # weight_packed (E,N,K/2) u8 + weight_scale (E,N,K/g) u8 (E8M0 exponent, 1 byte).
        # post_load widens that scale to fp16 on BOTH arms -- see `post_load_delta_bytes`.
        _require_div(K, g, "MXFP4 K")
        return GemmBytes(E * N * (K // 2), E * N * (K // g), 0, E, kind,
                         f"E*N*K/2 + E*N*K/{g}*1")
    if kind == SCHEME_NVFP4:
        # weight_packed (E,N,K/2) u8 + weight_scale (E,N,K/g) f16 (the FOLDED single-level scale).
        _require_div(K, g, "NVFP4 K")
        return GemmBytes(E * N * (K // 2), E * N * (K // g) * 2, 0, E, kind,
                         f"E*N*K/2 + E*N*K/{g}*2")

    pf = 32 // scheme.bits  # int4 -> 8 nibbles per int32
    if kind == SCHEME_GPTQ:
        # qweight (E,K/pf,N) i32 + scales (E,K/g,N) f16 + qzeros (E,K/g,N/pf) i32. GPTQ always
        # ships zeros (symmetric GPTQ still stores the constant), matching the container.
        _require_div(K, pf, "GPTQ K")
        _require_div(K, g, "GPTQ K")
        _require_div(N, pf, "GPTQ N")
        return GemmBytes(E * (K // pf) * N * 4, E * (K // g) * N * 2,
                         E * (K // g) * (N // pf) * 4, E, kind,
                         f"E*K/{pf}*N*4 + E*K/{g}*N*2 + E*K/{g}*N/{pf}*4")
    if kind == SCHEME_AWQ:
        # qweight (E,K,N/pf) i32 + scales (E,K/g,N) f16 + qzeros (E,K/g,N/pf) i32. AWQ is
        # asymmetric by construction -> zeros ALWAYS present.
        _require_div(N, pf, "AWQ N")
        _require_div(K, g, "AWQ K")
        return GemmBytes(E * K * (N // pf) * 4, E * (K // g) * N * 2,
                         E * (K // g) * (N // pf) * 4, E, kind,
                         f"E*K*N/{pf}*4 + E*K/{g}*N*2 + E*K/{g}*N/{pf}*4")
    if kind == SCHEME_CT_INT4:
        # weight_packed (E,N,K/pf) i32 + weight_scale (E,N,K/g) f16, and weight_zero_point
        # (E,N/pf,K/g) i32 ONLY when the checkpoint is asymmetric. A symmetric CT checkpoint
        # synthesises constant-8 zeros in post_load and DOES allocate them: a real
        # `torch.empty((E,G,N/pf), int32)`, not a broadcast. Being expert-INVARIANT keeps them out of
        # the GRANULE, but they are fully RESIDENT and the arena must hold them -- so they are
        # charged in `post_load_delta_bytes`, not here, and not dropped.
        _require_div(K, pf, "CT int4 K")
        _require_div(K, g, "CT int4 K")
        _require_div(N, pf, "CT int4 N")
        zero = 0 if scheme.sym else E * (N // pf) * (K // g) * 4
        return GemmBytes(E * N * (K // pf) * 4, E * N * (K // g) * 2, zero, E, kind,
                         f"E*N*K/{pf}*4 + E*N*K/{g}*2"
                         + ("" if scheme.sym else f" + E*N/{pf}*K/{g}*4"))
    raise ValueError(f"no analytic byte model for scheme {kind!r}")


def analytic_gemm_rows(gb: GemmBytes, prefix: str) -> Tuple[Tuple[str, int], ...]:
    """One container's arena ROWS from the closed-form model: `((name, nbytes), ...)`, summing to
    `gb.total`.

    A row is ONE COMPONENT's stacked slab (`moe_interpose._plan_items` calls `alloc_like` per
    storage, never per expert), so the closed form's three terms are three rows and the `post_load`
    delta is a fourth -- except for MXFP4, whose delta WIDENS the existing scale row rather than
    adding one. That distinction is not cosmetic: two 24 MiB rows pack into a chunk tail that one
    48 MiB row does not, so modelling a widened row as a separate row would make the reservation
    optimistic in exactly the place the never-straddle rule bites.
    """
    rows = [(f"{prefix}.weight", gb.weight), (f"{prefix}.scale", gb.scale),
            (f"{prefix}.zero", gb.zero)]
    delta = gb.post_load_resident
    if delta:
        if gb.scheme == SCHEME_MXFP4:
            # E8M0 u8 -> fp16 on the SAME buffer: the scale row grows, it does not gain a sibling.
            rows[1] = (rows[1][0], rows[1][1] + delta)
        else:
            # CT-symmetric's `_zeros_op` is a real `torch.empty` post_load allocates from nothing,
            # so it is a row of its own, carved like any other.
            rows.append((f"{prefix}.post_load", delta))
    return tuple((n, b) for n, b in rows if b > 0)


def _meta_spec_rows(spec: Any, prefix: str) -> Tuple[Tuple[str, int], ...]:
    """The meta-built container's rows, straight off the walker's component/replicated lists."""
    comps = getattr(spec, "components", None)
    if comps is None:
        return ()
    n = int(getattr(spec, "num_granules", 0) or 0) or 1
    rows = [(f"{prefix}.{c.name}", int(getattr(c, "nbytes", 0) or 0) * n) for c in comps]
    rows += [
        (f"{prefix}.{r.name}", int(getattr(r, "nbytes", 0) or 0))
        for r in (getattr(spec, "replicated", ()) or ())
    ]
    return tuple((name, nb) for name, nb in rows if nb > 0)


def _apply_post_load_to_rows(
    rows: Tuple[Tuple[str, int], ...], scheme: ExpertScheme, delta: "PostLoadDelta",
    num_experts: int, out_features: int, in_features: int, prefix: str,
) -> Tuple[Tuple[str, int], ...]:
    """Correct meta-derived rows (which are `__init__` shapes) for what `post_load()` does.

    The meta model builds the real container and therefore cannot drift on SHAPES, but it can never
    run `post_load()` -- so it is blind to the two shipped containers that are not byte-invariant
    across it. `expert_stack_bytes` already corrects the TOTAL; the rows have to be corrected the
    same way or the reservation is enumerated against tensors that will not exist.
    """
    if not rows or not delta.resident:
        return rows
    if scheme.kind == SCHEME_MXFP4 and scheme.group_size:
        target = num_experts * out_features * (in_features // scheme.group_size)
        out = list(rows)
        for i, (name, nb) in enumerate(out):
            if nb == target:
                out[i] = (name, nb + delta.resident)
                return tuple(out)
        # The u8 scale row was not where the closed form says it is -- do not guess which row grew;
        # charge it as its own row, which is the conservative reading for the TOTAL and only ever
        # optimistic about packing by one row's worth.
        return tuple(out) + ((f"{prefix}.post_load", delta.resident),)
    return rows + ((f"{prefix}.post_load", delta.resident),)


def _require_div(value: int, divisor: int, what: str) -> None:
    if divisor <= 0 or value % divisor:
        raise ValueError(f"{what}={value} must be divisible by {divisor}")


# ---------------------------------------------------------------------------------------------
# Meta byte model -- the real containers, allocated on `meta`, summed. Preferred when torch is
# importable, because it cannot drift from the allocator.
# ---------------------------------------------------------------------------------------------


def meta_gemm_spec(
    quant: Any, num_experts: int, out_features: int, in_features: int, *, fp8_experts: bool = False
) -> Optional[Any]:
    """Build the real expert container on `torch.device("meta")` and derive its `GranuleSpec`.

    Reuses the ONE walker -- `granule.derive_granule_spec(..., allow_meta=True)` -- rather than
    summing tensors here. That matters for more than tidiness: the walker knows which components are
    per-expert and which are replicated, so `granule_bytes` excludes shared rows that are read once
    per layer instead of once per routed expert. A local sum would over-charge the route.

    `allow_meta=True` is exactly the shapes-only sizing estimate the walker documents: every meta
    tensor reports `data_ptr()==0`, so aliasing and expert-invariance are NOT trustworthy here --
    which is why this is an ESTIMATE and `reconcile_measured_bytes` after `post_load()` is mandatory.

    Returns None (never raises) when torch or the layer stack is unavailable, so a torch-free host
    still gets a plan from the analytic model. Meta tensors allocate NOTHING -- no host RAM, no
    VRAM, no GPU context -- so this is safe to call at boot before any device is selected.
    """
    try:
        import torch
        from minisgl.layers.moe import create_moe_quant_method

        from .granule import spec_for_container
    except Exception:  # pragma: no cover - torch-free host
        return None
    try:
        with torch.device("meta"):
            method = create_moe_quant_method(quant, fp8_experts=fp8_experts)
            container = method.create_experts(num_experts, out_features, in_features)
        return spec_for_container(container, num_experts, allow_meta=True)
    except Exception:  # pragma: no cover - an unbuildable shape is the analytic model's problem
        return None


def meta_gemm_bytes(
    quant: Any, num_experts: int, out_features: int, in_features: int, *, fp8_experts: bool = False
) -> Optional[int]:
    """Total container bytes from the meta-built `GranuleSpec`, or None when torch is unavailable."""
    spec = meta_gemm_spec(quant, num_experts, out_features, in_features, fp8_experts=fp8_experts)
    return None if spec is None else int(spec.total_bytes)


# ---------------------------------------------------------------------------------------------
# The public entry point.
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ExpertStackBytes:
    """Bytes for ONE layer's full expert stack (w13 + w2), as this rank will hold it.

    Two byte counts, and conflating them is a ~50x error in either direction on the target shape
    (E=512, top_k=10):
      * `total` -- RESIDENT bytes. What the arena must store. The capacity unit.
      * `granule` -- bytes of ONE expert across BOTH GEMMs. What routing that expert costs in
        traffic. The bandwidth unit. Excludes expert-invariant components when the walker could
        prove them (meta source cannot prove invariance -- see `meta_gemm_spec`).
    """

    w13: int
    w2: int
    num_local_experts: int
    scheme: str
    source: str  # "meta" | "analytic"
    agreement: Optional[float]  # meta/analytic ratio when both ran, else None
    granule: int = 0
    detail: Tuple[str, ...] = ()
    # Every arena ROW this layer will ask for, in carve order, as `((name, nbytes), ...)`. Sums to
    # `total`. This is what lets the pinned arena be reserved by ENUMERATION instead of by
    # `chunk_plan.headroom_chunks`' worst-case bound (+28% on the target shape) -- see
    # `placement.LayerWeights.rows`. `()` when neither model could break the container down.
    rows: Tuple[Tuple[str, int], ...] = ()

    @property
    def total(self) -> int:
        return self.w13 + self.w2

    @property
    def per_expert(self) -> float:
        return self.total / self.num_local_experts if self.num_local_experts else 0.0

    @property
    def granule_bytes(self) -> int:
        """Co-demanded bytes for one expert; falls back to the flat share when unknown.

        FLOOR, not round: `placement.LayerWeights` requires `resident >= granule * E`, and rounding
        up can break that by a few bytes on a stack whose size is not an exact multiple of E. The
        error is under one byte per expert and it falls on the safe side of the invariant."""
        return self.granule or (self.total // self.num_local_experts if self.num_local_experts
                                else 0)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "w13": self.w13,
            "w2": self.w2,
            "total": self.total,
            "granule": self.granule_bytes,
            "num_local_experts": self.num_local_experts,
            "scheme": self.scheme,
            "source": self.source,
            "agreement": self.agreement,
        }


def expert_stack_bytes(
    *,
    quant: Any,
    num_local_experts: int,
    hidden_size: int,
    intermediate_size_per_partition: int,
    fp8_experts: bool = False,
    compute_dtype_bytes: int = 2,
    prefer_meta: bool = True,
) -> ExpertStackBytes:
    """Bytes for one MoE layer's (w13, w2) pair as MoELayer.__init__ will allocate them.

    Shapes mirror `MoELayer.__init__` exactly: w13 is (E, 2*I_part, H) and w2 is (E, H, I_part),
    where I_part is `intermediate_size` under EP (each rank owns whole experts) and
    `intermediate_size // tp_size` under plain TP. The caller resolves that; getting it wrong is a
    2x error in the arena size, so it is not re-derived here.

    A quant scheme this file does not RECOGNISE is not an error here. `scheme_from_quant` is asked
    non-strictly and the meta model -- which builds the container through the real
    `create_moe_quant_method` -- answers on its own. The repo rule is that a new weight format is a
    WLoad policy on the shared core, so it must require nothing of downstream infrastructure; the
    previous strict resolution meant a format added to `layers/moe.py` took `resolve_weight_plan`
    down, and with it every serve, offload requested or not. Only when BOTH models fail is this an
    error, and then the message names both remedies rather than only the transcription.
    """
    if num_local_experts <= 0:
        raise ValueError(f"num_local_experts must be >= 1, got {num_local_experts}")
    scheme = scheme_from_quant(quant, fp8_experts=fp8_experts,
                               compute_dtype_bytes=compute_dtype_bytes, strict=False)
    shapes = (
        ("w13", 2 * intermediate_size_per_partition, hidden_size),
        ("w2", hidden_size, intermediate_size_per_partition),
    )

    analytic = {}
    analytic_err = None
    try:
        for name, out_f, in_f in shapes:
            analytic[name] = analytic_gemm_bytes(scheme, num_local_experts, out_f, in_f)
    except ValueError as exc:
        analytic_err = str(exc)

    # The meta model builds the container through `create_experts` and NEVER runs `post_load()` --
    # it cannot, on meta tensors. So it reports `__init__` shapes and must be corrected by the same
    # `post_load_delta_bytes` the analytic model uses, or the two shipped non-invariant containers
    # (CT-symmetric's synthesised `_zeros_op`, MXFP4's u8->fp16 scale) under-reserve the arena on the
    # PRODUCTION path, where `prefer_meta=True`. See `PostLoadDelta`.
    meta: Dict[str, Any] = {}
    meta_granule = 0
    meta_delta_note = ""
    meta_rows: Tuple[Tuple[str, int], ...] = ()
    meta_rows_complete = True
    if prefer_meta:
        for name, out_f, in_f in shapes:
            spec = meta_gemm_spec(quant, num_local_experts, out_f, in_f,
                                  fp8_experts=fp8_experts)
            if spec is None:
                meta = {}
                meta_granule = 0
                meta_rows = ()
                break
            d = post_load_delta_bytes(scheme, num_local_experts, out_f, in_f)
            meta[name] = int(spec.total_bytes) + d.resident
            # A routed expert demands its slice of BOTH GEMMs, so the granules ADD -- this is
            # `granule.total_granule_bytes` over the pair, inlined to avoid importing torch here.
            meta_granule += int(spec.granule_bytes) + d.granule
            meta_delta_note = meta_delta_note or d.note
            r = _apply_post_load_to_rows(
                _meta_spec_rows(spec, name), scheme, d, num_local_experts, out_f, in_f, name
            )
            meta_rows_complete = meta_rows_complete and bool(r)
            meta_rows += r
        # All or nothing: half a row list reserves exactly for the components it can see and nothing
        # for the rest, which is short in the silent direction.
        if not meta_rows_complete or sum(nb for _, nb in meta_rows) != sum(meta.values()):
            meta_rows = ()

    if meta and analytic:
        # Compare CHECKPOINT totals, not resident ones: the meta model's subject is `__init__`, and
        # netting the delta out of both sides is what makes the ratio a real drift signal instead of
        # a constant 1.03 that trains people to ignore the warning.
        a_total = analytic["w13"].checkpoint_total + analytic["w2"].checkpoint_total
        m_total = (meta["w13"] + meta["w2"]) - (
            analytic["w13"].post_load_resident + analytic["w2"].post_load_resident
        )
        agreement = (m_total / a_total) if a_total else None
        return ExpertStackBytes(
            meta["w13"], meta["w2"], num_local_experts, scheme.kind, "meta", agreement,
            meta_granule, (analytic["w13"].formula, analytic["w2"].formula),
            # The analytic breakdown is the fallback ONLY when it totals the same bytes the meta
            # model priced. Rows that do not sum to `total` would reserve the arena for a different
            # weight set than the capacity budget was spent on.
            rows=meta_rows or (
                (analytic_gemm_rows(analytic["w13"], "w13")
                 + analytic_gemm_rows(analytic["w2"], "w2"))
                if analytic["w13"].total + analytic["w2"].total == meta["w13"] + meta["w2"]
                else ()
            ),
        )
    if meta:
        return ExpertStackBytes(
            meta["w13"], meta["w2"], num_local_experts, scheme.kind, "meta", None, meta_granule,
            (meta_delta_note,) if meta_delta_note else (),
            rows=meta_rows,
        )
    if analytic:
        # The analytic granule is the CHECKPOINT flat share plus only the PER-EXPERT part of the
        # post_load delta. Falling through to `granule_bytes`'s `total // E` default would divide
        # the resident total instead, which charges a routed expert for CT-symmetric's
        # expert-invariant `_zeros_op` -- capacity bytes billed as bandwidth, i.e. the same
        # conflation in the opposite direction, inflating the projected step time. For every
        # byte-invariant scheme this is bit-identical to the old default.
        a_granule = (
            (analytic["w13"].checkpoint_total + analytic["w2"].checkpoint_total)
            // num_local_experts
            + analytic["w13"].post_load_granule
            + analytic["w2"].post_load_granule
        )
        return ExpertStackBytes(
            analytic["w13"].total, analytic["w2"].total, num_local_experts, scheme.kind,
            "analytic", None, a_granule,
            (analytic["w13"].formula, analytic["w2"].formula),
            rows=analytic_gemm_rows(analytic["w13"], "w13")
            + analytic_gemm_rows(analytic["w2"], "w2"),
        )
    raise ValueError(
        f"cannot size a {scheme.kind} expert stack "
        f"(E={num_local_experts} H={hidden_size} I={intermediate_size_per_partition}): "
        f"{analytic_err}\n"
        "BOTH byte models refused. Either the meta model could not run (torch or "
        "minisgl.layers.moe unimportable, or create_moe_quant_method itself rejects this quant "
        "config -- in which case the model will not build either), or the shape is genuinely "
        "invalid. If layers/moe.py gained a container this file has no arm for, the meta model "
        "should have covered it; check that `prefer_meta` was not turned off."
    )
