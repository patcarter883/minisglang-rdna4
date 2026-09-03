"""Cross-check: the DERIVED granule (live tensors) against the ANALYTIC byte model (`sizing.py`).

Two independent models of the same quantity, built from different sources: `sizing.analytic_gemm_
bytes` is a hand-transcribed closed form over the container `__init__`s (it has to be — the boot-time
capacity abort runs BEFORE any weight exists), while `granule.derive_granule_spec` reads the tensors
that actually survived `post_load`. If they disagree, one of them is wrong and the offload plan either
over- or under-reserves. That is precisely the failure the plan's boot assertion
`|delta memory_allocated - plan.device_bytes| < tol` is meant to catch — better to catch it here.

THE TWO MODELS MUST AGREE ON THE **RESIDENT** TOTAL, FOR EVERY FORMAT. That is the load-bearing
assertion here — `spec.total_bytes == analytic.total` — because the resident total is what the arena
reserves and what `required_device_bytes`/`feasible` are computed from. They agree on the CHECKPOINT
total only where `post_load` is byte-invariant, which is why `GemmBytes` carries both numbers.

Two formats are NOT byte-invariant, and until 2026-09-03 this file pinned that divergence as
"deliberate" while `sizing.py` published the checkpoint figure as the arena size — so the divergence
was documented and never charged:

  * MXFP4's `weight_scale` is a raw E8M0 exponent (uint8) in the checkpoint and an fp16 group scale
    after `post_load`. Checkpoint 1 byte/group, resident 2. Asserted as an exact 2x on the scale
    term AND as a `post_load_resident` the analytic total now includes.
  * A SYMMETRIC compressed-tensors checkpoint ships no zero-points, and `post_load` allocates a real
    `(E, G, N/pf) int32` of 0x88. The granule walker proves it expert-invariant and excludes it from
    the GRANULE — correctly, a routed expert never re-reads it — but it is fully RESIDENT and
    `moe_interpose._plan_items` copies it into the arena like any other tensor. Both the granule
    exclusion and the residency charge are asserted below.

GPU-free.
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
from minisgl.weights import sizing
from minisgl.weights.granule import spec_for_container

E, N, K = 4, 64, 256


# Deliberately a LOCAL copy of `test_granule_spec.py`'s two helpers rather than a cross-test import:
# nothing under `tests/` is a package (no `__init__.py`, matching the rest of the repo), so importing
# between test modules would need a sys.path trick or a shared conftest. Twenty lines is cheaper.
def _q(method: str, **kw) -> QuantConfig:
    base = {"method": method, "bits": 4, "group_size": 32, "sym": True}
    base.update(kw)
    return QuantConfig(**base)  # type: ignore[arg-type]


def _fill(t: torch.Tensor, seed: int, kind: str = "random") -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    if kind == "nibble14":  # GPTQ qzeros stores zero_point-1, so every nibble must be <= 14
        nib = torch.randint(0, 15, t.shape + (8,), generator=g, dtype=torch.int64)
        packed = torch.zeros(t.shape, dtype=torch.int64)
        for j in range(8):
            packed |= nib[..., j] << (4 * j)
        t.copy_((packed - (1 << 32) * (packed >> 31)).to(torch.int32))
    elif kind == "e8m0":  # MXFP4's scale is a raw exponent; keep it inside the fp16 store
        t.copy_(torch.randint(120, 132, t.shape, generator=g, dtype=torch.int64).to(t.dtype))
    elif t.dtype in (torch.int32, torch.uint8):
        hi = 2**31 - 1 if t.dtype == torch.int32 else 255
        t.copy_(torch.randint(0, hi, t.shape, generator=g, dtype=torch.int64).to(t.dtype))
    elif t.dtype == torch.float8_e4m3fn:
        t.copy_(torch.randn(t.shape, generator=g).to(torch.float8_e4m3fn))
    else:
        t.copy_(torch.randn(t.shape, generator=g).to(t.dtype))
    return t


def _load(container, seed: int = 0):
    for i, (name, v) in enumerate(list(vars(container).items())):
        if isinstance(v, torch.Tensor):
            kind = "random"
            if isinstance(container, _GroupedGPTQExperts) and name == "qzeros":
                kind = "nibble14"
            elif isinstance(container, _GroupedMxFp4Experts) and name == "weight_scale":
                kind = "e8m0"
            _fill(v, seed + i, kind)
    container.post_load()
    return container


def _built(monkeypatch, cls, quant, *, shape=(N, K), **kw):
    monkeypatch.setattr(moe_mod.kernels, "MOE_W4A16", "0", raising=False)
    monkeypatch.setattr(moe_mod.kernels, "RXF_REGDIRECT", False, raising=False)
    monkeypatch.setattr(moe_mod.kernels, "MOE_MXFP4_REGDIRECT", False, raising=False)
    monkeypatch.setattr(moe_mod.kernels, "MOE_W8A8_REGDIRECT", False, raising=False)
    n, k = shape
    args = (E, n, k) if quant is None else (E, n, k, quant)
    return _load(cls(*args, **kw))


CASES = {
    "gptq": (_GroupedGPTQExperts, lambda: _q("gptq", group_size=32, sym=False)),
    "awq": (_GroupedAWQExperts, lambda: _q("awq", group_size=32, sym=False)),
    "rxf": (_GroupedRXFExperts, lambda: _q("rxf", group_size=32)),
    "ct_int4_sym": (_GroupedCompressedTensorsExperts, lambda: _q("compressed-tensors", sym=True)),
    "ct_int4_asym": (_GroupedCompressedTensorsExperts, lambda: _q("compressed-tensors", sym=False)),
    "nvfp4": (
        _GroupedNvFp4Experts,
        lambda: _q("compressed-tensors", group_size=16, weight_type="float",
                   ct_format="nvfp4-pack-quantized"),
    ),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_derived_resident_bytes_match_the_analytic_model(name, monkeypatch):
    """`spec.total_bytes` is what the arena must hold; `analytic.total` is what it reserves.

    `total_bytes`, not `stacked_bytes`: a replicated component is excluded from the GRANULE but is
    still resident, and `moe_interpose._plan_items` copies it. Comparing the stacked figure is how
    CT-symmetric's `_zeros_op` stayed uncharged.
    """
    cls, mk = CASES[name]
    quant = mk()
    c = _built(monkeypatch, cls, quant)
    spec = spec_for_container(c, E)
    scheme = sizing.scheme_from_quant(quant)
    analytic = sizing.analytic_gemm_bytes(scheme, E, N, K)
    assert spec.total_bytes == analytic.total, (
        f"{name}: derived resident {spec.total_bytes} B vs analytic {analytic.total} B "
        f"({analytic.formula}); components={[(x.name, x.nbytes) for x in spec.components]}; "
        f"replicated={[(r.name, r.nbytes) for r in spec.replicated]}"
    )
    # And the granule (bandwidth) must exclude exactly the replicated part, no more and no less.
    assert spec.stacked_bytes == analytic.total - spec.replicated_bytes


def test_meta_sizing_matches_the_live_post_load_container(monkeypatch):
    """The end-to-end claim, on the PRODUCTION path: `expert_stack_bytes(prefer_meta=True)` is the
    number `resolve_weight_plan` reserves the arena from, and the arena holds these tensors.

    The meta model builds the container through `create_experts` and cannot run `post_load` on meta
    tensors, so without `post_load_delta_bytes` it reports `__init__` shapes — which for CT-symmetric
    is 3.1% short of what the walker below finds on the real, loaded container.
    """
    # `MoELayer.__init__`'s two GEMMs, at H=K and I_part=N: w13 is (E, 2*I, H), w2 is (E, H, I).
    quant = _q("compressed-tensors", sym=True)
    w13 = _built(monkeypatch, _GroupedCompressedTensorsExperts, quant, shape=(2 * N, K))
    w2 = _built(monkeypatch, _GroupedCompressedTensorsExperts, quant, shape=(K, N))
    live = spec_for_container(w13, E).total_bytes + spec_for_container(w2, E).total_bytes
    sized = sizing.expert_stack_bytes(
        quant=quant, num_local_experts=E, hidden_size=K,
        intermediate_size_per_partition=N, prefer_meta=True,
    )
    assert sized.source == "meta"
    assert sized.total == live
    # The agreement ratio is a DRIFT signal, so it must compare like with like (both `__init__`
    # shapes) and stay at 1.0 rather than sitting at a constant 1.03 nobody reads.
    assert sized.agreement == pytest.approx(1.0)


def test_fp8_matches_the_analytic_model(monkeypatch):
    c = _built(monkeypatch, _GroupedFP8Experts, None)
    spec = spec_for_container(c, E)
    analytic = sizing.analytic_gemm_bytes(sizing.ExpertScheme(sizing.SCHEME_FP8, bits=8), E, N, K)
    assert spec.total_bytes == analytic.total == analytic.checkpoint_total


def test_unquantized_matches_the_analytic_model():
    t = torch.zeros(E, N, K, dtype=torch.bfloat16)
    t[:] = torch.arange(E, dtype=torch.bfloat16).view(E, 1, 1)  # distinct rows
    spec = spec_for_container(t, E)
    analytic = sizing.analytic_gemm_bytes(
        sizing.ExpertScheme(sizing.SCHEME_UNQUANTIZED, bits=16, elem_bytes=2), E, N, K
    )
    assert spec.stacked_bytes == analytic.total == E * N * K * 2


def test_symmetric_ct_zeros_are_replicated_in_the_granule_but_CHARGED_as_resident(monkeypatch):
    """The two halves of the same fact, and getting either one wrong is a wrong number.

    The granule walker proves the synthesised `_zeros_op` expert-invariant and keeps it out of the
    granule — right, because a routed expert never re-reads it, so charging it would inflate the
    projected step time. But it is a materialized `(E, G, N/pf) int32`, `_plan_items` copies it into
    the arena, and `sizing` must charge it as RESIDENT — which it did not until 2026-09-03.
    """
    c = _built(monkeypatch, _GroupedCompressedTensorsExperts, _q("compressed-tensors", sym=True))
    spec = spec_for_container(c, E)
    assert "_zeros_op" not in {x.name for x in spec.components}
    rep = [r for r in spec.replicated if r.name == "_zeros_op"]
    assert len(rep) == 1 and rep[0].reason == "expert-invariant"
    assert spec.total_bytes == spec.stacked_bytes + rep[0].nbytes

    scheme = sizing.scheme_from_quant(_q("compressed-tensors", sym=True))
    analytic = sizing.analytic_gemm_bytes(scheme, E, N, K)
    assert analytic.post_load_resident == rep[0].nbytes  # charged to capacity...
    assert analytic.post_load_granule == 0               # ...and not to bandwidth
    assert analytic.checkpoint_total == spec.stacked_bytes
    assert analytic.total == spec.total_bytes


def test_mxfp4_scale_widens_at_post_load(monkeypatch):
    """The MXFP4 checkpoint's E8M0 group scale is 1 byte; `post_load` converts it to an fp16 group
    scale, so the OFFLOADED bytes are 2 per group. Pinned on both sides so neither can move alone:
    the checkpoint term stays 1 byte/group, and the RESIDENT total carries the widening."""
    quant = _q("compressed-tensors", group_size=32, weight_type="float",
               ct_format="mxfp4-pack-quantized")
    c = _built(monkeypatch, _GroupedMxFp4Experts, quant)
    spec = spec_for_container(c, E)
    analytic = sizing.analytic_gemm_bytes(
        sizing.ExpertScheme(sizing.SCHEME_MXFP4, bits=4, group_size=32), E, N, K
    )
    by_name = {x.name: x for x in spec.components}
    weight_bytes = by_name["_w_op"].nbytes * E
    scale_bytes = by_name["_scales_op"].nbytes * E
    assert weight_bytes == analytic.weight
    assert scale_bytes == 2 * analytic.scale, (
        "MXFP4 post_load no longer widens E8M0 -> fp16, or sizing.py changed: "
        f"derived scale {scale_bytes} B vs analytic {analytic.scale} B"
    )
    assert analytic.post_load_resident == analytic.scale
    # Unlike CT's zeros this widening IS per-expert, so it is charged to the granule too.
    assert analytic.post_load_granule == analytic.scale // E
    assert spec.stacked_bytes == analytic.total
    assert spec.total_bytes == analytic.total
