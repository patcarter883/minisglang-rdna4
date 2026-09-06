"""Desync guards: the ways a granule can quietly stop describing what the kernel reads.

Every case here is a WRONG-NUMBERS bug with no crash if it slips through — a scale left behind, a
w13/w2 pair placed as if they had the same components, two TP ranks disagreeing about what an expert
is. The tests assert each is DETECTED, and (where it matters) that the detector is non-vacuous: it
also has to accept the legitimate case, or "raises on everything" would pass.

GPU-free. Exercises the real `MoELayer` seam (`gate_up_proj` / `down_proj`) at tp_size=1.
"""

from __future__ import annotations

import pytest
import torch
from minisgl.layers import moe as moe_mod
from minisgl.layers.moe import _GroupedCompressedTensorsExperts, _GroupedFP8Experts
from minisgl.quant.config import QuantConfig
from minisgl.weights.granule import (
    ExpertComponent,
    GranuleError,
    GranuleSpec,
    assert_granule_pair_consistent,
    spec_for_container,
    total_granule_bytes,
)

E = 4
HIDDEN = 128
INTER = 64


def _lds(monkeypatch):
    monkeypatch.setattr(moe_mod.kernels, "MOE_W4A16", "0", raising=False)
    monkeypatch.setattr(moe_mod.kernels, "MOE_W8A8_REGDIRECT", False, raising=False)


def _ct(sym: bool) -> QuantConfig:
    return QuantConfig(method="compressed-tensors", bits=4, group_size=32, sym=sym)


def _pair(monkeypatch, sym: bool = False):
    """A real w13/w2 pair with the shapes `MoELayer.__init__` builds: w13 is (2*inter, hidden),
    w2 is (hidden, inter). Different shapes, SAME component set — the legitimate case."""
    _lds(monkeypatch)
    out = []
    for n_out, n_in, seed in ((2 * INTER, HIDDEN, 1), (HIDDEN, INTER, 2)):
        c = _GroupedCompressedTensorsExperts(E, n_out, n_in, _ct(sym))
        g = torch.Generator().manual_seed(seed)
        for name, v in list(vars(c).items()):
            if isinstance(v, torch.Tensor):
                v.copy_(
                    torch.randint(0, 2**31 - 1, v.shape, generator=g, dtype=torch.int64).to(v.dtype)
                    if v.dtype in (torch.int32, torch.uint8)
                    else torch.randn(v.shape, generator=g).to(v.dtype)
                )
        c.post_load()
        out.append(c)
    return out


# =====================================================================================
# The w13/w2 pair
# =====================================================================================


def test_matched_pair_is_accepted(monkeypatch):
    """Non-vacuity: the detector must PASS on a real, correctly-matched pair with different shapes."""
    w13, w2 = _pair(monkeypatch)
    s13, s2 = spec_for_container(w13, E), spec_for_container(w2, E)
    assert [c.name for c in s13.components] == [c.name for c in s2.components]
    assert tuple(s13.components[0].shape) != tuple(s2.components[0].shape)  # shapes really differ
    assert_granule_pair_consistent(s13, s2, where="test")
    # The co-demanded granule is BOTH GEMMs' slices of one expert.
    assert total_granule_bytes([s13, s2]) == s13.granule_bytes + s2.granule_bytes


def test_half_applied_repack_knob_is_caught(monkeypatch):
    """The realistic desync: one container repacked to `_w_rep`, the other still on `_w_op`.

    That is a live possibility — the repack sits behind `kernels.MOE_W4A16` and is applied per
    container in `post_load`, so a half-applied state has nothing else to trip over. Placed as one
    unit, the two halves of a layer would land on different media with different byte budgets."""
    w13, w2 = _pair(monkeypatch)
    w13._w_rep = w13._w_op
    del w13._w_op
    with pytest.raises(GranuleError, match="component sets differ"):
        assert_granule_pair_consistent(spec_for_container(w13, E), spec_for_container(w2, E))


def test_expert_count_mismatch_is_caught(monkeypatch):
    """An EP-sharded rank and a replicated one must never be paired."""
    w13, w2 = _pair(monkeypatch)
    s13 = spec_for_container(w13, E)
    s2 = spec_for_container(w2, E)
    with pytest.raises(GranuleError, match="expert counts differ"):
        assert_granule_pair_consistent(s13, GranuleSpec(s2.kind, E // 2, s2.components))


def test_dtype_mismatch_is_caught(monkeypatch):
    w13, w2 = _pair(monkeypatch)
    s13 = spec_for_container(w13, E)
    s2 = spec_for_container(w2, E)
    bad = tuple(
        ExpertComponent(c.name, torch.bfloat16 if i == 1 else c.dtype, c.shape,
                        aliases=c.aliases, stacked_shape=c.stacked_shape)
        for i, c in enumerate(s2.components)
    )
    with pytest.raises(GranuleError, match="dtypes differ"):
        assert_granule_pair_consistent(s13, GranuleSpec(s2.kind, s2.num_experts, bad))


def test_kind_mismatch_is_caught(monkeypatch):
    w13, _ = _pair(monkeypatch)
    s13 = spec_for_container(w13, E)
    _lds(monkeypatch)
    fp8 = _GroupedFP8Experts(E, HIDDEN, INTER)
    fp8.weight.copy_(torch.randn(fp8.weight.shape).to(torch.float8_e4m3fn))
    fp8.weight_scale.copy_(torch.randn(fp8.weight_scale.shape))
    fp8.post_load()
    with pytest.raises(GranuleError, match="container kinds differ"):
        assert_granule_pair_consistent(s13, spec_for_container(fp8, E))


# =====================================================================================
# Fingerprint: two ranks must agree on what an expert IS
# =====================================================================================


def test_fingerprint_distinguishes_a_dropped_scale(monkeypatch):
    """Dropping the scales is exactly the silent-corruption case; the fingerprint must move."""
    from dataclasses import replace

    w13, _ = _pair(monkeypatch)
    full = spec_for_container(w13, E)
    assert "_scales_op" in [c.name for c in full.components]
    without = replace(
        full, components=tuple(c for c in full.components if c.name != "_scales_op")
    )
    assert without.fingerprint() != full.fingerprint()
    assert without.granule_bytes < full.granule_bytes


def test_fingerprint_distinguishes_symmetric_from_asymmetric_zeros(monkeypatch):
    """A symmetric checkpoint's zeros are expert-invariant (excluded); an asymmetric one's are not.

    The two must not fingerprint the same, because they place different byte counts per expert."""
    sym13, _ = _pair(monkeypatch, sym=True)
    asym13, _ = _pair(monkeypatch, sym=False)
    s_sym = spec_for_container(sym13, E)
    s_asym = spec_for_container(asym13, E)
    assert s_sym.fingerprint() != s_asym.fingerprint()
    assert s_sym.granule_bytes < s_asym.granule_bytes
    assert [r.reason for r in s_sym.replicated] == ["expert-invariant"]


def test_fingerprint_is_insensitive_to_tensor_CONTENTS(monkeypatch):
    """It describes the LAYOUT, not the weights: two ranks hold different shards of the same format
    and must still agree. (Contents-sensitivity would make it useless as a cross-rank check.)

    `add_(1)` on a per-expert buffer is the WEAK version of this — it perturbs bytes without
    changing how any of them classify, so on its own it cannot catch a content-dependent walker.
    `test_fingerprint_survives_a_shard_whose_rows_happen_to_match` below is the sharp one."""
    a13, _ = _pair(monkeypatch, sym=False)
    b13, _ = _pair(monkeypatch, sym=False)
    b13._w_op.add_(1)
    assert spec_for_container(a13, E).fingerprint() == spec_for_container(b13, E).fingerprint()


def test_fingerprint_survives_a_shard_whose_rows_happen_to_match(monkeypatch):
    """REGRESSION (TP desync). Rank 0 and rank 1 hold DIFFERENT SHARDS of the same tensors — w13 is
    column-split, w2 row-split, and under EP each rank holds different experts entirely. So a walker
    that drops a component because *its* slice happens to be row-identical lets one rank exempt a
    buffer the other keeps.

    Nothing downstream can catch that: each rank derives its spec with no collective, and
    `placement.LayerWeights` turns `granule_bytes`/`fingerprint()` straight into that rank's byte
    budget (`placement.py:24`, `host_capacity.py:178` — "ranks place granules at different offsets
    and the collectives hang"). Here rank B's asymmetric `_zeros_op` is uniform across experts by
    coincidence; the two specs must still agree byte for byte.

    The pre-fix walker excluded B's zeros as "expert-invariant" and kept A's — different
    fingerprints, different granule_bytes, silently."""
    a13, _ = _pair(monkeypatch, sym=False)
    b13, _ = _pair(monkeypatch, sym=False)
    assert "_zeros_op" in {c.name for c in spec_for_container(a13, E).components}
    b13._zeros_op[:] = b13._zeros_op[0]  # this rank's shard: every expert row identical

    sa = spec_for_container(a13, E)
    sb = spec_for_container(b13, E)
    assert sb.fingerprint() == sa.fingerprint()
    assert sb.granule_bytes == sa.granule_bytes
    assert "_zeros_op" in {c.name for c in sb.components}
    # ...and the coincidence is still SURFACED, just not acted on.
    assert "_zeros_op" in sb.content_invariant and "_zeros_op" not in sa.content_invariant
    # content_invariant must not leak into the hash, or it is the same divergence by another route.
    assert sb.content_invariant != sa.content_invariant


# =====================================================================================
# The MoELayer seam itself
# =====================================================================================


@pytest.fixture()
def tp1():
    """tp_size=1, EP off — the smallest state `MoELayer.__init__` needs. `set_tp_info` refuses a
    second call, so this is process-idempotent by design."""
    from minisgl.distributed import info as dinfo

    if dinfo.try_get_tp_info() is None:
        dinfo.set_tp_info(0, 1)
    return dinfo.get_tp_info()


def _moe_layer(monkeypatch, quant):
    _lds(monkeypatch)
    return moe_mod.MoELayer(
        num_experts=E, top_k=2, hidden_size=HIDDEN, intermediate_size=INTER, quant=quant
    )


def test_moe_layer_granule_specs(tp1, monkeypatch):
    """The seam: `forward` reads `w13, w2 = self.gate_up_proj, self.down_proj`, so the residency
    layer needs exactly these two containers and nothing format-specific."""
    layer = _moe_layer(monkeypatch, _ct(sym=False))
    for c in layer.expert_containers().values():
        g = torch.Generator().manual_seed(5)
        for name, v in list(vars(c).items()):
            if isinstance(v, torch.Tensor):
                v.copy_(
                    torch.randint(0, 2**31 - 1, v.shape, generator=g, dtype=torch.int64).to(v.dtype)
                    if v.dtype in (torch.int32, torch.uint8)
                    else torch.randn(v.shape, generator=g).to(v.dtype)
                )
        c.post_load()

    specs = layer.granule_specs()
    assert set(specs) == {"gate_up_proj", "down_proj"}
    for name, s in specs.items():
        assert s.num_experts == layer.local_num_experts == E
        assert s.components, f"{name}: empty granule"
    assert layer.co_demanded_granule_bytes() == sum(s.granule_bytes for s in specs.values())
    # w13 is (2*inter, hidden), w2 is (hidden, inter): the gate|up merge makes w13 the larger half.
    assert specs["gate_up_proj"].granule_bytes > specs["down_proj"].granule_bytes


def test_moe_layer_unquantized_bare_tensor_containers(tp1, monkeypatch):
    """`_UnquantizedMoEMethod.create_experts` returns a BARE `torch.empty(E, out, in)` — there is no
    object to hang attributes on — so `gate_up_proj` IS a tensor. The seam must still describe it and
    the pair check must still fire, WITHOUT the plan's `_GroupedUnquantizedExperts` wrapper (which
    cannot land without also touching the loader; see the note in `granule.py`)."""
    layer = _moe_layer(monkeypatch, None)
    assert isinstance(layer.gate_up_proj, torch.Tensor)
    for t in layer.expert_containers().values():
        t.copy_(torch.arange(t.numel(), dtype=t.dtype).reshape(t.shape))
    specs = layer.granule_specs()
    for s in specs.values():
        assert [c.name for c in s.components] == ["weight"]
        assert s.num_experts == E
    assert layer.co_demanded_granule_bytes() == sum(s.granule_bytes for s in specs.values())


def test_moe_layer_refuses_a_meta_build(tp1, monkeypatch):
    """`engine.py` builds the model on the meta device; deriving there would silently mis-dedupe."""
    with torch.device("meta"):
        layer = _moe_layer(monkeypatch, _ct(sym=False))
    with pytest.raises(GranuleError, match="META"):
        layer.granule_specs()
    # Shapes-only sizing is still available for the boot-time capacity abort.
    specs = layer.granule_specs(allow_meta=True)
    assert all(s.meta for s in specs.values())
    assert specs["gate_up_proj"].granule_bytes > 0
