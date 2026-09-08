"""NVFP4's TWO-LEVEL scale, end to end on the host: split -> merge -> stack -> container -> op args.

`quant/nvfp4.py` used to collapse NVFP4's e4m3 block scale and its per-tensor global into one fp16
per-group scale, and its docstring claimed the collapse was exact. It is not: measured against an
fp64 golden on real `Qwen3.8-Flash-Next-NVFP4` layer-0 experts the fold carries 4.37e-04 max relative
error on every weight, where keeping both levels is exact to f32 round-off (4.2e-08). The fold was
also BIGGER -- 2 bytes per 16-element group against 1 byte plus 4 bytes per output channel.

WHAT THESE TESTS ARE FOR. Every failure mode of this change is silent. A dropped global, a global
folded in the wrong direction, a global sharded on the wrong axis, a `.to(int32)` where a
`.view(int32)` was meant, an arena row that was never counted -- all of them load, all of them run,
all of them produce finite plausible numbers. So each one is asserted here rather than left to a
serve to discover. No GPU: this is all shapes, dtypes, bytes and f64 arithmetic.
"""

from __future__ import annotations

import pytest
import torch
from minisgl.quant import nvfp4
from minisgl.quant.mxfp4 import FP4_E2M1_LUT

G = nvfp4.NVFP4_GROUP_SIZE  # 16


def _fake_leaf(n: int, k: int, seed: int = 0):
    """A checkpoint-shaped NVFP4 leaf: packed E2M1 bytes, an e4m3 block scale, a f32 global.

    The block scale is built by ROUND-TRIPPING through e4m3 (`.to(float8_e4m3fn)`), because that is
    what a real checkpoint stores -- an arbitrary f32 would let the split look exact for the wrong
    reason (there would be no 4-bit significand to preserve).
    """
    gen = torch.Generator().manual_seed(seed)
    packed = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, generator=gen)
    raw = torch.rand((n, k // G), generator=gen) * 200.0 + 1.0
    block = raw.to(torch.float8_e4m3fn)
    # modelopt's `weight_scale_2`: a MULTIPLIER, small.
    g = torch.tensor(2.0781e-4, dtype=torch.float32)
    return packed, block, g


def _golden_f64(packed, block, mul: float) -> torch.Tensor:
    """E2M1 magnitudes x e4m3 block scale x global, all in float64. The thing both arms approximate."""
    codes = nvfp4.unpack_e2m1_nibbles(packed).to(torch.int64)
    lut = torch.tensor(FP4_E2M1_LUT, dtype=torch.float64)
    return (
        lut[codes]
        * block.to(torch.float64).repeat_interleave(G, dim=-1)
        * torch.tensor(mul, dtype=torch.float64)
    )


# ---------------------------------------------------------------------------------------------
# 1. The accuracy claim, which is the whole reason for the change.
# ---------------------------------------------------------------------------------------------


def test_the_fp16_fold_is_lossy_and_the_split_is_not():
    """The claim the old docstring got backwards, as a measurement on the same bytes.

    Both arms are derived from ONE (block, global) pair and compared against ONE f64 golden, so the
    only difference between them is where the rounding happens. If a future change makes the fold
    exact (it cannot -- fp16 has an 11-bit significand and the product needs ~30) this test says so
    loudly instead of leaving the docstring wrong in the other direction.
    """
    packed, block, g = _fake_leaf(64, 256)
    mul = float(nvfp4.nvfp4_global_multiplier(g, global_field="weight_scale_2"))
    gold = _golden_f64(packed, block, mul)

    b_split, gvec = nvfp4.split_nvfp4_scale(block, g, global_field="weight_scale_2")
    conv = nvfp4.convert_nvfp4_weight(packed, b_split)
    w_split = nvfp4.dequantize_nvfp4_split(conv["w_packed"], conv["scales"], gvec)

    folded = nvfp4.fold_nvfp4_scale(block, g, global_field="weight_scale_2")
    w_fold = nvfp4.dequantize_nvfp4_folded(conv["w_packed"], folded)

    nz = gold != 0
    rel = lambda w: float(  # noqa: E731
        ((w.to(torch.float64) - gold).abs() / gold.abs().clamp_min(1e-300))[nz].max()
    )
    r_split, r_fold = rel(w_split), rel(w_fold)
    assert r_fold > 1e-4, f"the fp16 fold should be visibly lossy; got {r_fold:.3e}"
    assert r_split < 1e-6, f"the two-level split should be exact to f32 round-off; got {r_split:.3e}"
    assert r_fold / r_split > 1000.0


def test_the_block_scale_is_byte_verbatim():
    """No arithmetic touches the block scale, so there is no rounding step to argue about.

    Asserted on the BYTES, not on the values: a `.to(float8_e4m3fn)` round-trip of a value-converted
    tensor could coincidentally agree on most entries.
    """
    _, block, g = _fake_leaf(32, 128)
    out, _ = nvfp4.split_nvfp4_scale(block, g, global_field="weight_scale_2")
    assert out.dtype == torch.float8_e4m3fn
    assert torch.equal(out.view(torch.uint8), block.view(torch.uint8))


def test_a_uint8_block_scale_is_bitcast_not_value_converted():
    """Some exporters ship the block scale as raw bytes. `.to(float8_e4m3fn)` on those would read
    byte 126 as the NUMBER 126 and saturate to 448 -- plausible for small bytes, catastrophic for
    large ones, and this checkpoint's down_proj scales saturate at byte 126 on every expert."""
    _, block, g = _fake_leaf(16, 64)
    as_bytes = block.view(torch.uint8).clone()
    out, _ = nvfp4.split_nvfp4_scale(as_bytes, g, global_field="weight_scale_2")
    assert torch.equal(out.view(torch.uint8), as_bytes)


def test_a_folded_fp16_scale_cannot_be_fed_back_into_the_split():
    """Splitting an already-folded scale would bake the fold's 4.37e-04 into the 'exact' path."""
    _, block, g = _fake_leaf(16, 64)
    folded = nvfp4.fold_nvfp4_scale(block, g, global_field="weight_scale_2")
    with pytest.raises(ValueError, match="float8_e4m3fn"):
        nvfp4.split_nvfp4_scale(folded, g, global_field="weight_scale_2")


# ---------------------------------------------------------------------------------------------
# 2. Direction. The documented sign is the WRONG one for this checkpoint.
# ---------------------------------------------------------------------------------------------


def test_direction_is_normalised_to_a_multiplier_for_both_producers():
    """The kernel only ever multiplies, so the reciprocal must be taken host-side, exactly once."""
    g_mod = torch.tensor(2.0781e-4)  # modelopt: already a multiplier
    g_ct = torch.tensor(1.0 / 2.0781e-4)  # compressed-tensors: a divisor
    m_mod = float(nvfp4.nvfp4_global_multiplier(g_mod, global_field="weight_scale_2"))
    m_ct = float(nvfp4.nvfp4_global_multiplier(g_ct, global_field="weight_global_scale"))
    assert m_mod == pytest.approx(2.0781e-4, rel=1e-6)
    assert m_ct == pytest.approx(2.0781e-4, rel=1e-6)


def test_an_unknown_global_spelling_raises_instead_of_guessing():
    """A new producer's spelling is a ~7-orders-of-magnitude coin flip; refuse to call it."""
    with pytest.raises(ValueError, match="unknown NVFP4 global-scale field"):
        nvfp4.nvfp4_global_multiplier(torch.tensor(1.0), global_field="weight_scale_3")


@pytest.mark.parametrize("bad", [0.0, -1.0, float("inf"), float("nan")])
def test_a_non_positive_or_non_finite_global_is_refused(bad):
    """Both producers derive the global from an amax, so it is positive and finite by construction.
    A zero would make every expert output zero; a NaN would make them all NaN 48 layers later."""
    with pytest.raises(ValueError):
        nvfp4.nvfp4_global_multiplier(torch.tensor(bad), global_field="weight_scale_2")


# ---------------------------------------------------------------------------------------------
# 3. The MERGE. This is why the global is an N-vector and not a scalar.
# ---------------------------------------------------------------------------------------------


def test_the_global_survives_the_gate_up_merge_and_the_expert_stack():
    """VERIFIED, not assumed: run the loader's OWN merge and stack primitives over the leaves.

    `_gate_up_merge` + `torch.cat(dim=0)` is the gate|up merge; `_ExpertStacker` is the expert stack.
    Both are imported from `models/weight.py` rather than re-implemented, so this cannot pass against
    a private re-derivation of what the loader is believed to do.

    gate and up carry DIFFERENT globals here on purpose. On the real checkpoint they happen to be
    equal (1536/1536 experts measured), and a test built on that coincidence would pass just as well
    if the merge dropped one of them.
    """
    from minisgl.models.weight import (
        _ExpertStacker,
        _gate_up_merge,
        _get_expert_stack_info,
    )

    E, INTER, K = 4, 32, 128
    stacker = _ExpertStacker()
    stacked = None
    for e in range(E):
        parts = {}
        for proj, gval in (("gate_proj", 2.0781e-4), ("up_proj", 5.5e-4)):
            base = f"model.layers.0.mlp.experts.{e}.{proj}"
            _, block, _ = _fake_leaf(INTER, K, seed=e)
            leaves = nvfp4.nvfp4_leaf_scales(
                base, block, torch.tensor(gval), global_field="weight_scale_2"
            )
            names = [n for n, _ in leaves]
            assert names == [f"{base}.weight_scale", f"{base}.weight_global"], names
            gvec = dict(leaves)[f"{base}.weight_global"]
            assert gvec.shape == (INTER,), "the global must be a per-OUTPUT-CHANNEL vector"
            merged_key, slot = _gate_up_merge(f"{base}.weight_global")
            parts[slot] = gvec
        merged = torch.cat([parts["gate"], parts["up"]], dim=0)
        assert merged.shape == (2 * INTER,)
        # The merged vector is CONSTANT ON EACH CONTIGUOUS OUTPUT-CHANNEL RANGE -- the exact property
        # that makes an N-vector the right contract and a scalar the wrong one.
        assert torch.equal(merged[:INTER], torch.full((INTER,), 2.0781e-4))
        assert torch.equal(merged[INTER:], torch.full((INTER,), 5.5e-4))
        # `_get_expert_stack_info` is what turns `...experts.<e>.gate_up_proj.weight_global` into the
        # packed container key + expert index. Using it (rather than the merged per-expert key) is
        # the point: the global has to be recognised as a stackable per-expert leaf by the SAME rule
        # the packed weight is, or it would never reach the container at all.
        packed_key, idx = _get_expert_stack_info(merged_key)
        assert packed_key.endswith("experts.gate_up_proj.weight_global"), packed_key
        assert idx == e
        stacked = stacker.add(packed_key, idx, merged, E)
    assert stacked is not None and stacked.shape == (E, 2 * INTER)
    assert stacked.dtype == torch.float32
    assert not stacker, "the stack should be complete"


def test_a_dense_module_still_folds_because_its_kernel_cannot_take_two_levels():
    """RULE 5 fence, as a test. The dense e2m1 core still hardcodes `const __half* w_scales`, so a
    dense leaf must NOT start emitting e4m3 -- that would be read as halves and return garbage."""
    _, block, g = _fake_leaf(16, 64)
    dense = "model.layers.0.self_attn.q_proj"
    moe = "model.layers.0.mlp.experts.7.gate_proj"
    assert not nvfp4.nvfp4_leaf_splits(dense)
    assert nvfp4.nvfp4_leaf_splits(moe)
    d = nvfp4.nvfp4_leaf_scales(dense, block, g, global_field="weight_scale_2")
    assert len(d) == 1 and d[0][1].dtype == torch.float16
    m = nvfp4.nvfp4_leaf_scales(moe, block, g, global_field="weight_scale_2")
    assert len(m) == 2 and m[0][1].dtype == torch.float8_e4m3fn


@pytest.mark.parametrize("shard", ["qwen4_exp", "qwen3_5", "laguna"])
def test_the_global_shards_on_the_output_axis_which_is_not_its_weights_axis_for_down(shard):
    """THE ONE AXIS THAT DIFFERS. `weight_global` is per-OUTPUT-CHANNEL:

      * gate/up are COLUMN-parallel -> their output N splits, so the global splits on dim 0, the
        same axis as their weight and block scale.
      * down is ROW-parallel -> it splits the INPUT K, so its output N is FULL WIDTH on every rank
        and its global REPLICATES, while its weight and block scale split on dim 1.

    Getting this wrong does not raise on gate/up (a 1-D chunk is legal) and on down it either raises
    (chunking 1-D on dim 1) or, if a future rule matched it on dim 0, hands each rank a quarter of
    the output channels it actually computes -- fluent and wrong. So all three sharders are checked,
    and against the WEIGHT's axis, so the difference is the assertion rather than a comment.
    """
    import minisgl.models.weight as W

    fn, prefix = {
        "qwen4_exp": (W._shard_qwen4_exp, "model.language_model.layers.0.mlp.experts.7."),
        "qwen3_5": (W._shard_qwen3_5, "model.layers.0.mlp.experts.7."),
        "laguna": (W._shard_laguna, "model.layers.0.mlp.experts.7."),
    }[shard]

    class Cfg:
        num_experts = 8
        linear_key_head_dim = 128
        linear_num_key_heads = 2
        linear_value_head_dim = 128
        linear_num_value_heads = 2

        class quant:
            group_size = 16
            bits = 4

    N, K, n = 64, 128, 2
    gate_scale_key = prefix + ("gate_proj.weight_scale" if shard != "laguna"
                               else "gate_proj.weight_scale")
    gate_scale = torch.zeros((N, K // G))
    down_scale = torch.zeros((N, K // G))
    gvec = torch.arange(N, dtype=torch.float32)

    # gate/up: the global splits on dim 0, exactly like the block scale.
    got = fn(gate_scale_key.replace("weight_scale", "weight_global"), gvec, 1, n, Cfg)
    assert got.shape == (N // n,)
    assert torch.equal(got, gvec[N // n :]), "rank 1 must get the SECOND half of the channels"
    assert fn(gate_scale_key, gate_scale, 1, n, Cfg).shape == (N // n, K // G)

    # down: the global REPLICATES while its block scale splits on dim 1.
    down_g = fn(prefix + "down_proj.weight_global", gvec, 1, n, Cfg)
    assert down_g.shape == (N,), "down_proj's global is per-output-channel and must NOT split"
    assert torch.equal(down_g, gvec)
    ds = fn(prefix + ("down_proj.weight_scale"), down_scale, 1, n, Cfg)
    assert ds.shape == (N, (K // G) // n), "the down_proj block scale splits the INPUT axis"


# ---------------------------------------------------------------------------------------------
# 4. The container, and the bitcast that is one character from a silent zero.
# ---------------------------------------------------------------------------------------------


def test_post_load_bitcasts_the_global_and_keeps_the_block_scale_e4m3():
    from minisgl.layers.moe import _GroupedNvFp4Experts

    class Q:
        group_size = 16

    E, N, K = 3, 32, 128
    c = _GroupedNvFp4Experts(E, N, K, Q())
    assert c.weight_scale.dtype == torch.float8_e4m3fn
    assert c.weight_global.shape == (E, N) and c.weight_global.dtype == torch.float32
    c.weight_packed.random_(0, 256)
    c.weight_scale.view(torch.uint8).random_(1, 200)
    c.weight_global.uniform_(1e-4, 1e-3)
    want = c.weight_global.clone()
    c.post_load()
    assert c._scales_op.shape == (E, K // G, N)
    assert c._scales_op.dtype == torch.float8_e4m3fn
    assert c._global_op.shape == (E, N) and c._global_op.dtype == torch.int32
    # THE BITCAST. `.to(torch.int32)` would truncate 2.078e-04 to 0 and zero every expert output.
    assert torch.equal(c._global_op.view(torch.float32), want)
    assert not hasattr(c, "weight_global")


def test_the_op_argument_pair_is_checked_rather_than_trusted():
    """`w_zeros` carries two different tensors under two different policies. A mismatched pair is a
    wrong-stride pointer dereference, not a shape error the op would catch, so it is refused here."""
    from minisgl.quant.kernels import _check_moe_scale_pair

    E, N, Kw, Gn = 4, 64, 16, 8
    w = torch.zeros((E, N, Kw), dtype=torch.int32)
    e4m3 = torch.zeros((E, Gn, N), dtype=torch.float8_e4m3fn)
    fp16 = torch.zeros((E, Gn, N), dtype=torch.float16)
    glob = torch.zeros((E, N), dtype=torch.float32).view(torch.int32)
    zeros = torch.zeros((E, Gn, N // 8), dtype=torch.int32)

    _check_moe_scale_pair(w, fp16, None, "w13")  # MXFP4 / folded-NVFP4: symmetric
    _check_moe_scale_pair(w, fp16, zeros, "w13")  # AWQ: packed zeros
    _check_moe_scale_pair(w, e4m3, glob, "w13")  # NVFP4 native

    with pytest.raises(AssertionError, match="REQUIRE"):
        _check_moe_scale_pair(w, e4m3, None, "w13")  # global dropped -> block scale served alone
    with pytest.raises(AssertionError, match="BITCAST"):
        _check_moe_scale_pair(w, e4m3, zeros, "w13")  # AWQ zeros under the e4m3 policy
    with pytest.raises(AssertionError, match="BITCAST"):  # right shape, VALUE-converted not bitcast
        _check_moe_scale_pair(w, e4m3, torch.zeros((E, N), dtype=torch.float32), "w13")
    with pytest.raises(AssertionError, match="AWQ packed zeros"):
        _check_moe_scale_pair(w, fp16, glob, "w13")  # global handed to the fp16 policy


# ---------------------------------------------------------------------------------------------
# 5. The accounting. A new companion tensor is a new granule row.
# ---------------------------------------------------------------------------------------------


def test_the_meta_container_and_the_analytic_byte_model_agree_exactly():
    """THE UNDER-COUNTING GUARD, and the reason it is the strong one: the meta model allocates the
    REAL container through `create_moe_quant_method`, so it sees `weight_global` whether or not
    `sizing.py` was told about it. Agreement != 1.0 means the analytic arm forgot a tensor -- which
    under-reserves the arena, spills host-budgeted weights into VRAM, and sizes the KV pool off the
    same under-count, all without an error.
    """
    from minisgl.weights.sizing import expert_stack_bytes

    class Q:
        method = "modelopt"
        bits = 4
        group_size = 16
        sym = True
        weight_type = "float"
        ct_groups = ()
        is_fp8_w8a8 = False
        is_nvfp4 = True
        weight_is_e2m1 = False
        is_int4 = False
        is_gptq = False
        is_awq = False
        is_compressed_tensors = False

    sb = expert_stack_bytes(
        quant=Q(), num_local_experts=512, hidden_size=2560, intermediate_size_per_partition=640
    )
    assert sb.source == "meta"
    assert sb.agreement == pytest.approx(1.0), (
        f"meta/analytic = {sb.agreement}: the analytic NVFP4 arm is missing a tensor the container "
        f"actually allocates"
    )
    names = [n for n, _ in sb.rows]
    assert any(n.endswith("weight_global") for n in names), names
    assert sum(nb for _, nb in sb.rows) == sb.total
    # The granule is ONE expert across BOTH GEMMs and must include its slice of the global, or an
    # offloaded serve moves an expert's weights without the second half of its scale.
    H, I = 2560, 640
    assert sb.granule_bytes == (
        (2 * I) * (H // 2) + (2 * I) * (H // G) + (2 * I) * 4
        + H * (I // 2) + H * (I // G) + H * 4
    )


def test_the_48_layer_resident_saving_is_what_the_brief_claimed():
    """The byte claim, on the real `Qwen3.8-Flash-Next-NVFP4` shape, as arithmetic rather than prose.

    Briefed: +7.03 GiB gross scale saving, 0.3516 GiB spent on the global vector, 6.68 GiB net.
    """
    from minisgl.weights.sizing import SCHEME_NVFP4, ExpertScheme, analytic_gemm_bytes

    E, H, I, L, GiB = 512, 2560, 640, 48, 2**30
    sch = ExpertScheme(SCHEME_NVFP4, bits=4, group_size=16)
    new = analytic_gemm_bytes(sch, E, 2 * I, H).total + analytic_gemm_bytes(sch, E, H, I).total
    old = (
        E * (2 * I) * (H // 2) + E * (2 * I) * (H // G) * 2
        + E * H * (I // 2) + E * H * (I // G) * 2
    )
    assert (old - new) * L / GiB == pytest.approx(6.68, abs=0.01)
    gross = (E * (2 * I) * (H // G) + E * H * (I // G)) * L
    cost = (E * (2 * I) * 4 + E * H * 4) * L
    assert gross / GiB == pytest.approx(7.03, abs=0.01)
    assert cost / GiB == pytest.approx(0.3516, abs=0.001)
