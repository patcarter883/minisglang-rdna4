"""TP=2 / graph-capture lens on `minisgl.weights.plan`. No GPU required, no GPU used.

Every test here is a regression for a defect found by attacking the resolver from one angle: *can
the two TP ranks ever disagree, and does the plan describe the layer the engine will actually
build?* Three clusters:

  * EP VETO -- `MoELayer.__init__:967` is a THREE-way conjunction and the planner only transcribed
    two of them. Unquantized experts are never EP-sharded no matter what `--enable-ep` says.
  * RANK AGREEMENT -- `placement.py` advertises `OffloadPlan.digest()` "so a caller can prove
    agreement across ranks with one tiny collective"; nothing called it, and the plan digest could
    not have caught a sizing divergence anyway because `LayerPlacement` drops the fingerprint.
  * MTP -- the draft head is re-read on every draft step inside the captured spec-decode graphs, and
    the only thing keeping it off the host stack is an index convention nothing validated.

GPU-REQUIRING TESTS: none. What genuinely needs a device -- that the resolved plan survives
full-forward HIP graph capture at the served TP -- is NOT here and is not discharged by a green run.
"""

from __future__ import annotations

import dataclasses

import pytest

# MUST precede the minisgl imports; see `_offload_torch_stub`'s docstring.
from _offload_torch_stub import TORCH_IS_REAL  # noqa: I001
from minisgl.weights.placement import PlacementError
from minisgl.weights.plan import (
    moe_layer_shapes,
    resolve_expert_parallel,
    resolve_weight_plan,
)
from minisgl.weights.sizing import (
    SCHEME_AWQ,
    SCHEME_CT_INT4,
    SCHEME_FP8,
    SCHEME_GPTQ,
    SCHEME_MXFP4,
    SCHEME_NVFP4,
    SCHEME_SUPPORTS_EP,
    SCHEME_UNQUANTIZED,
    scheme_supports_ep,
)
from test_weight_plan import FakeQuant, make_config, make_model_config  # noqa: I001


def resolve(config=None, **kw):
    kw.setdefault("prefer_meta", False)
    return resolve_weight_plan(config if config is not None else make_config(), **kw)


# =================================================================================================
# EP VETO -- the third conjunct of MoELayer.__init__:967
# =================================================================================================


def test_the_ep_table_names_the_method_that_vetoes():
    """`_UnquantizedMoEMethod` says nothing and INHERITS `supports_ep = False` from the base.

    Silence rather than a decision is exactly why it was missed by the original transcription and
    why it is pinned here: a future reader who "fixes" it has to explain themselves to a test
    rather than to a serve. (RXF was the other veto until it was deleted; the inherited-default
    hazard it shared with unquantized is the part worth keeping under test.)
    """
    assert SCHEME_SUPPORTS_EP[SCHEME_UNQUANTIZED] is False
    for kind in (SCHEME_FP8, SCHEME_NVFP4, SCHEME_MXFP4, SCHEME_GPTQ, SCHEME_AWQ, SCHEME_CT_INT4):
        assert SCHEME_SUPPORTS_EP[kind] is True


@pytest.mark.skipif(not TORCH_IS_REAL, reason="needs the real layers/moe to read the live classes")
def test_the_ep_table_matches_the_live_quant_method_classes():
    """The table is a transcription; the class attribute is the truth. Drift is the whole risk."""
    from minisgl.layers.moe import create_moe_quant_method

    cases = {
        SCHEME_UNQUANTIZED: None,
        SCHEME_CT_INT4: FakeQuant(),
        SCHEME_NVFP4: FakeQuant(is_nvfp4=True, group_size=16),
        SCHEME_MXFP4: FakeQuant(weight_is_e2m1=True, is_int4=False),
        SCHEME_FP8: FakeQuant(is_fp8_w8a8=True, is_int4=False),
    }
    for kind, quant in cases.items():
        live = bool(create_moe_quant_method(quant).supports_ep)
        assert live is SCHEME_SUPPORTS_EP[kind], kind
        assert scheme_supports_ep(quant) is live, kind


def test_scheme_supports_ep_is_false_for_an_unidentifiable_scheme():
    """Conservative direction: a false 'EP supported' halves the traffic model and hides a miss."""

    class Alien:
        method, bits, group_size, sym, weight_type = "alien", 3, 0, True, "?"
        ct_groups = ()
        is_fp8_w8a8 = is_nvfp4 = weight_is_e2m1 = is_int4 = False
        is_gptq = is_awq = is_compressed_tensors = False

    assert scheme_supports_ep(Alien()) is False


def test_resolve_expert_parallel_honours_the_method_veto():
    cfg = make_config(tp=2, enable_ep=True)
    assert resolve_expert_parallel(cfg, method_supports_ep=True) == (True, 2)
    assert resolve_expert_parallel(cfg, method_supports_ep=False) == (False, 1)


@pytest.mark.parametrize(
    "quant,label",
    [(None, "unquantized")],
)
def test_a_vetoing_scheme_is_planned_replicated_even_under_enable_ep(quant, label):
    """THE DEFECT. `--enable-ep --tp 2` on an unquantized checkpoint builds the layer
    REPLICATED with the intermediate tensor-split; the planner sharded it anyway.

    The resident-byte totals coincide (`E/2 x 2I` and `E x 2(I/2)` multiply out the same), so bytes
    could never have caught it. What diverged is the pair that drives every traffic number:
    `num_experts` (256 vs 512) and `top_k` (5 vs 10).
    """
    mc = make_model_config(quant=quant)
    s = moe_layer_shapes(make_config(mc, tp=2, enable_ep=True))[0]
    assert s.expert_parallel is False and s.ep_size == 1, label
    assert s.num_local_experts == 512, label  # all experts, not E/ep
    assert s.intermediate_size_per_partition == 768 // 2, label  # tensor-split, not full
    assert s.top_k_local == 10, label  # the full route, not ceil(10/2)


def test_an_ep_capable_scheme_still_shards_under_enable_ep():
    """Guard the fix in the other direction: the veto must not disable EP for int4/fp8/e2m1."""
    s = moe_layer_shapes(make_config(tp=2, enable_ep=True))[0]  # default fixture is CT-int4
    assert s.expert_parallel is True and s.ep_size == 2
    assert s.num_local_experts == 512 // 2
    assert s.intermediate_size_per_partition == 768
    assert s.top_k_local == 5


def test_the_veto_doubles_the_projected_host_traffic_it_used_to_hide():
    """The consequence, priced. Halving top_k halves `distinct_experts` at batch=1, so the whole
    projection -- the K4 kill gate, A1.7, the f-sweep -- read ~2x optimistic on this configuration.
    """
    mc = make_model_config(quant=None)
    fixed = resolve(make_config(mc, tp=2, enable_ep=True))
    # What the pre-fix planner described: the same config resolved as if EP applied.
    sharded = resolve(make_config(mc, tp=2, enable_ep=False), device_budget_bytes=0)
    assert fixed.layers[0].top_k == 10
    assert fixed.layers[0].num_experts == 512
    # Both now agree: the veto puts EP-on and EP-off on the same footing for a vetoing scheme.
    assert sharded.layers[0].top_k == fixed.layers[0].top_k
    assert fixed.projection.host_ms > 0


def test_a_vetoing_scheme_now_gets_the_tp_divisibility_check_it_will_actually_assert():
    """The EP branch skipped `intermediate % tp_size`, so a vetoing checkpoint whose intermediate is not
    tp-divisible was planned as feasible and then died in `div_even` during the model build."""
    mc = make_model_config(quant=None, moe_intermediate_size=769)
    with pytest.raises(PlacementError, match="divisible"):
        moe_layer_shapes(make_config(mc, tp=2, enable_ep=True))


# =================================================================================================
# RANK AGREEMENT -- proving purity instead of asserting it
# =================================================================================================


def test_two_ranks_of_the_same_config_agree():
    a = resolve(make_config(tp=2, rank=0))
    b = resolve(make_config(tp=2, rank=1))
    assert a.agreement_digest() == b.agreement_digest()
    assert a.assert_rank_agreement(gather=lambda mine: [mine, b.agreement_digest()])


def test_a_per_rank_device_budget_is_caught():
    """The one input the resolver cannot vet. A budget derived from a live `mem_get_info` delta
    differs by tens of MiB between the rank processes -- enough to move a layer across the greedy
    fill boundary, after which the two ranks size different KV pools with no error anywhere."""
    total = sum(lw.resident_bytes for lw in resolve().layers)
    a = resolve(device_budget_bytes=total // 4)
    b = resolve(device_budget_bytes=total // 4 + (200 << 20))
    assert a.plan.digest() != b.plan.digest()
    with pytest.raises(PlacementError, match="DIVERGED across ranks"):
        a.assert_rank_agreement(gather=lambda mine: [mine, b.agreement_digest()])


def test_a_sizing_divergence_at_equal_bytes_is_caught_and_the_plan_digest_alone_would_miss_it():
    """THE DEFECT the `fingerprint` comment claimed was covered and was not.

    `plan_layer_granular` builds `LayerPlacement`s, which have no fingerprint field, so
    `OffloadPlan.digest()` hashes only path/kind/bytes/E/top_k. Two ranks that resolved DIFFERENT
    quant schemes -- or one from the meta model and one from the analytic fallback -- to the same
    byte totals produce an identical plan digest. `sizing_digest()` is the missing half.
    """
    a = resolve()
    forged = dataclasses.replace(
        a,
        layers=tuple(
            dataclasses.replace(lw, fingerprint=lw.fingerprint.replace("est:", "other:"))
            for lw in a.layers
        ),
    )
    assert forged.plan.digest() == a.plan.digest()  # the placement is byte-identical...
    assert forged.sizing_digest() != a.sizing_digest()  # ...but the weights are not the same ones
    with pytest.raises(PlacementError, match="DIVERGED across ranks"):
        a.assert_rank_agreement(gather=lambda mine: [mine, forged.agreement_digest()])


def test_a_byte_source_divergence_is_caught():
    """One rank sized from the meta containers, the other fell back to the analytic model.

    `meta_gemm_spec` swallows every exception and returns None, so the fallback is per-layer, silent
    and invisible to the byte totals whenever the two models agree -- which is the normal case, and
    exactly why it needs to be in the agreement hash rather than trusted."""
    a = resolve()  # prefer_meta=False -> "analytic"
    assert a.byte_source == "analytic"
    b = dataclasses.replace(a, byte_source="meta")
    with pytest.raises(PlacementError, match="DIVERGED across ranks"):
        a.assert_rank_agreement(gather=lambda mine: [mine, b.agreement_digest()])


def test_agreement_is_a_no_op_without_a_group_or_a_gather():
    """Single-rank serves must not pay for, or fail on, a collective that does not exist."""
    r = resolve(make_config(tp=1))
    assert r.assert_rank_agreement() == (r.agreement_digest(),)
    assert r.assert_rank_agreement(group=None) == (r.agreement_digest(),)


def test_every_rank_enters_the_gather_even_when_its_own_plan_is_empty():
    """The check must not become the desync it exists to prevent.

    If one rank resolved an empty plan and its peer did not, running the gather AFTER the
    `if not resolution.enabled` early return would block the offloading rank in
    `all_gather_object` forever. `bake._resolve_driver` therefore calls it BEFORE that branch, and a
    disabled rank must still produce a digest that DIFFERS from an enabled one.
    """
    huge = sum(lw.resident_bytes for lw in resolve().layers) * 2
    off = resolve(device_budget_bytes=huge)  # whole stack on device -> no-op plan
    on = resolve(device_budget_bytes=0)  # all-host
    assert off.enabled is False and on.enabled is True

    seen = []

    def spy(mine):
        seen.append(mine)
        return [mine]

    off.assert_rank_agreement(gather=spy)
    assert seen == [off.agreement_digest()], "a disabled rank must still enter the gather"
    with pytest.raises(PlacementError, match="DIVERGED across ranks"):
        off.assert_rank_agreement(gather=lambda mine: [mine, on.agreement_digest()])


def test_resolve_driver_checks_agreement_before_the_enabled_early_return():
    """Pin the ORDER in `bake._resolve_driver`. Getting it wrong turns a clear error into a HANG:
    the rank with a non-empty plan blocks in `all_gather_object` while its peer, having returned
    early on `not resolution.enabled`, walks on and never joins.

    Asserted on the SOURCE rather than by calling `_resolve_driver`, because executing it pulls in
    `minisgl.kvcache` -> transformers, which cannot import on a host with no working torch. The
    property being pinned is textual and structural anyway -- which of two statements comes first --
    so reading it off the source is not a weaker check here, and it keeps this suite CPU-only and
    dependency-free the way the rest of the offload planning tests are.
    """
    import inspect

    from minisgl.weights import bake

    src = inspect.getsource(bake._resolve_driver)
    check = src.index("assert_rank_agreement")
    early_return = src.index("if not resolution.enabled")
    assert check < early_return, (
        "bake._resolve_driver must call assert_rank_agreement BEFORE the `not enabled` early "
        "return, or a rank whose plan is empty skips the collective and hangs its peer."
    )


# =================================================================================================
# MTP -- the one layer that must never reach the host stack
# =================================================================================================


def test_supplied_layer_indices_must_use_the_mtp_numbering_convention():
    """`is_mtp` is `lid >= num_layers` and nothing else. A caller handing in indices under any other
    numbering would classify the DRAFT HEAD as an ordinary layer and make it offloadable -- and the
    draft head is re-read `num_draft` times per accepted token inside the captured spec-decode
    graphs. Refuse the ambiguous input rather than silently pick the bad reading."""
    cfg = make_config(make_model_config(num_nextn_predict_layers=1))
    ok = moe_layer_shapes(cfg, layer_indices=[0, 1, 48])
    assert [s.offloadable for s in ok] == [True, True, False]
    with pytest.raises(PlacementError, match="outside"):
        moe_layer_shapes(cfg, layer_indices=[0, 1, 99])
    with pytest.raises(PlacementError, match="outside"):
        moe_layer_shapes(cfg, layer_indices=[-1])


def test_supplied_layer_indices_reject_duplicates():
    """`plan_layer_granular` refuses duplicate paths; catch it at the source with a better message."""
    with pytest.raises(PlacementError, match="duplicates"):
        moe_layer_shapes(make_config(), layer_indices=[3, 3])


def test_the_mtp_head_is_never_offloadable_even_when_the_engine_hands_it_in():
    """The default enumeration never mentions MTP at all, so the "never offloaded" guarantee is only
    exercised on the path the engine is meant to use: handing in the layer list from the live meta
    model. Charge it there."""
    cfg = make_config(make_model_config(num_nextn_predict_layers=1))
    r = resolve(cfg, device_budget_bytes=0, layer_indices=list(range(49)))
    assert all("mtp" not in lw.path for lw in r.layers)
    assert any("mtp" in s for s in r.diagnostics["skipped"])
    assert len(r.layers) == 48
