"""Tranche T0.3 / GATE-2 — numeric parity of `minisgl.layers.HyperConnection` against the
sglang reference implementation of the same architecture.

Run in the serve image (torch does not import on this host); mount the sglang tree read-only so the
reference source is importable, and the checkpoint if you want the REAL layer-10 tensors:

    docker run --rm --device /dev/kfd --device /dev/dri --group-add video \
      --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE \
      --ipc host --shm-size 16gb -e ROCR_VISIBLE_DEVICES=0 \
      -v <worktree>:/engine -v /home/pat/Projects/sglang-upstream:/sglang:ro \
      -v /home/pat/.cache/hf-q4e:/model:ro \
      --entrypoint bash minisgl-rdna4:m1b-20260903 -lc \
      'PYTHONPATH=/engine/python:/opt/kernels python -m pytest \
         /engine/tests/qwen4exp_hc_parity_test.py -q -o addopts=""'

WHAT IS BEING PINNED, AND WHY IT NEEDED ITS OWN TEST
----------------------------------------------------
The hyper-connection is the residual plumbing of all 48 layers, and every constant in it fails
SILENTLY rather than loudly — a `/hc` dropped, a `.mean` written as `.sum`, the `2 *` on the inject
gate, gating on the unnormed stream instead of the normed one. Each produces the right shapes and
plausible-looking activations. The only way to retire that risk is to run the reference.

So this test does not compare against a transcription of the reference — a transcription is the
thing under suspicion. It EXTRACTS THE REFERENCE SOURCE ITSELF out of
`sglang/srt/layers/hyperconnection.py` (`GroupedGemmaRMSNorm`, and the `_mix_compute` /
`_combine_compute` closures nested in `GatedResidual.__init__`) with `ast`, execs it in an isolated
namespace holding only `torch`/`nn`/`F`, and calls it. If upstream's math moves, this test breaks.

`mix` and `combine` are checked SEPARATELY, in fp32 (where the two implementations are the same
arithmetic and agree to fp32 round-off) and in bf16 (where they differ by one extra rounding: this
repo's shared `_rms_norm` core returns the normalized value in `x.dtype` before the `(1 + w)` gain,
one rounding earlier than the reference's fp32-until-the-end — see `GroupedRMSNorm`).

CPU by design. On GPU the linears route through the engine's WMMA/GEMV kernels, which are not
bit-identical to `F.linear`; that is a kernel question, and mixing it in here would blur the one
thing this file exists to answer, which is whether the MATH is the reference's math.
"""

from __future__ import annotations

import ast
import json
import math
import os
import textwrap
from typing import Optional

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from minisgl.distributed import set_tp_info, try_get_tp_info  # noqa: E402

# Process-global and settable once. Guarded so this file can be collected in the same pytest run as
# the other qwen4exp tests, which set it too (an unguarded second call aborts COLLECTION, taking the
# whole run down, not just this module).
if try_get_tp_info() is None:
    set_tp_info(0, 1)

from minisgl.layers import GroupedRMSNorm, HyperConnection  # noqa: E402

DEV = torch.device("cpu")

# Real-config numbers (RadixArk/Qwen3.8-Flash-Next-NVFP4 `text_config`).
HC_COUNT = 4
HIDDEN = 2560
LOWRANK = 320
EPS = 1e-6
WIDE = HC_COUNT * HIDDEN

REF_CANDIDATES = (
    os.environ.get("SGLANG_HC_REF"),
    "/sglang/python/sglang/srt/layers/hyperconnection.py",
    "/home/pat/Projects/sglang-upstream/python/sglang/srt/layers/hyperconnection.py",
)
CKPT = os.environ.get("QWEN4EXP_CKPT", "/model")
# The shard that holds `model.language_model.layers.10.{attn,mlp}_hyper_connection.*`.
REAL_LAYER = 10


# ---------------------------------------------------------------------------
# the reference, lifted out of upstream's source rather than re-typed
# ---------------------------------------------------------------------------


def _ref_path() -> str:
    for cand in REF_CANDIDATES:
        if cand and os.path.exists(cand):
            return cand
    pytest.skip(
        "sglang reference hyperconnection.py not found (set SGLANG_HC_REF, or mount the tree at "
        f"/sglang). Tried: {[c for c in REF_CANDIDATES if c]}"
    )


def _load_reference(path: str) -> dict:
    """Exec `GroupedGemmaRMSNorm`, `_mix_compute` and `_combine_compute` — verbatim upstream source
    — in a namespace with nothing but torch. `_mix_compute`/`_combine_compute` are closures defined
    inside `GatedResidual.__init__`, so they cannot simply be imported; `ast` reaches them, and
    reaching them is the whole point (the reference is the oracle, not a copy of it)."""
    src = open(path).read()
    lines = src.splitlines(keepends=True)
    tree = ast.parse(src)

    def segment(node) -> str:
        return textwrap.dedent("".join(lines[node.lineno - 1 : node.end_lineno]))

    want = ("GroupedGemmaRMSNorm", "_mix_compute", "_combine_compute")
    found: dict = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in want:
            found[node.name] = segment(node)
    missing = [n for n in want if n not in found]
    if missing:
        raise AssertionError(
            f"{path} no longer defines {missing} — the reference moved, so this parity test is "
            f"testing nothing. Re-point it at the new definition instead of relaxing it."
        )

    ns = {"torch": torch, "nn": torch.nn, "F": F, "Optional": Optional}
    for name in want:
        exec(compile(found[name], f"{path}::{name}", "exec"), ns)
    return ns


@pytest.fixture(scope="module")
def ref() -> dict:
    return _load_reference(_ref_path())


def ref_norm(ns: dict, x: torch.Tensor, w_norm: torch.Tensor) -> torch.Tensor:
    # qwen4_exp builds every hyper-connection with hc_per_branch_norm=True
    # (`sglang/srt/models/qwen4_exp.py`), i.e. norm_dim = hc*H and group_size = H, and `mix` then
    # calls `self.hc_norm(hyper_input)` on the flat wide vector.
    norm = ns["GroupedGemmaRMSNorm"](WIDE, eps=EPS, group_size=HIDDEN)
    with torch.no_grad():
        norm.weight.copy_(w_norm.float())
    norm.to(w_norm.dtype)
    return norm(x)


# ---------------------------------------------------------------------------
# weights: the real layer-10 tensors when the shard is on disk, else random
# ---------------------------------------------------------------------------


def _real_weights(dtype: torch.dtype):
    """The four `layers.10.attn_hyper_connection.*` tensors from the checkpoint, or None."""
    index = os.path.join(CKPT, "model.safetensors.index.json")
    if not os.path.exists(index):
        return None
    try:
        from safetensors.torch import safe_open
    except ImportError:
        return None
    weight_map = json.load(open(index))["weight_map"]
    prefix = f"model.language_model.layers.{REAL_LAYER}.attn_hyper_connection."
    leaves = (
        "hc_norm.weight",
        "input_mix_weight_down.weight",
        "input_mix_weight_up.weight",
        "block_inject_weight.weight",
    )
    keys = [prefix + leaf for leaf in leaves]
    if any(k not in weight_map for k in keys):
        return None
    shards = {weight_map[k] for k in keys}
    if any(not os.path.exists(os.path.join(CKPT, s)) for s in shards):
        return None  # body shards still downloading
    out = {}
    for shard in shards:
        with safe_open(os.path.join(CKPT, shard), framework="pt") as f:
            for k in keys:
                if weight_map[k] == shard:
                    out[k[len(prefix) :]] = f.get_tensor(k).to(DEV, dtype)
    return out


def _random_weights(dtype: torch.dtype, seed: int = 0):
    g = torch.Generator(device="cpu").manual_seed(seed)

    def r(*shape, scale):
        return (torch.randn(*shape, generator=g) * scale).to(DEV, dtype)

    return {
        # `hc_norm` is the Gemma `(1 + w)` convention, so the checkpoint's weights are centered on 0.
        "hc_norm.weight": r(WIDE, scale=0.05),
        "input_mix_weight_down.weight": r(LOWRANK, WIDE, scale=WIDE**-0.5),
        "input_mix_weight_up.weight": r(WIDE, LOWRANK, scale=LOWRANK**-0.5),
        "block_inject_weight.weight": r(HC_COUNT, WIDE, scale=WIDE**-0.5),
    }


@pytest.fixture(scope="module")
def weight_source() -> str:
    return "real" if _real_weights(torch.float32) is not None else "random"


def _weights(dtype: torch.dtype):
    w = _real_weights(dtype)
    if w is not None:
        return w, "real"
    return _random_weights(dtype), "random"


def _build(dtype: torch.dtype, *, use_combine: bool = True, post_load: bool = True):
    """`post_load=True` is the DEFAULT because it is what the serve runs.

    `HyperConnection.post_load` folds `/hc` into the weights and packs the down/inject projections
    into one buffer (see that method). Both are exact, but "exact" is a claim, and a parity test that
    only ever ran the un-finalized layer would be pinning a path production never takes. Every test
    below therefore drives the FINALIZED layer against the reference, and
    `test_post_load_is_exact_and_idempotent` pins the un-finalized one against it."""
    w, source = _weights(dtype)
    torch.set_default_dtype(dtype)
    try:
        hc = HyperConnection(
            hidden_size=HIDDEN,
            hc_count=HC_COUNT,
            hc_lowrank=LOWRANK,
            eps=EPS,
            use_combine=use_combine,
        )
    finally:
        torch.set_default_dtype(torch.float32)
    hc.hc_norm.weight = w["hc_norm.weight"].clone()
    hc.input_mix_weight_down.weight = w["input_mix_weight_down.weight"].clone()
    hc.input_mix_weight_up.weight = w["input_mix_weight_up.weight"].clone()
    if use_combine:
        hc.block_inject_weight.weight = w["block_inject_weight.weight"].clone()
    if post_load:
        hc.post_load()
    return hc, w, source


def _inputs(dtype: torch.dtype, tokens: int = 7, seed: int = 3):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(tokens, WIDE, generator=g).to(DEV, dtype)
    y = torch.randn(tokens, HIDDEN, generator=g).to(DEV, dtype)
    return x, y


def _maxdiff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def _max_ulps(got: torch.Tensor, want: torch.Tensor) -> float:
    """Worst absolute disagreement measured in ULPs of the reference tensor's LARGEST magnitude.

    A flat `atol` is the wrong instrument in bf16: its 8-bit significand makes one ULP 3.1e-2 at
    |x| ~ 10, so an absolute 1e-3 is unreachable for ANY implementation and a passing 2e-2 says
    nothing about whether the arithmetic matched. The honest question is how many representable
    steps apart the two answers are on the grid the output actually lives on — this repo's shared
    `_rms_norm` core rounds the normalized value to `x.dtype` before the `(1 + w)` gain, one
    rounding earlier than the reference's fp32-until-the-end, so a small constant number of ULPs is
    expected and anything beyond that is a real disagreement, at any dtype.

    Scaling by the tensor max rather than per element on purpose: near a zero crossing the per
    element ULP collapses and the ratio diverges without anything being wrong.
    """
    w = want.float()
    peak = w.abs().max().item()
    if peak == 0.0:
        return 0.0
    mantissa_bits = 8 if want.dtype is torch.bfloat16 else 23
    ulp = 2.0 ** (math.floor(math.log2(peak)) - mantissa_bits)
    return _maxdiff(got, w) / ulp


# ---------------------------------------------------------------------------
# the parity tests — mix and combine, separately
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype,atol,max_ulps", [(torch.float32, 2e-6, 1.0), (torch.bfloat16, 2e-2, 4.0)]
)
def test_mix_matches_reference(ref, dtype, atol, max_ulps, weight_source):
    hc, w, source = _build(dtype)
    x, _ = _inputs(dtype)

    got, residuals = hc.mix(x)
    # (unnormed stream, normed stream, inject logits) — the third element is the inject gate's
    # pre-sigmoid, produced by the same GEMV as the low-rank down-projection (see HCResidual).
    res_raw, res_normed = residuals[0], residuals[1]

    n = ref_norm(ref, x, w["hc_norm.weight"])
    want = ref["_mix_compute"](
        n,
        w["input_mix_weight_down.weight"],
        w["input_mix_weight_up.weight"],
        HC_COUNT,
        HIDDEN,
    )

    assert got.shape == (x.shape[0], HIDDEN)
    # `mix` must hand `combine` the UNNORMED stream (the thing the block output is added to) and the
    # NORMED one (the thing the inject gate reads). Swapping them is silent.
    assert res_raw is x
    print(
        f"\n[mix/{dtype} weights={source}] max|d|={_maxdiff(got, want):.3e} "
        f"({_max_ulps(got, want):.2f} ulp)  hc_norm max|d|={_maxdiff(res_normed, n):.3e} "
        f"({_max_ulps(res_normed, n):.2f} ulp)"
    )
    torch.testing.assert_close(res_normed, n, rtol=atol, atol=atol)
    torch.testing.assert_close(got, want, rtol=atol, atol=atol)
    assert _max_ulps(res_normed, n) <= max_ulps
    assert _max_ulps(got, want) <= max_ulps


@pytest.mark.parametrize(
    "dtype,atol,max_ulps", [(torch.float32, 2e-6, 1.0), (torch.bfloat16, 2e-2, 4.0)]
)
def test_combine_matches_reference(ref, dtype, atol, max_ulps, weight_source):
    hc, w, source = _build(dtype)
    x, y = _inputs(dtype)

    _, residuals = hc.mix(x)
    got = hc.combine(y, residuals)

    n = ref_norm(ref, x, w["hc_norm.weight"])
    want = ref["_combine_compute"](
        y, x, n, w["block_inject_weight.weight"], HC_COUNT, HIDDEN
    )

    assert got.shape == x.shape
    print(
        f"\n[combine/{dtype} weights={source}] max|d|={_maxdiff(got, want):.3e} "
        f"({_max_ulps(got, want):.2f} ulp)"
    )
    torch.testing.assert_close(got, want, rtol=atol, atol=atol)
    assert _max_ulps(got, want) <= max_ulps


def test_end_to_end_block_matches_reference(ref):
    """mix -> (block) -> combine as a decoder layer runs it, in one shot. Catches a wiring error
    that the two isolated tests cannot: e.g. `combine` re-normalizing, or `mix` returning a residual
    pair the block's own output has already been folded into."""
    hc, w, _ = _build(torch.float32)
    x, y = _inputs(torch.float32)

    mixed, residuals = hc.mix(x)
    got = hc.combine(y + mixed, residuals)

    n = ref_norm(ref, x, w["hc_norm.weight"])
    ref_mixed = ref["_mix_compute"](
        n,
        w["input_mix_weight_down.weight"],
        w["input_mix_weight_up.weight"],
        HC_COUNT,
        HIDDEN,
    )
    want = ref["_combine_compute"](
        y + ref_mixed, x, n, w["block_inject_weight.weight"], HC_COUNT, HIDDEN
    )
    torch.testing.assert_close(got, want, rtol=2e-6, atol=2e-6)


# ---------------------------------------------------------------------------
# the silent-failure guards: each load-bearing constant, shown to matter
# ---------------------------------------------------------------------------


def test_each_load_bearing_constant_changes_the_output(ref):
    """Parity is only worth something if the quantities it pins are ones a plausible mistake would
    move. Each variant below is a mistake someone could make reading the formula once; all four must
    disagree with the reference by a wide margin, or the parity test above is vacuous."""
    hc, w, _ = _build(torch.float32)
    x, y = _inputs(torch.float32)
    n = ref_norm(ref, x, w["hc_norm.weight"])
    w_down = w["input_mix_weight_down.weight"]
    w_up = w["input_mix_weight_up.weight"]
    w_inj = w["block_inject_weight.weight"]

    ref_mix = ref["_mix_compute"](n, w_down, w_up, HC_COUNT, HIDDEN)
    ref_comb = ref["_combine_compute"](y, x, n, w_inj, HC_COUNT, HIDDEN)

    def unflat(t):
        return t.unflatten(-1, (HC_COUNT, HIDDEN))

    # 1. the `/ hc` before the silu
    no_div = (
        torch.sigmoid(F.linear(F.silu(F.linear(n, w_down)), w_up)).unflatten(
            -1, (HC_COUNT, HIDDEN)
        )
        * unflat(n)
    ).mean(dim=-2)
    # 2. `.mean` over branches, not `.sum`
    as_sum = ref_mix * HC_COUNT
    # 3. the leading `2 *` on the inject gate
    half_gate = unflat(x) + y.unsqueeze(-2) * torch.sigmoid(
        F.linear(n, w_inj) / HC_COUNT
    ).unsqueeze(-1)
    # 4. combine adds to the UNNORMED stream, not the normed one
    onto_normed = unflat(n) + y.unsqueeze(-2) * (
        2 * torch.sigmoid(F.linear(n, w_inj) / HC_COUNT)
    ).unsqueeze(-1)

    for name, wrong, right in (
        ("no /hc before silu", no_div, ref_mix),
        ("sum instead of mean", as_sum, ref_mix),
        ("inject gate without the 2x", half_gate.flatten(-2), ref_comb),
        ("combine onto the normed stream", onto_normed.flatten(-2), ref_comb),
    ):
        d = _maxdiff(wrong, right)
        assert d > 1e-2, f"{name!r} moves the output by only {d:.3e} — this test proves nothing"

    # ...and the implementation under test agrees with the reference, not with any of them.
    got, residuals = hc.mix(x)
    torch.testing.assert_close(got, ref_mix, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(hc.combine(y, residuals), ref_comb, rtol=2e-6, atol=2e-6)


def test_hc_norm_is_grouped_not_full_width():
    """A full-width RMSNorm over the 10240 vector has the same shape and is a different function."""
    hc, w, _ = _build(torch.float32)
    x, _ = _inputs(torch.float32)
    grouped = hc.hc_norm.forward(x)
    full = GroupedRMSNorm(WIDE, group_size=WIDE, eps=EPS)
    full.weight = w["hc_norm.weight"].clone()
    assert _maxdiff(grouped, full.forward(x)) > 1e-2


# ---------------------------------------------------------------------------
# structure: the mixer variant, the parameter set, and shapes
# ---------------------------------------------------------------------------


def test_mixer_variant_has_three_tensors_and_refuses_to_combine(ref):
    """The top-level `hyper_connection_mixer` replaces the absent final `model.norm`: it is this same
    block with `use_combine=False`, so it ships 3 tensors and its `mix` is the 10240 -> 2560 fold the
    lm_head consumes."""
    mixer, w, _ = _build(torch.float32, use_combine=False)
    keys = set(mixer.state_dict().keys())
    assert keys == {
        "hc_norm.weight",
        "input_mix_weight_down.weight",
        "input_mix_weight_up.weight",
    }
    x, y = _inputs(torch.float32)
    out, _ = mixer.mix(x)
    assert out.shape == (x.shape[0], HIDDEN)
    n = ref_norm(ref, x, w["hc_norm.weight"])
    want = ref["_mix_compute"](
        n,
        w["input_mix_weight_down.weight"],
        w["input_mix_weight_up.weight"],
        HC_COUNT,
        HIDDEN,
    )
    torch.testing.assert_close(out, want, rtol=2e-6, atol=2e-6)
    with pytest.raises(RuntimeError, match="use_combine=False"):
        mixer.combine(y, (x, n, None))


def test_state_dict_keys_and_shapes_match_the_checkpoint():
    hc, _, _ = _build(torch.float32)
    sd = hc.state_dict()
    assert {k: tuple(v.shape) for k, v in sd.items()} == {
        "hc_norm.weight": (WIDE,),
        "input_mix_weight_down.weight": (LOWRANK, WIDE),
        "input_mix_weight_up.weight": (WIDE, LOWRANK),
        "block_inject_weight.weight": (HC_COUNT, WIDE),
    }


def test_post_load_is_exact_and_idempotent(ref):
    """`post_load` folds `/hc` into the weights and packs down+inject into one GEMV. Both are
    launch-count changes that must not move a single bit, and the second one must not happen twice.

    Three separate claims, because each fails differently:
      * EXACT — the finalized layer and the un-finalized one agree to the LAST BIT, not to a
        tolerance. A `/hc` fold is only exact because 4 is a power of two; if someone changes
        `hc_count` to 3 the fold must refuse, and this comparison is what would catch it silently
        succeeding instead.
      * IDEMPOTENT — a second `post_load()` must not divide by `hc` again. That failure is
        invisible: right shapes, right keys, a uniformly 4x-small pre-activation in all 48 layers.
      * PACKED — `input_mix_weight_down.weight` / `block_inject_weight.weight` keep their checkpoint
        shapes and become disjoint views of one buffer, so byte accounting is unchanged.
    """
    plain, w, _ = _build(torch.float32, post_load=False)
    fused, _, _ = _build(torch.float32)
    x, y = _inputs(torch.float32)

    # the fold happened, and it is the exact power-of-two one
    assert fused._scale_folded is True
    torch.testing.assert_close(
        fused.input_mix_weight_down.weight * HC_COUNT,
        plain.input_mix_weight_down.weight,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        fused.block_inject_weight.weight * HC_COUNT,
        plain.block_inject_weight.weight,
        rtol=0,
        atol=0,
    )

    # packed: same shapes, one storage, disjoint and contiguous
    down, inj = fused.input_mix_weight_down.weight, fused.block_inject_weight.weight
    assert tuple(down.shape) == (LOWRANK, WIDE) and tuple(inj.shape) == (HC_COUNT, WIDE)
    assert down.is_contiguous() and inj.is_contiguous()
    assert down.data_ptr() == fused._fused_w.data_ptr()
    assert inj.data_ptr() == down.data_ptr() + down.numel() * down.element_size()

    # The packed GEMV is DECLINED here, and that is the documented rule, not an accident: the pack
    # is only legal under a row-invariant kernel, and this oracle is fp32-on-CPU, i.e. `F.linear` ->
    # BLAS, whose accumulation blocking depends on N. See `HyperConnection._fused_ok`.
    assert fused._fused_ok(x) is False

    # EXACT, both halves, against the un-finalized layer
    a_mix, a_res = plain.mix(x)
    b_mix, b_res = fused.mix(x)
    torch.testing.assert_close(b_mix, a_mix, rtol=0, atol=0)
    torch.testing.assert_close(fused.combine(y, b_res), plain.combine(y, a_res), rtol=0, atol=0)

    # IDEMPOTENT
    before = fused.input_mix_weight_down.weight.clone()
    fused.post_load()
    torch.testing.assert_close(fused.input_mix_weight_down.weight, before, rtol=0, atol=0)
    torch.testing.assert_close(fused.mix(x)[0], b_mix, rtol=0, atol=0)

    # and the un-finalized layer is still the reference's arithmetic, so neither path is untested
    n = ref_norm(ref, x, w["hc_norm.weight"])
    torch.testing.assert_close(
        a_mix,
        ref["_mix_compute"](
            n, w["input_mix_weight_down.weight"], w["input_mix_weight_up.weight"],
            HC_COUNT, HIDDEN,
        ),
        rtol=2e-6,
        atol=2e-6,
    )


def test_grouped_rms_norm_gain_cache_follows_the_weight():
    """`GroupedRMSNorm` memoizes `1 + weight`. The cache must die when the weight is REBOUND (what
    `BaseOP.load_state_dict` does) and when it is mutated IN PLACE (`copy_`), or a layer would keep
    normalizing with the gain of the checkpoint it was loaded from before."""
    norm = GroupedRMSNorm(WIDE, group_size=HIDDEN, eps=EPS)
    x, _ = _inputs(torch.float32)

    norm.weight = torch.zeros(WIDE)
    first = norm.forward(x).clone()
    torch.testing.assert_close(first, norm.forward(x), rtol=0, atol=0)  # cache hit is exact

    norm.weight.copy_(torch.full((WIDE,), 1.0))  # in-place mutation
    after_inplace = norm.forward(x)
    torch.testing.assert_close(after_inplace, first * 2.0, rtol=1e-6, atol=1e-6)

    norm.weight = torch.full((WIDE,), 3.0)  # rebind, as the loader does
    torch.testing.assert_close(norm.forward(x), first * 4.0, rtol=1e-6, atol=1e-6)


def test_real_checkpoint_shapes_when_present():
    w = _real_weights(torch.bfloat16)
    if w is None:
        pytest.skip(f"layer-{REAL_LAYER} hyper-connection shard not on disk under {CKPT}")
    assert tuple(w["hc_norm.weight"].shape) == (WIDE,)
    assert tuple(w["input_mix_weight_down.weight"].shape) == (LOWRANK, WIDE)
    assert tuple(w["input_mix_weight_up.weight"].shape) == (WIDE, LOWRANK)
    assert tuple(w["block_inject_weight.weight"].shape) == (HC_COUNT, WIDE)


def test_empty_batch_keeps_shapes(ref):
    """An idle rank still has to return the right shapes so the collectives downstream line up."""
    hc, _, _ = _build(torch.float32)
    x = torch.zeros(0, WIDE, device=DEV)
    y = torch.zeros(0, HIDDEN, device=DEV)
    mixed, residuals = hc.mix(x)
    assert mixed.shape == (0, HIDDEN)
    assert hc.combine(y, residuals).shape == (0, WIDE)


def test_wrong_width_is_rejected():
    hc, _, _ = _build(torch.float32)
    with pytest.raises(ValueError, match="wide residual"):
        hc.mix(torch.zeros(2, HIDDEN, device=DEV))
    x, _ = _inputs(torch.float32)
    _, residuals = hc.mix(x)
    with pytest.raises(ValueError, match="wide block output"):
        hc.combine(torch.zeros(x.shape[0], WIDE, device=DEV), residuals)
