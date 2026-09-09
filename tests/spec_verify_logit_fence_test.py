"""The speculative verify lane read RAW logits, so the padded-vocab tail was reachable there.

MEASURED DEFECT (2026-09-10). A live serve of cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit logs at boot:

    sampler: masking padded vocab tail [248077, 248320) — 243 untrained ids fenced off

Those 243 lm_head rows are untrained padding; on an AWQ checkpoint their logits are dequant noise.
`Sampler.sample` fences them to -inf, so on the plain decode lane they are unreachable. Every
speculative verify forward bypassed that stack — the sampled lane went straight into
`probs_from_logits`, the greedy lane straight into `logits.argmax` — which put all 243 into the
top-k nucleus and into argmax contention. `MINISGL_SPEC_SAMPLED` defaults ON whenever spec-decode is
active (tools/serve.sh), so every non-greedy request took the sampled lane. CONSEQUENCE: a committed
token id outside the tokenizer's range — it detokenizes to nothing, or to a replacement character,
mid-answer.

2fba9ae closed this at ONE site (`Scheduler._spec_decode_step`) by inlining the sampler's block, and
left the three sibling verify forwards raw. `_spec_decode_step_tidar_fused` is the one that mattered
most: it commits its verify argmax directly, with no second linear verify behind it. The fix is one
shared `Sampler.condition_logits` that every lane calls.

    python3 -m pytest tests/spec_verify_logit_fence_test.py -q -o addopts=""
"""

from __future__ import annotations

import importlib
import os
import re

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from pathlib import Path  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

from minisgl.engine.sample import BatchSamplingArgs, Sampler, sample_impl  # noqa: E402
from minisgl.spec import probs_from_logits, verify_sampled  # noqa: E402

CPU = torch.device("cpu")

# A miniature of the shipped Qwen3.6-35B geometry: a real tokenizer range plus an untrained tail.
V = 64          # model / lm_head vocab (config vocab_size)
REAL = 60       # tokenizer vocab -> ids 60..63 are the untrained pad tail
PAD_ID = 62     # the pad-tail row we plant the dequant artifact in
K = 3           # draft length; verify logits are [K+1, V]


def _sampler(softcap: float | None = None, real: int | None = REAL) -> Sampler:
    return Sampler(device=CPU, vocab_size=V, real_vocab_size=real, logit_softcap=softcap)


def _verify_logits(artifact: float = 50.0) -> torch.Tensor:
    """[K+1, V] verify logits: a plausible in-range distribution, plus a huge pad-tail artifact.

    The artifact size is the point — an AWQ dequant outlier is not a near-tie, it dominates the row,
    so it wins argmax outright and takes essentially the whole softmax mass under any top_k >= 1.
    """
    torch.manual_seed(0)
    logits = torch.randn(K + 1, V) * 2.0
    logits[:, REAL:] = -5.0             # the untrained rows are otherwise unremarkable...
    logits[:, PAD_ID] = artifact        # ...except this one, which is noise the head never trained
    return logits


def _legacy_condition(smp: Sampler, logits: torch.Tensor) -> torch.Tensor:
    """The block Sampler.sample carried INLINE at 880d3cb, before condition_logits existed.

    Kept verbatim so the plain-lane no-change claim is checked against the OLD CODE rather than
    against a re-derivation of it (the repo's A/B rule: an emulated baseline proves nothing).
    """
    if os.environ.get("MINISGL_SANITIZE_LOGITS", "1") != "0":
        torch.nan_to_num_(logits, nan=-1e30, posinf=1e30, neginf=-1e30)
    if smp.logit_softcap:
        logits = torch.tanh(logits.float() / smp.logit_softcap) * smp.logit_softcap
    if smp.real_vocab_size is not None and smp.real_vocab_size < logits.shape[-1]:
        logits[:, smp.real_vocab_size:] = float("-inf")
    return logits


def _sampled_lane(logits: torch.Tensor, draft, seed: int = 7, temperature: float = 1.0,
                  top_k: int = 50, top_p: float = 0.95):
    """Exactly what Scheduler._spec_decode_step's unconstrained sampled branch does with a logit
    block: build p over the K+1 rows, then rejection-verify the draft against it."""
    gen = torch.Generator(device=CPU)
    gen.manual_seed(seed)
    p = probs_from_logits(logits, temperature, top_k, top_p)
    return verify_sampled(list(draft), p, gen)


# --- (a) the untrained pad tail must be unreachable on the sampled spec lane -----------------------

def test_raw_verify_logits_let_the_pad_tail_win_the_sampled_lane():
    """The defect itself. Without conditioning, the artifact takes the nucleus and IS emitted."""
    emitted = _sampled_lane(_verify_logits(), draft=[1, 2, 3]).emitted
    assert PAD_ID in emitted, (
        "fixture no longer reproduces the defect — the pad-tail artifact must dominate the raw "
        f"nucleus for this file to be testing anything (emitted={emitted})"
    )


def test_conditioned_verify_logits_make_the_pad_tail_unemittable():
    """The fix. Every seed, every draft: no emitted id may fall in the untrained tail."""
    smp = _sampler()
    for seed in range(64):
        logits = smp.condition_logits(_verify_logits())
        r = _sampled_lane(logits, draft=[1, 2, 3], seed=seed)
        assert all(t < REAL for t in r.emitted), (
            f"seed {seed}: sampled spec lane emitted an out-of-tokenizer id {r.emitted}"
        )


def test_a_drafted_pad_id_is_always_rejected_not_accepted():
    """A pad id can still be PROPOSED (a drafter with its own untrained head). Acceptance is
    p[i, draft_i], and the fence drives that to exactly 0, so it can never be accepted — the same
    guarantee the greedy lane gets from the masked argmax."""
    smp = _sampler()
    logits = smp.condition_logits(_verify_logits())
    for seed in range(32):
        r = _sampled_lane(logits, draft=[PAD_ID, 2, 3], seed=seed)
        assert r.num_accepted == 0, f"seed {seed}: accepted a pad-tail draft ({r})"
        assert r.emitted[0] < REAL


def test_pad_tail_cannot_win_the_greedy_verify_argmax():
    """The greedy verify lane (`logits.argmax`) reads the same conditioned tensor — including the
    on-device accept chain and the fused-TiDAR step, which commits its argmax with no re-verify."""
    smp = _sampler()
    argmax = smp.condition_logits(_verify_logits()).argmax(dim=-1)
    assert (argmax < REAL).all(), f"greedy verify argmax landed in the pad tail: {argmax.tolist()}"


def test_softcap_runs_before_the_fence_not_after():
    """Order is load-bearing: tanh(-inf)*cap is -cap, a perfectly samplable logit. Fencing after the
    softcap (as Sampler.sample has always done) is what keeps the tail at -inf."""
    smp = _sampler(softcap=30.0)
    out = smp.condition_logits(_verify_logits())
    assert torch.isinf(out[:, REAL:]).all() and (out[:, REAL:] < 0).all()
    assert out[:, :REAL].abs().max() <= 30.0


def test_no_fence_configured_leaves_every_id_reachable():
    """A tokenizer that could not be loaded must not silently disable sampling (see real_vocab_size)
    — real_vocab_size None means no mask, not an all -inf row."""
    out = _sampler(real=None).condition_logits(_verify_logits())
    assert torch.isfinite(out).all()


# --- (b) a NaN in the verify logits must not crown garbage ----------------------------------------

def test_nan_verify_row_poisons_the_whole_sampled_distribution_unconditioned():
    """Why the scrub belongs on this lane too: one NaN makes softmax return NaN across the ROW, so
    the target distribution is undefined and torch.multinomial's pick is meaningless."""
    logits = _verify_logits(artifact=1.0)
    logits[0, 7] = float("nan")
    p = probs_from_logits(logits, 1.0, 50, 0.95)
    assert torch.isnan(p[0]).any()


def test_nan_verify_row_is_scrubbed_and_the_real_max_wins():
    """After the scrub, NaN -> -1e30: removed from contention rather than crowned argmax-of-garbage.
    The row's true max is untouched and the distribution is a valid one."""
    smp = _sampler()
    raw = _verify_logits(artifact=1.0)
    want = int(raw[0, :REAL].argmax())
    raw[0, 7] = float("nan")
    raw[1, 11] = float("inf")           # the +Inf twin: it would otherwise BE the argmax
    out = smp.condition_logits(raw)
    p = probs_from_logits(out, 1.0, 50, 0.95)
    assert torch.isfinite(p).all() and torch.allclose(p.sum(-1), torch.ones(K + 1))
    assert int(out[0].argmax()) == want, "the scrub moved the argmax off the row's real maximum"
    assert out[0, 7].item() == pytest.approx(-1e30)
    for seed in range(32):
        assert all(t < REAL for t in _sampled_lane(out, [1, 2, 3], seed=seed).emitted)


def test_sanitize_off_still_fences_the_pad_tail():
    """MINISGL_SANITIZE_LOGITS=0 disables the SCRUB only. The fence is not env-gated — it is a
    correctness property of the checkpoint's geometry, not a diagnostic."""
    smp = _sampler()
    prev = os.environ.get("MINISGL_SANITIZE_LOGITS")
    os.environ["MINISGL_SANITIZE_LOGITS"] = "0"
    try:
        out = smp.condition_logits(_verify_logits())
    finally:
        if prev is None:
            os.environ.pop("MINISGL_SANITIZE_LOGITS", None)
        else:
            os.environ["MINISGL_SANITIZE_LOGITS"] = prev
    assert torch.isinf(out[:, REAL:]).all()


# --- (c) the plain lane must be byte-identical ----------------------------------------------------

@pytest.mark.parametrize("softcap", [None, 30.0])
@pytest.mark.parametrize("real", [REAL, None])
def test_condition_logits_matches_the_pre_refactor_inline_block(softcap, real):
    smp = _sampler(softcap=softcap, real=real)
    got = smp.condition_logits(_verify_logits())
    want = _legacy_condition(smp, _verify_logits())
    assert torch.equal(got, want), "the extracted helper is not the block it replaced"


def test_plain_greedy_sample_is_unchanged():
    smp = _sampler()
    got = smp.sample(_verify_logits(), BatchSamplingArgs(temperatures=None))
    want = _legacy_condition(smp, _verify_logits()).argmax(dim=-1)
    assert torch.equal(got, want.to(got.dtype))
    assert (got < REAL).all()


def test_plain_sampled_sample_is_unchanged():
    """Same seed, same draws: the refactor must not perturb the plain lane's RNG consumption either."""
    smp = _sampler()
    temps = torch.full((K + 1,), 0.8)
    torch.manual_seed(1234)
    got = smp.sample(_verify_logits(), BatchSamplingArgs(temperatures=temps))
    torch.manual_seed(1234)
    want = sample_impl(_legacy_condition(smp, _verify_logits()).float(), temps, None, None, None)
    assert torch.equal(got, want)


# --- the "one sibling only" guard: every verify forward must condition its output ------------------

# `minisgl` is a namespace package (no __init__, so no __file__) — anchor on a real module instead.
_SCHED = Path(importlib.import_module("minisgl.scheduler.scheduler").__file__).resolve()


def test_every_verify_forward_conditions_its_logits():
    """2fba9ae fenced ONE of the four verify forwards. This is the regression that catches a fifth
    one being added raw — grep, because there is no CPU-only way to drive a verify step."""
    src = _SCHED.read_text().splitlines()
    sites = [i for i, ln in enumerate(src) if "self.engine.forward_verify(" in ln
             and "logits" in ln and not ln.lstrip().startswith("#")]
    assert len(sites) >= 4, f"expected >=4 verify forwards in scheduler.py, found {len(sites)}"
    for i in sites:
        window = "\n".join(src[i : i + 14])
        assert "condition_logits" in window, (
            f"{_SCHED}:{i + 1} takes a verify forward's logits without "
            f"Sampler.condition_logits — the padded-vocab fence and NaN scrub are missing there:\n"
            + window
        )


def test_the_scheduler_holds_no_second_copy_of_the_conditioning():
    """One core, one call site's worth of policy. A scheduler that mentions real_vocab_size again is
    a re-fork of the helper (KERNEL_CORE_POLICY.md's rule, applied to the head)."""
    src = _SCHED.read_text()
    for token in ("real_vocab_size", "logit_softcap", "nan_to_num_"):
        assert not re.search(rf"\b{token}\b", src), (
            f"scheduler.py re-implements '{token}' — call Sampler.condition_logits instead"
        )
