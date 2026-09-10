"""`probs_from_logits` must build the same distribution WITHOUT sorting the vocabulary.

MEASURED DEFECT (2026-09-10). The function ran two `torch.sort`s over the full row — one for top_k,
a second for top_p — so 2 x O(V log V) per row, per position, for BOTH the target's p and the
drafter's q, at V ~ 150k. Turning on distribution-matched proposal cost **27% end-to-end** on
Qwen3.6-35B-A3B (81.4 -> 59.5 tok/s), which made the A/B for that feature meaningless: it was being
judged against a handicap, not on its merits.

It is a known cost, not a subtle one. FlashInfer measures PyTorch's sort-based top-k/top-p at ~20% of
serving time and vLLM and SGLang both replaced it. This repo had already solved it on the PLAIN lane
— that sampler is documented "no sort" — and the spec mirror never got the same treatment.

A rank-k mask needs SELECTION, not order. These tests pin that the selection form is equivalent to
the sort form, including where equivalence is subtle: ties at the k-th rank, rows already zeroed by
min_p, and the top_p-with-no-top_k path that still needs a prefix.

    python3 -m pytest tests/probs_from_logits_nosort_test.py -q -o addopts=""
"""

from __future__ import annotations

import pytest
import torch

from minisgl.spec.sampling import probs_from_logits  # noqa: E402

V = 4096


def reference(logits, temperature, top_k, top_p, min_p=0.0):
    """The ORIGINAL sort-based body, verbatim, as the oracle."""
    logits = logits.float()
    if temperature <= 0.0:
        out = torch.zeros_like(logits)
        out.scatter_(-1, logits.argmax(dim=-1, keepdim=True), 1.0)
        return out
    W = logits.shape[-1]
    probs = torch.softmax(logits / temperature, dim=-1)
    if min_p and min_p > 0.0:
        floor = probs.max(dim=-1, keepdim=True).values * min_p
        probs = probs.masked_fill(probs < floor, 0.0)
    if top_k and 0 < top_k < W:
        sp, si = torch.sort(probs, descending=True, dim=-1)
        ranks = torch.arange(W, device=probs.device).expand_as(sp)
        sp = sp.masked_fill(ranks >= top_k, 0.0)
        probs = torch.zeros_like(probs).scatter_(-1, si, sp)
    if top_p and 0.0 < top_p < 1.0:
        sp, si = torch.sort(probs, descending=True, dim=-1)
        cumsum = sp.cumsum(dim=-1)
        sp = sp.masked_fill((cumsum - sp) > top_p, 0.0)
        probs = torch.zeros_like(probs).scatter_(-1, si, sp)
    return probs / probs.sum(dim=-1, keepdim=True)


def rows(n=5, seed=0, scale=3.0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, V, generator=g, dtype=torch.float32) * scale


@pytest.mark.parametrize("top_k", [1, 20, 64, 1000, 0, -1])
@pytest.mark.parametrize("top_p", [0.95, 0.8, 1.0, 0.0])
@pytest.mark.parametrize("min_p", [0.0, 0.05])
@pytest.mark.parametrize("temperature", [1.0, 0.6])
def test_matches_the_sort_based_oracle(top_k, top_p, min_p, temperature):
    lg = rows()
    torch.testing.assert_close(
        probs_from_logits(lg, temperature, top_k, top_p, min_p),
        reference(lg, temperature, top_k, top_p, min_p), rtol=1e-6, atol=1e-7)


def test_the_SHIPPED_operating_point_is_bit_comparable():
    """Qwen3.6-35B's own generation_config: temperature 1.0, top_k 20, top_p 0.95."""
    lg = rows(n=16, seed=7)
    torch.testing.assert_close(probs_from_logits(lg, 1.0, 20, 0.95),
                               reference(lg, 1.0, 20, 0.95), rtol=0, atol=1e-8)


def test_greedy_is_untouched():
    lg = rows()
    torch.testing.assert_close(probs_from_logits(lg, 0.0, 20, 0.95),
                               reference(lg, 0.0, 20, 0.95), rtol=0, atol=0)


# ------------------------------------------------------------------ the subtle cases

def test_ties_at_the_kth_rank_still_yield_a_VALID_top_k():
    """`sort` and `topk` may break ties differently, so bit-equality is not the contract here — the
    contract is that exactly k tokens survive, all of them at or above the k-th value, and the row
    normalizes. A tie-free randn fixture cannot see this at all."""
    lg = torch.full((3, V), -20.0)
    lg[:, :40] = 5.0                                    # 40 tokens EXACTLY tied for a top-20 slot
    out = probs_from_logits(lg, 1.0, 20, 0.0)
    nz = (out > 0).sum(dim=-1)
    assert (nz == 20).all(), nz
    assert torch.allclose(out.sum(dim=-1), torch.ones(3), atol=1e-6)
    assert (out[:, :40][out[:, :40] > 0] > 0).all() and float(out[:, 40:].max()) == 0.0


def test_a_row_min_p_has_already_flattened_survives():
    """min_p can zero most of the row before top_k sees it; the surviving set must not include zeros
    that a rank mask would have admitted."""
    lg = torch.full((2, V), -30.0)
    lg[:, 0] = 10.0
    lg[:, 1] = 9.5
    out = probs_from_logits(lg, 1.0, 20, 0.0, min_p=0.5)
    ref = reference(lg, 1.0, 20, 0.0, min_p=0.5)
    torch.testing.assert_close(out, ref, rtol=1e-6, atol=1e-7)
    assert int((out > 0).sum()) <= 4


def test_top_p_with_no_top_k_takes_the_prefix_path_and_still_matches():
    """The branch with no rank mask: a bounded prefix must contain the whole nucleus."""
    lg = rows(n=4, seed=11)
    for tp in (0.5, 0.9, 0.95, 0.999):
        torch.testing.assert_close(probs_from_logits(lg, 1.0, -1, tp),
                                   reference(lg, 1.0, -1, tp), rtol=1e-6, atol=1e-7)


def test_a_deliberately_flat_row_keeps_the_right_MASS_even_though_ties_make_the_SET_ambiguous():
    """A uniform row is the worst case for the nucleus: every token is tied, so which ones the cut
    drops is arbitrary in BOTH implementations and bit-equality is not the contract — count and mass
    are. (This test originally asserted bit-equality and failed on 16 of 8192 elements, which is the
    tie-break difference and nothing else.)"""
    lg = torch.zeros(2, V)                               # perfectly uniform
    out, ref = probs_from_logits(lg, 1.0, -1, 0.999), reference(lg, 1.0, -1, 0.999)
    assert int((out > 0).sum()) == int((ref > 0).sum())
    torch.testing.assert_close(out.sum(dim=-1), torch.ones(2), rtol=0, atol=1e-6)


def test_the_prefix_FALLBACK_branch_is_exercised_and_stays_exact(monkeypatch):
    """Force the prefix to be too small for the nucleus, so the full-sort fallback must fire. Without
    this the fallback is dead code in every test — the shipped prefix is larger than any fixture."""
    import minisgl.spec.sampling as S
    monkeypatch.setattr(S, "_NUCLEUS_PREFIX", 8)         # 8 tokens cannot hold a 0.95 nucleus
    lg = rows(n=3, seed=13)
    torch.testing.assert_close(S.probs_from_logits(lg, 1.0, -1, 0.95),
                               reference(lg, 1.0, -1, 0.95), rtol=1e-6, atol=1e-7)


def test_no_filter_at_all_is_just_a_softmax():
    lg = rows()
    torch.testing.assert_close(probs_from_logits(lg, 1.0, -1, 1.0),
                               torch.softmax(lg, dim=-1), rtol=1e-6, atol=1e-7)


def test_every_row_is_a_distribution():
    for tk, tp in ((20, 0.95), (1, 1.0), (-1, 0.8), (64, 0.99)):
        out = probs_from_logits(rows(n=6, seed=3), 1.0, tk, tp)
        assert (out >= 0).all()
        torch.testing.assert_close(out.sum(dim=-1), torch.ones(6), rtol=0, atol=1e-6)
