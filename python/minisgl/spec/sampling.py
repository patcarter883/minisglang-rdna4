"""Sampled (rejection-sampling) speculative verify — the sampling analogue of verify_greedy.

Greedy spec accepts a draft iff it equals the target's argmax; that is lossless only for greedy
decoding, so a sampled request (temperature>0 / top_p<1 — e.g. every RSA rollout) falls back to plain
decode and the drafter never engages. Speculative sampling (Leviathan 2023 / Chen 2023) makes spec
lossless for SAMPLING: the emitted tokens are drawn from exactly the target's temp/top_k/top_p
distribution. See docs/SAMPLED_SPEC_VERIFY.md.

v1 treats the draft as a DETERMINISTIC proposal (q = onehot(draft_i)) — correct for any proposer with
no proposer changes; acceptance is `p_i(draft_i)` per position. Sampling the draft from the drafter's
own softmax (a better q, higher acceptance) is a v1.1 proposer opt-in.
"""

from __future__ import annotations

from typing import Sequence

import torch

from .accept import AcceptResult

__all__ = ["probs_from_logits", "verify_sampled"]


def probs_from_logits(
    logits: torch.Tensor, temperature: float, top_k: int, top_p: float
) -> torch.Tensor:
    """Build the target sampling distribution from logits, applying the request's temp/top_k/top_p the
    same way the fused sampler does (temperature scale -> top-k mask -> softmax -> top-p renormalize).
    ``logits`` [.., V] (any float); returns probs [.., V] fp32. temperature<=0 -> onehot(argmax)."""
    logits = logits.float()
    if temperature <= 0.0:
        out = torch.zeros_like(logits)
        out.scatter_(-1, logits.argmax(dim=-1, keepdim=True), 1.0)
        return out
    logits = logits / temperature
    V = logits.shape[-1]
    if top_k and 0 < top_k < V:
        kth = torch.topk(logits, top_k, dim=-1).values[..., -1:]  # kth-largest per row
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    if top_p and 0.0 < top_p < 1.0:
        sp, si = torch.sort(probs, descending=True, dim=-1)
        # keep a token iff the cumulative mass BEFORE it is < top_p (so the token that crosses top_p is
        # kept, matching nucleus sampling); everything after is dropped.
        keep = (sp.cumsum(dim=-1) - sp) < top_p
        sp = sp * keep
        sp = sp / sp.sum(dim=-1, keepdim=True)
        probs = torch.zeros_like(probs).scatter_(-1, si, sp)
    return probs


def verify_sampled(
    draft: Sequence[int], p: torch.Tensor, gen: torch.Generator
) -> AcceptResult:
    """Rejection-sampling acceptance for one request (deterministic-proposal q = onehot(draft)).

    ``p`` [K+1, V] fp32 = the target's per-position sampling distribution (from probs_from_logits) over
    the K draft positions plus the bonus position. Accept ``draft[i]`` with prob ``p[i, draft[i]]``
    (== min(1, p/q) with q_i(draft_i)=1); on the first reject at n, emit a residual sample from
    ``normalize(relu(p[n] - onehot(draft[n])))`` (= p[n] with draft[n] zeroed, renormalized); if all K
    accept, emit a bonus sample from ``p[K]``. Output tokens are distributed exactly as the target's
    sampler. All randomness draws from ``gen`` in a fixed order so TP ranks (identical drafts + p +
    seed) stay in lockstep without an outcome broadcast.
    """
    K = len(draft)
    assert p.shape[0] == K + 1, (p.shape, K)
    device = p.device
    n = K
    if K > 0:
        idx = torch.tensor(draft, dtype=torch.long, device=device)
        p_at = p[torch.arange(K, device=device), idx]          # [K] = p_i(draft_i)
        u = torch.rand(K, device=device, generator=gen)        # [K] accept draws (fixed order)
        rejected = u >= p_at                                    # accept iff u < p_i(draft_i)
        if bool(rejected.any()):
            n = int(rejected.float().argmax().item())           # first reject index
    if n < K:
        resid = p[n].clone()
        resid[int(draft[n])] = 0.0                              # relu(p - onehot) zeroes the draft
        s = resid.sum()
        dist = resid / s if float(s) > 0.0 else p[n]            # degenerate p==onehot -> fall back to p
        tok = int(torch.multinomial(dist, 1, generator=gen).item())
        return AcceptResult(emitted=list(draft[:n]) + [tok], num_accepted=n)
    tok = int(torch.multinomial(p[K], 1, generator=gen).item())  # all accepted -> bonus ~ p[K]
    return AcceptResult(emitted=list(draft[:K]) + [tok], num_accepted=K)
