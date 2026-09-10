"""Sampled (rejection-sampling) speculative verify — the sampling analogue of verify_greedy.

Greedy spec accepts a draft iff it equals the target's argmax; that is lossless only for greedy
decoding, so a sampled request (temperature>0 / top_p<1 — e.g. every RSA rollout) falls back to plain
decode and the drafter never engages. Speculative sampling (Leviathan 2023 / Chen 2023) makes spec
lossless for SAMPLING: the emitted tokens are drawn from exactly the target's temp/top_k/top_p
distribution. See docs/SAMPLED_SPEC_VERIFY.md.

Two proposal regimes reach `verify_sampled`, and the boot log says which one is live:

* DETERMINISTIC (``q=None``, ``q = onehot(draft_i)``) — correct for any proposer with no proposer
  changes, but acceptance is `p_i(draft_i)` per position, i.e. bounded by the TARGET'S ENTROPY.
* DISTRIBUTION-MATCHED (``q`` supplied) — acceptance becomes `1 - TV(p, q)`, which is not bounded
  by entropy. `verify_sampled` accepts ``q`` today; NO caller supplies one on this branch, so the
  deterministic regime is what runs. Wiring a proposer to report and draw from its own distribution
  is separate work.

Both emit exactly the target's distribution; they differ only in how often a draft survives.
"""

from __future__ import annotations

from typing import Sequence

import torch

from .accept import AcceptResult

# Prefix taken when top_p is set with NO top_k. Large enough that a nucleus of any realistic
# top_p sits inside it on a ~150k vocab; when it does not, the code falls back to the exact
# full sort -- so this is a performance constant and never a correctness one.
_NUCLEUS_PREFIX = 8192

__all__ = ["probs_from_logits", "verify_sampled"]


def probs_from_logits(
    logits: torch.Tensor, temperature: float, top_k: int, top_p: float, min_p: float = 0.0
) -> torch.Tensor:
    """Build the target sampling distribution from logits, applying the request's temp/min_p/top_k/top_p
    the same way the fused sampler does (temperature scale -> min-p floor -> top-k mask -> softmax ->
    top-p renormalize). ``logits`` [.., V] (any float); returns probs [.., V] fp32.
    temperature<=0 -> onehot(argmax).

    CONTRACT: ``logits`` must ALREADY be head-conditioned — NaN-scrubbed, softcapped, and with the
    untrained padded-vocab tail fenced to -inf. That is Sampler.condition_logits (engine/sample.py),
    which every verify forward runs on its output before anything here is reached. This function
    deliberately does not repeat it: it is called per request and per position, it has no view of
    real_vocab_size, and re-masking [K+1, V] rows once per req per step is exactly the cost that
    conditioning once per forward avoids. Fed RAW logits it puts a pad-tail dequant artifact into the
    nucleus, and top-k selects it like any other token."""
    logits = logits.float()
    if temperature <= 0.0:
        out = torch.zeros_like(logits)
        out.scatter_(-1, logits.argmax(dim=-1, keepdim=True), 1.0)
        return out
    V = logits.shape[-1]
    # Mirror engine/sample.py::sample_impl EXACTLY (the torch reference the fused HIP sampler matches):
    # softmax(logits/T) -> min_p (relative floor) -> top_k (rank mask, NO renorm) -> top_p (nucleus on
    # the un-renormalized probs) -> normalize. Applying top_p to un-renormalized top_k probs is
    # load-bearing: renormalizing between top_k and top_p shifts the nucleus and skews the distribution
    # (caught by validate_sampled_spec).
    probs = torch.softmax(logits / temperature, dim=-1)
    if min_p and min_p > 0.0:
        # min-p: drop tokens below min_p * max_prob for the row, BEFORE top-k/top-p (sample_impl's
        # processor order). The max-prob token itself always survives, so the row never zeroes out.
        floor = probs.max(dim=-1, keepdim=True).values * min_p
        probs = probs.masked_fill(probs < floor, 0.0)

    k = top_k if (top_k and 0 < top_k < V) else 0
    nucleus = top_p if (top_p and 0.0 < top_p < 1.0) else 0.0
    if not k and not nucleus:
        return probs / probs.sum(dim=-1, keepdim=True)

    # NO FULL-VOCAB SORT. This was two `torch.sort`s over the whole row -- one for top_k and a second
    # for top_p -- i.e. 2 x O(V log V) per row, per position, at V ~ 150k. FlashInfer measures
    # PyTorch's sort-based top-k/top-p at ~20% of serving time and vLLM and SGLang both replaced it;
    # this repo already solved it on the PLAIN lane, whose fused HIP sampler is documented "no sort",
    # and the spec mirror never got the same treatment.
    #
    # HONEST ABOUT THE PAYOFF: on Qwen3.6-35B-A3B (tp=2, dflash k=15, conc=2) this bought NOTHING
    # measurable -- 81.4 tok/s before, 80.3 after, inside a ~2.6% boot-to-boot band that a greedy
    # control leg exposed. It was landed on a 27% figure that turned out to be a COLD-BOOT artifact:
    # the first battery after a serve boot runs ~18% slow (65.7 vs 80.3 warm on the identical build),
    # and that cold run was being compared against a warm serve. Kept because O(V) selection beats
    # O(V log V) order asymptotically and vocabularies only grow -- not because it measured faster
    # here. Do not cite a speedup for it.
    #
    # A rank-k mask needs SELECTION, not order. `torch.topk` is O(V) with a k-heap and already returns
    # descending, so top_p's nucleus scan then runs over k elements instead of V -- both filters in one
    # pass. The processor ORDER is unchanged and still load-bearing: top_p sees UN-renormalized top_k
    # probs and the single normalize happens at the end (see the note above).
    if k:
        vals, idx = torch.topk(probs, k, dim=-1)
    else:
        # top_p with no top_k. A prefix whose mass already covers the nucleus contains every token the
        # nucleus can contain, so nothing outside it can change the answer -- the same "raise a pivot
        # until the remaining mass falls under the threshold" argument FlashInfer's sorting-free
        # sampler makes, bounded to one step. One sync confirms it held; the exact sort is the
        # fallback, never the path.
        cap = min(_NUCLEUS_PREFIX, V)
        vals, idx = torch.topk(probs, cap, dim=-1)
        if cap < V and not bool((vals.sum(dim=-1) >= nucleus).all()):
            vals, idx = torch.sort(probs, descending=True, dim=-1)
    if nucleus:
        cumsum = vals.cumsum(dim=-1)
        vals = vals.masked_fill((cumsum - vals) > nucleus, 0.0)  # keep the nucleus (matches _apply_top_p)
    probs = torch.zeros_like(probs).scatter_(-1, idx, vals)
    return probs / probs.sum(dim=-1, keepdim=True)


def verify_sampled(
    draft: Sequence[int], p: torch.Tensor, gen: torch.Generator,
    q: "torch.Tensor | None" = None,
) -> AcceptResult:
    """Rejection-sampling acceptance for one request.

    ``q`` [K, V] fp32 = the DRAFTER's per-position proposal distribution, the one each ``draft[i]``
    was actually drawn from. Standard speculative sampling: accept with ``min(1, p/q)``, and on the
    first reject emit from ``normalize(relu(p - q))``.

    ``q=None`` keeps the historical DETERMINISTIC-proposal form (q = onehot(draft)), which is exact
    but caps acceptance at ``p(draft)``: with a point-mass proposal the accept probability IS the
    target's probability of the drafted token, so even a perfect drafter proposing the target's own
    argmax is accepted only ``p(argmax)`` of the time. At this checkpoint's own temperature 1.0 that
    is a few tenths on ordinary prose, which bounds acceptance by the TARGET'S ENTROPY and has
    nothing to do with drafter quality — measured 0.266 accept rate on Qwen3.6-35B where the greedy
    lane on the same drafter is far higher. Supplying ``q`` lifts that bound: expected acceptance
    becomes ``1 - TV(p, q)``, which is high for a well-matched drafter at any entropy.

    Both forms emit exactly the target distribution; they differ only in how often the draft
    survives. Keep the ``None`` path for proposers that cannot report a distribution (n-gram, and
    any semi-autoregressive walk whose per-position conditional is not the one its token was drawn
    from) — using a soft q for a token that was drawn by argmax is NOT exact and would silently
    skew the output.

    ``p`` [K+1, V] fp32 = the target's per-position sampling distribution (from probs_from_logits) over
    the K draft positions plus the bonus position. Accept ``draft[i]`` with prob ``p[i, draft[i]]``
    (== min(1, p/q) with q_i(draft_i)=1); on the first reject at n, emit a residual sample from
    ``normalize(relu(p[n] - onehot(draft[n])))`` (= p[n] with draft[n] zeroed, renormalized); if all K
    accept, emit a bonus sample from ``p[K]``. Output tokens are distributed exactly as the target's
    sampler. All randomness draws from ``gen`` in a fixed order (rand(K), then ONE batched multinomial
    over all K+1 rows) so TP ranks (identical drafts + p + seed) stay in lockstep without an outcome
    broadcast.

    Everything runs on device and the outcome (first-reject index + the sample at every row) reaches
    host in ONE packed sync — a .item() per intermediate would serialize 3-4 stream stalls per req per
    spec step. Sampling every row's residual and selecting row n afterward leaves the emitted
    distribution unchanged: each row's draw uses disjoint generator output, so the sample at row n is
    exactly multinomial(resid_n) regardless of the other rows.
    """
    K = len(draft)
    assert p.shape[0] == K + 1, (p.shape, K)
    device = p.device
    if K == 0:
        tok = int(torch.multinomial(p[0], 1, generator=gen).item())  # bonus ~ p[0]
        return AcceptResult(emitted=[tok], num_accepted=0)
    rows = torch.arange(K, device=device)
    idx = torch.tensor(draft, dtype=torch.long, device=device)
    p_at = p[rows, idx]                                        # [K] = p_i(draft_i)
    u = torch.rand(K, device=device, generator=gen)            # [K] accept draws (fixed order)
    if q is None:
        ratio = p_at                                            # q_i(draft_i) == 1
    else:
        q_at = q[rows, idx]                                     # [K] = q_i(draft_i)
        # q_at is > 0 for a token actually drawn from q; the guard covers a proposal whose
        # distribution was reshaped after the draw (a truncation that excluded its own sample),
        # where the ratio is undefined and accepting is the conservative choice.
        ratio = torch.where(q_at > 0, p_at / q_at, torch.ones_like(p_at))
    rejected = u >= ratio                                       # accept iff u < min(1, p/q)
    # first reject index (argmax of all-False is 0, so gate it on any()); K == all accepted
    n_t = torch.where(
        rejected.any(), rejected.int().argmax(), torch.tensor(K, device=device, dtype=torch.long)
    )
    if q is None:
        resid = p.clone()
        resid[rows, idx] = 0.0                                  # relu(p - onehot) zeroes the draft
    else:
        # relu(p - q) on the K draft rows; the bonus row K has no proposal to subtract.
        resid = p.clone()
        resid[:K] = torch.clamp(p[:K] - q, min=0.0)
    # degenerate p==onehot -> fall back to p; row K stays p[K] (the bonus dist)
    dist = torch.where(resid.sum(dim=-1, keepdim=True) > 0.0, resid, p)
    toks = torch.multinomial(dist, 1, generator=gen).squeeze(-1)  # [K+1] per-row samples (normalizes)
    packed = torch.cat([n_t.reshape(1), toks]).cpu().tolist()   # the ONE sync
    n = int(packed[0])
    tok = int(packed[1 + n])                                    # residual at row n, or bonus at row K
    return AcceptResult(emitted=list(draft[:n]) + [tok], num_accepted=n)
