from __future__ import annotations

from typing import TYPE_CHECKING, List, Sequence

import torch

from .base import Proposer, ProposeContext

if TYPE_CHECKING:
    from minisgl.core import Req

__all__ = ["propose_ngram", "NgramProposer"]


def propose_ngram(
    token_ids: Sequence[int] | torch.Tensor,
    *,
    num_draft: int,
    max_ngram: int,
    min_ngram: int = 1,
) -> list[int]:
    """Prompt-lookup (n-gram) speculative proposal.

    Find the most recent *earlier* occurrence of the sequence's trailing n-gram and return
    the up-to-``num_draft`` tokens that followed it — a zero-model draft (no weights, no extra
    forward). This is the cheapest possible proposer and is what the spec-decode MVP runs on
    (see ``SPEC_DECODE.md``); swapping in an MTP/EAGLE draft head later only replaces this fn.

    The trailing n-gram is the "needle"; we try the largest window first (``max_ngram``) and
    shrink to ``min_ngram``, returning on the first window size that hits. Longer matches are
    rarer but far more likely to be accepted, so they're preferred. Among occurrences of a given
    window we take the **most recent** (largest start index) — in repetitive/structured output
    the nearest prior copy is the best continuation predictor.

    Args:
        token_ids: the full token sequence so far (prompt + generated), 1-D.
        num_draft: max number of draft tokens to return (the K in "K+1 verify").
        max_ngram: largest trailing-n-gram window to match on.
        min_ngram: smallest window to fall back to (>= 1).

    Returns:
        Up to ``num_draft`` proposed token ids (possibly empty when nothing matches). Fewer than
        ``num_draft`` when the match sits near the end of the sequence.
    """
    if num_draft <= 0:
        return []
    if isinstance(token_ids, torch.Tensor):
        t = token_ids.detach().to(device="cpu", dtype=torch.long).flatten()
    else:
        t = torch.as_tensor(token_ids, dtype=torch.long)
    L = int(t.numel())
    # Need at least one token before the needle to ever match, and one after to propose.
    if L < 2:
        return []

    hi = min(max_ngram, L - 1)
    lo = max(1, min_ngram)
    for n in range(hi, lo - 1, -1):
        # Candidate match starts s span t[s : s+n]; we need a token at s+n to propose, and we
        # must exclude the needle's own position (s = L-n), so s ranges over [0, L-n-1].
        if L - n - 1 < 0:
            continue
        needle = t[L - n :]
        windows = t.unfold(0, n, 1)[: L - n]  # rows 0 .. L-n-1, each an n-gram start
        matches = (windows == needle).all(dim=1).nonzero(as_tuple=True)[0]
        if matches.numel() == 0:
            continue
        s = int(matches[-1].item())  # most recent earlier occurrence
        start = s + n
        draft = t[start : start + num_draft]
        return [int(x) for x in draft.tolist()]
    return []


class NgramProposer(Proposer):
    """Prompt-lookup proposer: zero model, draft tokens come from `propose_ngram` over each req's
    own token sequence. Owns no draft state (no `on_accept` rollback)."""

    def __init__(self, num_draft: int, ngram_max: int, ngram_min: int = 1) -> None:
        self._num_draft = num_draft
        self._ngram_max = ngram_max
        self._ngram_min = ngram_min

    def propose(self, reqs: List["Req"], num_draft: int, ctx: ProposeContext) -> List[List[int]]:
        out: List[List[int]] = []
        for req in reqs:
            # Clamp to remain_len-1 so a full accept (K_i+1 emitted) stays within the req budget.
            k_i = max(0, min(num_draft, req.remain_len - 1))
            out.append(
                propose_ngram(
                    req.input_ids, num_draft=k_i, max_ngram=self._ngram_max, min_ngram=self._ngram_min
                )
                if k_i > 0
                else []
            )
        return out
