from __future__ import annotations

from typing import NamedTuple, Sequence

__all__ = ["AcceptResult", "verify_greedy"]


class AcceptResult(NamedTuple):
    emitted: list[int]
    """Tokens to commit this step: t0 (always) plus one per accepted draft (1 .. K+1 tokens)."""
    num_accepted: int
    """How many of the K draft tokens matched (0 .. K). The req advances by num_accepted+1."""


def verify_greedy(draft: Sequence[int], target: Sequence[int]) -> AcceptResult:
    """Greedy speculative acceptance for one request.

    The verify forward runs the target on the K+1 query positions
    ``[confirmed, draft_0, ..., draft_{K-1}]`` and yields one next-token argmax per position:
    ``target[i] = argmax(logits at query position i)``, so ``len(target) == len(draft) + 1``.

    Greedy decoding is lossless here: ``target[0]`` is exactly the token plain decode would emit
    after ``confirmed``. If it equals the first draft, the draft "guessed right" and ``target[1]``
    (the token after that draft) is itself a valid plain-decode step, and so on. We accept the
    longest matching prefix and always emit one extra "bonus/correction" token ``target[n]``:

        n = 0
        while n < K and target[n] == draft[n]:
            n += 1
        emitted = target[0 : n+1]          # n+1 tokens

    The first ``n`` emitted tokens equal ``draft[:n]`` (their KV, computed for the draft during
    verify, is valid and kept). The bonus ``target[n]`` differs from ``draft[n]`` (or n==K), so its
    KV must be recomputed next step — exactly like a normal decode of the confirmed token. The
    caller frees the KV of the ``K - n`` rejected draft positions (rollback).

    Output is bit-identical to non-speculative greedy decoding.
    """
    K = len(draft)
    assert len(target) == K + 1, f"target must be len(draft)+1, got {len(target)} vs {K}+1"
    n = 0
    while n < K and target[n] == draft[n]:
        n += 1
    return AcceptResult(emitted=list(target[: n + 1]), num_accepted=n)
