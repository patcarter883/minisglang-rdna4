"""CPU unit tests for the spec-decode algorithmic core (no GPU).

Run:  PYTHONPATH=python python tests/spec_core_test.py
Covers the two pure pieces that the engine/scheduler verify cycle is built on:
the n-gram proposer and greedy acceptance.
"""

from __future__ import annotations

import torch
from minisgl.spec import propose_ngram, verify_greedy


def _check(name: str, got, want) -> None:
    assert got == want, f"{name}: got {got!r}, want {want!r}"
    print(f"  ok  {name}")


def test_proposer() -> None:
    print("propose_ngram:")
    # Most-recent occurrence of the trailing bigram "a b" (1,2) is at index 4 -> propose 3,9.
    seq = [1, 2, 3, 7, 1, 2, 9, 8, 1, 2]
    _check("bigram-recent", propose_ngram(seq, num_draft=2, max_ngram=3), [9, 8])

    # No earlier occurrence of the trailing token -> empty.
    _check("no-match", propose_ngram([5, 6, 7, 8], num_draft=3, max_ngram=2), [])

    # Falls back from 3-gram (no hit) to 1-gram: trailing token 2 last seen at idx 1 -> propose 3.
    _check("ngram-fallback", propose_ngram([2, 3, 9, 9, 9, 2], num_draft=1, max_ngram=3), [3])

    # Truncates at sequence end: match at idx 0 ("1,2") is followed by 3,1,2 (only 3 tokens
    # exist, so a num_draft=4 request is truncated to those three).
    _check("truncate-at-end", propose_ngram([1, 2, 3, 1, 2], num_draft=4, max_ngram=2), [3, 1, 2])

    # num_draft<=0 and too-short sequences yield nothing.
    _check("zero-draft", propose_ngram(seq, num_draft=0, max_ngram=3), [])
    _check("too-short", propose_ngram([7], num_draft=2, max_ngram=2), [])

    # Accepts a torch tensor as well as a list.
    _check("tensor-input", propose_ngram(torch.tensor(seq), num_draft=2, max_ngram=3), [9, 8])

    # Longer match preferred over shorter even when the shorter one is more recent.
    # Trailing 3-gram "1,2,3" recurs only at idx 1 -> continuation 7.
    # Trailing 2-gram "2,3" most-recently recurs at idx 6 -> continuation 8.
    # We try the 3-gram first and hit, so we must get 7 (not the 2-gram's 8).
    seq2 = [9, 1, 2, 3, 7, 5, 2, 3, 8, 1, 2, 3]
    _check("prefer-longer", propose_ngram(seq2, num_draft=1, max_ngram=3), [7])


def test_accept() -> None:
    print("verify_greedy:")
    # All K drafts correct -> emit K+1 tokens, K accepted.
    r = verify_greedy(draft=[10, 11, 12], target=[10, 11, 12, 13])
    _check("all-accept.emitted", r.emitted, [10, 11, 12, 13])
    _check("all-accept.n", r.num_accepted, 3)

    # First draft wrong -> emit only the correction, 0 accepted.
    r = verify_greedy(draft=[10, 11], target=[99, 50, 51])
    _check("reject-first.emitted", r.emitted, [99])
    _check("reject-first.n", r.num_accepted, 0)

    # Partial: draft0 ok, draft1 wrong -> emit t0,t1 (t1 is the correction), 1 accepted.
    r = verify_greedy(draft=[10, 11, 12], target=[10, 77, 0, 0])
    _check("partial.emitted", r.emitted, [10, 77])
    _check("partial.n", r.num_accepted, 1)

    # K==0 (no drafts proposed) degrades to a plain decode step: emit the single token.
    r = verify_greedy(draft=[], target=[42])
    _check("no-draft.emitted", r.emitted, [42])
    _check("no-draft.n", r.num_accepted, 0)


if __name__ == "__main__":
    test_proposer()
    test_accept()
    print("\nALL SPEC CORE TESTS PASSED")
