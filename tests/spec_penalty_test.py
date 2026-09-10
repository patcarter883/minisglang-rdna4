"""Presence/frequency penalties: they must ACCUMULATE, and they must work under spec decode.

TWO MEASURED DEFECTS (2026-09-10).

1. The penalty was inert. `_penalty_plan` builds `args.pen_counts` with `torch.stack`, which COPIES;
   `_commit_penalty` did its `index_add_` into that copy, which is rebuilt every step and dropped at
   the end of it. The persistent `Sampler._pen_counts[uid]` buffer therefore never advanced past its
   creation-time seed — empty for a request that starts fresh — so `(c > 0)` was all-False and both
   penalties resolved to exactly 0.0 for the life of the request. Measured on the live serve at
   temperature 0 / top_k 1, presence_penalty=20.0:

       "banana banana banana banana banana banana banana banana banana banana"

   byte-identical to presence_penalty=0.01, reproducibly.

2. Penalised requests were refused the spec lane (`_req_spec_ok`), because every accept path reads
   the verify logits raw and the penalty would have been silently inert there too. That guard cost
   the whole spec lane the moment a serve set a non-zero presence default: with
   MINISGL_DEFAULT_PRESENCE_PENALTY=1.5 in serve.sh's table, EVERY request that did not send its own
   penalty was refused, and the spec counters read 0 drafts / 0 steps while the drafter sat loaded
   with its propose graphs captured.

   It was a guard around a missing processor, not a limit of the technique: row i of a verify block
   needs the history plus `drafts[:i]`, and the drafts are proposed and TP-broadcast BEFORE the
   verify forward, so the per-row prefix is known up front and no rollback is needed.

    python3 -m pytest tests/spec_penalty_test.py -q -o addopts=""
"""

from __future__ import annotations

import pytest
import torch

from minisgl.engine.sample import Sampler  # noqa: E402

V = 32


class FakeReq:
    def __init__(self, uid, generated=()):
        self.uid = uid
        self._gen = torch.tensor(list(generated), dtype=torch.long)

    @property
    def generated_ids(self):
        return self._gen


def sampler():
    return Sampler(device=torch.device("cpu"), vocab_size=V)


def plain_penalty(counts, presence, frequency):
    """The plain lane's subtraction, verbatim from Sampler.sample."""
    return presence * (counts > 0).float() + frequency * counts


# ------------------------------------------------------------------ 1. accumulation

def test_committed_tokens_advance_the_PERSISTENT_buffer():
    """The whole first defect. Before the fix this stayed 0.0 forever."""
    s = sampler()
    s.penalty_counts(FakeReq(1))
    s.commit_penalty_tokens(1, [3])
    assert float(s._pen_counts[1][3]) == 1.0
    s.commit_penalty_tokens(1, [3, 3])
    assert float(s._pen_counts[1][3]) == 3.0


def test_a_repeated_token_is_actually_penalised_after_it_is_said():
    """The banana case, at the unit level: once a token is committed, presence must bite."""
    s = sampler()
    c = s.penalty_counts(FakeReq(2))
    assert float(plain_penalty(c, 20.0, 0.0)[7]) == 0.0      # never said -> no penalty
    s.commit_penalty_tokens(2, [7])
    assert float(plain_penalty(s._pen_counts[2], 20.0, 0.0)[7]) == 20.0


def test_the_buffer_is_seeded_from_what_the_request_already_generated():
    s = sampler()
    c = s.penalty_counts(FakeReq(3, generated=[5, 5, 9]))
    assert (float(c[5]), float(c[9]), float(c[0])) == (2.0, 1.0, 0.0)


# ------------------------------------------------------------------ 2. the block penalty

@pytest.mark.parametrize("presence,frequency", [(1.5, 0.0), (0.0, 0.7), (1.5, 0.7), (0.0, 0.0)])
@pytest.mark.parametrize("drafts", [[], [4], [4, 4], [4, 9, 4], [1, 2, 3, 4, 5]])
def test_block_penalty_equals_applying_the_plain_penalty_position_BY_position(presence, frequency, drafts):
    """The core equivalence. Row i must see history + drafts[:i] — no more, no less."""
    s = sampler()
    base = torch.zeros(V); base[9] = 2.0            # a non-empty history, so (c>0) is exercised
    block = torch.zeros(len(drafts) + 1, V)
    s.penalise_block(block, base, presence, frequency, drafts)
    for i in range(len(drafts) + 1):
        expect_counts = base.clone()
        for t in drafts[:i]:
            expect_counts[t] += 1
        torch.testing.assert_close(-block[i], plain_penalty(expect_counts, presence, frequency),
                                   rtol=0, atol=1e-6)


def test_row_zero_is_penalised_exactly_like_the_plain_lane():
    """Position 0 of a verify block is the same position plain decode would sample — identical."""
    s = sampler()
    base = torch.zeros(V); base[2] = 3.0
    block = torch.zeros(4, V)
    s.penalise_block(block, base, 1.5, 0.7, [2, 2, 2])
    torch.testing.assert_close(-block[0], plain_penalty(base, 1.5, 0.7), rtol=0, atol=1e-6)


def test_a_zero_penalty_leaves_the_block_untouched():
    s = sampler()
    block = torch.zeros(4, V)
    s.penalise_block(block, torch.ones(V) * 5, 0.0, 0.0, [1, 2, 3])
    assert float(block.abs().max()) == 0.0


def test_padded_staged_rows_past_the_block_are_ignored_not_indexed():
    """`staged_drafts` can be padded to a captured width; the block is only q rows."""
    s = sampler()
    block = torch.zeros(3, V)                       # q=3 but 5 staged drafts
    s.penalise_block(block, torch.zeros(V), 1.0, 0.0, [1, 2, 3, 4, 5])
    assert torch.isfinite(block).all()


def test_a_draft_id_past_the_vocab_is_skipped():
    """A fenced pad id can be drafted; it can never be committed, and must not index out of range."""
    s = sampler()
    block = torch.zeros(3, V)
    s.penalise_block(block, torch.zeros(V), 1.0, 0.0, [V + 5, 1])
    assert torch.isfinite(block).all()


# ------------------------------------------------------------------ 3. rejection semantics

def test_a_rejected_tail_never_reaches_the_counts():
    """Verify rejects at n: only the accepted prefix plus the emitted token is committed, and the
    result must equal what plain decode would have accumulated over the same emitted tokens."""
    s = sampler()
    s.penalty_counts(FakeReq(4))
    s.commit_penalty_tokens(4, [8, 8, 3])           # accepted [8, 8] + emitted 3; drafts [8,8,5,5] rejected at 2
    ref = sampler()
    ref.penalty_counts(FakeReq(5))
    for t in (8, 8, 3):                             # plain decode, one token per step
        ref.commit_penalty_tokens(5, [t])
    torch.testing.assert_close(s._pen_counts[4], ref._pen_counts[5], rtol=0, atol=0)


def test_row_n_assumed_exactly_the_prefix_that_was_accepted():
    """Why no rollback is needed: the row the emitted token comes from assumed drafts[:n], and
    rows 0..n-1 were accepted, so that assumption is the truth."""
    s = sampler()
    base = torch.zeros(V)
    drafts = [6, 6, 7]
    block = torch.zeros(4, V)
    s.penalise_block(block, base, 2.0, 0.0, drafts)
    n = 2                                            # accepted 6, 6; emitted from row 2
    committed = base.clone()
    for t in drafts[:n]:
        committed[t] += 1
    torch.testing.assert_close(-block[n], plain_penalty(committed, 2.0, 0.0), rtol=0, atol=1e-6)
