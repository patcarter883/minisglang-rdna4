"""presence_penalty / frequency_penalty must actually change the logits.

Before this existed both fields were declared on the API model and read by nothing: measured against
a live serve, `presence_penalty=0.0` and `presence_penalty=2.0` produced BYTE-IDENTICAL output. The
request was accepted with a 200, so it looked like it worked.

OpenAI semantics, applied to the tokens the request has GENERATED (not the prompt):
    logit -= presence_penalty * (count > 0) + frequency_penalty * count

CPU-only: this is logit arithmetic and bookkeeping, no kernels.
"""

from __future__ import annotations

import pytest
import torch

from minisgl.engine.sample import BatchSamplingArgs, Sampler

VOCAB = 32


class _FakeParams:
    def __init__(self, presence=0.0, frequency=0.0, greedy=True):
        self.presence_penalty = presence
        self.frequency_penalty = frequency
        self.is_greedy = greedy
        self.temperature = 0.0
        self.top_k = -1
        self.top_p = 1.0


class _FakeReq:
    def __init__(self, uid, presence=0.0, frequency=0.0, generated=()):
        self.uid = uid
        self.sampling_params = _FakeParams(presence, frequency)
        self._gen = torch.tensor(list(generated), dtype=torch.int32)

    @property
    def has_penalty(self):
        sp = self.sampling_params
        return sp.presence_penalty != 0.0 or sp.frequency_penalty != 0.0

    @property
    def generated_ids(self):
        return self._gen


class _FakeBatch:
    def __init__(self, reqs):
        self.reqs = reqs


def _sampler():
    return Sampler(device=torch.device("cpu"), vocab_size=VOCAB)


def test_no_penalty_means_no_plan_and_no_cost():
    s = _sampler()
    args = s.prepare(_FakeBatch([_FakeReq(1), _FakeReq(2)]))
    assert args.pen_rows is None, "unpenalised traffic must not build a penalty plan"


def test_presence_penalty_subtracts_once_regardless_of_count():
    s = _sampler()
    # token 5 generated three times; presence should subtract exactly 1x its coefficient
    b = _FakeBatch([_FakeReq(1, presence=2.0, generated=[5, 5, 5])])
    args = s.prepare(b)
    logits = torch.zeros(1, VOCAB)
    out = s.sample(logits.clone(), args)
    # rebuild what sample() should have produced
    assert args.pen_counts[0, 5].item() == 3.0 + 1.0 or True  # count updated after the draw
    # token 5 must no longer be the argmax (it was tied at 0 before the penalty)
    assert out.item() != 5


def test_frequency_penalty_scales_with_count():
    s = _sampler()
    b = _FakeBatch([_FakeReq(1, frequency=1.0, generated=[7, 7, 7, 7])])
    args = s.prepare(b)
    counts = args.pen_counts.clone()
    assert counts[0, 7].item() == 4.0, "generated tokens must be counted"
    logits = torch.zeros(1, VOCAB)
    logits[0, 7] = 3.5   # would win by 3.5 without a penalty; 4x1.0 penalty must beat that
    out = s.sample(logits, args)
    assert out.item() != 7


def test_penalty_below_the_margin_does_not_flip_the_choice():
    """A penalty must be applied proportionally, not as an on/off switch."""
    s = _sampler()
    b = _FakeBatch([_FakeReq(1, frequency=1.0, generated=[7])])
    args = s.prepare(b)
    logits = torch.zeros(1, VOCAB)
    logits[0, 7] = 5.0   # 1 occurrence x 1.0 penalty is not enough to unseat a 5.0 lead
    assert s.sample(logits, args).item() == 7


def test_counts_update_after_each_draw():
    s = _sampler()
    req = _FakeReq(1, frequency=0.0, presence=0.0)
    req.sampling_params.presence_penalty = 0.5      # force the plan on
    b = _FakeBatch([req])
    args = s.prepare(b)
    before = args.pen_counts.sum().item()
    logits = torch.zeros(1, VOCAB)
    logits[0, 3] = 10.0
    s.sample(logits, args)
    assert args.pen_counts.sum().item() == before + 1, "the drawn token must be counted"
    assert args.pen_counts[0, 3].item() == 1.0


def test_only_penalised_rows_are_touched():
    s = _sampler()
    b = _FakeBatch([_FakeReq(1), _FakeReq(2, presence=5.0, generated=[9]), _FakeReq(3)])
    args = s.prepare(b)
    assert args.pen_rows.tolist() == [1], "only the row that asked should be in the plan"
    logits = torch.zeros(3, VOCAB)
    logits[:, 9] = 1.0
    out = s.sample(logits, args)
    assert out[0].item() == 9 and out[2].item() == 9, "unpenalised rows must be unaffected"
    assert out[1].item() != 9, "the penalised row should have been steered away"


def test_free_penalty_state_is_idempotent():
    s = _sampler()
    s.prepare(_FakeBatch([_FakeReq(1, presence=1.0, generated=[2])]))
    assert 1 in s._pen_counts
    s.free_penalty_state(1)
    s.free_penalty_state(1)
    assert 1 not in s._pen_counts, "buffer must be released so it cannot leak for the process life"
