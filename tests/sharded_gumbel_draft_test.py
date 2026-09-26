"""Sampled drafting over a vocab-parallel head draws from softmax(logits / T) (CPU, 2 simulated ranks).

The Gemma-4 assistant proposer samples each draft by Gumbel-max over each rank's vocab shard (each
rank with its OWN noise stream) and merges with ParallelLMHead.argmax_from_local's (max, id)
exchange; the rejection verify then needs q = softmax(gathered logits / T). Checks:
  * the merged sample's empirical distribution matches softmax(logits / T) (total variation);
  * the SAME noise seed on both ranks (the hazard the proposer avoids) is measurably biased;
  * gather_local_logits reassembles the full row exactly.

    PYTHONPATH=python python tests/sharded_gumbel_draft_test.py
"""
from __future__ import annotations

import sys

import torch

from minisgl.distributed import DistributedInfo
from minisgl.layers import embedding as E

VOCAB, HID, TP, T, N = 64, 16, 2, 0.7, 60000


def build_heads(weight):
    heads = []
    for rank in range(TP):
        E.get_tp_info = lambda r=rank: DistributedInfo(rank=r, size=TP)   # set_tp_info is set-once
        h = E.ParallelLMHead(VOCAB, HID)
        start, count = h.vocab_range
        w = torch.zeros(h.num_embeddings_tp, HID)
        w[:count] = weight[start:start + count]
        h.weight = w
        heads.append(h)
    return heads


class _Gather:
    def __init__(self, parts_fn):
        self.parts_fn = parts_fn

    def all_gather(self, _mine):
        return torch.cat(self.parts_fn(), dim=0)


def sample(heads, x, seeds):
    """N merged Gumbel-max draws of one row; rank r uses generator seed seeds[r]."""
    gens = [torch.Generator().manual_seed(s) for s in seeds]
    local = [h.logits_local_shard(x.expand(N, -1)) for h in heads]            # [N, count] per rank
    scores = []
    for g, lo in zip(gens, local):
        u = torch.rand(lo.shape, generator=g).clamp_(min=1e-20)
        scores.append(lo / T - torch.log(-torch.log(u)))

    def pairs():
        out = []
        for h, sc in zip(heads, scores):
            v, i = sc.max(dim=-1)
            out.append(torch.stack([v, (i + h.vocab_range[0]).float()], dim=-1))
        return out

    heads[0]._comm = _Gather(pairs)
    return heads[0].argmax_from_local(scores[0])


def main() -> int:
    E._lm_head_linear = lambda x, w, b: x.float() @ w.float().t()
    g = torch.Generator().manual_seed(1)
    weight = torch.randn(VOCAB, HID, generator=g)
    x = torch.randn(1, HID, generator=g) * 0.6
    heads = build_heads(weight)
    target = torch.softmax((x @ weight.t())[0] / T, dim=-1)

    ok = True

    def tv(toks):
        emp = torch.bincount(toks, minlength=VOCAB).float() / N
        return 0.5 * (emp - target).abs().sum().item()

    tv_ok = tv(sample(heads, x, [11, 12]))
    tv_bad = tv(sample(heads, x, [11, 11]))
    # sampling noise alone: TV of N draws from `target` itself
    tv_ref = tv(torch.multinomial(target, N, replacement=True, generator=g))
    print(f"  TV distinct seeds {tv_ok:.4f}   same seed {tv_bad:.4f}   multinomial reference {tv_ref:.4f}")
    ok &= tv_ok < 2.0 * tv_ref + 0.005
    ok &= tv_bad > 2.0 * tv_ref + 0.01

    locals_ = [h.logits_local_shard(x.expand(3, -1)) for h in heads]
    heads[0]._comm = _Gather(lambda: [lo for lo in locals_])
    full = heads[0].gather_local_logits(locals_[0])
    counts = [h.vocab_range[1] for h in heads]
    exact = torch.equal(full, torch.cat([lo[:, :c] for lo, c in zip(locals_, counts)], dim=-1))
    print(f"  gather_local_logits reassembles the full row: {exact}")
    ok &= exact
    print("ALL CHECKS PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
