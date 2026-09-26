"""ParallelLMHead.argmax_all_rows == argmax over the full vocab row, for every TP rank (CPU).

Two ranks are simulated in one process: each rank's head holds its own vocab shard and the
all_gather is replaced by a stub that stacks both ranks' (max, id) pairs, exactly as the collective
would. Covers an odd vocab (padded last shard), a tie inside a shard and a tie across the shard
boundary (the full-row argmax takes the lower id).

    PYTHONPATH=python python tests/lm_head_argmax_all_rows_test.py
"""
from __future__ import annotations

import sys

import torch

from minisgl.distributed import DistributedInfo
from minisgl.layers import embedding as E

VOCAB, HID, ROWS, TP = 1001, 64, 6, 2


def _lm_head_linear_ref(x, weight, bias):
    return x.float() @ weight.float().t()


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
    """all_gather over the simulated ranks: every rank contributes its own pair tensor."""

    def __init__(self, heads, x):
        self.heads, self.x = heads, x

    def all_gather(self, _mine):
        parts = []
        for h in self.heads:
            local = h.logits_local_shard(self.x)
            v, i = local.max(dim=-1)
            parts.append(torch.stack([v, (i + h.vocab_range[0]).float()], dim=-1))
        return torch.cat(parts, dim=0)


def main() -> int:
    E._lm_head_linear = _lm_head_linear_ref       # CPU: plain fp32 matmul
    g = torch.Generator().manual_seed(0)
    weight = torch.randn(VOCAB, HID, generator=g)
    x = torch.randn(ROWS, HID, generator=g)
    # row 1: exact tie inside rank 0's shard; row 2: exact tie across the shard boundary.
    weight[40] = weight[7]
    x[1] = weight[7] * 3
    boundary = build_heads(weight)[1].vocab_range[0]
    weight[boundary + 3] = weight[boundary - 5]
    x[2] = weight[boundary - 5] * 3

    heads = build_heads(weight)
    full = (x @ weight.t()).argmax(dim=-1)
    ok = True
    for rank, h in enumerate(heads):
        h._comm = _Gather(heads, x)
        got = h.argmax_all_rows(x)
        match = torch.equal(got, full)
        ok &= match
        print(f"  {'OK  ' if match else 'FAIL'}  rank {rank}: {got.tolist()} vs full-row {full.tolist()}")
    assert int(full[1]) == 7 and int(full[2]) == boundary - 5, "fixture ties did not form"
    print("ALL CHECKS PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
