"""Paged draft-KV attention for the MTP proposers — the drafter runs on the HIP decode kernel.

The MTP heads used to attend in PLAIN TORCH: every draft step, every layer, `k_buf[slot_rows]`
GATHERED (copied) the slot's whole draft ring [B, R, heads, dim], then einsum scores + softmax over
all R columns with an additive -inf mask for columns not written / no longer valid. The cost scaled
with the ring's CAPACITY, not with the live draft context, on every draft step.

Now the ring is handed to `attn_decode.flash_decode_paged` IN PLACE, viewed as page_size-1 pages
([max_slots*R, 1, heads, dim], no copy). Each row's block table lists its ring columns in POSITION
order from a per-row `base`, so logical key k is absolute position base+k at column (base+k) % R, and
`ctx_len = q + 1 - base`. The kernel reads exactly those keys — no gather, no mask, no full-width
softmax. All of it is computed on the device from static shapes, so the draft step stays capturable.

WHICH KEYS ARE VISIBLE — exactly the torch mask's set, taken FROM that mask. The proposer still keeps
the per-column absolute position (pos_buf) and computes keep = written & pa <= q & q - pa < R. That
set is CONTIGUOUS in position by construction: columns the previous slot owner or an unseeded prompt
prefix left behind are older than everything this request wrote, and columns rejected drafts
overwrote are the oldest in the window. So it is exactly positions [q - L + 1, q] with L = keep.sum(),
and the block table starts there. No approximation, no stale or foreign key ever addressed.
(Found independently by the GLM drafter port; a scalar sliding window cannot express the per-row
hole below a seeded prompt tail, which is why an earlier page-16 + window design was dropped.)
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class DraftAttnMeta:
    block_table: torch.Tensor   # [B, R] int32 — ring column (flattened with the slot) per logical key
    ctx_lens: torch.Tensor      # [B] int32 — number of visible keys, q + 1 - base


class DraftAttnBuilder:
    """Per-proposer constants + the per-step (device-only) metadata computation."""

    def __init__(self, ring: int, device: torch.device):
        self.ring = int(ring)
        self._k = torch.arange(self.ring, device=device, dtype=torch.int64)   # [R]

    def meta(self, slots: torch.Tensor, q_abs: torch.Tensor, keep: torch.Tensor) -> DraftAttnMeta:
        """slots [B] ring slot per row; q_abs [B] absolute position being written (and attended
        FROM); keep [B, R] bool, the proposer's visibility mask. All on device, no host read."""
        L = keep.sum(dim=-1)                                                            # [B]
        base = q_abs - L + 1                                                            # first visible pos
        cols = torch.remainder(base.unsqueeze(1) + self._k, self.ring)                  # [B, R]
        bt = (slots.unsqueeze(1) * self.ring + cols).to(torch.int32)
        return DraftAttnMeta(bt, L.to(torch.int32))


_decode = None


def paged_draft_attention(q: torch.Tensor, k_buf: torch.Tensor, v_buf: torch.Tensor,
                          meta: DraftAttnMeta, scale: float) -> torch.Tensor:
    """q [B, nq, hd] -> [B, nq, hd]; k_buf/v_buf [max_slots, R, nkv, hd], the ring, viewed in place.
    GQA: q head h reads kv head h // (nq // nkv) — the mapping the torch path's repeat_interleave used."""
    global _decode
    if _decode is None:
        import attn_decode   # the canonical HIP decode package; a missing build must RAISE, not fall back
        _decode = attn_decode.flash_decode_paged
    ms, R, nkv, hd = k_buf.shape
    return _decode(q.contiguous(), k_buf.view(ms * R, 1, nkv, hd), v_buf.view(ms * R, 1, nkv, hd),
                   meta.block_table, meta.ctx_lens, scale, 0)
