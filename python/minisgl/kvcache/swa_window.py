"""SWA-radix window snapshot/restore — the sliding-window analog of gdn_state/cca_state clone_slot.

A SWA-hybrid model (Laguna) keeps its sliding layers in a window-bounded ring pool (last W tokens per
sequence). That ring is transient per `table_idx` and overwritten as a sequence generates, so a radix
hit that reuses a prefix would find the boundary window gone — exactly the GDN/CCA recurrent-state
problem. The fix is identical: SNAPSHOT the window (the last min(L, W) tokens' sliding K/V, all sliding
layers, in ascending absolute position) at the page-aligned prefix-commit boundary, and RESTORE it for
the reusing request.

Restore does two things:
  * seed the reusing request's ring block so a later DECODE reads the correct window, and
  * hand the extend-prefill kernel the window K/V (via metadata.swa_prefix) so the new chunk attends
    across the boundary. With a BC(=32) front-pad the extend is BIT-IDENTICAL to a cold prefill
    (tools/swa_prefix_extend_validate.py: 0.000e+00 vs cold at every boundary).

Feature-gated by the scheduler (MINISGL_SWA_RADIX); inert unless a SWA-hybrid model opts in.
"""
from __future__ import annotations

import torch

# Flash BC tile for head_dim 64/128 (attn_kernels_hip.hip kBC). The online-softmax reduces in BC-wide
# blocks; aligning the first real query row to (L mod BC) makes the extend reduction match cold's
# block grouping exactly -> bit-identical. The extend kernel front-pads with (L-Wp) % BC masked rows.
BC_ALIGN = 32


class SWAWindowSnapshotter:
    """Clone/restore the sliding-window ring KV for SWA-radix prefix caching."""

    def __init__(self, swa_kv, window: int, ring_stride: int | None = None) -> None:
        self.swa_kv = swa_kv          # MHAKVCache ring pool ([num_slots, 1, kv_heads, head_dim]/layer)
        self.W = int(window)          # window SIZE — how many recent positions the snapshot holds
        # Per-seq ring STRIDE. == W without spec (Track A ring); == window + num_draft + 1 under spec
        # (engine.py widens the ring so a K+1 verify's speculative block gets disjoint slots). The ring
        # is addressed EVERYWHERE (store/gather/decode-read in rdna4.py) as slot = table_idx*R + pos%R,
        # so the snapshot MUST use R for both the per-seq base offset AND the modulo — using W would read
        # the wrong block base (table_idx*W) and alias positions the widened ring keeps disjoint,
        # corrupting the restore. Defaults to W (spec-off) so Track A is byte-unchanged.
        self.R = int(ring_stride) if ring_stride else self.W
        self.num_layers = swa_kv.num_layers

    def _window_slots(self, table_idx: int, boundary: int, Wp: int) -> torch.Tensor:
        """Ring slots for absolute positions [boundary-Wp, boundary), in ascending position order.
        Strided by R (the ring stride), matching rdna4.py's store/gather/decode addressing exactly."""
        base = table_idx * self.R
        pos = torch.arange(boundary - Wp, boundary, device=self.swa_kv.device, dtype=torch.long)
        return base + (pos % self.R)

    def clone(self, table_idx: int, boundary: int):
        """Snapshot the window at a page-aligned prefix boundary. Precondition: the ring currently
        holds the last W positions of this sequence at device_len == boundary (true at a chunk/prefill
        commit). Returns (boundary, k_snap, v_snap) with k/v_snap [num_layers, Wp, kv_heads, head_dim]
        in ascending absolute position (Wp = min(boundary, W)). Slot-agnostic (cloned tensors)."""
        Wp = min(boundary, self.W)
        slots = self._window_slots(table_idx, boundary, Wp)
        ks, vs = [], []
        for lid in range(self.num_layers):
            ks.append(self.swa_kv.k_cache(lid)[slots, 0].clone())  # [Wp, Hk, D]
            vs.append(self.swa_kv.v_cache(lid)[slots, 0].clone())
        return (boundary, torch.stack(ks), torch.stack(vs))        # [num_layers, Wp, Hk, D]

    def restore_ring(self, table_idx: int, snap) -> None:
        """Scatter a window snapshot into table_idx's ring block (so a later DECODE reads the correct
        window). The extend-prefill reads metadata.swa_prefix, not the ring, but decode reads the ring;
        seeding here keeps both correct. Positions the reusing request itself will overwrite (its new
        tokens) are re-written by store_kv in _swa_forward after this — correct ring semantics."""
        boundary, k_snap, v_snap = snap
        Wp = k_snap.shape[1]
        slots = self._window_slots(table_idx, boundary, Wp)
        for lid in range(self.num_layers):
            self.swa_kv.k_cache(lid)[slots, 0] = k_snap[lid].to(self.swa_kv.dtype)
            self.swa_kv.v_cache(lid)[slots, 0] = v_snap[lid].to(self.swa_kv.dtype)

    @staticmethod
    def pad_for(boundary: int, Wp: int) -> int:
        """BC front-pad count so the first new-token row lands at (boundary mod BC): (L-Wp) % BC."""
        return (boundary - Wp) % BC_ALIGN
