"""The QSA index-key state: a tiny per-request RING of raw keys, and the compressed-key cache.

THE ECONOMY OF THIS DESIGN, stated up front because it is the reason the memory bill is small:
**raw index keys are never a per-token cache.** A token's raw index key `k_tok` is needed only
until its group of `r` completes; once the group is averaged, normed and roped into ONE compressed
key, its members are dead. So the raw keys live in a ring of exactly `r` slots per REQUEST ROW —
`table_idx * r + pos % r` — not one slot per token. At r=4, 128 dims, bf16, 12 index layers and 64
request rows that is 64*4*128*2*12 = 786 KiB for the whole engine.

The compressed cache is one row per GROUP of r tokens, i.e. `num_kv_slots / r` rows per index
layer: at 69k KV slots that is 17.2k rows * 128 * 2 B * 12 layers = 53 MiB. It is addressed by
the **DSV4 identity**

        compressed_slot = physical_kv_slot // r

which needs no allocator, no free list and no ownership state — the KV allocator's decisions are
inherited verbatim. `config.QSAProfile.require_page_size` is what makes the identity legal (a
group must not straddle a page); it raises rather than falling back.

ONE EXTRA ROW AT THE END is the graph-capture scratch. Under cudagraph capture the compression
step has to run at a FIXED shape every step, but only one decode row in `r` actually completes a
group. Non-boundary rows therefore write to `scratch_slot` — the last row, which the DSV4 identity
can never name, because it is past `num_kv_slots // r`. (The upstream reference writes slot ZERO
for non-boundaries, which is only safe if slot 0 is a reserved null; this engine's allocator hands
out slot 0 to a real request, so a dedicated row at the end is used instead.)
"""

from __future__ import annotations

import torch

from .config import QSAProfile


class QSAIndexCache:
    """Per-layer raw-key ring + compressed-key cache for every full-attention layer."""

    def __init__(
        self,
        profile: QSAProfile,
        num_index_layers: int,
        num_req_rows: int,
        num_kv_slots: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        r = profile.compress_ratio
        d = profile.head_dim
        self.profile = profile
        self.ratio = r
        self.head_dim = d
        self.device = device
        self.dtype = dtype
        self.num_index_layers = num_index_layers
        self.num_req_rows = num_req_rows
        # ---- pending raw keys: r slots per request row, per layer -------------------------------
        self.pending = torch.zeros(
            (num_index_layers, num_req_rows * r, d), dtype=dtype, device=device
        )
        # RoPE coordinate of each pending slot. LAYER-INDEPENDENT (all index layers see the same
        # positions), so one table serves all of them — a per-layer copy would be 12 identical
        # tensors kept in sync by convention.
        self.pending_pos = torch.zeros((num_req_rows * r,), dtype=torch.int32, device=device)
        # ---- compressed keys: [slots+1, 1, 1, d] so the paged scoring op reads it as page_size=1 --
        self.num_comp_slots = num_kv_slots // r
        self.scratch_slot = self.num_comp_slots  # the capture no-op row (see the module docstring)
        self.compressed = torch.zeros(
            (num_index_layers, self.num_comp_slots + 1, 1, 1, d), dtype=dtype, device=device
        )

    # -- accessors the indexer uses (named, so a wrong layer index is a KeyError not a silent read) --
    def pending_keys(self, index_layer: int) -> torch.Tensor:
        return self.pending[index_layer]

    def compressed_keys(self, index_layer: int) -> torch.Tensor:
        """[num_comp_slots+1, 1, 1, head_dim] — the layout `qsa_index_score_paged` expects."""
        return self.compressed[index_layer]

    def ring_slots(self, table_idx: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """`table_idx * r + pos % r`, the pending-ring slot a token's raw key is written to."""
        return table_idx * self.ratio + torch.remainder(positions, self.ratio)

    def bytes(self) -> int:
        return self.pending.numel() * self.pending.element_size() + (
            self.compressed.numel() * self.compressed.element_size()
        ) + self.pending_pos.numel() * self.pending_pos.element_size()

    def __repr__(self) -> str:
        return (
            f"QSAIndexCache(layers={self.num_index_layers}, req_rows={self.num_req_rows}, "
            f"ratio={self.ratio}, head_dim={self.head_dim}, comp_slots={self.num_comp_slots}, "
            f"MiB={self.bytes() / 2**20:.1f})"
        )


__all__ = ["QSAIndexCache"]
