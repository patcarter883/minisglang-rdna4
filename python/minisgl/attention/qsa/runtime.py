"""Per-forward QSA plan + the selection driver every index layer runs.

WHAT IS SHARED AND WHAT IS PER-LAYER, because getting that split wrong is the difference between
12x and 1x of this work. Positions, the request->row mapping, the per-row visible-block window, the
compressed page table, the pending-ring slots and the compression PLAN (which groups complete this
forward, and where they are written) depend only on the BATCH — they are built ONCE per forward by
`QSAPlan.build`. What is per-layer is only: project qk, norm+rope q, store the raw key, compress the
completed groups' keys, score, top-k, expand, and turn the selected token positions into physical KV
slots. Twelve layers therefore pay one index-arithmetic bill, not twelve.

THE ORDER OF THE FOUR STAGES IS LOAD-BEARING and none of it is checkable from the output text:

  1. store the raw `k_tok` into the pending ring at `table_idx*r + pos%r`, with its rope coordinate;
  2. COMPRESS every group that completes at this forward — mean in FP32 over the r members, THEN
     the Gemma (1+w) norm, THEN rope at the group's FIRST (oldest) member's position. Mean-then-norm
     is not interchangeable with norm-then-mean, and roping at the newest member instead of the
     oldest is invisible: selection just quietly degrades;
  3. SCORE against blocks [0, (pos+1)//r) — the blocks FULLY visible to this query. The group that
     completed in step 2 is included, which is why compression must precede scoring;
  4. top-k -> expand -> physical slots.

SPARSITY IS A MEASURED OUTPUT, NOT AN ASSUMPTION. `QSASelection.visited` is the number of KV slots
the attention kernel will actually read; `dense` is what it would have read. Below the budget they
are equal BY CONSTRUCTION (the selection contains every visible token — that is the free
correctness gate), and above it the ratio is the proof that anything was skipped at all.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import torch

from minisgl.utils import init_logger

from . import ops
from .cache import QSAIndexCache
from .config import QSAProfile

if TYPE_CHECKING:
    from minisgl.core import Batch
    from minisgl.layers.rotary import RotaryEmbedding

logger = init_logger(__name__)

# Bound the dominant [query_rows, compressed_blocks] fp32 scoring workspace. Top-k is
# row-independent, so a big prefill chunk is scored in row TILES without changing any selection.
_LOGITS_BUDGET_BYTES = 128 * 1024 * 1024
# Bound the sparse-attention workspace too: the sparse call runs one kernel "sequence" per QUERY
# ROW, and attn_decode's split-KV path allocates [rows, q_heads, splits, head_dim] fp32 partials.
# At 2048 rows x 24 heads x 256 dims that is 200 MB per split, so prefill attention is row-tiled.
_ATTN_ROW_TILE = 256


@dataclass
class QSASelection:
    """One layer's selected KV, in the form the paged attention core consumes."""

    slots: torch.Tensor      # [rows, index_width] int32 — PHYSICAL kv slots, -1 padded at the tail
    lens: torch.Tensor       # [rows] int32 — valid entries, contiguous at the front of `slots`
    visited: int             # total KV rows the attention kernel will read
    dense: int               # total it would have read densely — `visited < dense` IS the sparsity


class QSAPlan:
    """Batch-derived, layer-independent QSA metadata for ONE forward."""

    __slots__ = (
        "profile", "rows", "bs", "row_seq", "logical_pos", "rope_pos", "seq_len_row",
        "row_starts", "row_ends", "max_blocks", "comp_page_table", "ring_slots",
        "write_locs", "member_rows", "member_ring", "row_table_idx", "page_stride",
        "is_prefill", "dense_total",
    )

    @staticmethod
    def build(
        profile: QSAProfile,
        cache: QSAIndexCache,
        batch: "Batch",
        page_table: torch.Tensor,
    ) -> "QSAPlan":
        """Derive everything the index layers need from `batch` + the global page table.

        `page_table` is the engine's GLOBAL table, which always treats page_size = 1: row
        `table_idx`, column `pos` holds the physical KV slot of that request's token `pos`. Every
        address in this plan is derived from it, so QSA inherits the KV allocator's decisions and
        keeps no allocation state of its own.
        """
        self = QSAPlan()
        r = profile.compress_ratio
        dev = page_table.device
        reqs = batch.reqs
        self.profile = profile
        self.bs = len(reqs)
        self.is_prefill = batch.is_prefill
        self.page_stride = page_table.shape[1]

        q_lens = [req.extend_len for req in reqs]
        starts = [req.cached_len for req in reqs]
        table = [req.table_idx for req in reqs]
        seq_lens = [req.device_len for req in reqs]
        self.rows = int(sum(q_lens))
        if self.rows == 0:
            raise ValueError("QSA plan built for an empty batch")

        # Per-QUERY-ROW vectors. Built on the host from O(bs) integers, then one H2D — the same
        # shape of construction the SWA/decode static fills use.
        row_seq: List[int] = []
        logical: List[int] = []
        for i, (q, s) in enumerate(zip(q_lens, starts)):
            row_seq.extend([i] * q)
            logical.extend(range(s, s + q))
        self.row_seq = torch.tensor(row_seq, dtype=torch.int32, device=dev)
        self.logical_pos = torch.tensor(logical, dtype=torch.int64, device=dev)
        # RoPE coordinate == logical position on a text-only serve. Kept as its own name because
        # they are NOT the same thing under speculative decode, where the model's rope coordinate
        # advances independently of the physical paged-KV position.
        self.rope_pos = self.logical_pos
        seq_len_t = torch.tensor(seq_lens, dtype=torch.int32, device=dev)
        self.seq_len_row = seq_len_t.index_select(0, self.row_seq.to(torch.long))
        table_t = torch.tensor(table, dtype=torch.int64, device=dev)
        self.row_table_idx = table_t.index_select(0, self.row_seq.to(torch.long))

        # Visible-block window: blocks [0, (pos+1)//r) are the groups FULLY visible to this query.
        self.row_starts = torch.zeros(self.rows, dtype=torch.int32, device=dev)
        self.row_ends = torch.div(
            self.logical_pos + 1, r, rounding_mode="floor"
        ).to(torch.int32)
        self.max_blocks = int(self.row_ends.max().item())

        # Compressed page table: block g of sequence s -> compressed slot, via the DSV4 identity
        # `physical_slot // r` on the group's FIRST token.
        if self.max_blocks > 0:
            first = page_table[table_t][:, : self.max_blocks * r : r]
            self.comp_page_table = torch.div(
                first, r, rounding_mode="floor"
            ).to(torch.int32).contiguous()
        else:
            self.comp_page_table = torch.zeros((self.bs, 1), dtype=torch.int32, device=dev)

        # Pending-ring slot of each token of this forward.
        self.ring_slots = cache.ring_slots(self.row_table_idx, self.logical_pos).to(torch.int32)

        # ---- compression plan: which groups complete at this forward, and where they are written --
        done = ((self.logical_pos + 1) % r) == 0
        done_rows = torch.nonzero(done, as_tuple=False).flatten()
        if done_rows.numel() == 0:
            self.write_locs = torch.zeros(0, dtype=torch.int32, device=dev)
            self.member_rows = None
            self.member_ring = None
        else:
            group_first_pos = self.logical_pos.index_select(0, done_rows) - (r - 1)
            g_table = self.row_table_idx.index_select(0, done_rows)
            first_slot = page_table.reshape(-1).index_select(
                0, g_table * self.page_stride + group_first_pos
            )
            self.write_locs = torch.div(first_slot, r, rounding_mode="floor").to(torch.int32)
            off = torch.arange(r, device=dev, dtype=torch.int64)
            if batch.is_prefill:
                # A prefill chunk is page-aligned (cached_len % page_size == 0) and page_size is a
                # multiple of r, so EVERY member of every completing group is a row of THIS
                # forward: source them from the packed chunk directly, no cache round trip.
                for req in reqs:
                    if req.cached_len % r != 0:
                        raise ValueError(
                            f"QSA prefill chunk is not group-aligned: cached_len="
                            f"{req.cached_len} is not a multiple of compress_ratio={r}. The KV "
                            f"page size must be a multiple of r so a group cannot straddle a chunk."
                        )
                self.member_rows = (done_rows - (r - 1))[:, None] + off[None, :]
                self.member_ring = None
            else:
                # Decode rows complete at most one group each, and its members are exactly the
                # pending ring window of that request row.
                base = g_table * r
                self.member_ring = base[:, None] + torch.remainder(
                    group_first_pos[:, None] + off[None, :], r
                )
                self.member_rows = None

        # What DENSE attention would have visited, for the sparsity ledger. It is the CAUSAL count
        # `min(pos+1, seq_len)`, not `seq_len`: a prefill row at position 7 of a 2048-token prompt
        # attends 8 keys, not 2048. Using seq_len here reported a prefill as "50% sparse" purely
        # because of the causal triangle, which would have been a fabricated win.
        self.dense_total = int(
            torch.minimum(self.logical_pos + 1, self.seq_len_row.to(torch.int64))
            .sum()
            .item()
        )
        return self


class QSARuntime:
    """Engine-lifetime QSA state: the profile, the index cache, and the index RoPE."""

    def __init__(
        self,
        profile: QSAProfile,
        cache: QSAIndexCache,
        rotary: "RotaryEmbedding",
        page_table: torch.Tensor,
    ) -> None:
        self.profile = profile
        self.cache = cache
        self.rotary = rotary
        self.page_table = page_table
        self.plan: Optional[QSAPlan] = None
        self.scale = ops.sqrt_scale(profile.head_dim)
        self.tap = os.environ.get("MINISGL_QSA_TAP", "0") != "0"
        self._last: dict = {}
        # Running sparsity ledger. `visited` is the number of KV rows the attention kernel actually
        # read; `dense` is what it would have read. This is the ONLY evidence that anything was
        # skipped — a "sparse" path that quietly selected everything passes every coherence test.
        self.total_visited = 0
        self.total_dense = 0
        self.last_sparsity: Optional[float] = None

    # -- per-forward -------------------------------------------------------------------------
    def prepare(self, batch: "Batch") -> None:
        if torch.cuda.is_current_stream_capturing():
            # NAMED refusal, not a confusing capture failure. `QSAPlan.build` reads request lengths
            # on the HOST, does an H2D of the derived vectors, and syncs on `row_ends.max().item()`
            # to size the block window — none of which a captured graph admits. Making it capturable
            # is the static-buffer treatment every other per-step-varying structure here already has
            # (decode's cache_seqlens/page_table, SWA's ring, GDN's state_indices): a fixed
            # `max_blocks = ceil(max_seq/r)`, per-bs preallocated logits/blocks/tokens/slots buffers,
            # and a `prepare_for_replay` that refreshes their CONTENTS in place. The SPARSE ATTENTION
            # itself is already capture-safe — `forward_sparse` calls one kernel through fixed
            # pointers at a fixed `index_width` — so this is a selection-side gap only.
            raise NotImplementedError(
                "qwen4_exp QSA selection cannot run inside a cudagraph capture (plan T5.1: it needs "
                "static per-bs workspaces and a fixed block window instead of a host .max()). Serve "
                "with --graph 0 / --cuda-graph-max-bs 0, or set MINISGL_QSA=0 to fall back to the "
                "dense path — which is bit-equivalent ONLY at or below indexer_budget and refuses "
                "above it."
            )
        self.plan = QSAPlan.build(self.profile, self.cache, batch, self.page_table)
        self._last = {}

    def select(
        self,
        index_layer: int,
        q: torch.Tensor,
        token_k: torch.Tensor,
        k_layernorm,
    ) -> QSASelection:
        """Stages 1-4 for ONE index layer. `q` is [rows, Hi, D] already normed + roped."""
        plan = self.plan
        assert plan is not None, "QSARuntime.select before prepare()"
        r = self.profile.compress_ratio
        cache = self.cache

        # 1. raw key -> pending ring (raw: NOT normed, NOT roped), with its rope coordinate.
        pend = cache.pending_keys(index_layer)
        slots = plan.ring_slots.to(torch.long)
        pend.index_copy_(0, slots, token_k.to(pend.dtype))
        cache.pending_pos.index_copy_(0, slots, plan.rope_pos.to(torch.int32))

        # 2. compress the groups that complete at this forward.
        if plan.write_locs.numel() > 0:
            if plan.member_rows is not None:
                members = token_k.index_select(0, plan.member_rows.reshape(-1))
                first_pos = plan.rope_pos.index_select(0, plan.member_rows[:, 0])
            else:
                members = pend.index_select(0, plan.member_ring.reshape(-1))
                first_pos = cache.pending_pos.index_select(
                    0, plan.member_ring[:, 0]
                ).to(torch.int64)
            pooled = members.reshape(-1, r, self.profile.head_dim).float().mean(dim=1)
            normed = k_layernorm.forward(pooled.to(token_k.dtype))
            roped = self.rotary.forward_one(first_pos, normed)
            comp = cache.compressed_keys(index_layer)
            comp.reshape(comp.shape[0], -1).index_copy_(
                0, plan.write_locs.to(torch.long), roped.to(comp.dtype)
            )

        # 3. score every row against its visible compressed blocks, THROUGH the page table.
        rows = plan.rows
        cols = max(plan.max_blocks, 1)
        block_topk = self.profile.block_topk
        blocks = torch.full((rows, block_topk), -1, dtype=torch.int32, device=q.device)
        if plan.max_blocks > 0:
            comp = cache.compressed_keys(index_layer)
            tile = _logits_row_tile(rows, cols)
            logits = torch.empty((tile, cols), dtype=torch.float32, device=q.device)
            for lo in range(0, rows, tile):
                hi = min(lo + tile, rows)
                view = logits[: hi - lo]
                ops.score_paged(
                    q[lo:hi].contiguous(), comp, plan.comp_page_table,
                    plan.row_starts[lo:hi], plan.row_ends[lo:hi], plan.row_seq[lo:hi],
                    view, self.scale,
                )
                # 4a. exact top-k of the row's window (lengths == row_ends, rows start at 0).
                ops.topk(view, plan.row_starts[lo:hi], plan.row_ends[lo:hi], blocks[lo:hi])

        # 4b. blocks -> token positions -> PHYSICAL kv slots.
        width = self.profile.index_width
        sel_tokens = torch.empty((rows, width), dtype=torch.int32, device=q.device)
        ops.expand(
            blocks, plan.logical_pos.to(torch.int32), plan.seq_len_row, sel_tokens,
            r, self.profile.budget,
        )
        lens = (sel_tokens >= 0).sum(dim=1).to(torch.int32)
        if self.tap:
            # MINISGL_QSA_TAP=1: keep this layer's selection inputs+outputs so a test can re-derive
            # them in float64 on the host. OFF by default — it pins a [rows, block_topk] and a
            # [rows, index_width] tensor per index layer, which is not something a serve should hold.
            self._last[index_layer] = {
                "q": q.detach().clone(),
                "blocks": blocks.detach().clone(),
                "tokens": sel_tokens.detach().clone(),
                "row_ends": plan.row_ends.detach().clone(),
                "comp_page_table": plan.comp_page_table.detach().clone(),
                "logical_pos": plan.logical_pos.detach().clone(),
                "seq_len_row": plan.seq_len_row.detach().clone(),
                "row_seq": plan.row_seq.detach().clone(),
                "compressed": cache.compressed_keys(index_layer).detach().clone(),
            }
        flat = (
            plan.row_table_idx[:, None] * plan.page_stride
            + sel_tokens.clamp(min=0).to(torch.int64)
        )
        sel_slots = self.page_table.reshape(-1).index_select(0, flat.reshape(-1)).reshape(rows, width)
        sel_slots = sel_slots.to(torch.int32).contiguous()
        visited = int(lens.to(torch.int64).sum().item())
        self.total_visited += visited
        self.total_dense += plan.dense_total
        self.last_sparsity = visited / plan.dense_total if plan.dense_total else None
        return QSASelection(
            slots=sel_slots, lens=lens, visited=visited, dense=plan.dense_total
        )


def _logits_row_tile(rows: int, cols: int) -> int:
    """Row tile that keeps the fp32 [rows, blocks] scoring workspace under the budget."""
    if rows <= 0 or cols <= 0:
        return max(rows, 1)
    per_row = cols * 4
    return max(1, min(rows, max(1, _LOGITS_BUDGET_BYTES // per_row)))


__all__ = ["QSAPlan", "QSARuntime", "QSASelection", "_ATTN_ROW_TILE"]
