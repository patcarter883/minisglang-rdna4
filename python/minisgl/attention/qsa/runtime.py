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

CUDAGRAPH CAPTURE — THE STATIC DECODE PLAN (was a named NotImplementedError until 2026-09-06)
--------------------------------------------------------------------------------------------
A captured graph admits no host reads, no host syncs, no host->device copy of a python list and no
python branch on a device value. `QSAPlan.build` does all four. So a DECODE batch takes a second,
STATIC path — the same treatment decode's `cache_seqlens`/page_table, SWA's ring and GDN's
`state_indices` already have — installed by `QSARuntime.init_capture` and refreshed in place by
`QSARuntime.prepare_for_replay`:

  * `max_blocks` is `ceil(max_seq_len / r)`, a BUILD-TIME constant, not `row_ends.max().item()`.
    Every launch grid downstream is therefore shape-derived and constant across capture and replay:
    the scorer's is `(rows, ceil(max_blocks/256))` and the split top-k's `num_splits` is a policy on
    the logits tensor's STATIC column width (`qsa_index` README, "caller obligation"). A row whose
    window is shorter simply leaves the trailing key tiles/slabs writing -inf and doing no work.
  * logits / blocks / tokens / slots / lens live in ONE preallocated set at MAX width, reused by all
    12 index layers (each layer's selection is consumed by that layer's attention before the next
    index layer runs — the same lifetime argument `_get_out_buf` makes).
  * COMPRESSION IS UNCONDITIONAL. Only one decode row in r actually completes a group, but a graph
    cannot branch, so every row runs the mean/norm/rope and the rows that completed nothing write to
    `QSAIndexCache.scratch_slot` — the row past `num_kv_slots // r`, which the DSV4 identity
    `physical_slot // r` can never name. Padded (dummy) rows are forced to the scratch slot too.
  * THE SPARSITY LEDGER STAYS ALIVE, on the device. `visited`/`dense` were `.item()` syncs per
    layer; under the static path they accumulate into int64 device counters that the host reads only
    when asked. A ledger that had to be switched off for capture would mean every captured number
    was taken with the one instrument that proves the path was sparse at all turned off.

The static path is used for EVERY decode batch once `init_capture` has run, not only for captured
ones. That is deliberate: it makes the eager and captured legs run the identical ops at the identical
launch policies, so an eager-vs-captured comparison measures CAPTURE and nothing else. (Had eager
kept the dynamic width, the top-k would have taken the one-CTA path on one leg and the split path on
the other — bit-identical by that kernel's own gate, but no longer the same-kernel comparison this
repo's capture-identity rule asks for.)
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
        "is_prefill", "dense_total", "static", "pos_i32", "ring_slots_long",
        "member_ring_flat", "write_locs_long", "dense_total_dev",
    )

    def __init__(self) -> None:
        # Defaults for the fields only one of the two builders sets. Explicit because __slots__
        # gives no class-level fallback: reading an unset slot is an AttributeError, and the two
        # builders (dynamic `build`, static `QSARuntime._fill_decode_plan`) populate different sets.
        self.static = False
        self.pos_i32 = None
        self.ring_slots_long = None
        self.member_ring_flat = None
        self.write_locs_long = None
        self.dense_total_dev = None

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
        #: Device-side sparsity accumulator for the DYNAMIC path. The static path has had one
        #: since capture landed (`visited_acc`); this is the same treatment here. Lazily made
        #: on first `select` so it lands on the right device without this __init__ knowing it.
        self._dyn_acc: Optional[torch.Tensor] = None
        #: Most recent step's (visited-on-device, dense-on-host), for the `last_sparsity`
        #: property. Held, not read: reading is a sync and only the gates ever ask.
        self._last_sp_num: Optional[torch.Tensor] = None
        self._last_sp_den: int = 0
        # ---- static decode plan (cudagraph capture); None until init_capture() ------------------
        self.max_graph_bs = 0
        self.static_max_blocks = 0
        self._s: dict = {}
        self._static_plan: Optional[QSAPlan] = None
        self.static_steps = 0        # PROVENANCE: decode forwards that took the static path
        self.captured_steps = 0      # ...of which ran inside a capture

    # -- cudagraph capture ---------------------------------------------------------------------
    def init_capture(self, max_bs: int, max_seq_len: int) -> None:
        """Allocate the static decode workspaces. Must run OUTSIDE capture (it allocates).

        `max_blocks` is `ceil(max_seq_len / r)` — the block window a request at the engine's own
        context ceiling would have. It is a build-time constant, which is the whole point: every grid
        downstream (the scorer's key tiles, the split top-k's `num_splits`) is derived from a tensor
        SHAPE, so it is identical at capture and at replay. Deriving it from `row_ends.max()` instead
        is exactly the width-inference bug class this repo already shipped once (61d96cf/0972e387).
        """
        if max_bs <= 0 or self._s:
            return
        p, dev = self.profile, self.page_table.device
        r = p.compress_ratio
        mb = max(1, (max_seq_len + r - 1) // r)
        self.max_graph_bs = int(max_bs)
        self.static_max_blocks = mb
        w, k = p.index_width, p.block_topk
        z = lambda *shape, dt=torch.int32: torch.zeros(shape, dtype=dt, device=dev)  # noqa: E731
        self._s = {
            # plan
            "row_seq": torch.arange(max_bs, dtype=torch.int32, device=dev),
            "row_starts": z(max_bs),
            "row_ends": z(max_bs),
            "pos": z(max_bs, dt=torch.int64),
            "pos_i32": z(max_bs),
            "seq_len": z(max_bs),
            "table": z(max_bs, dt=torch.int64),
            "ring": z(max_bs),
            "ring_long": z(max_bs, dt=torch.int64),
            "cpt": z(max_bs, mb),
            "cpt_idx": z(max_bs, mb, dt=torch.int64),
            "cols": (torch.arange(mb, dtype=torch.int64, device=dev) * r),
            "off": torch.arange(r, dtype=torch.int64, device=dev),
            "member_ring": z(max_bs, r, dt=torch.int64),
            "write": z(max_bs),
            "write_long": z(max_bs, dt=torch.int64),
            "scratch": torch.full((max_bs,), self.cache.scratch_slot, dtype=torch.int64, device=dev),
            "dense": z(1, dt=torch.int64),
            # 1 for a REAL row, 0 for a dummy padded one — the sparsity ledger must not count the
            # padding, or a bs=1 request replaying a bs=2 graph would report 2x the visited slots.
            "real": z(max_bs, dt=torch.int64),
            # selection workspace, shared by all index layers (see the module docstring)
            "logits": torch.zeros((max_bs, mb), dtype=torch.float32, device=dev),
            "blocks": z(max_bs, k),
            "tokens": z(max_bs, w),
            "slots": z(max_bs, w),
            "lens": z(max_bs),
            "flat": z(max_bs, w, dt=torch.int64),
            # device-side sparsity ledger (no per-layer .item())
            "visited_acc": z(1, dt=torch.int64),
            "dense_acc": z(1, dt=torch.int64),
            # pinned host staging for the ONE H2D per forward
            "host": torch.zeros((3, max_bs), dtype=torch.int64, pin_memory=True),
            "host_dev": z(3, max_bs, dt=torch.int64),
        }
        logger.info_rank0(
            f"QSA graph capture ARMED: max_bs={max_bs} max_blocks={mb} (max_seq_len={max_seq_len}, "
            f"r={r}) logits={max_bs * mb * 4 / 2**20:.1f} MiB, scratch_slot={self.cache.scratch_slot}"
        )

    @property
    def capture_ready(self) -> bool:
        return bool(self._s)

    def _use_static(self, batch: "Batch") -> bool:
        return (
            bool(self._s)
            and batch.is_decode
            and len(batch.padded_reqs) <= self.max_graph_bs
            and all(req.extend_len == 1 for req in batch.padded_reqs)
        )

    def _fill_decode_plan(self, batch: "Batch") -> QSAPlan:
        """Refresh the static decode buffers IN PLACE from `batch.padded_reqs`. Eager, never inside a
        capture. One pinned H2D of (table_idx, position, seq_len); everything else is device math."""
        s = self._s
        reqs = batch.padded_reqs
        n_real, P = batch.size, len(reqs)
        r = self.profile.compress_ratio
        host = s["host"]
        for i, req in enumerate(reqs):
            host[0, i] = req.table_idx
            host[1, i] = req.cached_len          # decode: extend_len == 1, so pos == cached_len
            host[2, i] = req.device_len
        s["host_dev"][:, :P].copy_(host[:, :P], non_blocking=True)
        tbl, pos, slen = s["host_dev"][0, :P], s["host_dev"][1, :P], s["host_dev"][2, :P]

        s["real"][:P].zero_()
        s["real"][:n_real] = 1
        s["pos"][:P].copy_(pos)
        s["pos_i32"][:P].copy_(pos)
        s["seq_len"][:P].copy_(slen)
        s["table"][:P].copy_(tbl)
        s["row_ends"][:P].copy_(torch.div(pos + 1, r, rounding_mode="floor"))
        s["ring"][:P].copy_(tbl * r + torch.remainder(pos, r))
        s["ring_long"][:P].copy_(s["ring"][:P])

        stride = self.page_table.shape[1]
        flat_pt = self.page_table.reshape(-1)
        # compressed page table: block g of this row -> compressed slot, via `physical_slot // r` on
        # the group's FIRST token. Refreshed at FULL static width; columns past the row's own window
        # hold stale slots and are never read (the scorer bounds j by row_ends).
        torch.add(tbl[:, None] * stride, s["cols"][None, :], out=s["cpt_idx"][:P])
        s["cpt"][:P].copy_(
            torch.div(
                flat_pt.index_select(0, s["cpt_idx"][:P].reshape(-1)).reshape(P, -1),
                r, rounding_mode="floor",
            )
        )
        # compression plan: EVERY row runs it; only the rows that actually complete a group (and are
        # real, not dummy padding) write to a live compressed slot. The rest write to `scratch_slot`.
        gfp = pos - (r - 1)                                   # group's first member position
        boundary = torch.remainder(pos + 1, r) == 0
        if n_real < P:
            boundary[n_real:] = False
        first_slot = flat_pt.index_select(0, tbl * stride + gfp.clamp(min=0))
        s["write_long"][:P].copy_(
            torch.where(boundary, torch.div(first_slot, r, rounding_mode="floor").to(torch.int64),
                        s["scratch"][:P])
        )
        s["write"][:P].copy_(s["write_long"][:P])
        s["member_ring"][:P].copy_(
            (tbl * r)[:, None] + torch.remainder(gfp[:, None] + s["off"][None, :], r)
        )
        # dense causal count, on the device (was an .item()): min(pos+1, seq_len) over REAL rows.
        d = torch.minimum(pos + 1, slen)
        if n_real < P:
            d = d[:n_real]
        torch.sum(d, dim=0, keepdim=True, out=s["dense"])

        plan = QSAPlan()
        plan.profile, plan.static, plan.is_prefill = self.profile, True, False
        plan.rows, plan.bs, plan.page_stride = P, P, stride
        plan.row_seq = s["row_seq"][:P]
        plan.logical_pos = plan.rope_pos = s["pos"][:P]
        plan.pos_i32 = s["pos_i32"][:P]
        plan.seq_len_row = s["seq_len"][:P]
        plan.row_table_idx = s["table"][:P]
        plan.row_starts = s["row_starts"][:P]
        plan.row_ends = s["row_ends"][:P]
        plan.max_blocks = self.static_max_blocks
        plan.comp_page_table = s["cpt"][:P]
        plan.ring_slots = s["ring"][:P]
        plan.ring_slots_long = s["ring_long"][:P]
        plan.write_locs = s["write"][:P]
        plan.write_locs_long = s["write_long"][:P]
        plan.member_ring = s["member_ring"][:P]
        plan.member_ring_flat = s["member_ring"][:P].reshape(-1)
        plan.member_rows = None
        plan.dense_total = -1
        plan.dense_total_dev = s["dense"]
        self._static_plan = plan
        return plan

    def prepare_for_replay(self, batch: "Batch") -> None:
        """`GraphRunner.replay`'s pre-hook (via `BaseLLMModel.prepare_for_replay`). Runs EAGER,
        outside the graph, and writes the exact tensors the captured kernels read through their baked
        pointers. The captured `model.forward()` never re-enters `prepare()`."""
        assert self._s, "QSARuntime.prepare_for_replay before init_capture"
        assert self._use_static(batch), (
            f"QSA: a decode batch reached the graph replay path that the static plan cannot serve "
            f"(padded_size={len(batch.padded_reqs)} > max_graph_bs={self.max_graph_bs}, or a "
            f"multi-query row). can_use_cuda_graph should have routed this to eager."
        )
        self.plan = self._fill_decode_plan(batch)
        self._last = {}
        self.static_steps += 1

    # -- per-forward -------------------------------------------------------------------------
    def prepare(self, batch: "Batch") -> None:
        if torch.cuda.is_current_stream_capturing():
            # Inside a capture the plan CONTENTS are irrelevant (only the op stream is recorded) and
            # the fill is exactly the host work a capture forbids — the warmup forward that precedes
            # every capture already filled the same buffers from the same batch. So: reuse, don't
            # refill. Anything that is not a static-plan decode is a genuine refusal.
            assert self._static_plan is not None and self._use_static(batch), (
                "qwen4_exp QSA: only a DECODE batch with a static plan is capturable "
                f"(is_decode={batch.is_decode}, padded={len(batch.padded_reqs)}, "
                f"max_graph_bs={self.max_graph_bs}). Prefill is eager everywhere in this engine."
            )
            # AND IT HAS TO BE **THIS** BUCKET'S PLAN. `_static_plan` is a single mutable slot that
            # every `_fill_decode_plan` overwrites, and reusing it here is sound only because
            # `GraphRunner`'s capture loop runs an EAGER WARMUP at the same bs immediately before
            # each capture. That ordering belongs to GraphRunner, not to this class, and nothing
            # else enforces it — so without this line a reordered or added capture step would bake
            # the PREVIOUS bucket's row count into the graph: a different `num_splits` for the split
            # top-k, a different scorer grid, a different row count in every static view, and no
            # error at capture OR at replay. Silently-wrong attention, which is the failure class
            # this whole file is written against. Assert the identity instead of documenting the
            # ordering, because an ordering nobody checks is an ordering that eventually changes.
            assert self._static_plan.rows == len(batch.padded_reqs), (
                "qwen4_exp QSA: the static plan on hand was built for a DIFFERENT batch width "
                f"(plan.rows={self._static_plan.rows}, this capture's padded batch="
                f"{len(batch.padded_reqs)}). Every capture must be immediately preceded by an eager "
                "warmup at its own bs — that warmup is what refreshes the plan. Capturing here "
                "would bake the other bucket's row count and would never raise."
            )
            self.plan = self._static_plan
            self._last = {}
            self.captured_steps += 1
            return
        if self._use_static(batch):
            self.plan = self._fill_decode_plan(batch)
            self._last = {}
            self.static_steps += 1
            return
        self.plan = QSAPlan.build(self.profile, self.cache, batch, self.page_table)
        self._last = {}

    # -- the sparsity ledger, which survives capture ---------------------------------------------
    def sparsity_totals(self) -> "tuple[int, int]":
        """(visited, dense) over the whole run — the dynamic path's host counters PLUS the static
        path's device counters. Syncs once, when asked; never per layer."""
        v, d = self.total_visited, self.total_dense
        if self._dyn_acc is not None:
            v += int(self._dyn_acc.item())
        if self._s:
            v += int(self._s["visited_acc"].item())
            d += int(self._s["dense_acc"].item())
        return v, d

    @property
    def last_sparsity(self) -> "Optional[float]":
        """visited/dense for the most recent `select`. SYNCS ON READ — never per step. The serve
        never touches it; the gates do, and one sync when a test asks is free."""
        if self._last_sp_num is None or not self._last_sp_den:
            return None
        return int(self._last_sp_num.item()) / self._last_sp_den

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
        if plan.static:
            return self._select_static(index_layer, q, token_k, k_layernorm, plan)
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
        # THE LEDGER, ON THE DEVICE. This was `int(lens.to(torch.int64).sum().item())` — a host
        # sync per INDEX LAYER, i.e. 12 pipeline drains per decode token on this checkpoint, for a
        # number nothing in the forward pass reads. py-spy put it at 36% of the scheduler rank's
        # samples and the serving path measured 131 ms/token against the harness's 60.5
        # [PERF-2026-09-06]. The static path stopped doing this when capture landed; the dynamic
        # path — the one a GRAPH_BS=0 serve actually runs — never got the same fix.
        # `sparsity_totals()` and `last_sparsity` still sync, ONCE, when something asks.
        vis = lens.to(torch.int64).sum()
        if self._dyn_acc is None:
            self._dyn_acc = torch.zeros((), dtype=torch.int64, device=vis.device)
        self._dyn_acc += vis
        self.total_dense += plan.dense_total
        self._last_sp_num, self._last_sp_den = vis, plan.dense_total
        # visited=-1 for the same reason the static path returns -1: it is not knowable without a
        # sync, and no consumer reads it — the ledger is read through sparsity_totals().
        return QSASelection(
            slots=sel_slots, lens=lens, visited=-1, dense=plan.dense_total
        )

    def _select_static(
        self, index_layer: int, q: torch.Tensor, token_k: torch.Tensor, k_layernorm,
        plan: QSAPlan,
    ) -> QSASelection:
        """The SAME four stages, on the preallocated buffers, with no host read and no branch.

        Every difference from the dynamic body above is one of exactly three things, and none of
        them changes a value:
          * the workspaces are the preallocated ones (allocation is not a value);
          * the logits width is the STATIC `max_blocks` instead of this step's `row_ends.max()` —
            the scorer bounds `j` by `row_ends` and writes -inf elsewhere, and the top-k reads only
            `[row_starts, row_starts+row_ends)`, so the extra columns are never read;
          * the compression is unconditional, with the non-boundary and padded rows aimed at
            `scratch_slot` (a row the DSV4 identity cannot name) instead of skipped.
        """
        s, r, cache = self._s, self.profile.compress_ratio, self.cache
        P, d = plan.rows, self.profile.head_dim

        # 1. raw key -> pending ring.
        pend = cache.pending_keys(index_layer)
        pend.index_copy_(0, plan.ring_slots_long, token_k.to(pend.dtype))
        cache.pending_pos.index_copy_(0, plan.ring_slots_long, plan.pos_i32)

        # 2. compress — UNCONDITIONAL; see the docstring. fp32 mean -> Gemma norm -> rope at the
        #    group's OLDEST member. Order is load-bearing and invisible when wrong.
        members = pend.index_select(0, plan.member_ring_flat)
        first_pos = cache.pending_pos.index_select(0, plan.member_ring[:, 0]).to(torch.int64)
        pooled = members.reshape(-1, r, d).float().mean(dim=1)
        normed = k_layernorm.forward(pooled.to(token_k.dtype))
        roped = self.rotary.forward_one(first_pos, normed)
        comp = cache.compressed_keys(index_layer)
        comp.reshape(comp.shape[0], -1).index_copy_(0, plan.write_locs_long, roped.to(comp.dtype))

        # 3. score + 4a. top-k, at the STATIC width (no row tiling: P <= max_graph_bs).
        logits = s["logits"][:P]
        blocks = s["blocks"][:P]
        ops.score_paged(
            q.contiguous(), comp, plan.comp_page_table,
            plan.row_starts, plan.row_ends, plan.row_seq, logits, self.scale,
        )
        ops.topk(logits, plan.row_starts, plan.row_ends, blocks)

        # 4b. blocks -> token positions -> PHYSICAL kv slots.
        sel_tokens = s["tokens"][:P]
        ops.expand(
            blocks, plan.pos_i32, plan.seq_len_row, sel_tokens, r, self.profile.budget,
        )
        lens = s["lens"][:P]
        torch.sum((sel_tokens >= 0).to(torch.int32), dim=1, out=lens)
        torch.add(
            plan.row_table_idx[:, None] * plan.page_stride,
            sel_tokens.clamp(min=0).to(torch.int64),
            out=s["flat"][:P],
        )
        sel_slots = s["slots"][:P]
        sel_slots.copy_(
            self.page_table.reshape(-1)
            .index_select(0, s["flat"][:P].reshape(-1))
            .reshape(P, -1)
        )
        # The ledger, on the device. `visited` counts REAL rows only — a padded dummy row's
        # selection is discarded by the sampler and counting it would inflate the sparsity ratio.
        s["visited_acc"] += (lens.to(torch.int64) * s["real"][:P]).sum()
        s["dense_acc"] += plan.dense_total_dev
        return QSASelection(slots=sel_slots, lens=lens, visited=-1, dense=-1)


def _logits_row_tile(rows: int, cols: int) -> int:
    """Row tile that keeps the fp32 [rows, blocks] scoring workspace under the budget."""
    if rows <= 0 or cols <= 0:
        return max(rows, 1)
    per_row = cols * 4
    return max(1, min(rows, max(1, _LOGITS_BUDGET_BYTES // per_row)))


__all__ = ["QSAPlan", "QSARuntime", "QSASelection", "_ATTN_ROW_TILE"]
