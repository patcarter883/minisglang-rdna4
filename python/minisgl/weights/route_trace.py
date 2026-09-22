"""Per-(layer, expert, step) MoE routing capture. MEASURE-ONLY, default OFF.

WHAT QUESTION THIS ANSWERS. Our P2-prime scoping measured a LINEAR expert-miss curve (cliff_index
0.090/0.094 against 0.100 pure-linear) and concluded per-expert placement is worth 1.013x at the
reachable operating point — i.e. routing is not skewed and an LRU hot-expert VRAM cache cannot beat
static placement. A third party reports the opposite on comparable hardware. A linear miss curve and
a 27-30 tok/s LRU result cannot both be true. This module records the ground truth: which expert
slabs the grouped GEMM actually dereferenced, per layer, per forward step, in order.

WHERE THE ROUTE COMES FROM, AND WHY IT MUST. `quant.kernels.w4a8_moe` calls `_route_align`
(`moe_hip.moe_route_align`) which does softmax + top-k + renormalize + moe_align inside ONE kernel.
A torch `softmax().topk()` is NOT tie-identical to it: `layers/moe.py:1185-1200` records the measured
disagreement on THIS checkpoint (layer 0, 8-token prefill, experts 324 and 366 both at logit
-5.09375 straddling k=10; the kernel kept 324, torch.topk kept 366; 2 of 8 MoE calls in one forward
diverged). bf16 gate logits over 512 experts make ties ordinary. So the trace is taken from the
kernel's OWN topk_ids at the point the GEMM consumes it, never re-derived.

WHY THERE IS A DEVICE RING. The forbidden implementation is a per-layer `.item()`/`.tolist()` on the
decode path. This repo has a measured price for exactly that: commit ffa1d8c6, QSA's dynamic select
ended with `int(lens.to(torch.int64).sum().item())` — one host sync per index layer. py-spy put that
single line at 36% of scheduler-rank samples, and deleting it took the serve from 131 to 85 ms/token.
Extrapolated across 48 MoE layers that is roughly +180 ms on a 60 ms step, ~4x slower. The decode
path here does one device slice-assign per layer and syncs nothing; the host sees the ids once per
`drain_every` steps, in one D2H.

PREFILL IS DIFFERENT ON PURPOSE. A prefill call's expert union is variable-length (up to E), so it
falls out of the fixed-width ring and takes a single blocking `.tolist()`. A prefill chunk is ~35 ms
of work and there are ~28 of them per 28k prompt; one sync there is noise. Prefill records are
REQUIRED for the pollution analysis (a prefill touches most of the 512 experts and would flush a
naive LRU), so they are not optional.

CAPTURE-SAFE, VIA A STAGE BUFFER AND A HARVEST AT THE STEP BOUNDARY. This module used to be
capture-UNSAFE by design, and said so: every write was guarded on
`torch.cuda.is_current_stream_capturing()`, so an accidental capture degraded to a GAP in the trace
rather than an illegal op. That gap stopped being acceptable the moment this ring became the expert
cache's only input (`set_observer`) instead of just a measurement fixture. Under capture the ring
recorded nothing, the cache observed nothing, and it went INERT while still holding its entire
budget (2.5 GiB/rank on the shipped arm) — i.e. re-freezing exactly the stall that was just fixed at
real cost (h 0.3256 -> 0.4067 measured, install rate 0.39 -> ~5/tick). Capture itself is worth a
measured 3.45-3.66 ms of a ~60 ms decode step (`tools/serve.sh`, `[QSA-CAP-2026-09-06]`), so
"capture or the cache, pick one" was not a trade worth keeping.

Two things made the ring un-capturable. Both are fixed here:

  1. THE RING INDEX WAS A HOST INT. `record` wrote `ids_ring[self.slot, lid, :n]` and `self.slot` is
     host Python, computed in `begin_forward`. Under capture that Python runs ONCE, so the slot is
     baked as a constant and every replay writes the SAME slot; `self.meta[slot]` (a host dict) is
     never updated at replay either. FIX: `record` writes a per-step STAGE buffer at
     `stage[lid, :n]` — constant indices, no slot, capture-safe — and `harvest()` copies stage into
     `ids_ring[slot]` at the step boundary: in `begin_forward`, which is outside any capture region
     and the one point every forward call site passes through. `lid` never needed fixing — each
     layer's wrapper runs at capture with its own lid, so the graph holds one correctly-addressed
     write per layer. The harvest is one [num_layers, ring_width] int32 D2D copy, ~4 KiB/step,
     noise against a ~60 ms step. Device-side indexing of the ring (a `slot` tensor the scheduler
     bumps) was deliberately NOT taken: it buys nothing here and the staging copy is far easier to
     prove correct.
  2. A CAPTURED BUCKET'S PADDED ROWS CARRY GARBAGE ROUTING. Capture is BUCKETED: a bs=2 graph
     replayed for a 1-request step pushes a padded row through routing too, and whichever experts
     that row lands on were referenced by NO request. Feeding them to the policy would admit slabs
     nothing reads and evict ones something does. `record` cannot mask them — at capture time it
     knows only the bucket width — so `harvest` does it on the host, where the step's REAL row count
     is known (`begin_forward(num_rows=...)`, from `Scheduler._step_boundary`).
     `topk_ids.reshape(-1)` is row-major, so rows 0..M-1 are exactly the first M*top_k entries.

The HOST path (prefill, and any forward too wide for the ring) is still capture-guarded and must
stay that way: it ends in a blocking `.tolist()`, which is illegal under capture. Nothing captured
takes it — `engine/graph.py` captures decode and verify only, and `record` routes both to the ring.

`tools/moe_route_stats.sh:17-20` remains the precedent that a routing statistic is legitimately
taken from the forward at all: routing is a deterministic function of the hidden states, identical
captured or not.

TESTING THIS WITHOUT A GPU. Everything above is host logic over tensors; `device=torch.device("cpu")`
constructs a working tracer (the pinned D2H staging buffer is only pinned on a cuda device, see
__init__), and the capture branch is reachable two ways: call `_record_captured` directly, or
monkeypatch `torch.cuda.is_current_stream_capturing`. `tests/route_trace_verify_test.py` is the
existing shape of such a test.
"""

from __future__ import annotations

import os
import struct
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from minisgl.kvcache._envutil import env_int

# ---- format ---------------------------------------------------------------------------------
MAGIC = b"MSGLRT01"
VERSION = 1
HEADER_FMT = "<8sIIIIIIQIIQ8x"      # 64 bytes
RECORD_FMT = "<IIHBBHH"             # 16 bytes, then u16[num_ids]
NUM_RECORDS_OFFSET = 32             # byte offset of the u64 num_records field
assert struct.calcsize(HEADER_FMT) == 64
assert struct.calcsize(RECORD_FMT) == 16

KIND_PREFILL, KIND_DECODE, KIND_OTHER = 0, 1, 2
# A speculative VERIFY forward. Distinct from KIND_DECODE in the trace FILE (its M is
# bs*(K+1), and analyses that assume one token per record would silently mis-weight it) but
# treated as a decode by the OBSERVER: a verify is the decode band, and its routed-expert
# union is exactly the working set the cache is sized for.
KIND_VERIFY = 3

_FNV_OFFSET = 0xCBF29CE484222325
_FNV_PRIME = 0x100000001B3
_MASK64 = (1 << 64) - 1


def fnv1a64(s: str) -> int:
    h = _FNV_OFFSET
    for b in s.encode("utf-8"):
        h = ((h ^ b) * _FNV_PRIME) & _MASK64
    return h


# ---- module state ---------------------------------------------------------------------------
# The scheduler loop is single-threaded, which is what makes a module global a legal channel from
# the MoELayer wrapper down to w4a8_moe. Stated, not assumed: `_LOCK` guards only the file, which
# the drain touches, and an assert in `begin_forward` catches a second driving thread.
_TRACER: "Optional[RouteTracer]" = None
_CUR_LID: Optional[int] = None
_CUR_CHUNK: Dict[int, int] = {}     # lid -> next chunk_idx within THIS forward (C7)


def tracer() -> "Optional[RouteTracer]":
    return _TRACER


def enabled() -> bool:
    return _TRACER is not None


def begin_forward(is_prefill: bool, req_uid: int, *, is_verify: bool = False,
                  num_rows: "Optional[int]" = None) -> None:
    """Step boundary. Called from Scheduler._forward for EVERY loop (there are eight), and from the
    four speculative VERIFY sites. `is_verify` is not cosmetic: without it `_cur_kind` keeps the
    previous forward's value -- a PREFILL on a spec serve -- and every verify record early-returns
    before reaching the ring, which is why the expert cache saw nothing under spec.

    `num_rows` is the REAL query-row count of the forward about to run (padding excluded). It is what
    lets `harvest` mask a captured bucket's padded rows; None means "unknown", which is correct-but-
    unmasked and is counted in `rows_unknown` rather than assumed harmless."""
    t = _TRACER
    if t is not None:
        t.begin_forward(is_prefill, req_uid, is_verify=is_verify, num_rows=num_rows)


def close() -> None:
    global _TRACER
    t = _TRACER
    if t is not None:
        _TRACER = None
        t.close()


# ---- the tracer -----------------------------------------------------------------------------
class RouteTraceError(RuntimeError):
    """The tracer cannot honour its own invariants. Never downgraded to a warning: a measurement
    fixture that silently degrades is how an earlier ghost-oracle fixture ended up with one
    root-owned, months-stale file that nobody noticed."""


class RouteTracer:
    def __init__(
        self,
        out_dir: str,
        *,
        model_slug: str,
        num_layers: int,
        num_experts: int,
        top_k: int,
        tp_rank: int,
        dp_rank: int,
        expert_bytes: int,
        ring_steps: int,
        # How many FORWARD ROWS one ring entry can hold: the WIDEST decode/verify the engine can
        # produce. A plain decode carries one row per running request, a captured decode carries its
        # BUCKET's padded rows, a speculative verify carries bs*(K+1) — `engine._route_trace_ring_rows`
        # takes the max of the three. The default of 1 is for a direct caller/test only; it used to be
        # the engine's non-spec value and that starved the observer on every concurrent decode step.
        ring_rows: int = 1,
        drain_every: int,
        max_steps: int,
        record_prefill: bool,
        blockmap_checks: int,
        device: torch.device,
    ) -> None:
        # `out_dir=None` is OBSERVE-ONLY: ring + drain + observer, no file. That is the mode the
        # expert cache runs in production — it needs "which experts did layer L read", which this
        # ring already collects with no extra D2H, but it must NOT write a 35 MB trace per rank on
        # every serve. Capture mode (a real dir) is unchanged and still writes the fixture.
        if out_dir is not None:
            if not os.path.isdir(out_dir):
                raise RouteTraceError(
                    f"MINISGL_MOE_ROUTE_TRACE={out_dir!r} is not a directory. The trace is a "
                    f"durable measurement fixture and must land on a path the operator pre-created "
                    f"and mounted (`install -d -o 1000 -g 1000 <dir>`), never in a container layer "
                    f"or a worktree that gets removed. Refusing to boot rather than degrading "
                    f"silently."
                )
            if not os.access(out_dir, os.W_OK):
                raise RouteTraceError(f"route trace dir {out_dir!r} is not writable by this uid")
        if drain_every > ring_steps:
            raise RouteTraceError(
                f"drain_every ({drain_every}) > ring_steps ({ring_steps}): the ring would wrap "
                f"before it is read and steps would be lost with no error."
            )
        self.num_layers = int(num_layers)
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.ring_steps = int(ring_steps)
        self.drain_every = int(drain_every)
        self.max_steps = int(max_steps)
        self.record_prefill = bool(record_prefill)
        self.blockmap_checks = int(blockmap_checks)
        self.device = device
        self.ops: Dict[int, Any] = {}
        self.disarmed = False
        self.step_id = -1
        self.slot = -1
        self.n_since_drain = 0
        self.records_written = 0
        self.blockmap_seen = 0
        self.blockmap_violations = 0
        self._thread = threading.get_ident()
        self._lock = threading.Lock()
        # Per-step host state for the step CURRENTLY in flight. Initialised here rather than first
        # assigned in `begin_forward` so `harvest` and a CPU test can read them before any step has
        # run, and so `record` can never see a half-built object.
        self._cur_uid = 0
        self._cur_kind = KIND_OTHER
        self._cur_rows: Optional[int] = None    # REAL rows of this forward; None = unknown
        self._harvested = True                  # nothing staged yet -> nothing to harvest

        # RING WIDTH. One decode row needs top_k slots; a speculative VERIFY carries M = bs*(K+1)
        # rows and its union is what the cache must see, so the ring is widened to hold `ring_rows`
        # rows' worth. Rows beyond that are dropped with a counter rather than silently truncating
        # the union (see `record`), because an under-reported union makes the cache look BETTER than
        # it is -- the same failure mode the M==1 gate was protecting against.
        self.ring_rows = max(1, int(ring_rows))
        self.ring_width = self.top_k * self.ring_rows
        # DEVICE ring: ids only. -1 = "this (step, layer) had no ring record".
        self.ids_ring = torch.full(
            (self.ring_steps, self.num_layers, self.ring_width), -1, dtype=torch.int32, device=device
        )
        # PER-STEP STAGE. This, not the ring, is what `record` writes, and it is the whole reason a
        # captured decode can be traced: the write is `stage[lid, :n]`, two CONSTANT indices, so the
        # graph bakes a correct address. The ring's step index cannot appear inside a graph (it is
        # host Python; a replay would rewrite one frozen slot forever) — `harvest` moves stage into
        # `ids_ring[slot]` at the step boundary instead. Allocated ONCE, before any capture, so the
        # address the graph bakes stays valid for the life of the serve (the same argument
        # expert_cache.py's "GRAPH CAPTURE" note makes for `slot_of`).
        self.stage = torch.full(
            (self.num_layers, self.ring_width), -1, dtype=torch.int32, device=device
        )
        # Pinned host staging for the ONE D2H per drain (the qsa runtime.py:331-333 pattern).
        # pin_memory only on a cuda device: pinning is a hipHostMalloc and RAISES with no GPU, which
        # would make this whole class unconstructable in a CPU test for no benefit (a cpu->cpu
        # `copy_` neither needs nor uses the pinning).
        self.ids_host = torch.empty(
            (self.ring_steps, self.num_layers, self.ring_width), dtype=torch.int32,
            pin_memory=(torch.device(device).type == "cuda"),
        )
        # Verify rows that did not fit `ring_rows`. Reported at close; a nonzero value means the
        # cache was fed a PARTIAL union and any hit rate measured against it is optimistic.
        self.rows_dropped = 0
        # Steps harvested with `num_rows=None`, i.e. with no real-row count to mask a captured
        # bucket's padding against. Reported at close because it is the one way padded routing can
        # still reach the policy: a caller that forgot to plumb the count.
        self.rows_unknown = 0
        # Steps where `record` saw a row count that disagreed with the one `begin_forward` was told.
        # One int compare per layer per step buys the difference between "the mask is wrong" showing
        # up as a counter and showing up as a quietly optimistic hit rate.
        self.rows_mismatch = 0
        # MoE calls that reached `record` under capture with no layer id in scope (an unwrapped MoE
        # being captured — an MTP draft head is the plausible one). Not fatal: a missing record
        # degrades the cache, it cannot corrupt it. Loud once, then counted.
        self.capture_unwrapped = 0
        # HOST meta, ONE TUPLE PER STEP: (step, uid, kind, rows). Every ring record of a given step
        # shares all four — `record` used to store a per-lid dict entry, 48 dict setitems per step on
        # the decode path, to hold 48 copies of the same tuple plus a lid-keyed "did this layer
        # record" bit. `harvest` now owns it, one store per step, and `drain` recovers the per-layer
        # bit from the ring row itself (all -1 => that layer never recorded). None = no ring record
        # for this slot.
        self.step_meta: List[Optional[Tuple[int, int, int, int]]] = [None] * self.ring_steps
        # Prefill/large-M records fall out of the ring entirely (variable num_ids).
        self.oversize: List[Tuple[int, int, int, int, int, int, List[int]]] = []

        slug = "".join(c if c.isalnum() or c in "-._" else "_" for c in (model_slug or "unknown"))
        if out_dir is None:
            self.path = None
            self._fh = None
            return
        self.path = os.path.join(out_dir, f"route.{slug}.rank{tp_rank}.bin")
        self._fh = open(self.path, "wb")
        self._fh.write(
            struct.pack(
                HEADER_FMT,
                MAGIC,
                VERSION,
                self.num_layers,
                self.num_experts,
                self.top_k,
                int(tp_rank),
                int(dp_rank),
                0,                      # num_records, patched on every drain and at close
                int(expert_bytes),
                1,                      # flags: bit0 = ids are the deduped sorted union (always 1)
                fnv1a64(model_slug or "unknown"),
            )
        )
        self._fh.flush()

    # -- install ------------------------------------------------------------------------------
    def install_hooks(self, model: Any) -> None:
        """Shadow MoELayer.forward with an instance attribute, outermost.

        AFTER ExpertStreamTier.install_hooks (engine.py:376 -> bake.py:674), so this wrapper is the
        outer one and `_CUR_LID` is set before the tier's own staging runs.
        """
        from minisgl.weights.moe_interpose import discover_moe_layers
        from minisgl.weights.plan import is_mtp_path
        from minisgl.weights.stream_tier import layer_index_of_path

        found = [(p, o) for p, o in discover_moe_layers(model) if not is_mtp_path(p)]
        for path, op in found:
            lid = layer_index_of_path(path)      # raises on an unparseable path (C2)
            if lid in self.ops:
                raise RouteTraceError(f"two MoE layers claim layer index {lid}: {path!r}")
            inner = op.forward

            def traced_forward(hidden_states, router_logits=None, *args, _lid=lid, _inner=inner,
                               **kwargs):
                global _CUR_LID
                _CUR_LID = _lid
                # NO per-entry reset. `_CUR_CHUNK` is per-(STEP, layer) and is cleared once at the
                # step boundary. Resetting here was WRONG and self-defeating: the row split lives
                # ABOVE this function -- qwen3_5_moe.py:172 `rowchunked_ar_span(...)` ->
                # tp_overlap.py:405 `produce(x[lo:hi], ...)` -> qwen3_5_moe.py:113 `_fused_partial`
                # -> THIS wrapper, once PER CHUNK. So a >=256-row prefill enters here twice and a
                # per-entry reset stamps chunk_idx=0 on both, which is the exact duplicate key C7
                # exists to prevent. With MAX_PREFILL_LENGTH=1024 every prefill chunk splits 2x512,
                # so this fired on EVERY prefill layer, and a lenient reader would have silently
                # dropped ~half of each layer's prefill expert union -- the pollution term that
                # decides whether LRU survives prefill.
                try:
                    return _inner(hidden_states, router_logits, *args, **kwargs)
                finally:
                    _CUR_LID = None

            op.forward = traced_forward
            self.ops[lid] = op

        # Assert, do not assume. qwen4exp.py:131 ModelConfig.from_hf REFUSES --spec-algorithm mtp for
        # this architecture and serve.sh sets spec_default=none, so the MTP filter should remove
        # nothing here — but a count of 49 means an MTP MoELayer slipped through and every layer id
        # after it would be a lie.
        if len(self.ops) != self.num_layers:
            raise RouteTraceError(
                f"route trace wrapped {len(self.ops)} non-MTP MoE layers, expected "
                f"{self.num_layers}: {sorted(self.ops)}"
            )
        if sorted(self.ops) != list(range(self.num_layers)):
            raise RouteTraceError(f"non-contiguous MoE layer ids: {sorted(self.ops)}")

    # -- step boundary ------------------------------------------------------------------------
    def begin_forward(self, is_prefill: bool, req_uid: int, *, is_verify: bool = False,
                      num_rows: "Optional[int]" = None) -> None:
        if self.disarmed:
            return
        assert threading.get_ident() == self._thread, (
            "route trace: a second thread drove a forward. The module globals that carry the layer "
            "id are only legal because the scheduler loop is single-threaded."
        )
        # HARVEST FIRST, then drain. The step that just finished left its ids in `stage`; they have to
        # reach `ids_ring[self.slot]` BEFORE the drain reads the ring and before this step's records
        # overwrite the stage. Both run here for the same reason: `begin_forward` is outside the
        # forward, outside any capture region, and the only point every forward call site passes
        # through. _hp_tick is NOT such a point (it runs only under MINISGL_HOSTPROF) — see C5.
        self.harvest()
        if self.n_since_drain >= self.drain_every:
            self.drain()
        self.step_id += 1
        if self.step_id >= self.max_steps:
            self.disarmed = True
            self.close()
            return
        self.slot = self.step_id % self.ring_steps
        self.step_meta[self.slot] = None
        self._cur_uid = int(req_uid) & 0xFFFFFFFF
        self._cur_kind = (KIND_PREFILL if is_prefill
                          else KIND_VERIFY if is_verify else KIND_DECODE)
        # The REAL row count of the forward about to run. `harvest` masks everything past it, which
        # is how a captured bucket's padded rows are kept out of the policy — see (2) in the module
        # docstring. None is "unknown": harvested unmasked and counted, never assumed to be 1.
        self._cur_rows = None if num_rows is None else max(0, int(num_rows))
        self._harvested = False
        self.n_since_drain += 1
        _CUR_CHUNK.clear()

    def harvest(self) -> None:
        """Move the finished step's staged ids into its ring slot, masking padded rows. IDEMPOTENT.

        WHY THIS EXISTS: see (1) and (2) in the module docstring. `record` cannot address the ring
        under capture (the slot is host Python) and cannot know the real row count (it sees the
        bucket width), so both jobs land here — host side, at the step boundary, outside capture.

        THREE DEVICE OPS PER STEP, and the eager path got CHEAPER in exchange: `record` no longer
        clears the stale tail of its row (the stage is reset to -1 here, once, for all layers), so
        the per-layer cost went from two device ops to one. Net on a 48-layer decode: 96 -> 51.

        Idempotent via `_harvested` because `drain()` calls it too — a drain that ran without a
        following `begin_forward` (every direct `drain()` in a test, and `close()`) must still see
        the last step's ids, and a second harvest of the same step must not re-copy a stage that has
        already been reset."""
        if self.disarmed or self.slot < 0 or self._harvested:
            return
        self._harvested = True
        if self._cur_kind not in (KIND_DECODE, KIND_VERIFY):
            # A PREFILL step never writes the stage (every prefill record takes the host path), so
            # there is nothing to move and nothing to reset -- the stage is all -1 by induction, since
            # only a decode/verify step writes it and that step's own harvest reset it. Returning
            # early keeps a 28-chunk prompt from issuing a pointless copy+fill per chunk.
            return
        # THE MASK, and the one invariant it rests on: REAL ROWS COME FIRST. `topk_ids.reshape(-1)`
        # is row-major, so rows 0..rows-1 are exactly the leading rows*top_k ids — and the padding is
        # at the END because `GraphRunner.pad_batch` builds `padded_reqs = batch.reqs + [dummy_req]*k`.
        # A future change that interleaved or prepended dummies would make this silently keep garbage
        # and drop real routings, with nothing failing.
        rows = self._cur_rows
        if rows is None:
            self.rows_unknown += 1
            n = self.ring_width
        else:
            n = min(rows * self.top_k, self.ring_width)
        slot = self.slot
        if n > 0:
            self.ids_ring[slot, :, :n].copy_(self.stage[:, :n])
        if n < self.ring_width:
            # The tail is PADDING (a captured bucket wider than the live batch) or a narrower step's
            # unused width. Either way it must not reach the policy, and it must not be left holding
            # a wider previous step's ids in this reused slot.
            self.ids_ring[slot, :, n:].fill_(-1)
        self.stage.fill_(-1)
        self.step_meta[slot] = (self.step_id, self._cur_uid, self._cur_kind,
                                self.ring_rows if rows is None else rows)

    # -- the hot path -------------------------------------------------------------------------
    def record(self, topk_ids: torch.Tensor, num_tokens: int, expert_ids=None, ntp=None,
               block_m: int = 0) -> None:
        """One MoE call. DECODE: one device slice-assign, no sync, no .item(), no .tolist()."""
        if self.disarmed:
            return
        if torch.cuda.is_current_stream_capturing():
            # UNDER CAPTURE this is the ONLY legal path, and it must not be skipped: whatever is
            # recorded here is the whole of what every later replay will do. It cannot touch any
            # per-step host state -- `self.slot` is -1 at capture time (capture runs at engine init,
            # before the first `begin_forward`) and `_cur_kind` describes no real step -- so it is a
            # separate method that touches only the stage. Costs the eager path nothing: this is the
            # same single `is_current_stream_capturing()` call the old early-return made.
            self._record_captured(topk_ids)
            return
        if self.slot < 0:
            return
        lid = _CUR_LID
        if lid is None:
            raise RouteTraceError(
                "route trace: a MoE call reached w4a8_moe with no layer id in scope. A missing "
                "wrapper must be LOUD — a mislabeled trace is worse than no trace."
            )
        chunk = _CUR_CHUNK.get(lid, 0)
        _CUR_CHUNK[lid] = chunk + 1          # C7: tp_overlap can call a layer twice per forward

        M = int(topk_ids.shape[0])
        if chunk == 0 and self._cur_kind in (KIND_DECODE, KIND_VERIFY) and M <= self.ring_rows:
            # RING PATH, device-side, no sync. It used to be gated on M == 1 because
            # `reshape(-1)[:top_k]` keeps row 0 only and silently under-reports the union (C6) --
            # so the ring is now `top_k * ring_rows` wide and takes ALL M rows' ids, with the
            # dedupe in `drain()` collapsing them. That is what lets a spec VERIFY reach the
            # observer at all: under MTP every target forward is a verify with M = bs*(K+1), so an
            # M == 1 gate meant the ring never fired once and the expert cache sat inert, holding
            # its whole budget at fill=0.000 for zero hits.
            #
            # IT WRITES THE STAGE, NOT THE RING, and for the eager path that is not merely harmless
            # but one device op cheaper: `harvest` resets the whole stage to -1 once per step, so the
            # per-row tail clear this used to do (a second dispatch per layer per step, 48 of them)
            # is gone. The ring's step index cannot appear here at all -- see (1) in the module
            # docstring.
            n = M * self.top_k
            self.stage[lid, :n] = topk_ids.reshape(-1)[:n]
            if self._cur_rows is not None and M != self._cur_rows:
                # The row count `harvest` will mask against disagrees with the one this MoE call
                # actually carried. One int compare per layer per step, and it is worth it: a wrong
                # mask either truncates a real row's union (the cache looks BETTER than it is) or
                # admits a padded row's experts, and neither is visible in a hit rate.
                # NONZERO IS NOT AUTOMATICALLY A PLUMBING BUG. Two legitimate causes, both of which
                # genuinely DO give the policy a partial union and are worth surfacing:
                #   * a tp_overlap ROW SPLIT (>= 256 rows): chunk 0 is the only one the ring takes,
                #     so M is the chunk, not the forward;
                #   * EP: `MoELayer` sees the EP group's gathered rows (dp_size*bs), not this
                #     replica's. Before this change such a forward missed `ring_rows` entirely and
                #     the policy saw NOTHING, so masking to the local rows is strictly more.
                self.rows_mismatch += 1
            if self.blockmap_seen < self.blockmap_checks and ntp is not None and block_m:
                self._blockmap_check(topk_ids, expert_ids, ntp, block_m)
            return

        # HOST PATH: prefill chunks, and any decode/verify too wide for the ring. One blocking
        # .tolist(); a prefill chunk is ~35 ms of work so this is noise, and there are only ~28 per
        # 28k prompt. A VERIFY landing here is a sizing miss, not a normal path -- count it, because
        # these records never reach the observer and the cache would be fed a partial union.
        if self._cur_kind == KIND_VERIFY:
            self.rows_dropped += 1
        if self._cur_kind == KIND_PREFILL and not self.record_prefill:
            return
        ids = sorted({int(e) for row in topk_ids.tolist() for e in (row if isinstance(row, list) else [row])})
        kind = self._cur_kind if M > 1 or self._cur_kind != KIND_DECODE else KIND_OTHER
        self.oversize.append(
            (self.step_id, self._cur_uid, lid, kind, chunk, min(M, 0xFFFF), ids)
        )

    def _record_captured(self, topk_ids: torch.Tensor) -> None:
        """The ring write as it is RECORDED INTO A GRAPH. Also the CPU test seam for capture.

        Everything a graph bakes has to be a constant: the destination `stage[lid, :n]` is, because
        `lid` comes from this layer's own wrapper (the wrapper runs at capture, once per layer, so
        each layer's write is separately and correctly addressed) and `n` comes from the BUCKET width,
        which is fixed for this graph. Nothing per-step is read or written -- no slot, no kind, no
        meta, no `_CUR_CHUNK` -- because none of that host state exists at capture time and none of it
        would be re-evaluated at replay if it did.

        NO CHUNK BOOKKEEPING, and that is safe rather than sloppy: `rowchunked_ar_span` falls back to
        an unsplit `produce(x)` under capture (`tp_overlap.py::_overlappable` excludes capturing
        explicitly, since a side-stream collective cannot be recorded), so a captured forward calls
        each MoE layer exactly once. Leaving `_CUR_CHUNK` untouched also keeps the capture-time
        warmup from poisoning the first real step's chunk counters.

        THE BUCKET MUST FIT. `n > ring_width` would bake a graph whose write is silently truncated --
        every replay forever feeding the policy a partial union, which makes the cache look better
        than it is. `engine._route_trace_ring_rows` sizes the ring from `cuda_graph_max_bs` precisely
        so this cannot happen, so if it fires it is a sizing bug and must stop the boot, not the
        serve.
        """
        lid = _CUR_LID
        if lid is None:
            # An unwrapped MoE is being captured -- an MTP draft head is the plausible one (they are
            # filtered out of `self.ops` by design). Not fatal: a missing record starves the cache,
            # it cannot corrupt it. Loud once, then counted, because the eager path RAISES on this
            # and silently differing under capture is how a wrapper gap survives.
            self.capture_unwrapped += 1
            if self.capture_unwrapped == 1:
                print(
                    "[route-trace] a MoE call was CAPTURED with no layer id in scope: its routings "
                    "will never reach the expert cache. Check that every MoE layer this graph runs "
                    "is wrapped by install_hooks (MTP/draft heads are excluded on purpose).",
                    flush=True,
                )
            return
        n = int(topk_ids.shape[0]) * self.top_k
        if n > self.ring_width:
            raise RouteTraceError(
                f"route trace: capturing a {int(topk_ids.shape[0])}-row forward but the ring holds "
                f"{self.ring_rows} rows ({self.ring_width} ids). The graph would bake a truncated "
                f"write and every replay would feed the expert cache a partial expert union. Size "
                f"ring_rows from cuda_graph_max_bs (engine._route_trace_ring_rows)."
            )
        self.stage[lid, :n] = topk_ids.reshape(-1)[:n]

    def _blockmap_check(self, topk_ids, expert_ids, ntp, block_m: int) -> None:
        """T1, bounded: does moe_align ever emit a block under an expert NOT in topk_ids?

        If it does, the GEMM dereferences a slab this trace does not record and the cache looks
        BETTER than it is. Costs one ntp.item() sync, so it runs for the first N decode records only
        and then never again. A violation is a finding to report, not something to swallow.
        """
        self.blockmap_seen += 1
        live = int(ntp.item()) // int(block_m)
        extra = set(expert_ids[:live].tolist()) - set(topk_ids.flatten().tolist())
        extra.discard(self.num_experts)      # the aligner's padding sentinel
        if extra:
            self.blockmap_violations += 1
            print(
                f"[route-trace] BLOCK-MAP VIOLATION step={self.step_id} lid={_CUR_LID} "
                f"experts {sorted(extra)} have aligner blocks but are not in topk_ids. The GEMM "
                f"reads slabs this trace does not record; flags bit1 must be set and the analyzer "
                f"must union the block map.",
                flush=True,
            )

    # -- drain --------------------------------------------------------------------------------
    def set_observer(self, fn) -> None:
        """Subscribe a consumer to the drained records — this is how the expert cache learns routes.

        THE POINT IS THAT IT IS THE SAME RING. The cache needs "which experts did layer L read",
        which is exactly what this tracer already collects with no host sync; building a second
        observation path would duplicate the D2H. The consumer sees records N steps late by
        construction, and that lag is measured to be free (h 0.8558 at lag 0 vs 0.8563 at lag 64).
        """
        self._observer = fn

    def drain(self) -> None:
        """ONE D2H of the filled ring, then dedupe+sort per (step, layer) on host and append."""
        if torch.cuda.is_current_stream_capturing():
            # BEFORE the harvest, not after: harvesting under capture would record the staging copy
            # into the graph, which is precisely the mistake this module is being fixed to avoid.
            return
        if self.n_since_drain == 0 and not self.oversize:
            return
        # The step in flight staged its ids and nothing has moved them into the ring yet. `drain`
        # called from `begin_forward` has already harvested (and `harvest` is idempotent); `drain`
        # called from `close()` or straight from a test has not, and without this the last step --
        # the only one in a single-step test -- would be dropped.
        self.harvest()
        with self._lock:
            n = min(self.n_since_drain, self.ring_steps)
            self.ids_host.copy_(self.ids_ring, non_blocking=False)
            ids_np = self.ids_host.numpy()
            out = bytearray()
            first = self.step_id - n + 1
            for s in range(first, self.step_id + 1):
                slot = s % self.ring_steps
                sm = self.step_meta[slot]
                if sm is not None:
                    step, uid, kind, ntok = sm
                    # ALL LAYERS, and the ring row decides which ones actually recorded. `harvest`
                    # stores one tuple per STEP rather than the old per-lid dict entry, because every
                    # ring record of a step shares it -- and because under capture there is no host
                    # loop to build a per-lid dict at all (the graph writes the stage with no Python
                    # running). An all -1 row means that layer never recorded; it is skipped, exactly
                    # as a missing dict key used to be, rather than emitting a zero-expert record.
                    #
                    # TWO COLUMN BOUNDS, both of which matter now that `ring_rows` is sized from
                    # max_running_req/cuda_graph_max_bs instead of 1. The row is `top_k*ring_rows`
                    # wide, but only the leading `ntok*top_k` ids can be real, and the dedupe used to
                    # walk the FULL width in Python: at max_running_req 256 and top_k 10 that is
                    # 64 steps x 48 layers x 2560 = 7.8M interpreter iterations PER DRAIN, on the
                    # scheduler thread, to extract at most a few hundred ids. Slicing to the real
                    # width and deduping in numpy (`np.unique` returns sorted-unique, so the result is
                    # identical to the old `sorted({...})`) makes the drain proportional to the
                    # routing it actually carries. MEASURED host-side, CPU only, 2026-09-23, 48
                    # layers / top_k 10 / ring_rows 256 / 64 steps of 2 real rows: 50.1 ms for the
                    # whole 64-step window with the bound, 97.3 ms without it -- i.e. the unbounded
                    # scan is ~47 ms of scheduler-thread stall per drain, and that is WITH numpy
                    # doing the dedupe.
                    # (ntok == 0 -- an empty batch -- gives ncols 0, an empty slice, and no records.)
                    ncols = min(max(int(ntok), 0) * self.top_k, self.ring_width)
                    for lid in range(self.num_layers):
                        row = np.unique(ids_np[slot, lid, :ncols])
                        ids = [int(e) for e in row if 0 <= e < self.num_experts]
                        if not ids:
                            continue
                        obs = getattr(self, "_observer", None)
                        if obs is not None and kind in (KIND_DECODE, KIND_VERIFY):
                            # DECODE ONLY. A prefill chunk touches nearly every expert in the layer,
                            # so feeding it to the policy would look like one enormous sweep; the
                            # oracle measured that arm separately (prefill pollution, -0.0008 for
                            # SLRU) and the manager is sized for the decode working set.
                            obs(lid, ids)
                        out += struct.pack(RECORD_FMT, step & 0xFFFFFFFF, uid, lid, kind, 0,
                                           min(ntok, 0xFFFF), len(ids))
                        out += struct.pack(f"<{len(ids)}H", *ids)
                        self.records_written += 1
                self.step_meta[slot] = None
            for (step, uid, lid, kind, chunk, ntok, ids) in self.oversize:
                ids = ids[:0xFFFF]
                out += struct.pack(RECORD_FMT, step & 0xFFFFFFFF, uid, lid, kind, chunk,
                                   ntok, len(ids))
                out += struct.pack(f"<{len(ids)}H", *ids)
                self.records_written += 1
            self.oversize.clear()
            if self._fh is None:
                # OBSERVE-ONLY: the observer above has already been fed, which is the whole point of
                # the drain in this mode. Reset the ring and skip every file operation — `out` is
                # built unconditionally because the packing loop is also what computes `ids`, and
                # splitting it would give the cache and the capture fixture two different notions of
                # which experts a step touched.
                self.step_meta = [None] * self.ring_steps
                self.ids_ring.fill_(-1)
                self.n_since_drain = 0
                return
            self._fh.write(out)
            self._fh.flush()
            # Patch num_records IN PLACE (8 aligned bytes), then seek back to append. NOT
            # os.replace: that would rewrite a 35 MB file every drain for an 8-byte field. Readers
            # must ALSO tolerate num_records == 0 and scan to EOF, so a SIGTERM'd run is readable.
            end = self._fh.tell()
            self._fh.seek(NUM_RECORDS_OFFSET)
            self._fh.write(struct.pack("<Q", self.records_written))
            self._fh.flush()
            os.fsync(self._fh.fileno())
            self._fh.seek(end)
            self.ids_ring.fill_(-1)
            self.n_since_drain = 0

    def close(self) -> None:
        try:
            self.drain()
        finally:
            if self._fh is not None and not self._fh.closed:
                self._fh.close()
        # EVERY WAY THE UNION CAN BE WRONG, in one line. Each of these makes the cache look BETTER
        # than it is (a truncated or padding-polluted union), and none of them fails anything:
        #   rows_dropped     a forward too wide for ring_rows fell to the host path -> not observed
        #   rows_unknown     a step harvested with no real-row count -> padded rows NOT masked
        #   rows_mismatch    the plumbed row count disagreed with the forward's own M
        #   capture_unwrapped a captured MoE call had no layer id -> its layer is never observed
        warn = (f" rows_dropped={self.rows_dropped} rows_unknown={self.rows_unknown} "
                f"rows_mismatch={self.rows_mismatch} capture_unwrapped={self.capture_unwrapped}")
        if self._fh is None:
            print(
                f"[route-trace] observe-only: {self.step_id + 1} steps fed to the observer, "
                f"no fixture written;{warn}",
                flush=True,
            )
            return
        print(
            f"[route-trace] {self.records_written} records -> {self.path} "
            f"(steps={self.step_id + 1}, blockmap_checked={self.blockmap_seen}, "
            f"violations={self.blockmap_violations});{warn}",
            flush=True,
        )

    def stats(self) -> Dict[str, Any]:
        return {
            "route_trace_path": self.path,
            "route_trace_records": self.records_written,
            "route_trace_steps": self.step_id + 1,
            "route_trace_blockmap_checked": self.blockmap_seen,
            "route_trace_blockmap_violations": self.blockmap_violations,
            # See close(): the four ways a reported union can be narrower or dirtier than the truth.
            "route_trace_rows_dropped": self.rows_dropped,
            "route_trace_rows_unknown": self.rows_unknown,
            "route_trace_rows_mismatch": self.rows_mismatch,
            "route_trace_capture_unwrapped": self.capture_unwrapped,
        }


# ---- construction ---------------------------------------------------------------------------
def _derive_shape(model: Any) -> "tuple[int, int, int]":
    """(num_layers, num_experts, top_k) READ OFF THE OPS THAT WILL ACTUALLY BE WRAPPED.

    The plan originally took these from `config.model_config`. Deriving them from the discovered
    ops is strictly better and is why the caller no longer passes them: a transcribed config can
    disagree with what was wrapped (models/utils.py's `MoEMLP` builds its MoELayer with no `quant=`
    at all, which is the same class of transcription error `resolve_weight_plan` documents), and a
    header that disagrees with the trace body silently mis-scales every hit-rate the oracle prints.
    """
    from minisgl.weights.moe_interpose import discover_moe_layers
    from minisgl.weights.plan import is_mtp_path

    found = [(p, o) for p, o in discover_moe_layers(model) if not is_mtp_path(p)]
    if not found:
        raise RouteTraceError(
            "route trace armed but no non-MTP MoE layer was discovered — refusing to write a "
            "header describing a model this tracer is not attached to."
        )
    experts = {int(getattr(o, "num_experts", 0)) for _, o in found}
    topks = {int(getattr(o, "top_k", 0)) for _, o in found}
    if len(experts) != 1 or len(topks) != 1 or 0 in experts or 0 in topks:
        raise RouteTraceError(
            f"route trace: MoE layers disagree on shape (num_experts={sorted(experts)}, "
            f"top_k={sorted(topks)}). The trace format assumes one (E, top_k) for the whole model."
        )
    return len(found), experts.pop(), topks.pop()


def maybe_install(model: Any, *, model_slug: str, tp_rank: int, dp_rank: int,
                  device: torch.device, num_layers: "Optional[int]" = None,
                  num_experts: "Optional[int]" = None, top_k: "Optional[int]" = None,
                  expert_bytes: int = 1382400,
                  ring_rows: int = 1,
                  observe_only: bool = False) -> "Optional[RouteTracer]":
    """Arm the tracer iff MINISGL_MOE_ROUTE_TRACE names an existing writable dir.

    `observe_only=True` arms the ring with NO output file, for the expert cache: it needs the same
    references the fixture records and reusing this ring costs no extra D2H. If the env var ALSO
    names a dir, the fixture still wins — one ring, and a capture run keeps writing its trace while
    the cache observes the same records.

    Env convention follows kvcache/ghost_cache.py:64-83 and kvcache/_envutil.py: every knob in
    docker-compose.yml is declared `FOO: "${FOO:-}"`, so an unset variable arrives SET-BUT-EMPTY and
    `int(os.environ.get(...))` would raise at import. Use env_int. An EXPLICIT dir is required — no
    default into the repo layer, for exactly the reason ghost_oracle_path spells out.
    """
    global _TRACER
    d = os.environ.get("MINISGL_MOE_ROUTE_TRACE", "").strip() or None
    if d is None and not observe_only:
        return None
    if _TRACER is not None:
        raise RouteTraceError("route trace already armed")
    if num_layers is None or num_experts is None or top_k is None:
        num_layers, num_experts, top_k = _derive_shape(model)
    ring = env_int("MINISGL_MOE_ROUTE_TRACE_RING", 1024)
    # RING ROWS and the byte budget. One entry is `top_k * ring_rows` int32 per (step, layer), so
    # width scales with the widest forward the engine can produce (`engine._route_trace_ring_rows`:
    # max_running_req, cuda_graph_max_bs, and the spec K+1 factor) and the ring is
    # steps x layers x width x 4 B. A wide config would otherwise allocate gigabytes silently -- e.g.
    # max_running_req 64 at K=15 is 1024 rows, 10240 wide, 2 GB. Rows come first (an under-wide ring
    # drops decode/verify records to the host path, where they never reach the observer, and under
    # CAPTURE it is worse than that: `_record_captured` refuses to bake a truncated write and stops
    # the boot); `ring_steps` is what gives way, and loudly. The device cost on top of the ring is one
    # `stage` of layers x width x 4 B -- 1/ring_steps of it, i.e. unbudgeted on purpose.
    rows = max(1, int(ring_rows))
    width = max(1, int(top_k)) * rows
    budget = env_int("MINISGL_MOE_ROUTE_TRACE_MAX_MB", 64) * (1 << 20)
    per_step = max(1, int(num_layers)) * width * 4
    if per_step * ring > budget:
        shrunk = max(8, budget // per_step)
        _logger.info_rank0(
            f"[route-trace] ring {ring} -> {shrunk} steps to hold {rows} rows/entry "
            f"({width} ids x {num_layers} layers x 4 B = {per_step / 1024:.1f} KiB/step, "
            f"budget {budget >> 20} MiB). Rows are not negotiable: a verify that does not fit the "
            f"ring falls to the host path and never reaches the expert cache."
        )
        ring = shrunk
    t = RouteTracer(
        d,
        model_slug=model_slug,
        num_layers=num_layers,
        num_experts=num_experts,
        top_k=top_k,
        tp_rank=tp_rank,
        dp_rank=dp_rank,
        expert_bytes=expert_bytes,
        ring_steps=ring,
        ring_rows=rows,
        # The drain interval is the cache's LEARNING RATE: no reference reaches the policy until a
        # drain runs, so at the capture default of 512 a serve does hundreds of decode steps with
        # slot_of stuck at -1 and the cache cannot warm at all. The oracle measured a 64-step
        # observation lag as free (h 0.8558 -> 0.8563), so observe-only drains at 64.
        drain_every=env_int("MINISGL_MOE_ROUTE_TRACE_DRAIN",
                            min(512, ring) if d is not None else min(64, ring)),
        # max_steps bounds the FIXTURE (a 35 MB file is a measurement artefact, not a log). In
        # observe-only mode there is no file and the cache needs references for the life of the
        # serve, so it must never disarm — a tracer that quietly stopped at step 40,000 would
        # freeze the residency map and the hit rate would decay with no error anywhere.
        max_steps=(1 << 62) if d is None else env_int("MINISGL_MOE_ROUTE_TRACE_MAX", 40000),
        # Prefill records are never fed to the observer (decode-only, see drain()) and in
        # observe-only mode nothing else consumes them, so collecting them is pure overhead.
        record_prefill=(d is not None
                        and os.environ.get("MINISGL_MOE_ROUTE_TRACE_PREFILL", "1") != "0"),
        blockmap_checks=env_int("MINISGL_MOE_ROUTE_TRACE_BLOCKMAP", 64),
        device=device,
    )
    t.install_hooks(model)
    _TRACER = t
    print(
        f"[route-trace] ARMED: {t.path or 'observe-only (no fixture)'} ring={t.ring_steps} "
        f"drain={t.drain_every} max_steps={t.max_steps} prefill={t.record_prefill} "
        f"layers={len(t.ops)}",
        flush=True,
    )
    return t
