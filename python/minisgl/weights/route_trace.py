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

NOT CAPTURE-SAFE, AND THAT IS NOT A CONSTRAINT ON THIS ARM. `tools/serve.sh:754` sets `GRAPH_BS=0`
as the qwen4exp DEFAULT, with a measured rationale (capture buys 3.45-3.66 ms of a ~60 ms step and
costs 78% of context reach at MEM_RATIO 0.90). Under bucketed capture the Python in
`MoELayer.forward` runs once at capture and is dead at replay, and a bs=2 bucket's padded row 1
carries garbage routing that would pollute the trace. Every write here is nevertheless guarded on
`torch.cuda.is_current_stream_capturing()` so an accidental capture degrades to a gap in the trace,
never to an illegal op or a poisoned record. `tools/moe_route_stats.sh:17-20` is the precedent that a
routing statistic is legitimately taken eager: routing is a deterministic function of the hidden
states, identical captured or not.
"""

from __future__ import annotations

import os
import struct
import threading
from typing import Any, Dict, List, Optional, Tuple

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


def begin_forward(is_prefill: bool, req_uid: int) -> None:
    """Step boundary. Called from Scheduler._forward for EVERY loop (there are eight)."""
    t = _TRACER
    if t is not None:
        t.begin_forward(is_prefill, req_uid)


def close() -> None:
    global _TRACER
    t = _TRACER
    if t is not None:
        _TRACER = None
        t.close()


# ---- the tracer -----------------------------------------------------------------------------
class RouteTraceError(RuntimeError):
    """The tracer cannot honour its own invariants. Never downgraded to a warning: a measurement
    fixture that silently degrades is how /home/pat/fixtures/minisgl-ghost-oracle ended up with one
    root-owned Aug-6 file that nobody noticed was stale."""


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

        # DEVICE ring: ids only. -1 = "this (step, layer) had no decode record".
        self.ids_ring = torch.full(
            (self.ring_steps, self.num_layers, self.top_k), -1, dtype=torch.int32, device=device
        )
        # Pinned host staging for the ONE D2H per drain (the qsa runtime.py:331-333 pattern).
        self.ids_host = torch.empty(
            (self.ring_steps, self.num_layers, self.top_k), dtype=torch.int32, pin_memory=True
        )
        # HOST meta (C8): every field is host-known at the call, so staging it on device would be a
        # second dispatch per layer per step for nothing. slot -> lid -> (step, uid, kind, chunk, ntok)
        self.meta: List[Dict[int, Tuple[int, int, int, int, int]]] = [
            {} for _ in range(self.ring_steps)
        ]
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
    def begin_forward(self, is_prefill: bool, req_uid: int) -> None:
        if self.disarmed:
            return
        assert threading.get_ident() == self._thread, (
            "route trace: a second thread drove a forward. The module globals that carry the layer "
            "id are only legal because the scheduler loop is single-threaded."
        )
        # Drain FIRST, at the step boundary: outside the forward, outside any capture region, and
        # the only point every one of the eight _forward call sites passes through. _hp_tick is NOT
        # such a point (it runs only under MINISGL_HOSTPROF) — see C5.
        if self.n_since_drain >= self.drain_every:
            self.drain()
        self.step_id += 1
        if self.step_id >= self.max_steps:
            self.disarmed = True
            self.close()
            return
        self.slot = self.step_id % self.ring_steps
        self.meta[self.slot] = {}
        self._cur_uid = int(req_uid) & 0xFFFFFFFF
        self._cur_kind = KIND_PREFILL if is_prefill else KIND_DECODE
        self.n_since_drain += 1
        _CUR_CHUNK.clear()

    # -- the hot path -------------------------------------------------------------------------
    def record(self, topk_ids: torch.Tensor, num_tokens: int, expert_ids=None, ntp=None,
               block_m: int = 0) -> None:
        """One MoE call. DECODE: one device slice-assign, no sync, no .item(), no .tolist()."""
        if self.disarmed or self.slot < 0:
            return
        if torch.cuda.is_current_stream_capturing():
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
        if M == 1 and self._cur_kind == KIND_DECODE and chunk == 0:
            # RING PATH. Gated on M == 1, not `M <= 8`: at M>1 `reshape(-1)[:top_k]` keeps row 0
            # only and silently under-reports the union (C6).
            self.ids_ring[self.slot, lid, :] = topk_ids.reshape(-1)[: self.top_k]
            self.meta[self.slot][lid] = (self.step_id, self._cur_uid, KIND_DECODE, 0, 1)
            if self.blockmap_seen < self.blockmap_checks and ntp is not None and block_m:
                self._blockmap_check(topk_ids, expert_ids, ntp, block_m)
            return

        # HOST PATH: prefill chunks, bs>1 decode, spec-verify rows. One blocking .tolist(); a
        # prefill chunk is ~35 ms of work so this is noise, and there are only ~28 per 28k prompt.
        if self._cur_kind == KIND_PREFILL and not self.record_prefill:
            return
        ids = sorted({int(e) for row in topk_ids.tolist() for e in (row if isinstance(row, list) else [row])})
        kind = self._cur_kind if M > 1 or self._cur_kind != KIND_DECODE else KIND_OTHER
        self.oversize.append(
            (self.step_id, self._cur_uid, lid, kind, chunk, min(M, 0xFFFF), ids)
        )

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
        if self.n_since_drain == 0 and not self.oversize:
            return
        if torch.cuda.is_current_stream_capturing():
            return
        with self._lock:
            n = min(self.n_since_drain, self.ring_steps)
            self.ids_host.copy_(self.ids_ring, non_blocking=False)
            ids_np = self.ids_host.numpy()
            out = bytearray()
            first = self.step_id - n + 1
            for s in range(first, self.step_id + 1):
                slot = s % self.ring_steps
                for lid, (step, uid, kind, chunk, ntok) in sorted(self.meta[slot].items()):
                    row = ids_np[slot, lid]
                    ids = sorted({int(e) for e in row if 0 <= int(e) < self.num_experts})
                    obs = getattr(self, "_observer", None)
                    if obs is not None and kind == KIND_DECODE:
                        # DECODE ONLY. A prefill chunk touches nearly every expert in the layer, so
                        # feeding it to the policy would look like one enormous sweep; the oracle
                        # measured that arm separately (prefill pollution, -0.0008 for SLRU) and the
                        # manager is sized for the decode working set.
                        obs(lid, ids)
                    out += struct.pack(RECORD_FMT, step & 0xFFFFFFFF, uid, lid, kind, chunk,
                                       ntok, len(ids))
                    out += struct.pack(f"<{len(ids)}H", *ids)
                    self.records_written += 1
                self.meta[slot] = {}
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
                self.meta = [{} for _ in range(self.ring_steps)]
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
        if self._fh is None:
            print(
                f"[route-trace] observe-only: {self.step_id + 1} steps fed to the observer, "
                f"no fixture written",
                flush=True,
            )
            return
        print(
            f"[route-trace] {self.records_written} records -> {self.path} "
            f"(steps={self.step_id + 1}, blockmap_checked={self.blockmap_seen}, "
            f"violations={self.blockmap_violations})",
            flush=True,
        )

    def stats(self) -> Dict[str, Any]:
        return {
            "route_trace_path": self.path,
            "route_trace_records": self.records_written,
            "route_trace_steps": self.step_id + 1,
            "route_trace_blockmap_checked": self.blockmap_seen,
            "route_trace_blockmap_violations": self.blockmap_violations,
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
