# Routing-trace capture hook — implementation plan (NOT APPLIED)

**Status:** plan only. No python file in this worktree is edited by this document. The simulator is
being written in parallel by another agent; the capture seam sits on the served decode hot path and
must be reviewed before it lands.

**Worktree:** `/home/pat/code/minisgl-rdna4-oracle` (branch `task/routing-oracle`, HEAD `70b7ee85`).
**Model:** `qwen4_exp` = Qwen3.8-Flash-Next NVFP4, 48 MoE layers, E=512, top_k=10, TP=2.

**What this hook decides.** Our own P2-prime scoping concluded the miss curve is *linear*
(cliff_index 0.090/0.094 vs 0.100 pure-linear) — i.e. hit rate == resident fraction, routing is not
skewed, and an LRU hot-expert cache cannot beat static placement. A third party reports vLLM + LRU
on 2x3080-20GB reaching 27-30 tok/s at ~44% residency, which a linear miss curve cannot produce.
One of those is wrong. This hook produces the real per-(layer, expert, step) access stream that
settles it. Everything downstream of the trace is CPU-only.

---

## 0. Corrections to the design-phase spec (read these first)

I checked every anchor the spec cited against the tree. Seven are wrong or incomplete, and three of
them would produce a **silently wrong or silently empty trace**. The diff below implements the
corrected version.

| # | Spec said | Tree says | Consequence if followed literally |
|---|---|---|---|
| C1 | `is_mtp_path` at `moe_interpose.py:276` | `is_mtp_path` is at **`weights/plan.py:434`**; `discover_moe_layers` is at `moe_interpose.py:275` | ImportError at boot (loud, harmless) |
| C2 | parse lid from the dotted path in the tracer | `stream_tier.layer_index_of_path()` (**`stream_tier.py:73`**) already does exactly this and *raises* rather than defaulting | a second parser to keep in sync; reuse it |
| C3 | `self._step_id` on Scheduler | **does not exist.** No forward counter on Scheduler at all | `AttributeError` at the first forward |
| C4 | 2-line edit at `scheduler.py:1148-1156` | there are **eight** `self._forward(...)` call sites: `1093, 1132, 1150, 2989, 3015, 3115, 3137, 4546` | the hook covers 1 of 8 loops. The overlap loop (`1093`) and the DP/spec loops would emit records with a stale `_CUR` — **mislabeled, not empty** |
| C5 | `maybe_drain()` next to `_hp_tick()` | `_hp_tick()` is only reached when `self._hp is not None`, i.e. only under `MINISGL_HOSTPROF` | **the ring never drains on a normal serve; the trace file stays empty.** This is the ghost-oracle failure signature again |
| C6 | decode ring row = `topk_ids.reshape(-1)[:top_k]` for `num_tokens <= 8` | at bs>1 decode `topk_ids` is `(M, top_k)`; `reshape(-1)[:10]` silently keeps **row 0 only** | at CONC>1 the trace under-reports the union → **LRU looks better than it is.** Ring path must be gated on `M == 1`, everything else takes the host path |
| C7 | `_ROUTE_TRACE.record(topk_ids, M, 0)` — chunk_idx hardcoded 0 **[FIXED 2026-09-08: the proposed fix ALSO failed — it reset the counter per forward-entry, but the row split calls the wrapper once per chunk, so both chunks still stamped 0. The reset is now only at the step boundary.]** | `tp_overlap._MIN_TOKENS = 256` (`tp_overlap.py:153`), so a >=256-row prefill calls `w4a8_moe` **twice** for one layer | two records with identical `(step_id, layer_id, chunk_idx)` → the analyzer's dedupe key collides and one chunk's experts vanish. `chunk_idx` must be a per-(step, layer) counter the tracer owns |
| C8 | `meta_ring` int32 on device | every meta field (`step_id`, `req_uid`, `kind`, `chunk_idx`, `num_tokens`) is **host-known** at the call | a pointless second device dispatch per layer per step. Keeping meta host-side **halves** the hot-path cost (96 → 48 dispatches/step) |

Two further findings that are not spec bugs but must be settled before the simulator quotes a number:

* **T0 — `expert_bytes` reconciliation.** The header's `expert_bytes = 1382400` is the correct
  rank-local granule (w13 `2*320*2560/2` = 819200 + w2 `2560*320/2` = 409600 + e4m3 group-16 block
  scales `(1638400+819200)/16` = 153600). But `48 * 512 * 1382400 * 2 ranks = 63.28 GiB`, and the
  brief quotes **68.42 GiB** total expert bytes — an 8.1% gap. The residency fraction (6.7 GiB/rank
  ÷ total) is the x-axis of the entire miss curve, so an 8% error in the denominator moves the
  operating point. Reconcile before the sim runs: either the brief includes the NVFP4 per-expert
  global scales / the gate / the shared expert, or one of the two is wrong. **The simulator must use
  the `expert_bytes` the header declares, and the header must be the reconciled value.**
* **T1 — the aligner's block map.** `stream_tier.route_ids()` (`stream_tier.py:349-376`) deliberately
  unions `expert_ids[:live]` into the route because *"a future aligner that emits a padding block
  under some expert id"* would make the GEMM dereference a slab that is not in `topk_ids`. If that
  ever happens, the GEMM reads **more** bytes than this trace records, and the cache looks better
  than it is. Getting `live` costs `ntp.item()` — one host sync, forbidden on the decode path.
  Resolution: a bounded self-check, `MINISGL_MOE_ROUTE_TRACE_BLOCKMAP=<N>` (default 64), pays the
  sync for the **first N decode records only** and asserts `set(expert_ids[:live]) ⊆ set(topk_ids)`.
  If all N pass, `flags` bit1 stays 0 and the analyzer may trust `topk_ids` alone for the whole run;
  if any fails it is a finding, not something to swallow.

---

## 1. Insertion points, and why this source is authoritative

### 1.1 The route producer — `python/minisgl/quant/kernels.py:657-659`, inside `w4a8_moe()`

```
655            "align", lambda: moe_hip.moe_align(topk_ids, E, block_m)
656        )
657    P = sorted_ids.shape[0]
658    if _ROUTE_STATS_PATH:
659        _route_stats(topk_ids, E, block_m, ntp)
```

This is the **one authoritative producer** of the route on the served NVFP4 path, for four
independently checked reasons:

1. **The served arm reaches `w4a8_moe` with `topk_ids=None`.** `MoELayer.forward`
   (`layers/moe.py:1280`) only precomputes a route when `self.enable_ep` (`moe.py:1355`); `enable_ep`
   is False for qwen4_exp because `tools/serve.sh` never passes `--enable-ep` on this arm. So control
   reaches `kernels.py:645-648`, `_route_align(...)`, which is `moe_hip.moe_route_align` — softmax +
   top-k + renormalize + `moe_align` in **one** op (`kernels.py:401-425`).
2. **By line 657 the route is already computed.** Recording here reuses the tensor the GEMM is about
   to consume. Zero extra `route_align` launches, zero extra plumbing through the seven
   `MoEQuantMethod.apply` implementations, no signature change anywhere.
3. **The Python `torch.topk` route is NOT tie-identical to the kernel's.** `moe.py:1185-1200` records
   the measured disagreement *on this checkpoint*: layer 0 of an 8-token prefill, experts **324 and
   366 both scored exactly `-5.09375`** at ranks 9 and 10 straddling the k=10 cut; the kernel kept the
   lower index (324), `torch.topk` kept the higher (366). **Two of the eight MoE calls in that single
   forward diverged by one expert.** The gate logits are bf16 (8 mantissa bits) spread over 512
   experts, so ties are ordinary, not exotic. That docstring closes with the rule this plan obeys
   verbatim: *"Anything that needs the set of experts the GEMM will actually dereference (a routed
   weight gather, an offload prefetch) must call `quant.kernels._route_align` and read its
   `topk_ids`/`expert_ids`, not re-derive it here."* `stream_tier.py:41-49` and `:349-376` exist
   because of the same measurement.
4. **Precedent.** `_ROUTE_STATS_PATH` (`kernels.py:166`, probe body `:173-221`) is an existing
   env-gated routing probe at this exact line. The new hook sits beside it, one `is not None` when
   unarmed.

### 1.2 The layer id — a wrapper on `MoELayer.forward`, installed after the stream tier

`w4a8_moe` does not know its layer. The lid arrives through a module global set by an outermost
wrapper on `MoELayer.forward`, mirroring `ExpertStreamTier.install_hooks`
(`stream_tier.py:378-405`): an **instance attribute shadows the bound method**, so no model file is
edited.

* Layers come from `discover_moe_layers` (`moe_interpose.py:275-291`), keyed by the **structural**
  dotted path `model.layers.{lid}.mlp.experts`. That function's own docstring says why: *"the MTP
  draft head builds its own `MoELayer`, so a counter renumbers every layer after it."* **Never a call
  counter, never `call_index % 48`.**
* MTP layers are dropped with `plan.is_mtp_path()` (`plan.py:434-442`, segment equality on `mtp`).
  `qwen4exp.py` refuses `--spec-algorithm mtp` for this architecture and `serve.sh` sets
  `spec_default=none`, so the filter should remove nothing — **assert the count is exactly
  `num_layers` (48) anyway.** A count of 49 means an MTP `MoELayer` slipped the filter.
* lid is recovered with the existing `stream_tier.layer_index_of_path()` (`stream_tier.py:73-87`),
  which raises on an unparseable path rather than defaulting (C2).

**Install order.** `RouteTracer.install_hooks` must run **after**
`WeightOffloadSession.install_stream_hooks()` (`bake.py:652-675`, called from
`engine/engine.py:376`, itself after `seal()`). Later shadowing = outermost wrapper, so `_CUR_LID` is
set before the stream tier's staging runs and the tier's own `route_ids()` call is inside the
labelled window. Installing before it would leave the tier's staging unlabelled.

### 1.3 The step boundary — inside `Scheduler._forward()`, `scheduler.py:2843-2846`

Not `normal_loop`. There are **eight** `_forward` call sites (C4) and the instrumented tail the spec
pointed at is dead unless `MINISGL_HOSTPROF` is set (C5). `_forward()` is the single choke point that
every loop — normal, overlap (`:1093`), hostprof (`:1150`), DP-dummy (`:3115`, `:3137`), spec
(`:2989`, `:3015`, `:4546`) — funnels through, and at line 2844-2845 `batch`, `_step_is_pf` and
`_step_bs` are already in hand, read *before* `forward_batch` mutates them (the file's own comment at
`:2833-2838` explains why they are read there).

The **drain also lives at that boundary**, at the top of `begin_forward`: it is outside the forward,
outside any capture region, and it is the only point guaranteed to run on every loop. It is guarded
on `not torch.cuda.is_current_stream_capturing()` regardless.

---

## 2. The exact diff (ready to apply — DO NOT APPLY YET)

### 2.1 New file — `python/minisgl/weights/route_trace.py`

```python
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
        if not os.path.isdir(out_dir):
            raise RouteTraceError(
                f"MINISGL_MOE_ROUTE_TRACE={out_dir!r} is not a directory. The trace is a durable "
                f"measurement fixture and must land on a path the operator pre-created and mounted "
                f"(`install -d -o 1000 -g 1000 <dir>`), never in a container layer or a worktree "
                f"that gets removed. Refusing to boot rather than degrading silently."
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
            if not self._fh.closed:
                self._fh.close()
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
def maybe_install(model: Any, *, model_slug: str, num_layers: int, num_experts: int, top_k: int,
                  tp_rank: int, dp_rank: int, device: torch.device,
                  expert_bytes: int = 1382400) -> "Optional[RouteTracer]":
    """Arm the tracer iff MINISGL_MOE_ROUTE_TRACE names an existing writable dir.

    Env convention follows kvcache/ghost_cache.py:64-83 and kvcache/_envutil.py: every knob in
    docker-compose.yml is declared `FOO: "${FOO:-}"`, so an unset variable arrives SET-BUT-EMPTY and
    `int(os.environ.get(...))` would raise at import. Use env_int. An EXPLICIT dir is required — no
    default into the repo layer, for exactly the reason ghost_oracle_path spells out.
    """
    global _TRACER
    d = os.environ.get("MINISGL_MOE_ROUTE_TRACE", "").strip()
    if not d:
        return None
    if _TRACER is not None:
        raise RouteTraceError("route trace already armed")
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
        drain_every=env_int("MINISGL_MOE_ROUTE_TRACE_DRAIN", min(512, ring)),
        max_steps=env_int("MINISGL_MOE_ROUTE_TRACE_MAX", 40000),
        record_prefill=os.environ.get("MINISGL_MOE_ROUTE_TRACE_PREFILL", "1") != "0",
        blockmap_checks=env_int("MINISGL_MOE_ROUTE_TRACE_BLOCKMAP", 64),
        device=device,
    )
    t.install_hooks(model)
    _TRACER = t
    print(
        f"[route-trace] ARMED: {t.path} ring={t.ring_steps} drain={t.drain_every} "
        f"max_steps={t.max_steps} prefill={t.record_prefill} layers={len(t.ops)}",
        flush=True,
    )
    return t
```

### 2.2 `python/minisgl/quant/kernels.py` — 3 added lines

```diff
--- a/python/minisgl/quant/kernels.py
+++ b/python/minisgl/quant/kernels.py
@@ -655,6 +655,12 @@
             "align", lambda: moe_hip.moe_align(topk_ids, E, block_m)
         )
     P = sorted_ids.shape[0]
     if _ROUTE_STATS_PATH:
         _route_stats(topk_ids, E, block_m, ntp)
+    # ROUTING TRACE (measure-only, OFF unless MINISGL_MOE_ROUTE_TRACE names a dir). Reuses the route
+    # JUST computed by _route_align — the op the GEMM below actually consumes — so it costs zero
+    # extra launches and needs no change to any of the seven MoEQuantMethod.apply signatures. The
+    # layer id arrives via a module global set by route_trace's MoELayer.forward wrapper.
+    if _route_trace.enabled():
+        _route_trace.tracer().record(topk_ids, M, expert_ids, ntp, block_m)
 
     _e2m1 = "+e2m1" if weight_is_e2m1 else ""
```

and, beside the existing probe declaration at `kernels.py:166`:

```diff
--- a/python/minisgl/quant/kernels.py
+++ b/python/minisgl/quant/kernels.py
@@ -164,6 +164,8 @@
 # Recorded per MoE call; rows are dumped when the sample cap is hit.
 _ROUTE_STATS_PATH = _os.environ.get("MINISGL_MOE_ROUTE_STATS", "")
+from minisgl.weights import route_trace as _route_trace  # measure-only; `enabled()` is one `is not None`
+
 _ROUTE_STATS_MAX = int(_os.environ.get("MINISGL_MOE_ROUTE_STATS_N") or 4000)
```

> **Import-cycle check for review:** `minisgl.weights.route_trace` imports `torch`,
> `minisgl.kvcache._envutil` and nothing from `minisgl.quant` at module scope —
> `moe_interpose` / `plan` / `stream_tier` are imported *inside* `install_hooks`. If the cycle bites
> anyway, move this to a lazy module-level `_route_trace = None` + import in `maybe_install`.

### 2.3 `python/minisgl/scheduler/scheduler.py` — 4 added lines, ONE choke point

```diff
--- a/python/minisgl/scheduler/scheduler.py
+++ b/python/minisgl/scheduler/scheduler.py
@@ -2841,6 +2841,10 @@
         if STEP_LOG_SYNC:
             torch.cuda.synchronize()
         _step_t0 = time.perf_counter()
         _step_is_pf = bool(batch.is_prefill)
         _step_bs = int(batch.size)
+        # ROUTING TRACE step boundary (no-op — one `is not None` — unless armed). HERE and not in
+        # normal_loop: there are EIGHT `self._forward(...)` call sites (overlap, hostprof, DP-dummy,
+        # spec) and this is the only point all of them pass through. The drain runs at the top of
+        # begin_forward: outside the forward, outside any capture region.
+        _route_trace.begin_forward(_step_is_pf, batch.reqs[0].uid if batch.reqs else 0)
         _step_r0 = self.engine.graph_runner.replays
```

plus the import at the top of `scheduler.py`:

```diff
+from minisgl.weights import route_trace as _route_trace
```

### 2.4 `python/minisgl/engine/engine.py` — 8 added lines, after the stream tier

```diff
--- a/python/minisgl/engine/engine.py
+++ b/python/minisgl/engine/engine.py
@@ -373,6 +373,17 @@
         # AFTER seal, which is the moment residency stops changing. The hook is a forward-path
         # wrapper, not a residency change, but installing it inside the window would put file I/O in
         # the middle of the gates that are still measuring the window's device cost.
         self._woff.install_stream_hooks()
+        # ROUTING TRACE, AFTER the stream tier so this wrapper is the OUTERMOST one and _CUR_LID is
+        # set before ExpertStreamTier's staging runs. Measure-only; returns None unless
+        # MINISGL_MOE_ROUTE_TRACE names an existing writable dir, and RAISES if it names one that
+        # does not exist rather than degrading to a silent no-trace.
+        from minisgl.weights import route_trace as _route_trace
+
+        _route_trace.maybe_install(
+            self.model,
+            model_slug=str(config.model_config.model_path),
+            num_layers=<the MoE layer count from config>,
+            num_experts=<config num_experts>,
+            top_k=<config top_k>,
+            tp_rank=self.tp_rank,
+            dp_rank=self.dp_rank,
+            device=self.device,
+        )
         if self._woff.enabled:
             _mem_probe("after weight offload")
```

> The four `<...>` placeholders are the one part I could not pin without reading the qwen4_exp config
> plumbing end-to-end; the implementer should take them from the same `config.model_config` fields
> `MoELayer.__init__` reads (`moe.py:1050-1058`), or simply derive `num_layers`, `num_experts`,
> `top_k` from the discovered ops themselves (`op.num_experts`, `op.top_k`, `len(found)`) and drop
> them from the signature. Deriving from the ops is strictly better — it cannot disagree with what
> was actually wrapped.

Finally, `close()` should be called from the engine/scheduler shutdown path so a clean stop patches
the header; the reader tolerates `num_records == 0` and scans to EOF, so a `SIGKILL` still yields a
readable file.

---

## 3. Gating — all default OFF

Convention taken from `kvcache/ghost_cache.py:57-83` (an **explicit** dir, no default into the repo)
and `kvcache/_envutil.py:1-26` (compose declares every knob `FOO: "${FOO:-}"`, so unset arrives as
**set-but-empty** and bare `int()` raises at import — use `env_int`).

| Env var | Default | Meaning |
|---|---|---|
| `MINISGL_MOE_ROUTE_TRACE` | `""` (OFF) | Output dir. Arms the tracer. **Must exist and be writable, else RAISE AT BOOT.** |
| `MINISGL_MOE_ROUTE_TRACE_RING` | `1024` | Ring depth in steps |
| `MINISGL_MOE_ROUTE_TRACE_DRAIN` | `512` | Drain cadence; must be `<= ring` (checked, raises) |
| `MINISGL_MOE_ROUTE_TRACE_MAX` | `40000` | Auto-disarm + final flush; bounds the file and the run |
| `MINISGL_MOE_ROUTE_TRACE_PREFILL` | `1` | Record prefill calls. **Required for the pollution analysis — do not turn off** |
| `MINISGL_MOE_ROUTE_TRACE_BLOCKMAP` | `64` | T1 self-check budget: decode records that pay one `ntp.item()` to verify the aligner emits no off-route block. `0` disables |

**Never silently degrade.** A named-but-missing dir raises; `drain > ring` raises; a MoE call with no
lid in scope raises; a wrapped-layer count `!= 48` raises. `/home/pat/fixtures/minisgl-ghost-oracle/BEFORE-ladder4-cap41.bin`
is root-owned and dated Aug 6 — this repo's known silent-install-failure signature — and every raise
above exists to make that impossible here.

**Compose** (`docker-compose.yml`), mirroring the `MINISGL_GHOST_ORACLE*` + `/ghost` block at
`:107-114`, `:49` and `:670`:

```diff
   volumes:
     - ${GHOST_ORACLE_DIR:-/home/pat/fixtures/minisgl-ghost-oracle}:/ghost
+    - ${ROUTE_TRACE_DIR:-/home/pat/fixtures/minisgl-route-trace-2026-09-08}:/route
```
```diff
     MINISGL_GHOST_ORACLE_DIR: "${MINISGL_GHOST_ORACLE_DIR:-/ghost}"
+    # ---- MoE routing trace (MEASURE-ONLY; settles linear-miss-curve vs LRU) ----
+    MINISGL_MOE_ROUTE_TRACE: "${MINISGL_MOE_ROUTE_TRACE:-}"
+    MINISGL_MOE_ROUTE_TRACE_RING: "${MINISGL_MOE_ROUTE_TRACE_RING:-}"
+    MINISGL_MOE_ROUTE_TRACE_DRAIN: "${MINISGL_MOE_ROUTE_TRACE_DRAIN:-}"
+    MINISGL_MOE_ROUTE_TRACE_MAX: "${MINISGL_MOE_ROUTE_TRACE_MAX:-}"
+    MINISGL_MOE_ROUTE_TRACE_PREFILL: "${MINISGL_MOE_ROUTE_TRACE_PREFILL:-}"
+    MINISGL_MOE_ROUTE_TRACE_BLOCKMAP: "${MINISGL_MOE_ROUTE_TRACE_BLOCKMAP:-}"
```

The `serve` service at `:668-670` **redefines** volumes rather than inheriting `*lean-common`, so the
`/route` mount must be added in **both** places — the same trap the `/ghost` comment at `:668` calls
out. Missing the second one is the five-hops metric-plumbing failure: it exports nothing, not an error.

---

## 4. Measured hot-path cost

**Per layer per decode step (device):** one slice-assign of `top_k=10` int32 = **40 B**. Meta is
host-side (C8), so there is **no second device op**. That is **48 dispatches/step**, not 96.

Repo-measured dispatch floors on this box (`kernels.py:411-417`): smallest possible torch dispatch
(a 1-element `add_`) = **3.098 us**; work-free kernel floor = **3.955 us**.

| | per step | on a 60.06 ms step |
|---|---|---|
| 48 slice-assigns @ 3.098 us | 0.149 ms | **0.25%** |
| 48 slice-assigns @ 3.955 us | 0.190 ms | **0.32%** |
| (spec's 2-op version, for contrast) | 0.30–0.38 ms | 0.50–0.63% |

**D2H count: one per `drain_every` steps.** Ring = `1024 x 48 x 10 x int32` = **1.966 MB**. At the
measured host bandwidths (card 0 root port 28.93 GB/s, card 1 **14.48 GB/s** — card 1's root port
trained Gen4 x8, so it gates every TP=2 host transfer): **0.068 ms (card 0) / 0.136 ms (card 1)**,
once every 512 steps = **0.00027 ms/step amortized**, i.e. under 0.0005%. Host-side dedupe+sort of
512x48 rows of 10 and the file append are ~10-25 ms of pure CPU once per 512 steps, on the scheduler
thread; if that shows up in `[hostprof]`, move the drain to a worker thread (the file is already
mutex-guarded) rather than shrinking the cadence.

**Bytes written:** decode record = 16 B header + 10 x u16 = **36 B**; x48 layers = **1,728 B/step**;
20,000 steps = **34.6 MB**. Prefill records are capped at 16 + 2 x num_ids <= ~1,040 B per layer per
chunk; a 28k-token prompt at `MAX_PREFILL_LENGTH=1024` is 28 chunks x 48 layers ~= **1.4 MB**. Whole
run comfortably **under 100 MB**, ~4x smaller after zstd.

**Prefill cost:** one blocking `.tolist()` per (layer, chunk). ~28 chunks x 48 layers per 28k prompt,
each ~20 KB. Adds seconds to a prefill that already takes minutes. Irrelevant to a routing statistic.

### Keeping it off the critical path

* **The device ring is the mechanism** — there is no host sync on the decode path at all. The
  forbidden alternative has a price tag in this repo: commit `ffa1d8c6`, QSA's
  `int(lens.to(torch.int64).sum().item())` was **36% of scheduler-rank py-spy samples**, and deleting
  it took the serve **131 → 85 ms/token**. Across 48 MoE layers that is roughly **+180 ms/step, ~4x
  slower**.
* **Drain at the step boundary**, before the forward is issued, guarded on
  `is_current_stream_capturing()`.
* **A stride knob ("capture every Nth step") is deliberately NOT offered as an admissible setting.**
  Subsampling steps destroys the reuse-distance structure the LRU simulation is built on — it
  fabricates locality by deleting the intervening accesses that would have evicted. If the 0.25%
  ever needs cutting, cut **layers** (trace a contiguous subset and extrapolate), never steps. A
  strided trace is inadmissible for the LRU question, full stop.
* **The overhead denominator must be re-measured, not quoted.** `tools/serve.sh` carries an
  "[ALL-HOST 2026-09-07]" note (commit `cd40a9ab`, all MoE layers from system RAM) that may have
  superseded the 38+10 split behind the 60.06 ms figure. The split changes nothing in the simulator
  (it re-derives placement itself) but it *is* the ms/step denominator above — record the booted
  split from `ExpertStreamTier.stats()` into the sidecar and recompute the percentage.

---

## 5. Trace format, and how it maps onto the simulator

Durable fixture dir, pre-created and **never inside a worktree** (worktrees get removed — the same
reasoning `ghost_cache.ghost_oracle_path` spells out at `:64-83`):

```
install -d -o 1000 -g 1000 /home/pat/fixtures/minisgl-route-trace-2026-09-08
```
```
/home/pat/fixtures/minisgl-route-trace-2026-09-08/
  route.<slug>.rank0.bin        route.<slug>.rank0.meta.json
  route.<slug>.rank1.bin        route.<slug>.rank1.meta.json
```

Binary, little-endian, uncompressed (zstd afterwards if wanted, ~4x).

**HEADER — 64 bytes, `struct '<8sIIIIIIQIIQ8x'`** (verified `struct.calcsize` == 64):

| field | type | value |
|---|---|---|
| `magic` | `char[8]` | `b"MSGLRT01"` |
| `version` | u32 | 1 |
| `num_layers` | u32 | 48 |
| `num_experts` | u32 | 512 |
| `top_k` | u32 | 10 |
| `tp_rank` / `dp_rank` | u32, u32 | |
| `num_records` | u64 | patched at every drain and at close (offset **32**). **Readers MUST also tolerate 0 and scan to EOF** so a SIGTERM'd run is still readable |
| `expert_bytes` | u32 | `1382400` rank-local — see **T0**, reconcile against the brief's 68.42 GiB total first |
| `flags` | u32 | bit0 = ids are the deduped sorted union for the call (always 1 in v1); bit1 = the aligner block map was OR'd in (0 unless the T1 check fires) |
| `model_hash` | u64 | FNV-1a of the model slug — a trace from another checkpoint is incomparable |
| reserved | 8 B | zero |

**RECORD — 16 B header + payload, `struct '<IIHBBHH'` then `u16[num_ids]`** (verified == 16):

| field | type | meaning |
|---|---|---|
| `step_id` | u32 | monotonic forward counter **owned by the tracer** (the Scheduler has none — C3) |
| `req_uid` | u32 | low 32 bits of `batch.reqs[0].uid` (`scheduler/utils.py:16`), 0 if none — lets the analyzer prove >=12 distinct prompts |
| `layer_id` | u16 | **STRUCTURAL** index from `model.layers.{lid}.mlp.experts` via `layer_index_of_path`. Never a call counter |
| `kind` | u8 | 0 prefill chunk, 1 decode, 2 other/spec-verify/bs>1 |
| `chunk_idx` | u8 | per-(step, layer) call counter (C7: `tp_overlap._MIN_TOKENS=256` means prefill can be 2) |
| `num_tokens` | u16 | real rows, padded rows excluded |
| `num_ids` | u16 | payload length |
| `payload` | `u16[num_ids]` | deduped, **ascending** global expert ids (E=512 fits u16; non-EP so local id == global id, `moe.py:1050-1058`) |

**Why the deduped union and not per-row.** The grouped GEMM reads each distinct expert slab **once
per call** — that is the byte event a cache must model. It also enforces the arXiv 2608.07911
requirement to count one access per `(layer, expert, step)` rather than per `(token, expert)`, which
otherwise inflates recency policies by 27-29%. At decode M=1 the union **is** the row, so decode is
lossless.

**Mapping onto the simulator's event stream.** Each record is exactly one cache event group:

```
for rec in records:                       # already in file order == step order
    if rec.kind != DECODE: ...            # prefill = the pollution arm; keep it separate
    for e in rec.payload:                 # ascending, deduped
        key   = (rec.layer_id, e)         # 48 x 512 = 24,576 distinct cache lines
        bytes = header.expert_bytes       # 1,382,400 B rank-local, uniform
        t     = rec.step_id               # recency clock; NOT wall time
```

The cache line is `(layer, expert)`, not `expert`: the layers are disjoint weight sets, and the
capacity budget (6.7 GiB/rank ÷ 1.3824 MB = **~5,082 slots of 24,576**, ~20.7%) is global across
them, which is precisely the freedom static placement does not have and LRU does. Miss cost is
`expert_bytes / host_BW`; the arm's measured per-layer costs (device-resident 0.114 ms, host-streamed
~0.93 ms, CPU-tier ~1.02 ms) are the calibration targets — a full-miss layer at top_k=10 must
reproduce ~0.93 ms or the sim's bandwidth model is wrong.

**Invariants a reader MUST assert on load** (fail loud, do not repair):

1. `magic == b"MSGLRT01"` and `version == 1`.
2. every `layer_id < num_layers`; every expert id `< num_experts`.
3. at `kind == 1`: `num_tokens == 1` **and** `num_ids == top_k`. `num_ids < top_k` at decode means
   the route returned a duplicate expert id — **a routing bug worth reporting, not silently
   accepting**.
4. `step_id` non-decreasing (records are appended in drain order; ring records precede that drain's
   oversize records within one drain, so sort by `(step_id, layer_id, chunk_idx)` before analysis).
5. payload strictly ascending within a record.
6. `model_hash` matches the sidecar's model slug.
7. `expert_bytes` matches the value the sim's byte model uses (**T0**).

**META SIDECAR** (JSON, hand-written by the harness, **not** by the tracer). *A trace without this
file is not admissible.* Contents: worktree commit sha, kernels sha, image tag,
`MODEL/TP/GRAPH_BS/CONC/MEM_RATIO/PAGE_SIZE`, **the offload split actually booted** (host-streamed
layer list and device-resident layer list, read from `ExpertStreamTier.stats()`), sampler params
(temperature / top_p / top_k / **seed**), the prompt fixture filenames driven, per-request
`completion_tokens`, wall clock, and **the card ids from the lease** (a mismatched pair is a real
failure mode on this box — card 1's root port is Gen4 x8 at half the bandwidth).

---

## 6. Run recipe — BLOCKED, this is the only GPU step in the plan

> **Blocked right now.** Both cards are held by a serve A/B. Everything above and below this section
> is CPU-only. Nothing in this document has been executed against a GPU.

* **Worktree:** `/home/pat/code/minisgl-rdna4-oracle` (branch `task/routing-oracle`), mounted.
  **Never `$PWD`.**
* **Lease:** `gpu-lease -n 2 -- <harness>` — TP=2, and `-n 2` is correct **here and only here**.
* **Config:** `MODEL=qwen4exp`, `GRAPH_BS=0` (the shipped default at `tools/serve.sh:754`), `CONC=1`,
  the arm's shipped `MEM_RATIO` and offload split. **Do NOT set `GRAPH_BS=2`** — under bucketed
  capture the Python in `MoELayer.forward` runs once at capture and is dead at replay, and the bs=2
  bucket's padded row 1 carries garbage routing that would pollute the trace.
* **Drive:** all 12 fixtures in `/home/pat/fixtures/minisgl-real-traffic/requests`
  (`real_{3k,7k,14k,28k}_{0,1,2}.json`) plus ~8 short varied prompts, **one request at a time**.
* **SAMPLED, NEVER GREEDY:** `temperature=0.8`, `top_p=0.95`, fixed seed recorded in the sidecar.
  Greedy on a repeated prompt produces verbatim-identical continuations; arXiv 2608.07911 measures
  that this shifts early-window recency effects by **19.4-31.9 pp** and inflates recency policies by
  **27-29%**. A greedy trace is inadmissible.
* **Volume:** >= 20,000 decode steps across >= 12 distinct prompts and >= 3 length classes. At
  16.65 tok/s that is ~1,200 s of decode plus prefill — one ~25-35 min window. Adequacy: 48x512 =
  24,576 (layer, expert) slots; 10 draws x 48 layers x 20k steps = 200k draws/layer ≈ **390
  observations per expert per layer**.
* **Both ranks this first run.** Routing is expected bit-identical: the gate is `LinearReplicated`
  (`qwen3_5_moe.py:67`), non-EP so `local_num_experts=512` and `local_expert_offset=0` on both ranks
  (`moe.py:1050-1058`), and experts are TP-sharded on the intermediate dim 640→320 — which is exactly
  why the granule is 1,382,400 B rank-local. **Diff the two id streams offline; byte-identical is the
  confirmation** and it costs one extra 35 MB file. Subsequent runs can be rank 0 only.
* **Verify after:** files **not root-owned**; rank0/rank1 id streams byte-identical; header
  invariants pass; distinct `req_uid` >= 12; decode records >= `20000 * 48`; T1 blockmap violations
  == 0.
* **A/B provenance gate:** run ~200 **greedy** tokens with the tracer ON and OFF on the same prompt
  and assert the token ids are **identical**. The tracer is a pure read of an already-computed
  tensor and must not perturb the route — assert it, do not assume it. (Greedy is invalid for the
  *trace*; it is the right tool for this determinism check.)

---

## 7. HOW THIS COULD LIE

Every way this capture could produce a trace that makes the LRU cache look better or worse than it
really is. Each has a detector or an accepted residual.

### Makes LRU look BETTER than it is (the dangerous direction — it argues for building something)

1. **Capturing the Python route instead of the kernel's.** A `torch.softmax().topk()` re-derivation
   disagrees with `moe_route_align` on exact bf16 ties at the k-th boundary — **measured on this
   checkpoint**: experts 324 and 366 both at `-5.09375`, 2 of 8 calls in one forward diverged. A
   trace built from the Python route would show one expert the GEMM never read and miss one it did.
   *Detector:* structural — we record `topk_ids` from `_route_align` itself. Any future refactor that
   moves the record away from `kernels.py:657` reintroduces this.
2. **bs>1 decode collapsed to row 0 (C6).** The spec's `reshape(-1)[:top_k]` at `num_tokens <= 8`
   silently keeps one row's 10 experts and drops the rest of the union. At CONC>1 that under-reports
   bytes and inflates the hit rate. *Detector:* the ring path is gated on `M == 1`, and the reader
   asserts `kind==1 ⇒ num_tokens==1 and num_ids==top_k`.
3. **The aligner's off-route block map (T1).** `stream_tier.route_ids()` unions `expert_ids[:live]`
   because a padding block emitted under a real expert id makes the GEMM dereference a slab that is
   not in `topk_ids` — more bytes read than the trace records. *Detector:* the bounded
   `MINISGL_MOE_ROUTE_TRACE_BLOCKMAP=64` check; a violation prints loudly and sets the analysis
   requirement that bit1 be honoured.
4. **Duplicate `(step, layer, chunk)` keys colliding (C7).** A >=256-row prefill calls a layer twice
   (`tp_overlap._MIN_TOKENS=256`); with `chunk_idx` hardcoded to 0 the analyzer's dedupe drops one
   chunk's experts, shrinking the prefill's apparent working set — which is exactly the pollution
   term that decides whether LRU survives a prefill. *Detector:* tracer-owned per-(step, layer)
   counter, plus a reader assert that `(step_id, layer_id, chunk_idx)` is unique.
5. **Warmup / capture / dummy steps recorded as real traffic.** Boot warmup forwards and DP-dummy
   batches route on synthetic hidden states. Their routing is not traffic, and if they are uniform
   they *dilute* skew (looks worse); if they are degenerate they *concentrate* it (looks better).
   *Detector:* every record carries `req_uid`; the analyzer must **drop `req_uid == 0`** and the
   first N steps before the first real request's uid appears. This must be an explicit filter in the
   simulator, not an assumption.
6. **A single prompt, or one length class, being unrepresentative.** One prompt's continuation can
   sit in a narrow topical basin and route to a small expert subset all run — that *is* skew, but
   skew of a sample, not of the traffic. The reported hit rate would be an artifact of n=1.
   *Detector:* >= 12 distinct prompts across >= 3 length classes, asserted by counting distinct
   `req_uid`; and the analyzer must report the miss curve **per prompt** as well as pooled, with the
   spread. A pooled curve whose per-prompt spread is wide is not a result.
7. **Greedy sampling.** Verbatim-identical continuations on a repeated prompt manufacture reuse the
   real serve would never see; arXiv 2608.07911 measures 19.4-31.9 pp of early-window recency shift
   and 27-29% recency-policy inflation. *Detector:* sidecar records temperature/top_p/seed; a
   sidecar showing `temperature: 0` invalidates the trace.
8. **Per-row instead of per-call accounting.** Counting `(token, expert)` rather than
   `(layer, expert, step)` inflates recency policies by 27-29% (same paper). *Detector:* `flags`
   bit0 asserts the payload is the per-call deduped union; the reader must check it.

### Makes LRU look WORSE than it is

9. **Prefill records mixed into the decode stream.** A 1024-token prefill chunk touches most of the
   512 experts and would flush any LRU. That is a *real* effect and must be modeled — but modeling
   it as if it happened at *decode* cadence over-weights it enormously. *Detector:* `kind` is on
   every record; the simulator must run three arms (decode-only, decode+prefill, decode+prefill with
   prefill non-resident/bypassing the cache) and report all three. Reporting only the pooled arm
   would be the mirror-image error of #6.
10. **Ring wrap losing steps.** `drain_every > ring_steps` would overwrite un-drained slots and
    delete accesses, which breaks reuse distances in the direction of *more* misses. *Detector:*
    constructor raises; the reader asserts `step_id` continuity and reports any gap.
11. **Records dropped under capture.** Every write is guarded on `is_current_stream_capturing()` and
    returns silently. On this arm `GRAPH_BS=0` so capture never happens — but if someone runs with
    capture, the trace acquires holes with no error. *Detector:* the sidecar records `GRAPH_BS`; the
    reader asserts `GRAPH_BS == 0` and that `records == steps * 48`.
12. **The tracer perturbing the route it measures.** 48 extra dispatches/step could in principle
    change stream ordering. *Detector:* the ON/OFF greedy token-id equality A/B in §6. It is a pure
    read of an already-computed tensor, so this should be exact — assert it.

### Makes the *answer* wrong without making the trace wrong

13. **`expert_bytes` mismatch (T0).** `48*512*1382400*2 = 63.28 GiB` vs the brief's 68.42 GiB, an
    8.1% gap. The residency fraction is the x-axis of the miss curve; an 8% error in the denominator
    moves the operating point and can flip a marginal verdict. *Detector:* reconcile before the sim
    runs; the header value and the sim's byte model must be the same number.
14. **The comparison arm is not comparable.** The 27-30 tok/s report is 2x3080-20GB, PCIe 4.0 x8,
    96 GB DDR5, a different model and a different quantization. **Our** card 1 root port trained
    Gen4 x8 (14.48 vs 28.93 GB/s), so a TP=2 host-bandwidth ceiling here is card-1-gated in a way
    theirs may not be. Even a correct trace showing exploitable skew does not transfer their tok/s
    to this box. *Detector:* the simulator must be calibrated against **our** measured per-layer
    costs (0.114 / ~0.93 / ~1.02 ms) and report tok/s in those units, never quote theirs.
15. **The stale ms/step denominator.** The 60.06 ms / 38+10 split may have been superseded by the
    "[ALL-HOST 2026-09-07]" note (commit `cd40a9ab`). It changes nothing the simulator computes, but
    every overhead percentage in §4 is quoted against it. *Detector:* record the booted split from
    `ExpertStreamTier.stats()` in the sidecar and recompute.

---

## 8. Review checklist before this lands

- [ ] C1-C8 above are all reflected in the applied code (they are in the diff; verify no reviewer
      "restores" the spec's version).
- [ ] Import cycle `quant.kernels` → `weights.route_trace` does not bite; if it does, go lazy.
- [ ] `num_layers`/`num_experts`/`top_k` derived from the discovered ops, not from a config field
      that could disagree with what was wrapped.
- [ ] `close()` wired into the shutdown path.
- [ ] `/route` mount added in **both** compose places (`:49` and the `serve` service at `:670`).
- [ ] Fixture dir pre-created `-o 1000 -g 1000`; post-run ownership verified non-root.
- [ ] Reader/invariant checker written and green on a synthetic file **before** the GPU window opens.
- [ ] ON/OFF greedy token-id A/B green.
