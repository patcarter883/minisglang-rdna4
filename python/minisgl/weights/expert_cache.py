"""Per-expert VRAM residency cache — the host half of the `slot_of[]` indirection.

WHAT THIS IS. The MoE decode GEMV reads 10 routed experts per layer per token, and on this box a
host-streamed layer costs ~0.93 ms against ~0.114 ms device-resident, gated by card 1's permanently
Gen4 x8 link. The shipped placement is LAYER-granular: whole layers pinned in VRAM until the budget
runs out, which at 6.7 GiB/rank covers 21.2% of the expert set and buys h = 0.208 by construction.
Measured on a real 20,004-step route trace (`tools/offload/EXPERT_CACHE_DECISION_2026-09-08.md`),
an SLRU cache over the SAME bytes reaches **h = 0.864**, i.e. 16.65 -> 29.10 tok/s.

WHY A RECENCY POLICY AND NOT A HOT-LIST. The checkpoint carries `router_aux_loss_coef: 0.001` and it
works: per-layer Gini is only 0.61 and the marginals are near-flat, so the DISTRIBUTION-based
policies fail — static-prior reaches 0.556 and LFU only 0.360. But an aux loss constrains marginals,
not the AUTOCORRELATION of the route sequence, and `frac(stack distance <= cache size) = 0.8404` IS
the LRU hit condition. Routing here is near-uniform AND highly cacheable; those are different
properties and the project's prior analysis conflated them.

THE MANAGER IS ASYNCHRONOUS, AND THAT IS MEASURED, NOT ASSUMED. Deciding residency from a
synchronous `topk_ids` read would cost a D2H per MoE layer per step (~0.13 ms x 48 = ~6 ms/step,
eating a third of the win). Simulated with `expert_cache_oracle.py --lag-steps`, a manager that
observes references 64 steps LATE loses nothing: h 0.8558 -> 0.8563, because the per-layer stack
distance is P50 29.9 / P90 138.4 and the hot set is stable over hundreds of steps. So this drains
the route ring in the background and NEVER synchronises the forward.

THE ONE CORRECTNESS INVARIANT, which the whole design rests on:

    slot_of[e] >= 0  =>  slot `slot_of[e]` of the VRAM slabs HOLDS EXPERT e's bytes
    slot_of[e] <  0  =>  read expert e from the host/pinned base

A stale table is safe in exactly ONE direction. It may under-report residency (a host read: correct,
just slower). It must NEVER over-report, because the kernel would then read another expert's bytes
and produce plausible wrong numbers with no error anywhere. Every write below therefore lands in
this order, and `promote()` documents it inline:

    copy bytes into slot  ->  fence the copy  ->  publish slot_of[e] = slot

and eviction is the reverse: retract `slot_of[e] = -1` and fence BEFORE the slot is handed out
again. A slot is never reused while a launch that may still read it is in flight.

GRAPH CAPTURE. The slabs and the per-layer `slot_of` tensors are allocated ONCE, so their addresses
are stable for the life of the serve; capture bakes those pointers and the manager mutates the table
contents underneath. That is why this needs no VMM (which is dead on gfx1201 anyway — a remap at a
live VA serves the stale page and returns hipSuccess) and no re-capture on promotion.
"""

from __future__ import annotations

import collections
import os
import threading
from typing import Any, Deque, Dict, List, Optional, Tuple

import torch

from .cpu_tier import CpuTierError


def _env_int(name: str, default: int) -> int:
    """Empty-safe. compose declares every knob `FOO: "${FOO:-}"`, so unset arrives SET-BUT-EMPTY
    and a bare int() raises at import — the documented failure mode in this tree."""
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


class ExpertCacheError(RuntimeError):
    pass


class _SLRU:
    """Segmented LRU: a PROBATION segment that new experts enter, and a PROTECTED segment they are
    promoted into on a second hit. Measured best on the real trace (h 0.8644 vs plain LRU 0.8558).

    The reason it wins here is prefill: a prefill chunk touches nearly every expert in a layer once,
    and plain LRU lets that single sweep evict the decode working set. Probation absorbs it — a
    one-touch expert never reaches protected, so it can only displace other one-touch entries. The
    oracle measured the prefill-pollution term at -0.0008 for SLRU against +0.028 for LFU.
    """

    __slots__ = ("cap", "prot_cap", "probation", "protected")

    def __init__(self, cap: int, protected_frac: float = 0.8):
        self.cap = cap
        self.prot_cap = int(cap * protected_frac)
        self.probation: Dict[int, None] = {}
        self.protected: Dict[int, None] = {}

    def __contains__(self, key: int) -> bool:
        return key in self.probation or key in self.protected

    def touch(self, key: int) -> None:
        """Record a hit on a resident key: probation -> protected, protected -> most-recent."""
        if key in self.protected:
            self.protected.pop(key)
            self.protected[key] = None
            return
        if key in self.probation:
            self.probation.pop(key)
            self.protected[key] = None
            if len(self.protected) > self.prot_cap:          # demote, never drop
                old, _ = next(iter(self.protected.items()))
                self.protected.pop(old)
                self.probation[old] = None

    def admit(self, key: int) -> Optional[int]:
        """Insert a MISS. Returns the key evicted to make room, or None."""
        victim = None
        if len(self.probation) + len(self.protected) >= self.cap:
            if self.probation:
                victim, _ = next(iter(self.probation.items()))
                self.probation.pop(victim)
            else:
                victim, _ = next(iter(self.protected.items()))
                self.protected.pop(victim)
        self.probation[key] = None
        return victim

    def take_victim(self) -> Optional[int]:
        """Pop the least-valuable resident key WITHOUT admitting anything.

        `admit()` couples "make room" to "insert this key", which is wrong for a cache whose slots
        come back asynchronously: the manager needs to free capacity AHEAD of demand, then place
        keys into it as references arrive. Probation first, exactly as `admit` chooses, so the
        segmented behaviour the trace was scored on is unchanged.
        """
        if self.probation:
            victim, _ = next(iter(self.probation.items()))
            self.probation.pop(victim)
            return victim
        if self.protected:
            victim, _ = next(iter(self.protected.items()))
            self.protected.pop(victim)
            return victim
        return None

    def __len__(self) -> int:
        return len(self.probation) + len(self.protected)

    def evict_key(self, key: int) -> None:
        self.probation.pop(key, None)
        self.protected.pop(key, None)


class ExpertResidencyCache:
    """One rank's per-expert VRAM cache across all offloaded MoE layers.

    `layers` maps layer_id -> (host_w, host_s, host_z) — the pinned-arena tensors the kernel reads
    today. The cache allocates VRAM slabs of `slots` entries and hands the kernel a per-layer
    `slot_of` table via `moe_hip`/`fp8_wmma`'s `set_expert_slot_map`.
    """

    def __init__(self, *, num_experts: int, expert_bytes: int, budget_bytes: int,
                 device: torch.device, protected_frac: float = 0.8) -> None:
        self.num_experts = int(num_experts)
        self.expert_bytes = int(expert_bytes)
        self.device = device
        self.slots = int(budget_bytes // max(1, expert_bytes))
        if self.slots < 1:
            raise ExpertCacheError(
                f"expert cache budget {budget_bytes} B holds no whole expert "
                f"({expert_bytes} B each). Configure a larger budget or disable the cache."
            )
        self._policy = _SLRU(self.slots, protected_frac)
        # key -> slot, and the inverse. Keys are GLOBAL: layer * num_experts + expert, so one pool
        # spans every layer. The oracle measured `--pool global` against `--pool per-layer`; a
        # global pool lets a hot layer borrow slots from a cold one, which per-layer cannot.
        self._slot_of_key: Dict[int, int] = {}
        self._key_of_slot: List[int] = [-1] * self.slots
        self._free: List[int] = list(range(self.slots))
        self._layers: Dict[int, Dict[str, torch.Tensor]] = {}
        self._slabs: Dict[str, torch.Tensor] = {}
        self._copy_stream: Optional[torch.cuda.Stream] = None
        self.stats = {"hits": 0, "misses": 0, "promotions": 0, "evictions": 0, "drains": 0,
                      "dropped_refs": 0, "manager_batches": 0}

        # ---- THE ASYNCHRONOUS MANAGER -----------------------------------------------------------
        # A promotion is a 1.4 MB H2D copy and a cold fill is thousands of them. Run on the
        # scheduler thread they land INSIDE the forward: measured 2026-09-08, fwd_launch went to
        # 5356 ms/step (92% of a 5800 ms step) against a 68.5 ms baseline. So the expensive half
        # runs here, on its own thread, and only two cheap things stay on the scheduler.
        #
        # THE HANDSHAKE, and why it is two-phase. PyTorch's "current stream" is PER THREAD, and
        # writing a device tensor enqueues on the CALLER's stream — so a manager thread that wrote
        # `slot_of` directly would order that write against nothing the compute stream can see. The
        # split that fixes it:
        #
        #   manager thread : policy, and the COPIES (on `_copy_stream`), each tailed by an event
        #   scheduler thread (`apply_pending`, once per step, at the step boundary):
        #       (a) RETRACT victims  -> `slot_of[v] = -1` on the compute stream, event recorded
        #       (b) PUBLISH arrivals -> `slot_of[e] = slot`, but ONLY for copies whose event has
        #           already completed (`query()`, never `synchronize()`)
        #
        # A slot goes manager -> retract-queue -> (scheduler retracts, records event) -> free-list
        # -> manager copies (after waiting that event) -> publish-queue -> (scheduler publishes).
        # The slot is therefore never written while a launch that may still read it is in flight,
        # and a publish is never visible before its bytes land. Both directions of the one
        # invariant, preserved across two threads.
        self._q: Deque = collections.deque(maxlen=max(64, _env_int("MINISGL_EXPERT_CACHE_QUEUE", 4096)))
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._stopping = False
        #: (victim_key, slot) the manager wants retired; the scheduler retracts them.
        self._to_retract: List[Tuple[int, int]] = []
        #: slot -> retraction event; the manager must wait it before reusing the slot.
        self._retract_ev: Dict[int, Any] = {}
        #: (key, slot, event) copies in flight; the scheduler publishes those that have landed.
        self._inflight: List[Tuple[int, int, Any]] = []
        self._compute_stream = None
        #: Manager batches between summary lines. 0 disables.
        self._report_every = _env_int("MINISGL_EXPERT_CACHE_REPORT", 200)
        #: Free slots the manager tries to keep ready. ~0.5% of the pool: enough to absorb a
        #: scheduler round trip at decode rates, small enough that the residency given up is noise.
        self._low_water = max(8, _env_int("MINISGL_EXPERT_CACHE_LOW_WATER", self.slots // 200))
        self._refill_batch = max(8, self._low_water)
        #: OUTSTANDING COPIES CEILING. Unbounded, the manager queues the whole cold fill onto the
        #: copy stream at once — measured 2026-09-08: 4,418 in-flight x 1.36 MiB = 6.2 GB, which
        #: saturates the same card-1-gated PCIe link the forward streams its OWN experts over. The
        #: scheduler then stalls (ticks froze at 66), so nothing is published, so no slot is freed,
        #: so the cache jams with `free=0` and a climbing `deferred`. A cache that starves the path
        #: it is accelerating is worse than no cache. 64 x 1.36 MiB = ~87 MiB of PCIe in flight.
        self._max_inflight = max(8, _env_int("MINISGL_EXPERT_CACHE_MAX_INFLIGHT", 64))
        #: apply_pending() calls. In the summary because a stalled scheduler half and a stalled
        #: manager look identical from the outside, and this separates them.
        self.stats["ticks"] = 0

    # -- setup -----------------------------------------------------------------------------------
    def register_layer(self, layer_id: int, gate_up: "tuple", down: "tuple") -> None:
        """Bind one layer's HOST-resident expert tensors and allocate its `slot_of` table.

        The table is allocated ONCE and never reallocated, because its ADDRESS is what graph capture
        bakes. Initialised to -1: every expert starts non-resident and reads from the host base, so
        an unpopulated cache is exactly today's behaviour rather than a wrong answer.
        """
        if layer_id in self._layers:
            raise ExpertCacheError(f"layer {layer_id} registered twice")
        for name, (w, s_, _z) in (("gate_up", gate_up), ("down", down)):
            if w.shape[0] != self.num_experts:
                raise ExpertCacheError(
                    f"layer {layer_id} {name}: host weight has {w.shape[0]} experts, cache was "
                    f"built for {self.num_experts} — a mismatch would index the wrong slab row."
                )
        slot_of = torch.full((self.num_experts,), -1, dtype=torch.int32, device=self.device)
        self._layers[layer_id] = {"gate_up": gate_up, "down": down, "slot_of": slot_of}
        if not self._slabs:
            self._alloc_slabs(gate_up, down)

    def _alloc_slabs(self, gate_up: "tuple", down: "tuple") -> None:
        """One VRAM slab per tensor kind, `slots` rows each, laid out exactly like the host tensor's
        per-expert row so the kernel's existing `wq_expert(base, idx, ...)` arithmetic is unchanged —
        that is what makes this a (base, index) redirect rather than a new addressing scheme."""
        def slab(t):
            return (torch.empty((self.slots, *t.shape[1:]), dtype=t.dtype, device=self.device)
                    if t is not None else None)
        # TWO PLANES, ONE SLOT INDEX. gemm1_silu reads gate_up, the scatter/down GEMV reads down;
        # an expert is promoted as ONE granule spanning both, so "expert e is resident" stays a
        # single fact. A half-resident expert would read one tensor from VRAM and the other from
        # host at the same slot number — wrong bytes, no error.
        for plane, (w, sc, z) in (("gate_up", gate_up), ("down", down)):
            self._slabs[f"{plane}_w"] = slab(w)
            self._slabs[f"{plane}_s"] = slab(sc)
            self._slabs[f"{plane}_z"] = slab(z)
        # A DEDICATED copy stream: promotions must not serialise behind the forward, which is the
        # entire point of the asynchronous manager. On CPU (the selftest) there are no streams and
        # the ordering is trivially sequential — the TABLE and SLAB mutations, which is what the
        # invariant test checks, are identical either way.
        self._copy_stream = (torch.cuda.Stream(device=self.device)
                             if self.device.type == "cuda" else None)

    # -- the hot-path binding --------------------------------------------------------------------
    def install(self, layer_id: int, set_map_op) -> None:
        """Publish this layer's map for the NEXT MoE launch. Called immediately before the op.

        Cheap by construction: four already-materialised tensors handed to a setter that stores raw
        pointers host-side. No sync, no allocation, no device work.
        """
        L = self._layers.get(layer_id)
        if L is None:
            # Unregistered layer -> cache OFF for it: the kernel reads the host base exactly as
            # today. A layer the cache does not manage must never see a stale map from the
            # PREVIOUS layer's install, which is why this clears rather than returning early.
            set_map_op(None, None, None, None, None, None, None)
            return
        S = self._slabs
        set_map_op(L["slot_of"],
                   S["gate_up_w"], S["gate_up_s"], S["gate_up_z"],
                   S["down_w"], S["down_s"], S["down_z"])

    # -- the asynchronous manager ----------------------------------------------------------------
    def observe(self, layer_id: int, expert_ids) -> None:
        """SCHEDULER THREAD. Hand the reference to the manager and return — nothing else.

        This is called from `route_trace.drain()`, which runs in `begin_forward` on the scheduler
        thread, so it must not touch the policy, allocate, or issue a copy. It appends to a bounded
        deque and returns.

        THE QUEUE DROPS RATHER THAN BLOCKS, and that is the safe direction: a dropped reference
        makes the policy staler, and a stale table only ever under-reports residency (a host read —
        correct, merely slower). Blocking the scheduler to keep the policy perfectly informed would
        trade the one thing this design exists to protect for the one thing it can afford to lose.
        """
        if self._thread is None:
            # No manager running (CPU selftest, or start() not called): do it inline. Same
            # semantics, and it is what every unit test in expert_cache_test.py exercises.
            self._observe_now(layer_id, expert_ids)
            return
        ids = tuple(int(e) for e in expert_ids)
        with self._lock:
            if len(self._q) == self._q.maxlen:
                self.stats["dropped_refs"] += len(ids)
            self._q.append((int(layer_id), ids))
        self._wake.set()

    def _observe_now(self, layer_id: int, expert_ids) -> None:
        """The policy update itself. MANAGER THREAD (or inline when there is no manager)."""
        base = layer_id * self.num_experts
        for e in expert_ids:
            key = base + int(e)
            if key in self._policy:
                self._policy.touch(key)
                self.stats["hits"] += 1
            else:
                self.stats["misses"] += 1
                self._promote(key)

    # -- thread lifecycle ------------------------------------------------------------------------
    def start(self) -> None:
        """Start the manager. MUST be called from the SCHEDULER THREAD.

        The compute stream is captured HERE, from the calling thread, because
        `torch.cuda.current_stream()` is per-thread: read on the manager it would return the
        manager's own stream and every fence would order against nothing.
        """
        if self._thread is not None or self.device.type != "cuda":
            return
        self._compute_stream = torch.cuda.current_stream(self.device)
        self._thread = threading.Thread(target=self._run, name="expert-cache-mgr", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stopping = True
        self._wake.set()
        t, self._thread = self._thread, None
        if t is not None:
            t.join(timeout=5.0)

    def _run(self) -> None:
        torch.cuda.set_device(self.device)          # a fresh thread has no current device
        # INFERENCE MODE IS PER-THREAD AND THE WEIGHTS ARE INFERENCE TENSORS. The scheduler loop is
        # wrapped in `@torch.inference_mode()`, so every tensor the slabs are filled from (and the
        # slabs themselves, allocated inside it) carries the inference flag; an in-place `copy_`
        # from a thread that is NOT in inference mode raises "Inplace update to inference tensor
        # outside InferenceMode is not allowed". Measured 2026-09-08: the manager died on its very
        # first promotion, the cache degraded to host reads exactly as designed, and the arm
        # therefore measured 69.18 ms against a 68.5 ms baseline — i.e. correct, silent, and
        # useless. The failure was visible ONLY in the manager's own error line.
        with torch.inference_mode():
            self._run_loop()

    def _run_loop(self) -> None:
        while not self._stopping:
            self._wake.wait(timeout=0.25)
            self._wake.clear()
            while True:
                with self._lock:
                    if not self._q:
                        break
                    layer_id, ids = self._q.popleft()
                try:
                    self._observe_now(layer_id, ids)
                except Exception as e:                # never take the serve down from here
                    print(f"[expert-cache] manager error (cache degrades to host reads): {e!r}",
                          flush=True)
                    return
                self.stats["manager_batches"] += 1
                # TOP UP THE FREE POOL. A slot needs a scheduler round trip to come back, so
                # reclaiming only when a miss needs one makes every miss pay that latency — which
                # is exactly how replacement froze at 8 evictions in 240k references. Kept small:
                # a reclaimed slot is a resident expert given up, so over-reclaiming lowers the hit
                # rate for nothing.
                with self._lock:
                    short = self._low_water - len(self._free) - len(self._to_retract)
                if short > 0:
                    self._reclaim(min(short, self._refill_batch))
                # THE INSTRUMENT. Without a hit rate a flat A/B is uninterpretable: "the cache does
                # not help" and "the cache never warmed" produce the same TPOT, and this project has
                # already spent three runs on the second one wearing the first one's face. Printed
                # from the manager thread, so it also proves the thread is ALIVE — a dead manager
                # degrades to host reads silently and the line simply stops.
                if self._report_every and self.stats["manager_batches"] % self._report_every == 0:
                    print(self.summary(), flush=True)

    # -- the scheduler-thread half of the handshake ----------------------------------------------
    def apply_pending(self) -> None:
        """SCHEDULER THREAD, once per step. Retract victims, publish landed copies. Cheap.

        Both operations write `slot_of`, and they are here rather than on the manager precisely
        because this thread owns the compute stream the kernels read it from.

        `event.query()` and NEVER `synchronize()`: a copy that has not landed is simply published
        next step. Waiting here would put the PCIe transfer back on the critical path, which is the
        whole defect this thread split exists to remove.
        """
        if self.device.type != "cuda":
            return
        self.stats["ticks"] += 1
        with self._lock:
            retract, self._to_retract = self._to_retract, []
            inflight, self._inflight = self._inflight, []
        for _victim, slot in retract:
            ev = torch.cuda.Event()
            ev.record(self._compute_stream)         # ordered after every launch already queued
            self._retract_ev[slot] = ev
            with self._lock:
                self._free.append(slot)
        still = []
        for key, slot, ev in inflight:
            if not ev.query():
                still.append((key, slot, ev))
                continue
            layer_id, expert = divmod(key, self.num_experts)
            L = self._layers.get(layer_id)
            if L is None:
                continue
            if self._slot_of_key.get(key) != slot:
                # The manager already evicted this key and reassigned the slot while the copy was
                # in flight. Publishing now would point `expert` at another expert's bytes — the
                # exact wrong-numbers bug this design exists to prevent. Drop it; the expert simply
                # reads the host base until it is promoted again.
                self.stats["stale_publishes_dropped"] = \
                    self.stats.get("stale_publishes_dropped", 0) + 1
                continue
            L["slot_of"][expert] = slot             # compute stream: this thread owns it
            self.stats["promotions"] += 1
        if still:
            with self._lock:
                self._inflight.extend(still)

    def _promote(self, key: int) -> None:
        """MANAGER THREAD. Place one expert into a slot we ALREADY HAVE. Inline when unthreaded.

        THE ORDER IS THE CORRECTNESS ARGUMENT, and it spans two threads:
          retract victim (scheduler) -> fence -> copy (here) -> fence -> publish (scheduler).
        Never publish before the copy's fence, and never write a slot before its retraction is
        visible. Both are wrong-numbers bugs with no error.

        WHAT THIS DELIBERATELY NO LONGER DOES: call `admit()` to "make room" and then drop the key.
        That coupling stalled replacement dead — on a full cache every miss evicted a victim and
        discarded its OWN reference, so a promotion needed two misses AND a scheduler tick between
        them. Measured 2026-09-08: 240,000 references produced 5,161 promotions and 8 evictions,
        i.e. the cache filled once and then froze (observed_h 0.51 against the oracle's 0.864).
        Capacity is now freed AHEAD of demand by `_reclaim`, and this function only places.
        """
        layer_id, expert = divmod(key, self.num_experts)
        L = self._layers.get(layer_id)
        if L is None:
            return
        cuda = self.device.type == "cuda"
        threaded = self._thread is not None and cuda

        with self._lock:
            # RATE LIMIT FIRST. Backpressure belongs before the slot is taken: taking one and then
            # refusing to copy would strand it out of the pool.
            if len(self._inflight) >= self._max_inflight:
                self.stats["throttled"] = self.stats.get("throttled", 0) + 1
                return
            slot = self._free.pop() if self._free else None
        if slot is None:
            if not threaded:
                # UNTHREADED (selftests): keep the original synchronous behaviour — evict inline.
                victim = self._policy.take_victim()
                if victim is None:
                    return
                vslot = self._slot_of_key.pop(victim, None)
                if vslot is None:
                    return
                vlayer, vexpert = divmod(victim, self.num_experts)
                self._layers[vlayer]["slot_of"][vexpert] = -1
                self._key_of_slot[vslot] = -1
                self.stats["evictions"] += 1
                slot = vslot
            else:
                # No slot yet. Do NOT touch the policy: the reference is simply not placed this
                # time, and `_reclaim` will have capacity ready shortly. Counting it keeps the
                # stall visible instead of silent.
                self.stats["deferred"] = self.stats.get("deferred", 0) + 1
                return

        # We hold a slot, so this cannot evict — but handle it rather than assume it.
        victim = self._policy.admit(key)
        if victim is not None:
            self._queue_retract(victim)

        compute = self._compute_stream if threaded else (
            torch.cuda.current_stream(self.device) if cuda else None)
        stream = self._copy_stream
        if cuda:
            ev = self._retract_ev.pop(slot, None)
            if ev is not None:
                stream.wait_event(ev)      # this slot's own retraction, not the whole stream
            else:
                stream.wait_stream(compute)
        import contextlib
        with (torch.cuda.stream(stream) if cuda else contextlib.nullcontext()):
            for plane in ("gate_up", "down"):
                w, sc, z = L[plane]
                self._slabs[f"{plane}_w"][slot].copy_(w[expert], non_blocking=cuda)
                self._slabs[f"{plane}_s"][slot].copy_(sc[expert], non_blocking=cuda)
                if self._slabs[f"{plane}_z"] is not None and z is not None:
                    self._slabs[f"{plane}_z"][slot].copy_(z[expert], non_blocking=cuda)

        if threaded:
            done = torch.cuda.Event()
            done.record(stream)
            self._slot_of_key[key] = slot          # host claim now; device publish after the fence
            self._key_of_slot[slot] = key
            with self._lock:
                self._inflight.append((key, slot, done))
            return
        if cuda:
            compute.wait_stream(stream)
        L["slot_of"][expert] = slot
        self._slot_of_key[key] = slot
        self._key_of_slot[slot] = key
        self.stats["promotions"] += 1

    def _queue_retract(self, victim: int) -> bool:
        """Hand one victim's slot to the scheduler for retraction. MANAGER THREAD."""
        slot = self._slot_of_key.pop(victim, None)
        if slot is None:
            return False
        self._key_of_slot[slot] = -1
        self.stats["evictions"] += 1
        with self._lock:
            self._to_retract.append((victim, slot))
        return True

    def _reclaim(self, want: int) -> None:
        """Free capacity AHEAD of demand. MANAGER THREAD.

        The whole point of the split: a slot takes a scheduler round trip to come back, so if
        reclamation only starts when a miss needs a slot, every miss pays that latency and the
        cache stops adapting. Keeping a small pool of slots in flight makes replacement continuous
        — the policy decides WHO leaves, this decides WHEN, and they are different questions.
        """
        for _ in range(want):
            victim = self._policy.take_victim()
            if victim is None:
                return
            if not self._queue_retract(victim):
                continue

    def summary(self) -> str:
        tot = self.stats["hits"] + self.stats["misses"]
        h = self.stats["hits"] / tot if tot else 0.0
        return (f"[expert-cache] slots={self.slots} resident={len(self._slot_of_key)} "
                f"fill={len(self._slot_of_key) / max(1, self.slots):.3f} "
                f"observed_h={h:.4f} refs={tot} promotions={self.stats['promotions']} "
                f"evictions={self.stats['evictions']} "
                f"inflight={len(self._inflight)} free={len(self._free)} "
                f"ticks={self.stats['ticks']} deferred={self.stats.get('deferred', 0)} "
                f"throttled={self.stats.get('throttled', 0)} "
                f"dropped_refs={self.stats['dropped_refs']} "
                f"stale_pub={self.stats.get('stale_publishes_dropped', 0)}")
