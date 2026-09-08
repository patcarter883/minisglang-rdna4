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

import os
from typing import Dict, List, Optional, Tuple

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
        self.stats = {"hits": 0, "misses": 0, "promotions": 0, "evictions": 0, "drains": 0}

    # -- setup -----------------------------------------------------------------------------------
    def register_layer(self, layer_id: int, host_w: torch.Tensor,
                       host_s: torch.Tensor, host_z: Optional[torch.Tensor]) -> None:
        """Bind one layer's HOST-resident expert tensors and allocate its `slot_of` table.

        The table is allocated ONCE and never reallocated, because its ADDRESS is what graph capture
        bakes. Initialised to -1: every expert starts non-resident and reads from the host base, so
        an unpopulated cache is exactly today's behaviour rather than a wrong answer.
        """
        if layer_id in self._layers:
            raise ExpertCacheError(f"layer {layer_id} registered twice")
        if host_w.shape[0] != self.num_experts:
            raise ExpertCacheError(
                f"layer {layer_id}: host weight has {host_w.shape[0]} experts, cache was built for "
                f"{self.num_experts} — a mismatch would index the wrong slab row."
            )
        slot_of = torch.full((self.num_experts,), -1, dtype=torch.int32, device=self.device)
        self._layers[layer_id] = {"w": host_w, "s": host_s, "z": host_z, "slot_of": slot_of}
        if not self._slabs:
            self._alloc_slabs(host_w, host_s, host_z)

    def _alloc_slabs(self, w: torch.Tensor, s: torch.Tensor, z: Optional[torch.Tensor]) -> None:
        """One VRAM slab per tensor kind, `slots` rows each, laid out exactly like the host tensor's
        per-expert row so the kernel's existing `wq_expert(base, idx, ...)` arithmetic is unchanged —
        that is what makes this a (base, index) redirect rather than a new addressing scheme."""
        def slab(t: torch.Tensor) -> torch.Tensor:
            return torch.empty((self.slots, *t.shape[1:]), dtype=t.dtype, device=self.device)
        self._slabs["w"] = slab(w)
        self._slabs["s"] = slab(s)
        self._slabs["z"] = slab(z) if z is not None else None
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
            set_map_op(None, None, None, None)      # unregistered layer -> cache off, host reads
            return
        set_map_op(L["slot_of"], self._slabs["w"], self._slabs["s"], self._slabs["z"])

    # -- the asynchronous manager ----------------------------------------------------------------
    def observe(self, layer_id: int, expert_ids) -> None:
        """Feed the policy references the manager has OBSERVED. Never on the critical path.

        `expert_ids` is a host-side sequence, drained from the route ring N steps after the fact.
        Measured: N=64 costs nothing (h 0.8558 -> 0.8563).
        """
        base = layer_id * self.num_experts
        for e in expert_ids:
            key = base + int(e)
            if key in self._policy:
                self._policy.touch(key)
                self.stats["hits"] += 1
            else:
                self.stats["misses"] += 1
                self._promote(key)

    def _promote(self, key: int) -> None:
        """Bring one expert into VRAM. THE ORDER HERE IS THE CORRECTNESS ARGUMENT.

        retract victim -> copy bytes -> fence -> publish. Never publish before the fence: the kernel
        would read a slot whose copy is still in flight, which is a wrong-numbers bug with no error.
        And never reuse a slot without retracting it first, or an in-flight launch reading the old
        expert would silently get the new one's bytes.
        """
        layer_id, expert = divmod(key, self.num_experts)
        L = self._layers.get(layer_id)
        if L is None:
            return
        victim = self._policy.admit(key)
        cuda = self.device.type == "cuda"
        compute = torch.cuda.current_stream(self.device) if cuda else None
        stream = self._copy_stream
        if victim is not None:
            vlayer, vexpert = divmod(victim, self.num_experts)
            slot = self._slot_of_key.pop(victim)
            # (1) RETRACT on the COMPUTE stream, so it is ordered against the launches that read it.
            self._layers[vlayer]["slot_of"][vexpert] = -1
            self._key_of_slot[slot] = -1
            self.stats["evictions"] += 1
        elif self._free:
            slot = self._free.pop()
        else:
            self._policy.evict_key(key)
            return

        # (2) The copy must not overwrite the slot until the retraction is visible to anything
        #     already queued. Without this a launch still reading the victim would get the new
        #     expert's bytes: right shapes, wrong numbers, no error.
        if cuda:
            stream.wait_stream(compute)
        import contextlib
        with (torch.cuda.stream(stream) if cuda else contextlib.nullcontext()):
            self._slabs["w"][slot].copy_(L["w"][expert], non_blocking=cuda)
            self._slabs["s"][slot].copy_(L["s"][expert], non_blocking=cuda)
            if self._slabs["z"] is not None and L["z"] is not None:
                self._slabs["z"][slot].copy_(L["z"][expert], non_blocking=cuda)
        # (3) FENCE: the publish must not become visible before the bytes land, or the kernel reads
        #     a slot whose copy is still in flight. This is the invariant the whole design rests on.
        if cuda:
            compute.wait_stream(stream)
        L["slot_of"][expert] = slot
        self._slot_of_key[key] = slot
        self._key_of_slot[slot] = key
        self.stats["promotions"] += 1

    def summary(self) -> str:
        tot = self.stats["hits"] + self.stats["misses"]
        h = self.stats["hits"] / tot if tot else 0.0
        return (f"[expert-cache] slots={self.slots} resident={len(self._slot_of_key)} "
                f"observed_h={h:.4f} promotions={self.stats['promotions']} "
                f"evictions={self.stats['evictions']} drains={self.stats['drains']}")
