"""Pinned host-RAM arena for recurrent-state snapshots (the "ladder"), + its async handle.

WHY THIS EXISTS. The recurrent-radix snapshot store is a VRAM reservation subtracted from the KV
pool at boot (`engine.py::_rec_snapshot_store_bytes` -> `engine.py:1003-1011`). On the 35B at
TP=2/CONC=4 that is 20 snapshots x 16.406 MiB = 0.32 GiB, which at 5,120 B per KV token is
**67,200 KV tokens** the pool never gets — and since the context ceiling is `min(max_position_
embeddings, pool)`, i.e. pool-bound, those tokens are also context. Moving the store to pinned
host RAM hands them back.

WHY THE ARITHMETIC WORKS. A snapshot is a fixed 16.4 MiB *regardless of how much context it
represents*, it is touched once at a prefix hit, and it is never on the decode path. So the
comparison is PCIe-vs-recompute, not PCIe-vs-VRAM: ~0.33 ms over PCIe 5.0 x16 against re-running
the recurrence over thousands of tokens.

THREE INVARIANTS, each of which is a silent-corruption bug if broken:

1. **TP determinism.** Free-list membership must be a pure function of the Python-level snapshot
   lifecycle. `alloc`/`release` must NEVER consult `event.query()`. Every TP rank runs the
   identical batch sequence, so the Python capture/drop sequence is identical — but a completion
   query is timing-dependent, so ranks would make *different* drop decisions, `match_prefix` would
   return different `cached_len` per rank, the ranks would build differently-shaped batches, and
   the TP collectives would HANG. Reuse safety therefore comes from a stream `wait_event` on
   alloc, never from a host query on free.
2. **No side-stream op under graph capture.** A side-stream copy cannot be recorded into a
   captured graph (same rule as `layers/tp_overlap.py:235-247`). Callers assert this.
3. **Ring ordering.** `flush_ring` before any D2H of `ssm_state`, `reset_ring` after any H2D into
   it (`gdn_state.py:67-78`). Violating either is silently wrong text, not a crash.

Nothing here ever READS the pinned bytes on the host — the only consumers are the restore H2D (a
device op) and the free list. So there is no host synchronisation in this module at all, and
`wait()` is a stream order, exactly like `AsyncAllReduce.wait` (`tp_overlap.py:214-218`).
"""

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch

from ._envutil import env_int

# DMA-friendly and comfortably above any dtype's alignment requirement, so every component view is
# safe to reinterpret from the uint8 slab.
_ALIGN = 256

_side_stream: "torch.cuda.Stream | None" = None


def host_tier_enabled() -> bool:
    """Whether the recurrent snapshot store lives in pinned host RAM. Default ON.

    Read in ONE place because the ENGINE uses it to size the VRAM reservation and the STATE CACHE
    uses it to decide where a clone goes — the same drift hazard `resolve_prefix_cache` and
    `swa_radix_enabled` were centralised to avoid. If these two disagreed, the pool would be sized
    for a store that does not exist (or vice versa).
    """
    return os.environ.get("MINISGL_REC_SNAP_HOST", "1") != "0"


def stage_ring_depth() -> int:
    """Device staging buffers kept for the host tier. Default 1.

    Depth 0 means "no staging": the D2H would read LIVE conv_state/ssm_state, so the next forward
    must wait on the whole PCIe leg instead of just the ~30 us gather, AND the copy becomes 60
    per-layer calls because `conv_state[lid, s]` is contiguous only per layer. That buys the last
    3,360 KV tokens (2.4% of the pool) for ~0.6 ms per capture on the critical path — a bad trade,
    which is why the default is 1 rather than 0. Depth >= 2 only overlaps gather(N+1) with DMA(N),
    30 us against 330 us, and consecutive captures serialise on the single PCIe link anyway.
    """
    return max(0, env_int("MINISGL_REC_SNAP_STAGE_RING", 1))


def get_snapshot_side_stream() -> "torch.cuda.Stream":
    """The one process-wide side stream carrying snapshot D2H traffic.

    Deliberately NOT `tp_overlap.get_ar_side_stream()`: a 0.33 ms snapshot DMA queued ahead of a TP
    collective on the shared stream would serialise comms behind PCIe. One stream (not one per
    capture) so concurrent captures serialise against each other for free — they contend for the
    same physical link regardless.
    """
    global _side_stream
    if _side_stream is None:
        _side_stream = torch.cuda.Stream()
    return _side_stream


@dataclass(frozen=True)
class FrameComponent:
    name: str
    dtype: torch.dtype
    shape: Tuple[int, ...]

    @property
    def nbytes(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n * self.dtype.itemsize


class FrameLayout:
    """A byte layout shared by the pinned host frame AND the device staging buffer.

    Sharing it is the point: identical geometry on both sides makes a capture/restore ONE
    contiguous `copy_` of the whole frame rather than a per-component (or worse, per-layer) loop.
    """

    __slots__ = ("components", "offsets", "nbytes")

    def __init__(self, components: Sequence[FrameComponent]) -> None:
        self.components: Tuple[FrameComponent, ...] = tuple(components)
        self.offsets: List[int] = []
        off = 0
        for c in self.components:
            self.offsets.append(off)
            off += (c.nbytes + _ALIGN - 1) // _ALIGN * _ALIGN
        self.nbytes = off

    def views(self, flat_u8: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Typed views over a flat uint8 buffer of exactly `nbytes`. Built ONCE per buffer, so a
        capture on the hot path creates no tensors."""
        assert flat_u8.dtype == torch.uint8 and flat_u8.numel() >= self.nbytes
        out: Dict[str, torch.Tensor] = {}
        for c, off in zip(self.components, self.offsets):
            out[c.name] = flat_u8[off : off + c.nbytes].view(c.dtype).view(c.shape)
        return out


class PinnedFrameArena:
    """Fixed-size pinned frames + a FIFO free list.

    One frame size per process: `resolve_prefix_cache` makes a model GDN xor CCA xor SWA, so there
    is never more than one snapshot kind alive and no size classes are needed.

    FIFO, not LIFO (the slot allocators in `gdn_state.py:105` / `cca_state.py:58` are LIFO because
    locality matters there). Here the opposite holds: a frame's last DMA may still be in flight, so
    FIFO gives every frame a full lap around the ring before reuse, which makes the `wait_event` on
    alloc almost always free.
    """

    def __init__(self, layout: FrameLayout, num_frames: int) -> None:
        self.layout = layout
        self.num_frames = int(num_frames)
        self.frame_bytes = layout.nbytes
        # One allocation, not num_frames of them: a single large cudaHostRegister is far cheaper
        # than thousands of small ones, and it keeps the pinned footprint exactly predictable.
        self._slab = torch.empty(self.num_frames * self.frame_bytes, dtype=torch.uint8, pin_memory=True)
        self._flat: List[torch.Tensor] = []
        self._views: List[Dict[str, torch.Tensor]] = []
        for i in range(self.num_frames):
            f = self._slab[i * self.frame_bytes : (i + 1) * self.frame_bytes]
            self._flat.append(f)
            self._views.append(layout.views(f))
        self._free: "deque[int]" = deque(range(self.num_frames))
        self._lastuse: List["torch.cuda.Event | None"] = [None] * self.num_frames
        self.drops = 0
        self.high_water = 0

    # -- capacity -------------------------------------------------------------

    @property
    def in_use(self) -> int:
        return self.num_frames - len(self._free)

    @property
    def nbytes(self) -> int:
        return self.num_frames * self.frame_bytes

    # -- allocation -----------------------------------------------------------

    def alloc(self, write_stream: "torch.cuda.Stream | None" = None) -> int | None:
        """Take a frame, or None when full. NEVER blocks and NEVER queries an event (invariant 1).

        `write_stream` is the stream that will WRITE this frame — it must be the one that waits,
        not merely the current one. The two accesses run on different streams: a capture's D2H is
        issued on the snapshot side stream, while a restore's H2D READS the frame on engine.stream.
        So a frame can be handed out while a previous restore is still reading it, and waiting on
        the wrong stream would let the new D2H overwrite bytes that read is still consuming —
        cross-contaminating an unrelated, already-stored snapshot. `lastuse` therefore tracks the
        last access of EITHER kind (see `HostSnapshot.note_read`), and the writer waits on it.

        Deterministic across ranks, and free in the common case because FIFO gives every frame a
        full lap around the ring before reuse.
        """
        if not self._free:
            self.drops += 1
            return None
        idx = self._free.popleft()
        ev = self._lastuse[idx]
        if ev is not None:
            (write_stream or torch.cuda.current_stream()).wait_event(ev)
            self._lastuse[idx] = None
        self.high_water = max(self.high_water, self.in_use)
        return idx

    def release(self, idx: int) -> None:
        """Return a frame IMMEDIATELY and UNCONDITIONALLY (invariant 1).

        No completion check, no retire queue. A DMA still in flight on this frame is made safe by
        `alloc`'s `wait_event`, not by delaying the free — delaying it here would make free-list
        membership timing-dependent and desync the TP ranks.
        """
        if 0 <= idx < self.num_frames:
            self._free.append(idx)

    def set_lastuse(self, idx: int, event: "torch.cuda.Event") -> None:
        self._lastuse[idx] = event

    def views(self, idx: int) -> Dict[str, torch.Tensor]:
        return self._views[idx]

    def flat(self, idx: int) -> torch.Tensor:
        return self._flat[idx]

    def stats(self) -> str:
        return (
            f"frames={self.num_frames} in_use={self.in_use} high_water={self.high_water} "
            f"drops={self.drops} pinned={self.nbytes / (1 << 30):.2f} GiB"
        )


class HostSnapshot:
    """A snapshot whose bytes live in a pinned host frame; its D2H may still be in flight.

    Modelled on `AsyncAllReduce` (`tp_overlap.py:200-223`): `wait()` is idempotent and orders the
    CURRENT STREAM after the capture, never blocking the host.

    Release is by REFCOUNT (`__del__`), and that is the entire reason `radix_cache.py` needs no
    changes at all. The store drops snapshots at five independent sites — `_enforce_rec_cap`
    (`radix_cache.py:277`), evict (`:331`), `_free_req_resources` (`scheduler.py:1459-1460`), the
    ladder FIFO trim (`:1510`), and node replacement — and every one of them is a plain
    `x = None`. Refcounting means all five keep working untouched, and the `rec_state: Any`
    contract at `radix_cache.py:45` stays honest. An explicit `release()` at each site was
    rejected: one missed site leaks a frame silently, and the store would stop being opaque.
    """

    __slots__ = ("_arena", "_idx", "views", "flat", "_event", "_done", "meta")

    def __init__(self, arena: PinnedFrameArena, idx: int, event=None, meta=None) -> None:
        self._arena = arena
        self._idx = idx
        self.views = arena.views(idx)
        self.flat = arena.flat(idx)
        self._event = event
        self._done = event is None
        self.meta = meta

    def wait(self) -> "HostSnapshot":
        if not self._done:
            torch.cuda.current_stream().wait_event(self._event)
            self._done = True
        return self

    def set_event(self, event) -> None:
        self._event = event
        self._done = False

    def note_read(self, event) -> None:
        """Record that something READ this frame (a restore's H2D), so the next writer of the frame
        waits for that read to retire. Without this, a snapshot dropped mid-restore could have its
        frame recycled and overwritten by an incoming capture while the outbound H2D is still in
        flight — and because restores are non-destructive, nothing else would ever notice."""
        self._arena.set_lastuse(self._idx, event)

    def __del__(self) -> None:
        # Interpreter shutdown can null out module globals before the last snapshot dies, so this
        # must not raise — a failed release here would surface as an unrelated ignored exception.
        try:
            self._arena.release(self._idx)
        except Exception:
            pass
