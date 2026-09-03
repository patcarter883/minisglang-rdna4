"""Pure-integer chunk / bump arithmetic for the pinned weight arena.

**NO torch, NO HIP, NO /proc reads.** Everything in this file is an integer function of its
arguments, which is why it carries the unit tests that run without a GPU (and, on this box, without
even an importable torch).

WHY A CHUNK TABLE AT ALL. The host tier is built from `hipHostMalloc(...Mapped)` allocations, and
P3b committed 34 GiB per rank as **17 × 2 GiB** chunks (median 4.88 GB/s of pinning, p25/p75
4.31/5.35, min 3.33, max 5.53 — `p3b.json`, arm `hipHostMalloc` ranks=1). Each chunk is an
INDEPENDENT mapping: `hipHostGetDevicePointer` is called per chunk and the resulting device
addresses are **not contiguous with each other**. So:

    >>> A REGION MAY NEVER STRADDLE A CHUNK BOUNDARY. <<<

A straddling region would read the right bytes for its first half and an unrelated chunk's bytes
for its second — plausible weights, no crash, exactly the silent-corruption class Phase 0 kept
running into. `BumpAllocator` therefore does **next-fit**: when a request does not fit in the tail
of the current chunk it abandons that tail and starts the next chunk. The abandoned bytes are
accounted (`abandoned_bytes`), never hidden.

WHY FORWARD-ONLY. P5b measured that torch's free callback fires **zero** times for a live pool
block *and* zero times for a cached/dropped block plus a subsequent `empty_cache()`. Memory is
therefore never handed back, so a free list would be dead code that could only ever be wrong. The
allocator is a forward-only bump pointer and sizing must be right at construction — the same shape
P5b validated, and the same shape `kvcache/host_arena.py` uses for its slab (one allocation, views
attached to it), minus the FIFO free list that module needs and this one provably does not.

WHY THE ORDER MATTERS AND MUST BE RANK-IDENTICAL. Placement is a pure function of
`(ordered request list, chunk_bytes, align)`. Every TP rank must derive the identical plan or the
ranks disagree about where a granule lives; `ChunkPlan.digest()` exists so a caller can prove
equality across ranks (log it, or compare it on the CPU group) instead of assuming it. Nothing in
this file consults a clock, an environment variable, `MemAvailable`, or a HIP return code — the
three timing-dependent inputs that would make two ranks plan differently.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import List, Sequence, Tuple

KIB = 1 << 10
MIB = 1 << 20
GIB = 1 << 30

# 512 B, not `kvcache/host_arena.py`'s 256: it is DMA-friendly, comfortably above every torch dtype's
# alignment requirement (so any component view is safe to reinterpret), AND it is the alignment P5b's
# validated allocator callback used when serving `torch.empty()` from the arena bump pointer. One
# constant for both carving paths so a granule and a torch tensor can never disagree about padding.
ALIGN = 512

# Chunk sizes are rounded to this so a chunk is huge-page friendly and `hipHostMalloc` never sees a
# ragged size.
CHUNK_GRANULE = 2 * MIB

# THE MEASURED CHUNK SIZE. `p3b.json` committed 34.0 GiB/rank and 62.0 GiB across two ranks with
# `chunk_bytes = 2147483648`. 4 GiB is *not* a demonstrated size on this box — nothing above 2 GiB has
# ever been pinned here, and P3's file-backed 8 GiB / 2 GiB-chunk attempt spun >711 s of user CPU with
# no progress. Do not raise this default without re-running P3b at the new size.
DEFAULT_CHUNK_BYTES = 2 * GIB
MIN_CHUNK_BYTES = CHUNK_GRANULE

# torch's caching allocator does NOT ask the arena for a tensor's byte count. `MemPool` allocations
# go through `CUDACachingAllocator::malloc`, which rounds the request (`round_size`, 512 B) and then
# asks the backing allocator for a whole SEGMENT sized by `get_allocation_size`: `kSmallBuffer`
# (2 MiB) for requests <= 1 MiB, `kLargeBuffer` (20 MiB) for requests < 10 MiB, and
# `round_up(size, kRoundLarge = 2 MiB)` above that. So a planned row of `n` bytes costs the arena at
# least `round_up(max(n, 2 MiB), 2 MiB)`, and a reservation enumerated in tensor bytes is short by
# the rounding.
TORCH_ALLOC_GRANULE = 2 * MIB


def torch_allocation_bytes(nbytes: int) -> int:
    """Arena bytes ONE torch `MemPool` allocation of `nbytes` consumes. See `TORCH_ALLOC_GRANULE`.

    MODELLED, NEVER MEASURED — this is the single largest assumption in the exact-packing
    reservation and a GPU run must replace it (compare `ArenaMemPool.stats()`'s served sizes with
    `OffloadPlan.host_row_requests()`). Two deliberate choices:

      * `round_up(max(n, 2 MiB), 2 MiB)` mirrors `kRoundLarge`/`kSmallBuffer` exactly for every row
        this feature actually reserves (a component's stacked slab is tens to hundreds of MiB).
      * the `kLargeBuffer` bucket — a 6 MiB row costing a 20 MiB segment — is deliberately NOT
        modelled. torch SPLITS a large block, so a run of such rows shares one buffer, and charging
        every one of them 20 MiB over-reserves ~3x on a small-row shape. Over-reserving is not the
        polite failure here: it is charged against the live `MemAvailable` gate and refuses boots
        that would have fitted. The residual exposure is bounded by
        `sum(kLargeBuffer - modelled)` over rows in the (1 MiB, 10 MiB) band and, on the target
        shape, is absorbed by the tail next-fit already abandons in each chunk; a shape where it is
        not absorbed fails LOUDLY at the carve (`allocate_raw` refuses, `mark_populated` raises)
        rather than silently in VRAM, which is why the optimistic direction is affordable here and
        was not affordable for `headroom_chunks`.
    """
    n = int(nbytes)
    if n <= 0:
        return 0
    return round_up(max(n, TORCH_ALLOC_GRANULE), TORCH_ALLOC_GRANULE)


class ArenaLayoutError(RuntimeError):
    """A layout request that cannot be satisfied. Always carries the fix in the message."""


class RegionTooLargeError(ArenaLayoutError):
    pass


class ArenaExhaustedError(ArenaLayoutError):
    pass


def round_up(x: int, a: int) -> int:
    if a <= 0:
        raise ValueError(f"alignment must be positive, got {a}")
    return ((x + a - 1) // a) * a


def fmt_bytes(n: int) -> str:
    """Human-readable, and deliberately GiB-only above 1 GiB — every capacity number in the Phase 0
    artifacts is quoted in GiB, so mixing units here would make an operator diff the wrong numbers.

    Negatives are formatted in the same unit, not dumped as raw bytes: the single most important
    number this function ever prints is a *shortfall*, and "-10737418240 B" next to "70.00 GiB" is
    unreadable at exactly the moment it matters.
    """
    if n < 0:
        return "-" + fmt_bytes(-n)
    if n >= GIB:
        return f"{n / GIB:.2f} GiB"
    if n >= MIB:
        return f"{n / MIB:.1f} MiB"
    if n >= KIB:
        return f"{n / KIB:.1f} KiB"
    return f"{n} B"


@dataclass(frozen=True)
class RegionRequest:
    """One thing that needs a contiguous, device-readable home in the arena.

    For M1 a "region" is whatever the granule walker hands over — an expert stack, a scale stack, a
    dense weight. This module deliberately knows nothing about what is inside it: the walker owns
    "which tensors", the arena owns "where the bytes go".
    """

    name: str
    nbytes: int
    align: int = ALIGN
    # True == "this region is a FORECAST of an allocation somebody else will make ANONYMOUSLY".
    #
    # The weight rows are carved by torch's `MemPool` callback (`PinnedWeightArena.allocate_raw`),
    # which is a C ABI that carries a size and nothing else — it cannot name the region it is
    # serving. Before this flag the only way to reserve for it was `extra_bytes` (anonymous
    # headroom), which reserves `headroom_chunks(payload, chunk, max_row)` = a GUARANTEE bound that
    # costs +28% on the target shape because it assumes the worst row lands at the worst offset in
    # every chunk. A forecast region says instead "here is the row list, in carve order": the layout
    # is planned exactly with the same next-fit allocator that will carve it, so the reservation is
    # the real chunk count rather than a bound, and `ChunkPlan.digest()` stops collapsing to a hash
    # of the chunk COUNT.
    #
    # What it does NOT claim is that the carve will use this name. `verify_matches_plan()` therefore
    # checks a forecast region by ENVELOPE (the carve must not need more arena than was planned)
    # and by attribution (a raw carve landing exactly on a forecast placement adopts its name, so
    # `carve_digest()` covers real rows), never by name equality — see `PinnedWeightArena`.
    forecast: bool = False

    def __post_init__(self) -> None:
        if self.nbytes < 0:
            raise ValueError(f"region {self.name!r}: nbytes must be >= 0, got {self.nbytes}")
        if self.align <= 0 or (self.align & (self.align - 1)):
            raise ValueError(f"region {self.name!r}: align must be a power of two, got {self.align}")


@dataclass(frozen=True)
class Placement:
    """Where a region landed. `(chunk_index, offset)` — never a flat arena offset, because the arena
    has no flat address space (chunks are independent mappings).

    `align` is the alignment that PRODUCED this offset, carried so the live carve can reproduce the
    plan exactly. Without it, `RegionRequest.align` is an input the plan consumes and then forgets,
    and `PinnedWeightArena.allocate(name, nbytes)` — whose `align` argument defaults to the
    arena-wide `ALIGN` — silently places a 4096-aligned region at a 512-aligned offset. Every region
    after it shifts, no pointer leaves its chunk, nothing faults, and each granule reads the
    neighbouring granule's bytes.
    """

    name: str
    chunk_index: int
    offset: int
    nbytes: int
    align: int = ALIGN
    # Carried from `RegionRequest.forecast`. A forecast placement is never carved by name, so every
    # by-name check must skip it or it reports every row as "planned but never carved".
    forecast: bool = False

    @property
    def end(self) -> int:
        return self.offset + self.nbytes


class BumpAllocator:
    """Forward-only next-fit bump pointer over `n_chunks` equal chunks.

    This is the ONE piece of arithmetic used by both the up-front plan and the live arena, so the
    thing the unit tests exercise is literally the thing that runs at boot. `PinnedWeightArena`
    holds an instance of this bounded to the chunks it actually pinned; `plan_regions()` holds a
    growable one.
    """

    __slots__ = ("chunk_bytes", "align", "max_chunks", "_cursors", "_cur")

    def __init__(self, chunk_bytes: int, n_chunks: int | None = None, align: int = ALIGN) -> None:
        if chunk_bytes < MIN_CHUNK_BYTES:
            raise ValueError(f"chunk_bytes {chunk_bytes} < MIN_CHUNK_BYTES {MIN_CHUNK_BYTES}")
        if chunk_bytes % CHUNK_GRANULE:
            raise ValueError(
                f"chunk_bytes {chunk_bytes} is not a multiple of CHUNK_GRANULE {CHUNK_GRANULE}"
            )
        self.chunk_bytes = int(chunk_bytes)
        self.align = int(align)
        self.max_chunks = None if n_chunks is None else int(n_chunks)
        # Growable allocators start with one chunk; bounded ones start with all of them, so
        # `n_chunks` means "chunks that exist" in both cases.
        self._cursors: List[int] = [0] * (1 if self.max_chunks is None else self.max_chunks)
        self._cur = 0

    # -- geometry -------------------------------------------------------------

    @property
    def n_chunks(self) -> int:
        return len(self._cursors)

    @property
    def cursors(self) -> Tuple[int, ...]:
        return tuple(self._cursors)

    @property
    def reserved_bytes(self) -> int:
        return self.n_chunks * self.chunk_bytes

    @property
    def consumed_bytes(self) -> int:
        """Payload + intra-chunk alignment padding, i.e. everything below a cursor."""
        return sum(self._cursors)

    @property
    def abandoned_bytes(self) -> int:
        """Tails of chunks the allocator has moved past. This is the price of "never straddle"."""
        return sum(self.chunk_bytes - c for i, c in enumerate(self._cursors) if i < self._cur)

    @property
    def free_bytes(self) -> int:
        """Bytes still allocatable — the tail of the current chunk plus every untouched chunk. A
        request larger than the current tail can still be served from a later chunk, so this is an
        upper bound on what a *single* allocation can get, not a promise."""
        return self.reserved_bytes - self.consumed_bytes - self.abandoned_bytes

    @property
    def footprint_bytes(self) -> int:
        """High-water mark: every byte of arena this allocator has consumed OR abandoned.

        `consumed_bytes` alone is not comparable between two runs of a forward-only next-fit
        allocator, because an abandoned tail is arena that is gone even though no cursor counts it.
        This is the number `verify_matches_plan()` compares carve-against-plan on: it is monotone in
        the request sequence, so "the carve stayed inside the reservation" is exactly
        `carved.footprint_bytes <= planned.footprint_bytes` — a check that passes when the real
        allocations turn out SMALLER or fewer than forecast (torch splitting a block, a component
        the walker deduped) and fails whenever they need more room than was pinned for them.
        """
        if not self._cursors:
            return 0
        return self._cur * self.chunk_bytes + self._cursors[self._cur]

    @property
    def largest_free_run(self) -> int:
        """The biggest single region that can still be placed. This — not `free_bytes` — is what an
        exhaustion check must use."""
        if not self._cursors:
            return 0
        tail = self.chunk_bytes - self._cursors[self._cur]
        untouched = self.chunk_bytes if self._cur + 1 < self.n_chunks else 0
        if self.max_chunks is None:
            untouched = self.chunk_bytes
        return max(tail, untouched)

    # -- allocation -----------------------------------------------------------

    def try_allocate(self, nbytes: int, *, name: str = "", align: int | None = None) -> Placement | None:
        """Next-fit. Returns None when it cannot be placed (bounded allocator only)."""
        a = self.align if align is None else int(align)
        n = int(nbytes)
        if n < 0:
            raise ValueError(f"{name!r}: nbytes must be >= 0, got {n}")
        if n > self.chunk_bytes:
            raise RegionTooLargeError(
                f"region {name!r} needs {fmt_bytes(n)} but a chunk is only "
                f"{fmt_bytes(self.chunk_bytes)}, and a region may never straddle two chunks "
                f"(they are independent hipHostMalloc mappings with non-contiguous device "
                f"pointers). FIX: raise the chunk size to at least "
                f"{fmt_bytes(round_up(n, CHUNK_GRANULE))} "
                f"(MINISGL_WEIGHT_ARENA_CHUNK_MIB={round_up(n, CHUNK_GRANULE) // MIB}) — but note "
                f"nothing above 2 GiB has ever been pinned on this box (P3b), so re-measure before "
                f"trusting it."
            )
        if not self._cursors:
            # A zero-chunk arena — the "everything already fits in VRAM, the plan is empty" no-op
            # path — can place nothing, not even a zero-byte region: there is no chunk to point into.
            return None
        if n == 0:
            # A zero-byte region is legal (an empty component) and must not move the cursor, or two
            # ranks that disagree about whether a component exists would also disagree about every
            # subsequent offset.
            return Placement(name, self._cur, self._cursors[self._cur], 0, a)

        off = round_up(self._cursors[self._cur], a)
        if off + n > self.chunk_bytes:
            # Doesn't fit in this chunk's tail -> abandon the tail, move on.
            if self.max_chunks is None:
                self._cursors.append(0)
            elif self._cur + 1 >= self.n_chunks:
                return None
            self._cur += 1
            off = round_up(self._cursors[self._cur], a)
            if off + n > self.chunk_bytes:  # pragma: no cover - implied by the n > chunk_bytes guard
                return None
        self._cursors[self._cur] = off + n
        return Placement(name, self._cur, off, n, a)

    def allocate(self, nbytes: int, *, name: str = "", align: int | None = None) -> Placement:
        p = self.try_allocate(nbytes, name=name, align=align)
        if p is None:
            raise ArenaExhaustedError(
                f"arena exhausted placing region {name!r} ({fmt_bytes(nbytes)}): "
                f"{self.n_chunks} × {fmt_bytes(self.chunk_bytes)} chunks, "
                f"{fmt_bytes(self.free_bytes)} free but the largest placeable run is only "
                f"{fmt_bytes(self.largest_free_run)}. The arena is forward-only by design (P5b: "
                f"torch never returns a pool block), so it cannot be grown after attach. "
                f"FIX: reserve more up front — pass a larger `extra_bytes`/region list to reserve()."
            )
        return p


@dataclass(frozen=True)
class ChunkPlan:
    """The immutable answer to "how many chunks, and what goes where".

    Built BEFORE a single byte is pinned, so a plan that cannot fit fails in milliseconds instead of
    after a 68 GB load (the whole point of the reserve/attach split).
    """

    chunk_bytes: int
    n_chunks: int
    align: int
    placements: Tuple[Placement, ...]
    cursors: Tuple[int, ...]
    # Whole chunks reserved for allocations that could not be enumerated up front (the torch
    # `MemPool` path). Tracked separately so `abandoned_bytes` does not report them as waste.
    extra_chunks: int = 0
    # What was asked for as headroom, and the caller's bound on any one un-enumerated region (0 =
    # "unbounded", i.e. the reservation assumed perfect packing). Recorded rather than derived
    # because `extra_chunks` alone cannot distinguish "34 GiB with a 400 MiB bound" from
    # "44 GiB with none", and those two plans behave completely differently at carve time.
    extra_bytes: int = 0
    extra_max_region_bytes: int = 0

    # -- accounting -----------------------------------------------------------

    @property
    def reserved_bytes(self) -> int:
        """What must actually be pinned. THIS is the number the capacity gate consumes — not the sum
        of region sizes, which understates by the padding and the abandoned tails."""
        return self.n_chunks * self.chunk_bytes

    @property
    def payload_bytes(self) -> int:
        return sum(p.nbytes for p in self.placements)

    @property
    def consumed_bytes(self) -> int:
        return sum(self.cursors)

    @property
    def padding_bytes(self) -> int:
        return self.consumed_bytes - self.payload_bytes

    @property
    def last_used_chunk(self) -> int:
        used = [i for i, c in enumerate(self.cursors) if c > 0]
        return used[-1] if used else 0

    @property
    def abandoned_bytes(self) -> int:
        """Tails skipped because the next region would have straddled. Excludes the headroom chunks
        and the final chunk's unused tail, which are not waste — they are still allocatable."""
        return sum(self.chunk_bytes - c for c in self.cursors[: self.last_used_chunk])

    @property
    def footprint_bytes(self) -> int:
        """Arena consumed OR abandoned by the placements. See `BumpAllocator.footprint_bytes`."""
        if not self.placements:
            return 0
        last = max(p.chunk_index for p in self.placements)
        return last * self.chunk_bytes + self.cursors[last]

    @property
    def forecast_placements(self) -> Tuple[Placement, ...]:
        """Placements reserved for anonymous carves. See `RegionRequest.forecast`."""
        return tuple(p for p in self.placements if p.forecast)

    @property
    def named_placements(self) -> Tuple[Placement, ...]:
        return tuple(p for p in self.placements if not p.forecast)

    @property
    def forecast_bytes(self) -> int:
        return sum(p.nbytes for p in self.placements if p.forecast)

    @property
    def headroom_bytes(self) -> int:
        return self.extra_chunks * self.chunk_bytes

    @property
    def overhead_bytes(self) -> int:
        """Everything pinned that is not payload: padding + abandoned tails + unallocated tail."""
        return self.reserved_bytes - self.payload_bytes

    @property
    def fill_frac(self) -> float:
        return (self.payload_bytes / self.reserved_bytes) if self.reserved_bytes else 0.0

    def by_name(self) -> dict:
        return {p.name: p for p in self.placements}

    def digest(self) -> str:
        """Stable hash of the whole placement. Two TP ranks that print different digests have
        diverged, and every downstream offset is untrustworthy — log it, or compare it on the CPU
        group. Cheap insurance against the failure mode that hangs a collective instead of raising.
        """
        h = hashlib.sha256()
        h.update(f"chunk_bytes={self.chunk_bytes};align={self.align};n={self.n_chunks}\n".encode())
        for p in self.placements:
            h.update(f"{p.name}\x00{p.chunk_index}\x00{p.offset}\x00{p.nbytes}\n".encode())
        return h.hexdigest()[:16]

    def headroom_advisory(self) -> str | None:
        """Loud when anonymous headroom was reserved with no bound on the region size.

        This is the one place the arena's arithmetic is knowingly optimistic, and the consequence is
        not a slow boot: the rows that do not fit are served by `hipMalloc`, land in VRAM, and make
        the whole capacity plan a fiction that nothing downstream can detect except
        `ArenaMemPool.assert_clean()`. Say so at reserve() time, with the arithmetic, rather than
        letting an operator meet it as an unexplained fallback count after a 7 s pin and a full load.
        """
        if self.extra_chunks <= 0 or self.extra_max_region_bytes:
            return None
        return (
            f"anonymous headroom of {fmt_bytes(self.extra_bytes)} was reserved as {self.extra_chunks}"
            f" x {fmt_bytes(self.chunk_bytes)} ASSUMING PERFECT PACKING. Next-fit abandons a chunk's "
            f"tail whenever the next region does not fit, so for uniform r-byte regions only "
            f"floor(chunk/r)*r is usable — e.g. r=400 MiB leaves 48 MiB/chunk unusable and this "
            f"reservation is short by ~{fmt_bytes(self.extra_chunks * 48 * MIB)}. Regions that do "
            f"not fit fall back to hipMalloc (VRAM) and invalidate the capacity plan. FIX: pass "
            f"`extra_max_region_bytes=<largest single allocation>` to reserve()/plan_regions()."
        )

    def packing_advisory(self, threshold_frac: float = 0.05) -> str | None:
        """Warn when the never-straddle rule is costing real capacity.

        A 598 MiB region tiles three-to-a-2-GiB-chunk and abandons 204 MiB each time — 11 % of the
        arena, which on a feature whose binding constraint IS capacity (PHASE0 §3.4) is worth a
        whole point of device-tier fraction. The fix is *smaller granules* (per-expert rather than
        per-layer regions), which is the walker's decision, so this reports rather than acts.
        """
        if not self.n_chunks or self.abandoned_bytes <= threshold_frac * self.reserved_bytes:
            return None
        return (
            f"chunk packing wastes {fmt_bytes(self.abandoned_bytes)} "
            f"({self.abandoned_bytes / self.reserved_bytes * 100:.1f}% of the arena) to the "
            f"never-straddle rule at chunk={fmt_bytes(self.chunk_bytes)}. Smaller regions "
            f"(per-expert rather than per-layer granules) would reclaim most of it."
        )

    def describe(self) -> str:
        return (
            f"chunks={self.n_chunks}×{fmt_bytes(self.chunk_bytes)} "
            f"reserved={fmt_bytes(self.reserved_bytes)} payload={fmt_bytes(self.payload_bytes)} "
            f"pad={fmt_bytes(self.padding_bytes)} abandoned={fmt_bytes(self.abandoned_bytes)} "
            f"headroom={fmt_bytes(self.headroom_bytes)} "
            f"fill={self.fill_frac * 100:.1f}% regions={len(self.placements)}"
            f"(forecast={len(self.forecast_placements)}) "
            f"digest={self.digest()}"
        )


def suggest_chunk_bytes(
    requests: Sequence[RegionRequest],
    preferred: int = DEFAULT_CHUNK_BYTES,
    *,
    hard_max: int = 4 * GIB,
) -> int:
    """Smallest legal chunk size that is >= `preferred` and can hold the largest single region.

    Grows only when forced, and never silently past `hard_max` — beyond 2 GiB this box has no
    measurement at all, so an automatic jump to a 12 GiB chunk would be an unvalidated allocation
    dressed up as a default.
    """
    largest = max((r.nbytes for r in requests), default=0)
    want = max(round_up(int(preferred), CHUNK_GRANULE), round_up(largest, CHUNK_GRANULE))
    want = max(want, MIN_CHUNK_BYTES)
    if want > hard_max:
        raise RegionTooLargeError(
            f"largest region is {fmt_bytes(largest)}, which would force a "
            f"{fmt_bytes(want)} chunk — above the {fmt_bytes(hard_max)} cap. No pinned allocation "
            f"above 2 GiB has ever been measured on this box (P3b). FIX: split the region "
            f"(per-expert or per-layer granules are already the design), or raise `hard_max` "
            f"deliberately and re-run P3b at that size."
        )
    return want


def headroom_chunks(extra_bytes: int, chunk_bytes: int, max_region_bytes: int = 0) -> int:
    """How many whole chunks it takes to be SURE `extra_bytes` of un-enumerated regions will fit.

    `ceil(extra_bytes / chunk_bytes)` — the obvious answer — is WRONG, and wrong in the direction
    that costs a boot. Next-fit abandons the tail of a chunk whenever the next request does not fit
    in it, so the usable fraction of a chunk is `floor(chunk/r)*r` for uniform `r`-byte regions, not
    `chunk`. With the layer-granular rows this feature actually reserves — P3b's 2 GiB chunk, ~400
    MiB per component stack — that is 2000 of 2048 MiB usable, so a 34 GiB reservation of 17 chunks
    is ~800 MiB short and the last rows fall out of the arena entirely. They then come back as
    `hipMalloc` **VRAM**, which is the one outcome the whole capacity plan exists to prevent.

    With a bound `m` on any single region, next-fit leaves at most `m` bytes unused per chunk (the
    request that did not fit was no larger than `m`), so `chunk - m` bytes per chunk are guaranteed
    placeable. That bound is loose — it is a guarantee, not an estimate — which is why callers that
    know their largest row should pass it and callers that do not get the old arithmetic plus a loud
    advisory rather than a silently different number.
    """
    n = int(extra_bytes)
    if n <= 0:
        return 0
    c, m = int(chunk_bytes), int(max_region_bytes)
    if m < 0:
        raise ValueError(f"max_region_bytes must be >= 0, got {m}")
    if m >= c:
        raise RegionTooLargeError(
            f"headroom regions may be up to {fmt_bytes(m)} but a chunk is only {fmt_bytes(c)}, and a "
            f"region may never straddle two chunks. FIX: raise the chunk size to at least "
            f"{fmt_bytes(round_up(m + CHUNK_GRANULE, CHUNK_GRANULE))}, or split the regions."
        )
    usable = c - m if m else c
    return (n + usable - 1) // usable


def plan_regions(
    requests: Sequence[RegionRequest],
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    *,
    align: int = ALIGN,
    extra_bytes: int = 0,
    extra_max_region_bytes: int = 0,
) -> ChunkPlan:
    """Lay `requests` out, in the order given, into as many chunks as it takes.

    `extra_bytes` reserves trailing headroom that no named region owns — for allocations made later
    through the torch `MemPool` path, which cannot be enumerated up front. `extra_max_region_bytes`
    is an upper bound on any ONE of those allocations; pass it whenever it is known, because without
    it the reservation assumes perfect packing and next-fit does not pack perfectly. See
    `headroom_chunks`, and `ChunkPlan.headroom_advisory()` for what is printed when it is omitted.
    """
    names = [r.name for r in requests]
    if len(set(names)) != len(names):
        dupes = sorted({n for n in names if names.count(n) > 1})
        raise ArenaLayoutError(
            f"duplicate region names {dupes} — the region table is keyed by name, so a duplicate "
            f"silently overwrites the earlier entry and one granule ends up reading another's bytes"
        )
    b = BumpAllocator(chunk_bytes, n_chunks=None, align=align)
    placements = [
        replace(b.allocate(r.nbytes, name=r.name, align=r.align), forecast=r.forecast)
        if r.forecast
        else b.allocate(r.nbytes, name=r.name, align=r.align)
        for r in requests
    ]
    extra_chunks = headroom_chunks(extra_bytes, chunk_bytes, extra_max_region_bytes)
    # An empty plan must reserve ZERO chunks, not one. §6.2's design has the plan *derived*: when
    # everything fits in VRAM the plan is empty and the whole path must cost nothing, or the "it is
    # exercised on every serve and cannot rot" argument comes with a silent 2 GiB tax.
    #
    # `or placements` is load-bearing and is NOT the same test: a plan whose regions are ALL
    # zero-byte leaves every cursor at 0, so the `any(c > 0)` test alone reports "no chunks used"
    # while every `Placement` still names chunk 0. `attach()` would then pin nothing, and the first
    # `allocate()` — for a region the plan said was placed — would raise `ArenaExhaustedError` (or
    # index an empty chunk table). That is reachable at TP=2 without any all-empty checkpoint: the
    # zero-byte region exists precisely so two ranks that DISAGREE about whether a component is
    # present still agree on every subsequent offset, and a rank whose shard is entirely empty
    # (an expert-parallel split that gives it no experts) produces exactly this plan while its peer
    # produces a populated one. Reserve the chunk those placements point into.
    used_chunks = b.n_chunks if (any(c > 0 for c in b.cursors) or placements) else 0
    n_chunks = used_chunks + extra_chunks
    cursors = list(b.cursors[:used_chunks]) + [0] * extra_chunks
    return ChunkPlan(
        chunk_bytes=int(chunk_bytes),
        n_chunks=n_chunks,
        align=int(align),
        placements=tuple(placements),
        cursors=tuple(cursors),
        extra_chunks=extra_chunks,
        extra_bytes=max(0, int(extra_bytes)),
        extra_max_region_bytes=max(0, int(extra_max_region_bytes)),
    )
