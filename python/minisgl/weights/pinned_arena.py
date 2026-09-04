"""`PinnedWeightArena` — the host tier of the weight-offload arena.

WHAT THIS IS. A fixed set of `hipHostMalloc(...Mapped)` chunks whose device-visible addresses come
from `hipHostGetDevicePointer`, plus a forward-only bump allocator that carves named regions out of
them. Weights live here; a kernel reads them straight over PCIe at 28.93 GB/s (card 0) or
14.48 GB/s (card 1). That asymmetry is a **Gen5 x8 vs Gen4 x8 root-port** difference, not a driver
artifact, so every TP=2 ceiling is card-1-gated — plan against 14.48, never 28.93.

THE MECHANISM IS THE ONLY ONE THAT EXISTS ON THIS BOX. Phase 0 established that
`hipMemCreate(location=Host)` silently returns VRAM and that `hipMemUnmap`→`hipMemMap` serves stale
pages, both with `hipSuccess` throughout. So: no VMM, no mixed-media VA, no dynamic residency. See
`hipmem.py` for the full list of what is deliberately not bound.

--------------------------------------------------------------------------------------------------
WHAT WAS REUSED FROM `kvcache/host_arena.py`, AND WHAT CHANGED

REUSED (idiom, and in two cases the exact constant/shape):

* **The reserve-then-attach split.** `FrameLayout` there is pure byte arithmetic with no memory
  behind it; `PinnedFrameArena` then attaches one slab and hands out views onto it. Same split here:
  `chunk_plan.py` is pure integers, `reserve()` decides everything without a single HIP call, and
  `attach()` is the only function that pins. That split is what makes the failure *early* — a
  capacity abort costs milliseconds instead of 68 GB of load.
* **One big allocation, not many small ones**, with views attached afterwards. Same reason
  (`host_arena.py:148-150`): a few large pins are far cheaper than thousands of small ones and the
  pinned footprint stays exactly predictable.
* **Alignment discipline** — a single alignment constant applied to every carve so any component
  view is safe to reinterpret from the flat bytes (`_ALIGN = 256` there, `ALIGN = 512` here; see
  `chunk_plan.ALIGN` for why it grew).
* **No timing-dependent decisions.** `host_arena.py`'s invariant 1 forbids `event.query()` in
  alloc/release because two TP ranks would then make different decisions and the collectives would
  hang. The same hazard exists here in a different costume: `MemAvailable` is timing-dependent, so
  the capacity check may only ever *pass or raise* — never quietly reserve less on one rank.
* **`__del__` must not raise** — interpreter shutdown nulls module globals before the last object
  dies (`host_arena.py:268-274`).

CHANGED, and why:

* **No free list, no `wait_event`, no FIFO ring.** That module is a *fixed-frame, cold-path,
  recycling* allocator: 20 identical 16.4 MiB frames, reused thousands of times, where a frame's
  last DMA may still be in flight. This one is *huge-extent, variable-size, write-once*: ~34 GiB of
  weights, written at load, read for the process lifetime, never freed. P5b measured torch's free
  callback firing **zero** times for live *and* cached pool blocks, so a free list here could only
  ever be dead code that is wrong. Forward-only bump is the correct and only design — and it means
  sizing must be right at construction, which is exactly why `reserve()` exists.
* **Chunked, not one slab.** 34 GiB is not one `hipHostMalloc`; P3b demonstrated 2 GiB chunks and
  nothing larger has ever been pinned on this box. Chunks are independent mappings with
  non-contiguous device pointers, hence the never-straddle rule in `chunk_plan.py`.
* **A capacity gate at all.** `PinnedFrameArena` allocates 0.32 GiB and cannot plausibly fail. This
  one is the difference between booting and destroying the box (P3b: 114,813 pages swapped at the
  ceiling), so it has a `MemAvailable` floor, an all-local-ranks charge, and a swap tripwire.
* **An out-of-band data self-test.** Nothing in the snapshot arena checks that pinned pages actually
  store what was written. Here it is mandatory (plan §5.4 A1.4): Phase 0 produced FOUR independent
  cases of the driver reporting success over wrong state. **Assert on the data, never on a return
  code or a query.**
* **A freeze.** Snapshot frames are allocated and recycled for the whole run; weight chunks must all
  exist before `_determine_num_pages` and none after (rule R1).
--------------------------------------------------------------------------------------------------
"""

from __future__ import annotations

import ctypes
import hashlib
import statistics
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import hipmem
from .chunk_plan import (
    ALIGN,
    CHUNK_GRANULE,
    DEFAULT_CHUNK_BYTES,
    ArenaLayoutError,
    BumpAllocator,
    ChunkPlan,
    Placement,
    RegionRequest,
    fmt_bytes,
    plan_regions,
    suggest_chunk_bytes,
)
from .host_capacity import (
    DEFAULT_FLOOR_BYTES,
    CapacityVerdict,
    HostArenaCapacityError,
    SwapTripwire,
    check_capacity,
    mem_available_bytes,
)

PAGE = 4096


class ArenaPhase(str, Enum):
    NEW = "new"
    RESERVED = "reserved"      # plan built, capacity cleared, nothing pinned yet
    ATTACHED = "attached"      # chunks pinned and self-tested, regions being carved
    POPULATED = "populated"    # weights written; the self-test may never run again
    CLOSED = "closed"


class ArenaStateError(RuntimeError):
    """An operation was attempted in the wrong phase — including the one that would overwrite
    weights with self-test fingerprints."""


class ArenaSelfTestError(RuntimeError):
    """The pages did not store what was written, or the chunk table is structurally impossible.

    Data first — that is the ONLY trustworthy failure signal on this box. But the data check is a
    SAMPLE (ten 4-byte probes per 2 GiB chunk = 1.9e-8 coverage), so it is backed by an exact
    structural check on the pointers themselves. See `_verify_chunk_structure`.
    """


@dataclass(frozen=True)
class ArenaChunk:
    index: int
    host_ptr: int
    device_ptr: int
    nbytes: int
    fingerprint: int
    pin_seconds: float = 0.0
    touch_seconds: float = 0.0

    @property
    def same_va(self) -> bool:
        """True on this box (ROCm unified VA). Recorded, never assumed — P5b's caveat."""
        return self.host_ptr == self.device_ptr

    @property
    def pin_gb_s(self) -> Optional[float]:
        return self.nbytes / self.pin_seconds / 1e9 if self.pin_seconds > 0 else None


@dataclass(frozen=True)
class ArenaRegion:
    """A contiguous, device-readable home for one granule (or one torch allocation).

    Carries BOTH pointers because both are load-bearing: the device pointer is what a kernel and a
    torch tensor use, the host pointer is what `populate` writes through. On this box they are equal
    (unified VA) — which is exactly why they must be carried separately, so nothing accidentally
    depends on the equality.
    """

    name: str
    chunk_index: int
    offset: int
    nbytes: int
    host_ptr: int
    device_ptr: int

    def host_buffer(self) -> Any:
        """A writable ctypes view of the bytes, for CPU-side population.

        Safe: Phase 0 unknown #6 was answered for pinned zero-copy — CPU-write → kernel-read passed
        on both cards with no explicit flush, after a fence, after `clflush`, and at sub-64 B line
        granularity. (The plan's write-through-the-device-pointer mitigation still costs nothing and
        remains the recommendation for the bulk path; this view exists for population and
        verification.)
        """
        return (ctypes.c_ubyte * self.nbytes).from_address(self.host_ptr)

    def host_tensor(self, dtype: Any = None, shape: Sequence[int] | None = None) -> Any:
        """A CPU torch tensor aliasing the pinned bytes — no copy, no pin_memory round trip.

        torch is imported lazily so that everything above (planning, capacity, layout) stays usable
        on a host where torch is not importable at all.
        """
        import torch

        dtype = torch.uint8 if dtype is None else dtype
        t = torch.frombuffer(self.host_buffer(), dtype=dtype)
        return t if shape is None else t.view(*shape)


def verify_offsets(chunk_bytes: int) -> Tuple[int, ...]:
    """Head, tail and page-aligned interior sample points.

    Head+tail alone cannot see a chunk whose middle landed on someone else's physical pages — the
    exact P6 failure shape — so the interior is sampled too. Ten 4-byte probes cost ~0.1 ms/chunk.
    """
    offs = {0}
    for k in range(1, 8):
        o = (chunk_bytes * k // 8) & ~(PAGE - 1)
        if 0 < o <= chunk_bytes - 4:
            offs.add(o)
    offs.add(chunk_bytes - 4)
    return tuple(sorted(offs))


def chunk_fingerprint(rank: int, device: int, index: int) -> int:
    """A word unique per (rank, device, chunk).

    Uniqueness is the point: a read-back that returns some *other* chunk's fingerprint identifies
    the aliasing partner instead of merely reporting "mismatch", and folding the rank in means a
    cross-PROCESS physical-page mixup is also identifiable.

    THE FIELD WIDTHS ARE CHECKED, NOT MASKED. Masking `rank` to 4 bits is silent for rank <= 15 and
    catastrophic at 16: rank 16 and rank 0 would stamp the *same* word, so the one thing this
    fingerprint exists to detect — two processes' chunks landing on the same physical pages — would
    read back "correct" on both. A self-test that cannot fail is worse than no self-test, and the
    encoding must not be something a larger TP size silently outgrows. Same argument for `device`
    (16 HIP devices is not a hypothetical on a multi-GPU host) and for `index` at 65536 chunks.
    """
    if not 0 <= int(rank) <= 0xF:
        raise ValueError(
            f"rank {rank} does not fit the 4-bit fingerprint field. Masking it would make two ranks "
            f"stamp identical words and the cross-process aliasing check silently unfalsifiable — "
            f"widen the encoding (the top byte 0xA5 has room) before serving a TP size this large."
        )
    if not 0 <= int(device) <= 0xF:
        raise ValueError(f"device {device} does not fit the 4-bit fingerprint field; widen it")
    if not 0 <= int(index) <= 0xFFFF:
        raise ValueError(
            f"chunk index {index} does not fit the 16-bit fingerprint field "
            f"(65536 chunks); widen the encoding"
        )
    return 0xA5000000 | (int(rank) << 20) | (int(device) << 16) | int(index)


def decode_fingerprint(word: int) -> Optional[Dict[str, int]]:
    if (int(word) & 0xFF000000) != 0xA5000000:
        return None
    return {"rank": (word >> 20) & 0xF, "device": (word >> 16) & 0xF, "chunk": word & 0xFFFF}


@dataclass
class SelfTestResult:
    """`passed` is a POSITIVE statement: N chunks were probed at M offsets each and every word came
    back right. It is deliberately not "no failures were recorded".

    Those differ exactly when nothing was probed, and that case is reachable: `_rollback()` empties
    `self.chunks` on any attach failure, and a plan with `n_chunks == 0` never pins anything. A
    self-test that reports `selftest PASS chunks=0` in the `[serve]` banner is a green light that
    asserted nothing — the same species of evidence as the `hipSuccess` return codes Phase 0 spent
    six probes learning not to trust. `expected_chunks` is what makes the statement checkable.
    """

    passed: bool
    chunks_checked: int
    offsets_per_chunk: int
    failures: List[Dict[str, Any]] = field(default_factory=list)
    seconds: float = 0.0
    expected_chunks: int = 0

    @property
    def vacuous(self) -> bool:
        """Nothing was actually probed. Never a pass."""
        return self.chunks_checked == 0 or self.offsets_per_chunk == 0

    def summary(self) -> str:
        return (
            f"selftest {'PASS' if self.passed else 'FAIL'} "
            f"chunks={self.chunks_checked}/{self.expected_chunks} "
            f"offsets/chunk={self.offsets_per_chunk} "
            f"failures={len(self.failures)} in {self.seconds * 1e3:.0f} ms"
            + (" VACUOUS(nothing probed)" if self.vacuous else "")
        )


@dataclass(frozen=True)
class PlanVerification:
    """What `PinnedWeightArena.verify_matches_plan()` actually compared.

    A boolean pass is not enough for this check and never was: the failure it exists to catch is a
    layout drift, and the way it FAILED was not by returning False but by having nothing to compare
    — the shipping caller reserved anonymous headroom, so the planned table was empty and a green
    result carried zero information. `vacuous` is therefore part of the result, and
    `verify_matches_plan` treats it as an error rather than as a pass.
    """

    named_planned: int
    named_carved: int
    forecast_planned: int
    forecast_matched: int
    raw_carves: int
    planned_footprint: int
    carved_footprint: int
    planned_payload: int
    carved_bytes: int
    headroom_bytes: int = 0

    @property
    def vacuous(self) -> bool:
        """Nothing was checked. Never a pass."""
        return self.named_carved == 0 and self.forecast_matched == 0

    @property
    def forecast_coverage(self) -> float:
        return self.forecast_matched / self.forecast_planned if self.forecast_planned else 0.0

    def describe(self) -> str:
        return (
            f"named {self.named_carved}/{self.named_planned} checked, forecast "
            f"{self.forecast_matched}/{self.forecast_planned} attributed, {self.raw_carves} "
            f"anonymous carve(s), footprint {fmt_bytes(self.carved_footprint)} of "
            f"{fmt_bytes(self.planned_footprint)} planned"
            + (f", {fmt_bytes(self.headroom_bytes)} anonymous headroom" if self.headroom_bytes
               else "")
        )


class PinnedWeightArena:
    """The host tier. Build it, reserve it, attach it, carve it, populate it, freeze it.

    Lifecycle (each step refuses to run out of order — see `ArenaPhase`):

        arena = PinnedWeightArena(device_index=0, rank=0, local_ranks=2)
        arena.reserve([RegionRequest("layer0.w13", n), ...])   # pure; may raise capacity, fast
        arena.attach()                                          # the only step that pins
        r = arena.allocate("layer0.w13", n)                     # carve; matches the plan
        ...populate through r.host_buffer() / r.device_ptr...
        arena.mark_populated()                                  # self-test locked out from here
        arena.freeze()                                          # rule R1: no mapping after boot
    """

    def __init__(
        self,
        device_index: int,
        *,
        rank: int = 0,
        local_ranks: int = 1,
        chunk_bytes: int = DEFAULT_CHUNK_BYTES,
        align: int = ALIGN,
        floor_bytes: int = DEFAULT_FLOOR_BYTES,
        hip: Any = None,
        label: str = "weights",
        selftest_default: bool = True,
        first_touch_default: bool = True,
    ) -> None:
        # attach()'s defaults, so the resolved settings survive the trip from
        # `resolve_arena_settings()` through `create_pinned_weight_arena()` to the one function that
        # consumes them. See attach().
        self.selftest_default = bool(selftest_default)
        self.first_touch_default = bool(first_touch_default)
        self.device_index = int(device_index)
        self.rank = int(rank)
        self.local_ranks = int(local_ranks)
        self.chunk_bytes = int(chunk_bytes)
        self.align = int(align)
        self.floor_bytes = int(floor_bytes)
        self.label = label
        # Injectable so every non-pinning code path — planning, capacity, carving, the self-test's
        # own failure handling — is testable without a GPU. Production passes None and gets the real
        # binding, resolved lazily at attach() so that constructing an arena on a ROCm-less host
        # (e.g. to print a plan) does not dlopen libamdhip64.
        self._hip = hip
        self.phase = ArenaPhase.NEW

        self.plan: Optional[ChunkPlan] = None
        self.capacity: Optional[CapacityVerdict] = None
        # attach()'s own single-rank re-check, kept separate so the multi-rank reserve() verdict is
        # not silently overwritten by the weaker one.
        self.capacity_at_attach: Optional[CapacityVerdict] = None
        self.chunks: List[ArenaChunk] = []
        self._bump: Optional[BumpAllocator] = None
        self._regions: Dict[str, ArenaRegion] = {}
        # Names carved anonymously through `allocate_raw` (the torch MemPool callback). Tracked
        # explicitly rather than recognised by a `torch:` name prefix: `verify_matches_plan()` EXEMPTS
        # unplanned raw carves, and a prefix test would silently exempt a *granule* the walker
        # happened to name `torch:...` — an exemption is the one thing that must not be spoofable by
        # a string.
        self._raw_names: List[str] = []
        self._selftest: Optional[SelfTestResult] = None
        self.attach_seconds = 0.0
        self.device_identity: Dict[str, Any] = {}
        # Non-arena allocations the torch pool had to serve from hipMalloc. MUST be 0 after populate
        # — a fallback means weights silently landed in VRAM and the capacity plan is a fiction.
        self.torch_fallbacks = 0
        # Set by reserve() when the never-straddle rule is costing material capacity.
        self.packing_advisory: Optional[str] = None
        # Set by reserve() when anonymous headroom was sized without a region-size bound.
        self.headroom_advisory: Optional[str] = None
        # Set by reserve() when a region forced the chunk size up. Not an error: the whole point is
        # that a new checkpoint does not need an operator to set an env var.
        self.chunk_growth_advisory: Optional[str] = None
        # Bytes `allocate_raw` had to refuse. Non-zero means the anonymous headroom was
        # under-reserved and torch fell back to VRAM; recorded here so the DIAGNOSIS is available,
        # not just the symptom (`torch_fallbacks`).
        self.headroom_denied_bytes = 0
        self.headroom_denied_max = 0
        # How many raw (`allocate_raw`) carves landed exactly on a FORECAST placement and adopted
        # its name. See `RegionRequest.forecast`: the torch `MemPool` C ABI carries a size and
        # nothing else, so a forecast row cannot be carved BY NAME — attribution by coinciding
        # (chunk, offset) is what makes `carve_digest()` cover real weight rows instead of a run of
        # `torch:<n>` counters, and what makes `verify_matches_plan()` able to say the carve
        # followed the plan.
        self._forecast_cursor = 0
        self._forecast_matched = 0

    # -- phase plumbing -------------------------------------------------------

    def _require(self, *phases: ArenaPhase, what: str) -> None:
        if self.phase not in phases:
            raise ArenaStateError(
                f"{what} requires phase {' or '.join(p.value for p in phases)}, "
                f"but the arena is {self.phase.value}"
            )

    def _hip_or_bind(self) -> Any:
        if self._hip is None:
            self._hip = hipmem.get_hip()
        return self._hip

    # -- step 1: reserve (PURE — no HIP call, no pinning) ---------------------

    def reserve(
        self,
        requests: Sequence[RegionRequest],
        *,
        extra_bytes: int = 0,
        extra_max_region_bytes: int = 0,
        check: bool = True,
    ) -> ChunkPlan:
        """Build the plan and clear the capacity gate. **Fails in milliseconds, not after a load.**

        `extra_bytes` is headroom for allocations that cannot be enumerated up front (the torch
        `MemPool` path); `extra_max_region_bytes` bounds any one of them. See `plan_regions` and
        `headroom_chunks` — without the bound the reservation assumes perfect packing and the plan
        carries `headroom_advisory()` saying so.

        `check=False` is for printing a plan on a machine that is not the target box. It does not
        exist to let a failing plan through: `attach()` re-checks unconditionally.
        """
        self._require(ArenaPhase.NEW, what="reserve()")

        # A REGION LARGER THAN A CHUNK MUST NOT REQUIRE AN OPERATOR.
        #
        # `BumpAllocator` refuses to straddle, so a single region above `chunk_bytes` is unplaceable
        # and the old behaviour was to raise and tell the operator to set
        # MINISGL_WEIGHT_ARENA_CHUNK_MIB. That makes "a new checkpoint" a config task: a model with
        # more experts, a wider hidden size, a coarser granule, TP=1 instead of TP=2, or simply a
        # dense lm_head bigger than 2 GiB would all bounce off a default that nothing about the model
        # justifies. The chunk size is not a property of the checkpoint; it is a property of what
        # this box has been shown to pin.
        #
        # So grow it here instead. `suggest_chunk_bytes` is a PURE function of the request list, so
        # every TP rank derives the identical size from the identical regions — no rank can grow
        # while its peer does not, which is the one thing that would make this a desync. It still
        # refuses past 4 GiB, loudly, because nothing above 2 GiB has ever been pinned on this box.
        probe = list(requests)
        if extra_max_region_bytes > 0:
            # `+ CHUNK_GRANULE`, and it is load-bearing rather than a rounding nicety. A region may
            # never straddle, so `headroom_chunks` guarantees only `chunk - m` placeable bytes per
            # chunk — at `chunk == m` that is ZERO and the function raises `RegionTooLargeError`
            # ("headroom regions may be up to X but a chunk is only X"). `suggest_chunk_bytes` grows
            # the chunk to exactly `round_up(largest)`, so probing with `m` itself produced precisely
            # that degenerate equality whenever the largest headroom row exceeded the preferred chunk
            # — i.e. the caller did the right thing (passed its row bound) and got an exception
            # instead of a bigger chunk. Probe one granule above the bound so the grown chunk is
            # strictly larger than any row it must hold. `plan.effective_chunk_bytes` mirrors this
            # exactly, so the planner charges the chunks this will pin.
            probe.append(
                RegionRequest("__headroom_bound__", int(extra_max_region_bytes) + CHUNK_GRANULE)
            )
        want = suggest_chunk_bytes(probe, self.chunk_bytes)
        if want > self.chunk_bytes:
            self.chunk_growth_advisory = (
                f"chunk size grown {fmt_bytes(self.chunk_bytes)} -> {fmt_bytes(want)}: a region "
                f"needs more than one chunk's worth and may never straddle. This is a pure function "
                f"of the region list, so every TP rank derives the same size. NOTE: no pinned "
                f"allocation above 2 GiB has ever been measured on this box (P3b) — record the "
                f"pin rate from stats() if this run is a first."
            )
            self.chunk_bytes = want

        plan = plan_regions(
            requests,
            self.chunk_bytes,
            align=self.align,
            extra_bytes=extra_bytes,
            extra_max_region_bytes=extra_max_region_bytes,
        )
        # Surfaced, not acted on: the fix is smaller granules, which is the walker's call.
        self.packing_advisory = plan.packing_advisory()
        self.headroom_advisory = plan.headroom_advisory()
        if check:
            # Charge EVERY local rank: at TP=2 both ranks pin from the same host RAM, so a per-rank
            # check passes twice and the box still dies.
            self.capacity = check_capacity(
                plan.reserved_bytes, self.local_ranks, self.floor_bytes, raise_on_fail=True
            )
        self.plan = plan
        self.phase = ArenaPhase.RESERVED
        return plan

    # -- step 2: attach (the only function that pins) -------------------------

    def attach(self, *, selftest: bool | None = None, first_touch: bool | None = None) -> None:
        """Pin every chunk, prove each one stores what was written, then resweep them all.

        `first_touch=True` fills each whole chunk through the DEVICE pointer. Two reasons, both
        load-bearing: drivers commit pages lazily, so a `MemAvailable` delta sampled at allocation
        time can be ~0 and prove nothing; and the fill is device-issued, so it exercises the page
        table the kernels will walk rather than the CPU mapping. Cost is ~1 chunk-size of PCIe per
        chunk (~70 ms per 2 GiB on card 0, ~145 ms on card 1).

        `None` (the default for both) means "use what the constructor was given", which is what
        `create_pinned_weight_arena()` resolved from the environment. Before that, the factory read
        `MINISGL_WEIGHT_ARENA_SELFTEST` / `_FIRST_TOUCH` and then threw both away — every caller that
        did not separately call `resolve_arena_settings()` and re-pass the booleans by hand got the
        hardcoded defaults, and the knobs read as a silent no-op rather than an error. That is the
        same five-hop plumbing failure that makes a minisgl metric export a flat zero.
        """
        self._require(ArenaPhase.RESERVED, what="attach()")
        selftest = self.selftest_default if selftest is None else bool(selftest)
        first_touch = self.first_touch_default if first_touch is None else bool(first_touch)
        hip = self._hip_or_bind()

        # `hipSetDevice` is process-global state that torch also owns, and torch CACHES it
        # (`c10::cuda::GetDevice` returns a thread-local `targetDeviceIndex` rather than asking the
        # driver). Leaving the driver on a different device than torch believes it is on means torch
        # skips the `hipSetDevice` it would otherwise issue and launches kernels on the WRONG CARD —
        # which on this box is not a 5% effect, it is a Gen5-x8 card and a Gen4-x8 card. So bind it
        # for the duration of the pinning and put it back, on every exit path, rather than leaving
        # the process wherever the arena happened to need it. `attach()` is also the ONLY entry
        # point that can leave pinned chunks behind on an abort, so the same `finally` guarantees
        # the rollback: `SwapTripwire.check` used to raise straight out of the pinning loop with
        # every chunk so far still pinned — a failed boot that keeps 30 GiB of unevictable host RAM
        # is the box-destroying outcome `_rollback` exists to prevent.
        prev_device: Optional[int] = None
        try:
            prev_device = int(hip.current_device())
        except Exception:  # pragma: no cover - a runtime that cannot answer cannot be restored
            prev_device = None
        try:
            self._attach_pinning(hip, selftest=selftest, first_touch=first_touch)
        except BaseException:
            if self.phase is not ArenaPhase.CLOSED and self.chunks:
                self._rollback(hip)
            raise
        finally:
            # `>= 0` because a runtime that has never had a device selected answers with a sentinel,
            # and restoring a sentinel is worse than leaving the device where the arena put it.
            if prev_device is not None and prev_device >= 0 and prev_device != self.device_index:
                try:
                    hip.set_device(prev_device)
                except Exception:  # pragma: no cover - best effort on an already-failing path
                    pass

    def _attach_pinning(self, hip: Any, *, selftest: bool, first_touch: bool) -> None:
        """The body of `attach()`. Split out only so `attach()` can own the `finally` that restores
        the HIP device and guarantees rollback; every invariant lives here."""
        assert self.plan is not None
        t_start = time.perf_counter()
        hip.set_device(self.device_index)
        self.device_identity = {
            "hip_device": self.device_index,
            "pci_bus_id": hip.pci_bus_id(self.device_index),
            "name": hip.device_name(self.device_index),
            "free_vram_bytes": hip.free_vram(),
        }

        # Re-check unconditionally — the box is shared with every other agent on this machine — but
        # charge THIS RANK ONLY.
        #
        # Charging `local_ranks` here is a guaranteed rank-asymmetric false abort at TP=2. The ranks
        # are separate processes (engine/config.py: one device per `device_index`) and they never
        # attach simultaneously; pinning 34 GiB takes ~7 s (P3b). So rank 0 pins first, MemAvailable
        # drops by its whole arena, and rank 1 then re-charges rank 0's *already pinned* bytes a
        # second time against the already-reduced reading:
        #
        #   MemAvailable 90 GiB, floor 12, plan 34/rank, local_ranks=2
        #     reserve()  both ranks: 68 + 12 = 80 <= 90            -> pass
        #     rank 0 attaches 34 GiB                               -> MemAvailable 56
        #     rank 1 attach re-check: 68 + 12 = 80 > 56            -> ABORT
        #
        # Rank 0 boots, rank 1 dies, and the first collective hangs — the exact TP desync this
        # module's own docstrings say the capacity policy exists to prevent. The multi-rank charge is
        # only meaningful at reserve() time, before any rank has pinned; from attach() onward every
        # peer's pinned bytes are ALREADY subtracted from MemAvailable, which is precisely the
        # reasoning the per-chunk loop below already documents for itself. Box death is prevented by
        # that incremental floor check plus the swap tripwire, not by re-charging peers.
        self.capacity_at_attach = check_capacity(
            self.plan.reserved_bytes, 1, self.floor_bytes, raise_on_fail=True
        )
        # Keep reserve()'s multi-rank verdict as the reported one — it is the stronger statement and
        # the one `stats()` should show. Only fall back to the attach verdict when reserve() ran with
        # check=False and therefore recorded nothing.
        if self.capacity is None:
            self.capacity = self.capacity_at_attach
        # Scaled to THIS arena's reservation: a flat page count is a fraction of a percent of a
        # 30 GiB pin and aborts on ambient co-tenant swap traffic. See `SwapTripwire`.
        swap = SwapTripwire(pin_bytes=self.plan.reserved_bytes)

        for i in range(self.plan.n_chunks):
            # "Can I take the NEXT chunk and still be above the floor?" — P3b's incremental check,
            # shifted by one chunk so the abort happens before the allocation that would breach it.
            # Deliberately charges only THIS rank's next chunk: the other ranks' already-pinned
            # bytes are visible in MemAvailable already, and charging their *future* bytes again
            # would abort boots that would have succeeded. The full multi-rank charge lives in
            # reserve()/attach()'s up-front check, where no rank has allocated yet.
            avail = mem_available_bytes()
            if avail - self.chunk_bytes < self.floor_bytes:
                self._rollback(hip)
                raise HostArenaCapacityError(
                    f"WEIGHT OFFLOAD: pinning stopped at chunk {i}/{self.plan.n_chunks} "
                    f"({fmt_bytes(i * self.chunk_bytes)} pinned). MemAvailable is "
                    f"{fmt_bytes(avail)}; taking another {fmt_bytes(self.chunk_bytes)} would leave "
                    f"less than the {fmt_bytes(self.floor_bytes)} floor. This is exactly how P3b's "
                    f"rank 1 stopped at 28.0 of 34.0 GiB. Raise the device-tier fraction or free "
                    f"host RAM; see the reserve()-time message for the quantified table."
                )
            # `avail` is the reading taken one line above, so the arming decision and the floor
            # decision are made from the SAME sample — two /proc reads could straddle a co-tenant's
            # allocation and disagree about whether the box is short of memory.
            swap.check(
                f"at chunk {i}/{self.plan.n_chunks}", available=avail, floor=self.floor_bytes
            )

            t0 = time.perf_counter()
            try:
                host_ptr, dev_ptr = hip.host_alloc(self.chunk_bytes)
            except Exception:
                self._rollback(hip)
                raise
            t1 = time.perf_counter()

            fp = chunk_fingerprint(self.rank, self.device_index, i)
            t2 = t1
            if first_touch:
                try:
                    hip.memset_d32(dev_ptr, fp, self.chunk_bytes // 4)
                    hip.sync()
                except Exception:
                    self.chunks.append(ArenaChunk(i, host_ptr, dev_ptr, self.chunk_bytes, fp))
                    self._rollback(hip)
                    raise
                t2 = time.perf_counter()
            self.chunks.append(
                ArenaChunk(
                    index=i,
                    host_ptr=host_ptr,
                    device_ptr=dev_ptr,
                    nbytes=self.chunk_bytes,
                    fingerprint=fp,
                    pin_seconds=t1 - t0,
                    touch_seconds=(t2 - t1) if first_touch else 0.0,
                )
            )

        self._bump = BumpAllocator(self.chunk_bytes, n_chunks=self.plan.n_chunks, align=self.align)
        self.phase = ArenaPhase.ATTACHED
        self.attach_seconds = time.perf_counter() - t_start

        # `n_chunks == 0` is the legitimate empty arena (§6.2: an all-device plan must cost nothing).
        # It is skipped explicitly rather than allowed to "pass" a self-test over zero chunks — those
        # are different statements and only one of them is evidence.
        if selftest and self.plan.n_chunks > 0:
            if not first_touch:
                self._write_fingerprints(hip)
            res = self.selftest_light()
            if not res.passed:
                self._rollback(hip)
                raise ArenaSelfTestError(
                    "WEIGHT OFFLOAD: the pinned arena FAILED its out-of-band data check. Every HIP "
                    "call returned success — on this box that proves nothing (Phase 0 found four "
                    "independent cases of the driver reporting success over wrong state). Do NOT "
                    "load weights into these pages.\n"
                    f"  {res.summary()}\n"
                    + "\n".join(f"  {f}" for f in res.failures[:16])
                )

        # ALWAYS, including when the data self-test was switched off for a measurement. It costs
        # microseconds, it is exact rather than sampled, and `selftest=False` is precisely the
        # configuration in which nothing else is watching.
        try:
            self._verify_chunk_structure()
        except ArenaSelfTestError:
            self._rollback(hip)
            raise

    def _verify_chunk_structure(self) -> None:
        """EXACT checks on the pointers themselves. Runs after the data self-test, never instead.

        The data self-test is a SAMPLE: ten 4-byte probes per chunk = 40 B of 2 GiB, i.e. 1.9e-8
        coverage. Two things it therefore cannot be relied on to see, both of which produce plausible
        weights and no crash, are decidable exactly from the pointer table alone:

        1. **OVERLAPPING CHUNKS.** Two chunks whose ranges intersect means two regions ARE the same
           bytes. In the shipping path each component tensor is its own region, so an overlap is
           literally "the w13 stack and the w13_scale stack occupy the same address range": every
           expert then dequantizes against whatever was written last. The fingerprint resweep catches
           a *full* alias (that is `AliasingHip`), but a partial overlap that misses all ten sampled
           offsets reads clean. Sorting the ranges and comparing neighbours is O(n log n) on 17
           entries and is not a sample.

        2. **A CHUNK BASE THAT IS NOT `align`-ALIGNED.** `BumpAllocator` guarantees only that
           OFFSETS are aligned; the address a kernel actually gets is `chunk_base + offset`. If the
           base is not aligned, every region in that chunk is misaligned while every offset still
           "looks" right, and a `float4`/WMMA load off a packed-int4 stack then reads wrong-but-
           well-formed bytes. `hipHostMalloc` returns page-aligned memory in practice — which is
           exactly why this must be asserted rather than assumed: Phase 0's whole finding is that
           this box returns success over wrong state, and an assumption that is true today and
           silent when false is the shape of every bug in `PHASE0_REPORT.md`.

        Both are raised as `ArenaSelfTestError` because to a caller they mean the same thing: the
        pages the driver handed over are not the pages that were asked for. Do not load weights.
        """
        problems: List[str] = []
        a = self.align
        if a > 0:
            for c in self.chunks:
                if c.host_ptr % a or c.device_ptr % a:
                    problems.append(
                        f"chunk {c.index}: base host=0x{c.host_ptr:x} device=0x{c.device_ptr:x} is "
                        f"not {a} B aligned, so EVERY region carved from it is misaligned even "
                        f"though every offset is — a vector load off a packed weight stack then "
                        f"reads wrong-but-well-formed bytes"
                    )
        for kind in ("host_ptr", "device_ptr"):
            spans = sorted((getattr(c, kind), c.nbytes, c.index) for c in self.chunks)
            for (p0, n0, i0), (p1, n1, i1) in zip(spans, spans[1:]):
                if p1 < p0 + n0:
                    problems.append(
                        f"chunks {i0} and {i1} OVERLAP in {kind}: "
                        f"[0x{p0:x}, 0x{p0 + n0:x}) vs [0x{p1:x}, 0x{p1 + n1:x}) — "
                        f"{fmt_bytes(min(p0 + n0, p1 + n1) - p1)} of shared address space. Two "
                        f"regions carved into these chunks are the same bytes, which on a quantized "
                        f"checkpoint means one stack silently overwrites another"
                    )
        if problems:
            raise ArenaSelfTestError(
                "WEIGHT OFFLOAD: the pinned chunk table is structurally impossible. Every HIP call "
                "returned success; the pointers it returned do not describe distinct, aligned "
                "mappings. Do NOT load weights into these pages.\n"
                + "\n".join(f"  {p}" for p in problems[:16])
            )

    def _write_fingerprints(self, hip: Any) -> None:
        """Stamp only the verification offsets (used when `first_touch=False`)."""
        word = ctypes.c_uint32(0)
        for c in self.chunks:
            word.value = c.fingerprint
            for off in verify_offsets(c.nbytes):
                hip.memcpy(c.device_ptr + off, ctypes.addressof(word), 4,
                           hipmem.hipMemcpyHostToDevice)
        hip.sync()

    def _rollback(self, hip: Any) -> None:
        """Give the pages back before raising. A boot that aborts holding 30 GiB of pinned RAM turns
        one failed serve into a box every other agent has to wait out."""
        with hipmem.teardown_window("PinnedWeightArena._rollback"):
            for c in self.chunks:
                try:
                    hip.host_free(c.host_ptr)
                except Exception:  # pragma: no cover - best effort on an already-failing path
                    pass
        self.chunks = []
        self._bump = None
        self._regions = {}
        self._raw_names = []
        # CLOSED, not "still ATTACHED with zero chunks". A caller that catches the exception this
        # rollback precedes would otherwise pass every phase gate on an arena that owns no memory:
        # `selftest_light()` would probe zero chunks, `allocate()` would dereference a None bump
        # (an AssertionError today, and nothing at all under `python -O`). The pages are gone; say so.
        self.phase = ArenaPhase.CLOSED

    # -- step 3: the self-test (phase-gated so it can never eat weights) ------

    def selftest_light(self) -> SelfTestResult:
        """Read every chunk's fingerprint back THROUGH THE DEVICE POINTER, at head/interior/tail.

        Phase-gated to ATTACHED: after `mark_populated()` this would overwrite weights on the write
        side and compare weights against fingerprints on the read side. Plan §5.4 A1.4 requires the
        gate explicitly.

        This is a *resweep* — every chunk is re-read after ALL chunks were written, not right after
        its own write. Verifying a chunk immediately after writing it cannot detect later aliasing:
        if chunk 60's mapping silently reuses chunk 3's physical pages, chunk 60 verifies clean and
        chunk 3 is now corrupt and unexamined.
        """
        self._require(ArenaPhase.ATTACHED, what="selftest_light()")
        hip = self._hip_or_bind()
        t0 = time.perf_counter()
        failures: List[Dict[str, Any]] = []
        offsets = verify_offsets(self.chunk_bytes)
        for c in self.chunks:
            for off in offsets:
                got = hip.read_u32(c.device_ptr + off)
                if got != c.fingerprint:
                    alias = decode_fingerprint(got)
                    failures.append(
                        {
                            "chunk": c.index,
                            "offset": off,
                            "expected": hex(c.fingerprint),
                            "got": hex(got),
                            "looks_like": alias,
                            "note": (
                                "read-back returned another chunk's fingerprint — physical-page "
                                "aliasing" if alias else "read-back returned neither this chunk's "
                                "fingerprint nor any recognisable one"
                            ),
                        }
                    )
        expected = self.plan.n_chunks if self.plan is not None else len(self.chunks)
        if len(self.chunks) != expected:
            failures.append(
                {
                    "chunk": None,
                    "note": (
                        f"the self-test probed {len(self.chunks)} chunk(s) but the plan reserved "
                        f"{expected}. A partially-attached arena must never report PASS — that is "
                        f"the state _rollback() leaves behind"
                    ),
                }
            )
        res = SelfTestResult(
            # A POSITIVE statement, not the absence of failures: probing zero chunks records zero
            # failures, and `passed=not failures` would then print `selftest PASS` in the boot
            # banner over an arena that was never touched.
            passed=(not failures) and len(self.chunks) == expected and expected > 0
            and len(offsets) > 0,
            chunks_checked=len(self.chunks),
            offsets_per_chunk=len(offsets),
            failures=failures,
            seconds=time.perf_counter() - t0,
            expected_chunks=expected,
        )
        self._selftest = res
        return res

    # -- step 4: carve --------------------------------------------------------

    def allocate(self, name: str, nbytes: int, *, align: int | None = None) -> ArenaRegion:
        """Carve one region, and prove on the spot that it landed where `reserve()` said it would.

        Same `BumpAllocator` arithmetic the plan used, so the live offsets are the offsets the unit
        tests exercise — but "same arithmetic" only reproduces the plan if the caller supplies the
        same regions, in the same order, with the same sizes AND the same per-region `align`. Two of
        those three are easy to get wrong silently:

        * `RegionRequest.align` is per-request, while this signature defaults `align=None` -> the
          arena-wide `ALIGN`. A plan built with `RegionRequest(name, n, align=4096)` and carved with
          `allocate(name, n)` places the same region at two different offsets.
        * the walker can enumerate in one order for `reserve()` and another for the carve (a dict
          rebuilt, a dedupe that fires late, an expert-count branch).

        Either way the arena still "works", every pointer is inside a chunk, nothing faults, and
        every region is off by the drift — which on a quantized checkpoint is expert `e`'s nibbles
        dequantized against expert `e'`'s scale: plausible text, no crash, nothing downstream can
        detect it. `verify_matches_plan()` can catch it afterwards, but it is an OPTIONAL call that
        no production caller makes, so the check is done HERE, on the region that drifted, naming
        both placements. Defaulting the align from the plan also removes the first failure mode
        outright rather than reporting it.
        """
        self._require(ArenaPhase.ATTACHED, what="allocate()")
        assert self._bump is not None
        if name in self._regions:
            raise ArenaStateError(
                f"region {name!r} is already carved at chunk {self._regions[name].chunk_index}"
                f"+{self._regions[name].offset}; re-carving would hand two granules the same bytes"
            )
        planned = self.plan.by_name().get(name) if self.plan is not None else None
        if align is None and planned is not None:
            # The PLAN is the authority on this region's alignment. `Placement.align` carries the
            # `RegionRequest.align` that produced the planned offset, so a per-request alignment can
            # never silently degrade to the arena default at carve time.
            align = planned.align
        p = self._bump.allocate(nbytes, name=name, align=align)
        if planned is not None and (planned.chunk_index, planned.offset, planned.nbytes) != (
            p.chunk_index,
            p.offset,
            p.nbytes,
        ):
            raise ArenaStateError(
                f"WEIGHT OFFLOAD: region {name!r} was carved at chunk {p.chunk_index}+{p.offset} "
                f"({p.nbytes} B) but reserve() planned chunk {planned.chunk_index}+"
                f"{planned.offset} ({planned.nbytes} B). The carve order, the sizes or the "
                f"alignments have drifted from the plan (plan digest {self.plan.digest()}). Every "
                f"region from here on is at a different address than planned, which on a quantized "
                f"checkpoint means dequantizing one expert against another's scale — plausible "
                f"text, no crash. Refusing rather than continuing."
            )
        region = self._region_from_placement(p)
        self._regions[name] = region
        return region

    def allocate_raw(self, nbytes: int, *, align: int | None = None) -> Optional[ArenaRegion]:
        """Anonymous carve for the torch `MemPool` callback. Returns None instead of raising.

        Two differences from `allocate()`, both forced by the C ABI on the other side:
        * it is legal in POPULATED as well as ATTACHED, because capture warmup allocates activations
          after the weights are in;
        * it never raises. An exception inside a ctypes callback is swallowed and becomes a NULL
          return, which torch turns into a use-after-null deep in the allocator — a segfault with no
          diagnostic. The caller falls back and counts it instead.
        """
        if self.phase not in (ArenaPhase.ATTACHED, ArenaPhase.POPULATED) or self._bump is None:
            return None
        try:
            p = self._bump.try_allocate(
                int(nbytes), name=f"torch:{len(self._regions)}", align=align
            )
        except ArenaLayoutError:
            # `try_allocate` RAISES `RegionTooLargeError` (not returns None) for `nbytes >
            # chunk_bytes`, so the "never raises" contract above was not actually held: a single
            # torch allocation bigger than one chunk — a fused w13 stack is exactly that shape —
            # propagated an exception out of a ctypes callback. `ArenaMemPool._alloc` happens to
            # catch BaseException today, but a contract that is only kept because one caller is
            # defensive is not kept. Refuse the way exhaustion refuses, and count it identically so
            # the shortfall reporting below sees it.
            p = None
        if p is None:
            # Record the DIAGNOSIS, not just the symptom. The caller (`ArenaMemPool._fallback`) can
            # only report "a hipMalloc happened". There are two reasons and `stats()` distinguishes
            # them: on the enumerated path the row list did not describe what the bake asked for
            # (a component the sizing model does not know about, or torch asking for a bigger
            # segment than `chunk_plan.torch_allocation_bytes` models); on the anonymous-headroom
            # fallback the reservation was sized as ceil(bytes/chunk) with no bound on the region
            # size, so next-fit abandoned a tail per chunk and ran out early. The shortfall and the
            # largest refusal together name either the missing row or the missing
            # `extra_max_region_bytes`.
            self.headroom_denied_bytes += int(nbytes)
            self.headroom_denied_max = max(self.headroom_denied_max, int(nbytes))
            return None
        p = self._attribute_to_forecast(p)
        region = self._region_from_placement(p)
        self._regions[region.name] = region
        self._raw_names.append(region.name)
        return region

    def _attribute_to_forecast(self, p: Placement) -> Placement:
        """Give an anonymous carve the name of the FORECAST row it landed on, when it landed on one.

        The rows of the host tier are reserved by `reserve()` as forecast regions and then carved
        through torch's `MemPool` C ABI, which passes a size and no identity. Matching on the
        coinciding `(chunk_index, offset)` is exact and unspoofable — the forward-only allocator
        produced both numbers from the same request sequence, so they coincide iff the carve is
        following the plan — and it costs one comparison per carve.

        Two things depend on it. `carve_digest()` (the cross-rank layout proof, per its own
        docstring the only digest that is diagnostic at TP=2) starts naming real components instead
        of `torch:0, torch:1, ...`, so two ranks whose walks diverged print different digests for a
        reason an operator can read. And `verify_matches_plan()` gets a coverage number: "the carve
        followed N of the M rows the plan reserved" is a checkable statement where "the plan had
        nothing to compare" was not.

        On a mismatch the carve keeps its `torch:<n>` name and the cursor stops advancing, so a
        drifted carve is reported as zero further attribution rather than being silently re-aligned
        against a row it is not.
        """
        if self.plan is None:
            return p
        forecasts = self.plan.forecast_placements
        if self._forecast_cursor >= len(forecasts):
            return p
        f = forecasts[self._forecast_cursor]
        # Offset AND size. Offset alone is not attribution: the first carve of a run always lands at
        # chunk 0 offset 0 whatever its size, so a walk that enumerated something else entirely
        # would adopt the first row's name and then be reported as "1 of N followed the plan".
        if (f.chunk_index, f.offset, f.nbytes) != (
            p.chunk_index,
            p.offset,
            p.nbytes,
        ) or f.name in self._regions:
            return p
        self._forecast_cursor += 1
        self._forecast_matched += 1
        return Placement(f.name, p.chunk_index, p.offset, p.nbytes, p.align, forecast=True)

    def _region_from_placement(self, p: Placement) -> ArenaRegion:
        c = self.chunks[p.chunk_index]
        return ArenaRegion(
            name=p.name,
            chunk_index=p.chunk_index,
            offset=p.offset,
            nbytes=p.nbytes,
            host_ptr=c.host_ptr + p.offset,
            device_ptr=c.device_ptr + p.offset,
        )

    def region(self, name: str) -> ArenaRegion:
        try:
            return self._regions[name]
        except KeyError:
            raise KeyError(f"no region {name!r} in arena {self.label!r}") from None

    @property
    def regions(self) -> Dict[str, ArenaRegion]:
        return dict(self._regions)

    def owns_pointer(self, ptr: int, nbytes: int = 0) -> bool:
        """Does `[ptr, ptr+nbytes)` lie wholly inside one pinned chunk?

        The ONE residency question that is answerable on this box. `hipPointerGetAttributes` is
        unreliable in BOTH directions here (Phase 0 echoed "Host" for VRAM; P5b reported "Device"
        for the real host arena), so the only trustworthy test is arithmetic against the pointers
        `hipHostGetDevicePointer` actually returned. Used by the torch pool to prove a tensor really
        came from the arena instead of silently landing in VRAM.
        """
        p = int(ptr)
        n = max(0, int(nbytes))
        for c in self.chunks:
            if c.device_ptr <= p and p + n <= c.device_ptr + c.nbytes:
                return True
            if c.host_ptr <= p and p + n <= c.host_ptr + c.nbytes:
                return True
        return False

    def carve_digest(self) -> str:
        """Stable hash of what was ACTUALLY carved, in carve order.

        `ChunkPlan.digest()` is not sufficient as the cross-rank equality proof on the path that
        ships. `bake.py` reserves the host tier as anonymous headroom —
        `arena.reserve([], extra_bytes=plan.host_resident_bytes)` — so `placements` is empty and the
        plan digest collapses to `sha256("chunk_bytes=..;align=..;n=<chunk count>")`. Two ranks whose
        host byte totals round to the same chunk count print IDENTICAL plan digests while their real
        layouts differ, and `verify_matches_plan()` iterates an empty planned table and passes
        vacuously. The layout that matters is produced later, by the ORDER of `allocate_raw` calls
        from the walker — which is exactly the thing that can differ between ranks (a dict iteration
        order, an expert-count branch, a dedupe that fires on one rank only).

        This digest covers that: name, chunk, offset and size of every carved region, in carve order.
        Log it on both ranks and diff the banners; equal digests mean the two ranks placed the same
        bytes at the same offsets in the same chunks.
        """
        h = hashlib.sha256()
        h.update(f"chunk_bytes={self.chunk_bytes};align={self.align};"
                 f"chunks={len(self.chunks)};regions={len(self._regions)}\n".encode())
        for r in self._regions.values():  # dicts preserve insertion (carve) order
            h.update(f"{r.name}\x00{r.chunk_index}\x00{r.offset}\x00{r.nbytes}\n".encode())
        return h.hexdigest()[:16]

    def verification_coverage(self) -> "PlanVerification":
        """What `verify_matches_plan()` is in a position to check. See `PlanVerification`."""
        plan = self.plan
        raw = set(self._raw_names)
        named_planned = 0 if plan is None else len(plan.named_placements)
        named_carved = sum(
            1
            for name in self._regions
            if name not in raw and plan is not None and name in plan.by_name()
        )
        return PlanVerification(
            named_planned=named_planned,
            named_carved=named_carved,
            forecast_planned=0 if plan is None else len(plan.forecast_placements),
            forecast_matched=self._forecast_matched,
            raw_carves=len(self._raw_names),
            planned_footprint=0 if plan is None else plan.footprint_bytes,
            carved_footprint=0 if self._bump is None else self._bump.footprint_bytes,
            planned_payload=0 if plan is None else plan.payload_bytes,
            carved_bytes=sum(r.nbytes for r in self._regions.values()),
            headroom_bytes=0 if plan is None else plan.headroom_bytes,
        )

    def verify_matches_plan(self, *, require_coverage: bool = True) -> "PlanVerification":
        """Every carved region landed where `reserve()` said it would — and SOMETHING was checked.

        Cheap, and it closes a real drift path: if the walker enumerates regions in one order for
        the plan and another for the carve, the arena still "works" and every offset is wrong, which
        on a quantized checkpoint means dequantizing one expert against another's scale — plausible
        text, no crash.

        THIS USED TO PASS OVER THE SHIPPING SHAPE WITHOUT CHECKING ANYTHING, and that is the whole
        reason it had no production caller worth having. `bake.StageARuntime.attach_host_arena`
        reserved the host tier as anonymous headroom (`reserve([], extra_bytes=...)`), so
        `plan.placements` was empty, this method iterated an empty table, and a green result meant
        "no regions were planned" rather than "the layout is right". Now the tier is reserved as
        enumerated FORECAST rows, and there are three real checks:

          1. NAMED regions (`allocate(name, ...)`) must match the plan exactly, and every planned
             one must have been carved — unchanged, and still the strongest check available.
          2. FORECAST regions are checked by ENVELOPE and by ATTRIBUTION, never by name: the carve
             is anonymous (torch's `MemPool` C ABI passes a size and no identity), so the questions
             that can honestly be asked are "did the carve need more arena than was pinned for it"
             (`carved_footprint <= planned_footprint`, monotone in the request sequence, so a carve
             that turns out smaller or fewer — torch splitting a block, a component the walker
             deduped — passes) and "did any of it follow the plan at all"
             (`_attribute_to_forecast`).
          3. `require_coverage` refuses a VACUOUS pass. If the arena carved bytes but this method
             found nothing to compare them against, that is not a pass; it is the absence of a
             check, and reporting it as a pass is what let the defect above survive a green suite.
             Pass `require_coverage=False` only to inspect the coverage of a shape you already know
             is unverifiable.

        Returns the `PlanVerification` so a caller (or a test) can assert on WHAT was checked rather
        than only on the absence of an exception.
        """
        assert self.plan is not None
        planned = {n: p for n, p in self.plan.by_name().items() if not p.forecast}
        raw = set(self._raw_names)
        bad = []
        for name, r in self._regions.items():
            p = planned.get(name)
            if p is None:
                # Exempt only carves this arena actually made through `allocate_raw`. Testing
                # `name.startswith("torch:")` instead would let a GRANULE whose walker-assigned name
                # happens to start with `torch:` skip the plan check entirely — an exemption that a
                # string can spoof is not an exemption, and the thing being skipped is the check
                # that stops one expert's bytes landing on another's.
                if name in raw:
                    continue  # MemPool allocations are anonymous by construction, never planned
                bad.append(f"{name}: carved but not in the plan")
            elif (p.chunk_index, p.offset, p.nbytes) != (r.chunk_index, r.offset, r.nbytes):
                bad.append(
                    f"{name}: planned chunk {p.chunk_index}+{p.offset} ({p.nbytes} B) but carved "
                    f"chunk {r.chunk_index}+{r.offset} ({r.nbytes} B)"
                )
        missing = [n for n in planned if n not in self._regions]

        cov = self.verification_coverage()
        if cov.carved_footprint > cov.planned_footprint and cov.forecast_planned:
            bad.append(
                f"the carve consumed {fmt_bytes(cov.carved_footprint)} of arena but the plan "
                f"reserved {fmt_bytes(cov.planned_footprint)} for it "
                f"(+{fmt_bytes(cov.carved_footprint - cov.planned_footprint)}). The enumerated rows "
                f"are not the rows the bake asked for — either a component the sizing model does "
                f"not know about, or torch asking for a larger segment than "
                f"chunk_plan.torch_allocation_bytes models. The overflow lands in hipMalloc VRAM "
                f"while the capacity plan still calls it host-resident."
            )
        if self.headroom_denied_bytes:
            bad.append(
                f"the arena REFUSED {fmt_bytes(self.headroom_denied_bytes)} of carves (largest "
                f"single refusal {fmt_bytes(self.headroom_denied_max)}); those bytes went to "
                f"hipMalloc VRAM"
            )
        if cov.forecast_planned and cov.forecast_matched == 0 and cov.raw_carves:
            bad.append(
                f"none of the {cov.raw_carves} anonymous carve(s) landed on any of the "
                f"{cov.forecast_planned} forecast row(s). The carve order or the row sizes have "
                f"drifted from the plan, so every offset from the first divergence on is a "
                f"different address than was reserved."
            )
        if require_coverage and cov.vacuous and (cov.carved_bytes or cov.planned_payload):
            bad.append(
                f"NOTHING WAS VERIFIED. {cov.describe()}. A plan with no named and no forecast "
                f"regions cannot be compared against anything, so this call is not evidence the "
                f"layout is right — it is the absence of a check. Reserve the host rows as "
                f"RegionRequests (OffloadPlan.host_row_requests) instead of as anonymous "
                f"`extra_bytes` headroom."
            )
        if bad or missing:
            raise ArenaStateError(
                "WEIGHT OFFLOAD: carved layout does not match the reserved plan "
                f"(digest {self.plan.digest()}, carve {self.carve_digest()}).\n"
                + "\n".join(f"  {b}" for b in bad)
                + (f"\n  planned but never carved: {missing[:16]}" if missing else "")
            )
        return cov

    # -- step 5: lock down ----------------------------------------------------

    def mark_populated(self, *, allow_fallbacks: bool = False) -> None:
        """Weights are in. The self-test is locked out from here — it writes fingerprints.

        ENFORCES `torch_fallbacks == 0`, which `__init__` has always documented as a MUST and which
        nothing checked. `ArenaMemPool.assert_clean()` exists to make this assertion and has no
        caller: `StageARuntime.freeze()` calls `mark_populated()` and never touches the pool. So on
        the shipping path a fallback was counted, printed in `stats()` if anyone looked, and
        otherwise ignored.

        It is not ignorable. Every fallback is a `hipMalloc`, i.e. bytes that the plan budgeted as
        host-resident which actually landed in VRAM. Three things then quietly stop being true:
        the capacity verdict (it was computed against `plan.reserved_bytes`), the device-tier
        fraction the operating point was chosen at, and — because the bind happens inside the window
        `_determine_num_pages` measures as `old_free - new_free` — the KV pool is sized against a
        model term that is short by exactly those bytes. The failure surfaces much later as an OOM
        with an unrelated message. `mark_populated()` is the last moment at which the diagnosis is
        still one line long.

        `allow_fallbacks=True` exists for a deliberate measurement leg that wants the run to finish;
        it is never a production setting.
        """
        self._require(ArenaPhase.ATTACHED, what="mark_populated()")
        if self.torch_fallbacks and not allow_fallbacks:
            denied = (
                f" The arena refused {fmt_bytes(self.headroom_denied_bytes)} of carves (largest "
                f"single refusal {fmt_bytes(self.headroom_denied_max)}), so reserve() was too "
                f"small: either the enumerated rows "
                f"({len(self.plan.forecast_placements) if self.plan else 0} forecast) do not "
                f"describe what the bake asked for, or the anonymous-headroom fallback was sized "
                f"as the payload total with no allowance for the tails next-fit abandons at every "
                f"chunk boundary."
                if self.headroom_denied_bytes
                else ""
            )
            raise ArenaStateError(
                f"WEIGHT OFFLOAD: {self.torch_fallbacks} allocation(s) fell back to hipMalloc while "
                f"populating arena {self.label!r}, so bytes budgeted as host-resident are in VRAM. "
                f"The capacity verdict, the device-tier fraction and the KV pool sizing that "
                f"follows are all now computed from a number that is not true.{denied} FIX: raise "
                f"the reserved headroom. Pass allow_fallbacks=True only for a measurement leg."
            )
        self.phase = ArenaPhase.POPULATED

    def freeze(self, reason: str = "weight arena populated") -> None:
        """Rule R1: no mapping entry point may be called after this."""
        self._require(ArenaPhase.ATTACHED, ArenaPhase.POPULATED, what="freeze()")
        if self.phase is ArenaPhase.ATTACHED:
            self.phase = ArenaPhase.POPULATED
        hipmem.freeze(reason)

    def close(self, *, force: bool = False) -> None:
        """Release every chunk. Shutdown and tests only.

        Freeing while a torch tensor still points into the arena is the `c10::Error: invalid device
        pointer` abort P5b exists to avoid, so this is never called implicitly — no `__del__` frees.

        REFUSES once the mapping window is frozen unless `force=True`. Freeze marks the point after
        which weights are live: captured HIP graphs have baked these exact device pointers into their
        recorded kernel arguments (that address stability is the ONLY reason the arena is
        capture-safe at all — the graph never re-resolves a pointer, it replays the one it captured).
        `hipHostFree` here does not raise and does not invalidate the graph; the next `replay()`
        simply reads unmapped pages, and the failure surfaces as a GPU memory fault with no
        attribution to the shutdown path that caused it. A shutdown handler, an `atexit`, or a test
        fixture reaching for `close()` is precisely how that gets called by accident, so the guard
        makes it deliberate rather than documented.
        """
        if self.phase is ArenaPhase.CLOSED:
            return
        if hipmem.is_frozen() and not force:
            raise ArenaStateError(
                f"close() on arena {self.label!r} after the mapping window was frozen. Captured "
                f"HIP graphs hold these device pointers verbatim in their recorded kernel "
                f"arguments; freeing the chunks turns the next graph replay into an unattributed "
                f"GPU memory fault. Pass force=True only if you know no captured graph and no live "
                f"tensor references the arena (process teardown after the model is gone)."
            )
        if self.chunks:
            hip = self._hip_or_bind()
            with hipmem.teardown_window("PinnedWeightArena.close"):
                for c in self.chunks:
                    hip.host_free(c.host_ptr)
        self.chunks = []
        self._bump = None
        self._regions = {}
        self.phase = ArenaPhase.CLOSED

    # -- observability --------------------------------------------------------

    @property
    def pinned_bytes(self) -> int:
        return sum(c.nbytes for c in self.chunks)

    @property
    def carved_bytes(self) -> int:
        return sum(r.nbytes for r in self._regions.values())

    def stats(self) -> Dict[str, Any]:
        pin_rates = [c.pin_gb_s for c in self.chunks if c.pin_gb_s]
        return {
            "label": self.label,
            "phase": self.phase.value,
            "rank": self.rank,
            "local_ranks": self.local_ranks,
            "device": self.device_identity or {"hip_device": self.device_index},
            "chunk_bytes": self.chunk_bytes,
            "n_chunks": len(self.chunks),
            "pinned_bytes": self.pinned_bytes,
            "carved_bytes": self.carved_bytes,
            "n_regions": len(self._regions),
            "attach_seconds": round(self.attach_seconds, 3),
            "pin_gb_s_median": round(statistics.median(pin_rates), 3) if pin_rates else None,
            "same_va": all(c.same_va for c in self.chunks) if self.chunks else None,
            "plan": self.plan.describe() if self.plan else None,
            "plan_digest": self.plan.digest() if self.plan else None,
            # The digest that is actually diagnostic at TP=2 — see carve_digest()'s docstring for
            # why plan_digest alone cannot see a rank divergence on the shipping path.
            "carve_digest": self.carve_digest(),
            "capacity": self.capacity.summary() if self.capacity else None,
            "capacity_advisories": list(self.capacity.advisories) if self.capacity else [],
            "packing_advisory": self.packing_advisory,
            "headroom_advisory": self.headroom_advisory,
            "chunk_growth_advisory": self.chunk_growth_advisory,
            "headroom_denied_bytes": self.headroom_denied_bytes,
            "headroom_denied_max": self.headroom_denied_max,
            # What `verify_matches_plan()` is in a position to check. Printed because a reservation
            # that fell back to anonymous headroom verifies NOTHING, and that has to be visible in
            # the boot log rather than discovered by reading the reserve() call site.
            "verification": self.verification_coverage().describe() if self.plan else None,
            "selftest": self._selftest.summary() if self._selftest else None,
            "torch_hipmalloc_fallbacks": self.torch_fallbacks,
            "frozen": hipmem.is_frozen(),
        }

    def summary(self) -> str:
        """One line for the `[serve]` banner. Includes the digest so two ranks' banners can be
        diffed, and the pinned total so an operator can sanity-check it against `free -g`."""
        s = self.stats()
        return (
            f"[weight-arena:{self.label}] rank={self.rank}/{self.local_ranks} "
            f"dev={s['device'].get('pci_bus_id', self.device_index)} phase={s['phase']} "
            f"pinned={fmt_bytes(self.pinned_bytes)} in {len(self.chunks)}×"
            f"{fmt_bytes(self.chunk_bytes)} carved={fmt_bytes(self.carved_bytes)} "
            f"regions={len(self._regions)} attach={self.attach_seconds:.1f}s "
            f"pin={s['pin_gb_s_median']} GB/s digest={s['plan_digest']} "
            f"carve={s['carve_digest']} "
            f"selftest={s['selftest']} fallbacks={self.torch_fallbacks}"
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"<PinnedWeightArena {self.label} phase={self.phase.value} "
            f"chunks={len(self.chunks)} pinned={fmt_bytes(self.pinned_bytes)}>"
        )
