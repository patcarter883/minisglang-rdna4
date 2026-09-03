"""VRAM accounting for the weight-offload arena — and the assertions that keep it honest.

WHY THIS FILE IS TORCH-FREE. Every number here is an integer byte count. Keeping `import torch` out
means the arithmetic — the part that, if wrong, silently costs ~100k KV tokens — is unit-testable
anywhere. The caller (`weights/bake.py`, driven by `Engine.__init__`) takes the `mem_get_info` /
`memory_allocated` / `memory_reserved` samples and hands the raw integers in.

WHAT THE ACCOUNTING HAS TO GET RIGHT (`docs/WEIGHT_OFFLOAD_PLAN.md` §5.3)

`Engine._determine_num_pages` bills the model as

    device_used  = old_free - new_free                       # driver-visible draw across the load
    model_memory = memory_allocated() + max(0, device_used - memory_reserved())

and then subtracts five reservations from `memory_ratio * old_free`. The plan's rule is **no sixth
subtrahend**: the device tier is allocated inside the measured window, so it is already inside
`device_used`, and reserving for it again double-subtracts and at a 16 GB tier sizes the pool
negative. Under layer-granular placement the device tier is not even a new allocation — a
device-resident layer is simply left exactly where `load_state_dict` put it — so there is nothing
extra to bill at all.

What DOES need correcting is the opposite mistake. P5b validated the host arena through torch's
`_cuda_customAllocator` + `MemPool` (`weights/torch_pool.py`), and a pool-served tensor over
`hipHostGetDevicePointer` pages is an ordinary `cuda` tensor to `memory_allocated()`. Left alone,
tens of GB of HOST RAM would be billed as device memory, `available_memory` would go hard negative
and the engine would refuse to boot. So one correction, and it is MEASURED rather than predicted:
the growth in `allocated`/`reserved` across the bake, clamped to the bytes actually copied. When the
arena is invisible to the torch allocator the correction is exactly 0 and every serve that does not
offload is byte-identical.

THE CHECKS, AND THE FAILURE EACH ONE CATCHES

  1. `host arena costs 0 device bytes` — Phase 0 found `hipMemCreate(location=Host)` silently
     returning **VRAM** while the properties query echoed "Host" back verbatim (three independent
     probes), and P5b found `hipPointerGetAttributes` reporting `memory_type=Device` for a real host
     arena. **Both queries lie, in both directions.** This check does not ask the driver what the
     memory is; it asks the card how much of itself is left, across arena attach. It is the single
     highest-value assertion in the feature.
  2. `bytes copied == plan.host_resident_bytes` — the bake moved exactly what the plan priced. A
     mismatch means the KV pool is about to be sized against a plan that is not what happened.
  3. `device tier == plan.device_resident_bytes` — the device-resident bytes MEASURED off the live
     post-`post_load()` containers (`observed_device_bytes`), not an allocator statistic, so caching,
     fragmentation and pool bookkeeping cannot fool it. It must be a measurement: the older
     `offloadable_total - copied` form is algebraically `host_bytes - copied` (the plan guarantees
     `total = host + device`), i.e. check 2 restated, and could not catch the one failure this check
     exists for — a config-time byte model that disagrees with what `post_load()` really produced.
     `plan.WeightPlanResolution.assert_device_accounting` gates the same number at `seal()`, which is
     the boot assertion plan §5.3 requires as the price of having no sixth subtrahend.
  4. `torch released the originals` — `memory_allocated()` fell by the copied bytes, net of whatever
     the arena itself added. This is the only signal that catches a MISSED ALIAS: if a repack left
     two names on one storage (`_GroupedFP8Experts.post_load` sets `_w_op = weight.view(uint8)` and,
     under `MINISGL_ZAYA_OLDMOE=1`, keeps `weight`) and the rebind moved only one of them, the copy
     succeeds, the read-back passes, and VRAM simply never comes back down. Netting out
     `arena_torch_bytes` is what makes it visible: without that term the arena's own rows and the
     dropped originals cancel exactly whenever the arena is pool-served, and a completely undropped
     original reads as perfect.

A failing report prints the whole sample table, because the useful diagnostic is always "which of
the deltas moved".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------------------------
# Tolerance for the byte-agreement checks. The arena is chunked and page-aligned, the driver rounds
# allocations up, and `memory_allocated()` moves under any concurrent allocation, so exact equality
# would be a flake generator. 64 MiB is ~0.1% of a 50 GB arena and far below the smallest thing that
# could hide here (one 2 GiB backing chunk, or any single layer's granules).
#
# NOTE: the measured per-card host bandwidths, the pinned-host ceiling and the compute floor live in
# `weights/prior.OffloadPrior` — ONE frozen, provenance-carrying table for the whole feature.
# Duplicating 28.93/14.48 here is exactly how a planner starts projecting the FAST card's number on
# the rank that owns the slow one.
# ---------------------------------------------------------------------------------------------
DEFAULT_TOL_BYTES: int = 64 << 20


def gib(size: int) -> str:
    """`engine.graph.mem_GB`'s format, reimplemented here only so this module stays torch-free.

    Kept byte-identical (`f"{x:.2f} GiB"`) so a boot log mixing the two reads as one line."""
    return f"{size / (1024**3):.2f} GiB"


@dataclass(frozen=True)
class MemSample:
    """One `(free, allocated, reserved)` triple, taken at a named point in `Engine.__init__`.

    `free` is `torch.cuda.mem_get_info(device)[0]` — the DRIVER's own number, which is the only
    residency signal Phase 0 found trustworthy. `allocated`/`reserved` are torch's, which the arena
    may or may not perturb; that is measured below, never assumed."""

    tag: str
    free: int
    allocated: int
    reserved: int


@dataclass(frozen=True)
class AccountingCheck:
    name: str
    ok: bool
    actual: int
    expected: int
    tol: int
    detail: str

    def line(self) -> str:
        return (
            f"  [{'ok  ' if self.ok else 'FAIL'}] {self.name}: "
            f"actual={gib(self.actual)} expected={gib(self.expected)} "
            f"(tol {gib(self.tol)}) — {self.detail}"
        )


@dataclass(frozen=True)
class AccountingReport:
    checks: Tuple[AccountingCheck, ...]
    samples: Tuple[MemSample, ...]
    alloc_correction: int
    reserved_correction: int

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks)

    @property
    def failures(self) -> Tuple[AccountingCheck, ...]:
        return tuple(c for c in self.checks if not c.ok)

    def render(self) -> str:
        rows = [
            f"    {s.tag:<12} free={gib(s.free)} alloc={gib(s.allocated)} res={gib(s.reserved)}"
            for s in self.samples
        ]
        return "\n".join(
            ["weight-arena accounting:"]
            + [c.line() for c in self.checks]
            + ["  samples:"]
            + rows
            + [
                f"  model_memory correction: allocated -{gib(self.alloc_correction)} "
                f"reserved -{gib(self.reserved_correction)}"
            ]
        )


@dataclass
class WeightArenaAccounting:
    """The byte ledger across the Stage-A window.

    Sample points, in the order `bake.StageASession` takes them:

        pre_attach   -- immediately before the pinned host arena is attached (still pre-load)
        post_attach  -- immediately after it; `free` MUST NOT have moved
        pre_bake     -- after post_load(), immediately before the first granule copy
        post_bake    -- after every copy, the read-back self-test, the rebind and the drop

    `host_bytes` / `device_bytes` are the PLAN's numbers (`placement.OffloadPlan
    .host_resident_bytes` / `.device_resident_bytes`); `copied_bytes` and `offloadable_bytes` are
    what the bake actually did. The checks are the comparison.

    Nothing here allocates, frees or queries anything. It is a ledger.
    """

    host_bytes: int = 0
    device_bytes: int = 0
    offloadable_bytes: int = 0
    copied_bytes: int = 0
    # Bytes the arena handed to torch during the bake, as reported by the arena/pool itself. Used to
    # separate "the originals were released" from "the arena rows were added" in a single
    # `memory_allocated()` delta, which otherwise nets to zero when the arena is pool-served.
    arena_torch_bytes: int = 0
    # Device-tier bytes MEASURED off the live post-load containers (`moe_interpose.BindOutcome
    # .device_resident_bytes`), or None when no such measurement exists. See `device_tier_bytes`
    # for why the plan-derived fallback is not a measurement.
    #
    # `kw_only` so the positional arity of this constructor does not move. The window's ORDER guard
    # (`test_order_is_a_classvar_not_a_field`) works by proving one-too-many positional arguments
    # raises; a new positional field would silently absorb that extra argument and retire the guard.
    observed_device_bytes: Optional[int] = field(default=None, kw_only=True)
    tol_bytes: int = DEFAULT_TOL_BYTES
    _samples: Dict[str, MemSample] = field(default_factory=dict)
    _order: List[str] = field(default_factory=list)

    # ClassVar, NOT a field: a bare `ORDER: Tuple[...] = (...)` inside a @dataclass silently becomes
    # a per-instance field with a default, and the constructor would then accept an extra positional
    # argument that reorders the window.
    ORDER: ClassVar[Tuple[str, ...]] = ("pre_attach", "post_attach", "pre_bake", "post_bake")

    # -- ledger -------------------------------------------------------------------------------

    def sample(self, tag: str, *, free: int, allocated: int, reserved: int) -> MemSample:
        """Record one triple. Tags are fixed (`ORDER`) and must arrive in order.

        The ordering check is not pedantry: every derived quantity below is a DIFFERENCE between two
        named samples, so a sample taken at the wrong point yields a plausible number rather than an
        error — which is the exact failure mode this module exists to make impossible."""
        if tag not in self.ORDER:
            raise ValueError(f"unknown accounting sample {tag!r}; expected one of {self.ORDER}")
        if tag in self._samples:
            raise RuntimeError(f"accounting sample {tag!r} taken twice")
        if self.ORDER.index(tag) < len(self._order):
            raise RuntimeError(
                f"accounting sample {tag!r} taken after {self._order[-1]!r}; the Stage-A window is "
                f"ordered {self.ORDER} and every delta is a difference between two of them"
            )
        s = MemSample(tag, int(free), int(allocated), int(reserved))
        self._samples[tag] = s
        self._order.append(tag)
        return s

    def has(self, tag: str) -> bool:
        return tag in self._samples

    def get(self, tag: str) -> Optional[MemSample]:
        return self._samples.get(tag)

    def _delta(self, a: str, b: str, field_name: str) -> int:
        sa, sb = self._samples.get(a), self._samples.get(b)
        if sa is None or sb is None:
            return 0
        return getattr(sb, field_name) - getattr(sa, field_name)

    # -- derived quantities -------------------------------------------------------------------

    @property
    def host_arena_device_cost(self) -> int:
        """Device bytes the HOST arena consumed. MUST be ~0 — check 1 in the module docstring."""
        return -self._delta("pre_attach", "post_attach", "free")

    @property
    def device_tier_bytes(self) -> int:
        """Offloadable bytes left resident on the card, from the enumerated granules.

        Under layer-granular placement a device-resident layer is never reallocated (it stays
        exactly where `load_state_dict` put it), so there is no allocation event to observe — but
        there IS an exact byte count, and it must equal what the plan budgeted.

        `observed_device_bytes` is that count, summed over the LIVE post-load containers of the
        device-placed seams. It is the only form of this check that carries information. The
        `offloadable - copied` fallback below is kept for a caller that cannot measure (a test
        double, a driver without seams) but it is NOT a measurement: the plan guarantees
        `total = host + device`, so `total - copied` reduces to `host - copied`, which is the
        "bake moved plan.host_resident_bytes" check restated. Two checks, one fact — and the one
        thing neither of them could catch was a plan whose byte model disagrees with what
        `post_load()` really produced (`sizing.analytic_gemm_bytes` under-counts MXFP4's E8M0 ->
        fp16 scale widening by 2x today, which is exactly that failure).
        """
        if self.observed_device_bytes is not None:
            return max(0, int(self.observed_device_bytes))
        return max(0, self.offloadable_bytes - self.copied_bytes)

    @property
    def device_tier_is_measured(self) -> bool:
        """Whether `device_tier_bytes` came from the live containers or from plan arithmetic."""
        return self.observed_device_bytes is not None

    @property
    def alloc_delta_bake(self) -> int:
        """`memory_allocated()` change across the bake: arena rows added, originals dropped."""
        return self._delta("pre_bake", "post_bake", "allocated")

    @property
    def reserved_delta_bake(self) -> int:
        return self._delta("pre_bake", "post_bake", "reserved")

    @property
    def originals_released(self) -> int:
        """Torch-allocated bytes the DROP returned, net of what the arena itself added.

        `Δallocated = arena_rows_added - originals_released`, and `arena_rows_added` is reported by
        the arena (`arena_torch_bytes`, 0 when the rows are not pool-served), so the drop is
        recoverable from a single delta. Without that term the two movements cancel exactly whenever
        the arena IS pool-served, and a completely un-dropped original would look perfect."""
        return self.arena_torch_bytes - self.alloc_delta_bake

    @property
    def alloc_correction(self) -> int:
        """HOST-arena bytes `torch.cuda.memory_allocated()` is counting as device memory.

        Clamped to `[0, copied_bytes]`: an unclamped figure would also absorb anything else that
        allocated in the same window (nothing should, but a clamp turns a surprise into a bounded
        error rather than a negative KV pool). 0 when the arena is invisible to the torch allocator,
        which makes the whole correction a no-op.

        THAT LAST SENTENCE IS ONLY TRUE IF `arena_torch_bytes` IS MEASURED OFF TORCH. It is the
        driver's job to supply it that way, and `StageARuntime` used to supply the STACK allocator's
        handed-out total instead — which equals `copied_bytes` by construction no matter what torch
        did, so the "invisible to the torch allocator" case produced a full-size correction against a
        reading that never contained the bytes. `model_memory` then under-bills the model,
        `available_memory` is over-stated, and the KV pool is sized against VRAM that is gone; the
        failure lands on the first forward, not at boot. It is now `ArenaMemPool.served_bytes`."""
        return max(0, min(self.copied_bytes, self.arena_torch_bytes))

    @property
    def reserved_correction(self) -> int:
        """Same, for `memory_reserved()`. Applied to the non-torch term of `model_memory`.

        Measured from the reserved delta rather than reused from `alloc_correction`: torch reserves
        in segments, so the two are not equal, and reusing one for the other would mis-state the
        non-torch remainder (HIP context, kernel code objects) in whichever direction it is off.

        THE CAP IS THE ARENA'S TORCH-RESERVED FOOTPRINT, NOT `copied_bytes`. Torch reserves whole
        segments, so a pool-served arena shows up in `memory_reserved()` as `copied + pad` where the
        pad is that segment rounding. Capping the correction at `copied` alone leaves the pad behind
        in `reserved`, and `model_memory`'s non-torch term is
        `max(0, device_used - (reserved - correction))` — so an uncorrected HOST-memory pad is
        subtracted from a DEVICE-memory total, under-billing the HIP context by the pad and handing
        the KV pool that many bytes it does not have. The pad is not an unknown: it is exactly
        `torch_slack_bytes` (reserved-not-allocated arena bytes, the same quantity the scheduler's
        prefill guard subtracts), so the cap is `copied + slack` and no constant is invented. Still
        exactly 0 when the arena is invisible to the torch allocator, where `slack` is 0 too.
        """
        cap = self.copied_bytes + self.torch_slack_bytes
        return max(0, min(cap, self.reserved_delta_bake))

    @property
    def arena_visible_to_torch(self) -> bool:
        """Whether the arena shows up in torch's allocator stats (P5b's `MemPool` route does).

        Reported on the boot banner: it decides whether `model_memory` needs correcting at all, and
        an operator reading a KV-sizing line has to be able to tell which regime they are in."""
        return self.alloc_correction > self.tol_bytes

    @property
    def torch_slack_bytes(self) -> int:
        """Arena bytes torch has RESERVED but not ALLOCATED.

        `Scheduler._prefill_budget_now` computes affordable prefill tokens from
        `mem_get_info().free + memory_reserved() - memory_allocated()`, on the sound reasoning that
        the allocator's free cache is memory the next forward reuses without a new device mapping.
        Arena segments break that reasoning: they are reserved, not allocated, and NOT reusable — a
        host-backed segment is not device memory at all. Counting them inflates the guard's idea of
        free VRAM by up to the whole arena and turns a protective clamp into an OOM.

        0 whenever the arena is invisible to the torch allocator, so the guard is unchanged on every
        serve that does not offload."""
        return max(0, self.reserved_delta_bake - self.alloc_delta_bake - self.copied_bytes)

    def model_memory_correction(self) -> Tuple[int, int]:
        """`(allocated, reserved)` bytes to remove from `_determine_num_pages`'s model term.

        Subtract BOTH: `model_memory` is `allocated + max(0, device_used - reserved)`, so correcting
        `allocated` alone while leaving an inflated `reserved` would drive the non-torch term to 0
        and quietly under-bill the HIP context and kernel code objects.

        This is NOT the sixth subtrahend the plan forbids. Nothing extra is reserved; a host-RAM
        number is being removed from a device-memory total."""
        return self.alloc_correction, self.reserved_correction

    # -- the gate -----------------------------------------------------------------------------

    def report(self) -> AccountingReport:
        tol = self.tol_bytes
        checks = [
            AccountingCheck(
                name="host arena costs 0 device bytes",
                ok=abs(self.host_arena_device_cost) <= tol,
                actual=self.host_arena_device_cost,
                expected=0,
                tol=tol,
                detail=(
                    "free-VRAM delta across arena attach. Phase 0: hipMemCreate(location=Host) "
                    "silently returned VRAM with the property echoed back, and "
                    "hipPointerGetAttributes reports 'Device' for real host pages — neither query "
                    "is trusted, so this asks the card how much of itself is left"
                ),
            ),
            AccountingCheck(
                name="bake moved plan.host_resident_bytes",
                ok=abs(self.copied_bytes - self.host_bytes) <= tol,
                actual=self.copied_bytes,
                expected=self.host_bytes,
                tol=tol,
                detail=(
                    "bytes actually copied into the arena vs what the plan priced; a mismatch means "
                    "the KV pool is about to be sized against a plan that is not what happened"
                ),
            ),
            AccountingCheck(
                name="device tier == plan.device_resident_bytes",
                ok=abs(self.device_tier_bytes - self.device_bytes) <= tol,
                actual=self.device_tier_bytes,
                expected=self.device_bytes,
                tol=tol,
                detail=(
                    "device-resident bytes "
                    + (
                        "MEASURED off the live post-load containers"
                        if self.device_tier_is_measured
                        else "DERIVED from the plan (no live measurement available — this check "
                        "carries no independent information in that state)"
                    )
                    + ", not from an allocator statistic, so caching and fragmentation cannot fool "
                    "it. A mismatch means the config-time byte model and post_load() disagree and "
                    "the device tier is billed wrong inside model_memory"
                ),
            ),
            AccountingCheck(
                name="torch released the originals",
                ok=abs(self.originals_released - self.copied_bytes) <= tol,
                actual=self.originals_released,
                expected=self.copied_bytes,
                tol=tol,
                detail=(
                    "allocator delta across the bake, net of the arena rows it added. Short means a "
                    "live reference to a pre-offload weight survived and peak VRAM never comes down"
                ),
            ),
        ]
        return AccountingReport(
            checks=tuple(checks),
            samples=tuple(self._samples[t] for t in self._order),
            alloc_correction=self.alloc_correction,
            reserved_correction=self.reserved_correction,
        )


# ---------------------------------------------------------------------------------------------
# The one line of KV-sizing arithmetic the correction touches, lifted out of `Engine` so it is
# torch-free and therefore testable. This is the number that, if wrong, silently costs ~100k KV
# tokens or sizes a pool that OOMs on the first forward.
# ---------------------------------------------------------------------------------------------


def corrected_model_memory(
    *,
    allocated: int,
    reserved: int,
    device_used: int,
    alloc_correction: int,
    reserved_correction: int,
) -> Tuple[int, int, int]:
    """`Engine._determine_num_pages`'s model term, with the host-arena correction applied SAFELY.

    Returns `(model_memory, applied_alloc_correction, applied_reserved_correction)`; the two applied
    figures are what the boot log must print, because they can differ from what was requested.

    THE CLAMP IS NOT DEFENSIVE PROGRAMMING. `alloc_correction`/`reserved_correction` are deltas
    sampled during the Stage-A window (`seal()`), but `allocated`/`reserved` are read LATER, and
    `Engine._sync_get_memory()` calls `torch.cuda.empty_cache()` in between. `empty_cache` releases
    the segments the bake's dropped originals left behind, so `reserved` can legitimately fall by
    roughly the arena size between the sample and the read. Two failures follow from applying an
    unclamped delta to a shrunken reading, and they fail in OPPOSITE directions, which is why both
    ends are bounded:

      * `reserved - reserved_correction` goes negative, so `max(0, device_used - reserved)` — the
        non-torch remainder — is inflated by up to the whole arena. `available_memory` goes hard
        negative and the boot is refused with a message naming four unrelated causes.
      * `allocated - alloc_correction` goes negative, so `model_memory` under-bills the model,
        `available_memory` is over-stated and the KV pool is sized against VRAM that is gone. That
        one does not fail at boot; it fails on the first forward.

    A clamp cannot mask a real fault: the ledger's own gate at `seal()` already refused to reach
    this point if the arena and the plan disagreed about bytes. It only stops a stale delta from
    becoming a wrong pool size.

    With both corrections 0 (every serve that does not offload) this is byte-identical to the
    original `memory_allocated() + max(0, device_used - memory_reserved())`.
    """
    applied_alloc = max(0, min(int(alloc_correction), int(allocated)))
    applied_reserved = max(0, min(int(reserved_correction), int(reserved)))
    model_memory = max(0, int(allocated) - applied_alloc) + max(
        0, int(device_used) - (int(reserved) - applied_reserved)
    )
    return model_memory, applied_alloc, applied_reserved


# ---------------------------------------------------------------------------------------------
# The KV-sizing annotation (plan §5.3): one line an operator can read without re-deriving anything.
# ---------------------------------------------------------------------------------------------


def step_floor_ms(host_bytes_per_forward: int, host_read_gb_s: float) -> float:
    """Lower bound on forward latency imposed purely by streaming host-resident weights.

    P2prime measured the miss cost as EXACTLY linear in bytes at full PCIe rate — the marginal cost
    of one host-resident expert is one granule at the card's independently measured host bandwidth
    (102.8% of it on card 0, 101.3% on card 1) — so there is no cliff term, no concurrency term and
    no residency-probability term. The floor really is bytes / bandwidth.

    GB here is the vendor 1e9 GB the bandwidth figures are quoted in, NOT GiB; mixing the two is a
    silent 7% error."""
    if host_bytes_per_forward <= 0 or host_read_gb_s <= 0:
        return 0.0
    return (host_bytes_per_forward / 1e9) / host_read_gb_s * 1e3


def kv_sizing_annotation(
    *,
    device_bytes: int,
    host_bytes: int,
    tp_size: int,
    host_bytes_per_forward: int = 0,
    host_read_gb_s: float = 0.0,
) -> str:
    """The `weight-arena=...` suffix appended to the single `KV sizing:` line.

    Returns "" when there is no arena, so the existing line is byte-identical on every serve that
    does not offload. The step-floor clause is OMITTED, not guessed, when the caller has no measured
    per-forward host byte count or no bandwidth for this rank — a confident wrong number is worse
    than no number.

    Callers at tp>=2 must pass the SLOW rank's bandwidth (`prior.OffloadPrior.slow_host_gbps`): the
    two links are independent (P4, efficiency 0.999) so the slower rank sets the step, and quoting
    card 0's figure for the pair is wrong by ~2x."""
    if device_bytes <= 0 and host_bytes <= 0:
        return ""
    out = (
        f" weight-arena={gib(device_bytes)} (dev tier, inside model) "
        f"host={gib(host_bytes)}x tp{tp_size}"
    )
    ms = step_floor_ms(host_bytes_per_forward, host_read_gb_s)
    if ms > 0:
        out += f"; step floor {ms:.1f} ms -> <= {1e3 / ms:.1f} forwards/s"
    return out
