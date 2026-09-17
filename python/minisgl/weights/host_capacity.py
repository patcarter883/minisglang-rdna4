"""Host-RAM capacity accounting for the pinned weight arena — the LOUD, EARLY failure.

WHY THIS FILE IS THE MOST IMPORTANT ONE IN THE MODULE. Phase 0's binding constraint is **capacity,
not bandwidth** (`PHASE0_REPORT.md` §3.4). On an *idle* box with **no engine loaded**, P3b pinned
34.0 GiB for one rank but only **62.0 GiB across two** (34.0 + 28.0) — rank 1 stopped on the
`MemAvailable` floor with **114,813 pages swapped out**. The target model wants 68.8 GiB. So the
default expectation is that a naive all-host arena **does not fit**, and the difference between a
good failure and a terrible one is *when* it is detected:

* **terrible:** allocate lazily during `load_state_dict`, discover the shortfall 40 GiB and several
  minutes in, having already pushed the box into swap thrash and taken the page cache with it;
* **good:** compute the required bytes from the plan, compare against `MemAvailable` **before the
  first `hipHostMalloc`**, and abort in milliseconds with the numbers and the fixes printed.

This file only does the second. It is pure integer + `/proc` + `/sys/fs/cgroup` reading; no torch, no
HIP. The cgroup term matters because this repo ships as a container and `/proc/meminfo` is not
namespaced — see `mem_available_bytes`.

THREE RULES THAT ARE NOT NEGOTIABLE.

1. **NEVER auto-shrink.** The obvious "helpful" behaviour — notice the arena does not fit and
   quietly reserve less — is a TP-desync bug. `MemAvailable` is timing-dependent and differs between
   the rank processes, so rank 0 could shrink and rank 1 not; the two would then place granules at
   different offsets and the collectives would hang, or worse, serve different weights. The capacity
   check is therefore strictly **pass or raise**; the *plan* is a pure function of the region list
   (see `chunk_plan.py`) and is never influenced by a runtime measurement.
2. **Charge for every local rank.** At TP=2 both ranks pin from the *same* host RAM. A per-rank
   check passes twice and the box still dies. `local_ranks` multiplies the requirement.
3. **A floor, not zero.** Pinned pages are unevictable; driving `MemAvailable` to zero takes the
   page cache, the engine's own host allocations and the tokenizer process with it. P3b ran with a
   **12 GiB** floor and that is the default here, so the 62 GiB ceiling this module quotes was
   measured under the policy this module enforces.

`hipHostMalloc` failure is NOT a reliable signal to lean on: P3b's rank 1 never got a non-zero
return code — it was stopped by *this* check. And the box was already swapping by then.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

from .chunk_plan import GIB, fmt_bytes

MEMINFO_PATH = "/proc/meminfo"
VMSTAT_PATH = "/proc/vmstat"

# `/proc/meminfo` is NOT namespaced: inside a container it reports the HOST's memory, so a serve run
# under `docker run --memory=32g` on a 96 GiB box reads MemAvailable ~= 80 GiB, sails through this
# gate, and is OOM-killed by the cgroup partway through pinning — with no message from this module at
# all. This repo *ships as a container* (the Dockerfile IS the serve image), so the cgroup limit is
# the operative ceiling far more often than the host's. Both hierarchies are read because ROCm images
# still turn up on v1 hosts.
CGROUP_V2_MAX = "/sys/fs/cgroup/memory.max"
CGROUP_V2_CURRENT = "/sys/fs/cgroup/memory.current"
CGROUP_V2_STAT = "/sys/fs/cgroup/memory.stat"
CGROUP_V1_LIMIT = "/sys/fs/cgroup/memory/memory.limit_in_bytes"
CGROUP_V1_USAGE = "/sys/fs/cgroup/memory/memory.usage_in_bytes"
CGROUP_V1_STAT = "/sys/fs/cgroup/memory/memory.stat"

# cgroup v1 spells "no limit" as a huge sentinel (PAGE_COUNTER_MAX * PAGE_SIZE), not as a word; v2
# spells it "max". Anything at or above this is treated as unlimited.
CGROUP_UNLIMITED_MIN = 1 << 53

# P3b's floor, so the ceilings quoted below were measured under this policy.
DEFAULT_FLOOR_BYTES = 12 * GIB

# The largest pinned arena ever DEMONSTRATED on this box, per local-rank count. Advisory only —
# never a gate, because it is a property of this box on a given day while `MemAvailable` is the live
# truth. Used to warn when a plan is technically inside `MemAvailable` but outside anything anyone
# has actually reached.
#
# THE 1-RANK ENTRY IS NOW MEASURED ON THE SHIPPING PATH, not inherited from P3b. P3b's 34.0 GiB came
# from `hipMemCreate(location=Host)` in a standalone probe; the arena uses `hipHostMalloc` through
# `pinned_arena.attach()`, behind this module's floor and tripwire, and nobody had ever swept THAT.
# `tools/offload/pin_ceiling_sweep.py` did, one rank, card 0, 3 GiB chunks, 12 GiB floor, 2026-09-04:
# clean at 8/16/20/24/26/28/30/32/34/36/38/40/44/48/52 GiB targets, topping out at **51.0 GiB
# actually pinned**, with the per-chunk pin time flat throughout (median 6.4-9.3 s per 3 GiB chunk;
# max/median never above 2.0x, i.e. no rate collapse). 56 GiB is the first failure and it is not a
# `hipHostMalloc` refusal: at chunk 17/18 `MemAvailable` had fallen to 21.97 GiB — inside 2x the
# floor, so the swap tripwire ARMED — and 20,003,608 pages of swap-out tripped the 1 % rate
# threshold. The arena rolled back to 0 bytes and the box was fine.
#
# So the failure mode past the ceiling is a CLEAN, EARLY REFUSAL by this module, which is the whole
# design intent, and the real limit is node `MemAvailable` rather than anything in the allocator.
#
# THE 2-RANK ENTRY IS STILL P3b's AND IS NOW SUSPECT. 62.0 GiB across two ranks exceeds what ONE
# rank reached on this box today (51.0), and the ceiling is a node MemAvailable floor that does not
# grow with ranks (`plan.host_arena_ceiling_bytes` says exactly this). P3b measured it on an idle
# box; this box now idles with tens of GiB in zram. It is left unchanged because it has NOT been
# re-measured — sweeping two concurrent ranks is the obvious next probe — but a plan that leans on
# it is leaning on the older, more optimistic of two numbers.
PINNED_CEILING_BYTES: Dict[int, int] = {1: 51 * GIB, 2: 62 * GIB}
#: Back-compat alias. The name said P3B when only P3b's figures were in it; the 1-rank entry no
#: longer comes from P3b. Kept so existing importers (`weights/plan.py`, `tests/core/test_weight_plan
#: .py`) do not have to move in the same change that re-measured the number.
P3B_PINNED_CEILING_BYTES: Dict[int, int] = PINNED_CEILING_BYTES
P3B_NOTE = (
    "measured 2026-09-04 (tools/offload/pin_ceiling_sweep.py, card 0, 1 rank, 3 GiB chunks, 12 GiB "
    "floor): 51.0 GiB pinned clean with a flat per-chunk rate; 56.0 GiB aborted at chunk 17/18 on "
    "the swap tripwire with MemAvailable down to 21.97 GiB. 2-rank figure is still P3b's (idle box, "
    "no engine loaded): 62.0 GiB across two ranks (34.0 + 28.0), rank 1 stopped on the MemAvailable "
    "floor with 114,813 pages swapped — NOT re-measured, and above today's 1-rank result"
)

# Swap-out pages observed while pinning before we call it thrash. P3b saw 114,813 at the ceiling;
# 16,384 pages = 64 MiB is far below that and still far above idle noise on this box.
#
# THIS IS A FLOOR, NOT THE THRESHOLD — see `SwapTripwire`. As a flat constant it is a boot failure
# for any large arena: 30 GiB is 7.9 M pages, so 64 MiB is 0.2 % of the pin, a figure any co-tenant
# of a shared box crosses without the pinning being responsible for a single page of it (this box
# runs zram and idles with tens of GiB already swapped). Every 48-layer boot died on it.
DEFAULT_SWAP_TRIPWIRE_PAGES = 16 * 1024

# The RATE the tripwire gates on: pages evicted per page pinned. Above this the kernel is
# systematically making room for us rather than incidentally reclaiming, which is the box-destroying
# regime — pinned pages are themselves unevictable, so the eviction is one-way. 1 % of a 30 GiB pin
# is 78,643 pages, still below the 114,813 P3b measured AT the ceiling, so the fault P3b hit is
# still caught while a co-tenant's steady-state swap traffic is not.
DEFAULT_SWAP_TRIPWIRE_FRACTION = 0.01

# The tripwire is ARMED only once MemAvailable has fallen within this multiple of the floor.
#
# `pswpout` is box-wide and carries no attribution, so on a shared box a co-tenant's eviction storm
# reads as ours. Measured here, 2026-09-03: pinning 10 GiB with 71 GiB free and MemAvailable at
# 58 GiB coincided with 432,173 pages of swap-out — 1.65 GiB, 16 % of the pin — none of which our
# pinning could have caused, because the box was nowhere near short of memory (this box runs zram
# and its cumulative `pswpout` is in the hundreds of millions of pages). A threshold low enough to
# catch real thrash is far below that ambient traffic, so a swap-count-only tripwire on this box
# refuses every large arena for a reason that has nothing to do with the arena.
#
# Headroom is the discriminator that needs no attribution: swap-out matters only when memory is
# actually scarce. The tripwire exists because `MemAvailable` is an ESTIMATE that P3b proved
# optimistic — but P3b's rank 1 was AT its floor when it drove 114,813 pages out, which this arming
# condition catches, while "60 GiB available and the box is swapping anyway" is somebody else's
# problem and not a reason to refuse a boot.
DEFAULT_SWAP_TRIPWIRE_ARM_MULTIPLE = 2.0


class HostArenaCapacityError(RuntimeError):
    """The arena cannot be reserved. Raised BEFORE any pinning, or between chunks if the box moved."""


class HostArenaSwapThrashError(HostArenaCapacityError):
    """Pinning is pushing the box into swap. Continuing turns a slow boot into an unusable box."""


def _read_kv_kb(path: str, keys: Tuple[str, ...]) -> Dict[str, int | None]:
    out: Dict[str, int | None] = dict.fromkeys(keys)
    try:
        with open(path) as fh:
            text = fh.read()
    except OSError:
        return out
    return parse_kv_kb(text, keys)


def parse_kv_kb(text: str, keys: Tuple[str, ...]) -> Dict[str, int | None]:
    """Parse `/proc/meminfo`-style `Key:   1234 kB` lines into BYTES. Split out from the file read so
    it is testable without a `/proc`."""
    out: Dict[str, int | None] = dict.fromkeys(keys)
    for line in text.splitlines():
        parts = line.split(":", 1)
        if len(parts) == 2 and parts[0] in out:
            try:
                out[parts[0]] = int(parts[1].strip().split()[0]) * 1024
            except (ValueError, IndexError):
                pass
    return out


def read_meminfo() -> Dict[str, int | None]:
    return _read_kv_kb(MEMINFO_PATH, ("MemTotal", "MemFree", "MemAvailable", "Cached",
                                      "SwapTotal", "SwapFree", "Shmem"))


def parse_cgroup_available(
    limit_text: str | None,
    current_text: str | None,
    stat_text: str | None = None,
    *,
    reclaimable_keys: Tuple[str, ...] = ("inactive_file", "total_inactive_file"),
) -> int | None:
    """PURE. `None` means "this cgroup imposes no limit" (or the files are unreadable/garbage).

    Split out from the file reads for the same reason `parse_kv_kb` is: the policy has to be
    testable on a box whose own cgroup is unlimited, which is every dev box and neither of the two
    container shapes that matter.

    Reclaimable page cache is added back because a cgroup at its limit with 20 GiB of `inactive_file`
    is not actually out of memory — the kernel evicts that before it OOM-kills. Only *inactive* file
    pages are counted; anything more generous starts guessing.
    """
    limit = _parse_int(limit_text)
    if limit is None or limit >= CGROUP_UNLIMITED_MIN:
        return None
    current = _parse_int(current_text) or 0
    reclaimable = 0
    if stat_text:
        stats = _parse_flat_kv(stat_text)
        for key in reclaimable_keys:
            if key in stats:
                reclaimable = stats[key]
                break
    return max(0, limit - current + reclaimable)


def _parse_int(text: str | None) -> int | None:
    if text is None:
        return None
    token = text.strip().split()[0] if text.strip() else ""
    if token == "max":  # cgroup v2's "unlimited"
        return None
    try:
        return int(token)
    except ValueError:
        return None


def _parse_flat_kv(text: str) -> Dict[str, int]:
    """`memory.stat`'s `key value` lines. Same file format in v1 and v2."""
    out: Dict[str, int] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2:
            try:
                out[parts[0]] = int(parts[1])
            except ValueError:
                pass
    return out


def _read_text(path: str) -> str | None:
    try:
        with open(path) as fh:
            return fh.read()
    except OSError:
        return None


def cgroup_available_bytes() -> int | None:
    """Bytes this process's cgroup will still hand out, or `None` when it is unlimited.

    v2 first (every current container runtime), v1 as the fallback.
    """
    v2 = parse_cgroup_available(
        _read_text(CGROUP_V2_MAX), _read_text(CGROUP_V2_CURRENT), _read_text(CGROUP_V2_STAT)
    )
    if v2 is not None:
        return v2
    return parse_cgroup_available(
        _read_text(CGROUP_V1_LIMIT), _read_text(CGROUP_V1_USAGE), _read_text(CGROUP_V1_STAT)
    )


def mem_available_bytes() -> int:
    """The smaller of the host's `MemAvailable` and this cgroup's remaining allowance.

    0 when `/proc/meminfo` is unreadable. A 0 makes every capacity check FAIL, which is the correct
    direction: an unreadable `/proc/meminfo` means the gate is blind, and a blind gate must not wave
    68 GiB through.

    THE CGROUP TERM IS NOT OPTIONAL POLISH. `/proc/meminfo` is not namespaced, so inside the serve
    image it reports the *host's* memory. Reading it alone, a `--memory`-limited container passes
    this gate on a number that has nothing to do with what it is allowed to allocate, pins until the
    cgroup OOM-killer fires, and dies with `Killed` and no message from this module — which is
    exactly the "terrible" failure this file's docstring exists to prevent, one level of indirection
    down. An unlimited cgroup (the default `docker run`, and every bare-metal run) contributes
    nothing, so this costs three file reads and changes no existing behaviour.
    """
    host = int(read_meminfo().get("MemAvailable") or 0)
    if host <= 0:
        return 0
    cg = cgroup_available_bytes()
    return host if cg is None else min(host, cg)


def read_pswpout_pages() -> int:
    """Cumulative pages swapped OUT since boot. Deltas of this are the thrash tripwire."""
    try:
        with open(VMSTAT_PATH) as fh:
            for line in fh:
                if line.startswith("pswpout "):
                    return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return 0


@dataclass(frozen=True)
class CapacityVerdict:
    """The whole decision, with every input recorded so a failure is diagnosable from the log line
    alone rather than needing a repro."""

    fits: bool
    needed_per_rank_bytes: int
    local_ranks: int
    mem_available_bytes: int
    floor_bytes: int
    reason: str
    advisories: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def needed_total_bytes(self) -> int:
        return self.needed_per_rank_bytes * self.local_ranks

    @property
    def headroom_bytes(self) -> int:
        """What would be left above the floor if the reservation succeeded. Negative == shortfall."""
        return self.mem_available_bytes - self.needed_total_bytes - self.floor_bytes

    @property
    def shortfall_bytes(self) -> int:
        return max(0, -self.headroom_bytes)

    @property
    def single_rank_fits(self) -> bool:
        """Would THIS rank's own arena fit, ignoring the other local ranks?

        The difference between this and `fits` is the whole rank-skew story: `MemAvailable` already
        has every peer's *already pinned* bytes subtracted from it, so charging `local_ranks` after a
        peer has started pinning double-counts that peer. A verdict where `single_rank_fits` is True
        and `fits` is False is therefore ambiguous — a genuine multi-rank shortfall looks identical
        to a benign rank skew — and the operator has to be told which, or they will chase a phantom.
        """
        return self.needed_per_rank_bytes + self.floor_bytes <= self.mem_available_bytes

    def summary(self) -> str:
        return (
            f"weight-arena capacity: need {fmt_bytes(self.needed_per_rank_bytes)}/rank × "
            f"{self.local_ranks} = {fmt_bytes(self.needed_total_bytes)}, "
            f"MemAvailable {fmt_bytes(self.mem_available_bytes)}, "
            f"floor {fmt_bytes(self.floor_bytes)}, "
            f"headroom {fmt_bytes(self.headroom_bytes)} -> {'FITS' if self.fits else 'DOES NOT FIT'}"
        )

    def failure_message(self) -> str:
        """The message an operator reads at 3am. Numbers first, then the fixes, in the order of
        decreasing goodness — and the device-tier row table from PHASE0_REPORT §3.4 so the first fix
        is quantified rather than a suggestion."""
        return "\n".join(
            [
                "WEIGHT OFFLOAD: the pinned host arena does not fit. Aborting before any pinning.",
                f"  {self.summary()}",
                f"  reason: {self.reason}",
                f"  short by: {fmt_bytes(self.shortfall_bytes)}",
                f"  context: {P3B_NOTE}",
                *[f"  advisory: {a}" for a in self.advisories],
                *(
                    [
                        "  RANK SKEW CAVEAT: this rank's own arena "
                        f"({fmt_bytes(self.needed_per_rank_bytes)}) fits; only the "
                        f"{self.local_ranks}-rank total does not. MemAvailable already excludes "
                        "every byte a peer rank has ALREADY pinned, so if a peer got here first "
                        "this reading double-counts it and the shortfall is an artifact of boot "
                        "skew, not of the plan. Check the peer's [weight-arena] banner: if it "
                        "pinned successfully, re-run rather than shrinking the plan (a shrink on "
                        "one rank only is a TP desync).",
                    ]
                    if self.local_ranks > 1 and self.single_rank_fits
                    else []
                ),
                "  FIXES, best first:",
                "    1. Give the layer-granular DEVICE tier more of the model. Every GB on device is",
                "       a GB the host arena does not need. PHASE0_REPORT §3.4 (TP=2, card-1-gated",
                "       14.48 GB/s host): f=0.20 -> 55.0 GiB host total, ~22.5 tok/s; f=0.25 -> 51.6",
                "       GiB, ~23.7 tok/s; f=0.30 -> 48.2 GiB, ~25.0 tok/s. The device tier is a",
                "       CAPACITY prerequisite here, not an optimisation.",
                "    2. Free host RAM: stop other jobs on this box, drop the page cache, shrink the",
                "       ZFS ARC. `free -h` and `docker ps` show what else is resident.",
                "    3. Use a smaller checkpoint.",
                "    4. The file-backed overflow tier costs zero anonymous RAM (P3b: Cached delta =",
                "       1.000x committed) but is UNBUILT and uncharacterised above 4 GiB/rank (P3c).",
                "  NOT a fix: reserving less. The plan must be identical on every TP rank or the",
                "  ranks place granules at different offsets and the collectives hang.",
            ]
        )


def evaluate_capacity(
    needed_per_rank_bytes: int,
    local_ranks: int,
    mem_available_bytes: int,
    floor_bytes: int = DEFAULT_FLOOR_BYTES,
) -> CapacityVerdict:
    """PURE. No `/proc`, no clock — so the policy is unit-testable and reviewable in one place."""
    if local_ranks < 1:
        raise ValueError(f"local_ranks must be >= 1, got {local_ranks}")
    if needed_per_rank_bytes < 0 or floor_bytes < 0:
        raise ValueError("byte counts must be non-negative")

    total = needed_per_rank_bytes * local_ranks
    advisories: list[str] = []

    if mem_available_bytes <= 0:
        return CapacityVerdict(
            fits=False,
            needed_per_rank_bytes=needed_per_rank_bytes,
            local_ranks=local_ranks,
            mem_available_bytes=mem_available_bytes,
            floor_bytes=floor_bytes,
            reason=(
                "MemAvailable read as 0 — /proc/meminfo is unreadable or not mounted in this "
                "container. The gate is blind, and a blind gate must fail closed."
            ),
        )

    ceiling = P3B_PINNED_CEILING_BYTES.get(local_ranks)
    if ceiling is not None and total > ceiling:
        advisories.append(
            f"{fmt_bytes(total)} exceeds the largest pinned arena ever demonstrated on this box at "
            f"{local_ranks} rank(s) ({fmt_bytes(ceiling)}, and that was measured with NO engine "
            f"loaded). Even if MemAvailable says yes, expect swap."
        )
    if total > 0 and mem_available_bytes - total < floor_bytes * 2:
        advisories.append(
            "post-reservation headroom is under 2x the floor; the KV pool, page cache and the "
            "engine's own host allocations all come out of what is left"
        )

    fits = total + floor_bytes <= mem_available_bytes
    reason = (
        "fits with headroom above the floor"
        if fits
        else (
            f"needs {fmt_bytes(total)} + {fmt_bytes(floor_bytes)} floor = "
            f"{fmt_bytes(total + floor_bytes)}, but MemAvailable is only "
            f"{fmt_bytes(mem_available_bytes)}"
        )
    )
    return CapacityVerdict(
        fits=fits,
        needed_per_rank_bytes=needed_per_rank_bytes,
        local_ranks=local_ranks,
        mem_available_bytes=mem_available_bytes,
        floor_bytes=floor_bytes,
        reason=reason,
        advisories=tuple(advisories),
    )


def check_capacity(
    needed_per_rank_bytes: int,
    local_ranks: int = 1,
    floor_bytes: int = DEFAULT_FLOOR_BYTES,
    *,
    raise_on_fail: bool = True,
) -> CapacityVerdict:
    """`evaluate_capacity` against the live box. Raises `HostArenaCapacityError` by default."""
    v = evaluate_capacity(needed_per_rank_bytes, local_ranks, mem_available_bytes(), floor_bytes)
    if raise_on_fail and not v.fits:
        raise HostArenaCapacityError(v.failure_message())
    return v


class SwapTripwire:
    """Watches `pswpout` across the pinning loop. The threshold is a RATE, not a constant.

    `MemAvailable` is checked before every chunk, but it is an *estimate*: P3b drove 114,813 pages
    to swap while still nominally above its floor. Swap-out during pinning means the kernel is
    evicting to make room for pages that can never themselves be evicted — the box is being
    destroyed one chunk at a time. Abort while it is still recoverable.

    WHY A FLAT THRESHOLD WAS WRONG, AND HOW. `DEFAULT_SWAP_TRIPWIRE_PAGES` alone (64 MiB) is a
    fraction of a percent of the pins this feature exists to make: the target checkpoint asks for
    ~29.3 GiB per rank, so a flat 64 MiB aborts on 0.2 % of the payload's worth of swap-out — which a
    box that swaps at all crosses from ambient co-tenant traffic long before the pinning has done
    anything wrong. Every 48-layer boot died there. But raising the constant is not the fix either:
    it would make the tripwire progressively blinder as arenas grow, and the *small*-arena case (a
    64 MiB pin that drives 64 MiB of eviction) is genuinely pathological and must still fire.

    So `pin_bytes` turns it into a rate: abort when the eviction we caused exceeds
    `DEFAULT_SWAP_TRIPWIRE_FRACTION` of what we are pinning, never below the flat floor. Passing no
    `pin_bytes` reproduces the old flat behaviour exactly, which is what every existing caller and
    test gets.

    It cannot ATTRIBUTE the swap-out — `pswpout` is box-wide — which is why `check()` takes the live
    `available`/`floor` and arms only when memory is actually scarce. See
    `DEFAULT_SWAP_TRIPWIRE_ARM_MULTIPLE` for the measurement that forced that. Passing no headroom
    keeps the tripwire permanently armed, which is the old behaviour and the right default for a
    caller that has no `MemAvailable` reading to offer.

    ARMING GATES *WHEN* THE COMPARISON RUNS; IT MUST ALSO GATE *WHICH* SWAP IS COUNTED. Until the
    re-baseline below, it did not, and the two halves contradicted each other. `delta_pages()`
    counted from construction — the start of the pin — while `armed()` only decided whether to look.
    On a long pin the box is nowhere near its floor for most of the loop, so the tripwire sits
    disarmed while box-wide `pswpout` accumulates from co-tenants; then, near the END of the pin,
    `MemAvailable` finally falls inside `arm_multiple x floor`, the tripwire arms, and the FIRST
    armed comparison is charged the entire disarmed window. That is precisely the ambient traffic
    `DEFAULT_SWAP_TRIPWIRE_ARM_MULTIPLE` exists to exclude, billed a few chunks later.

    MEASURED, 2026-09-04, qwen4_exp 48 layers TP=2, 27.10 GiB/rank: aborted at chunk 33/37 claiming
    **21,660,130 pages = 82.63 GiB "swapped out while pinning"** a 27.10 GiB arena. The number
    refutes its own attribution — no 27.10 GiB pin evicts 82.63 GiB — and the run was killed by it
    after ten minutes of successful pinning. The identical configuration booted, generated coherent
    text and pinned all 37 chunks when the floor was lowered, which did not add capacity: it lowered
    the ARMING threshold enough that the pin finished before the window opened. A tripwire whose
    verdict depends on how late it happens to arm is not measuring anything.

    So the delta is re-baselined ONCE, on the disarmed -> armed transition, and the count that
    matters becomes the swap-out observed WHILE memory was scarce. Deliberately once and not per
    transition: `MemAvailable` oscillates around the arming threshold as chunks are pinned and page
    cache is reclaimed, and re-baselining on every re-arm would reset the count forever and blind the
    tripwire completely. And deliberately only when a disarmed check was actually seen
    (`disarmed_checks`), so a caller that is armed from its first check — every caller that passes no
    `available`, plus every existing test — keeps the construction baseline and behaves exactly as
    before.
    """

    __slots__ = (
        "threshold_pages",
        "baseline_pages",
        "enabled",
        "pin_bytes",
        "fraction",
        "arm_multiple",
        "disarmed_checks",
        "armed_baseline_taken",
    )

    def __init__(
        self,
        threshold_pages: int = DEFAULT_SWAP_TRIPWIRE_PAGES,
        *,
        pin_bytes: int = 0,
        fraction: float = DEFAULT_SWAP_TRIPWIRE_FRACTION,
        arm_multiple: float = DEFAULT_SWAP_TRIPWIRE_ARM_MULTIPLE,
        page_bytes: int = 4096,
    ) -> None:
        self.pin_bytes = max(0, int(pin_bytes))
        self.fraction = float(fraction)
        self.arm_multiple = float(arm_multiple)
        self.disarmed_checks = 0
        self.armed_baseline_taken = False
        scaled = int(self.pin_bytes / max(1, page_bytes) * self.fraction)
        self.threshold_pages = max(int(threshold_pages), scaled)
        self.baseline_pages = read_pswpout_pages()
        # A box with no swap configured cannot thrash; do not spend a /proc read per chunk on it.
        self.enabled = bool(read_meminfo().get("SwapTotal") or 0) and self.threshold_pages > 0

    def delta_pages(self) -> int:
        return max(0, read_pswpout_pages() - self.baseline_pages)

    def armed(self, available: int | None, floor: int) -> bool:
        """Is memory scarce enough that swap-out could plausibly be OUR fault?"""
        if available is None or floor <= 0 or self.arm_multiple <= 0:
            return True
        return available < floor * self.arm_multiple

    def check(
        self, context: str = "", *, available: int | None = None, floor: int = 0
    ) -> None:
        if not self.enabled:
            return
        if not self.armed(available, floor):
            self.disarmed_checks += 1
            return
        # The disarmed -> armed transition, once. See the class docstring: without this the first
        # armed comparison is charged every page a co-tenant evicted while the box had tens of GiB
        # free, which on a long pin is the whole pin. `disarmed_checks` guards it so an
        # always-armed caller keeps the construction baseline byte-for-byte.
        if self.disarmed_checks and not self.armed_baseline_taken:
            self.armed_baseline_taken = True
            self.baseline_pages = read_pswpout_pages()
        d = self.delta_pages()
        if d >= self.threshold_pages:
            scale = (
                f" (threshold is {self.fraction:.1%} of the {fmt_bytes(self.pin_bytes)} being "
                f"pinned, floored at {DEFAULT_SWAP_TRIPWIRE_PAGES} pages; armed because "
                f"MemAvailable {fmt_bytes(available or 0)} is within {self.arm_multiple:g}x the "
                f"{fmt_bytes(floor)} floor)"
                if self.pin_bytes
                else ""
            )
            raise HostArenaSwapThrashError(
                f"WEIGHT OFFLOAD: aborting the pinned host arena — {d} pages ({fmt_bytes(d * 4096)} "
                f"at 4 KiB) were swapped out while pinning{(' ' + context) if context else ''}"
                f"{scale}. "
                f"Pinned pages are unevictable, so continuing trades the whole box for a boot that "
                f"will not finish. Same fixes as a capacity failure: raise the device-tier fraction, "
                f"free host RAM, or use a smaller checkpoint."
            )
