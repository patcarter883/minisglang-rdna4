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

# What P3b actually reached, on an IDLE box with NO ENGINE LOADED. Advisory only — never a gate,
# because it is a property of that box on that day, while `MemAvailable` is the live truth. Used to
# warn when a plan is technically inside `MemAvailable` but outside anything ever demonstrated.
P3B_PINNED_CEILING_BYTES: Dict[int, int] = {1: 34 * GIB, 2: 62 * GIB}
P3B_NOTE = (
    "P3b (idle box, no engine loaded): 34.0 GiB pinned for 1 rank, 62.0 GiB across 2 ranks "
    "(34.0 + 28.0), rank 1 stopped on the MemAvailable floor with 114,813 pages swapped"
)

# Swap-out pages observed while pinning before we call it thrash. P3b saw 114,813 at the ceiling;
# 16,384 pages = 64 MiB is far below that and still far above idle noise on this box.
DEFAULT_SWAP_TRIPWIRE_PAGES = 16 * 1024


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
                "       ZFS ARC. `gpu-status` shows who else is resident.",
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
    """Watches `pswpout` across the pinning loop.

    `MemAvailable` is checked before every chunk, but it is an *estimate*: P3b drove 114,813 pages
    to swap while still nominally above its floor. Swap-out during pinning means the kernel is
    evicting to make room for pages that can never themselves be evicted — the box is being
    destroyed one chunk at a time. Abort while it is still recoverable.
    """

    __slots__ = ("threshold_pages", "baseline_pages", "enabled")

    def __init__(self, threshold_pages: int = DEFAULT_SWAP_TRIPWIRE_PAGES) -> None:
        self.threshold_pages = int(threshold_pages)
        self.baseline_pages = read_pswpout_pages()
        # A box with no swap configured cannot thrash; do not spend a /proc read per chunk on it.
        self.enabled = bool(read_meminfo().get("SwapTotal") or 0) and self.threshold_pages > 0

    def delta_pages(self) -> int:
        return max(0, read_pswpout_pages() - self.baseline_pages)

    def check(self, context: str = "") -> None:
        if not self.enabled:
            return
        d = self.delta_pages()
        if d >= self.threshold_pages:
            raise HostArenaSwapThrashError(
                f"WEIGHT OFFLOAD: aborting the pinned host arena — {d} pages ({fmt_bytes(d * 4096)} "
                f"at 4 KiB) were swapped out while pinning{(' ' + context) if context else ''}. "
                f"Pinned pages are unevictable, so continuing trades the whole box for a boot that "
                f"will not finish. Same fixes as a capacity failure: raise the device-tier fraction, "
                f"free host RAM, or use a smaller checkpoint."
            )
