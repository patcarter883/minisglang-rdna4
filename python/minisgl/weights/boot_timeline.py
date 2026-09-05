"""Boot attribution — where the wall clock and the host RSS go between `Engine(...)` and the first token.

WHY THIS EXISTS
---------------
The offloaded qwen4_exp boot takes 519-871 s. Stage B is 167-278 s of that; the remaining 352-593 s
had never been timed by anybody, and Stage B's own 105-174 MB/s against a 4.9 GB/s NVMe had never
been split into I/O vs compute vs placement. Boot time is the multiplier on every experiment this
repo runs — an agent that pays 9-15 minutes before the first token takes one contaminated big run
instead of five clean small ones — so "nobody has ever attributed boot" is itself the defect.

WHAT IT IS
----------
Two primitives and nothing else:

  * `phase(name)` — a coarse span around a boot stage. Records wall, and the RSS/device deltas
    across it. There are ~20 of these in `Engine.__init__`; each one costs two `perf_counter()`
    calls and four small `/proc` + torch reads, i.e. microseconds against seconds.
  * `tick(bucket, dt)` / `count(bucket, n)` — accumulators for the INNER loops (per-tensor read,
    per-tensor H2D, per-chunk `gc.collect`). These are hit ~10^5-10^6 times on a 48-layer boot, so
    they are a bare dict add against a `perf_counter()` delta the caller already has; no context
    manager, no allocation, no formatting.

NOT ENV-GATED. The recording is unconditional because it is unconditionally cheap and because an
instrument that only exists when someone remembers to set a variable is an instrument that is never
on for the run that mattered. `MINISGL_BOOT_TIMELINE_JSON` only chooses whether the finished report
is ALSO written to a file — it does not gate collection, and the human-readable report goes to the
ordinary boot log either way.

RSS IS SPLIT ANON vs FILE, DELIBERATELY
---------------------------------------
`VmHWM` on the Stage-B boot peaks at 33.3 GB while the design's live set is one layer (1.465 GiB),
and the two candidate explanations are not the same bug:

  * `RssAnon` — heap, torch CPU tensors, and the pinned arena (`hipHostMalloc` pins ordinary
    anonymous pages via userptr, so the arena IS anonymous RSS of this process). Growth here is REAL
    retention and is a leak or an over-large staging set.
  * `RssFile` — page-cache pages of the safetensors mmaps that this process has touched. Reading an
    84 GB checkpoint through `safe_open` maps and touches every byte of it, so this number climbs to
    the bytes read whether or not anything is retained. It is reclaimable under pressure and is NOT
    a leak.

Reporting only `VmHWM` cannot tell those apart, which is why the 33.3 GB figure has stood unexplained.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

__all__ = ["BootTimeline", "timeline", "phase", "tick", "count", "rss_bytes"]

_MIB = 1 << 20
_GIB = 1 << 30

#: The `/proc/self/status` fields worth carrying. VmHWM is the historical high-water mark (what the
#: Stage-B ledger already reports); the Rss* triple is the CURRENT split that says what it is made of.
_STATUS_FIELDS = ("VmRSS", "RssAnon", "RssFile", "RssShmem", "VmHWM")


def rss_bytes() -> Dict[str, int]:
    """Current RSS broken into anon / file / shmem, plus the peak. Zeros if `/proc` is unreadable."""
    out = {k: 0 for k in _STATUS_FIELDS}
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                name, _, rest = line.partition(":")
                if name in out:
                    out[name] = int(rest.split()[0]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return out


def _read_io() -> Dict[str, int]:
    """`/proc/self/io`. `read_bytes` is what actually came off the BLOCK DEVICE for this process.

    This is the number that answers "is anything read twice": `rchar` counts bytes the process asked
    for (including page-cache hits and mmap faults served from cache), `read_bytes` counts bytes the
    block layer actually moved. A `read_bytes` far above the payload is real re-reading from disk; a
    high `rchar` with a low `read_bytes` is a warm cache, which is a different (and much cheaper)
    finding.
    """
    out = {"rchar": 0, "read_bytes": 0, "syscr": 0}
    try:
        with open("/proc/self/io") as fh:
            for line in fh:
                name, _, rest = line.partition(":")
                if name in out:
                    out[name] = int(rest.strip())
    except (OSError, ValueError):
        pass
    return out


def _process_age_seconds() -> float:
    """Seconds since THIS process was forked, from `/proc`. -1.0 if unreadable.

    The timeline's own origin is the first `import minisgl.weights.boot_timeline`, which is inside
    `Engine.__init__` — so on its own it cannot see the seconds spent BEFORE the engine: the torch
    import, the HF config/tokenizer resolution, the harness's own setup. On a boot whose total is
    the number under investigation, "everything before the first phase" has to be a measured term
    and not a residual nobody named.
    """
    try:
        with open("/proc/self/stat") as fh:
            fields = fh.read().rsplit(") ", 1)[1].split()
        starttime_ticks = int(fields[19])  # field 22, 1-indexed, minus the 2 consumed by the rsplit
        with open("/proc/uptime") as fh:
            uptime = float(fh.read().split()[0])
        hz = os.sysconf("SC_CLK_TCK")
        return uptime - starttime_ticks / hz
    except (OSError, ValueError, IndexError):
        return -1.0


def _box_state() -> Dict[str, int]:
    """Box-wide memory + swap counters. THE confound this report must carry, not infer.

    Two boots of the IDENTICAL 37-host-layer configuration came in at 519.5 s and 871.1 s (+68%),
    and the slower one was on a box carrying two or three other agents' workloads. Pinning the arena
    is `hipHostMalloc`, i.e. the kernel locking pages down, and on a box that is already swapping
    that means reclaim — so "the arena pinned at 130 MiB/s" is only a statement about this code if
    the swap counters next to it say the box was not thrashing. `pswpin`/`pswpout` are cumulative
    page counts; the report carries the DELTA across the boot.
    """
    out = {"MemAvailable": 0, "SwapFree": 0, "SwapTotal": 0}
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                name, _, rest = line.partition(":")
                if name in out:
                    out[name] = int(rest.split()[0]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    try:
        with open("/proc/vmstat") as fh:
            for line in fh:
                k, _, v = line.partition(" ")
                if k in ("pswpin", "pswpout", "pgmajfault"):
                    out[k] = int(v)
    except (OSError, ValueError):
        pass
    return out


def _device_alloc() -> int:
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.is_initialized():
            return int(torch.cuda.memory_allocated())
    except Exception:  # pragma: no cover - torch not up yet is normal at the first phases
        pass
    return 0


@dataclass
class Span:
    name: str
    seconds: float
    rss_start: Dict[str, int]
    rss_end: Dict[str, int]
    dev_start: int
    dev_end: int
    io_start: Dict[str, int]
    io_end: Dict[str, int]
    depth: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "seconds": round(self.seconds, 3),
            "depth": self.depth,
            "rss_anon_delta": self.rss_end["RssAnon"] - self.rss_start["RssAnon"],
            "rss_file_delta": self.rss_end["RssFile"] - self.rss_start["RssFile"],
            "rss_anon_end": self.rss_end["RssAnon"],
            "rss_file_end": self.rss_end["RssFile"],
            "vmhwm_end": self.rss_end["VmHWM"],
            "device_alloc_delta": self.dev_end - self.dev_start,
            "disk_read_bytes": self.io_end["read_bytes"] - self.io_start["read_bytes"],
            "rchar": self.io_end["rchar"] - self.io_start["rchar"],
        }


class _PhaseCtx:
    __slots__ = ("tl", "name", "t0", "rss0", "dev0", "io0")

    def __init__(self, tl: "BootTimeline", name: str) -> None:
        self.tl = tl
        self.name = name

    def __enter__(self) -> "_PhaseCtx":
        self.tl._depth += 1
        self.t0 = time.perf_counter()
        self.rss0 = rss_bytes()
        self.dev0 = _device_alloc()
        self.io0 = _read_io()
        return self

    def __exit__(self, *exc: Any) -> bool:
        dt = time.perf_counter() - self.t0
        self.tl._depth -= 1
        self.tl.spans.append(
            Span(
                name=self.name,
                seconds=dt,
                rss_start=self.rss0,
                rss_end=rss_bytes(),
                dev_start=self.dev0,
                dev_end=_device_alloc(),
                io_start=self.io0,
                io_end=_read_io(),
                depth=self.tl._depth,
            )
        )
        return False


class BootTimeline:
    """Coarse phase spans + fine-grained accumulators for one process's boot."""

    def __init__(self) -> None:
        self.t_origin = time.perf_counter()
        self.spans: List[Span] = []
        #: bucket -> seconds. Hit from the per-tensor loops; a bare dict add.
        self.buckets: Dict[str, float] = {}
        #: bucket -> integer count (bytes, calls, files).
        self.counters: Dict[str, int] = {}
        #: free-form per-chunk rows from Stage B.
        self.rows: List[Dict[str, Any]] = []
        #: checkpoint path -> how many times it was opened, split by purpose. THE read-amplification
        #: instrument: the checkpoint is 196 per-layer/per-expert-range shards and Stage B chunks per
        #: layer, so "is any shard opened (and therefore re-read) by more than one chunk?" is a
        #: question about this dict and not about a wall-clock ratio.
        self.file_opens: Dict[str, Dict[str, int]] = {}
        self._depth = 0
        self.rss_at_start = rss_bytes()
        self.io_at_start = _read_io()
        #: Seconds this process had already been alive when the timeline started — i.e. everything
        #: before `Engine.__init__`: `import torch`, the HF config read, the harness's own setup.
        self.pre_engine_seconds = _process_age_seconds()
        self.box_at_start = _box_state()

    # -- primitives ---------------------------------------------------------------------------

    def phase(self, name: str) -> _PhaseCtx:
        return _PhaseCtx(self, name)

    def tick(self, bucket: str, dt: float) -> None:
        self.buckets[bucket] = self.buckets.get(bucket, 0.0) + dt

    def count(self, bucket: str, n: int = 1) -> None:
        self.counters[bucket] = self.counters.get(bucket, 0) + n

    def row(self, r: Dict[str, Any]) -> None:
        self.rows.append(r)

    def note_file(self, path: str, purpose: str) -> None:
        """One `safe_open` of `path`. `purpose` is `header` (keys only) or `tensors` (a real read)."""
        d = self.file_opens.setdefault(path, {})
        d[purpose] = d.get(purpose, 0) + 1

    def file_summary(self) -> Dict[str, Any]:
        tensor_opens = {p: d.get("tensors", 0) for p, d in self.file_opens.items()}
        reread = {p: n for p, n in tensor_opens.items() if n > 1}
        return {
            "distinct_files": len(self.file_opens),
            "header_opens": sum(d.get("header", 0) for d in self.file_opens.values()),
            "tensor_opens": sum(tensor_opens.values()),
            "files_opened_for_tensors_more_than_once": len(reread),
            "max_tensor_opens_of_one_file": max(tensor_opens.values(), default=0),
            "example_rereads": sorted(reread.items(), key=lambda kv: -kv[1])[:5],
        }

    # -- report -------------------------------------------------------------------------------

    def total_seconds(self) -> float:
        return time.perf_counter() - self.t_origin

    def as_dict(self) -> Dict[str, Any]:
        rss = rss_bytes()
        io = _read_io()
        return {
            "total_seconds": round(self.total_seconds(), 3),
            "pre_engine_seconds": round(self.pre_engine_seconds, 3),
            "process_age_seconds": round(_process_age_seconds(), 3),
            "phases": [s.as_dict() for s in self.spans],
            "buckets_seconds": {k: round(v, 3) for k, v in sorted(self.buckets.items())},
            "counters": dict(sorted(self.counters.items())),
            "rows": self.rows,
            "files": self.file_summary(),
            "rss_end": rss,
            "rss_start": self.rss_at_start,
            "box_start": self.box_at_start,
            "box_end": (box_end := _box_state()),
            "swap_in_pages": box_end.get("pswpin", 0) - self.box_at_start.get("pswpin", 0),
            "swap_out_pages": box_end.get("pswpout", 0) - self.box_at_start.get("pswpout", 0),
            "major_faults": box_end.get("pgmajfault", 0) - self.box_at_start.get("pgmajfault", 0),
            "disk_read_bytes_total": io["read_bytes"] - self.io_at_start["read_bytes"],
            "rchar_total": io["rchar"] - self.io_at_start["rchar"],
        }

    def describe(self) -> str:
        d = self.as_dict()
        lines = [
            f"[boot-timeline] total {d['total_seconds']:.1f} s inside Engine.__init__, "
            f"{d['pre_engine_seconds']:.1f} s before it (torch import + config), "
            f"process age now {d['process_age_seconds']:.1f} s"
        ]
        for p in d["phases"]:
            pad = "  " * p["depth"]
            lines.append(
                f"[boot-timeline] {pad}{p['name']:<34} {p['seconds']:8.2f} s  "
                f"anon{p['rss_anon_delta'] / _GIB:+7.2f} file{p['rss_file_delta'] / _GIB:+7.2f} "
                f"dev{p['device_alloc_delta'] / _GIB:+7.2f} GiB  "
                f"disk {p['disk_read_bytes'] / _GIB:6.2f} GiB"
            )
        if d["buckets_seconds"]:
            lines.append("[boot-timeline] --- inner-loop buckets (seconds) ---")
            for k, v in sorted(d["buckets_seconds"].items(), key=lambda kv: -kv[1]):
                lines.append(f"[boot-timeline]   {k:<40} {v:8.2f} s")
        if d["counters"]:
            lines.append("[boot-timeline] --- counters ---")
            for k, v in d["counters"].items():
                extra = f"  ({v / _GIB:.2f} GiB)" if k.endswith("_bytes") else ""
                lines.append(f"[boot-timeline]   {k:<40} {v:>16,d}{extra}")
        fs = d["files"]
        lines.append(
            f"[boot-timeline] checkpoint files: {fs['distinct_files']} distinct, "
            f"{fs['header_opens']} header opens, {fs['tensor_opens']} tensor opens, "
            f"{fs['files_opened_for_tensors_more_than_once']} opened >1x for tensors "
            f"(max {fs['max_tensor_opens_of_one_file']}x)"
        )
        r = d["rss_end"]
        lines.append(
            f"[boot-timeline] RSS end: total {r['VmRSS'] / _GIB:.2f} GiB "
            f"(anon {r['RssAnon'] / _GIB:.2f} + file {r['RssFile'] / _GIB:.2f} "
            f"+ shmem {r['RssShmem'] / _GIB:.2f}), peak VmHWM {r['VmHWM'] / _GIB:.2f} GiB"
        )
        lines.append(
            f"[boot-timeline] disk read_bytes {d['disk_read_bytes_total'] / _GIB:.2f} GiB, "
            f"rchar {d['rchar_total'] / _GIB:.2f} GiB"
        )
        lines.append(
            f"[boot-timeline] box: MemAvailable {d['box_start']['MemAvailable'] / _GIB:.1f} -> "
            f"{d['box_end']['MemAvailable'] / _GIB:.1f} GiB, swap used "
            f"{(d['box_start']['SwapTotal'] - d['box_start']['SwapFree']) / _GIB:.1f} -> "
            f"{(d['box_end']['SwapTotal'] - d['box_end']['SwapFree']) / _GIB:.1f} GiB; "
            f"swapped in {d['swap_in_pages'] * 4096 / _GIB:.2f} GiB, out "
            f"{d['swap_out_pages'] * 4096 / _GIB:.2f} GiB during boot"
        )
        return "\n".join(lines)

    def dump(self, path: Optional[str] = None) -> Optional[str]:
        """Write the report as JSON. `path` defaults to `$MINISGL_BOOT_TIMELINE_JSON`.

        Rank-suffixed, because a TP boot has one timeline PER PROCESS and the two ranks do not do the
        same work (rank 0 reads the tqdm-visible stream; both pin their own arena). Overwriting one
        file from two processes would produce a report that is a torn mix of both.
        """
        path = path or os.environ.get("MINISGL_BOOT_TIMELINE_JSON") or ""
        if not path:
            return None
        try:
            from minisgl.distributed.info import get_tp_info

            rank = get_tp_info().rank
        except Exception:
            rank = 0
        stem, ext = os.path.splitext(path)
        out = f"{stem}.rank{rank}{ext or '.json'}"
        try:
            os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
            with open(out, "w") as fh:
                json.dump(self.as_dict(), fh, indent=1)
        except OSError:
            return None
        return out


#: THE process-wide timeline. A module singleton rather than something threaded through twelve call
#: sites: the inner accumulators are hit from `models/weight.py`, which must not grow an engine
#: dependency, and a boot is single-threaded per rank by construction (one process per rank).
_TIMELINE = BootTimeline()


def timeline() -> BootTimeline:
    return _TIMELINE


def phase(name: str) -> _PhaseCtx:
    return _TIMELINE.phase(name)


def tick(bucket: str, dt: float) -> None:
    _TIMELINE.tick(bucket, dt)


def count(bucket: str, n: int = 1) -> None:
    _TIMELINE.count(bucket, n)
