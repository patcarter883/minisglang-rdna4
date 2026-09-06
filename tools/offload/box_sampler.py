"""Sample box-wide memory/reclaim counters at 1 Hz so a boot's stalls can be TIME-ALIGNED.

Why this exists: the 48-layer TP=2 boot's `ckpt.h2d` bucket spikes on 10 specific layers, and the
spikes are IDENTICAL on both ranks to within 1% -- two independent processes on two different cards.
At layer 0 the same ~33 s stall landed in `stageb.sink_place` on rank 0 and in `ckpt.h2d` on rank 1.
A stall that lands in a DIFFERENT bucket on each rank is not a defect of either bucket's operation;
it is the box stopping both processes. This sampler is what turns that inference into evidence: it
records `/proc/vmstat` reclaim + zram-swap counters and `/proc/meminfo` on a monotonic clock that the
boot timeline's own `time.perf_counter()` rows can be lined up against.

Runs on the HOST, outside the container, so it survives the container and costs one 4 KiB read per
second. No GPU, no lease.
"""

from __future__ import annotations

import json
import sys
import time

VMSTAT_KEYS = (
    "pswpin",
    "pswpout",
    "pgmajfault",
    "pgscan_kswapd",
    "pgsteal_kswapd",
    "pgscan_direct",
    "pgsteal_direct",
    "pgalloc_normal",
    "compact_stall",
    "nr_free_pages",
)
MEMINFO_KEYS = ("MemFree", "MemAvailable", "Cached", "SwapFree", "Dirty", "Writeback")


def _vmstat() -> dict:
    out = {}
    with open("/proc/vmstat") as fh:
        for line in fh:
            k, _, v = line.partition(" ")
            if k in VMSTAT_KEYS:
                out[k] = int(v)
    return out


def _meminfo() -> dict:
    out = {}
    with open("/proc/meminfo") as fh:
        for line in fh:
            k, _, rest = line.partition(":")
            if k in MEMINFO_KEYS:
                out[k] = int(rest.split()[0]) * 1024
    return out


def _arc() -> int:
    try:
        with open("/proc/spl/kstat/zfs/arcstats") as fh:
            for line in fh:
                if line.startswith("size "):
                    return int(line.split()[-1])
    except OSError:
        pass
    return 0


def main() -> None:
    path = sys.argv[1]
    period = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    t0 = time.perf_counter()
    wall0 = time.time()
    with open(path, "w") as out:
        out.write(json.dumps({"t0_wall": wall0, "period": period}) + "\n")
        out.flush()
        while True:
            row = {"t": round(time.perf_counter() - t0, 3), "wall": round(time.time(), 3)}
            row.update(_vmstat())
            row.update(_meminfo())
            row["arc"] = _arc()
            try:
                with open("/proc/loadavg") as fh:
                    row["load1"] = float(fh.read().split()[0])
            except OSError:
                pass
            out.write(json.dumps(row) + "\n")
            out.flush()
            time.sleep(period)


if __name__ == "__main__":
    main()
