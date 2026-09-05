#!/usr/bin/env python3
"""ISOLATE `arena_pin_attach`: hipHostMalloc + device-issued memset, one chunk at a time.

The 48-layer boot spends 241.5 s (41%) in `PinnedWeightArena.attach()`, split 65.7 s of
`hipHostMalloc` and 171.8 s of `hip.memset_d32` over the mapped device pointer. The code's own
comment prices the second at "~70 ms/2 GiB" (≈28 GB/s, i.e. PCIe); measured it is 144 MiB/s, 200x
off. Two explanations that call for opposite fixes:

  * the PRIMITIVE is slow on gfx1201 — a device-issued fill into freshly hipHostMalloc'd host
    memory runs at ~144 MiB/s no matter what else is happening, in which case first_touch is simply
    a 172-second debug fill and the fix is to stop doing it;
  * the BOX is the cause — it only collapses once tens of GiB are pinned and the kernel is
    reclaiming into zram (`vm.swappiness=100`, 26.9 GiB already in zram), in which case the pin
    rate is a memory-state problem and the arena code is innocent.

This separates them by running the identical two calls on a quiet box, printing the rate PER CHUNK
so the trend across a growing pin is visible, with the box's swap counters beside each row. Nothing
else is resident: no model, no checkpoint, no torch.
"""
import ctypes
import os
import sys
import time

sys.path.insert(0, "/engine/python")
from minisgl.weights.hipmem import get_hip  # noqa: E402

MIB = 1 << 20
GIB = 1 << 30
CHUNK = int(os.environ.get("PROBE_CHUNK_MIB", "1372")) * MIB
N = int(os.environ.get("PROBE_CHUNKS", "18"))


def vm():
    d = {"pswpin": 0, "pswpout": 0, "pgmajfault": 0}
    with open("/proc/vmstat") as fh:
        for line in fh:
            k, _, v = line.partition(" ")
            if k in d:
                d[k] = int(v)
    return d


def mem_avail():
    with open("/proc/meminfo") as fh:
        for line in fh:
            if line.startswith("MemAvailable"):
                return int(line.split()[1]) * 1024
    return 0


hip = get_hip()
hip.set_device(0)
print(f"device: {hip.device_name(0)}  chunk={CHUNK / MIB:.0f} MiB  n={N}", flush=True)
print(
    f"{'chunk':>5}{'alloc_s':>9}{'alloc MiB/s':>13}{'touch_s':>9}{'touch MiB/s':>13}"
    f"{'memcpyH2D MiB/s':>17}{'MemAvail GiB':>14}{'swpout GiB':>12}",
    flush=True,
)

held = []
# A device-side source for the H2D leg: the SAME bytes moved by hipMemcpy rather than by a
# device-issued memset, so "the link is slow" and "this particular fill is slow" are separable.
dev_src = hip.dev_alloc(min(CHUNK, 512 * MIB))

t_alloc = t_touch = t_copy = 0.0
try:
    for i in range(N):
        v0, a0 = vm(), mem_avail()
        t0 = time.perf_counter()
        hptr, dptr = hip.host_alloc(CHUNK)
        t1 = time.perf_counter()
        held.append(hptr)
        hip.memset_d32(dptr, 0xA5A5A5A5, CHUNK // 4)
        hip.sync()
        t2 = time.perf_counter()
        n = min(CHUNK, 512 * MIB)
        hip.memcpy(dptr, dev_src, n, 3)  # hipMemcpyDeviceToDevice-typed D->host-mapped
        hip.sync()
        t3 = time.perf_counter()
        v1 = vm()
        t_alloc += t1 - t0
        t_touch += t2 - t1
        t_copy += t3 - t2
        print(
            f"{i:>5}{t1 - t0:9.2f}{CHUNK / MIB / (t1 - t0):13.1f}"
            f"{t2 - t1:9.2f}{CHUNK / MIB / (t2 - t1):13.1f}"
            f"{n / MIB / (t3 - t2):17.1f}"
            f"{a0 / GIB:14.1f}{(v1['pswpout'] - v0['pswpout']) * 4096 / GIB:12.2f}",
            flush=True,
        )
finally:
    print(
        f"TOTAL pinned {len(held) * CHUNK / GIB:.2f} GiB: alloc {t_alloc:.1f} s "
        f"({len(held) * CHUNK / MIB / max(t_alloc, 1e-9):.0f} MiB/s), "
        f"touch {t_touch:.1f} s ({len(held) * CHUNK / MIB / max(t_touch, 1e-9):.0f} MiB/s)",
        flush=True,
    )
    for p in held:
        try:
            hip.host_free(p)
        except Exception:
            pass
