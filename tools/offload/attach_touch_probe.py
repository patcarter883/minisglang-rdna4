#!/usr/bin/env python3
"""Is `arena.first_touch` slow because the FILL is slow, or because the PAGES are not yet there?

THE NUMBER BEING EXPLAINED. The 48-layer TP=2 boot spends 241.5 s in `PinnedWeightArena.attach()`
— `arena.host_alloc` 65.7 s + `arena.first_touch` 171.8 s (rank 1: 65.8 + 175.7). It is the largest
attributed cost in the whole boot, bigger than `ckpt.h2d` (116.7 s) and bigger than `graph_capture`
(103.0 s).

WHAT THE PREVIOUS PROBE ALREADY SETTLED. `pin_probe.py`, ONE process, 18 chunks (24.12 GiB), quiet
box: alloc 64.3 s, touch 21.6 s. `alloc` reproduces the boot EXACTLY (65.7 s) and `touch` is 8x
CHEAPER than in the boot. So the fill primitive is not intrinsically slow — the per-chunk rows show
it hitting 27,000 MiB/s (PCIe speed, exactly what the code's "~70 ms per 2 GiB" comment predicts) on
most chunks and collapsing to 150-650 MiB/s on precisely the chunks whose row also shows GiB of
`pswpout`. The difference between 21.6 s and 171.8 s is the box: at TP=2 there are TWO arenas, 48.24
GiB of unevictable pinned pages appear on a 91 GiB box that already has 21 GiB in use and 27 GiB in
zram, and the boot's own vmstat delta is 39.7M pages OUT and 38.7M pages IN.

THE HYPOTHESIS THIS PROBE TESTS. `hipMemsetD32` over a freshly `hipHostMalloc`'d mapped pointer is
the FIRST WRITE to those pages, so it is the write that makes the kernel find 24 GiB of physical
memory — direct reclaim, zram compression, and page faults, all driven from a single device-issued
stream. That work is unavoidable (the pages must be committed) but it is not intrinsically serial:
zram compression is per-CPU and this box has 16 of them. Fault the pages from a CPU THREAD POOL
first, and the device fill that follows should run at the 27,000 MiB/s it reaches on an unpressured
chunk.

If that is right, the fix keeps EVERY existing guarantee — the device-issued whole-chunk fill still
happens, still last, still verified by the resweep through the device pointer — and only adds a
cheap parallel pre-fault in front of it.

ARMS (run back-to-back in one process so the box state is shared, A first so B is measured on the
MORE pressured box — i.e. the comparison is biased AGAINST the proposed fix):

  A  alloc -> hipMemsetD32(device ptr) -> sync            [what ships today]
  B  alloc -> parallel CPU memset(host ptr) -> hipMemsetD32(device ptr) -> sync   [proposed]

`PROBE_CHUNKS` defaults to 36 = 2 x 18, i.e. the same 48.24 GiB of pinned host RAM the real TP=2
boot creates, so the box-level pressure is the boot's and not a single rank's. The floor guard is
the arena's own: stop before MemAvailable would drop under 10 GiB, because this probe can destroy
the box for every other agent on it exactly as easily as a bad boot can.
"""
import ctypes
import ctypes.util
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "/engine/python")
from minisgl.weights.hipmem import get_hip  # noqa: E402

MIB = 1 << 20
GIB = 1 << 30
CHUNK = int(os.environ.get("PROBE_CHUNK_MIB", "1372")) * MIB
N = int(os.environ.get("PROBE_CHUNKS", "36"))
THREADS = int(os.environ.get("PROBE_THREADS", str(os.cpu_count() or 8)))
FLOOR = int(os.environ.get("PROBE_FLOOR_GIB", "10")) * GIB

_libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=False)
_libc.memset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
_libc.memset.restype = ctypes.c_void_p


def cpu_touch_parallel(host_ptr: int, nbytes: int, pool: ThreadPoolExecutor, nthreads: int) -> None:
    """Fault every page of the chunk from `nthreads` CPUs at once.

    `ctypes` foreign calls release the GIL, so these really do run concurrently; the point is that
    direct reclaim and zram compression are per-CPU work and a single device-issued fill drives them
    from one context.
    """
    span = (nbytes // nthreads) & ~0xFFF
    futs = []
    for t in range(nthreads):
        off = t * span
        n = nbytes - off if t == nthreads - 1 else span
        futs.append(pool.submit(_libc.memset, ctypes.c_void_p(host_ptr + off), 0, n))
    for f in futs:
        f.result()


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
print(
    f"device: {hip.device_name(0)}  chunk={CHUNK / MIB:.0f} MiB  n={N}  threads={THREADS}  "
    f"floor={FLOOR / GIB:.0f} GiB",
    flush=True,
)


def run(arm: str, cpu_pre: bool):
    held = []
    pool = ThreadPoolExecutor(max_workers=THREADS) if cpu_pre else None
    t_alloc = t_cpu = t_dev = 0.0
    print(f"\n=== ARM {arm}: alloc -> {'CPU parallel touch -> ' if cpu_pre else ''}"
          f"hipMemsetD32 -> sync ===", flush=True)
    print(f"{'chunk':>5}{'alloc_s':>9}{'cpu_s':>8}{'dev_s':>8}{'dev MiB/s':>11}"
          f"{'MemAvail GiB':>14}{'swpout GiB':>12}{'majflt k':>10}", flush=True)
    try:
        for i in range(N):
            avail = mem_avail()
            if avail - CHUNK < FLOOR:
                print(f"  STOP at chunk {i}: MemAvailable {avail / GIB:.1f} GiB would breach the "
                      f"{FLOOR / GIB:.0f} GiB floor", flush=True)
                break
            v0 = vm()
            t0 = time.perf_counter()
            hptr, dptr = hip.host_alloc(CHUNK)
            t1 = time.perf_counter()
            held.append(hptr)
            if cpu_pre:
                cpu_touch_parallel(hptr, CHUNK, pool, THREADS)
            t2 = time.perf_counter()
            hip.memset_d32(dptr, 0xA5A5A5A5, CHUNK // 4)
            hip.sync()
            t3 = time.perf_counter()
            v1 = vm()
            t_alloc += t1 - t0
            t_cpu += t2 - t1
            t_dev += t3 - t2
            print(
                f"{i:>5}{t1 - t0:9.2f}{t2 - t1:8.2f}{t3 - t2:8.2f}"
                f"{CHUNK / MIB / max(t3 - t2, 1e-9):11.1f}"
                f"{avail / GIB:14.1f}{(v1['pswpout'] - v0['pswpout']) * 4096 / GIB:12.2f}"
                f"{(v1['pgmajfault'] - v0['pgmajfault']) / 1e3:10.1f}",
                flush=True,
            )
    finally:
        gib = len(held) * CHUNK / GIB
        total = t_alloc + t_cpu + t_dev
        print(
            f"ARM {arm}: {len(held)} chunks / {gib:.2f} GiB -- alloc {t_alloc:.1f} s | "
            f"cpu_touch {t_cpu:.1f} s | dev_fill {t_dev:.1f} s | TOTAL {total:.1f} s "
            f"({gib * 1024 / max(total, 1e-9):.0f} MiB/s)",
            flush=True,
        )
        if pool is not None:
            pool.shutdown(wait=True)
        t = time.perf_counter()
        for p in held:
            try:
                hip.host_free(p)
            except Exception:
                pass
        print(f"ARM {arm}: freed in {time.perf_counter() - t:.1f} s, "
              f"MemAvailable now {mem_avail() / GIB:.1f} GiB", flush=True)
    return {"arm": arm, "chunks": len(held), "alloc_s": t_alloc, "cpu_s": t_cpu,
            "dev_s": t_dev, "total_s": t_alloc + t_cpu + t_dev}


res = [run("A_device_only", False)]
time.sleep(10.0)  # let the kernel settle so B is not measured mid-reclaim of A's teardown
res.append(run("B_cpu_prefault", True))

print("\n=== SUMMARY ===", flush=True)
for r in res:
    print(f"  {r['arm']:<16} chunks={r['chunks']:>2}  alloc {r['alloc_s']:7.1f} s  "
          f"cpu {r['cpu_s']:7.1f} s  dev {r['dev_s']:7.1f} s  TOTAL {r['total_s']:7.1f} s",
          flush=True)
if len(res) == 2 and res[0]["chunks"] == res[1]["chunks"] and res[0]["total_s"] > 0:
    d = res[0]["total_s"] - res[1]["total_s"]
    print(f"  B - A = {-d:+.1f} s ({-100 * d / res[0]['total_s']:+.1f}%) over the same "
          f"{res[0]['chunks']} chunks", flush=True)
