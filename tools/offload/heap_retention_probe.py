#!/usr/bin/env python3
"""Does Stage B's 222k transient CPU clones grow the heap and NEVER give it back?

THE SIGNAL BEING EXPLAINED. `stage_b_peak_host_rss` is 29.21 GiB against Stage B's design claim of
a 1.465 GiB live set. Subtracting the 24.12 GiB pinned arena still leaves ~5 GiB of transient anon,
and rank 0's `VmHWM` reaches 35.04 GiB. On a 91 GiB box that already carries 22 GiB of other
tenants and 26 GiB in zram, and where BOTH ranks pin 24.12 GiB, every GiB of avoidable peak is
bought back at the 8-190x degradation factor this round measured on every memory-touching phase.

THE MECHANISM. `_shard_qwen4_exp` ends in `.clone()`, so Stage B allocates and frees ~222,252 CPU
tensors averaging 176 KiB (and up to a few MiB). glibc's mmap threshold is DYNAMIC: it starts at
128 KiB, but every time an mmap'd block is freed glibc raises the threshold towards 32 MiB on the
theory that the program will reuse that size from the heap. Once the threshold passes the clone
size, the clones come from the brk heap instead — and the heap only ever shrinks from the TOP, so
one long-lived allocation above a freed block pins the whole span. The result is a heap that grows
to the high-water mark of concurrently-live clones plus fragmentation, and stays there for the
process lifetime.

If that is what is happening, `mallopt(M_MMAP_THRESHOLD, 128 KiB)` pins the threshold so every
clone is its own mapping that `munmap` returns to the OS immediately. It is a five-line, allocator-
only change: the same bytes, in the same order, at the same addresses relative to each tensor.

This probe reproduces the allocation PATTERN (not the data) with torch's own CPU allocator and
reports RSS after the same number of alloc/free cycles, with and without the mallopt.
"""
import ctypes
import ctypes.util
import os
import sys

sys.path.insert(0, "/engine/python")
import torch  # noqa: E402

MIB = 1 << 20
N = int(os.environ.get("HEAP_N", "222252"))
# The boot's mean leaf is 176 KiB; the spread matters because it is the spread that fragments.
SIZES_KIB = [40, 80, 176, 176, 320, 640, 1280]

_libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
_libc.mallopt.argtypes = [ctypes.c_int, ctypes.c_int]
M_TRIM_THRESHOLD, M_MMAP_THRESHOLD = -1, -3


def rss_kb(field="VmRSS"):
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith(field):
                return int(line.split()[1])
    return 0


def churn(n):
    """Stage B's shape: allocate a clone, hand it downstream, drop it. A few stay alive at once
    (`fold_buf`, `concat_buf`), which is what leaves holes in the heap."""
    live = []
    for i in range(n):
        kib = SIZES_KIB[i % len(SIZES_KIB)]
        t = torch.empty(kib * 1024, dtype=torch.uint8)
        t[0] = 1
        t[-1] = 2
        live.append(t)
        if len(live) > 3:
            live.pop(0)
    live.clear()


mode = os.environ.get("HEAP_MODE", "default")
if mode == "mallopt":
    rc1 = _libc.mallopt(M_MMAP_THRESHOLD, 128 * 1024)
    rc2 = _libc.mallopt(M_TRIM_THRESHOLD, 128 * 1024)
    print(f"mallopt(M_MMAP_THRESHOLD, 128K)={rc1}  mallopt(M_TRIM_THRESHOLD, 128K)={rc2}")

base = rss_kb()
print(f"mode={mode}  N={N}  RSS before {base / 1024:.1f} MiB", flush=True)
churn(N)
after = rss_kb()
hwm = rss_kb("VmHWM")
print(f"mode={mode}  RSS after  {after / 1024:.1f} MiB   VmHWM {hwm / 1024:.1f} MiB   "
      f"RETAINED {(after - base) / 1024:.1f} MiB", flush=True)
