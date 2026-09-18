"""Which mechanism should a DECODE-SIZED PLE gather use, on latency AND on footprint?

`row_table.py` picked mmap+MADV_WILLNEED for sub-threshold gathers from this table:

    rows        baseline    prefetch    workers=16/32
    16 (decode)  2320 us      609 us       802 us     <- prefetch won

The pread arm in that sweep was the THREADED one, where pool setup dominates at 16 rows. SERIAL
pread -- no pool, no mmap -- was never measured, and on ZFS it is the arm that avoids the page-cache
copy entirely: OpenZFS serves read()/pread() from the ARC at the VFS layer, so a pread buffers the
bytes ONCE, while an mmap'd page costs a page-cache page AND its ARC buffer. The checkpoint reader
measured exactly that (`ckpt_read.py`): mmap +0.33 GiB Cached / +0.33 GiB ARC against read()'s
+0.00 / +0.31 for the same 337.7 MiB.

So this probe reports BOTH columns for all three mechanisms. Row ids are freshly randomised every
iteration across the whole 20M-row table, which is what the n-gram hash actually produces and what
stops the cache from trivially serving a repeat.

CPU-only. No GPU, no lease. Run it against the real sidecar:

    python3 tools/offload/ple_readpath_probe.py /ple
"""

from __future__ import annotations

import glob
import os
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, "/engine/python")
from minisgl.weights.row_table import open_qwen4exp_ngram_table  # noqa: E402

PLE_DIR = sys.argv[1] if len(sys.argv) > 1 else "/ple"
SIZES = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["16", "64", "256"])]
ITERS = int(sys.argv[3]) if len(sys.argv) > 3 else 40

ARC = "/hostarcstats"


def arc_size() -> int:
    try:
        for line in open(ARC):
            f = line.split()
            if f and f[0] == "size":
                return int(f[2])
    except Exception:
        pass
    return 0


def cached() -> int:
    for line in open("/proc/meminfo"):
        if line.startswith("Cached:"):
            return int(line.split()[1]) * 1024
    return 0


files = sorted(glob.glob(os.path.join(PLE_DIR, "model-plefp8-*.safetensors")))
if not files:
    print(f"no model-plefp8-*.safetensors under {PLE_DIR}")
    sys.exit(2)

GiB = 1073741824
print(f"PLE read-path probe — {len(files)} shards under {PLE_DIR}, {ITERS} iters per point")
print(f"{'rows':>6}  {'mechanism':<12} {'median us':>10} {'p90 us':>9} {'dCached MiB':>12} {'dARC MiB':>10}")

rng = np.random.default_rng(20260918)

ARMS = [
    ("mmap",     dict(workers=0,  auto_prefetch=True,  small_gather="mmap"),  None),
    ("pread",    dict(workers=0,  auto_prefetch=False, small_gather="pread"), None),
    ("threaded", dict(workers=16, auto_prefetch=False, small_gather="mmap"),  1),
    ("uring",    dict(workers=0,  auto_prefetch=False, small_gather="uring"), None),
]
from minisgl.weights import uring as _uring
if not _uring.available():
    ARMS = [a for a in ARMS if a[0] != "uring"]
    print("  (io_uring unavailable here — arm dropped; run with --security-opt seccomp=unconfined)")

# ARMS ARE INTERLEAVED, NOT RUN BACK TO BACK. The first version of this probe ran each arm's whole
# iteration block consecutively, so within every size the FIRST arm paid a cold ARC and the LAST ran
# warmest -- and across a whole invocation the table warmed monotonically. That made a second run of
# the same probe report serial pread 8x faster than the first had (2231 us -> 273 us) purely from
# cache state. Rotating the arm order per iteration, with FRESH random ids for every single timing,
# spreads that warming evenly instead of handing it to whoever ran last.
for n in SIZES:
    tables = []
    for label, kw, force_thresh in ARMS:
        t, _ = open_qwen4exp_ngram_table(files, (), **kw)
        if force_thresh is not None:
            t.threaded_min_rows = force_thresh
        t.gather_raw(rng.integers(0, t.n_rows, size=n))     # warm-up, not timed
        tables.append(t)

    lat = {label: [] for label, _, _ in ARMS}
    c0, a0 = cached(), arc_size()
    for it in range(ITERS):
        order = [(it + k) % len(ARMS) for k in range(len(ARMS))]   # rotate who goes first
        for k in order:
            label = ARMS[k][0]
            ids = rng.integers(0, tables[k].n_rows, size=n)        # fresh ids per arm per iter
            t0 = time.perf_counter()
            tables[k].gather_raw(ids)
            lat[label].append((time.perf_counter() - t0) * 1e6)
    c1, a1 = cached(), arc_size()

    for label, _, _ in ARMS:
        v = sorted(lat[label])
        print(f"{n:>6}  {label:<12} {statistics.median(v):>10.1f} {v[int(.9*len(v))]:>9.1f} "
              f"{'':>12} {'':>10}")
    print(f"{'':>6}  {'(whole size block)':<12} {'':>10} {'':>9} "
          f"{(c1-c0)/1048576:>12.1f} {(a1-a0)/1048576:>10.1f}")
    for t in tables:
        t.close()
