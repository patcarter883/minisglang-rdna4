"""All three row-gather mechanisms must return BYTE-IDENTICAL rows.

WHY THIS EXISTS. `ShardedRowTable` now has three ways to fetch rows, chosen by request size and by
`small_gather`: fancy-indexing an mmap'd view, serial `os.pread`, and threaded `os.pread`. They are a
performance/footprint trade and NOTHING ELSE — the moment one returns different bytes from another,
the PLE n-gram block is fed wrong rows and the model degrades with no error anywhere. That failure is
invisible: a wrong n-gram row is still a plausible row.

The serial pread path is the new one (added 2026-09-18 to drop mmap's page-cache copy on ZFS, where
an mmap'd page costs a page-cache page AND its ARC buffer). This pins it against the two that were
already shipping, including the cases most likely to break an offset calculation:

  * rows in the SECOND shard, and rows straddling a shard boundary — `_locate` does the global-id to
    (shard, local-row) arithmetic and an off-by-one there is silent;
  * a SPARSE, non-monotonic shard_id set, which is the real checkpoint's layout (shard 0, 1, 10..19
    in one file, not 0..12) and the case a naive slot==id assumption gets wrong;
  * repeated and out-of-order ids, which the mmap path handles by fancy-index and pread by a loop.

Synthetic table, CPU-only, no GPU, no 51 GB sidecar, runs in well under a second.

    PYTHONPATH=python python3 tests/ple_row_table_pread_parity_test.py
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
from minisgl.weights.row_table import ShardedRowTable, TensorLoc  # noqa: E402

FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'OK  ' if ok else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    if not ok:
        FAILED.append(name)


ROWS_PER_SHARD = 64
ROW_ELEMS = 32                      # F8_E4M3 -> 1 byte/elem -> 32-byte rows
HEADER_PAD = 137                    # deliberately NOT page- or row-aligned: offsets must be honoured


def build(tmp: str, n_shards: int):
    """Two shards per file, at a non-aligned base offset, filled with per-row unique bytes."""
    locs, truth = [], {}
    for s in range(n_shards):
        path = os.path.join(tmp, f"shard-{s:03d}.bin")
        payload = np.empty((ROWS_PER_SHARD, ROW_ELEMS), dtype=np.uint8)
        for r in range(ROWS_PER_SHARD):
            payload[r] = (np.arange(ROW_ELEMS, dtype=np.int64) + r * 7 + s * 31) & 0xFF
        with open(path, "wb") as f:
            f.write(b"\xAB" * HEADER_PAD)
            f.write(payload.tobytes())
        locs.append(TensorLoc(path=path, offset=HEADER_PAD,
                              shape=[ROWS_PER_SHARD, ROW_ELEMS], dtype="F8_E4M3"))
        truth[s] = payload
    return locs, truth


print("=" * 90)
print("PLE row table: mmap / serial-pread / threaded-pread must agree byte for byte")
print("=" * 90)

with tempfile.TemporaryDirectory() as tmp:
    # A SPARSE, non-monotonic shard id set — the real checkpoint's layout.
    shard_ids = [0, 1, 10, 11]
    locs, truth = build(tmp, len(shard_ids))

    def table(**kw):
        return ShardedRowTable(locs, shard_ids=shard_ids, auto_prefetch=True, **kw)

    def expect(ids):
        out = np.empty((len(ids), ROW_ELEMS), dtype=np.uint8)
        for i, gid in enumerate(ids):
            sid, local = divmod(gid, ROWS_PER_SHARD)
            out[i] = truth[shard_ids.index(sid)][local]
        return out

    CASES = {
        "first shard": [0, 1, 2, 63],
        "second shard": [64, 65, 127],
        "straddles a shard boundary": [63, 64],
        "sparse high shard ids (10, 11)": [10 * 64, 10 * 64 + 5, 11 * 64 + 63],
        "repeated and out of order": [65, 0, 65, 700, 3, 0],
        "single row": [700],
    }

    t_mmap = table(workers=0, small_gather="mmap")
    t_pread = table(workers=0, small_gather="pread")
    # io_uring is optional: an old kernel, a foreign arch or a seccomp profile that blocks the
    # setup syscall all make it genuinely unavailable, and this test must then skip that arm rather
    # than fail. `available()` deliberately does NOT swallow programming errors, so a False here
    # means unsupported, not broken.
    from minisgl.weights import uring as _uring
    HAVE_URING = _uring.available()
    t_uring = table(workers=0, small_gather="uring") if HAVE_URING else None
    print(f"  (io_uring available: {HAVE_URING})")
    t_thread = table(workers=4, small_gather="mmap")
    t_thread.threaded_min_rows = 1          # force the threaded arm even for tiny gathers

    for label, ids in CASES.items():
        want = expect(ids)
        a = t_mmap.gather_raw(ids)
        b = t_pread.gather_raw(ids)
        c = t_thread.gather_raw(ids)
        check(f"{label}: mmap matches the known-correct bytes",
              np.array_equal(a, want), f"ids={ids}")
        check(f"{label}: serial pread is byte-identical to mmap",
              np.array_equal(b, a), f"ids={ids}")
        check(f"{label}: threaded pread is byte-identical to mmap",
              np.array_equal(c, a), f"ids={ids}")
        if HAVE_URING:
            # THE ONE THAT MATTERS FOR io_uring: the ring's index arithmetic is hand-rolled, and a
            # mistake there does not raise — it reads a different file offset into the right buffer
            # slot. The first version of this backend collapsed every SQE in a batch onto ring
            # index 0 (it used the mask's BYTE OFFSET as the mask) and returned the last row N
            # times, with every completion reporting success. Only a byte comparison catches that.
            check(f"{label}: io_uring is byte-identical to mmap",
                  np.array_equal(t_uring.gather_raw(ids), a), f"ids={ids}")

    # ASYNC path: the pool writes into a caller-owned buffer while the caller does other work, so
    # a wrong fill is a partially-written buffer rather than an exception. Pin it against the
    # synchronous result for the same ids, including the sub-threshold fallback (which fills
    # synchronously) and the threaded path (which does not).
    for label, tbl in (("below threshold", t_mmap), ("threaded", t_thread)):
        for ids in ([0, 65, 700, 3], list(range(0, 127, 3)) + list(range(640, 700, 3)), [700]):
            want = tbl.gather_raw(ids)
            out = np.empty((len(ids), tbl.row_bytes), dtype=np.uint8)
            h = tbl.gather_raw_into_async(ids, out)
            got = tbl.gather_wait(h)
            check(f"async gather ({label}, n={len(ids)}) matches the synchronous gather",
                  np.array_equal(got, want) and np.array_equal(out, want))
            check(f"async handle ({label}, n={len(ids)}) reports itself drained after the wait",
                  not h.pending)

    # A mis-sized landing buffer must be refused, not silently partially filled.
    try:
        t_thread.gather_raw_into_async([1, 2, 3], np.empty((2, t_thread.row_bytes), dtype=np.uint8))
        check("a mis-sized async landing buffer is refused", False, "no ValueError")
    except ValueError:
        check("a mis-sized async landing buffer is refused", True)

    # decode_raw is the SHARED decoder: gather() must be exactly decode_raw(gather_raw()).
    ids = [0, 65, 700, 3]
    check("gather() == decode_raw(gather_raw()) — one decoder, not two",
          np.array_equal(t_mmap.gather(ids), t_mmap.decode_raw(t_mmap.gather_raw(ids)),
                         equal_nan=True))

    # Dequantised output, not just raw bytes — the decoder runs on whatever the gather returned.
    ids = [0, 65, 700, 3]
    dq_p, dq_m = t_pread.gather(ids), t_mmap.gather(ids)
    # equal_nan: F8_E4M3 has NaN encodings (0x7F/0xFF) and this fixture's arbitrary bytes hit them,
    # so a bare array_equal compares NaN to NaN and reports a difference that is not one. The bytes
    # are already pinned identical above; this checks the DECODER sees the same input either way.
    check("dequantised gather agrees across mechanisms",
          np.array_equal(dq_p, dq_m, equal_nan=True),
          f"nan count p={np.isnan(dq_p).sum()} m={np.isnan(dq_m).sum()}")

    # A bad mechanism name must be refused at construction, not silently ignored.
    try:
        table(workers=0, small_gather="preadd")
        check("an unknown small_gather is refused", False, "no ValueError raised")
    except ValueError:
        check("an unknown small_gather is refused", True)

    for t in (t_mmap, t_pread, t_thread, t_uring):
        if t is not None:
            t.close()

print()
if FAILED:
    print(f"FAILED ({len(FAILED)}): " + "; ".join(FAILED))
    sys.exit(1)
print("ALL CHECKS PASS")
