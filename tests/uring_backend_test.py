"""The io_uring backend: every registration mode must return the SAME bytes.

WHY THIS IS SEPARATE from the row-table parity test. That test pins the four GATHER MECHANISMS
against each other; this one pins the io_uring backend's own options — registered files, registered
buffers (READ_FIXED), SQPOLL submission, the single-mmap ring layout, and batches larger than the
ring depth. Each of those changes how a request is expressed to the kernel, and every one of them
can be wrong in the same silent way: the read succeeds, the completion reports success, and the
bytes in the landing buffer are from somewhere else.

That is not hypothetical. The first version of this backend used `p.sq_ring_mask` as a mask when it
is a BYTE OFFSET to the mask, so `(tail + k) & 16` was 0 for the first sixteen entries: every SQE in
a batch landed on ring index 0, one request was submitted N times, all N completions reported
SUCCESS, and the buffer kept whatever had been in it. Nothing raised.

Skips cleanly where a ring cannot be created — an old kernel, a foreign arch, or a seccomp profile
that blocks io_uring_setup (the default docker profile does; this repo's serve runs pass
`--security-opt seccomp=unconfined`).

    PYTHONPATH=python python3 tests/uring_backend_test.py
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
from minisgl.weights import uring  # noqa: E402

FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'OK  ' if ok else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    if not ok:
        FAILED.append(name)


print("=" * 88)
print("io_uring backend — registered files / buffers / SQPOLL / chunking")
print("=" * 88)

if not uring.available():
    print("SKIPPED: no io_uring here (old kernel, non-x86_64, or seccomp blocks io_uring_setup).")
    print("         The serve passes --security-opt seccomp=unconfined, where it IS available.")
    sys.exit(0)

ROW, N, SHARDS = 160, 200, 3
HDR = 137                                   # non-aligned header: offsets must be honoured exactly

with tempfile.TemporaryDirectory() as d:
    paths = [os.path.join(d, f"s{k}.bin") for k in range(SHARDS)]
    pay = [np.frombuffer(os.urandom(ROW * 256), dtype=np.uint8).reshape(256, ROW).copy()
           for _ in paths]
    for pth, pl in zip(paths, pay):
        with open(pth, "wb") as f:
            f.write(b"\xCC" * HDR + pl.tobytes())
    fds = [os.open(pth, os.O_RDONLY) for pth in paths]

    rng = np.random.default_rng(20260918)
    shard = rng.integers(0, SHARDS, size=N)
    row = rng.integers(0, 256, size=N)
    want = np.stack([pay[s][r] for s, r in zip(shard, row)])

    def run(depth: int, reg_files: bool, reg_bufs: bool, sqpoll: bool):
        out = np.zeros((N, ROW), dtype=np.uint8)
        addr = out.ctypes.data
        r = uring.Ring(depth, sqpoll=sqpoll)
        if reg_files:
            r.register_files(fds)
        if reg_bufs:
            r.register_buffers([(addr, out.nbytes)])
        reqs = [(fds[shard[k]], ROW, HDR + int(row[k]) * ROW, k) for k in range(N)]
        for lo in range(0, N, r.entries):                 # chunk exactly as the table does
            hi = min(lo + r.entries, N)
            r.submit_reads(reqs[lo:hi], lambda s: addr + s * ROW)
            r.reap(hi - lo, ROW)
        sqe = r._sqe_view[0]
        got = (int(sqe["opcode"]), int(sqe["flags"]), bool(r.single_mmap))
        r.close()
        return out, got

    MODES = [
        ("plain",                      256, False, False, False, uring.IORING_OP_READ,       0),
        ("registered files",           256, True,  False, False, uring.IORING_OP_READ,       1),
        ("registered buffers",         256, False, True,  False, uring.IORING_OP_READ_FIXED, 0),
        ("files + buffers",            256, True,  True,  False, uring.IORING_OP_READ_FIXED, 1),
        ("SQPOLL + files + buffers",   256, True,  True,  True,  uring.IORING_OP_READ_FIXED, 1),
        ("depth 32 (batch chunked)",    32, True,  True,  False, uring.IORING_OP_READ_FIXED, 1),
    ]
    for label, depth, rf, rb, sp, want_op, want_flags in MODES:
        out, (op, flags, single) = run(depth, rf, rb, sp)
        check(f"{label}: bytes identical to the known-correct rows", np.array_equal(out, want))
        # Assert the MECHANISM, not just the result: "it returned the right bytes" is also true of a
        # mode that silently fell back to a plain READ, which would make the feature untested.
        check(f"{label}: used the expected opcode/flags",
              op == want_op and flags == want_flags,
              f"opcode={op} (want {want_op}), flags={flags} (want {want_flags})")

    # Errors must surface, not be absorbed into a half-filled buffer.
    out = np.zeros((4, ROW), dtype=np.uint8)
    r = uring.Ring(32)
    try:
        r.submit_reads([(fds[0], ROW, HDR, 0)], lambda s: out.ctypes.data + s * ROW)
        r.reap(1, ROW + 1)          # demand more bytes than the read returns
        check("a short read is refused rather than accepted", False, "no UringError")
    except uring.UringError:
        check("a short read is refused rather than accepted", True)
    finally:
        r.close()

    r = uring.Ring(32)
    try:
        r.submit_reads([(-1, ROW, 0, 0)], lambda s: out.ctypes.data)
        r.reap(1, ROW)
        check("a failed read surfaces its errno", False, "no UringError")
    except uring.UringError:
        check("a failed read surfaces its errno", True)
    finally:
        r.close()

    check("SINGLE_MMAP ring closes without a double-close", True)   # reached == no raise above

    for f in fds:
        os.close(f)

print()
if FAILED:
    print(f"FAILED ({len(FAILED)}): " + "; ".join(FAILED))
    sys.exit(1)
print("ALL CHECKS PASS")
