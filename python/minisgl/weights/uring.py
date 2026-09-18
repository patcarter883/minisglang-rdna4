"""A minimal io_uring batch-pread backend, driven straight through `ctypes`.

WHY NOT liburing. The serve image ships neither `liburing.h` nor `liburing.so` (checked
2026-09-18), so a C extension against it would mean a Dockerfile change and an image rebuild — and
a `.so` built in one image will not load in another, which is this repo's most expensive build trap.
The io_uring syscall ABI is stable and small enough to drive directly, so this module needs no
liburing, no compiler, and no build step: it is importable from the same source tree the serve
already mounts.

WHAT IT BUYS, and it is ONE thing: N scattered row reads cost ONE `io_uring_enter` instead of N
`pread` syscalls, and they are submitted without a thread pool. The threaded path gets its I/O
concurrency by parking 16 OS threads in `pread`; every one of those threads has to re-acquire the
GIL to store its result. Here a batch is submitted by the calling thread and reaped by the calling
thread, so there is no cross-thread GIL traffic at all.

WHAT IT DOES NOT BUY: the reads still cost what the drive costs. At a decode-sized gather the whole
operation is ~300 us against a ~50 ms forward, so this cannot move TPOT; it removes syscall and GIL
overhead from a path that is already a rounding error. It exists because the same machinery is what
a PREFILL-sized gather (32k rows a chunk) would use, and because `begin`/`reap` map exactly onto the
overlap split in `PLEEmbeddingSource`.

THE RISK IS SILENT WRONG DATA, which is why this is not a default and why
`tests/ple_row_table_pread_parity_test.py` pins every mechanism byte-for-byte against mmap. An
off-by-one in the submission-queue index arithmetic does not raise — it reads the wrong file offset
into the right buffer slot, and a wrong n-gram row is still a plausible n-gram row. Every completion
is checked for a short or failed read before the batch is accepted.

x86_64 only (the syscall numbers are hard-coded); `available()` reports false anywhere else, and on
any kernel that refuses the setup call (containers with a seccomp profile that blocks io_uring, which
is increasingly common — this repo's own runs pass `--security-opt seccomp=unconfined`).
"""

from __future__ import annotations

import ctypes
import errno
import mmap
import os
import platform
import struct

import numpy as np
from typing import List, Sequence

# --- syscall numbers (x86_64). NOT dunder-prefixed: a `__NR_x` name referenced inside a class
# body is mangled to `_Ring__NR_x` and raises NameError at call time, which `available()` would
# then report as "io_uring unsupported". ------------------------------------------------------------------
_NR_IO_URING_SETUP = 425
_NR_IO_URING_ENTER = 426

# --- mmap offsets, from include/uapi/linux/io_uring.h ------------------------------------------
IORING_OFF_SQ_RING = 0
IORING_OFF_CQ_RING = 0x8000000
IORING_OFF_SQES = 0x10000000

IORING_OP_READ = 22
IORING_ENTER_GETEVENTS = 1

_SQE_SIZE = 64
_CQE_SIZE = 16

_libc = ctypes.CDLL(None, use_errno=True)
_syscall = _libc.syscall
_syscall.restype = ctypes.c_long


class UringError(RuntimeError):
    pass


class _Params(ctypes.Structure):
    """struct io_uring_params — 120 bytes. Only the ring-offset blocks are read back."""

    _fields_ = [
        ("sq_entries", ctypes.c_uint32), ("cq_entries", ctypes.c_uint32),
        ("flags", ctypes.c_uint32), ("sq_thread_cpu", ctypes.c_uint32),
        ("sq_thread_idle", ctypes.c_uint32), ("features", ctypes.c_uint32),
        ("wq_fd", ctypes.c_uint32), ("resv", ctypes.c_uint32 * 3),
        # struct io_sqring_offsets
        ("sq_head", ctypes.c_uint32), ("sq_tail", ctypes.c_uint32),
        ("sq_ring_mask", ctypes.c_uint32), ("sq_ring_entries", ctypes.c_uint32),
        ("sq_flags", ctypes.c_uint32), ("sq_dropped", ctypes.c_uint32),
        ("sq_array", ctypes.c_uint32), ("sq_resv1", ctypes.c_uint32),
        ("sq_resv2", ctypes.c_uint64),
        # struct io_cqring_offsets
        ("cq_head", ctypes.c_uint32), ("cq_tail", ctypes.c_uint32),
        ("cq_ring_mask", ctypes.c_uint32), ("cq_ring_entries", ctypes.c_uint32),
        ("cq_overflow", ctypes.c_uint32), ("cq_cqes", ctypes.c_uint32),
        ("cq_flags", ctypes.c_uint32), ("cq_resv1", ctypes.c_uint32),
        ("cq_resv2", ctypes.c_uint64),
    ]


def available() -> bool:
    """True if a ring can actually be created here. Cheap, and it ANSWERS rather than guesses —
    seccomp, an old kernel and an unsupported arch all fail at the same call."""
    if platform.machine() != "x86_64":
        return False
    try:
        r = Ring(8)
    except (OSError, UringError):
        return False          # genuinely unsupported: old kernel, seccomp, no permission
    r.close()
    return True
    # Deliberately NOT a bare `except Exception`: a NameError or TypeError in this module is a BUG,
    # and swallowing it here reports a broken backend as an unavailable one, which is how the
    # name-mangling defect above survived its first smoke test.


class Ring:
    """One io_uring, sized for `entries` in-flight reads. Not thread-safe: one owner, one batch."""

    def __init__(self, entries: int = 256) -> None:
        if entries & (entries - 1):
            entries = 1 << (entries - 1).bit_length()     # the kernel requires a power of two
        self.entries = int(entries)
        p = _Params()
        fd = _syscall(_NR_IO_URING_SETUP, ctypes.c_int(self.entries), ctypes.byref(p))
        if fd < 0:
            e = ctypes.get_errno()
            raise UringError(f"io_uring_setup({self.entries}) failed: {errno.errorcode.get(e, e)}")
        self.fd = int(fd)
        self._p = p
        try:
            # p.sq_entries / p.cq_entries are COUNTS; p.sq_ring_entries / p.cq_ring_entries are
            # BYTE OFFSETS to where those counts live inside the ring. Using the offsets here sizes
            # the mapping far too small (it only survived because mmap rounds up to a page).
            sq_len = p.sq_array + p.sq_entries * 4
            cq_len = p.cq_cqes + p.cq_entries * _CQE_SIZE
            self._sq = mmap.mmap(self.fd, sq_len, flags=mmap.MAP_SHARED,
                                 prot=mmap.PROT_READ | mmap.PROT_WRITE, offset=IORING_OFF_SQ_RING)
            self._cq = mmap.mmap(self.fd, cq_len, flags=mmap.MAP_SHARED,
                                 prot=mmap.PROT_READ | mmap.PROT_WRITE, offset=IORING_OFF_CQ_RING)
            self._sqes = mmap.mmap(self.fd, p.sq_entries * _SQE_SIZE, flags=mmap.MAP_SHARED,
                                   prot=mmap.PROT_READ | mmap.PROT_WRITE, offset=IORING_OFF_SQES)
        except Exception:
            os.close(self.fd)
            raise
        # EVERY field in io_sqring_offsets/io_cqring_offsets is a BYTE OFFSET INTO THE RING, not a
        # value — including the masks. Assigning p.sq_ring_mask directly gives 16, so
        # `(tail + k) & 16` is 0 for the first sixteen entries and every SQE in a batch lands on
        # index 0: the ring then submits ONE request N times, the completions all carry the last
        # user_data, and the landing buffer keeps whatever was in it. No error is raised anywhere.
        #: Structured view over the SQE array so a batch is filled with vector stores, not a
        #: per-field Python loop. Offsets are struct io_uring_sqe's, itemsize is the full 64 bytes
        #: so untouched tail fields stay zeroed by the `sq[idx] = 0` above.
        self._sqe_dtype = np.dtype({
            "names": ["opcode", "flags", "ioprio", "fd", "off", "addr", "len", "rw_flags",
                      "user_data"],
            "formats": [np.uint8, np.uint8, np.uint16, np.int32, np.uint64, np.uint64,
                        np.uint32, np.uint32, np.uint64],
            "offsets": [0, 1, 2, 4, 8, 16, 24, 28, 32],
            "itemsize": _SQE_SIZE,
        })
        self._sqe_view = np.frombuffer(self._sqes, dtype=self._sqe_dtype, count=p.sq_entries)
        self._sq_array_view = np.frombuffer(
            self._sq, dtype=np.uint32, count=p.sq_entries, offset=p.sq_array)
        self._sq_mask = self._u32(self._sq, p.sq_ring_mask)
        self._cq_mask = self._u32(self._cq, p.cq_ring_mask)
        self._closed = False

    # -- ring word access ---------------------------------------------------
    def _u32(self, buf, off: int) -> int:
        return struct.unpack_from("<I", buf, off)[0]

    def _set_u32(self, buf, off: int, v: int) -> None:
        struct.pack_into("<I", buf, off, v & 0xFFFFFFFF)

    def submit_reads(self, reqs: Sequence[tuple], buf_addr_of) -> int:
        """Queue one IORING_OP_READ per `(fd, nbytes, offset, slot)` and submit them all.

        Returns the number submitted. Does NOT wait — call `reap`. `buf_addr_of(slot)` gives the
        destination address for that request, so the caller keeps ownership of the landing buffer.
        """
        if self._closed:
            raise UringError("ring is closed")
        n = len(reqs)
        if n > self.entries:
            raise UringError(f"batch of {n} exceeds ring depth {self.entries}")
        p = self._p
        tail = self._u32(self._sq, p.sq_tail)
        idx = (np.arange(n, dtype=np.uint32) + tail) & np.uint32(self._sq_mask)
        # VECTORISED SQE FILL. The obvious loop — one struct.pack_into per field per request — costs
        # five GIL-held Python calls per row, which is what a batch pread costs MINUS the syscall.
        # MEASURED 2026-09-18 on the 51.2 GB table: with the loop, io_uring was 3130 us at 256 rows
        # against the thread pool's 2484 (26% WORSE); the syscall saving is real but the packing ate
        # it. Writing the fields as numpy columns over a structured view of the SQE mmap makes the
        # per-row cost a handful of vector stores instead.
        sq = self._sqe_view
        sq[idx] = 0                       # SQEs are reused; stale fields are live fields
        sq["opcode"][idx] = IORING_OP_READ
        sq["fd"][idx] = np.asarray([r[0] for r in reqs], dtype=np.int32)
        sq["off"][idx] = np.asarray([r[2] for r in reqs], dtype=np.uint64)
        sq["addr"][idx] = np.asarray([buf_addr_of(r[3]) for r in reqs], dtype=np.uint64)
        sq["len"][idx] = np.asarray([r[1] for r in reqs], dtype=np.uint32)
        sq["user_data"][idx] = np.asarray([r[3] for r in reqs], dtype=np.uint64)
        self._sq_array_view[idx] = idx
        self._set_u32(self._sq, p.sq_tail, tail + n)
        got = _syscall(_NR_IO_URING_ENTER, ctypes.c_int(self.fd), ctypes.c_uint(n),
                       ctypes.c_uint(0), ctypes.c_uint(0), None, ctypes.c_size_t(0))
        if got < 0:
            e = ctypes.get_errno()
            raise UringError(f"io_uring_enter(submit={n}) failed: {errno.errorcode.get(e, e)}")
        return int(got)

    def reap(self, n: int, expect_bytes: int) -> List[int]:
        """Block until `n` completions are available; return their user_data (row indices).

        Every completion is checked: a negative `res` is the read's errno, and a short read is
        refused rather than accepted as a partly-filled row. Both would otherwise be invisible —
        the destination buffer simply keeps whatever was there.
        """
        if self._closed:
            raise UringError("ring is closed")
        p = self._p
        done: List[int] = []
        while len(done) < n:
            head = self._u32(self._cq, p.cq_head)
            tail = self._u32(self._cq, p.cq_tail)
            if head == tail:
                r = _syscall(_NR_IO_URING_ENTER, ctypes.c_int(self.fd), ctypes.c_uint(0),
                             ctypes.c_uint(n - len(done)), ctypes.c_uint(IORING_ENTER_GETEVENTS),
                             None, ctypes.c_size_t(0))
                if r < 0:
                    e = ctypes.get_errno()
                    if e == errno.EINTR:
                        continue
                    raise UringError(f"io_uring_enter(wait) failed: {errno.errorcode.get(e, e)}")
                continue
            while head != tail and len(done) < n:
                off = p.cq_cqes + (head & self._cq_mask) * _CQE_SIZE
                user_data, res, _flags = struct.unpack_from("<QiI", self._cq, off)
                if res < 0:
                    self._set_u32(self._cq, p.cq_head, head + 1)
                    raise UringError(
                        f"row {user_data}: read failed: {errno.errorcode.get(-res, -res)}")
                if res != expect_bytes:
                    self._set_u32(self._cq, p.cq_head, head + 1)
                    raise UringError(
                        f"row {user_data}: short read, {res} of {expect_bytes} bytes")
                done.append(int(user_data))
                head += 1
            self._set_u32(self._cq, p.cq_head, head)
        return done

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for m in ("_sqes", "_cq", "_sq"):
            try:
                getattr(self, m).close()
            except Exception:  # noqa: BLE001
                pass
        try:
            os.close(self.fd)
        except OSError:
            pass

    def __enter__(self) -> "Ring":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


__all__ = ["Ring", "UringError", "available"]
