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
_NR_IO_URING_REGISTER = 427

# --- mmap offsets, from include/uapi/linux/io_uring.h ------------------------------------------
IORING_OFF_SQ_RING = 0
IORING_OFF_CQ_RING = 0x8000000
IORING_OFF_SQES = 0x10000000

IORING_OP_READ_FIXED = 4
IORING_OP_READ = 22

IORING_ENTER_GETEVENTS = 1
IORING_ENTER_SQ_WAKEUP = 2

IORING_SETUP_SQPOLL = 2

IOSQE_FIXED_FILE = 1

IORING_REGISTER_BUFFERS = 0
IORING_UNREGISTER_BUFFERS = 1
IORING_REGISTER_FILES = 2
IORING_UNREGISTER_FILES = 3

IORING_FEAT_SINGLE_MMAP = 1
IORING_SQ_NEED_WAKEUP = 1

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

    def __init__(self, entries: int = 256, *, sqpoll: bool = False,
                 sq_thread_idle_ms: int = 1000) -> None:
        if entries & (entries - 1):
            entries = 1 << (entries - 1).bit_length()     # the kernel requires a power of two
        self.entries = int(entries)
        p = _Params()
        if sqpoll:
            # A kernel thread polls the submission queue, so `submit_reads` needs NO SYSCALL at all
            # unless that thread has idled out (IORING_SQ_NEED_WAKEUP). Verified usable unprivileged
            # on this kernel; older kernels required CAP_SYS_NICE, hence the caller-visible flag
            # rather than an unconditional default.
            p.flags = IORING_SETUP_SQPOLL
            p.sq_thread_idle = int(sq_thread_idle_ms)
        self.sqpoll = bool(sqpoll)
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
            # IORING_FEAT_SINGLE_MMAP (5.4+): the CQ ring lives inside the SQ mapping, so ONE mmap
            # covers both and every cq_* offset is relative to the same base. Two mappings still
            # work on such a kernel, but they cost an extra VMA for nothing.
            self.single_mmap = bool(p.features & IORING_FEAT_SINGLE_MMAP)
            if self.single_mmap:
                self._sq = mmap.mmap(self.fd, max(sq_len, cq_len), flags=mmap.MAP_SHARED,
                                     prot=mmap.PROT_READ | mmap.PROT_WRITE,
                                     offset=IORING_OFF_SQ_RING)
                self._cq = self._sq
            else:
                self._sq = mmap.mmap(self.fd, sq_len, flags=mmap.MAP_SHARED,
                                     prot=mmap.PROT_READ | mmap.PROT_WRITE,
                                     offset=IORING_OFF_SQ_RING)
                self._cq = mmap.mmap(self.fd, cq_len, flags=mmap.MAP_SHARED,
                                     prot=mmap.PROT_READ | mmap.PROT_WRITE,
                                     offset=IORING_OFF_CQ_RING)
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
                      "user_data", "buf_index"],
            "formats": [np.uint8, np.uint8, np.uint16, np.int32, np.uint64, np.uint64,
                        np.uint32, np.uint32, np.uint64, np.uint16],
            "offsets": [0, 1, 2, 4, 8, 16, 24, 28, 32, 40],
            "itemsize": _SQE_SIZE,
        })
        self._sqe_view = np.frombuffer(self._sqes, dtype=self._sqe_dtype, count=p.sq_entries)
        self._cqe_dtype = np.dtype({
            "names": ["user_data", "res", "flags"],
            "formats": [np.uint64, np.int32, np.uint32],
            "offsets": [0, 8, 12],
            "itemsize": _CQE_SIZE,
        })
        self._cqe_view = np.frombuffer(
            self._cq, dtype=self._cqe_dtype, count=p.cq_entries, offset=p.cq_cqes)
        self._sq_array_view = np.frombuffer(
            self._sq, dtype=np.uint32, count=p.sq_entries, offset=p.sq_array)
        self._sq_mask = self._u32(self._sq, p.sq_ring_mask)
        self._cq_mask = self._u32(self._cq, p.cq_ring_mask)
        self._closed = False
        #: Registered-file table: actual fd -> index, set by `register_files`. Empty == unused.
        self._file_index = {}
        self._fd_lut = None
        #: Registered-buffer table: (addr, length) -> index, set by `register_buffers`.
        self._buf_index = {}
        self._buf_ranges = []

    # -- ring word access ---------------------------------------------------
    def _u32(self, buf, off: int) -> int:
        return struct.unpack_from("<I", buf, off)[0]

    def _set_u32(self, buf, off: int, v: int) -> None:
        struct.pack_into("<I", buf, off, v & 0xFFFFFFFF)

    # -- registration -------------------------------------------------------
    #
    # Both of these move per-operation work OUT of the submission path and do it ONCE:
    #   * registered FILES  — the kernel resolves and refcounts the struct file at registration, so
    #     an SQE carries an index instead of an fd and skips the per-op lookup/fget/fput.
    #   * registered BUFFERS — the pages are pinned and the iovec mapped once, so a READ_FIXED skips
    #     the per-op get_user_pages/unpin of the destination.
    # This path reads the SAME ten shard fds into the SAME landing buffer on every gather, which is
    # exactly the case they exist for.

    def register_files(self, fds: Sequence[int]) -> None:
        """Register `fds` once; later reads that name one of them use IOSQE_FIXED_FILE."""
        arr = (ctypes.c_int * len(fds))(*[int(f) for f in fds])
        r = _syscall(_NR_IO_URING_REGISTER, ctypes.c_int(self.fd),
                     ctypes.c_uint(IORING_REGISTER_FILES), ctypes.byref(arr),
                     ctypes.c_uint(len(fds)))
        if r < 0:
            e = ctypes.get_errno()
            raise UringError(f"register_files({len(fds)}) failed: {errno.errorcode.get(e, e)}")
        self._file_index = {int(f): i for i, f in enumerate(fds)}
        # Dense fd -> registered-index table; -1 means "not registered". fds are small integers, so
        # this is a few hundred bytes and turns the per-batch remap into one numpy gather.
        self._fd_lut = np.full(max(int(f) for f in fds) + 1, -1, dtype=np.int32)
        for i, f in enumerate(fds):
            self._fd_lut[int(f)] = i

    def register_buffers(self, bufs: Sequence[tuple]) -> None:
        """Register `(addr, length)` landing buffers; reads into them become IORING_OP_READ_FIXED."""
        class _IoVec(ctypes.Structure):
            _fields_ = [("base", ctypes.c_void_p), ("len", ctypes.c_size_t)]

        arr = (_IoVec * len(bufs))()
        for i, (addr, ln) in enumerate(bufs):
            arr[i].base, arr[i].len = int(addr), int(ln)
        r = _syscall(_NR_IO_URING_REGISTER, ctypes.c_int(self.fd),
                     ctypes.c_uint(IORING_REGISTER_BUFFERS), ctypes.byref(arr),
                     ctypes.c_uint(len(bufs)))
        if r < 0:
            e = ctypes.get_errno()
            raise UringError(f"register_buffers({len(bufs)}) failed: {errno.errorcode.get(e, e)}")
        self._buf_index = {(int(a), int(ln)): i for i, (a, ln) in enumerate(bufs)}
        self._buf_ranges = [(int(a), int(a) + int(ln), i) for i, (a, ln) in enumerate(bufs)]

    def _fixed_buf_for(self, addr: int, nbytes: int):
        """Index of the registered buffer wholly containing [addr, addr+nbytes), else None."""
        for lo, hi, idx in self._buf_ranges:
            if lo <= addr and addr + nbytes <= hi:
                return idx
        return None

    def submit_reads(self, reqs: Sequence[tuple], buf_addr_of) -> int:
        """Queue one IORING_OP_READ per `(fd, nbytes, offset, slot)` and submit them all.

        Returns the number submitted. Does NOT wait — call `reap`. `buf_addr_of(slot)` gives the
        destination address for that request, so the caller keeps ownership of the landing buffer.
        """
        if self._closed:
            raise UringError("ring is closed")
        return self.submit_reads_arrays(
            np.asarray([r[0] for r in reqs], dtype=np.int32),
            np.asarray([r[2] for r in reqs], dtype=np.uint64),
            np.asarray([buf_addr_of(r[3]) for r in reqs], dtype=np.uint64),
            np.asarray([r[1] for r in reqs], dtype=np.uint32),
            np.asarray([r[3] for r in reqs], dtype=np.uint64),
        )

    def submit_reads_arrays(self, fds, offsets, addrs, lens, user_data) -> int:
        """Array form of `submit_reads`: NO PER-ROW PYTHON ANYWHERE.

        `submit_reads` above takes a list of (fd, nbytes, offset, slot) tuples, which forces five
        O(N) Python passes before a single vector store happens — building the tuples, then one
        comprehension per column. MEASURED on ARC-resident data, where the drive is out of the
        picture and only per-batch overhead remains, the fill was 1073 us of a 4735 us 2048-row
        batch. That is the whole of what a C submission path could have removed, and it did not need
        C: it needed the caller to stop materialising tuples. With this API plus the contiguous SQE
        fast path the fill is 531.6 us — HALF, in pure Python:

            rows    fill (tuple API)   fill (this API)
              16         29.7 us            15.2 us
             256        162.3 us            74.6 us
            2048       1073   us           531.6 us

        What is left is the vector stores themselves, so a C submission path is now worth ~8% of a
        real gather rather than ~18%.

        Every argument is a numpy array of length n. Callers that already hold their data as columns
        (the row table does — `_locate` returns arrays) should use this and never build tuples.
        """
        if self._closed:
            raise UringError("ring is closed")
        n = int(fds.shape[0])
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
        raw_fds = fds

        # Fixed FILE: the whole batch uses it only if every fd in it is registered — a partially
        # fixed batch would need per-row flags, and this path always reads the same shard set.
        # The membership test and the fd->index remap are BOTH vectorised: written as
        # `all(f in self._file_index for f in raw_fds)` plus a list comprehension they are two more
        # O(N) Python passes over the batch, and MEASURED 2026-09-18 that made the registered arm
        # 8-10% SLOWER than the same backend with registration off at every size. `_fd_lut` is a
        # dense lookup table built once at registration, so the remap is a single gather.
        use_fixed_file = False
        fd_col = raw_fds
        if self._fd_lut is not None and raw_fds.size:
            lo, hi = int(raw_fds.min()), int(raw_fds.max())
            if lo >= 0 and hi < self._fd_lut.size:
                mapped = self._fd_lut[raw_fds]
                if mapped.min() >= 0:
                    use_fixed_file, fd_col = True, mapped

        # Fixed BUFFER: only when every destination falls inside one registered range.
        bidx = None
        if self._buf_ranges:
            # VECTORISED containment test. The obvious form — a set comprehension calling
            # `_fixed_buf_for` per request — is a Python loop over the whole batch, and it cost more
            # than registration saved: MEASURED 2026-09-18, registered was 3018 us at 256 rows
            # against 2947 for the same backend with registration OFF, i.e. the feature made it
            # SLOWER. Same lesson as the SQE fill: at this batch size the kernel-side win is small
            # and any per-row Python erases it.
            ends = addrs + lens.astype(np.uint64)
            for lo, hi, i in self._buf_ranges:
                if addrs.min() >= lo and ends.max() <= hi:
                    bidx = i
                    break

        sq = self._sqe_view
        # CONTIGUOUS FAST PATH. `idx` wraps only when a batch straddles the ring's end, which is the
        # uncommon case; the rest of the time the entries are consecutive and a SLICE assignment
        # replaces a fancy-index scatter. numpy's fancy indexing allocates an index array and
        # scatters element by element, while a slice is a strided copy — MEASURED below.
        start = int(idx[0])
        contiguous = (start + n <= sq.shape[0]) and (int(idx[-1]) == start + n - 1)
        sel = slice(start, start + n) if contiguous else idx
        sq[sel] = 0                       # SQEs are reused; stale fields are live fields
        sq["opcode"][sel] = IORING_OP_READ_FIXED if bidx is not None else IORING_OP_READ
        sq["flags"][sel] = IOSQE_FIXED_FILE if use_fixed_file else 0
        sq["fd"][sel] = fd_col
        sq["off"][sel] = offsets
        sq["addr"][sel] = addrs
        sq["len"][sel] = lens
        sq["user_data"][sel] = user_data
        if bidx is not None:
            sq["buf_index"][sel] = bidx
        self._sq_array_view[sel] = idx
        self._set_u32(self._sq, p.sq_tail, tail + n)
        if self.sqpoll:
            # The kernel poller picks the entries up on its own. A syscall is needed ONLY if it has
            # idled out, which it advertises in sq_flags. This is the whole point of SQPOLL: in the
            # steady state submission costs zero syscalls.
            if self._u32(self._sq, p.sq_flags) & IORING_SQ_NEED_WAKEUP:
                r = _syscall(_NR_IO_URING_ENTER, ctypes.c_int(self.fd), ctypes.c_uint(0),
                             ctypes.c_uint(0), ctypes.c_uint(IORING_ENTER_SQ_WAKEUP),
                             None, ctypes.c_size_t(0))
                if r < 0:
                    e = ctypes.get_errno()
                    raise UringError(f"io_uring_enter(wakeup) failed: {errno.errorcode.get(e, e)}")
            return n
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
            # VECTORISED REAP. Unpacking one CQE per iteration is an O(N) Python pass over the
            # batch — the same shape as the SQE fill, which cost 26% at 256 rows before it was
            # turned into vector stores. The completions are read as numpy columns instead, and the
            # validity checks are two whole-array comparisons.
            avail = min(tail - head, n - len(done))
            pos = (np.arange(avail, dtype=np.uint32) + head) & np.uint32(self._cq_mask)
            cq = self._cqe_view
            res = cq["res"][pos]
            ud = cq["user_data"][pos]
            bad = np.flatnonzero(res != expect_bytes)
            if bad.size:
                k = int(bad[0])
                r0, u0 = int(res[k]), int(ud[k])
                self._set_u32(self._cq, p.cq_head, head + k + 1)
                if r0 < 0:
                    raise UringError(
                        f"row {u0}: read failed: {errno.errorcode.get(-r0, -r0)}")
                raise UringError(f"row {u0}: short read, {r0} of {expect_bytes} bytes")
            done.extend(ud.tolist())
            head += avail
            self._set_u32(self._cq, p.cq_head, head)
        return done

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        seen = set()
        for m in ("_sqes", "_cq", "_sq"):
            mm = getattr(self, m, None)
            if mm is None or id(mm) in seen:
                continue          # under SINGLE_MMAP _cq IS _sq; closing it twice raises
            seen.add(id(mm))
            try:
                mm.close()
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
