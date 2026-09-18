"""ctypes binding for the host-tier HIP entry points. Styled on `utils/roctx.py`, but WITHOUT its
graceful degradation: a missing symbol raises.

WHY NO GRACEFUL DEGRADATION. `roctx.py` degrades because a missing profiler marker costs nothing. A
missing `hipHostGetDevicePointer` costs correctness: the fallback would be to hand the kernel a
plain host pointer, which is not device-addressable, and the failure surfaces as a GPU page fault
(SIGABRT) with no attribution. Fail at bind time, in one line, with the symbol name.

WHAT IS BOUND — and, more importantly, WHAT IS NOT.

Bound: `hipHostMalloc` / `hipHostGetDevicePointer` / `hipHostFree` (the host tier), `hipMalloc` /
`hipFree` (the layer-granular device tier that M2 needs and that Phase 0 promoted onto M1's critical
path as a *capacity* prerequisite), plus the plumbing needed to validate them
(`hipMemcpy`, `hipMemsetD32`, `hipDeviceSynchronize`, `hipMemGetInfo`, device identity).

**NOT bound: `hipMemAddressReserve` / `hipMemCreate` / `hipMemMap` / `hipMemSetAccess`.** Phase 0
established, with three independent probes, that `hipMemCreate(location.type=hipMemLocationTypeHost)`
**silently returns device VRAM** while echoing "Host" back from the properties query, and that
`hipMemUnmap`→`hipMemMap` at a used VA serves the *stale physical page* with every call returning
`hipSuccess`. There is no mixed-media VA on this box. Binding those symbols here would be an
invitation to rebuild the mechanism that does not exist.

**NEVER gate anything on `hipPointerGetAttributes`.** P5b measured it reporting
`memory_type = 1 ("Device")` for the *real* host arena — the exact inverse of the Phase 0 trap where
it echoed "Host" for VRAM. It is unreliable in BOTH directions. It is not bound.

RULE R1 — FREEZE. Every entry point that creates or destroys a mapping raises after `freeze()`.
Mapping after boot is a correctness bug, not just untidy: a mapping made after
`engine.py:_determine_num_pages` is invisible to `_prefill_budget_now`'s `reserved − allocated`
correction and silently collapses the prefill budget, with the warning pointing the operator at
`--memory-ratio`. Freeze makes that unrepresentable instead of documented.
"""

from __future__ import annotations

import ctypes
import errno
import ctypes.util
import threading
from contextlib import contextmanager
from typing import Iterator, Tuple

# --- HIP constants (from /opt/rocm/include/hip/hip_runtime_api.h) --------------------------------
hipHostMallocPortable = 0x1
hipHostMallocMapped = 0x2
hipHostMallocNonCoherent = 0x80000000

hipMemcpyHostToDevice = 1
hipMemcpyDeviceToHost = 2
hipMemcpyDeviceToDevice = 3

# The flags P1/P5b measured: Portable so the mapping is valid for every device in the process
# (TP ranks share a process only under some launchers, but Portable costs nothing), Mapped so
# `hipHostGetDevicePointer` has something to return. NonCoherent is deliberately NOT set —
# unknown #6 was answered by measuring the COHERENT variant, on both cards, in both directions,
# with no explicit flush; switching to NonCoherent would invalidate that result.
HOST_ALLOC_FLAGS = hipHostMallocPortable | hipHostMallocMapped


_LIBC = ctypes.CDLL(None, use_errno=True)
#: Set once we have warned, so a box without the headroom logs the reason ONCE, not per chunk.
_MLOCK_WARNED = False


def _mlock_region(addr: int, nbytes: int) -> None:
    """`mlock` a host allocation so the kernel cannot page it out.

    THE ARENA IS NOT PINNED BY THE DRIVER, WHICH IS THE OPPOSITE OF WHAT ITS NAME SAYS. KFD's
    userptr path (`KFD_IOC_ALLOC_MEM_FLAGS_USERPTR`, which is what `hipHostMalloc` takes here) is
    HMM-managed: it registers an MMU notifier and re-validates on invalidation rather than taking a
    page pin. `/proc/<pid>/status` confirms it — VmPin and VmLck are both 0 for a rank holding a
    29.9 GiB resident arena. The pages are ordinary swappable anonymous memory.

    MEASURED CONSEQUENCE, 2026-09-18, qwen4exp TP=2 (2 x 27.94 GiB arena on a 91.8 GiB box): a
    request sat with `prefill_computed_tokens_total` at ZERO for 12+ minutes while the box moved
    50-100 MB/s of swap in BOTH directions continuously. Swap occupancy sat frozen at 55.0 GiB
    (37.0 GiB of it SwapCached — faulted back in but with the slot retained, so occupancy cannot
    fall), and the ranks' resident anon oscillated 46.5 <-> 62.7 GiB: the arena being evicted and
    dragged back, over and over, before prefill computed a single token.

    The existing swap tripwire cannot catch this. It watches ALLOCATION; this happens long after,
    on first use. So lock the pages at the point they are created.

    NOT FATAL on failure. A box without the headroom should degrade to the old behaviour with a loud
    line, not refuse to boot — but it warns once, because silently swapping a 56 GiB arena is the
    failure this exists to prevent and it must never be inferred from a slow serve again.
    """
    global _MLOCK_WARNED
    if _LIBC.mlock(ctypes.c_void_p(addr), ctypes.c_size_t(nbytes)) == 0:
        return
    if not _MLOCK_WARNED:
        _MLOCK_WARNED = True
        e = ctypes.get_errno()
        print(
            f"[hipmem] mlock({nbytes / (1 << 30):.2f} GiB) failed: {errno.errorcode.get(e, e)}. "
            f"The host arena is HMM-managed userptr memory, NOT driver-pinned, so unlocked pages "
            f"are swappable — on this box that showed up as 50-100 MB/s of bidirectional swap "
            f"before prefill computed a single token. Raise RLIMIT_MEMLOCK (ulimit -l) or shrink "
            f"the arena.",
            flush=True,
        )


class HipError(RuntimeError):
    pass


class HipFrozenError(RuntimeError):
    """A mapping entry point was called after `freeze()` (rule R1)."""


_FROZEN = False
_FREEZE_REASON = ""
# Depth, NOT a saved-value stack. `teardown_window` used to save `_FROZEN` on entry and restore it
# on exit; two overlapping windows then interleave as
#     A: was=True  -> _FROZEN=False
#     B: was=False -> _FROZEN=False
#     A exit: _FROZEN=True
#     B exit: _FROZEN=False        <-- R1 silently and PERMANENTLY defeated
# and nothing ever raises again. torch's pluggable-allocator callback can fire on any thread, and
# `ArenaMemPool._fallback` opens a window from inside it, so the overlap is reachable at runtime,
# not just in tests. A depth counter is order-independent: the window is open iff depth > 0.
_TEARDOWN_DEPTH = 0
_LOCK = threading.RLock()


def freeze(reason: str = "boot complete") -> None:
    """Close the mapping window. Idempotent."""
    global _FROZEN, _FREEZE_REASON
    with _LOCK:
        _FROZEN = True
        _FREEZE_REASON = reason


def is_frozen() -> bool:
    with _LOCK:
        return _FROZEN and _TEARDOWN_DEPTH == 0


def _check_not_frozen(what: str) -> None:
    if is_frozen():
        raise HipFrozenError(
            f"{what} was called after the weight-arena mapping window closed "
            f"({_FREEZE_REASON!r}). Rule R1: all pinning happens between post_load() and "
            f"_determine_num_pages(); a mapping made later is invisible to the prefill budget's "
            f"reserved-minus-allocated correction and silently collapses it."
        )


@contextmanager
def teardown_window(who: str) -> Iterator[None]:
    """Temporarily reopen the window so an arena can free its chunks at shutdown or in a test.

    Deliberately explicit and named: unfreezing is exactly the thing R1 forbids, so it may not
    happen implicitly inside a destructor path that a reader would skim past.

    Nesting/overlap safe: it bumps a depth counter rather than saving and restoring `_FROZEN`, so
    the latch cannot be lost by two windows interleaving (see the `_TEARDOWN_DEPTH` note above).
    `_FROZEN` itself is never cleared here, so the freeze reason survives for the error message.
    """
    global _TEARDOWN_DEPTH
    with _LOCK:
        _TEARDOWN_DEPTH += 1
    try:
        yield
    finally:
        with _LOCK:
            _TEARDOWN_DEPTH -= 1
            assert _TEARDOWN_DEPTH >= 0, f"teardown_window depth underflow after {who}"


def teardown_depth() -> int:
    """Diagnostics/tests only."""
    with _LOCK:
        return _TEARDOWN_DEPTH


class Hip:
    """Thin, explicit binding. One instance per process (`get_hip()`)."""

    _SIGNATURES = (
        ("hipGetDeviceCount", [ctypes.POINTER(ctypes.c_int)]),
        ("hipSetDevice", [ctypes.c_int]),
        ("hipGetDevice", [ctypes.POINTER(ctypes.c_int)]),
        ("hipDeviceGetPCIBusId", [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]),
        ("hipDeviceGetName", [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]),
        ("hipMemGetInfo", [ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]),
        ("hipHostMalloc", [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint]),
        ("hipHostGetDevicePointer", [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p,
                                     ctypes.c_uint]),
        ("hipHostFree", [ctypes.c_void_p]),
        ("hipMalloc", [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]),
        ("hipFree", [ctypes.c_void_p]),
        ("hipMemcpy", [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]),
        ("hipMemsetD32", [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]),
        ("hipDeviceSynchronize", []),
    )

    def __init__(self, lib_path: str | None = None) -> None:
        path = lib_path or ctypes.util.find_library("amdhip64") or "libamdhip64.so"
        try:
            self.lib = ctypes.CDLL(path)
        except OSError as exc:
            raise HipError(
                f"cannot load {path!r}: {exc}. The weight arena needs the HIP runtime; this module "
                f"must not be imported on a host without ROCm."
            ) from exc
        self.so_path = path
        self.lib.hipGetErrorString.restype = ctypes.c_char_p
        self.lib.hipGetErrorString.argtypes = [ctypes.c_int]
        for name, args in self._SIGNATURES:
            if not hasattr(self.lib, name):
                raise HipError(
                    f"{path} lacks {name} — the pinned weight arena cannot be built on this "
                    f"runtime. No fallback exists: the alternative is handing a kernel a "
                    f"non-device-addressable pointer, which faults with no attribution."
                )
            fn = getattr(self.lib, name)
            fn.argtypes = args
            fn.restype = ctypes.c_int

    # -- errors ---------------------------------------------------------------

    def err(self, rc: int) -> str:
        try:
            s = self.lib.hipGetErrorString(ctypes.c_int(rc))
            return s.decode() if s else f"rc={rc}"
        except Exception:  # pragma: no cover - defensive
            return f"rc={rc}"

    def check(self, rc: int, what: str) -> None:
        if rc != 0:
            raise HipError(f"{what} failed: rc={rc} ({self.err(rc)})")

    # -- device identity ------------------------------------------------------

    def device_count(self) -> int:
        n = ctypes.c_int(0)
        self.check(self.lib.hipGetDeviceCount(ctypes.byref(n)), "hipGetDeviceCount")
        return int(n.value)

    def set_device(self, dev: int) -> int:
        """Pin *libamdhip64's* current device for these ctypes calls.

        `torch.cuda.set_device` does not necessarily bind this thread's HIP context, and a mismatch
        attributes the arena to the wrong physical card — which on this box is a 2x bandwidth error
        (card 0 Gen5 x8 / card 1 Gen4 x8), not a cosmetic one.
        """
        self.check(self.lib.hipSetDevice(ctypes.c_int(int(dev))), f"hipSetDevice({dev})")
        cur = ctypes.c_int(-1)
        self.lib.hipGetDevice(ctypes.byref(cur))
        return int(cur.value)

    def current_device(self) -> int:
        cur = ctypes.c_int(-1)
        self.check(self.lib.hipGetDevice(ctypes.byref(cur)), "hipGetDevice")
        return int(cur.value)

    def pci_bus_id(self, dev: int) -> str:
        buf = ctypes.create_string_buffer(64)
        rc = self.lib.hipDeviceGetPCIBusId(buf, ctypes.c_int(64), ctypes.c_int(int(dev)))
        return buf.value.decode(errors="replace") if rc == 0 else f"<rc={rc}>"

    def device_name(self, dev: int) -> str:
        buf = ctypes.create_string_buffer(256)
        rc = self.lib.hipDeviceGetName(buf, ctypes.c_int(256), ctypes.c_int(int(dev)))
        return buf.value.decode(errors="replace") if rc == 0 else f"<rc={rc}>"

    def free_vram(self) -> int:
        free, total = ctypes.c_size_t(0), ctypes.c_size_t(0)
        rc = self.lib.hipMemGetInfo(ctypes.byref(free), ctypes.byref(total))
        return int(free.value) if rc == 0 else 0

    def sync(self) -> None:
        self.check(self.lib.hipDeviceSynchronize(), "hipDeviceSynchronize")

    # -- THE MECHANISM (gated by R1) ------------------------------------------

    def host_alloc(self, nbytes: int, flags: int = HOST_ALLOC_FLAGS) -> Tuple[int, int]:
        """`hipHostMalloc(...Mapped)` -> `hipHostGetDevicePointer`. Returns `(host_ptr, device_ptr)`.

        The only mechanism Phase 0 found that puts REAL host pages behind an address a kernel can
        read: 28.93 GB/s on card 0, 14.48 GB/s on card 1 (P1), reproduced to three digits by P2'
        through a real MoE GEMV and to within 4-8% by P5b through torch.

        On this box `host_ptr == device_ptr` (ROCm's unified VA), and P5b recorded exactly that for
        the arm that passed every coherence test. **Do not bake the aliasing in** — always call
        `hipHostGetDevicePointer` and use what it returns.
        """
        _check_not_frozen("hipHostMalloc")
        hp = ctypes.c_void_p()
        self.check(
            self.lib.hipHostMalloc(ctypes.byref(hp), ctypes.c_size_t(int(nbytes)),
                                   ctypes.c_uint(int(flags))),
            f"hipHostMalloc({nbytes} B, flags=0x{flags:x})",
        )
        dp = ctypes.c_void_p()
        self.check(
            self.lib.hipHostGetDevicePointer(ctypes.byref(dp), hp, ctypes.c_uint(0)),
            "hipHostGetDevicePointer",
        )
        if not hp.value or not dp.value:
            raise HipError(
                "hipHostMalloc/hipHostGetDevicePointer returned hipSuccess with a NULL pointer — "
                "this box has repeatedly returned success over wrong state; never trust the rc"
            )
        _mlock_region(int(hp.value), int(nbytes))
        return int(hp.value), int(dp.value)

    def host_free(self, host_ptr: int) -> None:
        _check_not_frozen("hipHostFree")
        self.check(self.lib.hipHostFree(ctypes.c_void_p(int(host_ptr))), "hipHostFree")

    def dev_alloc(self, nbytes: int, *, allow_after_freeze: bool = False) -> int:
        """`allow_after_freeze` exists for exactly one caller: `ArenaMemPool._fallback`.

        That path runs inside torch's C-ABI allocator callback, where a raise is swallowed into a
        NULL return and a segfault with no diagnostic. It used to reach for
        `hipmem.teardown_window()` — i.e. it flipped the PROCESS-GLOBAL R1 latch, at arbitrary
        runtime, on an arbitrary thread, from inside a C callback. That is the one thing R1 exists to
        forbid, and it also opened the window for every other thread for the duration. An explicit
        per-call bypass keeps the latch untouched and puts the violation at the call site where a
        reader can see it; the fallback counts and shouts, and `assert_clean()` fails the merge gate
        on it.
        """
        if not allow_after_freeze:
            _check_not_frozen("hipMalloc")
        p = ctypes.c_void_p()
        self.check(self.lib.hipMalloc(ctypes.byref(p), ctypes.c_size_t(int(nbytes))),
                   f"hipMalloc({nbytes})")
        if not p.value:
            raise HipError("hipMalloc returned hipSuccess with a NULL pointer")
        return int(p.value)

    def dev_free(self, ptr: int) -> None:
        _check_not_frozen("hipFree")
        self.check(self.lib.hipFree(ctypes.c_void_p(int(ptr))), "hipFree")

    # -- data movement, used by the out-of-band self-test ---------------------

    def memcpy(self, dst: int, src: int, nbytes: int, kind: int) -> None:
        self.check(
            self.lib.hipMemcpy(ctypes.c_void_p(int(dst)), ctypes.c_void_p(int(src)),
                               ctypes.c_size_t(int(nbytes)), ctypes.c_int(int(kind))),
            "hipMemcpy",
        )

    def memset_d32(self, dptr: int, word: int, n_words: int) -> None:
        """DEVICE-issued fill through the device pointer.

        Device-issued on purpose: it is the only way to prove the GPU's view of the page, and the
        GPU's view is the one the kernels use. A CPU `memset` would validate the CPU mapping and say
        nothing about the page table the device walks — which is precisely what was wrong in P6.
        """
        w = int(word) & 0xFFFFFFFF
        signed = w - (1 << 32) if w >> 31 else w
        self.check(
            self.lib.hipMemsetD32(ctypes.c_void_p(int(dptr)), ctypes.c_int(signed),
                                  ctypes.c_size_t(int(n_words))),
            "hipMemsetD32",
        )

    def read_u32(self, dptr: int) -> int:
        """One 4-byte read back through the DEVICE pointer."""
        buf = ctypes.c_uint32(0)
        self.memcpy(ctypes.addressof(buf), int(dptr), 4, hipMemcpyDeviceToHost)
        return int(buf.value)


_HIP: Hip | None = None


def get_hip() -> Hip:
    global _HIP
    if _HIP is None:
        with _LOCK:
            if _HIP is None:
                _HIP = Hip()
    return _HIP
