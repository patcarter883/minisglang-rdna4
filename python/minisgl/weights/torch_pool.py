"""Hand the pinned host arena to torch as ordinary `device='cuda'` tensors.

This is the P5b-validated plumbing, and *only* that: `torch._C._cuda_customAllocator` +
`torch.cuda.MemPool` + `torch.cuda.use_mem_pool`, with a forward-only bump allocator over the
arena's chunks and a no-op free. P5b ran it in the serve image on both physical cards, under HIP
graph capture, over a real `hipHostMalloc(Mapped)` + `hipHostGetDevicePointer` arena, and measured:

* `t.data_ptr()` == the `hipHostGetDevicePointer` address exactly; 8/8 allocations served from the
  arena, **0 `hipMalloc` fallbacks** in all four legs;
* the arena survives `torch.cuda.empty_cache()` — pointer stable, contents intact, 6/6 reps
  (`engine/graph.py:314` calls it, so this mattered);
* the free callback fires **zero** times, for a live pool block AND for a cached/dropped block plus
  a subsequent `empty_cache()`;
* 24 graph replays → one distinct sha256, matching a CPU ground truth; 25/25 varying-input replays
  byte-exact, proving each replay genuinely re-reads the host pages.

So the ~30-line ctypes route works and the C++ `torch::from_blob` extension (+3 days) is not needed.

THREE THINGS THAT WILL SEGFAULT IF YOU CHANGE THEM.

1. **Nothing here may ever be garbage collected.** The two `CFUNCTYPE` trampolines, the python
   closures they wrap, the allocator object, the pool, and every tensor cut from the pool go into
   `_KEEP` forever. A collected trampoline is a dangling function pointer called from C++.
2. **A pool tensor must never outlive its pool.** Tearing down a `MemPool` with live blocks aborts
   with `c10::Error: invalid device pointer`. That is why `close()` on the arena is explicit and
   never runs from a destructor.
3. **The alloc callback must never raise and never return NULL.** An exception inside a ctypes
   callback is swallowed and becomes a NULL return, which torch dereferences. Every path is wrapped;
   the last resort is a real `hipMalloc`, counted loudly.

WHY A FALLBACK AT ALL, GIVEN IT IS WRONG. Returning NULL is a segfault with no diagnostic; falling
back is a *counted* correctness problem. `arena.torch_fallbacks != 0` after populate means bytes
that were budgeted as host-resident silently landed in VRAM, so the capacity plan is a fiction —
assert it is zero, do not merely log it.
"""

from __future__ import annotations

import ctypes
import sys
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

from .pinned_arena import PinnedWeightArena

# Module-level, permanent, never cleared. See note 1 above.
_KEEP: List[Any] = []

_ALLOC_T = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p)
_FREE_T = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p)


def _shout(msg: str) -> None:
    """stderr, unconditionally. The alloc callback runs under the C ABI where a raise becomes a NULL
    return and a segfault with no diagnostic; a returned NULL must at minimum leave a line behind
    naming the arena, or the crash is unattributable."""
    try:
        print(f"[weight-arena] {msg}", file=sys.stderr, flush=True)
    except BaseException:  # pragma: no cover - stderr closed during interpreter teardown
        pass


def _capturing() -> bool:
    """Is a HIP stream capture in progress on this thread's current stream?

    Guarded to the last comma: this runs inside the allocator callback, so it may not raise and may
    not import anything heavy. A torch too old to have `is_current_stream_capturing` answers False,
    which reproduces today's behaviour rather than blocking the pool.
    """
    try:
        import torch

        fn = getattr(torch.cuda, "is_current_stream_capturing", None)
        return bool(fn()) if fn is not None else False
    except BaseException:
        return False


class ArenaMemPool:
    """A `torch.cuda.MemPool` whose backing store is `arena`.

    One per arena. Constructing it does not allocate anything; allocations happen when torch asks,
    inside `use()`.
    """

    def __init__(self, arena: PinnedWeightArena, *, allow_hipmalloc_fallback: bool = True) -> None:
        import torch

        if not hasattr(torch._C, "_cuda_customAllocator"):
            raise RuntimeError(
                "this torch has no torch._C._cuda_customAllocator — the arena cannot be handed to "
                "torch without the C++ from_blob extension (plan §5.1, +3 days). P5b confirmed the "
                "symbol exists in the serve image; check you are running in it."
            )
        self.arena = arena
        self.allow_fallback = bool(allow_hipmalloc_fallback)
        self.alloc_events = 0
        self.free_events = 0
        self.served_from_arena = 0
        self.served_bytes = 0
        # Set if the alloc callback ever fires while a HIP stream capture is in progress. Recorded
        # rather than merely refused, because a capture-time arena allocation is a design error the
        # merge gate has to see even if the fallback happened to succeed.
        self.alloc_during_capture = 0
        self._last_error: Optional[str] = None

        alloc_cb, free_cb = self._install_callbacks()
        allocator = torch._C._cuda_customAllocator(
            ctypes.cast(alloc_cb, ctypes.c_void_p).value,
            ctypes.cast(free_cb, ctypes.c_void_p).value,
        )
        _KEEP.append(allocator)
        try:
            pool = torch.cuda.MemPool(allocator)
        except TypeError:  # older signature
            pool = torch.cuda.MemPool(allocator=allocator)
        _KEEP.append(pool)
        self.allocator = allocator
        self.pool = pool

    # -- the C ABI side -------------------------------------------------------

    def _install_callbacks(self):
        arena = self.arena

        def _alloc(size, device, stream):  # noqa: ANN001 - C ABI
            try:
                self.alloc_events += 1
                size = int(size)
                if _capturing():
                    # A raw hipMalloc from inside the allocator callback INVALIDATES an active
                    # stream capture (HIP, like CUDA, forbids cudaMalloc/hipMalloc during capture),
                    # and the abort names graph capture, not the arena. Serving from the arena
                    # during capture is fine pointer-wise but is still a design error: the bump
                    # allocator never frees, so every capture bucket would permanently consume
                    # arena headroom. Count it so assert_clean() fails loudly at the merge gate.
                    self.alloc_during_capture += 1
                # The arena is a set of hipHostMalloc mappings registered for ONE device. Serving
                # them to a different device index hands torch memory the asking device may not
                # address — silent corruption, not a slowdown. Fall back instead.
                if int(device) == arena.device_index:
                    r = arena.allocate_raw(size)
                    if r is not None:
                        self.served_from_arena += 1
                        self.served_bytes += r.nbytes
                        return r.device_ptr
                return self._fallback(size, reason=(
                    "wrong device" if int(device) != arena.device_index else "arena exhausted"))
            except BaseException as exc:  # a raise here would silently become a NULL return
                self._last_error = f"{type(exc).__name__}: {exc}"
                try:
                    return self._fallback(int(size), reason="callback error")
                except BaseException:
                    return 0

        def _free(ptr, size, device, stream):  # noqa: ANN001 - C ABI
            # Deliberate no-op: the arena lives for the whole process and hipHostFree is never
            # called from here. P5b measured this callback firing zero times anyway; the counter
            # exists so that a future torch that DOES call it is visible rather than surprising.
            try:
                self.free_events += 1
            except BaseException:
                pass

        a, f = _ALLOC_T(_alloc), _FREE_T(_free)
        _KEEP.extend([_alloc, _free, a, f])
        return a, f

    def _fallback(self, size: int, *, reason: str) -> int:
        self.arena.torch_fallbacks += 1
        if not self.allow_fallback:
            self._last_error = f"fallback disabled; returning NULL for {size} B after {reason}"
            _shout(self._last_error)
            return 0
        if _capturing():
            # Do NOT hipMalloc here. It would invalidate the in-flight capture, and the resulting
            # error is raised out of torch.cuda.graph.__exit__ pointing at graph capture with no
            # mention of the weight arena. NULL is also fatal, but it is fatal HERE, with a message.
            self._last_error = (
                f"alloc callback needed a {size} B fallback ({reason}) DURING an active HIP stream "
                f"capture. hipMalloc is illegal mid-capture, so no fallback is possible. The arena "
                f"MemPool must not be active during graph capture — wrap only weight "
                f"materialisation in use()."
            )
            _shout(self._last_error)
            return 0
        hip = self.arena._hip_or_bind()
        try:
            # Deliberately NOT wrapped in hipmem.teardown_window(). Opening the process-global R1
            # window from inside a C callback, on an arbitrary thread, at arbitrary runtime, is the
            # one thing R1 exists to forbid — and it opened the window for every other thread for
            # the duration. The per-call bypass keeps the latch untouched; the violation is counted
            # above, shouted below, and fails assert_clean().
            return hip.dev_alloc(size, allow_after_freeze=True)
        except Exception as exc:
            self._last_error = f"fallback hipMalloc({size}) after {reason}: {exc}"
            _shout(self._last_error)
            return 0

    # -- the torch side -------------------------------------------------------

    @contextmanager
    def use(self) -> Iterator[None]:
        """Route `torch.empty` through the arena — ON THE ARENA'S DEVICE, and never mid-capture.

        `torch.cuda.use_mem_pool(pool)` binds the pool to the CURRENT device, not to the device the
        allocations name. At TP=2 rank 1's arena is `cuda:1`, and if the current device is still
        `cuda:0` when `use()` is entered, the pool is installed on device 0 while every
        `torch.empty(device="cuda:1")` inside it is served by the ORDINARY caching allocator. The
        result is not an error and not a fallback: the alloc callback is never invoked at all, so
        `alloc_events == 0`, `torch_fallbacks == 0`, `assert_clean()` passes — and 34 GiB of weights
        that the capacity plan booked against host RAM are sitting in 16 GiB of VRAM. It presents
        much later as an unrelated OOM on rank 1 only. Bind the device explicitly, and assert the
        current device agrees so a torch without the `device=` kwarg still cannot get this wrong.

        Capture guard: the arena's bump allocator never frees, so an allocation served during graph
        capture permanently consumes headroom, and the fallback path cannot run mid-capture at all
        (hipMalloc is illegal during capture). Only weight materialisation belongs inside `use()`.
        """
        import torch

        dev = self.arena.device_index
        cur = torch.cuda.current_device()
        if int(cur) != int(dev):
            raise RuntimeError(
                f"WEIGHT OFFLOAD: ArenaMemPool.use() entered with current device cuda:{cur} but the "
                f"arena is pinned for cuda:{dev}. torch.cuda.use_mem_pool binds the pool to the "
                f"CURRENT device, so the pool would be installed on the wrong one and every "
                f"allocation would silently bypass the arena into VRAM with zero fallbacks counted. "
                f"Call torch.cuda.set_device({dev}) first."
            )
        if _capturing():
            raise RuntimeError(
                "WEIGHT OFFLOAD: ArenaMemPool.use() entered during an active HIP stream capture. "
                "The arena is a forward-only bump allocator with a no-op free, so capture-time "
                "allocations are never reclaimed, and its hipMalloc fallback would invalidate the "
                "capture outright. Wrap only weight materialisation in use()."
            )
        try:
            ctx = torch.cuda.use_mem_pool(self.pool, device=dev)
        except TypeError:  # older torch: no device kwarg — the assert above is the guard
            ctx = torch.cuda.use_mem_pool(self.pool)
        with ctx:
            yield

    def empty(self, *shape: int, dtype: Any = None) -> Any:
        """`torch.empty` served from the arena, anchored so it can never outlive the pool.

        The `data_ptr()` check is P5b's own assertion, promoted from a probe to an invariant. It is
        the ONLY residency test that is trustworthy on this box: `hipPointerGetAttributes` reported
        "Host" for VRAM in Phase 0 and "Device" for the real host arena in P5b, so it is wrong in
        both directions, while "is this address inside a range `hipHostGetDevicePointer` returned"
        is pure arithmetic. Without it, every way the pool can quietly stop routing (wrong device,
        a torch upgrade changing `MemPool` semantics, a `use_mem_pool` that silently no-ops) is
        invisible — `assert_clean()` sees zero fallbacks and passes.
        """
        import torch

        with self.use():
            t = torch.empty(*shape, dtype=dtype, device=f"cuda:{self.arena.device_index}")
        if t.numel() and not self.arena.owns_pointer(t.data_ptr(), t.numel() * t.element_size()):
            raise RuntimeError(
                f"WEIGHT OFFLOAD: torch.empty inside the arena MemPool returned "
                f"0x{t.data_ptr():x}, which is NOT inside any pinned chunk. The pool is not "
                f"routing: the allocation went to VRAM while the capacity plan booked it against "
                f"host RAM. stats={self.stats()}"
            )
        _KEEP.append(t)
        return t

    def stats(self) -> Dict[str, Any]:
        return {
            "alloc_callbacks": self.alloc_events,
            "served_from_arena": self.served_from_arena,
            "served_bytes": self.served_bytes,
            "free_callbacks": self.free_events,
            "hipmalloc_fallbacks": self.arena.torch_fallbacks,
            "alloc_during_capture": self.alloc_during_capture,
            "last_error": self._last_error,
        }

    def assert_clean(self, *, expect_served_bytes: int = 0) -> None:
        """Merge gate: every torch allocation came from the arena — and some actually did.

        A non-zero fallback count means host-budgeted bytes went to VRAM — the exact failure the
        capacity plan cannot see, and one that presents later as an OOM with an unrelated message.

        The zero checks matter just as much, and the original gate could not see them. Every way the
        pool stops routing entirely (installed on the wrong device, a torch that changes `MemPool`
        semantics, a caller that forgot `use()`) produces `alloc_events == 0` and
        `torch_fallbacks == 0` — a perfectly clean-looking ledger over a serve that put the whole
        host tier in VRAM. `expect_served_bytes` lets the caller assert against the plan's
        host-resident total; with it left 0 the gate still refuses the "nothing was ever served"
        shape whenever the arena was pinned at all.
        """
        problems = []
        if self.arena.torch_fallbacks or self._last_error:
            problems.append(
                "allocations escaped the arena (fallback to hipMalloc): bytes budgeted as "
                "host-resident have landed in VRAM, so the capacity plan is no longer valid. "
                "Raise the reserved headroom (`reserve(..., extra_bytes=...)`)."
            )
        if self.alloc_during_capture:
            problems.append(
                f"{self.alloc_during_capture} allocation(s) hit the arena callback DURING HIP "
                "graph capture. The bump allocator never frees, so capture-time allocations leak "
                "arena headroom permanently; only weight materialisation may run inside use()."
            )
        if self.arena.pinned_bytes and self.alloc_events == 0:
            problems.append(
                f"the arena has {self.arena.pinned_bytes} B pinned but the alloc callback NEVER "
                "fired. The MemPool is not routing at all — the allocations went to VRAM through "
                "the ordinary caching allocator and nothing counted them. Check that use() ran on "
                f"cuda:{self.arena.device_index} and that this torch honours use_mem_pool."
            )
        if expect_served_bytes and self.served_bytes < expect_served_bytes:
            problems.append(
                f"only {self.served_bytes} B were served from the arena but the plan expected at "
                f"least {expect_served_bytes} B host-resident."
            )
        if problems:
            raise RuntimeError(
                "WEIGHT OFFLOAD: torch MemPool did not stay inside the arena.\n"
                + "\n".join(f"  - {p}" for p in problems)
                + f"\n  stats={self.stats()}"
            )
