"""ROCTx markers for marker-gated GPU profiling. Inert unless MINISGL_ROCTX=1.

Why this exists
---------------
Profiling a spec step is otherwise two separate problems:

* **Scope.** rocprofv3 over a whole serve captures model load and graph-capture warmup as well as
  decode — measured once at 919,939 dispatches over 35.8 s, of which the steady decode phase was a
  7 s tail that had to be recovered by post-filtering a 533 MB CSV on dispatch density. Wrapping the
  interesting steps in `roctxProfilerResume(0)` / `roctxProfilerPause(0)` and running rocprofv3 with
  `--selected-regions` collects only those steps.

* **Attribution.** propose and verify both replay from CUDA graphs, so the torch profiler sees one
  launch and no kernels, and a flat kernel ranking cannot say which phase a kernel belongs to.
  `roctxRangePush`/`Pop` around each phase, plus `--marker-trace`, lets every dispatch be assigned to
  the enclosing range by timestamp.

Implementation notes
--------------------
`torch.cuda.nvtx` maps to ROCTx on ROCm and would cover push/pop, but NOT ProfilerResume/Pause, which
is the half that controls scope — so bind the library directly and get both from one place.

Everything degrades to a no-op: unset env, missing library, or missing symbol. A profiling aid must
never be able to break a serve.
"""
from __future__ import annotations

import ctypes
import os
from typing import Optional

_ENABLED = os.environ.get("MINISGL_ROCTX") == "1"
_lib: Optional[ctypes.CDLL] = None

# The SDK roctx first (matches the rocprofiler-sdk that rocprofv3 drives); libroctx64 is the older
# name kept as a fallback for images that ship only that.
_CANDIDATES = (
    "librocprofiler-sdk-roctx.so.1",
    "librocprofiler-sdk-roctx.so",
    "libroctx64.so.4",
    "libroctx64.so",
)

if _ENABLED:
    for _name in _CANDIDATES:
        try:
            _lib = ctypes.CDLL(_name)
            break
        except OSError:
            continue
    if _lib is not None:
        try:
            _lib.roctxRangePushA.argtypes = [ctypes.c_char_p]
            _lib.roctxRangePushA.restype = ctypes.c_int
            _lib.roctxRangePop.restype = ctypes.c_int
            _lib.roctxProfilerResume.argtypes = [ctypes.c_uint64]
            _lib.roctxProfilerPause.argtypes = [ctypes.c_uint64]
        except AttributeError:
            _lib = None


def enabled() -> bool:
    return _lib is not None


def push(name: str) -> None:
    """Open a named range. Every push MUST be matched by a pop on every path, including exceptions —
    an unbalanced stack silently mis-attributes every later dispatch to the wrong phase."""
    if _lib is not None:
        _lib.roctxRangePushA(name.encode())


def pop() -> None:
    if _lib is not None:
        _lib.roctxRangePop()


def resume() -> None:
    """Begin collection (rocprofv3 --selected-regions). Argument 0 = the calling thread's process."""
    if _lib is not None:
        _lib.roctxProfilerResume(0)


def pause() -> None:
    if _lib is not None:
        _lib.roctxProfilerPause(0)


class range_:  # noqa: N801 - used as a context manager, reads as `with roctx.range_("propose")`
    """Context-managed range so the pop survives an exception in the profiled phase."""

    __slots__ = ("_name",)

    def __init__(self, name: str) -> None:
        self._name = name

    def __enter__(self) -> "range_":
        push(self._name)
        return self

    def __exit__(self, *exc) -> None:  # noqa: ANN002
        pop()
