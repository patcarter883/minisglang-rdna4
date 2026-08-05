"""TP comms/compute overlap — one primitive, usable by any model, safe under graph capture.

WHAT THIS REPLACES
------------------
`Qwen3_5MoeSparseBlock._forward_async_ar` was the only comms/compute overlap in the repo: it split the
fused MoE partial into 2 disjoint row chunks and hid chunk-0's all_reduce on a side stream behind
chunk-1's expert GEMM. It was correct, and it was stuck — Qwen3.5-MoE only, one hard-coded shape, and
structurally incompatible with the cudagraph capture that has since landed on the canvas step.

The generalization is NOT "the same trick, copy-pasted per model". It is to notice that the trick has
exactly one primitive in it -- **an all_reduce whose completion is deferred to a later point in the
program** -- and to expose that primitive, so that "what independent compute do we hide it behind" is a
decision the CALLER makes from its own dataflow, not something baked into the mechanism.

    handle = async_all_reduce(comm, partial)   # issued now, on the side stream
    ...                                        # anything independent of `partial` -- it overlaps
    reduced = handle.wait()                    # main stream is safe to read it here

Two application patterns fall out, and they compose:

  BRANCH overlap  -- a layer with two independent branches (Gemma4: dense MLP and MoE, summed only
                     after separate norms) reduces one branch while it computes the other. NO row
                     split at all, so it is bit-exact in the strongest possible sense: the identical
                     collective runs on the identical tensor and only its STREAM differs.
  ROW-CHUNK overlap -- a row-independent span reduces chunk i while computing chunk i+1. This is the
                     original Qwen3.5-MoE trick, and it is LOSSY on this box. See below.

`rowchunked_ar_span` below is the second pattern expressed on the first primitive, and Qwen3.5-MoE now
calls it instead of carrying its own copy.

THE BIT-EXACTNESS RESULT (measured; tools/tp_overlap_bitexact.py)
-----------------------------------------------------------------
The inherited justification for row chunking was: "disjoint rows => BIT-EXACT, because each row's
all_reduce is an independent 2-rank elementwise SUM". That argument is sound, and it is about the
COLLECTIVE. It says nothing about the PRODUCER, and the producer is where it fails.

Splitting rows changes M for every GEMM inside the span, and a GEMM's tiling/algorithm selection is
M-dependent. Measured, on a bare `torch.mm` -- no MoE, no atomics, just rocBLAS:

    rows=3200, split into 2   ->  max|delta| = 1.562e-02   (producer alone, collective not involved)
    rows=2048, split into 2   ->  max|delta| = 0            (this shape happens to pick the same kernel)
    rows=256,  split into 2   ->  max|delta| = 7.812e-03

So the row split is **not lossless**, it merely *looked* lossless at whichever shape was first tried.
`MINISGL_TP_AR_CHUNKS` therefore defaults to **1** (no split): overlap comes from branch independence,
which is exact. Row chunking remains available for callers with no independent branch -- Qwen3.5-MoE is
exactly that case -- but it is opt-in and it is a lossy/perf trade, not a free one.

The no-split path measures max|delta| = 0.000e+00 at every shape tested, including two collectives
outstanding at once.

THE CAPTURE PROBLEM, AND WHY THERE IS NO GATE IN THE MODELS
-----------------------------------------------------------
Side-stream collectives cannot be captured into a CUDA graph. The canvas step IS captured. Those two
facts do not compose, and the honest resolution is not to make the models choose.

Instead the primitive is **capture-transparent**: when the current stream is capturing (or overlap is
disabled, or the tensor is too small to be worth a side stream), `async_all_reduce` performs the
ordinary in-place all_reduce on the CURRENT stream and returns an already-satisfied handle whose
`wait()` is a no-op. The captured graph therefore contains exactly the collectives it contained before
this file existed, in exactly the same order, and the eager path overlaps them.

So the model source has ONE form. There is no `if is_current_stream_capturing()` in any model, no
second code path to keep in sync, and no way for a future capture to silently lose its collectives.
Overlap is a stream-scheduling decision made inside the primitive, invisible above it. Captured decode
gets no overlap -- that is a real limit, stated plainly -- but chunked prefill, which is always eager
(`GraphRunner.can_use_cuda_graph` requires `batch.is_decode`), gets all of it.

THE ORDERING INVARIANT (this is what a bug here would look like)
----------------------------------------------------------------
A collective library matches calls by ENQUEUE order per communicator, and both TP ranks run the same
Python, so they enqueue in the same order by construction. The hazard is not order, it is CONCURRENCY:
two all_reduces of the same communicator in flight at once. For RCCL that can deadlock; for custom_ar
it is worse than that, because `one_shot_ar` handshakes through ONE shared flag buffer and two
concurrent calls would corrupt each other's flags and silently return half-reduced data.

The invariant that prevents it: **every outstanding handle must be waited before the next collective is
issued on the main stream.** Async collectives all share ONE side stream, so they serialize against each
other; `wait()` makes the main stream block on the side stream, so anything the main stream issues
afterwards is ordered after them. `ar_span()` enforces this for its own handles by construction --
it cannot return with one outstanding -- and is the recommended way to use this module.

Env:
  MINISGL_TP_OVERLAP=0            disable entirely (every async_all_reduce becomes a plain one)
  MINISGL_TP_OVERLAP_MIN_TOKENS   rows below which overlap is not worth the doubled launch (default 256)
  MINISGL_TP_AR_CHUNKS            row chunks for rowchunked_ar_span (default 2)
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Callable, List

import torch

if TYPE_CHECKING:
    from minisgl.distributed import DistributedCommunicator

__all__ = [
    "AsyncAllReduce",
    "ar_span",
    "async_all_reduce",
    "rowchunked_ar_span",
    "tp_overlap_chunks",
    "tp_overlap_enabled",
    "tp_overlap_min_tokens",
]

def _env_int(name: str, default: int) -> int:
    """An UNSET var and a var set to the empty string mean the same thing: take the default. Harnesses
    that forward `-e VAR=$VAR` propagate empty strings for unset vars, and `int("")` would take down
    the whole engine at import time for a knob nobody set."""
    v = os.environ.get(name, "").strip()
    return int(v) if v else default


_ENABLED = os.environ.get("MINISGL_TP_OVERLAP", "").strip() != "0"
# Below this many rows the collective is latency-dominated (measured: a [1, 2816] bf16 all_reduce is
# 13.7 us and a [256, 2816] one is 139.8 us, against a per-call side-stream cost of ~2 events plus a
# cross-stream wait), so the doubled launch outweighs anything it could hide.
_MIN_TOKENS = _env_int("MINISGL_TP_OVERLAP_MIN_TOKENS", 256)
# DEFAULT 1 = no row split. See "THE BIT-EXACTNESS RESULT" above: splitting rows is NOT lossless on
# this box, because the producer's GEMM is M-dependent. Row chunking is opt-in and lossy.
_CHUNKS = max(1, _env_int("MINISGL_TP_AR_CHUNKS", 1))
_side_stream: "torch.cuda.Stream | None" = None
_announced = False


def tp_overlap_enabled() -> bool:
    """Whether comms/compute overlap is engaged at all (MINISGL_TP_OVERLAP, default ON)."""
    return _ENABLED


def tp_overlap_min_tokens() -> int:
    """Row count below which a collective is not worth putting on a side stream."""
    return _MIN_TOKENS


def tp_overlap_chunks() -> int:
    """Row chunks used by rowchunked_ar_span (MINISGL_TP_AR_CHUNKS, default 2)."""
    return _CHUNKS


def get_ar_side_stream() -> "torch.cuda.Stream":
    """The one process-wide side stream carrying every overlapped collective.

    ONE stream, deliberately. Multiple side streams would let two collectives of the same communicator
    run concurrently, which is exactly the failure this module's ordering invariant exists to prevent.
    With a single stream they serialize against each other for free, and `wait()` orders the main
    stream after them."""
    global _side_stream
    if _side_stream is None:
        _side_stream = torch.cuda.Stream()
    return _side_stream


class AsyncAllReduce:
    """A handle for an all_reduce that has been ISSUED but may not have COMPLETED.

    `wait()` returns the reduced tensor and is idempotent. Until it is called, the tensor must not be
    read on the main stream -- reading it early is the one way to get a half-reduced value out of this
    module, and it is why `ar_span` exists to make forgetting impossible."""

    __slots__ = ("_tensor", "_event", "_done")

    def __init__(self, tensor: torch.Tensor, event: "torch.cuda.Event | None") -> None:
        self._tensor = tensor
        self._event = event
        self._done = event is None

    def wait(self) -> torch.Tensor:
        if not self._done:
            torch.cuda.current_stream().wait_event(self._event)
            self._done = True
        return self._tensor

    @property
    def tensor(self) -> torch.Tensor:
        """The underlying storage WITHOUT waiting. For plumbing only (shape/dtype); never read it."""
        return self._tensor


def overlap_active(x: torch.Tensor) -> bool:
    """Whether an overlapped collective on a tensor shaped like `x` would actually run on a side
    stream. Callers use it to decide whether a ROW SPLIT is worth doing at all: under capture, below
    the threshold, or with overlap off, every collective is inline, so splitting rows buys nothing and
    costs extra launches (and, for MoE, a different grouped-GEMM M). Same predicate the primitive uses,
    so the caller's decision and the primitive's cannot disagree."""
    return _overlappable(x)


def _overlappable(x: torch.Tensor) -> bool:
    """Whether this collective should go to the side stream rather than the current one.

    Capturing is the hard exclusion: a side-stream collective cannot be recorded into a CUDA graph, so
    under capture we issue the ordinary in-place collective and the graph is exactly what it always
    was."""
    return (
        _ENABLED
        and x.is_cuda
        and x.shape[0] >= _MIN_TOKENS
        and not torch.cuda.is_current_stream_capturing()
    )


def async_all_reduce(comm: "DistributedCommunicator", x: torch.Tensor) -> AsyncAllReduce:
    """Issue an in-place all_reduce of `x`, overlapped with whatever the caller does next.

    BIT-EXACT, unconditionally: this changes WHICH STREAM the collective runs on and nothing else. The
    same communicator performs the same reduction over the same two rank-local addends of the same
    tensor, so every output element is produced by the identical arithmetic. There is no reassociation
    to argue about -- at TP=2 an all_reduce is a 2-addend elementwise sum, which has no ordering.

    Returns a handle; the value is only safe to read after `handle.wait()`."""
    if not _overlappable(x):
        return AsyncAllReduce(comm.all_reduce(x), None)

    global _announced
    if not _announced:
        # PROVENANCE. An A/B that only sets MINISGL_TP_OVERLAP proves the env var was set, not that any
        # collective moved. This fires from the first one that actually does, so a log grep can tell
        # the two arms apart for real.
        _announced = True
        from minisgl.utils import init_logger

        init_logger(__name__).info_rank0(
            f"TP comms/compute overlap ENGAGED (side stream; first tensor {tuple(x.shape)}, "
            f"min_tokens={_MIN_TOKENS}, row chunks={_CHUNKS})"
        )

    main = torch.cuda.current_stream()
    side = get_ar_side_stream()
    ready = torch.cuda.Event()
    ready.record(main)      # the producer of `x` has been submitted on main
    side.wait_event(ready)  # ...and the collective must not start before it lands
    with torch.cuda.stream(side):
        # Tell the caching allocator the side stream also uses this block, so it cannot be handed to
        # another main-stream tensor while the collective is still reading it.
        x.record_stream(side)
        comm.all_reduce(x)
        done = torch.cuda.Event()
        done.record(side)
    return AsyncAllReduce(x, done)


class ar_span:
    """A scope in which async all_reduces may be outstanding, guaranteeing none escapes it.

    The ordering invariant this module depends on -- no collective issued on the main stream while
    another is in flight -- is easy to state and easy to violate by an early return or an exception. As
    a context manager it holds structurally:

        with ar_span(comm) as span:
            a = span.all_reduce(mlp_partial)     # issued on the side stream
            moe_partial = experts(...)           # independent -> overlaps `a`
            b = span.all_reduce(moe_partial)
            dense, moe = norm1(a.wait()), norm2(b.wait())
        # every handle is waited here, whether or not the body did it and even if it raised

    Waiting twice is free (`AsyncAllReduce.wait` is idempotent), so the body should wait at the natural
    consumption point -- the exit wait is a backstop, not the intended synchronization."""

    __slots__ = ("_comm", "_handles")

    def __init__(self, comm: "DistributedCommunicator") -> None:
        self._comm = comm
        self._handles: List[AsyncAllReduce] = []

    def all_reduce(self, x: torch.Tensor) -> AsyncAllReduce:
        h = async_all_reduce(self._comm, x)
        self._handles.append(h)
        return h

    def __enter__(self) -> "ar_span":
        return self

    def __exit__(self, *exc) -> None:
        for h in self._handles:
            h.wait()
        self._handles.clear()


def rowchunked_ar_span(
    comm: "DistributedCommunicator",
    x: torch.Tensor,
    produce: Callable[[torch.Tensor], torch.Tensor],
    *,
    num_chunks: int | None = None,
) -> torch.Tensor:
    """Run a ROW-INDEPENDENT producer over disjoint row chunks, reducing chunk i while computing i+1.

    `produce(rows) -> partial` must be row-independent (every output row a function of the same input
    row only) and must return the UNREDUCED rank-local partial -- typically by passing `reduce=False`
    to a row-parallel projection or MoE layer.

    NOT BIT-EXACT, and default-off (MINISGL_TP_AR_CHUNKS=1) for that reason. The COLLECTIVE half is
    exact -- disjoint rows make each row's all_reduce an independent 2-rank elementwise SUM. The
    PRODUCER half is not: fewer rows means a different M, and GEMM kernel selection is M-dependent, so
    `produce(x[:n//2])` and the first half of `produce(x)` can differ in the last bits. Measured at up
    to 1.6e-2 on bf16 for a plain `torch.mm`; see the module docstring and tools/tp_overlap_bitexact.py.
    Use this only where there is no independent branch to overlap against and the trade is worth it.

    Both TP ranks derive the split from `x.shape[0]`, which they hold identically (it is
    post-attention-all_reduce), so they chunk the same way and submit the same collectives in the same
    order -- the split is deterministic even though it is not lossless.

    Falls back to the plain `produce(x)` + one all_reduce when overlap is off, under capture, below the
    token threshold, or at num_chunks == 1 -- so callers need no gate of their own."""
    n = x.shape[0]
    k = tp_overlap_chunks() if num_chunks is None else num_chunks
    if k <= 1 or not _overlappable(x) or n < k:
        return comm.all_reduce(produce(x))

    bounds = [(n * i) // k for i in range(k + 1)]
    parts: List[AsyncAllReduce] = []
    with ar_span(comm) as span:
        for i in range(k):
            lo, hi = bounds[i], bounds[i + 1]
            # produce() runs on the MAIN stream; the PREVIOUS chunk's collective is on the side stream
            # and overlaps it. The last chunk's collective has nothing after it and stays exposed --
            # that is the (k-1)/k ceiling on what this can hide.
            parts.append(span.all_reduce(produce(x[lo:hi])))
    return torch.cat([p.wait() for p in parts], dim=0)
