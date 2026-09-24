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
                     original Qwen3.5-MoE trick. It IS bit-exact for every producer the engine puts
                     inside it, provided the chunks stay above the producer's kernel crossovers --
                     which `rowchunked_ar_span` now enforces structurally. See below.

`rowchunked_ar_span` below is the second pattern expressed on the first primitive, and Qwen3.5-MoE now
calls it instead of carrying its own copy.

THE BIT-EXACTNESS RESULT (measured; tools/quant_m_invariance.py, 2026-08-05, RX 9070 XT)
----------------------------------------------------------------------------------------
CORRECTION (this section previously said the opposite; the evidence for that was invalid).

The inherited justification for row chunking was: "disjoint rows => BIT-EXACT, because each row's
all_reduce is an independent 2-rank elementwise SUM". That argument is sound and it is about the
COLLECTIVE; it says nothing about the PRODUCER. Splitting rows changes M for every GEMM inside the
span, and kernel selection IS M-dependent -- so the question is real. It was then answered with the
WRONG INSTRUMENT: a bare `torch.mm` (rows=3200 split 2 -> max|delta| = 1.562e-02), and the default
was set to 1 on that basis.

No engine path puts a `torch.mm`/`F.linear` inside a row-split span:
  * the UNQUANTIZED producers route through `minv_linear` (layers/minv.py), which exists precisely
    because rocBLAS picks its kernel by shape. It is M-invariant BY CONSTRUCTION (full K-reduction
    per output tile, fixed order, no split-K), and its three `dense_gemm` arms (rd/pipe/lds) are
    bit-identical to each other at 0.000e+00 across every crossover, M=129/192 included.
  * the QUANTIZED producers route through quant/kernels.py -- measured now, for the first time:

    DENSE W4A8, 7 shipped shapes (Gemma4 o_proj 2048/4096/8192 x 2816, qkv, gate_up, dense_down at
    g=32 fp16; Qwen3.6-35B o_proj/gate_up at g=128 bf16), rows 256/512/1024/2048, split into 2:
        max|delta| = 0.000e+00  at every one.
    MoE W4A8 (E=128, top_k=8, hidden 2816, inter 352 -- the Qwen3.5-MoE-shaped producer this span
    actually wraps), rows 256/512/1024/2048, split into 2:
        max|delta| = 0.000e+00  at every one, INCLUDING rows=2048, where the split changes the
        grouped tile `_moe_block_m` from 128 to 64. The workload-derived tile is bit-NEUTRAL; it had
        been assumed lossy and it is not.

WHAT THE ACTUAL CONSTRAINT IS. Both producers are M-dependent -- just not at these row counts. Every
individual kernel arm is M-invariant on its own (verified: rows[0:m] computed alone == the same rows
inside a batch of 2048, max|delta| = 0, for all three dense arms). The ONLY way a token's value
changes is if the row count moves it onto a DIFFERENT ARM:

    dense: `prefill_wmma` and `wmma_tiled_tuned` are BIT-IDENTICAL to each other (0.000e+00 at every
           M on every shape), so the M=64 crossover is free. The one lossy dense crossover is
           decode_gemv <-> WMMA at M <= _W4A8_GEMV_MAX_INT4 (measured up to 1.953e-3 abs, fp16).
    MoE:   the lossy crossovers are gemm1 gemv <-> wmma at M <= _MOE_GEMM1_GEMV_MAX (32, measured up
           to 4.9e-4 abs) and the gemm2 atomic scatter at M <= 2 (which is additionally
           non-deterministic run to run, by design and already documented in quant/kernels.py).

This span only engages at `n >= _MIN_TOKENS` (256), so at the shipped `k=2` every chunk is >= 128
rows -- above every crossover, which is exactly why the measurement is 0. The guard that was needed
is therefore not "never split", it is "never split so finely that a chunk lands below 33 rows", and
that is now enforced in code (`_MIN_CHUNK_ROWS`) instead of by a blanket default of 1. Default
restored to 2.

The no-split path measures max|delta| = 0.000e+00 at every shape tested, including two collectives
outstanding at once.

THE CAPTURE PROBLEM, AND WHY THERE IS NO GATE IN THE MODELS
-----------------------------------------------------------
Side-stream collectives DO capture into a CUDA graph (event fork/join), but on ROCm 7.2 the captured
canvas step measured 13% SLOWER with them (see `_overlappable`). The canvas step IS captured, so the
honest resolution is not to make the models choose.

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
  MINISGL_TP_AR_CHUNKS            row chunks for rowchunked_ar_span (default 2; clamped so no chunk
                                  falls below _MIN_CHUNK_ROWS, which is what keeps it bit-exact)
"""

from __future__ import annotations

import contextlib
import os
from typing import TYPE_CHECKING, Callable, List

import torch

if TYPE_CHECKING:
    from minisgl.distributed import DistributedCommunicator

__all__ = [
    "AsyncAllReduce",
    "ar_span",
    "async_all_reduce",
    "inline_collectives",
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
# DEFAULT 2 = the original two-chunk split. See "THE BIT-EXACTNESS RESULT" above: the row split is
# bit-exact for both real producers (dense W4A8 and the grouped MoE) at every row count this span
# engages at. It was defaulted to 1 on a `torch.mm` measurement of a kernel the engine never calls.
_CHUNKS = max(1, _env_int("MINISGL_TP_AR_CHUNKS", 2))
# Rows below which a chunk changes the producer's KERNEL ARM rather than just its grid, which is the
# one thing that makes a row split lossy. Set by the highest measured crossover of any producer that
# can sit inside this span: the grouped MoE gemm1 swaps its scalar GEMV for WMMA at
# `quant/kernels.py::_MOE_GEMM1_GEMV_MAX` = 32 rows (the dense crossovers are lower: 8/16 for
# decode_gemv, and prefill_wmma == wmma_tiled_tuned bit-for-bit so the 64 crossover is free).
# The chunk count is clamped against this, so an operator raising MINISGL_TP_AR_CHUNKS cannot
# silently turn an exact split into a lossy one -- it just gets fewer chunks.
_MIN_CHUNK_ROWS = 33
_side_stream: "torch.cuda.Stream | None" = None
_announced = False
# Set only by `inline_collectives()` below -- a measurement scope, not an operator knob.
_FORCE_INLINE = False


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

    Capturing is the hard exclusion, and it is a MEASURED one, not a capability limit: the event
    fork/join below does capture (2026-09-25, DiffusionGemma canvas, TP=2, the graph-vs-eager gate
    stayed 0.000e+00), but the captured step got SLOWER, 87.8 -> 99.3 ms (branch overlap) / 99.9 ms
    (row chunks) on ROCm 7.2 — the spinning one-shot collective competes with the grouped GEMM for
    CUs, and the branched graph loses more than the hidden collective saves. So under capture we
    issue the ordinary in-place collective and the graph is exactly what it always was."""
    return (
        _ENABLED
        and not _FORCE_INLINE
        and x.is_cuda
        and x.shape[0] >= _MIN_TOKENS
        and not torch.cuda.is_current_stream_capturing()
    )


@contextlib.contextmanager
def inline_collectives():
    """Run the enclosed forward in the collective regime a CAPTURED GRAPH contains.

    `_overlappable` excludes capture, so a captured region holds inline collectives and no row split
    while the eager path holds side-stream collectives and (at chunks>1) a row split. That is by
    design, but it means "graph vs eager" is TWO differences at once: the graph mechanism, and a
    different program. A bit-exactness gate that does not hold the second one fixed cannot say which
    it caught — which is exactly how `[canvas-graph]` came to report a 1.575e+01 delta with nothing
    to attribute it to.

    This is a MEASUREMENT scope, not a knob: nothing in the serve loop enters it, and it takes no
    env var. Its one caller is `GraphRunner.replay_canvas`'s reference forward."""
    global _FORCE_INLINE
    prev = _FORCE_INLINE
    _FORCE_INLINE = True
    try:
        yield
    finally:
        _FORCE_INLINE = prev


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
    produce: Callable[..., torch.Tensor],
    *,
    num_chunks: int | None = None,
    row_aligned: "tuple[torch.Tensor | None, ...] | None" = None,
) -> torch.Tensor:
    """Run a ROW-INDEPENDENT producer over disjoint row chunks, reducing chunk i while computing i+1.

    `produce(rows) -> partial` must be row-independent (every output row a function of the same input
    row only) and must return the UNREDUCED rank-local partial -- typically by passing `reduce=False`
    to a row-parallel projection or MoE layer.

    BIT-EXACT as long as every chunk stays >= `_MIN_CHUNK_ROWS`, which this function ENFORCES by
    clamping `k`. The COLLECTIVE half is exact unconditionally -- disjoint rows make each row's
    all_reduce an independent 2-rank elementwise SUM. The PRODUCER half is exact because each
    quantized/minv kernel ARM is itself M-invariant, and a chunk that stays above the crossovers picks
    the same arm as the whole. Measured 0.000e+00 for the dense W4A8 producer and for the grouped MoE
    producer at rows 256/512/1024/2048 split into 2 -- including the case where the split changes
    `_moe_block_m` 128 -> 64. See the module docstring and tools/quant_m_invariance.py.

    (This docstring previously said "NOT BIT-EXACT ... measured at up to 1.6e-2 on bf16 for a plain
    `torch.mm`". That measurement was of rocBLAS, which no producer inside this span uses -- the
    unquantized ones go through `minv_linear` and the quantized ones through quant/kernels.py.)

    Both TP ranks derive the split from `x.shape[0]`, which they hold identically (it is
    post-attention-all_reduce), so they chunk the same way and submit the same collectives in the same
    order.

    `row_aligned` is a tuple of tensors whose dim-0 is THE SAME ROW AXIS as `x` -- e.g. the producer-
    side act-quant pair (x_fp8 (n,K), act_scales (n,)) -- and which must therefore be sliced by the
    SAME bounds and handed to `produce` alongside the row chunk: `produce(x_chunk, *aligned_chunks)`.
    A `None` entry passes through as `None` (unsliced), so a caller with no pair needs no branch.
    This exists because slicing `x` while passing the pair whole would scale every token by another
    token's amax -- wrong numbers, right shapes, no error. Making the alignment the primitive's job
    is what keeps that impossible to get wrong at a call site.

    Falls back to the plain `produce(x)` + one all_reduce when overlap is off, under capture, below the
    token threshold, or at num_chunks == 1 -- so callers need no gate of their own."""
    aligned = tuple(row_aligned or ())
    for t in aligned:
        if t is not None and t.shape[0] != x.shape[0]:
            raise AssertionError(
                f"rowchunked_ar_span: row_aligned entry has {t.shape[0]} rows, x has {x.shape[0]}"
            )
    n = x.shape[0]
    k = tp_overlap_chunks() if num_chunks is None else num_chunks
    # Clamp so no chunk drops below the producer's kernel-arm crossover. A chunk that crosses would
    # make the split lossy, and it is not the caller's job to know where the crossovers are.
    k = min(k, max(1, n // _MIN_CHUNK_ROWS))
    if k <= 1 or not _overlappable(x) or n < k:
        return comm.all_reduce(produce(x, *aligned))

    bounds = [(n * i) // k for i in range(k + 1)]
    parts: List[AsyncAllReduce] = []
    with ar_span(comm) as span:
        for i in range(k):
            lo, hi = bounds[i], bounds[i + 1]
            # produce() runs on the MAIN stream; the PREVIOUS chunk's collective is on the side stream
            # and overlaps it. The last chunk's collective has nothing after it and stays exposed --
            # that is the (k-1)/k ceiling on what this can hide.
            sl = tuple(None if t is None else t[lo:hi] for t in aligned)
            parts.append(span.all_reduce(produce(x[lo:hi], *sl)))
    return torch.cat([p.wait() for p in parts], dim=0)
