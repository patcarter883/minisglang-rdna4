"""M-invariant dense GEMM — the engine's standing rule for bf16/fp16 linears.

WHY THIS EXISTS
    torch's F.linear dispatches to rocBLAS, whose kernel / tile / split-K reduction ORDER is chosen by
    the problem shape (M,N,K). Float addition is non-associative, so linear(x[:m],W) is NOT bit-identical
    to linear(x[:M],W)[:m] when m != M: the same token's output changes by ~1 bf16 ULP depending on how
    many tokens share the batch. This silently corrupts every pathway where a token is computed at
    different M and expected to match:
      * prefix / radix caching  (cached continuation at M=new_len vs fresh at M=full_len)
      * chunked prefill         (a long prompt split by max_extend runs each chunk at a different M)
      * spec-decode VERIFY      (K+1 tokens/seq at variable M vs sequential decode at M=1)
    A coarse int8/fp8 activation downcast downstream amplifies the ULP seed into flipped greedy tokens
    (observed: ZAYA GSM8K 45->25 with CCA prefix caching on).

THE FIX
    Route bf16/fp16 dense linears through our OWN fixed-tile WMMA GEMM (the moe_bf16_wmma spine driven
    dense: single expert, identity routing). It does the FULL K-reduction per output tile in a fixed
    order with NO split-K, so the reduction order does not depend on M -> M-invariant BY CONSTRUCTION,
    and WE own the tile so it cannot drift under a rocBLAS/hipBLASLt upgrade. Verified bit-exact across
    M for the ZAYA projection shapes; parity vs F.linear is ~1 bf16 ULP (a different but fixed order).

SCOPE ("as close to M-invariant as realistically possible")
    Applied for eager bf16/fp16 GEMMs whose K is WMMA-friendly (IN % 16 == 0). Falls back to F.linear
    (with a one-time warning) for: fp32/other dtypes, IN not a multiple of 16, or under cudagraph
    capture (static shapes are already self-consistent, and the arange/route tensors would allocate
    mid-capture). Integer (int8) matmuls are already exact/M-invariant; quantized-expert and attention
    kernels are already fixed-tile HIP.
"""
from __future__ import annotations

import os

import torch

_BLOCK_M = int(os.environ.get("MINISGL_MINV_BLOCK_M", "64"))  # WMMA M-tile (mult of 16, <=128)
_BN = int(os.environ.get("MINISGL_MINV_BN", "64"))            # WMMA N-tile (divides most OUT dims)
# ---------------------------------------------------------------------------------------------
# WHICH KERNEL, AND WHY. All three dense_gemm kernels are bit-identical (identical fixed 16-wide K
# order, no split-K), so this dispatch may pick purely on speed and stays M-invariant whatever it
# picks. It used to be ONE constant, `_PIPE_M`, and one constant cannot express this surface:
#
#   rd (register-direct: no LDS, no barrier) re-reads the WHOLE B panel once per 16 rows of M — it has
#     no LDS staging and no M-register-blocking, so its cost grows as ~ceil(M/16)*OUT*IN. That is free
#     when the panel is small or M is small and catastrophic otherwise. The ISA says why, and it is
#     NOT this repo's usual answer: rd does not spill (VGPR 38-82, scratch 0, LDS 0 — the best
#     occupancy of the three). Per 16-wide K step at BN=64 it issues 5 global b128 loads, 4 WMMAs and
#     4 `s_wait_loadcnt` — a dependent load->wait->mma chain with ZERO prefetch depth, so every WMMA
#     is gated on a fresh ~300-cycle global load and only wave count hides any of it.
#   pipe prefetches a whole K-chunk ahead (A into a register double-buffer, B into a double-buffered
#     LDS tile), so its inner loop is ds_read + WMMA with the global loads a chunk in front: 32 WMMAs
#     per 6 `s_wait_loadcnt` at BN=64, and B read once per block_m (128-256) instead of once per 16.
#     The price is a fixed ~25-60 us floor (VGPR 153-217, LDS staging, a barrier per K-chunk) and a
#     coarse grid, so it needs BOTH a wide OUT and real M before it pays for itself.
#   lds is the general fallback and the only one that can mask a ragged OUT tile.
#
# Thresholds are from dense_gemm/local/sweep_policy.py: a CUDA-graph-timed sweep of every callable
# tile over the shapes this engine actually dispatches, with the weights rotated past the 64 MB
# Infinity Cache. Both of those matter. A per-call `synchronize()` has a ~40 us wall floor on this
# box — larger than most of these kernels, so it makes them all look identical below M~192. And a
# hot-weight benchmark flatters `rd` specifically, because re-reading B costs nothing when B never
# leaves cache; under a 30-layer ~700 MB weight working set it is not in cache.
#
# Result vs rocBLAS (best-of-family / rocBLAS, cold, RX 9070) on the four shapes Gemma4 and
# DiffusionGemma actually route here: gate_up 0.76-1.21x, down 1.05-1.19x, router 1.12-1.26x,
# lm_head 0.93-1.09x — we BEAT it at M=64..128 on gate_up and at M<=128 on the LM head. The residual
# deficit is bounded by ~1.26x and is kernel debt to close, not a reason to call rocBLAS: F.linear is
# reached only through minv_supported()'s dtype / ragged-K fallback.
_RD_MAX_M = int(os.environ.get("MINISGL_MINV_RD_MAX_M", "128"))
# ...but M alone is not enough, which is the flaw a single threshold cannot fix: rd's B re-read scales
# with OUT too, so an LM-head-width output leaves rd's regime almost immediately. At the SAME M=128,
# rd is the fastest kernel on gate_up (35.5 us vs pipe 51.0) and the slowest on the LM head (2216 us
# vs pipe 1242) — the ranking inverts on OUT, not on M. M*OUT is the term that separates them; 512K
# puts every real shape on the correct side with an order of magnitude of margin (gate_up M=128 ->
# 270K rd, down M=128 -> 360K rd, lm_head M=32 -> 4.2M pipe).
_RD_MAX_MN = int(os.environ.get("MINISGL_MINV_RD_MAX_MN", str(512 * 1024)))
# Narrow OUT: pipe's grid is (ceil(OUT/BN), ceil(M/block_m)). Gemma4's router is OUT=128 — TWO N-tiles
# at BN=64 — so pipe runs 2-4 workgroups on a 64-CU part and costs a FLAT ~42 us at every M, while rd
# does the same work in ~16 us. This is the regression the previous single `_PIPE_M=256` shipped: it
# sent every router call at the 256-token canvas to pipe, 2.6x slower, 30 calls per step.
_PIPE_MIN_OUT = int(os.environ.get("MINISGL_MINV_PIPE_MIN_OUT", "512"))
_PIPE_MI = int(os.environ.get("MINISGL_MINV_PIPE_MI", "2"))   # pipe register-block M-subtiles/warp
# pipe K-chunk. IN % PBK != 0 is FINE now — the kernel grew a zero-filled K-tail. It did not have one,
# and that silently excluded a whole layer class: a row-parallel down_proj shards IN by the TP degree,
# so Gemma4's intermediate 2112 becomes IN=1056 at TP=2, which is not a multiple of 64, so `mlp.down`
# could never reach pipe at any M and sat on rd forever (M=1024: 110 us rd -> 78 us pipe).
_PIPE_PBK = int(os.environ.get("MINISGL_MINV_PIPE_PBK", "64"))
_warned: set[str] = set()


def _warn_once(key: str, msg: str) -> None:
    if key not in _warned:
        _warned.add(key)
        try:
            from minisgl.utils import init_logger

            init_logger(__name__).warning(msg)
        except Exception:
            pass


# Decode fast path through the shared bf16/fp16 GEMV — see the note in minv_linear.
#
# UNCONDITIONAL — no env gate. Both sides of the threshold are individually M-invariant (the GEMV
# per-(row,col) in a fixed K-order; dense_gemm with a fixed full-K reduction), so the only hazard is
# the CROSSING between them. MAXM=16 is the M ceiling the GEMV accepts, so ordinary decode and
# spec-decode VERIFY (M=K+1) land on the SAME kernel rather than opposite sides of the threshold —
# that is what keeps verify bit-matching sequential decode.
# Measured vs the old threshold of 2: +19.8% at conc=4, +7.9% at conc=8, neutral at bs=1
# (tools/_maxm_ab.sh).
_DECODE_GEMV_MAXM = 16
_decode_gemv_fn = None
_decode_gemv_probed = False


def _get_decode_gemv():
    """Lazily resolve fp8_wmma.dense_bf16_gemv (None if the kernel package is unavailable)."""
    global _decode_gemv_fn, _decode_gemv_probed
    if not _decode_gemv_probed:
        _decode_gemv_probed = True
        try:
            from fp8_wmma import dense_bf16_gemv

            _decode_gemv_fn = dense_bf16_gemv
        except Exception:
            _decode_gemv_fn = None
    return _decode_gemv_fn


def minv_supported(x: torch.Tensor, weight: torch.Tensor) -> bool:
    """True iff `minv_linear` will run the M-invariant kernel (else it falls back to F.linear)."""
    if weight.dtype not in (torch.bfloat16, torch.float16):
        return False
    if weight.shape[-1] % 16 != 0:  # IN must be WMMA-friendly (full-K reduction in 16-wide steps)
        return False
    # Under CUDA-graph capture we STILL use dense_gemm (the whole point of removing rocBLAS): its only
    # allocations are F.pad + the output tensor, both via torch's caching allocator, which is capture-
    # safe (the graph memory pool). The old F.linear fallback here reintroduced rocBLAS/hipblaslt AND
    # faulted the spec-verify capture (hipblaslt's own workspace malloc is illegal mid-capture). Set
    # MINISGL_MINV_CAPTURE=0 to restore the fallback if a capture regression appears.
    if (os.environ.get("MINISGL_MINV_CAPTURE", "1") == "0"
            and torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()):
        return False
    return True


def minv_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None,
                *, block_m: int = _BLOCK_M, BN: int = _BN) -> torch.Tensor:
    """M-invariant drop-in for F.linear: C = x @ weight^T (+bias). Accepts N-D x (flattened to 2-D).
    Falls back to F.linear when the M-invariant kernel is unavailable for this dtype/shape/context."""
    import torch.nn.functional as F

    if not minv_supported(x, weight):
        if weight.dtype in (torch.bfloat16, torch.float16) and weight.shape[-1] % 16 != 0:
            _warn_once(f"K{weight.shape[-1]}",
                       f"minv_linear: IN={weight.shape[-1]} not a multiple of 16 -> F.linear fallback "
                       f"(this GEMM is NOT M-invariant; see layers/minv.py)")
        return F.linear(x, weight, bias)

    # DECODE fast path: the WMMA-tiled dense_gemm is built for large M and stalls at M=1 — it pads M
    # up to block_m (63/64 of the tile is padding) and the decode profile puts this chokepoint at
    # 2.58 ms/step (dense_gemm_rd, 160 calls) + 1.03 ms/step (the ragged dense_gemm on OUT=1
    # shared_expert_gate, 40 calls) = 3.6 ms of a 13.19 ms kernel step. The shared decode GEMV
    # (gemv_decode_core<Bf16GemvLoader>, packed v_dot2_f32_bf16) runs the same shapes at 1.5-3x —
    # it is the same trade the LM head already took (embedding.py) and the GDN projections took in
    # gdn/layer.py::_GemvLinear.
    #
    # M-invariant (row i is a per-(row,col) fp32 dot in a fixed K-order, bit-identical at any M —
    # measured max|d|=0.0 across M=1/2/8/16). It is NOT bit-identical to the dense_gemm family, so
    # the threshold is a crossing; MAXM=16 puts decode AND spec-verify on this side of it.
    if (x.dim() == 2 and x.shape[0] <= _DECODE_GEMV_MAXM and bias is None
            and x.dtype == weight.dtype):          # minv_supported() gated weight.dtype, not x's
        gemv = _get_decode_gemv()
        if gemv is not None and weight.shape[-1] % 8 == 0:
            return gemv(x.contiguous(), weight)

    import dense_gemm as _dg

    orig_shape = x.shape
    x2 = x.reshape(-1, orig_shape[-1]).contiguous()
    M, IN = x2.shape
    OUT = weight.shape[0]
    # All three dense_gemm kernels (rd / pipe / lds) run the FULL K-reduction per output tile in the
    # identical fixed 16-wide order with NO split-K, so every one is bit-identical per output row and
    # interchangeable -> the dispatch below is M-invariant regardless of which kernel a given M picks.
    # Full-tile kernels need block_m | M and BN | OUT; pad M up to the tile (padded rows are sliced off
    # and never touch the real rows). See the threshold block at the top of this file for the measured
    # surface each clause encodes.
    def _padded(bm: int) -> torch.Tensor:
        Mp = ((M + bm - 1) // bm) * bm
        return x2 if Mp == M else torch.nn.functional.pad(x2, (0, 0, 0, Mp - M))

    if OUT % BN != 0:
        # ragged OUT: only the LDS kernel masks a partial N-tile (a direct fragment load cannot).
        out = _dg.dense_gemm(x2, weight, block_m, BN)
    elif OUT < _PIPE_MIN_OUT or (M <= _RD_MAX_M and M * OUT <= _RD_MAX_MN):
        # rd's regime: either the N-grid is too narrow for pipe to fill the machine (routers), or the
        # B re-read volume is still small. rd BEATS rocBLAS through most of this band.
        out = _dg.dense_gemm_rd(_padded(block_m), weight, block_m, BN)[:M]
    else:
        # pipe: B read once per pbm rows, prefetched a K-chunk ahead.
        #   pbm=128 wins M=192..384 (grid stays wide, no wasted M-padding); pbm=256 wins from M>=512,
        #   and already from M>=192 on an LM-head-width OUT, where the N-grid is thousands of tiles
        #   wide so the only remaining lever is active warps per block (8 at pbm=256 vs 4 at 128).
        pbm = 256 if (M >= 512 or (OUT >= 65536 and M >= 192)) else 128
        out = _dg.dense_gemm_pipe(_padded(pbm), weight, pbm, BN, _PIPE_MI, _PIPE_PBK)[:M]
    if bias is not None:
        out = out + bias
    return out.reshape(*orig_shape[:-1], OUT)
