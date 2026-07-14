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
    kernels are already fixed-tile HIP. Global off-switch: MINISGL_MINV_GEMM=0.
"""
from __future__ import annotations

import os

import torch

_MINV_ON = os.environ.get("MINISGL_MINV_GEMM", "1") != "0"
_BLOCK_M = int(os.environ.get("MINISGL_MINV_BLOCK_M", "64"))  # WMMA M-tile (mult of 16, <=128)
_BN = int(os.environ.get("MINISGL_MINV_BN", "64"))            # WMMA N-tile (divides most OUT dims)
# Large-M path: the deep-pipelined dense_gemm_pipe kernel runs at rocBLAS parity at large M (gate_up
# M=4096: ~113 vs rocBLAS ~114 TFLOPS, cca_q/down M=1024 BEAT it) after the RDNA4 LDS bank-conflict pad.
# It is BIT-IDENTICAL to the register-direct (rd) and LDS kernels (same 16-wide K-reduction order, no
# split-K), so switching rd<->pipe by M stays M-invariant. rd still wins the small-M decode hot path
# (register-direct, no LDS staging / __syncthreads overhead), so we only reach for pipe at M >= _PIPE_M.
_PIPE_M = int(os.environ.get("MINISGL_MINV_PIPE_M", "512"))   # M threshold to switch rd -> pipe
_PIPE_MI = int(os.environ.get("MINISGL_MINV_PIPE_MI", "2"))   # pipe register-block M-subtiles/warp
_PIPE_PBK = int(os.environ.get("MINISGL_MINV_PIPE_PBK", "64"))  # pipe K-chunk (needs IN % PBK == 0)
_warned: set[str] = set()


def _warn_once(key: str, msg: str) -> None:
    if key not in _warned:
        _warned.add(key)
        try:
            from minisgl.utils import init_logger

            init_logger(__name__).warning(msg)
        except Exception:
            pass


def minv_supported(x: torch.Tensor, weight: torch.Tensor) -> bool:
    """True iff `minv_linear` will run the M-invariant kernel (else it falls back to F.linear)."""
    if not _MINV_ON:
        return False
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
        if _MINV_ON and weight.dtype in (torch.bfloat16, torch.float16) and weight.shape[-1] % 16 != 0:
            _warn_once(f"K{weight.shape[-1]}",
                       f"minv_linear: IN={weight.shape[-1]} not a multiple of 16 -> F.linear fallback "
                       f"(this GEMM is NOT M-invariant; see layers/minv.py)")
        return F.linear(x, weight, bias)

    import dense_gemm as _dg

    orig_shape = x.shape
    x2 = x.reshape(-1, orig_shape[-1]).contiguous()
    M, IN = x2.shape
    OUT = weight.shape[0]
    # All three dense_gemm kernels (rd / pipe / lds) run the FULL K-reduction per output tile in the
    # identical fixed 16-wide order with NO split-K, so every one is bit-identical per output row and
    # interchangeable -> the dispatch below is M-invariant regardless of which kernel a given M picks.
    #   * ragged OUT (OUT % BN != 0): the LDS kernel masks it (register-direct can't mask a ragged tile).
    #   * large M (>= _PIPE_M, full tiles, IN % PBK == 0): the deep-pipelined kernel at ~rocBLAS parity.
    #   * else (small-M decode hot path): register-direct (LDS-bypass), which wins there.
    # Full-tile kernels need block_m | M and BN | OUT; pad M up to the tile (padded rows are sliced off
    # and never touch the real rows).
    if OUT % BN != 0:
        out = _dg.dense_gemm(x2, weight, block_m, BN)  # LDS kernel: handles ragged OUT
    elif M >= _PIPE_M and IN % _PIPE_PBK == 0:
        pbm = 256 if M >= 1024 else 128                # deeper M -> more warps sharing the LDS B tile
        pbn = 128 if OUT % 128 == 0 else BN            # wider N-tile when OUT allows (fewer A reloads)
        Mp = ((M + pbm - 1) // pbm) * pbm
        xp = x2 if Mp == M else torch.nn.functional.pad(x2, (0, 0, 0, Mp - M))
        out = _dg.dense_gemm_pipe(xp, weight, pbm, pbn, _PIPE_MI, _PIPE_PBK)[:M]
    else:
        Mp = ((M + block_m - 1) // block_m) * block_m
        xp = x2 if Mp == M else torch.nn.functional.pad(x2, (0, 0, 0, Mp - M))
        out = _dg.dense_gemm_rd(xp, weight, block_m, BN)[:M]
    if bias is not None:
        out = out + bias
    return out.reshape(*orig_shape[:-1], OUT)
