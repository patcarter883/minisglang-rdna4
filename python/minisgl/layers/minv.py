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
    # A malloc (arange/route tensors) is illegal mid-capture; static-shape decode is self-consistent.
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
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
    # ONE fixed tile (block_m=BN => trivially M-invariant: single algorithm, single reduction order).
    # Register-direct (LDS-bypass) beats rocBLAS across the serving-relevant M range (decode batch +
    # chunked prefill) but needs full tiles (block_m | M, BN | OUT). Pad M up to block_m (rows sliced
    # off; they never touch the real rows) when OUT is BN-aligned; otherwise use the LDS kernel, which
    # masks ragged OUT. Both kernels share the identical K-reduction order -> bit-identical.
    if OUT % BN == 0:
        Mp = ((M + block_m - 1) // block_m) * block_m
        xp = x2 if Mp == M else torch.nn.functional.pad(x2, (0, 0, 0, Mp - M))
        out = _dg.dense_gemm_rd(xp, weight, block_m, BN)[:M]
    else:
        out = _dg.dense_gemm(x2, weight, block_m, BN)  # LDS kernel: handles ragged OUT
    if bias is not None:
        out = out + bias
    return out.reshape(*orig_shape[:-1], OUT)
