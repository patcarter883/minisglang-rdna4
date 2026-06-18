"""Swappable quantized-kernel provider.

ALL quantized GEMM call sites in the engine go through this module, so the parallel
custom kernel framework can be dropped in by swapping these impls — without touching
layers / models / the weight loader. The current impl wraps the in-repo
`w4a8_fp8_wmma` kernel (int4 weights + fp8 activations -> native RDNA4 fp8 WMMA;
nothing dequants to F16 — the fp16 I/O is the op's staging boundary, compute is e4m3).

Weight-layout conversion (AWQ g128 -> op g32, AutoGPTQ bit order, transpose;
compressed-tensors g32 ~= native) also belongs here — ported from
vllm-gfx1201/w4a8_fp8_wmma/{vllm_adapter.py,moe_experts.py} — TODO Phase 2c.
"""
from __future__ import annotations

import torch


def _pick_dense_version(m: int, k: int, group_size: int) -> int:
    """Per-M kernel-variant ladder (simplified from vllm_adapter.py). The full adapter
    also has env tuning + a Triton W4A16 small-M/large-group fallback (PERF_NOTES)."""
    if m <= 2 and k % 1024 == 0 and group_size % 32 == 0:
        return 11  # decode
    if group_size in (32, 128):
        return 10  # mid/prefill workhorse
    return 5  # large-M


def w4a8_linear(
    x: torch.Tensor,
    w_packed: torch.Tensor,  # (N, K/8) int32, op layout
    scales: torch.Tensor,  # (N, K/group) fp16
    w_zeros: torch.Tensor | None,  # (N/8, K/group) int32 (AWQ asym) or None (sym)
    group_size: int,
    version: int | None = None,
) -> torch.Tensor:
    """Dense W4A8 GEMM: (M, K) @ (N, K)^T -> (M, N). Returns the op's fp16 output;
    the caller casts back to the activation dtype."""
    import w4a8_fp8_wmma

    x2d = x if x.dtype == torch.float16 else x.to(torch.float16)  # op computes in fp16
    k = x2d.shape[-1]
    if version is None:
        version = _pick_dense_version(x2d.shape[0], k, group_size)
    return w4a8_fp8_wmma.mmq_fp8_gemm(x2d, w_packed, scales, version=version, w_zeros=w_zeros)
