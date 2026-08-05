from typing import Tuple

import torch

from . import _tail_hip
from .base import BaseOP, StateLessOP


def _rms_norm(
    x: torch.Tensor, weight: torch.Tensor | None, eps: float, plus_one: bool = False
) -> torch.Tensor:
    # fp32-internal RMSNorm over the last dim; in-dtype out (no fp32 dequant).
    # plus_one: the (1 + weight) gain convention (Qwen3.5 / Gemma — weight is centered on 0).
    # weight=None: the UNWEIGHTED norm (transformers `with_scale=False`), for which the checkpoint
    # ships no tensor. The native kernel takes an optional gain, so it serves that case too.
    if _tail_hip.active(x, weight):
        return _tail_hip.rms_norm(x.contiguous(), weight, eps, int(plus_one))
    dtype = x.dtype
    xf = x.float()
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    normed = (xf * torch.rsqrt(var + eps)).to(dtype)
    if weight is None:
        return normed
    return normed * (weight + 1.0) if plus_one else normed * weight


class RMSNorm(BaseOP):
    def __init__(self, size: int, eps: float, *, plus_one: bool = False) -> None:
        self.eps = eps
        self.plus_one = plus_one
        self.weight = torch.empty(size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _rms_norm(x, self.weight, self.eps, self.plus_one)

    def forward_inplace(self, x: torch.Tensor) -> None:
        x.copy_(_rms_norm(x, self.weight, self.eps, self.plus_one))


class RMSNormNoScale(StateLessOP):
    """RMSNorm with NO learned gain (transformers' `with_scale=False`).

    Gemma4 uses three of these — `self_attn.v_norm`, `router.norm`, and the vision embedder's
    pre-projection norm — and because `with_scale=False` creates no parameter, the checkpoint ships
    NO weight tensor for them. That makes them easy to skip by accident: nothing in the state dict
    hints they exist, and dropping one leaves the model running with un-normalized V (or un-normalized
    router input) — plausible output, quietly wrong. Being a StateLessOP keeps it out of the state
    dict, so the loader's exact-key check stays honest."""

    def __init__(self, eps: float) -> None:
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Same native kernel as the weighted norm, with a null gain — NOT a separate torch
        # decomposition. Gemma4 runs two of these per layer (v_norm + the router norm), so leaving
        # them in torch would have kept 60 multi-launch norms per step after the weighted ones moved.
        return _rms_norm(x, None, self.eps)


class RMSNormFused(BaseOP):
    def __init__(self, size: int, eps: float, *, plus_one: bool = False) -> None:
        self.eps = eps
        self.plus_one = plus_one
        self.weight = torch.empty(size)

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return _rms_norm(x, self.weight, self.eps, self.plus_one), x
        # fused residual-add: residual <- x + residual (new residual); out = rmsnorm(sum)
        if _tail_hip.active(x, residual, self.weight) and residual.is_contiguous():
            # rms_norm_add mutates `residual` in place to (x + residual) and returns rmsnorm(sum).
            # residual must stay the SAME storage (residual stream), so it is never .contiguous()'d.
            out = _tail_hip.rms_norm_add(
                x.contiguous(), residual, self.weight, self.eps, int(self.plus_one)
            )
            return out, residual
        residual.add_(x)
        x.copy_(_rms_norm(residual, self.weight, self.eps, self.plus_one))
        return x, residual
