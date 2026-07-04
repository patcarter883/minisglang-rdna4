from typing import Tuple

import torch

from . import _tail_hip
from .base import BaseOP


def _rms_norm(
    x: torch.Tensor, weight: torch.Tensor, eps: float, plus_one: bool = False
) -> torch.Tensor:
    # fp32-internal RMSNorm over the last dim; bf16/f16 in -> same dtype out (no F16 dequant).
    # plus_one: the (1 + weight) gain convention (Qwen3.5 / Gemma — weight is centered on 0).
    if _tail_hip.active(x, weight):
        return _tail_hip.rms_norm(x.contiguous(), weight, eps, int(plus_one))
    dtype = x.dtype
    xf = x.float()
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    normed = (xf * torch.rsqrt(var + eps)).to(dtype)
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
