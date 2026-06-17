from typing import Tuple

import torch

from .base import BaseOP


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    # fp32-internal RMSNorm over the last dim; bf16/f16 in -> same dtype out (no F16 dequant).
    dtype = x.dtype
    xf = x.float()
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    normed = (xf * torch.rsqrt(var + eps)).to(dtype)
    return normed * weight


class RMSNorm(BaseOP):
    def __init__(self, size: int, eps: float) -> None:
        self.eps = eps
        self.weight = torch.empty(size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _rms_norm(x, self.weight, self.eps)

    def forward_inplace(self, x: torch.Tensor) -> None:
        x.copy_(_rms_norm(x, self.weight, self.eps))


class RMSNormFused(BaseOP):
    def __init__(self, size: int, eps: float) -> None:
        self.eps = eps
        self.weight = torch.empty(size)

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return _rms_norm(x, self.weight, self.eps), x
        # fused residual-add: residual <- x + residual (new residual); x <- rmsnorm(sum)
        residual.add_(x)
        x.copy_(_rms_norm(residual, self.weight, self.eps))
        return x, residual
