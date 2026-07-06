from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F

from . import _tail_hip


def _gated(
    x: torch.Tensor, act: Callable[[torch.Tensor], torch.Tensor], out: torch.Tensor | None
):
    # gated activation over a [..., 2d] tensor: act(x[..., :d]) * x[..., d:].
    # fp32-internal, in-dtype out (no F16 dequant).
    d = x.shape[-1] // 2
    a, b = x[..., :d], x[..., d:]
    result = (act(a.float()) * b.float()).to(x.dtype)
    if out is not None:
        out.copy_(result)
        return out
    return result


def silu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None):
    if _tail_hip.active(x):
        result = _tail_hip.silu_and_mul(x.contiguous())  # act(x[:,:d]) * x[:,d:]
        if out is not None:
            out.copy_(result)
            return out
        return result
    return _gated(x, F.silu, out)


def gelu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None):
    # `_tail_hip.gelu_and_mul` is None when the baked kernel is silu-only (see _tail_hip.py) — fall
    # back to the torch path in that case rather than calling None.
    if _tail_hip.active(x) and _tail_hip.gelu_and_mul is not None:
        result = _tail_hip.gelu_and_mul(x.contiguous())  # exact (erf) gelu(x[:,:d]) * x[:,d:]
        if out is not None:
            out.copy_(result)
            return out
        return result
    return _gated(x, lambda t: F.gelu(t, approximate="none"), out)


__all__ = ["silu_and_mul", "gelu_and_mul"]
