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


def gelu_tanh_and_mul(x: torch.Tensor, out: torch.Tensor | None = None):
    # HF `gelu_pytorch_tanh` (Gemma4's routed-expert + dense-MLP activation): the TANH APPROXIMATION
    # 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3))), which is NOT what `gelu_and_mul` above computes —
    # that one is the exact erf gelu, in both its native (`_tail_hip.gelu_and_mul`) and torch
    # (`approximate="none"`) arms. The two agree to ~1e-3 absolute, so substituting one for the other
    # never crashes and never trips a shape/dtype check; it just quietly shifts every expert output.
    # Hence a SEPARATE entry point rather than a keyword on gelu_and_mul: a caller that means "tanh"
    # must not be able to land on the erf path by defaulting.
    # The native kernel is the SAME gated-mul core as silu/erf-gelu with a different activation
    # policy (KERNEL_CORE_POLICY: a new activation is a policy on the shared core, not a new
    # kernel). `is not None` because an older baked .so predates the op.
    if _tail_hip.active(x) and _tail_hip.gelu_tanh_and_mul is not None:
        result = _tail_hip.gelu_tanh_and_mul(x.contiguous())
        if out is not None:
            out.copy_(result)
            return out
        return result
    return _gated(x, lambda t: F.gelu(t, approximate="tanh"), out)


__all__ = ["silu_and_mul", "gelu_and_mul", "gelu_tanh_and_mul"]
