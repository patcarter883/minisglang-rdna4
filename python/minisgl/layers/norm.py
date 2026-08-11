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


def _rms_norm_quant(
    x: torch.Tensor, weight: torch.Tensor | None, eps: float, plus_one: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """The SAME norm, plus the fp8-e4m3 form of its own output row.

    PRODUCER-SIDE ACTIVATION QUANT. A w4a8/w8a8 linear needs per-token fp8 activations; today the
    GEMM op launches its own `compute_act_fp8_and_scales_kernel` and re-reads the whole (M,K)
    activation to build them. The norm that produced those values already held them in registers, one
    block per row, with a block reduce — so the quant is an EPILOGUE there, not a kernel anywhere.
    It is deliberately NOT a standalone elementwise op (that trades one dispatch for another), and
    the quant stays SEPARATE from the GEMM (the core keeps consuming `x_fp8` + `act_scales`).

    Returns (out, x_fp8, act_scales) — a bit-identical `out`, and a pair `w4a8_linear` consumes
    bit-identically — or None when the native kernel is unavailable (a torch-decomposition dtype, or
    a tail_hip build predating the op). None is the ONLY fallback, it is explicit at the call site,
    and it changes performance, never numerics.
    """
    if not _tail_hip.active(x, weight) or not hasattr(_tail_hip, "rms_norm_quant"):
        return None
    return _tail_hip.rms_norm_quant(x.contiguous(), weight, eps, int(plus_one))


def _rms_norm_add_quant(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor | None, eps: float,
    plus_one: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """`_rms_norm_quant` for the fused residual-add norm. `residual` is mutated in place to
    (x + residual) exactly as `rms_norm_add` does, and must stay the SAME storage."""
    if (not _tail_hip.active(x, residual, weight) or not residual.is_contiguous()
            or not hasattr(_tail_hip, "rms_norm_add_quant")):
        return None
    return _tail_hip.rms_norm_add_quant(x.contiguous(), residual, weight, eps, int(plus_one))


class RMSNorm(BaseOP):
    def __init__(self, size: int, eps: float, *, plus_one: bool = False) -> None:
        self.eps = eps
        self.plus_one = plus_one
        self.weight = torch.empty(size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _rms_norm(x, self.weight, self.eps, self.plus_one)

    def forward_quant(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """`forward`, plus the fp8-e4m3 form of the output for the w4a8 linears it feeds.

        The un-fused twin of `RMSNormFused.forward_quant`, for the PRE-norm position that has no
        residual to add — which is where an attention block's q/k/v projections get their input.
        Returns (out, x_fp8, act_scales); the pair is None when the native kernel does not apply, and
        the caller then simply does not pass it, which re-quantizes exactly as before.

        WHY THIS ONE MATTERS MORE THAN ITS SHAPE SUGGESTS. In Gemma4 `input_layernorm`'s output feeds
        THREE separate dense linears — q_proj/k_proj/v_proj are not merged, the checkpoint ships them
        apart — and each was launching its own `compute_act_fp8_and_scales_kernel` over the SAME
        (M, K) rows. One producer quant therefore replaces THREE consumer quants, not one.

        `out` is bit-identical to `forward` either way, so a call site can switch on this without a
        numerics review."""
        r = _rms_norm_quant(x, self.weight, self.eps, self.plus_one)
        if r is None:
            return _rms_norm(x, self.weight, self.eps, self.plus_one), None, None
        return r[0], r[1], r[2]

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

    def forward_inplace(self, x: torch.Tensor) -> None:
        """In-place form, so a scaleless norm can stand in for the weighted `q_norm`/`k_norm` an
        `AttentionLayer` applies to the split-out q/k views. Muse-Glimmer's QK-norm is exactly this:
        `with_scale=False`, hence no `q_norm.weight`/`k_norm.weight` in its checkpoint."""
        x.copy_(_rms_norm(x, None, self.eps))


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

    def forward_quant(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """`forward`, plus the fp8-e4m3 form of the output for a downstream w4a8/w8a8 linear.

        Returns (out, residual, x_fp8, act_scales). The pair is None when the fused kernel does not
        apply, and the caller then just does not pass it to the linear — which re-quantizes exactly
        as it always did. `out` and `residual` are bit-identical to `forward` either way, so a call
        site can switch on this without a numerics review.

        See `minisgl.quant.kernels.w4a8_linear`'s x_fp8/act_scales for why this lives on the PRODUCER:
        the norm already holds the row in registers, so the alternative is a separate act-quant
        dispatch plus an (M,K) re-read per dense linear.
        """
        if residual is None:
            r = _rms_norm_quant(x, self.weight, self.eps, self.plus_one)
            if r is None:
                return _rms_norm(x, self.weight, self.eps, self.plus_one), x, None, None
            return r[0], x, r[1], r[2]
        r = _rms_norm_add_quant(x, residual, self.weight, self.eps, self.plus_one)
        if r is None:
            out, res = self.forward(x, residual)
            return out, res, None, None
        return r[0], residual, r[1], r[2]
