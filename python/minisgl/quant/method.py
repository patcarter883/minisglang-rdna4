from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import torch
import torch.nn.functional as F

from . import kernels
from .config import QuantConfig

if TYPE_CHECKING:
    from minisgl.layers.base import BaseOP


@runtime_checkable
class LinearMethod(Protocol):
    """How a parallel-linear layer allocates its weights and computes its matmul.
    The layer owns sharding + collectives; the method owns weight layout + the GEMM."""

    def create_weights(self, layer: "BaseOP", out_features: int, in_features: int) -> None:
        """Declare weight tensors on `layer` as plain (non-underscore) attributes so
        BaseOP's __dict__ introspection serializes/loads them. Shapes/dtypes MUST match
        the checkpoint exactly (BaseOP.load_state_dict asserts both)."""
        ...

    def apply(
        self, layer: "BaseOP", x: torch.Tensor, bias: torch.Tensor | None
    ) -> torch.Tensor: ...


class UnquantizedLinearMethod:
    """bf16/f16 dense `F.linear` — the default, unchanged behavior."""

    def create_weights(self, layer: "BaseOP", out_features: int, in_features: int) -> None:
        layer.weight = torch.empty(out_features, in_features)

    def apply(
        self, layer: "BaseOP", x: torch.Tensor, bias: torch.Tensor | None
    ) -> torch.Tensor:
        return F.linear(x, layer.weight, bias)


class W4A8LinearMethod:
    """int4-weight / fp8-activation WMMA via the swappable kernel provider.

    Phase 2 scaffold. The op consumes weights in its native layout
    (w_packed (N, K/8) i32, scales (N, K/32) f16, zeros (N/8, K/32) i32 | None).
    Checkpoints arrive in AWQ (g128, AutoGPTQ bit order, K-major) or compressed-tensors
    (g32) layout, so weights are declared in CHECKPOINT layout (to load) then converted
    to op layout once after load.

    TODO Phase 2c: implement create_weights (exact AWQ/CT buffer shapes) +
    process_weights_after_load (port the conversion from
    vllm-gfx1201/w4a8_fp8_wmma/{vllm_adapter.py:_awq_to_op_layout, moe_experts.py}).
    apply() below is the final call shape and is ready once op-layout buffers exist.
    """

    def __init__(self, quant: QuantConfig) -> None:
        self.quant = quant

    def create_weights(self, layer: "BaseOP", out_features: int, in_features: int) -> None:
        # Declare buffers in CHECKPOINT (AWQ "gemm") layout so BaseOP load matches:
        #   qweight (K, N//pf) i32, scales (K//group, N) f16, qzeros (K//group, N//pf) i32.
        # (N=out, K=in are the LOCAL/per-TP sizes; TP-quant sharding is a follow-up.)
        pf = 32 // self.quant.bits
        g = self.quant.group_size
        N, K = out_features, in_features
        assert N % pf == 0 and K % g == 0, f"W4A8 needs N%{pf}==0,K%{g}==0; got N={N},K={K}"
        layer.qweight = torch.empty((K, N // pf), dtype=torch.int32)
        layer.scales = torch.empty((K // g, N), dtype=torch.float16)
        if not self.quant.sym:
            layer.qzeros = torch.empty((K // g, N // pf), dtype=torch.int32)

    def process_weights_after_load(self, layer: "BaseOP") -> None:
        qz = getattr(layer, "qzeros", None)
        w_packed, scales_op, zeros_op = kernels.awq_to_op_layout(
            layer.qweight, layer.scales, qz, bits=self.quant.bits  # type: ignore[attr-defined]
        )
        # op-layout buffers are derived (underscore -> not re-serialized); free the loaded ones.
        layer._w_packed_op = w_packed
        layer._scales_op = scales_op
        layer._zeros_op = zeros_op
        del layer.qweight, layer.scales
        if qz is not None:
            del layer.qzeros

    def apply(
        self, layer: "BaseOP", x: torch.Tensor, bias: torch.Tensor | None
    ) -> torch.Tensor:
        out = kernels.w4a8_linear(
            x,
            layer._w_packed_op,  # type: ignore[attr-defined]
            layer._scales_op,  # type: ignore[attr-defined]
            layer._zeros_op,  # type: ignore[attr-defined]
            self.quant.group_size,
        )
        out = out.to(x.dtype)
        if bias is not None:
            out = out + bias
        return out


def create_linear_method(
    quant: QuantConfig | None, *, quantized: bool = True
) -> LinearMethod:
    """Pick the method for a linear layer. `quantized=False` (e.g. lm_head) always
    stays unquantized even when the model is quantized."""
    if quant is None or not quantized:
        return UnquantizedLinearMethod()
    return W4A8LinearMethod(quant)
