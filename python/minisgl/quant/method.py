from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import torch
import torch.nn.functional as F

from . import kernels
from .config import QuantConfig

if TYPE_CHECKING:
    from minisgl.layers.base import BaseOP


def _ct_packed_is_uint4b8(packed: torch.Tensor) -> bool:
    """Decide a compressed-tensors int4 checkpoint's packed sign convention from the nibble
    distribution. Returns True if the packed nibbles are already uint4b8 (q+8; mode at 8 for
    symmetric weights) -> pass through; False for two's-complement (mode at 0) -> XOR 0x88.
    Samples a slice (the decision is uniform across a tensor's nibbles)."""
    flat = packed.flatten()
    sample = flat[: min(flat.numel(), 1 << 16)].to(torch.int64) & 0xFFFFFFFF
    nib = torch.cat([(sample >> (4 * p)) & 0xF for p in range(8)])
    counts = torch.bincount(nib, minlength=16)
    return bool(counts[8] >= counts[0])


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
        # Declare buffers in CHECKPOINT layout so BaseOP load matches. (N=out, K=in are the
        # LOCAL/per-TP sizes; TP-quant sharding is a follow-up.)
        pf = 32 // self.quant.bits
        g = self.quant.group_size
        N, K = out_features, in_features
        if self.quant.is_gptq:
            # GPTQ "gemm" layout: qweight int32 packed along INPUT (K//pf, N), per-group
            # scales (K//g, N), and qzeros (K//g, N//pf) ALWAYS present (even symmetric — the
            # constant zero is stored explicitly; MoeWNA16 likewise loads then folds it).
            assert K % pf == 0 and K % g == 0 and N % pf == 0, (
                f"GPTQ needs K%{pf}==0,K%{g}==0,N%{pf}==0; got N={N},K={K}"
            )
            layer.qweight = torch.empty((K // pf, N), dtype=torch.int32)
            layer.scales = torch.empty((K // g, N), dtype=torch.float16)
            layer.qzeros = torch.empty((K // g, N // pf), dtype=torch.int32)
            if self.quant.desc_act:
                layer.g_idx = torch.empty((K,), dtype=torch.int32)
            return
        if self.quant.is_compressed_tensors:
            # compressed-tensors W4A16 DENSE linear: weight_packed (N, K//pf) int32 (8 SIGNED int4 per
            # int32, natural K order) + weight_scale (N, K//g). Already the op's natural nibble order,
            # so post_load is a whole-tensor fixup (XOR 0x88 + constant zero-point 8) — the same
            # conversion _GroupedCompressedTensorsExperts uses, minus the E dim. `weight_shape` in the
            # checkpoint is ignored by the loader.
            assert K % pf == 0 and K % g == 0 and N % pf == 0, (
                f"CT dense needs K%{pf}==0,K%{g}==0,N%{pf}==0; got N={N},K={K}"
            )
            layer.weight_packed = torch.empty((N, K // pf), dtype=torch.int32)
            # pack-quantized scales ship fp16 (some heads ship bf16); the loader normalizes both to
            # fp16 (engine._cast) so this one declared dtype matches every CT checkpoint.
            layer.weight_scale = torch.empty((N, K // g), dtype=torch.float16)
            if not self.quant.sym:
                # ASYMMETRIC: per-group weight_zero_point, int4-packed 8-per-int32 along the OUTPUT
                # dim (shape [N//pf, G]) — already the op's zeros layout. Loaded + used in process().
                layer.weight_zero_point = torch.empty((N // pf, K // g), dtype=torch.int32)
            return
        # AWQ "gemm" layout: qweight (K, N//pf) i32, scales (K//group, N) f16,
        # qzeros (K//group, N//pf) i32 (asymmetric only).
        assert N % pf == 0 and K % g == 0, f"W4A8 needs N%{pf}==0,K%{g}==0; got N={N},K={K}"
        layer.qweight = torch.empty((K, N // pf), dtype=torch.int32)
        layer.scales = torch.empty((K // g, N), dtype=torch.float16)
        if not self.quant.sym:
            layer.qzeros = torch.empty((K // g, N // pf), dtype=torch.int32)

    def process_weights_after_load(self, layer: "BaseOP") -> None:
        if self.quant.is_compressed_tensors:
            # CT DENSE -> op layout (constant zero-point 8; zeros_op all 0x88; scales as-is).
            # The op wants weights as uint4b8 (nibble = q + 8). compressed-tensors "pack-quantized"
            # ships int4 in one of TWO packings, per producer, that we must distinguish per checkpoint:
            #   * two's-complement signed int4 (nibble = q & 0xF): convert to uint4b8 by flipping each
            #     nibble's top bit — XOR 0x88 per byte — since (q&0xF)^8 == q+8 for q in [-8,7].
            #   * already-offset uint4b8 (nibble = q + 8; AWQ-style zero_point=8, e.g. cyankiwi's
            #     Qwen3.5/3.6 "AWQ-*-INT4" dense checkpoints): pass through UNCHANGED — an XOR here
            #     would scramble it (it re-flips the top bit) and produce garbage.
            # Symmetric weights make the two trivially separable by the packed nibble distribution:
            # uint4b8 is a bell curve with its mode at 8 (q=0); two's-complement's mode is at 0.
            pf = 32 // self.quant.bits
            N, Kp = layer.weight_packed.shape  # type: ignore[attr-defined]
            G = layer.weight_scale.shape[-1]  # type: ignore[attr-defined]
            wp = layer.weight_packed.contiguous()  # type: ignore[attr-defined]
            uint4b8 = _ct_packed_is_uint4b8(wp)
            if uint4b8:
                layer._w_packed_op = wp
            else:
                layer._w_packed_op = (wp.view(torch.uint8) ^ 0x88).view(torch.int32).contiguous()
            layer._scales_op = layer.weight_scale.to(torch.float16).contiguous()  # type: ignore[attr-defined]
            zp = getattr(layer, "weight_zero_point", None)
            if zp is None:
                # SYMMETRIC: constant zero-point 8 (uint4b8), zeros_op all 0x88.
                zeros = torch.empty((N // pf, G), dtype=torch.int32)
                zeros.view(torch.uint8).fill_(0x88)
                layer._zeros_op = zeros.to(wp.device)
            else:
                # ASYMMETRIC: real per-group zero_point, already int4-packed [N//pf, G] along N (the
                # op's zeros layout). It shares the weight's sign convention (same quantizer), so apply
                # the SAME uint4b8-vs-two's-complement transform: W_u and Z_u then live in one unsigned
                # domain and the op computes scale*(W_u - Z_u) = scale*(q - zp), exact.
                zp = zp.contiguous()
                layer._zeros_op = zp if uint4b8 else (zp.view(torch.uint8) ^ 0x88).view(torch.int32).contiguous()
                del layer.weight_zero_point
            del layer.weight_packed, layer.weight_scale
            return
        qz = getattr(layer, "qzeros", None)
        if self.quant.is_gptq:
            # GPTQ -> op layout (2M-2). desc_act is asserted off (g_idx identity) by the converter.
            assert not self.quant.desc_act, "GPTQ desc_act (act-order) not supported"
            w_packed, scales_op, zeros_op = kernels.gptq_to_op_layout(
                layer.qweight, layer.scales, qz, bits=self.quant.bits  # type: ignore[attr-defined]
            )
        else:
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


class RXFLinearMethod:
    """RXF ("Rotated eXtra Fast") W4(NL codebook)-A8(int8) linear, native HIP (rxf_hip).

    The checkpoint already ships op-layout (no AWQ/GPTQ unpack-transpose-repack): a uint8
    weight_packed (N, K/2) of NL indices and an fp16 per-group weight_scale (N, K/32), group=32.
    The weights were rotated offline by a fixed block-diagonal Hadamard (FWHT-span); apply()
    rotates+int8-quantizes the activation with the SAME span so the rotation cancels in the dot
    (and spreads activation outliers to tighten the 4-bit scale). NL codebook is model-wide
    (kernels._rxf_nl). No zero-points (the NL codebook is symmetric)."""

    def __init__(self, quant: QuantConfig) -> None:
        self.quant = quant

    def create_weights(self, layer: "BaseOP", out_features: int, in_features: int) -> None:
        N, K = out_features, in_features
        span = self.quant.rotation_span
        assert K % 32 == 0 and K % span == 0 and K % 2 == 0, (
            f"RXF needs K%32==0,K%span({span})==0,K%2==0; got N={N},K={K}"
        )
        layer.weight_packed = torch.empty((N, K // 2), dtype=torch.uint8)
        layer.weight_scale = torch.empty((N, K // 32), dtype=torch.float16)

    def apply(
        self, layer: "BaseOP", x: torch.Tensor, bias: torch.Tensor | None
    ) -> torch.Tensor:
        out = kernels.rxf_linear(
            x,
            layer.weight_packed,  # type: ignore[attr-defined]
            layer.weight_scale,  # type: ignore[attr-defined]
            bias,
            self.quant.rotation_span,
        )
        return out.to(x.dtype)


def create_linear_method(
    quant: QuantConfig | None, *, quantized: bool = True
) -> LinearMethod:
    """Pick the method for a linear layer. `quantized=False` (e.g. lm_head) always
    stays unquantized even when the model is quantized."""
    if quant is None or not quantized:
        return UnquantizedLinearMethod()
    if quant.is_rxf:
        return RXFLinearMethod(quant)
    # fp8 W8A8 (compressed-tensors float-quantized 8-bit) has no dense linear kernel here — only the
    # MoE expert path (create_moe_quant_method) implements it. ZAYA's dense/attn linears are all in
    # the quant `ignore` list, so a quantized fp8 config never reaches a dense linear; guard anyway so
    # it can't silently misroute into the int4 W4A8 layout.
    if quant.is_fp8_w8a8:
        raise NotImplementedError(
            "fp8 W8A8 dense linear not implemented (MoE-expert-only); fp8 modules must be unquantized"
        )
    return W4A8LinearMethod(quant)
