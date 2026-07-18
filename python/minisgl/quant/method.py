from __future__ import annotations

import os
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import torch
import torch.nn.functional as F

from . import kernels
from .config import QuantConfig

if TYPE_CHECKING:
    from minisgl.layers.base import BaseOP

# Fused dense gate_up + silu_and_mul (mmq_fp8_gemm_silu) for a MERGED gate_up projection at decode:
# one kernel writes silu(gate)*up, dropping the separate silu launch + the [.., 2*inter] HBM round-trip.
# Bit-exact to w4a8_linear(gate_up) + silu_and_mul. Only the decode-gemv path is fused (M<=16, K%512==0,
# group_size%32==0); otherwise apply_swiglu returns None and Linear.forward_swiglu falls back unfused.
# MINISGL_DENSE_FUSED_SILU=0 reverts.
_DENSE_FUSED_SILU = os.environ.get("MINISGL_DENSE_FUSED_SILU", "1") != "0"


def _fused_swiglu_ok(x: torch.Tensor, w_packed: torch.Tensor, group_size: int) -> bool:
    """Shape gate for the fused dense gemm+silu decode kernel (see mmq_fp8_gemm_silu constraints)."""
    if not _DENSE_FUSED_SILU:
        return False
    N = w_packed.shape[0]
    return x.shape[0] <= 16 and x.shape[1] % 512 == 0 and group_size % 32 == 0 and N % 2 == 0


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
    """bf16/f16 dense linear. Routes through the engine's M-invariant WMMA GEMM (layers/minv.py) so a
    chunked / prefix-cached / spec-verify forward matches a fresh one bit-for-bit; falls back to
    F.linear for dtypes/shapes/contexts the kernel doesn't cover (fp32, IN%16!=0, cudagraph capture).
    This is the single chokepoint every unquantized Linear in every model flows through."""

    def create_weights(self, layer: "BaseOP", out_features: int, in_features: int) -> None:
        layer.weight = torch.empty(out_features, in_features)

    def apply(
        self, layer: "BaseOP", x: torch.Tensor, bias: torch.Tensor | None
    ) -> torch.Tensor:
        from minisgl.layers.minv import minv_linear

        return minv_linear(x, layer.weight, bias)


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
        if kernels.MOE_W4A16 != "0":
            # W4A16 (fp16-act) dense path: repack -> register-direct wide weights, drop the op-layout.
            import fp8_wmma

            N, K8 = w_packed.shape
            wide = kernels._w4a16_wide(self.quant.group_size)
            w_rep = fp8_wmma.repack_int4_to_w_rep(w_packed, N, K8 * 8)
            layer._w_rep_wide = fp8_wmma.repack_w_rep_wide(w_rep, wide)
            layer._n_out = N
            del layer._w_packed_op
        del layer.qweight, layer.scales
        if qz is not None:
            del layer.qzeros

    def apply(
        self, layer: "BaseOP", x: torch.Tensor, bias: torch.Tensor | None
    ) -> torch.Tensor:
        if getattr(layer, "_w_rep_wide", None) is not None:
            out = kernels.w4a16_linear(
                x,
                layer._w_rep_wide,  # type: ignore[attr-defined]
                layer._scales_op,  # type: ignore[attr-defined]
                layer._zeros_op,  # type: ignore[attr-defined]
                self.quant.group_size,
                layer._n_out,  # type: ignore[attr-defined]
            )
        else:
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

    def apply_swiglu(self, layer: "BaseOP", x: torch.Tensor) -> torch.Tensor | None:
        """FUSED gate_up + silu_and_mul at decode (this linear is a merged gate_up). Returns None ->
        caller falls back to the unfused silu_and_mul(apply(...)) when the fused kernel doesn't apply:
        the wide-W4A16 path, prefill (M>16), or an unsupported shape. Bit-exact when it fires."""
        if getattr(layer, "_w_rep_wide", None) is not None:
            return None
        w = layer._w_packed_op  # type: ignore[attr-defined]
        if not _fused_swiglu_ok(x, w, self.quant.group_size):
            return None
        return kernels.w4a8_linear_silu(
            x, w, layer._scales_op, layer._zeros_op, self.quant.group_size  # type: ignore[attr-defined]
        ).to(x.dtype)


class MxFp4LinearMethod:
    """MXFP4 (OCP E2M1 weights + E8M0 per-32-block scale) dense linear, served through the SAME
    W4A8 fp8-WMMA kernel as int4 with `weight_is_e2m1=True` (per-token fp8 activations). The
    checkpoint (compressed-tensors `mxfp4-pack-quantized`) ships weights ALREADY in a compact
    packed form — weight_packed uint8 (N, K//2) 2 E2M1 nibbles/byte + weight_scale uint8 (N, K//32)
    E8M0 group exponent — so `process_weights_after_load` runs the MXFP4 converter (nibbles ->
    (N,K//8) int32 codes verbatim; E8M0 -> fp16 group scale) and drops the checkpoint copies.
    Symmetric (no zero-points). Config-selected purely from `quant.weight_is_e2m1`."""

    def __init__(self, quant: QuantConfig) -> None:
        self.quant = quant

    def create_weights(self, layer: "BaseOP", out_features: int, in_features: int) -> None:
        N, K = out_features, in_features
        g = self.quant.group_size  # 32 (OCP MX block)
        assert K % 2 == 0 and K % g == 0 and N % 8 == 0, (
            f"MXFP4 needs K%2==0,K%{g}==0,N%8==0; got N={N},K={K}"
        )
        # CHECKPOINT layout (uint8), so BaseOP load matches. E8M0 scale is an integer exponent, NOT
        # a float — declared uint8 so the engine's _cast leaves it untouched (see engine._cast).
        layer.weight_packed = torch.empty((N, K // 2), dtype=torch.uint8)
        layer.weight_scale = torch.empty((N, K // g), dtype=torch.uint8)

    def process_weights_after_load(self, layer: "BaseOP") -> None:
        from . import mxfp4

        conv = mxfp4.convert_mxfp4_weight(layer.weight_packed, layer.weight_scale)  # type: ignore[attr-defined]
        info = conv["scale_info"]
        if not info["fp16_range_ok"]:
            from minisgl.utils import init_logger

            init_logger("mxfp4").info_rank0(
                f"[mxfp4] E8M0 group scales exceed the fp16 store on "
                f"{getattr(layer, 'prefix', '<linear>')} (exp {info['exp_min']}..{info['exp_max']}, "
                f"{info['fp16_overflow_groups']} overflow / {info['e8m0_nan_groups']} e8m0-NaN "
                f"groups); an fp32 group-scale path may be needed for this checkpoint."
            )
        layer._w_packed_op = conv["w_packed"]  # (N, K//8) int32
        layer._scales_op = conv["scales"]  # (N, K//32) fp16
        del layer.weight_packed, layer.weight_scale

    def apply(
        self, layer: "BaseOP", x: torch.Tensor, bias: torch.Tensor | None
    ) -> torch.Tensor:
        out = kernels.w4a8_linear(
            x,
            layer._w_packed_op,  # type: ignore[attr-defined]
            layer._scales_op,  # type: ignore[attr-defined]
            None,  # symmetric — no zero-points
            self.quant.group_size,
            weight_is_e2m1=True,
        )
        out = out.to(x.dtype)
        if bias is not None:
            out = out + bias
        return out

    def apply_swiglu(self, layer: "BaseOP", x: torch.Tensor) -> torch.Tensor | None:
        """FUSED gate_up + silu (MXFP4 / E2M1, symmetric) at decode; None -> caller falls back."""
        w = layer._w_packed_op  # type: ignore[attr-defined]
        if not _fused_swiglu_ok(x, w, self.quant.group_size):
            return None
        return kernels.w4a8_linear_silu(
            x, w, layer._scales_op, None, self.quant.group_size, weight_is_e2m1=True  # type: ignore[attr-defined]
        ).to(x.dtype)


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
    if quant.is_nvfp4:
        # gfx1201 has no FP4 hardware -> upconvert NVFP4 to the fp8 W8A8 path at load (see NvFp4LinearMethod).
        return NvFp4LinearMethod(quant)
    if quant.is_rxf:
        return RXFLinearMethod(quant)
    # MXFP4 (compressed-tensors float-quantized 4-bit, OCP E2M1) -> the W4A8 kernel with the e2m1
    # decode. Config-selected from the DECLARED scheme (no model-name branch); disjoint from the int4
    # W4A8 path below (weight_type=="int") and the fp8 W8A8 path (bits==8).
    if quant.weight_is_e2m1:
        return MxFp4LinearMethod(quant)
    # fp8 W8A8 (compressed-tensors float-quantized 8-bit) dense linear — served through the SAME
    # W8A8 WMMA core as the fp8 MoE experts (single-expert grouped GEMM). ZAYA's dense/attn linears
    # stay in the quant `ignore` list (-> unquantized), so this only fires for a checkpoint that
    # actually declares fp8-W8A8 dense linears (e.g. RedHatAI *-FP8-dynamic).
    if quant.is_fp8_w8a8:
        return Fp8W8A8LinearMethod(quant)
    return W4A8LinearMethod(quant)


class NvFp4LinearMethod:
    """NVFP4 (compressed-tensors 'nvfp4-pack-quantized') dense linear, served through the SAME e2m1
    W4A8 kernel as MXFP4 — weights stay 4-bit; the E2M1 codes decode to fp8 e4m3 in-register at the
    WMMA (no VRAM upconvert). NVFP4 differs from MXFP4 only in the scale, which the WEIGHT LOADER folds
    to one fp16 per-group scale at the leaf (`nvfp4.fold_nvfp4_scale`: e4m3 block / per-tensor global),
    dropping the global tensors. So by the time this method loads, the checkpoint is MXFP4-shaped:
    weight_packed uint8 (N,K//2) 2 E2M1 nibbles/byte + weight_scale fp16 (N,K//16). `process_weights_
    after_load` packs the nibbles to (N,K//8) int32 codes (verbatim) and passes the fp16 scale through;
    `apply` calls the e2m1 kernel at group_size 16. Symmetric (no zero-points). From quant.is_nvfp4."""

    def __init__(self, quant: QuantConfig) -> None:
        self.quant = quant

    def create_weights(self, layer: "BaseOP", out_features: int, in_features: int) -> None:
        # POST-FOLD checkpoint layout (the loader already folded the scale to fp16 and dropped the
        # per-tensor globals): weight_packed uint8, weight_scale fp16 at group_size 16.
        N, K = out_features, in_features
        g = self.quant.group_size  # 16 (NVFP4 block)
        assert K % 2 == 0 and K % g == 0 and N % 8 == 0, (
            f"NVFP4 needs K%2==0,K%{g}==0,N%8==0; got N={N},K={K}"
        )
        layer.weight_packed = torch.empty((N, K // 2), dtype=torch.uint8)
        layer.weight_scale = torch.empty((N, K // g), dtype=torch.float16)

    def process_weights_after_load(self, layer: "BaseOP") -> None:
        from . import nvfp4

        conv = nvfp4.convert_nvfp4_weight(layer.weight_packed, layer.weight_scale)  # type: ignore[attr-defined]
        layer._w_packed_op = conv["w_packed"]  # (N, K//8) int32 E2M1 codes
        layer._scales_op = conv["scales"]  # (N, K//16) fp16 per-group
        del layer.weight_packed, layer.weight_scale

    def apply(
        self, layer: "BaseOP", x: torch.Tensor, bias: torch.Tensor | None
    ) -> torch.Tensor:
        out = kernels.w4a8_linear(
            x,
            layer._w_packed_op,  # type: ignore[attr-defined]
            layer._scales_op,  # type: ignore[attr-defined]
            None,  # symmetric — no zero-points
            self.quant.group_size,
            weight_is_e2m1=True,
        )
        out = out.to(x.dtype)
        if bias is not None:
            out = out + bias
        return out


class Fp8W8A8LinearMethod:
    """Dense fp8 W8A8 linear: per-output-channel fp8 (e4m3) weights + dynamic per-token fp8
    activations (the RedHatAI *-FP8-dynamic scheme: weights `strategy:channel`, activations
    `dynamic:token`). Served through kernels.w8a8_dense_linear — the same validated W8A8 WMMA core as
    the fp8 MoE experts, reused as a single-expert grouped GEMM (no dedicated dense kernel needed).
    Config-selected purely from `quant.is_fp8_w8a8` (float-quantized, 8-bit)."""

    def __init__(self, quant: QuantConfig) -> None:
        self.quant = quant

    def create_weights(self, layer: "BaseOP", out_features: int, in_features: int) -> None:
        # CHECKPOINT layout so the BaseOP loader lands tensors directly: fp8 e4m3 weight (N,K) +
        # per-output-channel scale (N,1). compressed-tensors fp8 ships the scale fp16 (some heads
        # bf16); declare fp16 like the CT W4A8 path — the engine._cast normalizes both to fp16 —
        # then process_weights_after_load promotes it to the f32 the kernel ABI wants. (N=out, K=in
        # are the LOCAL/per-TP sizes.)
        N, K = out_features, in_features
        layer.weight = torch.empty((N, K), dtype=torch.float8_e4m3fn)
        layer.weight_scale = torch.empty((N, 1), dtype=torch.float16)

    def process_weights_after_load(self, layer: "BaseOP") -> None:
        # op layout == natural row-major e4m3 (the GEMM indexes [n*K+k]); the kernel takes the e4m3
        # bytes as uint8 (zero-copy bitcast preserving the bit pattern) + a flat (N,) f32 channel
        # scale — the same op layout _GroupedFP8Experts.post_load builds, minus the E dim.
        layer._w_op = layer.weight.contiguous().view(torch.uint8)  # type: ignore[attr-defined]
        layer._scales_op = layer.weight_scale.squeeze(-1).contiguous().float()  # (N,1)->(N,)
        del layer.weight, layer.weight_scale

    def apply(
        self, layer: "BaseOP", x: torch.Tensor, bias: torch.Tensor | None
    ) -> torch.Tensor:
        out = kernels.w8a8_dense_linear(
            x, layer._w_op, layer._scales_op  # type: ignore[attr-defined]
        )
        if bias is not None:
            out = out + bias
        return out
