from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
from minisgl.core import get_global_ctx
from minisgl.distributed import (
    DistributedCommunicator,
    get_dp_info,
    get_tp_info,
    is_ep_enabled,
)
from minisgl.utils import div_even

from .base import BaseOP

if TYPE_CHECKING:
    from minisgl.quant.config import QuantConfig


class _GroupedGPTQExperts(BaseOP):
    """Per-expert grouped GPTQ buffers for one of the two MoE GEMMs (w13 or w2).

    Declared in CHECKPOINT layout, STACKED over the E experts so the streaming loader's
    merge->stack path lands tensors here directly (keys `<...>.{qweight,scales,qzeros}`):
        qweight (E, K//pf, N) i32   scales (E, K//g, N) f16   qzeros (E, K//g, N//pf) i32
    where N=out, K=in are per-expert. `post_load` converts each expert to the op's native
    grouped layout (the same gptq_to_op_layout used for dense linears) and stacks:
        _w_op (E, N, K//pf) i32   _scales_op (E, N, K//g) f16   _zeros_op (E, N//pf, K//g) i32
    which is exactly what `kernels.w4a8_moe` consumes for w13 / w2."""

    def __init__(self, num_experts: int, out_features: int, in_features: int, quant: "QuantConfig"):
        pf = 32 // quant.bits
        g = quant.group_size
        N, K = out_features, in_features
        assert K % pf == 0 and K % g == 0 and N % pf == 0, (
            f"grouped GPTQ needs K%{pf}==0,K%{g}==0,N%{pf}==0; got N={N},K={K}"
        )
        self.qweight = torch.empty((num_experts, K // pf, N), dtype=torch.int32)
        self.scales = torch.empty((num_experts, K // g, N), dtype=torch.float16)
        self.qzeros = torch.empty((num_experts, K // g, N // pf), dtype=torch.int32)
        self._quant = quant

    def forward(self, *args, **kwargs):  # pragma: no cover - storage container, never called
        raise RuntimeError("_GroupedGPTQExperts holds weights; call kernels.w4a8_moe instead")

    def post_load(self) -> None:
        from minisgl.quant import kernels

        assert not self._quant.desc_act, "GPTQ desc_act (act-order) not supported"
        E = self.qweight.shape[0]
        w_op, s_op, z_op = [], [], []
        for e in range(E):
            w, s, z = kernels.gptq_to_op_layout(
                self.qweight[e], self.scales[e], self.qzeros[e], bits=self._quant.bits
            )
            w_op.append(w)
            s_op.append(s)
            z_op.append(z)
        self._w_op = torch.stack(w_op, dim=0)
        self._scales_op = torch.stack(s_op, dim=0)
        self._zeros_op = torch.stack(z_op, dim=0)
        del self.qweight, self.scales, self.qzeros


class _GroupedAWQExperts(BaseOP):
    """AWQ-gemm experts for one MoE GEMM (w13 or w2), STACKED over E (checkpoint layout).

    AWQ packs int4 along the OUTPUT N with the GEMM interleave: qweight (E, K, N//pf) int32,
    per-group scales (E, K//g, N), qzeros (E, K//g, N//pf) (AWQ is asymmetric -> zeros ALWAYS
    present). `post_load` runs the proven `awq_to_op_layout` per expert (undo interleave,
    transpose, repack along K) and stacks to the op's grouped layout
    (_w_op (E, N, K//pf), _scales_op (E, N, K//g), _zeros_op (E, N//pf, K//g)). N=out, K=in."""

    def __init__(self, num_experts: int, out_features: int, in_features: int, quant: "QuantConfig"):
        pf = 32 // quant.bits
        g = quant.group_size
        N, K = out_features, in_features
        assert N % pf == 0 and K % g == 0, f"AWQ experts need N%{pf}==0,K%{g}==0; got N={N},K={K}"
        self.qweight = torch.empty((num_experts, K, N // pf), dtype=torch.int32)
        self.scales = torch.empty((num_experts, K // g, N), dtype=torch.float16)
        self.qzeros = torch.empty((num_experts, K // g, N // pf), dtype=torch.int32)
        self._quant = quant

    def forward(self, *args, **kwargs):  # pragma: no cover - storage container, never called
        raise RuntimeError("_GroupedAWQExperts holds weights; call kernels.w4a8_moe instead")

    def post_load(self) -> None:
        from minisgl.quant import kernels

        E = self.qweight.shape[0]
        w_op, s_op, z_op = [], [], []
        for e in range(E):
            w, s, z = kernels.awq_to_op_layout(
                self.qweight[e], self.scales[e], self.qzeros[e], bits=self._quant.bits
            )
            w_op.append(w)
            s_op.append(s)
            z_op.append(z)
        self._w_op = torch.stack(w_op, dim=0)
        self._scales_op = torch.stack(s_op, dim=0)
        self._zeros_op = torch.stack(z_op, dim=0)
        del self.qweight, self.scales, self.qzeros


class _GroupedRXFExperts(BaseOP):
    """RXF W4(NL)-A8 experts for one MoE GEMM (w13 or w2), STACKED over E.

    RXF ships op-layout already (no AWQ/GPTQ unpack-transpose-repack), so these buffers are
    loaded as-is and need no post_load: weight_packed (E, N, K/2) uint8 NL indices, weight_scale
    (E, N, K/32) fp16 per-group scale, group=32, symmetric NL codebook (no zero-points). N=out,
    K=in per expert. Consumed by kernels.rxf_moe."""

    def __init__(self, num_experts: int, out_features: int, in_features: int, quant: "QuantConfig"):
        N, K = out_features, in_features
        span = quant.rotation_span
        assert K % 32 == 0 and K % span == 0 and K % 2 == 0, (
            f"grouped RXF needs K%32==0,K%span({span})==0,K%2==0; got N={N},K={K}"
        )
        self.weight_packed = torch.empty((num_experts, N, K // 2), dtype=torch.uint8)
        self.weight_scale = torch.empty((num_experts, N, K // 32), dtype=torch.float16)
        self._quant = quant

    def forward(self, *args, **kwargs):  # pragma: no cover - storage container, never called
        raise RuntimeError("_GroupedRXFExperts holds weights; call kernels.rxf_moe instead")


class _GroupedCompressedTensorsExperts(BaseOP):
    """compressed-tensors int4 *weight-only* (W4A16) experts for one MoE GEMM (w13 or w2), STACKED
    over E. The format `cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit` ships (despite "AWQ" in its name):
        weight_packed (E, N, K//pf) int32 — 8 SIGNED int4 per int32, packed along INPUT K in natural
            order (K-index k -> column k//pf, nibble k%pf); symmetric, so NO zero-points.
        weight_scale  (E, N, K//g)  bf16  — per-(output-row, input-group) scale, group g=32.
    (N=out, K=in per expert.) This is structurally the op's grouped `_w_op (E,N,K//pf)` /
    `_scales_op (E,N,K//g)` layout ALREADY (same natural nibble order as gptq_to_op_layout's output),
    so `post_load` is a cheap whole-tensor fixup rather than a per-expert unpack/transpose:
      * signed int4 -> the kernel's `w = scale*(q_unsigned - zero)` convention by flipping each
        nibble's top bit (XOR 0x8) and using a CONSTANT zero-point of 8: for every nibble value
        `(n ^ 8) - 8 == signed_int4(n)` exactly (n<8 -> n, else n-16). XOR 0x8 per nibble == XOR 0x88
        per byte, done via a uint8 view (no int32 overflow).
      * zeros_op is all-8 (every nibble 8 -> every int32 0x88888888), shape (E, N//pf, K//g).
    Then `kernels.w4a8_moe` consumes `_w_op/_scales_op/_zeros_op` exactly as for GPTQ/AWQ (the
    activations are quantized to int8 by that kernel — same W4A16-weights-through-W4A8-kernel path the
    AWQ experts already use)."""

    def __init__(self, num_experts: int, out_features: int, in_features: int, quant: "QuantConfig"):
        pf = 32 // quant.bits  # 8
        g = quant.group_size  # 32
        N, K = out_features, in_features
        assert K % pf == 0 and K % g == 0 and N % pf == 0, (
            f"grouped compressed-tensors needs K%{pf}==0,K%{g}==0,N%{pf}==0; got N={N},K={K}"
        )
        self.weight_packed = torch.empty((num_experts, N, K // pf), dtype=torch.int32)
        self.weight_scale = torch.empty((num_experts, N, K // g), dtype=torch.bfloat16)
        self._quant = quant

    def forward(self, *args, **kwargs):  # pragma: no cover - storage container, never called
        raise RuntimeError("_GroupedCompressedTensorsExperts holds weights; call kernels.w4a8_moe")

    def post_load(self) -> None:
        pf = 32 // self._quant.bits
        E, N, Kp = self.weight_packed.shape
        G = self.weight_scale.shape[-1]
        # signed int4 -> unsigned (q+8) by flipping each nibble's top bit (XOR 0x88 per byte).
        flipped = (self.weight_packed.contiguous().view(torch.uint8) ^ 0x88).view(torch.int32)
        self._w_op = flipped.contiguous()
        self._scales_op = self.weight_scale.to(torch.float16).contiguous()
        # symmetric zero-point == 8 for every (output, group): every packed nibble 8 -> 0x88888888.
        zeros = torch.empty((E, N // pf, G), dtype=torch.int32)
        zeros.view(torch.uint8).fill_(0x88)
        self._zeros_op = zeros.to(self.weight_packed.device)
        del self.weight_packed, self.weight_scale


class _GroupedMxFp4Experts(BaseOP):
    """MXFP4 (OCP E2M1 weights + E8M0 per-32-block scale) experts for one MoE GEMM (w13 or w2),
    STACKED over E. The compressed-tensors `mxfp4-pack-quantized` checkpoint ships (per expert,
    merged gate|up into w13 / down into w2 by the loader):
        weight_packed (E, N, K//2) uint8 — 2 E2M1 nibbles/byte, low nibble = lower K index.
        weight_scale  (E, N, K//32) uint8 — E8M0, one shared exponent per 32-element block.
    (N=out, K=in per expert.) `post_load` runs the MXFP4 converter (nibbles -> (E,N,K//8) int32 codes
    verbatim; E8M0 -> fp16 group scale) so `kernels.w4a8_moe(..., weight_is_e2m1=True)` consumes
    `_w_op/_scales_op` exactly as the int4 experts do. Symmetric -> no zero-points."""

    def __init__(self, num_experts: int, out_features: int, in_features: int, quant: "QuantConfig"):
        g = quant.group_size  # 32
        N, K = out_features, in_features
        assert K % 2 == 0 and K % g == 0 and N % 8 == 0, (
            f"grouped MXFP4 needs K%2==0,K%{g}==0,N%8==0; got N={N},K={K}"
        )
        # CHECKPOINT layout (uint8); E8M0 scale is an integer exponent, NOT a float, so the engine's
        # _cast leaves both uint8 buffers untouched.
        self.weight_packed = torch.empty((num_experts, N, K // 2), dtype=torch.uint8)
        self.weight_scale = torch.empty((num_experts, N, K // g), dtype=torch.uint8)
        self._quant = quant

    def forward(self, *args, **kwargs):  # pragma: no cover - storage container, never called
        raise RuntimeError("_GroupedMxFp4Experts holds weights; call kernels.w4a8_moe instead")

    def post_load(self) -> None:
        from minisgl.quant import mxfp4

        conv = mxfp4.convert_mxfp4_moe(self.weight_packed, self.weight_scale)
        info = conv["scale_info"]
        if not info["fp16_range_ok"]:
            from minisgl.utils import init_logger

            init_logger("mxfp4").info_rank0(
                f"[mxfp4-moe] E8M0 group scales exceed the fp16 store "
                f"(exp {info['exp_min']}..{info['exp_max']}, {info['fp16_overflow_groups']} overflow "
                f"/ {info['e8m0_nan_groups']} e8m0-NaN groups); an fp32 group-scale path may be needed."
            )
        self._w_op = conv["w_packed"]  # (E, N, K//8) int32
        self._scales_op = conv["scales"]  # (E, N, K//32) fp16
        del self.weight_packed, self.weight_scale


class _GroupedFP8Experts(BaseOP):
    """Weight-only fp8 (F8_E4M3) experts for one MoE GEMM (w13 or w2), STACKED over E.

    ZAYA's experts are compressed-tensors *float-quant*: each expert weight is F8_E4M3 with a
    per-output-channel (dim-0) F32 `weight_scale`. Storing the raw fp8 (~8 GB total) instead of
    dequantizing to bf16 (~16 GB) is what lets the 8B fit a single 16 GB gfx1201 card. At compute the
    fp8 weights feed the native W8A8 grouped-MoE kernel DIRECTLY (no dequant): `post_load` bitcasts
    them to the kernel's uint8 op layout (`_w_op`/`_scales_op`) and drops the checkpoint copies.
    `dequant()` (the full fp8->bf16 stack) is the gated A/B reference ONLY (`MINISGL_ZAYA_OLDMOE=1`) —
    it re-materializes the WHOLE (E,N,K) bf16 stack per GEMM per forward, so it is decidedly NOT the
    hot path. Buffers are declared in CHECKPOINT dtype/shape so the BaseOP loader's dtype assertion
    passes and the streaming stack path lands tensors here directly:
        weight (E, N, K) f8_e4m3   weight_scale (E, N, 1) f32     (N=out, K=in per expert)."""

    def __init__(self, num_experts: int, out_features: int, in_features: int):
        N, K = out_features, in_features
        self.weight = torch.empty((num_experts, N, K), dtype=torch.float8_e4m3fn)
        self.weight_scale = torch.empty((num_experts, N, 1), dtype=torch.float32)

    def forward(self, *args, **kwargs):  # pragma: no cover - storage container, never called
        raise RuntimeError("_GroupedFP8Experts holds weights; dequant per-expert at compute")

    def dequant(self, dtype: torch.dtype) -> torch.Tensor:
        """A/B-reference ONLY (`MINISGL_ZAYA_OLDMOE=1`): dequantize ALL experts to `dtype` -> (E,N,K).
        This materializes the full bf16 stack (every expert, both GEMMs) per forward — the transient
        the fp8 storage scheme exists to avoid. The default path never calls this (native W8A8 kernel
        consumes `_w_op`/`_scales_op` directly).

        VECTORIZED (2026-07-04): one whole-tensor dequant instead of a Python per-expert loop+stack.
        The old `torch.stack([... for e in range(E)])` issued ~3E tiny kernels PER dequant × 2 GEMMs ×
        40 layers = the ~9,300-launch/step op-flood that made the fused OLDMOE step 284ms (vs 55ms
        native fp8); this collapses it to 3 ops. `weight_scale` (E,N,1) broadcasts over `weight` (E,N,K)
        exactly as the per-expert `[e]` slices did — bit-identical result."""
        return (self.weight.float() * self.weight_scale).to(dtype)

    def post_load(self) -> None:
        """Build the native W8A8 kernel's op-layout buffers and drop the checkpoint copies.

        The op layout IS the natural (E, N, K) f8_e4m3 row-major checkpoint layout (the GEMM reads
        `w_fp8 + e*N*K` and indexes `[n*K + k]`), so `_w_op` is just a contiguous view; the kernel
        wants the per-output-channel scale as a flat (E, N) f32. Underscore-prefixed so the BaseOP
        state walk skips them. The kernel binding takes the e4m3 bytes as a uint8 tensor (it copies
        them straight to LDS for the fp8 WMMA intrinsic), so reinterpret the f8_e4m3 storage as uint8
        — a zero-copy bitcast that preserves the exact e4m3 bit pattern."""
        self._w_op = self.weight.contiguous().view(torch.uint8)
        self._scales_op = self.weight_scale.squeeze(-1).contiguous().float()  # (E, N, 1) -> (E, N)
        # A/B toggle: MINISGL_ZAYA_OLDMOE=1 keeps the checkpoint fp8 weights for the legacy
        # dequant->Triton path (forward branch). Default drops them (native W8A8 kernel only).
        if os.environ.get("MINISGL_ZAYA_OLDMOE", "0") == "0":
            del self.weight, self.weight_scale


# =====================================================================================
# Config-driven MoE-expert quant selector (mirrors quant.method.create_linear_method for the
# dense linears and the GDN in_proj dispatch). ALL three module families now pick their scheme +
# kernel from the SAME declared QuantConfig the same way — no scattered per-scheme `if`s in the
# layer, no env var that substitutes a different scheme than the checkpoint declares, no model-name
# branches. A `MoEQuantMethod` owns one scheme family: which per-expert weight CONTAINER to
# allocate (__init__) and which grouped kernel to run (forward, both the plain TP path and the
# per-rank EP shard). `create_moe_quant_method` maps the config to the subclass.
# =====================================================================================
class MoEQuantMethod:
    """How a MoE expert GEMM pair (w13 gate|up, w2 down) allocates its weights and runs its matmul.
    The MoELayer owns routing, EP dispatch/combine and the TP all-reduce; the method owns the
    per-expert weight layout + the grouped GEMM."""

    supports_ep: bool = False  # can this scheme run the EP all_gather/mask/all_reduce shard path?
    needs_precomputed_route: bool = False  # True -> forward MUST be handed topk_weights/topk_ids

    def create_experts(self, num_experts: int, out_features: int, in_features: int):
        """Allocate the per-expert weight container for ONE GEMM (STACKED over `num_experts` on
        dim 0; caller passes the LOCAL count under EP). Returns a BaseOP container (quantized) or a
        plain stacked bf16/fp16 tensor (unquantized)."""
        raise NotImplementedError

    def apply(
        self, w13, w2, hidden_states, *, router_logits, topk_weights, topk_ids,
        top_k: int, renormalize: bool, activation: str, apply_router_weight_on_input: bool,
    ) -> "torch.Tensor":
        """Plain (non-EP) forward over the full replicated expert stack."""
        raise NotImplementedError

    def ep_local(
        self, w13, w2, g_hidden, local_weights, local_ids, *, top_k: int, renormalize: bool
    ) -> "torch.Tensor":
        """EP per-rank shard kernel: run THIS rank's local expert stack (w13/w2 already the
        [E_local,...] shard) over the all_gather'd tokens with local-remapped ids/weights."""
        raise NotImplementedError(f"{type(self).__name__} does not support expert parallelism")


class _UnquantizedMoEMethod(MoEQuantMethod):
    """bf16/fp16 stacked experts — the fused moe_backend (or the precomputed-route stacked kernel)."""

    def create_experts(self, num_experts, out_features, in_features):
        return torch.empty(num_experts, out_features, in_features)

    def apply(self, w13, w2, hidden_states, *, router_logits, topk_weights, topk_ids,
              top_k, renormalize, activation, apply_router_weight_on_input):
        if topk_ids is not None:
            # Route computed in the model (Zaya top-1 + MOD, GLM noaux_tc). The moe_backend fuses
            # softmax+topk internally so it can't take a precomputed route — call the stacked kernel.
            from minisgl.moe.fused import fused_experts_impl

            return fused_experts_impl(
                hidden_states, w13, w2, topk_weights, topk_ids,
                activation=activation, apply_router_weight_on_input=apply_router_weight_on_input,
            )
        ctx = get_global_ctx()
        return ctx.moe_backend.forward(
            hidden_states=hidden_states, w1=w13, w2=w2, gating_output=router_logits,
            topk=top_k, renormalize=renormalize, activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )


class _W4A8MoEMethod(MoEQuantMethod):
    """int4-weight grouped experts through the shared `kernels.w4a8_moe` (int4 weight x per-token fp8
    act). Covers GPTQ (K-major qweight), AWQ-gemm (N-major, interleaved, asymmetric) and
    compressed-tensors int4 (W4A16 weights served through the same W4A8 kernel) — they differ only in
    CHECKPOINT layout (the container's post_load converts each to the op's grouped triple)."""

    supports_ep = True

    def __init__(self, quant: "QuantConfig"):
        self._quant = quant
        if quant.is_gptq:
            self._cls = _GroupedGPTQExperts
        elif quant.is_awq:
            self._cls = _GroupedAWQExperts
        elif quant.is_compressed_tensors:
            self._cls = _GroupedCompressedTensorsExperts
        else:
            raise AssertionError(f"W4A8 MoE unsupported quant method: {quant.method}")

    def create_experts(self, num_experts, out_features, in_features):
        return self._cls(num_experts, out_features, in_features, self._quant)

    def apply(self, w13, w2, hidden_states, *, router_logits, topk_weights, topk_ids,
              top_k, renormalize, activation, apply_router_weight_on_input):
        assert activation == "silu" and not apply_router_weight_on_input, (
            "MoE W4A8 path is silu-only without router-weight-on-input"
        )
        from minisgl.quant import kernels

        return kernels.w4a8_moe(
            hidden_states, w13._w_op, w13._scales_op, w13._zeros_op,
            w2._w_op, w2._scales_op, w2._zeros_op,
            router_logits, top_k, renormalize, topk_weights=topk_weights, topk_ids=topk_ids,
        )

    def ep_local(self, w13, w2, g_hidden, local_weights, local_ids, *, top_k, renormalize):
        from minisgl.quant import kernels

        return kernels.w4a8_moe(
            g_hidden, w13._w_op, w13._scales_op, w13._zeros_op,
            w2._w_op, w2._scales_op, w2._zeros_op,
            None, top_k, renormalize, topk_weights=local_weights, topk_ids=local_ids,
        )


class _MxFp4MoEMethod(MoEQuantMethod):
    """MXFP4 (OCP E2M1) grouped experts through the shared `kernels.w4a8_moe` with
    `weight_is_e2m1=True` — the same W4A8 fp8-WMMA kernel the int4 experts use, only a different
    4-bit decode table + E8M0->fp16 group scale (done at load in `_GroupedMxFp4Experts.post_load`).
    Symmetric, so no zero-points (None). EP-capable exactly like the int4 W4A8 path (E on dim 0 of
    every expert buffer; a shard is a pure dim-0 slice)."""

    supports_ep = True

    def __init__(self, quant: "QuantConfig"):
        self._quant = quant

    def create_experts(self, num_experts, out_features, in_features):
        return _GroupedMxFp4Experts(num_experts, out_features, in_features, self._quant)

    def apply(self, w13, w2, hidden_states, *, router_logits, topk_weights, topk_ids,
              top_k, renormalize, activation, apply_router_weight_on_input):
        assert activation == "silu" and not apply_router_weight_on_input, (
            "MoE MXFP4 path is silu-only without router-weight-on-input"
        )
        from minisgl.quant import kernels

        return kernels.w4a8_moe(
            hidden_states, w13._w_op, w13._scales_op, None,
            w2._w_op, w2._scales_op, None,
            router_logits, top_k, renormalize, topk_weights=topk_weights, topk_ids=topk_ids,
            weight_is_e2m1=True,
        )

    def ep_local(self, w13, w2, g_hidden, local_weights, local_ids, *, top_k, renormalize):
        from minisgl.quant import kernels

        return kernels.w4a8_moe(
            g_hidden, w13._w_op, w13._scales_op, None,
            w2._w_op, w2._scales_op, None,
            None, top_k, renormalize, topk_weights=local_weights, topk_ids=local_ids,
            weight_is_e2m1=True,
        )


class _RXFMoEMethod(MoEQuantMethod):
    """RXF W4(NL)-A8 grouped experts (`kernels.rxf_moe`). No EP path (RXF has no precomputed-topk
    shard route, which EP requires) — stays replicated."""

    supports_ep = False

    def __init__(self, quant: "QuantConfig"):
        self._quant = quant

    def create_experts(self, num_experts, out_features, in_features):
        return _GroupedRXFExperts(num_experts, out_features, in_features, self._quant)

    def apply(self, w13, w2, hidden_states, *, router_logits, topk_weights, topk_ids,
              top_k, renormalize, activation, apply_router_weight_on_input):
        assert activation == "silu" and not apply_router_weight_on_input, (
            "MoE RXF path is silu-only without router-weight-on-input"
        )
        from minisgl.quant import kernels

        return kernels.rxf_moe(
            hidden_states, w13.weight_packed, w13.weight_scale, w2.weight_packed, w2.weight_scale,
            router_logits, top_k, renormalize, topk_weights=topk_weights, topk_ids=topk_ids,
            span=self._quant.rotation_span,
        )


class _FP8MoEMethod(MoEQuantMethod):
    """fp8 W8A8 experts (ZAYA): F8_E4M3 weights + a per-output-channel f32 scale, fed to the native
    `kernels.w8a8_moe` fp8-WMMA kernel with per-token fp8 activations — the CHECKPOINT-DECLARED
    scheme, and the DEFAULT. Two opt-outs, both env-gated PERF toggles (never a scheme substitution
    picked silently): MINISGL_ZAYA_W8A16=1 -> fp8 weights dequantized in-register to bf16 acts
    (W8A16, quality/latency trade); MINISGL_ZAYA_OLDMOE=1 -> legacy fp8->bf16 dequant->fused Triton
    A/B reference. W8A16 used to be the default (an env silently swapping the declared act scheme) —
    that was a bug; it is now an explicit opt-in."""

    supports_ep = True
    needs_precomputed_route = True  # ZAYA top-1 + MOD route is computed model-side

    def __init__(self):
        self._oldmoe = os.environ.get("MINISGL_ZAYA_OLDMOE", "0") == "1"
        # W8A16 is now OPT-IN (=1); default is the checkpoint-declared native W8A8 fp8-act kernel.
        self._w8a16_fn = None
        if not self._oldmoe and os.environ.get("MINISGL_ZAYA_W8A16", "0") == "1":
            try:  # fail-safe: an env without the built extension falls back to native W8A8
                from moe_w8a16_wmma import fused_moe_w8a16

                self._w8a16_fn = fused_moe_w8a16
            except ImportError:
                self._w8a16_fn = None

    def create_experts(self, num_experts, out_features, in_features):
        return _GroupedFP8Experts(num_experts, out_features, in_features)

    def apply(self, w13, w2, hidden_states, *, router_logits, topk_weights, topk_ids,
              top_k, renormalize, activation, apply_router_weight_on_input):
        assert topk_ids is not None, "fp8 experts use the precomputed-route path (ZAYA top-1 + MOD)"
        from minisgl.quant import kernels

        if self._w8a16_fn is not None:
            # W8A16 opt-in: dequant the fp8 weight tile to bf16 IN-REGISTER (no full-stack
            # materialize), routed experts only; bf16 acts. Uses the always-present op-layout buffers.
            # The kernel requires bf16 activations, so guard the cast (no-op when the model is already
            # bf16 — the norm for W8A16 checkpoints) instead of hardcoding an unconditional conversion.
            acts = hidden_states if hidden_states.dtype == torch.bfloat16 else hidden_states.to(torch.bfloat16)
            return self._w8a16_fn(
                acts,
                w13._w_op, w13._scales_op, w2._w_op, w2._scales_op,
                topk_weights, topk_ids.to(torch.int32),
            )
        if self._oldmoe:
            # A/B reference: legacy fp8->bf16-dequant->Triton path (weights kept in post_load).
            from minisgl.moe.fused import fused_experts_impl

            return fused_experts_impl(
                hidden_states, w13.dequant(hidden_states.dtype), w2.dequant(hidden_states.dtype),
                topk_weights, topk_ids,
                activation=activation, apply_router_weight_on_input=apply_router_weight_on_input,
            )
        return kernels.w8a8_moe(
            hidden_states, w13._w_op, w13._scales_op, w2._w_op, w2._scales_op,
            None, top_k, renormalize, topk_weights=topk_weights, topk_ids=topk_ids,
        )

    def ep_local(self, w13, w2, g_hidden, local_weights, local_ids, *, top_k, renormalize):
        from minisgl.quant import kernels

        if self._w8a16_fn is not None:
            # Kernel requires bf16 acts; guard the cast (no-op when already bf16) — see forward().
            acts = g_hidden if g_hidden.dtype == torch.bfloat16 else g_hidden.to(torch.bfloat16)
            return self._w8a16_fn(
                acts,
                w13._w_op, w13._scales_op, w2._w_op, w2._scales_op, local_weights, local_ids,
            )
        return kernels.w8a8_moe(
            g_hidden, w13._w_op, w13._scales_op, w2._w_op, w2._scales_op,
            None, top_k, renormalize, topk_weights=local_weights, topk_ids=local_ids,
        )


def create_moe_quant_method(
    quant: "QuantConfig | None", *, fp8_experts: bool = False
) -> MoEQuantMethod:
    """Pick the MoE-expert quant method from the checkpoint's DECLARED scheme (the analogue of
    quant.method.create_linear_method for dense linears). Route:
      * fp8 W8A8 (compressed-tensors float-quantized 8-bit, or the explicit `fp8_experts` signal) ->
        native w8a8_moe (per-token fp8 acts; W8A16 is an env opt-in, never the default);
      * RXF -> rxf_moe;
      * MXFP4 (compressed-tensors float-quantized 4-bit, OCP E2M1) -> the shared w4a8_moe kernel with
        weight_is_e2m1=True (same kernel, e2m1 decode + E8M0->fp16 group scale);
      * int4 AWQ / GPTQ / compressed-tensors int4 -> the shared w4a8_moe kernel;
      * no quant -> the unquantized fused backend.
    Selection is purely config-driven: no model-name branch, and no env that substitutes a different
    scheme than the checkpoint declares."""
    if fp8_experts or (quant is not None and quant.is_fp8_w8a8):
        return _FP8MoEMethod()
    if quant is None:
        return _UnquantizedMoEMethod()
    if quant.is_rxf:
        return _RXFMoEMethod(quant)
    if quant.weight_is_e2m1:
        return _MxFp4MoEMethod(quant)
    if quant.is_int4:
        return _W4A8MoEMethod(quant)
    raise AssertionError(
        f"MoE: unsupported declared quant scheme (method={quant.method}, bits={quant.bits}, "
        f"weight_type={quant.weight_type})"
    )


class MoELayer(BaseOP):
    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        renormalize: bool = True,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        quant: "QuantConfig | None" = None,
        fp8_experts: bool = False,
        force_no_ep: bool = False,
    ):
        super().__init__()

        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self._comm = DistributedCommunicator()

        tp_info = get_tp_info()
        self.tp_size = tp_size = tp_info.size
        # Expert parallelism: each DP replica OWNS only experts [dp_rank*E/dp : (dp_rank+1)*E/dp], so
        # the per-expert weight buffers are sized to the LOCAL count (E/dp); the dispatch/combine
        # collective in forward() reconstructs the full result. enable_ep is the process-global toggle
        # (set by the Engine before model build); off => full replicated count. Every expert buffer
        # (fp8 AND quantized: _w_op/_scales_op/_zeros_op / weight_packed) puts E on dim 0, and every
        # grouped kernel reads E = w13.shape[0], so a shard is a pure dim-0 slice — EP is quant-agnostic
        # for the W4A8 (GPTQ/AWQ) and W4A16 (compressed-tensors) op layouts (shared w4a8_moe kernel) and
        # the fp8 W8A8/W8A16 layouts. RXF is excluded (no precomputed-topk path, which EP requires).
        # `force_no_ep` keeps a specific layer replicated even under EP — used for the tiny MTP draft
        # head, whose EP-sharding would make spec-decode propose issue data-dependent collectives.
        # Config-driven expert quant method (mirrors create_linear_method for the dense linears): it
        # owns the container class + the grouped kernel, and declares whether the scheme can run EP.
        self._moe_method = create_moe_quant_method(quant, fp8_experts=fp8_experts)
        self.enable_ep = (
            is_ep_enabled() and self._moe_method.supports_ep and not force_no_ep
        )
        dp_info = get_dp_info()
        self.ep_dp_rank = dp_info.dp_rank
        self.ep_dp_size = dp_info.dp_size
        if self.enable_ep:
            assert num_experts % dp_info.dp_size == 0, (
                f"EP needs num_experts ({num_experts}) divisible by dp_size ({dp_info.dp_size})"
            )
            self.local_num_experts = num_experts // dp_info.dp_size
            self.local_expert_offset = dp_info.dp_rank * self.local_num_experts
        else:
            self.local_num_experts = num_experts
            self.local_expert_offset = 0
        self.renormalize = renormalize
        self.activation = activation
        self.apply_router_weight_on_input = apply_router_weight_on_input
        self.quant = quant
        self.fp8_experts = fp8_experts
        intermediate_size_per_partition = div_even(intermediate_size, tp_size)
        # The method allocates the per-expert container for each GEMM (config-driven: fp8 F8_E4M3,
        # int4 W4A8/W4A16 grouped, RXF NL, or a plain stacked bf16/fp16 tensor). EP: size to the LOCAL
        # expert shard (E/dp) so this replica loads + runs only its experts; the streaming loader skips
        # non-local ids (weight.py mirror). enable_ep off (unquantized, RXF, force_no_ep) =>
        # local_num_experts == num_experts (full replicated). Both GEMMs, w13 = gate|up (2*inter), w2 =
        # down (hidden), share the method and the silu_and_mul convention.
        self.gate_up_proj = self._moe_method.create_experts(
            self.local_num_experts, 2 * intermediate_size_per_partition, hidden_size
        )
        self.down_proj = self._moe_method.create_experts(
            self.local_num_experts, hidden_size, intermediate_size_per_partition
        )

    def _ep_route(
        self,
        router_logits: "torch.Tensor | None",
        topk_weights: "torch.Tensor | None",
        topk_ids: "torch.Tensor | None",
    ):
        """Return (topk_weights f32, topk_ids i32) for the EP dispatch. EP must all_gather a route, so
        it can't defer to the kernel's fused softmax+topk — precompute it here (matching the kernel's
        own torch route: softmax -> top_k -> optional renorm, kernels.py w4a8_moe._route). A model that
        already provides a route (GLM/DeepSeek noaux_tc) passes it through unchanged; renormalize must
        happen over ALL top_k here, BEFORE the per-rank local-expert masking in _ep_dispatch."""
        if topk_ids is not None:
            assert topk_weights is not None, "topk_weights required when topk_ids is given"
            return topk_weights.to(torch.float32), topk_ids.to(torch.int32)
        assert router_logits is not None, "EP route needs router_logits or a precomputed topk"
        probs = torch.softmax(router_logits.float(), dim=-1)
        tw, ti = torch.topk(probs, self.top_k, dim=-1)
        if self.renormalize:
            tw = tw / (tw.sum(dim=-1, keepdim=True) + 1e-20)
        return tw.contiguous(), ti.to(torch.int32).contiguous()

    def _ep_dispatch(self, hidden_states, topk_weights, topk_ids, local_kernel):
        """Expert-parallel dispatch/combine, quant-agnostic. all_gather every replica's token rows +
        route so each rank sees ALL tokens -> remap global expert id to local (mask non-local by zeroing
        the route weight) -> run the format-specific grouped kernel over THIS rank's expert shard ->
        all_reduce(SUM) (each token's k experts each live on exactly one rank, so the sum reconstructs
        the full top-k result) -> slice our rows. `local_kernel(g_hidden, local_weights, local_ids_i32)
        -> (dp*N, H)` is the only per-format part (fp8 W8A16/W8A8, W4A8/W4A16 w4a8_moe). Replaces the TP
        all_reduce epilogue. See MoELayer.forward for the closures; N self-coordination is Part A."""
        ep = get_global_ctx().ep
        assert ep is not None, "EP enabled but ctx.ep group not built"
        real_n = hidden_states.shape[0]
        ep_w = topk_weights.contiguous()
        ep_i = topk_ids.to(torch.int32).contiguous()
        hs = hidden_states
        # Common token count N every replica pads to before the all_gather (RCCL needs equal shapes):
        #  1. graph decode: pre-padded to a captured bs -> already equal, no host sync.
        #  2. eager with a scheduler pre-agreement (ep_loop prefill): ep.pad_tokens.
        #  3. eager, no pre-agreement (spec-verify / prefix-seed): SELF-COORDINATE via one tiny
        #     all_gather of real_n. This is what lets EP survive spec-decode (Part A).
        if torch.cuda.is_current_stream_capturing():
            common_n = real_n
        elif ep.pad_tokens is not None:
            common_n = max(ep.pad_tokens, real_n)
        else:
            counts = ep.all_gather(
                torch.tensor([real_n], device=hs.device, dtype=torch.int64)
            )
            # Inherent host sync: `common_n` sizes the `pad` for the torch.cat below, so the padded
            # tensor's shape must be known host-side — it cannot be removed without a device sync.
            common_n = int(counts.max().item())
        if common_n > real_n:
            pad = common_n - real_n
            hs = torch.cat([hs, hs.new_zeros(pad, hs.shape[1])], dim=0)
            ep_w = torch.cat([ep_w, ep_w.new_zeros(pad, ep_w.shape[1])], dim=0)
            ep_i = torch.cat([ep_i, ep_i.new_zeros(pad, ep_i.shape[1])], dim=0)
        N = hs.shape[0]  # common token count, identical on every rank
        g_hidden = ep.all_gather(hs)  # (dp*N, H)
        g_weights = ep.all_gather(ep_w)  # (dp*N, top_k)
        g_ids = ep.all_gather(ep_i)  # (dp*N, top_k)
        lo, hi = self.local_expert_offset, self.local_expert_offset + self.local_num_experts
        is_local = (g_ids >= lo) & (g_ids < hi)
        local_ids = torch.where(is_local, g_ids - lo, torch.zeros_like(g_ids))
        local_weights = torch.where(is_local, g_weights, torch.zeros_like(g_weights))
        partial = local_kernel(g_hidden, local_weights, local_ids.to(torch.int32))  # (dp*N, H)
        partial = ep.all_reduce(partial)  # SUM across ranks -> full result for every token
        return partial[self.ep_dp_rank * N : self.ep_dp_rank * N + real_n]  # drop padding

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor | None = None,
        *,
        topk_weights: torch.Tensor | None = None,
        topk_ids: torch.Tensor | None = None,
        reduce: bool = True,
    ):
        # Either pass raw `router_logits` (fused softmax+topk inside the kernel) OR a precomputed
        # `topk_weights`/`topk_ids` route (GLM/DeepSeek noaux_tc, ZAYA top-1 computed in the model).
        # The expert scheme + kernel are owned by the config-driven `self._moe_method`; MoELayer only
        # owns routing + the EP dispatch/combine + the TP all-reduce.
        method = self._moe_method
        w13, w2 = self.gate_up_proj, self.down_proj
        if self.enable_ep:
            # Expert-parallel dispatch/combine (only schemes with method.supports_ep reach here). EP
            # can't defer routing to the kernel (it must all_gather a route), so precompute topk here —
            # renormalized over ALL top_k BEFORE the per-rank local-expert masking, then passed
            # precomputed. fp8 already REQUIRES the model-side route; W4A8/W4A16 derive it via _ep_route.
            # The all_gather/mask/all_reduce scaffold + N self-coordination live in _ep_dispatch; only
            # the local-shard kernel (method.ep_local) is scheme-specific.
            if method.needs_precomputed_route:
                assert topk_ids is not None, "EP fp8 experts need the model-side precomputed route"
                ep_w, ep_i = topk_weights, topk_ids
            else:
                ep_w, ep_i = self._ep_route(router_logits, topk_weights, topk_ids)
            final_hidden_states = self._ep_dispatch(
                hidden_states, ep_w, ep_i,
                lambda gh, lw, li: method.ep_local(
                    w13, w2, gh, lw, li, top_k=self.top_k, renormalize=self.renormalize
                ),
            )
        else:
            final_hidden_states = method.apply(
                w13, w2, hidden_states,
                router_logits=router_logits, topk_weights=topk_weights, topk_ids=topk_ids,
                top_k=self.top_k, renormalize=self.renormalize,
                activation=self.activation,
                apply_router_weight_on_input=self.apply_router_weight_on_input,
            )
        # EP already all_reduce'd over the dp/EP group (which subsumes any per-replica TP reduce —
        # ZAYA is tp_size=1 anyway), so skip the TP epilogue when the EP path ran. reduce=False also
        # skips it so the caller can fuse this partial with another row-parallel partial (shared expert)
        # and all_reduce once — the row-parallel down-proj output is a per-rank partial either way.
        if self.tp_size > 1 and not self.enable_ep and reduce:
            final_hidden_states = self._comm.all_reduce(final_hidden_states)
        return final_hidden_states
