from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
from minisgl.core import get_global_ctx
from minisgl.distributed import DistributedCommunicator, get_tp_info
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
        consumes `_w_op`/`_scales_op` directly)."""
        E = self.weight.shape[0]
        return torch.stack(
            [(self.weight[e].float() * self.weight_scale[e]).to(dtype) for e in range(E)],
            dim=0,
        )

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
    ):
        super().__init__()

        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self._comm = DistributedCommunicator()

        tp_info = get_tp_info()
        self.tp_size = tp_size = tp_info.size
        self.renormalize = renormalize
        self.activation = activation
        self.apply_router_weight_on_input = apply_router_weight_on_input
        self.quant = quant
        self.fp8_experts = fp8_experts
        intermediate_size_per_partition = div_even(intermediate_size, tp_size)
        if fp8_experts:
            assert quant is None, "fp8_experts is its own storage path (not a QuantConfig)"
            # Weight-only fp8 (ZAYA): store F8_E4M3 + per-channel F32 scale (~8 GB) and feed the native
            # W8A8 grouped-MoE kernel directly (post_load builds the op layout). The legacy
            # dequant->Triton path is kept only behind MINISGL_ZAYA_OLDMOE=1 (A/B reference).
            self.gate_up_proj = _GroupedFP8Experts(
                num_experts, 2 * intermediate_size_per_partition, hidden_size
            )
            self.down_proj = _GroupedFP8Experts(
                num_experts, hidden_size, intermediate_size_per_partition
            )
        elif quant is not None:
            # int4 W4A8 grouped experts (silu-only SwiGLU MoE, the proven w4a8_fp8_wmma path).
            # GPTQ (K-major qweight) and AWQ-gemm (N-major qweight, interleaved, asymmetric) differ
            # only in checkpoint layout; both convert to the op's grouped layout in post_load.
            assert activation == "silu" and not apply_router_weight_on_input, (
                "MoE W4A8 path is silu-only without router-weight-on-input"
            )
            if quant.is_gptq:
                Experts = _GroupedGPTQExperts
            elif quant.is_awq:
                Experts = _GroupedAWQExperts
            elif quant.is_rxf:
                Experts = _GroupedRXFExperts
            else:
                raise AssertionError(f"MoE W4A8 unsupported quant method: {quant.method}")
            self.gate_up_proj = Experts(
                num_experts, 2 * intermediate_size_per_partition, hidden_size, quant
            )
            self.down_proj = Experts(
                num_experts, hidden_size, intermediate_size_per_partition, quant
            )
        else:
            self.gate_up_proj = torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size,
            )
            self.down_proj = torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor | None = None,
        *,
        topk_weights: torch.Tensor | None = None,
        topk_ids: torch.Tensor | None = None,
    ):
        # Either pass raw `router_logits` (fused softmax+topk inside the kernel) OR a precomputed
        # `topk_weights`/`topk_ids` route (GLM/DeepSeek noaux_tc computed in the model).
        precomputed = topk_ids is not None
        if self.fp8_experts:
            # Weight-only fp8 (ZAYA): native W8A8-fp8 grouped-MoE HIP kernel (fp8 e4m3 weights +
            # per-output-channel f32 scale, fp8 activations) — no bf16 dequant spike, WMMA compute.
            # Replaces the old fp8->bf16-dequant->Triton path; experts stay ~8 GB fp8 in HBM.
            from minisgl.quant import kernels

            assert precomputed, "fp8 experts use the precomputed-route path (ZAYA top-1 + MOD)"
            w13, w2 = self.gate_up_proj, self.down_proj
            if os.environ.get("MINISGL_ZAYA_OLDMOE", "0") == "1":
                # A/B reference: legacy fp8->bf16-dequant->Triton path (weights kept in post_load).
                from minisgl.moe.fused import fused_experts_impl

                w13_bf16 = w13.dequant(hidden_states.dtype)
                w2_bf16 = w2.dequant(hidden_states.dtype)
                final_hidden_states = fused_experts_impl(
                    hidden_states,
                    w13_bf16,
                    w2_bf16,
                    topk_weights,
                    topk_ids,
                    activation=self.activation,
                    apply_router_weight_on_input=self.apply_router_weight_on_input,
                )
            else:
                final_hidden_states = kernels.w8a8_moe(
                    hidden_states,
                    w13._w_op,
                    w13._scales_op,
                    w2._w_op,
                    w2._scales_op,
                    None,
                    self.top_k,
                    self.renormalize,
                    topk_weights=topk_weights,
                    topk_ids=topk_ids,
                )
        elif self.quant is not None:
            from minisgl.quant import kernels

            w13, w2 = self.gate_up_proj, self.down_proj
            if self.quant.is_rxf:
                assert not precomputed, "RXF MoE precomputed-topk path not wired yet"
                final_hidden_states = kernels.rxf_moe(
                    hidden_states,
                    w13.weight_packed,
                    w13.weight_scale,
                    w2.weight_packed,
                    w2.weight_scale,
                    router_logits,
                    self.top_k,
                    self.renormalize,
                    span=self.quant.rotation_span,
                )
            else:
                final_hidden_states = kernels.w4a8_moe(
                    hidden_states,
                    w13._w_op,
                    w13._scales_op,
                    w13._zeros_op,
                    w2._w_op,
                    w2._scales_op,
                    w2._zeros_op,
                    router_logits,
                    self.top_k,
                    self.renormalize,
                    topk_weights=topk_weights,
                    topk_ids=topk_ids,
                )
        elif precomputed:
            # Unquantized experts with a route computed in the model (Zaya top-1 + MOD, GLM
            # noaux_tc). The moe_backend fuses softmax+topk internally, so it can't take a
            # precomputed route — call the stacked-expert kernel directly with our topk tensors.
            from minisgl.moe.fused import fused_experts_impl

            final_hidden_states = fused_experts_impl(
                hidden_states,
                self.gate_up_proj,
                self.down_proj,
                topk_weights,
                topk_ids,
                activation=self.activation,
                apply_router_weight_on_input=self.apply_router_weight_on_input,
            )
        else:
            ctx = get_global_ctx()
            final_hidden_states = ctx.moe_backend.forward(
                hidden_states=hidden_states,
                w1=self.gate_up_proj,
                w2=self.down_proj,
                gating_output=router_logits,
                topk=self.top_k,
                renormalize=self.renormalize,
                activation=self.activation,
                apply_router_weight_on_input=self.apply_router_weight_on_input,
            )
        if self.tp_size > 1:
            final_hidden_states = self._comm.all_reduce(final_hidden_states)
        return final_hidden_states
