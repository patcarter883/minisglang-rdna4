from __future__ import annotations

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
        intermediate_size_per_partition = div_even(intermediate_size, tp_size)
        if quant is not None:
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

    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor):
        if self.quant is not None:
            from minisgl.quant import kernels

            w13, w2 = self.gate_up_proj, self.down_proj
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
