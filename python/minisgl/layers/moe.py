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
        # Expert parallelism: each DP replica OWNS only experts [dp_rank*E/dp : (dp_rank+1)*E/dp], so
        # the per-expert weight buffers are sized to the LOCAL count (E/dp). EP is fp8-only (ZAYA);
        # the dispatch/combine collective in forward() reconstructs the full result. enable_ep is the
        # process-global toggle (set by the Engine before model build); off => full replicated count.
        self.enable_ep = is_ep_enabled() and fp8_experts
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
        if fp8_experts:
            assert quant is None, "fp8_experts is its own storage path (not a QuantConfig)"
            # Weight-only fp8 (ZAYA): store F8_E4M3 + per-channel F32 scale (~8 GB) and feed the native
            # W8A8 grouped-MoE kernel directly (post_load builds the op layout). The legacy
            # dequant->Triton path is kept only behind MINISGL_ZAYA_OLDMOE=1 (A/B reference).
            # EP: size to the LOCAL expert shard (E/dp); EP off => full E. The streaming loader
            # (_store_expert) yields a stack of exactly this many experts.
            self.gate_up_proj = _GroupedFP8Experts(
                self.local_num_experts, 2 * intermediate_size_per_partition, hidden_size
            )
            self.down_proj = _GroupedFP8Experts(
                self.local_num_experts, hidden_size, intermediate_size_per_partition
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
            elif quant.is_compressed_tensors:
                # int4 weight-only (W4A16) experts — convert to the op layout in post_load and run
                # through the same w4a8_moe kernel as GPTQ/AWQ (the `else` forward branch).
                Experts = _GroupedCompressedTensorsExperts
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
            # ZAYA fp8 experts. DEFAULT = W8A16 (fp8 weights + bf16 acts): bf16-activation correctness
            # at ~native-fp8 speed (autotuned ~56ms fused). Opt-outs: W8A16=0 -> native W8A8 fp8-act
            # kernel (~7% faster AR, fp8-act quality); OLDMOE=1 -> legacy dequant->Triton reference. The
            # EP (TP>1) path ALSO uses W8A16 (quality) on the local expert shard when the kernel is
            # built (see the enable_ep branch below); falls back to native W8A8 if it isn't.
            from minisgl.quant import kernels

            assert precomputed, "fp8 experts use the precomputed-route path (ZAYA top-1 + MOD)"
            w13, w2 = self.gate_up_proj, self.down_proj
            oldmoe = os.environ.get("MINISGL_ZAYA_OLDMOE", "0") == "1"
            w8a16 = os.environ.get("MINISGL_ZAYA_W8A16", "1") != "0"  # DEFAULT ON (opt out with =0)
            fused_moe_w8a16 = None
            if w8a16 and not oldmoe:
                try:  # fail-safe: envs without the built extension fall back to native W8A8 below
                    from moe_w8a16_wmma import fused_moe_w8a16
                except ImportError:
                    fused_moe_w8a16 = None
            if fused_moe_w8a16 is not None and not self.enable_ep:
                # W8A16: dequant the fp8 weight tile to bf16 IN-REGISTER (no full-stack materialize),
                # routed experts only; bf16 acts. Uses the always-present op-layout fp8 buffers
                # (_w_op/_scales_op). Fixes the fused-TiDAR OLDMOE=1 284ms/step dequant flood.
                final_hidden_states = fused_moe_w8a16(
                    hidden_states.to(torch.bfloat16),
                    w13._w_op, w13._scales_op, w2._w_op, w2._scales_op,
                    topk_weights, topk_ids.to(torch.int32),
                )
            elif oldmoe:
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
            elif self.enable_ep:
                # Expert-parallel dispatch/combine (graph-capturable, fixed shapes):
                #   1. all_gather every replica's token rows + top-1 route so each rank sees ALL
                #      tokens (rank r's own rows are the contiguous slice [r*N : (r+1)*N]).
                #   2. remap global expert id -> local (gid - offset); a token whose expert is NOT
                #      on this rank is masked by ZEROING its route weight (and clamping its id to a
                #      valid local 0) so the gather-reduce contributes nothing for it.
                #   3. the grouped MoE kernel over the LOCAL expert tensors with the remapped ids —
                #      W8A16 (quality) when the kernel is built, else native W8A8.
                #   4. all_reduce(SUM) the partial outputs: each token's top-1 expert lives on
                #      exactly one rank, so the sum reconstructs the full result; slice OUR rows.
                # This REPLACES the tp all_reduce epilogue (the EP all_reduce subsumes it).
                ep = get_global_ctx().ep
                assert ep is not None, "EP enabled but ctx.ep group not built"
                real_n = hidden_states.shape[0]
                # DECODE (graph): batch is already padded to a captured bs -> equal N on every rank,
                # pad_tokens is None. EAGER PREFILL: token counts differ, so zero-pad rows up to the
                # scheduler-agreed common N (all_gather requires equal N). Slice back real_n at the end.
                ep_w = topk_weights.contiguous()
                ep_i = topk_ids.to(torch.int32).contiguous()
                hs = hidden_states
                if ep.pad_tokens is not None and ep.pad_tokens > real_n:
                    pad = ep.pad_tokens - real_n
                    hs = torch.cat([hs, hs.new_zeros(pad, hs.shape[1])], dim=0)
                    ep_w = torch.cat([ep_w, ep_w.new_zeros(pad, ep_w.shape[1])], dim=0)
                    # padded rows get expert id 0 with weight 0 -> zero contribution after masking.
                    ep_i = torch.cat([ep_i, ep_i.new_zeros(pad, ep_i.shape[1])], dim=0)
                N = hs.shape[0]  # common token count, identical on every rank
                g_hidden = ep.all_gather(hs)  # (dp*N, H)
                g_weights = ep.all_gather(ep_w)  # (dp*N, top_k)
                g_ids = ep.all_gather(ep_i)  # (dp*N, top_k)
                lo, hi = self.local_expert_offset, self.local_expert_offset + self.local_num_experts
                is_local = (g_ids >= lo) & (g_ids < hi)
                local_ids = torch.where(is_local, g_ids - lo, torch.zeros_like(g_ids))
                local_weights = torch.where(is_local, g_weights, torch.zeros_like(g_weights))
                if fused_moe_w8a16 is not None:
                    # W8A16 (quality) on the LOCAL expert shard — same EP contract as W8A8 below but
                    # bf16 activations. w13._w_op is already this rank's [E_local,...] shard and
                    # local_ids are remapped into [0, E_local), so the kernel's moe_align groups over
                    # exactly the local experts; non-local tokens carry weight 0 -> zero contribution
                    # after the topk-weighted scatter (identical masking to the W8A8 path).
                    partial = fused_moe_w8a16(
                        g_hidden.to(torch.bfloat16),
                        w13._w_op, w13._scales_op, w2._w_op, w2._scales_op,
                        local_weights, local_ids.to(torch.int32),
                    )  # (dp*N, H)
                else:
                    partial = kernels.w8a8_moe(
                        g_hidden,
                        w13._w_op,
                        w13._scales_op,
                        w2._w_op,
                        w2._scales_op,
                        None,
                        self.top_k,
                        self.renormalize,
                        topk_weights=local_weights,
                        topk_ids=local_ids,
                    )  # (dp*N, H) — only this rank's local-expert tokens are non-zero
                partial = ep.all_reduce(partial)  # SUM across ranks -> full result for every token
                # Our rows are the contiguous [dp_rank*N : dp_rank*N+real_n] slice (drop any padding).
                final_hidden_states = partial[self.ep_dp_rank * N : self.ep_dp_rank * N + real_n]
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
        # EP already all_reduce'd over the dp/EP group (which subsumes any per-replica TP reduce —
        # ZAYA is tp_size=1 anyway), so skip the TP epilogue when the EP path ran.
        if self.tp_size > 1 and not self.enable_ep:
            final_hidden_states = self._comm.all_reduce(final_hidden_states)
        return final_hidden_states
