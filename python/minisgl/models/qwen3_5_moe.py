"""Phase 3M-1 — Qwen3.5-MoE GDN-hybrid model (text path of Qwen3_5MoeForConditionalGeneration).

Identical to the 4B `qwen3_5` GDN-hybrid decoder (linear_attention/full_attention interleave,
gated partial-rotary attention, (1+w) RMSNorm) EXCEPT the per-layer MLP is a Qwen2-MoE-style
sparse block: a top-k router over `num_experts` grouped experts PLUS an always-on shared expert
weighted by a sigmoid gate (`final = routed(x) + sigmoid(shared_gate(x))·shared(x)`).

In the canonical checkpoint (cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit, refs/main = AWQ-gemm g32) ONLY
the routed experts are quantized; the GDN/attention, the shared expert, both gates, and lm_head
are F16 (per the quant `modules_to_not_convert`). So the qwen3_5 BACKBONE is built UNQUANTIZED
(quant=None) and the real AWQ quant is handed only to the MoE experts. Reuses qwen3_5's
decoder/model/head via their `mlp_factory` seam — the MoE block is the sole structural difference.
Serve is gated on TP=2 (the 35B does not fit one 16 GB card); weight map = Phase 3M-3.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from minisgl.layers import (
    BaseOP,
    LinearColParallelMerged,
    LinearReplicated,
    LinearRowParallel,
    MoELayer,
    silu_and_mul,
)
from minisgl.quant import create_linear_method
from minisgl.utils import nvtx_annotate

from .qwen3_5 import Qwen3_5ForConditionalGeneration, _lp_timed

if TYPE_CHECKING:
    from minisgl.quant.config import QuantConfig

    from .config import ModelConfig


class Qwen3_5MoeSharedExpert(BaseOP):
    """Always-on shared expert: an UNQUANTIZED (F16) SwiGLU at shared_expert_intermediate_size."""

    def __init__(self, config: "ModelConfig"):
        inter = config.shared_expert_intermediate_size
        qm = create_linear_method(None)  # F16
        self.gate_up_proj = LinearColParallelMerged(
            config.hidden_size, [inter, inter], has_bias=False, quant_method=qm
        )
        self.down_proj = LinearRowParallel(
            inter, config.hidden_size, has_bias=False, quant_method=qm
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj.forward(silu_and_mul(self.gate_up_proj.forward(x)))


class Qwen3_5MoeSparseBlock(BaseOP):
    """Router gate + shared gate + shared expert are F16 (in the checkpoint's quant ignore list);
    only the routed experts carry the int4/mxfp4 quant (MoELayer's grouped path). `expert_quant` is
    threaded in separately so the routed-expert precision is explicit here — the surrounding backbone
    config now keeps the real quant (per-module is_module_quantized gates each backbone linear)."""

    def __init__(self, config: "ModelConfig", expert_quant: "QuantConfig | None"):
        self.gate = LinearReplicated(config.hidden_size, config.num_experts, has_bias=False)
        self.experts = MoELayer(
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            # ALWAYS renormalize: the Qwen3.5-MoE router (transformers Qwen3_5MoeTopKRouter) divides
            # the top-k softmax probs by their sum UNCONDITIONALLY — there is no norm_topk_prob toggle
            # for this architecture, and the config omits the key (minisgl would default it False).
            # Skipping it leaves the routed weights summing to <1 -> mis-scaled experts -> degenerate.
            renormalize=True,
            quant=expert_quant,
        )
        self.shared_expert = Qwen3_5MoeSharedExpert(config)
        self.shared_expert_gate = LinearReplicated(config.hidden_size, 1, has_bias=False)

    @nvtx_annotate("MoE")
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # "shared" sub-bucket of the layer-prof "ffn" total (MINISGL_LAYER_PROF). Summed over all
        # layers, reported per-step -> direct per-step shared-expert cost (Task B #18 attribution).
        shared_out = _lp_timed("shared", self.shared_expert.forward, hidden_states)
        shared_out = torch.sigmoid(self.shared_expert_gate.forward(hidden_states)) * shared_out
        router_logits = self.gate.forward(hidden_states)
        routed_out = self.experts.forward(
            hidden_states=hidden_states, router_logits=router_logits
        )
        return (routed_out + shared_out).view(num_tokens, hidden_dim)


class Qwen3_5MoeForConditionalGeneration(Qwen3_5ForConditionalGeneration):
    def __init__(self, config: "ModelConfig"):
        # Config-driven per-module quant across the WHOLE model. The backbone keeps the real quant
        # config; each GDN/attention projection is gated by is_module_quantized (the checkpoint's
        # ignore list), so the AWQ 35B (entire backbone in modules_to_not_convert) stays bf16 while
        # the MXFP4 checkpoint (GDN in_proj_qkv/z/a/b + out_proj quantized; self_attn, conv1d, norm,
        # gates, shared_expert in the ignore list) builds exactly those projections mxfp4. The routed
        # experts still get the quant via the mlp_factory closure. No model-name branch.
        expert_quant = config.quant
        backbone_cfg = config

        def mlp_factory(cfg: "ModelConfig") -> BaseOP:
            return Qwen3_5MoeSparseBlock(cfg, expert_quant)

        # The MTP head follows the checkpoint's precision: quantized only if mtp.* is NOT in the quant
        # ignore list. A bf16/fp16 MTP head on a quantized backbone builds unquantized so its full-
        # precision experts load (QuantConfig.is_module_quantized; universal across quant methods).
        mtp_quant = expert_quant
        if expert_quant is not None and not expert_quant.is_module_quantized(
            "mtp.layers.0.mlp.experts.0.gate_proj"
        ):
            mtp_quant = None

        def mtp_mlp_factory(cfg: "ModelConfig") -> BaseOP:
            return Qwen3_5MoeSparseBlock(cfg, mtp_quant)

        super().__init__(backbone_cfg, mlp_factory=mlp_factory, mtp_mlp_factory=mtp_mlp_factory)


__all__ = ["Qwen3_5MoeForConditionalGeneration"]
