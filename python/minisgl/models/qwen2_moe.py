from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
from minisgl.core import get_global_ctx
from minisgl.layers import (
    BaseOP,
    LinearColParallelMerged,
    LinearReplicated,
    LinearRowParallel,
    MoELayer,
    OPList,
    ParallelLMHead,
    RMSNormFused,
    VocabParallelEmbedding,
    silu_and_mul,
)
from minisgl.quant import create_linear_method
from minisgl.utils import nvtx_annotate

from .base import BaseLLMModel
from .utils import RopeAttn

if TYPE_CHECKING:
    from .config import ModelConfig


class Qwen2MoeSharedExpert(BaseOP):
    """The always-on shared expert: a quantized SwiGLU MLP at shared_expert_intermediate_size.
    Same compute as GatedMLP but with an explicit (larger) intermediate size, so it can't reuse
    GatedMLP (which is hardwired to config.intermediate_size)."""

    def __init__(self, config: ModelConfig):
        qm = create_linear_method(config.quant)
        inter = config.shared_expert_intermediate_size
        self.gate_up_proj = LinearColParallelMerged(
            config.hidden_size, [inter, inter], has_bias=False, quant_method=qm
        )
        self.down_proj = LinearRowParallel(
            inter, config.hidden_size, has_bias=False, quant_method=qm
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj.forward(silu_and_mul(self.gate_up_proj.forward(x)))


class Qwen2MoeSparseBlock(BaseOP):
    """Qwen2-MoE block: top-k routed experts + an always-on shared expert weighted by a
    sigmoid gate. final = routed(x) + sigmoid(shared_gate(x)) * shared(x)."""

    def __init__(self, config: ModelConfig):
        self.gate = LinearReplicated(config.hidden_size, config.num_experts, has_bias=False)
        self.experts = MoELayer(
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
        )
        self.shared_expert = Qwen2MoeSharedExpert(config)
        self.shared_expert_gate = LinearReplicated(config.hidden_size, 1, has_bias=False)

    @nvtx_annotate("MoE")
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        shared_out = self.shared_expert.forward(hidden_states)
        shared_out = torch.sigmoid(self.shared_expert_gate.forward(hidden_states)) * shared_out
        router_logits = self.gate.forward(hidden_states)
        routed_out = self.experts.forward(
            hidden_states=hidden_states, router_logits=router_logits
        )
        return (routed_out + shared_out).view(num_tokens, hidden_dim)


class Qwen2MoeDecoderLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        # Qwen2 attention: qkv carry a (real) bias; o_proj's bias is an all-zero GPTQ
        # placeholder (skipped on load). No q/k norm.
        self.self_attn = RopeAttn(config, layer_id, has_attn_bias=True, has_qk_norm=False)
        self.mlp = Qwen2MoeSparseBlock(config)
        self.input_layernorm = RMSNormFused(size=config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNormFused(
            size=config.hidden_size, eps=config.rms_norm_eps
        )
        self._layer_id = layer_id

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, residual = self.input_layernorm.forward(x, residual)
        x = self.self_attn.forward(x)
        x, residual = self.post_attention_layernorm.forward(x, residual)
        x = self.mlp.forward(x)
        return x, residual


class Qwen2MoeModel(BaseOP):
    def __init__(self, config: ModelConfig):
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size, embedding_dim=config.hidden_size
        )
        self.layers = OPList(
            [Qwen2MoeDecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]
        )
        self.norm = RMSNormFused(size=config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens.forward(input_ids)
        residual: torch.Tensor | None = None
        for layer in self.layers.op_list:
            x, residual = layer.forward(x, residual)
        return self.norm.forward(x, residual)[0]


class Qwen2MoeForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.model = Qwen2MoeModel(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )
        super().__init__()

    def forward(self) -> torch.Tensor:
        output = self.model.forward(get_global_ctx().batch.input_ids)
        return self.lm_head.forward(output)


__all__ = ["Qwen2MoeForCausalLM"]
