from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

import torch
from minisgl.core import get_global_ctx
from minisgl.layers import BaseOP, OPList, ParallelLMHead, RMSNormFused, VocabParallelEmbedding
from minisgl.utils import nvtx_annotate

from .base import BaseLLMModel
from .utils import GatedMLP as Qwen3MLP
from .utils import RopeAttn as Qwen3Attn

if TYPE_CHECKING:
    from .config import ModelConfig


class Qwen3DecoderLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        self.self_attn = Qwen3Attn(config, layer_id, has_qk_norm=True)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
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


class Qwen3Model(BaseOP):
    def __init__(self, config: ModelConfig):
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = OPList(
            [Qwen3DecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]
        )
        self.norm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )
        # Spec-decode aux capture: decoder-layer ids whose output hidden is stashed (None = off).
        self._capture_layer_ids: List[int] | None = None

    def set_capture_layers(self, ids: List[int] | None) -> None:
        self._capture_layer_ids = list(ids) if ids else None

    def forward(
        self, input_ids: torch.Tensor, return_hidden: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor | None]:
        x = self.embed_tokens.forward(input_ids)
        residual: torch.Tensor | None = None
        # Aux capture is OFF unless return_hidden AND layers are programmed: zero cost otherwise.
        cap = self._capture_layer_ids if return_hidden else None
        cap_set = set(cap) if cap else None
        grabbed: dict[int, torch.Tensor] = {}
        for lid, layer in enumerate(self.layers.op_list):
            x, residual = layer.forward(x, residual)
            if cap_set is not None and lid in cap_set:
                # output hidden of layer lid = the residual stream after it (feeds the next layer).
                grabbed[lid] = residual.clone()
        final = self.norm.forward(x, residual)[0]
        if return_hidden:
            # stack in the programmed id order so a consumer can index aux by position; None if empty.
            aux_stack = torch.stack([grabbed[i] for i in cap], dim=0) if cap else None
            return final, aux_stack
        return final


class Qwen3ForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.model = Qwen3Model(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )
        super().__init__()

    def forward(self, return_hidden: bool = False):
        input_ids = get_global_ctx().batch.input_ids
        if return_hidden:
            # last_hidden = post-final-norm hidden (pre-lm_head); aux = stacked captured layers.
            last_hidden, aux_hidden = self.model.forward(input_ids, return_hidden=True)
            return self.lm_head.forward(last_hidden), last_hidden, aux_hidden
        return self.lm_head.forward(self.model.forward(input_ids))

    def set_capture_layers(self, ids: List[int] | None) -> None:
        self.model.set_capture_layers(ids)


__all__ = ["Qwen3ForCausalLM"]
