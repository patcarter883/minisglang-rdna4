"""GLM-4.7-Flash (Glm4MoeLiteForCausalLM, model_type=glm4_moe_lite) — MLA + fine-grained MoE.

WIP / GPU-UNVALIDATED. Brings up GLM-4.7-Flash on the absorbed-MLA path (mla_hip kernels +
MLAKVCache latent pool) with the existing AWQ/W4A8 quant path on the MLP linears. Attention
(MLA) stays bf16. Validate against an HF reference before trusting outputs.

Architecture (from config.json):
  - MLA attention: q_a_proj(→q_lora 768)→q_a_layernorm→q_b_proj(→H·256); kv_a_proj_with_mqa
    (→kv_lora 512 ‖ k_rope 64)→kv_a_layernorm(512); kv_b_proj(512→H·(192+256)). Per head qk=256
    (nope 192 + rope 64), v=256. RoPE on the 64-dim rope sub-vector only. Decode is ABSORBED
    (q_nope·W_UK over the latent, then W_UV); prefill MATERIALIZES full per-head K/V from the latent.
  - MoE: layer 0 dense (first_k_dense_replace=1); layers 1+ = 64 routed experts (top-4, noaux_tc:
    sigmoid + e_score_correction_bias, n_group=1 → plain top-4, normalize, ×routed_scaling_factor)
    PLUS one always-on shared expert (added, not gated).

QUANT SPLIT (mirrors qwen3_5_moe + the [shared-expert-keep-bf16] convention): ONLY the routed
experts are quantized (W4A8/AWQ). The MLA attention, router gate, the always-on shared expert, the
dense layer-0 MLP, and lm_head stay bf16. If a given AWQ checkpoint instead quantizes the shared
expert / dense layers, flip those modules to the model quant. TP=1 only (AWQ INT4 ~9 GB fits one
card); MLA TP sharding is a follow-up. The MTP head (num_nextn_predict_layers) is skipped.
"""
from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Tuple

import torch
import torch.nn.functional as F
from minisgl.core import get_global_ctx
from minisgl.distributed import get_tp_info
from minisgl.layers import (
    BaseOP,
    LinearColParallelMerged,
    LinearReplicated,
    LinearRowParallel,
    MoELayer,
    OPList,
    ParallelLMHead,
    RMSNorm,
    RMSNormFused,
    VocabParallelEmbedding,
    get_rope,
    silu_and_mul,
)
from minisgl.quant import create_linear_method
from minisgl.utils import nvtx_annotate

from .base import BaseLLMModel
from .utils import GatedMLP

if TYPE_CHECKING:
    from .config import ModelConfig


class GLMMLAAttention(BaseOP):
    """Multi-head latent attention. Projections + RoPE + W_UK/W_UV absorption live here; the paged
    latent cache + the mla_hip kernels live in the MLABackend (ctx.attn_backend)."""

    def __init__(self, config: "ModelConfig", layer_id: int):
        assert get_tp_info().size == 1, "GLM MLA path is TP=1 only (sharding is a follow-up)"
        self._layer_id = layer_id
        self.num_heads = H = config.num_qo_heads
        self.qk_nope = config.qk_nope_head_dim
        self.qk_rope = config.qk_rope_head_dim
        self.qk_head_dim = self.qk_nope + self.qk_rope
        self.v_head_dim = config.v_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        eps = config.rms_norm_eps

        self.q_a_proj = LinearReplicated(config.hidden_size, config.q_lora_rank, has_bias=False)
        self.q_a_layernorm = RMSNorm(config.q_lora_rank, eps=eps)
        self.q_b_proj = LinearReplicated(config.q_lora_rank, H * self.qk_head_dim, has_bias=False)
        self.kv_a_proj_with_mqa = LinearReplicated(
            config.hidden_size, self.kv_lora_rank + self.qk_rope, has_bias=False
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=eps)
        self.kv_b_proj = LinearReplicated(
            self.kv_lora_rank, H * (self.qk_nope + self.v_head_dim), has_bias=False
        )
        self.o_proj = LinearReplicated(H * self.v_head_dim, config.hidden_size, has_bias=False)

        rc = config.rotary_config
        # RoPE over the 64-dim rope sub-vector only (full rotary on that 64-wide head).
        self.rotary = get_rope(
            head_dim=self.qk_rope,
            rotary_dim=self.qk_rope,
            max_position=rc.max_position,
            base=rc.base,
            rope_scaling=None,
        )

    def post_load(self) -> None:
        super().post_load()
        # Split kv_b_proj [H·(qk_nope+v), kv_lora] into the absorption tensors (views, no copy of
        # the matmul weight beyond the reshape). W_UK maps the latent c_KV -> per-head k_nope;
        # W_UV maps it -> per-head v.
        H, kv_lora = self.num_heads, self.kv_lora_rank
        w = self.kv_b_proj.weight.view(H, self.qk_nope + self.v_head_dim, kv_lora)
        self._w_uk = w[:, : self.qk_nope, :].contiguous()  # [H, qk_nope, kv_lora]
        self._w_uv = w[:, self.qk_nope :, :].contiguous()  # [H, v_head_dim, kv_lora]

    @nvtx_annotate("MLA")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        batch = ctx.batch
        backend = ctx.attn_backend
        metadata = batch.attn_metadata
        T = x.shape[0]
        H, nope, rope, vhd = self.num_heads, self.qk_nope, self.qk_rope, self.v_head_dim

        # ---- projections ----
        q = self.q_b_proj.forward(self.q_a_layernorm.forward(self.q_a_proj.forward(x)))
        q = q.view(T, H, self.qk_head_dim)
        q_nope, q_rope = q[..., :nope], q[..., nope:]  # [T,H,nope], [T,H,rope]

        kv = self.kv_a_proj_with_mqa.forward(x)  # [T, kv_lora + rope]
        c_kv = self.kv_a_layernorm.forward(kv[:, : self.kv_lora_rank].contiguous())  # [T, kv_lora]
        k_rope = kv[:, self.kv_lora_rank :]  # [T, rope] (shared across heads / MQA)

        # ---- RoPE on the rope sub-vectors (q: H heads, k: 1 head) ----
        q_rope, k_rope = self.rotary.forward(
            batch.positions, q_rope.reshape(T, H * rope).contiguous(), k_rope.contiguous()
        )
        q_rope = q_rope.view(T, H, rope)  # [T,H,rope]

        # latent stored in the cache = [c_KV (normed) ‖ k_rope (roped)], shared across heads.
        latent = torch.cat([c_kv, k_rope], dim=-1)  # [T, kv_lora + rope]
        backend.store_latent(latent, batch.out_loc, self._layer_id)

        if batch.is_decode:
            # ABSORBED decode: q_nope·W_UK -> latent space, attend over the paged latent, then ·W_UV.
            q_absorbed = torch.einsum("thn,hnl->thl", q_nope, self._w_uk)  # [T,H,kv_lora]
            q_full = torch.cat([q_absorbed, q_rope], dim=-1)  # [T,H,kv_lora+rope]
            o_latent = backend.decode(q_full, self._layer_id, metadata)  # [T,H,kv_lora]
            o = torch.einsum("thl,hdl->thd", o_latent, self._w_uv)  # [T,H,v]
        else:
            # MATERIALIZED prefill: rebuild full per-head K/V for each seq from the latent cache
            # (includes the new tokens just stored), then varlen flash attention with prefix-offset
            # causal (cu_seqlens_q = new tokens, cu_seqlens_k = full KV).
            q_full = torch.cat([q_nope, q_rope], dim=-1)  # [T,H,qk]  (new tokens only)
            latent_flat = ctx.kv_cache.latent_cache(self._layer_id).view(-1, self.kv_lora_rank + rope)
            page_table = ctx.page_table  # raw per-token slots (page_size=1 indexing)
            k_list, v_list = [], []
            for req in batch.padded_reqs:
                slots = page_table[req.table_idx, : req.device_len].long()
                lat = latent_flat[slots]  # [L, kv_lora + rope]
                kv = self.kv_b_proj.forward(lat[:, : self.kv_lora_rank].contiguous())
                kv = kv.view(-1, H, nope + vhd)
                k_nope, v = kv[..., :nope], kv[..., nope:]  # [L,H,nope], [L,H,v]
                kr = lat[:, self.kv_lora_rank :].unsqueeze(1).expand(-1, H, rope)  # [L,H,rope]
                k_list.append(torch.cat([k_nope, kr], dim=-1))  # [L,H,qk]
                v_list.append(v)
            k_all = torch.cat(k_list, dim=0)
            v_all = torch.cat(v_list, dim=0)
            o = backend.prefill(
                q_full, k_all, v_all,
                metadata.cu_seqlens_q, metadata.cu_seqlens_k, metadata.max_seqlen_q,
            )  # [T,H,v]

        return self.o_proj.forward(o.reshape(T, H * vhd))


class GLMTopkGate(BaseOP):
    """noaux_tc router: a replicated [E, hidden] linear PLUS a per-expert correction bias used
    only for top-k SELECTION (the routing weights are the un-biased sigmoid scores)."""

    def __init__(self, hidden_size: int, num_experts: int):
        self.weight = torch.empty(num_experts, hidden_size)
        self.e_score_correction_bias = torch.empty(num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight)  # [T, E] logits


class GLMSharedExpert(BaseOP):
    """Always-on shared expert (SwiGLU). Quantized with the model quant when present (AWQ)."""

    def __init__(self, config: "ModelConfig"):
        inter = config.moe_intermediate_size * max(1, config.n_shared_experts)
        qm = create_linear_method(config.quant)
        self.gate_up_proj = LinearColParallelMerged(
            config.hidden_size, [inter, inter], has_bias=False, quant_method=qm
        )
        self.down_proj = LinearRowParallel(
            inter, config.hidden_size, has_bias=False, quant_method=qm
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj.forward(silu_and_mul(self.gate_up_proj.forward(x)))


class GLMSparseBlock(BaseOP):
    """noaux_tc router + W4A8 routed experts + always-on shared expert (added, not gated).

    `expert_quant` is threaded in separately: the surrounding backbone is built unquantized
    (quant=None) so the gate + shared expert stay bf16; only the routed experts are quantized."""

    def __init__(self, config: "ModelConfig", expert_quant):
        self.gate = GLMTopkGate(config.hidden_size, config.num_experts)
        self.experts = MoELayer(
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=False,  # noaux_tc normalize is done here, weights passed in precomputed
            quant=expert_quant,
        )
        self.shared_experts = GLMSharedExpert(config)
        self.top_k = config.num_experts_per_tok
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor

    def _noaux_tc(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # logits [T, E]. Returns (topk_weights [T,top_k] f32, topk_ids [T,top_k] i32).
        scores = logits.float().sigmoid()  # routing weights come from the UN-biased scores
        choice = scores + self.gate.e_score_correction_bias.float()  # bias only steers selection
        if self.n_group > 1:
            T, E = choice.shape
            grp = choice.view(T, self.n_group, -1)
            group_scores = grp.topk(2, dim=-1).values.sum(dim=-1)  # [T, n_group] (top-2 per group)
            keep = group_scores.topk(self.topk_group, dim=-1).indices  # [T, topk_group]
            mask = torch.zeros_like(group_scores).scatter_(1, keep, 1.0).bool()
            mask = mask.unsqueeze(-1).expand(T, self.n_group, E // self.n_group).reshape(T, E)
            choice = choice.masked_fill(~mask, float("-inf"))
        topk_ids = choice.topk(self.top_k, dim=-1).indices  # [T, top_k]
        topk_weights = scores.gather(1, topk_ids)  # original sigmoid scores at selected experts
        if self.norm_topk_prob:
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_weights.float().contiguous(), topk_ids.int().contiguous()

    @nvtx_annotate("MoE")
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        topk_weights, topk_ids = self._noaux_tc(self.gate.forward(hidden_states))
        routed = self.experts.forward(
            hidden_states, topk_weights=topk_weights, topk_ids=topk_ids
        )
        shared = self.shared_experts.forward(hidden_states)
        return (routed + shared).view(num_tokens, hidden_dim)


class GLMDecoderLayer(BaseOP):
    def __init__(self, config: "ModelConfig", layer_id: int, expert_quant):
        self.self_attn = GLMMLAAttention(config, layer_id)
        # first_k_dense_replace early layers use a dense MLP; the rest are sparse MoE blocks.
        if layer_id < config.first_k_dense_replace:
            self.mlp = GatedMLP(config)  # bf16 (config is the unquantized backbone)
        else:
            self.mlp = GLMSparseBlock(config, expert_quant)
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


class GLMModel(BaseOP):
    def __init__(self, config: "ModelConfig", expert_quant):
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size, embedding_dim=config.hidden_size
        )
        self.layers = OPList(
            [GLMDecoderLayer(config, layer_id, expert_quant) for layer_id in range(config.num_layers)]
        )
        self.norm = RMSNormFused(size=config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens.forward(input_ids)
        residual: torch.Tensor | None = None
        for layer in self.layers.op_list:
            x, residual = layer.forward(x, residual)
        return self.norm.forward(x, residual)[0]


class Glm4MoeLiteForCausalLM(BaseLLMModel):
    def __init__(self, config: "ModelConfig"):
        # Only the routed experts are quantized; build the rest of the model (MLA attention, gate,
        # shared expert, dense layer-0, lm_head) unquantized and hand the quant to the experts.
        expert_quant = config.quant
        backbone_cfg = dataclasses.replace(config, quant=None)
        self.model = GLMModel(backbone_cfg, expert_quant)
        config = backbone_cfg
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


__all__ = ["Glm4MoeLiteForCausalLM"]
