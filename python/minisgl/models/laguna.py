"""Laguna (poolside/Laguna-XS.2) — sliding-window-attention hybrid MoE. This module carries the
SWA-specific, CONFIG-DRIVEN attention wiring (per-layer QO head count, dual RoPE, per-layer sliding
window, and the compact KV-pool index that routes full layers to the main pool and sliding layers to
the window-bounded ring pool). It is model-name-free: every per-layer choice falls out of ModelConfig
(attn_head_counts, layer_types, sliding_window, sliding_rotary_config) exactly as parsed from the
checkpoint — no branch on model_type.

Scope: this file provides `LagunaAttention` (the gated GQA mixer, ready for the Phase-3 decoder) and
the `laguna_layer_plan` helper that computes each layer's (kind, heads, rope, window, compact-kv-id).
The full `LagunaForCausalLM` (MoE INT4/INT8 experts, dense layer-0, Hadamard R1, untied lm_head) is
the separate 5-phase bring-up in docs/laguna-port/LAGUNA_SERVING_PLAN.md — its MoE body is not built
here. What IS here is the attention subsystem the SWA feature exists to serve, and it is unit-checked
by meta-instantiation (tools/test_laguna_attn.py) against the real Laguna config shape.
"""
from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple

import torch
from minisgl.core import get_global_ctx
from minisgl.distributed import get_tp_info
from minisgl.layers import (
    AttentionLayer,
    BaseOP,
    LinearColParallelMerged,
    LinearOProj,
    LinearReplicated,
    MoELayer,
    OPList,
    ParallelLMHead,
    RMSNorm,
    RMSNormFused,
    VocabParallelEmbedding,
    silu_and_mul,
)
from minisgl.quant import create_linear_method
from minisgl.utils import div_even, nvtx_annotate

from .base import BaseLLMModel
from .utils import GatedMLP

if TYPE_CHECKING:
    from minisgl.models import ModelConfig, RotaryConfig


@dataclass(frozen=True)
class LayerPlan:
    """The per-layer attention descriptor, derived purely from ModelConfig."""

    is_sliding: bool
    num_qo_heads: int
    sliding_window: int  # 0 for a full layer
    kv_id: int  # compact index: full layers -> main pool; sliding layers -> SWA ring pool
    rotary_config: "RotaryConfig"


def laguna_layer_plan(config: "ModelConfig", layer_id: int) -> LayerPlan:
    """Compute layer `layer_id`'s attention plan. Full layers use rotary_config (yarn) + the full
    main pool (compact id = position among full-attn layers); sliding layers use sliding_rotary_config
    (default) + a `sliding_window`-token ring (compact id = position among sliding layers). The two
    id spaces are disjoint because the two pools are physically separate tensors."""
    assert config.layer_types is not None, "Laguna requires a per-layer attention schedule"
    is_sliding = config.layer_types[layer_id] == "sliding_attention"
    # Per-layer QO head count (Laguna: 48 full / 64 sliding); fall back to the uniform count.
    heads = (
        config.attn_head_counts[layer_id]
        if config.attn_head_counts is not None
        else config.num_qo_heads
    )
    if is_sliding:
        window = config.sliding_window or 0
        kv_id = config.swa_layer_ids.index(layer_id)
        rope = config.sliding_rotary_config or config.rotary_config
    else:
        window = 0
        kv_id = config.full_attn_layer_ids.index(layer_id)
        rope = config.rotary_config
    return LayerPlan(
        is_sliding=is_sliding,
        num_qo_heads=heads,
        sliding_window=window,
        kv_id=kv_id,
        rotary_config=rope,
    )


class LagunaAttention(BaseOP):
    """Gated GQA with per-layer heads + per-layer RoPE + optional sliding window.

    Structural differences from a plain GQA that this class threads generically from the config:
      * `num_qo_heads` is PER LAYER (48 full / 64 sliding) — sizes q_proj / o_proj / g_proj.
      * `rotary_config` is PER LAYER (full → yarn partial-0.5; sliding → default full-rotary).
      * `sliding_window` is PER LAYER (0 full / 512 sliding) → the AttentionLayer masks + routes the
        window-bounded ring pool for sliding layers.
      * a SEPARATE `g_proj` per-head output gate (Laguna: `self_attn.g_proj [heads, hidden]`),
        sigmoid-applied to the attention output before o_proj.
      * QK-norm (per-head RMSNorm on q/k before RoPE).
    The compact KV-pool id (full-main vs swa-ring) is passed to AttentionLayer as layer_id.
    """

    def __init__(self, config: "ModelConfig", layer_id: int):
        plan = laguna_layer_plan(config, layer_id)
        head_dim = config.head_dim
        nqo, nkv = plan.num_qo_heads, config.num_kv_heads
        q = config.quant
        prefix = f"model.layers.{layer_id}"

        def _method(module: str):
            name = f"{prefix}.self_attn.{module}"
            quantized = q is not None and q.is_module_quantized(name)
            return create_linear_method(q, quantized=quantized)

        # q/k/v are separate column-parallel projections (per-layer head count). All attention
        # projections are bf16 in the Laguna checkpoint (self_attn is ignore-listed) — the quant
        # method still falls out of the config, no model-name branch.
        self.q_proj = LinearColParallelMerged(
            config.hidden_size, [nqo * head_dim], has_bias=False, quant_method=_method("q_proj")
        )
        self.k_proj = LinearColParallelMerged(
            config.hidden_size, [nkv * head_dim], has_bias=False, quant_method=_method("k_proj")
        )
        self.v_proj = LinearColParallelMerged(
            config.hidden_size, [nkv * head_dim], has_bias=False, quant_method=_method("v_proj")
        )
        # Per-head output gate: g_proj emits one scalar per QO head ([nqo, hidden]).
        self.g_proj = LinearColParallelMerged(
            config.hidden_size, [nqo], has_bias=False, quant_method=_method("g_proj")
        )
        self.q_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)
        self.attn = AttentionLayer(
            layer_id=plan.kv_id,  # compact pool id (full → main pool, sliding → SWA ring pool)
            head_dim=head_dim,
            num_qo_heads=nqo,
            num_kv_heads=nkv,
            rotary_config=plan.rotary_config,  # per-layer rope (full=yarn / sliding=default)
            q_norm=self.q_norm,
            k_norm=self.k_norm,
            sliding_window=plan.sliding_window,  # 0 full / 512 sliding
        )
        self.o_proj = LinearOProj(
            head_dim * nqo, config.hidden_size, has_bias=False, quant_method=_method("o_proj")
        )
        self._head_dim = head_dim
        self._nqo_local = div_even(nqo, get_tp_info().size)
        self.plan = plan

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        q = self.q_proj.forward(x)
        k = self.k_proj.forward(x)
        v = self.v_proj.forward(x)
        gate = self.g_proj.forward(x)  # [n, nqo_local]
        qkv = torch.cat([q, k, v], dim=-1)
        o = self.attn.forward(qkv)  # [n, nqo_local*head_dim]
        # Laguna gated attention: per-head SOFTPLUS gate (NOT sigmoid), computed in fp32 then cast,
        # broadcast across head_dim, applied BEFORE o_proj (reference modeling_laguna.py
        # `F.softplus(g_proj(x).float())`). softplus ∈ [0,∞) — sigmoid would wrongly cap the gate at 1.
        gate = torch.nn.functional.softplus(gate.float()).to(o.dtype)
        o = o.view(n, self._nqo_local, self._head_dim) * gate.unsqueeze(-1)
        return self.o_proj.forward(o.reshape(n, self._nqo_local * self._head_dim))


class LagunaTopKRouter(BaseOP):
    """Laguna MoE router: a replicated [E, hidden] linear PLUS a per-expert `e_score_correction_bias`
    used ONLY for top-k SELECTION (arXiv:2408.15664 aux-loss-free balancing); the routing weights are
    the UN-biased sigmoid scores. Mirrors the reference `LagunaTopKRouter` (the checkpoint stores the
    bias at `mlp.experts.e_score_correction_bias`; the loader remaps it here to `mlp.gate`)."""

    def __init__(self, hidden_size: int, num_experts: int):
        self.weight = torch.empty(num_experts, hidden_size)
        self.e_score_correction_bias = torch.empty(num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # M-invariant GEMM (layers/minv.py) not raw F.linear: the router logits drive top-k expert
        # SELECTION, so a ~1-ULP M-dependence (a chunked/prefix-reused forward runs at a different M
        # than a cold one) could flip a near-tie selection -> different experts -> divergent output.
        # Routing through the fixed-tile WMMA GEMM makes a reused-prefix forward bit-identical to cold.
        from minisgl.layers.minv import minv_linear

        return minv_linear(x, self.weight)  # [T, E] logits


class LagunaSharedExpert(BaseOP):
    """Always-on shared expert (SwiGLU), ADDED (not gated) to the routed output. NVFP4-quantized in
    the XS-2.1 checkpoint (quant targets include `shared_expert.{gate,up,down}_proj`); the loader
    merges gate_proj+up_proj -> gate_up_proj at the leaf. REPLICATED across TP ranks (not sharded):
    each rank computes the full shared output, added to the already-all-reduced routed output."""

    def __init__(self, config: "ModelConfig", quant):
        inter = config.shared_expert_intermediate_size
        qm = create_linear_method(quant)
        self.gate_up_proj = LinearReplicated(
            config.hidden_size, 2 * inter, has_bias=False, quant_method=qm
        )
        self.down_proj = LinearReplicated(inter, config.hidden_size, has_bias=False, quant_method=qm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj.forward(silu_and_mul(self.gate_up_proj.forward(x)))


class LagunaSparseBlock(BaseOP):
    """Sigmoid-routed fine-grained MoE + always-on shared expert (Laguna `LagunaSparseMoeBlock`).

    Route: sigmoid(router_logits) scores; select top-k on scores+correction_bias; weights = the
    UN-biased sigmoid scores at the selected experts; normalize (norm_topk_prob); then the routed sum
    is scaled by routed_scaling_factor (2.5) — applied here to the per-token weights, which is exact
    (scaling a linear combination == scaling its weights) — and the shared expert is added, ungated.
    `expert_quant` (NVFP4) is threaded to the experts + shared expert; the router gate stays bf16."""

    def __init__(self, config: "ModelConfig", expert_quant):
        self.gate = LagunaTopKRouter(config.hidden_size, config.num_experts)
        self.experts = MoELayer(
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=False,  # normalize + scaling done here; weights passed in precomputed
            quant=expert_quant,
        )
        self.shared_expert = LagunaSharedExpert(config, expert_quant)
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor

    def _route(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = logits.float().sigmoid()  # routing weights = UN-biased sigmoid scores
        choice = scores + self.gate.e_score_correction_bias.float()  # bias steers SELECTION only
        topk_ids = choice.topk(self.top_k, dim=-1).indices  # [T, top_k]
        topk_weights = scores.gather(1, topk_ids)  # [T, top_k]
        if self.norm_topk_prob:
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_weights.float().contiguous(), topk_ids.int().contiguous()

    @nvtx_annotate("MoE")
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        topk_weights, topk_ids = self._route(self.gate.forward(hidden_states))
        routed = self.experts.forward(
            hidden_states, topk_weights=topk_weights, topk_ids=topk_ids
        )
        shared = self.shared_expert.forward(hidden_states)
        return (routed + shared).view(num_tokens, hidden_dim)


class LagunaDecoderLayer(BaseOP):
    """Pre-norm gated-attention + (dense L0 | sparse MoE) decoder layer."""

    def __init__(self, config: "ModelConfig", layer_id: int, expert_quant):
        self.self_attn = LagunaAttention(config, layer_id)
        # mlp_layer_types[0] == "dense" (mlp_only_layers=[0]) -> a dense SwiGLU; the rest are sparse.
        if layer_id < config.first_k_dense_replace:
            self.mlp = GatedMLP(config)  # bf16 (backbone is the unquantized config)
        else:
            self.mlp = LagunaSparseBlock(config, expert_quant)
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


class LagunaModel(BaseOP):
    def __init__(self, config: "ModelConfig", expert_quant):
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size, embedding_dim=config.hidden_size
        )
        self.layers = OPList(
            [
                LagunaDecoderLayer(config, layer_id, expert_quant)
                for layer_id in range(config.num_layers)
            ]
        )
        self.norm = RMSNormFused(size=config.hidden_size, eps=config.rms_norm_eps)
        self._capture_layer_ids: list[int] | None = None

    def set_capture_layers(self, ids: list[int] | None) -> None:
        self._capture_layer_ids = list(ids) if ids else None

    def forward(
        self, input_ids: torch.Tensor, return_hidden: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        x = self.embed_tokens.forward(input_ids)
        residual: torch.Tensor | None = None
        cap = self._capture_layer_ids if return_hidden else None
        cap_set = set(cap) if cap else None
        grabbed: dict[int, torch.Tensor] = {}
        _aux_xr = cap_set is None or os.environ.get("MINISGL_EAGLE3_AUX_MODE", "xr") == "xr"
        for lid, layer in enumerate(self.layers.op_list):
            x, residual = layer.forward(x, residual)
            if cap_set is not None and lid in cap_set:
                grabbed[lid] = (x + residual).clone() if _aux_xr else residual.clone()
        pre_norm = (x + residual).clone() if return_hidden else None
        final = self.norm.forward(x, residual)[0]
        if return_hidden:
            aux_stack = torch.stack([grabbed[i] for i in cap], dim=0) if cap else None
            return final, pre_norm, aux_stack
        return final


class LagunaForCausalLM(BaseLLMModel):
    """poolside/Laguna-XS-2.1 — SWA-hybrid gated-attention MoE with NVFP4 routed + shared experts.

    Only the routed experts + shared expert are quantized (NVFP4); the backbone (gated SWA attention,
    router gate, dense layer-0, embed/norm/untied lm_head) is bf16. Built like Glm4MoeLiteForCausalLM
    (backbone at quant=None, expert quant handed to the MoE blocks) but with LagunaAttention (per-layer
    heads, dual RoPE, sliding-window ring KV pool, per-head output gate) in place of MLA."""

    def __init__(self, config: "ModelConfig"):
        expert_quant = config.quant
        backbone_cfg = dataclasses.replace(config, quant=None)
        self.model = LagunaModel(backbone_cfg, expert_quant)
        config = backbone_cfg
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,  # Laguna: untied (False)
            tied_embedding=None,
        )
        super().__init__()

    def forward(self, return_hidden: bool = False):
        input_ids = get_global_ctx().batch.input_ids
        if return_hidden:
            final, pre_norm, aux_hidden = self.model.forward(input_ids, return_hidden=True)
            return self.lm_head.forward(final), pre_norm, aux_hidden
        return self.lm_head.forward(self.model.forward(input_ids))

    def set_capture_layers(self, ids: list[int] | None) -> None:
        self.model.set_capture_layers(ids)


__all__ = ["LagunaForCausalLM"]
