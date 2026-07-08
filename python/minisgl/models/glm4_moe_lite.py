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

QUANT SPLIT: the routed experts AND the always-on shared expert are quantized (W4A8/AWQ) — real AWQ
checkpoints (QuantTrio/GLM-4.7-Flash-AWQ) quantize the shared expert too, leaving only the MLA
attention, the router gate, the dense layer-0 MLP, and lm_head in bf16 (their
modules_to_not_convert = {self_attn, mlp.gate, layers.0}). The shared expert follows expert_quant,
so a bf16 (non-AWQ) checkpoint keeps it bf16. TP=1 only (AWQ INT4 fits one card); MLA TP sharding is
a follow-up. The MTP head (layers.<num_layers>, num_nextn_predict_layers) is skipped by the loader.
"""
from __future__ import annotations

import dataclasses
import os
from typing import TYPE_CHECKING, Tuple

import torch
import torch.nn.functional as F
from minisgl.core import get_global_ctx
from minisgl.distributed import get_tp_info
from minisgl.layers import (
    BaseOP,
    LinearColParallelMerged,
    LinearOProj,
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
from minisgl.utils import div_even, nvtx_annotate

from .base import BaseLLMModel
from .utils import GatedMLP

if TYPE_CHECKING:
    from .config import ModelConfig


class GLMMLAAttention(BaseOP):
    """Multi-head latent attention. Projections + RoPE + W_UK/W_UV absorption live here; the paged
    latent cache + the mla_hip kernels live in the MLABackend (ctx.attn_backend)."""

    def __init__(self, config: "ModelConfig", layer_id: int):
        self._layer_id = layer_id
        self.qk_nope = config.qk_nope_head_dim
        self.qk_rope = config.qk_rope_head_dim
        self.qk_head_dim = self.qk_nope + self.qk_rope
        self.v_head_dim = config.v_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        eps = config.rms_norm_eps
        # TP: the q/kv up-projections + o_proj are HEAD-parallel (each rank owns num_qo_heads/tp
        # heads). The q_lora/kv_lora bottlenecks and the SHARED MQA latent (kv_a) are replicated —
        # the latent KV cache is shared across heads, so it is replicated too (no per-head split).
        Hfull = config.num_qo_heads
        self.num_heads = H = div_even(Hfull, get_tp_info().size)  # heads on THIS rank

        self.q_a_proj = LinearReplicated(config.hidden_size, config.q_lora_rank, has_bias=False)
        self.q_a_layernorm = RMSNorm(config.q_lora_rank, eps=eps)
        self.q_b_proj = LinearColParallelMerged(
            config.q_lora_rank, [Hfull * self.qk_head_dim], has_bias=False
        )
        self.kv_a_proj_with_mqa = LinearReplicated(
            config.hidden_size, self.kv_lora_rank + self.qk_rope, has_bias=False
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=eps)
        self.kv_b_proj = LinearColParallelMerged(
            self.kv_lora_rank, [Hfull * (self.qk_nope + self.v_head_dim)], has_bias=False
        )
        self.o_proj = LinearOProj(Hfull * self.v_head_dim, config.hidden_size, has_bias=False)

        rc = config.rotary_config
        # RoPE over the 64-dim rope sub-vector only (full rotary on that 64-wide head).
        self.rotary = get_rope(
            head_dim=self.qk_rope,
            rotary_dim=self.qk_rope,
            max_position=rc.max_position,
            base=rc.base,
            # Config-driven, not hardcoded None: a long-context GLM variant (yarn) carries scaling.
            rope_scaling=tuple(rc.scaling.items()) if rc.scaling else None,
            interleave=rc.interleave,  # GLM-4.x uses interleaved RoPE (rope_interleave=True)
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
            # ABSORBED form: q_nope·W_UK -> latent space, attend over the paged latent, then ·W_UV.
            # q_len == 1 -> single-token decode kernel; q_len > 1 -> spec-decode multi-query VERIFY
            # (confirmed + drafts) over the same paged latent (no prefix re-materialization).
            q_absorbed = torch.einsum("thn,hnl->thl", q_nope, self._w_uk)  # [T,H,kv_lora]
            q_full = torch.cat([q_absorbed, q_rope], dim=-1)  # [T,H,kv_lora+rope]
            if metadata.max_seqlen_q == 1:
                o_latent = backend.decode(q_full, self._layer_id, metadata)  # [T,H,kv_lora]
            else:
                o_latent = backend.verify(q_full, self._layer_id, metadata)  # [T,H,kv_lora]
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
                if lat.dtype != q_full.dtype:  # fp8 KV cache -> dequant the stored latent (scale 1.0)
                    lat = lat.to(q_full.dtype)
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
    """Always-on shared expert (SwiGLU). REPLICATED across TP ranks (not sharded): a row-parallel
    down_proj would have K = moe_intermediate/tp = 768, but the W4A8 dense kernel needs K % 512 == 0
    (1536 only un-sharded). Each rank computes the full shared output, which is added to the
    already-all-reduced routed output (no double-count, no extra collective). Quantized with the
    model quant when present (AWQ)."""

    def __init__(self, config: "ModelConfig"):
        inter = config.moe_intermediate_size * max(1, config.n_shared_experts)
        qm = create_linear_method(config.quant)
        self.gate_up_proj = LinearReplicated(
            config.hidden_size, 2 * inter, has_bias=False, quant_method=qm
        )
        self.down_proj = LinearReplicated(inter, config.hidden_size, has_bias=False, quant_method=qm)

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
        # The always-on shared expert follows the routed-expert quant: real AWQ GLM-4.7-Flash
        # checkpoints (e.g. QuantTrio/GLM-4.7-Flash-AWQ) quantize it alongside the routed experts
        # (their modules_to_not_convert keeps only attn + gate + dense layer-0 bf16). When the MoE is
        # NOT quantized (expert_quant is None — bf16 checkpoint), it stays bf16. (This intentionally
        # relaxes commit 7b113b9's "shared expert always bf16" guess, which predated a real AWQ ckpt.)
        self.shared_experts = GLMSharedExpert(dataclasses.replace(config, quant=expert_quant))
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
        # Spec-decode aux capture: decoder-layer ids whose output hidden is stashed (None = off).
        self._capture_layer_ids: list[int] | None = None

    def set_capture_layers(self, ids: list[int] | None) -> None:
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
        # EAGLE3 aux = the full residual stream entering the NEXT layer = x + residual (SGLang
        # captures `hidden_states + residual`). `residual` alone MISSES this layer's mlp output, which
        # the layer returns un-added in `x` (folded into the stream by the next layer's input_norm).
        # MINISGL_EAGLE3_AUX_MODE=r captures `residual` only (diagnostic).
        _aux_xr = cap_set is None or os.environ.get("MINISGL_EAGLE3_AUX_MODE", "xr") == "xr"
        for lid, layer in enumerate(self.layers.op_list):
            x, residual = layer.forward(x, residual)
            if cap_set is not None and lid in cap_set:
                grabbed[lid] = (x + residual).clone() if _aux_xr else residual.clone()
        # MTP seed = the PRE-final-norm residual stream (x + residual), the standard GLM/DeepSeek
        # NextN `previous_hidden_states` input (the MTP's own hnorm re-normalizes it). Snapshot it
        # before self.norm mutates `residual` in place. Only materialized when capturing.
        pre_norm = (x + residual).clone() if return_hidden else None
        final = self.norm.forward(x, residual)[0]
        if return_hidden:
            # stack in the programmed id order so a consumer can index aux by position; None if empty.
            aux_stack = torch.stack([grabbed[i] for i in cap], dim=0) if cap else None
            return final, pre_norm, aux_stack
        return final


class GLMMTPAttention(GLMMLAAttention):
    """MLA attention for the self-contained MTP draft chain.

    Reuses the decoder MLA projections (q_a/q_b, kv_a/kv_b, o_proj, RoPE, W_UK/W_UV absorption via
    post_load) but runs a MATERIALIZED causal attention over the SHORT per-request draft chain
    (<= K tokens, freshly built each propose), never touching the engine's paged latent cache or the
    attn backend. `forward_draft(x, positions)` processes one autoregressive step for all B requests
    (x: [B, hidden]); it appends each step's per-head k/v latent to a running cache the caller owns."""

    def forward_draft(
        self, x: torch.Tensor, positions: torch.Tensor, cache: "list", step: int
    ) -> torch.Tensor:
        # x: [B, hidden] (one MTP token per request); positions: [B] absolute RoPE positions.
        # cache: list growing per step, each entry (k_full [B,H,qk], v [B,H,vhd]); returns [B, hidden].
        T = x.shape[0]
        H, nope, rope, vhd = self.num_heads, self.qk_nope, self.qk_rope, self.v_head_dim
        q = self.q_b_proj.forward(self.q_a_layernorm.forward(self.q_a_proj.forward(x)))
        q = q.view(T, H, self.qk_head_dim)
        q_nope, q_rope = q[..., :nope], q[..., nope:]
        kv = self.kv_a_proj_with_mqa.forward(x)
        c_kv = self.kv_a_layernorm.forward(kv[:, : self.kv_lora_rank].contiguous())
        k_rope = kv[:, self.kv_lora_rank :]
        q_rope, k_rope = self.rotary.forward(
            positions, q_rope.reshape(T, H * rope).contiguous(), k_rope.contiguous()
        )
        q_rope = q_rope.view(T, H, rope)
        # Materialize per-head k_nope / v from the latent (drop the absorption — chain is tiny).
        kvb = self.kv_b_proj.forward(c_kv).view(T, H, nope + vhd)
        k_nope, v = kvb[..., :nope], kvb[..., nope:]  # [T,H,nope], [T,H,vhd]
        k_full = torch.cat([k_nope, k_rope.unsqueeze(1).expand(T, H, rope)], dim=-1)  # [T,H,qk]
        q_full = torch.cat([q_nope, q_rope], dim=-1)  # [T,H,qk]
        cache.append((k_full, v))
        # Causal attention over the chain so far (steps 0..step). Stack -> [S,T,H,*].
        Ks = torch.stack([c[0] for c in cache], dim=0)  # [S,T,H,qk]
        Vs = torch.stack([c[1] for c in cache], dim=0)  # [S,T,H,vhd]
        # scores[t,h,s] = q[t,h]·k[s,t,h]; per (t,h): attend keys 0..step (all causal, current incl.).
        scores = torch.einsum("thd,sthd->ths", q_full, Ks) * self.scale_attn  # [T,H,S]
        probs = scores.softmax(dim=-1).to(Vs.dtype)
        o = torch.einsum("ths,sthd->thd", probs, Vs)  # [T,H,vhd]
        return self.o_proj.forward(o.reshape(T, H * vhd))

    def seed_kv(
        self, x: torch.Tensor, positions: torch.Tensor
    ) -> "list[Tuple[torch.Tensor, torch.Tensor]]":
        """Per-position (k_full, v) for a batch of prompt positions WITHOUT attention — used to SEED
        the persistent MTP draft KV from the prompt prefill (MTPProposer.seed_prefill). The q/k/v +
        RoPE math MUST mirror ``forward_draft`` exactly (keep in sync); only the attention is dropped
        (the cache stores k/v; attention runs at propose time over the stacked cache).

        x: [S, hidden] (already ``input_layernorm``'d, like forward_draft's input); positions: [S].
        Returns a list of S ``(k_full [1,H,qk], v [1,H,vhd])`` entries — exactly what forward_draft
        appends, so the proposer can stack them directly."""
        S = x.shape[0]
        H, nope, rope, vhd = self.num_heads, self.qk_nope, self.qk_rope, self.v_head_dim
        q = self.q_b_proj.forward(self.q_a_layernorm.forward(self.q_a_proj.forward(x)))
        q = q.view(S, H, self.qk_head_dim)
        q_rope = q[..., nope:]
        kv = self.kv_a_proj_with_mqa.forward(x)
        c_kv = self.kv_a_layernorm.forward(kv[:, : self.kv_lora_rank].contiguous())
        k_rope = kv[:, self.kv_lora_rank :]
        _, k_rope = self.rotary.forward(
            positions, q_rope.reshape(S, H * rope).contiguous(), k_rope.contiguous()
        )
        kvb = self.kv_b_proj.forward(c_kv).view(S, H, nope + vhd)
        k_nope, v = kvb[..., :nope], kvb[..., nope:]  # [S,H,nope], [S,H,vhd]
        k_full = torch.cat([k_nope, k_rope.unsqueeze(1).expand(S, H, rope)], dim=-1)  # [S,H,qk]
        return [(k_full[s : s + 1], v[s : s + 1]) for s in range(S)]

    def post_load(self) -> None:
        super().post_load()
        self.scale_attn = float(self.qk_head_dim) ** -0.5


class GLMMTPHead(BaseOP):
    """GLM-4.x MTP (next-token-prediction) self-speculation head — a FULL MLA+MoE decoder layer at
    model.layers.<num_layers> plus its own untied embed/lm_head and the enorm/hnorm/eh_proj fuser:

        h_mtp = layer( eh_proj( concat[ enorm(embed(tok)), hnorm(last_hidden) ] ) )
        logits = shared_head.head( shared_head.norm(h_mtp) )

    Run K times autoregressively (own short draft chain, no paged KV); see MTPProposer."""

    def __init__(self, config: "ModelConfig", layer_id: int, expert_quant):
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size, embedding_dim=config.hidden_size
        )
        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # eh_proj: concat[enorm(e), hnorm(h)] (2*hidden) -> hidden. Replicated (no TP split).
        self.eh_proj = LinearReplicated(2 * config.hidden_size, config.hidden_size, has_bias=False)
        # The MTP decoder layer mirrors GLMDecoderLayer but uses the draft MLA attention.
        self.self_attn = GLMMTPAttention(config, layer_id)
        self.mlp = GLMSparseBlock(config, expert_quant)
        self.input_layernorm = RMSNormFused(size=config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNormFused(
            size=config.hidden_size, eps=config.rms_norm_eps
        )
        self.shared_head = GLMMTPSharedHead(config)
        self._layer_id = layer_id

    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens.forward(tokens)

    def fuse(self, embed_e: torch.Tensor, last_hidden: torch.Tensor) -> torch.Tensor:
        # concat[ enorm(e), hnorm(h) ] -> eh_proj -> hidden
        e = self.enorm.forward(embed_e)
        h = self.hnorm.forward(last_hidden)
        return self.eh_proj.forward(torch.cat([e, h], dim=-1))

    def step(
        self, fused: torch.Tensor, positions: torch.Tensor, cache: "list", step: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One MTP decoder-layer step over the fused [B, hidden] input. Returns (logits, hidden)
        where hidden feeds the NEXT step's hnorm and logits gives the next draft token."""
        # GLMDecoderLayer-shaped: input_layernorm(no residual on the fused input) -> attn ->
        # post_attention_layernorm(residual) -> mlp(residual). The fused vector is the layer input.
        x, residual = self.input_layernorm.forward(fused, None)
        x = self.self_attn.forward_draft(x, positions, cache, step)
        x, residual = self.post_attention_layernorm.forward(x, residual)
        x = self.mlp.forward(x)
        hidden = x + residual  # residual stream after the layer
        logits = self.shared_head.forward(hidden)
        return logits, hidden

    @torch.inference_mode()
    def seed_kv(
        self, tokens: torch.Tensor, prev_hidden: torch.Tensor, positions: torch.Tensor
    ) -> "list[Tuple[torch.Tensor, torch.Tensor]]":
        """Seed the persistent MTP draft KV from the prompt: for each prompt position build the same
        fused layer input ``step`` would, then compute its k/v (no attention). Returns the list of
        (k_full, v) entries the proposer stacks into its per-uid cache.

        tokens: [S] (the token at each seeded position p); prev_hidden: [S, hidden] (the target hidden
        h_{p-1} that produced it — the standard MTP ``previous_hidden_states``); positions: [S] RoPE
        positions (= p). Mirrors ``step``'s fuse + input_layernorm before the attention's seed_kv."""
        fused = self.fuse(self.embed(tokens), prev_hidden)
        x = self.input_layernorm.forward(fused, None)[0]
        return self.self_attn.seed_kv(x, positions)


class GLMMTPSharedHead(BaseOP):
    """The MTP head's own (untied) final norm + lm_head."""

    def __init__(self, config: "ModelConfig"):
        self.norm = RMSNormFused(size=config.hidden_size, eps=config.rms_norm_eps)
        self.head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=False,
            tied_embedding=None,
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        normed = self.norm.forward(hidden, None)[0]
        # Full-vocab logits over all rows (TP all_gather, no prefill reduction) — every rank MUST see
        # the SAME full-vocab argmax or the per-rank drafts desync the verify batch.
        return self.head.logits_all_rows(normed)


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
        # MTP self-speculation head (model.layers.<num_layers>). Built only when the checkpoint
        # ships one (num_nextn_predict_layers>0); the routed experts in its MoE follow expert_quant.
        self.mtp = (
            GLMMTPHead(backbone_cfg, layer_id=config.num_layers, expert_quant=expert_quant)
            if config.num_nextn_predict_layers > 0
            else None
        )
        super().__init__()

    def forward(self, return_hidden: bool = False):
        input_ids = get_global_ctx().batch.input_ids
        if return_hidden:
            # The model returns (post-norm hidden for lm_head, pre-norm residual for MTP, aux). The
            # MTP seed is the PRE-final-norm residual stream (last_hidden); lm_head uses the post-norm.
            final, pre_norm, aux_hidden = self.model.forward(input_ids, return_hidden=True)
            return self.lm_head.forward(final), pre_norm, aux_hidden
        return self.lm_head.forward(self.model.forward(input_ids))

    def set_capture_layers(self, ids: list[int] | None) -> None:
        self.model.set_capture_layers(ids)


__all__ = ["Glm4MoeLiteForCausalLM"]
