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
from minisgl._hip_engage import engaged
from minisgl.layers.minv import minv_linear
from minisgl.quant import create_linear_method, kernels
from minisgl.utils import div_even, nvtx_annotate

from .base import BaseLLMModel
from .utils import GatedMLP, mlp_accepts_producer_actquant, norm_then_mlp

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
        # NB: the two einsums below stay on rocBLAS DELIBERATELY. They are batched-over-heads
        # per-head projections, so the 2-D `minv` seam cannot express them, and the obvious
        # conclusion from the kernel sweep — "Tensile runs them at 7.8% occupancy, our GEMV runs the
        # same band at 75-100%, therefore route them" — was BUILT AND MEASURED, and is FALSE. See
        # the rejected-lever note in rdna4-hip-kernels/KERNEL_CORE_POLICY.md for the numbers.

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
            engaged("torch.einsum(ROCBLAS_BMM:mla_absorb_uk)")
            q_absorbed = torch.einsum("thn,hnl->thl", q_nope, self._w_uk)  # [T,H,kv_lora]
            q_full = torch.cat([q_absorbed, q_rope], dim=-1)  # [T,H,kv_lora+rope]
            if metadata.max_seqlen_q == 1:
                o_latent = backend.decode(q_full, self._layer_id, metadata)  # [T,H,kv_lora]
            else:
                o_latent = backend.verify(q_full, self._layer_id, metadata)  # [T,H,kv_lora]
            engaged("torch.einsum(ROCBLAS_BMM:mla_absorb_uv)")
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
                if lat.dtype != q_full.dtype:
                    # fp8 latent cache -> DEQUANT with this layer's calibrated descale, the same one
                    # the store divided by. A bare `.to(bf16)` here reads every cached token 1/scale
                    # too small; it is only invisible while the scale happens to be 1.0, which is
                    # exactly the shape of the SWA-ring bug the MHA path already had.
                    lat = lat.to(q_full.dtype) * ctx.kv_cache.latent_descale[self._layer_id].to(
                        q_full.dtype
                    )
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
        # NOT F.linear. This is a bf16 [T,2048] x [64,2048] with no bias — precisely what
        # minv_linear's decode arm routes to the shared dense GEMV — but it was written against
        # a bare `torch.empty` weight rather than a Linear, so it bypassed the dispatch seam
        # entirely and landed on a rocBLAS Tensile MT16x16x32 solution that fields FOUR workgroups
        # (0.2% occupancy) on a 64-CU part: ~13.5 us of essentially pure launch overhead, once per
        # MoE layer per step. minv_linear keeps its own F.linear fallback for shapes the GEMV
        # declines, so this is a routing change, not a new constraint.
        return minv_linear(x, self.weight)  # [T, E] logits


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
        # fused gate_up + silu at decode (bit-exact); falls back to silu_and_mul(forward) otherwise.
        return self.down_proj.forward(self.gate_up_proj.forward_swiglu(x))


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
        if self.n_group <= 1:
            # Every checkpoint we serve (zai-org/GLM-4.7-Flash and QuantTrio/GLM-4.7-Flash-AWQ both
            # ship n_group=1) takes this path: the group-topk block below is DEAD on them, and the
            # remaining chain is byte-for-byte Laguna's. ONE launch instead of twelve.
            #
            # Both tensors go in at their NATIVE dtype, and THE ALREADY-TRUNCATED BIAS IS DELIBERATE
            # — this is bit-identical to the torch chain below. The checkpoint ships
            # e_score_correction_bias as F32, but engine.py:608 downcasts it to the MODEL dtype
            # because there is no special case for it (QuantTrio/GLM-4.7-Flash-AWQ is fp16), and the
            # model-side buffer is at the model dtype too — so the `.float()` below widens a value
            # that was already truncated. Feeding the F32 checkpoint value would change expert
            # SELECTION on near-ties: different, arguably better, and invisible to a tok/s A/B.
            # Raise it as its own commit if at all.
            return kernels.moe_route_sigmoid_bias(
                logits,
                self.gate.e_score_correction_bias,
                self.top_k,
                self.norm_topk_prob,
                self.routed_scaling_factor,
            )
        # n_group > 1: DeepSeek-style group-limited routing. The kernel has no group-topk policy, so
        # this stays a LIVE torch fallback — not an assert. A future checkpoint gets a slow-but-
        # correct serve instead of a crash, and never a silently wrong router.
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

    def accepts_producer_actquant(self) -> bool:
        """Can this block consume the feeding RMSNorm's (x_fp8, act_scales) pair?

        GLM differs from Qwen3.5-MoE here: real AWQ GLM-4.7-Flash checkpoints quantize the SHARED
        expert alongside the routed ones, so in principle one producer could feed two consumers.
        Only the routed-expert GEMM1 is wired today — the shared expert is a dense `LinearTP` and
        the dense linear layers do not take the pair yet (`LinearMethod.apply` has no parameter for
        it). That second consumer is the obvious follow-up and it is free once the dense seam exists.
        """
        e = self.experts
        return bool(getattr(e._moe_method, "supports_producer_actquant", False)) and not e.enable_ep

    @nvtx_annotate("MoE")
    def forward(self, hidden_states: torch.Tensor,
                x_fp8: torch.Tensor | None = None,
                act_scales: torch.Tensor | None = None) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        if x_fp8 is not None and x_fp8.shape[0] != hidden_states.shape[0]:
            raise AssertionError(
                f"producer pair has {x_fp8.shape[0]} rows, hidden_states has {hidden_states.shape[0]}"
            )
        topk_weights, topk_ids = self._noaux_tc(self.gate.forward(hidden_states))
        routed = self.experts.forward(
            hidden_states, topk_weights=topk_weights, topk_ids=topk_ids,
            x_fp8=x_fp8, act_scales=act_scales,
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
        # Decided ONCE at construction: quant scheme and EP topology are fixed at load.
        self._fuse_actquant = mlp_accepts_producer_actquant(self.mlp)

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, residual = self.input_layernorm.forward(x, residual)
        x = self.self_attn.forward(x)
        x, residual = norm_then_mlp(self.post_attention_layernorm, self.mlp, x, residual,
                                    fuse_actquant=self._fuse_actquant)
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
    """MLA attention for the self-contained MTP draft chain, on the ``mla_hip.mla_decode`` kernel.

    Reuses the decoder MLA projections (q_a/q_b, kv_a/kv_b, o_proj, RoPE, W_UK/W_UV absorption via
    post_load) and runs the ABSORBED form — the same one the target's decode uses — over a persistent
    per-slot LATENT draft ring owned by the proposer (spec/mtp.py). It never touches the engine's paged
    latent cache or the attn backend.

    WHY LATENT, NOT MATERIALIZED. The draft ring used to hold per-head materialized K [H, qk] and V
    [H, v] per token (TP=2 GLM-4.7-Flash: 10 x (256 + 256) = 5120 elements) and attended in PLAIN
    TORCH: ``k_buf[slot_rows]`` gathered (copied) the slot's whole ring every draft step, then einsum +
    softmax over all R columns with an additive -inf mask — work sized from the ring's CAPACITY, not the
    live draft context. The latent row is [c_KV (kv_lora) | roped k_rope] = 576 elements shared by
    every head (8.9x fewer bytes/key), the kv_b_proj up-projection of the new token disappears from
    both the step and the prompt seed, and ``mla_decode`` reads the ring IN PLACE (page_size-1 pages,
    block table from ``DraftAttnMeta``) over exactly the live keys. Absorbed == materialized
    mathematically: q_nope·(W_UK c) = (q_nope W_UK)·c and Σp·(W_UV c) = W_UV·(Σp c)."""

    def draft_buffer_dims(self) -> "tuple[int, int, int, int]":
        """(n_k_heads, k_dim, n_v_heads, v_dim) for the GLOBAL persistent draft ring the proposer
        allocates. The LATENT form stores ONE shared [c_KV | k_rope] row per token (1 "head", 576
        wide) and NO separate V: mla_decode reads V as the latent's first kv_lora_rank dims. So the V
        buffer is zero-sized and nothing indexes it."""
        return 1, self.kv_lora_rank + self.qk_rope, 0, 0

    def _latent(self, x: torch.Tensor, positions: torch.Tensor) -> "Tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
        """q_nope [T,H,nope], q_rope [T,H,rope] (roped) and latent [T, kv_lora+rope] (normed c_KV ‖
        roped k_rope) — the exact rows GLMMLAAttention.forward stores in the target's latent cache."""
        T = x.shape[0]
        H, nope, rope = self.num_heads, self.qk_nope, self.qk_rope
        q = self.q_b_proj.forward(self.q_a_layernorm.forward(self.q_a_proj.forward(x)))
        q = q.view(T, H, self.qk_head_dim)
        q_nope, q_rope = q[..., :nope], q[..., nope:]
        kv = self.kv_a_proj_with_mqa.forward(x)
        c_kv = self.kv_a_layernorm.forward(kv[:, : self.kv_lora_rank].contiguous())
        q_rope, k_rope = self.rotary.forward(
            positions, q_rope.reshape(T, H * rope).contiguous(), kv[:, self.kv_lora_rank :].contiguous()
        )
        return q_nope, q_rope.view(T, H, rope), torch.cat([c_kv, k_rope], dim=-1)

    def forward_draft_masked(
        self,
        x: torch.Tensor,           # [B, hidden] — ONE draft token per row (post input_layernorm)
        positions: torch.Tensor,   # [B] absolute RoPE position per row
        k_buf: torch.Tensor,       # [max_slots, R, 1, kv_lora+rope] GLOBAL persistent LATENT draft ring
        v_buf: torch.Tensor,       # zero-sized (see draft_buffer_dims) — unused
        slot_rows: torch.Tensor,   # [B] slot (= req.table_idx) per row
        write_col: torch.Tensor,   # [B] ring column this token's latent is written at, per row
        meta,                      # spec.draft_attn.DraftAttnMeta: block_table [B,R] i32, ctx_lens [B] i32
    ) -> torch.Tensor:
        """One capturable draft step: store this token's latent at (slot_rows, write_col), then absorbed
        MLA decode over the row's visible keys (``meta``, built on device by the proposer from its keep
        mask — exactly the keys the old -inf mask kept). Static shapes, no host reads."""
        B = x.shape[0]
        H, vhd = self.num_heads, self.v_head_dim
        q_nope, q_rope, latent = self._latent(x, positions)
        k_buf[slot_rows, write_col, 0] = latent
        engaged("torch.einsum(ROCBLAS_BMM:mla_absorb_uk)")
        q_abs = torch.einsum("thn,hnl->thl", q_nope, self._w_uk)       # [B,H,kv_lora]
        q_full = torch.cat([q_abs, q_rope], dim=-1).contiguous()       # [B,H,kv_lora+rope]
        ms, R, _, D = k_buf.shape
        engaged("mla_hip.mla_decode(draft)")
        o_latent = self._mla_decode(q_full, k_buf.view(ms * R, 1, D), meta.block_table,
                                    meta.ctx_lens, self.scale_attn, 0, 0, self.qk_rope)  # [B,H,kv_lora]
        engaged("torch.einsum(ROCBLAS_BMM:mla_absorb_uv)")
        o = torch.einsum("thl,hdl->thd", o_latent, self._w_uv)          # [B,H,v]
        return self.o_proj.forward(o.reshape(B, H * vhd))

    def seed_kv_masked(
        self,
        x: torch.Tensor,          # [S, hidden] — post input_layernorm prompt-prefix rows
        positions: torch.Tensor,  # [S] absolute RoPE position per row
        k_buf: torch.Tensor,      # [max_slots, R, 1, kv_lora+rope] GLOBAL persistent LATENT draft ring
        v_buf: torch.Tensor,      # zero-sized — unused
        slot: int,                # slot (= req.table_idx) to seed
        start_col: int,           # first column to write
    ) -> None:
        """Seed the latent ring from the prompt prefix WITHOUT attention. Only the latent is needed —
        no q projection, no kv_b_proj — and k_rope is rotated alone (``forward_one`` is bit-identical
        to the key half of ``forward``), so it matches what forward_draft_masked stores exactly."""
        S = x.shape[0]
        kv = self.kv_a_proj_with_mqa.forward(x)
        c_kv = self.kv_a_layernorm.forward(kv[:, : self.kv_lora_rank].contiguous())
        k_rope = self.rotary.forward_one(positions, kv[:, self.kv_lora_rank :].contiguous())
        k_buf[slot, start_col : start_col + S, 0] = torch.cat([c_kv, k_rope], dim=-1)

    def post_load(self) -> None:
        super().post_load()
        self.scale_attn = float(self.qk_head_dim) ** -0.5
        # Resolved at LOAD: a missing mla_hip build must fail the boot, never fall back to torch.
        import mla_hip
        self._mla_decode = mla_hip.mla_decode


class GLMMTPHead(BaseOP):
    """GLM-4.x MTP (next-token-prediction) self-speculation head — a FULL MLA+MoE decoder layer at
    model.layers.<num_layers> plus its own untied embed/lm_head and the enorm/hnorm/eh_proj fuser:

        h_mtp = layer( eh_proj( concat[ enorm(embed(tok)), hnorm(last_hidden) ] ) )
        logits = shared_head.head( shared_head.norm(h_mtp) )

    Run K+1 times per propose over a persistent LATENT draft ring (see MTPProposer)."""

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
        self._fuse_actquant = mlp_accepts_producer_actquant(self.mlp)
        self.hidden_size = config.hidden_size  # for the buffered-propose seed buffer alloc (spec/mtp.py)

    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens.forward(tokens)

    def fuse(self, embed_e: torch.Tensor, last_hidden: torch.Tensor) -> torch.Tensor:
        # concat[ enorm(e), hnorm(h) ] -> eh_proj -> hidden
        e = self.enorm.forward(embed_e)
        h = self.hnorm.forward(last_hidden)
        return self.eh_proj.forward(torch.cat([e, h], dim=-1))

    def step_masked(
        self,
        fused: torch.Tensor,
        positions: torch.Tensor,
        k_buf: torch.Tensor,
        v_buf: torch.Tensor,
        slot_rows: torch.Tensor,
        write_col: torch.Tensor,
        meta,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One capturable MTP decoder-layer step over the fused [B, hidden] input: input_layernorm ->
        forward_draft_masked (mla_hip decode over the GLOBAL latent draft ring, visible keys given by
        ``meta``) -> post_attention_layernorm + MoE -> own norm + lm-head. Returns (logits, hidden);
        hidden feeds the NEXT step's hnorm. Drives the K-step chain in MTPProposer.propose_body."""
        x, residual = self.input_layernorm.forward(fused, None)
        x = self.self_attn.forward_draft_masked(
            x, positions, k_buf, v_buf, slot_rows, write_col, meta)
        x, residual = norm_then_mlp(self.post_attention_layernorm, self.mlp, x, residual,
                                    fuse_actquant=self._fuse_actquant)
        hidden = x + residual
        logits = self.shared_head.forward(hidden)
        return logits, hidden

    @torch.inference_mode()
    def seed_buffered(
        self,
        tokens: torch.Tensor,
        prev_hidden: torch.Tensor,
        positions: torch.Tensor,
        k_buf: torch.Tensor,
        v_buf: torch.Tensor,
        slot: int,
        start_col: int,
    ) -> None:
        """Seed the GLOBAL latent draft ring from the prompt prefix (no attention): for each prompt
        position build the same fused layer input step_masked would (fuse + input_layernorm), then
        store its latent at k_buf[slot, start_col:]. tokens [S] = token at position p; prev_hidden
        [S, hidden] = the target hidden h_{p-1} that produced it; positions [S] = p."""
        fused = self.fuse(self.embed(tokens), prev_hidden)
        x = self.input_layernorm.forward(fused, None)[0]
        self.self_attn.seed_kv_masked(x, positions, k_buf, v_buf, slot, start_col)


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
        # MTP self-speculation head (model.layers.<num_layers>). Built only when the checkpoint ships
        # one (num_nextn_predict_layers>0). Its MoE follows the checkpoint's precision: quantized only
        # if the head's module is NOT in the quant ignore list. A checkpoint may keep the MTP head
        # bf16/fp16 on a quantized backbone (some GLM checkpoints keep layers.<num_layers>
        # unquantized) — then
        # build it unquantized so its full-precision weights load (QuantConfig.is_module_quantized).
        mtp_quant = expert_quant
        if expert_quant is not None and not expert_quant.is_module_quantized(
            f"model.layers.{config.num_layers}.mlp.experts.0.gate_proj"
        ):
            mtp_quant = None
        self.mtp = (
            GLMMTPHead(backbone_cfg, layer_id=config.num_layers, expert_quant=mtp_quant)
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
