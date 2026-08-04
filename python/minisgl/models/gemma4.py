"""Gemma4 — the shared backbone of `gemma4` (autoregressive) and `diffusion_gemma` (block
diffusion). Both checkpoints carry the SAME 30-layer stack; only the head and the namespace differ
(`model.language_model.*` vs `model.decoder.*`), so everything structural lives here and the heads
stay thin.

Five things about this architecture diverge from every other model in the repo, and four of the five
fail SILENTLY — plausible, grammatical output that is quietly wrong. They are called out at each
site, but collected here because they are what a reviewer should check first:

  1. The two layer types have DIFFERENT head_dim AND kv-head counts — 25 sliding layers at
     (16 q, 8 kv, D=256) and 5 full layers at (16 q, 2 kv, D=512). Not just a per-layer head COUNT
     the way Laguna varies it. ModelConfig carries the full geometry in `head_dim`/`num_kv_heads`
     (they size the main paged pool) and the sliding geometry in `swa_head_dim`/`swa_num_kv_heads`.
  2. The softmax scale is 1.0, NOT 1/sqrt(head_dim) (reference `self.scaling = 1.0`). The
     temperature is folded into the LEARNED k_norm, whose weight is a near-constant 0.1260 on
     sliding / 0.0623 on full layers. Applying head_dim**-0.5 on top roughly doubles or halves the
     logit temperature depending on the layer type.
  3. The full-attention layers ship NO v_proj (`attention_k_eq_v`). V is not an alias of the cached
     K — it is the PRE-norm, PRE-RoPE k_proj output passed through an UNWEIGHTED RMSNorm, while K is
     k_norm'd and RoPE'd. Both must be materialised and cached.
  4. The dense MLP and the routed experts are PARALLEL branches off the same residual, summed, under
     three separate norms (post_feedforward_layernorm_1 on the dense output,
     post_feedforward_layernorm_2 on the MoE output, post_feedforward_layernorm on their sum). The
     router reads the RAW residual, not either branch's normalized input.
  5. `layer_scalar` rescales the ENTIRE residual stream at the end of every layer (~0.7 typical,
     0.0986 at layer 0). Dropping it compounds to ~0.7^30 and destroys the model.

Not applicable to the shipping checkpoints, deliberately not implemented: per-layer-embeddings
(`hidden_size_per_layer_input` is 0) and KV sharing (`num_kv_shared_layers` is 0). Both short-circuit
in the reference. The vision tower is skipped by the loader — this engine is text-only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple

import torch
from minisgl.core import get_global_ctx
from minisgl.layers import (
    AttentionLayer,
    BaseOP,
    LinearColParallelMerged,
    LinearOProj,
    LinearRowParallel,
    MoELayer,
    OPList,
    ParallelLMHead,
    RMSNorm,
    VocabParallelEmbedding,
    gelu_tanh_and_mul,
)
from minisgl.layers.norm import RMSNormNoScale
from minisgl.quant import create_linear_method
from minisgl.utils import div_even, nvtx_annotate

from .base import BaseLLMModel

if TYPE_CHECKING:
    from minisgl.quant.config import QuantConfig

    from .config import ModelConfig, RotaryConfig


@dataclass(frozen=True)
class Gemma4LayerPlan:
    """Per-layer attention descriptor, derived purely from ModelConfig — no model-name branch."""

    is_sliding: bool
    num_kv_heads: int
    head_dim: int
    sliding_window: int  # 0 for a full layer
    kv_id: int  # compact index: full -> main pool; sliding -> SWA ring pool
    rotary_config: "RotaryConfig"
    has_v_proj: bool  # False on the full layers (attention_k_eq_v)


def gemma4_layer_plan(config: "ModelConfig", layer_id: int) -> Gemma4LayerPlan:
    """Full layers: 2 kv heads at head_dim 512, the proportional RoPE, the full-context main pool.
    Sliding layers: 8 kv heads at head_dim 256, the default RoPE, a 1024-token ring pool. The two
    compact kv id spaces are disjoint because the two pools are physically separate tensors."""
    assert config.layer_types is not None, "Gemma4 requires a per-layer attention schedule"
    is_sliding = config.layer_types[layer_id] == "sliding_attention"
    if is_sliding:
        return Gemma4LayerPlan(
            is_sliding=True,
            # swa_* are None when a checkpoint gives both layer types one geometry; fall back rather
            # than assume the split, so a future uniform Gemma4 variant still builds.
            num_kv_heads=config.swa_num_kv_heads or config.num_kv_heads,
            head_dim=config.swa_head_dim or config.head_dim,
            sliding_window=config.sliding_window or 0,
            kv_id=config.swa_layer_ids.index(layer_id),
            rotary_config=config.sliding_rotary_config or config.rotary_config,
            has_v_proj=True,
        )
    return Gemma4LayerPlan(
        is_sliding=False,
        num_kv_heads=config.num_kv_heads,
        head_dim=config.head_dim,
        sliding_window=0,
        kv_id=config.full_attn_layer_ids.index(layer_id),
        rotary_config=config.rotary_config,
        # `attention_k_eq_v` drops v_proj on the GLOBAL layers only; the sliding layers keep theirs.
        has_v_proj=not config.attention_k_eq_v,
    )


class Gemma4Attention(BaseOP):
    """GQA with per-layer head_dim, per-layer RoPE, per-layer window, and — on the full layers —
    no v_proj at all."""

    def __init__(self, config: "ModelConfig", layer_id: int):
        plan = gemma4_layer_plan(config, layer_id)
        head_dim, nkv = plan.head_dim, plan.num_kv_heads
        nqo = config.num_qo_heads
        q = config.quant
        prefix = f"model.layers.{layer_id}.self_attn"

        def _method(module: str):
            name = f"{prefix}.{module}"
            return create_linear_method(
                q, quantized=q is not None and q.is_module_quantized(name)
            )

        self.q_proj = LinearColParallelMerged(
            config.hidden_size, [nqo * head_dim], has_bias=False, quant_method=_method("q_proj")
        )
        self.k_proj = LinearColParallelMerged(
            config.hidden_size, [nkv * head_dim], has_bias=False, quant_method=_method("k_proj")
        )
        # Building a v_proj the checkpoint does not ship would fail the loader's exact-key check,
        # which is the intended behaviour: the absence of the tensor IS the architecture signal.
        self.v_proj = (
            LinearColParallelMerged(
                config.hidden_size, [nkv * head_dim], has_bias=False,
                quant_method=_method("v_proj"),
            )
            if plan.has_v_proj
            else None
        )
        self.q_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)
        # `with_scale=False` in the reference: no weight tensor exists in the checkpoint, and V is
        # normalized on EVERY layer — including the sliding ones that do have a v_proj.
        self._v_norm = RMSNormNoScale(eps=config.rms_norm_eps)
        self.attn = AttentionLayer(
            layer_id=plan.kv_id,
            head_dim=head_dim,
            num_qo_heads=nqo,
            num_kv_heads=nkv,
            rotary_config=plan.rotary_config,
            q_norm=self.q_norm,  # applied BEFORE RoPE, matching the reference op order
            k_norm=self.k_norm,
            sliding_window=plan.sliding_window,
        )
        self.o_proj = LinearOProj(
            head_dim * nqo, config.hidden_size, has_bias=False, quant_method=_method("o_proj")
        )
        self._head_dim = head_dim
        # Local kv-head count after TP sharding; AttentionLayer already computed it (and may have
        # REPLICATED rather than split when nkv < tp_size, which the full layers' 2 heads can hit).
        self._nkv_local = self.attn.num_kv_heads
        self.plan = plan

    @nvtx_annotate("MHA")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        q = self.q_proj.forward(x)
        k = self.k_proj.forward(x)
        # On a full layer V reuses k_proj's OUTPUT, not the cached key: the key that reaches the KV
        # pool has since been k_norm'd (a learned gain) and RoPE'd, neither of which V gets. Reading
        # it back off the key would silently rotate the values.
        v_src = self.v_proj.forward(x) if self.v_proj is not None else k
        v = self._v_norm.forward(v_src.view(n, self._nkv_local, self._head_dim)).view(n, -1)
        # AttentionLayer q/k-norms and RoPEs the q,k slices in place; v is a copy made by the cat,
        # so the pre-RoPE value computed above survives untouched.
        o = self.attn.forward(torch.cat([q, k, v], dim=-1))
        return self.o_proj.forward(o)


class Gemma4Router(BaseOP):
    """Softmax-over-all-experts router with two learned rescalings that are easy to miss.

    `scale` is a per-HIDDEN-DIM vector applied after an unweighted RMSNorm and further multiplied by
    hidden_size**-0.5; `per_expert_scale` is a per-EXPERT vector applied to the top-k weights AFTER
    they are renormalized to sum to 1 — so the final gate weights deliberately do NOT sum to 1.
    Order matters: renormalize first, then scale."""

    def __init__(self, hidden_size: int, num_experts: int, eps: float):
        self.weight = torch.empty(num_experts, hidden_size)  # <- checkpoint `router.proj.weight`
        self.scale = torch.empty(hidden_size)
        self.per_expert_scale = torch.empty(num_experts)
        self._norm = RMSNormNoScale(eps=eps)
        self._scalar_root_size = hidden_size**-0.5
        self._top_k = 0  # set by the block (top_k is a model-level config, not a router weight)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        from minisgl.layers.minv import minv_linear

        h = self._norm.forward(x)
        h = h * self.scale * self._scalar_root_size
        # M-invariant GEMM, as for every other router in this repo: these logits drive top-k expert
        # SELECTION, so a ~1-ULP M-dependence between a cold forward and a prefix-reused one could
        # flip a near-tie and route to a different expert.
        logits = minv_linear(h, self.weight)
        # softmax is monotonic, so top-k SELECTION is identical whether it runs here or on the raw
        # logits; fp32 only tightens the weights the kernel will combine with.
        probs = torch.softmax(logits.float(), dim=-1)
        w, idx = torch.topk(probs, k=self._top_k, dim=-1)
        w = w / w.sum(dim=-1, keepdim=True)
        return w * self.per_expert_scale.float()[idx], idx


class Gemma4DenseMLP(BaseOP):
    """The always-on dense branch (intermediate_size 2112). Unquantized in the shipping checkpoints
    — the whole `mlp.*` namespace is in the compressed-tensors ignore list — but gated on the config
    rather than hard-coded, so a future fully-quantized Gemma4 builds correctly."""

    def __init__(self, config: "ModelConfig", layer_id: int):
        q = config.quant
        name = f"model.layers.{layer_id}.mlp.gate_proj"
        qm = create_linear_method(
            q, quantized=q is not None and q.is_module_quantized(name)
        )
        inter = config.intermediate_size
        self.gate_up_proj = LinearColParallelMerged(
            config.hidden_size, [inter, inter], has_bias=False, quant_method=qm
        )
        self.down_proj = LinearRowParallel(
            inter, config.hidden_size, has_bias=False, quant_method=qm
        )

    def forward(self, x: torch.Tensor, reduce: bool = True) -> torch.Tensor:
        # The checkpoint declares `hidden_activation: gelu_pytorch_tanh`, i.e. the TANH approximation.
        # `gelu_and_mul` is the exact erf gelu — close enough to look right and never to fail, which
        # is precisely why it must not be used here. Same activation as the routed experts.
        return self.down_proj.forward(gelu_tanh_and_mul(self.gate_up_proj.forward(x)), reduce=reduce)


class Gemma4DecoderLayer(BaseOP):
    """Attribute names mirror the checkpoint's module paths 1:1, so the weight loader is a namespace
    rewrite plus expert stacking rather than a mapping table."""

    def __init__(self, config: "ModelConfig", layer_id: int, expert_quant: "QuantConfig | None"):
        eps, hidden = config.rms_norm_eps, config.hidden_size
        self.input_layernorm = RMSNorm(hidden, eps=eps)
        self.self_attn = Gemma4Attention(config, layer_id)
        self.post_attention_layernorm = RMSNorm(hidden, eps=eps)

        self.pre_feedforward_layernorm = RMSNorm(hidden, eps=eps)
        self.mlp = Gemma4DenseMLP(config, layer_id)
        self.post_feedforward_layernorm_1 = RMSNorm(hidden, eps=eps)

        self.pre_feedforward_layernorm_2 = RMSNorm(hidden, eps=eps)
        self.router = Gemma4Router(hidden, config.num_experts, eps)
        self.router._top_k = config.num_experts_per_tok
        self.experts = MoELayer(
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=hidden,
            intermediate_size=config.moe_intermediate_size,
            # The router already renormalized AND applied per_expert_scale; a second renormalize in
            # the kernel would divide that scale straight back out.
            renormalize=False,
            activation="gelu",
            quant=expert_quant,
        )
        self.post_feedforward_layernorm_2 = RMSNorm(hidden, eps=eps)

        self.post_feedforward_layernorm = RMSNorm(hidden, eps=eps)
        self.layer_scalar = torch.empty(1)

    @nvtx_annotate("Layer")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.input_layernorm.forward(x)
        h = self.self_attn.forward(h)
        # Post-norm on the attention OUTPUT, then the residual add — not the usual pre-norm order,
        # so the fused rmsnorm+residual-add op does not apply here.
        h = self.post_attention_layernorm.forward(h)
        h = residual + h

        residual = h
        dense = self.mlp.forward(self.pre_feedforward_layernorm.forward(h))
        dense = self.post_feedforward_layernorm_1.forward(dense)

        # The router reads the RAW residual, not the norm-2 output the experts consume.
        topk_weights, topk_ids = self.router.forward(residual)
        moe = self.experts.forward(
            hidden_states=self.pre_feedforward_layernorm_2.forward(residual),
            topk_weights=topk_weights,
            topk_ids=topk_ids,
        )
        moe = self.post_feedforward_layernorm_2.forward(moe)

        h = self.post_feedforward_layernorm.forward(dense + moe)
        h = residual + h
        # Rescales the WHOLE residual stream, not just the FFN branch.
        return h * self.layer_scalar


class Gemma4Model(BaseOP):
    def __init__(self, config: "ModelConfig", expert_quant: "QuantConfig | None"):
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = OPList(
            [Gemma4DecoderLayer(config, i, expert_quant) for i in range(config.num_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self._embed_scale = config.embed_scale
        self._embed_scale_cache: torch.Tensor | None = None

    def _scale_tensor(self, like: torch.Tensor) -> torch.Tensor:
        # The reference casts sqrt(hidden) to the WEIGHT dtype before multiplying, so in fp16 the
        # constant is 53.0625 and not 53.0660. Materializing it as a tensor of `like`'s dtype
        # reproduces that rounding; a bare python float would multiply at fp32 precision.
        cache = self._embed_scale_cache
        if cache is None or cache.dtype != like.dtype or cache.device != like.device:
            cache = torch.tensor(self._embed_scale, dtype=like.dtype, device=like.device)
            self._embed_scale_cache = cache
        return cache

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        h = self.embed_tokens.forward(input_ids)
        if self._embed_scale is not None:
            h = h * self._scale_tensor(h)
        for layer in self.layers.op_list:
            h = layer.forward(h)
        return self.norm.forward(h)


class Gemma4ForConditionalGeneration(BaseLLMModel):
    def __init__(self, config: "ModelConfig"):
        # Only the routed experts are quantized; the attention projections, the dense MLP, the
        # router and lm_head are all in the checkpoint's ignore list. The backbone keeps the real
        # quant config so each linear can gate itself via is_module_quantized (no model-name branch),
        # and the experts get it explicitly.
        self.model = Gemma4Model(config, expert_quant=config.quant)
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )
        self._softcap = config.final_logit_softcapping
        super().__init__()

    def forward(self, return_hidden: bool = False):
        hidden = self.model.forward(get_global_ctx().batch.input_ids)
        logits = self.lm_head.forward(hidden)
        if self._softcap is not None:
            # logits = c * tanh(logits / c). Applied to the EMITTED logits only, after lm_head —
            # the reference caps in Gemma4ForCausalLM.forward, not inside the text model. Note the
            # lm_head consumes the UNSCALED embedding matrix; the sqrt(hidden) factor is applied in
            # the embedding forward only, so a tied head must not re-apply it.
            logits = torch.tanh(logits / self._softcap) * self._softcap
        if return_hidden:
            # `hidden` is post-final-norm / pre-lm_head, which is what the draft heads seed from.
            # No aux capture: set_capture_layers is unimplemented for Gemma4, so the base class
            # raises if a proposer ever asks for capture layers rather than silently returning None.
            return logits, hidden, None
        return logits


__all__ = ["Gemma4ForConditionalGeneration", "gemma4_layer_plan", "Gemma4LayerPlan"]
