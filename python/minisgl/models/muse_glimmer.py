"""Muse-Glimmer (meta-models/Muse-Glimmer-30B) — dense SWA-hybrid decoder with gated attention,
NoPE global layers and Gemma-style sandwich norms. Text-only: the vision tower is skipped by the
loader, exactly as it is for Gemma4, Mistral3 and Qwen3.5.

Full spec, and the provenance of every constant here, is docs/MUSE_GLIMMER_PORT.md. Five details in
this architecture produce plausible-but-wrong output rather than a crash, so each is called out at
its call site below:

  1. TWO RMSNorm conventions. The four per-layer norms are `(1 + weight)`; the FINAL norm is a plain
     `* weight`. Same class, different flag.
  2. TWO epsilons per layer. The sandwich POST-norms use `post_norm_eps` (1e-8); the input-side
     norms use `rms_norm_eps` (1e-5).
  3. QK-norm is WEIGHTLESS — it ships no tensor, so nothing in the state dict hints it exists — and
     runs BEFORE RoPE. Its Q-side `qk_scale_factor` (3.87) is folded into `attn_softmax_scale`.
  4. `self_attn.gate_proj` is a SIGMOID gate on the attention output, driven by the layer input, and
     shares a leaf name with the (completely unrelated) `mlp.gate_proj`.
  5. The 13 full-attention layers are NoPE — no positional encoding at all.
"""
from __future__ import annotations

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
    OPList,
    ParallelLMHead,
    RMSNorm,
    VocabParallelEmbedding,
)
from minisgl.layers.norm import RMSNormNoScale
from minisgl.quant import create_linear_method
from minisgl.utils import div_even, nvtx_annotate

from .base import BaseLLMModel
from .utils import GatedMLP

if TYPE_CHECKING:
    from minisgl.models import ModelConfig, RotaryConfig


@dataclass(frozen=True)
class LayerPlan:
    """Per-layer attention descriptor, derived purely from ModelConfig."""

    is_sliding: bool
    sliding_window: int  # 0 for a full layer
    kv_id: int  # compact index: full -> main paged pool; sliding -> SWA ring pool
    rotary_config: "RotaryConfig | None"  # None == NoPE


def muse_glimmer_layer_plan(config: "ModelConfig", layer_id: int) -> LayerPlan:
    """Compute layer `layer_id`'s attention plan.

    Sliding layers (39 of 52) get a 2048-token window, a compact id into the SWA ring pool, and
    RoPE; full layers (13) get the main paged pool and NO positional encoding.

    NoPE is read from `layer_rope_theta[i] == 0` rather than from `is_sliding`. In this checkpoint
    the two coincide exactly, but that is a property of its config — both fall out of the same
    `(num_layers - 1 - i) % 4 == 0` rule — not a structural invariant, and aliasing them would hide
    a future checkpoint that separates them.
    """
    assert config.layer_types is not None, "Muse-Glimmer requires a per-layer attention schedule"
    is_sliding = config.layer_types[layer_id] == "sliding_attention"
    nope = layer_id in config.nope_layer_ids
    return LayerPlan(
        is_sliding=is_sliding,
        sliding_window=(config.sliding_window or 0) if is_sliding else 0,
        kv_id=(
            config.swa_layer_ids.index(layer_id)
            if is_sliding
            else config.full_attn_layer_ids.index(layer_id)
        ),
        rotary_config=None if nope else config.rotary_config,
    )


class MuseGlimmerAttention(BaseOP):
    """GQA with a weightless QK-norm, an optional sliding window, optional NoPE, and a sigmoid
    output gate.

    `hidden_size (6656) != num_qo_heads * head_dim (4096)`, so q/gate/o are all non-square — none of
    the usual "attention dim == hidden dim" shortcuts apply.
    """

    def __init__(self, config: "ModelConfig", layer_id: int):
        plan = muse_glimmer_layer_plan(config, layer_id)
        head_dim = config.head_dim
        nqo, nkv = config.num_qo_heads, config.num_kv_heads
        q = config.quant
        # Checkpoint-space name, so the `ignore` list is queried in the space it was written in.
        prefix = f"model.layers.{layer_id}.self_attn"

        def _method(module: str):
            name = f"{prefix}.{module}"
            return create_linear_method(q, quantized=q is not None and q.is_module_quantized(name))

        # q/k/v stay SEPARATE (the checkpoint ships them apart, and merging them would have to
        # interleave the NVFP4 group scales too).
        self.q_proj = LinearColParallelMerged(
            config.hidden_size, [nqo * head_dim], has_bias=False, quant_method=_method("q_proj")
        )
        self.k_proj = LinearColParallelMerged(
            config.hidden_size, [nkv * head_dim], has_bias=False, quant_method=_method("k_proj")
        )
        self.v_proj = LinearColParallelMerged(
            config.hidden_size, [nkv * head_dim], has_bias=False, quant_method=_method("v_proj")
        )
        # [4] The attention output gate. PER-CHANNEL over all nqo*head_dim (NOT per-head, and NOT
        # softplus — that is Laguna's `g_proj`, a different gate with a different shape). Column-
        # parallel exactly like q_proj, so its shard lines up with the attention output's.
        self.gate_proj = LinearColParallelMerged(
            config.hidden_size, [nqo * head_dim], has_bias=False, quant_method=_method("gate_proj")
        )
        # [3] WEIGHTLESS QK-norm (`with_scale=False`), per-head over head_dim, applied BEFORE RoPE.
        # It ships no tensor, so a port that simply never built it would load cleanly and run wrong.
        # One instance serves both q and k: it is stateless, so sharing it is not aliasing state.
        # Its Q-side `qk_scale_factor` (3.87) is NOT applied here — it is folded into
        # `config.attn_softmax_scale` (= 3.87/sqrt(head_dim)), which is exact because scaling Q by c
        # then dotting is identical to scaling the logits by c, and saves a tensor multiply/layer.
        qk_norm = RMSNormNoScale(eps=config.rms_norm_eps)
        self.attn = AttentionLayer(
            layer_id=plan.kv_id,  # compact pool id (full -> main pool, sliding -> SWA ring)
            head_dim=head_dim,
            num_qo_heads=nqo,
            num_kv_heads=nkv,
            rotary_config=plan.rotary_config,  # [5] None on the 13 full-attention layers == NoPE
            q_norm=qk_norm,
            k_norm=qk_norm,
            sliding_window=plan.sliding_window,
        )
        self.o_proj = LinearOProj(
            head_dim * nqo, config.hidden_size, has_bias=False, quant_method=_method("o_proj")
        )
        self._attn_dim_local = div_even(nqo, get_tp_info().size) * head_dim
        self.plan = plan

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q_proj.forward(x)
        k = self.k_proj.forward(x)
        v = self.v_proj.forward(x)
        # [4] The gate reads the LAYER INPUT (the input_layernorm output), not the attention result.
        gate = self.gate_proj.forward(x)
        o = self.attn.forward(torch.cat([q, k, v], dim=-1))
        # sigmoid in fp32 then cast back, matching the reference's `torch.sigmoid` on an fp32-upcast
        # activation; applied elementwise to the concatenated head outputs BEFORE o_proj.
        o = o * torch.sigmoid(gate.float()).to(o.dtype)
        return self.o_proj.forward(o.view(-1, self._attn_dim_local))


class MuseGlimmerDecoderLayer(BaseOP):
    """Gemma-style SANDWICH-norm layer: the post-norms sit on the sublayer OUTPUT, before the
    residual add. That ordering is why this does not use `RMSNormFused` (whose whole point is to
    fuse the norm with a residual add that here happens AFTER it) and why `forward` threads a plain
    `x` rather than the usual `(x, residual)` pair — same shape as Gemma4's layer."""

    def __init__(self, config: "ModelConfig", layer_id: int):
        hidden = config.hidden_size
        # [1] `plus_one=True`: these four are the CENTERED convention, `normed * (1 + weight)`.
        # [2] and the two POST-norms carry their own, much tighter epsilon.
        eps_in = config.rms_norm_eps
        eps_post = config.post_norm_eps if config.post_norm_eps is not None else eps_in
        self.input_layernorm = RMSNorm(hidden, eps=eps_in, plus_one=True)
        self.post_attention_layernorm = RMSNorm(hidden, eps=eps_post, plus_one=True)
        self.pre_feedforward_layernorm = RMSNorm(hidden, eps=eps_in, plus_one=True)
        self.post_feedforward_layernorm = RMSNorm(hidden, eps=eps_post, plus_one=True)
        self.self_attn = MuseGlimmerAttention(config, layer_id)
        self.mlp = GatedMLP(config)
        self._layer_id = layer_id

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.input_layernorm.forward(x)
        h = self.self_attn.forward(h)
        h = self.post_attention_layernorm.forward(h)
        h = residual + h

        residual = h
        h = self.pre_feedforward_layernorm.forward(h)
        h = self.mlp.forward(h)
        h = self.post_feedforward_layernorm.forward(h)
        return residual + h


class MuseGlimmerModel(BaseOP):
    def __init__(self, config: "ModelConfig"):
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = OPList(
            [MuseGlimmerDecoderLayer(config, i) for i in range(config.num_layers)]
        )
        # [1] The FINAL norm is the PLAIN convention (`normed * weight`, plus_one=False) while the
        # 4x52 layer norms above are centered. Same class, opposite flag; getting it backwards is a
        # silent quality bug with no shape error to catch it.
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # A WEIGHTLESS RMSNorm on the embedding output. This is NOT Gemma's sqrt(hidden) multiplier
        # (`embed_scale`) — reaching for that instead would be wrong by a large constant factor. The
        # reference keeps it unfused from the embedding matrix on purpose, so the DFlash drafter can
        # embed without it.
        self._embed_norm = RMSNormNoScale(eps=config.rms_norm_eps)
        # Spec-decode aux capture: decoder-layer ids whose output hidden is stashed (None = off).
        self._capture_layer_ids: "list[int] | None" = None

    def set_capture_layers(self, ids: "list[int] | None") -> None:
        self._capture_layer_ids = list(ids) if ids else None

    def forward(
        self, input_ids: torch.Tensor, return_hidden: bool = False
    ) -> "torch.Tensor | Tuple[torch.Tensor, torch.Tensor | None]":
        h = self._embed_norm.forward(self.embed_tokens.forward(input_ids))
        # Aux capture is OFF unless return_hidden AND layers are programmed: zero cost otherwise.
        cap = self._capture_layer_ids if return_hidden else None
        cap_set = set(cap) if cap else None
        grabbed: "dict[int, torch.Tensor]" = {}
        for lid, layer in enumerate(self.layers.op_list):
            h = layer.forward(h)
            if cap_set is not None and lid in cap_set:
                # The layer's RETURN VALUE is the output hidden, full stop — this decoder threads a
                # plain `x` and does its residual adds internally (sandwich norms forbid the fused
                # rmsnorm+residual-add). So there is no `x` vs `x + residual` ambiguity here, and no
                # equivalent of EAGLE3's MINISGL_EAGLE3_AUX_MODE toggle is needed or meaningful.
                grabbed[lid] = h.clone()
        final = self.norm.forward(h)
        if return_hidden:
            # Stack in the PROGRAMMED id order, not sorted order: the drafter's `fc` concatenates its
            # aux inputs in the order it was trained on, so this ordering is part of the contract.
            aux_stack = torch.stack([grabbed[i] for i in cap], dim=0) if cap else None
            return final, aux_stack
        return final


class MuseGlimmerForConditionalGeneration(BaseLLMModel):
    """Muse-Glimmer 30B. The `ForConditionalGeneration` name is the checkpoint's (it is a vision
    model); this serves the text decoder only."""

    def __init__(self, config: "ModelConfig"):
        self.model = MuseGlimmerModel(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,  # untied
            tied_embedding=None,
        )
        self._softcap = config.final_logit_softcapping
        self._multiplier = config.output_multiplier
        super().__init__()

    def _logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """`T * tanh(lm_head(h) * m / T)`.

        Both constants land on the RETURNED logits, so they change SAMPLING — not just a training
        loss scale. `output_multiplier` (1/sqrt(26)) is applied BEFORE the cap, so it is not
        interchangeable with folding it into the cap constant. Any spec-decode verify path has to
        reproduce this transform or its accept test compares against differently-scaled logits."""
        logits = self.lm_head.forward(hidden)
        if self._multiplier is not None:
            logits = logits * self._multiplier
        if self._softcap is not None:
            logits = torch.tanh(logits / self._softcap) * self._softcap
        return logits

    def forward(self, return_hidden: bool = False):
        input_ids = get_global_ctx().batch.input_ids
        if return_hidden:
            # last_hidden = post-final-norm hidden (pre-lm_head); aux = stacked captured layers.
            hidden, aux_hidden = self.model.forward(input_ids, return_hidden=True)
            return self._logits(hidden), hidden, aux_hidden
        return self._logits(self.model.forward(input_ids))

    def set_capture_layers(self, ids: "list[int] | None") -> None:
        self.model.set_capture_layers(ids)


__all__ = ["MuseGlimmerForConditionalGeneration"]
