from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict
from transformers import PretrainedConfig

from minisgl.quant.config import QuantConfig


@dataclass(frozen=True)
class RotaryConfig:
    head_dim: int
    rotary_dim: int
    max_position: int
    base: float
    scaling: Dict[str, Any] | None


@dataclass(frozen=True)
class ModelConfig:
    num_layers: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int
    vocab_size: int
    intermediate_size: int
    rms_norm_eps: float
    rotary_config: RotaryConfig
    hidden_act: str
    tie_word_embeddings: bool
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    norm_topk_prob: bool
    # Qwen2-MoE-style shared expert (always-on, runs alongside the top-k routed experts) +
    # its sigmoid gate. 0 for models without a shared expert (Qwen3-MoE, Mixtral).
    shared_expert_intermediate_size: int
    model_type: str
    architectures: list[str]
    quant: QuantConfig | None = None
    # ---- MLA (multi-head latent attention; DeepSeek / GLM-4.x MoE). None for non-MLA models. ----
    # Set ONLY when the config carries kv_lora_rank, so every other model keeps is_mla=False. The
    # absorbed-decode latent dim is kv_lora_rank + qk_rope_head_dim; the per-head qk dim is
    # qk_nope_head_dim + qk_rope_head_dim (== head_dim, overwritten in from_hf for MLA).
    kv_lora_rank: int | None = None
    q_lora_rank: int | None = None
    qk_nope_head_dim: int | None = None
    qk_rope_head_dim: int | None = None
    v_head_dim: int | None = None
    # ---- fine-grained MoE routing (GLM-4.x / DeepSeek "noaux_tc": sigmoid score + correction
    # bias + group-limited top-k, normalize, scale). Defaults are the no-op / plain-top-k case. ----
    n_group: int = 1
    topk_group: int = 1
    routed_scaling_factor: float = 1.0
    first_k_dense_replace: int = 0  # first K decoder layers use a dense MLP, not MoE
    n_shared_experts: int = 0  # always-on shared experts (added, not gated — GLM/DeepSeek style)
    num_nextn_predict_layers: int = 0  # GLM/DeepSeek MTP heads appended after the decoder
    mtp_num_hidden_layers: int = 0  # Qwen3.5 MTP head (mtp.* namespace); 0 = no MTP head
    # ---- GDN / linear-attention (Qwen3-Next / Qwen3.5 hybrid). None for dense models. ----
    # `layer_types[i]` is "linear_attention" (GDN) or "full_attention". Populated by from_hf
    # ONLY when the config carries linear-attention dims, so the dense path stays untouched.
    linear_num_key_heads: int | None = None
    linear_num_value_heads: int | None = None
    linear_key_head_dim: int | None = None
    linear_value_head_dim: int | None = None
    linear_conv_kernel_dim: int | None = None
    layer_types: tuple[str, ...] | None = None
    # ---- ZAYA CCA hybrid (cross-channel attention conv front-end + EDA/MOD MoE). None for non-Zaya.
    # Populated by from_hf ONLY when model_type == "zaya", so every other model keeps is_cca_hybrid
    # False. The schedule is implicit (even layer -> CCA attention, odd -> MoE), so there is no
    # layer_types list; cca_layer_ids derives it from num_layers. State dtype is fp32.
    is_cca: bool = False
    cca_time0: int | None = None  # conv kernel of conv_qk.0 (depthwise); padding TP0 = cca_time0-1
    cca_time1: int | None = None  # conv kernel of conv_qk.1 (grouped);  padding TP1 = cca_time1-1
    cca_num_k_heads: int | None = None  # num_query_groups (k/v heads) = 2
    cca_num_q_heads: int | None = None  # num_attention_heads (q heads)  = 8
    cca_head_dim: int | None = None  # 128
    cca_clamp_temp: bool = False  # if True key temp is exp(clamp(temp,1e-7,2.0)); else raw temp
    zaya_mlp_expansion: int | None = None  # router down_proj width (e.g. 256)
    zaya_use_eda: bool = False  # expert-decision-aggregation: thread prev router hidden across MoE
    zaya_use_mod: bool = False  # mixture-of-depths: extra "skip" expert at index num_experts
    scale_residual_merge: bool = False  # affine on the fp32 residual stream before each input_norm
    residual_in_fp32: bool = False  # carry the residual stream in fp32 across all layers

    @property
    def is_moe(self) -> bool:
        # model_type=="zaya" carries no "moe" substring, but Zaya IS a (CCA-hybrid) MoE model and the
        # engine must build the moe_backend for its unquantized/precomputed-route experts.
        return "moe" in self.model_type or self.is_cca_hybrid

    @property
    def is_mla(self) -> bool:
        """True for a multi-head latent-attention model (DeepSeek / GLM-4.x MoE)."""
        return self.kv_lora_rank is not None

    @property
    def is_gdn_hybrid(self) -> bool:
        """True for a GDN/linear-attention hybrid (interleaved linear + full layers)."""
        return self.layer_types is not None

    @property
    def gdn_layer_ids(self) -> list[int]:
        """Global indices of the linear-attention (GDN) layers, in order. The GDN state
        cache is indexed by position in THIS list (gdn_layer_id), not the global layer_id."""
        if self.layer_types is None:
            return []
        return [i for i, t in enumerate(self.layer_types) if t == "linear_attention"]

    @property
    def num_gdn_layers(self) -> int:
        return len(self.gdn_layer_ids)

    @property
    def gdn_conv_dim(self) -> int:
        """conv_dim = 2*key_dim + value_dim — the causal-conv channel count."""
        assert self.linear_key_head_dim is not None and self.linear_num_key_heads is not None
        key_dim = self.linear_key_head_dim * self.linear_num_key_heads
        value_dim = self.linear_value_head_dim * self.linear_num_value_heads
        return key_dim * 2 + value_dim

    @property
    def is_cca_hybrid(self) -> bool:
        """True for a ZAYA CCA hybrid (even layers = CCA attention, odd layers = MoE)."""
        return self.is_cca

    @property
    def cca_layer_ids(self) -> list[int]:
        """Global indices of the CCA (attention-bearing) layers, in order. The CCA conv-state
        cache AND the paged KV pool are indexed by position in THIS list (cca_layer_id), which is
        contiguous over the attention-bearing layers."""
        if not self.is_cca:
            return []
        return [i for i in range(self.num_layers) if i % 2 == 0]

    @property
    def num_cca_layers(self) -> int:
        return len(self.cca_layer_ids)

    @property
    def cca_conv_dim(self) -> int:
        """conv channel count C = (num_q_heads + num_k_heads) * head_dim (= 1280 for Zaya)."""
        assert self.cca_num_q_heads is not None and self.cca_num_k_heads is not None
        assert self.cca_head_dim is not None
        return (self.cca_num_q_heads + self.cca_num_k_heads) * self.cca_head_dim

    @property
    def cca_conv_width(self) -> int:
        """conv_states width TP = (cca_time0-1) + (cca_time1-1) (= 2 for Zaya)."""
        assert self.cca_time0 is not None and self.cca_time1 is not None
        return (self.cca_time0 - 1) + (self.cca_time1 - 1)

    @classmethod
    def from_hf(cls, config: PretrainedConfig) -> ModelConfig:
        quant = QuantConfig.from_hf(config)  # quantization_config is top-level
        if hasattr(config, "text_config") and config.text_config is not None:
            top = config
            config = config.text_config
            for attr in ("architectures", "rope_theta", "rope_scaling"):
                if not getattr(config, attr, None) and getattr(top, attr, None):
                    setattr(config, attr, getattr(top, attr))

        model_type = getattr(config, "model_type", "llama")
        # PORT_PLAN §detection: a Zaya CCA hybrid is identified by model_type=="zaya" OR an explicit
        # top-level `cca` flag — NOT the conjunction. AND silently disabled the entire CCA port for a
        # `zaya` checkpoint that omits `cca` (no error, garbage output); OR matches the spec and the
        # shipping checkpoints (which carry both).
        is_cca = (model_type == "zaya") or bool(getattr(config, "cca", False))
        # Zaya names kv heads `num_query_groups` (not num_key_value_heads) and top-k `moe_router_topk`.
        num_kv_heads = (
            getattr(config, "num_query_groups", None)
            if is_cca
            else getattr(config, "num_key_value_heads", None)
        ) or config.num_attention_heads
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        tie_word_embeddings = getattr(config, "tie_word_embeddings", False)
        if is_cca:
            # ZAYA ties the lm_head to embed_tokens (no separate lm_head.weight in the checkpoint),
            # but the HF config omits tie_word_embeddings; force it on.
            tie_word_embeddings = True
        # All-MoE models (e.g. qwen3_5_moe, mlp_only_layers=[]) carry no dense `intermediate_size`.
        # ZAYA stores its (merged gate+up) FFN width as `ffn_hidden_size`; the per-gate width is half.
        intermediate_size = getattr(config, "intermediate_size", 0)
        # Routed-expert count: Mixtral/Qwen use num_local_experts/num_experts; GLM-4.x & DeepSeek
        # MoE use n_routed_experts.
        num_experts = (
            getattr(config, "num_local_experts", None)
            or getattr(config, "num_experts", None)
            or getattr(config, "n_routed_experts", 0)
        )
        num_experts_per_tok = getattr(config, "num_experts_per_tok", 0)
        moe_intermediate_size = getattr(config, "moe_intermediate_size", 0)
        if is_cca:
            # ZAYA: top-k from moe_router_topk; per-expert FFN width = ffn_hidden_size//2 (the
            # checkpoint's linear_fc1 is the MERGED gate+up of width ffn_hidden_size).
            num_experts_per_tok = getattr(config, "moe_router_topk", 1) or 1
            ffn_hidden_size = getattr(config, "ffn_hidden_size", 0) or 0
            moe_intermediate_size = ffn_hidden_size // 2
        norm_topk_prob = getattr(config, "norm_topk_prob", False)
        shared_expert_intermediate_size = getattr(config, "shared_expert_intermediate_size", 0)
        architectures = getattr(config, "architectures", ["LlamaForCausalLM"])

        # MLA (DeepSeek / GLM-4.x MoE): present only when kv_lora_rank is set. When MLA, head_dim
        # has no meaningful config value (hidden/heads is non-integral), so derive it from the
        # per-head qk dim (qk_nope + qk_rope) — the value the absorbed/materialized kernels use.
        kv_lora_rank = getattr(config, "kv_lora_rank", None)
        q_lora_rank = getattr(config, "q_lora_rank", None)
        qk_nope_head_dim = getattr(config, "qk_nope_head_dim", None)
        qk_rope_head_dim = getattr(config, "qk_rope_head_dim", None)
        v_head_dim = getattr(config, "v_head_dim", None)
        if kv_lora_rank is not None:
            head_dim = qk_nope_head_dim + qk_rope_head_dim
        # noaux_tc / fine-grained MoE knobs (defaults = plain top-k, no shared expert).
        n_group = getattr(config, "n_group", 1) or 1
        topk_group = getattr(config, "topk_group", 1) or 1
        routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0) or 1.0
        first_k_dense_replace = getattr(config, "first_k_dense_replace", 0) or 0
        n_shared_experts = getattr(config, "n_shared_experts", 0) or 0
        num_nextn_predict_layers = getattr(config, "num_nextn_predict_layers", 0) or 0
        mtp_num_hidden_layers = getattr(config, "mtp_num_hidden_layers", 0) or 0

        # RMSNorm eps: Llama/Qwen use `rms_norm_eps`; ZAYA names it `norm_epsilon`.
        rms_norm_eps = getattr(config, "rms_norm_eps", None)
        if rms_norm_eps is None:
            rms_norm_eps = getattr(config, "norm_epsilon", 1e-5)

        # Llama/Qwen: rope_theta is a direct attr; Mistral: it's inside rope_scaling dict;
        # Qwen3.5: a single `rope_parameters` dict (rope_theta + partial_rotary_factor + mrope).
        # Normalize falsy values to None: some configs (e.g. ZAYA) ship `rope_scaling: false`,
        # which must be treated as "no scaling" — not a dict to .get() on.
        rope_scaling = getattr(config, "rope_scaling", None) or None
        rope_params = getattr(config, "rope_parameters", None) or None
        rope_theta = getattr(config, "rope_theta", None)
        if rope_theta is None and rope_params is not None:
            rope_theta = rope_params.get("rope_theta")
        if rope_theta is None and rope_scaling is not None:
            rope_theta = rope_scaling["rope_theta"]
        # Partial rotary (Qwen3.5: rotary_dim = head_dim * partial_rotary_factor, e.g. 0.25 -> 64).
        # Plain models leave it None -> full rotary (rotary_dim == head_dim).
        partial = getattr(config, "partial_rotary_factor", None)
        if partial is None and rope_params is not None:
            partial = rope_params.get("partial_rotary_factor")
        rotary_dim = int(head_dim * partial) if partial is not None else head_dim

        # Rope scaling: only a real scheme (llama3/yarn/...) becomes RotaryConfig.scaling.
        # A "default" rope_type carries no scaling -> None (so _get_rope takes the plain-rotary
        # path, identical to the default branch). Qwen3.5's rope_parameters is rope_type
        # "default" + an mrope_section LIST; mrope is multimodal-only (text serving uses plain
        # partial rotary, already set via rotary_dim above) and the list is unhashable in the
        # cached rope builder, so it must NOT be folded into scaling.
        rope_dict = rope_scaling if rope_scaling is not None else rope_params
        scaling = (
            rope_dict
            if rope_dict is not None and rope_dict.get("rope_type") not in (None, "default")
            else None
        )

        # GDN / linear-attention hybrid (Qwen3-Next / Qwen3.5). Only populated when the config
        # actually carries linear-attention dims, so dense models keep layer_types=None even if
        # they define a (sliding/full) layer_types list of their own.
        linear_num_key_heads = getattr(config, "linear_num_key_heads", None)
        layer_types = None
        if linear_num_key_heads is not None:
            layer_types = getattr(config, "layer_types", None)
            if layer_types is None:
                # Implicit pattern: a full-attention layer every `full_attention_interval`.
                interval = getattr(config, "full_attention_interval", 4)
                layer_types = [
                    "full_attention" if (i + 1) % interval == 0 else "linear_attention"
                    for i in range(config.num_hidden_layers)
                ]
            layer_types = tuple(layer_types)

        return cls(
            num_layers=config.num_hidden_layers,
            num_qo_heads=config.num_attention_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            intermediate_size=intermediate_size,
            hidden_act=getattr(config, "hidden_act", "silu"),
            rms_norm_eps=rms_norm_eps,
            tie_word_embeddings=tie_word_embeddings,
            rotary_config=RotaryConfig(
                head_dim=head_dim,
                rotary_dim=rotary_dim,
                max_position=config.max_position_embeddings,
                base=rope_theta,
                scaling=scaling,
            ),
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            moe_intermediate_size=moe_intermediate_size,
            norm_topk_prob=norm_topk_prob,
            shared_expert_intermediate_size=shared_expert_intermediate_size,
            model_type=model_type,
            architectures=architectures,
            quant=quant,
            linear_num_key_heads=linear_num_key_heads,
            linear_num_value_heads=getattr(config, "linear_num_value_heads", None),
            linear_key_head_dim=getattr(config, "linear_key_head_dim", None),
            linear_value_head_dim=getattr(config, "linear_value_head_dim", None),
            linear_conv_kernel_dim=getattr(config, "linear_conv_kernel_dim", None),
            layer_types=layer_types,
            kv_lora_rank=kv_lora_rank,
            q_lora_rank=q_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            n_group=n_group,
            topk_group=topk_group,
            routed_scaling_factor=routed_scaling_factor,
            first_k_dense_replace=first_k_dense_replace,
            n_shared_experts=n_shared_experts,
            num_nextn_predict_layers=num_nextn_predict_layers,
            mtp_num_hidden_layers=mtp_num_hidden_layers,
            is_cca=is_cca,
            cca_time0=getattr(config, "cca_time0", 2) if is_cca else None,
            cca_time1=getattr(config, "cca_time1", 2) if is_cca else None,
            cca_num_k_heads=(getattr(config, "num_query_groups", None) if is_cca else None),
            cca_num_q_heads=(config.num_attention_heads if is_cca else None),
            cca_head_dim=(head_dim if is_cca else None),
            cca_clamp_temp=bool(getattr(config, "clamp_temp", False)) if is_cca else False,
            zaya_mlp_expansion=(getattr(config, "zaya_mlp_expansion", 256) if is_cca else None),
            zaya_use_eda=bool(getattr(config, "zaya_use_eda", False)) if is_cca else False,
            zaya_use_mod=bool(getattr(config, "zaya_use_mod", False)) if is_cca else False,
            scale_residual_merge=bool(getattr(config, "scale_residual_merge", False)),
            residual_in_fp32=bool(getattr(config, "residual_in_fp32", False)),
        )