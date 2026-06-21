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
    model_type: str
    architectures: list[str]
    quant: QuantConfig | None = None
    # ---- GDN / linear-attention (Qwen3-Next / Qwen3.5 hybrid). None for dense models. ----
    # `layer_types[i]` is "linear_attention" (GDN) or "full_attention". Populated by from_hf
    # ONLY when the config carries linear-attention dims, so the dense path stays untouched.
    linear_num_key_heads: int | None = None
    linear_num_value_heads: int | None = None
    linear_key_head_dim: int | None = None
    linear_value_head_dim: int | None = None
    linear_conv_kernel_dim: int | None = None
    layer_types: tuple[str, ...] | None = None

    @property
    def is_moe(self) -> bool:
        return "moe" in self.model_type

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

    @classmethod
    def from_hf(cls, config: PretrainedConfig) -> ModelConfig:
        quant = QuantConfig.from_hf(config)  # quantization_config is top-level
        if hasattr(config, "text_config") and config.text_config is not None:
            top = config
            config = config.text_config
            for attr in ("architectures", "rope_theta", "rope_scaling"):
                if not getattr(config, attr, None) and getattr(top, attr, None):
                    setattr(config, attr, getattr(top, attr))

        num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        tie_word_embeddings = getattr(config, "tie_word_embeddings", False)
        model_type = getattr(config, "model_type", "llama")
        num_experts = getattr(config, "num_local_experts", getattr(config, "num_experts", 0))
        num_experts_per_tok = getattr(config, "num_experts_per_tok", 0)
        moe_intermediate_size = getattr(config, "moe_intermediate_size", 0)
        norm_topk_prob = getattr(config, "norm_topk_prob", False)
        architectures = getattr(config, "architectures", ["LlamaForCausalLM"])

        # Llama/Qwen: rope_theta is a direct attr; Mistral: it's inside rope_scaling dict;
        # Qwen3.5: a single `rope_parameters` dict (rope_theta + partial_rotary_factor + mrope).
        rope_scaling = getattr(config, "rope_scaling", None)
        rope_params = getattr(config, "rope_parameters", None)
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
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            rms_norm_eps=config.rms_norm_eps,
            tie_word_embeddings=tie_word_embeddings,
            rotary_config=RotaryConfig(
                head_dim=head_dim,
                rotary_dim=rotary_dim,
                max_position=config.max_position_embeddings,
                base=rope_theta,
                scaling=rope_scaling,
            ),
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            moe_intermediate_size=moe_intermediate_size,
            norm_topk_prob=norm_topk_prob,
            model_type=model_type,
            architectures=architectures,
            quant=quant,
            linear_num_key_heads=linear_num_key_heads,
            linear_num_value_heads=getattr(config, "linear_num_value_heads", None),
            linear_key_head_dim=getattr(config, "linear_key_head_dim", None),
            linear_value_head_dim=getattr(config, "linear_value_head_dim", None),
            linear_conv_kernel_dim=getattr(config, "linear_conv_kernel_dim", None),
            layer_types=layer_types,
        )