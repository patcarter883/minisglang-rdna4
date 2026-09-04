import importlib

from .config import ModelConfig

_MODEL_REGISTRY = {
    "LlamaForCausalLM": (".llama", "LlamaForCausalLM"),
    "Qwen2ForCausalLM": (".qwen2", "Qwen2ForCausalLM"),
    "Qwen2MoeForCausalLM": (".qwen2_moe", "Qwen2MoeForCausalLM"),
    "Qwen3ForCausalLM": (".qwen3", "Qwen3ForCausalLM"),
    "Qwen3MoeForCausalLM": (".qwen3_moe", "Qwen3MoeForCausalLM"),
    "Qwen3_5ForConditionalGeneration": (".qwen3_5", "Qwen3_5ForConditionalGeneration"),
    "Qwen3_5MoeForConditionalGeneration": (".qwen3_5_moe", "Qwen3_5MoeForConditionalGeneration"),
    # Qwen3.8-Flash-Next (`qwen4_exp`). Text decoder only, like every other multimodal checkpoint
    # here. It is a GDN hybrid *and* a hyper-connection model, so it gets its own entry rather than
    # reusing the Qwen3.5 class — that class has no hyper-connections and would build a final `norm`
    # this checkpoint does not ship.
    "Qwen4ExpForConditionalGeneration": (".qwen4exp", "Qwen4ExpForConditionalGeneration"),
    "Glm4MoeLiteForCausalLM": (".glm4_moe_lite", "Glm4MoeLiteForCausalLM"),
    "ZayaForCausalLM": (".zaya", "ZayaForCausalLM"),
    "MistralForCausalLM": (".mistral", "MistralForCausalLM"),
    "Mistral3ForConditionalGeneration": (".mistral", "MistralForCausalLM"),
    "LagunaForCausalLM": (".laguna", "LagunaForCausalLM"),
    # Gemma4 and DiffusionGemma share ONE backbone; the block-diffusion head is a separate
    # execution mode, not a separate decoder stack.
    "Gemma4ForConditionalGeneration": (".gemma4", "Gemma4ForConditionalGeneration"),
    "Gemma4ForCausalLM": (".gemma4", "Gemma4ForConditionalGeneration"),
    "DiffusionGemmaForBlockDiffusion": (".diffusion_gemma", "DiffusionGemmaForBlockDiffusion"),
    # Muse-Glimmer is a vision model; like every other multimodal checkpoint served here, only its
    # text decoder is built (the loader skips the vision tower).
    "MuseGlimmerForConditionalGeneration": (".muse_glimmer", "MuseGlimmerForConditionalGeneration"),
}


def get_model_class(model_architecture: str, model_config: ModelConfig):
    if model_architecture not in _MODEL_REGISTRY:
        raise ValueError(f"Model architecture {model_architecture} not supported")
    module_path, class_name = _MODEL_REGISTRY[model_architecture]
    module = importlib.import_module(module_path, package=__package__)
    model_cls = getattr(module, class_name)
    return model_cls(model_config)


__all__ = ["get_model_class"]