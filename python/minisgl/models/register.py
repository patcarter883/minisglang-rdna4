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
    "Glm4MoeLiteForCausalLM": (".glm4_moe_lite", "Glm4MoeLiteForCausalLM"),
    "ZayaForCausalLM": (".zaya", "ZayaForCausalLM"),
    "MistralForCausalLM": (".mistral", "MistralForCausalLM"),
    "Mistral3ForConditionalGeneration": (".mistral", "MistralForCausalLM"),
    "LagunaForCausalLM": (".laguna", "LagunaForCausalLM"),
}


def get_model_class(model_architecture: str, model_config: ModelConfig):
    if model_architecture not in _MODEL_REGISTRY:
        raise ValueError(f"Model architecture {model_architecture} not supported")
    module_path, class_name = _MODEL_REGISTRY[model_architecture]
    module = importlib.import_module(module_path, package=__package__)
    model_cls = getattr(module, class_name)
    return model_cls(model_config)


__all__ = ["get_model_class"]