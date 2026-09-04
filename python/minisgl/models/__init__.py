from .base import BaseLLMModel
from .config import ModelConfig, RotaryConfig
from .register import get_model_class
from .weight import (
    cast_checkpoint_tensor,
    chunked_weight_source,
    expert_row_source,
    load_weight,
)


def create_model(model_config: ModelConfig) -> BaseLLMModel:
    return get_model_class(model_config.architectures[0], model_config)


__all__ = [
    "cast_checkpoint_tensor",
    "chunked_weight_source",
    "create_model",
    "expert_row_source",
    "load_weight",
    "RotaryConfig",
]
