"""Block-diffusion decoding: the sampler and per-request denoising state.

Deliberately independent of the engine. A denoising step is `logits -> (next canvas, emit
candidate, done?)` and nothing else; the scheduler decides when to run one and the attention
backend decides how the forward is shaped. Keeping the arithmetic here means it can be replayed
against the HuggingFace reference on CPU, bit for bit, with no GPU and no scheduler.
"""

from .sampler import (
    CanvasState,
    DiffusionSamplerConfig,
    DiffusionStep,
    categorical_entropy,
    normalized_probs,
)

__all__ = [
    "CanvasState",
    "DiffusionSamplerConfig",
    "DiffusionStep",
    "categorical_entropy",
    "normalized_probs",
]
