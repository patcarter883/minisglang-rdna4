from __future__ import annotations

import functools
import math
from typing import Any, Callable, Dict, Tuple

import torch

from .base import StateLessOP


class RotaryEmbedding(StateLessOP):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        post_process: None | Callable[[torch.Tensor], torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        # Partial rotary (Qwen3.5: rotary_dim < head_size): rotate the first rotary_dim dims of
        # each head, pass the rest through. rotary_dim == head_size is the full-rotary default.
        assert rotary_dim <= head_size and rotary_dim % 2 == 0
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        if post_process is not None:
            inv_freq = post_process(inv_freq)
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        # buffer, so don't load/save
        self._cos_sin_cache = torch.cat((cos, sin), dim=-1)
        assert self.head_size in [64, 128, 256, 512]

    def _apply(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # x: [n, num_heads * head_size] -> NeoX rotate-half RoPE, fp32 internal, in-dtype out.
        # cos/sin are rotary_dim wide; only the first rotary_dim dims of each head are rotated
        # (partial rotary), the remaining [rotary_dim:head_size] pass through unchanged.
        n = x.shape[0]
        orig_dtype = x.dtype
        x = x.view(n, -1, self.head_size).float()
        rd = self.rotary_dim
        d = rd // 2
        x_rot = x[..., :rd]
        x1, x2 = x_rot[..., :d], x_rot[..., d:]
        rot = torch.cat((-x2, x1), dim=-1)
        out_rot = x_rot * cos.view(n, 1, rd) + rot * sin.view(n, 1, rd)
        out = torch.cat((out_rot, x[..., rd:]), dim=-1) if rd < self.head_size else out_rot
        return out.view(n, -1).to(orig_dtype)

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # cos_sin_cache row = cat(cos[d/2], sin[d/2]); NeoX needs each repeated to head_size.
        cos_half, sin_half = self._cos_sin_cache[positions].chunk(2, dim=-1)
        cos = torch.cat((cos_half, cos_half), dim=-1)
        sin = torch.cat((sin_half, sin_half), dim=-1)
        return self._apply(query, cos, sin), self._apply(key, cos, sin)


def _get_rope(
    head_dim: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: Dict[str, Any] | None = None,
) -> RotaryEmbedding:
    if rope_scaling is None:
        return RotaryEmbedding(head_dim, rotary_dim, max_position, base)
    # need to test some cases:
    match rope_scaling["rope_type"]:
        case "default":
            return RotaryEmbedding(head_dim, rotary_dim, max_position, base)

        case "llama3":
            scaling_factor: float = rope_scaling["factor"]
            low_freq_factor: float = rope_scaling["low_freq_factor"]
            high_freq_factor: float = rope_scaling["high_freq_factor"]
            original_max_position: int = rope_scaling["original_max_position_embeddings"]

            def post_process(inv_freq: torch.Tensor) -> torch.Tensor:
                # no smooth if low_freq_factor == high_freq_factor
                wave_len = 2 * math.pi / inv_freq
                if low_freq_factor == high_freq_factor:
                    return torch.where(
                        wave_len < original_max_position / high_freq_factor,
                        inv_freq,
                        inv_freq / scaling_factor,
                    )

                delta = high_freq_factor - low_freq_factor
                smooth = (original_max_position / wave_len - low_freq_factor) / delta
                smooth = torch.clamp(smooth, 0, 1)
                factor = (1 - smooth) / scaling_factor + smooth
                return factor * inv_freq

            return RotaryEmbedding(head_dim, rotary_dim, max_position, base, post_process)

        case "yarn":
            factor: float = rope_scaling["factor"]
            beta_fast: float = rope_scaling.get("beta_fast", 32.0)
            beta_slow: float = rope_scaling.get("beta_slow", 1.0)
            orig_max_pos: int = rope_scaling["original_max_position_embeddings"]

            def _find_correction_dim(num_rotations: float) -> float:
                return rotary_dim * math.log(orig_max_pos / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

            low = max(math.floor(_find_correction_dim(beta_fast)), 0)
            high = min(math.ceil(_find_correction_dim(beta_slow)), rotary_dim // 2 - 1)

            def post_process(inv_freq: torch.Tensor) -> torch.Tensor:
                ramp = torch.clamp(
                    (torch.arange(rotary_dim // 2, dtype=torch.float32) - low) / max(high - low, 1),
                    0, 1,
                )
                return (inv_freq / factor) * ramp + inv_freq * (1 - ramp)

            return RotaryEmbedding(head_dim, rotary_dim, max_position, base, post_process)

    raise ValueError(f"Unsupported {rope_scaling = }")


_ROPE_DEVICE: torch.device | None = None


def set_rope_device(device: torch.device):
    global _ROPE_DEVICE
    _ROPE_DEVICE = device


@functools.cache
def get_rope(
    head_dim: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: Tuple[Tuple[str, Any], ...] | None = None,
) -> RotaryEmbedding:
    rope_map = dict(rope_scaling) if rope_scaling is not None else None
    t = torch.tensor([])
    if t.device == torch.device("meta"):
        # we cannot use meta device for rope
        if _ROPE_DEVICE is None:
            raise RuntimeError(
                "We cannot use meta device for rope. Please call set_rope_device() first."
            )
        with torch.device(_ROPE_DEVICE):
            return _get_rope(head_dim, rotary_dim, max_position, base, rope_map)
    return _get_rope(head_dim, rotary_dim, max_position, base, rope_map)


__all__ = ["get_rope", "RotaryEmbedding", "set_rope_device"]