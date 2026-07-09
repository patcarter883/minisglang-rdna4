from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from minisgl.core import get_global_ctx
from minisgl.distributed import get_tp_info
from minisgl.utils import div_even

from .base import StateLessOP
from .rotary import get_rope

if TYPE_CHECKING:
    from minisgl.layers import RMSNorm
    from minisgl.models import RotaryConfig


class AttentionLayer(StateLessOP):
    def __init__(
        self,
        layer_id: int,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rotary_config: RotaryConfig,
        q_norm: RMSNorm | None = None,
        k_norm: RMSNorm | None = None,
        sliding_window: int = 0,
    ):
        assert num_qo_heads % num_kv_heads == 0
        self.layer_id = layer_id
        self.head_dim = head_dim
        # Per-layer sliding window (Laguna SWA layers: 512). 0 = global/full attention (dense, MLA,
        # GDN full layers, and every non-SWA model). When >0 the backend (a) masks the attention to
        # the last `sliding_window` keys and (b) routes this layer's paged KV to the window-bounded
        # SWA ring pool instead of the full-context main pool. `layer_id` then indexes the SWA pool.
        self.sliding_window = sliding_window
        tp_size = get_tp_info().size
        self.num_qo_heads = div_even(num_qo_heads, tp_size)
        self.num_kv_heads = div_even(num_kv_heads, tp_size, allow_replicate=True)
        self.qo_attn_dim = self.num_qo_heads * head_dim
        self.kv_attn_dim = self.num_kv_heads * head_dim
        self.rotary = get_rope(
            head_dim=head_dim,
            rotary_dim=rotary_config.rotary_dim,
            max_position=rotary_config.max_position,
            base=rotary_config.base,
            rope_scaling=tuple(rotary_config.scaling.items()) if rotary_config.scaling else None,
            interleave=rotary_config.interleave,  # GLM-style interleaved RoPE, else NeoX
        )
        self.q_norm = q_norm
        self.k_norm = k_norm

    def forward(self, qkv: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        q, k, v = qkv.split([self.qo_attn_dim, self.kv_attn_dim, self.kv_attn_dim], dim=-1)
        if self.q_norm is not None:
            self.q_norm.forward_inplace(q.view(-1, self.num_qo_heads, self.head_dim))
        if self.k_norm is not None:
            self.k_norm.forward_inplace(k.view(-1, self.num_kv_heads, self.head_dim))
        q, k = self.rotary.forward(ctx.batch.positions, q, k)
        q = q.view(-1, self.num_qo_heads, self.head_dim)
        o = ctx.attn_backend.forward(
            q, k, v, self.layer_id, ctx.batch, sliding_window=self.sliding_window
        )
        return o.view(-1, self.qo_attn_dim)
