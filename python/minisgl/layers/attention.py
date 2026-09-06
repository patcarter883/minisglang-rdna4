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
        rotary_config: RotaryConfig | None,
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
        # `rotary_config is None` == NoPE: this layer applies NO positional encoding at all. Not a
        # zero-frequency rope (which is numerically an identity but still costs a kernel launch per
        # layer) — the rope is simply never built and never called. Muse-Glimmer marks its 13 full-
        # attention layers NoPE via `layer_rope_theta[i] == 0`, letting the global layers mix context
        # without a positional prior while the 39 sliding layers carry RoPE.
        self.rotary = (
            None
            if rotary_config is None
            else get_rope(
                head_dim=head_dim,
                rotary_dim=rotary_config.rotary_dim,
                max_position=rotary_config.max_position,
                base=rotary_config.base,
                rope_scaling=(
                    tuple(rotary_config.scaling.items()) if rotary_config.scaling else None
                ),
                interleave=rotary_config.interleave,  # GLM-style interleaved RoPE, else NeoX
            )
        )
        self.q_norm = q_norm
        self.k_norm = k_norm

    def forward(self, qkv: torch.Tensor, selection: object | None = None) -> torch.Tensor:
        """`selection` is a `minisgl.attention.qsa.QSASelection` on a QSA full-attention layer.

        It changes exactly ONE thing: which backend entry point the (already normed, already roped)
        q/k/v go to. Everything above that line — the q/k norms, the partial rotary, the contiguity
        invariant — is byte-identical between the dense and the sparse call, which is what makes
        "sparse == dense when the selection is everything" a statement about the attention kernel
        alone rather than about two independently-assembled forwards."""
        ctx = get_global_ctx()
        q, k, v = qkv.split([self.qo_attn_dim, self.kv_attn_dim, self.kv_attn_dim], dim=-1)
        if self.q_norm is not None:
            self.q_norm.forward_inplace(q.view(-1, self.num_qo_heads, self.head_dim))
        if self.k_norm is not None:
            self.k_norm.forward_inplace(k.view(-1, self.num_kv_heads, self.head_dim))
        if self.rotary is not None:
            q, k = self.rotary.forward(ctx.batch.positions, q, k)
        else:
            # NoPE. `qkv.split` hands back NON-CONTIGUOUS views (stride = the fused row width), and
            # every roped model is handed contiguous q/k only as a SIDE EFFECT of rope, which does
            # `query.contiguous()` internally and returns fresh tensors. With rope skipped that
            # invariant silently lapses, and the HIP decode kernel rejects it — `attn_decode: q must
            # be contiguous`, raised during graph capture, i.e. at boot rather than in a way any
            # numeric test would surface. Restore the invariant explicitly instead of relying on a
            # neighbouring op to launder it.
            q, k = q.contiguous(), k.contiguous()
        q = q.view(-1, self.num_qo_heads, self.head_dim)
        if selection is not None:
            o = ctx.attn_backend.forward_sparse(
                q, k, v, self.layer_id, ctx.batch, selection.slots, selection.lens
            )
        else:
            o = ctx.attn_backend.forward(
                q, k, v, self.layer_id, ctx.batch, sliding_window=self.sliding_window
            )
        return o.view(-1, self.qo_attn_dim)
