from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from minisgl.core import get_global_ctx
from minisgl.distributed import get_tp_info
from minisgl.utils import div_even

from . import _tail_hip
from .base import StateLessOP
from .norm import RMSNorm, RMSNormNoScale, _rms_norm
from .rotary import get_rope

if TYPE_CHECKING:
    from minisgl.models import RotaryConfig


class AttentionLayer(StateLessOP):
    def __init__(
        self,
        layer_id: int,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rotary_config: RotaryConfig | None,
        q_norm: RMSNorm | RMSNormNoScale | None = None,
        k_norm: RMSNorm | RMSNormNoScale | None = None,
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
        # The q/k norm + RoPE front end, shared with the drafters that attend outside this class.
        self.qk_prep = QKNormRope(
            self.num_qo_heads, self.num_kv_heads, head_dim, q_norm, k_norm, self.rotary
        )

    def forward(self, qkv: torch.Tensor, selection: object | None = None) -> torch.Tensor:
        q, k, v = qkv.split([self.qo_attn_dim, self.kv_attn_dim, self.kv_attn_dim], dim=-1)
        return self.forward_qkv(q, k, v, selection=selection)

    def forward_qkv(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        v_norm_eps: float | None = None,
        selection: object | None = None,
    ) -> torch.Tensor:
        """q/k norm -> RoPE -> attention from separate q/k/v (views are fine: [n, heads*hd] with any
        row stride, or q as [n, heads, hd] with any head stride). `v_norm_eps` also applies the
        scale-less norm to v; v may be k. `selection` (QSA) only changes the backend entry point."""
        ctx = get_global_ctx()
        q, k, v = self.qk_prep.forward(q, k, v, ctx.batch.positions, v_norm_eps=v_norm_eps)
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


class QKNormRope(StateLessOP):
    """Per-head q/k norm, then RoPE (and optionally v's scale-less norm) — the attention front end
    shared by AttentionLayer and the drafters. One tail_hip.qk_norm_rope launch where the kernel
    covers the layer, else the op chain; bit-identical either way. Holds references to the norms."""

    def __init__(
        self,
        num_qo_heads: int,   # LOCAL (post-TP) head counts
        num_kv_heads: int,
        head_dim: int,
        q_norm: RMSNorm | RMSNormNoScale | None,
        k_norm: RMSNorm | RMSNormNoScale | None,
        rotary: object | None,  # layers.rotary RoPE, or None for NoPE
    ):
        self.num_qo_heads, self.num_kv_heads, self.head_dim = num_qo_heads, num_kv_heads, head_dim
        self.qo_attn_dim, self.kv_attn_dim = num_qo_heads * head_dim, num_kv_heads * head_dim
        self.q_norm, self.k_norm, self.rotary = q_norm, k_norm, rotary
        self._prep: tuple | None = None  # the kernel's per-layer constants, resolved on first use

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor | None,
        positions: torch.Tensor,
        *,
        v_norm_eps: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """q: [n, hq*hd] (any row stride) or [n, hq, hd] (any head stride); k/v: [n, hk*hd] or
        [n, hk, hd]. Returns contiguous [n, hq*hd] q and [n, hk*hd] k, and v — normed and contiguous
        when `v_norm_eps` is given, else passed through untouched. v may BE k (a Gemma-4 full layer's V
        is k_proj's raw output): nothing is written into the inputs before v has been read. The
        fallback chain norms q/k IN PLACE, so callers must not rely on their q/k inputs afterwards."""
        vs = () if v is None else (v,)
        prep = self._fused_prep(q.dtype, v_norm_eps) if _tail_hip.active(q, k, *vs) else False
        if prep and _rows_aligned(q, k, *vs):
            q_w, k_w, cache, rd, do_norm, plus_one, eps = prep
            q, k, vn = _tail_hip.qk_norm_rope(
                q, k, v if v_norm_eps is not None else None, q_w, k_w,
                positions.to(torch.int32), cache, self.num_qo_heads, self.num_kv_heads,
                self.head_dim, rd, do_norm, plus_one, eps,
            )
            return q, k, (vn if v_norm_eps is not None else v)
        n = q.shape[0]
        q = q.reshape(n, self.qo_attn_dim)
        if v_norm_eps is not None:
            v = _rms_norm(v.reshape(-1, self.head_dim), None, v_norm_eps).view(n, -1)
        if self.q_norm is not None:
            self.q_norm.forward_inplace(q.view(-1, self.num_qo_heads, self.head_dim))
        if self.k_norm is not None:
            self.k_norm.forward_inplace(k.view(-1, self.num_kv_heads, self.head_dim))
        if self.rotary is not None:
            q, k = self.rotary.forward(positions, q, k.reshape(n, self.kv_attn_dim))
        else:
            # NoPE. `qkv.split` hands back NON-CONTIGUOUS views (stride = the fused row width), and
            # every roped model is handed contiguous q/k only as a SIDE EFFECT of rope, which does
            # `query.contiguous()` internally and returns fresh tensors. With rope skipped that
            # invariant silently lapses, and the HIP decode kernel rejects it — `attn_decode: q must
            # be contiguous`, raised during graph capture, i.e. at boot rather than in a way any
            # numeric test would surface. Restore the invariant explicitly instead of relying on a
            # neighbouring op to launder it.
            q, k = q.contiguous(), k.reshape(n, self.kv_attn_dim).contiguous()
        return q, k, v

    def _fused_prep(self, dtype: torch.dtype, v_norm_eps: float | None) -> tuple | bool:
        """The layer's arguments to `tail_hip.qk_norm_rope`, or False when the fused front end
        cannot express this layer — which then runs the op chain in `forward`, unchanged. Resolved once,
        after load (the norm gains' dtype is only final then)."""
        if self._prep is not None and self._prep[0] == (dtype, v_norm_eps):
            return self._prep[1]
        args = self._resolve_prep(dtype, v_norm_eps)
        self._prep = ((dtype, v_norm_eps), args)
        return args

    def _resolve_prep(self, dtype: torch.dtype, v_norm_eps: float | None) -> tuple | bool:
        if not hasattr(_tail_hip, "qk_norm_rope") or dtype not in (torch.float16, torch.bfloat16):
            return False
        rot = self.rotary
        # GLM's interleaved pairing is not NeoX rotate-half; the kernel's 8-wide chunks need rd % 16.
        if self.head_dim % 8 or (rot is not None and (rot.interleave or rot.rotary_dim % 16)):
            return False
        qn, kn = self.q_norm, self.k_norm
        if qn is None and kn is None:
            if v_norm_eps is not None:
                return False
            do_norm, q_w, k_w, plus_one, eps = 0, None, None, False, 0.0
        elif type(qn) is not type(kn) or qn.eps != kn.eps:
            return False
        elif isinstance(qn, RMSNormNoScale):
            do_norm, q_w, k_w, plus_one, eps = 1, None, None, False, qn.eps
        elif isinstance(qn, RMSNorm):
            if (qn.plus_one != kn.plus_one or qn.weight.dtype != dtype or kn.weight.dtype != dtype
                    or not qn.weight.is_cuda):
                return False
            do_norm, q_w, k_w, plus_one, eps = 1, qn.weight, kn.weight, qn.plus_one, qn.eps
        else:
            return False
        if v_norm_eps is not None and do_norm and v_norm_eps != eps:
            return False
        rd, cache = (0, None) if rot is None else (rot.rotary_dim, rot._cos_sin_cache)
        return q_w, k_w, cache, rd, do_norm, int(plus_one), (v_norm_eps if v_norm_eps is not None else eps)


def _rows_aligned(*ts: torch.Tensor) -> bool:
    """qk_norm_rope's b128 row reads need 16-byte-aligned row/head starts (it refuses otherwise)."""
    for t in ts:
        es = t.element_size()
        if t.stride(-1) != 1 or t.data_ptr() % 16 or any((st * es) % 16 for st in t.stride()[:-1]):
            return False
    return True
