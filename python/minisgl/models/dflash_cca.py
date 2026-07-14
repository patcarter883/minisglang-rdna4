"""DFlash CCA-aware drafter (architecture ``DFlashCCADraftModel``) — the trained ZAYA drafter.

A block-diffusion drafter whose layers are CCA-recurrent (Zyphra Compressed-Convolutional Attention),
NOT Qwen3 GQA (that is models/dflash.py, for the z-lab attention drafters). Ported from
vllm-gfx1201-zaya-dflash/zaya/dflash/cca_drafter_model.py to the minisgl BaseOP world.

Per layer, over the block ``x: [S, T, H]`` (T = 1 + num_spec) with a single per-sequence seed
``seed: [S, H]`` = fc(target aux at the committed position):

    h = input_layernorm(x); if seed: hin = cat([input_layernorm(seed)[:,None], h]) else hin = h
    q = linear_q(hin); k = linear_k(hin); v = val_proj(hin)          # CCA latent projections
    qk = conv_qk(cat([q,k]))                                          # bidirectional depthwise conv (kernel 3, 'same')
    q,k = per-head-RMSNorm(q,k) * sqrt(head_dim); k *= temp           # CCA norm + per-k-head key temperature
    out = softmax(q·kᵀ/√d) · v   (block_mixer='attn')   OR   v        (block_mixer='conv')
    out = o_proj(out)[:, drop seed col]; h = x + out                  # residual
    h = h + down_proj(silu(gate_proj(h2)) * up_proj(h2))             # SwiGLU

The seed is the CCA drafter's ENTIRE cross-block context (a single committed-position aux, folded via
fc) — it was TRAINED on one seed, so the Phase-1 full-context prefix does NOT apply here. Tied vocab:
borrows the target embed_tokens + lm_head. Verification stays the linear verify_greedy (or DDTree).
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F
from minisgl.layers import RMSNorm
from minisgl.layers.base import BaseOP

from .dflash import _PlainLinear


def _rmsnorm_heads(x: torch.Tensor, num_heads: int, head_dim: int, sqrt_head_dim: float,
                   eps: float = 1e-12) -> torch.Tensor:
    """CCA per-head RMS normalize: normalize each head vector in fp32, then scale by sqrt(head_dim).
    Mirrors cca.py::_rms_normalize_qk (unit weight). x: [..., num_heads*head_dim]."""
    shape = x.shape
    xh = x.float().view(*shape[:-1], num_heads, head_dim)
    norm = torch.linalg.vector_norm(xh, ord=2, dim=-1, keepdim=True)
    xh = xh * torch.rsqrt(norm * norm + eps) * sqrt_head_dim
    return xh.view(shape).to(x.dtype)


class _CCADrafterLayer(BaseOP):
    """One CCA-aware drafter layer over the (1+num_spec) block with a prepended seed column."""

    def __init__(self, hidden_size: int, intermediate_size: int, num_q_heads: int,
                 num_k_heads: int, head_dim: int, conv_kernel: int, rms_norm_eps: float,
                 clamp_temp: bool, block_mixer: str) -> None:
        self.num_q_heads = num_q_heads
        self.num_k_heads = num_k_heads
        self.head_dim = head_dim
        self.gqa_groups = num_q_heads // num_k_heads
        self.latent_q = num_q_heads * head_dim
        self.latent_k = num_k_heads * head_dim
        self.sqrt_head_dim = float(head_dim) ** 0.5
        self.clamp_temp = clamp_temp
        self.block_mixer = block_mixer
        self._conv_kernel = conv_kernel
        self._conv_pad = conv_kernel // 2

        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.linear_q = _PlainLinear(hidden_size, self.latent_q)
        self.linear_k = _PlainLinear(hidden_size, self.latent_k)
        self.val_proj = _PlainLinear(hidden_size, self.latent_k)
        self.o_proj = _PlainLinear(self.latent_q, hidden_size)
        # depthwise conv over [q|k] (in_out_ch = latent_q + latent_k), kernel `conv_kernel`, 'same' pad.
        self.conv_qk_weight = torch.empty(self.latent_q + self.latent_k, 1, conv_kernel)
        self.conv_qk_bias = torch.empty(self.latent_q + self.latent_k)
        self.temp = torch.empty(num_k_heads)  # per-k-head key temperature (raw)
        self.gate_proj = _PlainLinear(hidden_size, intermediate_size)
        self.up_proj = _PlainLinear(hidden_size, intermediate_size)
        self.down_proj = _PlainLinear(intermediate_size, hidden_size)

    def _temp_eff(self) -> torch.Tensor:
        t = self.temp.float()
        if self.clamp_temp:
            t = torch.exp(torch.clamp(t, 1e-7, 2.0))
        return t

    def forward(self, x: torch.Tensor, seed: Optional[torch.Tensor]) -> torch.Tensor:
        # x: [S, T, H]; seed: [S, H] or None
        S, T, H = x.shape
        residual = x
        h = self.input_layernorm.forward(x)
        if seed is not None:
            s = self.input_layernorm.forward(seed).unsqueeze(1)  # [S,1,H]
            hin = torch.cat([s, h], dim=1)                       # [S, 1+T, H]
            off = 1
        else:
            hin = h
            off = 0
        L = hin.shape[1]

        q = self.linear_q.forward(hin)   # [S, L, latent_q]
        k = self.linear_k.forward(hin)   # [S, L, latent_k]
        v = self.val_proj.forward(hin)   # [S, L, latent_k]

        # Bidirectional depthwise conv over time (local CCA mixing), trimmed to L.
        qk = torch.cat([q, k], dim=-1).transpose(1, 2)  # [S, C, L]
        qk = F.conv1d(qk, self.conv_qk_weight, self.conv_qk_bias, padding=self._conv_pad,
                      groups=self.latent_q + self.latent_k)[..., :L].transpose(1, 2)  # [S, L, C]
        q = qk[..., :self.latent_q]
        k = qk[..., self.latent_q:]

        q = _rmsnorm_heads(q, self.num_q_heads, self.head_dim, self.sqrt_head_dim)
        k = _rmsnorm_heads(k, self.num_k_heads, self.head_dim, self.sqrt_head_dim)
        temp = self._temp_eff().view(1, 1, self.num_k_heads, 1)

        qh = q.view(S, L, self.num_q_heads, self.head_dim).float()
        kh = k.view(S, L, self.num_k_heads, self.head_dim).float() * temp
        vh = v.view(S, L, self.num_k_heads, self.head_dim).float()
        kh = kh.repeat_interleave(self.gqa_groups, dim=2)  # GQA expand to q heads
        vh = vh.repeat_interleave(self.gqa_groups, dim=2)

        if self.block_mixer == "conv":
            out = vh.reshape(S, L, self.latent_q).to(x.dtype)
        else:
            attn = torch.einsum("slhd,smhd->shlm", qh, kh) / self.sqrt_head_dim
            attn = torch.softmax(attn, dim=-1)
            out = torch.einsum("shlm,smhd->slhd", attn, vh).reshape(S, L, self.latent_q).to(x.dtype)

        out = self.o_proj.forward(out)[:, off:, :]  # drop seed column -> [S, T, H]
        h = residual + out
        h2 = self.post_attention_layernorm.forward(h)
        gated = F.silu(self.gate_proj.forward(h2)) * self.up_proj.forward(h2)
        return h + self.down_proj.forward(gated)


class DFlashCCADraftModel(BaseOP):
    """CCA-aware DFlash drafter trunk: fc aux-combiner + N _CCADrafterLayers + norm. Tied vocab
    (borrows the target embed_tokens + lm_head). Replicated across TP ranks (tiny drafter)."""

    def __init__(self, hidden_size: int, intermediate_size: int, num_layers: int, num_q_heads: int,
                 num_k_heads: int, head_dim: int, num_aux_layers: int, target_hidden: int,
                 conv_kernel: int, rms_norm_eps: float, clamp_temp: bool, block_mixer: str) -> None:
        self.hidden_size = hidden_size
        self.num_aux_layers = num_aux_layers
        self.num_layers = num_layers
        self.fc = _PlainLinear(num_aux_layers * target_hidden, hidden_size)
        self.norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.layers = [
            _CCADrafterLayer(hidden_size, intermediate_size, num_q_heads, num_k_heads, head_dim,
                             conv_kernel, rms_norm_eps, clamp_temp, block_mixer)
            for _ in range(num_layers)
        ]
        self._embed = None
        self._lm_head = None

    def bind_embed(self, embed) -> None:
        self._embed = embed

    def bind_lm_head(self, lm_head) -> None:
        self._lm_head = lm_head

    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        assert self._embed is not None, "CCA draft embed not bound"
        return self._embed.forward(tokens)

    def head(self, hidden: torch.Tensor) -> torch.Tensor:
        assert self._lm_head is not None, "CCA draft lm_head not bound"
        return self._lm_head.logits_all_rows(hidden)

    def fuse_aux(self, aux: torch.Tensor) -> torch.Tensor:
        """Fold the concat aux -> per-sequence seed. aux: [S, n_aux, target_hidden] -> [S, hidden].
        NB: unlike the Qwen drafter there is NO hidden_norm — the CCA seed is fc(aux) directly."""
        flat = aux.reshape(aux.shape[0], -1)  # [S, n_aux*target_hidden]
        return self.fc.forward(flat)

    @torch.inference_mode()
    def denoise(self, noise_embed: torch.Tensor, seed: Optional[torch.Tensor]) -> torch.Tensor:
        """One block forward. noise_embed: [S, T, H]; seed: [S, H] or None -> [S, T, H]."""
        x = noise_embed
        for layer in self.layers:
            x = layer.forward(x, seed)
        return self.norm.forward(x)


__all__ = ["DFlashCCADraftModel"]
