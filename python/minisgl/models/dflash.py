"""DFlash block-diffusion draft model (z-lab DFlashDraftModel).

A SELF-CONTAINED N-layer Qwen3/Llama-GQA draft trunk that proposes a whole BLOCK of candidate
tokens in ONE bidirectional forward (block-diffusion denoising), NOT autoregressively. It is loaded
directly from a DFlash safetensors checkpoint by the DFlashProposer (spec/dflash.py) — NOT through
the engine's target-weight machinery — because it is a *different* architecture from the target
(standard GQA decoder layers, not the target's GDN-hybrid / MLA mixer).

Two checkpoint dialects exist (see SPEC_DECODE.md / the proposer). This module covers the tied-vocab
z-lab variant (Checkpoint A, z-lab/Qwen3.5-4B-DFlash): NO embed_tokens, NO lm_head, NO d2t/t2d — it
borrows the target's embed_tokens + lm_head (vocab identical). The pruned-vocab "speculators" variant
(Checkpoint B, poolside/Laguna) additionally ships its own embed/lm_head(pruned)/d2t/t2d; this class
accepts an optional own embed/lm_head + d2t to cover it, but the validated path is Checkpoint A.

Forward mechanism (z-lab dflash/model.py DFlashDraftModel.forward):

    target_hidden = hidden_norm( fc( captured_concat ) )      # [P, hidden]  fc: N_aux*hidden -> hidden
    hidden_states = noise_embedding                           # embed([anchor, mask, mask, ...])  [B, hidden]
    for layer in layers:
        hidden_states = layer(hidden_states, target_hidden, positions)
    return norm(hidden_states)                                # [B, hidden] -> lm_head -> block logits

The KV-INJECTION seam is in each layer's attention (Qwen3DFlashAttention):

    Q = q_proj(hidden_states)                                 # draft noise queries
    K = cat([ k_proj(target_hidden), k_proj(hidden_states) ]) # target-context KV PREFIX + noise KV
    V = cat([ v_proj(target_hidden), v_proj(hidden_states) ])
    attn(Q, K, V, is_causal=False)                            # BIDIRECTIONAL within the block

So the fc-projected captured target hidden states are injected as a per-layer KV PREFIX (concatenated
BEFORE the noise KV), and the noise block attends bidirectionally over [target prefix + whole block].
A single denoising forward emits block_size hidden vectors; lm_head over positions 1..B-1 yields the
B-1 candidate draft tokens (position 0 is the already-known anchor). Verification stays the existing
linear verify_greedy (DFlash is a linear block, not a tree).
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F
from minisgl.layers import RMSNorm, get_rope, silu_and_mul
from minisgl.layers.base import BaseOP


class _PlainLinear(BaseOP):
    """A replicated nn.Linear-shaped weight [out, in], no bias, no TP sharding. The draft is tiny and
    REPLICATED on every rank — its argmax is identical per rank, so drafts stay in sync with no
    collective.

    Optional weight-only quant (fp8 E4M3 or int8, per-output-channel scale) via `load_quant`: the
    drafter's output is verified by the target, so this is LOSSLESS — it trades a little draft
    acceptance for ~half the drafter memory (needed to fit big drafters replicated on 16 GB cards).
    Dequant to the activation dtype happens in-forward (no fp8 GEMM needed)."""

    def __init__(self, in_features: int, out_features: int) -> None:
        self.weight = torch.empty(out_features, in_features)
        self._wq = None   # [out, in] fp8_e4m3 / int8 quantized weight
        self._ws = None   # [out, 1] per-output-channel scale (compute dtype)

    def load_quant(self, w: torch.Tensor, mode: str, compute_dtype, device) -> None:
        """RTN weight-only quant of a loaded [out, in] weight (pass it on CPU so the fp16 transient
        stays in host RAM and only the 1-byte packed weight lands on GPU). mode: 'fp8' | 'int8'."""
        wf = w.float()
        amax = wf.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)  # [out,1]
        if mode == "fp8":
            fmax = 448.0  # E4M3 max
            s = amax / fmax
            wq = (wf / s).clamp(-fmax, fmax).to(torch.float8_e4m3fn)
        else:  # int8
            s = amax / 127.0
            wq = (wf / s).round().clamp(-127, 127).to(torch.int8)
        self._wq = wq.contiguous().to(device)
        self._ws = s.to(compute_dtype).to(device)
        self.weight = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._wq is not None:
            w = self._wq.to(x.dtype) * self._ws  # dequant [out,in] * [out,1]
            return F.linear(x, w)
        return F.linear(x, self.weight)


class _DFlashLayer(BaseOP):
    """One DFlash Qwen3-style GQA decoder layer with the target-context KV-prefix injection.

    Pre-norm residual: input_layernorm -> attn(noise Q over [target-prefix K | noise K]) -> residual,
    post_attention_layernorm -> SwiGLU MLP -> residual. Per-head q_norm/k_norm (Qwen3) over head_dim
    precede rotary. is_causal=False: the block attends bidirectionally over the target prefix + itself.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rms_norm_eps: float,
        rotary,
    ) -> None:
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.q_dim = num_heads * head_dim
        self.kv_dim = num_kv_heads * head_dim
        self.scale = head_dim ** -0.5
        self._rotary = rotary

        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.q_proj = _PlainLinear(hidden_size, self.q_dim)
        self.k_proj = _PlainLinear(hidden_size, self.kv_dim)
        self.v_proj = _PlainLinear(hidden_size, self.kv_dim)
        self.o_proj = _PlainLinear(self.q_dim, hidden_size)
        # Qwen3 per-head RMSNorm over head_dim (plain-weight; the draft is model_type=qwen3, NOT the
        # target's (1+weight) Qwen3.5 convention).
        self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps)
        self.gate_proj = _PlainLinear(hidden_size, intermediate_size)
        self.up_proj = _PlainLinear(hidden_size, intermediate_size)
        self.down_proj = _PlainLinear(intermediate_size, hidden_size)

    def _mlp(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj.forward(x)
        up = self.up_proj.forward(x)
        return self.down_proj.forward(silu_and_mul(torch.cat([gate, up], dim=-1)))

    def forward(
        self,
        hidden: torch.Tensor,         # [B, hidden]  noise block hidden
        target_hidden: torch.Tensor,  # [P, hidden]  fc+hidden_norm'd captured context (shared)
        block_pos: torch.Tensor,      # [B]  RoPE positions for the noise block
        ctx_pos: torch.Tensor,        # [P]  RoPE positions for the target prefix
    ) -> torch.Tensor:
        B = hidden.shape[0]
        P = target_hidden.shape[0]
        H, Hkv, hd = self.num_heads, self.num_kv_heads, self.head_dim

        residual = hidden
        x = self.input_layernorm.forward(hidden)

        # Noise queries + KV; target-context KV prefix (projected through THIS layer's k/v_proj).
        q = self.q_proj.forward(x).view(B, H, hd)
        k_noise = self.k_proj.forward(x).view(B, Hkv, hd)
        v_noise = self.v_proj.forward(x).view(B, Hkv, hd)
        k_ctx = self.k_proj.forward(target_hidden).view(P, Hkv, hd)
        v_ctx = self.v_proj.forward(target_hidden).view(P, Hkv, hd)

        # Per-head q_norm/k_norm over head_dim, then rotary (noise at block_pos, prefix at ctx_pos).
        self.q_norm.forward_inplace(q)
        self.k_norm.forward_inplace(k_noise)
        self.k_norm.forward_inplace(k_ctx)
        q_flat, kn_flat = self._rotary.forward(
            block_pos, q.reshape(B, H * hd).contiguous(), k_noise.reshape(B, Hkv * hd).contiguous()
        )
        # The prefix shares k_proj; apply rotary to it at its own positions (q unused -> reuse a slot).
        _, kc_flat = self._rotary.forward(
            ctx_pos, k_ctx.reshape(P, Hkv * hd).contiguous(), k_ctx.reshape(P, Hkv * hd).contiguous()
        )
        q = q_flat.view(B, H, hd)
        k_noise = kn_flat.view(B, Hkv, hd)
        k_ctx = kc_flat.view(P, Hkv, hd)

        # K/V = [ctx prefix | noise]  along the key sequence; bidirectional (no causal mask).
        K = torch.cat([k_ctx, k_noise], dim=0)  # [P+B, Hkv, hd]
        V = torch.cat([v_ctx, v_noise], dim=0)  # [P+B, Hkv, hd]
        group = H // Hkv
        K = K.repeat_interleave(group, dim=1)  # [P+B, H, hd]
        V = V.repeat_interleave(group, dim=1)
        # scores[b,h,s] = q[b,h] . K[s,h]; full (bidirectional) attention over the P+B keys.
        scores = torch.einsum("bhd,shd->bhs", q, K) * self.scale  # [B, H, P+B]
        probs = scores.softmax(dim=-1).to(V.dtype)
        attn = torch.einsum("bhs,shd->bhd", probs, V)  # [B, H, hd]
        attn_out = self.o_proj.forward(attn.reshape(B, H * hd))  # [B, hidden]

        hidden = residual + attn_out
        residual = hidden
        normed = self.post_attention_layernorm.forward(hidden)
        return residual + self._mlp(normed)


class DFlashDraftModel(BaseOP):
    """DFlash block-diffusion draft trunk (REPLICATED across TP ranks).

    fc(N_aux*hidden -> hidden) + hidden_norm materialize the captured target concat into the per-layer
    KV-prefix feature; the N draft layers run the bidirectional denoising block; norm is the final
    pre-head norm. Tied-vocab variant borrows the target embed_tokens + lm_head (bind_embed/bind_head);
    pruned-vocab variant installs its own (load handled by the proposer). The proposer owns the block
    assembly, the single forward, sampling, and (compressed variant) the d2t remap.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_layers: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        num_aux_layers: int,
        rms_norm_eps: float,
        rope_theta: float,
        max_position: int,
        *,
        draft_vocab_size: Optional[int] = None,
        own_embed: bool = False,
    ) -> None:
        self.hidden_size = hidden_size
        self.num_aux_layers = num_aux_layers
        self.num_layers = num_layers

        # fc fuses the N captured target aux layers (N*hidden) -> hidden; hidden_norm norms it.
        self.fc = _PlainLinear(num_aux_layers * hidden_size, hidden_size)
        self.hidden_norm = RMSNorm(hidden_size, eps=rms_norm_eps)

        rotary = get_rope(
            head_dim=head_dim,
            rotary_dim=head_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=None,
        )
        self.layers = [
            _DFlashLayer(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                rms_norm_eps=rms_norm_eps,
                rotary=rotary,
            )
            for _ in range(num_layers)
        ]
        self.norm = RMSNorm(hidden_size, eps=rms_norm_eps)

        # Own embed/lm_head + d2t only for the pruned-vocab (Checkpoint B) variant.
        self.draft_vocab_size = draft_vocab_size
        self._own_embed = None
        self._own_lm_head = None
        if own_embed:
            assert draft_vocab_size is not None
            # Own full-target-vocab embed is sized by the proposer at load (target_vocab); allocate
            # lm_head over the pruned draft vocab here.
            self._own_lm_head = _PlainLinear(hidden_size, draft_vocab_size)
        self.d2t: Optional[torch.Tensor] = None
        self.t2d: Optional[torch.Tensor] = None

        # Borrowed target embed/lm_head (tied-vocab variant). Installed before first propose.
        self._embed = None
        self._lm_head = None

    # ---- binding to the target (tied-vocab variant) ----
    def bind_embed(self, embed) -> None:
        self._embed = embed

    def bind_lm_head(self, lm_head) -> None:
        self._lm_head = lm_head

    def set_own_embed(self, embed_weight: torch.Tensor) -> None:
        self._own_embed = _PlainLinear(0, 0)
        self._own_embed.weight = embed_weight  # [target_vocab, hidden]

    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        if self._own_embed is not None:
            return F.embedding(tokens, self._own_embed.weight)
        assert self._embed is not None, "DFlash draft embed not bound (call bind_embed)"
        return self._embed.forward(tokens)

    def head(self, hidden: torch.Tensor) -> torch.Tensor:
        """Block logits [B, vocab] over the draft head's vocab (pruned for B, target for A)."""
        if self._own_lm_head is not None:
            return self._own_lm_head.forward(hidden)
        assert self._lm_head is not None, "DFlash draft lm_head not bound (call bind_lm_head)"
        # Borrowed target head: logits over the full (identical) target vocab for ALL rows.
        return self._lm_head.logits_all_rows(hidden)

    def fuse_aux(self, aux: torch.Tensor) -> torch.Tensor:
        """fc + hidden_norm of the captured concat. aux: [P, N_aux, hidden] -> [P, hidden]."""
        flat = aux.reshape(aux.shape[0], -1)  # [P, N_aux*hidden]
        return self.hidden_norm.forward(self.fc.forward(flat))

    @torch.inference_mode()
    def denoise(
        self,
        noise_embed: torch.Tensor,    # [B, hidden]  embed([anchor, mask, ...])
        target_hidden: torch.Tensor,  # [P, hidden]  fuse_aux output
        block_pos: torch.Tensor,      # [B]
        ctx_pos: torch.Tensor,        # [P]
    ) -> torch.Tensor:
        """One bidirectional denoising forward -> [B, hidden] (pre-head-normed block hidden)."""
        hidden = noise_embed
        for layer in self.layers:
            hidden = layer.forward(hidden, target_hidden, block_pos, ctx_pos)
        return self.norm.forward(hidden)


__all__ = ["DFlashDraftModel"]
