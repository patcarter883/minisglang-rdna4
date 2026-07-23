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
        *,
        gated: bool = False,
        sliding_window: int = 0,
    ) -> None:
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.q_dim = num_heads * head_dim
        self.kv_dim = num_kv_heads * head_dim
        self.scale = head_dim ** -0.5
        self._rotary = rotary
        # Laguna gated attention: a per-head SOFTPLUS output gate (self_attn.g_proj [num_heads, hidden])
        # applied to the attention output before o_proj — matches the base LagunaAttention. Qwen3 z-lab
        # drafters have no gate (gated=False) and this path is byte-identical to before.
        self.gated = gated
        self.sliding_window = sliding_window
        self.g_proj = _PlainLinear(hidden_size, num_heads) if gated else None

        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.q_proj = _PlainLinear(hidden_size, self.q_dim)
        self.k_proj = _PlainLinear(hidden_size, self.kv_dim)
        self.v_proj = _PlainLinear(hidden_size, self.kv_dim)
        self.o_proj = _PlainLinear(self.q_dim, hidden_size)
        # Per-head RMSNorm over head_dim (plain-weight; both the Qwen3 z-lab draft and the Laguna
        # draft use the plain-weight convention, NOT the (1+weight) Qwen3.5 one).
        self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps)
        self.gate_proj = _PlainLinear(hidden_size, intermediate_size)
        self.up_proj = _PlainLinear(hidden_size, intermediate_size)
        self.down_proj = _PlainLinear(intermediate_size, hidden_size)

    def _mlp(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj.forward(x)
        up = self.up_proj.forward(x)
        return self.down_proj.forward(silu_and_mul(torch.cat([gate, up], dim=-1)))

    def project_ctx(
        self,
        target_hidden: torch.Tensor,  # [m, hidden]  fc+hidden_norm'd captured context
        ctx_pos: torch.Tensor,        # [m]  RoPE positions for these prefix rows
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project prefix rows through THIS layer's k/v_proj (+ k_norm + rotary). The result depends
        ONLY on the (fixed) captured target hidden of already-committed positions, so it is CACHEABLE:
        the persistent-KV proposer projects each committed position exactly ONCE and appends it, instead
        of re-projecting the whole context every step (the O(P)-per-token re-feed this replaces).
        Returns (k_ctx [m, Hkv, hd] post-rotary, v_ctx [m, Hkv, hd])."""
        m = target_hidden.shape[0]
        Hkv, hd = self.num_kv_heads, self.head_dim
        k_ctx = self.k_proj.forward(target_hidden).view(m, Hkv, hd)
        v_ctx = self.v_proj.forward(target_hidden).view(m, Hkv, hd)
        self.k_norm.forward_inplace(k_ctx)
        _, kc_flat = self._rotary.forward(
            ctx_pos, k_ctx.reshape(m, Hkv * hd).contiguous(), k_ctx.reshape(m, Hkv * hd).contiguous()
        )
        return kc_flat.view(m, Hkv, hd), v_ctx

    def attend_block(
        self,
        hidden: torch.Tensor,     # [B, hidden]  noise block hidden
        block_pos: torch.Tensor,  # [B]  RoPE positions for the noise block
        k_ctx: torch.Tensor,      # [P, Hkv, hd]  post-rotary prefix K (cached or freshly projected)
        v_ctx: torch.Tensor,      # [P, Hkv, hd]  prefix V
        attn_mask: Optional[torch.Tensor] = None,  # [B, P+B] additive mask (0/-inf); None => bidirectional
    ) -> torch.Tensor:
        """The block half of the layer forward: project the noise queries/KV, then attend over
        [prefix K/V | noise K/V]. `attn_mask` None => bidirectional (z-lab Qwen); a [B, P+B] additive
        causal+sliding-window mask => Laguna (causal=true, window=512). `k_ctx`/`v_ctx` is the (possibly
        persistent) target-context prefix from `project_ctx`."""
        B = hidden.shape[0]
        H, Hkv, hd = self.num_heads, self.num_kv_heads, self.head_dim

        residual = hidden
        x = self.input_layernorm.forward(hidden)

        q = self.q_proj.forward(x).view(B, H, hd)
        k_noise = self.k_proj.forward(x).view(B, Hkv, hd)
        v_noise = self.v_proj.forward(x).view(B, Hkv, hd)

        # Per-head q_norm/k_norm over head_dim, then rotary on the noise block (prefix already rotated).
        self.q_norm.forward_inplace(q)
        self.k_norm.forward_inplace(k_noise)
        q_flat, kn_flat = self._rotary.forward(
            block_pos, q.reshape(B, H * hd).contiguous(), k_noise.reshape(B, Hkv * hd).contiguous()
        )
        q = q_flat.view(B, H, hd)
        k_noise = kn_flat.view(B, Hkv, hd)

        # K/V = [ctx prefix | noise]  along the key sequence.
        K = torch.cat([k_ctx, k_noise], dim=0)  # [P+B, Hkv, hd]
        V = torch.cat([v_ctx, v_noise], dim=0)  # [P+B, Hkv, hd]
        group = H // Hkv
        K = K.repeat_interleave(group, dim=1)  # [P+B, H, hd]
        V = V.repeat_interleave(group, dim=1)
        # scores[b,h,s] = q[b,h] . K[s,h]; attention over the P+B keys (masked for Laguna causal+SWA).
        scores = torch.einsum("bhd,shd->bhs", q, K) * self.scale  # [B, H, P+B]
        if attn_mask is not None:
            scores = scores + attn_mask.unsqueeze(1)  # [B, 1, P+B] broadcast over heads
        probs = scores.softmax(dim=-1).to(V.dtype)
        attn = torch.einsum("bhs,shd->bhd", probs, V)  # [B, H, hd]
        if self.gated:
            # Per-head softplus output gate (fp32, matches base LagunaAttention) before o_proj.
            gate = torch.nn.functional.softplus(self.g_proj.forward(x).float()).to(attn.dtype)  # [B,H]
            attn = attn * gate.unsqueeze(-1)
        attn_out = self.o_proj.forward(attn.reshape(B, H * hd))  # [B, hidden]

        hidden = residual + attn_out
        residual = hidden
        normed = self.post_attention_layernorm.forward(hidden)
        return residual + self._mlp(normed)

    def forward(
        self,
        hidden: torch.Tensor,         # [B, hidden]  noise block hidden
        target_hidden: torch.Tensor,  # [P, hidden]  fc+hidden_norm'd captured context (shared)
        block_pos: torch.Tensor,      # [B]  RoPE positions for the noise block
        ctx_pos: torch.Tensor,        # [P]  RoPE positions for the target prefix
        attn_mask: Optional[torch.Tensor] = None,  # [B, P+B] additive mask (0/-inf); None => bidirectional
    ) -> torch.Tensor:
        # Recompute-every-step path (no persistent KV): project the whole prefix, then attend. Kept
        # byte-identical for the legacy/window fallback; the fast path caches project_ctx across steps.
        k_ctx, v_ctx = self.project_ctx(target_hidden, ctx_pos)
        return self.attend_block(hidden, block_pos, k_ctx, v_ctx, attn_mask)


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
        decoder_layer_type: str = "qwen3",
        sliding_window: int = 0,
        causal: bool = False,
        per_aux_norm: bool = False,
    ) -> None:
        self.hidden_size = hidden_size
        self.num_aux_layers = num_aux_layers
        self.num_layers = num_layers
        # Laguna (decoder_layer_type='laguna_xs'): gated attention, causal block mask + sliding window,
        # and a per-captured-layer RMSNorm on each aux BEFORE fc (aux_hidden_norms). z-lab Qwen3
        # (default): ungated, bidirectional, single hidden_norm after fc.
        gated = decoder_layer_type == "laguna_xs"
        self.causal = causal
        self.sliding_window = sliding_window

        # fc fuses the N captured target aux layers (N*hidden) -> hidden; hidden_norm norms it.
        self.fc = _PlainLinear(num_aux_layers * hidden_size, hidden_size)
        self.hidden_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        # Per-aux-layer RMSNorm applied to each captured target hidden before concat+fc (Laguna).
        self.aux_hidden_norms = (
            [RMSNorm(hidden_size, eps=rms_norm_eps) for _ in range(num_aux_layers)]
            if per_aux_norm
            else None
        )

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
                gated=gated,
                sliding_window=sliding_window,
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
        """fc + hidden_norm of the captured concat. aux: [P, N_aux, hidden] -> [P, hidden].
        Laguna additionally RMS-norms each captured aux (aux_hidden_norms[i]) before concat."""
        if self.aux_hidden_norms is not None:
            parts = [
                self.aux_hidden_norms[i].forward(aux[:, i, :])
                for i in range(self.num_aux_layers)
            ]
            flat = torch.cat(parts, dim=-1)  # [P, N_aux*hidden]
        else:
            flat = aux.reshape(aux.shape[0], -1)  # [P, N_aux*hidden]
        return self.hidden_norm.forward(self.fc.forward(flat))

    def _block_mask(self, P: int, B: int, device: torch.device) -> Optional[torch.Tensor]:
        """Additive [B, P+B] mask (0 keep / -inf drop) for the Laguna causal + sliding-window block.
        The keys are the P prefix positions followed by the B block positions, all contiguous in
        absolute position (prefix at [base-P .. base-1], block at [base .. base+B-1]); so a query at
        block index i sits at concatenated position i+P and, causally + FlashAttention moving-query
        SWA, attends to keys j with (i+P-window) < j <= (i+P). Returns None when not causal (z-lab
        bidirectional path -> byte-identical to before)."""
        if not self.causal:
            return None
        qpos = torch.arange(B, device=device).view(B, 1) + P  # [B,1] concat position of each query
        kpos = torch.arange(P + B, device=device).view(1, P + B)  # [1,P+B]
        keep = kpos <= qpos
        if self.sliding_window > 0:
            keep = keep & ((qpos - kpos) < self.sliding_window)
        return torch.where(
            keep, torch.zeros((), device=device), torch.full((), float("-inf"), device=device)
        ).float()

    @torch.inference_mode()
    def denoise(
        self,
        noise_embed: torch.Tensor,    # [B, hidden]  embed([anchor, mask, ...])
        target_hidden: torch.Tensor,  # [P, hidden]  fuse_aux output
        block_pos: torch.Tensor,      # [B]
        ctx_pos: torch.Tensor,        # [P]
    ) -> torch.Tensor:
        """One denoising forward -> [B, hidden] (pre-head-normed block hidden). Bidirectional (z-lab)
        or causal+SWA-masked (Laguna) per self.causal."""
        hidden = noise_embed
        mask = self._block_mask(target_hidden.shape[0], noise_embed.shape[0], noise_embed.device)
        for layer in self.layers:
            hidden = layer.forward(hidden, target_hidden, block_pos, ctx_pos, mask)
        return self.norm.forward(hidden)

    @torch.inference_mode()
    def project_prefix(
        self,
        aux_slice: torch.Tensor,   # [m, num_aux, hidden]  captured target aux for m committed positions
        positions: torch.Tensor,   # [m]  absolute RoPE positions of those rows
    ) -> List[tuple]:
        """fc+hidden_norm the captured aux of m committed positions, then project each layer's prefix
        K/V once. Returns a per-layer list of (k_ctx [m, Hkv, hd], v_ctx [m, Hkv, hd]). The persistent-
        KV proposer calls this ONCE per position (on the newly-accepted tail each step) and appends the
        result to its cache — turning the O(P) per-step re-feed into O(new)."""
        target_hidden = self.fuse_aux(aux_slice)  # [m, hidden]
        return [layer.project_ctx(target_hidden, positions) for layer in self.layers]

    @torch.inference_mode()
    def denoise_cached(
        self,
        noise_embed: torch.Tensor,  # [B, hidden]
        prefix_kv: List[tuple],     # per-layer (k_ctx [P, Hkv, hd], v_ctx [P, Hkv, hd]) from project_prefix
        block_pos: torch.Tensor,    # [B]
    ) -> torch.Tensor:
        """Denoising forward against a PRECOMPUTED (persistent) per-layer prefix K/V — the fast path.
        Byte-identical to `denoise` for the same effective prefix, but the prefix projection is reused
        across decode steps instead of recomputed."""
        hidden = noise_embed
        P = prefix_kv[0][0].shape[0]
        mask = self._block_mask(P, noise_embed.shape[0], noise_embed.device)
        for layer, (k_ctx, v_ctx) in zip(self.layers, prefix_kv):
            hidden = layer.attend_block(hidden, block_pos, k_ctx, v_ctx, mask)
        return self.norm.forward(hidden)


__all__ = ["DFlashDraftModel"]
