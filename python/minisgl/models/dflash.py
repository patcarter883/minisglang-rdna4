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


# THE drafter linear now lives in ONE place — models/draft_linear.py — shared by DFlash, the CCA
# drafter (which imports this name) and GLM-EAGLE3 (which used to carry its own diverged copy).
# The class body that used to sit here moved there verbatim, plus a TP `shard` policy; the weight
# FORMAT (bf16 / fp8 / int8 / nvfp4-e2m1) stays a load policy on that one core, never a subclass,
# per KERNEL_CORE_POLICY.md. Aliased so the 12 call sites below are untouched.
from .draft_linear import DraftLinear as _PlainLinear  # noqa: E402


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
        # ONE rope launch + ONE [m, Hkv*hd] staging copy. The old call handed the SAME k tensor in as
        # both `query` and `key` and threw the query result away, i.e. it paid two identical
        # `tail_hip.rope` launches and two `.contiguous()` copies per layer per step for one result.
        # Bit-identical: same kernel, same input, same positions.
        kc_flat = self._rotary.forward_one(ctx_pos, k_ctx.reshape(m, Hkv * hd).contiguous())
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
        K = torch.cat([k_ctx, k_noise], dim=0)  # [S, Hkv, hd], S = P + B
        V = torch.cat([v_ctx, v_noise], dim=0)  # [S, Hkv, hd]
        group = H // Hkv
        # The group-expanded K/V stays. NOT an oversight — MEASURED, min-of-7, in
        # tools/dflash_gqa_formulation_probe.py: the "carry a group axis in the einsum" rewrite (`bkgd,skd->bkgs` / `bkgs,skd->bkgd`)
        # changes the underlying bmm from (batch=H, M=B, K=hd) to (batch=Hkv, M=B*group, K=hd) and the
        # AV product from (batch=H, K=S) to (batch=Hkv, K=S). rocBLAS partitions those differently, so
        # it is NOT bit-identical (dmax 2e-6..5e-4 on the attention output, ~1 bf16 ULP), and in
        # tools/dflash_window_parity.py that was enough to FLIP a drafted token's argmax at P=512.
        # A stride-0 broadcast `matmul` formulation IS bit-identical (dmax exactly 0 at every S) but
        # torch materialises the broadcast anyway and it runs SLOWER than this (0.235 vs 0.160 ms at
        # S=528). And the expansion is no longer the cost it was: once the window slice above caps
        # S at sliding_window + block, these two copies are ~8.6 MB each per layer, not the ~492 MB
        # they were at a 30k prefix — the grouped einsum's whole remaining edge at S=528 is 3.8% of
        # the attention core (0.154 vs 0.160 ms), which does not buy a drafted-token flip.
        K = K.repeat_interleave(group, dim=1)  # [S, H, hd]
        V = V.repeat_interleave(group, dim=1)
        # scores[b,h,s] = q[b,h] . K[s,h]; attention over the S keys (masked for Laguna causal+SWA).
        scores = torch.einsum("bhd,shd->bhs", q, K) * self.scale  # [B, H, S]
        if attn_mask is not None:
            scores = scores + attn_mask.unsqueeze(1)  # [B, 1, S] broadcast over heads
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

    def attend_block_batched(
        self,
        hidden: torch.Tensor,     # [N, Q, hidden]  noise block hidden, N requests x Q block rows
        block_pos: torch.Tensor,  # [N, Q] int32 RoPE positions for the noise block
        k_ctx: torch.Tensor,      # [N, C, Hkv, hd]  post-rotary prefix K (a ring slice of the pool)
        v_ctx: torch.Tensor,      # [N, C, Hkv, hd]
        attn_mask: torch.Tensor,  # [N, Q, C+Q] additive 0/-inf
    ) -> torch.Tensor:
        """BATCHED, CUDA-graph-capturable twin of `attend_block`.

        Three differences from the per-request form, each forced by capture (see spec/capture.py):
          * N requests in ONE forward — the per-request Python loop is host control flow, which a
            graph cannot contain, and it was also serialising every request's ~150 launches.
          * the prefix is a FIXED [N, C, ...] ring slice with an additive mask, not a variable [P,...]
            slice: a data-dependent contraction dim cannot be captured.
          * GROUPED-query contraction instead of `repeat_interleave(group)`. The expansion would
            materialise [N, C+Q, H, hd] — 8x larger — INSIDE the graph, where every allocation is
            charged to the graph's private pool permanently (this is the same allocation that OOM'd
            the MTP propose pool before it was grouped). At the O(window) shape it is 8.6 MB/layer
            expanded, so the trade that kept `repeat_interleave` in the eager path (bit-identity at
            a 3.8% cost) does not survive multiplication by the batch and the pool.

        NOT bit-identical to `attend_block` — the reduction regroups (different bmm shapes, longer
        masked-out key axis). It IS bit-identical to ITSELF eager vs replayed, which is the property
        capture has to have; losslessness of the emitted tokens comes from verify, as always."""
        N, Q = hidden.shape[0], hidden.shape[1]
        H, Hkv, hd = self.num_heads, self.num_kv_heads, self.head_dim
        group = H // Hkv
        T = N * Q

        flat = hidden.reshape(T, -1)
        residual = flat
        x = self.input_layernorm.forward(flat)

        q = self.q_proj.forward(x).view(T, H, hd)
        k_noise = self.k_proj.forward(x).view(T, Hkv, hd)
        v_noise = self.v_proj.forward(x).view(T, Hkv, hd)
        self.q_norm.forward_inplace(q)
        self.k_norm.forward_inplace(k_noise)
        q_flat, kn_flat = self._rotary.forward(
            block_pos.reshape(T), q.reshape(T, H * hd).contiguous(),
            k_noise.reshape(T, Hkv * hd).contiguous()
        )
        # [N, Q, Hkv, group, hd] — the regrouping that matches repeat_interleave's head mapping
        # (expanded head h reads kv head h // group, so q head h = kv*group + r).
        qg = q_flat.view(N, Q, Hkv, group, hd)
        k_noise = kn_flat.view(N, Q, Hkv, hd)
        v_noise = v_noise.view(N, Q, Hkv, hd)

        K = torch.cat([k_ctx, k_noise], dim=1)   # [N, S, Hkv, hd], S = C + Q
        V = torch.cat([v_ctx, v_noise], dim=1)
        scores = torch.einsum("nqgrd,nsgd->nqgrs", qg, K) * self.scale       # [N,Q,Hkv,group,S]
        scores = scores + attn_mask.view(N, Q, 1, 1, -1)
        probs = scores.softmax(dim=-1).to(V.dtype)
        attn = torch.einsum("nqgrs,nsgd->nqgrd", probs, V).reshape(T, H, hd)
        if self.gated:
            gate = torch.nn.functional.softplus(self.g_proj.forward(x).float()).to(attn.dtype)
            attn = attn * gate.unsqueeze(-1)
        attn_out = self.o_proj.forward(attn.reshape(T, H * hd))

        h = residual + attn_out
        return (h + self._mlp(self.post_attention_layernorm.forward(h))).view(N, Q, -1)

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

    def window_prefix(self, P: int) -> int:
        """How many TRAILING prefix rows the block can actually attend to. 0-cost, host-only.

        Under the Laguna causal + sliding-window mask (`_block_mask`) the FIRST block query sits at
        concatenated position P and keeps only keys j with P - j < W, so every row j <= P - W is
        `-inf` for EVERY query in the block. Those rows contribute exp(-inf - max) == 0.0 exactly and
        0.0 * V == 0.0, so materialising and reading them is pure traffic — at a 30k prefix, ~4.9 GB
        of transient alloc and ~13 GB of HBM traffic per propose, to compute something that depends
        on 512 keys.

        NO position-base shift is needed after slicing, and this is the one place it is easy to get
        wrong. `_block_mask` indexes the CONCATENATION, not absolute positions. Dropping the oldest
        D = P - W keys maps key j -> j - D and query i from concat position i + P to i + W:
          * distance  (i + P) - j  ==  (i + W) - (j - D)         -> the SWA test is preserved;
          * causality  j <= i + P  <=>  j - D <= i + W           -> the causal test is preserved.
        So `_block_mask(min(P, W), B)` IS the sliced mask, unshifted. RoPE is unaffected regardless:
        every prefix row was rotated at its own TRUE absolute position when it was projected
        (`project_ctx`), so a row carries its phase with it and slicing moves no phase.

        (Row P - W itself is also always masked — for query i, qpos - kpos = i + W >= W — so W - 1
        rows would suffice; W is kept as the conservative, easier-to-reason-about bound.)"""
        if not self.causal or self.sliding_window <= 0:
            return P  # z-lab bidirectional drafter: NO mask at all, every key is live. Never slice.
        return min(P, self.sliding_window)

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
        P_full = target_hidden.shape[0]
        P = self.window_prefix(P_full)
        if P < P_full:  # windowed: drop the prefix rows the mask discards BEFORE projecting them
            target_hidden = target_hidden[P_full - P :]
            ctx_pos = ctx_pos[P_full - P :]
        mask = self._block_mask(P, noise_embed.shape[0], noise_embed.device)
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
        P_full = prefix_kv[0][0].shape[0]
        P = self.window_prefix(P_full)
        if P < P_full:
            # SLICE BEFORE THE CAT: a contiguous view of the newest W rows, so `attend_block`'s
            # torch.cat + einsums see [W + B] keys instead of [P + B]. Constant-shaped once P >= W.
            d = P_full - P
            prefix_kv = [(k[d:], v[d:]) for (k, v) in prefix_kv]
        mask = self._block_mask(P, noise_embed.shape[0], noise_embed.device)
        for layer, (k_ctx, v_ctx) in zip(self.layers, prefix_kv):
            hidden = layer.attend_block(hidden, block_pos, k_ctx, v_ctx, mask)
        return self.norm.forward(hidden)

    # ---- CUDA-graph-capturable batched propose (see spec/capture.py, spec/dflash.py) -------------
    def project_prefix_into(
        self,
        aux: torch.Tensor,          # [m, num_aux, hidden]  captured target aux, m rows
        rope_pos: torch.Tensor,     # [m] int32 absolute RoPE position of each row
        k_pool: List[torch.Tensor],  # per-layer [slots, C, Hkv, hd] persistent ring
        v_pool: List[torch.Tensor],
        wslot: torch.Tensor,        # [m] destination slot per row (NULL slot = discard)
        wcol: torch.Tensor,         # [m] destination ring column per row
    ) -> None:
        """fc+hidden_norm the captured aux of m committed positions and SCATTER each layer's prefix
        K/V straight into the persistent ring. Fixed-shape and sync-free, so it runs INSIDE the
        captured body — which is what makes the whole propose captured rather than "captured except
        for the part that keeps the drafter's context up to date".

        Rows whose ``wslot`` is the NULL slot are computed and thrown away. That is deliberate: how
        MANY positions were newly committed varies per request per step (1..block), and a variable
        row count is exactly what a graph cannot express. Projecting a FIXED block-sized tail every
        step and discarding the overhang is idempotent — the scheduler only ever APPENDS accepted
        positions to the aux buffer, so re-projecting an already-projected position reproduces the
        identical K/V, bit for bit."""
        target_hidden = self.fuse_aux(aux)  # [m, hidden]
        for l, layer in enumerate(self.layers):
            k, v = layer.project_ctx(target_hidden, rope_pos)
            k_pool[l][wslot, wcol] = k
            v_pool[l][wslot, wcol] = v

    def denoise_batched(
        self,
        noise_embed: torch.Tensor,   # [N, Q, hidden]
        block_pos: torch.Tensor,     # [N, Q] int32
        k_pool: List[torch.Tensor],  # per-layer [slots, C, Hkv, hd]
        v_pool: List[torch.Tensor],
        slots: torch.Tensor,         # [N] which ring slot each request reads
        mask: torch.Tensor,          # [N, Q, C+Q] additive 0/-inf
    ) -> torch.Tensor:
        """One BATCHED denoising forward over the persistent ring -> [N, Q, hidden].

        The per-layer gather ``k_pool[l][slots]`` is issued INSIDE the layer loop, not hoisted: the
        caching allocator then reuses one layer's [N, C, Hkv, hd] transient for the next, so the
        graph's private pool holds one layer's worth rather than all of them."""
        hidden = noise_embed
        for l, layer in enumerate(self.layers):
            hidden = layer.attend_block_batched(
                hidden, block_pos, k_pool[l][slots], v_pool[l][slots], mask)
        return self.norm.forward(hidden.reshape(-1, hidden.shape[-1])).view_as(hidden)


__all__ = ["DFlashDraftModel"]
