"""EAGLE3 draft model for GLM-4.7-Flash (thoughtworks/GLM-4.7-Flash-Eagle3).

A SELF-CONTAINED, single-layer Llama GQA draft head that proposes a linear chain of tokens for the
GLM-4.7-Flash (glm4_moe_lite) MLA target. It is NOT loaded through the engine's target-weight
machinery — it is a separate 15-tensor checkpoint with a *different* architecture (standard GQA, not
MLA) and its OWN compressed draft vocab (32000). The DraftModelProposer (spec/draft_model.py) owns
this module, loads its weights directly, binds the target's embed table, and runs it autoregressively.

Architecture (config.json: model_type=llama, LlamaForCausalLMEagle3, hidden=2048 == target hidden):
  - fc(3*hidden -> hidden): fuses the 3 captured target decoder-layer aux states into one feature.
  - midlayer: a Llama GQA decoder layer with a WIDENED first-layer input. q/k/v_proj take in=2*hidden,
    consuming concat[ input_layernorm(embed(tok)), hidden_norm(fc_out) ]. 16 q-heads / 4 kv-heads /
    head_dim 128 (GQA), full RoPE (rope_theta 1e6) on the whole 128-dim head, SwiGLU MLP (8192).
  - norm + lm_head(32000): the draft's OWN final norm and untied head over the COMPRESSED draft vocab.
  - d2t [32000] (delta): target_id = draft_id + d2t[draft_id]; t2d is the inverse membership mask
    (loaded but unused at inference — the proposer maps draft->target ids via d2t).

The chain is run with a tiny per-request causal cache (built fresh each propose, like GLMMTPAttention
.forward_draft) — it never touches the engine's paged KV. Step 0 fuses the captured target aux +
the confirmed token's embedding; subsequent steps feed the draft's OWN output hidden + the new draft
token's embedding (standard EAGLE3 autoregression). See spec/draft_model.py for the loop and the
d2t / target-vocab mapping.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn.functional as F
from minisgl.layers import RMSNorm, get_rope, silu_and_mul
from minisgl.layers.base import BaseOP


class _PlainLinear(BaseOP):
    """A replicated nn.Linear-shaped weight [out, in] with no bias and no TP sharding. The draft is
    tiny (278 MB) so it is replicated on every rank; its lm_head over the 32000 draft vocab produces
    the same argmax on each rank (drafts stay in sync, no collective needed)."""

    def __init__(self, in_features: int, out_features: int) -> None:
        self.weight = torch.empty(out_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight)


class GLMEagle3DraftModel(BaseOP):
    """One-layer EAGLE3 Llama-GQA draft head over the GLM-4.7-Flash target's hidden space.

    Owns every weight EXCEPT the token embedding, which it borrows from the target (the checkpoint
    ships no embed_tokens; draft hidden == target hidden == 2048). `bind_embed` installs the borrowed
    embedding before the first propose. The whole module is REPLICATED across TP ranks.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        num_aux_layers: int,
        draft_vocab_size: int,
        target_vocab_size: int,
        rms_norm_eps: float,
        rope_theta: float,
        max_position: int,
    ) -> None:
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.q_dim = num_heads * head_dim
        self.kv_dim = num_kv_heads * head_dim
        self.scale = head_dim ** -0.5
        self.draft_vocab_size = draft_vocab_size

        # fc: fuse the N captured target aux layers (N*hidden) -> hidden.
        self.fc = _PlainLinear(num_aux_layers * hidden_size, hidden_size)

        # midlayer norms. input_layernorm norms the token-EMBEDDING branch; hidden_norm norms the
        # fused-aux (fc output) branch. Both feed the WIDENED qkv input (concat of the two).
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.hidden_norm = RMSNorm(hidden_size, eps=rms_norm_eps)

        # GQA attention with widened input (in = 2*hidden). q/k/v separate (Llama GQA), no bias.
        self.q_proj = _PlainLinear(2 * hidden_size, self.q_dim)
        self.k_proj = _PlainLinear(2 * hidden_size, self.kv_dim)
        self.v_proj = _PlainLinear(2 * hidden_size, self.kv_dim)
        self.o_proj = _PlainLinear(self.q_dim, hidden_size)

        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)

        # SwiGLU MLP (separate gate/up).
        self.gate_proj = _PlainLinear(hidden_size, intermediate_size)
        self.up_proj = _PlainLinear(hidden_size, intermediate_size)
        self.down_proj = _PlainLinear(intermediate_size, hidden_size)

        # Final norm + own untied head over the COMPRESSED draft vocab.
        self.norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.lm_head = _PlainLinear(hidden_size, draft_vocab_size)

        # Full RoPE over the whole head_dim (rope_type=default, theta=rope_theta). Contrast the
        # target MLA's 64-dim partial rope.
        self.rotary = get_rope(
            head_dim=head_dim,
            rotary_dim=head_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=None,
        )

        # Borrowed target embedding (installed by bind_embed). The draft ships no embed_tokens.
        self._embed = None  # type: ignore[assignment]
        # d2t (delta) draft->target id map and t2d membership mask, loaded directly into the proposer.
        self.d2t = torch.empty(draft_vocab_size, dtype=torch.int64)
        self.t2d = torch.empty(target_vocab_size, dtype=torch.bool)

    def bind_embed(self, embed) -> None:
        """Install the borrowed target embedding module (VocabParallelEmbedding). The draft does not
        ship its own embed_tokens — draft hidden == target hidden, so the table is dimension-clean."""
        self._embed = embed

    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        assert self._embed is not None, "draft embed not bound (call bind_embed)"
        return self._embed.forward(tokens)

    def fuse_aux(self, aux: torch.Tensor) -> torch.Tensor:
        """Fuse the captured target aux states into one feature. aux: [B, num_aux_layers, hidden]
        (or [num_aux_layers, hidden] for a single request). Returns [B, hidden]."""
        flat = aux.reshape(aux.shape[0], -1)  # [B, num_aux_layers*hidden]
        return self.fc.forward(flat)

    def _mlp(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj.forward(x)
        up = self.up_proj.forward(x)
        return self.down_proj.forward(silu_and_mul(torch.cat([gate, up], dim=-1)))

    def step(
        self,
        embed_e: torch.Tensor,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        cache: List,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One autoregressive EAGLE3 draft step over all B requests.

        embed_e:   [B, hidden]  the embedding of THIS step's input token.
        hidden:    [B, hidden]  the previous-feature hidden — for step 0 this is fc(aux) (the fused
                   target feature); for later steps it is the draft's OWN output hidden. It is BOTH
                   the hidden_norm branch input AND the block residual (standard llama_eagle3: the
                   midlayer's residual is its hidden-state input).
        positions: [B] absolute RoPE positions.
        cache:     list growing per step; each entry (k [B,Hkv,hd], v [B,Hkv,hd]).
        Returns (draft_logits [B, draft_vocab], output_hidden [B, hidden]) — output_hidden feeds the
        NEXT step's hidden + the head produces the draft token.
        """
        B = embed_e.shape[0]
        H, Hkv, hd = self.num_heads, self.num_kv_heads, self.head_dim

        widened = self._widened(embed_e, hidden)  # [B, 2*hidden]

        q = self.q_proj.forward(widened).view(B, H, hd)
        k = self.k_proj.forward(widened).view(B, Hkv, hd)
        v = self.v_proj.forward(widened).view(B, Hkv, hd)

        # Full RoPE over head_dim. The shared rotary applies to q (H heads) and k (Hkv heads) jointly.
        q_flat, k_flat = self.rotary.forward(
            positions, q.reshape(B, H * hd).contiguous(), k.reshape(B, Hkv * hd).contiguous()
        )
        q = q_flat.view(B, H, hd)
        k = k_flat.view(B, Hkv, hd)

        cache.append((k, v))
        group = H // Hkv
        Ks = torch.stack([c[0] for c in cache], dim=0)  # [S, B, Hkv, hd]
        Vs = torch.stack([c[1] for c in cache], dim=0)  # [S, B, Hkv, hd]
        Ks = Ks.repeat_interleave(group, dim=2)  # [S, B, H, hd]
        Vs = Vs.repeat_interleave(group, dim=2)  # [S, B, H, hd]
        # scores[b,h,s] = q[b,h]·k[s,b,h]; attend keys 0..step (causal, current included).
        scores = torch.einsum("bhd,sbhd->bhs", q, Ks) * self.scale  # [B, H, S]
        probs = scores.softmax(dim=-1).to(Vs.dtype)
        attn = torch.einsum("bhs,sbhd->bhd", probs, Vs)  # [B, H, hd]
        attn_out = self.o_proj.forward(attn.reshape(B, H * hd))  # [B, hidden]

        # Llama post-norm residual: residual = hidden (the midlayer's hidden input).
        residual = hidden + attn_out
        normed = self.post_attention_layernorm.forward(residual)
        out_hidden = residual + self._mlp(normed)  # [B, hidden]
        logits = self.lm_head.forward(self.norm.forward(out_hidden))  # [B, draft_vocab]
        return logits, out_hidden

    # ---- CUDA-graph-capturable twins of step()/seed_kv() (see spec/capture.py) -------------------
    def draft_buffer_dims(self) -> Tuple[int, int, int, int]:
        """(n_k_heads, k_dim, n_v_heads, v_dim) for the shared propose draft-KV buffer. Llama GQA:
        nkv heads with a symmetric head_dim for both K and V."""
        return self.num_kv_heads, self.head_dim, self.num_kv_heads, self.head_dim

    def _widened(self, embed_e: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """The widened first-layer input concat[ input_layernorm(embed), hidden_norm(hidden) ].
        Shared by step / step_masked / seed_buffered so the three can never drift apart."""
        return torch.cat(
            [self.input_layernorm.forward(embed_e), self.hidden_norm.forward(hidden)], dim=-1)

    def step_masked(
        self,
        embed_e: torch.Tensor,     # [B, hidden]
        hidden: torch.Tensor,      # [B, hidden]
        positions: torch.Tensor,   # [B] absolute RoPE position of this token per row
        k_buf: torch.Tensor,       # [max_slots, max_ctx, Hkv, hd] GLOBAL persistent draft K
        v_buf: torch.Tensor,       # [max_slots, max_ctx, Hkv, hd] GLOBAL persistent draft V
        slot_rows: torch.Tensor,   # [B] which global slot (= req.table_idx) each row uses
        write_col: torch.Tensor,   # [B] column this token is written at, per row
        mask_bias: torch.Tensor,   # [B, max_ctx] additive: 0 for cols <= write_col, -inf beyond
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """CUDA-graph-capturable equivalent of ``step``: fixed-shape masked attention over a GLOBAL
        persistent draft-KV buffer keyed by ``slot_rows``, instead of a ``torch.stack`` over a growing
        Python list (a dynamic contraction dim AND a host-side list mutation — two capture blockers).

        Rows have DIFFERENT context lengths, which is exactly why one sliced tensor cannot serve the
        batch: each row writes its k/v at its own ``write_col`` and attends the whole ``max_ctx``
        window with ``-inf`` beyond it. ``softmax(-inf) == 0`` makes that arithmetically the same
        attention as the sliced stack over the row's real prefix (validated per row by
        tools/eagle3_masked_parity.py).

        NOTE ON SCOPE, and it matters for how an A/B is read: unlike DFlash, EAGLE3 is NOT a
        sliding-window drafter — ``step`` attends every cached key with no mask. So bounding the
        buffer at ``max_ctx`` is a real (if lossless) change for prefixes longer than the window: the
        drafts can differ. Losslessness comes from verify gating every emitted token, NOT from
        byte-equality of the drafts. Gate an EAGLE3 change on accept-len at a stated window."""
        B = embed_e.shape[0]
        H, Hkv, hd = self.num_heads, self.num_kv_heads, self.head_dim
        widened = self._widened(embed_e, hidden)

        q = self.q_proj.forward(widened).view(B, H, hd)
        k = self.k_proj.forward(widened).view(B, Hkv, hd)
        v = self.v_proj.forward(widened).view(B, Hkv, hd)
        q_flat, k_flat = self.rotary.forward(
            positions, q.reshape(B, H * hd).contiguous(), k.reshape(B, Hkv * hd).contiguous()
        )
        q = q_flat.view(B, H, hd)
        k = k_flat.view(B, Hkv, hd)
        # Persist this token's k/v into its slot at write_col (dynamic tensor index — capturable:
        # the graph records the buffer pointer, the indices come from static tensors).
        k_buf[slot_rows, write_col] = k
        v_buf[slot_rows, write_col] = v
        rep = H // Hkv
        # GROUPED-query attention WITHOUT expanding K/V to H heads. The expansion would materialize
        # [B, max_ctx, H, hd] inside the graph — rep x larger and charged to the graph's private pool
        # permanently. Group q as [B, Hkv, rep, hd] and contract against the Hkv-head K/V instead.
        qg = q.view(B, Hkv, rep, hd)
        Ks = k_buf[slot_rows]                                 # [B, max_ctx, Hkv, hd]
        Vs = v_buf[slot_rows]
        scores = torch.einsum("bgrd,bsgd->bgrs", qg, Ks) * self.scale   # [B, Hkv, rep, max_ctx]
        scores = scores + mask_bias.view(B, 1, 1, -1)
        probs = scores.softmax(dim=-1).to(Vs.dtype)
        attn = torch.einsum("bgrs,bsgd->bgrd", probs, Vs).reshape(B, H * hd)
        attn_out = self.o_proj.forward(attn)

        residual = hidden + attn_out
        normed = self.post_attention_layernorm.forward(residual)
        out_hidden = residual + self._mlp(normed)
        logits = self.lm_head.forward(self.norm.forward(out_hidden))  # [B, draft_vocab]
        return logits, out_hidden

    @torch.inference_mode()
    def seed_buffered(
        self,
        embed_e: torch.Tensor,     # [S, hidden]
        hidden: torch.Tensor,      # [S, hidden]
        positions: torch.Tensor,   # [S]
        k_buf: torch.Tensor,
        v_buf: torch.Tensor,
        slot: int,
        start_col: int = 0,
    ) -> None:
        """Seed the GLOBAL draft-KV buffer from the prompt prefill, WITHOUT attention — the buffered
        twin of ``seed_kv``. q/k/v + RoPE mirror ``step_masked`` exactly (via ``_widened``); only the
        attention/MLP/head are dropped, which is a no-op for k/v storage."""
        S = embed_e.shape[0]
        H, Hkv, hd = self.num_heads, self.num_kv_heads, self.head_dim
        widened = self._widened(embed_e, hidden)
        k = self.k_proj.forward(widened).view(S, Hkv, hd)
        v = self.v_proj.forward(widened).view(S, Hkv, hd)
        # The shared rotary rotates q (H heads) and k (Hkv heads) jointly; q is computed to satisfy
        # the call and discarded (the seed stores k/v, not q).
        q = self.q_proj.forward(widened).view(S, H, hd)
        _, k_flat = self.rotary.forward(
            positions, q.reshape(S, H * hd).contiguous(), k.reshape(S, Hkv * hd).contiguous()
        )
        k_buf[slot, start_col : start_col + S] = k_flat.view(S, Hkv, hd)
        v_buf[slot, start_col : start_col + S] = v

    @torch.inference_mode()
    def seed_kv(
        self, embed_e: torch.Tensor, hidden: torch.Tensor, positions: torch.Tensor
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Compute per-position (k, v) for a batch of prompt positions WITHOUT attention — used to
        SEED the persistent draft KV from the prompt prefill (DraftModelProposer.seed_prefill), so the
        first draft sees full prompt context instead of a cold cache. The q/k/v projection MUST mirror
        ``step`` exactly (keep in sync) — only the attention/MLP/head are dropped (the cache only stores
        k/v; attention runs at propose time over the stacked cache).

        embed_e/hidden: [S, hidden] (S = number of prompt positions seeded); positions: [S] RoPE pos.
        Returns a list of S ``(k [1, Hkv, hd], v [1, Hkv, hd])`` entries — same shape/order ``step``
        appends, so the proposer can stack them directly."""
        S = embed_e.shape[0]
        H, Hkv, hd = self.num_heads, self.num_kv_heads, self.head_dim
        widened = self._widened(embed_e, hidden)  # [S, 2*hidden]
        k = self.k_proj.forward(widened).view(S, Hkv, hd)
        v = self.v_proj.forward(widened).view(S, Hkv, hd)
        # RoPE on k uses the shared rotary, which rotates q (H heads) and k (Hkv heads) jointly; compute
        # q only to satisfy the call and discard it (the seed needs k/v, not q).
        q = self.q_proj.forward(widened).view(S, H, hd)
        _, k_flat = self.rotary.forward(
            positions, q.reshape(S, H * hd).contiguous(), k.reshape(S, Hkv * hd).contiguous()
        )
        k = k_flat.view(S, Hkv, hd)
        return [(k[s : s + 1], v[s : s + 1]) for s in range(S)]


__all__ = ["GLMEagle3DraftModel"]
