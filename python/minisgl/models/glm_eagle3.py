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

The chain attends over a persistent per-slot draft-KV RING owned by the proposer (spec/draft_model.py),
read IN PLACE by the attn_decode.flash_decode_paged HIP kernel (spec/draft_attn.py) — it never touches
the engine's paged KV. Step 0 fuses the captured target aux +
the confirmed token's embedding; subsequent steps feed the draft's OWN output hidden + the new draft
token's embedding (standard EAGLE3 autoregression). See spec/draft_model.py for the loop and the
d2t / target-vocab mapping.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
from minisgl.layers import RMSNorm, get_rope, silu_and_mul
from minisgl._hip_engage import engaged
from minisgl.layers.base import BaseOP
from minisgl.spec.draft_attn import paged_draft_attention


# The drafter linear now lives in ONE place (models/draft_linear.py), shared by every draft trunk.
# This was a near-identical COPY of DFlash's class that then diverged, and the divergence cost real
# things: this copy allocated its scaffold in fp32 ON THE CARD (no device="meta"), and it carried no
# weight-only quant path at all — so the EAGLE3 drafter could not be quantized, purely because the
# class had been copy-pasted. Aliased rather than renamed at the 10 call sites, so at tp_size==1 this
# is a pure consolidation with no behavioural delta.
from .draft_linear import DraftLinear as _PlainLinear  # noqa: E402


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

    # ---- the capturable draft step + prompt seed (see spec/capture.py) --------------------------
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
        meta,                      # spec.draft_attn.DraftAttnMeta: block_table [B,R] i32, ctx_lens [B] i32
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One capturable autoregressive EAGLE3 draft step over all B rows.

        embed_e:   the embedding of THIS step's input token.
        hidden:    the previous-feature hidden — fc(aux) at step 0, the draft's OWN output hidden after.
                   It is BOTH the hidden_norm branch input AND the block residual (llama_eagle3).
        Stores this token's k/v into the GLOBAL ring at (slot_rows, write_col), then GQA decode
        attention over the row's visible keys on ``attn_decode.flash_decode_paged``: the ring is read
        in place as page_size-1 pages, the block table / lengths (``meta``) are built on device by the
        proposer from its keep mask, so the kernel reads exactly the keys the old additive -inf mask
        kept — no ``k_buf[slot_rows]`` gather, no work sized from the ring's capacity.
        Returns (draft_logits [B, draft_vocab], output_hidden [B, hidden]).

        NOTE ON SCOPE, and it matters for how an A/B is read: EAGLE3 is NOT a sliding-window drafter,
        so bounding its context at the ring window is a real (if lossless) change for prefixes longer
        than the window: the drafts can differ. Losslessness comes from verify gating every emitted
        token, NOT from byte-equality of the drafts. Gate an EAGLE3 change on accept-len at a stated
        window."""
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
        engaged("attn_decode.flash_decode_paged(eagle3_draft)")
        attn = paged_draft_attention(q, k_buf, v_buf, meta, self.scale).reshape(B, H * hd)
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
        """Seed the GLOBAL draft-KV ring from the prompt prefill, WITHOUT attention. k/v + RoPE mirror
        ``step_masked`` exactly (via ``_widened``); only the attention/MLP/head are dropped, which is a
        no-op for k/v storage. Writes S rows at k_buf/v_buf[slot, start_col:start_col+S]."""
        S = embed_e.shape[0]
        Hkv, hd = self.num_kv_heads, self.head_dim
        widened = self._widened(embed_e, hidden)
        k = self.k_proj.forward(widened)
        v = self.v_proj.forward(widened).view(S, Hkv, hd)
        # k rotated alone: forward_one is bit-identical to the key half of forward(), so no q
        # projection is computed just to be discarded.
        k_flat = self.rotary.forward_one(positions, k.contiguous())
        k_buf[slot, start_col : start_col + S] = k_flat.view(S, Hkv, hd)
        v_buf[slot, start_col : start_col + S] = v


__all__ = ["GLMEagle3DraftModel"]
