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

import os
from typing import List, NamedTuple, Optional

import torch
import torch.nn.functional as F
from minisgl.layers import RMSNorm, get_rope, silu_and_mul
from minisgl.layers.attention import QKNormRope
from minisgl.layers.base import BaseOP


# THE drafter linear now lives in ONE place — models/draft_linear.py — shared by DFlash, the CCA
# drafter (which imports this name) and GLM-EAGLE3 (which used to carry its own diverged copy).
# The class body that used to sit here moved there verbatim, plus a TP `shard` policy; the weight
# FORMAT (bf16 / fp8 / int8 / nvfp4-e2m1) stays a load policy on that one core, never a subclass,
# per KERNEL_CORE_POLICY.md. Aliased so the 12 call sites below are untouched.
from .draft_linear import (  # noqa: E402
    SHARD_COL, SHARD_NONE, SHARD_ROW, DraftLinear as _PlainLinear)

# ---- drafter attention: the native HIP paged flash-prefill kernel, never torch -----------------
# Every DFlash/DSpark/CCA drafter attention call has the same shape: N short query blocks (Q rows
# each), each attending over ITS OWN key sequence [prefix | block] with an optional additive mask.
# That is exactly attn_prefill_paged.flash_prefill_paged's varlen contract with one PAGE PER
# SEQUENCE: k/v_cache [pages, page_len, Hkv, hd], block_table [N, 1] naming each sequence's page,
# cu_seqlens_q = arange(N+1)*Q, context_lens = page_len. It runs causal=0 and lets `mask_bias`
# ([N*Q, page_len] fp32, 0 / -inf) carry all block structure (causal+SWA, liveness of ring columns),
# so there is ONE code path whatever the layer's (causal, window) is. GQA is native (no
# repeat_interleave), scores never materialise, and it is capture-safe: every argument is a device
# tensor or a serve-lifetime constant, and `split_ctx` is the page length — the SAME number on the
# eager and captured paths, which is what the op requires of it.
#
# The dense attn_hip.flash_prefill does NOT fit: it assumes q_len == k_len (one square [seq, seq]
# bias), so a 16-row block over an 8k prefix would have to pad the query to 8k rows.
_HIP_ATTN_HEAD_DIMS = (64, 128, 256, 512)   # the kernel's own TORCH_CHECK set (attention/rdna4.py)
_PAGED_PREFILL = None
# `split_ctx` floor for a drafter call. The op splits the key axis only when split_ctx exceeds its
# MINISGL_ATTN_SPLIT_MIN_CTX gate (1024) — a gate tuned for ENGINE prefill, where a short context
# usually arrives with a large query count that already fills the GPU. A drafter block never does:
# it is <= block*group rows per KV head, a handful of CTAs. Measured on gfx1201 (RX 9070 XT, 16 q /
# 4 kv heads, hd128): a 528-key page runs 132.5 us single-pass vs 35.0 us split; a 519-key block-7
# page 131.3 vs 36.9 us. So the drafter always states a bound of at least this. It is a pure
# function of the page length, so eager and captured calls still pass the same number, and every
# ring page on the served pairs (2062 / 4240 / 8224 keys) is above it already — the captured path
# is unaffected; this is the eager window/fallback path.
_SPLIT_CTX_FLOOR = 2048


def drafter_attn_op(head_dim: int):
    """Resolve the HIP op AT LOAD, and refuse loudly if this geometry cannot run on it. There is no
    torch fallback: a drafter the kernel cannot serve is a load error, not a silent slow path."""
    global _PAGED_PREFILL
    if head_dim not in _HIP_ATTN_HEAD_DIMS:
        raise ValueError(
            f"drafter head_dim={head_dim} is not covered by attn_prefill_paged "
            f"(supports {_HIP_ATTN_HEAD_DIMS}); the drafter attention has no torch fallback")
    if _PAGED_PREFILL is None:
        try:
            import attn_prefill_paged
        except ImportError as e:  # pragma: no cover - image/PYTHONPATH defect
            raise ImportError(
                "the DFlash drafter attention runs on the attn_prefill_paged HIP kernel, which is "
                "not importable — PYTHONPATH must include /opt/kernels (append it AFTER "
                "/engine/python; see the tail_hip namespace-package note)") from e
        _PAGED_PREFILL = attn_prefill_paged.flash_prefill_paged
    return _PAGED_PREFILL


class DrafterAttnMeta(NamedTuple):
    """Page metadata for one drafter attention forward — shared by every layer that reads pages of
    the same length. `group` is the GQA fan-out folded into the query axis (see `drafter_attend`)."""
    block_table: torch.Tensor   # [n, 1] int32: sequence i reads page block_table[i]
    cu_seqlens_q: torch.Tensor  # [n+1] int32: arange * (q * group)
    context_lens: torch.Tensor  # [n] int32: page_len
    q: int                      # query rows per sequence BEFORE the fold
    group: int                  # num_heads // num_kv_heads


def drafter_attn_meta(n: int, q: int, page_len: int, device, group: int,
                      pages: Optional[torch.Tensor] = None) -> DrafterAttnMeta:
    """Metadata for n sequences of q query rows, each over ONE page of `page_len` keys. `pages`
    names each sequence's page in the cache (default: page i for sequence i). Device-side
    construction only — no H2D, no host sync — so it is legal inside a captured body. Build it ONCE
    per forward and share it across layers."""
    i32 = torch.int32
    rows = q * group
    bt = (torch.arange(n, device=device, dtype=i32) if pages is None
          else pages.to(i32)).view(n, 1)
    cu = torch.arange(0, (n + 1) * rows, rows, device=device, dtype=i32)
    cl = torch.full((n,), page_len, device=device, dtype=i32)
    return DrafterAttnMeta(bt, cu, cl, q, group)


def drafter_fold_mask(mask: Optional[torch.Tensor], group: int) -> Optional[torch.Tensor]:
    """[n, q, page_len] (or [n*q, page_len]) additive mask -> the folded [n*q*group, page_len] fp32
    rows the kernel indexes. Every head of a GQA group shares its query's mask row. Do this ONCE per
    distinct mask per forward (layers share masks), not per layer."""
    if mask is None:
        return None
    L = mask.shape[-1]
    m = mask.reshape(-1, 1, L).to(torch.float32)
    return m.expand(m.shape[0], group, L).reshape(-1, L).contiguous()


def drafter_attend(q, k_cache, v_cache, meta: DrafterAttnMeta, scale: float,
                   mask: Optional[torch.Tensor], split_ctx: Optional[int] = None) -> torch.Tensor:
    """q [n*meta.q, H, hd] -> [n*meta.q, H, hd].

    k/v_cache [pages, page_len, Hkv, hd] (may be a view whose PAGES stride, e.g. a ring pool's
    [slots, cap+Q, Hkv, hd]; rows within a page must be packed). mask: None (bidirectional over the
    whole page) or the FOLDED additive fp32 [n*meta.q*group, page_len] from `drafter_fold_mask`.
    split_ctx: the op's split-K bound; default max(page_len, _SPLIT_CTX_FLOOR) (see there).

    GQA IS FOLDED INTO THE QUERY AXIS. A drafter block is ~16 query rows against thousands of keys:
    handed to the kernel as H heads x 16 rows, every CTA fills half of a 32-row tile and each KV head's
    slab is re-streamed for every q-head pair. Handed over as Hkv heads x (16 * group) rows — the
    `group` q-heads that share a KV head become extra query ROWS of that head — the tiles fill and K/V
    is staged once per KV head. Same math (each row still sees exactly its own query vector, its own
    mask row and its KV head's keys). Measured on gfx1201 (RX 9070, 16 q / 4 kv heads, hd128, block
    16): 4240 keys 323.8 -> 176.0 us, 8224 keys 720.7 -> 315.4 us, and bit-identical to the unfolded
    call (max|delta| 0.0) at block 16."""
    from minisgl._hip_engage import engaged
    engaged("attn_prefill_paged.flash_prefill_paged[drafter]")
    page_len = k_cache.shape[1]
    Hkv, hd = k_cache.shape[2], k_cache.shape[3]
    for c in (k_cache, v_cache):   # rows inside a page must be packed; only the PAGE may stride
        assert c.stride(3) == 1 and c.stride(2) == hd and c.stride(1) == Hkv * hd, (
            f"drafter KV page must have packed rows, got strides {tuple(c.stride())}")
    assert v_cache.stride(0) == k_cache.stride(0), "K and V pages must share one page stride"
    T, H = q.shape[0], q.shape[1]
    G = meta.group
    assert H == Hkv * G, f"q heads {H} != kv heads {Hkv} x group {G}"
    n = T // meta.q
    rows = meta.q * G
    if mask is not None:
        assert mask.shape == (n * rows, page_len) and mask.dtype == torch.float32 \
            and mask.is_contiguous(), (
                f"mask must be drafter_fold_mask()'d: [{n * rows}, {page_len}] fp32, got "
                f"{tuple(mask.shape)} {mask.dtype}")
    qf = q.view(n, meta.q, Hkv, G, hd).transpose(2, 3).reshape(n * rows, Hkv, hd)
    out = _PAGED_PREFILL(
        qf, k_cache, v_cache, meta.block_table, meta.cu_seqlens_q, meta.context_lens, float(scale),
        0, 0, int(rows),
        int(split_ctx if split_ctx is not None else max(page_len, _SPLIT_CTX_FLOOR)),
        int(k_cache.stride(0)), mask)
    return out.view(n, meta.q, G, Hkv, hd).transpose(2, 3).reshape(T, H, hd)


# DSpark Markov walk: how many unbiased top candidates each block position re-scores. The bias is a
# low-rank ADDITIVE term, so the biased argmax can only move within tokens whose unbiased logit is
# within max|bias| of the top — re-scoring the full vocab streams the whole [vocab, rank] w2 matrix
# once per position, sequentially, for a result the candidate set already contains. A fixed C keeps
# the shapes static (capture-safe). MINISGL_DSPARK_MARKOV_FULL=1 restores the exact full-vocab walk
# for parity checks.
_MARKOV_TOPC = int(os.environ.get("MINISGL_DSPARK_MARKOV_TOPC") or 64)
_MARKOV_FULL = os.environ.get("MINISGL_DSPARK_MARKOV_FULL") == "1"


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
        from minisgl.distributed import get_tp_info

        tp = get_tp_info()
        # ATTENTION sharding needs BOTH head counts to divide the TP size: the GQA group mapping
        # (q head h reads kv head h // group) is only preserved when both split by the same factor.
        # If either does not divide, attention stays REPLICATED and only the MLP shards — which is
        # where most of the mass is anyway, so the fallback is degraded, not useless.
        self._attn_sharded = (
            tp.size > 1 and num_heads % tp.size == 0 and num_kv_heads % tp.size == 0)
        _asplit = tp.size if self._attn_sharded else 1

        self.hidden_size = hidden_size
        # LOCAL head counts — every reshape and the attention call below are expressed in these, so the math
        # is per-rank by construction and needs no other edit.
        self.num_heads = num_heads // _asplit
        self.num_kv_heads = num_kv_heads // _asplit
        self.head_dim = head_dim
        self.q_dim = self.num_heads * head_dim
        self.kv_dim = self.num_kv_heads * head_dim
        # FULL dims are kept for the LOADERS: a checkpoint tensor always arrives whole and
        # DraftLinear.load()/load_quant() slice it to this rank.
        self.full_q_dim = num_heads * head_dim
        self.full_kv_dim = num_kv_heads * head_dim
        # Plain 1/sqrt(d), and it stays plain even when the TARGET's family folds an extra factor
        # into its softmax scale (Muse-Glimmer: qk_scale_factor 3.87 over a WEIGHTLESS QK-norm). A
        # DFlash drafter is not that architecture: it carries its OWN learned q_norm/k_norm, so
        # training sets the temperature through those weights. Both upstream engines agree —
        # vllm/model_executor/models/qwen3_dflash.py:190 `self.scaling = self.head_dim**-0.5` with
        # learned q_norm/k_norm at :235-236, and sglang/srt/models/dflash.py:158 identically — and
        # neither reads any family scale factor for a drafter. Do not "inherit" the target's scale.
        self.scale = head_dim ** -0.5
        # The attention runs on the HIP paged flash kernel; resolve it (and refuse an uncovered
        # head_dim) HERE, at load, rather than on the first propose.
        drafter_attn_op(head_dim)
        self._rotary = rotary
        _acol = SHARD_COL if self._attn_sharded else SHARD_NONE
        _arow = SHARD_ROW if self._attn_sharded else SHARD_NONE
        # Laguna gated attention: a per-head SOFTPLUS output gate (self_attn.g_proj [num_heads, hidden])
        # applied to the attention output before o_proj — matches the base LagunaAttention. Qwen3 z-lab
        # drafters have no gate (gated=False) and this path is byte-identical to before.
        self.gated = gated
        self.sliding_window = sliding_window
        self.g_proj = _PlainLinear(hidden_size, num_heads, shard=_acol) if gated else None

        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.q_proj = _PlainLinear(hidden_size, self.full_q_dim, shard=_acol)
        self.k_proj = _PlainLinear(hidden_size, self.full_kv_dim, shard=_acol)
        self.v_proj = _PlainLinear(hidden_size, self.full_kv_dim, shard=_acol)
        self.o_proj = _PlainLinear(self.full_q_dim, hidden_size, shard=_arow)
        # Per-head RMSNorm over head_dim (plain-weight; both the Qwen3 z-lab draft and the Laguna
        # draft use the plain-weight convention, NOT the (1+weight) Qwen3.5 one).
        self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps)
        # The shared attention front end (q/k norm + RoPE, one launch where it applies) over the
        # norms above. `_`-prefixed: it holds references, and must stay out of the state dict.
        self._qk = QKNormRope(self.num_heads, self.num_kv_heads, head_dim, self.q_norm, self.k_norm,
                              rotary)
        self.gate_proj = _PlainLinear(hidden_size, intermediate_size, shard=SHARD_COL)
        self.up_proj = _PlainLinear(hidden_size, intermediate_size, shard=SHARD_COL)
        self.down_proj = _PlainLinear(intermediate_size, hidden_size, shard=SHARD_ROW)

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
        attn_mask: Optional[torch.Tensor] = None,  # additive 0/-inf; None => bidirectional. [B, P+B]
                                  # when meta is None, else already drafter_fold_mask()'d
        meta=None,                # drafter_attn_meta(1, B, P+B, group) — shared across layers
    ) -> torch.Tensor:
        """The block half of the layer forward: project the noise queries/KV, then attend over
        [prefix K/V | noise K/V]. `attn_mask` None => bidirectional (z-lab Qwen); a [B, P+B] additive
        causal+sliding-window mask => Laguna (causal=true, window=512). `k_ctx`/`v_ctx` is the (possibly
        persistent) target-context prefix from `project_ctx`.

        Attention is the HIP paged flash kernel over ONE page holding [prefix | block] (see
        `drafter_attend`). It is not bit-identical to the torch einsum/softmax it replaced — a
        different reduction order — and that is not a correctness property of a DRAFTER: a drafted
        token only changes acceptance, and the target verifies every one. The contract is numerical
        closeness to an fp32 reference (tools/dflash_window_parity.py)."""
        B = hidden.shape[0]
        H, Hkv, hd = self.num_heads, self.num_kv_heads, self.head_dim

        residual = hidden
        x = self.input_layernorm.forward(hidden)

        q = self.q_proj.forward(x).view(B, H, hd)
        k_noise = self.k_proj.forward(x).view(B, Hkv, hd)
        v_noise = self.v_proj.forward(x).view(B, Hkv, hd)

        # Per-head q_norm/k_norm over head_dim, then rotary on the noise block (prefix already rotated).
        q_flat, kn_flat, _ = self._qk.forward(q, k_noise, None, block_pos)
        q = q_flat.view(B, H, hd)
        k_noise = kn_flat.view(B, Hkv, hd)

        # K/V = [ctx prefix | noise] along the key sequence, as ONE page of S = P + B rows.
        K = torch.cat([k_ctx, k_noise], dim=0).unsqueeze(0)  # [1, S, Hkv, hd]
        V = torch.cat([v_ctx, v_noise], dim=0).unsqueeze(0)
        if meta is None:   # standalone call; the model forwards build these once for all layers
            meta = drafter_attn_meta(1, B, K.shape[1], hidden.device, H // Hkv)
            attn_mask = drafter_fold_mask(attn_mask, H // Hkv)
        attn = drafter_attend(q, K, V, meta, self.scale, attn_mask)  # [B, H, hd]
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
        k_pool: torch.Tensor,     # [slots, C+Q, Hkv, hd]  this layer's ring: C prefix cols + Q scratch
        v_pool: torch.Tensor,
        slots: torch.Tensor,      # [N] int64 ring slot each request reads (NULL slot for padding)
        attn_mask: torch.Tensor,  # [N*Q*group, C+Q] additive 0/-inf, drafter_fold_mask()'d
        meta,                     # drafter_attn_meta(N, Q, C+Q, group, pages=slots), shared
    ) -> torch.Tensor:
        """BATCHED, CUDA-graph-capturable twin of `attend_block`.

        Differences from the per-request form, each forced by capture (see spec/capture.py):
          * N requests in ONE forward — the per-request Python loop is host control flow, which a
            graph cannot contain, and it was also serialising every request's ~150 launches.
          * the prefix is a FIXED-capacity ring row with an additive mask, not a variable [P,...]
            slice: a data-dependent key length cannot be captured.
          * ZERO-COPY keys. Each ring row carries Q SCRATCH columns after its C prefix columns; the
            block's own K/V are written there, and the kernel reads [prefix | block] straight out of
            the pool with the request's slot as its page. The old path gathered `k_pool[slots]` and
            `torch.cat`ed the block onto it — two full [N, C+Q, Hkv, hd] copies per tensor per layer,
            all of it charged to the graph's private pool. Padding rows share the NULL slot and so
            share its scratch; their outputs are discarded, and every value there is finite.

        Deterministic: eager == replayed bit for bit for the real rows (the kernel's split decision
        keys on `split_ctx` = C+Q, a serve constant, never on anything that differs under capture)."""
        N, Q = hidden.shape[0], hidden.shape[1]
        H, Hkv, hd = self.num_heads, self.num_kv_heads, self.head_dim
        T = N * Q
        C = k_pool.shape[1] - Q

        flat = hidden.reshape(T, -1)
        residual = flat
        x = self.input_layernorm.forward(flat)

        q = self.q_proj.forward(x).view(T, H, hd)
        k_noise = self.k_proj.forward(x).view(T, Hkv, hd)
        v_noise = self.v_proj.forward(x).view(T, Hkv, hd)
        q_flat, kn_flat, _ = self._qk.forward(q, k_noise, None, block_pos.reshape(T))
        # The block's K/V into each request's scratch columns [C, C+Q) of its own ring row.
        k_pool[slots, C:] = kn_flat.view(N, Q, Hkv, hd)
        v_pool[slots, C:] = v_noise.view(N, Q, Hkv, hd)
        attn = drafter_attend(q_flat.view(T, H, hd), k_pool, v_pool, meta, self.scale,
                              attn_mask)  # [T, H, hd]
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
        meta=None,                    # drafter_attn_meta(1, B, P+B), shared across layers
    ) -> torch.Tensor:
        # Recompute-every-step path (no persistent KV): project the whole prefix, then attend. Kept
        # byte-identical for the legacy/window fallback; the fast path caches project_ctx across steps.
        k_ctx, v_ctx = self.project_ctx(target_hidden, ctx_pos)
        return self.attend_block(hidden, block_pos, k_ctx, v_ctx, attn_mask, meta)


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
        rope_scaling: "tuple | None" = None,
        draft_vocab_size: Optional[int] = None,
        own_embed: bool = False,
        decoder_layer_type: str = "qwen3",
        sliding_window: int = 0,
        causal: bool = False,
        per_aux_norm: bool = False,
        layer_causal: Optional[List[bool]] = None,
        layer_window: Optional[List[int]] = None,
    ) -> None:
        self.hidden_size = hidden_size
        self.num_aux_layers = num_aux_layers
        self.num_layers = num_layers
        # Laguna (decoder_layer_type='laguna_xs'): gated attention, causal block mask + sliding window,
        # and a per-captured-layer RMSNorm on each aux BEFORE fc (aux_hidden_norms). z-lab Qwen3
        # (default): ungated, bidirectional, single hidden_norm after fc.
        gated = decoder_layer_type == "laguna_xs"
        # THE MASK IS PER LAYER, and this is not a minisgl invention — it is the convention both
        # merged upstream engines implement, independently and identically:
        #
        #   vllm/model_executor/models/qwen3_dflash.py::_dflash_layer_causal
        #       "``dflash_config.causal`` overrides all layers; else only SWA layers causal."
        #   sglang/srt/models/dflash.py
        #       full_attention  -> AttentionType.ENCODER_ONLY (non-causal), window -1
        #       sliding_attention -> AttentionType.DECODER (causal), window sliding_window-1
        #
        # i.e. a `sliding_attention` layer is CAUSAL with a left-only window, and a `full_attention`
        # layer is BIDIRECTIONAL and unbounded. A single global (causal, sliding_window) pair cannot
        # express a checkpoint that mixes the two — which every z-lab DFlash drafter except the
        # all-full one does. `causal`/`sliding_window` remain the UNIFORM fallback for a drafter that
        # declares no `layer_types`; when the caller derives the per-layer lists they win.
        self.layer_causal = list(layer_causal) if layer_causal else [causal] * num_layers
        self.layer_window = list(layer_window) if layer_window else [sliding_window] * num_layers
        assert len(self.layer_causal) == len(self.layer_window) == num_layers, (
            f"per-layer mask lists must cover all {num_layers} layers, got "
            f"{len(self.layer_causal)}/{len(self.layer_window)}")
        # Uniform view, kept for the callers that legitimately need ONE answer for the whole model:
        # `window_prefix` (how many prefix rows are worth projecting at all) and the proposer's
        # decision on whether a fixed-capacity ring can hold this drafter's prefix. A drafter is
        # BOUNDED only if EVERY layer is windowed — one full_attention layer reads the whole prefix
        # and makes the model unbounded no matter what the other layers do.
        self.bounded = all(w > 0 for w in self.layer_window)
        # Every layer masks identically. The CAPTURED propose body (`denoise_batched`) records ONE
        # mask tensor for the whole trunk, so a drafter whose layers disagree cannot ride it; the
        # proposer gates capture on this rather than silently applying layer 0's mask to all of them.
        # Uniform is the common case: it holds for every all-`sliding_attention` and every
        # all-`full_attention` checkpoint, and fails only for the mixed z-lab drafters.
        self.uniform_mask = (
            len(set(zip(self.layer_causal, self.layer_window))) <= 1)
        self.causal = causal
        self.sliding_window = max(self.layer_window) if self.bounded else 0

        # fc fuses the N captured target aux layers (N*hidden) -> hidden; hidden_norm norms it.
        self.fc = _PlainLinear(num_aux_layers * hidden_size, hidden_size)
        self.hidden_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        # Per-aux-layer RMSNorm applied to each captured target hidden before concat+fc (Laguna).
        self.aux_hidden_norms = (
            [RMSNorm(hidden_size, eps=rms_norm_eps) for _ in range(num_aux_layers)]
            if per_aux_norm
            else None
        )

        # rope_scaling was hardcoded None, so a drafter declaring a real scheme silently ran PLAIN
        # rope. That is not a long-context-only concern: YaRN's frequency ramp reshapes inv_freq at
        # EVERY position, so a YaRN-trained drafter fed default rope mis-encodes position from token
        # one and drafts badly with nothing in the logs to say so. Measured on Qwen3.8-27B-DSpark
        # (rope_type=yarn, factor=32, original_max_position_embeddings=8192).
        rotary = get_rope(
            head_dim=head_dim,
            rotary_dim=head_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=rope_scaling,
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
                sliding_window=self.layer_window[i],
            )
            for i in range(num_layers)
        ]
        self.norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        # GQA fan-out the attention kernel folds into its query axis (per rank; identical per layer).
        self._group = self.layers[0].num_heads // self.layers[0].num_kv_heads

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

        # DSpark: optional low-rank bigram ("Markov") logit-bias head. None for plain DFlash; the
        # proposer installs it at load when the checkpoint ships `markov_head.*` — detected from
        # TENSORS, like every other variant decision on this path.
        self._markov_w1 = None  # [vocab, rank]  latent for the PREVIOUS token
        self._markov_w2 = None  # [vocab, rank]  Linear(rank -> vocab).weight, projects back
        # DSpark: optional confidence head (AcceptRatePredictor) — a single Linear over
        # [block hidden | markov latent of the previous token] predicting the per-position
        # acceptance probability. Consumed by the proposer to TRUNCATE the drafted block where
        # the predicted cumulative survival drops below threshold (adaptive draft length).
        self._conf_w = None  # [1, hidden (+ markov_rank when trained with_markov)]  fp32
        self._conf_b = None  # [1] fp32

    # ---- binding to the target (tied-vocab variant) ----
    def bind_embed(self, embed) -> None:
        self._embed = embed

    def bind_lm_head(self, lm_head) -> None:
        self._lm_head = lm_head

    def set_own_embed(self, embed_weight: torch.Tensor) -> None:
        self._own_embed = _PlainLinear(0, 0)
        self._own_embed.weight = embed_weight  # [target_vocab, hidden]

    # ---- DSpark Markov (bigram) head ----
    def set_markov(self, w1: torch.Tensor, w2: torch.Tensor) -> None:
        self._markov_w1, self._markov_w2 = w1, w2

    @property
    def has_markov(self) -> bool:
        return self._markov_w1 is not None

    def markov_bias(self, prev_ids: torch.Tensor) -> torch.Tensor:
        """Logit bias for the token FOLLOWING `prev_ids`: w2 @ w1[prev] -> [*, vocab].

        Rank-256 factorization, so it is two skinny matmuls rather than a [vocab, vocab] bigram
        table (which at this vocab would be 61 G params)."""
        return F.linear(F.embedding(prev_ids, self._markov_w1), self._markov_w2)

    # ---- DSpark confidence head (AcceptRatePredictor) ----
    def set_confidence(self, weight: torch.Tensor, bias: Optional[torch.Tensor]) -> None:
        self._conf_w, self._conf_b = weight, bias

    @property
    def has_confidence(self) -> bool:
        return self._conf_w is not None

    def confidence(self, rows: torch.Tensor, prev_ids: torch.Tensor) -> torch.Tensor:
        """Per-position predicted acceptance probability, sigmoid(Linear(features)) in fp32.

        `rows` [..., k, hidden] are the SAME post-norm block hidden states the head scores (the
        reference feeds the logits' input to the confidence head — sglang dspark_planner.py
        `compute_confidence(draft_hidden=...)` where draft_hidden is the sampler's hidden_states).
        `prev_ids` [..., k] is the teacher-forcing sequence [anchor, draft_0, .., draft_{k-2}] —
        the same one-step-back conditioning the Markov walk uses. When the head was trained
        with_markov (our checkpoint: in_dim = hidden + rank), the feature is the concat of the row
        with markov_w1[prev]; a head trained without it is just the row. fp32 throughout: the
        weight is [1, in_dim], so this costs one skinny GEMV per block."""
        from minisgl._hip_engage import engaged
        engaged("spec_dflash.dspark_confidence")
        feat = rows.float()
        if self._conf_w.shape[1] > rows.shape[-1]:
            assert self._markov_w1 is not None, "with_markov confidence head needs the Markov head"
            feat = torch.cat([feat, F.embedding(prev_ids, self._markov_w1).float()], dim=-1)
        raw = F.linear(feat, self._conf_w, self._conf_b).squeeze(-1)
        return torch.sigmoid(raw)

    def markov_block_argmax(
        self, block_logits: torch.Tensor, prev: torch.Tensor
    ) -> torch.Tensor:
        """Semi-autoregressive block decode: bias each position by the token chosen at the previous
        one, then argmax. `block_logits` is [..., k, vocab]; `prev` is the anchor id, shaped [...].

        WHY: DFlash scores every block position in ONE forward, so given the block hidden states its
        positions are conditionally INDEPENDENT — which is precisely why its acceptance decays toward
        the back of a block ("suffix decay"). Restoring one-step dependency is the whole DSpark idea.
        The loop is a FIXED trip count over k with static shapes and `prev` never leaves the device,
        so it adds no host sync and stays capture-safe. The candidate set per position is the top
        `_MARKOV_TOPC` UNBIASED tokens, computed in one batched topk; the walk then re-scores only
        those — a [C, rank] gather of w2 per position instead of the full [vocab, rank] stream and
        a full-vocab argmax, which is almost entirely wasted bandwidth for a low-rank additive bias
        (see `_MARKOV_TOPC`)."""
        from minisgl._hip_engage import engaged
        engaged("spec_dflash.dspark_markov")
        k = block_logits.shape[-2]
        if _MARKOV_FULL:
            ids = []
            for j in range(k):
                tok = (block_logits[..., j, :] + self.markov_bias(prev)).argmax(dim=-1)
                ids.append(tok)
                prev = tok
            return torch.stack(ids, dim=-1) if ids else prev.new_empty(prev.shape + (0,))
        C = min(_MARKOV_TOPC, block_logits.shape[-1])
        top_v, top_i = block_logits.topk(C, dim=-1)      # [..., k, C], one pass over the logits
        cand_w2 = F.embedding(top_i, self._markov_w2)    # [..., k, C, rank]
        ids = []
        for j in range(k):
            lat = F.embedding(prev, self._markov_w1)     # [..., rank]
            bias = (cand_w2[..., j, :, :] * lat.unsqueeze(-2)).sum(dim=-1)  # [..., C]
            sel = (top_v[..., j, :] + bias).argmax(dim=-1)
            tok = top_i[..., j, :].gather(-1, sel.unsqueeze(-1)).squeeze(-1)
            ids.append(tok)
            prev = tok
        return torch.stack(ids, dim=-1) if ids else prev.new_empty(prev.shape + (0,))

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

    def _block_mask(
        self, P: int, B: int, device: torch.device, causal: bool, window: int
    ) -> Optional[torch.Tensor]:
        """Additive [B, P+B] mask (0 keep / -inf drop) for ONE layer's block attention.

        The keys are the P prefix positions followed by the B block positions, all contiguous in
        absolute position (prefix at [base-P .. base-1], block at [base .. base+B-1]); so a query at
        block index i sits at concatenated position i+P and, causally + FlashAttention moving-query
        SWA, attends to keys j with (i+P-window) < j <= (i+P).

        `(qpos - kpos) < window` is exactly SGLang's `window_left = sliding_window - 1` convention
        (`speculative/dflash_utils.py`: "HF sliding windows include the current token; SGLang stores
        window_left"), so a declared 4096 keeps distances 0..4095. Returns None only for a layer that
        is BOTH non-causal and unwindowed — a `full_attention` layer — which attends the whole prefix
        bidirectionally and needs no mask at all."""
        if not causal and window <= 0:
            return None
        qpos = torch.arange(B, device=device).view(B, 1) + P  # [B,1] concat position of each query
        kpos = torch.arange(P + B, device=device).view(1, P + B)  # [1,P+B]
        keep = kpos <= qpos if causal else torch.ones_like(kpos, dtype=torch.bool).expand(B, P + B)
        if window > 0:
            keep = keep & ((qpos - kpos) < window)
        return torch.where(
            keep, torch.zeros((), device=device), torch.full((), float("-inf"), device=device)
        ).float()

    def _fold_masks(self, masks: List[Optional[torch.Tensor]]) -> List[Optional[torch.Tensor]]:
        """drafter_fold_mask each DISTINCT mask once (layers share mask tensors), preserving the
        per-layer sharing, so a uniform drafter pays one fold per forward, not one per layer."""
        done: dict = {}
        out = []
        for m in masks:
            k = id(m)
            if k not in done:
                done[k] = drafter_fold_mask(m, self._group)
            out.append(done[k])
        return out

    def layer_masks(self, P: int, B: int, device: torch.device) -> List[Optional[torch.Tensor]]:
        """One additive mask per layer, built once per distinct (causal, window) pair.

        A uniform drafter (every layer `sliding_attention`, or every layer `full_attention` — which
        is every DFlash checkpoint except the mixed z-lab ones) therefore still materialises exactly
        ONE mask and hands the same tensor to every layer, so this costs nothing where the old single
        global mask was already correct."""
        cache: dict = {}
        out: List[Optional[torch.Tensor]] = []
        for c, w in zip(self.layer_causal, self.layer_window):
            key = (bool(c), int(w))
            if key not in cache:
                cache[key] = self._block_mask(P, B, device, bool(c), int(w))
            out.append(cache[key])
        return out

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
        rows would suffice; W is kept as the conservative, easier-to-reason-about bound.)

        MIXED drafters may not slice. The bound is a property of the WHOLE model, not of the widest
        layer: a single `full_attention` layer reads every prefix row, so dropping rows that the
        windowed layers discard would change ITS output rather than skip dead work. Hence `bounded`
        (all layers windowed), not `max(layer_window)`."""
        if not self.bounded:
            return P  # an unwindowed layer is present: every key is live somewhere. Never slice.
        return min(P, max(self.layer_window))

    @torch.inference_mode()
    def denoise(
        self,
        noise_embed: torch.Tensor,    # [B, hidden]  embed([anchor, mask, ...])
        target_hidden: torch.Tensor,  # [P, hidden]  fuse_aux output
        block_pos: torch.Tensor,      # [B]
        ctx_pos: torch.Tensor,        # [P]
    ) -> torch.Tensor:
        """One denoising forward -> [B, hidden] (pre-head-normed block hidden). Each layer gets its
        OWN mask: causal+SWA for a `sliding_attention` layer, none (bidirectional, unbounded) for a
        `full_attention` one — see `layer_masks`."""
        hidden = noise_embed
        P_full = target_hidden.shape[0]
        P = self.window_prefix(P_full)
        if P < P_full:  # windowed: drop the prefix rows the mask discards BEFORE projecting them
            target_hidden = target_hidden[P_full - P :]
            ctx_pos = ctx_pos[P_full - P :]
        B = noise_embed.shape[0]
        masks = self._fold_masks(self.layer_masks(P, B, noise_embed.device))
        meta = drafter_attn_meta(1, B, P + B, noise_embed.device, self._group)  # [prefix | block]
        for layer, mask in zip(self.layers, masks):
            hidden = layer.forward(hidden, target_hidden, block_pos, ctx_pos, mask, meta)
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
            # torch.cat + attention see [W + B] keys instead of [P + B]. Constant-shaped once P >= W.
            d = P_full - P
            prefix_kv = [(k[d:], v[d:]) for (k, v) in prefix_kv]
        B = noise_embed.shape[0]
        masks = self._fold_masks(self.layer_masks(P, B, noise_embed.device))
        meta = drafter_attn_meta(1, B, P + B, noise_embed.device, self._group)
        for layer, (k_ctx, v_ctx), mask in zip(self.layers, prefix_kv, masks):
            hidden = layer.attend_block(hidden, block_pos, k_ctx, v_ctx, mask, meta)
        return self.norm.forward(hidden)

    # ---- CUDA-graph-capturable batched propose (see spec/capture.py, spec/dflash.py) -------------
    def project_prefix_into(
        self,
        aux: torch.Tensor,          # [m, num_aux, hidden]  captured target aux, m rows
        rope_pos: torch.Tensor,     # [m] int32 absolute RoPE position of each row
        k_pool: List[torch.Tensor],  # per-layer [slots, C+Q, Hkv, hd] ring (cols >= C are scratch)
        v_pool: List[torch.Tensor],
        wslot: torch.Tensor,        # [m] destination slot per row (NULL slot = discard)
        wcol,                       # [m] ring column per row, or a PER-LAYER list of them
        layer_ids=None,             # restrict to these layer indices (rings of one capacity)
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
        # `wcol` may be per-layer: a mixed drafter's rings have DIFFERENT capacities (a windowed
        # layer needs window+slack rows, a `full_attention` layer needs a max-context-capped ring),
        # so `pos % C` differs per layer. One shared column vector only works when every layer's ring
        # is the same size. `fuse_aux` still runs ONCE for all layers either way.
        # `layer_ids` exists for the REBUILD path, which must feed a different number of rows to
        # rings of different capacity: a ring of capacity c can only hold its newest c rows, and
        # scattering more than that writes several positions to the same column. That is not merely
        # wasteful — `_pk`/`_pv` and `_ppos` are SEPARATE scatters, so with duplicate indices they
        # can disagree about which position a column holds, and the mask would then admit a key
        # believing it is at a position it is not (wrong RoPE phase, wrong content).
        per_layer_col = isinstance(wcol, (list, tuple))
        target_hidden = self.fuse_aux(aux)  # [m, hidden]
        for l in (range(len(self.layers)) if layer_ids is None else layer_ids):
            layer = self.layers[l]
            k, v = layer.project_ctx(target_hidden, rope_pos)
            c = wcol[l] if per_layer_col else wcol
            k_pool[l][wslot, c] = k
            v_pool[l][wslot, c] = v

    def denoise_batched(
        self,
        noise_embed: torch.Tensor,   # [N, Q, hidden]
        block_pos: torch.Tensor,     # [N, Q] int32
        k_pool: List[torch.Tensor],  # per-layer [slots, C+Q, Hkv, hd]: C ring cols + Q block scratch
        v_pool: List[torch.Tensor],
        slots: torch.Tensor,         # [N] which ring slot each request reads
        mask,                        # [N, Q, C+Q] additive 0/-inf, or a PER-LAYER list of them
    ) -> torch.Tensor:
        """One BATCHED denoising forward over the persistent ring -> [N, Q, hidden].

        No per-layer gather: the attention kernel reads each request's ring row IN PLACE, the slot
        being its page (see `attend_block_batched`). The page metadata depends only on (N, Q, page
        length, slots), so it is built once per distinct ring capacity and shared by every layer."""
        masks = self._fold_masks(
            list(mask) if isinstance(mask, (list, tuple)) else [mask] * len(self.layers))
        N, Q = noise_embed.shape[0], noise_embed.shape[1]
        metas: dict = {}
        hidden = noise_embed
        for l, layer in enumerate(self.layers):
            page_len = k_pool[l].shape[1]
            if page_len not in metas:
                metas[page_len] = drafter_attn_meta(
                    N, Q, page_len, noise_embed.device, self._group, pages=slots)
            hidden = layer.attend_block_batched(
                hidden, block_pos, k_pool[l], v_pool[l], slots, masks[l], metas[page_len])
        return self.norm.forward(hidden.reshape(-1, hidden.shape[-1])).view_as(hidden)


__all__ = ["DFlashDraftModel"]
