"""TiDAR structured attention mask + fused-forward layout — backend-neutral (C.3).

Ported verbatim from the reference `vllm-gfx1201-tidar-fused/zaya/tidar/tidar_mask.py` (the single
source of truth, pure-torch, no deps) so the FUSED single-forward math is defined ONCE and consumed
identically by every attention path. See docs/PHASE_C_FUSED_FORWARD.md for how this plugs into the
minisgl paged serve.

Produces, from one construction:
  (1) ``build_allow_matrix``  — boolean [q_len, kv_len] allow/deny (ground truth for tests).
  (2) ``additive_bias``       — float [q_len, kv_len], 0 where allowed, -inf where denied. Added to
                                QK^T BEFORE softmax. This is the tensor the attn kernel `mask_bias`
                                arg consumes (attn_hip already; attn_prefill_paged after C.1).
  (3) ``square_additive_bias``— [L,L] variant for self-attention kernels whose q and k seqs are the
                                same length (the attn_hip dense cold-prefill prototype path).
  (4) ``MaskDescriptor``      — compact layout integers for an inline-predicate kernel path.

LAYOUT (one sequence; prefix = committed KV, cached & not re-queried):
    new-token region, q_len = block_len * (1 + block_len):
      [ S : sampling block (B drafts to AR-verify) | R_0 | R_1 | ... | R_{B-1} ]
    Keys = [ prefix (cached) | S | R_0 | ... | R_{B-1} ].
  R_r = B mask-token replicas pre-drafting the NEXT block conditioned on r accepted drafts of S;
  after verify yields accepted length k, the runner SELECTS replica R_k as the next drafts.

PREDICATE (q attends k iff allow[q,k]):
  - prefix query -> causal over prefix.
  - S[i]  -> prefix (all) + S[j], j<=i (causal).                         NOT any R_r.
  - R_r[m]-> prefix (all) + S[j], j < r (the r accepted drafts) + own replica R_r (bidir).
            NOT other replicas, NOT S[j>=r].

The unresolved-but-pinned choices are FLAGS on MaskDescriptor (replica_offset, sampling_causal,
mask_sees_prefix), confirmed against the conversion checkpoint upstream.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

NEG_INF = float("-inf")

__all__ = [
    "MaskDescriptor",
    "build_allow_matrix",
    "additive_bias",
    "build_square_allow_matrix",
    "square_additive_bias",
    "select_next_drafts_row_range",
    "fused_forward_position_ids",
    "parse_fused_forward_logits",
    "fused_paged_layout",
    "fused_paged_layout_segmented",
]


@dataclass(frozen=True)
class MaskDescriptor:
    """Compact layout integers. All positions are GLOBAL key-axis indices (prefix occupies
    [0, prefix_len); the new region occupies [prefix_len, prefix_len+q_len)); a query-axis index
    q in [0, q_len) maps to global key position prefix_len + q."""

    prefix_len: int
    block_len: int
    replica_offset: int = 0
    sampling_causal: bool = True
    mask_sees_prefix: bool = True

    @property
    def q_len(self) -> int:
        return self.block_len * (1 + self.block_len)

    @property
    def kv_len(self) -> int:
        return self.prefix_len + self.q_len

    @property
    def s_start(self) -> int:
        return 0

    @property
    def s_end(self) -> int:  # exclusive
        return self.block_len

    def replica_start(self, r: int) -> int:
        return self.block_len + r * self.block_len

    def replica_drafts(self, r: int) -> int:
        """How many sampling-block drafts replica r is conditioned on."""
        return r + self.replica_offset

    def seg_of(self, new_pos: int) -> tuple[str, int, int]:
        """Classify a new-region position: ('S'|'R', replica_idx_or_-1, local_idx)."""
        if new_pos < self.block_len:
            return ("S", -1, new_pos)
        rel = new_pos - self.block_len
        r = rel // self.block_len
        return ("R", r, rel % self.block_len)


def _allow_pair(d: MaskDescriptor, q_new: int, k_global: int) -> bool:
    """Reference predicate: does query (new-region index q_new) attend key (global k)?"""
    in_prefix = k_global < d.prefix_len
    k_new = k_global - d.prefix_len
    q_seg, q_r, q_i = d.seg_of(q_new)
    if q_seg == "S":
        if in_prefix:
            return True
        k_seg, k_r, k_i = d.seg_of(k_new)
        if k_seg == "S":
            return (k_i <= q_i) if d.sampling_causal else True
        return False
    # q_seg == "R"
    if in_prefix:
        return d.mask_sees_prefix
    k_seg, k_r, k_i = d.seg_of(k_new)
    if k_seg == "S":
        return k_i < d.replica_drafts(q_r)
    return k_r == q_r


def build_allow_matrix(d: MaskDescriptor, device=None) -> torch.Tensor:
    """Dense boolean [q_len, kv_len] allow matrix (ground truth); vectorised == _allow_pair."""
    B, P = d.block_len, d.prefix_len
    ql, kl = d.q_len, d.kv_len

    new_idx = torch.arange(ql, device=device)
    is_S = new_idx < B
    rel = (new_idx - B).clamp(min=0)
    rep = torch.where(is_S, torch.full_like(new_idx, -1), rel // B)
    loc = torch.where(is_S, new_idx, rel % B)
    q_is_S = is_S.view(ql, 1)
    q_rep = rep.view(ql, 1)
    q_loc = loc.view(ql, 1)

    k_arange = torch.arange(kl, device=device)
    k_in_prefix = (k_arange < P).view(1, kl)
    k_new = (k_arange - P).clamp(min=0)
    k_is_S = ((k_arange >= P) & (k_arange < P + B)).view(1, kl)
    k_rel = (k_new - B).clamp(min=0)
    k_rep = torch.where(k_arange < P + B, torch.full_like(k_arange, -1), k_rel // B).view(1, kl)
    k_loc = torch.where(k_arange - P < B, (k_arange - P).clamp(min=0), k_rel % B).view(1, kl)

    allow = torch.zeros(ql, kl, dtype=torch.bool, device=device)
    qS = q_is_S
    allow |= qS & k_in_prefix
    if d.sampling_causal:
        allow |= qS & k_is_S & (k_loc <= q_loc)
    else:
        allow |= qS & k_is_S
    qR = ~q_is_S
    if d.mask_sees_prefix:
        allow |= qR & k_in_prefix
    drafts = q_rep + d.replica_offset
    allow |= qR & k_is_S & (k_loc < drafts)
    same_rep = (k_rep == q_rep) & (k_rep >= 0)
    allow |= qR & same_rep
    return allow


def additive_bias(d: MaskDescriptor, dtype=torch.float32, device=None) -> torch.Tensor:
    """Float [q_len, kv_len] bias: 0 where allowed, -inf where denied. Add to QK^T pre-softmax."""
    allow = build_allow_matrix(d, device=device)
    bias = torch.zeros_like(allow, dtype=dtype)
    bias.masked_fill_(~allow, NEG_INF)
    return bias


def build_square_allow_matrix(d: MaskDescriptor, device=None) -> torch.Tensor:
    """Square [L,L] allow matrix over the FULL window L = prefix_len + q_len, prefix laid out as
    query rows too (causal). Needed by self-attention kernels (attn_hip.flash_prefill) whose q and k
    seqs are the same length; only the new-region output rows [prefix_len:] are used by the runner."""
    P, ql = d.prefix_len, d.q_len
    L = P + ql
    allow = torch.zeros(L, L, dtype=torch.bool, device=device)
    if P > 0:
        idx = torch.arange(P, device=device)
        allow[:P, :P] = idx[:, None] >= idx[None, :]
    allow[P:L, :] = build_allow_matrix(d, device=device)
    return allow


def square_additive_bias(d: MaskDescriptor, dtype=torch.float32, device=None) -> torch.Tensor:
    """Square [L,L] additive bias (0 / -inf) for self-attention kernels."""
    allow = build_square_allow_matrix(d, device=device)
    bias = torch.zeros_like(allow, dtype=dtype)
    bias.masked_fill_(~allow, NEG_INF)
    return bias


def select_next_drafts_row_range(d: MaskDescriptor, accepted_k: int) -> tuple[int, int]:
    """After verify yields ``accepted_k``, the q-axis row range of the replica to read as the next
    block's drafts. Maps acceptance length -> replica index (clamped)."""
    r = max(0, min(d.block_len - 1, accepted_k - d.replica_offset))
    start = d.replica_start(r)
    return (start, start + d.block_len)


def fused_forward_position_ids(d: MaskDescriptor) -> list[int]:
    """Absolute RoPE position_ids for the NEW-region queries of the fused forward over
    [prefix | S | R_0..R_{B-1}] — length q_len. §7.6 (fp32 bit-exact upstream):
      * S local index j -> prefix_len + j (ordinary causal continuation);
      * R_r token m     -> prefix_len + replica_drafts(r) + m (the block r would occupy if exactly
        replica_drafts(r) drafts accept). The naive "all replicas at prefix_len+B" placed RoPE B-r
        positions too far for r<B -> off-distribution drafts. The runner prepends range(prefix_len)."""
    P = d.prefix_len
    pos = [0] * d.q_len
    for new_pos in range(d.q_len):
        kind, r, local = d.seg_of(new_pos)
        if kind == "S":
            pos[new_pos] = P + local
        else:
            pos[new_pos] = P + d.replica_drafts(r) + local
    return pos


def fused_paged_layout(cached_len: int, block_len: int, device=None):
    """Layout for the minisgl PAGED fused step: query ``[confirmed | S | R_0..R_{B-1}]`` over the
    cached prefix ``[0..cached_len-1]``. ``confirmed`` is the prior step's bonus token (re-queried, so
    its logit row gives p_ar[0] — mirrors the two-forward verify that already works; no carried-logit
    state). Keys during the forward = ``[cached | confirmed@cached_len | S | R*]`` (confirmed's KV is
    stored this step). Reduces to ``MaskDescriptor(prefix_len=cached_len+1)`` for the S/R predicate.

    Returns ``(positions, mask_bias, n_query, block_len)``:
      * ``positions``  — int list, length ``1 + B + B²``: confirmed@cached_len; S_j@cached_len+1+j;
        R_r[m]@cached_len+1+replica_drafts(r)+m (§7.6).
      * ``mask_bias``  — fp32 ``[n_query, context_len]`` additive (0/-inf), context_len =
        cached_len+1+B+B². Row 0 = confirmed (causal over prefix+itself); rows 1.. = S|R* predicate.
      * parse: ``p_ar = logits[0:B+1]`` (confirmed + S rows), ``replica = logits[B+1:].reshape(B,B,V)``.
    """
    B = block_len
    d = MaskDescriptor(prefix_len=cached_len + 1, block_len=B)  # [cached | confirmed] = prefix
    q_new = d.q_len            # B + B²  (S | R*)
    n_query = 1 + q_new        # + confirmed
    context_len = cached_len + 1 + q_new
    positions = [cached_len] + fused_forward_position_ids(d)
    mask = torch.full((n_query, context_len), NEG_INF, dtype=torch.float32, device=device)
    mask[0, : cached_len + 1] = 0.0                            # confirmed: causal over prefix+self
    mask[1:, :] = additive_bias(d, dtype=torch.float32, device=device)  # [q_new, context_len]
    return positions, mask, n_query, B


def fused_paged_layout_segmented(cached_len: int, block_len: int, tp: int, device=None):
    """SEGMENTED-conv fused layout: ``[confirmed | S | (ctx_0, R_0) | (ctx_1, R_1) | ...]``.

    The flat layout gives each replica ``R_r`` the WRONG conv left-context (packed neighbours = tail of
    the previous replica), crushing draft acceptance. This inserts, immediately before each ``R_r``,
    its ``tp`` correct conv-context tokens — the last ``tp`` tokens of ``[committed|confirmed|drafts[:r]]``
    (absolute positions ``c0+1+r-tp .. c0+r``) — MASKED from attention (self-only) so they ONLY feed the
    causal conv. minisgl's CCA conv reads packed neighbours (``qk_new[row-tp..row]``), so ``R_r``'s first
    mask then convs over its correct context. Mirrors ``cca.py::_decode_verify_spec`` / the reference
    ``single_forward_ours.build_segmented``.

    Returns ``dict`` with:
      positions   : list[int] RoPE positions, len = n_query = 1 + B + B*(tp+B)
      mask        : fp32 ``[n_query, context_len]`` additive (0/-inf), context_len = cached_len + n_query
                    (each query token stored at KV col cached_len+packed_row)
      n_query     : int
      p_ar_rows   : (0, B+1) slice — confirmed + S rows for beta_verify
      ctx_src     : list[(packed_row, abs_pos)] — ctx tokens; scheduler fills token value = token@abs_pos
      replica_rows: list of (r, [packed rows of R_r's B masks]) — read R_k logits from these
    """
    B, tp2 = block_len, tp
    c0 = cached_len
    # --- packed rows: (kind, r, local, abs_pos). kind in {confirmed,S,ctx,R} ---
    rows = [("confirmed", -1, 0, c0)]
    positions = [c0]
    for j in range(B):
        rows.append(("S", -1, j, c0 + 1 + j)); positions.append(c0 + 1 + j)
    ctx_src = []
    replica_rows = []
    for r in range(B):
        for t in range(tp2):
            abs_pos = max(0, c0 + 1 + r - tp2 + t)   # last tp toks of [committed|confirmed|drafts[:r]]
            pr = len(rows)
            rows.append(("ctx", r, t, abs_pos)); positions.append(abs_pos)
            ctx_src.append((pr, abs_pos))
        rr = []
        for m in range(B):
            pr = len(rows)
            rows.append(("R", r, m, c0 + 1 + r + m)); positions.append(c0 + 1 + r + m)  # §7.6
            rr.append(pr)
        replica_rows.append((r, rr))
    n_query = len(rows)
    context_len = c0 + n_query

    # --- allow matrix [n_query, context_len]; KV col of query packed row j = c0 + j ---
    # STEP-0.5 cost fix: build on CPU (per-cell writes are cheap host ops) then move to device ONCE.
    # Building on `device` did ~n_query² tiny GPU kernel launches per step (a big slice of the ~287ms
    # step); the tensor is small so H2D of one [n_query, context_len] fp32 mask is negligible.
    allow = torch.zeros(n_query, context_len, dtype=torch.bool)  # CPU
    for qi in range(n_query):
        qkind, qr, qloc, qabs = rows[qi]
        allow[qi, c0 + qi] = True                         # self
        if qkind in ("confirmed", "S", "R"):
            allow[qi, :c0] = True                         # prefix (cached committed)
        if qkind == "confirmed":
            continue                                      # prefix + self only (causal @ c0)
        if qkind == "ctx":
            # ctx tokens feed R_r's conv, so their qk must match the TRUE token@qabs qk -> attend
            # CAUSALLY over committed+confirmed+S up to qabs (KV col index == position here, so the
            # causal prefix is the contiguous cols [0, qabs)). self-only (the reference) drifts too far
            # on deep ZAYA -> garbage conv context -> 0 acceptance.
            allow[qi, :qabs] = True
            continue
        for kj in range(n_query):
            kkind, kr, kloc, _ = rows[kj]
            kcol = c0 + kj
            if qkind == "S":
                if kkind == "confirmed" or (kkind == "S" and kloc <= qloc):
                    allow[qi, kcol] = True                # confirmed + S[:i] causal
            else:  # qkind == "R"
                if kkind == "confirmed" or (kkind == "S" and kloc < qr) or (kkind == "R" and kr == qr):
                    allow[qi, kcol] = True                # confirmed + S[:r] + own replica block

    mask = torch.zeros(n_query, context_len, dtype=torch.float32)  # CPU
    mask.masked_fill_(~allow, NEG_INF)
    if device is not None:
        mask = mask.to(device, non_blocking=True)
    return {
        "positions": positions, "mask": mask, "n_query": n_query,
        "p_ar_rows": (0, B + 1), "ctx_src": ctx_src, "replica_rows": replica_rows, "rows": rows,
    }


def parse_fused_forward_logits(logits, d: MaskDescriptor, *, prefix_queried: bool,
                               prefix_tail_logit=None):
    """Split a fused forward's logits into (p_ar [B+1,V] verify rows, replica_logits [B,B,V]).

    prefix_queried=True  (HF dense forward over the whole seq): p_ar = logits[P-1 : P+B],
                         replica_logits = logits[P+B:].
    prefix_queried=False (KV-cached runner querying ONLY [S|R*] — the minisgl live path): carry the
                         first-new-position prediction from the prior step's bonus-token logit row as
                         ``prefix_tail_logit`` [V]; p_ar = cat([prefix_tail_logit, logits[:B]]),
                         replica_logits = logits[B:]."""
    B, P, V = d.block_len, d.prefix_len, logits.shape[-1]
    if prefix_queried:
        assert logits.shape[0] == d.kv_len, (logits.shape, d.kv_len)
        p_ar = logits[P - 1: P + B]
        replica_logits = logits[P + B:].reshape(B, B, V)
    else:
        assert logits.shape[0] == d.q_len, (logits.shape, d.q_len)
        assert prefix_tail_logit is not None, "new-region-only forward needs the carried prefix-tail logit"
        p_ar = torch.cat([prefix_tail_logit.reshape(1, V), logits[:B]], dim=0)
        replica_logits = logits[B:].reshape(B, B, V)
    return p_ar, replica_logits
