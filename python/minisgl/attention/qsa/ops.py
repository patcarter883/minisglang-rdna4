"""The QSA selection ops — native HIP (`qsa_index`) with a torch reference behind one switch.

THE HIP KERNELS ARE THE PRODUCT. `rdna4-hip-kernels/qsa_index` is Triton-free and tilelang-free:

    qsa_index_score_paged(q, k_cache, page_table, row_starts, row_ends, row_seq, logits,
                          scale, num_key_cols)
        logits[m, j] = ( SUM_h relu(q[m,h] . key_j) ) / scale   for j in [row_starts[m], row_ends[m])
        columns inside [0, num_key_cols) but outside the row window are written -inf.
    qsa_index_topk(logits, row_starts, lengths, out)
        the `out.shape[1]` largest of each row's window, as RELATIVE indices, ASCENDING, -1 padded,
        with a TOTAL tie order (value descending, index ascending) so two ranks agree entry-for-entry.
    qsa_index_expand(block_indices, query_positions, seq_lens, out, ratio, token_topk)
        block b -> tokens b*ratio .. b*ratio+ratio-1 capped at token_topk, then the query's own
        trailing partial group; valid entries contiguous at the front, -1 padded.

WHY A TORCH PATH EXISTS AT ALL, and what it is not. It is NOT a serving fallback — `MINISGL_QSA_OPS`
defaults to `hip` and a missing `qsa_index` package RAISES. It exists so that (a) the gate tests can
run the identical selection on a box/build without the package and still be meaningful, and (b) a
kernel/torch A/B is one env var, which is how the kernels' index-set equality was checked against
something not derived from them. It is deliberately written from `config`'s math, using torch.sort
with an explicit (value desc, index asc) lexicographic key so it reproduces the kernel's TOTAL tie
order rather than numpy's or torch.topk's unspecified one.
"""

from __future__ import annotations

import math
import os
from typing import Optional

import torch

_BACKEND: Optional[str] = None
_OPS = None


def backend() -> str:
    """"hip" (default) or "torch" (MINISGL_QSA_OPS=torch). Resolved once."""
    global _BACKEND, _OPS
    if _BACKEND is None:
        want = os.environ.get("MINISGL_QSA_OPS", "hip").lower()
        if want not in ("hip", "torch"):
            raise ValueError(f"MINISGL_QSA_OPS must be 'hip' or 'torch', got {want!r}")
        if want == "hip":
            import qsa_index  # noqa: F401  — raises with the real ImportError if absent

            _OPS = qsa_index
        _BACKEND = want
    return _BACKEND


def score_paged(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    row_seq: torch.Tensor,
    logits: torch.Tensor,
    scale: float,
) -> None:
    """Score every query row against its sequence's compressed keys, THROUGH the page table.

    q         [rows, Hi, D]                (already normed + roped)
    k_cache   [n_comp_slots, 1, 1, D]      compressed-key cache, page_size 1
    page_table[bs, max_blocks] int32       block g of sequence s -> compressed slot
    row_*     [rows] int32                 this row's [start, end) block window
    row_seq   [rows] int32                 query row -> sequence index
    logits    [rows, num_key_cols] float32 OUT (caller-owned; the served path is graph-captured)
    """
    if backend() == "hip":
        _OPS.qsa_index_score_paged(
            q, k_cache, page_table, row_starts, row_ends, row_seq, logits, scale,
            logits.shape[1],
        )
        return
    _torch_score_paged(q, k_cache, page_table, row_starts, row_ends, row_seq, logits, scale)


def topk(logits: torch.Tensor, row_starts: torch.Tensor, lengths: torch.Tensor,
         out: torch.Tensor) -> None:
    """Exact top-`out.shape[1]` of each row's window, RELATIVE, ASCENDING, -1 padded."""
    if backend() == "hip":
        _OPS.qsa_index_topk(logits, row_starts, lengths, out)
        return
    _torch_topk(logits, row_starts, lengths, out)


def expand(block_indices: torch.Tensor, query_positions: torch.Tensor, seq_lens: torch.Tensor,
           out: torch.Tensor, ratio: int, token_topk: int) -> None:
    """Selected BLOCKS -> selected TOKEN positions, valid entries contiguous at the front."""
    if backend() == "hip":
        _OPS.qsa_index_expand(block_indices, query_positions, seq_lens, out, ratio, token_topk)
        return
    _torch_expand(block_indices, query_positions, seq_lens, out, ratio, token_topk)


# ---------------------------------------------------------------------------------------------
# torch reference (MINISGL_QSA_OPS=torch) — written from the math in config.py's docstring.
# ---------------------------------------------------------------------------------------------

def _torch_score_paged(q, k_cache, page_table, row_starts, row_ends, row_seq, logits, scale) -> None:
    rows, _, dim = q.shape
    cols = logits.shape[1]
    logits.fill_(-float("inf"))
    if rows == 0 or cols == 0:
        return
    starts = row_starts.to(torch.long)
    ends = torch.clamp(row_ends.to(torch.long), max=cols)
    seqs = row_seq.to(torch.long)
    keys = k_cache.reshape(k_cache.shape[0], dim)
    col_ix = torch.arange(cols, device=q.device)
    # page_size is 1 here, so the "page table" IS the slot map: gather each row's window directly.
    slots = page_table.to(torch.long)[seqs][:, :cols]                      # [rows, cols]
    valid = (col_ix[None, :] >= starts[:, None]) & (col_ix[None, :] < ends[:, None])
    gathered = keys[torch.where(valid, slots, torch.zeros_like(slots))]    # [rows, cols, D]
    scores = torch.einsum("mhd,mcd->mch", q.float(), gathered.float())
    vals = torch.relu(scores).sum(dim=-1) / scale
    logits.copy_(torch.where(valid, vals, torch.full_like(vals, -float("inf"))))


def _torch_topk(logits, row_starts, lengths, out) -> None:
    k = out.shape[1]
    out.fill_(-1)
    starts = row_starts.to(torch.long).tolist()
    lens = lengths.to(torch.long).tolist()
    for row in range(out.shape[0]):
        start, length = int(starts[row]), int(lens[row])
        if length <= 0:
            continue
        if length <= k:
            out[row, :length] = torch.arange(length, dtype=out.dtype, device=out.device)
            continue
        seg = logits[row, start:start + length].float()
        # TOTAL order: value DESCENDING, index ASCENDING. torch.sort(stable=True) on the negated
        # value keeps original (ascending-index) order among equals, which IS the tie rule.
        order = torch.argsort(-seg, stable=True)[:k]
        out[row] = torch.sort(order)[0].to(out.dtype)


def _torch_expand(block_indices, query_positions, seq_lens, out, ratio, token_topk) -> None:
    out.fill_(-1)
    final = out.shape[1]
    blocks = block_indices.to(torch.long)
    pos = query_positions.to(torch.long).reshape(-1)
    lens = seq_lens.to(torch.long).reshape(-1)
    n_block = token_topk // ratio
    # block b -> tokens b*ratio + o, dropped when b < 0 or the token is past the sequence.
    b = blocks[:, :n_block]                                                # [rows, n_block]
    off = torch.arange(ratio, device=out.device)
    tok = (b[:, :, None] * ratio + off[None, None, :]).reshape(b.shape[0], -1)   # [rows, token_topk]
    ok = (b[:, :, None] >= 0).expand(-1, -1, ratio).reshape(tok.shape)
    ok = ok & (tok >= 0) & (tok < lens[:, None])
    # trailing partial group of the query's own position
    visible = pos + 1
    tail_start = (visible // ratio) * ratio
    n_tail = torch.clamp(
        torch.minimum(visible - tail_start, torch.clamp(lens - tail_start, min=0)),
        min=0, max=ratio - 1,
    )
    tail_off = torch.arange(ratio - 1, device=out.device)
    tail_tok = tail_start[:, None] + tail_off[None, :]
    tail_ok = tail_off[None, :] < n_tail[:, None]
    all_tok = torch.cat([tok, tail_tok], dim=1)
    all_ok = torch.cat([ok, tail_ok], dim=1)
    # compact valid entries to the front, order-preserving
    order = torch.argsort((~all_ok).to(torch.int8), dim=1, stable=True)
    packed = torch.gather(all_tok, 1, order)
    packed_ok = torch.gather(all_ok, 1, order)
    packed = torch.where(packed_ok, packed, torch.full_like(packed, -1))
    out.copy_(packed[:, :final].to(out.dtype))


def sqrt_scale(head_dim: int) -> float:
    return math.sqrt(float(head_dim))


__all__ = ["backend", "score_paged", "topk", "expand", "sqrt_scale"]
