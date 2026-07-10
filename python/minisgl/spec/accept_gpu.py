from __future__ import annotations

from typing import NamedTuple

import torch

__all__ = [
    "OnDeviceAccept",
    "OnDeviceCommit",
    "OnDeviceTruncate",
    "accept_greedy_ondevice",
    "build_commit_ondevice",
    "gather_committed_ids",
    "truncate_at_eos_ondevice",
]


class OnDeviceAccept(NamedTuple):
    """Pure-GPU result of segmented greedy speculative acceptance.

    All tensors live on ``device``; nothing here was pulled to host. A later phase can consume
    ``committed_flat`` + ``committed_offsets`` to commit tokens without a serializing sync.
    """

    num_accepted: torch.Tensor
    """int32 [num_reqs]: drafts accepted per req, in ``[0, K_r]``. Req r advances by this + 1."""
    committed_flat: torch.Tensor
    """int32 [sum(num_accepted_r + 1)]: committed token ids for all reqs, concatenated."""
    committed_offsets: torch.Tensor
    """int32 [num_reqs]: exclusive prefix-sum of ``committed_lens`` — start of each req's slice."""
    committed_lens: torch.Tensor
    """int32 [num_reqs]: ``num_accepted_r + 1``, the number of committed tokens for req r."""


def _exclusive_cumsum(x: torch.Tensor) -> torch.Tensor:
    """Exclusive prefix sum: out[i] = sum(x[:i]). Same dtype/shape as ``x``."""
    return torch.cumsum(x, dim=0) - x


def _leading_zero_run_per_segment(
    stop_flag: torch.Tensor,
    seg: torch.Tensor,
    seg_offsets: torch.Tensor,
    seg_lens: torch.Tensor,
    num_reqs: int,
) -> torch.Tensor:
    """Per-segment length of the leading run of zeros in a flat 0/1 ``stop_flag`` buffer.

    The flat buffer is partitioned into ``num_reqs`` contiguous segments; ``seg[j]`` is the owning
    segment of flat row ``j``, ``seg_offsets`` the exclusive prefix-sum of ``seg_lens`` (each
    segment's start). For each segment this returns the number of positions BEFORE the first
    ``stop_flag == 1`` (equivalently: the local index of that first 1). A segment with no 1 returns
    its full length ``seg_lens[r]``.

    Vectorized "segmented arg-first-true": a global inclusive cumsum of the flag, rebased to each
    segment's start, is ``0`` exactly on the leading zero-run; summing that 0/1 carry per segment
    counts the run. This is the same trick Phase-1 uses for the accepted-prefix length. Everything
    stays on ``stop_flag.device`` — no host sync.
    """
    total = stop_flag.shape[0]
    if total == 0:
        # Empty flat buffer (every segment length 0 — e.g. all reqs have K=0, an ngram all-miss
        # batch). Each segment's leading-zero run is its length, which is 0. Guard before indexing
        # ``excl`` (size 0) below.
        return torch.zeros(num_reqs, dtype=torch.int32, device=stop_flag.device)
    incl = torch.cumsum(stop_flag, dim=0)  # [total] global inclusive
    excl = incl - stop_flag  # [total] global exclusive
    # Flag-count strictly BEFORE each segment starts (0 for the first). Clamp guards a possible
    # offset==total for a trailing empty segment (which contributes no rows to the sum anyway).
    start_idx = seg_offsets.to(torch.int64).clamp_max(max(total - 1, 0))
    seg_start_excl = excl[start_idx]  # [num_reqs]
    seg_start_excl_flat = torch.repeat_interleave(seg_start_excl, seg_lens.to(torch.int64))
    seg_incl = incl - seg_start_excl_flat  # [total] segmented inclusive cumsum
    carry = (seg_incl == 0).to(torch.int32)  # [total] 1 while still in the leading zero-run
    run = torch.zeros(num_reqs, dtype=torch.int32, device=stop_flag.device)
    run.index_add_(0, seg, carry)  # segment-sum of carry == leading-zero-run length
    return run


def gather_committed_ids(
    target_argmax: torch.Tensor,
    target_offsets: torch.Tensor,
    committed_lens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather each req's committed prefix ``target[target_offset_r : target_offset_r + len_r]``
    into one flat buffer, fully on-device (no host sync in this function).

    Returns ``(committed_flat, committed_offsets)`` where committed_offsets is the exclusive
    prefix-sum of committed_lens.
    """
    device = target_argmax.device
    num_reqs = committed_lens.shape[0]
    committed_offsets = _exclusive_cumsum(committed_lens)

    # Expand per-req scalars to one entry per committed token WITHOUT knowing the total length on
    # host: repeat_interleave(tensor, repeats_tensor) is a device op; we never call .item().
    seg_ids = torch.arange(num_reqs, device=device, dtype=torch.int64)
    committed_seg = torch.repeat_interleave(seg_ids, committed_lens.to(torch.int64))  # [total]

    # arange(total) built as cumsum-of-ones so we never need the host-side total.
    ones = torch.ones_like(committed_seg)
    global_pos = torch.cumsum(ones, dim=0) - 1  # 0,1,2,...,total-1

    within = global_pos - committed_offsets.to(torch.int64)[committed_seg]  # index inside its req
    gather_idx = target_offsets.to(torch.int64)[committed_seg] + within
    committed_flat = target_argmax[gather_idx].to(torch.int32)
    return committed_flat, committed_offsets.to(torch.int32)


def accept_greedy_ondevice(
    target_argmax: torch.Tensor,
    drafts_gpu: torch.Tensor,
    q_lens: torch.Tensor,
    device: torch.device,
) -> OnDeviceAccept:
    """Vectorized, on-device greedy speculative acceptance over a flattened multi-req batch.

    Layout (mirrors the host path in ``_spec_decode_step``):
      * ``q_lens[r] = K_r + 1``       -- verify query rows for req r.
      * ``target_argmax`` is ``[sum(q_len)]`` -- per-position argmax; req r occupies the slice
        ``[target_offset_r : target_offset_r + q_len_r]`` where target_offset = cumsum(q_len).
      * ``drafts_gpu`` is ``[sum(K)]``       -- req r occupies ``[draft_offset_r : +K_r]`` where
        draft_offset = cumsum(K), K_r = q_len_r - 1.

    For each req: ``num_accepted_r`` = length of the leading run of ``target[i] == draft[i]``
    (i in 0..K_r-1); committed = ``target[target_offset_r : target_offset_r + num_accepted_r + 1]``.
    This matches ``spec.accept.verify_greedy`` exactly, per req.

    No ``.cpu()/.item()/.tolist()`` anywhere — everything stays on ``device``.
    """
    q_lens = q_lens.to(device=device, dtype=torch.int32)
    target_argmax = target_argmax.to(device=device, dtype=torch.int32)
    drafts_gpu = drafts_gpu.to(device=device, dtype=torch.int32)

    num_reqs = q_lens.shape[0]
    K = q_lens - 1  # [num_reqs] drafts per req (>= 0)
    S = drafts_gpu.shape[0]  # sum(K); a tensor SHAPE, so a host int without a device sync

    draft_offsets = _exclusive_cumsum(K)  # [num_reqs]
    target_offsets = _exclusive_cumsum(q_lens)  # [num_reqs]

    # --- 1. draft->target alignment. Each draft row j (global, in [0,S)) belongs to req seg[j],
    # with within-req index i = j - draft_offset[seg]. Its comparison target row is at
    # target_offset[seg] + i (the target has one EXTRA "bonus" row per req, so the layouts differ
    # only by that per-req shift).
    seg_ids = torch.arange(num_reqs, device=device, dtype=torch.int64)
    seg = torch.repeat_interleave(seg_ids, K.to(torch.int64))  # [S] owning-req per draft row
    global_draft_idx = torch.arange(S, device=device, dtype=torch.int64)  # 0..S-1
    within = global_draft_idx - draft_offsets.to(torch.int64)[seg]  # [S] within-req index
    target_gather_idx = target_offsets.to(torch.int64)[seg] + within  # [S]

    matched = (target_argmax[target_gather_idx] == drafts_gpu).to(torch.int32)  # [S] 1 if match
    mismatch = 1 - matched  # [S]

    # --- 2. segmented "longest leading-true prefix" == leading-zero run of the mismatch flag.
    # num_accepted_r = number of draft rows before the first mismatch (they form a prefix run).
    num_accepted = _leading_zero_run_per_segment(mismatch, seg, draft_offsets, K, num_reqs)

    committed_lens = num_accepted + 1  # always emit the bonus/correction token
    committed_flat, committed_offsets = gather_committed_ids(
        target_argmax, target_offsets, committed_lens
    )
    return OnDeviceAccept(
        num_accepted=num_accepted,
        committed_flat=committed_flat,
        committed_offsets=committed_offsets,
        committed_lens=committed_lens,
    )


class OnDeviceTruncate(NamedTuple):
    """Pure-GPU result of EOS truncation of a committed multi-req batch.

    All tensors live on ``device``; nothing here was pulled to host.
    """

    kept_lens: torch.Tensor
    """int32 [num_reqs]: committed length after truncating at (and including) the first kept EOS."""
    kept_finished_eos: torch.Tensor
    """bool [num_reqs]: True where an EOS was kept (i.e. the req finished on EOS this step)."""
    kept_flat: torch.Tensor
    """int32 [sum(kept_lens)]: the truncated committed token ids, concatenated across reqs."""
    kept_offsets: torch.Tensor
    """int32 [num_reqs]: exclusive prefix-sum of ``kept_lens`` — start of each req's kept slice."""


def truncate_at_eos_ondevice(
    committed_flat: torch.Tensor,
    committed_offsets: torch.Tensor,
    committed_lens: torch.Tensor,
    eos_token_id: int,
    ignore_eos_mask: torch.Tensor,
    device: torch.device,
) -> OnDeviceTruncate:
    """Vectorized, on-device EOS truncation of an already-committed flattened multi-req batch.

    Mirrors the host ``keep``-loop in ``_spec_decode_step``:

        keep = []
        for tok in emitted:
            keep.append(tok)
            if (not ignore_eos) and tok == eos_token_id:
                break

    i.e. for each req, keep the committed prefix up to and INCLUDING the first EOS token, unless the
    req has ``ignore_eos`` (then keep everything). ``kept_finished_eos[r]`` is True iff an EOS was
    kept for req r.

    Args:
      * ``committed_flat`` / ``committed_offsets`` / ``committed_lens``: the Phase-1 committed buffer
        (``committed_offsets`` = exclusive prefix-sum of ``committed_lens``; each req occupies
        ``committed_flat[off_r : off_r + len_r]``). ``committed_lens`` are all ``>= 1``.
      * ``ignore_eos_mask``: bool ``[num_reqs]``, True where the req should NOT truncate at EOS.

    Vectorization: build a per-flat-position ``is_eos AND not ignore_eos(seg)`` stop-flag, then take
    the leading-zero-run per segment (the SAME segmented arg-first-true trick Phase-1 uses). That run
    length is the local index of the first kept EOS; ``kept_len = min(committed_len, run + 1)`` keeps
    through and including it (and collapses to ``committed_len`` when there is no EOS, since then
    ``run == committed_len``). ``has_eos = run < committed_len``.

    No ``.cpu()/.item()/.tolist()`` anywhere — everything stays on ``device``.
    """
    committed_flat = committed_flat.to(device=device, dtype=torch.int32)
    committed_offsets = committed_offsets.to(device=device, dtype=torch.int32)
    committed_lens = committed_lens.to(device=device, dtype=torch.int32)
    ignore_eos_mask = ignore_eos_mask.to(device=device, dtype=torch.bool)

    num_reqs = committed_lens.shape[0]

    # Owning-req per flat committed row (repeat_interleave is a device op; never .item()).
    seg_ids = torch.arange(num_reqs, device=device, dtype=torch.int64)
    seg = torch.repeat_interleave(seg_ids, committed_lens.to(torch.int64))  # [total]

    # Stop flag: 1 where this position is an EOS token AND its req does not ignore EOS.
    not_ignore_seg = (~ignore_eos_mask)[seg]  # [total] bool, per-position "eos counts here"
    is_eos = (committed_flat == eos_token_id) & not_ignore_seg  # [total] bool
    stop_flag = is_eos.to(torch.int32)

    # first_eos_local[r] = #positions before the first kept EOS = leading-zero run of stop_flag.
    # When a req has no (counted) EOS the run == committed_lens[r].
    first_eos_local = _leading_zero_run_per_segment(
        stop_flag, seg, committed_offsets, committed_lens, num_reqs
    )
    has_eos = first_eos_local < committed_lens  # [num_reqs] bool
    kept_lens = torch.minimum(committed_lens, first_eos_local + 1)  # [num_reqs] int32

    # Re-gather the truncated ids: each kept slice is the first kept_len rows of its committed slice,
    # so gather_committed_ids over (source=committed_flat, base=committed_offsets, lens=kept_lens).
    kept_flat, kept_offsets = gather_committed_ids(committed_flat, committed_offsets, kept_lens)
    return OnDeviceTruncate(
        kept_lens=kept_lens,
        kept_finished_eos=has_eos,
        kept_flat=kept_flat,
        kept_offsets=kept_offsets,
    )


class OnDeviceCommit(NamedTuple):
    """Pure-GPU result of laying out the token-pool scatter + advancing per-req lengths.

    All tensors live on ``device``. ``scatter_rows``/``scatter_cols``/``scatter_vals`` are a
    ready-to-apply flattened indexed assignment into the ``[num_seqs, max_ctx]`` token pool:
    ``token_pool[scatter_rows, scatter_cols] = scatter_vals``. The rest are per-req vectors the
    scheduler previously advanced with a host loop (``req.cached_len``, ``req.device_len``, the GDN
    install ``t_index``, and the draft-head seed row).
    """

    scatter_rows: torch.Tensor
    """int64 [sum(kept_lens)]: token-pool ROW (== req.table_idx) per committed token."""
    scatter_cols: torch.Tensor
    """int64 [sum(kept_lens)]: token-pool COLUMN (== c0 + 1 + within-req index) per committed token."""
    scatter_vals: torch.Tensor
    """int32 [sum(kept_lens)]: the committed token ids (== the input ``kept_flat``), in row-major
    req-then-token order — the SAME order the host ``c_rows/c_cols/c_vals`` loop produced."""
    new_cached_len: torch.Tensor
    """int32 [num_reqs]: ``c0 + kept_lens`` — KV valid through ``new_cached_len - 1``."""
    new_device_len: torch.Tensor
    """int32 [num_reqs]: ``new_cached_len + 1`` (the bonus row recomputed next step)."""
    gdn_t_index: torch.Tensor
    """int32 [num_reqs]: ``kept_lens - 1`` — the captured scratch index to install (state AFTER the
    last committed token). Only meaningful for still-running reqs (caller masks by ``finished``)."""
    seed_rows: torch.Tensor
    """int64 [num_reqs]: ``target_offset_r + kept_lens - 1`` — the verify-output row that produced
    each req's last committed token (draft-head hidden-state seed). Caller masks by ``finished``."""


def build_commit_ondevice(
    kept_flat: torch.Tensor,
    kept_lens: torch.Tensor,
    table_idx: torch.Tensor,
    cached_len_c0: torch.Tensor,
    target_offsets: torch.Tensor,
    device: torch.device,
) -> OnDeviceCommit:
    """Lay out the GPU token-pool scatter + advance per-req lengths, fully on-device.

    Replaces the host commit loop in ``scheduler._spec_decode_step`` (the
    ``c_rows/c_cols/c_vals`` append + ``req.cached_len``/``req.device_len`` advance + the GDN
    ``install`` ``t_index`` + the draft-head seed row), all of which previously read
    ``.tolist()``-ed per-req scalars on the host.

    Layout (mirrors the host loop, per req r with ``c0 = req.cached_len`` BEFORE this step):
      * committed token j of req r lands at ``token_pool[table_idx[r], c0[r] + 1 + j]``.
      * ``req.cached_len`` advances to ``c0[r] + kept_lens[r]`` (KV valid through cached_len-1).
      * ``req.device_len`` advances to ``cached_len + 1``.
      * the GDN/CCA install index is ``kept_lens[r] - 1`` (state after the last committed token).
      * the draft-head seed row is ``target_offset[r] + kept_lens[r] - 1`` (``target_offset`` is the
        req's ``block_start`` in the ``[sum(q_len)]`` verify output — cumsum of ``q_len``).

    Args:
      * ``kept_flat`` / ``kept_lens``: the Phase-2a truncated committed buffer (``kept_lens`` all
        ``>= 1``, in req order; ``kept_flat`` concatenated in req-then-token order).
      * ``table_idx``: int ``[num_reqs]`` — each req's token-pool / page-table row (``req.table_idx``).
      * ``cached_len_c0``: int ``[num_reqs]`` — each req's ``cached_len`` BEFORE the commit.
      * ``target_offsets``: int ``[num_reqs]`` — the verify-output ``block_start`` per req
        (``_exclusive_cumsum(q_lens)`` from Phase-1). Used only for the draft-head seed row.

    The scatter triple comes out in req-then-token order (``repeat_interleave`` of ``arange``), the
    identical order the host ``for i in reqs: for j in keep`` loop produced — so a validator can
    compare element-wise, not just as a set.

    No ``.cpu()/.item()/.tolist()`` anywhere — everything stays on ``device``.
    """
    kept_flat = kept_flat.to(device=device, dtype=torch.int32)
    kept_lens = kept_lens.to(device=device, dtype=torch.int32)
    table_idx = table_idx.to(device=device, dtype=torch.int64)
    cached_len_c0 = cached_len_c0.to(device=device, dtype=torch.int32)
    target_offsets = target_offsets.to(device=device, dtype=torch.int32)

    num_reqs = kept_lens.shape[0]
    kept_offsets = _exclusive_cumsum(kept_lens)  # [num_reqs] start of each req's kept slice

    # Owning-req per committed token (repeat_interleave is a device op; never .item()).
    seg_ids = torch.arange(num_reqs, device=device, dtype=torch.int64)
    seg = torch.repeat_interleave(seg_ids, kept_lens.to(torch.int64))  # [total]

    # within-req token index, built cumsum-of-ones so we never need the host-side total length.
    ones = torch.ones_like(seg)
    global_pos = torch.cumsum(ones, dim=0) - 1  # 0..total-1
    within = global_pos - kept_offsets.to(torch.int64)[seg]  # [total] j inside its req

    scatter_rows = table_idx[seg]  # [total]
    scatter_cols = cached_len_c0.to(torch.int64)[seg] + 1 + within  # [total] c0 + 1 + j
    scatter_vals = kept_flat  # already req-then-token order

    new_cached_len = cached_len_c0 + kept_lens  # [num_reqs]
    new_device_len = new_cached_len + 1
    gdn_t_index = kept_lens - 1
    seed_rows = (target_offsets + kept_lens - 1).to(torch.int64)

    return OnDeviceCommit(
        scatter_rows=scatter_rows,
        scatter_cols=scatter_cols,
        scatter_vals=scatter_vals,
        new_cached_len=new_cached_len,
        new_device_len=new_device_len,
        gdn_t_index=gdn_t_index,
        seed_rows=seed_rows,
    )
