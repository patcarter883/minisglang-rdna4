"""Per-batch metadata for the ZAYA CCA conv front-end (mirrors gdn/metadata.py).

minisgl schedules HOMOGENEOUS batches — ``Batch.phase`` is prefill XOR decode — so unlike the vLLM
CCA reference (which splits one mixed batch into decode-first + prefill regions) there is NO mixed
batch and NO decode/prefill split inside the CCA layer; dispatch is purely on ``batch.phase``.

What the CCA layer (`models/zaya.py: ZayaCCAAttn.forward`) consumes:

  * ``query_start_loc`` — cu_seqlens, int32 ``(num_seqs+1,)``. Decode: ``arange(num_seqs+1)`` (one
    token per seq). Prefill: cumulative ``extend_len`` (tokens processed THIS pass — a chunk may be
    a slice of a longer prompt).
  * ``state_indices`` — int32 ``(num_seqs,)``, the CCA conv-state slot per sequence, in
    ``batch.reqs`` order. OWNED by the scheduler's slot allocator (CCASlotManager); passed in here.
    Every entry is ``>= 1`` — slot 0 is the reserved NULL block (see `CCAStateCache`).
  * ``has_initial_state`` — bool ``(num_seqs,)``, prefill only. True ⟺ this sequence already carries
    recurrent conv/prev_hs state to continue from (chunked-prefill continuation). With the non-radix
    cache forced for CCA models, ``cached_len > 0`` is exactly that predicate.

Scope: TP=1, no spec-decode/MTP (v0). CCA cudagraph capture is wired (CCAGraphCapture provides the
recurrent-state static buffers, same shape as GDN's) and shares the grad-free capture fix that
unblocked GDN — Zaya end-to-end capture validation is still pending (needs the TP=2 boot), but the
mechanism is present and the "eager only" note no longer reflects a hard limitation. The kernel
computes everything else it needs (seg_pos / req_id / is_last for prefill) from these.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List

import torch

if TYPE_CHECKING:
    from minisgl.core import Batch, Req


@dataclass
class CCAMetadata:
    is_prefill: bool
    num_seqs: int
    query_start_loc: torch.Tensor  # int32 (num_seqs+1,) device — cu_seqlens
    state_indices: torch.Tensor  # int32 (num_seqs,) device — CCA conv slot per seq (all >= 1)
    has_initial_state: torch.Tensor | None = None  # bool (num_seqs,) device, prefill only
    # ---- spec-decode verify capture (mirrors gdn/metadata.py). When capture_verify_state is True the
    # CCA layer runs the varlen (prefill-style) path AND stashes, per cca_layer_id, the conv window +
    # prev-hidden state AFTER each of the K+1 verify tokens into scratch. `verify_max_qlen` = max(K+1).
    # The scheduler installs the state after the accepted prefix (index = accepted_count-1) into the
    # slot (CCAStateCache.install_verify_state) — no snapshot + re-advance, bit-exact vs 1-token decode.
    # Unlike GDN this needs NO new HIP kernel: conv_state is raw qk_new columns (a rolling window — see
    # cca_kernel.hip), so per-token windows reconstruct in torch; prev_hs is just the prior input hs.
    capture_verify_state: bool = False
    verify_max_qlen: int = 0
    conv_scratch: Dict[int, torch.Tensor] = field(default_factory=dict)  # id -> [Q, N, C, TP] fp32
    prev_scratch: Dict[int, torch.Tensor] = field(default_factory=dict)  # id -> [Q, N, hidden] fp32
    seg_lens: List[int] | None = None  # host per-seq extend_len (prefill/verify); sync-free segment walk


def build_cca_metadata(
    batch: Batch,
    state_indices: torch.Tensor,
    device: torch.device,
) -> CCAMetadata:
    """Build per-batch CCA metadata from a homogeneous `Batch` + its conv-state slots.

    ``state_indices`` is the scheduler-owned int32 slot id per sequence, in ``batch.reqs`` order
    (length == ``batch.size``). It is packaged as-is; ``query_start_loc`` and ``has_initial_state``
    are derived from the reqs. Eager-only, so ``batch.reqs`` are used directly (no padding rows).
    """
    reqs: List[Req] = batch.reqs
    num_seqs = len(reqs)
    assert state_indices.numel() == num_seqs, (
        f"state_indices ({state_indices.numel()}) must match batch.size ({num_seqs})"
    )

    if batch.is_decode and not batch.spec_verify:
        # one token per sequence: cu_seqlens = [0, 1, 2, ..., num_seqs]
        query_start_loc = torch.arange(num_seqs + 1, dtype=torch.int32, device=device)
        return CCAMetadata(
            is_prefill=False,
            num_seqs=num_seqs,
            query_start_loc=query_start_loc,
            state_indices=state_indices,
            has_initial_state=None,
        )

    # prefill: cu_seqlens over the tokens processed THIS pass (extend_len per seq). Pin host staging
    # only for a live CUDA/HIP target (page-locked H2D overlap); CPU unit tests skip it.
    pin = device.type == "cuda" and torch.cuda.is_available()
    extend_lens = [req.extend_len for req in reqs]
    qsl_host = torch.zeros(num_seqs + 1, dtype=torch.int32, pin_memory=pin)
    torch.cumsum(torch.tensor(extend_lens, dtype=torch.int32), dim=0, out=qsl_host[1:])
    query_start_loc = qsl_host.to(device, non_blocking=pin)

    # has_initial_state: True ⟺ continuation (cached_len > 0). With the non-radix cache forced for
    # CCA models, cached_len > 0 happens only for chunked-prefill continuations.
    has_initial = torch.tensor(
        [req.cached_len > 0 for req in reqs], dtype=torch.bool, pin_memory=pin
    ).to(device, non_blocking=pin)

    return CCAMetadata(
        is_prefill=True,
        num_seqs=num_seqs,
        query_start_loc=query_start_loc,
        state_indices=state_indices,
        has_initial_state=has_initial,
        seg_lens=extend_lens,
    )


def capture_cca_verify_state(
    md: CCAMetadata,
    cca_layer_id: int,
    qk_new: torch.Tensor,       # [N, C] fp32 — packed q|k inputs to the conv front-end
    init_states: torch.Tensor,  # [num_seqs, C, TP] fp32 — cached conv window per seq (0 if fresh)
    hs: torch.Tensor,           # [N, hidden] — input hidden states (model dtype)
) -> None:
    """Reconstruct + stash the per-token CCA recurrent state during a spec-decode verify forward.

    NO HIP kernel: ``conv_state`` is a rolling window of RAW ``qk_new`` columns (cca_kernel.hip does
    ``window[c,TP]=qk_new[s,c]``; ``conv_states[slot,c,i]=window[c,i+1]`` — roll left, new token at
    tail). So the window AFTER seq i's token j is a plain unfold of ``[init_window ++ qk_new_i]``.
    ``prev_hs`` after token j is just ``hs[token j]`` (decode does ``prev[slot]=hs``). Stashes into
    ``md.conv_scratch/prev_scratch[cca_layer_id]`` as ``[Q, num_seqs, ...]`` (Q = verify_max_qlen);
    the scheduler gathers index ``accepted_count-1`` (state AFTER the last accepted token).
    """
    num_seqs = md.num_seqs
    Q = md.verify_max_qlen
    C = qk_new.shape[1]
    TP = init_states.shape[2]
    hidden = hs.shape[1]
    seg = md.seg_lens
    assert seg is not None and len(seg) == num_seqs, "verify capture needs host seg_lens"
    # v2 S2 (cudagraph): if the scratch is pre-allocated (a persistent static buffer supplied by the
    # verify-graph capturer), WRITE IN-PLACE so the captured graph's pointer stays valid across replays.
    # Otherwise (eager path) allocate fresh, as before. Either way the fill loop below is identical.
    conv_scr = md.conv_scratch.get(cca_layer_id)
    prev_scr = md.prev_scratch.get(cca_layer_id)
    if conv_scr is None:
        conv_scr = qk_new.new_zeros((Q, num_seqs, C, TP))            # fp32 (qk_new is fp32)
        prev_scr = qk_new.new_zeros((Q, num_seqs, hidden))          # fp32, matches prev_hs dtype
        md.conv_scratch[cca_layer_id] = conv_scr
        md.prev_scratch[cca_layer_id] = prev_scr
    else:
        conv_scr.zero_()                                            # persistent buffer: reset last step
        prev_scr.zero_()
    off = 0
    for i in range(num_seqs):
        Li = int(seg[i])
        s, e = off, off + Li
        off = e
        if Li <= 0:
            continue
        x = qk_new[s:e]                                             # [Li, C]
        # stream positions: [oldest_init, ..., newest_init, x0, x1, ...] -> [TP+Li, C]
        stream = torch.cat([init_states[i].transpose(0, 1), x], dim=0)  # [TP+Li, C]
        windows = stream.unfold(0, TP, 1)                          # [Li+1, C, TP]; win[m]=stream[m:m+TP]
        conv_scr[:Li, i] = windows[1 : 1 + Li]                     # window AFTER token j == win[j+1]
        prev_scr[:Li, i] = hs[s:e].float()                        # prev after token j == hs[token j]
    # (scratch already bound above: fresh-alloc path set the dict; persistent-buffer path wrote in place)


__all__ = ["CCAMetadata", "build_cca_metadata", "capture_cca_verify_state"]
