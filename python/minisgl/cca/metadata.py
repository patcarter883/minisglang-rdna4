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

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

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
    )


__all__ = ["CCAMetadata", "build_cca_metadata"]
