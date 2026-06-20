"""Phase 3c-1: minisgl-native GDN attention metadata.

The vLLM `GDNAttentionMetadataBuilder` exists mostly to *split* one mixed batch into
prefill + decode subsets (`split_decodes_and_prefills`) and to handle spec-decode. minisgl
schedules HOMOGENEOUS batches — `Batch.phase` is `prefill` XOR `decode` — so that split is
already done by the scheduler. What's left is the per-batch tensors the GDN layer consumes
explicitly (see `gdn/layer.py`):

  * ``query_start_loc`` — cu_seqlens, int32 ``(num_seqs+1,)``. Prefill: cumulative
    ``extend_len`` (tokens processed THIS pass — a chunk may be a slice of a longer prompt).
    Decode: ``arange(0, num_seqs+1)`` (one token per sequence).
  * ``state_indices`` — int32 ``(num_seqs,)``, the GDN state-cache slot per sequence, in
    ``batch.reqs`` order. OWNED by the scheduler's slot allocator (3c-2); passed in here.
    Every entry is ``>= 1`` — slot 0 is the reserved NULL block (see `GDNStateCache`).
  * ``has_initial_state`` — bool ``(num_seqs,)``, prefill only. True ⟺ this sequence already
    has recurrent state to continue from. With the non-radix cache forced for GDN models
    (3c-2), ``cached_len > 0`` ⟺ chunk continuation, so that is exactly the predicate.

The FLA chunk metadata (`chunk_indices`/`chunk_offsets`) and the causal-conv1d metadata
(`nums_dict`/`batch_ptr`/`token_chunk_offset_ptr`) are PERF precomputation only: the kernels
compute them on the fly from ``cu_seqlens`` when omitted (verified in 3b-3 with
``metadata=None``; `cumsum.py` calls `prepare_chunk_indices` itself when `chunk_indices`
is None). MVP leaves them None; precompute is a later optimization to drop the GPU↔CPU sync.

Scope (3c): TP=1, eager (no cudagraph — GDN capture is out of scope this phase), no
spec-decode/MTP. DS-native conv state end-to-end (we own both cache and layer; 3b-3 proved
DS is bit-exact, so no SD transpose is threaded).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch

if TYPE_CHECKING:
    from minisgl.core import Batch, Req


@dataclass
class GDNMetadata:
    is_prefill: bool
    num_seqs: int
    query_start_loc: torch.Tensor  # int32 (num_seqs+1,) device — cu_seqlens
    state_indices: torch.Tensor  # int32 (num_seqs,) device — GDN slot per seq (all >= 1)
    has_initial_state: torch.Tensor | None = None  # bool (num_seqs,) device, prefill only

    # ---- perf precomputation (None == kernel computes on the fly; MVP leaves None) ----
    chunk_indices: torch.Tensor | None = None
    chunk_offsets: torch.Tensor | None = None
    nums_dict: dict | None = None
    batch_ptr: torch.Tensor | None = None
    token_chunk_offset_ptr: torch.Tensor | None = None


def build_gdn_metadata(
    batch: Batch,
    state_indices: torch.Tensor,
    device: torch.device,
) -> GDNMetadata:
    """Build per-batch GDN metadata from a homogeneous `Batch` + its state slots.

    ``state_indices`` is the scheduler-owned int32 slot id per sequence, in ``batch.reqs``
    order (length == ``batch.size``). It is packaged as-is; ``query_start_loc`` and
    ``has_initial_state`` are derived from the reqs. Eager-only, so ``batch.reqs`` (the real
    sequences) are used directly — no cudagraph padding rows.
    """
    reqs: List[Req] = batch.reqs
    num_seqs = len(reqs)
    assert state_indices.numel() == num_seqs, (
        f"state_indices ({state_indices.numel()}) must match batch.size ({num_seqs})"
    )

    if batch.is_decode:
        # one token per sequence: cu_seqlens = [0, 1, 2, ..., num_seqs]
        query_start_loc = torch.arange(num_seqs + 1, dtype=torch.int32, device=device)
        return GDNMetadata(
            is_prefill=False,
            num_seqs=num_seqs,
            query_start_loc=query_start_loc,
            state_indices=state_indices,
            has_initial_state=None,
        )

    # prefill: cu_seqlens over the tokens processed THIS pass (extend_len per seq).
    # Pin the host staging only for a CUDA target (page-locked H2D overlap); pinning needs
    # a live GPU, so the CPU path (unit tests) skips it.
    pin = device.type == "cuda" and torch.cuda.is_available()
    extend_lens = [req.extend_len for req in reqs]
    qsl_host = torch.zeros(num_seqs + 1, dtype=torch.int32, pin_memory=pin)
    torch.cumsum(
        torch.tensor(extend_lens, dtype=torch.int32), dim=0, out=qsl_host[1:]
    )
    query_start_loc = qsl_host.to(device, non_blocking=pin)

    # has_initial_state: True ⟺ continuation (cached_len > 0). With the non-radix cache
    # forced for GDN models, cached_len > 0 happens only for chunked-prefill continuations.
    has_initial = torch.tensor(
        [req.cached_len > 0 for req in reqs], dtype=torch.bool, pin_memory=pin
    ).to(device, non_blocking=pin)

    return GDNMetadata(
        is_prefill=True,
        num_seqs=num_seqs,
        query_start_loc=query_start_loc,
        state_indices=state_indices,
        has_initial_state=has_initial,
    )


__all__ = ["GDNMetadata", "build_gdn_metadata"]
