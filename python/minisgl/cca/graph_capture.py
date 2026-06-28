"""Static-buffer CCA decode metadata for cudagraph capture/replay (the CCA analog of
`gdn/graph_capture.py`).

A captured graph records kernel launches with fixed argument POINTERS; the CCA decode kernel
(``torch.ops.zaya_cca.cca_decode_qk``) reads ``state_indices`` (conv-state slot per sequence)
through those pointers, so for capture they must be PERSISTENT tensors whose CONTENTS we refresh in
place before each replay (the conv/prev_hs state buffers themselves are already persistent, indexed
by these slots).

Decode is one token per sequence, so ``query_start_loc`` is a fixed ``arange`` and only
``state_indices`` varies per step. Padding rows (cudagraph batch-size rounding) point at the reserved
NULL slot 0 — a real buffer row that no live sequence ever owns (the free-list starts at 1). The CCA
decode layer derives ``is_pad = (slot == 0)`` inside the graph, so padded rows are flagged pad: the
kernel skips their state roll and their garbage output is discarded (only ``batch.size`` rows are
read back).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .metadata import CCAMetadata

if TYPE_CHECKING:
    from minisgl.core import Batch


class CCAGraphCapture:
    def __init__(self, device: torch.device, max_bs: int) -> None:
        self._state_indices = torch.zeros(max_bs, dtype=torch.int32, device=device)
        self._cu = torch.arange(max_bs + 1, dtype=torch.int32, device=device)

    def _metadata(self, bs: int) -> CCAMetadata:
        return CCAMetadata(
            is_prefill=False,
            num_seqs=bs,
            query_start_loc=self._cu[: bs + 1],
            state_indices=self._state_indices[:bs],
            has_initial_state=None,
        )

    def prepare_for_capture(self, batch: "Batch") -> None:
        # Dummy capture batch: all rows -> NULL slot 0 (valid reserved row, no live slot touched).
        self._state_indices[: batch.padded_size].fill_(0)
        batch.cca_metadata = self._metadata(batch.padded_size)

    def prepare_for_replay(self, batch: "Batch") -> None:
        # Reuse the scheduler-built real slots (batch.cca_metadata.state_indices); pad with NULL 0.
        real = batch.cca_metadata.state_indices
        n = real.numel()
        self._state_indices[:n].copy_(real)
        self._state_indices[n : batch.padded_size].fill_(0)
        batch.cca_metadata = self._metadata(batch.padded_size)


__all__ = ["CCAGraphCapture"]
