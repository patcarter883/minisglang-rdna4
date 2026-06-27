"""Static-buffer GDN decode metadata for cudagraph capture/replay.

A captured graph records kernel launches with fixed argument POINTERS; the GDN decode kernels read
``state_indices`` (recurrent-state slot per sequence) and ``query_start_loc`` through those pointers,
so for capture they must be PERSISTENT tensors whose CONTENTS we refresh in place before each replay
(the conv/ssm state buffers themselves are already persistent, indexed by these slots).

Decode is one token per sequence, so ``query_start_loc`` is a fixed ``arange`` (every q-length is 1)
and only ``state_indices`` varies per step. Padding rows (cudagraph batch-size rounding) point at the
reserved NULL slot 0 — a real buffer row that no live sequence ever owns (the free-list starts at 1),
so writing it is harmless and its garbage output is discarded (only ``batch.size`` rows are read back).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .metadata import GDNMetadata

if TYPE_CHECKING:
    from minisgl.core import Batch


class GDNGraphCapture:
    def __init__(self, device: torch.device, max_bs: int) -> None:
        self._state_indices = torch.zeros(max_bs, dtype=torch.int32, device=device)
        self._cu = torch.arange(max_bs + 1, dtype=torch.int32, device=device)

    def _metadata(self, bs: int) -> GDNMetadata:
        return GDNMetadata(
            is_prefill=False,
            num_seqs=bs,
            query_start_loc=self._cu[: bs + 1],
            state_indices=self._state_indices[:bs],
            has_initial_state=None,
        )

    def prepare_for_capture(self, batch: "Batch") -> None:
        # Dummy capture batch: all rows -> NULL slot 0 (valid reserved row, no live slot touched).
        self._state_indices[: batch.padded_size].fill_(0)
        batch.gdn_metadata = self._metadata(batch.padded_size)

    def prepare_for_replay(self, batch: "Batch") -> None:
        # Reuse the scheduler-built real slots (batch.gdn_metadata.state_indices); pad with NULL 0.
        real = batch.gdn_metadata.state_indices
        n = real.numel()
        self._state_indices[:n].copy_(real)
        self._state_indices[n : batch.padded_size].fill_(0)
        batch.gdn_metadata = self._metadata(batch.padded_size)


__all__ = ["GDNGraphCapture"]
