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


class CCAVerifyGraphCapture:
    """Static-buffer CCA metadata for cudagraph capture of the spec-VERIFY forward (v2 S2).

    Like ``CCAGraphCapture`` but for ``qlen = K+1`` query tokens/seq WITH per-token verify-state
    capture. Holds, as PERSISTENT buffers the captured graph's pointers reference across replays:
      * ``state_indices`` (conv slot/seq) + ``query_start_loc`` (= arange*(K+1)) — refreshed in place;
      * per-CCA-layer conv/prev SCRATCH ``[Q, max_bs, C, TP]`` / ``[Q, max_bs, hidden]`` that
        ``capture_cca_verify_state`` writes IN-PLACE (metadata.py: it writes into a pre-bound buffer
        rather than allocating fresh — the fix that makes it capturable);
      * fixed host ``seg_lens = [K+1]*bs`` (uniform verify → the per-seq segment walk is static, so the
        capture's Python loop has fixed iteration/shape).
    The scheduler gathers the accepted-prefix state (index ``accepted-1``) from the scratch EAGERLY
    after ``g.replay()`` (``install_verify_state``) — same as the eager path. Only capturable when every
    req has exactly ``K`` drafts (``GraphRunner.can_use_verify_graph``); partial-K steps stay eager.
    """

    def __init__(self, device: torch.device, max_bs: int, num_draft: int,
                 cca_layer_ids, conv_dim: int, conv_width: int, hidden: int) -> None:
        Q = num_draft + 1
        self._Q = Q
        self._state_indices = torch.zeros(max_bs, dtype=torch.int32, device=device)
        self._cu = torch.arange(max_bs + 1, dtype=torch.int32, device=device) * Q
        # has_initial_state (verify is a multi-query PREFILL-path forward, build_cca_metadata is_prefill
        # =True): True ⟺ cached_len>0 (always so for a verify continuation). Static bool buf refreshed
        # per replay; the captured cca_prefill_qk's `torch.where(has_init, init_conv, 0)` reads it live.
        self._has_init = torch.zeros(max_bs, dtype=torch.bool, device=device)
        self._seg_full = [Q] * max_bs  # host, fixed (uniform K+1 verify)
        self._conv = {int(lid): torch.zeros(Q, max_bs, conv_dim, conv_width, dtype=torch.float32,
                                            device=device) for lid in cca_layer_ids}
        self._prev = {int(lid): torch.zeros(Q, max_bs, hidden, dtype=torch.float32, device=device)
                      for lid in cca_layer_ids}

    def _metadata(self, bs: int) -> CCAMetadata:
        return CCAMetadata(
            is_prefill=True,  # verify is the multi-query varlen path (matches build_cca_metadata)
            num_seqs=bs,
            query_start_loc=self._cu[: bs + 1],
            state_indices=self._state_indices[:bs],
            has_initial_state=self._has_init[:bs],
            capture_verify_state=True,
            verify_max_qlen=self._Q,
            conv_scratch={lid: buf[:, :bs] for lid, buf in self._conv.items()},
            prev_scratch={lid: buf[:, :bs] for lid, buf in self._prev.items()},
            seg_lens=self._seg_full[:bs],
        )

    def prepare_verify_for_capture(self, batch: "Batch") -> None:
        # Dummy capture batch: all rows -> NULL slot 0 (a valid reserved row no live seq owns);
        # cached_len 0 -> has_initial_state False (the where() records with the static tensor either way).
        self._state_indices[: batch.padded_size].fill_(0)
        self._has_init[: batch.padded_size].fill_(False)
        batch.cca_metadata = self._metadata(batch.padded_size)

    def prepare_verify_for_replay(self, batch: "Batch") -> None:
        # Reuse the scheduler-built real slots + has_initial_state (batch.cca_metadata); pad NULL/False.
        md = batch.cca_metadata
        real = md.state_indices
        n = real.numel()
        self._state_indices[:n].copy_(real)
        self._state_indices[n : batch.padded_size].fill_(0)
        if md.has_initial_state is not None:
            self._has_init[:n].copy_(md.has_initial_state)
        else:
            self._has_init[:n].fill_(False)
        self._has_init[n : batch.padded_size].fill_(False)
        batch.cca_metadata = self._metadata(batch.padded_size)


__all__ = ["CCAGraphCapture", "CCAVerifyGraphCapture"]
