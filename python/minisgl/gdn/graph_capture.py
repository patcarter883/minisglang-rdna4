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


def _replay_verify_available(qmax: int) -> bool:
    """Will the GDN layers take the ReplaySSM verify (ring append) for a window of `qmax` tokens?

    Same three conditions the layer checks (ring present, op built, window fits the ring), asked at
    capture time so the capturer knows whether to allocate the per-token SSM scratch at all."""
    try:
        import gdn_hip as gdn

        return (hasattr(gdn, "gdn_verify_replay")
                and hasattr(gdn, "gdn_decode_conv_gated_replay")
                and int(qmax) <= int(gdn.REPLAY_RING_LEN))
    except Exception:
        return False


class GDNVerifyGraphCapture:
    """Static-buffer GDN metadata for cudagraph capture of the spec-VERIFY forward (the GDN analog of
    ``CCAVerifyGraphCapture``).

    Like ``GDNGraphCapture`` but for ``qlen = K+1`` query tokens/seq WITH per-token verify-state
    capture. Holds, as PERSISTENT buffers the captured graph's pointers reference across replays:
      * ``state_indices`` (conv/ssm slot/seq) + ``query_start_loc`` (= arange*(K+1)) — refreshed in
        place per replay;
      * ``has_initial_state`` — bool/seq; verify is the varlen (prefill-style) recurrent path so the
        kernels read it live (True ⟺ cached_len>0, always so for a verify continuation). Static bool
        buffer refreshed per replay;
      * per-GDN-layer conv/ssm SCRATCH ``[Q, max_bs, C, W-1]`` (fp32) / ``[Q, max_bs, HV, V, K]``
        (ssm dtype) that the captured verify forward writes IN-PLACE. The GDN verify kernels return a
        FRESH scratch tensor (unlike CCA's torch reconstruction), so the model bridge (qwen3_5.py)
        COPIES the kernel output into these pre-bound buffers when they are present — that copy is the
        in-graph write that keeps the pointers valid.
    The scheduler gathers the accepted-prefix state (index ``accepted-1``) from the scratch EAGERLY
    after ``g.replay()`` (``GDNStateCache.install_verify_state``) — same as the eager path. Only
    capturable when every req has exactly ``K`` drafts (``GraphRunner.can_use_verify_graph``);
    partial-K steps stay eager.
    """

    def __init__(self, device: torch.device, max_bs: int, num_draft: int,
                 gdn_layer_ids, conv_dim: int, conv_width: int,
                 num_v_heads: int, head_v_dim: int, head_k_dim: int,
                 ssm_dtype: torch.dtype) -> None:
        # `num_draft` may be a LIST of widths (adaptive verify width, spec/width.py). The per-layer
        # conv/ssm scratch is the DOMINANT per-width allocation on a GDN hybrid, so it is allocated
        # ONCE at the widest Q and every narrower width takes the leading `[:Q]` slice — a prefix view
        # of the outermost dim, exactly like the existing `[:, :bs]` batch slice. Only `query_start_loc`
        # (arange*Q) is genuinely per-width, and it is [max_bs+1] ints.
        widths = [num_draft] if isinstance(num_draft, int) else sorted(set(int(w) for w in num_draft))
        Qmax = max(widths) + 1
        self._Qmax = Qmax
        self._Q = Qmax
        self._state_indices = torch.zeros(max_bs, dtype=torch.int32, device=device)
        self._cu_by_q = {
            w + 1: torch.arange(max_bs + 1, dtype=torch.int32, device=device) * (w + 1)
            for w in widths
        }
        self._cu = self._cu_by_q[Qmax]
        self._has_init = torch.zeros(max_bs, dtype=torch.bool, device=device)
        self._conv = {int(lid): torch.zeros(Qmax, max_bs, conv_dim, conv_width, dtype=torch.float32,
                                            device=device) for lid in gdn_layer_ids}
        # SSM scratch is allocated ONLY for the materialising verify. Under ReplaySSM the draft
        # window lives in the ring and the accept is a cursor rewind, so there are no per-token
        # states to stage — and this is the DOMINANT capture allocation on a GDN hybrid
        # (Qmax * max_bs * HV * V * K per layer: 10 MB/layer at Q=5, bs=4, 16 v-heads, 128x128 bf16,
        # i.e. ~300 MB across 30 layers). Skipping it is the VRAM half of the replay-verify win, and
        # VRAM is what makes spec decode fail to boot on this box.
        self._replay_verify = _replay_verify_available(Qmax)
        self._ssm = {} if self._replay_verify else {
            int(lid): torch.zeros(Qmax, max_bs, num_v_heads, head_v_dim, head_k_dim,
                                  dtype=ssm_dtype, device=device) for lid in gdn_layer_ids}

    def set_width(self, qlen: int) -> None:
        """Select the captured width (`qlen` query rows/seq) whose metadata the next capture/replay
        should use. Called by GraphRunner right before prepare_verify_for_capture / _for_replay."""
        self._Q = qlen
        self._cu = self._cu_by_q[qlen]

    def _metadata(self, bs: int) -> GDNMetadata:
        Q = self._Q
        return GDNMetadata(
            is_prefill=True,  # verify is the multi-query varlen path (matches build_gdn_metadata)
            num_seqs=bs,
            query_start_loc=self._cu[: bs + 1],
            state_indices=self._state_indices[:bs],
            has_initial_state=self._has_init[:bs],
            capture_verify_state=True,
            verify_max_qlen=Q,
            conv_scratch={lid: buf[:Q, :bs] for lid, buf in self._conv.items()},
            # empty under ReplaySSM — the scheduler reads its emptiness as "commit by rollback"
            ssm_scratch={lid: buf[:Q, :bs] for lid, buf in self._ssm.items()},
        )

    def prepare_verify_for_capture(self, batch: "Batch") -> None:
        # Dummy capture batch: all rows -> NULL slot 0 (a valid reserved row no live seq owns);
        # cached_len 0 -> has_initial_state False.
        self._state_indices[: batch.padded_size].fill_(0)
        self._has_init[: batch.padded_size].fill_(False)
        batch.gdn_metadata = self._metadata(batch.padded_size)

    def prepare_verify_for_replay(self, batch: "Batch") -> None:
        # Reuse the scheduler-built real slots + has_initial_state (batch.gdn_metadata); pad NULL/False.
        md = batch.gdn_metadata
        real = md.state_indices
        n = real.numel()
        self._state_indices[:n].copy_(real)
        self._state_indices[n : batch.padded_size].fill_(0)
        if md.has_initial_state is not None:
            self._has_init[:n].copy_(md.has_initial_state)
        else:
            self._has_init[:n].fill_(False)
        self._has_init[n : batch.padded_size].fill_(False)
        batch.gdn_metadata = self._metadata(batch.padded_size)


__all__ = ["GDNGraphCapture", "GDNVerifyGraphCapture"]
