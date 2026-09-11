"""PLE (Qwen4-Exp n-gram) participation in cudagraph capture/replay.

WHY THIS EXISTS AT ALL, given that `ple/runtime.py` already says the decode path is capturable.

`GraphRunner._capture_graphs` builds a synthetic decode batch (`bs` copies of `dummy_req`) and calls
`model.forward()` twice — once eager as warmup, once inside `torch.cuda.graph`. It stages the attn
backend's metadata and, for a hybrid, GDN/CCA/CAM state — and NOTHING ELSE. `Qwen4ExpPLE.forward`
reads `get_global_ctx().ple.batch` and RAISES when it is None (deliberately: skipping the block would
drop the n-gram features silently). So on qwen4_exp the capture-time warmup forward raised before a
single graph was recorded, and cudagraph capture was structurally impossible for this model.

WHAT THE SPLIT IS. Nothing about the PLE block moves into the graph that was not already there; the
device arithmetic was always shape-static (index_select / cat / four scaled adds / index_copy_). The
gap was purely that capture had no way to STAGE a batch. So:

  capture   -> `prepare_for_capture` stages `bs` rows on the reserved NULL slot 0 with a one-token
               EOS context, which is exactly the convention `GDNGraphCapture` uses for its state
               slots and exactly what `Scheduler._stage_ple` writes for a cudagraph PADDING row. The
               gathered embeddings land in `PLEEmbeddingSource.embeddings[:bs]` and the slot ids in
               `PLERuntime._slot_idx[:bs]` — both static buffers, both at offset 0, so the pointers
               the graph bakes are the ones every later replay refreshes IN PLACE.
  after     -> `after_capture` DISCARDS the staged batch. It must not be committed (that would push
               EOS onto slot 0's n-gram history, which is meaningless but also not free of meaning:
               `commit_staged` is the once-per-forward invariant and a capture-time commit would put
               the prepare/commit counters permanently out of step). Clearing it also restores the
               property that a forward issued without a fresh `Scheduler._stage_ple` RAISES.
  replay    -> `prepare_for_replay` does NO staging work. The scheduler already staged this batch in
               `_finish_prepare`, over `padded_reqs`, i.e. including the cudagraph padding rows. What
               this does is ASSERT the two facts a replay silently depends on: that a batch was
               staged with exactly `padded_size` rows, and that it landed at the SAME ADDRESSES the
               capture baked. The second one is cheap and is the only thing standing between a
               refactor that allocates a fresh staging tensor and a graph that replays 48 layers
               against stale n-gram embeddings with no error anywhere.

The address assertion is deliberately an equality on `data_ptr()`, not a subclass/shape check: shape
and dtype would still match after such a refactor, and the output would still be finite text.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from minisgl.core import Batch

    from .runtime import PLERuntime


class PLEGraphCapture:
    def __init__(self, ple: "PLERuntime") -> None:
        self._ple = ple
        # The addresses the FIRST capture staged at, checked against every later capture and replay.
        self._embed_ptr: int | None = None
        self._slot_ptr: int | None = None
        self._conv_ptr = int(ple.state.conv_state.data_ptr())

    # -- capture ------------------------------------------------------------------------------

    def prepare_for_capture(self, batch: "Batch") -> None:
        # Discard the PREVIOUS captured size's staged batch before staging this one. `_capture_graphs`
        # walks the bucket list largest-first and never runs a scheduler step in between, so without
        # this the prepare/commit/discard ledger would silently gain one un-accounted `prepare` per
        # captured batch size and stop being an invariant anything can assert on.
        self._ple.discard_staged()
        bs = batch.padded_size
        eos = int(self._ple.state.eos_token_id)
        seed = np.full(1, eos, dtype=np.int64)
        # All rows on the reserved NULL slot 0: its conv window is zero and its history is EOS, so
        # nothing a live sequence owns is read or written. `is_decode=True` explicitly rather than
        # inferred, so a future change to the inference rule cannot capture the varlen branch.
        b = self._ple.prepare([0] * bs, [seed] * bs, is_decode=True)
        self._check_addresses(b, bs, "capture")

    def prepare_for_verify_capture(self, batch: "Batch", bs: int, qlen: int) -> None:
        """Stage the synthetic batch for a SPEC-VERIFY capture warmup.

        Separate from `prepare_for_capture` for one reason that is easy to get wrong: a verify
        forward carries `bs * qlen` tokens, not `bs`. The PLE block consumes one embedding per
        TOKEN and checks the two counts agree, so staging the decode shape here fails with
        "PLE staged N token embeddings but the forward carries M" — which is what it did.

        `PLERuntime.prepare` counts SEQUENCES, not tokens (it is capped at `max_seqs`), and a
        multi-token pass is one slot carrying a token ARRAY — the same shape `Scheduler._stage_ple`
        builds. So this stages `bs` sequences of `qlen` tokens each, giving bs*qlen embeddings.
        Passing bs*qlen single-token sequences instead trips "10 sequences > max_seqs 4".

        Values are irrelevant (a synthetic warmup on the reserved NULL slot 0, discarded by
        `after_capture`); only the token COUNT and the buffer addresses matter, because the
        addresses are what the captured graph bakes in."""
        self._ple.discard_staged()
        eos = int(self._ple.state.eos_token_id)
        seed = np.full(qlen, eos, dtype=np.int64)
        # is_decode=False: these rows carry qlen>1 tokens, which is the varlen shape, not the
        # single-token static-conv decode shape.
        b = self._ple.prepare([0] * bs, [seed] * bs, is_decode=(qlen == 1))
        self._check_addresses(b, bs, "verify-capture")

    def after_capture(self) -> None:
        """Drop the capture-staged batch. See the module docstring: NOT a commit."""
        self._ple.discard_staged()

    # -- replay -------------------------------------------------------------------------------

    def prepare_for_replay(self, batch: "Batch") -> None:
        b = self._ple.batch
        if b is None:
            raise RuntimeError(
                "qwen4_exp PLE: a captured decode graph is about to replay with no staged batch. "
                "`Scheduler._stage_ple` runs in `_finish_prepare`, before the forward, and "
                "`PLERuntime.commit_staged` clears it after — so this means the replay was reached "
                "on a path that skips one of them. The graph would silently read whatever n-gram "
                "embeddings the previous step left in the staging buffer."
            )
        if not b.is_decode:
            raise RuntimeError(
                f"qwen4_exp PLE: the staged batch is not a decode batch (seq_lens={b.seq_lens[:8]}), "
                f"but the decode graph was captured against the single-token static conv path. "
                f"`GraphRunner.can_use_cuda_graph` gates on `batch.is_decode`, so the two disagree."
            )
        if len(b.slots) != batch.padded_size:
            raise RuntimeError(
                f"qwen4_exp PLE: staged {len(b.slots)} rows but the graph replays "
                f"{batch.padded_size} (batch.size={batch.size}). `_stage_ple` must stage over "
                f"`padded_reqs` — the cudagraph padding rows each need their own NULL-slot row or "
                f"the layer reads one sequence's embedding for another's token."
            )
        self._check_addresses(b, batch.padded_size, "replay")

    # -- the pointer gate ---------------------------------------------------------------------

    def _check_addresses(self, b, bs: int, where: str) -> None:
        ep, sp = int(b.embeddings.data_ptr()), int(b.state_indices.data_ptr())
        cp = int(self._ple.state.conv_state.data_ptr())
        if self._embed_ptr is None:
            self._embed_ptr, self._slot_ptr = ep, sp
            return
        if (ep, sp, cp) != (self._embed_ptr, self._slot_ptr, self._conv_ptr):
            raise RuntimeError(
                f"qwen4_exp PLE [{where}, bs={bs}]: the buffers the captured graph reads MOVED "
                f"(embeddings {self._embed_ptr:#x} -> {ep:#x}, slot index {self._slot_ptr:#x} -> "
                f"{sp:#x}, conv state {self._conv_ptr:#x} -> {cp:#x}). A captured graph holds these "
                f"addresses as constants; replaying it now reads freed or unrelated memory and "
                f"produces finite, plausible, WRONG text. `PLEEmbeddingSource` allocates its host, "
                f"fp32 landing and model-dtype buffers exactly once and `stage_rows` must return a "
                f"leading VIEW of that one allocation — not a fresh tensor."
            )


__all__ = ["PLEGraphCapture"]
