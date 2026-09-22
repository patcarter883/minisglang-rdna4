"""Per-batch PLE plumbing: what the scheduler stages before a forward, and what the layer reads.

`Qwen4ExpPLE.forward` takes ONE argument (the wide hidden stream), like every other block in this
engine. Everything else it needs — the gathered n-gram embeddings, which state slot each sequence
owns, how the flat token buffer splits into sequences — arrives through `Context.ple`, the same way
GDN metadata and the KV pool do.

The split of work is the point:

    HOST, BEFORE the forward (never inside a captured graph)
        PLERuntime.prepare(slots, token_lists, is_decode)
          -> hash the recent tokens          (ple/hashing.py, VERIFIED against the reference)
          -> gather 16 rows/token from NVMe  (weights/row_table.py)
          -> ONE H2D into a static buffer    (ple/source.py)
          -> write the slot ids into a static device index buffer

    DEVICE, INSIDE the forward (capturable)
        Qwen4ExpPLE.forward(hidden)
          -> two GEMMs, three grouped norms, a gate, a dilated depthwise conv against
             PLEStateCache.conv_state, all at fixed addresses

The decode path is deliberately shape-static: `prepare` is told the PADDED sequence count, the
index buffer is a fixed-size slice, and the conv is an `index_select` / `index_copy_` pair — so a
cudagraph capture of a decode step replays without re-recording anything. Padded rows point at the
NULL slot 0, whose state is zero and whose output is therefore discarded harmlessly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np
import torch

from .source import PLEEmbeddingSource
from .state import PLEStateCache


@dataclass
class PLEBatch:
    """What the PLE layer reads for the current forward. Rebuilt (in place) every step."""

    embeddings: torch.Tensor  # (T, ple_embed_dim) view of the static staging buffer
    state_indices: torch.Tensor  # (N,) int64 device view of the static slot-index buffer
    #: The same slot ids on the HOST. Carried explicitly so the varlen prefill path can index the
    #: conv state per sequence without a `.tolist()` on the device tensor — that would be a
    #: device->host SYNC inside a model forward, which is exactly the kind of stall this engine
    #: spends its time removing.
    slots: List[int]
    seq_lens: List[int]  # tokens per sequence, sums to T
    is_decode: bool  # every seq_len == 1 -> the static single-token conv path
    #: The token ids that were staged, per sequence. Kept so `commit_staged` can advance exactly
    #: what `prepare` hashed, rather than have the caller re-derive it from the requests AFTER the
    #: forward — by which point `complete_one()` has moved `cached_len` to `device_len` and the
    #: `[cached_len:device_len]` span that defined this pass no longer exists.
    #: Defaults empty so a test can hand-build a batch to drive `Qwen4ExpPLE.compute` (which never
    #: reads it) without inventing token ids; `commit_staged` REFUSES an empty one rather than
    #: quietly advancing nothing.
    tokens: List[np.ndarray] = field(default_factory=list)
    #: Spec-decode VERIFY pass: this batch's tokens are `[confirmed, d0 .. d_{K-1}]` and the drafts
    #: are NOT committed yet, so neither the n-gram history nor the conv window may be advanced over
    #: them until the accept decision is known. `commit_staged` parks such a batch instead of
    #: advancing it, and `commit_verified` finishes the job with the accepted lengths.
    defer_commit: bool = False
    #: slot -> the `[state | chunk]` window (wide, state_len + n) the verify forward built, stashed
    #: by `Qwen4ExpPLE._short_conv` so the accepted-prefix conv state can be recovered exactly
    #: (`PLEStateCache.install_verify_conv`) rather than approximated or recomputed.
    conv_windows: dict = field(default_factory=dict)


class PLERuntime:
    """Owns the n-gram source and the per-sequence state, and stages one batch at a time."""

    def __init__(
        self,
        *,
        source: PLEEmbeddingSource,
        state: PLEStateCache,
        max_seqs: int,
        device: torch.device,
    ) -> None:
        self.source = source
        self.state = state
        self.max_seqs = int(max_seqs)
        #: A deferred spec-verify batch awaiting its accept decision (see commit_staged/commit_verified).
        self._pending: "PLEBatch | None" = None
        # Static index buffer: `index_select`/`index_copy_` inside a captured graph must read a
        # tensor at a fixed address. Long, because that is what index_copy_ wants everywhere.
        self._slot_idx = torch.zeros(self.max_seqs, dtype=torch.long, device=device)
        self.batch: PLEBatch | None = None
        #: THE ONCE-PER-FORWARD LEDGER. `prepare` (scheduler `_finish_prepare`) and `commit_staged`
        #: (scheduler `_forward`, right after the forward returns) sit far apart in the loop, and the
        #: pairing between them is what keeps every slot's n-gram history exactly one pass behind its
        #: conv state. Under cudagraph capture that pairing acquires a THIRD participant —
        #: `PLEGraphCapture` stages one batch per captured batch size and DISCARDS it instead of
        #: committing — so "commit_staged ran exactly once per forward" stopped being checkable by
        #: reading the two call sites. These counters make it arithmetic; at any quiescent point:
        #:     prepares == commits + discards      and      commit_noops == 0
        #: A non-zero `commit_noops` is a forward that ran with nothing staged (or a batch committed
        #: twice) — the failure that freezes the n-gram context with no error anywhere.
        self.prepares = 0
        #: Set by `prepare_begin`, consumed by `prepare_finish`. Not None == reads in flight.
        self._begun = None
        self.commits = 0
        self.discards = 0
        self.commit_noops = 0

    # -- per-batch ---------------------------------------------------------

    def prepare(
        self,
        slots: Sequence[int],
        token_lists: Sequence[np.ndarray],
        *,
        is_decode: bool | None = None,
        defer_commit: bool = False,
    ) -> PLEBatch:
        """Stage the host side of one forward. Call BEFORE the model forward, never inside capture.

        `slots` is the per-sequence PLE/GDN state slot (0 for a cudagraph padding row), and
        `token_lists` the token ids this pass feeds each sequence. Does NOT advance the token
        history — call `commit` after the forward has actually run, so an aborted batch cannot
        leave a slot's n-gram history one chunk ahead of its conv state.
        """
        self.prepare_begin(slots, token_lists, is_decode=is_decode, defer_commit=defer_commit)
        return self.prepare_finish()

    def prepare_begin(
        self,
        slots: Sequence[int],
        token_lists: Sequence[np.ndarray],
        *,
        is_decode: bool | None = None,
        defer_commit: bool = False,
    ) -> None:
        """Submit the n-gram row reads and return WITHOUT waiting for them.

        Split out of `prepare` so the caller can issue the NVMe reads as soon as the slots exist and
        then keep preparing the batch while they land -- `os.pread` releases the GIL, so rows in
        flight cost the calling thread nothing. Pair with `prepare_finish`; `prepare` is the two
        back to back and is what every caller that does not care about overlap should use.

        Nothing here advances the token history, exactly as in `prepare`: an aborted batch between
        begin and finish leaves no state behind except the in-flight gather, which `prepare_finish`
        (or the next `prepare_begin`, which refuses) surfaces.
        """
        if len(slots) > self.max_seqs:
            raise ValueError(f"{len(slots)} sequences > max_seqs {self.max_seqs}")
        seq_lens = [int(np.size(t)) for t in token_lists]
        if is_decode is None:
            is_decode = all(n == 1 for n in seq_lens)
        self.source.begin_rows(self.source.batch_row_ids(self.state, slots, token_lists))
        self._begun = (list(slots), list(token_lists), seq_lens, bool(is_decode),
                       bool(defer_commit))

    def prepare_finish(self) -> PLEBatch:
        """Wait for the reads issued by `prepare_begin` and publish the batch."""
        if self._begun is None:
            raise RuntimeError("prepare_finish called without a preceding prepare_begin")
        slots, token_lists, seq_lens, is_decode, defer_commit = self._begun
        self._begun = None
        embeddings = self.source.finish_rows()
        n = len(slots)
        self._slot_idx[:n].copy_(
            torch.as_tensor(np.asarray(slots, dtype=np.int64)), non_blocking=True
        )
        self.batch = PLEBatch(
            embeddings=embeddings,
            state_indices=self._slot_idx[:n],
            slots=[int(s) for s in slots],
            seq_lens=seq_lens,
            is_decode=bool(is_decode),
            tokens=[np.asarray(t, dtype=np.int64).reshape(-1) for t in token_lists],
            defer_commit=bool(defer_commit),
        )
        self.prepares += 1
        return self.batch

    def ledger_fault(self) -> str | None:
        """The ONCE-PER-FORWARD invariant, evaluated. None when healthy, else why it is not.

        The invariant is stated in the ledger's own comment above and was, until this method existed,
        NEVER EVALUATED ANYWHERE: `prepares`, `commits`, `discards` and `commit_noops` were all
        incremented and nothing in the engine read them. The module documents the exact arithmetic
        for "the failure that freezes the n-gram context with no error anywhere" and then left the
        arithmetic undone — the same shape as this repo's other silent instruments (a metric that
        exports a flat zero because one plumbing hop is missing; a kernel gap closed with no caller).

        Why it matters more here than a lost counter would: a frozen n-gram history does not crash
        and does not corrupt the WEIGHTS. It corrupts the model's representation of ITS OWN CONTEXT,
        so the reply stays fluent and confidently wrong — parametric knowledge intact, retrieval
        broken. That is the exact signature under investigation (`@docs/ui-plan` ->
        `docs/22909-74486-01`, Hermes session 5d26424a4be3).

        Checked at a QUIESCENT point only (no batch staged, nothing in flight); mid-forward the two
        halves are legitimately unequal.
        """
        if self.batch is not None or self._begun is not None:
            return None  # mid-forward: the pairing is open by construction
        if self.commit_noops:
            return (f"commit_noops={self.commit_noops}: a forward ran with nothing staged (or a "
                    f"batch was committed twice). Every slot in such a forward re-hashed a stale "
                    f"token context, so its n-gram features are frozen")
        if self.prepares != self.commits + self.discards:
            return (f"prepares={self.prepares} != commits={self.commits} + "
                    f"discards={self.discards}: a staged batch was neither committed nor discarded, "
                    f"which leaves that slot's n-gram history behind its conv state")
        return None

    def ledger(self) -> dict:
        """The raw counters, for logging and metric export."""
        return {"prepares": self.prepares, "commits": self.commits,
                "discards": self.discards, "commit_noops": self.commit_noops,
                "pending": self._pending is not None}

    def commit(self, slots: Sequence[int], token_lists: Sequence[np.ndarray]) -> None:
        """Advance each slot's n-gram token history. Call AFTER a successful forward."""
        self.source.advance(self.state, slots, token_lists)

    def commit_staged(self) -> None:
        """`commit` the batch this runtime last staged, exactly once.

        The scheduler's pairing is `prepare` in `_finish_prepare` -> forward -> here, and the two
        sites are far apart in the loop. Re-deriving the token spans at the second site is not
        merely redundant, it is WRONG: `Engine.forward_batch` calls `complete_one()` on every row,
        which sets `cached_len = device_len`, so the `[cached_len:device_len]` slice that defined
        the pass is empty by then and the history would silently stop advancing (every step would
        re-hash the same stale 2-token context and the n-gram features would freeze). Clearing
        `self.batch` here is what makes a second call a no-op AND what makes the PLE layer refuse
        loudly if a forward is ever issued without a fresh `prepare`.
        """
        b = self.batch
        if b is None:
            self.commit_noops += 1
            return
        self.batch = None
        self.commits += 1
        if b.defer_commit:
            # Spec VERIFY: park it. The history must not advance over drafts that the accept loop
            # is about to reject, and the conv window the forward already wrote is the "after every
            # draft" one. `commit_verified` (scheduler, next to the GDN accepted-prefix install)
            # finishes both halves with the accepted lengths. The ledger still counts this as the
            # commit for this forward, so `prepares == commits` stays the once-per-forward invariant.
            if self._pending is not None:
                raise RuntimeError(
                    "PLE commit_staged: a deferred verify batch was already parked and never "
                    "resolved. `commit_verified` must run after every deferred forward — a second "
                    "one means the accept path was skipped, which would silently freeze the n-gram "
                    "history at the earlier pass."
                )
            self._pending = b
            return
        if len(b.tokens) != len(b.slots):
            raise RuntimeError(
                f"PLE commit_staged: {len(b.slots)} slots but {len(b.tokens)} token lists. The "
                f"staged batch did not come from `prepare`, so there is nothing to advance the "
                f"n-gram history WITH — committing it would leave every slot's context frozen at "
                f"its EOS seed for the rest of the sequence, silently."
            )
        self.source.advance(self.state, b.slots, b.tokens)

    def commit_verified(self, keep_by_slot: "dict[int, int]") -> None:
        """Resolve the parked spec-verify batch against the accepted lengths.

        `keep_by_slot` maps each real slot to how many tokens of that pass were COMMITTED — the
        confirmed token plus the accepted drafts, i.e. `len(keep)` at the accept site, which is the
        same quantity GDN installs as `t_index = len(keep) - 1`.

        Both halves move together, which is the whole point: the n-gram history advances over
        exactly the committed tokens, and the conv window is rolled back to the state after those
        same tokens. Advancing one without the other is the "history one chunk ahead of its conv
        state" hazard that `prepare`/`commit` were split apart to prevent.

        A slot missing from the map keeps the full-pass advance the forward already did. That is the
        right fallback for a row this step did not adjudicate (a cudagraph padding row on the NULL
        slot), and it is what the pre-fix behaviour was for everything.
        """
        b = self._pending
        if b is None:
            return
        self._pending = None
        slots: "list[int]" = []
        toks: "list[np.ndarray]" = []
        for slot, tok in zip(b.slots, b.tokens):
            if slot == 0:            # reserved NULL slot: padding rows own no sequence
                continue
            keep = keep_by_slot.get(slot)
            if keep is None:
                keep = int(tok.size)  # not adjudicated -> behave exactly as before
            keep = max(0, min(int(keep), int(tok.size)))
            win = b.conv_windows.get(slot)
            if win is not None and keep != int(tok.size):
                self.state.install_verify_conv(slot, win, keep)
            if keep:
                slots.append(slot)
                toks.append(tok[:keep])
        b.conv_windows.clear()
        if slots:
            self.source.advance(self.state, slots, toks)

    def discard_staged(self) -> None:
        """Drop the staged batch WITHOUT advancing any history. Cudagraph capture only.

        `PLEGraphCapture.prepare_for_capture` stages a synthetic all-NULL-slot batch so the
        capture-time warmup forward has something to read; committing it would push an EOS onto slot
        0's history (meaningless) and, worse, would make `prepares == commits` while one of those
        commits corresponded to no forward at all — destroying the one invariant that says the n-gram
        history is advancing in lockstep with the conv state. Counted separately for that reason.
        """
        if self.batch is None:
            return
        self.batch = None
        self.discards += 1

    def counters(self) -> dict:
        """The prepare/commit/discard ledger. See `__init__`; a harness asserts on this."""
        return {
            "ple_prepares": self.prepares,
            "ple_commits": self.commits,
            "ple_discards": self.discards,
            "ple_commit_noops": self.commit_noops,
        }

    def reset_slot(self, slot: int) -> None:
        self.state.reset_slot(slot)

    def close(self) -> None:
        self.source.close()


# ======================= engine-side construction =======================
# One builder, used by BOTH the Engine and the bring-up harnesses. Not because it saves lines: the
# four things below (slot count, staging width, n-gram-table location, hasher seed) each have a
# silent failure mode, and having two call sites derive them independently is exactly how a harness
# comes to "pass" against a configuration the serve never runs.

#: The file-name marker of the 51.2 GB n-gram table shards. The weight loader skips these by the
#: same marker (`weight.py::_QWEN4EXP_PLE_FILE_MARK`) — the table is mmap'd, never streamed into
#: device memory — so the two must agree or one of them is looking at the wrong set of files.
PLE_TABLE_GLOB = "model-plefp8-*.safetensors"


def ple_device_bytes(config, *, num_slots: int, max_tokens: int, dtype: torch.dtype) -> int:
    """DEVICE bytes a `PLERuntime` for `config` will allocate, from config alone.

    Config-only on purpose, and for the same reason `_recurrent_state_bytes` is: the engine has to
    answer this while it is SIZING the KV pool, i.e. before the runtime exists. The pinned HOST
    staging buffer is deliberately not counted — it is host RAM and does not come out of the device
    budget.
    """
    embed = int(config.ple_embed_dim or 0)
    if not config.ple_layer_ids or not embed:
        return 0
    # source.py: `_dev_f32` (fp32, the H2D landing zone) + `embeddings` (model dtype, what the layer
    # reads). Both are max_tokens x ple_embed_dim and both are allocated once.
    staging = max_tokens * embed * (4 + dtype.itemsize)
    # state.py: conv_state [num_slots, hc*hidden, (k-1)*ngram_size]. token_history is host numpy.
    state_len = (int(config.ple_conv_kernel_size) - 1) * int(config.ngram_size)
    conv = num_slots * int(config.hc_hidden_size) * state_len * dtype.itemsize
    return staging + conv


def build_ple_runtime(
    *,
    model,
    config,
    model_path: str,
    num_slots: int,
    max_tokens: int,
    device: torch.device,
    dtype: torch.dtype,
) -> "PLERuntime":
    """Build the PLE runtime for a loaded qwen4_exp model.

    `num_slots` MUST be the same slot count the `GDNStateCache` was given: `PLEStateCache` is
    indexed by the slot id the GDN slot manager hands out (see `ple/state.py`), so a smaller PLE
    cache would IndexError on the highest-numbered sequence and a larger one silently wastes
    memory. `max_tokens` must cover the largest single forward (the chunked-prefill budget).

    The n-gram table is located from `MINISGL_PLE_FILES` / `MINISGL_PLE_META_FILES` when they are
    set, otherwise from `model_path` itself. The env path exists because the table is routinely
    staged on a different filesystem from the body (it is 51.2 GB of NVMe-resident weights that are
    never loaded), not as a feature flag: with neither present this RAISES rather than running a
    model with the n-gram features silently zeroed.
    """
    import glob as _glob

    from .hashing import Qwen4ExpNGramHasher
    from .source import ENV_PLE_FILES, ENV_PLE_META_FILES, PLEEmbeddingSource, _env_paths

    block = model.ple_block()
    files = _env_paths(ENV_PLE_FILES) or sorted(_glob.glob(f"{model_path}/{PLE_TABLE_GLOB}"))
    if not files:
        raise FileNotFoundError(
            f"qwen4_exp declares a PLE layer at decoder index {sorted(config.ple_layer_ids)} but no "
            f"n-gram table shards were found: {ENV_PLE_FILES} is unset and {model_path!r} holds no "
            f"{PLE_TABLE_GLOB}. That table is 51.2 GB and stays NVMe-resident, so it is NOT part of "
            f"the weight stream and its absence cannot be detected by the loader — name it in "
            f"{ENV_PLE_FILES} (colon-separated), with the bf16 shard carrying "
            f"`ngram_heads_offsets`/`ngram_heads_vocab_sizes` in {ENV_PLE_META_FILES}."
        )
    # The head-band metadata lives in a bf16 shard, not in the plefp8 set (see source.py). Default to
    # the model dir's own bf16 shards, which is where it is for an intact checkpoint.
    meta = _env_paths(ENV_PLE_META_FILES) or sorted(
        _glob.glob(f"{model_path}/model-bf16-*.safetensors")
    )
    hasher = Qwen4ExpNGramHasher.from_checkpoint_multipliers(
        ngram_size=config.ngram_size,
        heads_per_ngram=config.heads_per_ngram,
        eos_token_id=config.ngram_eos_token_id,
        # The CHECKPOINT's multipliers, not a re-derivation from the seed. They agree
        # (tests/qwen4exp_ple_hash_test.py asserts the equality against a checked-in fixture), and
        # taking them from the loaded parameter is what keeps them agreeing for a future checkpoint
        # that was generated with a different seed than its config declares.
        checkpoint_multipliers=block.ple_embedding.layer_multipliers.cpu().numpy(),
        vocab_size=config.vocab_size,
        ple_layer_index=0,
        seed=config.ngram_seed,
    )
    source = PLEEmbeddingSource(
        ple_files=files,
        meta_files=meta,
        hasher=hasher,
        embed_dim=config.ple_embed_dim,
        max_tokens=max_tokens,
        device=device,
        dtype=dtype,
    )
    state = block.make_state_cache(
        num_slots=num_slots,
        eos_token_id=config.ngram_eos_token_id,
        device=device,
        dtype=dtype,
    )
    return PLERuntime(source=source, state=state, max_seqs=num_slots, device=device)


__all__ = ["PLEBatch", "PLERuntime", "PLE_TABLE_GLOB", "build_ple_runtime", "ple_device_bytes"]
