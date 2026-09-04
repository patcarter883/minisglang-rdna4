"""Host side of the PLE block: hash -> NVMe row gather -> ONE H2D copy into a static buffer.

WHAT THIS OWNS
--------------
Everything between "the scheduler knows this batch's token ids" and "the PLE layer has a device
tensor of n-gram embeddings":

    per sequence:  previous_context ++ tokens  --hash-->  (S, 16) global row ids
    whole batch:   (T*16,) row ids  --ShardedRowTable.gather_into-->  pinned fp32 (T, 2560)
                   --one non_blocking H2D-->  static device fp32 (T, 2560)  --cast-->  bf16

WHAT IT DELIBERATELY DOES NOT OWN
---------------------------------
The table. `weights/row_table.py` already mmaps the 51.2 GB `model-plefp8-*.safetensors` set, decodes
F8_E4M3, applies the single global scale, addresses the 128 shards, and picks prefetch-vs-threadpool
per request size (measured: 609 us for a 16-row decode step with prefetch; the pool wins above ~256
rows). None of that is re-implemented here — this module opens it and calls `gather_into`.

WHY A STATIC BUFFER, NOT A FRESH TENSOR PER STEP
------------------------------------------------
The gather is a HOST operation (page-cache / NVMe reads), so its result has to cross the PCIe link
before the decoder can use it. Under cudagraph capture the decoder's input addresses are frozen, so
the embeddings must land at a FIXED device address that the replay reads — exactly like
`Batch.input_ids`. Allocating per step would make the PLE layer uncapturable, which by this repo's
rules means unfinished. So: one pinned host buffer, one device buffer, both allocated once, and
`gather_into` writes straight into the pinned one (the dequant's output IS the staging buffer — no
intermediate array).

The staging buffer is fp32 because that is what `ShardedRowTable.gather` produces and writing it
directly is what avoids a second host pass; the cast to the model dtype happens on the DEVICE, where
it is a trivially bandwidth-bound elementwise kernel. Cost of the fp32 link traffic: 10 KB/token, so
~60 KB for a 6-sequence decode step (unmeasurable) and ~21 MB for a 2048-token prefill chunk
(~1.5 ms on this box's Gen4 x8 link to card 1) against a prefill that takes far longer than that.
"""
from __future__ import annotations

import os
from typing import List, Sequence

import numpy as np
import torch

from minisgl.weights.row_table import NgramHeads, ShardedRowTable, open_qwen4exp_ngram_table

from .hashing import Qwen4ExpNGramHasher
from .state import PLEStateCache

#: Colon-separated `model-plefp8-*.safetensors` paths (the n-gram table shards).
ENV_PLE_FILES = "MINISGL_PLE_FILES"
#: Colon-separated extra shards holding `ngram_heads_offsets` / `ngram_heads_vocab_sizes` — they live
#: in a bf16 shard, NOT in the plefp8 set, so they must be named separately.
ENV_PLE_META_FILES = "MINISGL_PLE_META_FILES"


def _env_paths(name: str) -> List[str]:
    raw = os.environ.get(name, "")
    return [p for p in raw.split(":") if p]


class PLEEmbeddingSource:
    """Opens the n-gram table and turns a batch's tokens into a device embedding tensor."""

    def __init__(
        self,
        *,
        ple_files: Sequence[str],
        meta_files: Sequence[str] = (),
        hasher: Qwen4ExpNGramHasher,
        embed_dim: int,
        max_tokens: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        workers: int = 16,
        scale_override: float | None = None,
        _table: ShardedRowTable | None = None,
        _heads: NgramHeads | None = None,
    ) -> None:
        if _table is not None:
            table, heads = _table, _heads
        else:
            table, heads = open_qwen4exp_ngram_table(
                ple_files, meta_files, workers=workers, scale_override=scale_override
            )
        if heads is None:
            raise KeyError(
                f"the n-gram head metadata (`ngram_heads_offsets` / `ngram_heads_vocab_sizes`) was "
                f"not in the given files. It lives in a bf16 shard, not in the plefp8 set — name it "
                f"in {ENV_PLE_META_FILES}. Without it the row ids cannot be formed at all (there is "
                f"no safe default: guessing the band layout reads another head's embeddings)."
            )
        if heads.n_heads != hasher.ngram_heads:
            raise ValueError(
                f"checkpoint ships {heads.n_heads} n-gram head bands but the config implies "
                f"(ngram_size-1)*heads_per_ngram = {hasher.ngram_heads}"
            )
        if table.row_elems * heads.n_heads != embed_dim:
            raise ValueError(
                f"n-gram row width {table.row_elems} x {heads.n_heads} heads = "
                f"{table.row_elems * heads.n_heads} != ple_embed_dim {embed_dim}. The PLE embedding "
                f"is the per-head rows CONCATENATED, so these must agree exactly."
            )
        self.table = table
        self.heads = heads
        self.hasher = hasher
        self.embed_dim = int(embed_dim)
        self.max_tokens = int(max_tokens)
        self.dtype = dtype
        self.device = device

        # One pinned host buffer + one device buffer, allocated once. `gather_into` writes into the
        # host one through a (max_tokens*n_heads, row_elems) view, which is the same memory as the
        # (max_tokens, embed_dim) view the copy reads — the head concatenation is free.
        pin = device.type == "cuda" and torch.cuda.is_available()
        self._host = torch.empty(
            (self.max_tokens, self.embed_dim), dtype=torch.float32, pin_memory=pin
        )
        self._host_rows = self._host.numpy().reshape(self.max_tokens * heads.n_heads, table.row_elems)
        self._dev_f32 = torch.empty(
            (self.max_tokens, self.embed_dim), dtype=torch.float32, device=device
        )
        #: The tensor the layer reads. Static address, cast target of `_dev_f32`.
        self.embeddings = torch.zeros(
            (self.max_tokens, self.embed_dim), dtype=dtype, device=device
        )

    # -- row ids -----------------------------------------------------------

    def batch_row_ids(
        self, state: PLEStateCache, slots: Sequence[int], token_lists: Sequence[np.ndarray]
    ) -> np.ndarray:
        """(T, n_heads) global row ids for a whole batch, sequences concatenated in `slots` order.

        Does NOT advance the token history — call `advance` after the forward, so a batch that is
        aborted mid-flight cannot leave a slot's history one chunk ahead of its conv state.
        """
        if len(slots) != len(token_lists):
            raise ValueError(f"{len(slots)} slots vs {len(token_lists)} token lists")
        out = [
            self.hasher.row_ids(
                self.heads, state.full_history(int(slot), toks), int(np.size(toks))
            )
            for slot, toks in zip(slots, token_lists)
        ]
        return (
            np.concatenate(out, axis=0)
            if out
            else np.empty((0, self.heads.n_heads), dtype=np.int64)
        )

    def advance(
        self, state: PLEStateCache, slots: Sequence[int], token_lists: Sequence[np.ndarray]
    ) -> None:
        """Commit this pass's tokens to each slot's n-gram history."""
        for slot, toks in zip(slots, token_lists):
            state.push_tokens(int(slot), toks)

    # -- staging -----------------------------------------------------------

    def stage_rows(self, row_ids: np.ndarray) -> torch.Tensor:
        """(T, n_heads) row ids -> the first T rows of `self.embeddings`, filled.

        One gather, one H2D. Returns a VIEW of the static buffer, so the caller must not keep it
        across steps.
        """
        ids = np.asarray(row_ids, dtype=np.int64)
        n_tokens = ids.shape[0]
        if n_tokens > self.max_tokens:
            raise ValueError(
                f"PLE staging buffer holds {self.max_tokens} tokens, batch has {n_tokens}. Size it "
                f"from the same max-token budget the scheduler uses (chunked-prefill size, or "
                f"max_running_req for decode)."
            )
        if n_tokens == 0:
            return self.embeddings[:0]
        flat = ids.reshape(-1)
        self.table.gather_into(flat, self._host_rows[: flat.size])
        self._dev_f32[:n_tokens].copy_(self._host[:n_tokens], non_blocking=True)
        self.embeddings[:n_tokens].copy_(self._dev_f32[:n_tokens])
        return self.embeddings[:n_tokens]

    def stage_batch(
        self, state: PLEStateCache, slots: Sequence[int], token_lists: Sequence[np.ndarray]
    ) -> torch.Tensor:
        """`batch_row_ids` + `stage_rows`. Does not advance the history — see `batch_row_ids`."""
        return self.stage_rows(self.batch_row_ids(state, slots, token_lists))

    def close(self) -> None:
        self.table.close()

    # -- construction from the environment ---------------------------------

    @classmethod
    def from_env(
        cls,
        *,
        hasher: Qwen4ExpNGramHasher,
        embed_dim: int,
        max_tokens: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        workers: int = 16,
    ) -> "PLEEmbeddingSource":
        files = _env_paths(ENV_PLE_FILES)
        if not files:
            raise RuntimeError(
                f"{ENV_PLE_FILES} is unset. Qwen3.8-Flash-Next's 51.2 GB n-gram table is NOT loaded "
                f"with the rest of the weights — it stays NVMe-resident and is mmap'd from the "
                f"`model-plefp8-*.safetensors` set, which must be named explicitly (colon-separated) "
                f"along with {ENV_PLE_META_FILES} for the bf16 shard holding the head metadata."
            )
        return cls(
            ple_files=files,
            meta_files=_env_paths(ENV_PLE_META_FILES),
            hasher=hasher,
            embed_dim=embed_dim,
            max_tokens=max_tokens,
            device=device,
            dtype=dtype,
            workers=workers,
        )


__all__ = ["ENV_PLE_FILES", "ENV_PLE_META_FILES", "PLEEmbeddingSource"]
