"""Per-sequence PLE state: the dilated short-conv window, and the n-gram token history.

The PLE block on decoder layer 1 carries TWO pieces of recurrent state per sequence, and the
reference keeps both in the same cache entry as that layer's GDN conv state, at different
`state_idx` (`modeling_qwen4_exp.py`: `state_idx=1` for the short conv, `state_idx=2` for the token
history). This class mirrors that: it is indexed by the SAME slot id the `GDNStateCache` hands out,
so a sequence has one slot number for its whole recurrent footprint and nothing has to be kept in
sync.

  * `conv_state`  DEVICE  [num_slots, wide, state_len]  — the last `state_len = (k - 1) * dilation`
    = `(4 - 1) * 3` = 9 columns of the normed gated value the depthwise conv slides over. Zero for a
    fresh slot, which is exactly what the reference's left zero-pad produces.
  * `token_history`  HOST  [num_slots, context_len] int64 — the last `ngram_size - 1` = 2 token ids.
    Host, because the hash that consumes it is host-side (it addresses an mmap'd NVMe table), and
    int64 because the multipliers are.

★ Slot 0 is the NULL slot, same convention as `GDNStateCache` — never handed to a real sequence, and
kept zero/EOS so a padded cudagraph row reads something defined instead of another request's state.

GRAPH CAPTURE. `conv_state` is allocated ONCE with a stable device address and is only ever read via
`index_select(0, idx)` and written via `index_copy_(0, idx, ...)` with a device index tensor — both
capturable. `token_history` is host state, updated OUTSIDE the graph on the same schedule as
`Batch.input_ids` itself, so it never appears in a captured region.
"""
from __future__ import annotations

import numpy as np
import torch


class PLEStateCache:
    def __init__(
        self,
        *,
        num_slots: int,
        wide: int,
        state_len: int,
        context_len: int,
        eos_token_id: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        if num_slots < 2:
            raise ValueError("num_slots must leave room for the reserved NULL slot 0 plus one seq")
        self.num_slots = int(num_slots)
        self.wide = int(wide)
        self.state_len = int(state_len)
        self.context_len = int(context_len)
        self.eos_token_id = int(eos_token_id)
        self.conv_state = torch.zeros(
            (self.num_slots, self.wide, self.state_len), dtype=dtype, device=device
        )
        # EOS, not zero: the reference seeds `previous_context` with `eos_token_id`, and token 0 is a
        # real token in this vocab. Seeding zeros would hash the first two tokens of every request
        # against a real n-gram instead of the "start of segment" one — a silent quality loss.
        self.token_history = np.full(
            (self.num_slots, self.context_len), self.eos_token_id, dtype=np.int64
        )

    @property
    def nbytes_device(self) -> int:
        return self.conv_state.numel() * self.conv_state.element_size()

    def reset_slot(self, slot: int) -> None:
        """Hand a slot to a new sequence. MUST be called at prefill: a recycled slot still holds the
        previous request's conv window and token history, which would leak that request's lexical
        context into this one's first few tokens."""
        if not 1 <= slot < self.num_slots:
            raise IndexError(f"slot {slot} outside [1, {self.num_slots}) (0 is the NULL slot)")
        self.conv_state[slot].zero_()
        self.token_history[slot] = self.eos_token_id

    def history(self, slot: int) -> np.ndarray:
        return self.token_history[slot]

    def push_tokens(self, slot: int, tokens: np.ndarray) -> None:
        """Record this pass's tokens so the next pass can reconstruct its n-gram context.

        Keeps the LAST `context_len` of `history ++ tokens` — equivalent to the reference's
        `update_conv_state(..., state_idx=2, conv_kernel_size=context_len)` including its
        EOS-left-pad for a first chunk shorter than `context_len` (our history already starts as
        EOS, so the concatenation does that for free).
        """
        t = np.asarray(tokens, dtype=np.int64).reshape(-1)
        if t.size >= self.context_len:
            self.token_history[slot] = t[-self.context_len :]
        else:
            keep = self.context_len - t.size
            self.token_history[slot, :keep] = self.token_history[slot, self.context_len - keep :]
            self.token_history[slot, keep:] = t

    def full_history(self, slot: int, tokens: np.ndarray) -> np.ndarray:
        """`previous_context ++ tokens` — what `Qwen4ExpNGramHasher` hashes. Does NOT mutate."""
        return np.concatenate(
            [self.token_history[slot], np.asarray(tokens, dtype=np.int64).reshape(-1)]
        )


__all__ = ["PLEStateCache"]
