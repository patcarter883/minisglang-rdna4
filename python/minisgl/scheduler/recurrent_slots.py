"""Shared recurrent-state slot lifecycle for hybrid models (GDN, CCA/Zaya, ...).

minisgl's scheduler already splits work into homogeneous prefill / decode batches, so the hard
part of wiring a recurrent-state model into the engine is NOT a subset-split — it is threading ONE
fixed recurrent-state slot through a sequence's whole life, across the cases that share a slot:

  * fresh prefill        — first time we see a uid: allocate + zero a slot.
  * chunked continuation — a long prompt prefilled over several passes. Each pass builds a NEW
    `Req`/`ChunkedReq` object, so the per-chunk object is NOT a stable identity. The slot must
    follow the PERSISTENT identity (the uid), and must NOT be re-zeroed on continuation
    (``cached_len > 0``) or the prior chunk's recurrent state is lost.
  * prefill -> decode    — the same uid moves from the prefill manager into the decode manager's
    running set; the slot carries over unchanged.
  * finish / abort       — free the slot. Overlap scheduling can free the same uid twice, so free
    MUST be idempotent (else two live sequences alias one freed slot).

This base owns that lifecycle, keyed by uid (the only identity stable across all four). It is
duck-typed on the underlying state cache, which only needs ``device``, ``alloc_many(n)`` (returns
an int32 device tensor of slots, all ``>= 1`` — slot 0 is the reserved NULL block),
``reset_slots(slots)`` and ``free([slots])``. It is deliberately scheduler-side and model-free so
it is unit-testable on CPU without a GPU boot.

★ Correctness precondition (enforced at the scheduler/engine level): recurrent-state hybrid models
MUST run with the non-radix ("naive") prefix cache. Recurrent state is not prefix-cacheable; a
radix hit would yield ``cached_len > 0`` with no recurrent state behind it (silent garbage). Under
the naive cache, ``cached_len > 0`` happens ONLY for a chunked continuation — exactly when
``has_initial_state`` must be True and the slot reused.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Protocol

import torch

if TYPE_CHECKING:
    from minisgl.core import Batch, Req


class _RecurrentStateCache(Protocol):
    device: torch.device

    def alloc_many(self, n: int) -> torch.Tensor: ...
    def reset_slots(self, slots: torch.Tensor) -> None: ...
    def free(self, slots: List[int]) -> None: ...


class RecurrentSlotManager:
    def __init__(self, state_cache: _RecurrentStateCache) -> None:
        self.state_cache = state_cache
        self.device = state_cache.device
        # uid -> the (>=1) recurrent state slot held for that sequence's whole life.
        self._slot_of: Dict[int, int] = {}

    def state_indices(self, batch: "Batch") -> torch.Tensor:
        """Return the int32 slot-per-sequence tensor for `batch`, in `batch.reqs` order.

        On a PREFILL batch, any uid seen for the first time is allocated a fresh, zeroed slot; uids
        already known (chunk continuations) keep their slot and state untouched. On a DECODE batch
        every uid is already known (it was prefilled), so this is a pure lookup — a missing uid is a
        wiring bug and surfaces as a KeyError, not silent garbage.
        """
        reqs: List[Req] = batch.reqs
        if batch.is_prefill:
            self._ensure_slots(reqs)
        idx = [self._slot_of[req.uid] for req in reqs]
        return torch.tensor(idx, dtype=torch.int32, device=self.device)

    def _ensure_slots(self, reqs: List["Req"]) -> None:
        new_uids = [req.uid for req in reqs if req.uid not in self._slot_of]
        if not new_uids:
            return
        slots = self.state_cache.alloc_many(len(new_uids))  # int32 device tensor, all >= 1
        for uid, slot in zip(new_uids, slots.tolist()):
            self._slot_of[uid] = slot
        # Zero ONLY the freshly allocated slots (cached_len == 0 sequences). Continuations are not
        # in `new_uids`, so their accumulated recurrent state is preserved.
        self.state_cache.reset_slots(slots)

    def slot_for(self, uid: int) -> int | None:
        """The (>=1) recurrent slot currently held for `uid`, or None if the uid is not active.
        Used by recurrent-radix prefix caching to clone/restore a sequence's state at its slot."""
        return self._slot_of.get(uid)

    def free(self, uid: int) -> None:
        """Release the slot for `uid`. Idempotent: a second free (overlap scheduling) is a no-op,
        so a freed slot can't be reused while a stale free is still in flight."""
        slot = self._slot_of.pop(uid, None)
        if slot is not None:
            self.state_cache.free([slot])

    @property
    def num_active(self) -> int:
        return len(self._slot_of)


__all__ = ["RecurrentSlotManager"]
