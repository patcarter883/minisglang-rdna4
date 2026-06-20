"""Phase 3c-2: GDN recurrent-state slot lifecycle (the spine of the phase).

minisgl's scheduler already splits work into homogeneous prefill / decode batches, so the
hard part of wiring GDN into the engine is NOT a subset-split — it is threading one fixed
recurrent-state slot through a sequence's whole life, across the cases that share a slot:

  * fresh prefill        — first time we see a uid: allocate + zero a slot.
  * chunked continuation — a long prompt prefilled over several passes. Each pass builds a
    NEW `Req`/`ChunkedReq` object (`PrefillAdder._add_one_req`), so the per-chunk object is
    NOT a stable identity. The slot must follow the PERSISTENT identity (the uid), and must
    NOT be re-zeroed on continuation (``cached_len > 0``) or the prior chunk's state is lost.
  * prefill -> decode    — the same uid moves from the prefill manager into the decode
    manager's running set; the slot carries over unchanged.
  * finish / abort       — free the slot. Overlap scheduling can try to free the same uid
    twice, so free MUST be idempotent (else two live sequences alias one freed slot).

This manager owns that lifecycle, keyed by uid (the only identity stable across all four).
``GDNStateCache`` underneath reserves slot 0 as the NULL block, so every allocated slot is
``>= 1`` (the load-bearing 3c constraint). It is deliberately scheduler-side and model-free
so it is unit-testable on CPU without a GPU boot (see ``tools/gdn_slots_test.py``).

★ Correctness precondition (enforced at the scheduler/engine level, 3c-2b): GDN-hybrid
models MUST run with the non-radix ("naive") prefix cache. GDN state is not prefix-cacheable;
a radix hit would yield ``cached_len > 0`` with no recurrent state behind it (silent garbage,
same failure class as slot-0). Under the naive cache, ``cached_len > 0`` happens ONLY for a
chunked continuation — exactly when ``has_initial_state`` must be True and the slot reused.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List

import torch

if TYPE_CHECKING:
    from minisgl.core import Batch, Req
    from minisgl.kvcache.gdn_state import GDNStateCache


class GDNSlotManager:
    def __init__(self, state_cache: GDNStateCache) -> None:
        self.state_cache = state_cache
        self.device = state_cache.device
        # uid -> the (>=1) GDN state slot held for that sequence's whole life.
        self._slot_of: Dict[int, int] = {}

    def state_indices(self, batch: Batch) -> torch.Tensor:
        """Return the int32 slot-per-sequence tensor for `batch`, in `batch.reqs` order.

        On a PREFILL batch, any uid seen for the first time is allocated a fresh, zeroed
        slot; uids already known (chunk continuations) keep their slot and state untouched.
        On a DECODE batch every uid is already known (it was prefilled), so this is a pure
        lookup — a missing uid is a wiring bug and surfaces as a KeyError, not silent garbage.
        """
        reqs: List[Req] = batch.reqs
        if batch.is_prefill:
            self._ensure_slots(reqs)
        idx = [self._slot_of[req.uid] for req in reqs]
        return torch.tensor(idx, dtype=torch.int32, device=self.device)

    def _ensure_slots(self, reqs: List[Req]) -> None:
        new_uids = [req.uid for req in reqs if req.uid not in self._slot_of]
        if not new_uids:
            return
        slots = self.state_cache.alloc_many(len(new_uids))  # int32 device tensor, all >= 1
        for uid, slot in zip(new_uids, slots.tolist()):
            self._slot_of[uid] = slot
        # Zero ONLY the freshly allocated slots (cached_len == 0 sequences). Continuations
        # are not in `new_uids`, so their accumulated state is preserved.
        self.state_cache.reset_slots(slots)

    def free(self, uid: int) -> None:
        """Release the slot for `uid`. Idempotent: a second free (overlap scheduling) is a
        no-op, so a freed slot can't be reused while a stale free is still in flight."""
        slot = self._slot_of.pop(uid, None)
        if slot is not None:
            self.state_cache.free([slot])

    @property
    def num_active(self) -> int:
        return len(self._slot_of)


__all__ = ["GDNSlotManager"]
