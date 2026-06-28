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

from .recurrent_slots import RecurrentSlotManager


class GDNSlotManager(RecurrentSlotManager):
    """The GDN (conv_state + ssm_state) recurrent-slot lifecycle.

    Identical lifecycle to every recurrent hybrid — the per-model difference is the buffer set the
    underlying ``GDNStateCache`` declares, which the slot manager never touches. The four lifecycle
    cases plus the idempotent-free and naive-cache preconditions live in ``RecurrentSlotManager``.
    Still unit-testable on CPU without a GPU boot (see ``tools/gdn_slots_test.py``).
    """


__all__ = ["GDNSlotManager"]
