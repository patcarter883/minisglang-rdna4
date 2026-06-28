"""ZAYA CCA recurrent-state slot lifecycle — the CCA analog of `gdn_slots.py`.

minisgl's scheduler splits work into homogeneous prefill / decode batches, so wiring CCA into the
engine is NOT a subset-split — it is threading ONE fixed recurrent-state slot (conv_states +
prev_hs) through a sequence's whole life, across the cases that share a slot:

  * fresh prefill        — first time we see a uid: allocate + zero a slot.
  * chunked continuation — a long prompt prefilled over several passes. Each pass builds a NEW
    `Req`/`ChunkedReq` object, so the per-chunk object is NOT a stable identity. The slot must
    follow the PERSISTENT identity (the uid), and must NOT be re-zeroed on continuation
    (``cached_len > 0``) or the prior chunk's conv/prev_hs state is lost.
  * prefill -> decode    — the same uid moves from the prefill manager into the decode manager's
    running set; the slot carries over unchanged.
  * finish / abort       — free the slot. Overlap scheduling can free the same uid twice, so free
    MUST be idempotent (else two live sequences alias one freed slot).

This manager owns that lifecycle, keyed by uid (the only identity stable across all four).
``CCAStateCache`` underneath reserves slot 0 as the NULL block, so every allocated slot is ``>= 1``
(the load-bearing constraint the decode kernel relies on for is_pad). It is deliberately
scheduler-side and model-free.

★ Correctness precondition (enforced at the scheduler level): CCA-hybrid models MUST run with the
non-radix ("naive") prefix cache. CCA recurrent state is not prefix-cacheable; a radix hit would
yield ``cached_len > 0`` with no recurrent state behind it (silent garbage). Under the naive cache,
``cached_len > 0`` happens ONLY for a chunked continuation — exactly when ``has_initial_state`` must
be True and the slot reused.
"""

from __future__ import annotations

from .recurrent_slots import RecurrentSlotManager


class CCASlotManager(RecurrentSlotManager):
    """The CCA (conv_states + prev_hs) recurrent-slot lifecycle.

    Behaviour is identical to GDN's — the only per-model difference is the buffer set the underlying
    ``CCAStateCache`` declares, which the slot manager never touches. All four lifecycle cases (fresh
    prefill / chunked continuation / prefill->decode / finish-abort) plus the idempotent-free and
    naive-cache preconditions live in ``RecurrentSlotManager``.
    """


__all__ = ["CCASlotManager"]
