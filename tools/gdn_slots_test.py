#!/usr/bin/env python
"""Phase 3c-2 unit test: the GDN slot lifecycle (`GDNSlotManager`). CPU-only (no GPU/lease).

Exercises the four cases that share one recurrent-state slot — the spine of 3c:
  (1) fresh multi-seq prefill   -> distinct, zeroed, >=1 slots
  (2) chunked continuation      -> same slot, state NOT re-zeroed
  (3) prefill -> decode handoff -> same slot (pure lookup)
  (4) idempotent free + reuse   -> double-free is a no-op; a reused slot IS re-zeroed
plus the load-bearing invariant: slot 0 (NULL_BLOCK_ID) is never handed out.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch

from minisgl.kvcache.gdn_state import GDNStateCache
from minisgl.scheduler.gdn_slots import GDNSlotManager


def fake_batch(uids, *, prefill):
    # GDNSlotManager only reads batch.reqs / batch.is_prefill/.is_decode and req.uid.
    reqs = [SimpleNamespace(uid=u) for u in uids]
    return SimpleNamespace(reqs=reqs, is_prefill=prefill, is_decode=not prefill)


def main() -> None:
    cache = GDNStateCache(
        num_gdn_layers=2, num_slots=6, conv_dim=8192, conv_kernel=4,
        num_v_heads=32, head_v_dim=128, head_k_dim=128,
        dtype=torch.bfloat16, device=torch.device("cpu"),
    )
    mgr = GDNSlotManager(cache)

    # (1) fresh multi-seq prefill -> 3 distinct, >=1, zeroed slots
    idx = mgr.state_indices(fake_batch([10, 20, 30], prefill=True))
    assert idx.dtype == torch.int32
    assert len(set(idx.tolist())) == 3 and (idx >= 1).all(), idx
    assert mgr.num_active == 3
    slot10, slot20, slot30 = (mgr._slot_of[u] for u in (10, 20, 30))
    assert (cache.ssm(0)[slot10] == 0).all()  # freshly zeroed

    # write a sentinel into uid 10's state (simulates prefill having advanced the SSM state)
    cache.ssm(0)[slot10] = 7.0
    cache.ssm(1)[slot10] = 7.0

    # (2) chunked continuation: uid 10 reappears in a later prefill pass -> SAME slot,
    #     and the accumulated state must NOT be re-zeroed.
    idx2 = mgr.state_indices(fake_batch([10], prefill=True))
    assert idx2.tolist() == [slot10], (idx2, slot10)
    assert mgr.num_active == 3  # no new slot allocated
    assert (cache.ssm(0)[slot10] == 7.0).all(), "continuation must not re-zero the slot"

    # (3) prefill -> decode handoff: same uids in a decode batch -> pure lookup, same slots
    didx = mgr.state_indices(fake_batch([10, 20, 30], prefill=False))
    assert didx.tolist() == [slot10, slot20, slot30]
    assert (cache.ssm(0)[slot10] == 7.0).all()  # untouched by decode lookup

    # decode lookup of an unknown uid is a wiring bug -> surfaces loudly (no silent garbage)
    try:
        mgr.state_indices(fake_batch([999], prefill=False))
        raise AssertionError("expected KeyError for an un-prefilled uid in a decode batch")
    except KeyError:
        pass

    # (4a) idempotent free: freeing uid 20 twice must not double-release its slot.
    #      5 usable slots, 3 allocated -> 2 free; freeing one -> 3 free.
    mgr.free(20)
    assert mgr.num_active == 2 and cache.num_free == 3, cache.num_free
    mgr.free(20)  # no-op: must NOT push slot 20 back a second time
    assert mgr.num_active == 2 and cache.num_free == 3, cache.num_free

    # (4b) a reused slot IS re-zeroed. Free uid 10 (sentinel slot), then a NEW uid must
    #      reuse it cleared.
    mgr.free(10)
    assert (slot10 not in mgr._slot_of.values())
    new_idx = mgr.state_indices(fake_batch([40], prefill=True))
    reused = new_idx.tolist()[0]
    if reused == slot10:  # LIFO makes this the expected reuse
        assert (cache.ssm(0)[slot10] == 0).all(), "reused slot must be re-zeroed"
        assert (cache.ssm(1)[slot10] == 0).all()
    assert (new_idx >= 1).all()

    print("GDN slot manager OK: fresh/continuation/handoff/idempotent-free/reuse-zero + "
          "slot-0 reservation all pass")
    print("PASS")


if __name__ == "__main__":
    main()
