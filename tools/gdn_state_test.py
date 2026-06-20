#!/usr/bin/env python
"""Phase 3a unit test: the GDN recurrent state-slot allocator. CPU-only (no GPU/lease).
Validates buffer shapes, alloc/free/reuse, slot isolation, reset, and exhaustion."""
from __future__ import annotations

import torch

from minisgl.kvcache.gdn_state import GDNStateCache


def main() -> None:
    # 35B GDN dims (TP=1): conv_dim = 128*16*2 + 128*32 = 8192; ssm (32,128,128).
    c = GDNStateCache(
        num_gdn_layers=30,
        num_slots=8,
        conv_dim=8192,
        conv_kernel=4,
        num_v_heads=32,
        head_v_dim=128,
        head_k_dim=128,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )
    assert tuple(c.conv_state.shape) == (30, 8, 8192, 3), c.conv_state.shape
    assert tuple(c.ssm_state.shape) == (30, 8, 32, 128, 128), c.ssm_state.shape
    # Slot 0 is the reserved NULL block: 8 buffer slots -> 7 allocatable.
    assert c.num_free == 7

    s = c.alloc_many(3)
    assert c.num_free == 4 and len(set(s.tolist())) == 3
    # ★ slot 0 (NULL_BLOCK_ID) must NEVER be handed out (the 3c load-bearing constraint).
    assert (s != 0).all(), f"allocated slot 0 (NULL_BLOCK_ID): {s.tolist()}"

    # slot isolation: writing one slot leaves the others zero
    c.ssm(0)[s[0]] = 1.0
    assert (c.ssm(0)[s[1]] == 0).all()
    assert c.conv(5)[s[2]].shape == (8192, 3)

    # free -> reuse
    c.free(s)
    assert c.num_free == 7
    # exhaust ALL allocatable slots and confirm slot 0 is never among them
    s2 = c.alloc_many(7)
    assert c.num_free == 0
    assert (s2 != 0).all(), f"allocated slot 0 (NULL_BLOCK_ID): {s2.tolist()}"

    # reset clears a reused slot's prior state
    c.reset_slots(s2[:1])
    assert (c.ssm(0)[s2[0]] == 0).all() and (c.conv(0)[s2[0]] == 0).all()

    # exhaustion raises
    try:
        c.alloc_many(1)
        raise AssertionError("expected RuntimeError on exhaustion")
    except RuntimeError:
        pass

    mem_mb = (c.conv_state.numel() * 2 + c.ssm_state.numel() * 2) / 1e6
    print(f"GDN state cache OK: shapes/alloc/free/reuse/reset/exhaustion + slot-0 "
          f"reservation all pass ({mem_mb:.0f} MB for 30 layers x 8 slots)")
    print("PASS")


if __name__ == "__main__":
    main()
