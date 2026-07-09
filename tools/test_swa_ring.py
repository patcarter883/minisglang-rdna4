"""Pure-Python (no torch/GPU) proof of the SWA ring addressing used by
RDNA4Backend._build_swa_metadata: for a sequence of length S and window W, the ring
(slot = pos % W) must, after storing positions 0..S-1, hold EXACTLY the last min(S,W)
positions, and the decode read block [base .. base+min(S,W)) with cache_seqlen=min(S,W)
must select exactly the windowed-attention key set { p : S-1-p < W }.

This validates item 2's addressing math on CPU; the GPU harness (tools/swa_gpu_validate.py)
confirms the HIP kernel computes the attention over that key set. Run: python tools/test_swa_ring.py
"""
from __future__ import annotations

import sys

FAILS = []


def ring_survivors(S: int, W: int, table_idx: int = 0):
    """Replicate the ring store: positions 0..S-1 written to slot table_idx*W + pos%W; later
    positions overwrite. Return {slot: surviving_position}."""
    base = table_idx * W
    slot_to_pos = {}
    for p in range(S):
        slot_to_pos[base + (p % W)] = p  # later p overwrites
    return slot_to_pos


def read_block(S: int, W: int, table_idx: int = 0):
    """Replicate _build_swa_metadata's read block for a seq of length S: the first min(S,W) slots
    of the request's window block, with cache_seqlen = min(S,W)."""
    base = table_idx * W
    cnt = min(S, W)
    return [base + s for s in range(cnt)], cnt


def expected_window_positions(S: int, W: int):
    """Windowed-causal key set for the newest query (position S-1): { p : (S-1)-p < W }."""
    return set(range(max(0, S - W), S))


def check(name, ok, extra=""):
    print(f"  [{'ok' if ok else 'FAIL'}] {name} {extra}")
    if not ok:
        FAILS.append(name)


def main() -> int:
    W = 512
    for S in [1, 100, 511, 512, 513, 700, 1024, 5000]:
        surv = ring_survivors(S, W)
        block, cnt = read_block(S, W)
        surv_positions = set(surv.values())
        want = expected_window_positions(S, W)
        # 1. the ring holds exactly the last min(S,W) positions
        c1 = surv_positions == want
        # 2. the read block covers exactly the occupied slots (block == keys of surv), same count
        c2 = set(block) == set(surv.keys()) and cnt == len(want)
        check(f"S={S:5d} W={W}: ring=last-{min(S,W)} positions", c1,
              f"(|surv|={len(surv_positions)}, want=[{min(want)}..{max(want)}])")
        check(f"S={S:5d} W={W}: read block == occupied slots, ctx_len={cnt}", c2)

    # multi-sequence: each request's block is disjoint (base = table_idx*W)
    b0, _ = read_block(600, W, table_idx=0)
    b1, _ = read_block(600, W, table_idx=1)
    check("multi-seq ring blocks disjoint", set(b0).isdisjoint(set(b1)),
          f"(seq0 max={max(b0)}, seq1 min={min(b1)})")

    print()
    if FAILS:
        print(f"FAILED: {FAILS}")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
