"""Equivalence parity for native moe_hip.moe_align vs vLLM moe_align_block_size (GPU).

moe_align groups routed tokens per expert and pads each run to block_size with sentinel = numel.
Intra-expert token ORDER is arbitrary (downstream gather/scatter-reduce is per original token), so we
check EQUIVALENCE, not bit-identity: same P (sorted_ids length), same num_tokens_post_pad, same
per-block expert_ids over the used range, and the same valid-token SET per expert — both vs each other
and vs the ground-truth routing.

Run inside vllm22-w4a8:combined under a 1-card lease (see attn_decode_parity.py for the docker line).
"""
from __future__ import annotations

import torch

import op  # noqa: F401  loads moe_hip_C + registers torch.ops.moe_hip.*
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size

DEV = "cuda"
torch.manual_seed(0)


def _groups(sorted_ids, expert_ids, ntp, numel, bs):
    """expert -> sorted list of valid tokens, derived from a (sorted_ids, expert_ids) alignment."""
    si = sorted_ids.tolist()
    ei = expert_ids.tolist()
    nblk = int(ntp.item()) // bs
    d: dict[int, list[int]] = {}
    for b in range(nblk):
        e = ei[b]
        for r in range(bs):
            tok = si[b * bs + r]
            if tok < numel:
                d.setdefault(e, []).append(tok)
    return {e: sorted(v) for e, v in d.items()}


def _ground_truth(topk_ids, numel):
    flat = topk_ids.reshape(-1).tolist()
    d: dict[int, list[int]] = {}
    for t in range(numel):
        d.setdefault(flat[t], []).append(t)
    return {e: sorted(v) for e, v in d.items()}


def check(name, M, top_k, E, bs=16) -> bool:
    topk_ids = torch.randint(0, E, (M, top_k), device=DEV, dtype=torch.int32)
    numel = M * top_k

    si, ei, ntp = torch.ops.moe_hip.moe_align(topk_ids, E, bs)
    rsi, rei, rntp = moe_align_block_size(topk_ids, bs, E, None, pad_sorted_ids=True)

    gt = _ground_truth(topk_ids, numel)
    mine = _groups(si, ei, ntp, numel, bs)
    ref = _groups(rsi, rei, rntp, numel, bs)

    ok = True
    ok &= si.shape == rsi.shape                          # same P
    ok &= int(ntp.item()) == int(rntp.item())            # same total padded
    nblk = int(ntp.item()) // bs
    ok &= torch.equal(ei[:nblk], rei[:nblk])             # same per-block expert (ascending) over used range
    ok &= (mine == gt) and (ref == gt)                   # both reproduce the routing exactly
    # sentinel padding: every slot >= ntp, and every in-run padding slot, must be the numel sentinel
    ok &= bool((si[int(ntp.item()):] == numel).all().item())
    P = si.shape[0]
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:30s} P={P} ntp={int(ntp.item())} "
          f"(ref P={rsi.shape[0]} ntp={int(rntp.item())}) experts_hit={len(mine)}")
    return ok


def main() -> None:
    print("=== moe_hip.moe_align equivalence vs vLLM moe_align_block_size ===")
    ok = True
    ok &= check("decode M1 tk8 E128", 1, 8, 128)
    ok &= check("decode M2 tk8 E128", 2, 8, 128)
    ok &= check("M4 tk4 E64", 4, 4, 64)
    ok &= check("M8 tk2 E8 (collisions)", 8, 2, 8)
    ok &= check("prefill M512 tk8 E128", 512, 8, 128)
    ok &= check("prefill M2048 tk8 E128", 2048, 8, 128)
    ok &= check("M128 tk2 E8 bs32", 128, 2, 8, bs=32)
    ok &= check("M1 tk1 E128 (single)", 1, 1, 128)
    ok &= check("M16 tk8 E256", 16, 8, 256)
    print("RESULT:", "ALL PASS" if ok else "FAILURES PRESENT")


if __name__ == "__main__":
    main()
