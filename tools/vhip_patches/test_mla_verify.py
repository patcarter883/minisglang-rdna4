"""Isolate mla_verify (multi-query spec verify) from the vLLM glue that drives it.

mla_decode is already validated in-serve (coherent + faster than TRITON_MLA), so it is the reference
here. The invariant: verifying query row j of a sequence with causal bound `kb` must equal decoding
that sequence with exactly `kb` cached tokens. If that holds, a degenerate spec serve is the glue's
index construction; if it fails, it is the kernel.
"""

import torch
import mla_hip

torch.manual_seed(0)
DEV = "cuda"
H, LAT, ROPE = 8, 512, 64
QK = LAT + ROPE
PAGE = 16


def build(B, ctx_len, max_blocks):
    nblocks = B * max_blocks
    cache = torch.randn(nblocks, PAGE, QK, dtype=torch.bfloat16, device=DEV)
    # give each sequence its own (deliberately non-contiguous) pages
    perm = torch.randperm(nblocks, device=DEV).to(torch.int32)
    block_table = perm.view(B, max_blocks).contiguous()
    return cache, block_table


def main():
    B, ctx, QLEN = 3, 91, 3          # ctx includes the QLEN new tokens, as vLLM's seq_lens does
    max_blocks = (ctx + PAGE - 1) // PAGE
    cache, block_table = build(B, ctx, max_blocks)
    scale = QK ** -0.5

    q = torch.randn(B * QLEN, H, QK, dtype=torch.bfloat16, device=DEV)
    seq_lens = torch.full((B,), ctx, dtype=torch.int32, device=DEV)

    # --- exactly the construction mla_vllm/decode.py::_build_decode does -------------------
    q_seq_idx = torch.arange(B, device=DEV, dtype=torch.int32).repeat_interleave(QLEN)
    within = torch.arange(QLEN, device=DEV, dtype=torch.int32).repeat(B)
    q_kbound = seq_lens.repeat_interleave(QLEN) - QLEN + within + 1
    print("q_seq_idx:", q_seq_idx.tolist())
    print("q_kbound :", q_kbound.tolist())

    got = mla_hip.mla_verify(q, cache, block_table, q_seq_idx, q_kbound, scale, 0)

    # --- reference: each row decoded on its own, with a context truncated to its bound ------
    worst = 0.0
    for b in range(B):
        for j in range(QLEN):
            row = b * QLEN + j
            kb = int(q_kbound[row])
            want = mla_hip.mla_decode(
                q[row : row + 1],                      # [1, H, QK]
                cache,
                block_table[b : b + 1],
                torch.tensor([kb], dtype=torch.int32, device=DEV),
                scale,
                0,
            )
            err = (got[row].float() - want[0].float()).abs().max().item()
            worst = max(worst, err)
            print(f"  seq {b} query {j} (kbound={kb:3d}): max|verify-decode| = {err:.5f}")
    print(f"\nworst = {worst:.5f}")
    assert worst < 0.02, "mla_verify disagrees with mla_decode -> KERNEL bug"
    print("mla_verify MATCHES mla_decode -> kernel + index construction are correct")


main()
