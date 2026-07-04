"""Autotune (block_m, BN) for the W8A16 fused MoE at the real ZAYA1-8B TiDAR shapes.

The fused-TiDAR decode calls fused_moe_w8a16 with M≈29 query tokens (1+B+B², seg B=4), top_k=1 over
E=16 experts, K=2048, inter=4096. block_m controls moe_align padding (each active expert's few tokens
pad up to block_m) so small block_m wins on tiny-M decode; BN is the N-tile width. This times the full
gemm1→SiLU→gemm2 over a config grid (warmup + median) and prints the winner. Current default = (64,128).

Run inside the combined ROCm image UNDER a 1-card lease:
  cd /engine && PYTHONPATH=/engine python moe_w8a16_wmma/moe_w8a16_autotune.py
"""
from __future__ import annotations

import time

import torch

from moe_w8a16_wmma import fused_moe_w8a16

DEV = "cuda"
torch.manual_seed(0)

# ZAYA1-8B TiDAR MoE dims.
E, K, INTER, TOP_K = 16, 2048, 4096, 1
M_LIST = [29]                     # fused seg B=4 query tokens (1 + B + B²)
BLOCK_MS = [16, 32, 64]
BNS = [32, 64, 128]
ITERS = 30


def make_fp8(E, OUT, IN):
    w_f8 = (torch.randn(E, OUT, IN, device=DEV) * (IN ** -0.5)).to(torch.float8_e4m3fn)
    scales = (torch.rand(E, OUT, device=DEV) * 0.5 + 0.5).to(torch.float32)
    return w_f8.view(torch.uint8).contiguous(), scales


def bench(M, block_m, BN, w13, w13s, w2, w2s):
    x = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)
    gate = torch.randn(M, E, device=DEV)
    topk_ids = gate.topk(TOP_K, dim=-1).indices.to(torch.int32)
    topk_w = torch.rand(M, TOP_K, device=DEV, dtype=torch.float32)
    # warmup
    for _ in range(5):
        fused_moe_w8a16(x, w13, w13s, w2, w2s, topk_w, topk_ids, block_m=block_m, BN=BN)
    torch.cuda.synchronize()
    ts = []
    for _ in range(ITERS):
        torch.cuda.synchronize(); t = time.perf_counter()
        fused_moe_w8a16(x, w13, w13s, w2, w2s, topk_w, topk_ids, block_m=block_m, BN=BN)
        torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
    ts.sort()
    return ts[len(ts) // 2] * 1e3  # median ms


def main():
    w13, w13s = make_fp8(E, 2 * INTER, K)     # gate_up: (E, 8192, 2048)
    w2, w2s = make_fp8(E, K, INTER)           # down:    (E, 2048, 4096)
    print(f"ZAYA MoE E={E} K={K} inter={INTER} top_k={TOP_K}; per-layer fused_moe_w8a16 median ms")
    best = {}
    for M in M_LIST:
        print(f"\n M={M}:")
        rows = []
        for bm in BLOCK_MS:
            for bn in BNS:
                try:
                    ms = bench(M, bm, bn, w13, w13s, w2, w2s)
                    rows.append((ms, bm, bn))
                    print(f"   block_m={bm:3d} BN={bn:3d}  {ms:.3f} ms")
                except Exception as e:  # noqa: BLE001
                    print(f"   block_m={bm:3d} BN={bn:3d}  ERR {e}")
        rows.sort()
        best[M] = rows[0]
        cur = next((r for r in rows if r[1] == 64 and r[2] == 128), None)
        print(f"  WINNER M={M}: block_m={rows[0][1]} BN={rows[0][2]} = {rows[0][0]:.3f} ms"
              + (f"  (default 64/128 = {cur[0]:.3f} ms, {cur[0]/rows[0][0]:.2f}× slower)" if cur else ""))
    print("\nBEST per M:", {m: (b[1], b[2], round(b[0], 3)) for m, b in best.items()})


if __name__ == "__main__":
    main()
