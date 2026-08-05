#!/usr/bin/env python
"""What (block_m x BN) should the GROUPED W4A8 MoE GEMM run? Sweep it, and gate the chooser on it.

WHY THIS EXISTS. The dense tile became a shape-derived cost-model choice (tile_select.h). The MoE
tile did not: `block_m` came from `_moe_block_m`, which keys on ROWS-PER-EXPERT ALONE -- no N, no CU
count, no LDS/occupancy term -- and `BN` was an env-defaulted constant 64 (or 128), which is the
grouped half of exactly the bug the dense `bm=256, bn=128` hard-wire was. BN sets ceil(N/BN), i.e.
the dispatch granularity on the output axis, so a constant BN cannot know whether the grid covers
the machine: at MoE gemm2 (N = hidden/TP) BN=64 and BN=256 differ by 4x in workgroup count.

WHAT IT MEASURES, in order (the order matters -- see CONTROL):

  CONTROL. baseline vs a SECOND recording of the SAME build, per shape. The gemm2 scatter epilogue
     is fp32 atomicAdd and is order-nondeterministic AGAINST ITSELF, so a bit-identity gate on it
     would fail for reasons that have nothing to do with the tile. Run the control FIRST and print
     it, so a later delta is read against a measured floor rather than against 0.

  A. SURFACE. every (block_m, BN) the kernel can express, graph-replay timed with the expert weight
     stacks rotated past the 64 MB MALL BY BYTE COUNT, over the real routed shapes of the shipped
     MoE checkpoints x the M ladder the engine dispatches.

  B. PROVENANCE. what the chooser (fp8_wmma.moe_tile_choose, host-side, no GPU) picks for each cell,
     what `_moe_block_m` + BN=64 picked before it, and what the swept oracle was.

  C. BIT-IDENTITY / TOLERANCE. gemm1 (non-scatter, deterministic store epilogue) must be
     max|delta| = 0 across tiles -- the tile is a launch parameter, not a numerics one. gemm2's
     scatter is gated by TOLERANCE under an ATOMIC. key, against the control floor from above.

    gpu-lease -n 1 --timeout 10800 -- \
      TILE_TOOL=tools/w4a8_moe_tile_surface.py bash tools/w4a8_dense_tile_surface_run.sh \
        --out /engine/_moe_tile_surface.txt --csv /engine/_moe_tile_surface.csv
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import torch

DEV = torch.device("cuda:0")
ROT_TARGET_BYTES = 160 << 20
ROT_MAX_COPIES = 24
ROT_MAX_VRAM = 6 << 30
CELL_BUDGET_US = 20_000.0
LDS_MAX = 65536

BM_SET = (16, 32, 64, 128)
BN_SET = (16, 32, 64, 96, 128, 192, 256)

# (name, E, top_k, hidden, inter). The grouped GEMMs are then
#   gemm1: N = 2*inter, K = hidden      gemm2: N = hidden, K = inter
# Every shipped W4A8 MoE checkpoint, at the TP the engine serves it at.
SHAPES = [
    # Qwen3.6-35B-A3B, TP=2: hidden 2048, moe_intermediate 768 -> 1536/768 per rank
    ("q35.moe    tp2", 256, 8, 2048, 768, 128),
    ("q35.moe    tp1", 256, 8, 2048, 1536, 128),
    # A GLM-4.7-Flash-shaped point (E=128, top_k=8). NOTE inter must be a multiple of the group
    # size: gemm2 contracts over K=inter, so inter=704 at g=128 makes the runtime group_size 140 and
    # the kernel (correctly, loudly) refuses every tile.
    ("glm.moe    tp2", 128, 8, 2048, 768, 128),
    # Gemma4 / a group-32 checkpoint, to keep the g axis in the surface
    ("g32.moe    tp2", 128, 4, 2048, 1024, 32),
    # A narrow-N grid point: this is where a constant BN is worst (few N-blocks -> ragged wave)
    ("narrowN    tp2", 64, 4, 1024, 512, 128),
]
MS = [1, 2, 8, 32, 64, 256, 1024]


def pack_uint4_3d(w: torch.Tensor) -> torch.Tensor:
    E, N, K = w.shape
    w = w.to(torch.int32)
    packed = torch.zeros((E, N, K // 8), dtype=torch.int32, device=w.device)
    for i in range(8):
        packed |= (w[:, :, i::8] & 0xF) << (i * 4)
    return packed


def expert_stack(E: int, N: int, K: int, g: int):
    wp = pack_uint4_3d(torch.randint(0, 16, (E, N, K), dtype=torch.int8, device=DEV))
    sc = (torch.randn(E, N, K // g, device=DEV).abs() * 0.02 + 0.002).to(torch.float16)
    return wp, sc


def rotation(E, hidden, inter, g, M, top_k):
    """R distinct (w13, w2) expert stacks whose TOTAL bytes exceed the 64 MB MALL, sized in BYTES."""
    n1, k1 = 2 * inter, hidden
    n2, k2 = hidden, inter
    wb = E * (n1 * (k1 // 8) * 4 + n1 * (k1 // g) * 2 + n2 * (k2 // 8) * 4 + n2 * (k2 // g) * 2)
    percall = wb + M * hidden * 4 * 3
    R = max(2, min(ROT_MAX_COPIES, -(-ROT_TARGET_BYTES // wb) + 1))
    R = max(2, min(R, max(2, ROT_MAX_VRAM // max(percall, 1))))
    ws = []
    for _ in range(R):
        w13p, w13s = expert_stack(E, n1, k1, g)
        w2p, w2s = expert_stack(E, n2, k2, g)
        ws.append((w13p, w13s, w2p, w2s))
    return ws, R, wb


def _eager_us(fn, ws) -> float:
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for w in ws:
        fn(w)
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) * 1e3 / len(ws)


def time_graph(fn, ws, budget_us: float = CELL_BUDGET_US, max_reps: int = 20, min_reps: int = 2):
    """Graph-replay device timing. A per-call synchronize() has a ~40 us wall floor on this box --
    larger than most of these kernels -- so every number here is a replay of len(ws) back-to-back
    calls, event-bracketed, divided by the call count."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            for w in ws:
                fn(w)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    est = max(_eager_us(fn, ws), 1.0)
    reps = int(max(min_reps, min(max_reps, budget_us / (est * len(ws)))))
    graph = torch.cuda.CUDAGraph()
    pool = torch.cuda.graph_pool_handle()
    with torch.cuda.graph(graph, pool=pool):
        for w in ws:
            fn(w)
    torch.cuda.synchronize()
    for _ in range(2):
        graph.replay()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(reps):
        graph.replay()
    e1.record()
    torch.cuda.synchronize()
    us = e0.elapsed_time(e1) * 1e3 / (reps * len(ws))
    del graph, pool
    torch.cuda.synchronize()
    return us, reps


def moe_legal(bm, bn, g, gtile=4, a_in_lds=False):
    """Mirror the launcher's clamps: WARPS_N must divide NFRAG, the workgroup is <= 8 warps, and the
    B staging LDS (GTILE groups of it, clamped at 40 KB) must fit."""
    nwm = bm // 16
    nfrag = bn // 16
    if nwm < 1 or nwm > 8 or nfrag < 1:
        return False
    wn = min(8 // nwm, nfrag)
    wn = 4 if wn >= 4 else (2 if wn >= 2 else 1)
    while wn > 1 and nfrag % wn:
        wn //= 2
    if a_in_lds:
        return (bm + bn) * (g + 8) <= LDS_MAX
    gt = max(1, gtile)
    per = bn * (g + 8)
    while gt > 1 and per * gt > 40960:
        gt -= 1
    return per * gt <= LDS_MAX


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--csv", default="")
    ap.add_argument("--ms", default="")
    ap.add_argument("--shapes", default="")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    fh = open(args.out, "w") if args.out else None
    ch = open(args.csv, "w") if args.csv else None

    def out(s=""):
        print(s, flush=True)
        if fh:
            fh.write(s + "\n")
            fh.flush()

    def csv(s):
        if ch:
            ch.write(s + "\n")
            ch.flush()

    sys.path.insert(0, "/engine/python")
    import fp8_wmma as W
    from minisgl.quant.kernels import w4a8_moe

    def prev_block_m(M: int, E: int, top_k: int) -> int:
        """The heuristic this work replaces, kept HERE as the explicit baseline arm rather than
        imported: `_moe_block_m` now calls the shared chooser, so importing it would compare the new
        rule against itself. The rule was: the largest of {16,32,64,128} not exceeding the average
        rows-per-expert. One variable, no N, no CU count, no LDS term."""
        rpe = (M * top_k) / max(E, 1)
        bm = 16
        for c in (16, 32, 64, 128):
            if c <= rpe:
                bm = c
        return bm

    torch.manual_seed(0)
    ms = [int(v) for v in args.ms.split(",")] if args.ms else list(MS)
    shapes = [s for s in SHAPES if (not args.shapes or any(t in s[0] for t in args.shapes.split(",")))]
    if args.smoke:
        shapes, ms = shapes[:1], ms[:2]
    csv("name,E,top_k,hidden,inter,g,M,cand,us,reps,R")

    out(f"tiles {len(BM_SET)}x{len(BN_SET)}   shapes {len(shapes)}   M {ms}")
    out(f"chooser CU is PINNED (tile_select.h); torch {torch.__version__}\n")

    for name, E, top_k, hidden, inter, g in shapes:
        n1, k1 = 2 * inter, hidden
        n2, k2 = hidden, inter
        out(f"=== {name}  E={E} top_k={top_k} hidden={hidden} inter={inter} g={g} ===")
        out(f"    gemm1 N={n1} K={k1}   gemm2 N={n2} K={k2}")
        for M in ms:
            ws, R, wb = rotation(E, hidden, inter, g, M, top_k)
            x = torch.randn(M, hidden, device=DEV, dtype=torch.bfloat16) * 0.05
            gate = torch.randn(M, E, device=DEV, dtype=torch.bfloat16)

            # --- CONTROL FIRST: the same build, recorded twice, so a later delta has a floor ---
            base = None
            rows = []
            for bm in BM_SET:
                for bn in BN_SET:
                    if not moe_legal(bm, bn, g):
                        continue
                    os.environ["VLLM_W4A8_MOE_BN"] = str(bn)
                    fn = lambda ww, _bm=bm: w4a8_moe(
                        x, ww[0], ww[1], None, ww[2], ww[3], None, gate, top_k, True,
                        kernel="wmma", block_m=_bm,
                    )
                    try:
                        us, reps = time_graph(fn, ws)
                    except Exception as exc:  # a tile the kernel refuses is INFORMATION, not a skip
                        out(f"  M={M:>5} {bm}x{bn}: REFUSED {type(exc).__name__}: {str(exc)[:110]}")
                        continue
                    rows.append((us, f"{bm}x{bn}"))
                    csv(f"{name},{E},{top_k},{hidden},{inter},{g},{M},{bm}x{bn},{us:.3f},{reps},{R}")
                    if base is None:
                        base = us
            os.environ.pop("VLLM_W4A8_MOE_BN", None)
            if not rows:
                out(f"  M={M:>5}  no legal tile")
                continue
            rows.sort()
            best_us, best_c = rows[0]
            # what shipped before this change, and what the chooser picks now
            prev_bm = prev_block_m(M, E, top_k)
            prev_c = f"{prev_bm}x64"
            prev_us = next((u for u, c in rows if c == prev_c), float("nan"))
            ch_bm, ch_bn1, ch_bn2 = W.moe_tile_choose(M * top_k, E, n1, k1, g, n2, k2, g)
            # the surface sweeps ONE BN for both GEMMs; score the chooser's block_m at its gemm1 BN
            ch_c = f"{ch_bm}x{ch_bn1}"
            ch_us = next((u for u, c in rows if c == ch_c), float("nan"))
            out(
                f"  M={M:>5} R={R:>3} ({wb >> 20} MB)  oracle {best_c:>7} {best_us:8.2f}us | "
                f"prev {prev_c:>7} {prev_us:8.2f}us | chooser {ch_c:>7} {ch_us:8.2f}us"
                f"  gain {prev_us / ch_us if ch_us == ch_us else float('nan'):.2f}x"
                f"  vs oracle {ch_us / best_us if ch_us == ch_us else float('nan'):.2f}x"
            )
            del ws
            torch.cuda.empty_cache()
        out()

    out("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
