#!/usr/bin/env python
"""An ANALYTIC cost model for `wmma_tiled_tuned`'s tile, validated against the swept surface.

    python tools/w4a8_dense_tile_model.py _tile_surface.csv [--cu 64] [--fit]

WHY A MODEL AND NOT A TABLE. A fitted table of swept cells is only correct on shapes we happened to
bench; the next checkpoint arrives with a different (N, K, group_size) and falls off the table into
whatever the default was, quietly, at 2-8x. Every term below is a property of the LAUNCH, computable
for any shape, so the selector extrapolates:

    NWARPS   = BM/16                          warps per workgroup (one warp per 16 rows)
    THREADS  = 32*NWARPS
    NFRAG    = BN/16                          independent WMMA accumulators per warp (ILP)
    LDS      = (BM+BN)*(group_size+8)         A and B staging tiles, dynamic
    WGS      = ceil(M/BM)*ceil(N/BN)          the grid
    BLK/CU   = min(LDS_BUDGET/LDS, WAVES_PER_CU/NWARPS, VGPR-limited waves / NWARPS)
    ROUNDS   = ceil(WGS / (CU*BLK_PER_CU))    dispatch quantisation -- a ragged last round costs
                                              a whole round, which is the term the N=2048 case
                                              (16 workgroups on 64 CUs) gets catastrophically wrong

and per workgroup, per K-group, per thread:

    A-stage  = ceil(BM*BK/4 / THREADS) = BK/8              (independent of BM: more rows, more threads)
    B-stage  = ceil(BN*(BK/8) / THREADS)                   weight dequant+LDS store, the hot loop
    WMMA     = (BK/16)*NFRAG                               per warp

    cost = ROUNDS * (K/BK) * (cA*A + cB*B + cW*WMMA)  , floored by the HBM traffic
    hbm  = ceil(M/BM) * (N*K/2 + N*(K/BK)*2)  bytes  ->  hbm / BW

Only cA, cB, cW and BW are fitted, and they are RELATIVE ISSUE COSTS + one bandwidth -- not shape
thresholds. The script prints predicted-vs-measured for every cell; disagreement is a missing term,
not a cell to special-case.
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict

ARMS = ("prefill_wmma", "prefill_wmma_ashuffle", "prefill_wmma:smallm_off")
LDS_BUDGET = 65536
# gfx1201: 2 SIMD32 per CU, 16 wave32 slots per SIMD -> 32 waves/CU; 1536 VGPRs per SIMD in wave32.
WAVES_PER_CU = 32
VGPR_PER_SIMD = 1536
WAVES_PER_SIMD_MAX = 16


def parse_tile(c):
    a, b = c.split("x")
    return int(a), int(b)


def ceildiv(a, b):
    return -(-a // b)


def gm(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def vgprs(bn: int) -> int:
    """VGPRs a wave of the tiled-tuned kernel needs, as a function of BN alone.

    The accumulator state is the whole story: `running[NFRAG][8]` (8*NFRAG) + `acc[NFRAG]` v8f
    (8*NFRAG) + the double-buffered LDS operands `b_cur/b_nx[NFRAG]` v2i (4*NFRAG) = 20*NFRAG, plus
    a fixed ~24 for addresses/indices. BM does not appear -- each warp owns 16 rows whatever BM is.
    """
    return 24 + 20 * (bn // 16)


def blocks_per_cu(bm, bn, g):
    nwarps = bm // 16
    lds = (bm + bn) * (g + 8) + 4 * bn + 4 * bm
    by_lds = LDS_BUDGET // max(lds, 1)
    waves_simd = min(WAVES_PER_SIMD_MAX, VGPR_PER_SIMD // max(vgprs(bn), 1))
    by_waves = (waves_simd * 2) // max(nwarps, 1)
    return max(0, min(by_lds, by_waves))


def model_cost(M, N, K, g, bm, bn, cu, cA, cB, cW, LAT, bw=0.0):
    """Modelled time for one call at tile (bm, bn). Three multiplicative terms, one floor.

    WORK      total thread-instruction slots the launch issues. Note the A-staging term is the one
              that scales with 1/BN (the activation tile is re-staged once per N-block), and the
              WMMA term is BN-invariant -- so on WORK alone a wide BN always wins.
    UNDERFILL the machine is CU*blocks_per_cu workgroup slots wide; a launch with fewer workgroups
              than that leaves slots empty for its whole life, and a launch that does not divide it
              pays a whole extra round for the remainder. This is the term the hard-wired BN=128
              gets catastrophically wrong at narrow N (16 workgroups on a 64-CU card) and the one
              that makes the optimum move with N and with the CU count.
    LATENCY   the inner loop is LDS/global-latency bound, hidden by waves resident on the CU. A
              wide BN costs accumulator VGPRs (20 per NFRAG) which costs waves, so the wide tile
              only pays for itself once there is enough real work to amortise it -- i.e. at large M.
              This is why the optimum tile grows with M rather than tracking it.

    Co-residency does NOT multiply issue throughput -- a CU has a fixed issue width -- so blocks/CU
    enters through UNDERFILL and LATENCY only, never as a divisor of WORK. (Getting that wrong is
    what made the first version of this model prefer BN=32 at M=2048, where it loses 1.6x.)
    """
    if (bm + bn) * (g + 8) > LDS_BUDGET:
        return None
    bpc = blocks_per_cu(bm, bn, g)
    if bpc <= 0:
        return None
    nwarps, threads, nfrag = bm // 16, 32 * (bm // 16), bn // 16
    mblk, nblk = ceildiv(M, bm), ceildiv(N, bn)
    wgs = mblk * nblk
    bk = g
    # thread-instructions ONE workgroup issues, per K-group, summed over its threads:
    #   A stage  bm*bk/4 dwords   B stage  bn*bk/8 dequant+store   WMMA  threads*(bk/16)*NFRAG
    per_wg = K * (cA * (bm / 4.0) + cB * (bn / 8.0) + cW * (threads * bn / 256.0))
    # ROUNDS: workgroup slots each CU must serve. The ceiling is the ragged last wave -- a grid of
    # 16 workgroups and a grid of 64 both cost one round on a 64-CU card, which is exactly why the
    # hard-wired BN=128 collapses at narrow N.
    rounds = ceildiv(wgs, cu)
    # OCCUPANCY: waves resident on a CU = the blocks it actually gets (never more than the grid
    # supplies, never more than LDS/VGPRs allow) x the warps per block.
    live = min(bpc, rounds)
    occ = min(WAVES_PER_CU, live * nwarps)
    return rounds * per_wg * (1.0 + LAT / occ)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--cu", type=int, default=64)
    ap.add_argument("--fit", action="store_true")
    args = ap.parse_args()

    t = defaultdict(dict)
    for r in csv.DictReader(open(args.csv)):
        t[(r["name"], int(r["K"]), int(r["N"]), int(r["g"]), int(r["M"]))][r["cand"]] = float(
            r["us"]
        )
    tiles = sorted({parse_tile(c) for d in t.values() for c in d if c not in ARMS})
    oracle = {k: min(((c, v) for c, v in d.items() if c not in ARMS), key=lambda p: p[1])
              for k, d in t.items() if any(c not in ARMS for c in d)}
    keys = sorted(oracle)

    def evaluate(cA, cB, cW, LAT, verbose=False):
        rs, worst, wk = [], 1.0, None
        for k in keys:
            name, K, N, g, M = k
            best, bc = None, None
            for bm, bn in tiles:
                c = model_cost(M, N, K, g, bm, bn, args.cu, cA, cB, cW, LAT)
                if c is None:
                    continue
                if best is None or c < best:
                    best, bc = c, f"{bm}x{bn}"
            if bc not in t[k]:
                continue
            r = t[k][bc] / oracle[k][1]
            rs.append(r)
            if r > worst:
                worst, wk = r, (name, N, M, bc, oracle[k][0])
        return gm(rs), worst, wk, sum(1 for x in rs if x > 1.10), len(rs)

    best = None
    if args.fit:
        print("fitting the four relative-cost constants ...")
        for cA in (0.5, 1.0, 2.0, 4.0):
            for cB in (0.5, 1.0, 2.0, 4.0, 8.0):
                for cW in (0.25, 0.5, 1.0, 2.0, 4.0):
                    for LAT in (0, 4, 8, 16, 32, 64):
                        g_, w_, wk_, bad_, n_ = evaluate(cA, cB, cW, LAT)
                        if best is None or (g_, w_) < (best[0], best[1]):
                            best = (g_, w_, wk_, bad_, n_, cA, cB, cW, LAT)
        print(f"  best: cA={best[5]} cB={best[6]} cW={best[7]} LAT={best[8]}")
    else:
        best = evaluate(1.0, 1.0, 1.0, 16) + (1.0, 1.0, 1.0, 16)

    g_, w_, wk_, bad_, n_ = best[:5]
    cA, cB, cW, LAT = best[5:]
    print(f"\nMODEL vs per-cell oracle: geomean {g_:.4f}x   worst {w_:.2f}x   "
          f">1.10x on {bad_}/{n_} cells")
    if wk_:
        print(f"  worst cell: {wk_[0]} N={wk_[1]} M={wk_[2]} -> model picked {wk_[3]}, "
              f"oracle {wk_[4]}")

    # ---- per-cell agreement, the thing to chase ----------------------------------------------
    print("\nPREDICTED vs MEASURED, every cell (model pick / oracle pick / ratio)")
    Ns = sorted({k[2] for k in keys})
    Ms = sorted({k[4] for k in keys})
    print(f"{'M':>6}  " + "".join(f"{n:>14}" for n in Ns))
    for M in Ms:
        row = []
        for N in Ns:
            ks = [k for k in keys if k[4] == M and k[2] == N]
            if not ks:
                row.append("-")
                continue
            k = ks[0]
            name, K, Nn, g, _ = k
            bc, bcost = None, None
            for bm, bn in tiles:
                c = model_cost(M, N, K, g, bm, bn, args.cu, cA, cB, cW, LAT)
                if c is not None and (bcost is None or c < bcost):
                    bcost, bc = c, f"{bm}x{bn}"
            r = t[k].get(bc, float("nan")) / oracle[k][1]
            mark = "" if r < 1.05 else ("!" if r < 1.3 else "!!")
            row.append(f"{bc}/{r:.2f}{mark}")
        print(f"{M:>6}  " + "".join(f"{c:>14}" for c in row))

    # ---- rank correlation: does the model order the WHOLE tile set, or just find the min? -----
    print("\nRANK QUALITY -- Spearman of model cost vs measured us, per cell (mean over cells)")
    cors = []
    for k in keys:
        name, K, N, g, M = k
        pairs = []
        for bm, bn in tiles:
            c = f"{bm}x{bn}"
            mc = model_cost(M, N, K, g, bm, bn, args.cu, cA, cB, cW, LAT)
            if mc is not None and c in t[k]:
                pairs.append((mc, t[k][c]))
        if len(pairs) < 6:
            continue
        rx = {v: i for i, v in enumerate(sorted(p[0] for p in pairs))}
        ry = {v: i for i, v in enumerate(sorted(p[1] for p in pairs))}
        n = len(pairs)
        d2 = sum((rx[a] - ry[b]) ** 2 for a, b in pairs)
        cors.append(1 - 6 * d2 / (n * (n * n - 1)))
    print(f"  mean Spearman rho = {sum(cors)/len(cors):.3f} over {len(cors)} cells "
          f"(1.0 = the model orders every tile correctly)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
