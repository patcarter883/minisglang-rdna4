#!/usr/bin/env python
"""Which VARIABLES does the tile rule need? Fit nested rule families and price each term. CPU-only.

    python tools/w4a8_dense_tile_fit.py _tile_surface.csv [--cu 64]

The discipline this enforces: a term is only allowed into the rule if the surface pays for it. Each
family below adds exactly one variable to the one above, and the printed geomean/worst is the price
of leaving that variable out. Three constants have already been shipped in this repo encoding the
WRONG variable, so "it seemed like the deciding term" is not evidence.

  F0  one constant tile                      (what ships today: 256x128)
  F1  tile(M)                                the padded-rows / waves-per-workgroup term
  F2  tile(M, workgroup count)               + does the grid cover the CUs?  (brings in N and CU)
  F3  per-cell oracle                        the ceiling
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict

import _w4a8_tile_policy as _policy

# "auto" is the CHOOSER'S OWN PICK (--auto), not a tile: it has no "BMxBN" to parse and it is
# not an arm either. Treat it as a non-tile candidate everywhere, or `parse_tile` raises on it
# and the whole analysis refuses to run against any surface swept with --auto.
ARMS = ("prefill_wmma", "prefill_wmma_ashuffle", "prefill_wmma:smallm_off", "auto")
LDS_MAX = 65536


def parse_tile(c):
    a, b = c.split("x")
    return int(a), int(b)


def legal(bm, bn, g):
    """LDS fit. Delegated to _w4a8_tile_policy so this file cannot carry a stale copy of the
    formula — the previous inline `(bm + bn) * (g + 8)` under-reports by 8x at g=16, because the
    staging round is GROUPS_PER_STAGE quant groups deep and there is a static scale array too."""
    return _policy.tile_lds(bm, bn, g) <= _policy.LDS_BUDGET


def ceildiv(a, b):
    return -(-a // b)


def gm(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--cu", type=int, default=64)
    args = ap.parse_args()

    t = defaultdict(dict)
    for r in csv.DictReader(open(args.csv)):
        t[(r["name"], int(r["K"]), int(r["N"]), int(r["g"]), int(r["M"]))][r["cand"]] = float(
            r["us"]
        )
    tiles = sorted({parse_tile(c) for d in t.values() for c in d if c not in ARMS})
    oracle = {}
    for key, d in t.items():
        dt = {c: v for c, v in d.items() if c not in ARMS}
        if dt:
            b = min(dt, key=dt.get)
            oracle[key] = (b, dt[b])
    keys = sorted(oracle)
    Ms = sorted({k[4] for k in keys})

    def cost_of(fn, label, verbose=True):
        rs, worst, wk, miss = [], 1.0, None, 0
        for k in keys:
            name, K, N, g, M = k
            pick = fn(M, N, K, g)
            if pick not in t[k]:
                miss += 1
                continue
            r = t[k][pick] / oracle[k][1]
            rs.append(r)
            if r > worst:
                worst, wk = r, (name, N, M, pick, oracle[k][0])
        if verbose:
            s = f"{label:<44}{gm(rs):>9.4f}x{worst:>9.2f}x{sum(1 for x in rs if x>1.1):>9}"
            if wk:
                s += f"   worst: {wk[0]} N={wk[1]} M={wk[2]} -> {wk[3]} (oracle {wk[4]})"
            print(s)
        return gm(rs), worst

    print(f"cells {len(keys)}   tiles {len(tiles)}   CU={args.cu}")
    print(f"\n{'family':<44}{'geomean':>10}{'worst':>9}{'>1.10x':>9}")

    # ---- F0: one constant tile ---------------------------------------------------------------
    best_c, best_gm = None, 1e9
    for bm, bn in tiles:
        c = f"{bm}x{bn}"
        rs = [t[k][c] / oracle[k][1] for k in keys if c in t[k]]
        if len(rs) < 0.9 * len(keys):
            continue
        if gm(rs) < best_gm:
            best_gm, best_c = gm(rs), c
    cost_of(lambda M, N, K, g: "256x128", "F0  256x128 (SHIPPED default)")
    cost_of(lambda M, N, K, g: best_c, f"F0' {best_c} (best possible CONSTANT)")

    # ---- F1: tile(M) -- fitted per M, then read off as bands ---------------------------------
    perM = {}
    for M in Ms:
        km = [k for k in keys if k[4] == M]
        bt, bg = None, 1e9
        for bm, bn in tiles:
            c = f"{bm}x{bn}"
            rs = [t[k][c] / oracle[k][1] for k in km if c in t[k]]
            if len(rs) < 0.9 * len(km):
                continue
            if gm(rs) < bg:
                bg, bt = gm(rs), c
        perM[M] = (bt, bg)
    cost_of(lambda M, N, K, g: perM[M][0], "F1  tile(M), fitted per M")
    print("\n  F1 fitted table   " + "  ".join(f"M<={M}:{perM[M][0]}({perM[M][1]:.2f}x)" for M in Ms))

    # ---- F2: tile(M, workgroup count) --------------------------------------------------------
    # BN is capped so the grid covers the CUs: the largest BN with ceil(M/BM)*ceil(N/BN) >= CU.
    # BM comes from the F1 fit (the M term). This adds N and the CU count, nothing else.
    def f2(M, N, K, g):
        bm0, bn0 = parse_tile(perM[M][0])
        ok = [(a, b) for (a, b) in tiles if legal(a, b, g) and a == bm0]
        feas = [(a, b) for (a, b) in ok if ceildiv(M, a) * ceildiv(N, b) >= args.cu]
        if not feas:
            return perM[M][0]
        bn = max(b for _, b in feas)
        return f"{bm0}x{min(bn, bn0)}"

    cost_of(f2, "F2  F1 with BN capped by 'grid covers CUs'")

    # ---- F2b: the closed form -- both BM and BN from M, BN floored by the CU cover ------------
    def closed(M, N, K, g, mband=None):
        # BM: waves per workgroup. At M < BM the grid has ONE row-block, so BM/16 is the only
        # source of waves; below 64 there are too few to hide the staging latency.
        if M <= 64:
            bm = 64
        elif M <= 256:
            bm = 128
        elif M <= 1024:
            bm = 256
        else:
            bm = 512
        # BN: accumulator registers (NFRAG = BN/16 -> 16*NFRAG VGPRs) trade occupancy against WMMA
        # ILP + weight reuse. Small M is latency-bound -> small BN; large M has the reuse to pay
        # for a wide tile.
        bn = 32 if M <= 96 else (64 if M <= 192 else 128)
        # ... but never so wide that the grid stops covering the machine.
        while bn > 32 and ceildiv(M, bm) * ceildiv(N, bn) < args.cu:
            bn //= 2
        while bn < 128 and ceildiv(M, bm) * ceildiv(N, bn) > 8 * args.cu:
            bn *= 2
        while not legal(bm, bn, g) and bn > 32:
            bn //= 2
        if (bm, bn) not in tiles:
            cand = [x for x in tiles if x[1] == bn and legal(*x, g)]
            bm = min(cand, key=lambda x: abs(x[0] - bm))[0] if cand else bm
        return f"{bm}x{bn}"

    cost_of(closed, "F2b closed-form BM(M) x BN(M, N/CU)")

    # ---- how much of the gap is BM vs BN? ----------------------------------------------------
    print("\nWHICH HALF OF THE TILE CARRIES THE WIN?")
    for fixname, fixfn in (
        ("oracle BN, BM from F1", lambda M, N, K, g: f"{parse_tile(perM[M][0])[0]}x?"),
        ("oracle BM, BN from F1", None),
    ):
        pass
    rs_bn, rs_bm = [], []
    for k in keys:
        name, K, N, g, M = k
        obm, obn = parse_tile(oracle[k][0])
        # best time with BN forced to the oracle's, BM free
        c1 = [c for c in t[k] if c not in ARMS and parse_tile(c)[1] == obn]
        c2 = [c for c in t[k] if c not in ARMS and parse_tile(c)[0] == obm]
        # ... and the reverse: what does the SHIPPED value of each half cost?
        s1 = [c for c in t[k] if c not in ARMS and parse_tile(c)[0] == 256]  # BM pinned to 256
        s2 = [c for c in t[k] if c not in ARMS and parse_tile(c)[1] == 128]  # BN pinned to 128
        if s1:
            rs_bm.append(min(t[k][c] for c in s1) / oracle[k][1])
        if s2:
            rs_bn.append(min(t[k][c] for c in s2) / oracle[k][1])
    print(f"  BM pinned to the shipped 256 (BN free) : geomean {gm(rs_bm):.4f}x  worst {max(rs_bm):.2f}x")
    print(f"  BN pinned to the shipped 128 (BM free) : geomean {gm(rs_bn):.4f}x  worst {max(rs_bn):.2f}x")
    print("  -> the larger number is the half of the tile the hard-wire was actually costing.")

    # ---- oracle tile as a function of M and N, printed as the surface it is -------------------
    print("\nORACLE TILE, M down / N across")
    Ns = sorted({k[2] for k in keys})
    print(f"{'M':>6}  " + "".join(f"{n:>9}" for n in Ns))
    for M in Ms:
        row = []
        for N in Ns:
            hits = {oracle[k][0] for k in keys if k[4] == M and k[2] == N}
            row.append("/".join(sorted(hits)) if hits else "-")
        print(f"{M:>6}  " + "".join(f"{c:>9}" for c in row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
