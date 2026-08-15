#!/usr/bin/env python
"""Derive the shape-derived tile rule for `wmma_tiled_tuned` from the swept surface. CPU-only.

    python tools/w4a8_dense_tile_rule.py _tile_surface.csv [--cu 64]

The rule family is STRUCTURAL, not a table of thresholds. `wmma_tiled_tuned` launches

    grid = (ceil(M/BM), ceil(N/BN))   workgroups of (BM/16) warps each

so a tile choice sets two independent things, and the surface moves with both:

  * WORKGROUPS = ceil(M/BM)*ceil(N/BN).  Below the CU count, part of the machine is idle for the
    whole kernel. This is the term that makes BN track N, and it is why the mid-band wanted a
    SMALLER BN than the shipped 128 -- at N=2048, BN=128 is 16 workgroups on a 64-CU card.
  * WAVES per CU = WORKGROUPS*(BM/16)/CU.  A workgroup is BM/16 waves, and at M < BM the grid has
    exactly one row-block, so BM is the ONLY source of waves for latency hiding. This is why BM
    does NOT simply track M: a 16-row tile at M=1 is one wave per workgroup and the memory-bound
    inner loop has nothing to hide behind.

Everything else (larger BN = more independent WMMA accumulators per warp, and fewer redundant
weight-slab passes) argues monotonically for the LARGEST tile, so the rule is: take the largest BN
that still satisfies both occupancy constraints, then the smallest BM that satisfies the wave
constraint. The two constants are the two constraint targets; this script fits them over the
measured surface and prints the sensitivity, so neither is a magic number nor a fourth constant
encoding the wrong variable.
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
SHIPPED = {
    (64, 64), (64, 128), (80, 128), (96, 128), (112, 128), (128, 64),
    (128, 128), (128, 256), (192, 128), (256, 64), (256, 128), (256, 192),
}


def parse_tile(c):
    a, b = c.split("x")
    return int(a), int(b)


def legal(bm, bn, g):
    """LDS fit, from the shared policy — see _w4a8_tile_policy on why the inline formula is wrong."""
    return _policy.tile_lds(bm, bn, g) <= _policy.LDS_BUDGET


def ceildiv(a, b):
    return -(-a // b)


def make_rule(wg_mult: float, wave_target: float):
    """Largest BN, then smallest BM, subject to WGs >= wg_mult*CU and waves/CU >= wave_target."""

    def r(M, N, K, g, cu, tiles):
        ok = [t for t in tiles if legal(t[0], t[1], g)]
        feas = []
        for bm, bn in ok:
            wgs = ceildiv(M, bm) * ceildiv(N, bn)
            waves = wgs * (bm // 16) / cu
            if wgs >= wg_mult * cu and waves >= wave_target:
                feas.append((bm, bn, wgs, waves))
        if not feas:
            # nothing satisfies both: maximise waves/CU, the harder constraint to recover from
            best = max(ok, key=lambda t: (ceildiv(M, t[0]) * ceildiv(N, t[1]) * (t[0] // 16), t[1]))
            return f"{best[0]}x{best[1]}"
        bn_max = max(f[1] for f in feas)
        cand = [f for f in feas if f[1] == bn_max]
        bm_min = min(f[0] for f in cand)
        return f"{bm_min}x{bn_max}"

    return r


def score(rule, oracle, t, tiles, cu, restrict=None):
    ratios, worst, wk, exact, n = [], 1.0, None, 0, 0
    for key, (ob, ous) in oracle.items():
        name, K, N, g, M = key
        tl = [x for x in tiles if restrict is None or x in restrict]
        pick = rule(M, N, K, g, cu, tl)
        d = t[key]
        if pick not in d:
            continue
        n += 1
        rr = d[pick] / ous
        ratios.append(rr)
        exact += pick == ob
        if rr > worst:
            worst, wk = rr, (name, N, M, pick, ob)
    if not ratios:
        return None
    gm = math.exp(sum(math.log(x) for x in ratios) / len(ratios))
    return gm, worst, wk, exact, n, sum(1 for x in ratios if x > 1.10)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--cu", type=int, default=64)
    args = ap.parse_args()

    t = defaultdict(dict)
    for r in csv.DictReader(open(args.csv)):
        t[(r["name"], int(r["K"]), int(r["N"]), int(r["g"]), int(r["M"]))][r["cand"]] = float(r["us"])
    tiles = sorted({parse_tile(c) for d in t.values() for c in d if c not in ARMS})

    oracle, oracle_ship = {}, {}
    for key, d in t.items():
        dt = {c: v for c, v in d.items() if c not in ARMS}
        if not dt:
            continue
        b = min(dt, key=dt.get)
        oracle[key] = (b, dt[b])
        ds = {c: v for c, v in dt.items() if parse_tile(c) in SHIPPED}
        if ds:
            bs = min(ds, key=ds.get)
            oracle_ship[key] = (bs, ds[bs])

    print(f"cells: {len(oracle)}   tiles measured: {len(tiles)}   CU={args.cu}")

    # ---- what does the tile knob buy at all? -------------------------------------------------
    hard = [t[k]["256x128"] / o[1] for k, o in oracle.items() if "256x128" in t[k]]
    hard_s = [t[k]["256x128"] / o[1] for k, o in oracle_ship.items() if "256x128" in t[k]]
    print(f"\nhard-wired 256x128 vs per-cell oracle (ALL tiles)     : "
          f"geomean {math.exp(sum(map(math.log, hard))/len(hard)):.4f}x  worst {max(hard):.2f}x")
    print(f"hard-wired 256x128 vs per-cell oracle (SHIPPED 12 only): "
          f"geomean {math.exp(sum(map(math.log, hard_s))/len(hard_s)):.4f}x  worst {max(hard_s):.2f}x")
    ext = [oracle_ship[k][1] / oracle[k][1] for k in oracle if k in oracle_ship]
    print(f"what the ADDED tiles buy over the shipped 12 (oracle vs oracle): "
          f"geomean {math.exp(sum(map(math.log, ext))/len(ext)):.4f}x  best {max(ext):.2f}x")

    # ---- fit the two constraint targets ------------------------------------------------------
    print("\nFITTING the two structural constants over the measured surface")
    print(f"{'wg>=cu*':>9}{'waves/CU>=':>12}{'geomean':>10}{'worst':>9}{'>1.10x':>8}{'exact':>7}")
    grid = []
    for a in (0.5, 1.0, 1.5, 2.0, 3.0):
        for b in (1, 2, 3, 4, 6, 8, 12, 16):
            s = score(make_rule(a, b), oracle, t, tiles, args.cu)
            if s:
                grid.append((s[0], a, b, s))
                print(f"{a:>9.1f}{b:>12}{s[0]:>9.4f}x{s[1]:>8.2f}x{s[5]:>8}{s[3]:>7}")
    grid.sort()
    gm, a, b, s = grid[0]
    print(f"\nBEST: WGs >= {a}*CU and waves/CU >= {b}  -> geomean {gm:.4f}x, worst {s[1]:.2f}x "
          f"({s[3]}/{s[4]} exact)")
    if s[2]:
        print(f"  worst cell: {s[2][0]} N={s[2][1]} M={s[2][2]} picked {s[2][3]}, oracle {s[2][4]}")
    print("  runners-up:  " + "   ".join(f"({x[1]},{x[2]})={x[0]:.4f}x" for x in grid[1:5]))

    # same fit restricted to the shipped 12, so the rule can ship without the new instantiations
    s12 = score(make_rule(a, b), oracle, t, tiles, args.cu, restrict=SHIPPED)
    print(f"\nsame rule restricted to the SHIPPED 12 tiles: geomean {s12[0]:.4f}x, worst {s12[1]:.2f}x")
    grid12 = []
    for aa in (0.5, 1.0, 1.5, 2.0, 3.0):
        for bb in (1, 2, 3, 4, 6, 8, 12, 16):
            ss = score(make_rule(aa, bb), oracle, t, tiles, args.cu, restrict=SHIPPED)
            if ss:
                grid12.append((ss[0], aa, bb, ss))
    grid12.sort()
    print(f"best over the shipped 12 alone: WGs >= {grid12[0][1]}*CU, waves/CU >= {grid12[0][2]}"
          f"  -> geomean {grid12[0][0]:.4f}x, worst {grid12[0][3][1]:.2f}x")

    # ---- can a CONSTANT express it? ----------------------------------------------------------
    print("\nCAN A SINGLE CONSTANT TILE EXPRESS IT?  (best fixed tile over the whole surface)")
    fixed = []
    for bm, bn in tiles:
        c = f"{bm}x{bn}"
        rs = [t[k][c] / o[1] for k, o in oracle.items() if c in t[k]]
        if len(rs) < 0.8 * len(oracle):
            continue
        fixed.append((math.exp(sum(map(math.log, rs)) / len(rs)), max(rs), c, len(rs)))
    fixed.sort()
    for gmv, wv, c, n in fixed[:6]:
        print(f"   {c:>9}: geomean {gmv:.4f}x  worst {wv:.2f}x  ({n} cells)")
    print(f"   -> the best CONSTANT is {fixed[0][2]} at {fixed[0][0]:.4f}x geomean / "
          f"{fixed[0][1]:.2f}x worst; the derived rule is {gm:.4f}x / {s[1]:.2f}x.")

    # ---- does the rule need an M term, an N term, both? --------------------------------------
    print("\nDOES THE OPTIMUM MOVE WITH M, WITH N, OR BOTH?  (oracle tile, marginalised)")
    byM, byN, byg = defaultdict(set), defaultdict(set), defaultdict(set)
    for (name, K, N, g, M), (ob, _) in oracle.items():
        byM[M].add(ob)
        byN[N].add(ob)
        byg[g].add(ob)
    print("  BN by M   : " + "  ".join(
        f"M={m}:{sorted({parse_tile(x)[1] for x in byM[m]})}" for m in sorted(byM)))
    print("  BN by N   : " + "  ".join(
        f"N={n}:{sorted({parse_tile(x)[1] for x in byN[n]})}" for n in sorted(byN)))
    print("  BM by M   : " + "  ".join(
        f"M={m}:{sorted({parse_tile(x)[0] for x in byM[m]})}" for m in sorted(byM)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
