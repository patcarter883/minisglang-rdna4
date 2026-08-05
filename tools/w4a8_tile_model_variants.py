#!/usr/bin/env python
"""Price every STRUCTURAL variant of tile_select.h's cost model against BOTH swept surfaces. CPU.

    python tools/w4a8_tile_model_variants.py \
        --dense tools/_fixtures/dense_tile_surface.csv \
        --moe   tools/_fixtures/moe_tile_surface.csv

WHY THIS EXISTS. tile_select.h has six fitted numbers and one structural form. "Which constant is
wrong" is the cheap question and it is usually the wrong one: the shipped constants are already at
the optimum of their family on the dense surface (verified below, F0), so a column that fits badly
is evidence about the FORM, not the values. This script makes changing the form as cheap as changing
a constant, and scores every candidate on the same cells.

SCORING -- restricted argmin. The model ranges over a 77-tile lattice; each surface measured ~25.
If a variant is allowed to pick an unmeasured tile, that cell silently drops out of its score, so a
variant that picks unmeasured tiles more often looks better for no reason. Every variant here picks
the argmin over THE TILES THAT CELL MEASURED, so all variants are scored on 100% of cells against
the same per-cell oracle and the geomeans are comparable. What this can falsify is RANKING; what it
cannot see is extrapolation, which is what the live re-time in w4a8_*_tile_verify.py is for.

THE TWO SURFACES DISAGREE ABOUT THE LATENCY TERM, which is the whole finding:
  * dense wants LAT=32 with no ILP credit (any ILP/split/rounds-softening variant costs more on the
    195-cell g=32 column than it buys on the 15-cell g=128 one);
  * MoE at M<=32 is 2-3x off with LAT=32, and the direction is always the same -- the model takes a
    TALL block_m for the occupancy and pays for it in masked padding rows.
Both are the same defect seen from two sides: `1 + LAT/OCC` is a steady-state latency-hiding law,
and it is applied to launches that are 3 waves deep. See ROUNDS_FLOOR below.
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict

ARMS = ("prefill_wmma", "prefill_wmma_ashuffle", "prefill_wmma:smallm_off")
LDS_BUDGET = 65536
WAVES_PER_CU = 32
VGPR_PER_SIMD = 1536
WAVES_PER_SIMD_MAX = 16
HW_BLOCKS_PER_CU = 4
MOE_MAX_WARPS = 8


def cd(a, b):
    return -(-a // b)


def gm(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def core(bm, bn, warps_n, g, row_blocks, n_blocks, z_blocks, k_groups, lds, shuffled, cu, p):
    """Byte-for-byte the arithmetic of tile_select.h::core_terms, with the structural knobs of the
    variant grid spliced in at the two points that are being questioned (ROUNDS and LATENCY)."""
    if lds > LDS_BUDGET or bm % 16 or bn % 16 or warps_n < 1:
        return None
    nwarps_m, nfrag = bm // 16, bn // 16
    if nfrag % warps_n:
        return None
    nfrag_w = nfrag // warps_n
    nwarps = nwarps_m * warps_n
    threads = 32 * nwarps
    if g <= 0 or g % 16:
        return None
    k_steps = g // 16
    vgpr = 24 + 20 * nfrag_w
    wps = min(WAVES_PER_SIMD_MAX, VGPR_PER_SIMD // vgpr)
    bpc = min((wps * 2) // nwarps, LDS_BUDGET // max(lds, 1), HW_BLOCKS_PER_CU)
    if bpc <= 0:
        return None
    wgs = row_blocks * n_blocks * z_blocks
    if wgs <= 0:
        return None
    rounds = cd(wgs, cu)
    live = min(bpc, rounds)
    occ = min(WAVES_PER_CU, live * nwarps)
    a_it = k_steps if shuffled else cd(bm * g // 4, threads)
    b_it = cd(bn * (g // 8), threads)
    lds_it = k_steps * (nfrag_w if shuffled else nfrag_w + 1)
    pt = p["cA"] * a_it + p["cB"] * b_it + p["cL"] * lds_it + p["cW"] * k_steps * nfrag_w
    # ---- LATENCY, the term under test ----------------------------------------------------------
    # `1 + LAT/OCC` says a CU with OCC resident waves hides LAT units of latency. That is a
    # STEADY-STATE law and it is being applied to launches that are a handful of waves deep, where
    # a workgroup's latency is paid once at the head of the pipe and never amortised again. DEPTH
    # caps how much of LAT occupancy is allowed to hide: a launch of ROUNDS rounds cannot amortise
    # more latency than it has rounds to amortise it over.
    if p["lat"] == "const":
        lat = 1.0 + p["LAT"] / occ
    elif p["lat"] == "depth":
        eff = min(float(occ), p["LAT"] * min(1.0, rounds / p["ROUNDS_FLOOR"]) + 1.0)
        lat = 1.0 + p["LAT"] / max(eff, 1.0)
    elif p["lat"] == "ilp":
        lat = 1.0 + p["LAT"] / (occ * (1.0 + p["alpha"] * (nfrag_w - 1)))
    elif p["lat"] == "cap":
        lat = 1.0 + p["LAT"] / max(occ, p["OCC_FLOOR"])
    else:
        raise SystemExit(f"unknown lat mode {p['lat']}")
    return rounds * k_groups * threads * pt * lat


def dense_cost(M, N, K, g, bm, bn, wn, cu, p):
    lds = (bm + bn) * (g + 8) + 4 * bn
    return core(bm, bn, wn, g, cd(M, bm), cd(N, bn), 1, K / g, lds, False, cu, p)


def moe_warps_n(bm, bn):
    nwm, nfrag = bm // 16, bn // 16
    if nwm < 1 or nfrag < 1:
        return 0
    wn = min(MOE_MAX_WARPS // nwm, nfrag)
    wn = 4 if wn >= 4 else (2 if wn >= 2 else 1)
    while wn > 1 and nfrag % wn:
        wn //= 2
    return wn


def moe_padded_rows(rows, E, bm):
    rows = max(rows, 1)
    hit = max(1, min(E, rows))
    return hit * cd(cd(rows, hit), bm) * bm


def moe_cost(rows, E, N, K, g, bm, bn, cu, p, gtile=4):
    wn = moe_warps_n(bm, bn)
    if wn < 1 or bm // 16 > MOE_MAX_WARPS:
        return None
    kg = K // g
    if kg <= 0:
        return None
    gt = max(1, min(gtile, kg))
    per = bn * (g + 8)
    while gt > 1 and per * gt > 40960:
        gt -= 1
    rb = moe_padded_rows(rows, E, bm) // bm
    return core(bm, bn, wn, g, rb, cd(N, bn), 1, kg, per * gt, True, cu, p)


def load(path, moe):
    t = defaultdict(dict)
    for r in csv.DictReader(open(path)):
        if r["cand"] in ARMS:
            continue
        if moe:
            k = (r["name"], int(r["E"]), int(r["top_k"]), int(r["hidden"]), int(r["inter"]),
                 int(r["g"]), int(r["M"]))
        else:
            k = (r["name"], int(r["K"]), int(r["N"]), int(r["g"]), int(r["M"]))
        t[k][r["cand"]] = float(r["us"])
    return {k: d for k, d in t.items() if len(d) >= 4}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense", default="tools/_fixtures/dense_tile_surface.csv")
    ap.add_argument("--moe", default="tools/_fixtures/moe_tile_surface.csv")
    ap.add_argument("--cu", type=int, default=64)
    args = ap.parse_args()

    D = load(args.dense, moe=False)
    Mo = load(args.moe, moe=True)
    print(f"dense cells {len(D)}   moe cells {len(Mo)}   CU={args.cu}")

    def score(p):
        res = {}
        for tag, cells, isd in (("dense", D, True), ("moe", Mo, False)):
            per_g, allr, small, big = defaultdict(list), [], [], []
            for k, d in cells.items():
                best, bc = None, None
                for c in d:
                    bm, bn = (int(v) for v in c.split("x"))
                    if isd:
                        name, K, N, g, M = k
                        # the dense surface was swept at WARPS_N=1 (the only thing that shipped)
                        cst = dense_cost(M, N, K, g, bm, bn, 1, args.cu, p)
                    else:
                        name, E, tk, hid, inter, g, M = k
                        # score on gemm1's shape; gemm2 rides the same block_m and BN sweep
                        cst = moe_cost(M * tk, E, 2 * inter, hid, g, bm, bn, args.cu, p)
                    if cst is None:
                        continue
                    if best is None or cst < best:
                        best, bc = cst, c
                if bc is None:
                    continue
                r = d[bc] / min(d.values())
                allr.append(r)
                per_g[k[3] if isd else k[5]].append(r)
                (small if k[-1] <= 32 else big).append(r)
            res[tag] = dict(gm=gm(allr), worst=max(allr) if allr else 0,
                            g32=gm(per_g.get(32, [])), g128=gm(per_g.get(128, [])),
                            small=gm(small), big=gm(big), n=len(allr))
        return res

    def show(tag, p):
        s = score(p)
        d, m = s["dense"], s["moe"]
        print(f"{tag:<46} DENSE gm={d['gm']:.4f} worst={d['worst']:.2f} "
              f"g32={d['g32']:.3f} g128={d['g128']:.3f} | "
              f"MOE gm={m['gm']:.4f} worst={m['worst']:.2f} "
              f"M<=32={m['small']:.3f} M>32={m['big']:.3f}")
        return s

    base = dict(cA=1.0, cB=16.0, cL=0.25, cW=0.25, LAT=32.0, alpha=0.25,
                lat="const", OCC_FLOOR=8, ROUNDS_FLOOR=8)

    print("\n=== SHIPPED ===")
    show("const LAT=32", base)

    print("\n=== the LATENCY constant alone ===")
    for L in (4, 8, 12, 16, 24, 32, 48):
        show(f"const LAT={L}", dict(base, LAT=L))

    print("\n=== OCC_FLOOR: cap how much low occupancy is punished ===")
    for of in (4, 6, 8, 12, 16):
        for L in (16, 32):
            show(f"cap OCC_FLOOR={of} LAT={L}", dict(base, lat="cap", OCC_FLOOR=of, LAT=L))

    print("\n=== DEPTH: a shallow launch cannot amortise steady-state latency ===")
    for rf in (2, 4, 8, 16, 32):
        for L in (16, 32):
            show(f"depth ROUNDS_FLOOR={rf} LAT={L}", dict(base, lat="depth", ROUNDS_FLOOR=rf, LAT=L))

    print("\n=== ILP credit ===")
    for a in (0.1, 0.25, 0.5, 1.0):
        show(f"ilp alpha={a}", dict(base, lat="ilp", alpha=a))

    print("\n=== joint grid, ranked by max(dense gm, moe gm) ===")
    rows = []
    for lat in ("const", "cap", "depth", "ilp"):
        for L in (4, 8, 12, 16, 24, 32):
            for cB in (8.0, 16.0, 24.0):
                for cA in (0.5, 1.0, 2.0, 4.0):
                    extras = ([{}] if lat == "const"
                              else [{"OCC_FLOOR": v} for v in (4, 6, 8, 12, 16)] if lat == "cap"
                              else [{"ROUNDS_FLOOR": v} for v in (2, 4, 8, 16, 32)] if lat == "depth"
                              else [{"alpha": v} for v in (0.1, 0.25, 0.5, 1.0)])
                    for ex in extras:
                        p = dict(base, lat=lat, LAT=L, cB=cB, cA=cA, **ex)
                        s = score(p)
                        rows.append((max(s["dense"]["gm"], s["moe"]["gm"]), s, p))
    rows.sort(key=lambda r: r[0])
    for mx, s, p in rows[:15]:
        d, m = s["dense"], s["moe"]
        ex = {k: p[k] for k in ("OCC_FLOOR", "ROUNDS_FLOOR", "alpha") if p["lat"] in
              {"cap": ("OCC_FLOOR",), "depth": ("ROUNDS_FLOOR",), "ilp": ("alpha",),
               "const": ()}.get(p["lat"], ())}
        print(f"  max={mx:.4f}  D gm={d['gm']:.4f}/w{d['worst']:.2f} "
              f"g32={d['g32']:.3f} g128={d['g128']:.3f} | "
              f"M gm={m['gm']:.4f}/w{m['worst']:.2f} sm={m['small']:.3f} big={m['big']:.3f}"
              f"   lat={p['lat']} LAT={p['LAT']} cA={p['cA']} cB={p['cB']} {ex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
