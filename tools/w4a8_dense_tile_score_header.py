#!/usr/bin/env python
"""Score the SHIPPED C++ chooser -- the one in tile_select.h -- against the recorded surface. CPU.

    docker run --rm -v <engine wt>:/engine -v <kernels wt>/fp8_wmma/torch-ext/fp8_wmma:/opt/kernels/fp8_wmma:ro \
      --entrypoint bash minisgl-rdna4:lean -lc \
      'PYTHONPATH=/opt/kernels:/engine/python python /engine/tools/w4a8_dense_tile_score_header.py'

WHY THIS EXISTS SEPARATELY FROM w4a8_tile_model_variants.py. That tool scores a PYTHON
RE-IMPLEMENTATION of the cost model, which is the right shape for exploring variants and the wrong
shape for deciding what ships: a term can be right in the harness and mistyped in the header, and
the harness would never say so. `fp8_wmma.dense_tile_explain` asks the SHIPPING header itself,
host-side, with no GPU and no kernel launch -- so this is the check that the C++ and the Python
agree, run on all 300 recorded cells rather than on a spot example.

It needs no lease. It is not a substitute for the live re-time in w4a8_dense_tile_verify.py: the
chooser ranges over a 77-tile lattice and the surface measured 36, so on the cells where the
chooser picks an UNMEASURED tile this tool can only say "unmeasured", never "good". Those cells are
exactly the ones the GPU verify exists to time. What this tool decides is the other question --
whether a header change moved a pick it should not have.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import defaultdict


def gm(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--surface", default="/engine/tools/_fixtures/dense_tile_surface_card0.csv")
    ap.add_argument("--cu", type=int, default=64)
    ap.add_argument("--baseline", default="",
                    help="a picks CSV written by an earlier run of this tool; every pick that "
                         "MOVED is printed, with the measured cost of the move")
    ap.add_argument("--out-picks", default="")
    args = ap.parse_args()

    import fp8_wmma as W
    print(f"fp8_wmma: {W.__file__}")
    if not hasattr(W, "dense_tile_explain"):
        print("REFUSING: this fp8_wmma has no dense_tile_explain; nothing to score.")
        return 2

    cells = defaultdict(dict)
    for r in csv.DictReader(open(args.surface)):
        try:
            k = (r["name"], int(r["K"]), int(r["N"]), int(r["g"]), int(r["M"]))
            cells[k][r["cand"]] = float(r["us"])
        except (ValueError, KeyError):
            continue
    cells = {k: d for k, d in cells.items() if len(d) >= 4}
    print(f"surface cells: {len(cells)}  CU={args.cu}\n")

    base = {}
    if args.baseline:
        for r in csv.DictReader(open(args.baseline)):
            base[(r["name"], int(r["K"]), int(r["N"]), int(r["g"]), int(r["M"]))] = r["pick"]

    ratios, unmeasured, moved = [], 0, []
    rows = []
    for k in sorted(cells):
        name, K, N, g, M = k
        info = W.dense_tile_explain(M, N, K, g, args.cu)
        pick = info if isinstance(info, str) else str(info)
        # dense_tile_explain returns a text block; the tile is the "BMxBN[xWN]" token in it.
        tok = None
        for w in pick.replace("\n", " ").split():
            p = w.split("x")
            if 2 <= len(p) <= 3 and all(s.isdigit() for s in p):
                tok = w
                break
        if tok is None:
            print(f"  UNPARSEABLE explain for {k}: {pick[:120]}")
            continue
        d = cells[k]
        oracle = min(d.values())
        us = d.get(tok)
        rows.append(dict(name=name, K=K, N=N, g=g, M=M, pick=tok,
                         us=("" if us is None else f"{us:.3f}"), oracle=f"{oracle:.3f}"))
        if us is None:
            unmeasured += 1
        else:
            ratios.append(us / oracle)
        if base and base.get(k) not in (None, tok):
            moved.append((k, base[k], tok, d.get(base[k]), us))

    print(f"picks scored on a MEASURED tile: {len(ratios)}   picked an UNMEASURED tile: {unmeasured}")
    if ratios:
        print(f"  geomean vs per-cell oracle = {gm(ratios):.4f}   worst = {max(ratios):.2f}")
    if base:
        print(f"\npicks that MOVED vs {args.baseline}: {len(moved)}")
        for k, b, t, ub, ut in moved[:40]:
            d = "" if (ub is None or ut is None) else f"   {ub:.1f} -> {ut:.1f} us ({ut/ub:.2f}x)"
            print(f"  {k}  {b} -> {t}{d}")
    if args.out_picks:
        with open(args.out_picks, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["name", "K", "N", "g", "M", "pick", "us", "oracle"])
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.out_picks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
