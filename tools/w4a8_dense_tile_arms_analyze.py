#!/usr/bin/env python
"""Reduce the tile-EQUALIZED arm sweep to the two questions it exists to answer. CPU-only.

    python tools/w4a8_dense_tile_arms_analyze.py _tile_arms.csv

(a) at the SAME tile, is any arm algorithmically ahead?  (b) at each arm's OWN best tile, does any
arm still earn its maintenance? These are different questions and the shipped three-way dispatch
answered neither -- it compared arms at their own frozen, DIFFERENT 256x128 defaults.
"""
from __future__ import annotations
import csv, math, sys
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "_tile_arms.csv"
t = defaultdict(dict)
for r in csv.DictReader(open(path)):
    t[(r["name"], int(r["N"]), int(r["M"]))][r["cand"]] = float(r["us"])
keys = sorted(t)
gm = lambda x: math.exp(sum(map(math.log, x)) / len(x))

TILES = ["16x128", "32x128", "64x128", "128x128", "256x128", "256x256", "384x128", "512x128"]

print("(a) TILE-EQUALIZED: tiled_tuned vs ashuffle at the SAME tile")
print(f"    {'tile':>9}{'tiled wins':>12}{'ash wins':>10}{'geomean ash/tiled':>20}{'ash best':>10}")
for tl in TILES:
    rs, tw, aw = [], 0, 0
    for k in keys:
        a, b = t[k].get(f"tiled@{tl}"), t[k].get(f"ash@{tl}")
        if a and b:
            rs.append(b / a); tw += a < b; aw += b < a
    if rs:
        print(f"    {tl:>9}{tw:>12}{aw:>10}{gm(rs):>19.3f}x{min(rs):>9.3f}x")
allr = [t[k][f"ash@{tl}"] / t[k][f"tiled@{tl}"] for k in keys for tl in TILES
        if f"ash@{tl}" in t[k] and f"tiled@{tl}" in t[k]]
print(f"    OVERALL at equal tile: ash/tiled geomean {gm(allr):.3f}x over {len(allr)} pairs "
      f"(>1 = tiled faster)")

print("\n(b) EACH ARM AT ITS OWN BEST TILE")
best = {}
for k in keys:
    d = {}
    for arm, pref in (("tiled", "tiled@"), ("ash", "ash@"), ("prefill", "prefill")):
        sub = {c: v for c, v in t[k].items() if c.startswith(pref)}
        if sub:
            bc = min(sub, key=sub.get); d[arm] = (bc, sub[bc])
    best[k] = d
wins = defaultdict(int); marg = defaultdict(list)
for k, d in best.items():
    w = min(d, key=lambda a: d[a][1]); wins[w] += 1
    for a in d:
        marg[a].append(d[a][1] / d[w][1])
for a in ("tiled", "ash", "prefill"):
    if a in marg:
        print(f"    {a:>8}: wins {wins[a]:>3}/{len(best)} cells;  geomean cost vs the per-cell "
              f"best arm {gm(marg[a]):.3f}x;  worst {max(marg[a]):.2f}x")
print("\n    cells where an arm OTHER than tiled wins at own-best:")
n = 0
for k, d in sorted(best.items()):
    w = min(d, key=lambda a: d[a][1])
    if w != "tiled":
        n += 1
        print(f"      {k[0]:<20} N={k[1]:<7} M={k[2]:<6} {w} {d[w][0]} {d[w][1]:.1f} us vs "
              f"tiled {d['tiled'][0]} {d['tiled'][1]:.1f} us  ({d['tiled'][1]/d[w][1]:.3f}x)")
if n == 0:
    print("      NONE.")

print("\n(c) prefill_wmma's small-M tile ON vs OFF -- was its mid-band edge the TILE?")
rs = [(k, t[k]["prefill"], t[k]["prefill:smallm_off"]) for k in keys
      if "prefill" in t[k] and "prefill:smallm_off" in t[k]]
mid = [(k, a, b) for k, a, b in rs if k[2] < 128]
print(f"    mid-band cells (M<128): small-M tile ON is {gm([b/a for _, a, b in mid]):.3f}x faster "
      f"than OFF (geomean over {len(mid)} cells)")
print(f"    and at its OWN best tile prefill_wmma is still {gm(marg['prefill']):.3f}x off the "
      f"per-cell best arm.")
