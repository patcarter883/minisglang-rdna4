#!/usr/bin/env python
"""Reduce the tile surface CSV to a tile rule, and re-ask the ARM question. CPU-only, no GPU lease.

    python tools/w4a8_dense_tile_analyze.py _tile_surface.csv [--cu 64]

Three questions, in order:
  1. What is the ORACLE tile per (shape, M), and how much is today's hard-wired 256x128 leaving?
  2. Which variables does the oracle move with? A rule is only allowed a term it can name.
  3. With the rule's tile on `wmma_tiled_tuned`, do `prefill_wmma` / `prefill_wmma_ashuffle` win
     anything -- and does `prefill_wmma`'s advantage survive VLLM_W4A8_DENSE_SMALLM_OFF=1?
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


def parse_tile(c: str):
    """(BM, BN, WARPS_N). The surface carries "BMxBN" AND "BMxBNxWN" since the WARPS_N axis was
    swept; a two-field unpack raised on every WN row, which is why this tool stopped running at
    all against the card-0 fixtures."""
    p = c.split("x")
    return int(p[0]), int(p[1]), int(p[2]) if len(p) > 2 else 1


def legal(bm, bn, g):
    """LDS fit. Delegated to _w4a8_tile_policy so this file cannot carry a stale copy of the
    formula — the previous inline `(bm + bn) * (g + 8)` under-reports by 8x at g=16, because the
    staging round is GROUPS_PER_STAGE quant groups deep and there is a static scale array too."""
    return _policy.tile_lds(bm, bn, g) <= _policy.LDS_BUDGET


# ---------------------------------------------------------------- candidate rules
def rule_shipped(M, N, K, g, cu, tiles):
    return "256x128"


def tname(t) -> str:
    return f"{t[0]}x{t[1]}" if t[2] == 1 else f"{t[0]}x{t[1]}x{t[2]}"


def _pick(tiles, g, want_bm, want_bn):
    """Nearest instantiated tile at or above (want_bm, want_bn), preferring exact BM.

    These pre-cost-model heuristics have no WARPS_N opinion, so they pick among the wn=1 tiles and
    are scored honestly against a surface that also contains wn>1 -- rather than being silently
    handed a WN choice they never expressed."""
    ok = [t for t in tiles if legal(t[0], t[1], g) and t[2] == 1]
    cand = [t for t in ok if t[0] >= want_bm and t[1] >= want_bn]
    if not cand:
        cand = ok
    cand.sort(key=lambda t: (t[0] * t[1], t[0]))
    return tname(cand[0])


def rule_bm_from_m(M, N, K, g, cu, tiles):
    """BM covers M (no padded warps); BN fixed at the shipped 128."""
    return _pick(tiles, g, min(256, max(16, 1 << (max(M, 16) - 1).bit_length())), 128)


def make_rule_wg(min_wg_mult=1.0, prefer_bn=128):
    """BM covers M; then BN is shrunk until ceil(M/BM)*ceil(N/BN) covers the CU count."""

    def r(M, N, K, g, cu, tiles):
        bm_want = min(256, max(16, 1 << (max(M, 16) - 1).bit_length()))
        ok = sorted([t for t in tiles if legal(t[0], t[1], g) and t[2] == 1])
        best = None
        for bm, bn, _wn in ok:
            if bm < bm_want and bm * 2 <= bm_want:
                continue
            wg = -(-M // bm) * (-(-N // bn))
            if wg < cu * min_wg_mult:
                continue
            # among the legal ones prefer the largest BN (WMMA ILP), then BM closest to bm_want
            key = (-bn, abs(bm - bm_want))
            if best is None or key < best[0]:
                best = (key, (bm, bn, 1))
        if best is None:
            return _pick(ok, g, bm_want, 32)
        return tname(best[1])

    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--cu", type=int, default=64)
    ap.add_argument("--top", type=int, default=4)
    args = ap.parse_args()

    t = defaultdict(dict)  # (name,K,N,g,M) -> {cand: us}
    for r in csv.DictReader(open(args.csv)):
        key = (r["name"], int(r["K"]), int(r["N"]), int(r["g"]), int(r["M"]))
        t[key][r["cand"]] = float(r["us"])

    tiles = sorted({parse_tile(c) for d in t.values() for c in d if c not in ARMS})
    shapes = sorted({k[:4] for k in t}, key=lambda s: (s[2], s[1]))
    Ms = sorted({k[4] for k in t})

    # ---------------------------------------------------------------- 1. oracle
    print("=" * 110)
    print("ORACLE TILE per (shape, M)   [tiles only; arms excluded]   "
          "'x' = 256x128 (today's hard-wired default) is oracle")
    print("=" * 110)
    print(f"{'shape':<20}{'K':>6}{'N':>7}{'g':>4}  " + "".join(f"{m:>10}" for m in Ms))
    oracle = {}
    for name, K, N, g in shapes:
        cells = []
        for m in Ms:
            d = {c: v for c, v in t.get((name, K, N, g, m), {}).items() if c not in ARMS}
            if not d:
                cells.append("-")
                continue
            b = min(d, key=d.get)
            oracle[(name, K, N, g, m)] = (b, d[b])
            ship = d.get("256x128")
            cells.append(f"{b}" + (f"/{ship/d[b]:.2f}" if ship else ""))
        print(f"{name:<20}{K:>6}{N:>7}{g:>4}  " + "".join(f"{c:>10}" for c in cells))
    print("\n  cell = oracle tile / (256x128 time  ÷ oracle time), i.e. what the hard-wire costs.")

    # ---------------------------------------------------------------- 2. rules
    rules = {
        "256x128 (shipped)": rule_shipped,
        "BM<-M, BN=128": rule_bm_from_m,
        "BM<-M then BN<-CU (1.0x)": make_rule_wg(1.0),
        "BM<-M then BN<-CU (2.0x)": make_rule_wg(2.0),
    }
    print("\n" + "=" * 110)
    print(f"RULE SCORING vs the per-cell oracle   (CU={args.cu}, {len(oracle)} cells)")
    print("=" * 110)
    print(f"{'rule':<28}{'geomean vs oracle':>20}{'worst cell':>14}{'>1.10x cells':>14}"
          f"{'exact hits':>12}")
    scored = {}
    for rname, fn in rules.items():
        ratios, worst, worstkey = [], 1.0, None
        exact = 0
        for key, (ob, ous) in oracle.items():
            name, K, N, g, M = key
            pick = fn(M, N, K, g, args.cu, tiles)
            d = t[key]
            if pick not in d:
                continue
            rr = d[pick] / ous
            ratios.append(rr)
            if pick == ob:
                exact += 1
            if rr > worst:
                worst, worstkey = rr, (name, N, M, pick, ob)
        gm = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
        bad = sum(1 for r in ratios if r > 1.10)
        scored[rname] = (gm, worst, worstkey)
        print(f"{rname:<28}{gm:>19.4f}x{worst:>13.2f}x{bad:>14}{exact:>12}")
    for rname, (gm, worst, wk) in scored.items():
        if wk:
            print(f"  worst for {rname!r}: {wk[0]} N={wk[1]} M={wk[2]} picked {wk[3]}, "
                  f"oracle {wk[4]} ({worst:.2f}x)")

    # ---------------------------------------------------------------- 3. arms
    print("\n" + "=" * 110)
    print("THE ARM QUESTION -- do the prefill arms beat the ORACLE tile anywhere?")
    print("=" * 110)
    for arm in ARMS:
        wins, cells, best = [], 0, 1.0
        for key, (ob, ous) in oracle.items():
            d = t[key]
            if arm not in d:
                continue
            cells += 1
            if d[arm] < ous:
                wins.append((key, ous / d[arm], ob))
                best = max(best, ous / d[arm])
        print(f"\n  {arm}: wins {len(wins)}/{cells} cells vs the oracle tile; best margin {best:.3f}x")
        wins.sort(key=lambda w: -w[1])
        for (name, K, N, g, M), marg, ob in wins[:12]:
            print(f"      {name:<20} N={N:<7} M={M:<6} {marg:.3f}x  (oracle tile {ob})")

    # prefill_wmma with its small-M tile ON vs OFF -- is the arm's edge the TILE?
    print("\n  prefill_wmma small-M tile ON vs OFF (VLLM_W4A8_DENSE_SMALLM_OFF=1):")
    print(f"    {'shape':<20}{'N':>8}{'M':>6}{'smallm ON':>12}{'OFF':>10}{'ON gain':>10}")
    rows = []
    for key in sorted(oracle):
        d = t[key]
        if "prefill_wmma" in d and "prefill_wmma:smallm_off" in d:
            rows.append((key, d["prefill_wmma"], d["prefill_wmma:smallm_off"]))
    for (name, K, N, g, M), on, off in rows:
        if M > 128:
            continue
        print(f"    {name:<20}{N:>8}{M:>6}{on:>12.1f}{off:>10.1f}{off/on:>9.2f}x")

    # ---------------------------------------------------------------- 4. where is the tile knee?
    print("\n" + "=" * 110)
    print("ORACLE BM vs M  (does BM track M, and where does 'small M' end?)")
    print("=" * 110)
    bym = defaultdict(list)
    for (name, K, N, g, M), (ob, us) in oracle.items():
        bym[M].append(parse_tile(ob))
    print(f"{'M':>6}  {'BM histogram':<40}{'BN histogram':<24}{'WARPS_N histogram'}")
    for M in sorted(bym):
        bmh, bnh, wnh = defaultdict(int), defaultdict(int), defaultdict(int)
        for a, b, w in bym[M]:
            bmh[a] += 1
            bnh[b] += 1
            wnh[w] += 1
        s1 = " ".join(f"{k}:{v}" for k, v in sorted(bmh.items()))
        s2 = " ".join(f"{k}:{v}" for k, v in sorted(bnh.items()))
        s3 = " ".join(f"{k}:{v}" for k, v in sorted(wnh.items()))
        print(f"{M:>6}  {s1:<40}{s2:<24}{s3}")

    print("\nORACLE BN vs N  (does BN track the dispatch width?)")
    byn = defaultdict(lambda: defaultdict(int))
    for (name, K, N, g, M), (ob, us) in oracle.items():
        byn[N][parse_tile(ob)[1]] += 1
    print(f"{'N':>8}  {'ceil(N/128)/CU':>16}  BN histogram")
    for N in sorted(byn):
        s = " ".join(f"{k}:{v}" for k, v in sorted(byn[N].items()))
        print(f"{N:>8}  {(-(-N//128))/args.cu:>16.2f}  {s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
