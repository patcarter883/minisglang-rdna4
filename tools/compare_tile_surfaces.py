#!/usr/bin/env python3
"""Compare two tile-surface CSVs cell by cell — the compiler A/B.

Same source, same shapes, same tiles; the only difference is the toolchain that compiled the
kernel. So a per-cell ratio is a clean read on what the compiler did, and the SPREAD matters as
much as the mean: a compiler that helps the winning tile and hurts a losing one changes which tile
the chooser should pick, which is a different (and more expensive) fact than "everything got 5%
faster".

    python compare_sweeps.py base.csv new.csv [--label-a rocm7.2.1 --label-b rocm7.14]
"""
from __future__ import annotations
import argparse, csv, math
from collections import defaultdict


def load(path):
    """(name, M, candidate) -> us. Schema: name,K,N,g,dtype,M,cand,us,reps,R,dev,cu,card,..."""
    out = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            try:
                us = float(r["us"])
            except (KeyError, ValueError):
                continue
            out[(r["name"].strip(), int(r["M"]), r["cand"].strip())] = us
    return out


def geo(xs):
    xs = [x for x in xs if x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--label-a", default="A"); ap.add_argument("--label-b", default="B")
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()
    A, B = load(args.a), load(args.b)
    common = sorted(set(A) & set(B))
    if not common:
        print("no overlapping cells — check the two CSVs are the same sweep"); return 1
    print(f"{len(common)} common cells  ({len(A)} in {args.label_a}, {len(B)} in {args.label_b})")

    # Speedup = A/B: >1 means B (the new toolchain) is FASTER.
    sp = {k: A[k] / B[k] for k in common if B[k] > 0}
    print(f"\noverall geomean speedup {args.label_b} vs {args.label_a}: {geo(list(sp.values())):.4f}")

    print(f"\n-- by candidate (geomean, n) --")
    by_c = defaultdict(list)
    for (s, m, c), v in sp.items():
        by_c[c].append(v)
    for c, vs in sorted(by_c.items(), key=lambda kv: -geo(kv[1]))[: args.top]:
        print(f"   {c:<26} {geo(vs):6.3f}  n={len(vs)}")
    print("   ...")
    for c, vs in sorted(by_c.items(), key=lambda kv: -geo(kv[1]))[-5:]:
        print(f"   {c:<26} {geo(vs):6.3f}  n={len(vs)}")

    print(f"\n-- by shape (geomean) --")
    by_s = defaultdict(list)
    for (s, m, c), v in sp.items():
        by_s[s].append(v)
    for s, vs in sorted(by_s.items(), key=lambda kv: -geo(kv[1])):
        print(f"   {s:<22} {geo(vs):6.3f}  n={len(vs)}")

    # DOES THE ORACLE MOVE? The expensive question: if the best tile per cell changes, the chooser
    # has to be refitted per toolchain, not just rescaled.
    def oracle(D):
        best = {}
        for (s, m, c), us in D.items():
            if c in ("auto",) or c.startswith("prefill_"):
                continue
            k = (s, m)
            if k not in best or us < best[k][1]:
                best[k] = (c, us)
        return best
    oa, ob = oracle(A), oracle(B)
    keys = sorted(set(oa) & set(ob))
    moved = [k for k in keys if oa[k][0] != ob[k][0]]
    print(f"\n-- oracle tile --")
    print(f"   {len(moved)}/{len(keys)} cells change their best tile under {args.label_b}")
    # How much does it COST to keep A's oracle tile on B?
    loss = []
    for k in keys:
        t = oa[k][0]
        if (k[0], k[1], t) in B:
            loss.append(B[(k[0], k[1], t)] / ob[k][1])
    if loss:
        print(f"   keeping {args.label_a}'s oracle tile on {args.label_b}: "
              f"geomean {geo(loss):.4f}x, worst {max(loss):.3f}x")
    for k in moved[:10]:
        print(f"     {k[0]:<22} M={k[1]:<5} {oa[k][0]:>12} -> {ob[k][0]:<12} "
              f"({A[k[0],k[1],oa[k][0]]:.1f} -> {ob[k][1]:.1f} us)")

    # And what production actually gets.
    au = [sp[k] for k in sp if k[2] == "auto"]
    if au:
        print(f"\n-- the chooser's own pick ('auto'): geomean {geo(au):.4f}, n={len(au)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
