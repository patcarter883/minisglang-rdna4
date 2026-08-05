#!/usr/bin/env python
"""Reduce the mid-band surface CSV to a winner matrix and a dispatch rule. CPU-only, no GPU lease.

    python tools/w4a8_dense_midband_analyze.py _surface/surface.csv
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict

A, B = "prefill_wmma", "wmma_tiled_tuned"


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else "_surface/surface.csv"
    t = defaultdict(dict)  # (name,K,N,M) -> {arm: us}
    meta = {}
    for r in csv.DictReader(open(path)):
        key = (r["name"], int(r["K"]), int(r["N"]), int(r["M"]))
        t[key][r["arm"]] = float(r["us"])
        meta[(r["name"], int(r["K"]), int(r["N"]))] = r["dtype"]

    shapes = sorted({(k[0], k[1], k[2]) for k in t}, key=lambda s: (s[2], s[1]))
    Ms = sorted({k[3] for k in t})

    print("WINNER MATRIX  (T = wmma_tiled_tuned, P = prefill_wmma; ratio = loser/winner)")
    print(f"{'shape':<22}{'K':>6}{'N':>7}  " + "".join(f"{m:>7}" for m in Ms))
    for name, K, N in shapes:
        cells = []
        for m in Ms:
            d = t.get((name, K, N, m), {})
            if A not in d or B not in d:
                cells.append("  -")
                continue
            w = "T" if d[B] < d[A] else "P"
            ratio = max(d[A], d[B]) / min(d[A], d[B])
            cells.append(f"{w}{ratio:5.2f}")
        print(f"{name:<22}{K:>6}{N:>7}  " + "".join(f"{c:>7}" for c in cells))

    # --- where does prefill_wmma win, and by how much? ---
    print("\nprefill_wmma WINS (mid-band M only, 17..63):")
    wins = [
        (name, K, N, m, t[(name, K, N, m)][A], t[(name, K, N, m)][B])
        for (name, K, N, m) in t
        if 17 <= m <= 63 and A in t[(name, K, N, m)] and B in t[(name, K, N, m)]
        and t[(name, K, N, m)][A] < t[(name, K, N, m)][B]
    ]
    if not wins:
        print("  NONE. prefill_wmma has no mid-band regime on any measured (M, N, K).")
    else:
        wins.sort(key=lambda w: -(w[5] / w[4]))
        print(f"  {'shape':<22}{'K':>6}{'N':>7}{'M':>5}{'prefill':>10}{'tiled':>10}{'gain':>8}")
        for name, K, N, m, a, b in wins:
            print(f"  {name:<22}{K:>6}{N:>7}{m:>5}{a:>10.1f}{b:>10.1f}{b/a:>7.2f}x")

    # --- candidate boundary: smallest N at which prefill_wmma ever wins in the mid-band ---
    byN = defaultdict(lambda: [0, 0])
    for (name, K, N, m), d in t.items():
        if not (17 <= m <= 63) or A not in d or B not in d:
            continue
        byN[N][0 if d[A] < d[B] else 1] += 1
    print("\nmid-band cells by N:   N    prefill_wins  tiled_wins")
    for N in sorted(byN):
        p, q = byN[N]
        print(f"                    {N:>6}      {p:>6}      {q:>6}")

    # --- and by M, restricted to the N where prefill can win ---
    byM = defaultdict(lambda: [0, 0])
    for (name, K, N, m), d in t.items():
        if A not in d or B not in d:
            continue
        byM[m][0 if d[A] < d[B] else 1] += 1
    print("\nall cells by M:        M    prefill_wins  tiled_wins")
    for m in sorted(byM):
        p, q = byM[m]
        print(f"                    {m:>6}      {p:>6}      {q:>6}")

    # --- score a candidate rule: prefill_wmma iff N >= NTHRESH and M <= MTHRESH ---
    print("\nRULE SCORING (mid-band cells only): total us lost vs the per-cell oracle")
    cells = [
        (name, K, N, m, d[A], d[B])
        for (name, K, N, m), d in t.items()
        if 17 <= m <= 63 and A in d and B in d
    ]
    oracle = sum(min(a, b) for *_, a, b in cells)

    def cost(rule):
        return sum((a if rule(K, N, m) else b) for name, K, N, m, a, b in cells)

    cands = [
        ("always wmma_tiled_tuned (retire prefill)", lambda K, N, m: False),
        ("always prefill_wmma (today's rule)", lambda K, N, m: True),
    ]
    for nt in (4096, 6144, 8192, 10240, 11264, 16384):
        cands.append((f"prefill iff N>={nt}", (lambda nt: lambda K, N, m: N >= nt)(nt)))
        for mt in (24, 32, 40, 48, 63):
            cands.append(
                (
                    f"prefill iff N>={nt} and M<={mt}",
                    (lambda nt, mt: lambda K, N, m: N >= nt and m <= mt)(nt, mt),
                )
            )
    # The weight (N * K/2 bytes of int4) against the 64 MB MALL: the plausible physical term, since
    # the arms differ in how they stream B and only a weight that does NOT fit the cache is streamed.
    for mb in (16, 24, 32, 48, 64, 96):
        thr = mb << 20
        cands.append(
            (f"prefill iff weight>={mb}MB", (lambda t: lambda K, N, m: N * K // 2 >= t)(thr))
        )
        for mt in (24, 32, 40, 48, 63):
            cands.append(
                (
                    f"prefill iff weight>={mb}MB and M<={mt}",
                    (lambda t, mt: lambda K, N, m: N * K // 2 >= t and m <= mt)(thr, mt),
                )
            )
    # The wave-quantization predicate. wmma_tiled_tuned tiles N by BN=128 and, in the mid-band
    # (M < BM=256), launches exactly ceil(N/128) workgroups -- one dispatch wave per 64 CUs. Its
    # occupancy of the LAST wave is the fraction below; prefill_wmma tiles N by 64 and so quantizes
    # at twice the resolution. This is the physical term the surface is non-monotonic in.
    CU = 64

    def tiled_wave_occ(N):
        tiles = -(-N // 128)
        return tiles / (CU * -(-tiles // CU))

    for occ in (0.55, 0.65, 0.72, 0.80, 0.90):
        for mt in (24, 32, 40, 48, 63):
            cands.append(
                (
                    f"prefill iff tiled last-wave occ<{occ} and M<={mt}",
                    (lambda o, mt: lambda K, N, m: tiled_wave_occ(N) < o and m <= mt)(occ, mt),
                )
            )
        cands.append(
            (
                f"prefill iff tiled last-wave occ<{occ}",
                (lambda o: lambda K, N, m: tiled_wave_occ(N) < o)(occ),
            )
        )
        for nt in (4096, 6144, 8192, 9216):
            for mt in (32, 40, 48, 56, 63):
                cands.append(
                    (
                        f"prefill iff N>={nt} and tiled occ<{occ} and M<={mt}",
                        (
                            lambda o, nt, mt: lambda K, N, m: (
                                N >= nt and tiled_wave_occ(N) < o and m <= mt
                            )
                        )(occ, nt, mt),
                    )
                )
    print("\ntiled last-wave occupancy by N (ceil(N/128) tiles over 64-CU waves):")
    for N in sorted({n for _, _, n in shapes}):
        tiles = -(-N // 128)
        print(
            f"   N={N:>6}  tiles={tiles:>4}  waves={-(-tiles//CU):>2}  occ={tiled_wave_occ(N):.3f}"
        )

    def worst(rule):
        """Worst SINGLE-CELL regression vs the oracle -- the metric that matters for a served
        layer, since an aggregate hides a 1.6x hit on one shape behind 40 shapes it gets right."""
        w, where = 1.0, None
        for name, K, N, m, a, b in cells:
            got = a if rule(K, N, m) else b
            r = got / min(a, b)
            if r > w:
                w, where = r, (name, N, m)
        return w, where

    scored = sorted(((cost(f) - oracle, nm, f) for nm, f in cands))
    for loss, nm, f in scored:
        wr, where = worst(f)
        loc = f"  worst cell {wr:.2f}x at {where[0].strip()} N={where[1]} M={where[2]}" if where else ""
        print(f"  {loss:>10.0f} us over oracle ({100*loss/oracle:5.2f}%)   {nm}{loc}")

    print("\nmid-band cells by WEIGHT MB (N*K/2):  MB   prefill_wins  tiled_wins")
    byW = defaultdict(lambda: [0, 0])
    for (name, K, N, m), d in t.items():
        if not (17 <= m <= 63) or A not in d or B not in d:
            continue
        byW[round(N * K / 2 / 1e6, 1)][0 if d[A] < d[B] else 1] += 1
    for w in sorted(byW):
        p, q = byW[w]
        print(f"                              {w:>8}      {p:>6}      {q:>6}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
