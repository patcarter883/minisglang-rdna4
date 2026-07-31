#!/usr/bin/env python3
"""Analyze the expert-divergence probe dumps (tools/route_stats_*.rank*.json).

THE QUESTION (CONTINUANCE §2 finding 3): a spec-verify batch is qlen=K+1 rows into a 256-expert
top-8 MoE. If those rows route to near-DISJOINT expert sets, the grouped GEMM genuinely has to
stream ~8*qlen expert slabs and the verify cost is inherent. If they overlap heavily, the measured
14x MoE cost at M=17 is a kernel/alignment bug.

Reads the raw rows and reports, per M (batch height):
  pairs        M*top_k, the (token,expert) assignments
  distinct     |union of expert ids| -- what an ideal kernel must stream
  blocks       ntp/block_m, what moe_align ACTUALLY launches
  ideal        sum_e ceil(rows_e/block_m), the best alignment could do
  cost vs M=1  distinct/8 -- the predicted MoE slowdown IF cost ~ expert slabs streamed
No GPU, no torch: pure json. Host python3 is fine.
"""
import glob
import json
import statistics as st
import sys


def load(pat):
    rows = []
    for p in sorted(glob.glob(pat)):
        with open(p) as f:
            rows += json.load(f)
    return rows


def report(tag, rows):
    by_m = {}
    for r in rows:
        by_m.setdefault(r["M"], []).append(r)
    print(f"\n### {tag}  ({len(rows)} MoE calls, E={rows[0]['E']}, "
          f"top_k={rows[0]['top_k']}, block_m={rows[0]['block_m']})")
    print(f"{'M':>4} {'n':>5} {'pairs':>6} {'distinct':>18} {'blocks':>8} {'ideal':>7} "
          f"{'blk/dist':>9} {'dist/pairs':>11}")
    for M in sorted(by_m):
        g = by_m[M]
        d = [r["distinct"] for r in g]
        b = [r["blocks"] for r in g]
        i = [r["ideal_blocks"] for r in g]
        pairs = g[0]["pairs"]
        dm = st.mean(d)
        print(f"{M:>4} {len(g):>5} {pairs:>6} "
              f"{dm:>8.1f} [{min(d):>3}-{max(d):>3}] {st.mean(b):>8.1f} {st.mean(i):>7.1f} "
              f"{st.mean(b)/dm:>9.2f} {dm/pairs:>11.3f}")
    return by_m


def union_curve(tag, by_m):
    """Marginal new experts per added row: the ONLY thing that says whether drafts share experts."""
    for M in sorted(by_m):
        if M < 4:
            continue
        g = by_m[M]
        n = len(g[0]["union_curve"])
        avg = [st.mean(r["union_curve"][j] for r in g) for j in range(n)]
        marg = [avg[0]] + [avg[j] - avg[j - 1] for j in range(1, n)]
        tk = g[0]["top_k"]
        print(f"\n{tag} M={M}: union after row i (top_k={tk}, disjoint would be {tk}*i)")
        print("   i:  " + " ".join(f"{j+1:>5}" for j in range(n)))
        print("   |U|:" + " ".join(f"{v:>5.1f}" for v in avg))
        print("   new:" + " ".join(f"{v:>5.2f}" for v in marg))
        print(f"   -> row 1 brings {marg[0]:.2f} experts, the LAST row brings {marg[-1]:.2f} "
              f"({100*marg[-1]/tk:.0f}% of top_k are new)")


if __name__ == "__main__":
    base = sys.argv[1] if len(sys.argv) > 1 else "/home/pat/code/minisgl-rdna4-specod/tools"
    for tag in ("plain", "k7", "k15"):
        files = f"{base}/route_stats_{tag}.rank0.json"
        rows = load(files)
        if not rows:
            print(f"\n### {tag}: no data")
            continue
        by_m = report(tag, rows)
        union_curve(tag, by_m)
