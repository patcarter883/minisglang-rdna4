#!/usr/bin/env python
"""Fold the per-leg JSON blobs from ab_tile_serve_bench.py into one BEFORE/AFTER table.

Medians of the recorded repeats, plus the min/max spread, so a reader can see whether a ratio
clears the run-to-run noise rather than having to take the median on trust.

  python tools/ab_tile_serve_report.py BEFORE.json AFTER.json [--name QWEN]
"""
from __future__ import annotations

import argparse
import json
import statistics


def _fmt(vals):
    return f"{statistics.median(vals):8.2f} [{min(vals):.2f}-{max(vals):.2f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("before")
    ap.add_argument("after")
    ap.add_argument("--name", default="")
    args = ap.parse_args()
    B = json.load(open(args.before))
    A = json.load(open(args.after))

    print(f"\n===== {args.name} BEFORE={B['label']} vs AFTER={A['label']} "
          f"(medians of {B['reps']}/{A['reps']} reps; [min-max]) =====")
    print(f"{'cell':>16} {'metric':>9} {'BEFORE':>22} {'AFTER':>22} {'AFTER/BEFORE':>13}")
    for wl, metrics in (("prefill", ("ttft_ms", "tok_s")), ("decode", ("tpot_ms", "tok_s"))):
        for cell in sorted(set(B[wl]) | set(A[wl])):
            for m in metrics:
                b = B[wl].get(cell, {}).get(m)
                a = A[wl].get(cell, {}).get(m)
                if not b or not a:
                    print(f"{wl + ' ' + cell:>16} {m:>9} "
                          f"{(_fmt(b) if b else 'ABSENT'):>22} "
                          f"{(_fmt(a) if a else 'ABSENT'):>22} {'--':>13}")
                    continue
                mb, ma = statistics.median(b), statistics.median(a)
                # ratio oriented so >1 always means AFTER is BETTER
                r = (mb / ma) if m.endswith("_ms") else (ma / mb)
                print(f"{wl + ' ' + cell:>16} {m:>9} {_fmt(b):>22} {_fmt(a):>22} {r:12.4f}x")


if __name__ == "__main__":
    main()
