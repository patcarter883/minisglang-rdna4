#!/usr/bin/env python3
"""Restricted accept-len: only the region of a request where the prompt seed can still be IN the
drafter's 512-key window (P<512) and where it dominates that window (P<128, P<64).

Why this is THE statistic: a whole-request accept-len at max_tokens=1600 on a 3.5k prompt is ~75%
steps with P>512, where the prompt seed has been evicted from the drafter's fixed 512-key window BY
CONSTRUCTION and cannot be causal. Any tail-vs-tail difference in that region is the greedy content
lottery (each leg wanders into a different continuation), not the knob. The restricted columns are
the only ones the seed can move.

Usage: python3 tools/seedtail_window_analysis.py tools/seedtail.t*.server.log
"""
import re, sys, collections, os

pat = re.compile(r"\[spec-dbg\] uid=(\d+) c0=(\d+) dev=(\d+) conf=(-?\d+) k=(\d+) n=(-?\d+) emit=\[(.*?)\]")
NAME = {1: "CODE1", 2: "CODE2", 3: "SHORT1", 4: "SHORT2"}
print(f"{'leg':>8} {'req':>7} {'plen':>6} {'ALL':>7} {'P<512':>7} {'P<128':>7} {'P<64':>7} {'steps':>6}")
for p in sys.argv[1:]:
    if not os.path.exists(p):
        continue
    leg = os.path.basename(p).split(".")[1]
    by = collections.defaultdict(list)
    for line in open(p, errors="ignore"):
        m = pat.search(line)
        if m:
            uid, c0, dev, conf, k, n, emit = m.groups()
            by[int(uid)].append((int(c0), int(k), int(n),
                                 len([x for x in emit.split(",") if x.strip()])))
    for uid in sorted(by):
        rows = sorted(by[uid])
        if len(rows) < 20:
            continue
        c0 = rows[0][0]

        def acc(lim):
            sel = [r for r in rows if r[0] - c0 < lim]
            return sum(r[3] for r in sel) / len(sel) if sel else float("nan")
        print(f"{leg:>8} {NAME.get(uid, uid):>7} {c0:>6} {acc(10**9):>7.3f} {acc(512):>7.3f} "
              f"{acc(128):>7.3f} {acc(64):>7.3f} {len(rows):>6}")
