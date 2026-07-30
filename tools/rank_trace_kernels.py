"""Rank kernels by GPU time from a MINISGL_PROFILE Chrome trace.

The engine's built-in torch-profiler window is the right tool for "which kernel dominates a decode
step" on this box: rocprofv3 --pmc hangs, a --collection-period window is fragile against a short
bench, and rpd's LD_PRELOAD does not reach the server (the engine launches it under setsid). This
reads the trace the engine already knows how to emit.
"""
import collections
import json
import sys

path = sys.argv[1]
d = json.load(open(path))
ev = d["traceEvents"] if isinstance(d, dict) else d

gpu = collections.Counter()
cnt = collections.Counter()
cats = collections.Counter()
for e in ev:
    if e.get("ph") != "X":
        continue
    cat = (e.get("cat") or "").lower()
    cats[cat] += 1
    if "kernel" not in cat and "gpu" not in cat:
        continue
    name = e.get("name", "?")
    gpu[name] += e.get("dur", 0)
    cnt[name] += 1

if not gpu:
    print("no GPU/kernel-category events; categories present:")
    for c, n in cats.most_common(12):
        print(f"  {c or '<none>':24s} {n}")
    sys.exit(1)

tot = sum(gpu.values())
print(f"GPU total over the window: {tot/1000:.2f} ms across {sum(cnt.values())} dispatches")
print(f"{'share':>7} {'us_total':>10} {'n':>7} {'us_each':>9}  kernel")
for k, v in gpu.most_common(20):
    print(f"{100*v/tot:6.2f}% {v:10.1f} {cnt[k]:7d} {v/cnt[k]:9.2f}  {k[:86]}")
