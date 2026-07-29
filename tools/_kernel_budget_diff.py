#!/usr/bin/env python3
"""Side-by-side per-kernel decode budget for two engines running the SAME HIP kernels.

The tight existence proof for the minisgl decode gap is vhip: vLLM 0.24 + our gdn_hip /
tail_hip / w4a8_fp8_wmma plugins reaches 82.3 tok/s where minisgl reaches 51.2. Same kernels,
same card, same model — so the delta must be in WHICH kernels get launched, HOW MANY times,
and at WHAT shape. This prints exactly that.

Usage: _kernel_budget_diff.py <traceA.gz> <stepsA> <labelA> <traceB.gz> <stepsB> <labelB>
"""
import collections
import gzip
import json
import sys


def load(path, steps):
    with gzip.open(path) as f:
        d = json.load(f)
    ev = [e for e in d["traceEvents"] if e.get("cat") == "kernel" and "dur" in e]
    dur = collections.Counter()
    cnt = collections.Counter()
    for e in ev:
        n = norm(e["name"])
        dur[n] += e["dur"] / 1000.0 / steps   # ms/step
        cnt[n] += e["count"] if "count" in e else 1
    for k in cnt:
        cnt[k] /= steps
    return dur, cnt


def norm(n):
    """Collapse a mangled kernel symbol to a comparable family name."""
    n = n.split("(")[0].replace("void ", "").strip()
    for pre in ("at::native::", "(anonymous namespace)::"):
        n = n.replace(pre, "")
    n = n.split("<")[0]
    n = n.replace(" [clone .kd]", "").strip()
    # rocBLAS assembly kernels carry their tile in the name — keep the tile, drop the rest
    if n.startswith("Cijk"):
        parts = [p for p in n.split("_") if p.startswith("MT")]
        n = "rocBLAS_" + (parts[0] if parts else "asm")
    return n or "(unnamed)"


def main():
    a_p, a_s, a_l, b_p, b_s, b_l = sys.argv[1:7]
    da, ca = load(a_p, int(a_s))
    db, cb = load(b_p, int(b_s))
    ta, tb = sum(da.values()), sum(db.values())

    print(f"{a_l}: {ta:6.2f} ms/step of GPU kernel time")
    print(f"{b_l}: {tb:6.2f} ms/step of GPU kernel time")
    print(f"delta: {ta-tb:+.2f} ms/step\n")

    keys = set(da) | set(db)
    rows = sorted(keys, key=lambda k: -(da.get(k, 0) - db.get(k, 0)))
    print(f"{'kernel':<50}{a_l+' ms':>12}{'/step':>8}{b_l+' ms':>12}{'/step':>8}{'DELTA ms':>11}")
    print("-" * 101)
    for k in rows:
        x, y = da.get(k, 0.0), db.get(k, 0.0)
        if max(x, y) < 0.02:
            continue
        print(f"{k[:49]:<50}{x:>12.3f}{ca.get(k,0):>8.1f}{y:>12.3f}{cb.get(k,0):>8.1f}{x-y:>+11.3f}")

    print()
    print(f"ONLY IN {a_l} (pure overhead vs the reference):")
    for k in rows:
        if da.get(k, 0) >= 0.05 and k not in db:
            print(f"  {da[k]:6.3f} ms/step  {ca[k]:6.1f}/step  {k[:70]}")
    print(f"ONLY IN {b_l}:")
    for k in rows:
        if db.get(k, 0) >= 0.05 and k not in da:
            print(f"  {db[k]:6.3f} ms/step  {cb[k]:6.1f}/step  {k[:70]}")


if __name__ == "__main__":
    main()
