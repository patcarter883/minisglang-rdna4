#!/usr/bin/env python3
"""Analyze vLLM torch-profiler traces (*.pt.trace.json.gz) from run_profile.sh — HOST-side, no GPU.

Reports the numbers that settle "launch/gap-bound vs kernel-bound" and rank the split targets:
  - GPU-active fraction  (union of GPU-kernel intervals / wall)  -> low = idle-gap/launch bound
  - dispatches/token     (GPU kernel count / decode tokens)      -> the launch-storm size
  - top kernels by total GPU time                                -> WHICH kernel to split first
Picks the busiest rank. Usage: analyze_trace.py <trace_dir> [decode_tokens]
For deeper roofline/Python->GPU linkage use TraceLens on the same files (see CLAUDE.md)."""
import glob, gzip, json, os, sys

d = sys.argv[1] if len(sys.argv) > 1 else "/tmp/claude-1000/-home-pat-code-minisgl-rdna4/prof"
NTOK = int(sys.argv[2]) if len(sys.argv) > 2 else int(os.environ.get("PROF_DECODE", "30"))

files = sorted(glob.glob(os.path.join(d, "**", "*.pt.trace.json*"), recursive=True))
if not files:
    print("no *.pt.trace.json* under", d); sys.exit(1)


def load(f):
    op = gzip.open if f.endswith(".gz") else open
    with op(f, "rt") as fh:
        return json.load(fh)


def kernels(ev):
    # torch-profiler Chrome trace: GPU kernels carry cat=="kernel"; each has ts,dur (microseconds)
    out = []
    for e in ev:
        if e.get("ph") != "X":
            continue
        cat = (e.get("cat") or "").lower()
        if cat in ("kernel", "gpu_memcpy", "gpu_memset"):
            out.append((e["ts"], e.get("dur", 0), e.get("name", "?")))
    return out


def union_busy(iv):
    iv = sorted(iv)
    busy = 0; cs = ce = None
    for s, dur, _ in iv:
        e = s + dur
        if cs is None: cs, ce = s, e
        elif s <= ce: ce = max(ce, e)
        else: busy += ce - cs; cs, ce = s, e
    if cs is not None: busy += ce - cs
    return busy


best = None
for f in files:
    try:
        ks = kernels(load(f).get("traceEvents", []))
    except Exception as ex:
        print("  skip", os.path.basename(f), ex); continue
    if not ks: continue
    span = (max(s + d for s, d, _ in ks) - min(s for s, _, _ in ks))
    if best is None or len(ks) > len(best[1]):
        best = (f, ks, span)

if best is None:
    print("no GPU kernel events found in any trace (is enforce_eager hiding them? check cat names)")
    # dump distinct cats to help
    for f in files[:1]:
        cats = {}
        for e in load(f).get("traceEvents", []):
            cats[e.get("cat")] = cats.get(e.get("cat"), 0) + 1
        print("  cats in", os.path.basename(f), ":", cats)
    sys.exit(1)

f, ks, span = best
busy = union_busy(ks)
wall_ms, busy_ms = span / 1e3, busy / 1e3
print(f"busiest rank trace: {os.path.relpath(f, d)}  ({len(files)} rank files)")
print(f"\n=== DECODE window ({NTOK} tokens) ===")
print(f"  GPU kernel dispatches : {len(ks)}   -> {len(ks)/NTOK:.1f} dispatches/token")
print(f"  wall (kernel span)    : {wall_ms:8.2f} ms  = {wall_ms/NTOK:.3f} ms/token")
print(f"  GPU-active (union)    : {busy_ms:8.2f} ms  = {100*busy/span:5.1f}% busy  -> {100-100*busy/span:5.1f}% IDLE-GAP")
agg = {}
for _, dur, n in ks:
    key = n.split("(")[0].strip()[:64]
    agg[key] = agg.get(key, 0) + dur
print("\n  top GPU kernels by total time (ms, % of active) — SPLIT TARGETS, most-costly first:")
for k, v in sorted(agg.items(), key=lambda x: -x[1])[:20]:
    print(f"    {v/1e3:8.3f}  {100*v/busy:5.1f}%  {k}")
