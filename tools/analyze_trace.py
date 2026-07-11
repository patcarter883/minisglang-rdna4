#!/usr/bin/env python
"""Direct chrome-trace analyzer for a minisgl decode/verify profile: aggregate GPU kernel time,
split COMM (rccl/all-reduce/all-gather) vs COMPUTE, and list the top kernels. Answers 'where does the
per-step time go, comm or compute' without TraceLens (which can't parse the gfx1201 torch trace).

  python tools/analyze_trace.py tools/tp2_results/mtp_prof.pt.trace.json [n_steps]
"""
import json
import sys
from collections import defaultdict

path = sys.argv[1]
n_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 50

with open(path) as f:
    data = json.load(f)
evs = data.get("traceEvents", data if isinstance(data, list) else [])

# Categories present (diagnostic)
cats = defaultdict(int)
for e in evs:
    if isinstance(e, dict):
        cats[e.get("cat", "?")] += 1
print("categories:", dict(sorted(cats.items(), key=lambda x: -x[1])[:12]))

# GPU kernel events: cat == 'kernel' (torch ROCm). Aggregate dur (us) by name.
COMM_HINTS = ("rccl", "nccl", "allreduce", "all_reduce", "allgather", "all_gather",
              "reduce_scatter", "reducescatter", "ncclDevKernel", "ncclKernel", "AllReduce",
              "AllGather", "collective", "c10d")
by_name = defaultdict(lambda: [0.0, 0])   # name -> [total_us, count]
gpu_total = 0.0
comm_total = 0.0
for e in evs:
    if not isinstance(e, dict) or e.get("ph") != "X":
        continue
    if e.get("cat") != "kernel":
        continue
    dur = float(e.get("dur", 0))
    name = e.get("name", "?")
    by_name[name][0] += dur
    by_name[name][1] += 1
    gpu_total += dur
    if any(h.lower() in name.lower() for h in COMM_HINTS):
        comm_total += dur

if gpu_total == 0:
    print("\nNo cat=='kernel' events; sampling names on GPU-like streams instead:")
    # fallback: any event with dur on a tid that looks like a stream
    for e in evs[:5]:
        print("  sample:", {k: e.get(k) for k in ("name", "cat", "ph", "dur", "pid", "tid")})
    sys.exit(0)

comp_total = gpu_total - comm_total
print(f"\n=== GPU-active kernel time over the {n_steps}-step window ===")
print(f"  total GPU kernel time : {gpu_total/1e3:8.1f} ms  ({gpu_total/n_steps:7.1f} us/step)")
print(f"  COMM  (collectives)   : {comm_total/1e3:8.1f} ms  ({100*comm_total/gpu_total:4.1f}%)  "
      f"({comm_total/n_steps:7.1f} us/step)")
print(f"  COMPUTE               : {comp_total/1e3:8.1f} ms  ({100*comp_total/gpu_total:4.1f}%)  "
      f"({comp_total/n_steps:7.1f} us/step)")

print(f"\n=== top 25 kernels by total time (us/step, count/step) ===")
print(f"{'us/step':>9} {'%':>5} {'n/step':>7}  kernel")
for name, (tot, cnt) in sorted(by_name.items(), key=lambda x: -x[1][0])[:25]:
    tag = "COMM" if any(h.lower() in name.lower() for h in COMM_HINTS) else ""
    print(f"{tot/n_steps:9.1f} {100*tot/gpu_total:5.1f} {cnt/n_steps:7.1f}  {tag:4} {name[:80]}")
