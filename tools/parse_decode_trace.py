"""Parse a minisgl MINISGL_PROFILE chrome trace into the decode-latency localization split.

Host-side, no GPU. Computes per-decode-step: wall time, GPU-busy (union of kernel intervals on the
device streams), GPU-idle/host-gap, and a top-kernel attribution bucketed into
{LM-head, GDN, MoE, attention, allreduce/allgather, sampler, other}.

  python tools/parse_decode_trace.py tools/loads/decode_trace.json [--steps 50]
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import defaultdict


def load(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        d = json.load(f)
    return d["traceEvents"] if isinstance(d, dict) else d


def union_busy(intervals):
    """Total length of the union of [start,end) intervals (microseconds)."""
    if not intervals:
        return 0.0
    intervals.sort()
    total = 0.0
    cs, ce = intervals[0]
    for s, e in intervals[1:]:
        if s > ce:
            total += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    total += ce - cs
    return total


def bucket(name):
    n = name.lower()
    if any(k in n for k in ("dense_gemm", "lmhead", "lm_head")):
        # dense_gemm is used by BOTH lm_head and (some) dense proj; tag broadly, refine by size later
        return "gemm(dense/lmhead)"
    if any(k in n for k in ("gdn", "conv_gated", "ssm", "recurrent", "chunk_gated", "gated_delta")):
        return "GDN"
    if any(k in n for k in ("moe", "w4a16", "w4a8", "grouped", "expert", "topk", "silu", "swiglu")):
        return "MoE"
    if any(k in n for k in ("attn", "attention", "flash", "paged", "rope", "rotary")):
        return "attention"
    if any(k in n for k in ("all_reduce", "allreduce", "all_gather", "allgather", "nccl", "rccl", "custom_ar", "ccl")):
        return "comm(ar/ag)"
    if any(k in n for k in ("sample", "argmax", "softmax", "multinomial", "topk_topp")):
        return "sampler"
    if any(k in n for k in ("elementwise", "vectorized", "norm", "rms", "add", "cast", "copy", "memcpy", "memset")):
        return "elementwise/norm/copy"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--steps", type=int, default=50, help="MINISGL_PROFILE_STEPS window size")
    args = ap.parse_args()

    events = load(args.path)
    # GPU kernel events: torch profiler tags device work cat in {"kernel","gpu_memcpy","gpu_memset"}.
    kinds = {"kernel", "gpu_memcpy", "gpu_memset"}
    gpu = [e for e in events if e.get("cat") in kinds and e.get("ph") == "X" and "dur" in e]
    if not gpu:
        # fallback: some exporters use cat "Kernel"
        gpu = [e for e in events if str(e.get("cat", "")).lower().startswith("kernel") and "dur" in e]
    if not gpu:
        print("NO GPU KERNEL EVENTS FOUND; event cats present:",
              sorted({str(e.get('cat')) for e in events})[:20])
        return 1

    # group by device stream (pid,tid) to union per-stream then combine
    ts0 = min(e["ts"] for e in gpu)
    ts1 = max(e["ts"] + e["dur"] for e in gpu)
    window = ts1 - ts0

    # busy = union across ALL gpu streams (concurrent kernels overlap -> wall busy is the union)
    all_intervals = [(e["ts"], e["ts"] + e["dur"]) for e in gpu]
    busy = union_busy(all_intervals)

    # per-bucket summed kernel time (sum of durations, NOT union — shows compute share)
    by_bucket = defaultdict(float)
    by_name = defaultdict(float)
    by_name_cnt = defaultdict(int)
    for e in gpu:
        by_bucket[bucket(e["name"])] += e["dur"]
        by_name[e["name"]] += e["dur"]
        by_name_cnt[e["name"]] += 1

    steps = args.steps
    step_wall = window / steps
    step_busy = busy / steps
    idle = window - busy
    idle_frac = idle / window * 100 if window else 0

    print(f"=== decode-step localization (window {window/1e3:.2f} ms over {steps} steps) ===")
    print(f"per-step WALL      : {step_wall:9.1f} us   ({1e6/step_wall:6.1f} tok/s equiv)")
    print(f"per-step GPU-busy  : {step_busy:9.1f} us   ({busy/window*100:5.1f}% of wall)")
    print(f"per-step GPU-idle  : {(idle/steps):9.1f} us   ({idle_frac:5.1f}% of wall)  <- host/launch gap")
    print()
    print("--- GPU kernel time by bucket (sum of durations / step) ---")
    tot = sum(by_bucket.values())
    for b, v in sorted(by_bucket.items(), key=lambda x: -x[1]):
        print(f"  {b:24} {v/steps:9.1f} us/step   {v/tot*100:5.1f}% of GPU compute")
    print()
    print("--- top 20 kernels by total time ---")
    for name, v in sorted(by_name.items(), key=lambda x: -x[1])[:20]:
        print(f"  {v/steps:8.1f} us/step  x{by_name_cnt[name]:5d}  {name[:80]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
