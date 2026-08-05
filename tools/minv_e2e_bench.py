"""End-to-end tok/s guard for a layers/minv.py dispatch change, on a running serve at :1919.

minv_linear is a SHARED engine path — every model's unquantized bf16 linears, routers and LM head go
through it — so a dispatch change has to be answered end to end on both served models, not just in a
kernel microbench. Warms up first (the first request after boot is not the measurement) and reports
the median of N timed completions.

    python tools/minv_e2e_bench.py --max-tokens 256 --reps 3 [--port 1919] [--prompt ...]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.request

PROMPT = ("Explain, in careful detail, how a tensor-parallel transformer splits its attention and "
          "MLP weights across two GPUs, and what has to be all-reduced and why.")


def complete(port: int, prompt: str, max_tokens: int, temperature: float) -> tuple[int, float, str]:
    body = json.dumps({
        "model": "default", "prompt": prompt, "max_tokens": max_tokens,
        "temperature": temperature, "stream": False,
    }).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=1800) as r:
        out = json.load(r)
    dt = time.perf_counter() - t0
    n = out.get("usage", {}).get("completion_tokens") or 0
    return n, dt, out["choices"][0].get("text", "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=1919)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    for i in range(args.warmup):
        n, dt, _ = complete(args.port, args.prompt, args.max_tokens, args.temperature)
        print(f"  warmup {i}: {n} tok in {dt:.2f}s = {n / dt:.1f} tok/s (not counted)")
    rates, rows = [], []
    for i in range(args.reps):
        n, dt, txt = complete(args.port, args.prompt, args.max_tokens, args.temperature)
        r = n / dt if dt else 0.0
        rates.append(r)
        rows.append((n, dt, r))
        print(f"  rep {i}: {n} tok in {dt:.2f}s = {r:.1f} tok/s")
    med = statistics.median(rates)
    print(f"\nRESULT{' ' + args.tag if args.tag else ''}: median {med:.1f} tok/s "
          f"over {args.reps} reps (min {min(rates):.1f}, max {max(rates):.1f}); "
          f"first 80 chars: {rows[-1] and txt[:80]!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
