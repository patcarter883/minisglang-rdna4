#!/usr/bin/env python3
"""Drive a PLAIN-decode serve through two fixed-length phases, so a step-indexed ROCTx window can be
placed inside each one.

WHY PHASES OF A FIXED TOKEN COUNT. The collection windows (MINISGL_ROCTX_WINDOWS) are placed by
scheduler-LOOP-ITERATION index, because that is the only counter the loop has before it knows what it
is about to run. That index is deterministic only if the traffic is: `ignore_eos` + a fixed
`max_tokens` makes one request cost exactly `max_tokens` decode iterations, so phase boundaries land
where they were computed to land. A phase driven by "generate until EOS" would move the windows.

WHY THE LAST PHASE IS EXPECTED TO DIE. rocprofv3 is LD_PRELOADed into the engine, so it flushes only
on a NORMAL interpreter exit and any signal aborts the write (see tools/propose_rocprof.sh).
MINISGL_EXIT_AFTER_STEPS is the only shutdown that works, and it fires mid-request by construction —
so the final stream raises, and the timings for that phase come from the chunk timestamps collected
before it died. The trace is the artifact; the completion is not.

Timings here are the PROFILED wall, not the true serving wall — rocprofv3 is attached. The unprofiled
control leg is a separate run against the same commit (--no-profiler in serve_busy_trace.sh).
"""
from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
import urllib.request

PROMPT = (
    "Explain, step by step and in depth, how a modern CPU executes a program: cover fetch/decode/"
    "execute, pipelining, branch prediction, caches, and out-of-order execution. Use clear prose."
)


def _stream(url: str, max_tokens: int, barrier: threading.Barrier, out: list, idx: int) -> None:
    payload = json.dumps({
        "model": "", "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens, "temperature": 0.0, "ignore_eos": True, "stream": True,
    }).encode()
    req = urllib.request.Request(url + "/v1/chat/completions", data=payload,
                                 headers={"Content-Type": "application/json"})
    stamps: list[float] = []
    err = None
    barrier.wait()
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            for raw in r:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if body == "[DONE]":
                    break
                stamps.append(time.perf_counter())
    except Exception as exc:                                   # noqa: BLE001
        err = f"{type(exc).__name__}: {exc}"
    out[idx] = {"t0": t0, "stamps": stamps, "err": err}


def phase(url: str, m: int, max_tokens: int, label: str) -> dict:
    barrier = threading.Barrier(m)
    out: list = [None] * m
    threads = [threading.Thread(target=_stream, args=(url, max_tokens, barrier, out, i))
               for i in range(m)]
    w0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - w0
    gaps = []
    ntok = 0
    for r in out:
        s = r["stamps"]
        ntok += len(s)
        gaps += [(s[i] - s[i - 1]) * 1e3 for i in range(1, len(s))]
    # MEDIAN inter-token gap, not the mean: a phase that is cut off by the step bound, or that
    # absorbs a scheduler hiccup, puts a few multi-second gaps in the tail and the mean follows them.
    res = {
        "label": label, "M": m, "max_tokens": max_tokens, "wall_s": wall, "tokens": ntok,
        "gap_ms_median": statistics.median(gaps) if gaps else None,
        "gap_ms_mean": (sum(gaps) / len(gaps)) if gaps else None,
        "gap_ms_p10": (statistics.quantiles(gaps, n=10)[0] if len(gaps) > 10 else None),
        "gap_ms_p90": (statistics.quantiles(gaps, n=10)[8] if len(gaps) > 10 else None),
        "n_gaps": len(gaps),
        "errors": [r["err"] for r in out if r["err"]],
    }
    # At M streams the SERVER runs one decode step per token PER STREAM in a batch of M, so the
    # per-STEP wall is the per-stream inter-token gap; the aggregate token rate is M times that.
    res["step_ms_from_gaps"] = res["gap_ms_median"]
    res["agg_tok_s"] = (ntok / wall) if wall > 0 else None
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--warmup-tokens", type=int, default=16)
    ap.add_argument("--phase-a-tokens", type=int, default=600)
    ap.add_argument("--phase-b-tokens", type=int, default=600)
    ap.add_argument("--phase-b-m", type=int, default=8)
    ap.add_argument("--ready-timeout", type=int, default=900)
    args = ap.parse_args()

    deadline = time.time() + args.ready_timeout
    ready = False
    while time.time() < deadline:
        try:
            urllib.request.urlopen(args.url + "/v1/models", timeout=3).read()
            ready = True
            break
        except Exception:                                      # noqa: BLE001
            time.sleep(3)
    print(f"[drive] ready={ready}", flush=True)
    if not ready:
        raise SystemExit(f"server at {args.url} not ready within {args.ready_timeout}s")

    results = []
    # Warmup is COUNTED, not free: it costs loop iterations like any other traffic, and the window
    # placement is arithmetic over those iterations.
    w = phase(args.url, 1, args.warmup_tokens, "warmup")
    print(f"[drive] warmup: {json.dumps(w)}", flush=True)
    results.append(w)
    time.sleep(2.0)                       # the loop BLOCKS when idle, so this consumes no iterations

    a = phase(args.url, 1, args.phase_a_tokens, "A_bs1")
    print(f"[drive] A_bs1: {json.dumps(a)}", flush=True)
    results.append(a)
    time.sleep(2.0)

    b = phase(args.url, args.phase_b_m, args.phase_b_tokens, f"B_bs{args.phase_b_m}")
    print(f"[drive] B: {json.dumps(b)}", flush=True)
    results.append(b)

    with open(args.out, "w") as f:
        json.dump({"url": args.url, "phases": results}, f, indent=1)
    print(f"[drive] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
