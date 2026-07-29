#!/usr/bin/env python3
"""Concurrency decode driver for minisgl /generate — aggregate tok/s at N in-flight requests.

Every measurement in the decode push so far was bs=1, which is exactly where the shared decode
GEMV is strongest (it is an M=1-shaped kernel and loses to WMMA from about M=4). Raising
*_GEMV_MAXM puts concurrent batches on the GEMV too, so the cost of that has to be measured at
M>1 or it is invisible.

SAMPLED, ignore_eos, fixed max_tokens; timing starts at each stream's first token so prefill and
graph capture are excluded. Usage: _conc_driver.py <conc> [max_tokens]
"""
import json
import os
import sys
import threading
import time
import urllib.request

BASE = "http://127.0.0.1:1919"
PROMPT = (
    "Explain, in detail and step by step, how a modern GPU executes a matrix multiplication, "
    "covering memory hierarchy, warps, and tiling. Be thorough."
)


def one(max_tokens: int, out: list) -> None:
    body = json.dumps({"prompt": PROMPT, "max_tokens": max_tokens, "ignore_eos": True}).encode()
    req = urllib.request.Request(BASE + "/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    n, t_first = 0, None
    with urllib.request.urlopen(req, timeout=900) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace")
            if not line.startswith("data: ") or line[6:].rstrip("\n") == "[DONE]":
                continue
            if t_first is None:
                t_first = time.perf_counter()
            n += 1
    out.append((n, time.perf_counter() - (t_first or time.perf_counter())))


def main() -> None:
    conc = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    # warmup so capture/autotune are not in the measured window
    threading.Thread(target=one, args=(16, [])).start()
    time.sleep(8)

    out: list = []
    t0 = time.perf_counter()
    ths = [threading.Thread(target=one, args=(max_tokens, out)) for _ in range(conc)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    wall = time.perf_counter() - t0
    total = sum(n for n, _ in out)
    print(f"[conc={conc}] streams={len(out)} tokens={total} wall={wall:.3f}s "
          f"agg_tok/s={total / wall:.1f}")


if __name__ == "__main__":
    main()
