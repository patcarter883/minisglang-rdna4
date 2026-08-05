#!/usr/bin/env python3
"""Decode-throughput driver at a fixed CONCURRENCY, for the MoE producer act-quant A/B.

`bs` here is the number of SIMULTANEOUS in-flight requests, which is what sets the M the grouped
GEMM actually sees at decode: bs=1 -> M=1, and bs=n -> M=n once all n are past prefill. That is the
axis this change is measured on, because it removes a fixed number of DISPATCHES per step rather
than any amount of work per row — so its relative value should FALL as bs rises and the step gets
more compute to hide behind.

SAMPLED (the checkpoint's own generation_config), not greedy: greedy takes an argmax while sampled
runs the top-k/top-p sampler path, and serving is sampled. Reports aggregate decode tok/s across the
concurrent stream, measured AFTER a warmup request so prefill / graph capture / autotune are out.
"""
import json
import sys
import threading
import time
import urllib.request

BASE = "http://127.0.0.1:1919"
PROMPT = (
    "Explain, in detail and step by step, how a modern GPU executes a matrix multiplication, "
    "covering memory hierarchy, warps, and tiling. Be thorough."
)


def one(max_tokens: int, out: list, idx: int):
    body = json.dumps({
        "prompt": PROMPT, "max_tokens": max_tokens, "ignore_eos": True, "stream": True,
    }).encode()
    req = urllib.request.Request(BASE + "/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    n, t_first, t_last = 0, None, None
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            now = time.perf_counter()
            if t_first is None:
                t_first = now
            t_last = now
            n += 1
    out[idx] = (n, t_first, t_last)


def run(bs: int, max_tokens: int):
    res = [None] * bs
    ts = [threading.Thread(target=one, args=(max_tokens, res, i)) for i in range(bs)]
    t0 = time.perf_counter()
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    t1 = time.perf_counter()
    got = [r for r in res if r]
    total = sum(r[0] for r in got)
    # DECODE tok/s only: exclude each stream's own TTFT by measuring from its FIRST token. The
    # aggregate window is first-first-token to last-last-token, which is what a served decode
    # actually occupies.
    first = min(r[1] for r in got)
    last = max(r[2] for r in got)
    dec = max(last - first, 1e-9)
    return total, total / dec, t1 - t0


if __name__ == "__main__":
    bs = int(sys.argv[1])
    toks = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    one(32, [None], 0)  # warmup: prefill / capture / autotune out of the measurement
    n, tps, wall = run(bs, toks)
    print(f"bs={bs} tokens={n} decode_tok_s={tps:.2f} wall={wall:.2f}s")
