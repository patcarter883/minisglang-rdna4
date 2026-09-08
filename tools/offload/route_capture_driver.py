#!/usr/bin/env python3
"""Drive a serve to produce a PROVENANCE-VALID routing trace.

The oracle's G5 gate refuses a decision table below 12 distinct req_uids and 20000 decode steps,
because 4 prompts / 456 steps characterises nothing (llama.cpp's comparand trace is 87,889 lines).
This sends N distinct prompts SEQUENTIALLY -- one in flight at a time -- so every decode batch is
M == 1 and takes the tracer's ring path. That matters twice: the ring is the un-perturbed path (the
host path does a blocking .tolist() per layer), and bs=1 is the operating point gates G1/G3 are
defined on. A concurrent driver would put M=2 batches on the host path and mix two interleaved
sequences into one locality measurement.

Stdlib only; runs on the lease host, outside the container.
"""
from __future__ import annotations
import argparse, json, sys, time, urllib.request

PROMPTS = [
    "Explain how a heat pump moves thermal energy against a temperature gradient, in detail.",
    "Write a technical description of how B-trees keep lookups balanced as data is inserted.",
    "Describe the process of photosynthesis from photon capture to glucose, step by step.",
    "Explain the CAP theorem and give three real distributed systems that pick different corners.",
    "Walk through how a modern CPU branch predictor works and why mispredictions are costly.",
    "Describe the chemistry of bread fermentation and what each ingredient contributes.",
    "Explain how GPS receivers correct for relativistic time dilation, with the arithmetic.",
    "Give a detailed account of how vaccines train the adaptive immune system.",
    "Explain the mathematics behind public-key cryptography using RSA as the worked example.",
    "Describe how an internal combustion engine's four strokes convert fuel into torque.",
    "Explain how ocean thermohaline circulation works and why it affects European climate.",
    "Describe how a compiler performs register allocation, including graph colouring.",
    "Explain the physics of how noise-cancelling headphones cancel sound waves.",
    "Give a detailed explanation of how DNS resolution works from browser to authoritative server.",
    "Explain how lithium-ion batteries store and release charge, including degradation modes.",
    "Describe how error-correcting codes let a scratched CD still play perfectly.",
]

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:1919")
    ap.add_argument("--max-tokens", type=int, default=1800)
    ap.add_argument("--target-steps", type=int, default=21000)
    ap.add_argument("--timeout", type=float, default=900.0)
    a = ap.parse_args()

    total = 0
    t0 = time.perf_counter()
    for i, prompt in enumerate(PROMPTS):
        if total >= a.target_steps:
            break
        body = json.dumps({
            "model": "x",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": a.max_tokens,
            # SAMPLED, not greedy: temp 0 exercises a different sampler path and this repo's own
            # rule forbids characterising behaviour at temp 0.
            "temperature": 1.0, "top_k": 20, "top_p": 0.95,
        }).encode()
        req = urllib.request.Request(a.url.rstrip("/") + "/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        s = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=a.timeout) as r:
                d = json.loads(r.read())
        except Exception as exc:  # noqa: BLE001
            print(f"[capture] prompt {i} FAILED: {type(exc).__name__}: {exc}", flush=True)
            continue
        n = int((d.get("usage") or {}).get("completion_tokens") or 0)
        total += n
        el = time.perf_counter() - s
        print(f"[capture] prompt {i:2d}: {n:5d} tokens in {el:6.1f}s "
              f"({n/el if el else 0:5.2f} tok/s)  cumulative {total}/{a.target_steps}", flush=True)
    print(f"[capture] DONE {total} decode steps from {len(PROMPTS)} prompts in "
          f"{time.perf_counter()-t0:.0f}s", flush=True)
    return 0 if total > 0 else 1

if __name__ == "__main__":
    sys.exit(main())
