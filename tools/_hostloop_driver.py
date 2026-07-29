#!/usr/bin/env python3
"""bs=1 decode driver.

SAMPLED by default (temperature 1.0 / top_k 20 / top_p 0.95 — the checkpoint's own
generation_config), because real serving is not greedy: greedy takes an argmax while sampled runs
the top-k/top-p sampler path, which is work that never appeared in a greedy measurement.
Set GREEDY=1 for the argmax path (useful only for byte-identity checks between two builds).

Sends a fixed prompt (greedy, ignore_eos, fixed max_tokens) to /generate, streams the SSE
response, and reports: full output text (byte-identity token stream) + end-to-end decode tok/s.
A warmup request precedes the measured one so prefill / graph-capture / autotune are excluded.
"""
import json
import sys
import time
import os
import urllib.request

GREEDY = os.environ.get("GREEDY", "0") == "1"
BASE = "http://127.0.0.1:1919"
PROMPT = (
    "Explain, in detail and step by step, how a modern GPU executes a matrix multiplication, "
    "covering memory hierarchy, warps, and tiling. Be thorough."
)


def run(max_tokens: int):
    req_body = {"prompt": PROMPT, "max_tokens": max_tokens, "ignore_eos": True}
    if GREEDY:
        req_body["temperature"] = 0.0          # argmax; for byte-identity checks between builds
    # else: send NOTHING — the server resolves the checkpoint's generation_config
    # (do_sample=true, temperature 1.0, top_k 20, top_p 0.95), which is the real serving path.
    body = json.dumps(req_body).encode()
    req = urllib.request.Request(BASE + "/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    chunks = []
    t_first = None
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace")
            if not line.startswith("data: "):
                continue
            payload = line[6:].rstrip("\n")
            if payload == "[DONE]":
                break
            if t_first is None:
                t_first = time.perf_counter()
            chunks.append(payload)
    t_end = time.perf_counter()
    text = "".join(chunks)
    n_chunks = len(chunks)
    decode_wall = t_end - (t_first if t_first is not None else t0)
    # first chunk is the first decoded token; remaining (n_chunks-1) tokens define steady-state rate
    toks = max(n_chunks - 1, 1)
    return text, n_chunks, decode_wall, toks / decode_wall if decode_wall > 0 else 0.0


def main():
    max_tokens = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    out_path = sys.argv[2] if len(sys.argv) > 2 else "/engine/scratch_out.txt"
    print("[driver] warmup...", flush=True)
    run(64)
    time.sleep(1.0)
    print(f"[driver] measured run, max_tokens={max_tokens}...", flush=True)
    text, n_chunks, wall, tps = run(max_tokens)
    with open(out_path, "w") as f:
        f.write(text)
    print(f"[driver] chunks={n_chunks} decode_wall={wall:.3f}s tok/s={tps:.2f}", flush=True)
    print(f"[driver] output_len_chars={len(text)} sha_head={hash(text) & 0xffffffff:08x}", flush=True)
    print(f"[driver] wrote output to {out_path}", flush=True)


if __name__ == "__main__":
    main()
