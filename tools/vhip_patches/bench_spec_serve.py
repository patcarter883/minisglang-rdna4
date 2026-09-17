"""Serve-side check for the vhip GDN spec path: coherence, bs=1 decode tok/s, concurrency-4 tok/s.

Reports the same three things the CONTINUANCE table tracks, so a run here is directly comparable:

    config                    bs=1        concurrency-4
    no-spec                   75.1        222 tok/s
    fla-Triton spec           98.0        HANGS
    gdn_hip spec (gather)     49.6        156 peak / 134 mean

Usage:  python tools/vhip_patches/bench_spec_serve.py [--port 8000] [--gen 200] [--trials 3]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import time
import urllib.request

COHERENCE = [
    # The corruption this fix targets looked like ' Paris!!!!!!' — one correct token then token 0 —
    # so what matters here is that the text stays fluent well past the first token.
    ("What is the capital of France? Answer in one short sentence.", 400),
    ("List the first 8 prime numbers, comma separated, nothing else.", 400),
    ("Write one sentence explaining what a speculative-decoding drafter does.", 400),
]
DECODE_PROMPT = (
    "Write a detailed technical explanation of how a gated delta-net linear-attention layer "
    "maintains its recurrent state across tokens. Be thorough and precise."
)


def post(port, prompt, max_tokens, temperature=0.0):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"http://localhost:{port}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.load(r)
    dt = time.perf_counter() - t0
    msg = out["choices"][0]["message"]
    # Qwen3.6 is a thinking model: on a max_tokens-truncated response `content` is null and the whole
    # generation sits in `reasoning`. Either one is what we want to eyeball for coherence.
    txt = msg.get("content") or msg.get("reasoning") or msg.get("reasoning_content") or ""
    return txt, out["usage"]["completion_tokens"], dt


def models(port):
    with urllib.request.urlopen(f"http://localhost:{port}/v1/models", timeout=30) as r:
        return json.load(r)["data"][0]["id"]


ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8000)
ap.add_argument("--gen", type=int, default=200)
ap.add_argument("--trials", type=int, default=3)
ap.add_argument("--conc", type=int, default=4)
args = ap.parse_args()
MODEL = models(args.port)
print(f"model={MODEL}\n")

print("== coherence ==")
for p, n in COHERENCE:
    txt, ntok, dt = post(args.port, p, n)
    one = " ".join(txt.split())
    print(f"  Q: {p[:56]}\n  A: {one[:160]}\n     ({ntok} tok, {dt:.1f}s, {ntok/dt:.1f} tok/s)")

print("\n== bs=1 decode ==")
post(args.port, DECODE_PROMPT, 16)  # warm
bs1 = []
for i in range(args.trials):
    _, ntok, dt = post(args.port, DECODE_PROMPT, args.gen)
    bs1.append(ntok / dt)
    print(f"  trial {i + 1}: {ntok} tok in {dt:.2f}s -> {bs1[-1]:.1f} tok/s")
print(f"  bs=1 mean {sum(bs1) / len(bs1):.1f} tok/s   peak {max(bs1):.1f}")

print(f"\n== concurrency-{args.conc} ==")
conc = []
for i in range(args.trials):
    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(args.conc) as ex:
        futs = [ex.submit(post, args.port, DECODE_PROMPT + f" (variant {j})", args.gen)
                for j in range(args.conc)]
        res = [f.result() for f in futs]
    wall = time.perf_counter() - t0
    tot = sum(r[1] for r in res)
    conc.append(tot / wall)
    print(f"  trial {i + 1}: {tot} tok in {wall:.2f}s -> {conc[-1]:.1f} tok/s aggregate")
print(f"  conc-{args.conc} mean {sum(conc) / len(conc):.1f} tok/s   peak {max(conc):.1f}   "
      f"{len(conc)}/{args.trials} trials completed")
