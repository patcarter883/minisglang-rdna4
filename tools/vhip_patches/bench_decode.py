"""bs=1 decode benchmark reporting BOTH metrics the wiring table tracks.

    decode = (tokens - 1) / delta(vllm:request_decode_time_seconds_sum)   <- server-side, no TTFT
    e2e    =  tokens / wall                                               <- client-side, TTFT charged

The two differ by ~4% on this stack, which is enough to invert a comparison — so every row in
docs/CONTINUE_rdna4_vllm_wiring.md must say which one it used. This prints both, from one request.

Usage:  python tools/vhip_patches/bench_decode.py [--port 8000] [--gen 200] [--trials 3]
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request

PROMPT = (
    "Write a detailed technical explanation of how a gated delta-net linear-attention layer "
    "maintains its recurrent state across tokens. Be thorough and precise."
)


def _get(url: str, timeout: int = 60) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode()


def counters(port: int) -> dict[str, float]:
    """Scrape the two decode-time counters. Both are cumulative across all requests, so a
    difference across a single serialised bs=1 request is that request's own decode time."""
    out: dict[str, float] = {}
    for line in _get(f"http://localhost:{port}/metrics").splitlines():
        if line.startswith("#"):
            continue
        for key in ("vllm:request_decode_time_seconds_sum",
                    "vllm:request_decode_time_seconds_count"):
            if line.startswith(key):
                out[key] = out.get(key, 0.0) + float(line.rsplit(" ", 1)[1])
    return out


def post(port: int, prompt: str, max_tokens: int, model: str) -> tuple[str, int, float]:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"http://localhost:{port}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=900) as r:
        out = json.load(r)
    dt = time.perf_counter() - t0
    msg = out["choices"][0]["message"]
    txt = msg.get("content") or msg.get("reasoning") or msg.get("reasoning_content") or ""
    return txt, out["usage"]["completion_tokens"], dt


ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8000)
ap.add_argument("--gen", type=int, default=200)
ap.add_argument("--trials", type=int, default=3)
ap.add_argument("--label", default="")
args = ap.parse_args()

model = json.loads(_get(f"http://localhost:{args.port}/v1/models"))["data"][0]["id"]
post(args.port, PROMPT, 16, model)  # warm (first call pays graph/JIT one-offs)

rows = []
for i in range(args.trials):
    c0 = counters(args.port)
    txt, ntok, wall = post(args.port, PROMPT, args.gen, model)
    c1 = counters(args.port)
    d_sum = c1["vllm:request_decode_time_seconds_sum"] - c0["vllm:request_decode_time_seconds_sum"]
    decode = (ntok - 1) / d_sum if d_sum > 0 else float("nan")
    e2e = ntok / wall
    rows.append((decode, e2e))
    print(f"  trial {i + 1}: {ntok} tok  decode {decode:6.1f} tok/s   e2e {e2e:6.1f} tok/s")

d = [r[0] for r in rows]
e = [r[1] for r in rows]
print(f"\n{args.label or 'result'}:  decode mean {sum(d)/len(d):.1f} (peak {max(d):.1f})   "
      f"e2e mean {sum(e)/len(e):.1f} (peak {max(e):.1f})")
print(f"\nlast sample: {' '.join(txt.split())[:220]}")
