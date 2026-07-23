"""SWA graph-capture validation client (runs inside the lean container against localhost:1919).

Greedy /generate (temperature=0, ignore_eos=True -> exactly max_tokens deterministic tokens). Prints:
  * sha256 of the decoded output text (identical greedy tokens => identical bytes) — the byte-identity
    gate between an EAGER (--cuda-graph-max-bs 0) and a CAPTURED (--cuda-graph-max-bs N) server;
  * decode tok/s = max_tokens / (total - ttft) — the eager-vs-captured perf metric.

Two cases: `short` (< window, the byte-identity HARD GATE — the model's long-forward is non-deterministic
run-to-run, a pre-existing fp8-MoE property, so cross-run byte-identity is only meaningful on a short
generation) and `long` (decode-heavy, for the tok/s number).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request

URL = "http://localhost:1919/generate"


def build_prompt(case: str) -> str:
    if case == "long":
        unit = "The quantum harmonic oscillator exhibits discrete energy levels spaced evenly apart. "
        return ("A rigorous study of physics. " + unit * 40).strip()
    return "A rigorous study of physics. The quantum harmonic oscillator has discrete energy levels."


def gen(prompt: str, max_tokens: int):
    body = json.dumps({"prompt": prompt, "max_tokens": max_tokens, "ignore_eos": True}).encode()
    req = urllib.request.Request(URL, data=body, headers={"content-type": "application/json"})
    t0 = time.time()
    ttft = None
    raw = bytearray()
    with urllib.request.urlopen(req, timeout=900) as r:
        for chunk in r:
            raw += chunk
            if ttft is None and chunk.strip() not in (b"", b"data:", b"data: "):
                ttft = time.time() - t0
    text = "".join(
        ln[6:] for ln in raw.decode("utf-8", "replace").split("\n")
        if ln.startswith("data: ") and "[DONE]" not in ln
    )
    return text, (ttft or 0.0), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True)  # free label: eager / captured
    ap.add_argument("--case", choices=["long", "short"], required=True)
    ap.add_argument("--max-tokens", type=int, default=48)
    a = ap.parse_args()
    prompt = build_prompt(a.case)
    text, ttft, total = gen(prompt, a.max_tokens)
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    decode_s = max(total - ttft, 1e-9)
    toks = a.max_tokens
    print(f"RESULT mode={a.mode} case={a.case} sha={sha} ttft={ttft:.4f} total={total:.4f} "
          f"decode_tok_s={toks / decode_s:.2f} ntok={toks} nchars={len(text)}")
    print(f"  OUT[{a.mode}/{a.case}]: {text[:200]!r}")


if __name__ == "__main__":
    main()
