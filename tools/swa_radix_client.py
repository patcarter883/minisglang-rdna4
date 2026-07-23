"""SWA-radix serve validation client (runs inside the lean container against localhost:1919).

Greedy /generate (SamplingParams default temperature=0 -> deterministic). Two modes:
  * reuse: WARM the prefix P (so its sliding-window snapshot is cached), then generate B = P+suffix,
    which REUSES P via the sliding-layer window extend (rdna4.py::_swa_prefill_extend).
  * cold:  generate B directly against an empty/naive cache (no reuse) — the byte-identity reference.

Byte-identity gate: sha256 of the raw SSE body (identical greedy output => identical bytes) must match
between reuse and cold, for a prefix LONGER than the window (>512) and SHORTER (<512). Also reports
TTFT (time to first token) so the reuse prefill-skip benefit is measurable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request

URL = "http://localhost:1919/generate"


def build_prompts(case: str):
    if case == "long":                     # prefix > window (W=512 tokens)
        unit = "The quantum harmonic oscillator exhibits discrete energy levels spaced evenly apart. "
        P = "A rigorous study of physics. " + unit * 60
    else:                                  # prefix < window
        P = "A rigorous study of physics. The quantum harmonic oscillator has discrete energy levels."
    suffix = " In summary, the single most important practical consequence for working engineers is that"
    return P.strip(), (P + suffix).strip()


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
    # Reconstruct the DECODED TEXT: the server yields exactly `data: <incremental_output>\n` per chunk
    # (stream_generate). Strip the fixed 6-char "data: " prefix EXACTLY (NOT lstrip — the token text may
    # legitimately start with spaces, and lstrip + variable SSE framing made identical greedy output
    # hash differently). Join the payloads; that content is framing-independent.
    text = "".join(
        ln[6:] for ln in raw.decode("utf-8", "replace").split("\n")
        if ln.startswith("data: ") and "[DONE]" not in ln
    )
    return text, (ttft or 0.0), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["reuse", "cold"], required=True)
    ap.add_argument("--case", choices=["long", "short"], required=True)
    ap.add_argument("--max-tokens", type=int, default=48)
    a = ap.parse_args()
    P, B = build_prompts(a.case)

    if a.mode == "reuse":
        gen(P, 4)          # warm: prefill+cache P (creates the page-aligned window snapshot)
        time.sleep(0.7)    # let the async cache_req/attach settle before B reuses it

    text, ttft, total = gen(B, a.max_tokens)
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]   # hash the DECODED text
    print(f"RESULT mode={a.mode} case={a.case} sha={sha} ttft={ttft:.4f} "
          f"total={total:.4f} nchars={len(text)}")
    print(f"  OUT[{a.mode}/{a.case}]: {text[:160]!r}")


if __name__ == "__main__":
    main()
