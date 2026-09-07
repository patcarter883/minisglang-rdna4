#!/usr/bin/env python3
"""Is this serve ACTUALLY generating? Exit 0 iff one chat completion returns non-empty content.

`/health` is not an answer to that question and never was: the API process binds it before the
engine is warm and keeps answering 200 after the worker dies — container `Up`, `RestartCount` 0,
health green, zero tokens. Every readiness loop in this repo that trusted it reported a healthy
serve for the whole timeout and then benched nothing.

SAMPLED, NOT GREEDY. temperature 1.0 / top_k 20 / top_p 0.95 is the operating point these serves are
measured at; a temp-0 probe exercises a different sampler path and is exactly the shape of test the
repo's own rule ("never QUALITY-test at temp=0") says not to write.

Stdlib only — it runs on the lease host, outside any container.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", default="x")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--prompt", default="Say READY and nothing else.")
    # 64, NOT 8. THIS SERVE IS A THINKING MODEL AND 8 TOKENS NEVER LEAVES THE THINK SPAN.
    # Measured 2026-09-08 against the live CPU-tier serve: "Say READY and nothing else."
    # cost 24 reasoning tokens before the 1-token answer (29 completion tokens total). At
    # max_tokens=8 the span is still open, so `reasoning.py` files the WHOLE reply under
    # `reasoning_content` with `content=""` -- which the check below then read as "generated
    # nothing". A healthy serve answering in 6.4 s was reported NOT READY for the whole
    # timeout, and the boot it took 70 s of staged load to reach was thrown away.
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    body = json.dumps({
        "model": a.model,
        "messages": [{"role": "user", "content": a.prompt}],
        "max_tokens": a.max_tokens,
        "temperature": 1.0,
        "top_k": 20,
        "top_p": 0.95,
    }).encode()
    req = urllib.request.Request(
        a.url.rstrip("/") + "/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=a.timeout) as r:
            d = json.loads(r.read())
        msg = d["choices"][0]["message"]
        content = (msg.get("content") or "")
        # Reasoning IS generation. The question this probe exists to answer is "did the engine
        # produce tokens", and on a reasoning model an unterminated think span is a complete
        # answer to that -- the tokens went through the sampler either way. Reading only
        # `content` makes readiness depend on the model finishing its thought inside
        # max_tokens, which is a property of the PROMPT, not of engine health.
        reasoning = (msg.get("reasoning_content") or "")
    except Exception as exc:  # noqa: BLE001 - every failure mode here means "not ready"
        if not a.quiet:
            print(f"not-ready: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if not content.strip() and not reasoning.strip():
        if not a.quiet:
            print("not-ready: empty content AND empty reasoning_content (the engine answered but "
                  "generated nothing)", file=sys.stderr)
        return 1
    if not a.quiet:
        if content.strip():
            print(content.strip()[:200])
        else:
            print(f"[reasoning-only, span still open] {reasoning.strip()[:200]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
