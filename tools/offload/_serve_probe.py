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
    ap.add_argument("--max-tokens", type=int, default=8)
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
        content = (d["choices"][0]["message"].get("content") or "")
    except Exception as exc:  # noqa: BLE001 - every failure mode here means "not ready"
        if not a.quiet:
            print(f"not-ready: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if not content.strip():
        if not a.quiet:
            print("not-ready: empty content (the engine answered but generated nothing)",
                  file=sys.stderr)
        return 1
    if not a.quiet:
        print(content.strip()[:200])
    return 0


if __name__ == "__main__":
    sys.exit(main())
