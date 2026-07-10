"""Fire N chat-completion requests CONCURRENTLY at the in-container server (localhost:1919) and print
one line per prompt: ``<idx>\t<repr(content)>`` (or ``<idx>\tERR:<reason>``), sorted by idx.

Run INSIDE the serve container (docker exec) so it hits the internal :1919 directly — the host
docker-published port drops simultaneous connection bursts, which made an 8-way `curl &` harness fail
symmetrically. A single async client with one connection pool does not. Deterministic greedy
(temperature 0) so ONDEVICE=1 and =0 runs must be byte-identical.

Uses only stdlib (urllib + threads) — the lean image has no aiohttp/httpx guaranteed.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import os
import sys
import urllib.request

PROMPTS = [
    "Name five colors.",
    "Count from 1 to 5 in words.",
    "What is the capital of Japan?",
    "List three fruits.",
    "Say hello in French.",
    "Add two and three.",
    "Give one fact about cats.",
    "How many days are in a week?",
]

URL = "http://localhost:1919/v1/chat/completions"


def one(idx: int, prompt: str) -> tuple[int, str]:
    body = json.dumps(
        {
            "model": "x",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 48,
            "seed": 0,
        }
    ).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            d = json.loads(r.read())
        return idx, repr(d["choices"][0]["message"]["content"])
    except Exception as e:  # noqa: BLE001 - report any failure inline
        return idx, f"ERR:{type(e).__name__}:{str(e)[:80]}"


def main() -> int:
    # NREQ caps how many of the fixed prompts to fire concurrently (target load = 3-4 on this box;
    # N=8 exceeds the GDN spec-verify memory headroom and OOMs — a known constraint, not a code bug).
    n = int(os.environ.get("NREQ", str(len(PROMPTS))))
    prompts = PROMPTS[:n]
    results: dict[int, str] = {}
    # fire ALL prompts at once -> they arrive together and batch into multi-req verify steps
    with cf.ThreadPoolExecutor(max_workers=len(prompts)) as ex:
        futs = [ex.submit(one, i, p) for i, p in enumerate(prompts)]
        for f in cf.as_completed(futs):
            i, txt = f.result()
            results[i] = txt
    for i in sorted(results):
        print(f"{i}\t{results[i]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
