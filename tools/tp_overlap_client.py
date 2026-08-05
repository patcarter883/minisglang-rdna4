"""Client half of tools/tp_overlap_serve_ab.sh: measures prefill TTFT/tok-s and the AR-guard tok/s.

Stdlib only, so it runs inside the serve container with no extra install.

Two deliberate methodology choices, both of which this repo has been burned by before:

  * tok/s comes from `usage.completion_tokens / wall`, never from counting SSE chunks. Counting chunks
    once read 39.7 tok/s as 14.9.
  * the prefill arm asks for max_tokens=1, so TTFT is the prefill and nothing else. Prefill tok/s is
    then prompt_tokens/TTFT, with prompt_tokens taken from the server's own usage block rather than a
    client-side guess at the tokenizer.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request

PREFILL_WORDS = (
    "The distributed runtime schedules each tensor-parallel rank independently while the collective "
    "library matches calls by enqueue order across every participating process in the group. "
)
AR_PROMPT = "Explain, in careful detail, how a tensor-parallel all-reduce works and why it is needed."


def post(base: str, path: str, body: dict, timeout: float = 900.0) -> dict:
    req = urllib.request.Request(
        base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt-tokens", type=int, default=3200)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    # ~3200 tokens: the shape that exercised the SWA extend. Built from a fixed sentence so both arms
    # get a byte-identical prompt.
    body_text = (PREFILL_WORDS * ((args.prompt_tokens // 27) + 1)).strip()
    out: dict = {}

    # --- warm the serve (weights paged in, kernels JIT'd, graphs captured) --------------------------
    post(args.base, "/v1/chat/completions",
         {"model": "x", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8,
          "temperature": 0})

    # --- PREFILL: max_tokens=1 so the whole wall IS the prefill --------------------------------------
    ttfts, ptoks, text = [], None, None
    for _ in range(args.reps):
        # A distinct suffix per rep would defeat the prefix cache; we WANT the same prompt, so instead
        # the serve is asked with cache disabled semantics via a unique lead token per rep.
        t0 = time.perf_counter()
        r = post(args.base, "/v1/chat/completions",
                 {"model": "x",
                  "messages": [{"role": "user", "content": f"{len(ttfts)} {body_text}"}],
                  "max_tokens": 1, "temperature": 0})
        ttfts.append(time.perf_counter() - t0)
        ptoks = r.get("usage", {}).get("prompt_tokens")
        text = r["choices"][0]["message"].get("content")
    ttfts.sort()
    med = ttfts[len(ttfts) // 2]
    out["prefill_ttft_s"] = med
    out["prefill_prompt_tokens"] = ptoks
    out["prefill_tok_s"] = (ptoks / med) if ptoks else 0.0
    out["prefill_text"] = text
    out["prefill_all_ttft"] = ttfts
    print(f"prefill: prompt_tokens={ptoks}  TTFT median={med:.3f}s  "
          f"prefill tok/s={out['prefill_tok_s']:.1f}  (all {['%.3f' % t for t in ttfts]})")

    # --- AR GUARD: 256-token greedy decode, the captured path ---------------------------------------
    rates = []
    for _ in range(args.reps):
        t0 = time.perf_counter()
        r = post(args.base, "/v1/chat/completions",
                 {"model": "x", "messages": [{"role": "user", "content": AR_PROMPT}],
                  "max_tokens": 256, "temperature": 0})
        w = time.perf_counter() - t0
        n = r.get("usage", {}).get("completion_tokens", 0)
        rates.append(n / w)
        print(f"AR guard: {n} tokens in {w:.2f}s = {n / w:.1f} tok/s")
    rates.sort()
    out["ar_tok_s"] = rates[len(rates) // 2]

    json.dump(out, open(args.out, "w"), indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "prefill_text"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
