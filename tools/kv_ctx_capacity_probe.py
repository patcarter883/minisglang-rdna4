"""Equal-VRAM context probe — what the fp8 KV cache actually BUYS.

The accuracy question ("is fp8 KV as good as bf16 at the same context") is the wrong frame, because
the two configurations do not get the same context: at identical VRAM an fp8 cache holds TWICE the
tokens (measured on Qwen3.6-35B TP=2: 53,440 vs 26,720 tokens in the pool). So the deployment
question is: at equal VRAM, does fp8-plus-more-context beat bf16-with-less?

This probe answers it the only way that is not an opinion — sweep the context length on a RUNNING
serve and record, per length, whether the request is served at all and whether the model still finds
a fact buried in the middle of it. A bf16 serve does not answer a 32k-token request WORSE; it does
not answer it.

    python3 tools/kv_ctx_capacity_probe.py --base http://127.0.0.1:1919 --lengths 4000,8000,16000,24000,32000

Each case builds `N` filler lines of "item K is VALUE", hides the needle at ~50% depth (the hardest
position for a long-context model, and the one fp8-KV was previously measured to lose at 7.7k), and
asks for it back. Reported per length: HTTP outcome, prompt tokens the serve actually saw, and
whether the answer contains the needle value.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def build_prompt(approx_tokens: int, needle_idx: int) -> tuple[str, int]:
    """~7 tokens per filler line, so line count ~= tokens/7. The needle's VALUE is derived from its
    index so a model that pattern-matches the question instead of retrieving is still wrong."""
    n_lines = max(16, approx_tokens // 7)
    idx = int(n_lines * 0.5) if needle_idx < 0 else needle_idx
    val = 7919 + idx * 13
    lines = [f"item {i} is {7919 + i * 13}" for i in range(n_lines)]
    body = "\n".join(lines)
    q = f"\n\nWhat is the value of item {idx}? Answer with just the number."
    return body + q, val


def ask(base: str, prompt: str, max_tokens: int, timeout: int) -> dict:
    body = {
        "model": "x",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": False,
    }
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read())
        return {"ok": True, "wall": time.time() - t0, "resp": d}
    except urllib.error.HTTPError as e:
        return {"ok": False, "wall": time.time() - t0, "error": f"HTTP {e.code}: {e.read()[:300]!r}"}
    except Exception as e:  # timeout, reset, refused — all "the serve did not answer this"
        return {"ok": False, "wall": time.time() - t0, "error": f"{type(e).__name__}: {e}"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:1919")
    ap.add_argument("--lengths", default="4000,8000,16000,24000,32000")
    ap.add_argument("--max-tokens", type=int, default=2048, help="a reasoning serve spends most of this thinking")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default=None, help="write results as JSON here")
    a = ap.parse_args()

    rows = []
    for L in [int(x) for x in a.lengths.split(",")]:
        prompt, val = build_prompt(L, -1)
        r = ask(a.base, prompt, a.max_tokens, a.timeout)
        if not r["ok"]:
            print(f"  ctx~{L:>6}: NOT SERVED after {r['wall']:.1f}s — {r['error'][:160]}")
            rows.append({"ctx": L, "served": False, "found": False, "error": r["error"][:300]})
            continue
        d = r["resp"]
        txt = d["choices"][0]["message"].get("content") or ""
        ptok = d.get("usage", {}).get("prompt_tokens", -1)
        found = str(val) in txt
        print(
            f"  ctx~{L:>6}: served ({ptok} prompt tokens, {r['wall']:.1f}s)  needle {'FOUND' if found else 'MISSED'}"
            f"  answer={txt.strip()[:60]!r}"
        )
        rows.append(
            {"ctx": L, "served": True, "prompt_tokens": ptok, "found": found,
             "wall": round(r["wall"], 2), "answer": txt.strip()[:120],
             "finish": d["choices"][0]["finish_reason"]}
        )

    served = sum(1 for r in rows if r["served"])
    found = sum(1 for r in rows if r["found"])
    print(f"\n{a.label or 'result'}: served {served}/{len(rows)}, needle found {found}/{len(rows)}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"label": a.label, "base": a.base, "rows": rows}, f, indent=2)
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
