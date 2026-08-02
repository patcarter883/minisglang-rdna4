"""Is recurrent-radix-under-spec WRONG, or merely DIFFERENT?

Byte-equality cannot answer that on this stack: the serve is already non-deterministic past ~32
output tokens with the cache OFF (measured: 4 distinct outputs in 5 identical greedy runs at 256
tokens — moe_align's atomicAdd scatter, see the noise-floor memory). So "cold != warm" is the
baseline condition, not evidence of damage.

This measures ACCURACY instead, on a task with objective ground truth that is answerable ONLY from
the cached prefix:

    <900 facts: "item i has value (i*7)%97">   +   "what is the value of item K?"

Every question shares the same long prefix, so a prefix cache is exercised on every request after the
first; the answer is a number this script computes independently. Run it with the cache ON and OFF
and compare accuracy — the same shape as the GSM8K comparison that justified the original CCA gate
(radix-on 25.5%/37.5% vs naive 45% = systematically worse).

    python3 tools/rec_radix_quality.py --label cache-ON
    # restart the serve with --no-gdn-radix for the cache-off leg
    python3 tools/rec_radix_quality.py --label cache-OFF
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request

N_FACTS = 900


def val(i: int) -> int:
    return i * 7 % 97


def ask(base, model, prompt, max_tokens):
    r = urllib.request.Request(base + "/v1/chat/completions",
                               data=json.dumps({"model": model,
                                                "messages": [{"role": "user", "content": prompt}],
                                                "max_tokens": max_tokens, "temperature": 0.0,
                                                "seed": 1234, "stream": False}).encode(),
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=1800) as resp:
        d = json.loads(resp.read())
    m = d["choices"][0]["message"]
    return (m.get("content") or ""), d["usage"]["prompt_tokens"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:1919")
    ap.add_argument("--label", default="run")
    ap.add_argument("--n", type=int, default=24, help="questions asked")
    ap.add_argument("--maxtok", type=int, default=420)
    a = ap.parse_args()
    B = a.base
    model = json.loads(urllib.request.urlopen(B + "/v1/models", timeout=30).read())["data"][0]["id"]
    prefix = "\n".join(f"Fact {i}: item {i} has value {val(i)}." for i in range(N_FACTS))

    # Spread the probes across the prefix so a partially-reused prefix cannot be masked by only ever
    # querying the head or the tail.
    items = [int(N_FACTS * (k + 0.5) / a.n) for k in range(a.n)]
    ok, bad, walls = 0, [], []
    t_all = time.perf_counter()
    for it in items:
        q = (f"\n\nUsing ONLY the facts above, what is the value of item {it}? "
             "Reply with just the number.")
        t0 = time.perf_counter()
        text, ptok = ask(B, model, prefix + q, a.maxtok)
        walls.append(time.perf_counter() - t0)
        nums = re.findall(r"-?\d+", text)
        got = int(nums[-1]) if nums else None
        if got == val(it):
            ok += 1
        else:
            bad.append((it, val(it), got))
    total = time.perf_counter() - t_all

    print(f"\n=== {a.label} ===")
    print(f"  model {model}   {a.n} questions over a ~{N_FACTS}-fact shared prefix")
    print(f"  ACCURACY  {ok}/{a.n} = {100*ok/a.n:.1f}%")
    print(f"  wall      total {total:.1f}s   median/req {sorted(walls)[len(walls)//2]:.2f}s")
    if bad:
        show = ", ".join(f"item{i}: want {w} got {g}" for i, w, g in bad[:6])
        print(f"  misses    {show}" + (" ..." if len(bad) > 6 else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
