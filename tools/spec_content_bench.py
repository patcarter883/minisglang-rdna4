#!/usr/bin/env python3
"""Serve bench for SPECULATIVE DECODE, measured the two ways that actually decide the answer:
by CONTENT FAMILY and SAMPLED.

Both are corrections to how this is usually measured here, and both are load-bearing:

  * SAMPLED, not greedy. `ab_tile_serve_bench.py` runs `temperature: 0.0`. Greedy inflates draft
    acceptance, so a greedy bench over-recommends K and over-sells every spec arm. Serving is
    sampled, so the number that predicts production is the sampled one.
  * PER CONTENT FAMILY. Acceptance is a property of the WORKLOAD, not of the drafter. This repo's
    own committed measurement: DFlash accepted 2.78 on prose (+1.2% throughput) and 5.98 on
    math/code (+62%, at width 15). A bench that runs one content type reports one point on that
    spread and calls it "the" speedup. So four families run every time:

        code    high-structure source with obvious continuations   (spec-FRIENDLY)
        math    stepwise arithmetic derivation                     (spec-FRIENDLY)
        struct  a long JSON array with a fixed schema              (spec-FRIENDLY, most regular)
        prose   open-ended explanation                             (spec-HOSTILE control)

    The prose arm is not padding: it is the control that stops a friendly-only result from being
    read as a general speedup.

Every prompt carries a fresh random prefix so no two requests share a radix prefix — otherwise the
second repeat measures a prefix-cache hit instead of a prefill.

Spec accounting comes from the server's own counters (`vllm:spec_decode_*`) sampled before and
after each family, so acceptance is the ENGINE's number, not inferred from wall time. Acceptance is
reported per DRAFT OFFERED, because raw accepted-token counts double-count whenever a controller
widens the proposal.

    python3 tools/spec_content_bench.py --url http://localhost:1919 --model Qwen3.8-27B \
        --label 27b-dflash --out /tmp/27b-dflash.json

Stdlib only. No GPU use of its own.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import threading
import time
import urllib.request

# --------------------------------------------------------------------------------------------
# Content families. Each returns a prompt that should produce a LONG, continuable generation, so
# decode dominates the measurement and a drafter has runway to be right or wrong repeatedly.
# --------------------------------------------------------------------------------------------

_CODE = """Write a complete Python module implementing a fixed-size ring buffer class called \
`RingBuffer`. Include: __init__, __len__, __iter__, is_empty, is_full, push, pop, peek, clear, \
extend, to_list, and __repr__. Every method needs a docstring and full implementation with bounds \
checks. Then write a second class `TimestampedRingBuffer` that subclasses it and adds `push_at`, \
`window`, and `decay`. Output only code."""

_MATH = """Compute the following step by step, showing every intermediate line in the form \
`step N: <expression> = <value>`. Do not skip steps and do not summarise.
Start with x = 17. Repeat 40 times: multiply by 3, add 11, then subtract 4. \
After each iteration print the running total and the iteration index."""

_STRUCT = """Output a JSON array of 40 objects and nothing else. Every object must have exactly \
these keys in this order: "id" (integer, sequential from 1000), "name" (a short lowercase \
identifier), "category" (one of "alpha", "beta", "gamma"), "score" (a float with two decimals \
between 0 and 100), "active" (boolean), "tags" (an array of exactly two short strings). \
Output valid JSON only, no prose, no code fence."""

_PROSE = """Explain, in continuous prose with no lists and no headings, how a modern inference \
engine schedules concurrent requests against a paged key-value cache: admission, batching, \
eviction, and what changes when the model is a hybrid of attention and recurrent layers. Write at \
least 600 words and do not use bullet points."""

FAMILIES = {"code": _CODE, "math": _MATH, "struct": _STRUCT, "prose": _PROSE}
FRIENDLY = ("code", "math", "struct")


def _unique(prompt: str, tag: str) -> str:
    """A fresh prefix per request so no two share a radix prefix. `os.urandom` rather than a
    counter: a counter shares a long common prefix across repeats, which is the very thing this
    is defeating."""
    return f"[req {os.urandom(8).hex()} {tag}]\n{prompt}"


def _post(url, body, timeout):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def _stream_one(url, model, prompt, max_tokens, args, barrier, out, i):
    """One streamed request -> (ttft, wall, completion_tokens). Barrier so an M>1 rung starts
    together and measures a genuinely concurrent batch rather than a staggered one."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        # SAMPLED. See the module docstring: greedy over-states draft acceptance.
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "ignore_eos": True,   # fixed decode length, so tok/s is not a length lottery
    }
    barrier.wait()
    t0 = time.perf_counter()
    ttft = None
    ntok = 0
    usage_tok = None
    try:
        r = _post(f"{url}/v1/chat/completions", body, args.timeout)
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                d = json.loads(payload)
            except Exception:
                continue
            if d.get("usage"):
                usage_tok = d["usage"].get("completion_tokens")
            ch = (d.get("choices") or [{}])[0]
            delta = ch.get("delta") or {}
            if delta.get("content") or delta.get("reasoning_content"):
                if ttft is None:
                    ttft = time.perf_counter() - t0
                ntok += 1
    except Exception as e:
        out[i] = {"error": f"{type(e).__name__}: {e}"}
        return
    wall = time.perf_counter() - t0
    out[i] = {"ttft": ttft, "wall": wall,
              "tokens": usage_tok if usage_tok else ntok,
              "chunks": ntok}


def _rung(url, model, family, M, max_tokens, args):
    barrier = threading.Barrier(M)
    out = [None] * M
    ths = []
    for i in range(M):
        p = _unique(FAMILIES[family], f"{family}-{i}")
        t = threading.Thread(target=_stream_one,
                             args=(url, model, p, max_tokens, args, barrier, out, i))
        t.start()
        ths.append(t)
    for t in ths:
        t.join()
    return out


def _metrics(url):
    """`vllm:*` counters as a flat dict. Returns {} for an engine that exposes none (ours)."""
    try:
        r = urllib.request.urlopen(f"{url}/metrics", timeout=10)
        txt = r.read().decode("utf-8", "replace")
    except Exception:
        return {}
    out = {}
    for line in txt.splitlines():
        if line.startswith("#") or " " not in line:
            continue
        name, _, val = line.rpartition(" ")
        try:
            out[name.split("{")[0]] = out.get(name.split("{")[0], 0.0) + float(val)
        except ValueError:
            pass
    return out


def _spec_delta(before, after):
    """Accepted tokens and drafts across a window, plus acceptance PER DRAFT OFFERED.

    Per draft offered, not per accepted token: a controller that widens its proposal raises the raw
    accepted count without necessarily doing better, so raw accept-len flatters a widening arm.
    """
    def d(k):
        return after.get(k, 0.0) - before.get(k, 0.0)
    acc = d("vllm:spec_decode_num_accepted_tokens_total") or d("vllm:spec_decode_num_accepted_tokens_per_pos")
    drafts = d("vllm:spec_decode_num_drafts_total")
    draft_tok = d("vllm:spec_decode_num_draft_tokens_total")
    out = {"accepted_tokens": acc, "drafts": drafts, "draft_tokens": draft_tok}
    if drafts > 0:
        out["accepted_per_draft"] = acc / drafts
    if draft_tok > 0:
        out["draft_token_accept_rate"] = acc / draft_tok
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--families", default="code,math,struct,prose")
    ap.add_argument("--decode-m", default="1,4")
    ap.add_argument("--decode-tokens", type=int, default=256)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--timeout", type=int, default=1200)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    fams = [f for f in args.families.split(",") if f in FAMILIES]
    ms = [int(x) for x in args.decode_m.split(",")]

    print(f"label={args.label or '(none)'} model={args.model} url={args.url}")
    print(f"sampled: temperature={args.temperature} top_p={args.top_p} top_k={args.top_k}; "
          f"decode_tokens={args.decode_tokens} reps={args.reps}\n")

    for _ in range(args.warmup):
        _rung(args.url, args.model, "prose", 1, 32, args)

    rows = []
    for fam in fams:
        for M in ms:
            before = _metrics(args.url)
            reps = []
            for _ in range(args.reps):
                res = [r for r in _rung(args.url, args.model, fam, M, args.decode_tokens, args)
                       if r and not r.get("error")]
                if not res:
                    continue
                wall = max(r["wall"] for r in res)
                toks = sum(r["tokens"] for r in res)
                ttfts = [r["ttft"] for r in res if r["ttft"] is not None]
                reps.append({"wall": wall, "tokens": toks,
                             "tok_s": toks / wall if wall else 0.0,
                             "ttft": statistics.median(ttfts) if ttfts else None})
            after = _metrics(args.url)
            if not reps:
                print(f"  {fam:7s} M={M}  ALL REPS FAILED")
                continue
            tok_s = statistics.median(r["tok_s"] for r in reps)
            ttfts = [r["ttft"] for r in reps if r["ttft"] is not None]
            row = {"family": fam, "M": M, "tok_s": tok_s,
                   "ttft_med": statistics.median(ttfts) if ttfts else None,
                   "reps": reps, "spec": _spec_delta(before, after)}
            rows.append(row)
            sp = row["spec"]
            extra = ""
            if sp.get("accepted_per_draft"):
                extra = (f"  accepted/draft={sp['accepted_per_draft']:.2f}"
                         f" rate={sp.get('draft_token_accept_rate', 0):.3f}")
            elif sp.get("accepted_tokens"):
                extra = f"  accepted_tokens={sp['accepted_tokens']:.0f}"
            ttft_s = f"{row['ttft_med']:.3f}s" if row["ttft_med"] is not None else "n/a"
            print(f"  {fam:7s} M={M}  {tok_s:7.2f} tok/s  ttft={ttft_s}"
                  + (extra or "   (no spec counters)"))

    friendly = [r["tok_s"] for r in rows if r["family"] in FRIENDLY and r["M"] == ms[0]]
    hostile = [r["tok_s"] for r in rows if r["family"] == "prose" and r["M"] == ms[0]]
    if friendly and hostile:
        print(f"\n  spec-friendly median {statistics.median(friendly):.2f} tok/s vs "
              f"prose control {hostile[0]:.2f} tok/s at M={ms[0]} "
              f"({statistics.median(friendly) / hostile[0]:.2f}x)")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"label": args.label, "model": args.model, "url": args.url,
                       "sampling": {"temperature": args.temperature, "top_p": args.top_p,
                                    "top_k": args.top_k},
                       "decode_tokens": args.decode_tokens, "reps": args.reps,
                       "rows": rows}, f, indent=1)
        print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
