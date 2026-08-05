#!/usr/bin/env python
"""A/B serve bench for the dense-tile / act-quant question: does a faster W4A8 dense GEMM move the
SERVED path? Byte-identical copy lives in both legs' worktrees so the two runs differ only in the
engine+kernels under test.

Why not serve_matrix_bench.py: that harness reuses ONE fixed long prompt for every prefill cell, so
after its own warmup every subsequent prefill is a RADIX PREFIX-CACHE HIT and the reported TTFT is
the cost of a cache lookup, not of a prefill. That is fine for its purpose and fatal for this one —
the whole hypothesis is about prefill GEMM cost. Every prompt here is prefixed with fresh random
hex, so no two requests share a prefix and every prefill is genuinely computed.

  prefill rung : unique long prompt, max_tokens=1, stream -> TTFT, prefill tok/s = prompt_tok / TTFT
  decode  rung : unique short prompt, max_tokens=D, ignore_eos -> TPOT, tok/s = D*M / wall

Stdlib only. Emits a human table plus a JSON blob (--out) with every raw repeat, so medians are
recomputable from the fixture rather than trusted.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import threading
import time
import urllib.request

_CTX_PARA = (
    "The transformer architecture, introduced in 2017, replaced recurrence with self-attention, "
    "letting every token attend to every other token in a sequence in parallel. This removed the "
    "sequential bottleneck of RNNs and made large-scale pre-training practical on modern accelerators. "
    "A transformer block interleaves multi-head attention with a position-wise feed-forward network, "
    "each wrapped in residual connections and layer normalization. Attention computes query, key, and "
    "value projections, scores every query against every key, normalizes with softmax, and mixes the "
    "values accordingly. Multiple heads let the model attend to different relationships at once. "
    "Because attention is permutation-invariant, positional information is injected through learned or "
    "rotary position encodings. Scaling laws showed that loss falls predictably as parameters, data, "
    "and compute grow together, motivating ever-larger models. Mixture-of-experts layers scale capacity "
    "without scaling per-token compute: a router sends each token to a small subset of expert feed-"
    "forward networks, so only a fraction of the weights activate per token. Linear-attention and "
    "state-space hybrids further cut the quadratic cost of full attention for long contexts. Inference "
    "is dominated by the autoregressive decode loop, where each step produces one token conditioned on "
    "all previous ones, making it memory-bandwidth bound at small batch sizes. Speculative decoding "
    "accelerates this by drafting several tokens cheaply and verifying them in a single parallel pass. "
)
_SHORT = ("Explain, step by step and in depth, how a modern CPU executes a program: cover fetch/"
          "decode/execute, pipelining, branch prediction, caches, and out-of-order execution.")


def _nonce() -> str:
    """Fresh, high-entropy prefix. Defeats the radix prefix cache: the match is from token 0, so a
    differing FIRST token makes the whole prompt a miss and the prefill real."""
    return "Reference id " + os.urandom(16).hex() + ". "


def _long(words: int) -> str:
    """Context of ~`words` words, sliced at WORD granularity rather than by repeating the whole
    paragraph. Paragraph-granular repetition quantizes the prompt to ~333-token steps, and the
    band this A/B is about (the dense GEMM's M, i.e. the prompt length) includes specific points —
    M~512 is where the one known dense regression sits — that a 333-token grid simply steps over."""
    w = _CTX_PARA.split()
    body = (w * (words // len(w) + 1))[:words]
    return (_nonce() + " ".join(body)
            + "\nBased on the passage above, explain speculative decoding in your own words.")


def _stream(url, prompt, max_tokens, ignore_eos, barrier):
    payload = json.dumps({
        "model": "", "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0.0, "ignore_eos": ignore_eos,
        "stream": True, "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(url + "/v1/chat/completions", data=payload,
                                 headers={"Content-Type": "application/json"})
    barrier.wait()
    t0 = time.perf_counter()
    stamps, usage = [], None
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            try:
                d = json.loads(body)
            except Exception:  # noqa: BLE001
                continue
            # include_usage sends a FINAL choices-empty chunk carrying usage; it is not a token.
            if d.get("choices"):
                stamps.append(time.perf_counter())
            if d.get("usage"):
                usage = d["usage"]
    return {"ttft": (stamps[0] - t0) if stamps else None, "n": len(stamps),
            "stamps": stamps, "usage": usage}


def _run(url, M, prompt_fn, max_tokens, ignore_eos):
    barrier = threading.Barrier(M)
    out = [None] * M
    prompts = [prompt_fn() for _ in range(M)]   # every stream gets its OWN uncached prompt

    def work(i):
        try:
            out[i] = _stream(url, prompts[i], max_tokens, ignore_eos, barrier)
        except Exception as e:  # noqa: BLE001
            out[i] = {"error": repr(e)}

    ths = [threading.Thread(target=work, args=(i,)) for i in range(M)]
    w0 = time.perf_counter()
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    wall = time.perf_counter() - w0
    ok = [r for r in out if r and "error" not in r and r["n"] > 0]
    return ok, wall, [r for r in out if not r or "error" in r]


def _tpot_ms(ok):
    gaps = []
    for r in ok:
        s = r["stamps"]
        gaps += [(s[i] - s[i - 1]) * 1000 for i in range(1, len(s))]
    return statistics.mean(gaps) if gaps else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--decode-tokens", type=int, default=128)
    # ~128 / ~512 / ~1024 / ~2048 PROMPT TOKENS on this prose (~1.4 tok/word + chat template).
    # That is the M band the dense tile work targets, and it lands a rung on M~512 where the one
    # known dense regression (glm.gate_up tp2, 0.8141x) was measured in isolation.
    ap.add_argument("--prefill-words", default="90,360,730,1460")
    ap.add_argument("--decode-m", default="1,4,16")
    ap.add_argument("--out", default="")
    ap.add_argument("--ready-timeout", type=int, default=900)
    args = ap.parse_args()

    dl = time.time() + args.ready_timeout
    while time.time() < dl:
        try:
            urllib.request.urlopen(args.url + "/v1", timeout=3)
            break
        except Exception:  # noqa: BLE001
            time.sleep(3)
    else:
        raise SystemExit("server not ready")

    words = [int(w) for w in args.prefill_words.split(",")]
    dms = [int(m) for m in args.decode_m.split(",")]
    D = args.decode_tokens

    print(f"[warmup] {args.label}", flush=True)
    _run(args.url, 1, lambda: _long(words[0]), 1, False)
    _run(args.url, 1, lambda: _nonce() + _SHORT, D, True)

    rec = {"label": args.label, "reps": args.reps, "decode_tokens": D,
           "prefill": {}, "decode": {}}

    for rep in range(args.reps):
        print(f"\n===== rep {rep + 1}/{args.reps}  [{args.label}] =====", flush=True)
        print(f"{'workload':>12} {'ptok':>6} {'TTFT ms':>9} {'TPOT ms':>9} {'tok/s':>10}")
        for w in words:
            ok, wall, bad = _run(args.url, 1, lambda w=w: _long(w), 1, False)
            if not ok:
                print(f"{'prefill w' + str(w):>12} FAILED {bad}")
                continue
            ptok = int((ok[0]["usage"] or {}).get("prompt_tokens") or 0)
            ttft = ok[0]["ttft"] * 1000
            tps = ptok / ok[0]["ttft"] if ptok else float("nan")
            print(f"{'prefill w' + str(w):>12} {ptok:>6} {ttft:9.2f} {'--':>9} {tps:10.1f}",
                  flush=True)
            k = f"w{w}"
            rec["prefill"].setdefault(k, {"ptok": [], "ttft_ms": [], "tok_s": []})
            rec["prefill"][k]["ptok"].append(ptok)
            rec["prefill"][k]["ttft_ms"].append(ttft)
            rec["prefill"][k]["tok_s"].append(tps)
        for M in dms:
            ok, wall, bad = _run(args.url, M, lambda: _nonce() + _SHORT, D, True)
            if not ok:
                print(f"{'decode M' + str(M):>12} FAILED {bad}")
                continue
            tpot = _tpot_ms(ok)
            tps = D * len(ok) / wall
            print(f"{'decode M' + str(M):>12} {'--':>6} {'--':>9} {tpot:9.2f} {tps:10.1f}",
                  flush=True)
            k = f"M{M}"
            rec["decode"].setdefault(k, {"tpot_ms": [], "tok_s": []})
            rec["decode"][k]["tpot_ms"].append(tpot)
            rec["decode"][k]["tok_s"].append(tps)

    print(f"\n===== MEDIANS ({args.reps} reps) [{args.label}] =====")
    print(f"{'cell':>14} {'ptok':>6} {'TTFT ms':>9} {'TPOT ms':>9} {'tok/s':>10}")
    for k, v in rec["prefill"].items():
        print(f"{'prefill ' + k:>14} {int(statistics.median(v['ptok'])):>6} "
              f"{statistics.median(v['ttft_ms']):9.2f} {'--':>9} "
              f"{statistics.median(v['tok_s']):10.1f}")
    for k, v in rec["decode"].items():
        print(f"{'decode ' + k:>14} {'--':>6} {'--':>9} "
              f"{statistics.median(v['tpot_ms']):9.2f} {statistics.median(v['tok_s']):10.1f}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(rec, f, indent=1)
        print(f"[out] {args.out}")


if __name__ == "__main__":
    main()
