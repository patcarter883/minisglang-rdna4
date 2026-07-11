#!/usr/bin/env python
"""Serving benchmark matrix for the minisgl OpenAI server: prefill / decode / mixed, TPOT +
throughput, swept over concurrency M in {1,2,4,8,16}. Streams /v1/chat/completions and timestamps
each SSE token chunk (TTFT = first chunk; TPOT = mean inter-token gap; throughput = total output
tokens / wall). Stdlib only (urllib + threads); no HF tokenizer load.

  python tools/serve_matrix_bench.py --url http://127.0.0.1:21009 --label 35b [--m 1,2,4,8,16] \
      [--workloads prefill,decode,mixed] [--decode-tokens 128] [--prefill-words 480]

Workloads (concurrency M = simultaneous streams, fired via a barrier):
  prefill : long prompt, max_tokens=1   -> TTFT (= prefill latency) + prefill tok/s (prompt/ TTFT)
  decode  : short prompt, max_tokens=D, ignore_eos -> TPOT + decode tok/s
  mixed   : long prompt, max_tokens=D, ignore_eos   -> TTFT + TPOT + total output tok/s
"""
from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.request

# Real-world-representative prompts (chat format), NOT "benchmark benchmark …" garbage — spec
# accept-len depends heavily on how predictable the generated text is, so a repetitive filler prompt
# gives an unrepresentative (usually too-easy) accept-len. A genuine instruction produces structured
# natural-language generation, which is what a served workload actually decodes.
_DECODE_PROMPT = (
    "Explain, step by step and in depth, how a modern CPU executes a program: cover fetch/decode/"
    "execute, pipelining, branch prediction, caches, and out-of-order execution. Use clear prose."
)
# A realistic long CONTEXT (~480 tokens of coherent prose) for the prefill / mixed (RAG-style) workloads.
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


def _messages(prompt: str):
    return [{"role": "user", "content": prompt}]


def _stream(url: str, prompt: str, max_tokens: int, ignore_eos: bool, barrier: threading.Barrier):
    payload = json.dumps({
        "model": "", "messages": _messages(prompt), "max_tokens": max_tokens,
        "temperature": 0.0, "ignore_eos": ignore_eos, "stream": True,
    }).encode()
    req = urllib.request.Request(url + "/v1/chat/completions", data=payload,
                                 headers={"Content-Type": "application/json"})
    barrier.wait()                       # release all M streams together
    t0 = time.perf_counter()
    stamps = []                          # arrival time of each token chunk
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            stamps.append(time.perf_counter())
    return {"t0": t0, "ttft": (stamps[0] - t0) if stamps else None,
            "n": len(stamps), "stamps": stamps, "end": stamps[-1] if stamps else t0}


def _run(url: str, M: int, prompt: str, max_tokens: int, ignore_eos: bool):
    barrier = threading.Barrier(M)
    out, threads = [None] * M, []

    def work(i):
        try:
            out[i] = _stream(url, prompt, max_tokens, ignore_eos, barrier)
        except Exception as e:  # noqa: BLE001
            out[i] = {"error": repr(e)}

    wall0 = time.perf_counter()
    for i in range(M):
        t = threading.Thread(target=work, args=(i,))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    wall = time.perf_counter() - wall0
    ok = [r for r in out if r and "error" not in r and r["n"] > 0]
    return ok, wall, [r for r in out if not r or "error" in r]


def _agg_tpot(ok):
    # mean inter-token gap (ms) across all streams with >=2 tokens
    gaps = []
    for r in ok:
        s = r["stamps"]
        gaps += [(s[i] - s[i - 1]) * 1000 for i in range(1, len(s))]
    return (sum(gaps) / len(gaps)) if gaps else float("nan")


def _prompt_tokens(url: str, prompt: str) -> int:
    """Exact prompt-token count from one non-stream request's usage.prompt_tokens; word-count
    fallback if the server omits usage. PREFILL throughput = these tokens / TTFT, NOT the 1 output
    token — the old harness reported output tok/s for prefill (~M/wall), which is meaningless."""
    payload = json.dumps({"model": "", "messages": _messages(prompt), "max_tokens": 1,
                          "temperature": 0.0, "stream": False}).encode()
    req = urllib.request.Request(url + "/v1/chat/completions", data=payload,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            d = json.loads(r.read())
        pt = (d.get("usage") or {}).get("prompt_tokens")
        if pt:
            return int(pt)
    except Exception:  # noqa: BLE001
        pass
    return max(1, len(prompt.split()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--m", default="1,2,4,8,16")
    ap.add_argument("--max-concurrency", type=int, default=0,
                    help="the concurrency ceiling = min(--max-running-requests, --cuda-graph-max-bs): "
                         "beyond max-running a request QUEUES (not concurrent), beyond graph-max-bs a "
                         "decode batch falls to EAGER (not graph-captured). When set (>0) the sweep is "
                         "capped to M <= this AND this value is appended as the top point, so every "
                         "measurement is genuinely concurrent AND graph-captured.")
    ap.add_argument("--workloads", default="prefill,decode,mixed")
    ap.add_argument("--decode-tokens", type=int, default=128)
    ap.add_argument("--prefill-words", type=int, default=480)
    ap.add_argument("--ready-timeout", type=int, default=600)
    args = ap.parse_args()
    # wait for the server to come up (load can take minutes for the 35B)
    deadline = time.time() + args.ready_timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(args.url + "/v1", timeout=3)
            break
        except Exception:  # noqa: BLE001
            time.sleep(3)
    else:
        raise SystemExit(f"server at {args.url} not ready within {args.ready_timeout}s")

    Ms = [int(x) for x in args.m.split(",")]
    if args.max_concurrency > 0:
        # Keep only sweep points the server can run WITHOUT queueing, and include the ceiling itself
        # as the top point (real max-concurrency throughput). E.g. cap=24 -> …,16,24; cap=6 -> 1,2,4,6.
        dropped = [m for m in Ms if m > args.max_concurrency]
        Ms = sorted({m for m in Ms if m <= args.max_concurrency} | {args.max_concurrency})
        if dropped:
            print(f"[bench] max-running-requests={args.max_concurrency}: dropped M={dropped} "
                  f"(would queue, not concurrent); sweeping M={Ms}", flush=True)
    wls = args.workloads.split(",")
    # decode: a genuine instruction (short prompt, real generation). prefill/mixed: a realistic long
    # CONTEXT (~prefill_words tokens of coherent prose, by repeating the paragraph) + a question over it.
    short_prompt = _DECODE_PROMPT
    reps = max(1, round(args.prefill_words / max(1, len(_CTX_PARA.split()))))
    long_prompt = (_CTX_PARA * reps) + "\nBased on the passage above, explain speculative decoding " \
                                       "and why decode is memory-bandwidth bound, in your own words."
    D = args.decode_tokens

    # Warm up: absorb first-request JIT/cold-compile so it doesn't skew the M=1 prefill cell.
    print("[warmup] firing 2 warmup requests (prefill + decode shapes) ...", flush=True)
    _run(args.url, 1, long_prompt, 1, False)
    _run(args.url, 1, short_prompt, D, True)

    spec = {
        "prefill": (long_prompt, 1, False),
        "decode": (short_prompt, D, True),
        "mixed": (long_prompt, D, True),
    }
    # PREFILL throughput needs the real prompt-token count (measured once from the server's usage).
    prefill_ptoks = _prompt_tokens(args.url, long_prompt)
    print(f"\n===== serve matrix [{args.label}] {args.url}  "
          f"(prefill={prefill_ptoks} prompt tok, decode={D} tok) =====")
    print("  tok/s: prefill = prompt tokens processed / wall (prompt throughput); "
          "decode/mixed = output tokens / wall")
    for wl in wls:
        prompt, maxtok, ieos = spec[wl]
        print(f"\n--- {wl} ---")
        print(f"{'M':>3} {'TTFT ms':>9} {'TPOT ms':>9} {'tok/s':>12} {'fails':>6}")
        for M in Ms:
            ok, wall, bad = _run(args.url, M, prompt, maxtok, ieos)
            if not ok:
                print(f"{M:>3} {'--':>9} {'--':>9} {'--':>12} {len(bad):>6}  ALL FAILED")
                continue
            ttft = sum(r["ttft"] for r in ok) / len(ok) * 1000
            tpot = _agg_tpot(ok)
            # prefill: total PROMPT tokens processed / wall (real prefill throughput). decode/mixed:
            # output tokens / wall. wall is the concurrent wall-clock for all M streams.
            if wl == "prefill":
                tps = prefill_ptoks * len(ok) / wall
            else:
                tps = sum(r["n"] for r in ok) / wall
            tpot_s = f"{tpot:9.2f}" if tpot == tpot else f"{'n/a':>9}"
            print(f"{M:>3} {ttft:9.2f} {tpot_s} {tps:12.1f} {len(bad):>6}")


if __name__ == "__main__":
    main()
