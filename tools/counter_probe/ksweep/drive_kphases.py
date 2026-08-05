#!/usr/bin/env python3
"""Drive a serve through FIXED-LENGTH phases so step-indexed ROCTx windows land inside each one.

Phases, in order (each costs a KNOWN number of scheduler loop iterations, which is what the
MINISGL_ROCTX_WINDOWS arithmetic is placed against):

  warmup   M=1   WARM tokens          ->  1 + WARM iters
  A        M=1   TOK   tokens         ->  1 + TOK  iters      (served decode, batch 1)
  B        M=5   TOK   tokens         ->  <=5 + TOK iters     (served decode, batch 5)
  C        M=6   TOK   tokens         ->  <=6 + TOK iters     (served decode, batch 6 == max-running)
  P        NREQ sequential prefills   ->  NREQ * (1 + PTOK) iters

Why the phase lengths must be fixed: the window index is a scheduler LOOP-ITERATION counter, the only
counter the loop has before it knows what it is about to run. `ignore_eos` + a fixed `max_tokens`
makes one request cost exactly `max_tokens` decode iterations, so the boundaries land where they were
computed to land. "Generate until EOS" would move every window.

Why phase P's prompts are UNIQUE: a shared prefix would be served out of the radix cache and the
"prefill" phase would measure a cache hit, not prefill. Each prompt carries a distinct nonce prefix
and is sized to sit under the 8192-token chunk budget, so it is exactly ONE prefill iteration.

Why the LAST phase is expected to die: rocprofv3 is LD_PRELOADed into the engine and flushes only on
a normal interpreter exit, so MINISGL_EXIT_AFTER_STEPS is the only shutdown that writes the trace —
and it fires mid-request by construction. The trace is the artifact; the completion is not.

Timings here are the PROFILED wall when a profiler is attached. The unprofiled control leg is a
separate run of the same script against the same image.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import threading
import time
import urllib.request

DECODE_PROMPT = (
    "Explain, step by step and in depth, how a modern CPU executes a program: cover fetch/decode/"
    "execute, pipelining, branch prediction, caches, and out-of-order execution. Use clear prose."
)

# A word pool that tokenizes densely enough that ~1400 words lands near ~2000 tokens, which is well
# under the 8192-token chunk-prefill budget => exactly one prefill iteration per request.
_WORDS = (
    "memory bandwidth kernel occupancy register scratch workgroup dispatch latency throughput cache "
    "coherence pipeline scheduler tensor gradient attention embedding quantization inference decode "
    "prefill speculative rollback allocator fragmentation topology interconnect firmware microcode"
).split()


def _long_prompt(rng: random.Random, n_words: int) -> str:
    nonce = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(24))
    body = " ".join(rng.choice(_WORDS) for _ in range(n_words))
    return f"[{nonce}] Summarize the following technical notes in one word.\n{body}"


def _stream(url: str, prompt: str, max_tokens: int, barrier, out: list, idx: int) -> None:
    payload = json.dumps({
        "model": "", "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0.0, "ignore_eos": True, "stream": True,
    }).encode()
    req = urllib.request.Request(url + "/v1/chat/completions", data=payload,
                                 headers={"Content-Type": "application/json"})
    stamps: list[float] = []
    err = None
    if barrier is not None:
        barrier.wait()
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            for raw in r:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                if line[5:].strip() == "[DONE]":
                    break
                stamps.append(time.perf_counter())
    except Exception as exc:                                   # noqa: BLE001
        err = f"{type(exc).__name__}: {exc}"
    out[idx] = {"t0": t0, "ttft_s": (stamps[0] - t0) if stamps else None,
                "stamps": stamps, "err": err}


def _summarize(label: str, m: int, max_tokens: int, out: list, wall: float) -> dict:
    gaps: list[float] = []
    ntok = 0
    ttfts = [r["ttft_s"] for r in out if r and r.get("ttft_s") is not None]
    for r in out:
        if not r:
            continue
        s = r["stamps"]
        ntok += len(s)
        gaps += [(s[i] - s[i - 1]) * 1e3 for i in range(1, len(s))]
    res = {
        "label": label, "M": m, "max_tokens": max_tokens, "wall_s": wall, "tokens": ntok,
        # MEDIAN, not mean: a phase cut off by the step bound, or one that absorbs a scheduler
        # hiccup, puts multi-second gaps in the tail and the mean follows them.
        "gap_ms_median": statistics.median(gaps) if gaps else None,
        "gap_ms_p10": (statistics.quantiles(gaps, n=10)[0] if len(gaps) > 10 else None),
        "gap_ms_p90": (statistics.quantiles(gaps, n=10)[8] if len(gaps) > 10 else None),
        "n_gaps": len(gaps),
        "ttft_s_median": statistics.median(ttfts) if ttfts else None,
        "errors": [r["err"] for r in out if r and r.get("err")][:4],
    }
    # At M streams the server runs ONE decode step per token PER STREAM in a batch of M, so the
    # per-STEP wall is the per-stream inter-token gap and the aggregate rate is M times that.
    res["step_ms_from_gaps"] = res["gap_ms_median"]
    res["agg_tok_s"] = (ntok / wall) if wall > 0 else None
    return res


def decode_phase(url: str, m: int, max_tokens: int, label: str) -> dict:
    barrier = threading.Barrier(m)
    out: list = [None] * m
    threads = [threading.Thread(target=_stream,
                                args=(url, DECODE_PROMPT, max_tokens, barrier, out, i))
               for i in range(m)]
    w0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return _summarize(label, m, max_tokens, out, time.perf_counter() - w0)


def prefill_phase(url: str, nreq: int, n_words: int, ptok: int, seed: int, label: str) -> dict:
    """SEQUENTIAL long-prompt requests, one at a time, each a single prefill iteration.

    Sequential (not concurrent) so the iteration cost per request is deterministic: `1 + ptok`. A
    concurrent burst would let the scheduler co-batch an unknown number of prefills per iteration.
    """
    rng = random.Random(seed)
    out: list = [None] * nreq
    w0 = time.perf_counter()
    for i in range(nreq):
        _stream(url, _long_prompt(rng, n_words), ptok, None, out, i)
    res = _summarize(label, 1, ptok, out, time.perf_counter() - w0)
    res["prompt_words"] = n_words
    res["n_requests"] = nreq
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--warmup-tokens", type=int, default=16)
    ap.add_argument("--tok", type=int, default=500)
    ap.add_argument("--m-list", default="1,5,6")
    ap.add_argument("--prefill-reqs", type=int, default=100)
    ap.add_argument("--prefill-words", type=int, default=1400)
    ap.add_argument("--prefill-tokens", type=int, default=3)
    ap.add_argument("--skip-prefill", action="store_true")
    ap.add_argument("--ready-timeout", type=int, default=1200)
    args = ap.parse_args()

    deadline = time.time() + args.ready_timeout
    ready = False
    while time.time() < deadline:
        try:
            urllib.request.urlopen(args.url + "/v1/models", timeout=3).read()
            ready = True
            break
        except Exception:                                      # noqa: BLE001
            time.sleep(3)
    print(f"[drive] ready={ready}", flush=True)
    if not ready:
        raise SystemExit(f"server at {args.url} not ready within {args.ready_timeout}s")

    results = []
    iters = 0

    def record(r: dict, cost: int) -> None:
        nonlocal iters
        r["iter_start_est"] = iters + 1
        iters += cost
        r["iter_end_est"] = iters
        print(f"[drive] {r['label']}: iters~{r['iter_start_est']}..{r['iter_end_est']} "
              f"{json.dumps({k: v for k, v in r.items() if k != 'stamps'})}", flush=True)
        results.append(r)

    # Warmup is COUNTED, not free: it costs loop iterations like any other traffic, and the window
    # placement is arithmetic over those iterations.
    w = decode_phase(args.url, 1, args.warmup_tokens, "warmup")
    record(w, 1 + args.warmup_tokens)
    time.sleep(2.0)          # the loop BLOCKS when idle, so idling consumes no iterations

    for m in [int(x) for x in args.m_list.split(",") if x.strip()]:
        r = decode_phase(args.url, m, args.tok, f"decode_bs{m}")
        record(r, m + args.tok)
        time.sleep(2.0)

    if not args.skip_prefill:
        p = prefill_phase(args.url, args.prefill_reqs, args.prefill_words,
                          args.prefill_tokens, 1234, "prefill")
        record(p, args.prefill_reqs * (1 + args.prefill_tokens))

    with open(args.out, "w") as f:
        json.dump({"url": args.url,
                   "phases": [{k: v for k, v in r.items() if k != "stamps"} for r in results]},
                  f, indent=1)
    print(f"[drive] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
