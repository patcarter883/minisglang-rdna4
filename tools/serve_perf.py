"""Comprehensive PERFORMANCE profile of a RUNNING minisgl serve.

Read-only over HTTP — starts/stops/reconfigures nothing, so it needs no GPU lease of its own.

    python3 tools/serve_perf.py [--base http://localhost:1919] [--conc 4] [--container <name>]

METHODOLOGY (each rule here exists because breaking it produced a wrong number in this repo):

  * TRUE tok/s = usage.completion_tokens / wall. NEVER count SSE chunks: under spec decode ONE chunk
    carries a whole accepted block, so chunk-counting under-reports by the accept-len factor (that
    bug once read 39.7 tok/s as 14.9).
  * WARMUP is discarded everywhere. The first request pays graph-replay warmup and lazy allocation.
  * Concurrency is driven at the serve's OWN max_running_requests. Above it you measure the
    admission queue, not the engine.
  * Prompts carry a DISTINCT suffix per request, so concurrent requests do not collapse onto one
    shared radix prefix and accidentally measure prefix-cache hits instead of decode. The prefix-cache
    section deliberately does the opposite.
  * A REASONING model spends most of its budget thinking; reasoning tokens are counted in
    completion_tokens and ARE real decode work, so they belong in tok/s. But max_tokens must be large
    enough to finish, or the run measures a truncated thought.
  * TTFT is measured from the first delta of EITHER reasoning_content or content — on a reasoning
    model the first *answer* token can be hundreds of tokens into the stream.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

SHORT = ("Explain how speculative decoding works in a language model inference engine, "
         "covering the drafter, the verify step, and why acceptance matters.")


def post(base, body, timeout=1800):
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def gen(base, model, prompt, max_tokens, temperature=0.0, seed=1234):
    t0 = time.perf_counter()
    d = post(base, {"model": model, "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens, "temperature": temperature, "seed": seed,
                    "stream": False})
    w = time.perf_counter() - t0
    return d["usage"]["completion_tokens"], d["usage"]["prompt_tokens"], w


def stream_ttft(base, model, prompt, max_tokens):
    """(ttft, total_wall, n_tokens). TTFT = first delta of reasoning OR content."""
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0.0, "seed": 1234, "stream": True}
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft, n = None, 0
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            p = line[5:].strip()
            if p == "[DONE]":
                break
            delta = json.loads(p)["choices"][0].get("delta", {})
            if delta.get("content") or delta.get("reasoning_content"):
                if ttft is None:
                    ttft = time.perf_counter() - t0
                n += 1
    return ttft, time.perf_counter() - t0, n


def filler(n_facts):
    return "\n".join(f"Fact {i}: item {i} has value {i * 7 % 97}." for i in range(n_facts))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:1919")
    ap.add_argument("--conc", type=int, default=4)
    ap.add_argument("--container", default="lease-minisgl-serve-serve")
    ap.add_argument("--maxtok", type=int, default=512)
    a = ap.parse_args()
    B, MT = a.base, a.maxtok
    model = json.loads(urllib.request.urlopen(B + "/v1/models", timeout=30).read())["data"][0]["id"]
    print(f"model: {model}\nconc:  {a.conc}   max_tokens: {MT}\n" + "=" * 76)

    gen(B, model, "Say hello.", 32)  # warmup, discarded

    print("\n1. SINGLE-STREAM (bs=1) — the latency-sensitive case")
    runs = [gen(B, model, SHORT, MT) for _ in range(3)]
    tps = [t / w for t, _, w in runs]
    print(f"   tokens={runs[0][0]}  wall={statistics.median(w for _,_,w in runs):.2f}s")
    print(f"   tok/s  median={statistics.median(tps):.1f}  min={min(tps):.1f}  max={max(tps):.1f}"
          f"  spread={100*(max(tps)-min(tps))/statistics.median(tps):.1f}%")

    print("\n2. TTFT / TPOT (streaming; TTFT counts the first reasoning OR content delta)")
    # TPOT MUST be per TOKEN, not per CHUNK. Under spec decode one SSE chunk carries a whole accepted
    # block, so dividing by the chunk count reports ~1/accept-len of the real rate (measured: 30.7
    # ms/"token" = 32.6 tok/s against a true 84 tok/s). Take the token count from a matched
    # NON-streamed call — greedy + same seed makes it the identical continuation.
    tt = [stream_ttft(B, model, SHORT, MT) for _ in range(3)]
    ntok = gen(B, model, SHORT, MT)[0]
    ttfts = [x[0] for x in tt if x[0]]
    tpots = [(w - f) / max(ntok - 1, 1) * 1000 for f, w, n in tt if f]
    nchunk = statistics.median(n for _, _, n in tt)
    print(f"   TTFT median={statistics.median(ttfts)*1000:.0f} ms   "
          f"TPOT median={statistics.median(tpots):.2f} ms/token "
          f"({1000/statistics.median(tpots):.1f} tok/s decode-only)")
    print(f"   [{ntok} tokens arrived in {nchunk:.0f} SSE chunks = {ntok/nchunk:.2f} tok/chunk — "
          f"chunk-counting would have reported {1000/((statistics.median(w-f for f,w,_ in tt))/max(nchunk-1,1)*1000):.1f} tok/s]")

    print("\n3. THROUGHPUT SCALING (distinct prompts — no radix sharing)")
    base_tps = None
    for n in sorted({1, 2, a.conc}):
        def one(i):
            return gen(B, model, SHORT + f"\n\n(Variant {i}: emphasise point {i+1}.)", MT)[0]
        t0 = time.perf_counter()
        with ThreadPoolExecutor(n) as ex:
            toks = sum(ex.map(one, range(n)))
        w = time.perf_counter() - t0
        if base_tps is None:
            base_tps = toks / w
        print(f"   bs={n:<2} {toks:5d} tok / {w:6.2f}s = {toks/w:7.1f} tok/s"
              f"   ({toks/w/base_tps:.2f}x vs bs=1,  {toks/w/n:6.1f} tok/s per stream)")

    print("\n4. PREFILL vs PROMPT LENGTH (prefill tok/s = prompt_tokens / TTFT)")
    for nf in (0, 200, 600, 1200):
        p = (filler(nf) + "\n\n" if nf else "") + "Summarise in one sentence."
        f, w, _ = stream_ttft(B, model, p, 128)
        pt = gen(B, model, p, 8)[1]
        print(f"   {pt:6d} prompt tok -> TTFT {f*1000:7.0f} ms = {pt/f:8.0f} prefill tok/s")

    print("\n5. DECODE vs CONTEXT DEPTH (same output budget, deeper context)")
    for nf in (0, 400, 1000):
        p = (filler(nf) + "\n\n" if nf else "") + SHORT
        t, pt, w = gen(B, model, p, MT)
        print(f"   ctx~{pt:6d} tok -> {t:4d} out in {w:6.2f}s = {t/w:7.1f} tok/s")

    print("\n6. PREFIX CACHE (radix): same long prefix, different suffix")
    pref = filler(800)
    cold = gen(B, model, pref + "\n\nQuestion A: summarise.", 64)
    warm = [gen(B, model, pref + f"\n\nQuestion {c}: summarise.", 64) for c in "BCD"]
    print(f"   cold  {cold[1]:6d} prompt tok  {cold[2]:6.2f}s")
    print(f"   warm  median {statistics.median(w for _,_,w in warm):6.2f}s  "
          f"-> {100*(1-statistics.median(w for _,_,w in warm)/cold[2]):+.0f}% wall vs cold")

    print("\n7. LATENCY PERCENTILES UNDER SUSTAINED LOAD")
    lat = []
    def timed(i):
        t0 = time.perf_counter()
        gen(B, model, SHORT + f"\n\n(Run {i}.)", 192)
        return time.perf_counter() - t0
    with ThreadPoolExecutor(a.conc) as ex:
        lat = list(ex.map(timed, range(a.conc * 3)))
    lat.sort()
    p = lambda q: lat[min(int(q * len(lat)), len(lat) - 1)]
    print(f"   n={len(lat)}  p50={p(.5):.2f}s  p90={p(.9):.2f}s  p99={p(.99):.2f}s  max={lat[-1]:.2f}s")

    print("\n8. SPEC-DECODE EFFECTIVENESS (DELTA over a controlled workload)")
    # The engine's [spec] counters are CUMULATIVE SINCE BOOT. Reading them raw mixes in whatever else
    # the serve has handled — an acceptance run full of max_tokens=8 requests inflates the
    # zero-draft/eager share, because a near-exhausted budget leaves nothing to draft. Snapshot,
    # drive a known workload, and report the difference.
    def _counters():
        try:
            o = subprocess.run(["docker", "logs", "--tail", "6000", a.container],
                               capture_output=True, text=True, timeout=60)
            t = o.stdout + o.stderr
            w = [l for l in t.splitlines() if "verify-width[" in l]
            if not w:
                return None
            r = re.search(r"replay=(\d+) eager=(\d+)", w[-1])
            h = re.search(r"verify-width\[([^\]]*)\]", w[-1])
            return (int(r.group(1)), int(r.group(2)), h.group(1)) if r else None
        except Exception:  # noqa: BLE001
            return None
    before = _counters()
    for i in range(a.conc):
        gen(B, model, SHORT + f"\n\n(Spec probe {i}.)", MT)
    after = _counters()
    if before and after:
        dr, de = after[0] - before[0], after[1] - before[1]
        tot = dr + de
        print(f"   over {a.conc} full-length requests: verify replay={dr} eager={de}"
              + (f"  ({100*de/tot:.1f}% EAGER)" if tot else ""))
        print(f"   realized widths now: {after[2]}")
        print(f"   (cumulative-since-boot would have read: replay={after[0]} eager={after[1]})")
    else:
        print("   (no [spec] counters in the log — MINISGL_SPEC_DEBUG not enabled?)")
    try:
        out = subprocess.run(["docker", "logs", "--tail", "4000", a.container],
                             capture_output=True, text=True, timeout=60)
        log = out.stdout + out.stderr
        step = [l for l in log.splitlines() if "[spec] step=" in l]
        width = [l for l in log.splitlines() if "verify-width[" in l]
        if step:
            m = re.search(r"accept_rate=([\d.]+).*emitted/step=([\d.]+).*reqs/step=([\d.]+)", step[-1])
            if m:
                print(f"   accept_rate={m.group(1)}  emitted/step={m.group(2)}  reqs/step={m.group(3)}")
        if width:
            w = re.search(r"verify-width\[([^\]]*)\]", width[-1])
            r = re.search(r"replay=(\d+) eager=(\d+)", width[-1])
            if w:
                print(f"   realized widths: {w.group(1)}")
            if r:
                print(f"   verify graph: replay={r.group(1)} eager={r.group(2)}"
                      + ("  <-- EAGER FALLBACK" if int(r.group(2)) else "  (fully captured)"))
        if not step and not width:
            print("   (no [spec] counters in the log — MINISGL_SPEC_DEBUG not enabled?)")
    except Exception as e:  # noqa: BLE001
        print(f"   (could not read container log: {type(e).__name__})")

    print("\n" + "=" * 76)
    return 0


if __name__ == "__main__":
    sys.exit(main())
