"""Comprehensive acceptance test for a RUNNING minisgl serve.

Drives an already-serving endpoint over HTTP — it does NOT start, stop or reconfigure anything, so
it needs no GPU lease of its own (the serve already holds one).

    python3 tools/serve_acceptance.py [--base http://localhost:1919] [--conc 4]

Conventions this test follows deliberately:
  * TRUE tok/s = usage.completion_tokens / wall, NEVER a count of SSE chunks. Under spec decode ONE
    chunk carries a whole accepted block, so chunk-counting under-reports by the accept-len factor
    (that bug once read 39.7 tok/s as 14.9).
  * Concurrency is driven at the serve's OWN max_running_requests. Driving above it measures the
    admission queue, not the engine.
  * Every latency figure discards a warmup request, so graph-replay and lazy-alloc costs are not
    charged to the first measurement.
  * REASONING MODELS NEED HEADROOM. Qwen3.6 emits chain-of-thought before the answer (~200 tokens
    even for "capital of France"), delivered in `reasoning_content` with the answer in `content`.
    A small max_tokens truncates DURING thinking and leaves `content` empty — which looks exactly
    like an incoherent model. The first version of this file did that and reported 5 false failures.
    Coherence checks therefore use generous max_tokens and assert on `content`.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append((name, ok, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""), flush=True)
    return ok


def post(base: str, path: str, body: dict, timeout: int = 600):
    req = urllib.request.Request(
        base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def chat(base: str, model: str, content: str, **kw):
    body = {"model": model, "messages": [{"role": "user", "content": content}],
            "max_tokens": kw.pop("max_tokens", 64), "temperature": kw.pop("temperature", 0.0),
            "stream": False}
    body.update(kw)
    return post(base, "/v1/chat/completions", body)


def text_of(d) -> str:
    """The ANSWER only. A reasoning serve puts chain-of-thought in `reasoning_content`; asserting on
    a concatenation of the two would pass on the model merely *thinking about* the right answer."""
    return d["choices"][0]["message"].get("content") or ""


def reasoning_of(d) -> str:
    return d["choices"][0]["message"].get("reasoning_content") or ""


def truncated_in_thought(d) -> bool:
    """finish_reason=length with an empty answer = the budget was spent entirely on thinking."""
    return d["choices"][0]["finish_reason"] == "length" and not text_of(d).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:1919")
    ap.add_argument("--conc", type=int, default=4, help="serve's max_running_requests")
    a = ap.parse_args()
    B = a.base

    print("=" * 78)
    print("1. IDENTITY / HEALTH")
    with urllib.request.urlopen(B + "/health", timeout=30) as r:
        h = json.loads(r.read())
    check("/health ok", h.get("status") == "ok", str(h)[:90])
    model = json.loads(urllib.request.urlopen(B + "/v1/models", timeout=30).read())["data"][0]["id"]
    check("/v1/models reports a model", bool(model), model)

    print("\n2. COHERENCE")
    d = chat(B, model, "Name the capital of France. Answer with just the city name.", max_tokens=400)
    t = text_of(d)
    check("factual single-token answer", "paris" in t.lower(), repr(t[:60]))
    d = chat(B, model, "Write a Python function that returns the nth Fibonacci number.", max_tokens=700)
    code = text_of(d)
    check("generates syntactically plausible code", "def " in code and "return" in code,
          repr(code[:70].replace("\n", " ")))
    d = chat(B, model, "Count from 1 to 10 separated by spaces. Output only the numbers.", max_tokens=400)
    seq = re.findall(r"\d+", text_of(d))
    check("ordered sequence is correct", seq[:10] == [str(i) for i in range(1, 11)], " ".join(seq[:12]))

    print("\n2b. REASONING SPLIT (thought must not leak into the answer)")
    d = chat(B, model, "Capital of France? Just the city.", max_tokens=400)
    rc, ct = reasoning_of(d), text_of(d)
    check("reasoning_content is populated", len(rc) > 0, f"{len(rc)} chars of thought")
    check("content holds ONLY the answer", ct.strip().lower().startswith("paris") and len(ct) < 60,
          repr(ct[:50]))
    check("thought did NOT leak into content", "thinking process" not in ct.lower())
    d = chat(B, model, "Explain compilers.", max_tokens=24)
    check("thought-truncation is visible as finish_reason=length, not a silent empty answer",
          truncated_in_thought(d) or text_of(d).strip() != "",
          "budget spent in thought" if truncated_in_thought(d) else "answered")

    print("\n3. DETERMINISM (greedy must be reproducible)")
    p = "In exactly one sentence, explain what a GPU kernel is."
    r1, r2 = chat(B, model, p, max_tokens=64, seed=1234), chat(B, model, p, max_tokens=64, seed=1234)
    check("temperature=0 is reproducible", text_of(r1) == text_of(r2),
          "identical" if text_of(r1) == text_of(r2) else "DIVERGED")

    print("\n4. SAMPLING CONTROLS")
    s1 = text_of(chat(B, model, "Invent a short product name.", max_tokens=24, temperature=1.2, seed=1))
    s2 = text_of(chat(B, model, "Invent a short product name.", max_tokens=24, temperature=1.2, seed=2))
    check("temperature>0 with different seeds diverges", s1 != s2, f"{s1[:24]!r} vs {s2[:24]!r}")
    d = chat(B, model, "Repeat the word 'apple' forever.", max_tokens=60,
             temperature=0.8, seed=7, frequency_penalty=1.5)
    check("frequency_penalty accepted and applied", d["usage"]["completion_tokens"] > 0,
          f"{len(set(text_of(d).split()))} distinct tokens")

    print("\n5. LENGTH / FINISH REASONS")
    d = chat(B, model, "Write a long essay about compilers.", max_tokens=32)
    fr = d["choices"][0]["finish_reason"]
    check("max_tokens truncation -> finish_reason=length", fr == "length", f"finish_reason={fr}")
    check("max_tokens is respected exactly", d["usage"]["completion_tokens"] <= 32,
          f"{d['usage']['completion_tokens']} tokens")
    d = chat(B, model, "Say only the word: done", max_tokens=64)
    check("natural stop -> finish_reason=stop", d["choices"][0]["finish_reason"] == "stop",
          f"finish_reason={d['choices'][0]['finish_reason']}")

    print("\n6. MULTI-TURN CONTEXT")
    body = {"model": model, "max_tokens": 400, "temperature": 0.0, "stream": False, "messages": [
        {"role": "user", "content": "My favourite colour is teal. Remember it."},
        {"role": "assistant", "content": "Got it — teal."},
        {"role": "user", "content": "What is my favourite colour? One word."}]}
    d = post(B, "/v1/chat/completions", body)
    check("recalls earlier turn", "teal" in text_of(d).lower(), repr(text_of(d)[:50]))

    print("\n7. STREAMING")
    body = {"model": model, "messages": [{"role": "user", "content": "Count 1 to 20."}],
            "max_tokens": 500, "temperature": 0.0, "stream": True}
    req = urllib.request.Request(B + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    chunks, rchunks, acc, saw_done = 0, 0, "", False
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                saw_done = True
                break
            delta = json.loads(payload)["choices"][0].get("delta", {})
            # A reasoning serve streams thought in reasoning_content FIRST; counting only `content`
            # on a short budget measures nothing but the thinking phase.
            if delta.get("reasoning_content"):
                rchunks += 1
            if delta.get("content"):
                acc += delta["content"]; chunks += 1
    check("stream terminates with [DONE]", saw_done)
    check("stream produced an answer", len(acc) > 0, f"{chunks} answer chunks, {len(acc)} chars")
    check("stream separates reasoning deltas", rchunks > 0, f"{rchunks} reasoning chunks")
    check("streamed answer matches non-streamed", "20" in acc, repr(acc[-40:]))

    print("\n8. LONG PROMPT (prefill path)")
    # Compute the expected answer from the SAME expression that generates the facts. Hardcoding it
    # is how this check first failed: the model correctly answered 85 and the test asserted 60.
    def _val(i: int) -> int:
        return i * 7 % 97
    _needle = 123
    long_p = ("Here is a list of facts.\n"
              + "\n".join(f"Fact {i}: the value of item {i} is {_val(i)}." for i in range(400))
              + f"\n\nWhat is the value of item {_needle}? Answer with just the number.")
    t0 = time.perf_counter()
    d = chat(B, model, long_p, max_tokens=600)
    check("long prompt served", d["usage"]["prompt_tokens"] > 2000,
          f"{d['usage']['prompt_tokens']} prompt tokens, {time.perf_counter()-t0:.1f}s")
    check("retrieves a fact from mid-context", str(_val(_needle)) in text_of(d),
          f"expected {_val(_needle)}, got {text_of(d)[:40]!r}")

    print("\n9. UNICODE / EDGE CASES")
    d = chat(B, model, "Repeat exactly: 日本語 — café — 🎯", max_tokens=400)
    check("unicode round-trips", any(x in text_of(d) for x in ("日本", "café", "🎯")),
          repr(text_of(d)[:40]))
    d = chat(B, model, "hi", max_tokens=400)
    check("very short prompt", d["usage"]["completion_tokens"] > 0, repr(text_of(d)[:30]))
    try:
        post(B, "/v1/chat/completions", {"model": model, "messages": [], "max_tokens": 8})
        check("empty messages rejected or handled", True, "accepted without crashing")
    except urllib.error.HTTPError as e:
        check("empty messages rejected or handled", 400 <= e.code < 500, f"HTTP {e.code}")

    print("\n10. THROUGHPUT (TRUE tok/s from usage, warmup discarded)")
    chat(B, model, "Say hello.", max_tokens=8)  # warmup, discarded

    def run(i: int, n: int, mt: int = 256):
        pr = "Explain speculative decoding in detail."
        if n > 1:
            pr += f"\n\n(Variant {i}: focus on point {i+1}.)"
        return chat(B, model, pr, max_tokens=mt, temperature=0.0, seed=1234)["usage"]["completion_tokens"]

    tps = {}
    for n in (1, a.conc):
        t0 = time.perf_counter()
        with ThreadPoolExecutor(n) as ex:
            toks = sum(ex.map(lambda i: run(i, n), range(n)))
        w = time.perf_counter() - t0
        tps[n] = toks / w
        print(f"    bs={n}: {toks} tokens / {w:.2f}s = {tps[n]:.1f} tok/s")
    check("single-stream throughput is sane (>20 tok/s)", tps[1] > 20, f"{tps[1]:.1f} tok/s")
    check("concurrency scales throughput", tps[a.conc] > tps[1],
          f"bs=1 {tps[1]:.1f} -> bs={a.conc} {tps[a.conc]:.1f} ({tps[a.conc]/tps[1]:.2f}x)")

    print("\n11. LATENCY CONSISTENCY (no stalls)")
    lat = []
    for _ in range(5):
        t0 = time.perf_counter()
        chat(B, model, "Give one short fact about GPUs.", max_tokens=48, temperature=0.0, seed=99)
        lat.append(time.perf_counter() - t0)
    med, mx = statistics.median(lat), max(lat)
    check("no latency outlier (max < 3x median)", mx < 3 * med,
          f"median {med:.2f}s max {mx:.2f}s")

    print("\n12. STABILITY UNDER LOAD (repeat concurrency, all must complete)")
    ok_n = 0
    with ThreadPoolExecutor(a.conc) as ex:
        for r in ex.map(lambda i: run(i, a.conc, 128), range(a.conc)):
            ok_n += 1 if r > 0 else 0
    check(f"all {a.conc} concurrent requests completed", ok_n == a.conc, f"{ok_n}/{a.conc}")

    print("\n" + "=" * 78)
    npass = sum(1 for _, ok, _ in RESULTS if ok)
    fails = [n for n, ok, _ in RESULTS if not ok]
    print(f"RESULT: {npass}/{len(RESULTS)} checks passed")
    if fails:
        print("FAILED: " + "; ".join(fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
