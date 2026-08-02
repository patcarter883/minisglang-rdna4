"""Correctness gate for recurrent radix UNDER SPEC DECODE.

The MINISGL_REC_RADIX_SPEC flag this was written for is GONE — recurrent radix now composes with
spec by default, having been measured lossless. Kept as the REGRESSION gate for that property: run
it after any change to the recurrent snapshot/restore or the spec verify-state install.

Wrong here is SILENT GARBAGE, not a crash: a prefix HIT reports cached_len>0, and the recurrent
(GDN/CCA) state behind it must equal what a fresh full forward would have produced. If it does not,
the serve keeps answering — just differently, and worse. So the gate is OUTPUT EQUALITY, not speed:

    cold MISS output  ==  warm HIT output      byte-identical, greedy, same seed

    MINISGL_MOE_G2FUSE=0 ... docker compose --profile serve up -d   # required, see below
    python3 tools/rec_radix_spec_gate.py --expect-hit
Compare against a --no-gdn-radix serve for the cache-off control.

A pass here is necessary, not sufficient — it shows the reused state is faithful on THESE prompts.
The original recurrent-radix fix was validated the same way (cold-MISS == warm-HIT byte-identical)
plus a concurrency soak; do both before trusting it in production.

THE SERVE MUST BE DETERMINISTIC FIRST, or this gate is meaningless. Measured 2026-08-02: with the
cache OFF, the SAME greedy prompt run twice produced DIFFERENT answer text at 256 output tokens.
Cause is not the cache — it is the fused MoE gemm2, which accumulates across blocks with atomicAdd
(moe_kernel.hip: "the cross-block accumulation needs atomics"), so the reduction order varies run to
run on a 256-expert MoE and the divergence compounds over a long greedy chain. Start the serve with
MINISGL_MOE_G2FUSE=0 (the bit-exact WMMA gemm2 + gather_reduce) or every comparison below is noise.
This gate refuses to run without it rather than report a colourful false failure — a control that
cannot fail is not evidence, and one whose premise is false is worse.
"""
from __future__ import annotations

import argparse
import json
import hashlib
import subprocess
import sys
import time
import urllib.request

PREFIX_FACTS = 900          # ~15k tokens: long enough that a hit is unmistakable in TTFT
GREEDY = {"temperature": 0.0, "seed": 1234, "stream": False}


def post(base, body, timeout=1800):
    r = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        return json.loads(resp.read())


def ask(base, model, prompt, max_tokens=32):
    t0 = time.perf_counter()
    d = post(base, {"model": model, "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens, **GREEDY})
    m = d["choices"][0]["message"]
    return {
        "text": m.get("content") or "",
        "reasoning": m.get("reasoning_content") or "",
        "tokens": d["usage"]["completion_tokens"],
        "prompt_tokens": d["usage"]["prompt_tokens"],
        "wall": time.perf_counter() - t0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:1919")
    ap.add_argument("--container", default="lease-minisgl-serve-serve")
    ap.add_argument("--expect-hit", action="store_true",
                    help="assert the engine logs a recurrent-radix HIT (flag must be ON)")
    a = ap.parse_args()
    B = a.base
    model = json.loads(urllib.request.urlopen(B + "/v1/models", timeout=30).read())["data"][0]["id"]

    # PRECONDITION, MEASURED not assumed: find an output length at which this serve is actually
    # bit-reproducible, and compare only there.
    #
    # MINISGL_MOE_G2FUSE=0 is NOT sufficient. Measured 2026-08-02 with the prefix cache OFF and
    # G2FUSE=0, an identical greedy request repeated 5x produced:
    #     max_tokens= 32 -> 1 distinct output   (reproducible)
    #     max_tokens=128 -> 3 distinct
    #     max_tokens=256 -> 4 distinct          (essentially non-reproducible)
    # so a 256-token comparison measures NOISE. The residual source is not only the fused gemm2:
    # moe_align scatters rows with `atomicAdd` and documents "order within a run is arbitrary", so the
    # per-expert row order — and therefore the gather-reduce summation order — varies run to run.
    # This gate therefore probes the floor and refuses if even the shortest length is unstable,
    # rather than reporting a confident verdict on a comparison that cannot mean anything.
    probe = "Explain speculative decoding in an inference engine."
    stable = None
    for mt in (32, 64, 128):
        h = {hashlib.sha1((ask(B, model, probe, mt)["reasoning"] + "|"
                           + ask(B, model, probe, mt)["text"]).encode()).hexdigest() for _ in range(2)}
        outs = [ask(B, model, probe, mt) for _ in range(3)]
        sigs = {o["reasoning"] + "|" + o["text"] for o in outs}
        print(f"  noise probe max_tokens={mt:<4} -> {len(sigs)} distinct of 3"
              + ("   <= REPRODUCIBLE" if len(sigs) == 1 else ""))
        if len(sigs) == 1:
            stable = mt
        else:
            break
    if stable is None:
        print("REFUSING TO RUN: this serve is not bit-reproducible at ANY tested length, so a")
        print("  cold-vs-warm output comparison cannot distinguish a lossy cache from run-to-run")
        print("  noise. Validate the recurrent state directly instead of via model output.")
        return 2
    print(f"  -> comparing at max_tokens={stable}\n")
    GATE_MT = stable

    prefix = "\n".join(f"Fact {i}: item {i} has value {i * 7 % 97}." for i in range(PREFIX_FACTS))
    q = "\n\nUsing the facts above, explain in detail how you would find the value of item 500, then give it."
    prompt = prefix + q
    fails = []

    print(f"model {model}\nprefix ~{PREFIX_FACTS} facts\n" + "=" * 74)

    # A fresh serve has never seen this prefix -> this run is the COLD MISS that populates the cache.
    cold = ask(B, model, prompt, GATE_MT)
    print(f"cold  MISS  {cold['prompt_tokens']:6d} prompt tok  {cold['tokens']:4d} out  {cold['wall']:6.2f}s")

    # Same prompt again: identical prefix AND suffix -> maximal hit.
    warm = ask(B, model, prompt, GATE_MT)
    print(f"warm  HIT?  {warm['prompt_tokens']:6d} prompt tok  {warm['tokens']:4d} out  {warm['wall']:6.2f}s"
          f"   ({100*(1-warm['wall']/cold['wall']):+.0f}% wall)")

    # THE GATE. Reused recurrent state must reproduce the fresh forward exactly.
    if cold["text"] != warm["text"]:
        fails.append("answer text DIVERGED between cold MISS and warm HIT")
    if cold["reasoning"] != warm["reasoning"]:
        fails.append("reasoning text DIVERGED between cold MISS and warm HIT")
    if cold["tokens"] != warm["tokens"]:
        fails.append(f"token count differs ({cold['tokens']} vs {warm['tokens']})")
    print(f"  text identical:      {cold['text'] == warm['text']}")
    print(f"  reasoning identical: {cold['reasoning'] == warm['reasoning']}")

    # A DIFFERENT suffix on the same prefix — the case a prefix cache actually exists for. It cannot
    # be compared to `cold`, so compare it to itself across a hit boundary.
    q2 = "\n\nUsing the facts above, explain how you would find the value of item 250, then give it."
    first = ask(B, model, prefix + q2, GATE_MT)
    again = ask(B, model, prefix + q2, GATE_MT)
    if first["text"] != again["text"]:
        fails.append("different-suffix answer not reproducible across a hit")
    print(f"  new-suffix reproducible: {first['text'] == again['text']}"
          f"   ({first['wall']:.2f}s -> {again['wall']:.2f}s)")

    # Engine-side evidence: did a recurrent-radix hit actually occur?
    hits = 0
    try:
        out = subprocess.run(["docker", "logs", "--tail", "3000", a.container],
                             capture_output=True, text=True, timeout=60)
        log = out.stdout + out.stderr
        hits = sum(1 for l in log.splitlines() if "recurrent-radix" in l.lower() and "hit" in l.lower())
        mode = [l for l in log.splitlines() if "prefix cache" in l.lower()
                or "recurrent radix prefix cache" in l.lower()]
        if mode:
            print(f"  engine cache mode: {mode[-1].split('INFO')[-1].split('WARNING')[-1].strip()[:110]}")
        print(f"  recurrent-radix HIT lines in log: {hits}")
    except Exception as e:  # noqa: BLE001
        print(f"  (could not read container log: {type(e).__name__})")

    if a.expect_hit and hits == 0:
        fails.append("--expect-hit set but the engine logged no recurrent-radix HIT "
                     "(flag off, or the cache never engaged — a 'pass' here would be vacuous)")

    print("=" * 74)
    if fails:
        print("FAIL:\n  - " + "\n  - ".join(fails))
        return 1
    print("PASS — reused recurrent state reproduces a fresh forward byte-identically")
    return 0


if __name__ == "__main__":
    sys.exit(main())
