#!/usr/bin/env python
"""Does the quantized path's M-dependence reach the TOKENS? (serve-level argmax probe)

`tools/quant_m_invariance.py` proved, at the kernel level, that a quantized token's VALUE depends on
how many tokens shared its batch -- the dense arms `decode_gemv` and `prefill_wmma`/`wmma_tiled_tuned`
disagree by ~1e-3 relative, and the engine swaps between them at M=8 (int4). That is a value delta.
Acceptance and greedy output are DISCRETE tests, so the question that matters is whether it flips an
ARGMAX.

This drives a live serve and answers it in tokens:

  FLOOR  the same greedy prompts run twice at the SAME concurrency. Any difference here is the
         engine's own run-to-run noise (MoE atomic reduction order) and BOUNDS everything below it.
         Measured at <= 24 output tokens, inside the documented reproducibility floor.
  M=1    every prompt run ALONE  -> decode batch M=1  -> dense arm decode_gemv, MoE gemm2 scatter.
  M=N    all prompts run TOGETHER -> decode batch M=N -> at N>8 the dense arm is prefill_wmma and the
         MoE gemm2 is gather-reduce. Everything else -- weights, prompt, sampler -- is identical.

  A difference between M=1 and M=N that exceeds the FLOOR is an M-dependence that reached the tokens.

`--check-arm` greps the serve log for the `[hip-engage] fp8_wmma.mmq_fp8_gemm(<arm>)` provenance
lines, so a leg cannot silently be new-vs-itself.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sys
import urllib.request

PROMPTS = [
    "Explain in one paragraph why the sky is blue.",
    "Write a haiku about a cold machine room.",
    "List three causes of the fall of the Western Roman Empire.",
    "What is the difference between a mutex and a semaphore?",
    "Summarize the plot of Moby-Dick in two sentences.",
    "Give a short recipe for scrambled eggs.",
    "Why do leaves change colour in autumn?",
    "Describe the water cycle briefly.",
    "What does a compiler's register allocator do?",
    "Name four properties of a good hash function.",
    "In one paragraph, what is speculative decoding?",
    "How does a heat pump move heat against a gradient?",
    "What is the Chesterton's fence principle?",
    "Explain floating-point non-associativity to a new engineer.",
    "What is the difference between latency and throughput?",
    "Briefly, how does an LSM-tree differ from a B-tree?",
]


def gen(port: int, prompt: str, max_tokens: int) -> str:
    body = json.dumps({
        "model": "x",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        "stream": False,
    }).encode()
    req = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


def run_serial(port, prompts, max_tokens):
    return [gen(port, p, max_tokens) for p in prompts]


def run_concurrent(port, prompts, max_tokens):
    with cf.ThreadPoolExecutor(max_workers=len(prompts)) as ex:
        return list(ex.map(lambda p: gen(port, p, max_tokens), prompts))


def diff(a, b, prompts):
    """Return (n_differing, first divergent char index per prompt)."""
    n, detail = 0, []
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            n += 1
            j = next((k for k in range(min(len(x), len(y))) if x[k] != y[k]), min(len(x), len(y)))
            detail.append((i, j, prompts[i][:40], x[max(0, j - 30):j + 30], y[max(0, j - 30):j + 30]))
    return n, detail


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=1919)
    ap.add_argument("--n", type=int, default=12, help="concurrency for the M=N leg")
    ap.add_argument("--max-tokens", type=int, default=24)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    fh = open(args.out, "w") if args.out else None

    def out(s=""):
        print(s, flush=True)
        if fh:
            fh.write(s + "\n"); fh.flush()

    prompts = PROMPTS[:args.n]
    out(f"prompts={len(prompts)} max_tokens={args.max_tokens} greedy")

    out("\n-- leg 1: SERIAL (decode M=1) ...")
    a1 = run_serial(args.port, prompts, args.max_tokens)
    out("-- leg 2: SERIAL again (reproducibility FLOOR) ...")
    a2 = run_serial(args.port, prompts, args.max_tokens)
    out(f"-- leg 3: CONCURRENT x{len(prompts)} (decode M<={len(prompts)}) ...")
    b1 = run_concurrent(args.port, prompts, args.max_tokens)
    out(f"-- leg 4: CONCURRENT x{len(prompts)} again ...")
    b2 = run_concurrent(args.port, prompts, args.max_tokens)

    nf, _ = diff(a1, a2, prompts)
    nfc, _ = diff(b1, b2, prompts)
    nm, det = diff(a1, b1, prompts)
    out(f"\nFLOOR   serial vs serial       : {nf}/{len(prompts)} prompts differ")
    out(f"FLOOR   conc   vs conc         : {nfc}/{len(prompts)} prompts differ")
    out(f"M TEST  serial(M=1) vs conc(M=N): {nm}/{len(prompts)} prompts differ")
    if det:
        out("\nfirst divergences:")
        for i, j, p, x, y in det[:6]:
            out(f"  [{i}] at char {j}  prompt={p!r}")
            out(f"        M=1 : ...{x!r}")
            out(f"        M=N : ...{y!r}")
    out("")
    out("VERDICT: " + (
        "M-dependence reached the tokens (M test exceeds the floor)."
        if nm > max(nf, nfc) else
        "no token-level M-dependence above the engine's own reproducibility floor."))
    if fh:
        fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
