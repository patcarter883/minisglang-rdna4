#!/usr/bin/env python
"""gdn_hip #22 — real-serve coherence check for the WMMA chunked prefill (single card, TP=1).

Loads the GDN 4B once and greedy-generates each prompt TWICE: first on the recurrent prefill
(GDN_HIP_WMMA_PREFILL=0, the proven oracle path), then on the matrix-core WMMA prefill (default).
The forward_prefill op is chosen per-call from os.environ, so flipping it between generate() calls
exercises both kernels on the SAME loaded weights. Greedy => deterministic; we diff the token-id
sequences. The op-level parity (tools/gdn_hip_parity.py) already proved max|Δ|~1e-3 vs recurrent;
this confirms the fp16 prefill doesn't perturb real-model token sampling enough to diverge.

Run inside the combined image UNDER a 1-card lease:
    .../gpu-lease.sh -n 1 -- bash -c 'docker run ... python /engine/tools/gdn_wmma_serve_smoke.py'
"""
from __future__ import annotations

import argparse
import os

import torch

from minisgl.core import SamplingParams
from minisgl.llm import LLM

PROMPTS = [
    "The capital of France is",
    "Once upon a time, in a small village,",
    "The three primary colors are red, blue, and",
    "Q: What is 2 + 2? A:",
    "In a single sentence, explain why the sky appears blue:",
    "List the first five prime numbers:",
]


def _gen(llm: LLM, prompts, max_tokens: int):
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)  # greedy
    out = llm.generate(prompts, sp)
    return [{"text": o["text"], "ids": list(o["token_ids"])} for o in out]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--max-running-req", type=int, default=16)
    ap.add_argument("--memory-ratio", type=float, default=0.6)
    args = ap.parse_args()

    print(f"[smoke] loading {args.model} (GDN 4B, TP=1) ...", flush=True)
    llm = LLM(
        model_path=args.model,
        dtype=torch.bfloat16,
        cuda_graph_max_bs=0,
        page_size=16,
        memory_ratio=args.memory_ratio,
        attention_backend="auto",
        max_running_req=args.max_running_req,
    )

    os.environ["GDN_HIP_WMMA_PREFILL"] = "0"
    print("\n[smoke] === recurrent prefill (oracle) ===", flush=True)
    rec = _gen(llm, PROMPTS, args.max_tokens)

    os.environ["GDN_HIP_WMMA_PREFILL"] = "1"
    print("[smoke] === WMMA prefill (matrix-core) ===", flush=True)
    wmma = _gen(llm, PROMPTS, args.max_tokens)

    n_match = 0
    for i, (p, r, w) in enumerate(zip(PROMPTS, rec, wmma)):
        same = r["ids"] == w["ids"]
        # longest common prefix of token ids (how far they agree before any divergence)
        lcp = 0
        for a, b in zip(r["ids"], w["ids"]):
            if a != b:
                break
            lcp += 1
        n_match += same
        verdict = "IDENTICAL" if same else f"diverge@{lcp}/{len(r['ids'])}"
        print(f"\n=== prompt {i}: {p!r}  {verdict}")
        print(f"  recurrent: {r['text']!r}")
        print(f"  wmma     : {w['text']!r}")

    print("\n" + "=" * 60)
    print(f"RESULT: {n_match}/{len(PROMPTS)} token-identical to the recurrent oracle.")
    print("(Identical => the WMMA prefill is serve-safe as the default. A late divergence on a few "
          "prompts is the expected fp16 effect; full text incoherence would be a real bug.)")


if __name__ == "__main__":
    main()
