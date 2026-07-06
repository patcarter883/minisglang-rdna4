#!/usr/bin/env python
"""End-to-end smoke for the perf-backlog branch on the live engine path.

Exercises (on a real GDN model, bf16, single card):
  * [B2] high --max-running-req boots WITHOUT OOM (the recurrent-state reservation) — watch the log
    for "Reserved ... for GDN/CCA recurrent state".
  * [A3] vectorized page-table, [A2] output-buffer reuse, [S5] embedding mask, [A4] fused bf16
    store_kv — all active on the greedy decode path; validated by COHERENT output.
  * [S4] fused sampler_hip — a sampled (temperature>0) generation routes through sample_impl ->
    the fused kernel (greedy uses argmax and bypasses it).

Run under a 1-card lease inside the lean image (see the invoking shell command).
"""
from __future__ import annotations

import argparse

from minisgl.core import SamplingParams
from minisgl.llm import LLM

PROMPTS = [
    "The capital of France is",
    "The three primary colors are red, blue, and",
    "Q: What is 2 + 2? A:",
    "List the first five prime numbers:",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--max-running-req", type=int, default=256)  # high -> stresses [B2]
    ap.add_argument("--memory-ratio", type=float, default=0.85)
    args = ap.parse_args()

    print(f"[smoke] loading {args.model} | max_running_req={args.max_running_req} (B2 stress)")
    llm = LLM(
        args.model,
        attention_backend="hip",  # production native-HIP path
        max_running_req=args.max_running_req,
        memory_ratio=args.memory_ratio,
    )
    print("[smoke] BOOT OK (no OOM at high max_running_req -> [B2] reservation works)")

    print("\n[smoke] === greedy (A3/A2/S5/store_kv-bf16 live) ===")
    greedy = llm.generate(PROMPTS, SamplingParams(temperature=0.0, max_tokens=args.max_tokens))
    for p, o in zip(PROMPTS, greedy):
        print(f"  {p!r} -> {o['text']!r}")

    print("\n[smoke] === sampled temp=0.8 top_p=0.95 (fused sampler [S4] live) ===")
    sampled = llm.generate(
        PROMPTS, SamplingParams(temperature=0.8, top_p=0.95, top_k=40, max_tokens=args.max_tokens)
    )
    for p, o in zip(PROMPTS, sampled):
        print(f"  {p!r} -> {o['text']!r}")

    # crude coherence gate: the France prompt must mention Paris on the greedy (deterministic) path.
    france = greedy[0]["text"].lower()
    ok = "paris" in france
    print(f"\n[smoke] coherence gate (France->Paris): {'PASS' if ok else 'FAIL'} ({france!r})")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
