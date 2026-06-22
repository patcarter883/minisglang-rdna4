"""Phase 3d-4 — vLLM greedy reference for Qwen3.5-4B, and token-diff vs minisgl's saved output.

Runs the combined image's vLLM on the SAME prompts (greedy, eager) and compares the generated
token ids to /engine/tmp/qwen3_5_ours.json (from boot_smoke). GPU via the lease.
"""
from __future__ import annotations

import argparse
import json

from vllm import LLM, SamplingParams

PROMPTS = [
    "The capital of France is",
    "Once upon a time, in a small village,",
    "The three primary colors are red, blue, and",
    "Q: What is 2 + 2? A:",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--ours", default="/engine/tmp/qwen3_5_ours.json")
    args = ap.parse_args()

    llm = LLM(model=args.model, dtype="bfloat16", enforce_eager=True,
              gpu_memory_utilization=0.85, max_model_len=2048)
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    out = llm.generate(PROMPTS, sp)
    ref = [{"prompt": p, "text": o.outputs[0].text, "token_ids": list(o.outputs[0].token_ids)}
           for p, o in zip(PROMPTS, out)]

    ours = None
    try:
        with open(args.ours) as f:
            ours = {r["prompt"]: r for r in json.load(f)}
    except FileNotFoundError:
        print(f"[warn] {args.ours} not found; printing vLLM ref only")

    for r in ref:
        print(f"\n=== {r['prompt']!r}")
        print(f"  vLLM : {r['text']!r}")
        print(f"  vids : {r['token_ids'][:24]}")
        if ours is not None and r["prompt"] in ours:
            o = ours[r["prompt"]]
            oids, vids = o["token_ids"], r["token_ids"]
            n = min(len(oids), len(vids))
            match = sum(1 for i in range(n) if oids[i] == vids[i])
            first_div = next((i for i in range(n) if oids[i] != vids[i]), None)
            print(f"  ours : {o['text']!r}")
            print(f"  oids : {oids[:24]}")
            print(f"  MATCH {match}/{n} greedy tokens; first divergence @ {first_div}")


if __name__ == "__main__":
    main()
