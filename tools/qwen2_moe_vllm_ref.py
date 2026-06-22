"""Phase 2M-4 — vLLM greedy reference for Qwen1.5-MoE-A2.7B-GPTQ-Int4, token-diff vs minisgl.

vLLM is the loading reference AND parity oracle for this checkpoint (it serves it fine via
GPTQConfig -> MoeWNA16). Runs vLLM on the SAME prompts (greedy, eager, quantization auto-detected
from config) and token-diffs against /engine/tmp/qwen2_moe_ours.json (from boot_smoke). GPU via the
lease. NOT expected to be bit-identical: minisgl runs the int4 weights through the fp8-activation
WMMA kernel (e4m3 compute) while vLLM dequants to bf16 — same posture as Phase 1a/3d-4. Parity =
coherent output + high greedy-token agreement (divergence only deep in decode where tiny logit
deltas flip the argmax).
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
    ap.add_argument("--model", default="Qwen/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4")
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--ours", default="/engine/tmp/qwen2_moe_ours.json")
    args = ap.parse_args()

    # dtype="auto" -> read torch_dtype from config; quantization auto-detected (gptq -> MoeWNA16).
    llm = LLM(model=args.model, dtype="auto", enforce_eager=True,
              gpu_memory_utilization=0.85, max_model_len=2048, trust_remote_code=True)
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

    total_match = total_n = 0
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
            total_match += match
            total_n += n
            print(f"  ours : {o['text']!r}")
            print(f"  oids : {oids[:24]}")
            print(f"  MATCH {match}/{n} greedy tokens; first divergence @ {first_div}")
    if total_n:
        print(f"\n[parity] overall {total_match}/{total_n} greedy tokens "
              f"({100*total_match/total_n:.1f}%)")


if __name__ == "__main__":
    main()
