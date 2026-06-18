#!/usr/bin/env python
"""Reference: run vLLM greedy on the SAME input token-ids as the RDNA4 engine and
diff the output token streams. Bit-divergence early => a shim/kernel bug; long
agreement => Phase 1a is numerically sound. (Our engine uses torch RMSNorm/RoPE/SwiGLU
shims while vLLM uses its own ops, so late divergence is acceptable.)"""
from __future__ import annotations

import json

from vllm import LLM, SamplingParams


def main() -> None:
    with open("/engine/tools/cmp_ours.json") as f:
        ours = json.load(f)
    input_ids = [r["input_ids"] for r in ours]

    llm = LLM(
        model="Qwen/Qwen3-0.6B",
        dtype="bfloat16",
        enforce_eager=True,
        gpu_memory_utilization=0.5,
        max_model_len=2048,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=48)
    try:
        from vllm import TokensPrompt

        prompts = [TokensPrompt(prompt_token_ids=ids) for ids in input_ids]
        outs = llm.generate(prompts, sp)
    except Exception:
        outs = llm.generate(prompt_token_ids=input_ids, sampling_params=sp)
    vllm_ids = [list(o.outputs[0].token_ids) for o in outs]
    with open("/engine/tools/cmp_vllm.json", "w") as f:
        json.dump(vllm_ids, f)

    print("\n=== token-diff: RDNA4 engine vs vLLM (same input ids, greedy) ===", flush=True)
    all_identical = True
    for i, (r, v) in enumerate(zip(ours, vllm_ids)):
        o = r["output_ids"]
        common = min(len(o), len(v))
        m = 0
        for a, b in zip(o, v):
            if a == b:
                m += 1
            else:
                break
        identical = m == len(o) == len(v)
        all_identical &= identical
        status = "IDENTICAL" if identical else f"match {m}/{common} then diverge"
        print(f"p{i}: {status}  (ours_len={len(o)} vllm_len={len(v)})", flush=True)
        if not identical:
            print(f"    ours[{m}:{m+6}] = {o[m:m+6]}", flush=True)
            print(f"    vllm[{m}:{m+6}] = {v[m:m+6]}", flush=True)
    print("\nRESULT:", "ALL TOKEN-IDENTICAL" if all_identical else "diverges (see above)", flush=True)


if __name__ == "__main__":
    main()
