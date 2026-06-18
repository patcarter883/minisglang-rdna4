#!/usr/bin/env python
"""Logit oracle, part 1: capture the RDNA4 engine's first-token logits (post-lm_head,
last position per prompt) by hooking the sampler, and save them with the input ids.
Run each prompt singly (batch=1) so the saved order is unambiguous."""
from __future__ import annotations

import torch
from transformers import AutoTokenizer

from minisgl.core import SamplingParams
from minisgl.llm import LLM

MODEL = "Qwen/Qwen3-0.6B"
PROMPTS = [
    "The capital of France is",
    "Once upon a time, in a small village,",
    "The three primary colors are red, blue, and",
    "Q: What is 2 + 2? A:",
]


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL)
    input_ids = [tok.encode(p) for p in PROMPTS]
    llm = LLM(MODEL, dtype=torch.bfloat16, cuda_graph_max_bs=0, page_size=16, memory_ratio=0.4)

    cap: dict[str, torch.Tensor] = {}
    orig = llm.engine.sampler.sample

    def hook(logits, args):
        cap["l"] = logits.detach().float().cpu().clone()
        return orig(logits, args)

    llm.engine.sampler.sample = hook  # type: ignore[method-assign]

    sp = SamplingParams(temperature=0.0, max_tokens=1)
    ours = []
    for ids in input_ids:
        cap.clear()
        llm.generate([ids], sp)
        ours.append(cap["l"][0])  # [vocab]
    logits = torch.stack(ours)
    torch.save({"input_ids": input_ids, "logits": logits}, "/engine/tools/ours_logits.pt")
    print(f"[oracle_ours] saved logits {tuple(logits.shape)} for {len(input_ids)} prompts")


if __name__ == "__main__":
    main()
