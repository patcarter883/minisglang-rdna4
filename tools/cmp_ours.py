#!/usr/bin/env python
"""Generate greedy continuations with the RDNA4 engine on FIXED input token-ids
(tokenized once, shared with the vLLM reference) and save {input_ids, output_ids}."""
from __future__ import annotations

import json

import torch
from transformers import AutoTokenizer

from minisgl.core import SamplingParams
from minisgl.llm import LLM

MODEL = "Qwen/Qwen3-0.6B"
MAXTOK = 48
PROMPTS = [
    "The capital of France is",
    "Once upon a time, in a small village,",
    "The three primary colors are red, blue, and",
    "Q: What is 2 + 2? A:",
]


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL)
    input_ids = [tok.encode(p) for p in PROMPTS]
    llm = LLM(MODEL, dtype=torch.bfloat16, cuda_graph_max_bs=0, page_size=16, memory_ratio=0.5)
    sp = SamplingParams(temperature=0.0, max_tokens=MAXTOK)
    out = llm.generate(input_ids, sp)  # generate accepts List[List[int]]
    res = [
        {"input_ids": list(iid), "output_ids": list(o["token_ids"])}
        for iid, o in zip(input_ids, out)
    ]
    with open("/engine/tools/cmp_ours.json", "w") as f:
        json.dump(res, f)
    for i, r in enumerate(res):
        print(f"[ours] p{i} in={len(r['input_ids'])} out[:16]={r['output_ids'][:16]}", flush=True)


if __name__ == "__main__":
    main()
