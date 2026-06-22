#!/usr/bin/env python
"""Phase 1a functional boot smoke test.

Loads a small dense model on RDNA4 with the lifted ``triton_rdna4`` attention backend
(eager, bf16 KV) and greedy-generates. Exercises the whole stripped path end-to-end:
embedding -> RMSNorm -> RoPE -> tuned Triton attention -> MLP -> sampling -> paged KV +
radix scheduler. Prints token ids for a token-diff against the combined image / HF.

Run inside the combined image (deps pip-installed), TP=1, GPUs assigned. __main__-guarded
(TP=1 spawns no workers, but keep the guard per repo footgun).
"""
from __future__ import annotations

import argparse
import json

import torch

from minisgl.core import SamplingParams
from minisgl.llm import LLM

PROMPTS = [
    "The capital of France is",
    "Once upon a time, in a small village,",
    "The three primary colors are red, blue, and",
    "Q: What is 2 + 2? A:",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--page-size", type=int, default=16)
    ap.add_argument("--memory-ratio", type=float, default=0.6)
    # GDN-hybrid models size a fixed recurrent-state slot per running req (max_running_req+2
    # slots, conv+ssm per GDN layer); the default 256 slots is GiB-scale and OOMs a 16 GB card
    # for a smoke. Lower it for GDN; dense models ignore the cost. None -> engine default (256).
    ap.add_argument("--max-running-req", type=int, default=None)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--prompt", action="append", default=None, help="override PROMPTS (repeatable)")
    args = ap.parse_args()
    prompts = args.prompt if args.prompt else PROMPTS

    print(f"[boot] loading {args.model} (bf16, eager, attention=auto) ...", flush=True)
    extra = {} if args.max_running_req is None else {"max_running_req": args.max_running_req}
    llm = LLM(
        model_path=args.model,
        dtype=torch.bfloat16,
        cuda_graph_max_bs=0,  # eager: triton_rdna4 has no graph capture yet
        page_size=args.page_size,  # triton_rdna4 requires a multiple of 16
        memory_ratio=args.memory_ratio,
        attention_backend="auto",  # -> triton_rdna4 on ROCm
        **extra,
    )
    try:
        print(
            f"[boot] backend={llm.config.attention_backend} page_size={llm.config.page_size}",
            flush=True,
        )
    except Exception:
        pass

    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)  # greedy
    out = llm.generate(prompts, sp)

    results = []
    for p, o in zip(prompts, out):
        print(f"\n=== {p!r}\n--> {o['text']!r}", flush=True)
        print(f"    ids[:24]: {list(o['token_ids'])[:24]}", flush=True)
        results.append({"prompt": p, "text": o["text"], "token_ids": list(o["token_ids"])})

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[boot] wrote {args.json_out}", flush=True)
    print("\n[boot] DONE", flush=True)


if __name__ == "__main__":
    main()
