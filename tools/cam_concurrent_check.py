#!/usr/bin/env python
"""Phase-3 validation: concurrent memory + non-memory requests in ONE batch, each correct.

Submits several prompts together in a single generate() so they prefill/decode as one batch. Each
carries its OWN mem_subject (or none) — the per-row bank buffer must give each row its own tap bank
(the graph decode path fills bank_buf[i] from req i). Success = every memory request delivers ITS
object and the plain request is unperturbed, all from the same in-flight batch.

Run in the lean image on a lease (scratchpad/run_concurrent.sh):
    python /engine/tools/cam_concurrent_check.py --model Qwen/Qwen3.5-4B --graph 4
"""
from __future__ import annotations

import argparse
import torch

from minisgl.core import SamplingParams
from minisgl.llm import LLM

# (subject or None, prompt, expected object or None)
REQS = [
    ("Lionel Jospin", "The mother tongue of Lionel Jospin is", "Dutch"),
    ("Oleg Kotov",    "The mother tongue of Oleg Kotov is",    "English"),
    ("Marie NDiaye",  "The mother tongue of Marie NDiaye is",  "Russian"),
    (None,            "The capital of France is",              None),      # plain, must be unperturbed
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--graph", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=12)
    ap.add_argument("--max-running-req", type=int, default=8)
    ap.add_argument("--memory-ratio", type=float, default=0.8)
    args = ap.parse_args()

    llm = LLM(model_path=args.model, dtype=torch.bfloat16, cuda_graph_max_bs=args.graph, page_size=16,
              memory_ratio=args.memory_ratio, attention_backend="hip",
              max_running_req=args.max_running_req)
    cam = llm.engine.cam
    assert cam is not None and cam.enabled
    tok = llm.tokenizer
    encsp = lambda s: list(tok(" " + s, add_special_tokens=False).input_ids)
    for s, _, o in REQS:
        if s is not None:
            cam._write(encsp(s), encsp(o))

    # plain baseline for the France prompt (no memory anywhere) to compare the concurrent plain row
    plain_sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    france_solo = llm.generate([REQS[3][1]], plain_sp)[0]["text"].replace("\n", " ").strip()

    prompts = [p for _, p, _ in REQS]
    sps = [SamplingParams(temperature=0.0, max_tokens=args.max_tokens,
                          mem_subject=s) for s, _, _ in REQS]
    outs = llm.generate(prompts, sps)   # ALL in one in-flight batch

    n_hit = 0
    n_mem = sum(1 for s, _, _ in REQS if s is not None)
    for (s, p, o), out in zip(REQS, outs):
        txt = out["text"].replace("\n", " ").strip()
        if o is not None:
            hit = o.lower() in txt.lower()
            n_hit += hit
            print(f"\n=== mem  {p!r} (subj={s!r})\n    -> {txt!r}   [delivered '{o}': {hit}]", flush=True)
        else:
            same = txt == france_solo
            print(f"\n=== plain {p!r}\n    -> {txt!r}\n    solo -> {france_solo!r}   "
                  f"[unperturbed: {same}]", flush=True)

    print(f"\nCAM-CONCURRENT memory-delivery {n_hit}/{n_mem}", flush=True)


if __name__ == "__main__":
    main()
