#!/usr/bin/env python
"""Phase-2 validation: the CAM decode tap runs INSIDE the captured CUDA graph.

Boots with cuda-graph capture ON (cuda_graph_max_bs>0) so decode replays a captured graph. Two checks
per fact:
  * ALWAYS-INJECT (MINISGL_CAM_ALWAYS_INJECT=1): injection persists into decode, so the object token
    REPEATS in the tail — this is only possible if the captured decode graph actually injects the tap
    (with seed-once the object lands at PREFILL and decode would be a no-op, proving nothing). Repetition
    => the static per-row bank buffer + tap ops were captured and replay correctly.
  * SEED-ONCE (default): clean fluent delivery with graphs on (no breakage from the buffer path).

Run in the lean image on a lease (scratchpad/run_graph.sh):
    python /engine/tools/cam_graph_check.py --model Qwen/Qwen3.5-4B --graph 2
"""
from __future__ import annotations

import argparse
import os
import torch

from minisgl.core import SamplingParams
from minisgl.llm import LLM

FACTS = [("Lionel Jospin", "The mother tongue of Lionel Jospin is", "Dutch"),
         ("Oleg Kotov",    "The mother tongue of Oleg Kotov is",    "English"),
         ("Marie NDiaye",  "The mother tongue of Marie NDiaye is",  "Russian")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--graph", type=int, default=2, help="cuda_graph_max_bs (>0 enables capture)")
    ap.add_argument("--max-tokens", type=int, default=14)
    ap.add_argument("--max-running-req", type=int, default=2)
    ap.add_argument("--memory-ratio", type=float, default=0.8)
    args = ap.parse_args()

    llm = LLM(model_path=args.model, dtype=torch.bfloat16, cuda_graph_max_bs=args.graph, page_size=16,
              memory_ratio=args.memory_ratio, attention_backend="hip",
              max_running_req=args.max_running_req)
    cam = llm.engine.cam
    assert cam is not None and cam.enabled, "CAM not built (MINISGL_CAM=1 + ckpt?)"
    captured = llm.engine.graph_runner.cam_capture is not None
    print(f"[graph] cuda_graph_max_bs={args.graph} cam_capture={'BUILT' if captured else 'NONE'} "
          f"tap_layer={cam.tap_layer}", flush=True)
    assert captured, "CAMGraphCapture was not built — no graph tap to validate"
    tok = llm.tokenizer
    encsp = lambda s: list(tok(" " + s, add_special_tokens=False).input_ids)
    for s, _, o in FACTS:
        cam._write(encsp(s), encsp(o))

    def gen(prompt, subj, always):
        os.environ["MINISGL_CAM_ALWAYS_INJECT"] = "1" if always else "0"
        sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, mem_subject=subj)
        return llm.generate([prompt], sp)[0]["text"].replace("\n", " ").strip()

    n_seed = n_decode_inject = 0
    for s, p, o in FACTS:
        seed = gen(p, s, always=False)
        alw = gen(p, s, always=True)
        seed_hit = o.lower() in seed.lower()
        # decode-injection proof: the object repeats (>=2 occurrences) only if the graph decode injects
        reps = alw.lower().count(o.lower())
        decode_inject = reps >= 2
        n_seed += seed_hit
        n_decode_inject += decode_inject
        print(f"\n=== {p!r}", flush=True)
        print(f"    SEED  -> {seed!r}   [delivered '{o}': {seed_hit}]", flush=True)
        print(f"    ALWAYS-> {alw!r}   [obj x{reps} -> graph-decode injects: {decode_inject}]", flush=True)

    print(f"\nCAM-GRAPH seed-once={n_seed}/{len(FACTS)} decode-inject={n_decode_inject}/{len(FACTS)}",
          flush=True)


if __name__ == "__main__":
    main()
