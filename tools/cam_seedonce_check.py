#!/usr/bin/env python
"""Phase-1 productionized path: seed-once tap delivery driven by a REQUEST (sampling_params.mem_subject).

Unlike cam_serve_check.py (which staged banks by hand to prove parity), this exercises the real
scheduler wiring: mem_subject rides SamplingParams -> Req; the scheduler reads the bank at prefill
(_prepare_cam), stages it before each forward (_stage_cam), and clears it once the object's first token
lands (seed-once, _process_last_data). Success = the object is delivered AND the tail is fluent (NOT the
repetitive "English English English" of an always-on tap).

Run in the lean image on a lease (scratchpad/run_seedonce.sh):
    python /engine/tools/cam_seedonce_check.py --model Qwen/Qwen3.5-4B
"""
from __future__ import annotations

import argparse
import torch

from minisgl.core import SamplingParams
from minisgl.llm import LLM

FACTS = [("Lionel Jospin", "The mother tongue of Lionel Jospin is", "Dutch"),
         ("Oleg Kotov",    "The mother tongue of Oleg Kotov is",    "English"),
         ("Marie NDiaye",  "The mother tongue of Marie NDiaye is",  "Russian")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--max-tokens", type=int, default=14)
    ap.add_argument("--max-running-req", type=int, default=4)
    ap.add_argument("--memory-ratio", type=float, default=0.85)
    args = ap.parse_args()

    llm = LLM(model_path=args.model, dtype=torch.bfloat16, cuda_graph_max_bs=0, page_size=16,
              memory_ratio=args.memory_ratio, attention_backend="hip",
              max_running_req=args.max_running_req)
    cam = llm.engine.cam
    assert cam is not None and cam.enabled, "CAM not built (MINISGL_CAM=1 + ckpt?)"
    tok = llm.tokenizer
    encsp = lambda s: list(tok(" " + s, add_special_tokens=False).input_ids)
    for s, _, o in FACTS:
        cam._write(encsp(s), encsp(o))
    print(f"[cam] wrote {len(FACTS)} facts; tap_layer={cam.tap_layer}", flush=True)

    n_hit = 0
    for s, p, o in FACTS:
        off_sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
        mem_sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, mem_subject=s)
        off = llm.generate([p], off_sp)[0]["text"].replace("\n", " ").strip()
        on = llm.generate([p], mem_sp)[0]["text"].replace("\n", " ").strip()
        hit = o.lower() in on.lower()
        n_hit += hit
        print(f"\n=== {p!r}  (mem_subject={s!r})", flush=True)
        print(f"    OFF  -> {off!r}", flush=True)
        print(f"    MEM  -> {on!r}   [delivered '{o}': {hit}]", flush=True)

    print(f"\nCAM-SEEDONCE REQUEST-DRIVEN DELIVERY {n_hit}/{len(FACTS)}", flush=True)


if __name__ == "__main__":
    main()
