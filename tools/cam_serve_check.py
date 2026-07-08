#!/usr/bin/env python
"""Phase-1 PARITY probe: does the CAM residual tap deliver through minisgl's OWN served forward?

tap_check.py validated the tap against the HF base. This is the first test through minisgl's native
kernels (the real serving path). We boot the engine (CAM built in Engine.__init__, model-share), write
a fact directly into the store (bypassing the base-logits write gate — not what's under test here),
stage the subject's bank on the served model, and greedy-generate via the normal scheduler batch loop.
The staged bank persists across all decode steps (always-on tap → repetition expected; seed-once
quality is already proven in tap_check). If the object is delivered, the HF-trained tap composes with
minisgl's forward and Phase 1 is de-risked; the rest is ZMQ/request plumbing.

Run in the lean image on a lease (see scratchpad/run_camserve.sh):
    python /engine/tools/cam_serve_check.py --model Qwen/Qwen3.5-4B
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
    eng = llm.engine
    cam = eng.cam
    assert cam is not None and cam.enabled, "CAM not built on the engine (MINISGL_CAM=1 + ckpt?)"
    inner = eng.model.model
    tok = llm.tokenizer
    encsp = lambda s: list(tok(" " + s, add_special_tokens=False).input_ids)
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)  # greedy

    # write all facts into the standing store (direct _write; gate is not under test here)
    for s, _, o in FACTS:
        cam._write(encsp(s), encsp(o))
    print(f"[cam] wrote {len(FACTS)} facts; tap_layer={cam.tap_layer} n_banks={cam.n_banks}", flush=True)

    n_hit = 0
    for s, p, o in FACTS:
        # OFF baseline
        inner.clear_cam()
        off = llm.generate([p], sp)[0]["text"].replace("\n", " ").strip()
        # TAP: stage this subject's bank so the L24 tap injects every forward
        bank, conf = cam.read(encsp(s))
        inner.stage_cam(cam, bank, conf)
        on = llm.generate([p], sp)[0]["text"].replace("\n", " ").strip()
        inner.clear_cam()
        hit = o.lower() in on.lower()
        n_hit += hit
        print(f"\n=== {p!r}", flush=True)
        print(f"    OFF -> {off!r}", flush=True)
        print(f"    TAP -> {on!r}   [delivered '{o}': {hit}]", flush=True)

    print(f"\nCAM-SERVE-FORWARD TAP DELIVERY {n_hit}/{len(FACTS)}", flush=True)


if __name__ == "__main__":
    main()
