"""Isolated captured-forward REPLAY vs end-to-end serving A/B (measurement 2's method).

Single-card / single-process via the offline LLM path. Discriminates whether the decode gap is
in-model GPU compute or host (scheduler/sample/detok):
  * pure_replay_us  : just g.replay() of the captured decode graph, in a tight loop (GPU forward only)
  * full_replay_us  : graph_runner.replay(batch) — pure replay + the per-step host copy_from/prepare
  * generate_tpot_us: end-to-end offline generate() per-output-token (adds sampler + scheduler + detok)
If pure_replay << generate_tpot -> the gap is host-side (launch bubbles / sampling / detok / scheduler).
If pure_replay ~= generate_tpot -> the decode is in-model GPU-bound.

NOTE: run on the 35B TP=2 needs the multiprocess serve launcher (offline LLM is single-process); this
harness is the SINGLE-CARD proxy (default Qwen3.5-4B GDN-hybrid, same decode structure & vocab family).
  gpu-lease -n 1 -- bash -c 'docker run ... python /engine/tools/isolated_forward_ab.py [MODEL]'
"""
from __future__ import annotations

import os
import sys
import time

import torch
from minisgl.core import SamplingParams
from minisgl.llm import LLM

MODEL = sys.argv[1] if len(sys.argv) > 1 else "cyankiwi/Qwen3.5-4B-AWQ-BF16-INT4"


def main() -> int:
    gbs = int(os.environ.get("GBS", "8"))
    print(f"[iso-fwd] model={MODEL} graph_bs={gbs} torch={torch.__version__} hip={torch.cuda.is_available()}",
          flush=True)
    llm = LLM(
        model_path=MODEL,
        dtype=torch.bfloat16,
        attention_backend="hip",
        cuda_graph_max_bs=gbs,
        memory_ratio=0.80,
        max_running_req=1,
        gdn_radix=(os.environ.get("GDN_RADIX", "0") == "1"),  # off by default: 2nd generate() + radix segfaults offline
    )
    gr = llm.engine.graph_runner

    # ---- hook: stash a steady-state decode batch, then time pure g.replay() -------------------
    stash = {"batch": None, "n": 0, "measure": False, "full_us": None}
    orig_replay = gr.replay

    def timed_replay(batch):
        stash["n"] += 1
        if stash["measure"] and stash["n"] == 25 and gr.can_use_cuda_graph(batch):  # steady decode step
            # Time the FULL production per-step forward: copy_from + prepare_for_replay + g.replay().
            # (A bare g.replay() loop without prepare_for_replay corrupts the paged-attn metadata and
            #  segfaults, so we measure exactly what the serve pays each decode step.)
            for _ in range(20):
                orig_replay(batch)
            torch.cuda.synchronize()
            tic, toc = torch.cuda.Event(True), torch.cuda.Event(True)
            tic.record()
            for _ in range(300):
                orig_replay(batch)
            toc.record(); toc.synchronize()
            stash["full_us"] = tic.elapsed_time(toc) / 300 * 1e3
        return orig_replay(batch)

    gr.replay = timed_replay  # type: ignore

    prompt = ("Explain, step by step and in depth, how a modern CPU executes a program: cover "
              "fetch/decode/execute, pipelining, branch prediction, caches, and out-of-order execution.")
    # warmup
    _ = llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=8))

    # gen#1: CLEAN timed generate (no timing loop) -> honest serving TPOT (single process)
    stash["measure"] = False
    n_tok = 128
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    outs = llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=n_tok))
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    n = len(outs[0]["token_ids"])
    gen_tpot_us = dt / max(n - 1, 1) * 1e6

    # gen#2: install timing loop -> isolated full-replay us (its own TPOT is discarded)
    stash["measure"] = True
    stash["n"] = 0
    _ = llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=64))

    full = stash["full_us"]
    print("\n==== ISOLATED FORWARD vs SERVING A/B (single-card proxy) ====", flush=True)
    if full:
        print(f"full replay(batch)   : {full:9.1f} us/step   ({1e6/full:7.1f} tok/s)  [captured forward + host copy/prepare]",
              flush=True)
    print(f"generate() TPOT      : {gen_tpot_us:9.1f} us/step   ({1e6/gen_tpot_us:7.1f} tok/s)  "
          f"[+sampler+scheduler+detok]", flush=True)
    if full:
        gap = gen_tpot_us - full
        print(f"\nsched+sample+detok gap: {gap:9.1f} us/step  ({gap/gen_tpot_us*100:.1f}% of serving TPOT)",
              flush=True)
        print(f"in-forward (replay)   : {full:9.1f} us/step  ({full/gen_tpot_us*100:.1f}% of serving TPOT)",
              flush=True)
    print(f"\n[OUTPUT] {outs[0]['text'][:120]!r}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
