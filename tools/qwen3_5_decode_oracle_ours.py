#!/usr/bin/env python
"""Phase 3e — minisgl side of the GDN-hybrid decode-path logit oracle.

Runs the Qwen3.5-4B GDN-hybrid in minisgl, greedy, for N steps on a single prompt, and captures
the *per-step* first-token-style logits by hooking the sampler — sample() fires once per generated
token (prefill -> step 0, then each forward_decode -> steps 1..N-1), so accumulating across ONE
generate yields a [N, vocab] stack where:

  * step 0  == the prefill first-token logit (the dense path's logit-oracle target, task 3e-1), and
  * steps 1..N-1 == the GDN recurrent decode path (causal_conv1d_update +
    fused_sigmoid_gating_delta_rule_update, in-place ssm_state) — the real 3e-2 target.

Saved with the input ids and the greedy gen ids so the HF cmp side can teacher-force on minisgl's
OWN generated sequence (each step then conditioned on an identical prefix, apples-to-apples even if
greedy later diverges). Run in the combined image, batch=1, via the GPU lease.
"""
from __future__ import annotations

import argparse

import torch
from transformers import AutoTokenizer

from minisgl.core import SamplingParams
from minisgl.llm import LLM


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--max-running-req", type=int, default=16)  # GDN slot OOM guard (16 GB card)
    ap.add_argument("--memory-ratio", type=float, default=0.6)
    ap.add_argument("--out", default="/engine/tools/qwen3_5_decode_ours.pt")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    ids = tok.encode(args.prompt)
    print(f"[ours] prompt={args.prompt!r}")
    print(f"[ours] input_ids={ids}")

    llm = LLM(
        model_path=args.model,
        dtype=torch.bfloat16,
        cuda_graph_max_bs=0,       # GDN models force eager anyway
        page_size=16,
        memory_ratio=args.memory_ratio,
        attention_backend="auto",  # -> rdna4 on ROCm
        max_running_req=args.max_running_req,
    )

    cap: list[torch.Tensor] = []
    orig = llm.engine.sampler.sample

    def hook(logits, sargs):
        cap.append(logits.detach().float().cpu()[0].clone())  # [vocab] for this step
        return orig(logits, sargs)

    llm.engine.sampler.sample = hook  # type: ignore[method-assign]

    sp = SamplingParams(temperature=0.0, max_tokens=args.steps)  # greedy
    out = llm.generate([ids], sp)
    gen_ids = list(out[0]["token_ids"])
    logits = torch.stack(cap)  # [num_captured, vocab]

    print(f"[ours] gen_text={out[0]['text']!r}")
    print(f"[ours] gen_ids={gen_ids}")
    print(f"[ours] captured {logits.shape[0]} step logits (vocab={logits.shape[1]})")
    # captured count should equal generated count; align defensively
    n = min(logits.shape[0], len(gen_ids))
    torch.save(
        {"model": args.model, "prompt": args.prompt, "input_ids": ids,
         "gen_ids": gen_ids[:n], "logits": logits[:n]},
        args.out,
    )
    print(f"[ours] saved {args.out}")


if __name__ == "__main__":
    main()
