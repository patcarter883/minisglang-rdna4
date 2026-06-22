#!/usr/bin/env python
"""Phase 3e — HF side of the GDN-hybrid decode-path logit oracle.

Loads qwen3_5_decode_ours.pt (from qwen3_5_decode_oracle_ours.py) and teacher-forces HF over
minisgl's OWN generated sequence in ONE forward (use_cache=False), then slices the per-step logits
and compares to minisgl's captured per-step logits.

Why teacher-force on minisgl's gen_ids: at step t minisgl produced gen_ids[t] from the context
[prompt + gen_ids[:t]]. Feeding HF the full [prompt + gen_ids[:-1]] (use_cache=False) gives, at
position (len(prompt)-1 + t), HF's logit conditioned on EXACTLY that same prefix — so each step is
apples-to-apples regardless of any later greedy divergence. HF's single full-sequence forward takes
the CHUNK gated-delta-rule path; minisgl's steps 1.. take the RECURRENT decode kernels — a per-step
match validates that the decode recurrence reproduces the chunk math step by step (drift growing
with t would betray a state-update bug the coherence check masks).

Step 0 is the prefill first-token oracle (task 3e-1); steps 1.. are the decode path (3e-2).
Run in the combined image, via the GPU lease. Separate process from `ours` (two 4B copies OOM 16 GB).
"""
from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="/engine/tools/qwen3_5_decode_ours.pt")
    ap.add_argument("--cos-prefill", type=float, default=0.999)  # task 3e-1 bar
    ap.add_argument("--cos-decode", type=float, default=0.99)    # per-step decode bar
    args = ap.parse_args()

    d = torch.load(args.inp)
    model = d["model"]
    input_ids = d["input_ids"]            # list[int], length P
    gen_ids = d["gen_ids"]                # list[int], length N
    ours = d["logits"]                    # [N, vocab]
    P, N = len(input_ids), len(gen_ids)
    print(f"[cmp] model={model} prompt={d['prompt']!r}")
    print(f"[cmp] P(prompt)={P} N(steps)={N}")

    hf = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16, device_map="cuda").eval()

    # Feed [prompt + gen_ids[:-1]]: positions P-1+t (t=0..N-1) are the per-step reference logits.
    seq = input_ids + gen_ids[:-1]
    t_in = torch.tensor([seq], device="cuda")
    with torch.no_grad():
        hf_logits = hf(t_in, use_cache=False).logits[0].float().cpu()  # [len(seq), vocab]

    print("=== GDN-hybrid decode-path logit oracle: minisgl vs HF (teacher-forced, bf16) ===")
    print("  step  cos       top1   top5   maxabs   |ours_argmax==gen?|")
    # A top-1 flip only signals a REAL divergence when the logit vectors actually differ; at
    # cos >= TIE_COS the two leading logits are tied to within bf16 noise and the argmax flip is a
    # degenerate tie, not a numerical defect (the exact brittleness the logit oracle exists to see
    # past — cf. oracle_cmp.py: cos-sim is tolerant of sub-ULP rounding, greedy token-identity is not).
    TIE_COS = 0.9995
    worst_decode_cos = 1.0
    prefill_cos = None
    real_top1_break = False
    ties = 0
    for t in range(N):
        o = ours[t]
        h = hf_logits[P - 1 + t]
        cos = F.cosine_similarity(o.unsqueeze(0), h.unsqueeze(0)).item()
        top1 = o.argmax().item() == h.argmax().item()
        ov = len(set(o.topk(5).indices.tolist()) & set(h.topk(5).indices.tolist()))
        md = (o - h).abs().max().item()
        if not top1:
            if cos >= TIE_COS:
                ties += 1
            else:
                real_top1_break = True
        if t == 0:
            prefill_cos = cos
        else:
            worst_decode_cos = min(worst_decode_cos, cos)
        tag = "PREFILL" if t == 0 else f"decode{t}"
        flag = "OK " if top1 else ("TIE" if cos >= TIE_COS else "DIF")
        print(
            f"  {tag:>7} cos={cos:.5f} top1={flag} "
            f"top5={ov}/5 maxabs={md:6.3f}  hf_argmax={h.argmax().item():>6} "
            f"ours_argmax={o.argmax().item():>6} gen={gen_ids[t]:>6}"
        )

    print("\n--- SUMMARY ---")
    print(f"  prefill (step 0) cos = {prefill_cos:.5f}  (3e-1 bar {args.cos_prefill})")
    print(f"  worst decode-step cos = {worst_decode_cos:.5f}  (3e-2 bar {args.cos_decode})")
    print(f"  top-1 ties (flip under cos>={TIE_COS}, benign) = {ties}; real top-1 breaks = {real_top1_break}")
    ok = prefill_cos >= args.cos_prefill and worst_decode_cos >= args.cos_decode and not real_top1_break
    print("VERDICT:", "DECODE PATH NUMERICALLY SOUND" if ok else "INVESTIGATE")


if __name__ == "__main__":
    main()
