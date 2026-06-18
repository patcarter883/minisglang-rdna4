#!/usr/bin/env python
"""Logit oracle, part 2: compute HF transformers' first-token logits on the SAME input ids
and compare to the RDNA4 engine's (cosine-sim, top-1, top-5 overlap, max-abs-diff). This is the
standing numerical oracle — tolerant of sub-ULP rounding, unlike greedy token-identity."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

MODEL = "Qwen/Qwen3-0.6B"


def main() -> None:
    d = torch.load("/engine/tools/ours_logits.pt")
    input_ids = d["input_ids"]
    ours = d["logits"]  # [N, vocab]

    hf = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to("cuda").eval()
    hl = []
    for ids in input_ids:
        t = torch.tensor([ids], device="cuda")
        with torch.no_grad():
            hl.append(hf(t).logits[0, -1, :].float().cpu())
    hf_logits = torch.stack(hl)

    print("=== logit oracle: RDNA4 engine vs HF transformers (bf16) ===")
    worst_cos = 1.0
    all_top1 = True
    for i in range(len(input_ids)):
        o, h = ours[i], hf_logits[i]
        cos = F.cosine_similarity(o.unsqueeze(0), h.unsqueeze(0)).item()
        top1 = o.argmax().item() == h.argmax().item()
        ov = len(set(o.topk(5).indices.tolist()) & set(h.topk(5).indices.tolist()))
        md = (o - h).abs().max().item()
        worst_cos = min(worst_cos, cos)
        all_top1 &= top1
        print(
            f"p{i}: cos={cos:.5f} top1={'OK' if top1 else 'DIFF'} "
            f"top5={ov}/5 maxabs={md:.3f} "
            f"argmax(ours={o.argmax().item()}, hf={h.argmax().item()})"
        )
    print(f"\nSUMMARY: worst cos-sim={worst_cos:.5f}, all top-1 match={all_top1}")
    print("VERDICT:", "NUMERICALLY SOUND" if (worst_cos > 0.99 and all_top1) else "INVESTIGATE")


if __name__ == "__main__":
    main()
