"""Phase 3d-4 — HF reference forward for Qwen3.5-4B (ground truth for the minisgl port).

Loads the model via plain transformers and runs ONE forward with output_hidden_states, printing:
  * the prompt token ids (so we can confirm minisgl tokenizes identically),
  * per-layer LAST-TOKEN residual-stream norm (== minisgl's `residual + x`, tagged hs[ii]),
  * the top-8 next-token logits for the last position.

Run in the combined image. Uses the SAME vendored FLA/conv kernels as minisgl (when on GPU),
so a divergence isolates minisgl's wiring/loading, not the kernels. GPU via the lease.
"""
from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--save", default=None, help="save last-token hidden_states stack to this .pt")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    ids = tok(args.prompt, return_tensors="pt").input_ids
    print(f"[ref] prompt={args.prompt!r}")
    print(f"[ref] token_ids={ids[0].tolist()}")

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    ids = ids.to(model.device)
    with torch.no_grad():
        out = model(ids, output_hidden_states=True, use_cache=False)

    hs = out.hidden_states  # tuple len num_layers+1, each (1, seq, hidden)
    last_stack = []
    for i, h in enumerate(hs):
        hl = h[0, -1].float()
        last_stack.append(hl)
        print(
            f"[ref] hs[{i:02d}]: norm={hl.norm().item():.3e} absmax={hl.abs().max().item():.3e}"
        )

    logits = out.logits[0, -1].float()
    top = logits.topk(8)
    print(
        f"[ref] logits last: top_ids={top.indices.tolist()} "
        f"top_vals={[round(v,2) for v in top.values.tolist()]}"
    )
    if args.save:
        torch.save(torch.stack(last_stack).cpu(), args.save)
        print(f"[ref] saved {args.save}")


if __name__ == "__main__":
    main()
