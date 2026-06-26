#!/usr/bin/env python
"""A/B validation for the native-HIP serve path (decode / cold-prefill / extend-prefill / tail).

Runs the dense engine under env flags and compares HIP-on vs HIP-off:
  ON  (default)            -> attn_decode + attn_hip + attn_prefill_paged + tail_hip
  OFF (MINISGL_ATTN_HIP=0,
       MINISGL_TAIL_HIP=0) -> pure Triton attention + torch elementwise

Two gates:
  1. FIRST-TOKEN LOGITS (decode-free, no accumulation): captured by hooking the sampler at
     max_tokens=1. Round1 = long prompts (>= 3 pages) -> COLD prefill. Round2 = those prompts
     EXTENDED by a suffix; round1 already cached the >= 3-page prefix, so round2 is an EXTEND
     prefill through the paged/chunked kernel (attn_prefill_paged). Comparing on-vs-off logits
     isolates the prefill + tail wiring (cos / top-1) with zero decode drift -> the tight gate.
  2. GREEDY TOKEN-DIFF (informational): 16 greedy steps; late flips between two correct bf16
     kernels are expected, an EARLY/structural divergence is a wiring bug.

The live backend's _hip_decode / _hip_prefill / _hip_prefill_paged are wrapped with counters so the
run PROVES which native paths fired (only present when MINISGL_ATTN_HIP != 0).

Writes <tag>.pt {cold_logits, extend_logits, token_ids, counts, env}. Compare with --diff a.pt b.pt.
"""
from __future__ import annotations

import argparse
import os

import torch

from minisgl.core import SamplingParams
from minisgl.llm import LLM

MODEL = os.environ.get("MINISGL_VALIDATE_MODEL", "Qwen/Qwen3-0.6B")

# Round 1: long prompts (>= ~48 tokens => >= 3 pages at page_size 16) so their prefix is radix-
# cacheable. Round 2: each EXTENDS its round-1 prompt verbatim, so the cached >=3-page prefix hits
# (cached_len>0) and the prefill is chunked/paged (attn_prefill_paged) rather than cold.
ROUND1 = [
    "The history of the Roman Empire spans many centuries and is one of the most studied "
    "subjects in all of Western civilization, beginning with the legendary founding of the city "
    "and continuing through the republic and then the empire, until",
    "In modern computer science, a sorting algorithm is a well defined computational method that "
    "takes a sequence of elements and rearranges them into a specific order, and over the decades "
    "researchers have developed many such algorithms, including",
    "The water cycle, also known as the hydrological cycle, describes the continuous movement of "
    "water on, above, and below the surface of the Earth, and it begins when energy from the sun "
    "causes water at the surface to",
]
ROUND2 = [p + " a series of remarkable and well documented events that historians describe as"
          for p in ROUND1]


def _instrument(backend) -> dict:
    counts = {"decode": 0, "cold_prefill": 0, "paged_prefill": 0}
    for key, attr in (("decode", "_hip_decode"), ("cold_prefill", "_hip_prefill"),
                      ("paged_prefill", "_hip_prefill_paged")):
        fn = getattr(backend, attr, None)
        if fn is None:
            continue

        def make(fn, key):
            def wrapped(*a, **k):
                counts[key] += 1
                return fn(*a, **k)
            return wrapped

        setattr(backend, attr, make(fn, key))
    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-dir", default="/engine/tools/.hip_ab")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    attn = os.environ.get("MINISGL_ATTN_HIP", "1")
    tail = os.environ.get("MINISGL_TAIL_HIP", "1")
    print(f"[ab] tag={args.tag} MINISGL_ATTN_HIP={attn} MINISGL_TAIL_HIP={tail} model={MODEL}",
          flush=True)

    llm = LLM(MODEL, dtype=torch.bfloat16, cuda_graph_max_bs=0, page_size=16, memory_ratio=0.4)
    counts = _instrument(llm.engine.attn_backend)
    print(f"[ab] backend={type(llm.engine.attn_backend).__name__} wired={list(counts)}", flush=True)

    # Sampler hook -> capture first-token logits per generate() (single-prompt batches keep order).
    cap: dict[str, torch.Tensor] = {}
    orig = llm.engine.sampler.sample

    def hook(logits, a):
        cap["l"] = logits.detach().float().cpu().clone()
        return orig(logits, a)

    llm.engine.sampler.sample = hook  # type: ignore[method-assign]

    def first_token_logits(prompts):
        sp = SamplingParams(temperature=0.0, max_tokens=1)
        out = []
        for p in prompts:
            cap.clear()
            llm.generate([p], sp)
            out.append(cap["l"][0])  # [vocab]
        return torch.stack(out)

    cold_logits = first_token_logits(ROUND1)      # cold prefill + tail
    print(f"[ab] cold (round1) done; counts={counts}", flush=True)
    extend_logits = first_token_logits(ROUND2)    # extend/paged prefill + tail
    print(f"[ab] extend (round2) done; counts={counts}", flush=True)

    # Informational greedy token-diff (16 steps).
    sp16 = SamplingParams(temperature=0.0, max_tokens=16)
    token_ids = [list(o["token_ids"]) for o in llm.generate(ROUND1, sp16)]

    path = os.path.join(args.out_dir, f"{args.tag}.pt")
    torch.save({"tag": args.tag, "env": {"attn": attn, "tail": tail},
                "cold_logits": cold_logits, "extend_logits": extend_logits,
                "token_ids": token_ids, "counts": counts}, path)
    print(f"[ab] wrote {path}  counts={counts}", flush=True)


def diff(a_path: str, b_path: str) -> None:
    import torch.nn.functional as F
    a = torch.load(a_path)
    b = torch.load(b_path)
    print(f"=== HIP-on vs HIP-off  ({a['tag']} env={a['env']}  vs  {b['tag']} env={b['env']}) ===")
    print(f"ON  path counts: {a['counts']}")
    print(f"OFF path counts: {b['counts']}")
    paged_on = a["counts"].get("paged_prefill", 0)

    def logit_gate(name, la, lb):
        worst_cos, all_top1 = 1.0, True
        for i in range(la.shape[0]):
            cos = F.cosine_similarity(la[i].unsqueeze(0), lb[i].unsqueeze(0)).item()
            top1 = la[i].argmax().item() == lb[i].argmax().item()
            worst_cos = min(worst_cos, cos)
            all_top1 &= top1
            print(f"  {name} p{i}: cos={cos:.5f} top1={'OK' if top1 else 'DIFF'} "
                  f"argmax(on={la[i].argmax().item()}, off={lb[i].argmax().item()})")
        ok = worst_cos > 0.999 and all_top1
        print(f"  {name}: worst cos={worst_cos:.5f} all-top1={all_top1} -> "
              f"{'PASS' if ok else 'INVESTIGATE'}")
        return ok

    print("--- first-token logits: COLD prefill + tail (decode-free) ---")
    cold_ok = logit_gate("cold", a["cold_logits"], b["cold_logits"])
    print("--- first-token logits: EXTEND/paged prefill + tail (decode-free) ---")
    ext_ok = logit_gate("extend", a["extend_logits"], b["extend_logits"])

    ta, tb = a["token_ids"], b["token_ids"]
    ident = sum(ta[i] == tb[i] for i in range(min(len(ta), len(tb))))
    print(f"--- greedy 16-step token-diff (informational): {ident}/{len(ta)} identical ---")
    for i in range(len(ta)):
        if ta[i] != tb[i]:
            j = next((k for k in range(min(len(ta[i]), len(tb[i]))) if ta[i][k] != tb[i][k]), -1)
            print(f"    seq {i}: first divergence @ {j} (0-3 => structural bug; late => bf16 drift)")

    print(f"\nattn_prefill_paged fired in ON run: {paged_on} "
          f"({'OK' if paged_on > 0 else 'NOT EXERCISED'})")
    print("VERDICT:", "PASS" if (cold_ok and ext_ok and paged_on > 0) else "INVESTIGATE")


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 4 and sys.argv[1] == "--diff":
        diff(sys.argv[2], sys.argv[3])
    else:
        main()
