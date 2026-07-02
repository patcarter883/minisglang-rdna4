"""Phase 4-4 follow-up — vLLM greedy reference for the Qwen3.6-35B-A3B GDN-MoE, token-diffed
against minisgl's saved TP=2 serve output.

minisgl already booted the 35B TP=2 and greedy-generated the fixed prompts (the GDN forward on
torch.ops.gdn_hip.*, the routed experts on the W4A8/compressed-tensors path); that output lives in
tools/tp2_results/s2_moe_35b_tp2.json. This script runs the SAME model + prompts + greedy sampling
through the combined image's vLLM (which natively supports Qwen3_5MoeForConditionalGeneration and
runs GDN on its own fla Triton kernels), then diffs the two token/text streams.

Posture (matches the rest of Phase 4): minisgl and vLLM use different GDN kernels (native HIP WMMA
prefill / recurrent decode vs fla Triton chunk-scan) AND different MoE/quant paths, so they are NOT
expected to be bit-identical. A long common greedy prefix + coherent agreement is the pass signal; a
divergence at token 0 or incoherent text would flag a real mapping/numerics bug. vLLM token_ids are
dumped for the record.

GPU work — run UNDER a 2-card lease (TP=2), inside the combined image:
    gpu-lease -n 2 -- bash -c 'docker run ... \
        python /engine/tools/qwen3_5_moe_vllm_parity.py'
"""
from __future__ import annotations

import argparse
import json

from vllm import LLM, SamplingParams

# MUST match tools/tp_serve_probe.py PROMPTS + MAX_TOKENS so the diff is apples-to-apples.
PROMPTS = [
    "The capital of France is",
    "Once upon a time, in a small village,",
    "The three primary colors are red, blue, and",
    "Q: What is 2 + 2? A:",
]


def _prefix_chars(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--memory-ratio", type=float, default=0.92)
    ap.add_argument("--ours", default="/engine/tools/tp2_results/s2_moe_35b_tp2.json")
    ap.add_argument("--out", default="/engine/tools/tp2_results/s2_moe_35b_vllm_tp2.json")
    args = ap.parse_args()

    print(f"[parity] loading {args.model} on vLLM (TP={args.tp}, eager) ...", flush=True)
    # This checkpoint registers as Qwen3_5MoeForConditionalGeneration => vLLM treats it as multimodal:
    # it would load the vision tower AND run a max-image-size profiling forward + reserve an encoder
    # cache, which (a) burns ~15 min and (b) leaves no room for the KV cache on a 16 GiB card (OOM at
    # cache-block sizing). minisgl loads it as pure text; match that here by forbidding mm inputs so
    # the profiling run is text-only and the encoder-cache reservation drops to ~0.
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        dtype="bfloat16",
        enforce_eager=True,
        gpu_memory_utilization=args.memory_ratio,
        max_model_len=2048,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 0, "video": 0},
    )
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    outs = llm.generate(PROMPTS, sp)
    vllm = [{"prompt": p, "text": o.outputs[0].text, "token_ids": list(o.outputs[0].token_ids)}
            for p, o in zip(PROMPTS, outs)]
    with open(args.out, "w") as f:
        json.dump({"model": args.model, "tp": args.tp, "generations": vllm}, f, indent=2)
    print(f"[parity] vLLM output written to {args.out}", flush=True)

    # minisgl side (text only — the OpenAI serve probe stored text per prompt).
    ours = {}
    try:
        with open(args.ours) as f:
            ours = {g["prompt"]: g["text"] for g in json.load(f).get("generations", [])}
    except FileNotFoundError:
        print(f"[warn] {args.ours} not found; printing vLLM ref only", flush=True)

    print("\n=== token-diff: minisgl 35B (saved TP=2) vs vLLM 35B (greedy, same prompts) ===", flush=True)
    exact = 0
    fracs = []
    for v in vllm:
        p, vt = v["prompt"], v["text"]
        ot = ours.get(p)
        print(f"\n=== {p!r}")
        print(f"  minisgl: {ot!r}")
        print(f"  vllm   : {vt!r}")
        if ot is None:
            continue
        if ot == vt:
            exact += 1
            print("  -> IDENTICAL text")
        else:
            m = _prefix_chars(ot, vt)
            frac = m / max(len(ot), len(vt), 1)
            fracs.append(frac)
            print(f"  -> diverge@char {m}/{max(len(ot), len(vt))} (prefix agreement {frac:.0%})")
    if fracs or exact:
        denom = exact + len(fracs)
        mean_prefix = (exact + sum(fracs)) / denom if denom else 0.0
        print("\n" + "=" * 60, flush=True)
        print(f"RESULT: {exact}/{denom} text-identical; mean prefix agreement {mean_prefix:.0%}", flush=True)
        print("(Different GDN kernels + MoE/quant paths => not bit-identical by design; a long common "
              "prefix + coherent text is the pass signal.)", flush=True)


if __name__ == "__main__":
    main()
