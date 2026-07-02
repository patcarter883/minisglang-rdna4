"""Phase 4-4 follow-up (HTTP client) — token-diff minisgl's saved 35B output vs the vLLM production
serve.

Runs on the HOST (stdlib only, no GPU, no lease). Assumes the vLLM OpenAI server is already up via
the vllm-gfx1201 compose `serve` profile (VLLM_MODEL_ID=cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit) — that
path carries the production env that keys the warm GDN autotune cache AND the memory flags
(kv-cache-dtype=fp8, util 0.95, max-num-seqs 8) that a hand-rolled run kept OOMing without.

Uses /v1/completions (raw text continuation, NO chat template) to match minisgl's raw-prompt serve
output saved in tools/tp2_results/s2_moe_35b_tp2.json. Greedy (temperature 0). Diffs text + dumps
vLLM token_ids. Posture: different GDN kernels + MoE/quant paths => not bit-identical; a long common
greedy prefix + coherent agreement is the pass signal.

STATUS (2026-06-25): tooling complete, run NOT yet captured — vLLM 35B boot on this box is the
blocker, not the diff. Hard-won boot recipe for whoever finishes it (4 failed attempts distilled):
  - Launch the vLLM serve via the vllm-gfx1201 compose `serve` profile, NOT a hand-rolled docker run:
    only compose carries the env (VLLM_ROCM_USE_W4A8_FP8_WMMA, VLLM_ROCM_W4A8_LAYOUT=single,
    TRITON_CACHE_AUTOTUNING=1, CU_NUM, VLLM_TP_CU_WEIGHTS) that keys the warm GDN autotune cache.
  - MUST pass memory flags or it OOMs at KV-cache sizing on 16 GB TP=2 (the vision tower loads to
    12.27 GiB/rank): --kv-cache-dtype=fp8 --gpu-memory-utilization=0.95 --max-num-seqs=8
    --max-num-batched-tokens=4096 --max-model-len=2048 --limit-mm-per-prompt={"image":0,"video":0}.
  - Add --enforce-eager: the default (cudagraph FULL_AND_PIECEWISE) capture HANGS on ROCm for this
    hybrid (vLLM #19579/#39010, per the compose comments). Eager skips compile+capture.
  - Even eager, expect a SLOW boot: ~82 s weight load + ~15-20 min CPU-bound Triton kernel compile
    for the GDN/FLA chunk_* kernels at the eager shapes (autotune *config* cache hits, but the kernel
    *binaries* for these shapes aren't warm). Give the health poll a >=40-min budget; the workers peg
    CPU (~183%) at 3% GPU during this — that's compile, not a hang. Do NOT kill it early.
Example (under a 2-card lease, from ~/code/vllm-gfx1201):
    VLLM_MODEL_ID=cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit gpu-lease -n 2 --name serve35be -- \\
      docker compose --profile serve run --rm --service-ports serve \\
        cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit --host=0.0.0.0 --port=8000 --tensor-parallel-size=2 \\
        --enforce-eager --kv-cache-dtype=fp8 --gpu-memory-utilization=0.95 --max-num-seqs=8 \\
        --max-num-batched-tokens=4096 --max-model-len=2048 \\
        '--limit-mm-per-prompt={"image":0,"video":0}' --trust-remote-code
then (host): python3 tools/qwen3_5_moe_vllm_parity_http.py
"""
from __future__ import annotations

import argparse
import json
import urllib.request

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


def _completion(base: str, prompt: str, max_tokens: int, timeout: float) -> dict:
    payload = {"model": "", "prompt": prompt, "max_tokens": max_tokens,
               "temperature": 0.0, "stream": False}
    req = urllib.request.Request(f"{base}/v1/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read().decode())
    ch = d["choices"][0]
    return {"text": ch.get("text"), "token_ids": ch.get("token_ids")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--ours", default="tools/tp2_results/s2_moe_35b_tp2.json")
    ap.add_argument("--out", default="tools/tp2_results/s2_moe_35b_vllm_tp2.json")
    args = ap.parse_args()

    vllm = []
    for p in PROMPTS:
        c = _completion(args.base, p, args.max_tokens, args.timeout)
        vllm.append({"prompt": p, "text": c["text"], "token_ids": c["token_ids"]})
        print(f"[vllm] {p!r} -> {c['text']!r}", flush=True)
    with open(args.out, "w") as f:
        json.dump({"model": "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit (vLLM serve)", "generations": vllm}, f, indent=2)
    print(f"\n[parity] vLLM output written to {args.out}", flush=True)

    with open(args.ours) as f:
        ours = {g["prompt"]: g["text"] for g in json.load(f).get("generations", [])}

    print("\n=== token-diff: minisgl 35B (saved TP=2) vs vLLM 35B serve (greedy, /v1/completions) ===",
          flush=True)
    exact = 0
    fracs = []
    for v in vllm:
        p, vt, ot = v["prompt"], v["text"] or "", ours.get(v["prompt"])
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
    denom = exact + len(fracs)
    mean_prefix = (exact + sum(fracs)) / denom if denom else 0.0
    print("\n" + "=" * 60, flush=True)
    print(f"RESULT: {exact}/{denom} text-identical; mean prefix agreement {mean_prefix:.0%}", flush=True)
    print("(Different GDN kernels + MoE/quant paths => not bit-identical by design; a long common "
          "prefix + coherent text is the pass signal.)", flush=True)


if __name__ == "__main__":
    main()
