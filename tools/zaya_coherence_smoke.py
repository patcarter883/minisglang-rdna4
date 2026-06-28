"""ZAYA1-8B-fp8 offline coherence smoke — loads on GPU (single card, eager, --attn hip
equivalent for the CCA-hybrid path) and emits a short greedy completion.

Run INSIDE vllm22-w4a8:combined via the gpu-lease wrapper:
    PYTHONPATH=/engine/python:/engine python /engine/tools/zaya_coherence_smoke.py
"""

from __future__ import annotations

import sys

import torch
from minisgl.core import SamplingParams
from minisgl.llm import LLM

MODEL = "/models/ZAYA1-8B-fp8"


def main() -> int:
    print(f"[zaya-smoke] torch={torch.__version__} hip_avail={torch.cuda.is_available()}", flush=True)
    # CCA recurrent state is forced fp32 inside the engine (CCAStateCache dtype=float32 — the
    # mamba-cache-dtype float32 equivalent). Model compute dtype = bf16 (the dense/router path).
    llm = LLM(
        model_path=MODEL,
        dtype=torch.bfloat16,
        attention_backend="hip",   # native-HIP attention backend (production, non-graph first try)
        cuda_graph_max_bs=0,       # eager (CCA graph capture is a follow-up)
        memory_ratio=0.80,
        max_running_req=4,
    )
    print("[zaya-smoke] engine constructed; running generation", flush=True)

    prompts = [
        "Q: What is the capital of France?\nA:",
        "Q: What is 2 + 2?\nA:",
    ]
    sp = SamplingParams(temperature=0.0, max_tokens=48)
    outs = llm.generate(prompts, sp)
    ok = True
    for p, o in zip(prompts, outs):
        text = o["text"]
        print(f"\n[PROMPT] {p!r}\n[OUTPUT] {text!r}", flush=True)
        if not text.strip():
            ok = False
    print(f"\n[zaya-smoke] RESULT: {'PASS' if ok else 'EMPTY-OUTPUT'}", flush=True)
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
