"""ZAYA1-8B-fp8 CUDA-graph-capture smoke — loads on GPU (single card, --attn hip,
graph capture ENABLED) and emits a short greedy completion, validating that the CCA
decode kernel (torch.ops.zaya_cca.cca_decode_qk, which rolls conv_states in place) and
the MoE path are graph-capturable and stay coherent.

Run INSIDE vllm22-w4a8:combined via the gpu-lease wrapper, with MINISGL_MOE_SCATTER=0:
    PYTHONPATH=/engine/python:/engine MINISGL_MOE_SCATTER=0 \
      python /engine/tools/zaya_graph_smoke.py
"""

from __future__ import annotations

import sys

import torch
from minisgl.core import SamplingParams
from minisgl.llm import LLM

MODEL = "/models/ZAYA1-8B-fp8"


def main() -> int:
    print(f"[zaya-graph] torch={torch.__version__} hip_avail={torch.cuda.is_available()}", flush=True)
    llm = LLM(
        model_path=MODEL,
        dtype=torch.bfloat16,
        attention_backend="hip",   # native-HIP attention backend (the only capture-capable path)
        cuda_graph_max_bs=8,       # capture decode graphs for bs in [1,2,4,8]
        memory_ratio=0.80,
        max_running_req=8,
    )
    print("[zaya-graph] engine constructed (graph capture done); running generation", flush=True)

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
    print(f"\n[zaya-graph] RESULT: {'PASS' if ok else 'EMPTY-OUTPUT'}", flush=True)
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
