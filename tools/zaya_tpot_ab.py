"""ZAYA1-8B-fp8 decode TPOT probe (M=1) for the W8A8 A/B comparison.

Toggled by MINISGL_ZAYA_OLDMOE (0 = native W8A8 kernel, 1 = legacy dequant->Triton). Runs a
single-request greedy decode, warms up, then times a fixed decode length and reports per-output-token
latency (TPOT, ms). Eager (cuda_graph_max_bs=0) so we measure the kernel, not graph replay.

Run INSIDE vllm22-w4a8:combined via the gpu-lease wrapper:
    PYTHONPATH=/engine/python:/engine MINISGL_MOE_SCATTER=0 MINISGL_ZAYA_OLDMOE=<0|1> \
      python /engine/tools/zaya_tpot_ab.py
"""

from __future__ import annotations

import os
import sys
import time

import torch
from minisgl.core import SamplingParams
from minisgl.llm import LLM

MODEL = "/models/ZAYA1-8B-fp8"


def main() -> int:
    kern = "OLD dequant->Triton" if os.environ.get("MINISGL_ZAYA_OLDMOE", "0") == "1" else "NEW w8a8"
    # MINISGL_ZAYA_GRAPH_BS: 0 = eager (time the MoE kernel); >0 = capture decode graphs up to that bs
    # and time graph replay (the production decode path).
    gbs = int(os.environ.get("MINISGL_ZAYA_GRAPH_BS", "0"))
    mode = f"{kern} [{'graph bs=' + str(gbs) if gbs else 'eager'}]"
    print(f"[zaya-tpot] mode={mode!r} torch={torch.__version__} hip={torch.cuda.is_available()}", flush=True)
    llm = LLM(
        model_path=MODEL,
        dtype=torch.bfloat16,
        attention_backend="hip",
        cuda_graph_max_bs=gbs,  # 0 = eager; >0 = graph replay
        memory_ratio=0.80,
        max_running_req=1,
    )
    prompt = "Q: Tell me a short story about a robot.\nA:"

    # Warmup (triggers any lazy compile / autotune; result discarded).
    _ = llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=16))

    # Timed run: single request, M=1 decode. Wall time / (out_tokens-1) ~= TPOT (first token is
    # prefill-bound; subtracting one approximates the steady-state decode cost).
    n_tok = 128
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    outs = llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=n_tok))
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    out_ids = outs[0]["token_ids"]
    n = len(out_ids)
    tpot_ms = (dt / max(n - 1, 1)) * 1e3
    print(f"\n[OUTPUT] {outs[0]['text']!r}", flush=True)
    print(f"[zaya-tpot] mode={mode!r} out_tokens={n} total={dt*1e3:.1f}ms "
          f"TPOT={tpot_ms:.2f}ms ({1e3/tpot_ms:.1f} tok/s)", flush=True)
    print(f"[zaya-tpot] RESULT_TPOT mode={mode} tpot_ms={tpot_ms:.3f} toks={n}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
