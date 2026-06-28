"""ZAYA1-8B-fp8 serving matrix: concurrency (batch) x context-length -> prefill / decode TPOT /
throughput / KV-fit, for working out real-world RSA serving limits.

The binding limit is the KV token pool (num_pages*page_size): all concurrently-decoding sequences
share it, so concurrency B at context L needs ~B*(L+steps) <= KV_budget. RSA fans one user request
into N rollouts (default N=16) => N concurrent decode seqs; the Markovian variant caps each rollout's
carried context to a ~4000-token tail (+ fresh generation), so the RSA-relevant region is roughly
B in {4..16} x L in {4k..12k}, NOT 32k.

Decode TPOT is isolated as t(max_tokens=1+STEPS) - t(max_tokens=1) over STEPS, at batch B / context L
(the shared prefill cancels). prefill_ms ~= t(max_tokens=1) (prefill of B*L + 1 decode).

Run INSIDE vllm22-w4a8:combined via the gpu-lease wrapper (graph capture on, scatter off for capture):
    PYTHONPATH=/engine/python:/engine MINISGL_MOE_SCATTER=0 python /engine/tools/zaya_serving_matrix.py
"""
from __future__ import annotations

import os
import sys
import time

import torch
from minisgl.core import SamplingParams
from minisgl.llm import LLM

MODEL = "/models/ZAYA1-8B-fp8"
CONC = [int(x) for x in os.environ.get("CONC", "1,2,4,8,16").split(",")]
CTX = [int(x) for x in os.environ.get("CTX", "1024,4096,8192,16384,32768").split(",")]
STEPS = int(os.environ.get("STEPS", "16"))
# Chunked prefill: cap each prefill FORWARD to CHUNK tokens so prefill-activation memory is bounded
# (the scheduler chunks any longer prompt). This is the production path — without it, B prompts pack
# into one giant prefill forward whose activations OOM well before the decode KV budget is the limit.
CHUNK = int(os.environ.get("CHUNK", "2048"))
MAXLEN = max(CTX) + STEPS + 16


def _prompt(n: int) -> list[int]:
    # Valid, non-special token ids; content is irrelevant to timing.
    return [(i % 50000) + 5 for i in range(n)]


def _time(llm: LLM, prompts, max_tokens: int) -> float:
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    llm.generate(prompts, sp)
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def main() -> int:
    gbs = max(CONC)
    print(f"[matrix] CONC={CONC} CTX={CTX} STEPS={STEPS} graph_bs={gbs} maxlen={MAXLEN}", flush=True)
    llm = LLM(
        model_path=MODEL,
        dtype=torch.bfloat16,
        attention_backend="hip",
        cuda_graph_max_bs=gbs,
        memory_ratio=0.90,
        max_running_req=max(CONC),
        max_seq_len_override=MAXLEN,
        max_extend_tokens=CHUNK,  # chunked prefill -> bounded prefill-activation memory
    )
    print(f"[matrix] chunked prefill: max_extend_tokens={CHUNK}", flush=True)
    eng = llm.engine
    page_size = getattr(eng, "page_size", None) or 1
    kv_budget = int(eng.num_pages * page_size)
    print(f"[matrix] KV_BUDGET_TOKENS={kv_budget} (num_pages={eng.num_pages} page_size={page_size})", flush=True)

    rows = []
    for L in CTX:
        for B in CONC:
            need = B * (L + STEPS)
            if need > int(kv_budget * 0.98):
                rows.append((B, L, "OVER_KV", need, None, None, None))
                print(f"[cell] B={B:<3} L={L:<6} SKIP over-KV (need {need} > {kv_budget})", flush=True)
                continue
            try:
                prompts = [_prompt(L) for _ in range(B)]
                _ = _time(llm, prompts, 1)  # warm (graph + this shape)
                t1 = _time(llm, prompts, 1)
                tT = _time(llm, prompts, 1 + STEPS)
                decode_ms = (tT - t1) * 1e3
                tpot_ms = decode_ms / STEPS
                thrpt = (B * 1e3 / tpot_ms) if tpot_ms > 0 else 0.0
                prefill_ms = t1 * 1e3
                rows.append((B, L, "OK", need, prefill_ms, tpot_ms, thrpt))
                print(f"[cell] B={B:<3} L={L:<6} prefill={prefill_ms:8.1f}ms "
                      f"TPOT={tpot_ms:7.2f}ms thrpt={thrpt:7.1f}tok/s (kv {need})", flush=True)
            except Exception as e:  # noqa: BLE001
                rows.append((B, L, f"ERR:{type(e).__name__}", need, None, None, None))
                print(f"[cell] B={B:<3} L={L:<6} ERROR {type(e).__name__}: {str(e)[:120]}", flush=True)

    # ---- emit matrices ----
    print(f"\n[matrix] KV_BUDGET_TOKENS={kv_budget}", flush=True)
    def grid(title, idx, fmt):
        print(f"\n===== {title} (rows=context L, cols=concurrency B) =====", flush=True)
        print("L \\ B    | " + " ".join(f"{b:>9}" for b in CONC), flush=True)
        for L in CTX:
            cells = []
            for B in CONC:
                r = next(x for x in rows if x[0] == B and x[1] == L)
                if r[2] != "OK":
                    cells.append(f"{r[2][:9]:>9}")
                else:
                    cells.append(f"{fmt(r[idx]):>9}")
            print(f"{L:>8} | " + " ".join(cells), flush=True)
    # row tuple = (B[0], L[1], status[2], need[3], prefill_ms[4], tpot_ms[5], thrpt[6])
    grid("DECODE TPOT (ms)", 5, lambda v: f"{v:.1f}")
    grid("DECODE THROUGHPUT (tok/s, aggregate)", 6, lambda v: f"{v:.0f}")
    grid("PREFILL (ms, B*L tokens)", 4, lambda v: f"{v:.0f}")
    print("\n[matrix] DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
