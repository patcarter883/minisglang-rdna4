#!/usr/bin/env python
"""What does it COST to pin the quantized dense path to ONE arm across the decode/verify band?

`tools/quant_m_invariance.py` established that the three dense arms are NOT interchangeable:
`prefill_wmma` and `wmma_tiled_tuned` are bit-identical to each other everywhere, but `decode_gemv`
differs from both. So the ONLY M at which a token's value changes is the gemv<->WMMA crossover
(`_W4A8_GEMV_MAX_INT4 = 8`, `_W4A8_GEMV_MAX_E2M1 = 16`) -- and that crossover sits inside the
spec-decode verify band (verify M = bs*(K+1); MTP K=4 at bs>=2 is M>=10, DFlash K=15 is M=16).

The candidate fix is the same one `layers/minv.py` already applies to the bf16 path, where
`_DECODE_GEMV_MAXM = 16` exists verbatim so "ordinary decode and spec-decode VERIFY (M=K+1) land on
the SAME kernel rather than opposite sides of the threshold": raise the int4 cap 8 -> 16 so the whole
M<=16 band is one arm. This measures what that costs.

TIMING. Per-call `synchronize()` has a ~40 us floor on this box -- larger than these kernels -- so
every number here is a CUDA-graph replay of R back-to-back calls, event-bracketed, divided by R.
The R calls rotate over R DISTINCT weight copies sized in BYTES to exceed the 64 MB MALL, because a
single re-read weight sits in Infinity Cache and flatters whichever arm re-reads B most.

    gpu-lease -n 1 -- bash tools/quant_m_invariance_run.sh   # (edit the tool path) or run directly
"""
from __future__ import annotations

import argparse
import sys

import torch

DEV = torch.device("cuda:0")
MALL_BYTES = 96 << 20


def pack_uint4_2d(w: torch.Tensor) -> torch.Tensor:
    N, K = w.shape
    w = w.to(torch.int32)
    packed = torch.zeros((N, K // 8), dtype=torch.int32, device=w.device)
    for i in range(8):
        packed |= (w[:, i::8] & 0xF) << (i * 4)
    return packed


def rotation(N: int, K: int, g: int):
    """R distinct weight sets whose TOTAL bytes exceed the MALL."""
    wbytes = N * (K // 8) * 4 + N * (K // g) * 2
    R = max(2, min(24, (MALL_BYTES + wbytes - 1) // wbytes + 1))
    ws = []
    for _ in range(R):
        wp = pack_uint4_2d(torch.randint(0, 16, (N, K), dtype=torch.int8, device=DEV))
        sc = (torch.randn(N, K // g, device=DEV).abs() * 0.02 + 0.002).to(torch.float16)
        ws.append((wp, sc))
    return ws, R, wbytes


def time_graph(fn, ws, reps=30, warmup=5):
    """Capture one replay of len(ws) calls; return us per call."""
    import torch.cuda

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warmup):
            for w in ws:
                fn(w)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    pool = torch.cuda.graph_pool_handle()
    with torch.cuda.graph(graph, pool=pool):
        for w in ws:
            fn(w)
    torch.cuda.synchronize()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(reps):
        graph.replay()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) * 1e3 / (reps * len(ws))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    fh = open(args.out, "w") if args.out else None

    def out(s=""):
        print(s, flush=True)
        if fh:
            fh.write(s + "\n"); fh.flush()

    import fp8_wmma as W

    torch.manual_seed(0)
    out(f"device: {torch.cuda.get_device_name(0)}   fp8_wmma: {W.__file__}")
    out("graph-replay timed, weights rotated past the 64 MB MALL; us per call\n")

    SHAPES = [
        # Gemma4-26B-A4B qat-AWQ-INT4 (g=32, fp16), TP=2 per rank
        ("g4.o_proj(local)",  2048, 2816, 32, torch.float16),
        ("g4.o_proj(global)", 4096, 2816, 32, torch.float16),
        ("g4.qkv_q",          2816, 2048, 32, torch.float16),
        ("g4.gate_up",        2816, 2112, 32, torch.float16),
        ("g4.dense_down",     1056, 2816, 32, torch.float16),
        # Qwen3.6-35B-A3B-AWQ-4bit (g=128, bf16), TP=2 per rank
        ("q35.o_proj",        2048, 4096, 128, torch.bfloat16),
        ("q35.gate_up",       4096, 3072, 128, torch.bfloat16),
    ]
    MS = [1, 2, 4, 8, 9, 10, 12, 16, 20, 32, 64]
    ARMS = ("decode_gemv", "prefill_wmma", "wmma_tiled_tuned")

    for name, K, N, g, dt in SHAPES:
        ws, R, wbytes = rotation(N, K, g)
        out(f"=== {name}  K={K} N={N} g={g} {str(dt).split('.')[-1]}  "
            f"(rotation {R} x {wbytes/1e6:.1f} MB = {R*wbytes/1e6:.0f} MB) ===")
        out(f"    {'M':>4} {'decode_gemv':>12} {'prefill_wmma':>13} {'wmma_tiled':>12}   "
            f"{'gemv speedup vs best WMMA':>26}")
        for M in MS:
            x = (torch.randn(M, K, device=DEV) * 0.3).to(dt)
            row = {}
            for arm in ARMS:
                if arm == "decode_gemv" and M > 16:
                    continue
                fn = (lambda a: (lambda w: W.mmq_fp8_gemm(x, w[0], w[1], kernel=a, w_zeros=None,
                                                          weight_is_e2m1=False)))(arm)
                try:
                    row[arm] = time_graph(fn, ws)
                except Exception as e:  # noqa: BLE001
                    row[arm] = float("nan")
                    out(f"      (M={M} {arm}: {type(e).__name__}: {e})")
            best_wmma = min(row.get("prefill_wmma", float("inf")),
                            row.get("wmma_tiled_tuned", float("inf")))
            sp = f"{best_wmma / row['decode_gemv']:.2f}x" if "decode_gemv" in row else "-"
            def s(k):
                return f"{row[k]:.2f}" if k in row else "-"
            out(f"    {M:>4} {s('decode_gemv'):>12} {s('prefill_wmma'):>13} "
                f"{s('wmma_tiled_tuned'):>12}   {sp:>26}")
            del x
        del ws
        torch.cuda.empty_cache()
        out("")

    out("done.")
    if fh:
        fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
