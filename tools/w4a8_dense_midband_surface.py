#!/usr/bin/env python
"""Does the w4a8 dense MID-BAND rule (`gemv_max < M < 64 -> prefill_wmma`) have ANY regime?

`_pick_dense_kernel` sends the mid-band to `prefill_wmma` on the claim that "its conservative config
still wins the small-M/wide-N corner (re-bench: tiled loses only at N>=6144, M<=32)". Nothing in the
repo ever measured that corner: the shapes benched by `tools/w4a8_dense_arm_cost.py` top out at
N=4096, and on those `wmma_tiled_tuned` wins the whole mid-band. But N>=6144 dense quantized layers
ARE shipped (Qwen3.6-27B gate_up N=34816/17408, Laguna-XS.2 gate_up 16384/8192, GLM-4.7-Flash
gate_up 20480/10240, Qwen3.5-4B gate_up 18432/9216), so the corner is a real regime and the claim
has to be tested there, not assumed away.

This sweeps the (M, N) surface across the mid-band on BOTH sides of the claimed N=6144 boundary, at
the K values the engine actually uses, and reports the winning arm per cell.

TIMING (both traps this repo has already paid for):
  * per-call `synchronize()` has a ~40 us wall floor on this box, a large fraction of a 77 us kernel,
    so every number is a CUDA-graph replay of R back-to-back calls, event-bracketed, / R.
  * the R weight copies are sized in BYTES to exceed the 64 MB MALL. A previous sweep capped the
    rotation at 24 COPIES, which for a 0.72 MB weight is 17 MB -- entirely resident, so its "cold"
    baseline was hot. Here R is derived from the target byte count and the achieved MB is PRINTED,
    so a rotation that failed to bust the MALL is visible in the output rather than silent.
  * `mmq_fp8_gemm` requires an explicit `kernel=`; its old default silently ran the ~1000x-slower
    scalar reference.

    gpu-lease -n 1 --timeout 3600 -- bash tools/w4a8_dense_midband_surface_run.sh --out /engine/_surface.txt
"""
from __future__ import annotations

import argparse
import sys

import torch

DEV = torch.device("cuda:0")
# 64 MB MALL (Infinity Cache) on gfx1201. Target 2.5x so the rotation is unambiguously cold even if
# the cache is more effective than its nominal size.
ROT_TARGET_BYTES = 160 << 20
ROT_MAX_COPIES = 160  # only a graph-size / VRAM guard; NEVER the binding constraint (see below)


def pack_uint4_2d(w: torch.Tensor) -> torch.Tensor:
    N, K = w.shape
    w = w.to(torch.int32)
    packed = torch.zeros((N, K // 8), dtype=torch.int32, device=w.device)
    for i in range(8):
        packed |= (w[:, i::8] & 0xF) << (i * 4)
    return packed


def rotation(N: int, K: int, g: int, zeros: bool):
    """R distinct weight sets whose TOTAL bytes exceed ROT_TARGET_BYTES (sized in BYTES, not copies)."""
    wbytes = N * (K // 8) * 4 + N * (K // g) * 2 + (((N // 8) * (K // g) * 4) if zeros else 0)
    R = max(2, min(ROT_MAX_COPIES, -(-ROT_TARGET_BYTES // wbytes) + 1))
    ws = []
    for _ in range(R):
        wp = pack_uint4_2d(torch.randint(0, 16, (N, K), dtype=torch.int8, device=DEV))
        sc = (torch.randn(N, K // g, device=DEV).abs() * 0.02 + 0.002).to(torch.float16)
        wz = (
            torch.randint(0, 1 << 30, (N // 8, K // g), dtype=torch.int32, device=DEV)
            if zeros
            else None
        )
        ws.append((wp, sc, wz))
    return ws, R, wbytes


def time_graph(fn, ws, reps=20, warmup=3):
    """Capture one replay of len(ws) calls; return us per call."""
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
    us = e0.elapsed_time(e1) * 1e3 / (reps * len(ws))
    del graph, pool
    return us


# Every quantized DENSE linear this engine can dispatch, from the cached checkpoints, at TP=1 and
# TP=2 per-rank. `lm_head` is deliberately absent: minisgl's ParallelLMHead carries no quant_method,
# so the wide-N logits GEMM never reaches `_pick_dense_kernel` (it is a plain embedding matmul).
# (name, K, N, group, dtype, awq_zeros)
REAL_SHAPES = [
    # --- Gemma4-26B-A4B qat-AWQ-INT4 (compressed-tensors, symmetric, g=32, fp16) ---
    ("g4.q_proj      tp1", 2816, 4096, 32, torch.float16, False),
    ("g4.q_proj      tp2", 2816, 2048, 32, torch.float16, False),
    ("g4.kv_proj     tp2", 2816, 1024, 32, torch.float16, False),
    ("g4.o_proj      tp1", 4096, 2816, 32, torch.float16, False),
    ("g4.o_proj      tp2", 2048, 2816, 32, torch.float16, False),
    ("g4.gate_up     tp1", 2816, 4224, 32, torch.float16, False),
    ("g4.gate_up     tp2", 2816, 2112, 32, torch.float16, False),
    ("g4.down        tp2", 1056, 2816, 32, torch.float16, False),
    # --- Qwen3.6-35B-A3B-AWQ-4bit (AWQ asym, g=32, bf16) ---
    ("q35.q_proj     tp1", 2048, 4096, 32, torch.bfloat16, True),
    ("q35.q_proj     tp2", 2048, 2048, 32, torch.bfloat16, True),
    ("q35.o_proj     tp2", 2048, 2048, 32, torch.bfloat16, True),
    # --- Qwen3.6-27B AWQ-INT4 DENSE (g=32, bf16) -- the genuinely wide-N shipped model ---
    ("q27.q_proj     tp1", 5120, 6144, 32, torch.bfloat16, True),
    ("q27.q_proj     tp2", 5120, 3072, 32, torch.bfloat16, True),
    ("q27.o_proj     tp2", 3072, 5120, 32, torch.bfloat16, True),
    ("q27.gate_up    tp1", 5120, 34816, 32, torch.bfloat16, True),
    ("q27.gate_up    tp2", 5120, 17408, 32, torch.bfloat16, True),
    ("q27.down       tp2", 8704, 5120, 32, torch.bfloat16, True),
    # --- Laguna-XS.2-AWQ-INT4 (g=32, bf16) ---
    ("lag.gate_up    tp1", 2048, 16384, 32, torch.bfloat16, True),
    ("lag.gate_up    tp2", 2048, 8192, 32, torch.bfloat16, True),
    ("lag.q_proj     tp1", 2048, 6144, 32, torch.bfloat16, True),
    ("lag.down       tp2", 4096, 2048, 32, torch.bfloat16, True),
    # --- GLM-4.7-Flash-AWQ (g=128, bf16) ---
    ("glm.gate_up    tp2", 2048, 10240, 128, torch.bfloat16, True),
    # --- Qwen3.5-4B-AWQ-BF16-INT4 (g=32, bf16) ---
    ("q35b4.gate_up  tp2", 2560, 9216, 32, torch.bfloat16, True),
    ("q35b4.down     tp2", 4608, 2560, 32, torch.bfloat16, True),
]

# Synthetic (M, N) surface: N spans both sides of the claimed 6144 boundary at the real K values.
GRID_K = [2048, 2816, 5120]
GRID_N = [1024, 2048, 2816, 4096, 6144, 8192, 11264, 16384]

MID_M = [17, 20, 24, 32, 33, 40, 48, 56, 63]
BOUNDARY_M = [64, 80, 96, 128]

ARMS = ("prefill_wmma", "wmma_tiled_tuned", "prefill_wmma_ashuffle")


def call(W, arm, x, w, e2m1=False):
    return W.mmq_fp8_gemm(x, w[0], w[1], kernel=arm, w_zeros=w[2], weight_is_e2m1=e2m1)


def sweep(W, out, label, shapes, MS, arms, bitcheck=True):
    out(f"\n########## {label} ##########")
    rows = []
    for name, K, N, g, dt, zeros in shapes:
        ws, R, wbytes = rotation(N, K, g, zeros)
        tot = R * wbytes / 1e6
        flag = "" if tot >= 64.0 else "  <-- ROTATION DID NOT BUST THE MALL"
        out(
            f"\n=== {name}  K={K} N={N} g={g} {str(dt).split('.')[-1]} "
            f"zeros={'awq' if zeros else 'sym'}  "
            f"(rotation {R} x {wbytes/1e6:.2f} MB = {tot:.0f} MB){flag} ==="
        )
        hdr = f"    {'M':>4} " + "".join(f"{a:>22}" for a in arms) + f"   {'winner':>22} {'margin':>8}"
        out(hdr)
        for M in MS:
            x = (torch.randn(M, K, device=DEV) * 0.3).to(dt)
            t = {}
            for arm in arms:
                try:
                    t[arm] = time_graph(lambda w, a=arm: call(W, a, x, w), ws)
                except Exception as e:  # noqa: BLE001
                    out(f"      (M={M} {arm}: {type(e).__name__}: {e})")
            if not t:
                continue
            win = min(t, key=t.get)
            second = sorted(t.values())[1] if len(t) > 1 else t[win]
            margin = second / t[win]
            cells = "".join(f"{t.get(a, float('nan')):>22.2f}" for a in arms)
            out(f"    {M:>4} {cells}   {win:>22} {margin:>7.2f}x")
            for a in arms:
                if a in t:
                    rows.append((name, K, N, g, str(dt).split(".")[-1], M, a, t[a]))
            del x
        if bitcheck:
            # Bit-identity across the WMMA arms, at a mid-band M and a boundary M.
            for M in (min(MS), max(MS)):
                x = (torch.randn(M, K, device=DEV) * 0.3).to(dt)
                ref = None
                deltas = []
                for arm in arms:
                    try:
                        y = call(W, arm, x, ws[0]).float()
                    except Exception:  # noqa: BLE001
                        continue
                    if ref is None:
                        ref, refname = y, arm
                    else:
                        deltas.append(f"{arm}-vs-{refname}: {(y - ref).abs().max().item():.3e}")
                if deltas:
                    out(f"    bit-identity M={M}: " + "   ".join(deltas))
                del x
        del ws
        torch.cuda.empty_cache()
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--csv", default="")
    ap.add_argument("--only", default="all", choices=["all", "grid", "real"])
    ap.add_argument("--ashuffle", action="store_true", help="also time prefill_wmma_ashuffle")
    ap.add_argument("--smoke", action="store_true", help="2 shapes x 2 M -- wiring check only")
    args = ap.parse_args()
    fh = open(args.out, "w") if args.out else None

    def out(s=""):
        print(s, flush=True)
        if fh:
            fh.write(s + "\n")
            fh.flush()

    import fp8_wmma as W

    torch.manual_seed(0)
    arms = ARMS if args.ashuffle else ARMS[:2]
    out(f"device: {torch.cuda.get_device_name(0)}   fp8_wmma: {W.__file__}")
    out(f"arms: {arms}")
    out("graph-replay timed; rotation sized in BYTES to exceed the 64 MB MALL; us per call")

    MS = MID_M + BOUNDARY_M
    real, gk, gn = REAL_SHAPES, GRID_K, GRID_N
    if args.smoke:
        MS = [20, 64]
        real = [REAL_SHAPES[1], REAL_SHAPES[15]]
        gk, gn = [2048], [1024, 16384]
    rows = []
    if args.only in ("all", "real"):
        rows += sweep(W, out, "REAL SHIPPED SHAPES", real, MS, arms)
    if args.only in ("all", "grid"):
        grid = [
            (f"grid K={K} N={N}", K, N, 32, torch.bfloat16, False) for K in gk for N in gn
        ]
        rows += sweep(W, out, "SYNTHETIC (M, N) SURFACE  (bf16, g=32, sym)", grid, MS, arms)

    if args.csv:
        with open(args.csv, "w") as f:
            f.write("name,K,N,g,dtype,M,arm,us\n")
            for r in rows:
                f.write(",".join(str(v) for v in r) + "\n")
    out("\ndone.")
    if fh:
        fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
