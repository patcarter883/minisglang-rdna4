#!/usr/bin/env python
"""What tile should `wmma_tiled_tuned` run? Sweep the whole (BM x BN) space it supports.

`wmma_tiled_tuned` is templated on its tile with dynamic LDS (NWARPS = BM/16), and the launcher
picks the tile from `VLLM_W4A8_V7_CFG` with a HARD-WIRED default of 256x128 -- a large-M choice
("BM=256 amortizes staging") applied to every M the engine dispatches. `prefill_wmma`, the other
arm, already derives a small-M tile (BM=64 below M=128). The mid-band sweep measured the cost of
that asymmetry at up to 2.06x on a single cell, which is larger than the whole three-way ARM
dispatch it motivated.

So this sweeps the tile, not the arm:

  * every instantiated (BM, BN) that fits LDS at the shape's group size --
    LDS = (BM+BN)*(group_size+8) bytes, so a tile legal at g=32 can be illegal at g=128;
  * against BOTH prefill arms at their shipped defaults;
  * and against `prefill_wmma` with VLLM_W4A8_DENSE_SMALLM_OFF=1, which removes the ONLY thing
    that arm has and the tiled kernel lacks. If prefill_wmma's mid-band win disappears when its
    small-M tile is switched off, the win was never algorithmic and the arm can be retired.

TIMING -- the two traps this repo has already paid for, plus one this sweep adds:
  * per-call `synchronize()` has a ~40 us wall floor on this box, so every number is a CUDA-graph
    replay of R back-to-back calls, event-bracketed, / R.
  * the R weight copies are sized in BYTES to exceed the 64 MB MALL, and the achieved MB is printed.
  * the M range here spans 1..2048, i.e. ~4 orders of magnitude in work per call. A fixed rep count
    either wastes an hour on M=2048 or measures noise at M=17, so the rep count is DERIVED from a
    per-cell time budget after a cheap eager estimate. `reps` is recorded in the CSV so a cell timed
    at the floor is visible rather than silent.

    gpu-lease -n 1 --timeout 10800 -- bash tools/w4a8_dense_tile_surface_run.sh \
        --out /engine/_tile_surface.txt --csv /engine/_tile_surface.csv
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

DEV = torch.device("cuda:0")
ROT_TARGET_BYTES = 160 << 20
ROT_MAX_COPIES = 160
ROT_MAX_VRAM = 5 << 30  # weights + per-call outputs held live in the graph pool
CELL_BUDGET_US = 20_000.0  # GPU time per timed cell, before overhead
LDS_MAX = 65536

# Instantiated in the kernel BEFORE this work (w4a8_fp8_wmma_kernel.hip, DK::WmmaTiledTuned).
TILES_SHIPPED = [
    (64, 64), (64, 128), (80, 128), (96, 128), (112, 128), (128, 64),
    (128, 128), (128, 256), (192, 128), (256, 64), (256, 128), (256, 192),
]
# Added by this work. The shipped set's optimum sat ON its small corner (64x64 / 256x64 won nearly
# every measured mid-band cell), which means the set was fencing the answer, not containing it.
# These extend both edges: BM below 64 (a BM=256 tile at M=17 runs 16 warps of WMMA for 2 warps of
# real rows) and BN below 64 (BN sets ceil(N/BN) = the dispatch granularity; at N=2048 a BN=128 grid
# is 16 workgroups on a 64-CU card).
TILES_EXT = [
    (16, 64), (16, 128), (32, 32), (32, 64), (32, 128), (64, 32),
    (128, 32), (256, 32), (256, 256), (384, 128), (512, 64), (512, 128),
]

# (name, K, N, group, dtype, awq_zeros). Every quantized dense linear the engine dispatches, at TP=1
# and TP=2 per-rank, spanning N from 2048 to the LM head's 131072 and K from 2048 to 8704.
SHAPES = [
    ("g4.q_proj    tp2", 2816, 2048, 32, torch.float16, False),
    ("g4.o_proj    tp2", 2048, 2816, 32, torch.float16, False),
    ("g4.gate_up   tp1", 2816, 4224, 32, torch.float16, False),
    ("q35.q_proj   tp1", 2048, 4096, 32, torch.bfloat16, True),
    ("q27.down     tp2", 8704, 5120, 32, torch.bfloat16, True),
    ("q27.q_proj   tp1", 5120, 6144, 32, torch.bfloat16, True),
    ("lag.gate_up  tp2", 2048, 8192, 32, torch.bfloat16, True),
    ("q35b4.gate_up tp2", 2560, 9216, 32, torch.bfloat16, True),
    ("glm.gate_up  tp2", 2048, 10240, 128, torch.bfloat16, True),
    ("grid N=11264", 2816, 11264, 32, torch.bfloat16, False),
    ("lag.gate_up  tp1", 2048, 16384, 32, torch.bfloat16, True),
    ("q27.gate_up  tp2", 5120, 17408, 32, torch.bfloat16, True),
    ("q27.gate_up  tp1", 5120, 34816, 32, torch.bfloat16, True),
    ("lm_head      tp2", 2816, 131072, 32, torch.float16, False),
]
MS = [1, 8, 17, 24, 32, 48, 63, 64, 96, 128, 192, 256, 512, 1024, 2048]

ARMS = ["prefill_wmma", "prefill_wmma_ashuffle", "prefill_wmma:smallm_off"]


def pack_uint4_2d(w: torch.Tensor) -> torch.Tensor:
    N, K = w.shape
    w = w.to(torch.int32)
    packed = torch.zeros((N, K // 8), dtype=torch.int32, device=w.device)
    for i in range(8):
        packed |= (w[:, i::8] & 0xF) << (i * 4)
    return packed


def rotation(N: int, K: int, g: int, zeros: bool, M: int, outbytes: int = 2):
    """R distinct weight sets whose TOTAL bytes exceed ROT_TARGET_BYTES (sized in BYTES).

    Each graphed call also owns an output (M, N), so the rotation is additionally capped so
    R*(weight + output) stays inside ROT_MAX_VRAM -- at M=2048, N=131072 one output alone is 0.5 GB.
    """
    wbytes = N * (K // 8) * 4 + N * (K // g) * 2 + (((N // 8) * (K // g) * 4) if zeros else 0)
    percall = wbytes + M * N * outbytes
    R = max(2, min(ROT_MAX_COPIES, -(-ROT_TARGET_BYTES // wbytes) + 1))
    R = max(2, min(R, ROT_MAX_VRAM // percall))
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


def _eager_us(fn, ws) -> float:
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for w in ws:
        fn(w)
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) * 1e3 / len(ws)


def time_graph(fn, ws, budget_us: float = CELL_BUDGET_US, max_reps: int = 20, min_reps: int = 2):
    """Capture one replay of len(ws) calls; return (us per call, reps). reps derived from a budget."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            for w in ws:
                fn(w)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    est = max(_eager_us(fn, ws), 1.0)
    reps = int(max(min_reps, min(max_reps, budget_us / (est * len(ws)))))

    graph = torch.cuda.CUDAGraph()
    pool = torch.cuda.graph_pool_handle()
    with torch.cuda.graph(graph, pool=pool):
        for w in ws:
            fn(w)
    torch.cuda.synchronize()
    for _ in range(2):
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
    torch.cuda.synchronize()
    return us, reps


def legal(bm: int, bn: int, g: int) -> bool:
    return (bm + bn) * (g + 8) <= LDS_MAX


def make_call(W, cand: str, x, e2m1: bool):
    """Return a fn(w) that runs `cand` (an arm name or a "BMxBN" tile) on weight set w.

    The kernel reads its tile / small-M switch from the environment at LAUNCH time, and a captured
    graph freezes whatever was launched, so the env must be set around capture -- which is what the
    caller does.
    """
    if cand in ("prefill_wmma", "prefill_wmma:smallm_off"):
        arm = "prefill_wmma"
    elif cand == "prefill_wmma_ashuffle":
        arm = "prefill_wmma_ashuffle"
    else:
        arm = "wmma_tiled_tuned"
    return lambda w: W.mmq_fp8_gemm(
        x, w[0], w[1], kernel=arm, w_zeros=w[2], weight_is_e2m1=e2m1
    )


def set_env(cand: str) -> None:
    os.environ.pop("VLLM_W4A8_V7_CFG", None)
    os.environ.pop("VLLM_W4A8_DENSE_SMALLM_OFF", None)
    if cand == "prefill_wmma:smallm_off":
        os.environ["VLLM_W4A8_DENSE_SMALLM_OFF"] = "1"
    elif cand not in ARMS:
        os.environ["VLLM_W4A8_V7_CFG"] = cand


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--csv", default="")
    ap.add_argument("--tiles", default="all", choices=["all", "shipped", "ext"])
    ap.add_argument("--ms", default="", help="comma list; default the full 1..2048 ladder")
    ap.add_argument("--shapes", default="", help="comma list of substrings to keep")
    ap.add_argument("--no-arms", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    fh = open(args.out, "w") if args.out else None

    def out(s=""):
        print(s, flush=True)
        if fh:
            fh.write(s + "\n")
            fh.flush()

    import fp8_wmma as W

    torch.manual_seed(0)
    tiles = {"all": TILES_SHIPPED + TILES_EXT, "shipped": TILES_SHIPPED, "ext": TILES_EXT}[
        args.tiles
    ]
    tiles = sorted(set(tiles))
    ms = [int(v) for v in args.ms.split(",")] if args.ms else list(MS)
    shapes = SHAPES
    if args.shapes:
        keys = [s.strip() for s in args.shapes.split(",")]
        shapes = [s for s in SHAPES if any(k in s[0] for k in keys)]
    if args.smoke:
        shapes, ms, tiles = shapes[:2], [17, 256], tiles[:4]

    out(f"device: {torch.cuda.get_device_name(0)}   fp8_wmma: {W.__file__}")
    out(f"tiles ({len(tiles)}): " + " ".join(f"{a}x{b}" for a, b in tiles))
    out(f"arms: {'(skipped)' if args.no_arms else ARMS}")
    out("graph-replay timed; rotation sized in BYTES past the 64 MB MALL; us per call\n")

    cf = open(args.csv, "w") if args.csv else None
    if cf:
        cf.write("name,K,N,g,dtype,M,cand,us,reps,R\n")

    for name, K, N, g, dt, zeros in shapes:
        legal_tiles = [f"{a}x{b}" for a, b in tiles if legal(a, b, g)]
        skipped = [f"{a}x{b}" for a, b in tiles if not legal(a, b, g)]
        cands = ([] if args.no_arms else list(ARMS)) + legal_tiles
        out(f"=== {name}  K={K} N={N} g={g} {str(dt).split('.')[-1]} "
            f"zeros={'awq' if zeros else 'sym'} ===")
        if skipped:
            out(f"    LDS-illegal at g={g} (>{LDS_MAX} B): " + " ".join(skipped))
        prev_M = None
        ws = R = wbytes = None
        for M in ms:
            # rotation depends on M (output bytes), so rebuild when the VRAM class changes
            if ws is None or prev_M is None or M * N * 2 > 64 << 20:
                del ws
                ws = None
                torch.cuda.empty_cache()
                ws, R, wbytes = rotation(N, K, g, zeros, M)
                tot = R * wbytes / 1e6
                flag = "" if tot >= 64.0 else "  <-- ROTATION DID NOT BUST THE MALL"
                out(f"    [M={M}] rotation {R} x {wbytes/1e6:.2f} MB = {tot:.0f} MB{flag}")
            prev_M = M
            x = (torch.randn(M, K, device=DEV) * 0.3).to(dt)
            t, rp = {}, {}
            for cand in cands:
                set_env(cand)
                try:
                    t[cand], rp[cand] = time_graph(make_call(W, cand, x, False), ws)
                except Exception as e:  # noqa: BLE001
                    out(f"      (M={M} {cand}: {type(e).__name__}: {str(e)[:90]})")
                    torch.cuda.synchronize()
                set_env("prefill_wmma")  # clear
            if not t:
                continue
            order = sorted(t, key=t.get)
            best = order[0]
            line = f"    M={M:>5}  best {best:>9} {t[best]:>9.2f} us"
            if len(order) > 1:
                line += f"   (2nd {order[1]:>9} {t[order[1]]:>9.2f}, {t[order[1]]/t[best]:.2f}x)"
            worst_tile = max((c for c in t if c not in ARMS), key=t.get, default=None)
            if worst_tile:
                line += f"   worst tile {worst_tile} {t[worst_tile]:.2f} ({t[worst_tile]/t[best]:.2f}x)"
            out(line)
            out("        " + "  ".join(f"{c}={t[c]:.1f}" for c in order))
            if cf:
                for c in order:
                    cf.write(
                        f"{name},{K},{N},{g},{str(dt).split('.')[-1]},{M},{c},"
                        f"{t[c]:.3f},{rp[c]},{R}\n"
                    )
                cf.flush()
            del x
        del ws
        torch.cuda.empty_cache()
        out("")

    out("done.")
    if fh:
        fh.close()
    if cf:
        cf.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
