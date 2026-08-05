#!/usr/bin/env python
"""Validate the ANALYTIC tile chooser against the measured surface, and gate it on bit-identity.

Three things, in order:

  A. PROVENANCE + PREDICTION. For every shape x M: what tile does the kernel's own chooser pick
     (asked host-side through `fp8_wmma.dense_tile_explain`, no GPU), what did the hard-wired
     256x128 do, and what was the swept oracle over the 24 tiles the surface measured? The chooser
     ranges over a 77-tile LATTICE, so on ~26% of cells it picks a tile the surface never timed --
     those are exactly the cells this tool exists to time. A model that only agrees where it was
     fitted has not been validated.

  B. COST. Graph-replay timed, weights rotated past the 64 MB MALL by BYTE count:
     chooser tile vs the shipped 256x128 vs the previously-measured oracle tile.

  C. BIT-IDENTITY, through `w4a8_linear`'s auto-dispatch rather than the raw op. The tile is a
     launch parameter, not a numerics one, so every tile must produce the SAME bytes -- that is the
     whole licence for changing it. Checked at several M per shape.

    gpu-lease -n 1 --timeout 3600 -- \
      TILE_TOOL=tools/w4a8_dense_tile_verify.py bash tools/w4a8_dense_tile_surface_run.sh \
        --out /engine/_tile_verify.txt
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from w4a8_dense_tile_surface import DEV, rotation, time_graph  # noqa: E402

SHAPES = [
    ("g4.q_proj    tp2", 2816, 2048, 32, torch.float16, False),
    ("g4.o_proj    tp2", 2048, 2816, 32, torch.float16, False),
    ("g4.gate_up   tp1", 2816, 4224, 32, torch.float16, False),
    ("q27.down     tp2", 8704, 5120, 32, torch.bfloat16, True),
    ("q27.q_proj   tp1", 5120, 6144, 32, torch.bfloat16, True),
    ("lag.gate_up  tp2", 2048, 8192, 32, torch.bfloat16, True),
    ("q35b4.gate_up tp2", 2560, 9216, 32, torch.bfloat16, True),
    ("glm.gate_up  tp2", 2048, 10240, 128, torch.bfloat16, True),
    ("lag.gate_up  tp1", 2048, 16384, 32, torch.bfloat16, True),
    ("q27.gate_up  tp1", 5120, 34816, 32, torch.bfloat16, True),
    ("lm_head      tp2", 2816, 131072, 32, torch.float16, False),
]
MS = [1, 17, 32, 64, 128, 256, 512, 2048]
BITID_MS = [1, 17, 33, 64, 129, 512]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--oracle-csv", default="/engine/_tile_surface.csv")
    args = ap.parse_args()
    fh = open(args.out, "w") if args.out else None

    def out(s=""):
        print(s, flush=True)
        if fh:
            fh.write(s + "\n")
            fh.flush()

    import fp8_wmma as W

    from minisgl.quant.kernels import w4a8_linear

    # previously measured per-cell oracle over the 24-tile surface, for reference
    oracle = {}
    try:
        import csv as _csv

        best = {}
        for r in _csv.DictReader(open(args.oracle_csv)):
            if "prefill" in r["cand"]:
                continue
            k = (int(r["N"]), int(r["K"]), int(r["M"]))
            v = float(r["us"])
            if k not in best or v < best[k][1]:
                best[k] = (r["cand"], v)
        oracle = best
    except Exception as e:  # noqa: BLE001
        out(f"(no oracle csv: {e})")

    torch.manual_seed(0)
    p = torch.cuda.get_device_properties(0)
    out(f"device: {p.name}   WGPs={p.multi_processor_count} -> {2*p.multi_processor_count} CUs")
    out(f"fp8_wmma: {W.__file__}")
    out("us per call; graph-replay timed; rotation sized in BYTES past the 64 MB MALL\n")

    tot_new = tot_old = 0.0
    worst = (1.0, None)
    for name, K, N, g, dt, zeros in SHAPES:
        out(f"=== {name}  K={K} N={N} g={g} {str(dt).split('.')[-1]} ===")
        out(f"    {'M':>6}{'chooser':>10}{'shipped':>10}{'sweptOracle':>13}"
            f"{'chooser us':>12}{'256x128 us':>12}{'gain':>8}{'vs oracle':>11}")
        for M in MS:
            ws, R, wb = rotation(N, K, g, zeros, M)
            x = (torch.randn(M, K, device=DEV) * 0.3).to(dt)
            expl = W.dense_tile_explain(M, N, K, g)
            pick = expl.split("tile        = ")[1].split()[0]
            os.environ.pop("VLLM_W4A8_V7_CFG", None)
            t_new, _ = time_graph(
                lambda w: W.mmq_fp8_gemm(
                    x, w[0], w[1], kernel="wmma_tiled_tuned", w_zeros=w[2], weight_is_e2m1=False
                ),
                ws,
            )
            os.environ["VLLM_W4A8_V7_CFG"] = "256x128"
            t_old, _ = time_graph(
                lambda w: W.mmq_fp8_gemm(
                    x, w[0], w[1], kernel="wmma_tiled_tuned", w_zeros=w[2], weight_is_e2m1=False
                ),
                ws,
            )
            ob = oracle.get((N, K, M))
            t_orc = None
            if ob:
                os.environ["VLLM_W4A8_V7_CFG"] = ob[0]
                try:
                    t_orc, _ = time_graph(
                        lambda w: W.mmq_fp8_gemm(
                            x, w[0], w[1], kernel="wmma_tiled_tuned", w_zeros=w[2],
                            weight_is_e2m1=False,
                        ),
                        ws,
                    )
                except Exception:  # noqa: BLE001
                    torch.cuda.synchronize()
            os.environ.pop("VLLM_W4A8_V7_CFG", None)
            tot_new += t_new
            tot_old += t_old
            ratio = (t_new / t_orc) if t_orc else float("nan")
            if t_orc and t_new / t_orc > worst[0]:
                worst = (t_new / t_orc, f"{name} M={M} chooser {pick} vs oracle {ob[0]}")
            out(f"    {M:>6}{pick:>10}{'256x128':>10}{(ob[0] if ob else '-'):>13}"
                f"{t_new:>12.2f}{t_old:>12.2f}{t_old/t_new:>7.2f}x{ratio:>10.2f}x")
            del x, ws
            torch.cuda.empty_cache()

        # ---- bit-identity THROUGH w4a8_linear's auto-dispatch ----
        ws, R, wb = rotation(N, K, g, zeros, max(BITID_MS))
        deltas = []
        for M in BITID_MS:
            x = (torch.randn(M, K, device=DEV) * 0.3).to(dt)
            os.environ.pop("VLLM_W4A8_V7_CFG", None)
            a = w4a8_linear(x, ws[0][0], ws[0][1], ws[0][2], g).float()
            os.environ["VLLM_W4A8_V7_CFG"] = "256x128"
            b = w4a8_linear(x, ws[0][0], ws[0][1], ws[0][2], g).float()
            os.environ.pop("VLLM_W4A8_V7_CFG", None)
            deltas.append(f"M={M}:{(a - b).abs().max().item():.3e}")
            del x, a, b
        out("    bit-identity through w4a8_linear (chooser tile vs 256x128): " + "  ".join(deltas))
        del ws
        torch.cuda.empty_cache()
        out("")

    out(f"TOTAL over the measured cells: chooser {tot_new:.0f} us vs shipped 256x128 "
        f"{tot_old:.0f} us = {tot_old/tot_new:.3f}x")
    out(f"worst cell vs the swept oracle: {worst[0]:.2f}x  ({worst[1]})")
    out("\ndone.")
    if fh:
        fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
