#!/usr/bin/env python
"""Does the optimal tile differ between the two physical cards, and does the PINNED CU cost anything?

This box is a mismatched pair — RX 9070 XT (64 CUs) and RX 9070 (56 CUs) — and the arbiter leases
whichever is free, so a per-device dispatch rule makes the same job dispatch differently run to run.
The chooser therefore reasons about a PINNED 64 on both ranks (mirroring minisgl's
`_PINNED_DISPATCH_CU`). That is a deliberate trade and this tool prices it:

  * the chooser's tile at the pinned CU=64  (what ships)
  * the chooser's tile at the card's TRUE CU count (VLLM_W4A8_TILE_CU), i.e. what a per-device rule
    would have picked
  * the per-cell oracle over the measured tile set, on THIS card

If column 2 beats column 1 by more than the run-to-run spread, pinning costs something real on this
card and that number belongs in the record. If it does not, the pin is free and the reproducibility
argument wins outright.

    gpu-lease -n 2 --timeout 3600 -- bash -c 'for c in 0 1; do \
      ROCR_VISIBLE_DEVICES=$c TILE_TOOL=tools/w4a8_dense_tile_crosscard.py \
      bash tools/w4a8_dense_tile_surface_run.sh --out /engine/_tile_crosscard_$c.txt; done'
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from w4a8_dense_tile_surface import DEV, TILES_EXT, TILES_SHIPPED, rotation, time_graph  # noqa: E402

SHAPES = [
    ("g4.q_proj    tp2", 2816, 2048, 32, torch.float16, False),
    ("q27.q_proj   tp1", 5120, 6144, 32, torch.bfloat16, True),
    ("lag.gate_up  tp2", 2048, 8192, 32, torch.bfloat16, True),
    ("glm.gate_up  tp2", 2048, 10240, 128, torch.bfloat16, True),
    ("lag.gate_up  tp1", 2048, 16384, 32, torch.bfloat16, True),
]
MS = [1, 17, 32, 64, 128, 512]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    fh = open(args.out, "w") if args.out else None

    def out(s=""):
        print(s, flush=True)
        if fh:
            fh.write(s + "\n")
            fh.flush()

    import fp8_wmma as W

    torch.manual_seed(0)
    p = torch.cuda.get_device_properties(0)
    true_cu = 2 * p.multi_processor_count
    out(f"device: {p.name}   WGPs={p.multi_processor_count} -> TRUE CU={true_cu}   pinned CU=64")
    out(f"fp8_wmma: {W.__file__}")
    out("us per call; graph-replay timed; rotation sized in BYTES past the 64 MB MALL\n")

    tiles = [f"{a}x{b}" for a, b in sorted(set(TILES_SHIPPED + TILES_EXT))]
    tot_pin = tot_true = tot_orc = 0.0
    diff_cells = 0
    for name, K, N, g, dt, zeros in SHAPES:
        out(f"=== {name}  K={K} N={N} g={g} ===")
        out(f"    {'M':>6}{'pinned64':>10}{'true'+str(true_cu):>10}{'oracle':>10}"
            f"{'pin us':>10}{'true us':>10}{'oracle us':>11}{'pin/true':>10}{'pin/orc':>9}")
        for M in MS:
            ws, R, wb = rotation(N, K, g, zeros, M)
            x = (torch.randn(M, K, device=DEV) * 0.3).to(dt)

            def run(cfg=None):
                if cfg:
                    os.environ["VLLM_W4A8_V7_CFG"] = cfg
                else:
                    os.environ.pop("VLLM_W4A8_V7_CFG", None)
                t, _ = time_graph(
                    lambda w: W.mmq_fp8_gemm(
                        x, w[0], w[1], kernel="wmma_tiled_tuned", w_zeros=w[2],
                        weight_is_e2m1=False,
                    ),
                    ws,
                )
                os.environ.pop("VLLM_W4A8_V7_CFG", None)
                return t

            pin = W.dense_tile_explain(M, N, K, g, 0).split("tile        = ")[1].split()[0]
            tru = W.dense_tile_explain(M, N, K, g, true_cu).split("tile        = ")[1].split()[0]
            t_pin = run()
            t_tru = run(tru) if tru != pin else t_pin
            best_t, best_c = None, None
            for c in tiles:
                try:
                    v = run(c)
                except Exception:  # noqa: BLE001
                    torch.cuda.synchronize()
                    continue
                if best_t is None or v < best_t:
                    best_t, best_c = v, c
            diff_cells += tru != pin
            tot_pin += t_pin
            tot_true += t_tru
            tot_orc += best_t
            out(f"    {M:>6}{pin:>10}{tru:>10}{best_c:>10}{t_pin:>10.2f}{t_tru:>10.2f}"
                f"{best_t:>11.2f}{t_pin/t_tru:>9.3f}x{t_pin/best_t:>8.3f}x")
            del x, ws
            torch.cuda.empty_cache()
        out("")

    out(f"cells where the TRUE-CU rule would pick a different tile: {diff_cells}/"
        f"{len(SHAPES)*len(MS)}")
    out(f"total: pinned {tot_pin:.0f} us   true-CU {tot_true:.0f} us   oracle {tot_orc:.0f} us")
    out(f"  cost of PINNING the CU count on this card: {tot_pin/tot_true:.4f}x")
    out(f"  chooser (pinned) vs this card's per-cell oracle: {tot_pin/tot_orc:.4f}x")
    out("\ndone.")
    if fh:
        fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
