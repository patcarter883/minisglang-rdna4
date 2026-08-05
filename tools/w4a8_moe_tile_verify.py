#!/usr/bin/env python
"""Gate the grouped-MoE tile chooser: what it picks, what it costs, and that it changes no bytes.

Four things, in this order, and the order is load-bearing:

  0. CONTROL. The SAME build, recorded twice, per shape. The gemm2 decode arm is an fp32 atomic
     SCATTER and is order-nondeterministic AGAINST ITSELF, so a bit-identity gate on it fails for
     reasons that have nothing to do with the tile. Measure that floor FIRST and print it; every
     later delta is read against it, under an `ATOMIC.` key, instead of against 0.

  1. PROVENANCE. What block_m/BN the chooser picks (host-side, fp8_wmma.moe_tile_choose, no GPU),
     versus the rows-per-expert heuristic it replaces. A chooser that agrees with the old rule
     everywhere has changed nothing; a chooser that disagrees needs (2).

  2. COST. Graph-replay device timing (a per-call synchronize() has a ~40 us wall floor on this box),
     expert weight stacks rotated past the 64 MB MALL BY BYTE COUNT: chooser vs the prior heuristic.

  3. BIT-IDENTITY. The tile is a LAUNCH parameter, not a numerics one -- that is the entire licence
     for changing it. gemm1 and the M>2 gather-reduce gemm2 must be max|delta| = 0.000e+00 across
     tiles; the M<=2 scatter is gated by TOLERANCE against the control floor from step 0.

    gpu-lease -n 1 --timeout 5400 -- \
      TILE_TOOL=tools/w4a8_moe_tile_verify.py bash tools/w4a8_dense_tile_surface_run.sh \
        --out /engine/_moe_tile_verify.txt
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from w4a8_moe_tile_surface import DEV, SHAPES, rotation, time_graph  # noqa: E402

MS = [1, 2, 8, 32, 64, 256, 1024]
BITID_MS = [1, 2, 3, 17, 33, 64, 129]


def prev_block_m(M: int, E: int, top_k: int) -> int:
    """The rows-per-expert rule the chooser replaces -- kept here as the baseline arm."""
    rpe = (M * top_k) / max(E, 1)
    bm = 16
    for c in (16, 32, 64, 128):
        if c <= rpe:
            bm = c
    return bm


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
    from minisgl.quant.kernels import w4a8_moe

    torch.manual_seed(0)
    p = torch.cuda.get_device_properties(0)
    out(f"device: {p.name}   WGPs={p.multi_processor_count} -> {2 * p.multi_processor_count} CUs")
    out(f"fp8_wmma: {W.__file__}")
    out("us per call; graph-replay timed; rotation sized in BYTES past the 64 MB MALL\n")

    tot_new = tot_old = 0.0
    for name, E, top_k, hidden, inter, g in SHAPES:
        n1, k1 = 2 * inter, hidden
        n2, k2 = hidden, inter
        out(f"=== {name}  E={E} top_k={top_k} hidden={hidden} inter={inter} g={g} ===")
        out(f"    gemm1 N={n1} K={k1}   gemm2 N={n2} K={k2}")
        # THREE arms, because the change has two separable halves and they must be attributed
        # separately: BN comes from the chooser INSIDE the launcher (given whatever block_m it was
        # handed), block_m comes from the chooser in the engine BEFORE moe_align. `prev+chooserBN`
        # isolates the first; `chooser` is both.
        out(f"    {'M':>6}{'chooser':>16}{'prev':>10}{'chooser us':>12}{'prev us':>10}"
            f"{'prev+chBN':>11}{'gain':>8}{'BN only':>9}")
        for M in MS:
            ws, R, wb = rotation(E, hidden, inter, g, M, top_k)
            x = torch.randn(M, hidden, device=DEV, dtype=torch.bfloat16) * 0.05
            gate = torch.randn(M, E, device=DEV, dtype=torch.bfloat16)
            bm, bn1, bn2 = W.moe_tile_choose(M * top_k, E, n1, k1, g, n2, k2, g)
            pbm = prev_block_m(M, E, top_k)

            def run(w, block_m=None):
                return w4a8_moe(
                    x, w[0], w[1], None, w[2], w[3], None, gate, top_k, True,
                    kernel="wmma", block_m=block_m,
                )

            os.environ.pop("VLLM_W4A8_MOE_BN", None)
            t_new, _ = time_graph(lambda w: run(w), ws)          # chooser: block_m AND BN
            os.environ["VLLM_W4A8_MOE_BN"] = "64"                # the prior constant
            t_old, _ = time_graph(lambda w: run(w, pbm), ws)     # prior heuristic block_m + BN=64
            os.environ.pop("VLLM_W4A8_MOE_BN", None)
            t_bn, _ = time_graph(lambda w: run(w, pbm), ws)      # prior block_m, chooser's BN
            tot_new += t_new
            tot_old += t_old
            out(f"    {M:>6}{f'{bm}x{bn1}/{bn2}':>16}{f'{pbm}x64':>10}"
                f"{t_new:>12.2f}{t_old:>10.2f}{t_bn:>11.2f}"
                f"{t_old / t_new:>7.2f}x{t_old / t_bn:>8.2f}x")
            del x, gate, ws
            torch.cuda.empty_cache()

        # ---- step 0 + 3: the control floor, then the tile delta, on the same key -----------------
        ws, R, _ = rotation(E, hidden, inter, g, max(BITID_MS), top_k)
        w = ws[0]
        line_ctl, line_tile = [], []
        for M in BITID_MS:
            x = torch.randn(M, hidden, device=DEV, dtype=torch.bfloat16) * 0.05
            gate = torch.randn(M, E, device=DEV, dtype=torch.bfloat16)

            def one(block_m=None, bn=None):
                if bn:
                    os.environ["VLLM_W4A8_MOE_BN"] = str(bn)
                else:
                    os.environ.pop("VLLM_W4A8_MOE_BN", None)
                r = w4a8_moe(x, w[0], w[1], None, w[2], w[3], None, gate, top_k, True,
                             kernel="wmma", block_m=block_m)
                os.environ.pop("VLLM_W4A8_MOE_BN", None)
                return r

            a = one()
            ctl = (one() - a).abs().max().item()           # SAME config twice: the floor
            b = one(prev_block_m(M, E, top_k), 64)         # the tile actually changed
            d = (b - a).abs().max().item()
            atomic = M <= 2                                # the scatter arm
            key = "ATOMIC." if atomic else ""
            line_ctl.append(f"M={M}:{key}{ctl:.3e}")
            line_tile.append(f"M={M}:{key}{d:.3e}")
            if not atomic and d != 0.0:
                out(f"    !! BIT-IDENTITY FAILED at M={M}: max|delta| = {d:.3e} "
                    f"(control floor {ctl:.3e}) -- the tile changed the BYTES, not just the launch")
            del x, gate
        out("    control (same build, recorded twice): " + "  ".join(line_ctl))
        out("    chooser tile vs prev tile:            " + "  ".join(line_tile))
        del ws
        torch.cuda.empty_cache()
        out()

    out(f"TOTAL over the measured cells: chooser {tot_new:.0f} us vs prev {tot_old:.0f} us "
        f"= {tot_old / tot_new:.3f}x")
    out("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
