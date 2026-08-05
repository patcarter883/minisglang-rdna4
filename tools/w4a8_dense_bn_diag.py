#!/usr/bin/env python
"""WHY is 32x128x4 four times 32x64x4 at lm_head M=1, when every term the model has says wash?

    gpu-lease -n 2 --timeout 5400 -- env TILE_EXCLUSIVE=1 \
        TILE_TOOL=tools/w4a8_dense_bn_diag.py TILE_WT=<worktree> TILE_KERNELS=<kernels wt> \
        bash tools/w4a8_dense_tile_surface_run.sh --out ... --csv ...

The cost model prices tile choice as ROUNDS * K/bk * THREADS * per_thread * (1 + LAT/OCC). For
32x64x4 vs 32x128x4 at M=1 every one of those is identical or exactly compensating: same BM, same
WARPS_N, same NWARPS=8, same blocks/CU, same OCC, and doubling BN halves ROUNDS while doubling the
per-thread B stage. The model puts them 4% apart. The card puts them 4x apart. So the separating
mechanism is in a stage the model does not price at all, and this tool finds WHICH.

THREE ORTHOGONAL EXPERIMENTS, all on the same held lease:

  STAGE  -- the kernel already carries VLLM_W4A8_V7_DIAG, a bitmask that ablates one stage at a
            time (1 skip WMMA, 2 skip global loads, 4 skip LDS writes, 8 skip LDS operand reloads,
            16 skip the per-group w_scale read). Whichever bit COLLAPSES the 64-vs-128 gap names
            the stage. This is attribution by ablation, not by hypothesis: no bit collapsing it
            would itself falsify every stage-local story.

  NSWEEP -- hold K and g, sweep N over a decade. A term in BN alone must show the gap at every N;
            a term that needs the grid to cover the machine many times over shows it only at large
            N. The one cell in the whole 300-cell surface that carries the 4x is the one with
            N=131072, so this decides whether "BN" or "BN x grid depth" is the real variable.

  KSWEEP -- hold N and BN, sweep K at fixed g. K/g is the k-loop trip count AND the stride of the
            w_scales array (K/g halves per weight row). If the effect tracks K/g rather than g, the
            mechanism is the SCALE array's line reuse, not the weight array's.

Timing conventions are the surface tool's, imported rather than re-implemented: graph replay,
rotation sized in BYTES past the 64 MB MALL, reps derived from a per-cell budget, card selected by
PROPERTIES (64 CU), provenance stamped per row through csv.writer.
"""
from __future__ import annotations

import argparse
import csv as csvmod
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import w4a8_dense_tile_surface as S  # noqa: E402  (timing + rotation + provenance, one source)

DIAG_BITS = [
    (0, "full"),
    (1, "no_wmma"),
    (2, "no_global"),
    (4, "no_ldswrite"),
    (8, "no_ldsread"),
    (16, "no_wscale"),
    (18, "no_global+no_wscale"),
]


def timed(W, tile, x, e2m1, ws, diag, swiz=1):
    os.environ["VLLM_W4A8_V7_CFG"] = tile
    os.environ["VLLM_W4A8_V7_DIAG"] = str(diag)
    os.environ["VLLM_W4A8_V7_SWIZ"] = str(swiz)
    fn = lambda w: W.mmq_fp8_gemm(x, w[0], w[1], kernel="wmma_tiled_tuned",
                                  w_zeros=w[2], weight_is_e2m1=e2m1)
    try:
        return S.time_graph(fn, ws)
    except Exception as exc:                      # a refused tile is INFORMATION, not a skip
        return None, f"{type(exc).__name__}: {str(exc)[:90]}"
    finally:
        for k in ("VLLM_W4A8_V7_CFG", "VLLM_W4A8_V7_DIAG", "VLLM_W4A8_V7_SWIZ"):
            os.environ.pop(k, None)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--csv", default="")
    ap.add_argument("--tiles", default="32x64x4,32x128x4",
                    help="comma list of BMxBNxWN to contrast")
    ap.add_argument("--exp", default="stage,nsweep,ksweep")
    ap.add_argument("--ms", default="1,32,48")
    ap.add_argument("--ns", default="4096,8192,16384,32768,65536,131072")
    ap.add_argument("--ks", default="1024,2048,2816,4096,8192")
    ap.add_argument("--gs", default="32,128")
    ap.add_argument("--base-k", type=int, default=2816)
    ap.add_argument("--base-n", type=int, default=131072)
    ap.add_argument("--swiz", default="1,0",
                    help="PRODUCTION DEFAULT IS 1. Forcing 0 measures a launch the\n                          engine never makes, and at ceil(M/BM)>1 the two differ by up\n                          to 3.9x -- an earlier revision of this tool defaulted to 0 and\n                          produced a fixture-disagreement that was the TOOL, not the data.")
    args = ap.parse_args()

    fh = open(args.out, "w") if args.out else None
    ch = open(args.csv, "w", newline="") if args.csv else None
    cw = csvmod.writer(ch) if ch else None

    def out(s=""):
        print(s, flush=True)
        if fh:
            fh.write(s + "\n")
            fh.flush()

    def row(fields):
        if cw:
            cw.writerow(fields)
            ch.flush()

    sys.path.insert(0, "/engine/python")
    import fp8_wmma as W

    # Card selection BY PROPERTIES -- the chooser ships a 64-CU pin, so a 56-CU card measures a
    # knowingly mis-tiled kernel. Never by ordinal: under a two-card lease the ordinals are both live.
    want = None
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        n = p.multi_processor_count * 2
        out(f"  visible cuda:{i} = {p.name}  CUs={n}")
        if n == 64 and want is None:
            want = i
    if want is None:
        out("REFUSING TO TIME: no 64-CU card visible. Re-lease until card 0 is held.")
        return 2
    torch.cuda.set_device(want)
    S.DEV = torch.device(f"cuda:{want}")
    dev_name = torch.cuda.get_device_name(want)
    cu = torch.cuda.get_device_properties(want).multi_processor_count * 2
    card, lease = S.provenance_cards(want)
    excl = os.environ.get("TILE_EXCLUSIVE", "0")
    out(f"device: {dev_name}   CUs={cu}   physical card={card}   lease={lease}   "
        f"exclusive_box={excl}   fp8_wmma: {W.__file__}")
    out(f"tiles: {args.tiles}   experiments: {args.exp}\n")
    row(["exp", "K", "N", "g", "M", "tile", "diag", "diagname", "swiz", "us", "reps", "R",
         "dev", "cu", "card", "lease", "exclusive"])

    tiles = [t for t in args.tiles.split(",") if t]
    exps = set(args.exp.split(","))
    DT, E2M1, ZEROS = torch.float16, False, False

    def run(exp, K, N, g, M, diaglist, swizlist=(1,)):
        ws, R, wb = S.rotation(N, K, g, ZEROS, M)
        tot = R * wb / 1e6
        out(f"--- {exp}  K={K} N={N} g={g} M={M}   rotation {R} x {wb/1e6:.1f} MB = {tot:.0f} MB"
            f"{'' if tot >= 64 else '   <-- DID NOT BUST THE MALL'}")
        x = (torch.randn(M, K, device=S.DEV) * 0.3).to(DT)
        for swiz in swizlist:
            for diag, dname in diaglist:
                line, base = [], None
                for t in tiles:
                    bm, bn, wn = S.tile3([int(v) for v in t.split("x")])
                    if not S.legal(bm, bn, g, wn):
                        line.append(f"{t}=ILLEGAL")
                        continue
                    us, reps = timed(W, t, x, E2M1, ws, diag, swiz)
                    if us is None:
                        line.append(f"{t}=REFUSED({reps})")
                        continue
                    row([exp, K, N, g, M, t, diag, dname, swiz, f"{us:.3f}", reps, R,
                         dev_name, cu, card, lease, excl])
                    line.append(f"{t}={us:8.1f}")
                    if base is None:
                        base = us
                    elif base:
                        line[-1] += f" ({us/base:.2f}x)"
                sw = f" swiz={swiz}" if len(swizlist) > 1 else ""
                out(f"    diag={diag:<3}{dname:<20}{sw}  " + "   ".join(line))
        del x, ws
        torch.cuda.empty_cache()

    if "stage" in exps:
        out("=" * 100)
        out("STAGE ABLATION -- which stage carries the gap? (VLLM_W4A8_V7_DIAG)")
        out("=" * 100)
        for g in [int(v) for v in args.gs.split(",")]:
            for M in [int(v) for v in args.ms.split(",")]:
                run("stage", args.base_k, args.base_n, g, M, DIAG_BITS)

    if "cells" in exps:
        out("=" * 100)
        out("CELL RE-TIME -- one shape, the tiles that matter, diag off. CONTROL FIRST: the same")
        out("build recorded twice, so a later delta has a noise floor to clear.")
        out("=" * 100)
        for g in [int(v) for v in args.gs.split(",")]:
            for M in [int(v) for v in args.ms.split(",")]:
                run("cellsA", args.base_k, args.base_n, g, M, [(0, "full")])
                run("cellsB", args.base_k, args.base_n, g, M, [(0, "full")])

    if "swiz" in exps:
        out("=" * 100)
        out("SWIZZLE -- does putting M on the fast grid axis move it?")
        out("=" * 100)
        for M in [int(v) for v in args.ms.split(",")]:
            run("swiz", args.base_k, args.base_n, 32, M, [(0, "full")],
                swizlist=[int(v) for v in args.swiz.split(",")])

    if "nsweep" in exps:
        out("=" * 100)
        out("N SWEEP -- is the variable BN, or BN x how deeply the grid covers the machine?")
        out("=" * 100)
        for g in [int(v) for v in args.gs.split(",")]:
            for N in [int(v) for v in args.ns.split(",")]:
                run("nsweep", args.base_k, N, g, 1, [(0, "full")])

    if "ksweep" in exps:
        out("=" * 100)
        out("K SWEEP at fixed g -- does the gap track K/g (the w_scales row stride) or g alone?")
        out("=" * 100)
        for g in [int(v) for v in args.gs.split(",")]:
            for K in [int(v) for v in args.ks.split(",")]:
                if K % g or K % 8:
                    continue
                run("ksweep", K, args.base_n, g, 1, [(0, "full")])

    out("\ndone")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
