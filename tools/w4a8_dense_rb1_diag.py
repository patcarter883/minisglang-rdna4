#!/usr/bin/env python
"""WHY is a dense w4a8 tile SLOWER at row_blocks==1 than at row_blocks==2, with HALF the work?

    gpu-lease -n 2 --timeout 5400 -- env TILE_EXCLUSIVE=1 \
        TILE_TOOL=tools/w4a8_dense_rb1_diag.py TILE_WT=<engine wt> TILE_KERNELS=<kernels wt> \
        bash tools/w4a8_dense_tile_surface_run.sh --out ... --csv ...

THE FACT, straight off the recorded card-0 surface (tools/_fixtures/dense_tile_surface_card0.csv),
`lag.gate_up tp1` (K=2048 N=16384 g=32 bf16 AWQ-zeros) at tile 32x64x4:

    M      1     8    17    24    32 |   48    63    64 |   96   128
    rb     1     1     1     1     1 |    2     2     2 |    3     4
    us   304   323   324   342   355 |  141   149   147 |  209   267
    us/rb 304   323   324   342   355 |   70    75    73 |   70    67

Every rb>=2 point costs ~70 us per row-block. The rb==1 points cost ~5x that. M=32 -> M=48 is
354.7 -> 140.6 us: 1.5x the work, 2.5x FASTER. The shipped cost model has NO term that can produce
a discontinuity there -- it prices cost as ROUNDS * ... with ROUNDS = ceil(row_blocks*n_blocks/CU),
so it predicts rb==2 costs TWICE rb==1. It is wrong by ~5x on that one cell, and that error is the
single largest thing standing between the chooser and the WARPS_N axis: widening WN_SET regresses
30 cells and SIX of the worst eight are this shape at rb==1.

WHAT IS ALREADY KNOWN, from tools/_fixtures/bn_diag/ -- do not re-derive:
  * At rb==1, swiz=0 and swiz=1 are the SAME (613.1 vs 624.9 us; 206.7 vs 212.6 us). They must be:
    the dense swizzle puts the ROW-BLOCK axis on grid.x, so at row_blocks==1 grid.x==1 and there is
    nothing to interleave -- BOTH orders are the un-swizzled order. This FALSIFIES a grid-dimension
    / shader-engine degeneracy at gridDim.x==1 (that story predicts the two orders differ).
  * At rb==2 the swizzle is worth a great deal: lm_head 32x128x4, 1180.7 us at swiz=1 vs 4637.5 at
    swiz=0, i.e. 3.9x. Turn the swizzle off and the rb==2 point returns to exactly 2x the rb==1
    point -- linear, no cliff. So the cliff IS the swizzle's reuse, present at rb>=2 and structurally
    unavailable at rb==1.
  * For lm_head that reuse is accounted: 32x128x4 retains 128 scale lines x 128 B = 16 KB per slab
    and 256 resident blocks x 16 KB = 4.19 MB just clears L2 (4 MB), while rb==2 halves the live
    slab count to 2.10 MB and fits. 32x64x4 is 8 KB/slab, fits at every rb, and shows NO cliff.
    That correction (count what is CONCURRENTLY live, not the shape's total) is real but INERT on
    the shape that actually regresses: lag.gate_up tp1 at BN=64 has a live scale set of only
    2.1 MB, so the scale-line term never fires there and cannot be the carrier. Hence this tool.

SO WHAT IS THE CARRIER ON lag.gate_up? The candidates that survive the above, and the experiment
that separates them:

  STAGE   -- VLLM_W4A8_V7_DIAG ablates one stage at a time (1 no WMMA, 2 no global loads, 4 no LDS
             writes, 8 no LDS operand reloads, 16 no per-group w_scale read). Whichever bit
             COLLAPSES the rb1-vs-rb2 ratio names the stage. This is the method that found the BN
             term; two hypothesis-first attempts (real-row fraction, occupancy re-pointing) were
             both falsified. Ablate first.
  BSLAB   -- if the carrier is the WEIGHT slab rather than its scales, the reuse is
             BN*K/2 bytes per n-block, not BN*128 -- 64 KB per slab here, 16.8 MB live at rb==1.
             `no_wscale` leaves that read ON and `no_global` kills BOTH, so the PAIR separates them:
             a cliff that survives diag=16 but dies at diag=2 is the weight data, not the scales.
  RBSWEEP -- sweep M continuously across the boundary. A cache/reuse story predicts the step lands
             exactly at M=BM+1 and that rb=3,4 are FLAT per row-block (reuse saturates at 2).
             A launch/occupancy story predicts a slope, not a step.
  NSWEEP  -- hold the tile and rb==1, sweep N. Reuse pressure scales with the number of resident
             DISTINCT slabs, which is capped by cu*blocks_per_cu, so a capacity story predicts the
             cliff APPEARS once n_blocks exceeds that cap and is absent below it.

CONVENTIONS. Timing/rotation/provenance are the surface tool's, imported not re-implemented. Two
traps this file exists to avoid, both of which have already produced a confident wrong verdict in
this campaign:
  * SHAPE FIDELITY. The shapes carry their real (dtype, awq_zeros) from the surface's own table --
    w4a8_dense_bn_diag.py hardcodes fp16-without-zeros, and `lag.gate_up tp1` is bf16 WITH AWQ
    zeros. Running the wrong one is a different kernel path.
  * THE FIXTURE GATE. Nothing is interpreted until the probe reproduces the RECORDED surface cells
    it is reasoning about. A probe that silently disagrees with the fixture is measuring its own
    configuration, not the card.
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

# (name, K, N, g, dtype, awq_zeros) -- COPIED from w4a8_dense_tile_surface.SHAPES so the cells this
# tool times are the cells the fixture recorded. `lag.gate_up tp1` is the shape that carries six of
# the eight worst WN_SET regressions; `lm_head tp2` is the one whose cliff the scale-line term
# already explains, kept as the POSITIVE CONTROL that the method reproduces a known answer.
SHAPES = {
    "lag":    ("lag.gate_up  tp1", 2048, 16384, 32, torch.bfloat16, True),
    "lmhead": ("lm_head      tp2", 2816, 131072, 32, torch.float16, False),
    "q27gu":  ("q27.gate_up  tp1", 5120, 34816, 32, torch.bfloat16, True),
}


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


def load_fixture(path):
    """{(name, M, cand): us} from the recorded card-0 surface, for the GATE."""
    ref = {}
    try:
        for r in csvmod.DictReader(open(path)):
            try:
                ref[(r["name"], int(r["M"]), r["cand"])] = float(r["us"])
            except (ValueError, KeyError):
                continue
    except OSError:
        return {}
    return ref


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--csv", default="")
    ap.add_argument("--shapes", default="lag,lmhead")
    ap.add_argument("--tiles", default="32x64x4,128x64",
                    help="the tile the chooser WANTS at rb==1 and the tile the oracle picks")
    ap.add_argument("--exp", default="gate,stage,rbsweep,nsweep")
    ap.add_argument("--fixture", default="/engine/tools/_fixtures/dense_tile_surface_card0.csv")
    ap.add_argument("--gate-tol", type=float, default=0.25,
                    help="fractional disagreement with the recorded fixture that ABORTS the run")
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

    # Card selection BY PROPERTIES. torch reports WGPs on RDNA, so a 64-CU card answers
    # multi_processor_count == 32; asserting 64 on that field REJECTS the correct card.
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
    row(["exp", "shape", "K", "N", "g", "dtype", "zeros", "M", "rb", "tile", "diag", "diagname",
         "swiz", "us", "reps", "R", "dev", "cu", "card", "lease", "exclusive"])

    ref = load_fixture(args.fixture)
    out(f"fixture: {args.fixture}  ({len(ref)} recorded cells)\n")
    tiles = [t for t in args.tiles.split(",") if t]
    exps = set(args.exp.split(","))
    shapes = [SHAPES[s] for s in args.shapes.split(",") if s in SHAPES]

    def run(exp, name, K, N, g, dt, zeros, M, tilelist, diaglist, swizlist=(1,), gate=False):
        ws, R, wb = S.rotation(N, K, g, zeros, M, outbytes=2)
        tot = R * wb / 1e6
        out(f"--- {exp}  {name}  K={K} N={N} g={g} {str(dt).split('.')[-1]} "
            f"zeros={'awq' if zeros else 'sym'}  M={M}   rotation {R} x {wb/1e6:.1f} MB = {tot:.0f} MB"
            f"{'' if tot >= 64 else '   <-- DID NOT BUST THE MALL'}")
        x = (torch.randn(M, K, device=S.DEV) * 0.3).to(dt)
        ok = True
        for swiz in swizlist:
            for diag, dname in diaglist:
                line, base = [], None
                for t in tilelist:
                    bm, bn, wn = S.tile3([int(v) for v in t.split("x")])
                    if not S.legal(bm, bn, g, wn):
                        line.append(f"{t}=ILLEGAL")
                        continue
                    us, reps = timed(W, t, x, False, ws, diag, swiz)
                    if us is None:
                        line.append(f"{t}=REFUSED({reps})")
                        continue
                    rb = -(-M // bm)
                    row([exp, name, K, N, g, str(dt).split('.')[-1], int(zeros), M, rb, t, diag,
                         dname, swiz, f"{us:.3f}", reps, R, dev_name, cu, card, lease, excl])
                    tag = f"{t}={us:8.1f}"
                    if gate and diag == 0 and swiz == 1:
                        r = ref.get((name, M, t))
                        if r is None:
                            tag += " [no fixture row]"
                        else:
                            dev = abs(us - r) / r
                            tag += f" [fixture {r:.1f}, {dev*100:+.0f}%]"
                            if dev > args.gate_tol:
                                ok = False
                    line.append(tag)
                    if base is None:
                        base = us
                    elif base:
                        line[-1] += f" ({us/base:.2f}x)"
                sw = f" swiz={swiz}" if len(swizlist) > 1 else ""
                out(f"    diag={diag:<3}{dname:<20}{sw}  " + "   ".join(line))
        del x, ws
        torch.cuda.empty_cache()
        return ok

    # ---- GATE: reproduce the recorded surface before interpreting anything -----------------------
    if "gate" in exps:
        out("=" * 100)
        out("GATE -- the probe must reproduce the RECORDED fixture cells it reasons about.")
        out("A probe that silently disagrees with the fixture is measuring its own configuration.")
        out("=" * 100)
        allok = True
        for name, K, N, g, dt, zeros in shapes:
            for M in (32, 48):
                allok &= run("gate", name, K, N, g, dt, zeros, M, tiles, [(0, "full")], gate=True)
        if not allok:
            out(f"\nGATE FAILED: a cell disagreed with the fixture by more than "
                f"{args.gate_tol*100:.0f}%. REFUSING to interpret the ablation below.")
            out("done")
            return 3
        out("\nGATE PASSED.\n")

    # ---- STAGE ABLATION: which stage carries the cliff ------------------------------------------
    if "stage" in exps:
        out("=" * 100)
        out("STAGE ABLATION -- rb==1 (M=32) vs rb==2 (M=48) at every DIAG bit, BOTH swizzles.")
        out("The bit that collapses the rb1-vs-rb2 ratio NAMES the stage. diag=16 leaves the weight")
        out("DATA read on and kills only the scales; diag=2 kills both -- the pair separates them.")
        out("=" * 100)
        for name, K, N, g, dt, zeros in shapes:
            for M in (32, 48):
                run("stage", name, K, N, g, dt, zeros, M, tiles, DIAG_BITS, swizlist=(1, 0))

    # ---- RBSWEEP: is it a STEP at M=BM, or a slope? ---------------------------------------------
    if "rbsweep" in exps:
        out("=" * 100)
        out("RB SWEEP -- M walked across the boundary. A reuse story predicts a STEP exactly at")
        out("M=BM+1 and FLAT us/row-block for rb>=2 (the reuse saturates); a launch/occupancy story")
        out("predicts a slope.")
        out("=" * 100)
        for name, K, N, g, dt, zeros in shapes:
            for M in (16, 28, 31, 32, 33, 36, 40, 48, 64, 65, 96, 128):
                run("rbsweep", name, K, N, g, dt, zeros, M, tiles, [(0, "full")])

    # ---- NSWEEP at rb==1: does the cliff need the grid to cover the machine? ---------------------
    if "nsweep" in exps:
        out("=" * 100)
        out("N SWEEP at rb==1 (M=32) and rb==2 (M=48) -- the number of DISTINCT resident slabs is")
        out("min(n_blocks, cu*blocks_per_cu). A capacity story predicts the cliff is ABSENT while")
        out("n_blocks is below that cap and APPEARS above it.")
        out("=" * 100)
        name, K, N0, g, dt, zeros = SHAPES["lag"]
        for N in (2048, 4096, 8192, 16384, 32768):
            for M in (32, 48):
                run("nsweep", name, K, N, g, dt, zeros, M, tiles, [(0, "full")])

    out("\ndone")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
