#!/usr/bin/env python
"""Is `prefill_wmma_ashuffle`'s mid-band band a REGIME, or a repeat of the timing noise floor?

The mid-band is a three-way surface, not a two-way threshold: over 432 measured mid-band cells the
undispatched third arm `prefill_wmma_ashuffle` takes 67 of them — MORE than `prefill_wmma`'s 59. But
cell COUNT is the wrong statistic for a dispatch decision. prefill_wmma's cells are worth up to
1.57x; ashuffle's median cell is worth 1.038x and its best is 1.131x, which is close enough to the
graph-replay run-to-run spread (~1.5% observed between two runs of the same shape) that "wins the
cell" and "won the coin flip" are not distinguishable from one pass.

So before adding a third arm to a hot-path dispatch, two questions the single sweep cannot answer:
  1. REPEATABILITY -- does the same (K, N, M) pick ashuffle on 3 independent passes, and is the gain
     stable, or does the winner move between passes?
  2. PORTABILITY -- the prefill_wmma corner MOVES between this box's two cards (64 vs 56 CUs). Does
     ashuffle's band move too? A band fitted on one card and routed for both is the same mistake as
     the fixed N threshold it replaces.

Controls are included on purpose: shapes where the single sweep says ashuffle LOSES must keep losing.

    gpu-lease -n 2 -- <per-card container wrapper>
"""
from __future__ import annotations

import argparse
import statistics
import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import w4a8_dense_midband_surface as S  # noqa: E402

# (label, K, N) -- the band the single sweep credits to ashuffle, plus controls it credits to tiled.
SHAPES = [
    ("BAND  q27.q_proj tp1", 5120, 6144),
    ("BAND  q27.down   tp2", 8704, 5120),
    ("BAND  grid",           5120, 8192),
    ("CTRL  grid",           5120, 1024),
    ("CTRL  q35.o_proj tp2", 2048, 2048),
    ("CTRL  glm.gate_up tp2", 2048, 10240),
]
MS = [17, 24, 32, 40, 48, 63]
ARMS = ("prefill_wmma", "wmma_tiled_tuned", "prefill_wmma_ashuffle")
PASSES = 3


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
    out(f"device: {p.name}  CUs={p.multi_processor_count*2}   {PASSES} independent passes per cell")
    out("gain = best(prefill_wmma, wmma_tiled_tuned) / ashuffle, per pass; >1 means ashuffle wins\n")

    flips = stable = 0
    for label, K, N in SHAPES:
        ws, R, wb = S.rotation(N, K, 32, False)
        out(f"=== {label}  K={K} N={N}  (rotation {R} x {wb/1e6:.1f} MB = {R*wb/1e6:.0f} MB) ===")
        out(f"    {'M':>4}  {'gain pass1':>11}{'pass2':>9}{'pass3':>9}   {'spread':>8}  "
            f"{'winner (3 passes)':>22}")
        for M in MS:
            x = (torch.randn(M, K, device=S.DEV) * 0.3).to(torch.bfloat16)
            gains, wins = [], []
            for _ in range(PASSES):
                t = {
                    a: S.time_graph(
                        lambda w, a=a: W.mmq_fp8_gemm(
                            x, w[0], w[1], kernel=a, w_zeros=None, weight_is_e2m1=False
                        ),
                        ws,
                    )
                    for a in ARMS
                }
                gains.append(min(t[ARMS[0]], t[ARMS[1]]) / t[ARMS[2]])
                wins.append(min(t, key=t.get))
            uniq = set(wins)
            if len(uniq) == 1:
                stable += 1
                verdict = wins[0]
            else:
                flips += 1
                verdict = "FLIPPED: " + "/".join(w[:4] for w in wins)
            out(
                f"    {M:>4}  " + "".join(f"{g:>{11 if i==0 else 9}.3f}" for i, g in enumerate(gains))
                + f"   {max(gains)-min(gains):>8.3f}  {verdict:>22}"
            )
            del x
        del ws
        torch.cuda.empty_cache()
        out("")

    out(f"cells whose winner was STABLE across {PASSES} passes: {stable}; FLIPPED: {flips}")
    out("\ndone.")
    if fh:
        fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
