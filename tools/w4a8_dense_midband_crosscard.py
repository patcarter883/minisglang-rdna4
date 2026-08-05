#!/usr/bin/env python
"""Does the mid-band boundary MOVE between the box's two cards?

The surface is non-monotonic in N, and the shape of the non-monotonicity says why: in the mid-band
`wmma_tiled_tuned` launches exactly ceil(N/128) workgroups, so it is fast when that count is a
multiple of the CU count and slow when the last dispatch wave is half empty. It wins at N=8192 and
N=16384 (64 and 128 tiles = 1.00 and 2.00 waves of 64 CUs) and loses at 9216/10240/11264/17408.

If that reading is right, the boundary is a function of the DEVICE, not of the shape alone -- and
this box's two compute cards do not have the same CU count (RX 9070 XT = 64, RX 9070 = 56). A
dispatch rule keyed on "N/128 is a multiple of 64" would then be right on card 0 and wrong on card 1
of the SAME TP=2 job. This runs the identical grid on both, so the rule is chosen knowing that.

    gpu-lease -n 2 --timeout 1800 -- <container wrapper>   # genuinely needs BOTH cards
"""
from __future__ import annotations

import argparse
import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import w4a8_dense_midband_surface as S  # noqa: E402

NS = [6144, 8192, 9216, 10240, 11264, 16384, 17408]
MS = [17, 32, 40, 63]
K = 2048
ARMS = ("prefill_wmma", "wmma_tiled_tuned")


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
    for dev_i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(dev_i)
        S.DEV = torch.device(f"cuda:{dev_i}")
        torch.cuda.set_device(dev_i)
        out(f"\n########## cuda:{dev_i}  {p.name}  CUs={p.multi_processor_count} ##########")
        out(
            f"{'N':>7}{'tiles':>7}{'occ64':>7}{'occCU':>7}  "
            + "".join(f"{'M=%d P/T' % m:>18}" for m in MS)
        )
        for N in NS:
            ws, R, wb = S.rotation(N, K, 32, False)
            tiles = -(-N // 128)
            cu = p.multi_processor_count
            occ64 = tiles / (64 * -(-tiles // 64))
            occcu = tiles / (cu * -(-tiles // cu))
            cells = []
            for M in MS:
                x = (torch.randn(M, K, device=S.DEV) * 0.3).to(torch.bfloat16)
                t = {
                    a: S.time_graph(
                        lambda w, a=a: W.mmq_fp8_gemm(
                            x, w[0], w[1], kernel=a, w_zeros=None, weight_is_e2m1=False
                        ),
                        ws,
                    )
                    for a in ARMS
                }
                win = "P" if t[ARMS[0]] < t[ARMS[1]] else "T"
                cells.append(f"{t[ARMS[0]]:.0f}/{t[ARMS[1]]:.0f} {win}")
                del x
            out(
                f"{N:>7}{tiles:>7}{occ64:>7.2f}{occcu:>7.2f}  "
                + "".join(f"{c:>18}" for c in cells)
            )
            del ws
            torch.cuda.empty_cache()
    out("\ndone.")
    if fh:
        fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
