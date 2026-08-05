#!/usr/bin/env python
"""Verify the NEW mid-band dispatch through the engine entry point, not through the kernel names.

Three things, on whichever card the lease hands us (the rule is device-derived, so both are valid):

  A. PROVENANCE -- what `w4a8_linear`'s auto-dispatch actually selects at every real shape x M, and
     what the OLD rule would have selected. If these two columns are identical the A/B below is
     measuring nothing, so the tool says so rather than reporting a green 1.00x.
  B. BIT-IDENTITY -- old arm vs new arm THROUGH `w4a8_linear`, at the M values where they differ.
     The dispatch flip is only free if the numbers do not move; this is the gate on that.
  C. COST -- graph-replay timed, weights rotated past the MALL by byte count. new / old.

    gpu-lease -n 1 --timeout 1800 -- \
      MIDBAND_TOOL=tools/w4a8_dense_midband_verify.py bash tools/w4a8_dense_midband_surface_run.sh
"""
from __future__ import annotations

import argparse
import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import w4a8_dense_midband_surface as S  # noqa: E402

# The shapes the corner actually decides, plus narrow-N controls the old rule got wrong.
SHAPES = [
    ("g4.q_proj     tp2", 2816, 2048, 32, torch.float16, False),
    ("g4.down       tp2", 1056, 2816, 32, torch.float16, False),
    ("q35.o_proj    tp2", 2048, 2048, 32, torch.bfloat16, True),
    ("q27.q_proj    tp1", 5120, 6144, 32, torch.bfloat16, True),   # ashuffle box
    ("q27.down      tp2", 8704, 5120, 32, torch.bfloat16, True),   # ashuffle box
    ("lag.gate_up   tp2", 2048, 8192, 32, torch.bfloat16, True),
    ("q35b4.gate_up tp2", 2560, 9216, 32, torch.bfloat16, True),
    ("glm.gate_up   tp2", 2048, 10240, 128, torch.bfloat16, True),
    ("lag.gate_up   tp1", 2048, 16384, 32, torch.bfloat16, True),
    ("q27.gate_up   tp2", 5120, 17408, 32, torch.bfloat16, True),
]
MS = [17, 20, 32, 40, 48, 56, 63]


def old_rule(m, gemv_max=16):
    """The rule this change replaces: the WHOLE mid-band on prefill_wmma, with no N term at all."""
    if m <= gemv_max:
        return "decode_gemv"
    return "wmma_tiled_tuned" if m >= 64 else "prefill_wmma"


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

    from minisgl.quant.kernels import _device_cu_count, _pick_dense_kernel, w4a8_linear

    torch.manual_seed(0)
    p = torch.cuda.get_device_properties(0)
    out(f"device: {p.name}   MPs(WGPs)={p.multi_processor_count}   CUs={_device_cu_count()}")
    out(f"fp8_wmma: {W.__file__}")
    out("us per call, graph-replay timed, rotation sized in BYTES past the 64 MB MALL\n")

    differ = 0
    tot_new = tot_old = 0.0
    for name, K, N, g, dt, zeros in SHAPES:
        ws, R, wb = S.rotation(N, K, g, zeros)
        out(
            f"=== {name}  K={K} N={N} g={g}  tiles={-(-N//128)}  "
            f"last-wave occ={-(-N//128)/(_device_cu_count()*-(-(-(-N//128))//_device_cu_count())):.3f}"
            f"   (rotation {R} x {wb/1e6:.1f} MB = {R*wb/1e6:.0f} MB) ==="
        )
        out(
            f"    {'M':>4} {'NEW dispatch':>22}{'OLD dispatch':>22}"
            f"{'new us':>10}{'old us':>10}{'new/old':>9}  {'max|delta|':>11}"
        )
        for M in MS:
            new = _pick_dense_kernel(M, False, g, k=K, n=N)
            old = old_rule(M)
            x = (torch.randn(M, K, device=S.DEV) * 0.3).to(dt)
            tn = S.time_graph(
                lambda w: w4a8_linear(x, w[0], w[1], w[2], g), ws
            )
            to = S.time_graph(
                lambda w: W.mmq_fp8_gemm(
                    x, w[0], w[1], kernel=old, w_zeros=w[2], weight_is_e2m1=False
                ),
                ws,
            )
            d = (
                w4a8_linear(x, ws[0][0], ws[0][1], ws[0][2], g).float()
                - W.mmq_fp8_gemm(
                    x, ws[0][0], ws[0][1], kernel=old, w_zeros=ws[0][2], weight_is_e2m1=False
                ).float()
            ).abs().max().item()
            differ += new != old
            tot_new += tn
            tot_old += to
            out(
                f"    {M:>4} {new:>22}{old:>22}{tn:>10.2f}{to:>10.2f}{tn/to:>8.2f}x  {d:>11.3e}"
            )
            del x
        del ws
        torch.cuda.empty_cache()
        out("")

    out(f"cells where the dispatch CHANGED: {differ} / {len(SHAPES)*len(MS)}")
    if differ == 0:
        out("!! the A/B compared the new rule against ITSELF -- the numbers above mean nothing")
    out(f"mid-band total: new {tot_new:.0f} us  vs  old {tot_old:.0f} us  "
        f"= {tot_old/tot_new:.2f}x")
    out("\ndone.")
    if fh:
        fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
