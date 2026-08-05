#!/usr/bin/env python
"""Is the mid-band a KERNEL choice, or just a TILE choice on one kernel?

The surface sweep says `prefill_wmma` beats `wmma_tiled_tuned` in a mid-band corner that is NOT
monotonic in N: tiled wins at N=8192 and N=16384 and loses at 9216/10240/11264/17408/34816. Read the
launcher and that stops being mysterious --

    wmma_tiled_tuned : BM=256, BN=128 -> grid = (ceil(M/256), ceil(N/128)), 16 warps/WG
    prefill_wmma     : at M<128 takes a SMALL-M tile, SBM=64, V2_BN=64 -> grid = (ceil(N/64), 1),
                       8 warps/WG

-- both land the same total warp count, but tiled's workgroups are 2x coarser in N and its BM=256
tile discards (256-M)/256 of its rows as padding in the mid-band. `ceil(N/128)` is a multiple of the
64 CUs exactly at N=8192 and N=16384, which is exactly where tiled wins. So the hypothesis is that
this was never an ALGORITHM difference (they are bit-identical) but a TILE-SHAPE difference, and
`wmma_tiled_tuned` already accepts its tile at runtime through VLLM_W4A8_V7_CFG.

If tiled at BM=64 wins the corner too, the mid-band arm has no regime that a tile choice on the
shared core would not serve better -- which is what KERNEL_CORE_POLICY asks us to establish before
keeping a second kernel alive.

    gpu-lease -n 1 --timeout 3600 -- \
      MIDBAND_TOOL=tools/w4a8_dense_midband_tilecfg.py bash tools/w4a8_dense_midband_surface_run.sh
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from w4a8_dense_midband_surface import DEV, rotation, time_graph  # noqa: E402

# (name, K, N, g, dtype, awq_zeros) -- the corner prefill_wmma won, plus the two N at which
# ceil(N/128) is an exact multiple of 64 CUs and tiled won.
SHAPES = [
    ("q27.q_proj    tp1", 5120, 6144, 32, torch.bfloat16, True),
    ("lag.gate_up   tp2", 2048, 8192, 32, torch.bfloat16, True),  # 64 tiles -> tiled's best case
    ("q35b4.gate_up tp2", 2560, 9216, 32, torch.bfloat16, True),
    ("glm.gate_up   tp2", 2048, 10240, 128, torch.bfloat16, True),
    ("grid          N=11264", 2816, 11264, 32, torch.bfloat16, False),
    ("lag.gate_up   tp1", 2048, 16384, 32, torch.bfloat16, True),  # 128 tiles -> tiled's best case
    ("q27.gate_up   tp2", 5120, 17408, 32, torch.bfloat16, True),
    ("q27.gate_up   tp1", 5120, 34816, 32, torch.bfloat16, True),
    ("g4.q_proj     tp2", 2816, 2048, 32, torch.float16, False),  # narrow-N control
]
MS = [17, 20, 32, 40, 63, 64, 128, 256]
CFGS = ["256x128", "128x128", "64x128", "64x64", "256x64"]


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
    out(f"device: {torch.cuda.get_device_name(0)}   fp8_wmma: {W.__file__}")
    out("us per call; graph-replay timed, rotation sized in BYTES past the 64 MB MALL")
    out("tiled@BMxBN = wmma_tiled_tuned with VLLM_W4A8_V7_CFG set to that tile\n")

    cols = ["prefill_wmma"] + [f"tiled@{c}" for c in CFGS]
    for name, K, N, g, dt, zeros in SHAPES:
        ws, R, wbytes = rotation(N, K, g, zeros)
        ntile = -(-N // 128)
        out(
            f"=== {name}  K={K} N={N} g={g}  (rotation {R} x {wbytes/1e6:.1f} MB "
            f"= {R*wbytes/1e6:.0f} MB)   ceil(N/128)={ntile} = {ntile/64:.2f} x 64 CUs ==="
        )
        out(f"    {'M':>4} " + "".join(f"{c:>16}" for c in cols) + f"   {'winner':>16}")
        for M in MS:
            x = (torch.randn(M, K, device=DEV) * 0.3).to(dt)
            t = {}
            os.environ.pop("VLLM_W4A8_V7_CFG", None)
            t["prefill_wmma"] = time_graph(
                lambda w: W.mmq_fp8_gemm(
                    x, w[0], w[1], kernel="prefill_wmma", w_zeros=w[2], weight_is_e2m1=False
                ),
                ws,
            )
            for c in CFGS:
                os.environ["VLLM_W4A8_V7_CFG"] = c
                try:
                    t[f"tiled@{c}"] = time_graph(
                        lambda w: W.mmq_fp8_gemm(
                            x, w[0], w[1], kernel="wmma_tiled_tuned", w_zeros=w[2],
                            weight_is_e2m1=False,
                        ),
                        ws,
                    )
                except Exception as e:  # noqa: BLE001
                    out(f"      (M={M} {c}: {type(e).__name__}: {e})")
            os.environ.pop("VLLM_W4A8_V7_CFG", None)
            win = min(t, key=t.get)
            out(
                f"    {M:>4} "
                + "".join(f"{t.get(c, float('nan')):>16.2f}" for c in cols)
                + f"   {win:>16}"
            )
            del x
        # bit-identity across every tile config -- a tile is not allowed to change the numbers
        x = (torch.randn(20, K, device=DEV) * 0.3).to(dt)
        os.environ.pop("VLLM_W4A8_V7_CFG", None)
        ref = W.mmq_fp8_gemm(
            x, ws[0][0], ws[0][1], kernel="prefill_wmma", w_zeros=ws[0][2], weight_is_e2m1=False
        ).float()
        ds = []
        for c in CFGS:
            os.environ["VLLM_W4A8_V7_CFG"] = c
            try:
                y = W.mmq_fp8_gemm(
                    x, ws[0][0], ws[0][1], kernel="wmma_tiled_tuned", w_zeros=ws[0][2],
                    weight_is_e2m1=False,
                ).float()
            except Exception:  # noqa: BLE001
                continue
            ds.append(f"{c}:{(y - ref).abs().max().item():.3e}")
        os.environ.pop("VLLM_W4A8_V7_CFG", None)
        out("    bit-identity vs prefill_wmma (M=20): " + "  ".join(ds))
        del x, ws
        torch.cuda.empty_cache()
        out("")

    out("done.")
    if fh:
        fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
