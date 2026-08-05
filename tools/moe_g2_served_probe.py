"""Which gemm2 kernel does the ENGINE actually launch at each served decode M — and how big is
its grid?

The counter scorecard measured `moe_gemm2_gather_reduce_core` and labelled `M=1` "the served decode
case". `python/minisgl/quant/kernels.py` says otherwise: `w4a8_moe` takes the `M <= 2` ATOMIC SCATTER
branch first, so M=1 and M=2 never reach the fused gather-reduce at all. This probe settles it by
calling the real dispatch entry point (`minisgl.quant.kernels.w4a8_moe`) at the real served shape and
recording the `engaged(...)` name per M, rather than reading the branch and hoping.

Shape is Qwen3.6-35B-A3B-AWQ-4bit at TP=2, from its config.json (NOT the probe defaults the
scorecard used, which were E=32 group=128):
    hidden 2048 · moe_intermediate 512 -> 256 per rank · E 256 · top_k 8 · group_size 32 · 40 layers
so gemm2 is (M, K=256) x (E, N=2048, 256) int4-asym.

  PYTHONPATH=/opt/kernels:/engine/python:/engine python tools/moe_g2_served_probe.py [--m 1,5,6]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

# Capture every engaged() name per call instead of the one-shot log.
import minisgl._hip_engage as _eng

_FIRED: list[str] = []
_orig_engaged = _eng.engaged


def _spy(name: str) -> None:
    _FIRED.append(name)


_eng.engaged = _spy
import minisgl.quant.kernels as qk  # noqa: E402

qk.engaged = _spy


# Qwen3.6-35B-A3B-AWQ-4bit @ TP=2 (config.json text_config + quantization_config).
HIDDEN = 2048
INTER_FULL = 512
E = 256
TOP_K = 8
GROUP = 32


def build(dev: torch.device, dtype: torch.dtype, inter: int):
    """Real-layout AWQ->op int4 asymmetric weights. Values are random; SHAPES are the served ones."""
    k1 = HIDDEN
    w13 = torch.randint(-(2**31), 2**31 - 1, (E, 2 * inter, k1 // 8), dtype=torch.int32, device=dev)
    w13_s = (torch.rand((E, 2 * inter, k1 // GROUP), device=dev, dtype=torch.float16) * 0.02 + 0.001)
    w13_z = torch.full((E, (2 * inter) // 8, k1 // GROUP), 0x88888888 - (1 << 32), dtype=torch.int32, device=dev)
    w2 = torch.randint(-(2**31), 2**31 - 1, (E, k1, inter // 8), dtype=torch.int32, device=dev)
    w2_s = (torch.rand((E, k1, inter // GROUP), device=dev, dtype=torch.float16) * 0.02 + 0.001)
    w2_z = torch.full((E, k1 // 8, inter // GROUP), 0x88888888 - (1 << 32), dtype=torch.int32, device=dev)
    return w13, w13_s, w13_z, w2, w2_s, w2_z


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", default="1,2,3,4,5,6,8,16,32")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--trace-only", action="store_true", help="one call per M, for rocprofv3")
    args = ap.parse_args()

    dev = torch.device("cuda")
    dtype = getattr(torch, args.dtype)
    inter = INTER_FULL // args.tp
    torch.manual_seed(0)

    print(f"[probe] Qwen3.6-35B-A3B-AWQ TP={args.tp}: hidden={HIDDEN} inter={inter} "
          f"E={E} top_k={TOP_K} group={GROUP} dtype={dtype}")
    w13, w13_s, w13_z, w2, w2_s, w2_z = build(dev, dtype, inter)
    print(f"[probe] weights resident: "
          f"{sum(t.numel() * t.element_size() for t in (w13, w13_s, w13_z, w2, w2_s, w2_z)) / 2**20:.0f} MiB")

    for M in [int(x) for x in args.m.split(",")]:
        x = torch.randn((M, HIDDEN), device=dev, dtype=dtype) * 0.05
        gate = torch.randn((M, E), device=dev, dtype=dtype)

        def call():
            return qk.w4a8_moe(x, w13, w13_s, w13_z, w2, w2_s, w2_z, gate, TOP_K, True)

        _FIRED.clear()
        out = call()
        torch.cuda.synchronize()
        fired = list(dict.fromkeys(_FIRED))
        blk = qk._moe_block_m(M, E, TOP_K)
        g2 = [f for f in fired if "gemm_scatter" in f or "gather_reduce" in f or "gemm(" in f]

        if args.trace_only:
            print(f"M={M:<3} block_m={blk} kernels={fired}")
            continue

        for _ in range(args.warmup):
            call()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.iters):
            call()
        torch.cuda.synchronize()
        us = (time.perf_counter() - t0) / args.iters * 1e6
        print(f"M={M:<3} block_m={blk:<4} split_k={qk._moe_split_k(M)}  whole-MoE {us:8.2f} us   "
              f"out={tuple(out.shape)} {out.dtype}")
        print(f"        gemm2 arm: {g2}")
        print(f"        all      : {fired}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
