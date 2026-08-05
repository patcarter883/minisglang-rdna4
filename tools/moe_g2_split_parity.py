"""Bit-exactness gate for the fused-gemm2 top_k split (grid.z), through the REAL dispatch path.

The claim being gated: splitting the top_k reduction across grid.z changes max|Δ| by EXACTLY ZERO at
every slice count and every M, because a slice writes per-k terms and the last-arriving block folds
all top_k in ascending k — the same left fold the unsplit kernel does in registers.

ONE PROCESS, ONE SET OF INPUTS. The first version of this harness ran each slice count in its own
subprocess and rebuilt the inputs from a seed in each. That is not a parity test, it is a parity test
XOR an input-reproducibility test, and it reported failures at M>=16 that turned out to include the
harness itself: two legs with the SAME split count disagreed. The launcher reads
MINISGL_MOE_G2_SPLIT_K with getenv on every call, so every leg can run against the byte-identical
tensors in one process — which is what makes a delta attributable to the kernel.

CONTROLS, both mandatory:
  * `sk1b` — a second recording of the unsplit configuration. If sk1b != sk1 the table means nothing.
  * self-determinism — each leg is called twice and the two results compared, because the
    neighbouring scatter arm is NOT deterministic against itself (minv.py records 9.5e-7..2.4e-4
    between identical calls) and a split that quietly introduced an atomic would still look
    "bit-exact vs baseline" on a lucky run.

  PYTHONPATH=/opt/kernels:/engine/python:/engine python tools/moe_g2_split_parity.py
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

HIDDEN, INTER_FULL, E, TOP_K, GROUP, BLOCK_M = 2048, 512, 256, 8, 32, 16


def build(M: int, inter: int, dtype, dev, seed: int):
    import moe_hip

    gen = torch.Generator(device=dev).manual_seed(seed)
    gate = torch.randn((M, E), device=dev, dtype=dtype, generator=gen)
    tw, ti, sti, eid, ntp = moe_hip.moe_route_align(gate, TOP_K, True, E, BLOCK_M)
    buf2 = (torch.randn((sti.shape[0], inter), device=dev, dtype=dtype, generator=gen) * 0.05).contiguous()
    return dict(buf2=buf2, sti=sti, eid=eid, ntp=ntp,
                twf=tw.reshape(-1).float().contiguous(), P=int(sti.shape[0]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", default="3,4,5,6,8,16,32")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    import fp8_wmma

    dev = torch.device("cuda")
    dtype = getattr(torch, args.dtype)
    inter = INTER_FULL // args.tp
    wgen = torch.Generator(device=dev).manual_seed(7)
    w2 = torch.randint(-(2**31), 2**31 - 1, (E, HIDDEN, inter // 8), dtype=torch.int32,
                       device=dev, generator=wgen)
    w2_s = torch.rand((E, HIDDEN, inter // GROUP), device=dev, dtype=torch.float16,
                      generator=wgen) * 0.02 + 0.001
    w2_z = torch.full((E, HIDDEN // 8, inter // GROUP), 0x88888888 - (1 << 32),
                      dtype=torch.int32, device=dev)

    legs = ["sk1", "sk1b", "auto", "sk2", "sk4", "sk8"]
    ok = True
    print(f"{'M':>4} {'P':>6} {'leg':>6} {'max|d| vs sk1':>16} {'self-det':>10}")
    for M in [int(x) for x in args.m.split(",")]:
        v = build(M, inter, dtype, dev, seed=1000 + M)

        def call():
            return fp8_wmma.mmq_fp8_moe_gemm2_gather_reduce(
                v["buf2"], w2, w2_s, v["sti"], v["eid"], v["ntp"], v["twf"], TOP_K, BLOCK_M,
                w_zeros=w2_z, weight_is_e2m1=False)

        ref = None
        for leg in legs:
            if leg == "auto":
                os.environ.pop("MINISGL_MOE_G2_SPLIT_K", None)
            else:
                os.environ["MINISGL_MOE_G2_SPLIT_K"] = leg.replace("sk", "").replace("b", "")
            a, b = call(), call()
            torch.cuda.synchronize()
            det = torch.equal(a, b)
            if ref is None:
                ref = a
            d = (a.double() - ref.double()).abs().max().item()
            bad = (d != 0.0) or (not det)
            ok = ok and not bad
            print(f"{M:>4} {v['P']:>6} {leg:>6} {d:>16.3e} {str(det):>10}"
                  + ("   <-- FAIL" if bad else ""))
    os.environ.pop("MINISGL_MOE_G2_SPLIT_K", None)
    print("\nGATE:", "PASS — max|d| = 0 at every slice count and every M, and deterministic"
          if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
