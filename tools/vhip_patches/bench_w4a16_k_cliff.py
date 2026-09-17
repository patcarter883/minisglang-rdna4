"""Test the lane-occupancy model of mmq_regdirect_w4a16_moe_gemv by sweeping K at fixed N.

MODEL (static read of w4a8_fp8_wmma_kernel.hip:1717): the K loop is
    for (int base = lane * 4; base < ppr_chunk; base += 32 * 4)
with ppr_chunk = K/8, so a lane's slot covers 4 int32 = 32 k-values and the wave needs
K >= 32 lanes * 32 = 1024 to put every lane to work. Below that, only K/32 lanes are active
and the rest of the wave idles through a full 5-step __shfl_xor reduction anyway.

PREDICTION: at fixed N, time should be roughly FLAT from K=256 to K=1024 (4x the weight bytes
for free, as idle lanes fill up), then scale ~linearly beyond K=1024. A flat region is the
signature of lane starvation; linear-from-256 would falsify the model and mean the cost is
something else (reduction tree, launch, LDS).

    VHIP_CMD='export PYTHONPATH=/opt/kernels; python3 /engine/tools/vhip_patches/bench_w4a16_k_cliff.py' \
    gpu-lease -n 1 -- docker compose --profile vhip run --rm --no-deps vhip
"""
from __future__ import annotations

import torch
import w4a8_fp8_wmma

DEV = "cuda"
G = 32
TOP_K = 8
BLOCK_M = 8
N = 2048          # gemm2's output width (= hidden)
E = TOP_K         # one padded block per routed expert


def time_call(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
    ev0.record()
    for _ in range(iters):
        fn()
    ev1.record()
    torch.cuda.synchronize()
    return ev0.elapsed_time(ev1) / iters * 1000.0  # us


def main():
    P = E * BLOCK_M
    sorted_ids = torch.arange(P, dtype=torch.int32, device=DEV)
    expert_ids = torch.arange(E, dtype=torch.int32, device=DEV)
    ntp = torch.tensor([P], dtype=torch.int32, device=DEV)

    print(f"N={N} E={E} P={P} block_m={BLOCK_M}  (gemm2 is the K=256 row)")
    print(f"{'K':>6} {'lanes':>6} {'MB':>7} {'us':>8} {'GB/s':>8}  {'us/MB':>7}")
    prev = None
    for K in (256, 512, 1024, 2048, 4096):
        g = torch.Generator(device=DEV).manual_seed(0)
        w = torch.randint(-(2**31), 2**31 - 1, (E, N, K // 8), generator=g, device=DEV,
                          dtype=torch.int32)
        s = (torch.rand((E, N, K // G), generator=g, device=DEV) * 0.02 + 0.005).to(torch.float16)
        z = torch.randint(0, 2**31 - 1, (E, N // 8, K // G), generator=g, device=DEV,
                          dtype=torch.int32)
        x = torch.randn((P, K), device=DEV, dtype=torch.float16)

        def call():
            return w4a8_fp8_wmma.mmq_regdirect_w4a16_moe_gemv(
                x, w, s, sorted_ids, expert_ids, ntp, N, 1, BLOCK_M, w_zeros=z)

        us = time_call(call)
        mb = E * N * K / 2 / 1e6
        active = min(32, max(1, K // 32))
        tag = ""
        if prev is not None:
            tag = f"   {us/prev[0]:.2f}x time for {mb/prev[1]:.0f}x bytes"
        print(f"{K:>6} {active:>6} {mb:>7.2f} {us:>8.1f} {mb/us/1e3:>8.1f} {us/mb:>7.1f}{tag}")
        prev = (us, mb)

    print("\nFLAT 256->1024 => lane starvation confirmed (idle lanes absorb the extra work).")
    print("LINEAR from 256  => model wrong; cost is the reduction/launch, not occupancy.")


if __name__ == "__main__":
    main()
