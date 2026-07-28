"""Launch-config sweep for `mmq_regdirect_w4a16_moe_gemv` on the two Qwen decode MoE shapes.

WHY: a cudagraph decode trace of the W4A16 MoE wiring showed gemm1 (K=2048, N=512) at 15.4us but
gemm2 (K=256, N=2048) at 64.4us -- ~33 GB/s of effective weight bandwidth, i.e. nowhere near
streaming. Short K with wide N is the per-column cross-lane-reduction floor: each lane ends up
owning ~one int32 of packed weight per column, so the __shfl tree, not the weight read, sets the
time. The launcher picks NWARPS/COLS from a heuristic tuned on the long-K shape
(`run_w4a16_moe_gemv`, w4a8_fp8_wmma_kernel.hip). This sweep measures whether that heuristic is
simply mis-tuned for gemm2 (a launcher fix) or whether the shape needs a different kernel
(GEMV-by-lanes: one column per lane, full-K serial dot, no cross-lane reduction).

The kernel reads VLLM_W4A16_MOE_GEMV_{NWARPS,COLS,BK} via getenv on EVERY launch, so the sweep can
set them per timed run with no rebuild.

    VHIP_CMD='export PYTHONPATH=/opt/kernels; python3 /engine/tools/vhip_patches/bench_w4a16_moe_gemv.py' \
    gpu-lease -n 1 -- docker compose --profile vhip run --rm --no-deps vhip
"""
from __future__ import annotations

import os

import torch
import w4a8_fp8_wmma

DEV = "cuda"
G = 32
E_TOT = 256      # Qwen3.6-35B-A3B
TOP_K = 8
BLOCK_M = 8


def build(N, K, E):
    g = torch.Generator(device=DEV).manual_seed(0)
    w = torch.randint(-(2**31), 2**31 - 1, (E, N, K // 8), generator=g, device=DEV,
                      dtype=torch.int32)
    s = (torch.rand((E, N, K // G), generator=g, device=DEV) * 0.02 + 0.005).to(torch.float16)
    z = torch.randint(0, 2**31 - 1, (E, N // 8, K // G), generator=g, device=DEV, dtype=torch.int32)
    return w, s, z


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


def sweep(label, M_rows, N, K, top_k, weight_bytes, n_experts):
    """M_rows = rows of the A operand actually presented to the kernel.

    n_experts is the number of ROUTED experts (one padded block each), which is top_k for gemm1
    but is NOT top_k for gemm2 -- gemm2 runs with top_k=1 (identity gather) over the same 8 routed
    expert blocks. Deriving one from the other benchmarks 1/8th of the real weight read."""
    E = n_experts
    w, s, z = build(N, K, E)
    P = E * BLOCK_M
    x = torch.randn((M_rows, K), device=DEV, dtype=torch.float16)
    sorted_ids = torch.arange(P, dtype=torch.int32, device=DEV)
    if M_rows == 1:  # gemm1: sorted ids index (token*top_k + slot); one real row per expert block
        sorted_ids = torch.zeros(P, dtype=torch.int32, device=DEV)
        for e in range(E):
            sorted_ids[e * BLOCK_M] = e
            sorted_ids[e * BLOCK_M + 1:(e + 1) * BLOCK_M] = M_rows * top_k  # invalid -> masked
    expert_ids = torch.arange(E, dtype=torch.int32, device=DEV)
    ntp = torch.tensor([P], dtype=torch.int32, device=DEV)

    def call():
        return w4a8_fp8_wmma.mmq_regdirect_w4a16_moe_gemv(
            x, w, s, sorted_ids, expert_ids, ntp, N, top_k, BLOCK_M, w_zeros=z)

    print(f"\n=== {label}: K={K} N={N} P={P} rows={M_rows} "
          f"(weights read {weight_bytes/1e6:.2f} MB)")
    base = None
    results = []
    for nw in (4, 8, 16, 32):
        for cols in (1, 2, 4, 8):
            os.environ["VLLM_W4A16_MOE_GEMV_NWARPS"] = str(nw)
            os.environ["VLLM_W4A16_MOE_GEMV_COLS"] = str(cols)
            try:
                us = time_call(call)
            except Exception as e:
                print(f"  nw={nw:<3d} cols={cols}: FAILED {str(e)[:60]}")
                continue
            results.append((us, nw, cols))
            if base is None:
                base = us
    os.environ.pop("VLLM_W4A16_MOE_GEMV_NWARPS", None)
    os.environ.pop("VLLM_W4A16_MOE_GEMV_COLS", None)
    auto = time_call(call)
    results.sort()
    print(f"  auto (launcher heuristic): {auto:7.1f} us  -> {weight_bytes/auto/1e3:6.1f} GB/s")
    for us, nw, cols in results[:6]:
        print(f"  nw={nw:<3d} cols={cols:<2d}          : {us:7.1f} us  -> "
              f"{weight_bytes/us/1e3:6.1f} GB/s   {auto/us:.2f}x vs auto")
    return auto, results[0]


def main():
    hidden, inter = 2048, 256  # TP=2 shard
    # gemm1: x (1, hidden) -> (P, 2*inter). Reads top_k experts' w13.
    b1 = TOP_K * 2 * inter * hidden // 2
    a1, best1 = sweep("gemm1 (w13)", 1, 2 * inter, hidden, TOP_K, b1, n_experts=TOP_K)
    # gemm2: x (P, inter) -> (P, hidden), identity gather (top_k=1).
    b2 = TOP_K * hidden * inter // 2
    a2, best2 = sweep("gemm2 (w2)", TOP_K * BLOCK_M, hidden, inter, 1, b2, n_experts=TOP_K)

    print(f"\nper-layer decode MoE: auto {a1 + a2:.1f} us   "
          f"best-config {best1[0] + best2[0]:.1f} us   "
          f"(stock Triton fused_moe measured ~42 us/layer at decode)")


if __name__ == "__main__":
    main()
