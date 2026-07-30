"""Isolate the MoE (w4a8_moe / MXFP4) cost at decode (M=1) vs spec-verify (M=5, M=9) token counts.

Prime suspect for the ~41ms fixed spec-verify tax: at M tokens x top_k=8, up to M*8 of the 256 experts
activate (vs 8 for a 1-token decode), and the grouped GEMM fires one segment per active expert. This
times the exact `kernels.w4a8_moe` call the MXFP4 35B uses, at its real geometry, for M in {1,5,9},
so we can see how the MoE scales from decode to verify. x40 layers -> multiply per-call ms.

Timing only depends on shapes/dtypes/route, so random int4 packs + scales are fine.

Run under a 1-card lease:
  gpu-lease -n 1 -- bash -c 'docker run ... python /engine/tools/moe_verify_microbench.py'
"""
from __future__ import annotations

import time
import torch
from minisgl.quant import kernels

DEV = "cuda"
torch.manual_seed(0)
# Real Qwen3.6-35B-A3B-MXFP4 MoE geometry (config.json text_config):
E = 256            # num_experts
K = 2048           # hidden_size
INTER = 512        # moe_intermediate_size
TOP_K = 8
G = 32             # MXFP4 E8M0 group size
NLAYERS = 40
MS = [1, 5, 9]     # decode(1), K=4 verify(5), K=8 verify(9)
ITERS = 50


def _mk_weights():
    # op-layout packed int4 weights (values irrelevant for timing) + fp16 group scales.
    w13 = torch.randint(-2**31, 2**31 - 1, (E, 2 * INTER, K // 8), dtype=torch.int32, device=DEV)
    w13_s = torch.randn(E, 2 * INTER, K // G, dtype=torch.float16, device=DEV)
    w2 = torch.randint(-2**31, 2**31 - 1, (E, K, INTER // 8), dtype=torch.int32, device=DEV)
    w2_s = torch.randn(E, K, INTER // G, dtype=torch.float16, device=DEV)
    return w13, w13_s, w2, w2_s


def _bench(fn, iters=ITERS):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1e3  # ms/call


def main():
    assert torch.cuda.is_available()
    w13, w13_s, w2, w2_s = _mk_weights()
    print(f"=== MoE (w4a8_moe MXFP4) decode-vs-verify microbench ({torch.cuda.get_device_name()}) ===")
    print(f"E={E} K={K} inter={INTER} top_k={TOP_K}  — ms per single-layer call, x{NLAYERS} layers")
    print(f"{'M':>3} | {'active experts (<=M*8, cap E)':>28} | {'moe ms':>8} | {'vs M=1':>7} | {'x40':>8}")
    print("-" * 72)
    base = None
    for M in MS:
        x = torch.randn(M, K, dtype=torch.bfloat16, device=DEV)
        gating = torch.randn(M, E, dtype=torch.float32, device=DEV)

        def run():
            kernels.w4a8_moe(x, w13, w13_s, None, w2, w2_s, None,
                             gating, TOP_K, True, weight_is_e2m1=True)

        # count distinct experts this route actually hits (for context)
        ids = gating.topk(TOP_K, dim=-1).indices
        n_active = int(ids.unique().numel())
        t = _bench(run)
        base = base or t
        print(f"{M:>3} | {n_active:>28} | {t:>8.3f} | {t/base:>6.2f}x | {t*NLAYERS:>7.2f}ms")
    print("\nIf moe(M=5) >> moe(M=1): MoE expert fan-out is the fixed verify tax. x40 => total MoE ms.")


if __name__ == "__main__":
    main()
