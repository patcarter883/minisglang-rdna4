#!/usr/bin/env python
"""gemm2 (W4A8 down-proj) decode SCATTER latency: vendored w4a8_fp8_wmma vs minisgl split-K
(moe_splitk_hip) at the real Qwen3.6-35B decode shape. Task A (#17).

gemm2 at M=1 is occupancy-starved (grid = (N/64, P/16) ~= a few hundred blocks, ~15% peak BW).
Split-K carves the K=inter contraction across `split_k` grid.z blocks. Both kernels run the same
internal fp8 act-quant, so this is an apples-to-apples gemm2 timing.

  PYTHONPATH=/engine/python:/engine python /engine/tools/gemm2_splitk_bench.py   (1 card)
"""
from __future__ import annotations

import torch

import moe_hip
import moe_splitk_hip  # noqa: F401  registers torch.ops.moe_splitk_hip
import w4a8_fp8_wmma


def t(fn, iters=300, warmup=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    e.synchronize()
    return s.elapsed_time(e) / iters * 1000  # us/call


def pack_int4(q):  # (E,N,K)->(E,N,K//8) i32, nibble j = k8*8+j
    E, N, K = q.shape
    packed = torch.zeros((E, N, K // 8), dtype=torch.int32, device=q.device)
    for j in range(8):
        packed |= (q[:, :, j::8].to(torch.int32) & 0xF) << (j * 4)
    return packed


def main():
    dev = "cuda"
    E, K, inter, g, block_m = 256, 2048, 512, 32, 16  # real 35B gemm2: N=hidden=K, contraction=inter
    num_groups = inter // g
    print(f"gemm2 decode shape: E={E} hidden={K} inter={inter} g={g} block_m={block_m} "
          f"num_groups={num_groups}")
    w2 = pack_int4(torch.randint(0, 16, (E, K, inter), device=dev))            # (E,K,inter//8)
    s2 = (torch.rand(E, K, inter // g, device=dev) * 0.02 + 0.005).to(torch.float16)

    for M, top_k in [(1, 8), (2, 8)]:
        torch.manual_seed(M)
        # distinct experts per token (decode: few real rows -> the occupancy-starved regime)
        ti = torch.stack([torch.randperm(E, device=dev)[:top_k] for _ in range(M)]).to(torch.int32)
        sorted_ids, expert_ids, ntp = moe_hip.moe_align(ti, E, block_m)
        P = sorted_ids.shape[0]
        buf2 = (torch.randn(P, inter, device=dev) * 0.1).to(torch.float16)
        tw = torch.rand(M * top_k, device=dev, dtype=torch.float32)
        acc = torch.zeros((M, K), dtype=torch.float32, device=dev)

        def base():
            acc.zero_()
            w4a8_fp8_wmma.mmq_fp8_moe_gemm_scatter(
                buf2, w2, s2, sorted_ids, expert_ids, ntp, tw, acc, top_k, block_m,
                kernel="wmma", w_zeros=None)

        def splitk(S):
            def f():
                acc.zero_()
                torch.ops.moe_splitk_hip.moe_gemm_splitk_scatter(
                    buf2, w2, s2, None, sorted_ids, expert_ids, ntp, tw, acc, top_k, block_m, S)
            return f

        tb = t(base)
        print(f"\nM={M} top_k={top_k} P={P} (blocks base = {(K + 63)//64} x {P//block_m})")
        print(f"  vendored wmma scatter : {tb:7.1f} us")
        for S in (2, 4, 8, 16):
            if S > num_groups:
                continue
            ts = t(splitk(S))
            print(f"  split-K S={S:<2}           : {ts:7.1f} us   ({tb/ts:.2f}x)")


if __name__ == "__main__":
    main()
