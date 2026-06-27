#!/usr/bin/env python
"""RXF (int8/NL/Hadamard) vs fp8 W4A8 latency at matched shapes — shows the int8 WMMA path
reaches parity with the validated fp8 kernel. Dense GEMM (GEMM-only and full incl rotate_quant)
and grouped MoE (decode + prefill).

  PYTHONPATH=/engine/python:/engine python /engine/tools/rxf_bench.py
"""
from __future__ import annotations

import torch

from minisgl.quant import kernels

SPAN = 32


def t(fn, iters=100, warmup=20):
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
    return s.elapsed_time(e) / iters * 1000  # us


def rxf_w(N, K, dev):
    idx = torch.randint(0, 16, (N, K), device=dev, dtype=torch.uint8)
    wp = (idx[:, 0::2] | (idx[:, 1::2] << 4)).contiguous()
    ws = ((torch.rand(N, K // SPAN, device=dev) - 0.5) * 0.04).to(torch.float16)
    return wp, ws


def fp8_w(N, K, dev):  # op-layout int4 (N,K/8) i32 + (N,K/32) f16, symmetric (zeros=None)
    q = torch.randint(0, 16, (N, K), device=dev, dtype=torch.int32)
    wp = torch.zeros((N, K // 8), dtype=torch.int32, device=dev)
    for j in range(8):
        wp |= (q[:, j::8] & 0xF) << (j * 4)
    ws = (torch.rand(N, K // 32, device=dev) * 0.02 + 0.005).to(torch.float16)
    return wp, ws


def main():
    dev = "cuda"
    nl = kernels._rxf_nl(dev)
    print("DENSE  (us/call)        fp8-W4A8   RXF-gemm   RXF-full(+rot)")
    for M, N, K in [(1, 4096, 4096), (64, 4096, 4096), (256, 4096, 4096), (64, 11008, 4096)]:
        x = (torch.randn(M, K, device=dev) * 0.3).to(torch.bfloat16)
        fwp, fws = fp8_w(N, K, dev)
        rwp, rws = rxf_w(N, K, dev)
        q, asc = torch.ops.rxf_hip.rotate_quant_int8(x.contiguous(), SPAN)
        t_fp8 = t(lambda: kernels.w4a8_linear(x, fwp, fws, None, 32))
        t_gemm = t(lambda: torch.ops.rxf_hip.linear(q, asc, rwp, rws, nl, None))
        t_full = t(lambda: kernels.rxf_linear(x, rwp, rws, None, SPAN))
        print(f"  M={M:4d} N={N:5d} K={K:5d}   {t_fp8:7.1f}    {t_gemm:7.1f}    {t_full:7.1f}")

    print("\nMoE    (us/call)        fp8-W4A8   RXF")
    E, K, inter, top_k = 32, 4096, 1408, 4
    fw13, fw13s = fp8_w(E * 2 * inter, K, dev)
    fw13 = fw13.reshape(E, 2 * inter, K // 8)
    fw13s = fw13s.reshape(E, 2 * inter, K // 32)
    fw2, fw2s = fp8_w(E * K, inter, dev)
    fw2 = fw2.reshape(E, K, inter // 8)
    fw2s = fw2s.reshape(E, K, inter // 32)
    rw13 = torch.stack([rxf_w(2 * inter, K, dev)[0] for _ in range(E)])
    rw13s = torch.stack([rxf_w(2 * inter, K, dev)[1] for _ in range(E)])
    rw2 = torch.stack([rxf_w(K, inter, dev)[0] for _ in range(E)])
    rw2s = torch.stack([rxf_w(K, inter, dev)[1] for _ in range(E)])
    for M in (1, 16, 128):
        x = (torch.randn(M, K, device=dev) * 0.3).to(torch.bfloat16)
        g = torch.randn(M, E, device=dev)
        t_fp8 = t(lambda: kernels.w4a8_moe(x, fw13, fw13s, None, fw2, fw2s, None, g, top_k, True))
        t_rxf = t(lambda: kernels.rxf_moe(x, rw13, rw13s, rw2, rw2s, g, top_k, True, span=SPAN))
        print(f"  M={M:4d}                   {t_fp8:7.1f}    {t_rxf:7.1f}")

    # --- breakdown of rxf_moe sub-ops at M=128 (prefill path) ---
    print("\nrxf_moe M=128 breakdown (us):")
    from vllm import _custom_ops as vops
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
    M, block_m = 128, 16
    x = (torch.randn(M, K, device=dev) * 0.3).to(torch.bfloat16)
    g = torch.randn(M, E, device=dev)
    tw = torch.empty(M, top_k, dtype=torch.float32, device=dev)
    ti = torch.empty(M, top_k, dtype=torch.int32, device=dev)
    tei = torch.empty(M, top_k, dtype=torch.int32, device=dev)
    vops.topk_softmax(tw, ti, tei, g.float(), True)
    sorted_ids, expert_ids, ntp = moe_align_block_size(ti, block_m, E, None, pad_sorted_ids=True)
    P = sorted_ids.shape[0]
    tw_flat = tw.reshape(-1).float().contiguous()
    q, asc = torch.ops.rxf_hip.rotate_quant_int8(x.contiguous(), SPAN)
    out1 = torch.ops.rxf_hip.moe_gemm(q, asc, rw13, rw13s, nl, sorted_ids, expert_ids, ntp, top_k, block_m, M * top_k)
    buf2 = torch.ops.tail_hip.silu_and_mul(out1.contiguous())
    q2, asc2 = torch.ops.rxf_hip.rotate_quant_int8(buf2.contiguous(), SPAN)
    ident = torch.arange(P, dtype=torch.int32, device=dev)
    out2 = torch.ops.rxf_hip.moe_gemm(q2, asc2, rw2, rw2s, nl, ident, expert_ids, ntp, 1, block_m, P)
    print(f"  rotate1    {t(lambda: torch.ops.rxf_hip.rotate_quant_int8(x.contiguous(), SPAN)):7.1f}")
    print(f"  gemm1 wmma {t(lambda: torch.ops.rxf_hip.moe_gemm(q, asc, rw13, rw13s, nl, sorted_ids, expert_ids, ntp, top_k, block_m, M*top_k)):7.1f}")
    print(f"  silu       {t(lambda: torch.ops.tail_hip.silu_and_mul(out1.contiguous())):7.1f}")
    print(f"  rotate2    {t(lambda: torch.ops.rxf_hip.rotate_quant_int8(buf2.contiguous(), SPAN)):7.1f}")
    print(f"  gemm2 wmma {t(lambda: torch.ops.rxf_hip.moe_gemm(q2, asc2, rw2, rw2s, nl, ident, expert_ids, ntp, 1, block_m, P)):7.1f}")
    print(f"  gather_red {t(lambda: torch.ops.rxf_hip.moe_gather_reduce(out2, sorted_ids, tw_flat, ntp, M, top_k, M*top_k)):7.1f}")


if __name__ == "__main__":
    main()
