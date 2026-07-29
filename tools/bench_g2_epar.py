"""moe_gemm2_gather_reduce: serial vs EPAR at the SERVED decode shape.

The served gemm2 for Qwen3.6-35B-A3B at TP=2 is K = moe_intermediate = 256, N = hidden = 2048,
E = 256, top_k = 8. At K=256 the K-on-lanes sweep gives ppr = K/8 = 32 int32 and starts at
base = lane*4, so only lanes 0-7 have work while 24/32 idle -- and all 32 still pay the 5-step
__shfl_xor. Profiled in-serve at 25.1 us x 40 layers = 1.0 ms/step = 6.1% of a 16.5 ms bs=1 step.

BYLANE is NOT bit-exact vs K-on-lanes (different, still fixed, dot summation order), so this reports
the delta rather than asserting equality. It IS M-invariant, which is what the engine requires.
"""
import os
import time

import torch

import fp8_wmma
import moe_hip


def build(M, K, N, E, top_k, block_m, group_size=128, dev="cuda"):
    torch.manual_seed(0)
    ng = K // group_size
    w = torch.randint(-(2**31), 2**31 - 1, (E, N, K // 8), dtype=torch.int32, device=dev)
    s = (torch.rand(E, N, ng, device=dev) * 0.02 + 0.005).to(torch.float16)
    z = torch.randint(-(2**31), 2**31 - 1, (E, N // 8, ng), dtype=torch.int32, device=dev)
    ids = torch.randint(0, E, (M, top_k), dtype=torch.int32, device=dev)
    sid, eid, ntp = moe_hip.moe_align(ids, E, block_m)
    P = sid.shape[0]                                   # buf2 is PRE-SORTED: one row per padded slot
    x = (torch.randn(P, K, device=dev) * 0.3).to(torch.bfloat16)
    tw = torch.rand(M, top_k, device=dev, dtype=torch.float32)
    return x, w, s, z, sid, eid, ntp, tw


def run(x, w, s, z, sid, eid, ntp, tw, top_k, block_m):
    return fp8_wmma.mmq_fp8_moe_gemm2_gather_reduce(
        x, w, s, sid, eid, ntp, tw.reshape(-1).contiguous(), top_k, block_m, w_zeros=z)


def bench(fn, iters=100, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e6 / iters


SHAPES = [
    ("SERVED decode  M=1  K=256 N=2048", 1, 256, 2048, 256, 8, 16),
    ("decode         M=2  K=256 N=2048", 2, 256, 2048, 256, 8, 16),
    ("decode         M=8  K=256 N=2048", 8, 256, 2048, 256, 8, 16),
    ("long-K control M=1  K=1024 N=2048", 1, 1024, 2048, 256, 8, 16),
]

print(f"{'shape':<36}{'serial us':>12}{'EPAR us':>11}{'speedup':>9}{'max|d|':>11}{'rel':>10}")
print("-" * 90)
for name, M, K, N, E, tk, bm in SHAPES:
    args = build(M, K, N, E, tk, bm)
    os.environ["VLLM_W4A8_MOE_G2FUSE_EPAR"]="0"
    ref = run(*args, tk, bm).clone()
    t_k = bench(lambda: run(*args, tk, bm))
    os.environ["VLLM_W4A8_MOE_G2FUSE_EPAR"]="1"
    got = run(*args, tk, bm).clone()
    t_b = bench(lambda: run(*args, tk, bm))
    os.environ["VLLM_W4A8_MOE_G2FUSE_EPAR"]="0"
    d = (got - ref).abs().max().item()
    rel = d / max(ref.abs().max().item(), 1e-9)
    print(f"{name:<36}{t_k:>12.1f}{t_b:>11.1f}{t_k/t_b:>8.2f}x{d:>11.3e}{rel:>10.2e}")

print("\nx40 layers, M=1: serial vs EPAR ms/step")
args = build(1, 256, 2048, 256, 8, 16)
os.environ["VLLM_W4A8_MOE_G2FUSE_EPAR"]="0"
a = bench(lambda: run(*args, 8, 16))
os.environ["VLLM_W4A8_MOE_G2FUSE_EPAR"]="1"
b = bench(lambda: run(*args, 8, 16))
print(f"  serial {a*40/1000:.3f} ms/step -> BYLANE {b*40/1000:.3f} ms/step "
      f"(saves {(a-b)*40/1000:.3f} ms/step)")
