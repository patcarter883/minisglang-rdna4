"""Micro-bench: FUSED mmq_fp8_moe_gemm1_silu(gemv) [1 launch, (P,inter) out] vs UNFUSED
mmq_fp8_moe_gemm(gemv) + tail_hip.silu_and_mul [2 launches + (P,2*inter) out1 round-trip]. Decode shapes.
  gpu-lease -n 1 -- python this.py  (PYTHONPATH=<fusion w4a8 torch-ext>:/opt/kernels)
"""
import sys
import time
import torch

import w4a8_fp8_wmma as W  # noqa: E402
import tail_hip  # noqa: E402

DEV = torch.device("cuda:0")
torch.manual_seed(0)
print(f"w4a8 from {W.__file__}")


def pack_uint4_3d(w):
    E, N, K = w.shape
    w = w.to(torch.int32)
    packed = torch.zeros((E, N, K // 8), dtype=torch.int32, device=w.device)
    for i in range(8):
        packed |= (w[:, :, i::8] & 0xF) << (i * 4)
    return packed


def moe_align(topk_ids, block_m, E):
    T, top_k = topk_ids.shape
    num_valid = T * top_k
    flat = topk_ids.reshape(-1)
    sorted_ids, expert_ids = [], []
    for e in range(E):
        slots = torch.nonzero(flat == e, as_tuple=False).flatten().tolist()
        n = len(slots)
        npad = ((n + block_m - 1) // block_m) * block_m
        for i in range(npad):
            sorted_ids.append(slots[i] if i < n else num_valid)
        expert_ids.extend([e] * (npad // block_m))
    dev = topk_ids.device
    return (torch.tensor(sorted_ids, dtype=torch.int32, device=dev),
            torch.tensor(expert_ids, dtype=torch.int32, device=dev),
            torch.tensor([len(sorted_ids)], dtype=torch.int32, device=dev), num_valid)


def bench(fn, n=3000):
    for _ in range(100):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e6  # us/call


def run(T, E, inter, K, top_k, block_m, g, dtype):
    N = 2 * inter
    x = torch.randn(T, K, dtype=dtype, device=DEV) * 0.3
    w13 = pack_uint4_3d(torch.randint(0, 16, (E, N, K), dtype=torch.int8, device=DEV))
    G = K // g
    scales = torch.randn(E, N, G, dtype=torch.float16, device=DEV).abs() * 0.02 + 0.001
    topk_ids = torch.stack([torch.randperm(E, device=DEV)[:top_k] for _ in range(T)]).to(torch.int32)
    sti, eids, ntp, _ = moe_align(topk_ids, block_m, E)

    def unfused():
        out1 = W.mmq_fp8_moe_gemm(x, w13, scales, sti, eids, ntp, top_k, block_m, kernel="gemv")
        return tail_hip.silu_and_mul(out1.contiguous())

    def fused():
        return W.mmq_fp8_moe_gemm1_silu(x, w13, scales, sti, eids, ntp, top_k, block_m, kernel="gemv")

    u = bench(unfused)
    f = bench(fused)
    print(f"  T={T} E={E} inter={inter} K={K} tk={top_k} bm={block_m} {str(dtype).split('.')[-1]}: "
          f"unfused(gemm1+silu)={u:6.2f}us  fused={f:6.2f}us  "
          f"SPEEDUP={u/f:.2f}x  ({u-f:+.2f}us/MoE-layer/token)")


def main():
    print("=== moe_gemv_decode_silu decode bench (unfused gemm1+silu vs fused) ===")
    # Qwen3.6-35B-A3B-ish: 128 experts, top_k=8, inter~768, K=2048; plus a couple smaller MoE shapes.
    for dtype in (torch.float16, torch.bfloat16):
        run(1, 128, 768, 2048, 8, 16, 128, dtype)
        run(2, 128, 768, 2048, 8, 16, 128, dtype)
        run(1, 64, 512, 1024, 6, 16, 128, dtype)


if __name__ == "__main__":
    main()
