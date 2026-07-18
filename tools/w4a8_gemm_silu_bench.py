"""Micro-bench: FUSED mmq_fp8_gemm_silu [1 launch, (M,inter)] vs UNFUSED mmq_fp8_gemm(decode_gemv) +
tail_hip.silu_and_mul [2 launches + (M,2*inter) round-trip]. Dense SwiGLU decode shapes.
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


def pack_uint4_2d(w):
    N, K = w.shape
    w = w.to(torch.int32)
    packed = torch.zeros((N, K // 8), dtype=torch.int32, device=w.device)
    for i in range(8):
        packed |= (w[:, i::8] & 0xF) << (i * 4)
    return packed


def bench(fn, n=3000):
    for _ in range(100):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e6


def run(M, inter, K, g, dtype):
    N = 2 * inter
    x = torch.randn(M, K, dtype=dtype, device=DEV) * 0.3
    w13 = pack_uint4_2d(torch.randint(0, 16, (N, K), dtype=torch.int8, device=DEV))
    scales = torch.randn(N, K // g, dtype=torch.float16, device=DEV).abs() * 0.02 + 0.001

    def unfused():
        out1 = W.mmq_fp8_gemm(x, w13, scales, kernel="decode_gemv")
        return tail_hip.silu_and_mul(out1.contiguous())

    def fused():
        return W.mmq_fp8_gemm_silu(x, w13, scales)

    u, f = bench(unfused), bench(fused)
    print(f"  M={M} inter={inter} K={K} {str(dtype).split('.')[-1]}: "
          f"unfused={u:6.2f}us fused={f:6.2f}us  SPEEDUP={u/f:.2f}x  ({u-f:+.2f}us/MLP/token)")


def main():
    print("=== dense gate_up+silu decode bench (unfused vs fused) ===")
    for dtype in (torch.float16, torch.bfloat16):
        run(1, 768, 2048, 128, dtype)     # dense MLP-ish
        run(1, 1536, 4096, 128, dtype)    # larger dense MLP
        run(1, 512, 1536, 128, dtype)     # GLM shared-expert-ish
        run(2, 768, 2048, 128, dtype)


if __name__ == "__main__":
    main()
