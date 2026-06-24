"""Validate the rocWMMA GEMM probe vs torch matmul (gfx1201). Run under a 1-card lease."""
import glob
import os

import torch

_so = glob.glob(os.path.join(os.path.dirname(__file__), "wmma_probe_C*.so"))
assert _so, "build first: GPU_ARCHS=gfx1201 python setup.py build_ext --inplace"
torch.ops.load_library(_so[0])

torch.manual_seed(0)
for (M, K, N) in [(16, 16, 16), (64, 64, 64), (32, 128, 64), (128, 128, 128)]:
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    D = torch.ops.wmma_probe.gemm(A, B)
    ref = (A.float() @ B.float())
    d = (D - ref).abs().max().item()
    rel = d / ref.abs().max().item()
    print(f"  [{'PASS' if rel < 2e-2 else 'FAIL'}] {M}x{K}x{N}  max|Δ|={d:.3e}  rel={rel:.2e}")
print("rocWMMA GEMM probe done (fp16 in / fp32 acc; ~1e-2 rel is expected for fp16 inputs).")
