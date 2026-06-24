"""Validate the rocWMMA GEMM probe vs torch matmul (gfx1201). Run under a 1-card lease."""
import glob
import os

import torch

_so = glob.glob(os.path.join(os.path.dirname(__file__), "wmma_probe_C*.so"))
assert _so, "build first: GPU_ARCHS=gfx1201 python setup.py build_ext --inplace"
torch.ops.load_library(_so[0])

torch.manual_seed(0)


def _chk(name, got, ref):
    d = (got - ref).abs().max().item()
    rel = d / max(ref.abs().max().item(), 1e-9)
    print(f"  [{'PASS' if rel < 2e-2 else 'FAIL'}] {name:28s} max|Δ|={d:.3e}  rel={rel:.2e}")


# shapes incl. the chunked gated-delta-rule matmul shapes (C=16, Dk=Dv=128)
for (M, K, N) in [(16, 16, 16), (64, 64, 64), (128, 128, 128), (16, 128, 16), (16, 128, 128)]:
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    _chk(f"NN {M}x{K}x{N} (A@B)", torch.ops.wmma_probe.gemm(A, B), A.float() @ B.float())
    Bnt = torch.randn(N, K, device="cuda", dtype=torch.float16)  # [N,K]
    _chk(f"NT {M}x{K}x{N} (A@B^T)", torch.ops.wmma_probe.gemm_nt(A, Bnt), A.float() @ Bnt.float().T)
    Atn = torch.randn(K, M, device="cuda", dtype=torch.float16)  # [K,M]
    _chk(f"TN {M}x{K}x{N} (A^T@B)", torch.ops.wmma_probe.gemm_tn(Atn, B), Atn.float().T @ B.float())
print("rocWMMA GEMM toolkit (NN/NT/TN) done — the matmul primitives for the chunked kernel.")

# Triangular solve U = (I + L)^-1 B (the one non-matmul piece of the chunked GDN prefill). L is
# strictly-lower CxC (unit diagonal implied), B is CxN. Validate vs torch.linalg.solve_triangular.
print("\n--- tri_solve U = (I + tril(.,-1))^-1 B (forward substitution) ---")
for (C, N) in [(16, 128), (16, 16), (8, 128), (16, 1)]:
    Lraw = torch.randn(C, C, device="cuda", dtype=torch.float32)
    L = torch.tril(Lraw, diagonal=-1)            # strict lower; diagonal ignored by the kernel
    B = torch.randn(C, N, device="cuda", dtype=torch.float32)
    got = torch.ops.wmma_probe.tri_solve(L, B)
    ref = torch.linalg.solve_triangular(L + torch.eye(C, device="cuda"), B, upper=False,
                                        unitriangular=True)
    _chk(f"tri_solve C={C} N={N}", got, ref)
print("tri_solve done — forward-substitution solver for the U = (I+L)^-1 B step.")
