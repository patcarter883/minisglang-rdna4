"""Validate driving moe_bf16_wmma.moe_bf16_gemm as a DENSE bf16 projection GEMM:
  C[M,OUT] = A[M,IN] @ W[OUT,IN]^T   (single expert, identity routing, fixed block_m).

Two checks:
  1. PARITY vs F.linear (rocBLAS): values should match to bf16 ULP (different reduction order, so not
     bit-identical, but numerically faithful).
  2. M-INVARIANCE: dense_gemm(x[:m]) == dense_gemm(x)[:m] BIT-EXACT across m. This is the property
     rocBLAS lacks and the fix needs. Fixed block_m/BN => full-K WMMA reduction, order-independent of M.
"""
import torch, torch.nn.functional as F
import moe_bf16_wmma as mb


def dense_gemm_bf16(x, weight, block_m=32, BN=64):
    """x [M,IN] bf16, weight [OUT,IN] bf16 -> [M,OUT] bf16, M-invariant."""
    M, IN = x.shape
    OUT = weight.shape[0]
    P = ((M + block_m - 1) // block_m) * block_m
    dev = x.device
    # identity routing: row i gathers A[i]; padded rows [M,P) get offs>=M -> masked.
    sorted_ids = torch.arange(P, device=dev, dtype=torch.int32)
    expert_ids = torch.zeros(P // block_m, device=dev, dtype=torch.int32)
    ntp = torch.tensor([P], device=dev, dtype=torch.int32)
    w = weight.unsqueeze(0).contiguous()  # [1, OUT, IN]
    out = mb.moe_bf16_gemm(x.contiguous(), w, sorted_ids, expert_ids, ntp, None,
                           1, block_m, M, BN, 0)  # top_k=1, num_valid=M, mul_weight=0
    return out[:M]


def main():
    dev = "cuda"
    torch.manual_seed(0)
    for (Nout, name) in [(1024, "q"), (256, "k"), (128, "v")]:
        W = (torch.randn(Nout, 2048, device=dev) * 0.02).to(torch.bfloat16)
        xfull = (torch.randn(520, 2048, device=dev) * 1.0).to(torch.bfloat16)
        print(f"\n===== {name} proj  Nout={Nout} =====")
        for block_m, BN in [(32, 64), (16, 64), (32, 128)]:
            ref = dense_gemm_bf16(xfull, W, block_m, BN)  # M=520
            # parity vs rocBLAS F.linear
            fl = F.linear(xfull, W)
            dpar = (ref.float() - fl.float()).abs()
            # M-invariance: recompute prefixes, compare to ref[:m]
            worst = 0.0; worstm = -1
            for m in (33, 48, 60, 131, 200, 384, 511):
                part = dense_gemm_bf16(xfull[:m], W, block_m, BN)
                d = (part.float() - ref[:m].float()).abs().max().item()
                if d > worst:
                    worst = d; worstm = m
            print(f"  block_m={block_m} BN={BN}: parity_vs_F.linear max|Δ|={dpar.max().item():.3e} "
                  f"(mean {dpar.mean().item():.2e}) | M-invariance max|Δ|={worst:.3e} @M={worstm} "
                  f"{'<-- BIT-EXACT M-INVARIANT' if worst==0 else '<-- NOT invariant'}")


if __name__ == "__main__":
    main()
