"""shared_expert_gate is a 2048->1 linear run 40x/step. The decode profile puts it at 26.8 us/call
= 1.07 ms/step = 6.5% of the bs=1 step -- to read 4 KB of weight and produce one scalar per token.

Why it is slow: OUT=1 means OUT % BN != 0, so minv_linear falls to the ragged `dense_gemm` LDS
kernel, which runs a full 64x64 tile (grid (1,1) = ONE workgroup) with LDS staging + a barrier per
16-wide K step -- 128 staged iterations for a single dot product.

The constraint on replacing it is M-INVARIANCE, not bit-identity to today: minv's own docstring says
its parity vs F.linear is "~1 bf16 ULP (a different but fixed order)". So any candidate is admissible
if it uses ONE fixed reduction order at every M. This probe measures the prize and checks exactly
that property for each candidate.
"""
import time

import torch
import torch.nn.functional as F

from minisgl.layers.minv import minv_linear

IN, OUT = 2048, 1
MS = [1, 2, 8, 64, 512, 2048]


def bench(fn, iters=200, warmup=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e6 / iters


def m_invariant(fn, W, dev, Mfull=512, probes=(1, 2, 8, 64, 200)):
    """Does row i come out identical whether computed at M=i+1 or at M=Mfull?"""
    x = (torch.randn(Mfull, IN, device=dev) * 0.5).to(torch.bfloat16)
    full = fn(x, W)
    worst = 0.0
    for m in probes:
        part = fn(x[:m], W)
        d = (part - full[:m]).abs().max().item()
        worst = max(worst, d)
    return worst


def main():
    dev = "cuda"
    W = (torch.randn(OUT, IN, device=dev) * 0.02).to(torch.bfloat16)

    cands = {
        "minv_linear (SHIPPED)": lambda x, w: minv_linear(x, w),
        "F.linear (rocBLAS)": lambda x, w: F.linear(x, w),
        "(x*w).sum(-1) torch": lambda x, w: (x.float() * w.float()).sum(-1, keepdim=True).to(x.dtype),
        "x @ w.t() torch": lambda x, w: (x @ w.t()),
    }

    print(f"IN={IN} OUT={OUT}   (40 of these per decode step)")
    print(f"{'candidate':<26}" + "".join(f"{'M=%d' % m:>10}" for m in MS) + f"{'M-inv max|d|':>14}")
    print("-" * 100)
    for name, fn in cands.items():
        row = []
        for M in MS:
            x = (torch.randn(M, IN, device=dev) * 0.5).to(torch.bfloat16)
            try:
                row.append(bench(lambda: fn(x, W)))
            except Exception:
                row.append(float("nan"))
        try:
            inv = m_invariant(fn, W, dev)
            invs = f"{inv:.3e}" + ("  OK" if inv == 0.0 else "  BREAKS")
        except Exception as e:  # noqa: BLE001
            invs = f"err {e}"[:20]
        cells = "".join(f"{t:>10.1f}" if t == t else f"{'-':>10}" for t in row)
        print(f"{name:<26}{cells}{invs:>14}")

    print("\nus/call at M=1 x 40 layers = ms/step:")
    for name, fn in cands.items():
        x = (torch.randn(1, IN, device=dev) * 0.5).to(torch.bfloat16)
        try:
            t = bench(lambda: fn(x, W))
            print(f"  {name:<26}{t:7.1f} us -> {t*40/1000:5.3f} ms/step")
        except Exception:
            pass


if __name__ == "__main__":
    main()
