"""Perf: our M-invariant WMMA GEMM (minv_linear) vs rocBLAS F.linear, across the real serving shapes
and prefill M values. Goal (north star): meet or beat rocBLAS so M-invariance is free -> rocBLAS-free.

Reports median us/call for each. block_m/BN are the current minv defaults (reusing the MoE spine); a
dedicated dense kernel would shed the routing-tensor + M-pad overhead this still carries."""
import torch, torch.nn.functional as F
from minisgl.layers.minv import minv_linear

def bench(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    import time
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        fn(); torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e6)
    ts.sort()
    return ts[len(ts)//2]

SHAPES = [
    ("cca_q  ", 2048, 1024), ("cca_k  ", 2048, 256), ("cca_v  ", 2048, 128),
    ("router0", 2048, 512),  ("moe_gate_up", 2048, 2816), ("moe_down", 1408, 2048),
    ("lm_head", 2048, 152064),
]
MS = [16, 64, 256, 1024, 4096]

def main():
    dev = "cuda"
    print(f"{'shape':<12}{'M':>6}  {'F.linear us':>12}{'minv us':>10}{'ratio':>8}")
    for name, IN, OUT in SHAPES:
        W = (torch.randn(OUT, IN, device=dev) * 0.02).to(torch.bfloat16)
        for M in MS:
            if name == "lm_head" and M > 1024:
                continue
            x = (torch.randn(M, IN, device=dev) * 1.0).to(torch.bfloat16)
            t_fl = bench(lambda: F.linear(x, W))
            best = None
            for bm in (16, 32, 64):
                for bn in (64, 128):
                    t = bench(lambda: minv_linear(x, W, block_m=bm, BN=bn))
                    if best is None or t < best[0]:
                        best = (t, bm, bn)
            t_mv, bm, bn = best
            r = t_mv / t_fl
            flag = "  <-- minv slower" if r > 1.15 else ("  ok" if r <= 1.05 else "")
            print(f"{name:<12}{M:>6}  {t_fl:>12.1f}{t_mv:>10.1f}{r:>7.2f}x  best(bm={bm},bn={bn}){flag}")

if __name__ == "__main__":
    main()
