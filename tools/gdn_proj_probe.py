"""The GDN projections are the largest kernel group in decode and they bypass minv entirely.

gdn/layer.py::_make_proj returns a plain nn.Linear for the unquantized case ("the 35B path"), so
in_proj_qkvz / in_proj_ba / out_proj go F.linear -> rocBLAS. The decode profile:
    rocBLAS MT128x128x32   3.185 ms/step   60 calls  53.1 us   (in_proj_qkvz + out_proj)
    rocBLAS MT16x16x32     0.422 ms/step   30 calls  14.1 us   (in_proj_ba)
= 3.61 ms/step, 49% of the 7.34 ms bf16-dense bucket, 24% of the whole 14.75 ms kernel step.

They therefore have NO M-invariance guarantee today (rocBLAS picks its kernel by shape AND M), so
moving them to any fixed-order kernel IMPROVES the guarantee rather than trading it away. This asks:
which fixed-order candidate is actually fastest at the served shapes, and is it M-invariant?

Amortized timing + cache-busting (a single re-read weight sits in the 64 MB Infinity Cache and
reports above the 674 GB/s peak -- the tell that the measurement is wrong).
"""
import time

import torch
import torch.nn.functional as F

from minisgl.layers.minv import minv_linear

PEAK = 674.0
CACHE_BYTES = 96 << 20
# per-rank at TP=2: hidden 2048, 16 k-heads x128, 32 v-heads x128, 30 GDN layers
SHAPES = [
    ("gdn_in_proj_qkvz", 2048, 6144, 30),
    ("gdn_out_proj", 2048, 2048, 30),
    ("gdn_in_proj_ba", 2048, 32, 30),
]


def rotation(OUT, IN, dev):
    wb = max(OUT * IN * 2, 1)
    n = max(2, min(24, (CACHE_BYTES + wb - 1) // wb + 1))
    return [(torch.randn(OUT, IN, device=dev) * 0.02).to(torch.bfloat16) for _ in range(n)]


def bench(call, ws, iters=60, warmup=12):
    for i in range(warmup):
        call(ws[i % len(ws)])
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(iters):
        call(ws[i % len(ws)])
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e6 / iters


def m_inv(fn, W, IN, dev, Mfull=512, probes=(1, 2, 8, 64, 200)):
    x = (torch.randn(Mfull, IN, device=dev) * 0.5).to(torch.bfloat16)
    full = fn(x, W)
    worst = 0.0
    for m in probes:
        worst = max(worst, (fn(x[:m], W) - full[:m]).abs().max().item())
    return worst


def main():
    dev = "cuda"
    try:
        from fp8_wmma import dense_bf16_gemv
    except Exception:
        dense_bf16_gemv = None

    cands = [
        ("F.linear (SHIPPED)", lambda x, w: F.linear(x, w)),
        ("minv_linear", lambda x, w: minv_linear(x, w)),
    ]
    if dense_bf16_gemv is not None:
        cands.append(("dense_bf16_gemv", lambda x, w: dense_bf16_gemv(x.contiguous(), w)))

    for M in (1, 8):
        print(f"\n===== M={M} =====")
        print(f"{'shape':<18}{'OUT':>6} | " + "".join(f"{n[:17]:>19}" for n, _ in cands))
        print("-" * (26 + 19 * len(cands)))
        totals = [0.0] * len(cands)
        for name, IN, OUT, nl in SHAPES:
            ws = rotation(OUT, IN, dev)
            x = (torch.randn(M, IN, device=dev)).to(torch.bfloat16)
            cells = []
            for i, (cn, fn) in enumerate(cands):
                try:
                    t = bench(lambda w: fn(x, w), ws)
                    gb = (IN * OUT * 2) / (t * 1e-6) / 1e9
                    cells.append(f"{t:>11.1f}us{gb:>6.0f}")
                    totals[i] += t * nl
                except Exception:
                    cells.append(f"{'n/a':>19}")
                    totals[i] = float("nan")
            print(f"{name:<18}{OUT:>6} | " + "".join(cells))
        print(f"{'x30 layers ms/step':<25} | " + "".join(f"{t/1000:>13.3f} ms   " for t in totals))

    print("\nM-INVARIANCE (max|d| of row i computed at M=i+1 vs at M=512; 0 = invariant)")
    for name, IN, OUT, _ in SHAPES:
        W = (torch.randn(OUT, IN, device=dev) * 0.02).to(torch.bfloat16)
        out = []
        for cn, fn in cands:
            try:
                out.append(f"{cn}={m_inv(fn, W, IN, dev):.2e}")
            except Exception:
                out.append(f"{cn}=n/a")
        print(f"  {name:<18} " + "  ".join(out))
    print(f"\n(HBM peak {PEAK:.0f} GB/s; profile: these are 3.61 ms/step = 24% of the kernel step)")


if __name__ == "__main__":
    main()
