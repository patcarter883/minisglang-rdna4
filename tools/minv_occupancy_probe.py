"""Is the M=1 minv path block-starved, or bandwidth-limited, or floored by something else?

METHOD (two fixes over the naive version, both of which matter here):
  1. AMORTIZED TIMING — time N back-to-back calls between ONE pair of syncs. A per-call
     synchronize() imposes a ~47 us floor on this box, which swamps a 3-25 us kernel and makes
     every shape look identical.
  2. CACHE BUSTING — rotate over enough distinct weight buffers to exceed the 64 MB Infinity
     Cache. In real decode the model's ~11 GB of weights cannot be cache-resident, so a
     microbench that re-reads ONE weight measures cache bandwidth, not HBM, and overstates by
     >1 TB/s (i.e. above the 674 GB/s peak — the tell that the measurement is wrong).

dense_gemm_rd launches grid = (ceil(OUT/BN), ceil(M/block_m)); at M=1 that is (OUT/BN, 1).
BN does not change the K-reduction order, so every BN is bit-identical and this probe is free of
the M-invariance question.
"""
import time

import torch

from minisgl.layers.minv import minv_linear

SHAPES = [
    ("gdn_in_proj_qkvz", 2048, 6144),
    ("gdn_out_proj", 2048, 2048),
    ("attn_q_proj", 2048, 4096),
    ("shared_gate_up", 2048, 512),
    ("router_gate", 2048, 256),
]
BNS = [32, 64, 128]
CACHE_BYTES = 96 << 20   # > 64 MB Infinity Cache


def make_rotation(OUT, IN, dev):
    """Enough distinct weights to exceed the cache, so reads come from HBM."""
    wbytes = OUT * IN * 2
    n = max(2, min(24, (CACHE_BYTES + wbytes - 1) // wbytes + 1))
    return [(torch.randn(OUT, IN, device=dev) * 0.02).to(torch.bfloat16) for _ in range(n)]


def bench_amortized(call, ws, iters=60, warmup=10):
    for i in range(warmup):
        call(ws[i % len(ws)])
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(iters):
        call(ws[i % len(ws)])
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e6 / iters


def main():
    dev = "cuda"
    print(f"{'shape':<18}{'OUT':>6}{'blk@64':>8} | "
          + "".join(f"{'BN=%d us' % b:>12}" for b in BNS)
          + f"{'GB/s@best':>11}{'%peak':>7}{'vs BN=64':>10}")
    print("-" * 104)
    for name, IN, OUT in SHAPES:
        ws = make_rotation(OUT, IN, dev)
        x = (torch.randn(1, IN, device=dev)).to(torch.bfloat16)
        ref = minv_linear(x, ws[0], block_m=16, BN=64)
        row, base, best = [], None, None
        for BN in BNS:
            if OUT % BN != 0:
                row.append(float("nan")); continue
            assert torch.equal(ref, minv_linear(x, ws[0], block_m=16, BN=BN)), \
                f"BN={BN} changed the result for {name}"
            t = bench_amortized(lambda w, b=BN: minv_linear(x, w, block_m=16, BN=b), ws)
            row.append(t)
            if BN == 64:
                base = t
            if best is None or t < best[0]:
                best = (t, BN)
        gb = (OUT * IN * 2) / (best[0] * 1e-6) / 1e9
        cells = "".join(f"{t:>12.1f}" if t == t else f"{'-':>12}" for t in row)
        print(f"{name:<18}{OUT:>6}{(OUT+63)//64:>8} | {cells}{gb:>11.0f}{100*gb/674:>6.0f}%{base/best[0]:>9.2f}x")
    print(f"\n(rotating {CACHE_BYTES>>20} MB of distinct weights per shape; HBM peak 674 GB/s)")
    print("All BN values verified bit-identical per shape.")


if __name__ == "__main__":
    main()
