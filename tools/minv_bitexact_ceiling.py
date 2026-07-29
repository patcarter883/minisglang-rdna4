"""What is the BIT-IDENTICAL performance ceiling for the bf16 dense decode path?

dense_gemm_rd accumulates with WMMA (Wmma<T>::mma over 16-wide K steps). The summation inside a
WMMA op is hardware-defined, so any replacement must ALSO use WMMA with the same 16-wide chunking
to stay bit-identical -- which restricts us to changing the SCHEDULE and TILING, not the reduction
order. (Split-K is excluded too: S>1 reorders the partials.)

So this sweeps everything that is legal under that constraint -- kernel variant (rd / pipe / lds)
x block_m x BN -- at the real served shapes, ASSERTS bit-identity against the shipped default, and
reports the best. That bounds the option "keep the guarantee, get the speed by rescheduling".

Amortized timing + cache-busting rotation (a single re-read weight sits in the 64 MB Infinity Cache
and reports >1 TB/s, which is above the 674 GB/s peak -- the tell that the measurement is wrong).
"""
import time

import torch

import dense_gemm as dg

PEAK = 674.0
CACHE_BYTES = 96 << 20

# per-rank at TP=2 (hidden 2048). These are the linears that go through minv_linear today.
SHAPES = [
    ("attn_q_proj", 2048, 4096, 10),
    ("attn_o_proj", 2048, 2048, 10),
    ("attn_k_proj", 2048, 256, 10),
    ("shared_gate_up", 2048, 512, 40),
    ("shared_down", 256, 2048, 40),
    ("router_gate", 2048, 256, 40),
]


def rotation(OUT, IN, dev):
    wb = OUT * IN * 2
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


def variants(M, IN, OUT):
    """Every (name, fn) that is legal under the bit-identity constraint."""
    out = []
    for bm in (16, 32, 64):
        for bn in (32, 64, 128):
            if OUT % bn != 0:
                out.append((f"lds bm{bm} bn{bn}", bm, bn, "lds"))
                continue
            out.append((f"rd  bm{bm} bn{bn}", bm, bn, "rd"))
            out.append((f"lds bm{bm} bn{bn}", bm, bn, "lds"))
    return out


def call_of(kind, bm, bn):
    def f(x, w):
        M = x.shape[0]
        Mp = ((M + bm - 1) // bm) * bm
        xp = x if Mp == M else torch.nn.functional.pad(x, (0, 0, 0, Mp - M))
        if kind == "rd":
            return dg.dense_gemm_rd(xp, w, bm, bn)[:M]
        return dg.dense_gemm(xp, w, bm, bn)[:M]
    return f


def main():
    dev = "cuda"
    print(f"{'shape':<16}{'OUT':>6} | {'shipped us':>11}{'GB/s':>7} | {'best bit-identical':<20}"
          f"{'us':>8}{'GB/s':>7}{'gain':>7}")
    print("-" * 92)
    tot_ship = tot_best = 0.0
    for name, IN, OUT, nlayers in SHAPES:
        ws = rotation(OUT, IN, dev)
        x = (torch.randn(1, IN, device=dev)).to(torch.bfloat16)
        ship = call_of("rd" if OUT % 64 == 0 else "lds", 64, 64)
        ref = ship(x, ws[0])
        t_ship = bench(lambda w: ship(x, w), ws)
        best = (t_ship, "shipped rd/lds bm64 bn64")
        for label, bm, bn, kind in variants(1, IN, OUT):
            f = call_of(kind, bm, bn)
            try:
                got = f(x, ws[0])
            except Exception:
                continue
            if not torch.equal(got, ref):      # bit-identity is the whole constraint
                continue
            t = bench(lambda w: f(x, w), ws)
            if t < best[0]:
                best = (t, label)
        gb_s, gb_b = (IN * OUT * 2) / (t_ship * 1e-6) / 1e9, (IN * OUT * 2) / (best[0] * 1e-6) / 1e9
        print(f"{name:<16}{OUT:>6} | {t_ship:>11.1f}{gb_s:>7.0f} | {best[1]:<20}{best[0]:>8.1f}"
              f"{gb_b:>7.0f}{t_ship/best[0]:>6.2f}x")
        tot_ship += t_ship * nlayers
        tot_best += best[0] * nlayers
    print("-" * 92)
    print(f"modelled minv total per step (M=1): shipped {tot_ship/1000:.2f} ms -> "
          f"best bit-identical {tot_best/1000:.2f} ms  ({tot_ship/tot_best:.2f}x, "
          f"saves {(tot_ship-tot_best)/1000:.2f} ms/step)")
    print(f"(profile says dense_gemm_rd+dense_gemm = 3.74 ms/step; the bf16 dense bucket is 7.34; "
          f"peak {PEAK:.0f} GB/s)")


if __name__ == "__main__":
    main()
