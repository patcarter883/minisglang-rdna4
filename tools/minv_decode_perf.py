"""Decode-M (M<=16) perf + bit-parity for the three bf16 dense paths, at the REAL Qwen3.6-35B-A3B
shapes at TP=2.

Why this exists: tools/minv_perf.py measures M in {16,64,256,1024,4096} on the ZAYA shapes — it never
measured the decode case (M=1..8) and never included the bf16 decode GEMV. The bs=1 decode gap lives
entirely at M<=8, where minv pads M up to block_m (default 64) and runs a WMMA tile whose M-extent is
63/64 padding.

Three paths compared:
  F.linear          rocBLAS/hipBLASLt        (what the GDN projections use today)
  minv_linear       dense_gemm_rd + M-pad    (what every other unquantized Linear uses today)
  dense_bf16_gemv   gemv_decode_core         (what the LM head already uses; M<=16 only)

Also reports bit-parity between the paths, because minv exists to be M-invariant: swapping a kernel
in at M<=16 is only safe if it does not change results relative to the path used at other M.
"""
import time

import torch
import torch.nn.functional as F

from minisgl.layers.minv import minv_linear

# (name, IN, OUT) — per-rank at TP=2, hidden=2048, head_dim=256, 40 layers (30 GDN + 10 full-attn).
SHAPES = [
    ("gdn_in_proj_qkvz", 2048, 6144, 30),   # 45% of decode bytes — currently plain nn.Linear
    ("gdn_out_proj",     2048, 2048, 30),
    ("attn_q_proj",      2048, 4096, 10),   # q + gate interleaved
    ("attn_k_proj",      2048,  256, 10),
    ("attn_v_proj",      2048,  256, 10),
    ("attn_o_proj",      2048, 2048, 10),
    ("shared_gate_up",   2048,  512, 40),
    ("shared_down",       256, 2048, 40),
    ("router_gate",      2048,  256, 40),
    ("lm_head",          2048, 124160, 1),
]
MS = [1, 2, 4, 8, 16]


def bench(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e6)
    ts.sort()
    return ts[len(ts) // 2]


def gbps(IN, OUT, us):
    """Weight-streaming bandwidth: the weight is the dominant read at decode M."""
    return (IN * OUT * 2) / (us * 1e-6) / 1e9


def main():
    dev = "cuda"
    try:
        from fp8_wmma import dense_bf16_gemv
    except Exception as e:  # noqa: BLE001
        print(f"!! dense_bf16_gemv unavailable ({e}) — GEMV column skipped")
        dense_bf16_gemv = None

    print(f"{'shape':<18}{'M':>3} {'IN':>6}{'OUT':>8} | "
          f"{'F.linear':>9}{'GB/s':>7} | {'minv64':>9}{'GB/s':>7} | "
          f"{'minv16':>9}{'GB/s':>7} | {'gemv':>9}{'GB/s':>7} | {'gemv/minv64':>12}")
    print("-" * 132)

    totals = {"flin": 0.0, "minv64": 0.0, "minv16": 0.0, "gemv": 0.0}
    for name, IN, OUT, nlayers in SHAPES:
        W = (torch.randn(OUT, IN, device=dev) * 0.02).to(torch.bfloat16)
        for M in MS:
            x = (torch.randn(M, IN, device=dev) * 1.0).to(torch.bfloat16)

            t_fl = bench(lambda: F.linear(x, W))
            t_m64 = bench(lambda: minv_linear(x, W, block_m=64, BN=64))
            t_m16 = bench(lambda: minv_linear(x, W, block_m=16, BN=64))
            t_gv = float("nan")
            if dense_bf16_gemv is not None and M <= 16:
                try:
                    t_gv = bench(lambda: dense_bf16_gemv(x.contiguous(), W))
                except Exception:
                    t_gv = float("nan")

            ratio = (t_m64 / t_gv) if t_gv == t_gv and t_gv > 0 else float("nan")
            print(f"{name:<18}{M:>3} {IN:>6}{OUT:>8} | "
                  f"{t_fl:>9.1f}{gbps(IN,OUT,t_fl):>7.0f} | {t_m64:>9.1f}{gbps(IN,OUT,t_m64):>7.0f} | "
                  f"{t_m16:>9.1f}{gbps(IN,OUT,t_m16):>7.0f} | {t_gv:>9.1f}{gbps(IN,OUT,t_gv):>7.0f} | "
                  f"{ratio:>11.2f}x")

            if M == 1:  # per-step model cost at bs=1 decode: every layer of this kind, once
                totals["flin"] += t_fl * nlayers
                totals["minv64"] += t_m64 * nlayers
                totals["minv16"] += t_m16 * nlayers
                totals["gemv"] += (t_gv if t_gv == t_gv else t_m64) * nlayers
        print()

    print("=" * 132)
    print("MODELLED bs=1 per-step cost of the bf16 dense linears (sum over all layers, M=1):")
    for k in ("flin", "minv64", "minv16", "gemv"):
        print(f"  {k:<8} {totals[k]/1000:8.2f} ms/step")
    print("  (shipped config = minv64 for everything except gdn_* which is F.linear, and lm_head=gemv)")

    # ---- bit-parity: is the GEMV interchangeable with the minv kernels? ----
    print()
    print("BIT-PARITY (does swapping the kernel change the result?) — minv exists to be M-invariant,")
    print("so a kernel swap at M<=16 is only safe if it is bit-identical to the path used at other M.")
    if dense_bf16_gemv is not None:
        for name, IN, OUT, _ in SHAPES[:4]:
            W = (torch.randn(OUT, IN, device=dev) * 0.02).to(torch.bfloat16)
            x = (torch.randn(8, IN, device=dev)).to(torch.bfloat16)
            a = minv_linear(x, W, block_m=64, BN=64)
            b = minv_linear(x, W, block_m=16, BN=64)
            c = dense_bf16_gemv(x.contiguous(), W)
            print(f"  {name:<18} minv64-vs-minv16 max|d|={(a-b).abs().max().item():.3e}  "
                  f"minv64-vs-gemv max|d|={(a-c).abs().max().item():.3e}  "
                  f"exact={'YES' if torch.equal(a, c) else 'NO'}")


if __name__ == "__main__":
    main()
