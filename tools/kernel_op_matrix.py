"""Op-level performance matrix for the canonical gfx1201 (RDNA4) HIP kernels: per-kernel achieved
throughput (TFLOP/s compute-bound, GB/s bandwidth-bound), % of the empirical roofline, and speedup vs
the vendor baseline (rocBLAS/hipBLASLt for GEMMs via torch; torch-native for elementwise). Prefill
(compute-bound, large M) AND decode (M=1, latency/bandwidth-bound) regimes.

Rooflines are MEASURED on THIS card (not spec-sheet): the bf16 hipBLASLt peak = practical compute ceiling;
a large-tensor stream = HBM bandwidth ceiling. Triton baselines need the (Triton-bearing) combined image
and are produced by the companion --triton pass; this file is the HIP-vs-rocBLAS in-process matrix.

  gpu-lease -n 1 -- python this.py       (PYTHONPATH=<fusion gdn+w4a8>:/opt/kernels:/engine/python:/engine)
"""
import sys
import time
import torch
import torch.nn.functional as F

sys.path.insert(0, "/engine/python"); sys.path.insert(0, "/engine")
from minisgl.distributed import set_tp_info      # noqa: E402
set_tp_info(0, 1)                                # single-process TP=1 (engaged() logger needs it)
from minisgl.quant import kernels as K          # noqa: E402
from minisgl.layers.minv import minv_linear     # noqa: E402
import tail_hip                                  # noqa: E402

DEV = torch.device("cuda:0")
torch.manual_seed(0)
E4M3_MAX = 448.0


def bench(fn, n=300, warm=50):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e6  # us/call


def pack_uint4_2d(w):
    N, Kk = w.shape
    w = w.to(torch.int32)
    p = torch.zeros((N, Kk // 8), dtype=torch.int32, device=w.device)
    for i in range(8):
        p |= (w[:, i::8] & 0xF) << (i * 4)
    return p


# ---------------------------------------------------------------- rooflines (measured on-card)
def rooflines():
    a = torch.randn(8192, 8192, device=DEV, dtype=torch.bfloat16)
    b = torch.randn(8192, 8192, device=DEV, dtype=torch.bfloat16)
    us = bench(lambda: torch.matmul(a, b), n=50, warm=20)
    tflops_peak = 2 * 8192**3 / (us * 1e-6) / 1e12
    # HBM: read+write a big tensor (copy). bytes = 2 * N * 2(bf16).
    x = torch.randn(1 << 26, device=DEV, dtype=torch.bfloat16)  # 64Mi elems = 128 MiB
    us_c = bench(lambda: x.mul(1.0001), n=100, warm=30)
    gbs_peak = (2 * x.numel() * 2) / (us_c * 1e-6) / 1e9
    print(f"ROOFLINE (measured): bf16 hipBLASLt peak = {tflops_peak:.1f} TFLOP/s   "
          f"HBM bandwidth = {gbs_peak:.0f} GB/s\n")
    return tflops_peak, gbs_peak


ROWS = []


def gemm_row(name, M, N, Kk, hip_fn, base_fn, tflops_peak, gbs_peak, wbytes_per_elem):
    """Prefill (M large) -> compute-bound: report TFLOP/s + %compute-roof. Decode (M==1) ->
    bandwidth-bound (weight-read dominated): report weight GB/s + %HBM-roof. Both vs rocBLAS."""
    hu = bench(hip_fn)
    bu = bench(base_fn)
    if M > 1:
        flop = 2 * M * N * Kk
        ht = flop / (hu * 1e-6) / 1e12
        bt = flop / (bu * 1e-6) / 1e12
        ROWS.append((name, f"{M}x{N}x{Kk}", f"{hu:.1f}", f"{ht:.1f} TF/s", f"{100*ht/tflops_peak:.0f}%",
                     f"{bu:.1f}", f"{bt:.1f} TF/s", f"{bu/hu:.2f}x"))
    else:
        wb = N * Kk * wbytes_per_elem                 # weight bytes read (dominates decode GEMV)
        bb = N * Kk * 2                                # rocBLAS reads bf16 weights
        hg = wb / (hu * 1e-6) / 1e9
        bg = bb / (bu * 1e-6) / 1e9
        ROWS.append((name, f"1x{N}x{Kk}", f"{hu:.1f}", f"{hg:.0f} GB/s", f"{100*hg/gbs_peak:.0f}%",
                     f"{bu:.1f}", f"{bg:.0f} GB/s", f"{bu/hu:.2f}x"))


def elt_row(name, bytes_moved, hip_fn, base_fn, gbs_peak):
    hu = bench(hip_fn)
    bu = bench(base_fn)
    hg = bytes_moved / (hu * 1e-6) / 1e9
    bg = bytes_moved / (bu * 1e-6) / 1e9
    ROWS.append((name, "-", f"{hu:.1f}", f"{hg:.0f} GB/s", f"{100*hg/gbs_peak:.0f}%",
                 f"{bu:.1f}", f"{bg:.0f} GB/s", f"{bu/hu:.2f}x"))


def main():
    if not torch.cuda.is_available():
        print("no HIP device"); sys.exit(1)
    print(f"Device: {torch.cuda.get_device_name(0)}\n")
    tflops_peak, gbs_peak = rooflines()

    # ============ GEMM family: dense bf16 / W4A16 regdirect / W4A8 int4 / W8A8 fp8 =========
    # Prefill-shaped (compute-bound) + decode (M=1). N=K=4096. Each vs its RIGHT baseline: bf16->rocBLAS
    # bf16; fp8->hipBLASLt fp8 (_scaled_mm). int4 has no vendor path -> vs rocBLAS bf16 (dtype-native cost).
    import fp8_wmma as W4  # merged package: W4A8 int4/e2m1 + W8A8 fp8 (torch.ops.fp8_wmma_C)
    N = Kk = 4096
    g = 128
    w_int4 = torch.randint(0, 16, (N, Kk), dtype=torch.int8, device=DEV)
    w4 = pack_uint4_2d(w_int4)
    s4 = torch.randn(N, Kk // g, device=DEV, dtype=torch.float16).abs() * 0.02 + 0.001
    wide = K._w4a16_wide(g)
    w_rep = W4.repack_int4_to_w_rep(w4, N, Kk)
    w_rep_wide = W4.repack_w_rep_wide(w_rep, wide)
    wf8 = (torch.randn(N, Kk, device=DEV) * 0.05).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    s8 = torch.randn(N, device=DEV, dtype=torch.float32).abs() * 0.02 + 0.001
    # M=4096 = the large-M / long-context prefill regime (the wmma_tiled_tuned reroute path, and on
    # the served MoE the block_m==128 flag regime); M=2048 mid prefill; M=1 decode (bandwidth-bound).
    for M in (4096, 2048, 1):
        tag = f"prefill{M}" if M > 1 else "decode"
        x_bf = torch.randn(M, Kk, device=DEV, dtype=torch.bfloat16) * 0.1
        x_f16 = x_bf.to(torch.float16)
        w_bf = torch.randn(N, Kk, device=DEV, dtype=torch.bfloat16) * 0.05
        rocblas_bf16 = lambda: F.linear(x_bf, w_bf)  # noqa: E731

        gemm_row(f"dense_gemm bf16 [{tag}]", M, N, Kk,
                 lambda: minv_linear(x_bf, w_bf, None), rocblas_bf16, tflops_peak, gbs_peak, 2)
        # W4A16 register-direct (mmq_regdirect_w4a16_wide) — fp16 act direct, the served AWQ-W4A16 path
        gemm_row(f"w4a16 regdirect int4 [{tag}]", M, N, Kk,
                 lambda: K.w4a16_linear(x_f16, w_rep_wide, s4, None, g, N), rocblas_bf16,
                 tflops_peak, gbs_peak, 0.5)
        # W4A8 int4xfp8 (mmq_fp8_gemm) — the DEFAULT served dense path (decode-tuned)
        gemm_row(f"w4a8_fp8_wmma int4 [{tag}]", M, N, Kk,
                 lambda: K.w4a8_linear(x_bf, w4, s4, None, g), rocblas_bf16, tflops_peak, gbs_peak, 0.5)
        # W8A8 fp8xfp8 vs hipBLASLt fp8 (_scaled_mm, the RIGHT vendor baseline for fp8)
        xf8 = (x_bf / (x_bf.abs().amax().clamp(min=1e-4) / E4M3_MAX)).to(torch.float8_e4m3fn)
        sc = torch.ones(1, device=DEV)
        def rocblas_fp8():  # noqa: E731
            return torch._scaled_mm(xf8, wf8.t(), scale_a=sc, scale_b=sc, out_dtype=torch.bfloat16)
        gemm_row(f"w8a8_fp8_wmma fp8 [{tag}]", M, N, Kk,
                 lambda: K.w8a8_dense_linear(x_bf, wf8.view(torch.uint8), s8),
                 rocblas_fp8, tflops_peak, gbs_peak, 1)

    # ============ Elementwise (bandwidth-bound) vs torch =================================
    # 16384x4096 bf16 = 128 MiB >> L2, so reads hit HBM (defeats the tight-loop cache-reuse artifact).
    M, Nn = 16384, 4096
    xe = torch.randn(M, Nn, device=DEV, dtype=torch.bfloat16)
    we = torch.randn(Nn, device=DEV, dtype=torch.bfloat16)
    # rms_norm: read x + write y (weight negligible) = 2*M*N*2 bytes
    elt_row("tail_hip.rms_norm", 2 * M * Nn * 2,
            lambda: tail_hip.rms_norm(xe, we, 1e-6, True),
            lambda: (xe * torch.rsqrt(xe.float().pow(2).mean(-1, keepdim=True) + 1e-6)).to(xe.dtype) * we,
            gbs_peak)
    # silu_and_mul: read 2*M*N (gate|up) + write M*N = 3*M*N*2 bytes
    xg = torch.randn(M, 2 * Nn, device=DEV, dtype=torch.bfloat16)
    elt_row("tail_hip.silu_and_mul", 3 * M * Nn * 2,
            lambda: tail_hip.silu_and_mul(xg),
            lambda: F.silu(xg[:, :Nn].float()).mul(xg[:, Nn:].float()).to(xg.dtype),
            gbs_peak)

    # ---------------------------------------------------------------- report
    hdr = ("kernel", "MxNxK", "HIP us", "HIP TFLOP/s|BW", "%roof", "rocBLAS us", "base T/s|BW", "vs rocBLAS")
    print(f"{hdr[0]:30}{hdr[1]:>13}{hdr[2]:>9}{hdr[3]:>16}{hdr[4]:>7}{hdr[5]:>12}{hdr[6]:>14}{hdr[7]:>12}")
    print("-" * 113)
    for r in ROWS:
        print(f"{r[0]:30}{r[1]:>13}{r[2]:>9}{r[3]:>16}{r[4]:>7}{r[5]:>12}{r[6]:>14}{r[7]:>12}")
    print("\nNotes: W4A8/W8A8 'vs rocBLAS' compares against bf16 hipBLASLt (there is no int4/fp8 rocBLAS "
          "path); the speedup reflects the quant kernel doing the same GEMM at lower precision/bytes.")


if __name__ == "__main__":
    main()
