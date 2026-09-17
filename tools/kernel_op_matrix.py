"""Op-level performance AND COVERAGE matrix for the canonical gfx1201 (RDNA4) HIP kernels.

Two jobs, and the second is why this file exists in its current shape:

  1. PERF — per-kernel achieved throughput (TFLOP/s compute-bound, GB/s bandwidth-bound), % of the
     empirical roofline, and speedup vs the vendor baseline (rocBLAS/hipBLASLt via torch; torch-native
     for elementwise). Prefill (compute-bound, large M) AND decode (M<=16, bandwidth-bound) regimes.

  2. COVERAGE — it enumerates the served (weight format x group_size) space and reports, per row,
     WHICH HIP ARM ACTUALLY FIRED (from the `engaged()` ledger) plus the ENGINE-SIDE GATE verdict.
     A format that is missing an arm, that silently falls back to a slower one, or that the kernel
     accepts while an engine predicate blocks it, shows up here as a gap.

     This half was added 2026-08-18 after NVFP4 (e2m1 at group_size=16) was found to have been
     serving for weeks with (a) the serial per-nibble E2M1 decode long after a bit-exact v_perm
     decoder existed, and (b) NO fused gate_up+silu at all — the kernel's gate was relaxed to
     `group_size % 16` on 2026-07-23 explicitly for NVFP4, while the engine's `_fused_swiglu_ok`
     still required `% 32`, so nothing ever called it. The old version of this file benched ONE
     hardcoded group_size and had no e2m1 arm, so the one tool whose job was to enumerate the op
     space could not have seen either. See rdna4-hip-kernels/KERNEL_CORE_POLICY.md RULE 5.

Rooflines are MEASURED on THIS card (not spec-sheet): the bf16 hipBLASLt peak = practical compute
ceiling; a large-tensor stream = HBM bandwidth ceiling. Triton baselines need the (Triton-bearing)
combined image and are produced by the companion --triton pass; this file is the HIP-vs-rocBLAS
in-process matrix.

  gpu-lease -n 1 -- python this.py       (PYTHONPATH=<fusion gdn+w4a8>:/opt/kernels:/engine/python:/engine)
"""
import sys
import time
import torch
import torch.nn.functional as F

sys.path.insert(0, "/engine/python"); sys.path.insert(0, "/engine")
from minisgl.distributed import set_tp_info      # noqa: E402
set_tp_info(0, 1)                                # single-process TP=1 (engaged() logger needs it)
from minisgl import _hip_engage as ENGAGE        # noqa: E402
from minisgl.quant import kernels as K           # noqa: E402
from minisgl.quant import method as QM           # noqa: E402
from minisgl.layers.minv import minv_linear      # noqa: E402
import tail_hip                                  # noqa: E402

# The arm column is read out of engaged()'s dedup set, which is ONLY populated when the log is on.
# Assert rather than silently print an empty column — a blank coverage column is the exact failure
# mode this file exists to catch.
assert ENGAGE._ON, "MINISGL_HIP_ENGAGE_LOG=0 blanks the arm ledger; unset it before running this"

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


def arms_of(fn):
    """The HIP arms one call engages. engaged() is one-shot per name, so the ledger is cleared
    first — that is what makes a SILENT fallback a table column instead of an invisible deficit."""
    ENGAGE._seen.clear()
    fn()
    torch.cuda.synchronize()
    return sorted(ENGAGE._seen)


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


# ---------------------------------------------------------------- the configuration space
# One entry per (weight format x group_size) the engine can actually SERVE today. Enumerating them
# is the point: a config that is absent from this list is a config nothing measures.
#
#   int4 g128  GLM-4.7-Flash-AWQ, Qwen3.8-27B-AWQ-INT4 at g=128 checkpoints
#   int4 g32   the served Qwen-AWQ family (compressed-tensors pack-quantized, asymmetric)
#   e2m1 g32   MXFP4 (E8M0 block exponent folded to an fp16 group scale at load)
#   e2m1 g16   NVFP4 (e4m3 block scale x per-tensor global, folded to fp16 at load) — Laguna,
#              Muse-Glimmer, Qwen3.8-27B-MTP-NVFP4. The narrowest group in the fleet, and the one
#              every group_size-sensitive cost (scale traffic, k_sub, LDS pad) is worst at.
CONFIGS = [
    # label            fmt     group  zeros(asym)
    ("bf16",          "bf16",     0,  False),
    ("int4 g128",     "int4",   128,  True),
    ("int4 g32",      "int4",    32,  True),
    ("e2m1 g32",      "e2m1",    32,  False),   # MXFP4 — symmetric, no zero-points
    ("e2m1 g16",      "e2m1",    16,  False),   # NVFP4
    ("fp8 w8a8",      "fp8",      0,  False),
]

# M=1 and M=16 bracket the decode band (both _W4A8_GEMV_MAX_INT4 and _W4A8_GEMV_MAX_E2M1 are 16, so
# M=16 is the top of the GEMV band and M>16 is the wmma_tiled_tuned reroute). 2048/4096 = mid and
# long-context prefill.
M_BANDS = (4096, 2048, 16, 1)

ROWS = []       # perf table
COVER = []      # (row label, arms fired, note)


def wbytes_per_elem(fmt, group, zeros):
    """Weight-side bytes per weight ELEMENT, INCLUDING the per-group scale (and packed zero-point).
    Counting scales is not pedantry: at group 16 the fp16 scale is 2B/16 = 0.125 B/elem on top of a
    0.5 B/elem weight, i.e. a 25% tax that a weights-only figure hides — and hiding it flatters
    exactly the config that pays it most."""
    if fmt == "bf16":
        return 2.0
    if fmt == "fp8":
        return 1.0                                   # + a per-CHANNEL scale, negligible
    b = 0.5 + 2.0 / group                            # 4-bit weight + fp16 group scale
    if zeros:
        b += 0.5 / group                             # (K/g, N/8) int32 packed zero-points
    return b


def gemm_row(name, cfg, M, N, Kk, hip_fn, base_fn, tflops_peak, gbs_peak, wbpe):
    """Prefill (M>16) -> compute-bound: TFLOP/s + %compute-roof. Decode band (M<=16) ->
    bandwidth-bound (weight-read dominated): weight+scale GB/s + %HBM-roof. Both vs rocBLAS."""
    arms = arms_of(hip_fn)
    hu = bench(hip_fn)
    bu = bench(base_fn)
    if M > 16:
        flop = 2 * M * N * Kk
        ht = flop / (hu * 1e-6) / 1e12
        bt = flop / (bu * 1e-6) / 1e12
        ROWS.append((name, cfg, f"{M}x{N}x{Kk}", f"{hu:.1f}", f"{ht:.1f} TF/s",
                     f"{100*ht/tflops_peak:.0f}%", f"{bu:.1f}", f"{bt:.1f} TF/s", f"{bu/hu:.2f}x"))
    else:
        wb = N * Kk * wbpe                            # weight+scale bytes (dominates the decode band)
        bb = N * Kk * 2                               # rocBLAS reads bf16 weights
        hg = wb / (hu * 1e-6) / 1e9
        bg = bb / (bu * 1e-6) / 1e9
        ROWS.append((name, cfg, f"{M}x{N}x{Kk}", f"{hu:.1f}", f"{hg:.0f} GB/s",
                     f"{100*hg/gbs_peak:.0f}%", f"{bu:.1f}", f"{bg:.0f} GB/s", f"{bu/hu:.2f}x"))
    COVER.append((f"{name} [{cfg}]", arms, ""))


def gap(name, cfg, why):
    """Record a config the op cannot serve. A recorded exclusion is information; a skipped row that
    nobody notices is how NVFP4 lost a fusion for a month."""
    ROWS.append((name, cfg, "-", "-", "GAP", "-", "-", "-", "-"))
    COVER.append((f"{name} [{cfg}]", [], why))


def elt_row(name, bytes_moved, hip_fn, base_fn, gbs_peak):
    arms = arms_of(hip_fn)
    hu = bench(hip_fn)
    bu = bench(base_fn)
    hg = bytes_moved / (hu * 1e-6) / 1e9
    bg = bytes_moved / (bu * 1e-6) / 1e9
    ROWS.append((name, "-", "-", f"{hu:.1f}", f"{hg:.0f} GB/s", f"{100*hg/gbs_peak:.0f}%",
                 f"{bu:.1f}", f"{bg:.0f} GB/s", f"{bu/hu:.2f}x"))
    COVER.append((name, arms, ""))


def build_weights(fmt, group, zeros, N, Kk):
    """Op-layout weights for one config. Scales are GROUP-MAJOR (K/group, N) and packed zeros are
    (K/group, N/8) — the layout the bindings have asserted since rdna4-hip-kernels abcbb09
    (2026-08-07), where the output channel is the contiguous axis so the 16-lane fragment read is
    ONE request. NOTE: several engine docstrings still describe the pre-abcbb09 (N, K/group), which
    is what the previous revision of this file built — and the binding rejects it outright."""
    if fmt == "bf16":
        return {"w": torch.randn(N, Kk, device=DEV, dtype=torch.bfloat16) * 0.05}
    if fmt == "fp8":
        wf8 = (torch.randn(N, Kk, device=DEV) * 0.05).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
        s8 = torch.randn(N, device=DEV, dtype=torch.float32).abs() * 0.02 + 0.001
        return {"w": wf8, "s": s8}
    # int4 / e2m1 share the packed-nibble container; only the DECODE of the nibble differs, which is
    # the whole point of the shared WLoad core.
    codes = torch.randint(0, 16, (N, Kk), dtype=torch.int8, device=DEV)
    w4 = pack_uint4_2d(codes)
    sc = torch.randn(Kk // group, N, device=DEV, dtype=torch.float16).abs() * 0.02 + 0.001
    # zp=8 in every nibble: the VALUE is perf-irrelevant, the LOAD is not — an asym config must
    # exercise the zero-point fetch or it under-reports int4 against symmetric e2m1. -0x77777778 is
    # 0x88888888 reinterpreted as a signed int32 (the packed container's dtype).
    wz = torch.full((Kk // group, N // 8), -0x77777778, dtype=torch.int32,
                    device=DEV) if zeros else None
    return {"w": w4, "s": sc, "z": wz}


def main():
    if not torch.cuda.is_available():
        print("no HIP device"); sys.exit(1)
    print(f"Device: {torch.cuda.get_device_name(0)}\n")
    tflops_peak, gbs_peak = rooflines()

    import fp8_wmma as W4  # merged package: W4A8 int4/e2m1 + W8A8 fp8 (torch.ops.fp8_wmma_C)
    N = Kk = 4096

    W = {label: build_weights(fmt, g, z, N, Kk) for label, fmt, g, z in CONFIGS}

    # ============ GEMM family, swept over the FULL (format x group) space ==================
    # Each vs its RIGHT baseline: bf16 -> rocBLAS bf16; fp8 -> hipBLASLt fp8 (_scaled_mm); 4-bit has
    # no vendor path -> vs rocBLAS bf16 (the dtype-native cost of the same GEMM).
    for M in M_BANDS:
        tag = f"prefill{M}" if M > 16 else f"decode{M}"
        x_bf = torch.randn(M, Kk, device=DEV, dtype=torch.bfloat16) * 0.1
        x_f16 = x_bf.to(torch.float16)
        w_bf = W["bf16"]["w"]
        rocblas_bf16 = lambda: F.linear(x_bf, w_bf)  # noqa: E731

        for label, fmt, g, z in CONFIGS:
            d = W[label]
            wbpe = wbytes_per_elem(fmt, g, z)

            if fmt == "bf16":
                gemm_row(f"dense_gemm bf16 [{tag}]", label, M, N, Kk,
                         lambda: minv_linear(x_bf, w_bf, None), rocblas_bf16,
                         tflops_peak, gbs_peak, wbpe)
                continue

            if fmt == "fp8":
                xf8 = (x_bf / (x_bf.abs().amax().clamp(min=1e-4) / E4M3_MAX)).to(torch.float8_e4m3fn)
                sc1 = torch.ones(1, device=DEV)
                def rocblas_fp8():  # noqa: E731
                    return torch._scaled_mm(xf8, d["w"].t(), scale_a=sc1, scale_b=sc1,
                                            out_dtype=torch.bfloat16)
                gemm_row(f"w8a8_fp8_wmma fp8 [{tag}]", label, M, N, Kk,
                         lambda: K.w8a8_dense_linear(x_bf, d["w"].view(torch.uint8), d["s"]),
                         rocblas_fp8, tflops_peak, gbs_peak, wbpe)
                continue

            e2m1 = fmt == "e2m1"
            # W4A8 (mmq_fp8_gemm) — the DEFAULT served dense path for every 4-bit checkpoint.
            gemm_row(f"w4a8_fp8_wmma [{tag}]", label, M, N, Kk,
                     lambda: K.w4a8_linear(x_bf, d["w"], d["s"], d.get("z"), g,
                                           weight_is_e2m1=e2m1),
                     rocblas_bf16, tflops_peak, gbs_peak, wbpe)

            # W4A16 — unquantized activations, the MINISGL_MOE_W4A16 path. NO GROUP EXCLUSION any
            # more: this used to be the register-direct wide arm, which needed group_size % 32 for
            # its b128 wide load and so recorded a gap at g=16 (NVFP4's own group). `w4a16_linear`
            # now dispatches a decode GEMV below M=16 and the tiled A16 core above it, both on the
            # SAME `w_packed` this row already holds and both carrying a runtime group size.
            gemm_row(f"w4a16 [{tag}]", label, M, N, Kk,
                     lambda: K.w4a16_linear(x_f16, d["w"], d["s"], d.get("z"), g, N),
                     rocblas_bf16, tflops_peak, gbs_peak, wbpe)

    # ============ Fused gate_up + SiLU (decode-only) — kernel arm AND engine gate ===========
    # The op takes a [gate|up] weight of (2*inter, K); with N=4096 that is inter=2048. Measured
    # against the unfused pair it replaces, so the row states what the fusion is WORTH.
    #
    # The `engine gate` column is the load-bearing one: it evaluates the real predicate the serve
    # path uses (quant.method._fused_swiglu_ok). A row that is FAST and BLOCKED is a fusion the
    # kernel supports and no model can reach — precisely the NVFP4 case, where the kernel accepts
    # group_size % 16 and the engine predicate demands % 32.
    print("\nFUSED gate_up+silu (mmq_fp8_gemm_silu) — decode band only\n" + "-" * 96)
    print(f"{'config':12}{'M':>4}{'fused us':>11}{'unfused us':>12}{'speedup':>10}"
          f"{'engine gate':>14}   arms")
    for label, fmt, g, z in CONFIGS:
        if fmt not in ("int4", "e2m1"):
            continue
        d, e2m1 = W[label], fmt == "e2m1"
        for M in (16, 1):
            x_bf = torch.randn(M, Kk, device=DEV, dtype=torch.bfloat16) * 0.1
            fused = lambda: K.w4a8_linear_silu(x_bf, d["w"], d["s"], d.get("z"), g,
                                               weight_is_e2m1=e2m1)
            unfused = lambda: tail_hip.silu_and_mul(
                K.w4a8_linear(x_bf, d["w"], d["s"], d.get("z"), g, weight_is_e2m1=e2m1))
            gate = "ok" if QM._fused_swiglu_ok(x_bf, d["w"], g) else "BLOCKED"
            try:
                arms = arms_of(fused)
                fu, uu = bench(fused), bench(unfused)
                print(f"{label:12}{M:>4}{fu:>11.1f}{uu:>12.1f}{uu/fu:>9.2f}x{gate:>14}   "
                      f"{','.join(a.split('.')[-1] for a in arms)}")
                COVER.append((f"w4a8_linear_silu [{label} M={M}]", arms,
                              "" if gate == "ok" else "engine gate _fused_swiglu_ok BLOCKS this config"))
            except Exception as e:                      # kernel refuses the config outright
                print(f"{label:12}{M:>4}{'-':>11}{'-':>12}{'-':>10}{gate:>14}   REFUSED: {e}")
                COVER.append((f"w4a8_linear_silu [{label} M={M}]", [], f"kernel refused: {e}"))

    # ============ Elementwise (bandwidth-bound) vs torch =================================
    # 16384x4096 bf16 = 128 MiB >> L2, so reads hit HBM (defeats the tight-loop cache-reuse artifact).
    Me, Nn = 16384, 4096
    xe = torch.randn(Me, Nn, device=DEV, dtype=torch.bfloat16)
    we = torch.randn(Nn, device=DEV, dtype=torch.bfloat16)
    # rms_norm: read x + write y (weight negligible) = 2*M*N*2 bytes
    elt_row("tail_hip.rms_norm", 2 * Me * Nn * 2,
            lambda: tail_hip.rms_norm(xe, we, 1e-6, True),
            lambda: (xe * torch.rsqrt(xe.float().pow(2).mean(-1, keepdim=True) + 1e-6)).to(xe.dtype) * we,
            gbs_peak)
    # silu_and_mul: read 2*M*N (gate|up) + write M*N = 3*M*N*2 bytes
    xg = torch.randn(Me, 2 * Nn, device=DEV, dtype=torch.bfloat16)
    elt_row("tail_hip.silu_and_mul", 3 * Me * Nn * 2,
            lambda: tail_hip.silu_and_mul(xg),
            lambda: F.silu(xg[:, :Nn].float()).mul(xg[:, Nn:].float()).to(xg.dtype),
            gbs_peak)

    # ---------------------------------------------------------------- report
    hdr = ("kernel", "config", "MxNxK", "HIP us", "HIP TFLOP/s|BW", "%roof", "base us",
           "base T/s|BW", "vs base")
    print(f"\n{hdr[0]:28}{hdr[1]:>11}{hdr[2]:>16}{hdr[3]:>9}{hdr[4]:>16}{hdr[5]:>7}"
          f"{hdr[6]:>10}{hdr[7]:>14}{hdr[8]:>10}")
    print("-" * 121)
    for r in ROWS:
        print(f"{r[0]:28}{r[1]:>11}{r[2]:>16}{r[3]:>9}{r[4]:>16}{r[5]:>7}{r[6]:>10}{r[7]:>14}{r[8]:>10}")

    print("\nARM LEDGER — which HIP arm each row actually engaged (from engaged()).")
    print("An empty arm list on a row that produced a number = the op fell back OFF the HIP path.")
    print("-" * 121)
    for name, arms, note in COVER:
        shown = ", ".join(a.split(".")[-1] for a in arms) if arms else "(none)"
        print(f"{name:52} {shown}")
        if note:
            print(f"{'':52} ^^ {note}")

    print("\nNotes: 4-bit/fp8 'vs base' compares against bf16 hipBLASLt (there is no int4/fp8 rocBLAS "
          "path); the speedup reflects the quant kernel doing the same GEMM at lower precision/bytes.\n"
          "Decode-band GB/s counts weight + per-group scale (+ packed zero-point) bytes, so the "
          "group-16 scale tax is visible rather than hidden.")


if __name__ == "__main__":
    main()
