"""Parity: FUSED mmq_fp8_gemm_silu  ==  UNFUSED mmq_fp8_gemm(decode_gemv) + silu_and_mul (dense W4A8).

The fused dense kernel (mmq_fp8_gemv_decode_silu) reuses the unfused decode-gemv's exact accumulation AND
the moe_silu_and_mul_h epilogue (replicated as w4a8_silu_and_mul_h), so vs (unfused gemv -> (M,2*inter))
followed by that same canonical silu it should be BIT-EXACT. Gate against the w4a8_silu_and_mul_h replica
applied to the unfused out1 (isolates the fusion; any deviation is a real gemv/index bug, not silu rounding).
Decode matrix: M in {1,2,4}, int4 SYM/ASYM + e2m1, fp16/bf16.  gpu-lease -n 1 -- python this.py
"""
import sys
import torch
import torch.nn.functional as F

import w4a8_fp8_wmma as W  # noqa: E402
import tail_hip  # noqa: E402

DEV = torch.device("cuda:0")
torch.manual_seed(0)
FAILS = []
print(f"w4a8 from {W.__file__}")


def pack_uint4_2d(w):  # (N,K) int8 -> (N,K/8) int32
    N, K = w.shape
    w = w.to(torch.int32)
    packed = torch.zeros((N, K // 8), dtype=torch.int32, device=w.device)
    for i in range(8):
        packed |= (w[:, i::8] & 0xF) << (i * 4)
    return packed


def pack_zeros_2d(z):  # (N,G) int8 -> (N/8,G) int32
    N, G = z.shape
    z = z.to(torch.int32)
    packed = torch.zeros((N // 8, G), dtype=torch.int32, device=z.device)
    for i in range(8):
        packed |= (z[i::8, :] & 0xF) << (i * 4)
    return packed


def check(M, inter, K, g, asym, e2m1, dtype):
    N = 2 * inter
    x = torch.randn(M, K, dtype=dtype, device=DEV) * 0.3
    w_packed = pack_uint4_2d(torch.randint(0, 16, (N, K), dtype=torch.int8, device=DEV))
    G = K // g
    scales = torch.randn(N, G, dtype=torch.float16, device=DEV).abs() * 0.02 + 0.001
    zp_packed = None
    if asym:
        zp_packed = pack_zeros_2d(torch.randint(0, 16, (N, G), dtype=torch.int8, device=DEV))

    out1 = W.mmq_fp8_gemm(x, w_packed, scales, kernel="decode_gemv", w_zeros=zp_packed,
                          weight_is_e2m1=e2m1)                     # (M, 2*inter)
    g_h, u_h = out1[:, :inter], out1[:, inter:]
    silu_h = (g_h.float() / (1.0 + torch.exp(-g_h.float()))).to(dtype)
    ref_silu_h = silu_h * u_h                                      # w4a8_silu_and_mul_h replica
    buf_tail = tail_hip.silu_and_mul(out1.contiguous())
    buf_torch = (F.silu(g_h.float()) * u_h.float()).to(dtype)

    fused = W.mmq_fp8_gemm_silu(x, w_packed, scales, w_zeros=zp_packed, weight_is_e2m1=e2m1)  # (M, inter)

    tag = f"M={M} inter={inter} K={K} g={g} {'ASYM' if asym else 'SYM'}" \
          f"{'/e2m1' if e2m1 else ''} {str(dtype).split('.')[-1]}"
    a = fused.float()
    for name, ref, gate in (("vs silu_h replica", ref_silu_h, True),
                            ("vs tail_hip.silu", buf_tail, False),
                            ("vs torch.silu", buf_torch, False)):
        b = ref.float()
        diff = (a - b).abs()
        mx = diff.max().item()
        mean_rel = (diff / b.abs().clamp(min=1e-4)).mean().item()
        if gate:
            ok = (mx == 0.0) and (mean_rel == 0.0)
            flag = "PASS" if ok else "FAIL"
            if not ok:
                FAILS.append(f"{tag} {name}")
        else:
            flag = "info"
        print(f"  [{flag}] {tag:<44} {name:<18} max={mx:.3e} mean_rel={mean_rel:.2e} |ref|={b.abs().mean():.4f}")


def main():
    if not torch.cuda.is_available():
        print("FAIL: no CUDA/HIP device"); sys.exit(1)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print("=== dense mmq_fp8_gemm_silu FUSED parity (decode: M<=16, gemv) ===")
    shapes = [
        (1, 256, 512, 128),
        (2, 512, 1024, 128),
        (4, 768, 2048, 128),
        (1, 512, 1536, 128),   # K=1536 (GLM shared-expert-ish, %512==0)
        (1, 384, 1024, 32),    # group=32
    ]
    for dtype in (torch.float16, torch.bfloat16):
        for (M, inter, K, g) in shapes:
            check(M, inter, K, g, asym=False, e2m1=False, dtype=dtype)
            check(M, inter, K, g, asym=True, e2m1=False, dtype=dtype)
            check(M, inter, K, g, asym=False, e2m1=True, dtype=dtype)
    print()
    if FAILS:
        print(f"FAILED ({len(FAILS)}):")
        for f in FAILS:
            print("   ", f)
        sys.exit(1)
    print("ALL PARITY PASS — dense fused gemm+silu BIT-EXACT vs unfused gemv + silu")


if __name__ == "__main__":
    main()
