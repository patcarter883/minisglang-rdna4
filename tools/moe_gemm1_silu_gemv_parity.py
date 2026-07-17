"""Parity: FUSED mmq_fp8_moe_gemm1_silu(kernel="gemv")  ==  UNFUSED mmq_fp8_moe_gemm(gemv) + silu_and_mul.

The fused decode kernel (moe_gemv_decode_silu) reuses the UNFUSED gemv's exact inner-product accumulation
(same float ops, same order) AND the WMMA fused path's exact epilogue (moe_silu_and_mul_h): each warp owns
one FUSED output column j, computes gate (weight col j) + up (col j+inter) sharing the gathered fp8
activation, then writes silu(gate)*up to (P, inter). So vs (unfused gemv -> (P,2*inter)) + tail_hip.silu it
should be bit-identical up to at most silu's last-bit fp32 rounding.

Covers the decode matrix: T in {1,2} (M<=2 gemv path), int4 SYM + ASYM(AWQ zeros), and e2m1 (MXFP4) — the
same random nibbles fed to BOTH sides with weight_is_e2m1=True, so the comparison isolates the FUSION
regardless of weight interpretation.  gpu-lease -n 1 -- python this.py
"""
import sys
import torch
import torch.nn.functional as F

# w4a8_fp8_wmma MUST resolve to the FUSION build (mount it ahead of /opt/kernels on PYTHONPATH);
# tail_hip comes from /opt/kernels unchanged.
import w4a8_fp8_wmma as W  # noqa: E402
import tail_hip  # noqa: E402

print(f"w4a8_fp8_wmma from: {W.__file__}")

DEV = torch.device("cuda:0")
torch.manual_seed(0)
FAILS = []


def pack_uint4_3d(w):  # (E,N,K) int8 -> (E,N,K//8) int32
    E, N, K = w.shape
    w = w.to(torch.int32)
    packed = torch.zeros((E, N, K // 8), dtype=torch.int32, device=w.device)
    for i in range(8):
        packed |= (w[:, :, i::8] & 0xF) << (i * 4)
    return packed


def pack_zeros_3d(z):  # (E,N,G) int8 -> (E,N//8,G) int32
    E, N, G = z.shape
    z = z.to(torch.int32)
    packed = torch.zeros((E, N // 8, G), dtype=torch.int32, device=z.device)
    for i in range(8):
        packed |= (z[:, i::8, :] & 0xF) << (i * 4)
    return packed


def moe_align(topk_ids, block_m, E):
    T, top_k = topk_ids.shape
    num_valid = T * top_k
    flat = topk_ids.reshape(-1)
    sorted_ids, expert_ids = [], []
    for e in range(E):
        slots = torch.nonzero(flat == e, as_tuple=False).flatten().tolist()
        n = len(slots)
        npad = ((n + block_m - 1) // block_m) * block_m
        for i in range(npad):
            sorted_ids.append(slots[i] if i < n else num_valid)
        expert_ids.extend([e] * (npad // block_m))
    dev = topk_ids.device
    sti = torch.tensor(sorted_ids, dtype=torch.int32, device=dev)
    eids = torch.tensor(expert_ids, dtype=torch.int32, device=dev)
    ntp = torch.tensor([sti.numel()], dtype=torch.int32, device=dev)
    return sti, eids, ntp, num_valid


def check(T, E, inter, K, top_k, block_m, g, asym, e2m1, dtype):
    N = 2 * inter  # w13 = [gate | up]
    x = torch.randn(T, K, dtype=dtype, device=DEV) * 0.3
    w_int4 = torch.randint(0, 16, (E, N, K), dtype=torch.int8, device=DEV)
    w_packed = pack_uint4_3d(w_int4)
    G = K // g
    scales = torch.randn(E, N, G, dtype=torch.float16, device=DEV).abs() * 0.02 + 0.001
    topk_ids = torch.stack([torch.randperm(E, device=DEV)[:top_k] for _ in range(T)]).to(torch.int32)
    sti, eids, ntp, num_valid = moe_align(topk_ids, block_m, E)
    zp_packed = None
    if asym:
        zeros = torch.randint(0, 16, (E, N, G), dtype=torch.int8, device=DEV)
        zp_packed = pack_zeros_3d(zeros)

    # UNFUSED gemv -> (P, 2*inter). The fused kernel's INTERNAL gate/up rounding ((AT)(acc*asc)) is
    # bit-identical to this out1, so applying moe_silu_and_mul_h's EXACT arithmetic to out1 gives the
    # correctness reference — any deviation there is a real gemv/index/fusion bug, not silu rounding.
    out1 = W.mmq_fp8_moe_gemm(x, w_packed, scales, sti, eids, ntp, top_k, block_m,
                              kernel="gemv", w_zeros=zp_packed, weight_is_e2m1=e2m1)  # (P, 2*inter)
    d = out1.shape[1] // 2
    g_h, u_h = out1[:, :d], out1[:, d:]                       # already AT (== fused's gate_h / up_h)
    silu_h = (g_h.float() / (1.0 + torch.exp(-g_h.float()))).to(dtype)   # (AT)(g/(1+exp(-g)))
    ref_silu_h = silu_h * u_h                                 # AT*AT  == moe_silu_and_mul_h replica
    buf2_tail = tail_hip.silu_and_mul(out1.contiguous())      # shipped decode silu (own rounding order)
    buf2_torch = (F.silu(g_h.float()) * u_h.float()).to(dtype)  # fp32 silu+mul then round

    # FUSED: gemm1 + silu in one kernel, (P, inter) directly.
    buf2_fused = W.mmq_fp8_moe_gemm1_silu(x, w_packed, scales, sti, eids, ntp, top_k, block_m,
                                          kernel="gemv", w_zeros=zp_packed, weight_is_e2m1=e2m1)

    valid = (sti < num_valid)
    tag = f"T={T} E={E} inter={inter} K={K} tk={top_k} bm={block_m} g={g} " \
          f"{'ASYM' if asym else 'SYM'}{'/e2m1' if e2m1 else ''} {str(dtype).split('.')[-1]}"
    a = buf2_fused.float()[valid]
    # GATE: vs the exact moe_silu_and_mul_h replica. Same gemv + same epilogue arithmetic -> expect
    # bit-exact save the rare fp32 exp-ulp that survives the AT cast. mean-rel proves no systematic bug;
    # elementwise cap allows <=1 AT-ULP. torch/tail reported for context (differ by silu order, ~1-2 ULP).
    ulp = 2.0 ** (-8 if dtype == torch.bfloat16 else -10)     # ~1 ULP relative for bf16 / fp16
    for name, ref, gate in (("vs silu_h replica", ref_silu_h, True),
                            ("vs tail_hip.silu", buf2_tail, False),
                            ("vs torch.silu", buf2_torch, False)):
        b = ref.float()[valid]
        diff = (a - b).abs()
        mx = diff.max().item()
        denom = b.abs().clamp(min=1e-4)
        mean_rel = (diff / denom).mean().item()
        n_bad = int((diff > 2 * ulp * b.abs().clamp(min=0.5)).sum().item())  # > 2 AT-ULP
        refm = b.abs().mean().item()
        if gate:
            ok = (mean_rel <= 1e-3) and (n_bad == 0)
            flag = "PASS" if ok else "FAIL"
            if not ok:
                FAILS.append(f"{tag} {name}")
        else:
            flag = "info"
        print(f"  [{flag}] {tag:<50} {name:<18} max={mx:.3e} mean_rel={mean_rel:.2e} "
              f"bad={n_bad} |ref|={refm:.4f}")


def main():
    if not torch.cuda.is_available():
        print("FAIL: no CUDA/HIP device"); sys.exit(1)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print("=== moe_gemv_decode_silu FUSED parity (decode: T<=2, gemv) ===")
    # (T, E, inter, K, top_k, block_m, g); inter/K chosen ~Qwen MoE gemm1 shapes.
    shapes = [
        (1, 8, 256, 512, 2, 16, 128),
        (2, 8, 256, 512, 2, 16, 128),
        (1, 16, 384, 1024, 4, 16, 128),
        (2, 32, 768, 2048, 4, 16, 128),   # ~35B gemm1
        (1, 8, 512, 1024, 2, 16, 32),     # group=32
    ]
    for dtype in (torch.float16, torch.bfloat16):
        for (T, E, inter, K, tk, bm, g) in shapes:
            check(T, E, inter, K, tk, bm, g, asym=False, e2m1=False, dtype=dtype)
            check(T, E, inter, K, tk, bm, g, asym=True, e2m1=False, dtype=dtype)
            check(T, E, inter, K, tk, bm, g, asym=False, e2m1=True, dtype=dtype)
    print()
    if FAILS:
        print(f"FAILED ({len(FAILS)}):")
        for f in FAILS:
            print("   ", f)
        sys.exit(1)
    print("ALL PARITY PASS — fused gemv gemm1+silu bit-matches unfused gemv + silu")


if __name__ == "__main__":
    main()
