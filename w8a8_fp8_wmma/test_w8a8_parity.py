"""W8A8-fp8 grouped-MoE parity test (gfx1201).

Validates the native W8A8 MoE kernel (w8a8_fp8_wmma torch.ops) against two references,
for BOTH the prefill (M large -> WMMA, kernel=6) and decode (M<=2 -> GEMV gemm1 + scatter
gemm2) dispatch paths, top_k==1 (the ZAYA contract):

  Reference A (the CURRENT path being replaced): dequant fp8->bf16 (w.float()*scale) then the
    minisgl fused_experts_impl (Triton). This is the bf16-dequant->Triton MoE the kernel
    replaces; it is itself bf16-lossy so we compare at fp8 tolerance.
  Reference B (fp32 oracle): an fp32 torch reference that MATCHES the kernel's quantization
    semantics exactly — per-token dynamic act-fp8 quant (e4m3, max/448), per-N weight scale,
    fp32 accumulation, silu_and_mul gate|up, then gemm2 + topk-weighted reduce.

The kernel pipeline (gemm1 -> tail silu_and_mul -> gemm2 -> reduce/scatter) is driven through
the RAW w8a8 ops the same way python/minisgl/quant/kernels.py:w4a8_moe drives w4a8 (per-GEMM
kernel pick: gemm1 gemv if M<=2 else wmma; gemm2 wmma; decode scatter vs prefill gather_reduce).

Run ON GPU (lease 1 card, container recipe per CLAUDE.md):
    PYTHONPATH=/engine/python:/engine python /engine/w8a8_fp8_wmma/test_w8a8_parity.py

PASS iff parity holds for prefill AND decode. Exits nonzero on FAIL.
"""
import sys

import torch
import torch.nn.functional as F

import w8a8_fp8_wmma  # noqa: F401  loads torch.ops.w8a8_fp8_wmma
from w8a8_fp8_wmma import (
    mmq_w8a8_moe_gather_reduce,
    mmq_w8a8_moe_gemm,
    mmq_w8a8_moe_gemm_scatter,
)

E4M3_MAX = 448.0
BLOCK_M = 16


# ---- fp8 e4m3 round-trip matching the kernel's __builtin_amdgcn_cvt_*_fp8 ----
def quant_e4m3(x_f32: torch.Tensor) -> torch.Tensor:
    """f32 -> e4m3 -> f32, the value the kernel actually multiplies (cast round-trip)."""
    return x_f32.to(torch.float8_e4m3fn).float()


def act_quant_rowwise(x_f16: torch.Tensor):
    """Per-token dynamic act-fp8 quant exactly as moe_compute_act_fp8_kernel:
    scale = max(|row|)/448 (floored 1e-8); returns (x_fp8_roundtrip f32, act_scale (T,))."""
    xf = x_f16.float()
    amax = xf.abs().amax(dim=1)
    scale = (amax / E4M3_MAX).clamp_min(1e-8)
    xq = quant_e4m3(xf / scale[:, None])
    return xq, scale


def moe_align(topk_ids: torch.Tensor, E: int, block_m: int):
    import moe_hip

    return moe_hip.moe_align(topk_ids, E, block_m)


# ---------------------------------------------------------------------------
# Reference B: fp32 oracle with kernel-matching quantization.
# ---------------------------------------------------------------------------
def ref_b_fp32(x_f16, w13_fp8_rt, w13_scale, w2_fp8_rt, w2_scale, topk_ids, topk_weights):
    """x (M,K) f16; w13 (E,2I,K) f32 e4m3-roundtrip; w13_scale (E,2I); w2 (E,K,I); w2_scale (E,K).
    Returns (M,K) f32. Matches: act-fp8 per-token quant, per-N weight scale, fp32 accum,
    silu(gate)*up, gemm2 + topk-weighted reduce. top_k arbitrary (loops experts per slot)."""
    M, K = x_f16.shape
    E, twoI, _ = w13_fp8_rt.shape
    I = twoI // 2
    top_k = topk_ids.shape[1]
    out = torch.zeros(M, K, dtype=torch.float32, device=x_f16.device)

    xq, a_scale = act_quant_rowwise(x_f16)  # (M,K) f32 roundtrip, (M,)
    for m in range(M):
        for s in range(top_k):
            e = int(topk_ids[m, s])
            w = float(topk_weights[m, s])
            # gemm1: (2I,) = (xq[m] @ w13[e].T) * a_scale[m] * w13_scale[e]
            g1 = (xq[m] @ w13_fp8_rt[e].T) * a_scale[m] * w13_scale[e]  # (2I,)
            gate, up = g1[:I], g1[I:]
            inter = (F.silu(gate) * up)  # (I,) f32
            # gemm2 input is re-quantized to act-fp8 (the kernel re-runs act quant on buf2)
            inter16 = inter.to(torch.float16)
            iq, ia_scale = act_quant_rowwise(inter16[None, :])
            iq = iq[0]
            o = (iq @ w2_fp8_rt[e].T) * ia_scale[0] * w2_scale[e]  # (K,)
            out[m] += w * o
    return out


# ---------------------------------------------------------------------------
# Reference A: the current dequant fp8->bf16 + fused_experts_impl path.
# ---------------------------------------------------------------------------
def ref_a_dequant_triton(x_f16, w13_fp8, w13_scale, w2_fp8, w2_scale, topk_ids, topk_weights):
    """w13_fp8 (E,2I,K) float8_e4m3fn; w13_scale (E,2I) f32. Dequant per expert to bf16, run
    fused_experts_impl. Returns (M,K) in x dtype (bf16-lossy)."""
    from minisgl.moe.fused import fused_experts_impl

    E = w13_fp8.shape[0]
    dt = torch.bfloat16
    w13_bf = torch.stack([(w13_fp8[e].float() * w13_scale[e][:, None]).to(dt) for e in range(E)])
    w2_bf = torch.stack([(w2_fp8[e].float() * w2_scale[e][:, None]).to(dt) for e in range(E)])
    return fused_experts_impl(
        x_f16.to(dt).contiguous(), w13_bf.contiguous(), w2_bf.contiguous(),
        topk_weights.to(torch.float32), topk_ids.to(torch.int32), activation="silu",
        apply_router_weight_on_input=False,
    )


# ---------------------------------------------------------------------------
# The kernel under test: raw-op pipeline mirroring kernels.w4a8_moe.
# ---------------------------------------------------------------------------
def run_w8a8(x_f16, w13_fp8_u8, w13_scale, w2_fp8_u8, w2_scale, topk_ids, topk_weights, E):
    """w13_fp8_u8 (E,2I,K) uint8 e4m3 bytes; w13_scale (E,2I) f32. Returns (M,K) f32."""
    M, K = x_f16.shape
    dev = x_f16.device
    top_k = topk_ids.shape[1]

    gemm1_kernel = "gemv" if M <= 2 else "wmma"
    gemm2_kernel = "wmma"

    sorted_ids, expert_ids, ntp = moe_align(topk_ids.to(torch.int32), E, BLOCK_M)
    P = sorted_ids.shape[0]

    x16 = x_f16.to(torch.float16).contiguous()
    out1 = mmq_w8a8_moe_gemm(
        x16, w13_fp8_u8, w13_scale, sorted_ids, expert_ids, ntp, top_k, BLOCK_M, gemm1_kernel
    )  # (P, 2I) f16
    d = out1.shape[1] // 2
    import tail_hip  # noqa: F401  registers torch.ops.tail_hip.*

    buf2 = torch.ops.tail_hip.silu_and_mul(out1.contiguous())  # (P, I) f16

    tw_flat = topk_weights.reshape(-1).float().contiguous()
    if M <= 2:
        # DECODE: fused gemm2 + scatter into pre-zeroed (M,K) fp32.
        acc = torch.zeros((M, K), dtype=torch.float32, device=dev)
        mmq_w8a8_moe_gemm_scatter(
            buf2, w2_fp8_u8, w2_scale, sorted_ids, expert_ids, ntp, tw_flat, acc,
            top_k, BLOCK_M, gemm2_kernel
        )
        return acc
    # PREFILL: unfused gemm2 (P,K) then contention-free gather_reduce.
    ident = torch.arange(P, dtype=torch.int32, device=dev)
    out2 = mmq_w8a8_moe_gemm(
        buf2, w2_fp8_u8, w2_scale, ident, expert_ids, ntp, 1, BLOCK_M, gemm2_kernel
    )  # (P, K) f16
    acc = mmq_w8a8_moe_gather_reduce(out2.contiguous(), sorted_ids, tw_flat, ntp, top_k)
    return acc


# ---------------------------------------------------------------------------
def build_case(M, E, K, inter, top_k, dev, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    x = (torch.randn(M, K, generator=g, device=dev, dtype=torch.float32) * 0.5).half()

    # Per-expert fp8 (e4m3) weights + per-output-channel f32 scale.
    def mkw(N, Kin):
        raw = torch.randn(E, N, Kin, generator=g, device=dev, dtype=torch.float32)
        wq = raw.to(torch.float8_e4m3fn)            # stored e4m3
        scale = (torch.rand(E, N, generator=g, device=dev, dtype=torch.float32) * 0.02 + 0.002)
        return wq, scale

    w13_fp8, w13_scale = mkw(2 * inter, K)
    w2_fp8, w2_scale = mkw(K, inter)

    # uint8 byte view for the kernel ABI (op wants uint8 e4m3 bytes).
    w13_u8 = w13_fp8.view(torch.uint8).contiguous()
    w2_u8 = w2_fp8.view(torch.uint8).contiguous()
    # e4m3-roundtrip f32 for the fp32 oracle.
    w13_rt = w13_fp8.float()
    w2_rt = w2_fp8.float()

    # top_k routing (random experts; weights ~1 for top_k=1 ZAYA, else softmax-ish).
    ids = torch.randint(0, E, (M, top_k), generator=g, device=dev, dtype=torch.int32)
    if top_k == 1:
        tw = torch.ones(M, 1, device=dev, dtype=torch.float32)
    else:
        tw = torch.rand(M, top_k, generator=g, device=dev, dtype=torch.float32)
        tw = tw / tw.sum(dim=1, keepdim=True)
    return dict(
        x=x, w13_fp8=w13_fp8, w13_scale=w13_scale, w13_u8=w13_u8, w13_rt=w13_rt,
        w2_fp8=w2_fp8, w2_scale=w2_scale, w2_u8=w2_u8, w2_rt=w2_rt, ids=ids, tw=tw,
    )


def stats(name, a, b):
    a = a.float()
    b = b.float()
    d = (a - b).abs()
    max_abs = d.max().item()
    denom = b.abs().mean().clamp_min(1e-6)
    rel = (d.mean() / denom).item()
    print(f"    {name:18s} max_abs={max_abs:.4e}  rel_mean={rel:.4e}")
    return max_abs, rel


def run_phase(tag, M, E, K, inter, top_k, dev):
    print(f"\n=== {tag}: M={M} E={E} K={K} inter={inter} top_k={top_k} ===")
    c = build_case(M, E, K, inter, top_k, dev, seed=1234 + M)

    out_k = run_w8a8(c["x"], c["w13_u8"], c["w13_scale"], c["w2_u8"], c["w2_scale"],
                     c["ids"], c["tw"], E)
    ref_b = ref_b_fp32(c["x"], c["w13_rt"], c["w13_scale"], c["w2_rt"], c["w2_scale"],
                       c["ids"], c["tw"])
    ref_a = ref_a_dequant_triton(c["x"], c["w13_fp8"], c["w13_scale"], c["w2_fp8"],
                                 c["w2_scale"], c["ids"], c["tw"])

    # kernel vs fp32 oracle (B): the tight gate — same quant semantics, only fp16-staging +
    # WMMA accumulation-order noise separate them.
    mab_b, rel_b = stats("kernel vs refB(fp32)", out_k, ref_b)
    # kernel vs current path (A): both fp8-lossy AND A is bf16-dequant->Triton; looser.
    mab_a, rel_a = stats("kernel vs refA(bf16)", out_k, ref_a)
    # sanity: refA vs refB should agree at fp8/bf16 tolerance (direction check).
    mab_ab, rel_ab = stats("refA vs refB", ref_a, ref_b)

    # Tolerances: fp8 act+weight quant accumulated over K (qk-style). Oracle B shares the
    # kernel's exact quant, so rel is tight; A adds bf16 dequant + Triton path differences.
    ok_b = rel_b < 5e-2
    ok_a = rel_a < 1.5e-1
    ok_ab = rel_ab < 1.5e-1
    ok = ok_b and ok_a and ok_ab
    print(f"    -> refB rel<5e-2:{ok_b}  refA rel<1.5e-1:{ok_a}  A-vs-B rel<1.5e-1:{ok_ab}  "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def main():
    assert torch.cuda.is_available(), "no HIP GPU visible (check lease / device passthrough)"
    dev = "cuda"
    print("device:", torch.cuda.get_device_name(0))

    # ZAYA dims: E=16, K=2048, inter=4096. (N for w13 = 2*inter = 8192, for w2 = K = 2048.)
    E, K, inter = 16, 2048, 4096

    results = []
    # PREFILL: M large -> WMMA (kernel=6) for both GEMMs.
    results.append(("prefill", run_phase("PREFILL (wmma)", 128, E, K, inter, 1, dev)))
    # DECODE: M<=2 -> GEMV gemm1 + scatter gemm2.
    results.append(("decode-1", run_phase("DECODE (gemv+scatter) M=1", 1, E, K, inter, 1, dev)))
    results.append(("decode-2", run_phase("DECODE (gemv+scatter) M=2", 2, E, K, inter, 1, dev)))

    print("\n================ SUMMARY ================")
    all_ok = True
    for name, ok in results:
        print(f"  {name:12s} {'PASS' if ok else 'FAIL'}")
        all_ok = all_ok and ok
    print(f"\nOVERALL: {'PASS' if all_ok else 'FAIL'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
