#!/usr/bin/env python
"""Is the QUANTIZED dense/MoE path M-INVARIANT? (i.e. does a token's output depend on how many
tokens shared its batch?)

WHY THIS EXISTS
    `layers/minv.py` guarantees M-invariance for the UNQUANTIZED bf16/fp16 linears, and names the
    three pathways that depend on it: chunked prefill, prefix/radix caching, spec-decode VERIFY.
    The QUANTIZED path has no such guarantee and it is the path every served model actually uses for
    o_proj / gate_up / down_proj / the experts. `quant/kernels.py::_pick_dense_kernel` swaps between
    THREE DIFFERENT ALGORITHMS as a function of M:

        M <= 8  (int4) / <= 16 (e2m1)   ->  decode_gemv
        8  < M < 64                     ->  prefill_wmma
        M >= 64                         ->  wmma_tiled_tuned

    and `w4a8_moe` swaps gemm1 gemv->wmma at M>32, changes the grouped tile via `_moe_block_m(M,..)`,
    and takes an ATOMIC-SCATTER gemm2 at M<=2. Nobody had measured whether the arms agree.

    The spec-decode angle is the sharpest: greedy acceptance is `draft == target.argmax`, a DISCRETE
    test. Plain decode runs the target at M=1 (decode_gemv). Verify runs K+1 tokens per sequence in
    ONE forward -- MTP/EAGLE3 K=4 at bs=2 is M=10, DFlash K=15 is M=16 -- which crosses the M=8
    boundary onto a different kernel. If the arms disagree at all, acceptance is depressed by an
    amount no drafter improvement can recover.

WHAT IT MEASURES (dense and MoE, at the real per-rank TP=2 shapes)
    A. ARM AGREEMENT      -- same x, every legal arm, pairwise max|delta|.
    B. ARM SELF-INVARIANCE-- one arm, rows [0:m] alone vs as part of a batch of M.
    C. ENGINE INVARIANCE  -- the auto-dispatched engine path (`_pick_dense_kernel`), same test.
                             THIS is what chunked prefill / radix / verify actually experience.
    D. ROW SPLIT          -- `rowchunked_ar_span`'s producer: cat(auto(x[:h]), auto(x[h:])) vs auto(x).
    E. ARGMAX AGREEMENT   -- the acceptance-relevant metric: fraction of rows whose argmax over the
                             output columns changes between the M=1 arm and the M=K+1 arm.

Run (1 card, serve image):
    gpu-lease -n 1 -- bash tools/quant_m_invariance_run.sh
"""
from __future__ import annotations

import argparse
import sys

import torch

DEV = torch.device("cuda:0")
ARMS = ("decode_gemv", "prefill_wmma", "wmma_tiled_tuned")
GEMV_MAX_M = 16  # decode_gemv asserts M<=16 in-kernel


# ---------------------------------------------------------------------------------------------
# weight construction: compressed-tensors pack-quantized int4, group 32, SYMMETRIC (Gemma4 qat-AWQ)
# ---------------------------------------------------------------------------------------------
def pack_uint4_2d(w: torch.Tensor) -> torch.Tensor:  # (N,K) int8 -> (N,K/8) int32
    N, K = w.shape
    w = w.to(torch.int32)
    packed = torch.zeros((N, K // 8), dtype=torch.int32, device=w.device)
    for i in range(8):
        packed |= (w[:, i::8] & 0xF) << (i * 4)
    return packed


def pack_uint4_3d(w: torch.Tensor) -> torch.Tensor:  # (E,N,K) int8 -> (E,N,K/8) int32
    E, N, K = w.shape
    w = w.to(torch.int32)
    packed = torch.zeros((E, N, K // 8), dtype=torch.int32, device=w.device)
    for i in range(8):
        packed |= (w[:, :, i::8] & 0xF) << (i * 4)
    return packed


def make_dense(N: int, K: int, g: int, dtype):
    w = torch.randint(0, 16, (N, K), dtype=torch.int8, device=DEV)
    wp = pack_uint4_2d(w)
    sc = (torch.randn(N, K // g, device=DEV).abs() * 0.02 + 0.002).to(torch.float16)
    return wp, sc


def fmt(v: float) -> str:
    return "0" if v == 0.0 else f"{v:.3e}"


def rel(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    d = (a.float() - b.float()).abs()
    denom = b.float().abs().clamp(min=1e-4)
    return d.max().item(), (d / denom).max().item()


# ---------------------------------------------------------------------------------------------
# DENSE
# ---------------------------------------------------------------------------------------------
def dense_suite(shapes, ms, dtype, group, out):
    import fp8_wmma as W
    from minisgl.quant.kernels import _pick_dense_kernel, w4a8_linear

    MMAX = max(ms)
    for name, K, N in shapes:
        wp, sc = make_dense(N, K, group, dtype)
        x = (torch.randn(MMAX, K, device=DEV) * 0.3).to(dtype)

        out(f"\n=== DENSE {name}  K={K} N={N} g={group} {str(dtype).split('.')[-1]} ===")

        # ---- A. arm agreement at each M -------------------------------------------------------
        out("  [A] ARM AGREEMENT (same x, different kernel)")
        out(f"      {'M':>6} {'engine arm':<18} {'gemv~pwmma':>12} {'gemv~tiled':>12} {'pwmma~tiled':>12}")
        for M in ms:
            xm = x[:M].contiguous()
            res = {}
            for arm in ARMS:
                if arm == "decode_gemv" and M > GEMV_MAX_M:
                    continue
                res[arm] = W.mmq_fp8_gemm(xm, wp, sc, kernel=arm, w_zeros=None, weight_is_e2m1=False)
            picked = _pick_dense_kernel(M, False, group, k=K)
            def d(a, b):
                if a not in res or b not in res:
                    return "-"
                return fmt((res[a].float() - res[b].float()).abs().max().item())
            out(f"      {M:>6} {picked:<18} {d('decode_gemv','prefill_wmma'):>12} "
                f"{d('decode_gemv','wmma_tiled_tuned'):>12} {d('prefill_wmma','wmma_tiled_tuned'):>12}")

        # ---- B. per-arm self invariance -------------------------------------------------------
        out("  [B] ARM SELF-INVARIANCE  rows[0:m] alone  vs  inside a batch of Mref")
        for arm in ARMS:
            mref = GEMV_MAX_M if arm == "decode_gemv" else MMAX
            ref = W.mmq_fp8_gemm(x[:mref].contiguous(), wp, sc, kernel=arm, w_zeros=None,
                                 weight_is_e2m1=False)
            worst, worst_m = 0.0, 0
            for m in [v for v in ms if v <= mref]:
                sub = W.mmq_fp8_gemm(x[:m].contiguous(), wp, sc, kernel=arm, w_zeros=None,
                                     weight_is_e2m1=False)
                dv = (ref[:m].float() - sub.float()).abs().max().item()
                if dv > worst:
                    worst, worst_m = dv, m
            out(f"      {arm:<20} Mref={mref:<5} worst max|delta| = {fmt(worst)}"
                + (f"  (at m={worst_m})" if worst else ""))

        # ---- C. engine auto-dispatch invariance ----------------------------------------------
        out("  [C] ENGINE AUTO-DISPATCH INVARIANCE  (what chunked prefill / radix / verify see)")
        out(f"      {'m':>6} {'arm(m)':<18} {'arm(Mref)':<18} {'max|d|':>12} {'max rel':>10} {'argmax flips':>14}")
        big = w4a8_linear(x[:MMAX].contiguous(), wp, sc, None, group)
        for m in ms:
            sub = w4a8_linear(x[:m].contiguous(), wp, sc, None, group)
            mx, rl = rel(sub, big[:m])
            fl = (sub.argmax(-1) != big[:m].argmax(-1)).sum().item()
            out(f"      {m:>6} {_pick_dense_kernel(m, False, group, k=K):<18} "
                f"{_pick_dense_kernel(MMAX, False, group, k=K):<18} {fmt(mx):>12} {fmt(rl):>10} "
                f"{fl:>7}/{m:<6}")

        # ---- D. row split (the rowchunked_ar_span producer) ------------------------------------
        out("  [D] ROW SPLIT  cat(auto(x[:h]), auto(x[h:]))  vs  auto(x)   [rowchunked_ar_span]")
        for M in [v for v in ms if v >= 256]:
            full = w4a8_linear(x[:M].contiguous(), wp, sc, None, group)
            h = M // 2
            split = torch.cat([w4a8_linear(x[:h].contiguous(), wp, sc, None, group),
                               w4a8_linear(x[h:M].contiguous(), wp, sc, None, group)], dim=0)
            mx, _ = rel(split, full)
            fl = (split.argmax(-1) != full.argmax(-1)).sum().item()
            out(f"      rows={M:<6} split 2 -> max|delta| = {fmt(mx):<12} argmax flips {fl}/{M}")

        del wp, sc, x
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------------------------
# SPEC: M=1 decode vs M=K+1 verify, ARGMAX AGREEMENT
# ---------------------------------------------------------------------------------------------
def spec_suite(shapes, dtype, group, out):
    """The acceptance-relevant test. Greedy spec accepts iff draft == target.argmax at that position.
    Row 0 of a verify batch is the SAME token the previous decode step emitted at M=1, so if
    auto(x[:1]) != auto(x[:K+1])[0] the two forwards disagree about that position."""
    from minisgl.quant.kernels import _pick_dense_kernel, w4a8_linear

    # verify M = bs * (K+1). MTP/EAGLE3 K=4, DFlash K=15.
    CASES = [(1, 4), (2, 4), (4, 4), (8, 4), (1, 15), (2, 15), (1, 7), (4, 7)]
    out("\n=== SPEC: plain decode (M=1 per seq) vs verify (M=bs*(K+1)) ===")
    for name, K, N in shapes:
        wp, sc = make_dense(N, K, group, dtype)
        out(f"\n  {name}  K={K} N={N}")
        out(f"      {'bs':>3} {'K':>3} {'Mver':>5} {'arm@1':<14} {'arm@Mver':<18} "
            f"{'max|d| row0':>12} {'argmax flips (all rows)':>24}")
        for bs, kk in CASES:
            Mver = bs * (kk + 1)
            x = (torch.randn(Mver, K, device=DEV) * 0.3).to(dtype)
            ver = w4a8_linear(x.contiguous(), wp, sc, None, group)
            # sequential decode: each row on its own at M=1
            dec = torch.cat([w4a8_linear(x[i:i + 1].contiguous(), wp, sc, None, group)
                             for i in range(Mver)], dim=0)
            mx0 = (ver[0].float() - dec[0].float()).abs().max().item()
            fl = (ver.argmax(-1) != dec.argmax(-1)).sum().item()
            out(f"      {bs:>3} {kk:>3} {Mver:>5} {_pick_dense_kernel(1, False, group, k=K):<14} "
                f"{_pick_dense_kernel(Mver, False, group, k=K):<18} {fmt(mx0):>12} "
                f"{fl:>10}/{Mver:<6} ({100.0*fl/Mver:.2f}%)")
        del wp, sc
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------------------------
# MoE
# ---------------------------------------------------------------------------------------------
def moe_suite(ms, dtype, out, E=128, top_k=8, hidden=2816, inter=352, group=32, activation="gelu"):
    from minisgl.quant.kernels import _MOE_GEMM1_GEMV_MAX, _moe_block_m, w4a8_moe

    MMAX = max(ms)
    torch.manual_seed(1234)
    w13 = pack_uint4_3d(torch.randint(0, 16, (E, 2 * inter, hidden), dtype=torch.int8, device=DEV))
    s13 = (torch.randn(E, 2 * inter, hidden // group, device=DEV).abs() * 0.02 + 0.002).to(torch.float16)
    w2 = pack_uint4_3d(torch.randint(0, 16, (E, hidden, inter), dtype=torch.int8, device=DEV))
    s2 = (torch.randn(E, hidden, inter // group, device=DEV).abs() * 0.02 + 0.002).to(torch.float16)

    x = (torch.randn(MMAX, hidden, device=DEV) * 0.3).to(dtype)
    # ROUTE IS FIXED AND ROW-LOCAL: precompute once for all rows and slice, so the ONLY variable
    # under test is the grouped-GEMM's M-dependence (kernel + block_m + scatter), not the router.
    gate = torch.randn(MMAX, E, device=DEV)
    tw, ti = torch.topk(torch.softmax(gate.float(), -1), top_k, dim=-1)
    ti = ti.to(torch.int32)

    def run(m):
        return w4a8_moe(x[:m].contiguous(), w13, s13, None, w2, s2, None, None, top_k, False,
                        topk_weights=tw[:m].contiguous(), topk_ids=ti[:m].contiguous(),
                        activation=activation)

    out(f"\n=== MoE  E={E} top_k={top_k} hidden={hidden} inter={inter} g={group} "
        f"act={activation} {str(dtype).split('.')[-1]} ===")
    out(f"  gemm1 arm = gemv at M<={_MOE_GEMM1_GEMV_MAX}, else wmma;  gemm2 = ATOMIC SCATTER at M<=2")
    out(f"  {'m':>6} {'gemm1':>6} {'block_m':>8} {'gemm2':>9} {'max|d| vs batch':>17} {'max rel':>10} "
        f"{'argmax flips':>14}")
    big = run(MMAX)
    for m in ms:
        sub = run(m)
        mx, rl = rel(sub, big[:m])
        fl = (sub.argmax(-1) != big[:m].argmax(-1)).sum().item()
        g1 = "gemv" if m <= _MOE_GEMM1_GEMV_MAX else "wmma"
        g2 = "scatter" if m <= 2 else "gather"
        out(f"  {m:>6} {g1:>6} {_moe_block_m(m, E, top_k):>8} {g2:>9} {fmt(mx):>17} {fmt(rl):>10} "
            f"{fl:>7}/{m:<6}")

    # determinism of the M<=2 atomic scatter: same call twice
    out("  [determinism] same input, two consecutive calls:")
    for m in (1, 2, 4, 64):
        a, b = run(m), run(m)
        out(f"      m={m:<5} max|delta| = {fmt((a.float()-b.float()).abs().max().item())}")

    # row split
    out("  [D] ROW SPLIT (rowchunked_ar_span producer, the Qwen3.5-MoE case)")
    for M in [v for v in ms if v >= 256]:
        full = run(M)
        h = M // 2
        lo = w4a8_moe(x[:h].contiguous(), w13, s13, None, w2, s2, None, None, top_k, False,
                      topk_weights=tw[:h].contiguous(), topk_ids=ti[:h].contiguous(),
                      activation=activation)
        hi = w4a8_moe(x[h:M].contiguous(), w13, s13, None, w2, s2, None, None, top_k, False,
                      topk_weights=tw[h:M].contiguous(), topk_ids=ti[h:M].contiguous(),
                      activation=activation)
        split = torch.cat([lo, hi], dim=0)
        mx, _ = rel(split, full)
        out(f"      rows={M:<6} split 2 -> max|delta| = {fmt(mx)}")

    del w13, s13, w2, s2, x
    torch.cuda.empty_cache()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--skip-moe", action="store_true")
    ap.add_argument("--only-moe", action="store_true")
    args = ap.parse_args()

    fh = open(args.out, "w") if args.out else None

    def out(s=""):
        print(s, flush=True)
        if fh:
            fh.write(s + "\n")
            fh.flush()

    torch.manual_seed(0)
    # the engine's `engaged()` provenance logger reads get_tp_info(); this harness is single-process.
    from minisgl.distributed import info as _info

    _info._TP_INFO = _info.DistributedInfo(rank=0, size=1)  # type: ignore[attr-defined]
    out(f"device: {torch.cuda.get_device_name(0)}")
    import fp8_wmma as W
    out(f"fp8_wmma: {W.__file__}")
    import minisgl.quant.kernels as QK
    out(f"minisgl.quant.kernels: {QK.__file__}")
    out(f"crossovers: gemv_max_int4={QK._W4A8_GEMV_MAX_INT4} gemv_max_e2m1={QK._W4A8_GEMV_MAX_E2M1} "
        f"tiled_min={QK._W4A8_PREFILL_TILED_MIN} gemv_K_mult={QK._W4A8_GEMV_K_MULTIPLE} "
        f"moe_gemm1_gemv_max={QK._MOE_GEMM1_GEMV_MAX}")

    MS = [1, 2, 4, 5, 8, 9, 10, 16, 17, 20, 32, 33, 64, 65, 128, 129, 192, 256, 512, 1024, 2048]

    # Gemma4-26B-A4B qat-AWQ-INT4: hidden 2816, head_dim 256 (global 512), nq 16, kv 8,
    # dense inter 2112, moe inter 704, E=128, top_k 8, group_size 32, dtype float16. TP=2 per rank.
    G4 = [
        ("g4.o_proj(local,rank)", 2048, 2816),   # 16*256/2 in, hidden out   [row-parallel]
        ("g4.o_proj(global,rank)", 4096, 2816),  # 16*512/2 in
        ("g4.o_proj(global,tp1)", 8192, 2816),   # un-sharded global o_proj
        ("g4.qkv_q(rank)", 2816, 2048),          # hidden in, q out          [col-parallel]
        ("g4.gate_up(rank)", 2816, 2112),        # hidden in, 2*1056 out
        ("g4.dense_down(rank)", 1056, 2816),     # 2112/2 in, hidden out     [row-parallel]
    ]
    if not args.only_moe:
        dense_suite(G4, MS, torch.float16, 32, out)
        spec_suite(G4, torch.float16, 32, out)

    # Qwen3.6-35B-A3B-AWQ-4bit: group_size 128, bf16 activations. One shape to prove the result is
    # not a group-32/fp16 artifact.
    Q35 = [("q35.o_proj(rank)", 2048, 4096), ("q35.gate_up(rank)", 4096, 3072)]
    if not args.only_moe:
        dense_suite(Q35, MS, torch.bfloat16, 128, out)
        spec_suite(Q35, torch.bfloat16, 128, out)

    if not args.skip_moe:
        moe_suite(MS, torch.float16, out, activation="gelu")     # Gemma4 routed experts
        moe_suite(MS, torch.float16, out, activation="silu")     # the fused-epilogue path

    out("\ndone.")
    if fh:
        fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
