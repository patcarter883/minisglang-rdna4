"""Swappable quantized-kernel provider.

ALL quantized GEMM call sites in the engine go through this module, so the parallel
custom kernel framework can be dropped in by swapping these impls — without touching
layers / models / the weight loader. The current impl wraps the in-repo
`w4a8_fp8_wmma` kernel (int4 weights + fp8 activations -> native RDNA4 fp8 WMMA;
nothing dequants to F16 — the fp16 I/O is the op's staging boundary, compute is e4m3).

Weight-layout conversion (AWQ g128 -> op g32, AutoGPTQ bit order, transpose;
compressed-tensors g32 ~= native) also belongs here — ported from
vllm-gfx1201/w4a8_fp8_wmma/{vllm_adapter.py,moe_experts.py} — TODO Phase 2c.
"""
from __future__ import annotations

import torch

from minisgl._hip_engage import engaged

# AWQ "gemm" reverse pack order (undo the [0,2,4,6,1,3,5,7] interleave). From
# vllm-gfx1201/w4a8_fp8_wmma/moe_experts.py:_REVERSE_AWQ_PACK_ORDER.
_REVERSE_AWQ_PACK_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]

# --- env-gated W4A8-MoE sub-step GPU-time attribution (diagnostics only) ------------------------
# MINISGL_MOE_PROF=<N> times each step of w4a8_moe (route/align/cast/gemm1/silu/gemm2/gather) with
# CUDA events and logs the split every N calls. Inert (zero overhead) when unset.
import os as _os
from collections import defaultdict as _dd

_MOE_EVERY = int(_os.environ["MINISGL_MOE_PROF"]) if _os.environ.get("MINISGL_MOE_PROF", "").isdigit() else 0
_moe_buckets: dict = _dd(float)
_moe_calls = 0

# Decode-path gemm2+gather fusion via mmq_fp8_moe_gemm_scatter (atomic scatter). OFF by default so
# CUDA-graph decode capture works out of the box for MoE models (the scatter's atomicAdd is NOT
# graph-capture-safe; the unfused gemm2 + gather_reduce fallback reaches ~the same memory bandwidth).
# Set MINISGL_MOE_SCATTER=1 to force the fused scatter for an eager (non-graph) deployment.
_MOE_SCATTER = _os.environ.get("MINISGL_MOE_SCATTER", "0") != "0"

# W4A16 MoE (fp16 activations, no act-quant) for the routed experts — the fix for the fp8-act decode
# degradation on activation-sensitive models (GLM-4.7-Flash). "1" = all M; "decode" = M<=2 only
# (mirrors vLLM's low-M W4A16 crossover). Off by default. Requires group_size>=64 (g=128 -> wide 8).
MOE_W4A16 = _os.environ.get("MINISGL_MOE_W4A16", "0")

# W8A8-fp8 MoE register-direct b128 (LDS-bypass) path for the routed experts — the fp8 twin of the
# W4A16 register-direct kernel. Bit-exact vs the LDS-staged w8a8_moe (same act-fp8 quant, same fp8
# e4m3 weights, same WMMA chain; weights just pre-permuted into WMMA-B lane order) and ~2x faster at
# ZAYA decode (gemm1 T1 148->55us). ON by default (the served ZAYA1-8B-fp8 path); =0 reverts to the
# LDS-staged w8a8_moe. Weights are pre-repacked to _w_rep in _GroupedFP8Experts.post_load.
MOE_W8A8_REGDIRECT = _os.environ.get("MINISGL_MOE_W8A8_REGDIRECT", "1") != "0"

# RXF (W4-NL x int8) MoE register-direct b128 — the RXF twin of the above (moe_gemm_regdirect, no
# LDS-staged B / no K-loop barrier). Bit-exact vs the LDS-staged rxf_moe, gemm1 T1 89->44us (2.02x).
# ON by default; =0 reverts to LDS-staged rxf_moe. Weights repacked to _w_rep in post_load.
RXF_REGDIRECT = _os.environ.get("MINISGL_RXF_REGDIRECT", "1") != "0"

# MXFP4 (OCP E2M1) MoE register-direct b128 — routes the e2m1 experts through the register-direct
# W4A16 kernel (mmq_regdirect_w4a16_moe, weight_is_e2m1=True): fp16 acts DIRECT (no act-quant, unlike
# the w4a8_moe e2m1 path which fp8-quantizes acts) + b128 weight loads. Aims to be faster AND higher
# quality (fp16 acts, matching vLLM/MLX MXFP4) — but it is NOT yet validated on the MXFP4 target, so
# it is OPT-IN (=1). Default OFF -> the validated LDS-staged W4A8 path w4a8_moe(weight_is_e2m1=True)
# (fp8 acts). Weights repacked to _w_rep/_scales_rd in post_load only when this is on.
MOE_MXFP4_REGDIRECT = _os.environ.get("MINISGL_MOE_MXFP4_REGDIRECT", "0") != "0"

# Decode gemm2 split-K (Task A #17): MINISGL_MOE_SPLITK=<S> (S>=2) routes the decode scatter gemm2
# to the minisgl-local moe_splitk_hip kernel, carving the K=inter contraction across S grid.z blocks
# to lift occupancy (gemm2 is ~15% of peak BW at M=1). 0/unset = the vendored w4a8_fp8_wmma scatter.
# Like the base scatter, the atomicAdd is NOT cuda-graph-capture-safe (eager decode only).
_MOE_SPLITK = int(_os.environ["MINISGL_MOE_SPLITK"]) if _os.environ.get("MINISGL_MOE_SPLITK", "").isdigit() else 0

# Native HIP moe_align (moe_hip) replacing the vLLM moe_align_block_size host op. On by default;
# MINISGL_MOE_ALIGN=0 reverts to the vLLM reference.
_MOE_ALIGN_HIP = _os.environ.get("MINISGL_MOE_ALIGN", "1") != "0"

# Native HIP fp16 silu_and_mul (tail_hip) for the MoE intermediates. Same MINISGL_TAIL_HIP=0 opt-out
# as the layer path; gated locally (don't import minisgl.layers from quant — it pulls the full layer
# stack). silu_and_mul is dtype-generic (fp16 MoE intermediates / bf16 / fp32).
_TAIL_HIP = _os.environ.get("MINISGL_TAIL_HIP", "1") != "0"
_SILU_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
# Fused gemm1 + silu_and_mul decode path (mmq_fp8_moe_gemm1_silu, kernel="gemv" -> moe_gemv_decode_silu):
# one warp per FUSED output col computes gate (weight col j) AND up (col j+inter) sharing the gathered fp8
# activation, then writes silu(gate)*up to (P, inter) directly -- dropping the separate silu launch AND the
# (P, 2*inter) out1 HBM round-trip. Reuses the WMMA fused path's exact epilogue (moe_silu_and_mul_h), so it
# is bit-identical to that already-validated fused kernel; vs the tail_hip.silu_and_mul reference it can
# differ only in silu's last-bit fp32 rounding. fp16/bf16 only. MINISGL_MOE_FUSED_SILU=0 reverts to
# gemm1 + tail_hip.silu_and_mul.
_MOE_FUSED_SILU = _os.environ.get("MINISGL_MOE_FUSED_SILU", "1") != "0"
_FUSED_SILU_DTYPES = (torch.float16, torch.bfloat16)
# MoE gemm1 (wide 2*inter output) decode crossover: the scalar GEMV wins gemm1 through M~32 (measured
# w4a8 E128/tk8/inter768: 1.9-3.5x over WMMA at M=4-32; WMMA reclaims M=64) — far past the old M<=2 gate.
# w8a8 gemm1 shares the same gemv structure so uses the same crossover (inferred from the w4a8 measurement).
_MOE_GEMM1_GEMV_MAX = 32
# Register-tiled "flag" grouped MoE GEMM (moe_gemm_flag) for PREFILL gemm2 (down-proj, non-scatter): a
# 64x64/64x32 register-macro-tile port of the flagship dense kernel -> ~1.1-1.5x over the tiled wmma at
# block_m==128 (bit-exact). Gated block_m==128 (the macro-tile needs it) + group_size==128 for w4 (E's
# validated config; MXFP4 group=32 + decode/small-M stay on tiled/gemv). MINISGL_MOE_FLAG=0 reverts.
_MOE_FLAG = _os.environ.get("MINISGL_MOE_FLAG", "1") != "0"
# DECODE gemm2 (down-proj) + gather-reduce fused into ONE graph-safe kernel (mmq_fp8_moe_gemm2_gather_
# reduce): drops the (P,N) out2 HBM round-trip + the gather_reduce launch, and skips the alignment-
# padding rows the WMMA gemm2 computes. Uses gemv-math down-proj -> ~1e-4 vs the WMMA path (fp32
# accumulation order, ~1000x below the fp8 quant-noise floor; user-accepted, NOT max|Δ|=0).
# MINISGL_MOE_G2FUSE=0 reverts to the bit-exact WMMA gemm2 + gather_reduce.
_MOE_G2FUSE = _os.environ.get("MINISGL_MOE_G2FUSE", "1") != "0"


def _moe_time(bucket: str, fn):
    if not _MOE_EVERY:
        return fn()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    r = fn()
    e.record()
    e.synchronize()
    _moe_buckets[bucket] += s.elapsed_time(e)
    return r


def _moe_report() -> None:
    global _moe_calls
    if not _MOE_EVERY:
        return
    _moe_calls += 1
    if _moe_calls % _MOE_EVERY == 0 and _moe_buckets:
        from minisgl.utils import init_logger

        tot = sum(_moe_buckets.values()) or 1.0
        parts = "  ".join(
            f"{k}={v / _MOE_EVERY * 1000:.0f}us({100 * v / tot:.0f}%)"
            for k, v in sorted(_moe_buckets.items(), key=lambda x: -x[1])
        )
        init_logger("moe_prof").info_rank0(f"[moe-prof] per-call over {_MOE_EVERY} calls: {parts}")
        _moe_buckets.clear()


def awq_to_op_layout(
    qweight: torch.Tensor,  # (K, N//pf) int32, AWQ-packed along output
    scales: torch.Tensor,  # (K//group, N) fp16
    qzeros: torch.Tensor | None,  # (K//group, N//pf) int32, AWQ-packed; None if symmetric
    *,
    bits: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Convert ONE dense AWQ matrix to the op's native layout:
    w_packed (N, K//pf) int32, scales (N, K//group) fp16, zeros (N//pf, K//group) int32.
    Ported from w4a8_fp8_wmma/moe_experts.py:_awq_to_op_layout_single (per-matrix).
    NOTE: one-shot unpack — transient (N, K) tensor; chunk for very large matrices
    (PERF_NOTES; the 27B OOM)."""
    pf = 32 // bits  # 8
    mask = (1 << bits) - 1
    K, Np = qweight.shape
    N = Np * pf
    dev = qweight.device
    rev = torch.tensor(_REVERSE_AWQ_PACK_ORDER[:pf], dtype=torch.long, device=dev)
    shifts = torch.arange(0, 32, bits, dtype=torch.int32, device=dev)

    # qweight: unpack AWQ -> (K, N) uint4 (natural channel order) -> (N, K) -> repack along K
    uw = (qweight.unsqueeze(-1) >> shifts) & mask  # (K, Np, pf)
    uw = uw[:, :, rev].reshape(K, N).t().contiguous().to(torch.int32)  # (N, K)
    w_packed = torch.zeros((N, K // pf), dtype=torch.int32, device=dev)
    for j in range(pf):
        w_packed |= (uw[:, j::pf] & mask) << (j * bits)

    scales_op = scales.t().contiguous().to(torch.float16)  # (G, N) -> (N, G)

    zeros_op = None
    if qzeros is not None:
        G = qzeros.shape[0]
        uz = (qzeros.unsqueeze(-1) >> shifts) & mask  # (G, Np, pf)
        uz = uz[:, :, rev].reshape(G, N).t().contiguous().to(torch.int32)  # (N, G)
        zeros_op = torch.zeros((N // pf, G), dtype=torch.int32, device=dev)
        for j in range(pf):
            zeros_op |= (uz[j::pf, :] & mask) << (j * bits)

    return w_packed, scales_op, zeros_op


def gptq_to_op_layout(
    qweight: torch.Tensor,  # (K//pf, N) int32, GPTQ-packed along INPUT (natural nibble order)
    scales: torch.Tensor,  # (K//group, N) fp16
    qzeros: torch.Tensor | None,  # (K//group, N//pf) int32, GPTQ-packed along N; +1 = zero point
    *,
    bits: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert ONE dense GPTQ matrix to the op's native layout
    (w_packed (N, K//pf) int32, scales (N, K//group) fp16, zeros (N//pf, K//group) int32).

    GPTQ vs AWQ: int32 is packed along INPUT (K) with NATURAL nibble order (no AWQ interleave),
    and qzeros are ALWAYS present — the dequant zero point is `unpacked_qzeros + 1` (AutoGPTQ's
    historical off-by-one; symmetric int4 stores 7 -> zero=8). We fold the +1 and emit an EXPLICIT
    zeros tensor so the proven asymmetric op path (w = scale*(q - zero)) is exact regardless of the
    `sym` flag. Asserts the folded zero fits 4 bits (true for this checkpoint: all 8)."""
    pf = 32 // bits  # 8
    mask = (1 << bits) - 1
    Kp, N = qweight.shape
    K = Kp * pf
    dev = qweight.device
    shifts = torch.arange(0, 32, bits, dtype=torch.int32, device=dev)  # [0,4,...,28]

    # qweight: unpack (K//pf, N, pf) NATURAL (channel = ki*pf + j) -> (K, N) -> (N, K) -> repack/K
    uw = (qweight.unsqueeze(-1) >> shifts) & mask  # (Kp, N, pf)
    uw = uw.permute(0, 2, 1).reshape(K, N)  # (K, N): row ki*pf+j
    uw = uw.t().contiguous().to(torch.int32)  # (N, K)
    w_packed = torch.zeros((N, K // pf), dtype=torch.int32, device=dev)
    for j in range(pf):
        w_packed |= (uw[:, j::pf] & mask) << (j * bits)

    scales_op = scales.t().contiguous().to(torch.float16)  # (G, N) -> (N, G)

    # qzeros: unpack (G, N//pf, pf) NATURAL along N -> (G, N), fold +1, -> (N, G), repack/N.
    assert qzeros is not None, "GPTQ always ships qzeros"
    G = qzeros.shape[0]
    uz = (qzeros.unsqueeze(-1) >> shifts) & mask  # (G, N//pf, pf)
    uz = uz.reshape(G, N) + 1  # (G, N) actual zero point; col = np*pf + j (natural)
    assert int(uz.max()) <= mask, f"GPTQ zero+1 overflows {bits}b (max={int(uz.max())})"
    uz = uz.t().contiguous().to(torch.int32)  # (N, G)
    zeros_op = torch.zeros((N // pf, G), dtype=torch.int32, device=dev)
    for j in range(pf):
        zeros_op |= (uz[j::pf, :] & mask) << (j * bits)

    return w_packed, scales_op, zeros_op


# Grouped-MoE WMMA tile heights. These are HARDWARE/KERNEL limits, not tunables: the WMMA M-dim is
# WMMA_DIM=16 and the kernel launches up to MV5_MAX_WARPS=8 warps (block_m = warps * 16, so 128 max).
_MOE_WMMA_DIM = 16
_MOE_BLOCK_M_CHOICES = (16, 32, 64, 128)  # multiples of WMMA_DIM up to 8 warps


def _moe_block_m(num_tokens: int, num_experts: int, top_k: int) -> int:
    """Choose the grouped-MoE tile height from the WORKLOAD rather than a fixed constant.

    ``moe_align`` pads EACH expert's routed rows up to a multiple of ``block_m`` and the WMMA kernel
    then grinds one expert's weight slab per ``block_m``-row tile (n_warps = block_m/16). So the tile
    height trades padding waste against weight reuse + occupancy:
      * a tile taller than an expert's row count is mostly PADDING — decode (M<=2 => ~0 rows/expert)
        wants the 16-row minimum (and gemm1 is GEMV there anyway);
      * prefill routes thousands of rows/expert, so a taller tile reuses each fp8 weight slab across
        more rows and runs more warps => higher throughput.
    Pick the largest supported tile (16..128) not exceeding the average rows-per-expert
    (num_tokens*top_k / num_experts). Pure function of static shapes => cudagraph-capture-safe (a
    captured decode graph always sees small M => 16, exactly as before). ``MINISGL_MOE_BLOCK_M`` forces
    a value (16/32/64/128) for autotuning / to restore the historical fixed 16.
    """
    forced = _os.environ.get("MINISGL_MOE_BLOCK_M")
    if forced:
        bm = int(forced)
        assert bm in _MOE_BLOCK_M_CHOICES, f"MINISGL_MOE_BLOCK_M must be one of {_MOE_BLOCK_M_CHOICES}"
        return bm
    rows_per_expert = (num_tokens * top_k) / max(num_experts, 1)
    bm = _MOE_WMMA_DIM
    for choice in _MOE_BLOCK_M_CHOICES:
        if choice <= rows_per_expert:
            bm = choice
    return bm


def _softmax_topk_route(
    gating_output: torch.Tensor, top_k: int, renormalize: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused softmax + top-k (+ optional renormalize) route from raw router logits.

    Returns (topk_weights f32 (M, top_k), topk_ids i32 (M, top_k)) — what a model that does NOT
    precompute its route (Qwen3.5-MoE: router_logits only, no noaux_tc) hands the grouped-MoE kernel.
    The lean (vllm-free) image has no fused kernel, so this is the pure-torch chain matching the
    vLLM _moe_C.topk_softmax op; when vLLM IS present (the legacy combined image) prefer its fused
    kernel — a single launch vs the torch chain. Shared by w4a8_moe and w4a16_moe."""
    M = gating_output.shape[0]
    dev = gating_output.device
    try:
        from vllm import _custom_ops as vllm_ops
    except ImportError:
        probs = torch.softmax(gating_output.float(), dim=-1)
        tw, ti = torch.topk(probs, top_k, dim=-1)
        if renormalize:
            tw = tw / (tw.sum(dim=-1, keepdim=True) + 1e-20)
        return tw.contiguous(), ti.to(torch.int32).contiguous()

    tw = torch.empty(M, top_k, dtype=torch.float32, device=dev)
    ti = torch.empty(M, top_k, dtype=torch.int32, device=dev)
    tei = torch.empty(M, top_k, dtype=torch.int32, device=dev)  # token_expert_indices scratch
    vllm_ops.topk_softmax(tw, ti, tei, gating_output.float(), renormalize)
    return tw, ti


def w4a8_moe(
    x: torch.Tensor,  # (M, K) activations
    w13: torch.Tensor,  # (E, 2*inter, K//8) i32 — gate|up stacked
    w13_scales: torch.Tensor,  # (E, 2*inter, K//g) f16
    w13_zeros: torch.Tensor | None,
    w2: torch.Tensor,  # (E, K, inter//8) i32
    w2_scales: torch.Tensor,  # (E, K, inter//g) f16
    w2_zeros: torch.Tensor | None,
    gating_output: torch.Tensor | None,  # (M, E); ignored when topk_ids/topk_weights are given
    top_k: int,
    renormalize: bool,
    *,
    topk_weights: torch.Tensor | None = None,  # (M, top_k) f32 — precomputed route (e.g. noaux_tc)
    topk_ids: torch.Tensor | None = None,  # (M, top_k) i32 — precomputed expert ids
    kernel: str = "wmma",
    block_m: int | None = None,  # None -> derive the WMMA tile height from the workload (_moe_block_m)
    weight_is_e2m1: bool = False,  # True -> decode w13/w2 nibbles as MXFP4 (OCP E2M1), zeros must be None
) -> torch.Tensor:
    """Grouped W4A8 MoE forward: topk -> moe_align -> grouped GEMM(w13) -> silu_and_mul
    -> grouped GEMM(w2) -> topk-weighted gather-reduce. Mirrors the proven
    w4a8_fp8_wmma `_run_grouped_moe` (non-GEMV, unfused-silu) path. Returns (M, K).
    `weight_is_e2m1=True` selects the kernel's MXFP4 (E2M1) weight decode instead of uniform int4
    (the scales are the E8M0 group exponents folded to fp16; w13_zeros/w2_zeros MUST be None).
    NOTE: imports vLLM's moe_align_block_size from the image (a small util) — port to a
    torch/Triton implementation later (PERF_NOTES)."""
    import torch.nn.functional as F
    import fp8_wmma

    M, K = x.shape
    E = w13.shape[0]
    dev = x.device
    if block_m is None:  # derive the grouped-GEMM tile from the workload (16 at decode, up to 128 at prefill)
        block_m = _moe_block_m(M, E, top_k)
    # Decode fast path is PER-GEMM: the two grouped GEMMs want OPPOSITE kernels at M<=2 (measured on
    # gfx1201, Qwen3.6-35B). gemm1 (w13, wide 2*inter output, gather-by-sorted, top_k>1): the scalar
    # GEMV is ~8.7x faster than the prefill WMMA (35us vs 306us). gemm2 (w2, K output, identity
    # gather, top_k==1): WMMA is ~6.8x faster than GEMV (50us vs 339us) — the GEMV path
    # underperforms for that shape. So pick gemv for gemm1, keep WMMA for gemm2. Together ~85us vs
    # ~356us with the old all-WMMA default. Prefill (M>2) keeps the passed/default kernel for both.
    gemm1_kernel = "gemv" if M <= _MOE_GEMM1_GEMV_MAX else kernel
    gemm2_kernel = kernel

    # Precomputed route (e.g. GLM/DeepSeek noaux_tc: sigmoid + correction bias + group top-k +
    # normalize + scale, done in the model). Otherwise fall back to fused softmax+topk here.
    if topk_ids is None:
        topk_weights, topk_ids = _moe_time(
            "route", lambda: _softmax_topk_route(gating_output, top_k, renormalize)
        )
    else:
        assert topk_weights is not None, "topk_weights required when topk_ids is given"
        topk_weights = topk_weights.to(torch.float32).contiguous()
        topk_ids = topk_ids.to(torch.int32).contiguous()

    # moe_align: native HIP (moe_hip). The former MINISGL_MOE_ALIGN=0 vLLM reference is gone — the
    # lean image has no vllm, and moe_hip.moe_align is the validated drop-in for that host op.
    import moe_hip

    engaged("moe_hip.moe_align")
    sorted_ids, expert_ids, ntp = _moe_time("align", lambda: moe_hip.moe_align(topk_ids, E, block_m))
    P = sorted_ids.shape[0]

    _e2m1 = "+e2m1" if weight_is_e2m1 else ""
    # The W4A8 kernel is now activation-dtype-generic (fp16 OR bf16), so pass activations in their
    # NATIVE dtype — a bf16 model no longer round-trips bf16->fp16->bf16 here (out1 follows x's dtype).
    x16 = _moe_time("cast", lambda: x.contiguous())
    # Gated gemm1 + SiLU-mul. FUSED path (default): one kernel writes silu(gate)*up -> (P, inter),
    # dropping the separate silu launch and the (P, 2*inter) out1 round-trip. The gemm1-epilogue fusion
    # is now available at DECODE too via the moe_gemv_decode_silu kernel (kernel="gemv"), not just the
    # WMMA prefill path. UNFUSED fallback (MINISGL_MOE_FUSED_SILU=0, or fp32) = gemm1 -> (P,2*inter) then
    # tail_hip.silu_and_mul (fp32-internal HIP) / the torch silu+mul reference.
    if _MOE_FUSED_SILU and x16.dtype in _FUSED_SILU_DTYPES:
        # PREFILL (block_m in {64,128}): the silu-fused flagship register-tiled gemm1 flag — bit-exact
        # to the tiled gemm1_silu (max|Δ|=0), W4 wins 1.28x @128 / 1.58x @64 (the 53% real-traffic
        # band). Decode/small-M (block_m<64) stays on the tiled/gemv fused path.
        _flag1 = _MOE_FLAG and block_m in (64, 128) and \
            (w13.shape[-1] * 8) // w13_scales.shape[-1] in (32, 64, 128)
        if _flag1:
            engaged(f"fp8_wmma.mmq_fp8_moe_gemm1_silu_flag{_e2m1}")
            buf2 = _moe_time(
                "gemm1silu",
                lambda: fp8_wmma.mmq_fp8_moe_gemm1_silu_flag(
                    x16, w13, w13_scales, sorted_ids, expert_ids, ntp, top_k, block_m,
                    w_zeros=w13_zeros, weight_is_e2m1=weight_is_e2m1,
                ),
            )  # (P, inter) in x's dtype
        else:
            engaged(f"fp8_wmma.mmq_fp8_moe_gemm1_silu({gemm1_kernel}{_e2m1})")
            buf2 = _moe_time(
                "gemm1silu",
                lambda: fp8_wmma.mmq_fp8_moe_gemm1_silu(
                    x16, w13, w13_scales, sorted_ids, expert_ids, ntp, top_k, block_m,
                    kernel=gemm1_kernel, w_zeros=w13_zeros, weight_is_e2m1=weight_is_e2m1,
                ),
            )  # (P, inter) in x's dtype
    else:
        engaged(f"fp8_wmma.mmq_fp8_moe_gemm({gemm1_kernel}{_e2m1})")
        out1 = _moe_time(
            "gemm1",
            lambda: fp8_wmma.mmq_fp8_moe_gemm(
                x16, w13, w13_scales, sorted_ids, expert_ids, ntp, top_k, block_m,
                kernel=gemm1_kernel, w_zeros=w13_zeros, weight_is_e2m1=weight_is_e2m1,
            ),
        )  # (P, 2*inter) in x's dtype
        d = out1.shape[1] // 2
        if _TAIL_HIP and out1.dtype in _SILU_DTYPES:
            import tail_hip  # canonical package: silu_and_mul is a module-level callable

            engaged("tail_hip.silu_and_mul")
            buf2 = _moe_time("silu", lambda: tail_hip.silu_and_mul(out1.contiguous()))
        else:
            buf2 = _moe_time(
                "silu",
                lambda: (F.silu(out1[:, :d].float()) * out1[:, d:].float()).to(out1.dtype).contiguous(),
            )

    tw_flat = topk_weights.reshape(-1).float().contiguous()
    # DECODE fast path: fuse gemm2 + topk-weight + reduce into ONE kernel (mmq_fp8_moe_gemm_scatter):
    # it computes gemm2 (identity-gather over buf2) and atomic-scatters topk_weights[r]*(buf2[r]@W) into
    # a pre-zeroed fp32 (M,K), removing BOTH the (P,K) out2 materialization AND the separate
    # gather_reduce launch. The atomicAdd scatter is NOT HIP-graph-capture-safe, so it is gated to the
    # eager decode path (M<=2; the serve default is cuda_graph_max_bs=0) and can be turned off with
    # MINISGL_MOE_SCATTER=0. Prefill (M>2) keeps the unfused gemm2 + contention-free gather_reduce
    # (graph-safe; also the (P,K) out2 reuse amortizes better at larger M).
    if M <= 2 and _MOE_SCATTER:
        acc = torch.zeros((M, K), dtype=torch.float32, device=dev)
        if _MOE_SPLITK >= 2 and M == 1:  # split-K only helps the M==1 grid (M>=2 has enough blocks)
            assert not weight_is_e2m1, "moe_splitk scatter has no MXFP4 (e2m1) decode path"
            import moe_splitk_hip  # canonical package: op is a module-level callable

            engaged("moe_splitk_hip.moe_gemm_splitk_scatter")
            _moe_time(
                "gemm2scat",
                lambda: moe_splitk_hip.moe_gemm_splitk_scatter(
                    buf2, w2, w2_scales, w2_zeros, sorted_ids, expert_ids, ntp, tw_flat, acc,
                    top_k, block_m, _MOE_SPLITK,
                ),
            )  # writes acc in place (atomic scatter over experts AND split_k K-slices)
        else:
            engaged(f"fp8_wmma.mmq_fp8_moe_gemm_scatter{_e2m1}")
            _moe_time(
                "gemm2scat",
                lambda: fp8_wmma.mmq_fp8_moe_gemm_scatter(
                    buf2, w2, w2_scales, sorted_ids, expert_ids, ntp, tw_flat, acc, top_k, block_m,
                    kernel=gemm2_kernel, w_zeros=w2_zeros, weight_is_e2m1=weight_is_e2m1,
                ),
            )  # writes acc in place
        _moe_report()
        return acc.to(x.dtype)

    # DECODE (small M): fuse gemm2 (down-proj) + gather-reduce into ONE launch. Reads the pre-sorted buf2
    # directly + does the per-token top_k reduce in-kernel, dropping the (P,N) out2 HBM round-trip + the
    # gather_reduce launch AND skipping the alignment-padding rows the WMMA gemm2 computes. gemv-math
    # down-proj -> ~1e-4 vs the WMMA path (accumulation order; user-accepted). Prefill (M>threshold) keeps
    # the flag/WMMA gemm2 + gather_reduce below.
    if _MOE_G2FUSE and M <= _MOE_GEMM1_GEMV_MAX and block_m != 128 \
            and hasattr(fp8_wmma, "mmq_fp8_moe_gemm2_gather_reduce"):
        engaged(f"fp8_wmma.mmq_fp8_moe_gemm2_gather_reduce{_e2m1}")
        acc = _moe_time(
            "gemm2gather",
            lambda: fp8_wmma.mmq_fp8_moe_gemm2_gather_reduce(
                buf2, w2, w2_scales, sorted_ids, expert_ids, ntp, tw_flat, top_k, block_m,
                w_zeros=w2_zeros, weight_is_e2m1=weight_is_e2m1,
            ),
        )  # (M, K) fp32 — down-proj + top_k reduce in one
        _moe_report()
        return acc.to(x.dtype)

    ident = torch.arange(P, dtype=torch.int32, device=dev)
    # PREFILL gemm2 (non-scatter): register-tiled flag kernel at block_m==128 + group 128 (bit-exact,
    # ~1.1-1.4x); else the tiled wmma. Group = inter // (inter//group) = (w2 packed inter*8) / scale K-dim.
    _flag2 = _MOE_FLAG and block_m == 128 and \
        (w2.shape[-1] * 8) // w2_scales.shape[-1] in (32, 64, 128)  # flag supports group 32/64/128
    if _flag2:
        engaged(f"fp8_wmma.mmq_fp8_moe_gemm_flag{_e2m1}")
        out2 = _moe_time(
            "gemm2",
            lambda: fp8_wmma.mmq_fp8_moe_gemm_flag(
                buf2, w2, w2_scales, ident, expert_ids, ntp, 1, block_m,
                w_zeros=w2_zeros, weight_is_e2m1=weight_is_e2m1,
            ),
        )  # (P, K)
    else:
        engaged(f"fp8_wmma.mmq_fp8_moe_gemm({gemm2_kernel}{_e2m1})")
        out2 = _moe_time(
            "gemm2",
            lambda: fp8_wmma.mmq_fp8_moe_gemm(
                buf2, w2, w2_scales, ident, expert_ids, ntp, 1, block_m,
                kernel=gemm2_kernel, w_zeros=w2_zeros, weight_is_e2m1=weight_is_e2m1,
            ),
        )  # (P, K)
    engaged("fp8_wmma.mmq_fp8_moe_gather_reduce")
    acc = _moe_time(
        "gather",
        lambda: fp8_wmma.mmq_fp8_moe_gather_reduce(
            out2.contiguous(), sorted_ids, tw_flat, ntp, top_k
        ),
    )  # (M, K) fp32
    _moe_report()
    return acc.to(x.dtype)


def _w4a16_wide(group_size: int) -> int:
    """Pick the W4A16 register-direct b-load width. The kernel processes group_size/16 k-tiles per
    group and requires `wide` to divide that: 8=2x b128 (g=128), 4=b128 (g=64), 2=b64 (g=32; added
    in rdna4-hip-kernels 13fba94). g must be a multiple of 32."""
    ks = group_size // 16
    if ks % 8 == 0:
        return 8
    if ks % 4 == 0:
        return 4
    if ks % 2 == 0:
        return 2
    raise AssertionError(f"W4A16 wide MoE kernel needs group_size>=32 & %32==0 (got {group_size})")


def w4a16_moe(
    x: torch.Tensor,  # (M, K) activations — kept fp16, NO activation quant (the whole point)
    w13_rep: torch.Tensor,  # (E, 2*inter/16, K/16, 32, wide) register-direct (built in post_load)
    w13_scales: torch.Tensor,  # (E, 2*inter, K//g) fp16
    w13_zeros: torch.Tensor | None,  # (E, (2*inter)//8, K//g) int32 (AWQ) or None (symmetric)
    w2_rep: torch.Tensor,  # (E, K/16, inter/16, 32, wide)
    w2_scales: torch.Tensor,  # (E, K, inter//g) fp16
    w2_zeros: torch.Tensor | None,
    hidden: int,
    inter: int,
    group_size: int,
    *,
    topk_weights: torch.Tensor | None = None,  # (M, top_k) f32 precomputed route (GLM noaux_tc); None -> route here
    topk_ids: torch.Tensor | None = None,  # (M, top_k) i32 precomputed ids; None -> softmax+topk from router_logits
    router_logits: torch.Tensor | None = None,  # (M, E) raw gate logits — used ONLY when topk_ids is None
    top_k: int = 0,  # experts/token — required when routing here (topk_ids is None)
    renormalize: bool = False,  # renormalize the top-k weights (Qwen3.5-MoE: always True)
    weight_is_e2m1: bool = False,  # True -> decode w13/w2 nibbles as MXFP4 (OCP E2M1)
) -> torch.Tensor:
    """Grouped W4A16 MoE: fp16 activations DIRECT (no act-quant) via the register-direct
    mmq_regdirect_w4a16_moe kernel. This is the fix for the fp8-activation decode degradation on
    activation-sensitive models (GLM-4.7-Flash): int4 weights, fp16 acts, matching vLLM's W4A16.
    Weights are pre-repacked to w_rep_wide in post_load. block_m is fixed 16; `wide` from group_size.
    Route is precomputed for noaux_tc models (GLM) and the EP path; for a model that hands raw router
    logits (Qwen3.5-MoE MXFP4) pass `router_logits`+`top_k` and the fused softmax+topk runs here (same
    fallback as w4a8_moe). `weight_is_e2m1=True` decodes the weights as MXFP4 (OCP E2M1) — the
    register-direct MXFP4 path (fp16 acts, e2m1 weight decode, symmetric so no zeros). Returns (M, K)."""
    import moe_hip
    import fp8_wmma

    M = x.shape[0]
    E = w13_rep.shape[0]
    dev = x.device
    block_m = 16
    wide = _w4a16_wide(group_size)
    # Precomputed route (GLM/DeepSeek noaux_tc, or the EP path). Otherwise softmax+topk here — Qwen3.5-MoE
    # hands us raw router_logits with no model-side route, same fallback the w4a8_moe LDS path has.
    if topk_ids is None:
        assert router_logits is not None and top_k > 0, (
            "w4a16_moe needs a precomputed route (topk_ids/topk_weights) or router_logits + top_k"
        )
        topk_weights, topk_ids = _moe_time(
            "route", lambda: _softmax_topk_route(router_logits, top_k, renormalize)
        )
    top_k = topk_ids.shape[1]
    tw = topk_weights.to(torch.float32).contiguous()
    ti = topk_ids.to(torch.int32).contiguous()
    _empty = torch.empty(0, dtype=torch.int32, device=dev)

    _e2m1 = "+e2m1" if weight_is_e2m1 else ""
    engaged("moe_hip.moe_align")
    sorted_ids, expert_ids, ntp = _moe_time("align", lambda: moe_hip.moe_align(ti, E, block_m))
    P = sorted_ids.shape[0]
    x16 = _moe_time("cast", lambda: x.to(torch.float16).contiguous())

    engaged(f"fp8_wmma.mmq_regdirect_w4a16_moe{_e2m1}")
    out1 = _moe_time(
        "gemm1",
        lambda: fp8_wmma.mmq_regdirect_w4a16_moe(
            x16, w13_rep, w13_scales, w13_zeros if w13_zeros is not None else _empty,
            sorted_ids, expert_ids, ntp, 2 * inter, top_k, block_m, wide,
            weight_is_e2m1=weight_is_e2m1,
        ),
    )  # (P, 2*inter) fp16
    d = out1.shape[1] // 2
    if _TAIL_HIP and out1.dtype in _SILU_DTYPES:
        import tail_hip

        engaged("tail_hip.silu_and_mul")
        buf2 = _moe_time("silu", lambda: tail_hip.silu_and_mul(out1.contiguous()))
    else:
        buf2 = _moe_time(
            "silu",
            lambda: (F.silu(out1[:, :d].float()) * out1[:, d:].float()).to(torch.float16).contiguous(),
        )

    tw_flat = tw.reshape(-1).contiguous()
    # DECODE fast path: fused gemm2 + topk-weight + atomic scatter (NOT graph-capture-safe -> gated to
    # eager M<=2 + MINISGL_MOE_SCATTER). Otherwise the graph-safe unfused gemm2 + gather_reduce.
    if M <= 2 and _MOE_SCATTER:
        output = torch.zeros((M, hidden), dtype=torch.float32, device=dev)
        engaged(f"fp8_wmma.mmq_regdirect_w4a16_moe_scatter{_e2m1}")
        _moe_time(
            "gemm2scat",
            lambda: fp8_wmma.mmq_regdirect_w4a16_moe_scatter(
                buf2.contiguous(), w2_rep, w2_scales, sorted_ids, expert_ids, ntp, tw_flat, output,
                hidden, top_k, block_m, wide, w_zeros=w2_zeros, weight_is_e2m1=weight_is_e2m1,
            ),
        )  # writes output in place
        _moe_report()
        return output.to(x.dtype)

    ident = torch.arange(P, dtype=torch.int32, device=dev)
    engaged(f"fp8_wmma.mmq_regdirect_w4a16_moe{_e2m1}")
    out2 = _moe_time(
        "gemm2",
        lambda: fp8_wmma.mmq_regdirect_w4a16_moe(
            buf2.contiguous(), w2_rep, w2_scales, w2_zeros if w2_zeros is not None else _empty,
            ident, expert_ids, ntp, hidden, 1, block_m, wide, weight_is_e2m1=weight_is_e2m1,
        ),
    )  # (P, hidden) fp16
    engaged("fp8_wmma.mmq_fp8_moe_gather_reduce")
    output = _moe_time(
        "gather",
        lambda: fp8_wmma.mmq_fp8_moe_gather_reduce(
            out2.contiguous(), sorted_ids, tw_flat, ntp, top_k
        ),
    )  # (M, hidden) fp32
    _moe_report()
    return output.to(x.dtype)


def w4a16_linear(
    x: torch.Tensor,  # (M, K) fp16 activations — DIRECT (no act-quant)
    w_rep_wide: torch.Tensor,  # register-direct wide weights (built in process_weights_after_load)
    scales: torch.Tensor,  # (N, K//g) fp16
    w_zeros: torch.Tensor | None,  # (N//8, K//g) int32 (AWQ) or None (symmetric)
    group_size: int,
    N: int,
) -> torch.Tensor:
    """Dense W4A16 GEMM (fp16 acts direct) via mmq_regdirect_w4a16_wide — the fp16-act twin of
    w4a8_linear, for the GLM shared expert / dense layers when MINISGL_MOE_W4A16 is on."""
    import fp8_wmma

    wide = _w4a16_wide(group_size)
    x16 = x if x.dtype == torch.float16 else x.to(torch.float16)
    z = w_zeros if w_zeros is not None else torch.empty(0, dtype=torch.int32, device=x.device)
    engaged("fp8_wmma.mmq_regdirect_w4a16_wide")
    return fp8_wmma.mmq_regdirect_w4a16_wide(x16.contiguous(), w_rep_wide, scales, z, N, wide)


def w8a8_moe(
    x: torch.Tensor,  # (M, K) activations
    w13: torch.Tensor,  # (E, 2*inter, K) f8_e4m3 — gate|up stacked, op layout (natural)
    w13_scales: torch.Tensor,  # (E, 2*inter) f32 per-output-channel
    w2: torch.Tensor,  # (E, K, inter) f8_e4m3
    w2_scales: torch.Tensor,  # (E, K) f32 per-output-channel
    gating_output: torch.Tensor | None,  # (M, E); ignored when topk_ids/topk_weights are given
    top_k: int,
    renormalize: bool,
    *,
    topk_weights: torch.Tensor | None = None,  # (M, top_k) f32 — precomputed route
    topk_ids: torch.Tensor | None = None,  # (M, top_k) i32 — precomputed expert ids
    kernel: str = "wmma",
    block_m: int | None = None,  # None -> derive the WMMA tile height from the workload (_moe_block_m)
) -> torch.Tensor:
    """Grouped W8A8-fp8 MoE forward: topk -> moe_align -> grouped GEMM(w13) -> silu_and_mul
    -> grouped GEMM(w2) -> topk-weighted gather-reduce. A strict simplification of `w4a8_moe`
    (no zeros, no per-K-group scale: fp8 weights carry a per-output-channel f32 scale folded
    once in the epilogue). Returns (M, K)."""
    import torch.nn.functional as F
    import fp8_wmma

    M, K = x.shape
    E = w13.shape[0]
    dev = x.device
    if block_m is None:  # derive the grouped-GEMM tile from the workload (16 at decode, up to 128 at prefill)
        block_m = _moe_block_m(M, E, top_k)
    # Per-GEMM kernel pick (== w4a8_moe): at decode (M<=2) gemm1's wide 2*inter output over a few
    # real tokens is far faster as a per-token GEMV than WMMA over mostly-padding tiles; gemm2's
    # K output favours WMMA. Prefill (M>2) keeps the passed/default kernel for both.
    gemm1_kernel = "gemv" if M <= _MOE_GEMM1_GEMV_MAX else kernel
    gemm2_kernel = kernel

    def _route():
        try:
            from vllm import _custom_ops as vllm_ops
        except ImportError:
            probs = torch.softmax(gating_output.float(), dim=-1)
            tw, ti = torch.topk(probs, top_k, dim=-1)
            if renormalize:
                tw = tw / (tw.sum(dim=-1, keepdim=True) + 1e-20)
            return tw.contiguous(), ti.to(torch.int32).contiguous()

        tw = torch.empty(M, top_k, dtype=torch.float32, device=dev)
        ti = torch.empty(M, top_k, dtype=torch.int32, device=dev)
        tei = torch.empty(M, top_k, dtype=torch.int32, device=dev)  # token_expert_indices scratch
        vllm_ops.topk_softmax(tw, ti, tei, gating_output.float(), renormalize)
        return tw, ti

    # Precomputed route (ZAYA top-1 + MOD) or fused softmax+topk fallback.
    if topk_ids is None:
        topk_weights, topk_ids = _moe_time("route", _route)
    else:
        assert topk_weights is not None, "topk_weights required when topk_ids is given"
        topk_weights = topk_weights.to(torch.float32).contiguous()
        topk_ids = topk_ids.to(torch.int32).contiguous()

    # moe_align: native HIP (moe_hip). The former MINISGL_MOE_ALIGN=0 vLLM reference is gone — the
    # lean image has no vllm, and moe_hip.moe_align is the validated drop-in for that host op.
    import moe_hip

    engaged("moe_hip.moe_align")
    sorted_ids, expert_ids, ntp = _moe_time("align", lambda: moe_hip.moe_align(topk_ids, E, block_m))
    P = sorted_ids.shape[0]

    x16 = _moe_time("cast", lambda: x.to(torch.float16).contiguous())
    engaged(f"fp8_wmma.mmq_w8a8_moe_gemm({gemm1_kernel})")
    out1 = _moe_time(
        "gemm1",
        lambda: fp8_wmma.mmq_w8a8_moe_gemm(
            x16, w13, w13_scales, sorted_ids, expert_ids, ntp, top_k, block_m, gemm1_kernel,
        ),
    )  # (P, 2*inter)
    d = out1.shape[1] // 2
    # Gated SiLU-mul via the dtype-generic native HIP tail_hip.silu_and_mul (one launch, fp32
    # internal, no temps); MINISGL_TAIL_HIP=0 reverts to the torch ref. (The gemm1-epilogue fused
    # silu is wmma-only -> unusable at decode where gemm1 must be gemv.)
    if _TAIL_HIP and out1.dtype in _SILU_DTYPES:
        import tail_hip  # canonical package: silu_and_mul is a module-level callable

        engaged("tail_hip.silu_and_mul")
        buf2 = _moe_time("silu", lambda: tail_hip.silu_and_mul(out1.contiguous()))
    else:
        buf2 = _moe_time(
            "silu",
            lambda: (F.silu(out1[:, :d].float()) * out1[:, d:].float()).to(torch.float16).contiguous(),
        )

    tw_flat = topk_weights.reshape(-1).float().contiguous()
    # DECODE fast path: fuse gemm2 + topk-weight + reduce into ONE atomic-scatter kernel (NOT
    # HIP-graph-capture-safe -> gated to eager decode M<=2 + MINISGL_MOE_SCATTER). Prefill (M>2)
    # keeps the unfused gemm2 + contention-free gather_reduce (graph-safe).
    if M <= 2 and _MOE_SCATTER:
        acc = torch.zeros((M, K), dtype=torch.float32, device=dev)
        engaged(f"fp8_wmma.mmq_w8a8_moe_gemm_scatter({gemm2_kernel})")
        _moe_time(
            "gemm2scat",
            lambda: fp8_wmma.mmq_w8a8_moe_gemm_scatter(
                buf2, w2, w2_scales, sorted_ids, expert_ids, ntp, tw_flat, acc, top_k, block_m,
                gemm2_kernel,
            ),
        )  # writes acc in place
        _moe_report()
        return acc.to(x.dtype)

    ident = torch.arange(P, dtype=torch.int32, device=dev)
    # PREFILL gemm2: register-tiled flag kernel at block_m==128 (fp8, per-channel scale — no group gate);
    # bit-exact, ~1.18-1.30x over tiled wmma. else the tiled wmma.
    if _MOE_FLAG and block_m == 128:
        engaged("fp8_wmma.mmq_w8a8_moe_gemm_flag")
        out2 = _moe_time(
            "gemm2",
            lambda: fp8_wmma.mmq_w8a8_moe_gemm_flag(
                buf2, w2, w2_scales, ident, expert_ids, ntp, 1, block_m,
            ),
        )  # (P, K)
    else:
        engaged(f"fp8_wmma.mmq_w8a8_moe_gemm({gemm2_kernel})")
        out2 = _moe_time(
            "gemm2",
            lambda: fp8_wmma.mmq_w8a8_moe_gemm(
                buf2, w2, w2_scales, ident, expert_ids, ntp, 1, block_m, gemm2_kernel,
            ),
        )  # (P, K)
    engaged("fp8_wmma.mmq_w8a8_moe_gather_reduce")
    acc = _moe_time(
        "gather",
        lambda: fp8_wmma.mmq_w8a8_moe_gather_reduce(
            out2.contiguous(), sorted_ids, tw_flat, ntp, top_k
        ),
    )  # (M, K) fp32
    _moe_report()
    return acc.to(x.dtype)


def w8a8_moe_regdirect(
    x: torch.Tensor,  # (M, K) activations (fp16/bf16); act-fp8-quantized inside the op
    w13_rep: torch.Tensor,  # (E, 2*inter/16, K/16, 32, wide) register-direct fp8 (built in post_load)
    w13_scales: torch.Tensor,  # (E, 2*inter) f32 per-output-channel
    w2_rep: torch.Tensor,  # (E, K/16, inter/16, 32, wide)
    w2_scales: torch.Tensor,  # (E, K) f32 per-output-channel
    top_k: int,
    *,
    topk_weights: torch.Tensor,  # (M, top_k) f32 — precomputed route (ZAYA top-1 + MOD)
    topk_ids: torch.Tensor,  # (M, top_k) i32
    block_m: int = 16,
    wide: int = 2,  # b128 (fp8: 2 K16-tiles/lane); 1=b64, 4=2x b128
) -> torch.Tensor:
    """Register-direct b128 (LDS-bypass) W8A8-fp8 grouped MoE — the fp8 twin of w4a16_moe and a
    drop-in for w8a8_moe. gemm1 (mmq_regdirect_w8a8_moe) -> silu_and_mul -> gemm2 (fused atomic
    scatter at eager decode, or unfused gemm2 + gather_reduce for the graph-safe / prefill path).
    Bit-exact vs w8a8_moe (same act-fp8 quant + fp8 weights + WMMA chain; weights pre-permuted into
    WMMA-B lane order). Route is always precomputed (ZAYA). Returns (M, K)."""
    import torch.nn.functional as F
    import moe_hip
    import fp8_wmma

    M, K = x.shape
    E = w13_rep.shape[0]
    dev = x.device
    N13 = w13_scales.shape[1]  # 2*inter (gemm1 output width)
    tw = topk_weights.to(torch.float32).contiguous()
    ti = topk_ids.to(torch.int32).contiguous()

    engaged("moe_hip.moe_align")
    sorted_ids, expert_ids, ntp = _moe_time("align", lambda: moe_hip.moe_align(ti, E, block_m))
    P = sorted_ids.shape[0]

    x16 = _moe_time("cast", lambda: x.to(torch.float16).contiguous())
    engaged("fp8_wmma.mmq_regdirect_w8a8_moe")
    out1 = _moe_time(
        "gemm1",
        lambda: fp8_wmma.mmq_regdirect_w8a8_moe(
            x16, w13_rep, w13_scales, sorted_ids, expert_ids, ntp, N13, top_k, block_m, wide,
        ),
    )  # (P, 2*inter) fp16
    d = out1.shape[1] // 2
    if _TAIL_HIP and out1.dtype in _SILU_DTYPES:
        import tail_hip

        engaged("tail_hip.silu_and_mul")
        buf2 = _moe_time("silu", lambda: tail_hip.silu_and_mul(out1.contiguous()))
    else:
        buf2 = _moe_time(
            "silu",
            lambda: (F.silu(out1[:, :d].float()) * out1[:, d:].float()).to(torch.float16).contiguous(),
        )

    tw_flat = tw.reshape(-1).contiguous()
    # DECODE fast path: fused gemm2 + topk-weight + atomic scatter (NOT graph-capture-safe -> gated to
    # eager M<=2 + MINISGL_MOE_SCATTER). Otherwise the graph-safe unfused gemm2 + gather_reduce.
    if M <= 2 and _MOE_SCATTER:
        acc = torch.zeros((M, K), dtype=torch.float32, device=dev)
        engaged("fp8_wmma.mmq_regdirect_w8a8_moe_scatter")
        _moe_time(
            "gemm2scat",
            lambda: fp8_wmma.mmq_regdirect_w8a8_moe_scatter(
                buf2.contiguous(), w2_rep, w2_scales, sorted_ids, expert_ids, ntp, tw_flat, acc,
                K, top_k, block_m, wide,
            ),
        )  # writes acc in place
        _moe_report()
        return acc.to(x.dtype)

    ident = torch.arange(P, dtype=torch.int32, device=dev)
    engaged("fp8_wmma.mmq_regdirect_w8a8_moe")
    out2 = _moe_time(
        "gemm2",
        lambda: fp8_wmma.mmq_regdirect_w8a8_moe(
            buf2, w2_rep, w2_scales, ident, expert_ids, ntp, K, 1, block_m, wide,
        ),
    )  # (P, K)
    engaged("fp8_wmma.mmq_w8a8_moe_gather_reduce")
    acc = _moe_time(
        "gather",
        lambda: fp8_wmma.mmq_w8a8_moe_gather_reduce(
            out2.contiguous(), sorted_ids, tw_flat, ntp, top_k
        ),
    )  # (M, K) fp32
    _moe_report()
    return acc.to(x.dtype)


# --- RXF (Rotated eXtra Fast) W4(NL)-A8(int8) path -------------------------------------------
# Distinct from the fp8 W4A8 above: weights are an NL (non-uniform) int4 codebook, activations are
# int8 (not fp8), and a fixed block-diagonal Hadamard rotation is applied to the activation at
# runtime (and was applied to the weights offline) so it cancels in the dot while spreading
# activation outliers. Native HIP via fp8_wmma (rxf_* ops; folded from the old rxf_hip package).

_RXF_NL: dict = {}


def _rxf_nl(dev: torch.device) -> torch.Tensor:
    """Cached int8[16] NL codebook on `dev` (fp8_wmma.RXF_NL_DEFAULT == the Triton _NL_DEFAULT)."""
    key = str(dev)
    t = _RXF_NL.get(key)
    if t is None:
        import fp8_wmma  # rxf folded into fp8_wmma

        t = torch.tensor(fp8_wmma.RXF_NL_DEFAULT, dtype=torch.int8, device=dev)
        _RXF_NL[key] = t
    return t


def rxf_linear(
    x: torch.Tensor,  # (M, K) activations (bf16/fp16)
    w_packed: torch.Tensor,  # (N, K/2) uint8 NL indices
    w_scale: torch.Tensor,  # (N, K/32) fp16 per-group weight scale
    bias: torch.Tensor | None,
    span: int = 32,
) -> torch.Tensor:
    """Dense RXF W4A8: rotate+int8-quant the activation, then int8 . NL-int4 GEMM -> bf16 (M,N).
    The rotate_quant fuses FWHT-span + per-token int8 quant; linear picks WMMA (M>2) / GEMV (M<=2)."""
    import fp8_wmma  # rxf folded into fp8_wmma

    engaged("fp8_wmma.rxf_rotate_quant_int8")
    q, a_scale = fp8_wmma.rxf_rotate_quant_int8(x.contiguous(), span)
    engaged("fp8_wmma.rxf_linear")
    return fp8_wmma.rxf_linear(q, a_scale, w_packed, w_scale, _rxf_nl(x.device), bias)


def rxf_moe(
    x: torch.Tensor,  # (M, K) activations
    w13: torch.Tensor,  # (E, 2*inter, K/2) uint8
    w13_scales: torch.Tensor,  # (E, 2*inter, K/32) fp16
    w2: torch.Tensor,  # (E, K, inter/2) uint8
    w2_scales: torch.Tensor,  # (E, K, inter/32) fp16
    gating_output: torch.Tensor | None,  # (M, E); ignored when topk_ids/topk_weights are given
    top_k: int,
    renormalize: bool,
    *,
    topk_weights: torch.Tensor | None = None,  # (M, top_k) f32 — precomputed route (e.g. noaux_tc)
    topk_ids: torch.Tensor | None = None,  # (M, top_k) i32 — precomputed expert ids
    span: int = 32,
    block_m: int = 16,
) -> torch.Tensor:
    """Grouped RXF W4A8 MoE: rotate+quant -> grouped GEMM(w13) -> silu_and_mul -> rotate+quant
    -> grouped GEMM(w2) -> topk-weighted gather-reduce. Mirrors w4a8_moe's dispatch (moe_align),
    int8/NL on the GEMMs. Returns (M, K).

    Either pass raw ``gating_output`` (softmax+topk computed here) OR a precomputed
    ``topk_weights``/``topk_ids`` route (GLM/DeepSeek noaux_tc computed in the model — its normalize
    + scaling are already folded in, so pass them through unchanged; renormalize is ignored)."""
    import torch.nn.functional as F
    import moe_hip
    import fp8_wmma  # rxf folded into fp8_wmma

    M, K = x.shape
    E = w13.shape[0]
    dev = x.device
    nl = _rxf_nl(dev)

    # Precomputed route (GLM/DeepSeek noaux_tc) or torch softmax+topk fallback (lean image has no
    # vllm; matches the former _moe_C.topk_softmax).
    if topk_ids is not None:
        assert topk_weights is not None, "topk_weights required when topk_ids is given"
        tw = topk_weights.to(torch.float32).contiguous()
        ti = topk_ids.to(torch.int32).contiguous()
    else:
        probs = torch.softmax(gating_output.float(), dim=-1)
        tw, ti = torch.topk(probs, top_k, dim=-1)
        if renormalize:
            tw = tw / (tw.sum(dim=-1, keepdim=True) + 1e-20)
        tw = tw.contiguous()
        ti = ti.to(torch.int32).contiguous()

    # align: native HIP drop-in for the former vLLM moe_align_block_size host op.
    engaged("moe_hip.moe_align")
    sorted_ids, expert_ids, ntp = moe_hip.moe_align(ti, E, block_m)
    P = sorted_ids.shape[0]

    # gemm1: rotate+quant the activation (gathered by sorted_ids inside the GEMM), grouped over w13.
    # Per-GEMM kernel selection (== w4a8_moe): at decode (M<=2) gemm1's wide 2*inter output over a
    # few real tokens is far faster as a per-token GEMV than WMMA over mostly-padding tiles.
    engaged("fp8_wmma.rxf_rotate_quant_int8")
    q, a_scale = fp8_wmma.rxf_rotate_quant_int8(x.contiguous(), span)
    gemm1 = fp8_wmma.rxf_moe_gemv if M <= 2 else fp8_wmma.rxf_moe_gemm
    engaged(f"fp8_wmma.rxf_{'moe_gemv' if M <= 2 else 'moe_gemm'}")
    out1 = gemm1(
        q, a_scale, w13, w13_scales, nl, sorted_ids, expert_ids, ntp, top_k, block_m, M * top_k
    )  # (P, 2*inter) bf16
    d = out1.shape[1] // 2

    from minisgl.layers import _tail_hip

    if _tail_hip.active(out1):
        buf2 = _tail_hip.silu_and_mul(out1.contiguous())
    else:
        buf2 = (F.silu(out1[:, :d].float()) * out1[:, d:].float()).to(torch.bfloat16).contiguous()

    # gemm2: rotate+quant the intermediate (w2 was rotated offline too).
    q2, a_scale2 = fp8_wmma.rxf_rotate_quant_int8(buf2.contiguous(), span)
    tw_flat = tw.reshape(-1).float().contiguous()

    # DECODE fast path (mirrors w4a8_moe): fuse gemm2 + topk-weight + reduce into ONE kernel via the
    # atomic scatter, removing the (P,K) out2 materialization + the separate gather. The atomicAdd is
    # NOT HIP-graph-capture-safe -> gated to eager decode (M<=2) and MINISGL_MOE_SCATTER.
    if M <= 2 and _MOE_SCATTER:
        acc = torch.zeros((M, K), dtype=torch.float32, device=dev)
        engaged("fp8_wmma.rxf_moe_gemm_scatter")
        fp8_wmma.rxf_moe_gemm_scatter(
            q2, a_scale2, w2, w2_scales, nl, sorted_ids, expert_ids, ntp, tw_flat, acc,
            top_k, block_m, M * top_k
        )  # writes acc in place
        return acc.to(x.dtype)

    # PREFILL: unfused gemm2 (identity gather) + contention-free gather-reduce (graph-safe).
    ident = torch.arange(P, dtype=torch.int32, device=dev)
    engaged("fp8_wmma.rxf_moe_gemm")
    out2 = fp8_wmma.rxf_moe_gemm(
        q2, a_scale2, w2, w2_scales, nl, ident, expert_ids, ntp, 1, block_m, P
    )  # (P, K) bf16
    engaged("fp8_wmma.rxf_moe_gather_reduce")
    acc = fp8_wmma.rxf_moe_gather_reduce(
        out2, sorted_ids, tw_flat, ntp, M, top_k, M * top_k
    )  # (M, K) fp32
    return acc.to(x.dtype)


def rxf_moe_regdirect(
    x: torch.Tensor,  # (M, K) activations
    w13_rep: torch.Tensor,  # (E, 2*inter/16, ktiles/wide, 32, wide) register-direct NL codes
    w13_scales: torch.Tensor,  # (E, 2*inter, K/32) fp16 per-group scale
    w2_rep: torch.Tensor,  # (E, K/16, ktiles/wide, 32, wide)
    w2_scales: torch.Tensor,  # (E, K, inter/32) fp16
    top_k: int,
    *,
    topk_weights: torch.Tensor,  # (M, top_k) f32 — precomputed route
    topk_ids: torch.Tensor,  # (M, top_k) i32
    span: int = 32,
    block_m: int = 16,
    wide: int = 4,  # b128 (RXF: 4 K16-tiles/lane); 2=b64
) -> torch.Tensor:
    """Register-direct b128 (LDS-bypass) RXF W4(NL)-A8 grouped MoE — a drop-in for rxf_moe. Same
    rotate+quant -> grouped GEMM -> silu_and_mul -> rotate+quant -> grouped GEMM -> gather-reduce
    contract, but the WMMA weight operand is loaded direct from global into registers (b128) with no
    B in LDS and no K-loop __syncthreads. moe_gemm_regdirect has no fused-scatter twin, so this always
    uses the graph-safe unfused gemm2 + gather_reduce (no eager-only atomic path). Bit-exact vs
    rxf_moe. Route is always precomputed. Returns (M, K)."""
    import torch.nn.functional as F
    import moe_hip
    import fp8_wmma  # rxf folded into fp8_wmma

    M, K = x.shape
    E = w13_rep.shape[0]
    dev = x.device
    nl = _rxf_nl(dev)
    tw = topk_weights.to(torch.float32).contiguous()
    ti = topk_ids.to(torch.int32).contiguous()

    engaged("moe_hip.moe_align")
    sorted_ids, expert_ids, ntp = moe_hip.moe_align(ti, E, block_m)
    P = sorted_ids.shape[0]

    engaged("fp8_wmma.rxf_rotate_quant_int8")
    q, a_scale = fp8_wmma.rxf_rotate_quant_int8(x.contiguous(), span)
    engaged("fp8_wmma.rxf_moe_gemm_regdirect")
    out1 = fp8_wmma.rxf_moe_gemm_regdirect(
        q, a_scale, w13_rep, w13_scales, nl, sorted_ids, expert_ids, ntp,
        top_k, block_m, M * top_k, wide,
    )  # (P, 2*inter) bf16
    d = out1.shape[1] // 2

    from minisgl.layers import _tail_hip

    if _tail_hip.active(out1):
        buf2 = _tail_hip.silu_and_mul(out1.contiguous())
    else:
        buf2 = (F.silu(out1[:, :d].float()) * out1[:, d:].float()).to(torch.bfloat16).contiguous()

    q2, a_scale2 = fp8_wmma.rxf_rotate_quant_int8(buf2.contiguous(), span)
    tw_flat = tw.reshape(-1).float().contiguous()
    ident = torch.arange(P, dtype=torch.int32, device=dev)
    engaged("fp8_wmma.rxf_moe_gemm_regdirect")
    out2 = fp8_wmma.rxf_moe_gemm_regdirect(
        q2, a_scale2, w2_rep, w2_scales, nl, ident, expert_ids, ntp,
        1, block_m, P, wide,
    )  # (P, K) bf16
    engaged("fp8_wmma.rxf_moe_gather_reduce")
    acc = fp8_wmma.rxf_moe_gather_reduce(out2, sorted_ids, tw_flat, ntp, M, top_k, M * top_k)  # (M, K) f32
    return acc.to(x.dtype)


def _pick_dense_kernel(m: int, weight_is_e2m1: bool = False, group_size: int = 128) -> str:
    """Per-M dense-linear kernel selection, at the MEASURED crossovers (gfx1201).

    - m <= gemv_max -> decode_gemv: a streaming GEMV that reads each weight once and dots it against
      all M rows (amortizing the weight read), writing straight to `out`. Serves the decode/decode-
      batch band; measured crossover int4 M<=8, e2m1 M<=16 (also the only e2m1-capable small-M kernel;
      decode_gemv's in-kernel cap is 16).
    - m >= _W4A8_PREFILL_TILED_MIN -> wmma_tiled_tuned: the tuned tiled WMMA GEMM. Dominates the prefill
      regime ~2-4x over prefill_wmma (re-bench), for BOTH int4 AND e2m1 now that the tiled kernel is
      e2m1-bit-exact and carries the packed-store. Bit-exact + graph-capture-safe (writes to `out`,
      no in-op at::zeros). One dtype-generic rule — no int4-vs-e2m1 branch.
    - mid-band (gemv_max < m < _W4A8_PREFILL_TILED_MIN) -> prefill_wmma: its conservative config still
      wins the small-M/wide-N corner (re-bench: tiled loses only at N>=6144, M<=32).

    The dead small-M WMMA variants (nsplit/splitk/regdirect_shuffle) were REMOVED (they allocated an
    in-op at::zeros((M,N),f32) that blew up VRAM under CUDA-graph capture); no override knob remains.
    """
    gemv_max = _W4A8_GEMV_MAX_E2M1 if weight_is_e2m1 else _W4A8_GEMV_MAX_INT4
    if m <= gemv_max:
        return "decode_gemv"
    return "wmma_tiled_tuned" if m >= _W4A8_PREFILL_TILED_MIN else "prefill_wmma"


# gemv<->wmma crossover per decode path (measured). decode_gemv asserts M<=16 in-kernel, so E2M1 caps
# at 16; int4 crosses lower because its WMMA tile reclaims M=16 on a dense model. Both >= the M<=2
# decode fast path, so the served decode batch (<= max_running_req) stays on the faster kernel.
_W4A8_GEMV_MAX_INT4 = 8
_W4A8_GEMV_MAX_E2M1 = 16
# Prefill regime: wmma_tiled_tuned dominates from here up (~2-4x prefill_wmma, bit-exact, graph-safe,
# both dtypes). Below it (small-M/wide-N mid-band) prefill_wmma's conservative config still wins
# (re-bench: tiled loses only at N>=6144, M<=32). True prefill/chunked-prefill M is always >> 64.
_W4A8_PREFILL_TILED_MIN = 64


def w4a8_linear(
    x: torch.Tensor,
    w_packed: torch.Tensor,  # (N, K/8) int32, op layout
    scales: torch.Tensor,  # (N, K/group) fp16
    w_zeros: torch.Tensor | None,  # (N/8, K/group) int32 (AWQ asym) or None (sym)
    group_size: int,
    kernel: str | None = None,
    weight_is_e2m1: bool = False,  # True -> decode nibbles as MXFP4 (OCP E2M1); w_zeros must be None
) -> torch.Tensor:
    """Dense W4A8 GEMM: (M, K) @ (N, K)^T -> (M, N). Output follows the activation dtype (fp16 OR
    bf16 — the kernel is activation-dtype-generic), so a bf16 model runs cast-free (the caller's
    out.to(x.dtype) is then a no-op). `weight_is_e2m1=True` selects the kernel's MXFP4 (E2M1) weight
    decode instead of uniform int4 (scales are the E8M0 group exponents folded to fp16; w_zeros MUST
    be None — the op asserts symmetric)."""
    import fp8_wmma

    x2d = x  # native dtype straight into the op (fp16 or bf16); no bf16->fp16 round-trip
    if kernel is None:
        kernel = _pick_dense_kernel(x2d.shape[0], weight_is_e2m1, group_size)
    engaged(f"fp8_wmma.mmq_fp8_gemm({kernel}{'+e2m1' if weight_is_e2m1 else ''})")
    return fp8_wmma.mmq_fp8_gemm(
        x2d, w_packed, scales, kernel=kernel, w_zeros=w_zeros, weight_is_e2m1=weight_is_e2m1
    )


def w4a8_linear_silu(
    x: torch.Tensor,
    w_packed: torch.Tensor,  # (2*inter, K/8) int32 [gate|up], op layout
    scales: torch.Tensor,  # (2*inter, K/group) fp16
    w_zeros: torch.Tensor | None,  # ((2*inter)/8, K/group) int32 (AWQ) or None (sym)
    group_size: int,
    weight_is_e2m1: bool = False,
) -> torch.Tensor:
    """FUSED dense gate_up GEMV + silu_and_mul: (M, K) @ (2*inter, K)^T -> silu(gate)*up -> (M, inter).
    ONE launch, no (M, 2*inter) HBM round-trip. Decode-only (M<=16, K%512==0, group_size%32==0);
    BIT-EXACT to w4a8_linear(gate_up) + silu_and_mul. Output follows x's dtype (fp16/bf16)."""
    import fp8_wmma

    engaged(f"fp8_wmma.mmq_fp8_gemm_silu({'e2m1' if weight_is_e2m1 else 'int4'})")
    return fp8_wmma.mmq_fp8_gemm_silu(
        x, w_packed, scales, w_zeros=w_zeros, weight_is_e2m1=weight_is_e2m1
    )


def w8a8_dense_linear(
    x: torch.Tensor,  # (M, K) fp16/bf16 activations
    w_fp8: torch.Tensor,  # (N, K) uint8 (e4m3 bits), op layout (natural row-major)
    scales: torch.Tensor,  # (N,) f32 per-output-channel weight scale
    kernel: str | None = None,
) -> torch.Tensor:
    """Dense W8A8-fp8 linear: (M, K) @ (N, K)^T -> (M, N). e4m3 weights carrying a per-output-channel
    f32 scale, with dynamic per-token fp8 activations quantized INSIDE the kernel (RedHatAI
    *-FP8-dynamic scheme). Runs the GENUINE dense kernel `mmq_w8a8_gemm` — flagship register-tiled
    prefill + dense fp8 gemv decode, activation-dtype-generic (bf16 goes straight in, NO fp16 cast).
    This is a real dense GEMM: NO single-expert / sorted_token_ids / expert grouping, NO *_moe_* kernel.
    `kernel=None` auto-dispatches (M<=8 -> decode_gemv, else prefill_tiled)."""
    import fp8_wmma

    engaged(f"fp8_wmma.mmq_w8a8_gemm({kernel or 'auto'})")
    return fp8_wmma.mmq_w8a8_gemm(x, w_fp8, scales, kernel)
