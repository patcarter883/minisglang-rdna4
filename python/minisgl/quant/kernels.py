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
    block_m: int = 16,
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
    import w4a8_fp8_wmma

    M, K = x.shape
    E = w13.shape[0]
    dev = x.device
    # Decode fast path is PER-GEMM: the two grouped GEMMs want OPPOSITE kernels at M<=2 (measured on
    # gfx1201, Qwen3.6-35B). gemm1 (w13, wide 2*inter output, gather-by-sorted, top_k>1): the scalar
    # GEMV is ~8.7x faster than the prefill WMMA (35us vs 306us). gemm2 (w2, K output, identity
    # gather, top_k==1): WMMA is ~6.8x faster than GEMV (50us vs 339us) — the GEMV path
    # underperforms for that shape. So pick gemv for gemm1, keep WMMA for gemm2. Together ~85us vs
    # ~356us with the old all-WMMA default. Prefill (M>2) keeps the passed/default kernel for both.
    gemm1_kernel = "gemv" if M <= 2 else kernel
    gemm2_kernel = kernel

    # softmax + top-k (+ renormalize) route. The lean (vllm-free) image has no fused kernel, so this
    # is pure torch (softmax -> topk -> optional renorm -> int32 ids), matching what the vLLM
    # _moe_C.topk_softmax fused op computed. When vLLM IS present (the legacy combined image), prefer
    # its fused kernel — a single launch vs the torch chain.
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

    # Precomputed route (e.g. GLM/DeepSeek noaux_tc: sigmoid + correction bias + group top-k +
    # normalize + scale, done in the model). Otherwise fall back to fused softmax+topk here.
    if topk_ids is None:
        topk_weights, topk_ids = _moe_time("route", _route)
    else:
        assert topk_weights is not None, "topk_weights required when topk_ids is given"
        topk_weights = topk_weights.to(torch.float32).contiguous()
        topk_ids = topk_ids.to(torch.int32).contiguous()

    # moe_align: native HIP (moe_hip). The former MINISGL_MOE_ALIGN=0 vLLM reference is gone — the
    # lean image has no vllm, and moe_hip.moe_align is the validated drop-in for that host op.
    import moe_hip

    sorted_ids, expert_ids, ntp = _moe_time("align", lambda: moe_hip.moe_align(topk_ids, E, block_m))
    P = sorted_ids.shape[0]

    # The W4A8 kernel is now activation-dtype-generic (fp16 OR bf16), so pass activations in their
    # NATIVE dtype — a bf16 model no longer round-trips bf16->fp16->bf16 here (out1 follows x's dtype).
    x16 = _moe_time("cast", lambda: x.contiguous())
    out1 = _moe_time(
        "gemm1",
        lambda: w4a8_fp8_wmma.mmq_fp8_moe_gemm(
            x16, w13, w13_scales, sorted_ids, expert_ids, ntp, top_k, block_m,
            kernel=gemm1_kernel, w_zeros=w13_zeros, weight_is_e2m1=weight_is_e2m1,
        ),
    )  # (P, 2*inter) in x's dtype
    d = out1.shape[1] // 2
    # Gated SiLU-mul: the fp16 MoE intermediates (out1 is fp16) now route through the dtype-generic
    # native HIP tail_hip.silu_and_mul (one launch, fp32-internal, no temps) — replacing the multi-op
    # torch chain (silu+mul+float+cast+contiguous). MINISGL_TAIL_HIP=0 reverts to the torch ref.
    # (The gemm1-epilogue fused silu is wmma-only -> unusable at decode where gemm1 must be gemv.)
    if _TAIL_HIP and out1.dtype in _SILU_DTYPES:
        import tail_hip  # canonical package: silu_and_mul is a module-level callable

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

            _moe_time(
                "gemm2scat",
                lambda: moe_splitk_hip.moe_gemm_splitk_scatter(
                    buf2, w2, w2_scales, w2_zeros, sorted_ids, expert_ids, ntp, tw_flat, acc,
                    top_k, block_m, _MOE_SPLITK,
                ),
            )  # writes acc in place (atomic scatter over experts AND split_k K-slices)
        else:
            _moe_time(
                "gemm2scat",
                lambda: w4a8_fp8_wmma.mmq_fp8_moe_gemm_scatter(
                    buf2, w2, w2_scales, sorted_ids, expert_ids, ntp, tw_flat, acc, top_k, block_m,
                    kernel=gemm2_kernel, w_zeros=w2_zeros, weight_is_e2m1=weight_is_e2m1,
                ),
            )  # writes acc in place
        _moe_report()
        return acc.to(x.dtype)

    ident = torch.arange(P, dtype=torch.int32, device=dev)
    out2 = _moe_time(
        "gemm2",
        lambda: w4a8_fp8_wmma.mmq_fp8_moe_gemm(
            buf2, w2, w2_scales, ident, expert_ids, ntp, 1, block_m,
            kernel=gemm2_kernel, w_zeros=w2_zeros, weight_is_e2m1=weight_is_e2m1,
        ),
    )  # (P, K)
    acc = _moe_time(
        "gather",
        lambda: w4a8_fp8_wmma.mmq_fp8_moe_gather_reduce(
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
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    """Grouped W4A16 MoE: fp16 activations DIRECT (no act-quant) via the register-direct
    mmq_regdirect_w4a16_moe kernel. This is the fix for the fp8-activation decode degradation on
    activation-sensitive models (GLM-4.7-Flash): int4 weights, fp16 acts, matching vLLM's W4A16.
    Weights are pre-repacked to w_rep_wide in post_load. block_m is fixed 16; `wide` from group_size.
    Route is always precomputed (GLM noaux_tc). Returns (M, K)."""
    import moe_hip
    import w4a8_fp8_wmma

    M = x.shape[0]
    E = w13_rep.shape[0]
    dev = x.device
    block_m = 16
    wide = _w4a16_wide(group_size)
    top_k = topk_ids.shape[1]
    tw = topk_weights.to(torch.float32).contiguous()
    ti = topk_ids.to(torch.int32).contiguous()
    _empty = torch.empty(0, dtype=torch.int32, device=dev)

    sorted_ids, expert_ids, ntp = _moe_time("align", lambda: moe_hip.moe_align(ti, E, block_m))
    x16 = _moe_time("cast", lambda: x.to(torch.float16).contiguous())

    out1 = _moe_time(
        "gemm1",
        lambda: w4a8_fp8_wmma.mmq_regdirect_w4a16_moe(
            x16, w13_rep, w13_scales, w13_zeros if w13_zeros is not None else _empty,
            sorted_ids, expert_ids, ntp, 2 * inter, top_k, block_m, wide,
        ),
    )  # (P, 2*inter) fp16
    d = out1.shape[1] // 2
    if _TAIL_HIP and out1.dtype in _SILU_DTYPES:
        import tail_hip

        buf2 = _moe_time("silu", lambda: tail_hip.silu_and_mul(out1.contiguous()))
    else:
        buf2 = _moe_time(
            "silu",
            lambda: (F.silu(out1[:, :d].float()) * out1[:, d:].float()).to(torch.float16).contiguous(),
        )

    tw_flat = tw.reshape(-1).contiguous()
    # gemm2 + topk-weight + reduce via the fused scatter epilogue (eager decode). The atomicAdd is
    # not HIP-graph-safe; a gather twin (mmq_regdirect_w4a16_moe + gather_reduce) is the graph path.
    output = torch.zeros((M, hidden), dtype=torch.float32, device=dev)
    _moe_time(
        "gemm2scat",
        lambda: w4a8_fp8_wmma.mmq_regdirect_w4a16_moe_scatter(
            buf2.contiguous(), w2_rep, w2_scales, sorted_ids, expert_ids, ntp, tw_flat, output,
            hidden, top_k, block_m, wide, w_zeros=w2_zeros,
        ),
    )  # writes output in place
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
    import w4a8_fp8_wmma

    wide = _w4a16_wide(group_size)
    x16 = x if x.dtype == torch.float16 else x.to(torch.float16)
    z = w_zeros if w_zeros is not None else torch.empty(0, dtype=torch.int32, device=x.device)
    return w4a8_fp8_wmma.mmq_regdirect_w4a16_wide(x16.contiguous(), w_rep_wide, scales, z, N, wide)


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
    block_m: int = 16,
) -> torch.Tensor:
    """Grouped W8A8-fp8 MoE forward: topk -> moe_align -> grouped GEMM(w13) -> silu_and_mul
    -> grouped GEMM(w2) -> topk-weighted gather-reduce. A strict simplification of `w4a8_moe`
    (no zeros, no per-K-group scale: fp8 weights carry a per-output-channel f32 scale folded
    once in the epilogue). Returns (M, K)."""
    import torch.nn.functional as F
    import w8a8_fp8_wmma

    M, K = x.shape
    E = w13.shape[0]
    dev = x.device
    # Per-GEMM kernel pick (== w4a8_moe): at decode (M<=2) gemm1's wide 2*inter output over a few
    # real tokens is far faster as a per-token GEMV than WMMA over mostly-padding tiles; gemm2's
    # K output favours WMMA. Prefill (M>2) keeps the passed/default kernel for both.
    gemm1_kernel = "gemv" if M <= 2 else kernel
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

    sorted_ids, expert_ids, ntp = _moe_time("align", lambda: moe_hip.moe_align(topk_ids, E, block_m))
    P = sorted_ids.shape[0]

    x16 = _moe_time("cast", lambda: x.to(torch.float16).contiguous())
    out1 = _moe_time(
        "gemm1",
        lambda: w8a8_fp8_wmma.mmq_w8a8_moe_gemm(
            x16, w13, w13_scales, sorted_ids, expert_ids, ntp, top_k, block_m, gemm1_kernel,
        ),
    )  # (P, 2*inter)
    d = out1.shape[1] // 2
    # Gated SiLU-mul via the dtype-generic native HIP tail_hip.silu_and_mul (one launch, fp32
    # internal, no temps); MINISGL_TAIL_HIP=0 reverts to the torch ref. (The gemm1-epilogue fused
    # silu is wmma-only -> unusable at decode where gemm1 must be gemv.)
    if _TAIL_HIP and out1.dtype in _SILU_DTYPES:
        import tail_hip  # canonical package: silu_and_mul is a module-level callable

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
        _moe_time(
            "gemm2scat",
            lambda: w8a8_fp8_wmma.mmq_w8a8_moe_gemm_scatter(
                buf2, w2, w2_scales, sorted_ids, expert_ids, ntp, tw_flat, acc, top_k, block_m,
                gemm2_kernel,
            ),
        )  # writes acc in place
        _moe_report()
        return acc.to(x.dtype)

    ident = torch.arange(P, dtype=torch.int32, device=dev)
    out2 = _moe_time(
        "gemm2",
        lambda: w8a8_fp8_wmma.mmq_w8a8_moe_gemm(
            buf2, w2, w2_scales, ident, expert_ids, ntp, 1, block_m, gemm2_kernel,
        ),
    )  # (P, K)
    acc = _moe_time(
        "gather",
        lambda: w8a8_fp8_wmma.mmq_w8a8_moe_gather_reduce(
            out2.contiguous(), sorted_ids, tw_flat, ntp, top_k
        ),
    )  # (M, K) fp32
    _moe_report()
    return acc.to(x.dtype)


# --- RXF (Rotated eXtra Fast) W4(NL)-A8(int8) path -------------------------------------------
# Distinct from the fp8 W4A8 above: weights are an NL (non-uniform) int4 codebook, activations are
# int8 (not fp8), and a fixed block-diagonal Hadamard rotation is applied to the activation at
# runtime (and was applied to the weights offline) so it cancels in the dot while spreading
# activation outliers. Native HIP via the vendored rxf_hip package (rxf_hip.*).

_RXF_NL: dict = {}


def _rxf_nl(dev: torch.device) -> torch.Tensor:
    """Cached int8[16] NL codebook on `dev` (rxf_hip.NL_DEFAULT == the Triton _NL_DEFAULT)."""
    key = str(dev)
    t = _RXF_NL.get(key)
    if t is None:
        import rxf_hip

        t = torch.tensor(rxf_hip.NL_DEFAULT, dtype=torch.int8, device=dev)
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
    import rxf_hip  # noqa: F401  registers rxf_hip.*

    q, a_scale = rxf_hip.rotate_quant_int8(x.contiguous(), span)
    return rxf_hip.linear(q, a_scale, w_packed, w_scale, _rxf_nl(x.device), bias)


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
    import rxf_hip  # noqa: F401

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
    sorted_ids, expert_ids, ntp = moe_hip.moe_align(ti, E, block_m)
    P = sorted_ids.shape[0]

    # gemm1: rotate+quant the activation (gathered by sorted_ids inside the GEMM), grouped over w13.
    # Per-GEMM kernel selection (== w4a8_moe): at decode (M<=2) gemm1's wide 2*inter output over a
    # few real tokens is far faster as a per-token GEMV than WMMA over mostly-padding tiles.
    q, a_scale = rxf_hip.rotate_quant_int8(x.contiguous(), span)
    gemm1 = rxf_hip.moe_gemv if M <= 2 else rxf_hip.moe_gemm
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
    q2, a_scale2 = rxf_hip.rotate_quant_int8(buf2.contiguous(), span)
    tw_flat = tw.reshape(-1).float().contiguous()

    # DECODE fast path (mirrors w4a8_moe): fuse gemm2 + topk-weight + reduce into ONE kernel via the
    # atomic scatter, removing the (P,K) out2 materialization + the separate gather. The atomicAdd is
    # NOT HIP-graph-capture-safe -> gated to eager decode (M<=2) and MINISGL_MOE_SCATTER.
    if M <= 2 and _MOE_SCATTER:
        acc = torch.zeros((M, K), dtype=torch.float32, device=dev)
        rxf_hip.moe_gemm_scatter(
            q2, a_scale2, w2, w2_scales, nl, sorted_ids, expert_ids, ntp, tw_flat, acc,
            top_k, block_m, M * top_k
        )  # writes acc in place
        return acc.to(x.dtype)

    # PREFILL: unfused gemm2 (identity gather) + contention-free gather-reduce (graph-safe).
    ident = torch.arange(P, dtype=torch.int32, device=dev)
    out2 = rxf_hip.moe_gemm(
        q2, a_scale2, w2, w2_scales, nl, ident, expert_ids, ntp, 1, block_m, P
    )  # (P, K) bf16
    acc = rxf_hip.moe_gather_reduce(
        out2, sorted_ids, tw_flat, ntp, M, top_k, M * top_k
    )  # (M, K) fp32
    return acc.to(x.dtype)


def _pick_dense_kernel(m: int) -> str:
    """Per-M dense kernel selection. The served WMMA prefill kernel handles all M; the
    scalar-dot GEMV is the decode (M<=2) fast path. (The full vllm_adapter also has env
    tuning + a Triton W4A16 small-M/large-group fallback — PERF_NOTES.)"""
    return "decode_gemv" if m <= 2 else "prefill_wmma"


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
    import w4a8_fp8_wmma

    x2d = x  # native dtype straight into the op (fp16 or bf16); no bf16->fp16 round-trip
    if kernel is None:
        kernel = _pick_dense_kernel(x2d.shape[0])
    return w4a8_fp8_wmma.mmq_fp8_gemm(
        x2d, w_packed, scales, kernel=kernel, w_zeros=w_zeros, weight_is_e2m1=weight_is_e2m1
    )
