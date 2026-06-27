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

# Decode-path gemm2+gather fusion via mmq_fp8_moe_gemm_scatter (atomic scatter). On by default;
# MINISGL_MOE_SCATTER=0 reverts to the unfused gemm2 + gather_reduce (set this if CUDA graphs are
# enabled for decode — the scatter's atomicAdd is not graph-capture-safe).
_MOE_SCATTER = _os.environ.get("MINISGL_MOE_SCATTER", "1") != "0"

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
    gating_output: torch.Tensor,  # (M, E)
    top_k: int,
    renormalize: bool,
    *,
    kernel: str = "wmma",
    block_m: int = 16,
) -> torch.Tensor:
    """Grouped W4A8 MoE forward: topk -> moe_align -> grouped GEMM(w13) -> silu_and_mul
    -> grouped GEMM(w2) -> topk-weighted gather-reduce. Mirrors the proven
    w4a8_fp8_wmma `_run_grouped_moe` (non-GEMV, unfused-silu) path. Returns (M, K).
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

    # Fused softmax+topk(+renormalize) in one kernel (vLLM _moe_C.topk_softmax), replacing
    # torch.softmax+torch.topk+manual-renorm+int32-cast. NB: this image has no sgl_kernel and no
    # _C.silu_and_mul, but vLLM's _moe_C MoE ops ARE present (same source as the moe_align import).
    # ids come out int32 directly; renormalize is folded into the kernel.
    def _route():
        from vllm import _custom_ops as vllm_ops

        tw = torch.empty(M, top_k, dtype=torch.float32, device=dev)
        ti = torch.empty(M, top_k, dtype=torch.int32, device=dev)
        tei = torch.empty(M, top_k, dtype=torch.int32, device=dev)  # token_expert_indices scratch
        vllm_ops.topk_softmax(tw, ti, tei, gating_output.float(), renormalize)
        return tw, ti

    topk_weights, topk_ids = _moe_time("route", _route)

    # moe_align: native HIP (moe_hip) by default, vLLM reference under MINISGL_MOE_ALIGN=0.
    if _MOE_ALIGN_HIP:
        import moe_hip

        sorted_ids, expert_ids, ntp = _moe_time("align", lambda: moe_hip.moe_align(topk_ids, E, block_m))
    else:
        from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size

        sorted_ids, expert_ids, ntp = _moe_time(
            "align", lambda: moe_align_block_size(topk_ids, block_m, E, None, pad_sorted_ids=True)
        )
    P = sorted_ids.shape[0]

    x16 = _moe_time("cast", lambda: x.to(torch.float16).contiguous())
    out1 = _moe_time(
        "gemm1",
        lambda: w4a8_fp8_wmma.mmq_fp8_moe_gemm(
            x16, w13, w13_scales, sorted_ids, expert_ids, ntp, top_k, block_m,
            kernel=gemm1_kernel, w_zeros=w13_zeros,
        ),
    )  # (P, 2*inter)
    d = out1.shape[1] // 2
    # Gated SiLU-mul: the fp16 MoE intermediates (out1 is fp16) now route through the dtype-generic
    # native HIP tail_hip.silu_and_mul (one launch, fp32-internal, no temps) — replacing the multi-op
    # torch chain (silu+mul+float+cast+contiguous). MINISGL_TAIL_HIP=0 reverts to the torch ref.
    # (The gemm1-epilogue fused silu is wmma-only -> unusable at decode where gemm1 must be gemv.)
    if _TAIL_HIP and out1.dtype in _SILU_DTYPES:
        import tail_hip  # noqa: F401  registers torch.ops.tail_hip.*

        buf2 = _moe_time("silu", lambda: torch.ops.tail_hip.silu_and_mul(out1.contiguous()))
    else:
        buf2 = _moe_time(
            "silu",
            lambda: (F.silu(out1[:, :d].float()) * out1[:, d:].float()).to(torch.float16).contiguous(),
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
        _moe_time(
            "gemm2scat",
            lambda: w4a8_fp8_wmma.mmq_fp8_moe_gemm_scatter(
                buf2, w2, w2_scales, sorted_ids, expert_ids, ntp, tw_flat, acc, top_k, block_m,
                kernel=gemm2_kernel, w_zeros=w2_zeros,
            ),
        )  # writes acc in place
        _moe_report()
        return acc.to(x.dtype)

    ident = torch.arange(P, dtype=torch.int32, device=dev)
    out2 = _moe_time(
        "gemm2",
        lambda: w4a8_fp8_wmma.mmq_fp8_moe_gemm(
            buf2, w2, w2_scales, ident, expert_ids, ntp, 1, block_m,
            kernel=gemm2_kernel, w_zeros=w2_zeros,
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


# --- RXF (Rotated eXtra Fast) W4(NL)-A8(int8) path -------------------------------------------
# Distinct from the fp8 W4A8 above: weights are an NL (non-uniform) int4 codebook, activations are
# int8 (not fp8), and a fixed block-diagonal Hadamard rotation is applied to the activation at
# runtime (and was applied to the weights offline) so it cancels in the dot while spreading
# activation outliers. Native HIP via the vendored rxf_hip package (torch.ops.rxf_hip.*).

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
    import rxf_hip  # noqa: F401  registers torch.ops.rxf_hip.*

    q, a_scale = torch.ops.rxf_hip.rotate_quant_int8(x.contiguous(), span)
    return torch.ops.rxf_hip.linear(q, a_scale, w_packed, w_scale, _rxf_nl(x.device), bias)


def rxf_moe(
    x: torch.Tensor,  # (M, K) activations
    w13: torch.Tensor,  # (E, 2*inter, K/2) uint8
    w13_scales: torch.Tensor,  # (E, 2*inter, K/32) fp16
    w2: torch.Tensor,  # (E, K, inter/2) uint8
    w2_scales: torch.Tensor,  # (E, K, inter/32) fp16
    gating_output: torch.Tensor,  # (M, E)
    top_k: int,
    renormalize: bool,
    *,
    span: int = 32,
    block_m: int = 16,
) -> torch.Tensor:
    """Grouped RXF W4A8 MoE: rotate+quant -> grouped GEMM(w13) -> silu_and_mul -> rotate+quant
    -> grouped GEMM(w2) -> topk-weighted gather-reduce. Mirrors w4a8_moe's dispatch (vLLM
    topk_softmax + moe_align), int8/NL on the GEMMs. Returns (M, K)."""
    import torch.nn.functional as F
    import rxf_hip  # noqa: F401
    from vllm import _custom_ops as vllm_ops
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size

    M, K = x.shape
    E = w13.shape[0]
    dev = x.device
    nl = _rxf_nl(dev)

    tw = torch.empty(M, top_k, dtype=torch.float32, device=dev)
    ti = torch.empty(M, top_k, dtype=torch.int32, device=dev)
    tei = torch.empty(M, top_k, dtype=torch.int32, device=dev)
    vllm_ops.topk_softmax(tw, ti, tei, gating_output.float(), renormalize)

    sorted_ids, expert_ids, ntp = moe_align_block_size(ti, block_m, E, None, pad_sorted_ids=True)
    P = sorted_ids.shape[0]

    # gemm1: rotate+quant the activation (gathered by sorted_ids inside the GEMM), grouped over w13.
    # Per-GEMM kernel selection (== w4a8_moe): at decode (M<=2) gemm1's wide 2*inter output over a
    # few real tokens is far faster as a per-token GEMV than WMMA over mostly-padding tiles.
    q, a_scale = torch.ops.rxf_hip.rotate_quant_int8(x.contiguous(), span)
    gemm1 = torch.ops.rxf_hip.moe_gemv if M <= 2 else torch.ops.rxf_hip.moe_gemm
    out1 = gemm1(
        q, a_scale, w13, w13_scales, nl, sorted_ids, expert_ids, ntp, top_k, block_m, M * top_k
    )  # (P, 2*inter) bf16
    d = out1.shape[1] // 2

    from minisgl.layers import _tail_hip

    if _tail_hip.active(out1):
        buf2 = torch.ops.tail_hip.silu_and_mul(out1.contiguous())
    else:
        buf2 = (F.silu(out1[:, :d].float()) * out1[:, d:].float()).to(torch.bfloat16).contiguous()

    # gemm2: rotate+quant the intermediate (w2 was rotated offline too).
    q2, a_scale2 = torch.ops.rxf_hip.rotate_quant_int8(buf2.contiguous(), span)
    tw_flat = tw.reshape(-1).float().contiguous()

    # DECODE fast path (mirrors w4a8_moe): fuse gemm2 + topk-weight + reduce into ONE kernel via the
    # atomic scatter, removing the (P,K) out2 materialization + the separate gather. The atomicAdd is
    # NOT HIP-graph-capture-safe -> gated to eager decode (M<=2) and MINISGL_MOE_SCATTER.
    if M <= 2 and _MOE_SCATTER:
        acc = torch.zeros((M, K), dtype=torch.float32, device=dev)
        torch.ops.rxf_hip.moe_gemm_scatter(
            q2, a_scale2, w2, w2_scales, nl, sorted_ids, expert_ids, ntp, tw_flat, acc,
            top_k, block_m, M * top_k
        )  # writes acc in place
        return acc.to(x.dtype)

    # PREFILL: unfused gemm2 (identity gather) + contention-free gather-reduce (graph-safe).
    ident = torch.arange(P, dtype=torch.int32, device=dev)
    out2 = torch.ops.rxf_hip.moe_gemm(
        q2, a_scale2, w2, w2_scales, nl, ident, expert_ids, ntp, 1, block_m, P
    )  # (P, K) bf16
    acc = torch.ops.rxf_hip.moe_gather_reduce(
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
) -> torch.Tensor:
    """Dense W4A8 GEMM: (M, K) @ (N, K)^T -> (M, N). Returns the op's fp16 output;
    the caller casts back to the activation dtype."""
    import w4a8_fp8_wmma

    x2d = x if x.dtype == torch.float16 else x.to(torch.float16)  # op computes in fp16
    if kernel is None:
        kernel = _pick_dense_kernel(x2d.shape[0])
    return w4a8_fp8_wmma.mmq_fp8_gemm(x2d, w_packed, scales, kernel=kernel, w_zeros=w_zeros)
