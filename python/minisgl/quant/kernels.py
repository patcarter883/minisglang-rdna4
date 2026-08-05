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

# Decode-path gemm2+gather fusion via mmq_fp8_moe_gemm_scatter (atomic scatter). UNCONDITIONAL at
# M<=2 -- there is no flag. M<=2 is a WORKLOAD condition (decode vs prefill), not a toggle.
#
# This was once gated OFF on the claim that "the scatter's atomicAdd is NOT graph-capture-safe".
# THAT CLAIM WAS FALSE and it cost the served path a free win for as long as it stood. An atomicAdd
# is an ordinary instruction; nothing about it resists capture. What a captured region does require
# is that the accumulator be re-zeroed on every REPLAY, and it is: the `torch.zeros((M, K))` below is
# recorded INSIDE the captured region, so its fill kernel is part of the graph and runs on each
# replay. Measured 2026-07-30, Qwen3.6-35B-A3B-AWQ-4bit TP=2 --graph 16, M=1 decode, interleaved
# paired legs in one lease: all 6 decode graphs capture with the scatter engaged, and
#   gather_reduce -> 81.0, 81.1 tok/s    scatter -> 83.7, 83.5 tok/s   (1.0315x, non-overlap)
# Note the size of that win, because the isolated microbench for the same op said 1.75x
# (21.93us scatter vs 38.42us gather_reduce). e2e it is +3.2%. Trust the serve number.
#
# NOT bit-exact vs gather_reduce (the atomic reduction order varies), so it is tolerance-gated.

# The route is produced by moe_hip.moe_route_align (see _route_align below) — softmax + top-k +
# renormalize + moe_align_block_size in ONE op, unconditionally. The old `_get_fused_router()`
# capability probe and the pure-torch / vLLM fallback chain it guarded are DELETED, not left
# dormant: the lean image has no vLLM, tools/_bench_inner.sh already hard-fails a leg whose moe_hip
# does not import, and a silent try/except fallback is exactly how a slower leg becomes an
# accidental "baseline" (COMMANDMENT §1, routing-reality check).

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

# Decode gemm2 split-K, DERIVED FROM THE SHAPE (no flag), the same way _moe_block_m is. The scatter
# GEMM at M==1 is occupancy-starved -- grid is only (N/BN, P/block_m), a handful of blocks -- so the
# K=inter contraction is carved across grid.z and the scatter's atomicAdd (already the reduction)
# combines the slices for free. M>=2 has enough blocks and takes no slicing.
# This used to be MINISGL_MOE_SPLITK pointing at a whole separate moe_splitk_hip package. That
# package hardcoded fp16 activations, so on a bf16 model it raised on the first decode step -- its
# "1.767x" could never run in production. Split-K is now an axis on the shared fp8_wmma core, which
# is activation-dtype generic, so it works for fp16 AND bf16.
_MOE_SPLITK_DECODE = 4          # k-slices at M==1; clamped to K/group_size inside the launcher


def _moe_split_k(M: int) -> int:
    """K-slices for the decode scatter gemm2. Workload-derived: 1 = no slicing."""
    return _MOE_SPLITK_DECODE if M == 1 else 1

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

# NVFP4 (group-16 e2m1) decode-GEMV lever. DEFAULT ON: group-16 rides the unified decode-GEMV core
# (per-16-K-half scale fold). The earlier serve crash was the MoE fused kernel's own guards still
# asserting group_size%32==0 (moe_kernel.hip gemm1_silu/gemm2/scatter) while the shared accum already
# handled group-16 — fixed to allow group_size==16. Validated on Laguna-XS-2.1-NVFP4 (fe2e 2026-07-23):
# no crash, coherent greedy output, 21.71 -> 48.6 tok/s (2.24x, WMMA baseline was a mis-applied prefill
# kernel at M=1). MINISGL_NVFP4_GEMV=0 forces the old WMMA path (A/B + rollback).
_NVFP4_GEMV = _os.environ.get("MINISGL_NVFP4_GEMV", "1") != "0"


# ---- Expert-divergence probe (diagnostic; OFF unless MINISGL_MOE_ROUTE_STATS names a JSON path).
#
# Answers ONE question that the spec-decode cost analysis rests on: when a verify batch of qlen=K+1
# draft rows hits a 256-expert top-8 MoE, do those rows route to DISJOINT expert sets (so the grouped
# GEMM genuinely has to stream ~8*qlen expert slabs and the verify cost is inherent), or do they
# overlap heavily (in which case the cost is a kernel/alignment bug)?
#
# The kernel-side truth is `ntp` (num_tokens_post_padded) from moe_align: the grouped GEMM grinds
# ntp/block_m tiles, one expert slab per tile. So blocks/D tells you whether alignment amortizes the
# overlap that IS there. Recorded per MoE call; rows are dumped when the sample cap is hit.
_ROUTE_STATS_PATH = _os.environ.get("MINISGL_MOE_ROUTE_STATS", "")
_ROUTE_STATS_MAX = int(_os.environ.get("MINISGL_MOE_ROUTE_STATS_N") or 4000)
_ROUTE_STATS_MAXTOK = int(_os.environ.get("MINISGL_MOE_ROUTE_STATS_MAXTOK") or 64)
_route_stats_rows: list = []
_route_stats_done = False


def _route_stats(topk_ids: torch.Tensor, num_experts: int, block_m: int, ntp: torch.Tensor) -> None:
    """Record one MoE call's routing shape. Syncs (unique + .tolist()) — probe only."""
    global _route_stats_done
    if _route_stats_done:
        return
    if torch.cuda.is_current_stream_capturing():  # .item() is illegal mid-capture
        return
    M = topk_ids.shape[0]
    if M > _ROUTE_STATS_MAXTOK:  # prefill batch, not a decode/verify step
        return
    ids = topk_ids.tolist()  # [M, top_k]
    per_expert: dict = {}
    union_curve = []
    seen: set = set()
    for row in ids:
        seen.update(row)
        union_curve.append(len(seen))
        for e in row:
            per_expert[e] = per_expert.get(e, 0) + 1
    D = len(seen)
    pairs = M * len(ids[0]) if ids else 0
    ideal_blocks = sum(-(-c // block_m) for c in per_expert.values())
    _route_stats_rows.append(
        {
            "M": M,
            "top_k": len(ids[0]) if ids else 0,
            "E": num_experts,
            "block_m": block_m,
            "pairs": pairs,
            "distinct": D,
            "ntp": int(ntp.item()),
            "blocks": int(ntp.item()) // block_m,
            "ideal_blocks": ideal_blocks,
            "max_rows_per_expert": max(per_expert.values()) if per_expert else 0,
            "union_curve": union_curve,
        }
    )
    if len(_route_stats_rows) >= _ROUTE_STATS_MAX:
        import json

        _route_stats_done = True
        try:
            import torch.distributed as dist

            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
            path = f"{_ROUTE_STATS_PATH}.rank{rank}.json"
            with open(path, "w") as f:
                json.dump(_route_stats_rows, f)
            print(f"[route-stats] wrote {len(_route_stats_rows)} rows -> {path}", flush=True)
        except Exception as exc:  # a probe must never take the serve down
            print(f"[route-stats] dump failed: {exc}", flush=True)


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


def _route_align(
    gating_output: torch.Tensor, top_k: int, renormalize: bool, num_experts: int, block_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """THE WHOLE ROUTE IN ONE OP: softmax + top-k + renormalize + moe_align_block_size.

    Returns (topk_weights f32, topk_ids i32, sorted_ids, expert_ids, num_tokens_post_pad) — every
    tensor the grouped-MoE GEMM needs to start.

    WHY THIS EXISTS. Producing the route used to cost THREE dispatches per MoE layer:
    `gating_output.float()`, `moe_hip.moe_topk_softmax`, `moe_hip.moe_align`. At the served decode
    point (Qwen3.6-35B-A3B: M=1, E=256, top_k=8, 40 MoE layers, TP=2) every one of them is
    dominated by its own launch — measured amortized on GPU0: the router is 6.311 us of which the
    work-free floor is 3.955 us, and the smallest possible torch dispatch (a 1-element `add_`)
    under the same harness is 3.098 us. You cannot tile or de-barrier a dispatch floor away. The
    kernel package now exposes the whole pipeline as ONE op, and the launcher falls back to the
    two-kernel form IN CODE at large M (a single block cannot parallelise a 2048-token prefill
    route). There is no flag and no opt-in: the caller always calls this.

    The `.float()` is deleted rather than moved: the router core reads bf16/fp16 natively and the
    widening is lossless, so it is bit-identical (gated in the package's parity_route_align.py)."""
    import moe_hip

    engaged("moe_hip.moe_route_align")
    g = gating_output if gating_output.is_contiguous() else gating_output.contiguous()
    return moe_hip.moe_route_align(g, top_k, renormalize, num_experts, block_size)


def moe_route_sigmoid_bias(
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    top_k: int,
    renormalize: bool,
    routed_scaling_factor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The sigmoid + correction-bias route in ONE launch (Laguna, GLM-4.7-Flash n_group=1).

    Replaces a TWELVE-kernel torch chain per sparse layer over a single [T, E] row — `logits.float()`,
    `bias.float()`, `sigmoid`, `add`, `warpMergeSortTopK`, `bitonicSortKVInPlace`, `gather`, `sum`,
    `+1e-20`, `div`, `*sf`, `.int().contiguous()`. Laguna-XS-2.1 pays that 39x per step (468
    dispatches, 21.9% of ALL decode dispatches, ~1.9 us each of pure launch latency); GLM-4.7-Flash
    pays it 46x (552). Same kernel body as `moe_topk_softmax` under a different SCORING POLICY
    (KERNEL_CORE_POLICY) — not a second router, and not a per-model kernel: E comes from the tensor
    and top_k / renormalize / routed_scaling_factor are runtime arguments.

    SCOPE IS THE ROUTE ONLY. `moe_align` deliberately stays a separate call: folding it in needs
    `w4a8_moe`'s signature re-plumbed and would re-open the fused kernel's `extern __shared__`
    carve-up. So this is ~1020 route dispatches/step -> ~85, with ~85 aligns still standing.

    Pass the gate logits and the bias at their NATIVE dtype. Do NOT .float() either: the core widens
    bf16/fp16 in-register (lossless), and the engine's bias buffer is ALREADY at the model dtype
    (bf16 for Laguna-XS-2.1-NVFP4, fp16 for QuantTrio/GLM-4.7-Flash-AWQ), so widening it here would
    be a no-op that costs a dispatch. See the bias-dtype note at the call sites.
    """
    import moe_hip

    engaged("moe_hip.moe_topk_sigmoid_bias")
    g = gating_output if gating_output.is_contiguous() else gating_output.contiguous()
    b = correction_bias if correction_bias.is_contiguous() else correction_bias.contiguous()
    return moe_hip.moe_topk_sigmoid_bias(g, b, top_k, renormalize, routed_scaling_factor)


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
    activation: str = "silu",  # gated activation on the gemm1 [gate|up] output: "silu" | "gelu"
) -> torch.Tensor:
    """Grouped W4A8 MoE forward: topk -> moe_align -> grouped GEMM(w13) -> gated activation
    -> grouped GEMM(w2) -> topk-weighted gather-reduce. Mirrors the proven
    w4a8_fp8_wmma `_run_grouped_moe` (non-GEMV, unfused-silu) path. Returns (M, K).
    `weight_is_e2m1=True` selects the kernel's MXFP4 (E2M1) weight decode instead of uniform int4
    (the scales are the E8M0 group exponents folded to fp16; w13_zeros/w2_zeros MUST be None).
    `activation` picks the gated activation: "silu" (default, and the only one with a fused gemm1
    epilogue) or "gelu" == HF `gelu_pytorch_tanh` (Gemma4's routed experts), which forces the
    unfused gemm1 path below.
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
    # NVFP4 experts are group-16 and now ride the decode GEMV: the unified Int4Fp8GemvLoader folds a
    # per-16-K-half scale (group<32 branch), so the MoE gemm1+silu / gemm / scatter / gather-reduce GEMV
    # kernels serve group_size 16 exactly like group-32 MXFP4/int4 — one shared core, no fork. Runtime
    # group_size = K / n_groups, K = w13.shape[-1]*8 (int32 packs 8 e2m1 nibbles); %16 covers 16/32/128.
    _grp = (w13.shape[-1] * 8) // w13_scales.shape[-1]
    _gemv_ok = (_grp % 32 == 0) or (_grp % 16 == 0 and _NVFP4_GEMV)
    gemm1_kernel = "gemv" if (M <= _MOE_GEMM1_GEMV_MAX and _gemv_ok) else kernel
    gemm2_kernel = kernel

    # Precomputed route (e.g. GLM/DeepSeek noaux_tc: sigmoid + correction bias + group top-k +
    # normalize + scale, done in the model). Otherwise route AND align in ONE op — see _route_align:
    # the cast+router+sort trio was three dispatches per MoE layer and each one is ~3.1 us of pure
    # launch at the served M=1.
    import moe_hip

    if topk_ids is None:
        topk_weights, topk_ids, sorted_ids, expert_ids, ntp = _moe_time(
            "route_align",
            lambda: _route_align(gating_output, top_k, renormalize, E, block_m),
        )
    else:
        assert topk_weights is not None, "topk_weights required when topk_ids is given"
        topk_weights = topk_weights.to(torch.float32).contiguous()
        topk_ids = topk_ids.to(torch.int32).contiguous()
        # moe_align: native HIP (moe_hip). The former MINISGL_MOE_ALIGN=0 vLLM reference is gone —
        # the lean image has no vllm, and moe_hip.moe_align is the validated drop-in for that op.
        engaged("moe_hip.moe_align")
        sorted_ids, expert_ids, ntp = _moe_time(
            "align", lambda: moe_hip.moe_align(topk_ids, E, block_m)
        )
    P = sorted_ids.shape[0]
    if _ROUTE_STATS_PATH:
        _route_stats(topk_ids, E, block_m, ntp)

    _e2m1 = "+e2m1" if weight_is_e2m1 else ""
    # The W4A8 kernel is now activation-dtype-generic (fp16 OR bf16), so pass activations in their
    # NATIVE dtype — a bf16 model no longer round-trips bf16->fp16->bf16 here (out1 follows x's dtype).
    x16 = _moe_time("cast", lambda: x.contiguous())
    # Gated gemm1 + activation. FUSED path (default): one kernel writes silu(gate)*up -> (P, inter),
    # dropping the separate silu launch and the (P, 2*inter) out1 round-trip. The gemm1-epilogue fusion
    # is now available at DECODE too via the moe_gemv_decode_silu kernel (kernel="gemv"), not just the
    # WMMA prefill path. UNFUSED fallback (MINISGL_MOE_FUSED_SILU=0, fp32, or activation != silu) =
    # gemm1 -> (P,2*inter) then tail_hip.silu_and_mul (fp32-internal HIP) / the torch reference.
    #
    # `activation` is a POLICY over this one shared body, never a kernel fork (KERNEL_CORE_POLICY.md).
    # The `activation == "silu"` guard on the fused branch is load-bearing: the fused epilogue is
    # HARD-WIRED to silu, so letting a gelu model reach it would compute the wrong activation and
    # still return correctly-shaped, finite, plausible logits — a silent quality regression with no
    # crash to catch it. Cost of the guard is that gelu pays the extra (P, 2*inter) round-trip and a
    # second launch.
    # FOLLOW-UP (KERNEL_CORE_POLICY.md: a new activation is a policy on the existing core, NOT a new
    # kernel and NOT a permanent slow path): template `mmq_fp8_moe_gemm1_silu` / `_silu_flag` on the
    # epilogue functor so the activation becomes a kernel argument and gelu runs at fused speed. Until
    # that lands, gelu is correct-but-slower here by construction.
    assert activation in ("silu", "gelu"), (
        f"w4a8_moe supports activation 'silu' or 'gelu' (gelu_pytorch_tanh); got {activation!r}"
    )
    if activation == "silu" and _MOE_FUSED_SILU and x16.dtype in _FUSED_SILU_DTYPES:
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
        if activation == "gelu":
            # `gelu_tanh_and_mul`, NOT `gelu_and_mul` — the latter is the exact erf gelu (both its
            # native tail_hip arm and its torch arm), while every gelu-MoE checkpoint we serve
            # declares HF `gelu_pytorch_tanh`. Torch-only: there is no native tanh-gelu tail kernel.
            from minisgl.layers.activation import gelu_tanh_and_mul

            buf2 = _moe_time("gelu", lambda: gelu_tanh_and_mul(out1.contiguous()).contiguous())
        elif _TAIL_HIP and out1.dtype in _SILU_DTYPES:
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
    # gather_reduce launch. It IS graph-capturable (the old "atomicAdd cannot be
    # captured" claim was false) and is ON by default -- the `torch.zeros` below is captured too, so
    # the accumulator is re-zeroed on every replay. Prefill (M>2) keeps the unfused gemm2 +
    # contention-free gather_reduce (the (P,K) out2 reuse amortizes better at larger M).
    # NOT bit-exact vs gather_reduce: the atomic reduction order varies, so this is a tolerance-gated
    # path, never a bit-exact one.
    if M <= 2:
        acc = torch.zeros((M, K), dtype=torch.float32, device=dev)
        # split_k is an AXIS on the shared scatter core (workload-derived), not a second kernel and
        # not a second package: same op, same weights, same epilogue, one extra grid dimension.
        split_k = _moe_split_k(M)
        engaged(f"fp8_wmma.mmq_fp8_moe_gemm_scatter{_e2m1}")
        _moe_time(
            "gemm2scat",
            lambda: fp8_wmma.mmq_fp8_moe_gemm_scatter(
                buf2, w2, w2_scales, sorted_ids, expert_ids, ntp, tw_flat, acc, top_k, block_m,
                kernel=gemm2_kernel, w_zeros=w2_zeros, weight_is_e2m1=weight_is_e2m1,
                split_k=split_k,
            ),
        )  # writes acc in place (atomic scatter over experts AND the split_k K-slices)
        _moe_report()
        return acc.to(x.dtype)

    # DECODE (small M): fuse gemm2 (down-proj) + gather-reduce into ONE launch. Reads the pre-sorted buf2
    # directly + does the per-token top_k reduce in-kernel, dropping the (P,N) out2 HBM round-trip + the
    # gather_reduce launch AND skipping the alignment-padding rows the WMMA gemm2 computes. gemv-math
    # down-proj -> ~1e-4 vs the WMMA path (accumulation order; user-accepted). Prefill (M>threshold) keeps
    # the flag/WMMA gemm2 + gather_reduce below.
    if _MOE_G2FUSE and M <= _MOE_GEMM1_GEMV_MAX and block_m != 128 and _gemv_ok \
            and hasattr(fp8_wmma, "mmq_fp8_moe_gemm2_gather_reduce"):
        # _gemv_ok: the decode gemm2 gather-reduce is gemv-math (now group_size%16, incl. NVFP4 group-16
        # via the unified loader's per-16-K-half scale fold).
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
    sorted_ids = None
    if topk_ids is None:
        assert router_logits is not None and top_k > 0, (
            "w4a16_moe needs a precomputed route (topk_ids/topk_weights) or router_logits + top_k"
        )
        # ONE op for cast+route+sort (see _route_align).
        topk_weights, topk_ids, sorted_ids, expert_ids, ntp = _moe_time(
            "route_align",
            lambda: _route_align(router_logits, top_k, renormalize, E, block_m),
        )
    top_k = topk_ids.shape[1]
    tw = topk_weights.to(torch.float32).contiguous()
    ti = topk_ids.to(torch.int32).contiguous()
    _empty = torch.empty(0, dtype=torch.int32, device=dev)

    _e2m1 = "+e2m1" if weight_is_e2m1 else ""
    if sorted_ids is None:
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
    # DECODE fast path: fused gemm2 + topk-weight + atomic scatter (graph-capturable, ON by default --
    # gated to M<=2 by workload, not by a flag). Otherwise the unfused gemm2 + gather_reduce.
    if M <= 2:
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

    # Precomputed route (ZAYA top-1 + MOD), else route AND align in ONE op (see _route_align).
    # This site used to skip the native HIP router entirely and fall through to vLLM/torch — dead
    # code in the lean image, and 4+ launches where there is now one.
    import moe_hip

    if topk_ids is None:
        topk_weights, topk_ids, sorted_ids, expert_ids, ntp = _moe_time(
            "route_align",
            lambda: _route_align(gating_output, top_k, renormalize, E, block_m),
        )
    else:
        assert topk_weights is not None, "topk_weights required when topk_ids is given"
        topk_weights = topk_weights.to(torch.float32).contiguous()
        topk_ids = topk_ids.to(torch.int32).contiguous()
        # moe_align: native HIP (moe_hip). The former MINISGL_MOE_ALIGN=0 vLLM reference is gone —
        # the lean image has no vllm, and moe_hip.moe_align is the validated drop-in for that op.
        engaged("moe_hip.moe_align")
        sorted_ids, expert_ids, ntp = _moe_time(
            "align", lambda: moe_hip.moe_align(topk_ids, E, block_m)
        )
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
    # DECODE fast path: fuse gemm2 + topk-weight + reduce into ONE atomic-scatter kernel
    # (graph-capturable, unconditional at M<=2). Prefill (M>2) keeps the
    # unfused gemm2 + contention-free gather_reduce.
    if M <= 2:
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
    # DECODE fast path: fused gemm2 + topk-weight + atomic scatter (graph-capturable, ON by default --
    # gated to M<=2 by workload, not by a flag). Otherwise the unfused gemm2 + gather_reduce.
    if M <= 2:
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
    # atomic scatter, removing the (P,K) out2 materialization + the separate gather. Graph-capturable,
    # gated to M<=2 by workload, not by a flag.
    if M <= 2:
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


def _tiled_last_wave_occupancy(n: int) -> float:
    """How full is `wmma_tiled_tuned`'s LAST dispatch wave at output width `n`?

    In the mid-band (M < its BM=256) the tiled kernel's grid is exactly ceil(N / BN) workgroups —
    one row-block, one column-block per BN=128 slice of N — so the whole launch is `tiles` blocks
    over the device's CUs. When `tiles` is a multiple of the CU count every wave is full; when it is
    not, the last wave runs at `tiles / (waves * CU)` occupancy and the kernel pays a full wave for a
    fraction of a wave's work. That is the term the mid-band surface is non-monotonic in, and it is a
    property of the DEVICE as much as of the shape — see `_pick_dense_kernel`.

    Returns 1.0 when the CU count cannot be read (no HIP device: CPU tests, meta init), which makes
    the caller prefer `wmma_tiled_tuned` — the right default, since it is the mid-band winner
    everywhere the wave is full.
    """
    cu = _device_cu_count()
    if cu <= 0:
        return 1.0
    tiles = -(-n // _W4A8_TILED_BN)
    return tiles / (cu * -(-tiles // cu))


_CU_COUNT: dict[int, int] = {}


def _device_cu_count() -> int:
    """CUs on the current device. torch reports RDNA's WORKGROUP PROCESSORS in
    `multi_processor_count` (32 on the RX 9070 XT, 28 on the RX 9070) and a WGP is 2 CUs, so the
    dispatch unit count is twice it — 64 and 56 respectively. Measured: the mid-band boundary
    tracks 2x this number on BOTH cards and tracks neither `multi_processor_count` alone."""
    try:
        idx = torch.cuda.current_device()
    except Exception:  # noqa: BLE001  — no HIP device at all
        return 0
    cu = _CU_COUNT.get(idx)
    if cu is None:
        try:
            cu = torch.cuda.get_device_properties(idx).multi_processor_count * 2
        except Exception:  # noqa: BLE001
            cu = 0
        _CU_COUNT[idx] = cu
    return cu


def _pick_dense_kernel(
    m: int,
    weight_is_e2m1: bool = False,
    group_size: int = 128,
    k: int | None = None,
    n: int | None = None,
) -> str:
    """Per-M dense-linear kernel selection, at the MEASURED crossovers (gfx1201).

    - m <= gemv_max -> decode_gemv: a streaming GEMV that reads each weight once and dots it against
      all M rows (amortizing the weight read), writing straight to `out`. Serves the decode/decode-
      batch band; measured crossover int4 M<=8, e2m1 M<=16 (also the only e2m1-capable small-M kernel;
      decode_gemv's in-kernel cap is 16).
    - m >= _W4A8_PREFILL_TILED_MIN -> wmma_tiled_tuned: the tuned tiled WMMA GEMM. Dominates the prefill
      regime ~2-4x over prefill_wmma (re-bench), for BOTH int4 AND e2m1 now that the tiled kernel is
      e2m1-bit-exact and carries the packed-store. Bit-exact + graph-capture-safe (writes to `out`,
      no in-op at::zeros). One dtype-generic rule — no int4-vs-e2m1 branch.
    - mid-band (gemv_max < m < _W4A8_PREFILL_TILED_MIN) -> a THREE-WAY surface: wmma_tiled_tuned for
      the large majority, prefill_wmma in a wide-N corner, prefill_wmma_ashuffle in a tall-K box.
      The old rule sent the WHOLE mid-band to prefill_wmma on the claim that "its conservative config
      still wins the small-M/wide-N corner (tiled loses only at N>=6144, M<=32)". Measured, the claim
      is right that a corner exists and wrong about where it is, and it has no N term at all — so a
      wide-N conclusion was applied at every width, and the rule was BACKWARDS on essentially every
      shape this engine dispatches.

    THE MID-BAND SURFACE (measured 2026-08-05, tools/w4a8_dense_midband_surface.py, CUDA-graph-replay
    timed with the weights rotated past the 64 MB MALL by BYTE count; 24 shipped shapes x 24 synthetic
    (M,N) grid cells x 13 M values x 3 arms, RX 9070 XT; fixture _midband_surface.txt). Of the 432
    mid-band cells wmma_tiled_tuned wins 306, prefill_wmma_ashuffle 67, prefill_wmma 59.

        rule                                    total us over the per-cell oracle   worst single cell
        always prefill_wmma  (THE OLD RULE)                 +40.6%                  3.09x  N=1024 M=63
        always wmma_tiled_tuned (retire both)               + 4.8%                  1.57x  N=10240 M=17
        + wide-N corner                                     + 1.36%                 1.17x  N=9216  M=56
        + ashuffle box            (THE NEW RULE)            + 0.55%                 1.17x  N=9216  M=56

      So the old rule was not a small mis-tune: on a narrow-N layer at M=63 it ran the mid-band 3x
      slower than the arm sitting next to it. Both prefill arms nevertheless HAVE regimes — retiring
      them costs 1.57x on GLM-4.7-Flash's gate_up (N=10240, M=17: 98.4 us vs 154.3 us) and 1.44x on
      Qwen3.5-4B's (N=9216) — so this is `keep, bounded` for both, not `retire`.
      CAUTION on ashuffle: it wins more CELLS than prefill_wmma but its median cell is worth only
      1.038x, which is near the timing floor. Only the part of its band that survived a 3-pass
      repeat on BOTH cards is routed; see _W4A8_ASHUFFLE_MIN_K.

    WHY NO SINGLE CONSTANT EXPRESSES IT. The surface is NOT monotonic in N. wmma_tiled_tuned wins at
    N=8192 and N=16384 and loses at 9216 / 10240 / 11264 / 17408 / 34816, which is not noise: in the
    mid-band its grid is exactly ceil(N/128) workgroups, and 8192 and 16384 are precisely the widths
    at which that count (64, 128) is a whole number of 64-CU dispatch waves. Everywhere else the last
    wave runs half empty and prefill_wmma — which tiles N by 64, i.e. quantizes at twice the
    resolution, and takes a BM=64 small-M tile instead of padding M up to 256 — wins.
    The deciding term is therefore the DEVICE's CU count, not a shape constant, and this box proves
    it: on the RX 9070 (56 CUs, GPU 1) prefill_wmma wins at N=8192 AND N=16384, exactly where it
    LOSES on the RX 9070 XT (64 CUs, GPU 0). Same shape, same M, opposite arm — so the two ranks of
    one TP=2 job genuinely want different kernels, and any fixed N threshold is wrong on one of them.
    Measured on both cards, 14/14 cells agree with the CU-derived rule and 0/14 with a fixed N
    (tools/w4a8_dense_midband_crosscard.py; fixtures _midband_crosscard_{0,1}.txt).

    THE STRUCTURAL FIX IS IN THE KERNEL, NOT HERE. This whole corner exists because
    wmma_tiled_tuned's tile is HARD-WIRED to BM=256 x BN=128 while prefill_wmma already derives a
    small-M tile (BM=64) from M. The tiled kernel accepts its tile at runtime (VLLM_W4A8_V7_CFG), and
    swept over that knob it beats BOTH shipped arms across nearly the whole mid-band
    (tools/w4a8_dense_midband_tilecfg.py, fixture _midband_tilecfg.txt; bit-identical at every tile):

        N=9216  M=17 :  prefill 160.2   tiled@256x128 235.6 (shipped)   tiled@64x64 114.5   -> 2.06x
        N=10240 M=17 :  prefill  98.9   tiled@256x128 155.9 (shipped)   tiled@64x64  91.4   -> 1.71x
        N=8192  M=17 :  prefill 145.4   tiled@256x128 127.8 (shipped)   tiled@64x64  98.2   -> 1.30x
        N=2048  M=32 :  prefill 140.0   tiled@256x128  97.0 (shipped)   tiled@256x64 84.3   -> 1.15x

      i.e. the tile is worth up to 2.06x and the ARM is worth at most ~1.6x — the arm choice this
      function makes is the smaller half of the win, and it only exists because the bigger half was
      left on the floor. The right fix is a shape-derived tile inside `mmq_fp8_gemm`'s tiled
      launcher (it already does exactly this for prefill_wmma's SBM=64 path), after which the
      mid-band arms plausibly have no regime at all and BOTH prefill arms can be retired. Until then
      this dispatch is the best available from the three arms as shipped.

    The dead small-M WMMA variants (nsplit/splitk/regdirect_shuffle) were REMOVED (they allocated an
    in-op at::zeros((M,N),f32) that blew up VRAM under CUDA-graph capture); no override knob remains.

    DO THE ARMS AGREE? (measured 2026-08-05, tools/quant_m_invariance.py, RX 9070 XT, 7 shipped
    shapes: Gemma4 o_proj 2048/4096/8192 x 2816, qkv, gate_up, dense_down at g=32 fp16; Qwen3.6-35B
    o_proj/gate_up at g=128 bf16.)  Nobody had ever asked, and the answer decides whether a quantized
    model is M-invariant at all -- the property layers/minv.py exists to guarantee for the bf16 path,
    on which chunked prefill, prefix/radix caching and spec-decode VERIFY all depend.

        prefill_wmma  vs  wmma_tiled_tuned :  max|delta| = 0.000e+00  at EVERY M, EVERY shape.
                                              (re-confirmed on the 2026-08-05 mid-band sweep: 0.000e+00
                                              at every one of the 48 shapes, and also for the third,
                                              UNDISPATCHED arm prefill_wmma_ashuffle and for every
                                              VLLM_W4A8_V7_CFG tile of wmma_tiled_tuned. So the whole
                                              mid-band question is pure performance.)
        decode_gemv   vs  either WMMA arm  :  up to 1.953e-3 abs (fp16 g32) / 2.441e-4 (bf16 g128),
                                              ~1e-3 relative.
        each arm against ITSELF across M   :  0.000e+00 (rows[0:m] alone == the same rows inside a
                                              batch of 2048, for all three arms).

    So: every arm is individually M-invariant, and the M >= 64 crossover is FREE because the two WMMA
    arms are the same numbers. The quantized dense path has exactly ONE lossy seam -- decode_gemv <->
    WMMA at `gemv_max` -- and it sits inside the spec-verify band (verify M = bs*(K+1)). That is why
    _W4A8_GEMV_MAX_INT4 is 16 and not 8; see its comment.
    """
    # NVFP4 (group-16) e2m1 now rides decode_gemv in the decode band: the unified Int4Fp8GemvLoader
    # folds a per-16-K-half scale (group<32 branch), so the streaming GEMV serves group-16 at M<=gemv_max
    # just like group-32 MXFP4/int4. Above the band, prefill_wmma/ashuffle STILL hard-require
    # group_size%32, so group-16 must use wmma_tiled_tuned (the only runtime-group_size WMMA, BKT=0).
    # A/B + rollback: force group-16 e2m1 back onto WMMA when the lever is off.
    if weight_is_e2m1 and group_size % 32 != 0 and not _NVFP4_GEMV:
        return "wmma_tiled_tuned"
    gemv_max = _W4A8_GEMV_MAX_E2M1 if weight_is_e2m1 else _W4A8_GEMV_MAX_INT4
    # decode_gemv's K granularity. Its inner warp-step consumes K in 32-k units (4x int32 b128 per
    # lane), so a tail chunk is covered exactly when the lanes divide it; the b128 weight load
    # additionally needs the (N, K/8) row int32-offset 16-byte aligned, i.e. K % 32 == 0.
    #
    # This guard is a CORRECTNESS backstop, not a performance policy. Falling through to the WMMA
    # GEMM would put the served M=1 band on a prefill kernel — the decode-through-a-GEMM
    # anti-pattern this repo has already measured as an occupancy collapse at bs=1 — so any shape
    # that lands in the decode band and gets refused here is a KERNEL bug to fix in the tail, not a
    # shape to route around. (`k=None`: caller did not pass K; keep the historical behaviour.)
    if m <= gemv_max and (k is None or k % _W4A8_GEMV_K_MULTIPLE == 0):
        return "decode_gemv"
    # BOTH prefill arms hard-require group_size % 32 == 0 in-kernel; only wmma_tiled_tuned carries a
    # runtime group size (its BKT=0 instantiation). This used to be checked for e2m1 alone, which
    # left an int4 group-16 checkpoint — they exist, CohereLabs North-Mini-Code w4a16 ships g=16 —
    # able to reach prefill_wmma in the mid-band. It is a property of the ARMS, not of e2m1.
    if group_size % 32 != 0:
        return "wmma_tiled_tuned"
    if m >= _W4A8_PREFILL_TILED_MIN:
        return "wmma_tiled_tuned"
    # THE MID-BAND — a three-way surface, not a threshold. Over the 432 measured mid-band cells
    # wmma_tiled_tuned takes 306; the other 126 split between TWO DISJOINT regions belonging to two
    # different arms, neither of which a single constant can name. `n=None` (caller did not pass N)
    # keeps wmma_tiled_tuned, which is both the safe arm and the majority one.
    if n is None:
        return "wmma_tiled_tuned"
    # (a) THE WIDE-N CORNER — tiled's coarse N tiling leaves a ragged dispatch wave. Device-dependent.
    if (
        n >= _W4A8_PREFILL_WIDE_N
        and m <= _W4A8_PREFILL_WIDE_MAX_M
        and _tiled_last_wave_occupancy(n) < _W4A8_TILED_WAVE_FULL
    ):
        return "prefill_wmma"
    # (b) THE TALL-K / MID-N BOX — prefill_wmma_ashuffle, a third arm that shipped in the package but
    #     appeared in no dispatch, wins here at EVERY mid-band M on BOTH cards.
    if (
        k is not None
        and k >= _W4A8_ASHUFFLE_MIN_K
        and _W4A8_ASHUFFLE_MIN_N <= n <= _W4A8_ASHUFFLE_MAX_N
    ):
        return "prefill_wmma_ashuffle"
    return "wmma_tiled_tuned"


# gemv<->wmma crossover per decode path. decode_gemv asserts M<=16 in-kernel, so 16 is the ceiling for
# BOTH weight formats — and both now sit AT it. This used to be 8 for int4, on the claim that "its WMMA
# tile reclaims M=16 on a dense model". That claim is FALSE at every shape this engine dispatches, and
# it cost twice: it left performance on the floor AND it put a numerics seam in the middle of the
# spec-verify band.
#
# THE NUMERICS HALF. `_pick_dense_kernel`'s docstring records the measurement: the two WMMA arms are
# bit-identical to each other, and each arm is M-invariant on its own, so decode_gemv <-> WMMA is the
# only crossover at which a token's value depends on how many tokens shared its batch. Spec VERIFY runs
# M = bs*(K+1) while plain decode runs M=1, so with the cap at 8 an MTP K=4 verify at bs>=2 (M=10) or a
# DFlash K=15 verify at bs=1 (M=16) ran a DIFFERENT KERNEL from the decode it is compared against. At 16
# the whole band the kernel can serve is one arm — exactly the reasoning layers/minv.py already applies
# to the bf16 path (`_DECODE_GEMV_MAXM = 16`, "so that ordinary decode and spec-decode VERIFY land on
# the SAME kernel rather than opposite sides of the threshold"). Honest limit: this MOVES the seam to
# M=16, it does not remove it — verify wider than 16 rows still crosses, and the kernel cannot go higher.
#
# THE PERF HALF (graph-replay timed, weights rotated past the 64 MB MALL; tools/w4a8_dense_arm_cost.py,
# 2026-08-05, RX 9070 XT). decode_gemv is FASTER than both WMMA arms at every M in 1..16, on every
# shape — the M=9..16 band was being handed to `prefill_wmma`, the slowest of the three:
#     us at M=16          decode_gemv   prefill_wmma (dispatched)   wmma_tiled   -> speedup
#     g4.o_proj(local)        40.6            135.6                    77.4         3.34x
#     g4.o_proj(global)       65.2            188.4                   139.0         2.89x
#     g4.qkv_q                38.5            122.8                    96.9         3.19x
#     g4.gate_up              44.5            124.6                    97.0         2.80x
#     g4.dense_down           35.3            112.2                    41.7         3.18x
#     q35.o_proj              50.6             65.1                    66.2         1.29x
#     q35.gate_up             66.0            115.3                   115.6         1.75x
# so the fix is not a correctness tax, it is a strict win in the band it touches.
_W4A8_GEMV_MAX_INT4 = 16
_W4A8_GEMV_MAX_E2M1 = 16
# K granularity the decode GEMV can consume. Tracks the kernel's own precondition — raise/lower this
# ONLY together with the kernel, never to route a shape away from the decode path (see
# _pick_dense_kernel: a refused decode-band shape means the kernel tail needs fixing).
# 32 is what the b128 weight load actually requires: the 4-word read must stay in-row and 16-byte
# aligned, i.e. (K/8) % 4 == 0. The kernel's K-on-lanes sweep already drops out-of-row lanes to a
# zero contribution, so a partial final wave was always handled — shipped shapes like K=8704 (8 full
# waves + 16 lanes) exercise it. The previous 512 was a stale inheritance from the LDS K-tiling that
# consolidation retired, and it cost Gemma4 (K=2816) the decode GEMV entirely.
_W4A8_GEMV_K_MULTIPLE = 32
# Prefill regime: wmma_tiled_tuned dominates from here up (~2-4x prefill_wmma, bit-exact, graph-safe,
# both dtypes) at EVERY N measured, on BOTH cards. True prefill/chunked-prefill M is always >> 64.
_W4A8_PREFILL_TILED_MIN = 64

# ---- the mid-band wide-N corner (all three terms measured; see _pick_dense_kernel's docstring) ----
# These are NOT three independent knobs to tune: they are one measured region, and the region is
# non-convex in N, so it takes a shape term, an M term AND a device term to fence. Moving any of them
# without re-running tools/w4a8_dense_midband_surface.py re-opens the mistake this replaced — a rule
# whose comment asserted a corner ("tiled loses only at N>=6144, M<=32") that had never been measured
# and that, applied to the whole mid-band, cost up to 3.09x on a single shape.
#
# N floor. Below this width wmma_tiled_tuned wins the mid-band on every shape and both cards,
# regardless of how ragged its last wave is — the launch is under one wave either way, so the tail
# never costs a full wave. Measured: at N=6144 tiled wins 4/4 shapes on GPU 0 and 1/1 on GPU 1.
_W4A8_PREFILL_WIDE_N = 8192
# M ceiling. prefill_wmma's edge is its BM=64 small-M tile; it decays as M fills wmma_tiled_tuned's
# BM=256 tile and is gone by M~56. The per-shape crossover ranges 32..63 (N=9216 and N=10240 keep
# prefill_wmma ahead through 63; N=11264/17408/34816 give it up at 48), so this is a compromise, not
# a cliff. 48 is the optimum on BOTH scores over the measured surface -- 0.45% off the per-cell
# oracle and a 1.17x worst single cell, against 0.50%/1.30x at 40 and 0.70%/1.13x at 56.
_W4A8_PREFILL_WIDE_MAX_M = 48
# wmma_tiled_tuned's N tile, i.e. the BN in its `dim3 grid(ceil(M/BM), ceil(N/BN))`. Tracks the
# kernel's VLLM_W4A8_V7_CFG default of "256x128"; if that default changes, this must change with it.
_W4A8_TILED_BN = 128
# What counts as "the last dispatch wave is full enough that the tail costs nothing". At 1.0 this is
# exactly "ceil(N/BN) divides the CU count"; 0.9 leaves a margin so a near-multiple (e.g. 63 tiles on
# a 64-CU card) is still treated as full. Every measured cell sits far from this edge — the ragged
# cases run 0.56-0.85 and the full ones exactly 1.00 — so the threshold is not load-bearing between
# them; it exists so an unmeasured width degrades toward wmma_tiled_tuned rather than away from it.
_W4A8_TILED_WAVE_FULL = 0.9

# ---- the mid-band tall-K / mid-N box, served by prefill_wmma_ashuffle --------------------------
# The third arm. `prefill_wmma_ashuffle` has been in the fp8_wmma package the whole time (it gets A
# out of LDS with a warp-shuffle transpose, leaving LDS to B alone and double-buffering it) and was
# in NO dispatch — so the mid-band was being argued as a two-way threshold when it is a three-way
# surface. It takes 67 of the 432 measured mid-band cells, MORE than prefill_wmma's 59.
#
# Cell count is the wrong statistic for a dispatch, though, and it nearly bought a third arm for
# nothing: ashuffle's MEDIAN winning cell is worth 1.038x against a ~1.5% run-to-run spread, i.e.
# most of those 67 cells are coin flips, while prefill_wmma's are worth up to 1.57x. What survives a
# 3-pass repeat on BOTH cards (tools/w4a8_dense_midband_ashuffle.py, fixtures
# _midband_ashuffle_{0,1}.txt — 71/72 cells kept the same winner all three passes) is a narrow box:
#
#     K=5120 N=6144  (Qwen3.6-27B q_proj, TP=1) : ashuffle wins EVERY M 17..63, both cards,
#                                                 1.00-1.09x on GPU 0, 1.08-1.16x on GPU 1
#     K=8704 N=5120  (Qwen3.6-27B down,   TP=2) : ashuffle wins EVERY M 17..63, both cards,
#                                                 1.01-1.12x on GPU 0, 1.03-1.16x on GPU 1
#     K=5120 N=8192                             : NOT in the box — GPU 0 gives it to ashuffle at
#                                                 M<=32 and GPU 1 to prefill_wmma at every M, so it
#                                                 is card-split and is left to the (a) rule above
#     K=5120 N=1024 / K=2048 N=2048 (controls)  : ashuffle LOSES 1.3-1.8x — hence a two-sided N box
#
# Adding it takes the mid-band residual from 1.36% to 0.55% of the per-cell oracle and does NOT
# raise the worst cell (1.17x either way). The bounds are deliberately TIGHT: outside them this arm
# loses by up to 1.8x, and the box is fitted to the two shipped shape families that populate it, so
# widening it is a measurement, not an edit.
_W4A8_ASHUFFLE_MIN_K = 5120
_W4A8_ASHUFFLE_MIN_N = 5120
_W4A8_ASHUFFLE_MAX_N = 6144


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
        # w_packed is (N, K/8) int32, so K is 8x its last dim and N is its leading dim. BOTH are
        # dispatch terms: K for decode_gemv's granularity precondition (else the kernel asserts), N
        # for the mid-band wide-N corner (which is not expressible in M alone — see the selector).
        kernel = _pick_dense_kernel(
            x2d.shape[0],
            weight_is_e2m1,
            group_size,
            k=w_packed.shape[-1] * 8,
            n=w_packed.shape[0],
        )
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
    ONE launch, no (M, 2*inter) HBM round-trip. Decode-only (M<=16, K%32==0, group_size%16==0);
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
