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

import os
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


# MXFP4 (OCP E2M1) MoE register-direct b128 — routes the e2m1 experts through the register-direct
# W4A16 kernel (mmq_regdirect_w4a16_moe, weight_is_e2m1=True): fp16 acts DIRECT (no act-quant, unlike
# the w4a8_moe e2m1 path which fp8-quantizes acts) + b128 weight loads. Aims to be faster AND higher
# quality (fp16 acts, matching vLLM/MLX MXFP4) — but it is NOT yet validated on the MXFP4 target, so
# it is OPT-IN (=1). Default OFF -> the validated LDS-staged W4A8 path w4a8_moe(weight_is_e2m1=True)
# (fp8 acts). Weights repacked to _w_rep/_scales_rd in post_load only when this is on.
MOE_MXFP4_REGDIRECT = _os.environ.get("MINISGL_MOE_MXFP4_REGDIRECT", "0") != "0"

# Decode gemm2 split-K. This used to be MINISGL_MOE_SPLITK pointing at a whole separate
# moe_splitk_hip package. That package hardcoded fp16 activations, so on a bf16 model it raised on
# the first decode step -- its "1.767x" could never run in production. Split-K is now an axis on the
# shared fp8_wmma core, which is activation-dtype generic, so it works for fp16 AND bf16.
#
# THE COUNT IS NO LONGER DECIDED HERE. It used to be `4 if M == 1 else 1`, described in its own
# docstring as "workload-derived" while actually being a constant plus an equality test:
#   * no sweep produced the 4, and it was never swept across M at all;
#   * `M == 1` left M=2 -- an equally occupancy-starved shape -- with NO split, purely because of
#     the shape of the test rather than anything measured;
#   * and the thing it has to be derived FROM (the launched grid: ceil(N/BN) x P/block_m blocks of
#     nwm*WARPS_N waves, against this card's SIMD count) is not visible from Python at all. BN and
#     WARPS_N are chosen inside the kernel launcher.
# So the engine now asks for 0 = "derive it", and `moe_split_slices` in moe_kernel.hip resolves it
# where the grid is actually known -- the same rule, and the same function, that carves the fused
# gather-reduce gemm2's top_k reduction.
_MOE_SPLITK_AUTO = 0
# A/B handle, not a feature gate — the twin of MINISGL_MOE_G2_SPLIT_K on the fused arm. `legacy`
# reproduces the retired `4 if M == 1 else 1` so a control leg can measure what the shipped constant
# actually cost end-to-end; any integer forces that slice count. Unset = derive.
_MOE_SPLITK_FORCE = _os.environ.get("MINISGL_MOE_SPLITK_SCATTER", "")


def _moe_split_k(M: int) -> int:
    """K-slices for the decode scatter gemm2. 0 = derive from the launched grid (see above)."""
    if _MOE_SPLITK_FORCE == "legacy":
        return 4 if M == 1 else 1
    if _MOE_SPLITK_FORCE.isdigit():
        return int(_MOE_SPLITK_FORCE)
    return _MOE_SPLITK_AUTO

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
# ROUTING-TRACE hook (measure-only). `enabled()` is one `is not None`; the module imports torch
# and kvcache._envutil only, nothing from minisgl.quant, so this cannot cycle.
from minisgl.weights import route_trace as _route_trace
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
    w_packed (N, K//pf) int32, scales (K//group, N) fp16, zeros (K//group, N//pf) int32.

    Scales/zeros keep the CHECKPOINT's group-major order: the op indexes them `[g*N + n]` so that a
    fragment's 16 lanes (which differ only in `n`) coalesce into one request. AWQ already ships
    (K//group, N), so the scale conversion is now a dtype cast with no transpose at all.
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

    scales_op = scales.contiguous().to(torch.float16)  # (G, N) — already group-major

    zeros_op = None
    if qzeros is not None:
        G = qzeros.shape[0]
        uz = (qzeros.unsqueeze(-1) >> shifts) & mask  # (G, Np, pf)
        uz = uz[:, :, rev].reshape(G, N).contiguous().to(torch.int32)  # (G, N)
        # Repack 8 CHANNELS per int32 along N, group-major: channel n = i*pf + j lands in word i,
        # nibble j — exactly what the kernel's `(wz[g*(N/8) + n/8] >> ((n%8)*4)) & 0xF` decodes.
        zeros_op = torch.zeros((G, N // pf), dtype=torch.int32, device=dev)
        for j in range(pf):
            zeros_op |= (uz[:, j::pf] & mask) << (j * bits)

    return w_packed, scales_op, zeros_op


def gptq_to_op_layout(
    qweight: torch.Tensor,  # (K//pf, N) int32, GPTQ-packed along INPUT (natural nibble order)
    scales: torch.Tensor,  # (K//group, N) fp16
    qzeros: torch.Tensor | None,  # (K//group, N//pf) int32, GPTQ-packed along N; +1 = zero point
    *,
    bits: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert ONE dense GPTQ matrix to the op's native layout
    (w_packed (N, K//pf) int32, scales (K//group, N) fp16, zeros (K//group, N//pf) int32).
    Scales/zeros stay GROUP-MAJOR so the op's 16-lane scale read coalesces (see awq_to_op_layout).

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

    scales_op = scales.contiguous().to(torch.float16)  # (G, N) — already group-major

    # qzeros: unpack (G, N//pf, pf) NATURAL along N -> (G, N), fold +1, repack 8 channels/int32 -> (G, N//pf).
    assert qzeros is not None, "GPTQ always ships qzeros"
    G = qzeros.shape[0]
    uz = (qzeros.unsqueeze(-1) >> shifts) & mask  # (G, N//pf, pf)
    uz = uz.reshape(G, N) + 1  # (G, N) actual zero point; col = np*pf + j (natural)
    assert int(uz.max()) <= mask, f"GPTQ zero+1 overflows {bits}b (max={int(uz.max())})"
    uz = uz.contiguous().to(torch.int32)  # (G, N)
    zeros_op = torch.zeros((G, N // pf), dtype=torch.int32, device=dev)
    for j in range(pf):
        zeros_op |= (uz[:, j::pf] & mask) << (j * bits)

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

    WHY THIS IS STILL ONE VARIABLE, with the shared analytic chooser sitting next to it. The dense
    tile became a cost-model choice (fp8_wmma/tile_select.h) and the same model now covers the
    grouped MoE core — ``fp8_wmma.moe_tile_choose(rows, E, n1, k1, g1, n2, k2, g2)`` returns the
    (block_m, BN, BN) this function would defer to, host-side and cudagraph-safe. It is NOT wired
    yet because it was measured, over 28 (shape, M) cells x 25 tiles (tools/w4a8_moe_tile_surface.py,
    graph-replay timed, expert stacks rotated past the 64 MB MALL by BYTE count):

        M >= 64 : 1.80x - 3.01x FASTER than this rule    (which asks for 16 rows where the oracle
                                                          wants 32x192 — a 2.3-3.0x miss)
        M <= 32 : 0.76x - 0.85x, i.e. up to 24% SLOWER   <-- the served decode point

    At M<=32 an expert holds ~1 routed row, so a block is nearly all masked padding, and no term in
    that model prices it. This rule is near-oracle exactly there (1.00x on almost every M<=32 cell)
    and 2.3-3.0x off exactly where the model is right. Trading 1.9x of prefill for 24% of decode is
    the wrong trade, so the rule stays until the model has the missing term — see the OPEN note in
    tile_select.h, and tools/w4a8_moe_tile_verify.py for the three-arm A/B that produced these.
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


def _check_moe_scale_pair(
    w: torch.Tensor, scales: torch.Tensor, zeros: "torch.Tensor | None", which: str
) -> None:
    """Assert the (scales dtype, zeros slot) PAIR is one of the two the kernel can dispatch.

    THE KERNEL PICKS ITS WScale POLICY OFF THE SCALES DTYPE and nothing else
    (`moe_kernel.hip::MOE_SCALE_FMT_DISPATCH`; `torch_binding.cpp` gates the same three dtypes). The
    `w_zeros` pointer slot is then read as AWQ packed zeros under `Fp16GroupScale` and as the
    per-output-channel f32 global under `E4m3GroupScaleGlobal`. Those are different tensors of
    different rank in the same argument, so a mismatched pair is not a shape error the op would catch
    — it is a pointer the kernel dereferences with the wrong stride. Both failure directions are
    silent and finite:

      * e4m3 scales with AWQ-shaped zeros -> the epilogue reads a packed-nibble word as an f32 global
        and every output channel is multiplied by a denormal or a huge number;
      * fp16 scales with an (E, N) global -> `Fp16GroupScale::wz_base` strides by `G*(N/8)`, so the
        zero-point read runs off the end of a tensor 1/(G/8) its expected size.

    So the pair is checked here, on the ONE call path both formats share, rather than trusted to
    line up because the container that built them happened to be consistent.

    KERNEL-SIDE STATUS (was a gap; CLOSED in rdna4-hip-kernels @ 38ec157, "the e4m3 bindings
    rejected the scale they were written for"). `torch_binding.cpp` used to validate the zeros slot
    as `w_zeros.dim() == 3 && w_zeros.size(2)*8 == N && w_zeros.size(1) == scales.size(1)` on every
    grouped-MoE entry point, which REJECTED the (E, N) global with a message naming a tensor that was
    not the one being passed. `check_moe_w_zeros` now picks the predicate off the SCALES DTYPE — the
    same single fact `MOE_SCALE_FMT_DISPATCH` selects the WScale policy from — on the three launchers
    that have an e4m3 arm (`mmq_fp8_moe_gemm`, `_gemm1_silu`, `_gemm_scatter`). This check is
    therefore no longer the thing standing between an NVFP4 checkpoint and a shape error; it is kept
    because it is the ONE call path both formats share and it still catches a MISMATCHED pair
    (fp16 scales + (E,N) global) that the binding's fp16 arm would accept as a rank-3 failure only by
    accident. Verify with a `--build-context kernels=` at or past 38ec157; an older kernels tree
    still fails at the binding, loudly.
    """
    if scales.dtype == torch.float16:
        if zeros is not None and zeros.dim() != 3:
            raise AssertionError(
                f"w4a8_moe {which}: fp16 (folded) scales pair with AWQ packed zeros (E, G, N/8) "
                f"int32, but got a {zeros.dim()}-D {zeros.dtype} tensor {tuple(zeros.shape)}. A 2-D "
                f"zeros tensor here is an NVFP4 per-output-channel global handed to the fp16 policy, "
                f"which would stride it as if it were (E, G, N/8)."
            )
        return
    if scales.dtype not in (torch.float8_e4m3fn, torch.uint8):
        raise AssertionError(
            f"w4a8_moe {which}: scales must be fp16 (folded per-group), float8_e4m3fn/uint8 "
            f"(NVFP4 two-level block scale) or uint8 with no zeros (MXFP4 E8M0); "
            f"got {scales.dtype}"
        )
    E, N = w.shape[0], w.shape[1]
    # MXFP4 E8M0: a 1-BYTE scale with NO zeros slot. Both live formats are one byte wide, so dtype
    # alone cannot separate them — the discriminator is the zeros slot, which is exactly the rule
    # the kernel's own `MOE_SCALE_FMT_DISPATCH_G` uses (`w_zeros.defined() && w_zeros.numel() > 0`).
    # Keeping the two in agreement is the whole point of checking here.
    #
    # THIS ARM IS WHY: MXFP4 group scales used to be WIDENED to fp16 at load, so they returned above
    # and never reached this check. Passing the E8M0 byte through natively (2026-09-08) made them
    # uint8, and this guard then read every MXFP4 MoE checkpoint as an e4m3 pair missing its global
    # and refused to serve it. Caught 2026-09-09 booting ZAYA1-8B-MXFP4; the qwen4_exp serve never
    # saw it because that checkpoint is genuinely NVFP4 and does carry the global.
    if scales.dtype == torch.uint8 and zeros is None:
        return
    if zeros is None:
        raise AssertionError(
            f"w4a8_moe {which}: e4m3 block scales REQUIRE the per-output-channel f32 global in the "
            f"zeros slot — the two-level scale is incomplete without it, and the kernel's "
            f"`epi(nullptr)` returns 1.0f, i.e. it would serve the block scale alone. That is a "
            f"~1/global error per weight (4.8e3x on this checkpoint), fluent and finite."
        )
    if zeros.dtype != torch.int32 or tuple(zeros.shape) != (E, N):
        raise AssertionError(
            f"w4a8_moe {which}: the NVFP4 global must be the (E, N)=({E}, {N}) f32 vector BITCAST to "
            f"int32 (the `w_zeros` slot is typed `const int*` and the kernel does "
            f"reinterpret_cast<const float*>); got {zeros.dtype} {tuple(zeros.shape)}. A "
            f"`.to(torch.int32)` instead of `.view(torch.int32)` truncates every global to 0."
        )


def w4a8_moe(
    x: torch.Tensor,  # (M, K) activations
    w13: torch.Tensor,  # (E, 2*inter, K//8) i32 — gate|up stacked
    w13_scales: torch.Tensor,  # (E, K//g, 2*inter) GROUP-MAJOR f16 (folded) | f8_e4m3 (NVFP4 block)
    w13_zeros: torch.Tensor | None,  # AWQ zeros (E,G,N/8) i32 | NVFP4 global (E,N) f32-as-i32 | None
    w2: torch.Tensor,  # (E, K, inter//8) i32
    w2_scales: torch.Tensor,  # (E, inter//g, K) GROUP-MAJOR f16 | f8_e4m3
    w2_zeros: torch.Tensor | None,
    gating_output: torch.Tensor | None,  # (M, E); ignored when topk_ids/topk_weights are given
    top_k: int,
    renormalize: bool,
    *,
    topk_weights: torch.Tensor | None = None,  # (M, top_k) f32 — precomputed route (e.g. noaux_tc)
    topk_ids: torch.Tensor | None = None,  # (M, top_k) i32 — precomputed expert ids
    kernel: str = "wmma",
    block_m: int | None = None,  # None -> derive the WMMA tile height from the workload (_moe_block_m)
    weight_is_e2m1: bool = False,  # True -> decode w13/w2 nibbles as MXFP4/NVFP4 (OCP E2M1)
    activation: str = "silu",  # gated activation on the gemm1 [gate|up] output: "silu" | "gelu"
    x_fp8: torch.Tensor | None = None,  # PRODUCER-quantized activations — see below
    act_scales: torch.Tensor | None = None,
) -> torch.Tensor:
    """Grouped W4A8 MoE forward: topk -> moe_align -> grouped GEMM(w13) -> gated activation
    -> grouped GEMM(w2) -> topk-weighted gather-reduce. Mirrors the proven
    w4a8_fp8_wmma `_run_grouped_moe` (non-GEMV, unfused-silu) path. Returns (M, K).
    `weight_is_e2m1=True` selects the kernel's MXFP4/NVFP4 (E2M1) weight decode instead of uniform
    int4. The SCALE FORMAT is then chosen by the scales DTYPE, not by a flag: fp16 = one folded
    per-group scale (MXFP4's E8M0 exponents, or NVFP4's legacy fold) and the zeros slot must be
    empty; float8_e4m3fn = NVFP4's native per-16 block scale, and the zeros slot carries its
    per-output-channel f32 global bitcast to int32. `_check_moe_scale_pair` enforces the pairing.
    `activation` picks the gated activation: "silu" (default, and the only one with a fused gemm1
    epilogue) or "gelu" == HF `gelu_pytorch_tanh` (Gemma4's routed experts), which forces the
    unfused gemm1 path below.
    PRODUCER-SIDE ACTIVATION QUANT. `x_fp8`/`act_scales` are the (M,K) e4m3 bits + (M,) f32 per-row
    scale returned by `tail_hip.rms_norm_quant` / `rms_norm_add_quant` — the post-attention RMSNorm
    feeding this sparse block, which already held the rows in registers. Supplying them deletes
    GEMM1's `moe_compute_act_fp8_kernel` dispatch and its (M,K) activation re-read, BIT-IDENTICALLY
    (verified max|Δ|=0 at M=1/5/6/30 across int4-sym/int4-asym/e2m1 and every gemm1 arm).

    Only GEMM1 can take it, and that is structural, not an oversight: GEMM2's input is the (P,inter)
    post-activation buffer written by GEMM1's epilogue, so no norm ever holds those rows. One
    producer therefore removes exactly ONE dispatch per MoE layer per step — this is a LAUNCH-COUNT
    win, and it is worth having only because decode at bs=1 is gap-bound, not bandwidth-bound.

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
    # scales are GROUP-major (E, K//group, N) -> the group count is shape[1], NOT shape[-1]. Reading
    # shape[-1] here yields N, which makes _grp nonsense, turns _gemv_ok False, and silently drops
    # decode gemm1 onto the prefill WMMA arm (~8.7x slower on this GEMM). Cost the serve A/B 1.5-1.8x
    # of decode before it was caught -- the kernel bench and the bit-identity gate both pass shapes
    # explicitly, so neither can see a DISPATCH regression. Measured 2026-08-07.
    _grp = (w13.shape[-1] * 8) // w13_scales.shape[1]
    # A wrong axis here does not raise -- it just quietly picks a slower kernel, which is invisible to
    # every parity/bench harness (they pass shapes explicitly). Make it loud instead.
    assert _grp in (16, 32, 64, 128), (
        f"w4a8_moe: derived group_size={_grp} from K={w13.shape[-1] * 8} and "
        f"G={w13_scales.shape[1]}; scales must be GROUP-major (E, K//group, N), got "
        f"{tuple(w13_scales.shape)}"
    )
    _gemv_ok = (_grp % 32 == 0) or (_grp % 16 == 0 and _NVFP4_GEMV)
    gemm1_kernel = "gemv" if (M <= _MOE_GEMM1_GEMV_MAX and _gemv_ok) else kernel
    gemm2_kernel = kernel
    _check_moe_scale_pair(w13, w13_scales, w13_zeros, "w13")
    _check_moe_scale_pair(w2, w2_scales, w2_zeros, "w2")
    # WHICH ENTRY POINTS CAN READ A TWO-LEVEL SCALE. `E4m3GroupScaleGlobal` reached the kernel bodies
    # and the THREE launchers that dispatch on the scale format — `mmq_fp8_moe_gemm`,
    # `mmq_fp8_moe_gemm1_silu`, `mmq_fp8_moe_gemm_scatter` (moe_kernel.hip's
    # `MOE_SCALE_FMT_DISPATCH`). The other three grouped launchers are still fp16-only on BOTH sides:
    # `mmq_fp8_moe_gemm1_silu_flag`, `mmq_fp8_moe_gemm_flag` and `mmq_fp8_moe_gemm2_gather_reduce`
    # each begin `TORCH_CHECK(scales.scalar_type() == at::kHalf)`.
    #
    # The two `_flag` arms were already unreachable for NVFP4 — they gate on group in {32,64,128} and
    # NVFP4 is group 16 — but that is an ACCIDENT of the group size, not a decision, so it is spelled
    # out below rather than relied on. `mmq_fp8_moe_gemm2_gather_reduce` was NOT excluded by anything
    # and it is selected at 3 <= M <= _MOE_GEMM1_GEMV_MAX, i.e. by any short prefill: a 48-layer
    # NVFP4 boot died there on its first prompt with "scales must be fp16".
    #
    # Excluded rather than emulated. The fallback is the unfused `mmq_fp8_moe_gemm` + the weightless
    # `mmq_fp8_moe_gather_reduce`, which is the bit-exact reference path this file already documents
    # `MINISGL_MOE_G2FUSE=0` as selecting — so the cost is one extra launch and a (P,K) round-trip at
    # small-M prefill, and nothing numeric. Lifting it is a KERNEL change (give the remaining three
    # launchers the same `MOE_SCALE_FMT_DISPATCH` the other three have), not an engine one.
    #
    # DERIVED FROM BOTH CONTAINERS, not just w2. `_flag1` gates the w13 (gemm1) launcher, so reading
    # only `w2_scales` here made w13's format a statement about a tensor w13 has nothing to do with.
    # On this checkpoint the two always agree (one `convert_nvfp4_moe` produces both), and the group
    # test on `_flag1` fences NVFP4 a second time, so this has never fired — which is exactly why it
    # is worth writing down rather than leaving as two accidents stacked. The predicate can only ever
    # turn a fused/flag arm OFF, so a false positive costs a launch, never a wrong number.
    _E4M3_SCALE_DTYPES = (torch.float8_e4m3fn, torch.uint8)
    _two_level = (w13_scales.dtype in _E4M3_SCALE_DTYPES) or (w2_scales.dtype in _E4M3_SCALE_DTYPES)

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
    # Reuses the route `_route_align` JUST produced — the ids the GEMM below actually
    # dereferences — so it is authoritative (the Python torch.topk route in _ep_route breaks
    # exact ties the OTHER way; see moe.py::_ep_route) and costs zero extra launches.
    if _route_trace.enabled():
        _route_trace.tracer().record(topk_ids, M, expert_ids, ntp, block_m)

    _e2m1 = "+e2m1" if weight_is_e2m1 else ""
    # The W4A8 kernel is now activation-dtype-generic (fp16 OR bf16), so pass activations in their
    # NATIVE dtype — a bf16 model no longer round-trips bf16->fp16->bf16 here (out1 follows x's dtype).
    x16 = _moe_time("cast", lambda: x.contiguous())
    # The producer pair describes `x` AS GIVEN. If `.contiguous()` had to copy, or if a caller ever
    # reshapes x here, the pair would silently describe a different tensor — so tie them together
    # loudly rather than trusting the invariant to hold forever.
    if x_fp8 is not None:
        assert act_scales is not None, "w4a8_moe: x_fp8 and act_scales must be supplied together"
        assert x_fp8.shape == (M, K) and act_scales.shape == (M,), (
            f"w4a8_moe: producer pair must be (M,K)=({M},{K}) and (M,)=({M},); "
            f"got {tuple(x_fp8.shape)} and {tuple(act_scales.shape)}"
        )
    elif act_scales is not None:
        raise AssertionError("w4a8_moe: act_scales given without x_fp8")
    _pq = "+prequant" if x_fp8 is not None else ""
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
        _flag1 = _MOE_FLAG and not _two_level and block_m in (64, 128) and \
            (w13.shape[-1] * 8) // w13_scales.shape[1] in (32, 64, 128)  # scales (E, G, N): G=shape[1]
        if _flag1:
            engaged(f"fp8_wmma.mmq_fp8_moe_gemm1_silu_flag{_e2m1}{_pq}")
            buf2 = _moe_time(
                "gemm1silu",
                lambda: fp8_wmma.mmq_fp8_moe_gemm1_silu_flag(
                    x16, w13, w13_scales, sorted_ids, expert_ids, ntp, top_k, block_m,
                    w_zeros=w13_zeros, weight_is_e2m1=weight_is_e2m1,
                    x_fp8=x_fp8, act_scales=act_scales,
                ),
            )  # (P, inter) in x's dtype
        else:
            engaged(f"fp8_wmma.mmq_fp8_moe_gemm1_silu({gemm1_kernel}{_e2m1}{_pq})")
            buf2 = _moe_time(
                "gemm1silu",
                lambda: fp8_wmma.mmq_fp8_moe_gemm1_silu(
                    x16, w13, w13_scales, sorted_ids, expert_ids, ntp, top_k, block_m,
                    kernel=gemm1_kernel, w_zeros=w13_zeros, weight_is_e2m1=weight_is_e2m1,
                    x_fp8=x_fp8, act_scales=act_scales,
                ),
            )  # (P, inter) in x's dtype
    else:
        engaged(f"fp8_wmma.mmq_fp8_moe_gemm({gemm1_kernel}{_e2m1}{_pq})")
        out1 = _moe_time(
            "gemm1",
            lambda: fp8_wmma.mmq_fp8_moe_gemm(
                x16, w13, w13_scales, sorted_ids, expert_ids, ntp, top_k, block_m,
                kernel=gemm1_kernel, w_zeros=w13_zeros, weight_is_e2m1=weight_is_e2m1,
                x_fp8=x_fp8, act_scales=act_scales,
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
    #
    # AND THAT IS WHY `MINISGL_MOE_G2FUSE=0` GATES IT TOO (plan §5.4 A1.0). The knob's stated job is
    # "revert to the bit-exact WMMA gemm2 + gather_reduce", and it did not cover this branch — so at
    # bs<=2, which is every decode step of a bs=1 serve, there was NO deterministic reference in the
    # binary at all: two runs of the same weights on the same input differ, and every weight-offload
    # numerics gate had to be scored against a measured noise floor with no way to check that the
    # floor was not hiding a real difference. With the knob off, M<=2 falls through to the
    # gather_reduce path below (its own `_MOE_G2FUSE` test also fails), which is the bit-exact one.
    # Default is unchanged: the fused scatter is still ON for every serve.
    if _MOE_G2FUSE and M <= 2:
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
    if _MOE_G2FUSE and not _two_level and M <= _MOE_GEMM1_GEMV_MAX and block_m != 128 and _gemv_ok \
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
    _flag2 = _MOE_FLAG and not _two_level and block_m == 128 and \
        (w2.shape[-1] * 8) // w2_scales.shape[1] in (32, 64, 128)  # group 32/64/128; scales (E,G,N)
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
    scales: torch.Tensor,  # (K//g, N) fp16 GROUP-MAJOR
    w_zeros: torch.Tensor | None,  # (K//g, N//8) int32 (AWQ) or None (symmetric)
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
    N13 = w13_scales.shape[2]  # 2*inter (gemm1 output width); scales are (E, K//group, 2*inter)
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


def _device_cu_count() -> int:
    """CU count the dispatch reasons about — PINNED to `_PINNED_DISPATCH_CU` so both ranks of a TP
    job agree regardless of which physical card each was leased. See the note above.

    The physical values, for reference: torch reports RDNA's WORKGROUP PROCESSORS in
    `multi_processor_count` (32 on the RX 9070 XT, 28 on the RX 9070) and a WGP is 2 CUs, so the
    dispatch unit count is twice it — 64 and 56 respectively. Measured: the mid-band boundary
    tracks 2x this number on BOTH cards and tracks neither `multi_processor_count` alone."""
    env = _os.environ.get("MINISGL_DISPATCH_CU")
    if env is None:
        return _PINNED_DISPATCH_CU
    if env not in ("0", "auto"):
        return int(env)
    # Explicit opt-in to the PHYSICAL count: heterogeneous dispatch across a mismatched pair, and
    # not reproducible across a re-lease. Only for measuring the per-device surface.
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

    MINISGL_FORCE_TILED_LINEARS=1 (debug, corruption bisection): return wmma_tiled_tuned at EVERY M,
    bypassing the decode_gemv family entirely — the prefill-proven kernels serve M=1. Slower; the
    point is that teacher-forced probes only ever exercise the tiled arm, so this is how the decode
    GEMVs get isolated on a live generation workload.

    - m <= gemv_max -> decode_gemv: a streaming GEMV that reads each weight once and dots it against
      all M rows (amortizing the weight read), writing straight to `out`. Serves the decode/decode-
      batch band; measured crossover int4 M<=8, e2m1 M<=16 (also the only e2m1-capable small-M kernel;
      decode_gemv's in-kernel cap is 16).
    - m > gemv_max -> wmma_tiled_tuned, at EVERY M, N, K and group size. One arm, no surface.

    THIS USED TO BE A THREE-WAY ARM DISPATCH and it was measured to be one — a wide-N corner to
    prefill_wmma, a tall-K/mid-N box to prefill_wmma_ashuffle, the rest to wmma_tiled_tuned, fitted
    over 432 mid-band cells. Every one of those measurements was CONFOUNDED, in a way that is worth
    writing down because it is a general trap: the three arms each carried their OWN tile knob, their
    OWN tile SET, and their OWN independently-frozen 256x128 default. Only prefill_wmma had a small-M
    tile (a hardcoded `M<128 -> BM=64`), and only ashuffle shipped BM=16/32. So "which arm wins at
    M=17" was answering "which arm happens to expose a tile near M=17" — an accident of which
    launcher had been tuned, not a property of any algorithm. The arms are bit-identical to each
    other; there was never any numerics in it.

    The fix was in the kernel: wmma_tiled_tuned now derives (BM x BN) from the shape via an analytic
    cost model over a 77-tile lattice (rdna4-hip-kernels fp8_wmma_rocm/tile_select.h). Against a
    per-cell oracle over 210 graph-replay-timed cells the hard-wired 256x128 was 1.331x off (worst
    2.95x); the best possible CONSTANT tile is 1.120x; the chooser is 1.042x. Re-timed live it is
    1.296x faster than the hard-wire over 88 cells, peaking at 2.89x.

    WITH THE TILE DERIVED, THE ARMS HAVE NO REGIME (tools/w4a8_dense_tile_surface.py and
    tools/w4a8_dense_tile_arms.py; fixtures _tile_surface.csv, _tile_arms.csv, _tile_verify.txt):

        at their SHIPPED tiles, 210 cells       prefill_wmma 0 wins, ashuffle 0 wins
        TILE-EQUALIZED, 610 same-tile pairs     tiled 1.225x faster geomean than ashuffle
        each arm at its OWN best tile, 80 cells tiled 70, ashuffle 5, prefill_wmma 5
                                                 — and all 10 losses are an artifact of restricting
                                                 tiled to ashuffle's 8-tile set; over the full
                                                 lattice it takes them back (q27.gate_up tp1
                                                 N=34816 M=17: prefill 800 us, tiled@256x128 986 us,
                                                 the chooser's actual pick 128x64 634 us)
        prefill_wmma small-M tile OFF           2.204x SLOWER across the mid-band than ON

      ashuffle IS algorithmically different (B-only LDS, double-buffered, A-layout shuffle) and it
      shows: at the 256x256 tile it beats tiled by 1.64x, the one place its staging pays. That is a
      reason to fold its staging into the shared core as a policy, not a reason to keep a second
      kernel and a dispatch rule — its own-best never catches tiled's own-best at any cell.

      ONE regime survives at ~1.06x: GLM (group_size=128, N=10240) at M=17..24, prefill 100.8 us vs
      the chooser's 106.9. That is ~6 us — one kernel dispatch on this box — and it is precisely
      what the tiled path pays and prefill does not: prefill FUSES the activation fp8-quant into its
      prologue, while wmma_tiled_tuned takes pre-quantized activations and so eats a separate
      `compute_act_fp8_and_scales_kernel` launch plus an (M,K) uint8 HBM round-trip inside
      `mmq_fp8_gemm`. Producer-side fusion is the fix; a second GEMM is not. Note the tiled arm was
      carrying that handicap in every number above and still won.

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
    if os.environ.get("MINISGL_FORCE_TILED_LINEARS") == "1":
        return "wmma_tiled_tuned"
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
    # THE MID-BAND IS NO LONGER A SURFACE, so there is no longer an M threshold here either.
    #
    # It used to be a three-way arm dispatch — a wide-N corner to prefill_wmma, a tall-K/mid-N box
    # to prefill_wmma_ashuffle, the rest to wmma_tiled_tuned. That whole surface was an artifact of
    # a MISSING TILE SELECTOR, not of anything algorithmic. The three arms each carried their own
    # tile knob, their own tile SET, and their own independently-frozen 256x128 default, so "which
    # arm wins at M=17" was really "which arm happens to expose a tile near M=17": prefill_wmma was
    # the only arm with a small-M branch (M<128 -> BM=64) and ashuffle the only one shipping
    # BM=16/32. wmma_tiled_tuned now derives its tile from the shape (an analytic cost model over a
    # 77-tile lattice; kernels' tile_select.h), the asymmetry is gone, and so is the surface.
    #
    # RE-MEASURED with the chooser live (graph-replay timed, weights rotated past the 64 MB MALL by
    # BYTE count; tools/w4a8_dense_tile_surface.py + tools/w4a8_dense_tile_arms.py, fixtures
    # _tile_surface.csv / _tile_arms.csv / _tile_verify.txt):
    #
    #   * at their SHIPPED tiles, over 210 cells (14 dense shapes x M=1..2048), prefill_wmma and
    #     prefill_wmma_ashuffle win ZERO cells against the tiled arm's per-cell oracle tile.
    #   * TILE-EQUALIZED over ashuffle's whole 8-tile set (which wmma_tiled_tuned now instantiates
    #     in full), 610 same-tile pairs: tiled is 1.225x faster geomean. ashuffle leads at exactly
    #     one tile, 256x256 (1.64x its way) — real, and it is where its B-only double-buffered
    #     staging pays — but its own-best never catches tiled's own-best.
    #   * at EACH ARM'S OWN BEST TILE, 80 cells: tiled wins 70. All 10 losses are an artifact of
    #     restricting tiled to the 8-tile common set; over the full lattice it takes them back —
    #     the loudest, q27.gate_up tp1 N=34816 M=17, is prefill 800 us vs tiled@256x128 986 us, and
    #     the chooser's actual pick there (128x64) is 634 us.
    #   * prefill_wmma with VLLM_W4A8_DENSE_SMALLM_OFF=1 is 2.204x SLOWER across the mid-band than
    #     with it on. Its entire mid-band edge was that one hardcoded tile.
    #
    # ONE regime used to survive, at ~1.06x: GLM (group_size=128, N=10240) at M=17..24 — prefill
    # 100.8 us vs the chooser's 106.9. That gap was ~6 us, i.e. ONE kernel dispatch on this box, and
    # it was exactly what the tiled path paid and prefill did not: prefill FUSES the activation
    # fp8-quant into its prologue, while wmma_tiled_tuned consumes pre-quantized activations and
    # therefore ate a separate `compute_act_fp8_and_scales_kernel` launch plus an (M,K) uint8 HBM
    # round-trip inside `mmq_fp8_gemm`. The prescribed fix was producer-side fusion (quantise in the
    # RMSNorm/residual that already has the values in registers), not a second GEMM.
    #
    # THAT FIX IS BUILT, AND IT CLOSES THE REGIME. tail_hip `rms_norm_quant`/`rms_norm_add_quant`
    # emit the e4m3 row as an EPILOGUE on a kernel that was already one-block-per-row with a block
    # reduce (no extra LDS, no grid change), and `w4a8_linear(x_fp8=, act_scales=)` feeds them to the
    # SAME core — a WLoad-style activation POLICY, not a new kernel. Bit-identical at
    # M=1/17/33/64/129/512 in bf16 and fp16 through this auto-dispatch.
    #
    # RE-MEASURED end-to-end (producer + linear, graph-replay, card 0, rotation past the 64 MB MALL;
    # tools/w4a8_producer_actquant.py, fixture tools/_fixtures/producer_actquant_card0.csv). The
    # probe reproduces dense_tile_arms.csv on the shipped arms first:
    #
    #   * glm.gate_up tp2 M=17: prefill 102.8 / chooser 108.4 (1.054x — the recorded 1.06x, hit
    #     almost exactly) / chooser+FUSED 103.9. The fusion is worth 1.043x here, and 1.045x at
    #     M=24, leaving prefill ahead by 1.010x / 1.014x — noise.
    #   * q27.gate_up tp1 M=17/24/32: the CHOOSER (612-627 us) was ALREADY 1.24-1.32x FASTER than
    #     prefill (764-822 us) before any fusion. The fixture's "prefill wins 1.11-1.23x" there was
    #     measured against the best of the RESTRICTED common tile set (841-984 us), never against
    #     the chooser — the same restriction artifact called out three bullets up.
    #
    # => 5/5 cells. `prefill_wmma` no longer wins anywhere on this surface and is RETIRABLE. The
    # measured saving is a flat ~3-5 us per dense linear (one dispatch), so it reads as ~4% at
    # glm's 108 us and as noise at q27's 620 — it is a DISPATCH win, not a bandwidth one, and it
    # scales with launch count, not with shape.
    #
    # `k` and `n` are still taken and still used above (decode_gemv's K precondition); they no
    # longer select an arm. The two prefill arms remain in the package, reachable by explicit
    # `kernel=`, so the sweeps above stay reproducible — but nothing dispatches to them.
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
# The prefill/mid-band ARM constants that used to live here — _W4A8_PREFILL_TILED_MIN,
# _W4A8_PREFILL_WIDE_N, _W4A8_PREFILL_WIDE_MAX_M, _W4A8_TILED_BN, _W4A8_TILED_WAVE_FULL,
# _W4A8_ASHUFFLE_{MIN_K,MIN_N,MAX_N}, and the _tiled_last_wave_occupancy() they fed — are DELETED,
# not merely unused. All eight described a THREE-WAY ARM SURFACE that only existed because
# wmma_tiled_tuned's tile was hard-wired to 256x128 while the other two arms carried tiles of their
# own. With the tile derived from the shape in the kernel (tile_select.h), the surface is gone and
# every one of those constants is a fossil that would silently re-open the mistake the moment
# someone "tuned" it. `_W4A8_TILED_BN = 128` in particular was a COPY of the kernel's default tile
# living in a second file — exactly the kind of duplicated constant that goes stale; the tile is
# now asked for, not assumed (`fp8_wmma.dense_tile_explain(M, N, K, group_size)`).


def w4a8_linear(
    x: torch.Tensor,
    w_packed: torch.Tensor,  # (N, K/8) int32, op layout
    scales: torch.Tensor,  # (K/group, N) fp16 GROUP-MAJOR — N contiguous (binding asserts size(1)==N)
    w_zeros: torch.Tensor | None,  # (K/group, N/8) int32 (AWQ asym) or None (sym)
    group_size: int,
    kernel: str | None = None,
    weight_is_e2m1: bool = False,  # True -> decode nibbles as MXFP4 (OCP E2M1); w_zeros must be None
    x_fp8: torch.Tensor | None = None,  # PRODUCER-quantized activations — see below
    act_scales: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dense W4A8 GEMM: (M, K) @ (N, K)^T -> (M, N). Output follows the activation dtype (fp16 OR
    bf16 — the kernel is activation-dtype-generic), so a bf16 model runs cast-free (the caller's
    out.to(x.dtype) is then a no-op). `weight_is_e2m1=True` selects the kernel's MXFP4 (E2M1) weight
    decode instead of uniform int4 (scales are the E8M0 group exponents folded to fp16; w_zeros MUST
    be None — the op asserts symmetric).

    PRODUCER-SIDE ACTIVATION QUANT. `x_fp8`/`act_scales` are the (M,K) e4m3 bits + (M,) f32 per-row
    scale returned by `tail_hip.rms_norm_quant` / `rms_norm_add_quant` — the RMSNorm or residual-add
    feeding this linear, which already held the values in registers. Supplying them deletes the op's
    own `compute_act_fp8_and_scales_kernel` dispatch and the (M,K) activation re-read, and the result
    is BIT-IDENTICAL (the producer runs the same amax/448 with the same 1e-8 floor, the same
    reciprocal-multiply and the same UNCLAMPED f32_to_e4m3, over the same rounded output row).
    This closes the one regime `_pick_dense_kernel` still credits to `prefill_wmma`. BOTH
    auto-dispatched arms (`decode_gemv`, `wmma_tiled_tuned`) consume the pair, so ONE producer fusion
    covers 100% of the served dense surface; the two prefill arms quantize in-prologue and the op
    REFUSES the pair loudly rather than ignoring it."""
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
    engaged(
        f"fp8_wmma.mmq_fp8_gemm({kernel}{'+e2m1' if weight_is_e2m1 else ''}"
        f"{'+prequant' if x_fp8 is not None else ''})"
    )
    return fp8_wmma.mmq_fp8_gemm(
        x2d, w_packed, scales, kernel=kernel, w_zeros=w_zeros, weight_is_e2m1=weight_is_e2m1,
        x_fp8=x_fp8, act_scales=act_scales,
    )


def w4a8_linear_silu(
    x: torch.Tensor,
    w_packed: torch.Tensor,  # (2*inter, K/8) int32 [gate|up], op layout
    scales: torch.Tensor,  # (K/group, 2*inter) fp16 GROUP-MAJOR
    w_zeros: torch.Tensor | None,  # (K/group, (2*inter)/8) int32 (AWQ) or None (sym)
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

    if kernel is None and os.environ.get("MINISGL_FORCE_TILED_LINEARS") == "1":
        kernel = "prefill_tiled"  # bisection: bypass the fp8 decode gemv (see _pick_dense_kernel)
    engaged(f"fp8_wmma.mmq_w8a8_gemm({kernel or 'auto'})")
    return fp8_wmma.mmq_w8a8_gemm(x, w_fp8, scales, kernel)
