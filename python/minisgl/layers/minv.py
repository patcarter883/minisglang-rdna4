"""M-invariant dense GEMM — the engine's standing rule for bf16/fp16 linears.

WHY THIS EXISTS
    torch's F.linear dispatches to rocBLAS, whose kernel / tile / split-K reduction ORDER is chosen by
    the problem shape (M,N,K). Float addition is non-associative, so linear(x[:m],W) is NOT bit-identical
    to linear(x[:M],W)[:m] when m != M: the same token's output changes by ~1 bf16 ULP depending on how
    many tokens share the batch. This silently corrupts every pathway where a token is computed at
    different M and expected to match:
      * prefix / radix caching  (cached continuation at M=new_len vs fresh at M=full_len)
      * chunked prefill         (a long prompt split by max_extend runs each chunk at a different M)
      * spec-decode VERIFY      (K+1 tokens/seq at variable M vs sequential decode at M=1)
    A coarse int8/fp8 activation downcast downstream amplifies the ULP seed into flipped greedy tokens
    (observed: ZAYA GSM8K 45->25 with CCA prefix caching on).

THE FIX
    Route bf16/fp16 dense linears through our OWN fixed-tile WMMA GEMM (the moe_bf16_wmma spine driven
    dense: single expert, identity routing). It does the FULL K-reduction per output tile in a fixed
    order with NO split-K, so the reduction order does not depend on M -> M-invariant BY CONSTRUCTION,
    and WE own the tile so it cannot drift under a rocBLAS/hipBLASLt upgrade. Verified bit-exact across
    M for the ZAYA projection shapes; parity vs F.linear is ~1 bf16 ULP (a different but fixed order).

SCOPE ("as close to M-invariant as realistically possible")
    Applied for eager bf16/fp16 GEMMs whose K is WMMA-friendly (IN % 16 == 0). Falls back to F.linear
    (with a one-time warning) for: fp32/other dtypes, IN not a multiple of 16, or under cudagraph
    capture (static shapes are already self-consistent, and the arange/route tensors would allocate
    mid-capture). Integer (int8) matmuls are already exact/M-invariant.

    "quantized-expert and attention kernels are already fixed-tile HIP" USED TO STAND HERE. It was an
    assertion, never a measurement, and it is only half true (measured 2026-08-05,
    tools/quant_m_invariance.py, 7 shipped dense shapes + the Gemma4-shaped grouped MoE):
      * each quantized kernel ARM is indeed fixed-tile and M-invariant on its own — rows[0:m] computed
        alone are bit-identical to the same rows inside a batch of 2048, max|delta| = 0, for all three
        dense arms — and the grouped-MoE `_moe_block_m` tile (16/32/64/128) is bit-NEUTRAL;
      * but quant/kernels.py DISPATCHES BETWEEN ARMS as a function of M, and the arms are not all the
        same numbers. Dense: prefill_wmma == wmma_tiled_tuned bit-for-bit, but decode_gemv differs from
        both by up to 1.953e-3 abs (~1e-3 rel). MoE: gemm1 swaps gemv<->wmma at M=32 (up to 4.9e-4),
        and gemm2 swaps its gather-reduce for an ATOMIC SCATTER at M<=2 which is not even deterministic
        against itself (measured 9.5e-7 to 2.4e-4 between two identical consecutive calls).
    So a quantized model is M-invariant only WITHIN an arm band. `_W4A8_GEMV_MAX_INT4` was raised
    8 -> 16 for exactly the reason `_DECODE_GEMV_MAXM = 16` exists below — to put ordinary decode and
    spec-decode verify on the same arm. The MoE gemm2 seam at M<=2 has no such fix and is a standing
    limit: on a MoE target, verify (M>=3) is ALWAYS on a different gemm2 arm from plain decode (M=1),
    at every K and every batch size.
"""
from __future__ import annotations

import os

import torch

# OVERRIDES, not defaults. These were the pinned tile for every shape and every M; the tile is now
# derived per shape from a measured surface (see the selection block in minv_linear). 0 = derive.
# They stay as an escape hatch and as the knob the sweep harness drives.
_BLOCK_M_OVERRIDE = int(os.environ.get("MINISGL_MINV_BLOCK_M", "0"))  # WMMA M-tile (mult of 16, <=128)
_BN_OVERRIDE = int(os.environ.get("MINISGL_MINV_BN", "0"))            # WMMA N-tile (must divide OUT)
# The signature defaults below keep working for direct callers that pass neither.
_BLOCK_M = _BLOCK_M_OVERRIDE or 64
_BN = _BN_OVERRIDE or 64
# ---------------------------------------------------------------------------------------------
# WHICH KERNEL, AND WHY. All three dense_gemm kernels are bit-identical (identical fixed 16-wide K
# order, no split-K), so this dispatch may pick purely on speed and stays M-invariant whatever it
# picks. It used to be ONE constant, `_PIPE_M`, and one constant cannot express this surface:
#
#   rd (register-direct: no LDS, no barrier) re-reads the WHOLE B panel once per 16 rows of M — it has
#     no LDS staging and no M-register-blocking, so its cost grows as ~ceil(M/16)*OUT*IN. That is free
#     when the panel is small or M is small and catastrophic otherwise. The ISA says why, and it is
#     NOT this repo's usual answer: rd does not spill (VGPR 38-82, scratch 0, LDS 0 — the best
#     occupancy of the three). Per 16-wide K step at BN=64 it issues 5 global b128 loads, 4 WMMAs and
#     4 `s_wait_loadcnt` — a dependent load->wait->mma chain with ZERO prefetch depth, so every WMMA
#     is gated on a fresh ~300-cycle global load and only wave count hides any of it.
#   pipe prefetches a whole K-chunk ahead (A into a register double-buffer, B into a double-buffered
#     LDS tile), so its inner loop is ds_read + WMMA with the global loads a chunk in front: 32 WMMAs
#     per 6 `s_wait_loadcnt` at BN=64, and B read once per block_m (128-256) instead of once per 16.
#     The price is a fixed ~25-60 us floor (VGPR 153-217, LDS staging, a barrier per K-chunk) and a
#     coarse grid, so it needs BOTH a wide OUT and real M before it pays for itself.
#   lds is the general fallback and the only one that can mask a ragged OUT tile.
#
# Thresholds are from dense_gemm/local/sweep_policy.py: a CUDA-graph-timed sweep of every callable
# tile over the shapes this engine actually dispatches, with the weights rotated past the 64 MB
# Infinity Cache. Both of those matter. A per-call `synchronize()` has a ~40 us wall floor on this
# box — larger than most of these kernels, so it makes them all look identical below M~192. And a
# hot-weight benchmark flatters `rd` specifically, because re-reading B costs nothing when B never
# leaves cache; under a 30-layer ~700 MB weight working set it is not in cache.
#
# Result vs rocBLAS (best-of-family / rocBLAS, cold, RX 9070) on the four shapes Gemma4 and
# DiffusionGemma actually route here: gate_up 0.76-1.21x, down 1.05-1.19x, router 1.12-1.26x,
# lm_head 0.93-1.09x — we BEAT it at M=64..128 on gate_up and at M<=128 on the LM head. The residual
# deficit is bounded by ~1.26x and is kernel debt to close, not a reason to call rocBLAS: F.linear is
# reached only through minv_supported()'s dtype / ragged-K fallback.
_RD_MAX_M = int(os.environ.get("MINISGL_MINV_RD_MAX_M", "128"))
# ...but M alone is not enough, which is the flaw a single threshold cannot fix: rd's B re-read scales
# with OUT too, so an LM-head-width output leaves rd's regime almost immediately. At the SAME M=128,
# rd is the fastest kernel on gate_up (35.5 us vs pipe 51.0) and the slowest on the LM head (2216 us
# vs pipe 1242) — the ranking inverts on OUT, not on M. M*OUT is the term that separates them; 512K
# puts every real shape on the correct side with an order of magnitude of margin (gate_up M=128 ->
# 270K rd, down M=128 -> 360K rd, lm_head M=32 -> 4.2M pipe).
_RD_MAX_MN = int(os.environ.get("MINISGL_MINV_RD_MAX_MN", str(512 * 1024)))
# Narrow OUT: pipe's grid is (ceil(OUT/BN), ceil(M/block_m)). Gemma4's router is OUT=128 — TWO N-tiles
# at BN=64 — so pipe runs 2-4 workgroups on a 64-CU part and costs a FLAT ~42 us at every M, while rd
# does the same work in ~16 us. This is the regression the previous single `_PIPE_M=256` shipped: it
# sent every router call at the 256-token canvas to pipe, 2.6x slower, 30 calls per step.
_PIPE_MIN_OUT = int(os.environ.get("MINISGL_MINV_PIPE_MIN_OUT", "512"))
_PIPE_MI = int(os.environ.get("MINISGL_MINV_PIPE_MI", "2"))   # pipe register-block M-subtiles/warp
# pipe K-chunk. IN % PBK != 0 is FINE now — the kernel grew a zero-filled K-tail. It did not have one,
# and that silently excluded a whole layer class: a row-parallel down_proj shards IN by the TP degree,
# so Gemma4's intermediate 2112 becomes IN=1056 at TP=2, which is not a multiple of 64, so `mlp.down`
# could never reach pipe at any M and sat on rd forever (M=1024: 110 us rd -> 78 us pipe).
_PIPE_PBK = int(os.environ.get("MINISGL_MINV_PIPE_PBK", "64"))
# ---- SPLIT-K REDUCTION ORDER --------------------------------------------------------------------
# A narrow-OUT, deep-K GEMM (the router) has no parallelism to give: rd's loop is a dependent
# global-load -> s_wait -> WMMA chain IN/16 deep, so it costs ~IN/16 L2 latencies and is FLAT in M
# (18.3-19.6 us from M=16 to M=1024 at IN=2816/OUT=128). Cutting K into SK slices divides that chain
# and multiplies the grid, and it is the ONLY thing that has ever beaten rocBLAS on this shape: 0.57x
# at M=16, 0.83x at M=384, where the whole non-split tile lattice's own oracle never got below 1.19x.
#
# TWO SEPARATE DECISIONS, and conflating them is what stalled this for a session:
#
#  1. WHICH SHAPES use the split-K reduction order. This IS a bit-move against plain rd, so it is a
#     per-shape commitment that must hold at EVERY M — never a function of M. `split_k_slices(IN)`
#     reads IN and nothing else, by construction, so one order serves the whole M ladder.
#  2. HOW that order is scheduled. `grid_split=True` fans slices across blockIdx.z and reduces fp32
#     partials; `False` walks the same slices in one block with a running total. These are
#     BIT-IDENTICAL (gated below), so this one is free to depend on M — and must, because the
#     fan-out wins only while M cannot fill the machine on its own (2.79x rocBLAS at M=2048).
#
# Gates, all green on card 0, 2026-08-07 (fixture: rdna4-hip-kernels
# tools/_fixtures/splitk_router_ladder_card0.txt):
#   * schedule bit-identity, every M on the ladder 16..8192: torch.equal == True
#   * M-invariance ACROSS the schedule boundary: M=1024 (slice) vs chunks of 16/32/64/128/192/256
#     (grid) -> max|d| 0.000e+00
#   * EXPERT-FLIP, the criterion that actually decides an MoE router (dense_gemm/local/
#     splitk_router_flips.py against real captured gemma4_router.pt, 5 layers / 2800 real rows):
#     0/22400 top-k index mismatches. A "max rel delta 3e-7" would have said nothing here — top-k is
#     a step function and one ULP can reroute a token.
# Cost above the crossover: the sliced schedule tracks plain rd within +/-1% (16 extra VGPRs at BN=32).
#
# ANY NEW SHAPE that enters this band changes numerics and needs splitk_router_flips.py re-run
# against a real capture of ITS inputs. OUT<=256 keeps the band to the router class deliberately.
# IN >= 1024 guarantees split_k_slices() >= 2 — below it the op falls back to PLAIN RD ORDER
# internally, which would silently re-mix the two orders for the same weight.
# rd N-tile selection (see the block in minv_linear). _CUS is the gfx1201 compute-unit count — note
# torch reports WGPs, not CUs, so do NOT read this from device properties.
_CUS = 64
# A-traffic must be worth the halved grid. Fitted between mlp.down M=192 (M*IN=203K, wants bn64) and
# mlp.gate_up M=64 (180K, wants bn32).
_RD_BN64_MIN_MN = 192 * 1024
# ...and the halved grid must still leave the machine reasonably busy. THIS IS THE WEAKEST CONSTANT
# IN THE FILE: the fixture has exactly two shapes in the regime where it bites and they disagree —
# o_proj M=128 (44 tiles at bn64) wants bn64, mlp.gate_up M=128 (33 tiles) wants bn32 — so any value
# in (33, 44] fits and nothing in the data picks one. Re-fit it if a third shape lands in the band.
_RD_BN64_MIN_TILES = 40
_SPLITK_MAX_OUT = int(os.environ.get("MINISGL_MINV_SPLITK_MAX_OUT", "256"))
_SPLITK_MIN_IN = int(os.environ.get("MINISGL_MINV_SPLITK_MIN_IN", "1024"))
_SPLITK_GRID_MAX_M = int(os.environ.get("MINISGL_MINV_SPLITK_GRID_MAX_M", "448"))
_warned: set[str] = set()


def _warn_once(key: str, msg: str) -> None:
    if key not in _warned:
        _warned.add(key)
        try:
            from minisgl.utils import init_logger

            init_logger(__name__).warning(msg)
        except Exception:
            pass


# Decode fast path through the shared bf16/fp16 GEMV — see the note in minv_linear.
#
# UNCONDITIONAL — no env gate. Both sides of the threshold are individually M-invariant (the GEMV
# per-(row,col) in a fixed K-order; dense_gemm with a fixed full-K reduction), so the only hazard is
# the CROSSING between them. MAXM=16 is the M ceiling the GEMV accepts, so ordinary decode and
# spec-decode VERIFY (M=K+1) land on the SAME kernel rather than opposite sides of the threshold —
# that is what keeps verify bit-matching sequential decode.
# Measured vs the old threshold of 2: +19.8% at conc=4, +7.9% at conc=8, neutral at bs=1
# (tools/_maxm_ab.sh).
_DECODE_GEMV_MAXM = 16
_fallback_seen: set = set()
_decode_gemv_fn = None
_decode_gemv_probed = False


def _get_decode_gemv():
    """Lazily resolve fp8_wmma.dense_bf16_gemv (None if the kernel package is unavailable)."""
    global _decode_gemv_fn, _decode_gemv_probed
    if not _decode_gemv_probed:
        _decode_gemv_probed = True
        try:
            from fp8_wmma import dense_bf16_gemv

            _decode_gemv_fn = dense_bf16_gemv
        except Exception:
            _decode_gemv_fn = None
    return _decode_gemv_fn


def minv_supported(x: torch.Tensor, weight: torch.Tensor) -> bool:
    """True iff `minv_linear` will run the M-invariant kernel (else it falls back to F.linear)."""
    if weight.dtype not in (torch.bfloat16, torch.float16):
        return False
    if weight.shape[-1] % 16 != 0:  # IN must be WMMA-friendly (full-K reduction in 16-wide steps)
        return False
    # Under CUDA-graph capture we STILL use dense_gemm (the whole point of removing rocBLAS): its only
    # allocations are F.pad + the output tensor, both via torch's caching allocator, which is capture-
    # safe (the graph memory pool). The old F.linear fallback here reintroduced rocBLAS/hipblaslt AND
    # faulted the spec-verify capture (hipblaslt's own workspace malloc is illegal mid-capture). Set
    # MINISGL_MINV_CAPTURE=0 to restore the fallback if a capture regression appears.
    if (os.environ.get("MINISGL_MINV_CAPTURE", "1") == "0"
            and torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()):
        return False
    return True


def minv_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None,
                *, block_m: int | None = None, BN: int | None = None) -> torch.Tensor:
    """M-invariant drop-in for F.linear: C = x @ weight^T (+bias). Accepts N-D x (flattened to 2-D).
    Falls back to F.linear when the M-invariant kernel is unavailable for this dtype/shape/context."""
    import torch.nn.functional as F

    from minisgl._hip_engage import engaged

    if not minv_supported(x, weight):
        if weight.dtype in (torch.bfloat16, torch.float16) and weight.shape[-1] % 16 != 0:
            _warn_once(f"K{weight.shape[-1]}",
                       f"minv_linear: IN={weight.shape[-1]} not a multiple of 16 -> F.linear fallback "
                       f"(this GEMM is NOT M-invariant; see layers/minv.py)")
        # Announce the fallback. The engage ledger covered the QUANTIZED surface but was blind to
        # bf16, so the ~190 dense_bf16_gemv dispatches/step on GLM — and every rocBLAS escape like
        # this one — were invisible to the "profile what is DISPATCHED" check that the ledger exists
        # to serve. engaged() is first-call-per-name only, so this costs one set lookup per call.
        # Build the tag only once per (dtype, K): engaged() is first-call-per-NAME, but the
        # f-string feeding it would otherwise be evaluated on every call of a fallback path.
        _fb_key = (weight.dtype, weight.shape[-1])
        if _fb_key not in _fallback_seen:
            _fallback_seen.add(_fb_key)
            engaged(f"torch.F_linear(ROCBLAS_FALLBACK:dt={weight.dtype},K={weight.shape[-1]})")
        return F.linear(x, weight, bias)

    # DECODE fast path: the WMMA-tiled dense_gemm is built for large M and stalls at M=1 — it pads M
    # up to block_m (63/64 of the tile is padding) and the decode profile puts this chokepoint at
    # 2.58 ms/step (dense_gemm_rd, 160 calls) + 1.03 ms/step (the ragged dense_gemm on OUT=1
    # shared_expert_gate, 40 calls) = 3.6 ms of a 13.19 ms kernel step. The shared decode GEMV
    # (gemv_decode_core<Bf16GemvLoader>, packed v_dot2_f32_bf16) runs the same shapes at 1.5-3x —
    # it is the same trade the LM head already took (embedding.py) and the GDN projections took in
    # gdn/layer.py::_GemvLinear.
    #
    # M-invariant (row i is a per-(row,col) fp32 dot in a fixed K-order, bit-identical at any M —
    # measured max|d|=0.0 across M=1/2/8/16). It is NOT bit-identical to the dense_gemm family, so
    # the threshold is a crossing; MAXM=16 puts decode AND spec-verify on this side of it.
    if (x.dim() == 2 and x.shape[0] <= _DECODE_GEMV_MAXM and bias is None
            and x.dtype == weight.dtype):          # minv_supported() gated weight.dtype, not x's
        gemv = _get_decode_gemv()
        if gemv is not None and weight.shape[-1] % 8 == 0:
            engaged("fp8_wmma.dense_bf16_gemv(minv_decode)")
            return gemv(x.contiguous(), weight)

    import dense_gemm as _dg

    orig_shape = x.shape
    x2 = x.reshape(-1, orig_shape[-1]).contiguous()
    M, IN = x2.shape
    OUT = weight.shape[0]
    # All three dense_gemm kernels (rd / pipe / lds) run the FULL K-reduction per output tile in the
    # identical fixed 16-wide order with NO split-K, so every one is bit-identical per output row and
    # interchangeable -> the dispatch below is M-invariant regardless of which kernel a given M picks.
    # Full-tile kernels need block_m | M and BN | OUT; pad M up to the tile (padded rows are sliced off
    # and never touch the real rows). See the threshold block at the top of this file for the measured
    # surface each clause encodes.
    def _padded(bm: int) -> torch.Tensor:
        Mp = ((M + bm - 1) // bm) * bm
        return x2 if Mp == M else torch.nn.functional.pad(x2, (0, 0, 0, Mp - M))

    # ---- TILE SELECTION: block_m and BN are per-shape now, not module constants -------------------
    # They used to be `_BLOCK_M = 64` and `_BN = 64` for EVERY shape and every M, so two of the four
    # real axes were pinned. The ARM rule below is unchanged — it is right on 34 of 36 cells — but the
    # tile it hands the arm was a constant, and the surface says the tile is where the loss is.
    #
    # Fixture: tools/_fixtures/dense_gemm_surface_card0.csv in rdna4-hip-kernels (2,844 cells: the 6
    # served shapes x M in {16,64,128,192,256,512} x the full lds/rd/pipe lattice INCLUDING MI=1,
    # cold weights rotated past the 64 MB MALL, CUDA-graph-replay device time, rocBLAS in every cell).
    # Scored against the per-cell oracle over our own arms:
    #
    #                      vs oracle   vs rocBLAS   beats rocBLAS   worst cell
    #   pinned (before)      1.0812      1.1580         9/36          1.89x
    #   THIS RULE            1.0204      1.0928        13/36          1.33x     and ZERO regressions
    #
    #   * rd's block_m TRACKS M. The oracle picks bm16/48 at M=16, bm64 at M=64, bm128 at M=128 on
    #     every shape — obvious in hindsight, since rd's warps each own 16 rows and a bm below M just
    #     launches more row-blocks of the same work while a bm above M pads.
    #   * rd's BN follows OUT, not M. Wide outputs want 64, narrow want 32 (the router's whole grid is
    #     OUT/BN tiles, so halving BN doubles the only parallelism it has).
    #   * pipe needed one more staircase step: at an LM-head-width OUT the oracle is bm128 from M=128,
    #     but the old staircase held bm64 until M=256, which was the single worst cell in the set
    #     (lm_head M=128, 1.89x rocBLAS). The existing `OUT>=65536 and M>=192 -> 256` clause is right
    #     and is kept; this adds the step below it.
    #
    # NOTE the MI=1 trap this fixture had to be rebuilt to avoid: `mi = _PIPE_MI if pbm >= 256 else 1`
    # means the engine runs MI=1 everywhere below pbm=256, but the sweep's config list predated those
    # instantiations and carried only MI in {2,4}. Scoring against it silently graded the policy on
    # configs it never launches. If you extend the lattice here, extend PIPE_INST with it.
    _wide = OUT >= 65536
    if OUT < _PIPE_MIN_OUT or (M <= _RD_MAX_M and M * OUT <= _RD_MAX_MN):
        # bm MUST NOT force an M-pad. `_padded()` rounds M up to bm and that pad is a real
        # allocate+copy inside minv_linear — but the fitting surface hoisted it OUT of the timed
        # region (sweep_policy.py builds `xp = {bm: pad(x, bm)}` once, before gbench), so every
        # config whose bm does not divide M was priced too cheap and this rule then picked one.
        # Measured end-to-end: router M=192 chose bm128 (pad 192->256) at 25.3 us against 21.2 for
        # the unpadded bm64 it replaced — a +19.1% REGRESSION — while the surface claimed
        # bm128/bn32 there was 16.07 us, i.e. a ~9 us pad the surface never charged.
        # bm barely moves the router (15.6-16.1 us across the whole bm lattice at bn32), so the
        # right objective is: minimise PADDED ROWS first, then take the largest such tile (fewest
        # row-blocks). Exact for every M that is a multiple of any lattice entry.
        bm = min((16, 32, 48, 64, 96, 128), key=lambda b: (-(-M // b) * b, -b))
        # rd's N-tile. `OUT >= 4096 -> 64 else 32` was wrong because OUT alone cannot express the
        # trade; there are two effects pulling opposite ways and only one of them scales with M:
        #   * A-TRAFFIC. rd re-reads the whole A panel once per N-tile, so A bytes ~ (OUT/BN)*M*IN.
        #     Doubling BN halves it. GROWS with M.
        #   * PARALLELISM. The grid is (OUT/BN)*ceil(M/bm) blocks. Doubling BN halves it, and below
        #     the 64-CU count that is pure loss. FLAT in M.
        # So the crossover lives in M and is different per OUT. Fitted on the 30 rd cells of
        # dense_gemm_surface_card0.csv: right on every cell the engine can actually dispatch (rd is
        # reachable only at M<=128, plus the router at every M) except mlp.gate_up M=128.
        # The cell this fixes: o_proj M=128 was taking bn32 at 57.6 us against 43.3 at bn64 (-25%),
        # the worst cell in the whole A/B at 2.05x rocBLAS.
        rb = -(-M // bm)
        _t32, _t64 = (OUT // 32) * rb, (OUT // 64) * rb
        if OUT % 64 or _t32 < _CUS:
            # Router class: the machine is not full even at the finest tile, so every block counts
            # and halving the grid can only hurt. (OUT=128 is 4 tiles at bn32, 2 at bn64.)
            bn = 32
        elif _t64 >= _CUS or (M * IN >= _RD_BN64_MIN_MN and _t64 >= _RD_BN64_MIN_TILES):
            bn = 64
        else:
            bn = 32
    else:
        bn = 128 if _wide else 64
        bm = 256 if (M >= 512 or (_wide and M >= 192)) else (
            128 if (M >= 256 or (_wide and M >= 128)) else 64)
    # Precedence: explicit argument > env override > derived. No engine call site passes either
    # (all eight go through `minv_linear(x, weight[, bias])`), so the derived tile is what serves;
    # the arguments exist for the sweep harness and for a caller that knows better.
    bn = BN or _BN_OVERRIDE or bn
    bm = block_m or _BLOCK_M_OVERRIDE or bm

    if OUT % bn != 0:
        # ragged OUT: only the LDS kernel masks a partial N-tile (a direct fragment load cannot).
        # Checked against the tile actually chosen, not a module constant — that is the whole point.
        out = _dg.dense_gemm(x2, weight, bm, bn)
    elif OUT <= _SPLITK_MAX_OUT and IN >= _SPLITK_MIN_IN:
        # Split-K reduction order — committed by SHAPE (see the note at the top of this file), so
        # every M on this weight reduces identically. Only the SCHEDULE below reads M, and it may,
        # because the two schedules are bit-identical.
        out = _dg.dense_gemm_rd_sk(_padded(bm), weight, bm, bn, 0,
                                   M < _SPLITK_GRID_MAX_M)[:M]
    elif OUT < _PIPE_MIN_OUT or (M <= _RD_MAX_M and M * OUT <= _RD_MAX_MN):
        # rd's regime: either the N-grid is too narrow for pipe to fill the machine (routers), or the
        # B re-read volume is still small. rd BEATS rocBLAS through most of this band.
        out = _dg.dense_gemm_rd(_padded(bm), weight, bm, bn)[:M]
    else:
        # pipe: B read once per pbm rows, prefetched a K-chunk ahead.
        #   pbm=128 wins M=192..384 (grid stays wide, no wasted M-padding); pbm=256 wins from M>=512,
        #   and already from M>=192 on an LM-head-width OUT, where the N-grid is thousands of tiles
        #   wide so the only remaining lever is active warps per block (8 at pbm=256 vs 4 at 128).
        # pbm=64 additionally wins M<256 once MI=1 removes the register pressure: the tile shrinks
        # but the grid widens, and at M=192 that is the trade that pays (gate_up 45.3/47.4 us at
        # bm64 vs >47.9 at bm128; mlp.down 24.8 vs 25.9; qkv 66.1 best). At M=256 bm128 retakes it
        # (mlp.down 26.2 vs 27.4), which is where this steps up.
        pbm = bm   # chosen above, with the extra wide-OUT step at M>=128
        # MI is the per-warp M-register-blocking factor, and it is the OCCUPANCY knob — the
        # accumulator is acc[MI][NFRAG], i.e. MI*NFRAG*8 VGPRs, which at BN=64/MI=2 is 64 of the
        # kernel's 156 and at BN=128/MI=2 is 128 of 220. MI=1 halves it: measured 156->89 VGPR
        # (9->16 waves/SIMD) at BN=64/PBK=64 and 220->121 (6->12) at BN=128/PBK=64, 0 scratch in
        # every case. It costs no REUSE — B is staged in LDS once per block and read from there by
        # every warp, so halving MI only doubles the per-warp ds_read while block_m = n_warps*MI*16
        # is held constant by doubling n_warps.
        #
        # It wins where the machine is not yet saturated, i.e. the short-M end, and loses to MI=2 at
        # pbm=256 where 8 warps x MI=2 is what fills the block. Best-of-family vs rocBLAS (cold,
        # CUDA-graph-replay device time): gate_up M=192 45.3 us MI=1 vs 47.9 MI=2 (0.82x rocBLAS),
        # mlp.down M=192 24.8 MI=1, M=256 26.2 MI=1; qkv M=192 66.1 MI=1.
        #
        # SAFE TO GATE ON M: every (MI, ADIV, PBK, BN) instantiation issues the identical WMMA
        # sequence into the identical accumulator chain, so they are bit-identical to each other and
        # to lds/rd (verified 0.000e+00 over 36-54 configs x 4 shapes x 4 M). This is the same class
        # of threshold as the pbm one above, NOT the kind the split-K arm would have needed.
        mi = _PIPE_MI if pbm >= 256 else 1
        out = _dg.dense_gemm_pipe(_padded(pbm), weight, pbm, bn, mi, _PIPE_PBK)[:M]
    if bias is not None:
        out = out + bias
    return out.reshape(*orig_shape[:-1], OUT)
