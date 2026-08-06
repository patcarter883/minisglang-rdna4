#!/usr/bin/env python
"""Price every STRUCTURAL variant of tile_select.h's cost model against BOTH swept surfaces. CPU.

    python tools/w4a8_tile_model_variants.py \
        --dense tools/_fixtures/dense_tile_surface.csv \
        --moe   tools/_fixtures/moe_tile_surface.csv

WHY THIS EXISTS. tile_select.h has six fitted numbers and one structural form. "Which constant is
wrong" is the cheap question and it is usually the wrong one: the shipped constants are already at
the optimum of their family on the dense surface (verified below, F0), so a column that fits badly
is evidence about the FORM, not the values. This script makes changing the form as cheap as changing
a constant, and scores every candidate on the same cells.

SCORING -- restricted argmin. The model ranges over a 77-tile lattice; each surface measured ~25.
If a variant is allowed to pick an unmeasured tile, that cell silently drops out of its score, so a
variant that picks unmeasured tiles more often looks better for no reason. Every variant here picks
the argmin over THE TILES THAT CELL MEASURED, so all variants are scored on 100% of cells against
the same per-cell oracle and the geomeans are comparable. What this can falsify is RANKING; what it
cannot see is extrapolation, which is what the live re-time in w4a8_*_tile_verify.py is for.

THE TWO SURFACES DISAGREE ABOUT THE LATENCY TERM, which is the whole finding:
  * dense wants LAT=32 with no ILP credit (any ILP/split/rounds-softening variant costs more on the
    195-cell g=32 column than it buys on the 15-cell g=128 one);
  * MoE at M<=32 is 2-3x off with LAT=32, and the direction is always the same -- the model takes a
    TALL block_m for the occupancy and pays for it in masked padding rows.
Both are the same defect seen from two sides: `1 + LAT/OCC` is a steady-state latency-hiding law,
and it is applied to launches that are 3 waves deep. See ROUNDS_FLOOR below.
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict

ARMS = ("prefill_wmma", "prefill_wmma_ashuffle", "prefill_wmma:smallm_off")
LDS_BUDGET = 65536
LINE = 128            # gfx1201 cache line, bytes
WAVES_PER_CU = 32
VGPR_PER_SIMD = 1536
WAVES_PER_SIMD_MAX = 16
HW_BLOCKS_PER_CU = 4
MOE_MAX_WARPS = 8


def cd(a, b):
    return -(-a // b)


def gm(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def core(bm, bn, warps_n, g, row_blocks, n_blocks, z_blocks, k_groups, lds, shuffled, cu, p,
         real_rows=None):
    """Byte-for-byte the arithmetic of tile_select.h::core_terms, with the structural knobs of the
    variant grid spliced in at the three points that are being questioned (ROUNDS, LATENCY, and the
    REAL-ROW fraction).

    `real_rows` is how many rows of the launch hold DATA: M for dense, the routed row total for MoE.
    `row_blocks*bm - real_rows` is masked padding, and the two kernels treat it the same way -- the
    staging loop and the WMMAs are NOT predicated on a row being live, so the padding ISSUES at full
    price, but `stage_act_word(x, M, ...)` / `if (a_valid)` elide its global read and the epilogue
    `if (abs_m >= M) continue` / `if (offs_token >= num_valid_tokens) continue` elide its store. A
    masked M-warp therefore costs issue slots and earns NO latency to hide."""
    if lds > LDS_BUDGET or bm % 16 or bn % 16 or warps_n < 1:
        return None
    nwarps_m, nfrag = bm // 16, bn // 16
    if nfrag % warps_n:
        return None
    nfrag_w = nfrag // warps_n
    nwarps = nwarps_m * warps_n
    threads = 32 * nwarps
    if g <= 0 or g % 16:
        return None
    k_steps = g // 16
    vgpr = 24 + 20 * nfrag_w
    wps = min(WAVES_PER_SIMD_MAX, VGPR_PER_SIMD // vgpr)
    bpc = min((wps * 2) // nwarps, LDS_BUDGET // max(lds, 1), HW_BLOCKS_PER_CU)
    if bpc <= 0:
        return None
    wgs = row_blocks * n_blocks * z_blocks
    if wgs <= 0:
        return None
    rounds = cd(wgs, cu)
    live = min(bpc, rounds)
    # ---- THE REAL-ROW TERM, the one under test -------------------------------------------------
    # A block covers `rpb` real rows on average; the WMMA M axis is 16 rows wide, so only
    # ceil(rpb/16) of the block's BM/16 M-warps hold any data at all. The rest issue, but they load
    # nothing and store nothing, so they generate no latency for co-residency to hide. Counting them
    # in OCC is the model paying a tall tile for waves that are pure waste -- which is exactly
    # backwards, and it is the half that actively rewards the wrong tile.
    rr = p.get("rr", "off")
    rpb = (real_rows / row_blocks) if real_rows else float(bm)
    live_m = min(nwarps_m, max(1, cd(int(math.ceil(rpb)), 16)))
    phi = live_m / nwarps_m
    eff_nwarps = (live_m * warps_n) if rr in ("occ", "both") else nwarps
    occ = min(WAVES_PER_CU, live * eff_nwarps)
    a_it = k_steps if shuffled else cd(bm * g // 4, threads)
    b_it = cd(bn * (g // 8), threads)
    # ---- SCALE-LINE TRAFFIC, the term that separates BN at high WARPS_N ---------------------------
    # MEASURED, by ablation, not hypothesised: at lm_head g=32 M=1 the 32x64x4 / 32x128x4 gap is
    # 3.71x with everything on, 4.17x with the WMMAs off, 4.77x with the LDS writes off, 4.17x with
    # the LDS reloads off -- and 1.17x with the GLOBAL READS off, 1.61x with ONLY the per-group
    # w_scale read off. The gap lives entirely in the per-group weight-scale read.
    #
    # WHY it is a BN term while every other term washes. Both kernels read
    #     w_scales[abs_n * num_k_groups + g]
    # once per weight row per k-group. Consecutive rows are `2*k_groups` BYTES apart, so as soon as
    # that stride reaches a 128 B cache line EVERY row is a SEPARATE line and the read pulls 128 B
    # to use 2. One such line spans 64 consecutive k-groups, so the reuse is there to be had -- but
    # only if the line SURVIVES, and what the block must retain to collect it is
    #     blocks_per_cu * BN lines * 128 B,
    # in which BN is the only tile term. Doubling BN halves the block count and doubles the lines
    # each block retains: the ISSUE washes exactly (which is why the model saw a wash) and the
    # RETENTION does not wash at all -- it doubles the footprint against a fixed cache. Past the
    # cache the access is cyclic, and cyclic LRU past capacity collects NOTHING, so this is a cliff
    # and not a slope.
    #
    # The second gate is where a miss lands. If the shape's WHOLE live scale footprint (N lines,
    # independent of BN) still fits L2, the reuse is collected in L2 instead and the cliff never
    # fires. That is measured too: sweeping N at g=32 with everything else fixed, the 64-vs-128 gap
    # is 1.48x / 1.45x / 1.31x at N = 4096 / 8192 / 16384 and 3.55x / 4.25x / 3.98x at N = 32768 /
    # 65536 / 131072 -- a step exactly where N*128 crosses 4 MB.
    #
    # Both sizes are HARDWARE, taken from the ablation's own knees, not fitted to the surface.
    scale_it = 0.0
    if p.get("sline", "off") != "off":
        stride = 2.0 * k_groups                       # bytes between adjacent rows' group scales
        lines = float(bn) if stride >= LINE else max(1.0, bn * stride / LINE)
        span = min(float(k_groups), LINE / 2.0)       # k-groups one retained line serves
        # ---- THE SWIZZLE DIVISOR: what the CU must retain DEPENDS ON row_blocks -----------------
        # MEASURED by ablation (tools/_fixtures/rb1_diag/stage.txt), lag.gate_up tp1 32x64x4,
        # swiz=1, the shipped launch:
        #     M=32 (rb==1)  full 323.1 us   no_wscale 113.0 us  ->  the w_scale read is 65% of it
        #     M=48 (rb==2)  full 136.5 us   no_wscale 116.0 us  ->  the w_scale read is 15% of it
        # 1.5x the work, 2.4x FASTER, and 210 of the 323 us that vanish are the per-group w_scale
        # global read. No other bit moves it: no_wmma -2%, no_ldsread -1%, no_ldswrite -22%.
        # The carrier is the SAME scale line the BN term already prices -- what is new is that its
        # RETENTION is a function of row_blocks.
        #
        # WHY. swiz=1 puts the row-block axis on grid.x, so the `row_blocks` workgroups that share
        # one block_n slab are dispatched CONSECUTIVELY and share its scale lines. At row_blocks==1
        # grid.x==1: the swizzle is a structural NO-OP, and every resident workgroup retains its own
        # slab. So the retained set is bpc/row_blocks slabs, not bpc.
        # The same ablation proves the swizzle is the carrier from the other side: at rb==2, turning
        # the swizzle OFF puts the cliff straight back -- 493.9 us full / 157.4 no_wscale at swiz=0
        # against 136.5 / 116.0 at swiz=1. And at rb==1 the two orders are IDENTICAL (320.2 vs
        # 323.1), which is what "the swizzle is a no-op there" means, and which independently
        # falsifies a gridDim.x==1 dispatch/shader-engine story.
        rb_div = max(1, row_blocks) if (p.get("sline") == "rb" and not shuffled) else 1
        foot = cd(bpc, rb_div) * lines * LINE          # bytes the CU must retain
        # ---- WHAT IS ACTUALLY LIVE IN L2, and the ROW_BLOCKS==1 CLIFF it explains --------------
        # `n_blocks * lines * LINE` is the shape's TOTAL scale footprint, and a total is not what a
        # cache has to hold: the cache has to hold what is CONCURRENTLY LIVE. The launch has
        # `cu * blocks_per_cu` workgroups resident at once, and under the dense swizzle (swiz=1,
        # which is what ships) the ROW-BLOCK axis is grid.x -- the FAST axis -- so the `row_blocks`
        # workgroups that share one block_n weight slab are dispatched CONSECUTIVELY and are
        # co-resident. The number of DISTINCT slabs whose scale lines must be retained at once is
        # therefore ceil(resident / row_blocks), not n_blocks.
        #
        # That single correction is the whole ROW_BLOCKS == 1 cliff, and it is DERIVED, not fitted:
        # at row_blocks == 1 the swizzle is a NO-OP (grid.x == 1, nothing to interleave), every
        # resident workgroup streams its own slab, and the live set is the full `resident` count.
        # At row_blocks == 2 it HALVES, and where that halving steps back across L2 the cliff
        # switches off -- which is exactly the discontinuity the surface shows.
        #
        # MEASURED, on the recorded ablation fixtures (tools/_fixtures/bn_diag/), lm_head g=32:
        #   32x128x4 (BN=128, 128 lines x 128 B = 16 KB/slab, bpc=4 -> resident 256):
        #       rb==1  256 slabs x 16 KB = 4.19 MB > L2 4 MB -> MISS   2481-2505 us
        #       rb==2  128 slabs x 16 KB = 2.10 MB < L2      -> HELD   1181-1272 us
        #                                                    (2x the work, 2.0x FASTER)
        #     and the swiz A/B confirms the carrier: at rb==2, swiz=1 1180.7 vs swiz=0 4637.5
        #     (3.9x) -- turn the swizzle off and the rb==2 point returns to the un-reused price,
        #     2x the rb==1 point, exactly linear.
        #   32x64x4 (BN=64, 8 KB/slab): rb==1 is 256 x 8 KB = 2.10 MB < L2 -> HELD at EVERY rb,
        #     so this tile shows NO cliff (622 -> 1368 us, plain 2x) and the swizzle buys only
        #     1.19x. The term correctly stays silent there.
        # The falsified alternative is recorded too: a grid-DIMENSION degeneracy at gridDim.x==1
        # (all workgroups landing on one shader engine) predicts swiz=0 and swiz=1 differ at
        # rb==1. They do not -- 613.1 vs 624.9 and 206.7 vs 212.6 us, i.e. identical -- because at
        # rb==1 BOTH orders are the un-swizzled one. The dispatcher is innocent; the cache is not.
        if p.get("sline") == "rb" and not shuffled:
            resident = cu * bpc
            slabs = min(float(n_blocks), float(cd(resident, max(1, row_blocks))))
        else:
            slabs = float(n_blocks)
        live_ws = slabs * lines * LINE                # concurrently-live scale footprint
        # `<=` vs `<` is not a taste question here: the access is CYCLIC (the block re-walks the
        # same `lines` lines once per k-group), and cyclic LRU over a working set EQUAL to capacity
        # collects nothing -- the line it needs next is always the one just evicted. So a footprint
        # that exactly equals L0 is a MISS, not a hit. That distinction decides the shape this whole
        # investigation is about: lag.gate_up tp1 at 32x64x4 has foot = 4 x 64 lines x 128 B =
        # 32768 B = L0 EXACTLY, and it is measured missing (the w_scale read is 65% of the kernel).
        l0_held = (foot < p["L0"]) if p.get("sline") == "rb" else (foot <= p["L0"])
        held = l0_held or (live_ws < p["L2"] and not p.get("no_l2_rescue"))
        # A HELD line is a cache hit and adds no traffic, so the term is ZERO there -- it is a pure
        # penalty that fires only where the physics says the cliff is, and can never nudge a pick on
        # a shape that has no cliff. Past the cliff the block re-fetches a full LINE per row per
        # k-group, for the 2 bytes it wanted.
        per_block = 0.0 if held else lines * LINE
        # into the SAME units as a_it/b_it: dwords per thread per k-group, so it rides the existing
        # C_BSTAGE and introduces no new fitted constant.
        scale_it = per_block / 4.0 / threads
    lds_it = k_steps * (nfrag_w if shuffled else nfrag_w + 1)
    wmma_it = k_steps * nfrag_w
    if rr in ("work", "both"):
        # the other half of the question: charge the A-gather and the WMMAs only for the M-warps
        # that hold data. NOTE this is NOT what the ISA does -- the masked warps issue -- so it is
        # here to be falsified, not assumed.
        a_it *= phi
        wmma_it *= phi
    # cL/cW are PER STAGING POLICY in tile_select.h (C_LDSRD_STAGED/C_WMMA_STAGED = 0.25 vs
    # C_LDSRD_SHUF/C_WMMA_SHUF = 1.0): the LdsStaged k-loop prefetches a_nx/b_nx so its ds_reads sit
    # off the WMMA dependency chain, the Shuffled one reads b_cur immediately before the mma that
    # consumes it. Scoring both surfaces at the staged constants -- which this harness did -- prices
    # a model that does not ship, and under-charges the MoE side's per-warp WMMA/ds_read by 4x,
    # which is precisely the term that grows with BM.
    cL = p["cL_SHUF"] if (shuffled and p.get("cL_SHUF") is not None) else p["cL"]
    cW = p["cW_SHUF"] if (shuffled and p.get("cW_SHUF") is not None) else p["cW"]
    # Split per_thread into the two halves the LATENCY term treats differently:
    #   pt_chain -- the k-loop's own dependency chain, walked by EVERY warp: the A fragment, the
    #               NFRAG_W ds_reads and the NFRAG_W mmas. It is proportional to NFRAG_W and it is
    #               what a warp waits on.
    #   pt_coop  -- `stage_b`, which the WHOLE workgroup cooperates on before a barrier, so it
    #               shrinks as THREADS grows and is a bandwidth cost, not a latency chain.
    pt_chain = p["cA"] * a_it + cL * lds_it + cW * wmma_it
    pt = pt_chain + p["cB"] * (b_it + scale_it)
    # ---- LATENCY, the term under test ----------------------------------------------------------
    # `1 + LAT/OCC` says a CU with OCC resident waves hides LAT units of latency. That is a
    # STEADY-STATE law and it is being applied to launches that are a handful of waves deep, where
    # a workgroup's latency is paid once at the head of the pipe and never amortised again. DEPTH
    # caps how much of LAT occupancy is allowed to hide: a launch of ROUNDS rounds cannot amortise
    # more latency than it has rounds to amortise it over.
    # LAT follows the ACTIVATION-STAGING POLICY in the shipped header (LAT_LDS_STAGED=32,
    # LAT_SHUFFLED=8), so the harness must too -- scoring the MoE surface at the dense constant
    # measures a model that does not ship. `LAT_SHUF=None` collapses to one constant for the
    # single-LAT sweeps below.
    L = p["LAT"] if (not shuffled or p.get("LAT_SHUF") is None) else p["LAT_SHUF"]
    if p["lat"] == "const":
        div = occ
    elif p["lat"] == "depth":
        div = max(min(float(occ), L * min(1.0, rounds / p["ROUNDS_FLOOR"]) + 1.0), 1.0)
    elif p["lat"] == "ilp":
        div = occ * (1.0 + p["alpha"] * (nfrag_w - 1))
    elif p["lat"] == "cap":
        div = max(occ, p["OCC_FLOOR"])
    elif p["lat"] in ("wg", "wgmix"):
        # ---- THE BARRIER IS A WHOLE-WORKGROUP STALL, so WARPS INSIDE IT CANNOT COVER IT ----------
        # This is not a new hypothesis; it is what the shipped header ALREADY SAYS the term means:
        #   "Every wave in the workgroup stops at that barrier together, so nothing inside the
        #    workgroup hides the staging latency and ONLY CO-RESIDENCY CAN. That is exactly the
        #    regime LAT/OCC describes."
        # But the code then divides by `occ = live * nwarps`, and `nwarps` is precisely the count of
        # warps INSIDE the workgroup -- the ones that sentence says cannot cover it. The covering
        # resource is `live`: the number of OTHER RESIDENT WORKGROUPS on the CU.
        #
        # That mis-statement is a SIGN FLIP on exactly the axis that regresses. WARPS_N multiplies
        # `nwarps` while adding not one resident workgroup, and it actively REDUCES co-residency
        # (bpc = wps*2/nwarps), so the shipped denominator hands a WARPS_N=4 tile a ~4x latency
        # credit for a change that makes the covering resource WORSE. Hence "short-rounds launches
        # preferring a tall tile": where `rounds` clamps `live` low, the spurious nwarps factor is
        # the only thing left moving, and it points the wrong way.
        #
        # ---- VERDICT: FALSIFIED. Kept, because the next agent will otherwise derive it again. ----
        # Scored on the 300-cell card-0 dense surface against the shipped WN={1} chooser:
        #     shipped + scale-line, WN={1,2,4}   dense gm 1.0753  worst 2.48  REGRESSED 30
        #     wg    div=live        LAT 2..12    dense gm 1.54-1.63  worst 4.26  REGRESSED 196-233
        #     wgmix f=0.5  LAT=16               dense gm 1.1004  worst 2.48  REGRESSED 37
        #     wgmix f=0.75 LAT=8                dense gm 1.1401  worst 2.35  REGRESSED 36
        # Every variant is WORSE than what ships. The reading: `occ` is doing DOUBLE DUTY. It stands
        # for the barrier's covering resource (co-resident workgroups, which is what the header
        # describes) AND for memory-level parallelism (independent in-flight loads, which extra
        # warps genuinely do provide). Removing the nwarps factor removes both, and the second one
        # is real. A correct term has to SPLIT the stall, not re-point the whole denominator.
        #
        # What the surface actually says, and what the next attempt should start from: the anomaly
        # is a ROW_BLOCKS == 1 CLIFF, not a WARPS_N effect. Measuring us(rb==1) against that same
        # tile's own per-row-block trend us(rb==2)/2 over the whole surface:
        #     WARPS_N=1  n=452  geomean 1.577   WARPS_N=2  n=160  geomean 1.516
        #     WARPS_N=4  n=80   geomean 1.646   worst 5.04x (lag.gate_up tp1 32x64x4, M=32->48:
        #                                       354.7us -> 140.6us, i.e. 2.5x FASTER with 2x the work)
        # It is present at EVERY WARPS_N, so it is not a WN term -- but it is a term the model does
        # not have at all, and it is WN-COUPLED IN EFFECT because a tall tile stays at rb==1 over a
        # much longer M range than a short one. That coupling is the "short-rounds launches
        # preferring a tall tile" signature, made quantitative.
        #
        # DERIVED, not fitted: the structure follows from the barrier, and LAT stays ONE constant.
        # Its SCALE necessarily changes with the denominator's units (dividing by ~4-8x less), so
        # the sweep re-reads it on its plateau -- that is a re-scale of an existing constant, not a
        # new one.
        #   "wg"    -- the whole stall is barrier-bound: div = live.
        #   "wgmix" -- only WG_F of the stall is the barrier; the rest is per-wave memory latency
        #              that co-resident WAVES do hide. div = live * (1 + (nwarps-1)*(1-WG_F)).
        if p["lat"] == "wg":
            div = max(float(live), 1.0)
        else:
            f = p.get("WG_F", 1.0)
            div = max(float(live) * (1.0 + (eff_nwarps - 1) * (1.0 - f)), 1.0)
    else:
        raise SystemExit(f"unknown lat mode {p['lat']}")
    # `stall` = "pt" reproduces the shipped `ISSUE * (1 + LAT/OCC)` exactly. "chain" says the
    # exposed stall is the k-loop DEPENDENCY CHAIN, not the cooperative staging that rides in front
    # of a barrier -- i.e. a tile that spreads the same B stage over twice the threads does not
    # thereby halve its latency.
    stall_pt = pt if p.get("stall", "pt") == "pt" else pt_chain
    issue = rounds * k_groups * threads * pt
    stall = rounds * k_groups * threads * stall_pt * L / div
    return issue + stall


def dense_cost(M, N, K, g, bm, bn, wn, cu, p):
    lds = (bm + bn) * (g + 8) + 4 * bn
    # dense real rows = M exactly; the padding is ceil(M/BM)*BM - M, paid once on the last row-block
    return core(bm, bn, wn, g, cd(M, bm), cd(N, bn), 1, K / g, lds, False, cu, p, real_rows=M)


def moe_warps_n(bm, bn):
    nwm, nfrag = bm // 16, bn // 16
    if nwm < 1 or nfrag < 1:
        return 0
    wn = min(MOE_MAX_WARPS // nwm, nfrag)
    wn = 4 if wn >= 4 else (2 if wn >= 2 else 1)
    while wn > 1 and nfrag % wn:
        wn //= 2
    return wn


def moe_padded_rows(rows, E, bm):
    rows = max(rows, 1)
    hit = max(1, min(E, rows))
    return hit * cd(cd(rows, hit), bm) * bm


def moe_cost(rows, E, N, K, g, bm, bn, cu, p, gtile=4):
    wn = moe_warps_n(bm, bn)
    if wn < 1 or bm // 16 > MOE_MAX_WARPS:
        return None
    kg = K // g
    if kg <= 0:
        return None
    gt = max(1, min(gtile, kg))
    per = bn * (g + 8)
    while gt > 1 and per * gt > 40960:
        gt -= 1
    rb = moe_padded_rows(rows, E, bm) // bm
    # MoE real rows = the ROUTED row total; `moe_align` pads each expert up to bm, so at decode
    # (rows < E) every one of the rb blocks holds ONE real row no matter how tall bm is.
    return core(bm, bn, wn, g, rb, cd(N, bn), 1, kg, per * gt, True, cu, p, real_rows=rows)


def tile_of(cand):
    """A measured candidate is "BMxBN" (WARPS_N=1) or "BMxBNxWN". Returns (bm, bn, wn) or None."""
    parts = cand.split("x")
    if len(parts) not in (2, 3):
        return None
    try:
        bm, bn = int(parts[0]), int(parts[1])
        wn = int(parts[2]) if len(parts) == 3 else 1
    except ValueError:
        return None
    return bm, bn, wn


def load(path, moe, paths=()):
    """Load one or more surface CSVs into {cell: {cand: us}}.

    Rows whose `cand` is an ARM (a whole-kernel alternative, not a tile) are dropped, as are rows
    that do not parse -- a fixture written by two concurrent processes can carry a torn line, and a
    silently mis-parsed row would enter the oracle as a fake best.
    """
    t = defaultdict(dict)
    for p in (path, *paths):
        if not p:
            continue
        for r in csv.DictReader(open(p)):
            if r["cand"] in ARMS or tile_of(r["cand"]) is None:
                continue
            try:
                if moe:
                    k = (r["name"], int(r["E"]), int(r["top_k"]), int(r["hidden"]),
                         int(r["inter"]), int(r["g"]), int(r["M"]))
                else:
                    k = (r["name"], int(r["K"]), int(r["N"]), int(r["g"]), int(r["M"]))
                us = float(r["us"])
            except (ValueError, TypeError, KeyError):
                continue
            if us <= 0.0:
                continue
            t[k][r["cand"]] = us
    return {k: d for k, d in t.items() if len(d) >= 4}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense", default="tools/_fixtures/dense_tile_surface_card0.csv")
    ap.add_argument("--dense-extra", default="", help="comma list of extra dense surface CSVs "
                    "(the widened g=128 column and the WARPS_N re-sweep are separate files)")
    ap.add_argument("--moe", default="tools/_fixtures/moe_tile_surface_card0.csv")
    ap.add_argument("--cu", type=int, default=64)
    args = ap.parse_args()

    extra = tuple(x for x in args.dense_extra.split(",") if x)
    D = load(args.dense, moe=False, paths=extra)
    Mo = load(args.moe, moe=True)
    print(f"dense cells {len(D)}   moe cells {len(Mo)}   CU={args.cu}")

    def score(p):
        res = {}
        for tag, cells, isd in (("dense", D, True), ("moe", Mo, False)):
            per_g, allr, small, big = defaultdict(list), [], [], []
            lmh, lmh_sm = [], []          # the LM head is its own surface: N=131072 per rank
            for k, d in cells.items():
                best, bc = None, None
                for c in d:
                    bm, bn, wn = tile_of(c)
                    if isd:
                        name, K, N, g, M = k
                        # WARPS_N comes from the CANDIDATE now: the surface carries "BMxBNxWN"
                        # tiles since the WN axis was swept, and scoring one at wn=1 would price a
                        # launch that never happened.
                        cst = dense_cost(M, N, K, g, bm, bn, wn, args.cu, p)
                    else:
                        name, E, tk, hid, inter, g, M = k
                        # score on gemm1's shape; gemm2 rides the same block_m and BN sweep
                        cst = moe_cost(M * tk, E, 2 * inter, hid, g, bm, bn, args.cu, p)
                    if cst is None:
                        continue
                    if best is None or cst < best:
                        best, bc = cst, c
                if bc is None:
                    continue
                r = d[bc] / min(d.values())
                allr.append(r)
                per_g[k[3] if isd else k[5]].append(r)
                (small if k[-1] <= 32 else big).append(r)
                if isd and k[2] >= 100000:          # k = (name, K, N, g, M)
                    lmh.append(r)
                    if k[4] <= 32:
                        lmh_sm.append(r)
            res[tag] = dict(gm=gm(allr), worst=max(allr) if allr else 0,
                            g32=gm(per_g.get(32, [])), g128=gm(per_g.get(128, [])),
                            small=gm(small), big=gm(big), n=len(allr),
                            lmh=gm(lmh), lmh_sm=gm(lmh_sm))
        return res

    def show(tag, p):
        s = score(p)
        d, m = s["dense"], s["moe"]
        print(f"{tag:<40} DENSE gm={d['gm']:.4f} w={d['worst']:.2f} "
              f"g32={d['g32']:.3f} g128={d['g128']:.3f} lmh={d['lmh']:.3f}/{d['lmh_sm']:.3f} | "
              f"MOE gm={m['gm']:.4f} w={m['worst']:.2f} "
              f"M<=32={m['small']:.3f} M>32={m['big']:.3f}")
        return s

    base = dict(cA=1.0, cB=16.0, cL=0.25, cW=0.25, LAT=32.0, alpha=0.25,
                lat="const", OCC_FLOOR=8, ROUNDS_FLOOR=8, rr="off",
                cL_SHUF=None, cW_SHUF=None, LAT_SHUF=None)

    # The model as it actually SHIPS in tile_select.h: every constant that follows the activation-
    # staging policy takes its shuffled value on the MoE surface.
    ship = dict(base, LAT=32.0, LAT_SHUF=8.0, cL_SHUF=1.0, cW_SHUF=1.0)

    print("\n=== SHIPPED (per-policy LAT: staged 32 / shuffled 8, rr=off) ===")
    show("SHIPPED", ship)

    # ------------------------------------------------------------------------------------------
    # THE REAL-ROW TERM. Three surfaces (dense g=128, the LM head at M<=32, MoE at M<=32) were all
    # off in the same direction and the model has no term for the fraction of a tile's rows that
    # hold data. `rr=occ` denies masked M-warps their occupancy credit; `rr=work` instead discounts
    # the issue cost; `rr=both` does both. Only one of them is what the ISA does.
    print("\n=== REAL-ROW term, on the SHIPPED constants (32/8) ===")
    for mode in ("off", "occ", "work", "both"):
        show(f"rr={mode}", dict(ship, rr=mode))

    print("\n=== stall=chain: the exposed stall is the k-loop chain, not the cooperative stage ===")
    for mode in ("off", "occ", "work", "both"):
        show(f"stall=chain rr={mode}", dict(ship, stall="chain", rr=mode))

    print("\n=== stall=chain, refitting the two per-policy LATENCY constants ===")
    best = []
    for st in ("pt", "chain"):
        for mode in ("off", "occ", "work", "both"):
            for Ld in (4, 8, 16, 24, 32, 48, 64, 96, 128):
                for Ls in (1, 2, 4, 8, 16, 32, 64):
                    p = dict(ship, stall=st, rr=mode, LAT=float(Ld), LAT_SHUF=float(Ls))
                    s = score(p)
                    best.append((max(s["dense"]["gm"], s["moe"]["gm"]), s, p))
    best.sort(key=lambda r: r[0])
    for mx, s, p in best[:20]:
        d, m = s["dense"], s["moe"]
        print(f"  max={mx:.4f} stall={p['stall']:<5} rr={p['rr']:<4} LATd={p['LAT']:<5g} "
              f"LATs={p['LAT_SHUF']:<4g} | D gm={d['gm']:.4f}/w{d['worst']:.2f} "
              f"g32={d['g32']:.3f} g128={d['g128']:.3f} lmh={d['lmh']:.3f}/{d['lmh_sm']:.3f}"
              f" | M gm={m['gm']:.4f}/w{m['worst']:.2f} sm={m['small']:.3f} big={m['big']:.3f}")

    print("\n=== the LATENCY constant alone ===")
    for L in (4, 8, 12, 16, 24, 32, 48):
        show(f"const LAT={L}", dict(base, LAT=L))

    print("\n=== OCC_FLOOR: cap how much low occupancy is punished ===")
    for of in (4, 6, 8, 12, 16):
        for L in (16, 32):
            show(f"cap OCC_FLOOR={of} LAT={L}", dict(base, lat="cap", OCC_FLOOR=of, LAT=L))

    print("\n=== DEPTH: a shallow launch cannot amortise steady-state latency ===")
    for rf in (2, 4, 8, 16, 32):
        for L in (16, 32):
            show(f"depth ROUNDS_FLOOR={rf} LAT={L}", dict(base, lat="depth", ROUNDS_FLOOR=rf, LAT=L))

    print("\n=== ILP credit ===")
    for a in (0.1, 0.25, 0.5, 1.0):
        show(f"ilp alpha={a}", dict(base, lat="ilp", alpha=a))

    print("\n=== joint grid, ranked by max(dense gm, moe gm) ===")
    rows = []
    for lat in ("const", "cap", "depth", "ilp"):
        for L in (4, 8, 12, 16, 24, 32):
            for cB in (8.0, 16.0, 24.0):
                for cA in (0.5, 1.0, 2.0, 4.0):
                    extras = ([{}] if lat == "const"
                              else [{"OCC_FLOOR": v} for v in (4, 6, 8, 12, 16)] if lat == "cap"
                              else [{"ROUNDS_FLOOR": v} for v in (2, 4, 8, 16, 32)] if lat == "depth"
                              else [{"alpha": v} for v in (0.1, 0.25, 0.5, 1.0)])
                    for ex in extras:
                        p = dict(base, lat=lat, LAT=L, cB=cB, cA=cA, **ex)
                        s = score(p)
                        rows.append((max(s["dense"]["gm"], s["moe"]["gm"]), s, p))
    rows.sort(key=lambda r: r[0])
    for mx, s, p in rows[:15]:
        d, m = s["dense"], s["moe"]
        ex = {k: p[k] for k in ("OCC_FLOOR", "ROUNDS_FLOOR", "alpha") if p["lat"] in
              {"cap": ("OCC_FLOOR",), "depth": ("ROUNDS_FLOOR",), "ilp": ("alpha",),
               "const": ()}.get(p["lat"], ())}
        print(f"  max={mx:.4f}  D gm={d['gm']:.4f}/w{d['worst']:.2f} "
              f"g32={d['g32']:.3f} g128={d['g128']:.3f} | "
              f"M gm={m['gm']:.4f}/w{m['worst']:.2f} sm={m['small']:.3f} big={m['big']:.3f}"
              f"   lat={p['lat']} LAT={p['LAT']} cA={p['cA']} cB={p['cB']} {ex}")

    # ==============================================================================================
    # THE WN_SET QUESTION -- can the chooser be allowed WARPS_N > 1 WITHOUT regressing a cell?
    # ==============================================================================================
    # Scored the way the chooser actually decides: argmin over the tiles the chooser is ALLOWED to
    # pick (WN restricted to wn_set) intersected with the tiles the cell measured, then compared to
    # that cell's measured oracle over ALL tiles. Restricting the argmin to measured tiles is what
    # keeps the variants comparable -- an unrestricted argmin silently changes which cells score.
    # `REGRESSED` is the only number that decides shippability: a cell that the wn=1 chooser served
    # well and the widened one serves worse is the exact failure this chooser exists to prevent.
    def chooser(p, wn_set, cells, isd):
        picks = {}
        for k, d in cells.items():
            best = bc = None
            costs = {}
            for c in d:
                bm, bn, wn = tile_of(c)
                if wn not in wn_set:
                    continue
                if isd:
                    name, K, N, g, M = k
                    cst = dense_cost(M, N, K, g, bm, bn, wn, args.cu, p)
                else:
                    name, E, tk, hid, inter, g, M = k
                    cst = moe_cost(M * tk, E, 2 * inter, hid, g, bm, bn, args.cu, p)
                if cst is None:
                    continue
                costs[c] = cst
                if best is None or cst < best:
                    best, bc = cst, c
            # ---- THE NEAR-TIE BAND (tile_select.h::NEAR_TIE_BAND) ------------------------------
            # An argmin acts on any margin, however small; the model's measured PAIRWISE ranking
            # error is 1.10 (below a predicted 1.10x it calls the pair right 45-58% of the time --
            # a coin flip -- and the tile it prefers is measured 0.96-1.05x the one it rejects). So
            # inside that band the chooser keeps the INCUMBENT tile 256x128, the one the dense core
            # was hard-wired to and tuned around, instead of coin-flipping. `tie_break` is the
            # incumbent tile, `tie_band` the multiplier; absent, this is the plain argmin.
            tb = p.get("tie_break")
            if isd and tb and bc is not None and tile_of(bc)[:2] != tb:
                inc = f"{tb[0]}x{tb[1]}"
                if inc in costs and costs[inc] <= best * p.get("tie_band", 1.0):
                    bc = inc
            if bc is not None:
                picks[k] = (bc, d[bc] / min(d.values()))
        return picks

    print("\n" + "=" * 110)
    print("WN_SET -- widening the chooser's WARPS_N axis, with and without the SCALE-LINE term")
    print("=" * 110)
    L0L2 = dict(L0=32768, L2=4 << 20)
    variants = [
        ("shipped model, WN={1}          ", dict(ship), (1,)),
        ("shipped model, WN={1,2,4}      ", dict(ship), (1, 2, 4)),
        ("+ scale-line term, WN={1}      ", dict(ship, sline="on", **L0L2), (1,)),
        ("+ scale-line term, WN={1,2}    ", dict(ship, sline="on", **L0L2), (1, 2)),
        ("+ scale-line term, WN={1,2,4}  ", dict(ship, sline="on", **L0L2), (1, 2, 4)),
        # ---- the SAME term, with the live set counted CONCURRENTLY (the rb==1 cliff) -----------
        # No new constant: L0/L2 are unchanged and `resident/row_blocks` is the swizzle the header
        # already documents. This is the only change between "on" and "rb".
        ("+ rb-swizzle sline, WN={1}     ", dict(ship, sline="rb", **L0L2), (1,)),
        ("+ rb-swizzle sline, WN={1,2}   ", dict(ship, sline="rb", **L0L2), (1, 2)),
        ("+ rb-swizzle sline, WN={1,2,4} ", dict(ship, sline="rb", **L0L2), (1, 2, 4)),
        ("+ rb-swizzle, no L2 rescue {1} ", dict(ship, sline="rb", no_l2_rescue=True, **L0L2), (1,)),
        ("+ rb-swizzle, no L2 rescue{1,2}", dict(ship, sline="rb", no_l2_rescue=True, **L0L2), (1, 2)),
        ("+ rb-swizzle, no L2 resc{1,2,4}", dict(ship, sline="rb", no_l2_rescue=True, **L0L2),
         (1, 2, 4)),
    ]
    ref = None
    for tag, p, wns in variants:
        line = [f"{tag}"]
        for label, cells, isd in (("DENSE", D, True), ("MOE", Mo, False)):
            pk = chooser(p, wns, cells, isd)
            rs = [r for _, r in pk.values()]
            line.append(f"{label} gm={gm(rs):.4f} w={max(rs):.2f} n={len(rs)}")
        print("  ".join(line))
        if ref is None:
            ref = {k: v[1] for k, v in chooser(p, wns, D, True).items()}
            refm = {k: v[1] for k, v in chooser(p, wns, Mo, False).items()}
            continue
        for label, cells, isd, base in (("dense", D, True, ref), ("moe", Mo, False, refm)):
            pk = chooser(p, wns, cells, isd)
            bad = sorted(((v[1] / base[k], k, v[0]) for k, v in pk.items()
                          if k in base and v[1] > base[k] * 1.02), reverse=True)
            if bad:
                print(f"      REGRESSED {len(bad)} {label} cells vs the shipped WN={{1}} chooser; "
                      f"worst {bad[0][0]:.2f}x")
                for f, k, c in bad[:8]:
                    print(f"         {f:5.2f}x  {k}  picked {c}")
            else:
                print(f"      REGRESSED 0 {label} cells vs the shipped WN={{1}} chooser")

    # ==============================================================================================
    # THE OCCUPANCY TERM: what actually covers a WHOLE-WORKGROUP BARRIER STALL
    # ==============================================================================================
    # Baseline for "REGRESSED" is the SAME reference the block above used: the shipped model
    # restricted to WN={1}, i.e. what ships today. A widened chooser is shippable only if it
    # regresses nothing against that.
    print("\n" + "=" * 110)
    print("OCC DENOMINATOR -- `live*nwarps` (shipped) vs `live` (co-resident WORKGROUPS, derived)")
    print("=" * 110)
    print("The shipped header already states the physics: the LdsStaged core hits __syncthreads()")
    print("once per K-group and 'nothing inside the workgroup hides the staging latency, only")
    print("co-residency can'. The code then divides by live*NWARPS -- the warps inside it. WARPS_N")
    print("multiplies that numerator while REDUCING bpc=wps*2/nwarps, so the shipped term credits")
    print("WARPS_N ~4x for making the covering resource worse. Below, the denominator is `live`.")
    print("LAT is re-read on its plateau because the denominator's UNITS changed; it stays ONE")
    print("constant and no new one is introduced.\n")
    occ_variants = []
    for lat in (2.0, 3.0, 4.0, 6.0, 8.0, 12.0):
        occ_variants.append(
            (f"wg   div=live            LAT={lat:>4}",
             dict(ship, sline="on", lat="wg", LAT=lat, LAT_SHUF=lat / 4.0, **L0L2)))
    for f in (0.5, 0.75, 0.9):
        for lat in (4.0, 8.0, 16.0):
            occ_variants.append(
                (f"wgmix f={f} LAT={lat:>4}      ",
                 dict(ship, sline="on", lat="wgmix", WG_F=f, LAT=lat, LAT_SHUF=lat / 4.0, **L0L2)))
    best = None
    for tag, p in occ_variants:
        row = [tag]
        nreg = {}
        for label, cells, isd, base in (("D", D, True, ref), ("M", Mo, False, refm)):
            pk = chooser(p, (1, 2, 4), cells, isd)
            rs = [r for _, r in pk.values()]
            bad = [(v[1] / base[k], k, v[0]) for k, v in pk.items()
                   if k in base and v[1] > base[k] * 1.02]
            nreg[label] = sorted(bad, reverse=True)
            row.append(f"{label} gm={gm(rs):.4f} w={max(rs):.2f} reg={len(bad):>3}"
                       + (f"/{bad and max(b[0] for b in bad) or 1.0:.2f}x" if bad else "/-    "))
        print("  ".join(row))
        score = (len(nreg["D"]) + len(nreg["M"]),
                 gm([r for _, r in chooser(p, (1, 2, 4), D, True).values()]))
        if best is None or score < best[0]:
            best = (score, tag, p, nreg)
    if best:
        print(f"\nBEST on (fewest regressions, then dense geomean): {best[1].strip()}")
        for label in ("D", "M"):
            bad = best[3][label]
            print(f"  {label}: {len(bad)} regressed"
                  + (f", worst {bad[0][0]:.2f}x" if bad else ""))
            for f_, k, c in bad[:12]:
                print(f"     {f_:5.2f}x  {k}  picked {c}")

    # ==============================================================================================
    # THE NEAR-TIE BAND -- do not switch tiles on a margin the model cannot resolve
    # ==============================================================================================
    # This is the ONE variant that changes no term of the cost model: same costs, same lattice, a
    # different ARGMIN POLICY. A pure argmin acts on a 3% predicted gap as though it were a decision,
    # and it is not -- measured pairwise over this surface (every ordered pair of measured WN=1
    # tiles, bucketed by the model's predicted margin) the model ranks a pair right 45-58% of the
    # time below 1.10x and the tile it prefers measures 0.96-1.05x the one it rejects. From 1.10 up,
    # accuracy climbs monotonically (66.7% / 74.0% / 79.3% / 82.2% / 91.0% / 95.4%). So 1.10 is the
    # model's ranking error, and inside it the chooser keeps the INCUMBENT 256x128 -- the tile
    # wmma_tiled_tuned was hard-wired to, and around which every other part of the core was tuned.
    #
    # This shipped as a REGRESSION first: at glm.gate_up tp2 g=128 M=448..512 the argmin preferred
    # 256x64 by a predicted 1.036x and the measurement says 256x128 is 1.19x faster -- +2.4% on a
    # ~516-token GLM TP=2 prefill TTFT, disjoint ranges.
    #
    # NOTE the restricted argmin makes this a HARDER test than the lattice-wide one, not an easier
    # one: 256x128 is in the measured set of essentially every cell, so every capture here is a
    # move between two MEASURED tiles and cannot hide in an unmeasured pick.
    print("\n" + "=" * 110)
    print("NEAR-TIE BAND -- keep the incumbent 256x128 when the argmin's margin is inside the")
    print("model's own measured pairwise ranking error. Cost model UNCHANGED; argmin POLICY changed.")
    print("=" * 110)

    # ---- WHERE THE BAND COMES FROM. Not fitted, and deliberately NOT the 1.036 geomean-vs-oracle
    # number: geomean regret measures how good the SELECTION ends up, and a tie threshold needs how
    # well the model ORDERS A PAIR. Measured directly -- every ordered pair of measured WN=1 tiles in
    # every cell, bucketed by the model's PREDICTED margin, scored against the MEASUREMENT.
    edges = [1.02, 1.04, 1.06, 1.08, 1.10, 1.15, 1.20, 1.30, 1.50, float("inf")]

    def rank_table(label, cellf, tilef):
        buck = defaultdict(lambda: [0, 0, []])
        for k, d in D.items():
            if not cellf(k):
                continue
            name, K, N, g, M = k
            ts = []
            for c, us in d.items():
                bm, bn, wn = tile_of(c)
                if wn != 1 or not tilef(bm, bn):     # the chooser may only pick WN=1
                    continue
                cst = dense_cost(M, N, K, g, bm, bn, 1, args.cu, ship)
                if cst:
                    ts.append((cst, us))
            for i, (ca, ua) in enumerate(ts):
                for j, (cb, ub) in enumerate(ts):
                    if i == j or cb <= ca:
                        continue
                    e = next(x for x in edges if cb / ca < x)
                    b = buck[e]
                    b[1] += 1
                    b[0] += (ua < ub)
                    b[2].append(ua / ub)
        lo, cols = 1.0, []
        for e in edges:
            if e not in buck:
                continue
            right, tot, rat = buck[e]
            cols.append(f"[{lo:.2f},{min(e, 9.99):.2f}) n={tot:<5} {100*right/tot:5.1f}% gm={gm(rat):.3f}")
            lo = e
        print(f"  {label:<18} " + "  ".join(cols))

    print("\nPAIRWISE RANKING ACCURACY -- 'model RIGHT %' and the MEASURED us(model's pick)/us(other),")
    print("bucketed by the model's own PREDICTED margin. Below ~1.10 the model is at CHANCE and the")
    print("tile it prefers is not faster; from 1.10 up, accuracy climbs monotonically. THAT is the band.")
    rank_table("ALL", lambda k: True, lambda bm, bn: True)
    rank_table("g=32", lambda k: k[3] == 32, lambda bm, bn: True)
    rank_table("g=128", lambda k: k[3] == 128, lambda bm, bn: True)
    rank_table("M<=64", lambda k: k[4] <= 64, lambda bm, bn: True)
    rank_table("M>=96 PREFILL", lambda k: k[4] >= 96, lambda bm, bn: True)
    rank_table("no lm_head", lambda k: k[2] < 100000, lambda bm, bn: True)
    rank_table("BM>=64 only", lambda k: True, lambda bm, bn: bm >= 64)
    print()
    tie_ref = None
    for band in (1.0, 1.02, 1.04, 1.06, 1.10, 1.15, 1.20, 1.30):
        p = dict(ship, tie_break=(256, 128), tie_band=band)
        pk = chooser(p, (1,), D, True)
        rs = [r for _, r in pk.values()]
        if tie_ref is None:
            tie_ref = {k: v for k, v in pk.items()}
        moved = [k for k in pk if pk[k][0] != tie_ref[k][0]]
        worse = [(pk[k][1] / tie_ref[k][1], k, tie_ref[k][0], pk[k][0])
                 for k in moved if pk[k][1] > tie_ref[k][1] * 1.02]
        better = [(pk[k][1] / tie_ref[k][1], k, tie_ref[k][0], pk[k][0])
                  for k in moved if pk[k][1] < tie_ref[k][1] * 0.98]
        print(f"  band={band:<5} DENSE gm={gm(rs):.4f} w={max(rs):.2f} n={len(rs)}  "
              f"moved={len(moved):<3} BETTER={len(better):<3} WORSE={len(worse)}")
        for f_, k, b, t in sorted(better)[:6]:
            print(f"       BETTER {f_:5.3f}x  {k}  {b} -> {t}")
        for f_, k, b, t in sorted(worse, reverse=True)[:6]:
            print(f"       WORSE  {f_:5.3f}x  {k}  {b} -> {t}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
