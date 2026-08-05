#!/usr/bin/env python3
"""Assemble the ranked deliverables from the artifacts the sweep produced.

FOUR TABLES, in the order they are worth reading:

  1. ISOLATED vs IN-SERVE, ranked by GAP. The primary question: the same kernels are far slower in
     the serve than in isolation and nobody has explained it. The join is per (op, shape) between
     iso_replay.csv and the in-serve per-dispatch time from the kernel trace. Ranked by ratio, not by
     absolute, because the ratio is what needs a mechanism.
  2. STARVATION suspects — kernels whose own share is negligible but which leave a large hole behind
     them (`gap_after`) or which set another kernel's launch geometry. Self-cost ranking buries these
     by construction, which is why they get their own table.
  3. SELF-COST ranking, share_of_step x deficit. `deficit` is the worst normalised shortfall among
     the triage flags that fired, so a kernel at 2% occupancy that is 0.1% of the step ranks as the
     noise it is.
  4. CLEAN — kernels with material share and NO flag. Saying which kernels are healthy is worth as
     much as the deficits, because it stops the next person re-profiling them.

Every number carries which route produced it: TRACE (live serve, auto clocks, no counters), ISO
(isolated replay, same image and toolchain, auto clocks), STATIC (code-object metadata), or COUNTER
(ROCm 7.14 at profile_standard — ratios only, never times).

  python3 rank_report.py --results-dir <dir> --tag qwen-normal
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

N_CU = 64
ROOFLINE_GBS = 706.6

# Kernel symbol -> the iso_replay op that replays it. Substring rules, applied in order; the first
# match wins. Deliberately explicit: a fuzzy automatic join between a C++ template signature and a
# torch op name would silently pair the wrong two kernels and the headline number would be a
# mis-join rather than a measurement.
SYMBOL_TO_OP = [
    (r"gemv_decode_core<.*Bf16GemvLoader",              "fp8_wmma.dense_bf16_gemv"),
    (r"gemv_decode_core<.*Fp8DenseGemvLoader",          "fp8_wmma.mmq_fp8_gemm"),
    (r"gemv_decode_core<.*Int4Fp8GemvLoader",           "fp8_wmma.mmq_fp8_moe_gemm1_silu"),
    (r"moe_gemm2_gather_reduce",                        "fp8_wmma.mmq_fp8_moe_gemm2_gather_reduce"),
    (r"moe_gather_reduce",                              "fp8_wmma.mmq_fp8_moe_gather_reduce"),
    (r"moe_gemm1_silu",                                 "fp8_wmma.mmq_fp8_moe_gemm1_silu"),
    (r"dense_gemm_pipe_kernel",                         "dense_gemm.dense_gemm_pipe"),
    (r"dense_gemm_rd",                                  "dense_gemm.dense_gemm_rd"),
    (r"flash_decode_paged_fp8",                         "attn_decode.flash_decode_paged_fp8"),
    (r"flash_decode_paged",                             "attn_decode.flash_decode_paged"),
    (r"flash_prefill_paged",                            "attn_prefill_paged.flash_prefill_paged"),
    (r"flash_prefill",                                  "attn_hip.flash_prefill"),
    (r"mla_decode",                                     "mla_hip.mla_decode"),
    (r"gdn_decode_conv_gated",                          "gdn_hip.gdn_decode_conv_gated"),
    (r"causal_conv1d_update",                           "gdn_hip.causal_conv1d_update"),
    (r"rmsnorm_gated",                                  "gdn_hip.rmsnorm_gated"),
    (r"moe_align",                                      "moe_hip.moe_align"),
    (r"topk_softmax|moe_topk",                          "moe_hip.moe_topk_softmax"),
    (r"rms_norm_add",                                   "tail_hip.rms_norm_add"),
    (r"rms_norm",                                       "tail_hip.rms_norm"),
    (r"silu_and_mul",                                   "tail_hip.silu_and_mul"),
    (r"store_kv",                                       "tail_hip.store_kv"),
    (r"\brope\b|rotary",                                "tail_hip.rope"),
]


def op_for(symbol: str) -> str | None:
    for pat, op in SYMBOL_TO_OP:
        if re.search(pat, symbol):
            return op
    return None


def load_kernels(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            for k in ("n_per_step", "ms_per_step", "share_of_busy", "share_of_wall",
                      "ns_per_dispatch_mean", "ns_per_dispatch_median", "ns_per_dispatch_min",
                      "gap_after_ns_median", "gap_after_ns_total_per_step", "occ_upper_pct"):
                r[k] = float(r[k]) if r.get(k) else 0.0
            for k in ("workgroups", "waves", "vgpr", "scratch_b", "lds_b", "threads_per_wg"):
                r[k] = int(float(r[k])) if r.get(k) else 0
            rows.append(r)
    return rows


def load_iso(path: Path) -> dict:
    """op -> condition -> ns (median). Tolerant of a CSV whose exact column names moved."""
    out: dict = defaultdict(dict)
    if not path.exists():
        return out
    with open(path) as f:
        lines = [ln for ln in f if not ln.startswith("#")]
    for r in csv.DictReader(lines):
        op = r.get("op") or r.get("name") or ""
        cond = r.get("condition") or r.get("cond") or ""
        ns = r.get("ns_median") or r.get("ns") or r.get("gemv_ns")
        if not (op and cond and ns):
            continue
        try:
            out[op][cond] = float(ns)
        except ValueError:
            continue
        out[op].setdefault("_shape", r.get("shape", ""))
        out[op].setdefault("_pct_roof", r.get("pct_roofline", ""))
    return out


def deficit_of(r: dict) -> tuple[float, str]:
    """The WORST normalised shortfall among the flags that fired, in [0,1].

    Normalised so the flags are commensurable: a kernel at 8 workgroups on a 64-CU part scores the
    same 0.875 launch deficit whatever its occupancy story is, and the largest single shortfall is
    the one that names the kernel's problem."""
    cands: list[tuple[float, str]] = []
    if r["workgroups"] < N_CU:
        cands.append((1.0 - r["workgroups"] / N_CU, f"WG={r['workgroups']}<{N_CU}"))
    if r["occ_upper_pct"] < 25:
        cands.append((1.0 - r["occ_upper_pct"] / 25.0, f"occ={r['occ_upper_pct']:.1f}%"))
    if r["scratch_b"] > 0:
        # 632 B/lane is the figure the standing lead quoted; normalise against it so the known case
        # scores ~1.0 and the dense_gemm 2676 B case saturates rather than dominating by 4x.
        cands.append((min(1.0, r["scratch_b"] / 632.0), f"spill={r['scratch_b']}B"))
    if not cands:
        return 0.0, ""
    return max(cands)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--window", default=None, help="only this window id (e.g. 120-319)")
    ap.add_argument("--min-share", type=float, default=0.002)
    args = ap.parse_args()

    rd = Path(args.results_dir)
    kf = rd / f"{args.tag}.kernels.csv"
    if not kf.exists():
        raise SystemExit(f"missing {kf}")
    rows = load_kernels(kf)
    iso = load_iso(rd / "iso_replay.csv")

    windows = sorted({r["window"] for r in rows})
    print(f"tag={args.tag}  windows={windows}")
    for win in windows:
        if args.window and win != args.window:
            continue
        wrows = [r for r in rows if r["window"] == win]
        agents = sorted({r["agent"] for r in wrows})
        for ag in agents:
            ar = [r for r in wrows if r["agent"] == ag]
            prod = ar[0]["product"] if ar else "?"
            classes = ar[0]["classes"] if ar else "?"
            print(f"\n{'='*118}\nWINDOW {win}  {ag} [{prod}]  classes={classes}\n{'='*118}")

            # ---- 3. self-cost ranking -------------------------------------------------------------
            scored = []
            for r in ar:
                d, why = deficit_of(r)
                scored.append((r["share_of_busy"] * d, d, why, r))
            scored.sort(key=lambda x: -x[0])
            print("\n--- SELF-COST RANKING (share_of_busy x deficit)  [route: TRACE + STATIC] ---")
            print(f"{'score':>7} {'%busy':>6} {'ms/step':>8} {'n/step':>7} {'WG':>6} {'waves':>7} "
                  f"{'occ%':>6} {'vgpr':>5} {'scr':>5} {'ns/disp':>9}  flag / kernel")
            for score, d, why, r in scored[:25]:
                if r["share_of_busy"] < args.min_share and score < 1e-4:
                    continue
                print(f"{score:7.4f} {100*r['share_of_busy']:6.2f} {r['ms_per_step']:8.4f} "
                      f"{r['n_per_step']:7.1f} {r['workgroups']:6d} {r['waves']:7d} "
                      f"{r['occ_upper_pct']:6.1f} {r['vgpr']:5d} {r['scratch_b']:5d} "
                      f"{r['ns_per_dispatch_median']:9.0f}  {why:<18} {r['kernel'][:52]}")

            # ---- 2. starvation suspects ------------------------------------------------------------
            print("\n--- STARVATION SUSPECTS: cheap but flagged, or leaving a hole behind them "
                  "[route: TRACE] ---")
            sus = [r for r in ar if r["flags"] and r["share_of_busy"] < 0.05]
            sus.sort(key=lambda r: -r["gap_after_ns_total_per_step"])
            print(f"{'%busy':>6} {'gapAfter/step':>14} {'gapAfter med':>13} {'n/step':>7} "
                  f"{'WG':>6} {'occ%':>6}  kernel  [flags]")
            for r in sus[:15]:
                print(f"{100*r['share_of_busy']:6.2f} "
                      f"{r['gap_after_ns_total_per_step']/1e6:14.4f} "
                      f"{r['gap_after_ns_median']:13.0f} {r['n_per_step']:7.1f} "
                      f"{r['workgroups']:6d} {r['occ_upper_pct']:6.1f}  "
                      f"{r['kernel'][:52]}  [{r['flags']}]")

            # ---- 1. isolated vs in-serve ------------------------------------------------------------
            if iso:
                print("\n--- ISOLATED vs IN-SERVE, ranked by GAP  [route: ISO + TRACE] ---")
                joined = []
                for r in ar:
                    op = op_for(r["kernel"])
                    if not op or op not in iso:
                        continue
                    hot = iso[op].get("hot")
                    if not hot:
                        continue
                    serve = r["ns_per_dispatch_median"]
                    joined.append((serve / hot, op, r, iso[op]))
                joined.sort(key=lambda x: -x[0])
                print(f"{'gap x':>7} {'serve ns':>9} {'hot ns':>9} {'rot ns':>9} {'evict ns':>9} "
                      f"{'neigh ns':>9} {'%busy':>6}  op / kernel")
                for gap, op, r, ic in joined[:20]:
                    print(f"{gap:7.2f} {r['ns_per_dispatch_median']:9.0f} {ic.get('hot',0):9.0f} "
                          f"{ic.get('rotated',0) or 0:9.0f} {ic.get('evict',0) or 0:9.0f} "
                          f"{ic.get('neighbours',0) or 0:9.0f} {100*r['share_of_busy']:6.2f}  "
                          f"{op}  {r['kernel'][:34]}")
                if not joined:
                    print("  (no joins — iso_replay.csv present but no op matched a served symbol)")
            else:
                print("\n--- ISOLATED vs IN-SERVE: iso_replay.csv absent, gap table not built ---")

            # ---- 4. clean --------------------------------------------------------------------------
            clean = [r for r in ar if not r["flags"] and r["share_of_busy"] >= args.min_share]
            clean.sort(key=lambda r: -r["share_of_busy"])
            print("\n--- CONFIRMED CLEAN (material share, no flag fired)  [route: TRACE + STATIC] ---")
            print(f"{'%busy':>6} {'ms/step':>8} {'n/step':>7} {'WG':>6} {'waves':>7} {'occ%':>6} "
                  f"{'vgpr':>5}  kernel")
            for r in clean[:20]:
                print(f"{100*r['share_of_busy']:6.2f} {r['ms_per_step']:8.4f} {r['n_per_step']:7.1f} "
                      f"{r['workgroups']:6d} {r['waves']:7d} {r['occ_upper_pct']:6.1f} "
                      f"{r['vgpr']:5d}  {r['kernel'][:62]}")


if __name__ == "__main__":
    main()
