#!/usr/bin/env python3
"""Per-kernel deficit table for a REAL serve, from a rocprofv3 --kernel-trace + --marker-trace.

WHAT THIS GETS WITHOUT HARDWARE COUNTERS, AND WHY THAT MATTERS
`--pmc` cannot be run against the serve image (ROCm 7.2.1 hangs on it, counters need 7.14, and a .so
built in one image will not load in the other — silently, as a 0%-GPU wedge). But the kernel trace
already carries, PER DISPATCH: Grid_Size_{X,Y,Z}, Workgroup_Size_{X,Y,Z}, VGPR_Count,
Accum_VGPR_Count, SGPR_Count, Scratch_Size, LDS_Block_Size. Workgroup count, wave count, spills and
register-limited occupancy are therefore MEASURED on the served path, not inferred from a replay.

OCCUPANCY MODEL, and its calibration. gfx1201: 32 WGP x 4 SIMD = 128 SIMDs, 16 wave slots each =
2048 wave slots; wave32; VGPR file 1536 per SIMD, granule 24 =>
    waves_per_simd(v) = min(16, 1536 // (ceil(v/24)*24))          [the established law]
    occ_upper% = 100 * min(waves_launched, 128*waves_per_simd) / 2048
Checked against the five hardware `OccupancyPercent` readings in docs/COUNTER_SCORECARD.md for the
fused MoE gemm2 (3.1 vs 2.2, 15.6 vs 12.3, 18.8 vs 14.8, 62.5 vs 54.9, 25.0 vs 18.1): this model is a
consistent UPPER BOUND, and the counter reads 0.71-0.88x of it because it time-averages over ramp-up
and tail. Quote it as a bound, never as the counter value.

BUSY IS THE UNION OF THE KERNEL INTERVALS, NOT THE SUM. Kernels on different streams run
concurrently, so summing double-counts and can exceed the wall it is divided by — which is how an
overhead-bound step gets mistaken for a saturated one. Both are printed so the double-count is
visible rather than assumed away.

PER AGENT (= per card). A TP=2 serve traces both ranks through the inherited LD_PRELOAD and their
kernels are disjoint device streams; unioning across agents would merge two cards into one timeline.

  python3 parse_ksweep.py <rpv3-dir> --tag qwen-normal --out-dir <dir>
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

STEP_RE = re.compile(r"^(overlap_step|normal_step|spec_step)#(\d+)$")

# gfx1201 geometry. `multi_processor_count` reports WGPs, so the 64-CU card answers 32 — never assert
# 64 from that field. 32 WGP x 2 CU x 2 SIMD = 128 SIMDs.
N_SIMD = 128
WAVE_SLOTS_PER_SIMD = 16
TOTAL_WAVE_SLOTS = N_SIMD * WAVE_SLOTS_PER_SIMD          # 2048
VGPR_FILE = 1536
VGPR_GRANULE = 24
WAVE_SIZE = 32
N_CU = 64

# A dispatch belongs to a PREFILL step if the step contains a kernel matching this. Classifying by
# duration alone would misfile a slow decode step; classifying by kernel identity is what the step
# actually did.
PREFILL_KERNEL_RE = re.compile(r"prefill|extend|varlen|chunk|_fwd_kernel|flash_attn_fwd", re.I)


def waves_per_simd_by_vgpr(vgpr: int) -> int:
    if vgpr <= 0:
        return WAVE_SLOTS_PER_SIMD
    granulated = math.ceil(vgpr / VGPR_GRANULE) * VGPR_GRANULE
    return max(1, min(WAVE_SLOTS_PER_SIMD, VGPR_FILE // granulated))


def union_ns(intervals: list[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    intervals.sort()
    total = 0
    cs, ce = intervals[0]
    for s, e in intervals[1:]:
        if s > ce:
            total += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    total += ce - cs
    return total


def short(name: str) -> str:
    """Collapse a C++ template signature to something a table can carry, WITHOUT merging kernels that
    differ. Keeps the outermost template name and drops the argument list."""
    n = name.strip()
    n = re.sub(r"^void\s+", "", n)
    # `(anonymous namespace)::` is part of the NAME, not the argument list. Cutting at the first
    # unnested "(" therefore truncated every anonymous-namespace kernel to the empty string — and
    # those include moe_align, moe_route_align, moe_topk_softmax, rms_norm_add and the paged flash
    # attention kernels, i.e. several of the largest entries in the table rendered as a BLANK row.
    n = n.replace("(anonymous namespace)::", "anon::")
    # drop the argument list of the outermost function
    depth = 0
    for i, ch in enumerate(n):
        if ch == "(" and depth == 0:
            n = n[:i]
            break
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
    return n[:150]


def load(d: Path):
    kfiles = sorted(d.glob("*kernel_trace.csv"))
    mfiles = sorted(d.glob("*marker_api_trace.csv"))
    if not kfiles:
        sys.exit(f"no kernel trace in {d}")
    steps: list[tuple[int, int, int, str]] = []
    for mf in mfiles:
        for row in csv.DictReader(open(mf)):
            m = STEP_RE.match(row.get("Function") or "")
            if m:
                steps.append((int(m.group(2)), int(row["Start_Timestamp"]),
                              int(row["End_Timestamp"]), m.group(1)))
    steps.sort()

    disp = defaultdict(list)
    torn = 0
    for kf in kfiles:
        for row in csv.DictReader(open(kf)):
            if row.get("Kind") != "KERNEL_DISPATCH":
                continue
            # BOTH TP ranks inherit the rocprofv3 LD_PRELOAD and write the SAME csv, so a handful of
            # lines are interleaved mid-record and parse as one row carrying two records' fields.
            # Measured: 2 torn rows in ~2.9M on a TP=2 run. Skip and COUNT them — silently coercing
            # them would put a kernel name where a grid dimension belongs, and silently dropping
            # them would hide a corruption rate if it ever stopped being negligible.
            if row.get(None) is not None:
                torn += 1
                continue
            try:
                gx = int(row["Grid_Size_X"]); gy = int(row["Grid_Size_Y"])
                gz = int(row["Grid_Size_Z"])
                bx = int(row["Workgroup_Size_X"]); by = int(row["Workgroup_Size_Y"])
                bz = int(row["Workgroup_Size_Z"])
            except (TypeError, ValueError):
                torn += 1
                continue
            disp[row["Agent_Id"]].append({
                "s": int(row["Start_Timestamp"]), "e": int(row["End_Timestamp"]),
                "name": row["Kernel_Name"],
                # rocprofv3 reports Grid_Size in WORK-ITEMS, block size in work-items per group.
                "grid": (gx, gy, gz), "block": (bx, by, bz),
                "vgpr": int(row.get("VGPR_Count") or 0),
                "agpr": int(row.get("Accum_VGPR_Count") or 0),
                "sgpr": int(row.get("SGPR_Count") or 0),
                "scratch": int(row.get("Scratch_Size") or 0),
                "lds": int(row.get("LDS_Block_Size") or 0),
            })
    if torn:
        print(f"WARNING: skipped {torn} torn CSV rows (concurrent TP-rank writes)")
    # COVERAGE GUARD. Under TP=2 both rank processes inherit the rocprofv3 LD_PRELOAD and write the
    # same output, and ONE OF THEM STOPS RECORDING EARLY — nondeterministically which. Measured:
    # the normal-loop trace kept Agent 1 for 98.58 s and dropped Agent 2 after 3.28 s; the
    # overlap-loop trace kept Agent 2 for 95.67 s and dropped Agent 1 after 26.49 s.
    # Reading per-kernel shares off a truncated agent understates its busy time and manufactures a
    # "this configuration does less GPU work" result. Print the span so the reader can see which
    # agent is authoritative, and never compare two agents' per-step counts without checking it.
    spans = {a: (v[-1]["e"] - v[0]["s"]) / 1e9 for a, v in disp.items() if v}
    if spans:
        full = max(spans.values())
        for a, s in sorted(spans.items()):
            mark = "FULL" if s > 0.8 * full else "*** TRUNCATED — DO NOT USE ***"
            print(f"  coverage {a}: {len(disp[a]):8d} dispatches over {s:6.2f} s   {mark}")
    for a in disp:
        disp[a].sort(key=lambda r: r["s"])

    agent_name = {}
    for af in d.glob("*agent_info.csv"):
        for row in csv.DictReader(open(af)):
            if row.get("Agent_Type") == "GPU":
                agent_name[f"Agent {row['Logical_Node_Id']}"] = row.get("Product_Name", "")
    return steps, disp, agent_name


def geometry(r: dict) -> dict:
    gx, gy, gz = r["grid"]; bx, by, bz = r["block"]
    threads_per_wg = max(1, bx * by * bz)
    total_threads = max(1, gx * gy * gz)
    n_wg = max(1, total_threads // threads_per_wg)
    waves_per_wg = math.ceil(threads_per_wg / WAVE_SIZE)
    waves = n_wg * waves_per_wg
    wps = waves_per_simd_by_vgpr(r["vgpr"])
    resident = min(waves, N_SIMD * wps)
    return {
        "workgroups": n_wg, "threads_per_wg": threads_per_wg, "waves_per_wg": waves_per_wg,
        "waves": waves, "waves_per_simd_by_vgpr": wps,
        "occ_upper_pct": 100.0 * resident / TOTAL_WAVE_SLOTS,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rpdir")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--top", type=int, default=45)
    args = ap.parse_args()

    d = Path(args.rpdir)
    outdir = Path(args.out_dir); outdir.mkdir(parents=True, exist_ok=True)
    steps, disp, agent_name = load(d)
    if not steps:
        sys.exit("no `<loop>#<n>` step ranges — plain-decode ROCTx markers did not fire "
                 "(MINISGL_ROCTX=1? windows inside a driven phase?)")

    loops = Counter(s[3] for s in steps)
    print(f"trace: {d}")
    print(f"steps: {len(steps)}   loop kinds: {dict(loops)}   agents: {sorted(disp)}")
    for a, nm in sorted(agent_name.items()):
        print(f"  {a} = {nm}")

    # Contiguous runs of iteration indices = the collection windows.
    windows: list[list[tuple]] = [[steps[0]]]
    for s in steps[1:]:
        if s[0] == windows[-1][-1][0] + 1:
            windows[-1].append(s)
        else:
            windows.append([s])

    rows_out = []
    report = {"rpdir": str(d), "tag": args.tag, "agents": agent_name,
              "loop_kinds": dict(loops), "windows": []}

    for w in windows:
        if len(w) < 3:
            continue
        idx0, idx1 = w[0][0], w[-1][0]
        wstart, wend = w[0][1], w[-1][1]
        span_ns = wend - wstart
        nper = len(w) - 1

        # Classify each step as prefill or decode by the kernels inside it. A single window may
        # legitimately hold both (the prefill phase is 1 prefill + PTOK decode iterations per
        # request), so the classes are separated rather than the window being labelled.
        busiest = max(disp, key=lambda a: sum(r["e"] - r["s"] for r in disp[a]
                                              if r["e"] > wstart and r["s"] < wend))
        step_bounds = [(w[i][1], w[i + 1][1]) for i in range(len(w) - 1)]
        step_class = []
        for s0, s1 in step_bounds:
            has_pf = any(PREFILL_KERNEL_RE.search(r["name"])
                         for r in disp[busiest] if r["s"] >= s0 and r["s"] < s1)
            step_class.append("prefill" if has_pf else "decode")
        cls_count = Counter(step_class)

        entry = {"iters": [idx0, idx1], "n_steps": len(w),
                 "wall_ms_per_step_marker": span_ns / nper / 1e6,
                 "step_classes": dict(cls_count), "agents": {}}
        print(f"\n=== window iters {idx0}..{idx1} ({len(w)} steps) "
              f"marker wall/step {span_ns/nper/1e6:.3f} ms  classes={dict(cls_count)} ===")

        for agent in sorted(disp):
            ks = [r for r in disp[agent] if r["e"] > wstart and r["s"] < wend]
            if not ks:
                continue
            iv = [(max(r["s"], wstart), min(r["e"], wend)) for r in ks]
            iv = [(s, e) for s, e in iv if e > s]
            u = union_ns(iv)
            tot = sum(e - s for s, e in iv)

            # Per-kernel aggregation, plus the GAP AFTER each dispatch on this agent's timeline.
            # `gap_after` is the induced-cost signal: a cheap kernel that consistently leaves a large
            # hole behind it is holding something up even though its own share is negligible.
            per = defaultdict(lambda: {"n": 0, "ns": 0, "durs": [], "gaps": [], "geo": None,
                                       "grids": Counter(), "grid_ns": Counter(), "grid_geo": {},
                                       "vgpr": 0, "scratch": 0, "lds": 0})
            seq = [r for r in disp[agent] if r["s"] >= wstart and r["s"] < wend]
            for i, r in enumerate(seq):
                k = short(r["name"])
                p = per[k]
                p["n"] += 1
                dur = r["e"] - r["s"]
                p["ns"] += dur
                p["durs"].append(dur)
                if i + 1 < len(seq):
                    p["gaps"].append(max(0, seq[i + 1]["s"] - r["e"]))
                # Keep geometry PER GRID and pick the time-DOMINANT one later. Keeping the worst
                # geometry seen and pairing it with the kernel's whole time share is how a symbol
                # that spends 57% of its time at 100% occupancy gets reported as "0.8% occupied":
                # `dense_bf16_gemv` runs 8 distinct grids per step, from 1 workgroup (N=512) to 384
                # (N=196608), and only ~3% of its time is in the 1-workgroup shape.
                p["grids"][f"{r['grid']}/{r['block']}"] += 1
                p["grid_ns"][f"{r['grid']}/{r['block']}"] += dur
                p["grid_geo"][f"{r['grid']}/{r['block']}"] = geometry(r)
                p["vgpr"] = max(p["vgpr"], r["vgpr"])
                p["scratch"] = max(p["scratch"], r["scratch"])
                p["lds"] = max(p["lds"], r["lds"])

            busy_ms = u / nper / 1e6
            a = {"product": agent_name.get(agent, "?"),
                 "kernels_per_step": len(iv) / nper,
                 "busy_union_ms_per_step": busy_ms,
                 "busy_sum_ms_per_step": tot / nper / 1e6,
                 "busy_frac_union": u / span_ns, "idle_frac": 1.0 - u / span_ns}
            print(f"  {agent} [{a['product']}]: {a['kernels_per_step']:.0f} kernels/step  "
                  f"busy(UNION) {busy_ms:.3f} ms = {100*a['busy_frac_union']:.1f}%  "
                  f"IDLE {100*a['idle_frac']:.1f}%")

            ranked = sorted(per.items(), key=lambda x: -x[1]["ns"])
            a["kernels"] = []
            for k, p in ranked:
                # DOMINANT grid = the one carrying the most TIME, not the most launches and not the
                # smallest grid. The flags below then describe the geometry the kernel actually
                # spends its time in.
                dom_grid, dom_ns = p["grid_ns"].most_common(1)[0]
                g = p["grid_geo"][dom_grid]
                p["geo"] = g
                dom_frac = dom_ns / p["ns"] if p["ns"] else 0.0
                # How much of this symbol's time is spent LAUNCH-STARVED, across all its grids. This
                # is the honest version of the "workgroups < CU count" flag for a symbol that runs
                # many shapes: the flag fires on the dominant grid, this says how much is affected.
                starved_ns = sum(ns for gk, ns in p["grid_ns"].items()
                                 if p["grid_geo"][gk]["workgroups"] < N_CU)
                flags = []
                if g["workgroups"] < N_CU:
                    flags.append(f"WG<{N_CU}")
                if g["occ_upper_pct"] < 25:
                    flags.append("OCC<25")
                if p["scratch"] > 0:
                    flags.append(f"SPILL{p['scratch']}B")
                rec = {
                    "kernel": k, "n_per_step": p["n"] / nper,
                    "ms_per_step": p["ns"] / nper / 1e6,
                    "share_of_busy": p["ns"] / u if u else 0.0,
                    "share_of_wall": p["ns"] / span_ns,
                    "ns_per_dispatch_mean": p["ns"] / p["n"],
                    "ns_per_dispatch_median": statistics.median(p["durs"]),
                    "ns_per_dispatch_min": min(p["durs"]),
                    "gap_after_ns_median": (statistics.median(p["gaps"]) if p["gaps"] else 0),
                    "gap_after_ns_total_per_step": (sum(p["gaps"]) / nper if p["gaps"] else 0),
                    "workgroups": g["workgroups"], "threads_per_wg": g["threads_per_wg"],
                    "waves": g["waves"], "occ_upper_pct": g["occ_upper_pct"],
                    "vgpr": p["vgpr"], "waves_per_simd_by_vgpr": g["waves_per_simd_by_vgpr"],
                    "scratch_b": p["scratch"], "lds_b": p["lds"],
                    "top_grid": dom_grid,
                    "dom_grid_time_frac": dom_frac,
                    "starved_time_frac": (starved_ns / p["ns"]) if p["ns"] else 0.0,
                    "n_distinct_grids": len(p["grids"]),
                    "flags": ",".join(flags),
                }
                a["kernels"].append(rec)
                rows_out.append({"tag": args.tag, "window": f"{idx0}-{idx1}",
                                 "classes": "/".join(f"{k2}:{v2}" for k2, v2 in cls_count.items()),
                                 "agent": agent, "product": a["product"], **rec})
            entry["agents"][agent] = a

            print(f"    {'ms/step':>8} {'n/step':>7} {'%busy':>6} {'ns/disp':>9} "
                  f"{'gapAfter':>9} {'domWG':>7} {'waves':>7} {'occ%':>6} {'dom%':>5} "
                  f"{'strv%':>5} {'grids':>5} {'vgpr':>5} {'scr':>5}  kernel")
            for rec in a["kernels"][:args.top]:
                print(f"    {rec['ms_per_step']:8.4f} {rec['n_per_step']:7.1f} "
                      f"{100*rec['share_of_busy']:6.2f} {rec['ns_per_dispatch_median']:9.0f} "
                      f"{rec['gap_after_ns_median']:9.0f} {rec['workgroups']:7d} "
                      f"{rec['waves']:7d} {rec['occ_upper_pct']:6.1f} "
                      f"{100*rec['dom_grid_time_frac']:5.0f} {100*rec['starved_time_frac']:5.0f} "
                      f"{rec['n_distinct_grids']:5d} {rec['vgpr']:5d} "
                      f"{rec['scratch_b']:5d}  {rec['kernel'][:60]}  {rec['flags']}")
        report["windows"].append(entry)

    jf = outdir / f"{args.tag}.kernels.json"
    jf.write_text(json.dumps(report, indent=1))
    cf = outdir / f"{args.tag}.kernels.csv"
    if rows_out:
        with open(cf, "w", newline="") as f:
            wcsv = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
            wcsv.writeheader()
            wcsv.writerows(rows_out)
    print(f"\nwrote {jf}\nwrote {cf}")


if __name__ == "__main__":
    main()
