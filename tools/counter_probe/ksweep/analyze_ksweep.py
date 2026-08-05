#!/usr/bin/env python3
"""The three trace-side questions that no isolated bench can answer.

1. MoE PADDING WASTE (`moe_align`, the induced-cost suspect). moe_align produces
   `sorted_token_ids`/`expert_ids`/`num_tokens_post_pad`; the padded row count P is what sets the MoE
   GEMM grid. A cheap kernel choosing an expensive kernel's launch geometry is the textbook induced
   cost, and the repo has attributed MoE decode's poor showing to "block_m padding + gather at decode
   (structural, moe_align side)" without ever quantifying it. This measures P from the DISPATCH GRID
   of the MoE GEMMs on the served path, and compares it with the real row count M*top_k.

2. RANK SKEW around the TP=2 all-reduce. OBSERVATION ONLY — the vectorised custom_ar is already the
   fastest TP=2 path on this hardware and is not an optimisation target. But a collective is a sync
   point regardless of its quality, and the pair is MISMATCHED (GPU0 64 CU/320 W, GPU1 56 CU/260 W),
   so the slower rank gates every collective ~40-48x per step. That is a deployment fact, not a bug.

3. ELEMENTWISE / NORM BYTE VOLUME per step — the input that SIZES the cache-eviction experiment.
   The hypothesis is that individually trivial elementwise ops each do a full HBM round-trip of the
   activation and evict the weight working set between GEMMs, so a GEMM that is MALL-warm in a
   microbench is cold in the serve. To test that in isolation we need to know how many bytes the
   serve actually pushes between two GEMMs. Work-items come from the trace; the bytes-per-work-item
   assumption is stated explicitly rather than hidden, and both a 2-byte and a 4-byte accounting are
   printed so the reader can see the width of the estimate.

  python3 analyze_ksweep.py <rpv3-dir> --tag qwen-normal --out-dir <dir>
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from parse_ksweep import STEP_RE, load, short          # same directory

# Kernel-name families. Deliberately broad and then REPORTED, so a miscategorisation is visible in
# the printed membership list rather than silently folded into a total.
GEMM_RE = re.compile(r"gemm|gemv|matmul|wmma|linear|lm_head|moe_|w4a8|fp8_|int4|dense", re.I)
ELEM_RE = re.compile(r"elementwise|vectorized_elementwise|norm|rms|silu|swiglu|mul|add|rope|rotary|"
                     r"cast|copy|convert|scatter|gather|index|fill|reduce|softmax|sigmoid|act", re.I)
AR_RE = re.compile(r"all_?reduce|allreduce|custom_ar|one_?shot|two_?shot|all_?gather", re.I)
MOE_RE = re.compile(r"moe", re.I)
ALIGN_RE = re.compile(r"align|sort_token|expert_id|num_tokens_post", re.I)


def windows_of(steps):
    ws = [[steps[0]]]
    for s in steps[1:]:
        if s[0] == ws[-1][-1][0] + 1:
            ws[-1].append(s)
        else:
            ws.append([s])
    return ws


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rpdir")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--top-k", type=int, default=8, help="MoE experts per token (Qwen3.6-A3B = 8)")
    args = ap.parse_args()

    d = Path(args.rpdir)
    outdir = Path(args.out_dir); outdir.mkdir(parents=True, exist_ok=True)
    steps, disp, agent_name = load(d)
    out: dict = {"tag": args.tag, "rpdir": str(d), "agents": agent_name, "windows": []}

    for w in windows_of(steps):
        if len(w) < 3:
            continue
        idx0, idx1, wstart, wend = w[0][0], w[-1][0], w[0][1], w[-1][1]
        nper = len(w) - 1
        span = wend - wstart
        wrec: dict = {"iters": [idx0, idx1], "n_steps": len(w),
                      "wall_ms_per_step": span / nper / 1e6, "agents": {}}
        print(f"\n########## {args.tag}  window {idx0}..{idx1}  "
              f"({len(w)} steps, {span/nper/1e6:.3f} ms/step) ##########")

        for agent in sorted(disp):
            ks = [r for r in disp[agent] if r["s"] >= wstart and r["s"] < wend]
            if not ks:
                continue
            tot_ns = sum(r["e"] - r["s"] for r in ks)
            arec: dict = {"product": agent_name.get(agent, "?")}
            print(f"\n--- {agent} [{arec['product']}] ---")

            # ---- 1. MoE padding waste ------------------------------------------------------------
            moe = [r for r in ks if MOE_RE.search(r["name"])]
            if moe:
                fam = defaultdict(list)
                for r in moe:
                    fam[short(r["name"])].append(r)
                moe_out = {}
                print("  MoE dispatches (grid = work-items; wg = grid/block):")
                for k, rs in sorted(fam.items(), key=lambda x: -sum(e["e"] - e["s"] for e in x[1])):
                    g = Counter(f"{r['grid']}|{r['block']}" for r in rs)
                    gg, n = g.most_common(1)[0]
                    grid, block = gg.split("|")
                    gx, gy, gz = eval(grid)                            # noqa: S307 - our own string
                    bx, by, bz = eval(block)                           # noqa: S307
                    nwg = max(1, (gx * gy * gz) // max(1, bx * by * bz))
                    ns = sum(r["e"] - r["s"] for r in rs)
                    moe_out[k] = {"n_per_step": len(rs) / nper, "ms_per_step": ns / nper / 1e6,
                                  "grid": grid, "block": block, "workgroups": nwg,
                                  "grid_y": gy, "n_distinct_grids": len(g),
                                  "share_of_busy_ns": ns / tot_ns}
                    print(f"    {ns/nper/1e6:8.4f} ms/step  x{len(rs)/nper:6.1f}  "
                          f"grid={grid} block={block} -> {nwg} WG  (grid.y={gy})  {k[:70]}")
                arec["moe"] = moe_out
                # grid.y IS the padded-row dimension of the fused gemm2. Real rows = M * top_k.
                arec["moe_note"] = (
                    "grid.y of the fused gemm2 is the padded row count; real rows = M*top_k="
                    f"{args.top_k}*M. Compare grid.y against that to get the padding waste.")
            align = [r for r in ks if ALIGN_RE.search(r["name"])]
            if align:
                fam = Counter(short(r["name"]) for r in align)
                arec["align_kernels"] = {k: v / nper for k, v in fam.items()}
                print(f"  align-family kernels/step: "
                      f"{ {k: round(v/nper, 2) for k, v in fam.items()} }")

            # ---- 3. Byte volume by family --------------------------------------------------------
            fams = {"gemm": 0, "elementwise": 0, "allreduce": 0, "other": 0}
            fam_items = {k: 0 for k in fams}
            fam_n = {k: 0 for k in fams}
            members = defaultdict(Counter)
            for r in ks:
                nm = r["name"]
                if AR_RE.search(nm):
                    f = "allreduce"
                elif GEMM_RE.search(nm):
                    f = "gemm"
                elif ELEM_RE.search(nm):
                    f = "elementwise"
                else:
                    f = "other"
                fams[f] += r["e"] - r["s"]
                gx, gy, gz = r["grid"]
                fam_items[f] += gx * gy * gz
                fam_n[f] += 1
                members[f][short(nm)] += 1
            arec["families"] = {
                f: {"ms_per_step": fams[f] / nper / 1e6, "share_of_kernel_time": fams[f] / tot_ns,
                    "dispatches_per_step": fam_n[f] / nper,
                    "workitems_per_step": fam_items[f] / nper,
                    # An elementwise op reads its input and writes its output, so >=2 accesses per
                    # work-item. 2 B = bf16/fp16 activations, 4 B = fp32 accumulators/router. These
                    # BRACKET the traffic; they are not a measurement of it.
                    "MB_per_step_at_2B_rw": fam_items[f] / nper * 2 * 2 / 1e6,
                    "MB_per_step_at_4B_rw": fam_items[f] / nper * 4 * 2 / 1e6}
                for f in fams}
            print("  family                ms/step  %kernel-time  disp/step   workitems/step   "
                  "MB/step(2B r+w)  MB/step(4B r+w)")
            for f in ("gemm", "elementwise", "allreduce", "other"):
                v = arec["families"][f]
                print(f"    {f:<18} {v['ms_per_step']:8.3f} {100*v['share_of_kernel_time']:12.1f} "
                      f"{v['dispatches_per_step']:10.1f} {v['workitems_per_step']:16.0f} "
                      f"{v['MB_per_step_at_2B_rw']:16.1f} {v['MB_per_step_at_4B_rw']:16.1f}")
            arec["family_members"] = {f: dict(members[f].most_common(12)) for f in fams}

            wrec["agents"][agent] = arec

        # ---- 2. Rank skew around the all-reduce ---------------------------------------------------
        # Pair the Nth all-reduce dispatch on each agent within the window. They are the SAME
        # collective, so start-time difference is arrival skew and end-time difference is exit skew.
        ar_by_agent = {a: [r for r in disp[a] if r["s"] >= wstart and r["s"] < wend
                           and AR_RE.search(r["name"])] for a in disp}
        ar_by_agent = {a: v for a, v in ar_by_agent.items() if v}
        if len(ar_by_agent) == 2:
            a0, a1 = sorted(ar_by_agent)
            n = min(len(ar_by_agent[a0]), len(ar_by_agent[a1]))
            if n >= 10:
                start_skew = [ar_by_agent[a1][i]["s"] - ar_by_agent[a0][i]["s"] for i in range(n)]
                dur0 = [r["e"] - r["s"] for r in ar_by_agent[a0][:n]]
                dur1 = [r["e"] - r["s"] for r in ar_by_agent[a1][:n]]
                skew = {
                    "n_paired": n, "per_step": n / nper,
                    "agent_early": a0, "agent_late": a1,
                    "start_skew_ns_median": statistics.median(start_skew),
                    "start_skew_ns_p90": sorted(start_skew)[int(0.9 * n)],
                    "abs_start_skew_ns_median": statistics.median([abs(x) for x in start_skew]),
                    f"dur_ns_median_{a0}": statistics.median(dur0),
                    f"dur_ns_median_{a1}": statistics.median(dur1),
                    "total_abs_skew_ms_per_step":
                        sum(abs(x) for x in start_skew) / nper / 1e6,
                }
                wrec["ar_skew"] = skew
                print(f"\n  AR rank skew (OBSERVATION ONLY — custom_ar is not an optimisation "
                      f"target): {n/nper:.0f} collectives/step")
                print(f"    median |arrival skew| {skew['abs_start_skew_ns_median']:.0f} ns   "
                      f"signed median {skew['start_skew_ns_median']:+.0f} ns "
                      f"({a1} minus {a0})")
                print(f"    AR duration median: {a0}={statistics.median(dur0):.0f} ns  "
                      f"{a1}={statistics.median(dur1):.0f} ns")
                print(f"    total |skew| if fully exposed: "
                      f"{skew['total_abs_skew_ms_per_step']:.3f} ms/step")
        out["windows"].append(wrec)

    jf = outdir / f"{args.tag}.analysis.json"
    jf.write_text(json.dumps(out, indent=1, default=str))
    print(f"\nwrote {jf}")


if __name__ == "__main__":
    main()
