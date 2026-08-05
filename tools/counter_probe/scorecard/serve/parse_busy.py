#!/usr/bin/env python3
"""GPU-busy vs wall per decode STEP, from a rocprofv3 --kernel-trace + --marker-trace of a real serve.

BUSY IS THE UNION OF THE KERNEL INTERVALS, NOT THE SUM. Kernels on different streams/queues run
concurrently, so summing durations double-counts and can report a "busy" time larger than the wall it
is divided by — which is how an overhead-bound step gets mistaken for a saturated one. Both are
printed here so the size of the double-count is visible rather than assumed away.

WALL COMES FROM THE MARKERS. Each plain-decode loop iteration is wrapped in a ROCTx range named
`<loop>#<iteration>` (scheduler.py::_rtx_step_begin), and rocprofv3 stamps marker and kernel records
from the SAME clock — so consecutive range starts are the per-step wall boundaries, in the same time
base as the dispatches. That avoids correlating a host stopwatch in one process against GPU
timestamps in another.

Per AGENT (= per card). A TP=2 serve traces both ranks through the inherited LD_PRELOAD, and their
kernels are disjoint device streams; unioning across agents would merge two cards' work into one
timeline and understate idle.

  python3 parse_busy.py <rpv3-dir> [--json-out out.json]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# `canvas_step` is the block-diffusion loop (scheduler/diffusion.py::_canvas_step). It is a peer of
# the plain-decode loops, opened through the same `_rtx_step_begin`, so it carries the same
# `<loop>#<iteration>` shape and the busy/wall arithmetic below applies to it unchanged.
STEP_RE = re.compile(r"^(overlap_step|normal_step|canvas_step)#(\d+)$")


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rpdir")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()

    d = Path(args.rpdir)
    kfiles = sorted(d.glob("*kernel_trace.csv"))
    mfiles = sorted(d.glob("*marker_api_trace.csv"))
    if not kfiles or not mfiles:
        sys.exit(f"missing kernel/marker trace in {d}: kernel={kfiles} marker={mfiles}")

    # --- step markers ----------------------------------------------------------------------------
    steps: list[tuple[int, int, int]] = []            # (iteration, start_ns, end_ns)
    for mf in mfiles:
        for row in csv.DictReader(open(mf)):
            m = STEP_RE.match(row.get("Function") or "")
            if m:
                steps.append((int(m.group(2)), int(row["Start_Timestamp"]),
                              int(row["End_Timestamp"])))
    steps.sort()
    if not steps:
        sys.exit(f"no `<loop>#<n>` step ranges in {mfiles} — the plain-decode roctx markers did not "
                 f"fire (MINISGL_ROCTX=1? MINISGL_ROCTX_WINDOWS placed inside a driven phase?)")

    # Contiguous runs of iteration indices = the collection windows.
    windows: list[list[tuple[int, int, int]]] = [[steps[0]]]
    for s in steps[1:]:
        if s[0] == windows[-1][-1][0] + 1:
            windows[-1].append(s)
        else:
            windows.append([s])

    # --- kernel dispatches -----------------------------------------------------------------------
    by_agent: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    for kf in kfiles:
        for row in csv.DictReader(open(kf)):
            if row.get("Kind") != "KERNEL_DISPATCH":
                continue
            by_agent[row["Agent_Id"]].append(
                (int(row["Start_Timestamp"]), int(row["End_Timestamp"]), row["Kernel_Name"]))

    # Agent_Id -> product name, so a per-card row says WHICH card. The pair is not symmetric here
    # (RX 9070 XT + RX 9070), so an unnamed "Agent 2" is not interchangeable with "Agent 1".
    agent_name = {}
    for af in d.glob("*agent_info.csv"):
        for row in csv.DictReader(open(af)):
            if row.get("Agent_Type") == "GPU":
                agent_name[f"Agent {row['Logical_Node_Id']}"] = row.get("Product_Name", "")

    report = {"rpdir": str(d), "agents": agent_name, "windows": []}
    print(f"trace: {d}")
    print(f"steps captured: {len(steps)} in {len(windows)} window(s); "
          f"agents: {sorted(by_agent)}")
    print()

    for w in windows:
        n = len(w)
        if n < 3:
            continue
        idx0, idx1 = w[0][0], w[-1][0]
        # Wall/step from consecutive range STARTS over the window, excluding the last range (whose
        # own end is not a step boundary). This is the served step period, not a per-range duration.
        span_ns = w[-1][1] - w[0][1]
        nper = n - 1
        wall_ms = span_ns / nper / 1e6
        wstart, wend = w[0][1], w[-1][1]

        entry = {"iters": [idx0, idx1], "n_steps": n, "wall_ms_per_step": wall_ms, "agents": {}}
        print(f"=== window iters {idx0}..{idx1}  ({n} steps)  "
              f"wall/step = {wall_ms:.3f} ms ===")
        for agent, ks in sorted(by_agent.items()):
            iv = [(max(s, wstart), min(e, wend)) for s, e, _ in ks if e > wstart and s < wend]
            iv = [(s, e) for s, e in iv if e > s]
            if not iv:
                continue
            u = union_ns(iv)
            tot = sum(e - s for s, e in iv)
            names = Counter(nm for s, e, nm in ks if e > wstart and s < wend)
            dur = defaultdict(int)
            for s, e, nm in ks:
                if e > wstart and s < wend:
                    dur[nm] += min(e, wend) - max(s, wstart)
            busy_ms = u / nper / 1e6
            sum_ms = tot / nper / 1e6
            a = {
                "kernels_per_step": len(iv) / nper,
                "busy_union_ms_per_step": busy_ms,
                "busy_sum_ms_per_step": sum_ms,
                "busy_frac_union": u / span_ns,
                "busy_frac_sum": tot / span_ns,
                "idle_frac": 1.0 - u / span_ns,
                "top_kernels": [
                    {"name": nm[:90], "n_per_step": names[nm] / nper,
                     "ms_per_step": dur[nm] / nper / 1e6}
                    for nm, _ in sorted(dur.items(), key=lambda x: -x[1])[:args.top]],
            }
            entry["agents"][agent] = a
            print(f"  {agent} [{agent_name.get(agent, '?')}]: "
                  f"{a['kernels_per_step']:.1f} kernels/step   "
                  f"busy(UNION) {busy_ms:.3f} ms = {100*a['busy_frac_union']:.1f}%   "
                  f"busy(sum) {sum_ms:.3f} ms = {100*a['busy_frac_sum']:.1f}%   "
                  f"IDLE {100*a['idle_frac']:.1f}%")
        report["windows"].append(entry)
        print()

    # Kernel ranking for the FIRST window's busiest agent — what the step is actually made of.
    if report["windows"]:
        w0 = report["windows"][0]
        if w0["agents"]:
            ag = max(w0["agents"], key=lambda a: w0["agents"][a]["busy_union_ms_per_step"])
            print(f"top kernels, window {w0['iters']}, {ag} (ms/step, count/step):")
            for k in w0["agents"][ag]["top_kernels"]:
                print(f"  {k['ms_per_step']:7.3f} ms  x{k['n_per_step']:6.1f}  {k['name']}")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=1))
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
