"""Decompose a block-diffusion canvas step from a rocprofv3 kernel+marker CSV trace.

Answers three things a flat kernel ranking cannot:

  1. WHERE THE STEP GOES BY PHASE. Every dispatch is assigned to the innermost enclosing ROCTx range
     (canvas_fwd / canvas_sampler / canvas_soft_embed, inside canvas_step#N), so "the backbone" and
     "the tail" are separated by construction rather than by guessing from kernel names.
  2. HOW MANY STEPS THE WINDOW ACTUALLY CAUGHT. `canvas_step#N` range starts ARE the per-step wall
     boundaries — marker and kernel timestamps come off the same rocprofv3 clock — so per-step
     averages are divided by a COUNTED number of steps, not by a wall-clock guess.
  3. WHAT EACH KERNEL FAMILY COSTS PER STEP, with dispatch counts. Counts are exact; kernel-trace
     inflates SMALL kernels more than large ones, so elementwise device times read as upper bounds
     and big-GEMM times as lower bounds. That asymmetry is stated rather than silently averaged.

GAP vs BUSY. Summing kernel durations double-counts nothing but also ignores overlap and idle. The
report gives both: `busy` = sum of dispatch durations, `wall` = the marker range span. busy/wall is
the occupancy of the step by SOME kernel, and a low ratio is the host/gap story.

  python3 tools/canvas_trace_report.py <trace_dir> [--top N]
"""
from __future__ import annotations

import bisect
import collections
import csv
import glob
import os
import re
import sys


def _find(d: str, *suffixes: str) -> str | None:
    for s in suffixes:
        hits = glob.glob(os.path.join(d, "**", f"*{s}"), recursive=True)
        if hits:
            return sorted(hits, key=lambda p: -os.path.getsize(p))[0]
    return None


def _rows(path: str):
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            yield r


def _num(r: dict, *names: str) -> int:
    for n in names:
        if n in r and r[n] not in (None, ""):
            try:
                return int(float(r[n]))
            except ValueError:
                pass
    return 0


# Kernel names are C++ mangled/templated and carry the whole template argument list; grouping on the
# raw name explodes into hundreds of near-duplicates and hides the family that actually costs. Strip
# template args and the arg list, then bucket into the families the engine reasons about.
_FAMILY = [
    (re.compile(r"one_shot_ar|custom_ar|all_reduce|allreduce|_ar_kernel", re.I), "COLLECTIVE"),
    (re.compile(r"moe_gemm|moe_gemm1|moe_gemm2|grouped_gemm|mmq_fp8_moe", re.I), "MoE GEMM"),
    (re.compile(r"w4a8|mmq_fp8_gemm|wmma_tiled|prefill_wmma|ashuffle", re.I), "DENSE GEMM (W4A8)"),
    (re.compile(r"dense_gemm|minv|gemv_decode|Cijk|gemm_kernel|hgemm|s_gemm", re.I), "DENSE GEMM (bf16/fp16)"),
    (re.compile(r"flash_prefill|flash_decode|attn_|attention", re.I), "ATTENTION"),
    (re.compile(r"rms_norm|rope_kernel|store_kv|gated_mul|silu|gelu", re.I), "TAIL (native)"),
    (re.compile(r"topk|sort|cumsum|softmax|multinomial|argmax|entropy|sampler", re.I), "SAMPLER"),
    (re.compile(r"embed|gather|index_select|scatter", re.I), "EMBED/GATHER"),
]


def family(name: str) -> str:
    for rx, fam in _FAMILY:
        if rx.search(name):
            return fam
    return "elementwise/copy/other"


def short(name: str) -> str:
    """Collapse template args and the arg list but KEEP the function name.

    Naively stripping `<...>` and `(...)` turns `void ns::kern<A,B>(args)` into the string `"void "`
    for every templated kernel in the trace -- which silently MERGED attention and the tail kernels
    into one 16 ms/step row here. Strip the arg list first, then peel balanced template brackets, and
    if what survives is only a storage-class keyword, fall back to the raw prefix."""
    # `(anonymous namespace)::` comes BEFORE the real name and contains parentheses, so stripping
    # from the first `(` would erase the whole name. Every HIP kernel in this repo is in an anonymous
    # namespace, so this is the common case, not an edge one.
    n = name.strip().strip('"').replace("(anonymous namespace)::", "")
    d = 0
    out = []
    for ch in n:  # drop everything inside balanced <> at any depth
        if ch == "<":
            d += 1
            if d == 1:
                out.append("<>")
        elif ch == ">":
            d = max(0, d - 1)
        elif d == 0:
            out.append(ch)
    n = "".join(out)
    n = re.sub(r"\(.*$", "", n).strip()          # the (arg list) and anything after it
    n = re.sub(r"^(void|__global__)\s+", "", n)  # storage class carries no information
    return (n or name.strip().strip('"'))[:76]


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    d = sys.argv[1]
    topn = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 18

    kf = _find(d, "kernel_trace.csv")
    mf = _find(d, "marker_api_trace.csv", "marker_trace.csv")
    if not kf:
        print(f"no kernel_trace.csv under {d}")
        return 1
    print(f"kernel_trace : {kf}")
    print(f"marker_trace : {mf}")

    # ---- markers -------------------------------------------------------------------------------
    ranges: list[tuple[int, int, str]] = []
    steps: list[tuple[int, int, str]] = []
    if mf:
        for r in _rows(mf):
            # rocprofv3 names the marker column `Function` in the marker_api_trace CSV; `Name` is the
            # kernel-trace spelling. Accept both rather than assume a version.
            nm = (r.get("Function") or r.get("Name") or r.get("Marker_Name") or "").strip().strip('"')
            if not nm:
                continue
            s = _num(r, "Start_Timestamp", "Start_Timestamp(ns)")
            e = _num(r, "End_Timestamp", "End_Timestamp(ns)")
            if e <= s:
                continue
            if nm.startswith("canvas_step#"):
                steps.append((s, e, nm))
            elif nm.startswith("canvas_"):
                ranges.append((s, e, nm))
    steps.sort()
    ranges.sort()
    # Innermost-enclosing lookup: phases never nest inside each other, so a sorted start list + a
    # bisect is enough and is O(log n) per dispatch instead of O(n).
    r_starts = [x[0] for x in ranges]

    def phase_of(ts: int) -> str:
        """Innermost enclosing phase range, or a named bucket when there is none.

        The three phase ranges are SEQUENTIAL within a step, not nested, so the candidate is simply
        the last range that started at or before `ts`. Walking further back is bounded (8) so a
        dispatch that fell in a gap between phases cannot cost a linear scan of the whole trace —
        and it is reported as a gap rather than silently attributed to the previous phase, because
        'which kernels run OUTSIDE any phase' is itself a finding (async/deferred work)."""
        i = bisect.bisect_right(r_starts, ts) - 1
        for j in range(i, max(-1, i - 8), -1):
            s, e, nm = ranges[j]
            if s <= ts < e:
                return nm
        return "(between phases: gap / async)"

    nsteps = len(steps)
    if nsteps:
        wall = sum(e - s for s, e, _ in steps)
        print(f"\ncanvas steps in the window : {nsteps}"
              f"   (marker wall {wall/1e6:.2f} ms total, {wall/nsteps/1e6:.2f} ms/step)")
        lo, hi = steps[0][0], steps[-1][1]
    else:
        print("\n!! no canvas_step# markers — the collection window never opened over the canvas loop")
        lo, hi = 0, 1 << 62

    # ---- kernels -------------------------------------------------------------------------------
    by_phase: dict[str, int] = collections.Counter()
    n_phase: dict[str, int] = collections.Counter()
    by_fam: dict[str, int] = collections.Counter()
    n_fam: dict[str, int] = collections.Counter()
    by_k: dict[tuple[str, str], int] = collections.Counter()
    n_k: dict[tuple[str, str], int] = collections.Counter()
    # A TP=2 serve traces BOTH ranks through the inherited LD_PRELOAD. Their kernels are disjoint
    # device streams, so summing across agents reports two cards' work as one step and doubles every
    # per-step figure. Default to the busiest single agent (= one rank = one card, which is what a
    # per-step cost should mean) and say so; `--agent all` opts into the sum deliberately.
    want_agent = sys.argv[sys.argv.index("--agent") + 1] if "--agent" in sys.argv else None
    agents: dict[str, int] = collections.Counter()
    for r in _rows(kf):
        s = _num(r, "Start_Timestamp", "Start_Timestamp(ns)")
        e = _num(r, "End_Timestamp", "End_Timestamp(ns)")
        if e > s and lo <= s <= hi:
            agents[(r.get("Agent_Id") or "?").strip().strip('"')] += e - s
    if agents:
        print("\nagents (ranks/cards) in the window:  "
              + ", ".join(f"{a}={v/1e6:.1f}ms" for a, v in agents.most_common()))
    if want_agent is None and len(agents) > 1:
        want_agent = agents.most_common(1)[0][0]
        print(f"  -> reporting AGENT {want_agent} only (one rank = one card). "
              f"Use --agent all to sum both ranks.")
    if want_agent == "all":
        want_agent = None

    total = ndisp = 0
    for r in _rows(kf):
        s = _num(r, "Start_Timestamp", "Start_Timestamp(ns)")
        e = _num(r, "End_Timestamp", "End_Timestamp(ns)")
        if e <= s or not (lo <= s <= hi):
            continue
        if want_agent and (r.get("Agent_Id") or "?").strip().strip('"') != want_agent:
            continue
        nm = (r.get("Kernel_Name") or r.get("Name") or "?").strip().strip('"')
        dur = e - s
        ph, fam = phase_of(s), family(nm)
        total += dur
        ndisp += 1
        by_phase[ph] += dur
        n_phase[ph] += 1
        by_fam[fam] += dur
        n_fam[fam] += 1
        by_k[(fam, short(nm))] += dur
        n_k[(fam, short(nm))] += 1

    if not ndisp:
        print("no dispatches inside the marker window")
        return 1
    per = max(nsteps, 1)
    print(f"\ndispatches in the window   : {ndisp}  ({ndisp/per:.0f}/step)")
    print(f"summed kernel busy         : {total/1e6:.2f} ms  ({total/per/1e6:.2f} ms/step)")
    if nsteps:
        print(f"busy / marker wall         : {100*total/wall:.1f}%   "
              "(<100% is gap+idle; >100% means concurrent kernels or overlapping streams)")

    print(f"\n=== BY PHASE (ROCTx range) — per step over {per} steps")
    print(f"{'share':>7} {'ms/step':>9} {'disp/step':>10}  phase")
    for k, v in by_phase.most_common():
        print(f"{100*v/total:6.1f}% {v/per/1e6:9.3f} {n_phase[k]/per:10.1f}  {k}")

    print(f"\n=== BY KERNEL FAMILY — per step over {per} steps")
    print(f"{'share':>7} {'ms/step':>9} {'disp/step':>10} {'us/disp':>9}  family")
    for k, v in by_fam.most_common():
        print(f"{100*v/total:6.1f}% {v/per/1e6:9.3f} {n_fam[k]/per:10.1f} "
              f"{v/n_fam[k]/1e3:9.2f}  {k}")

    print(f"\n=== TOP {topn} KERNELS — per step over {per} steps")
    print(f"{'share':>7} {'ms/step':>9} {'disp/step':>10} {'us/disp':>9}  family / kernel")
    for (fam, nm), v in by_k.most_common(topn):
        print(f"{100*v/total:6.1f}% {v/per/1e6:9.3f} {n_k[(fam, nm)]/per:10.1f} "
              f"{v/n_k[(fam, nm)]/1e3:9.2f}  [{fam}] {nm}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
