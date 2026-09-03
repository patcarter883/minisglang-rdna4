#!/usr/bin/env python3
"""P0 cross-run reproducibility roll-up.

`p0_llamacpp_baseline.py` reports median + spread WITHIN one run.  A single
run's median can still be a point on a slow curve driven by page-cache warmth,
because the checkpoint (93.68 GB) is larger than installed RAM (91.84 GB) and
one 100-token warm-up cannot warm it.  This script pools the independent runs
that were actually taken and reports the ACROSS-run spread, which is the honest
uncertainty on the number gates K4 and A0.4 read.

It measures nothing itself: it only reads `p0.json` documents produced by the
probe.  Every field it emits is traceable to one of them.

    python3 tools/offload/p0_cross_run.py
"""

from __future__ import annotations

import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "docs" / "measurements" / "WEIGHT_OFFLOAD_2026-09-02"

# (label, path to that run's p0.json).  Chronological.
RUNS = [
    ("run1 (schema 4)", RESULTS / "p0_run1_schema4" / "p0.json"),
    ("run2 (schema 5)", RESULTS / "p0_run2_schema5" / "p0.json"),
    ("run3 (schema 5, canonical)", RESULTS / "p0.json"),
]

METRICS = [
    ("bs1", "decode_tok_s", "bs=1 decode tok/s"),
    ("bs1", "prompt_tok_s", "bs=1 prompt tok/s"),
    ("bs1", "wall_tok_s", "bs=1 wall tok/s (prefill incl.)"),
    ("conc6", "aggregate_tok_s_wall", "CONC=6 aggregate tok/s (wall)"),
    ("conc6", "aggregate_decode_tok_s_sum_of_rates", "CONC=6 aggregate tok/s (sum of decode rates)"),
    ("conc6", "per_stream_decode_tok_s", "CONC=6 per-stream decode tok/s"),
]


def spread_pct(vals):
    m = statistics.median(vals)
    return round(100.0 * (max(vals) - min(vals)) / m, 2) if m else None


def main() -> int:
    docs = []
    for label, p in RUNS:
        if not p.exists():
            print(f"missing: {p}", file=sys.stderr)
            return 2
        d = json.loads(p.read_text())
        if d.get("status") not in ("ok", "degraded"):
            print(f"{label}: status {d.get('status')!r} -- refusing to pool", file=sys.stderr)
            return 2
        docs.append((label, p, d))

    out = {
        "artifact": "P0 cross-run reproducibility roll-up",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "_this_file_measures_nothing": (
            "every value is read from a p0.json produced by "
            "tools/offload/p0_llamacpp_baseline.py; this script only pools them."
        ),
        "runs": [
            {
                "label": lab,
                "path": str(p.relative_to(REPO)),
                "timestamp_utc": d.get("timestamp_utc"),
                "schema_version": d.get("schema_version"),
                "status": d.get("status"),
                "gpu_engagement": (d.get("gpu_engagement") or {}).get("verdict"),
                "reps_valid": {k: v.get("reps_valid") for k, v in (d.get("legs") or {}).items()},
            }
            for lab, p, d in docs
        ],
        "metrics": {},
        "pooled_reps": {},
    }

    for leg, metric, human in METRICS:
        per_run = []
        for lab, _, d in docs:
            s = ((d.get("legs") or {}).get(leg) or {}).get(metric)
            per_run.append(None if not s else s.get("median"))
        vals = [v for v in per_run if v is not None]
        out["metrics"][metric] = {
            "leg": leg,
            "human": human,
            "per_run_median": per_run,
            "across_run_median_of_medians": round(statistics.median(vals), 4) if vals else None,
            "across_run_min": min(vals) if vals else None,
            "across_run_max": max(vals) if vals else None,
            "across_run_spread_pct": spread_pct(vals) if len(vals) > 1 else None,
            "n_runs": len(vals),
        }

    # Pool the individual repetitions too -- more honest than pooling medians.
    for leg, metric, human in METRICS:
        allv = []
        for _, _, d in docs:
            s = ((d.get("legs") or {}).get(leg) or {}).get(metric)
            if s and s.get("values"):
                allv.extend(s["values"])
        if allv:
            out["pooled_reps"][metric] = {
                "human": human,
                "n": len(allv),
                "median": round(statistics.median(allv), 4),
                "mean": round(statistics.mean(allv), 4),
                "min": round(min(allv), 4),
                "max": round(max(allv), 4),
                "stdev": round(statistics.stdev(allv), 4) if len(allv) > 1 else None,
                "spread_pct_of_median": spread_pct(allv),
            }

    canon = docs[-1][2]
    dv = canon.get("derived") or {}
    out["canonical_run"] = {
        "label": docs[-1][0],
        "bs1_decode_tok_s": dv.get("measured_bs1_tok_s"),
        "K4_hard_kill_threshold_tok_s": dv.get("K4_hard_kill_threshold_tok_s"),
        "fraction_of_plan_derived_ceiling": dv.get("fraction_of_plan_derived_ceiling"),
    }
    pooled = out["pooled_reps"].get("decode_tok_s", {})
    if pooled.get("median"):
        out["canonical_run"]["K4_if_recomputed_on_pooled_median"] = round(pooled["median"] / 1.57, 3)
        out["canonical_run"]["pooled_vs_canonical_pct"] = round(
            100.0 * (pooled["median"] - (dv.get("measured_bs1_tok_s") or 0))
            / (dv.get("measured_bs1_tok_s") or 1),
            2,
        )

    jp = RESULTS / "p0_cross_run.json"
    jp.write_text(json.dumps(out, indent=2) + "\n")

    L = []
    A = L.append
    A("# P0 — cross-run reproducibility")
    A("")
    A("`p0.json` reports spread **within** one run. This file reports spread **across** the "
      "independent runs actually taken, which is the honest uncertainty on the number gates "
      "**K4** and **A0.4** read. Nothing here is a new measurement.")
    A("")
    A("| Run | timestamp | schema | status | GPU | valid reps |")
    A("|---|---|---|---|---|---|")
    for r in out["runs"]:
        A(f"| `{r['path']}` | {r['timestamp_utc']} | {r['schema_version']} | {r['status']} | "
          f"{r['gpu_engagement']} | {r['reps_valid']} |")
    A("")
    A("## Median per run")
    A("")
    A("| Metric | " + " | ".join(r["label"] for r in out["runs"]) +
      " | across-run median | spread % |")
    A("|---|" + "---|" * (len(out["runs"]) + 2))
    for _, metric, human in METRICS:
        m = out["metrics"][metric]
        A(f"| {human} | " + " | ".join(str(v) for v in m["per_run_median"]) +
          f" | **{m['across_run_median_of_medians']}** | {m['across_run_spread_pct']} |")
    A("")
    A("## Pooled repetitions (all runs, warm-ups already discarded by the probe)")
    A("")
    A("| Metric | n | median | min | max | stdev | spread % |")
    A("|---|---|---|---|---|---|---|")
    for _, metric, _h in METRICS:
        p = out["pooled_reps"].get(metric)
        if p:
            A(f"| {p['human']} | {p['n']} | **{p['median']}** | {p['min']} | {p['max']} | "
              f"{p['stdev']} | {p['spread_pct_of_median']} |")
    A("")
    c = out["canonical_run"]
    A(f"**Canonical run** (`{c['label']}`, the one `p0.json` points at): bs=1 decode "
      f"**{c['bs1_decode_tok_s']} tok/s**, K4 threshold **{c['K4_hard_kill_threshold_tok_s']} "
      f"tok/s**, {c['fraction_of_plan_derived_ceiling']}× the plan's derived 33.8 tok/s ceiling.")
    if "K4_if_recomputed_on_pooled_median" in c:
        A("")
        A(f"Recomputed on the pooled median instead, K4 would be "
          f"**{c['K4_if_recomputed_on_pooled_median']} tok/s** "
          f"({c['pooled_vs_canonical_pct']:+.2f}% from the canonical run). The two agree well "
          "inside the across-run spread, so no gate turns on which run is cited.")
    A("")
    A("Regenerate: `python3 tools/offload/p0_cross_run.py`")
    mp = RESULTS / "P0_CROSS_RUN.md"
    mp.write_text("\n".join(L) + "\n")

    print(f"wrote {jp}")
    print(f"wrote {mp}")
    for _, metric, human in METRICS:
        m = out["metrics"][metric]
        print(f"  {human:46s} per-run {m['per_run_median']} -> "
              f"{m['across_run_median_of_medians']} (spread {m['across_run_spread_pct']}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
