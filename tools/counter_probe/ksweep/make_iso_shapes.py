#!/usr/bin/env python3
"""Turn the MEASURED serve trace into the isolated-replay spec.

The one thing this must get right is `evict_bytes`. The cache-eviction hypothesis says the serve's
elementwise/norm traffic between two GEMMs displaces the weight working set, so the isolated `evict`
condition is only a fair reproduction if it streams the SAME number of bytes the serve actually
streams between those two GEMMs. Guessing that number would make the whole experiment circular — it
would be tuned until it reproduced the answer we were looking for. So it comes from
analyze_ksweep.py's per-window family accounting, per batch size, and the derivation is printed.

Per-GEMM, not per-step: the eviction a single GEMM experiences is the elementwise traffic that ran
between it and the previous GEMM, i.e. (elementwise bytes per step) / (GEMM dispatches per step).
Both accountings are written into the spec so the sensitivity is visible.

  python3 make_iso_shapes.py --analysis <tag>.analysis.json --template iso_replay_example.json \
      --out iso_shapes.json [--window 0]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", required=True)
    ap.add_argument("--template", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--window", type=int, default=0, help="index into analysis['windows']")
    ap.add_argument("--dtype-bytes", type=int, default=2,
                    help="bytes per element for the elementwise accounting (2=bf16 activations)")
    args = ap.parse_args()

    an = json.loads(Path(args.analysis).read_text())
    w = an["windows"][args.window]
    # The BUSIEST agent — a TP=2 serve traces both ranks and their kernels are disjoint device
    # streams, so averaging across cards would understate the traffic one card actually sees.
    agent = max(w["agents"], key=lambda a: w["agents"][a]["families"]["gemm"]["ms_per_step"])
    fam = w["agents"][agent]["families"]
    key = f"MB_per_step_at_{args.dtype_bytes}B_rw"
    elem_mb = fam["elementwise"][key] + fam["other"][key]
    gemm_disp = fam["gemm"]["dispatches_per_step"]
    per_step_bytes = int(elem_mb * 1e6)
    per_gemm_bytes = int(per_step_bytes / gemm_disp) if gemm_disp else 0

    print(f"window {w['iters']} ({w['n_steps']} steps, {w['wall_ms_per_step']:.3f} ms/step)")
    print(f"agent {agent}: elementwise+other {elem_mb:.1f} MB/step at {args.dtype_bytes}B r+w, "
          f"{gemm_disp:.0f} GEMM dispatches/step")
    print(f"  evict_bytes per STEP = {per_step_bytes:,}")
    print(f"  evict_bytes per GEMM = {per_gemm_bytes:,}   <- the one the spec uses")

    spec = json.loads(Path(args.template).read_text())
    spec["_PROVENANCE"] = {
        "derived_from": args.analysis, "window": w["iters"], "agent": agent,
        "elementwise_plus_other_MB_per_step": elem_mb,
        "gemm_dispatches_per_step": gemm_disp,
        "evict_bytes_per_step": per_step_bytes,
        "evict_bytes_per_gemm": per_gemm_bytes,
        "dtype_bytes_assumed": args.dtype_bytes,
        "note": ("evict_bytes is the elementwise+other traffic BETWEEN two GEMM dispatches, "
                 "measured from the serve trace, not chosen. The per-step figure is recorded too "
                 "so the sensitivity of the conclusion to this choice is visible."),
    }
    for c in spec.get("cases", []):
        c["evict_bytes"] = per_gemm_bytes
    Path(args.out).write_text(json.dumps(spec, indent=1))
    print(f"wrote {args.out} ({len(spec.get('cases', []))} cases)")


if __name__ == "__main__":
    main()
