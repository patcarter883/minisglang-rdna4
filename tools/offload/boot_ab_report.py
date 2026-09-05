#!/usr/bin/env python3
"""Side-by-side of two boot-timeline legs, plus the invariants that decide whether the A/B COUNTS.

A faster boot that loads different weights is worthless, so the bit-identity block is printed with
the same weight as the timing block and any divergence is named. The fields checked are the ones
that would move if the read path changed WHAT was loaded rather than HOW: the arena plan digest and
region forecast (the carve order IS the layout), `arena_pinned_bytes` / `arena_carved_bytes`,
`torch_fallbacks`, `seam_pointer_checked`, `stage_b_keys_filled`, and the greedy token ids from both
prompts on both ranks.

The box columns are printed next to the timings on purpose: `disk_read_bytes_total`,
`swap_out_pages` and `major_faults` are how a read-path change is told apart from a box-state
difference, and a leg whose MemAvailable started 10 GiB lower is not the same experiment.

Host-side post-processing. No GPU, no lease.

    python tools/offload/boot_ab_report.py <outdir> <before-tag> <after-tag>
"""
from __future__ import annotations

import json
import os
import re
import sys

OUT, A, B = sys.argv[1], sys.argv[2], sys.argv[3]
GIB = 1 << 30


def load(tag: str, suffix: str):
    p = os.path.join(OUT, f"{tag}.{suffix}")
    return json.load(open(p)) if os.path.exists(p) else None


def row(name, a, b, fmt="{:9.2f}"):
    if a is None and b is None:
        return
    a = 0.0 if a is None else a
    b = 0.0 if b is None else b
    d = f"{(b - a) / a * 100:+7.1f}%" if a else "      -"
    print(f"  {name:<36} {fmt.format(a)} {fmt.format(b)}   {d}")


for rank in (0, 1):
    ba, bb = load(A, f"boot.rank{rank}.json"), load(B, f"boot.rank{rank}.json")
    if not ba or not bb:
        print(f"rank{rank}: MISSING timeline ({A}={bool(ba)} {B}={bool(bb)})")
        continue
    print(f"\n=== rank {rank}   {A} -> {B} ===")
    print(f"  {'':<36} {'before':>9} {'after':>9}   delta")
    row("TOTAL boot", ba.get("total_seconds"), bb.get("total_seconds"))

    pa = {p["name"]: p["seconds"] for p in ba.get("phases", [])}
    pb = {p["name"]: p["seconds"] for p in bb.get("phases", [])}
    for k in sorted(set(pa) | set(pb), key=lambda k: -max(pa.get(k, 0), pb.get(k, 0))):
        if max(pa.get(k, 0), pb.get(k, 0)) >= 1.0:
            row("phase  " + k, pa.get(k), pb.get(k))

    ka, kb = ba.get("buckets_seconds", {}), bb.get("buckets_seconds", {})
    for k in sorted(set(ka) | set(kb), key=lambda k: -max(ka.get(k, 0), kb.get(k, 0))):
        if max(ka.get(k, 0), kb.get(k, 0)) >= 1.0:
            row("bucket " + k, ka.get(k), kb.get(k))

    ca, cb = ba.get("counters", {}), bb.get("counters", {})
    for k in sorted(set(ca) | set(cb)):
        print(f"  count  {k:<29} {ca.get(k, 0):>15} {cb.get(k, 0):>15}")
    for k in ("swap_in_pages", "swap_out_pages", "major_faults", "disk_read_bytes_total"):
        print(f"  box    {k:<29} {ba.get(k, 0):>15} {bb.get(k, 0):>15}")
    for k in ("RssAnon", "VmHWM"):
        va = ba.get("rss_end", {}).get(k, 0) / GIB
        vb = bb.get("rss_end", {}).get(k, 0) / GIB
        print(f"  rss    {k:<29} {va:>14.2f}G {vb:>14.2f}G")
    for k in ("MemAvailable",):
        va = ba.get("box_start", {}).get(k, 0) / GIB
        vb = bb.get("box_start", {}).get(k, 0) / GIB
        print(f"  box    {k+' at start':<29} {va:>14.2f}G {vb:>14.2f}G")

ta, tb = load(A, "test.json"), load(B, "test.json")
print("\n=== BIT-IDENTITY ===")
if not ta or not tb:
    print(f"  MISSING test json ({A}={bool(ta)} {B}={bool(tb)})")
else:
    # WEIGHT IDENTITY FIRST, because it is the only field here that is a statement about the DATA.
    # Everything under it is an accounting identity, and every one of those passes over an arena
    # holding the right number of the wrong bytes.
    same_keys = ("weight_digest", "weight_digest_tensors", "weight_digest_bytes",
                 "weight_digest_host_tensors", "weight_digest_host_bytes",
                 "weight_digest_device_tensors", "weight_digest_device_bytes",
                 "weight_digest_arena_owned_tensors", "weight_digest_layer_buckets",
                 # BEHAVIOUR, at the ONLY width where it is evidence. `parity_*_ids` is the 12-step
                 # bs=1 probe, captured and eager.
                 "parity_captured_ids", "parity_eager_ids",
                 "arena_pinned_bytes", "arena_carved_bytes", "arena_torch_fallbacks",
                 "seam_pointer_checked", "stage_b_keys_filled", "stage_b_host_layers",
                 "stage_b_device_layers", "plan_host_bytes", "plan_device_bytes",
                 "copied_bytes", "stage_b_chunks", "engaged",
                 "seam_host_bytes", "seam_device_bytes", "kv_pages")
    # REPORTED, NEVER GATED — and `token_ids` MOVED HERE [BOOT-2026-09-05].
    #
    # `out["token_ids"]` is the `[5]` probe, which submits BOTH prompts in ONE `llm.generate([...])`
    # with `max_running_req=2`. They therefore decode TOGETHER at bs=2, where the MoE gemm2 is
    # `mmq_fp8_moe_gemm_scatter` — an ATOMIC scatter, whose accumulation order is not fixed and which
    # is documented as not bit-exact. Those six ids are not a function of the weights and they vary
    # boot to boot on unchanged code: `L48_r2_before`, `L48_r3_before`, `L48_r2_after`, `L48_r3_after`
    # all recorded [11751, 13, 11751, 369, 264, 3177] while `L48_id_before`/`L48_id_before2` — the
    # SAME 02957062 checkout as `L48_r3_before` — both recorded [11751, 13, 561, 6511, 314, 9564].
    # Gating on them, as the r2/r3 reports did, produces a green that is a coin landing the same way
    # twice and a red that indicts a correct change.
    report_only = ("token_ids", "text")
    show_keys = ("boot_seconds", "stage_b_seconds", "stage_b_peak_host_rss",
                 "weight_digest_seconds")
    bad = []
    for i, (ra, rb) in enumerate(zip(ta["ranks"], tb["ranks"])):
        for k in same_keys:
            va, vb = ra.get(k), rb.get(k)
            if va is None and vb is None:
                continue  # the leg did not run this probe; say nothing rather than "SAME None"
            ok = va == vb
            print(f"  rank{i} {k:<32} {'SAME' if ok else 'DIFFER'}  "
                  f"{va if ok else (va, vb)}")
            if not ok:
                bad.append(f"rank{i}.{k}")
        # PER-LAYER, so a divergence names the layer instead of only existing. 48 layers + body.
        la, lb = ra.get("weight_digest_per_layer"), rb.get("weight_digest_per_layer")
        if la and lb:
            diff = sorted(k for k in set(la) | set(lb) if la.get(k) != lb.get(k))
            print(f"  rank{i} {'weight_digest_per_layer':<32} "
                  f"{'ALL ' + str(len(la)) + ' MATCH' if not diff else 'DIFFER: ' + str(diff[:8])}")
            if diff:
                bad.append(f"rank{i}.per_layer[{len(diff)}]")
        for k in report_only:
            va, vb = ra.get(k), rb.get(k)
            if va is None and vb is None:
                continue
            tag = "same" if va == vb else "VARIES (bs=2 atomic scatter; not a gate)"
            print(f"  rank{i} {k+' [report-only]':<32} {tag}")
            if va != vb:
                print(f"        {va}\n     -> {vb}")
        for k in show_keys:
            if ra.get(k) is not None or rb.get(k) is not None:
                print(f"  rank{i} {k:<32} {ra.get(k)} -> {rb.get(k)}")
    print(f"  failures: {ta.get('failures')} -> {tb.get('failures')}")
    print("  *** DIVERGED: " + ", ".join(bad) if bad else "  *** every identity field matches")

    # --- BEHAVIOUR AND THROUGHPUT -------------------------------------------------------------
    # Separate block, because these are not identity claims of the same kind. Greedy ids either
    # match or they do not; tok/s is a measurement with spread and is only allowed to be "unchanged"
    # within it. `repro_engine_is_reproducible` is the FLOOR: if the engine does not reproduce its
    # own ids inside one boot, an ids-match across two boots is not evidence of anything.
    print("\n=== BEHAVIOUR / THROUGHPUT ===")
    for i, (ra, rb) in enumerate(zip(ta["ranks"], tb["ranks"])):
        for k in ("repro_engine_is_reproducible", "parity_verdict",
                  "throughput_captured_tok_per_s", "throughput_eager_tok_per_s",
                  "throughput_captured_spread_s", "throughput_eager_spread_s",
                  "throughput_captured_samples_s", "throughput_eager_samples_s"):
            va, vb = ra.get(k), rb.get(k)
            if va is None and vb is None:
                continue
            print(f"  rank{i} {k:<32} {va} -> {vb}")
        for k in ("repro_captured", "repro_eager"):
            v = ra.get(k), rb.get(k)
            if v[0] is None and v[1] is None:
                continue
            print(f"  rank{i} {k+'.all_identical':<32} "
                  f"{None if v[0] is None else v[0].get('all_identical')} -> "
                  f"{None if v[1] is None else v[1].get('all_identical')}")
            print(f"  rank{i} {k+'.distinct_outputs':<32} "
                  f"{None if v[0] is None else v[0].get('distinct_outputs')} -> "
                  f"{None if v[1] is None else v[1].get('distinct_outputs')}")

for tag in (A, B):
    p = os.path.join(OUT, f"{tag}.run.log")
    txt = open(p, errors="replace").read() if os.path.exists(p) else ""
    print(f"\n=== {tag} log invariants ===")
    for label, pat in (("digest", r"digest=([0-9a-f]+)"),
                       ("regions", r"regions=(\d+\(forecast=\d+\))"),
                       ("reserved", r"reserved=([0-9.]+ GiB)"),
                       ("fallbacks", r"torch_fallbacks?[= ](\d+)"),
                       ("selftest", r"(selftest[^\n]{0,90})")):
        g = sorted(set(re.findall(pat, txt)))
        if g:
            print(f"  {label:<10} {g[:4]}")
