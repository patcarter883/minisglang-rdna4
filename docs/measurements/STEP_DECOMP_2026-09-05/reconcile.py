#!/usr/bin/env python3
"""Reconcile step_decomp.py outputs into a decomposition that SUMS, plus the marginal host layer.

Reads one or more step_decomp JSONs. The operating-point run (mode=full) supplies the breakdown;
every run supplies (host_layers, wall_ms_per_step) for the regression.
"""
import json
import sys


def rank_pick(d):
    """The GATING rank. TP=2 runs in lockstep, so the slower rank sets the step."""
    rs = [r for r in d["ranks"] if "captured" in r and "wall_ms_per_step" in r.get("captured", {})]
    if not rs:
        return None
    return max(rs, key=lambda r: r["captured"]["wall_ms_per_step"])


def main(paths):
    runs = []
    for p in paths:
        d = json.load(open(p))
        r = rank_pick(d)
        if r is None:
            print(f"!! {p}: no usable rank ({[x.get('error','?')[:120] for x in d['ranks']]})")
            continue
        runs.append((p, d, r))

    print("=" * 100)
    print("RUNS")
    print("=" * 100)
    for p, d, r in runs:
        c, e = r["captured"], r["eager"]
        print(f"{p.split('/')[-1]:34s} dev={r['device_layers']:2d} host={r['host_layers']:2d} "
              f"card={r['card'][-12:]:12s} rank={r['rank']} "
              f"cap_wall={c['wall_ms_per_step']:7.3f} dev_busy={c.get('device_busy_ms_per_step', float('nan')):7.3f} "
              f"eag_wall={e['wall_ms_per_step']:7.3f}")
        for pp, dd, rr in [(p, d, x) for x in d["ranks"] if x is not r and "captured" in x]:
            cc = rr["captured"]
            print(f"{'  (other rank ' + str(rr['rank']) + ')':34s} "
                  f"card={rr['card'][-12:]:12s} "
                  f"cap_wall={cc['wall_ms_per_step']:7.3f} "
                  f"dev_busy={cc.get('device_busy_ms_per_step', float('nan')):7.3f}")

    # ---- marginal host layer -----------------------------------------------------------------
    pts = sorted({(r["host_layers"], r["captured"]["wall_ms_per_step"]) for _p, _d, r in runs})
    print()
    print("=" * 100)
    print("MARGINAL HOST LAYER  d(step)/d(host layer)")
    print("=" * 100)
    for h, w in pts:
        print(f"  host_layers={h:2d}  wall={w:7.3f} ms/step")
    slopes = []
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            dh = pts[j][0] - pts[i][0]
            if dh:
                s = (pts[j][1] - pts[i][1]) / dh
                slopes.append(s)
                print(f"  {pts[i][0]:2d}->{pts[j][0]:2d} layers: {s:+.4f} ms per host layer")
    if slopes:
        m = sum(slopes) / len(slopes)
        print(f"  mean marginal = {m:+.4f} ms/host-layer")
        r0 = runs[0][2]
        b = r0.get("host_active_bytes_per_token_per_rank", 0) / max(r0["host_layers"], 1)
        if b and m > 0:
            print(f"  bytes/host-layer/rank/token = {b:,.0f}")
            print(f"  implied bandwidth = {b / (m * 1e-3) / 1e9:.2f} GB/s "
                  f"(card1 Gen4 x8 measured ceiling 14.48; card0 Gen5 x8 28.93)")

    # ---- the breakdown, from the full run ------------------------------------------------------
    full = [(p, d, r) for p, d, r in runs if r.get("mode") == "full"]
    if not full:
        return 0
    p, d, r = full[0]
    c, e = r["captured"], r["eager"]
    W = c["wall_ms_per_step"]
    print()
    print("=" * 100)
    print(f"LEVEL 1 — partition of the {W:.3f} ms CAPTURED step (wall, difference method)")
    print("=" * 100)
    s = c["stage_ms_per_step"]
    tot = 0.0
    for k in ("recv", "sched", "fwd_launch", "gpu_wait", "commit"):
        tot += s[k]
        print(f"  {k:14s} {s[k]:8.3f} ms  {100*s[k]/W:5.1f}%")
    print(f"  {'loop_residual':14s} {c['loop_residual_ms_per_step']:8.3f} ms  "
          f"{100*c['loop_residual_ms_per_step']/W:5.1f}%   (generate/run_forever outside the 5 stages)")
    print(f"  {'SUM':14s} {tot + c['loop_residual_ms_per_step']:8.3f} ms  vs wall {W:.3f}")
    db = c.get("device_busy_ms_per_step")
    if db:
        print()
        print(f"  device_busy (HIP event pair around forward_batch) {db:8.3f} ms  {100*db/W:5.1f}%")
        print(f"  gpu_idle    (wall - device_busy)                  {W-db:8.3f} ms  {100*(W-db)/W:5.1f}%")
        print(f"  gpu_wait vs device_busy (independent instruments): "
              f"{s['gpu_wait']:.3f} vs {db:.3f}  -> delta {s['gpu_wait']-db:+.3f} ms")
        print(f"  host-only stages that CANNOT overlap the device (recv+sched+commit): "
              f"{c['host_only_stages_ms']:.3f} ms")

    print()
    print("=" * 100)
    print("LEVEL 2 — inside device_busy (eager HIP-event regions)")
    print("=" * 100)
    g = r.get("region_ms_per_step_eager", {})
    k = r.get("capture_factor_device")
    print(f"  eager wall {e['wall_ms_per_step']:.3f} ms/step, eager device_busy "
          f"{e.get('device_busy_ms_per_step', float('nan')):.3f} ms/step")
    print(f"  capture factor: wall {r.get('capture_factor_wall')}  device {k}")
    print(f"  instrument overhead {r.get('instrument_overhead_ms_per_step')} ms/step "
          f"(wrapped_off {r.get('eager_wrapped_off_ms_per_step')} -> "
          f"wrapped_on {r.get('eager_wrapped_on_ms_per_step')})")
    print()
    leaves = [x for x in g if not x.startswith("MODEL_TOTAL") and not x.startswith("moe.total")]
    ssum = sum(g[x] for x in leaves)
    print(f"  {'region':26s} {'eager ms':>9s} {'/step':>7s} {'scaled':>8s}")
    for kk in sorted(leaves, key=lambda x: -g[x]):
        sc = g[kk] / k if k else float('nan')
        print(f"  {kk:26s} {g[kk]:9.3f} {r.get('regions_per_step',{}).get(kk,0):7.1f} {sc:8.3f}")
    print(f"  {'LEAF SUM':26s} {ssum:9.3f}")
    print(f"  {'MODEL_TOTAL':26s} {g.get('MODEL_TOTAL', float('nan')):9.3f}")
    print(f"  {'unattributed in model':26s} "
          f"{g.get('MODEL_TOTAL', 0) - sum(g[x] for x in leaves if x != 'sampler'):9.3f}")

    print()
    print("=" * 100)
    print("LEVEL 1b — inside sched/commit (pure host wall, captured leg)")
    print("=" * 100)
    hs = r.get("host_substages", {})
    print(f"  wrapper overhead {hs.get('wrapper_overhead_ms_per_step')} ms/step "
          f"(off {hs.get('wall_wrapped_off')} -> on {hs.get('wall_wrapped_on')})")
    for kk, vv in (hs.get("site_ms_per_step") or {}).items():
        print(f"  {kk:42s} {vv:8.4f} ms  x{hs.get('site_calls_per_step', {}).get(kk, 0)}")

    print()
    print("=" * 100)
    print("LEVEL 3 — the host layer, four independent estimators")
    print("=" * 100)
    ab = r.get("ablation", {})
    if ab.get("host_points"):
        print("  (a) IN-BOOT ABLATION, eager wall slope (removes the routed-expert call):")
        for p in ab["host_points"]:
            print(f"        host ablated {p['ablated']:2d} -> {p['ms_per_step']} ms/step")
        for p in ab.get("device_points", []):
            print(f"        dev  ablated {p['ablated']:2d} -> {p['ms_per_step']} ms/step")
        print(f"      host {ab.get('ms_per_host_layer_eager')} ms/layer, "
              f"device {ab.get('ms_per_device_layer_eager')} ms/layer, "
              f"PREMIUM {ab.get('host_premium_ms_per_layer_eager')} ms/layer")
    print(f"  (b) IN-SITU REGION (eager device time): host/layer "
          f"{r.get('moe_routed_ms_per_host_layer_eager')}, dev/layer "
          f"{r.get('moe_routed_ms_per_device_layer_eager')}, premium "
          f"{(r.get('moe_routed_ms_per_host_layer_eager') or 0) - (r.get('moe_routed_ms_per_device_layer_eager') or 0):.4f}")
    iso = r.get("isolate", {})
    print(f"  (c) ISOLATION REPLAY (live weights, 1 event pair per pass): host/layer "
          f"{iso.get('experts_ms_per_host_layer')}, dev/layer {iso.get('experts_ms_per_dev_layer')}, "
          f"premium {(iso.get('experts_ms_per_host_layer') or 0) - (iso.get('experts_ms_per_dev_layer') or 0):.4f}")
    if len(pts) > 1:
        print(f"  (d) CROSS-BOOT PLANNER SWEEP: {slopes and round(sum(slopes)/len(slopes),4)} ms/layer")
    else:
        print("  (d) CROSS-BOOT PLANNER SWEEP: NOT AVAILABLE (only one boot fit in RAM)")

    b = r.get("host_active_bytes_per_token_per_rank", 0) / max(r["host_layers"], 1)
    print(f"\n  bytes read per host layer per rank per token: {b:,.0f}")
    for name, v in (("ablation total", ab.get("ms_per_host_layer_eager")),
                    ("ablation premium", ab.get("host_premium_ms_per_layer_eager")),
                    ("region total", r.get("moe_routed_ms_per_host_layer_eager")),
                    ("isolate total", iso.get("experts_ms_per_host_layer"))):
        if b and v:
            print(f"    implied GB/s from {name:18s} {b/(v*1e-3)/1e9:7.2f}   "
                  f"(card1 Gen4x8 = 14.48, card0 Gen5x8 = 28.93)")

    print()
    print("=" * 100)
    print("CROSS-CHECK: the all-reduce line is bandwidth or SKEW?")
    print("=" * 100)
    ar = r.get("synthetic_all_reduce", {})
    n_ar = (r.get("regions_per_step") or {}).get("moe.all_reduce", 0)
    reg = (g.get("moe.all_reduce") or 0)
    if n_ar and ar.get("ms_per_call"):
        print(f"  in-step:   {reg:.3f} ms over {n_ar:.0f} calls = {reg/n_ar:.4f} ms/call")
        print(f"  synthetic: {ar['ms_per_call']:.4f} ms/call at shape {ar.get('shape')} "
              f"({ar.get('reps')} reps, both ranks in lockstep)")
        print(f"  => {reg - n_ar*ar['ms_per_call']:.3f} ms/step of the all-reduce line is CROSS-RANK "
              f"SKEW absorbed at the collective, not collective cost "
              f"({100*(1 - ar['ms_per_call']*n_ar/reg):.0f}% of the line)")

    print()
    print("CROSS-CHECK: hyper-connections, region sum vs one-pair isolation replay")
    hc_reg = sum(g.get(x, 0) for x in ("hc.mix", "hc.combine", "hc.final_mix"))
    n_hc = sum((r.get("regions_per_step") or {}).get(x, 0)
               for x in ("hc.mix", "hc.combine", "hc.final_mix"))
    print(f"  region sum {hc_reg:.3f} ms over {n_hc:.0f} event-paired regions/step")
    print(f"  isolate hc_full {iso.get('hc_full_ms')} ms for {iso.get('hc_blocks')} blocks "
          f"(2 boundaries total)")
    if iso.get("hc_full_ms"):
        print(f"  => {hc_reg - iso['hc_full_ms']:+.3f} ms of the region figure is the instrument's "
              f"own per-boundary cost ({n_hc:.0f} pairs)")

    print()
    print("  sample text:", (r.get("sample_text") or "")[:300].replace("\n", " "))
    print("  loadavg during the captured leg:", c.get("loadavg_1m"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
