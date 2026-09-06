"""Join a boot's per-chunk rows against the host's 1 Hz box sample, and print the two ranks side by
side on ONE wall clock.

The question it answers: when a chunk row costs 14 s instead of 1.8 s, was the box reclaiming?
Usage:  python3 tools/offload/boot_join.py <outdir> <TAG>
"""

from __future__ import annotations

import json
import os
import sys

PAGE = 4096
GIB = 1 << 30


def load_rows(path):
    d = json.load(open(path))
    return d


def main() -> None:
    outdir, tag = sys.argv[1], sys.argv[2]
    r0 = load_rows(os.path.join(outdir, f"{tag}.boot.rank0.json"))
    r1p = os.path.join(outdir, f"{tag}.boot.rank1.json")
    r1 = load_rows(r1p) if os.path.exists(r1p) else None

    samp = []
    sp = os.path.join(outdir, f"{tag}.box_sampler.jsonl")
    if os.path.exists(sp):
        with open(sp) as fh:
            next(fh, None)
            for line in fh:
                try:
                    samp.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    samp.sort(key=lambda s: s["wall"])

    def window(t0, t1):
        """Reclaim work the BOX did in (t0, t1]: swap pages in/out, direct-reclaim scans."""
        sel = [s for s in samp if t0 < s["wall"] <= t1]
        if len(sel) < 2:
            return None
        a, b = sel[0], sel[-1]
        return {
            "swin_gib": (b["pswpin"] - a["pswpin"]) * PAGE / GIB,
            "swout_gib": (b["pswpout"] - a["pswpout"]) * PAGE / GIB,
            "majflt_m": (b["pgmajfault"] - a["pgmajfault"]) / 1e6,
            "direct_m": (b["pgscan_direct"] - a["pgscan_direct"]) / 1e6,
            "kswapd_m": (b["pgscan_kswapd"] - a["pgscan_kswapd"]) / 1e6,
            "avail_gib": b["MemAvailable"] / GIB,
            "arc_gib": b["arc"] / GIB,
            "load": b.get("load1", 0.0),
        }

    print(f"total rank0={r0['total_seconds']}s" + (f"  rank1={r1['total_seconds']}s" if r1 else ""))
    print("buckets rank0:")
    for k, v in sorted(r0["buckets_seconds"].items(), key=lambda kv: -kv[1])[:12]:
        print(f"   {k:32s} {v:9.3f}")
    if r1:
        print("buckets rank1:")
        for k, v in sorted(r1["buckets_seconds"].items(), key=lambda kv: -kv[1])[:12]:
            print(f"   {k:32s} {v:9.3f}")
    print("phases rank0:")
    for p in r0["phases"]:
        print(f"   {' ' * p['depth']}{p['name']:32s} {p['seconds']:9.3f}")

    rows0 = r0["rows"]
    rows1 = r1["rows"] if r1 else [None] * len(rows0)
    print()
    hdr = (
        f"{'chunk':22s} {'s0':>7s} {'s1':>7s} {'h2d0':>7s} {'h2d1':>7s} {'shd0':>6s} "
        f"{'swin':>6s} {'swout':>6s} {'majfM':>6s} {'dirM':>6s} {'availG':>7s} {'arcG':>6s} {'ld':>5s}"
    )
    print(hdr)
    for a, b in zip(rows0, rows1):
        w = None
        if "t_end_wall" in a:
            w = window(a["t_end_wall"] - a["seconds"], a["t_end_wall"])
        s = a["split"]
        sb = b["split"] if b else {}
        cells = (
            f"{a['name'][:22]:22s} {a['seconds']:7.2f} {(b['seconds'] if b else 0):7.2f} "
            f"{s.get('ckpt.h2d', 0):7.2f} {sb.get('ckpt.h2d', 0):7.2f} {s.get('ckpt.shard', 0):6.2f} "
        )
        if w:
            cells += (
                f"{w['swin_gib']:6.2f} {w['swout_gib']:6.2f} {w['majflt_m']:6.2f} "
                f"{w['direct_m']:6.2f} {w['avail_gib']:7.2f} {w['arc_gib']:6.2f} {w['load']:5.1f}"
            )
        print(cells)


if __name__ == "__main__":
    main()
