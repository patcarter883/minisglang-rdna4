#!/usr/bin/env python3
"""MEASURE the largest pinned host arena ONE RANK can take on this box, and the failure mode past it.

WHY THIS EXISTS. Every number the weight-offload capacity policy quotes for a per-rank ceiling comes
from P3b — 34.0 GiB on an *idle* box with *no engine loaded*, using `hipMemCreate(location=Host)`.
The shipping arena does not use that call: it uses `hipHostMalloc` through
`weights/pinned_arena.py`, in `MINISGL_WEIGHT_ARENA_CHUNK_MIB`-sized chunks, behind a `MemAvailable`
floor and a swap tripwire. Nobody had ever swept THAT path. `26 GiB` — the figure the M1 work
carried — is simply the largest size anyone happened to try; it is not a measured ceiling, and a
plan sized against a guess is a plan that either wastes host RAM or dies mid-boot.

WHAT IT MEASURES, per size in the sweep:
  * whether `PinnedWeightArena.reserve() + attach()` completes,
  * `pswpout` delta across the pin (pages the kernel evicted while we pinned),
  * `MemAvailable` before/after and its minimum across the chunk loop,
  * wall time per chunk and the derived pin rate, so DEGRADATION is visible as a rate collapse
    rather than only as an outright failure,
  * which card the arena was bound to, because card 1's root port is Gen4 x8 (14.48 GB/s vs card
    0's 28.93) and the first-touch fill runs over it.

DEGRADATION, not just failure, is the verdict this probe exists to produce. `hipHostMalloc`
succeeding is a weak signal: the box can keep saying yes while every chunk costs ten times what the
first one did, which on a shared box is indistinguishable from a hang to everyone else on it. So the
sweep records the per-chunk rate profile at every size and the caller reads the knee, and it stops
climbing on the FIRST of: a raise, a rate collapse past `--rate-collapse`, or the `MemAvailable`
floor — whichever comes first.

SAFETY. Each size is a SEPARATE PROCESS (`--_worker`), so a failed size releases every pinned byte
back to the box by process exit rather than by unwinding a half-built arena — the whole point being
that this probe must not be the thing that wedges the box. The parent samples `/proc` between sizes
and waits for `MemAvailable` to recover before the next one.

Not wrapped in `gpu-lease`: the lease is waived for this task. It touches ONE card (default 0) and
must never run concurrently with another GPU job.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

GIB = 1 << 30


def _meminfo() -> dict:
    out = {}
    with open("/proc/meminfo") as fh:
        for line in fh:
            k, _, v = line.partition(":")
            try:
                out[k] = int(v.strip().split()[0]) * 1024
            except (ValueError, IndexError):
                pass
    return out


def _pswpout() -> int:
    with open("/proc/vmstat") as fh:
        for line in fh:
            if line.startswith("pswpout "):
                return int(line.split()[1])
    return 0


def _gib(n) -> float:
    return round(n / GIB, 3)


# ---------------------------------------------------------------------------
# worker: one size, one process
# ---------------------------------------------------------------------------


def worker(args) -> int:
    sys.path.insert(0, "/engine/python")
    from minisgl.weights.chunk_plan import RegionRequest
    from minisgl.weights.pinned_arena import PinnedWeightArena

    target = int(args.gib * GIB)
    chunk = int(args.chunk_mib) * (1 << 20)
    # One region per chunk-sized slab. The arena's own planner then lays them out one-per-chunk,
    # which is the same shape a real granule plan produces and keeps the reservation exactly the
    # size asked for rather than a rounded-up approximation of it.
    n = max(1, target // chunk)
    # Leave a granule so the region fits inside the chunk after alignment; a region may never
    # straddle, and a region of exactly chunk_bytes forces the planner to grow the chunk.
    region_bytes = chunk - (2 << 20)
    reqs = [RegionRequest(f"slab{i}", region_bytes) for i in range(n)]

    rec: dict = {
        "target_gib": args.gib,
        "chunk_mib": args.chunk_mib,
        "n_regions": n,
        "device_index": args.device,
        "mem_available_before": _meminfo().get("MemAvailable", 0),
        "pswpout_before": _pswpout(),
    }
    arena = PinnedWeightArena(
        device_index=args.device,
        rank=0,
        local_ranks=1,
        chunk_bytes=chunk,
        floor_bytes=int(args.floor_gib * GIB),
    )
    t0 = time.perf_counter()
    try:
        plan = arena.reserve(reqs, check=True)
        rec["reserved_gib"] = _gib(plan.reserved_bytes)
        rec["n_chunks"] = plan.n_chunks
        arena.attach(selftest=args.selftest, first_touch=True)
        rec["ok"] = True
    except BaseException as e:  # noqa: BLE001 - the failure MODE is the measurement
        rec["ok"] = False
        rec["error_type"] = type(e).__name__
        rec["error"] = str(e)[:4000]
    rec["wall_s"] = round(time.perf_counter() - t0, 2)
    rec["pinned_gib"] = _gib(arena.pinned_bytes)
    rec["mem_available_after"] = _meminfo().get("MemAvailable", 0)
    rec["pswpout_after"] = _pswpout()
    rec["pswpout_delta_pages"] = rec["pswpout_after"] - rec["pswpout_before"]
    rec["swap_delta_gib"] = _gib(rec["pswpout_delta_pages"] * 4096)
    rec["mem_available_before_gib"] = _gib(rec["mem_available_before"])
    rec["mem_available_after_gib"] = _gib(rec["mem_available_after"])
    # Per-chunk timing: the DEGRADATION signal. `pin_seconds` is the hipHostMalloc alone;
    # `touch_seconds` is the device-issued first-touch fill over PCIe.
    chunks = [
        {
            "i": c.index,
            "pin_s": round(c.pin_seconds, 4),
            "touch_s": round(c.touch_seconds, 4),
            "pin_gb_s": round(c.pin_gb_s, 2) if c.pin_gb_s else None,
        }
        for c in arena.chunks
    ]
    rec["chunks"] = chunks
    if chunks:
        pins = [c["pin_s"] for c in chunks]
        rec["pin_s_first"] = pins[0]
        rec["pin_s_last"] = pins[-1]
        rec["pin_s_max"] = max(pins)
        rec["pin_s_median"] = sorted(pins)[len(pins) // 2]
        rec["pin_rate_collapse"] = (
            round(rec["pin_s_max"] / rec["pin_s_median"], 2) if rec["pin_s_median"] else None
        )
    try:
        rec["card"] = arena.device_identity.get("name")
        rec["pci"] = arena.device_identity.get("pci_bus_id")
    except Exception:
        pass

    print("@@JSON@@" + json.dumps(rec), flush=True)
    # Release explicitly so the parent's post-size MemAvailable sample is meaningful even before
    # the process exits.
    try:
        arena.close(force=True)
    except Exception:
        pass
    return 0 if rec["ok"] else 3


# ---------------------------------------------------------------------------
# parent: the sweep
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sizes", default="8,16,20,24,26,28,30,32,34",
                    help="comma-separated per-rank GiB targets, ascending")
    ap.add_argument("--chunk-mib", type=int, default=3072)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--floor-gib", type=float, default=12.0)
    ap.add_argument("--selftest", action="store_true",
                    help="run the arena's read-back self-test at each size (slow, ~1 chunk of PCIe)")
    ap.add_argument("--rate-collapse", type=float, default=8.0,
                    help="stop climbing when max/median per-chunk pin time exceeds this")
    ap.add_argument("--settle-s", type=float, default=8.0)
    ap.add_argument("--recover-gib", type=float, default=4.0,
                    help="required MemAvailable recovery vs the size's own start before the next")
    ap.add_argument("--timeout-s", type=float, default=900.0)
    ap.add_argument("--json", default="")
    ap.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--gib", type=float, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args._worker:
        return worker(args)

    sizes = [float(s) for s in args.sizes.split(",") if s.strip()]
    results = []
    verdict = "no size attempted"
    for gib in sizes:
        mi = _meminfo()
        avail = mi.get("MemAvailable", 0)
        print(
            f"\n=== target {gib:.1f} GiB | MemAvailable {_gib(avail)} GiB | "
            f"SwapFree {_gib(mi.get('SwapFree', 0))} GiB ===",
            flush=True,
        )
        if avail < (gib + args.floor_gib) * GIB:
            verdict = (
                f"stopped BEFORE {gib:.1f} GiB: MemAvailable {_gib(avail)} GiB cannot cover "
                f"{gib:.1f} + {args.floor_gib:.1f} floor. The box, not the arena, is the limit."
            )
            print(f"  SKIP: {verdict}", flush=True)
            break
        cmd = [
            sys.executable, os.path.abspath(__file__), "--_worker",
            "--gib", str(gib), "--chunk-mib", str(args.chunk_mib),
            "--device", str(args.device), "--floor-gib", str(args.floor_gib),
        ] + (["--selftest"] if args.selftest else [])
        t0 = time.perf_counter()
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout_s)
            stdout, stderr, rc = p.stdout, p.stderr, p.returncode
        except subprocess.TimeoutExpired as e:
            stdout = (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
            stderr = "TIMEOUT"
            rc = -9
        rec = None
        for line in stdout.splitlines():
            if line.startswith("@@JSON@@"):
                rec = json.loads(line[len("@@JSON@@"):])
        if rec is None:
            rec = {"target_gib": gib, "ok": False, "error_type": "no-json",
                   "error": (stderr or stdout)[-3000:], "wall_s": round(time.perf_counter() - t0, 2)}
        rec["rc"] = rc
        results.append(rec)
        print(
            f"  ok={rec.get('ok')} pinned={rec.get('pinned_gib')} GiB "
            f"wall={rec.get('wall_s')}s swap_delta={rec.get('swap_delta_gib')} GiB "
            f"({rec.get('pswpout_delta_pages')} pages) "
            f"pin_s med={rec.get('pin_s_median')} max={rec.get('pin_s_max')} "
            f"collapse={rec.get('pin_rate_collapse')}",
            flush=True,
        )
        if not rec.get("ok"):
            verdict = (
                f"FAILED at {gib:.1f} GiB with {rec.get('error_type')}: "
                f"{str(rec.get('error'))[:300]}"
            )
            print(f"  {verdict}", flush=True)
            break
        collapse = rec.get("pin_rate_collapse")
        if collapse is not None and collapse > args.rate_collapse:
            verdict = (
                f"DEGRADED at {gib:.1f} GiB: per-chunk pin time max/median = {collapse}x "
                f"(> {args.rate_collapse}x). The allocation still SUCCEEDS; it is the rate that "
                f"collapsed, which is the failure mode a success/fail sweep would have missed."
            )
            print(f"  {verdict}", flush=True)
            break
        verdict = f"clean through {gib:.1f} GiB"

        # let the box give the pages back before the next, larger, attempt
        start_avail = rec.get("mem_available_before", 0)
        deadline = time.time() + 60
        while time.time() < deadline:
            time.sleep(args.settle_s)
            now = _meminfo().get("MemAvailable", 0)
            if now >= start_avail - args.recover_gib * GIB:
                break
        print(f"  settled: MemAvailable {_gib(_meminfo().get('MemAvailable', 0))} GiB", flush=True)

    clean = [r for r in results if r.get("ok")]
    out = {
        "sizes": sizes,
        "chunk_mib": args.chunk_mib,
        "device_index": args.device,
        "floor_gib": args.floor_gib,
        "card": (clean[-1].get("card") if clean else None),
        "pci": (clean[-1].get("pci") if clean else None),
        "max_clean_pin_gib": (clean[-1]["pinned_gib"] if clean else 0.0),
        "verdict": verdict,
        "results": results,
    }
    print("\n" + "=" * 78)
    print(f"MAX CLEAN PIN (1 rank, card {args.device}): {out['max_clean_pin_gib']} GiB")
    print(f"VERDICT: {verdict}")
    print("=" * 78)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
