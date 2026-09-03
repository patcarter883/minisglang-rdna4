#!/usr/bin/env python3
"""P3b -- SUPPLEMENTARY to P3.  Not a substitute for it, and not one of the plan's forks.

P3 established that `hipMemCreate(location=hipMemLocationTypeHost)` on this box
(ROCm 7.2.4, gfx1201) returns DEVICE VRAM: the location field is echoed back by
hipMemGetAllocationPropertiesFromHandle and ignored, sysfs mem_info_vram_used tracks the
arena 1:1, host MemAvailable never moves, and the first touch runs at ~303 GB/s against a
28 GB/s PCIe ceiling.  P3 is therefore INVALIDATED as a host-capacity measurement and
emits no fork.

That leaves the design question the fork existed to answer still open:

    can this box give TWO ranks ~34 GiB each of host memory that the GPU can read?

...but by the route that actually works.  This probe measures the surviving routes:

  A) hipHostMalloc (pinned host, device-visible through hipHostGetDevicePointer)
  B) mmap(MAP_SHARED) on a real file + hipHostRegister -- the plan's stated fallback

For each: capacity climb to target or to a MemAvailable floor, allocation rate, the
MemAvailable delta (proving the commit is eager and lands in host RAM), sysfs VRAM delta
(proving it costs no device memory), and a DEVICE-ISSUED write through the device pointer
whose bandwidth must land at PCIe speed -- the same three discriminators P3's gate uses,
because "it returned hipSuccess" proves nothing here.

Every timing records which physical card.  Results are durable, in the worktree.
"""
import ctypes
import json
import mmap
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ["ROCR_VISIBLE_DEVICES"] = "0,1"
os.environ.pop("HIP_VISIBLE_DEVICES", None)

GIB = 1 << 30
MIB = 1 << 20
REPO = Path(__file__).resolve().parents[2]
OUTDIR = REPO / "docs" / "measurements" / "WEIGHT_OFFLOAD_2026-09-02"
# Durable scratch for the file-backed arm.  NOT /tmp: that is tmpfs on this box, which
# would make a "file-backed" arm secretly a RAM arm and the measurement a lie.
SCRATCH = REPO / "tools" / "offload" / "_build" / "p3b_scratch"

PCIE_MAX_GB_S = 100.0     # above this it did not cross PCIe


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def meminfo():
    o = {}
    for line in open("/proc/meminfo"):
        k, v = line.split(":", 1)
        if k in ("MemAvailable", "MemFree", "SwapFree", "Cached"):
            o[k] = int(v.split()[0]) * 1024
    return o


def vmstat_pswpout():
    for line in open("/proc/vmstat"):
        if line.startswith("pswpout "):
            return int(line.split()[1])
    return 0


def gpus():
    out = []
    for dev in sorted(Path("/sys/class/drm").glob("card*/device")):
        if not (dev / "mem_info_vram_total").exists():
            continue
        e = {"sysfs": str(dev)}
        for k, f in (("vram_total", "mem_info_vram_total"), ("vram_used", "mem_info_vram_used"),
                     ("gtt_used", "mem_info_gtt_used")):
            try:
                e[k] = int((dev / f).read_text().strip())
            except (OSError, ValueError):
                e[k] = None
        try:
            for line in (dev / "uevent").read_text().splitlines():
                if line.startswith("PCI_SLOT_NAME="):
                    e["pci_slot"] = line.split("=", 1)[1].strip()
        except OSError:
            pass
        e["discrete"] = bool((e.get("vram_total") or 0) >= 8 * GIB)
        out.append(e)
    return [e for e in out if e["discrete"]]


def box(tag):
    return {"tag": tag, "t": now(), **meminfo(), "pswpout": vmstat_pswpout(),
            "gpus": gpus()}


def hip():
    lib = ctypes.CDLL("libamdhip64.so")
    sigs = {
        "hipSetDevice": [ctypes.c_int],
        "hipDeviceSynchronize": [],
        "hipHostMalloc": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint],
        "hipHostFree": [ctypes.c_void_p],
        "hipHostRegister": [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint],
        "hipHostUnregister": [ctypes.c_void_p],
        "hipHostGetDevicePointer": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p,
                                    ctypes.c_uint],
        "hipMemsetD32": [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t],
        "hipMemcpy": [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int],
        "hipDeviceGetPCIBusId": [ctypes.c_char_p, ctypes.c_int, ctypes.c_int],
        "hipDeviceGetName": [ctypes.c_char_p, ctypes.c_int, ctypes.c_int],
    }
    for n, a in sigs.items():
        getattr(lib, n).argtypes = a
        getattr(lib, n).restype = ctypes.c_int
    return lib


def card_of(lib, dev):
    b = ctypes.create_string_buffer(64)
    lib.hipDeviceGetPCIBusId(b, 64, dev)
    pci = b.value.decode(errors="replace")
    n = ctypes.create_string_buffer(256)
    lib.hipDeviceGetName(n, 256, dev)
    sysfs = next((g for g in gpus()
                  if (g.get("pci_slot") or "").lower() == pci.lower()), None)
    return {"hip_dev": dev, "pci_bus_id": pci, "name": n.value.decode(errors="replace"),
            "sysfs": (sysfs or {}).get("sysfs")}


# ---------------------------------------------------------------- worker ----
def worker(arm, device, target_bytes, chunk_bytes, floor_bytes):
    lib = hip()
    lib.hipSetDevice(device)
    rec = {"arm": arm, "device": device, "card": card_of(lib, device),
           "target_bytes": target_bytes, "chunk_bytes": chunk_bytes,
           "floor_bytes": floor_bytes, "chunks": [], "pid": os.getpid()}
    held = []      # (host_ptr, dev_ptr, mm_or_None, fd_or_None)
    stop = None
    n = target_bytes // chunk_bytes
    SCRATCH.mkdir(parents=True, exist_ok=True)

    def say(o):
        sys.stdout.write(json.dumps(o) + "\n")
        sys.stdout.flush()

    say({"ev": "ready", **{k: rec[k] for k in ("arm", "device", "card")}})

    for i in range(n):
        mi = meminfo()
        if mi["MemAvailable"] < floor_bytes:
            stop = {"reason": "mem_available_floor", "i": i,
                    "mem_available": mi["MemAvailable"]}
            break
        c = {"i": i}
        hp = ctypes.c_void_p()
        mm = fd = None
        t0 = time.perf_counter()
        if arm == "hipHostMalloc":
            rc = lib.hipHostMalloc(ctypes.byref(hp), chunk_bytes, 0)
            if rc:
                stop = {"reason": "hipHostMalloc", "rc": rc, "i": i}
                break
        else:
            path = SCRATCH / f"arena_r{device}_{i:04d}.bin"
            fd = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o600)
            try:
                os.ftruncate(fd, chunk_bytes)
                mm = mmap.mmap(fd, chunk_bytes, flags=mmap.MAP_SHARED,
                               prot=mmap.PROT_READ | mmap.PROT_WRITE)
            except OSError as exc:
                os.close(fd)
                stop = {"reason": "mmap", "errno": str(exc), "i": i}
                break
            hp = ctypes.c_void_p(ctypes.addressof(ctypes.c_char.from_buffer(mm)))
            rc = lib.hipHostRegister(hp, chunk_bytes, 1)   # hipHostRegisterPortable
            if rc:
                stop = {"reason": "hipHostRegister", "rc": rc, "i": i}
                mm.close()
                os.close(fd)
                break
        t1 = time.perf_counter()

        dp = ctypes.c_void_p()
        rc = lib.hipHostGetDevicePointer(ctypes.byref(dp), hp, 0)
        c["rc_devptr"] = rc
        if rc:
            stop = {"reason": "hipHostGetDevicePointer", "rc": rc, "i": i}
            break
        # DEVICE-issued first touch through the device pointer, with a per-chunk
        # fingerprint so a later resweep can catch aliasing.
        word = (0xB1B10000 | (device << 12) | (i & 0x0FFF)) & 0xFFFFFFFF
        t2 = time.perf_counter()
        rc_t = lib.hipMemsetD32(dp, ctypes.c_int(word - (1 << 32) if word >> 31 else word).value,
                                chunk_bytes // 4)
        rc_s = lib.hipDeviceSynchronize()
        t3 = time.perf_counter()
        if rc_t or rc_s:
            stop = {"reason": "device_first_touch", "rc_memset": rc_t, "rc_sync": rc_s, "i": i}
            break
        c.update({
            "alloc_s": t1 - t0, "touch_s": t3 - t2,
            "alloc_gb_s": chunk_bytes / (t1 - t0) / 1e9 if t1 > t0 else None,
            "touch_gb_s": chunk_bytes / (t3 - t2) / 1e9 if t3 > t2 else None,
        })
        held.append((hp, dp, mm, fd, word))
        rec["chunks"].append(c)
        if i % 4 == 0:
            say({"ev": "progress", "i": i, "held_gib": len(held) * chunk_bytes / GIB,
                 "box": box(f"{arm}_r{device}_{i}")})

    rec["stop"] = stop or {"reason": "reached_target", "i": n}
    rec["committed_bytes"] = len(held) * chunk_bytes
    rec["committed_gib"] = rec["committed_bytes"] / GIB
    say({"ev": "peak", "committed_gib": rec["committed_gib"], "box": box(f"{arm}_r{device}_peak")})

    # resweep EVERY chunk: head/mid/tail must return that chunk's own fingerprint
    buf = (ctypes.c_uint32 * 4)()
    fails = []
    for idx, (hp, dp, mm, fd, word) in enumerate(held):
        for off in (0, chunk_bytes // 2, chunk_bytes - 16):
            rc = lib.hipMemcpy(ctypes.cast(buf, ctypes.c_void_p),
                               ctypes.c_void_p(dp.value + off), 16, 2)
            if rc or buf[0] != word:
                fails.append({"chunk": idx, "off": off, "rc": rc,
                              "expected": hex(word),
                              "got": None if rc else hex(buf[0])})
    rec["resweep_chunks"] = len(held)
    rec["resweep_failures"] = len(fails)
    rec["resweep_sample"] = fails[:16]

    for hp, dp, mm, fd, word in held:
        if mm is None:
            lib.hipHostFree(hp)
        else:
            lib.hipHostUnregister(hp)
            mm.close()
            os.close(fd)
    if arm != "hipHostMalloc":
        for p in SCRATCH.glob(f"arena_r{device}_*.bin"):
            try:
                p.unlink()
            except OSError:
                pass
    say({"ev": "done", **rec})
    return 0


def summarize(vals):
    vs = sorted(v for v in vals if v is not None)
    if not vs:
        return {"n": 0}
    import statistics
    q = statistics.quantiles(vs, n=4) if len(vs) > 3 else [vs[0], statistics.median(vs), vs[-1]]
    return {"n": len(vs), "median": statistics.median(vs), "p25": q[0], "p75": q[2],
            "min": vs[0], "max": vs[-1]}


def run_arm(arm, ranks, target_gib, chunk_mib, floor_gib, timeout):
    procs, results = [], []
    b0 = box(f"{arm}_{ranks}rank_before")
    for r in range(ranks):
        procs.append(subprocess.Popen(
            [sys.executable, os.path.abspath(__file__), "--_worker", arm, str(r),
             str(int(target_gib * GIB)), str(chunk_mib * MIB), str(int(floor_gib * GIB))],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
    peak = None
    for p in procs:
        try:
            out, err = p.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            p.kill()
            out, err = p.communicate()
        lines = [json.loads(l) for l in out.splitlines() if l.startswith("{")]
        done = next((l for l in reversed(lines) if l.get("ev") == "done"), None)
        pk = next((l for l in reversed(lines) if l.get("ev") == "peak"), None)
        if pk and (peak is None or pk["committed_gib"] > peak.get("committed_gib", 0)):
            peak = pk
        results.append({"returncode": p.returncode, "done": done,
                        "progress_boxes": [l["box"] for l in lines if l.get("ev") == "progress"],
                        "peak_box": (pk or {}).get("box"),
                        "stderr_tail": (err or "").strip()[-600:],
                        "faulted": "Memory access fault" in (err or "")})
    b1 = box(f"{arm}_{ranks}rank_after")
    per = [(r["done"] or {}).get("committed_bytes", 0) for r in results]
    alloc = [c.get("alloc_gb_s") for r in results for c in ((r["done"] or {}).get("chunks") or [])]
    touch = [c.get("touch_gb_s") for r in results for c in ((r["done"] or {}).get("chunks") or [])]
    tsum = summarize(touch)
    vram_growth = None
    try:
        a = [g["vram_used"] for g in b0["gpus"]]
        b = [g["vram_used"] for g in (peak or {}).get("box", b1)["gpus"]]
        vram_growth = max(y - x for x, y in zip(a, b))
    except Exception:
        pass
    peak_box = (peak or {}).get("box") or b1
    host_delta = b0["MemAvailable"] - peak_box["MemAvailable"]
    cached_delta = peak_box["Cached"] - b0["Cached"]
    committed = sum(per)
    return {
        "arm": arm, "ranks": ranks, "target_gib_per_rank": target_gib,
        "chunk_mib": chunk_mib, "floor_gib": floor_gib,
        "committed_bytes_total": committed, "committed_gib_total": committed / GIB,
        "committed_gib_per_rank": [p / GIB for p in per],
        "committed_gib_per_rank_min": min(per) / GIB if per else 0,
        "stops": [(r["done"] or {}).get("stop") for r in results],
        "resweep_failures": sum((r["done"] or {}).get("resweep_failures", 0) for r in results),
        "resweep_chunks": sum((r["done"] or {}).get("resweep_chunks", 0) for r in results),
        "cards": [(r["done"] or {}).get("card") for r in results],
        "alloc_gb_s": summarize(alloc), "device_touch_gb_s": tsum,
        "host_mem_available_delta_bytes": host_delta,
        "host_commit_accounted_frac": (host_delta / committed) if committed else None,
        "page_cache_delta_bytes": cached_delta,
        "page_cache_accounted_frac": (cached_delta / committed) if committed else None,
        # MemAvailable is the right meter for PINNED pages (unreclaimable, so it drops
        # ~1:1) but the WRONG one for file-backed pages: page cache is reclaimable, so
        # MemAvailable barely moves and the rise shows up in Cached instead. Scoring both
        # arms on MemAvailable produced a false "not host memory" for the mmap arm.
        "host_accounting_meter": ("MemAvailable" if arm == "hipHostMalloc" else "Cached"),
        "host_accounting_frac": ((host_delta / committed) if arm == "hipHostMalloc"
                                 else (cached_delta / committed)) if committed else None,
        "vram_growth_bytes": vram_growth,
        "vram_growth_frac_of_committed": (vram_growth / committed)
                                         if (committed and vram_growth is not None) else None,
        # Absolute floor as well as a fraction: on a shared box a couple of hundred MB of
        # ambient VRAM movement is normal, and at small arena sizes that alone tripped the
        # fraction test.
        "vram_tolerance_bytes": max(256 * MIB, int(0.03 * committed)),
        "vram_growth_within_tolerance": (vram_growth is None
                                         or vram_growth < max(256 * MIB, int(0.03 * committed))),
        "device_touch_is_pcie_speed": (tsum.get("median") is not None
                                       and tsum["median"] < PCIE_MAX_GB_S),
        # The three discriminators that hold for BOTH arms: the data survives, the device
        # reaches it at PCIe speed (so it is across the bus, i.e. not VRAM), and it cost
        # no device memory.
        "is_really_host_memory": bool(
            committed
            and sum((r["done"] or {}).get("resweep_failures", 0) for r in results) == 0
            and tsum.get("median") is not None and tsum["median"] < PCIE_MAX_GB_S
            and (vram_growth is None
                 or vram_growth < max(256 * MIB, int(0.03 * committed)))),
        "box_before": b0, "box_after": b1, "box_at_peak": (peak or {}).get("box"),
        "workers": results,
    }


def main():
    import argparse
    ap = argparse.ArgumentParser(description="P3b -- supplementary host-memory fallback capacity")
    ap.add_argument("--target-gib", type=float, default=34.0)
    # The scratch filesystem here is ZFS.  A full 2x34 GiB file-backed arena would push
    # 68 GiB through the ARC and the pool, which measures ZFS writeback rather than the
    # mapping route.  The mmap arm's job is viability + bandwidth, not capacity, so it
    # runs small by default; capacity is answered by the hipHostMalloc arm.
    ap.add_argument("--mmap-target-gib", type=float, default=8.0)
    ap.add_argument("--chunk-mib", type=int, default=2048)
    ap.add_argument("--floor-gib", type=float, default=12.0)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--settle-s", type=float, default=8.0)
    ap.add_argument("--arms", default="",
                    help="comma-separated subset, e.g. 'hipHostMalloc' or "
                         "'mmap_shared_hipHostRegister:1'")
    ap.add_argument("--outdir", default=str(OUTDIR))
    a = ap.parse_args()
    out = {"probe": "P3b", "schema_version": 1, "started_utc": now(),
           "title": "P3b supplementary: can the SURVIVING host-memory routes reach 34 GiB x 2?",
           "relation_to_p3": ("P3 is INVALIDATED: hipMemCreate(location=Host) returns device "
                              "VRAM on this box. P3b measures the routes that do work, so the "
                              "design question P3's fork existed to answer is still answered. "
                              "P3b is NOT a P3 fork and must not be quoted as one."),
           "args": vars(a), "baseline": box("baseline"), "arms": []}

    od = Path(a.outdir)
    od.mkdir(parents=True, exist_ok=True)
    jpath = od / "p3b.json"

    def flush():
        """Write after EVERY arm. The mmap arm on ZFS is slow enough to outlive a harness
        timeout, and an arm that ran but left no artifact is a wasted measurement."""
        out["finished_utc"] = now()
        jpath.write_text(json.dumps(out, indent=2))

    plan = [("hipHostMalloc", 1), ("hipHostMalloc", 2),
            ("mmap_shared_hipHostRegister", 1), ("mmap_shared_hipHostRegister", 2)]
    if a.arms:
        want = set(a.arms.split(","))
        plan = [(arm, r) for arm, r in plan if f"{arm}:{r}" in want or arm in want]
    flush()
    for arm, ranks in plan:
        tgt = a.target_gib if arm == "hipHostMalloc" else a.mmap_target_gib
        print(f"[p3b] arm={arm} ranks={ranks} target={tgt} GiB/rank "
              f"floor={a.floor_gib} GiB", file=sys.stderr)
        r = run_arm(arm, ranks, tgt, a.chunk_mib, a.floor_gib, a.timeout)
        out["arms"].append(r)
        print(f"[p3b]   committed={r['committed_gib_total']:.1f} GiB total "
              f"(per rank {['%.1f' % x for x in r['committed_gib_per_rank']]}) "
              f"stops={r['stops']} resweep_fails={r['resweep_failures']} "
              f"alloc={r['alloc_gb_s'].get('median')} GB/s "
              f"devtouch={r['device_touch_gb_s'].get('median')} GB/s "
              f"really_host={r['is_really_host_memory']}", file=sys.stderr)
        flush()
        time.sleep(a.settle_s)

    out["final"] = box("final")
    flush()
    print(f"[p3b] wrote {jpath}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--_worker":
        sys.exit(worker(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]),
                        int(sys.argv[5]), int(sys.argv[6])))
    sys.exit(main())
