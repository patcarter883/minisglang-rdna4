#!/usr/bin/env python3
"""P1 -- kernel-read bandwidth from HOST-located pages behind a DEVICE VA (gfx1201).

Spec: docs/WEIGHT_OFFLOAD_PLAN.md section 3, row P1, and section 11 items 1 and 6.

WHAT THIS ANSWERS
  The weight-offload plan's entire ceiling multiplies through one number: how
  fast can a KERNEL read weights that live on host-located physical pages
  mapped behind a device virtual address (hipMemAddressReserve +
  hipMemCreate(location=Host) + hipMemMap + hipMemSetAccess)?

  KILL CRITERION (plan section 3): if that read is below 0.5x the copy-engine
  figure (~13 GB/s absolute), the placement design's ceiling falls under
  llama.cpp and tiers T1/T2 are dead.

  The prior 127.2 GB/s from vmm_probe2.py is WITHDRAWN: it was a 2 MiB
  L2-resident artifact.  This probe uses a >= 256 MB working set (default
  512 MiB, 8x the 64 MB MALL) so nothing can be cache-resident, and it
  checksums every read so a bandwidth number can never be printed for a read
  that did not actually land on the pages under test.

PLACEMENT PROVENANCE -- why this probe is not just a bandwidth loop
  hipMemCreate(location.type=hipMemLocationTypeHost) can return hipSuccess,
  hipMemMap and hipMemSetAccess can both succeed, a kernel can read the VA
  correctly -- and the pages can still be in VRAM.  A bandwidth number taken
  from that arm is an HBM figure wearing a host-memory label, and it would
  hand the plan a fabricated ceiling.  So every candidate region is CLASSIFIED
  before any number is allowed into the verdict:
    * PRIMARY: the per-card amdgpu counters for this exact PCI BDF
      (/sys/class/drm/cardN/device/mem_info_vram_used and mem_info_gtt_used),
      sampled across backing AND again after a full device-side touch (drivers
      commit pages lazily).  Host-backed pages move gtt_used; VRAM pages move
      vram_used.  Unlike /proc/meminfo these cannot be perturbed by the other
      agents sharing this box.
    * CROSS-CHECK: hipMemGetInfo free-VRAM and /proc/meminfo MemAvailable
      deltas, always recorded so a disagreement is visible.
    * hipPointerGetAttributes recorded but never trusted alone.
    * PHYSICS, both directions.  Bytes that crossed PCIe cannot materially
      outrun the measured copy engine (>1.5x) nor approach HBM (>=0.25x the
      measured HBM reference): a candidate breaching either bound is
      reclassified device-resident whatever the counters said.  Run forward,
      the same physics UPGRADES a region that is provably not in VRAM and
      reads at PCIe speed to host-resident, so a noisy MemAvailable cannot
      demote the plan's own mechanism to "unproven".
    * a CONTROL: the known-device VMM region must classify device_resident, or
      the method itself is declared unreliable and every placement claim voids.
  Only a mechanism with PROVEN host-resident pages can answer P1.  If the
  plan's nominal mechanism fails that test, the verdict is computed from the
  substitute (hipHostMalloc zero-copy) and says so loudly, because the
  substitute cannot give T2's mixed device/host single VA stack.

TIMING PROVENANCE
  Every config is hipEvent-timed AND cross-checked against the wall clock over
  the whole rep loop.  Wall time strictly exceeds device time, so event-derived
  GB/s may exceed wall-derived GB/s only by the fixed per-rep host overhead; a
  ratio beyond TIMING_SANITY_FACTOR means the TIMER produced the number, not
  the kernel.  Such configs are excluded from `best` and raise a blocking
  error, because a garbage hipEventElapsedTime is otherwise indistinguishable
  from a spectacular result (the checksum still passes).

ARMS
  vmm_host          kernel read, host-located VMM pages behind a device VA   <-- THE ANSWER
  vmm_host_alt      same, via HostNuma/HostNumaCurrent, only if that variant
                    is verified host-resident when the nominal one is not
  vmm_dev           kernel read, device-located VMM pages (same mechanism, HBM reference
                    AND the control for the placement method)
  dev_plain         kernel read, ordinary hipMalloc (HBM reference, no VMM)
  pinned_default    kernel read, hipHostMalloc(Mapped) + hipHostGetDevicePointer
  pinned_noncoh     kernel read, hipHostMalloc(Mapped|NonCoherent) + device pointer
  copy engine       hipMemcpyAsync H2D from ordinary pinned host memory, plus
                    hipMemcpyAsync from the identical host-located VMM VA
  coherence         CPU-write -> kernel-read (and the reverse), per 128 B line,
                    to settle plan section 11 item 6

PATTERNS (each covers a prefix of the buffer, every covered byte exactly once)
  linear     grid-stride uint4 stream
  tiled      contiguous `seg` bursts walked TRANSPOSED across `stride` -- the
             grouped-GEMM's K-walk over an N-major packed expert row
  random128  bijective (odd-multiplier mod 2^k) permutation of 128 B lines

USAGE
  ROCR_VISIBLE_DEVICES=0,1 python3 tools/offload/p1_host_read_bw.py --devices 0,1
  python3 tools/offload/p1_host_read_bw.py --selftest      # no GPU work at all

The script sets ROCR_VISIBLE_DEVICES=0,1 and unsets HIP_VISIBLE_DEVICES itself
(re-exec'ing if needed) so the Ryzen iGPU (ROCm device 2, 47 GB of GTT) can
never enter enumeration and poison a "biggest free pool" decision.
"""

from __future__ import annotations

import os
import sys

# ---------------------------------------------------------------------------
# Device-visibility fence.  MUST happen before libamdhip64 is loaded.
# ---------------------------------------------------------------------------
REQUIRED_ROCR = "0,1"
_REEXEC_FLAG = "_P1_REEXEC"


def _fence_device_visibility() -> dict:
    """Force ROCR_VISIBLE_DEVICES=0,1 with HIP_VISIBLE_DEVICES unset."""
    inherited = {
        "ROCR_VISIBLE_DEVICES": os.environ.get("ROCR_VISIBLE_DEVICES"),
        "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES"),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "GPU_DEVICE_ORDINAL": os.environ.get("GPU_DEVICE_ORDINAL"),
    }
    bad = (
        inherited["ROCR_VISIBLE_DEVICES"] != REQUIRED_ROCR
        or inherited["HIP_VISIBLE_DEVICES"] is not None
        or inherited["CUDA_VISIBLE_DEVICES"] is not None
        or inherited["GPU_DEVICE_ORDINAL"] is not None
    )
    if bad and os.environ.get(_REEXEC_FLAG) != "1":
        env = dict(os.environ)
        # Preserve what we were ACTUALLY launched with, so the artifact records
        # the operator's environment and not our own correction of it.
        env["_P1_ORIG_VIS"] = repr(inherited)
        env["ROCR_VISIBLE_DEVICES"] = REQUIRED_ROCR
        for k in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL"):
            env.pop(k, None)
        env[_REEXEC_FLAG] = "1"
        os.execve(sys.executable, [sys.executable, os.path.abspath(__file__)] + sys.argv[1:], env)
        # not reached
    reexeced = os.environ.get(_REEXEC_FLAG) == "1"
    orig = os.environ.get("_P1_ORIG_VIS")
    return {
        "as_launched": orig if orig else repr(inherited),
        "effective": {"ROCR_VISIBLE_DEVICES": os.environ.get("ROCR_VISIBLE_DEVICES"),
                      "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES")},
        "reexeced": reexeced,
        "note": ("the probe forces ROCR_VISIBLE_DEVICES=0,1 with HIP_VISIBLE_DEVICES "
                 "unset so the Ryzen iGPU (ROCm device 2, 47 GB GTT) can never enter "
                 "enumeration"),
    }


# IMPORTANT: only fence when run as a script.  The fence re-execs the process,
# so doing it at import time would make `import p1_host_read_bw` silently launch
# a full GPU measurement run with default arguments.  (It did, once.)
if __name__ == "__main__":
    _ENV_FENCE = _fence_device_visibility()
else:
    _ENV_FENCE = {"as_launched": None, "effective": None, "reexeced": False,
                  "note": "module imported, not executed: no device-visibility "
                          "fence applied and no GPU work will be started"}

import argparse  # noqa: E402
import ctypes  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import statistics  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

SCHEMA_VERSION = 4
PROBE_ID = "P1"
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DEFAULT_OUTDIR = os.path.join(REPO, "docs", "measurements", "WEIGHT_OFFLOAD_2026-09-02")
BUILD_DIR = os.path.join(HERE, "_build")
HIP_SRC = os.path.join(HERE, "p1_kernels.hip")
SO_PATH = os.path.join(BUILD_DIR, "p1_kernels.so")

# Plan section 3: "< 0.5x the copy-engine figure (< ~13 GB/s)".
ABS_KILL_GBPS = 13.0
REL_KILL_FRACTION = 0.5

# Measured prior (plan section 0 / brief): pinned-host H2D via hipMemcpyAsync is
# 28.1 GB/s at 256 MiB on this box.  If the copy engine measures far below that,
# the RELATIVE threshold is being derived from a broken denominator and a host
# read could clear "0.5x copy engine" while being objectively slow.
COPY_ENGINE_PRIOR_GBPS = 28.1
COPY_ENGINE_SANITY_FLOOR_GBPS = 0.5 * COPY_ENGINE_PRIOR_GBPS

# --- provenance cross-checks -------------------------------------------------
# Physics: bytes that cross PCIe cannot materially outrun the copy engine, and
# cannot come close to HBM.  The previous 2.0x-over-copy-engine check was far
# too loose: a 50 GB/s read (impossible over this link, ~2x the 28 GB/s copy
# engine) would have slipped through and been reported as a host-page number.
PROVENANCE_MAX_OVER_COPY = 1.5      # > this x copy engine -> NOT crossing PCIe
PROVENANCE_MAX_FRAC_OF_HBM = 0.25   # >= this fraction of the HBM reference -> in VRAM
# The same physics run FORWARD: a region that is provably not in VRAM and reads
# at PCIe speed IS host-backed, whatever a noisy MemAvailable delta said.
PROVENANCE_HOST_MAX_OVER_COPY = 1.25
PROVENANCE_HOST_MAX_FRAC_OF_HBM = 0.15

# hipEvent timings are cross-checked against the wall clock over the whole rep
# loop.  Wall time strictly exceeds device time (it carries launch + memset +
# sync + ctypes overhead), so event-derived GB/s can legitimately be a little
# ABOVE wall-derived GB/s -- but never by this factor.  A broken/garbage
# hipEventElapsedTime is otherwise indistinguishable from a spectacular result.
TIMING_SANITY_FACTOR = 4.0

PATTERNS = ("linear", "tiled", "random128")
PATTERN_ID = {"linear": 0, "tiled": 1, "random128": 2}

# hipHostMalloc flags
HHM_PORTABLE = 0x1
HHM_MAPPED = 0x2
HHM_COHERENT = 0x40000000
HHM_NONCOHERENT = 0x80000000

# hipMemAllocationType / location / access
HIP_MEM_ALLOCATION_TYPE_PINNED = 0x1
HIP_MEM_LOCATION_TYPE_DEVICE = 1
HIP_MEM_LOCATION_TYPE_HOST = 2
HIP_MEM_LOCATION_TYPE_HOST_NUMA = 3
HIP_MEM_LOCATION_TYPE_HOST_NUMA_CURRENT = 4
HIP_MEM_ACCESS_FLAGS_PROT_READWRITE = 3

LOC_NAMES = {
    HIP_MEM_LOCATION_TYPE_DEVICE: "hipMemLocationTypeDevice",
    HIP_MEM_LOCATION_TYPE_HOST: "hipMemLocationTypeHost",
    HIP_MEM_LOCATION_TYPE_HOST_NUMA: "hipMemLocationTypeHostNuma",
    HIP_MEM_LOCATION_TYPE_HOST_NUMA_CURRENT: "hipMemLocationTypeHostNumaCurrent",
}
HIP_MEMORY_TYPE = {0: "Host", 1: "Device", 2: "Array", 3: "Unified", 4: "Managed"}

# Placement classification thresholds, as a fraction of the region size.
PLACEMENT_VRAM_DEVICE_FRAC = 0.5     # VRAM dropped by >= this -> device-resident
PLACEMENT_VRAM_HOST_FRAC = 0.25      # VRAM dropped by <  this -> not in VRAM
PLACEMENT_MEMAVAIL_HOST_FRAC = 0.5   # MemAvailable dropped by >= this -> host RAM


class ProbeError(RuntimeError):
    """Precondition or measurement failure.  Always fatal, always non-zero exit."""


def _fail(msg: str) -> None:
    raise ProbeError(msg)


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


# ---------------------------------------------------------------------------
# Box state -- recorded before and after.  A capacity/bandwidth number from an
# unrecorded box is worthless (this box is NOT idle).
# ---------------------------------------------------------------------------

def _read_text(path: str) -> str | None:
    try:
        with open(path, "r") as fh:
            return fh.read()
    except OSError:
        return None


def _run_cmd(cmd: list[str], timeout: float = 20.0) -> dict:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return {"cmd": " ".join(cmd), "rc": p.returncode,
                "stdout": p.stdout.strip(), "stderr": p.stderr.strip()[:2000]}
    except (OSError, subprocess.SubprocessError) as e:
        return {"cmd": " ".join(cmd), "rc": None, "error": f"{type(e).__name__}: {e}"}


def _parse_kv_kb(text: str | None, keys: tuple[str, ...]) -> dict:
    out: dict[str, int | None] = {k: None for k in keys}
    if not text:
        return out
    for line in text.splitlines():
        parts = line.split(":")
        if len(parts) == 2 and parts[0] in out:
            try:
                out[parts[0]] = int(parts[1].strip().split()[0])
            except (ValueError, IndexError):
                pass
    return out


def _parse_vmstat(keys: tuple[str, ...]) -> dict:
    out: dict[str, int | None] = {k: None for k in keys}
    text = _read_text("/proc/vmstat")
    if not text:
        return out
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] in out:
            try:
                out[parts[0]] = int(parts[1])
            except ValueError:
                pass
    return out


SYSFS_MEM_FIELDS = ("mem_info_vram_used", "mem_info_vram_total",
                    "mem_info_vis_vram_used", "mem_info_vis_vram_total",
                    "mem_info_gtt_used", "mem_info_gtt_total")


def sysfs_amdgpu_cards() -> dict:
    """PCI BDF -> that card's amdgpu memory accounting.

    This is the AUTHORITATIVE placement signal and the reason it exists here:
    hipMemGetInfo's "free VRAM" does not necessarily account for VMM
    (hipMemCreate) allocations, and /proc/meminfo MemAvailable is whole-box
    noise on a machine that is never idle.  amdgpu exports, PER CARD:
      mem_info_vram_used  -- bytes actually committed in that card's HBM
      mem_info_gtt_used   -- bytes of HOST RAM pinned and GPU-mapped for it
    A hipMemCreate(location=Host) region that really lands in host RAM moves
    gtt_used, not vram_used.  One that secretly lands in VRAM moves vram_used.
    Neither counter is perturbed by unrelated host-RAM churn elsewhere on the
    box, which is exactly the failure mode MemAvailable has.
    """
    out: dict = {}
    base = "/sys/class/drm"
    try:
        names = sorted(os.listdir(base))
    except OSError:
        return out
    for name in names:
        if not name.startswith("card") or "-" in name:
            continue
        dev = os.path.join(base, name, "device")
        try:
            bdf = os.path.basename(os.path.realpath(dev))
        except OSError:
            continue
        rec: dict = {"drm_node": name, "pci_bdf": bdf}
        present = False
        for f in SYSFS_MEM_FIELDS:
            t = _read_text(os.path.join(dev, f))
            v = None
            if t and t.strip().lstrip("-").isdigit():
                v = int(t.strip())
                present = True
            rec[f] = v
        if present:
            out[bdf.lower()] = rec
    return out


def card_mem(bdf: str | None) -> dict:
    """Snapshot one card's amdgpu counters, keyed by the PCI BDF HIP reported."""
    if not bdf:
        return {}
    return sysfs_amdgpu_cards().get(bdf.strip().lower(), {})


def pcie_link_chain(bdf: str | None) -> dict:
    """Every PCIe link between the CPU root complex and this card, with the
    NARROWEST/SLOWEST one identified.

    This is load-bearing provenance for a host-page bandwidth number, not trivia.
    `lspci` on the card's own function reports the width of the card's ON-PACKAGE
    bridge (x16 here), NOT the slot link, so reading it alone tells you nothing
    about the achievable PCIe rate.  The real link is the upstream root port.

    Measured on this box: the two gfx1201 cards are NOT symmetric.  Card 0's root
    port (0000:00:01.1) is trained at Gen5 x8 (32 GT/s) and delivers 28.7 GB/s;
    card 1's (0000:00:01.3) is trained at Gen4 x8 (16 GT/s, though max_link_speed
    reads 32) and delivers 14.3 GB/s -- exactly half.  Sampling the link during a
    sustained copy showed it never retrains, so this is a persistent platform
    asymmetry rather than idle downtraining.  Recording it makes each bandwidth
    figure self-describing about which link it actually crossed.
    """
    if not bdf:
        return {}
    dev = "/sys/bus/pci/devices/" + bdf.strip().lower()
    try:
        chain_path = os.path.realpath(dev)
    except OSError:
        return {}
    hops: list[dict] = []
    parts = chain_path.split(os.sep)
    # .../devices/pci0000:00/0000:00:01.1/0000:01:00.0/.../0000:03:00.0
    for i, seg in enumerate(parts):
        if not re.fullmatch(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9a-f]", seg):
            continue
        p = os.sep.join(parts[: i + 1])

        def _rd(f: str):
            v = _read_text(os.path.join(p, f))
            return v.strip() if v else None

        def _w(f: str):
            v = _rd(f)
            try:
                return int(v)
            except (TypeError, ValueError):
                return None

        def _gts(f: str):
            v = _rd(f) or ""
            mm = re.match(r"([\d.]+)\s*GT/s", v)
            return float(mm.group(1)) if mm else None

        hops.append({
            "bdf": seg,
            "role": ("root_port" if not hops else
                     "endpoint" if seg == bdf.strip().lower() else "bridge"),
            "current_link_width": _w("current_link_width"),
            "current_link_speed_gts": _gts("current_link_speed"),
            "max_link_width": _w("max_link_width"),
            "max_link_speed_gts": _gts("max_link_speed"),
        })
    if not hops:
        return {}
    # Per-lane GT/s -> GB/s: Gen3 8b/10b, Gen4+ 128b/130b.
    def _bw(h: dict):
        w, s = h.get("current_link_width"), h.get("current_link_speed_gts")
        if not w or not s:
            return None
        return round(w * s * (128 / 130 if s >= 8 else 0.8) / 8, 2)

    for h in hops:
        h["theoretical_gbps"] = _bw(h)
    rated = [h for h in hops if h.get("theoretical_gbps") is not None]
    bottleneck = min(rated, key=lambda h: h["theoretical_gbps"]) if rated else None
    return {
        "hops": hops,
        "root_port": hops[0]["bdf"],
        "bottleneck": bottleneck,
        "bottleneck_theoretical_gbps": (bottleneck or {}).get("theoretical_gbps"),
        "downtrained": bool(bottleneck) and (
            (bottleneck.get("max_link_speed_gts") or 0) >
            (bottleneck.get("current_link_speed_gts") or 0)),
        "note": ("the card's own function reports its ON-PACKAGE bridge width (x16), "
                 "not the slot link; the governing link is the upstream root port"),
    }


class LinkSampler:
    """Poll the PCIe link chain in the background while a sustained load runs.

    An IDLE link speed is the wrong number to file next to a loaded bandwidth
    figure.  These root ports do dynamic link power management: card 1's port
    was observed at 2.5 GT/s (Gen1) at deep idle, 16.0 GT/s throughout a
    sustained H2D copy, and back down afterwards.  Reading the link once, at
    device-record time, would therefore have recorded 2.5 GT/s -- ~2 GB/s
    theoretical -- beside a measured 14.3 GB/s, which is not merely useless but
    actively contradictory.  So the governing figure is the MAXIMUM state seen
    while the copy engine is actually saturating the link.
    """

    def __init__(self, bdf: str | None, interval: float = 0.01):
        self.bdf, self.interval = bdf, interval
        self._stop = threading.Event()
        self._seen: list[dict] = []
        self._thread: threading.Thread | None = None

    def _poll(self) -> None:
        while not self._stop.is_set():
            lk = pcie_link_chain(self.bdf)
            bn = lk.get("bottleneck")
            if bn:
                self._seen.append(bn)
            self._stop.wait(self.interval)

    def __enter__(self) -> "LinkSampler":
        # Re-entrant: the copy-engine loop enters this once per copy arm, and the
        # samples accumulate so `result()` is the max over all of them.  The stop
        # flag MUST be cleared here or every entry after the first would start a
        # thread that exits immediately and silently sample nothing.
        self._stop.clear()
        if self.bdf:
            self._thread = threading.Thread(target=self._poll, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def result(self) -> dict:
        if not self._seen:
            return {}
        best = max(self._seen, key=lambda h: h.get("theoretical_gbps") or 0.0)
        states = sorted({(h.get("current_link_width"), h.get("current_link_speed_gts"))
                         for h in self._seen},
                        key=lambda t: (t[1] or 0, t[0] or 0))
        return {
            "samples": len(self._seen),
            "under_load_width": best.get("current_link_width"),
            "under_load_speed_gts": best.get("current_link_speed_gts"),
            "under_load_theoretical_gbps": best.get("theoretical_gbps"),
            "rated_width": best.get("max_link_width"),
            "rated_speed_gts": best.get("max_link_speed_gts"),
            "reached_rated_speed": (best.get("current_link_speed_gts") ==
                                    best.get("max_link_speed_gts")),
            "all_states_observed": [{"width": w, "speed_gts": s} for w, s in states],
            "note": ("max link state observed while the copy engine was saturating "
                     "the link; these ports down-train at idle, so a single "
                     "at-rest reading would understate the path"),
        }


def collect_box_state(label: str, with_smi: bool = True) -> dict:
    st: dict = {
        "label": label,
        "utc": _utc(),
        "meminfo_kb": _parse_kv_kb(
            _read_text("/proc/meminfo"),
            ("MemTotal", "MemFree", "MemAvailable", "Buffers", "Cached",
             "SwapTotal", "SwapFree", "Dirty", "Mlocked", "Shmem"),
        ),
        "vmstat": _parse_vmstat(
            ("pswpin", "pswpout", "pgmajfault", "nr_mlock", "nr_free_pages")),
        "loadavg": (_read_text("/proc/loadavg") or "").strip() or None,
        "free_g": _run_cmd(["free", "-g"]) if shutil.which("free") else None,
        "amdgpu_cards": sysfs_amdgpu_cards(),
        "zfs_arc_size": None,
    }
    arc = _read_text("/proc/spl/kstat/zfs/arcstats")
    if arc:
        for line in arc.splitlines():
            f = line.split()
            if len(f) == 3 and f[0] in ("size", "c_max"):
                st.setdefault("zfs_arc", {})[f[0]] = int(f[2])
        st["zfs_arc_size"] = (st.get("zfs_arc") or {}).get("size")
    if with_smi and shutil.which("rocm-smi"):
        st["rocm_smi"] = _run_cmd(
            ["rocm-smi", "--showmeminfo", "vram", "--showuse", "--showbus",
             "--showproductname"], timeout=40.0)
    else:
        st["rocm_smi"] = None
    return st


def collect_static_env() -> dict:
    git = {}
    for name, cmd in (("sha", ["git", "rev-parse", "HEAD"]),
                      ("branch", ["git", "rev-parse", "--abbrev-ref", "HEAD"]),
                      ("dirty", ["git", "status", "--porcelain"])):
        r = _run_cmd(["git", "-C", REPO] + cmd[1:])
        git[name] = r.get("stdout") if r.get("rc") == 0 else None
    git["dirty"] = bool(git.get("dirty"))
    return {
        "hostname": platform.node(),
        "kernel": platform.release(),
        "python": sys.version.split()[0],
        "worktree": REPO,
        "git": git,
        "amdgpu_module_version": (_read_text("/sys/module/amdgpu/version") or "").strip() or None,
        "rocm_version_file": (_read_text("/opt/rocm/.info/version") or "").strip() or None,
        "env_fence": _ENV_FENCE,
        "env_seen": {k: v for k, v in os.environ.items()
                     if k.startswith(("ROCR_", "HIP_", "HSA_", "GPU_", "AMD_", "PYTORCH_"))},
    }


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def _sha256(path: str) -> str | None:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def build_kernels(arch: str, rebuild: bool, hipcc: str) -> dict:
    if not os.path.isfile(HIP_SRC):
        _fail(f"kernel source missing: {HIP_SRC}")
    if not (os.path.isfile(hipcc) or shutil.which(hipcc)):
        _fail(f"hipcc not found: {hipcc} (set --hipcc)")
    os.makedirs(BUILD_DIR, exist_ok=True)
    cmd = [hipcc, "-O3", "-std=c++17", f"--offload-arch={arch}",
           "-fPIC", "-shared", "-o", SO_PATH, HIP_SRC]
    need = rebuild or not os.path.isfile(SO_PATH) or (
        os.path.getmtime(SO_PATH) < os.path.getmtime(HIP_SRC))
    info = {"hipcc": hipcc, "arch": arch, "cmd": " ".join(cmd), "so": SO_PATH,
            "rebuilt": bool(need), "src_sha256": _sha256(HIP_SRC)}
    ver = _run_cmd([hipcc, "--version"])
    info["hipcc_version"] = (ver.get("stdout") or "").splitlines()[:3]
    if need:
        t0 = time.perf_counter()
        p = subprocess.run(cmd, capture_output=True, text=True)
        info["compile_seconds"] = round(time.perf_counter() - t0, 2)
        info["compile_stderr"] = p.stderr.strip()[:4000]
        if p.returncode != 0:
            _fail(f"hipcc failed (rc={p.returncode}):\n{p.stderr}")
    info["so_sha256"] = _sha256(SO_PATH)
    info["so_bytes"] = os.path.getsize(SO_PATH) if os.path.isfile(SO_PATH) else None
    return info


REQUIRED_SYMBOLS = (
    "p1_err", "p1_runtime_version", "p1_driver_version", "p1_device_count",
    "p1_set_device", "p1_sync", "p1_mem_info", "p1_device_info",
    "p1_malloc_device", "p1_free_device", "p1_host_malloc", "p1_host_free",
    "p1_host_dev_ptr", "p1_fill_range", "p1_run", "p1_memcpy_bw",
    "p1_line_checksums", "p1_cpu_fence", "p1_cpu_flush", "p1_pointer_attrs",
)


def load_kernels(check_only: bool = False) -> ctypes.CDLL:
    lib = ctypes.CDLL(SO_PATH)
    missing = [s for s in REQUIRED_SYMBOLS if not hasattr(lib, s)]
    if missing:
        _fail(f"{SO_PATH} is missing symbols: {missing}")
    if check_only:
        return lib
    c_sz, c_szp = ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)
    vp, vpp = ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)
    ip, up = ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_uint)
    fp = ctypes.POINTER(ctypes.c_float)
    lib.p1_err.restype = ctypes.c_char_p
    lib.p1_err.argtypes = [ctypes.c_int]
    for fn, args in (
        ("p1_runtime_version", [ip]), ("p1_driver_version", [ip]),
        ("p1_device_count", [ip]), ("p1_set_device", [ctypes.c_int]),
        ("p1_sync", []), ("p1_mem_info", [c_szp, c_szp]),
        ("p1_device_info", [ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                            ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
                            ctypes.c_int, ip, ip, ip,
                            ctypes.POINTER(ctypes.c_ulonglong)]),
        ("p1_malloc_device", [vpp, c_sz]), ("p1_free_device", [vp]),
        ("p1_host_malloc", [vpp, c_sz, ctypes.c_uint]), ("p1_host_free", [vp]),
        ("p1_host_dev_ptr", [vpp, vp]),
        ("p1_fill_range", [vp, c_sz, c_sz, ctypes.c_uint]),
        ("p1_run", [ctypes.c_int, vp, c_sz, ctypes.c_int, ctypes.c_int,
                    c_sz, c_sz, c_sz, up, c_szp, fp]),
        ("p1_memcpy_bw", [vp, vp, c_sz, ctypes.c_int, ctypes.c_int, fp]),
        ("p1_line_checksums", [vp, c_sz, c_sz, ctypes.c_uint, up]),
        ("p1_pointer_attrs", [vp, ip, ip, vpp, vpp, ip, up]),
        ("p1_cpu_flush", [vp, c_sz]),
    ):
        getattr(lib, fn).argtypes = args
        if fn != "p1_cpu_flush":
            getattr(lib, fn).restype = ctypes.c_int
    lib.p1_cpu_fence.argtypes = []
    lib.p1_cpu_fence.restype = None
    lib.p1_cpu_flush.restype = None
    return lib


def K(lib: ctypes.CDLL, rc: int, what: str) -> None:
    if rc != 0:
        msg = lib.p1_err(rc)
        _fail(f"{what} -> rc={rc} ({msg.decode() if msg else '?'})")


# ---------------------------------------------------------------------------
# HIP VMM (ctypes, direct against libamdhip64)
# ---------------------------------------------------------------------------

class hipMemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class _AllocFlags(ctypes.Structure):
    _fields_ = [("compressionType", ctypes.c_ubyte),
                ("gpuDirectRDMACapable", ctypes.c_ubyte),
                ("usage", ctypes.c_ushort)]


class hipMemAllocationProp(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("requestedHandleType", ctypes.c_int),
                ("location", hipMemLocation), ("win32HandleMetaData", ctypes.c_void_p),
                ("allocFlags", _AllocFlags)]


class hipMemAccessDesc(ctypes.Structure):
    _fields_ = [("location", hipMemLocation), ("flags", ctypes.c_int)]


def load_hip() -> ctypes.CDLL:
    try:
        hip = ctypes.CDLL("libamdhip64.so")
    except OSError as e:
        _fail(f"cannot load libamdhip64.so: {e}")
    sz, szp = ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)
    vp, vpp = ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)
    pp = ctypes.POINTER(hipMemAllocationProp)
    for fn, args in (
        ("hipMemGetAllocationGranularity", [szp, pp, ctypes.c_int]),
        ("hipMemAddressReserve", [vpp, sz, sz, vp, ctypes.c_ulonglong]),
        ("hipMemAddressFree", [vp, sz]),
        ("hipMemCreate", [vpp, sz, pp, ctypes.c_ulonglong]),
        ("hipMemMap", [vp, sz, sz, vp, ctypes.c_ulonglong]),
        ("hipMemUnmap", [vp, sz]),
        ("hipMemSetAccess", [vp, sz, ctypes.POINTER(hipMemAccessDesc), sz]),
        ("hipMemRelease", [vp]),
    ):
        if not hasattr(hip, fn):
            _fail(f"libamdhip64.so lacks {fn} -- VMM API unavailable, P1 cannot run")
        getattr(hip, fn).argtypes = args
        getattr(hip, fn).restype = ctypes.c_int
    return hip


def _prop(loc_type: int, dev: int) -> hipMemAllocationProp:
    p = hipMemAllocationProp()
    ctypes.memset(ctypes.byref(p), 0, ctypes.sizeof(p))
    p.type = HIP_MEM_ALLOCATION_TYPE_PINNED
    p.location.type = loc_type
    p.location.id = dev
    return p


def _vram_free(lib) -> int:
    f, t = ctypes.c_size_t(0), ctypes.c_size_t(0)
    K(lib, lib.p1_mem_info(ctypes.byref(f), ctypes.byref(t)), "p1_mem_info")
    return int(f.value)


def _mem_available_bytes() -> int:
    kb = _parse_kv_kb(_read_text("/proc/meminfo"), ("MemAvailable",))["MemAvailable"]
    return (kb or 0) * 1024


def pointer_attrs(lib, ptr: int) -> dict:
    mt, dv, mg = ctypes.c_int(-1), ctypes.c_int(-1), ctypes.c_int(-1)
    dp, hp = ctypes.c_void_p(), ctypes.c_void_p()
    fl = ctypes.c_uint(0)
    rc = lib.p1_pointer_attrs(ctypes.c_void_p(ptr), ctypes.byref(mt), ctypes.byref(dv),
                              ctypes.byref(dp), ctypes.byref(hp), ctypes.byref(mg),
                              ctypes.byref(fl))
    if rc != 0:
        msg = lib.p1_err(rc)
        return {"rc": rc, "error": msg.decode() if msg else "?",
                "note": "hipPointerGetAttributes refused this pointer"}
    return {"rc": 0, "memory_type": mt.value,
            "memory_type_name": HIP_MEMORY_TYPE.get(mt.value, f"?{mt.value}"),
            "device": dv.value,
            "device_pointer": f"0x{dp.value:x}" if dp.value else None,
            "host_pointer": f"0x{hp.value:x}" if hp.value else None,
            "is_managed": mg.value, "allocation_flags": fl.value,
            "caveat": "a query can succeed with a fabricated value; cross-checked "
                      "against the VRAM/MemAvailable deltas and the measured bandwidth"}


def classify_placement(size: int, vram_delta: int, memavail_delta: int,
                       sysfs_before: dict | None = None,
                       sysfs_after: dict | None = None) -> dict:
    """Classify where a region's physical pages actually live.

    `vram_delta`/`memavail_delta` are (before - after), i.e. bytes consumed, from
    hipMemGetInfo and /proc/meminfo respectively.

    PRIMARY signal is the per-card amdgpu sysfs accounting when it is available
    (`mem_info_vram_used` / `mem_info_gtt_used`), because it is the only one of
    the three that is (a) authoritative about the driver's own bookkeeping and
    (b) immune to unrelated host-RAM churn from the other agents on this box.
    hipMemGetInfo + MemAvailable are retained as a corroborating fallback and
    are always recorded, so a disagreement is visible in the artifact.
    """
    rec: dict = {
        "size_bytes": size,
        "vram_consumed_bytes": vram_delta,
        "vram_consumed_frac": round(vram_delta / size, 3) if size else None,
        "memavailable_consumed_bytes": memavail_delta,
        "memavailable_consumed_frac": round(memavail_delta / size, 3) if size else None,
    }

    sys_cls = None
    if sysfs_before and sysfs_after:
        sv = ((sysfs_after.get("mem_info_vram_used") or 0)
              - (sysfs_before.get("mem_info_vram_used") or 0))
        sg = ((sysfs_after.get("mem_info_gtt_used") or 0)
              - (sysfs_before.get("mem_info_gtt_used") or 0))
        have = (sysfs_before.get("mem_info_vram_used") is not None
                and sysfs_after.get("mem_info_vram_used") is not None
                and sysfs_before.get("mem_info_gtt_used") is not None
                and sysfs_after.get("mem_info_gtt_used") is not None)
        rec["sysfs_vram_used_delta_bytes"] = sv if have else None
        rec["sysfs_gtt_used_delta_bytes"] = sg if have else None
        rec["sysfs_vram_used_delta_frac"] = round(sv / size, 3) if (have and size) else None
        rec["sysfs_gtt_used_delta_frac"] = round(sg / size, 3) if (have and size) else None
        rec["sysfs_card"] = sysfs_after.get("pci_bdf") or sysfs_before.get("pci_bdf")
        if have:
            if sv >= PLACEMENT_VRAM_DEVICE_FRAC * size:
                sys_cls = "device_resident"
            elif (sg >= PLACEMENT_MEMAVAIL_HOST_FRAC * size
                  and sv < PLACEMENT_VRAM_HOST_FRAC * size):
                sys_cls = "host_resident"
            elif sv < PLACEMENT_VRAM_HOST_FRAC * size:
                sys_cls = "not_in_vram_but_host_delta_unclear"
            else:
                sys_cls = "indeterminate"
    rec["sysfs_classification"] = sys_cls

    in_vram = vram_delta >= PLACEMENT_VRAM_DEVICE_FRAC * size
    not_in_vram = vram_delta < PLACEMENT_VRAM_HOST_FRAC * size
    in_host = memavail_delta >= PLACEMENT_MEMAVAIL_HOST_FRAC * size
    if in_vram:
        hip_cls = "device_resident"
    elif not_in_vram and in_host:
        hip_cls = "host_resident"
    elif not_in_vram:
        hip_cls = "not_in_vram_but_host_delta_unclear"
    else:
        hip_cls = "indeterminate"
    rec["hip_memgetinfo_classification"] = hip_cls

    if sys_cls is not None:
        rec["classification"] = sys_cls
        rec["classification_source"] = "amdgpu sysfs (mem_info_vram_used / mem_info_gtt_used)"
    else:
        rec["classification"] = hip_cls
        rec["classification_source"] = "hipMemGetInfo + /proc/meminfo MemAvailable (sysfs unavailable)"
    rec["classifications_agree"] = (sys_cls is None or sys_cls == hip_cls)
    rec["method"] = (
        "per-card amdgpu sysfs mem_info_vram_used/mem_info_gtt_used sampled "
        "immediately before and after backing the region (PRIMARY -- per card, "
        "so unrelated host-RAM churn cannot move it), cross-checked against "
        "hipMemGetInfo free-VRAM and /proc/meminfo MemAvailable deltas")
    return rec


class VmmRegion:
    """A VIRGIN reserved VA backed by fresh physical handles.  Never remapped."""

    def __init__(self, hip, lib, dev: int, nbytes: int, loc_type: int,
                 chunk_bytes: int, align: int = 2 << 20, bdf: str | None = None):
        self.hip, self.lib, self.dev = hip, lib, dev
        self.bdf = bdf
        self.loc_type = loc_type
        self.nbytes = nbytes
        self.handles: list[ctypes.c_void_p] = []
        self.mapped: list[tuple[int, int]] = []
        self.va = ctypes.c_void_p()
        self.info: dict = {"loc_type": loc_type,
                           "loc_type_name": LOC_NAMES.get(loc_type, str(loc_type)),
                           "requested_bytes": nbytes}

        gran = ctypes.c_size_t(0)
        prop = _prop(loc_type, dev if loc_type == HIP_MEM_LOCATION_TYPE_DEVICE else 0)
        rc = hip.hipMemGetAllocationGranularity(ctypes.byref(gran), ctypes.byref(prop), 0)
        if rc != 0:
            _fail(f"hipMemGetAllocationGranularity(loc={loc_type}) -> {rc}")
        g = max(int(gran.value), 1)
        self.info["granularity"] = g
        # The query can succeed with a fabricated value, so we assert on the
        # actual operation (hipMemCreate) below, never on the query.
        if nbytes % g:
            nbytes += g - (nbytes % g)
        self.nbytes = nbytes
        self.info["aligned_bytes"] = nbytes
        align = max(align, g)

        # Back the WHOLE reservation.  If the requested chunk size is refused we
        # retry with smaller chunks -- but each retry takes a FRESH reservation,
        # because hipMemUnmap->hipMemMap at an already-used VA silently serves
        # the stale physical page on this box (plan section 0 fact 1).  Reusing
        # the VA would risk measuring the wrong physical medium while every call
        # returns hipSuccess.  If no chunk size works the probe aborts loudly; a
        # bandwidth number over a partially-backed range is never reported.
        candidates: list[int] = []
        for c in (chunk_bytes, 64 << 20, 16 << 20, 4 << 20, 2 << 20, g):
            c = max(c, g)
            c -= c % g
            if c > 0 and c not in candidates:
                candidates.append(c)
        attempts: list[dict] = []
        for chunk in candidates:
            rc = hip.hipMemAddressReserve(ctypes.byref(self.va), nbytes, align, None, 0)
            if rc != 0:
                _fail(f"hipMemAddressReserve({nbytes}) -> {rc}")
            # PLACEMENT PROVENANCE: sample VRAM and host MemAvailable immediately
            # either side of the backing.  hipMemCreate can return hipSuccess for
            # location.type=Host while actually allocating VRAM, in which case a
            # "host-page read bandwidth" number would be a pure fabrication.
            vram0, avail0 = _vram_free(lib), _mem_available_bytes()
            sys0 = card_mem(self.bdf)
            err = self._back(prop, chunk)
            if err is None:
                self._baseline = (vram0, avail0, sys0)
                self.info["placement"] = classify_placement(
                    nbytes, vram0 - _vram_free(lib), avail0 - _mem_available_bytes(),
                    sys0, card_mem(self.bdf))
            attempts.append({"chunk_bytes": chunk, "rc": err,
                             "va": f"0x{self.va.value:x}"})
            if err is None:
                self.info["chunk_bytes"] = chunk
                self.info["n_chunks"] = len(self.handles)
                self.info["va"] = f"0x{self.va.value:x}"
                self.info["virgin_va"] = True
                break
            self._unback()
            hip.hipMemAddressFree(self.va, nbytes)
            self.va = ctypes.c_void_p()
        else:
            self.info["backing_attempts"] = attempts
            _fail(f"could not back a {nbytes} B region with location.type={loc_type} at "
                  f"any chunk size {candidates}: {attempts}")
        self.info["backing_attempts"] = attempts

        desc = hipMemAccessDesc()
        desc.location.type = HIP_MEM_LOCATION_TYPE_DEVICE
        desc.location.id = dev
        desc.flags = HIP_MEM_ACCESS_FLAGS_PROT_READWRITE
        rc = hip.hipMemSetAccess(self.va, nbytes, ctypes.byref(desc), 1)
        if rc != 0:
            self.close()
            _fail(f"hipMemSetAccess(dev={dev}, loc={loc_type}) -> {rc}")
        self.info["set_access"] = "ok"
        self.info["pointer_attrs"] = pointer_attrs(lib, self.ptr)
        self.info["maps_entry"] = maps_entry_for(self.ptr)

    def resample_placement(self, lib) -> dict | None:
        """Re-classify AFTER every page has been touched.

        A driver may commit pages lazily, so the delta measured at backing time
        can be ~0 for both media.  Call this once the region has been fully
        written through the device pointer.
        """
        base = getattr(self, "_baseline", None)
        if base is None:
            return None
        vram0, avail0, sys0 = base
        p = classify_placement(self.nbytes, vram0 - _vram_free(lib),
                               avail0 - _mem_available_bytes(),
                               sys0, card_mem(self.bdf))
        p["sampled"] = "after every page was touched by a device-side fill"
        self.info["placement_after_touch"] = p
        return p

    def _back(self, prop, chunk: int) -> int | None:
        """Create+map handles covering the whole reservation. None on success."""
        off = 0
        while off < self.nbytes:
            n = min(chunk, self.nbytes - off)
            h = ctypes.c_void_p()
            rc = self.hip.hipMemCreate(ctypes.byref(h), n, ctypes.byref(prop), 0)
            if rc != 0:
                return rc
            self.handles.append(h)
            rc = self.hip.hipMemMap(ctypes.c_void_p(self.va.value + off), n, 0, h, 0)
            if rc != 0:
                return rc
            self.mapped.append((self.va.value + off, n))
            off += n
        return None

    def _unback(self) -> None:
        for addr, n in self.mapped:
            self.hip.hipMemUnmap(ctypes.c_void_p(addr), n)
        self.mapped = []
        for h in self.handles:
            self.hip.hipMemRelease(h)
        self.handles = []

    @property
    def ptr(self) -> int:
        return int(self.va.value)

    def close(self) -> None:
        self._unback()
        if self.va and self.va.value:
            self.hip.hipMemAddressFree(self.va, self.nbytes)
            self.va = ctypes.c_void_p()


# ---------------------------------------------------------------------------
# CPU-accessibility probes that cannot segfault (the kernel returns EFAULT
# instead of raising SIGSEGV when a syscall touches a bad user pointer).
# ---------------------------------------------------------------------------

def _cpu_read_probe(addr: int, nbytes: int) -> tuple[dict, bytes | None]:
    """Copy `nbytes` out of `addr` THROUGH THE KERNEL, so a bad pointer yields
    EFAULT instead of SIGSEGV.  Returns (classification, bytes-read-or-None).

    The sink must be one the kernel genuinely copies *from user space* into.
    `/dev/null` is NOT: its write handler discards the payload and returns the
    count without ever touching the user pages, so it reports EVERY pointer as
    readable.  Measured on this box against a guaranteed-unmapped VA:

        os.write(/dev/null, buf) -> returned 4096   (no fault, wrong answer)
        os.write(regular file)   -> OSError EFAULT  (correct)
        os.write(pipe)           -> OSError EFAULT  (correct)

    That false positive is what made this probe SIGSEGV rather than report an
    inaccessible VA: it was used as the guard in front of a raw `ctypes.string_at`
    and in front of the coherence arm's raw `from_address` sweep.  A regular file
    in BUILD_DIR is used here so the copy is real and the answer is trustworthy.
    """
    buf = memoryview((ctypes.c_char * nbytes).from_address(addr))
    os.makedirs(BUILD_DIR, exist_ok=True)
    try:
        with tempfile.TemporaryFile(dir=BUILD_DIR) as fh:
            done = 0
            while done < nbytes:
                n = os.write(fh.fileno(), buf[done:])
                if n <= 0:
                    # A short/zero write is not proof of readability; refuse to
                    # claim the region is readable on a partial copy.
                    return ({"readable": False, "read_errno": None,
                             "read_strerror": f"short write after {done} of {nbytes} B"},
                            None)
                done += n
            fh.seek(0)
            return ({"readable": True, "read_errno": None, "read_strerror": None},
                    fh.read(nbytes))
    except OSError as e:
        return ({"readable": False, "read_errno": e.errno,
                 "read_strerror": e.strerror}, None)


def cpu_readable(addr: int, nbytes: int) -> dict:
    return _cpu_read_probe(addr, nbytes)[0]


def cpu_writable(addr: int, nbytes: int, restore: bool = True) -> dict:
    """Probe CPU writability without SIGSEGV (readv returns EFAULT instead).

    This probe is DESTRUCTIVE -- it stores `nbytes` into the region.  Every
    caller here runs it on a buffer that is either about to be re-filled or has
    already been swept, but a probe that silently corrupts the buffer under test
    is one refactor away from producing a wrong checksum (or, worse, a right one
    over the wrong bytes), so it restores the original content by default.

    Both the save and the restore go through the kernel (`os.write` into a file,
    `os.readv` back out of one).  Neither may be a raw CPU dereference: this
    function runs on VAs whose CPU accessibility is precisely the unknown being
    measured, so `ctypes.string_at`/`memmove` here would fault the process on a
    device-only mapping instead of recording "not CPU-accessible".
    """
    buf = (ctypes.c_char * nbytes).from_address(addr)
    orig = _cpu_read_probe(addr, nbytes)[1] if restore else None
    os.makedirs(BUILD_DIR, exist_ok=True)
    with tempfile.TemporaryFile(dir=BUILD_DIR) as fh:
        fh.write(b"\0" * nbytes)
        fh.flush()
        fh.seek(0)
        try:
            os.readv(fh.fileno(), [memoryview(buf)])
            out = {"writable": True, "write_errno": None, "write_strerror": None}
        except OSError as e:
            out = {"writable": False, "write_errno": e.errno,
                   "write_strerror": e.strerror}
    out["restored_original_bytes"] = False
    if orig is not None and out["writable"]:
        try:
            with tempfile.TemporaryFile(dir=BUILD_DIR) as fh:
                fh.write(orig)
                fh.flush()
                fh.seek(0)
                os.readv(fh.fileno(), [memoryview(buf)])
            out["restored_original_bytes"] = True
        except OSError:
            out["restored_original_bytes"] = False
    return out


def maps_entry_for(addr: int) -> str | None:
    text = _read_text("/proc/self/maps")
    if not text:
        return None
    for line in text.splitlines():
        try:
            lo, hi = line.split()[0].split("-")
            if int(lo, 16) <= addr < int(hi, 16):
                return line
        except (ValueError, IndexError):
            continue
    return None


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

def expected_prefix_sum(cover_bytes: int) -> int:
    """Buffer holds dword[i] == i, so a prefix cover sums to M(M-1)/2 mod 2^32."""
    m = cover_bytes // 4
    return (m * (m - 1) // 2) % (1 << 32)


def _stats(vals: list[float]) -> dict:
    s = sorted(vals)
    n = len(s)
    q = {
        "n": n,
        "median": statistics.median(s),
        "min": s[0],
        "max": s[-1],
        "mean": statistics.fmean(s),
        "stdev": statistics.stdev(s) if n > 1 else 0.0,
    }
    q["spread_pct"] = (100.0 * (s[-1] - s[0]) / s[0]) if s[0] > 0 else None
    if n >= 4:
        q["p25"], q["p75"] = s[n // 4], s[(3 * n) // 4]
    return q


def measure_pattern(lib, ptr: int, nbytes: int, pattern: str, blocks: int,
                    threads: int, row_bytes: int, seg_bytes: int,
                    stride_bytes: int, reps: int, arm: str, device: int,
                    max_config_seconds: float = 0.0) -> dict:
    pid = PATTERN_ID[pattern]
    seg = 128 if pattern == "random128" else seg_bytes
    csum = ctypes.c_uint(0)
    cover = ctypes.c_size_t(0)
    ms = ctypes.c_float(0.0)

    def one() -> tuple[float, int, int]:
        rc = lib.p1_run(pid, ctypes.c_void_p(ptr), nbytes, blocks, threads,
                        row_bytes, seg, stride_bytes,
                        ctypes.byref(csum), ctypes.byref(cover), ctypes.byref(ms))
        K(lib, rc, f"p1_run(arm={arm}, pattern={pattern}, blocks={blocks}, threads={threads})")
        return float(ms.value), int(csum.value), int(cover.value)

    base: dict = {
        "arm": arm,
        "device": device,
        "pattern": pattern,
        "blocks": blocks,
        "threads": threads,
        "waves_in_flight": blocks * threads // 32,
        "buffer_bytes": nbytes,
        "timing_source": "hipEvent",
    }

    # Warm-up rep is DISCARDED (lazy HIP init / first-touch page setup).
    warm_ms, warm_sum, cov = one()
    expect = expected_prefix_sum(cov)

    # Time budget.  A low-occupancy random-128B read of 512 MiB over PCIe is
    # latency-bound and can run for MINUTES per rep; the full 6-arm x 54-config
    # sweep would then look hung for hours.  The warm-up rep is the estimator.
    # Skipping only ever REMOVES slow configs, and `best` is a max over the
    # sweep, so this can never bias the reported bandwidth upward.
    reps_run = reps
    reps_note = None
    if max_config_seconds and max_config_seconds > 0:
        warm_s = warm_ms / 1000.0
        if warm_s > max_config_seconds:
            base.update({
                "cover_bytes": cov,
                "reps": 0,
                "reps_requested": reps,
                "measured": False,
                "skipped_reason": (
                    f"a single rep took {warm_s:.1f}s > --max-config-seconds "
                    f"{max_config_seconds:.1f}s; recorded from the warm-up rep only "
                    f"and EXCLUDED from `best` (skipping only removes slow points)"),
                "warmup_discarded_ms": warm_ms,
                "warmup_gbps": round(cov / (warm_ms * 1e-3) / 1e9, 3) if warm_ms > 0 else None,
                "ms": [],
                "gbps_median": None,
                "gbps_max": None,
                "gbps_min": None,
                "gbps_stdev": None,
                "ms_spread_pct": None,
                "wall_gbps": None,
                "event_vs_wall_ratio": None,
                "checksum_expected": expect,
                "checksum_got": warm_sum,
                "checksum_ok": bool(warm_sum == expect),
                "timing_sane": None,
                "ms_median": None,
            })
            return base
        if reps * warm_s > max_config_seconds:
            reps_run = max(3, int(max_config_seconds // warm_s))
            if reps_run < reps:
                reps_note = (f"reps reduced {reps}->{reps_run} to stay inside "
                             f"--max-config-seconds {max_config_seconds:.1f}s "
                             f"({warm_s:.2f}s per rep)")

    times, sums = [], []
    t_wall0 = time.perf_counter()
    for _ in range(reps_run):
        m, s, cov = one()
        times.append(m)
        sums.append(s)
    wall = time.perf_counter() - t_wall0

    ok = all(s == expect for s in sums) and warm_sum == expect
    st = _stats(times)
    # A non-positive event time is a broken timer, not an infinitely fast read:
    # score it 0 so the JSON stays valid and `timing_sane` (below) is what flags it.
    gbps = [cov / (t * 1e-3) / 1e9 if t > 0 else 0.0 for t in times]
    gst = _stats(gbps)
    wall_gbps = (reps_run * cov) / wall / 1e9 if wall > 0 else 0.0

    # TIMING FENCE.  hipEventElapsedTime is the only thing standing between this
    # probe and a spectacular fabricated number: if it returns ~0 ms the derived
    # GB/s is astronomical and every other check (checksum, placement) still
    # passes.  Wall time strictly exceeds device time, so event-derived GB/s can
    # exceed wall-derived GB/s only by the fixed per-rep host overhead.  Anything
    # beyond TIMING_SANITY_FACTOR is a broken timer, not a fast kernel.
    timing_sane = (
        all(t > 0.0 for t in times)
        and wall_gbps > 0.0
        and gst["median"] <= TIMING_SANITY_FACTOR * wall_gbps
    )

    base.update({
        "cover_bytes": cov,
        "reps": reps_run,
        "reps_requested": reps,
        "reps_note": reps_note,
        "measured": True,
        "warmup_discarded_ms": warm_ms,
        "ms": [round(t, 4) for t in times],
        "ms_stats": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in st.items()},
        "gbps_median": round(gst["median"], 3),
        "gbps_max": round(gst["max"], 3),
        "gbps_min": round(gst["min"], 3),
        "gbps_stdev": round(gst["stdev"], 3),
        "ms_spread_pct": (round(st["spread_pct"], 2) if st["spread_pct"] is not None else None),
        "wall_gbps": round(wall_gbps, 3),
        "event_vs_wall_ratio": (round(gst["median"] / wall_gbps, 3) if wall_gbps > 0 else None),
        "timing_sane": bool(timing_sane),
        "timing_note": ("hipEvent GB/s cross-checked against wall-clock GB/s over the "
                        f"whole rep loop; event/wall must be <= {TIMING_SANITY_FACTOR} "
                        "(wall carries launch+memset+sync+ctypes overhead, so it is "
                        "always the smaller figure). A config that fails this is "
                        "excluded from `best` and raises a blocking error."),
        "checksum_expected": expect,
        "checksum_got": sums[-1] if sums else None,
        "checksum_ok": bool(ok),
        "checksum_note": ("buffer holds dword[i]==i and every pattern covers a prefix "
                          "exactly once, so a wrong sum means the read did not land on "
                          "the pages under test"),
        "ms_median": round(st["median"], 4),
    })
    return base


def sweep_arm(lib, ptr: int, nbytes: int, arm: str, device: int, args) -> list[dict]:
    out = []
    for pattern in args.patterns:
        # linear has no lane constraint; tiled/random cooperate lanes-per-burst
        lanes = 0 if pattern == "linear" else (
            (128 if pattern == "random128" else args.seg_bytes) // 16)
        for threads in args.threads:
            for mult in args.block_mults:
                blocks = max(1, args.cu_units * mult)
                if lanes and threads % lanes:
                    continue
                out.append(measure_pattern(
                    lib, ptr, nbytes, pattern, blocks, threads,
                    args.row_bytes, args.seg_bytes, args.stride_bytes,
                    args.reps, arm, device,
                    max_config_seconds=args.max_config_seconds))
    return out


# ---------------------------------------------------------------------------
# Coherence arm (plan section 11 item 6)
# ---------------------------------------------------------------------------

COH_BYTES = 1 << 20          # 1 MiB region
COH_LINE = 128


def _line_sums_host(lib, ptr: int, off: int, nlines: int, line: int) -> list[int]:
    buf = (ctypes.c_uint * nlines)()
    rc = lib.p1_line_checksums(ctypes.c_void_p(ptr), off, nlines, line, buf)
    K(lib, rc, "p1_line_checksums")
    return list(buf)


def _expected_lines(words: list[int], line_words: int) -> list[int]:
    return [sum(words[i:i + line_words]) & 0xFFFFFFFF
            for i in range(0, len(words), line_words)]


def coherence_arm(lib, host_ptr: int, dev_ptr: int, arm: str, device: int,
                  cpu_ok: dict) -> dict:
    """CPU-write -> kernel-read and kernel-write -> CPU-read, at 128 B line detail.

    `host_ptr` is what the CPU dereferences; `dev_ptr` is what the kernel is
    given.  For host-located VMM pages these are the same address; for
    hipHostMalloc they may differ (hipHostGetDevicePointer).
    """
    res: dict = {"arm": arm, "device": device, "region_bytes": COH_BYTES,
                 "line_bytes": COH_LINE, "cpu_access": cpu_ok,
                 "host_va": f"0x{host_ptr:x}", "device_va": f"0x{dev_ptr:x}",
                 "maps_entry": maps_entry_for(host_ptr), "tests": {}}
    if not cpu_ok.get("readable") or not cpu_ok.get("writable"):
        res["skipped"] = "VA is not CPU-accessible; CPU-side arms not applicable"
        return res

    nlines = COH_BYTES // COH_LINE
    line_words = COH_LINE // 4
    nwords = COH_BYTES // 4
    view = (ctypes.c_uint * nwords).from_address(host_ptr)

    # --- 1. device write -> CPU read (the direction vmm_probe2 already proved)
    rc = lib.p1_fill_range(ctypes.c_void_p(dev_ptr), 0, COH_BYTES, 0xA0000000)
    K(lib, rc, "p1_fill_range(coherence A)")
    K(lib, lib.p1_sync(), "p1_sync")
    want_a = [(0xA0000000 + j) & 0xFFFFFFFF for j in range(nwords)]
    got_a = list(view)
    bad_a = [i for i in range(nwords) if got_a[i] != want_a[i]]
    res["tests"]["dev_write_then_cpu_read"] = {
        "mismatch_words": len(bad_a),
        "first_mismatch_word": bad_a[0] if bad_a else None,
        "pass": not bad_a,
    }

    # --- 2. CPU write -> kernel read, with the GPU caches ALREADY populated by
    #        a prior read of the same lines (this is what makes staleness visible).
    prior = _line_sums_host(lib, dev_ptr, 0, nlines, COH_LINE)
    exp_prior = _expected_lines(want_a, line_words)
    res["tests"]["prewarm_read_matches_device_write"] = {
        "mismatch_lines": sum(1 for a, b in zip(prior, exp_prior) if a != b),
        "pass": prior == exp_prior,
    }

    want_b = [(0xB0000000 + j) & 0xFFFFFFFF for j in range(nwords)]
    for i in range(nwords):
        view[i] = want_b[i]
    exp_b = _expected_lines(want_b, line_words)

    got_nofence = _line_sums_host(lib, dev_ptr, 0, nlines, COH_LINE)
    stale_nofence = [i for i in range(nlines)
                     if got_nofence[i] != exp_b[i] and got_nofence[i] == exp_prior[i]]
    res["tests"]["cpu_write_then_kernel_read_no_explicit_flush"] = {
        "mismatch_lines": sum(1 for a, b in zip(got_nofence, exp_b) if a != b),
        "lines_serving_stale_prior": len(stale_nofence),
        "first_stale_line": stale_nofence[0] if stale_nofence else None,
        "pass": got_nofence == exp_b,
    }

    # --- 3. same, after a CPU seq-cst fence
    lib.p1_cpu_fence()
    got_fence = _line_sums_host(lib, dev_ptr, 0, nlines, COH_LINE)
    res["tests"]["cpu_write_then_kernel_read_after_fence"] = {
        "mismatch_lines": sum(1 for a, b in zip(got_fence, exp_b) if a != b),
        "pass": got_fence == exp_b,
    }

    # --- 4. same, after an explicit clflush of the range
    lib.p1_cpu_flush(ctypes.c_void_p(host_ptr), COH_BYTES)
    got_flush = _line_sums_host(lib, dev_ptr, 0, nlines, COH_LINE)
    res["tests"]["cpu_write_then_kernel_read_after_clflush"] = {
        "mismatch_lines": sum(1 for a, b in zip(got_flush, exp_b) if a != b),
        "pass": got_flush == exp_b,
    }

    # --- 5. sub-line granularity: rewrite only the FIRST 64 B of each 128 B line.
    want_c = list(want_b)
    half = line_words // 2
    for ln in range(nlines):
        for j in range(half):
            idx = ln * line_words + j
            want_c[idx] = (0xC0000000 + idx) & 0xFFFFFFFF
            view[idx] = want_c[idx]
    lib.p1_cpu_fence()
    exp_c = _expected_lines(want_c, line_words)
    got_c = _line_sums_host(lib, dev_ptr, 0, nlines, COH_LINE)
    whole_line_stale = sum(1 for i in range(nlines)
                           if got_c[i] != exp_c[i] and got_c[i] == exp_b[i])
    res["tests"]["cpu_partial_line_write_then_kernel_read"] = {
        "written_bytes_per_line": COH_LINE // 2,
        "mismatch_lines": sum(1 for a, b in zip(got_c, exp_c) if a != b),
        "lines_stale_at_whole_line_granularity": whole_line_stale,
        "pass": got_c == exp_c,
        "note": ("mismatch with lines_stale_at_whole_line_granularity == mismatch_lines "
                 "means visibility is coarser than 64 B"),
    }

    res["verdict"] = {
        "cpu_write_visible_to_kernel_without_flush":
            res["tests"]["cpu_write_then_kernel_read_no_explicit_flush"]["pass"],
        "cpu_write_visible_after_clflush":
            res["tests"]["cpu_write_then_kernel_read_after_clflush"]["pass"],
        "device_write_visible_to_cpu":
            res["tests"]["dev_write_then_cpu_read"]["pass"],
    }
    return res


# ---------------------------------------------------------------------------
# Placement discovery -- WHICH hipMemCreate location type actually lands in
# host RAM.  Without this the probe can report an HBM number as a "host-page
# read": hipMemCreate(location.type=Host) is known to return hipSuccess while
# allocating VRAM on this box.
# ---------------------------------------------------------------------------

HOST_LOC_CANDIDATES = (HIP_MEM_LOCATION_TYPE_HOST,
                       HIP_MEM_LOCATION_TYPE_HOST_NUMA,
                       HIP_MEM_LOCATION_TYPE_HOST_NUMA_CURRENT)


def discover_host_placement(hip, lib, dev: int, args, bdf: str | None = None) -> list[dict]:
    size = args.placement_probe_bytes
    out: list[dict] = []
    for loc in HOST_LOC_CANDIDATES:
        rec: dict = {"loc_type": loc, "loc_type_name": LOC_NAMES[loc],
                     "probe_bytes": size}
        region = None
        try:
            region = VmmRegion(hip, lib, dev, size, loc, size, bdf=bdf)
        except ProbeError as e:
            rec["usable"] = False
            rec["error"] = str(e)
            out.append(rec)
            continue
        try:
            for k in ("granularity", "placement", "pointer_attrs", "va", "maps_entry",
                      "chunk_bytes"):
                rec[k] = region.info.get(k)
            # NOTE: cpu_writable is destructive; it restores the original bytes and
            # in any case runs BEFORE the fill below, so the checksum is unaffected.
            rec["cpu_access"] = {**cpu_readable(region.ptr, 4096),
                                 **cpu_writable(region.ptr, 4096)}
            K(lib, lib.p1_fill_range(ctypes.c_void_p(region.ptr), 0, size, 0),
              f"fill(placement probe loc={loc})")
            K(lib, lib.p1_sync(), "p1_sync")
            rec["placement_after_touch"] = region.resample_placement(lib)
            m = measure_pattern(lib, region.ptr, size, "linear",
                                max(1, args.cu_units * 8), 256,
                                args.row_bytes, args.seg_bytes, args.stride_bytes,
                                3, f"placement_probe_loc{loc}", dev,
                                max_config_seconds=args.max_config_seconds)
            rec["probe_read_gbps_median"] = m["gbps_median"]
            rec["probe_read_checksum_ok"] = m["checksum_ok"]
            rec["probe_read_timing_sane"] = m.get("timing_sane")
            rec["probe_read_note"] = ("classification signal only -- this probe size may "
                                      "be partly cache-resident and is NOT a reported "
                                      "bandwidth")
            rec["usable"] = True
        finally:
            region.close()
        out.append(rec)
    return out


def placement_of(rec: dict | None) -> str:
    """Prefer the after-touch sample: lazy page commit makes the at-backing
    delta ~0 for BOTH media, which would read as 'indeterminate'."""
    if not rec or not rec.get("usable", True):
        return "unusable"
    for key in ("placement_after_touch", "placement"):
        cls = (rec.get(key) or {}).get("classification")
        if cls and cls != "indeterminate":
            return cls
    return ((rec.get("placement") or {}).get("classification")) or "indeterminate"


def placement_record_of(rec: dict | None) -> dict:
    """The classify_placement dict that `placement_of` actually answered from."""
    if not rec:
        return {}
    for key in ("placement_after_touch", "placement"):
        p = rec.get(key) or {}
        if p.get("classification") and p["classification"] != "indeterminate":
            return p
    return rec.get("placement") or {}


# ---------------------------------------------------------------------------
# Per-device run
# ---------------------------------------------------------------------------

def run_device(lib, hip, dev: int, args, checkpoint=lambda d=None: None) -> dict:
    K(lib, lib.p1_set_device(dev), f"p1_set_device({dev})")

    name = ctypes.create_string_buffer(256)
    arch = ctypes.create_string_buffer(256)
    pci = ctypes.create_string_buffer(64)
    mp = ctypes.c_int(0)
    warp = ctypes.c_int(0)
    clk = ctypes.c_int(0)
    tot = ctypes.c_ulonglong(0)
    K(lib, lib.p1_device_info(dev, name, 256, arch, 256, pci, 64,
                              ctypes.byref(mp), ctypes.byref(warp),
                              ctypes.byref(clk), ctypes.byref(tot)),
      "p1_device_info")
    freeb = ctypes.c_size_t(0)
    totb = ctypes.c_size_t(0)
    K(lib, lib.p1_mem_info(ctypes.byref(freeb), ctypes.byref(totb)), "p1_mem_info")

    d: dict = {
        "hip_device_index": dev,
        "physical_card_note": ("ROCR_VISIBLE_DEVICES=0,1 so HIP index == physical "
                               "gfx1201 card index; card 0 = RX 9070 XT, card 1 = RX 9070"),
        "name": name.value.decode(),
        "gcn_arch": arch.value.decode(),
        "pci_bus_id": pci.value.decode(),
        "multiprocessor_count_reported": mp.value,
        "multiprocessor_count_note": "HIP reports WGPs on RDNA, not CUs (64-CU card answers 32)",
        "warp_size": warp.value,
        "clock_khz": clk.value,
        "total_global_mem": int(tot.value),
        "vram_free_before": int(freeb.value),
        "vram_total": int(totb.value),
        "cu_units_basis": max(1, mp.value),
        "amdgpu_sysfs_before": {},
        "measurements": [],
        "copy_engine": {},
        "coherence": [],
        "vmm": {},
        "placement_discovery": [],
        "findings": [],
        "errors": [],       # BLOCKING: invalidate the bandwidth answer
        "warnings": [],     # non-blocking: recorded, but do not void the verdict
        "best": {},
        # Placeholder so a device record that is checkpointed and then dies on a
        # hard GPU fault still satisfies the artifact schema instead of turning a
        # real failure into an unreadable "schema error".
        "verdict": {"result": "NOT_MEASURED",
                    "reason": "device run did not complete"},
    }
    bdf = (d["pci_bus_id"] or "").strip().lower()
    d["amdgpu_sysfs_before"] = card_mem(bdf)
    # Which PCIe link do this card's host-page numbers actually cross?  The two
    # cards are trained at DIFFERENT speeds on this box, so a host bandwidth
    # figure is only interpretable next to its link.
    d["pcie_link"] = pcie_link_chain(bdf)
    d["amdgpu_sysfs_note"] = (
        "per-card amdgpu accounting for THIS physical card, matched by the PCI BDF "
        "hipDeviceGetPCIBusId reported; this is the primary placement signal")

    checkpoint(d)   # publish the device record early: a hard GPU fault (which
                    # would kill the process outright) then still leaves evidence
    if not d["gcn_arch"].startswith("gfx12"):
        _fail(f"device {dev} is '{d['gcn_arch']}' -- not a gfx12xx compute card. "
              f"Refusing to run (the Ryzen iGPU must never be a target).")
    if not d["amdgpu_sysfs_before"]:
        d["warnings"].append(
            f"no amdgpu sysfs counters found for PCI {bdf!r}; placement falls back to "
            f"hipMemGetInfo + MemAvailable, which is host-RAM-noise sensitive")

    n = args.bytes
    # Worst case device residency: vmm_dev + dev_plain + memcpy_dst, PLUS vmm_host
    # if hipMemCreate(location=Host) turns out to allocate VRAM (which is exactly
    # the failure this probe exists to detect), plus the transient discovery probe.
    need_dev = 4 * n + args.placement_probe_bytes + (1 << 29)
    if int(freeb.value) < need_dev:
        _fail(f"device {dev}: only {freeb.value/2**30:.2f} GiB VRAM free, need "
              f"{need_dev/2**30:.2f} GiB (4 x working set + placement probe + 512 MiB "
              f"slack; the 4th copy is vmm_host in the case where location=Host "
              f"silently allocates VRAM). Refusing to run rather than report a "
              f"partial arm set.")
    avail_kb = (collect_box_state("precheck", with_smi=False)["meminfo_kb"]["MemAvailable"] or 0)
    if avail_kb * 1024 < 3 * n + (2 << 30):
        _fail(f"host: MemAvailable {avail_kb/2**20:.1f} GiB < needed "
              f"{(3*n + (2<<30))/2**30:.1f} GiB of pinnable RAM. Refusing to run.")

    dev_plain = ctypes.c_void_p()
    pinned_def = ctypes.c_void_p()
    pinned_nc = ctypes.c_void_p()
    memcpy_dst = ctypes.c_void_p()
    vmm_host = vmm_dev = None
    vmm_alt = None
    try:
        # ---- STEP 0: which location type actually lands in host RAM?
        print("  [placement] probing hipMemCreate location types ...", flush=True)
        d["placement_discovery"] = discover_host_placement(hip, lib, dev, args, bdf=bdf)
        for rec in d["placement_discovery"]:
            print(f"    loc={rec['loc_type']} {rec['loc_type_name']}: "
                  f"{placement_of(rec)}"
                  f"{'  probe_read=' + str(rec.get('probe_read_gbps_median')) + ' GB/s' if rec.get('usable') else ''}",
                  flush=True)
        by_loc = {r["loc_type"]: r for r in d["placement_discovery"]}
        # a fallback host location type, only if it verifiably lands in host RAM
        alt_loc = next((loc for loc in HOST_LOC_CANDIDATES[1:]
                        if placement_of(by_loc.get(loc)) == "host_resident"), None)

        # ---- full-size VMM regions, one at a time: create -> touch every page
        # through the device pointer -> resample placement, BEFORE the next
        # allocation exists.  Resampling later would attribute every subsequent
        # allocation's bytes to this region.
        def _make_vmm(key: str, loc: int):
            reg = VmmRegion(hip, lib, dev, n, loc, args.chunk_bytes, bdf=bdf)
            K(lib, lib.p1_fill_range(ctypes.c_void_p(reg.ptr), 0, n, 0), f"fill({key})")
            K(lib, lib.p1_sync(), "p1_sync")
            reg.resample_placement(lib)
            d["vmm"][key] = reg.info
            print(f"  [vmm:{key}] {LOC_NAMES.get(loc, loc)} -> "
                  f"{placement_of(reg.info)}", flush=True)
            return reg

        vmm_host = _make_vmm("host", HIP_MEM_LOCATION_TYPE_HOST)
        if placement_of(vmm_host.info) == "device_resident":
            d["findings"].append({
                "id": "vmm_host_location_not_honored",
                "severity": "decision-critical",
                "summary": ("hipMemCreate(location.type=hipMemLocationTypeHost) returned "
                            "hipSuccess for every chunk, and hipMemMap/hipMemSetAccess "
                            "succeeded, but the pages consumed VRAM rather than host RAM"),
                "evidence": {
                    "full_region": vmm_host.info.get("placement_after_touch"),
                    "at_backing": vmm_host.info.get("placement"),
                    "discovery_probe": (by_loc.get(HIP_MEM_LOCATION_TYPE_HOST) or {}).get(
                        "placement_after_touch"),
                    "pointer_attrs": vmm_host.info.get("pointer_attrs"),
                },
                "consequence": ("the plan's mixed device/host single-VA stack (sections 2 "
                                "and 9) has no host tier through this API; any "
                                "'host-page read bandwidth' taken from this arm would be "
                                "an HBM number in disguise, so it is excluded from the "
                                "verdict"),
            })
        vmm_dev = _make_vmm("device", HIP_MEM_LOCATION_TYPE_DEVICE)
        if alt_loc is not None:
            vmm_alt = _make_vmm("host_alt", alt_loc)

        # Control: the device-located region MUST classify as device_resident.
        # If it does not, the placement method itself is unreliable here and no
        # placement claim in this record can be trusted.
        if placement_of(vmm_dev.info) != "device_resident":
            d["findings"].append({
                "id": "placement_method_control_failed",
                "severity": "blocking",
                "summary": ("the known-device VMM region did not classify as "
                            f"device_resident (got {placement_of(vmm_dev.info)}); the "
                            "VRAM/MemAvailable delta method is not reliable on this box "
                            "right now (concurrent GPU or RAM activity?)"),
                "evidence": vmm_dev.info.get("placement_after_touch"),
                "consequence": ("every placement classification in this record is "
                                "untrustworthy; re-run on a quieter box before drawing "
                                "any conclusion from the vmm_host arm"),
            })
            d["errors"].append("placement method control failed (vmm_dev not classified "
                               "device_resident) -- placement claims are untrustworthy")

        K(lib, lib.p1_malloc_device(ctypes.byref(dev_plain), n), "hipMalloc(dev_plain)")
        K(lib, lib.p1_malloc_device(ctypes.byref(memcpy_dst), n), "hipMalloc(memcpy_dst)")
        # hipHostMalloc is host RAM by construction, but measure it anyway: the
        # verdict may rest on it, and an asserted placement is not a measured one.
        d["pinned"] = {}
        pdef_dev = ctypes.c_void_p()
        pnc_dev = ctypes.c_void_p()
        for key, out, dptr, flags in (
            ("pinned_default", pinned_def, pdef_dev, HHM_MAPPED | HHM_PORTABLE),
            ("pinned_noncoh", pinned_nc, pnc_dev,
             HHM_MAPPED | HHM_PORTABLE | HHM_NONCOHERENT),
        ):
            v0, a0, s0 = _vram_free(lib), _mem_available_bytes(), card_mem(bdf)
            K(lib, lib.p1_host_malloc(ctypes.byref(out), n, flags),
              f"hipHostMalloc({key}, flags=0x{flags:x})")
            K(lib, lib.p1_host_dev_ptr(ctypes.byref(dptr), out),
              f"hipHostGetDevicePointer({key})")
            K(lib, lib.p1_fill_range(ctypes.c_void_p(dptr.value), 0, n, 0),
              f"touch({key})")
            K(lib, lib.p1_sync(), "p1_sync")
            d["pinned"][key] = {
                "flags": f"0x{flags:x}",
                "host_ptr": f"0x{out.value:x}",
                "device_ptr": f"0x{dptr.value:x}",
                "same_va": out.value == dptr.value,
                "pointer_attrs": pointer_attrs(lib, int(dptr.value)),
                "placement": classify_placement(n, v0 - _vram_free(lib),
                                                a0 - _mem_available_bytes(),
                                                s0, card_mem(bdf)),
            }
            print(f"  [pinned:{key}] -> {placement_of(d['pinned'][key])}", flush=True)
            if placement_of(d["pinned"][key]) == "device_resident":
                d["findings"].append({
                    "id": f"{key}_not_host_resident",
                    "severity": "blocking",
                    "summary": (f"hipHostMalloc({key}) consumed VRAM rather than host "
                                f"RAM -- the fallback host mechanism is not host either"),
                    "evidence": d["pinned"][key]["placement"],
                    "consequence": "no host-resident mechanism remains; P1 is unanswerable",
                })

        arms = [
            ("vmm_host", vmm_host.ptr),
            ("vmm_dev", vmm_dev.ptr),
            ("dev_plain", int(dev_plain.value)),
            ("pinned_default", int(pdef_dev.value)),
            ("pinned_noncoh", int(pnc_dev.value)),
        ]
        if vmm_alt is not None:
            arms.insert(1, ("vmm_host_alt", vmm_alt.ptr))
        if args.arms:
            arms = [a for a in arms if a[0] in args.arms]
            if not arms:
                _fail(f"--arms {args.arms} selected nothing")
            if not any(a[0] in ("vmm_dev", "dev_plain") for a in arms):
                d["warnings"].append(
                    "--arms excluded both HBM reference arms (vmm_dev, dev_plain): the "
                    "provenance cross-check loses its 'is this arm reading at HBM speed?' "
                    "half and falls back to the copy-engine bound alone")

        # ---- fill every buffer THROUGH THE DEVICE POINTER (dword[i] = i).
        # Never CPU stores into host-located pages: coherence granularity is
        # exactly what arm 6 is measuring, so the fill must not depend on it.
        for arm, ptr in arms:
            K(lib, lib.p1_fill_range(ctypes.c_void_p(ptr), 0, n, 0), f"fill({arm})")
        K(lib, lib.p1_sync(), "p1_sync(after fill)")

        # ---- bandwidth sweeps
        for arm, ptr in arms:
            t0 = time.perf_counter()
            ms = sweep_arm(lib, ptr, n, arm, dev, args)
            d["measurements"].extend(ms)
            bad = [m for m in ms if not m["checksum_ok"]]
            if bad:
                d["errors"].append(
                    f"{arm}: {len(bad)}/{len(ms)} configs returned a WRONG checksum "
                    f"-- the read did not land on the pages under test")
            insane = [m for m in ms if m.get("measured") and not m.get("timing_sane")]
            if insane:
                d["errors"].append(
                    f"{arm}: {len(insane)}/{len(ms)} configs failed the hipEvent-vs-wall "
                    f"timing sanity fence (event GB/s > {TIMING_SANITY_FACTOR}x wall "
                    f"GB/s, or a non-positive event time) -- the timer, not the kernel, "
                    f"produced those numbers; they are excluded from `best`")
            skipped = [m for m in ms if not m.get("measured", True)]
            if skipped:
                d["warnings"].append(
                    f"{arm}: {len(skipped)}/{len(ms)} configs exceeded "
                    f"--max-config-seconds {args.max_config_seconds}s and were recorded "
                    f"from the warm-up rep only (excluded from `best`; skipping only "
                    f"ever removes SLOW points, so it cannot inflate the answer)")
            usable = [m["gbps_median"] for m in ms
                      if m.get("measured") and m.get("checksum_ok")
                      and m.get("timing_sane") and m.get("gbps_median") is not None]
            print(f"  [{arm}] {len(ms)} configs in {time.perf_counter()-t0:.1f}s  "
                  f"best={max(usable, default=0.0):.2f} GB/s"
                  f"{'  CHECKSUM FAIL' if bad else ''}"
                  f"{'  TIMING INSANE' if insane else ''}"
                  f"{('  ' + str(len(skipped)) + ' too-slow') if skipped else ''}",
                  flush=True)
            checkpoint()

        # ---- copy-engine references (these OVERWRITE memcpy_dst only).
        # reps+1 issued, the FIRST discarded as warm-up (first-touch / lazy setup).
        reps = max(args.reps, 3)
        nbuf = reps + 1
        buf = (ctypes.c_float * nbuf)()
        ce = {}
        csum = ctypes.c_uint(0)
        ccov = ctypes.c_size_t(0)
        cms = ctypes.c_float(0.0)
        link_watch = LinkSampler(bdf)
        for label, src, kind in (
            ("h2d_pinned_default", pinned_def, 1),
            ("h2d_pinned_noncoherent", pinned_nc, 1),
            ("from_vmm_host_va", ctypes.c_void_p(vmm_host.ptr), 4),
            ("d2d_vmm_dev", ctypes.c_void_p(vmm_dev.ptr), 3),
        ):
            # Poison the destination first: a copy that silently moved nothing
            # would otherwise report an enormous rate, and this figure is the
            # DENOMINATOR of the relative kill threshold.
            K(lib, lib.p1_fill_range(memcpy_dst, 0, n, 0xDEAD0000),
              f"poison(memcpy_dst before {label})")
            K(lib, lib.p1_sync(), "p1_sync")
            with link_watch:      # sample the link while it is actually saturated
                rc = lib.p1_memcpy_bw(memcpy_dst, src, n, kind, nbuf, buf)
            if rc != 0:
                msg = lib.p1_err(rc)
                ce[label] = {"supported": False, "rc": rc,
                             "error": msg.decode() if msg else "?"}
                continue
            # Every source buffer holds dword[i]==i, so the destination must too.
            K(lib, lib.p1_run(0, memcpy_dst, n, max(1, args.cu_units * 8), 256,
                              args.row_bytes, args.seg_bytes, args.stride_bytes,
                              ctypes.byref(csum), ctypes.byref(ccov), ctypes.byref(cms)),
              f"verify(memcpy_dst after {label})")
            moved = int(csum.value) == expected_prefix_sum(int(ccov.value))
            allt = [float(x) for x in buf]
            times = allt[1:]
            gb = [n / (t * 1e-3) / 1e9 if t > 0 else 0.0 for t in times]
            ce[label] = {"supported": True, "bytes": n, "reps": reps,
                         "warmup_discarded_ms": round(allt[0], 4),
                         "ms": [round(t, 4) for t in times],
                         "gbps_median": round(statistics.median(gb), 3),
                         "gbps_max": round(max(gb), 3),
                         "gbps_min": round(min(gb), 3),
                         "destination_verified": bool(moved),
                         "destination_note": ("destination poisoned before the copy and "
                                              "checksummed after: proves the bytes really "
                                              "moved, so the rate is not a no-op's rate")}
            if not moved:
                d["errors"].append(
                    f"copy_engine[{label}]: the destination checksum is WRONG after the "
                    f"copy -- the reported {ce[label]['gbps_median']} GB/s is not a rate "
                    f"for moving these bytes, and it is the denominator of the relative "
                    f"kill threshold")
        # The H2D figure is the DENOMINATOR of the relative kill threshold.  A
        # degraded copy engine would silently lower the bar a host read has to
        # clear, so flag any large departure from the recorded 28.1 GB/s prior.
        h2d = ce.get("h2d_pinned_default") or {}
        if h2d.get("supported"):
            g = h2d["gbps_median"]
            h2d["prior_gbps"] = COPY_ENGINE_PRIOR_GBPS
            h2d["frac_of_prior"] = round(g / COPY_ENGINE_PRIOR_GBPS, 3)
            if g < COPY_ENGINE_SANITY_FLOOR_GBPS:
                d["warnings"].append(
                    f"copy-engine H2D measured {g:.2f} GB/s, far below the recorded "
                    f"{COPY_ENGINE_PRIOR_GBPS} GB/s prior for this box: the RELATIVE kill "
                    f"threshold (0.5x) is derived from a degraded denominator and is "
                    f"correspondingly easy to clear. Weight the absolute "
                    f"{ABS_KILL_GBPS} GB/s floor instead.")
            if g > 2.0 * COPY_ENGINE_PRIOR_GBPS:
                d["warnings"].append(
                    f"copy-engine H2D measured {g:.2f} GB/s, implausibly far ABOVE the "
                    f"{COPY_ENGINE_PRIOR_GBPS} GB/s prior for this PCIe link; suspect the "
                    f"timer or the source buffer's placement before trusting the relative "
                    f"threshold derived from it")
        # from_vmm_host_va is only an H2D copy if vmm_host is actually host-backed;
        # label it with the measured placement so the table cannot be misread.
        if "from_vmm_host_va" in ce:
            ce["from_vmm_host_va"]["source_placement"] = placement_of(
                (d.get("vmm") or {}).get("host"))
        d["copy_engine"] = ce
        # The link state that the H2D figures above actually crossed.  Merged into
        # the idle reading taken at device-record time so the artifact carries both.
        d.setdefault("pcie_link", {})["under_load"] = link_watch.result()
        checkpoint()

        # ---- coherence arm (destroys the pattern; runs last)
        coh_targets = [("vmm_host", vmm_host.ptr, vmm_host.ptr)]
        if vmm_alt is not None:
            coh_targets.append(("vmm_host_alt", vmm_alt.ptr, vmm_alt.ptr))
        coh_targets += [
            ("pinned_default", int(pinned_def.value), int(pdef_dev.value)),
            ("pinned_noncoh", int(pinned_nc.value), int(pnc_dev.value)),
        ]
        for arm, hostptr, devptr in coh_targets:
            if args.arms and arm not in args.arms:
                continue
            acc = {}
            # Probe the FULL region the coherence arm will raw-dereference, not a
            # 4 KiB prefix: the arm builds a ctypes view over all COH_BYTES, so a
            # VA whose first page is CPU-accessible while a later page is not
            # would fault the process instead of being recorded as inaccessible.
            acc.update(cpu_readable(hostptr, COH_BYTES))
            acc.update(cpu_writable(hostptr, COH_BYTES))
            acc["probe_bytes"] = COH_BYTES
            acc["same_va"] = hostptr == devptr
            try:
                d["coherence"].append(
                    coherence_arm(lib, hostptr, devptr, arm, dev, acc))
            except ProbeError as e:
                d["coherence"].append({"arm": arm, "device": dev, "error": str(e)})
                # NON-BLOCKING: the coherence arm answers plan section 11 item 6.
                # It has no bearing on whether the measured bandwidth is real, so it
                # must not turn a good P1 answer into INVALID and force a re-run.
                d["warnings"].append(f"coherence[{arm}]: {e}")
            checkpoint()

        K(lib, lib.p1_mem_info(ctypes.byref(freeb), ctypes.byref(totb)), "p1_mem_info")
        d["vram_free_after"] = int(freeb.value)
    finally:
        for p in (dev_plain, memcpy_dst):
            if p and p.value:
                lib.p1_free_device(p)
        for p in (pinned_def, pinned_nc):
            if p and p.value:
                lib.p1_host_free(p)
        for r in (vmm_host, vmm_dev, vmm_alt):
            if r is not None:
                r.close()

    d["best"] = summarize_best(d["measurements"])
    d["verdict"] = device_verdict(d)
    return d


def summarize_best(ms: list[dict]) -> dict:
    """Best occupancy point per (arm, pattern).

    A config is only eligible if it (a) read the right bytes (checksum), (b) was
    actually timed to completion (`measured`), and (c) passed the hipEvent-vs-wall
    timing fence.  Without (c) a broken timer's absurd GB/s would win the max and
    become "the answer" -- the exact confident-wrong-number this probe exists to
    prevent.
    """
    best: dict = {}
    for m in ms:
        if not m.get("checksum_ok"):
            continue
        if not m.get("measured", True):
            continue
        if not m.get("timing_sane", True):
            continue
        if m.get("gbps_median") is None:
            continue
        key = f"{m['arm']}.{m['pattern']}"
        cur = best.get(key)
        if cur is None or m["gbps_median"] > cur["gbps_median"]:
            best[key] = {"gbps_median": m["gbps_median"], "blocks": m["blocks"],
                         "threads": m["threads"], "waves_in_flight": m["waves_in_flight"],
                         "cover_bytes": m["cover_bytes"], "reps": m["reps"],
                         "gbps_min": m["gbps_min"], "gbps_max": m["gbps_max"],
                         "ms_spread_pct": m["ms_spread_pct"],
                         "event_vs_wall_ratio": m.get("event_vs_wall_ratio")}
    return best


def device_verdict(d: dict) -> dict:
    """Verdict on the P1 kill criterion, gated on VERIFIED host placement.

    A number is only allowed to answer 'how fast can a kernel read host-resident
    weights' if the pages it came from are proven to be in host RAM.  Otherwise
    the arm is reported, flagged, and excluded from the verdict.
    """
    best = d.get("best", {})
    ce = (d.get("copy_engine") or {}).get("h2d_pinned_default") or {}
    # Only a copy whose destination checksummed clean may set the relative bar.
    copy_ok = bool(ce.get("supported")) and ce.get("destination_verified", True)
    copy_gbps = ce.get("gbps_median") if copy_ok else None
    rel_thr = (REL_KILL_FRACTION * copy_gbps) if copy_gbps else None
    by_loc = {r["loc_type"]: r for r in d.get("placement_discovery", [])}

    def trio(arm: str) -> dict:
        return {p: (best.get(f"{arm}.{p}") or {}).get("gbps_median") for p in PATTERNS}

    # Candidate host mechanisms, most-desired first.  'placement' is how we know
    # the bytes really came from host RAM.
    vmm = d.get("vmm") or {}
    # Prefer the FULL-SIZE region's own classification; fall back to the small
    # discovery probe if the full region was never built.
    host_pl = (placement_of(vmm["host"]) if "host" in vmm
               else placement_of(by_loc.get(HIP_MEM_LOCATION_TYPE_HOST)))
    host_ev = (placement_record_of(vmm["host"]) if "host" in vmm
               else placement_record_of(by_loc.get(HIP_MEM_LOCATION_TYPE_HOST)))
    cands: list[dict] = [
        {"arm": "vmm_host",
         "mechanism": "hipMemCreate(location=Host) + hipMemMap onto a device VA",
         "placement": host_pl,
         "placement_source": ("per-card amdgpu sysfs vram_used/gtt_used delta across "
                              "backing + a full device-side touch of the working-set "
                              "region, cross-checked against hipMemGetInfo/MemAvailable"),
         "placement_evidence": host_ev},
    ]
    if "host_alt" in vmm and placement_of(vmm["host_alt"]) in (
            "host_resident", "not_in_vram_but_host_delta_unclear"):
        cands.append({
            "arm": "vmm_host_alt",
            "mechanism": (f"hipMemCreate(location="
                          f"{vmm['host_alt'].get('loc_type_name')}) + hipMemMap"),
            "placement": placement_of(vmm["host_alt"]),
            "placement_source": "per-card amdgpu sysfs delta on the full region",
            "placement_evidence": placement_record_of(vmm["host_alt"])})
    # hipHostMalloc is host RAM by construction; a MEASURED contradiction still
    # wins, but absence of a clean delta does not disqualify it.
    pinned_meas = placement_of((d.get("pinned") or {}).get("pinned_default"))
    cands.append({
        "arm": "pinned_default",
        "mechanism": "hipHostMalloc(Mapped) + hipHostGetDevicePointer (zero-copy)",
        "placement": ("device_resident" if pinned_meas == "device_resident"
                      else "host_resident"),
        "placement_source": (f"host by construction (hipHostMalloc); measured "
                             f"classification was '{pinned_meas}'"),
        "placement_evidence": placement_record_of((d.get("pinned") or {}).get("pinned_default")),
        "limitation": ("cannot be interleaved with device-located pages inside one "
                       "reserved VA range, so it does not give the plan's T2 mixed "
                       "device/host single stack")})

    # HBM reference from THIS run: the fastest checksum-clean, timing-sane read of
    # a known-device buffer.  A "host" arm approaching this is not crossing PCIe.
    hbm_ref = max([g for g in ((best.get("vmm_dev.linear") or {}).get("gbps_median"),
                               (best.get("dev_plain.linear") or {}).get("gbps_median"))
                   if g is not None], default=None)

    control_ok = (placement_of(vmm.get("device")) == "device_resident") if "device" in vmm else False

    for c in cands:
        c["gbps"] = trio(c["arm"])
        t = c["gbps"].get("tiled")
        lin = c["gbps"].get("linear")
        # ---- NEGATIVE cross-check (physics): bytes that crossed PCIe cannot
        # materially outrun the copy engine, and cannot approach HBM.  The old
        # 2.0x-copy-engine bound was far too loose -- a 50 GB/s read on a ~28 GB/s
        # link would have passed it and been reported as a host-page number.
        # Both patterns are tested; `linear` is the fastest and most sensitive.
        viol = []
        for pname, g in (("linear", lin), ("tiled", t)):
            if g is None:
                continue
            if copy_gbps and g > PROVENANCE_MAX_OVER_COPY * copy_gbps:
                viol.append(f"{pname} read {g:.1f} GB/s > {PROVENANCE_MAX_OVER_COPY}x the "
                            f"copy-engine H2D figure {copy_gbps:.1f} GB/s")
            if hbm_ref and g >= PROVENANCE_MAX_FRAC_OF_HBM * hbm_ref:
                viol.append(f"{pname} read {g:.1f} GB/s >= "
                            f"{PROVENANCE_MAX_FRAC_OF_HBM}x the measured HBM reference "
                            f"{hbm_ref:.1f} GB/s")
        if viol:
            c["provenance_violation"] = (
                "; ".join(viol) + " -- these bytes cannot have crossed PCIe, so the "
                "region is device-resident whatever the placement deltas said")
            c["placement"] = "device_resident"
            c["placement_source"] = (str(c.get("placement_source", "")) +
                                     " | OVERRIDDEN by the bandwidth cross-check")

        # ---- POSITIVE corroboration (the same physics, forward).  MemAvailable is
        # whole-box noise; a region can be genuinely host-backed and still fail to
        # show a clean delta, which would wrongly demote the plan's own mechanism to
        # "unproven" and hand the verdict to the substitute.  A region that (a) the
        # control-validated method says is NOT in VRAM and (b) reads at PCIe speed
        # and ~20x below HBM is host-backed by physics, not by a noisy counter.
        if (not viol and c["placement"] == "not_in_vram_but_host_delta_unclear"
                and control_ok and lin is not None and copy_gbps and hbm_ref):
            if (lin <= PROVENANCE_HOST_MAX_OVER_COPY * copy_gbps
                    and lin <= PROVENANCE_HOST_MAX_FRAC_OF_HBM * hbm_ref):
                c["placement"] = "host_resident_by_bandwidth"
                c["placement_upgrade"] = (
                    f"not in VRAM (control-validated placement method) AND linear read "
                    f"{lin:.1f} GB/s <= {PROVENANCE_HOST_MAX_OVER_COPY}x copy engine "
                    f"({copy_gbps:.1f}) and <= {PROVENANCE_HOST_MAX_FRAC_OF_HBM}x HBM "
                    f"({hbm_ref:.1f}): the bytes are crossing PCIe, so they are in host RAM")

        c["eligible"] = (c["placement"] in ("host_resident", "host_resident_by_bandwidth")
                         and t is not None)

    chosen = next((c for c in cands if c["eligible"]), None)

    v = {
        "kill_criterion": (
            "KILL iff a KERNEL read of the grouped-GEMM tiled pattern from pages PROVEN "
            "to live in host RAM falls below BOTH thresholds: 0.5x the copy-engine H2D "
            f"figure measured in this same run, AND the plan's {ABS_KILL_GBPS} GB/s "
            "absolute floor. Clearing only one of the two is a MARGINAL pass and is "
            "flagged as such (pass_marginal) -- do not read it as a clean PASS."),
        "copy_engine_h2d_gbps": copy_gbps,
        "copy_engine_prior_gbps": COPY_ENGINE_PRIOR_GBPS,
        "relative_threshold_gbps": (round(rel_thr, 3) if rel_thr else None),
        "absolute_threshold_gbps": ABS_KILL_GBPS,
        "host_mechanism_candidates": cands,
        "placement_method_control_passed": control_ok,
        "hbm_reference_gbps": {
            "vmm_dev_linear": (best.get("vmm_dev.linear") or {}).get("gbps_median"),
            "dev_plain_linear": (best.get("dev_plain.linear") or {}).get("gbps_median"),
            "used_for_provenance_cross_check": hbm_ref,
        },
        "provenance_cross_check": {
            "max_over_copy_engine": PROVENANCE_MAX_OVER_COPY,
            "max_frac_of_hbm": PROVENANCE_MAX_FRAC_OF_HBM,
            "host_upgrade_max_over_copy_engine": PROVENANCE_HOST_MAX_OVER_COPY,
            "host_upgrade_max_frac_of_hbm": PROVENANCE_HOST_MAX_FRAC_OF_HBM,
            "note": ("bytes that crossed PCIe cannot materially outrun the copy engine "
                     "nor approach HBM; a candidate breaching either bound is "
                     "reclassified device_resident and cannot answer P1"),
        },
    }
    if copy_gbps is None:
        v["result"] = "INDETERMINATE"
        v["reason"] = ("the copy-engine reference did not measure, or its destination "
                       "failed its post-copy checksum; no relative threshold is derivable")
        return v
    if chosen is None:
        v["result"] = "INDETERMINATE"
        v["reason"] = ("no mechanism produced a tiled read from PROVEN host-resident "
                       "pages; every candidate was device-resident, indeterminate or "
                       "unmeasured. See placement_discovery and host_mechanism_candidates.")
        return v

    tiled = chosen["gbps"]["tiled"]
    pass_abs = tiled >= ABS_KILL_GBPS
    pass_rel = tiled >= rel_thr
    v["mechanism_used"] = chosen["arm"]
    v["mechanism_description"] = chosen["mechanism"]
    v["mechanism_substituted"] = chosen["arm"] != "vmm_host"
    v["mechanism_placement"] = chosen["placement"]
    v["host_kernel_read_gbps"] = chosen["gbps"]
    v["pass_absolute"] = bool(pass_abs)
    v["pass_relative"] = bool(pass_rel)
    v["pass_marginal"] = bool(pass_abs != pass_rel)
    # Per the probe contract: KILL only when BOTH thresholds are missed.
    v["result"] = "KILL" if (not pass_abs and not pass_rel) else "PASS"
    v["reason"] = (
        f"tiled kernel-read from proven host-resident pages via {chosen['arm']} "
        f"({chosen['mechanism']}) = {tiled:.2f} GB/s vs copy-engine {copy_gbps:.2f} GB/s "
        f"(0.5x = {rel_thr:.2f}) and absolute floor {ABS_KILL_GBPS} GB/s -> {v['result']}")
    if v["pass_marginal"]:
        v["marginal_warning"] = (
            f"MARGINAL: cleared the {'absolute' if pass_abs else 'relative'} threshold "
            f"but MISSED the {'relative' if pass_abs else 'absolute'} one "
            f"({tiled:.2f} GB/s vs relative {rel_thr:.2f} / absolute {ABS_KILL_GBPS}). "
            f"The plan treats the two as the same bar; a run that separates them is "
            f"sitting on the kill line and must not be reported as a clean PASS.")
        v["reason"] += "  " + v["marginal_warning"]
    if v["mechanism_substituted"]:
        v["reason"] += ("  NOTE: the plan's nominal mechanism (hipMemCreate location=Host "
                        "behind a device VA) did NOT provide host-resident pages; this "
                        "number comes from a substitute mechanism that cannot be "
                        "interleaved with device pages in one VA range, so T2's mixed "
                        "device/host single stack is not available via this route.")
    if chosen["placement"] == "host_resident_by_bandwidth":
        v["reason"] += ("  NOTE: host residency for this arm was established by the "
                        "bandwidth cross-check (not in VRAM + reads at PCIe speed), "
                        "because the MemAvailable delta was inconclusive on this "
                        "not-idle box. See host_mechanism_candidates[].placement_upgrade.")
    # INVALID is reserved for the two conditions in the probe contract: the
    # placement-method control failing, or a checksum/timing failure.  Those are
    # exactly what lands in d["errors"]; incidental problems (a coherence-arm
    # exception, an unsupported copy kind) go to d["warnings"] and must NOT void
    # an otherwise sound bandwidth answer.
    if d.get("errors"):
        v["result"] = "INVALID"
        v["reason"] = ("blocking errors recorded on this device: "
                       + "; ".join(d["errors"])
                       + " | pre-invalidation verdict was: " + str(v.get("reason")))
    if d.get("warnings"):
        v["warnings"] = list(d["warnings"])
    return v


# ---------------------------------------------------------------------------
# Result assembly / validation / reporting
# ---------------------------------------------------------------------------

TOP_KEYS = ("probe", "schema_version", "status", "started_utc", "finished_utc",
            "duration_s", "argv", "config", "static_env", "box_state_before",
            "box_state_after", "build", "devices", "verdict", "errors", "warnings")


def validate_result(r: dict) -> None:
    missing = [k for k in TOP_KEYS if k not in r]
    if missing:
        _fail(f"result JSON missing keys: {missing}")
    if r["probe"] != PROBE_ID:
        _fail("result.probe must be 'P1'")
    if not isinstance(r["schema_version"], int):
        _fail("schema_version must be an int")
    if r["status"] not in ("ok", "selftest", "error"):
        _fail(f"bad status {r['status']!r}")
    if not isinstance(r["devices"], list):
        _fail("devices must be a list")
    for d in r["devices"]:
        for k in ("hip_device_index", "name", "gcn_arch", "pci_bus_id",
                  "measurements", "copy_engine", "coherence", "verdict",
                  "placement_discovery", "findings", "vmm"):
            if k not in d:
                _fail(f"device entry missing key {k!r}")
        if not isinstance(d["measurements"], list):
            _fail("device.measurements must be a list")
        for m in d["measurements"]:
            for k in ("arm", "pattern", "blocks", "threads", "gbps_median",
                      "checksum_ok", "cover_bytes", "reps", "ms", "measured",
                      "timing_sane"):
                if k not in m:
                    _fail(f"measurement missing key {k!r}")
            if m.get("measured") and m.get("gbps_median") is None:
                _fail(f"measurement {m['arm']}.{m['pattern']} claims measured=True but "
                      f"carries no gbps_median")
    for k in ("result", "reason"):
        if k not in r["verdict"]:
            _fail(f"verdict missing key {k!r}")
    if r["verdict"]["result"] not in ("PASS", "KILL", "INDETERMINATE", "INVALID",
                                      "NOT_MEASURED"):
        _fail(f"bad verdict.result {r['verdict']['result']!r}")


def overall_verdict(devices: list[dict]) -> dict:
    if not devices:
        return {"result": "NOT_MEASURED", "reason": "no device was measured"}
    per = {str(d["hip_device_index"]): (d.get("verdict") or {}).get("result", "NOT_MEASURED")
           for d in devices}
    primary = devices[0]
    v = dict(primary.get("verdict") or {})
    v.setdefault("result", "NOT_MEASURED")
    v.setdefault("reason", "primary device produced no verdict")
    v["per_device"] = per
    v["primary_device"] = primary.get("hip_device_index")
    v["primary_card"] = f"{primary.get('name')} ({primary.get('pci_bus_id')})"
    v["per_device_note"] = ("the two cards are NOT identical (RX 9070 XT vs RX 9070); "
                            "every timing in this record is tagged with its HIP index "
                            "and PCI BDF")
    if any(x == "INVALID" for x in per.values()):
        v["result"] = "INVALID"
        v["reason"] = f"at least one device produced blocking errors: {per}"
    elif any(x == "KILL" for x in per.values()):
        v["result"] = "KILL"
        v["reason"] = f"{v.get('reason','')} | per-device: {per}"
    elif all(x in ("INDETERMINATE", "NOT_MEASURED") for x in per.values()):
        v["result"] = "INDETERMINATE"
        v["reason"] = (f"no device produced a tiled read from proven host-resident "
                       f"pages: {per} | {v.get('reason','')}")
    elif any(x in ("INDETERMINATE", "NOT_MEASURED") for x in per.values()):
        # Some card answered and some did not: report the answer, but never let the
        # unanswered card disappear from the headline.
        v["reason"] = (f"{v.get('reason','')} | PARTIAL: not every card produced an "
                       f"answer: {per}")
        v["partial"] = True
    if any((d.get("verdict") or {}).get("pass_marginal") for d in devices):
        v["pass_marginal_any_device"] = True
    return v


def summary_blob(r: dict) -> dict:
    """Compact machine-readable summary printed to stdout (full record in p1.json)."""
    def _delta(key: str):
        a = (r.get("box_state_before") or {}).get("vmstat", {}).get(key)
        b = (r.get("box_state_after") or {}).get("vmstat", {}).get(key)
        return (b - a) if (a is not None and b is not None) else None

    devs = []
    for d in r["devices"]:
        devs.append({
            "hip_device_index": d.get("hip_device_index"),
            "card": d.get("name"),
            "pci_bus_id": d.get("pci_bus_id"),
            "gcn_arch": d.get("gcn_arch"),
            "cu_units_basis": d.get("cu_units_basis"),
            "vram_free_before_gib": (round(d["vram_free_before"] / 2 ** 30, 2)
                                     if d.get("vram_free_before") is not None else None),
            "best_gbps": {k: v["gbps_median"] for k, v in d.get("best", {}).items()},
            "best_config": {k: f"{v['blocks']}x{v['threads']}"
                            for k, v in d.get("best", {}).items()},
            "copy_engine_gbps": {k: (v.get("gbps_median") if v.get("supported")
                                     else f"unsupported:{v.get('error')}")
                                 for k, v in (d.get("copy_engine") or {}).items()},
            "coherence": {c["arm"]: c.get("verdict", c.get("skipped", c.get("error")))
                          for c in d.get("coherence", [])},
            "placement_discovery": {r["loc_type_name"]: {
                "classification": placement_of(r),
                "vram_consumed_frac": ((r.get("placement_after_touch") or
                                        r.get("placement") or {})
                                       .get("vram_consumed_frac")),
                "memavailable_consumed_frac": ((r.get("placement_after_touch") or
                                                r.get("placement") or {})
                                               .get("memavailable_consumed_frac")),
                "probe_read_gbps": r.get("probe_read_gbps_median"),
            } for r in d.get("placement_discovery", [])},
            "placement_full_regions": {k: placement_of(v)
                                       for k, v in (d.get("vmm") or {}).items()},
            "findings": d.get("findings", []),
            "checksum_failures": sum(1 for m in d.get("measurements", [])
                                     if not m.get("checksum_ok")),
            "timing_fence_failures": sum(1 for m in d.get("measurements", [])
                                         if m.get("measured") and not m.get("timing_sane")),
            "configs_skipped_too_slow": sum(1 for m in d.get("measurements", [])
                                            if not m.get("measured", True)),
            "configs_measured": len(d.get("measurements", [])),
            "verdict": (d.get("verdict") or {}).get("result", "NOT_MEASURED"),
            "verdict_reason": (d.get("verdict") or {}).get("reason"),
            "warnings": d.get("warnings", []),
        })
    return {
        "probe": r["probe"],
        "schema_version": r["schema_version"],
        "status": r["status"],
        "duration_s": r["duration_s"],
        "config": {k: r["config"][k] for k in
                   ("bytes", "row_bytes", "seg_bytes", "stride_bytes", "reps",
                    "threads", "block_mults", "patterns", "devices")},
        "box": {
            "mem_available_gib_before": round(
                (r["box_state_before"]["meminfo_kb"]["MemAvailable"] or 0) / 2 ** 20, 1),
            "mem_available_gib_after": round(
                ((r.get("box_state_after") or {}).get("meminfo_kb", {}).get(
                    "MemAvailable") or 0) / 2 ** 20, 1),
            "pswpout_delta": _delta("pswpout"),
            "pgmajfault_delta": _delta("pgmajfault"),
            "loadavg_before": r["box_state_before"]["loadavg"],
        },
        "devices": devs,
        "verdict": r["verdict"],
        "errors": r["errors"],
        "warnings": r.get("warnings", []),
        "artifacts": r.get("artifacts"),
        "note": "compact summary; the full per-config record is in the .json artifact",
    }


def render_md(r: dict) -> str:
    L: list[str] = []
    A = L.append
    A("# P1 — kernel-read bandwidth from host-located pages behind a device VA")
    A("")
    A(f"*Probe spec:* `docs/WEIGHT_OFFLOAD_PLAN.md` §3 row **P1** (+ §11 items 1 and 6).  ")
    A(f"*Run:* `{r['started_utc']}` → `{r['finished_utc']}` ({r['duration_s']} s).  ")
    A(f"*Raw:* `p1.json` (schema {r['schema_version']}).  ")
    g = r["static_env"]["git"]
    A(f"*Worktree:* `{r['static_env']['worktree']}` @ `{g.get('sha')}` "
      f"({g.get('branch')}, dirty={g.get('dirty')}).")
    A("")
    v = r["verdict"]
    A(f"## Verdict: **{v['result']}**")
    A("")
    A(f"> {v.get('reason','')}")
    A("")
    A(f"Kill criterion: {v.get('kill_criterion','(n/a)')}")
    A("")
    if v.get("mechanism_used"):
        A(f"Mechanism the verdict is based on: `{v['mechanism_used']}` — "
          f"{v.get('mechanism_description')} (placement: `{v.get('mechanism_placement')}`)"
          f"{'  **(SUBSTITUTED — not the plan nominal mechanism)**' if v.get('mechanism_substituted') else ''}")
        A("")
    if v.get("marginal_warning"):
        A(f"> ⚠ **{v['marginal_warning']}**")
        A("")
    if r.get("warnings"):
        A("Warnings (recorded, non-blocking): " + "; ".join(str(w) for w in r["warnings"]))
        A("")
    if r["status"] == "selftest":
        A("**This is a `--selftest` run: no GPU was touched and no bandwidth was "
          "measured.** The JSON shape and argument handling were validated only.")
        A("")
        return "\n".join(L) + "\n"

    bs = r["box_state_before"]["meminfo_kb"]
    A("## Box state at start (the box is NOT idle — record or the number is worthless)")
    A("")
    A("| MemTotal | MemAvailable | MemFree | SwapFree | pswpout | loadavg |")
    A("|---|---|---|---|---|---|")
    A(f"| {bs['MemTotal']/2**20:.1f} GiB | {bs['MemAvailable']/2**20:.1f} GiB | "
      f"{bs['MemFree']/2**20:.1f} GiB | {bs['SwapFree']/2**20:.1f} GiB | "
      f"{r['box_state_before']['vmstat']['pswpout']} | "
      f"{r['box_state_before']['loadavg']} |")
    A("")
    aft = r["box_state_after"]["vmstat"]["pswpout"]
    bef = r["box_state_before"]["vmstat"]["pswpout"]
    if bef is not None and aft is not None:
        A(f"`pswpout` delta over the run: **{aft - bef}** pages "
          f"({'no swapping' if aft == bef else 'SWAPPING OCCURRED — treat numbers with suspicion'}).")
        A("")

    cfg = r["config"]
    A(f"Working set **{cfg['bytes']/2**20:.0f} MiB** "
      f"(≥ 256 MB required so nothing is MALL/L2-resident); "
      f"tiled pattern = {cfg['seg_bytes']} B bursts, {cfg['stride_bytes']} B stride, "
      f"{cfg['row_bytes']} B expert row; {cfg['reps']} timed reps + 1 discarded warm-up.")
    A("")

    for d in r["devices"]:
        A(f"## HIP device {d.get('hip_device_index')} — {d.get('name')} "
          f"(`{d.get('gcn_arch')}`, PCI `{d.get('pci_bus_id')}`)")
        A("")
        A(f"VRAM {(d.get('vram_free_before') or 0)/2**30:.2f} / "
          f"{(d.get('vram_total') or 0)/2**30:.2f} GiB free "
          f"at start. HIP reports {d.get('multiprocessor_count_reported')} "
          f"multiprocessors (WGPs on RDNA, not CUs). "
          f"Verdict: **{(d.get('verdict') or {}).get('result')}**.")
        A("")
        lk = d.get("pcie_link") or {}
        ul = lk.get("under_load") or {}
        bn = lk.get("bottleneck") or {}
        if ul.get("under_load_speed_gts"):
            A(f"**PCIe path:** governing link is root port `{lk.get('root_port')}`, "
              f"measured **UNDER LOAD** at "
              f"**x{ul['under_load_width']} @ {ul['under_load_speed_gts']} GT/s** "
              f"(~{ul['under_load_theoretical_gbps']} GB/s theoretical)"
              + ("" if ul.get("reached_rated_speed") else
                 f", which is **below its rated x{ul.get('rated_width')} @ "
                 f"{ul.get('rated_speed_gts')} GT/s**")
              + ". These ports down-train at idle "
                f"(states seen: "
              + ", ".join(f"x{s['width']}@{s['speed_gts']}GT/s"
                          for s in ul.get("all_states_observed", []))
              + "), so this is the max observed while the copy engine was saturating "
                "the link, not an at-rest reading. The card's own function advertises "
                "its on-package bridge as x16; that is not the slot link and bounds "
                "nothing.")
            A("")
        elif bn:
            A(f"**PCIe path:** root port `{lk.get('root_port')}` read at idle as "
              f"x{bn.get('current_link_width')} @ {bn.get('current_link_speed_gts')} GT/s "
              f"(~{bn.get('theoretical_gbps')} GB/s). No under-load sample was taken, so "
              f"treat this as a floor: these ports down-train at idle.")
            A("")
        nskip = sum(1 for m in d.get("measurements", []) if not m.get("measured", True))
        ninsane = sum(1 for m in d.get("measurements", [])
                      if m.get("measured") and not m.get("timing_sane"))
        if nskip or ninsane:
            A(f"Sweep hygiene: **{ninsane}** config(s) failed the hipEvent-vs-wall timing "
              f"fence and **{nskip}** exceeded the per-config time budget; both classes are "
              f"excluded from the `best` figures below.")
            A("")
        if d.get("warnings"):
            A("### Warnings (recorded, non-blocking)")
            A("")
            for w in d["warnings"]:
                A(f"- {w}")
            A("")
        if d.get("findings"):
            A("### Findings")
            A("")
            for f in d["findings"]:
                A(f"- **{f['id']}** ({f['severity']}): {f['summary']}  ")
                A(f"  *Consequence:* {f['consequence']}")
            A("")
        if d.get("placement_discovery"):
            A("### Placement provenance — does `hipMemCreate(location=Host)` "
              "actually give host RAM?")
            A("")
            A("| region | location type | classification | sysfs vram_used Δ | "
              "sysfs gtt_used Δ | hipMemGetInfo VRAM Δ | MemAvailable Δ | CPU can read VA |")
            A("|---|---|---|---|---|---|---|---|")

            def _pl_row(label: str, loc_name, pl: dict, cpu) -> None:
                A(f"| {label} | `{loc_name}` | **{pl.get('classification')}** | "
                  f"{pl.get('sysfs_vram_used_delta_frac')}× | "
                  f"{pl.get('sysfs_gtt_used_delta_frac')}× | "
                  f"{pl.get('vram_consumed_frac')}× | "
                  f"{pl.get('memavailable_consumed_frac')}× | {cpu} |")

            for pd in d["placement_discovery"]:
                _pl_row(f"probe {pd['probe_bytes']/2**20:.0f} MiB", pd.get("loc_type_name"),
                        placement_record_of(pd),
                        (pd.get("cpu_access") or {}).get("readable"))
            for key, info in (d.get("vmm") or {}).items():
                _pl_row(f"full `vmm_{key}`", info.get("loc_type_name"),
                        placement_record_of(info), "—")
            for key, info in (d.get("pinned") or {}).items():
                _pl_row(f"`{key}`", "hipHostMalloc", placement_record_of(info), "—")
            A("")
            A("Placement is decided from the **per-card amdgpu counters** "
              "(`mem_info_vram_used` / `mem_info_gtt_used` for this exact PCI BDF), which "
              "unrelated host-RAM churn cannot move; `hipMemGetInfo` and `MemAvailable` "
              "are shown as a cross-check. A region that consumes VRAM is device memory "
              "no matter what the API returned; a bandwidth number taken from it is an HBM "
              "number in disguise. `vmm_device` is the control: it MUST read "
              "`device_resident` or the method itself is unreliable and every placement "
              "claim here is void. Independently, any candidate whose measured read beats "
              f"{PROVENANCE_MAX_OVER_COPY}× the copy engine or reaches "
              f"{PROVENANCE_MAX_FRAC_OF_HBM}× the HBM reference is reclassified "
              "device-resident on physics alone.")
            A("")
        A("### Best kernel-read bandwidth (GB/s, median over reps, best occupancy point)")
        A("")
        A("| arm | linear | tiled (grouped-GEMM) | random 128 B |")
        A("|---|---|---|---|")
        for arm in ("vmm_host", "vmm_host_alt", "vmm_dev", "dev_plain",
                    "pinned_default", "pinned_noncoh"):
            row = [arm]
            any_present = False
            for p in PATTERNS:
                e = (d.get("best") or {}).get(f"{arm}.{p}")
                if e:
                    any_present = True
                    row.append(f"**{e['gbps_median']:.1f}** "
                               f"<br><sub>{e['blocks']}×{e['threads']} "
                               f"({e['waves_in_flight']} waves), "
                               f"{e['gbps_min']:.1f}–{e['gbps_max']:.1f} over "
                               f"{e['reps']} reps</sub>")
                else:
                    row.append("—")
            if any_present:
                A("| " + " | ".join(row) + " |")
        A("")
        A("### Copy-engine reference (`hipMemcpyAsync`, same byte count)")
        A("")
        A("| path | GB/s (median) | destination checksummed after the copy |")
        A("|---|---|---|")
        for k, e in (d.get("copy_engine") or {}).items():
            A(f"| {k} | " + (f"{e['gbps_median']:.2f}" if e.get("supported")
                             else f"unsupported ({e.get('error')})")
              + f" | {e.get('destination_verified')} |")
        A("")
        A("The destination is poisoned before each copy and checksummed after, so a copy "
          "that moved nothing cannot masquerade as an infinite rate — this figure is the "
          "denominator of the relative kill threshold.")
        A("")
        if d.get("coherence"):
            A("### Coherence (§11 item 6): CPU write → kernel read")
            A("")
            A("| buffer | CPU can read VA | CPU can write VA | dev-write→CPU-read | "
              "CPU-write→kernel-read (no flush) | after clflush | sub-64 B granularity |")
            A("|---|---|---|---|---|---|---|")
            # A skipped or errored arm still carries a `tests` key (an empty dict),
            # and a VA that is not CPU-accessible legitimately skips every CPU-side
            # test -- which is a RESULT, not a rendering failure.  Look up each cell
            # defensively so an unmeasurable arm prints as "—" plus its reason
            # instead of raising KeyError and voiding an otherwise valid run.
            def cell(c: dict, key: str) -> str:
                t = (c.get("tests") or {}).get(key)
                return "—" if not isinstance(t, dict) or "pass" not in t else str(t["pass"])

            for c in d["coherence"]:
                acc = c.get("cpu_access") or {}
                A("| {} | {} | {} | {} | {} | {} | {} |".format(
                    c["arm"],
                    acc.get("readable", "—"), acc.get("writable", "—"),
                    cell(c, "dev_write_then_cpu_read"),
                    cell(c, "cpu_write_then_kernel_read_no_explicit_flush"),
                    cell(c, "cpu_write_then_kernel_read_after_clflush"),
                    cell(c, "cpu_partial_line_write_then_kernel_read")))
            A("")
            notes = [(c["arm"], c.get("skipped") or c.get("error"))
                     for c in d["coherence"] if c.get("skipped") or c.get("error")]
            if notes:
                for arm, why in notes:
                    A(f"- `{arm}`: {why}")
                A("")
        if d.get("errors"):
            A("### Errors")
            A("")
            for e in d["errors"]:
                A(f"- {e}")
            A("")
    A("## How to reproduce")
    A("")
    A("```")
    A(r["config"]["reproduce_cmd"])
    A("```")
    A("")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_int_list(s: str, what: str) -> list[int]:
    try:
        vals = [int(x) for x in s.replace(" ", "").split(",") if x != ""]
    except ValueError:
        raise argparse.ArgumentTypeError(f"{what}: not a comma-separated int list: {s!r}")
    if not vals:
        raise argparse.ArgumentTypeError(f"{what}: empty list")
    return vals


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="p1_host_read_bw.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--devices", default="0",
                   help="comma-separated HIP device indices (default 0 = RX 9070 XT). "
                        "'0,1' measures both physical cards.")
    p.add_argument("--bytes", type=int, default=512 << 20,
                   help="working-set bytes per arm (default 512 MiB; MUST be >= 256 MB)")
    p.add_argument("--chunk-bytes", type=int, default=256 << 20,
                   help="hipMemCreate backing-chunk size (default 256 MiB)")
    p.add_argument("--placement-probe-bytes", type=int, default=256 << 20,
                   help="size of the transient region used to classify each "
                        "hipMemCreate location type as host- or device-resident "
                        "(default 256 MiB)")
    p.add_argument("--row-bytes", type=int, default=2949120,
                   help="tiled pattern: expert-row bytes (default 2949120 = 2.8125 MiB, "
                        "the co-demanded expert granule)")
    p.add_argument("--seg-bytes", type=int, default=512,
                   help="tiled pattern: contiguous burst bytes (default 512)")
    p.add_argument("--stride-bytes", type=int, default=32768,
                   help="tiled pattern: byte stride between consecutive bursts (default 32768)")
    p.add_argument("--threads", default="64,256,512",
                   help="workgroup sizes to sweep (default 64,256,512)")
    p.add_argument("--block-mults", default="1,2,4,8,16,32",
                   help="workgroup counts as multiples of the reported multiprocessor "
                        "count (default 1,2,4,8,16,32)")
    p.add_argument("--reps", type=int, default=7,
                   help="timed reps per config, after one discarded warm-up (default 7)")
    p.add_argument("--max-config-seconds", type=float, default=25.0,
                   help="per-config wall budget (default 25 s). A low-occupancy random-128B "
                        "read of a 512 MiB buffer over PCIe is latency-bound and can run "
                        "for minutes per rep; without a budget the 6-arm x 54-config sweep "
                        "looks hung for hours. Reps are reduced (min 3) to fit, and a "
                        "config whose single warm-up rep already blows the budget is "
                        "recorded from that rep alone and EXCLUDED from `best` -- which "
                        "only ever removes SLOW points, so it cannot inflate the answer. "
                        "0 disables the budget entirely.")
    p.add_argument("--patterns", default=",".join(PATTERNS),
                   help=f"subset of {PATTERNS}")
    p.add_argument("--arms", default="",
                   help="optional comma-separated arm subset "
                        "(vmm_host,vmm_dev,dev_plain,pinned_default,pinned_noncoh)")
    p.add_argument("--quick", action="store_true",
                   help="small sweep (threads=256, mults=4,16, reps=3) for a smoke run")
    p.add_argument("--outdir", default=DEFAULT_OUTDIR)
    p.add_argument("--tag", default="", help="filename suffix, e.g. --tag rerun -> p1.rerun.json")
    p.add_argument("--arch", default="gfx1201")
    p.add_argument("--hipcc", default="/opt/rocm/bin/hipcc")
    p.add_argument("--rebuild", action="store_true", help="force a hipcc rebuild")
    p.add_argument("--no-compile", action="store_true",
                   help="selftest only: skip the hipcc build")
    p.add_argument("--selftest", "--dry-run", dest="selftest", action="store_true",
                   help="validate args, build, symbols and JSON shape WITHOUT touching the GPU")
    p.add_argument("--print-full-json", action="store_true",
                   help="dump the entire per-config record to stdout instead of the "
                        "compact summary (the full record is always written to the .json)")
    return p


def finalize_args(args, parser) -> None:
    if args.quick:
        args.threads, args.block_mults, args.reps = "256", "4,16", 3
    args.threads = parse_int_list(args.threads, "--threads")
    args.block_mults = parse_int_list(args.block_mults, "--block-mults")
    args.devices = parse_int_list(args.devices, "--devices")
    args.patterns = [x for x in args.patterns.replace(" ", "").split(",") if x]
    args.arms = [x for x in args.arms.replace(" ", "").split(",") if x]

    bad = [p for p in args.patterns if p not in PATTERNS]
    if bad:
        parser.error(f"--patterns: unknown {bad}; choose from {list(PATTERNS)}")
    known_arms = ("vmm_host", "vmm_host_alt", "vmm_dev", "dev_plain",
                  "pinned_default", "pinned_noncoh")
    bad = [a for a in args.arms if a not in known_arms]
    if bad:
        parser.error(f"--arms: unknown {bad}; choose from {list(known_arms)}")
    if args.bytes < (256 * 1000 * 1000):
        parser.error(f"--bytes {args.bytes} < 256 MB: the plan REQUIRES a working set "
                     f">= 256 MB (4x the 64 MB MALL) or the result is a cache artifact "
                     f"like the withdrawn 127.2 GB/s figure")
    if args.bytes % (2 << 20):
        parser.error("--bytes must be a multiple of 2 MiB")
    if args.seg_bytes % 16 or args.stride_bytes % args.seg_bytes or \
            args.row_bytes % args.stride_bytes:
        parser.error("tiled pattern requires seg%16==0, stride%seg==0, row%stride==0 "
                     f"(got seg={args.seg_bytes} stride={args.stride_bytes} "
                     f"row={args.row_bytes})")
    if args.row_bytes > args.bytes:
        parser.error("--row-bytes exceeds --bytes")
    if args.placement_probe_bytes < (16 << 20) or args.placement_probe_bytes % 4096:
        parser.error("--placement-probe-bytes must be >= 16 MiB and 4 KiB-aligned")
    if args.row_bytes > args.placement_probe_bytes:
        parser.error("--row-bytes exceeds --placement-probe-bytes")
    for t in args.threads:
        if t % 32 or t < 32 or t > 1024:
            parser.error(f"--threads {t}: must be a multiple of 32 in [32, 1024]")
    lanes = args.seg_bytes // 16
    if not any(t % lanes == 0 for t in args.threads):
        parser.error(f"no --threads value is a multiple of seg_bytes/16 = {lanes}; "
                     f"the tiled pattern would have no runnable config")
    if not any(t % 8 == 0 for t in args.threads):
        parser.error("no --threads value is a multiple of 8 (random128 needs 8 lanes/line)")
    for m in args.block_mults:
        if m < 1:
            parser.error("--block-mults entries must be >= 1")
    if args.reps < 3:
        parser.error("--reps must be >= 3 to report a median and a spread")
    if args.max_config_seconds < 0:
        parser.error("--max-config-seconds must be >= 0 (0 disables the budget)")
    if any(d < 0 or d > 7 for d in args.devices):
        parser.error("--devices entries must be in [0, 7]")
    if len(set(args.devices)) != len(args.devices):
        parser.error("--devices contains duplicates")


def config_dict(args, argv: list[str]) -> dict:
    return {
        "bytes": args.bytes,
        "chunk_bytes": args.chunk_bytes,
        "placement_probe_bytes": args.placement_probe_bytes,
        "row_bytes": args.row_bytes,
        "seg_bytes": args.seg_bytes,
        "stride_bytes": args.stride_bytes,
        "threads": args.threads,
        "block_mults": args.block_mults,
        "reps": args.reps,
        "max_config_seconds": args.max_config_seconds,
        "patterns": args.patterns,
        "arms": args.arms or "all",
        "devices": args.devices,
        "arch": args.arch,
        # NOTE: per-device; the authoritative per-card value is
        # devices[i].cu_units_basis.  This field is the LAST device's value.
        "cu_units_basis": getattr(args, "cu_units", None),
        "reproduce_cmd": ("ROCR_VISIBLE_DEVICES=0,1 python3 "
                          + os.path.relpath(os.path.abspath(__file__), REPO) + " "
                          + " ".join(argv)),
        "working_set_note": ("must be >= 256 MB so nothing is MALL(64 MB)/L2-resident; "
                             "the withdrawn 127.2 GB/s figure was a 2 MiB L2 artifact"),
        "cache_residency_argument": (
            "Every pattern is a cyclic scan that covers the whole buffer in the SAME "
            "deterministic order each rep. With a working set 8x the 64 MB MALL and LRU "
            "replacement, the lines a rep starts on are precisely the ones evicted "
            "longest ago, so inter-rep cache reuse is ~0 rather than the naive "
            "cache/working-set = 12.5%. No inter-rep flush is therefore needed, and the "
            "reported GB/s is a memory-path figure, not a cache figure."),
        "random128_note": (
            "'random128' is a bijective odd-multiplier permutation of 128 B lines "
            "(required so the prefix checksum still holds). Consecutive work items land "
            "a constant 0x9E3779B1 (mod nlines) lines apart -- a large-stride scatter "
            "across the buffer, NOT an i.i.d. random address stream. Read it as "
            "'prefetcher-hostile scatter'."),
        "timing_note": (
            "hipEvent-timed on the null stream with an event fence around the launch; "
            f"cross-checked per config against wall-clock GB/s (event/wall must be <= "
            f"{TIMING_SANITY_FACTOR}). Configs failing that fence are excluded from "
            "`best` and raise a blocking error."),
    }


def main(argv: list[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    finalize_args(args, parser)
    args.cu_units = 1  # replaced with the reported multiprocessor count per device

    started = _utc()
    t0 = time.perf_counter()
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    suffix = f".{args.tag}" if args.tag else ""
    json_path = os.path.join(outdir, f"p1{'.selftest' if args.selftest else ''}{suffix}.json")
    md_path = os.path.join(outdir, f"p1{'.selftest' if args.selftest else ''}{suffix}.md")

    result: dict = {
        "probe": PROBE_ID,
        "schema_version": SCHEMA_VERSION,
        "status": "selftest" if args.selftest else "ok",
        "started_utc": started,
        "finished_utc": None,
        "duration_s": None,
        "argv": argv,
        "config": config_dict(args, argv),
        "static_env": collect_static_env(),
        "box_state_before": None,
        "box_state_after": None,
        "build": None,
        "devices": [],
        "verdict": {"result": "NOT_MEASURED", "reason": "run did not complete"},
        "errors": [],     # BLOCKING only: these void the measurement
        "warnings": [],   # recorded, non-blocking
    }

    try:
        result["box_state_before"] = collect_box_state("before")
        if args.selftest:
            if args.no_compile:
                result["build"] = {"skipped": True,
                                   "reason": "--no-compile", "src_sha256": _sha256(HIP_SRC)}
            else:
                result["build"] = build_kernels(args.arch, args.rebuild, args.hipcc)
                load_kernels(check_only=True)   # dlopen + symbol check, no HIP calls
                result["build"]["symbols_present"] = list(REQUIRED_SYMBOLS)
            result["box_state_after"] = collect_box_state("after")
            result["verdict"] = {
                "result": "NOT_MEASURED",
                "reason": "--selftest: argument handling, build, symbol table and JSON "
                          "shape validated; no GPU work was performed and no bandwidth "
                          "number was produced",
                "kill_criterion": ("kernel-read from host-located pages must be >= 0.5x "
                                   f"copy-engine and >= {ABS_KILL_GBPS} GB/s"),
            }
        else:
            result["build"] = build_kernels(args.arch, args.rebuild, args.hipcc)
            lib = load_kernels()
            hip = load_hip()

            ndev = ctypes.c_int(0)
            K(lib, lib.p1_device_count(ctypes.byref(ndev)), "hipGetDeviceCount")
            result["hip_device_count"] = ndev.value
            if ndev.value < 1:
                _fail("hipGetDeviceCount returned 0 -- no HIP device visible")
            if ndev.value > 2:
                _fail(f"hipGetDeviceCount = {ndev.value}: more than the two gfx1201 compute "
                      f"cards are enumerated. The Ryzen iGPU (47 GB GTT) must never be "
                      f"visible. Check ROCR_VISIBLE_DEVICES.")
            for d in args.devices:
                if d >= ndev.value:
                    _fail(f"--devices requests {d} but only {ndev.value} device(s) visible")
            rv = ctypes.c_int(0)
            K(lib, lib.p1_runtime_version(ctypes.byref(rv)), "hipRuntimeGetVersion")
            result["hip_runtime_version"] = rv.value

            partial_path = json_path + ".partial"

            def checkpoint(dev_rec=None, _p=partial_path) -> None:
                """Durable incremental evidence: a hard GPU fault kills the
                process outright, so flush what we have after every arm."""
                if dev_rec is not None and all(x is not dev_rec for x in result["devices"]):
                    result["devices"].append(dev_rec)
                try:
                    with open(_p + ".tmp", "w") as fh:
                        json.dump({**result, "status": "in_progress"}, fh,
                                  indent=2, default=str)
                    os.replace(_p + ".tmp", _p)
                except OSError:
                    pass

            for d in args.devices:
                print(f"[P1] device {d} ...", flush=True)
                K(lib, lib.p1_set_device(d), f"set_device({d})")
                mp = ctypes.c_int(0)
                K(lib, lib.p1_device_info(d, None, 0, None, 0, None, 0,
                                          ctypes.byref(mp), None, None, None),
                  "p1_device_info(mp)")
                args.cu_units = max(1, mp.value)
                result["config"]["cu_units_basis"] = args.cu_units
                run_device(lib, hip, d, args, checkpoint)
                checkpoint()

            result["box_state_after"] = collect_box_state("after")
            result["verdict"] = overall_verdict(result["devices"])
            for d in result["devices"]:
                idx = d.get("hip_device_index")
                result["errors"].extend(f"dev{idx}: {e}" for e in d.get("errors", []))
                result["warnings"].extend(f"dev{idx}: {w}" for w in d.get("warnings", []))
            # A run that swapped is not a valid bandwidth measurement.
            b = (result.get("box_state_before") or {}).get("vmstat", {}).get("pswpout")
            a = (result.get("box_state_after") or {}).get("vmstat", {}).get("pswpout")
            if b is not None and a is not None and a > b:
                result["warnings"].append(
                    f"the box swapped during the run (pswpout +{a - b} pages): host-side "
                    f"timings may be contaminated -- re-run on a quieter box before "
                    f"treating a marginal figure as decisive")

    except ProbeError as e:
        result["status"] = "error"
        result["errors"].append(str(e))
        result["verdict"] = {"result": "INVALID", "reason": f"probe aborted: {e}"}
        if result["box_state_after"] is None:
            try:
                result["box_state_after"] = collect_box_state("after-error")
            except Exception:  # noqa: BLE001
                result["box_state_after"] = None
    except Exception as e:  # noqa: BLE001
        result["status"] = "error"
        result["errors"].append(f"{type(e).__name__}: {e}")
        result["verdict"] = {"result": "INVALID", "reason": f"unhandled: {type(e).__name__}: {e}"}
        if result["box_state_after"] is None:
            result["box_state_after"] = None

    result["finished_utc"] = _utc()
    result["duration_s"] = round(time.perf_counter() - t0, 2)

    shape_error = None
    try:
        validate_result(result)
    except ProbeError as e:
        shape_error = str(e)
        result["errors"].append(f"SCHEMA: {e}")

    result["artifacts"] = {"json": json_path, "md": md_path}
    with open(json_path, "w") as fh:
        json.dump(result, fh, indent=2, sort_keys=False, default=str)
        fh.write("\n")
    # the final record supersedes any incremental checkpoint
    for stale in (json_path + ".partial", json_path + ".partial.tmp"):
        try:
            os.remove(stale)
        except OSError:
            pass
    try:
        with open(md_path, "w") as fh:
            fh.write(render_md(result))
    except Exception as e:  # noqa: BLE001
        result["errors"].append(f"markdown render failed: {type(e).__name__}: {e}")

    if args.print_full_json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(json.dumps(summary_blob(result), indent=2, default=str))
    print(f"\n[P1] json -> {json_path}\n[P1] md   -> {md_path}", file=sys.stderr)
    for w in result.get("warnings", []):
        print(f"[P1] WARNING: {w}", file=sys.stderr)
    print(f"[P1] verdict: {result['verdict']['result']} -- {result['verdict'].get('reason','')}",
          file=sys.stderr)

    if shape_error:
        print(f"[P1] FATAL: result JSON failed its own schema check: {shape_error}",
              file=sys.stderr)
        return 3
    if result["status"] == "error" or result["errors"]:
        print(f"[P1] FATAL: {len(result['errors'])} error(s): {result['errors']}",
              file=sys.stderr)
        return 2
    if args.selftest:
        print("[P1] SELFTEST OK (no GPU touched)", file=sys.stderr)
        return 0
    if result["verdict"]["result"] in ("INVALID", "INDETERMINATE"):
        return 2
    # A KILL is a valid, successful measurement -- exit 0 and let the plan act on it.
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except ProbeError as e:
        print(f"[P1] FATAL: {e}", file=sys.stderr)
        sys.exit(2)
