#!/usr/bin/env python3
"""P4 — PCIe topology confirmation + CONCURRENT two-card H2D under host memory pressure.

Plan reference: docs/WEIGHT_OFFLOAD_PLAN.md §3 (probe P4), §1 (corrected physics), §11 item 3.

WHAT THIS ANSWERS
-----------------
The topology is SETTLED and is *not* re-litigated here: both gfx1201 cards sit on a bifurcated
x16 direct to the CPU root complex (root ports 0000:00:01.1 / 0000:00:01.3, each x8 Gen5), and
the x16 reported at 03:00.0 / 07:00.0 is each card's OWN on-package bridge. This probe reads
that back from sysfs as a one-line confirmation and then spends all of its time on the open
question:

    With BOTH cards streaming pinned host->device at once, what is the per-card and the
    aggregate bandwidth -- with and without a synthetic CPU memory-bandwidth load -- and do the
    two cards get equal shares or does one starve?

Two ranks want ~57 GB/s from a ~45 GB/s-class dual-channel DDR5 bus, so the expectation is that
host DDR binds before PCIe does. Every ceiling in the plan's §1 table is a single-card IDLE
number; TP=2 is the served configuration. This probe fixes that denominator.

METHOD
------
* Torch-free: ctypes against libamdhip64.so. No torch lazy-init artifacts, no CachingHostAllocator.
* One OS process per card (models TP=2 ranks; no GIL between the two streaming loops).
* Fixed-WALL-CLOCK windows released by a shared multiprocessing.Barrier, so both cards are
  provably streaming at the same time. Per-rep overlap fraction is computed and a rep whose
  overlap is below --min-overlap is FAILED, not reported.
* One warm-up window per arm is discarded; >= --reps steady-state windows are kept; median and
  min/max spread are reported.
* Synthetic CPU load = N processes doing libc memmove between two buffers far larger than the
  96 MB L3, so the traffic is real DDR traffic. Their achieved copy bandwidth is measured over
  exactly the GPU window (counter read AFTER the barrier releases), and DDR traffic is reported
  as a BRACKET (2x for a non-temporal-store copy, 3x if the store path incurred RFO).
* PCIe link state (width/speed) is sampled BEFORE, DURING and AFTER every streaming window, at
  the endpoint AND at that endpoint's OWN root port. Every fraction-of-theoretical is computed
  against the link as TRAINED DURING TRANSFER, per card.

WHY THE PER-CARD LINK MATTERS ON THIS BOX
-----------------------------------------
Both root ports advertise max x8 @32 GT/s (= 31.5 GB/s), but they do NOT necessarily train
there: 0000:00:01.1 was observed at 32 GT/s while 0000:00:01.3 sat at 16 GT/s (Gen4, half the
ceiling), and 00:01.1 ASPM-downtrains to 2.5 GT/s at idle. A denominator taken from
`max_link_speed` therefore overstates the ceiling, and -- far worse -- a per-card bandwidth gap
caused by link training would be misread as concurrency starvation and would fire the plan's
fairness kill rule (b) for the wrong reason. So this probe reports BOTH the raw imbalance and a
`link_normalized_imbalance` (each card's GB/s divided by its OWN trained link), and refuses to
compute a theoretical fraction it cannot ground in a resolved root port.

SAFETY / HONESTY RULES OBSERVED
-------------------------------
* Forces ROCR_VISIBLE_DEVICES=0,1 and unsets HIP_VISIBLE_DEVICES itself, so the Ryzen iGPU
  (ROCm device 2, which advertises ~47 GB of GTT) can never enter enumeration.
* Refuses to run unless exactly two devices enumerate and each reports a plausible discrete VRAM
  total; an iGPU-shaped device aborts the run.
* Records which PHYSICAL card every number came from (HIP ordinal -> PCI BDF -> DRM card node ->
  lspci line), and resolves the topology from those MEASURED BDFs, never from a hardcoded pair.
* CORRECTNESS, not just hipSuccess. The pinned source carries a position-dependent per-page
  stamp; after every arm's timed windows (and never inside one) the destination is read back and
  memcmp'd against the source, and the page header is checked to match the source offset. On
  this box a HIP call returning hipSuccess is not evidence that the right bytes moved. An arm
  whose read-back fails reports NO bandwidth.
* Cache-residency guard: the working set is required to be >= 256 MB (4x the 64 MB MALL) and
  every chunk size must leave >= 2 distinct source slots, so nothing can be served from cache.
* Host-memory precondition + a per-window swap tripwire: a window in which the box swapped is
  FAILED, because pinned-H2D and host-memcpy numbers from a swapping box are meaningless.
* Records full box state (meminfo, vmstat swap counters, loadavg, rocm-smi VRAM) before and
  after, and per-window vmstat deltas.
* Records each issuing thread's own CPU time per window, so a drop under CPU load can be
  attributed to DDR bandwidth rather than to scheduling contention on the issuing thread.
* Any arm that does not fully succeed is written as status=FAILED with NO bandwidth number, and
  the process exits non-zero. It never prints a number it did not measure.
* --selftest / --dry-run validate argument handling, the sysfs/box-state parsers and the full
  JSON+MD shape WITHOUT touching the GPU, writing into a _selftest/ subdirectory whose filenames
  say SYNTHETIC-DO-NOT-CITE (never p4.json).

RUN (no lease needed for this development run; see the brief):
    ROCR_VISIBLE_DEVICES=0,1 python3 tools/offload/p4_pcie_concurrency.py
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

# ---------------------------------------------------------------------------------------------
# Device visibility. Forced at IMPORT time so that it is already in place for multiprocessing
# 'spawn' children (which re-import this module) and before any HIP library load.
# ---------------------------------------------------------------------------------------------
_PRIOR_ROCR = os.environ.get("ROCR_VISIBLE_DEVICES")
_PRIOR_HIP_VD = os.environ.get("HIP_VISIBLE_DEVICES")
os.environ["ROCR_VISIBLE_DEVICES"] = "0,1"
os.environ.pop("HIP_VISIBLE_DEVICES", None)

import multiprocessing as mp  # noqa: E402  (after env forcing, deliberately)

MiB = 1 << 20
GiB = 1 << 30
GRANULE_BYTES = 2867200  # ~2.8 MiB, the co-demanded expert granule from the plan's §1 table
SCHEMA_VERSION = 1
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "docs", "measurements", "WEIGHT_OFFLOAD_2026-09-02")

# PCIe per-lane payload rate, GB/s. 8b/10b below Gen3, 128b/130b at and above.
_LANE_GBPS = {2.5: 0.250, 5.0: 0.500, 8.0: 0.985, 16.0: 1.969, 32.0: 3.938, 64.0: 7.877}
_BDF_RE = re.compile(r"^[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-9a-fA-F]$")

# The settled topology, asserted as a one-line confirmation (NOT re-measured beyond sysfs).
# These are a CROSS-CHECK ONLY: the denominator for every ratio is resolved from the BDFs the HIP
# workers actually reported, by walking each card's own sysfs chain to its root port.
EXPECTED_ROOT_PORTS = ("0000:00:01.1", "0000:00:01.3")
DEFAULT_CARD_BDFS = ("0000:03:00.0", "0000:07:00.0")  # selftest / pre-worker fallback only

# The device MALL is 64 MB and the host L3 is ~96 MB. Any working set below ~4x the larger of
# those can be served from cache and is not a PCIe measurement. Enforced in validate_args.
MIN_WORKING_SET_BYTES = 256 * MiB


class ProbeError(RuntimeError):
    """A precondition failed. Always fatal, always non-zero exit."""


# =============================================================================================
# HIP ctypes binding -- no graceful degradation: a missing symbol raises.
# =============================================================================================
HIP_SUCCESS = 0
HIP_MEMCPY_HOST_TO_DEVICE = 1
HIP_MEMCPY_DEVICE_TO_HOST = 2
HIP_HOST_MALLOC_DEFAULT = 0

_HIP_SYMS = [
    ("hipInit", [ctypes.c_uint], ctypes.c_int),
    ("hipGetDeviceCount", [ctypes.POINTER(ctypes.c_int)], ctypes.c_int),
    ("hipSetDevice", [ctypes.c_int], ctypes.c_int),
    ("hipDeviceGetPCIBusId", [ctypes.c_char_p, ctypes.c_int, ctypes.c_int], ctypes.c_int),
    ("hipMemGetInfo", [ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)], ctypes.c_int),
    ("hipHostMalloc", [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint], ctypes.c_int),
    ("hipHostFree", [ctypes.c_void_p], ctypes.c_int),
    ("hipMalloc", [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t], ctypes.c_int),
    ("hipFree", [ctypes.c_void_p], ctypes.c_int),
    ("hipStreamCreate", [ctypes.POINTER(ctypes.c_void_p)], ctypes.c_int),
    ("hipStreamDestroy", [ctypes.c_void_p], ctypes.c_int),
    ("hipMemcpyAsync", [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                        ctypes.c_void_p], ctypes.c_int),
    ("hipMemcpy", [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int], ctypes.c_int),
    ("hipMemsetD8", [ctypes.c_void_p, ctypes.c_ubyte, ctypes.c_size_t], ctypes.c_int),
    ("hipStreamSynchronize", [ctypes.c_void_p], ctypes.c_int),
    ("hipDeviceSynchronize", [], ctypes.c_int),
    ("hipGetErrorString", [ctypes.c_int], ctypes.c_char_p),
]

# Cross-process timestamps MUST come from a clock with a shared origin. CLOCK_MONOTONIC is
# system-wide on Linux; time.perf_counter() only happens to be the same thing on CPython/Linux,
# and the overlap fraction (which decides whether a "concurrent" number is concurrent at all) is
# compared across processes. Take the guarantee explicitly rather than by implementation detail.
def mono() -> float:
    return time.clock_gettime(time.CLOCK_MONOTONIC)


class Hip:
    def __init__(self, soname: str = "libamdhip64.so"):
        try:
            self.lib = ctypes.CDLL(soname)
        except OSError as e:
            raise ProbeError(f"cannot load {soname}: {e}") from e
        for name, argtypes, restype in _HIP_SYMS:
            try:
                fn = getattr(self.lib, name)
            except AttributeError as e:
                raise ProbeError(f"{soname} is missing required symbol {name}") from e
            fn.argtypes = argtypes
            fn.restype = restype

    def err(self, rc: int) -> str:
        try:
            s = self.lib.hipGetErrorString(rc)
            return s.decode() if s else f"rc={rc}"
        except Exception:
            return f"rc={rc}"

    def chk(self, rc: int, what: str) -> None:
        if rc != HIP_SUCCESS:
            raise ProbeError(f"{what} failed: rc={rc} ({self.err(rc)})")


# =============================================================================================
# Host-side readers: topology, box state. All read-only, no GPU work, safe in --selftest.
# =============================================================================================
def _read_attr(path: str):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def _lane_rate(speed_str) -> float | None:
    """'32.0 GT/s PCIe' -> 3.938 GB/s per lane."""
    if not speed_str:
        return None
    m = re.match(r"\s*([0-9.]+)", speed_str)
    if not m:
        return None
    return _LANE_GBPS.get(float(m.group(1)))


def _link_node(sysfs_dir: str, bdf: str) -> dict:
    cur_w = _read_attr(os.path.join(sysfs_dir, "current_link_width"))
    cur_s = _read_attr(os.path.join(sysfs_dir, "current_link_speed"))
    max_w = _read_attr(os.path.join(sysfs_dir, "max_link_width"))
    max_s = _read_attr(os.path.join(sysfs_dir, "max_link_speed"))
    out = {
        "bdf": bdf,
        "current_link_width": int(cur_w) if cur_w and cur_w.isdigit() else None,
        "current_link_speed": cur_s,
        "max_link_width": int(max_w) if max_w and max_w.isdigit() else None,
        "max_link_speed": max_s,
    }
    lr_cur, lr_max = _lane_rate(cur_s), _lane_rate(max_s)
    out["current_theoretical_gbps"] = (
        round(lr_cur * out["current_link_width"], 2)
        if lr_cur and out["current_link_width"] else None)
    out["max_theoretical_gbps"] = (
        round(lr_max * out["max_link_width"], 2)
        if lr_max and out["max_link_width"] else None)
    return out


def pci_chain(bdf: str) -> list:
    """The endpoint and every PCIe ancestor up to the host bridge, nearest-first."""
    dev = os.path.join("/sys/bus/pci/devices", bdf)
    if not os.path.exists(dev):
        raise ProbeError(f"no sysfs entry for PCI device {bdf}")
    p = os.path.realpath(dev)
    chain = []
    while p and p != "/" and os.path.basename(p) != "sys":
        name = os.path.basename(p)
        if _BDF_RE.match(name):
            chain.append(_link_node(p, name))
        if name.startswith("pci0000:"):
            break
        p = os.path.dirname(p)
    return chain


def drm_card_for_bdf(bdf: str):
    try:
        for entry in sorted(os.listdir("/sys/class/drm")):
            if not re.match(r"^card\d+$", entry):
                continue
            link = os.path.join("/sys/class/drm", entry, "device")
            if os.path.exists(link) and os.path.basename(os.path.realpath(link)) == bdf:
                return entry
    except OSError:
        pass
    return None


def lspci_line(bdf: str):
    if not shutil.which("lspci"):
        return None
    try:
        out = subprocess.run(["lspci", "-s", bdf], capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


def root_port_for_bdf(bdf: str) -> str | None:
    """The LAST PCIe ancestor before the host bridge -- i.e. this card's own CPU root port.

    This is the only link in the chain that is a real slot link. The x16 reported at the
    endpoint (03:00.0 / 07:00.0) and at the intermediate bridges is the card's OWN on-package
    bridge and must never be used as a bandwidth denominator.
    """
    chain = pci_chain(bdf)
    return chain[-1]["bdf"] if chain else None


def sample_link_state(bdfs: list, tag: str) -> dict:
    """Cheap enough (a few sysfs reads) to call mid-transfer.

    Samples BOTH the endpoint and that endpoint's own root port, because the root port is where
    the slot link (and its trained speed) actually lives.
    """
    links = {}
    for b in bdfs:
        links[b] = _link_node(os.path.realpath(os.path.join("/sys/bus/pci/devices", b)), b)
        rp = _ROOT_PORT_CACHE.get(b)
        if rp is None:
            try:
                rp = root_port_for_bdf(b)
            except ProbeError:
                rp = None
            _ROOT_PORT_CACHE[b] = rp or ""
        if rp:
            links[b]["root_port"] = _link_node(
                os.path.realpath(os.path.join("/sys/bus/pci/devices", rp)), rp)
    return {"tag": tag, "t": time.time(), "mono": mono(), "links": links}


_ROOT_PORT_CACHE: dict = {}


def link_denominator(samples: list, bdf: str, measured_gbps: float) -> dict:
    """Resolve the bandwidth denominator for one card from ITS OWN root port.

    The link ASPM-downtrains to 2.5 GT/s at idle on this box (observed live), so the *max* over
    the pre/mid/post samples is the speed the link trained to while streaming. But a sysfs sample
    is a point read: it can miss the uptrained state entirely. Missing the uptrain is a SAMPLING
    artifact, not a measurement artifact, so it must degrade the denominator's provenance -- not
    throw away a perfectly good bandwidth number. The only hard failure is a measured rate above
    the link's *max* capability, which is physically impossible and means the byte accounting or
    the device mapping is wrong.

    Returns the denominator, where it came from, and any warning, so every ratio in the artifact
    carries its own provenance.
    """
    cur, mx = [], []
    for s in samples:
        n = (s.get("links", {}).get(bdf, {}) or {}).get("root_port")
        if not n:
            continue
        if n.get("current_theoretical_gbps"):
            cur.append(n["current_theoretical_gbps"])
        if n.get("max_theoretical_gbps"):
            mx.append(n["max_theoretical_gbps"])
    during = max(cur) if cur else None
    during_min = min(cur) if cur else None
    max_link = max(mx) if mx else None
    out = {"sampled_during_window_gbps": during,
           "sampled_min_in_window_gbps": during_min,
           "root_port_max_link_gbps": max_link,
           "denominator_gbps": None, "denominator_source": "unresolved",
           "warning": None, "impossible": False}
    if during and measured_gbps <= during * 1.05:
        out["denominator_gbps"] = during
        out["denominator_source"] = "sampled_during_window"
    elif max_link:
        out["denominator_gbps"] = max_link
        out["denominator_source"] = ("max_link_fallback_sample_missed_uptrain" if during
                                     else "max_link_fallback_no_sample")
        out["warning"] = (
            f"root-port link samples for {bdf} peaked at {during} GB/s but the card measured "
            f"{measured_gbps:.2f} GB/s, so the sysfs samples missed the uptrained state; the "
            f"denominator falls back to the port's max capability ({max_link} GB/s). Any "
            "fraction-of-link for this card is a LOWER bound.")
    if max_link and measured_gbps > 1.05 * max_link:
        out["impossible"] = True
    return out


def meminfo() -> dict:
    out = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                k, _, v = line.partition(":")
                parts = v.split()
                if parts and parts[0].isdigit():
                    out[k] = int(parts[0])  # kB
    except OSError as e:
        raise ProbeError(f"cannot read /proc/meminfo: {e}") from e
    return out


def vmstat_swap() -> dict:
    out = {}
    try:
        with open("/proc/vmstat") as f:
            for line in f:
                k, _, v = line.partition(" ")
                if k in ("pswpin", "pswpout", "pgmajfault"):
                    out[k] = int(v)
    except OSError as e:
        raise ProbeError(f"cannot read /proc/vmstat: {e}") from e
    return out


def rocm_smi_vram():
    """Verbatim rocm-smi capture. Informational only; never parsed into a measurement."""
    exe = shutil.which("rocm-smi") or "/opt/rocm/bin/rocm-smi"
    if not os.path.exists(exe):
        return {"available": False}
    try:
        r = subprocess.run([exe, "--showmeminfo", "vram", "--showproductname"],
                           capture_output=True, text=True, timeout=60)
        return {"available": True, "returncode": r.returncode,
                "stdout": r.stdout, "stderr": r.stderr}
    except Exception as e:  # noqa: BLE001 -- informational capture must never abort the probe
        return {"available": False, "error": repr(e)}


def box_state(tag: str, with_rocm_smi: bool = True) -> dict:
    mi = meminfo()
    st = {
        "tag": tag,
        "utc": datetime.now(timezone.utc).isoformat(),
        "mem_total_gib": round(mi.get("MemTotal", 0) / (1 << 20), 2),
        "mem_free_gib": round(mi.get("MemFree", 0) / (1 << 20), 2),
        "mem_available_gib": round(mi.get("MemAvailable", 0) / (1 << 20), 2),
        "buffers_cached_gib": round((mi.get("Buffers", 0) + mi.get("Cached", 0)) / (1 << 20), 2),
        "swap_total_gib": round(mi.get("SwapTotal", 0) / (1 << 20), 2),
        "swap_free_gib": round(mi.get("SwapFree", 0) / (1 << 20), 2),
        "shmem_gib": round(mi.get("Shmem", 0) / (1 << 20), 2),
        "mlocked_gib": round(mi.get("Mlocked", 0) / (1 << 20), 2),
        "vmstat": vmstat_swap(),
        "loadavg": _read_attr("/proc/loadavg"),
        "nproc": os.cpu_count(),
    }
    if with_rocm_smi:
        st["rocm_smi"] = rocm_smi_vram()
    return st


def host_facts() -> dict:
    cpu = None
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    rocm_ver = _read_attr("/opt/rocm/.info/version")
    return {
        "hostname": os.uname().nodename,
        "kernel": os.uname().release,
        "cpu_model": cpu,
        "nproc": os.cpu_count(),
        "rocm_version": rocm_ver,
        "python": sys.version.split()[0],
        "cwd": os.getcwd(),
        "script": os.path.abspath(__file__),
        "repo_root": REPO_ROOT,
        "git_sha": _git_sha(),
        "rocr_visible_devices_forced": os.environ.get("ROCR_VISIBLE_DEVICES"),
        "rocr_visible_devices_prior": _PRIOR_ROCR,
        "hip_visible_devices_prior": _PRIOR_HIP_VD,
    }


def _git_sha():
    try:
        r = subprocess.run(["git", "-C", REPO_ROOT, "rev-parse", "HEAD"],
                           capture_output=True, text=True, timeout=15)
        return r.stdout.strip() or None
    except Exception:
        return None


def topology_report(bdfs: list | None = None) -> dict:
    """Read-back confirmation of the SETTLED topology, anchored on the cards actually MEASURED.

    `bdfs` is the list of PCI BDFs the HIP workers reported. When it is given, the root port for
    each card is resolved by walking that card's OWN sysfs chain -- never from a hardcoded pair --
    so the bandwidth denominator can never describe a device that produced none of the numbers.
    EXPECTED_ROOT_PORTS is kept only as a cross-check that raises a mismatch note.
    """
    bdfs = list(bdfs) if bdfs else list(DEFAULT_CARD_BDFS)
    cards, roots, per_card, notes = {}, {}, {}, []
    for bdf in bdfs:
        if not os.path.exists(os.path.join("/sys/bus/pci/devices", bdf)):
            notes.append(f"measured card {bdf} has no sysfs entry; no link denominator for it")
            continue
        chain = pci_chain(bdf)
        rp = chain[-1]["bdf"] if chain else None
        cards[bdf] = {
            "chain": chain,
            "root_port": rp,
            "drm_card": drm_card_for_bdf(bdf),
            "numa_node": _read_attr(os.path.join("/sys/bus/pci/devices", bdf, "numa_node")),
            "lspci": lspci_line(bdf),
        }
        if rp:
            node = _link_node(os.path.realpath(os.path.join("/sys/bus/pci/devices", rp)), rp)
            roots[rp] = node
            per_card[bdf] = {
                "root_port": rp,
                "max_theoretical_gbps": node["max_theoretical_gbps"],
                "idle_current_theoretical_gbps": node["current_theoretical_gbps"],
                "max_link_speed": node["max_link_speed"],
                "idle_current_link_speed": node["current_link_speed"],
            }
        else:
            notes.append(f"could not resolve a root port for measured card {bdf}")

    unexpected = sorted(set(roots) - set(EXPECTED_ROOT_PORTS))
    missing = sorted(set(EXPECTED_ROOT_PORTS) - set(roots))
    if unexpected or missing:
        notes.append(
            f"root ports resolved from the measured cards ({sorted(roots)}) differ from the "
            f"settled expectation {list(EXPECTED_ROOT_PORTS)}: unexpected={unexpected}, "
            f"missing={missing}. Every denominator below uses the RESOLVED ports.")

    # ---- link asymmetry: the two slots are NOT guaranteed to have trained to the same speed.
    # Observed on this box: 00:01.1 trains x8 Gen5 (32 GT/s) while 00:01.3 sits at x8 Gen4
    # (16 GT/s) -- half the ceiling -- with BOTH reporting max = 32 GT/s. If that is not surfaced,
    # a per-card bandwidth gap gets misread as concurrency starvation.
    idle_theo = {b: v["idle_current_theoretical_gbps"] for b, v in per_card.items()
                 if v["idle_current_theoretical_gbps"]}
    downtrained = {
        b: v for b, v in per_card.items()
        if v["max_theoretical_gbps"] and v["idle_current_theoretical_gbps"]
        and v["idle_current_theoretical_gbps"] < 0.95 * v["max_theoretical_gbps"]}
    asym = {
        "per_card": per_card,
        "idle_current_theoretical_gbps": idle_theo,
        "idle_downtrained_cards": sorted(downtrained),
        "idle_asymmetry_ratio": (round(max(idle_theo.values()) / min(idle_theo.values()), 3)
                                 if len(idle_theo) > 1 and min(idle_theo.values()) else None),
        "warning": None,
    }
    if len(idle_theo) > 1 and min(idle_theo.values()) and \
            max(idle_theo.values()) / min(idle_theo.values()) > 1.05:
        asym["warning"] = (
            "THE TWO SLOTS ARE NOT SYMMETRIC AT IDLE. A per-card bandwidth gap in the concurrent "
            "arms may be a LINK-TRAINING asymmetry, not concurrency starvation. Read "
            "link_normalized_imbalance (per-card GB/s divided by that card's OWN trained "
            "theoretical) before applying the plan's fairness rule (b).")

    maxes = [v["max_theoretical_gbps"] for v in per_card.values() if v["max_theoretical_gbps"]]
    confirmation = (
        "root ports (resolved from the measured cards) " + ", ".join(
            f"{v['root_port']} for {b}: max x{roots[v['root_port']]['max_link_width']} "
            f"@{v['max_link_speed']} = {v['max_theoretical_gbps']} GB/s; idle-current "
            f"@{v['idle_current_link_speed']} = {v['idle_current_theoretical_gbps']} GB/s"
            for b, v in per_card.items())
        + " | the x16 reported at the endpoint is the card's OWN on-package bridge, not the slot"
        + " | idle 'current' is ASPM-downtrained; the denominator used for every ratio is the "
          "link state sampled DURING transfer -- see link_state_samples")
    return {
        "confirmation": confirmation,
        "root_ports": roots,
        "cards": cards,
        "link_asymmetry": asym,
        "per_card_max_theoretical_gbps": maxes,
        "aggregate_max_theoretical_gbps": round(sum(maxes), 2) if maxes else None,
        "notes": notes,
        "note": "topology is SETTLED per the brief; this is a read-back confirmation only.",
    }


# =============================================================================================
# Synthetic CPU memory-bandwidth load
# =============================================================================================
def cpu_load_worker(buf_bytes: int, counter, stop_ev, live) -> None:
    """libc memmove between two buffers each far larger than L3 -> real DDR traffic.

    The counter is bumped in SUB-BUFFER slices so that the parent's window sampling is not
    quantised to whole 256 MiB copies (which at 6 loaders would be several GB of quantisation
    error on a short window, i.e. a several-GB/s error in the headline DDR figure).
    """
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=False)
        a = ctypes.create_string_buffer(buf_bytes)
        b = ctypes.create_string_buffer(buf_bytes)
        ctypes.memset(a, 0x5A, buf_bytes)   # populate; never leave zero pages
        ctypes.memset(b, 0xA5, buf_bytes)
        pa, pb = ctypes.addressof(a), ctypes.addressof(b)
        libc.memmove.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        libc.memmove.restype = ctypes.c_void_p
        # Slice at >= 32 MiB so glibc still takes its large-copy (non-temporal) path, and the
        # slice remains far above L2 -- but small enough to keep counter quantisation ~1 ms.
        slice_b = min(buf_bytes, 32 * MiB)
        nslice = max(1, buf_bytes // slice_b)
        with live.get_lock():
            live.value += 1
        flip = False
        while not stop_ev.is_set():
            src, dst = (pb, pa) if flip else (pa, pb)
            for i in range(nslice):
                if stop_ev.is_set():
                    break
                off = i * slice_b
                libc.memmove(dst + off, src + off, slice_b)
                with counter.get_lock():
                    counter.value += slice_b
            flip = not flip
    except Exception:
        traceback.print_exc()
    finally:
        try:
            with live.get_lock():
                live.value -= 1
        except Exception:
            pass


class CpuLoad:
    def __init__(self, ctx, n_procs: int, buf_bytes: int):
        self.ctx = ctx
        self.n = n_procs
        self.buf = buf_bytes
        self.counter = ctx.Value("q", 0)
        self.live = ctx.Value("i", 0)
        self.stop = ctx.Event()
        self.procs = []

    def start(self) -> None:
        if self.n <= 0:
            return
        for _ in range(self.n):
            p = self.ctx.Process(target=cpu_load_worker,
                                 args=(self.buf, self.counter, self.stop, self.live), daemon=True)
            p.start()
            self.procs.append(p)
        # Let every loader allocate, first-touch its two buffers and reach steady state. A loader
        # still faulting in 256 MiB is not yet generating DDR traffic.
        deadline = time.time() + 60.0
        while time.time() < deadline and self.live_count() < self.n:
            time.sleep(0.05)
        if self.live_count() < self.n:
            raise ProbeError(f"only {self.live_count()} of {self.n} CPU load processes reached "
                             "steady state within 60 s; the loaded arms would understate the DDR "
                             "load and are not reported")
        time.sleep(1.0)

    def live_count(self) -> int:
        with self.live.get_lock():
            return self.live.value

    def read(self) -> int:
        with self.counter.get_lock():
            return self.counter.value

    def alive_procs(self) -> int:
        return sum(1 for p in self.procs if p.is_alive())

    def stop_all(self) -> None:
        if not self.procs:
            return
        self.stop.set()
        for p in self.procs:
            p.join(timeout=20)
            if p.is_alive():
                p.terminate()
        self.procs = []


# =============================================================================================
# GPU H2D worker: one process per card
# =============================================================================================
PATTERN_MAGIC = 0x50345F484432_0000  # 'P4_HD2' -- stamped per page, XORed with the page's offset


def stamp_pattern(addr: int, nbytes: int) -> None:
    """Give the pinned source a POSITION-DEPENDENT pattern.

    A uniform memset cannot distinguish "the DMA moved the right bytes" from "every copy landed
    at offset 0", "the copy was short", or "the source was never populated" -- all of which
    return hipSuccess and yield a perfectly plausible bandwidth. Each 4096 B page gets a header
    of (MAGIC ^ page_byte_offset, page_byte_offset), on top of a replicated tile so the body is
    not all-zero either.
    """
    libc = ctypes.CDLL("libc.so.6", use_errno=False)
    libc.memcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    libc.memcpy.restype = ctypes.c_void_p
    tile_b = min(nbytes, 1 * MiB)
    ntile = tile_b // 8
    tile = (ctypes.c_uint64 * ntile)()
    for i in range(ntile):
        tile[i] = (PATTERN_MAGIC ^ (i * 8)) & 0xFFFFFFFFFFFFFFFF
    tptr = ctypes.addressof(tile)
    off = 0
    while off < nbytes:
        n = min(tile_b, nbytes - off)
        libc.memcpy(addr + off, tptr, n)
        off += n
    pages = nbytes // 4096
    if pages:
        words = (ctypes.c_uint64 * (pages * 512)).from_address(addr)  # one view, 512 u64 / page
        for p in range(pages):
            po = p * 4096
            w = p * 512
            words[w] = (PATTERN_MAGIC ^ po) & 0xFFFFFFFFFFFFFFFF
            words[w + 1] = po


def cpu_seconds_self() -> float:
    t = os.times()
    return t.user + t.system


def gpu_worker(ordinal: int, cfg: dict, cmd_q, res_q, barrier) -> None:
    hip = None
    src = dst = stream_handles = None
    try:
        hip = Hip(cfg["hip_soname"])
        hip.chk(hip.lib.hipInit(0), "hipInit")
        n = ctypes.c_int()
        hip.chk(hip.lib.hipGetDeviceCount(ctypes.byref(n)), "hipGetDeviceCount")
        if n.value != cfg["expect_devices"]:
            raise ProbeError(
                f"expected exactly {cfg['expect_devices']} HIP devices under "
                f"ROCR_VISIBLE_DEVICES={os.environ.get('ROCR_VISIBLE_DEVICES')}, got {n.value}. "
                "A third device is the Ryzen iGPU and must never enter enumeration.")
        hip.chk(hip.lib.hipSetDevice(ordinal), "hipSetDevice")

        buf = ctypes.create_string_buffer(64)
        hip.chk(hip.lib.hipDeviceGetPCIBusId(buf, 64, ordinal), "hipDeviceGetPCIBusId")
        bdf = buf.value.decode().lower()

        free_b, total_b = ctypes.c_size_t(), ctypes.c_size_t()
        hip.chk(hip.lib.hipMemGetInfo(ctypes.byref(free_b), ctypes.byref(total_b)), "hipMemGetInfo")
        total_gib = total_b.value / GiB
        if not (cfg["vram_min_gib"] <= total_gib <= cfg["vram_max_gib"]):
            raise ProbeError(
                f"device {ordinal} ({bdf}) reports {total_gib:.1f} GiB total VRAM, outside the "
                f"discrete-card window [{cfg['vram_min_gib']}, {cfg['vram_max_gib']}] GiB. "
                "This is very likely the Ryzen iGPU (advertises ~47 GB of GTT). Refusing to run.")
        need = cfg["dst_bytes"] + (64 << 20)
        if free_b.value < need:
            raise ProbeError(
                f"device {ordinal} ({bdf}) has {free_b.value/GiB:.2f} GiB free VRAM, needs "
                f"{need/GiB:.2f} GiB. Another job is holding the card, or lower --dst-mb.")

        # Pinned host source. Steady state only -- the first pin is timed but never reported as
        # a bandwidth number.
        t_pin0 = time.perf_counter()
        src = ctypes.c_void_p()
        hip.chk(hip.lib.hipHostMalloc(ctypes.byref(src), cfg["src_bytes"], HIP_HOST_MALLOC_DEFAULT),
                f"hipHostMalloc({cfg['src_bytes']})")
        pin_s = time.perf_counter() - t_pin0
        # Position-dependent, NOT a uniform memset: see stamp_pattern().
        stamp_pattern(src.value, cfg["src_bytes"])

        dst = ctypes.c_void_p()
        hip.chk(hip.lib.hipMalloc(ctypes.byref(dst), cfg["dst_bytes"]),
                f"hipMalloc({cfg['dst_bytes']})")
        # Poison the destination so a copy that never happens cannot accidentally verify.
        hip.chk(hip.lib.hipMemsetD8(dst, 0xEE, cfg["dst_bytes"]),
                "hipMemsetD8(dst, 0xEE)")

        # Ordinary (pageable) host buffer for read-back verification. Never used in a timed path.
        vbuf = ctypes.create_string_buffer(cfg["max_chunk_bytes"])
        vptr = ctypes.addressof(vbuf)
        libc = ctypes.CDLL("libc.so.6", use_errno=False)
        libc.memcmp.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        libc.memcmp.restype = ctypes.c_int

        stream_handles = []
        for _ in range(cfg["streams"]):
            s = ctypes.c_void_p()
            hip.chk(hip.lib.hipStreamCreate(ctypes.byref(s)), "hipStreamCreate")
            stream_handles.append(s)

        memcpy_async = hip.lib.hipMemcpyAsync
        stream_sync = hip.lib.hipStreamSynchronize
        hip.chk(hip.lib.hipDeviceSynchronize(), "hipDeviceSynchronize")

        res_q.put({"kind": "ready", "ord": ordinal, "payload": {
            "hip_ordinal": ordinal,
            "pci_bdf": bdf,
            "drm_card": drm_card_for_bdf(bdf),
            "lspci": lspci_line(bdf),
            "vram_total_gib": round(total_gib, 3),
            "vram_free_gib_at_start": round(free_b.value / GiB, 3),
            "pinned_src_bytes": cfg["src_bytes"],
            "pin_seconds": round(pin_s, 3),
            "pin_gbps_first_call_NOT_steady_state": round(cfg["src_bytes"] / pin_s / 1e9, 3),
            "device_dst_bytes": cfg["dst_bytes"],
            "streams": cfg["streams"],
            "host_malloc_flags": HIP_HOST_MALLOC_DEFAULT,
            "pid": os.getpid(),
        }})

        while True:
            cmd = cmd_q.get()
            if cmd["action"] == "stop":
                break
            try:
                barrier.wait(timeout=cfg["barrier_timeout_s"])
            except Exception as e:  # broken barrier: a sibling died
                res_q.put({"kind": "error", "ord": ordinal,
                           "payload": f"barrier: {e!r}"})
                break

            if cmd["action"] == "idle":
                res_q.put({"kind": "result", "ord": ordinal,
                           "payload": {"active": False, "arm": cmd["arm"], "rep": cmd["rep"]}})
                continue

            # ---- read-back verification. Runs OUTSIDE every timed window (issued by the parent
            # after an arm's reps are collected) so its D2H traffic can never perturb a
            # measurement. On this box a HIP call returning hipSuccess is NOT evidence that the
            # bytes moved (the VMM remap serves stale pages and still returns success), so the
            # bandwidth number is only reported if the same buffers, same streams and same
            # hipMemcpyAsync path are shown to deliver the right bytes to the right offsets.
            if cmd["action"] == "verify":
                chunk = cmd["chunk_bytes"]
                n_src = max(1, cfg["src_bytes"] // chunk)
                n_dst = max(1, cfg["dst_bytes"] // chunk)
                checks, bad = [], None
                try:
                    for j in (0, max(0, min(n_src, n_dst) - 1), min(n_src, n_dst) // 2):
                        s_off, d_off = j * chunk, j * chunk
                        hip.chk(memcpy_async(
                            ctypes.c_void_p(dst.value + d_off),
                            ctypes.c_void_p(src.value + s_off),
                            chunk, HIP_MEMCPY_HOST_TO_DEVICE, stream_handles[0]),
                            "verify hipMemcpyAsync H2D")
                        hip.chk(stream_sync(stream_handles[0]), "verify hipStreamSynchronize")
                        hip.chk(hip.lib.hipMemcpy(
                            ctypes.c_void_p(vptr), ctypes.c_void_p(dst.value + d_off),
                            chunk, HIP_MEMCPY_DEVICE_TO_HOST), "verify hipMemcpy D2H")
                        same = libc.memcmp(ctypes.c_void_p(vptr),
                                           ctypes.c_void_p(src.value + s_off), chunk) == 0
                        hdr = (ctypes.c_uint64 * 2).from_address(vptr)
                        checks.append({
                            "src_off": s_off, "dst_off": d_off, "bytes": chunk,
                            "memcmp_equal": same,
                            "page_header_word0": int(hdr[0]),
                            "page_header_word1_expected_src_off": int(hdr[1]),
                            "header_offset_matches": int(hdr[1]) == s_off,
                        })
                        if not same or int(hdr[1]) != s_off:
                            bad = checks[-1]
                            break
                except ProbeError as e:
                    bad = {"error": str(e)}
                res_q.put({"kind": "result", "ord": ordinal, "payload": {
                    "active": False, "verify": True, "arm": cmd["arm"], "rep": cmd["rep"],
                    "chunk_bytes": chunk, "checks": checks,
                    "ok": bad is None, "failure": bad,
                }})
                continue

            chunk = cmd["chunk_bytes"]
            depth = cfg["queue_depth"]
            n_src = max(1, cfg["src_bytes"] // chunk)
            n_dst = max(1, cfg["dst_bytes"] // chunk)
            nstream = len(stream_handles)
            k = 0
            nbytes = 0
            cpu0 = cpu_seconds_self()
            t0 = mono()
            t_end = t0 + cmd["window_s"]
            rc_bad = 0
            while mono() < t_end:
                for _ in range(depth):
                    # Check the deadline INSIDE the batch: at chunk=256 MiB and depth=8 a whole
                    # batch is 2 GiB (~70 ms), and a full-batch overrun past t_end is time in
                    # which one card is still streaming while its sibling has stopped -- which
                    # would be counted as concurrent bandwidth.
                    if mono() >= t_end:
                        break
                    s_off = (k % n_src) * chunk
                    d_off = (k % n_dst) * chunk
                    rc = memcpy_async(
                        ctypes.c_void_p(dst.value + d_off),
                        ctypes.c_void_p(src.value + s_off),
                        chunk, HIP_MEMCPY_HOST_TO_DEVICE,
                        stream_handles[k % nstream])
                    if rc != HIP_SUCCESS:
                        rc_bad = rc
                        break
                    k += 1
                    nbytes += chunk
                if rc_bad:
                    break
                for s in stream_handles:
                    rc = stream_sync(s)
                    if rc != HIP_SUCCESS:
                        rc_bad = rc
                        break
                if rc_bad:
                    break
            t1 = mono()  # every issued copy has been stream-synchronised before this point
            cpu1 = cpu_seconds_self()
            if rc_bad:
                # Abort the barrier too, or the parent burns a full timeout on every later arm.
                try:
                    barrier.abort()
                except Exception:
                    pass
                res_q.put({"kind": "error", "ord": ordinal,
                           "payload": f"hipMemcpyAsync/StreamSynchronize: {hip.err(rc_bad)}"})
                break
            dt = t1 - t0
            if k == 0 or dt <= 0:
                try:
                    barrier.abort()
                except Exception:
                    pass
                res_q.put({"kind": "error", "ord": ordinal,
                           "payload": f"window completed {k} copies in {dt:.6f}s -- no measurement"})
                break
            res_q.put({"kind": "result", "ord": ordinal, "payload": {
                "active": True,
                "arm": cmd["arm"],
                "rep": cmd["rep"],
                "chunk_bytes": chunk,
                "bytes": nbytes,
                "seconds": dt,
                "gbps": nbytes / dt / 1e9,
                "n_copies": k,
                "issue_cpu_seconds": round(cpu1 - cpu0, 4),
                "issue_cpu_fraction": round((cpu1 - cpu0) / dt, 4),
                "t0": t0,
                "t1": t1,
            }})
    except Exception as e:  # noqa: BLE001 -- must reach the parent, never hang it
        try:
            barrier.abort()
        except Exception:
            pass
        try:
            res_q.put({"kind": "error", "ord": ordinal,
                       "payload": f"{type(e).__name__}: {e}\n{traceback.format_exc()}"})
        except Exception:
            pass
    finally:
        try:
            if hip is not None:
                if stream_handles:
                    for s in stream_handles:
                        hip.lib.hipStreamDestroy(s)
                if dst is not None and dst.value:
                    hip.lib.hipFree(dst)
                if src is not None and src.value:
                    hip.lib.hipHostFree(src)
        except Exception:
            pass


# =============================================================================================
# Orchestration
# =============================================================================================
def summarize(values: list) -> dict:
    vals = sorted(float(v) for v in values)
    if not vals:
        return {"n": 0}
    med = statistics.median(vals)
    return {
        "n": len(vals),
        "median": round(med, 3),
        "mean": round(statistics.fmean(vals), 3),
        "min": round(vals[0], 3),
        "max": round(vals[-1], 3),
        "stdev": round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0,
        "spread_pct_of_median": round((vals[-1] - vals[0]) / med * 100, 2) if med else None,
    }


class Runner:
    def __init__(self, args):
        self.args = args
        self.ctx = mp.get_context("spawn")
        self.cfg = {
            "hip_soname": args.hip_so,
            "expect_devices": args.expect_devices,
            "src_bytes": args.src_mb * MiB,
            "dst_bytes": args.dst_mb * MiB,
            "max_chunk_bytes": max(args.chunk_bytes),
            "streams": args.streams,
            "queue_depth": args.queue_depth,
            "vram_min_gib": args.vram_min_gib,
            "vram_max_gib": args.vram_max_gib,
            "barrier_timeout_s": args.window_s * 6 + 60.0,
        }
        self.ordinals = list(range(args.expect_devices))
        self.res_q = self.ctx.Queue()
        self.cmd_qs = {o: self.ctx.Queue() for o in self.ordinals}
        self.barrier = self.ctx.Barrier(len(self.ordinals) + 1)  # + the parent
        self.procs = {}
        self.devices = {}
        self.bdfs = []
        self.link_samples = []

    # -- lifecycle -----------------------------------------------------------------------
    def start_workers(self) -> None:
        for o in self.ordinals:
            p = self.ctx.Process(target=gpu_worker,
                                 args=(o, self.cfg, self.cmd_qs[o], self.res_q, self.barrier),
                                 daemon=True)
            p.start()
            self.procs[o] = p
        deadline = time.time() + self.args.startup_timeout_s
        seen = set()
        while len(seen) < len(self.ordinals):
            remaining = deadline - time.time()
            if remaining <= 0:
                raise ProbeError(
                    f"only {sorted(seen)} of {self.ordinals} GPU workers became ready within "
                    f"{self.args.startup_timeout_s}s")
            try:
                msg = self.res_q.get(timeout=min(remaining, 5.0))
            except Exception:
                continue
            if msg["kind"] == "error":
                raise ProbeError(f"worker {msg['ord']} failed during init:\n{msg['payload']}")
            if msg["kind"] != "ready":
                raise ProbeError(f"unexpected message before ready: {msg}")
            self.devices[msg["ord"]] = msg["payload"]
            seen.add(msg["ord"])
        self.bdfs = [self.devices[o]["pci_bdf"] for o in self.ordinals]
        if len(set(self.bdfs)) != len(self.bdfs):
            raise ProbeError(f"two HIP ordinals map to the same PCI BDF: {self.bdfs}")

    def stop_workers(self) -> None:
        for o in self.ordinals:
            try:
                self.cmd_qs[o].put({"action": "stop"})
            except Exception:
                pass
        for o, p in self.procs.items():
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()

    # -- one measurement window ----------------------------------------------------------
    def run_window(self, arm: str, rep: int, active: list, chunk_bytes, load: CpuLoad | None):
        for o in self.ordinals:
            if o in active:
                self.cmd_qs[o].put({"action": "copy", "arm": arm, "rep": rep,
                                    "chunk_bytes": chunk_bytes, "window_s": self.args.window_s})
            else:
                self.cmd_qs[o].put({"action": "idle", "arm": arm, "rep": rep})

        vm0 = vmstat_swap()
        pre_link = sample_link_state(self.bdfs, f"{arm}/rep{rep}/pre-window")
        try:
            self.barrier.wait(timeout=self.cfg["barrier_timeout_s"])
        except Exception as e:
            raise ProbeError(f"barrier broke entering arm={arm} rep={rep}: {e!r} "
                             "(a GPU worker most likely died -- see errors above)") from e
        p_t0 = mono()
        # Read the load counter only AFTER the barrier releases. Reading it before the barrier
        # would attribute pre-window loader bytes to a window whose denominator starts here --
        # a systematic OVERSTATEMENT of the very DDR figure decision rule (c) turns on.
        load_before = load.read() if load else None

        # Sample the link state mid-window: at idle these links ASPM-downtrain (observed on this
        # box: 00:01.1 drops to 2.5 GT/s), so the only meaningful 'current' reading is one taken
        # while traffic is flowing. With no active card (the cpu_load_alone arm) the workers
        # return at once, so the parent must itself hold the window open for the load counter to
        # be sampled over a full window.
        time.sleep(max(0.05, self.args.window_s * (0.5 if active else 1.0)))
        mid_link = sample_link_state(self.bdfs, f"{arm}/rep{rep}/mid-window")

        results = {}
        deadline = time.time() + self.cfg["barrier_timeout_s"]
        while len(results) < len(self.ordinals):
            remaining = deadline - time.time()
            if remaining <= 0:
                raise ProbeError(f"timed out collecting results for arm={arm} rep={rep}; "
                                 f"got {sorted(results)}")
            try:
                msg = self.res_q.get(timeout=min(remaining, 5.0))
            except Exception:
                continue
            if msg["kind"] == "error":
                raise ProbeError(f"worker {msg['ord']} error in arm={arm} rep={rep}:\n"
                                 f"{msg['payload']}")
            results[msg["ord"]] = msg["payload"]
        p_t1 = mono()
        load_after = load.read() if load else None
        post_link = sample_link_state(self.bdfs, f"{arm}/rep{rep}/post-window")
        vm1 = vmstat_swap()
        for s in (pre_link, mid_link, post_link):
            self.link_samples.append(s)

        rec = {"rep": rep, "cards": {}, "status": "OK"}
        rec["vmstat_delta"] = {k: vm1.get(k, 0) - vm0.get(k, 0) for k in set(vm0) | set(vm1)}
        act = [results[o] for o in active if results.get(o, {}).get("active")]
        if len(act) != len(active):
            raise ProbeError(f"arm={arm} rep={rep}: expected {len(active)} active card results, "
                             f"got {len(act)}")
        win_samples = [pre_link, mid_link, post_link]
        fails, warns = [], []
        for o in active:
            r = results[o]
            bdf = self.devices[o]["pci_bdf"]
            den = link_denominator(win_samples, bdf, r["gbps"])
            card = {
                "pci_bdf": bdf,
                "drm_card": self.devices[o]["drm_card"],
                "gbps": round(r["gbps"], 3),
                "bytes": r["bytes"],
                "seconds": round(r["seconds"], 4),
                "n_copies": r["n_copies"],
                "issue_cpu_fraction": r.get("issue_cpu_fraction"),
                # Denominator taken from THIS card's own root port, with explicit provenance.
                "link": den,
                "root_port_trained_gbps_during_window": den["denominator_gbps"],
                "link_denominator_source": den["denominator_source"],
            }
            if den["denominator_gbps"]:
                card["fraction_of_own_trained_link"] = \
                    round(r["gbps"] / den["denominator_gbps"], 4)
            if den["warning"]:
                warns.append(den["warning"])
            # A card cannot exceed the MAX capability of its own root port. If it appears to, the
            # byte accounting or the ordinal->BDF mapping is wrong and nothing may be published.
            if den["impossible"]:
                fails.append(
                    f"card {bdf}: measured {r['gbps']:.2f} GB/s above its root port's maximum "
                    f"capability {den['root_port_max_link_gbps']} GB/s -- physically impossible; "
                    "byte accounting or the HIP-ordinal-to-BDF mapping is wrong")
            rec["cards"][str(o)] = card
        if act:
            t0s = [r["t0"] for r in act]
            t1s = [r["t1"] for r in act]
            span = min(t1s) - max(t0s)
            mean_win = statistics.fmean(r["seconds"] for r in act)
            overlap = span / mean_win if mean_win > 0 else 0.0
            rec["overlap_fraction"] = round(overlap, 4)
            rec["aggregate_gbps"] = round(sum(r["gbps"] for r in act), 3)
            rec["gpu_window_s"] = round(mean_win, 4)
            if len(act) > 1:
                gs = [r["gbps"] for r in act]
                rec["min_share"] = round(min(gs) / sum(gs), 4)
                rec["max_share"] = round(max(gs) / sum(gs), 4)
                rec["imbalance_ratio_max_over_min"] = round(max(gs) / min(gs), 4) if min(gs) else None
                # Raw imbalance conflates concurrency starvation with a LINK-TRAINING asymmetry
                # (this box has one slot at Gen5 and one at Gen4). Normalising each card by its
                # OWN trained link separates the two; only the normalised figure answers the
                # plan's fairness rule (b).
                norm = [rec["cards"][str(o)].get("fraction_of_own_trained_link") for o in active]
                if all(n for n in norm):
                    rec["link_normalized_shares"] = [round(n, 4) for n in norm]
                    rec["link_normalized_imbalance"] = round(max(norm) / min(norm), 4)
                if overlap < self.args.min_overlap:
                    fails.append(
                        f"overlap {overlap:.3f} < --min-overlap {self.args.min_overlap}: the two "
                        "cards were not provably streaming at the same time, so the aggregate is "
                        "not a concurrency measurement")
        else:
            rec["overlap_fraction"] = None
            rec["aggregate_gbps"] = None
        if load is not None:
            dt = p_t1 - p_t0
            copied = load_after - load_before
            rec["cpu_load_gbps_memcpy"] = round(copied / dt / 1e9, 3)
            # glibc memmove uses non-temporal stores for copies this large, so DDR traffic is
            # ~2x the copied bytes (one read + one write). If it did NOT take the NT path the
            # destination also incurs a read-for-ownership and the traffic is ~3x. Publish the
            # BRACKET: "the DDR bus is saturated" is a decision rule, and it must not rest on a
            # silently-chosen constant.
            rec["cpu_load_ddr_gbps_est_2x_nt"] = round(2 * copied / dt / 1e9, 3)
            rec["cpu_load_ddr_gbps_est_3x_rfo"] = round(3 * copied / dt / 1e9, 3)
            rec["cpu_load_ddr_gbps_est"] = rec["cpu_load_ddr_gbps_est_2x_nt"]  # back-compat
            rec["cpu_load_window_s"] = round(dt, 3)
            rec["cpu_load_procs_alive"] = load.alive_procs()
            rec["cpu_load_procs_expected"] = load.n
            if load.alive_procs() < load.n or load.live_count() < load.n:
                fails.append(
                    f"only {load.alive_procs()}/{load.n} CPU load processes were alive during "
                    "the window; the host memory load was not what was configured")
            if act:
                cov = mean_win / dt if dt > 0 else 0.0
                rec["gpu_window_coverage_of_load_window"] = round(cov, 4)
                if cov < 0.85:
                    fails.append(
                        f"the CPU-load counter window ({dt:.3f}s) is {1/cov:.2f}x the GPU "
                        f"streaming window ({mean_win:.3f}s); the reported host bandwidth is not "
                        "the bandwidth achieved WHILE the GPUs were streaming")
        # MATERIAL swapping during a window invalidates the pinned-H2D and host-memcpy numbers.
        # This box has a steadily-advancing background pswpout of tens of pages/s from unrelated
        # work, so "any page at all" would fail every arm and yield nothing; the gate is a
        # configurable page budget, and the delta is ALWAYS recorded either way.
        swapped = (rec["vmstat_delta"].get("pswpin", 0) + rec["vmstat_delta"].get("pswpout", 0))
        rec["swap_pages_in_window"] = swapped
        if swapped > self.args.max_swap_pages:
            fails.append(
                f"the host swapped {swapped} pages (~{swapped * 4096 / MiB:.1f} MiB) during this "
                f"window, above the --max-swap-pages budget of {self.args.max_swap_pages}; no "
                "bandwidth number from a swapping box is usable")
        elif swapped > 0:
            warns.append(f"{swapped} background swap page(s) during this window (under budget)")
        if warns:
            rec["warnings"] = warns
        if fails:
            rec["status"] = "FAILED"
            rec["failure"] = " | ".join(fails)
        return rec

    def run_verify(self, arm: str, active: list, chunk_bytes) -> dict:
        """Prove the H2D path moved the RIGHT bytes to the RIGHT offsets. Never timed.

        Issued only after an arm's measurement windows are collected, so its D2H read-back
        cannot perturb any bandwidth number. A hipSuccess return is not evidence on this box.
        """
        for o in self.ordinals:
            if o in active:
                self.cmd_qs[o].put({"action": "verify", "arm": arm, "rep": "verify",
                                    "chunk_bytes": chunk_bytes})
            else:
                self.cmd_qs[o].put({"action": "idle", "arm": arm, "rep": "verify"})
        try:
            self.barrier.wait(timeout=self.cfg["barrier_timeout_s"])
        except Exception as e:
            raise ProbeError(f"barrier broke entering verification for arm={arm}: {e!r}") from e
        results = {}
        deadline = time.time() + self.cfg["barrier_timeout_s"]
        while len(results) < len(self.ordinals):
            remaining = deadline - time.time()
            if remaining <= 0:
                raise ProbeError(f"timed out collecting verification for arm={arm}")
            try:
                msg = self.res_q.get(timeout=min(remaining, 5.0))
            except Exception:
                continue
            if msg["kind"] == "error":
                raise ProbeError(f"worker {msg['ord']} error verifying arm={arm}:\n{msg['payload']}")
            results[msg["ord"]] = msg["payload"]
        out = {"ok": True, "cards": {}}
        for o in active:
            r = results.get(o, {})
            out["cards"][str(o)] = {"pci_bdf": self.devices[o]["pci_bdf"],
                                    "ok": bool(r.get("ok")), "checks": r.get("checks"),
                                    "failure": r.get("failure")}
            if not r.get("ok"):
                out["ok"] = False
        return out

    def run_arm(self, arm: str, active: list, chunk_bytes, load: CpuLoad | None) -> dict:
        entry = {
            "arm": arm,
            "active_hip_ordinals": active,
            "active_pci_bdfs": [self.devices[o]["pci_bdf"] for o in active],
            "chunk_bytes": chunk_bytes,
            "window_s": self.args.window_s,
            "warmup_windows_discarded": 1,
            "cpu_load": None if load is None else {"procs": load.n, "buf_bytes": load.buf},
            "reps": [],
            "status": "OK",
        }
        try:
            self.run_window(arm, -1, active, chunk_bytes, load)  # warm-up, DISCARDED
            for rep in range(self.args.reps):
                entry["reps"].append(self.run_window(arm, rep, active, chunk_bytes, load))
            # Correctness gate, AFTER all timed windows so its D2H traffic perturbs nothing.
            if active:
                entry["verification"] = self.run_verify(arm, active, chunk_bytes)
        except ProbeError as e:
            entry["status"] = "FAILED"
            entry["failure"] = str(e)
            return entry

        if active and not entry.get("verification", {}).get("ok"):
            entry["status"] = "FAILED"
            entry["failure"] = (
                "H2D read-back verification FAILED: the copies returned hipSuccess but the "
                "destination did not contain the source bytes at the expected offsets. No "
                "bandwidth number is reported. Details in `verification`.")
            return entry

        good = [r for r in entry["reps"] if r["status"] == "OK"]
        if len(good) < self.args.reps:
            entry["status"] = "FAILED"
            entry["failure"] = (
                f"{self.args.reps - len(good)} of {self.args.reps} reps failed their validity "
                "checks; no summary is reported. Reasons: "
                + " ;; ".join(r.get("failure", "?") for r in entry["reps"]
                              if r["status"] != "OK"))
            return entry
        if active:
            entry["summary"] = {
                "per_card_gbps": {
                    str(o): summarize([r["cards"][str(o)]["gbps"] for r in good]) for o in active},
                "aggregate_gbps": summarize([r["aggregate_gbps"] for r in good]),
                "overlap_fraction": summarize([r["overlap_fraction"] for r in good]),
                # Max over the arm's reps: the trained speed can only be read from a window in
                # which the card was streaming, and taking one rep's value would silently pick
                # an ASPM-idle reading if that rep happened to be sampled between bursts.
                "per_card_trained_link_gbps": {
                    str(o): (max([r["cards"][str(o)]["root_port_trained_gbps_during_window"]
                                  for r in good
                                  if r["cards"][str(o)].get(
                                      "root_port_trained_gbps_during_window")], default=None))
                    for o in active},
                "per_card_trained_link_gbps_min_over_reps": {
                    str(o): (min([r["cards"][str(o)]["root_port_trained_gbps_during_window"]
                                  for r in good
                                  if r["cards"][str(o)].get(
                                      "root_port_trained_gbps_during_window")], default=None))
                    for o in active},
                "per_card_fraction_of_own_trained_link": {
                    str(o): summarize([r["cards"][str(o)]["fraction_of_own_trained_link"]
                                       for r in good])
                    for o in active
                    if good[0]["cards"][str(o)].get("fraction_of_own_trained_link") is not None},
                "per_card_issue_cpu_fraction": {
                    str(o): summarize([r["cards"][str(o)]["issue_cpu_fraction"] for r in good])
                    for o in active
                    if good[0]["cards"][str(o)].get("issue_cpu_fraction") is not None},
                # Provenance of every fraction-of-link below: a "max_link_fallback_*" source
                # means the sysfs samples missed the uptrain and the fraction is a LOWER bound.
                "per_card_link_denominator_source": {
                    str(o): sorted({r["cards"][str(o)].get("link_denominator_source")
                                    for r in good}) for o in active},
            }
            wm = sorted({w for r in good for w in r.get("warnings", [])})
            if wm:
                entry["warnings"] = wm
            if len(active) > 1:
                entry["summary"]["min_share"] = summarize([r["min_share"] for r in good])
                entry["summary"]["imbalance_ratio_max_over_min"] = summarize(
                    [r["imbalance_ratio_max_over_min"] for r in good])
                if all("link_normalized_imbalance" in r for r in good):
                    entry["summary"]["link_normalized_imbalance"] = summarize(
                        [r["link_normalized_imbalance"] for r in good])
        else:
            entry["summary"] = {}
        if load is not None:
            entry["summary"]["cpu_load_gbps_memcpy"] = summarize(
                [r["cpu_load_gbps_memcpy"] for r in good])
            entry["summary"]["cpu_load_ddr_gbps_est_2x_nt"] = summarize(
                [r["cpu_load_ddr_gbps_est_2x_nt"] for r in good])
            entry["summary"]["cpu_load_ddr_gbps_est_3x_rfo"] = summarize(
                [r["cpu_load_ddr_gbps_est_3x_rfo"] for r in good])
            entry["summary"]["cpu_load_ddr_gbps_est"] = \
                entry["summary"]["cpu_load_ddr_gbps_est_2x_nt"]
        return entry


def build_plan(args, ordinals: list) -> list:
    """(arm_name, active_ordinals, chunk_bytes|None, needs_cpu_load)."""
    plan = []
    if args.cpu_load_procs > 0:
        plan.append(("cpu_load_alone", [], None, True))
    for chunk in args.chunk_bytes:
        for o in ordinals:
            plan.append((f"single_card{o}_idle", [o], chunk, False))
        plan.append(("concurrent_idle", list(ordinals), chunk, False))
        if args.cpu_load_procs > 0:
            # BOTH cards get a loaded single-card arm, not just card 0. The two slots on this box
            # are not guaranteed to have trained to the same PCIe generation, so a loaded arm on
            # only one card cannot separate CPU contention from a per-card link difference.
            singles = ordinals if args.loaded_singles == "all" else ordinals[:1]
            for o in singles:
                plan.append((f"single_card{o}_loaded", [o], chunk, True))
            plan.append(("concurrent_loaded", list(ordinals), chunk, True))
    return plan


def derive_headline(arms: list, topo: dict, ordinals: list) -> dict:
    """Only from arms that fully succeeded. Missing input -> the field is absent, never faked."""
    by_key = {(a["arm"], a["chunk_bytes"]): a for a in arms if a["status"] == "OK"}
    out = {"per_chunk": {}, "notes": []}

    load_alone = by_key.get(("cpu_load_alone", None))
    if load_alone and load_alone.get("summary", {}).get("cpu_load_gbps_memcpy"):
        s = load_alone["summary"]
        out["host_memcpy_copy_gbps_no_gpu"] = s["cpu_load_gbps_memcpy"]["median"]
        out["host_memcpy_ddr_gbps_no_gpu_bracket"] = [
            s["cpu_load_ddr_gbps_est_2x_nt"]["median"],
            s["cpu_load_ddr_gbps_est_3x_rfo"]["median"]]
        out["host_memcpy_ddr_gbps_no_gpu"] = s["cpu_load_ddr_gbps_est_2x_nt"]["median"]

    chunks = sorted({k[1] for k in by_key if k[1] is not None})
    for chunk in chunks:
        e = {}
        singles = {}
        for o in ordinals:
            a = by_key.get((f"single_card{o}_idle", chunk))
            if a:
                singles[o] = a["summary"]["per_card_gbps"][str(o)]["median"]
        if singles:
            e["single_card_idle_gbps"] = {str(o): v for o, v in singles.items()}
            e["sum_of_single_card_idle_gbps"] = round(sum(singles.values()), 3)

            e["single_card_idle_trained_link_gbps"] = {
                str(o): by_key[(f"single_card{o}_idle", chunk)]["summary"]
                        ["per_card_trained_link_gbps"][str(o)]
                for o in singles}

        conc = by_key.get(("concurrent_idle", chunk))
        if conc:
            e["concurrent_idle_per_card_gbps"] = {
                o: s["median"] for o, s in conc["summary"]["per_card_gbps"].items()}
            e["concurrent_idle_aggregate_gbps"] = conc["summary"]["aggregate_gbps"]["median"]
            e["concurrent_idle_min_share"] = conc["summary"]["min_share"]["median"]
            e["concurrent_idle_imbalance"] = \
                conc["summary"]["imbalance_ratio_max_over_min"]["median"]
            if "link_normalized_imbalance" in conc["summary"]:
                e["concurrent_idle_link_normalized_imbalance"] = \
                    conc["summary"]["link_normalized_imbalance"]["median"]
            e["concurrent_idle_trained_link_gbps"] = \
                conc["summary"]["per_card_trained_link_gbps"]
            e["concurrent_idle_link_denominator_source"] = \
                conc["summary"].get("per_card_link_denominator_source")
            e["concurrent_idle_fraction_of_own_trained_link"] = {
                o: s["median"] for o, s in
                conc["summary"].get("per_card_fraction_of_own_trained_link", {}).items()}
            e["concurrent_idle_issue_cpu_fraction"] = {
                o: s["median"] for o, s in
                conc["summary"].get("per_card_issue_cpu_fraction", {}).items()}
            if singles and len(singles) == len(ordinals):
                e["concurrency_scaling_efficiency"] = round(
                    e["concurrent_idle_aggregate_gbps"] / sum(singles.values()), 4)
            # Denominator from the links as TRAINED DURING TRANSFER, not from max_link_speed:
            # a slot sitting at Gen4 has half the ceiling its max advertises.
            tl = [v for v in (e.get("concurrent_idle_trained_link_gbps") or {}).values() if v]
            if len(tl) == len(ordinals):
                e["aggregate_trained_pcie_theoretical_gbps"] = round(sum(tl), 2)
                e["aggregate_fraction_of_trained_pcie"] = round(
                    e["concurrent_idle_aggregate_gbps"] / sum(tl), 4)

        cl = by_key.get(("concurrent_loaded", chunk))
        if cl:
            e["concurrent_loaded_per_card_gbps"] = {
                o: s["median"] for o, s in cl["summary"]["per_card_gbps"].items()}
            e["concurrent_loaded_aggregate_gbps"] = cl["summary"]["aggregate_gbps"]["median"]
            e["concurrent_loaded_min_share"] = cl["summary"]["min_share"]["median"]
            if "link_normalized_imbalance" in cl["summary"]:
                e["concurrent_loaded_link_normalized_imbalance"] = \
                    cl["summary"]["link_normalized_imbalance"]["median"]
            e["concurrent_loaded_cpu_ddr_gbps_bracket"] = [
                cl["summary"]["cpu_load_ddr_gbps_est_2x_nt"]["median"],
                cl["summary"]["cpu_load_ddr_gbps_est_3x_rfo"]["median"]]
            e["concurrent_loaded_cpu_ddr_gbps_est"] = \
                cl["summary"]["cpu_load_ddr_gbps_est_2x_nt"]["median"]
            e["concurrent_loaded_issue_cpu_fraction"] = {
                o: s["median"] for o, s in
                cl["summary"].get("per_card_issue_cpu_fraction", {}).items()}
            if conc:
                e["cpu_load_penalty_on_aggregate"] = round(
                    e["concurrent_loaded_aggregate_gbps"] / e["concurrent_idle_aggregate_gbps"], 4)

        for o in ordinals:
            sl = by_key.get((f"single_card{o}_loaded", chunk))
            if sl:
                e.setdefault("single_card_loaded_gbps", {})[str(o)] = \
                    sl["summary"]["per_card_gbps"][str(o)]["median"]
                e.setdefault("single_card_loaded_cpu_ddr_gbps_est", {})[str(o)] = \
                    sl["summary"]["cpu_load_ddr_gbps_est_2x_nt"]["median"]
                if o in singles:
                    e.setdefault("cpu_load_penalty_on_single_card", {})[str(o)] = round(
                        e["single_card_loaded_gbps"][str(o)] / singles[o], 4)
        out["per_chunk"][str(chunk)] = e

    agg_theo = topo.get("aggregate_max_theoretical_gbps")
    if agg_theo:
        out["aggregate_pcie_max_theoretical_gbps"] = agg_theo
        for ck, e in out["per_chunk"].items():
            if "concurrent_idle_aggregate_gbps" in e:
                e["aggregate_fraction_of_pcie_max_theoretical"] = round(
                    e["concurrent_idle_aggregate_gbps"] / agg_theo, 4)
    else:
        out["notes"].append(
            "NO PCIe theoretical denominator could be resolved from the measured cards' root "
            "ports; every fraction-of-theoretical field is deliberately absent rather than "
            "computed against an assumed 2 x 31.5 GB/s.")
    asym = (topo.get("link_asymmetry") or {})
    if asym.get("warning"):
        out["link_asymmetry_warning"] = asym["warning"]
        out["notes"].append(asym["warning"])
    out["notes"].append(
        "concurrency_scaling_efficiency = concurrent aggregate / (sum of the two single-card "
        "medians). ~1.0 => the two links are independent and host DDR did not bind. Materially "
        "< 1.0 => something upstream of the links binds (host DDR the leading candidate) and "
        "every single-card ceiling in the plan's §1 must be scaled by it for TP=2.")
    out["notes"].append(
        "aggregate_fraction_of_TRAINED_pcie uses the link speed sampled DURING transfer at each "
        "card's OWN root port. aggregate_fraction_of_pcie_MAX_theoretical uses max_link_speed and "
        "is an upper bound only: a slot trained one PCIe generation down has half the ceiling its "
        "max advertises, and on this box the two slots are not guaranteed to match.")
    out["notes"].append(
        "cpu_load_*_ddr_gbps_* is a BRACKET, not a counter reading: 2x the copied bytes if glibc "
        "memmove took its non-temporal-store path (read+write), 3x if it did not (read + "
        "read-for-ownership + write). Do not conclude 'the DDR bus is saturated' from the 2x end "
        "alone.")
    out["decisions"] = derive_decisions(out, ordinals)
    return out


def derive_decisions(hl: dict, ordinals: list) -> dict:
    """The plan's three hard decision rules, evaluated explicitly rather than left to the reader.

    Every rule is emitted with its inputs. A rule whose inputs are missing is UNDETERMINED --
    never silently 'passing'.
    """
    PLAN_ZERO_CACHE_TOK_S = 20.2   # plan §1, derived at the single-card idle H2D figure
    PER_RANK_FLOOR_GBPS = 22.0     # brief rule (a): below this, T1's TP=2 ceiling breaks
    MIN_SHARE_FLOOR = 0.40         # brief rule (b)
    out = {}
    for ck, e in hl.get("per_chunk", {}).items():
        d = {}
        eff = e.get("concurrency_scaling_efficiency")
        percard = e.get("concurrent_idle_per_card_gbps") or {}
        if eff is not None:
            worst = min(percard.values()) if percard else None
            d["rule_a_concurrency"] = {
                "concurrency_scaling_efficiency": eff,
                "concurrent_per_card_gbps": percard,
                "restated_zero_cache_dma_tok_s_tp2": round(PLAN_ZERO_CACHE_TOK_S * eff, 2),
                "verdict": ("SCALES" if eff >= 0.95 else
                            "DEGRADED" if eff >= 0.80 else "BINDS_UPSTREAM"),
                "p1_kill_threshold_must_be_rederived_against_gbps": worst,
                "t1_ceiling_at_risk": bool(worst is not None and worst < PER_RANK_FLOOR_GBPS),
            }
            ddr = hl.get("host_memcpy_ddr_gbps_no_gpu_bracket")
            if worst and ddr:
                d["rule_a_concurrency"]["break_even_hit_rate_h_bracket"] = sorted(
                    (round(1 - worst / ddr[0], 4), round(1 - worst / ddr[1], 4)))
                d["rule_a_concurrency"]["break_even_hit_rate_h_note"] = (
                    "plan §1(a): h > 1 - H/DDR, with H taken as the WORST concurrent per-rank "
                    "H2D figure (not the single-card idle one) and DDR as the measured host "
                    "bracket. The bracket is [2x-NT DDR estimate, 3x-RFO DDR estimate].")
        else:
            d["rule_a_concurrency"] = {"verdict": "UNDETERMINED",
                                       "reason": "missing single-card and/or concurrent arms"}
        ms = e.get("concurrent_idle_min_share")
        msl = e.get("concurrent_loaded_min_share")
        if ms is not None or msl is not None:
            worst_share = min(x for x in (ms, msl) if x is not None)
            d["rule_b_fairness"] = {
                "concurrent_idle_min_share": ms,
                "concurrent_loaded_min_share": msl,
                "raw_imbalance": e.get("concurrent_idle_imbalance"),
                "link_normalized_imbalance": e.get("concurrent_idle_link_normalized_imbalance"),
                "per_card_trained_link_gbps": e.get("concurrent_idle_trained_link_gbps"),
                "link_denominator_source": e.get("concurrent_idle_link_denominator_source"),
                "verdict": "STARVES" if worst_share < MIN_SHARE_FLOOR else "FAIR",
                "caveat": ("A raw imbalance driven by a LINK-TRAINING asymmetry is not "
                           "starvation. Read link_normalized_imbalance: ~1.0 there means both "
                           "cards got the same fraction of their own link and the gap is the "
                           "slot, not contention."),
            }
        else:
            d["rule_b_fairness"] = {"verdict": "UNDETERMINED",
                                    "reason": "no concurrent arm succeeded"}
        pen = e.get("cpu_load_penalty_on_aggregate")
        if pen is not None:
            d["rule_c_host_ddr"] = {
                "cpu_load_penalty_on_aggregate": pen,
                "concurrent_idle_aggregate_gbps": e.get("concurrent_idle_aggregate_gbps"),
                "concurrent_loaded_aggregate_gbps": e.get("concurrent_loaded_aggregate_gbps"),
                "host_ddr_alone_bracket": hl.get("host_memcpy_ddr_gbps_no_gpu_bracket"),
                "host_ddr_under_gpu_bracket": e.get("concurrent_loaded_cpu_ddr_gbps_bracket"),
                "issue_thread_cpu_fraction_idle": e.get("concurrent_idle_issue_cpu_fraction"),
                "issue_thread_cpu_fraction_loaded": e.get("concurrent_loaded_issue_cpu_fraction"),
                "verdict": ("DDR_BINDS" if pen < 0.90 else "DDR_DOES_NOT_BIND"),
                "confound_check": ("If issue_thread_cpu_fraction rose sharply under load, the "
                                   "drop may be CPU scheduling contention on the issuing thread "
                                   "rather than DDR bandwidth. Both are recorded."),
            }
        else:
            d["rule_c_host_ddr"] = {"verdict": "UNDETERMINED",
                                    "reason": "concurrent_idle and/or concurrent_loaded missing"}
        out[ck] = d
    return out


# =============================================================================================
# Reporting
# =============================================================================================
def write_markdown(doc: dict, path: str) -> None:
    a = doc["config"]
    L = []
    L.append(f"# P4 — concurrent two-card H2D under host memory pressure")
    L.append("")
    if doc.get("synthetic"):
        L.append("> **SELFTEST OUTPUT — SYNTHETIC, NOT A MEASUREMENT. Do not cite any number here.**")
        L.append("")
    L.append(f"* status: **{doc['status']}**")
    L.append(f"* run: `{doc['host']['hostname']}` {doc['started_utc']} → {doc['finished_utc']}")
    L.append(f"* repo: `{doc['host']['repo_root']}` @ `{doc['host']['git_sha']}`")
    L.append(f"* ROCR_VISIBLE_DEVICES forced to `{doc['host']['rocr_visible_devices_forced']}`; "
             f"HIP_VISIBLE_DEVICES left unset")
    L.append(f"* window {a['window_s']}s × {a['reps']} reps (+1 warm-up discarded), "
             f"{a['streams']} stream(s), queue depth {a['queue_depth']}, "
             f"pinned src {a['src_mb']} MiB/card")
    L.append("")
    L.append("## Cards measured")
    L.append("")
    L.append("| HIP ord | PCI BDF | DRM node | VRAM total | lspci |")
    L.append("|---|---|---|---|---|")
    for o, d in sorted(doc.get("devices", {}).items()):
        L.append(f"| {o} | `{d['pci_bdf']}` | {d['drm_card']} | {d['vram_total_gib']} GiB | "
                 f"{(d.get('lspci') or '').replace('|', '/')} |")
    L.append("")
    L.append("## Topology (read-back confirmation only — settled, not re-litigated)")
    L.append("")
    topo = doc.get("topology") or {}
    L.append(f"> {topo.get('confirmation', '(topology unavailable — the run failed early)')}")
    L.append("")
    for n in topo.get("notes", []):
        L.append(f"* **TOPOLOGY NOTE:** {n}")
    asym = topo.get("link_asymmetry") or {}
    if asym.get("warning"):
        L.append("")
        L.append(f"> **LINK ASYMMETRY:** {asym['warning']}")
        L.append(f">")
        L.append(f"> idle-current per-card theoretical: `{asym.get('idle_current_theoretical_gbps')}`"
                 f" (ratio {asym.get('idle_asymmetry_ratio')})")
    L.append("")
    L.append("## Correctness verification (H2D read-back)")
    L.append("")
    ver = [(a["arm"], a["chunk_bytes"], a.get("verification"))
           for a in doc.get("arms", []) if a.get("verification")]
    if not ver:
        L.append("* none recorded — no arm reached its verification pass.")
    for arm, ck, v in ver:
        L.append(f"* `{arm}` chunk={ck}: **{'PASS' if v.get('ok') else 'FAIL'}** — "
                 f"{ {o: c['ok'] for o, c in v.get('cards', {}).items()} }")
    L.append("")
    L.append("## Box state")
    L.append("")
    for st in (doc.get("box_state_before"), doc.get("box_state_after")):
        if not st or "mem_available_gib" not in st:
            L.append(f"* **{(st or {}).get('tag', '?')}**: unavailable")
            continue
        L.append(f"* **{st['tag']}**: MemAvailable {st['mem_available_gib']} GiB / "
                 f"{st['mem_total_gib']} GiB total, free {st['mem_free_gib']} GiB, "
                 f"cache {st['buffers_cached_gib']} GiB, swap free {st['swap_free_gib']}/"
                 f"{st['swap_total_gib']} GiB, mlocked {st['mlocked_gib']} GiB, "
                 f"pswpout {st['vmstat'].get('pswpout')}, loadavg `{st['loadavg']}`")
    L.append("")
    L.append("## Headline")
    L.append("")
    hl = doc.get("headline", {})
    if "host_memcpy_copy_gbps_no_gpu" in hl:
        br = hl.get("host_memcpy_ddr_gbps_no_gpu_bracket") or []
        L.append(f"* host memcpy with no GPU traffic: "
                 f"**{hl['host_memcpy_copy_gbps_no_gpu']} GB/s copied** → DDR traffic "
                 f"**{br[0] if br else '?'}–{br[1] if len(br) > 1 else '?'} GB/s** "
                 "(bracket: 2× if the copy took glibc's non-temporal-store path, 3× if it "
                 "incurred read-for-ownership). This is the plan's assumed ~45 GB/s DDR bus, "
                 "measured — do not quote a single point from it.")
    for ck, e in hl.get("per_chunk", {}).items():
        label = "2.8 MiB expert granule" if int(ck) == GRANULE_BYTES else f"{int(ck)//MiB} MiB"
        L.append("")
        L.append(f"### chunk = {label} ({ck} B)")
        L.append("")
        L.append("| metric | value |")
        L.append("|---|---|")
        for k, v in e.items():
            L.append(f"| {k} | {v} |")
    L.append("")
    L.append("## Arms")
    L.append("")
    L.append("| arm | chunk | cards | status | per-card GB/s (median) | aggregate GB/s | "
             "min share | overlap |")
    L.append("|---|---|---|---|---|---|---|---|")
    for arm in doc.get("arms", []):
        s = arm.get("summary", {})
        pc = s.get("per_card_gbps", {})
        pcs = ", ".join(f"ord{o}={v['median']}" for o, v in sorted(pc.items())) or "—"
        agg = s.get("aggregate_gbps", {}).get("median", "—")
        share = s.get("min_share", {}).get("median", "—")
        ov = s.get("overlap_fraction", {}).get("median", "—")
        ck = arm["chunk_bytes"]
        L.append(f"| {arm['arm']} | {ck if ck else '—'} | "
                 f"{arm['active_hip_ordinals']} | {arm['status']} | {pcs} | {agg} | {share} | {ov} |")
        if arm["status"] != "OK":
            L.append(f"| ↳ failure | | | | {arm.get('failure', '')} | | | |")
    L.append("")
    L.append("## Decision rules (plan §3 / brief kill criteria)")
    L.append("")
    dec = hl.get("decisions", {})
    if not dec:
        L.append("* UNDETERMINED — no chunk produced a complete arm set.")
    for ck, d in dec.items():
        L.append(f"**chunk {ck} B**")
        L.append("")
        for rule, body in d.items():
            L.append(f"* `{rule}` → **{body.get('verdict')}**")
            for k, v in body.items():
                if k == "verdict":
                    continue
                L.append(f"    * {k}: `{v}`")
        L.append("")
    L.append("## Notes")
    L.append("")
    for n in hl.get("notes", []):
        L.append(f"* {n}")
    L.append("")
    L.append(f"Raw JSON: `{os.path.basename(doc['artifacts']['json'])}`. "
             f"Reproduce with:\n\n```\n{doc['artifacts']['command']}\n```")
    L.append("")
    with open(path, "w") as f:
        f.write("\n".join(L))


def write_outputs(doc: dict, json_path: str, md_path: str) -> None:
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    doc.setdefault("artifacts", {})["json"] = json_path
    doc["artifacts"]["md"] = md_path
    with open(json_path, "w") as f:
        json.dump(doc, f, indent=2, sort_keys=False)
        f.write("\n")
    write_markdown(doc, md_path)


# =============================================================================================
# Selftest
# =============================================================================================
REQUIRED_TOP_KEYS = ("probe", "schema_version", "status", "started_utc", "finished_utc",
                     "host", "config", "topology", "devices", "box_state_before",
                     "box_state_after", "link_state_samples", "arms", "headline", "artifacts")


def synthetic_doc(args) -> dict:
    """A fabricated but structurally identical document, loudly marked as synthetic."""
    ordinals = [0, 1]
    devices = {
        0: {"hip_ordinal": 0, "pci_bdf": "0000:03:00.0", "drm_card": "card1",
            "lspci": "SYNTHETIC", "vram_total_gib": 15.98, "vram_free_gib_at_start": 15.5,
            "pinned_src_bytes": args.src_mb * MiB, "pin_seconds": 0.2,
            "pin_gbps_first_call_NOT_steady_state": 5.0,
            "device_dst_bytes": args.dst_mb * MiB, "streams": args.streams, "pid": -1},
        1: {"hip_ordinal": 1, "pci_bdf": "0000:07:00.0", "drm_card": "card2",
            "lspci": "SYNTHETIC", "vram_total_gib": 15.98, "vram_free_gib_at_start": 15.5,
            "pinned_src_bytes": args.src_mb * MiB, "pin_seconds": 0.2,
            "pin_gbps_first_call_NOT_steady_state": 5.0,
            "device_dst_bytes": args.dst_mb * MiB, "streams": args.streams, "pid": -1},
    }
    arms = []
    # Deliberately ASYMMETRIC fake links (Gen5 vs Gen4), matching what this box actually reports,
    # so the selftest exercises the link-normalisation path rather than a tidy symmetric case.
    fake_link = {0: 31.5, 1: 15.75}
    fake = {"single_card0_idle": [28.1], "single_card1_idle": [14.2],
            "concurrent_idle": [18.0, 13.4], "single_card0_loaded": [24.0],
            "single_card1_loaded": [13.1], "concurrent_loaded": [14.0, 11.5]}
    for arm, active, chunk, loaded in build_plan(args, ordinals):
        e = {"arm": arm, "active_hip_ordinals": active,
             "active_pci_bdfs": [devices[o]["pci_bdf"] for o in active],
             "chunk_bytes": chunk, "window_s": args.window_s,
             "warmup_windows_discarded": 1,
             "cpu_load": ({"procs": args.cpu_load_procs, "buf_bytes": args.cpu_load_buf_mb * MiB}
                          if loaded else None),
             "reps": [], "status": "OK"}
        vals = fake.get(arm, [])
        for rep in range(args.reps):
            rec = {"rep": rep, "cards": {}, "status": "OK"}
            gs, norm = [], []
            for i, o in enumerate(active):
                g = vals[i] * (1.0 + 0.004 * rep)
                gs.append(g)
                norm.append(g / fake_link[o])
                rec["cards"][str(o)] = {
                    "pci_bdf": devices[o]["pci_bdf"], "drm_card": devices[o]["drm_card"],
                    "gbps": round(g, 3), "bytes": int(g * 1e9 * args.window_s),
                    "seconds": args.window_s, "n_copies": 100,
                    "issue_cpu_fraction": 0.12,
                    "link": {"sampled_during_window_gbps": fake_link[o],
                             "sampled_min_in_window_gbps": 2.0,
                             "root_port_max_link_gbps": 31.5,
                             "denominator_gbps": fake_link[o],
                             "denominator_source": "sampled_during_window",
                             "warning": None, "impossible": False},
                    "root_port_trained_gbps_during_window": fake_link[o],
                    "link_denominator_source": "sampled_during_window",
                    "fraction_of_own_trained_link": round(g / fake_link[o], 4)}
            rec["overlap_fraction"] = 0.98 if active else None
            rec["aggregate_gbps"] = round(sum(gs), 3) if gs else None
            rec["vmstat_delta"] = {"pswpin": 0, "pswpout": 0, "pgmajfault": 0}
            if len(active) > 1:
                rec["min_share"] = round(min(gs) / sum(gs), 4)
                rec["max_share"] = round(max(gs) / sum(gs), 4)
                rec["imbalance_ratio_max_over_min"] = round(max(gs) / min(gs), 4)
                rec["link_normalized_shares"] = [round(n, 4) for n in norm]
                rec["link_normalized_imbalance"] = round(max(norm) / min(norm), 4)
            if loaded:
                rec["cpu_load_gbps_memcpy"] = 22.0
                rec["cpu_load_ddr_gbps_est_2x_nt"] = 44.0
                rec["cpu_load_ddr_gbps_est_3x_rfo"] = 66.0
                rec["cpu_load_ddr_gbps_est"] = 44.0
                rec["cpu_load_window_s"] = args.window_s
            e["reps"].append(rec)
        good = e["reps"]
        if active:
            e["summary"] = {
                "per_card_gbps": {str(o): summarize([r["cards"][str(o)]["gbps"] for r in good])
                                  for o in active},
                "aggregate_gbps": summarize([r["aggregate_gbps"] for r in good]),
                "overlap_fraction": summarize([r["overlap_fraction"] for r in good]),
                "per_card_trained_link_gbps": {str(o): fake_link[o] for o in active},
                "per_card_trained_link_gbps_min_over_reps": {
                    str(o): fake_link[o] for o in active},
                "per_card_fraction_of_own_trained_link": {
                    str(o): summarize([r["cards"][str(o)]["fraction_of_own_trained_link"]
                                       for r in good]) for o in active},
                "per_card_issue_cpu_fraction": {
                    str(o): summarize([r["cards"][str(o)]["issue_cpu_fraction"] for r in good])
                    for o in active},
                "per_card_link_denominator_source": {
                    str(o): ["sampled_during_window"] for o in active},
            }
            if len(active) > 1:
                e["summary"]["min_share"] = summarize([r["min_share"] for r in good])
                e["summary"]["imbalance_ratio_max_over_min"] = summarize(
                    [r["imbalance_ratio_max_over_min"] for r in good])
                e["summary"]["link_normalized_imbalance"] = summarize(
                    [r["link_normalized_imbalance"] for r in good])
            e["verification"] = {"ok": True, "cards": {
                str(o): {"pci_bdf": devices[o]["pci_bdf"], "ok": True,
                         "checks": [{"memcmp_equal": True, "header_offset_matches": True}],
                         "failure": None} for o in active}}
        else:
            e["summary"] = {}
        if loaded:
            e["summary"]["cpu_load_gbps_memcpy"] = summarize(
                [r["cpu_load_gbps_memcpy"] for r in good])
            e["summary"]["cpu_load_ddr_gbps_est_2x_nt"] = summarize(
                [r["cpu_load_ddr_gbps_est_2x_nt"] for r in good])
            e["summary"]["cpu_load_ddr_gbps_est_3x_rfo"] = summarize(
                [r["cpu_load_ddr_gbps_est_3x_rfo"] for r in good])
            e["summary"]["cpu_load_ddr_gbps_est"] = e["summary"]["cpu_load_ddr_gbps_est_2x_nt"]
        arms.append(e)

    topo = topology_report([d["pci_bdf"] for d in devices.values()])
    doc = {
        "probe": "P4",
        "schema_version": SCHEMA_VERSION,
        "synthetic": True,
        "WARNING": "SELFTEST OUTPUT — every bandwidth number here is FABRICATED. Not a measurement.",
        "status": "SELFTEST",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "host": host_facts(),
        "config": vars(args),
        "topology": topo,
        "devices": devices,
        "box_state_before": box_state("before", with_rocm_smi=False),
        "box_state_after": box_state("after", with_rocm_smi=False),
        "link_state_samples": [sample_link_state(
            [d["pci_bdf"] for d in devices.values()], "selftest")],
        "arms": arms,
        "headline": derive_headline(arms, topo, ordinals),
        "artifacts": {"command": "SELFTEST"},
    }
    return doc


def run_selftest(args) -> int:
    print("[P4 selftest] no GPU is touched; validating parsers, plan and artifact shape")
    fails = []

    # 1. sysfs / box-state parsers on the real box
    topo = topology_report()
    if not topo["root_ports"]:
        fails.append("topology_report() found no root ports; sysfs layout unexpected")
    for rp, node in topo["root_ports"].items():
        print(f"  root port {rp}: max x{node['max_link_width']} @{node['max_link_speed']} "
              f"-> {node['max_theoretical_gbps']} GB/s theoretical "
              f"(current x{node['current_link_width']} @{node['current_link_speed']} "
              f"-> {node['current_theoretical_gbps']} GB/s "
              "— idle ASPM downtrain is expected; a persistent one-generation-down training is "
              "NOT)")
        if node["max_theoretical_gbps"] is None:
            fails.append(f"root port {rp}: could not derive a theoretical bandwidth")
    asym = topo.get("link_asymmetry") or {}
    if asym.get("warning"):
        print(f"  *** LINK ASYMMETRY DETECTED AT IDLE: "
              f"{asym.get('idle_current_theoretical_gbps')} GB/s per card "
              f"(ratio {asym.get('idle_asymmetry_ratio')})")
        print(f"  *** {asym['warning']}")
    for bdf, c in topo["cards"].items():
        print(f"  card {bdf}: drm={c['drm_card']} chain="
              f"{[n['bdf'] for n in c['chain']]}")
    st = box_state("selftest", with_rocm_smi=False)
    print(f"  box: MemAvailable {st['mem_available_gib']} GiB, free {st['mem_free_gib']} GiB, "
          f"pswpout {st['vmstat'].get('pswpout')}, loadavg {st['loadavg']}")
    if st["mem_total_gib"] <= 0:
        fails.append("meminfo() returned a non-positive MemTotal")

    # 2. argument sanity (the same checks the real run applies)
    try:
        validate_args(args)
    except ProbeError as e:
        fails.append(f"validate_args rejected the selftest arguments: {e}")

    # 3. plan shape
    plan = build_plan(args, [0, 1])
    names = [p[0] for p in plan]
    print(f"  plan ({len(plan)} arms): {names}")
    for want in ("single_card0_idle", "single_card1_idle", "concurrent_idle"):
        if want not in names:
            fails.append(f"plan is missing the {want} arm")
    if args.cpu_load_procs > 0 and "concurrent_loaded" not in names:
        fails.append("plan is missing concurrent_loaded despite --cpu-load-procs > 0")

    # 4. full document shape + writers
    doc = synthetic_doc(args)
    for k in REQUIRED_TOP_KEYS:
        if k not in doc:
            fails.append(f"document is missing required top-level key {k!r}")
    hl = doc["headline"]["per_chunk"]
    for ck, e in hl.items():
        for k in ("concurrent_idle_aggregate_gbps", "concurrency_scaling_efficiency",
                  "concurrent_idle_min_share", "concurrent_idle_link_normalized_imbalance",
                  "aggregate_trained_pcie_theoretical_gbps",
                  "aggregate_fraction_of_trained_pcie"):
            if k not in e:
                fails.append(f"headline[{ck}] is missing {k}")
        for k, v in e.items():
            if isinstance(v, (int, float)) and (v != v or v in (float("inf"), float("-inf"))):
                fails.append(f"headline[{ck}][{k}] is not finite: {v}")

    # The three decision rules must be present and must never silently default to a pass.
    dec = doc["headline"].get("decisions", {})
    if not dec:
        fails.append("headline has no `decisions` block")
    for ck, d in dec.items():
        for rule in ("rule_a_concurrency", "rule_b_fairness", "rule_c_host_ddr"):
            if rule not in d:
                fails.append(f"decisions[{ck}] is missing {rule}")
            elif d[rule].get("verdict") is None:
                fails.append(f"decisions[{ck}][{rule}] has no verdict")
    # The synthetic fixture is an ASYMMETRIC-link case with a starving raw share; the
    # link-normalised figure must show the two cards got comparable fractions of their own links,
    # which is exactly the misreading this probe exists to prevent.
    for ck, d in dec.items():
        b = d.get("rule_b_fairness", {})
        if b.get("link_normalized_imbalance") is None:
            fails.append(f"decisions[{ck}] fairness rule has no link_normalized_imbalance; a raw "
                         "imbalance alone cannot distinguish starvation from link asymmetry")
    if not (doc["topology"].get("link_asymmetry") or {}).get("per_card"):
        fails.append("topology has no per-card link_asymmetry block")

    # Every arm that reports a bandwidth number must carry a passing read-back verification.
    for a in doc["arms"]:
        if a["status"] == "OK" and a["active_hip_ordinals"]:
            if not a.get("verification", {}).get("ok"):
                fails.append(f"arm {a['arm']} reports a bandwidth without a passing verification")

    # The pattern stamp must be position-dependent, or verification cannot detect an offset bug.
    buf = ctypes.create_string_buffer(2 * MiB)
    stamp_pattern(ctypes.addressof(buf), 2 * MiB)
    w = (ctypes.c_uint64 * (512 * (2 * MiB // 4096))).from_address(ctypes.addressof(buf))
    if w[0] == w[512] or int(w[1]) != 0 or int(w[513]) != 4096:
        fails.append("stamp_pattern() is not position-dependent; read-back verification would "
                     "not detect a copy landing at the wrong offset")
    print(f"  pattern stamp: page0 hdr=({w[0]:#x},{w[1]}) page1 hdr=({w[512]:#x},{w[513]}) — "
          "position-dependent")

    # Synthetic artifacts go in a clearly-marked subdirectory so a fabricated number can never be
    # mistaken for a measurement sitting next to the real p4.json.
    sdir = os.path.join(args.out_dir, "_selftest")
    os.makedirs(sdir, exist_ok=True)
    jpath = os.path.join(sdir, "p4.SELFTEST-SYNTHETIC-DO-NOT-CITE.json")
    mpath = os.path.join(sdir, "p4.SELFTEST-SYNTHETIC-DO-NOT-CITE.md")
    if os.path.abspath(jpath) == os.path.abspath(os.path.join(args.out_dir, "p4.json")):
        fails.append("selftest would overwrite the real p4.json")
    write_outputs(doc, jpath, mpath)
    with open(jpath) as f:
        rt = json.load(f)
    if rt.get("synthetic") is not True or rt.get("status") != "SELFTEST":
        fails.append("round-tripped selftest JSON lost its synthetic marker")
    print(f"  wrote {jpath}")
    print(f"  wrote {mpath}")

    # 5. the CPU load generator, briefly, with no GPU involved
    if args.cpu_load_procs > 0:
        ctx = mp.get_context("spawn")
        load = CpuLoad(ctx, min(2, args.cpu_load_procs), 64 * MiB)
        load.start()
        b0 = load.read()
        time.sleep(0.6)
        b1 = load.read()
        load.stop_all()
        gb = (b1 - b0) / 0.6 / 1e9
        print(f"  cpu load generator smoke: {gb:.1f} GB/s copied by 2 procs over 0.6 s")
        if b1 <= b0:
            fails.append("cpu load generator moved no bytes")

    if fails:
        print("\n[P4 selftest] FAILED:", file=sys.stderr)
        for f_ in fails:
            print(f"  - {f_}", file=sys.stderr)
        return 1
    print("\n[P4 selftest] PASS — argument handling, parsers, plan and JSON/MD shape are sound. "
          "No GPU was touched and no bandwidth was measured.")
    return 0


# =============================================================================================
# Args + main
# =============================================================================================
def parse_chunks(s: str) -> list:
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok.lower() in ("granule", "g"):
            out.append(GRANULE_BYTES)
        elif tok.lower().endswith("m"):
            out.append(int(float(tok[:-1]) * MiB))
        else:
            out.append(int(tok))
    return out


def validate_args(args) -> None:
    if args.reps < 5:
        raise ProbeError("--reps must be >= 5 (the brief requires median and spread over >= 5 "
                         "steady-state reps)")
    if args.window_s < 0.25:
        raise ProbeError("--window-s must be >= 0.25 s or the window is dominated by launch jitter")
    if not args.chunk_bytes:
        raise ProbeError("--chunks resolved to an empty list")
    for c in args.chunk_bytes:
        if c < 4096:
            raise ProbeError(f"chunk {c} is below one page")
        if c > args.dst_mb * MiB:
            raise ProbeError(f"chunk {c} exceeds --dst-mb ({args.dst_mb} MiB)")
        if c > args.src_mb * MiB:
            raise ProbeError(f"chunk {c} exceeds --src-mb ({args.src_mb} MiB)")
    if args.streams < 1:
        raise ProbeError("--streams must be >= 1")
    if args.queue_depth < 1:
        raise ProbeError("--queue-depth must be >= 1")
    if not (0.0 < args.min_overlap <= 1.0):
        raise ProbeError("--min-overlap must be in (0, 1]")
    if args.expect_devices != 2:
        raise ProbeError("P4 is a two-card concurrency probe; --expect-devices must be 2")
    if args.cpu_load_procs < 0:
        raise ProbeError("--cpu-load-procs must be >= 0")
    if args.cpu_load_procs and args.cpu_load_buf_mb < 128:
        raise ProbeError("--cpu-load-buf-mb must be >= 128 so the loaders' working set clears the "
                         "96 MB L3 and the traffic is real DDR traffic")
    if args.loaded_singles not in ("all", "first"):
        raise ProbeError("--loaded-singles must be 'all' or 'first'")

    # ---- cache-residency guard. The device MALL is 64 MB and the host L3 is ~96 MB. A working
    # set that fits in either is not a PCIe measurement (the withdrawn 127.2 GB/s figure in
    # vmm_probe2.py was exactly this artifact at 2 MiB).
    if args.src_mb * MiB < MIN_WORKING_SET_BYTES:
        raise ProbeError(
            f"--src-mb {args.src_mb} MiB is below the {MIN_WORKING_SET_BYTES // MiB} MiB "
            "minimum working set (4x the 64 MB MALL); the source could be cache-resident and "
            "the number would not be a PCIe measurement")
    if args.dst_mb * MiB < MIN_WORKING_SET_BYTES:
        raise ProbeError(
            f"--dst-mb {args.dst_mb} MiB is below the {MIN_WORKING_SET_BYTES // MiB} MiB "
            "minimum working set (4x the 64 MB MALL)")
    for c in args.chunk_bytes:
        if args.src_mb * MiB // c < 2:
            raise ProbeError(
                f"chunk {c} leaves only {args.src_mb * MiB // c} distinct source slot(s); the "
                "stream would re-read one region and could be served from cache. Raise --src-mb.")

    # ---- host memory precondition. Pinned pages cannot be reclaimed; if the run pushes the box
    # into swap, every number in it is worthless (and this box already has swap in use).
    need_gib = (2 * args.src_mb * MiB
                + (args.cpu_load_procs * 2 * args.cpu_load_buf_mb * MiB)) / GiB
    mi = meminfo()
    avail = mi.get("MemAvailable", 0) / (1 << 20)
    if avail < need_gib + args.mem_headroom_gib:
        raise ProbeError(
            f"the run needs ~{need_gib:.1f} GiB of host RAM ({args.src_mb} MiB pinned x 2 cards + "
            f"{args.cpu_load_procs} loaders x 2 x {args.cpu_load_buf_mb} MiB) plus "
            f"{args.mem_headroom_gib} GiB headroom, but MemAvailable is {avail:.1f} GiB. "
            "Refusing to run: a swapping box produces a confident wrong number.")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="P4 — concurrent two-card H2D under host memory pressure",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--reps", type=int, default=5, help="steady-state windows per arm (>=5)")
    p.add_argument("--window-s", type=float, default=3.0,
                   help="seconds per measurement window (long enough that a single in-flight "
                        "chunk is a negligible fraction of the window)")
    p.add_argument("--chunks", type=str, default="granule,256M",
                   help="comma list of H2D chunk sizes ('granule' = 2.8 MiB expert granule)")
    p.add_argument("--src-mb", type=int, default=1024, help="pinned host source per card, MiB")
    p.add_argument("--dst-mb", type=int, default=512, help="device destination per card, MiB")
    p.add_argument("--streams", type=int, default=1, help="HIP streams per card")
    p.add_argument("--queue-depth", type=int, default=8,
                   help="async copies issued before each stream sync")
    p.add_argument("--cpu-load-procs", type=int, default=6,
                   help="synthetic host memory-load processes (0 disables the loaded arms)")
    p.add_argument("--cpu-load-buf-mb", type=int, default=256,
                   help="per-loader buffer size, MiB (two buffers each)")
    p.add_argument("--min-overlap", type=float, default=0.90,
                   help="minimum measured overlap of the two cards' windows for a concurrent rep "
                        "to count")
    p.add_argument("--loaded-singles", type=str, default="all", choices=("all", "first"),
                   help="run the CPU-loaded single-card arm on both cards ('all') or only the "
                        "first ordinal ('first'). 'all' is needed to separate CPU contention "
                        "from a per-card PCIe link difference.")
    p.add_argument("--max-swap-pages", type=int, default=2048,
                   help="swap pages (in+out) tolerated within one measurement window before the "
                        "window is FAILED; 2048 pages = 8 MiB. This box has a background "
                        "pswpout of tens of pages/s from unrelated work, so 0 is not usable.")
    p.add_argument("--mem-headroom-gib", type=float, default=8.0,
                   help="host RAM that must remain available beyond the run's own footprint, or "
                        "the probe refuses to start")
    p.add_argument("--expect-devices", type=int, default=2,
                   help="HIP device count that MUST enumerate; anything else aborts")
    p.add_argument("--vram-min-gib", type=float, default=8.0,
                   help="lower bound on a device's VRAM total; an iGPU tripwire")
    p.add_argument("--vram-max-gib", type=float, default=20.0,
                   help="upper bound on a device's VRAM total; an iGPU tripwire")
    p.add_argument("--hip-so", type=str, default="libamdhip64.so")
    p.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR)
    p.add_argument("--startup-timeout-s", type=float, default=180.0)
    p.add_argument("--selftest", action="store_true",
                   help="validate args, parsers and artifact shape WITHOUT touching the GPU")
    p.add_argument("--dry-run", action="store_true", help="alias for --selftest")
    return p


def main(argv=None) -> int:
    args = build_argparser().parse_args(argv)
    args.chunk_bytes = parse_chunks(args.chunks)
    args.out_dir = os.path.abspath(args.out_dir)
    if (args.out_dir == "/tmp" or args.out_dir.startswith("/tmp/")
            or args.out_dir == "/dev/shm" or args.out_dir.startswith("/dev/shm/")
            or args.out_dir.startswith("/run/")):
        print(f"P4 PRECONDITION FAILURE: --out-dir={args.out_dir} is tmpfs/volatile. "
              "Measurement fixtures and results must be durable and recorded — write into the "
              "worktree (default: docs/measurements/WEIGHT_OFFLOAD_2026-09-02/).", file=sys.stderr)
        return 2
    os.makedirs(args.out_dir, exist_ok=True)

    if args.selftest or args.dry_run:
        try:
            return run_selftest(args)
        except ProbeError as e:
            print(f"P4 SELFTEST PRECONDITION FAILURE: {e}", file=sys.stderr)
            return 2

    try:
        validate_args(args)
    except ProbeError as e:
        print(f"P4 PRECONDITION FAILURE: {e}", file=sys.stderr)
        return 2

    json_path = os.path.join(args.out_dir, "p4.json")
    md_path = os.path.join(args.out_dir, "p4.md")
    command = ("ROCR_VISIBLE_DEVICES=0,1 python3 tools/offload/p4_pcie_concurrency.py "
               + " ".join(argv if argv is not None else sys.argv[1:]))

    doc = {
        "probe": "P4",
        "schema_version": SCHEMA_VERSION,
        "synthetic": False,
        "status": "INCOMPLETE",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "finished_utc": None,
        "host": host_facts(),
        "config": {k: v for k, v in vars(args).items()},
        "topology": None,
        "devices": {},
        "box_state_before": None,
        "box_state_after": None,
        "link_state_samples": [],
        "arms": [],
        "headline": {},
        "artifacts": {"command": command},
    }

    runner = None
    load = None
    rc = 0
    try:
        if _PRIOR_ROCR not in (None, "0,1") or _PRIOR_HIP_VD is not None:
            print(f"[P4] NOTE: overriding inherited device visibility "
                  f"(ROCR_VISIBLE_DEVICES={_PRIOR_ROCR!r}, HIP_VISIBLE_DEVICES={_PRIOR_HIP_VD!r}) "
                  "with ROCR_VISIBLE_DEVICES=0,1 and HIP_VISIBLE_DEVICES unset. P4 needs BOTH "
                  "discrete cards; if a lease assigned you one card, stop and re-lease -n 2.",
                  file=sys.stderr)
        doc["topology"] = topology_report()  # provisional; re-anchored once the workers report
        print("[P4] topology (provisional, fallback BDFs): " + doc["topology"]["confirmation"])
        doc["box_state_before"] = box_state("before")
        b = doc["box_state_before"]
        print(f"[P4] box before: MemAvailable {b['mem_available_gib']} GiB / "
              f"{b['mem_total_gib']} GiB, free {b['mem_free_gib']} GiB, "
              f"pswpout {b['vmstat'].get('pswpout')}, loadavg {b['loadavg']}")

        runner = Runner(args)
        runner.link_samples = doc["link_state_samples"]
        print(f"[P4] starting {args.expect_devices} GPU workers "
              f"(pinning {args.src_mb} MiB per card)...")
        runner.start_workers()
        doc["devices"] = {str(o): d for o, d in runner.devices.items()}
        for o, d in sorted(runner.devices.items()):
            print(f"[P4]   hip ordinal {o} -> {d['pci_bdf']} ({d['drm_card']}), "
                  f"VRAM {d['vram_total_gib']} GiB, pin took {d['pin_seconds']}s")

        # Re-derive the topology anchored on the BDFs the workers ACTUALLY reported. Until this
        # point the report was built from the fallback pair, which may describe cards that
        # produce none of the numbers below.
        doc["topology"] = topology_report(runner.bdfs)
        print("[P4] topology (anchored on measured cards): "
              + doc["topology"]["confirmation"])
        for n in doc["topology"].get("notes", []):
            print(f"[P4] TOPOLOGY NOTE: {n}", file=sys.stderr)
        warn = (doc["topology"].get("link_asymmetry") or {}).get("warning")
        if warn:
            print(f"[P4] *** {warn}", file=sys.stderr)

        plan = build_plan(args, runner.ordinals)
        for i, (arm, active, chunk, loaded) in enumerate(plan):
            if loaded:
                load = CpuLoad(runner.ctx, args.cpu_load_procs, args.cpu_load_buf_mb * MiB)
                print(f"[P4] ({i+1}/{len(plan)}) {arm}: starting {args.cpu_load_procs} "
                      f"CPU load procs x {args.cpu_load_buf_mb} MiB x2 ...")
                load.start()
            else:
                print(f"[P4] ({i+1}/{len(plan)}) {arm} "
                      f"chunk={chunk} cards={active} ...")
            try:
                entry = runner.run_arm(arm, active, chunk, load)
            finally:
                if load is not None:
                    load.stop_all()
                    load = None
            doc["arms"].append(entry)
            if entry["status"] != "OK":
                print(f"[P4]   arm {arm} FAILED: {entry.get('failure')}", file=sys.stderr)
                rc = 3
                dead = [o for o, p in runner.procs.items() if not p.is_alive()]
                if dead:
                    print(f"[P4] GPU worker(s) {dead} are dead — abandoning the remaining "
                          f"{len(plan)-i-1} arm(s); partial results are still written.",
                          file=sys.stderr)
                    break
                continue
            s = entry.get("summary", {})
            if active:
                pcs = ", ".join(f"ord{o}={v['median']:.2f}"
                                for o, v in sorted(s["per_card_gbps"].items()))
                extra = ""
                if len(active) > 1:
                    extra = (f", min_share={s['min_share']['median']:.3f}, "
                             f"imbalance={s['imbalance_ratio_max_over_min']['median']:.3f}")
                if "cpu_load_ddr_gbps_est" in s:
                    extra += f", cpu_ddr~{s['cpu_load_ddr_gbps_est']['median']:.1f} GB/s"
                print(f"[P4]   {pcs} | aggregate={s['aggregate_gbps']['median']:.2f} GB/s"
                      f" (spread {s['aggregate_gbps']['spread_pct_of_median']}%)"
                      f"{extra}")
            elif "cpu_load_gbps_memcpy" in s:
                print(f"[P4]   host memcpy alone: {s['cpu_load_gbps_memcpy']['median']:.2f} GB/s "
                      f"copied (~{s['cpu_load_ddr_gbps_est']['median']:.1f} GB/s DDR traffic)")

        doc["headline"] = derive_headline(doc["arms"], doc["topology"], runner.ordinals)
        doc["status"] = "OK" if rc == 0 else "PARTIAL"
    except ProbeError as e:
        doc["status"] = "FAILED"
        doc["failure"] = str(e)
        print(f"\nP4 FAILURE: {e}", file=sys.stderr)
        rc = 2
    except KeyboardInterrupt:
        doc["status"] = "ABORTED"
        doc["failure"] = "KeyboardInterrupt"
        print("\nP4 ABORTED by user", file=sys.stderr)
        rc = 130
    except Exception as e:  # noqa: BLE001
        doc["status"] = "FAILED"
        doc["failure"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        print(f"\nP4 UNEXPECTED FAILURE: {e}\n{traceback.format_exc()}", file=sys.stderr)
        rc = 2
    finally:
        if load is not None:
            load.stop_all()
        if runner is not None:
            try:
                runner.stop_workers()
            except Exception:
                pass
        try:
            doc["box_state_after"] = box_state("after")
        except ProbeError:
            doc["box_state_after"] = {"tag": "after", "error": "unreadable"}
        doc["finished_utc"] = datetime.now(timezone.utc).isoformat()
        try:
            write_outputs(doc, json_path, md_path)
            print(f"\n[P4] wrote {json_path}")
            print(f"[P4] wrote {md_path}")
        except Exception as e:  # noqa: BLE001
            print(f"P4: could not write artifacts: {e}", file=sys.stderr)
            rc = rc or 2

    if doc["status"] == "OK":
        print("\n=== P4 machine-readable headline ===")
        print(json.dumps({"probe": "P4", "status": doc["status"],
                          "devices": {o: d["pci_bdf"] for o, d in doc["devices"].items()},
                          "topology_confirmation": doc["topology"]["confirmation"],
                          "headline": doc["headline"]}, indent=2))
    else:
        print(f"\n=== P4 status: {doc['status']} — see {json_path} ===", file=sys.stderr)
    return rc


if __name__ == "__main__":
    sys.exit(main())
