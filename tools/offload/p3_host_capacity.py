#!/usr/bin/env python3
"""P3 -- Host-located VMM capacity probe.

Question (WEIGHT_OFFLOAD_PLAN.md sec.3, P3): does `hipMemCreate(location=Host)` deliver
34 GiB for one rank, and 68 GiB for two ranks *concurrently*, on THIS box, with the
engine loaded and ZFS ARC warm?  And at what allocation RATE?

FORKS THE DESIGN.
  * fits  -> the mixed device/host single stack (T1/T2) is viable.
  * fails -> fall back to file-backed mmap(MAP_SHARED) + hipHostRegister, which forfeits
             the device tier and drags route_E/align_E (sec.4.4) back into scope.

Design notes -- why the probe is shaped this way:
  * hipMemCreate may be LAZY.  A handle that costs 9 us and commits nothing would make a
    "68 GiB fits" answer a fiction.  So every chunk is create -> map -> setAccess ->
    first-touch-through-the-device-pointer -> read-back-verify, and MemAvailable is sampled
    at each phase boundary so commit timing is attributed, not assumed.
    (Plan trap: "a capability probe can PASS while the operation FAILS".)
  * First touch is `hipMemsetD32` through the device VA, never a CPU store -- the plan
    (sec.5.2) forbids CPU stores into host-located pages because the coherence granularity
    is unstated.
  * The two ranks are two PROCESSES growing in LOCKSTEP, one chunk each per round, with a
    single authoritative box-state sampler in the parent between rounds.  Concurrency is
    the actual question; a single process allocating 68 GiB would not answer it.
  * Stops at an 8 GiB MemAvailable floor, or on swap thrash measured against a baseline
    window (this box swaps at idle: pswpout is already ~4e8 pages), or on a HIP error.
    A partial climb is still the answer, so an abort still writes JSON.

Rates are DECIMAL GB/s (1e9 B/s) to be comparable with the recorded hipHostMalloc figure
of 4.9-5.6 GB/s.  Capacities are GiB (2^30) unless a field says otherwise.
"""

from __future__ import annotations

import os
import sys

# ---------------------------------------------------------------------------
# Device visibility.  MUST happen before libamdhip64 is dlopen'd.
# ROCm device 2 on this box is the Ryzen iGPU advertising ~47 GB of GTT; if it enters
# enumeration it poisons any "biggest free pool" logic.  Never a compute target.
# ---------------------------------------------------------------------------
_ORIG_ROCR = os.environ.get("ROCR_VISIBLE_DEVICES")
_ORIG_HIP = os.environ.get("HIP_VISIBLE_DEVICES")
os.environ["ROCR_VISIBLE_DEVICES"] = "0,1"
os.environ.pop("HIP_VISIBLE_DEVICES", None)

import argparse  # noqa: E402
import ctypes  # noqa: E402
import errno  # noqa: E402
import fcntl  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import re  # noqa: E402
import select  # noqa: E402
import signal  # noqa: E402
import socket  # noqa: E402
import statistics  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

SCHEMA_VERSION = 1
PROBE_ID = "P3"
LIBHIP = "libamdhip64.so"

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUTDIR = REPO / "docs" / "measurements" / "WEIGHT_OFFLOAD_2026-09-02"
LOCKFILE = Path(__file__).resolve().parent / ".p3.lock"

ENGINE_RSS_MIN_KIB = 2 * 1024 * 1024  # 2 GiB: below this a name match is not an engine

GIB = 1 << 30
MIB = 1 << 20
PAGE = 4096


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _log(msg: str) -> None:
    print(f"[p3 {time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


class ProbeError(RuntimeError):
    """Precondition or measurement failure.  Always fatal, always non-zero exit."""


# ===========================================================================
# Box state -- pure /proc + best-effort tools.  No fabrication: anything we
# could not read comes back as None, never as 0.
# ===========================================================================

def read_meminfo() -> dict:
    out = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            k, _, rest = line.partition(":")
            parts = rest.split()
            if parts:
                try:
                    out[k] = int(parts[0])  # kB
                except ValueError:
                    pass
    except OSError as exc:  # pragma: no cover
        raise ProbeError(f"cannot read /proc/meminfo: {exc}") from exc
    return out


def read_vmstat() -> dict:
    out = {}
    try:
        for line in Path("/proc/vmstat").read_text().splitlines():
            parts = line.split()
            if len(parts) == 2:
                try:
                    out[parts[0]] = int(parts[1])
                except ValueError:
                    pass
    except OSError as exc:  # pragma: no cover
        raise ProbeError(f"cannot read /proc/vmstat: {exc}") from exc
    return out


def read_zfs_arc() -> dict:
    p = Path("/proc/spl/kstat/zfs/arcstats")
    if not p.exists():
        return {"present": False, "size_kib": None, "c_max_kib": None}
    vals = {}
    try:
        for line in p.read_text().splitlines()[2:]:
            parts = line.split()
            if len(parts) >= 3:
                try:
                    vals[parts[0]] = int(parts[2])
                except ValueError:
                    pass
    except OSError:
        return {"present": True, "size_kib": None, "c_max_kib": None}
    size = vals.get("size")
    cmax = vals.get("c_max")
    return {
        "present": True,
        "size_kib": None if size is None else size // 1024,
        "c_max_kib": None if cmax is None else cmax // 1024,
        "arc_meta_used_kib": (vals["arc_meta_used"] // 1024) if "arc_meta_used" in vals else None,
    }


def read_zram() -> list:
    out = []
    for d in sorted(Path("/sys/block").glob("zram*")):
        entry = {"dev": d.name, "disksize_bytes": None, "orig_data_bytes": None,
                 "compr_data_bytes": None, "mem_used_total_bytes": None}
        try:
            entry["disksize_bytes"] = int((d / "disksize").read_text().strip())
        except OSError:
            pass
        try:
            mm = (d / "mm_stat").read_text().split()
            if len(mm) >= 3:
                entry["orig_data_bytes"] = int(mm[0])
                entry["compr_data_bytes"] = int(mm[1])
                entry["mem_used_total_bytes"] = int(mm[2])
        except (OSError, ValueError):
            pass
        out.append(entry)
    return out


def read_swaps() -> list:
    out = []
    try:
        lines = Path("/proc/swaps").read_text().splitlines()[1:]
    except OSError:
        return out
    for line in lines:
        parts = line.split()
        if len(parts) >= 5:
            out.append({"file": parts[0], "type": parts[1],
                        "size_kib": int(parts[2]), "used_kib": int(parts[3]),
                        "priority": int(parts[4])})
    return out


def detect_engine() -> dict:
    """Best-effort: is a serve/engine actually loaded?  The plan requires this probe run
    on a LOADED box, not an idle one.  We record the evidence rather than assert it."""
    procs = []
    pat = re.compile(r"(minisgl|vllm|sglang|llama[-_]?server|llama-cli|lemonade)", re.I)
    me = os.getpid()
    mine = {me, os.getppid()}
    try:
        for d in Path("/proc").iterdir():
            if not d.name.isdigit():
                continue
            pid = int(d.name)
            if pid in mine:
                continue
            try:
                cmd = (d / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
            except OSError:
                continue
            if not cmd or not pat.search(cmd):
                continue
            # This probe lives under a worktree whose path contains "minisgl": a bare regex
            # hit on our own file name is not an engine.
            if "p3_host_capacity.py" in cmd or "/tools/offload/" in cmd:
                continue
            # An agent shell that merely sourced a snapshot under a path containing
            # "minisgl" is not an engine.
            if re.match(r"^/(bin|usr/bin)/(ba|z|da)?sh\b", cmd):
                continue
            rss_kib = None
            try:
                for line in (d / "status").read_text().splitlines():
                    if line.startswith("VmRSS:"):
                        rss_kib = int(line.split()[1])
                        break
            except OSError:
                pass
            procs.append({"pid": int(d.name), "rss_kib": rss_kib, "cmd": cmd[:240]})
    except OSError:
        pass
    containers = None
    try:
        r = subprocess.run(["docker", "ps", "--format", "{{.Names}}\t{{.Image}}\t{{.Status}}"],
                           capture_output=True, text=True, timeout=20)
        if r.returncode == 0:
            containers = [ln for ln in r.stdout.splitlines() if ln.strip()]
    except (OSError, subprocess.SubprocessError):
        containers = None
    total_rss = sum(p["rss_kib"] for p in procs if p["rss_kib"]) or None
    # A name match with a trivial RSS is a shell or a grep, not a loaded engine.
    heavy = [p for p in procs if (p["rss_kib"] or 0) >= ENGINE_RSS_MIN_KIB]
    heavy_rss = sum(p["rss_kib"] for p in heavy if p["rss_kib"]) or None

    # A container called "minisgl-cloudflared" is not an engine.  Require a serve-shaped
    # IMAGE and exclude the box's standing infra.
    # Match the IMAGE, never the container NAME: this box runs "minisgl-cloudflared" and
    # "vllm-gfx1201-gpu-exporter", neither of which is an engine.
    serve_img = re.compile(r"(minisgl|vllm|sglang|llama|lemonade|rocm/)", re.I)
    infra = re.compile(r"(cloudflared|grafana|prometheus|postgres|rabbitmq|redis|"
                       r"firecrawl|playwright|exporter|loki|tempo|cadvisor)", re.I)
    engine_containers = None
    if containers is not None:
        engine_containers = []
        for line in containers:
            fields = line.split("\t")
            name = fields[0] if fields else ""
            image = fields[1] if len(fields) > 1 else ""
            if serve_img.search(image) and not infra.search(image) and not infra.search(name):
                engine_containers.append(line)

    # Ground truth: is anything actually resident on a discrete card?
    cards = read_gpu_vram_sysfs()
    discrete = [c for c in cards if c.get("discrete")]
    vram_used = [c["vram_used_bytes"] for c in discrete if c.get("vram_used_bytes") is not None]
    max_vram = max(vram_used) if vram_used else None
    vram_busy = max_vram is not None and max_vram >= ENGINE_VRAM_MIN_BYTES

    basis = []
    if vram_busy:
        basis.append("discrete-card VRAM in use")
    if heavy:
        basis.append("process rss")
    if engine_containers:
        basis.append("serve container")
    return {
        "matched_processes": procs,
        "matched_process_rss_kib": total_rss,
        "heavy_processes": heavy,
        "heavy_process_rss_kib": heavy_rss,
        "engine_rss_min_kib": ENGINE_RSS_MIN_KIB,
        "docker_ps": containers,
        "engine_containers": engine_containers,
        "discrete_cards": discrete,
        "max_discrete_vram_used_bytes": max_vram,
        "engine_vram_min_bytes": ENGINE_VRAM_MIN_BYTES,
        "engine_detected": bool(vram_busy or heavy or engine_containers),
        "detection_basis": basis or ["none"],
    }


def rocm_smi_snapshot(skip: bool) -> dict:
    if skip:
        return {"collected": False, "reason": "--skip-rocm-smi"}
    for argv in (["rocm-smi", "--showid", "--showbus", "--showmeminfo", "vram", "--json"],
                 ["rocm-smi", "--showid", "--showbus", "--showmeminfo", "vram"]):
        try:
            r = subprocess.run(argv, capture_output=True, text=True, timeout=40)
        except FileNotFoundError:
            return {"collected": False, "reason": "rocm-smi not on PATH"}
        except subprocess.SubprocessError as exc:
            return {"collected": False, "reason": f"rocm-smi failed: {exc}"}
        if r.returncode == 0 and r.stdout.strip():
            if argv[-1] == "--json":
                try:
                    return {"collected": True, "format": "json", "data": json.loads(r.stdout)}
                except json.JSONDecodeError:
                    continue
            return {"collected": True, "format": "text", "raw": r.stdout}
    return {"collected": False, "reason": "rocm-smi produced no usable output"}


def box_state(tag: str, skip_smi: bool = True) -> dict:
    mi = read_meminfo()
    vm = read_vmstat()
    return {
        "tag": tag,
        "t_utc": _now_utc(),
        "t_mono": time.monotonic(),
        "meminfo_kib": {k: mi.get(k) for k in (
            "MemTotal", "MemFree", "MemAvailable", "Buffers", "Cached", "SwapCached",
            "SwapTotal", "SwapFree", "Dirty", "Writeback", "AnonPages", "Mapped",
            "Shmem", "Slab", "SReclaimable", "SUnreclaim", "Mlocked", "Unevictable",
            "CommitLimit", "Committed_AS")},
        "vmstat": {k: vm.get(k) for k in (
            "pswpin", "pswpout", "pgpgin", "pgpgout", "pgfault", "pgmajfault",
            "nr_free_pages", "nr_anon_pages", "compact_stall", "allocstall_normal")},
        "loadavg": _loadavg(),
        "zfs_arc": read_zfs_arc(),
        "zram": read_zram(),
        "swaps": read_swaps(),
        "gpu_vram_sysfs": read_gpu_vram_sysfs(),
        "rocm_smi": rocm_smi_snapshot(skip_smi),
    }


def _loadavg():
    try:
        return [float(x) for x in Path("/proc/loadavg").read_text().split()[:3]]
    except (OSError, ValueError):
        return None


DISCRETE_VRAM_MIN_BYTES = 8 * GIB      # the two gfx1201 cards are 16 GiB; the iGPU is 2 GiB
ENGINE_VRAM_MIN_BYTES = 1 * GIB        # above this, something real is resident on a card


def read_gpu_vram_sysfs() -> list:
    """Per-card VRAM from sysfs.  This is the ground truth for 'is an engine loaded' and
    for 'host-located pages did not silently eat VRAM' -- and it costs nothing, needs no
    ROCm tool, and never touches /dev/kfd."""
    out = []
    for dev in sorted(Path("/sys/class/drm").glob("card*/device")):
        tot = dev / "mem_info_vram_total"
        if not tot.exists():
            continue
        entry = {"sysfs": str(dev), "vram_total_bytes": None, "vram_used_bytes": None,
                 "gtt_used_bytes": None, "gtt_total_bytes": None,
                 "pci_slot": None, "pci_id": None, "discrete": None}
        # GTT is where host memory pinned for the GPU is accounted.  Without it, an
        # allocation that landed in host RAM and one that landed nowhere both read as
        # "VRAM did not grow", which is precisely the ambiguity this probe must resolve.
        for key, f in (("vram_total_bytes", "mem_info_vram_total"),
                       ("vram_used_bytes", "mem_info_vram_used"),
                       ("gtt_used_bytes", "mem_info_gtt_used"),
                       ("gtt_total_bytes", "mem_info_gtt_total")):
            try:
                entry[key] = int((dev / f).read_text().strip())
            except (OSError, ValueError):
                pass
        try:
            for line in (dev / "uevent").read_text().splitlines():
                if line.startswith("PCI_SLOT_NAME="):
                    entry["pci_slot"] = line.split("=", 1)[1].strip()
                elif line.startswith("PCI_ID="):
                    entry["pci_id"] = line.split("=", 1)[1].strip()
        except OSError:
            pass
        t = entry["vram_total_bytes"]
        entry["discrete"] = None if t is None else bool(t >= DISCRETE_VRAM_MIN_BYTES)
        out.append(entry)
    return out


def _is_volatile(path: Path) -> bool:
    """True if `path` lives on a filesystem that does not survive a reboot."""
    try:
        rp = str(Path(path).resolve())
    except OSError:
        return True
    for prefix in ("/tmp/", "/dev/shm/", "/run/", "/var/tmp/"):
        if rp.startswith(prefix) or rp == prefix.rstrip("/"):
            return True
    best = ("", "")
    try:
        for line in Path("/proc/self/mounts").read_text().splitlines():
            parts = line.split()
            if len(parts) < 3:
                continue
            mnt, fstype = parts[1], parts[2]
            if (rp == mnt or rp.startswith(mnt.rstrip("/") + "/")) and len(mnt) > len(best[0]):
                best = (mnt, fstype)
    except OSError:
        return False
    return best[1] in ("tmpfs", "ramfs", "devtmpfs")


def host_info() -> dict:
    cpu = None
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {
        "hostname": socket.gethostname(),
        "kernel": platform.release(),
        "python": sys.version.split()[0],
        "cpu_model": cpu,
        "cpu_count": os.cpu_count(),
    }


# ===========================================================================
# HIP driver bindings (ctypes).  No graceful degradation: a missing symbol raises.
# ===========================================================================

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


HIP_MEM_ALLOCATION_TYPE_PINNED = 0x1
HIP_MEM_LOCATION_TYPE_DEVICE = 1
HIP_MEM_LOCATION_TYPE_HOST = 2
HIP_MEM_ACCESS_FLAGS_PROT_READWRITE = 3
HIP_MEMCPY_DEVICE_TO_HOST = 2

_SIGS = {
    "hipGetDeviceCount": ([ctypes.POINTER(ctypes.c_int)], ctypes.c_int),
    "hipSetDevice": ([ctypes.c_int], ctypes.c_int),
    "hipDeviceGet": ([ctypes.POINTER(ctypes.c_int), ctypes.c_int], ctypes.c_int),
    "hipDeviceGetName": ([ctypes.c_char_p, ctypes.c_int, ctypes.c_int], ctypes.c_int),
    "hipDeviceGetPCIBusId": ([ctypes.c_char_p, ctypes.c_int, ctypes.c_int], ctypes.c_int),
    "hipMemGetInfo": ([ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)], ctypes.c_int),
    "hipDeviceSynchronize": ([], ctypes.c_int),
    "hipMemGetAllocationGranularity": (
        [ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(hipMemAllocationProp), ctypes.c_int],
        ctypes.c_int),
    "hipMemAddressReserve": ([ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t,
                              ctypes.c_size_t, ctypes.c_void_p, ctypes.c_ulonglong], ctypes.c_int),
    "hipMemAddressFree": ([ctypes.c_void_p, ctypes.c_size_t], ctypes.c_int),
    "hipMemCreate": ([ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t,
                      ctypes.POINTER(hipMemAllocationProp), ctypes.c_ulonglong], ctypes.c_int),
    "hipMemMap": ([ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
                   ctypes.c_void_p, ctypes.c_ulonglong], ctypes.c_int),
    "hipMemUnmap": ([ctypes.c_void_p, ctypes.c_size_t], ctypes.c_int),
    "hipMemSetAccess": ([ctypes.c_void_p, ctypes.c_size_t,
                         ctypes.POINTER(hipMemAccessDesc), ctypes.c_size_t], ctypes.c_int),
    "hipMemRelease": ([ctypes.c_void_p], ctypes.c_int),
    "hipMemsetD32": ([ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t], ctypes.c_int),
    "hipMemcpy": ([ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int], ctypes.c_int),
    "hipHostMalloc": ([ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint], ctypes.c_int),
    "hipHostFree": ([ctypes.c_void_p], ctypes.c_int),
    "hipGetErrorString": ([ctypes.c_int], ctypes.c_char_p),
}


class Hip:
    """Thin ctypes wrapper.  `load_only=True` dlopens and checks symbols without making
    a single HIP call -- safe to run in --selftest, since it never touches /dev/kfd."""

    def __init__(self, load_only: bool = False):
        try:
            self.lib = ctypes.CDLL(LIBHIP)
        except OSError as exc:
            raise ProbeError(f"cannot dlopen {LIBHIP}: {exc}") from exc
        missing = []
        for name, (argtypes, restype) in _SIGS.items():
            try:
                fn = getattr(self.lib, name)
            except AttributeError:
                missing.append(name)
                continue
            fn.argtypes = argtypes
            fn.restype = restype
        if missing:
            raise ProbeError(f"{LIBHIP} is missing required symbols: {missing}")
        self.load_only = load_only

    def err(self, rc: int) -> str:
        try:
            s = self.lib.hipGetErrorString(rc)
            return s.decode() if s else f"hipError {rc}"
        except Exception:  # pragma: no cover
            return f"hipError {rc}"

    def ck(self, rc: int, what: str) -> None:
        if rc != 0:
            raise ProbeError(f"{what} -> hipError {rc} ({self.err(rc)})")

    # -- convenience -------------------------------------------------------
    def device_count(self) -> int:
        n = ctypes.c_int(0)
        self.ck(self.lib.hipGetDeviceCount(ctypes.byref(n)), "hipGetDeviceCount")
        return n.value

    def device_identity(self, ordinal: int) -> dict:
        name = None
        pci = None
        buf = ctypes.create_string_buffer(256)
        dev = ctypes.c_int(0)
        if self.lib.hipDeviceGet(ctypes.byref(dev), ordinal) == 0:
            if self.lib.hipDeviceGetName(buf, 256, dev.value) == 0:
                name = buf.value.decode(errors="replace")
        buf2 = ctypes.create_string_buffer(64)
        if self.lib.hipDeviceGetPCIBusId(buf2, 64, ordinal) == 0:
            pci = buf2.value.decode(errors="replace")
        free = ctypes.c_size_t(0)
        total = ctypes.c_size_t(0)
        vram = None
        if self.lib.hipMemGetInfo(ctypes.byref(free), ctypes.byref(total)) == 0:
            vram = {"free_bytes": free.value, "total_bytes": total.value}
        return {"hip_ordinal": ordinal, "name": name, "pci_bus_id": pci, "vram": vram}

    def vram(self) -> dict:
        free = ctypes.c_size_t(0)
        total = ctypes.c_size_t(0)
        if self.lib.hipMemGetInfo(ctypes.byref(free), ctypes.byref(total)) != 0:
            return {"free_bytes": None, "total_bytes": None}
        return {"free_bytes": free.value, "total_bytes": total.value}


def host_prop() -> hipMemAllocationProp:
    p = hipMemAllocationProp()
    ctypes.memset(ctypes.byref(p), 0, ctypes.sizeof(p))
    p.type = HIP_MEM_ALLOCATION_TYPE_PINNED
    p.location.type = HIP_MEM_LOCATION_TYPE_HOST
    p.location.id = 0
    return p


def rw_desc(device: int) -> hipMemAccessDesc:
    d = hipMemAccessDesc()
    ctypes.memset(ctypes.byref(d), 0, ctypes.sizeof(d))
    d.location.type = HIP_MEM_LOCATION_TYPE_DEVICE
    d.location.id = device
    d.flags = HIP_MEM_ACCESS_FLAGS_PROT_READWRITE
    return d


def _sysfs_card_for_pci(pci_bus_id):
    """Resolve a HIP PCI bus id ('0000:03:00.0') to its sysfs DRM card, so every timing
    records WHICH PHYSICAL CARD it came from (the two cards are an RX 9070 XT and an
    RX 9070; mismatched-card timings have burned this team before)."""
    if not pci_bus_id:
        return None
    want = pci_bus_id.strip().lower()
    for c in read_gpu_vram_sysfs():
        slot = (c.get("pci_slot") or "").strip().lower()
        if slot and slot == want:
            return c
    return None


def chunk_fingerprint(rank: int, idx: int) -> int:
    """Unique per (rank, chunk).  Rank is folded in so that a cross-PROCESS physical-page
    alias -- two ranks handed the same pages -- is detectable, not just an intra-rank one."""
    return 0xA5A50000 | ((rank & 0xF) << 12) | (idx & 0x0FFF)


def verify_offsets(chunk_bytes: int) -> list:
    """Head, tail, and page-aligned interior samples.  Head+tail alone cannot see a chunk
    whose middle pages were mapped somewhere else; on this box the driver returns
    hipSuccess even when the page table is wrong, so sampling breadth is the only defence.
    Each probe is 16 B, so ten of them cost ~0.1 ms per chunk."""
    offs = [0]
    for k in range(1, 8):
        o = (chunk_bytes * k // 8) & ~(PAGE - 1)
        if 0 < o <= chunk_bytes - 16:
            offs.append(o)
    offs.append(chunk_bytes - 16)
    return sorted(set(offs))


# ===========================================================================
# Rank worker -- one process per rank.  Speaks newline-delimited JSON on
# stdin/stdout; all human output goes to stderr.
# ===========================================================================

class RankWorker:
    def __init__(self, rank: int, device: int, chunk_bytes: int, reserve_bytes: int):
        self.rank = rank
        self.device = device
        self.chunk_bytes = chunk_bytes
        self.reserve_bytes = reserve_bytes
        self.hip = Hip()
        self.va_base = None
        self.va_size = 0
        self.handles = []          # list[ctypes.c_void_p], index == chunk index
        self.chunks = []           # list[dict] per-chunk records
        self.desc = rw_desc(device)
        self.granularity = None
        self.identity = None
        self.vram_at_start = None
        self.sysfs_card = None
        self.voffsets = verify_offsets(chunk_bytes)

    # -- lifecycle ---------------------------------------------------------
    def hello(self) -> dict:
        h = self.hip
        n = h.device_count()
        if self.device >= n:
            raise ProbeError(
                f"rank {self.rank} wants HIP device {self.device} but hipGetDeviceCount()=={n}. "
                f"ROCR_VISIBLE_DEVICES={os.environ.get('ROCR_VISIBLE_DEVICES')!r}")
        h.ck(h.lib.hipSetDevice(self.device), f"hipSetDevice({self.device})")
        self.identity = h.device_identity(self.device)
        self.vram_at_start = h.vram()

        # --- WHICH PHYSICAL CARD, and is it even a discrete one? -------------
        # ROCR_VISIBLE_DEVICES=0,1 is *supposed* to exclude the Ryzen iGPU (ROCm dev 2,
        # ~47 GB of GTT).  If enumeration ever shifts, this probe would run on an APU where
        # "host-located" and "device" memory are literally the same DRAM -- the VRAM check
        # below would pass vacuously and the capacity number would be meaningless.  Assert
        # on the DEVICE ITSELF, not on the env var we set.
        self.sysfs_card = _sysfs_card_for_pci(self.identity.get("pci_bus_id"))
        tot = (self.vram_at_start or {}).get("total_bytes")
        if tot is None or tot < DISCRETE_VRAM_MIN_BYTES:
            raise ProbeError(
                f"rank {self.rank}: HIP device {self.device} "
                f"(pci={self.identity.get('pci_bus_id')!r} name={self.identity.get('name')!r}) "
                f"reports total VRAM {tot} B, below the {DISCRETE_VRAM_MIN_BYTES} B discrete "
                f"threshold. This is not one of the two gfx1201 cards -- refusing to measure "
                f"host-located capacity against an integrated GPU.")
        if self.sysfs_card is not None and self.sysfs_card.get("discrete") is False:
            raise ProbeError(
                f"rank {self.rank}: HIP device {self.device} resolves to sysfs "
                f"{self.sysfs_card['sysfs']} which is NOT discrete "
                f"(vram_total={self.sysfs_card.get('vram_total_bytes')}).")

        gran = ctypes.c_size_t(0)
        prop = host_prop()
        h.ck(h.lib.hipMemGetAllocationGranularity(
            ctypes.byref(gran), ctypes.byref(prop), 0),
            "hipMemGetAllocationGranularity(host, MINIMUM)")
        self.granularity = gran.value
        if self.granularity == 0 or self.chunk_bytes % self.granularity != 0:
            raise ProbeError(
                f"chunk_bytes={self.chunk_bytes} is not a multiple of the measured host "
                f"allocation granularity {self.granularity}")

        va = ctypes.c_void_p()
        align = max(2 * MIB, self.granularity)
        h.ck(h.lib.hipMemAddressReserve(
            ctypes.byref(va), self.reserve_bytes, align, None, 0),
            f"hipMemAddressReserve({self.reserve_bytes})")
        if not va.value:
            raise ProbeError("hipMemAddressReserve returned hipSuccess with a NULL base")
        self.va_base = va.value
        self.va_size = self.reserve_bytes
        return {
            "ev": "ready", "rank": self.rank, "device": self.device,
            "identity": self.identity, "vram_at_start": self.vram_at_start,
            "sysfs_card": self.sysfs_card,
            "host_granularity_bytes": self.granularity,
            "va_base": self.va_base, "va_reserved_bytes": self.va_size,
            "pid": os.getpid(),
        }

    # -- one chunk ---------------------------------------------------------
    def _mem_avail_kib(self):
        try:
            return read_meminfo().get("MemAvailable")
        except ProbeError:
            return None

    def _verify_chunk(self, idx: int) -> dict:
        """Read the fingerprint back at head, tail and page-aligned interior offsets.

        The driver on this box returns hipSuccess while serving a STALE physical page, so
        `rc == 0` proves nothing; only the DATA does.  Because each chunk carries a
        (rank, idx)-unique word, a read-back that returns some *other* chunk's fingerprint
        identifies the aliasing partner instead of just saying "wrong"."""
        h = self.hip
        buf = (ctypes.c_uint32 * 4)()
        want = chunk_fingerprint(self.rank, idx)
        base = self.va_base + idx * self.chunk_bytes
        ok = True
        detail = []
        for boff in self.voffsets:
            rc = h.lib.hipMemcpy(ctypes.cast(buf, ctypes.c_void_p),
                                 ctypes.c_void_p(base + boff), 16,
                                 HIP_MEMCPY_DEVICE_TO_HOST)
            if rc != 0:
                ok = False
                detail.append({"offset": boff, "hip_rc": rc, "hip_err": h.err(rc)})
                continue
            vals = [buf[i] for i in range(4)]
            bad = [v for v in vals if v != want]
            if bad:
                # Which chunk did we actually get?  A recognisable foreign fingerprint is
                # a page-table ALIAS; anything else is garbage.
                alias = None
                v0 = vals[0]
                if (v0 & 0xFFFF0000) == 0xA5A50000:
                    alias = {"rank": (v0 >> 12) & 0xF, "idx": v0 & 0x0FFF}
                ok = False
                detail.append({"offset": boff, "expected": want, "got": vals,
                               "looks_like_chunk": alias})
        return {"ok": ok, "detail": detail or None, "expected": want,
                "offsets_probed": len(self.voffsets)}

    def resweep(self) -> dict:
        """Re-verify EVERY committed chunk at peak.

        Verifying a chunk only right after writing it cannot detect later aliasing: if
        chunk 60's mapping silently reuses chunk 3's physical pages, chunk 60 verifies
        clean (we just wrote there) and chunk 3 is now corrupt and unexamined.  That is
        exactly how a '68 GiB committed' number becomes fiction.  This sweep is the check
        that makes the capacity claim mean something."""
        t0 = time.perf_counter()
        failures = []
        for idx in range(len(self.handles)):
            r = self._verify_chunk(idx)
            if not r["ok"]:
                failures.append({"rank": self.rank, "idx": idx, "detail": r["detail"]})
        t1 = time.perf_counter()
        return {"ev": "resweep", "rank": self.rank,
                "chunks_reverified": len(self.handles),
                "offsets_per_chunk": len(self.voffsets),
                "failures": len(failures), "failure_detail": failures[:8],
                "t_resweep_s": t1 - t0}

    def add_chunk(self, attrib: bool) -> dict:
        h = self.hip
        idx = len(self.handles)
        off = idx * self.chunk_bytes
        if off + self.chunk_bytes > self.va_size:
            raise ProbeError(f"rank {self.rank}: chunk {idx} would exceed the reservation "
                             f"({self.va_size} B)")
        addr = ctypes.c_void_p(self.va_base + off)
        rec = {"idx": idx, "offset": off, "attrib": attrib}

        if attrib:
            rec["mem_avail_kib_before"] = self._mem_avail_kib()

        handle = ctypes.c_void_p()
        t0 = time.perf_counter()
        rc = h.lib.hipMemCreate(ctypes.byref(handle), self.chunk_bytes,
                                ctypes.byref(host_prop()), 0)
        t1 = time.perf_counter()
        if rc != 0:
            rec.update({"ok": False, "failed_at": "hipMemCreate", "hip_rc": rc,
                        "hip_err": h.err(rc), "t_create_s": t1 - t0})
            return rec
        rec["t_create_s"] = t1 - t0
        if attrib:
            rec["mem_avail_kib_after_create"] = self._mem_avail_kib()

        t2 = time.perf_counter()
        rc = h.lib.hipMemMap(addr, self.chunk_bytes, 0, handle, 0)
        t3 = time.perf_counter()
        if rc != 0:
            h.lib.hipMemRelease(handle)
            rec.update({"ok": False, "failed_at": "hipMemMap", "hip_rc": rc,
                        "hip_err": h.err(rc), "t_map_s": t3 - t2})
            return rec
        rec["t_map_s"] = t3 - t2

        t4 = time.perf_counter()
        rc = h.lib.hipMemSetAccess(addr, self.chunk_bytes, ctypes.byref(self.desc), 1)
        t5 = time.perf_counter()
        if rc != 0:
            h.lib.hipMemUnmap(addr, self.chunk_bytes)
            h.lib.hipMemRelease(handle)
            rec.update({"ok": False, "failed_at": "hipMemSetAccess", "hip_rc": rc,
                        "hip_err": h.err(rc), "t_setaccess_s": t5 - t4})
            return rec
        rec["t_setaccess_s"] = t5 - t4
        if attrib:
            rec["mem_avail_kib_after_map"] = self._mem_avail_kib()

        # The handle is now ours; record it so teardown always frees it even if the
        # touch below fails.
        self.handles.append(handle)

        # First touch THROUGH THE DEVICE POINTER (never a CPU store: plan sec.5.2).
        # This is what forces commit -- without it a lazy hipMemCreate would let us
        # report capacity we never actually took.
        pattern = chunk_fingerprint(self.rank, idx)      # compared unsigned on read-back
        pattern_i32 = ctypes.c_int32(pattern).value      # hipMemsetD32 takes a signed int
        t6 = time.perf_counter()
        rc = h.lib.hipMemsetD32(addr, pattern_i32, self.chunk_bytes // 4)
        if rc == 0:
            rc = h.lib.hipDeviceSynchronize()            # fence: the memset is async
        t7 = time.perf_counter()
        if rc != 0:
            rec.update({"ok": False, "failed_at": "hipMemsetD32/sync", "hip_rc": rc,
                        "hip_err": h.err(rc), "t_touch_s": t7 - t6})
            return rec
        rec["t_touch_s"] = t7 - t6
        # NOT an allocation rate: this is a DEVICE-ISSUED WRITE over PCIe into the freshly
        # mapped host pages, at 512 MiB (8x the 64 MB MALL, so nothing is cache-resident).
        # It doubles as the host-locatedness check -- host pages must land in the PCIe band
        # (~10-30 GB/s), never at HBM speed.
        rec["touch_gb_s"] = self.chunk_bytes / (t7 - t6) / 1e9 if t7 > t6 else None
        if attrib:
            rec["mem_avail_kib_after_touch"] = self._mem_avail_kib()

        t8 = time.perf_counter()
        v = self._verify_chunk(idx)
        t9 = time.perf_counter()
        rec["t_verify_s"] = t9 - t8
        rec["verify_ok"] = v["ok"]
        rec["verify_detail"] = v["detail"]
        rec["verify_offsets_probed"] = v["offsets_probed"]
        rec["ok"] = v["ok"]
        if not v["ok"]:
            rec["failed_at"] = "readback_verify"
        # Sum of the TIMED stages, not wall t9-t0: for attribution chunks the wall includes
        # two or three /proc/meminfo reads, which would otherwise inflate 1-in-16 chunks'
        # "end to end" cost and poison the min/max of the reported rate.
        rec["t_total_s"] = (rec["t_create_s"] + rec["t_map_s"] + rec["t_setaccess_s"]
                            + rec["t_touch_s"] + rec["t_verify_s"])
        rec["t_wall_incl_sampling_s"] = t9 - t0
        rec["end_to_end_gb_s"] = (self.chunk_bytes / rec["t_total_s"] / 1e9
                                  if rec["t_total_s"] > 0 else None)
        return rec

    def step(self, n: int) -> dict:
        added = []
        fatal = None
        for _ in range(n):
            idx = len(self.handles)
            attrib = idx < 4 or idx % 16 == 0
            rec = self.add_chunk(attrib)
            added.append(rec)
            self.chunks.append(rec)
            if not rec.get("ok"):
                fatal = rec
                break
        return {
            "ev": "step", "rank": self.rank, "added": added,
            "committed_chunks": len(self.handles),
            "committed_bytes": len(self.handles) * self.chunk_bytes,
            "fatal": fatal,
            "vram": self.hip.vram(),
        }

    # -- rate bench --------------------------------------------------------
    def ratebench(self, arm: str, chunk_bytes: int, total_bytes: int, reps: int) -> dict:
        """Steady-state allocation rate.  Rep 0 is a warmup and is EXCLUDED from the
        summary (it carries lazy HIP init -- the 200 MB/s artifact in the plan)."""
        h = self.hip
        if arm not in ("memcreate_host", "hipHostMalloc"):
            raise ProbeError(f"unknown ratebench arm {arm!r}")
        nchunks = max(1, total_bytes // chunk_bytes)
        samples = []
        verify_failures = []
        arm_error = None

        # ONE reservation covering EVERY rep, so each rep maps a VA window that has never
        # been mapped before.  The previous shape reserved/freed per rep, which hands the
        # allocator the chance to return the same VA -- i.e. exactly the
        # `unmap -> map at an already-used VA` pattern this box is documented to serve
        # STALE PHYSICAL PAGES for, with every call returning hipSuccess.  A rep served a
        # stale page does far less work and would report an inflated allocation rate: the
        # precise way this bench could produce a confident wrong "34 GB boots in N s".
        va = None
        total_reserve = 0
        if arm == "memcreate_host":
            total_reserve = nchunks * chunk_bytes * reps
            va = ctypes.c_void_p()
            h.ck(h.lib.hipMemAddressReserve(
                ctypes.byref(va), total_reserve,
                max(2 * MIB, self.granularity or 2 * MIB), None, 0),
                f"ratebench hipMemAddressReserve({total_reserve})")

        try:
            for rep in range(reps):
                if arm == "memcreate_host":
                    handles = []
                    rep_base = va.value + rep * nchunks * chunk_bytes   # virgin VA window
                    t_create = t_map = t_touch = 0.0
                    try:
                        for i in range(nchunks):
                            addr = ctypes.c_void_p(rep_base + i * chunk_bytes)
                            hh = ctypes.c_void_p()
                            a = time.perf_counter()
                            h.ck(h.lib.hipMemCreate(ctypes.byref(hh), chunk_bytes,
                                                    ctypes.byref(host_prop()), 0),
                                 "ratebench hipMemCreate")
                            b = time.perf_counter()
                            handles.append((addr, hh))
                            h.ck(h.lib.hipMemMap(addr, chunk_bytes, 0, hh, 0),
                                 "ratebench hipMemMap")
                            h.ck(h.lib.hipMemSetAccess(addr, chunk_bytes,
                                                       ctypes.byref(self.desc), 1),
                                 "ratebench hipMemSetAccess")
                            c = time.perf_counter()
                            word = ctypes.c_int32(0x5A5A0000 | ((rep & 0xF) << 12)
                                                  | (i & 0x0FFF)).value
                            h.ck(h.lib.hipMemsetD32(addr, word, chunk_bytes // 4),
                                 "ratebench hipMemsetD32")
                            h.ck(h.lib.hipDeviceSynchronize(), "ratebench sync")
                            d = time.perf_counter()
                            t_create += b - a
                            t_map += c - b
                            t_touch += d - c
                        # Data check, OUTSIDE the timed region.  Without it the arm cannot
                        # tell "allocated fast" from "did not allocate at all".
                        buf = (ctypes.c_uint32 * 4)()
                        probes = verify_offsets(chunk_bytes)
                        for i in range(nchunks):
                            want = (0x5A5A0000 | ((rep & 0xF) << 12) | (i & 0x0FFF)) & 0xFFFFFFFF
                            for boff in probes:
                                rc = h.lib.hipMemcpy(
                                    ctypes.cast(buf, ctypes.c_void_p),
                                    ctypes.c_void_p(rep_base + i * chunk_bytes + boff), 16,
                                    HIP_MEMCPY_DEVICE_TO_HOST)
                                if rc != 0 or buf[0] != want:
                                    verify_failures.append(
                                        {"rep": rep, "chunk": i, "offset": boff,
                                         "hip_rc": rc, "expected": want,
                                         "got": None if rc != 0 else buf[0]})
                    finally:
                        for addr, hh in handles:
                            h.lib.hipMemUnmap(addr, chunk_bytes)
                            h.lib.hipMemRelease(hh)
                    nbytes = nchunks * chunk_bytes
                    samples.append({
                        "rep": rep, "warmup": rep == 0, "bytes": nbytes,
                        "create_gb_s": nbytes / t_create / 1e9 if t_create > 0 else None,
                        "map_gb_s": nbytes / t_map / 1e9 if t_map > 0 else None,
                        "touch_gb_s": nbytes / t_touch / 1e9 if t_touch > 0 else None,
                        "end_to_end_gb_s": nbytes / (t_create + t_map + t_touch) / 1e9,
                    })
                else:
                    # The recorded reference arm: 4.9-5.6 GB/s in 4 GiB chunks.
                    ptrs = []
                    t_alloc = 0.0
                    t_touch = 0.0
                    try:
                        for _i in range(nchunks):
                            p = ctypes.c_void_p()
                            a = time.perf_counter()
                            h.ck(h.lib.hipHostMalloc(ctypes.byref(p), chunk_bytes, 0),
                                 "ratebench hipHostMalloc")
                            b = time.perf_counter()
                            ptrs.append(p)
                            ctypes.memset(p, 0x5A, chunk_bytes)
                            c = time.perf_counter()
                            t_alloc += b - a
                            t_touch += c - b
                    finally:
                        for p in ptrs:
                            h.lib.hipHostFree(p)
                    nbytes = nchunks * chunk_bytes
                    samples.append({
                        "rep": rep, "warmup": rep == 0, "bytes": nbytes,
                        "alloc_gb_s": nbytes / t_alloc / 1e9 if t_alloc > 0 else None,
                        "cpu_touch_gb_s": nbytes / t_touch / 1e9 if t_touch > 0 else None,
                        "end_to_end_gb_s": nbytes / (t_alloc + t_touch) / 1e9,
                    })
        except ProbeError as exc:
            # An arm that OOMs must not destroy the whole probe: the climb is the primary
            # measurement and it has not run yet.  Record the failure, keep what we have.
            arm_error = f"{type(exc).__name__}: {exc}"
        finally:
            if va is not None:
                h.lib.hipMemAddressFree(va, total_reserve)

        steady = [s for s in samples if not s["warmup"]]
        keys = sorted({k for s in samples for k in s if k.endswith("_gb_s")})
        return {
            "ev": "ratebench", "rank": self.rank, "arm": arm,
            "chunk_bytes": chunk_bytes, "chunks_per_rep": nchunks,
            "reps": reps, "reps_completed": len(samples), "warmup_reps_discarded": 1,
            "error": arm_error,
            "fresh_va_per_rep": arm == "memcreate_host",
            "verify_failures": len(verify_failures),
            "verify_failure_detail": verify_failures[:8],
            "trustworthy": arm_error is None and not verify_failures and len(steady) >= 1,
            "samples": samples,
            "summary": {k: summarize([s.get(k) for s in steady]) for k in keys},
        }

    # -- teardown ----------------------------------------------------------
    def release(self) -> dict:
        h = self.hip
        t0 = time.perf_counter()
        errs = []
        for i, handle in enumerate(self.handles):
            addr = ctypes.c_void_p(self.va_base + i * self.chunk_bytes)
            rc = h.lib.hipMemUnmap(addr, self.chunk_bytes)
            if rc != 0:
                errs.append({"i": i, "op": "unmap", "rc": rc})
            rc = h.lib.hipMemRelease(handle)
            if rc != 0:
                errs.append({"i": i, "op": "release", "rc": rc})
        n = len(self.handles)
        self.handles = []
        t1 = time.perf_counter()
        if self.va_base is not None:
            h.lib.hipMemAddressFree(ctypes.c_void_p(self.va_base), self.va_size)
            self.va_base = None
        return {"ev": "released", "rank": self.rank, "chunks_released": n,
                "t_release_s": t1 - t0,
                "release_gb_s": (n * self.chunk_bytes) / (t1 - t0) / 1e9 if t1 > t0 and n else None,
                "errors": errs or None, "vram": h.vram()}


def worker_main(argv) -> int:
    ap = argparse.ArgumentParser(prog="p3-worker")
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--device", type=int, required=True)
    ap.add_argument("--chunk-bytes", type=int, required=True)
    ap.add_argument("--reserve-bytes", type=int, required=True)
    a = ap.parse_args(argv)

    def emit(obj):
        sys.stdout.write(json.dumps(obj) + "\n")
        sys.stdout.flush()

    w = None
    try:
        w = RankWorker(a.rank, a.device, a.chunk_bytes, a.reserve_bytes)
        emit(w.hello())
    except Exception as exc:
        emit({"ev": "error", "rank": a.rank, "where": "init",
              "msg": f"{type(exc).__name__}: {exc}"})
        return 3

    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            msg = json.loads(line)
            cmd = msg.get("cmd")
            if cmd == "step":
                emit(w.step(int(msg.get("n", 1))))
            elif cmd == "ratebench":
                emit(w.ratebench(msg["arm"], int(msg["chunk_bytes"]),
                                 int(msg["total_bytes"]), int(msg["reps"])))
            elif cmd == "resweep":
                emit(w.resweep())
            elif cmd == "release":
                emit(w.release())
            elif cmd == "quit":
                if w.handles:
                    w.release()
                emit({"ev": "bye", "rank": a.rank})
                return 0
            else:
                emit({"ev": "error", "rank": a.rank, "where": "dispatch",
                      "msg": f"unknown cmd {cmd!r}"})
                return 4
    except Exception as exc:
        emit({"ev": "error", "rank": a.rank, "where": "loop",
              "msg": f"{type(exc).__name__}: {exc}"})
        try:
            if w is not None and w.handles:
                w.release()
        except Exception:
            pass
        return 5
    return 0


# ===========================================================================
# Parent side
# ===========================================================================

def summarize(vals) -> dict:
    """Median + spread.  Never invents a number: n==0 -> all None."""
    vals = [v for v in vals if v is not None]
    if not vals:
        return {"n": 0, "median": None, "mean": None, "min": None, "max": None,
                "p25": None, "p75": None, "stdev": None}
    s = sorted(vals)
    def _q(q):
        if len(s) == 1:
            return s[0]
        pos = q * (len(s) - 1)
        lo = int(pos)
        hi = min(lo + 1, len(s) - 1)
        return s[lo] + (s[hi] - s[lo]) * (pos - lo)
    return {
        "n": len(s), "median": statistics.median(s), "mean": statistics.fmean(s),
        "min": s[0], "max": s[-1], "p25": _q(0.25), "p75": _q(0.75),
        "stdev": statistics.stdev(s) if len(s) > 1 else 0.0,
    }


class LineReader:
    """Non-blocking newline-delimited JSON reader over a raw fd."""

    def __init__(self, fd: int, name: str):
        self.fd = fd
        self.name = name
        self.buf = b""
        self.eof = False

    def readline(self, deadline: float):
        while True:
            nl = self.buf.find(b"\n")
            if nl >= 0:
                line, self.buf = self.buf[:nl], self.buf[nl + 1:]
                return line.decode(errors="replace")
            if self.eof:
                return None
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ProbeError(f"timeout waiting for a reply from {self.name}")
            r, _, _ = select.select([self.fd], [], [], min(remaining, 1.0))
            if not r:
                continue
            try:
                data = os.read(self.fd, 1 << 16)
            except OSError as exc:
                if exc.errno == errno.EINTR:
                    continue
                raise
            if not data:
                self.eof = True
            else:
                self.buf += data


class Worker:
    def __init__(self, rank: int, device: int, chunk_bytes: int, reserve_bytes: int,
                 timeout_s: float):
        self.rank = rank
        self.device = device
        self.timeout_s = timeout_s
        self.proc = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--_worker",
             "--rank", str(rank), "--device", str(device),
             "--chunk-bytes", str(chunk_bytes), "--reserve-bytes", str(reserve_bytes)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None,
            env=dict(os.environ),
        )
        self.reader = LineReader(self.proc.stdout.fileno(), f"rank{rank}(pid {self.proc.pid})")
        self.ready = None

    def send(self, obj: dict) -> None:
        if self.proc.poll() is not None:
            raise ProbeError(f"rank {self.rank} worker died (rc={self.proc.returncode}) "
                             f"before command {obj.get('cmd')!r}")
        try:
            self.proc.stdin.write((json.dumps(obj) + "\n").encode())
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise ProbeError(f"rank {self.rank} worker pipe broke: {exc}") from exc

    def recv(self, timeout_s=None) -> dict:
        deadline = time.monotonic() + (timeout_s or self.timeout_s)
        line = self.reader.readline(deadline)
        if line is None:
            rc = self.proc.poll()
            raise ProbeError(f"rank {self.rank} worker closed its pipe (rc={rc})")
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ProbeError(f"rank {self.rank} emitted non-JSON: {line[:200]!r}") from exc
        if msg.get("ev") == "error":
            raise ProbeError(f"rank {self.rank} worker error at {msg.get('where')}: "
                             f"{msg.get('msg')}")
        return msg

    def ask(self, obj: dict, timeout_s=None) -> dict:
        self.send(obj)
        return self.recv(timeout_s)

    def kill(self) -> None:
        self._closed = True
        if self.proc.poll() is None:
            try:
                self.proc.send_signal(signal.SIGTERM)
                self.proc.wait(timeout=30)
            except (subprocess.TimeoutExpired, OSError):
                try:
                    self.proc.kill()
                    self.proc.wait(timeout=15)
                except (subprocess.TimeoutExpired, OSError):
                    pass
        for f in (self.proc.stdin, self.proc.stdout):
            try:
                if f is not None:
                    f.close()
            except OSError:
                pass


def _pswpout_rate(prev: dict, cur: dict):
    """MB/s of pages swapped OUT between two box_state samples.  None if unreadable."""
    a = prev["vmstat"].get("pswpout")
    b = cur["vmstat"].get("pswpout")
    dt = cur["t_mono"] - prev["t_mono"]
    if a is None or b is None or dt <= 0:
        return None
    return (b - a) * PAGE / dt / 1e6


def measure_swap_baseline(window_s: float) -> dict:
    """This box swaps at idle (pswpout ~4e8 at start), so 'any swapping' is not a signal.
    Measure the resting rate and gate against it."""
    a = box_state("swap_baseline_a", skip_smi=True)
    time.sleep(window_s)
    b = box_state("swap_baseline_b", skip_smi=True)
    return {"window_s": window_s, "pswpout_mb_s": _pswpout_rate(a, b),
            "pswpin_mb_s": (
                None if a["vmstat"].get("pswpin") is None or b["vmstat"].get("pswpin") is None
                else (b["vmstat"]["pswpin"] - a["vmstat"]["pswpin"]) * PAGE
                / max(b["t_mono"] - a["t_mono"], 1e-9) / 1e6)}


def wait_for_headroom(baseline_avail_kib, tol_gib: float, timeout_s: float) -> dict:
    """Wait for MemAvailable to come back to within `tol_gib` of the baseline.

    In --mode both, rank1 climbs to tens of GiB and releases; if the kernel has not
    actually given the memory back, rank2 measures a DEPRESSED box and its shortfall gets
    read as 'two ranks do not fit' when it is really rank1's residue.  That is a wrong
    design fork from a real measurement, so the wait -- and the residual deficit when the
    wait times out -- are both recorded."""
    t0 = time.monotonic()
    last = None
    while True:
        last = read_meminfo().get("MemAvailable")
        if baseline_avail_kib is None or last is None:
            break
        deficit_gib = (baseline_avail_kib - last) * 1024 / GIB
        if deficit_gib <= tol_gib or time.monotonic() - t0 > timeout_s:
            break
        time.sleep(1.0)
    deficit = (None if (baseline_avail_kib is None or last is None)
               else round((baseline_avail_kib - last) * 1024 / GIB, 3))
    return {
        "waited_s": round(time.monotonic() - t0, 1),
        "timeout_s": timeout_s,
        "tolerance_gib": tol_gib,
        "baseline_mem_available_gib": (None if baseline_avail_kib is None
                                       else baseline_avail_kib * 1024 / GIB),
        "mem_available_at_phase_start_gib": None if last is None else last * 1024 / GIB,
        "deficit_vs_baseline_gib": deficit,
        "recovered": None if deficit is None else bool(deficit <= tol_gib),
    }


def run_phase(name: str, ranks: int, args, swap_baseline: dict,
              ratebench: bool, phases: dict = None, baseline: dict = None,
              progress_path: Path = None) -> dict:
    """One capacity climb: `ranks` workers grow in lockstep until target or a stop.

    `phases` is mutated IN PLACE the moment the phase dict exists, so an abort, a worker
    timeout or a SIGTERM half an hour into a climb still leaves the partial measurement in
    the emitted JSON.  'A partial climb is still the answer' only holds if the answer
    survives the abort."""
    chunk = args.chunk_mib * MIB
    target_per_rank = int(round(args.per_rank_gib * GIB))
    nchunks_target = target_per_rank // chunk
    if nchunks_target < 1:
        raise ProbeError("--per-rank-gib is smaller than one chunk")
    reserve = (nchunks_target + 2) * chunk

    _log(f"phase {name}: ranks={ranks} target/rank={target_per_rank/GIB:.2f} GiB "
         f"chunk={args.chunk_mib} MiB -> {nchunks_target} chunks/rank")

    phase = {
        "name": name,
        "ranks": ranks,
        "target_bytes_per_rank": target_per_rank,
        "target_bytes_total": target_per_rank * ranks,
        "chunk_bytes": chunk,
        "chunks_target_per_rank": nchunks_target,
        "va_reserved_bytes_per_rank": reserve,
        "chunks_per_round": args.chunks_per_round,
        "floor_bytes": int(args.floor_gib * GIB),
        "swap_baseline": swap_baseline,
        "workers": [],
        "rounds": [],
        "ratebench": None,
        "stop_reason": None,
        "stop_detail": None,
        "committed_bytes_total": 0,
        "committed_bytes_per_rank": [],
        "wall_s": None,
        "rate": {},
        "commit_attribution": None,
        "verify": {"chunks_verified": 0, "failures": 0, "failure_detail": []},
        "release": [],
        "recovery": None,
        "box_state_at_peak": None,
        "peak_resweep": None,
        "start_headroom": None,
    }
    # Register BEFORE any work, so a crash/abort mid-climb still reports what was measured.
    if phases is not None:
        phases[name] = phase

    base_avail = ((baseline or {}).get("meminfo_kib") or {}).get("MemAvailable")
    phase["start_headroom"] = wait_for_headroom(
        base_avail, args.recovery_tolerance_gib, args.recovery_timeout_s)
    if phase["start_headroom"].get("recovered") is False:
        _log(f"  WARNING phase {name} starts {phase['start_headroom']['deficit_vs_baseline_gib']} "
             f"GiB below the baseline headroom; its capacity result is a LOWER BOUND")

    workers = []
    try:
        for r in range(ranks):
            w = Worker(r, r, chunk, reserve, args.worker_timeout)
            workers.append(w)
        for w in workers:
            w.ready = w.recv(timeout_s=args.worker_timeout)
            if w.ready.get("ev") != "ready":
                raise ProbeError(f"rank {w.rank}: unexpected first message {w.ready}")
            phase["workers"].append(w.ready)
            _log(f"  rank{w.rank} ready on HIP dev {w.ready['device']} "
                 f"pci={w.ready['identity']['pci_bus_id']} "
                 f"name={w.ready['identity']['name']} "
                 f"gran={w.ready['host_granularity_bytes']}B")

        if ratebench:
            arms = []
            for arm, cb, tot in (
                ("memcreate_host", 512 * MIB, 4 * GIB),
                ("memcreate_host", 4 * GIB, 4 * GIB),
                ("hipHostMalloc", 4 * GIB, 4 * GIB),
            ):
                _log(f"  ratebench {arm} chunk={cb//MIB} MiB total={tot//GIB} GiB "
                     f"reps={args.rate_reps} (rep 0 discarded)")
                res = workers[0].ask({"cmd": "ratebench", "arm": arm, "chunk_bytes": cb,
                                      "total_bytes": tot, "reps": args.rate_reps},
                                     timeout_s=max(args.worker_timeout, 600))
                arms.append(res)
            phase["ratebench"] = arms

        prev_state = box_state("round_0", skip_smi=True)
        t_start = time.monotonic()
        committed = [0] * ranks
        stop = None
        rnd = 0
        peak_state = prev_state

        def _emit_round(row):
            phase["rounds"].append(row)
            if progress_path is None:
                return
            # Durable, append-only progress: a SIGKILL (OOM killer, operator) must not
            # take the whole climb with it.
            try:
                with open(progress_path, "a") as fh:
                    fh.write(json.dumps({"phase": name, **row}) + "\n")
                    fh.flush()
                    os.fsync(fh.fileno())
            except OSError:
                pass

        # Round 0 = the pre-climb reference row.  Without it the VRAM check would compare
        # round 1 (already N chunks in) against round N and miss anything the first round
        # cost, and the commit accounting would have no origin.
        _emit_round({
            "round": 0,
            "committed_bytes_total": 0,
            "committed_gib_total": 0.0,
            "committed_bytes_per_rank": [0] * ranks,
            "mem_available_kib": prev_state["meminfo_kib"].get("MemAvailable"),
            "mem_free_kib": prev_state["meminfo_kib"].get("MemFree"),
            "swap_free_kib": prev_state["meminfo_kib"].get("SwapFree"),
            "pswpout_mb_s": None,
            "zfs_arc_size_kib": prev_state["zfs_arc"].get("size_kib"),
            "discrete_vram_used_bytes": [c.get("vram_used_bytes")
                                         for c in prev_state["gpu_vram_sysfs"]
                                         if c.get("discrete")],
            "t_mono": prev_state["t_mono"],
        })

        while stop is None:
            remaining = min(nchunks_target - (committed[r] // chunk) for r in range(ranks))
            if remaining <= 0:
                stop = ("target_reached", None)
                break
            n = min(args.chunks_per_round, remaining)
            rnd += 1

            replies = []
            for w in workers:
                w.send({"cmd": "step", "n": n})
            for w in workers:
                replies.append(w.recv())

            for rep in replies:
                committed[rep["rank"]] = rep["committed_bytes"]
                for c in rep["added"]:
                    if "verify_ok" in c:
                        phase["verify"]["chunks_verified"] += 1
                        if not c["verify_ok"]:
                            phase["verify"]["failures"] += 1
                            phase["verify"]["failure_detail"].append(
                                {"rank": rep["rank"], "idx": c["idx"],
                                 "detail": c.get("verify_detail")})
                phase.setdefault("_chunk_records", {}).setdefault(rep["rank"], []).extend(rep["added"])

            st = box_state(f"round_{rnd}", skip_smi=True)
            avail_kib = st["meminfo_kib"].get("MemAvailable")
            swap_mb_s = _pswpout_rate(prev_state, st)
            total = sum(committed)
            row = {
                "round": rnd,
                "committed_bytes_total": total,
                "committed_gib_total": total / GIB,
                "committed_bytes_per_rank": list(committed),
                "mem_available_kib": avail_kib,
                "mem_free_kib": st["meminfo_kib"].get("MemFree"),
                "swap_free_kib": st["meminfo_kib"].get("SwapFree"),
                "pswpout_mb_s": swap_mb_s,
                "zfs_arc_size_kib": st["zfs_arc"].get("size_kib"),
                "discrete_vram_used_bytes": [c.get("vram_used_bytes")
                                             for c in st["gpu_vram_sysfs"]
                                             if c.get("discrete")],
                "t_mono": st["t_mono"],
            }
            _emit_round(row)
            prev_state = st
            peak_state = st

            # ORDER MATTERS.  A chunk that fails read-back is also marked `fatal`, so
            # checking `fatal` first would label the single most important outcome of this
            # probe -- the driver serving a wrong page while returning hipSuccess -- as a
            # generic "hip_error", and the hard invalidator would be read off the wrong
            # stop_reason.  Verify failures win.
            if phase["verify"]["failures"]:
                stop = ("readback_verify_failed", phase["verify"]["failure_detail"][:4])
                break
            fatal = next((rep["fatal"] for rep in replies if rep.get("fatal")), None)
            if fatal is not None:
                stop = ("hip_error", fatal)
                break
            if avail_kib is not None and avail_kib * 1024 < args.floor_gib * GIB:
                stop = ("mem_available_floor",
                        {"mem_available_gib": avail_kib * 1024 / GIB,
                         "floor_gib": args.floor_gib})
                break
            base = swap_baseline.get("pswpout_mb_s")
            limit = max((base or 0.0) * args.swap_slack, args.swap_limit_mb_s)
            if swap_mb_s is not None and swap_mb_s > limit:
                stop = ("swap_thrash", {"pswpout_mb_s": swap_mb_s, "limit_mb_s": limit,
                                        "baseline_mb_s": base})
                break

        phase["wall_s"] = time.monotonic() - t_start
        phase["stop_reason"], phase["stop_detail"] = stop
        phase["committed_bytes_per_rank"] = list(committed)
        phase["committed_bytes_total"] = sum(committed)
        phase["box_state_at_peak"] = box_state("peak", skip_smi=args.skip_rocm_smi)
        _log(f"  phase {name} stopped: {phase['stop_reason']} at "
             f"{phase['committed_bytes_total']/GIB:.2f} GiB total "
             f"({phase['committed_bytes_total']/ranks/GIB:.2f} GiB/rank) in "
             f"{phase['wall_s']:.1f} s")

        # --- THE CHECK THAT MAKES THE CAPACITY NUMBER MEAN ANYTHING ----------
        # Every chunk was verified right after it was written, which cannot detect a chunk
        # whose pages were LATER re-handed to a different chunk.  Re-read all of them, at
        # peak, still mapped, against their (rank, idx)-unique fingerprints.  If the arena
        # is really N GiB of distinct physical pages this is quiet; if the driver has been
        # aliasing, this is where "68 GiB committed" turns out to be fiction.
        _log(f"  re-verifying all {phase['committed_bytes_total'] // chunk} committed "
             f"chunks at peak (aliasing sweep)")
        sweeps = []
        for w in workers:
            sweeps.append(w.ask({"cmd": "resweep"}, timeout_s=max(args.worker_timeout, 600)))
        sweep_fail = sum(s.get("failures", 0) for s in sweeps)
        phase["peak_resweep"] = {
            "per_rank": sweeps,
            "chunks_reverified": sum(s.get("chunks_reverified", 0) for s in sweeps),
            "failures": sweep_fail,
            "note": ("Re-reads every still-mapped chunk at peak against its unique "
                     "fingerprint. A failure here means the arena aliased physical pages "
                     "and the committed-bytes figure is fiction."),
        }
        if sweep_fail:
            phase["verify"]["failures"] += sweep_fail
            for s in sweeps:
                phase["verify"]["failure_detail"].extend(s.get("failure_detail") or [])
            phase["stop_reason"] = "readback_verify_failed"
            phase["stop_detail"] = {"origin": "peak_resweep",
                                    "previous_stop_reason": stop[0],
                                    "failures": sweep_fail}
            _log(f"  *** PEAK RESWEEP FAILED: {sweep_fail} chunks no longer hold their "
                 f"own fingerprint. The capacity number is VOID.")

        _summarize_chunks(phase)
        phase["vram_check"] = _vram_check(phase)
        phase["host_locatedness"] = _host_locatedness(phase)
        phase["commit_accounting"] = _commit_accounting(phase)

        for w in workers:
            phase["release"].append(w.ask({"cmd": "release"},
                                          timeout_s=max(args.worker_timeout, 600)))
        for w in workers:
            w.ask({"cmd": "quit"}, timeout_s=120)
    finally:
        for w in workers:
            w.kill()

    time.sleep(args.settle_s)
    after = box_state("after_release", skip_smi=True)
    phase["recovery"] = {
        "settle_s": args.settle_s,
        "mem_available_kib": after["meminfo_kib"].get("MemAvailable"),
        "mem_free_kib": after["meminfo_kib"].get("MemFree"),
        "reclaimed_vs_peak_gib": (
            None if after["meminfo_kib"].get("MemAvailable") is None
            or phase["box_state_at_peak"]["meminfo_kib"].get("MemAvailable") is None
            else (after["meminfo_kib"]["MemAvailable"]
                  - phase["box_state_at_peak"]["meminfo_kib"]["MemAvailable"]) * 1024 / GIB),
    }
    return phase


def _vram_check(phase: dict) -> dict:
    """Host-located pages must cost ZERO device memory.  If VRAM GREW with the climb, the
    allocation is not landing where we think it is and every conclusion here is void.

    Two corrections over the naive form:
      * baseline is round 0 (pre-climb), not round 1 -- otherwise whatever the first round
        cost is invisible;
      * only POSITIVE growth counts.  The plan requires this probe run with an engine
        loaded, and an engine freeing VRAM mid-climb produces a large NEGATIVE delta.
        Scoring on |delta| would let that trip the hard invalidator and void a perfectly
        good run for a reason that has nothing to do with where our pages landed.
    """
    rows = phase.get("rounds") or []
    first = next((r for r in rows if r.get("discrete_vram_used_bytes")), None)
    last = next((r for r in reversed(rows) if r.get("discrete_vram_used_bytes")), None)
    if not first or not last or first is last:
        return {"status": "not_measured"}
    a, b = first["discrete_vram_used_bytes"], last["discrete_vram_used_bytes"]
    n = min(len(a), len(b))
    deltas = [(b[i] - a[i]) for i in range(n)
              if a[i] is not None and b[i] is not None]
    if not deltas:
        return {"status": "not_measured"}
    worst_growth = max(deltas)
    committed = phase.get("committed_bytes_total") or 0
    # Tolerance: page-table/bookkeeping for a multi-GiB arena is legitimately non-zero, but
    # it cannot be a meaningful FRACTION of the arena.  Scale, with an absolute floor for
    # ambient engine noise.
    tol = max(256 * MIB, int(0.02 * committed))
    return {
        "status": "measured",
        "baseline_round": first.get("round"),
        "peak_round": last.get("round"),
        "per_card_delta_bytes": deltas,
        "worst_growth_bytes": worst_growth,
        "worst_growth_mib": worst_growth / MIB,
        "worst_growth_frac_of_committed": (worst_growth / committed) if committed else None,
        "tolerance_bytes": tol,
        "committed_host_bytes": committed,
        "host_pages_cost_zero_vram": bool(worst_growth < tol),
        "note": ("Positive growth only. A VRAM increase tracking the climb would mean "
                 "hipMemCreate(location=Host) is not actually host-located, which "
                 "invalidates the whole probe. A negative delta is a co-resident engine "
                 "releasing memory and is NOT a failure."),
    }


# Anything faster than this is not crossing PCIe: it is device memory, or the write never
# happened.  Measured PCIe H2D on this box is 26.8-28.7 GB/s; HBM is ~640 GB/s.
HOST_LOCATED_MAX_GB_S = 100.0


def _host_locatedness(phase: dict) -> dict:
    """Independent evidence that the pages really are on the host.

    The sysfs VRAM check can only say "device memory did not grow"; it cannot distinguish
    host-located pages from pages that were never committed at all.  The first touch is a
    DEVICE-ISSUED write through the device VA over 512 MiB (8x the 64 MB MALL, so no part
    of it is cache-resident).  If those pages are host-located it must run at PCIe speed.
    A figure in the hundreds of GB/s means we are writing to VRAM and the probe is void."""
    s = ((phase.get("rate") or {}).get("first_touch_gb_s") or {})
    med = s.get("median")
    if med is None:
        return {"status": "not_measured"}
    return {
        "status": "measured",
        "first_touch_median_gb_s": med,
        "first_touch_p25_gb_s": s.get("p25"),
        "first_touch_p75_gb_s": s.get("p75"),
        "pcie_reference_gb_s": {"h2d_1gib_contiguous": 28.7, "h2d_2p8mib_granule": 26.8},
        "hbm_reference_gb_s": 640.0,
        "max_plausible_host_gb_s": HOST_LOCATED_MAX_GB_S,
        "consistent_with_host_located": bool(med < HOST_LOCATED_MAX_GB_S),
        "note": ("Device-issued write over the device VA at 512 MiB. Not an allocation "
                 "rate. Host-located pages cannot exceed the PCIe link; a median in the "
                 "hundreds of GB/s means the pages are device-located."),
    }


def _commit_accounting(phase: dict) -> dict:
    """Did the RAM actually go anywhere?

    The per-stage MemAvailable deltas answer 'eager or lazy'.  This answers the blunter
    question the hard invalidator needs: across the whole climb, did the box lose roughly
    `committed_bytes` of available memory?  Two confounders are corrected explicitly,
    because both RAISE MemAvailable while we are consuming it and would otherwise make a
    real commit look like nothing happened:
      * ZFS ARC shrinking under pressure (ARC is partly reclaimable slab, i.e. inside
        MemAvailable), and
      * the kernel swapping other things out.
    """
    rows = phase.get("rounds") or []
    have = [r for r in rows if r.get("mem_available_kib") is not None]
    if len(have) < 2:
        return {"status": "not_measured"}
    first, last = have[0], min(have, key=lambda r: r["mem_available_kib"])
    committed = phase.get("committed_bytes_total") or 0
    drop = (first["mem_available_kib"] - last["mem_available_kib"]) * 1024
    arc0, arc1 = first.get("zfs_arc_size_kib"), last.get("zfs_arc_size_kib")
    arc_shrink = max(0, (arc0 - arc1) * 1024) if (arc0 is not None and arc1 is not None) else 0
    sf0, sf1 = first.get("swap_free_kib"), last.get("swap_free_kib")
    swapped_out = max(0, (sf0 - sf1) * 1024) if (sf0 is not None and sf1 is not None) else 0
    accounted = drop + arc_shrink + swapped_out
    ratio = (accounted / committed) if committed else None
    return {
        "status": "measured",
        "committed_bytes": committed,
        "mem_available_drop_bytes": drop,
        "zfs_arc_shrink_bytes": arc_shrink,
        "swapped_out_bytes": swapped_out,
        "accounted_bytes": accounted,
        "accounted_frac_of_committed": ratio,
        "commit_landed": None if ratio is None else bool(ratio >= 0.5),
        "note": ("accounted = MemAvailable drop + ARC shrink + swap growth, all measured. "
                 "A ratio near 0 with a non-zero committed figure means the pages were "
                 "never really taken and any capacity claim from this run is fiction."),
    }


def _summarize_chunks(phase: dict) -> None:
    """Fold the per-chunk records into medians/spread, and attribute WHERE commit happened."""
    recs = phase.pop("_chunk_records", {})
    flat = [c for rank_recs in recs.values() for c in rank_recs]
    chunk = phase["chunk_bytes"]
    # Chunk 0 of each rank carries first-call costs; report it separately, exclude it
    # from the steady-state medians.
    warm = [c for c in flat if c["idx"] == 0]
    steady = [c for c in flat if c["idx"] > 0 and c.get("ok")]

    def rate(key):
        return summarize([chunk / c[key] / 1e9 for c in steady
                          if c.get(key) and c[key] > 0])

    phase["rate"] = {
        "units": "decimal GB/s (1e9 B/s), per-chunk, warmup chunk excluded",
        "note": ("hipMemCreate/Map/SetAccess are ALLOCATION rates. first_touch is a "
                 "device-issued PCIe WRITE bandwidth into the new host pages, not an "
                 "allocation rate -- see host_locatedness. end_to_end is the sum of the "
                 "timed stages, excluding the /proc sampling done on attribution chunks."),
        "chunk_bytes": chunk,
        "n_steady_chunks": len(steady),
        "hipMemCreate_gb_s": rate("t_create_s"),
        "hipMemMap_gb_s": rate("t_map_s"),
        "hipMemSetAccess_gb_s": rate("t_setaccess_s"),
        "first_touch_gb_s": rate("t_touch_s"),
        "end_to_end_gb_s": rate("t_total_s"),
        "per_chunk_seconds": {
            "hipMemCreate": summarize([c.get("t_create_s") for c in steady]),
            "hipMemMap": summarize([c.get("t_map_s") for c in steady]),
            "hipMemSetAccess": summarize([c.get("t_setaccess_s") for c in steady]),
            "first_touch": summarize([c.get("t_touch_s") for c in steady]),
            "readback_verify": summarize([c.get("t_verify_s") for c in steady]),
        },
        "warmup_chunks": warm,
    }
    if phase["wall_s"] and phase["committed_bytes_total"]:
        phase["rate"]["aggregate_wall_gb_s"] = (
            phase["committed_bytes_total"] / phase["wall_s"] / 1e9)
        phase["rate"]["aggregate_wall_note"] = (
            "includes per-round /proc sampling and lockstep sync; NOT a pure allocation rate")

    # Where does the memory actually get committed?  create, map, or first touch?
    att = [c for c in flat if c.get("attrib") and c.get("ok")
           and c.get("mem_avail_kib_before") is not None
           and c.get("mem_avail_kib_after_touch") is not None]
    if att:
        d_create = [(c["mem_avail_kib_before"] - c["mem_avail_kib_after_create"]) * 1024
                    for c in att if c.get("mem_avail_kib_after_create") is not None]
        d_map = [(c["mem_avail_kib_after_create"] - c["mem_avail_kib_after_map"]) * 1024
                 for c in att if c.get("mem_avail_kib_after_create") is not None
                 and c.get("mem_avail_kib_after_map") is not None]
        d_touch = [(c["mem_avail_kib_after_map"] - c["mem_avail_kib_after_touch"]) * 1024
                   for c in att if c.get("mem_avail_kib_after_map") is not None]
        m_create = summarize(d_create)
        m_map = summarize(d_map)
        m_touch = summarize(d_touch)
        half = 0.5 * chunk
        if (m_create.get("median") or 0) >= half:
            timing = "eager_at_hipMemCreate"
        elif (m_touch.get("median") or 0) >= half:
            timing = "lazy_until_first_touch"
        elif (m_map.get("median") or 0) >= half:
            timing = "at_hipMemMap_setAccess"
        else:
            timing = "unattributed"
        phase["commit_attribution"] = {
            "n_sampled_chunks": len(att),
            "chunk_bytes": chunk,
            "note": ("MemAvailable delta per phase.  If ~chunk_bytes lands on hipMemCreate "
                     "the allocation is eager; if it lands on first_touch it is lazy and any "
                     "capacity claim that skipped the touch would be fiction.  Noisy when "
                     "ranks>1 because the peer rank allocates concurrently, and 'unattributed' "
                     "is expected when ARC shrink or swap masks the drop -- read it together "
                     "with commit_accounting, which is the invalidator."),
            "attribution_clean": phase["ranks"] == 1,
            "commit_timing": timing,
            "delta_bytes_hipMemCreate": m_create,
            "delta_bytes_hipMemMap_setAccess": m_map,
            "delta_bytes_first_touch": m_touch,
        }


# ===========================================================================
# Verdict, projection, reporting
# ===========================================================================

THRESHOLDS = [
    ("34GiB", 34 * GIB),
    ("34GB_decimal", 34_000_000_000),
]


# ===========================================================================
# Host-locatedness gate -- runs BEFORE any climb.
#
# Added 2026-09-03 after the first real run died with
#     "Memory access fault by GPU node-1 ... Reason: Page not present"
# and the diagnostics in p3_diagnostics_*.json established WHY:
#
#   * hipMemCreate(location.type=hipMemLocationTypeHost) on ROCm 7.2.4 / gfx1201
#     allocates DEVICE VRAM.  The location field is accepted, echoed back verbatim by
#     hipMemGetAllocationPropertiesFromHandle, and ignored.  sysfs mem_info_vram_used
#     tracks the arena 1:1 (0.75 -> 15.75 GiB over a 512 MiB-at-a-time climb) while
#     MemAvailable and mem_info_gtt_used never move, and the first touch runs at ~300
#     GB/s -- HBM speed, ~11x this box's 28 GB/s PCIe H2D ceiling.
#   * The arena therefore tops out at the card's 16 GB of VRAM, not at host RAM.
#   * Worse, the over-commit is NOT reported.  hipMemCreate/Map/SetAccess all return
#     hipSuccess past the point where the pages can be backed; the failure surfaces as an
#     unrecoverable GPU page fault (SIGABRT) on FIRST TOUCH.  A capacity climb built on
#     this API cannot fail gracefully -- which is exactly how it destroyed the first run.
#
# So the climb must not be attempted until the premise is checked.  The gate is fenced in
# a disposable subprocess for the same reason: its own first touch may hard-fault.
# ===========================================================================

# Anything above this is not crossing PCIe.  Measured H2D on this box: 26.8 GB/s at the
# 2.8 MiB expert granule, 28.7 GB/s at 1 GiB contiguous.
GATE_MAX_HOST_GB_S = 100.0
GATE_SIZE = 2 * GIB
GATE_ABSORB_FRAC = 0.5      # a pool that took >50% of the arena is where it landed


def _pool_snapshot() -> dict:
    mi = read_meminfo()
    cards = [c for c in read_gpu_vram_sysfs() if c.get("discrete")]
    return {
        "mem_available_bytes": (mi.get("MemAvailable") or 0) * 1024,
        "mem_free_bytes": (mi.get("MemFree") or 0) * 1024,
        "vram_used_bytes": [c.get("vram_used_bytes") for c in cards],
        "gtt_used_bytes": [c.get("gtt_used_bytes") for c in cards],
    }


def locatedness_gate_worker(device: int, size: int) -> int:
    """Measure WHERE one host-located allocation actually lands.  Three independent
    discriminators, because a return code proves nothing here: the driver returns
    hipSuccess for the whole sequence either way."""
    h = Hip()
    out = {"device": device, "size_bytes": size}
    h.ck(h.lib.hipSetDevice(device), f"hipSetDevice({device})")
    out["identity"] = h.device_identity(device)
    out["sysfs_card"] = _sysfs_card_for_pci(out["identity"].get("pci_bus_id"))

    s0 = _pool_snapshot()
    va = ctypes.c_void_p()
    h.ck(h.lib.hipMemAddressReserve(ctypes.byref(va), size, 2 * MIB, None, 0),
         "gate hipMemAddressReserve")
    handle = ctypes.c_void_p()
    t = time.perf_counter()
    h.ck(h.lib.hipMemCreate(ctypes.byref(handle), size, ctypes.byref(host_prop()), 0),
         "gate hipMemCreate(location=Host)")
    out["t_create_s"] = time.perf_counter() - t

    # What does the driver SAY it gave us?  (It echoes the request -- recorded so the
    # report can show the query passing while the operation does the opposite.)
    try:
        fn = h.lib.hipMemGetAllocationPropertiesFromHandle
        fn.argtypes = [ctypes.POINTER(hipMemAllocationProp), ctypes.c_void_p]
        fn.restype = ctypes.c_int
        q = hipMemAllocationProp()
        ctypes.memset(ctypes.byref(q), 0, ctypes.sizeof(q))
        rc = fn(ctypes.byref(q), handle)
        out["driver_reported_props"] = {
            "rc": rc, "alloc_type": q.type, "location_type": q.location.type,
            "location_id": q.location.id,
            "location_is_host": q.location.type == HIP_MEM_LOCATION_TYPE_HOST}
    except AttributeError:
        out["driver_reported_props"] = {"rc": None, "note": "symbol not present"}

    h.ck(h.lib.hipMemMap(va, size, 0, handle, 0), "gate hipMemMap")
    h.ck(h.lib.hipMemSetAccess(va, size, ctypes.byref(rw_desc(device)), 1),
         "gate hipMemSetAccess")
    time.sleep(0.4)
    s1 = _pool_snapshot()

    # Warm the memset path so the timed write is not paying first-kernel-launch cost,
    # then write the WHOLE range through the device VA (plan sec.5.2: never a CPU store).
    h.lib.hipMemsetD32(va, 0x1, 4096)
    h.lib.hipDeviceSynchronize()
    sys.stderr.write("[gate] about to first-touch; a hard GPU fault here is itself the "
                     "result, which is why this runs in a throwaway process\n")
    sys.stderr.flush()
    t = time.perf_counter()
    h.ck(h.lib.hipMemsetD32(va, 0x5A5A1234, size // 4), "gate hipMemsetD32")
    h.ck(h.lib.hipDeviceSynchronize(), "gate sync")
    dt = time.perf_counter() - t
    out["first_touch_s"] = dt
    out["first_touch_gb_s"] = size / dt / 1e9 if dt > 0 else None
    time.sleep(0.6)
    s2 = _pool_snapshot()

    buf = (ctypes.c_uint32 * 4)()
    ok = True
    for off in (0, size // 2, size - 16):
        rc = h.lib.hipMemcpy(ctypes.cast(buf, ctypes.c_void_p),
                             ctypes.c_void_p(va.value + off), 16, HIP_MEMCPY_DEVICE_TO_HOST)
        ok = ok and rc == 0 and buf[0] == 0x5A5A1234
    out["readback_ok"] = ok

    def dmax(key, invert=False):
        a, b = s0[key], s2[key]
        if isinstance(a, list):
            pairs = [(y - x) for x, y in zip(a, b) if x is not None and y is not None]
            return max(pairs) if pairs else None
        return (a - b) if invert else (b - a)

    dv, dg = dmax("vram_used_bytes"), dmax("gtt_used_bytes")
    dh = dmax("mem_available_bytes", invert=True)
    out["pool_snapshots"] = {"before": s0, "after_map": s1, "after_touch": s2}
    out["absorbed_bytes"] = {"vram": dv, "gtt": dg, "host_mem_available": dh}
    out["absorbed_fraction"] = {k: (None if v is None else round(v / size, 4))
                                for k, v in out["absorbed_bytes"].items()}

    bw = out["first_touch_gb_s"] or 0.0
    fr = out["absorbed_fraction"]
    out["bandwidth_says"] = ("device" if bw > GATE_MAX_HOST_GB_S else "host")
    out["accounting_says"] = (
        "device" if (fr.get("vram") or 0) > GATE_ABSORB_FRAC else
        "host" if (fr.get("host_mem_available") or 0) > GATE_ABSORB_FRAC else
        "gtt" if (fr.get("gtt") or 0) > GATE_ABSORB_FRAC else "unaccounted")
    out["is_host_located"] = (out["bandwidth_says"] == "host"
                              and out["accounting_says"] == "host")
    out["agree"] = out["bandwidth_says"] == out["accounting_says"]

    h.lib.hipMemUnmap(va, size)
    h.lib.hipMemRelease(handle)
    h.lib.hipMemAddressFree(va, size)
    sys.stdout.write(json.dumps(out) + "\n")
    sys.stdout.flush()
    return 0


def run_locatedness_gate(args) -> dict:
    """Parent side.  A hard GPU fault in the child is a RESULT, not a crash."""
    cmd = [sys.executable, os.path.abspath(__file__), "--_locgate",
           "--device", "0", "--size", str(GATE_SIZE)]
    t = time.perf_counter()
    try:
        pr = subprocess.run(cmd, capture_output=True, text=True,
                            timeout=max(120.0, args.worker_timeout))
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "is_host_located": False, "cmd": cmd,
                "detail": "the gate did not return; treat host-locatedness as unproven"}
    lines = [l for l in pr.stdout.splitlines() if l.startswith("{")]
    rec = None
    if lines:
        try:
            rec = json.loads(lines[-1])
        except ValueError:
            rec = None
    faulted = "Memory access fault" in (pr.stderr or "")
    if rec is None:
        return {
            "status": "faulted" if faulted else "failed",
            "is_host_located": False,
            "returncode": pr.returncode, "wall_s": time.perf_counter() - t,
            "stderr_tail": (pr.stderr or "").strip()[-1200:], "cmd": cmd,
            "detail": ("the gate process died before reporting"
                       + (" with a GPU memory access fault -- the allocation could not be "
                          "backed and the driver signalled it as an unrecoverable page "
                          "fault rather than an error return" if faulted else "")),
        }
    rec["status"] = "measured"
    rec["returncode"] = pr.returncode
    rec["wall_s"] = time.perf_counter() - t
    rec["size_gib"] = GATE_SIZE / GIB
    rec["max_plausible_host_gb_s"] = GATE_MAX_HOST_GB_S
    rec["pcie_reference_gb_s"] = {"h2d_1gib_contiguous": 28.7, "h2d_2p8mib_granule": 26.8}
    rec["cmd"] = cmd
    return rec

LEGAL_FORKS = {
    None,
    "MIXED_DEVICE_HOST_STACK_VIABLE",
    "MIXED_DEVICE_HOST_STACK_VIABLE_AT_DECIMAL_34GB_ONLY",
    "SINGLE_RANK_ONLY",
    "FILE_BACKED_MMAP_FALLBACK",
}


def build_verdict(phases: dict, gate: dict = None) -> dict:
    """Evaluate the design fork against BOTH readings of '34 GB'.  The climb records the
    max actually committed, so every threshold is evaluated post-hoc from one measurement."""
    out = {
        "question": ("P3 fork: does hipMemCreate(location=Host) deliver 34 GB for one rank "
                     "and 68 GB for two ranks concurrently on this box?"),
        "thresholds_gib": {k: v / GIB for k, v in THRESHOLDS},
        "per_phase": {},
        "fork": None,
        "fork_rationale": None,
    }
    for name, ph in phases.items():
        if ph is None:
            out["per_phase"][name] = {"status": "not_run"}
            continue
        if ph.get("synthetic") or ph.get("stop_reason") == "not_measured":
            # A synthetic phase carries no measurement; it must never produce a verdict.
            out["per_phase"][name] = {"status": "not_measured",
                                      "reason": "synthetic phase (selftest)"}
            continue
        # MIN across ranks, not the mean.  "two ranks concurrently at X each" is bounded by
        # the WEAKER rank; averaging a diverged pair overstates the per-rank capacity.
        per_rank_list = [b for b in (ph.get("committed_bytes_per_rank") or []) if b is not None]
        per_rank = (min(per_rank_list) if per_rank_list
                    else ((ph["committed_bytes_total"] // ph["ranks"]) if ph["ranks"] else 0))
        row = {
            "ranks": ph["ranks"],
            "committed_gib_total": ph["committed_bytes_total"] / GIB,
            "committed_gib_per_rank": per_rank / GIB,
            "committed_gib_per_rank_basis": "min across ranks",
            "stop_reason": ph["stop_reason"],
            "readback_failures": ph["verify"]["failures"],
            "peak_resweep_failures": (ph.get("peak_resweep") or {}).get("failures"),
            "meets": {k: bool(per_rank >= v and ph["verify"]["failures"] == 0)
                      for k, v in THRESHOLDS},
        }
        out["per_phase"][name] = row

    # ---- HARD INVALIDATORS -------------------------------------------------
    # If any of these fire, NO fork may be read from this run at all.  Previously a
    # read-back failure still produced a fork string (every `meets` went False, which the
    # chain below happily read as FILE_BACKED_MMAP_FALLBACK) -- a corrupt run recommending
    # a design change is the worst possible output of this probe.
    inval = []
    # The gate is checked FIRST and on its own: if the pages are not host-located there is
    # nothing for the climb to have measured, and a capacity number taken from a device
    # allocation would be a confident wrong answer to the design fork.
    if gate and gate.get("status") != "skipped":
        if gate.get("is_host_located") is False:
            if gate.get("status") == "measured":
                detail = (
                    f"hipMemCreate(location=hipMemLocationTypeHost) did not return host "
                    f"memory. A {gate.get('size_gib')} GiB allocation was absorbed "
                    f"{gate.get('absorbed_fraction', {}).get('vram')} into discrete VRAM, "
                    f"{gate.get('absorbed_fraction', {}).get('host_mem_available')} into "
                    f"host MemAvailable and "
                    f"{gate.get('absorbed_fraction', {}).get('gtt')} into GTT; the first "
                    f"touch through the device VA ran at {gate.get('first_touch_gb_s')} "
                    f"GB/s against a {GATE_MAX_HOST_GB_S} GB/s ceiling for anything "
                    f"crossing PCIe. The driver reported the location back as "
                    f"{gate.get('driver_reported_props', {}).get('location_type')} "
                    f"(host={gate.get('driver_reported_props', {}).get('location_is_host')})"
                    f" -- the request is echoed and ignored.")
            else:
                detail = (f"the host-locatedness gate did not complete "
                          f"(status={gate.get('status')}): {gate.get('detail')}")
            inval.append({"invalidator": "pages_are_not_host_located", "phase": "gate",
                          "detail": detail, "evidence": gate})
    for name, ph in phases.items():
        if ph is None or ph.get("synthetic") or ph.get("stop_reason") == "not_measured":
            continue
        if ph["verify"]["failures"]:
            inval.append({
                "invalidator": "readback_verify_failed", "phase": name,
                "detail": (f"{ph['verify']['failures']} chunk read-backs did not return "
                           f"their own fingerprint; the driver returns hipSuccess even when "
                           f"the page table is wrong, so committed bytes cannot be trusted."),
                "evidence": ph["verify"]["failure_detail"][:4]})
        vc = ph.get("vram_check") or {}
        if vc.get("host_pages_cost_zero_vram") is False:
            inval.append({
                "invalidator": "host_pages_cost_vram", "phase": name,
                "detail": (f"discrete VRAM grew {vc.get('worst_growth_mib')} MiB "
                           f"({vc.get('worst_growth_frac_of_committed')} of committed) during "
                           f"the climb: the pages are not host-located."),
                "evidence": vc})
        hl = ph.get("host_locatedness") or {}
        if hl.get("consistent_with_host_located") is False:
            inval.append({
                "invalidator": "first_touch_faster_than_pcie", "phase": name,
                "detail": (f"first touch through the device VA ran at "
                           f"{hl.get('first_touch_median_gb_s')} GB/s, above the "
                           f"{hl.get('max_plausible_host_gb_s')} GB/s ceiling for anything "
                           f"crossing PCIe: these pages are device memory, not host memory."),
                "evidence": hl})
        ca = ph.get("commit_accounting") or {}
        if ca.get("commit_landed") is False:
            inval.append({
                "invalidator": "commit_landed_nowhere", "phase": name,
                "detail": (f"only {ca.get('accounted_frac_of_committed')} of the claimed "
                           f"{ca.get('committed_bytes')} B is accounted for in MemAvailable "
                           f"(+ARC shrink +swap): nothing was really committed."),
                "evidence": ca})
    out["invalidated"] = bool(inval)
    out["invalidators"] = inval

    caveats = []
    for name, ph in phases.items():
        sh = (ph or {}).get("start_headroom") or {}
        if sh.get("recovered") is False:
            caveats.append(
                f"phase {name} started {sh.get('deficit_vs_baseline_gib')} GiB below the "
                f"baseline MemAvailable (previous phase had not fully released after "
                f"{sh.get('waited_s')} s). Its capacity figure is a LOWER BOUND.")
        for arm in ((ph or {}).get("ratebench") or []):
            if not arm.get("trustworthy"):
                caveats.append(
                    f"phase {name} ratebench arm {arm.get('arm')}@"
                    f"{arm.get('chunk_bytes', 0)//MIB} MiB is NOT trustworthy "
                    f"(error={arm.get('error')!r}, verify_failures="
                    f"{arm.get('verify_failures')}); do not quote its GB/s.")
    out["caveats"] = caveats

    r1 = out["per_phase"].get("rank1", {})
    r2 = out["per_phase"].get("rank2", {})
    measured = [r for r in out["per_phase"].values() if "meets" in r]
    # Invalidators are read BEFORE "was anything measured".  A gate failure legitimately
    # leaves zero phases, and reporting that as a bland "NOT MEASURED" would bury the one
    # finding that actually decides the design.
    if inval:
        out["fork"] = None
        out["fork_rationale"] = (
            "INVALIDATED -- no fork may be read from this run. "
            + "; ".join(f"{i['invalidator']} ({i['phase']}): {i['detail']}" for i in inval)
            + " Fix the cause and re-run; do NOT read a design decision off a run whose "
              "own data checks failed.")
        return out
    if not measured:
        out["fork"] = None
        out["fork_rationale"] = ("NOT MEASURED -- no phase produced a capacity measurement. "
                                 "This file cannot be used to decide the design fork.")
        return out
    # Both the SINGLE_RANK_ONLY and FILE_BACKED forks are CLAIMS ABOUT A PHASE THAT MAY NOT
    # HAVE RUN: 'one rank fits, two do not' needs rank2, and 'did not reach target even for
    # one rank' needs rank1.  Emitting either from a single-mode run is a confident wrong
    # design decision.
    have_r1 = "meets" in r1
    have_r2 = "meets" in r2
    if r2.get("meets", {}).get("34GiB"):
        out["fork"] = "MIXED_DEVICE_HOST_STACK_VIABLE"
        out["fork_rationale"] = (
            "2 x 34 GiB of host-located hipMemCreate pages committed concurrently with "
            "clean read-back. Proceed with host-located VMM pages; T1/T2 stand; NVMe stays "
            "deferred (plan sec.9).")
    elif r2.get("meets", {}).get("34GB_decimal"):
        out["fork"] = "MIXED_DEVICE_HOST_STACK_VIABLE_AT_DECIMAL_34GB_ONLY"
        out["fork_rationale"] = (
            "2 x 34e9 B fits but 2 x 34 GiB does not. Viable only if the real per-rank arena "
            "is <= 34e9 B; size the arena from the measured checkpoint bytes before relying "
            "on this.")
    elif (r1.get("meets", {}).get("34GiB") or r1.get("meets", {}).get("34GB_decimal")) and have_r2:
        out["fork"] = "SINGLE_RANK_ONLY"
        out["fork_rationale"] = (
            "One rank fits, two concurrently do not. TP=2 with host-located pages is out on "
            "this box's RAM; either TP=1, or fall back to file-backed mmap(MAP_SHARED) + "
            "hipHostRegister (forfeits the device tier, drags route_E/align_E back in).")
    elif have_r1 and have_r2:
        out["fork"] = "FILE_BACKED_MMAP_FALLBACK"
        out["fork_rationale"] = (
            "Host-located hipMemCreate did not reach the target even for one rank. Fall back "
            "to file-backed mmap(MAP_SHARED) + hipHostRegister; the device tier is forfeited "
            "and the plan's sec.4.4 route_E/align_E prerequisite returns.")
    else:
        missing = "rank2" if not have_r2 else "rank1"
        out["fork"] = None
        out["fork_rationale"] = (
            f"INCONCLUSIVE -- the {missing} phase was not measured (--mode="
            f"{'rank1' if missing == 'rank2' else 'rank2'}), and every remaining fork is a "
            f"claim about it: SINGLE_RANK_ONLY asserts two ranks fail, "
            f"FILE_BACKED_MMAP_FALLBACK asserts even one rank fails. Run --mode both.")
    if out["fork"] and caveats:
        out["fork_rationale"] += " CAVEATS: " + " ".join(caveats)
    return out


def build_projection(baseline: dict, phases: dict, args) -> dict:
    """What would be needed on a QUIET box.  Every input is measured; the arithmetic and
    its assumptions are spelled out so the number cannot be quoted without them."""
    mi = baseline["meminfo_kib"]
    total_kib = mi.get("MemTotal")
    avail_kib = mi.get("MemAvailable")
    arc = baseline["zfs_arc"].get("size_kib")
    eng = baseline.get("engine", {})
    eng_rss_kib = eng.get("heavy_process_rss_kib")
    required_total = int(round(args.per_rank_gib * GIB)) * 2

    def g(kib):
        return None if kib is None else kib * 1024 / GIB

    measured_headroom = g(avail_kib)
    plus_arc = None if (avail_kib is None or arc is None) else g(avail_kib + arc)
    non_probe_used = None if (total_kib is None or avail_kib is None) else g(total_kib - avail_kib)
    quiet_need = required_total / GIB + args.os_floor_gib + (
        (eng_rss_kib * 1024 / GIB) if eng_rss_kib else 0.0)

    return {
        "assumptions": [
            "MemAvailable is the kernel's own estimate of allocatable-without-swapping RAM.",
            "ZFS ARC is NOT counted in MemAvailable but IS reclaimable, so 'available + ARC' "
            "is an upper bound the kernel will only realise under pressure.",
            "engine_rss is the summed VmRSS of processes matching the engine regex; it is a "
            "floor for engine footprint, not a full accounting (it excludes page cache and "
            "any container-side allocations that did not match).",
            f"os_floor_gib={args.os_floor_gib} is an assumed reserve for the OS and page cache, "
            "not a measurement.",
        ],
        "mem_total_gib": g(total_kib),
        "baseline_mem_available_gib": measured_headroom,
        "baseline_used_excl_available_gib": non_probe_used,
        "zfs_arc_size_gib": g(arc),
        "zfs_arc_cap_gib": g(baseline["zfs_arc"].get("c_max_kib")),
        "engine_detected": eng.get("engine_detected"),
        "engine_rss_gib": None if not eng_rss_kib else eng_rss_kib * 1024 / GIB,
        "swap_used_gib": (
            None if mi.get("SwapTotal") is None or mi.get("SwapFree") is None
            else (mi["SwapTotal"] - mi["SwapFree"]) * 1024 / GIB),
        "required_total_gib": required_total / GIB,
        "estimated_headroom_as_measured_gib": measured_headroom,
        "estimated_headroom_plus_arc_gib": plus_arc,
        "quiet_box_requirement_gib": quiet_need,
        "quiet_box_requirement_formula": (
            "2 * per_rank_gib + os_floor_gib + engine_rss_gib"),
        "fits_on_this_box_if_quiet": (
            None if g(total_kib) is None else bool(g(total_kib) >= quiet_need)),
        "headroom_after_assumed_engine_gib": (
            None if measured_headroom is None else
            {f"engine_{n}gib": round(measured_headroom - n, 3) for n in (0, 4, 8, 16, 24)}),
        "headroom_ladder_note": (
            "Pure arithmetic on the MEASURED baseline headroom, not a measurement of an "
            "engine. Use the row matching the serve you intend to co-locate; if the probe "
            "ran with the engine already up, the engine_0gib row is the real one."),
        "shortfall_vs_measured_headroom_gib": (
            None if measured_headroom is None
            else round(required_total / GIB - measured_headroom, 3)),
    }


REQUIRED_TOP_KEYS = [
    "probe", "schema_version", "status", "title", "started_utc", "finished_utc",
    "duration_s", "host", "args", "env", "cards", "preconditions", "box_state",
    "swap_baseline", "host_locatedness_gate", "phases", "verdict",
    "quiet_box_projection", "notes",
]


def validate_result(res: dict) -> None:
    """Assert the JSON shape.  Exercised identically by --selftest and by a real run, so
    the Verify phase actually checks the shipped shape."""
    missing = [k for k in REQUIRED_TOP_KEYS if k not in res]
    if missing:
        raise ProbeError(f"result JSON is missing required keys: {missing}")
    if res["probe"] != PROBE_ID:
        raise ProbeError(f"result JSON has probe={res['probe']!r}, expected {PROBE_ID!r}")
    if res["status"] not in ("ok", "aborted", "failed", "selftest"):
        raise ProbeError(f"bad status {res['status']!r}")
    if not isinstance(res["phases"], dict):
        raise ProbeError("phases must be an object")
    for name, ph in res["phases"].items():
        if ph is None:
            continue
        for k in ("ranks", "committed_bytes_total", "stop_reason", "rate", "verify",
                  "rounds", "workers", "target_bytes_per_rank", "chunk_bytes"):
            if k not in ph:
                raise ProbeError(f"phase {name!r} is missing key {k!r}")
    for k in ("fork", "per_phase", "thresholds_gib", "invalidated", "invalidators",
              "caveats"):
        if k not in res["verdict"]:
            raise ProbeError(f"verdict is missing key {k!r}")
    if res["verdict"]["fork"] not in LEGAL_FORKS:
        raise ProbeError(f"verdict.fork={res['verdict']['fork']!r} is not one of the four "
                         f"forks the plan defines (or null): {sorted(LEGAL_FORKS)}")
    if res["verdict"]["invalidated"] and res["verdict"]["fork"] is not None:
        raise ProbeError("verdict is invalidated but still carries a fork; an invalidated "
                         "run must not recommend a design decision")
    if res["env"].get("ROCR_VISIBLE_DEVICES") != "0,1":
        raise ProbeError("env.ROCR_VISIBLE_DEVICES must be '0,1' -- the iGPU must never "
                         "enter enumeration")
    if res["env"].get("HIP_VISIBLE_DEVICES") is not None:
        raise ProbeError("env.HIP_VISIBLE_DEVICES must be unset")
    try:
        json.dumps(res)
    except (TypeError, ValueError) as exc:
        raise ProbeError(f"result is not JSON-serialisable: {exc}") from exc


def _fmt(v, unit="", nd=2):
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.{nd}f}{unit}"
    return f"{v}{unit}"


def render_md(res: dict) -> str:
    L = []
    A = L.append
    A(f"# P3 — Host-located VMM capacity ({res['status'].upper()})")
    A("")
    A(f"*{res['started_utc']} → {res['finished_utc']} · {_fmt(res['duration_s'], ' s', 1)} · "
      f"`{res['host']['hostname']}` · kernel {res['host']['kernel']}*")
    A("")
    A("**Question (plan §3, P3):** does `hipMemCreate(location=Host)` deliver 34 GB for one "
      "rank and 68 GB for two ranks *concurrently* on this box, and at what rate?")
    A("")
    v = res["verdict"]
    if v.get("invalidated"):
        A("## ⛔ INVALIDATED — no design fork may be read from this run")
        A("")
        for i in v.get("invalidators", []):
            A(f"- **`{i['invalidator']}`** (phase `{i['phase']}`): {i['detail']}")
        A("")
    A(f"## Verdict — `{v.get('fork') or 'NO FORK (see above)'}`")
    A("")
    A(v.get("fork_rationale") or "_no rationale_")
    A("")
    for c in v.get("caveats", []):
        A(f"> ⚠️ {c}")
        A("")
    A("| phase | ranks | committed total | per rank (min) | stop reason | ≥34 GiB/rank | ≥34 GB/rank | readback fails | peak-resweep fails |")
    A("|---|---|---|---|---|---|---|---|---|")
    for name, row in v["per_phase"].items():
        if "meets" not in row:
            A(f"| {name} | — | — | — | {row.get('status', 'not run')} | — | — | — | — |")
            continue
        A(f"| {name} | {row['ranks']} | {_fmt(row['committed_gib_total'], ' GiB')} | "
          f"{_fmt(row['committed_gib_per_rank'], ' GiB')} | `{row['stop_reason']}` | "
          f"{_fmt(row['meets']['34GiB'])} | {_fmt(row['meets']['34GB_decimal'])} | "
          f"{row['readback_failures']} | {_fmt(row.get('peak_resweep_failures'))} |")
    A("")

    # ---- host-locatedness gate ------------------------------------------
    g = res.get("host_locatedness_gate") or {}
    if g.get("status") not in (None, "skipped", "not_run"):
        A("## Gate — are these pages host memory at all?")
        A("")
        A(f"One {_fmt(g.get('size_gib'), ' GiB', 0)} `hipMemCreate("
          f"location=hipMemLocationTypeHost)` on HIP dev 0, mapped to a device VA, "
          f"first-touched through that VA. Three independent discriminators, because "
          f"**every call in the sequence returns `hipSuccess` either way**.")
        A("")
        A("| discriminator | measured | means |")
        A("|---|---|---|")
        af = g.get("absorbed_fraction") or {}
        A(f"| discrete VRAM absorbed | {_fmt(af.get('vram'))} of the arena | "
          f"{'**device memory**' if (af.get('vram') or 0) > 0.5 else 'not VRAM'} |")
        A(f"| host MemAvailable absorbed | {_fmt(af.get('host_mem_available'))} of the arena | "
          f"{'host RAM' if (af.get('host_mem_available') or 0) > 0.5 else '**not host RAM**'} |")
        A(f"| GTT absorbed | {_fmt(af.get('gtt'))} of the arena | "
          f"{'GTT' if (af.get('gtt') or 0) > 0.5 else 'not GTT'} |")
        A(f"| first touch via device VA | {_fmt(g.get('first_touch_gb_s'), ' GB/s')} | "
          f"PCIe H2D on this box is 26.8–28.7 GB/s; anything above "
          f"{_fmt(g.get('max_plausible_host_gb_s'), ' GB/s', 0)} did not cross PCIe |")
        p = g.get("driver_reported_props") or {}
        A(f"| driver's own `hipMemGetAllocationPropertiesFromHandle` | "
          f"location_type={p.get('location_type')} "
          f"(host={_fmt(p.get('location_is_host'))}) | the request is echoed back "
          f"verbatim — the query passes while the operation does the opposite |")
        A(f"| read-back after first touch | {_fmt(g.get('readback_ok'))} | data is intact; "
          f"correctness was never the problem |")
        A("")
        A(f"**Gate verdict: pages are host-located = {_fmt(g.get('is_host_located'))}** "
          f"(accounting says `{g.get('accounting_says')}`, bandwidth says "
          f"`{g.get('bandwidth_says')}`, the two "
          f"{'AGREE' if g.get('agree') else 'DISAGREE'}).")
        A("")
    elif g.get("status") == "skipped":
        A("## Gate — are these pages host memory at all?")
        A("")
        A(f"**SKIPPED.** {g.get('detail')}")
        A("")

    A("## Data-integrity checks (all must pass for the fork to be readable)")
    A("")
    A("| phase | chunks re-verified at peak | resweep fails | host pages cost 0 VRAM | first touch GB/s | consistent w/ host-located | commit accounted |")
    A("|---|---|---|---|---|---|---|")
    for name, ph in res["phases"].items():
        if not ph:
            continue
        rs = ph.get("peak_resweep") or {}
        vc = ph.get("vram_check") or {}
        hl = ph.get("host_locatedness") or {}
        ca = ph.get("commit_accounting") or {}
        A(f"| {name} | {_fmt(rs.get('chunks_reverified'))} | {_fmt(rs.get('failures'))} | "
          f"{_fmt(vc.get('host_pages_cost_zero_vram'))} | "
          f"{_fmt(hl.get('first_touch_median_gb_s'))} | "
          f"{_fmt(hl.get('consistent_with_host_located'))} | "
          f"{_fmt(ca.get('accounted_frac_of_committed'))} |")
    A("")

    A("## Cards — which physical card every timing came from")
    A("")
    A("| phase | rank | HIP dev | PCI bus id | sysfs | name | VRAM total |")
    A("|---|---|---|---|---|---|---|")
    for c in res["cards"]:
        vram = c.get("vram") or {}
        tot = vram.get("total_bytes")
        sc = c.get("sysfs_card") or {}
        A(f"| {c.get('phase')} | {c.get('rank')} | {c.get('hip_ordinal')} | "
          f"`{c.get('pci_bus_id')}` | `{sc.get('sysfs') or 'unresolved'}` | "
          f"{c.get('name')} | {_fmt(None if tot is None else tot / GIB, ' GiB')} |")
    A("")

    A("## Allocation rate")
    A("")
    A("Decimal GB/s (1e9 B/s). Reference to beat/refute: `hipHostMalloc` 4 GiB chunks = "
      "**4.9–5.6 GB/s** (recorded).")
    A("")
    for name, ph in res["phases"].items():
        if not ph:
            continue
        r = ph.get("rate") or {}
        A(f"**phase `{name}` — climb, per-chunk ({ph['chunk_bytes'] // MIB} MiB), "
          f"n={r.get('n_steady_chunks')} steady chunks (warm-up chunk excluded)**")
        A("")
        A("| stage | median GB/s | p25 | p75 | min | max |")
        A("|---|---|---|---|---|---|")
        for key, label in (("hipMemCreate_gb_s", "hipMemCreate"),
                           ("hipMemMap_gb_s", "hipMemMap"),
                           ("hipMemSetAccess_gb_s", "hipMemSetAccess"),
                           ("first_touch_gb_s", "first touch (D32 via device VA)"),
                           ("end_to_end_gb_s", "end-to-end per chunk")):
            s = r.get(key) or {}
            A(f"| {label} | {_fmt(s.get('median'))} | {_fmt(s.get('p25'))} | "
              f"{_fmt(s.get('p75'))} | {_fmt(s.get('min'))} | {_fmt(s.get('max'))} |")
        A("")
        rb = ph.get("ratebench")
        if rb:
            A(f"**phase `{name}` — dedicated rate bench "
              f"(reps={rb[0].get('reps')}, rep 0 discarded)**")
            A("")
            A("| arm | chunk | metric | median GB/s | min | max | n | trustworthy |")
            A("|---|---|---|---|---|---|---|---|")
            for arm in rb:
                for k, s in (arm.get("summary") or {}).items():
                    A(f"| `{arm['arm']}` | {arm['chunk_bytes'] // MIB} MiB | {k} | "
                      f"{_fmt(s.get('median'))} | {_fmt(s.get('min'))} | "
                      f"{_fmt(s.get('max'))} | {s.get('n')} | "
                      f"{_fmt(arm.get('trustworthy'))} |")
            A("")
            for arm in rb:
                if not arm.get("trustworthy"):
                    A(f"> ⚠️ arm `{arm['arm']}`@{arm['chunk_bytes'] // MIB} MiB is NOT "
                      f"trustworthy — error={arm.get('error')!r}, "
                      f"verify_failures={arm.get('verify_failures')}. Do not quote its GB/s.")
                    A("")
        ca = ph.get("commit_attribution")
        if ca:
            A(f"**phase `{name}` — where the memory is actually committed** "
              f"(chunk = {ca['chunk_bytes'] // MIB} MiB; attribution clean: "
              f"{_fmt(ca['attribution_clean'])})")
            A("")
            A("| stage | median MemAvailable drop |")
            A("|---|---|")
            for k, label in (("delta_bytes_hipMemCreate", "hipMemCreate"),
                             ("delta_bytes_hipMemMap_setAccess", "hipMemMap + setAccess"),
                             ("delta_bytes_first_touch", "first touch")):
                m = (ca.get(k) or {}).get("median")
                A(f"| {label} | {_fmt(None if m is None else m / MIB, ' MiB', 1)} |")
            A("")

    for name, ph in res["phases"].items():
        vc = (ph or {}).get("vram_check") or {}
        if vc.get("status") == "measured":
            A(f"**phase `{name}` — device memory during the climb:** worst per-card VRAM "
              f"delta {_fmt(vc.get('worst_abs_delta_mib'), ' MiB', 1)} while committing "
              f"{_fmt((vc.get('committed_host_bytes') or 0) / GIB, ' GiB')} of host pages → "
              f"host pages cost zero VRAM: **{_fmt(vc.get('host_pages_cost_zero_vram'))}**")
            A("")

    A("## Box state")
    A("")
    b = res["box_state"].get("baseline") or {}
    mi = b.get("meminfo_kib", {})
    A(f"- engine detected: **{_fmt((res.get('quiet_box_projection') or {}).get('engine_detected'))}** "
      f"(the plan requires this probe run on a LOADED box, not an idle one)")
    A(f"- MemTotal {_fmt((mi.get('MemTotal') or 0) * 1024 / GIB, ' GiB')} · "
      f"MemAvailable at baseline {_fmt((mi.get('MemAvailable') or 0) * 1024 / GIB, ' GiB')}")
    A(f"- ZFS ARC {_fmt(((b.get('zfs_arc') or {}).get('size_kib') or 0) * 1024 / GIB, ' GiB')} "
      f"(cap {_fmt(((b.get('zfs_arc') or {}).get('c_max_kib') or 0) * 1024 / GIB, ' GiB')})")
    A(f"- swap baseline pswpout rate: "
      f"{_fmt((res.get('swap_baseline') or {}).get('pswpout_mb_s'), ' MB/s')} "
      f"(this box swaps at idle; the stop gate is measured against this, not against zero)")
    A("")
    p = res.get("quiet_box_projection") or {}
    A("### What a quiet box would need")
    A("")
    A(f"- required for 2 ranks: **{_fmt(p.get('required_total_gib'), ' GiB')}**")
    A(f"- measured headroom now: {_fmt(p.get('estimated_headroom_as_measured_gib'), ' GiB')} "
      f"(+ARC: {_fmt(p.get('estimated_headroom_plus_arc_gib'), ' GiB')})")
    A(f"- quiet-box requirement (2·per-rank + OS floor + engine RSS): "
      f"**{_fmt(p.get('quiet_box_requirement_gib'), ' GiB')}** vs MemTotal "
      f"{_fmt(p.get('mem_total_gib'), ' GiB')} → fits: "
      f"{_fmt(p.get('fits_on_this_box_if_quiet'))}")
    A(f"- shortfall vs measured headroom: "
      f"{_fmt(p.get('shortfall_vs_measured_headroom_gib'), ' GiB')}")
    ladder = p.get("headroom_after_assumed_engine_gib")
    if ladder:
        A(f"- headroom if an engine of N GiB were also resident: "
          + ", ".join(f"{k.replace('engine_', '').replace('gib', ' GiB')} → "
                      f"{_fmt(v, ' GiB')}" for k, v in ladder.items()))
    A("")
    for a in p.get("assumptions", []):
        A(f"  - _{a}_")
    A("")
    if res.get("notes"):
        A("## Notes")
        A("")
        for n in res["notes"]:
            A(f"- {n}")
        A("")
    A(f"Raw: `p3.json` (schema {res['schema_version']}).")
    A("")
    return "\n".join(L)


# ===========================================================================
# Preconditions, CLI, main
# ===========================================================================

class _GateFailed(Exception):
    """Raised when the host-locatedness gate proves the pages are not host memory.
    Carries the gate record; unwinds to main's writer so the artifact is still produced."""

    def __init__(self, gate):
        super().__init__("host-locatedness gate failed")
        self.gate = gate


class _Aborted(Exception):
    pass


def _install_signals():
    def handler(signum, _frame):
        raise _Aborted(f"signal {signum}")
    for s in (signal.SIGINT, signal.SIGTERM):
        signal.signal(s, handler)


def acquire_lock(force: bool):
    """The box is shared and this probe commits tens of GiB of RAM.  Two of them at once
    would measure each other."""
    if force:
        return None
    LOCKFILE.parent.mkdir(parents=True, exist_ok=True)
    fh = open(LOCKFILE, "w")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        fh.close()
        raise ProbeError(
            f"another p3 run holds {LOCKFILE} ({exc}). Two concurrent host-capacity climbs "
            f"measure each other. Wait, or pass --force if you are certain.") from exc
    fh.write(f"pid={os.getpid()} started={_now_utc()}\n")
    fh.flush()
    return fh


def validate_args(args) -> None:
    """Argument-only checks.  Runs on BOTH the selftest and the real path, before any
    directory is created or any HIP symbol is resolved, so a bad flag fails cleanly
    instead of tearing on a ZeroDivisionError deep in a phase."""
    chunk = args.chunk_mib * MIB
    problems = []
    if args.chunk_mib <= 0 or chunk % PAGE != 0:
        problems.append(f"--chunk-mib must be positive and a page multiple, got {args.chunk_mib}")
    if args.per_rank_gib <= 0:
        problems.append(f"--per-rank-gib must be positive, got {args.per_rank_gib}")
    if chunk and args.per_rank_gib * GIB < chunk:
        problems.append(f"--per-rank-gib ({args.per_rank_gib}) is smaller than one "
                        f"{args.chunk_mib} MiB chunk")
    if args.floor_gib <= 0:
        problems.append(f"--floor-gib must be positive, got {args.floor_gib}")
    if args.min_headroom_gib < 0:
        problems.append(f"--min-headroom-gib must be >= 0, got {args.min_headroom_gib}")
    if args.chunks_per_round < 1:
        problems.append(f"--chunks-per-round must be >= 1, got {args.chunks_per_round}")
    if args.rate_reps < 2:
        problems.append(f"--rate-reps must be >= 2 (rep 0 is a discarded warm-up), "
                        f"got {args.rate_reps}")
    if args.settle_s < 0 or args.swap_baseline_s <= 0 or args.worker_timeout <= 0:
        problems.append("--settle-s must be >= 0; --swap-baseline-s and --worker-timeout "
                        "must be positive")
    if args.recovery_tolerance_gib < 0 or args.recovery_timeout_s < 0:
        problems.append("--recovery-tolerance-gib and --recovery-timeout-s must be >= 0")
    if _is_volatile(args.outdir):
        problems.append(f"--outdir {args.outdir} looks volatile (tmpfs/ramfs or under "
                        f"/tmp,/dev/shm,/run). Measurement fixtures must be durable and "
                        f"recorded -- write into the worktree.")
    if problems:
        raise ProbeError("invalid arguments:\n  - " + "\n  - ".join(problems))


def check_preconditions(args, baseline: dict) -> dict:
    checks = []

    def req(name, ok, detail=""):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})
        if not ok:
            raise ProbeError(f"PRECONDITION FAILED: {name} -- {detail}")

    req("outdir_writable", os.access(args.outdir, os.W_OK), str(args.outdir))
    req("outdir_is_durable", not _is_volatile(args.outdir),
        f"{args.outdir} looks volatile (tmpfs/ramfs or under /tmp,/dev/shm,/run). "
        f"Measurement fixtures must be durable and recorded -- write into the worktree.")
    req("rocr_visible_devices_is_0_1",
        os.environ.get("ROCR_VISIBLE_DEVICES") == "0,1",
        f"got {os.environ.get('ROCR_VISIBLE_DEVICES')!r}; the iGPU (ROCm device 2, 47 GB GTT) "
        f"must never enter enumeration")
    req("hip_visible_devices_unset", "HIP_VISIBLE_DEVICES" not in os.environ,
        f"got {os.environ.get('HIP_VISIBLE_DEVICES')!r}")

    chunk = args.chunk_mib * MIB
    req("chunk_is_page_multiple", chunk > 0 and chunk % PAGE == 0, f"chunk={chunk} B")
    req("per_rank_positive", args.per_rank_gib > 0, f"{args.per_rank_gib}")
    req("floor_positive", args.floor_gib > 0, f"{args.floor_gib}")

    mi = baseline["meminfo_kib"]
    avail_gib = (mi.get("MemAvailable") or 0) * 1024 / GIB
    need = args.floor_gib + args.min_headroom_gib
    req("baseline_headroom_above_floor", avail_gib >= need,
        f"MemAvailable={avail_gib:.1f} GiB but the floor is {args.floor_gib} GiB and the probe "
        f"needs at least {args.min_headroom_gib} GiB above it to measure anything. "
        f"Free memory or lower --per-rank-gib; do NOT lower --floor-gib to make this pass.")

    eng = baseline.get("engine", {})
    if args.require_engine:
        req("engine_loaded", eng.get("engine_detected"),
            "no engine/serve process or container detected; the plan requires this probe run "
            "with the engine loaded and ARC warm, not on an idle box. Load the engine, or pass "
            "--no-require-engine and accept the recorded caveat.")
    else:
        checks.append({"check": "engine_loaded", "ok": bool(eng.get("engine_detected")),
                       "detail": "advisory only (--no-require-engine)"})

    try:
        Hip(load_only=True)
        checks.append({"check": "libamdhip64_symbols", "ok": True,
                       "detail": "dlopen + all required symbols present (no HIP call made)"})
    except ProbeError as exc:
        req("libamdhip64_symbols", False, str(exc))

    return {"ok": True, "checked": checks}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="p3_host_capacity.py",
        description="P3 -- host-located hipMemCreate capacity, rate and design fork.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--mode", choices=["both", "rank1", "rank2"], default="both",
                    help="rank1 = one rank climbs; rank2 = two ranks climb concurrently")
    ap.add_argument("--per-rank-gib", type=float, default=34.0,
                    help="target arena per rank, in GiB (the strict reading of '34 GB')")
    ap.add_argument("--chunk-mib", type=int, default=512,
                    help="backing chunk size (plan sec.5.1 uses 512 MiB handles)")
    ap.add_argument("--chunks-per-round", type=int, default=1,
                    help="chunks each rank commits between box-state samples")
    ap.add_argument("--floor-gib", type=float, default=8.0,
                    help="stop when MemAvailable falls below this")
    ap.add_argument("--min-headroom-gib", type=float, default=4.0,
                    help="refuse to start unless MemAvailable exceeds floor by this much")
    ap.add_argument("--os-floor-gib", type=float, default=8.0,
                    help="assumed OS+page-cache reserve used ONLY in the quiet-box projection")
    ap.add_argument("--swap-limit-mb-s", type=float, default=64.0,
                    help="absolute pswpout rate that stops the climb")
    ap.add_argument("--swap-slack", type=float, default=4.0,
                    help="also stop if pswpout exceeds this multiple of the measured baseline")
    ap.add_argument("--swap-baseline-s", type=float, default=3.0,
                    help="window used to measure the box's resting swap-out rate")
    ap.add_argument("--rate-reps", type=int, default=6,
                    help="dedicated rate-bench reps; rep 0 is a discarded warm-up")
    ap.add_argument("--skip-locatedness-gate", action="store_true",
                    help="skip the pre-climb check that hipMemCreate(location=Host) "
                         "actually returns host memory. On this box it does NOT, so the "
                         "climb will measure VRAM and then die on a GPU page fault")
    ap.add_argument("--no-ratebench", action="store_true",
                    help="skip the dedicated rate bench (the climb still yields per-chunk rates)")
    ap.add_argument("--settle-s", type=float, default=10.0,
                    help="settle window after release, to measure reclaim")
    ap.add_argument("--recovery-tolerance-gib", type=float, default=4.0,
                    help="a phase may start this far below the baseline MemAvailable; "
                         "beyond it, its capacity result is flagged as a lower bound")
    ap.add_argument("--recovery-timeout-s", type=float, default=120.0,
                    help="how long to wait for the previous phase's RAM to come back")
    ap.add_argument("--worker-timeout", type=float, default=600.0,
                    help="seconds to wait for a worker reply before declaring it hung")
    ap.add_argument("--skip-rocm-smi", action="store_true",
                    help="do not shell out to rocm-smi for the baseline/peak snapshots")
    ap.add_argument("--require-engine", dest="require_engine", action="store_true", default=True,
                    help="fail if no engine/serve is running (the plan requires a loaded box)")
    ap.add_argument("--no-require-engine", dest="require_engine", action="store_false",
                    help="proceed on an idle box and record the caveat")
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR,
                    help="results directory (must be inside the worktree, never /tmp)")
    ap.add_argument("--tag", default=None,
                    help="optional suffix for the output basename (default: p3)")
    ap.add_argument("--force", action="store_true", help="bypass the single-instance lock")
    ap.add_argument("--selftest", action="store_true",
                    help="validate args, box-state readers, JSON shape and the markdown "
                         "renderer WITHOUT touching the GPU; writes p3.selftest.{json,md}")
    ap.add_argument("--dry-run", dest="selftest", action="store_true",
                    help="alias for --selftest")
    ap.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--rank", type=int, help=argparse.SUPPRESS)
    ap.add_argument("--device", type=int, help=argparse.SUPPRESS)
    ap.add_argument("--chunk-bytes", type=int, help=argparse.SUPPRESS)
    ap.add_argument("--reserve-bytes", type=int, help=argparse.SUPPRESS)
    return ap


def _args_dict(args) -> dict:
    return {k: (str(v) if isinstance(v, Path) else v)
            for k, v in sorted(vars(args).items())
            if not k.startswith("_") and k not in ("rank", "device", "chunk_bytes",
                                                   "reserve_bytes")}


def _synthetic_phase(name: str, ranks: int, args) -> dict:
    """A shape-only phase for --selftest.  Every numeric field is flagged synthetic and the
    status makes it impossible to mistake this for a measurement."""
    chunk = args.chunk_mib * MIB
    return {
        "synthetic": True,
        "name": name, "ranks": ranks,
        "target_bytes_per_rank": int(args.per_rank_gib * GIB),
        "target_bytes_total": int(args.per_rank_gib * GIB) * ranks,
        "chunk_bytes": chunk, "chunks_target_per_rank": int(args.per_rank_gib * GIB) // chunk,
        "va_reserved_bytes_per_rank": None, "chunks_per_round": args.chunks_per_round,
        "floor_bytes": int(args.floor_gib * GIB), "swap_baseline": None,
        "workers": [], "rounds": [], "ratebench": None,
        "stop_reason": "not_measured", "stop_detail": "selftest: no GPU work performed",
        "committed_bytes_total": 0, "committed_bytes_per_rank": [0] * ranks,
        "wall_s": None,
        "rate": {"units": "decimal GB/s (1e9 B/s), per-chunk, warmup chunk excluded",
                 "chunk_bytes": chunk, "n_steady_chunks": 0,
                 "hipMemCreate_gb_s": summarize([]), "hipMemMap_gb_s": summarize([]),
                 "hipMemSetAccess_gb_s": summarize([]), "first_touch_gb_s": summarize([]),
                 "end_to_end_gb_s": summarize([]),
                 "per_chunk_seconds": {}, "warmup_chunks": []},
        "commit_attribution": None,
        "verify": {"chunks_verified": 0, "failures": 0, "failure_detail": []},
        "release": [], "recovery": None, "box_state_at_peak": None,
        "peak_resweep": None, "start_headroom": None,
        "vram_check": {"status": "not_measured"},
        "host_locatedness": {"status": "not_measured"},
        "commit_accounting": {"status": "not_measured"},
    }


def run_selftest(args) -> int:
    _log("selftest: no GPU work will be performed")
    args.outdir.mkdir(parents=True, exist_ok=True)
    baseline = box_state("baseline", skip_smi=True)
    baseline["engine"] = detect_engine()
    notes = [
        "SELFTEST RUN -- no GPU work was performed. Every phase is marked synthetic:true and "
        "committed_bytes_total is 0. This file exists to exercise argument handling, the "
        "box-state readers, the JSON schema and the markdown renderer.",
    ]
    try:
        Hip(load_only=True)
        notes.append("libamdhip64.so dlopened and all required symbols resolved "
                     "(no HIP call was made, /dev/kfd untouched).")
        hip_ok = True
    except ProbeError as exc:
        notes.append(f"libamdhip64 symbol check FAILED: {exc}")
        hip_ok = False

    phases = {}
    if args.mode in ("both", "rank1"):
        phases["rank1"] = _synthetic_phase("rank1", 1, args)
    if args.mode in ("both", "rank2"):
        phases["rank2"] = _synthetic_phase("rank2", 2, args)

    res = {
        "probe": PROBE_ID, "schema_version": SCHEMA_VERSION, "status": "selftest",
        "title": "P3 host-located VMM capacity (SELFTEST -- not a measurement)",
        "started_utc": _now_utc(), "finished_utc": _now_utc(), "duration_s": 0.0,
        "host": host_info(), "args": _args_dict(args),
        "env": {"ROCR_VISIBLE_DEVICES": os.environ.get("ROCR_VISIBLE_DEVICES"),
                "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES"),
                "original_ROCR_VISIBLE_DEVICES": _ORIG_ROCR,
                "original_HIP_VISIBLE_DEVICES": _ORIG_HIP},
        "cards": [], "preconditions": {"ok": hip_ok, "checked": [
            {"check": "libamdhip64_symbols", "ok": hip_ok, "detail": "selftest"}]},
        "box_state": {"baseline": baseline, "final": None},
        "swap_baseline": {"window_s": 0.0, "pswpout_mb_s": None, "pswpin_mb_s": None},
        "host_locatedness_gate": {"status": "skipped", "detail": "selftest: no HIP call"},
        "phases": phases,
        "verdict": build_verdict(phases, None),
        "quiet_box_projection": build_projection(baseline, phases, args),
        "notes": notes,
    }
    validate_result(res)
    stem = args.tag or "p3"
    jpath = args.outdir / f"{stem}.selftest.json"
    mpath = args.outdir / f"{stem}.selftest.md"
    jpath.write_text(json.dumps(res, indent=2, sort_keys=False))
    mpath.write_text(render_md(res))
    print(json.dumps(res, indent=2))
    _log(f"selftest OK -> {jpath} and {mpath}")
    for real in (args.outdir / f"{stem}.json", args.outdir / f"{stem}.md"):
        if real.exists():
            _log(f"note: a real result already exists at {real} (untouched by --selftest)")
    _log(f"schema validated ({len(REQUIRED_TOP_KEYS)} required top-level keys)")
    return 0 if hip_ok else 2


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--_worker" in argv:
        argv.remove("--_worker")
        return worker_main(argv)
    if "--_locgate" in argv:
        gp = argparse.ArgumentParser()
        gp.add_argument("--_locgate", action="store_true")
        gp.add_argument("--device", type=int, default=0)
        gp.add_argument("--size", type=int, default=GATE_SIZE)
        ga = gp.parse_args(argv)
        return locatedness_gate_worker(ga.device, ga.size)

    ap = build_parser()
    args = ap.parse_args(argv)
    args.outdir = Path(args.outdir).resolve()
    validate_args(args)

    if args.selftest:
        return run_selftest(args)

    _install_signals()
    args.outdir.mkdir(parents=True, exist_ok=True)
    lock = acquire_lock(args.force)
    t0 = time.monotonic()
    started = _now_utc()
    notes = []
    phases = {}
    status = "ok"
    baseline = None
    gate = {"status": "not_run", "detail": "the run ended before the gate"}
    swap_base = {"window_s": 0.0, "pswpout_mb_s": None, "pswpin_mb_s": None}
    try:
        _log("collecting baseline box state (this is NOT an idle-box measurement by design)")
        baseline = box_state("baseline", skip_smi=args.skip_rocm_smi)
        baseline["engine"] = detect_engine()
        pre = check_preconditions(args, baseline)
        if not baseline["engine"]["engine_detected"]:
            notes.append("CAVEAT: no engine/serve process was detected. The plan requires P3 "
                         "run with the engine loaded and ARC warm; this result is therefore "
                         "OPTIMISTIC relative to the served configuration.")
        _log(f"measuring resting swap-out rate over {args.swap_baseline_s} s")
        swap_base = measure_swap_baseline(args.swap_baseline_s)
        _log(f"  baseline pswpout = {swap_base['pswpout_mb_s']}")

        progress = args.outdir / f"{args.tag or 'p3'}.progress.ndjson"
        notes.append(f"Per-round progress is streamed durably to {progress.name}; if this "
                     f"process is SIGKILLed the climb is still recoverable from it.")

        # ---- gate: are these pages host memory at all? ----------------------
        if args.skip_locatedness_gate:
            gate = {"status": "skipped",
                    "detail": "--skip-locatedness-gate: the climb's own vram_check and "
                              "host_locatedness are the only remaining defence."}
            notes.append("CAVEAT: the host-locatedness gate was SKIPPED. If the pages are "
                         "device memory, this run's capacity number is a VRAM number.")
        else:
            _log(f"host-locatedness gate: {GATE_SIZE / GIB:.0f} GiB on HIP dev 0 "
                 f"(fenced subprocess)")
            gate = run_locatedness_gate(args)
            _log(f"  gate: status={gate.get('status')} "
                 f"host_located={gate.get('is_host_located')} "
                 f"first_touch={gate.get('first_touch_gb_s')} GB/s "
                 f"absorbed={gate.get('absorbed_fraction')}")
        if gate.get("status") != "skipped" and not gate.get("is_host_located"):
            # Climbing now would burn the card's VRAM and then SIGABRT on first touch past
            # the ceiling.  Stop with the finding intact.
            notes.append(
                "PHASES NOT RUN: the host-locatedness gate failed, so there is no host "
                "arena to climb. hipMemCreate(location=Host) returns DEVICE memory on this "
                "box; a capacity climb would have measured VRAM, run out at the card's "
                "16 GB, and died on an unrecoverable GPU page fault (which is exactly how "
                "the first attempt at this probe ended). See verdict.invalidators.")
            _log("  gate FAILED -- skipping both climb phases (see verdict.invalidators)")
            raise _GateFailed(gate)

        if args.mode in ("both", "rank1"):
            run_phase("rank1", 1, args, swap_base, ratebench=not args.no_ratebench,
                      phases=phases, baseline=baseline, progress_path=progress)
        if args.mode in ("both", "rank2"):
            if args.mode == "both":
                _log(f"settling {args.settle_s} s before the two-rank phase")
                time.sleep(args.settle_s)
            run_phase("rank2", 2, args, swap_base, ratebench=False,
                      phases=phases, baseline=baseline, progress_path=progress)
        if args.mode != "both":
            notes.append(f"--mode={args.mode}: only one phase was measured. The fork is "
                         f"INCONCLUSIVE by construction -- SINGLE_RANK_ONLY and "
                         f"FILE_BACKED_MMAP_FALLBACK are both claims about the phase that "
                         f"did not run.")
    except _GateFailed:
        # Not a failure of the probe: the probe answered its question, and the answer is
        # that the premise does not hold on this box.  status stays "ok" so the artifact
        # reads as a completed measurement; verdict.invalidated carries the finding.
        status = "ok"
    except _Aborted as exc:
        status = "aborted"
        notes.append(f"ABORTED by {exc}. Phases already completed are reported; anything "
                     f"missing was never measured.")
        _log(f"ABORTED: {exc}")
    except ProbeError as exc:
        status = "failed"
        notes.append(f"FAILED: {exc}")
        _log(f"FAILED: {exc}")
    except Exception as exc:  # unexpected -- still emit what we have
        status = "failed"
        notes.append(f"FAILED (unexpected {type(exc).__name__}): {exc}")
        _log(f"FAILED (unexpected): {type(exc).__name__}: {exc}")
        pre = locals().get("pre", {"ok": False, "checked": []})
    finally:
        if lock is not None:
            try:
                lock.close()
            except OSError:
                pass

    pre = locals().get("pre", {"ok": False, "checked": [],
                               "note": "preconditions never completed"})
    cards = []
    for pname, ph in phases.items():
        for w in (ph or {}).get("workers", []):
            ident = dict(w.get("identity") or {})
            ident["rank"] = w.get("rank")
            ident["phase"] = pname
            ident["vram"] = (w.get("vram_at_start") or {})
            ident["sysfs_card"] = w.get("sysfs_card")
            ident["host_granularity_bytes"] = w.get("host_granularity_bytes")
            cards.append(ident)

    final = box_state("final", skip_smi=args.skip_rocm_smi) if baseline else None
    res = {
        "probe": PROBE_ID, "schema_version": SCHEMA_VERSION, "status": status,
        "title": "P3 host-located VMM capacity: does 2 x 34 GB fit?",
        "plan_ref": "docs/WEIGHT_OFFLOAD_PLAN.md sec.3 (P3), sec.9, sec.11 item 4",
        "started_utc": started, "finished_utc": _now_utc(),
        "duration_s": time.monotonic() - t0,
        "host": host_info(), "args": _args_dict(args),
        "env": {"ROCR_VISIBLE_DEVICES": os.environ.get("ROCR_VISIBLE_DEVICES"),
                "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES"),
                "original_ROCR_VISIBLE_DEVICES": _ORIG_ROCR,
                "original_HIP_VISIBLE_DEVICES": _ORIG_HIP},
        "cards": cards,
        "preconditions": pre,
        "box_state": {"baseline": baseline, "final": final},
        "swap_baseline": swap_base,
        "host_locatedness_gate": gate,
        "phases": phases,
        "verdict": build_verdict(phases, gate),
        "quiet_box_projection": (build_projection(baseline, phases, args)
                                 if baseline else {"status": "not_measured"}),
        "notes": notes,
    }
    validate_result(res)

    stem = args.tag or "p3"
    jpath = args.outdir / f"{stem}.json"
    mpath = args.outdir / f"{stem}.md"
    jpath.write_text(json.dumps(res, indent=2, sort_keys=False))
    mpath.write_text(render_md(res))
    print(json.dumps(res, indent=2))
    _log(f"wrote {jpath}")
    _log(f"wrote {mpath}")
    _log(f"status={status} fork={res['verdict'].get('fork')}")
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ProbeError as exc:
        print(f"[p3] FATAL: {exc}", file=sys.stderr)
        sys.exit(2)
