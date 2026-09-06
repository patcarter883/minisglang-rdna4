#!/usr/bin/env python3
"""P6 -- hipMemUnmap/hipMemMap conformance tripwire (weight-offload Phase 0).

WHAT THIS IS
------------
A torch-free ctypes probe that answers exactly one question:

    After  hipMemUnmap(VA)  followed by  hipMemMap(VA, handle_B),
    does the GPU serve handle B's physical page, or handle A's stale page?

It checks BOTH directions:

  * READ  -- map handle i at the VA, read, compare against the fingerprint that
             handle i is known to hold.
  * WRITE -- map handle i at the VA, write a marker through the VA, unmap, then
             look at every handle through its OWN permanently-mapped VA and see
             which physical page actually received the marker.

and both access paths:

  * the HIP runtime copy path (hipMemcpy / hipMemsetD32).
  * the shader (a hipRTC kernel doing system-scope loads/stores) -- the path the
    offload design actually depends on, and the only one whose cleanliness can
    turn `dynamic_residency_available` green.

FOUR GUARDS AGAINST A CONFIDENT WRONG ANSWER
--------------------------------------------
1. CONTROL FIRST.  `cross_va_control` writes each handle through its own
   permanent VA and reads it back through a THIRD, never-reused VA.  Only if
   that passes is a later stale read attributable to the REMAP rather than to
   "hipMemMap never bound the requested handle".  A control failure suppresses
   the parked verdict entirely (exit 2) -- the probe refuses to report.
2. CACHE FLUSH.  The working set fits inside the 64 MB MALL, so a read taken
   right after a remap could be served by a cached line from the OLD mapping.
   Every read-back is preceded by streaming a >=4x-MALL scratch buffer.  Without
   it, "stale page table" and "stale cache line" are the same observation, and
   the runtime copy path must NOT be assumed to bypass caches -- on ROCm it is
   often itself a blit kernel.
3. SHADER LIVENESS.  Every launch poisons its output buffer and the kernel
   writes a liveness token; a launch that returns hipSuccess without executing
   is recorded as `shader_dead`, never scored as non-conformant.  The shader is
   also proven read+write correct on ordinary memory before any observation --
   otherwise a broken shader arm would pin the tripwire red forever and the
   probe could never fire.
4. MULTI-OFFSET READS.  Every read samples dwords 0, 1, mid and last of the
   handle, so a range that was only PARTLY remapped (an 8 MiB handle is 2048
   VMM granules) cannot read as conformant.

Divergence between the two access paths localises the fault; the cache-flush arm
is what separates page-table staleness from cache staleness.

EXPECTED RESULT ON THIS BOX (ROCm 7.2.x, gfx1201): **BROKEN**.  Every HIP call
returns hipSuccess and the STALE page is served, for reads and for writes.  That
is why the weight-offload plan's residency is a boot-time *placement* decision
and never a per-step *scheduling* one (see docs/WEIGHT_OFFLOAD_PLAN.md sec. 2).

THIS IS NOT A SHIPPED CODE PATH.  It is (a) an upstream ROCm/HIP bug repro --
hence the exact runtime/driver/kernel versions in the output -- and (b) a
regression tripwire: if AMD ever fixes it, this probe flips to CONFORMANT and
dynamic page-level residency (plan tier T3) becomes available with no redesign.

EXIT CODES
----------
  0  measurement completed and matched --expect
  2  precondition failure / probe error  (never reports a number it did not measure;
     the payload lands on p6.PRECONDITION_FAILED.json so a failed run can never
     overwrite a good p6.json)
  3  expectation mismatch  (tripwire fired: use `--expect broken` in CI)

USAGE
-----
  python3 tools/offload/p6_vmm_remap_conformance.py                  # measure
  python3 tools/offload/p6_vmm_remap_conformance.py --selftest       # no GPU
  python3 tools/offload/p6_vmm_remap_conformance.py --expect broken  # tripwire

`--selftest` validates the JSON shape AND drives the real measurement code
against a mock HIP driver (`p6_mock_driver_test.py`) that simulates a conformant
remap, this box's stale remap, a control failure and a partial remap, requiring
the correct verdict from each.  Shape validation alone cannot show that a
tripwire is capable of firing.

The script pins ROCR_VISIBLE_DEVICES=0,1 itself and clears HIP_VISIBLE_DEVICES so
the Ryzen iGPU (ROCm device 2, gfx1036, 47 GB of GTT) can never enter
enumeration, and REFUSES to report a verdict from any card that is not one of the
two known discrete gfx1201 boards unless --allow-unknown-card is passed.
"""

from __future__ import annotations

# --- device pinning MUST happen before libamdhip64 is ever dlopen'd ----------
import os

_ENV_PIN = {
    "requested_rocr_visible_devices": "0,1",
    "prior_rocr_visible_devices": os.environ.get("ROCR_VISIBLE_DEVICES"),
    "prior_hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
    "prior_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
}
os.environ["ROCR_VISIBLE_DEVICES"] = "0,1"
os.environ.pop("HIP_VISIBLE_DEVICES", None)
os.environ.pop("CUDA_VISIBLE_DEVICES", None)
# ---------------------------------------------------------------------------

import argparse
import ctypes
import datetime
import json
import platform
import shutil
import statistics
import struct
import subprocess
import sys
import time
from pathlib import Path

SCHEMA_VERSION = 3
PROBE_ID = "P6"
PROBE_NAME = "vmm_remap_conformance"

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = REPO_ROOT / "docs" / "measurements" / "WEIGHT_OFFLOAD_2026-09-02"

# ---------------------------------------------------------------------------
# HIP enums / structs
# ---------------------------------------------------------------------------

HIP_SUCCESS = 0
hipMemAllocationTypePinned = 0x1
hipMemLocationTypeDevice = 1
hipMemLocationTypeHost = 2
hipMemAccessFlagsProtReadWrite = 3
hipMemAllocationGranularityMinimum = 0
hipMemAllocationGranularityRecommended = 1
hipMemcpyHostToDevice = 1
hipMemcpyDeviceToHost = 2


class HipMemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class _AllocFlags(ctypes.Structure):
    _fields_ = [
        ("compressionType", ctypes.c_ubyte),
        ("gpuDirectRDMACapable", ctypes.c_ubyte),
        ("usage", ctypes.c_ushort),
    ]


class HipMemAllocationProp(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("requestedHandleType", ctypes.c_int),
        ("location", HipMemLocation),
        ("win32HandleMetaData", ctypes.c_void_p),
        ("allocFlags", _AllocFlags),
    ]


class HipMemAccessDesc(ctypes.Structure):
    _fields_ = [("location", HipMemLocation), ("flags", ctypes.c_int)]


class HipError(RuntimeError):
    def __init__(self, what: str, rc: int, msg: str = ""):
        self.what = what
        self.rc = rc
        self.msg = msg
        super().__init__(f"{what} -> hipError {rc} ({msg or 'no string'})")


class Hip:
    """Thin ctypes binding for the HIP calls this probe needs."""

    def __init__(self, libpath: str | None = None):
        self.libpath = libpath or self._find_lib()
        if self.libpath is None:
            raise RuntimeError(
                "libamdhip64.so not found; set --hip-lib or install ROCm"
            )
        self.lib = ctypes.CDLL(self.libpath)
        self.resolved = (
            os.path.realpath(self.libpath)
            if os.path.isabs(self.libpath)
            else self._resolve_from_maps("libamdhip64") or self.libpath
        )
        L = self.lib
        v, sz, i, u64 = ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_ulonglong
        P = ctypes.POINTER
        L.hipGetErrorString.restype = ctypes.c_char_p
        L.hipGetErrorString.argtypes = [i]
        L.hipGetDeviceCount.argtypes = [P(i)]
        L.hipSetDevice.argtypes = [i]
        L.hipDeviceSynchronize.argtypes = []
        L.hipDeviceGetName.argtypes = [ctypes.c_char_p, i, i]
        L.hipDeviceGetPCIBusId.argtypes = [ctypes.c_char_p, i, i]
        L.hipDeviceTotalMem.argtypes = [P(sz), i]
        L.hipRuntimeGetVersion.argtypes = [P(i)]
        L.hipDriverGetVersion.argtypes = [P(i)]
        L.hipMalloc.argtypes = [P(v), sz]
        L.hipFree.argtypes = [v]
        L.hipMemcpy.argtypes = [v, v, sz, i]
        L.hipMemsetD32.argtypes = [v, i, sz]
        L.hipMemAddressReserve.argtypes = [P(v), sz, sz, v, u64]
        L.hipMemAddressFree.argtypes = [v, sz]
        L.hipMemCreate.argtypes = [P(v), sz, P(HipMemAllocationProp), u64]
        L.hipMemRelease.argtypes = [v]
        L.hipMemMap.argtypes = [v, sz, sz, v, u64]
        L.hipMemUnmap.argtypes = [v, sz]
        L.hipMemSetAccess.argtypes = [v, sz, P(HipMemAccessDesc), sz]
        L.hipMemGetAllocationGranularity.argtypes = [
            P(sz),
            P(HipMemAllocationProp),
            i,
        ]
        L.hipModuleLoadData.argtypes = [P(v), v]
        L.hipModuleGetFunction.argtypes = [P(v), v, ctypes.c_char_p]
        L.hipModuleLaunchKernel.argtypes = [
            v,
            ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
            ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
            ctypes.c_uint,
            v,
            P(v),
            P(v),
        ]

    @staticmethod
    def _resolve_from_maps(needle: str) -> str | None:
        txt = _read("/proc/self/maps") or ""
        for line in txt.splitlines():
            parts = line.split()
            if len(parts) >= 6 and needle in parts[-1]:
                return os.path.realpath(parts[-1])
        return None

    @staticmethod
    def _find_lib() -> str | None:
        cands = [
            os.environ.get("MINISGL_HIP_LIB"),
            "/opt/rocm/lib/libamdhip64.so",
            "/opt/rocm/lib64/libamdhip64.so",
            "/usr/lib/libamdhip64.so",
            "/usr/lib/x86_64-linux-gnu/libamdhip64.so",
            "libamdhip64.so",
        ]
        for c in cands:
            if not c:
                continue
            if os.path.isabs(c):
                if os.path.exists(c):
                    return c
                continue
            try:  # bare soname: let the dynamic loader resolve it
                ctypes.CDLL(c)
                return c
            except OSError:
                continue
        return None

    def errstr(self, rc: int) -> str:
        try:
            s = self.lib.hipGetErrorString(ctypes.c_int(rc))
            return s.decode() if s else ""
        except Exception:  # pragma: no cover - defensive
            return ""

    def ck(self, rc: int, what: str) -> None:
        if rc != HIP_SUCCESS:
            raise HipError(what, rc, self.errstr(rc))


# ---------------------------------------------------------------------------
# host-side fact collection (no GPU work)
# ---------------------------------------------------------------------------


def _read(path: str) -> str | None:
    try:
        with open(path) as fh:
            return fh.read().strip()
    except OSError:
        return None


def _run(cmd: list[str], timeout: float = 20.0) -> dict:
    exe = shutil.which(cmd[0])
    if exe is None:
        return {"cmd": " ".join(cmd), "available": False, "stdout": None, "rc": None}
    try:
        p = subprocess.run(
            [exe] + cmd[1:], capture_output=True, text=True, timeout=timeout
        )
        return {
            "cmd": " ".join(cmd),
            "available": True,
            "rc": p.returncode,
            "stdout": p.stdout.strip(),
            "stderr": p.stderr.strip()[:2000],
        }
    except Exception as e:  # pragma: no cover - environment dependent
        return {"cmd": " ".join(cmd), "available": True, "rc": None, "error": repr(e)}


def _meminfo() -> dict:
    out: dict[str, int] = {}
    txt = _read("/proc/meminfo") or ""
    for line in txt.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].endswith(":"):
            try:
                out[parts[0][:-1]] = int(parts[1])  # kB
            except ValueError:
                pass
    keys = (
        "MemTotal MemFree MemAvailable Buffers Cached Dirty Writeback "
        "SwapTotal SwapFree Shmem"
    ).split()
    return {k: out.get(k) for k in keys}


def _vmstat() -> dict:
    out: dict[str, int] = {}
    txt = _read("/proc/vmstat") or ""
    for line in txt.splitlines():
        k, _, v = line.partition(" ")
        try:
            out[k] = int(v)
        except ValueError:
            pass
    keys = "pswpin pswpout pgmajfault pgpgin pgpgout nr_free_pages".split()
    return {k: out.get(k) for k in keys}


def _gfx_arch_from_target_version(v: int) -> str:
    major, minor, step = v // 10000, (v // 100) % 100, v % 100
    return f"gfx{major}{minor:x}{step:x}"


def _kfd_nodes() -> list[dict]:
    """Enumerate KFD topology nodes -> pci slot, arch, simd count.  Pure file reads."""
    nodes = []
    base = Path("/sys/class/kfd/kfd/topology/nodes")
    if not base.is_dir():
        return nodes
    for nd in sorted(base.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else -1):
        props = {}
        txt = _read(str(nd / "properties")) or ""
        for line in txt.splitlines():
            k, _, v = line.partition(" ")
            try:
                props[k] = int(v)
            except ValueError:
                props[k] = v
        if not props.get("simd_count"):
            continue  # CPU node
        loc = int(props.get("location_id", 0))
        dom = int(props.get("domain", 0))
        slot = f"{dom:04x}:{(loc >> 8) & 0xFF:02x}:{(loc >> 3) & 0x1F:02x}.{loc & 0x7}"
        gtv = int(props.get("gfx_target_version", 0) or 0)
        nodes.append(
            {
                "kfd_node": int(nd.name),
                "pci_slot": slot,
                "gfx_target_version": gtv,
                "gfx_arch": _gfx_arch_from_target_version(gtv) if gtv else None,
                "simd_count": props.get("simd_count"),
                "device_id": props.get("device_id"),
                "vendor_id": props.get("vendor_id"),
                "max_engine_clk_fcompute_mhz": props.get("max_engine_clk_fcompute"),
                "drm_render_minor": props.get("drm_render_minor"),
                "name": _read(str(nd / "name")),
            }
        )
    return nodes


def _drm_cards() -> list[dict]:
    """Per-card VRAM + PCI identity straight from sysfs (no driver ioctl)."""
    cards = []
    for d in sorted(Path("/sys/class/drm").glob("card[0-9]*")):
        if "-" in d.name:
            continue
        dev = d / "device"
        uev = _read(str(dev / "uevent")) or ""
        slot = None
        for line in uev.splitlines():
            if line.startswith("PCI_SLOT_NAME="):
                slot = line.split("=", 1)[1]
        total = _read(str(dev / "mem_info_vram_total"))
        used = _read(str(dev / "mem_info_vram_used"))
        if total is None:
            continue
        cards.append(
            {
                "drm_card": d.name,
                "pci_slot": slot,
                "vram_total_bytes": int(total),
                "vram_used_bytes": int(used) if used is not None else None,
                "link_width": _read(str(dev / "current_link_width")),
                "link_speed": _read(str(dev / "current_link_speed")),
                "power_dpm_state": _read(str(dev / "power_dpm_state")),
            }
        )
    return cards


# Human labels for the two discrete gfx1201 cards on this box; identity is always
# also carried by pci_slot + simd_count so a relabelled box cannot mislead.
_CARD_LABELS = {"0000:03:00.0": "RX 9070 XT (16 GB)", "0000:07:00.0": "RX 9070 (16 GB)"}


def collect_box_state(include_smi: bool) -> dict:
    state = {
        "hostname": platform.node(),
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "loadavg": _read("/proc/loadavg"),
        "uptime_s": float((_read("/proc/uptime") or "0 0").split()[0]),
        "meminfo_kb": _meminfo(),
        "vmstat": _vmstat(),
        "free_g": _run(["free", "-g"])["stdout"] if shutil.which("free") else None,
        "drm_cards": _drm_cards(),
        "kfd_nodes": _kfd_nodes(),
        "rocm_smi": None,
    }
    mi = state["meminfo_kb"]
    if mi.get("MemTotal") and mi.get("MemAvailable"):
        state["mem_total_gib"] = round(mi["MemTotal"] / 1048576.0, 2)
        state["mem_available_gib"] = round(mi["MemAvailable"] / 1048576.0, 2)
        state["mem_used_gib"] = round(
            (mi["MemTotal"] - mi["MemAvailable"]) / 1048576.0, 2
        )
    if include_smi:
        state["rocm_smi"] = _run(["rocm-smi", "--showmeminfo", "vram"])
    return state


def collect_versions(hip: "Hip | None") -> dict:
    v = {
        "rocm_info_version": _read("/opt/rocm/.info/version"),
        "rocm_info_version_dev": _read("/opt/rocm/.info/version-dev"),
        "kernel_release": platform.uname().release,
        "kernel_version": _read("/proc/version"),
        "amdgpu_module_version": _read("/sys/module/amdgpu/version"),
        "amdgpu_module_srcversion": _read("/sys/module/amdgpu/srcversion"),
        "kfd_topology_generation": _read("/sys/class/kfd/kfd/topology/generation_id"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "libamdhip64_path": None,
        "libamdhip64_realpath": None,
        "hip_runtime_version": None,
        "hip_driver_version": None,
        "hiprtc_realpath": None,
    }
    for c in ("/opt/rocm/lib/libhiprtc.so", "/usr/lib/libhiprtc.so"):
        if os.path.exists(c):
            v["hiprtc_realpath"] = os.path.realpath(c)
            break
    if hip is not None:
        v["libamdhip64_path"] = hip.libpath
        v["libamdhip64_realpath"] = hip.resolved
        rt, dr = ctypes.c_int(), ctypes.c_int()
        if hip.lib.hipRuntimeGetVersion(ctypes.byref(rt)) == HIP_SUCCESS:
            v["hip_runtime_version"] = rt.value
        if hip.lib.hipDriverGetVersion(ctypes.byref(dr)) == HIP_SUCCESS:
            v["hip_driver_version"] = dr.value
    else:
        for c in (
            "/opt/rocm/lib/libamdhip64.so",
            "/usr/lib/libamdhip64.so",
            "/usr/lib/x86_64-linux-gnu/libamdhip64.so",
        ):
            if os.path.exists(c):
                v["libamdhip64_path"] = c
                v["libamdhip64_realpath"] = os.path.realpath(c)
                break
    return v


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------


def stats_us(samples_ns: list[int]) -> dict | None:
    """Median + spread in microseconds.  Returns None when nothing was measured."""
    if not samples_ns:
        return None
    xs = sorted(s / 1000.0 for s in samples_ns)
    n = len(xs)

    def pct(p: float) -> float:
        if n == 1:
            return xs[0]
        idx = min(n - 1, max(0, int(round(p * (n - 1)))))
        return xs[idx]

    return {
        "n": n,
        "median_us": round(statistics.median(xs), 3),
        "mean_us": round(statistics.fmean(xs), 3),
        "min_us": round(xs[0], 3),
        "p25_us": round(pct(0.25), 3),
        "p75_us": round(pct(0.75), 3),
        "p95_us": round(pct(0.95), 3),
        "max_us": round(xs[-1], 3),
        "stdev_us": round(statistics.stdev(xs), 3) if n > 1 else 0.0,
    }


# ---------------------------------------------------------------------------
# optional shader path (hipRTC) -- a volatile load/store from a real kernel
# ---------------------------------------------------------------------------

OUT_SLOTS = 8                 # dwords in the kernel's output buffer
OUT_POISON = 0xDEADBEEF       # written before every launch; must be overwritten
OUT_LIVENESS = 0x600DC0DE     # kernel's "I actually ran" token, in out[7]

# `volatile` is NOT a coherence fence on AMDGCN -- it suppresses compiler caching
# only.  For a host-located page read/written across PCIe, and for a shader store
# that a subsequent runtime-copy-path read must observe, we need system-scope
# loads and
# stores plus a system threadfence.  Anything weaker can lose a marker and be
# scored as "shader non-conformant", pinning this tripwire red for a reason that
# has nothing to do with hipMemMap.  The volatile form is kept only as a
# compile-fallback and the variant actually used is recorded in the JSON.
_KERNEL_SRC_SYSTEM = b"""
extern "C" __global__ void p6_touch(unsigned int* p, unsigned int* out,
                                    unsigned int marker, int do_write,
                                    unsigned long long n_dwords) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    unsigned long long mid  = n_dwords >> 1;
    unsigned long long last = n_dwords - 1;
    out[0] = __hip_atomic_load(p,        __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    out[1] = __hip_atomic_load(p + 1,    __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    out[2] = __hip_atomic_load(p + mid,  __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    out[3] = __hip_atomic_load(p + last, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    if (do_write) {
      __hip_atomic_store(p + 1, marker, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
      __threadfence_system();
    }
    out[7] = 0x600DC0DEu;
  }
}
"""

_KERNEL_SRC_VOLATILE = b"""
extern "C" __global__ void p6_touch(unsigned int* p, unsigned int* out,
                                    unsigned int marker, int do_write,
                                    unsigned long long n_dwords) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    volatile unsigned int* vp = (volatile unsigned int*)p;
    unsigned long long mid  = n_dwords >> 1;
    unsigned long long last = n_dwords - 1;
    out[0] = vp[0];
    out[1] = vp[1];
    out[2] = vp[mid];
    out[3] = vp[last];
    if (do_write) { vp[1] = marker; __threadfence_system(); }
    out[7] = 0x600DC0DEu;
  }
}
"""

KERNEL_SRC = _KERNEL_SRC_SYSTEM  # back-compat alias for the recorded source


class ShaderDead(RuntimeError):
    """The kernel launch returned hipSuccess but the kernel did not run."""


class ShaderPath:
    """hipRTC-compiled kernel used to read/write the remap VA from the shader side.

    Entirely optional: any failure is recorded with a reason and the probe falls
    back to the runtime-copy-path result alone.  It never fabricates a value.

    Two hardenings that the conformance verdict depends on:

      * every launch poisons the output buffer first and the kernel writes a
        liveness token, so a launch that returns hipSuccess without executing is
        reported as `shader_dead`, never as `shader_ok = False`;
      * `selfcheck()` proves read AND write work against an ordinary hipMalloc
        buffer before a single conformance observation is taken.  Without it a
        broken shader arm would hold `dynamic_residency_available` at False
        forever -- i.e. the tripwire could never fire.
    """

    def __init__(self, hip: Hip, arch: str, rtc_path: str | None):
        self.hip = hip
        self.arch = arch
        self.available = False
        self.reason: str | None = None
        self.variant: str | None = None
        self.func = None
        self.verified = False
        self.verify_detail: dict | None = None
        self.dead_launches = 0
        self.launches = 0
        self.out_dev = ctypes.c_void_p()
        self._out_host = (ctypes.c_uint32 * OUT_SLOTS)()
        self.src: bytes = b""
        try:
            self._build(rtc_path)
            self.available = True
        except Exception as e:
            self.reason = f"{type(e).__name__}: {e}"

    def _compile(self, rtc, src: bytes) -> bytes:
        prog = ctypes.c_void_p()
        rc = rtc.hiprtcCreateProgram(
            ctypes.byref(prog), src, b"p6_touch.hip", 0, None, None
        )
        if rc != 0:
            raise RuntimeError(f"hiprtcCreateProgram rc={rc}")
        opts = (ctypes.c_char_p * 1)(f"--offload-arch={self.arch}".encode())
        rc = rtc.hiprtcCompileProgram(prog, 1, opts)
        if rc != 0:
            sz = ctypes.c_size_t()
            rtc.hiprtcGetProgramLogSize(prog, ctypes.byref(sz))
            log = ctypes.create_string_buffer(max(1, sz.value))
            rtc.hiprtcGetProgramLog(prog, log)
            raise RuntimeError(
                f"hiprtcCompileProgram rc={rc} arch={self.arch}: "
                f"{log.value.decode(errors='replace')[:600]}"
            )
        sz = ctypes.c_size_t()
        rtc.hiprtcGetCodeSize(prog, ctypes.byref(sz))
        code = ctypes.create_string_buffer(sz.value)
        rtc.hiprtcGetCode(prog, code)
        return code

    def _build(self, rtc_path: str | None) -> None:
        path = rtc_path
        if path is None:
            for c in ("/opt/rocm/lib/libhiprtc.so", "libhiprtc.so"):
                try:
                    ctypes.CDLL(c)
                    path = c
                    break
                except OSError:
                    continue
        if path is None:
            raise RuntimeError("libhiprtc.so not found")
        rtc = ctypes.CDLL(path)
        self.rtc_path = path
        v, P, c_char_p = ctypes.c_void_p, ctypes.POINTER, ctypes.c_char_p
        rtc.hiprtcCreateProgram.argtypes = [P(v), c_char_p, c_char_p, ctypes.c_int, P(c_char_p), P(c_char_p)]
        rtc.hiprtcCompileProgram.argtypes = [v, ctypes.c_int, P(c_char_p)]
        rtc.hiprtcGetProgramLogSize.argtypes = [v, P(ctypes.c_size_t)]
        rtc.hiprtcGetProgramLog.argtypes = [v, c_char_p]
        rtc.hiprtcGetCodeSize.argtypes = [v, P(ctypes.c_size_t)]
        rtc.hiprtcGetCode.argtypes = [v, c_char_p]

        errs = []
        code = None
        for name, src in (("system_atomic", _KERNEL_SRC_SYSTEM),
                          ("volatile", _KERNEL_SRC_VOLATILE)):
            try:
                code = self._compile(rtc, src)
                self.variant, self.src = name, src
                break
            except Exception as e:
                errs.append(f"{name}: {e}")
        if code is None:
            raise RuntimeError(" | ".join(errs))
        if errs:  # the strong variant failed; that weakens the write arm -- say so
            self.reason = f"fell back to `{self.variant}` ({errs[0]})"

        mod = ctypes.c_void_p()
        self.hip.ck(
            self.hip.lib.hipModuleLoadData(ctypes.byref(mod), ctypes.cast(code, ctypes.c_void_p)),
            "hipModuleLoadData",
        )
        fn = ctypes.c_void_p()
        self.hip.ck(
            self.hip.lib.hipModuleGetFunction(ctypes.byref(fn), mod, b"p6_touch"),
            "hipModuleGetFunction",
        )
        self.module = mod
        self.func = fn
        self.hip.ck(
            self.hip.lib.hipMalloc(ctypes.byref(self.out_dev), 4 * OUT_SLOTS),
            "hipMalloc(out)",
        )

    def touch(self, va: int, marker: int | None, n_dwords: int) -> dict:
        """Read dwords 0, 1, mid, last at `va`; optionally store `marker` at dword 1.

        Raises ShaderDead if the launch returned hipSuccess but the kernel did not
        execute -- that is a probe fault, never evidence about hipMemMap.
        """
        hip = self.hip
        # Poison first: a launch that silently does not run must be detectable and
        # must never leave the PREVIOUS observation's values in the buffer.
        hip.ck(
            hip.lib.hipMemsetD32(self.out_dev, _as_int32(OUT_POISON), OUT_SLOTS),
            "hipMemsetD32(out poison)",
        )
        hip.ck(hip.lib.hipDeviceSynchronize(), "sync(out poison)")
        p_arg = ctypes.c_void_p(va)
        o_arg = ctypes.c_void_p(self.out_dev.value)
        m_arg = ctypes.c_uint32(marker if marker is not None else 0)
        w_arg = ctypes.c_int(1 if marker is not None else 0)
        n_arg = ctypes.c_ulonglong(n_dwords)
        params = (ctypes.c_void_p * 5)(
            ctypes.cast(ctypes.byref(p_arg), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(o_arg), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(m_arg), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(w_arg), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(n_arg), ctypes.c_void_p),
        )
        self.launches += 1
        hip.ck(
            hip.lib.hipModuleLaunchKernel(
                self.func, 1, 1, 1, 1, 1, 1, 0, None, params, None
            ),
            "hipModuleLaunchKernel",
        )
        hip.ck(hip.lib.hipDeviceSynchronize(), "sync(kernel)")
        hip.ck(
            hip.lib.hipMemcpy(
                ctypes.byref(self._out_host), self.out_dev, 4 * OUT_SLOTS,
                hipMemcpyDeviceToHost,
            ),
            "hipMemcpy(out)",
        )
        vals = [int(x) for x in self._out_host]
        if vals[7] != OUT_LIVENESS:
            self.dead_launches += 1
            raise ShaderDead(
                f"kernel launch returned hipSuccess but liveness token is "
                f"0x{vals[7]:08x} (expected 0x{OUT_LIVENESS:08x})"
            )
        return {"dw0": vals[0], "dw1": vals[1], "mid": vals[2], "last": vals[3]}

    def selfcheck(self, nbytes: int = 1 << 20) -> dict:
        """Prove the shader can read AND write ordinary device memory.

        Runs before any conformance observation.  If this fails the shader arm is
        disabled with a reason rather than silently scoring every observation as
        non-conformant (which would make the tripwire unable to ever fire).
        """
        hip = self.hip
        n_dw = nbytes // 4
        buf = ctypes.c_void_p()
        detail: dict = {"variant": self.variant, "bytes": nbytes}
        try:
            hip.ck(hip.lib.hipMalloc(ctypes.byref(buf), nbytes), "hipMalloc(selfcheck)")
            sentinel = 0x51F70001
            marker = 0x51F70002
            hip.ck(hip.lib.hipMemsetD32(buf, _as_int32(sentinel), n_dw),
                   "hipMemsetD32(selfcheck)")
            hip.ck(hip.lib.hipDeviceSynchronize(), "sync(selfcheck memset)")
            got = self.touch(buf.value, None, n_dw)
            detail["read"] = got
            detail["read_ok"] = all(got[k] == sentinel for k in ("dw0", "dw1", "mid", "last"))
            self.touch(buf.value, marker, n_dw)
            host = (ctypes.c_uint32 * 2)()
            hip.ck(
                hip.lib.hipMemcpy(ctypes.byref(host), buf, 8, hipMemcpyDeviceToHost),
                "hipMemcpy(selfcheck readback)",
            )
            detail["write_readback"] = [int(host[0]), int(host[1])]
            detail["write_ok"] = int(host[1]) == marker and int(host[0]) == sentinel
            detail["ok"] = bool(detail["read_ok"] and detail["write_ok"])
        except Exception as e:
            detail["ok"] = False
            detail["error"] = f"{type(e).__name__}: {e}"
        finally:
            if buf.value:
                try:
                    hip.lib.hipFree(buf)
                except Exception:
                    pass
        self.verified = bool(detail.get("ok"))
        self.verify_detail = detail
        if not self.verified:
            self.available = False
            self.reason = (
                (self.reason + "; " if self.reason else "")
                + f"shader selfcheck FAILED on ordinary device memory: {detail}"
            )
        return detail

    def usable(self) -> bool:
        return bool(self.available and self.verified)


class CacheFlusher:
    """Evict L2 + the 64 MB MALL by streaming a >=4x-MALL device buffer.

    Without this the probe cannot tell a stale PAGE TABLE from a stale CACHE
    LINE: the whole working set (handles x size, default 32 MiB, plus the parked
    copy) fits inside the MALL, so a read taken right after a remap can legally
    be served from a cached line belonging to the OLD mapping.  Every read-back
    is therefore taken after a flush, and the classification records that.
    """

    def __init__(self, hip: Hip, nbytes: int):
        self.hip = hip
        self.nbytes = nbytes
        self.buf = ctypes.c_void_p()
        self.available = False
        self.reason: str | None = None
        self.flushes = 0
        self._pat = 0
        if nbytes <= 0:
            self.reason = "disabled (--cache-flush-mib 0)"
            return
        try:
            hip.ck(hip.lib.hipMalloc(ctypes.byref(self.buf), nbytes),
                   f"hipMalloc(cache-flush scratch {nbytes} B)")
            self.available = True
        except Exception as e:
            self.reason = f"{type(e).__name__}: {e}"

    def flush(self) -> bool:
        if not self.available:
            return False
        # A different pattern every time, so the memset can never be elided.
        self._pat = (self._pat + 1) & 0xFFFF
        self.hip.ck(
            self.hip.lib.hipMemsetD32(
                self.buf, _as_int32(0x5CAC0000 | self._pat), self.nbytes // 4
            ),
            "hipMemsetD32(cache flush)",
        )
        self.hip.ck(self.hip.lib.hipDeviceSynchronize(), "sync(cache flush)")
        self.flushes += 1
        return True

    def info(self) -> dict:
        return {
            "available": self.available,
            "reason": self.reason,
            "bytes": self.nbytes if self.available else 0,
            "mib": (self.nbytes >> 20) if self.available else 0,
            "flushes": self.flushes,
            "note": "streams a >=4x-MALL buffer between the remap and every "
                    "read-back so a cached line from the OLD mapping cannot be "
                    "misreported as a stale page table",
        }

    def free(self) -> None:
        if self.buf.value:
            try:
                self.hip.lib.hipFree(self.buf)
            except Exception:
                pass
            self.buf = ctypes.c_void_p()
            self.available = False


# ---------------------------------------------------------------------------
# the probe proper
# ---------------------------------------------------------------------------


def _as_int32(v: int) -> int:
    return struct.unpack("<i", struct.pack("<I", v & 0xFFFFFFFF))[0]


class VariantProbe:
    """Runs the sub-tests for one physical-memory location (device or host).

    Order matters and is enforced by `run_variant`:

      1. `cross_va_control`  -- CONTROL.  Each handle is written through its own
         permanent park VA and read back through a THIRD, never-reused VA.  This
         is what separates "hipMemMap binds the handle you asked for" from
         "hipMemMap is a no-op that always hands back the same page".  Without it
         a `stale` result is unattributable.
      2. `minimal_remap`     -- the clean upstream repro; no handle is ever mapped
         at more than one VA while it runs.  Stands alone, and its result is kept
         even if a later sub-test's precondition fails.
      3. `parked_read` / `parked_write` -- resolve WHICH physical page was served
         / written, which requires each handle to be simultaneously parked.
    """

    def __init__(self, hip: Hip, dev: int, location: str, size: int, nh: int,
                 shader: ShaderPath | None, flusher: "CacheFlusher | None" = None):
        self.hip = hip
        self.dev = dev
        self.location = location
        self.size = size
        self.nh = nh
        self.shader = shader if (shader is not None and shader.usable()) else None
        self.flusher = flusher
        self.handles: list[ctypes.c_void_p] = []
        self.va_min = ctypes.c_void_p()
        self.va_park = ctypes.c_void_p()
        self.va_ctl = ctypes.c_void_p()
        self.rw = HipMemAccessDesc()
        self.rw.location.type = hipMemLocationTypeDevice
        self.rw.location.id = dev
        self.rw.flags = hipMemAccessFlagsProtReadWrite
        self.t_map: list[int] = []
        self.t_setaccess: list[int] = []
        self.t_unmap: list[int] = []
        self.t_cycle: list[int] = []
        self.nonzero_rcs: list[dict] = []
        self.shader_dead: list[dict] = []
        # Every fingerprint value this variant has EVER written, mapped back to the
        # (phase, handle) that wrote it.  A stale read is then attributable to a
        # specific earlier write instead of collapsing to "unrecognised".
        self.fp_ledger: dict[int, str] = {}
        self._buf = (ctypes.c_uint32 * 4)()
        # dword indices sampled on every read: first, second, middle, last.  An
        # 8 MiB handle is 2048 VMM granules (4096 B); sampling only dword 0 would
        # be blind to a partially-remapped range.
        self.n_dw = self.size // 4
        self._probe_dw = (0, 1, self.n_dw // 2, self.n_dw - 1)

    # -- low level helpers --------------------------------------------------
    def _prop(self) -> HipMemAllocationProp:
        p = HipMemAllocationProp()
        ctypes.memset(ctypes.byref(p), 0, ctypes.sizeof(p))
        p.type = hipMemAllocationTypePinned
        if self.location == "device":
            p.location.type = hipMemLocationTypeDevice
            p.location.id = self.dev
        else:
            p.location.type = hipMemLocationTypeHost
            p.location.id = 0
        return p

    def _map(self, va: int, h: ctypes.c_void_p, timed: bool) -> None:
        hip = self.hip
        t0 = time.perf_counter_ns()
        rc = hip.lib.hipMemMap(ctypes.c_void_p(va), self.size, 0, h, 0)
        t1 = time.perf_counter_ns()
        self._rc(rc, "hipMemMap")
        rc = hip.lib.hipMemSetAccess(
            ctypes.c_void_p(va), self.size, ctypes.byref(self.rw), 1
        )
        t2 = time.perf_counter_ns()
        self._rc(rc, "hipMemSetAccess")
        if timed:
            self.t_map.append(t1 - t0)
            self.t_setaccess.append(t2 - t1)
            self._pending_cycle = t2 - t0
        else:
            self._pending_cycle = None

    def _unmap(self, va: int, timed: bool) -> None:
        hip = self.hip
        # These are host-side driver calls, but time them only from a quiesced
        # device so no queued work is charged to the unmap.
        self._sync()
        t0 = time.perf_counter_ns()
        rc = hip.lib.hipMemUnmap(ctypes.c_void_p(va), self.size)
        t1 = time.perf_counter_ns()
        self._rc(rc, "hipMemUnmap")
        if timed:
            self.t_unmap.append(t1 - t0)
            if getattr(self, "_pending_cycle", None) is not None:
                self.t_cycle.append(self._pending_cycle + (t1 - t0))
                self._pending_cycle = None

    def _rc(self, rc: int, what: str) -> None:
        if rc != HIP_SUCCESS:
            self.nonzero_rcs.append({"call": what, "rc": rc, "msg": self.hip.errstr(rc)})
            raise HipError(what, rc, self.hip.errstr(rc))

    def _sync(self) -> None:
        self.hip.ck(self.hip.lib.hipDeviceSynchronize(), "hipDeviceSynchronize")

    def _flush(self) -> bool:
        return bool(self.flusher and self.flusher.flush())

    def _note_fp(self, val: int, who: str) -> None:
        self.fp_ledger[val & 0xFFFFFFFF] = who

    def _memset32(self, va: int, val: int, nbytes: int) -> None:
        self._rc(
            self.hip.lib.hipMemsetD32(
                ctypes.c_void_p(va), _as_int32(val), nbytes // 4
            ),
            "hipMemsetD32",
        )
        self._sync()

    def _peek(self, va: int) -> list[int]:
        """Read the sampled dwords via the runtime copy path."""
        out = []
        for dw in self._probe_dw:
            self._rc(
                self.hip.lib.hipMemcpy(
                    ctypes.byref(self._buf), ctypes.c_void_p(va + dw * 4), 4,
                    hipMemcpyDeviceToHost,
                ),
                "hipMemcpy(D2H)",
            )
            out.append(int(self._buf[0]))
        return out

    def _shader_read(self, va: int) -> tuple[list[int] | None, str | None]:
        """Returns (sampled dwords, dead-reason).  Never fabricates a value."""
        if self.shader is None:
            return None, None
        try:
            g = self.shader.touch(va, None, self.n_dw)
        except ShaderDead as e:
            self.shader_dead.append({"va": hex(va), "why": str(e)})
            return None, str(e)
        return [g["dw0"], g["dw1"], g["mid"], g["last"]], None

    def _read_observation(self, va: int, expected: int, flushed: bool) -> dict:
        """One read-back through `va`: runtime copy path + shader, all sampled offsets."""
        copy_vals = self._peek(va)
        sh_vals, sh_dead = self._shader_read(va)
        copy_ok = all(v == expected for v in copy_vals)
        shader_ok = None if sh_vals is None else all(v == expected for v in sh_vals)
        served = self.fp_ledger.get(copy_vals[0])
        return {
            "expected_value": expected,
            "expected_hex": f"0x{expected:08x}",
            "cache_flushed_before_read": flushed,
            "probe_dwords": list(self._probe_dw),
            "observed_copy": copy_vals[0],
            "observed_copy_hex": f"0x{copy_vals[0]:08x}",
            "observed_copy_all_offsets": copy_vals,
            "copy_offsets_agree": len(set(copy_vals)) == 1,
            "observed_shader": None if sh_vals is None else sh_vals[0],
            "observed_shader_all_offsets": sh_vals,
            "shader_dead": sh_dead,
            "served_by": served,
            "copy_ok": copy_ok,
            "shader_ok": shader_ok,
        }

    # -- lifecycle ----------------------------------------------------------
    def setup(self) -> dict:
        hip = self.hip
        prop = self._prop()
        gmin, grec = ctypes.c_size_t(), ctypes.c_size_t()
        rc_gmin = hip.lib.hipMemGetAllocationGranularity(
            ctypes.byref(gmin), ctypes.byref(prop), hipMemAllocationGranularityMinimum
        )
        rc_grec = hip.lib.hipMemGetAllocationGranularity(
            ctypes.byref(grec), ctypes.byref(prop), hipMemAllocationGranularityRecommended
        )
        for i in range(self.nh):
            h = ctypes.c_void_p()
            rc = hip.lib.hipMemCreate(ctypes.byref(h), self.size, ctypes.byref(prop), 0)
            if rc != HIP_SUCCESS:
                raise HipError(
                    f"hipMemCreate(location={self.location}, size={self.size})",
                    rc,
                    hip.errstr(rc),
                )
            self.handles.append(h)
        hip.ck(
            hip.lib.hipMemAddressReserve(
                ctypes.byref(self.va_min), self.size, 2 << 20, None, 0
            ),
            "hipMemAddressReserve(remap VA)",
        )
        hip.ck(
            hip.lib.hipMemAddressReserve(
                ctypes.byref(self.va_park), self.size * self.nh, 2 << 20, None, 0
            ),
            "hipMemAddressReserve(park VA)",
        )
        hip.ck(
            hip.lib.hipMemAddressReserve(
                ctypes.byref(self.va_ctl), self.size * self.nh, 2 << 20, None, 0
            ),
            "hipMemAddressReserve(control VA)",
        )
        return {
            # Recorded as a QUERY only.  Nothing is ever asserted on it -- a
            # granularity query can succeed with a fabricated value while the
            # operation it describes fails.
            "granularity_query_minimum": gmin.value if rc_gmin == HIP_SUCCESS else None,
            "granularity_query_recommended": grec.value if rc_grec == HIP_SUCCESS else None,
            "granularity_query_rc": [rc_gmin, rc_grec],
            "granularity_query_note": "query only; never asserted on -- the "
                                      "conformance verdict rests on observed DATA",
            "remap_va": hex(self.va_min.value),
            "park_va_base": hex(self.va_park.value),
            "control_va_base": hex(self.va_ctl.value),
            "handle_bytes": self.size,
            "probe_dwords": list(self._probe_dw),
        }

    def teardown(self) -> None:
        hip = self.hip
        for base in (self.va_park, self.va_ctl):
            if not base.value:
                continue
            for i in range(self.nh):
                try:
                    hip.lib.hipMemUnmap(
                        ctypes.c_void_p(base.value + i * self.size), self.size
                    )
                except Exception:
                    pass
        try:
            hip.lib.hipMemUnmap(self.va_min, self.size)
        except Exception:
            pass
        for va, sz in ((self.va_min, self.size),
                       (self.va_park, self.size * self.nh),
                       (self.va_ctl, self.size * self.nh)):
            if va.value:
                try:
                    hip.lib.hipMemAddressFree(va, sz)
                except Exception:
                    pass
        for h in self.handles:
            try:
                hip.lib.hipMemRelease(h)
            except Exception:
                pass
        self.handles = []

    # -- sub-test 0: CONTROL -- does a fresh map bind the handle you asked for? --
    def cross_va_control(self) -> dict:
        """Write each handle through park VA i, read it back through control VA i.

        Both VAs are virgin (never remapped), so this measures ONLY 'hipMemMap
        binds the requested handle'.  If it fails, nothing downstream can be
        interpreted -- 'stale' and 'hipMemMap never worked' are indistinguishable.
        """
        self.fp = [0xC0DE0000 | i for i in range(self.nh)]
        for i, v in enumerate(self.fp):
            self._note_fp(v, f"park/handle{i}")
        self.fp_back = {v: i for i, v in enumerate(self.fp)}
        for i, h in enumerate(self.handles):
            self._map(self.va_park.value + i * self.size, h, timed=False)
            self._memset32(self.va_park.value + i * self.size, self.fp[i], self.size)
        self._sync()
        park_read = [self._peek(self.va_park.value + i * self.size)
                     for i in range(self.nh)]
        park_ok = all(all(v == self.fp[i] for v in park_read[i]) for i in range(self.nh))

        ctl_obs = []
        for i, h in enumerate(self.handles):
            slot = self.va_ctl.value + i * self.size
            self._map(slot, h, timed=False)
            flushed = self._flush()
            self._sync()
            o = self._read_observation(slot, self.fp[i], flushed)
            o["handle"] = i
            o["control_va"] = hex(slot)
            ctl_obs.append(o)
            self._unmap(slot, timed=False)
        ctl_copy_ok = all(o["copy_ok"] for o in ctl_obs)
        sh = [o["shader_ok"] for o in ctl_obs if o["shader_ok"] is not None]
        return {
            "test": "cross_va_control",
            "park_readback": park_read,
            "park_distinct_pages_confirmed": park_ok,
            "control_observations": ctl_obs,
            "control_copy_ok": ctl_copy_ok,
            "control_shader_ok": (all(sh) if len(sh) == len(ctl_obs) else None),
            "ok": bool(park_ok and ctl_copy_ok),
            "meaning": "a fresh hipMemMap at a never-used VA binds the requested "
                       "handle and its data is visible through a second VA",
        }

    # -- sub-test 1: minimal single-mapping remap (the clean upstream repro) --
    def minimal_remap(self, reps: int, warmup: int) -> dict:
        """No handle is ever mapped at more than one VA.  Write fingerprints
        through the remap VA itself, then read them back through the same VA.

        NOTE ON READING THE COUNT: if the bug is present, every write in a rep
        lands on the SAME physical page, so the last handle's fingerprint is what
        the page holds and exactly 1 of `handles` read-backs "matches".  A score
        of 1/handles per rep is therefore the signature of a TOTALLY broken
        remap, not of a remap that works a quarter of the time -- see
        `single_page_hypothesis`.
        """
        obs = []
        va = self.va_min.value
        for rep in range(-warmup, reps):
            measured = rep >= 0
            # Every rep (warm-up included) uses a distinct fingerprint namespace so a
            # page left stale by an earlier rep can never be mistaken for a correct read.
            tag = rep + warmup
            fp = [0xA5A50000 | (tag << 8) | i for i in range(self.nh)]
            for i, v in enumerate(fp):
                self._note_fp(v, f"minimal/rep{tag}/handle{i}")
            for i, h in enumerate(self.handles):
                self._map(va, h, timed=measured)
                self._memset32(va, fp[i], self.size)
                self._unmap(va, timed=measured)
            for i, h in enumerate(self.handles):
                self._map(va, h, timed=measured)
                flushed = self._flush()
                self._sync()
                o = self._read_observation(va, fp[i], flushed)
                self._unmap(va, timed=measured)
                if measured:
                    o["rep"] = rep
                    o["rep_tag"] = tag
                    o["handle"] = i
                    obs.append(o)
        out = self._summarise_read(obs, "minimal_remap")
        out["expected_count_if_totally_broken"] = len(obs) // max(1, self.nh)
        out["single_page_hypothesis"] = (
            out["copy_conformant_count"] == out["expected_count_if_totally_broken"]
            and not out["copy_conformant"]
        )
        return out

    # -- sub-test 2/3: parked handles (lets us see WHERE a write landed) -----
    def _park_scan(self) -> list[list[int]]:
        self._flush()
        self._sync()
        return [self._peek(self.va_park.value + i * self.size) for i in range(self.nh)]

    def _park_restore(self) -> None:
        for i in range(self.nh):
            self._memset32(self.va_park.value + i * self.size, self.fp[i], self.size)

    def parked_read(self, reps: int, warmup: int) -> dict:
        obs = []
        va = self.va_min.value
        for rep in range(-warmup, reps):
            measured = rep >= 0
            for i, h in enumerate(self.handles):
                self._map(va, h, timed=False)
                flushed = self._flush()
                self._sync()
                o = self._read_observation(va, self.fp[i], flushed)
                self._unmap(va, timed=False)
                if measured:
                    o["rep"] = rep
                    o["handle"] = i
                    obs.append(o)
        after = [self._peek(self.va_park.value + i * self.size)[0] for i in range(self.nh)]
        out = self._summarise_read(obs, "parked_read")
        out["park_integrity_after"] = after == self.fp
        return out

    def parked_write(self, reps: int, warmup: int) -> dict:
        obs = []
        va = self.va_min.value
        for rep in range(-warmup, reps):
            measured = rep >= 0
            tag = rep + warmup
            for i, h in enumerate(self.handles):
                marker_copy = 0xBEEF0000 | (tag << 8) | i
                marker_shader = 0xFACE0000 | (tag << 8) | i
                self._note_fp(marker_copy, f"write-copy/rep{tag}/handle{i}")
                self._note_fp(marker_shader, f"write-shader/rep{tag}/handle{i}")
                self._map(va, h, timed=False)
                self._memset32(va, marker_copy, self.size)
                sh_dead = None
                sh_ok_arm = self.shader is not None
                if sh_ok_arm:
                    try:
                        self.shader.touch(va, marker_shader, self.n_dw)
                    except ShaderDead as e:
                        self.shader_dead.append({"va": hex(va), "why": str(e)})
                        sh_dead = str(e)
                        sh_ok_arm = False
                # Read the write back through the remap VA itself before unmapping:
                # a self-consistent stale mapping shows the marker here AND on the
                # stale page, which is a different fault from a lost write.
                self._flush()
                self._sync()
                at_va = self._peek(va)
                self._unmap(va, timed=False)
                scan = self._park_scan()
                landed_copy = [j for j, dw in enumerate(scan) if dw[0] == marker_copy]
                landed_shader = (
                    [j for j, dw in enumerate(scan) if dw[1] == marker_shader]
                    if sh_ok_arm
                    else None
                )
                if measured:
                    obs.append(
                        {
                            "rep": rep,
                            "handle": i,
                            "expected_landing": i,
                            "marker_copy_hex": f"0x{marker_copy:08x}",
                            "readback_at_remap_va": at_va,
                            "readback_at_remap_va_ok": all(v == marker_copy for v in at_va),
                            "park_scan": scan,
                            "landed_copy": landed_copy,
                            "landed_shader": landed_shader,
                            "shader_dead": sh_dead,
                            "copy_ok": landed_copy == [i],
                            "shader_ok": (landed_shader == [i]) if sh_ok_arm else None,
                        }
                    )
                self._park_restore()
        n = len(obs)
        copy_ok = sum(1 for o in obs if o["copy_ok"])
        sh_vals = [o["shader_ok"] for o in obs if o["shader_ok"] is not None]
        lost = sum(1 for o in obs if not o["landed_copy"])
        landings = sorted({tuple(o["landed_copy"]) for o in obs})
        return {
            "test": "parked_write",
            "n_observations": n,
            "copy_conformant_count": copy_ok,
            "copy_conformant": n > 0 and copy_ok == n,
            "shader_conformant_count": sum(1 for x in sh_vals if x),
            "shader_conformant": (len(sh_vals) == n and n > 0 and all(sh_vals)),
            "shader_exercised": len(sh_vals) > 0,
            "shader_fully_exercised": n > 0 and len(sh_vals) == n,
            "writes_lost_entirely": lost,
            "distinct_landing_sets": [list(x) for x in landings],
            "observations": obs,
        }

    # -- shared summariser --------------------------------------------------
    def _summarise_read(self, obs: list[dict], name: str) -> dict:
        n = len(obs)
        copy_ok = sum(1 for o in obs if o["copy_ok"])
        sh_vals = [o["shader_ok"] for o in obs if o["shader_ok"] is not None]
        served = [o["served_by"] for o in obs]
        unresolved = sum(1 for s in served if s is None)
        partial = sum(1 for o in obs if not o["copy_offsets_agree"])
        # Group by fingerprint namespace: each rep of minimal_remap writes a FRESH
        # set of fingerprints, so "one physical page served throughout" shows up as
        # one value PER REP, not one value overall.  Comparing raw values across
        # reps would classify the canonical broken case as `mixed_or_unclassified`.
        groups: dict[object, list[dict]] = {}
        for o in obs:
            groups.setdefault(o.get("rep_tag", "all"), []).append(o)
        group_vals = {
            k: sorted({x["observed_copy"] for x in g}) for k, g in groups.items()
        }
        all_groups_single = bool(groups) and all(len(v) == 1 for v in group_vals.values())
        if n == 0:
            pattern = "no_observations"
        elif copy_ok == n:
            pattern = "conformant"
        elif partial:
            pattern = (
                f"partially_remapped ({partial}/{n} reads disagreed ACROSS OFFSETS "
                "within one handle -- part of the range moved and part did not)"
            )
        elif all_groups_single:
            k0 = next(iter(groups))
            val = group_vals[k0][0]
            who = groups[k0][0]["served_by"] or "an unrecorded writer"
            pattern = (
                "always_serves_a_single_physical_page (within every fingerprint "
                f"namespace all {len(groups[k0])} remap reads returned one value; "
                f"e.g. namespace {k0} read 0x{val:08x}, which was written by {who} "
                "-- i.e. the LAST write into the one page the VA is stuck on)"
            )
        elif unresolved == n:
            pattern = "serves_unrecognised_data (no recorded write produced these values)"
        else:
            pattern = "mixed_or_unclassified"
        flushed = sum(1 for o in obs if o.get("cache_flushed_before_read"))
        return {
            "values_per_fingerprint_namespace": {
                str(k): [f"0x{x:08x}" for x in v] for k, v in group_vals.items()
            },
            "one_value_per_fingerprint_namespace": all_groups_single,
            "test": name,
            "n_observations": n,
            "copy_conformant_count": copy_ok,
            "copy_conformant": n > 0 and copy_ok == n,
            "shader_conformant_count": sum(1 for x in sh_vals if x),
            "shader_conformant": (len(sh_vals) == n and n > 0 and all(sh_vals)),
            "shader_exercised": len(sh_vals) > 0,
            "shader_fully_exercised": n > 0 and len(sh_vals) == n,
            "served_handle_sequence": served,
            "unresolved_reads": unresolved,
            "reads_with_offset_disagreement": partial,
            "reads_taken_after_cache_flush": flushed,
            "stale_pattern": pattern,
            "observations": obs,
        }


def run_variant(hip: Hip, dev: int, location: str, size: int, nh: int, reps: int,
                warmup: int, shader: ShaderPath | None,
                flusher: "CacheFlusher | None" = None) -> dict:
    """Run one location.  Sub-tests are staged so an early failure never discards
    a result that had already been measured independently of it."""
    vp = VariantProbe(hip, dev, location, size, nh, shader, flusher)
    res: dict = {
        "location": location,
        "supported": False,
        "error": None,
        "stages_completed": [],
        "complete": False,
        "control_ok": None,
    }
    try:
        res.update(vp.setup())
        res["supported"] = True

        # 1. minimal repro first: it must not depend on anything below it, and it
        #    must survive a later precondition failure.
        res["minimal_remap"] = vp.minimal_remap(reps, warmup)
        res["stages_completed"].append("minimal_remap")

        # 2. CONTROL: does a fresh map bind the requested handle at all?
        res["control"] = vp.cross_va_control()
        res["park"] = {
            "fingerprints": vp.fp,
            "readback": res["control"]["park_readback"],
            "distinct_pages_confirmed": res["control"]["park_distinct_pages_confirmed"],
        }
        res["control_ok"] = res["control"]["ok"]
        res["stages_completed"].append("cross_va_control")
        if not res["control"]["ok"]:
            raise RuntimeError(
                "PRECONDITION: the control failed -- either the parked handles at "
                "distinct VAs do not read back their own distinct fingerprints, or "
                "a handle read through a THIRD virgin VA does not return its own "
                "data. The probe cannot distinguish 'stale after remap' from "
                "'hipMemMap never bound the requested handle' -- refusing to report "
                "a conformance verdict for the parked sub-tests."
            )

        res["parked_read"] = vp.parked_read(reps, warmup)
        res["stages_completed"].append("parked_read")
        res["parked_write"] = vp.parked_write(reps, warmup)
        res["stages_completed"].append("parked_write")

        reads = [res["minimal_remap"], res["parked_read"]]
        res["read_conformant"] = all(r["copy_conformant"] for r in reads)
        res["read_conformant_shader"] = (
            all(r["shader_conformant"] for r in reads)
            if all(r["shader_fully_exercised"] for r in reads)
            else None
        )
        res["write_conformant"] = res["parked_write"]["copy_conformant"]
        res["write_conformant_shader"] = (
            res["parked_write"]["shader_conformant"]
            if res["parked_write"]["shader_fully_exercised"]
            else None
        )
        res["conformant"] = bool(res["read_conformant"] and res["write_conformant"])
        res["complete"] = True
    except HipError as e:
        res["error"] = {"type": "HipError", "call": e.what, "rc": e.rc, "msg": e.msg}
    except Exception as e:
        res["error"] = {"type": type(e).__name__, "msg": str(e)}
    finally:
        res["timing_us"] = {
            "map": stats_us(vp.t_map),
            "set_access": stats_us(vp.t_setaccess),
            "unmap": stats_us(vp.t_unmap),
            "full_cycle_map_setaccess_unmap": stats_us(vp.t_cycle),
            "note": "minimal_remap sub-test only (2 x handles x reps samples: one "
                    "write pass + one read pass); warm-up reps discarded; device "
                    "quiesced with hipDeviceSynchronize before each unmap timer",
        }
        res["nonzero_hip_return_codes"] = vp.nonzero_rcs
        res["all_calls_returned_hipSuccess"] = not vp.nonzero_rcs
        res["shader_dead_launches"] = vp.shader_dead
        res["fingerprint_ledger"] = {f"0x{k:08x}": v for k, v in sorted(vp.fp_ledger.items())}
        res["cache_flush"] = flusher.info() if flusher is not None else None
        # `minimal_remap` is a standalone upstream repro: keep its verdict even
        # when a later stage's precondition failed.
        mr = res.get("minimal_remap")
        res["minimal_only_verdict"] = (
            {
                "copy_conformant": mr["copy_conformant"],
                "shader_conformant": (mr["shader_conformant"]
                                      if mr["shader_fully_exercised"] else None),
                "stale_pattern": mr["stale_pattern"],
                "single_page_hypothesis": mr.get("single_page_hypothesis"),
            }
            if mr else None
        )
        vp.teardown()
    return res

# ---------------------------------------------------------------------------
# payload assembly / validation
# ---------------------------------------------------------------------------

REQUIRED_TOP = [
    "probe", "probe_name", "schema_version", "synthetic", "status", "verdict",
    "params", "env", "versions", "device", "box_state", "box_state_after",
    "box_state_delta", "variants", "shader_path", "cache_flush", "notes",
    "repro", "exit_code",
]
REQUIRED_VERDICT = [
    "primary_location", "read_conformant", "write_conformant",
    "dynamic_residency_available", "expected", "matched_expectation",
    "control_ok", "shader_arm_trustworthy",
]
# Statuses that carry no conformance measurement.  These never write the canonical
# p6.json -- a failed run must not silently overwrite a good one.
NON_MEASUREMENT_STATUSES = {"PRECONDITION_FAILED", "SELFTEST"}


def validate_payload(p: dict) -> list[str]:
    """Structural validation shared by the real and --selftest paths."""
    errs = []
    for k in REQUIRED_TOP:
        if k not in p:
            errs.append(f"missing top-level key: {k}")
    if p.get("probe") != PROBE_ID:
        errs.append("probe id mismatch")
    if not isinstance(p.get("schema_version"), int):
        errs.append("schema_version must be int")
    v = p.get("verdict", {})
    if not isinstance(v, dict):
        errs.append("verdict must be a dict")
    else:
        for k in REQUIRED_VERDICT:
            if k not in v:
                errs.append(f"missing verdict key: {k}")
    if p.get("status") not in {
        "BROKEN_STALE", "CONFORMANT", "MIXED", "PRECONDITION_FAILED", "SELFTEST",
    }:
        errs.append(f"bad status: {p.get('status')}")
    for which in ("box_state", "box_state_after"):
        bs = p.get(which) or {}
        for k in ("meminfo_kb", "vmstat", "drm_cards", "kfd_nodes"):
            if k not in bs:
                errs.append(f"missing {which}.{k}")
    for k in ("kernel_release", "rocm_info_version", "libamdhip64_realpath"):
        if k not in p.get("versions", {}):
            errs.append(f"missing versions.{k}")
    dev = p.get("device", {})
    if not p.get("synthetic") and p.get("status") != "PRECONDITION_FAILED":
        for k in ("hip_index", "pci_slot", "name", "card_label"):
            if k not in dev:
                errs.append(f"missing device.{k}")
        if not isinstance(p.get("variants"), list) or not p["variants"]:
            errs.append("variants must be a non-empty list")
    try:
        json.dumps(p)
    except (TypeError, ValueError) as e:
        errs.append(f"payload is not JSON-serialisable: {e}")
    return errs


NOTES = [
    "A 'stale' read means hipMemUnmap+hipMemMap left the old physical page behind "
    "the VA while every HIP call returned hipSuccess -- a silent-wrong-data class.",
    "TWO ACCESS PATHS, NOT TWO ENGINES: hipMemcpy/hipMemsetD32 is the HIP runtime's "
    "copy path and may itself be implemented as a blit KERNEL rather than SDMA, so "
    "it must NOT be assumed to bypass GPU caches. Localisation (page table vs cache) "
    "therefore rests on the explicit cache-flush arm, not on an assumed engine "
    "difference; the shader arm still matters independently because it is the path "
    "the offload design actually reads weights through.",
    "CACHE FLUSH: the working set (handles x size, plus the parked copy) fits inside "
    "the 64 MB MALL, so a read taken straight after a remap could be served by a "
    "cached line from the OLD mapping. Every read-back is preceded by streaming a "
    ">=4x-MALL scratch buffer; `reads_taken_after_cache_flush` records how many "
    "observations actually got one. A stale result that survives the flush is a "
    "page-table result, not a cache result.",
    "CONTROL FIRST: `cross_va_control` writes each handle through its own permanent "
    "VA and reads it back through a THIRD, never-reused VA. Only if that passes can "
    "a later stale read be attributed to the remap rather than to 'hipMemMap never "
    "bound the requested handle'. A control failure suppresses the parked verdict.",
    "The minimal_remap sub-test never maps a handle at more than one VA, so its "
    "result stands alone as an upstream repro and is retained even when the control "
    "fails (`minimal_only_verdict`). parked_read/parked_write additionally keep each "
    "handle mapped at its own VA, which is the only way to observe WHICH physical "
    "page a write actually landed on.",
    "READING minimal_remap's COUNT: when the bug is present every write in a rep "
    "lands on the same page, so exactly 1 of `handles` read-backs matches and the "
    "score is 1/handles per rep. That is the signature of a TOTALLY broken remap, "
    "not of one that works part of the time -- see `single_page_hypothesis`.",
    "SHADER LIVENESS: every kernel launch poisons its output buffer first and the "
    "kernel writes a liveness token; a launch that returns hipSuccess without "
    "executing is recorded as `shader_dead`, never scored as non-conformant. The "
    "shader arm is also proven read+write correct against ordinary hipMalloc memory "
    "(`shader_path.selfcheck`) before any observation -- otherwise a broken shader "
    "would hold `dynamic_residency_available` at False forever and the tripwire "
    "could never fire.",
    "Granularity is recorded as a QUERY and never asserted on: a capability query "
    "can succeed with a fabricated value while the operation it describes fails. "
    "Every verdict here rests on observed DATA.",
    "This probe is a tripwire, never a shipped code path. If it flips to CONFORMANT, "
    "dynamic page-level residency (WEIGHT_OFFLOAD_PLAN tier T3) becomes reachable.",
]

REPRO = {
    "summary": "hipMemUnmap(va, sz) then hipMemMap(va, sz, 0, other_handle, 0) + "
               "hipMemSetAccess -> reads and writes at `va` are served by the "
               "FIRST handle ever mapped there; all calls return hipSuccess.",
    "minimal_sequence": [
        "hipMemAddressReserve(&va, SZ, 2MiB, 0, 0)",
        "hipMemCreate(&h[i], SZ, {type=Pinned, location={Device,0}}, 0)  for i in 0..3",
        "for i: hipMemMap(va,SZ,0,h[i],0); hipMemSetAccess(va,SZ,RW,1); "
        "hipMemsetD32(va, 0xA5A5_0000|i, SZ/4); hipMemUnmap(va,SZ)",
        "for i: hipMemMap(va,SZ,0,h[i],0); hipMemSetAccess(va,SZ,RW,1); "
        "hipMemcpy(&x,va,4,D2H); hipMemUnmap(va,SZ)   -> x should be 0xA5A5_0000|i",
    ],
    "expected_by_spec": "x == 0xA5A50000|i for every i",
    "observed_on_this_box": "see variants[].minimal_remap.stale_pattern",
}


def box_state_delta(before: dict, after: dict) -> dict:
    """What the box did to itself while the probe ran.

    A capacity/latency number taken on a box that swapped or lost VRAM to another
    tenant mid-run is not the number it claims to be; record the movement rather
    than assuming there was none.
    """
    out: dict = {}
    b_vm, a_vm = (before.get("vmstat") or {}), (after.get("vmstat") or {})
    for k in ("pswpin", "pswpout", "pgmajfault"):
        if b_vm.get(k) is not None and a_vm.get(k) is not None:
            out[f"{k}_delta"] = a_vm[k] - b_vm[k]
    b_mi, a_mi = (before.get("meminfo_kb") or {}), (after.get("meminfo_kb") or {})
    for k in ("MemAvailable", "MemFree"):
        if b_mi.get(k) is not None and a_mi.get(k) is not None:
            out[f"{k}_delta_kb"] = a_mi[k] - b_mi[k]
    b_c = {c["pci_slot"]: c for c in (before.get("drm_cards") or [])}
    a_c = {c["pci_slot"]: c for c in (after.get("drm_cards") or [])}
    out["vram_used_delta_bytes"] = {
        slot: (a_c[slot].get("vram_used_bytes") or 0) - (b_c[slot].get("vram_used_bytes") or 0)
        for slot in a_c
        if slot in b_c
    }
    out["swapped_during_run"] = bool(out.get("pswpout_delta"))
    out["elapsed_s"] = round(
        (after.get("uptime_s") or 0) - (before.get("uptime_s") or 0), 3
    )
    return out


def build_payload(args, box_state, versions, device, variants, shader_info,
                  status, verdict, synthetic=False, error=None,
                  box_state_after=None, cache_flush=None) -> dict:
    after = box_state_after if box_state_after is not None else box_state
    return {
        "probe": PROBE_ID,
        "probe_name": PROBE_NAME,
        "schema_version": SCHEMA_VERSION,
        "synthetic": synthetic,
        "tag": args.tag,
        "status": status,
        "verdict": verdict,
        "params": {
            "size_bytes": args.size_mib << 20,
            "size_mib": args.size_mib,
            "handles": args.handles,
            "reps_measured": args.reps,
            "warmup_reps_discarded": args.warmup,
            "device_index": args.device,
            "locations_requested": args.location,
            "shader_path_requested": not args.no_shader,
            "cache_flush_mib": args.cache_flush_mib,
            "allow_unknown_card": args.allow_unknown_card,
            "expect": args.expect,
            "argv": sys.argv,
            "cwd": os.getcwd(),
        },
        "env": _ENV_PIN
        | {
            "effective_rocr_visible_devices": os.environ.get("ROCR_VISIBLE_DEVICES"),
            "effective_hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
        },
        "versions": versions,
        "device": device,
        "box_state": box_state,
        "box_state_after": after,
        "box_state_delta": box_state_delta(box_state, after),
        "variants": variants,
        "shader_path": shader_info,
        "cache_flush": cache_flush,
        "notes": NOTES,
        "repro": REPRO,
        "error": error,
        "exit_code": None,  # filled by main
    }


def resolve_out_path(out_json: Path, status: str) -> Path:
    """A run that measured nothing must never land on the canonical filename.

    Overwriting a good p6.json with a PRECONDITION_FAILED payload (whose Verdict
    block is all `None`) destroys the measurement and reads as a result.
    """
    if status in NON_MEASUREMENT_STATUSES:
        return out_json.with_name(f"{out_json.stem}.{status}{out_json.suffix}")
    return out_json


def write_outputs(payload: dict, out_json: Path, write_md: bool) -> list[Path]:
    out_json = resolve_out_path(out_json, payload.get("status", ""))
    payload["output_json_path"] = str(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    written = []
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for path in (out_json, out_json.with_name(f"{out_json.stem}_{ts}{out_json.suffix}")):
        path.write_text(json.dumps(payload, indent=2) + "\n")
        written.append(path)
    if write_md:
        md = out_json.with_suffix(".md")
        md.write_text(render_md(payload))
        written.append(md)
    return written


def render_md(p: dict) -> str:
    v = p["verdict"]
    d = p.get("device") or {}
    bs = p.get("box_state") or {}
    ver = p.get("versions") or {}
    sh = p.get("shader_path") or {}
    cf = p.get("cache_flush") or {}
    lines = [
        f"# P6 -- hipMemUnmap/hipMemMap conformance ({p['status']})",
        "",
        f"*Generated {bs.get('timestamp_utc')} by `tools/offload/p6_vmm_remap_conformance.py` "
        f"(schema {p['schema_version']}).*",
        "",
    ]
    if p.get("status") in NON_MEASUREMENT_STATUSES:
        lines += [
            f"> **NOT A MEASUREMENT -- `{p['status']}`.** No conformance number was "
            "produced; every `read/write conformant` field below is `None` by "
            "construction. Do not cite this file.",
            "",
        ]
    if p.get("synthetic"):
        lines += ["> **SYNTHETIC PAYLOAD (selftest) -- DO NOT CITE.**", ""]
    lines += [
        "## Verdict",
        "",
        f"- **status:** `{p['status']}`",
        f"- read conformant: `{v['read_conformant']}`  (shader: `{v.get('read_conformant_shader')}`)",
        f"- write conformant: `{v['write_conformant']}`  (shader: `{v.get('write_conformant_shader')}`)",
        f"- control (fresh map binds the requested handle): `{v.get('control_ok')}`",
        f"- shader arm trustworthy (selfchecked, no dead launches): "
        f"`{v.get('shader_arm_trustworthy')}` "
        f"(variant `{sh.get('variant')}`, available `{sh.get('available')}`)",
        f"- cache flush per read-back: `{cf.get('available')}` ({cf.get('mib')} MiB "
        f"x {cf.get('flushes')} flushes)",
        f"- **dynamic page-level residency (T3) available: `{v['dynamic_residency_available']}`**",
        f"- expectation `{v['expected']}` matched: `{v['matched_expectation']}`",
        "",
        "## Card / stack",
        "",
        f"- card: **{d.get('card_label')}** (HIP index {d.get('hip_index')}, "
        f"pci `{d.get('pci_slot')}`, arch `{d.get('gfx_arch')}`, simd_count {d.get('simd_count')})",
        f"- ROCm `{ver.get('rocm_info_version')}`, HIP runtime `{ver.get('hip_runtime_version')}`, "
        f"driver `{ver.get('hip_driver_version')}`",
        f"- libamdhip64: `{ver.get('libamdhip64_realpath')}`",
        f"- kernel: `{ver.get('kernel_release')}`",
        "",
        "## Box state at measurement",
        "",
        f"- RAM: {bs.get('mem_used_gib')} GiB used / {bs.get('mem_total_gib')} GiB total, "
        f"{bs.get('mem_available_gib')} GiB available",
        f"- vmstat pswpout: {(bs.get('vmstat') or {}).get('pswpout')}, "
        f"pswpin: {(bs.get('vmstat') or {}).get('pswpin')}",
        f"- loadavg: `{bs.get('loadavg')}`",
        f"- during the run: `{json.dumps(p.get('box_state_delta') or {}, sort_keys=True)}`",
        "",
        "## Per-location results",
        "",
        "| location | control | minimal read | parked read | write landing | stale pattern | all rc==hipSuccess |",
        "|---|---|---|---|---|---|---|",
    ]
    for var in p.get("variants") or []:
        mr = var.get("minimal_remap")
        if not var.get("complete"):
            mr_cell = (
                f"{mr['copy_conformant_count']}/{mr['n_observations']}"
                if mr else "not reached"
            )
            lines.append(
                f"| {var.get('location')} | `{var.get('control_ok')}` | {mr_cell} "
                f"| INCOMPLETE | INCOMPLETE "
                f"| `{(mr or {}).get('stale_pattern', 'n/a')}` "
                f"| {var.get('all_calls_returned_hipSuccess')} |"
            )
            lines.append(
                f"| | | | | | ERROR: `{var.get('error')}` | |"
            )
            continue
        pr, pw = var["parked_read"], var["parked_write"]
        lines.append(
            f"| {var['location']} "
            f"| `{var.get('control_ok')}` "
            f"| {mr['copy_conformant_count']}/{mr['n_observations']} "
            f"| {pr['copy_conformant_count']}/{pr['n_observations']} "
            f"| {pw['copy_conformant_count']}/{pw['n_observations']} "
            f"| `{mr['stale_pattern']}` "
            f"| {var['all_calls_returned_hipSuccess']} |"
        )
    lines += [
        "",
        "`minimal read` of 1/N per rep is the signature of a TOTALLY broken remap "
        "(all writes in a rep land on one page, so the last handle's fingerprint is "
        "what the page holds) -- see `single_page_hypothesis` in the JSON, not a "
        "remap that works part of the time.",
    ]
    lines += ["", "## Remap cycle cost (minimal sub-test, warm-up discarded)", "",
              "| location | map | setAccess | unmap | full cycle |", "|---|---|---|---|---|"]
    for var in p.get("variants") or []:
        t = var.get("timing_us")
        if not t:
            continue

        def f(k):
            s = t.get(k)
            return f"{s['median_us']:.1f} us (n={s['n']}, {s['min_us']:.1f}-{s['max_us']:.1f})" if s else "n/a"

        lines.append(
            f"| {var['location']} | {f('map')} | {f('set_access')} | {f('unmap')} "
            f"| {f('full_cycle_map_setaccess_unmap')} |"
        )
    lines += ["", "## Notes", ""] + [f"- {n}" for n in p["notes"]]
    lines += ["", "## Upstream repro", "", "```", p["repro"]["summary"], "```", ""]
    lines += [f"{i+1}. `{s}`" for i, s in enumerate(p["repro"]["minimal_sequence"])]
    raw = p.get("output_json_path") or "(not yet written)"
    lines += ["", f"Raw JSON: `{raw}`", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="P6: hipMemUnmap/hipMemMap conformance tripwire (torch-free ctypes).",
    )
    ap.add_argument("--device", type=int, default=0,
                    help="HIP device index (0 = first card visible after ROCR pin)")
    ap.add_argument("--size-mib", type=int, default=8, help="bytes per handle, MiB")
    ap.add_argument("--handles", type=int, default=4, help="distinct physical handles")
    ap.add_argument("--reps", type=int, default=5, help="measured reps (>=5 required)")
    ap.add_argument("--warmup", type=int, default=1, help="discarded warm-up reps (>=1)")
    ap.add_argument("--location", choices=["device", "host", "both"], default="both",
                    help="physical location of the hipMemCreate handles")
    ap.add_argument("--expect", choices=["broken", "conformant", "any"], default="any",
                    help="exit 3 if the measured verdict contradicts this (CI: broken)")
    ap.add_argument("--no-shader", action="store_true",
                    help="skip the hipRTC kernel path (runtime copy path only)")
    ap.add_argument("--cache-flush-mib", type=int, default=256,
                    help="scratch streamed before every read-back to evict L2+MALL "
                         "(default 256 = 4x the 64 MB MALL; 0 disables and the JSON "
                         "records that the stale/cache ambiguity was not resolved)")
    ap.add_argument("--allow-unknown-card", action="store_true",
                    help="permit a run on a card that is not one of the two known "
                         "discrete gfx1201 boards (default: refuse -- a verdict from "
                         "the Ryzen iGPU would be meaningless)")
    ap.add_argument("--arch", default=None,
                    help="override the gfx arch used for hipRTC (default: from KFD topology)")
    ap.add_argument("--hip-lib", default=None, help="path to libamdhip64.so")
    ap.add_argument("--hiprtc-lib", default=None, help="path to libhiprtc.so")
    ap.add_argument("--out", default=str(DEFAULT_OUT_DIR / "p6.json"),
                    help="JSON output path (a timestamped copy is written alongside)")
    ap.add_argument("--no-md", action="store_true", help="do not write the companion .md")
    ap.add_argument("--tag", default=None, help="free-form label recorded in the JSON")
    ap.add_argument("--selftest", "--dry-run", dest="selftest", action="store_true",
                    help="validate args/JSON shape AND drive the measurement code "
                         "against a mock HIP driver, without touching the GPU")
    ap.add_argument("--no-mock", action="store_true",
                    help="selftest only: skip the mock-driver verdict-logic test")
    return ap.parse_args(argv)


def expectation_matched(status: str, dynamic_ok: bool, expect: str) -> bool:
    """Tripwire contract: `--expect broken` fails (exit 3) the moment the bug is fixed."""
    if expect == "any":
        return True
    if expect == "broken":
        return status == "BROKEN_STALE"
    return bool(dynamic_ok)


def die(msg: str, code: int = 2) -> int:
    print(f"\nP6 FATAL: {msg}\n", file=sys.stderr)
    return code


def validate_args(args) -> str | None:
    if args.reps < 5:
        return "--reps must be >= 5 (the probe reports a median and spread)"
    if args.warmup < 1:
        return "--warmup must be >= 1 (the first remap cycle is never steady state)"
    if args.handles < 2:
        return "--handles must be >= 2 (a remap needs two distinct physical pages)"
    if args.reps + args.warmup > 255:
        return "--reps + --warmup must be <= 255 (the fingerprint encodes the rep in 8 bits)"
    if args.handles > 255:
        return "--handles must be <= 255 (the fingerprint encodes the handle in 8 bits)"
    if args.size_mib < 1:
        return "--size-mib must be >= 1"
    if args.size_mib < 1 or (args.size_mib << 20) % 4096:
        return "--size-mib must cover a whole number of 4096 B VMM granules"
    if args.device < 0:
        return "--device must be >= 0"
    if args.cache_flush_mib < 0:
        return "--cache-flush-mib must be >= 0"
    if 0 < args.cache_flush_mib < 256:
        return ("--cache-flush-mib must be 0 (disabled, recorded as such) or >= 256 "
                "-- anything smaller does not evict the 64 MB MALL and would let a "
                "cached line from the OLD mapping be reported as a stale page")
    return None


def selftest(args) -> int:
    print("P6 SELFTEST: no GPU is touched.")
    bad = validate_args(args)
    if bad:
        return die(f"argument validation: {bad}")
    for bogus in (["--reps", "1"], ["--warmup", "0"], ["--handles", "1"],
                  ["--reps", "300"], ["--handles", "9999"], ["--size-mib", "0"],
                  ["--cache-flush-mib", "16"], ["--cache-flush-mib", "-1"]):
        a = parse_args(bogus)
        if validate_args(a) is None:
            return die(f"argument validation failed to reject {bogus}")
    if validate_args(parse_args(["--cache-flush-mib", "0"])) is not None:
        return die("--cache-flush-mib 0 must be allowed (explicitly-disabled arm)")
    print("  arg validation ..................... OK (rejects reps<5, warmup<1, "
          "handles<2, sub-MALL cache flush)")

    box = collect_box_state(include_smi=False)
    ver = collect_versions(None)
    nodes = box["kfd_nodes"]
    if not nodes:
        print("  WARNING: no KFD topology nodes visible; arch autodetect would fail")
    discrete = [n for n in nodes if n["pci_slot"] in _CARD_LABELS]
    arch = args.arch or ((discrete or nodes)[0]["gfx_arch"] if nodes else None)
    print(f"  box state .......................... OK "
          f"(RAM {box.get('mem_used_gib')}/{box.get('mem_total_gib')} GiB used, "
          f"{len(box['drm_cards'])} drm cards, {len(nodes)} kfd gpu nodes)")
    print(f"  versions ........................... OK (ROCm {ver['rocm_info_version']}, "
          f"kernel {ver['kernel_release']}, hip lib {ver['libamdhip64_realpath']})")
    print(f"  arch autodetect .................... {arch}")
    print(f"  env pin ............................ "
          f"ROCR_VISIBLE_DEVICES={os.environ.get('ROCR_VISIBLE_DEVICES')!r} "
          f"HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES')!r}")

    # A structurally complete, explicitly SYNTHETIC payload: shape only, no numbers
    # that could be mistaken for a measurement.
    def _empty_read(name):
        return {
            "test": name, "n_observations": 0, "copy_conformant_count": 0,
            "copy_conformant": False, "shader_conformant_count": 0,
            "shader_conformant": False, "shader_exercised": False,
            "shader_fully_exercised": False, "served_handle_sequence": [],
            "unresolved_reads": 0, "reads_with_offset_disagreement": 0,
            "reads_taken_after_cache_flush": 0,
            "values_per_fingerprint_namespace": {},
            "one_value_per_fingerprint_namespace": False,
            "stale_pattern": "not-measured (selftest)", "observations": [],
        }

    fake_variant = {
        "location": "device",
        "supported": True,
        "complete": True,
        "SYNTHETIC": True,
        "stages_completed": [],
        "granularity_query_minimum": None,
        "granularity_query_recommended": None,
        "remap_va": None,
        "park_va_base": None,
        "control_va_base": None,
        "minimal_remap": _empty_read("minimal_remap"),
        "minimal_only_verdict": None,
        "control": {"test": "cross_va_control", "park_readback": [],
                    "park_distinct_pages_confirmed": False,
                    "control_observations": [], "control_copy_ok": False,
                    "control_shader_ok": None, "ok": False,
                    "meaning": "not measured (selftest)"},
        "control_ok": None,
        "park": {"fingerprints": [], "readback": [], "distinct_pages_confirmed": False},
        "parked_read": dict(_empty_read("parked_read"), park_integrity_after=False),
        "parked_write": {
            "test": "parked_write",
            "n_observations": 0, "copy_conformant_count": 0, "copy_conformant": False,
            "shader_conformant_count": 0, "shader_conformant": False,
            "shader_exercised": False, "shader_fully_exercised": False,
            "writes_lost_entirely": 0,
            "distinct_landing_sets": [], "observations": [],
        },
        "timing_us": {"map": None, "set_access": None, "unmap": None,
                      "full_cycle_map_setaccess_unmap": None,
                      "note": "not measured (selftest)"},
        "nonzero_hip_return_codes": [],
        "all_calls_returned_hipSuccess": False,
        "shader_dead_launches": [],
        "fingerprint_ledger": {},
        "cache_flush": None,
        "read_conformant": False, "read_conformant_shader": None,
        "write_conformant": False, "write_conformant_shader": None,
        "conformant": False,
    }
    verdict = {
        "primary_location": "device",
        "read_conformant": None, "read_conformant_shader": None,
        "write_conformant": None, "write_conformant_shader": None,
        "control_ok": None, "shader_arm_trustworthy": None,
        "stale_survives_cache_flush": None,
        "dynamic_residency_available": None,
        "expected": args.expect, "matched_expectation": None,
        "explanation": "selftest payload -- nothing was measured",
    }
    box_after = collect_box_state(include_smi=False)
    payload = build_payload(
        args, box, ver,
        {"hip_index": None, "pci_slot": None, "name": None, "card_label": None,
         "gfx_arch": arch, "simd_count": None, "total_mem_bytes": None},
        [fake_variant],
        {"available": False, "verified": False, "variant": None, "reason": "selftest"},
        "SELFTEST", verdict, synthetic=True,
        box_state_after=box_after,
        cache_flush={"available": False, "reason": "selftest", "bytes": 0, "mib": 0,
                     "flushes": 0},
    )
    payload["exit_code"] = 0
    errs = validate_payload(payload)
    if errs:
        return die("JSON schema validation failed:\n  - " + "\n  - ".join(errs))
    print(f"  json schema ........................ OK ({len(json.dumps(payload))} bytes, "
          f"{len(REQUIRED_TOP)} required keys present)")

    # Prove the markdown renderer works on the same shape.
    md = render_md(payload)
    if "P6 --" not in md:
        return die("markdown renderer produced no header")
    print(f"  markdown renderer .................. OK ({len(md.splitlines())} lines)")

    # Selftest never writes p6.json -- it must not be mistakable for a measurement.
    out = Path(args.out)
    st_json = out.with_name(out.stem + ".selftest.json")
    st_json.parent.mkdir(parents=True, exist_ok=True)
    st_json.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"  wrote selftest payload ............. {st_json}")
    print(f"  (the real run writes {out} -- untouched by selftest)")
    stats_probe = stats_us([100_000, 90_000, 95_000, 99_000, 71_700, 120_000])
    if not stats_probe or stats_probe["n"] != 6 or stats_us([]) is not None:
        return die("stats_us() is misbehaving")
    print(f"  stats helper ....................... OK (median {stats_probe['median_us']} us on a fixture)")

    cases = [
        ("BROKEN_STALE", False, "broken", True), ("BROKEN_STALE", False, "any", True),
        ("BROKEN_STALE", False, "conformant", False), ("CONFORMANT", True, "broken", False),
        ("CONFORMANT", True, "conformant", True), ("CONFORMANT", False, "conformant", False),
        ("MIXED", False, "broken", False), ("MIXED", False, "any", True),
    ]
    for st, dyn, exp, want in cases:
        got = expectation_matched(st, dyn, exp)
        if got is not want:
            return die(f"expectation_matched({st},{dyn},{exp}) = {got}, expected {want}")
    print(f"  tripwire exit-code contract ........ OK ({len(cases)} cases; "
          "`--expect broken` -> exit 3 the moment the bug is fixed)")

    fp_ns = {(r + 1) << 8 | i for r in range(-1, 5) for i in range(4)}
    if len(fp_ns) != 6 * 4:
        return die("fingerprint namespace collides across reps/handles")
    # ...and across the three marker namespaces used by the write arm.
    all_markers = (
        {0xA5A50000 | (t << 8) | i for t in range(6) for i in range(4)}
        | {0xBEEF0000 | (t << 8) | i for t in range(6) for i in range(4)}
        | {0xFACE0000 | (t << 8) | i for t in range(6) for i in range(4)}
        | {0xC0DE0000 | i for i in range(4)}
    )
    if len(all_markers) != 3 * 24 + 4:
        return die("fingerprint namespaces collide ACROSS sub-tests")
    print("  fingerprint namespace .............. OK (distinct per rep, handle AND "
          "sub-test)")

    # A non-measurement payload must never land on the canonical filename.
    out = Path(args.out)
    for st in ("PRECONDITION_FAILED", "SELFTEST"):
        if resolve_out_path(out, st) == out:
            return die(f"status {st} would overwrite the canonical {out.name}")
    for st in ("BROKEN_STALE", "CONFORMANT", "MIXED"):
        if resolve_out_path(out, st) != out:
            return die(f"status {st} must write the canonical {out.name}")
    print(f"  output-path guard .................. OK (PRECONDITION_FAILED/SELFTEST "
          f"cannot overwrite {out.name})")

    # The markdown for a non-measurement status must say so before any number.
    pf = dict(payload, status="PRECONDITION_FAILED")
    md_pf = render_md(pf)
    if "NOT A MEASUREMENT" not in md_pf.split("## Verdict")[0]:
        return die("PRECONDITION_FAILED markdown lacks an INVALID banner above the verdict")
    if "DO NOT CITE" not in md.split("## Verdict")[0]:
        return die("synthetic markdown lacks a DO-NOT-CITE banner above the verdict")
    print("  invalid-result banners ............. OK (INVALID/DO-NOT-CITE precede "
          "every conformance field)")

    # The tripwire must not be able to go green off the copy path alone.
    if expectation_matched("CONFORMANT", False, "conformant"):
        return die("a copy-path-only pass must not satisfy --expect conformant")
    print("  shader gate ........................ OK (dynamic_residency_available "
          "requires read AND write AND a trustworthy shader arm)")

    d = box_state_delta(
        {"vmstat": {"pswpout": 10, "pswpin": 1, "pgmajfault": 0},
         "meminfo_kb": {"MemAvailable": 100, "MemFree": 50}, "uptime_s": 5.0,
         "drm_cards": [{"pci_slot": "x", "vram_used_bytes": 1}]},
        {"vmstat": {"pswpout": 42, "pswpin": 1, "pgmajfault": 0},
         "meminfo_kb": {"MemAvailable": 60, "MemFree": 40}, "uptime_s": 9.0,
         "drm_cards": [{"pci_slot": "x", "vram_used_bytes": 5}]},
    )
    if (d["pswpout_delta"], d["swapped_during_run"], d["MemAvailable_delta_kb"],
            d["vram_used_delta_bytes"]["x"], d["elapsed_s"]) != (32, True, -40, 4, 4.0):
        return die(f"box_state_delta is misbehaving: {d}")
    print("  box-state delta .................... OK (swap/VRAM movement during the "
          "run is recorded, not assumed absent)")

    # The checks above prove the JSON SHAPE.  They cannot prove the probe would
    # reach the right VERDICT -- which for a tripwire is the whole point.  Drive
    # the real measurement code against a mock HIP driver (no GPU) that simulates
    # a conformant remap, this box's stale remap, a control failure, and a
    # partial remap, and require the expected verdict from each.
    if not args.no_mock:
        try:
            import p6_mock_driver_test as mock  # noqa: PLC0415
        except Exception:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            try:
                import p6_mock_driver_test as mock  # noqa: PLC0415
            except Exception as e:
                return die(f"cannot import the mock-driver logic test: {e}")
        print("\n  --- mock-driver verdict logic (no GPU) ---")
        if mock.main() != 0:
            return die("mock-driver logic test FAILED: the probe would not reach the "
                       "correct verdict on a simulated driver")

    print("\nP6 SELFTEST: PASS\n")
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.selftest:
        return selftest(args)

    bad = validate_args(args)
    if bad:
        return die(f"argument validation: {bad}")

    # ---- preconditions ----------------------------------------------------
    try:
        hip = Hip(args.hip_lib)
    except Exception as e:
        return die(f"cannot load libamdhip64.so: {e}")

    ndev = ctypes.c_int()
    rc = hip.lib.hipGetDeviceCount(ctypes.byref(ndev))
    if rc != HIP_SUCCESS:
        return die(f"hipGetDeviceCount -> {rc} ({hip.errstr(rc)}); "
                   f"ROCR_VISIBLE_DEVICES={os.environ.get('ROCR_VISIBLE_DEVICES')}")
    if args.device >= ndev.value:
        return die(f"--device {args.device} but only {ndev.value} HIP device(s) visible")
    rc = hip.lib.hipSetDevice(args.device)
    if rc != HIP_SUCCESS:
        return die(f"hipSetDevice({args.device}) -> {rc} ({hip.errstr(rc)})")

    name = ctypes.create_string_buffer(256)
    hip.lib.hipDeviceGetName(name, 256, args.device)
    busid = ctypes.create_string_buffer(64)
    hip.lib.hipDeviceGetPCIBusId(busid, 64, args.device)
    totmem = ctypes.c_size_t()
    hip.lib.hipDeviceTotalMem(ctypes.byref(totmem), args.device)
    pci = busid.value.decode().lower()

    box = collect_box_state(include_smi=True)
    versions = collect_versions(hip)
    node = next((n for n in box["kfd_nodes"] if n["pci_slot"].lower() == pci), None)
    device = {
        "hip_index": args.device,
        "name": name.value.decode(),
        "pci_slot": pci,
        "card_label": _CARD_LABELS.get(pci, "UNKNOWN CARD -- identify by pci_slot"),
        "gfx_arch": (node or {}).get("gfx_arch"),
        "simd_count": (node or {}).get("simd_count"),
        "kfd_node": (node or {}).get("kfd_node"),
        "total_mem_bytes": totmem.value,
        "hip_device_count_visible": ndev.value,
    }
    if node is None:
        return die(f"HIP device {args.device} reports pci {pci} but no KFD node matches; "
                   "refusing to report a measurement whose card is ambiguous")
    # HARD gate, not a warning.  ROCR agent ordering is not a contract: if the pin
    # ever fails to keep the Ryzen iGPU (gfx1036, 47 GB of GTT) out of enumeration,
    # a full conformance verdict measured on the iGPU would be published under a
    # label that merely says "UNKNOWN CARD".
    if device["card_label"].startswith("UNKNOWN") and not args.allow_unknown_card:
        return die(
            f"HIP device {args.device} is pci {pci} (arch {device['gfx_arch']}, "
            f"simd_count {device['simd_count']}), which is NOT one of the two known "
            f"discrete gfx1201 cards {sorted(_CARD_LABELS)}. A conformance verdict "
            "from the wrong device -- the Ryzen iGPU in particular -- is worthless. "
            "Re-check ROCR_VISIBLE_DEVICES, or pass --allow-unknown-card if this is "
            "deliberately a different box."
        )
    if device["card_label"].startswith("UNKNOWN"):
        print(f"P6 WARNING: --allow-unknown-card: pci {pci} is not one of the two "
              "known discrete cards; the verdict is NOT comparable to prior runs",
              file=sys.stderr)

    arch = args.arch or device["gfx_arch"]
    print(f"P6: device {args.device} = {device['name']} [{device['card_label']}] "
          f"pci={pci} arch={arch}")
    print(f"P6: ROCm {versions['rocm_info_version']} hip_runtime={versions['hip_runtime_version']} "
          f"lib={versions['libamdhip64_realpath']}")
    print(f"P6: size={args.size_mib} MiB x {args.handles} handles, "
          f"{args.warmup} warm-up + {args.reps} measured reps, location={args.location}")

    shader = None
    shader_info: dict = {"available": False, "verified": False,
                         "reason": "disabled by --no-shader"}
    if not args.no_shader:
        if arch is None:
            shader_info = {"available": False, "verified": False,
                           "reason": "gfx arch unknown; pass --arch"}
        else:
            shader = ShaderPath(hip, arch, args.hiprtc_lib)
            if shader.available:
                # Prove read AND write on ordinary device memory BEFORE taking a
                # single conformance observation.  A dead shader arm would hold
                # dynamic_residency_available at False forever, i.e. the tripwire
                # could never fire -- an unfalsifiable probe.
                shader.selfcheck()
            shader_info = {
                "available": shader.available,
                "verified": shader.verified,
                "variant": shader.variant,
                "selfcheck": shader.verify_detail,
                "reason": shader.reason,
                "arch": arch,
                "hiprtc_path": getattr(shader, "rtc_path", None),
                "kernel_src": (shader.src or KERNEL_SRC).decode(),
                "liveness_token": f"0x{OUT_LIVENESS:08x}",
            }
            if not shader.usable():
                print(f"P6 WARNING: shader path unusable ({shader.reason}); "
                      "runtime-copy-path results only, and the T3 tripwire CANNOT "
                      "turn green in this run", file=sys.stderr)
                shader = None
            else:
                print(f"P6: shader path OK (variant={shader_info['variant']}, "
                      "selfchecked read+write on ordinary device memory)")

    flusher = CacheFlusher(hip, args.cache_flush_mib << 20)
    if not flusher.available:
        print(f"P6 WARNING: cache flush unavailable/disabled ({flusher.reason}); "
              "a stale read cannot be separated from a cached line from the OLD "
              "mapping", file=sys.stderr)

    locations = ["device", "host"] if args.location == "both" else [args.location]
    variants = []
    for loc in locations:
        print(f"P6: running location={loc} ...")
        v = run_variant(hip, args.device, loc, args.size_mib << 20, args.handles,
                        args.reps, args.warmup, shader, flusher)
        variants.append(v)
        if v.get("control_ok") is False:
            print(f"     CONTROL FAILED: {v['control'].get('meaning')} did NOT hold",
                  file=sys.stderr)
        if v["complete"]:
            print(f"     minimal read  {v['minimal_remap']['copy_conformant_count']}"
                  f"/{v['minimal_remap']['n_observations']} correct "
                  f"({v['minimal_remap']['stale_pattern']})")
            print(f"     parked read   {v['parked_read']['copy_conformant_count']}"
                  f"/{v['parked_read']['n_observations']} correct")
            print(f"     write landing {v['parked_write']['copy_conformant_count']}"
                  f"/{v['parked_write']['n_observations']} correct")
        else:
            print(f"     INCOMPLETE: {v['error']}")

    # Snapshot the flush record BEFORE free() -- free() flips `available`, and
    # reading info() afterwards would claim the cache-flush arm never ran and
    # attach a spurious "stale vs cached line not excluded" caveat to the verdict.
    cache_flush_info = flusher.info()
    flusher.free()
    box_after = collect_box_state(include_smi=True)
    shader_info["dead_launches"] = (shader.dead_launches if shader else 0)
    shader_info["launches"] = (shader.launches if shader else 0)

    complete = [v for v in variants if v.get("complete")]
    if not complete:
        # Keep the standalone upstream repro visible even though the parked
        # sub-tests' precondition failed -- minimal_remap does not depend on it.
        minimal_only = {
            v["location"]: v.get("minimal_only_verdict") for v in variants
        }
        payload = build_payload(args, box, versions, device, variants, shader_info,
                                "PRECONDITION_FAILED",
                                {"primary_location": None, "read_conformant": None,
                                 "write_conformant": None,
                                 "read_conformant_shader": None,
                                 "write_conformant_shader": None,
                                 "control_ok": next(
                                     (v.get("control_ok") for v in variants
                                      if v.get("control_ok") is not None), None),
                                 "shader_arm_trustworthy": bool(
                                     shader is not None and shader.usable()
                                     and not shader.dead_launches),
                                 "dynamic_residency_available": None,
                                 "expected": args.expect, "matched_expectation": None,
                                 "minimal_remap_only": minimal_only,
                                 "explanation": "no location produced a complete "
                                                "verdict; NO conformance number was "
                                                "measured. minimal_remap_only carries "
                                                "whatever the standalone repro saw."},
                                error=[v.get("error") for v in variants],
                                box_state_after=box_after,
                                cache_flush=cache_flush_info)
        payload["exit_code"] = 2
        errs = validate_payload(payload)
        if errs:
            print("P6 WARNING: PRECONDITION payload failed schema validation:\n  - "
                  + "\n  - ".join(errs), file=sys.stderr)
        for p in write_outputs(payload, Path(args.out), not args.no_md):
            print(f"P6: wrote {p}")
        print(json.dumps(payload, indent=2))
        return die("no location produced a complete verdict. NOTHING was measured; "
                   f"the canonical {Path(args.out).name} was NOT overwritten -- see "
                   f"{resolve_out_path(Path(args.out), 'PRECONDITION_FAILED').name} "
                   "for the failing HIP call.")

    primary = next((v for v in complete if v["location"] == "device"), complete[0])
    read_ok = primary["read_conformant"]
    write_ok = primary["write_conformant"]
    sh_read = primary["read_conformant_shader"]
    sh_write = primary["write_conformant_shader"]
    shader_trustworthy = bool(
        shader is not None and shader.usable() and not shader.dead_launches
    )
    shader_clean = (sh_read is True and sh_write is True and shader_trustworthy)
    dynamic_ok = bool(read_ok and write_ok and shader_clean)

    if read_ok and write_ok:
        status = "CONFORMANT"
    elif not read_ok and not write_ok:
        status = "BROKEN_STALE"
    else:
        status = "MIXED"

    explanation = {
        "CONFORMANT": "remap serves the newly mapped handle in both directions",
        "BROKEN_STALE": "remap silently serves the stale physical page for reads AND writes",
        "MIXED": "reads and writes disagree -- read the per-observation records",
    }[status]
    if read_ok and write_ok and not shader_clean:
        explanation += (
            "; the runtime copy path is conformant but the SHADER arm was not "
            "verified clean" + ("" if shader_trustworthy else
                                " (and the shader arm is not trustworthy: "
                                f"usable={shader is not None and shader.usable()}, "
                                f"dead_launches={shader.dead_launches if shader else 'n/a'})")
            + ", so T3 stays blocked -- the offload design reads weights from the "
              "shader, not from hipMemcpy"
        )
    if not cache_flush_info.get("available") and status != "CONFORMANT":
        explanation += ("; NOTE: the cache-flush arm was unavailable, so a stale "
                        "CACHE LINE from the old mapping has not been excluded as "
                        "an alternative explanation to a stale PAGE TABLE")

    matched = expectation_matched(status, dynamic_ok, args.expect)
    verdict = {
        "primary_location": primary["location"],
        "read_conformant": read_ok,
        "read_conformant_shader": sh_read,
        "write_conformant": write_ok,
        "write_conformant_shader": sh_write,
        "control_ok": primary.get("control_ok"),
        "shader_arm_trustworthy": shader_trustworthy,
        "stale_survives_cache_flush": (
            None if not cache_flush_info.get("available")
            else (status != "CONFORMANT")
        ),
        "dynamic_residency_available": dynamic_ok,
        "expected": args.expect,
        "matched_expectation": matched,
        "explanation": explanation,
        "per_location": {
            v["location"]: {
                "complete": v.get("complete"),
                "control_ok": v.get("control_ok"),
                "stages_completed": v.get("stages_completed"),
                "read_conformant": v.get("read_conformant"),
                "write_conformant": v.get("write_conformant"),
                "minimal_remap_only": v.get("minimal_only_verdict"),
            }
            for v in variants
        },
    }

    payload = build_payload(args, box, versions, device, variants, shader_info,
                            status, verdict, box_state_after=box_after,
                            cache_flush=cache_flush_info)
    errs = validate_payload(payload)
    if errs:
        return die("internal: JSON schema validation failed:\n  - " + "\n  - ".join(errs))
    code = 0 if matched else 3
    payload["exit_code"] = code
    written = write_outputs(payload, Path(args.out), not args.no_md)

    print(json.dumps(payload, indent=2))
    print()
    for p in written:
        print(f"P6: wrote {p}")
    banner = (
        "*** VMM REMAP CONFORMANCE: STILL BROKEN (expected) -- residency must stay a "
        "BOOT-TIME PLACEMENT decision; tier T3 remains unreachable. ***"
        if status == "BROKEN_STALE" else
        "*** !!! VMM REMAP IS NOW CONFORMANT -- dynamic page-level residency (T3) is "
        "UNBLOCKED on this stack. Re-read WEIGHT_OFFLOAD_PLAN sec. 8. !!! ***"
        if dynamic_ok else
        f"*** VMM REMAP CONFORMANCE: {status} -- {explanation} ***"
    )
    print()
    print(banner)
    if code == 3:
        print(f"P6: exit 3 -- measured '{status}' contradicts --expect {args.expect}",
              file=sys.stderr)
    return code


if __name__ == "__main__":
    sys.exit(main())
