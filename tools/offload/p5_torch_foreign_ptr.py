#!/usr/bin/env python3
"""P5 — torch over a foreign device pointer, in the serve image, under graph capture.

Question (WEIGHT_OFFLOAD_PLAN.md section 3, row P5):

  Can torch be made to hand out tensors whose storage lives on a VA *we* reserved and backed
  ourselves (hipMemAddressReserve + hipMemCreate + hipMemMap + hipMemSetAccess), via
  `torch._C._cuda_customAllocator` + `torch.cuda.MemPool` + `torch.cuda.use_mem_pool`, such that
  `t.data_ptr() == reserved_base` --

    (a) inside the serve image (not just on the host python),
    (b) concurrently with `expandable_segments:True` (the compose default, docker-compose.yml:68),
    (c) with graph capture active, and
    (d) surviving `torch.cuda.empty_cache()` -- which `engine/graph.py:314` calls right before
        capture, i.e. after the offload arena would already exist. Does empty_cache() invoke the
        custom *free* callback for a live user-MemPool block?

A green P5 means the arena plumbing is ~30 lines of ctypes. A red P5 costs +3 days (a C++
`from_blob` extension and a build-system change). It is not fatal to the project either way, so
this probe must distinguish "the answer is no" (exit 2) from "the probe could not run" (exit 3).

KNOWN ABORT, GUARDED
--------------------
A tensor that outlives its MemPool raises `c10::Error: invalid device pointer` inside
HIPCachingAllocator and *aborts the process* (SIGABRT), which would destroy the measurement.
Guards, all of them mandatory and all present below:

  * the two `CFUNCTYPE` trampolines, the `_cuda_CUDAAllocator`, the `MemPool`, every tensor cut
    from it, and the VA reservation are held in the module-level `_KEEP` list forever;
  * the free callback is a pure no-op -- it never returns a page to anyone;
  * the alloc callback is a forward-only bump pointer inside our own reservation and NEVER
    returns NULL (a request it cannot serve falls back to hipMalloc and is *recorded as not
    served from the reservation*, so the arm reports FAIL instead of segfaulting later);
  * results are written and fsynced BEFORE any teardown, and the process then leaves via
    `os._exit()` so no destructor can run;
  * each measured leg runs in its own child process, so even an abort we failed to anticipate is
    recorded by the parent (returncode -6) instead of losing the run.

ADVERSARIAL HARDENING (why some checks look paranoid)
-----------------------------------------------------
A probe that produces a confident WRONG number is worse than one that produces none. Each of these
exists because the naive version would have done exactly that:

  * the `reservation` arm does NOT pass on `hipSuccess`. It writes two DIFFERENT byte patterns at
    head/middle/tail of the whole reservation and reads them back, because on this box a VMM call
    can return success and still serve a stale physical page. `host_backing_available` is derived
    from that data check, never from a return code or a granularity QUERY.
  * `correctness` checks three things, not one: a copy-engine roundtrip (which may never touch a
    shader), a shader READ, and a shader WRITE -- with two different contents in sequence.
  * `empty_cache_live` writes its own distinct payload per rep (and re-writes after the call to
    prove the pages are still writable). It used to compare against whatever the `correctness` arm
    left behind, which both created a false-KILL cascade and could not detect an aliased page.
  * `alloc_latency_us` is reported three ways. torch's caching allocator can serve a rep from a
    previously split block WITHOUT calling us; folding those into one median would understate the
    arena-allocation cost by an order of magnitude.
  * replay timings carry an explicit "this is NOT a bandwidth number" note -- the captured working
    set is a few MiB and therefore L2/MALL-resident. Bandwidth is P1's job, at >= 256 MB.
  * `expandable_segments` coexistence is only claimed by a leg that VERIFIED the setting is live
    via `memory_snapshot()`; an unread env var is not evidence.
  * the exit code distinguishes a measured NO (2, a real KILL) from "could not run" (3) by
    ATTRIBUTION, so a gating leg that cannot build its backing never reports a kill for a claim
    that was never measured.

LAYOUT
------
Parent process (no torch import, ever): collects box state, runs one child per leg, merges into
`p5.json` + `p5.md`. Child process (`--child-leg NAME`): the actual measurement, writes
`p5_leg_<NAME>.json` and `p5_leg_<NAME>.log`.

Legs cross `expandable_segments` on/off with device- vs host-located backing pages, because the
landable tier (T1) is 100 % host-backed and T2 mixes both.

USAGE
-----
    tools/offload/p5_run.sh                    # canonical: in the serve image, all legs
    python3 tools/offload/p5_torch_foreign_ptr.py --selftest    # no GPU, validates JSON shape
"""

from __future__ import annotations

# --- device visibility: forced before ANY chance of a torch/HIP import ---------------------------
# ROCm device 2 on this box is the Ryzen iGPU advertising ~47 GB of GTT. If it enters enumeration
# it poisons every "biggest free pool" heuristic. It is never a compute target.
import os as _os

_os.environ["ROCR_VISIBLE_DEVICES"] = _os.environ.get("MINISGL_P5_ROCR_DEVICES", "0,1")
_os.environ.pop("HIP_VISIBLE_DEVICES", None)
_os.environ.pop("CUDA_VISIBLE_DEVICES", None)

import argparse
import ctypes
import ctypes.util
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
import traceback

SCHEMA_VERSION = "p5/1"
PROBE_ID = "P5"

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
DEFAULT_OUT_DIR = os.path.join(_REPO_ROOT, "docs", "measurements", "WEIGHT_OFFLOAD_2026-09-02")

EXPANDABLE = "expandable_segments:True"

# leg name -> (alloc-conf value, backing media). "" alloc-conf means the vars are UNSET.
LEGS: dict[str, dict[str, str]] = {
    "expandable_host": {"alloc_conf": EXPANDABLE, "backing": "host"},
    "expandable_device": {"alloc_conf": EXPANDABLE, "backing": "device"},
    "plain_host": {"alloc_conf": "", "backing": "host"},
    "plain_device": {"alloc_conf": "", "backing": "device"},
}
DEFAULT_LEGS = "expandable_host,expandable_device,plain_host"

# Legs whose failure makes P5 red. The compose default is expandable_segments:True, and both
# media matter (T1 = host, T2 = device+host). A "plain" leg is a control: informative, not a gate.
GATING_LEGS = {"expandable_host", "expandable_device"}

EXIT_OK = 0
EXIT_MEASURED_FAIL = 2  # the probe ran and the answer is NO
EXIT_PRECONDITION = 3  # the probe could not run -- no number was produced

# --- HIP constants (from /opt/rocm/include/hip) --------------------------------------------------
hipMemAllocationTypePinned = 0x1
hipMemLocationTypeDevice = 1
hipMemLocationTypeHost = 2
hipMemAccessFlagsProtReadWrite = 3
hipMemAllocationGranularityMinimum = 0x0
hipMemAllocationGranularityRecommended = 0x1
hipMemcpyHostToDevice = 1
hipMemcpyDeviceToHost = 2

# --- module-level anti-GC anchor. NEVER cleared, NEVER shrunk. -----------------------------------
_KEEP: list = []
_ALLOC_EVENTS: list = []
_FREE_EVENTS: list = []
_ARENA = {"base": 0, "size": 0, "cursor": 0, "fallbacks": 0, "device": None}
_HIP = None  # set by the child before the allocator can ever be called


# =================================================================================================
# HIP ctypes layer
# =================================================================================================
class HipMemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class _HipAllocFlags(ctypes.Structure):
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
        ("allocFlags", _HipAllocFlags),
    ]


class HipMemAccessDesc(ctypes.Structure):
    _fields_ = [("location", HipMemLocation), ("flags", ctypes.c_int)]


class HipError(RuntimeError):
    pass


class Hip:
    """Thin ctypes binding for exactly the VMM entry points this probe needs."""

    def __init__(self) -> None:
        path = ctypes.util.find_library("amdhip64") or "libamdhip64.so"
        self.lib = ctypes.CDLL(path)
        self.so_path = path
        L = self.lib
        L.hipGetErrorString.restype = ctypes.c_char_p
        L.hipGetErrorString.argtypes = [ctypes.c_int]
        L.hipGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
        L.hipDeviceGetPCIBusId.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
        L.hipMemGetInfo.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
        L.hipMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        L.hipMemGetAllocationGranularity.argtypes = [
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(HipMemAllocationProp),
            ctypes.c_int,
        ]
        L.hipMemAddressReserve.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_ulonglong,
        ]
        L.hipMemCreate.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_size_t,
            ctypes.POINTER(HipMemAllocationProp),
            ctypes.c_ulonglong,
        ]
        L.hipMemMap.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_ulonglong,
        ]
        L.hipMemSetAccess.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.POINTER(HipMemAccessDesc),
            ctypes.c_size_t,
        ]
        L.hipMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        L.hipDeviceSynchronize.argtypes = []
        L.hipSetDevice.argtypes = [ctypes.c_int]
        L.hipGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int)]

    def err(self, rc: int) -> str:
        try:
            s = self.lib.hipGetErrorString(ctypes.c_int(rc))
            return s.decode() if s else f"rc={rc}"
        except Exception:
            return f"rc={rc}"

    def check(self, rc: int, what: str) -> None:
        if rc != 0:
            raise HipError(f"{what} failed: rc={rc} ({self.err(rc)})")

    # -- helpers ----------------------------------------------------------------------------------
    def device_count(self) -> int:
        n = ctypes.c_int(0)
        self.check(self.lib.hipGetDeviceCount(ctypes.byref(n)), "hipGetDeviceCount")
        return int(n.value)

    def pci_bus_id(self, dev: int) -> str:
        buf = ctypes.create_string_buffer(64)
        rc = self.lib.hipDeviceGetPCIBusId(buf, ctypes.c_int(64), ctypes.c_int(dev))
        if rc != 0:
            return f"<hipDeviceGetPCIBusId rc={rc}>"
        return buf.value.decode()

    def set_device(self, dev: int) -> int:
        """Pin libamdhip64's current device for OUR ctypes calls. Do not assume torch's
        set_device already did it for this thread -- a mismatch would validate the wrong card."""
        self.check(self.lib.hipSetDevice(ctypes.c_int(dev)), f"hipSetDevice({dev})")
        cur = ctypes.c_int(-1)
        self.lib.hipGetDevice(ctypes.byref(cur))
        return int(cur.value)

    def mem_info(self) -> dict:
        free = ctypes.c_size_t(0)
        total = ctypes.c_size_t(0)
        rc = self.lib.hipMemGetInfo(ctypes.byref(free), ctypes.byref(total))
        if rc != 0:
            return {"free_bytes": None, "total_bytes": None, "rc": rc}
        return {"free_bytes": int(free.value), "total_bytes": int(total.value)}

    def prop(self, backing: str, dev: int) -> HipMemAllocationProp:
        p = HipMemAllocationProp()
        ctypes.memset(ctypes.byref(p), 0, ctypes.sizeof(p))
        p.type = hipMemAllocationTypePinned
        p.requestedHandleType = 0
        if backing == "device":
            p.location.type = hipMemLocationTypeDevice
            p.location.id = dev
        elif backing == "host":
            p.location.type = hipMemLocationTypeHost
            p.location.id = 0
        else:
            raise ValueError(f"unknown backing {backing!r}")
        return p

    def granularity(self, prop: HipMemAllocationProp, flag: int) -> int:
        g = ctypes.c_size_t(0)
        self.check(
            self.lib.hipMemGetAllocationGranularity(ctypes.byref(g), ctypes.byref(prop), flag),
            "hipMemGetAllocationGranularity",
        )
        return int(g.value)

    # -- backing validation -----------------------------------------------------------------------
    def validate_backing(self, base: int, size: int, chunk: int = 1 << 20) -> dict:
        """Prove the mapped pages actually STORE what we write, at head/middle/tail.

        This exists because on this box a VMM call can return `hipSuccess` and still serve a stale
        physical page (see the P6 brief). A reservation arm that passes on `rc == 0` alone would
        report `host_backing_available: YES` with zero evidence, which is exactly the confident
        wrong number this probe must not produce.

        Two DIFFERENT patterns are written and read back in sequence per offset: a stale page that
        happened to already hold pattern A cannot also hold pattern B.
        """
        chunk = min(chunk, size)
        offsets = sorted({0, max(0, (size // 2) - (size // 2) % 4096), size - chunk})
        offsets = [o for o in offsets if 0 <= o <= size - chunk]
        src = (ctypes.c_ubyte * chunk)()
        dst = (ctypes.c_ubyte * chunk)()
        results = []
        ok_all = True
        for off in offsets:
            per_off = {"offset": off, "chunk_bytes": chunk, "patterns": []}
            for pat_idx, seed in enumerate((0x5A, 0xA5)):
                # every byte differs between the two patterns, plus position-dependent markers at
                # each 4 KiB page boundary so a page-level aliasing mixup cannot alias into a pass
                ctypes.memset(src, seed, chunk)
                for i in range(0, chunk, 4096):
                    src[i] = (seed + (off >> 12) + (i >> 12) + pat_idx) & 0xFF
                    src[min(i + 4095, chunk - 1)] = (seed ^ ((i >> 12) & 0xFF)) & 0xFF
                ctypes.memset(dst, 0, chunk)
                rc_w = self.lib.hipMemcpy(
                    ctypes.c_void_p(base + off), ctypes.cast(src, ctypes.c_void_p),
                    ctypes.c_size_t(chunk), ctypes.c_int(hipMemcpyHostToDevice))
                rc_s = self.lib.hipDeviceSynchronize()
                rc_r = self.lib.hipMemcpy(
                    ctypes.cast(dst, ctypes.c_void_p), ctypes.c_void_p(base + off),
                    ctypes.c_size_t(chunk), ctypes.c_int(hipMemcpyDeviceToHost))
                rc_s2 = self.lib.hipDeviceSynchronize()
                match = (bytes(src) == bytes(dst))
                ok_all = ok_all and match and rc_w == 0 and rc_r == 0 and rc_s == 0 and rc_s2 == 0
                per_off["patterns"].append({
                    "pattern": pat_idx, "seed": seed, "bytes_match": bool(match),
                    "rc_h2d": int(rc_w), "rc_d2h": int(rc_r),
                    "rc_sync": [int(rc_s), int(rc_s2)],
                })
            results.append(per_off)
        return {"validated": bool(ok_all), "offsets_checked": offsets, "detail": results}


def _round_up(x: int, a: int) -> int:
    return ((x + a - 1) // a) * a


# =================================================================================================
# the custom allocator: forward-only bump pointer inside OUR reservation, no-op free
# =================================================================================================
def _install_callbacks():
    """Build and permanently anchor the two CFUNCTYPE trampolines. Returns (alloc_cb, free_cb)."""
    ALLOC_T = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p)
    FREE_T = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p)

    def _alloc(size, device, stream):  # noqa: ANN001 - C ABI
        try:
            size = int(size)
            base = int(_ARENA["base"])
            cur = _round_up(int(_ARENA["cursor"]), 512)
            # The reservation is mapped and access-granted for ONE device only. Serving it for a
            # different device index would hand torch memory the asking device cannot address --
            # a silent-corruption path, not a measurement. Fall back instead.
            arena_dev = _ARENA.get("device")
            right_device = arena_dev is None or int(device) == int(arena_dev)
            if base and right_device and cur + size <= int(_ARENA["size"]):
                ptr = base + cur
                _ARENA["cursor"] = cur + size
                _ALLOC_EVENTS.append(
                    {
                        "size": size,
                        "device": int(device),
                        "stream": int(stream or 0),
                        "ptr": ptr,
                        "offset": cur,
                        "from_reservation": True,
                    }
                )
                return ptr
            # NEVER return NULL: a NULL here becomes a use-after-null deep inside torch. Serve it
            # from a real hipMalloc, record that it did NOT come from the reservation, and let the
            # arm report a clean FAIL.
            p = ctypes.c_void_p()
            rc = _HIP.lib.hipMalloc(ctypes.byref(p), ctypes.c_size_t(size)) if _HIP else -1
            _ARENA["fallbacks"] = int(_ARENA["fallbacks"]) + 1
            _ALLOC_EVENTS.append(
                {
                    "size": size,
                    "device": int(device),
                    "stream": int(stream or 0),
                    "ptr": int(p.value or 0),
                    "offset": None,
                    "from_reservation": False,
                    "hipMalloc_rc": int(rc),
                    "reason": (
                        "no reservation" if not base
                        else "wrong device" if not right_device
                        else "reservation exhausted"
                    ),
                }
            )
            return int(p.value or 0)
        except BaseException as exc:  # a raise inside a ctypes callback would silently return NULL
            _ALLOC_EVENTS.append({"error": f"{type(exc).__name__}: {exc}", "from_reservation": False})
            return 0

    def _free(ptr, size, device, stream):  # noqa: ANN001 - C ABI
        # Deliberate no-op. The pages belong to our reservation for the life of the process; the
        # hipMalloc fallbacks are intentionally leaked (short-lived probe) rather than risk a
        # double-free abort that would destroy the measurement.
        try:
            _FREE_EVENTS.append(
                {
                    "ptr": int(ptr or 0),
                    "size": int(size),
                    "device": int(device),
                    "stream": int(stream or 0),
                    "t": time.time(),
                }
            )
        except BaseException:
            pass

    a = ALLOC_T(_alloc)
    f = FREE_T(_free)
    _KEEP.extend([ALLOC_T, FREE_T, _alloc, _free, a, f])
    return a, f


# =================================================================================================
# box state
# =================================================================================================
def _read_first(path: str) -> str | None:
    try:
        with open(path) as fh:
            return fh.read()
    except Exception:
        return None


def _run(cmd: list[str], timeout: int = 20) -> dict:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return {"rc": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()[:2000]}
    except FileNotFoundError:
        return {"rc": None, "error": "not found"}
    except Exception as exc:
        return {"rc": None, "error": f"{type(exc).__name__}: {exc}"}


def box_state() -> dict:
    """Everything needed to judge whether this measurement was taken on a busy box."""
    st: dict = {"t_wall": time.time(), "t_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    st["hostname"] = platform.node()
    st["kernel"] = platform.release()
    st["in_container"] = os.path.exists("/.dockerenv")
    st["uid"] = os.getuid()

    meminfo = _read_first("/proc/meminfo") or ""
    mem: dict = {}
    for key in ("MemTotal", "MemFree", "MemAvailable", "Cached", "SwapTotal", "SwapFree", "Shmem"):
        m = re.search(rf"^{key}:\s+(\d+) kB", meminfo, re.M)
        mem[key + "_kB"] = int(m.group(1)) if m else None
    st["meminfo"] = mem

    vmstat = _read_first("/proc/vmstat") or ""
    vm: dict = {}
    for key in ("pswpin", "pswpout", "pgmajfault", "nr_free_pages"):
        m = re.search(rf"^{key} (\d+)", vmstat, re.M)
        vm[key] = int(m.group(1)) if m else None
    st["vmstat"] = vm

    st["loadavg"] = (_read_first("/proc/loadavg") or "").strip() or None
    st["free_g"] = _run(["free", "-g"])
    st["rocm_smi"] = _run(["rocm-smi", "--showmeminfo", "vram", "--showuse", "--csv"], timeout=40)
    st["env_visibility"] = {
        "ROCR_VISIBLE_DEVICES": os.environ.get("ROCR_VISIBLE_DEVICES"),
        "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES"),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        "PYTORCH_HIP_ALLOC_CONF": os.environ.get("PYTORCH_HIP_ALLOC_CONF"),
    }
    # provenance: WHICH image and WHICH source tree produced this number (injected by p5_run.sh)
    st["provenance"] = {
        "image": os.environ.get("P5_IMAGE"),
        "image_id": os.environ.get("P5_IMAGE_ID"),
        "git_sha": os.environ.get("P5_GIT_SHA"),
        "script": os.path.abspath(__file__),
        "schema": SCHEMA_VERSION,
    }
    st["image_markers"] = {
        "dockerenv": os.path.exists("/.dockerenv"),
        "opt_kernels": os.path.isdir("/opt/kernels"),
        "opt_minisgl_python": os.path.isdir("/opt/minisgl/python"),
        "opt_venv": os.path.isdir("/opt/venv"),
        "os_release": (_read_first("/etc/os-release") or "").strip()[:400] or None,
    }
    return st


def _stats(samples: list[float]) -> dict:
    """median + spread. Returns Nones (never a fabricated number) if there is nothing to reduce."""
    if not samples:
        return {"n": 0, "median": None, "mean": None, "stdev": None, "min": None, "max": None,
                "p90": None, "samples": []}
    s = sorted(samples)
    return {
        "n": len(s),
        "median": statistics.median(s),
        "mean": statistics.fmean(s),
        "stdev": (statistics.pstdev(s) if len(s) < 2 else statistics.stdev(s)),
        "min": s[0],
        "max": s[-1],
        "p90": s[min(len(s) - 1, int(round(0.9 * (len(s) - 1))))],
        "samples": samples,
    }


# =================================================================================================
# host-side verification -- the ONLY trustworthy check on this box
# =================================================================================================
class HostVerify:
    """Verify device bytes by copying them to a host buffer allocated ONCE, and comparing on the
    CPU.

    Why this exists (measured 2026-09-02, see p5_diagnostics/): the obvious check,

        torch.equal(t, torch.full_like(t, v))

    allocates its comparison operand -- and `torch.equal`'s own reduction output -- from torch's
    DEFAULT caching allocator. Under `expandable_segments:True` (the compose default) that
    allocator unmaps physical handles on `empty_cache()` and re-maps them at the SAME VA on the
    next allocation, and on this driver a remap at an already-used VA silently serves a STALE
    page. The freshly-filled comparison tensor then reads back as zeros and the check reports a
    failure of the tensor under test, which is innocent. That is a checker artifact, and it is
    exactly the shape of a confident wrong number: it fired on the arm wired to the KILL criterion.

    So: no device allocation participates in a verification. One host buffer, one blocking
    hipMemcpy D2H, one CPU compare of the FULL byte range.
    """

    def __init__(self, hip: "Hip", nbytes: int) -> None:
        import torch

        self._torch = torch
        self.nbytes = int(nbytes)
        self.hip = hip
        self.buf = (ctypes.c_ubyte * self.nbytes)()
        self.addr = ctypes.addressof(self.buf)
        # zero-copy CPU uint8 view over the ctypes buffer -- allocates no device memory
        self.view = torch.frombuffer(memoryview(self.buf), dtype=torch.uint8)

    def d2h(self, dptr: int, nbytes: int) -> None:
        if nbytes > self.nbytes:
            raise ValueError(f"HostVerify buffer is {self.nbytes} B, asked for {nbytes} B")
        self.hip.check(
            self.hip.lib.hipMemcpy(ctypes.c_void_p(self.addr), ctypes.c_void_p(dptr),
                                   ctypes.c_size_t(nbytes), hipMemcpyDeviceToHost),
            "hipMemcpy D2H (HostVerify)")

    def equals_fill(self, dptr: int, n_elem: int, dtype, value) -> dict:
        """Is [dptr, dptr+n_elem*itemsize) exactly `value` repeated, in `dtype`?"""
        torch = self._torch
        expect = torch.empty(n_elem, dtype=dtype).fill_(value)  # CPU tensor, no device alloc
        expect_b = expect.view(torch.uint8) if expect.element_size() > 1 else expect
        nb = int(expect.numel() * expect.element_size())
        self.d2h(dptr, nb)
        got = self.view[:nb]
        ok = bool(torch.equal(got, expect_b.reshape(-1)))
        det = {"n_elem": n_elem, "bytes_compared": nb, "value": value, "ok": ok,
               "verified": "host (ctypes hipMemcpy D2H + CPU compare)"}
        if not ok:
            neq = (got != expect_b.reshape(-1))
            idx = int(neq.nonzero()[0].item())
            det.update({"n_mismatched_bytes": int(neq.sum().item()),
                        "first_bad_byte_index": idx,
                        "first_bad_byte": int(got[idx].item()),
                        "expected_byte": int(expect_b.reshape(-1)[idx].item())})
        return det


# =================================================================================================
# result skeleton + schema validation
# =================================================================================================
ARM_NAMES = (
    "reservation",
    "pool_alloc",
    "correctness",
    "empty_cache_live",
    "empty_cache_cached",
    "graph_capture",
)


def new_arm(question: str) -> dict:
    return {"question": question, "status": "not_run", "passed": None, "detail": {}, "error": None}


def leg_skeleton(leg: str, cfg: dict, args) -> dict:
    return {
        "schema": SCHEMA_VERSION,
        "probe": PROBE_ID,
        "kind": "leg",
        "leg": leg,
        "config": {
            "alloc_conf": cfg["alloc_conf"],
            "backing": cfg["backing"],
            "device_index": args.device,
            "slot_bytes": args.slot_bytes,
            "reps": args.reps,
            "warmup": args.warmup,
            "replays": args.replays,
            "dtype": args.dtype,
        },
        "status": "not_run",
        "passed": None,
        "precondition_error": None,
        "error": None,
        "traceback": None,
        "torch": {},
        "device": {},
        "arms": {name: new_arm("") for name in ARM_NAMES},
        "allocator_events": {},
        "box_state": {},
        "synthetic": False,
    }


def validate_report(obj: dict) -> None:
    """Raise if the merged report is not the shape the Verify phase expects."""
    def need(d, k, types, where):
        if k not in d:
            raise ValueError(f"missing key {where}.{k}")
        if not isinstance(d[k], types):
            raise ValueError(f"{where}.{k} is {type(d[k]).__name__}, want {types}")

    need(obj, "schema", str, "root")
    if obj["schema"] != SCHEMA_VERSION:
        raise ValueError(f"schema mismatch: {obj['schema']!r} != {SCHEMA_VERSION!r}")
    for k in ("probe", "kind"):
        need(obj, k, str, "root")
    need(obj, "synthetic", bool, "root")
    need(obj, "exit_code", int, "root")
    need(obj, "box_state", dict, "root")
    need(obj, "legs", dict, "root")
    need(obj, "verdict", dict, "root")
    v = obj["verdict"]
    for k in (
        "p5_pass",
        "foreign_ptr_ok",
        "expandable_segments_coexists",
        "capture_ok",
        "empty_cache_invokes_free_cb_live_block",
        "empty_cache_invokes_free_cb_cached_block",
        "tensor_survives_empty_cache",
        "host_backing_available",
        "hipMalloc_fallbacks_seen",
    ):
        if k not in v:
            raise ValueError(f"missing verdict.{k}")
        if not isinstance(v[k], (bool, type(None))):
            raise ValueError(f"verdict.{k} must be bool or null, got {type(v[k]).__name__}")
    need(v, "reasons", list, "verdict")
    need(v, "per_leg_claims", dict, "verdict")
    for name, leg in obj["legs"].items():
        need(leg, "status", str, f"legs.{name}")
        need(leg, "arms", dict, f"legs.{name}")
        for arm_name, arm in leg["arms"].items():
            need(arm, "status", str, f"legs.{name}.arms.{arm_name}")
            if not isinstance(arm.get("passed"), (bool, type(None))):
                raise ValueError(f"legs.{name}.arms.{arm_name}.passed must be bool or null")


# =================================================================================================
# CHILD: the measurement
# =================================================================================================
def run_leg(leg: str, cfg: dict, args) -> dict:  # noqa: C901 - a probe is a straight line of arms
    res = leg_skeleton(leg, cfg, args)
    res["box_state"] = box_state()

    # ---- preconditions -------------------------------------------------------------------------
    if not args.allow_outside_image and not (
        os.path.exists("/.dockerenv") or os.path.isdir("/opt/kernels")
    ):
        res["status"] = "precondition_failed"
        res["precondition_error"] = (
            "not running inside the serve image (no /.dockerenv and no /opt/kernels). P5 must be "
            "confirmed IN-IMAGE; pass --allow-outside-image only for host-side debugging."
        )
        return res

    try:
        import torch
    except Exception as exc:
        res["status"] = "precondition_failed"
        res["precondition_error"] = f"import torch failed: {type(exc).__name__}: {exc}"
        return res

    res["torch"] = {
        "version": torch.__version__,
        "hip": getattr(torch.version, "hip", None),
        "has_customAllocator": hasattr(torch._C, "_cuda_customAllocator"),
        "has_MemPool": hasattr(torch.cuda, "MemPool"),
        "has_use_mem_pool": hasattr(torch.cuda, "use_mem_pool"),
    }
    missing = [k for k, v in res["torch"].items() if k.startswith("has_") and not v]
    if missing:
        res["status"] = "precondition_failed"
        res["precondition_error"] = f"torch lacks required API(s): {missing}"
        return res
    if not torch.cuda.is_available():
        res["status"] = "precondition_failed"
        res["precondition_error"] = (
            "torch.cuda.is_available() is False -- ROCm device passthrough missing "
            "(--device /dev/kfd --device /dev/dri --group-add video)"
        )
        return res

    global _HIP
    try:
        _HIP = Hip()
        _KEEP.append(_HIP)
    except Exception as exc:
        res["status"] = "precondition_failed"
        res["precondition_error"] = f"libamdhip64 load failed: {type(exc).__name__}: {exc}"
        return res

    dev = args.device
    if dev >= torch.cuda.device_count():
        res["status"] = "precondition_failed"
        res["precondition_error"] = (
            f"--device {dev} but torch sees only {torch.cuda.device_count()} device(s)"
        )
        return res

    torch.cuda.set_device(dev)
    # Force the primary context to exist and stay alive BEFORE any VMM call.
    _KEEP.append(torch.zeros(1024, device=f"cuda:{dev}"))
    torch.cuda.synchronize(dev)

    props = torch.cuda.get_device_properties(dev)
    res["device"] = {
        "torch_index": dev,
        "rocr_visible_devices": os.environ.get("ROCR_VISIBLE_DEVICES"),
        "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
        "name": props.name,
        "gcn_arch": getattr(props, "gcnArchName", None),
        "total_memory_bytes": int(props.total_memory),
        "multi_processor_count": int(getattr(props, "multi_processor_count", 0)),
        "pci_bus_id": _HIP.pci_bus_id(dev),
        "hip_device_count": _HIP.device_count(),
        "all_visible": [
            {"index": i, "name": torch.cuda.get_device_properties(i).name,
             "pci_bus_id": _HIP.pci_bus_id(i)}
            for i in range(torch.cuda.device_count())
        ],
        "hip_mem_info_at_start": _HIP.mem_info(),
        "hip_current_device": _HIP.set_device(dev),
    }
    if res["device"]["hip_current_device"] != dev:
        res["status"] = "precondition_failed"
        res["precondition_error"] = (
            f"hipSetDevice({dev}) left the current device at "
            f"{res['device']['hip_current_device']}; the ctypes VMM calls and torch would target "
            "different cards and every number below would be attributed to the wrong GPU"
        )
        return res
    # Which PHYSICAL card is this? ROCR_VISIBLE_DEVICES maps torch index -> physical id.
    try:
        rocr = [int(x) for x in (os.environ.get("ROCR_VISIBLE_DEVICES") or "").split(",") if x != ""]
        res["device"]["physical_card_index"] = rocr[dev] if dev < len(rocr) else None
    except Exception:
        res["device"]["physical_card_index"] = None

    # ---- is expandable_segments actually LIVE? (assert on behaviour, not on the env var) --------
    alloc_backend = None
    try:
        alloc_backend = torch.cuda.get_allocator_backend()
    except Exception:
        pass
    expandable_effective = None
    snapshot_probe = {}
    try:
        probe_t = torch.empty(64 << 20, dtype=torch.uint8, device=f"cuda:{dev}")
        _KEEP.append(probe_t)
        segs = torch.cuda.memory_snapshot()
        flags = [s.get("is_expandable") for s in segs if "is_expandable" in s]
        snapshot_probe = {"n_segments": len(segs), "n_with_is_expandable_key": len(flags),
                          "any_expandable": (any(flags) if flags else None),
                          "first_segment_keys": sorted(segs[0].keys()) if segs else []}
        if flags:
            expandable_effective = bool(any(flags))
    except Exception as exc:
        snapshot_probe = {"error": f"{type(exc).__name__}: {exc}"}
    res["torch"]["allocator_backend"] = alloc_backend
    res["torch"]["expandable_segments_env"] = {
        "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        "PYTORCH_HIP_ALLOC_CONF": os.environ.get("PYTORCH_HIP_ALLOC_CONF"),
    }
    res["torch"]["expandable_segments_effective"] = expandable_effective
    res["torch"]["expandable_segments_snapshot_probe"] = snapshot_probe
    # Tri-state, and the verdict MUST honour it: a leg that could not confirm the setting is live
    # cannot be quoted as evidence that the arena "coexists with expandable_segments". Reporting
    # coexistence off an unread env var is the confident-wrong-number failure mode.
    if cfg["alloc_conf"] != EXPANDABLE:
        res["torch"]["expandable_segments_verified"] = None  # leg does not claim it
    else:
        res["torch"]["expandable_segments_verified"] = (
            True if expandable_effective is True else (False if expandable_effective is False
                                                       else None)
        )
    if cfg["alloc_conf"] == EXPANDABLE and expandable_effective is False:
        # Loud: the leg claims to test coexistence and the setting is not actually live.
        res["status"] = "precondition_failed"
        res["precondition_error"] = (
            "leg requires expandable_segments:True but memory_snapshot() reports no expandable "
            f"segment (env={res['torch']['expandable_segments_env']}, backend={alloc_backend}). "
            "Refusing to report a coexistence result that was not actually exercised."
        )
        return res
    if cfg["alloc_conf"] == EXPANDABLE and expandable_effective is None:
        # The snapshot did not carry an `is_expandable` key at all (older torch). The leg still
        # answers the foreign-pointer question, but its coexistence claim is UNVERIFIED and the
        # verdict downgrades `expandable_segments_coexists` to null because of this flag.
        print("[P5] WARNING: could not verify expandable_segments is live "
              f"(snapshot_probe={snapshot_probe}); coexistence will be reported as NOT VERIFIED",
              file=sys.stderr, flush=True)

    # ---- ARM: reservation ------------------------------------------------------------------------
    arm = res["arms"]["reservation"] = new_arm(
        "hipMemAddressReserve + hipMemCreate(%s) + hipMemMap + hipMemSetAccess" % cfg["backing"]
    )
    try:
        prop = _HIP.prop(cfg["backing"], dev)
        gran_min = _HIP.granularity(prop, hipMemAllocationGranularityMinimum)
        gran_rec = _HIP.granularity(prop, hipMemAllocationGranularityRecommended)
        gran = max(gran_min, 4096)
        # Headroom x3: torch's caching allocator rounds a mid-size request up to kLargeBuffer and
        # then SPLITS it, so the bytes it asks us for exceed the bytes it hands back. Undersizing
        # the reservation would show up as hipMalloc fallbacks -- i.e. a false FAIL.
        n_slots = args.reps + args.warmup + 2
        want = args.reserve_bytes or (3 * n_slots * args.slot_bytes)
        size = _round_up(want, gran)

        va = ctypes.c_void_p()
        _HIP.check(
            _HIP.lib.hipMemAddressReserve(
                ctypes.byref(va), ctypes.c_size_t(size), ctypes.c_size_t(gran), None,
                ctypes.c_ulonglong(0)),
            "hipMemAddressReserve",
        )
        handle = ctypes.c_void_p()
        _HIP.check(
            _HIP.lib.hipMemCreate(
                ctypes.byref(handle), ctypes.c_size_t(size), ctypes.byref(prop),
                ctypes.c_ulonglong(0)),
            f"hipMemCreate(location={cfg['backing']})",
        )
        _HIP.check(
            _HIP.lib.hipMemMap(va, ctypes.c_size_t(size), ctypes.c_size_t(0), handle,
                               ctypes.c_ulonglong(0)),
            "hipMemMap",
        )
        desc = HipMemAccessDesc()
        desc.location.type = hipMemLocationTypeDevice
        desc.location.id = dev
        desc.flags = hipMemAccessFlagsProtReadWrite
        _HIP.check(
            _HIP.lib.hipMemSetAccess(va, ctypes.c_size_t(size), ctypes.byref(desc),
                                     ctypes.c_size_t(1)),
            "hipMemSetAccess",
        )
        # anchored forever: never unmapped, never released, never address-freed
        _KEEP.extend([va, handle, prop, desc])
        # ASSERT ON THE OPERATION, NOT THE QUERY. Every call above can return hipSuccess and still
        # leave a stale/unbacked page behind (documented on this box). Write two distinct patterns
        # at head/middle/tail and read them back before this arm is allowed to pass -- otherwise
        # `host_backing_available: YES` would be reported with no evidence at all.
        vb = _HIP.validate_backing(int(va.value), int(size))
        _ARENA["base"] = int(va.value)
        _ARENA["size"] = int(size)
        _ARENA["cursor"] = 0
        _ARENA["device"] = dev
        arm["detail"] = {
            "granularity_minimum": gran_min,
            "granularity_recommended": gran_rec,
            "granularity_used": gran,
            "reserved_base": int(va.value),
            "reserved_bytes": int(size),
            "backing": cfg["backing"],
            "backing_data_validated": vb["validated"],
            "backing_validation": vb,
        }
        arm["passed"] = bool(vb["validated"])
        arm["status"] = "ok" if arm["passed"] else "failed"
        if not arm["passed"]:
            arm["error"] = (
                "the reservation was created and mapped with hipSuccess but does NOT store what "
                "is written to it (stale/unbacked page). Treating this as a backing failure."
            )
            res["status"] = "precondition_failed"
            res["precondition_error"] = (
                f"{cfg['backing']}-backed reservation is not usable memory: {arm['error']} "
                "(this is a VMM/backing failure, NOT a torch-plumbing answer)"
            )
            res["allocator_events"] = _events_summary()
            return res
    except Exception as exc:
        arm["status"] = "failed"
        arm["passed"] = False
        arm["error"] = f"{type(exc).__name__}: {exc}"
        res["status"] = "precondition_failed"
        res["precondition_error"] = (
            f"could not build a {cfg['backing']}-backed reservation: {arm['error']} "
            "(this is a VMM/backing failure, NOT a torch-plumbing answer)"
        )
        res["allocator_events"] = _events_summary()
        return res

    reserved_base = int(_ARENA["base"])

    # ---- build the custom allocator + the MemPool (both anchored forever) -----------------------
    try:
        alloc_cb, free_cb = _install_callbacks()
        allocator = torch._C._cuda_customAllocator(
            ctypes.cast(alloc_cb, ctypes.c_void_p).value,
            ctypes.cast(free_cb, ctypes.c_void_p).value,
        )
        _KEEP.append(allocator)
        try:
            pool = torch.cuda.MemPool(allocator)
        except TypeError:
            pool = torch.cuda.MemPool(allocator=allocator)
        _KEEP.append(pool)
        res["torch"]["mempool_id"] = str(getattr(pool, "id", None))
    except Exception as exc:
        res["status"] = "failed"
        res["error"] = f"MemPool/customAllocator construction failed: {type(exc).__name__}: {exc}"
        res["traceback"] = traceback.format_exc()
        res["allocator_events"] = _events_summary()
        return res

    dtype = getattr(torch, args.dtype, None)
    if not isinstance(dtype, torch.dtype):
        res["status"] = "precondition_failed"
        res["precondition_error"] = f"--dtype {args.dtype!r} is not a torch dtype"
        return res
    itemsize = torch.empty(0, dtype=dtype).element_size()
    n_elem = args.slot_bytes // itemsize
    if n_elem == 0:
        res["status"] = "precondition_failed"
        res["precondition_error"] = f"--slot-bytes {args.slot_bytes} < one {args.dtype} element"
        return res
    devstr = f"cuda:{dev}"

    # ---- ARM: pool_alloc -- the core claim, t.data_ptr() == reserved_base -----------------------
    arm = res["arms"]["pool_alloc"] = new_arm(
        "does torch.empty() under use_mem_pool() land on OUR reserved VA, and what does it cost?"
    )
    slot_tensors: list = []
    try:
        latencies_us: list[float] = []
        lat_cb_us: list[float] = []      # reps that actually went through OUR allocator
        lat_cached_us: list[float] = []  # reps torch served from its own cache (NOT an arena cost)
        recs: list[dict] = []
        total = args.warmup + args.reps
        for i in range(total):
            n_alloc_before = len(_ALLOC_EVENTS)
            t0 = time.perf_counter_ns()
            with torch.cuda.use_mem_pool(pool):
                t = torch.empty(n_elem, dtype=dtype, device=devstr)
            t1 = time.perf_counter_ns()
            _KEEP.append(t)  # anchored: a pool tensor must NEVER outlive the pool
            slot_tensors.append(t)
            new_events = _ALLOC_EVENTS[n_alloc_before:]
            ptr = t.data_ptr()
            rec = {
                "rep": i,
                "warmup": i < args.warmup,
                "data_ptr": ptr,
                "in_reservation": reserved_base <= ptr < reserved_base + int(_ARENA["size"]),
                "offset_from_base": ptr - reserved_base,
                "alloc_callbacks": new_events,
                "latency_us": (t1 - t0) / 1e3,
            }
            rec["hit_custom_allocator"] = bool(new_events)
            recs.append(rec)
            if i >= args.warmup:
                latencies_us.append(rec["latency_us"])
                (lat_cb_us if new_events else lat_cached_us).append(rec["latency_us"])

        first = recs[0]
        eq_base = first["data_ptr"] == reserved_base
        all_in = all(r["in_reservation"] for r in recs)
        # NOT every rep must trigger a callback: torch rounds a mid-size request up and splits the
        # block, so later reps can be served from the remainder. What must hold is that every
        # callback that DID fire was served from our reservation.
        n_cb = sum(len(r["alloc_callbacks"]) for r in recs)
        served = all(e.get("from_reservation") for r in recs for e in r["alloc_callbacks"])
        arm["detail"] = {
            "reserved_base": reserved_base,
            "first_data_ptr": first["data_ptr"],
            "first_data_ptr_equals_reserved_base": eq_base,
            "all_data_ptrs_inside_reservation": all_in,
            "n_alloc_callbacks": n_cb,
            "every_alloc_served_from_reservation": served,
            "hipMalloc_fallbacks": int(_ARENA["fallbacks"]),
            "reps": recs,
            # THREE stats, because the aggregate alone is a mislabelled number: torch's caching
            # allocator can serve a rep out of a previously-split block WITHOUT calling us, and
            # that rep costs ~a dict lookup. Quoting the mixed median as "the cost of an arena
            # allocation" would be wrong by an order of magnitude.
            "alloc_latency_us": _stats(latencies_us),
            "alloc_latency_us_via_custom_allocator": _stats(lat_cb_us),
            "alloc_latency_us_served_from_torch_cache": _stats(lat_cached_us),
            "n_measured_reps_hitting_custom_allocator": len(lat_cb_us),
            "latency_note": (
                "alloc_latency_us mixes reps that invoked the custom allocator with reps torch "
                "served from its own cache; quote alloc_latency_us_via_custom_allocator for the "
                "arena-allocation cost. Host wall time around torch.empty() only -- no device "
                "work is launched by an allocation, so no event fence is applicable."
            ),
        }
        arm["passed"] = bool(eq_base and all_in and served and n_cb > 0
                             and _ARENA["fallbacks"] == 0)
        arm["status"] = "ok" if arm["passed"] else "failed"
        if not arm["passed"]:
            arm["error"] = (
                "torch did NOT allocate over the foreign pointer "
                f"(first data_ptr=0x{first['data_ptr']:x} vs base=0x{reserved_base:x}, "
                f"fallbacks={_ARENA['fallbacks']})"
            )
    except Exception as exc:
        arm["status"] = "failed"
        arm["passed"] = False
        arm["error"] = f"{type(exc).__name__}: {exc}"
        arm["detail"]["traceback"] = traceback.format_exc()

    have_tensor = bool(slot_tensors)
    arena = slot_tensors[0] if have_tensor else None

    # ---- ARM: correctness -- a kernel reads and writes those pages through the torch tensor -----
    arm = res["arms"]["correctness"] = new_arm(
        "does a kernel read/write the foreign-VA tensor correctly (device write, device read, D2H)?"
    )
    if not have_tensor:
        arm["status"] = "skipped"
        arm["error"] = "no pool tensor (pool_alloc arm did not produce one)"
    else:
        try:
            ref = (torch.arange(n_elem, dtype=torch.int64) % 101).to(dtype)
            ref_dev = ref.to(devstr)
            # (a) copy path: H2D/D2D into the reserved VA, then an exact D2H compare. NOTE this
            #     alone may be served by the SDMA copy engine, so it is NOT proof a shader can
            #     write the pages -- arms (b) and (c) below supply that.
            arena.copy_(ref_dev)
            torch.cuda.synchronize(dev)
            back = arena.detach().to("cpu")
            exact_roundtrip = bool(torch.equal(back, ref))
            # (b) kernel READ through the reserved VA
            n_check = min(n_elem, 1 << 16)
            diff = (arena[:n_check] - ref_dev[:n_check]).abs().max()
            torch.cuda.synchronize(dev)
            max_abs_err = float(diff.item())
            # (c) kernel WRITE through the reserved VA (elementwise, in place), exact D2H compare.
            #     Two distinct contents in sequence: a stale page holding (a)'s bytes cannot also
            #     pass (c).
            arena.add_(1)
            torch.cuda.synchronize(dev)
            back2 = arena.detach().to("cpu")
            ref_plus1 = ((torch.arange(n_elem, dtype=torch.int64) % 101) + 1).to(dtype)
            kernel_write_ok = bool(torch.equal(back2, ref_plus1))
            arm["detail"] = {
                "elements": n_elem,
                "bytes": n_elem * itemsize,
                "dtype": str(dtype),
                "exact_d2h_roundtrip": exact_roundtrip,
                "kernel_read_max_abs_err": max_abs_err,
                "kernel_read_elements_checked": n_check,
                "kernel_write_exact_d2h_roundtrip": kernel_write_ok,
                "note": (
                    "three independent checks: copy-engine roundtrip, shader READ, shader WRITE. "
                    "Two different contents are compared in sequence so a stale physical page "
                    "cannot pass by coincidence."
                ),
            }
            arm["passed"] = bool(exact_roundtrip and max_abs_err == 0.0 and kernel_write_ok)
            arm["status"] = "ok" if arm["passed"] else "failed"
            if not arm["passed"]:
                arm["error"] = (
                    "foreign-VA tensor did not read back what was written "
                    f"(copy_roundtrip={exact_roundtrip}, kernel_read_max_abs_err={max_abs_err}, "
                    f"kernel_write_roundtrip={kernel_write_ok})"
                )
        except Exception as exc:
            arm["status"] = "failed"
            arm["passed"] = False
            arm["error"] = f"{type(exc).__name__}: {exc}"
            arm["detail"]["traceback"] = traceback.format_exc()

    # ---- ARM: empty_cache with a LIVE user-MemPool block ----------------------------------------
    # engine/graph.py:314 calls torch.cuda.empty_cache() immediately before capture, i.e. after the
    # offload arena exists. Does it hand our pages back through the free callback?
    arm = res["arms"]["empty_cache_live"] = new_arm(
        "does torch.cuda.empty_cache() invoke the custom free callback for a LIVE pool block, "
        "and does the tensor survive it?"
    )
    if not have_tensor:
        arm["status"] = "skipped"
        arm["error"] = "no pool tensor"
    else:
        try:
            per_rep = []
            survived = True
            ptr0 = arena.data_ptr()
            hv = HostVerify(_HIP, n_elem * itemsize)
            _KEEP.append(hv)
            # SELF-CONTAINED, and a DIFFERENT payload every rep. The previous version compared
            # against the pattern the `correctness` arm happened to leave behind, which (a) made a
            # benign correctness-arm failure cascade into a spurious "arena did not survive
            # empty_cache" -- i.e. straight into the KILL criterion -- and (b) could not detect a
            # stale/aliased page, because the expected bytes never changed between reps.
            for i in range(args.warmup + args.reps):
                probe_err = None
                pre_val = float((i % 97) + 1)   # written BEFORE empty_cache, must survive it
                post_val = float((i % 89) + 3)  # written AFTER, proves it is still writable
                same_ptr = False
                probe_ok = None
                writable_after = None
                host_pre = host_post = None
                torch_pre = torch_post = None
                n0 = n1 = 0
                t0 = t1 = time.perf_counter_ns()
                try:
                    arena.fill_(pre_val)  # shader write through the reserved VA
                    torch.cuda.synchronize(dev)
                    n0 = len(_FREE_EVENTS)
                    t0 = time.perf_counter_ns()
                    torch.cuda.empty_cache()  # <- the engine/graph.py:314 call, for real
                    t1 = time.perf_counter_ns()
                    n1 = len(_FREE_EVENTS)
                    same_ptr = arena.data_ptr() == ptr0
                    # Contents survived? HOST-verified over the WHOLE arena. See HostVerify: the
                    # device-vs-device form of this check is unsound on this box under
                    # expandable_segments, and it is wired to the KILL criterion.
                    host_pre = hv.equals_fill(arena.data_ptr(), n_elem, dtype, pre_val)
                    probe_ok = bool(host_pre["ok"])
                    # still writable? (a page that was handed back would fail here or fault)
                    arena.fill_(post_val)
                    torch.cuda.synchronize(dev)
                    host_post = hv.equals_fill(arena.data_ptr(), n_elem, dtype, post_val)
                    writable_after = bool(host_post["ok"])
                    # Recorded, NOT used as the pass condition: the same comparison done the naive
                    # device-vs-device way. A disagreement is evidence of the torch-allocator
                    # defect, not of an arena failure -- see torch_side_comparison below.
                    torch_pre = bool(torch.equal(arena, torch.full_like(arena, post_val)))
                    torch_post = torch_pre
                except Exception as exc:  # a use-after-free would land here (or abort the child)
                    probe_ok = False if probe_ok is None else probe_ok
                    writable_after = False
                    probe_err = f"{type(exc).__name__}: {exc}"
                survived = survived and same_ptr and bool(probe_ok) and bool(writable_after)
                per_rep.append(
                    {
                        "rep": i,
                        "warmup": i < args.warmup,
                        "free_callbacks": n1 - n0,
                        "data_ptr_stable": same_ptr,
                        "pre_value": pre_val,
                        "post_value": post_val,
                        "contents_still_correct": probe_ok,
                        "writable_after_empty_cache": writable_after,
                        "host_check_pre": host_pre,
                        "host_check_post": host_post,
                        "torch_side_equal_says": torch_post,
                        "probe_error": probe_err,
                        "empty_cache_us": (t1 - t0) / 1e3,
                    }
                )
            measured = [r for r in per_rep if r.get("warmup") is False]
            fired = any(r["free_callbacks"] > 0 for r in measured)
            n_torch_disagree = sum(
                1 for r in measured
                if r.get("torch_side_equal_says") is False
                and bool(r.get("writable_after_empty_cache"))
            )
            arm["detail"] = {
                "verification_method": (
                    "HOST: ctypes hipMemcpy D2H of the full arena + CPU byte compare. No device "
                    "allocation participates in the check."
                ),
                "torch_side_comparison": {
                    "n_measured_reps": len(measured),
                    "n_reps_torch_equal_disagreed_with_host": n_torch_disagree,
                    "note": (
                        "reps where the naive device-vs-device torch.equal() said FALSE while the "
                        "host byte compare said the arena was correct. Non-zero here is a defect "
                        "in torch's DEFAULT allocator under expandable_segments:True (it re-maps a "
                        "freed handle at an already-used VA and this driver serves a stale page), "
                        "NOT a defect in the arena. See p5_diagnostics/."
                    ),
                },
                "free_cb_invoked_for_live_block": fired,
                "free_callback_counts": [r["free_callbacks"] for r in measured],
                "tensor_survived": survived,
                "contents_survived": all(bool(r["contents_still_correct"]) for r in measured),
                "writable_after": all(bool(r["writable_after_empty_cache"]) for r in measured),
                "data_ptr_stable": all(bool(r["data_ptr_stable"]) for r in measured),
                "empty_cache_us": _stats([r["empty_cache_us"] for r in measured]),
                "empty_cache_us_note": (
                    "host wall around torch.cuda.empty_cache(); it synchronises internally. Not a "
                    "device-time measurement and not comparable to a kernel duration."
                ),
                "reps": per_rep,
            }
            # The arm PASSES if the arena survives. Whether the callback fires is the ANSWER, not
            # the pass condition -- a firing callback on a live block would be the dangerous case.
            arm["passed"] = bool(survived)
            arm["status"] = "ok" if survived else "failed"
            if not survived:
                arm["error"] = "the live pool tensor did not survive torch.cuda.empty_cache()"
        except Exception as exc:
            arm["status"] = "failed"
            arm["passed"] = False
            arm["error"] = f"{type(exc).__name__}: {exc}"
            arm["detail"]["traceback"] = traceback.format_exc()

    # ---- ARM: empty_cache with a CACHED (dropped, not live) pool block --------------------------
    arm = res["arms"]["empty_cache_cached"] = new_arm(
        "when a pool tensor is dropped, does the free callback fire on del, or only on empty_cache?"
    )
    if args.skip_cached_release:
        arm["status"] = "skipped"
        arm["error"] = "--skip-cached-release"
    else:
        try:
            n_before = len(_FREE_EVENTS)
            with torch.cuda.use_mem_pool(pool):
                tmp = torch.empty(n_elem, dtype=dtype, device=devstr)
            tmp_ptr = tmp.data_ptr()
            n_after_alloc = len(_FREE_EVENTS)
            del tmp  # refcount drop -> torch caches the block inside the pool
            n_after_del = len(_FREE_EVENTS)
            torch.cuda.empty_cache()
            n_after_empty = len(_FREE_EVENTS)
            arm["detail"] = {
                "tmp_ptr": tmp_ptr,
                "tmp_ptr_in_reservation": reserved_base <= tmp_ptr < reserved_base + int(_ARENA["size"]),
                "free_cb_on_alloc": n_after_alloc - n_before,
                "free_cb_on_del": n_after_del - n_after_alloc,
                "free_cb_on_empty_cache": n_after_empty - n_after_del,
                "freed_ptrs": [e.get("ptr") for e in _FREE_EVENTS[n_after_alloc:]],
            }
            arm["passed"] = True  # observational: any outcome is a valid answer
            arm["status"] = "ok"
        except Exception as exc:
            arm["status"] = "failed"
            arm["passed"] = False
            arm["error"] = f"{type(exc).__name__}: {exc}"
            arm["detail"]["traceback"] = traceback.format_exc()

    # ---- ARM: graph capture over the foreign-VA tensor ------------------------------------------
    arm = res["arms"]["graph_capture"] = new_arm(
        "can a CUDA graph be captured and replayed reading the foreign-VA tensor, with the user "
        "MemPool live (and does capture's internal empty_cache/gc.collect disturb it)?"
    )
    if not have_tensor:
        arm["status"] = "skipped"
        arm["error"] = "no pool tensor"
    else:
        try:
            n_view = min(n_elem, args.capture_elems)
            view = arena[:n_view]
            out = torch.empty(n_view, dtype=dtype, device=devstr)
            _KEEP.append(out)
            hv_cap = HostVerify(_HIP, n_view * itemsize)
            _KEEP.append(hv_cap)

            # warm the op on a side stream (mandatory before capture)
            s = torch.cuda.Stream(device=dev)
            s.wait_stream(torch.cuda.current_stream(dev))
            with torch.cuda.stream(s):
                for _ in range(3):
                    out.copy_(view * 2)
            torch.cuda.current_stream(dev).wait_stream(s)
            torch.cuda.synchronize(dev)

            free_before_capture = len(_FREE_EVENTS)
            g = torch.cuda.CUDAGraph()
            _KEEP.append(g)
            # torch.cuda.graph.__enter__ does synchronize() + gc.collect() + empty_cache() --
            # this is exactly the engine/graph.py:314 situation, exercised for real.
            with torch.cuda.graph(g):
                out.copy_(view * 2)
            free_during_capture = len(_FREE_EVENTS) - free_before_capture

            replay_us: list[float] = []
            event_ms: list[float] = []
            numeric_ok = True
            per_replay = []
            for i in range(args.warmup + args.replays):
                val = float((i % 41) + 1)
                view.fill_(val)  # device-side write through the reserved VA, outside the graph
                torch.cuda.synchronize(dev)
                ev0 = torch.cuda.Event(enable_timing=True)
                ev1 = torch.cuda.Event(enable_timing=True)
                t0 = time.perf_counter_ns()
                ev0.record()
                g.replay()
                ev1.record()
                torch.cuda.synchronize(dev)
                t1 = time.perf_counter_ns()
                # HOST-verified, for the same reason as empty_cache_live: torch.graph.__enter__
                # runs empty_cache(), so a fresh device-side `expect` tensor is exactly the
                # allocation the expandable_segments defect corrupts. Compare the graph's OUTPUT
                # bytes on the host instead.
                hcap = hv_cap.equals_fill(out.data_ptr(), n_view, dtype, val * 2)
                ok = bool(hcap["ok"])
                numeric_ok = numeric_ok and ok
                if i >= args.warmup:
                    replay_us.append((t1 - t0) / 1e3)
                    event_ms.append(float(ev0.elapsed_time(ev1)))
                per_replay.append({"rep": i, "warmup": i < args.warmup, "value": val,
                                   "correct": ok,
                                   "host_check": (None if ok else hcap)})

            arm["detail"] = {
                "captured": True,
                "capture_elems": n_view,
                "free_cb_calls_during_capture_enter": free_during_capture,
                "replay_numerics_correct": numeric_ok,
                "replay_wall_us": _stats(replay_us),
                "replay_event_ms": _stats(event_ms),
                "replays": per_replay,
                "data_ptr_after_capture": arena.data_ptr(),
                "data_ptr_stable_after_capture": arena.data_ptr() == reserved_base,
                "working_set_bytes": n_view * itemsize,
                "timing_note": (
                    "replay_wall_us / replay_event_ms are a FUNCTIONAL latency for a "
                    f"{n_view * itemsize} B read+write, which is L2/MALL-resident (MALL is 64 MB). "
                    "They are NOT a bandwidth measurement and must NOT be quoted as host-read or "
                    "device-read bandwidth -- that number is P1's job, on a >=256 MB working set. "
                    "Wall is fenced by torch.cuda.synchronize(); event time is the on-device span."
                ),
            }
            arm["passed"] = bool(numeric_ok and arena.data_ptr() == reserved_base)
            arm["status"] = "ok" if arm["passed"] else "failed"
            if not arm["passed"]:
                arm["error"] = "graph replay over the foreign-VA tensor produced wrong results"
        except Exception as exc:
            arm["status"] = "failed"
            arm["passed"] = False
            arm["error"] = f"{type(exc).__name__}: {exc}"
            arm["detail"]["traceback"] = traceback.format_exc()

    # ---- wrap up ---------------------------------------------------------------------------------
    res["allocator_events"] = _events_summary()
    res["hipMalloc_fallbacks_total"] = int(_ARENA["fallbacks"])
    res["arena_bytes_consumed"] = int(_ARENA["cursor"])
    res["arena_bytes_reserved"] = int(_ARENA["size"])
    res["device"]["hip_mem_info_at_end"] = _HIP.mem_info()
    res["box_state_end"] = box_state()
    required = ["reservation", "pool_alloc", "correctness", "empty_cache_live", "graph_capture"]
    res["passed"] = all(res["arms"][a]["passed"] is True for a in required)
    res["status"] = "ok" if res["passed"] else "failed"
    return res


def _events_summary() -> dict:
    return {
        "n_alloc_callbacks": len(_ALLOC_EVENTS),
        "n_free_callbacks": len(_FREE_EVENTS),
        "hipMalloc_fallbacks": int(_ARENA["fallbacks"]),
        "arena": dict(_ARENA),
        "alloc_events": _ALLOC_EVENTS[:64],
        "free_events": _FREE_EVENTS[:64],
    }


# =================================================================================================
# PARENT: orchestrate legs, merge, report
# =================================================================================================
def child_env(cfg: dict, rocr: str | None = None) -> dict:
    env = dict(os.environ)
    rocr = rocr or os.environ.get("ROCR_VISIBLE_DEVICES") or "0,1"
    env["ROCR_VISIBLE_DEVICES"] = rocr
    # The child re-forces device visibility at import time, BEFORE torch, from this variable --
    # without it the child would silently reset to the "0,1" literal and ignore --rocr-devices.
    env["MINISGL_P5_ROCR_DEVICES"] = rocr
    env.pop("HIP_VISIBLE_DEVICES", None)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env["PYTHONUNBUFFERED"] = "1"
    if cfg["alloc_conf"]:
        env["PYTORCH_CUDA_ALLOC_CONF"] = cfg["alloc_conf"]
        env["PYTORCH_HIP_ALLOC_CONF"] = cfg["alloc_conf"]
    else:
        env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        env.pop("PYTORCH_HIP_ALLOC_CONF", None)
    return env


def run_children(args, legs: list[str]) -> dict:
    out: dict = {}
    for leg in legs:
        cfg = LEGS[leg]
        leg_json = os.path.join(args.out_dir, f"p5_leg_{leg}.json")
        leg_log = os.path.join(args.out_dir, f"p5_leg_{leg}.log")
        for p in (leg_json, leg_log):
            try:
                os.unlink(p)
            except FileNotFoundError:
                pass
        cmd = [
            sys.executable, os.path.abspath(__file__),
            "--child-leg", leg,
            "--out-dir", args.out_dir,
            "--device", str(args.device),
            "--slot-bytes", str(args.slot_bytes),
            "--reps", str(args.reps),
            "--warmup", str(args.warmup),
            "--replays", str(args.replays),
            "--capture-elems", str(args.capture_elems),
            "--dtype", args.dtype,
            "--reserve-bytes", str(args.reserve_bytes),
            "--rocr-devices", args.rocr_devices,
        ]
        if args.allow_outside_image:
            cmd.append("--allow-outside-image")
        if args.skip_cached_release:
            cmd.append("--skip-cached-release")
        print(f"[P5] leg {leg}: alloc_conf={cfg['alloc_conf'] or '<unset>'} "
              f"backing={cfg['backing']}", flush=True)
        t0 = time.time()
        proc = subprocess.run(cmd, env=child_env(cfg, args.rocr_devices), capture_output=True,
                              text=True, timeout=args.leg_timeout)
        dt = time.time() - t0
        try:
            with open(leg_log, "w") as fh:
                fh.write(f"$ {' '.join(cmd)}\n--- stdout ---\n{proc.stdout}\n"
                         f"--- stderr ---\n{proc.stderr}\n--- rc={proc.returncode} ---\n")
        except Exception:
            pass
        if os.path.exists(leg_json):
            with open(leg_json) as fh:
                leg_res = json.load(fh)
        else:
            leg_res = leg_skeleton(leg, cfg, args)
            leg_res["status"] = "no_output"
        leg_res["child"] = {
            "returncode": proc.returncode,
            "aborted": proc.returncode in (-6, 134, -11, 139),
            "wall_s": dt,
            "log": leg_log,
            "stderr_tail": proc.stderr[-4000:],
            "stdout_tail": proc.stdout[-2000:],
        }
        if proc.returncode in (-6, 134, -11, 139):
            leg_res["status"] = "aborted"
            leg_res["passed"] = False
            leg_res["error"] = (
                f"child process died with returncode {proc.returncode} -- see {leg_log}. "
                "This is the c10::Error/abort class the probe guards against."
            )
        out[leg] = leg_res
        print(f"[P5] leg {leg}: status={leg_res.get('status')} rc={proc.returncode} "
              f"({dt:.1f}s)", flush=True)
    return out


def _tri(vals: list) -> bool | None:
    """AND over tri-state observations: None if nothing was observed."""
    obs = [v for v in vals if v is not None]
    if not obs:
        return None
    return all(bool(v) for v in obs)


def _any_tri(vals: list) -> bool | None:
    obs = [v for v in vals if v is not None]
    if not obs:
        return None
    return any(bool(v) for v in obs)


def build_verdict(legs: dict) -> dict:
    reasons: list[str] = []

    def arm_pass(leg: dict, name: str):
        a = leg.get("arms", {}).get(name, {})
        return a.get("passed")

    def arm_detail(leg: dict, name: str, key: str):
        return leg.get("arms", {}).get(name, {}).get("detail", {}).get(key)

    ran = {k: v for k, v in legs.items() if v.get("status") not in ("not_run",)}
    usable = {k: v for k, v in ran.items() if v.get("status") in ("ok", "failed", "aborted")}

    # The KILL criterion is scoped: "in a leg whose RESERVATION arm PASSED, does torch.empty()
    # under use_mem_pool land on the reserved VA". So evaluate the claim over exactly those legs,
    # and AND them -- an earlier _any_tri() would have reported foreign_ptr_ok=True off one lucky
    # non-gating control leg while a gating leg was demonstrably failing the same claim.
    reserved_legs = {k: v for k, v in usable.items() if arm_pass(v, "reservation") is True}
    foreign = _tri([arm_pass(v, "pool_alloc") for v in reserved_legs.values()])
    # Capture is only a meaningful question in a leg that got a foreign-VA tensor at all.
    ptr_legs = {k: v for k, v in reserved_legs.items() if arm_pass(v, "pool_alloc") is True}
    capture = _tri([arm_pass(v, "graph_capture") for v in ptr_legs.values()])
    per_leg_claims = {
        k: {
            "reservation": arm_pass(v, "reservation"),
            "backing_data_validated": arm_detail(v, "reservation", "backing_data_validated"),
            "pool_alloc": arm_pass(v, "pool_alloc"),
            "correctness": arm_pass(v, "correctness"),
            "empty_cache_live": arm_pass(v, "empty_cache_live"),
            "graph_capture": arm_pass(v, "graph_capture"),
            "hipMalloc_fallbacks_total": v.get("hipMalloc_fallbacks_total"),
            "status": v.get("status"),
        }
        for k, v in ran.items()
    }

    # Coexistence may only be claimed by a leg that VERIFIED expandable_segments is actually live.
    exp_legs = {
        k: v for k, v in usable.items()
        if LEGS.get(k, {}).get("alloc_conf") == EXPANDABLE
        and v.get("torch", {}).get("expandable_segments_verified") is True
    }
    exp_unverified = [
        k for k, v in usable.items()
        if LEGS.get(k, {}).get("alloc_conf") == EXPANDABLE
        and v.get("torch", {}).get("expandable_segments_verified") is not True
    ]
    coexist = None
    if exp_legs:
        coexist = _tri(
            [arm_pass(v, "pool_alloc") for v in exp_legs.values()]
            + [arm_pass(v, "graph_capture") for v in exp_legs.values()]
        )
    if exp_unverified:
        reasons.append(
            "expandable_segments could NOT be verified live in leg(s) %s (memory_snapshot carried "
            "no is_expandable key); their coexistence evidence is discarded" % sorted(exp_unverified)
        )

    live_cb = _any_tri(
        [arm_detail(v, "empty_cache_live", "free_cb_invoked_for_live_block") for v in usable.values()]
    )
    cached_cb = _any_tri(
        [
            (None if arm_detail(v, "empty_cache_cached", "free_cb_on_empty_cache") is None
             else arm_detail(v, "empty_cache_cached", "free_cb_on_empty_cache") > 0)
            for v in usable.values()
        ]
    )
    survives = _tri([arm_detail(v, "empty_cache_live", "tensor_survived") for v in usable.values()])

    # COLLATERAL FINDING, not a P5 gate: reps where the naive device-vs-device torch.equal()
    # contradicted the host byte compare. That is torch's DEFAULT allocator handing back a stale
    # page after empty_cache() under expandable_segments:True -- a serve-wide hazard, since
    # engine/graph.py:314 calls empty_cache() and the compose default sets expandable_segments.
    tsc = {}
    for k, v in ran.items():
        d = arm_detail(v, "empty_cache_live", "torch_side_comparison")
        if isinstance(d, dict):
            tsc[k] = d.get("n_reps_torch_equal_disagreed_with_host")
    torch_defect = any(isinstance(n, int) and n > 0 for n in tsc.values())
    if torch_defect:
        reasons.append(
            "COLLATERAL (does NOT gate P5): torch's DEFAULT allocator returned stale/zero memory "
            "after empty_cache() under expandable_segments:True in leg(s) %s. The arena itself was "
            "host-verified correct in every rep. See p5_diagnostics/." % sorted(
                k for k, n in tsc.items() if isinstance(n, int) and n > 0)
        )

    # "host backing available" means host-located pages behind a device VA that actually STORE
    # data -- not merely that hipMemCreate returned hipSuccess.
    host_legs = {k: v for k, v in ran.items() if LEGS.get(k, {}).get("backing") == "host"}
    host_ok = None
    if host_legs:
        host_ok = _any_tri(
            [arm_detail(v, "reservation", "backing_data_validated") for v in host_legs.values()]
        )

    gating_present = [k for k in GATING_LEGS if k in legs]
    gating_ok = True
    for k in gating_present:
        st = legs[k].get("status")
        if st != "ok":
            gating_ok = False
            why = legs[k].get("error") or legs[k].get("precondition_error") or st
            reasons.append(f"gating leg {k}: {st} -- {why}")
    if not gating_present:
        gating_ok = False
        reasons.append("no gating leg was run (need at least one of %s)" % sorted(GATING_LEGS))

    fallbacks = {k: v.get("hipMalloc_fallbacks_total") for k, v in ran.items()}
    any_fallback = any(isinstance(n, int) and n > 0 for n in fallbacks.values())
    if any_fallback:
        reasons.append(
            "hipMalloc FALLBACKS occurred (%s): torch was handed memory that is NOT inside our "
            "reservation. Any latency or capture result from such a leg is suspect." % fallbacks
        )

    # `survives is True`, not `is not False`: an unmeasured survival is not a pass. graph.py:314
    # calling empty_cache() on the live arena is the single most destructive failure mode here.
    p5_pass = bool(gating_ok and foreign is True and capture is True and survives is True
                   and not any_fallback)
    if foreign is False:
        reasons.append("t.data_ptr() != reserved_base in every leg that ran -- "
                       "KILL P5: a C++ from_blob extension is required (+3 days)")
    elif foreign is None:
        reasons.append("the foreign-pointer claim was NOT MEASURED (no leg reached the arm) -- "
                       "this is not an answer, re-run the probe")
    if capture is False:
        reasons.append("graph capture/replay over the foreign-VA tensor did not pass")
    elif capture is None:
        reasons.append("graph capture over the foreign-VA tensor was NOT MEASURED")
    if survives is False:
        reasons.append("the arena tensor did NOT survive torch.cuda.empty_cache() "
                       "(engine/graph.py:314 would destroy the offload arena)")
    elif survives is None:
        reasons.append("arena survival across torch.cuda.empty_cache() was NOT MEASURED")
    if host_ok is False:
        reasons.append("host-located hipMemCreate pages behind a device VA did NOT validate "
                       "(written bytes did not read back) -- the T1 100%-host tier is not "
                       "available by this route")
    if p5_pass and not reasons:
        reasons.append("all gating legs green: foreign pointer, expandable_segments, capture, "
                       "and empty_cache survival")

    return {
        "p5_pass": p5_pass,
        "foreign_ptr_ok": foreign,
        "expandable_segments_coexists": coexist,
        "capture_ok": capture,
        "empty_cache_invokes_free_cb_live_block": live_cb,
        "empty_cache_invokes_free_cb_cached_block": cached_cb,
        "tensor_survives_empty_cache": survives,
        "host_backing_available": host_ok,
        "torch_default_allocator_stale_after_empty_cache": (torch_defect if tsc else None),
        "torch_side_disagreement_reps_per_leg": tsc,
        "hipMalloc_fallbacks_seen": any_fallback,
        "hipMalloc_fallbacks_per_leg": fallbacks,
        "per_leg_claims": per_leg_claims,
        "legs_with_validated_reservation": sorted(reserved_legs),
        "legs_with_verified_expandable_segments": sorted(exp_legs),
        "reasons": reasons,
        "gating_legs": sorted(gating_present),
    }


def classify_exit(verdict: dict, legs: dict) -> tuple[int, dict]:
    """Separate 'the answer is NO' (2, a KILL) from 'the probe could not run' (3, re-run).

    The previous rule was `any_precond and foreign_ptr_ok is None -> 3, else 2`, which produced a
    FALSE KILL in a case the brief names explicitly: if `expandable_host` cannot build a
    host-located reservation (an exit-3 condition, "host-located hipMemCreate unsupported") while
    `expandable_device` measures the foreign pointer fine, `foreign_ptr_ok` is not None and the run
    exits 2 -- reporting a KILL for something that was never measured. Attribution now drives it.
    """
    kill: list[str] = []
    cannot_run: list[str] = []

    if verdict.get("foreign_ptr_ok") is False:
        kill.append("foreign_ptr_ok is False (t.data_ptr() != reserved_base / hipMalloc fallback)")
    if verdict.get("capture_ok") is False:
        kill.append("capture_ok is False (graph replay over the foreign VA was numerically wrong)")
    if verdict.get("tensor_survives_empty_cache") is False:
        kill.append("tensor_survives_empty_cache is False (graph.py:314 would destroy the arena)")
    if verdict.get("hipMalloc_fallbacks_seen") is True:
        kill.append("hipMalloc fallbacks occurred inside the user MemPool")

    for name in sorted(GATING_LEGS):
        leg = legs.get(name)
        if leg is None:
            cannot_run.append(f"gating leg {name} was not run")
            continue
        st = leg.get("status")
        if st == "aborted":
            # the c10::Error 'invalid device pointer' class -- the brief calls this a KILL
            kill.append(f"gating leg {name} ABORTED (rc={leg.get('child', {}).get('returncode')})")
        elif st in ("precondition_failed", "no_output", "not_run"):
            cannot_run.append(f"gating leg {name}: {st} -- "
                              f"{leg.get('precondition_error') or leg.get('error')}")
        elif st == "crashed":
            cannot_run.append(f"gating leg {name}: crashed before producing an answer -- "
                              f"{leg.get('error')}")
        elif st == "failed":
            failed_arms = [a for a, v in leg.get("arms", {}).items()
                           if v.get("passed") is False]
            kill.append(f"gating leg {name} failed arm(s) {failed_arms}")

    if verdict.get("p5_pass"):
        return EXIT_OK, {"code": EXIT_OK, "kill_signals": [], "cannot_run": cannot_run,
                         "why": "all gating legs green"}
    if kill:
        return EXIT_MEASURED_FAIL, {"code": EXIT_MEASURED_FAIL, "kill_signals": kill,
                                    "cannot_run": cannot_run,
                                    "why": "a measured NEGATIVE -- this is a real KILL"}
    return EXIT_PRECONDITION, {
        "code": EXIT_PRECONDITION, "kill_signals": [], "cannot_run": cannot_run,
        "why": "the probe could not run to an answer -- NOT a kill, re-run (see cannot_run)",
    }


def render_md(report: dict) -> str:
    v = report["verdict"]
    bs = report.get("box_state", {})
    mem = bs.get("meminfo", {})

    def gb(k):
        x = mem.get(k)
        return f"{x/1024/1024:.1f} GB" if isinstance(x, int) else "n/a"

    def tri(x):
        return {True: "YES", False: "NO", None: "not measured"}[x if x in (True, False) else None]

    lines = [
        "# P5 — torch over a foreign device pointer (in the serve image, under capture)",
        "",
        f"**Run:** {bs.get('t_iso')} · host `{bs.get('hostname')}` · "
        f"in-container `{bs.get('in_container')}` · schema `{report['schema']}`",
        (
            "**THIS IS A SELFTEST ARTIFACT — no GPU was touched and NOTHING below was measured.**"
            if report.get("synthetic")
            else f"**Verdict:** **{'PASS' if v['p5_pass'] else 'FAIL'}** "
                 f"(exit {report['exit_code']})"
        ),
        "",
        "## Answers",
        "",
        "| Question | Answer |",
        "|---|---|",
        f"| `t.data_ptr() == reserved_base` under `use_mem_pool` | {tri(v['foreign_ptr_ok'])} |",
        f"| coexists with `expandable_segments:True` (compose default) | "
        f"{tri(v['expandable_segments_coexists'])} |",
        f"| graph capture + replay over the foreign VA is correct | {tri(v['capture_ok'])} |",
        f"| arena tensor survives `torch.cuda.empty_cache()` (graph.py:314) | "
        f"{tri(v['tensor_survives_empty_cache'])} |",
        f"| `empty_cache()` invokes the custom **free** cb for a **live** pool block | "
        f"{tri(v['empty_cache_invokes_free_cb_live_block'])} |",
        f"| `empty_cache()` invokes it for a **cached (dropped)** pool block | "
        f"{tri(v['empty_cache_invokes_free_cb_cached_block'])} |",
        f"| host-located `hipMemCreate` pages behind a device VA (**bytes verified**, not just "
        f"`hipSuccess`) | {tri(v['host_backing_available'])} |",
        f"| any `hipMalloc` fallback inside the user MemPool | "
        f"{tri(v.get('hipMalloc_fallbacks_seen'))} |",
        f"| **COLLATERAL** — torch's *own* allocator returns stale memory after `empty_cache()` "
        f"under `expandable_segments:True` | "
        f"{tri(v.get('torch_default_allocator_stale_after_empty_cache'))} "
        f"(reps/leg: {v.get('torch_side_disagreement_reps_per_leg')}) |",
        "",
        f"**Exit classification:** {v.get('exit_classification', {}).get('why', 'n/a')}",
        "",
        "**Reasons**",
        "",
    ]
    lines += [f"- {r}" for r in v["reasons"]]
    lines += [
        "",
        "## Legs",
        "",
        "| leg | alloc conf | backing | status | reservation (bytes verified) | pool_alloc | "
        "correctness | empty_cache | capture | granularity B | alloc µs (median, via our "
        "allocator) | replay µs (median) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, leg in report["legs"].items():
        cfg = LEGS.get(name, {})
        arms = leg.get("arms", {})

        def a(n):
            p = arms.get(n, {}).get("passed")
            return {True: "ok", False: "FAIL", None: "-"}[p if p in (True, False) else None]

        rdet = arms.get("reservation", {}).get("detail", {}) or {}
        pdet = arms.get("pool_alloc", {}).get("detail", {}) or {}
        # the CALLBACK-only figure: the mixed one includes reps torch served from its own cache
        al = pdet.get("alloc_latency_us_via_custom_allocator") or pdet.get("alloc_latency_us") or {}
        rp = arms.get("graph_capture", {}).get("detail", {}).get("replay_wall_us", {}) or {}
        fmt = lambda x: (f"{x:.1f}" if isinstance(x, (int, float)) else "-")  # noqa: E731
        lines.append(
            f"| {name} | {cfg.get('alloc_conf') or '<unset>'} | {cfg.get('backing')} | "
            f"{leg.get('status')} | {a('reservation')} | {a('pool_alloc')} | {a('correctness')} | "
            f"{a('empty_cache_live')} | {a('graph_capture')} | "
            f"{rdet.get('granularity_minimum', '-')} | {fmt(al.get('median'))} | "
            f"{fmt(rp.get('median'))} |"
        )
    lines += [
        "",
        "> Replay times are a functional latency over an L2/MALL-resident working set. They are "
        "**not** a bandwidth measurement — that is P1's job, on a ≥256 MB working set.",
        "> `alloc µs` counts only reps that actually reached the custom allocator; reps torch "
        "served from its own cache are reported separately in the JSON.",
    ]

    lines += ["", "## Box state at run start", "",
              f"- MemTotal {gb('MemTotal_kB')}, MemAvailable {gb('MemAvailable_kB')}, "
              f"MemFree {gb('MemFree_kB')}, Cached {gb('Cached_kB')}",
              f"- swap: SwapTotal {gb('SwapTotal_kB')}, SwapFree {gb('SwapFree_kB')}, "
              f"pswpout {bs.get('vmstat', {}).get('pswpout')}",
              f"- loadavg `{bs.get('loadavg')}`",
              ""]
    for name, leg in report["legs"].items():
        d = leg.get("device", {})
        if d:
            lines.append(
                f"- leg `{name}` ran on torch index {d.get('torch_index')} = "
                f"physical card {d.get('physical_card_index')} "
                f"(`{d.get('name')}`, PCI `{d.get('pci_bus_id')}`, "
                f"{d.get('multi_processor_count')} WGPs, "
                f"{(d.get('total_memory_bytes') or 0) / 1e9:.1f} GB, "
                f"ROCR_VISIBLE_DEVICES=`{d.get('rocr_visible_devices')}`)"
            )
    lines += [
        "",
        "Raw: `p5.json` (this directory), per-leg `p5_leg_<name>.json` + `.log`.",
        "",
    ]
    return "\n".join(lines)


def _chown_if_asked(path: str) -> None:
    uid = os.environ.get("P5_CHOWN_UID")
    gid = os.environ.get("P5_CHOWN_GID")
    if uid and gid and os.getuid() == 0:
        try:
            os.chown(path, int(uid), int(gid))
        except Exception:
            pass


def write_json(path: str, obj: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=2, sort_keys=False, default=str)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    _chown_if_asked(path)


def write_text(path: str, text: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    _chown_if_asked(path)


# =================================================================================================
# selftest
# =================================================================================================
def synthetic_report(args) -> dict:
    """A structurally complete report with NO measured numbers. Every number is null."""
    legs = {}
    for leg in parse_legs(args.legs):
        cfg = LEGS[leg]
        sk = leg_skeleton(leg, cfg, args)
        sk["status"] = "selftest"
        sk["synthetic"] = True
        for name in ARM_NAMES:
            sk["arms"][name]["status"] = "selftest"
            sk["arms"][name]["question"] = "selftest placeholder"
        legs[leg] = sk
    report = {
        "schema": SCHEMA_VERSION,
        "probe": PROBE_ID,
        "kind": "selftest",
        "synthetic": True,
        "note": "SELFTEST OUTPUT -- no GPU was touched, no value here was measured.",
        "argv": sys.argv,
        "config": vars(args),
        "box_state": box_state(),
        "legs": legs,
        "verdict": {
            "p5_pass": False,
            "foreign_ptr_ok": None,
            "expandable_segments_coexists": None,
            "capture_ok": None,
            "empty_cache_invokes_free_cb_live_block": None,
            "empty_cache_invokes_free_cb_cached_block": None,
            "tensor_survives_empty_cache": None,
            "host_backing_available": None,
            "hipMalloc_fallbacks_seen": None,
            "hipMalloc_fallbacks_per_leg": {},
            "per_leg_claims": {},
            "legs_with_validated_reservation": [],
            "legs_with_verified_expandable_segments": [],
            "reasons": ["selftest: nothing was measured"],
            "gating_legs": sorted(GATING_LEGS & set(legs)),
        },
        "exit_code": EXIT_OK,
    }
    return report


def selftest(args) -> int:
    problems: list[str] = []
    # 1. leg table sanity
    for name, cfg in LEGS.items():
        if cfg["backing"] not in ("host", "device"):
            problems.append(f"leg {name}: bad backing")
        if cfg["alloc_conf"] not in ("", EXPANDABLE):
            problems.append(f"leg {name}: bad alloc_conf")
    if not GATING_LEGS <= set(LEGS):
        problems.append("GATING_LEGS references an unknown leg")
    # 2. leg parsing rejects garbage
    try:
        parse_legs("not_a_leg")
        problems.append("parse_legs accepted an unknown leg name")
    except SystemExit:
        pass
    except ValueError:
        pass
    # 3. child env composition
    e_exp = child_env(LEGS["expandable_host"])
    e_pln = child_env(LEGS["plain_host"])
    if e_exp.get("PYTORCH_CUDA_ALLOC_CONF") != EXPANDABLE:
        problems.append("expandable leg did not set PYTORCH_CUDA_ALLOC_CONF")
    if e_exp.get("PYTORCH_HIP_ALLOC_CONF") != EXPANDABLE:
        problems.append("expandable leg did not set PYTORCH_HIP_ALLOC_CONF")
    if "PYTORCH_CUDA_ALLOC_CONF" in e_pln or "PYTORCH_HIP_ALLOC_CONF" in e_pln:
        problems.append("plain leg leaked an alloc-conf env var")
    if e_exp.get("ROCR_VISIBLE_DEVICES") != "0,1":
        problems.append(f"ROCR_VISIBLE_DEVICES not forced (got {e_exp.get('ROCR_VISIBLE_DEVICES')})")
    if "HIP_VISIBLE_DEVICES" in e_exp:
        problems.append("HIP_VISIBLE_DEVICES leaked into the child env")
    # --rocr-devices must actually reach the child (the child re-forces visibility at import time
    # from MINISGL_P5_ROCR_DEVICES, before torch, so this variable is the only channel).
    e_alt = child_env(LEGS["expandable_host"], "1")
    if e_alt.get("MINISGL_P5_ROCR_DEVICES") != "1" or e_alt.get("ROCR_VISIBLE_DEVICES") != "1":
        problems.append("child_env did not propagate an explicit ROCR device set")
    if "2" in [s.strip() for s in (e_exp.get("ROCR_VISIBLE_DEVICES") or "").split(",")]:
        problems.append("ROCm device 2 (the Ryzen iGPU) is in the child's visible set")
    # 4. ctypes struct sizes match the HIP ABI
    if ctypes.sizeof(HipMemLocation) != 8:
        problems.append(f"HipMemLocation is {ctypes.sizeof(HipMemLocation)} bytes, want 8")
    if ctypes.sizeof(HipMemAllocationProp) != 32:
        problems.append(
            f"HipMemAllocationProp is {ctypes.sizeof(HipMemAllocationProp)} bytes, want 32")
    if ctypes.sizeof(HipMemAccessDesc) != 12:
        problems.append(f"HipMemAccessDesc is {ctypes.sizeof(HipMemAccessDesc)} bytes, want 12")
    # 5. stats helper never fabricates
    empty = _stats([])
    if empty["median"] is not None or empty["n"] != 0:
        problems.append("_stats([]) fabricated a value")
    five = _stats([3.0, 1.0, 2.0, 5.0, 4.0])
    if five["median"] != 3.0 or five["min"] != 1.0 or five["max"] != 5.0:
        problems.append("_stats gave the wrong median/spread")
    # 6. the bump allocator's arithmetic, with no GPU in sight
    saved = dict(_ARENA)
    try:
        _ARENA.update({"base": 0x100000, "size": 4096, "cursor": 0, "fallbacks": 0})
        a_cb, f_cb = _install_callbacks()
        p1 = a_cb(1000, 0, None)
        p2 = a_cb(1000, 0, None)
        if p1 != 0x100000:
            problems.append(f"bump alloc did not return the base (got {p1:#x})")
        if p2 != 0x100000 + 1024:
            problems.append(f"bump alloc did not 512-align (got {p2:#x})")
        n_fb = int(_ARENA["fallbacks"])
        a_cb(1 << 30, 0, None)  # cannot be served; _HIP is None -> records a fallback, no NULL crash
        if int(_ARENA["fallbacks"]) != n_fb + 1:
            problems.append("oversize request was not recorded as a fallback")
        n_free = len(_FREE_EVENTS)
        f_cb(p1, 1000, 0, None)
        if len(_FREE_EVENTS) != n_free + 1:
            problems.append("free callback did not record the call")
    finally:
        _ARENA.clear()
        _ARENA.update(saved)
        _ALLOC_EVENTS.clear()
        _FREE_EVENTS.clear()
    # 7. verdict builder is tri-state clean on an empty world
    v = build_verdict({})
    if v["p5_pass"] is not False or v["foreign_ptr_ok"] is not None:
        problems.append("build_verdict({}) invented a result")

    # 7b. exit classification: a KILL must be attributable to a MEASURED negative, never to a leg
    #     that could not run. Both directions are tested because getting this wrong once reported a
    #     kill for something that was never measured.
    def _leg(status, arms=None, rc=0):
        return {"status": status, "arms": arms or {}, "child": {"returncode": rc}}

    ok_arms = {a: {"passed": True} for a in ARM_NAMES}
    bad_alloc = dict(ok_arms, pool_alloc={"passed": False})
    # (i) host reservation unsupported + device leg fine  -> 3 (could not run), NOT a kill
    legs_i = {
        "expandable_host": _leg("precondition_failed"),
        "expandable_device": _leg("ok", ok_arms),
    }
    c_i, why_i = classify_exit(build_verdict(legs_i), legs_i)
    if c_i != EXIT_PRECONDITION:
        problems.append(f"a gating leg that COULD NOT RUN was classified {c_i}, want "
                        f"{EXIT_PRECONDITION} ({why_i})")
    # (ii) the pointer claim measurably fails -> 2 (kill)
    legs_ii = {"expandable_host": _leg("failed", bad_alloc),
               "expandable_device": _leg("failed", bad_alloc)}
    c_ii, _ = classify_exit(build_verdict(legs_ii), legs_ii)
    if c_ii != EXIT_MEASURED_FAIL:
        problems.append(f"a measured foreign-pointer failure was classified {c_ii}, want "
                        f"{EXIT_MEASURED_FAIL}")
    # (iii) a gating leg ABORT (the c10::Error class) -> 2 (kill)
    legs_iii = {"expandable_host": _leg("aborted", {}, rc=-6),
                "expandable_device": _leg("ok", ok_arms)}
    c_iii, _ = classify_exit(build_verdict(legs_iii), legs_iii)
    if c_iii != EXIT_MEASURED_FAIL:
        problems.append(f"an ABORTED gating leg was classified {c_iii}, want {EXIT_MEASURED_FAIL}")
    # (iv) foreign_ptr_ok must not be rescued by a non-gating control leg
    legs_iv = {"expandable_host": _leg("failed", bad_alloc),
               "plain_host": _leg("ok", ok_arms)}
    v_iv = build_verdict(legs_iv)
    if v_iv["foreign_ptr_ok"] is not False:
        problems.append("a passing control leg masked a failing gating leg in foreign_ptr_ok")
    # (v) unmeasured survival must not pass
    no_ec = dict(ok_arms)
    no_ec["empty_cache_live"] = {"passed": True, "detail": {}}
    legs_v = {"expandable_host": _leg("ok", no_ec), "expandable_device": _leg("ok", no_ec)}
    if build_verdict(legs_v)["p5_pass"] is not False:
        problems.append("p5_pass was granted without a measured empty_cache survival")
    # 8. full report shape
    report = synthetic_report(args)
    report["exit_code"] = EXIT_OK if not problems else EXIT_PRECONDITION
    try:
        validate_report(report)
    except Exception as exc:
        problems.append(f"schema validation failed: {exc}")
    # 9. markdown renders
    try:
        md = render_md(report)
        if "P5" not in md:
            problems.append("markdown render lost its title")
    except Exception as exc:
        problems.append(f"render_md raised: {type(exc).__name__}: {exc}")
        md = ""
    # 10. output paths are writable, but NEVER onto the real p5.json
    os.makedirs(args.out_dir, exist_ok=True)
    jpath = os.path.join(args.out_dir, "p5_selftest.json")
    mpath = os.path.join(args.out_dir, "p5_selftest.md")
    report["problems"] = problems
    write_json(jpath, report)
    write_text(mpath, md)

    print(json.dumps({"selftest": "ok" if not problems else "FAILED", "problems": problems,
                      "wrote": [jpath, mpath]}, indent=2))
    if problems:
        print("SELFTEST FAILED", file=sys.stderr)
        return EXIT_PRECONDITION
    return EXIT_OK


# =================================================================================================
# CLI
# =================================================================================================
def parse_legs(spec: str) -> list[str]:
    names = [s.strip() for s in spec.split(",") if s.strip()]
    if not names:
        raise ValueError("no legs requested")
    bad = [n for n in names if n not in LEGS]
    if bad:
        raise ValueError(f"unknown leg(s) {bad}; known: {sorted(LEGS)}")
    return names


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                   help="where p5.json / p5.md / per-leg files land (default: %(default)s)")
    p.add_argument("--legs", default=DEFAULT_LEGS,
                   help=f"comma-separated subset of {sorted(LEGS)} (default: %(default)s)")
    p.add_argument("--device", type=int, default=0,
                   help="torch device index WITHIN the ROCR_VISIBLE_DEVICES set (default 0)")
    p.add_argument("--slot-bytes", type=int, default=8 << 20,
                   help="bytes per pool allocation (default 8 MiB, ~3x the 2.8 MiB expert granule)")
    p.add_argument("--reserve-bytes", type=int, default=0,
                   help="override the VA reservation size in bytes (0 = derive from slots)")
    p.add_argument("--reps", type=int, default=6,
                   help="measured repetitions (>=5 required) (default %(default)s)")
    p.add_argument("--warmup", type=int, default=1, help="discarded warm-up reps (default 1)")
    p.add_argument("--replays", type=int, default=20,
                   help="measured graph replays (default %(default)s)")
    p.add_argument("--capture-elems", type=int, default=1 << 20,
                   help="elements of the arena the captured graph reads (default 1Mi)")
    p.add_argument("--dtype", default="float32",
                   help="torch dtype name for the arena tensor (default %(default)s)")
    p.add_argument("--leg-timeout", type=int, default=900, help="per-leg child timeout, seconds")
    p.add_argument("--allow-outside-image", action="store_true",
                   help="permit running outside the serve image (debugging only; P5 must be "
                        "confirmed IN-IMAGE)")
    p.add_argument("--skip-cached-release", action="store_true",
                   help="skip the dropped-block empty_cache arm")
    p.add_argument("--rocr-devices", default="0,1",
                   help="ROCR_VISIBLE_DEVICES to force (default %(default)s; ROCm2 is the iGPU and "
                        "must never be enumerated)")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--selftest", action="store_true",
                   help="validate args, allocator arithmetic and JSON shape WITHOUT touching a GPU")
    g.add_argument("--dry-run", action="store_true", help="alias for --selftest")
    g.add_argument("--child-leg", default=None,
                   help=argparse.SUPPRESS)  # internal: run ONE leg in this process
    return p


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    if args.dry_run:
        args.selftest = True

    os.environ["ROCR_VISIBLE_DEVICES"] = args.rocr_devices
    os.environ.pop("HIP_VISIBLE_DEVICES", None)
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    if args.reps < 5 and not args.selftest:
        print(f"FATAL: --reps {args.reps} < 5; the spec requires >= 5 measured reps",
              file=sys.stderr)
        return EXIT_PRECONDITION
    if args.warmup < 1 and not args.selftest:
        print("FATAL: --warmup must be >= 1 (a warm-up iteration must be discarded)",
              file=sys.stderr)
        return EXIT_PRECONDITION
    try:
        legs = parse_legs(args.legs)
    except ValueError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return EXIT_PRECONDITION

    os.makedirs(args.out_dir, exist_ok=True)
    # A root-owned output DIRECTORY is the trap from CLAUDE.md: the files inside can be chowned but
    # a later non-root rm still fails on the directory's write bit. Chown the dir too.
    _chown_if_asked(args.out_dir)

    # ---- selftest --------------------------------------------------------------------------------
    if args.selftest:
        return selftest(args)

    # ---- child mode: ONE leg, in this process ---------------------------------------------------
    if args.child_leg:
        leg = args.child_leg
        if leg not in LEGS:
            print(f"FATAL: unknown leg {leg!r}", file=sys.stderr)
            return EXIT_PRECONDITION
        path = os.path.join(args.out_dir, f"p5_leg_{leg}.json")
        try:
            res = run_leg(leg, LEGS[leg], args)
        except BaseException as exc:
            res = leg_skeleton(leg, LEGS[leg], args)
            res["status"] = "crashed"
            res["passed"] = False
            res["error"] = f"{type(exc).__name__}: {exc}"
            res["traceback"] = traceback.format_exc()
            res["box_state"] = box_state()
            res["allocator_events"] = _events_summary()
        write_json(path, res)
        print(json.dumps({"leg": leg, "status": res.get("status"), "passed": res.get("passed"),
                          "error": res.get("error") or res.get("precondition_error"),
                          "json": path}, indent=2, default=str), flush=True)
        code = EXIT_OK if res.get("passed") else (
            EXIT_PRECONDITION if res.get("status") in ("precondition_failed", "crashed")
            else EXIT_MEASURED_FAIL)
        sys.stdout.flush()
        sys.stderr.flush()
        # HARD exit: no destructor may run. A MemPool teardown with live blocks aborts the process,
        # which would corrupt the exit code we just computed.
        os._exit(code)

    # ---- parent mode: run every leg, merge ------------------------------------------------------
    started = box_state()
    try:
        leg_results = run_children(args, legs)
    except subprocess.TimeoutExpired as exc:
        print(f"FATAL: a leg exceeded --leg-timeout {args.leg_timeout}s: {exc}", file=sys.stderr)
        return EXIT_PRECONDITION

    verdict = build_verdict(leg_results)
    code, code_reason = classify_exit(verdict, leg_results)
    verdict["exit_classification"] = code_reason

    report = {
        "schema": SCHEMA_VERSION,
        "probe": PROBE_ID,
        "kind": "measurement",
        "synthetic": False,
        "title": "P5 -- torch over a foreign device pointer, in the serve image, under capture",
        "plan_ref": "docs/WEIGHT_OFFLOAD_PLAN.md section 3, row P5",
        "argv": sys.argv,
        "config": vars(args),
        "box_state": started,
        "box_state_end": box_state(),
        "legs": leg_results,
        "verdict": verdict,
        "exit_code": code,
    }
    try:
        validate_report(report)
    except Exception as exc:
        print(f"FATAL: the report I built does not validate: {exc}", file=sys.stderr)
        report["schema_error"] = str(exc)
        write_json(os.path.join(args.out_dir, "p5.INVALID.json"), report)
        return EXIT_PRECONDITION

    jpath = os.path.join(args.out_dir, "p5.json")
    mpath = os.path.join(args.out_dir, "p5.md")
    write_json(jpath, report)
    write_text(mpath, render_md(report))
    print(json.dumps(report, indent=2, default=str))
    print(f"\n[P5] wrote {jpath}\n[P5] wrote {mpath}\n[P5] verdict: "
          f"{'PASS' if verdict['p5_pass'] else 'FAIL'} (exit {code})", file=sys.stderr)
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
