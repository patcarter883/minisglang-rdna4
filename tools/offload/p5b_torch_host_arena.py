#!/usr/bin/env python3
"""P5b -- torch over a hipHostGetDevicePointer address, in-image, under graph capture.

Question (Phase 0 report section 6, unknown #3; blocks M1-A):

  P5 answered "can torch hand out tensors on a VA we backed ourselves?" -- but it answered it
  over a `hipMemAddressReserve` + `hipMemCreate` reservation, and Phase 0 subsequently proved
  that `hipMemCreate(location.type = hipMemLocationTypeHost)` SILENTLY RETURNS DEVICE VRAM on
  this box (P1 + P2 + P3, three independent methods). So P5's "host" legs were device memory and
  **the SELECTED T1 mechanism has never been tested through torch at all.**

  The selected mechanism is:  hipHostMalloc(..., hipHostMallocMapped)  +  hipHostGetDevicePointer

  P5b re-runs P5's arms with the reservation replaced by that mechanism, and adds the placement
  proof P5 never needed. It must show, IN THE SERVE IMAGE:

    (a) torch._C._cuda_customAllocator + torch.cuda.MemPool + torch.cuda.use_mem_pool yields
        t.data_ptr() == the HOST-DERIVED DEVICE POINTER,
    (b) zero hipMalloc fallbacks inside the pool,
    (c) the arena survives torch.cuda.empty_cache()  (engine/graph.py:314 calls it),
    (d) the custom free callback does NOT fire for a live pool block,
    (e) tensors over it replay EXACTLY under full graph capture (>= 20 replays, bit-identical),
    (f) all of the above concurrently with PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
        (the compose default, docker-compose.yml:68),
    (g) on BOTH physical cards -- P2, P5 and P6 only ever ran card 0 and this box has burned
        people twice on cross-card assumptions.

  If P5b fails, M1-A needs a C++ `from_blob` extension instead (+3 days). SAY SO CLEARLY rather
  than working around it -- that sentence is in the brief and it is the whole point of the probe.

PLACEMENT IS NOT ASSUMED -- IT IS PROVEN, THREE WAYS, EVERY LEG
---------------------------------------------------------------
The first P2 was invalidated by exactly one mistake: it trusted a location field and a return
code. `hipMemCreate(location=Host)` returned hipSuccess, `hipMemGetAllocationPropertiesFromHandle`
echoed "Host" back verbatim, every kernel read the pages correctly -- and the pages were VRAM. A
bandwidth number taken from that arm is an HBM figure wearing a host-memory label.

So no arm in this probe is allowed to say "host" because an API said so. Every leg measures:

  1. VRAM DELTA -- hipMemGetInfo free-VRAM before/after the allocation and again after a full
     device-side touch (drivers commit lazily), plus, when the per-card amdgpu sysfs counters are
     readable, mem_info_vram_used / mem_info_gtt_used for THIS EXACT PCI BDF. The sysfs pair is
     the only signal that unrelated host-RAM churn from the other agents on this box cannot move.
  2. MemAvailable DELTA -- /proc/meminfo, before/after. Noisy on a shared box, so it is a
     corroborating signal, never the sole one.
  3. ACHIEVED BANDWIDTH vs the 28.7 GB/s PCIe ceiling -- a >= 256 MiB kernel read (never smaller;
     the MALL is 64 MB and P1's withdrawn 127.2 GB/s was a 2 MiB L2 artifact), checksummed so a
     number can never be printed for a read that did not land on the pages under test, measured
     against an in-process HBM reference and an in-process copy-engine reference on the SAME
     card. Bytes that crossed PCIe cannot materially outrun the copy engine (1.5x) nor approach
     HBM (0.25x). A candidate breaching either bound is DEVICE memory whatever the counters said.
  4. CPU ACCESSIBILITY -- real host pages are readable and writable by the CPU. The fake VMM
     "host" pages were `---s` on renderD128 and `os.write` returned EFAULT. This is the sharpest
     single discriminator available, and it is probed through a syscall against a REGULAR FILE so
     a bad pointer yields EFAULT instead of SIGSEGV (/dev/null is NOT a valid sink -- its write
     handler never touches the user pages and reports every pointer as readable).

If placement is not proven, the leg reports `precondition_failed` and the run exits 3, because a
torch-plumbing answer measured over the wrong medium is not an answer. It is NOT reported as a
KILL: a KILL means "we asked torch and torch said no".

KNOWN ABORT, GUARDED
--------------------
A tensor that outlives its MemPool raises `c10::Error: invalid device pointer` inside
HIPCachingAllocator and *aborts the process* (SIGABRT), destroying the measurement. Guards, all
mandatory and all present below:

  * the two CFUNCTYPE trampolines, the _cuda_CUDAAllocator, the MemPool, every tensor cut from
    it, and the host arena itself are module-level singletons in `_KEEP`, never cleared;
  * the free callback is a pure no-op -- it never returns a page and NEVER calls hipHostFree;
  * the arena is allocated ONCE, before the allocator can be invoked; the alloc callback is a
    forward-only bump pointer inside it and NEVER returns NULL (an unservable request falls back
    to hipMalloc and is recorded as NOT served from the arena, so the arm reports a clean FAIL
    instead of a use-after-null deep inside torch);
  * results are written and fsynced BEFORE any teardown and the process leaves via os._exit();
  * each leg runs in its own child process, so an abort we failed to anticipate is recorded by
    the parent (returncode -6) instead of losing the run.

LAYOUT
------
Parent (never imports torch): box state, one child per leg, merge into p5b.json + p5b.md.
Child (`--child-leg NAME`): the measurement; writes p5b_leg_<NAME>.json and .log.

USAGE
-----
    tools/offload/p5b_run.sh                     # canonical: in the serve image, all legs
    python3 tools/offload/p5b_torch_host_arena.py --selftest    # no GPU, arg + JSON-shape check
"""

from __future__ import annotations

# --- device visibility: forced before ANY chance of a torch/HIP import ---------------------------
# ROCm device 2 on this box is the Ryzen iGPU advertising ~47 GB of GTT. If it enters enumeration
# it poisons every "biggest free pool" heuristic and it is never a compute target.
import os as _os

_os.environ["ROCR_VISIBLE_DEVICES"] = _os.environ.get("MINISGL_P5B_ROCR_DEVICES", "0,1")
_os.environ.pop("HIP_VISIBLE_DEVICES", None)
_os.environ.pop("CUDA_VISIBLE_DEVICES", None)
_os.environ.pop("GPU_DEVICE_ORDINAL", None)

import argparse
import ctypes
import ctypes.util
import hashlib
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import tempfile
import time
import traceback

SCHEMA_VERSION = "p5b/1"
PROBE_ID = "P5b"

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
DEFAULT_OUT_DIR = os.path.join(_REPO_ROOT, "docs", "measurements", "WEIGHT_OFFLOAD_2026-09-02")
DEFAULT_SCRATCH_DIR = os.path.join(_HERE, "_build")

EXPANDABLE = "expandable_segments:True"

# leg -> (alloc-conf, torch device index within ROCR_VISIBLE_DEVICES). "" means the vars are UNSET.
# Legs cross the compose default on/off with BOTH physical cards. Phase 0 section 6 item 9: "card 1
# was never exercised by P2, P5 or P6" -- so a card-1 leg is not optional here, it is the point.
LEGS: dict[str, dict] = {
    "expandable_card0": {"alloc_conf": EXPANDABLE, "device": 0},
    "expandable_card1": {"alloc_conf": EXPANDABLE, "device": 1},
    "plain_card0": {"alloc_conf": "", "device": 0},
    "plain_card1": {"alloc_conf": "", "device": 1},
}
DEFAULT_LEGS = "expandable_card0,expandable_card1,plain_card0"

# Failure in these makes P5b red. The compose default is expandable_segments:True and the brief
# requires BOTH cards, so both expandable legs gate. A "plain" leg is a control.
GATING_LEGS = {"expandable_card0", "expandable_card1"}

EXIT_OK = 0
EXIT_MEASURED_FAIL = 2  # torch was asked and the answer is NO -> C++ from_blob extension, +3 days
EXIT_PRECONDITION = 3  # the probe could not run -- no number was produced

# --- HIP constants (from /opt/rocm/include/hip) --------------------------------------------------
hipHostMallocPortable = 0x1
hipHostMallocMapped = 0x2
hipHostMallocNonCoherent = 0x80000000
hipMemcpyHostToDevice = 1
hipMemcpyDeviceToHost = 2
HIP_MEMORY_TYPE = {0: "Host", 1: "Device", 2: "Array", 3: "Unified", 4: "Managed"}

# --- placement thresholds (identical to P1, deliberately -- one classifier, one meaning) ---------
PLACEMENT_VRAM_DEVICE_FRAC = 0.5     # VRAM grew by >= this x size -> device-resident
PLACEMENT_VRAM_HOST_FRAC = 0.25      # VRAM grew by <  this x size -> not in VRAM
PLACEMENT_HOST_SIGNAL_FRAC = 0.5     # GTT or MemAvailable moved >= this x size -> host RAM

# --- physics gates (identical to P1) -------------------------------------------------------------
# Bytes that cross PCIe cannot materially outrun the copy engine, and cannot come close to HBM.
PROVENANCE_MAX_OVER_COPY = 1.5      # arena read > this x measured copy engine -> NOT crossing PCIe
PROVENANCE_MAX_FRAC_OF_HBM = 0.25   # arena read >= this x measured HBM ref    -> in VRAM
# hipEvent timings are cross-checked against wall over the whole rep loop. Wall strictly exceeds
# device time, so event GB/s may exceed wall GB/s only by fixed per-rep host overhead; beyond this
# factor the TIMER produced the number, not the kernel.
TIMING_SANITY_FACTOR = 4.0

# Measured on this box (P1/P4, mid-DMA link sampling -- NEVER read link speed at idle, ASPM lies):
#   card 0 = RX 9070 XT, 0000:03:00.0, root port 0000:00:01.1, Gen5 x8 -> 28.70 GB/s H2D
#   card 1 = RX 9070,    0000:07:00.0, root port 0000:00:01.3, Gen4 x8 -> 14.34 GB/s H2D
# These are EXPECTATIONS used to flag a surprise, not gates -- the gates are the in-process
# copy-engine and HBM references measured on the same card in the same process.
PCIE_EXPECTED_GBPS = {0: 28.70, 1: 14.34}
PCIE_ABS_CEILING_GBPS = 28.70   # the fastest link on this box; nothing host-resident may beat it
PCIE_CEILING_SLACK = 1.15       # 15 % measurement slack before we call it impossible

# --- module-level anti-GC anchor. NEVER cleared, NEVER shrunk. -----------------------------------
_KEEP: list = []
_ALLOC_EVENTS: list = []
_FREE_EVENTS: list = []
# The arena is allocated ONCE, before any callback can fire ("one-shot alloc"): the callback is a
# forward-only bump pointer over it and never asks the driver for more host memory.
_ARENA = {"base": 0, "host": 0, "size": 0, "cursor": 0, "fallbacks": 0, "device": None}
_HIP = None  # set by the child before the allocator can ever be called


class ProbeError(RuntimeError):
    pass


# =================================================================================================
# HIP ctypes layer -- exactly the entry points the SELECTED mechanism needs
# =================================================================================================
class HipError(RuntimeError):
    pass


class Hip:
    def __init__(self) -> None:
        path = ctypes.util.find_library("amdhip64") or "libamdhip64.so"
        self.lib = ctypes.CDLL(path)
        self.so_path = path
        L = self.lib
        vp, vpp, sz = ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t
        L.hipGetErrorString.restype = ctypes.c_char_p
        L.hipGetErrorString.argtypes = [ctypes.c_int]
        for name, args in (
            ("hipGetDeviceCount", [ctypes.POINTER(ctypes.c_int)]),
            ("hipDeviceGetPCIBusId", [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]),
            ("hipSetDevice", [ctypes.c_int]),
            ("hipGetDevice", [ctypes.POINTER(ctypes.c_int)]),
            ("hipMemGetInfo", [ctypes.POINTER(sz), ctypes.POINTER(sz)]),
            ("hipMalloc", [vpp, sz]),
            ("hipHostMalloc", [vpp, sz, ctypes.c_uint]),
            ("hipHostGetDevicePointer", [vpp, vp, ctypes.c_uint]),
            ("hipHostFree", [vp]),
            ("hipMemcpy", [vp, vp, sz, ctypes.c_int]),
            ("hipMemset", [vp, ctypes.c_int, sz]),
            ("hipDeviceSynchronize", []),
            ("hipPointerGetAttributes", [vp, vp]),
        ):
            if not hasattr(L, name):
                raise HipError(f"libamdhip64 lacks {name} -- P5b cannot run on this runtime")
            getattr(L, name).argtypes = args
            getattr(L, name).restype = ctypes.c_int

    # -- error handling ---------------------------------------------------------------------------
    def err(self, rc: int) -> str:
        try:
            s = self.lib.hipGetErrorString(ctypes.c_int(rc))
            return s.decode() if s else f"rc={rc}"
        except Exception:
            return f"rc={rc}"

    def check(self, rc: int, what: str) -> None:
        if rc != 0:
            raise HipError(f"{what} failed: rc={rc} ({self.err(rc)})")

    # -- device helpers ---------------------------------------------------------------------------
    def device_count(self) -> int:
        n = ctypes.c_int(0)
        self.check(self.lib.hipGetDeviceCount(ctypes.byref(n)), "hipGetDeviceCount")
        return int(n.value)

    def pci_bus_id(self, dev: int) -> str:
        buf = ctypes.create_string_buffer(64)
        rc = self.lib.hipDeviceGetPCIBusId(buf, ctypes.c_int(64), ctypes.c_int(dev))
        return buf.value.decode() if rc == 0 else f"<hipDeviceGetPCIBusId rc={rc}>"

    def set_device(self, dev: int) -> int:
        """Pin libamdhip64's current device for OUR ctypes calls. torch.cuda.set_device does not
        necessarily bind this thread's HIP context, and a mismatch would attribute every number
        below to the wrong physical card."""
        self.check(self.lib.hipSetDevice(ctypes.c_int(dev)), f"hipSetDevice({dev})")
        cur = ctypes.c_int(-1)
        self.lib.hipGetDevice(ctypes.byref(cur))
        return int(cur.value)

    def mem_info(self) -> dict:
        free, total = ctypes.c_size_t(0), ctypes.c_size_t(0)
        rc = self.lib.hipMemGetInfo(ctypes.byref(free), ctypes.byref(total))
        if rc != 0:
            return {"free_bytes": None, "total_bytes": None, "rc": rc}
        return {"free_bytes": int(free.value), "total_bytes": int(total.value)}

    def free_vram(self) -> int:
        return int(self.mem_info().get("free_bytes") or 0)

    def sync(self) -> None:
        self.check(self.lib.hipDeviceSynchronize(), "hipDeviceSynchronize")

    # -- THE SELECTED MECHANISM -------------------------------------------------------------------
    def host_alloc(self, nbytes: int, flags: int) -> tuple[int, int]:
        """hipHostMalloc(...Mapped) -> hipHostGetDevicePointer. Returns (host_ptr, device_ptr).

        This is the ONLY mechanism Phase 0 found that puts real host pages behind an address a
        kernel can read (28.93 GB/s card 0, 14.48 GB/s card 1). It is a SEPARATE VA from any
        device allocation and cannot be interleaved with one -- that is why the design is two
        stacks and why the plan's mixed device/host VA is not constructible.
        """
        hp = ctypes.c_void_p()
        self.check(self.lib.hipHostMalloc(ctypes.byref(hp), ctypes.c_size_t(nbytes),
                                          ctypes.c_uint(flags)),
                   f"hipHostMalloc({nbytes} B, flags=0x{flags:x})")
        dp = ctypes.c_void_p()
        self.check(self.lib.hipHostGetDevicePointer(ctypes.byref(dp), hp, ctypes.c_uint(0)),
                   "hipHostGetDevicePointer")
        if not hp.value or not dp.value:
            raise HipError("hipHostMalloc/hipHostGetDevicePointer returned hipSuccess with a "
                           "NULL pointer")
        return int(hp.value), int(dp.value)

    def dev_alloc(self, nbytes: int) -> int:
        p = ctypes.c_void_p()
        self.check(self.lib.hipMalloc(ctypes.byref(p), ctypes.c_size_t(nbytes)),
                   f"hipMalloc({nbytes})")
        return int(p.value)

    def memset(self, dptr: int, val: int, nbytes: int) -> None:
        self.check(self.lib.hipMemset(ctypes.c_void_p(dptr), ctypes.c_int(val),
                                      ctypes.c_size_t(nbytes)), "hipMemset")

    def memcpy(self, dst: int, src: int, nbytes: int, kind: int) -> None:
        self.check(self.lib.hipMemcpy(ctypes.c_void_p(dst), ctypes.c_void_p(src),
                                      ctypes.c_size_t(nbytes), ctypes.c_int(kind)), "hipMemcpy")

    def pointer_attrs(self, ptr: int) -> dict:
        """hipPointerGetAttributes, RECORDED BUT NEVER TRUSTED ALONE.

        The struct is read out of an oversized zeroed buffer at fixed offsets rather than through a
        declared ctypes.Structure, so an ABI change in a future ROCm cannot overflow anything --
        the worst case is a garbage field, and this value gates nothing.
        """
        buf = (ctypes.c_ubyte * 256)()
        rc = self.lib.hipPointerGetAttributes(ctypes.cast(buf, ctypes.c_void_p),
                                              ctypes.c_void_p(ptr))
        if rc != 0:
            return {"rc": rc, "error": self.err(rc),
                    "note": "hipPointerGetAttributes refused this pointer"}
        raw = bytes(buf)
        mt = int.from_bytes(raw[0:4], "little")
        dev = int.from_bytes(raw[4:8], "little", signed=True)
        dptr = int.from_bytes(raw[8:16], "little")
        hptr = int.from_bytes(raw[16:24], "little")
        return {
            "rc": 0, "memory_type": mt, "memory_type_name": HIP_MEMORY_TYPE.get(mt, f"?{mt}"),
            "device": dev,
            "device_pointer": f"0x{dptr:x}" if dptr else None,
            "host_pointer": f"0x{hptr:x}" if hptr else None,
            "caveat": ("a query can succeed with a fabricated value -- this box echoed "
                       "location.type=Host back verbatim for pages that were VRAM. Gates nothing."),
        }

    # -- byte-level validation --------------------------------------------------------------------
    def validate_backing(self, dptr: int, size: int, chunk: int = 1 << 20) -> dict:
        """Prove the pages actually STORE what is written, at head/middle/tail, through the DEVICE
        pointer.

        Two DIFFERENT patterns per offset, with position-dependent markers at every 4 KiB boundary:
        a stale page that happened to hold pattern A cannot also hold pattern B, and a page-level
        aliasing mixup cannot alias into a pass. Nothing here passes on a return code.
        """
        chunk = min(chunk, size)
        offsets = sorted({0, max(0, (size // 2) - (size // 2) % 4096), size - chunk})
        offsets = [o for o in offsets if 0 <= o <= size - chunk]
        src = (ctypes.c_ubyte * chunk)()
        dst = (ctypes.c_ubyte * chunk)()
        results, ok_all = [], True
        for off in offsets:
            per_off = {"offset": off, "chunk_bytes": chunk, "patterns": []}
            for pat_idx, seed in enumerate((0x5A, 0xA5)):
                ctypes.memset(src, seed, chunk)
                for i in range(0, chunk, 4096):
                    src[i] = (seed + (off >> 12) + (i >> 12) + pat_idx) & 0xFF
                    src[min(i + 4095, chunk - 1)] = (seed ^ ((i >> 12) & 0xFF)) & 0xFF
                ctypes.memset(dst, 0, chunk)
                rc_w = self.lib.hipMemcpy(ctypes.c_void_p(dptr + off),
                                          ctypes.cast(src, ctypes.c_void_p),
                                          ctypes.c_size_t(chunk),
                                          ctypes.c_int(hipMemcpyHostToDevice))
                rc_s = self.lib.hipDeviceSynchronize()
                rc_r = self.lib.hipMemcpy(ctypes.cast(dst, ctypes.c_void_p),
                                          ctypes.c_void_p(dptr + off),
                                          ctypes.c_size_t(chunk),
                                          ctypes.c_int(hipMemcpyDeviceToHost))
                rc_s2 = self.lib.hipDeviceSynchronize()
                match = bytes(src) == bytes(dst)
                ok_all = ok_all and match and rc_w == 0 and rc_r == 0 and rc_s == 0 and rc_s2 == 0
                per_off["patterns"].append({
                    "pattern": pat_idx, "seed": seed, "bytes_match": bool(match),
                    "rc_h2d": int(rc_w), "rc_d2h": int(rc_r), "rc_sync": [int(rc_s), int(rc_s2)],
                })
            results.append(per_off)
        return {"validated": bool(ok_all), "offsets_checked": offsets, "detail": results}


def _round_up(x: int, a: int) -> int:
    return ((x + a - 1) // a) * a


# =================================================================================================
# PLACEMENT PROOF -- the part the first P2 did not have
# =================================================================================================
SYSFS_MEM_FIELDS = ("mem_info_vram_used", "mem_info_vram_total", "mem_info_gtt_used",
                    "mem_info_gtt_total")


def _read_text(path: str) -> str | None:
    try:
        with open(path) as fh:
            return fh.read()
    except OSError:
        return None


def sysfs_amdgpu_cards() -> dict:
    """PCI BDF -> that card's amdgpu memory accounting.

    PRIMARY placement signal, for the reason P1 documents: hipMemGetInfo's "free VRAM" does not
    necessarily account for every allocation path, and /proc/meminfo MemAvailable is whole-box
    noise on a machine that is never idle (other agents lease these same two cards). amdgpu
    exports, PER CARD, mem_info_vram_used (bytes committed in that card's HBM) and mem_info_gtt_used
    (bytes of HOST RAM pinned and GPU-mapped for it). Neither can be perturbed by unrelated host
    churn. hipHostMalloc(Mapped) pages should move gtt_used and leave vram_used flat.
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
    if not bdf:
        return {}
    return sysfs_amdgpu_cards().get(bdf.strip().lower(), {})


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


def mem_available_bytes() -> int:
    kb = _parse_kv_kb(_read_text("/proc/meminfo"), ("MemAvailable",))["MemAvailable"]
    return (kb or 0) * 1024


def placement_sample(hip: "Hip", bdf: str | None) -> dict:
    """One (VRAM, MemAvailable, per-card sysfs) sample. Take one before, one after the allocation,
    and one after every page has been touched -- drivers commit pages lazily, so a delta measured
    at allocation time can be ~0 for BOTH media and prove nothing."""
    return {
        "t": time.time(),
        "hip_free_vram_bytes": hip.free_vram(),
        "mem_available_bytes": mem_available_bytes(),
        "sysfs": card_mem(bdf),
    }


def classify_placement(size: int, before: dict, after: dict) -> dict:
    """Where do this region's physical pages actually live?

    Deltas are CONSUMPTION: (free before - free after) for VRAM/MemAvailable, and
    (used after - used before) for the sysfs counters. Every signal is recorded even when it is
    not the one that decided, so a disagreement is visible in the artifact rather than averaged
    away. This function deliberately mirrors P1's classifier -- one classifier, one meaning.
    """
    vram_delta = int((before.get("hip_free_vram_bytes") or 0)
                     - (after.get("hip_free_vram_bytes") or 0))
    avail_delta = int((before.get("mem_available_bytes") or 0)
                      - (after.get("mem_available_bytes") or 0))
    rec: dict = {
        "size_bytes": size,
        "vram_consumed_bytes": vram_delta,
        "vram_consumed_frac": round(vram_delta / size, 4) if size else None,
        "memavailable_consumed_bytes": avail_delta,
        "memavailable_consumed_frac": round(avail_delta / size, 4) if size else None,
    }

    sb, sa = before.get("sysfs") or {}, after.get("sysfs") or {}
    have = all(x is not None for x in (sb.get("mem_info_vram_used"), sa.get("mem_info_vram_used"),
                                       sb.get("mem_info_gtt_used"), sa.get("mem_info_gtt_used")))
    sys_cls = None
    if have:
        sv = int(sa["mem_info_vram_used"]) - int(sb["mem_info_vram_used"])
        sg = int(sa["mem_info_gtt_used"]) - int(sb["mem_info_gtt_used"])
        rec.update({
            "sysfs_card": sa.get("pci_bdf") or sb.get("pci_bdf"),
            "sysfs_vram_used_delta_bytes": sv, "sysfs_gtt_used_delta_bytes": sg,
            "sysfs_vram_used_delta_frac": round(sv / size, 4) if size else None,
            "sysfs_gtt_used_delta_frac": round(sg / size, 4) if size else None,
        })
        if sv >= PLACEMENT_VRAM_DEVICE_FRAC * size:
            sys_cls = "device_resident"
        elif sg >= PLACEMENT_HOST_SIGNAL_FRAC * size and sv < PLACEMENT_VRAM_HOST_FRAC * size:
            sys_cls = "host_resident"
        elif sv < PLACEMENT_VRAM_HOST_FRAC * size:
            sys_cls = "not_in_vram_but_host_delta_unclear"
        else:
            sys_cls = "indeterminate"
    else:
        rec["sysfs_card"] = sa.get("pci_bdf") or sb.get("pci_bdf")
        rec["sysfs_unavailable_note"] = (
            "per-card amdgpu counters were not readable (is /sys/class/drm mounted in this "
            "container?); placement falls back to hipMemGetInfo + MemAvailable + physics")
    rec["sysfs_classification"] = sys_cls

    if vram_delta >= PLACEMENT_VRAM_DEVICE_FRAC * size:
        hip_cls = "device_resident"
    elif vram_delta < PLACEMENT_VRAM_HOST_FRAC * size and avail_delta >= PLACEMENT_HOST_SIGNAL_FRAC * size:
        hip_cls = "host_resident"
    elif vram_delta < PLACEMENT_VRAM_HOST_FRAC * size:
        hip_cls = "not_in_vram_but_host_delta_unclear"
    else:
        hip_cls = "indeterminate"
    rec["hip_memgetinfo_classification"] = hip_cls

    rec["classification"] = sys_cls if sys_cls is not None else hip_cls
    rec["classification_source"] = (
        "amdgpu sysfs (mem_info_vram_used / mem_info_gtt_used), per PCI BDF"
        if sys_cls is not None else
        "hipMemGetInfo free-VRAM + /proc/meminfo MemAvailable (sysfs unavailable)")
    rec["classifications_agree"] = (sys_cls is None or sys_cls == hip_cls)
    rec["not_in_vram"] = bool(vram_delta < PLACEMENT_VRAM_HOST_FRAC * size
                              and (not have or rec.get("sysfs_vram_used_delta_bytes", 0)
                                   < PLACEMENT_VRAM_HOST_FRAC * size))
    rec["positive_host_signal"] = bool(
        (have and rec.get("sysfs_gtt_used_delta_bytes", 0) >= PLACEMENT_HOST_SIGNAL_FRAC * size)
        or avail_delta >= PLACEMENT_HOST_SIGNAL_FRAC * size)
    return rec


def physics_verdict(arena_gbps: float | None, copy_gbps: float | None,
                    hbm_gbps: float | None, physical_card: int | None) -> dict:
    """Is the measured read rate PHYSICALLY consistent with pages on the far side of PCIe?

    This is the check whose absence invalidated the first P2: there, device-backed read was
    117.9 GB/s and "host"-backed read was 118.2 GB/s -- ratio 1.00 -- and every return code was
    hipSuccess. A 23.9x medium ratio is not a subtlety you can miss; it just has to be LOOKED at.
    """
    out: dict = {
        "arena_read_gbps": arena_gbps,
        "copy_engine_gbps": copy_gbps,
        "hbm_reference_gbps": hbm_gbps,
        "expected_pcie_gbps_for_card": PCIE_EXPECTED_GBPS.get(physical_card),
        "abs_pcie_ceiling_gbps": PCIE_ABS_CEILING_GBPS,
        "arena_over_copy_engine": None,
        "arena_frac_of_hbm": None,
        "gates": {},
    }
    if arena_gbps is None:
        out["consistent_with_host_pages"] = None
        out["reason"] = "no arena read bandwidth was measured"
        return out
    gates: dict = {}
    if copy_gbps:
        out["arena_over_copy_engine"] = round(arena_gbps / copy_gbps, 4)
        gates["not_faster_than_copy_engine"] = arena_gbps <= PROVENANCE_MAX_OVER_COPY * copy_gbps
    if hbm_gbps:
        out["arena_frac_of_hbm"] = round(arena_gbps / hbm_gbps, 4)
        gates["not_near_hbm"] = arena_gbps < PROVENANCE_MAX_FRAC_OF_HBM * hbm_gbps
    gates["under_abs_pcie_ceiling"] = arena_gbps <= PCIE_ABS_CEILING_GBPS * PCIE_CEILING_SLACK
    out["gates"] = {k: bool(v) for k, v in gates.items()}
    out["consistent_with_host_pages"] = all(gates.values()) if gates else None
    failed = [k for k, v in gates.items() if not v]
    out["reason"] = (
        "read rate is consistent with host pages behind PCIe"
        if not failed else
        f"read rate BREACHES {failed}: {arena_gbps:.2f} GB/s cannot have crossed PCIe -- these "
        f"pages are DEVICE memory whatever the counters and the location field said. This is the "
        f"exact failure that invalidated the first P2.")
    return out


# =================================================================================================
# CPU accessibility -- the sharpest single host/device discriminator, and it cannot segfault
# =================================================================================================
def cpu_read_probe(addr: int, nbytes: int, scratch_dir: str) -> tuple[dict, bytes | None]:
    """Copy `nbytes` out of `addr` THROUGH THE KERNEL, so a bad pointer yields EFAULT not SIGSEGV.

    The sink must be one the kernel genuinely copies FROM user space into. /dev/null is NOT: its
    write handler discards the payload and returns the count without touching the user pages, so it
    reports EVERY pointer as readable (measured on this box in P1 -- it returned 4096 for a
    guaranteed-unmapped VA). A regular file is used so the copy is real.

    The fake VMM "host" pages of Phase 0 failed this probe with EFAULT. Real hipHostMalloc pages
    must pass it.
    """
    buf = memoryview((ctypes.c_char * nbytes).from_address(addr))
    try:
        os.makedirs(scratch_dir, exist_ok=True)
    except OSError:
        scratch_dir = None
    try:
        with tempfile.TemporaryFile(dir=scratch_dir) as fh:
            done = 0
            while done < nbytes:
                n = os.write(fh.fileno(), buf[done:])
                if n <= 0:
                    return ({"readable": False, "errno": None,
                             "strerror": f"short write after {done} of {nbytes} B"}, None)
                done += n
            fh.seek(0)
            return ({"readable": True, "errno": None, "strerror": None}, fh.read(nbytes))
    except OSError as e:
        return ({"readable": False, "errno": e.errno, "strerror": e.strerror}, None)


def cpu_write_probe(addr: int, nbytes: int, scratch_dir: str) -> dict:
    """Probe CPU WRITABILITY without SIGSEGV: readv() returns EFAULT for a bad destination.

    DESTRUCTIVE -- the caller must own the range and expect it clobbered. Only ever called on a
    range that is about to be overwritten anyway, and only after cpu_read_probe said readable.
    """
    try:
        os.makedirs(scratch_dir, exist_ok=True)
    except OSError:
        scratch_dir = None
    payload = bytes((i * 7 + 13) & 0xFF for i in range(nbytes))
    try:
        with tempfile.TemporaryFile(dir=scratch_dir) as fh:
            fh.write(payload)
            fh.seek(0)
            dst = memoryview((ctypes.c_char * nbytes).from_address(addr))
            got = fh.readinto(dst)
            if got != nbytes:
                return {"writable": False, "errno": None,
                        "strerror": f"short readinto: {got} of {nbytes}"}
            same = bytes((ctypes.c_char * nbytes).from_address(addr)) == payload
            return {"writable": True, "errno": None, "strerror": None,
                    "readback_matches": bool(same)}
    except OSError as e:
        return {"writable": False, "errno": e.errno, "strerror": e.strerror}


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


# =================================================================================================
# the custom allocator: forward-only bump pointer inside OUR host arena, no-op free
# =================================================================================================
def _install_callbacks():
    """Build and permanently anchor the two CFUNCTYPE trampolines. Returns (alloc_cb, free_cb).

    Both trampolines, and the python closures they wrap, go into `_KEEP` forever. If either were
    collected, torch would call a dangling function pointer -- a segfault with no diagnostic, in a
    probe whose entire job is to produce a diagnostic.
    """
    ALLOC_T = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p)
    FREE_T = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p)

    def _alloc(size, device, stream):  # noqa: ANN001 - C ABI
        try:
            size = int(size)
            base = int(_ARENA["base"])
            cur = _round_up(int(_ARENA["cursor"]), 512)
            # The arena is one hipHostMalloc mapping, registered for the device that created it.
            # Serving it to a different device index would hand torch memory the asking device may
            # not address -- a silent-corruption path, not a measurement. Fall back instead.
            arena_dev = _ARENA.get("device")
            right_device = arena_dev is None or int(device) == int(arena_dev)
            if base and right_device and cur + size <= int(_ARENA["size"]):
                ptr = base + cur
                _ARENA["cursor"] = cur + size
                _ALLOC_EVENTS.append({"size": size, "device": int(device),
                                      "stream": int(stream or 0), "ptr": ptr, "offset": cur,
                                      "from_arena": True})
                return ptr
            # NEVER return NULL: a NULL here becomes a use-after-null deep inside torch. Serve it
            # from a real hipMalloc, record that it did NOT come from the arena, and let the arm
            # report a clean FAIL.
            p = ctypes.c_void_p()
            rc = _HIP.lib.hipMalloc(ctypes.byref(p), ctypes.c_size_t(size)) if _HIP else -1
            _ARENA["fallbacks"] = int(_ARENA["fallbacks"]) + 1
            _ALLOC_EVENTS.append({
                "size": size, "device": int(device), "stream": int(stream or 0),
                "ptr": int(p.value or 0), "offset": None, "from_arena": False,
                "hipMalloc_rc": int(rc),
                "reason": ("no arena" if not base else
                           "wrong device" if not right_device else "arena exhausted"),
            })
            return int(p.value or 0)
        except BaseException as exc:  # a raise inside a ctypes callback silently returns NULL
            _ALLOC_EVENTS.append({"error": f"{type(exc).__name__}: {exc}", "from_arena": False})
            return 0

    def _free(ptr, size, device, stream):  # noqa: ANN001 - C ABI
        # Deliberate no-op. The arena belongs to this process for its whole life and hipHostFree is
        # NEVER called: freeing it while a torch tensor still points into it is precisely the
        # c10::Error "invalid device pointer" abort this probe exists to avoid. The hipMalloc
        # fallbacks are intentionally leaked (short-lived probe) rather than risk a double free.
        try:
            _FREE_EVENTS.append({"ptr": int(ptr or 0), "size": int(size), "device": int(device),
                                 "stream": int(stream or 0), "t": time.time()})
        except BaseException:
            pass

    a, f = ALLOC_T(_alloc), FREE_T(_free)
    _KEEP.extend([ALLOC_T, FREE_T, _alloc, _free, a, f])
    return a, f


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
# box state -- a number from an unrecorded box is worthless; this box is never idle
# =================================================================================================
def _run(cmd: list[str], timeout: int = 20) -> dict:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return {"rc": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()[:2000]}
    except FileNotFoundError:
        return {"rc": None, "error": "not found"}
    except Exception as exc:
        return {"rc": None, "error": f"{type(exc).__name__}: {exc}"}


def box_state(with_smi: bool = True) -> dict:
    st: dict = {"t_wall": time.time(), "t_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    st["hostname"] = platform.node()
    st["kernel"] = platform.release()
    st["in_container"] = os.path.exists("/.dockerenv")
    st["uid"] = os.getuid()

    meminfo = _read_text("/proc/meminfo") or ""
    mem: dict = {}
    for key in ("MemTotal", "MemFree", "MemAvailable", "Cached", "SwapTotal", "SwapFree", "Shmem"):
        m = re.search(rf"^{key}:\s+(\d+) kB", meminfo, re.M)
        mem[key + "_kB"] = int(m.group(1)) if m else None
    st["meminfo"] = mem

    vmstat = _read_text("/proc/vmstat") or ""
    vm: dict = {}
    for key in ("pswpin", "pswpout", "pgmajfault", "nr_free_pages"):
        m = re.search(rf"^{key} (\d+)", vmstat, re.M)
        vm[key] = int(m.group(1)) if m else None
    st["vmstat"] = vm

    st["loadavg"] = (_read_text("/proc/loadavg") or "").strip() or None
    if with_smi:
        st["rocm_smi"] = _run(["rocm-smi", "--showmeminfo", "vram", "--showuse", "--csv"],
                              timeout=40)
    st["amdgpu_sysfs"] = sysfs_amdgpu_cards()
    st["env_visibility"] = {
        "ROCR_VISIBLE_DEVICES": os.environ.get("ROCR_VISIBLE_DEVICES"),
        "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES"),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        "PYTORCH_HIP_ALLOC_CONF": os.environ.get("PYTORCH_HIP_ALLOC_CONF"),
    }
    st["provenance"] = {
        "image": os.environ.get("P5B_IMAGE"),
        "image_id": os.environ.get("P5B_IMAGE_ID"),
        "git_sha": os.environ.get("P5B_GIT_SHA"),
        "git_dirty_files": os.environ.get("P5B_GIT_DIRTY"),
        "script": os.path.abspath(__file__),
        "schema": SCHEMA_VERSION,
    }
    st["image_markers"] = {
        "dockerenv": os.path.exists("/.dockerenv"),
        "opt_kernels": os.path.isdir("/opt/kernels"),
        "opt_minisgl_python": os.path.isdir("/opt/minisgl/python"),
        "os_release": (_read_text("/etc/os-release") or "").strip()[:400] or None,
    }
    return st


def _stats(samples: list[float]) -> dict:
    """median + spread. Returns Nones -- never a fabricated number -- if there is nothing to
    reduce. The spec requires median and spread over >= 5 reps with a warm-up discarded."""
    if not samples:
        return {"n": 0, "median": None, "mean": None, "stdev": None, "min": None, "max": None,
                "p90": None, "samples": []}
    s = sorted(samples)
    return {
        "n": len(s), "median": statistics.median(s), "mean": statistics.fmean(s),
        "stdev": (statistics.pstdev(s) if len(s) < 2 else statistics.stdev(s)),
        "min": s[0], "max": s[-1],
        "p90": s[min(len(s) - 1, int(round(0.9 * (len(s) - 1))))],
        "samples": samples,
    }


# =================================================================================================
# host-side verification -- the ONLY trustworthy check on this box
# =================================================================================================
class HostVerify:
    """Verify device-visible bytes on the CPU, with NO device allocation anywhere in the check.

    Why (measured in P5, p5_diagnostics/): the obvious check,

        torch.equal(t, torch.full_like(t, v))

    allocates its comparison operand -- and torch.equal's own reduction output -- from torch's
    DEFAULT caching allocator. Under expandable_segments:True (the compose default) that allocator
    unmaps physical handles on empty_cache() and re-maps at the SAME VA on the next allocation, and
    this driver serves a STALE page for a remap at an already-used VA (P6). The freshly-filled
    comparison tensor then reads back as zeros and the check blames the innocent tensor under test.
    It fired on the arm wired to the KILL criterion.

    So: one host buffer allocated once, one blocking hipMemcpy D2H, one CPU compare of the FULL
    byte range. Because P5b's arena is genuinely host memory, a SECOND, independent view is
    available -- dereferencing the hipHostMalloc host pointer directly -- and it is recorded as a
    cross-check. The two disagreeing would itself be a finding (a coherence defect), so it is
    reported rather than silently preferred.
    """

    def __init__(self, hip: "Hip", nbytes: int) -> None:
        import torch

        self._torch = torch
        self.nbytes = int(nbytes)
        self.hip = hip
        self.buf = (ctypes.c_ubyte * self.nbytes)()
        self.addr = ctypes.addressof(self.buf)
        self.view = torch.frombuffer(memoryview(self.buf), dtype=torch.uint8)
        self.memcpy_failures = 0

    def d2h(self, dptr: int, nbytes: int) -> None:
        if nbytes > self.nbytes:
            raise ValueError(f"HostVerify buffer is {self.nbytes} B, asked for {nbytes} B")
        self.hip.check(self.hip.lib.hipMemcpy(ctypes.c_void_p(self.addr), ctypes.c_void_p(dptr),
                                              ctypes.c_size_t(nbytes), hipMemcpyDeviceToHost),
                       "hipMemcpy D2H (HostVerify)")

    def equals_fill(self, dptr: int, n_elem: int, dtype, value, host_addr: int | None = None):
        """Is [dptr, dptr + n_elem*itemsize) exactly `value` repeated, in `dtype`?"""
        torch = self._torch
        expect = torch.empty(n_elem, dtype=dtype).fill_(value)  # CPU tensor, no device alloc
        expect_b = expect.view(torch.uint8) if expect.element_size() > 1 else expect
        expect_b = expect_b.reshape(-1)
        nb = int(expect.numel() * expect.element_size())
        self.d2h(dptr, nb)
        got = self.view[:nb]
        ok = bool(torch.equal(got, expect_b))
        det = {"n_elem": n_elem, "bytes_compared": nb, "value": value, "ok": ok,
               "verified": "host (ctypes hipMemcpy D2H + full-range CPU compare)"}
        if host_addr:
            # Independent second view: the CPU pointer hipHostMalloc handed back. Only meaningful
            # for genuinely host-resident pages, which is the claim under test. Recorded as a
            # cross-check -- it never overrides the hipMemcpy answer, because a disagreement is a
            # COHERENCE finding and must be visible as one rather than silently resolved.
            try:
                direct = bytes((ctypes.c_char * nb).from_address(host_addr))
                det["direct_cpu_view_ok"] = bool(direct == expect_b.contiguous().numpy().tobytes())
                det["direct_cpu_view_agrees_with_memcpy"] = bool(det["direct_cpu_view_ok"] == ok)
            except Exception as exc:
                det["direct_cpu_view_error"] = f"{type(exc).__name__}: {exc}"
                det["direct_cpu_view_agrees_with_memcpy"] = None
        if not ok:
            neq = (got != expect_b)
            nz = neq.nonzero()
            idx = int(nz[0].item()) if nz.numel() else -1
            det.update({"n_mismatched_bytes": int(neq.sum().item()),
                        "first_bad_byte_index": idx,
                        "first_bad_byte": (int(got[idx].item()) if idx >= 0 else None),
                        "expected_byte": (int(expect_b[idx].item()) if idx >= 0 else None)})
        return det

    def sha256(self, dptr: int, nbytes: int) -> str:
        """Byte-exact fingerprint of a device-visible range, computed on the CPU.

        This is what makes "bit-identical over >= 20 replays" a real claim rather than a
        value-tolerance claim: two replays are identical iff their output bytes hash the same.
        """
        self.d2h(dptr, nbytes)
        return hashlib.sha256(bytes(self.buf[:nbytes])).hexdigest()


# =================================================================================================
# result skeleton + schema validation
# =================================================================================================
ARM_NAMES = (
    "arena",             # hipHostMalloc(Mapped) + hipHostGetDevicePointer, bytes verified
    "placement",         # VRAM delta + MemAvailable delta + CPU accessibility (+ physics, folded
                         # in after `bandwidth` -- this arm is finalised LAST, see finalize_*)
    "pool_alloc",        # t.data_ptr() == the host-derived device pointer
    "bandwidth",         # >=256 MiB checksummed kernel read vs HBM and copy-engine references
    "correctness",       # shader read + shader write + exact roundtrip through the arena
    "empty_cache_live",  # engine/graph.py:314, for real
    "empty_cache_cached",
    "graph_capture",     # >= 20 replays, bit-identical
)
# Arms whose failure makes the LEG fail. `empty_cache_cached` is observational.
REQUIRED_ARMS = ("arena", "placement", "pool_alloc", "bandwidth", "correctness",
                 "empty_cache_live", "graph_capture")


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
            "device_index": cfg["device"],
            "mechanism": "hipHostMalloc(Mapped|Portable) + hipHostGetDevicePointer",
            "host_flags": f"0x{args.host_flags:x}",
            "arena_bytes": args.arena_bytes,
            "bw_bytes": args.bw_bytes,
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


VERDICT_TRISTATE_KEYS = (
    "p5b_pass",
    "host_arena_placement_proven",
    "foreign_ptr_ok",
    "expandable_segments_coexists",
    "capture_ok",
    "capture_bit_identical",
    "tensor_survives_empty_cache",
    "empty_cache_invokes_free_cb_live_block",
    "empty_cache_invokes_free_cb_cached_block",
    "hipMalloc_fallbacks_seen",
    "both_cards_exercised",
)


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
    for k in VERDICT_TRISTATE_KEYS:
        if k not in v:
            raise ValueError(f"missing verdict.{k}")
        if not isinstance(v[k], (bool, type(None))):
            raise ValueError(f"verdict.{k} must be bool or null, got {type(v[k]).__name__}")
    need(v, "reasons", list, "verdict")
    need(v, "per_leg_claims", dict, "verdict")
    need(v, "bandwidth_gbps_per_leg", dict, "verdict")
    for name, leg in obj["legs"].items():
        need(leg, "status", str, f"legs.{name}")
        need(leg, "arms", dict, f"legs.{name}")
        for arm_name, arm in leg["arms"].items():
            need(arm, "status", str, f"legs.{name}.arms.{arm_name}")
            if not isinstance(arm.get("passed"), (bool, type(None))):
                raise ValueError(f"legs.{name}.arms.{arm_name}.passed must be bool or null")


# =================================================================================================
# timing helpers
# =================================================================================================
def time_device_op(torch, dev: int, fn, nbytes: int, reps: int, warmup: int) -> dict:
    """Time a device op both ways and REFUSE to trust the pair if they disagree.

    hipEventElapsedTime returning garbage is indistinguishable from a spectacular result if you
    only look at the event number -- the checksum still passes. Wall time strictly exceeds device
    time (it carries launch + sync + ctypes overhead), so event GB/s may sit a little above wall
    GB/s but never by TIMING_SANITY_FACTOR. A config that breaches it is excluded, loudly.
    """
    for _ in range(max(warmup, 1)):
        fn()
    torch.cuda.synchronize(dev)
    ev_ms: list[float] = []
    wall_ms: list[float] = []
    # NOTE: `fn` must NOT synchronise (no .item()). A blocking read inside the timed window folds
    # the sync into the measurement and makes the event/wall cross-check vacuous -- the two would
    # agree by construction and could no longer catch a broken hipEventElapsedTime. The checksum
    # is taken ONCE, outside this loop, by the caller.
    for _ in range(reps):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        t0 = time.perf_counter_ns()
        e0.record()
        fn()
        e1.record()
        torch.cuda.synchronize(dev)
        t1 = time.perf_counter_ns()
        ev_ms.append(float(e0.elapsed_time(e1)))
        wall_ms.append((t1 - t0) / 1e6)

    def gbps(ms_list):
        vals = [nbytes / (ms / 1e3) / 1e9 for ms in ms_list if ms and ms > 0]
        return _stats(vals)

    ev_g, wall_g = gbps(ev_ms), gbps(wall_ms)
    sane = True
    why = None
    if not ev_g["median"] or ev_g["median"] <= 0:
        sane, why = False, "hipEvent elapsed time was non-positive"
    elif wall_g["median"] and ev_g["median"] > TIMING_SANITY_FACTOR * wall_g["median"]:
        sane = False
        why = (f"event-derived {ev_g['median']:.1f} GB/s > {TIMING_SANITY_FACTOR}x wall-derived "
               f"{wall_g['median']:.1f} GB/s -- the TIMER produced this number, not the kernel")
    return {
        "bytes": nbytes, "reps": reps, "warmup_discarded": max(warmup, 1),
        "event_ms": _stats(ev_ms), "wall_ms": _stats(wall_ms),
        "gbps_event": ev_g, "gbps_wall": wall_g,
        "gbps": ev_g["median"] if sane else None,
        "timing_sane": sane, "timing_note": why,
    }


def time_copy_engine(hip: "Hip", dst: int, src_host: int, nbytes: int, reps: int,
                     warmup: int) -> dict:
    """Blocking hipMemcpy H2D from the arena's HOST pointer, wall-timed.

    This is the same measurement that produced P1's 28.70 / 14.34 GB/s per-card figures, taken
    in-process on the card actually under test so the physics gate has a LIVE denominator instead
    of a remembered one. At >= 256 MiB the copy is ~9 ms, so wall timing is not a limitation.
    """
    for _ in range(max(warmup, 1)):
        hip.memcpy(dst, src_host, nbytes, hipMemcpyHostToDevice)
    hip.sync()
    ms: list[float] = []
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        hip.memcpy(dst, src_host, nbytes, hipMemcpyHostToDevice)
        hip.sync()
        ms.append((time.perf_counter_ns() - t0) / 1e6)
    g = _stats([nbytes / (m / 1e3) / 1e9 for m in ms if m > 0])
    return {"bytes": nbytes, "reps": reps, "warmup_discarded": max(warmup, 1),
            "wall_ms": _stats(ms), "gbps_wall": g, "gbps": g["median"],
            "method": "blocking hipMemcpy(HostToDevice) from the arena host pointer"}


# =================================================================================================
# CHILD: the measurement
# =================================================================================================
def run_leg(leg: str, cfg: dict, args) -> dict:  # noqa: C901 - a probe is a straight line of arms
    res = leg_skeleton(leg, cfg, args)
    res["box_state"] = box_state()
    dev = int(cfg["device"])

    # ---- preconditions ---------------------------------------------------------------------------
    if not args.allow_outside_image and not (os.path.exists("/.dockerenv")
                                             or os.path.isdir("/opt/kernels")):
        res["status"] = "precondition_failed"
        res["precondition_error"] = (
            "not running inside the serve image (no /.dockerenv and no /opt/kernels). P5b must be "
            "confirmed IN-IMAGE -- that is half the question. Pass --allow-outside-image only for "
            "host-side debugging, and never quote the result.")
        return res

    avail = (res["box_state"].get("meminfo", {}).get("MemAvailable_kB") or 0) * 1024
    need = int(args.arena_bytes * 1.25) + (2 << 30)
    if avail and avail < need:
        res["status"] = "precondition_failed"
        res["precondition_error"] = (
            f"MemAvailable {avail/2**30:.1f} GiB < {need/2**30:.1f} GiB needed for a "
            f"{args.arena_bytes/2**20:.0f} MiB pinned arena plus headroom. Pinning into a tight "
            f"box swaps (P3b measured 114,813 pages out at the 62 GiB ceiling) and every latency "
            f"number below would be a swap measurement.")
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
        res["precondition_error"] = (
            f"torch lacks required API(s): {missing}. Without them the ~30-line ctypes arena is "
            f"impossible by construction and M1-A needs the C++ from_blob extension (+3 days).")
        return res
    if not torch.cuda.is_available():
        res["status"] = "precondition_failed"
        res["precondition_error"] = (
            "torch.cuda.is_available() is False -- ROCm device passthrough missing "
            "(--device /dev/kfd --device /dev/dri --group-add video)")
        return res

    global _HIP
    try:
        _HIP = Hip()
        _KEEP.append(_HIP)
    except Exception as exc:
        res["status"] = "precondition_failed"
        res["precondition_error"] = f"libamdhip64 bind failed: {type(exc).__name__}: {exc}"
        return res

    if dev >= torch.cuda.device_count():
        res["status"] = "precondition_failed"
        res["precondition_error"] = (
            f"leg {leg} wants torch device {dev} but torch sees only "
            f"{torch.cuda.device_count()} device(s). The BOTH-CARDS requirement cannot be met on "
            f"this enumeration -- check ROCR_VISIBLE_DEVICES.")
        return res

    torch.cuda.set_device(dev)
    _KEEP.append(torch.zeros(1024, device=f"cuda:{dev}"))  # force + hold the primary context
    torch.cuda.synchronize(dev)

    props = torch.cuda.get_device_properties(dev)
    bdf = _HIP.pci_bus_id(dev)
    rocr = [int(x) for x in (os.environ.get("ROCR_VISIBLE_DEVICES") or "").split(",") if x != ""]
    res["device"] = {
        "torch_index": dev,
        "physical_card_index": rocr[dev] if dev < len(rocr) else None,
        "rocr_visible_devices": os.environ.get("ROCR_VISIBLE_DEVICES"),
        "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
        "name": props.name,
        "gcn_arch": getattr(props, "gcnArchName", None),
        "total_memory_bytes": int(props.total_memory),
        # NOTE: torch reports WGPs, not CUs, on RDNA -- a 64-CU card answers 32. Recorded as-is.
        "multi_processor_count_wgps": int(getattr(props, "multi_processor_count", 0)),
        "pci_bus_id": bdf,
        "hip_device_count": _HIP.device_count(),
        "all_visible": [{"index": i, "name": torch.cuda.get_device_properties(i).name,
                         "pci_bus_id": _HIP.pci_bus_id(i)}
                        for i in range(torch.cuda.device_count())],
        "hip_mem_info_at_start": _HIP.mem_info(),
        "hip_current_device": _HIP.set_device(dev),
        "amdgpu_sysfs_for_this_card": card_mem(bdf),
    }
    if res["device"]["hip_current_device"] != dev:
        res["status"] = "precondition_failed"
        res["precondition_error"] = (
            f"hipSetDevice({dev}) left the current device at {res['device']['hip_current_device']}"
            "; our ctypes calls and torch would target different cards and every number below "
            "would be attributed to the wrong GPU")
        return res
    if not card_mem(bdf):
        print(f"[P5b] WARNING: no amdgpu sysfs counters for PCI {bdf!r} -- placement falls back to "
              "hipMemGetInfo + MemAvailable + physics (is /sys/class/drm visible in this "
              "container?)", file=sys.stderr, flush=True)

    # ---- is expandable_segments actually LIVE? (assert on behaviour, not on the env var) ---------
    alloc_backend = None
    try:
        alloc_backend = torch.cuda.get_allocator_backend()
    except Exception:
        pass
    expandable_effective = None
    snapshot_probe: dict = {}
    try:
        _KEEP.append(torch.empty(64 << 20, dtype=torch.uint8, device=f"cuda:{dev}"))
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
    # Tri-state, and the verdict honours it: a leg that could not confirm the setting is live may
    # not be quoted as evidence of coexistence. Claiming it off an unread env var is the
    # confident-wrong-number failure mode.
    res["torch"]["expandable_segments_verified"] = (
        None if cfg["alloc_conf"] != EXPANDABLE
        else (True if expandable_effective is True
              else (False if expandable_effective is False else None)))
    if cfg["alloc_conf"] == EXPANDABLE and expandable_effective is False:
        res["status"] = "precondition_failed"
        res["precondition_error"] = (
            "leg requires expandable_segments:True (the compose default) but memory_snapshot() "
            f"reports no expandable segment (env={res['torch']['expandable_segments_env']}, "
            f"backend={alloc_backend}). Refusing to report a coexistence result that was never "
            "actually exercised.")
        return res
    if cfg["alloc_conf"] == EXPANDABLE and expandable_effective is None:
        print("[P5b] WARNING: could not verify expandable_segments is live "
              f"(snapshot_probe={snapshot_probe}); coexistence will be reported NOT VERIFIED",
              file=sys.stderr, flush=True)

    # ---- ARM: arena -- THE SELECTED MECHANISM ---------------------------------------------------
    arm = res["arms"]["arena"] = new_arm(
        "hipHostMalloc(Mapped|Portable) + hipHostGetDevicePointer: does it produce a device "
        "pointer whose pages actually STORE what is written, and are they CPU-accessible?")
    place_before = placement_sample(_HIP, bdf)
    try:
        size = _round_up(int(args.arena_bytes), 1 << 21)
        host_ptr, dev_ptr = _HIP.host_alloc(size, args.host_flags)
        # anchored forever; hipHostFree is NEVER called (see the free callback's comment)
        _ARENA.update({"base": dev_ptr, "host": host_ptr, "size": size, "cursor": 0,
                       "fallbacks": 0, "device": dev})
        _KEEP.append(("arena", host_ptr, dev_ptr, size))
        place_after_alloc = placement_sample(_HIP, bdf)

        # Touch EVERY page through the DEVICE pointer before re-sampling: drivers commit lazily,
        # so a delta taken at allocation time can be ~0 for both media and prove nothing.
        _HIP.memset(dev_ptr, 0, size)
        _HIP.sync()
        place_after_touch = placement_sample(_HIP, bdf)

        vb = _HIP.validate_backing(dev_ptr, size)
        arm["detail"] = {
            "host_ptr": f"0x{host_ptr:x}",
            "device_ptr": f"0x{dev_ptr:x}",
            "same_va": host_ptr == dev_ptr,
            "arena_bytes": size,
            "flags": f"0x{args.host_flags:x}",
            "host_ptr_page_aligned": host_ptr % 4096 == 0,
            "maps_entry_for_host_ptr": maps_entry_for(host_ptr),
            "pointer_attrs": _HIP.pointer_attrs(dev_ptr),
            "backing_data_validated": vb["validated"],
            "backing_validation": vb,
            "note": ("two distinct patterns with per-4KiB position markers, written and read back "
                     "through the DEVICE pointer at head/middle/tail. Nothing here passes on a "
                     "return code -- on this box a memory call can return hipSuccess and still "
                     "serve a stale physical page (P6)."),
        }
        arm["passed"] = bool(vb["validated"])
        arm["status"] = "ok" if arm["passed"] else "failed"
        if not arm["passed"]:
            arm["error"] = ("the arena was allocated with hipSuccess but does NOT store what is "
                            "written to it")
            res["status"] = "precondition_failed"
            res["precondition_error"] = (
                f"the SELECTED mechanism is not usable memory on card "
                f"{res['device']['physical_card_index']}: {arm['error']}. This is a MECHANISM "
                f"failure, not a torch-plumbing answer.")
            res["allocator_events"] = _events_summary()
            return res
    except Exception as exc:
        arm["status"] = "failed"
        arm["passed"] = False
        arm["error"] = f"{type(exc).__name__}: {exc}"
        arm["detail"]["traceback"] = traceback.format_exc()
        res["status"] = "precondition_failed"
        res["precondition_error"] = (
            f"could not build the hipHostMalloc(Mapped) arena: {arm['error']} "
            "(MECHANISM failure, NOT a torch-plumbing answer)")
        res["allocator_events"] = _events_summary()
        return res

    arena_base = int(_ARENA["base"])
    arena_host = int(_ARENA["host"])
    arena_size = int(_ARENA["size"])

    # ---- ARM: placement -- deltas + CPU accessibility (physics folded in after `bandwidth`) ------
    arm = res["arms"]["placement"] = new_arm(
        "are these pages REALLY host RAM? VRAM delta, MemAvailable delta, per-card amdgpu GTT "
        "delta, CPU accessibility -- and (folded in after the bandwidth arm) achieved read rate "
        "against the 28.7 GB/s PCIe ceiling. NEVER a location field, NEVER a return code.")
    try:
        at_alloc = classify_placement(arena_size, place_before, place_after_alloc)
        at_touch = classify_placement(arena_size, place_before, place_after_touch)
        cpu_r, _ = cpu_read_probe(arena_host, 4096, args.scratch_dir)
        cpu_w = {"writable": None, "skipped": "cpu_read_probe said not readable"}
        if cpu_r.get("readable"):
            # destructive, on a range about to be overwritten by the pool anyway
            cpu_w = cpu_write_probe(arena_host, 4096, args.scratch_dir)
            _HIP.memset(arena_base, 0, 1 << 20)
            _HIP.sync()
        arm["detail"] = {
            "sample_before_alloc": place_before,
            "sample_after_alloc": place_after_alloc,
            "sample_after_full_touch": place_after_touch,
            "classification_at_alloc": at_alloc,
            "classification_after_touch": at_touch,
            "cpu_readable": cpu_r,
            "cpu_writable": cpu_w,
            "cpu_accessibility_note": (
                "probed through a syscall against a REGULAR FILE so a bad pointer returns EFAULT "
                "instead of SIGSEGV. /dev/null is NOT a valid sink -- it reports every pointer as "
                "readable. The fake VMM 'host' pages of Phase 0 failed this probe."),
            "physics": None,
            "physics_note": "filled in by the bandwidth arm; this arm is finalised last",
        }
        # provisional -- finalised in finalize_placement() once bandwidth exists
        arm["status"] = "pending_physics"
    except Exception as exc:
        arm["status"] = "failed"
        arm["passed"] = False
        arm["error"] = f"{type(exc).__name__}: {exc}"
        arm["detail"]["traceback"] = traceback.format_exc()

    # ---- build the custom allocator + the MemPool (both anchored forever) ------------------------
    try:
        alloc_cb, free_cb = _install_callbacks()
        allocator = torch._C._cuda_customAllocator(
            ctypes.cast(alloc_cb, ctypes.c_void_p).value,
            ctypes.cast(free_cb, ctypes.c_void_p).value)
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
    bw_elem = args.bw_bytes // itemsize
    slot_elem = args.slot_bytes // itemsize
    if bw_elem == 0 or slot_elem == 0:
        res["status"] = "precondition_failed"
        res["precondition_error"] = (
            f"--bw-bytes/--slot-bytes are smaller than one {args.dtype} element")
        return res
    devstr = f"cuda:{dev}"

    # ---- ARM: pool_alloc -- the core claim ------------------------------------------------------
    # The FIRST allocation is the bandwidth tensor, so it sits at offset 0 and its data_ptr() is
    # the claim: t.data_ptr() == the host-derived device pointer.
    arm = res["arms"]["pool_alloc"] = new_arm(
        "does torch.empty() under use_mem_pool() land on the hipHostGetDevicePointer address, and "
        "what does an arena allocation cost?")
    bw_t = None
    slot_tensors: list = []
    try:
        recs: list[dict] = []
        lat_all: list[float] = []
        lat_cb: list[float] = []      # reps that actually reached OUR allocator
        lat_cached: list[float] = []  # reps torch served from its own cache (NOT an arena cost)
        plan = [("bw", bw_elem)] + [("slot", slot_elem)] * (args.warmup + args.reps)
        for i, (kind, n) in enumerate(plan):
            n_before = len(_ALLOC_EVENTS)
            t0 = time.perf_counter_ns()
            with torch.cuda.use_mem_pool(pool):
                t = torch.empty(n, dtype=dtype, device=devstr)
            t1 = time.perf_counter_ns()
            _KEEP.append(t)  # a pool tensor must NEVER outlive the pool -- that is the SIGABRT
            if kind == "bw":
                bw_t = t
            else:
                slot_tensors.append(t)
            new_events = _ALLOC_EVENTS[n_before:]
            ptr = t.data_ptr()
            rec = {
                "index": i, "kind": kind, "elements": n,
                "warmup": kind == "slot" and (i - 1) < args.warmup,
                "data_ptr": ptr,
                "in_arena": arena_base <= ptr < arena_base + arena_size,
                "offset_from_base": ptr - arena_base,
                "alloc_callbacks": new_events,
                "hit_custom_allocator": bool(new_events),
                "latency_us": (t1 - t0) / 1e3,
            }
            recs.append(rec)
            if kind == "slot" and not rec["warmup"]:
                lat_all.append(rec["latency_us"])
                (lat_cb if new_events else lat_cached).append(rec["latency_us"])

        first = recs[0]
        eq_base = first["data_ptr"] == arena_base
        all_in = all(r["in_arena"] for r in recs)
        n_cb = sum(len(r["alloc_callbacks"]) for r in recs)
        served = all(e.get("from_arena") for r in recs for e in r["alloc_callbacks"])
        arm["detail"] = {
            "host_derived_device_ptr": arena_base,
            "host_ptr": arena_host,
            "first_data_ptr": first["data_ptr"],
            "first_data_ptr_equals_host_derived_device_ptr": eq_base,
            "all_data_ptrs_inside_arena": all_in,
            "n_alloc_callbacks": n_cb,
            "every_alloc_served_from_arena": served,
            "hipMalloc_fallbacks": int(_ARENA["fallbacks"]),
            "allocations": recs,
            # THREE stats, because the aggregate alone is a mislabelled number: torch's caching
            # allocator can serve a rep from a previously-split block WITHOUT calling us, and that
            # rep costs a dict lookup. Quoting the mixed median as "the cost of an arena
            # allocation" would be wrong by an order of magnitude.
            "alloc_latency_us": _stats(lat_all),
            "alloc_latency_us_via_custom_allocator": _stats(lat_cb),
            "alloc_latency_us_served_from_torch_cache": _stats(lat_cached),
            "latency_note": ("host wall around torch.empty() only. An allocation launches no "
                             "device work, so no event fence applies. Quote "
                             "alloc_latency_us_via_custom_allocator."),
        }
        arm["passed"] = bool(eq_base and all_in and served and n_cb > 0
                             and int(_ARENA["fallbacks"]) == 0)
        arm["status"] = "ok" if arm["passed"] else "failed"
        if not arm["passed"]:
            arm["error"] = (
                "torch did NOT allocate over the host-derived device pointer "
                f"(first data_ptr=0x{first['data_ptr']:x} vs arena base=0x{arena_base:x}, "
                f"fallbacks={_ARENA['fallbacks']}, every_alloc_served_from_arena={served})")
    except Exception as exc:
        arm["status"] = "failed"
        arm["passed"] = False
        arm["error"] = f"{type(exc).__name__}: {exc}"
        arm["detail"]["traceback"] = traceback.format_exc()

    # ---- ARM: bandwidth -- the placement physics, and the only number worth quoting --------------
    arm = res["arms"]["bandwidth"] = new_arm(
        "how fast does a KERNEL read the arena over >= 256 MiB, against an in-process HBM "
        "reference and an in-process copy-engine reference on THIS card? A read that outruns "
        "PCIe did not cross PCIe.")
    physics = None
    if bw_t is None:
        arm["status"] = "skipped"
        arm["error"] = "no arena tensor (pool_alloc produced none)"
    elif args.bw_bytes < (64 << 20):
        arm["status"] = "skipped"
        arm["error"] = (f"--bw-bytes {args.bw_bytes} is below the 64 MB MALL; any number would be "
                        f"a cache artifact (this is how the withdrawn 127.2 GB/s happened)")
    else:
        try:
            nbytes = bw_elem * itemsize
            # Fill with 1.0 so the reduction doubles as a CHECKSUM: a float64 accumulation of
            # bw_elem ones is exactly bw_elem. A bandwidth number can then never be printed for a
            # read that did not actually land on every byte of the pages under test.
            bw_t.fill_(1.0)
            torch.cuda.synchronize(dev)
            arena_sum = float(bw_t.sum(dtype=torch.float64).item())
            arena_checksum_ok = abs(arena_sum - float(bw_elem)) < 0.5
            arena_bw = time_device_op(torch, dev, lambda: bw_t.sum(dtype=torch.float64),
                                      nbytes, args.reps, args.warmup)

            # HBM reference: the SAME op, the SAME size, ordinary device memory, this card, this
            # process. A remembered 692 GB/s is not a denominator.
            hbm_t = torch.empty(bw_elem, dtype=dtype, device=devstr)
            _KEEP.append(hbm_t)
            hbm_t.fill_(1.0)
            torch.cuda.synchronize(dev)
            hbm_sum = float(hbm_t.sum(dtype=torch.float64).item())
            hbm_checksum_ok = abs(hbm_sum - float(bw_elem)) < 0.5
            hbm_bw = time_device_op(torch, dev, lambda: hbm_t.sum(dtype=torch.float64),
                                    nbytes, args.reps, args.warmup)

            # Copy-engine reference: pinned H2D out of the arena's HOST pointer -- P1's method.
            dst = _HIP.dev_alloc(nbytes)
            _KEEP.append(("copy_dst", dst))
            copy_bw = time_copy_engine(_HIP, dst, arena_host, nbytes, args.reps, args.warmup)

            physics = physics_verdict(arena_bw["gbps"], copy_bw["gbps"], hbm_bw["gbps"],
                                      res["device"]["physical_card_index"])
            arm["detail"] = {
                "working_set_bytes": nbytes,
                "op": "Tensor.sum(dtype=float64) -- a full-buffer streaming read",
                "arena_read": arena_bw,
                "arena_read_checksum_ok": arena_checksum_ok,
                "arena_read_checksum": {"observed": arena_sum, "expected": float(bw_elem),
                                        "why": ("the buffer is filled with 1.0 and reduced in "
                                                "float64, so the sum is EXACTLY the element count "
                                                "iff every byte was read. A bandwidth number can "
                                                "never be printed for a read that missed the "
                                                "pages under test.")},
                "hbm_reference": hbm_bw,
                "hbm_reference_checksum_ok": hbm_checksum_ok,
                "hbm_reference_checksum": {"observed": hbm_sum, "expected": float(bw_elem)},
                "copy_engine_reference": copy_bw,
                "physics": physics,
                "expected_for_card": PCIE_EXPECTED_GBPS.get(
                    res["device"]["physical_card_index"]),
                "note": ("the arena figure is a TORCH-LEVEL kernel read of the real T1 medium on "
                         "the card named in device.physical_card_index. It is not comparable to "
                         "P1's hand-written tiled kernel and must not be quoted as the T1 "
                         "ceiling; P1 owns that number. It is quoted here to PROVE PLACEMENT."),
            }
            arm["passed"] = bool(arena_checksum_ok and hbm_checksum_ok
                                 and arena_bw["timing_sane"] and hbm_bw["timing_sane"]
                                 and physics.get("consistent_with_host_pages") is True)
            arm["status"] = "ok" if arm["passed"] else "failed"
            if not arm["passed"]:
                bits = []
                if not arena_checksum_ok:
                    bits.append("the arena read returned a WRONG checksum -- it did not land on "
                                "the pages under test")
                if not hbm_checksum_ok:
                    bits.append("the HBM reference read returned a wrong checksum")
                if not arena_bw["timing_sane"]:
                    bits.append(f"arena timing insane: {arena_bw['timing_note']}")
                if not hbm_bw["timing_sane"]:
                    bits.append(f"HBM-reference timing insane: {hbm_bw['timing_note']}")
                if physics.get("consistent_with_host_pages") is not True:
                    bits.append(physics.get("reason") or "physics inconclusive")
                arm["error"] = " | ".join(bits)
        except Exception as exc:
            arm["status"] = "failed"
            arm["passed"] = False
            arm["error"] = f"{type(exc).__name__}: {exc}"
            arm["detail"]["traceback"] = traceback.format_exc()

    # ---- finalise the placement arm now that physics exists --------------------------------------
    parm = res["arms"]["placement"]
    if parm["status"] == "pending_physics":
        det = parm["detail"]
        det["physics"] = physics
        det["physics_note"] = "measured by the bandwidth arm on this card, in this process"
        at_touch = det.get("classification_after_touch") or {}
        at_alloc = det.get("classification_at_alloc") or {}
        not_in_vram = bool(at_touch.get("not_in_vram") and at_alloc.get("not_in_vram") is not False)
        host_signal = bool(at_touch.get("positive_host_signal")
                           or at_alloc.get("positive_host_signal"))
        cpu_ok = bool((det.get("cpu_readable") or {}).get("readable"))
        phys_ok = (physics or {}).get("consistent_with_host_pages")
        signals = {
            "not_in_vram (VRAM delta < 0.25x size, hipMemGetInfo and sysfs)": not_in_vram,
            "positive_host_signal (GTT or MemAvailable moved >= 0.5x size)": host_signal,
            "cpu_accessible (real host pages; the fake VMM pages gave EFAULT)": cpu_ok,
            "physics (read rate consistent with pages behind PCIe)": phys_ok,
        }
        # PASS needs: not in VRAM, physics consistent, and at LEAST ONE positive host signal.
        # MemAvailable alone is whole-box noise on this machine, and CPU accessibility alone could
        # in principle be satisfied by a coherent device mapping -- so neither is sufficient by
        # itself, and the VRAM and physics terms are mandatory.
        passed = bool(not_in_vram and phys_ok is True and (host_signal or cpu_ok))
        det["signals"] = signals
        det["decision_rule"] = (
            "PASS iff (VRAM did not grow) AND (read rate is PCIe-consistent) AND (GTT/MemAvailable "
            "moved OR the pages are CPU-accessible). A location field or a return code counts for "
            "nothing -- that is exactly what invalidated the first P2.")
        parm["passed"] = passed
        parm["status"] = "ok" if passed else "failed"
        if not passed:
            failed_sig = [k for k, v in signals.items() if v is not True]
            parm["error"] = (
                "PLACEMENT NOT PROVEN -- failing signal(s): " + "; ".join(failed_sig) +
                ". The arena may not be host memory; every torch answer in this leg would be "
                "measured over the wrong medium.")

    have_slot = bool(slot_tensors)
    arena_t = slot_tensors[0] if have_slot else None
    # host address of the slot: the arena is ONE mapping, so the slot's CPU-visible address is its
    # device offset applied to the host pointer. Used only as a cross-check, never as the gate.
    slot_host = (arena_host + (arena_t.data_ptr() - arena_base)) if have_slot else 0

    # ---- ARM: correctness -----------------------------------------------------------------------
    arm = res["arms"]["correctness"] = new_arm(
        "does a kernel read AND write the arena tensor correctly (copy roundtrip, shader read, "
        "shader write), host-verified?")
    if not have_slot:
        arm["status"] = "skipped"
        arm["error"] = "no arena slot tensor"
    else:
        try:
            n = slot_elem
            hv = HostVerify(_HIP, n * itemsize)
            _KEEP.append(hv)
            ref = (torch.arange(n, dtype=torch.int64) % 101).to(dtype)
            ref_dev = ref.to(devstr)
            # (a) copy path -- may be served by the SDMA copy engine, so it is NOT proof a shader
            #     can write these pages. (b) and (c) supply that.
            arena_t.copy_(ref_dev)
            torch.cuda.synchronize(dev)
            back = arena_t.detach().to("cpu")
            exact_roundtrip = bool(torch.equal(back, ref))
            # (b) shader READ through the arena
            n_check = min(n, 1 << 16)
            diff = (arena_t[:n_check] - ref_dev[:n_check]).abs().max()
            torch.cuda.synchronize(dev)
            max_abs_err = float(diff.item())
            # (c) shader WRITE through the arena, host-verified, with DIFFERENT content: a stale
            #     page holding (a)'s bytes cannot also pass (c).
            arena_t.fill_(7.0)
            torch.cuda.synchronize(dev)
            hc = hv.equals_fill(arena_t.data_ptr(), n, dtype, 7.0, host_addr=slot_host)
            arm["detail"] = {
                "elements": n, "bytes": n * itemsize, "dtype": str(dtype),
                "exact_d2h_roundtrip": exact_roundtrip,
                "kernel_read_max_abs_err": max_abs_err,
                "kernel_read_elements_checked": n_check,
                "kernel_write_host_verified": hc,
                "note": ("three independent checks with two different contents in sequence, so a "
                         "stale physical page cannot pass by coincidence. The shader-write check "
                         "is host-verified (hipMemcpy D2H + CPU compare) with a second, "
                         "independent view through the hipHostMalloc CPU pointer recorded as a "
                         "coherence cross-check."),
            }
            arm["passed"] = bool(exact_roundtrip and max_abs_err == 0.0 and hc["ok"])
            arm["status"] = "ok" if arm["passed"] else "failed"
            if not arm["passed"]:
                arm["error"] = (f"arena tensor did not read back what was written "
                                f"(roundtrip={exact_roundtrip}, read_err={max_abs_err}, "
                                f"shader_write_ok={hc['ok']})")
        except Exception as exc:
            arm["status"] = "failed"
            arm["passed"] = False
            arm["error"] = f"{type(exc).__name__}: {exc}"
            arm["detail"]["traceback"] = traceback.format_exc()

    # ---- ARM: empty_cache with a LIVE user-MemPool block -----------------------------------------
    # engine/graph.py:314 calls torch.cuda.empty_cache() immediately before capture -- i.e. after
    # the offload arena already exists. Does it hand our pages back through the free callback?
    arm = res["arms"]["empty_cache_live"] = new_arm(
        "does torch.cuda.empty_cache() invoke the custom free callback for a LIVE pool block, and "
        "does the arena tensor survive it?")
    if not have_slot:
        arm["status"] = "skipped"
        arm["error"] = "no arena slot tensor"
    else:
        try:
            hv = HostVerify(_HIP, slot_elem * itemsize)
            _KEEP.append(hv)
            ptr0 = arena_t.data_ptr()
            per_rep, survived = [], True
            for i in range(args.warmup + args.reps):
                # A DIFFERENT payload every rep, self-contained. Comparing against whatever a
                # previous arm left behind cannot detect a stale/aliased page (the expected bytes
                # never change) and makes an unrelated failure cascade into the KILL criterion.
                pre_val = float((i % 97) + 1)
                post_val = float((i % 89) + 3)
                same_ptr = probe_ok = writable_after = None
                host_pre = host_post = None
                probe_err = None
                n0 = n1 = 0
                t0 = t1 = time.perf_counter_ns()
                try:
                    arena_t.fill_(pre_val)
                    torch.cuda.synchronize(dev)
                    n0 = len(_FREE_EVENTS)
                    t0 = time.perf_counter_ns()
                    torch.cuda.empty_cache()   # <- the engine/graph.py:314 call, for real
                    t1 = time.perf_counter_ns()
                    n1 = len(_FREE_EVENTS)
                    same_ptr = arena_t.data_ptr() == ptr0
                    host_pre = hv.equals_fill(arena_t.data_ptr(), slot_elem, dtype, pre_val,
                                              host_addr=slot_host)
                    probe_ok = bool(host_pre["ok"])
                    arena_t.fill_(post_val)     # still writable? a returned page faults or rots
                    torch.cuda.synchronize(dev)
                    host_post = hv.equals_fill(arena_t.data_ptr(), slot_elem, dtype, post_val,
                                               host_addr=slot_host)
                    writable_after = bool(host_post["ok"])
                except Exception as exc:   # a use-after-free lands here (or aborts the child)
                    probe_ok = False if probe_ok is None else probe_ok
                    writable_after = False
                    probe_err = f"{type(exc).__name__}: {exc}"
                survived = survived and bool(same_ptr) and bool(probe_ok) and bool(writable_after)
                per_rep.append({
                    "rep": i, "warmup": i < args.warmup, "free_callbacks": n1 - n0,
                    "data_ptr_stable": same_ptr, "pre_value": pre_val, "post_value": post_val,
                    "contents_still_correct": probe_ok,
                    "writable_after_empty_cache": writable_after,
                    "host_check_pre": host_pre, "host_check_post": host_post,
                    "probe_error": probe_err, "empty_cache_us": (t1 - t0) / 1e3,
                })
            measured = [r for r in per_rep if r["warmup"] is False]
            fired = any(r["free_callbacks"] > 0 for r in measured)
            coh = [r.get("host_check_post", {}).get("direct_cpu_view_agrees_with_memcpy")
                   for r in measured if isinstance(r.get("host_check_post"), dict)]
            arm["detail"] = {
                "verification_method": ("HOST: ctypes hipMemcpy D2H of the full slot + CPU byte "
                                        "compare. No device allocation participates."),
                "free_cb_invoked_for_live_block": fired,
                "free_callback_counts": [r["free_callbacks"] for r in measured],
                "tensor_survived": survived,
                "contents_survived": all(bool(r["contents_still_correct"]) for r in measured),
                "writable_after": all(bool(r["writable_after_empty_cache"]) for r in measured),
                "data_ptr_stable": all(bool(r["data_ptr_stable"]) for r in measured),
                "cpu_view_agreed_with_memcpy_reps": [c for c in coh],
                "empty_cache_us": _stats([r["empty_cache_us"] for r in measured]),
                "empty_cache_us_note": ("host wall around torch.cuda.empty_cache(); it "
                                        "synchronises internally. Not a device-time measurement."),
                "reps": per_rep,
            }
            # The arm PASSES if the arena survives. Whether the callback fires is the ANSWER, not
            # the pass condition -- a firing callback on a LIVE block would be the dangerous case.
            arm["passed"] = bool(survived)
            arm["status"] = "ok" if survived else "failed"
            if not survived:
                arm["error"] = ("the live pool tensor did not survive torch.cuda.empty_cache() -- "
                                "engine/graph.py:314 would destroy the offload arena")
        except Exception as exc:
            arm["status"] = "failed"
            arm["passed"] = False
            arm["error"] = f"{type(exc).__name__}: {exc}"
            arm["detail"]["traceback"] = traceback.format_exc()

    # ---- ARM: empty_cache with a CACHED (dropped, not live) pool block ---------------------------
    arm = res["arms"]["empty_cache_cached"] = new_arm(
        "when a pool tensor is dropped, does the free callback fire on del, or only on "
        "empty_cache, or never?")
    if args.skip_cached_release or not have_slot:
        arm["status"] = "skipped"
        arm["error"] = "--skip-cached-release" if args.skip_cached_release else "no slot tensor"
    else:
        try:
            n_before = len(_FREE_EVENTS)
            with torch.cuda.use_mem_pool(pool):
                tmp = torch.empty(slot_elem, dtype=dtype, device=devstr)
            tmp_ptr = tmp.data_ptr()
            n_alloc = len(_FREE_EVENTS)
            del tmp                       # refcount drop -> torch caches the block in the pool
            n_del = len(_FREE_EVENTS)
            torch.cuda.empty_cache()
            n_empty = len(_FREE_EVENTS)
            arm["detail"] = {
                "tmp_ptr": tmp_ptr,
                "tmp_ptr_in_arena": arena_base <= tmp_ptr < arena_base + arena_size,
                "free_cb_on_alloc": n_alloc - n_before,
                "free_cb_on_del": n_del - n_alloc,
                "free_cb_on_empty_cache": n_empty - n_del,
                "freed_ptrs": [e.get("ptr") for e in _FREE_EVENTS[n_alloc:]],
            }
            arm["passed"] = True   # observational: any outcome is a valid answer
            arm["status"] = "ok"
        except Exception as exc:
            arm["status"] = "failed"
            arm["passed"] = False
            arm["error"] = f"{type(exc).__name__}: {exc}"
            arm["detail"]["traceback"] = traceback.format_exc()

    # ---- ARM: graph capture over the arena tensor ------------------------------------------------
    arm = res["arms"]["graph_capture"] = new_arm(
        "can a HIP graph be captured and replayed reading the arena, with the user MemPool live, "
        "and is every one of >= 20 replays BIT-IDENTICAL to a CPU-computed ground truth?")
    if not have_slot:
        arm["status"] = "skipped"
        arm["error"] = "no arena slot tensor"
    else:
        try:
            n_view = min(slot_elem, args.capture_elems)
            nb_view = n_view * itemsize
            ptr_before_capture = arena_t.data_ptr()
            view = arena_t[:n_view]
            out = torch.empty(n_view, dtype=dtype, device=devstr)  # ordinary device memory
            _KEEP.append(out)
            hv_cap = HostVerify(_HIP, nb_view)
            _KEEP.append(hv_cap)

            # warm the op on a side stream (mandatory before capture)
            s = torch.cuda.Stream(device=dev)
            s.wait_stream(torch.cuda.current_stream(dev))
            with torch.cuda.stream(s):
                for _ in range(3):
                    out.copy_(view * 2)
            torch.cuda.current_stream(dev).wait_stream(s)
            torch.cuda.synchronize(dev)

            free_before = len(_FREE_EVENTS)
            g = torch.cuda.CUDAGraph()
            _KEEP.append(g)
            # torch.cuda.graph.__enter__ does synchronize() + gc.collect() + empty_cache() --
            # exactly the engine/graph.py:314 situation, exercised for real.
            with torch.cuda.graph(g):
                out.copy_(view * 2)
            free_during_capture = len(_FREE_EVENTS) - free_before

            # ---- loop A: VARYING input. Proves each replay really re-reads the arena; a stale or
            #      constant page cannot track a changing value.
            replay_us: list[float] = []
            event_ms: list[float] = []
            vary_ok = True
            per_replay = []
            for i in range(args.warmup + args.replays):
                val = float((i % 41) + 1)
                view.fill_(val)
                torch.cuda.synchronize(dev)
                ev0 = torch.cuda.Event(enable_timing=True)
                ev1 = torch.cuda.Event(enable_timing=True)
                t0 = time.perf_counter_ns()
                ev0.record()
                g.replay()
                ev1.record()
                torch.cuda.synchronize(dev)
                t1 = time.perf_counter_ns()
                # HOST-verified against a CPU-computed expectation: torch.cuda.graph.__enter__
                # runs empty_cache(), so a fresh DEVICE-side `expect` tensor is exactly the
                # allocation the expandable_segments defect corrupts (P5's collateral finding).
                hc = hv_cap.equals_fill(out.data_ptr(), n_view, dtype, val * 2.0)
                ok = bool(hc["ok"])
                vary_ok = vary_ok and ok
                if i >= args.warmup:
                    replay_us.append((t1 - t0) / 1e3)
                    event_ms.append(float(ev0.elapsed_time(ev1)))
                per_replay.append({"rep": i, "warmup": i < args.warmup, "input_value": val,
                                   "bytes_exactly_correct": ok, "host_check": (None if ok else hc)})

            # ---- loop B: FIXED, position-dependent input, hashed. This is the "bit-identical over
            #      >= 20 replays" claim, measured as a byte fingerprint rather than a tolerance.
            #      Ground truth is computed ON THE CPU (x*2 is exact in float32), so no device
            #      allocation and no eager device reference participates in the comparison.
            pat = (torch.arange(n_view, dtype=torch.int64) % 97).to(dtype)
            pat_bytes = pat.contiguous().numpy().tobytes()
            src = (ctypes.c_ubyte * nb_view).from_buffer_copy(pat_bytes)
            _HIP.check(_HIP.lib.hipMemcpy(ctypes.c_void_p(view.data_ptr()),
                                          ctypes.cast(src, ctypes.c_void_p),
                                          ctypes.c_size_t(nb_view),
                                          ctypes.c_int(hipMemcpyHostToDevice)),
                       "hipMemcpy H2D (capture fixed pattern)")
            _HIP.sync()
            expect_hash = hashlib.sha256(
                (pat * 2).contiguous().numpy().tobytes()).hexdigest()
            hashes: list[str] = []
            for _ in range(args.replays):
                g.replay()
                torch.cuda.synchronize(dev)
                hashes.append(hv_cap.sha256(out.data_ptr(), nb_view))
            n_ident = len(set(hashes))
            bit_identical = bool(n_ident == 1 and len(hashes) >= 20)
            matches_truth = bool(hashes and hashes[0] == expect_hash)

            arm["detail"] = {
                "captured": True,
                "capture_elements": n_view,
                "capture_bytes": nb_view,
                "free_cb_calls_during_capture_enter": free_during_capture,
                "varying_input": {
                    "n_measured_replays": args.replays,
                    "all_replays_byte_exact": vary_ok,
                    "replays": per_replay,
                    "why": ("a changing input proves each replay re-reads the arena; a stale page "
                            "or a cached constant cannot track it"),
                },
                "bit_identical": {
                    "n_replays": len(hashes),
                    "n_distinct_output_hashes": n_ident,
                    "all_replays_identical": bit_identical,
                    "matches_cpu_ground_truth": matches_truth,
                    "expected_sha256": expect_hash,
                    "observed_sha256": hashes[0] if hashes else None,
                    "distinct_hashes": sorted(set(hashes))[:8],
                    "why": ("byte fingerprint of the full graph output, computed on the CPU from "
                            "a CPU-computed expectation (x*2 is exact in float32). No device "
                            "temporary participates, so the expandable_segments stale-page defect "
                            "cannot corrupt the comparison operand."),
                },
                "replay_wall_us": _stats(replay_us),
                "replay_event_ms": _stats(event_ms),
                "data_ptr_before_capture": ptr_before_capture,
                "data_ptr_after_capture": arena_t.data_ptr(),
                "data_ptr_stable_across_capture": arena_t.data_ptr() == ptr_before_capture,
                "timing_note": (
                    f"replay times are a FUNCTIONAL latency for a {nb_view} B read+write over an "
                    "L2/MALL-resident working set (MALL is 64 MB). They are NOT a bandwidth "
                    "measurement -- the bandwidth arm above owns that, at >= 256 MiB."),
            }
            arm["passed"] = bool(vary_ok and bit_identical and matches_truth
                                 and args.replays >= 20
                                 and arena_t.data_ptr() == ptr_before_capture)
            arm["status"] = "ok" if arm["passed"] else "failed"
            if not arm["passed"]:
                bits = []
                if not vary_ok:
                    bits.append("a varying-input replay produced wrong bytes")
                if not bit_identical:
                    bits.append(f"replays were NOT bit-identical ({n_ident} distinct output "
                                f"hashes over {len(hashes)} replays)")
                if not matches_truth:
                    bits.append("replay output did not match the CPU-computed ground truth")
                if args.replays < 20:
                    bits.append(f"--replays {args.replays} < 20 required by the brief")
                if arena_t.data_ptr() != ptr_before_capture:
                    bits.append("the arena tensor's data_ptr MOVED across graph capture")
                arm["error"] = " | ".join(bits)
        except Exception as exc:
            arm["status"] = "failed"
            arm["passed"] = False
            arm["error"] = f"{type(exc).__name__}: {exc}"
            arm["detail"]["traceback"] = traceback.format_exc()

    # ---- wrap up ---------------------------------------------------------------------------------
    res["allocator_events"] = _events_summary()
    res["hipMalloc_fallbacks_total"] = int(_ARENA["fallbacks"])
    res["arena_bytes_consumed"] = int(_ARENA["cursor"])
    res["arena_bytes_total"] = int(_ARENA["size"])
    res["device"]["hip_mem_info_at_end"] = _HIP.mem_info()
    res["device"]["amdgpu_sysfs_at_end"] = card_mem(bdf)
    res["box_state_end"] = box_state(with_smi=False)
    res["passed"] = all(res["arms"][a]["passed"] is True for a in REQUIRED_ARMS)
    res["status"] = "ok" if res["passed"] else "failed"
    # A leg whose PLACEMENT failed did not measure the selected MEDIUM, so its bandwidth,
    # correctness and capture numbers are void: report it as "could not run" (exit 3), not as a
    # torch KILL. But NEVER downgrade a leg where torch measurably refused the pointer -- the
    # pointer-identity question does not depend on which medium is behind the address, and hiding
    # a real KILL behind a precondition is a worse failure than the reverse.
    if res["arms"]["placement"]["passed"] is False:
        if res["arms"]["pool_alloc"]["passed"] is False:
            res["status"] = "failed"
            res["arms"]["pool_alloc"]["error"] = (
                str(res["arms"]["pool_alloc"].get("error") or "") +
                " [NOTE: placement was ALSO unproven in this leg, so every OTHER number here is "
                "void -- but the pointer-identity failure above stands on its own and is reported "
                "as a measured NO.]")
        else:
            res["status"] = "precondition_failed"
            res["precondition_error"] = (
                "PLACEMENT NOT PROVEN: " + str(res["arms"]["placement"].get("error")) +
                " Every torch answer in this leg was measured over an unproven medium and none of "
                "them may be quoted.")
    return res


# =================================================================================================
# PARENT: orchestrate legs, merge, report
# =================================================================================================
def child_env(cfg: dict, rocr: str | None = None) -> dict:
    env = dict(os.environ)
    rocr = rocr or os.environ.get("ROCR_VISIBLE_DEVICES") or "0,1"
    env["ROCR_VISIBLE_DEVICES"] = rocr
    # The child re-forces device visibility at import time, BEFORE torch, from this variable --
    # without it the child silently resets to the "0,1" literal and ignores --rocr-devices.
    env["MINISGL_P5B_ROCR_DEVICES"] = rocr
    for k in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL"):
        env.pop(k, None)
    env["PYTHONUNBUFFERED"] = "1"
    if cfg["alloc_conf"]:
        env["PYTORCH_CUDA_ALLOC_CONF"] = cfg["alloc_conf"]
        env["PYTORCH_HIP_ALLOC_CONF"] = cfg["alloc_conf"]
    else:
        env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        env.pop("PYTORCH_HIP_ALLOC_CONF", None)
    return env


def run_children(args, legs: list[str]) -> dict:
    """One child per leg, STRICTLY SERIALLY.

    The GPU lease is waived for this work by explicit user instruction, which makes serialisation
    this script's responsibility rather than the arbiter's: two GPU jobs must never run at once on
    this box. Legs also mutate global VRAM/host state, so overlapping them would corrupt every
    placement delta.
    """
    out: dict = {}
    for leg in legs:
        cfg = LEGS[leg]
        leg_json = os.path.join(args.out_dir, f"p5b_leg_{leg}.json")
        leg_log = os.path.join(args.out_dir, f"p5b_leg_{leg}.log")
        for p in (leg_json, leg_log):
            try:
                os.unlink(p)
            except FileNotFoundError:
                pass
        cmd = [
            sys.executable, os.path.abspath(__file__),
            "--child-leg", leg,
            "--out-dir", args.out_dir,
            "--scratch-dir", args.scratch_dir,
            "--arena-bytes", str(args.arena_bytes),
            "--bw-bytes", str(args.bw_bytes),
            "--slot-bytes", str(args.slot_bytes),
            "--reps", str(args.reps),
            "--warmup", str(args.warmup),
            "--replays", str(args.replays),
            "--capture-elems", str(args.capture_elems),
            "--dtype", args.dtype,
            "--host-flags", hex(args.host_flags),
            "--rocr-devices", args.rocr_devices,
        ]
        if args.allow_outside_image:
            cmd.append("--allow-outside-image")
        if args.skip_cached_release:
            cmd.append("--skip-cached-release")
        print(f"[P5b] leg {leg}: alloc_conf={cfg['alloc_conf'] or '<unset>'} "
              f"torch_device={cfg['device']}", flush=True)
        t0 = time.time()
        proc = subprocess.run(cmd, env=child_env(cfg, args.rocr_devices), capture_output=True,
                              text=True, timeout=args.leg_timeout)
        dt = time.time() - t0
        try:
            with open(leg_log, "w") as fh:
                fh.write(f"$ {' '.join(cmd)}\n--- stdout ---\n{proc.stdout}\n"
                         f"--- stderr ---\n{proc.stderr}\n--- rc={proc.returncode} ---\n")
            _chown_if_asked(leg_log)
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
            "wall_s": dt, "log": leg_log,
            "stderr_tail": proc.stderr[-4000:], "stdout_tail": proc.stdout[-2000:],
        }
        if proc.returncode in (-6, 134, -11, 139):
            leg_res["status"] = "aborted"
            leg_res["passed"] = False
            leg_res["error"] = (
                f"child died with returncode {proc.returncode} -- see {leg_log}. This is the "
                "c10::Error 'invalid device pointer' / SIGSEGV class the probe guards against, "
                "and it is a MEASURED NO: torch refused the host-derived pointer.")
        out[leg] = leg_res
        print(f"[P5b] leg {leg}: status={leg_res.get('status')} rc={proc.returncode} "
              f"({dt:.1f}s)", flush=True)
    return out


def _tri(vals: list) -> bool | None:
    """AND over tri-state observations. None if nothing was observed -- an unmeasured claim is
    NOT a pass, and must never be silently promoted to one."""
    obs = [v for v in vals if v is not None]
    return None if not obs else all(bool(v) for v in obs)


def _any_tri(vals: list) -> bool | None:
    obs = [v for v in vals if v is not None]
    return None if not obs else any(bool(v) for v in obs)


def build_verdict(legs: dict) -> dict:  # noqa: C901
    reasons: list[str] = []

    def arm_pass(leg: dict, name: str):
        return leg.get("arms", {}).get(name, {}).get("passed")

    def arm_detail(leg: dict, name: str, *keys):
        d = leg.get("arms", {}).get(name, {}).get("detail", {})
        for k in keys:
            if not isinstance(d, dict):
                return None
            d = d.get(k)
        return d

    ran = {k: v for k, v in legs.items() if v.get("status") != "not_run"}
    usable = {k: v for k, v in ran.items()
              if v.get("status") in ("ok", "failed", "aborted", "precondition_failed")}

    # Placement first: a leg whose medium was not proven cannot contribute ANY torch evidence.
    placement = _tri([arm_pass(v, "placement") for v in ran.values()])
    proven = {k: v for k, v in usable.items() if arm_pass(v, "placement") is True}
    unproven = sorted(k for k, v in usable.items() if arm_pass(v, "placement") is False)
    if unproven:
        reasons.append(
            "PLACEMENT NOT PROVEN in leg(s) %s -- hipHostMalloc(Mapped) pages did not demonstrate "
            "host residency (VRAM delta / MemAvailable delta / GTT delta / achieved bandwidth vs "
            "the 28.7 GB/s PCIe ceiling). Their torch results are DISCARDED, because a "
            "torch-plumbing answer measured over the wrong medium is not an answer -- that is "
            "exactly what invalidated the first P2." % unproven)

    # The core claim, evaluated ONLY over legs with a proven medium, and ANDed: a passing control
    # leg must never mask a failing gating leg.
    foreign = _tri([arm_pass(v, "pool_alloc") for v in proven.values()])
    ptr_legs = {k: v for k, v in proven.items() if arm_pass(v, "pool_alloc") is True}
    capture = _tri([arm_pass(v, "graph_capture") for v in ptr_legs.values()])
    bit_ident = _tri([arm_detail(v, "graph_capture", "bit_identical", "all_replays_identical")
                      for v in ptr_legs.values()])
    survives = _tri([arm_detail(v, "empty_cache_live", "tensor_survived")
                     for v in proven.values()])
    live_cb = _any_tri([arm_detail(v, "empty_cache_live", "free_cb_invoked_for_live_block")
                        for v in usable.values()])
    cached_cb = _any_tri([
        (None if arm_detail(v, "empty_cache_cached", "free_cb_on_empty_cache") is None
         else arm_detail(v, "empty_cache_cached", "free_cb_on_empty_cache") > 0)
        for v in usable.values()])

    # Coexistence may only be claimed by a leg that VERIFIED expandable_segments is live.
    exp_legs = {k: v for k, v in proven.items()
                if LEGS.get(k, {}).get("alloc_conf") == EXPANDABLE
                and v.get("torch", {}).get("expandable_segments_verified") is True}
    exp_unverified = [k for k, v in usable.items()
                      if LEGS.get(k, {}).get("alloc_conf") == EXPANDABLE
                      and v.get("torch", {}).get("expandable_segments_verified") is not True]
    coexist = None
    if exp_legs:
        coexist = _tri([arm_pass(v, "pool_alloc") for v in exp_legs.values()]
                       + [arm_pass(v, "graph_capture") for v in exp_legs.values()])
    if exp_unverified:
        reasons.append(
            "expandable_segments could NOT be verified live in leg(s) %s; their coexistence "
            "evidence is discarded (an unread env var is not evidence)" % sorted(exp_unverified))

    # BOTH CARDS. Phase 0 section 6 item 9 exists because this box has burned people on cross-card
    # assumptions twice; "we ran two legs" is not the claim, "two DIFFERENT physical cards
    # answered" is.
    cards = {k: (v.get("device", {}).get("physical_card_index"),
                 v.get("device", {}).get("pci_bus_id"))
             for k, v in ran.items() if v.get("device")}
    ok_cards = {k: c for k, c in cards.items()
                if k in proven and arm_pass(proven[k], "pool_alloc") is True}
    distinct_bdf = {c[1] for c in ok_cards.values() if c[1]}
    both_cards = None
    if not cards:
        both_cards = None
    else:
        both_cards = len(distinct_bdf) >= 2
    if both_cards is False:
        reasons.append(
            "only %d distinct physical card(s) (%s) produced a passing foreign-pointer result. "
            "P2/P5/P6 all ran card 0 only and this box's two cards are NOT symmetric (card 1's "
            "root port trains Gen4 x8, half the bandwidth) -- a single-card green does not "
            "generalise." % (len(distinct_bdf), sorted(distinct_bdf)))

    bandwidth = {k: {"arena_read_gbps": arm_detail(v, "bandwidth", "physics", "arena_read_gbps"),
                     "copy_engine_gbps": arm_detail(v, "bandwidth", "physics",
                                                    "copy_engine_gbps"),
                     "hbm_reference_gbps": arm_detail(v, "bandwidth", "physics",
                                                      "hbm_reference_gbps"),
                     "arena_frac_of_hbm": arm_detail(v, "bandwidth", "physics",
                                                     "arena_frac_of_hbm"),
                     "physical_card": v.get("device", {}).get("physical_card_index"),
                     "pci_bus_id": v.get("device", {}).get("pci_bus_id")}
                 for k, v in ran.items()}

    per_leg_claims = {
        k: {a: arm_pass(v, a) for a in ARM_NAMES} | {
            "status": v.get("status"),
            "physical_card": v.get("device", {}).get("physical_card_index"),
            "pci_bus_id": v.get("device", {}).get("pci_bus_id"),
            "hipMalloc_fallbacks_total": v.get("hipMalloc_fallbacks_total"),
            "placement_classification": arm_detail(v, "placement", "classification_after_touch",
                                                   "classification"),
        }
        for k, v in ran.items()
    }

    gating_present = [k for k in sorted(GATING_LEGS) if k in legs]
    gating_ok = bool(gating_present)
    if not gating_present:
        reasons.append("no gating leg was run (need %s -- both cards)" % sorted(GATING_LEGS))
    for k in gating_present:
        st = legs[k].get("status")
        if st != "ok":
            gating_ok = False
            why = legs[k].get("error") or legs[k].get("precondition_error") or st
            reasons.append(f"gating leg {k}: {st} -- {why}")

    fallbacks = {k: v.get("hipMalloc_fallbacks_total") for k, v in ran.items()}
    any_fallback = any(isinstance(n, int) and n > 0 for n in fallbacks.values())
    if any_fallback:
        reasons.append(
            "hipMalloc FALLBACKS occurred (%s): torch was handed memory that is NOT inside the "
            "host arena, so those tensors were on DEVICE memory and any latency, bandwidth or "
            "capture result from such a leg is measuring the wrong thing." % fallbacks)

    p5b_pass = bool(gating_ok and placement is True and foreign is True and capture is True
                    and bit_ident is True and survives is True and both_cards is True
                    and not any_fallback)

    if foreign is False:
        reasons.append(
            "KILL: t.data_ptr() != the hipHostGetDevicePointer address (or torch fell back to "
            "hipMalloc). The ~30-line ctypes arena does NOT work over the selected mechanism -- "
            "M1-A needs a C++ from_blob extension instead (+3 days). Do not work around this.")
    elif foreign is None:
        reasons.append("the foreign-pointer claim was NOT MEASURED over a proven-host medium -- "
                       "that is not an answer, re-run the probe")
    if capture is False or bit_ident is False:
        reasons.append(
            "KILL: graph capture over the host arena did not replay bit-identically. Graph "
            "capture is mandatory for a feature to count as complete in this repo, so an arena "
            "that cannot be captured is not a landable arena.")
    if survives is False:
        reasons.append("KILL: the arena tensor did NOT survive torch.cuda.empty_cache() -- "
                       "engine/graph.py:314 calls it and would destroy the offload arena")
    if live_cb is True:
        reasons.append(
            "the custom FREE callback fired for a LIVE pool block. It is a no-op here, so nothing "
            "broke -- but any real implementation that actually frees on that callback would hand "
            "back pages torch still points at.")
    if p5b_pass and len(reasons) == 0:
        reasons.append(
            "GREEN on both cards: hipHostMalloc(Mapped) pages are provably host-resident, torch "
            "allocates over the host-derived device pointer with zero fallbacks, the arena "
            "survives empty_cache(), and >= 20 graph replays are bit-identical to a CPU-computed "
            "ground truth. M1-A can use the ctypes arena; no C++ extension is needed.")

    return {
        "p5b_pass": p5b_pass,
        "host_arena_placement_proven": placement,
        "foreign_ptr_ok": foreign,
        "expandable_segments_coexists": coexist,
        "capture_ok": capture,
        "capture_bit_identical": bit_ident,
        "tensor_survives_empty_cache": survives,
        "empty_cache_invokes_free_cb_live_block": live_cb,
        "empty_cache_invokes_free_cb_cached_block": cached_cb,
        "hipMalloc_fallbacks_seen": any_fallback,
        "both_cards_exercised": both_cards,
        "hipMalloc_fallbacks_per_leg": fallbacks,
        "bandwidth_gbps_per_leg": bandwidth,
        "per_leg_claims": per_leg_claims,
        "legs_with_proven_placement": sorted(proven),
        "legs_with_verified_expandable_segments": sorted(exp_legs),
        "distinct_physical_cards": sorted(distinct_bdf),
        "reasons": reasons,
        "gating_legs": gating_present,
        "consequence_if_failed": (
            "M1-A cannot use the ~30-line ctypes arena and needs a C++ torch::from_blob extension "
            "plus a build-system change: +3 days on the critical path."),
    }


def classify_exit(verdict: dict, legs: dict) -> tuple[int, dict]:
    """Separate 'we asked torch and it said NO' (2, a KILL) from 'the probe could not run' (3).

    Attribution drives this, not a flag scan. A leg that could not prove its medium, or could not
    build the arena, has NOT produced a negative torch answer -- reporting one would be a KILL for
    something that was never measured, which is the specific mistake this probe family has already
    made once.
    """
    kill: list[str] = []
    cannot_run: list[str] = []

    if verdict.get("foreign_ptr_ok") is False:
        kill.append("foreign_ptr_ok is False: t.data_ptr() != hipHostGetDevicePointer address")
    if verdict.get("capture_ok") is False:
        kill.append("capture_ok is False: graph replay over the arena was wrong")
    if verdict.get("capture_bit_identical") is False:
        kill.append("capture_bit_identical is False: replays were not byte-reproducible")
    if verdict.get("tensor_survives_empty_cache") is False:
        kill.append("tensor_survives_empty_cache is False: graph.py:314 would destroy the arena")
    if verdict.get("hipMalloc_fallbacks_seen") is True:
        kill.append("hipMalloc fallbacks occurred inside the user MemPool")
    if verdict.get("host_arena_placement_proven") is False:
        cannot_run.append(
            "placement was NOT proven in at least one leg: the arena may not be host memory, so "
            "no torch answer from it counts. This is a MECHANISM problem, not a torch answer.")

    for name in sorted(GATING_LEGS):
        leg = legs.get(name)
        if leg is None:
            cannot_run.append(f"gating leg {name} was not run (BOTH cards are required)")
            continue
        st = leg.get("status")
        if st == "aborted":
            kill.append(f"gating leg {name} ABORTED (rc={leg.get('child', {}).get('returncode')})"
                        " -- the c10::Error 'invalid device pointer' class")
        elif st in ("precondition_failed", "no_output", "not_run", "crashed"):
            cannot_run.append(f"gating leg {name}: {st} -- "
                              f"{leg.get('precondition_error') or leg.get('error')}")
        elif st == "failed":
            failed_arms = [a for a, v in leg.get("arms", {}).items() if v.get("passed") is False]
            kill.append(f"gating leg {name} failed arm(s) {failed_arms}")

    if verdict.get("both_cards_exercised") is False and not kill:
        cannot_run.append("only one physical card produced a passing result; the brief requires "
                          "both, and these two cards are not symmetric")

    if verdict.get("p5b_pass"):
        return EXIT_OK, {"code": EXIT_OK, "kill_signals": [], "cannot_run": cannot_run,
                         "why": "all gating legs green on both cards"}
    if kill:
        return EXIT_MEASURED_FAIL, {
            "code": EXIT_MEASURED_FAIL, "kill_signals": kill, "cannot_run": cannot_run,
            "why": ("a MEASURED NEGATIVE from torch -- this is a real KILL. M1-A needs a C++ "
                    "from_blob extension instead (+3 days).")}
    return EXIT_PRECONDITION, {
        "code": EXIT_PRECONDITION, "kill_signals": [], "cannot_run": cannot_run,
        "why": "the probe could not run to an answer -- NOT a kill, re-run (see cannot_run)"}


# =================================================================================================
# output
# =================================================================================================
def _chown_if_asked(path: str) -> None:
    """Root-owned artifacts in a bind-mounted worktree are the CLAUDE.md trap: a later non-root rm
    fails, and the next run's 'clean' step silently reuses a stale file."""
    uid, gid = os.environ.get("P5B_CHOWN_UID"), os.environ.get("P5B_CHOWN_GID")
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


def render_md(report: dict) -> str:
    v = report["verdict"]
    bs = report.get("box_state", {})
    mem = bs.get("meminfo", {})

    def gb(k):
        x = mem.get(k)
        return f"{x/1024/1024:.1f} GB" if isinstance(x, int) else "n/a"

    def tri(x):
        return {True: "YES", False: "**NO**", None: "not measured"}[x if x in (True, False)
                                                                    else None]

    def num(x, fmt="{:.2f}"):
        return fmt.format(x) if isinstance(x, (int, float)) else "-"

    lines = [
        "# P5b — torch over a `hipHostGetDevicePointer` address (in-image, under capture)",
        "",
        f"**Run:** {bs.get('t_iso')} · host `{bs.get('hostname')}` · in-container "
        f"`{bs.get('in_container')}` · image `{bs.get('provenance', {}).get('image')}` "
        f"(`{str(bs.get('provenance', {}).get('image_id'))[:19]}`) · schema `{report['schema']}`",
        (
            "**THIS IS A SELFTEST ARTIFACT — no GPU was touched and NOTHING below was measured.**"
            if report.get("synthetic")
            else f"**Verdict:** **{'PASS' if v['p5b_pass'] else 'FAIL'}** "
                 f"(exit {report['exit_code']})"
        ),
        "",
        "> Mechanism under test: **`hipHostMalloc(Mapped|Portable)` + `hipHostGetDevicePointer`** —",
        "> the T1 tier Phase 0 selected after `hipMemCreate(location=Host)` was found to return",
        "> device VRAM silently. P5 validated torch plumbing over a VMM reservation, i.e. over",
        "> **device** memory; this is the first time the *selected* mechanism meets torch.",
        "",
        "## Answers",
        "",
        "| Question | Answer |",
        "|---|---|",
        f"| are the arena's pages **provably host-resident** (VRAM delta, MemAvailable/GTT delta, "
        f"CPU accessibility, read rate vs the PCIe ceiling)? | {tri(v['host_arena_placement_proven'])} |",
        f"| `t.data_ptr()` == the host-derived device pointer, under `use_mem_pool` | "
        f"{tri(v['foreign_ptr_ok'])} |",
        f"| zero `hipMalloc` fallbacks inside the pool | "
        f"{tri(None if v['hipMalloc_fallbacks_seen'] is None else not v['hipMalloc_fallbacks_seen'])} |",
        f"| coexists with `expandable_segments:True` (compose default) | "
        f"{tri(v['expandable_segments_coexists'])} |",
        f"| arena survives `torch.cuda.empty_cache()` (`engine/graph.py:314`) | "
        f"{tri(v['tensor_survives_empty_cache'])} |",
        f"| free callback fires for a **live** pool block | "
        f"{tri(v['empty_cache_invokes_free_cb_live_block'])} |",
        f"| free callback fires for a **cached (dropped)** pool block | "
        f"{tri(v['empty_cache_invokes_free_cb_cached_block'])} |",
        f"| graph capture + replay over the arena is correct | {tri(v['capture_ok'])} |",
        f"| ≥ 20 replays **bit-identical** to a CPU-computed ground truth | "
        f"{tri(v['capture_bit_identical'])} |",
        f"| **both physical cards** exercised | {tri(v['both_cards_exercised'])} "
        f"(cards: {v.get('distinct_physical_cards')}) |",
        "",
        f"**Exit classification:** {v.get('exit_classification', {}).get('why', 'n/a')}",
        "",
        "**If this probe is RED:** " + v.get("consequence_if_failed", ""),
        "",
        "**Reasons**",
        "",
    ]
    lines += [f"- {r}" for r in v["reasons"]] or ["- (none)"]

    lines += [
        "",
        "## Placement proof — per leg",
        "",
        "| leg | physical card | PCI | classification | VRAM Δ / size | MemAvail Δ / size | "
        "GTT Δ / size | CPU readable | arena read GB/s | copy engine GB/s | HBM ref GB/s | "
        "arena / HBM |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, leg in report["legs"].items():
        d = leg.get("device", {})
        pl = (leg.get("arms", {}).get("placement", {}).get("detail", {})
              .get("classification_after_touch") or {})
        cpu = (leg.get("arms", {}).get("placement", {}).get("detail", {})
               .get("cpu_readable") or {})
        bw = v.get("bandwidth_gbps_per_leg", {}).get(name, {})
        lines.append(
            f"| {name} | {d.get('physical_card_index')} | `{d.get('pci_bus_id')}` | "
            f"{pl.get('classification', '-')} | {num(pl.get('vram_consumed_frac'), '{:.3f}')} | "
            f"{num(pl.get('memavailable_consumed_frac'), '{:.3f}')} | "
            f"{num(pl.get('sysfs_gtt_used_delta_frac'), '{:.3f}')} | "
            f"{cpu.get('readable')} | {num(bw.get('arena_read_gbps'))} | "
            f"{num(bw.get('copy_engine_gbps'))} | {num(bw.get('hbm_reference_gbps'))} | "
            f"{num(bw.get('arena_frac_of_hbm'), '{:.4f}')} |")
    lines += [
        "",
        "> A read that outruns the copy engine by >1.5× or reaches ≥0.25× the HBM reference did "
        "**not** cross PCIe — those pages are device memory whatever the counters said. That check "
        "is the one the first P2 lacked (device 117.9 vs \"host\" 118.2 GB/s, ratio 1.00).",
        "> Expected PCIe ceilings on this box: **card 0 = 28.70 GB/s (Gen5 x8)**, "
        "**card 1 = 14.34 GB/s (Gen4 x8)** — sampled mid-DMA, never at idle (ASPM downtrains).",
        "",
        "## Arms — per leg",
        "",
        "| leg | alloc conf | status | arena | placement | pool_alloc | bandwidth | correctness | "
        "empty_cache | capture | alloc µs (via our allocator) | replay µs |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, leg in report["legs"].items():
        cfg = LEGS.get(name, {})
        arms = leg.get("arms", {})

        def a(n, _arms=arms):
            p = _arms.get(n, {}).get("passed")
            return {True: "ok", False: "**FAIL**", None: "-"}[p if p in (True, False) else None]

        al = (arms.get("pool_alloc", {}).get("detail", {})
              .get("alloc_latency_us_via_custom_allocator") or {})
        rp = arms.get("graph_capture", {}).get("detail", {}).get("replay_wall_us", {}) or {}
        lines.append(
            f"| {name} | {cfg.get('alloc_conf') or '&lt;unset&gt;'} | {leg.get('status')} | "
            f"{a('arena')} | {a('placement')} | {a('pool_alloc')} | {a('bandwidth')} | "
            f"{a('correctness')} | {a('empty_cache_live')} | {a('graph_capture')} | "
            f"{num(al.get('median'), '{:.1f}')} | {num(rp.get('median'), '{:.1f}')} |")

    lines += [
        "",
        "> `alloc µs` counts only reps that actually reached the custom allocator; reps torch "
        "served from its own cache are reported separately in the JSON.",
        "> Replay times are a functional latency over an L2/MALL-resident working set — **not** a "
        "bandwidth measurement. The bandwidth column above is the ≥256 MiB one.",
        "",
        "## Box state at run start",
        "",
        f"- MemTotal {gb('MemTotal_kB')}, MemAvailable {gb('MemAvailable_kB')}, "
        f"MemFree {gb('MemFree_kB')}, Cached {gb('Cached_kB')}",
        f"- swap: SwapTotal {gb('SwapTotal_kB')}, SwapFree {gb('SwapFree_kB')}, "
        f"pswpout {bs.get('vmstat', {}).get('pswpout')}",
        f"- loadavg `{bs.get('loadavg')}`",
        f"- GPU lease **waived by explicit user instruction**; legs run strictly serially, never "
        f"two GPU jobs at once.",
        "",
        "Raw: `p5b.json` (this directory), per-leg `p5b_leg_<name>.json` + `.log`.",
        "",
    ]
    return "\n".join(lines)


# =================================================================================================
# selftest -- validates argument handling, the decision logic, and the JSON shape, with NO GPU
# =================================================================================================
def synthetic_report(args) -> dict:
    """A structurally complete report in which EVERY measured value is null."""
    legs = {}
    for leg in parse_legs(args.legs):
        sk = leg_skeleton(leg, LEGS[leg], args)
        sk["status"] = "selftest"
        sk["synthetic"] = True
        for name in ARM_NAMES:
            sk["arms"][name]["status"] = "selftest"
            sk["arms"][name]["question"] = "selftest placeholder"
        legs[leg] = sk
    verdict = {k: (False if k == "p5b_pass" else None) for k in VERDICT_TRISTATE_KEYS}
    verdict.update({
        "hipMalloc_fallbacks_per_leg": {}, "bandwidth_gbps_per_leg": {}, "per_leg_claims": {},
        "legs_with_proven_placement": [], "legs_with_verified_expandable_segments": [],
        "distinct_physical_cards": [], "gating_legs": sorted(GATING_LEGS & set(legs)),
        "reasons": ["selftest: nothing was measured"],
        "consequence_if_failed": "selftest placeholder",
    })
    return {
        "schema": SCHEMA_VERSION, "probe": PROBE_ID, "kind": "selftest", "synthetic": True,
        "note": "SELFTEST OUTPUT -- no GPU was touched, no value here was measured.",
        "argv": sys.argv, "config": vars(args), "box_state": box_state(with_smi=False),
        "legs": legs, "verdict": verdict, "exit_code": EXIT_OK,
    }


def selftest(args) -> int:  # noqa: C901
    problems: list[str] = []

    def want(cond, msg):
        if not cond:
            problems.append(msg)

    # 1. leg table -------------------------------------------------------------------------------
    for name, cfg in LEGS.items():
        want(cfg["alloc_conf"] in ("", EXPANDABLE), f"leg {name}: bad alloc_conf")
        want(cfg["device"] in (0, 1), f"leg {name}: bad device index {cfg['device']}")
    want(GATING_LEGS <= set(LEGS), "GATING_LEGS references an unknown leg")
    want({LEGS[g]["device"] for g in GATING_LEGS} == {0, 1},
         "the gating legs do not cover BOTH cards -- the brief requires both")

    # 2. leg parsing rejects garbage --------------------------------------------------------------
    try:
        parse_legs("not_a_leg")
        problems.append("parse_legs accepted an unknown leg name")
    except (SystemExit, ValueError):
        pass
    try:
        parse_legs("")
        problems.append("parse_legs accepted an empty leg list")
    except (SystemExit, ValueError):
        pass

    # 3. child env composition --------------------------------------------------------------------
    e_exp = child_env(LEGS["expandable_card0"])
    e_pln = child_env(LEGS["plain_card0"])
    want(e_exp.get("PYTORCH_CUDA_ALLOC_CONF") == EXPANDABLE,
         "expandable leg did not set PYTORCH_CUDA_ALLOC_CONF")
    want(e_exp.get("PYTORCH_HIP_ALLOC_CONF") == EXPANDABLE,
         "expandable leg did not set PYTORCH_HIP_ALLOC_CONF (the compose default is the HIP one)")
    want("PYTORCH_CUDA_ALLOC_CONF" not in e_pln and "PYTORCH_HIP_ALLOC_CONF" not in e_pln,
         "plain leg leaked an alloc-conf env var")
    want(e_exp.get("ROCR_VISIBLE_DEVICES") == "0,1", "ROCR_VISIBLE_DEVICES not forced")
    want("HIP_VISIBLE_DEVICES" not in e_exp, "HIP_VISIBLE_DEVICES leaked into the child env")
    want("2" not in [s.strip() for s in (e_exp.get("ROCR_VISIBLE_DEVICES") or "").split(",")],
         "ROCm device 2 (the Ryzen iGPU, 47 GB GTT) is in the child's visible set")
    e_alt = child_env(LEGS["expandable_card1"], "1")
    want(e_alt.get("MINISGL_P5B_ROCR_DEVICES") == "1" and e_alt.get("ROCR_VISIBLE_DEVICES") == "1",
         "child_env did not propagate an explicit ROCR device set (the child re-forces "
         "visibility from MINISGL_P5B_ROCR_DEVICES before torch, so it is the only channel)")

    # 4. stats never fabricate --------------------------------------------------------------------
    empty = _stats([])
    want(empty["median"] is None and empty["n"] == 0, "_stats([]) fabricated a value")
    five = _stats([3.0, 1.0, 2.0, 5.0, 4.0])
    want(five["median"] == 3.0 and five["min"] == 1.0 and five["max"] == 5.0,
         "_stats gave the wrong median/spread")

    # 5. the bump allocator's arithmetic, with no GPU in sight ------------------------------------
    saved = dict(_ARENA)
    try:
        _ARENA.update({"base": 0x100000, "host": 0x200000, "size": 4096, "cursor": 0,
                       "fallbacks": 0, "device": 0})
        a_cb, f_cb = _install_callbacks()
        p1 = a_cb(1000, 0, None)
        p2 = a_cb(1000, 0, None)
        want(p1 == 0x100000, f"bump alloc did not return the arena base (got {p1:#x})")
        want(p2 == 0x100000 + 1024, f"bump alloc did not 512-align (got {p2:#x})")
        n_fb = int(_ARENA["fallbacks"])
        a_cb(1 << 30, 0, None)   # unservable: _HIP is None -> records a fallback, never a NULL
        want(int(_ARENA["fallbacks"]) == n_fb + 1,
             "an oversize request was not recorded as a hipMalloc fallback")
        n_fb = int(_ARENA["fallbacks"])
        a_cb(16, 1, None)        # wrong device: must NOT be served from this arena
        want(int(_ARENA["fallbacks"]) == n_fb + 1,
             "a request for a DIFFERENT device was served from the arena -- that hands torch "
             "memory the asking card may not address")
        n_free = len(_FREE_EVENTS)
        f_cb(p1, 1000, 0, None)
        want(len(_FREE_EVENTS) == n_free + 1, "free callback did not record the call")
    finally:
        _ARENA.clear()
        _ARENA.update(saved)
        _ALLOC_EVENTS.clear()
        _FREE_EVENTS.clear()

    # 6. the placement classifier, including THE TRAP that invalidated the first P2 ----------------
    SZ = 1 << 30

    def sample(vram_free, avail, vram_used=None, gtt_used=None):
        sysfs = ({} if vram_used is None else
                 {"pci_bdf": "0000:03:00.0", "mem_info_vram_used": vram_used,
                  "mem_info_gtt_used": gtt_used, "mem_info_vram_total": 16 << 30,
                  "mem_info_gtt_total": 32 << 30})
        return {"hip_free_vram_bytes": vram_free, "mem_available_bytes": avail, "sysfs": sysfs}

    # (a) THE TRAP: an allocation that CLAIMS host but consumes VRAM and no host RAM.
    trap = classify_placement(SZ, sample(8 << 30, 40 << 30, 0, 0),
                              sample((8 << 30) - SZ, 40 << 30, SZ, 0))
    want(trap["classification"] == "device_resident",
         f"the classifier did NOT catch a VRAM-consuming 'host' region (got "
         f"{trap['classification']}) -- this is exactly the defect that invalidated the first P2")
    want(trap["not_in_vram"] is False, "trap region was reported as not_in_vram")
    # (b) a genuine host region: VRAM flat, GTT and MemAvailable both move.
    good = classify_placement(SZ, sample(8 << 30, 40 << 30, 100, 0),
                              sample(8 << 30, (40 << 30) - SZ, 100, SZ))
    want(good["classification"] == "host_resident",
         f"a genuine host region was not classified host_resident (got {good['classification']})")
    want(good["not_in_vram"] and good["positive_host_signal"],
         "a genuine host region failed the not_in_vram / positive_host_signal terms")
    # (c) sysfs is PRIMARY and a disagreement is recorded rather than averaged away.
    dis = classify_placement(SZ, sample(8 << 30, 40 << 30, 0, 0),
                             sample((8 << 30) - SZ, (40 << 30) - SZ, 0, SZ))
    want(dis["classification_source"].startswith("amdgpu sysfs"),
         "sysfs did not take priority over hipMemGetInfo")
    want(dis["classifications_agree"] is False,
         "a sysfs/hipMemGetInfo disagreement was not flagged")
    # (d) no sysfs at all -> falls back, and says so
    nofs = classify_placement(SZ, sample(8 << 30, 40 << 30), sample(8 << 30, (40 << 30) - SZ))
    want(nofs["classification"] == "host_resident" and "hipMemGetInfo" in
         nofs["classification_source"], "the no-sysfs fallback path is broken")

    # 7. the physics gate ------------------------------------------------------------------------
    #    Every case below is a real measurement from this box's Phase 0.
    fake_host = physics_verdict(690.0, 28.7, 692.0, 0)      # P1: "host" VMM pages read at HBM
    want(fake_host["consistent_with_host_pages"] is False,
         "physics accepted a 690 GB/s read as host-resident (that is HBM)")
    p2_trap = physics_verdict(118.2, 28.7, 117.9, 0)        # P2: device 117.9 vs "host" 118.2
    want(p2_trap["consistent_with_host_pages"] is False,
         "physics accepted the P2 trap (ratio 1.00 against the device arm)")
    real0 = physics_verdict(28.9, 28.7, 692.0, 0)           # P1 card 0, the real thing
    want(real0["consistent_with_host_pages"] is True,
         "physics rejected a genuine card-0 host read at 28.9 GB/s")
    real1 = physics_verdict(14.5, 14.34, 690.0, 1)          # P1 card 1 (Gen4 x8 root port)
    want(real1["consistent_with_host_pages"] is True,
         "physics rejected a genuine card-1 host read at 14.5 GB/s")
    want(physics_verdict(None, 28.7, 692.0, 0)["consistent_with_host_pages"] is None,
         "physics invented a verdict with no measurement")
    want(physics_verdict(40.0, 28.7, 692.0, 0)["gates"]["under_abs_pcie_ceiling"] is False,
         "a 40 GB/s read passed the absolute PCIe ceiling gate")

    # 8. verdict + exit classification ------------------------------------------------------------
    v0 = build_verdict({})
    want(v0["p5b_pass"] is False and v0["foreign_ptr_ok"] is None,
         "build_verdict({}) invented a result")

    def _leg(status, arms=None, card=0, bdf="0000:03:00.0", rc=0, exp_verified=True):
        return {"status": status, "arms": arms or {},
                "device": {"physical_card_index": card, "pci_bus_id": bdf},
                "torch": {"expandable_segments_verified": exp_verified},
                "hipMalloc_fallbacks_total": 0, "child": {"returncode": rc}}

    ok_arms = {a: {"status": "ok", "passed": True, "detail": {}} for a in ARM_NAMES}
    ok_arms["empty_cache_live"] = {"status": "ok", "passed": True,
                                   "detail": {"tensor_survived": True,
                                              "free_cb_invoked_for_live_block": False}}
    ok_arms["graph_capture"] = {"status": "ok", "passed": True,
                                "detail": {"bit_identical": {"all_replays_identical": True}}}
    bad_alloc = dict(ok_arms, pool_alloc={"status": "failed", "passed": False, "detail": {}})
    bad_place = dict(ok_arms, placement={"status": "failed", "passed": False, "detail": {}})

    def two(a0, a1, s0="ok", s1="ok"):
        return {"expandable_card0": _leg(s0, a0, 0, "0000:03:00.0"),
                "expandable_card1": _leg(s1, a1, 1, "0000:07:00.0")}

    # (i) both cards green -> PASS, exit 0
    legs_ok = two(ok_arms, ok_arms)
    v_ok = build_verdict(legs_ok)
    want(v_ok["p5b_pass"] is True, f"a fully green two-card run did not pass: {v_ok['reasons']}")
    want(v_ok["both_cards_exercised"] is True, "two distinct BDFs were not counted as both cards")
    want(classify_exit(v_ok, legs_ok)[0] == EXIT_OK, "a green run did not exit 0")
    # (ii) placement unproven on a gating leg -> COULD NOT RUN (3), never a KILL
    legs_pl = two(bad_place, ok_arms, s0="precondition_failed")
    v_pl = build_verdict(legs_pl)
    c_pl, why_pl = classify_exit(v_pl, legs_pl)
    want(c_pl == EXIT_PRECONDITION,
         f"an UNPROVEN-PLACEMENT gating leg was classified {c_pl}, want {EXIT_PRECONDITION} "
         f"({why_pl}) -- reporting a torch KILL for a medium we never established is the exact "
         f"mistake that produced the first P2's void result")
    want(v_pl["foreign_ptr_ok"] is not False,
         "a leg with unproven placement contributed a FALSE foreign-pointer answer")
    # (iii) the pointer claim measurably fails -> KILL (2)
    legs_kill = two(bad_alloc, bad_alloc, s0="failed", s1="failed")
    want(classify_exit(build_verdict(legs_kill), legs_kill)[0] == EXIT_MEASURED_FAIL,
         "a measured foreign-pointer failure was not classified as a KILL")
    # (iii-b) placement unproven AND torch measurably refused the pointer -> still a KILL. Hiding a
    #         real torch NO behind a precondition would be worse than the reverse: the pointer
    #         identity question does not depend on which medium is behind the address.
    both_bad = dict(bad_alloc, placement={"status": "failed", "passed": False, "detail": {}})
    legs_both = two(both_bad, both_bad, s0="failed", s1="failed")
    want(classify_exit(build_verdict(legs_both), legs_both)[0] == EXIT_MEASURED_FAIL,
         "a pointer refusal was downgraded to a precondition because placement also failed")
    # (iv) an ABORTED gating leg (the c10::Error class) -> KILL (2)
    legs_ab = {"expandable_card0": _leg("aborted", {}, 0, "0000:03:00.0", rc=-6),
               "expandable_card1": _leg("ok", ok_arms, 1, "0000:07:00.0")}
    want(classify_exit(build_verdict(legs_ab), legs_ab)[0] == EXIT_MEASURED_FAIL,
         "an ABORTED gating leg was not classified as a KILL")
    # (v) a passing CONTROL leg must not mask a failing gating leg
    legs_mask = {"expandable_card0": _leg("failed", bad_alloc, 0, "0000:03:00.0"),
                 "plain_card0": _leg("ok", ok_arms, 0, "0000:03:00.0")}
    want(build_verdict(legs_mask)["foreign_ptr_ok"] is False,
         "a passing control leg masked a failing gating leg in foreign_ptr_ok")
    # (vi) an UNMEASURED empty_cache survival must not pass
    no_ec = dict(ok_arms, empty_cache_live={"passed": True, "detail": {}})
    legs_noec = two(no_ec, no_ec)
    want(build_verdict(legs_noec)["p5b_pass"] is False,
         "p5b_pass was granted without a MEASURED empty_cache survival")
    # (vii) one card only -> not a pass, and not a KILL either
    legs_one = {"expandable_card0": _leg("ok", ok_arms, 0, "0000:03:00.0")}
    v_one = build_verdict(legs_one)
    want(v_one["p5b_pass"] is False and v_one["both_cards_exercised"] is False,
         "a single-card run was allowed to pass the both-cards requirement")
    want(classify_exit(v_one, legs_one)[0] == EXIT_PRECONDITION,
         "a single-card run was classified as a KILL")
    # (viii) a hipMalloc fallback is a KILL even if every arm ticked
    legs_fb = two(ok_arms, ok_arms)
    legs_fb["expandable_card0"]["hipMalloc_fallbacks_total"] = 3
    v_fb = build_verdict(legs_fb)
    want(v_fb["hipMalloc_fallbacks_seen"] is True and v_fb["p5b_pass"] is False,
         "a hipMalloc fallback inside the pool did not fail the run")
    want(classify_exit(v_fb, legs_fb)[0] == EXIT_MEASURED_FAIL,
         "a hipMalloc fallback was not classified as a KILL")

    # 9. report shape + markdown -------------------------------------------------------------------
    report = synthetic_report(args)
    report["exit_code"] = EXIT_OK if not problems else EXIT_PRECONDITION
    try:
        validate_report(report)
    except Exception as exc:
        problems.append(f"schema validation failed: {exc}")
    md = ""
    try:
        md = render_md(report)
        want("P5b" in md, "markdown render lost its title")
        want("hipHostGetDevicePointer" in md, "markdown does not name the mechanism under test")
    except Exception as exc:
        problems.append(f"render_md raised: {type(exc).__name__}: {exc}")
    # a REAL-shaped report must render too -- the synthetic one exercises none of the tables
    try:
        real = dict(report)
        real["kind"] = "measurement"
        real["synthetic"] = False
        real["legs"] = legs_ok
        real["verdict"] = dict(v_ok, exit_classification=classify_exit(v_ok, legs_ok)[1])
        validate_report(real)
        want("PASS" in render_md(real), "a green report did not render as PASS")
    except Exception as exc:
        problems.append(f"a measurement-shaped report failed to validate/render: "
                        f"{type(exc).__name__}: {exc}")

    # 10. output paths -- NEVER onto the real p5b.json ---------------------------------------------
    os.makedirs(args.out_dir, exist_ok=True)
    _chown_if_asked(args.out_dir)
    jpath = os.path.join(args.out_dir, "p5b.selftest.json")
    mpath = os.path.join(args.out_dir, "p5b.selftest.md")
    report["problems"] = problems
    write_json(jpath, report)
    write_text(mpath, md)

    print(json.dumps({"selftest": "ok" if not problems else "FAILED", "n_problems": len(problems),
                      "problems": problems, "wrote": [jpath, mpath]}, indent=2))
    if problems:
        print("SELFTEST FAILED", file=sys.stderr)
        return EXIT_PRECONDITION
    return EXIT_OK


# =================================================================================================
# CLI
# =================================================================================================
def parse_legs(spec: str) -> list[str]:
    names = [s.strip() for s in (spec or "").split(",") if s.strip()]
    if not names:
        raise ValueError("no legs requested")
    bad = [n for n in names if n not in LEGS]
    if bad:
        raise ValueError(f"unknown leg(s) {bad}; known: {sorted(LEGS)}")
    return names


def _int_maybe_hex(s: str) -> int:
    return int(s, 0)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                   help="where p5b.json / p5b.md / per-leg files land (default: %(default)s)")
    p.add_argument("--scratch-dir", default=DEFAULT_SCRATCH_DIR,
                   help="writable dir for the CPU-accessibility syscall probes (must be a real "
                        "filesystem; /dev/null is not a valid sink). Default: %(default)s")
    p.add_argument("--legs", default=DEFAULT_LEGS,
                   help=f"comma-separated subset of {sorted(LEGS)} (default: %(default)s)")
    p.add_argument("--arena-bytes", type=int, default=0,
                   help="pinned host arena size in bytes (0 = derive from bw/slot/reps)")
    p.add_argument("--bw-bytes", type=int, default=256 << 20,
                   help="working set for the placement bandwidth check (default 256 MiB; must "
                        "exceed the 64 MB MALL or the number is a cache artifact)")
    p.add_argument("--slot-bytes", type=int, default=16 << 20,
                   help="bytes per pool allocation for the latency/correctness reps "
                        "(default 16 MiB, ~6x the 2.8 MiB expert granule)")
    p.add_argument("--reps", type=int, default=6, help="measured repetitions (>=5) (default 6)")
    p.add_argument("--warmup", type=int, default=1, help="discarded warm-up reps (>=1)")
    p.add_argument("--replays", type=int, default=24,
                   help="measured graph replays per loop (>=20 required by the brief)")
    p.add_argument("--capture-elems", type=int, default=1 << 20,
                   help="elements of the arena the captured graph reads (default 1Mi)")
    p.add_argument("--dtype", default="float32", help="torch dtype for the arena tensors")
    p.add_argument("--host-flags", type=_int_maybe_hex, default=hipHostMallocMapped
                   | hipHostMallocPortable,
                   help="hipHostMalloc flags (default 0x3 = Mapped|Portable). Mapped is "
                        "mandatory -- without it there is no device pointer.")
    p.add_argument("--leg-timeout", type=int, default=1800, help="per-leg child timeout, seconds")
    p.add_argument("--rocr-devices", default="0,1",
                   help="ROCR_VISIBLE_DEVICES to force (default %(default)s; ROCm device 2 is the "
                        "Ryzen iGPU and must never be enumerated)")
    p.add_argument("--allow-outside-image", action="store_true",
                   help="permit running outside the serve image (debugging only; P5b must be "
                        "confirmed IN-IMAGE and an outside result must never be quoted)")
    p.add_argument("--skip-cached-release", action="store_true",
                   help="skip the dropped-block empty_cache arm")
    p.add_argument("--no-lock", action="store_true",
                   help="skip the cross-process lock. The GPU lease is waived for this work, so "
                        "the lock is the only thing stopping two GPU probes running at once.")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--selftest", action="store_true",
                   help="validate args, allocator arithmetic, the placement/physics classifiers "
                        "and the JSON shape WITHOUT touching a GPU")
    g.add_argument("--dry-run", action="store_true", help="alias for --selftest")
    g.add_argument("--child-leg", default=None, help=argparse.SUPPRESS)  # internal: one leg here
    return p


def _derive_arena_bytes(args) -> int:
    """Headroom x3 on the slots: torch's caching allocator rounds a mid-size request up and then
    SPLITS it, so the bytes it asks us for exceed the bytes it hands back. Undersizing the arena
    surfaces as hipMalloc fallbacks -- i.e. a FALSE KILL."""
    n_slots = args.reps + args.warmup + 2
    return _round_up(args.bw_bytes + 3 * n_slots * args.slot_bytes, 1 << 21)


def main(argv: list[str]) -> int:  # noqa: C901
    args = build_parser().parse_args(argv)
    if args.dry_run:
        args.selftest = True
    if not args.arena_bytes:
        args.arena_bytes = _derive_arena_bytes(args)

    os.environ["ROCR_VISIBLE_DEVICES"] = args.rocr_devices
    for k in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL"):
        os.environ.pop(k, None)

    # ---- argument gates. A probe that runs with too few reps produces a confident wrong number. --
    if not args.selftest:
        fatal = []
        if args.reps < 5:
            fatal.append(f"--reps {args.reps} < 5: the spec requires >= 5 measured reps")
        if args.warmup < 1:
            fatal.append("--warmup must be >= 1 (a warm-up iteration must be discarded)")
        if args.replays < 20:
            fatal.append(f"--replays {args.replays} < 20: the brief requires >= 20 bit-identical "
                         f"replays")
        if args.bw_bytes < (64 << 20):
            fatal.append(f"--bw-bytes {args.bw_bytes} <= the 64 MB MALL: any bandwidth number "
                         f"would be a cache artifact, which is how the withdrawn 127.2 GB/s "
                         f"figure happened")
        if not (args.host_flags & hipHostMallocMapped):
            fatal.append(f"--host-flags 0x{args.host_flags:x} lacks hipHostMallocMapped (0x2); "
                         f"without it hipHostGetDevicePointer has nothing to return")
        need = args.bw_bytes + (args.reps + args.warmup + 1) * args.slot_bytes
        if args.arena_bytes < need:
            fatal.append(f"--arena-bytes {args.arena_bytes} < {need} needed for the bandwidth "
                         f"tensor plus every slot; torch would fall back to hipMalloc and the run "
                         f"would report a FALSE KILL")
        if args.capture_elems <= 0:
            fatal.append("--capture-elems must be positive")
        if fatal:
            for f in fatal:
                print(f"FATAL: {f}", file=sys.stderr)
            return EXIT_PRECONDITION
    try:
        legs = parse_legs(args.legs)
    except ValueError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return EXIT_PRECONDITION

    os.makedirs(args.out_dir, exist_ok=True)
    # A root-owned output DIRECTORY is the CLAUDE.md trap: the files inside can be chowned but a
    # later non-root rm still fails on the directory's write bit. Chown the dir too.
    _chown_if_asked(args.out_dir)

    if args.selftest:
        return selftest(args)

    # ---- child mode: ONE leg, in this process ---------------------------------------------------
    if args.child_leg:
        leg = args.child_leg
        if leg not in LEGS:
            print(f"FATAL: unknown leg {leg!r}", file=sys.stderr)
            return EXIT_PRECONDITION
        path = os.path.join(args.out_dir, f"p5b_leg_{leg}.json")
        try:
            res = run_leg(leg, LEGS[leg], args)
        except BaseException as exc:
            res = leg_skeleton(leg, LEGS[leg], args)
            res["status"] = "crashed"
            res["passed"] = False
            res["error"] = f"{type(exc).__name__}: {exc}"
            res["traceback"] = traceback.format_exc()
            res["box_state"] = box_state(with_smi=False)
            res["allocator_events"] = _events_summary()
        write_json(path, res)
        print(json.dumps({"leg": leg, "status": res.get("status"), "passed": res.get("passed"),
                          "error": res.get("error") or res.get("precondition_error"),
                          "json": path}, indent=2, default=str), flush=True)
        code = (EXIT_OK if res.get("passed") else
                EXIT_PRECONDITION if res.get("status") in ("precondition_failed", "crashed")
                else EXIT_MEASURED_FAIL)
        sys.stdout.flush()
        sys.stderr.flush()
        # HARD exit: no destructor may run. Tearing down a MemPool with live blocks aborts the
        # process, which would corrupt the exit code just computed and lose the artifact.
        os._exit(code)

    # ---- parent mode ----------------------------------------------------------------------------
    lock_fh = None
    if not args.no_lock:
        # The GPU lease is waived by explicit user instruction, so nothing else on this box stops
        # two P5b runs (or a P5b and a stray probe) from sharing the cards and corrupting every
        # placement delta. This lock is that guard. It is advisory and P5b-specific -- it does NOT
        # coordinate with other agents' jobs, which is why the run protocol also says "serial".
        try:
            import fcntl
            lock_path = os.path.join(args.out_dir, ".p5b.lock")
            lock_fh = open(lock_path, "w")
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            lock_fh.write(f"pid={os.getpid()} t={time.time()}\n")
            lock_fh.flush()
            _chown_if_asked(lock_path)
        except BlockingIOError:
            print(f"FATAL: another P5b run holds {args.out_dir}/.p5b.lock. Two GPU jobs must "
                  f"never run at once on this box. Wait, or pass --no-lock if you are certain.",
                  file=sys.stderr)
            return EXIT_PRECONDITION
        except Exception as exc:
            print(f"[P5b] WARNING: could not take the run lock ({type(exc).__name__}: {exc}); "
                  f"proceeding UNLOCKED -- make sure no other GPU job is running", file=sys.stderr)

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
        "title": ("P5b -- torch over a hipHostGetDevicePointer address, in the serve image, "
                  "under graph capture, on both cards"),
        "plan_ref": ("docs/measurements/WEIGHT_OFFLOAD_2026-09-02/PHASE0_REPORT.md section 6, "
                     "unknown #3 (blocks M1-A)"),
        "mechanism": "hipHostMalloc(Mapped|Portable) + hipHostGetDevicePointer",
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
        write_json(os.path.join(args.out_dir, "p5b.INVALID.json"), report)
        return EXIT_PRECONDITION

    jpath = os.path.join(args.out_dir, "p5b.json")
    mpath = os.path.join(args.out_dir, "p5b.md")
    write_json(jpath, report)
    write_text(mpath, render_md(report))
    print(json.dumps(report, indent=2, default=str))
    print(f"\n[P5b] wrote {jpath}\n[P5b] wrote {mpath}\n[P5b] verdict: "
          f"{'PASS' if verdict['p5b_pass'] else 'FAIL'} (exit {code})", file=sys.stderr)
    for r in verdict["reasons"]:
        print(f"[P5b]   - {r}", file=sys.stderr)
    if lock_fh is not None:
        lock_fh.close()
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
