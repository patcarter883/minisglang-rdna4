#!/usr/bin/env python
"""P2 — mixed-media grouped MoE GEMM: correctness first, then the miss-count SHAPE.

WHAT THIS ANSWERS (docs/WEIGHT_OFFLOAD_PLAN.md §3 P2, §11 unknown #2)
--------------------------------------------------------------------
Launch ONE real `minisgl.quant.kernels.w4a8_moe` grouped MoE forward over an expert stack whose
expert rows straddle DEVICE-located and HOST-located physical pages inside a single reserved device
VA range (the plan's placement-only architecture, §2). Then:

  1. CORRECTNESS. Bit-compare the mixed-media launch against an all-device-resident reference stack
     holding byte-identical weights and driven by an identical route. The M<=2 gemm2 arm is an fp32
     atomic SCATTER and is order-nondeterministic AGAINST ITSELF (`kernels.py:662-663`), so the gate
     is `max|arena - ref| <= ctl`, where `ctl` is the SAME-CONFIG reference-vs-reference floor
     measured in this very run. Where `ctl == 0` the gate is exact bit-equality.

  2. THE SHAPE. Per-layer wall time as a function of MISS COUNT m = how many of the top-`k` routed
     experts sit on host pages, swept over m in {0,1,2,5,9,10}. Report whether time is LINEAR in m
     or CLIFFED (the workgroup-completion effect: a layer runs at HBM speed only if ALL k routed
     experts are resident, so P(fast layer) = h^k and a per-expert device tier buys almost nothing).

KILL (plan §3, §10): incorrect results, or a cliff worse than linear in miss count. If the FIRST
miss already costs most of what TEN misses cost, T2's per-expert device tier is worthless and only
T1 (all-host) or explicit-copy layer-group streaming survive. The discriminator reported here is

    cliff_index = (t(1) - t(0)) / ((t(k) - t(0)) / k)        1.0 = perfectly linear, k = full cliff

MEASUREMENT TRAPS THIS SCRIPT IS BUILT AROUND
---------------------------------------------
* CACHE RESIDENCY. 10 experts x ~2.4 MB = ~24 MB fits inside the 64 MB MALL. A naive
  "same route, timed in a loop" measurement would time an Infinity-Cache hit and report host pages
  running at HBM speed. Every timed window therefore replays a ROTATION of `R` DISJOINT routes so
  the reuse distance for any expert is `R * k * granule` bytes; the run REFUSES to report a number
  unless that exceeds 4x MALL. A `cache_sensitivity` diagnostic (R=1 vs R=R at m=k) measures the
  effect directly rather than assuming it away.
* A CAPABILITY PROBE THAT PASSES WHILE THE OPERATION FAILS. `hipMemMap` returns `hipSuccess` on
  this box even when the page table is wrong (plan §0, §5.4 A1.4). So before any weight is written
  the arena is fingerprinted per row (first AND last element of every row of every component) with
  a value block that is DISJOINT per component, ALL components written before ANY is verified — a
  one-at-a-time check cannot see cross-component aliasing — and the markers are then proved
  globally distinct. After populate, every element of every row is bit-compared against the
  reference (not a 64-row sample), and a checksum is taken and RE-CHECKED after the timed run so a
  mid-flight clobber cannot pass as a measurement. Two media checks run: a contiguous 256 MB
  device- vs host-backed scratch read, AND a gather over >=256 MB of REAL weight rows from each
  half of the stack (the scratch handles are different `hipMemCreate` calls, so scratch alone does
  not prove the expert stack straddles two media). If "host" reads at device speed the probe ABORTS
  instead of reporting a beautiful flat miss-count curve.
* WARM-UP. Two eager passes on a side stream, then one full timed window measured and DISCARDED
  before any recorded rep; >=5 kept reps, reported as median + full spread (min/p10/p90/max/MAD),
  never a mean of everything. The eager cross-check is itself a median over several passes.
* NOISE MASQUERADING AS A CURVE. `cliff_index` is a ratio of two DIFFERENCES of noisy medians, and
  it prints to two decimal places whether or not it is resolvable. The classification bands are
  therefore applied to the noise-propagated INTERVAL, the first-miss cost must clear 3 combined
  MADs to count as measured, and an implied host bandwidth above 64 GB/s at m=k (the link ceiling
  here is 28.7 GB/s) means the "host" experts were never on the far side of PCIe.
* A BASELINE THAT IS NOT ORDINARY VRAM. t(0) is the denominator of everything, so the same m=0
  rotation is also timed against a plain torch-allocated stack: if the arena's ~1000 VMM handles
  cost page-table time, the span shrinks and the cliff index is distorted with nothing to show it.
* PROVENANCE. The `engaged()` ledger is cleared and snapshotted per leg; the arena leg and the
  reference leg MUST engage the identical kernel set, or a "correct and fast" result is really
  "silently dispatched somewhere else". Recorded in the JSON, and a mismatch is a hard failure. The
  full MINISGL_*/HIP_*/HSA_* environment is recorded too: this probe measures the SHIPPED defaults
  and pins nothing, so a stray knob must be visible in the artifact.

WHAT IT DOES NOT DO
-------------------
It does not answer P1 (host-page kernel-read bandwidth across access patterns) or P5 (torch over a
foreign device pointer via `MemPool`). It deliberately binds torch to the reservation through a
hand-built DLPack capsule, NOT `torch._C._cuda_customAllocator` + `torch.cuda.MemPool`, so that P2's
verdict does not depend on P5's open question.

RUN
---
    bash tools/offload/p2_run.sh            # see that script for the exact docker invocation

EXIT CODES
----------
    0  the probe ran and produced a verdict (read `verdict` in the JSON: PASS / KILL / AMBIGUOUS /
       NOT_MEASURED). PASS additionally requires the correctness, media, arena-baseline, execution-
       mode and noise gates to hold — a curve that cannot be classified is AMBIGUOUS, never PASS.
    1  unexpected exception (harness bug)
    2  a PRECONDITION failed — no verdict was reached. Whatever HAD completed is written to
       `p2.aborted.json` with verdict NOT_MEASURED; nothing is reported as if it were a result
    3  only with --fail-on-kill, when the measured verdict is a KILL
"""
from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone

PROBE = "P2"
SCRIPT_PATH = os.path.abspath(__file__)
DEFAULT_OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_PATH))),
    "docs", "measurements", "WEIGHT_OFFLOAD_2026-09-02",
)

MALL_BYTES = 64 << 20          # gfx1201 Infinity Cache
MALL_SAFETY = 4                # required reuse distance, in MALL multiples
SCRATCH_BYTES = 256 << 20      # media-separation arm working set (4x MALL)
HIP_SO = "libamdhip64.so"

# hipMemAllocationType / hipMemLocationType / hipMemAccessFlags
_PINNED, _LOC_DEV, _LOC_HOST, _ACCESS_RW = 0x1, 1, 2, 3
_GRAN_MINIMUM = 0

# DLPack
_DL_CPU, _DL_CUDA, _DL_ROCM = 1, 2, 10
_DL_INT, _DL_UINT, _DL_FLOAT, _DL_BFLOAT = 0, 1, 2, 4


class Precondition(RuntimeError):
    """A precondition failed. Nothing was measured; nothing may be reported."""


# --------------------------------------------------------------------------------------------
# environment
# --------------------------------------------------------------------------------------------
def fix_env() -> dict:
    """ROCm device 2 on this box is the Ryzen iGPU advertising ~47 GB of GTT. If it ever enters
    enumeration it poisons every 'biggest free pool' / auto-size decision. Pin the visible set to
    the two discrete gfx1201 cards HERE, before torch is imported, and clear HIP_VISIBLE_DEVICES so
    it cannot double-filter (CLAUDE.md: setting BOTH breaks whenever card 1 is assigned)."""
    before = {k: os.environ.get(k) for k in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES")}
    os.environ["ROCR_VISIBLE_DEVICES"] = "0,1"
    os.environ.pop("HIP_VISIBLE_DEVICES", None)
    # engaged() only populates its ledger when the log is ON; we diff that ledger per leg.
    os.environ["MINISGL_HIP_ENGAGE_LOG"] = "1"
    # Every MINISGL_* knob changes which kernel the served path dispatches (MOE_G2FUSE,
    # MOE_FUSED_SILU, MOE_BLOCK_M, NVFP4_GEMV, MOE_SPLITK_SCATTER, …). The probe deliberately does
    # NOT pin them — it must measure the SHIPPED defaults — so it records the environment it
    # actually ran under, or a curve measured with a stray knob set is indistinguishable from one
    # that was not.
    knobs = {k: v for k, v in sorted(os.environ.items())
             if k.startswith(("MINISGL_", "HIP_", "ROCR_", "AMD_", "HSA_", "PYTORCH_"))}
    return {"before": before, "after": {"ROCR_VISIBLE_DEVICES": "0,1", "HIP_VISIBLE_DEVICES": None},
            "runtime_knobs": knobs}


# --------------------------------------------------------------------------------------------
# box state — recorded with every measurement; the box is NOT idle
# --------------------------------------------------------------------------------------------
def _proc_kv(path: str, keys) -> dict:
    out = {k: None for k in keys}
    try:
        with open(path) as fh:
            for line in fh:
                parts = line.split()
                if not parts:
                    continue
                name = parts[0].rstrip(":")
                if name in out:
                    out[name] = int(parts[1]) if len(parts) > 1 and parts[1].lstrip("-").isdigit() else None
    except OSError as exc:
        out["_error"] = f"{type(exc).__name__}: {exc}"
    return out


def _cmd(argv, timeout=20) -> dict:
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return {"argv": argv, "rc": p.returncode, "stdout": p.stdout[-8000:], "stderr": p.stderr[-2000:]}
    except Exception as exc:  # noqa: BLE001 — a missing tool is data, not a failure
        return {"argv": argv, "rc": None, "error": f"{type(exc).__name__}: {exc}"}


def collect_box_state(when: str) -> dict:
    return {
        "when": when,
        "wall_clock_utc": datetime.now(timezone.utc).isoformat(),
        "monotonic": time.monotonic(),
        "meminfo_kb": _proc_kv("/proc/meminfo", (
            "MemTotal", "MemFree", "MemAvailable", "Buffers", "Cached", "Dirty",
            "SwapTotal", "SwapFree", "Shmem", "Mlocked",
        )),
        "vmstat": _proc_kv("/proc/vmstat", ("pswpin", "pswpout", "pgmajfault", "pgpgin", "pgpgout")),
        "loadavg": open("/proc/loadavg").read().strip() if os.path.exists("/proc/loadavg") else None,
        "free_g": _cmd(["free", "-g"]),
        "rocm_smi": _cmd(["rocm-smi", "--showmeminfo", "vram", "--showuse", "--showpower"]),
        "nproc": os.cpu_count(),
        "host": platform.node(),
        "kernel": platform.release(),
    }


# --------------------------------------------------------------------------------------------
# HIP VMM, via ctypes (torch-free surface)
# --------------------------------------------------------------------------------------------
class _MemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class _AllocFlags(ctypes.Structure):
    _fields_ = [("compressionType", ctypes.c_ubyte), ("gpuDirectRDMACapable", ctypes.c_ubyte),
                ("usage", ctypes.c_ushort)]


class _MemAllocationProp(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("requestedHandleType", ctypes.c_int),
                ("location", _MemLocation), ("win32HandleMetaData", ctypes.c_void_p),
                ("allocFlags", _AllocFlags)]


class _MemAccessDesc(ctypes.Structure):
    _fields_ = [("location", _MemLocation), ("flags", ctypes.c_int)]


class Hip:
    """Thin ctypes binding. NO graceful degradation: a missing symbol raises (plan §5.1)."""

    def __init__(self) -> None:
        self.lib = ctypes.CDLL(HIP_SO)
        sigs = {
            "hipMemGetAllocationGranularity": [ctypes.POINTER(ctypes.c_size_t),
                                               ctypes.POINTER(_MemAllocationProp), ctypes.c_int],
            "hipMemAddressReserve": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t,
                                     ctypes.c_size_t, ctypes.c_void_p, ctypes.c_ulonglong],
            "hipMemAddressFree": [ctypes.c_void_p, ctypes.c_size_t],
            "hipMemCreate": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t,
                             ctypes.POINTER(_MemAllocationProp), ctypes.c_ulonglong],
            "hipMemMap": [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p,
                          ctypes.c_ulonglong],
            "hipMemUnmap": [ctypes.c_void_p, ctypes.c_size_t],
            "hipMemSetAccess": [ctypes.c_void_p, ctypes.c_size_t,
                                ctypes.POINTER(_MemAccessDesc), ctypes.c_size_t],
            "hipMemRelease": [ctypes.c_void_p],
            "hipDeviceGetPCIBusId": [ctypes.c_char_p, ctypes.c_int, ctypes.c_int],
        }
        missing = []
        for name, argtypes in sigs.items():
            fn = getattr(self.lib, name, None)
            if fn is None:
                missing.append(name)
                continue
            fn.argtypes = argtypes
            fn.restype = ctypes.c_int
        if missing:
            raise Precondition(f"{HIP_SO} is missing required VMM symbols: {missing}")

    def ck(self, rc: int, what: str) -> None:
        if rc != 0:
            raise Precondition(f"{what} -> hipError {rc}")

    def prop(self, loc_type: int, dev: int) -> _MemAllocationProp:
        p = _MemAllocationProp()
        ctypes.memset(ctypes.byref(p), 0, ctypes.sizeof(p))
        p.type = _PINNED
        p.location.type = loc_type
        p.location.id = dev
        return p

    def granularity(self, loc_type: int, dev: int) -> int:
        g = ctypes.c_size_t(0)
        pr = self.prop(loc_type, dev)
        self.ck(self.lib.hipMemGetAllocationGranularity(ctypes.byref(g), ctypes.byref(pr),
                                                        _GRAN_MINIMUM),
                f"hipMemGetAllocationGranularity(loc={loc_type})")
        if g.value <= 0:
            raise Precondition(f"granularity query returned {g.value} for loc={loc_type}")
        return int(g.value)

    def pci_bus_id(self, dev: int) -> str:
        buf = ctypes.create_string_buffer(64)
        rc = self.lib.hipDeviceGetPCIBusId(buf, 64, dev)
        return buf.value.decode() if rc == 0 else f"<hipError {rc}>"


# --------------------------------------------------------------------------------------------
# DLPack: bind a torch tensor to a chosen VA, WITHOUT torch's allocator
# --------------------------------------------------------------------------------------------
class _DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int32), ("device_id", ctypes.c_int32)]


class _DLDataType(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8), ("lanes", ctypes.c_uint16)]


class _DLTensor(ctypes.Structure):
    _fields_ = [("data", ctypes.c_void_p), ("device", _DLDevice), ("ndim", ctypes.c_int32),
                ("dtype", _DLDataType), ("shape", ctypes.POINTER(ctypes.c_int64)),
                ("strides", ctypes.POINTER(ctypes.c_int64)), ("byte_offset", ctypes.c_uint64)]


class _DLManagedTensor(ctypes.Structure):
    pass


_DLDeleter = ctypes.CFUNCTYPE(None, ctypes.POINTER(_DLManagedTensor))
_DLManagedTensor._fields_ = [("dl_tensor", _DLTensor), ("manager_ctx", ctypes.c_void_p),
                             ("deleter", _DLDeleter)]

# These must outlive every tensor built from them: the capsule stores the name POINTER, and the
# deleter trampoline is called by torch when the tensor dies. Module-level, never collected —
# exactly the singleton discipline the plan flags for the P5 allocator trampolines (§3 P5).
_DL_KEEPALIVE: list = []
_DL_NAME = ctypes.c_char_p(b"dltensor")
_DL_NOOP_DELETER = _DLDeleter(lambda _p: None)
_DL_KEEPALIVE.append(_DL_NOOP_DELETER)
_DL_DEVICE_TYPE: "int | None" = None


def _dl_dtype(torch, dt):
    table = {
        torch.int32: (_DL_INT, 32), torch.int8: (_DL_INT, 8), torch.int64: (_DL_INT, 64),
        torch.uint8: (_DL_UINT, 8),
        torch.float16: (_DL_FLOAT, 16), torch.float32: (_DL_FLOAT, 32),
        torch.float64: (_DL_FLOAT, 64), torch.bfloat16: (_DL_BFLOAT, 16),
    }
    if dt not in table:
        raise Precondition(f"no DLPack dtype mapping for {dt}")
    return table[dt]


def _make_capsule(torch, ptr: int, shape, dtype, device_index: int, device_type: int):
    code, bits = _dl_dtype(torch, dtype)
    ndim = len(shape)
    shp = (ctypes.c_int64 * ndim)(*[int(s) for s in shape])
    strides = [1] * ndim
    for i in range(ndim - 2, -1, -1):
        strides[i] = strides[i + 1] * int(shape[i + 1])
    strd = (ctypes.c_int64 * ndim)(*strides)
    mt = _DLManagedTensor()
    ctypes.memset(ctypes.byref(mt), 0, ctypes.sizeof(mt))
    mt.dl_tensor.data = ctypes.c_void_p(ptr)
    mt.dl_tensor.device.device_type = device_type
    mt.dl_tensor.device.device_id = device_index
    mt.dl_tensor.ndim = ndim
    mt.dl_tensor.dtype.code = code
    mt.dl_tensor.dtype.bits = bits
    mt.dl_tensor.dtype.lanes = 1
    mt.dl_tensor.shape = shp
    mt.dl_tensor.strides = strd
    mt.dl_tensor.byte_offset = 0
    mt.manager_ctx = None
    mt.deleter = _DL_NOOP_DELETER
    _DL_KEEPALIVE.extend([mt, shp, strd])
    new = ctypes.pythonapi.PyCapsule_New
    new.restype = ctypes.py_object
    new.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    return new(ctypes.cast(ctypes.byref(mt), ctypes.c_void_p), _DL_NAME, None)


def tensor_at(torch, ptr: int, shape, dtype, device_index: int):
    """A torch tensor whose storage IS the given device VA. Non-owning; the arena outlives it."""
    global _DL_DEVICE_TYPE
    candidates = [_DL_DEVICE_TYPE] if _DL_DEVICE_TYPE is not None else [_DL_ROCM, _DL_CUDA]
    errors = []
    for dt in candidates:
        try:
            t = torch.utils.dlpack.from_dlpack(_make_capsule(torch, ptr, shape, dtype,
                                                             device_index, dt))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"device_type={dt}: {type(exc).__name__}: {exc}")
            continue
        if t.data_ptr() != ptr or not t.is_cuda or t.device.index != device_index:
            errors.append(f"device_type={dt}: bound wrong "
                          f"(ptr=0x{t.data_ptr():x} want 0x{ptr:x}, dev={t.device})")
            continue
        _DL_DEVICE_TYPE = dt
        return t
    raise Precondition("could not bind a torch tensor to the reservation via DLPack: "
                       + " | ".join(errors))


# --------------------------------------------------------------------------------------------
# layout — pure integer arithmetic, unit-testable without a GPU
# --------------------------------------------------------------------------------------------
class Component:
    """One expert-major weight component: row `e` lives at `base + e*row_bytes`, exactly as the
    kernels' implicit-contiguous pointer arithmetic assumes (plan §2, component-major [ADJ])."""

    __slots__ = ("name", "row_shape", "itemsize", "dtype_name", "row_bytes", "offset", "base")

    def __init__(self, name, row_shape, itemsize, dtype_name):
        self.name = name
        self.row_shape = tuple(int(v) for v in row_shape)
        self.itemsize = int(itemsize)
        self.dtype_name = dtype_name
        n = 1
        for v in self.row_shape:
            n *= v
        self.row_bytes = n * self.itemsize
        self.offset = None
        self.base = None

    def as_dict(self, E):
        return {"name": self.name, "row_shape": list(self.row_shape), "dtype": self.dtype_name,
                "row_bytes": self.row_bytes, "region_bytes": self.row_bytes * E,
                "va_offset": self.offset}


def build_components(hidden: int, inter: int, group: int) -> "list[Component]":
    """The w4a8 op layout, as produced by `awq_to_op_layout` / `_GroupedAWQExperts.post_load`:
        w13        (E, 2*inter, hidden//8)  int32     gate|up stacked
        w13_scales (E, hidden//group, 2*inter) fp16   GROUP-major — kernels.py:522-534 asserts it
        w2         (E, hidden, inter//8)    int32
        w2_scales  (E, inter//group, hidden)  fp16
    Symmetric int4 (no zeros): `w13_zeros`/`w2_zeros` are None, as every current shipped W4A8 MoE
    arm passes them."""
    if hidden % group or inter % group:
        raise Precondition(f"hidden={hidden} and inter={inter} must both be multiples of "
                           f"group={group} (gemm2 contracts over K=inter)")
    if hidden % 8 or inter % 8:
        raise Precondition(f"hidden={hidden} and inter={inter} must be multiples of 8 (int4 pack)")
    return [
        Component("w13", (2 * inter, hidden // 8), 4, "int32"),
        Component("w13_scales", (hidden // group, 2 * inter), 2, "float16"),
        Component("w2", (hidden, inter // 8), 4, "int32"),
        Component("w2_scales", (inter // group, hidden), 2, "float16"),
    ]


def plan_layout(components, E: int, gran: int, scratch_bytes: int) -> dict:
    """Assign VA offsets. Every component region base AND every row must be granularity-aligned:
    the device/host boundary is per-ROW here (the plan permits a split row, §2, but a probe that
    splits rows cannot attribute time to a miss count, so this one refuses instead)."""
    bad = [c.name for c in components if c.row_bytes % gran]
    if bad:
        raise Precondition(
            f"row_bytes not a multiple of the VMM granularity {gran} for {bad}: "
            + ", ".join(f"{c.name}={c.row_bytes}" for c in components)
            + " — pick shapes whose per-expert row is granularity-aligned, or the miss count is "
              "not attributable to a row.")
    off = 0
    for c in components:
        c.offset = off
        off += c.row_bytes * E
        off = ((off + gran - 1) // gran) * gran
    scratch = {}
    for name in ("scratch_device", "scratch_host"):
        scratch[name] = off
        off += ((scratch_bytes + gran - 1) // gran) * gran
    return {"total_bytes": off, "scratch_offsets": scratch,
            "granule_bytes": sum(c.row_bytes for c in components)}


def runs(mask) -> "list[tuple[int, int, bool]]":
    """Run-length encode a per-expert host mask into (start, count, is_host) so consecutive
    same-media experts share one physical handle."""
    out = []
    i = 0
    while i < len(mask):
        j = i
        while j < len(mask) and mask[j] == mask[i]:
            j += 1
        out.append((i, j - i, bool(mask[i])))
        i = j
    return out


# --------------------------------------------------------------------------------------------
# routes
# --------------------------------------------------------------------------------------------
def build_placement(E: int, top_k: int, R: int, placement: str, rng) -> "list[bool]":
    """True = host-backed. Half the stack on each medium, which is what makes R disjoint routes at
    every miss count constructible."""
    n_host = E // 2
    if placement == "blocked":
        return [e >= (E - n_host) for e in range(E)]
    if placement == "alternate":
        return [bool(e & 1) for e in range(E)]
    if placement != "random":
        raise Precondition(f"unknown --placement {placement!r}")
    idx = list(range(E))
    rng.shuffle(idx)
    mask = [False] * E
    for e in idx[:n_host]:
        mask[e] = True
    return mask


def build_routes(host_mask, top_k: int, miss: int, R: int, M: int, rng, disjoint: bool = True):
    """R routes, each token holding EXACTLY `miss` host-backed experts and `top_k - miss`
    device-backed ones.

    `disjoint=True` (TIMING): no expert is reused anywhere in the rotation, which is what pushes the
    reuse distance past the MALL — without it the timed loop measures Infinity Cache, not media.
    Refuses rather than silently overlapping if the stack is too small.

    `disjoint=False` (CORRECTNESS): experts may repeat across tokens/routes (never within a token).
    Correctness does not care about cache residency, and a 64-token batch at top_k=10 would need
    640 distinct host experts, which no reasonable E provides."""
    host = [e for e, h in enumerate(host_mask) if h]
    dev = [e for e, h in enumerate(host_mask) if not h]
    if miss > len(host) or (top_k - miss) > len(dev):
        raise Precondition(f"miss={miss}/top_k={top_k} needs {miss} host and {top_k - miss} device "
                           f"experts; the stack has {len(host)}/{len(dev)}")
    if not disjoint:
        return [[rng.sample(host, miss) + rng.sample(dev, top_k - miss) for _ in range(M)]
                for _ in range(R)]
    rng.shuffle(host)
    rng.shuffle(dev)
    need_h, need_d = R * M * miss, R * M * (top_k - miss)
    if need_h > len(host) or need_d > len(dev):
        raise Precondition(
            f"cannot build {R} disjoint routes at miss={miss}, M={M}, top_k={top_k}: need "
            f"{need_h} host / {need_d} device experts, have {len(host)}/{len(dev)}. "
            f"Lower --rotation or raise --experts.")
    hi = di = 0
    routes = []
    for _ in range(R):
        rows = []
        for _ in range(M):
            ids = host[hi:hi + miss] + dev[di:di + (top_k - miss)]
            hi += miss
            di += top_k - miss
            rng.shuffle(ids)
            rows.append(ids)
        routes.append(rows)
    return routes


def route_bytes(routes, host_mask, granule_bytes: int):
    """Bytes a single launch must read, computed from the ACTUAL route — never assumed. Returns
    (mean_total, mean_host, mean_device) over the rotation."""
    tot = hostb = devb = 0
    for rows in routes:
        distinct = {e for row in rows for e in row}
        h = sum(1 for e in distinct if host_mask[e])
        tot += len(distinct) * granule_bytes
        hostb += h * granule_bytes
        devb += (len(distinct) - h) * granule_bytes
    n = len(routes)
    return tot / n, hostb / n, devb / n


# --------------------------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------------------------
def linfit(xs, ys):
    n = len(xs)
    if n < 2:
        return None, None, None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return None, None, None
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    icept = my - slope * mx
    sst = sum((y - my) ** 2 for y in ys)
    ssr = sum((y - (slope * x + icept)) ** 2 for x, y in zip(xs, ys))
    r2 = None if sst == 0 else 1.0 - ssr / sst
    return slope, icept, r2


CLIFF_KILL = 3.0        # first miss costs >=3x the average marginal miss -> the h^k regime
CLIFF_LINEAR = 1.5      # within 1.5x of the average marginal miss -> linear enough for T2
MAX_REL_SPREAD = 0.25   # (max-min)/median for any timed arm; above this no shape call is honest
MIN_SNR = 3.0           # t(1)-t(0) must clear this many combined MADs to count as RESOLVED
# Implied host bandwidth above this at m=k means the "host" experts are NOT on the far side of
# PCIe. The measured link ceiling on this box is 28.7 GB/s contiguous / 26.8 GB/s at the granule;
# because a miss REPLACES a device-side read, the implied figure can sit slightly above the raw
# link rate (1/(1/26.8 - 1/700) = 27.9 GB/s) but never near HBM. 64 GB/s is comfortably above any
# legitimate value and comfortably below anything that is really device-resident.
HOST_BW_CEILING_GBPS = 64.0


def analyse_shape(curve: dict, top_k: int, host_bytes_by_m: dict, spreads=None) -> dict:
    """curve: {miss_count: median_us_per_launch}. host_bytes_by_m: {miss_count: MEASURED host bytes
    a single launch must read at that miss count, counted from the actual route}. spreads:
    {miss_count: that arm's `spread()` dict} — supplied on every real run, because `cliff_index` is
    a ratio of two DIFFERENCES and a difference of two noisy medians is reported to two decimal
    places whether or not it is resolvable. The classification bands are therefore applied to the
    noise-propagated INTERVAL [cliff_index_lo, cliff_index_hi], not to the point estimate."""
    ms = sorted(curve)
    sp = spreads or {}
    mad = {m: float((sp.get(m) or {}).get("mad") or 0.0) for m in ms}
    rel = {m: (sp.get(m) or {}).get("rel_spread") for m in ms}
    rels = [v for v in rel.values() if v is not None]
    max_rel = max(rels) if rels else None
    out = {"miss_counts": ms, "median_us": [curve[m] for m in ms],
           "cliff_index": None, "cliff_index_lo": None, "cliff_index_hi": None,
           "first_miss_fraction": None,
           "avg_marginal_us_per_miss": None, "first_marginal_us": None,
           "linear_slope_us_per_miss": None, "linear_intercept_us": None, "linear_r2": None,
           "implied_host_gbps_per_miss": {}, "shape": None,
           "arm_mad_us": mad, "arm_rel_spread": rel, "max_rel_spread": max_rel,
           "noise_ok": None, "first_miss_snr": None, "first_miss_resolved": None,
           "span_snr": None, "host_bw_implausible": None, "host_gbps_at_full_miss": None,
           "spreads_supplied": bool(spreads),
           "thresholds": {"cliff_kill": CLIFF_KILL, "cliff_linear": CLIFF_LINEAR,
                          "max_rel_spread": MAX_REL_SPREAD, "min_snr": MIN_SNR,
                          "host_bw_ceiling_gbps": HOST_BW_CEILING_GBPS}}
    if 0 not in curve or top_k not in curve:
        out["shape"] = "INCOMPLETE"
        return out
    t0, tk = curve[0], curve[top_k]
    span = tk - t0
    slope, icept, r2 = linfit([float(m) for m in ms], [curve[m] for m in ms])
    out["linear_slope_us_per_miss"], out["linear_intercept_us"], out["linear_r2"] = slope, icept, r2
    for m in ms:
        hb = host_bytes_by_m.get(m)
        if m > 0 and curve[m] > t0 and hb:
            out["implied_host_gbps_per_miss"][str(m)] = hb / ((curve[m] - t0) * 1e-6) / 1e9
    hb_k = out["implied_host_gbps_per_miss"].get(str(top_k))
    out["host_gbps_at_full_miss"] = hb_k
    # A "host" page that reads at device speed is the most dangerous failure mode in this probe:
    # the curve stays smooth and plausible while every number in it is about the wrong medium.
    out["host_bw_implausible"] = bool(hb_k is not None and hb_k > HOST_BW_CEILING_GBPS)
    out["noise_ok"] = True if max_rel is None else bool(max_rel <= MAX_REL_SPREAD)
    if span <= 0:
        out["shape"] = "NO_MEASURABLE_MISS_COST"
        return out
    sig_span = mad[0] + mad[top_k]
    out["span_snr"] = (span / sig_span) if sig_span > 0 else None
    avg_marg = span / top_k
    out["avg_marginal_us_per_miss"] = avg_marg
    if 1 not in curve:
        out["shape"] = "INCOMPLETE"
        return out
    first = curve[1] - t0
    sig_first = mad[0] + mad[1]
    out["first_marginal_us"] = first
    out["first_miss_fraction"] = first / span
    out["cliff_index"] = first / avg_marg
    out["first_miss_snr"] = (abs(first) / sig_first) if sig_first > 0 else None
    out["first_miss_resolved"] = bool(sig_first == 0.0 or abs(first) >= MIN_SNR * sig_first)
    lo_den = (span + sig_span) / top_k
    hi_den = (span - sig_span) / top_k
    out["cliff_index_lo"] = ((first - sig_first) / lo_den) if lo_den > 0 else None
    out["cliff_index_hi"] = ((first + sig_first) / hi_den) if hi_den > 0 else float("inf")
    ci, ci_lo, ci_hi = out["cliff_index"], out["cliff_index_lo"], out["cliff_index_hi"]
    if ci_lo is not None and ci_lo >= CLIFF_KILL:
        out["shape"] = "CLIFFED"           # the WHOLE noise interval sits past the kill line
    elif (ci_hi is not None and ci_hi <= CLIFF_LINEAR and r2 is not None and r2 >= 0.95
          and out["noise_ok"] and not out["host_bw_implausible"]):
        out["shape"] = "LINEAR"
    elif ci >= CLIFF_KILL:
        # point estimate past the kill line, interval straddling it: report the LEAN. Auto-killing
        # on a curve this run cannot resolve is a confident wrong answer in the other direction.
        out["shape"] = "CLIFFED_NOISY"
    elif ci <= CLIFF_LINEAR:
        out["shape"] = "LINEAR_NOISY"
    else:
        out["shape"] = "AMBIGUOUS"
    return out


def spread(vals) -> dict:
    v = sorted(vals)
    n = len(v)
    med = statistics.median(v)
    return {
        "n": n, "median": med, "min": v[0], "max": v[-1],
        "p10": v[max(0, int(0.10 * (n - 1)))], "p90": v[min(n - 1, int(0.90 * (n - 1) + 0.5))],
        "mad": statistics.median([abs(x - med) for x in v]),
        "rel_spread": (v[-1] - v[0]) / med if med else None,
        "samples": v,
    }


# --------------------------------------------------------------------------------------------
# result document — ONE builder, used by both the real run and the selftest, so the selftest
# genuinely exercises the shipped shape rather than a hand-written mock.
# --------------------------------------------------------------------------------------------
def build_result(args, envfix, cfg, layout, card, gran, media, preconds, correctness,
                 timing, shapes, diagnostics, box_before, box_after, ledgers, notes,
                 verdict, verdict_reasons, selftest: bool) -> dict:
    return {
        "probe": PROBE,
        "title": "mixed-media grouped MoE GEMM: correctness and the miss-count curve",
        "plan": "docs/WEIGHT_OFFLOAD_PLAN.md §3 P2 / §11 unknown #2",
        "script": SCRIPT_PATH,
        "selftest": selftest,
        "schema_version": 1,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv,
        "args": vars(args),
        "env_fix": envfix,
        "git": {"engine_sha": _git_sha(os.path.dirname(os.path.dirname(
            os.path.dirname(SCRIPT_PATH)))), "engine_dirty": _git_dirty(os.path.dirname(
                os.path.dirname(os.path.dirname(SCRIPT_PATH))))},
        "config": cfg,
        "layout": layout,
        "card": card,
        "vmm_granularity": gran,
        "media_separation": media,
        "preconditions": preconds,
        "correctness": correctness,
        "timing": timing,
        "shape_analysis": shapes,
        "diagnostics": diagnostics,
        "engaged_ledgers": ledgers,
        "box_state_before": box_before,
        "box_state_after": box_after,
        "notes": notes,
        "verdict": verdict,
        "verdict_reasons": verdict_reasons,
        "kill_criterion": (
            "KILL if any correctness arm exceeds its same-binary control floor, OR if "
            f"cliff_index = (t(1)-t(0)) / ((t(k)-t(0))/k) >= {CLIFF_KILL} (the first miss already "
            "costs most of what k misses cost -> layer time is governed by P(all top-k resident) = "
            "h^k, the per-expert device tier buys almost nothing, and only T1 (all-host) or "
            "explicit-copy layer-group streaming survive)."),
    }


REQUIRED_TOP = ("probe", "script", "selftest", "schema_version", "argv", "args", "env_fix", "git",
                "config", "layout", "card", "vmm_granularity", "media_separation", "preconditions",
                "correctness", "timing", "shape_analysis", "diagnostics", "engaged_ledgers",
                "box_state_before", "box_state_after", "notes", "verdict", "verdict_reasons",
                "kill_criterion")


def check_shape(doc: dict) -> "list[str]":
    problems = [f"missing top-level key: {k}" for k in REQUIRED_TOP if k not in doc]
    if not isinstance(doc.get("argv"), list):
        problems.append("argv must be a list")
    if doc.get("verdict") not in ("PASS", "KILL", "AMBIGUOUS", "NOT_MEASURED"):
        problems.append(f"bad verdict {doc.get('verdict')!r}")
    try:
        json.dumps(doc, default=str)
    except (TypeError, ValueError) as exc:
        problems.append(f"not JSON-serialisable: {exc}")
    return problems


def _git_sha(repo: str):
    p = _cmd(["git", "-C", repo, "rev-parse", "HEAD"])
    return (p.get("stdout") or "").strip() or None


def _git_dirty(repo: str):
    p = _cmd(["git", "-C", repo, "status", "--porcelain"])
    out = p.get("stdout")
    return None if out is None else bool(out.strip())


# --------------------------------------------------------------------------------------------
# the arena
# --------------------------------------------------------------------------------------------
class Arena:
    MAX_HANDLES = 8192   # a pathological placement must fail fast, not spend an hour pinning

    def __init__(self, hip: Hip, torch, card: int, components, E: int, host_mask, gran: int,
                 scratch_bytes: int):
        self.hip, self.torch, self.card, self.E = hip, torch, card, E
        self.components = components
        self.host_mask = host_mask
        self.gran = gran
        self.handles: list = []
        self.mapped: list = []          # (ptr, size) for teardown
        self.populated = False
        self.checksums: dict = {}       # per-component reduction, taken right after populate
        self.map_seconds = {_LOC_DEV: 0.0, _LOC_HOST: 0.0}
        self.set_access_seconds = 0.0
        n_runs = len(runs(host_mask))
        if n_runs * len(components) + 2 > self.MAX_HANDLES:
            raise Precondition(
                f"placement needs {n_runs * len(components) + 2} physical handles (> "
                f"{self.MAX_HANDLES}). Use --placement blocked, or lower --experts.")
        t_build = time.perf_counter()
        self.layout = plan_layout(components, E, gran, scratch_bytes)
        total = self.layout["total_bytes"]
        va = ctypes.c_void_p()
        hip.ck(hip.lib.hipMemAddressReserve(ctypes.byref(va), total, 2 << 20, None, 0),
               f"hipMemAddressReserve({total} B)")
        self.base = int(va.value)
        self.total = total
        self.desc = _MemAccessDesc()
        self.desc.location.type = _LOC_DEV
        self.desc.location.id = card
        self.desc.flags = _ACCESS_RW
        self.device_bytes = self.host_bytes = 0
        for c in components:
            c.base = self.base + c.offset
            self._back_component(c)
        self.scratch = {}
        for name, loc in (("scratch_device", _LOC_DEV), ("scratch_host", _LOC_HOST)):
            off = self.layout["scratch_offsets"][name]
            size = ((scratch_bytes + gran - 1) // gran) * gran
            self._map_chunk(self.base + off, size, loc)
            self.scratch[name] = (self.base + off, size)
        self._set_access_once()
        self.tensors = {}
        for c in components:
            t = tensor_at(torch, c.base, (E,) + c.row_shape, self._dt(c.dtype_name), card)
            if not t.is_contiguous():
                raise Precondition(f"arena view for {c.name} is not contiguous")
            if t.data_ptr() != c.base:
                raise Precondition(f"arena view for {c.name} bound at 0x{t.data_ptr():x}, "
                                   f"expected 0x{c.base:x}")
            self.tensors[c.name] = t
        self.build_seconds = time.perf_counter() - t_build

    def _dt(self, name):
        return {"int32": self.torch.int32, "float16": self.torch.float16,
                "bfloat16": self.torch.bfloat16, "float32": self.torch.float32}[name]

    def _map_chunk(self, ptr: int, size: int, loc: int) -> None:
        h = ctypes.c_void_p()
        # location.id is the DEVICE ordinal for device-located memory and the NUMA/host id for
        # host-located memory — both gfx1201 cards sit on NUMA node 0, so host handles take id 0.
        loc_id = self.card if loc == _LOC_DEV else 0
        t0 = time.perf_counter()
        self.hip.ck(self.hip.lib.hipMemCreate(ctypes.byref(h), size,
                                              ctypes.byref(self.hip.prop(loc, loc_id)), 0),
                    f"hipMemCreate({size} B, loc={loc}, id={loc_id})")
        self.handles.append(h)
        self.hip.ck(self.hip.lib.hipMemMap(ctypes.c_void_p(ptr), size, 0, h, 0),
                    f"hipMemMap(0x{ptr:x}, {size} B)")
        self.mapped.append((ptr, size))
        # hipMemSetAccess is DELIBERATELY NOT called here — see _set_access_once().
        self.map_seconds[loc] += time.perf_counter() - t0
        if loc == _LOC_DEV:
            self.device_bytes += size
        else:
            self.host_bytes += size

    def _set_access_once(self) -> None:
        """Enable device access with ONE hipMemSetAccess over the whole mapped span.

        MEASURED DRIVER DEFECT (gfx1201 / ROCm 7.2.1, this box, 2026-09-02 — see
        tools/offload/_p2_diag_flake.py and _p2_diag_fix.py): per-chunk hipMemSetAccess returns
        hipErrorInvalidValue (1) non-deterministically whenever two ADJACENT mappings in the same
        reservation have DIFFERENT sizes. hipMemCreate and hipMemMap both return hipSuccess; only
        hipMemSetAccess rejects, and a retry at the same VA rejects again. Measured rates:
            adjacent EQUAL-sized mappings, per-chunk SetAccess   0 / 3072 chunks failed
            adjacent MIXED-sized mappings, per-chunk SetAccess   ~33% of chunks, both media
            the (dev 1.5 MiB, host 7.5 MiB) pair alone           50 / 100 reservations
            one SetAccess per COMPONENT region (still adjacent)  ~60% of calls
            ONE SetAccess over the whole mapped span             0 / 4096 chunks failed
        P2's layout is inherently mixed-size (w13 rows are 1572864 B, w2_scales rows 24576 B), so
        the per-chunk pattern cannot be used here. Deferring to a single call is semantically
        identical — nothing touches the arena until after this returns — and is the only pattern
        measured to be reliable.

        This is a WORKAROUND, not a fix: it is only available because the layout leaves NO HOLES.
        If a future shape introduces padding the mapped set is no longer one range, and issuing two
        adjacent SetAccess calls would re-enter the defect — so that case ABORTS rather than
        silently flaking."""
        cov = sorted(self.mapped)
        merged = []
        for ptr, size in cov:
            if merged and merged[-1][0] + merged[-1][1] == ptr:
                merged[-1][1] += size
            else:
                merged.append([ptr, size])
        if len(merged) != 1 or merged[0][0] != self.base or merged[0][1] != self.total:
            raise Precondition(
                f"the arena is mapped as {len(merged)} disjoint range(s) covering "
                f"{sum(m[1] for m in merged)} of {self.total} B from 0x{self.base:x}; a single "
                f"hipMemSetAccess cannot cover it, and per-range SetAccess hits the measured "
                f"gfx1201 adjacent-mixed-size defect (~33-60% hipErrorInvalidValue). Make the "
                f"layout hole-free before measuring.")
        t0 = time.perf_counter()
        self.hip.ck(self.hip.lib.hipMemSetAccess(ctypes.c_void_p(self.base), self.total,
                                                 ctypes.byref(self.desc), 1),
                    f"hipMemSetAccess(0x{self.base:x}, {self.total} B) [single-call workaround]")
        self.set_access_seconds = time.perf_counter() - t0

    def _back_component(self, c: Component) -> None:
        for start, count, is_host in runs(self.host_mask):
            self._map_chunk(c.base + start * c.row_bytes, count * c.row_bytes,
                            _LOC_HOST if is_host else _LOC_DEV)

    # ---- self-tests -------------------------------------------------------------------------
    def _marker_slots(self):
        """Assign every component a DISJOINT block of marker values WITHIN ITS OWN DTYPE.

        The earlier version fingerprinted-and-verified one component at a time with the same value
        range in each. That cannot see a CROSS-COMPONENT alias: if `w2`'s region overlapped `w13`'s,
        w13 had already been verified and passed before w2 clobbered it, and w2 would then verify
        against its own freshly written values. Same-dtype components therefore get non-overlapping
        value blocks, everything is written FIRST, and everything is verified AFTERWARDS."""
        torch = self.torch
        exact = {torch.float16: 2048, torch.bfloat16: 256, torch.float32: 1 << 24}
        groups = {}
        for c in self.components:
            groups.setdefault(self._dt(c.dtype_name), []).append(c)
        slots = {}
        for dt, comps in groups.items():
            exact_max = exact.get(dt, 1 << 30)
            need = 2 * len(comps) * self.E          # head + tail marker per component per row
            if need > exact_max:
                raise Precondition(
                    f"cannot fingerprint the {dt} components {[c.name for c in comps]}: they need "
                    f"{need} DISTINCT exactly-representable integers (2 markers x {self.E} rows x "
                    f"{len(comps)} components) but {dt} represents only {exact_max} exactly. "
                    f"Lower --experts.")
            for i, c in enumerate(comps):
                slots[c.name] = (2 * i * self.E, (2 * i + 1) * self.E, exact_max)
        return slots

    def fingerprint_selftest(self) -> dict:
        """PRE-POPULATE ONLY. `hipMemMap` returns hipSuccess on this box even when the page table
        is wrong, so an out-of-band data check is mandatory (plan §5.4 A1.4). Writes a UNIQUE word
        into the first AND last element of every row of every component through the DEVICE VA —
        ALL components first — then reads every one of them back: catches row aliasing (the stale-
        physical-page failure), CROSS-COMPONENT aliasing, unmapped tails, and wrong region offsets.
        Phase-gated so it can never run after weights exist."""
        if self.populated:
            raise Precondition("fingerprint_selftest ran after populate — it would destroy weights")
        torch = self.torch
        slots = self._marker_slots()
        want = {}
        # --- write EVERYTHING first ---------------------------------------------------------
        for c in self.components:
            t = self.tensors[c.name].view(self.E, -1)
            n = t.shape[1]
            h_off, t_off, _ = slots[c.name]
            head = torch.arange(self.E, device=t.device, dtype=torch.int32) + h_off
            tail = torch.arange(self.E, device=t.device, dtype=torch.int32) + t_off
            t[:, 0] = head.to(t.dtype)
            t[:, n - 1] = tail.to(t.dtype)
            want[c.name] = (head, tail)
        torch.cuda.synchronize()
        # --- then verify EVERYTHING ---------------------------------------------------------
        res = {}
        per_dtype_seen = {}
        for c in self.components:
            t = self.tensors[c.name].view(self.E, -1)
            n = t.shape[1]
            head, tail = want[c.name]
            got_h = t[:, 0].to(torch.int32)
            got_t = t[:, n - 1].to(torch.int32)
            ok_h = bool(torch.equal(got_h, head))
            ok_t = bool(torch.equal(got_t, tail))
            uniq = int(torch.unique(got_h).numel())
            res[c.name] = {"head_ok": ok_h, "tail_ok": ok_t, "distinct_head_values": uniq,
                           "expected_distinct": self.E, "dtype": str(t.dtype),
                           "marker_block": slots[c.name][:2]}
            if not (ok_h and ok_t and uniq == self.E):
                raise Precondition(
                    f"arena fingerprint FAILED for component {c.name}: head_ok={ok_h} "
                    f"tail_ok={ok_t} distinct={uniq}/{self.E}. The mapping lied (every hip call "
                    f"returned hipSuccess). Nothing measured after this point would be meaningful.")
            per_dtype_seen.setdefault(str(t.dtype), []).extend([got_h, got_t])
        # --- and finally prove the markers are globally distinct within each dtype -----------
        for dt, tensors in per_dtype_seen.items():
            allv = torch.cat(tensors)
            n_uniq = int(torch.unique(allv).numel())
            res[f"_global_{dt}"] = {"markers": int(allv.numel()), "distinct": n_uniq}
            if n_uniq != int(allv.numel()):
                raise Precondition(
                    f"arena fingerprint FAILED globally for dtype {dt}: {n_uniq} distinct markers "
                    f"out of {int(allv.numel())} written. Two component regions ALIAS each other "
                    f"(every hip call returned hipSuccess). Nothing below would be meaningful.")
        return res

    def _checksums(self) -> dict:
        """A deterministic reduction over every byte of every component, read THROUGH the arena.
        Used to prove after the timed run that nothing clobbered the weights mid-flight — a silent
        mid-run corruption would change what the timed launches were reading without changing a
        single hipSuccess."""
        torch = self.torch
        out = {}
        for c in self.components:
            t = self.tensors[c.name]
            integral = t.dtype in (torch.int32, torch.int64, torch.int8, torch.uint8)
            # CHUNKED: a whole-tensor `.to(torch.int64)` on w13 would materialise ~1.6 GB of
            # transient VRAM next to a 1.2 GB reference stack. Accumulate on the host instead.
            step = max(1, min(self.E, (64 << 20) // max(c.row_bytes, 1)))
            acc = 0 if integral else 0.0
            for e0 in range(0, self.E, step):
                blk = t[e0:min(self.E, e0 + step)]
                if integral:
                    acc += int(blk.to(torch.int64).sum().item())
                else:
                    acc += float(blk.to(torch.float64).sum().item())
            out[c.name] = acc
        torch.cuda.synchronize()
        return out

    def verify_checksums(self) -> dict:
        """Re-read the arena and compare against the checksums taken right after populate."""
        if not self.checksums:
            return {"checked": False, "reason": "no baseline checksums"}
        now = self._checksums()
        bad = {k: {"after_populate": self.checksums[k], "now": now[k]}
               for k in self.checksums if now.get(k) != self.checksums[k]}
        out = {"checked": True, "baseline": self.checksums, "now": now,
               "identical": not bad, "mismatched": bad}
        if bad:
            raise Precondition(
                f"arena content CHANGED during the run: {sorted(bad)}. Every timed number above "
                f"was measured against weights that no longer match what was validated.")
        return out

    def populate_from(self, ref: dict) -> dict:
        """Copy each granule into its VA row THROUGH THE DEVICE POINTER. Never a CPU store into a
        host-located page: the CPU/GPU coherence granularity of those pages is unstated (plan §11
        unknown #6), and a CPU-written page can be stale in device caches."""
        torch = self.torch
        for c in self.components:
            self.tensors[c.name].copy_(ref[c.name])
        torch.cuda.synchronize()
        self.populated = True
        # FULL read-back, every row, every element — not a 64-row sample. The whole premise of this
        # probe is that the mapping API cannot be trusted, so a sampled check leaves 87% of the
        # stack unverified and a wrong page in an unsampled row would surface only as a "correct"
        # number later. Chunked so no single boolean temporary approaches the component size.
        out = {}
        for c in self.components:
            a, r = self.tensors[c.name], ref[c.name]
            mismatch = 0
            step = max(1, min(self.E, (64 << 20) // max(c.row_bytes, 1)))
            for e0 in range(0, self.E, step):
                e1 = min(self.E, e0 + step)
                mismatch += int((a[e0:e1] != r[e0:e1]).sum().item())
            out[c.name] = {"rows_compared": self.E, "elements": int(a.numel()),
                           "mismatching_elements": mismatch, "bitwise_equal": mismatch == 0}
            if mismatch:
                raise Precondition(
                    f"populate read-back MISMATCH for {c.name}: {mismatch}/{a.numel()} elements "
                    f"differ — the arena does not hold the bytes we wrote (every hip call returned "
                    f"hipSuccess).")
        self.checksums = self._checksums()
        return out

    def measure_weight_media_separation(self, min_bytes: int, reps: int = 5) -> dict:
        """Prove the WEIGHT rows themselves straddle two media.

        `measure_media_separation` proves it for the two SCRATCH handles, which are separate
        `hipMemCreate` calls from the ones backing the expert stack — a placement bug confined to
        the component regions would sail straight past it and produce a beautifully flat, entirely
        fictional miss-count curve. This gathers >= `min_bytes` of REAL weight rows from the host
        half and from the device half through identical `index_select` traffic (same write volume,
        so the difference isolates the READ medium) and refuses if they behave the same."""
        torch = self.torch
        c = max(self.components, key=lambda x: x.row_bytes)
        t = self.tensors[c.name].view(self.E, -1)
        n_rows = max(8, -(-int(min_bytes) // c.row_bytes))
        host_ids = [e for e, h in enumerate(self.host_mask) if h]
        dev_ids = [e for e, h in enumerate(self.host_mask) if not h]
        if len(host_ids) < n_rows or len(dev_ids) < n_rows:
            raise Precondition(
                f"weight media separation needs {n_rows} rows of {c.name} on EACH medium to clear "
                f"{min_bytes / 2**20:.0f} MiB; the placement has {len(host_ids)} host / "
                f"{len(dev_ids)} device experts. Raise --experts.")
        out = {"component": c.name, "rows_per_leg": n_rows,
               "bytes_per_leg": n_rows * c.row_bytes}
        buf = torch.empty((n_rows, t.shape[1]), dtype=t.dtype, device=t.device)
        for label, ids in (("device_rows", dev_ids[:n_rows]), ("host_rows", host_ids[:n_rows])):
            idx = torch.tensor(ids, dtype=torch.int64, device=t.device)
            samples = []
            for i in range(reps + 1):
                e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
                torch.cuda.synchronize()
                e0.record()
                torch.index_select(t, 0, idx, out=buf)
                e1.record()
                torch.cuda.synchronize()
                if i:                                  # discard the warm-up
                    samples.append(out["bytes_per_leg"] / (e0.elapsed_time(e1) * 1e-3) / 1e9)
            out[label] = {"gbps": spread(samples)}
        del buf
        d = out["device_rows"]["gbps"]["median"]
        h = out["host_rows"]["gbps"]["median"]
        out["device_over_host"] = d / h if h else None
        out["separated"] = bool(h > 0.5 and d > 2.0 * h)
        if not out["separated"]:
            raise Precondition(
                f"WEIGHT-row media separation FAILED: gathering {n_rows} device-backed {c.name} "
                f"rows ran at {d:.1f} GB/s vs {h:.1f} GB/s for the same number of host-backed rows "
                f"(ratio {d / h if h else float('nan'):.2f}). The expert stack is NOT straddling "
                f"two media, so every miss-count number below would be about one medium.")
        return out

    def measure_media_separation(self, reps: int = 7) -> dict:
        """Prove the two media are actually different. A contiguous 256 MB (4x MALL) read from
        device-backed pages vs the same from host-backed pages, through the SAME mechanism. If
        'host' reads at device speed the placement silently did not take, and every miss-count
        number below would be a flat line for the wrong reason."""
        torch = self.torch
        out = {}
        for name, (ptr, size) in self.scratch.items():
            n = size // 4
            t = tensor_at(torch, ptr, (n,), torch.float32, self.card)
            t.fill_(1.0)
            torch.cuda.synchronize()
            samples = []
            for i in range(reps + 1):
                # Device events, not perf_counter: a per-call synchronize() has a ~40 us wall floor
                # on this box, which is 10% of a 256 MB device-side read.
                e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
                torch.cuda.synchronize()
                e0.record()
                s = t.sum(dtype=torch.float64)
                e1.record()
                torch.cuda.synchronize()
                if i:                       # discard the warm-up
                    samples.append(size / (e0.elapsed_time(e1) * 1e-3) / 1e9)
                if float(s.item()) != float(n):
                    raise Precondition(f"{name}: read-back sum {s.item()} != {n} — the scratch "
                                       f"mapping is wrong")
            out[name] = {"gbps": spread(samples), "bytes": size}
        d = out["scratch_device"]["gbps"]["median"]
        h = out["scratch_host"]["gbps"]["median"]
        out["device_over_host"] = d / h if h else None
        out["separated"] = bool(h > 0.5 and d > 2.0 * h)
        if not out["separated"]:
            raise Precondition(
                f"media separation FAILED: device-backed read {d:.1f} GB/s vs host-backed "
                f"{h:.1f} GB/s (ratio {d / h if h else float('nan'):.2f}). The 'host' pages are "
                f"not behaving like host pages — the mixed-media placement did not take, so a "
                f"flat miss-count curve would be an artifact, not a finding.")
        return out

    def close(self) -> None:
        # clear IN PLACE: callers hold this very dict, and a torch tensor left pointing at an
        # unmapped VA is a segfault waiting for the next touch.
        self.tensors.clear()
        for ptr, size in reversed(self.mapped):
            self.hip.lib.hipMemUnmap(ctypes.c_void_p(ptr), size)
        for h in self.handles:
            self.hip.lib.hipMemRelease(h)
        self.hip.lib.hipMemAddressFree(ctypes.c_void_p(self.base), self.total)


# --------------------------------------------------------------------------------------------
# GPU run
# --------------------------------------------------------------------------------------------
def make_reference(torch, components, E: int, card: int, seed: int) -> dict:
    """Random int4-symmetric expert weights in the op's native layout. Dtype-agnostic on the
    ACTIVATION side; the weight containers are what the w4a8 op defines them to be."""
    g = torch.Generator(device=f"cuda:{card}").manual_seed(seed)
    ref = {}
    for c in components:
        shape = (E,) + c.row_shape
        if c.dtype_name == "int32":
            # Random packed int4 nibbles. The exact bit content is irrelevant to a bit-comparison
            # and to a bandwidth measurement; what matters is that experts differ from each other,
            # so a row-aliasing bug cannot hide behind identical weights.
            ref[c.name] = torch.randint(0, 2 ** 31 - 1, shape, generator=g,
                                        device=f"cuda:{card}", dtype=torch.int32)
        else:
            v = torch.rand(shape, generator=g, device=f"cuda:{card}", dtype=torch.float32)
            ref[c.name] = (v * 0.02 + 0.002).to(torch.float16)
    return ref


def _call(kern, x, w, ids, wts, top_k):
    return kern(x, w["w13"], w["w13_scales"], None, w["w2"], w["w2_scales"], None,
                None, top_k, False, topk_weights=wts, topk_ids=ids)


def time_rotation(torch, kern, weights, xs, id_ts, wt_ts, top_k, reps, warmup_windows,
                  min_window_us=2000.0, eager_reps=3):
    """One timed window = `window` back-to-back replays of a captured graph holding the whole
    rotation. Returns a dict — the caller must never have to remember a tuple order.

    Every window is fenced by device events with a `synchronize()` on both sides, so no host wall
    time is ever wrapped around async work. The eager cross-check is a MEDIAN of `eager_reps`
    passes: a single eager pass is one sample and would be quoted as if it were a measurement."""
    R = len(id_ts)

    def one_pass():
        for i in range(R):
            _call(kern, xs[i], weights, id_ts[i], wt_ts[i], top_k)

    # eager warm-up on a side stream: the FIRST pass carries lazy HIP/kernel-module init and must
    # never reach a recorded sample (torch.empty(pin_memory=True) once measured 200 MB/s this way).
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            one_pass()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    eager_samples = []
    for _ in range(max(1, eager_reps)):
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        torch.cuda.synchronize()
        e0.record()
        one_pass()
        e1.record()
        torch.cuda.synchronize()
        eager_samples.append(e0.elapsed_time(e1) * 1e3 / R)
    eager_sp = spread(eager_samples)
    eager_us = eager_sp["median"]

    graph = pool = None
    mode, err = "graph", None
    try:
        graph = torch.cuda.CUDAGraph()
        pool = torch.cuda.graph_pool_handle()
        with torch.cuda.graph(graph, pool=pool):
            one_pass()
        torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001
        graph = pool = None
        mode, err = "eager", f"{type(exc).__name__}: {exc}"

    window = max(1, int(math.ceil(min_window_us / max(eager_us * R, 1.0))))
    samples = []
    for i in range(reps + warmup_windows):
        torch.cuda.synchronize()
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        for _ in range(window):
            if graph is not None:
                graph.replay()
            else:
                one_pass()
        b.record()
        torch.cuda.synchronize()
        us = a.elapsed_time(b) * 1e3 / (window * R)
        if i >= warmup_windows:      # DISCARD the warm-up window(s)
            samples.append(us)
    del graph, pool
    torch.cuda.synchronize()
    sp = spread(samples)
    med = sp["median"]
    # Graph replay removes launch gaps, so it must be <= eager. Materially SLOWER under replay means
    # the captured graph is not doing what the eager pass did, and the whole curve is suspect.
    ratio = (med / eager_us) if eager_us else None
    return {
        "spread": sp, "mode": mode, "window": window, "capture_error": err,
        "eager_us": eager_us, "eager_spread": eager_sp,
        "graph_over_eager": ratio,
        "crosscheck_ok": bool(ratio is None or ratio <= 1.10),
    }


def run_gpu(args) -> int:
    envfix = fix_env()
    box_before = collect_box_state("before")
    notes, preconds = [], {}

    import random as _random
    rng = _random.Random(args.seed)

    hip = Hip()
    import torch  # noqa: PLC0415 — deliberately after fix_env()
    import torch.utils.dlpack  # noqa: F401,PLC0415

    if not torch.cuda.is_available():
        raise Precondition("torch.cuda.is_available() is False — no HIP GPU visible. ROCm device "
                           "passthrough (--device /dev/kfd --device /dev/dri --group-add video) is "
                           "mandatory or is_rocm() is False.")
    ndev = torch.cuda.device_count()
    if args.card >= ndev:
        raise Precondition(f"--card {args.card} but only {ndev} device(s) visible under "
                           f"ROCR_VISIBLE_DEVICES={os.environ.get('ROCR_VISIBLE_DEVICES')}")
    torch.cuda.set_device(args.card)
    torch.cuda.init()
    props = torch.cuda.get_device_properties(args.card)
    free_b, total_b = torch.cuda.mem_get_info(args.card)
    # ROCm device 2 on this box is the Ryzen iGPU advertising ~47 GB of GTT. ROCR_VISIBLE_DEVICES
    # is supposed to keep it out of enumeration, but that is an ENV VAR — assert on the ENUMERATED
    # DEVICE, never on the setting that was meant to shape it. Both discrete cards are 16 GB.
    if total_b > (32 << 30):
        raise Precondition(
            f"--card {args.card} reports {total_b / 2**30:.1f} GiB of memory. The two discrete "
            f"gfx1201 cards are 16 GiB each; a pool this large is the Ryzen iGPU's GTT aperture, "
            f"which is never a compute target. Check ROCR_VISIBLE_DEVICES / --device passthrough.")
    if total_b < (8 << 30):
        raise Precondition(
            f"--card {args.card} reports only {total_b / 2**30:.1f} GiB — not one of the two "
            f"16 GiB gfx1201 cards this probe is calibrated for.")
    card = {
        "rocr_visible_devices": os.environ.get("ROCR_VISIBLE_DEVICES"),
        "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
        "selected_index": args.card,
        "physical_card": args.card,   # == ROCR index, because we pinned ROCR_VISIBLE_DEVICES=0,1
        "name": props.name,
        "gcn_arch": getattr(props, "gcnArchName", None),
        "wgps_reported_by_torch": props.multi_processor_count,
        "cus": 2 * props.multi_processor_count,
        "total_vram_bytes": int(total_b),
        "free_vram_bytes_at_start": int(free_b),
        "pci_bus_id": hip.pci_bus_id(args.card),
        "torch_version": torch.__version__,
        "all_visible": [{"index": i, "name": torch.cuda.get_device_properties(i).name,
                         "gcn_arch": getattr(torch.cuda.get_device_properties(i), "gcnArchName",
                                             None),
                         "total_bytes": int(torch.cuda.mem_get_info(i)[1]),
                         "pci_bus_id": hip.pci_bus_id(i)} for i in range(ndev)],
    }
    notes.append(f"timed on ROCR device {args.card} = {props.name} @ {card['pci_bus_id']}")

    from minisgl.quant.kernels import w4a8_moe as kern
    import minisgl._hip_engage as HE

    gran_dev = hip.granularity(_LOC_DEV, args.card)
    gran_host = hip.granularity(_LOC_HOST, 0)
    gran = max(gran_dev, gran_host)
    gran_info = {"device": gran_dev, "host": gran_host, "used": gran}

    components = build_components(args.hidden, args.inter, args.group)
    E, k = args.experts, args.top_k
    R = args.rotation if args.rotation > 0 else max(1, (E // 2) // max(k, 1))
    granule = sum(c.row_bytes for c in components)
    reuse_distance = R * k * granule
    preconds["reuse_distance"] = {
        "bytes": reuse_distance, "mall_bytes": MALL_BYTES, "required_multiple": MALL_SAFETY,
        "actual_multiple": reuse_distance / MALL_BYTES,
        "ok": reuse_distance >= MALL_SAFETY * MALL_BYTES,
    }
    if not preconds["reuse_distance"]["ok"]:
        raise Precondition(
            f"rotation reuse distance {reuse_distance / 1e6:.0f} MB < {MALL_SAFETY}x the 64 MB "
            f"MALL. Every timed number would be an Infinity-Cache artifact (this is exactly the "
            f"withdrawn 127.2 GB/s failure). Raise --rotation / --experts.")

    host_mask = build_placement(E, k, R, args.placement, rng)
    miss_counts = sorted({int(v) for v in args.miss_counts.split(",") if v.strip() != ""})
    bad = [m for m in miss_counts if not 0 <= m <= k]
    if bad:
        raise Precondition(f"--miss-counts {bad} outside [0, top_k={k}]")
    # cliff_index is defined ONLY against t(0), t(1) and t(top_k). Without all three the script
    # would run to completion and then report shape=INCOMPLETE — refuse before the GPU work.
    for need in (0, 1, k):
        if need not in miss_counts:
            raise Precondition(
                f"--miss-counts must contain 0, 1 and top_k={k}: cliff_index = "
                f"(t(1)-t(0))/((t(k)-t(0))/k) is undefined without them. Got {miss_counts}.")
    timing_ms = [int(v) for v in args.timing_m.split(",") if v.strip() != ""]
    corr_ms = [int(v) for v in args.correctness_m.split(",") if v.strip() != ""]
    if not timing_ms or not corr_ms:
        raise Precondition("--timing-m and --correctness-m must each name at least one batch size; "
                           "an empty leg would let the verdict read PASS on zero evidence.")

    # FAIL FAST. Every route this run needs is pure integer arithmetic, so construct all of them
    # BEFORE the ~1000 hipMemCreate calls and the 1.2 GB populate. Otherwise an infeasible
    # (M, miss) cell raises deep inside the timing loop, after the correctness leg has already run,
    # and the whole run is thrown away.
    feasibility = []
    for M in timing_ms:
        rd = R * M * k * granule
        feasibility.append({"leg": "timing", "M": M, "reuse_distance_bytes": rd,
                            "mall_multiple": rd / MALL_BYTES})
        if rd < MALL_SAFETY * MALL_BYTES:
            raise Precondition(
                f"timing M={M}: reuse distance {rd / 2**20:.0f} MiB < {MALL_SAFETY}x the 64 MB "
                f"MALL — every number would be an Infinity-Cache artifact.")
        for m in miss_counts:
            build_routes(host_mask, k, m, R, M, _random.Random(args.seed + 1000 * m + M))
    for M in corr_ms:
        for m in sorted({0, min(1, k), k // 2, k} & set(range(k + 1))):
            build_routes(host_mask, k, m, 1, M,
                         _random.Random(args.seed + 100 * m + M), disjoint=False)
    preconds["route_feasibility"] = feasibility

    cfg = {
        "experts": E, "top_k": k, "hidden": args.hidden, "inter": args.inter,
        "group_size": args.group, "act_dtype": args.dtype, "rotation_R": R,
        "reps": args.reps, "warmup_windows": args.warmup, "placement": args.placement,
        "n_host_experts": sum(host_mask), "n_device_experts": E - sum(host_mask),
        "miss_counts": miss_counts, "timing_M": timing_ms, "correctness_M": corr_ms,
        "granule_bytes": granule, "seed": args.seed,
        "stack_bytes_total": granule * E,
        "note": ("granule = w13 + w13_scales + w2 + w2_scales for ONE expert = what must travel "
                 "together (plan §4.4 co-demanded granule)"),
    }

    # Defaults so that a precondition tripping mid-run can still emit an ABORTED artifact holding
    # whatever HAD been established (the correctness leg is expensive and its result is not made
    # worthless by a later timing precondition). Everything not measured stays empty, and the
    # verdict on such a document is NOT_MEASURED — never something that reads like a result.
    arena = None
    layout, media, ledgers, diagnostics, shapes = {}, {}, {}, {}, {}
    arena_device_control = None
    correctness = {"gate": None, "arms": [], "all_pass": None, "ledgers_all_identical": None}
    timing = {"unit": "microseconds per w4a8_moe layer launch", "arms": []}
    try:
        arena = Arena(hip, torch, args.card, components, E, host_mask, gran, args.scratch_bytes)
        layout = dict(arena.layout)
        layout["components"] = [c.as_dict(E) for c in components]
        layout["va_base"] = f"0x{arena.base:x}"
        layout["reserved_bytes"] = arena.total
        layout["mapped_device_bytes"] = arena.device_bytes
        layout["mapped_host_bytes"] = arena.host_bytes
        layout["handles"] = len(arena.handles)
        layout["build_seconds"] = arena.build_seconds
        layout["map_seconds_device"] = arena.map_seconds[_LOC_DEV]
        layout["map_seconds_host"] = arena.map_seconds[_LOC_HOST]
        layout["host_map_gbps"] = (arena.host_bytes / arena.map_seconds[_LOC_HOST] / 1e9
                                   if arena.map_seconds[_LOC_HOST] else None)
        layout["set_access_seconds"] = arena.set_access_seconds
        layout["set_access_pattern"] = (
            "ONE hipMemSetAccess over the whole mapped span. Per-chunk SetAccess is UNUSABLE on "
            "this box: with adjacent mappings of DIFFERENT sizes it returns hipErrorInvalidValue "
            "~33% of the time (per-component ~60%, the minimal mixed pair 50/100), while "
            "hipMemCreate/hipMemMap both report success. Measured 2026-09-02 by "
            "tools/offload/_p2_diag_flake.py + _p2_diag_fix.py; see Arena._set_access_once().")

        preconds["fingerprint"] = arena.fingerprint_selftest()
        media = arena.measure_media_separation(reps=max(5, args.reps))

        ref = make_reference(torch, components, E, args.card, args.seed)
        preconds["populate_readback"] = arena.populate_from(ref)
        # …and prove the WEIGHT pages (not just the two scratch handles) straddle two media.
        preconds["weight_media_separation"] = arena.measure_weight_media_separation(
            min_bytes=MALL_SAFETY * MALL_BYTES, reps=max(5, args.reps))

        act_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
        dev = f"cuda:{args.card}"

        def make_route_tensors(routes):
            ids, wts, xs = [], [], []
            gen = torch.Generator(device="cpu").manual_seed(args.seed + 1)
            for rows in routes:
                M = len(rows)
                ids.append(torch.tensor(rows, dtype=torch.int32, device=dev))
                w = torch.rand((M, k), generator=gen).abs() + 0.05
                wts.append((w / w.sum(dim=1, keepdim=True)).to(torch.float32).to(dev))
                xs.append((torch.randn(M, args.hidden, generator=gen) * 0.05).to(act_dtype).to(dev))
            return ids, wts, xs

        # ---- CORRECTNESS FIRST ---------------------------------------------------------------
        correctness = {"gate": (f"max|arena - ref| <= ctl, where ctl is the MAX over "
                                f"{args.control_reps} SAME-CONFIG reference-vs-reference re-runs "
                                f"measured in this very run; where ctl == 0 the gate is exact "
                                f"bit-equality. A ctl above {args.max_control_rel:.1%} of the "
                                f"reference absmax is flagged control_suspicious and blocks PASS, "
                                f"because a gate that loose would swallow a real corruption."),
                       "arms": []}
        arena_w = arena.tensors
        ledgers = {}
        for M in corr_ms:
            for m in sorted({0, min(1, k), k // 2, k} & set(range(k + 1))):
                # disjoint=False: correctness does not care about cache residency, and a 64-token
                # batch at top_k=10 would otherwise demand 640 distinct host experts.
                routes = build_routes(host_mask, k, m, 1, M,
                                      _random.Random(args.seed + 100 * m + M), disjoint=False)
                ids, wts, xs = make_route_tensors(routes)
                HE._seen.clear()
                a0 = _call(kern, xs[0], ref, ids[0], wts[0], k)
                led_ref = sorted(HE._seen)
                # The control floor is the reference disagreeing with ITSELF. ONE re-run is a single
                # draw from that distribution: if the two happen to land bit-identical the gate
                # silently hardens to exact equality and a legitimate atomic-order difference in the
                # arena leg reads as corruption. Take the MAX over several draws.
                ctl_samples = []
                for _ in range(max(1, args.control_reps)):
                    ctl_t = _call(kern, xs[0], ref, ids[0], wts[0], k)
                    ctl_samples.append(float((ctl_t - a0).abs().max().item()))
                    del ctl_t
                ctl = max(ctl_samples)
                HE._seen.clear()
                b0 = _call(kern, xs[0], arena_w, ids[0], wts[0], k)
                led_arena = sorted(HE._seen)
                delta = float((b0 - a0).abs().max().item())
                ref_absmax = float(a0.abs().max().item())
                # A gate on max|delta| is VACUOUS if the reference output is identically zero — the
                # arena could be all zeroes and still "pass". Refuse rather than report.
                if not (ref_absmax > 0.0) or not math.isfinite(ref_absmax):
                    raise Precondition(
                        f"reference output at M={M}, miss={m} has max|out| = {ref_absmax}; a "
                        f"bit-comparison against it would be vacuous. Check the synthetic weights.")
                arena_absmax = float(b0.abs().max().item())
                if not (arena_absmax > 0.0) or not math.isfinite(arena_absmax):
                    raise Precondition(
                        f"arena output at M={M}, miss={m} has max|out| = {arena_absmax} — the "
                        f"mixed-media launch produced an all-zero/non-finite tensor. A max|delta| "
                        f"gate cannot be trusted next to that.")
                ok = (delta == 0.0) if ctl == 0.0 else (delta <= ctl)
                # A control floor that is itself a large fraction of the signal makes the gate
                # meaningless — it would swallow a real, whole-expert corruption. Flag, do not pass.
                ctl_rel = ctl / ref_absmax if ref_absmax else None
                ctl_suspicious = bool(ctl_rel is not None and ctl_rel > args.max_control_rel)
                ledgers[f"M{M}_m{m}"] = {"reference": led_ref, "arena": led_arena,
                                         "identical": led_ref == led_arena}
                correctness["arms"].append({
                    "M": M, "miss": m, "control_floor": ctl, "control_samples": ctl_samples,
                    "control_floor_relative": ctl_rel,
                    "control_suspicious": ctl_suspicious,
                    "max_abs_delta": delta,
                    "reference_absmax": ref_absmax, "arena_absmax": arena_absmax,
                    "relative_delta": (delta / ref_absmax) if ref_absmax else None,
                    "exact": delta == 0.0, "pass": bool(ok),
                    "out_dtype": str(a0.dtype), "out_shape": list(a0.shape),
                    "ledger_identical": led_ref == led_arena,
                })
                del a0, b0
                torch.cuda.empty_cache()
        correctness["all_pass"] = all(a["pass"] for a in correctness["arms"])
        correctness["ledgers_all_identical"] = all(a["ledger_identical"]
                                                   for a in correctness["arms"])
        correctness["any_control_suspicious"] = any(a["control_suspicious"]
                                                    for a in correctness["arms"])
        correctness["arms_measured"] = len(correctness["arms"])

        # ---- ARENA-vs-TORCH DEVICE CONTROL, at m=0, while `ref` still exists -------------------
        # t(0) is the denominator of every number in this probe. If the arena's DEVICE-backed pages
        # are themselves slower than ordinary VRAM (1000-odd separate physical handles is a lot of
        # page-table pressure), t(0) is inflated, the span shrinks, and cliff_index is distorted —
        # invisibly, because there is nothing in the miss-count sweep to compare it against. So time
        # the SAME m=0 rotation against a plain torch-allocated stack holding the same bytes.
        M0 = timing_ms[0]
        r0 = build_routes(host_mask, k, 0, R, M0, _random.Random(args.seed + 1000 * 0 + M0))
        i0, w0, x0 = make_route_tensors(r0)
        t_arena = time_rotation(torch, kern, arena_w, x0, i0, w0, k, args.reps, args.warmup)
        t_ref = time_rotation(torch, kern, ref, x0, i0, w0, k, args.reps, args.warmup)
        ratio0 = (t_arena["spread"]["median"] / t_ref["spread"]["median"]
                  if t_ref["spread"]["median"] else None)
        arena_device_control = {
            "what": ("m=0 (all routed experts device-resident) timed over the arena's VMM-mapped "
                     "device pages vs a plain torch-allocated stack with the same bytes and the "
                     "same route. ~1.0 means the arena's device tier behaves like ordinary VRAM; a "
                     "ratio well above 1 means t(0) — the baseline of the whole curve — carries "
                     "arena overhead that has nothing to do with host pages."),
            "arena_us": t_arena["spread"]["median"], "arena_spread": t_arena["spread"],
            "torch_us": t_ref["spread"]["median"], "torch_spread": t_ref["spread"],
            "arena_over_torch": ratio0,
            "arena_mode": t_arena["mode"], "torch_mode": t_ref["mode"],
            "mode_consistent": t_arena["mode"] == t_ref["mode"],
            "tolerance": args.max_arena_overhead,
            # A ratio between a graph-replayed leg and an eager leg is a launch-overhead ratio, not
            # a page-table ratio — it must not be allowed to certify the baseline.
            "ok": bool(ratio0 is not None and ratio0 <= args.max_arena_overhead
                       and t_arena["mode"] == t_ref["mode"]),
        }
        del i0, w0, x0

        # The reference stack is only needed for correctness and the m=0 control; drop it so the
        # timing legs are not competing for VRAM with 1.2 GB of duplicate weights.
        del ref
        torch.cuda.empty_cache()

        # ---- THE MISS-COUNT CURVE ------------------------------------------------------------
        timing = {"unit": "microseconds per w4a8_moe layer launch", "arms": []}
        shapes = {}
        for M in timing_ms:
            curve, host_bytes_by_m, spreads_by_m = {}, {}, {}
            for m in miss_counts:
                routes = build_routes(host_mask, k, m, R, M,
                                      _random.Random(args.seed + 1000 * m + M))
                ids, wts, xs = make_route_tensors(routes)
                tot_b, host_b, dev_b = route_bytes(routes, host_mask, granule)
                t = time_rotation(torch, kern, arena_w, xs, ids, wts, k, args.reps, args.warmup)
                sp = t["spread"]
                med = sp["median"]
                timing["arms"].append({
                    "M": M, "miss": m, "rotation_R": R, "mode": t["mode"],
                    "replays_per_window": t["window"], "capture_error": t["capture_error"],
                    "us_per_launch": sp,
                    "eager_us_per_launch_crosscheck": t["eager_us"],
                    "eager_spread": t["eager_spread"],
                    "graph_over_eager": t["graph_over_eager"],
                    "crosscheck_ok": t["crosscheck_ok"],
                    "bytes_per_launch": tot_b, "host_bytes_per_launch": host_b,
                    "device_bytes_per_launch": dev_b,
                    "effective_gbps": tot_b / (med * 1e-6) / 1e9 if med else None,
                })
                curve[m] = med
                host_bytes_by_m[m] = host_b
                spreads_by_m[m] = sp
                del ids, wts, xs
                torch.cuda.empty_cache()
            shapes[f"M={M}"] = analyse_shape(curve, k, host_bytes_by_m, spreads_by_m)

        # Every point on a curve must have been produced the SAME way. A mix of graph-replay and
        # eager arms makes the differences that cliff_index is built from partly a launch-overhead
        # difference, and nothing downstream would show it.
        modes = sorted({a["mode"] for a in timing["arms"]})
        timing["modes"] = modes
        timing["mode_consistent"] = len(modes) <= 1
        timing["all_crosschecks_ok"] = all(a["crosscheck_ok"] for a in timing["arms"])

        # ---- diagnostics ---------------------------------------------------------------------
        diagnostics = {"arena_device_control": arena_device_control}
        routes1 = build_routes(host_mask, k, k, 1, M0, _random.Random(args.seed + 7))
        ids1, wts1, xs1 = make_route_tensors(routes1)
        t1 = time_rotation(torch, kern, arena_w, xs1, ids1, wts1, k, args.reps, args.warmup)
        sp1 = t1["spread"]
        rot_med = next((a["us_per_launch"]["median"] for a in timing["arms"]
                        if a["M"] == M0 and a["miss"] == k), None)
        diagnostics["cache_sensitivity"] = {
            "what": ("all-host route (m=k) timed with R=1 (fully MALL-resident) vs the R=%d "
                     "disjoint rotation. A large ratio proves the rotation is load-bearing; ~1.0 "
                     "means host reads are not being cached and the rotation is belt-and-braces."
                     % R),
            "R1_us": sp1["median"], "R1_mode": t1["mode"], "rotation_us": rot_med,
            "rotation_over_R1": (rot_med / sp1["median"]) if (rot_med and sp1["median"]) else None,
        }
        del ids1, wts1, xs1
        torch.cuda.empty_cache()
        diagnostics["vram_after"] = {"free_bytes": int(torch.cuda.mem_get_info(args.card)[0]),
                                     "torch_allocated": int(torch.cuda.memory_allocated()),
                                     "torch_reserved": int(torch.cuda.memory_reserved())}

        # The arena must still hold the bytes it was validated with. A mid-run clobber (a stray
        # write through a stale mapping, a graph pool landing on top of the reservation) would not
        # raise anything — it would just quietly change what the timed launches were reading.
        preconds["arena_integrity_after_timing"] = arena.verify_checksums()

        # ---- verdict -------------------------------------------------------------------------
        reasons = []
        # Nothing measured can never read PASS.
        if not correctness["arms"] or not timing["arms"] or not shapes:
            reasons.append("NOT_MEASURED: a leg produced no arms; there is nothing to conclude")
            not_measured = True
        else:
            not_measured = False
        if correctness["arms"] and not correctness["all_pass"]:
            reasons.append("KILL: a mixed-media launch disagreed with the all-device reference "
                           "beyond the same-binary control floor")
        if correctness["arms"] and not correctness["ledgers_all_identical"]:
            reasons.append("KILL: the arena leg engaged a different kernel set than the reference "
                           "leg — a dispatch divergence makes both numbers uninterpretable")
        cliffed = [key for key, s in shapes.items() if s.get("shape") == "CLIFFED"]
        ambiguous = [key for key, s in shapes.items()
                     if s.get("shape") in ("AMBIGUOUS", "INCOMPLETE", "NO_MEASURABLE_MISS_COST",
                                           "LINEAR_NOISY", "CLIFFED_NOISY")]
        if cliffed:
            reasons.append(f"KILL: superlinear miss cliff at {cliffed} "
                           f"(cliff_index lower bound >= {CLIFF_KILL}) — layer time is governed by "
                           f"P(all top-k resident); the per-expert device tier (T2) buys almost "
                           f"nothing. Fall back to layer-granular placement or T1.")
        # Everything below invalidates the CURVE without invalidating the correctness result: they
        # block PASS but must never be dressed up as a measured KILL.
        blockers = []
        if correctness.get("any_control_suspicious"):
            blockers.append(
                f"the reference-vs-reference control floor exceeds {args.max_control_rel:.1%} of "
                f"the reference absmax on at least one arm — a gate that loose could swallow a "
                f"whole-expert corruption")
        if not timing.get("mode_consistent", True):
            blockers.append(f"the timed arms did not all use the same execution mode "
                            f"({timing.get('modes')}); part of the miss-count difference is a "
                            f"launch-overhead difference")
        if not timing.get("all_crosschecks_ok", True):
            blockers.append("graph replay was materially SLOWER than the eager cross-check on at "
                            "least one arm — the captured graph is not doing what eager did")
        if not arena_device_control["ok"]:
            blockers.append(
                f"the arena's device-backed pages ran "
                f"{_fmt(arena_device_control['arena_over_torch'], 2)}x a plain torch stack at m=0 "
                f"(tolerance {args.max_arena_overhead}); t(0) — the baseline of the whole curve — "
                f"carries arena overhead unrelated to host pages")
        implaus = [key for key, s in shapes.items() if s.get("host_bw_implausible")]
        if implaus:
            blockers.append(
                f"implied host bandwidth at m=top_k exceeds {HOST_BW_CEILING_GBPS} GB/s at "
                f"{implaus} — far above the {28.7} GB/s link ceiling measured on this box, so the "
                f"'host' experts were not being read over PCIe")
        unresolved = [key for key, s in shapes.items() if s.get("first_miss_resolved") is False]
        if unresolved:
            blockers.append(f"t(1)-t(0) is under {MIN_SNR} combined MADs at {unresolved}: the FIRST "
                            f"miss cost is not resolvable, so cliff_index is not a measurement")
        reasons.extend(f"BLOCKS PASS: {b}" for b in blockers)

        if any(r.startswith("KILL:") for r in reasons):
            verdict = "KILL"          # a measured failure outranks an incomplete leg
        elif not_measured:
            verdict = "NOT_MEASURED"
        elif ambiguous or blockers:
            verdict = "AMBIGUOUS"
        else:
            verdict = "PASS"
        if verdict == "AMBIGUOUS" and ambiguous:
            reasons.append(f"shape is not cleanly linear at {ambiguous}; read the curve, do not "
                           f"auto-promote to T2")
        if verdict == "PASS":
            reasons.append("correct on every arm, and per-layer time is linear in miss count — "
                           "per-expert placement (T2) is admissible")

        box_after = collect_box_state("after")
        doc = build_result(args, envfix, cfg, layout, card, gran_info, media, preconds,
                           correctness, timing, shapes, diagnostics, box_before, box_after,
                           ledgers, notes, verdict, reasons, selftest=False)
    except Precondition as exc:
        # Preserve what WAS established, clearly flagged, under a separate basename so it can never
        # be mistaken for (or overwrite) a completed run.
        notes.append(f"ABORTED by a precondition: {exc}")
        if arena_device_control is not None:
            diagnostics.setdefault("arena_device_control", arena_device_control)
        try:
            partial = build_result(
                args, envfix, cfg, layout, card, gran_info, media, preconds, correctness, timing,
                shapes, diagnostics, box_before, collect_box_state("aborted"), ledgers, notes,
                "NOT_MEASURED",
                [f"ABORTED: {exc}",
                 "Sections present below were established BEFORE the abort and are real; every "
                 "section that is empty was never measured. This document is not a result."],
                selftest=False)
            partial["aborted"] = True
            partial["abort_reason"] = str(exc)
            emit(partial, args, basename="p2.aborted")
        except Exception as inner:  # noqa: BLE001 — never mask the original precondition
            print(f"[p2] could not write the aborted-run artifact: "
                  f"{type(inner).__name__}: {inner}", file=sys.stderr, flush=True)
        raise
    finally:
        if arena is not None:
            try:
                arena.close()
            except Exception as exc:  # noqa: BLE001
                notes.append(f"arena teardown: {type(exc).__name__}: {exc}")

    emit(doc, args, basename="p2")
    return 3 if (verdict == "KILL" and args.fail_on_kill) else 0


# --------------------------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------------------------
def _relax(path: str) -> None:
    """The probe runs as root inside the serve image but its artifacts live in a pat-owned
    worktree. A root-owned, root-writable result file is exactly the trap that made a previous
    root-owned snapshot look installed when it was not — make every artifact group/other writable
    so the next run (or the host user) can replace it."""
    try:
        os.chmod(path, 0o666)
    except OSError:
        pass


def _fmt(v, n=1):
    return "n/a" if v is None else f"{v:.{n}f}"


def to_markdown(doc: dict) -> str:
    c, L = doc["config"], doc["layout"]
    card = doc["card"]
    lines = [f"# P2 — mixed-media grouped MoE GEMM ({doc['verdict']})", ""]
    if doc["selftest"]:
        lines += ["> **SELFTEST RUN — nothing was measured.** No number below is real.", ""]
    lines += [
        f"- probe: `{doc['probe']}` · script `{doc['script']}`",
        f"- engine sha: `{(doc['git'].get('engine_sha') or '?')[:12]}` "
        f"(dirty={doc['git'].get('engine_dirty')})",
        f"- started: {doc['started_utc']}",
        "",
        "## Verdict",
        "",
        f"**{doc['verdict']}**",
        "",
    ] + [f"- {r}" for r in doc["verdict_reasons"]] + [
        "",
        f"Kill criterion: {doc['kill_criterion']}",
        "",
        "## Card / box",
        "",
        f"- physical card (ROCR index): **{card.get('physical_card')}** — {card.get('name')} "
        f"@ `{card.get('pci_bus_id')}`, {card.get('cus')} CUs, "
        f"{(card.get('total_vram_bytes') or 0) / 2**30:.1f} GiB VRAM",
        f"- ROCR_VISIBLE_DEVICES=`{card.get('rocr_visible_devices')}` "
        f"HIP_VISIBLE_DEVICES=`{card.get('hip_visible_devices')}`",
    ]
    mb = doc["box_state_before"].get("meminfo_kb", {})
    ma = doc["box_state_after"].get("meminfo_kb", {}) if doc["box_state_after"] else {}
    vb = doc["box_state_before"].get("vmstat", {})
    va = doc["box_state_after"].get("vmstat", {}) if doc["box_state_after"] else {}

    def _gb(kb):
        return "n/a" if kb is None else f"{kb / 2**20:.1f}"
    lines += [
        f"- MemAvailable before/after: {_gb(mb.get('MemAvailable'))} / "
        f"{_gb(ma.get('MemAvailable'))} GiB · MemFree {_gb(mb.get('MemFree'))} / "
        f"{_gb(ma.get('MemFree'))} GiB",
        f"- pswpout before/after: {vb.get('pswpout')} / {va.get('pswpout')} "
        f"(a rise means the host tier pushed the box into swap — read every timing below in that "
        f"light)",
        f"- loadavg at start: `{doc['box_state_before'].get('loadavg')}`",
        "",
        "## Configuration",
        "",
        f"- E={c['experts']} top_k={c['top_k']} hidden={c['hidden']} inter={c['inter']} "
        f"group={c['group_size']} act={c['act_dtype']}",
        f"- granule (w13+scales+w2+scales, one expert) = {c['granule_bytes'] / 2**20:.2f} MiB; "
        f"stack = {c['stack_bytes_total'] / 2**30:.2f} GiB",
        f"- placement `{c['placement']}` → {c['n_device_experts']} device / {c['n_host_experts']} "
        f"host experts; VMM granularity {doc['vmm_granularity'].get('used')} B",
        f"- rotation R={c['rotation_R']} disjoint routes → reuse distance "
        f"{doc['preconditions']['reuse_distance']['bytes'] / 2**20:.0f} MiB "
        f"({doc['preconditions']['reuse_distance']['actual_multiple']:.1f}x MALL)",
        f"- arena: {L.get('mapped_device_bytes', 0) / 2**20:.0f} MiB device-backed + "
        f"{L.get('mapped_host_bytes', 0) / 2**20:.0f} MiB host-backed over "
        f"{L.get('handles')} handles at VA {L.get('va_base')}",
        "",
        "## Media separation (precondition)",
        "",
    ]
    med = doc.get("media_separation") or {}
    if med:
        lines += [
            f"- SCRATCH, 256 MiB contiguous read: device-backed "
            f"**{_fmt((med.get('scratch_device') or {}).get('gbps', {}).get('median'))} GB/s** vs "
            f"host-backed **{_fmt((med.get('scratch_host') or {}).get('gbps', {}).get('median'))} "
            f"GB/s** (ratio {_fmt(med.get('device_over_host'), 2)})",
        ]
    wms = (doc.get("preconditions") or {}).get("weight_media_separation") or {}
    if wms:
        lines += [
            f"- WEIGHT ROWS, {wms.get('rows_per_leg')} x `{wms.get('component')}` "
            f"({(wms.get('bytes_per_leg') or 0) / 2**20:.0f} MiB) gathered: device-backed "
            f"**{_fmt((wms.get('device_rows') or {}).get('gbps', {}).get('median'))} GB/s** vs "
            f"host-backed **{_fmt((wms.get('host_rows') or {}).get('gbps', {}).get('median'))} "
            f"GB/s** (ratio {_fmt(wms.get('device_over_host'), 2)}) — this is the check that the "
            f"EXPERT STACK, not just the scratch handles, straddles two media",
        ]
    integ = (doc.get("preconditions") or {}).get("arena_integrity_after_timing") or {}
    if integ:
        lines += [f"- arena content unchanged across the whole timed run: "
                  f"**{integ.get('identical')}**"]
    lines.append("")
    lines += ["## Correctness", ""]
    if doc["correctness"].get("arms"):
        lines += ["| M | miss | control floor | rel | max abs delta | pass | ledger identical |",
                  "|---|---|---|---|---|---|---|"]
        for a in doc["correctness"]["arms"]:
            rel = a.get("control_floor_relative")
            relf = "n/a" if rel is None else f"{rel:.2e}"
            if a.get("control_suspicious"):
                relf = f"**{relf}**"
            lines.append(f"| {a['M']} | {a['miss']} | {a['control_floor']:.3e} | {relf} | "
                         f"{a['max_abs_delta']:.3e} | {'yes' if a['pass'] else '**NO**'} | "
                         f"{'yes' if a['ledger_identical'] else '**NO**'} |")
        lines.append("")
    lines += ["## Miss-count curve", ""]
    for key, s in (doc.get("shape_analysis") or {}).items():
        lines += [f"### {key}", "",
                  "| misses of top-k | median us/launch | spread (min–max) | host MB/launch | "
                  "effective GB/s |", "|---|---|---|---|---|"]
        for a in doc["timing"]["arms"]:
            if f"M={a['M']}" != key:
                continue
            sp = a["us_per_launch"]
            lines.append(f"| {a['miss']} | {sp['median']:.1f} | {sp['min']:.1f}–{sp['max']:.1f} | "
                         f"{a['host_bytes_per_launch'] / 2**20:.1f} | "
                         f"{_fmt(a.get('effective_gbps'), 1)} |")
        lines += [
            "",
            f"- **shape = {s.get('shape')}**, cliff_index = {_fmt(s.get('cliff_index'), 2)} "
            f"(noise interval {_fmt(s.get('cliff_index_lo'), 2)}–{_fmt(s.get('cliff_index_hi'), 2)}; "
            f"1.0 = linear, {c['top_k']} = full cliff; KILL when the LOWER bound >= {CLIFF_KILL})",
            f"- first miss costs {_fmt((s.get('first_miss_fraction') or 0) * 100, 1)}% of what "
            f"{c['top_k']} misses cost (linear would be "
            f"{100.0 / c['top_k']:.1f}%); resolved = {s.get('first_miss_resolved')} "
            f"(SNR {_fmt(s.get('first_miss_snr'), 1)}, need >= {MIN_SNR})",
            f"- noise: max per-arm rel spread {_fmt(s.get('max_rel_spread'), 3)} "
            f"(limit {MAX_REL_SPREAD}) → noise_ok = {s.get('noise_ok')}; span SNR "
            f"{_fmt(s.get('span_snr'), 1)}",
            f"- linear fit: {_fmt(s.get('linear_slope_us_per_miss'), 2)} us/miss, "
            f"R² = {_fmt(s.get('linear_r2'), 4)}",
            f"- implied host bandwidth per miss count: "
            + ", ".join(f"m={m}: {v:.1f} GB/s"
                        for m, v in (s.get("implied_host_gbps_per_miss") or {}).items())
            + f" · at m=top_k implausible (> {HOST_BW_CEILING_GBPS} GB/s) = "
              f"{s.get('host_bw_implausible')}",
            "",
        ]
    diag = doc.get("diagnostics") or {}
    if diag:
        lines += ["## Diagnostics", ""]
    adc = diag.get("arena_device_control")
    if adc:
        lines += [f"- arena-vs-torch device control at m=0: arena {_fmt(adc.get('arena_us'))} us vs "
                  f"plain torch stack {_fmt(adc.get('torch_us'))} us → ratio "
                  f"{_fmt(adc.get('arena_over_torch'), 3)} (tolerance {adc.get('tolerance')}, "
                  f"ok = {adc.get('ok')}). This is whether t(0), the baseline of the whole curve, "
                  f"is an ordinary-VRAM number."]
    if diag.get("cache_sensitivity"):
        cs = diag["cache_sensitivity"]
        lines += [f"- cache sensitivity (m=k): R=1 {_fmt(cs.get('R1_us'))} us vs rotation "
                  f"{_fmt(cs.get('rotation_us'))} us → ratio "
                  f"{_fmt(cs.get('rotation_over_R1'), 2)}"]
    tm = doc.get("timing") or {}
    if tm.get("modes") is not None:
        lines += [f"- execution modes across the timed arms: {tm.get('modes')} "
                  f"(consistent = {tm.get('mode_consistent')}); graph-vs-eager cross-checks all ok "
                  f"= {tm.get('all_crosschecks_ok')}"]
    lines.append("")
    lines += ["## Notes", ""] + [f"- {n}" for n in doc.get("notes", [])] + [""]
    return "\n".join(lines)


def emit(doc: dict, args, basename: str) -> None:
    """Write the raw JSON FIRST. A measured result must never be lost to a schema nit or a
    formatting bug in the renderer — those get reported loudly, not by discarding the data."""
    os.makedirs(args.out_dir, exist_ok=True)
    tag = f".{args.tag}" if args.tag else ""
    jpath = os.path.join(args.out_dir, f"{basename}{tag}.json")
    mpath = os.path.join(args.out_dir, f"{basename}{tag}.md")

    def _write_json():
        with open(jpath, "w") as fh:
            json.dump(doc, fh, indent=2, sort_keys=False, default=str)
            fh.write("\n")
        _relax(jpath)

    _write_json()
    problems = check_shape(doc)
    if problems:
        doc["schema_problems"] = problems
        _write_json()
        print("[p2] WARNING: the result document failed its own schema check: "
              + "; ".join(problems), file=sys.stderr, flush=True)
    try:
        with open(mpath, "w") as fh:
            fh.write(to_markdown(doc))
        _relax(mpath)
    except Exception as exc:  # noqa: BLE001
        doc["markdown_error"] = f"{type(exc).__name__}: {exc}"
        _write_json()
        print(f"[p2] WARNING: markdown rendering failed ({exc}); the JSON at {jpath} is complete",
              file=sys.stderr, flush=True)
    print(json.dumps(doc, indent=2, default=str), flush=True)
    print(f"\n[p2] verdict: {doc['verdict']}", flush=True)
    for r in doc["verdict_reasons"]:
        print(f"[p2]   {r}", flush=True)
    print(f"[p2] wrote {jpath}", flush=True)
    print(f"[p2] wrote {mpath}", flush=True)


# --------------------------------------------------------------------------------------------
# selftest — no GPU, no torch
# --------------------------------------------------------------------------------------------
def dlpack_cpu_check():
    """Exercise the hand-built DLPack capsule — struct layout, capsule name lifetime, stride
    computation, non-owning aliasing, and the no-op deleter trampoline — against a plain ctypes
    buffer on the CPU. Same code path as the device binding minus the device, so it can run in the
    image with NO GPU attached. Returns (pass|None-if-skipped, detail)."""
    try:
        import torch
        import torch.utils.dlpack  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return None, f"skipped: torch not importable ({type(exc).__name__}: {exc})"
    try:
        import gc
        buf = (ctypes.c_float * 12)(*[float(i) for i in range(12)])
        ptr = ctypes.addressof(buf)
        t = torch.utils.dlpack.from_dlpack(_make_capsule(torch, ptr, (3, 4), torch.float32,
                                                         0, _DL_CPU))
        facts = {
            "shape": tuple(t.shape) == (3, 4),
            "strides": tuple(t.stride()) == (4, 1),
            "dtype": t.dtype == torch.float32,
            "ptr": t.data_ptr() == ptr,
            "contiguous": t.is_contiguous(),
            "values": t.flatten().tolist() == [float(i) for i in range(12)],
        }
        buf[5] = 99.0                       # non-owning view: the tensor must SEE this
        facts["aliases_source"] = float(t[1, 1].item()) == 99.0
        i32 = torch.utils.dlpack.from_dlpack(
            _make_capsule(torch, ptr, (12,), torch.int32, 0, _DL_CPU))
        facts["int32_dtype"] = i32.dtype == torch.int32
        del t, i32
        gc.collect()                        # runs the no-op deleter; must not crash or free `buf`
        facts["survives_deleter"] = float(buf[5]) == 99.0
        bad = [k for k, v in facts.items() if not v]
        return (not bad), (f"failed: {bad}" if bad else "all DLPack facts hold")
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def run_selftest(args) -> int:
    checks = []

    def chk(name, cond, detail=""):
        if cond is None:
            checks.append({"check": name, "pass": None, "skipped": True, "detail": str(detail)})
            return True
        checks.append({"check": name, "pass": bool(cond), "detail": str(detail)})
        return bool(cond)

    # 1. layout arithmetic, at the real 4096 B measured granularity
    comps = build_components(args.hidden, args.inter, args.group)
    gran = 4096
    lay = plan_layout(comps, args.experts, gran, args.scratch_bytes)
    chk("row_bytes granularity-aligned", all(c.row_bytes % gran == 0 for c in comps),
        {c.name: c.row_bytes for c in comps})
    chk("region offsets granularity-aligned", all(c.offset % gran == 0 for c in comps),
        {c.name: c.offset for c in comps})
    chk("regions do not overlap",
        all(comps[i].offset + comps[i].row_bytes * args.experts <= comps[i + 1].offset
            for i in range(len(comps) - 1)))
    chk("granule bytes = sum of component rows",
        lay["granule_bytes"] == sum(c.row_bytes for c in comps), lay["granule_bytes"])

    # 2. a shape that must be REFUSED, not silently rounded
    try:
        plan_layout(build_components(args.hidden, 4, args.group), args.experts, 1 << 20,
                    args.scratch_bytes)
        chk("misaligned row refused", False, "plan_layout accepted a misaligned row")
    except Precondition as exc:
        chk("misaligned row refused", True, str(exc)[:120])

    # 3. placement + route construction: EXACT miss counts, disjoint across the rotation
    import random as _random
    rng = _random.Random(args.seed)
    E, k = args.experts, args.top_k
    R = args.rotation if args.rotation > 0 else max(1, (E // 2) // max(k, 1))
    for placement in ("random", "blocked", "alternate"):
        mask = build_placement(E, k, R, placement, _random.Random(args.seed))
        chk(f"placement/{placement} splits the stack in half", sum(mask) == E // 2, sum(mask))
        for m in [int(v) for v in args.miss_counts.split(",")]:
            routes = build_routes(mask, k, m, R, 1, _random.Random(args.seed))
            counts = {sum(1 for e in row if mask[e]) for rows in routes for row in rows}
            widths = {len(row) for rows in routes for row in rows}
            seen = [e for rows in routes for row in rows for e in row]
            chk(f"routes/{placement}/m={m} exact miss count", counts == {m}, counts)
            chk(f"routes/{placement}/m={m} width == top_k", widths == {k}, widths)
            chk(f"routes/{placement}/m={m} disjoint across rotation",
                len(seen) == len(set(seen)), f"{len(seen)} ids, {len(set(seen))} distinct")

    # 4. an infeasible DISJOINT rotation must RAISE, not silently overlap
    try:
        build_routes(build_placement(E, k, R, "random", rng), k, k, E, 1, rng, disjoint=True)
        chk("infeasible rotation refused", False)
    except Precondition:
        chk("infeasible rotation refused", True)

    # 4b. the correctness path (disjoint=False) must still hold the exact miss count at a batch
    #     that no disjoint rotation could serve
    mask_c = build_placement(E, k, R, "random", _random.Random(args.seed))
    big = build_routes(mask_c, k, k, 1, 64, _random.Random(args.seed), disjoint=False)
    chk("non-disjoint routes: exact miss count at M=64",
        {sum(1 for e in row if mask_c[e]) for rows in big for row in rows} == {k})
    chk("non-disjoint routes: no duplicate expert WITHIN a token",
        all(len(set(row)) == k for rows in big for row in rows))

    # 5. reuse distance guard arithmetic
    granule = lay["granule_bytes"]
    chk("reuse distance exceeds 4x MALL at the configured rotation",
        R * k * granule >= MALL_SAFETY * MALL_BYTES,
        f"{R * k * granule / 2**20:.0f} MiB vs {MALL_SAFETY * MALL_BYTES / 2**20:.0f} MiB")

    # 6. route_bytes counts DISTINCT experts
    mask = build_placement(E, k, R, "random", _random.Random(args.seed))
    routes = build_routes(mask, k, 3, R, 1, _random.Random(args.seed))
    tot, hb, db = route_bytes(routes, mask, granule)
    chk("route_bytes: host share == miss count", abs(hb - 3 * granule) < 1e-6, hb / granule)
    chk("route_bytes: total == top_k granules", abs(tot - k * granule) < 1e-6, tot / granule)
    chk("route_bytes: host + device == total", abs((hb + db) - tot) < 1e-6)

    # 7. the SHAPE classifier itself — a linear series and a cliffed series must be told apart,
    #    AND a series whose noise cannot support either call must refuse to make one.
    hb = {m: m * granule for m in (0, 1, 2, 5, 10)}
    lin = {m: 100.0 + 90.0 * m for m in (0, 1, 2, 5, 10)}
    cliff = {0: 100.0, 1: 900.0, 2: 930.0, 5: 960.0, 10: 1000.0}
    a_lin, a_cliff = analyse_shape(lin, 10, hb), analyse_shape(cliff, 10, hb)
    chk("classifier: linear series -> LINEAR", a_lin["shape"] == "LINEAR",
        f"{a_lin['shape']} cliff_index={a_lin['cliff_index']:.2f}")
    chk("classifier: cliffed series -> CLIFFED", a_cliff["shape"] == "CLIFFED",
        f"{a_cliff['shape']} cliff_index={a_cliff['cliff_index']:.2f}")
    chk("classifier: linear series implies a constant per-miss bandwidth",
        len({round(v, 3) for v in a_lin["implied_host_gbps_per_miss"].values()}) == 1,
        a_lin["implied_host_gbps_per_miss"])
    flat = {m: 100.0 for m in (0, 1, 2, 5, 10)}
    chk("classifier: flat series -> NO_MEASURABLE_MISS_COST",
        analyse_shape(flat, 10, hb)["shape"] == "NO_MEASURABLE_MISS_COST")
    chk("classifier: missing m=0 -> INCOMPLETE",
        analyse_shape({1: 1.0, 10: 2.0}, 10, hb)["shape"] == "INCOMPLETE")
    # noise arms: the SAME series, with per-arm spreads big enough to swallow the differences
    def _sp(mad, med, rel):
        return {"mad": mad, "median": med, "rel_spread": rel}
    noisy_spreads = {m: _sp(80.0, lin[m], 0.6) for m in lin}
    a_noisy = analyse_shape(lin, 10, hb, noisy_spreads)
    chk("classifier: a linear-LOOKING series with huge per-arm spread is NOT called LINEAR",
        a_noisy["shape"] != "LINEAR" and a_noisy["noise_ok"] is False,
        f"{a_noisy['shape']} noise_ok={a_noisy['noise_ok']}")
    chk("classifier: an unresolvable first miss is flagged",
        a_noisy["first_miss_resolved"] is False,
        f"snr={a_noisy['first_miss_snr']}")
    cliff_noisy = {m: _sp(300.0, cliff[m], 0.2) for m in cliff}
    a_cn = analyse_shape(cliff, 10, hb, cliff_noisy)
    chk("classifier: a cliff whose noise interval straddles the kill line is NOT a KILL",
        a_cn["shape"] == "CLIFFED_NOISY",
        f"{a_cn['shape']} ci={a_cn['cliff_index']:.2f} lo={a_cn['cliff_index_lo']}")
    tight = {m: _sp(1.0, cliff[m], 0.02) for m in cliff}
    chk("classifier: a cliff measured tightly IS a KILL",
        analyse_shape(cliff, 10, hb, tight)["shape"] == "CLIFFED")
    clean = {m: _sp(0.5, lin[m], 0.02) for m in lin}
    chk("classifier: a tightly-measured linear series is still LINEAR",
        analyse_shape(lin, 10, hb, clean)["shape"] == "LINEAR")
    # host bandwidth plausibility: same timings, but host bytes so large that the implied host
    # bandwidth is device-class -> the 'host' pages were never on the far side of PCIe
    hb_big = {m: m * granule * 100 for m in (0, 1, 2, 5, 10)}
    a_imp = analyse_shape(lin, 10, hb_big, clean)
    chk("classifier: device-class implied host bandwidth blocks LINEAR",
        a_imp["host_bw_implausible"] is True and a_imp["shape"] != "LINEAR",
        f"{a_imp['shape']} {a_imp['host_gbps_at_full_miss']:.0f} GB/s")

    # 8. spread()
    sp = spread([5.0, 1.0, 3.0, 2.0, 4.0])
    chk("spread median/min/max", sp["median"] == 3.0 and sp["min"] == 1.0 and sp["max"] == 5.0, sp)

    # 9. run-length encoding of the placement mask
    chk("runs() encodes media runs", runs([False, False, True, True, True, False])
        == [(0, 2, False), (2, 3, True), (5, 1, False)])

    # 9b. the DLPack binding — the single riskiest mechanism in this probe. Runs CPU-only, so the
    #     Verify phase can prove it works inside the serve image with no GPU attached.
    dl_ok, dl_detail = dlpack_cpu_check()
    chk("DLPack capsule binds a foreign pointer (CPU arm)", dl_ok, dl_detail)

    # 10. box state + the JSON document shape, end to end
    box = collect_box_state("selftest")
    chk("box state carries MemAvailable", box["meminfo_kb"].get("MemAvailable") is not None)
    chk("box state carries pswpout", box["vmstat"].get("pswpout") is not None)

    layout = dict(lay)
    layout["components"] = [c.as_dict(E) for c in comps]
    layout["va_base"] = None
    cfg = {"experts": E, "top_k": k, "hidden": args.hidden, "inter": args.inter,
           "group_size": args.group, "act_dtype": args.dtype, "rotation_R": R,
           "reps": args.reps, "warmup_windows": args.warmup, "placement": args.placement,
           "n_host_experts": E // 2, "n_device_experts": E - E // 2,
           "miss_counts": [int(v) for v in args.miss_counts.split(",")],
           "timing_M": [int(v) for v in args.timing_m.split(",")],
           "correctness_M": [int(v) for v in args.correctness_m.split(",")],
           "granule_bytes": granule, "seed": args.seed, "stack_bytes_total": granule * E,
           "note": "selftest — not measured"}
    doc = build_result(
        args, {"before": {}, "after": {}}, cfg, layout,
        {"physical_card": None, "name": None, "pci_bus_id": None, "cus": None,
         "total_vram_bytes": None, "rocr_visible_devices": None, "hip_visible_devices": None},
        {"device": None, "host": None, "used": None}, {},
        {"reuse_distance": {"bytes": R * k * granule, "mall_bytes": MALL_BYTES,
                            "required_multiple": MALL_SAFETY,
                            "actual_multiple": R * k * granule / MALL_BYTES, "ok": None}},
        {"gate": "not measured", "arms": [], "all_pass": None, "ledgers_all_identical": None},
        {"unit": "microseconds per w4a8_moe layer launch", "arms": []},
        {}, {}, box, None, {},
        ["SELFTEST: no GPU was touched; every measured field is null by construction"],
        "NOT_MEASURED", ["selftest only — no measurement was performed"], selftest=True)
    problems = check_shape(doc)
    chk("result document passes its own schema check", not problems, problems)
    chk("selftest document reports no measurements",
        doc["timing"]["arms"] == [] and doc["correctness"]["arms"] == [])
    chk("emit() rejects a malformed document",
        check_shape({**doc, "verdict": "GREAT"}) != [], "bad verdict slipped through")

    # 11. the markdown renderer, on a FULLY POPULATED document. This is a renderer unit test with
    #     synthetic inputs; the rendered text is asserted on and then DISCARDED — it is never
    #     written to disk and never printed, so no fabricated number can escape. Without it, a
    #     formatting bug would only surface AFTER a full GPU run had already been paid for.
    try:
        probe_doc = json.loads(json.dumps(doc, default=str))
        probe_doc["media_separation"] = {
            "scratch_device": {"gbps": spread([600.0, 610.0, 620.0, 605.0, 615.0]), "bytes": 1},
            "scratch_host": {"gbps": spread([26.0, 27.0, 28.0, 26.5, 27.5]), "bytes": 1},
            "device_over_host": 22.4, "separated": True}
        probe_doc["correctness"]["arms"] = [
            {"M": 1, "miss": 0, "control_floor": 1e-3, "max_abs_delta": 0.0,
             "reference_absmax": 1.0, "relative_delta": 0.0, "exact": True, "pass": True,
             "out_dtype": "torch.bfloat16", "out_shape": [1, 2048], "ledger_identical": True}]
        probe_doc["timing"]["arms"] = [
            {"M": 1, "miss": m, "rotation_R": R, "mode": "graph", "replays_per_window": 2,
             "capture_error": None, "us_per_launch": spread([lin[m] + d for d in (-1, 0, 1, 2, -2)]),
             "eager_us_per_launch_crosscheck": lin[m], "bytes_per_launch": k * granule,
             "host_bytes_per_launch": m * granule,
             "device_bytes_per_launch": (k - m) * granule, "effective_gbps": 1.0}
            for m in (0, 1, 2, 5, 10)]
        probe_doc["timing"]["modes"] = ["graph"]
        probe_doc["timing"]["mode_consistent"] = True
        probe_doc["timing"]["all_crosschecks_ok"] = True
        probe_doc["shape_analysis"] = {"M=1": analyse_shape(lin, 10, hb, clean)}
        probe_doc["preconditions"]["weight_media_separation"] = {
            "component": "w13", "rows_per_leg": 171, "bytes_per_leg": 256 << 20,
            "device_rows": {"gbps": spread([500.0, 510.0, 505.0, 495.0, 502.0])},
            "host_rows": {"gbps": spread([25.0, 26.0, 27.0, 26.5, 25.5])},
            "device_over_host": 19.4, "separated": True}
        probe_doc["preconditions"]["arena_integrity_after_timing"] = {
            "checked": True, "identical": True, "mismatched": {}}
        probe_doc["diagnostics"] = {
            "arena_device_control": {"what": "synthetic", "arena_us": 101.0, "torch_us": 100.0,
                                     "arena_over_torch": 1.01, "tolerance": 1.15, "ok": True},
            "cache_sensitivity": {
                "what": "synthetic", "R1_us": 100.0, "rotation_us": 900.0,
                "rotation_over_R1": 9.0}}
        probe_doc["box_state_after"] = collect_box_state("selftest-after")
        md = to_markdown(probe_doc)
        chk("markdown renderer handles a fully populated document",
            "Miss-count curve" in md and "cliff_index" in md and len(md) > 800, len(md))
    except Exception as exc:  # noqa: BLE001
        chk("markdown renderer handles a fully populated document", False,
            f"{type(exc).__name__}: {exc}")
    try:
        chk("markdown renderer handles an empty document", len(to_markdown(doc)) > 200)
    except Exception as exc:  # noqa: BLE001
        chk("markdown renderer handles an empty document", False, f"{type(exc).__name__}: {exc}")

    doc["selftest_checks"] = checks
    doc["selftest_passed"] = all(c["pass"] for c in checks if c["pass"] is not None)
    doc["selftest_skipped"] = [c["check"] for c in checks if c["pass"] is None]
    # A green selftest with the DLPack arm SKIPPED is exactly the "capability probe passes while
    # the operation fails" trap: torch is not importable on the bare host, so the single riskiest
    # mechanism in this probe went unexercised. Say so loudly instead of printing an unqualified
    # pass.
    doc["selftest_complete"] = not doc["selftest_skipped"]
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "p2.selftest.json")   # never p2.json — cannot clobber a run
    with open(path, "w") as fh:
        json.dump(doc, fh, indent=2, default=str)
        fh.write("\n")
    _relax(path)
    for c in checks:
        state = "SKIP" if c["pass"] is None else ("PASS" if c["pass"] else "FAIL")
        print(f"[selftest] {state}  {c['check']}"
              + (f"   {c['detail']}" if (c["detail"] and c["pass"] is not True) else ""),
              flush=True)
    if not doc["selftest_complete"]:
        print("[selftest] WARNING: this selftest is INCOMPLETE — "
              + ", ".join(doc["selftest_skipped"])
              + ". It was NOT run in an environment where torch imports, so the DLPack binding "
                "(the mechanism the whole probe stands on) is unexercised. Re-run it inside the "
                "serve image before trusting a green result.", file=sys.stderr, flush=True)
    print(json.dumps({"selftest_passed": doc["selftest_passed"],
                      "selftest_complete": doc["selftest_complete"],
                      "checks": len(checks),
                      "skipped": doc["selftest_skipped"],
                      "failed": [c["check"] for c in checks if c["pass"] is False],
                      "wrote": path}, indent=2), flush=True)
    return 0 if doc["selftest_passed"] else 2


# --------------------------------------------------------------------------------------------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog="p2_mixed_media_moe.py",
        description="P2: mixed-media grouped MoE GEMM — correctness and the miss-count shape",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--experts", type=int, default=512,
                    help="E. Half go on host pages; R disjoint routes need E/2 >= R*top_k")
    ap.add_argument("--top-k", type=int, default=10, help="routed experts per token")
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--inter", type=int, default=768, help="per-rank moe_intermediate (TP=2 shape)")
    ap.add_argument("--group", type=int, default=128, help="int4 quant group size")
    ap.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16"),
                    help="ACTIVATION dtype; the op is dtype-generic on this axis")
    ap.add_argument("--miss-counts", default="0,1,2,5,9,10",
                    help="host-resident experts among the top-k, swept (plan asks 0,1,2,5,10; 9 is "
                         "added because it directly answers 'does making ONE expert resident help')")
    ap.add_argument("--timing-m", default="1",
                    help="batch M values for the timed curve (bs=1 decode is the case that matters)")
    ap.add_argument("--correctness-m", default="1,2,8,64",
                    help="batch M values for the correctness matrix: 1,2 exercise the atomic "
                         "scatter gemm2 arm, 8 the fused gather-reduce, 64 the WMMA gemm1 arm")
    ap.add_argument("--rotation", type=int, default=0,
                    help="R disjoint routes per timed window; 0 = auto = (E/2)//top_k")
    ap.add_argument("--reps", type=int, default=7, help="recorded timed windows (>= 5)")
    ap.add_argument("--warmup", type=int, default=1, help="timed windows discarded before recording")
    ap.add_argument("--control-reps", type=int, default=3,
                    help="reference-vs-reference re-runs per correctness arm; the control floor is "
                         "the MAX over them (one draw is not a floor)")
    ap.add_argument("--max-control-rel", type=float, default=0.02,
                    help="a control floor above this fraction of the reference absmax makes the "
                         "correctness gate too loose to trust; blocks PASS")
    ap.add_argument("--max-arena-overhead", type=float, default=1.15,
                    help="tolerance on (arena m=0 time)/(plain torch stack m=0 time); above it the "
                         "curve's baseline carries arena overhead and PASS is blocked")
    ap.add_argument("--placement", default="random", choices=("random", "blocked", "alternate"),
                    help="how device/host experts interleave in the stack")
    ap.add_argument("--card", type=int, default=0,
                    help="ROCR index of the card to time on (0 = RX 9070 XT). Recorded in the JSON")
    ap.add_argument("--scratch-bytes", type=int, default=SCRATCH_BYTES,
                    help="working set for the media-separation precondition (must exceed 4x MALL)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--tag", default="", help="suffix for the output filenames")
    ap.add_argument("--fail-on-kill", action="store_true",
                    help="exit 3 when the measured verdict is KILL (default: exit 0, verdict in JSON)")
    ap.add_argument("--selftest", "--dry-run", dest="selftest", action="store_true",
                    help="validate arguments, layout arithmetic, route construction, the shape "
                         "classifier and the JSON schema WITHOUT touching the GPU")
    args = ap.parse_args(argv)
    if args.reps < 5:
        ap.error("--reps must be >= 5 (median and spread need at least five kept windows)")
    if args.warmup < 1:
        ap.error("--warmup must be >= 1 (a warm-up window is always discarded)")
    if args.top_k < 1 or args.top_k > args.experts:
        ap.error("--top-k out of range")
    if args.control_reps < 1:
        ap.error("--control-reps must be >= 1")
    # Enforced HERE as well as in run_gpu so `--selftest` refuses the same argument sets a real run
    # would — otherwise you discover an undefined cliff_index only after the container is up.
    _mc = sorted({int(v) for v in args.miss_counts.split(",") if v.strip() != ""})
    if not _mc:
        ap.error("--miss-counts is empty")
    if [m for m in _mc if not 0 <= m <= args.top_k]:
        ap.error(f"--miss-counts values must lie in [0, top_k={args.top_k}]; got {_mc}")
    missing = [m for m in (0, 1, args.top_k) if m not in _mc]
    if missing:
        ap.error(f"--miss-counts must contain {missing} as well: cliff_index = "
                 f"(t(1)-t(0))/((t(top_k)-t(0))/top_k) is undefined without t(0), t(1) and "
                 f"t(top_k={args.top_k})")
    if not [v for v in args.timing_m.split(",") if v.strip()]:
        ap.error("--timing-m must name at least one batch size")
    if not [v for v in args.correctness_m.split(",") if v.strip()]:
        ap.error("--correctness-m must name at least one batch size")
    if args.scratch_bytes < MALL_SAFETY * MALL_BYTES:
        ap.error(f"--scratch-bytes must be >= {MALL_SAFETY * MALL_BYTES} (4x MALL) or the media "
                 f"separation check measures Infinity Cache")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.selftest:
        return run_selftest(args)
    try:
        return run_gpu(args)
    except Precondition as exc:
        print(f"\n[p2] PRECONDITION FAILED: {exc}\n[p2] no verdict was reached. If any leg had "
              f"already completed it was written to p2.aborted.json with verdict NOT_MEASURED; "
              f"nothing was reported as if it had been measured.",
              file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
