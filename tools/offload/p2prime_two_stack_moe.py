#!/usr/bin/env python3
"""P2prime -- is per-layer MoE time LINEAR or CLIFFED in host-resident (missed) expert count?

This is P2's original question, asked again on a mechanism that EXISTS on this box.

WHY P2 COULD NOT ANSWER IT
  P2 built one `hipMemAddressReserve` VA and backed alternating expert rows with
  `hipMemCreate(location.type = hipMemLocationTypeHost)` handles.  Every call returned
  hipSuccess, the fingerprints were clean -- and the media-separation precondition came back
  1.00 (device 117.9 GB/s vs "host" 118.2 GB/s), because on this stack that location field is
  ACCEPTED, ECHOED and IGNORED: the pages were VRAM.  P1 and P3 confirmed it independently.
  So there is NO single device VA whose sub-ranges live on different media, and the mixed
  device/host stack the plan is built on is not constructible.  P2 aborted with `cliff_index`
  never computed, which is why the shape of the curve is still the top open unknown.

WHAT THIS PROBE BUILDS INSTEAD -- an EXPLICIT TWO-STACK layout
  * a DEVICE expert stack from ordinary `hipMalloc`;
  * a HOST expert stack from `hipHostMalloc(...Mapped)` + `hipHostGetDevicePointer`
    (P1: real host pages, kernel-read at 28.93 GB/s on card 0 / 14.48 GB/s on card 1, correct in
    both directions on both cards);
  * a per-expert POINTER TABLE selecting which stack each routed expert reads from.
  The two stacks are separate virtual addresses and cannot be interleaved -- that is the whole
  reason a table is needed rather than a placement.

WHAT RUNS -- the SHIPPED kernel, not a probe-shaped one
  `gemv_decode::gemv_decode_core` from rdna4-hip-kernels (the decode-GEMV core that the w4a16 MoE
  gemm1+SiLU and gemm2 launchers use), instantiated with a WLoad policy
  `TwoStackInt4A16GemvLoader` that inherits everything from the shipped
  `Int4A16GemvLoader<__half>` and overrides ONLY `wq_expert` / `ws_expert` / `wz_expert` to look
  the per-expert base up in the table.  Per KERNEL_CORE_POLICY.md a placement scheme is a loader
  policy on the shared core, never a fork -- so the inner loop, tiling and numerics being measured
  are production's by construction.  The probe A/Bs the indirection itself against the STOCK
  loader on identical data, because an A/B baseline must be the old CODE.

WHAT IS CHECKED BEFORE ANY NUMBER IS REPORTED (this is what P2 lacked)
  Placement is never taken from a return code or a location field.  Every stack is classified by
  (a) the per-card amdgpu sysfs `mem_info_vram_used` / `mem_info_gtt_used` delta, (b) the
  /proc/meminfo MemAvailable delta, (c) a checksummed kernel-read bandwidth, and (d) CPU
  accessibility.  The run ABORTS -- loudly, non-zero exit, artifact still written -- unless:
    * the device stack consumes VRAM and the host stack does not;
    * the host stack consumes host RAM and is CPU-accessible;
    * device-read / host-read bandwidth ratio >= P2PRIME_MIN_MEDIA_RATIO (P2 measured 1.00 here);
    * host-read bandwidth is PCIe-BOUNDED, i.e. below the measured copy engine x a small margin;
    * at full miss the kernel's achieved weight bandwidth does not EXCEED the PCIe ceiling
      (bytes that must cross the link cannot outrun it -- the single sharpest falsifier of a
      fake-host stack, and the one that would have caught P2 in seconds).

CORRECTNESS BEFORE TIMING
  The device and host stacks are filled with DIFFERENT deterministic content, and the mixed run
  must reproduce, BIT FOR BIT, an output assembled per expert-block from the all-device and
  all-host references.  That is a positive provenance test: it fails both if the table is ignored
  (mixed == all-device) and if it is mis-indexed (one expert dequantized against another's
  scale -- plausible numbers, no crash).  Only then are both stacks re-filled to identical
  content for the timing sweep, where every timed output is still checksum-compared.

WHAT THE ANSWER DECIDES
  cliff_index = (t(1) - t(0)) / (t(n) - t(0)) over misses out of top-k.
    ~0.1  LINEAR  -- misses cost additively; per-expert placement is worth up to ~2x and M2 should
                     carry a per-expert tier.
    ~1.0  CLIFFED -- the layer is gated by its slowest workgroup, so one host expert costs nearly
                     what ten do; per-expert placement buys approximately ZERO and M2 is
                     layer-granular, full stop.
  The prior is bad: under a cliff, per-expert placement needs P(all top-k resident) = h^k near 1,
  i.e. h >= 0.9895 for 90% of layers at k=10, which no realistic tier reaches.  The probe does not
  rely on that prior -- it reports the measured expectation E[t | hit rate h] directly from the
  measured t(m), and compares it against LAYER-GRANULAR placement at the same byte budget.  If the
  two are equal, per-expert placement is worthless whatever the curve is called.

USAGE
  python3 tools/offload/p2prime_two_stack_moe.py --selftest        # no GPU, validates shape+math
  python3 tools/offload/p2prime_two_stack_moe.py                   # both cards, full sweep

OPERATIONAL NOTES
  * The GPU lease is WAIVED for this workstream by explicit user instruction, but two GPU jobs must
    never run concurrently.  This probe runs its cards SERIALLY and records box state around each.
  * ROCR_VISIBLE_DEVICES is forced to "0,1" with HIP_VISIBLE_DEVICES unset (the Ryzen iGPU is ROCm
    device 2 and advertises 47 GB of GTT -- never a compute target, and poison for any
    "largest free pool" logic).
  * Every timing records WHICH PHYSICAL CARD it came from.  Card 1's root port is trained Gen4 x8
    against card 0's Gen5 x8, so its host-read ceiling is half; both are exercised because this box
    has burned people on cross-card assumptions twice.
"""

from __future__ import annotations

import os
import sys

# ---------------------------------------------------------------------------
# Device-visibility fence.  MUST happen before libamdhip64 is loaded, and before
# p1_host_read_bw is imported (it loads nothing at import time, but the ordering
# is the invariant, not the current implementation).
# ---------------------------------------------------------------------------
REQUIRED_ROCR = "0,1"
_REEXEC_FLAG = "_P2PRIME_REEXEC"


def _fence_device_visibility() -> dict:
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
        env["_P2PRIME_ORIG_VIS"] = repr(inherited)
        env["ROCR_VISIBLE_DEVICES"] = REQUIRED_ROCR
        for k in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL"):
            env.pop(k, None)
        env[_REEXEC_FLAG] = "1"
        os.execve(sys.executable, [sys.executable, os.path.abspath(__file__)] + sys.argv[1:], env)
    return {
        "as_launched": os.environ.get("_P2PRIME_ORIG_VIS") or repr(inherited),
        "effective": {"ROCR_VISIBLE_DEVICES": os.environ.get("ROCR_VISIBLE_DEVICES"),
                      "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES")},
        "reexeced": os.environ.get(_REEXEC_FLAG) == "1",
        "note": ("forced ROCR_VISIBLE_DEVICES=0,1 with HIP_VISIBLE_DEVICES unset so the Ryzen "
                 "iGPU (ROCm device 2, 47 GB GTT) can never enter enumeration"),
    }


# Only fence when run as a script: the fence re-execs, so doing it at import time would make
# `import p2prime_two_stack_moe` silently launch a GPU run.  (P1 shipped that bug once.)
if __name__ == "__main__":
    _ENV_FENCE = _fence_device_visibility()
else:
    _ENV_FENCE = {"as_launched": None, "effective": None, "reexeced": False,
                  "note": "module imported, not executed: no fence, no GPU work"}

import argparse  # noqa: E402
import ctypes  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import platform  # noqa: E402
import random  # noqa: E402
import shutil  # noqa: E402
import statistics  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# Reuse the Phase 0 scaffolding rather than re-deriving it.  p1_host_read_bw deliberately guards
# its own device fence behind `__name__ == "__main__"` so importing it is inert; these helpers are
# the ones every Phase 0 artifact's provenance is already expressed in, and re-implementing them
# here would let the two probes' placement classifications drift apart silently.
import p1_host_read_bw as p1  # noqa: E402

_read_text = p1._read_text
_run_cmd = p1._run_cmd
_parse_kv_kb = p1._parse_kv_kb
_sha256 = p1._sha256
sysfs_amdgpu_cards = p1.sysfs_amdgpu_cards
card_mem = p1.card_mem
pcie_link_chain = p1.pcie_link_chain
LinkSampler = p1.LinkSampler
collect_box_state = p1.collect_box_state
classify_placement = p1.classify_placement
cpu_readable = p1.cpu_readable
cpu_writable = p1.cpu_writable
maps_entry_for = p1.maps_entry_for
_mem_available_bytes = p1._mem_available_bytes

SCHEMA_VERSION = 1
PROBE_ID = "P2prime"
DEFAULT_OUTDIR = os.path.join(REPO, "docs", "measurements", "WEIGHT_OFFLOAD_2026-09-02")
BUILD_DIR = os.path.join(HERE, "_build")
HIP_SRC = os.path.join(HERE, "p2prime_kernels.hip")
SO_PATH = os.path.join(BUILD_DIR, "p2prime_kernels.so")
DEFAULT_KERNELS_SRC = "/home/pat/code/rdna4-hip-kernels/fp8_wmma/fp8_wmma_rocm"

# hipHostMalloc flags (hip_runtime_api.h)
HHM_PORTABLE = 0x1
HHM_MAPPED = 0x2

# ---- precondition thresholds ------------------------------------------------
# The device/host kernel-read ratio P2 required and measured as 1.00.  P1 measured the real
# separation at 692 / 28.93 = 23.9x on card 0 and 692 / 14.48 = 47.8x on card 1, so 4.0 is a very
# loose floor that still cannot be cleared by a fake-host (VRAM) stack.
MIN_MEDIA_RATIO = 4.0
# A host read must be PCIe-BOUNDED.  P4: 28.70 GB/s card 0, 14.34 GB/s card 1 at 256 MiB; under
# host DDR load both collapse to ~12.4.  Anything materially faster than the copy engine on the
# SAME card is not crossing the link.
HOST_BW_MAX_OVER_COPY = 1.25
HOST_BW_MIN_GBPS = 4.0
DEVICE_BW_MIN_GBPS = 100.0
# Placement classification (fractions of the region size).
PLACE_VRAM_DEVICE_FRAC = 0.5
PLACE_VRAM_HOST_FRAC = 0.25
PLACE_MEMAVAIL_HOST_FRAC = 0.5
# hipEvent totals are cross-checked against the wall clock over the whole rep loop.  Wall time
# strictly exceeds device time, so event-derived time may be a little BELOW wall time -- never by
# this factor.  A garbage hipEventElapsedTime is otherwise indistinguishable from a great result.
TIMING_SANITY_FACTOR = 4.0
# The curve must actually separate before its shape means anything.  If all-host is not meaningfully
# slower than all-device, the probe reports INDETERMINATE rather than a shape.
MIN_CURVE_SEPARATION = 1.20
# Verdict bands on cliff_index = (t(1) - t(0)) / (t(n) - t(0)).
CLIFF_INDEX_LINEAR_MAX = 0.25
CLIFF_INDEX_CLIFF_MIN = 0.60
# The pointer-table indirection must not itself move the number it is used to measure.
MAX_INDIRECTION_OVERHEAD = 0.05

HIP_MEMORY_TYPE = {0: "Host", 1: "Device", 2: "Array", 3: "Unified", 4: "Managed"}


class ProbeError(RuntimeError):
    """Precondition or measurement failure.  Always fatal, always non-zero exit."""


def _fail(msg: str) -> None:
    raise ProbeError(msg)


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _align_up(x: int, a: int) -> int:
    return ((x + a - 1) // a) * a


# ===========================================================================
# Pure layout / route / mask math.  No GPU, no ctypes -- all of this is what
# --selftest exercises.
# ===========================================================================

GRANULE_COMPONENT_ALIGN = 256    # every component base 256 B aligned (>= the 16 B v4i_t load)
GRANULE_STRIDE_ALIGN = 4096      # every expert granule starts on a page

# The table path's zero-point ENABLE flag.  `wz_expert(z, e)` keys off z being non-null and then
# ignores its value, returning c_wz_tab[e]; it is never dereferenced when use_table=1.  This is a
# deliberately un-mappable low value so that if it ever DID reach the stock loader it would fault
# immediately rather than read plausible bytes.
Z_ENABLED_SENTINEL = 0x1


def expert_layout(hidden: int, inter: int, group_size: int, zeros: bool) -> dict:
    """Byte layout of ONE expert's granule, expert-major (all of expert e's tensors contiguous).

    A GRANULE IS ONE EXPERT'S SLICE OF EVERY TENSOR IN THE CONTAINER.  Scales and zero-points must
    travel with the packed weights or an expert is dequantized against another expert's scale --
    silent wrong numbers, no crash.  Laying the granule out expert-major is what makes "this expert
    lives on the host stack" a single pointer rather than six independent decisions, and it is the
    layout a real implementation would ship.

    w13 (gemm1, gate|up fused):  (2*inter, hidden/8) int32     scales (hidden/g, 2*inter) fp16
                                 zeros  (hidden/g, 2*inter/8) int32
    w2  (gemm2, down):           (hidden, inter/8)  int32      scales (inter/g, hidden)  fp16
                                 zeros  (inter/g,  hidden/8)  int32
    Scale/zero planes are GROUP-MAJOR (E, G, N) -- that is the shared core's contract
    (`ws_e[(long)g * N + nc]`), not a choice this probe gets to make.
    """
    if hidden % 32 or inter % 32:
        _fail(f"hidden ({hidden}) and inter ({inter}) must be multiples of 32 for the int4 core")
    if group_size % 32 or hidden % group_size or inter % group_size:
        _fail(f"group_size {group_size} must be a multiple of 32 and divide both hidden and inter")
    n1, k1 = 2 * inter, hidden
    n2, k2 = hidden, inter
    g1, g2 = k1 // group_size, k2 // group_size
    comps = [
        ("w13", n1 * (k1 // 8) * 4),
        ("s13", g1 * n1 * 2),
        ("z13", g1 * (n1 // 8) * 4 if zeros else 0),
        ("w2", n2 * (k2 // 8) * 4),
        ("s2", g2 * n2 * 2),
        ("z2", g2 * (n2 // 8) * 4 if zeros else 0),
    ]
    off, offsets, sizes = 0, {}, {}
    for name, sz in comps:
        if sz == 0:
            offsets[name], sizes[name] = None, 0
            continue
        off = _align_up(off, GRANULE_COMPONENT_ALIGN)
        offsets[name], sizes[name] = off, sz
        off += sz
    return {
        "n1": n1, "k1": k1, "g1": g1, "n2": n2, "k2": k2, "g2": g2,
        "offsets": offsets, "sizes": sizes,
        "payload_bytes": sum(sizes.values()),
        "granule_stride": _align_up(off, GRANULE_STRIDE_ALIGN),
        # Bytes a single expert contributes to each GEMM's weight traffic.  These are the
        # denominators for every achieved-bandwidth cross-check, so they are computed once here
        # and never re-derived at a call site.
        "gemm1_bytes_per_expert": sizes["w13"] + sizes["s13"] + sizes["z13"],
        "gemm2_bytes_per_expert": sizes["w2"] + sizes["s2"] + sizes["z2"],
    }


def build_route(tokens: int, top_k: int, num_experts: int, block_m: int,
                experts: list[int]) -> dict:
    """Emulate `moe_align` for a given (token, k) -> expert assignment.

    Contract the shared core relies on (gemv_decode_impl, GATHER=true):
      sorted_token_ids[row0 + r] = the flattened offset `token*top_k + k`, or a value
          >= num_valid_tokens for a padding row;
      expert_ids[block] = that block's expert;
      num_tokens_post_padded[0] = n_blocks * block_m (the core early-returns past it);
      s_src = offs / top_k (non-scatter, A indexed by TOKEN) or row0 + r (scatter, A indexed by
          PADDED ROW) -- which is why the scatter and non-scatter arms take different A shapes.

    `experts` is the flat assignment, len == tokens*top_k, entry i = expert for
    (token = i // top_k, k = i % top_k).
    """
    num_valid = tokens * top_k
    if len(experts) != num_valid:
        _fail(f"route needs {num_valid} expert assignments, got {len(experts)}")
    if any(not (0 <= e < num_experts) for e in experts):
        _fail("route contains an out-of-range expert id")
    by_expert: dict[int, list[int]] = {}
    for offs, e in enumerate(experts):
        by_expert.setdefault(e, []).append(offs)
    sti: list[int] = []
    eid: list[int] = []
    block_expert: list[int] = []
    for e in sorted(by_expert):
        rows = by_expert[e]
        for i in range(0, len(rows), block_m):
            chunk = rows[i:i + block_m]
            sti.extend(chunk)
            sti.extend([num_valid] * (block_m - len(chunk)))   # pad marker: >= num_valid
            eid.append(e)
            block_expert.append(e)
    n_blocks = len(eid)
    return {
        "tokens": tokens, "top_k": top_k, "block_m": block_m,
        "num_valid_tokens": num_valid,
        "sorted_token_ids": sti,
        "expert_ids": eid,
        "num_tokens_post_padded": n_blocks * block_m,
        "n_blocks": n_blocks,
        "P": n_blocks * block_m,
        "block_expert": block_expert,
        "distinct_experts": sorted(by_expert),
    }


def draw_route_experts(rng: random.Random, tokens: int, top_k: int, num_experts: int,
                       mode: str) -> list[int]:
    """Draw the (token, k) -> expert assignment.

    `distinct` (default) gives every routed slot its own expert, so block <-> expert is 1:1 and a
    miss COUNT is unambiguous -- which is the whole point of the sweep.  `random` allows collisions
    (an expert serving several tokens shares one block), which is closer to real traffic but makes
    "5 misses" mean two different things depending on the draw; it exists so the shape can be
    re-checked under reuse, not as the headline arm.
    """
    n = tokens * top_k
    if mode == "distinct":
        if n > num_experts:
            _fail(f"distinct route needs tokens*top_k ({n}) <= num_experts ({num_experts})")
        return rng.sample(range(num_experts), n)
    if mode == "random":
        return [rng.randrange(num_experts) for _ in range(n)]
    _fail(f"unknown route mode {mode!r} (expected 'distinct' or 'random')")
    return []


def choose_host_blocks(rng: random.Random, route: dict, miss: int) -> list[int]:
    """Pick which BLOCKS read from the host stack, and return the experts to flip.

    Placement is per EXPERT (a granule is an expert), so when one expert owns several blocks
    (route mode `random`) flipping it misses all of them at once.  Blocks are chosen first and the
    achieved miss count is reported, because silently delivering 7 misses when 5 were asked for is
    exactly the sort of drift that makes a curve unreadable.
    """
    n_blocks = route["n_blocks"]
    if not (0 <= miss <= n_blocks):
        _fail(f"miss count {miss} outside [0, {n_blocks}]")
    order = list(range(n_blocks))
    rng.shuffle(order)
    host_experts: set[int] = set()
    for b in order:
        if len(host_experts) >= miss:
            break
        host_experts.add(route["block_expert"][b])
    achieved = sum(1 for b in range(n_blocks) if route["block_expert"][b] in host_experts)
    return sorted(host_experts), achieved


# ===========================================================================
# Statistics, curve fitting and the decision analysis.  Pure functions.
# ===========================================================================

def stats(vals: list[float]) -> dict:
    """Median plus an honest spread.  A confident wrong number is worse than no number, so every
    timing carries n, min, median, IQR and the full p10/p90 band rather than a bare mean."""
    if not vals:
        return {"n": 0}
    s = sorted(vals)
    n = len(s)

    def q(p: float) -> float:
        if n == 1:
            return s[0]
        i = p * (n - 1)
        lo, hi = int(math.floor(i)), int(math.ceil(i))
        return s[lo] + (s[hi] - s[lo]) * (i - lo)

    med = statistics.median(s)
    return {
        "n": n, "min": s[0], "max": s[-1], "median": med,
        "mean": statistics.fmean(s),
        "p10": q(0.10), "p25": q(0.25), "p75": q(0.75), "p90": q(0.90),
        "iqr": q(0.75) - q(0.25),
        "rel_spread": ((q(0.90) - q(0.10)) / med) if med else None,
        "stdev": statistics.pstdev(s) if n > 1 else 0.0,
    }


def _ols(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Ordinary least squares y = a + b*x, returning (a, b, r2)."""
    n = len(xs)
    if n < 2:
        return (ys[0] if ys else 0.0), 0.0, float("nan")
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = (sxy / sxx) if sxx else 0.0
    a = my - b * mx
    sst = sum((y - my) ** 2 for y in ys)
    sse = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    r2 = (1.0 - sse / sst) if sst > 0 else float("nan")
    return a, b, r2


def fit_curve(points: dict[int, float]) -> dict:
    """Fit the miss-count curve and name its shape.

    Two models, chosen because they are the two ANSWERS, not because they fit well:

      LINEAR          t(m) = t0 + s*m
          every host-resident expert adds its own transfer time; misses serialise.

      PARALLEL-SERVICE  t(m) = t0 + B * (1 + (m-1)/W)   for m >= 1
          the first miss costs B (the layer waits for a PCIe round trip it did not have to make
          before), and further misses are absorbed by W-way concurrency.  W ~ 1 collapses to
          LINEAR; W >> n means the layer is gated by its slowest workgroup and one miss costs what
          n do -- the CLIFF.  It is linear in (B, B/W), so it is an exact regression rather than a
          search, and W is reported as `effective_miss_concurrency`.

    The headline scalar is deliberately model-free:
        cliff_index = (t(1) - t(0)) / (t(n) - t(0))
    which is 1/n for a pure line and 1.0 for a pure cliff.
    """
    ms = sorted(points)
    if len(ms) < 2:
        return {"status": "INSUFFICIENT_POINTS", "n_points": len(ms)}
    m_lo, m_hi = ms[0], ms[-1]
    t0, tn = points[m_lo], points[m_hi]
    out: dict = {"m_min": m_lo, "m_max": m_hi, "t_at_m_min_ms": t0, "t_at_m_max_ms": tn,
                 "separation_ratio": (tn / t0) if t0 else None}

    a, b, r2 = _ols([float(m) for m in ms], [points[m] for m in ms])
    out["linear_fit"] = {"intercept_ms": a, "slope_ms_per_miss": b, "r2": r2,
                         "predicted_t0_ms": a, "measured_t0_ms": t0}

    tail = [m for m in ms if m >= 1]
    if len(tail) >= 2:
        aa, bb, rr = _ols([float(m - 1) for m in tail], [points[m] - t0 for m in tail])
        w = (aa / bb) if bb > 1e-9 else float("inf")
        out["parallel_service_fit"] = {
            "first_miss_cost_ms": aa,
            "marginal_ms_per_extra_miss": bb,
            "effective_miss_concurrency": w,
            "r2": rr,
            "note": ("W ~ 1 => misses serialise (LINEAR); W >= m_max => the layer is gated by its "
                     "slowest workgroup and one miss costs what m_max do (CLIFF)"),
        }

    span = tn - t0
    if t0 and (tn / t0) < MIN_CURVE_SEPARATION:
        out["cliff_index"] = None
        out["shape"] = "INDETERMINATE"
        out["shape_reason"] = (
            f"all-host is only {tn / t0:.3f}x all-device (< {MIN_CURVE_SEPARATION}x): the curve "
            "does not separate, so its shape carries no information. Either the layer is not "
            "weight-bound at this shape or the two stacks are not on different media -- check the "
            "precondition block before reading anything else here.")
        return out
    if span <= 0:
        out["cliff_index"] = None
        out["shape"] = "INDETERMINATE"
        out["shape_reason"] = "t(m_max) <= t(m_min): no measurable miss cost"
        return out

    t1 = points.get(1)
    if t1 is None:
        out["cliff_index"] = None
        out["shape"] = "INDETERMINATE"
        out["shape_reason"] = "m=1 was not measured; cliff_index is undefined without it"
        return out
    ci = (t1 - t0) / span
    out["cliff_index"] = ci
    out["cliff_index_pure_linear"] = 1.0 / m_hi if m_hi else None
    if ci <= CLIFF_INDEX_LINEAR_MAX:
        out["shape"] = "LINEAR"
    elif ci >= CLIFF_INDEX_CLIFF_MIN:
        out["shape"] = "CLIFFED"
    else:
        out["shape"] = "PARTIAL"
    out["shape_bands"] = {"linear_max": CLIFF_INDEX_LINEAR_MAX, "cliff_min": CLIFF_INDEX_CLIFF_MIN}
    return out


def hit_rate_analysis(points: dict[int, float], top_k: int,
                      hit_rates: list[float]) -> list[dict]:
    """The decision, computed from the MEASURED curve with no model in between.

    For a per-expert device tier with per-expert hit rate h, the routed miss count is
    Binomial(top_k, 1-h), so the expected layer time is a direct weighted sum of the measured
    t(m).  The comparison that matters is NOT against all-host -- it is against LAYER-GRANULAR
    placement holding the SAME byte budget, whose expected time is just h*t(0) + (1-h)*t(n)
    (a layer is all-device or all-host).  Layer-granular needs no route change, no `slot_of`, no
    second stack in the hot path and is immune to the h^top_k problem, so per-expert placement has
    to beat it to be worth building at all.
    """
    ms = sorted(points)
    out: list[dict] = []
    for h in hit_rates:
        p = 1.0 - h
        if not points:
            continue
        # Only the fully-measured 0..top_k sweep supports an exact expectation.
        exact = (ms == list(range(0, top_k + 1)))
        if exact:
            ev = sum(math.comb(top_k, m) * (p ** m) * (h ** (top_k - m)) * points[m]
                     for m in range(top_k + 1))
        else:
            # Fall back to linear interpolation over the measured points at the expected miss
            # count.  Flagged, because it is an approximation of the thing the exact arm computes.
            em = p * top_k
            lo = max([m for m in ms if m <= em], default=ms[0])
            hi = min([m for m in ms if m >= em], default=ms[-1])
            ev = points[lo] if hi == lo else (
                points[lo] + (points[hi] - points[lo]) * (em - lo) / (hi - lo))
        t_all_host, t_all_dev = points[ms[-1]], points[ms[0]]
        t_layer = h * t_all_dev + (1.0 - h) * t_all_host
        out.append({
            "hit_rate": h,
            "exact_binomial": exact,
            "expected_ms_per_expert_tier": ev,
            "expected_ms_layer_granular": t_layer,
            "speedup_vs_all_host": (t_all_host / ev) if ev else None,
            "layer_granular_speedup_vs_all_host": (t_all_host / t_layer) if t_layer else None,
            "per_expert_gain_over_layer_granular": (t_layer / ev) if ev else None,
            "prob_all_resident": h ** top_k,
        })
    return out


def decide(fit: dict, hit_rows: list[dict]) -> dict:
    """Turn the fitted shape into the M2 recommendation, in the words the plan will need."""
    shape = fit.get("shape")
    gains = [r["per_expert_gain_over_layer_granular"] for r in hit_rows
             if r.get("per_expert_gain_over_layer_granular") is not None]
    best_gain = max(gains) if gains else None
    if shape == "INDETERMINATE":
        return {"verdict": "INDETERMINATE", "m2_shape": None,
                "summary": fit.get("shape_reason", "curve did not separate"),
                "best_per_expert_gain_over_layer_granular": best_gain}
    if shape == "LINEAR":
        return {
            "verdict": "LINEAR",
            "m2_shape": "per-expert placement is worth building",
            "summary": (f"cliff_index {fit['cliff_index']:.3f} <= {CLIFF_INDEX_LINEAR_MAX}: misses "
                        f"cost additively, so a per-expert device tier converts byte hit rate into "
                        f"time almost 1:1 (best measured gain over layer-granular at equal byte "
                        f"budget: {best_gain:.3f}x)."),
            "best_per_expert_gain_over_layer_granular": best_gain,
        }
    if shape == "CLIFFED":
        return {
            "verdict": "CLIFFED",
            "m2_shape": "layer-granular only -- do NOT build a per-expert tier",
            "summary": (f"cliff_index {fit['cliff_index']:.3f} >= {CLIFF_INDEX_CLIFF_MIN}: one "
                        f"host-resident expert costs nearly what all of them do, so per-expert "
                        f"placement needs P(all top-k resident) = h^k near 1 to pay. Best measured "
                        f"gain over layer-granular at equal byte budget: {best_gain:.3f}x."),
            "best_per_expert_gain_over_layer_granular": best_gain,
        }
    return {
        "verdict": "PARTIAL",
        "m2_shape": "decide on the measured gain, not the label",
        "summary": (f"cliff_index {fit['cliff_index']:.3f} sits between the bands. The only figure "
                    f"that matters is the gain over layer-granular at equal byte budget: "
                    f"{best_gain:.3f}x."),
        "best_per_expert_gain_over_layer_granular": best_gain,
    }


# ===========================================================================
# Build + ctypes binding
# ===========================================================================

REQUIRED_SYMBOLS = (
    "p2p_err", "p2p_max_experts", "p2p_runtime_version", "p2p_driver_version",
    "p2p_device_count", "p2p_set_device", "p2p_sync", "p2p_mem_info", "p2p_device_info",
    "p2p_malloc_device", "p2p_free_device", "p2p_host_malloc", "p2p_host_free",
    "p2p_host_dev_ptr", "p2p_memcpy", "p2p_memset", "p2p_memcpy_bw", "p2p_fill_seq",
    "p2p_fill_hash", "p2p_fill_half_uniform", "p2p_fill_float_uniform", "p2p_read_bw",
    "p2p_thrash", "p2p_set_tables", "p2p_get_table_entry", "p2p_gemm1_silu", "p2p_gemm2",
    "p2p_select_tiling", "p2p_pointer_attrs",
)


def kernels_src_provenance(src_dir: str, allow_dirty: bool) -> dict:
    """Record -- and gate on -- the state of the SHARED rdna4-hip-kernels tree we compile against.

    The probe includes `gemv_decode.h` from a tree other agents edit concurrently.  That is the
    source-isolation hazard the repo rules exist for: hipcc copies whatever is on disk at compile
    time, so a dirty tree bakes somebody else's mid-edit core into a measurement that will be cited
    as "the shipped kernel".  Refuse by default; `--allow-dirty-kernels` is an explicit,
    recorded override.
    """
    files = ("gemv_decode.h", "tile_config.h")
    rec: dict = {"path": src_dir, "files": {}}
    for f in files:
        p = os.path.join(src_dir, f)
        if not os.path.isfile(p):
            _fail(f"kernels source missing: {p} (set --kernels-src to a clean worktree)")
        rec["files"][f] = {"sha256": _sha256(p), "bytes": os.path.getsize(p)}
    top = _run_cmd(["git", "-C", src_dir, "rev-parse", "--show-toplevel"])
    root = top.get("stdout") if top.get("rc") == 0 else None
    rec["git_toplevel"] = root
    if root:
        for name, args in (("sha", ["rev-parse", "HEAD"]),
                           ("branch", ["rev-parse", "--abbrev-ref", "HEAD"]),
                           ("porcelain", ["status", "--porcelain"])):
            r = _run_cmd(["git", "-C", root] + args)
            rec[name] = r.get("stdout") if r.get("rc") == 0 else None
        rec["dirty"] = bool(rec.get("porcelain"))
        rec["porcelain"] = (rec.get("porcelain") or "")[:4000]
    else:
        rec["dirty"] = None
    if rec.get("dirty") and not allow_dirty:
        _fail(f"kernels tree {root} is DIRTY -- compiling against a tree another agent may be "
              f"mid-edit on would bake unknown source into this measurement. Point --kernels-src "
              f"at a clean worktree, or pass --allow-dirty-kernels to record the override.\n"
              f"{rec['porcelain']}")
    rec["allow_dirty_override_used"] = bool(rec.get("dirty") and allow_dirty)
    return rec


def build_kernels(arch: str, rebuild: bool, hipcc: str, src_dir: str) -> dict:
    if not os.path.isfile(HIP_SRC):
        _fail(f"kernel source missing: {HIP_SRC}")
    if not (os.path.isfile(hipcc) or shutil.which(hipcc)):
        _fail(f"hipcc not found: {hipcc} (set --hipcc)")
    os.makedirs(BUILD_DIR, exist_ok=True)
    cmd = [hipcc, "-O3", "-std=c++17", f"--offload-arch={arch}", "-fPIC", "-shared",
           "-I", src_dir, "-o", SO_PATH, HIP_SRC]
    newest_dep = max([os.path.getmtime(HIP_SRC)] +
                     [os.path.getmtime(os.path.join(src_dir, f))
                      for f in ("gemv_decode.h", "tile_config.h")])
    need = rebuild or not os.path.isfile(SO_PATH) or os.path.getmtime(SO_PATH) < newest_dep
    info = {"hipcc": hipcc, "arch": arch, "cmd": " ".join(cmd), "so": SO_PATH,
            "rebuilt": bool(need), "src_sha256": _sha256(HIP_SRC), "include_dir": src_dir}
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


def load_kernels(check_only: bool = False) -> ctypes.CDLL:
    if not os.path.isfile(SO_PATH):
        _fail(f"{SO_PATH} not built")
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
    ull_p = ctypes.POINTER(ctypes.c_ulonglong)
    ci, cu, cf = ctypes.c_int, ctypes.c_uint, ctypes.c_float
    lib.p2p_err.restype = ctypes.c_char_p
    lib.p2p_err.argtypes = [ci]
    for fn, args in (
        ("p2p_max_experts", []),
        ("p2p_runtime_version", [ip]), ("p2p_driver_version", [ip]),
        ("p2p_device_count", [ip]), ("p2p_set_device", [ci]), ("p2p_sync", []),
        ("p2p_mem_info", [c_szp, c_szp]),
        ("p2p_device_info", [ci, ctypes.c_char_p, ci, ctypes.c_char_p, ci,
                             ctypes.c_char_p, ci, ip, ip, ip, ull_p]),
        ("p2p_malloc_device", [vpp, c_sz]), ("p2p_free_device", [vp]),
        ("p2p_host_malloc", [vpp, c_sz, cu]), ("p2p_host_free", [vp]),
        ("p2p_host_dev_ptr", [vpp, vp]),
        ("p2p_memcpy", [vp, vp, c_sz, ci]), ("p2p_memset", [vp, ci, c_sz]),
        ("p2p_memcpy_bw", [vp, vp, c_sz, ci, ci, fp]),
        ("p2p_fill_seq", [vp, c_sz, c_sz, cu]),
        ("p2p_fill_hash", [vp, c_sz, c_sz, cu]),
        ("p2p_fill_half_uniform", [vp, c_sz, cu, cf, cf]),
        ("p2p_fill_float_uniform", [vp, c_sz, cu, cf, cf]),
        ("p2p_read_bw", [vp, c_sz, ci, ci, up, c_szp, fp]),
        ("p2p_thrash", [vp, c_sz]),
        ("p2p_set_tables", [vpp, vpp, vpp, ci]),
        ("p2p_get_table_entry", [ci, ull_p]),
        ("p2p_gemm1_silu", [ci, vp, vp, vp, vp, vp, vp, vp, vp,
                            ci, ci, ci, ci, ci, ci, ci, ci,
                            ci, ci, ci, ci, ci, ci, fp]),
        ("p2p_gemm2", [ci, ci, vp, vp, vp, vp, vp, vp, vp, vp, fp, fp,
                       ci, ci, ci, ci, ci, ci, ci, ci,
                       ci, ci, ci, ci, ci, fp]),
        ("p2p_select_tiling", [ci, ci, ci, ip, ip, ip]),
        ("p2p_pointer_attrs", [vp, ip, ip, vpp, vpp, ip, up]),
    ):
        getattr(lib, fn).argtypes = args
        getattr(lib, fn).restype = ci
    return lib


def K(lib: ctypes.CDLL, rc: int, what: str) -> None:
    if rc != 0:
        msg = lib.p2p_err(rc)
        _fail(f"{what} -> rc={rc} ({msg.decode() if msg else '?'})")


def pointer_attrs(lib, ptr: int) -> dict:
    mt, dv, mg = ctypes.c_int(-1), ctypes.c_int(-1), ctypes.c_int(-1)
    dp, hp = ctypes.c_void_p(), ctypes.c_void_p()
    fl = ctypes.c_uint(0)
    rc = lib.p2p_pointer_attrs(ctypes.c_void_p(ptr), ctypes.byref(mt), ctypes.byref(dv),
                               ctypes.byref(dp), ctypes.byref(hp), ctypes.byref(mg),
                               ctypes.byref(fl))
    if rc != 0:
        msg = lib.p2p_err(rc)
        return {"rc": rc, "error": msg.decode() if msg else "?"}
    return {"rc": 0, "memory_type": mt.value,
            "memory_type_name": HIP_MEMORY_TYPE.get(mt.value, f"?{mt.value}"),
            "device": dv.value,
            "device_pointer": f"0x{dp.value:x}" if dp.value else None,
            "host_pointer": f"0x{hp.value:x}" if hp.value else None,
            "is_managed": mg.value, "allocation_flags": fl.value,
            "caveat": ("a query can succeed with a fabricated value -- this whole probe family "
                       "exists because one did; cross-checked against VRAM/MemAvailable deltas "
                       "and achieved bandwidth, never trusted alone")}


def _vram_free(lib) -> int:
    f, t = ctypes.c_size_t(0), ctypes.c_size_t(0)
    K(lib, lib.p2p_mem_info(ctypes.byref(f), ctypes.byref(t)), "p2p_mem_info")
    return int(f.value)


def expected_prefix_sum(cover_bytes: int) -> int:
    """Checksum of a p2p_fill_seq(base=0) region read exactly once: sum_{i<M} i mod 2^32."""
    m = cover_bytes // 4
    return (m * (m - 1) // 2) & 0xFFFFFFFF


# ===========================================================================
# The two stacks, each with its placement PROVEN rather than declared.
# ===========================================================================

class Stack:
    """One expert stack: `nbytes` of contiguous memory plus the device VA a kernel reads it at.

    `media` is what the caller ASKED for.  `placement` is what the box actually did, measured three
    independent ways.  P2's whole failure was that those two were assumed equal.
    """

    def __init__(self, lib, media: str, nbytes: int, bdf: str | None):
        self.lib, self.media, self.nbytes, self.bdf = lib, media, nbytes, bdf
        self.host_ptr = 0        # CPU-addressable base (host stacks only)
        self.dev_ptr = 0         # the address a kernel dereferences
        self.placement: dict = {}
        self._freed = False

    def alloc(self) -> dict:
        lib = self.lib
        sys_before = card_mem(self.bdf)
        vram_before = _vram_free(lib)
        mem_before = _mem_available_bytes()
        p = ctypes.c_void_p()
        if self.media == "device":
            K(lib, lib.p2p_malloc_device(ctypes.byref(p), self.nbytes),
              f"hipMalloc({self.nbytes} B)")
            self.dev_ptr = int(p.value)
            self.host_ptr = 0
        elif self.media == "host":
            flags = HHM_MAPPED | HHM_PORTABLE
            K(lib, lib.p2p_host_malloc(ctypes.byref(p), self.nbytes, flags),
              f"hipHostMalloc({self.nbytes} B, Mapped|Portable)")
            self.host_ptr = int(p.value)
            d = ctypes.c_void_p()
            K(lib, lib.p2p_host_dev_ptr(ctypes.byref(d), ctypes.c_void_p(self.host_ptr)),
              "hipHostGetDevicePointer")
            self.dev_ptr = int(d.value)
        else:
            _fail(f"unknown stack media {self.media!r}")
        K(lib, lib.p2p_sync(), "sync after stack alloc")
        # Touch the whole region from the DEVICE before sampling the counters: a lazy allocator can
        # defer physical backing until first touch, and a placement classified before the pages
        # exist is a classification of nothing.
        K(lib, lib.p2p_fill_seq(ctypes.c_void_p(self.dev_ptr), 0, self.nbytes, 0),
          "first-touch fill of stack")
        sys_after = card_mem(self.bdf)
        vram_after = _vram_free(lib)
        mem_after = _mem_available_bytes()
        rec = classify_placement(self.nbytes, vram_before - vram_after, mem_before - mem_after,
                                 sys_before, sys_after)
        rec["requested_media"] = self.media
        rec["host_ptr"] = f"0x{self.host_ptr:x}" if self.host_ptr else None
        rec["device_ptr"] = f"0x{self.dev_ptr:x}"
        # Recorded as DATA, never as a gate.  ROCm has a unified virtual address space, so
        # hipHostGetDevicePointer normally hands back the SAME VA (P1 saw exactly this on the arm
        # that measured 28.93 GB/s).  A False here is expected and means nothing on its own; the
        # property that matters is the two stacks not aliasing, checked in gate_preconditions().
        rec["separate_va"] = bool(self.host_ptr and self.host_ptr != self.dev_ptr)
        rec["separate_va_note"] = ("informational only -- ROCm's unified VA makes host_ptr == "
                                   "device_ptr the normal case; do NOT gate on this")
        rec["pointer_attrs"] = pointer_attrs(lib, self.dev_ptr)
        rec["maps_entry"] = maps_entry_for(self.dev_ptr)
        # CPU accessibility, probed THROUGH THE KERNEL so a device-only VA reports EFAULT instead
        # of killing the process.  A real host stack is CPU-readable; a device stack must not be.
        rec["cpu_read_probe_at_device_ptr"] = cpu_readable(self.dev_ptr, 4096)
        if self.host_ptr:
            rec["cpu_read_probe_at_host_ptr"] = cpu_readable(self.host_ptr, 4096)
            # The write probe STORES into the region.  It restores, but a failed restore would
            # silently corrupt the very bytes the checksummed bandwidth arm is about to read from
            # the head of the stack -- turning a placement probe into a false "wrong checksum".
            # Run it on the LAST page instead, which no measurement covers.
            tail = self.host_ptr + self.nbytes - 4096
            rec["cpu_write_probe_at_host_ptr_tail"] = cpu_writable(tail, 4096, restore=True)
            rec["cpu_write_probe_offset"] = self.nbytes - 4096
        self.placement = rec
        return rec

    def free(self) -> None:
        if self._freed or not self.dev_ptr:
            return
        if self.media == "device":
            self.lib.p2p_free_device(ctypes.c_void_p(self.dev_ptr))
        else:
            self.lib.p2p_host_free(ctypes.c_void_p(self.host_ptr))
        self._freed = True

    def base_of(self, expert: int, stride: int) -> int:
        return self.dev_ptr + expert * stride


def measure_read_bw(lib, ptr: int, nbytes: int, blocks: int, threads: int, reps: int) -> dict:
    """Checksummed linear kernel read.  A WRONG CHECKSUM VOIDS THE BANDWIDTH NUMBER.

    The region must have been filled by p2p_fill_seq(base=0) so the sum over a full prefix is
    known in closed form.  Without this, a read that silently landed on the wrong physical pages
    (P6's remap defect does exactly that, with every call returning hipSuccess) would still produce
    a plausible GB/s.
    """
    out_sum = ctypes.c_uint(0)
    cover = ctypes.c_size_t(0)
    ms = ctypes.c_float(0.0)
    samples: list[float] = []
    sums: list[int] = []
    t_wall0 = time.perf_counter()
    for _ in range(reps):
        K(lib, lib.p2p_read_bw(ctypes.c_void_p(ptr), nbytes, blocks, threads,
                               ctypes.byref(out_sum), ctypes.byref(cover), ctypes.byref(ms)),
          "p2p_read_bw")
        samples.append(float(ms.value))
        sums.append(int(out_sum.value))
    wall = time.perf_counter() - t_wall0
    exp = expected_prefix_sum(int(cover.value))
    bad = [s for s in sums if s != exp]
    st = stats(samples)
    gbps = (int(cover.value) / (st["median"] * 1e-3)) / 1e9 if st.get("median") else None
    wall_gbps = (int(cover.value) * reps / wall) / 1e9 if wall > 0 else None
    return {
        "cover_bytes": int(cover.value), "reps": reps,
        "ms": st, "gbps": gbps, "wall_gbps": wall_gbps,
        "checksum_expected": exp, "checksum_observed": sums[:4],
        "checksum_ok": not bad,
        "event_vs_wall_ratio": (gbps / wall_gbps) if (gbps and wall_gbps) else None,
    }


def measure_copy_engine(lib, dst_dev: int, src_host: int, nbytes: int, reps: int,
                        bdf: str | None) -> dict:
    """H2D copy-engine bandwidth -- the DENOMINATOR every host figure is judged against.

    Sampled with the PCIe link watched mid-DMA, never at idle: these root ports downtrain to
    2.5 GT/s at rest, so an at-rest reading would sit next to a 14 GB/s measurement claiming a
    ~2 GB/s link.  P4 established that card 1's root port trains Gen4 x8 against card 0's Gen5 x8
    (78/78 mid-DMA samples), which is why this is measured per card and never assumed symmetric.
    """
    arr = (ctypes.c_float * reps)()
    with LinkSampler(bdf) as sampler:
        K(lib, lib.p2p_memcpy_bw(ctypes.c_void_p(dst_dev), ctypes.c_void_p(src_host),
                                 nbytes, 1, reps, arr), "p2p_memcpy_bw H2D")
    st = stats([float(x) for x in arr])
    return {
        "bytes": nbytes, "reps": reps, "ms": st,
        "gbps": (nbytes / (st["median"] * 1e-3)) / 1e9 if st.get("median") else None,
        "link_under_load": sampler.result(),
    }


def gate_preconditions(dev_stack: Stack, host_stack: Stack, bw_dev: dict, bw_host: dict,
                       copy_engine: dict) -> dict:
    """Every check that must pass before a single timing number is allowed to mean anything.

    This block IS the probe's answer to why P2 is not being repeated: none of these were run
    before, and each one independently catches the failure that killed it.
    """
    checks: list[dict] = []

    def chk(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    dp, hp = dev_stack.placement, host_stack.placement

    chk("device_stack_consumes_vram",
        dp.get("classification") == "device_resident",
        f"classification={dp.get('classification')} via {dp.get('classification_source')}; "
        f"sysfs vram delta frac={dp.get('sysfs_vram_used_delta_frac')}")

    chk("host_stack_not_in_vram",
        (hp.get("sysfs_vram_used_delta_frac") is None
         or hp["sysfs_vram_used_delta_frac"] < PLACE_VRAM_HOST_FRAC),
        f"sysfs vram delta frac={hp.get('sysfs_vram_used_delta_frac')} "
        f"(must be < {PLACE_VRAM_HOST_FRAC}); THIS is the check P2 never made")

    # A host stack must be ORDINARY ANONYMOUS host memory.  /proc/self/maps settles this per
    # allocation with no shared-box noise: real pinned host pages map anonymous private
    # ("rw-p 00000000 00:00 0"), whereas P2's fake-host VMM pages mapped the device BAR
    # ("rw-s ... /dev/dri/renderD12N") -- exactly what the device stack still shows here.  This is
    # the card-local, non-statistical form of the evidence MemAvailable was standing in for.
    h_maps = (hp.get("maps_entry") or "")
    d_maps = (dp.get("maps_entry") or "")
    anon_ok = bool(h_maps) and "/dev/dri/" not in h_maps and "rw-p" in h_maps
    chk("host_stack_is_anonymous_host_mapping",
        anon_ok,
        f"host maps_entry={h_maps!r} (must be anonymous private rw-p, NOT /dev/dri/*); the device "
        f"stack for contrast maps {d_maps!r}")

    # CORROBORATING ONLY -- deliberately NOT a gate.  MemAvailable is a global kernel heuristic on
    # a box shared with dozens of services and ~18 GB of swap in use; concurrent reclaim easily
    # masks half a gigabyte over the sampling window.  It is also non-discriminating: P2's fake
    # host pages are already caught three independent ways above (not_in_vram, cpu_accessible,
    # anonymous_mapping) plus media_separation_ratio below.  Gating on it cost a real run --
    # card 1 measured 0.468 against a 0.5 threshold while its host stack read at 14.47 GB/s
    # (PCIe Gen4 x8-bounded, matching P1's independent 14.48) with a 27x media separation, and the
    # same allocation on card 0 cleared the threshold.  A shared-box estimate must not be able to
    # veto four direct measurements, so this is recorded and surfaced, never fatal.
    mem_frac = hp.get("memavailable_consumed_frac")
    checks.append({
        "check": "host_stack_consumes_host_ram",
        "ok": True,
        "advisory": True,
        "corroborated": bool((mem_frac or 0) >= PLACE_MEMAVAIL_HOST_FRAC),
        "detail": (f"MemAvailable delta frac={mem_frac} (corroborating target >= "
                   f"{PLACE_MEMAVAIL_HOST_FRAC}). ADVISORY ONLY on a shared box -- it does not "
                   f"gate. Host residency is established by host_stack_not_in_vram, "
                   f"host_stack_is_cpu_accessible, host_stack_is_anonymous_host_mapping and "
                   f"media_separation_ratio, all of which are card-local or direct."),
    })

    chk("host_stack_is_cpu_accessible",
        bool((hp.get("cpu_read_probe_at_host_ptr") or {}).get("readable")
             and (hp.get("cpu_write_probe_at_host_ptr_tail") or {}).get("writable")),
        f"cpu read at host ptr: {hp.get('cpu_read_probe_at_host_ptr')}; P1 found the fake-host "
        f"VMM pages were NOT CPU-accessible (---s on renderD128, os.write -> EFAULT)")

    # NOTE (2026-09-03): this check used to demand host_ptr != device_ptr WITHIN the host stack.
    # That premise is false on ROCm, which uses a UNIFIED virtual address space:
    # hipHostGetDevicePointer on hipHostMalloc'd memory legitimately returns the SAME VA, and P1
    # recorded exactly that (host_ptr == device_ptr == 0x7fe0d4a00000, same_va=True) for the very
    # arm that measured 28.93 GB/s and passed coherence in both directions.  Worse, the old check
    # had ZERO discriminating power: P2's fake-host VMM pages also had host_ptr == dev_ptr, so it
    # fired identically on the good mechanism and the bad one.  What actually must hold -- and what
    # would genuinely corrupt the miss curve if violated -- is that the two stacks do not ALIAS:
    # if the host stack's device-visible range overlapped the device stack's, a kernel indexing a
    # "host" expert would silently read device bytes and every miss would be scored as a hit.
    d_lo, d_hi = dev_stack.dev_ptr, dev_stack.dev_ptr + dev_stack.nbytes
    h_lo, h_hi = host_stack.dev_ptr, host_stack.dev_ptr + host_stack.nbytes
    overlap = max(0, min(d_hi, h_hi) - max(d_lo, h_lo))
    chk("stacks_do_not_alias",
        bool(d_lo and h_lo and overlap == 0),
        f"device VA [0x{d_lo:x},0x{d_hi:x}) vs host device-visible VA [0x{h_lo:x},0x{h_hi:x}); "
        f"overlap={overlap} B (must be 0). host_ptr={hp.get('host_ptr')} "
        f"device_ptr={hp.get('device_ptr')} (equal is NORMAL under ROCm's unified VA -- see P1); "
        f"the two stacks are separate ALLOCATIONS on different media and cannot be interleaved, "
        f"which is what makes this a two-stack design")

    chk("read_checksums_valid",
        bool(bw_dev.get("checksum_ok")) and bool(bw_host.get("checksum_ok")),
        f"device ok={bw_dev.get('checksum_ok')} host ok={bw_host.get('checksum_ok')}; a wrong "
        f"checksum means the read did not land on the pages we think it did")

    d_gbps, h_gbps = bw_dev.get("gbps"), bw_host.get("gbps")
    ce_gbps = copy_engine.get("gbps")
    ratio = (d_gbps / h_gbps) if (d_gbps and h_gbps) else None
    chk("media_separation_ratio",
        bool(ratio and ratio >= MIN_MEDIA_RATIO),
        f"device {d_gbps:.1f} / host {h_gbps:.1f} = {ratio:.2f}x (need >= {MIN_MEDIA_RATIO}x). "
        f"P2 measured 1.00x here and that is what proved the 'host' pages were VRAM"
        if (d_gbps and h_gbps) else f"device={d_gbps} host={h_gbps}")

    chk("device_read_is_hbm_speed",
        bool(d_gbps and d_gbps >= DEVICE_BW_MIN_GBPS),
        f"{d_gbps} GB/s (need >= {DEVICE_BW_MIN_GBPS})")

    chk("host_read_is_pcie_bounded",
        bool(h_gbps and ce_gbps and HOST_BW_MIN_GBPS <= h_gbps <= ce_gbps * HOST_BW_MAX_OVER_COPY),
        f"host {h_gbps} GB/s vs copy engine {ce_gbps} GB/s x {HOST_BW_MAX_OVER_COPY}; bytes that "
        f"cross PCIe cannot materially outrun the copy engine on the same card")

    ok = all(c["ok"] for c in checks)
    return {
        "passed": ok,
        "checks": checks,
        "failed": [c["check"] for c in checks if not c["ok"]],
        # An advisory that did not corroborate is reported loudly rather than buried: it does not
        # invalidate the run, but a reader deciding M2 on these numbers should see it.
        "advisories_not_corroborated": [c["check"] for c in checks
                                        if c.get("advisory") and not c.get("corroborated")],
        "device_read_gbps": d_gbps, "host_read_gbps": h_gbps,
        "copy_engine_gbps": ce_gbps, "media_separation_ratio": ratio,
        "pcie_ceiling_gbps": max([g for g in (ce_gbps, h_gbps) if g], default=None),
    }


# ===========================================================================
# The measurement harness for one card.
# ===========================================================================

KERNEL_ARMS = ("gemm1_silu", "gemm2")


class Buffers:
    """Everything that is not an expert stack: activations, outputs, route, thrash scratch."""

    def __init__(self, lib, cfg: dict, lay: dict, P: int):
        self.lib, self.cfg, self.lay, self.P = lib, cfg, lay, P
        self.ptrs: dict[str, int] = {}
        T, hidden, inter, top_k = cfg["tokens"], cfg["hidden"], cfg["inter"], cfg["top_k"]
        self.sizes = {
            # gemm1 A is indexed by TOKEN (s_src = offs/top_k), gemm2-scatter A by PADDED ROW
            # (s_src = row0+r) -- different shapes, so x2 is allocated at the larger P rows and the
            # non-scatter gemm2 arm simply reads its first T rows.
            "x1": T * hidden * 2,
            "x2": P * inter * 2,
            "out1": P * inter * 2,
            "out2": P * hidden * 2,
            "out_scatter": T * hidden * 4,
            "topk_w": T * top_k * 4,
            "sti": P * 4,
            "eid": max(P // cfg["block_m"], 1) * 4,
            "ntp": 4,
            "thrash": cfg["thrash_mb"] * (1 << 20),
        }

    def alloc(self) -> None:
        for name, sz in self.sizes.items():
            p = ctypes.c_void_p()
            K(self.lib, self.lib.p2p_malloc_device(ctypes.byref(p), sz),
              f"hipMalloc({name}, {sz} B)")
            self.ptrs[name] = int(p.value)
        # Bounded, finite activations and topk weights.  A hash fill reinterpreted as fp16 makes
        # inf/NaN, and one NaN turns every downstream bit-comparison into noise.
        K(self.lib, self.lib.p2p_fill_half_uniform(
            ctypes.c_void_p(self.ptrs["x1"]), self.sizes["x1"] // 2, 0xA1, -1.0, 1.0), "fill x1")
        K(self.lib, self.lib.p2p_fill_half_uniform(
            ctypes.c_void_p(self.ptrs["x2"]), self.sizes["x2"] // 2, 0xA2, -1.0, 1.0), "fill x2")
        K(self.lib, self.lib.p2p_fill_float_uniform(
            ctypes.c_void_p(self.ptrs["topk_w"]), self.sizes["topk_w"] // 4, 0xA3, 0.0, 1.0),
          "fill topk_weights")
        K(self.lib, self.lib.p2p_fill_hash(
            ctypes.c_void_p(self.ptrs["thrash"]), 0, self.sizes["thrash"], 0xA4), "fill thrash")

    def free(self) -> None:
        for p in self.ptrs.values():
            self.lib.p2p_free_device(ctypes.c_void_p(p))
        self.ptrs.clear()

    def upload_route(self, route: dict) -> None:
        sti = (ctypes.c_int * len(route["sorted_token_ids"]))(*route["sorted_token_ids"])
        eid = (ctypes.c_int * len(route["expert_ids"]))(*route["expert_ids"])
        ntp = (ctypes.c_int * 1)(route["num_tokens_post_padded"])
        for name, arr in (("sti", sti), ("eid", eid), ("ntp", ntp)):
            K(self.lib, self.lib.p2p_memcpy(ctypes.c_void_p(self.ptrs[name]),
                                            ctypes.cast(arr, ctypes.c_void_p),
                                            ctypes.sizeof(arr), 1), f"upload {name}")

    def thrash(self) -> None:
        K(self.lib, self.lib.p2p_thrash(ctypes.c_void_p(self.ptrs["thrash"]),
                                        self.sizes["thrash"]), "p2p_thrash")

    def readback(self, name: str) -> bytes:
        n = self.sizes[name]
        buf = ctypes.create_string_buffer(n)
        K(self.lib, self.lib.p2p_memcpy(ctypes.cast(buf, ctypes.c_void_p),
                                        ctypes.c_void_p(self.ptrs[name]), n, 2),
          f"readback {name}")
        return buf.raw

    def zero(self, name: str) -> None:
        K(self.lib, self.lib.p2p_memset(ctypes.c_void_p(self.ptrs[name]), 0, self.sizes[name]),
          f"zero {name}")


def fill_stack_weights(lib, stack: Stack, cfg: dict, lay: dict, seed: int) -> dict:
    """Give a stack deterministic, well-conditioned contents.

    Packed weights and zero-points get a splitmix32 stream over the global dword index, so every
    expert's granule differs from every other's for free.  Scales are filled separately with a
    BOUNDED fp16 uniform: a hash pattern reinterpreted as fp16 yields inf/NaN, and a single NaN
    would make the bit-exact provenance comparison meaningless.

    `seed` selects the content stream.  The provenance arm gives the two stacks DIFFERENT seeds
    (so reading the wrong stack is visible); the timing arm gives them the SAME seed (so every
    placement does numerically identical work and the only variable left is the media).
    """
    off, sz = lay["offsets"], lay["sizes"]
    E, stride = cfg["num_experts"], lay["granule_stride"]
    K(lib, lib.p2p_fill_hash(ctypes.c_void_p(stack.dev_ptr), 0, stack.nbytes, seed & 0xFFFFFFFF),
      "hash-fill stack")
    n_scale_fills = 0
    for e in range(E):
        base = stack.base_of(e, stride)
        # Component salt is a FIXED constant, never Python's hash(): PYTHONHASHSEED randomises
        # str hashing per process, so a hash()-derived seed would make the stack contents
        # irreproducible across runs -- and reproducible fixtures are the point.
        for comp, salt, lo, hi in (("s13", 0x5131, 0.002, 0.02), ("s2", 0x5232, 0.002, 0.02)):
            if off[comp] is None:
                continue
            K(lib, lib.p2p_fill_half_uniform(ctypes.c_void_p(base + off[comp]), sz[comp] // 2,
                                             (seed ^ (e * 0x9E3779B1) ^ salt) & 0xFFFFFFFF,
                                             lo, hi),
              f"fill {comp}[{e}]")
            n_scale_fills += 1
    K(lib, lib.p2p_sync(), "sync after stack fill")
    return {"seed": seed, "scale_fills": n_scale_fills,
            "note": "packed weights/zeros = splitmix32(index); scales = bounded fp16 uniform"}


def table_for(kernel: str, cfg: dict, lay: dict, dev: Stack, host: Stack,
              host_experts) -> tuple:
    """Build the three per-expert base-pointer arrays for one kernel arm.

    All three are built together and uploaded together.  A granule is one expert's slice of EVERY
    tensor, so weights, scales and zero-points must move as a unit -- splitting them dequantizes
    one expert against another's scale, which produces plausible text and no crash.
    """
    off = lay["offsets"]
    stride = lay["granule_stride"]
    wname, sname, zname = (("w13", "s13", "z13") if kernel == "gemm1_silu" else ("w2", "s2", "z2"))
    E = cfg["num_experts"]
    hs = set(host_experts)
    wq = (ctypes.c_void_p * E)()
    ws = (ctypes.c_void_p * E)()
    wz = (ctypes.c_void_p * E)()
    for e in range(E):
        src = host if e in hs else dev
        base = src.base_of(e, stride)
        wq[e] = ctypes.c_void_p(base + off[wname])
        ws[e] = ctypes.c_void_p(base + off[sname])
        wz[e] = ctypes.c_void_p(base + off[zname]) if off[zname] is not None else ctypes.c_void_p(0)
    return wq, ws, wz


def upload_tables(lib, tabs: tuple, E: int) -> None:
    wq, ws, wz = tabs
    K(lib, lib.p2p_set_tables(wq, ws, wz, E), "p2p_set_tables")


def verify_table_entry(lib, tabs: tuple, e: int) -> dict:
    """Read one uploaded table slot back THROUGH A KERNEL.

    hipMemcpyToSymbol returning hipSuccess is not evidence that the constant landed -- that is the
    exact class of lie (P3: hipSuccess with the wrong placement; P6: hipSuccess with the wrong
    page) this probe family exists to catch.
    """
    out = (ctypes.c_ulonglong * 3)()
    K(lib, lib.p2p_get_table_entry(e, out), "p2p_get_table_entry")
    wq, ws, wz = tabs
    want = (int(wq[e] or 0), int(ws[e] or 0), int(wz[e] or 0))
    got = (int(out[0]), int(out[1]), int(out[2]))
    return {"expert": e, "expected": [f"0x{x:x}" for x in want],
            "observed": [f"0x{x:x}" for x in got], "match": want == got}


def tiling_for(lib, kernel: str, cfg: dict, lay: dict) -> dict:
    """Pick the launch tiling production would pick for this shape.

    gemm1+SiLU defers to the SHARED host-side table `gemv_decode::select_gemv_tiling` (the same
    call `launch_mmq_regdirect_w4a16_moe_gemv_silu_gfx1201` makes, with the same env prefix, so a
    served override is inherited rather than silently diverged from).  gemm2 replicates
    `run_w4a16_moe_gemv`'s own ladder, which is what the shipped scatter launcher uses -- copying
    the table it does NOT use would measure a tiling nothing runs.
    """
    T, block_m, group = cfg["tokens"], cfg["block_m"], cfg["group_size"]
    mreal_cap = min(block_m, T)
    if kernel == "gemm1_silu":
        k = lay["k1"]
        bl, nw, cols = ctypes.c_int(0), ctypes.c_int(0), ctypes.c_int(0)
        K(lib, lib.p2p_select_tiling(k, mreal_cap, group, ctypes.byref(bl), ctypes.byref(nw),
                                     ctypes.byref(cols)), "p2p_select_tiling")
        bylane, nwarps = int(bl.value), int(nw.value)
        # SILU fuses gate|up into ONE output column, so the grid runs over `inter`, and the shipped
        # launcher pins COLS=1 regardless of what the table returns for the plain gemv.
        per_block = nwarps * 32 if bylane else nwarps
        inter = cfg["inter"]
        grid_x = (inter + per_block - 1) // per_block
        return {"kernel": kernel, "source": "gemv_decode::select_gemv_tiling (shared table)",
                "bylane": bylane, "nwarps": nwarps, "cols": 1, "table_cols_ignored": int(cols.value),
                "grid_x": grid_x, "per_block": per_block, "mreal_cap": mreal_cap,
                "n_columns": inter, "K": k}
    k = lay["k2"]
    nw = 8 if k <= 512 else 16
    if k <= 1024:
        cols = 4 if mreal_cap <= 8 else (2 if mreal_cap <= 16 else 1)
    else:
        cols = 2 if mreal_cap <= 4 else 1
    env_nw = os.environ.get("VLLM_W4A16_MOE_GEMV_NWARPS")
    env_co = os.environ.get("VLLM_W4A16_MOE_GEMV_COLS")
    if env_nw:
        nw = int(env_nw)
        if nw not in (4, 8, 16, 32):
            nw = 8 if k <= 512 else 16
    if env_co:
        cols = int(env_co)
        if cols not in (1, 2, 4, 8):
            cols = 1
    per_block = nw * cols
    n_cols = lay["n2"]
    return {"kernel": kernel, "source": "run_w4a16_moe_gemv ladder (shipped scatter launcher)",
            "bylane": 0, "nwarps": nw, "cols": cols, "grid_x": (n_cols + per_block - 1) // per_block,
            "per_block": per_block, "mreal_cap": mreal_cap, "n_columns": n_cols, "K": k,
            "env_overrides": {"NWARPS": env_nw, "COLS": env_co},
            "k_on_lanes_underfilled": k < 1024}


def launch_raw(lib, kernel: str, use_table: bool, cfg: dict, lay: dict, buf: Buffers,
               route: dict, tl: dict, w_base: int, s_base: int, z_base: int,
               reps: int, scatter: bool = False) -> list[float]:
    """One timed burst of `reps` launches of the SHIPPED core; returns per-launch hipEvent ms.

    `w_base`/`s_base`/`z_base` are the STOCK loader's component-major bases and are dereferenced
    only when `use_table` is false.  The table path passes zeros for them on purpose: if the table
    were somehow ignored, the run faults immediately on a null base instead of quietly reading a
    valid stack and reporting a device-speed "host" number.  A loud crash beats a plausible lie --
    that is the whole lesson of P2.
    """
    arr = (ctypes.c_float * reps)()
    grid_y = route["P"] // cfg["block_m"]
    p = buf.ptrs
    if kernel == "gemm1_silu":
        rc = lib.p2p_gemm1_silu(
            1 if use_table else 0,
            ctypes.c_void_p(p["x1"]), ctypes.c_void_p(w_base), ctypes.c_void_p(s_base),
            ctypes.c_void_p(z_base),
            ctypes.c_void_p(p["sti"]), ctypes.c_void_p(p["eid"]), ctypes.c_void_p(p["ntp"]),
            ctypes.c_void_p(p["out1"]),
            cfg["tokens"], lay["n1"], lay["k1"], cfg["group_size"], cfg["top_k"], cfg["block_m"],
            route["num_valid_tokens"], 1 if cfg["e2m1"] else 0,
            tl["nwarps"], tl["cols"], tl["bylane"], tl["grid_x"], grid_y, reps, arr)
        K(lib, rc, "p2p_gemm1_silu")
    elif kernel == "gemm2":
        rc = lib.p2p_gemm2(
            1 if use_table else 0, 1 if scatter else 0,
            ctypes.c_void_p(p["x2"]), ctypes.c_void_p(w_base), ctypes.c_void_p(s_base),
            ctypes.c_void_p(z_base),
            ctypes.c_void_p(p["sti"]), ctypes.c_void_p(p["eid"]), ctypes.c_void_p(p["ntp"]),
            ctypes.c_void_p(p["out2"]),
            ctypes.cast(ctypes.c_void_p(p["out_scatter"]), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(ctypes.c_void_p(p["topk_w"]), ctypes.POINTER(ctypes.c_float)),
            cfg["tokens"], lay["n2"], lay["k2"], cfg["group_size"], cfg["top_k"], cfg["block_m"],
            route["num_valid_tokens"], 1 if cfg["e2m1"] else 0,
            tl["nwarps"], tl["cols"], tl["grid_x"], grid_y, reps, arr)
        K(lib, rc, "p2p_gemm2")
    else:
        _fail(f"unknown kernel arm {kernel!r}")
    return [float(x) for x in arr]


def launch(lib, kernel: str, use_table: bool, cfg: dict, lay: dict, buf: Buffers,
           route: dict, tl: dict, scatter: bool, reps: int) -> list[float]:
    """Table-path launch: the per-expert bases come from the uploaded pointer table.

    `w`/`s` are passed NULL on purpose (see launch_raw).  `z` cannot be: the loader uses it as a
    FLAG rather than a base -- `wz_expert(z, e)` returns `c_wz_tab[e]` when z is non-null and
    `nullptr` when it is null.  Passing 0 therefore silently DISABLES zero-points on the whole
    table path: the zp planes are never read, the ~0.75% of granule bytes they contribute vanish
    from every timing point, and the scales-AND-ZEROS-travel-with-the-weights contract that
    `--zeros` exists to exercise is not exercised at all.  It is never dereferenced in table mode,
    so a non-null sentinel is both safe and truthful, and it makes the table loader bit-identical
    to the stock loader in indirection_control() instead of silently diverging from it.
    """
    z_flag = Z_ENABLED_SENTINEL if (use_table and cfg["zeros"]) else 0
    return launch_raw(lib, kernel, use_table, cfg, lay, buf, route, tl, 0, 0, z_flag, reps, scatter)


# ---------------------------------------------------------------------------
# The indirection A/B control.
# ---------------------------------------------------------------------------

class StockMirror:
    """A small COMPONENT-MAJOR device mirror, so the STOCK loader can be run as the A/B baseline.

    The shipped `Int4A16GemvLoader` computes `w + e*N*(K/8)` and `s + e*G*N`, i.e. it assumes each
    tensor is one contiguous (E, ...) array.  The two-stack granule layout is expert-major (one
    expert's whole slice contiguous) because that is what makes "this expert lives on the host
    stack" a single pointer.  The two layouts are incompatible by construction, so the baseline
    cannot simply reuse the granule stack.

    Rather than emulate the stock path (the repo rule: an A/B baseline must be the OLD CODE -- an
    emulated baseline once turned a claimed -50% into a real -9.6%), this allocates a genuine
    component-major mirror covering only the `n_ctl` experts the control routes to (10 experts is
    ~23 MiB, not 1.2 GiB) and runs BOTH legs on it:
        stock leg -- unmodified loader, base + e*stride
        table leg -- TwoStackInt4A16GemvLoader with the table filled from the SAME addresses
    Identical bytes, identical addresses, identical tiling; the only difference is how the
    per-expert base was obtained.  The delta is therefore the cost of the indirection itself and
    nothing else.
    """

    def __init__(self, lib, cfg: dict, lay: dict, n_ctl: int):
        self.lib, self.cfg, self.lay, self.n_ctl = lib, cfg, lay, n_ctl
        sz = lay["sizes"]
        self.strides = {
            "w13": sz["w13"], "s13": sz["s13"], "z13": sz["z13"],
            "w2": sz["w2"], "s2": sz["s2"], "z2": sz["z2"],
        }
        self.ptrs: dict[str, int] = {}
        self.bytes = 0

    def alloc(self) -> dict:
        for name, stride in self.strides.items():
            if stride == 0:
                self.ptrs[name] = 0
                continue
            n = stride * self.n_ctl
            p = ctypes.c_void_p()
            K(self.lib, self.lib.p2p_malloc_device(ctypes.byref(p), n),
              f"hipMalloc(stock mirror {name}, {n} B)")
            self.ptrs[name] = int(p.value)
            self.bytes += n
            if name.startswith("s"):
                K(self.lib, self.lib.p2p_fill_half_uniform(p, n // 2, 0xC0DE ^ stride,
                                                           0.002, 0.02), f"fill mirror {name}")
            else:
                K(self.lib, self.lib.p2p_fill_hash(p, 0, n, 0xBEEF ^ stride),
                  f"fill mirror {name}")
        K(self.lib, self.lib.p2p_sync(), "sync after mirror fill")
        return {"experts": self.n_ctl, "bytes": self.bytes,
                "layout": "component-major (E, ...) -- the shipped loader's stride contract"}

    def free(self) -> None:
        for p in self.ptrs.values():
            if p:
                self.lib.p2p_free_device(ctypes.c_void_p(p))
        self.ptrs.clear()

    def bases(self, kernel: str) -> tuple:
        w, sname, z = (("w13", "s13", "z13") if kernel == "gemm1_silu" else ("w2", "s2", "z2"))
        return self.ptrs[w], self.ptrs[sname], self.ptrs.get(z, 0)

    def table(self, kernel: str, num_experts: int) -> tuple:
        """Per-expert tables that reproduce the stock loader's arithmetic EXACTLY."""
        wn, sn, zn = (("w13", "s13", "z13") if kernel == "gemm1_silu" else ("w2", "s2", "z2"))
        wq = (ctypes.c_void_p * num_experts)()
        ws = (ctypes.c_void_p * num_experts)()
        wz = (ctypes.c_void_p * num_experts)()
        for e in range(num_experts):
            i = e % self.n_ctl          # experts outside the control route are never dereferenced
            wq[e] = ctypes.c_void_p(self.ptrs[wn] + i * self.strides[wn])
            ws[e] = ctypes.c_void_p(self.ptrs[sn] + i * self.strides[sn])
            wz[e] = (ctypes.c_void_p(self.ptrs[zn] + i * self.strides[zn])
                     if self.ptrs.get(zn) else ctypes.c_void_p(0))
        return wq, ws, wz


def indirection_control(lib, cfg: dict, lay: dict, buf: Buffers, reps: int) -> dict:
    """Measure what the pointer table itself costs, before believing any curve measured through it.

    One block-uniform `__constant__` load per workgroup should be free, but "should be" is not a
    measurement, and if it were NOT free every miss-count point would carry a constant offset that
    flattens `cliff_index` toward 1.0 -- i.e. the instrument would manufacture the very answer the
    prior expects.  Reported as a fraction; the run flags (not aborts) above
    MAX_INDIRECTION_OVERHEAD, because the honest response to a costly indirection is to widen the
    error bars on the shape, not to discard the run.
    """
    E = cfg["num_experts"]
    n_ctl = cfg["tokens"] * cfg["top_k"]
    mirror = StockMirror(lib, cfg, lay, n_ctl)
    info = mirror.alloc()
    out: dict = {"mirror": info, "arms": {}}
    try:
        experts = list(range(n_ctl))          # the mirror only covers these
        route = build_route(cfg["tokens"], cfg["top_k"], E, cfg["block_m"], experts)
        buf.upload_route(route)
        for kernel in KERNEL_ARMS:
            tl = tiling_for(lib, kernel, cfg, lay)
            wb, sb, zb = mirror.bases(kernel)
            out_name = "out1" if kernel == "gemm1_silu" else "out2"
            stock_ms, table_ms, same = [], [], None
            for i in range(reps + 1):         # rep 0 is a discarded warm-up
                buf.thrash()
                buf.zero(out_name)
                s_ms = launch_raw(lib, kernel, False, cfg, lay, buf, route, tl, wb, sb, zb, 1)
                stock_out = buf.readback(out_name)
                upload_tables(lib, mirror.table(kernel, E), E)
                buf.thrash()
                buf.zero(out_name)
                # z must carry the same ENABLE flag the stock leg got (zb non-null), or the two
                # legs are not comparable: the table leg would skip zero-points entirely and
                # `bit_identical_output` would report a divergence this probe itself created.
                t_ms = launch_raw(lib, kernel, True, cfg, lay, buf, route, tl, 0, 0,
                                  (Z_ENABLED_SENTINEL if zb else 0), 1)
                table_out = buf.readback(out_name)
                if i == 0:
                    same = (stock_out == table_out)
                    continue
                stock_ms += s_ms
                table_ms += t_ms
            st_s, st_t = stats(stock_ms), stats(table_ms)
            over = ((st_t["median"] - st_s["median"]) / st_s["median"]) if st_s["median"] else None
            out["arms"][kernel] = {
                "tiling": tl,
                "stock_loader_ms": st_s,
                "table_loader_ms": st_t,
                "indirection_overhead_frac": over,
                "bit_identical_output": same,
                "within_tolerance": (over is not None and abs(over) <= MAX_INDIRECTION_OVERHEAD),
                "baseline_is_old_code": True,
            }
    finally:
        mirror.free()
    overs = [a["indirection_overhead_frac"] for a in out["arms"].values()
             if a["indirection_overhead_frac"] is not None]
    out["max_overhead_frac"] = max(overs, default=None)
    out["ok"] = all(a["within_tolerance"] for a in out["arms"].values()) if out["arms"] else False
    out["bit_identical"] = all(a["bit_identical_output"] for a in out["arms"].values())
    return out


def gemm_block_row_bytes(kernel: str, cfg: dict, lay: dict) -> int:
    """Bytes of output owned by ONE expert-block, used to assemble the expected mixed output.

    Block b writes rows [b*block_m, (b+1)*block_m) of the padded output (`out[s_pad[r]*N + nc]`,
    s_pad = row0 + r), and nothing else touches those rows, so a bit-exact expected output can be
    assembled from the all-device and all-host references with a slice per block.  That is what
    makes the provenance check exact and free -- no third stack, no tolerance.
    """
    n_out = cfg["inter"] if kernel == "gemm1_silu" else lay["n2"]
    return cfg["block_m"] * n_out * 2


def assemble_expected(ref_dev: bytes, ref_host: bytes, route: dict, host_experts,
                      row_bytes: int) -> bytes:
    hs = set(host_experts)
    out = bytearray(ref_dev)
    for b, e in enumerate(route["block_expert"]):
        if e in hs:
            lo, hi = b * row_bytes, (b + 1) * row_bytes
            out[lo:hi] = ref_host[lo:hi]
    return bytes(out)


def correctness_arm(lib, cfg: dict, lay: dict, buf: Buffers, dev: Stack, host: Stack,
                    rng: random.Random) -> dict:
    """Prove the pointer table is HONOURED, not merely accepted.

    The two stacks hold DIFFERENT content here.  Three runs per kernel arm:
      ref_dev   every table entry -> device stack
      ref_host  every table entry -> host stack        (must DIFFER from ref_dev, else the test is
                                                        vacuous and the probe says so)
      mixed     m entries -> host
    and `mixed` must equal, BIT FOR BIT, the per-block assembly of ref_dev/ref_host.  This fails
    if the table is ignored (mixed == ref_dev), if it is mis-indexed (an expert reads another
    expert's granule), and if scales/zeros were left behind on the other stack (the granule would
    dequantize against the wrong scale -- plausible numbers, no crash, which is the failure mode
    this arm exists for).

    Only the NON-scatter arms are bit-compared: the scatter epilogue accumulates with atomicAdd, so
    its reduction order varies run to run and it is not bit-reproducible by construction.  It runs
    on the identical core and loader, so its correctness is inherited; it is timing-only here and
    that is stated rather than assumed.
    """
    results: dict = {}
    E = cfg["num_experts"]
    for kernel in KERNEL_ARMS:
        tl = tiling_for(lib, kernel, cfg, lay)
        experts = draw_route_experts(rng, cfg["tokens"], cfg["top_k"], E, cfg["route_mode"])
        route = build_route(cfg["tokens"], cfg["top_k"], E, cfg["block_m"], experts)
        buf.upload_route(route)
        out_name = "out1" if kernel == "gemm1_silu" else "out2"
        row_bytes = gemm_block_row_bytes(kernel, cfg, lay)

        def run(host_experts) -> bytes:
            upload_tables(lib, table_for(kernel, cfg, lay, dev, host, host_experts), E)
            buf.zero(out_name)
            launch(lib, kernel, True, cfg, lay, buf, route, tl, scatter=False, reps=1)
            return buf.readback(out_name)

        ref_dev = run([])
        ref_host = run(list(range(E)))
        arm: dict = {
            "tiling": tl,
            "n_blocks": route["n_blocks"],
            "references_differ": ref_dev != ref_host,
            "cases": [],
        }
        if ref_dev == ref_host:
            arm["status"] = "VACUOUS"
            arm["detail"] = ("the all-device and all-host references are byte-identical, so a "
                             "mixed run cannot distinguish which stack it read. Either the two "
                             "stacks were filled with the same content or the reads are not "
                             "landing where the table says.")
            results[kernel] = arm
            continue
        misses = sorted({1, max(1, route["n_blocks"] // 2), route["n_blocks"] - 1})
        ok = True
        for m in misses:
            host_experts, achieved = choose_host_blocks(rng, route, m)
            tabs = table_for(kernel, cfg, lay, dev, host, host_experts)
            probe_e = host_experts[0] if host_experts else 0
            entry = verify_table_entry(lib, tabs, probe_e)
            upload_tables(lib, tabs, E)
            entry_after = verify_table_entry(lib, tabs, probe_e)
            buf.zero(out_name)
            launch(lib, kernel, True, cfg, lay, buf, route, tl, scatter=False, reps=1)
            got = buf.readback(out_name)
            want = assemble_expected(ref_dev, ref_host, route, host_experts, row_bytes)
            match = (got == want)
            ok = ok and match and entry_after["match"]
            arm["cases"].append({
                "requested_misses": m, "achieved_misses": achieved,
                "host_experts": host_experts[:16],
                "bit_exact_vs_assembled_reference": match,
                "differs_from_all_device": got != ref_dev,
                "differs_from_all_host": got != ref_host,
                "first_mismatch_byte": (None if match else
                                        next((i for i, (a, b) in enumerate(zip(got, want))
                                              if a != b), None)),
                "table_readback_before_upload": entry,
                "table_readback_after_upload": entry_after,
            })
        arm["status"] = "PASS" if ok else "FAIL"
        results[kernel] = arm
    overall = all(v.get("status") == "PASS" for v in results.values())
    return {"status": "PASS" if overall else "FAIL", "arms": results,
            "method": ("device and host stacks hold different deterministic content; the mixed "
                       "output must bit-match a per-expert-block assembly of the all-device and "
                       "all-host references")}


# ===========================================================================
# The timing sweep -- the curve itself.
# ===========================================================================

def sweep_kernel(lib, kernel: str, cfg: dict, lay: dict, buf: Buffers, dev: Stack, host: Stack,
                 rng: random.Random, miss_counts: list[int], pcie_ceiling_gbps: float | None,
                 checkpoint) -> dict:
    """Per-layer wall time vs host-resident (missed) expert count, for one kernel arm.

    METHOD, and why each part is load-bearing:

      * ONE LAUNCH per measurement.  The whole question is whether the expert-blocks of a layer
        overlap, so splitting them across launches would destroy the very structure being measured.
        The two-stack table is what makes a single launch possible at all.

      * THE ROUTE IS REDRAWN EVERY REP.  Ten experts of gemm1 weights is ~16 MiB, comfortably
        inside this card's 64 MB Infinity Cache, so hammering one route would let rep 2 onward read
        "host" weights out of on-die cache and report a miss as free -- a fabricated cliff.  A
        fresh draw walks a different 16 MiB of a 1.2 GiB stack each time.

      * PLUS AN EXPLICIT CACHE THRASH between reps, because a redrawn route is a statistical
        argument and a 192 MiB streaming read is a structural one.

      * COLD IS THE HEADLINE.  Each rep is timed as a single launch after a thrash; that is the
        serving regime.  A `hot` figure (a back-to-back burst on one fixed route) is recorded
        separately because it bounds the other end -- what the layer would cost if the tier were
        perfectly warm -- and the gap between them is itself informative.

      * EVERY POINT CARRIES ITS IMPLIED BANDWIDTH.  At full miss, all of the arm's weight bytes
        must cross PCIe, so `bytes / t` there cannot exceed the link.  If it does, the "host" stack
        is not on the far side of the link and the whole run is void -- the single sharpest
        falsifier of the failure that killed P2, and it costs one division.
    """
    E = cfg["num_experts"]
    tl = tiling_for(lib, kernel, cfg, lay)
    per_expert_bytes = lay["gemm1_bytes_per_expert" if kernel == "gemm1_silu"
                           else "gemm2_bytes_per_expert"]
    scatter = bool(cfg["scatter"]) and kernel == "gemm2"
    out_name = "out1" if kernel == "gemm1_silu" else "out2"
    reps = cfg["reps"]
    points: list[dict] = []
    n_blocks_seen: set[int] = set()
    verify_ok = True

    for m in miss_counts:
        cold: list[float] = []
        achieved: list[int] = []
        wall0 = time.perf_counter()
        for rep in range(reps + 1):            # rep 0 = discarded warm-up
            experts = draw_route_experts(rng, cfg["tokens"], cfg["top_k"], E, cfg["route_mode"])
            route = build_route(cfg["tokens"], cfg["top_k"], E, cfg["block_m"], experts)
            n_blocks_seen.add(route["n_blocks"])
            if m > route["n_blocks"]:
                continue
            host_experts, ach = choose_host_blocks(rng, route, m)
            buf.upload_route(route)
            upload_tables(lib, table_for(kernel, cfg, lay, dev, host, host_experts), E)
            buf.thrash()
            ms = launch(lib, kernel, True, cfg, lay, buf, route, tl, scatter, 1)
            if rep == 0:
                continue
            cold.append(ms[0])
            achieved.append(ach)
        wall = time.perf_counter() - wall0

        # HOT: a burst on one fixed route, no thrash between launches.
        experts = draw_route_experts(rng, cfg["tokens"], cfg["top_k"], E, cfg["route_mode"])
        route = build_route(cfg["tokens"], cfg["top_k"], E, cfg["block_m"], experts)
        host_experts, ach_hot = choose_host_blocks(rng, route, min(m, route["n_blocks"]))
        buf.upload_route(route)
        upload_tables(lib, table_for(kernel, cfg, lay, dev, host, host_experts), E)
        buf.thrash()
        hot = launch(lib, kernel, True, cfg, lay, buf, route, tl, scatter, max(2, cfg["hot_reps"]))

        # ---- per-point output verification -----------------------------------------------
        # The two stacks are MIRRORED here (identical content), so a mixed placement must produce
        # the byte-identical result an all-device placement does.  Anything else means this point's
        # host reads returned garbage -- which is not hypothetical on this box: P6 showed the driver
        # serving the WRONG PHYSICAL PAGE while every call returned hipSuccess.  Run untimed, with
        # SCATTER off (the atomic epilogue is not bit-reproducible by construction), so it costs the
        # sweep nothing and can never be mistaken for a timing sample.
        buf.zero(out_name)
        launch(lib, kernel, True, cfg, lay, buf, route, tl, False, 1)
        mixed_out = buf.readback(out_name)
        upload_tables(lib, table_for(kernel, cfg, lay, dev, host, []), E)
        buf.zero(out_name)
        launch(lib, kernel, True, cfg, lay, buf, route, tl, False, 1)
        ref_out = buf.readback(out_name)
        verified = (mixed_out == ref_out)
        verify_ok = verify_ok and verified
        buf.zero(out_name)

        st = stats(cold)
        st_hot = stats(hot[1:] or hot)
        mean_ach = statistics.fmean(achieved) if achieved else float(m)
        host_bytes = mean_ach * per_expert_bytes
        total_bytes = statistics.fmean(n_blocks_seen) * per_expert_bytes if n_blocks_seen else 0
        med = st.get("median")
        points.append({
            "requested_misses": m,
            # At m == 0 the mixed table IS the all-device table, so this degenerates into a
            # determinism check rather than a placement check -- still worth having, and labelled
            # so nobody reads a trivially-true row as evidence about the host path.
            "output_bit_matches_all_device": verified,
            "verification_is_placement_sensitive": m > 0,
            "hot_achieved_misses": ach_hot,
            "achieved_misses_mean": mean_ach,
            "cold_ms": st,
            "hot_ms": st_hot,
            "host_bytes_per_launch": host_bytes,
            "total_weight_bytes_per_launch": total_bytes,
            "implied_host_gbps": (host_bytes / (med * 1e-3) / 1e9) if (med and host_bytes) else None,
            "implied_total_gbps": (total_bytes / (med * 1e-3) / 1e9) if med else None,
            "wall_seconds": wall,
            "event_total_ms": sum(cold),
            "event_vs_wall_ok": (sum(cold) <= wall * 1e3 * TIMING_SANITY_FACTOR),
        })
        checkpoint()

    curve = {p["requested_misses"]: p["cold_ms"]["median"] for p in points
             if p["cold_ms"].get("median") is not None}
    fit = fit_curve(curve)
    # The binomial trial count is the number of independent expert draws this layer makes -- i.e.
    # the block count -- which equals top_k only at tokens == 1.  Using top_k at tokens > 1 would
    # silently compute the expectation for a smaller layer than the one measured.
    n_trials = max(curve) if curve else cfg["top_k"]
    hit_rows = hit_rate_analysis(curve, n_trials, cfg["hit_rates"])

    # ---- the PCIe falsifier, applied at full miss ----------------------------
    full = max(curve) if curve else None
    pcie = None
    if full is not None and pcie_ceiling_gbps:
        pt = next(p for p in points if p["requested_misses"] == full)
        achieved_gbps = pt["implied_host_gbps"]
        limit = pcie_ceiling_gbps * 1.15
        pcie = {
            "at_miss_count": full,
            "host_bytes": pt["host_bytes_per_launch"],
            "achieved_gbps": achieved_gbps,
            "pcie_ceiling_gbps": pcie_ceiling_gbps,
            "limit_with_margin_gbps": limit,
            "ok": bool(achieved_gbps is not None and achieved_gbps <= limit),
            "note": ("at full miss every weight byte this arm reads must cross PCIe, so the "
                     "achieved rate cannot exceed the link. If it does, the 'host' stack is not "
                     "host -- which is exactly what P2 discovered too late."),
        }

    return {
        "kernel": kernel,
        "tiling": tl,
        "scatter": scatter,
        "bit_reproducible": not scatter,
        "bytes_per_expert": per_expert_bytes,
        "n_blocks_observed": sorted(n_blocks_seen),
        "workgroups_per_launch": [tl["grid_x"] * nb for nb in sorted(n_blocks_seen)],
        "points": points,
        "all_points_bit_verified": verify_ok,
        "curve_median_ms": {str(k): v for k, v in curve.items()},
        "fit": fit,
        "hit_rate_analysis": hit_rows,
        "decision": decide(fit, hit_rows),
        "pcie_falsifier": pcie,
    }


def combine_layer(arms: dict) -> dict:
    """The MoE layer is gemm1 + gemm2, so the decision must be made on their SUM.

    Reporting only the arm with the friendlier curve would be cherry-picking: a linear gemm1 next
    to a cliffed gemm2 is a layer that is mostly cliffed, and it is the layer that sets tok/s.
    """
    common = None
    for a in arms.values():
        ks = set(a["curve_median_ms"])
        common = ks if common is None else (common & ks)
    if not common:
        return {"status": "NO_COMMON_MISS_POINTS"}
    curve = {int(k): sum(a["curve_median_ms"][k] for a in arms.values()) for k in common}
    fit = fit_curve(curve)
    n = max(curve)
    rows = hit_rate_analysis(curve, n, [0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    return {
        "curve_median_ms": {str(k): v for k, v in curve.items()},
        "fit": fit,
        "hit_rate_analysis": rows,
        "decision": decide(fit, rows),
        "note": ("gemm1 + gemm2 summed at equal miss count; this is the per-layer number the tok/s "
                 "projection uses"),
    }


# ===========================================================================
# One card, end to end.
# ===========================================================================

def device_identity(lib, idx: int) -> dict:
    name = ctypes.create_string_buffer(256)
    arch = ctypes.create_string_buffer(128)
    pci = ctypes.create_string_buffer(64)
    mp, wp, clk = ctypes.c_int(0), ctypes.c_int(0), ctypes.c_int(0)
    tot = ctypes.c_ulonglong(0)
    K(lib, lib.p2p_device_info(idx, name, 256, arch, 128, pci, 64, ctypes.byref(mp),
                               ctypes.byref(wp), ctypes.byref(clk), ctypes.byref(tot)),
      "p2p_device_info")
    bdf = pci.value.decode().strip().lower()
    return {
        "rocm_index": idx,
        "name": name.value.decode(),
        "arch": arch.value.decode(),
        "pci_bdf": bdf,
        # torch/HIP report WGPs on RDNA, not CUs -- a 64-CU card answers 32.  Recorded raw with the
        # doubling noted so nobody silently halves an occupancy calculation.
        "multi_processor_count_raw": mp.value,
        "cu_count_if_wgp_reported": mp.value * 2,
        "warp_size": wp.value,
        "clock_khz": clk.value,
        "total_mem_bytes": int(tot.value),
        "pcie_link_chain": pcie_link_chain(bdf),
        "physical_card_note": ("card 0 = RX 9070 XT 0000:03:00.0 root port 0000:00:01.1 Gen5 x8; "
                               "card 1 = RX 9070 0000:07:00.0 root port 0000:00:01.3 Gen4 x8 -- "
                               "host bandwidth is HALF on card 1 and that is a platform state, "
                               "not a design property"),
    }


def run_device(lib, idx: int, cfg: dict, checkpoint) -> dict:
    K(lib, lib.p2p_set_device(idx), f"hipSetDevice({idx})")
    rec: dict = {"utc_start": _utc(), "status": "RUNNING"}
    ident = device_identity(lib, idx)
    rec.update(ident)
    bdf = ident["pci_bdf"]
    rec["box_state_before"] = collect_box_state(f"device{idx}_before", with_smi=False)
    checkpoint()

    lay = expert_layout(cfg["hidden"], cfg["inter"], cfg["group_size"], cfg["zeros"])
    stack_bytes = cfg["num_experts"] * lay["granule_stride"]
    rec["layout"] = dict(lay, stack_bytes=stack_bytes,
                         stack_gib=stack_bytes / (1 << 30))
    P_max = cfg["tokens"] * cfg["top_k"] * cfg["block_m"]
    rec["P_max"] = P_max

    free_vram = _vram_free(lib)
    need = stack_bytes + cfg["thrash_mb"] * (1 << 20) + (P_max * cfg["hidden"] * 4) + (1 << 28)
    rec["vram_budget"] = {"free_bytes": free_vram, "estimated_need_bytes": need,
                          "headroom_bytes": free_vram - need}
    if free_vram < need:
        _fail(f"card {idx} ({bdf}) has {free_vram / 2**30:.2f} GiB free VRAM but the run needs "
              f"~{need / 2**30:.2f} GiB (device expert stack {stack_bytes / 2**30:.2f} GiB + "
              f"{cfg['thrash_mb']} MiB thrash + buffers). Lower --num-experts or --thrash-mb, or "
              f"wait for the card to drain -- do NOT silently shrink the stack, because a stack "
              f"that fits in cache stops measuring memory at all.")

    dev = Stack(lib, "device", stack_bytes, bdf)
    host = Stack(lib, "host", stack_bytes, bdf)
    buf = None
    try:
        rec["placement"] = {"device_stack": dev.alloc(), "host_stack": host.alloc()}
        checkpoint()

        # ---- copy-engine denominator, link sampled MID-DMA (never at idle) ----
        copy_bytes = min(cfg["copy_mb"] * (1 << 20), stack_bytes)
        dst = ctypes.c_void_p()
        K(lib, lib.p2p_malloc_device(ctypes.byref(dst), copy_bytes), "hipMalloc(copy dst)")
        try:
            rec["copy_engine"] = measure_copy_engine(lib, int(dst.value), host.host_ptr,
                                                     copy_bytes, cfg["bw_reps"], bdf)
        finally:
            lib.p2p_free_device(dst)

        # ---- checksummed kernel-read bandwidth over each stack ----
        bw_bytes = min(cfg["bw_mb"] * (1 << 20), stack_bytes)
        bw_bytes -= bw_bytes % 16
        rec["read_bw"] = {
            "region_bytes": bw_bytes,
            "device_stack": measure_read_bw(lib, dev.dev_ptr, bw_bytes, cfg["bw_blocks"],
                                            cfg["bw_threads"], cfg["bw_reps"]),
            "host_stack": measure_read_bw(lib, host.dev_ptr, bw_bytes, cfg["bw_blocks"],
                                          cfg["bw_threads"], cfg["bw_reps"]),
        }
        rec["preconditions"] = gate_preconditions(dev, host, rec["read_bw"]["device_stack"],
                                                  rec["read_bw"]["host_stack"], rec["copy_engine"])
        checkpoint()
        if not rec["preconditions"]["passed"]:
            rec["status"] = "ABORTED_PRECONDITION"
            rec["abort_reason"] = (
                "precondition(s) failed: " + ", ".join(rec["preconditions"]["failed"]) +
                ". No curve is reported: an unevaluated criterion is not a passing one. This is "
                "the same gate P2 aborted on, and it is here so a fake-host stack can never reach "
                "the timing loop.")
            rec["utc_end"] = _utc()
            return rec

        pcie_ceiling = rec["preconditions"]["copy_engine_gbps"]

        buf = Buffers(lib, cfg, lay, P_max)
        buf.alloc()
        checkpoint()

        # ---- provenance: different content per stack, bit-exact assembled reference ----
        rng = random.Random(cfg["seed"])
        rec["stack_fill_provenance"] = {
            "device": fill_stack_weights(lib, dev, cfg, lay, cfg["seed_device"]),
            "host": fill_stack_weights(lib, host, cfg, lay, cfg["seed_host"]),
        }
        rec["correctness"] = correctness_arm(lib, cfg, lay, buf, dev, host, rng)
        checkpoint()
        if rec["correctness"]["status"] != "PASS":
            rec["status"] = "ABORTED_CORRECTNESS"
            rec["abort_reason"] = (
                "the mixed-media output did not bit-match the per-block assembly of the all-device "
                "and all-host references. A timing curve measured through a table the kernel is "
                "not honouring measures nothing.")
            rec["utc_end"] = _utc()
            return rec

        # ---- MIRROR the stacks for timing: identical content, so the only variable is media ----
        rec["stack_fill_timing"] = {
            "host_refilled_with_device_seed": fill_stack_weights(lib, host, cfg, lay,
                                                                 cfg["seed_device"]),
            "why": ("with identical content every miss count does numerically identical work, so a "
                    "time difference can only be the medium; it also means each timed output is "
                    "still comparable against the all-device reference"),
        }
        rec["mirror_check"] = verify_mirror(lib, cfg, lay, dev, host)
        if not rec["mirror_check"]["ok"]:
            rec["status"] = "ABORTED_MIRROR"
            rec["abort_reason"] = ("the two stacks were not byte-identical after the mirror refill, "
                                   "so a miss would change the arithmetic and not just the medium")
            rec["utc_end"] = _utc()
            return rec

        rec["indirection_control"] = indirection_control(lib, cfg, lay, buf, cfg["control_reps"])
        checkpoint()

        n_blocks_max = cfg["tokens"] * cfg["top_k"]
        miss_counts = [m for m in cfg["miss_counts"] if 0 <= m <= n_blocks_max]
        rec["miss_counts"] = miss_counts
        arms: dict = {}
        for kernel in KERNEL_ARMS:
            arms[kernel] = sweep_kernel(lib, kernel, cfg, lay, buf, dev, host, rng, miss_counts,
                                        pcie_ceiling, checkpoint)
            rec["arms"] = arms
            checkpoint()
        rec["layer_total"] = combine_layer(arms)

        bad_verify = [k for k, a in arms.items() if not a.get("all_points_bit_verified")]
        if bad_verify:
            rec["status"] = "ABORTED_TIMING_VERIFY"
            rec["abort_reason"] = (
                f"arm(s) {bad_verify} produced a mixed-placement output that differs from the "
                f"all-device result while the two stacks hold IDENTICAL bytes. The host reads "
                f"returned wrong data during the sweep, so the timings describe a broken read "
                f"path, not a medium.")
            rec["utc_end"] = _utc()
            rec["layer_total"] = rec.get("layer_total")
            return rec
        bad_pcie = [k for k, a in arms.items()
                    if a.get("pcie_falsifier") and not a["pcie_falsifier"]["ok"]]
        if bad_pcie:
            rec["status"] = "ABORTED_PCIE_FALSIFIER"
            rec["abort_reason"] = (
                f"arm(s) {bad_pcie} read their full-miss weight bytes FASTER than the measured "
                f"PCIe link allows. Bytes that must cross the link cannot outrun it, so those "
                f"reads did not come from host pages. Every number in this record is void.")
        else:
            rec["status"] = "OK"
            rec["verdict"] = rec["layer_total"].get("decision")
    finally:
        if buf is not None:
            buf.free()
        host.free()
        dev.free()
        lib.p2p_sync()
    rec["box_state_after"] = collect_box_state(f"device{idx}_after", with_smi=False)
    rec["utc_end"] = _utc()
    return rec


def verify_mirror(lib, cfg: dict, lay: dict, dev: Stack, host: Stack) -> dict:
    """Confirm the two stacks really are byte-identical before the timing sweep.

    Sampled rather than exhaustive (1.2 GiB x2 over PCIe would dominate the run), but sampled at
    the head AND tail of several experts' granules, which is where a short/over-run fill shows up.
    """
    stride = lay["granule_stride"]
    n = 4096
    experts = sorted({0, cfg["num_experts"] // 2, cfg["num_experts"] - 1})
    samples = []
    ok = True
    for e in experts:
        for label, off in (("head", 0), ("tail", max(0, stride - n))):
            a = ctypes.create_string_buffer(n)
            b = ctypes.create_string_buffer(n)
            K(lib, lib.p2p_memcpy(ctypes.cast(a, ctypes.c_void_p),
                                  ctypes.c_void_p(dev.base_of(e, stride) + off), n, 2),
              "mirror sample device")
            K(lib, lib.p2p_memcpy(ctypes.cast(b, ctypes.c_void_p),
                                  ctypes.c_void_p(host.base_of(e, stride) + off), n, 2),
              "mirror sample host")
            same = a.raw == b.raw
            ok = ok and same
            samples.append({"expert": e, "where": label, "identical": same})
    return {"ok": ok, "bytes_per_sample": n, "samples": samples,
            "method": "head+tail of several granules compared device-vs-host after the refill"}


# ===========================================================================
# CLI, artifact shape, rendering
# ===========================================================================

def parse_int_list(s: str, what: str) -> list[int]:
    out = []
    for tok in (s or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(int(tok))
        except ValueError:
            _fail(f"{what}: {tok!r} is not an integer")
    return out


def parse_float_list(s: str, what: str) -> list[float]:
    out = []
    for tok in (s or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(float(tok))
        except ValueError:
            _fail(f"{what}: {tok!r} is not a number")
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="p2prime_two_stack_moe.py",
        description="P2prime -- mixed-media grouped MoE decode-GEMV on explicit two-stack pointers",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("shape (defaults track the target checkpoint P2 used)")
    g.add_argument("--num-experts", type=int, default=512)
    g.add_argument("--top-k", type=int, default=10)
    g.add_argument("--tokens", type=int, default=1,
                   help="decode batch; 1 gives exactly top_k expert-blocks so a miss COUNT is "
                        "unambiguous")
    g.add_argument("--hidden", type=int, default=2048)
    g.add_argument("--inter", type=int, default=768)
    g.add_argument("--group-size", type=int, default=128)
    g.add_argument("--block-m", type=int, default=16,
                   help="moe_align tile height; 16 is what _moe_block_m picks at decode")
    g.add_argument("--e2m1", action="store_true",
                   help="MXFP4 e2m1 weight codes (symmetric); default is uniform int4 + zero-points")
    g.add_argument("--no-zeros", dest="zeros", action="store_false",
                   help="drop the zero-point tensors from the granule (default: keep them, so the "
                        "scales-and-zeros-travel-with-the-weights contract is actually exercised)")
    g.add_argument("--act-dtype", default="fp16", choices=["fp16"],
                   help="activation dtype; only fp16 is instantiated -- rejected explicitly rather "
                        "than silently substituted")
    g.add_argument("--route-mode", default="distinct", choices=["distinct", "random"])
    g.add_argument("--no-scatter", dest="scatter", action="store_false",
                   help="run gemm2 without the atomic scatter epilogue (default: scatter, which is "
                        "what production decode uses)")

    m = p.add_argument_group("sweep")
    m.add_argument("--miss-counts", default="",
                   help="comma-separated miss counts; default is the FULL range 0..tokens*top_k, "
                        "which is what makes the binomial hit-rate expectation exact")
    m.add_argument("--reps", type=int, default=20,
                   help="independent cold reps per point (route redrawn + cache thrashed each)")
    m.add_argument("--hot-reps", type=int, default=6)
    m.add_argument("--control-reps", type=int, default=8)
    m.add_argument("--hit-rates", default="0.25,0.5,0.75,0.9,0.95,0.99")

    b = p.add_argument_group("bandwidth / preconditions")
    b.add_argument("--bw-mb", type=int, default=256)
    b.add_argument("--bw-reps", type=int, default=5)
    b.add_argument("--bw-blocks", type=int, default=4096)
    b.add_argument("--bw-threads", type=int, default=256)
    b.add_argument("--copy-mb", type=int, default=256)
    b.add_argument("--thrash-mb", type=int, default=192,
                   help="streaming read between reps; must exceed the 64 MB Infinity Cache or the "
                        "probe measures cache, not media")

    r = p.add_argument_group("run")
    r.add_argument("--devices", default="0,1",
                   help="ROCm device indices, run SERIALLY; both cards by default because card 1's "
                        "root port is Gen4 and its curve may differ")
    r.add_argument("--seed", type=int, default=20260903)
    r.add_argument("--seed-device", type=lambda x: int(x, 0), default=0x0D0E1CE0)
    r.add_argument("--seed-host", type=lambda x: int(x, 0), default=0x40571234)
    r.add_argument("--kernels-src", default=DEFAULT_KERNELS_SRC,
                   help="rdna4-hip-kernels fp8_wmma_rocm dir supplying gemv_decode.h")
    r.add_argument("--allow-dirty-kernels", action="store_true")
    r.add_argument("--hipcc", default="/opt/rocm/bin/hipcc")
    r.add_argument("--arch", default="gfx1201")
    r.add_argument("--rebuild", action="store_true")
    r.add_argument("--outdir", default=DEFAULT_OUTDIR)
    r.add_argument("--tag", default="")
    r.add_argument("--selftest", action="store_true",
                   help="validate argument handling, layout/route/mask math, the curve fit and the "
                        "JSON shape WITHOUT touching the GPU")
    return p


def finalize_cfg(args) -> dict:
    if args.tokens < 1 or args.top_k < 1:
        _fail("--tokens and --top-k must be >= 1")
    if args.block_m % 8 or args.block_m > 64:
        _fail("--block-m must be a multiple of 8 and <= 64 (the core's contract)")
    if min(args.block_m, args.tokens) > 16:
        _fail("min(block_m, tokens) > 16: this decode probe instantiates the MMAX {8, 16} ladder "
              "only. Raising it is a one-line change in p2prime_kernels.hip, but silently clamping "
              "would measure a different kernel than the one named.")
    if args.route_mode == "distinct" and args.tokens * args.top_k > args.num_experts:
        _fail("--route-mode distinct needs tokens*top_k <= num_experts")
    n_blocks_max = args.tokens * args.top_k
    miss = parse_int_list(args.miss_counts, "--miss-counts") or list(range(n_blocks_max + 1))
    hit = parse_float_list(args.hit_rates, "--hit-rates")
    if any(not (0.0 <= h <= 1.0) for h in hit):
        _fail("--hit-rates must lie in [0, 1]")
    if args.thrash_mb < 64:
        _fail("--thrash-mb below 64 cannot evict this card's Infinity Cache; a smaller value would "
              "let a host-resident expert be served from on-die cache and report a miss as free")
    return {
        "num_experts": args.num_experts, "top_k": args.top_k, "tokens": args.tokens,
        "hidden": args.hidden, "inter": args.inter, "group_size": args.group_size,
        "block_m": args.block_m, "e2m1": bool(args.e2m1), "zeros": bool(args.zeros),
        "act_dtype": args.act_dtype, "route_mode": args.route_mode, "scatter": bool(args.scatter),
        "miss_counts": miss, "reps": args.reps, "hot_reps": args.hot_reps,
        "control_reps": args.control_reps, "hit_rates": hit,
        "bw_mb": args.bw_mb, "bw_reps": args.bw_reps, "bw_blocks": args.bw_blocks,
        "bw_threads": args.bw_threads, "copy_mb": args.copy_mb, "thrash_mb": args.thrash_mb,
        "devices": parse_int_list(args.devices, "--devices"),
        "seed": args.seed, "seed_device": args.seed_device, "seed_host": args.seed_host,
        "kernels_src": args.kernels_src, "hipcc": args.hipcc, "arch": args.arch,
        "outdir": args.outdir, "tag": args.tag,
    }


def collect_static_env() -> dict:
    git = {}
    for name, args in (("sha", ["rev-parse", "HEAD"]),
                       ("branch", ["rev-parse", "--abbrev-ref", "HEAD"]),
                       ("porcelain", ["status", "--porcelain"])):
        r = _run_cmd(["git", "-C", REPO] + args)
        git[name] = r.get("stdout") if r.get("rc") == 0 else None
    git["dirty"] = bool(git.get("porcelain"))
    git["porcelain"] = (git.get("porcelain") or "")[:2000]
    return {
        "hostname": platform.node(), "kernel": platform.release(),
        "python": sys.version.split()[0], "worktree": REPO, "git": git,
        "amdgpu_module_version": (_read_text("/sys/module/amdgpu/version") or "").strip() or None,
        "rocm_version_file": (_read_text("/opt/rocm/.info/version") or "").strip() or None,
        "env_fence": _ENV_FENCE,
        "env_seen": {k: v for k, v in os.environ.items()
                     if k.startswith(("ROCR_", "HIP_", "HSA_", "GPU_", "AMD_", "VLLM_",
                                      "MINISGL_", "PYTORCH_"))},
        "gpu_lease": ("WAIVED by explicit user instruction for the weight-offload workstream; "
                      "cards are used SERIALLY and box state is recorded around each"),
    }


REQUIRED_TOP_KEYS = ("schema_version", "probe_id", "utc_start", "status", "config", "argv",
                     "static_env", "build", "kernels_src", "box_state_before", "devices",
                     "overall")


def validate_result(r: dict) -> None:
    missing = [k for k in REQUIRED_TOP_KEYS if k not in r]
    if missing:
        _fail(f"result is missing required keys: {missing}")
    if not isinstance(r["devices"], list):
        _fail("result['devices'] must be a list")
    for d in r["devices"]:
        for k in ("rocm_index", "pci_bdf", "status"):
            if k not in d:
                _fail(f"device record missing {k!r}")
    json.dumps(r)     # must be serialisable, not merely well-shaped


def overall_verdict(devices: list[dict]) -> dict:
    per_card = {}
    for d in devices:
        per_card[str(d.get("rocm_index"))] = {
            "pci_bdf": d.get("pci_bdf"),
            "status": d.get("status"),
            "cliff_index": ((d.get("layer_total") or {}).get("fit") or {}).get("cliff_index"),
            "shape": ((d.get("layer_total") or {}).get("fit") or {}).get("shape"),
            "verdict": (d.get("verdict") or {}).get("verdict"),
            "best_gain_over_layer_granular":
                (d.get("verdict") or {}).get("best_per_expert_gain_over_layer_granular"),
            "abort_reason": d.get("abort_reason"),
        }
    ok = [v for v in per_card.values() if v["status"] == "OK"]
    shapes = {v["shape"] for v in ok if v["shape"]}
    if not ok:
        status = "NOT_MEASURED"
        summary = ("no card produced a curve. The failing preconditions are recorded per card; an "
                   "unevaluated criterion is NOT a passing one, so nothing here should be read as "
                   "evidence about the shape.")
    elif len(shapes) == 1:
        shape = shapes.pop()
        status = shape
        summary = (f"both cards agree: {shape}. " if len(ok) > 1 else f"{shape}. ")
        summary += ("Per-expert placement is worth building only if it beats LAYER-GRANULAR "
                    "placement at the same byte budget; the measured gain is in "
                    "`best_gain_over_layer_granular` per card.")
    else:
        status = "CARDS_DISAGREE"
        summary = (f"cards disagree on shape ({sorted(shapes)}). Card 1's root port is trained "
                   f"Gen4 x8 against card 0's Gen5, so its host reads are half speed and its curve "
                   f"can genuinely differ -- read both, and do not generalise from card 0 alone.")
    return {"status": status, "per_card": per_card, "summary": summary}


def _f(x, nd=3, dash="—"):
    if x is None:
        return dash
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return "inf" if math.isinf(x) else "nan"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def render_md(r: dict) -> str:
    L: list[str] = []
    A = L.append
    A("# P2prime — mixed-media grouped MoE GEMM on explicit two-stack pointers")
    A("")
    A("*Probe spec:* the Phase 0 gate report §6 unknown #1 (`docs/measurements/"
      "WEIGHT_OFFLOAD_2026-09-02/PHASE0_REPORT.md`) — **is per-layer time LINEAR or CLIFFED in "
      "miss count?**")
    A(f"*Run:* {r['utc_start']} → {r.get('utc_end')} · *status:* **{r['status']}**")
    A(f"*Worktree:* `{r['static_env']['worktree']}` @ `{r['static_env']['git'].get('sha')}` "
      f"(`{r['static_env']['git'].get('branch')}`)")
    A(f"*Kernels:* `{r['kernels_src']['path']}` @ `{r['kernels_src'].get('sha')}` "
      f"(dirty={r['kernels_src'].get('dirty')})")
    A("")
    A("## Verdict")
    A("")
    ov = r["overall"]
    A(f"**{ov['status']}** — {ov['summary']}")
    A("")
    A("| card | BDF | status | cliff_index | shape | gain over layer-granular |")
    A("|---|---|---|---|---|---|")
    for k, v in ov["per_card"].items():
        A(f"| {k} | `{v['pci_bdf']}` | {v['status']} | {_f(v['cliff_index'])} | "
          f"{v['shape'] or '—'} | {_f(v['best_gain_over_layer_granular'])}x |")
    A("")
    A("`cliff_index = (t(1) − t(0)) / (t(n) − t(0))`: **1/n ⇒ LINEAR** (misses cost additively, "
      "per-expert placement is worth building), **≈1.0 ⇒ CLIFFED** (one host expert costs what all "
      "of them do, per-expert placement buys ≈ zero and M2 is layer-granular).")
    A("")
    A("## Mechanism")
    A("")
    A("Two stacks, because a mixed-media single VA is **not constructible on this box** (P1/P2/P3: "
      "`hipMemCreate(location=Host)` silently returns VRAM):")
    A("")
    A("* device stack — `hipMalloc`;")
    A("* host stack — `hipHostMalloc(Mapped|Portable)` + `hipHostGetDevicePointer`;")
    A("* a per-expert `__constant__` pointer table (weights **and** scales **and** zero-points, "
      "always together — a granule is one expert's slice of every tensor).")
    A("")
    A("The kernel is the **shipped** `gemv_decode::gemv_decode_core`; the two-stack layout is a "
      "WLoad policy (`TwoStackInt4A16GemvLoader`) that inherits the shipped "
      "`Int4A16GemvLoader<__half>` and replaces only the three per-expert base-pointer hooks. Per "
      "`KERNEL_CORE_POLICY.md` a placement scheme is a loader policy, never a kernel fork.")
    A("")
    for d in r["devices"]:
        A(f"## ROCm device {d['rocm_index']} — {d.get('name')} `{d.get('pci_bdf')}`")
        A("")
        A(f"*status:* **{d['status']}**" + (f" — {d['abort_reason']}" if d.get("abort_reason") else ""))
        A("")
        pc = d.get("preconditions") or {}
        if pc:
            A("### Preconditions (placement is MEASURED, never declared)")
            A("")
            A("| check | ok | detail |")
            A("|---|---|---|")
            for c in pc["checks"]:
                A(f"| `{c['check']}` | {'✅' if c['ok'] else '❌'} | {c['detail']} |")
            A("")
            A(f"device read **{_f(pc.get('device_read_gbps'), 1)} GB/s** · host read "
              f"**{_f(pc.get('host_read_gbps'), 1)} GB/s** · copy engine "
              f"**{_f(pc.get('copy_engine_gbps'), 1)} GB/s** · separation "
              f"**{_f(pc.get('media_separation_ratio'), 2)}×** "
              f"(P2 measured 1.00× here, which is how it discovered the pages were VRAM)")
            A("")
        if d.get("correctness"):
            c = d["correctness"]
            A(f"### Correctness — **{c['status']}**")
            A("")
            for name, arm in c["arms"].items():
                A(f"* `{name}`: references differ = {arm.get('references_differ')}; "
                  + "; ".join(f"{cs['achieved_misses']} misses → bit-exact="
                              f"{cs['bit_exact_vs_assembled_reference']}"
                              for cs in arm.get("cases", [])))
            A("")
        ic = d.get("indirection_control") or {}
        if ic.get("arms"):
            A("### Pointer-table overhead (A/B against the STOCK loader — old CODE, not emulated)")
            A("")
            A("| arm | stock ms | table ms | overhead | bit-identical |")
            A("|---|---|---|---|---|")
            for name, a in ic["arms"].items():
                A(f"| `{name}` | {_f(a['stock_loader_ms'].get('median'), 4)} | "
                  f"{_f(a['table_loader_ms'].get('median'), 4)} | "
                  f"{_f((a['indirection_overhead_frac'] or 0) * 100, 2)}% | "
                  f"{a['bit_identical_output']} |")
            A("")
        for name, arm in (d.get("arms") or {}).items():
            A(f"### Curve — `{name}` ({arm['bytes_per_expert'] / 2**20:.2f} MiB/expert, "
              f"{'scatter' if arm['scatter'] else 'non-scatter'}, "
              f"bit-verified={arm.get('all_points_bit_verified')})")
            A("")
            A("| misses | cold ms (median) | p10–p90 | hot ms | implied host GB/s |")
            A("|---|---|---|---|---|")
            for pt in arm["points"]:
                cm = pt["cold_ms"]
                A(f"| {pt['requested_misses']} | {_f(cm.get('median'), 4)} | "
                  f"{_f(cm.get('p10'), 4)}–{_f(cm.get('p90'), 4)} | "
                  f"{_f(pt['hot_ms'].get('median'), 4)} | {_f(pt.get('implied_host_gbps'), 1)} |")
            A("")
            f = arm["fit"]
            A(f"**cliff_index {_f(f.get('cliff_index'))}** (pure-linear would be "
              f"{_f(f.get('cliff_index_pure_linear'))}) → **{f.get('shape')}**. "
              f"separation {_f(f.get('separation_ratio'), 2)}×. "
              f"effective miss concurrency W = "
              f"{_f((f.get('parallel_service_fit') or {}).get('effective_miss_concurrency'), 2)}.")
            if f.get("shape_reason"):
                A("")
                A(f"> {f['shape_reason']}")
            A("")
        lt = d.get("layer_total") or {}
        if lt.get("hit_rate_analysis"):
            A("### Layer total (gemm1 + gemm2) — the decision table")
            A("")
            A("| hit rate h | per-expert tier ms | layer-granular ms | per-expert gain | P(all resident) |")
            A("|---|---|---|---|---|")
            for row in lt["hit_rate_analysis"]:
                A(f"| {row['hit_rate']:.2f} | {_f(row['expected_ms_per_expert_tier'], 4)} | "
                  f"{_f(row['expected_ms_layer_granular'], 4)} | "
                  f"{_f(row['per_expert_gain_over_layer_granular'])}x | "
                  f"{row['prob_all_resident']:.2e} |")
            A("")
            A(f"**{lt['decision']['verdict']}** — {lt['decision']['summary']}")
            A("")
    A("## Provenance")
    A("")
    A(f"Stack: {r['static_env'].get('rocm_version_file')} · amdgpu "
      f"{r['static_env'].get('amdgpu_module_version')} · kernel "
      f"{r['static_env'].get('kernel')} · hipcc "
      f"{(r['build'].get('hipcc_version') or [''])[0]}")
    A("")
    A("GPU lease **waived by explicit user instruction** for this workstream; cards were used "
      "serially and box state is recorded around each. `ROCR_VISIBLE_DEVICES=0,1` with "
      "`HIP_VISIBLE_DEVICES` unset throughout, so the Ryzen iGPU never entered enumeration.")
    A("")
    return "\n".join(L) + "\n"


# ===========================================================================
# --selftest: everything that can be checked without a GPU
# ===========================================================================

def _synthetic_curve(kind: str, n: int, t0: float = 1.0, span: float = 9.0) -> dict[int, float]:
    if kind == "linear":
        return {m: t0 + span * m / n for m in range(n + 1)}
    if kind == "cliff":
        return {m: (t0 if m == 0 else t0 + span) for m in range(n + 1)}
    if kind == "partial":
        return {m: t0 + span * (0.45 + 0.55 * (m - 1) / (n - 1)) if m else t0
                for m in range(n + 1)}
    if kind == "flat":
        return {m: t0 for m in range(n + 1)}
    raise ValueError(kind)


def selftest(args) -> dict:
    """Validate argument handling, the layout/route/mask math, the fit and the JSON shape.

    This runs the same functions the GPU path runs, on synthetic inputs, so a broken analysis is
    caught before a card is ever booked. It deliberately asserts on the DIRECTION of each verdict
    (a synthetic pure line must classify LINEAR, a synthetic pure cliff must classify CLIFFED),
    because a fit that always says "PARTIAL" would look like a working probe and answer nothing.
    """
    checks: list[dict] = []

    def chk(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    # ---- argument handling ----
    p = build_parser()
    cfg = finalize_cfg(p.parse_args([]))
    chk("defaults_parse", cfg["num_experts"] == 512 and cfg["top_k"] == 10, json.dumps(cfg["miss_counts"]))
    chk("miss_counts_default_is_full_range", cfg["miss_counts"] == list(range(11)),
        "the exact binomial hit-rate expectation requires every m from 0..top_k")
    cfg2 = finalize_cfg(p.parse_args(["--miss-counts", "0,1,2,5,10"]))
    chk("miss_counts_override", cfg2["miss_counts"] == [0, 1, 2, 5, 10])
    for bad, why in ((["--block-m", "17"], "block_m not a multiple of 8"),
                     # exercised with a shape that clears the distinct-route check first,
                     # so the MMAX guard is what actually fires rather than an earlier rule
                     (["--tokens", "32", "--top-k", "2", "--block-m", "32"],
                      "min(block_m, tokens) > 16"),
                     (["--thrash-mb", "8"], "thrash below the Infinity Cache"),
                     (["--hit-rates", "1.5"], "hit rate out of [0,1]")):
        try:
            finalize_cfg(p.parse_args(bad))
            chk(f"rejects[{why}]", False, "accepted a bad argument")
        except ProbeError as e:
            chk(f"rejects[{why}]", True, str(e)[:160])

    # ---- layout ----
    lay = expert_layout(cfg["hidden"], cfg["inter"], cfg["group_size"], cfg["zeros"])
    chk("layout_component_sizes",
        lay["sizes"]["w13"] == 2 * cfg["inter"] * (cfg["hidden"] // 8) * 4
        and lay["sizes"]["w2"] == cfg["hidden"] * (cfg["inter"] // 8) * 4,
        json.dumps(lay["sizes"]))
    chk("layout_offsets_aligned",
        all(o is None or o % GRANULE_COMPONENT_ALIGN == 0 for o in lay["offsets"].values()),
        json.dumps(lay["offsets"]))
    chk("layout_offsets_disjoint_and_ordered",
        all(lay["offsets"][a] is None or lay["offsets"][b] is None
            or lay["offsets"][b] >= lay["offsets"][a] + lay["sizes"][a]
            for a, b in zip(["w13", "s13", "z13", "w2", "s2"], ["s13", "z13", "w2", "s2", "z2"])))
    chk("granule_stride_covers_payload",
        lay["granule_stride"] >= lay["payload_bytes"]
        and lay["granule_stride"] % GRANULE_STRIDE_ALIGN == 0,
        f"stride={lay['granule_stride']} payload={lay['payload_bytes']}")
    try:
        expert_layout(2048, 100, 128, True)
        chk("layout_rejects_bad_inter", False)
    except ProbeError:
        chk("layout_rejects_bad_inter", True)

    # ---- route ----
    rng = random.Random(7)
    experts = draw_route_experts(rng, cfg["tokens"], cfg["top_k"], cfg["num_experts"], "distinct")
    chk("distinct_route_is_distinct", len(set(experts)) == len(experts))
    route = build_route(cfg["tokens"], cfg["top_k"], cfg["num_experts"], cfg["block_m"], experts)
    chk("route_block_count", route["n_blocks"] == cfg["tokens"] * cfg["top_k"],
        f"n_blocks={route['n_blocks']}")
    chk("route_sti_length", len(route["sorted_token_ids"]) == route["P"])
    chk("route_padding_marker_is_ge_num_valid",
        all(v >= route["num_valid_tokens"] for b in range(route["n_blocks"])
            for v in route["sorted_token_ids"][b * cfg["block_m"] + 1:(b + 1) * cfg["block_m"]]),
        "the core treats offs >= num_valid_tokens as padding; a smaller marker would make a pad "
        "row compute against a real token")
    real = [v for v in route["sorted_token_ids"] if v < route["num_valid_tokens"]]
    chk("route_covers_every_routed_slot",
        sorted(real) == list(range(cfg["tokens"] * cfg["top_k"])))
    chk("route_ntp_matches_blocks",
        route["num_tokens_post_padded"] == route["n_blocks"] * cfg["block_m"])
    try:
        build_route(1, 4, 512, 16, [0, 1, 2])
        chk("route_rejects_short_assignment", False)
    except ProbeError:
        chk("route_rejects_short_assignment", True)

    # ---- collision route ----
    rr = build_route(2, 4, 8, 16, [3, 3, 3, 3, 5, 5, 6, 7])
    chk("collision_route_groups_by_expert", rr["n_blocks"] == 4 and rr["block_expert"] == [3, 5, 6, 7])

    # ---- mask ----
    for m in range(route["n_blocks"] + 1):
        he, ach = choose_host_blocks(random.Random(m), route, m)
        chk(f"mask_achieves[{m}]", ach == m, f"host_experts={len(he)} achieved={ach}")
    try:
        choose_host_blocks(rng, route, route["n_blocks"] + 1)
        chk("mask_rejects_overflow", False)
    except ProbeError:
        chk("mask_rejects_overflow", True)

    # ---- expected-output assembly ----
    row_bytes = gemm_block_row_bytes("gemm1_silu", cfg, lay)
    nb = route["n_blocks"]
    ref_d = bytes(bytearray([0xAA]) * row_bytes * nb)
    ref_h = bytes(bytearray([0xBB]) * row_bytes * nb)
    he, _ = choose_host_blocks(random.Random(1), route, 3)
    asm = assemble_expected(ref_d, ref_h, route, he, row_bytes)
    got_host_blocks = sum(1 for b in range(nb)
                          if asm[b * row_bytes:b * row_bytes + 1] == b"\xbb")
    chk("assembly_picks_exactly_the_host_blocks", got_host_blocks == 3,
        f"{got_host_blocks} of {nb} blocks came from the host reference")
    chk("assembly_differs_from_both_refs", asm != ref_d and asm != ref_h)

    # ---- stats ----
    st = stats([1.0, 2.0, 3.0, 4.0, 100.0])
    chk("stats_median_is_robust", st["median"] == 3.0 and st["max"] == 100.0, json.dumps(st))
    chk("stats_empty_is_safe", stats([]) == {"n": 0})

    # ---- fit: the direction of each verdict ----
    n = cfg["top_k"]
    fl = fit_curve(_synthetic_curve("linear", n))
    chk("fit_linear_classifies_LINEAR", fl["shape"] == "LINEAR",
        f"cliff_index={fl.get('cliff_index')}")
    chk("fit_linear_recovers_slope", abs(fl["linear_fit"]["slope_ms_per_miss"] - 0.9) < 1e-6)
    chk("fit_linear_W_is_about_one",
        abs(fl["parallel_service_fit"]["effective_miss_concurrency"] - 1.0) < 1e-6,
        _f(fl["parallel_service_fit"]["effective_miss_concurrency"]))
    fc = fit_curve(_synthetic_curve("cliff", n))
    chk("fit_cliff_classifies_CLIFFED", fc["shape"] == "CLIFFED",
        f"cliff_index={fc.get('cliff_index')}")
    chk("fit_cliff_W_is_infinite",
        math.isinf(fc["parallel_service_fit"]["effective_miss_concurrency"]))
    fp_ = fit_curve(_synthetic_curve("partial", n))
    chk("fit_partial_classifies_PARTIAL", fp_["shape"] == "PARTIAL",
        f"cliff_index={fp_.get('cliff_index')}")
    ff = fit_curve(_synthetic_curve("flat", n))
    chk("fit_flat_is_INDETERMINATE_not_a_shape", ff["shape"] == "INDETERMINATE",
        "a curve that does not separate must refuse to name a shape")
    chk("fit_needs_two_points", fit_curve({0: 1.0})["status"] == "INSUFFICIENT_POINTS")

    # ---- hit-rate analysis ----
    rows_l = hit_rate_analysis(_synthetic_curve("linear", n), n, cfg["hit_rates"])
    rows_c = hit_rate_analysis(_synthetic_curve("cliff", n), n, cfg["hit_rates"])
    chk("binomial_is_exact_over_full_sweep", all(r["exact_binomial"] for r in rows_l))
    chk("linear_curve_ties_per_expert_to_layer_granular",
        all(abs(r["per_expert_gain_over_layer_granular"] - 1.0) < 1e-9 for r in rows_l),
        "on a LINEAR curve, per-expert and layer-granular placement are worth exactly the same at "
        "equal byte budget -- the gain comes from the curve's CONVEXITY, so this is the correct "
        "null and it is asserted rather than assumed")
    chk("cliff_curve_makes_per_expert_LOSE",
        all(r["per_expert_gain_over_layer_granular"] <= 1.0 + 1e-9 for r in rows_c),
        json.dumps([_f(r["per_expert_gain_over_layer_granular"]) for r in rows_c]))
    chk("hit_rate_analysis_handles_partial_sweep",
        len(hit_rate_analysis({0: 1.0, 5: 5.0, 10: 10.0}, n, [0.5])) == 1
        and not hit_rate_analysis({0: 1.0, 5: 5.0, 10: 10.0}, n, [0.5])[0]["exact_binomial"])

    # ---- decisions ----
    chk("decide_linear", decide(fl, rows_l)["verdict"] == "LINEAR")
    chk("decide_cliff", decide(fc, rows_c)["verdict"] == "CLIFFED")
    chk("decide_indeterminate", decide(ff, [])["verdict"] == "INDETERMINATE")

    # ---- layer combination ----
    arms = {
        "gemm1_silu": {"curve_median_ms": {str(m): v
                                           for m, v in _synthetic_curve("linear", n).items()}},
        "gemm2": {"curve_median_ms": {str(m): v
                                      for m, v in _synthetic_curve("cliff", n, 0.5, 4.5).items()}},
    }
    comb = combine_layer(arms)
    chk("layer_total_sums_the_arms",
        abs(float(comb["curve_median_ms"]["10"]) - (10.0 + 5.0)) < 1e-9,
        json.dumps(comb["curve_median_ms"]))
    chk("layer_total_names_a_shape", comb["fit"]["shape"] in
        ("LINEAR", "PARTIAL", "CLIFFED", "INDETERMINATE"))
    chk("layer_total_handles_no_overlap",
        combine_layer({"a": {"curve_median_ms": {"0": 1.0}},
                       "b": {"curve_median_ms": {"3": 1.0}}})["status"] == "NO_COMMON_MISS_POINTS")

    # ---- artifact shape ----
    fake_dev = {
        "rocm_index": 0, "pci_bdf": "0000:03:00.0", "name": "selftest", "status": "OK",
        "arms": {}, "layer_total": {"fit": fc, "hit_rate_analysis": rows_c,
                                    "decision": decide(fc, rows_c),
                                    "curve_median_ms": {}},
        "verdict": decide(fc, rows_c),
        "preconditions": {"passed": True, "checks": [], "failed": [],
                          "device_read_gbps": 690.0, "host_read_gbps": 28.9,
                          "copy_engine_gbps": 28.7, "media_separation_ratio": 23.9},
        "correctness": {"status": "PASS", "arms": {}},
        "indirection_control": {"arms": {}},
    }
    result = {
        "schema_version": SCHEMA_VERSION, "probe_id": PROBE_ID, "utc_start": _utc(),
        "utc_end": _utc(), "status": "SELFTEST", "config": cfg, "argv": sys.argv,
        "static_env": collect_static_env(),
        "build": {"hipcc_version": ["selftest"]},
        "kernels_src": {"path": cfg["kernels_src"], "sha": None, "dirty": None},
        "box_state_before": {"label": "selftest"},
        "devices": [fake_dev],
        "overall": overall_verdict([fake_dev]),
    }
    try:
        validate_result(result)
        chk("json_shape_valid", True)
    except ProbeError as e:
        chk("json_shape_valid", False, str(e))
    try:
        result["_rendered_md_chars"] = len(render_md(result))
        chk("md_renders", result["_rendered_md_chars"] > 800,
            f"{result.get('_rendered_md_chars')} chars")
    except Exception as e:                                    # noqa: BLE001
        chk("md_renders", False, f"{type(e).__name__}: {e}")
    try:
        validate_result({"schema_version": 1})
        chk("validate_rejects_incomplete", False)
    except ProbeError:
        chk("validate_rejects_incomplete", True)

    # ---- the .so, if it has been built (compile is CPU-only; loading needs no GPU) ----
    if os.path.isfile(SO_PATH):
        try:
            load_kernels(check_only=True)
            chk("so_exports_every_required_symbol", True, SO_PATH)
        except ProbeError as e:
            chk("so_exports_every_required_symbol", False, str(e))
    else:
        chk("so_present", False, f"{SO_PATH} not built yet (run without --selftest, or --rebuild)")

    passed = all(c["ok"] for c in checks)
    return {
        "schema_version": SCHEMA_VERSION, "probe_id": PROBE_ID + "-selftest",
        "utc": _utc(), "status": "PASS" if passed else "FAIL",
        "n_checks": len(checks), "n_failed": sum(1 for c in checks if not c["ok"]),
        "checks": checks, "config_under_test": cfg,
        "artifact_shape_sample": result,
        "note": ("no GPU was touched: no device was set, no HIP allocation was made, and the "
                 "shared library was only inspected for symbols"),
    }


def render_selftest_md(r: dict) -> str:
    L = [f"# P2prime — selftest **{r['status']}**", "",
         f"{r['n_checks'] - r['n_failed']}/{r['n_checks']} checks passed · {r['utc']}", "",
         f"> {r['note']}", "", "| check | ok | detail |", "|---|---|---|"]
    for c in r["checks"]:
        L.append(f"| `{c['check']}` | {'✅' if c['ok'] else '❌'} | {c.get('detail', '')[:200]} |")
    L.append("")
    return "\n".join(L) + "\n"


# ===========================================================================
# main
# ===========================================================================

def _write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        fh.write(text)
    os.replace(tmp, path)


def main(argv: list[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv[1:])
    try:
        cfg = finalize_cfg(args)
    except ProbeError as e:
        print(f"[P2prime] ABORT (arguments): {e}", file=sys.stderr)
        return 2

    tag = ("." + cfg["tag"]) if cfg["tag"] else ""
    outdir = cfg["outdir"]

    if args.selftest:
        try:
            r = selftest(args)
        except ProbeError as e:
            print(f"[P2prime] selftest ABORT: {e}", file=sys.stderr)
            return 2
        _write(os.path.join(outdir, f"p2prime{tag}.selftest.json"), json.dumps(r, indent=2))
        _write(os.path.join(outdir, f"p2prime{tag}.selftest.md"), render_selftest_md(r))
        print(f"[P2prime] selftest {r['status']}: "
              f"{r['n_checks'] - r['n_failed']}/{r['n_checks']} checks")
        for c in r["checks"]:
            if not c["ok"]:
                print(f"    FAIL {c['check']}: {c.get('detail', '')[:200]}")
        return 0 if r["status"] == "PASS" else 1

    json_path = os.path.join(outdir, f"p2prime{tag}.json")
    md_path = os.path.join(outdir, f"p2prime{tag}.md")

    result: dict = {
        "schema_version": SCHEMA_VERSION, "probe_id": PROBE_ID, "utc_start": _utc(),
        "status": "RUNNING", "config": cfg, "argv": list(argv),
        "static_env": collect_static_env(),
        "build": {}, "kernels_src": {},
        "box_state_before": collect_box_state("run_before"),
        "devices": [],
        "overall": {"status": "RUNNING", "per_card": {}, "summary": "in progress"},
    }

    def checkpoint() -> None:
        """Persist the artifact after every stage.

        A run that dies mid-sweep must still leave evidence: the P2 attempt that aborted is only
        readable today because it wrote what it had. Durable and recorded, never tmpfs.
        """
        try:
            _write(json_path, json.dumps(result, indent=2, default=str))
        except OSError:
            pass

    rc = 0
    try:
        result["kernels_src"] = kernels_src_provenance(cfg["kernels_src"], args.allow_dirty_kernels)
        result["build"] = build_kernels(cfg["arch"], args.rebuild, cfg["hipcc"], cfg["kernels_src"])
        checkpoint()
        lib = load_kernels()
        n = ctypes.c_int(0)
        K(lib, lib.p2p_device_count(ctypes.byref(n)), "p2p_device_count")
        result["hip_device_count"] = int(n.value)
        maxe = lib.p2p_max_experts()
        if cfg["num_experts"] > maxe:
            _fail(f"--num-experts {cfg['num_experts']} exceeds the compiled table size {maxe} "
                  f"(P2P_MAX_E in p2prime_kernels.hip)")
        for idx in cfg["devices"]:
            if idx >= int(n.value):
                _fail(f"ROCm device {idx} does not exist (count={n.value}); with "
                      f"ROCR_VISIBLE_DEVICES=0,1 only the two compute cards are enumerated")
            print(f"[P2prime] === ROCm device {idx} ===", flush=True)
            try:
                d = run_device(lib, idx, cfg, checkpoint)
            except ProbeError as e:
                d = {"rocm_index": idx, "pci_bdf": None, "status": "ABORTED",
                     "abort_reason": str(e), "utc_end": _utc()}
                rc = 3
            result["devices"].append(d)
            print(f"[P2prime]     status={d['status']} "
                  f"shape={((d.get('layer_total') or {}).get('fit') or {}).get('shape')}",
                  flush=True)
            checkpoint()
        result["overall"] = overall_verdict(result["devices"])
        result["status"] = result["overall"]["status"]
        if any(d.get("status") != "OK" for d in result["devices"]):
            rc = rc or 3
    except ProbeError as e:
        result["status"] = "ABORTED"
        result["abort_reason"] = str(e)
        result["overall"] = {"status": "ABORTED", "per_card": {}, "summary": str(e)}
        print(f"[P2prime] ABORT: {e}", file=sys.stderr)
        rc = 2
    except KeyboardInterrupt:
        result["status"] = "INTERRUPTED"
        result["overall"] = {"status": "INTERRUPTED", "per_card": {}, "summary": "Ctrl-C"}
        rc = 130
    finally:
        result["utc_end"] = _utc()
        result["box_state_after"] = collect_box_state("run_after")
        try:
            validate_result(result)
            result["artifact_validated"] = True
        except ProbeError as e:
            result["artifact_validated"] = False
            result["artifact_validation_error"] = str(e)
            rc = rc or 4
        _write(json_path, json.dumps(result, indent=2, default=str))
        try:
            _write(md_path, render_md(result))
        except Exception as e:                                # noqa: BLE001
            _write(md_path, f"# P2prime — render failed\n\n`{type(e).__name__}: {e}`\n\n"
                            f"The JSON at `{json_path}` is authoritative.\n")
        print(f"[P2prime] status={result['status']} -> {json_path}")
        print(f"[P2prime]                          -> {md_path}")
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv))
