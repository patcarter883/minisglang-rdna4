#!/usr/bin/env python3
"""Isolated-replay harness: the SAME HIP kernel, at the SAME served shape, under FOUR cache states.

WHY THIS EXISTS
The decode GEMV reaches 55-81% of the 706.6 GB/s roofline when benched in isolation and 13.0-69.5%
when the identical kernel runs inside the minisgl serve at the shapes the serve actually issues. A
microbench that relaunches one kernel back-to-back over one resident weight tensor is not measuring
the served condition: it is measuring a 64 MB MALL that the previous launch just warmed. This script
holds the KERNEL and the SHAPE fixed and varies ONLY the cache state around it, so whatever is left
of the gap after condition D is genuinely in-kernel and not a cache-residency artifact.

THE FOUR CONDITIONS (the whole point -- see --conditions)
  hot         back-to-back launches over ONE set of weight tensors. The classic microbench, and the
              condition under which "81% of roofline" was measured.
  rotated     N copies of the read-heavy operands cycled round-robin so the working set exceeds the
              MALL by BYTES (--pool-bytes, default 192 MB = 3x the 64 MB MALL). Every launch reads
              cold. If ONE copy already exceeds the target the pool is still >= 2 copies and the
              actual copy count is reported in the CSV -- read it before concluding anything.
  evict       hot weights, but a cache-flushing stream of --evict-bytes between every launch. Set
              that from the MEASURED per-step elementwise/norm byte volume of the serve
              (results/serve/bytes-bs*.json), not from a guess.
  neighbours  hot weights, but between launches the ACTUAL ops that surround this one in the real
              layer (RMSNorm / SiLU / residual add / RoPE / store_kv), at the served shapes. The
              highest-fidelity reproduction of what the serve does between two GEMMs. Declared
              per-case in the shape spec ("neighbours": [ {op, shape}, ... ]).

TIMING RULES THAT ARE LOAD-BEARING
  * Per-dispatch device time comes from a hipEvent PAIR around EACH launch, so the flush/neighbour
    work between launches perturbs the CACHE without being counted as the op's time. The same
    instrumentation runs in every condition -- never compare an event-timed number against a
    wall-clock one.
  * The SAME statistic is reported for every condition: median / min / p90 over all per-dispatch
    samples from all measured rounds, after a discarded warmup. Taking min in one condition and mean
    in another manufactures a difference that is not there.
  * %-of-roofline is against 706.6 GB/s, the gfx1201 HBM spec ceiling. NEVER 640, never 644, never a
    previously achieved bench number -- an achieved figure as a denominator inflates every result.
  * bytes_model is computed from the SHAPE by a per-op byte function (weights read + activations
    read + output written), not measured. It is a model; the byte formula used is emitted per row so
    a wrong model can be seen and corrected rather than silently believed.

USAGE
  # no GPU needed -- validates argument construction and prints the planned calls
  python3 iso_replay.py --spec shapes.json --dry-run

  # real run (under a lease, in the serve image, PYTHONPATH=/opt/kernels:/engine/python:/engine)
  python3 iso_replay.py --spec shapes.json --out iso_replay.csv \
      --conditions hot,rotated,evict,neighbours --reps 50 --rounds 7

  python3 iso_replay.py --list-ops        # every op this harness can construct inputs for

SHAPE SPEC (JSON) -- shapes come from the serve trace, never from this file
  {
    "cases": [
      {"name": "lmhead_gemv",
       "op": "fp8_wmma.dense_bf16_gemv",
       "shape": {"M": 1, "K": 2048, "N": 75776, "dtype": "bf16"},
       "evict_bytes": 331776,
       "neighbours": [{"op": "tail_hip.rms_norm", "shape": {"T": 1, "H": 2048}}]}
    ]
  }
Per-case "evict_bytes" overrides --evict-bytes; per-case "neighbours" is required for condition D
(a case without it SKIPS D with a printed reason rather than silently running condition A again).
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import glob
import json
import os
import statistics
import sys
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

# gfx1201 HBM SPEC ceiling. Not an achieved bench figure -- see the module docstring.
ROOFLINE_GBS = 706.6
# 64 MB MALL (last-level cache) on gfx1201; the rotated pool must exceed it by bytes.
MALL_BYTES = 64 << 20
DEFAULT_POOL_BYTES = 3 * MALL_BYTES  # 192 MB


# --------------------------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------------------------
def read_perf_level() -> str:
    """`power_dpm_force_performance_level` for every DRM card, e.g. 'card1=high'. A run at a
    different DPM level is a different machine; recording it is what makes two CSVs comparable."""
    out = []
    for p in sorted(glob.glob("/sys/class/drm/card*/device/power_dpm_force_performance_level")):
        card = p.split("/")[4]
        try:
            with open(p) as f:
                out.append(f"{card}={f.read().strip()}")
        except Exception as e:  # container may not map /sys/class/drm
            out.append(f"{card}=unreadable({type(e).__name__})")
    return ";".join(out) if out else "unavailable"


# --------------------------------------------------------------------------------------------
# tensor construction context
# --------------------------------------------------------------------------------------------
_DTYPES = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "fp32": torch.float32,
    "float32": torch.float32,
    "fp8": torch.float8_e4m3fn,
    "fp8_e4m3": torch.float8_e4m3fn,
    "float8_e4m3fn": torch.float8_e4m3fn,
    "int8": torch.int8,
    "int32": torch.int32,
    "uint8": torch.uint8,
}


def dt(name: str) -> torch.dtype:
    try:
        return _DTYPES[str(name).lower()]
    except KeyError:
        raise SkipOp(f"unknown dtype {name!r}; known: {sorted(_DTYPES)}")


class SkipOp(Exception):
    """A case this harness cannot construct inputs for. Carries the reason; never crashes the run."""


@dataclass
class Ctx:
    device: torch.device
    dry: bool

    def rand(self, shape, dtype: torch.dtype) -> torch.Tensor:
        """Random tensor of `dtype`. On --dry-run everything is allocated on the META device: shapes
        and dtypes are fully checked, nothing is materialised, and a 620 MB LM-head weight costs
        nothing. Meta tensors are never passed to a kernel (dry-run does not call)."""
        if self.dry:
            return torch.empty(tuple(shape), dtype=dtype, device="meta")
        if dtype in (torch.float8_e4m3fn,):
            return (torch.randn(tuple(shape), device=self.device) * 0.25).to(dtype)
        if dtype.is_floating_point:
            return torch.randn(tuple(shape), device=self.device, dtype=torch.float32).to(dtype)
        if dtype in (torch.int8,):
            return torch.randint(-8, 8, tuple(shape), device=self.device, dtype=dtype)
        if dtype in (torch.uint8,):
            return torch.randint(0, 255, tuple(shape), device=self.device, dtype=dtype)
        return torch.randint(0, 1 << 20, tuple(shape), device=self.device, dtype=dtype)

    def zeros(self, shape, dtype: torch.dtype) -> torch.Tensor:
        if self.dry:
            return torch.empty(tuple(shape), dtype=dtype, device="meta")
        return torch.zeros(tuple(shape), device=self.device, dtype=dtype)

    def arange(self, n: int, dtype: torch.dtype) -> torch.Tensor:
        if self.dry:
            return torch.empty((n,), dtype=dtype, device="meta")
        return torch.arange(n, device=self.device, dtype=dtype)

    def randint(self, lo: int, hi: int, shape, dtype: torch.dtype) -> torch.Tensor:
        if self.dry:
            return torch.empty(tuple(shape), dtype=dtype, device="meta")
        return torch.randint(lo, hi, tuple(shape), device=self.device, dtype=dtype)


def esz(dtype: torch.dtype) -> int:
    return torch.empty(0, dtype=dtype).element_size()


@dataclass
class Plan:
    """One constructed call. `rotate_idx` names the LARGE READ-ONLY argument positions -- the ones
    whose residency the rotated condition is meant to defeat. For a GEMM that is the weight; for an
    elementwise op it is the activation, because that is what the op streams."""
    fn: Callable
    args: List[Any]
    rotate_idx: Tuple[int, ...]
    bytes_model: int
    bytes_formula: str
    shape_str: str
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def call(self, args: Optional[List[Any]] = None):
        return self.fn(*(self.args if args is None else args), **self.kwargs)

    def rotate_bytes(self) -> int:
        return sum(
            a.numel() * a.element_size()
            for i, a in enumerate(self.args)
            if i in self.rotate_idx and torch.is_tensor(a)
        )


# --------------------------------------------------------------------------------------------
# builders -- one per op. Argument ORDER and DTYPES mirror the engine call sites (the ground
# truth), not the schema alone: file:line of the site is quoted above each builder.
# --------------------------------------------------------------------------------------------
BUILDERS: Dict[str, Callable[[dict, Ctx], Plan]] = {}
UNSUPPORTED: Dict[str, str] = {}


def builder(name: str):
    def deco(f):
        BUILDERS[name] = f
        return f

    return deco


def _need(sh: dict, *keys) -> tuple:
    missing = [k for k in keys if k not in sh]
    if missing:
        raise SkipOp(f"shape spec missing key(s) {missing}; needs {list(keys)}")
    return tuple(sh[k] for k in keys)


def _imp(pkg: str):
    try:
        return __import__(pkg)
    except Exception as e:
        raise SkipOp(f"package {pkg!r} did not import: {type(e).__name__}: {e}")


# ---- dense GEMV / GEMM ----------------------------------------------------------------------

@builder("fp8_wmma.dense_bf16_gemv")
def _b_dense_bf16_gemv(sh, ctx):
    """python/minisgl/layers/minv.py:192 and gdn/layer.py:170 -> fn(x.contiguous(), w).
    w is the nn.Linear weight [out_features, in_features] = [N, K]; gate requires K % 8 == 0."""
    M, K, N = _need(sh, "M", "K", "N")
    d = dt(sh.get("dtype", "bf16"))
    if K % 8:
        raise SkipOp(f"dense_bf16_gemv requires K % 8 == 0 (engine gate w.shape[-1] % 8); K={K}")
    m = _imp("fp8_wmma")
    x = ctx.rand((M, K), d)
    w = ctx.rand((N, K), d)
    e = esz(d)
    return Plan(
        fn=m.dense_bf16_gemv, args=[x, w], rotate_idx=(1,),
        bytes_model=M * K * e + N * K * e + M * N * e,
        bytes_formula="M*K*e + N*K*e + M*N*e",
        shape_str=f"M={M},K={K},N={N},{sh.get('dtype','bf16')}",
    )


def _dense_gemm_common(sh, ctx, opname: str, extra: Sequence[Any] = ()):
    M, K, N = _need(sh, "M", "K", "N")
    block_m = int(sh.get("block_m", 16))
    BN = int(sh.get("BN", 64))
    d = dt(sh.get("dtype", "bf16"))
    dg = _imp("dense_gemm")
    fn = getattr(dg, opname, None)
    if fn is None:
        raise SkipOp(f"dense_gemm has no callable {opname!r} (loaded package predates it)")
    # minv.py pads M up to the tile for the full-tile kernels and slices the pad off afterwards.
    Mp = ((M + block_m - 1) // block_m) * block_m if opname != "dense_gemm" else M
    A = ctx.rand((Mp, K), d)
    W = ctx.rand((N, K), d)
    e = esz(d)
    return Plan(
        fn=fn, args=[A, W, block_m, BN, *extra], rotate_idx=(1,),
        bytes_model=Mp * K * e + N * K * e + Mp * N * e,
        bytes_formula="Mpad*K*e + N*K*e + Mpad*N*e",
        shape_str=f"M={M}(pad{Mp}),K={K},N={N},bm={block_m},BN={BN}",
    )


@builder("dense_gemm.dense_gemm")
def _b_dense_gemm(sh, ctx):
    """python/minisgl/layers/minv.py -- the ragged-OUT (LDS) arm."""
    return _dense_gemm_common(sh, ctx, "dense_gemm")


@builder("dense_gemm.dense_gemm_rd")
def _b_dense_gemm_rd(sh, ctx):
    """python/minisgl/layers/minv.py -- the register-direct arm (small M / narrow N)."""
    return _dense_gemm_common(sh, ctx, "dense_gemm_rd")


@builder("dense_gemm.dense_gemm_pipe")
def _b_dense_gemm_pipe(sh, ctx):
    """python/minisgl/layers/minv.py -- the pipelined arm; MI/PBK/ADIV are its occupancy knobs."""
    return _dense_gemm_common(
        sh, ctx, "dense_gemm_pipe",
        extra=(int(sh.get("MI", 1)), int(sh.get("PBK", 64)), int(sh.get("ADIV", 1))),
    )


# ---- quantized dense (W4A8 / W8A8) -----------------------------------------------------------

@builder("fp8_wmma.mmq_fp8_gemm")
def _b_mmq_fp8_gemm(sh, ctx):
    """python/minisgl/quant/kernels.py:1582 (w4a8_linear). w_packed (N, K/8) int32; scales (N, K/g)
    fp16; w_zeros (N/8, K/g) int32 for AWQ-asym or None for symmetric. `kernel` is a NAME -- the op
    has no safe default (the package raises rather than silently running the scalar reference)."""
    M, K, N = _need(sh, "M", "K", "N")
    g = int(sh.get("group_size", 128))
    kern = sh.get("kernel", "decode_gemv")
    e2m1 = bool(sh.get("weight_is_e2m1", False))
    asym = bool(sh.get("asym", False))
    d = dt(sh.get("dtype", "bf16"))
    if K % 8:
        raise SkipOp(f"w_packed is (N, K/8) int32; K must be a multiple of 8, got K={K}")
    if K % g:
        raise SkipOp(f"K must be a multiple of group_size; K={K} g={g}")
    if kern == "decode_gemv" and K % 32:
        raise SkipOp(f"decode_gemv requires K % 32 == 0 (_W4A8_GEMV_K_MULTIPLE); K={K}")
    if asym and e2m1:
        raise SkipOp("e2m1 (MXFP4) is symmetric by construction; w_zeros must be None")
    if asym and N % 8:
        raise SkipOp(f"AWQ w_zeros is (N/8, K/g) int32; N must be a multiple of 8, got N={N}")
    m = _imp("fp8_wmma")
    x = ctx.rand((M, K), d)
    wp = ctx.rand((N, K // 8), torch.int32)
    sc = ctx.rand((N, K // g), torch.float16)
    wz = ctx.rand((N // 8, K // g), torch.int32) if asym else None
    e = esz(d)
    return Plan(
        fn=m.mmq_fp8_gemm, args=[x, wp, sc], rotate_idx=(1, 2),
        kwargs={"kernel": kern, "w_zeros": wz, "weight_is_e2m1": e2m1},
        bytes_model=M * K * e + N * K // 2 + N * (K // g) * 2 + M * N * e,
        bytes_formula="M*K*e + N*K/2(int4) + N*(K/g)*2(scales) + M*N*e",
        shape_str=f"M={M},K={K},N={N},g={g},{kern}{'+e2m1' if e2m1 else ''}"
                  f"{'+asym' if asym else ''}",
    )


@builder("fp8_wmma.mmq_fp8_gemm_silu")
def _b_mmq_fp8_gemm_silu(sh, ctx):
    """python/minisgl/quant/kernels.py (w4a8_linear_silu): fused gate_up GEMV + silu_and_mul.
    w_packed is (2*inter, K/8); output is (M, inter)."""
    M, K, I = _need(sh, "M", "K", "inter")
    g = int(sh.get("group_size", 128))
    d = dt(sh.get("dtype", "bf16"))
    if K % 32 or K % g:
        raise SkipOp(f"needs K%32==0 and K%group_size==0; K={K} g={g}")
    m = _imp("fp8_wmma")
    x = ctx.rand((M, K), d)
    wp = ctx.rand((2 * I, K // 8), torch.int32)
    sc = ctx.rand((2 * I, K // g), torch.float16)
    e = esz(d)
    return Plan(
        fn=m.mmq_fp8_gemm_silu, args=[x, wp, sc], rotate_idx=(1, 2),
        kwargs={"w_zeros": None, "weight_is_e2m1": bool(sh.get("weight_is_e2m1", False))},
        bytes_model=M * K * e + 2 * I * K // 2 + 2 * I * (K // g) * 2 + M * I * e,
        bytes_formula="M*K*e + 2I*K/2 + 2I*(K/g)*2 + M*inter*e",
        shape_str=f"M={M},K={K},inter={I},g={g}",
    )


@builder("fp8_wmma.mmq_w8a8_gemm")
def _b_mmq_w8a8_gemm(sh, ctx):
    """fp8 (e4m3) weights + per-output-channel f32 scale; activations quantized in-op."""
    M, K, N = _need(sh, "M", "K", "N")
    d = dt(sh.get("dtype", "bf16"))
    m = _imp("fp8_wmma")
    x = ctx.rand((M, K), d)
    w = ctx.rand((N, K), torch.float8_e4m3fn)
    sc = ctx.rand((N,), torch.float32)
    e = esz(d)
    return Plan(
        fn=m.mmq_w8a8_gemm, args=[x, w, sc], rotate_idx=(1,),
        kwargs={"kernel": int(sh.get("kernel", -1))},
        bytes_model=M * K * e + N * K + N * 4 + M * N * e,
        bytes_formula="M*K*e + N*K(fp8) + N*4 + M*N*e",
        shape_str=f"M={M},K={K},N={N}",
    )


@builder("fp8_wmma.rxf_linear")
def _b_rxf_linear(sh, ctx):
    """python/minisgl/quant/kernels.py:1093 -- the int8-act / NL-int4-weight dense path. The
    rotate+quant producer is a SEPARATE dispatch; time it as its own case if you want it."""
    M, K, N = _need(sh, "M", "K", "N")
    span = int(sh.get("span", 32))
    if K % 2 or K % 32:
        raise SkipOp(f"w_packed (N, K/2) uint8 + w_scale (N, K/32); K={K}")
    m = _imp("fp8_wmma")
    q = ctx.rand((M, K), torch.int8)
    a_scale = ctx.rand((M,), torch.float32)
    wp = ctx.rand((N, K // 2), torch.uint8)
    ws = ctx.rand((N, K // 32), torch.float16)
    nl = ctx.rand((16,), torch.int8)
    return Plan(
        fn=m.rxf_linear, args=[q, a_scale, wp, ws, nl, None], rotate_idx=(2, 3),
        bytes_model=M * K + M * 4 + N * K // 2 + N * (K // 32) * 2 + M * N * 2,
        bytes_formula="M*K(int8) + M*4 + N*K/2(int4) + N*(K/32)*2 + M*N*2(bf16 out)",
        shape_str=f"M={M},K={K},N={N},span={span}",
    )


@builder("fp8_wmma.rxf_rotate_quant_int8")
def _b_rxf_rotate_quant(sh, ctx):
    M, K = _need(sh, "M", "K")
    span = int(sh.get("span", 32))
    d = dt(sh.get("dtype", "bf16"))
    m = _imp("fp8_wmma")
    x = ctx.rand((M, K), d)
    e = esz(d)
    return Plan(
        fn=m.rxf_rotate_quant_int8, args=[x, span], rotate_idx=(0,),
        bytes_model=M * K * e + M * K + M * 4,
        bytes_formula="M*K*e(read) + M*K(int8 out) + M*4(scale)",
        shape_str=f"M={M},K={K},span={span}",
    )


# ---- MoE ------------------------------------------------------------------------------------

def _moe_align(ctx, M, top_k, E, block_m):
    """The engine's route metadata (moe_hip.moe_align). On --dry-run the align kernel cannot run, so
    the derived tensors are ANALYTIC placeholders and the row is marked dry -- their exact contents
    only matter on device."""
    mh = _imp("moe_hip")
    if ctx.dry:
        P = M * top_k + E * (block_m - 1)
        return (torch.empty((P,), dtype=torch.int32, device="meta"),
                torch.empty((P // block_m,), dtype=torch.int32, device="meta"),
                torch.empty((1,), dtype=torch.int32, device="meta"),
                P)
    ids = torch.randint(0, E, (M, top_k), device=ctx.device, dtype=torch.int32)
    sti, eid, ntp = mh.moe_align(ids, E, block_m)
    return sti, eid, ntp, int(sti.numel())


@builder("moe_hip.moe_align")
def _b_moe_align(sh, ctx):
    M, top_k, E = _need(sh, "M", "top_k", "num_experts")
    block_m = int(sh.get("block_m", 16))
    mh = _imp("moe_hip")
    ids = ctx.randint(0, E, (M, top_k), torch.int32)
    return Plan(
        fn=mh.moe_align, args=[ids, E, block_m], rotate_idx=(0,),
        bytes_model=M * top_k * 4 * 2 + E * 4,
        bytes_formula="M*top_k*4(read) + ~M*top_k*4(sorted out) + E*4",
        shape_str=f"M={M},top_k={top_k},E={E},bm={block_m}",
    )


@builder("moe_hip.moe_topk_softmax")
def _b_moe_topk_softmax(sh, ctx):
    M, E, top_k = _need(sh, "M", "num_experts", "top_k")
    mh = _imp("moe_hip")
    g = ctx.rand((M, E), dt(sh.get("dtype", "bf16")))
    e = esz(dt(sh.get("dtype", "bf16")))
    return Plan(
        fn=mh.moe_topk_softmax, args=[g, top_k, bool(sh.get("renormalize", True))],
        rotate_idx=(0,),
        bytes_model=M * E * e + M * top_k * 8,
        bytes_formula="M*E*e + M*top_k*(4+4)",
        shape_str=f"M={M},E={E},top_k={top_k}",
    )


@builder("moe_hip.moe_route_align")
def _b_moe_route_align(sh, ctx):
    """python/minisgl/quant/kernels.py::_route_align -- router + sort + align FUSED into one op."""
    M, E, top_k = _need(sh, "M", "num_experts", "top_k")
    block_m = int(sh.get("block_m", 16))
    mh = _imp("moe_hip")
    g = ctx.rand((M, E), dt(sh.get("dtype", "bf16")))
    e = esz(dt(sh.get("dtype", "bf16")))
    return Plan(
        fn=mh.moe_route_align,
        args=[g, top_k, bool(sh.get("renormalize", True)), E, block_m], rotate_idx=(0,),
        bytes_model=M * E * e + M * top_k * 8 + (M * top_k + E * block_m) * 4,
        bytes_formula="M*E*e + M*top_k*8 + (M*top_k + E*block_m)*4",
        shape_str=f"M={M},E={E},top_k={top_k},bm={block_m}",
    )


@builder("fp8_wmma.mmq_fp8_moe_gemm1_silu")
def _b_moe_gemm1_silu(sh, ctx):
    """python/minisgl/quant/kernels.py::w4a8_moe gemm1 arm. w13 (E, 2*inter, K/8) int32."""
    M, K, I, E, top_k = _need(sh, "M", "K", "inter", "num_experts", "top_k")
    g = int(sh.get("group_size", 128))
    block_m = int(sh.get("block_m", 16))
    kern = sh.get("kernel", "gemv")
    d = dt(sh.get("dtype", "bf16"))
    if K % 8 or K % g:
        raise SkipOp(f"w13 is (E, 2*inter, K/8) int32 with (K/g) scales; K={K} g={g}")
    m = _imp("fp8_wmma")
    sti, eid, ntp, P = _moe_align(ctx, M, top_k, E, block_m)
    x = ctx.rand((M, K), d)
    w13 = ctx.rand((E, 2 * I, K // 8), torch.int32)
    sc = ctx.rand((E, 2 * I, K // g), torch.float16)
    e = esz(d)
    # Only the experts actually routed are read; at decode that is <= min(E, M*top_k).
    Eact = min(E, max(1, M * top_k))
    return Plan(
        fn=m.mmq_fp8_moe_gemm1_silu, args=[x, w13, sc, sti, eid, ntp, top_k, block_m],
        rotate_idx=(1, 2), kwargs={"kernel": kern, "w_zeros": None,
                                   "weight_is_e2m1": bool(sh.get("weight_is_e2m1", False))},
        bytes_model=M * top_k * K * e + Eact * 2 * I * K // 2
        + Eact * 2 * I * (K // g) * 2 + P * I * e,
        bytes_formula="M*top_k*K*e + Eact*2I*K/2 + Eact*2I*(K/g)*2 + P*inter*e "
                      f"(Eact=min(E,M*top_k)={Eact}, P={P})",
        shape_str=f"M={M},K={K},inter={I},E={E},top_k={top_k},g={g},bm={block_m},{kern}",
    )


@builder("fp8_wmma.mmq_fp8_moe_gemm2_gather_reduce")
def _b_moe_gemm2_gr(sh, ctx):
    """python/minisgl/quant/kernels.py -- fused down-proj + weighted gather-reduce. x is the
    PRE-SORTED (P, inter) post-silu buffer; w2 is (E, hidden, inter/8) int32."""
    M, K, I, E, top_k = _need(sh, "M", "K", "inter", "num_experts", "top_k")
    g = int(sh.get("group_size", 128))
    block_m = int(sh.get("block_m", 16))
    d = dt(sh.get("dtype", "bf16"))
    if I % 8 or I % g:
        raise SkipOp(f"w2 is (E, hidden, inter/8) int32 with (inter/g) scales; inter={I} g={g}")
    m = _imp("fp8_wmma")
    sti, eid, ntp, P = _moe_align(ctx, M, top_k, E, block_m)
    x = ctx.rand((P, I), d)
    w2 = ctx.rand((E, K, I // 8), torch.int32)
    sc = ctx.rand((E, K, I // g), torch.float16)
    tw = ctx.rand((M, top_k), torch.float32)
    e = esz(d)
    Eact = min(E, max(1, M * top_k))
    return Plan(
        fn=m.mmq_fp8_moe_gemm2_gather_reduce,
        args=[x, w2, sc, sti, eid, ntp, tw, top_k, block_m], rotate_idx=(1, 2),
        kwargs={"w_zeros": None, "weight_is_e2m1": bool(sh.get("weight_is_e2m1", False))},
        bytes_model=P * I * e + Eact * K * I // 2 + Eact * K * (I // g) * 2 + M * K * 4,
        bytes_formula="P*inter*e + Eact*hidden*inter/2 + Eact*hidden*(inter/g)*2 + M*hidden*4(f32 out)"
                      f" (Eact={Eact}, P={P})",
        shape_str=f"M={M},hidden={K},inter={I},E={E},top_k={top_k},g={g},bm={block_m}",
    )


@builder("fp8_wmma.mmq_fp8_moe_gather_reduce")
def _b_moe_gather_reduce(sh, ctx):
    M, N, E, top_k = _need(sh, "M", "N", "num_experts", "top_k")
    block_m = int(sh.get("block_m", 16))
    d = dt(sh.get("dtype", "bf16"))
    m = _imp("fp8_wmma")
    sti, eid, ntp, P = _moe_align(ctx, M, top_k, E, block_m)
    out2 = ctx.rand((P, N), d)
    tw = ctx.rand((M, top_k), torch.float32)
    e = esz(d)
    return Plan(
        fn=m.mmq_fp8_moe_gather_reduce, args=[out2, sti, tw, ntp, top_k], rotate_idx=(0,),
        bytes_model=M * top_k * N * e + M * N * 4 + M * top_k * 4,
        bytes_formula="M*top_k*N*e(read) + M*N*4(f32 out) + M*top_k*4",
        shape_str=f"M={M},N={N},E={E},top_k={top_k},P={P}",
    )


# ---- attention -------------------------------------------------------------------------------

def _paged_kv(ctx, sh, dtype):
    bs, Hkv, D, ctxlen = _need(sh, "bs", "num_kv_heads", "head_dim", "ctx_len")
    page = int(sh.get("page_size", 16))
    pages_per_seq = (ctxlen + page - 1) // page
    num_pages = int(sh.get("num_pages", max(1, bs * pages_per_seq)))
    if num_pages < bs * pages_per_seq:
        raise SkipOp(f"num_pages={num_pages} < bs*ceil(ctx/page)={bs * pages_per_seq}")
    k = ctx.rand((num_pages, page, Hkv, D), dtype)
    v = ctx.rand((num_pages, page, Hkv, D), dtype)
    if ctx.dry:
        bt = torch.empty((bs, pages_per_seq), dtype=torch.int32, device="meta")
    else:
        bt = (torch.arange(bs * pages_per_seq, device=ctx.device, dtype=torch.int32)
              % num_pages).view(bs, pages_per_seq)
    lens = (torch.empty((bs,), dtype=torch.int32, device="meta") if ctx.dry
            else torch.full((bs,), ctxlen, device=ctx.device, dtype=torch.int32))
    return k, v, bt, lens, num_pages, page, pages_per_seq


@builder("attn_decode.flash_decode_paged")
def _b_flash_decode_paged(sh, ctx):
    """python/minisgl/attention/rdna4.py:372 -> op(q, k_cache, v_cache, block_table, ctx_lens,
    scale, 0). k_cache/v_cache are the per-layer [num_pages, page_size, kv_heads, head_dim] views."""
    bs, Hq, Hkv, D, ctxlen = _need(sh, "bs", "num_q_heads", "num_kv_heads", "head_dim", "ctx_len")
    d = dt(sh.get("dtype", "bf16"))
    m = _imp("attn_decode")
    k, v, bt, lens, npages, page, pps = _paged_kv(ctx, sh, d)
    q = ctx.rand((bs, Hq, D), d)
    e = esz(d)
    return Plan(
        fn=m.flash_decode_paged,
        args=[q, k, v, bt, lens, float(D) ** -0.5, int(sh.get("sliding_window", 0))],
        rotate_idx=(1, 2),
        bytes_model=bs * Hq * D * e + 2 * bs * ctxlen * Hkv * D * e + bs * Hq * D * e
        + bs * pps * 4 + bs * 4,
        bytes_formula="q + 2*bs*ctx*Hkv*D*e(KV read) + out + block_table + ctx_lens",
        shape_str=f"bs={bs},Hq={Hq},Hkv={Hkv},D={D},ctx={ctxlen},page={page},pages={npages}",
    )


@builder("attn_decode.flash_decode_paged_fp8")
def _b_flash_decode_paged_fp8(sh, ctx):
    """Same site, fp8-KV arm: + per-head k_descale/v_descale [num_kv_heads] f32."""
    bs, Hq, Hkv, D, ctxlen = _need(sh, "bs", "num_q_heads", "num_kv_heads", "head_dim", "ctx_len")
    dq = dt(sh.get("dtype", "bf16"))
    m = _imp("attn_decode")
    k, v, bt, lens, npages, page, pps = _paged_kv(ctx, sh, torch.float8_e4m3fn)
    q = ctx.rand((bs, Hq, D), dq)
    kd = ctx.rand((Hkv,), torch.float32)
    vd = ctx.rand((Hkv,), torch.float32)
    e = esz(dq)
    return Plan(
        fn=m.flash_decode_paged_fp8,
        args=[q, k, v, bt, lens, float(D) ** -0.5, kd, vd, int(sh.get("sliding_window", 0))],
        rotate_idx=(1, 2),
        bytes_model=bs * Hq * D * e + 2 * bs * ctxlen * Hkv * D + bs * Hq * D * e
        + bs * pps * 4 + bs * 4,
        bytes_formula="q + 2*bs*ctx*Hkv*D*1(fp8 KV read) + out + block_table + ctx_lens",
        shape_str=f"bs={bs},Hq={Hq},Hkv={Hkv},D={D},ctx={ctxlen},page={page},pages={npages},fp8KV",
    )


@builder("attn_hip.flash_prefill")
def _b_flash_prefill(sh, ctx):
    """python/minisgl/attention/rdna4.py:394 -- dense single-sequence cold prefill."""
    T, Hq, Hkv, D = _need(sh, "tokens", "num_q_heads", "num_kv_heads", "head_dim")
    d = dt(sh.get("dtype", "bf16"))
    m = _imp("attn_hip")
    q = ctx.rand((T, Hq, D), d)
    k = ctx.rand((T, Hkv, D), d)
    v = ctx.rand((T, Hkv, D), d)
    e = esz(d)
    return Plan(
        fn=m.flash_prefill,
        args=[q, k, v, float(D) ** -0.5, int(sh.get("causal", 1)),
              int(sh.get("sliding_window", 0)), None],
        rotate_idx=(0, 1, 2),
        bytes_model=T * Hq * D * e + 2 * T * Hkv * D * e + T * Hq * D * e,
        bytes_formula="q + k + v + out (streamed once; tile re-reads are cache traffic, not HBM)",
        shape_str=f"T={T},Hq={Hq},Hkv={Hkv},D={D},causal={sh.get('causal', 1)}",
    )


@builder("attn_prefill_paged.flash_prefill_paged")
def _b_flash_prefill_paged(sh, ctx):
    """python/minisgl/attention/rdna4.py:448 -- extend / paged-prefix prefill."""
    T, bs, Hq, Hkv, D, ctxlen = _need(
        sh, "tokens", "bs", "num_q_heads", "num_kv_heads", "head_dim", "ctx_len")
    d = dt(sh.get("dtype", "bf16"))
    m = _imp("attn_prefill_paged")
    k, v, bt, lens, npages, page, pps = _paged_kv(ctx, sh, d)
    q = ctx.rand((T, Hq, D), d)
    per = T // bs
    if ctx.dry:
        cu = torch.empty((bs + 1,), dtype=torch.int32, device="meta")
    else:
        cu = (torch.arange(bs + 1, device=ctx.device, dtype=torch.int32) * per)
    e = esz(d)
    return Plan(
        fn=m.flash_prefill_paged,
        args=[q, k, v, bt, cu, lens, float(D) ** -0.5, int(sh.get("causal", 1)),
              int(sh.get("sliding_window", 0)), int(sh.get("max_seqlen_q", per))],
        rotate_idx=(1, 2),
        bytes_model=T * Hq * D * e + 2 * bs * ctxlen * Hkv * D * e + T * Hq * D * e,
        bytes_formula="q + 2*bs*ctx*Hkv*D*e + out",
        shape_str=f"T={T},bs={bs},Hq={Hq},Hkv={Hkv},D={D},ctx={ctxlen},page={page}",
    )


@builder("mla_hip.mla_decode")
def _b_mla_decode(sh, ctx):
    """python/minisgl/attention/mla.py:102 -- absorbed MLA decode over the paged LATENT cache
    [num_pages, page_size, kv_lora + qk_rope_head_dim]; q is [bs, H, latent_dim]."""
    bs, H, lora, rope, ctxlen = _need(sh, "bs", "num_q_heads", "kv_lora_rank",
                                      "qk_rope_head_dim", "ctx_len")
    d = dt(sh.get("dtype", "bf16"))
    latent = lora + rope
    page = int(sh.get("page_size", 16))
    pps = (ctxlen + page - 1) // page
    npages = int(sh.get("num_pages", max(1, bs * pps)))
    m = _imp("mla_hip")
    q = ctx.rand((bs, H, latent), d)
    lc = ctx.rand((npages, page, latent), d)
    if ctx.dry:
        bt = torch.empty((bs, pps), dtype=torch.int32, device="meta")
        lens = torch.empty((bs,), dtype=torch.int32, device="meta")
    else:
        bt = (torch.arange(bs * pps, device=ctx.device, dtype=torch.int32) % npages).view(bs, pps)
        lens = torch.full((bs,), ctxlen, device=ctx.device, dtype=torch.int32)
    e = esz(d)
    return Plan(
        fn=m.mla_decode,
        args=[q, lc, bt, lens, float(lora) ** -0.5, int(sh.get("sliding_window", 0)), 0, rope],
        rotate_idx=(1,),
        bytes_model=bs * H * latent * e + bs * ctxlen * latent * e + bs * H * lora * e,
        bytes_formula="q + bs*ctx*latent*e(ONE latent read, not K+V) + out",
        shape_str=f"bs={bs},H={H},lora={lora},rope={rope},ctx={ctxlen},page={page}",
    )


# ---- tail (elementwise / norm / rope / store) --------------------------------------------------

@builder("tail_hip.rms_norm")
def _b_rms_norm(sh, ctx):
    T, H = _need(sh, "T", "H")
    d = dt(sh.get("dtype", "bf16"))
    m = _imp("tail_hip")
    x = ctx.rand((T, H), d)
    w = ctx.rand((H,), d)
    e = esz(d)
    return Plan(
        fn=m.rms_norm, args=[x, w, float(sh.get("eps", 1e-6)), int(sh.get("plus_one", 0))],
        rotate_idx=(0,), bytes_model=2 * T * H * e + H * e,
        bytes_formula="T*H*e(read) + T*H*e(write) + H*e",
        shape_str=f"T={T},H={H}",
    )


@builder("tail_hip.rms_norm_add")
def _b_rms_norm_add(sh, ctx):
    T, H = _need(sh, "T", "H")
    d = dt(sh.get("dtype", "bf16"))
    m = _imp("tail_hip")
    x = ctx.rand((T, H), d)
    res = ctx.rand((T, H), d)
    w = ctx.rand((H,), d)
    e = esz(d)
    return Plan(
        fn=m.rms_norm_add, args=[x, res, w, float(sh.get("eps", 1e-6)),
                                 int(sh.get("plus_one", 0))],
        rotate_idx=(0, 1), bytes_model=4 * T * H * e + H * e,
        bytes_formula="read x + read residual + write residual + write out + H*e",
        shape_str=f"T={T},H={H}",
    )


@builder("tail_hip.rms_norm_quant")
def _b_rms_norm_quant(sh, ctx):
    """PRODUCER-side act-quant twin (quant/kernels.py consumes its (x_fp8, act_scales) pair)."""
    T, H = _need(sh, "T", "H")
    d = dt(sh.get("dtype", "bf16"))
    m = _imp("tail_hip")
    fn = getattr(m, "rms_norm_quant", None)
    if fn is None:
        raise SkipOp("loaded tail_hip has no rms_norm_quant (pre-producer-fusion .so)")
    x = ctx.rand((T, H), d)
    w = ctx.rand((H,), d)
    e = esz(d)
    return Plan(
        fn=fn, args=[x, w, float(sh.get("eps", 1e-6)), int(sh.get("plus_one", 0))],
        rotate_idx=(0,), bytes_model=T * H * e + T * H * e + T * H + T * 4 + H * e,
        bytes_formula="read T*H*e + write T*H*e(bf16 out) + write T*H(fp8) + write T*4(scale)",
        shape_str=f"T={T},H={H}",
    )


@builder("tail_hip.rms_norm_add_quant")
def _b_rms_norm_add_quant(sh, ctx):
    T, H = _need(sh, "T", "H")
    d = dt(sh.get("dtype", "bf16"))
    m = _imp("tail_hip")
    fn = getattr(m, "rms_norm_add_quant", None)
    if fn is None:
        raise SkipOp("loaded tail_hip has no rms_norm_add_quant (pre-producer-fusion .so)")
    x = ctx.rand((T, H), d)
    res = ctx.rand((T, H), d)
    w = ctx.rand((H,), d)
    e = esz(d)
    return Plan(
        fn=fn, args=[x, res, w, float(sh.get("eps", 1e-6)), int(sh.get("plus_one", 0))],
        rotate_idx=(0, 1),
        bytes_model=3 * T * H * e + T * H * e + T * H + T * 4 + H * e,
        bytes_formula="read x + read res + write res + write out + write fp8 + write scale",
        shape_str=f"T={T},H={H}",
    )


def _gated_mul(opname):
    def f(sh, ctx):
        T, I = _need(sh, "T", "inter")
        d = dt(sh.get("dtype", "bf16"))
        m = _imp("tail_hip")
        fn = getattr(m, opname, None)
        if fn is None:
            raise SkipOp(f"loaded tail_hip has no {opname} (it is optional -- see _tail_hip.py)")
        x = ctx.rand((T, 2 * I), d)
        e = esz(d)
        return Plan(fn=fn, args=[x], rotate_idx=(0,), bytes_model=3 * T * I * e,
                    bytes_formula="T*2I*e(read) + T*I*e(write)",
                    shape_str=f"T={T},inter={I}")
    return f


BUILDERS["tail_hip.silu_and_mul"] = _gated_mul("silu_and_mul")
BUILDERS["tail_hip.gelu_and_mul"] = _gated_mul("gelu_and_mul")
BUILDERS["tail_hip.gelu_tanh_and_mul"] = _gated_mul("gelu_tanh_and_mul")


@builder("tail_hip.rope")
def _b_rope(sh, ctx):
    """python/minisgl/layers/rotary.py:104 -- x is FLAT [T, num_heads*head_size]; cache is the full
    fp32 cat(cos,sin) table indexed by int32 positions."""
    T, Hn, D = _need(sh, "T", "num_heads", "head_size")
    rd = int(sh.get("rotary_dim", D))
    maxpos = int(sh.get("max_position", 4096))
    d = dt(sh.get("dtype", "bf16"))
    m = _imp("tail_hip")
    x = ctx.rand((T, Hn * D), d)
    pos = ctx.randint(0, maxpos, (T,), torch.int32)
    cache = ctx.rand((maxpos, rd), torch.float32)
    e = esz(d)
    return Plan(
        fn=m.rope, args=[x, pos, cache, D, rd], rotate_idx=(0,),
        bytes_model=2 * T * Hn * D * e + T * 4 + T * rd * 4,
        bytes_formula="T*H*D*e(read) + T*H*D*e(write) + T*4(pos) + T*rotary_dim*4(cos/sin gather)",
        shape_str=f"T={T},heads={Hn},D={D},rd={rd}",
    )


@builder("tail_hip.store_kv")
def _b_store_kv(sh, ctx):
    """python/minisgl/kvcache/mha_pool.py:189 -- fused cast + per-head 1/scale + scatter. The cache
    views are FLAT [num_pages*page_size, kv_heads, head_dim] at this call site."""
    T, Hkv, D = _need(sh, "T", "num_kv_heads", "head_dim")
    slots = int(sh.get("slots", 4096))
    d = dt(sh.get("dtype", "bf16"))
    cd = dt(sh.get("cache_dtype", sh.get("dtype", "bf16")))
    m = _imp("tail_hip")
    k = ctx.rand((T, Hkv, D), d)
    v = ctx.rand((T, Hkv, D), d)
    kc = ctx.rand((slots, Hkv, D), cd)
    vc = ctx.rand((slots, Hkv, D), cd)
    loc = ctx.randint(0, slots, (T,), torch.int32)
    fp8 = cd == torch.float8_e4m3fn
    ki = ctx.rand((Hkv,), torch.float32) if fp8 else None
    vi = ctx.rand((Hkv,), torch.float32) if fp8 else None
    e, ec = esz(d), esz(cd)
    return Plan(
        fn=m.store_kv, args=[k, v, kc, vc, loc, 1.0, 1.0, ki, vi], rotate_idx=(0, 1),
        bytes_model=2 * T * Hkv * D * e + 2 * T * Hkv * D * ec + T * 4,
        bytes_formula="2*T*Hkv*D*e(read k,v) + 2*T*Hkv*D*ec(write cache) + T*4(out_loc)",
        shape_str=f"T={T},Hkv={Hkv},D={D},slots={slots},cache={sh.get('cache_dtype', 'same')}",
    )


# ---- GDN -------------------------------------------------------------------------------------

def _gdn_dims(sh):
    nk, nv, hk, hv = _need(sh, "num_k_heads", "num_v_heads", "head_k_dim", "head_v_dim")
    key_dim, value_dim = nk * hk, nv * hv
    conv_dim = key_dim * 2 + value_dim
    return nk, nv, hk, hv, key_dim, value_dim, conv_dim


@builder("gdn_hip.gdn_decode_conv_gated")
def _b_gdn_decode_conv_gated(sh, ctx):
    """python/minisgl/gdn/layer.py:603 -- conv_update + gated-delta-rule + gated RMSNorm, ONE kernel.
    conv_weight and norm_weight are the CACHED fp32 copies (_conv_weights_fp32 / _norm_weight_fp32);
    state_indices arrives as int64 (state_indices.long()); z is FLAT [B*num_v_heads, head_v_dim]."""
    B, = _need(sh, "B")
    nk, nv, hk, hv, key_dim, value_dim, conv_dim = _gdn_dims(sh)
    ck = int(sh.get("conv_kernel_size", 4))
    slots = int(sh.get("num_slots", 64))
    d = dt(sh.get("dtype", "bf16"))
    sd = dt(sh.get("state_dtype", "fp32"))
    m = _imp("gdn_hip")
    fn = getattr(m, "gdn_decode_conv_gated", None)
    if fn is None:
        raise SkipOp("loaded gdn_hip has no gdn_decode_conv_gated (pre-fusion .so)")
    mixed = ctx.rand((B, conv_dim), d)
    cw = ctx.rand((conv_dim, ck), torch.float32)
    cstate = ctx.rand((slots, conv_dim, ck - 1), torch.float32)
    a = ctx.rand((B, nv), d)
    b = ctx.rand((B, nv), d)
    A_log = ctx.rand((nv,), torch.float32)
    dtb = ctx.rand((nv,), torch.float32)
    ssm = ctx.rand((slots, nv, hv, hk), sd)
    idx = (torch.empty((B,), dtype=torch.int64, device="meta") if ctx.dry
           else torch.arange(B, device=ctx.device, dtype=torch.int64))
    z = ctx.rand((B * nv, hv), d)
    nw = ctx.rand((hv,), torch.float32)
    e, es = esz(d), esz(sd)
    return Plan(
        fn=fn,
        args=[mixed, cw, None, cstate, a, b, A_log, dtb, ssm, idx, z, nw,
              float(sh.get("eps", 1e-5)), 1, float(hk) ** -0.5, 1],
        rotate_idx=(0,),
        bytes_model=(B * conv_dim * e + conv_dim * ck * 4 + 2 * B * conv_dim * (ck - 1) * 4
                     + 2 * B * nv * hv * hk * es + B * nv * hv * e + B * nv * hv * e),
        bytes_formula="mixed_qkv + conv_w + 2*B*conv_dim*(k-1)*4(conv_state rw) + "
                      "2*B*nv*hv*hk*es(ssm_state rw) + z + out",
        shape_str=f"B={B},conv_dim={conv_dim},ck={ck},nv={nv},hv={hv},hk={hk},"
                  f"state={sh.get('state_dtype', 'fp32')}",
    )


@builder("gdn_hip.causal_conv1d_update")
def _b_conv_update(sh, ctx):
    """python/minisgl/gdn/layer.py:613 -- the unfused conv arm (MINISGL_GDN_FUSED_CONV=0)."""
    B, = _need(sh, "B")
    nk, nv, hk, hv, key_dim, value_dim, conv_dim = _gdn_dims(sh)
    ck = int(sh.get("conv_kernel_size", 4))
    slots = int(sh.get("num_slots", 64))
    d = dt(sh.get("dtype", "bf16"))
    m = _imp("gdn_hip")
    mixed = ctx.rand((B, conv_dim), d)
    cw = ctx.rand((conv_dim, ck), torch.float32)
    cstate = ctx.rand((slots, conv_dim, ck - 1), torch.float32)
    idx = (torch.empty((B,), dtype=torch.int64, device="meta") if ctx.dry
           else torch.arange(B, device=ctx.device, dtype=torch.int64))
    e = esz(d)
    return Plan(
        fn=m.causal_conv1d_update, args=[mixed, cw, None, cstate, idx, 1], rotate_idx=(0,),
        bytes_model=2 * B * conv_dim * e + conv_dim * ck * 4 + 2 * B * conv_dim * (ck - 1) * 4,
        bytes_formula="read+write mixed + conv_w + conv_state read/write",
        shape_str=f"B={B},conv_dim={conv_dim},ck={ck}",
    )


@builder("gdn_hip.rmsnorm_gated")
def _b_rmsnorm_gated(sh, ctx):
    """python/minisgl/gdn/layer.py:311 -- norm-before-gate + SiLU over [rows, head_v_dim]."""
    rows, hv = _need(sh, "rows", "head_v_dim")
    d = dt(sh.get("dtype", "bf16"))
    m = _imp("gdn_hip")
    x = ctx.rand((rows, hv), d)
    z = ctx.rand((rows, hv), d)
    w = ctx.rand((hv,), torch.float32)
    e = esz(d)
    return Plan(
        fn=m.rmsnorm_gated, args=[x, z, w, float(sh.get("eps", 1e-5))], rotate_idx=(0, 1),
        bytes_model=3 * rows * hv * e + hv * 4,
        bytes_formula="read x + read z + write out + hv*4",
        shape_str=f"rows={rows},hv={hv}",
    )


# ---- misc ------------------------------------------------------------------------------------

@builder("swiglu_hip.fused_swiglu")
def _b_fused_swiglu(sh, ctx):
    """NOTE: registered but NOT called anywhere in python/minisgl -- see --list-ops. F.linear
    convention: w_gate_up [2*inter, hidden], w_down [hidden, inter]."""
    M, H, I = _need(sh, "M", "hidden", "inter")
    d = dt(sh.get("dtype", "bf16"))
    m = _imp("swiglu_hip")
    x = ctx.rand((M, H), d)
    wgu = ctx.rand((2 * I, H), d)
    wd = ctx.rand((H, I), d)
    e = esz(d)
    return Plan(
        fn=m.fused_swiglu, args=[x, wgu, wd], rotate_idx=(1, 2),
        bytes_model=M * H * e + 2 * I * H * e + H * I * e + M * H * e,
        bytes_formula="M*H*e + 2I*H*e + H*I*e + M*H*e",
        shape_str=f"M={M},hidden={H},inter={I}",
    )


@builder("sampler_hip.top_k_top_p_sampling_from_logits")
def _b_sampler(sh, ctx):
    """python/minisgl/engine/_sampler_hip.py:63 -- the in-place op; uniforms are caller-owned."""
    bs, vocab = _need(sh, "bs", "vocab")
    m = _imp("sampler_hip")
    fn = getattr(m, "top_k_top_p_sampling_from_logits_", None)
    if fn is None:
        raise SkipOp("sampler_hip has no top_k_top_p_sampling_from_logits_ callable")
    rounds = int(getattr(m, "MAX_ROUNDS", 32))
    logits = ctx.rand((bs, vocab), torch.float32)
    temps = (torch.empty((bs,), dtype=torch.float32, device="meta") if ctx.dry
             else torch.full((bs,), 0.7, device=ctx.device, dtype=torch.float32))
    tk = (torch.empty((bs,), dtype=torch.int32, device="meta") if ctx.dry
          else torch.full((bs,), int(sh.get("top_k", 50)), device=ctx.device, dtype=torch.int32))
    tp = (torch.empty((bs,), dtype=torch.float32, device="meta") if ctx.dry
          else torch.full((bs,), float(sh.get("top_p", 0.95)), device=ctx.device,
                          dtype=torch.float32))
    u = ctx.rand((rounds, bs), torch.float32)
    out = ctx.zeros((bs,), torch.int32)
    return Plan(
        fn=fn, args=[logits, temps, tk, tp, u, out], rotate_idx=(0,),
        bytes_model=bs * vocab * 4 + rounds * bs * 4 + bs * 4,
        bytes_formula="bs*vocab*4(logits) + MAX_ROUNDS*bs*4(uniforms) + bs*4(out)",
        shape_str=f"bs={bs},vocab={vocab},rounds={rounds}",
    )


# ---- deliberately unsupported -----------------------------------------------------------------
UNSUPPORTED.update({
    "custom_ar.one_shot_ar":
        "needs a PEER process: peer_buf_ptr/peer_flags_ptr are raw device pointers opened from "
        "another rank's IPC handle. Single-process replay would deadlock on the flag spin, not "
        "measure the kernel. Time it with the two-rank TP=2 serve, not here.",
    "custom_ar.all_gather_p2p": "same as one_shot_ar -- requires a live peer rank's IPC pointers.",
    "custom_ar.alloc_shared": "an allocator, not a compute kernel; no byte model applies.",
    "custom_ar.get_ipc_handle": "IPC plumbing, not a compute kernel.",
    "custom_ar.open_ipc_handle": "IPC plumbing, not a compute kernel.",
    "custom_ar.flag_probe": "IPC plumbing; also unused in python/minisgl.",
    "zaya_cca.cca_decode_qk":
        "ZAYA-only. The (w0,b0,w1,b1,temp_eff) MLP shapes are read off a loaded ZAYA checkpoint in "
        "models/zaya.py, not derivable from a shape triple; constructing them here would be a "
        "guess. Add a builder once a ZAYA case is actually needed, from that call site.",
    "zaya_cca.cca_decode_fused": "ZAYA-only -- same reason as cca_decode_qk.",
    "zaya_cca.cca_prefill_qk": "ZAYA-only -- same reason as cca_decode_qk.",
    "zaya_cca.zaya_merge_norm": "ZAYA-only -- per-layer scale/bias vectors come from the checkpoint.",
    "gdn_hip.gdn_decode_conv_gated_replay":
        "needs the ReplaySSM ring (k/vr/g/vn/len/s0n) built by gdn_hip.make_replay_ring against a "
        "live GDNStateCache; the ring's fill state changes what the kernel does, so a synthetic "
        "zero ring measures the empty-ring branch only. Build from GDNStateCache when needed.",
    "gdn_hip.gdn_verify_replay": "same ring dependency as gdn_decode_conv_gated_replay.",
    "gdn_hip.gdn_replay_flush": "same ring dependency; also only fires every L steps.",
    "gdn_hip.gdn_replay_rollback": "same ring dependency.",
    "gdn_hip.causal_conv1d_bwd": "TRAINING-only (autograd); never on the served path.",
    "gdn_hip.rmsnorm_gated_bwd": "TRAINING-only (autograd); never on the served path.",
})


# --------------------------------------------------------------------------------------------
# conditions
# --------------------------------------------------------------------------------------------
def _clone_argset(plan: Plan) -> List[Any]:
    """One rotation copy: the rotate_idx tensors are cloned, everything else is SHARED. Cloning the
    non-rotated args too would change the working set for a reason unrelated to the condition."""
    args = list(plan.args)
    for i in plan.rotate_idx:
        if torch.is_tensor(args[i]):
            args[i] = args[i].clone()
    return args


def plan_pool(plan: Plan, pool_bytes: int, pool_max_bytes: int, reps: int) -> Tuple[int, int, str]:
    """(copies, bytes_actually_touched, note) for the rotated condition.

    The cap at `reps` is not a nicety, it is the difference between a real condition and a lie: the
    timed loop indexes the pool `i % copies` for i in [0, reps), so copies BEYOND reps are allocated
    and never read. A 20 KB activation with a 192 MB target asks for 24576 copies but at reps=50
    touches 50 of them -- 1 MB, comfortably MALL-resident, i.e. condition B silently degenerates
    into condition A. Cap it and SAY SO, so the fix (raise --reps) is visible rather than a
    plausible-looking row."""
    wb = plan.rotate_bytes()
    if wb <= 0:
        return 1, 0, "no rotatable operand -- rotated == hot"
    needed = max(2, -(-pool_bytes // wb))
    copies, notes = needed, []
    if copies * wb > pool_max_bytes:
        copies = max(2, pool_max_bytes // wb)
        notes.append(f"pool capped {needed}->{copies} copies by --pool-max-bytes "
                     f"({needed * wb / 2**20:.0f} MB > {pool_max_bytes / 2**20:.0f} MB)")
    if copies > reps:
        notes.append(f"pool capped {copies}->{reps} copies by --reps (copies past reps are never "
                     f"launched)")
        copies = reps
    touched = copies * wb
    if touched < pool_bytes:
        notes.append(f"TOUCHED WORKING SET {touched / 2**20:.2f} MB < target "
                     f"{pool_bytes / 2**20:.0f} MB -- raise --reps to reach it")
    if wb >= MALL_BYTES:
        notes.append(f"ONE copy ({wb / 2**20:.1f} MB) already exceeds the "
                     f"{MALL_BYTES / 2**20:.0f} MB MALL -- 'hot' was never cache-resident here")
    return copies, touched, "; ".join(notes)


def build_pool(plan: Plan, copies: int) -> List[List[Any]]:
    return [plan.args] + [_clone_argset(plan) for _ in range(copies - 1)]


class Evictor:
    """Streams a fixed number of BYTES between launches. dst.copy_(src) over uint8 buffers of
    nbytes/2 each => exactly nbytes of traffic (nbytes/2 read + nbytes/2 written)."""

    def __init__(self, nbytes: int, device):
        half = max(1, nbytes // 2)
        self.src = torch.empty(half, dtype=torch.uint8, device=device)
        self.dst = torch.empty(half, dtype=torch.uint8, device=device)
        self.nbytes = half * 2

    def __call__(self):
        self.dst.copy_(self.src)


# --------------------------------------------------------------------------------------------
# timing
# --------------------------------------------------------------------------------------------
@dataclass
class Timing:
    ns_median: float
    ns_min: float
    ns_p90: float
    n_samples: int


def time_plan(plan: Plan, argsets: List[List[Any]], between: Optional[Callable],
              reps: int, rounds: int, warmup: int) -> Timing:
    """Per-dispatch device time from a hipEvent PAIR around EACH launch. `between` runs AFTER the
    closing event, so its traffic perturbs the cache without entering the op's measurement."""
    n = len(argsets)
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]

    for i in range(warmup):
        plan.call(argsets[i % n])
        if between is not None:
            between()
    torch.cuda.synchronize()

    samples: List[float] = []
    for _ in range(rounds):
        for i in range(reps):
            a = argsets[i % n]
            starts[i].record()
            plan.call(a)
            ends[i].record()
            if between is not None:
                between()
        torch.cuda.synchronize()
        samples.extend(starts[i].elapsed_time(ends[i]) * 1e6 for i in range(reps))

    samples.sort()
    p90 = samples[min(len(samples) - 1, int(round(0.90 * (len(samples) - 1))))]
    return Timing(statistics.median(samples), samples[0], p90, len(samples))


# --------------------------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------------------------
CONDITIONS = ("hot", "rotated", "evict", "neighbours")

CSV_FIELDS = [
    "date", "image", "kernels_commit", "engine_commit", "perf_level", "device_name",
    "case", "condition", "op", "shape", "dtype_note",
    "ns_median", "ns_min", "ns_p90", "n_samples",
    "bytes_model", "bytes_formula", "GB_s", "pct_roofline",
    "rotate_bytes_per_copy", "pool_copies", "pool_touched_bytes", "evict_bytes",
    "neighbour_ops", "note",
]


def build_plan(op: str, shape: dict, ctx: Ctx) -> Plan:
    if op in UNSUPPORTED:
        raise SkipOp(UNSUPPORTED[op])
    b = BUILDERS.get(op)
    if b is None:
        raise SkipOp(f"no builder for {op!r}; --list-ops shows what is constructible")
    return b(shape, ctx)


def run_case(case: dict, args, ctx: Ctx, prov: dict, rows: List[dict]) -> None:
    name = case.get("name", case.get("op", "?"))
    op = case.get("op")
    if not op:
        print(f"[SKIP] case {name!r}: no 'op' key")
        return
    shape = case.get("shape", {})
    try:
        plan = build_plan(op, shape, ctx)
    except SkipOp as e:
        print(f"[SKIP] {name} ({op}): {e}")
        return
    except Exception as e:
        print(f"[SKIP] {name} ({op}): builder raised {type(e).__name__}: {e}")
        if args.traceback:
            traceback.print_exc()
        return

    gbs_den = plan.bytes_model
    print(f"\n=== {name}  op={op}  {plan.shape_str}")
    print(f"    bytes_model = {plan.bytes_model:,}  [{plan.bytes_formula}]")
    print(f"    args        = {_describe_args(plan)}")
    print(f"    rotate_idx  = {plan.rotate_idx} ({plan.rotate_bytes() / 2**20:.2f} MB/copy)")

    if args.dry_run:
        for cond in args.conditions:
            extra = ""
            if cond == "rotated":
                c, touched, n_ = plan_pool(plan, args.pool_bytes, args.pool_max_bytes, args.reps)
                extra = (f"  -> {c} copies, {touched / 2**20:.2f} MB touched"
                         + (f"   [{n_}]" if n_ else ""))
            elif cond == "evict":
                extra = f"  -> {case.get('evict_bytes', args.evict_bytes):,} B/launch"
            elif cond == "neighbours":
                nb = case.get("neighbours")
                if not nb:
                    extra = "  -> SKIP (case declares no 'neighbours')"
                else:
                    built = []
                    for n_ in nb:
                        try:
                            build_plan(n_["op"], n_.get("shape", {}), ctx)
                            built.append(n_["op"])
                        except SkipOp as e:
                            built.append(f"{n_.get('op')}!SKIP({e})")
                    extra = "  -> " + ", ".join(built)
            print(f"    [dry] condition {cond}{extra}")
        return

    for cond in args.conditions:
        note = ""
        between = None
        argsets = [plan.args]
        # pool_touched is 0 for every condition but `rotated` -- there is no pool there, and
        # printing the weight size in that column would read as one.
        copies, pool_touched, ev_bytes, nb_names = 1, 0, 0, ""
        try:
            if cond == "rotated":
                copies, pool_touched, note = plan_pool(
                    plan, args.pool_bytes, args.pool_max_bytes, args.reps)
                argsets = build_pool(plan, copies) if copies > 1 else [plan.args]
            elif cond == "evict":
                ev_bytes = int(case.get("evict_bytes", args.evict_bytes))
                if ev_bytes <= 0:
                    print(f"[SKIP] {name}/{cond}: evict_bytes is 0 "
                          "(set --evict-bytes or the case's evict_bytes)")
                    continue
                ev = Evictor(ev_bytes, ctx.device)
                ev_bytes = ev.nbytes
                between = ev
            elif cond == "neighbours":
                nb = case.get("neighbours")
                if not nb:
                    print(f"[SKIP] {name}/neighbours: case declares no 'neighbours' list -- "
                          "condition D without one would silently re-run condition A")
                    continue
                nplans, names = [], []
                for n_ in nb:
                    try:
                        nplans.append(build_plan(n_["op"], n_.get("shape", {}), ctx))
                        names.append(n_["op"])
                    except SkipOp as e:
                        print(f"    [neighbour SKIP] {n_.get('op')}: {e}")
                if not nplans:
                    print(f"[SKIP] {name}/neighbours: no neighbour built")
                    continue
                nb_names = "|".join(names)

                def between():  # noqa: E306
                    for p in nplans:
                        p.call()

            t = time_plan(plan, argsets, between, args.reps, args.rounds, args.warmup)
        except SkipOp as e:
            print(f"[SKIP] {name}/{cond}: {e}")
            continue
        except Exception as e:
            print(f"[SKIP] {name}/{cond}: {type(e).__name__}: {e}")
            if args.traceback:
                traceback.print_exc()
            continue

        gbs = gbs_den / (t.ns_median * 1e-9) / 1e9
        rows.append({
            **prov, "case": name, "condition": cond, "op": op, "shape": plan.shape_str,
            "dtype_note": str(shape.get("dtype", "")),
            "ns_median": round(t.ns_median, 1), "ns_min": round(t.ns_min, 1),
            "ns_p90": round(t.ns_p90, 1), "n_samples": t.n_samples,
            "bytes_model": plan.bytes_model, "bytes_formula": plan.bytes_formula,
            "GB_s": round(gbs, 2), "pct_roofline": round(100.0 * gbs / ROOFLINE_GBS, 2),
            "rotate_bytes_per_copy": plan.rotate_bytes(), "pool_copies": copies,
            "pool_touched_bytes": pool_touched, "evict_bytes": ev_bytes,
            "neighbour_ops": nb_names, "note": note,
        })
        print(f"    {cond:<11} med {t.ns_median:>10.1f} ns  min {t.ns_min:>10.1f}  "
              f"p90 {t.ns_p90:>10.1f}  {gbs:>7.1f} GB/s  {100.0 * gbs / ROOFLINE_GBS:>5.1f}% "
              f"of {ROOFLINE_GBS}" + (f"   [{note}]" if note else ""))


def _describe_args(plan: Plan) -> str:
    out = []
    for a in plan.args:
        if torch.is_tensor(a):
            out.append(f"T{tuple(a.shape)}:{str(a.dtype).replace('torch.', '')}")
        else:
            out.append(repr(a))
    for k, v in plan.kwargs.items():
        out.append(f"{k}={'T' + str(tuple(v.shape)) if torch.is_tensor(v) else v!r}")
    return ", ".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", help="JSON shape spec (see module docstring). Required unless "
                                   "--list-ops.")
    ap.add_argument("--out", help="CSV output path")
    ap.add_argument("--ops", default="", help="comma-separated op filter (substring match on the "
                                              "case's op or name); default = every case")
    ap.add_argument("--conditions", default=",".join(CONDITIONS),
                    help=f"subset of {','.join(CONDITIONS)}")
    ap.add_argument("--reps", type=int, default=50, help="launches per measured round")
    ap.add_argument("--rounds", type=int, default=7, help="measured rounds (>=5 for a stable p90)")
    ap.add_argument("--warmup", type=int, default=20, help="discarded launches before timing")
    ap.add_argument("--pool-bytes", type=int, default=DEFAULT_POOL_BYTES,
                    help=f"rotated-condition target working set (default {DEFAULT_POOL_BYTES} "
                         f"= 3x the {MALL_BYTES >> 20} MB MALL)")
    ap.add_argument("--pool-max-bytes", type=int, default=4 << 30,
                    help="VRAM guard on the rotated pool (default 4 GiB)")
    ap.add_argument("--evict-bytes", type=int, default=0,
                    help="default evict-condition byte volume per launch; set it from the MEASURED "
                         "per-step elementwise/norm volume, e.g. results/serve/bytes-bs1.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry-run", action="store_true",
                    help="build every argument on the META device and print the planned calls; "
                         "no GPU, no kernel launch")
    ap.add_argument("--list-ops", action="store_true")
    ap.add_argument("--image", default=os.environ.get("ISO_REPLAY_IMAGE", "unknown"))
    ap.add_argument("--kernels-commit", default="8a8bca6")
    ap.add_argument("--engine-commit", default="4d719ad6")
    ap.add_argument("--traceback", action="store_true", help="print tracebacks for skipped cases")
    args = ap.parse_args(argv)

    if args.list_ops:
        print("CONSTRUCTIBLE (a builder exists; shape-spec keys in parentheses):")
        for k in sorted(BUILDERS):
            doc = (BUILDERS[k].__doc__ or "").strip().splitlines()
            print(f"  {k:<48} {doc[0] if doc else ''}")
        print("\nNOT CONSTRUCTIBLE (skipped with this reason):")
        for k in sorted(UNSUPPORTED):
            print(f"  {k:<48} {UNSUPPORTED[k]}")
        return 0

    if not args.spec:
        ap.error("--spec is required (or use --list-ops)")

    args.conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    bad = [c for c in args.conditions if c not in CONDITIONS]
    if bad:
        ap.error(f"unknown condition(s) {bad}; known: {list(CONDITIONS)}")

    with open(args.spec) as f:
        spec = json.load(f)
    cases = spec.get("cases", [])
    if args.ops:
        pats = [p.strip() for p in args.ops.split(",") if p.strip()]
        cases = [c for c in cases
                 if any(p in c.get("op", "") or p in c.get("name", "") for p in pats)]
    if not cases:
        print("no cases selected", file=sys.stderr)
        return 2

    dev_name = "n/a (dry-run)"
    if not args.dry_run:
        if not torch.cuda.is_available():
            print("ERROR: no HIP device visible. Run under `gpu-lease -n 1 --` with the ROCm "
                  "device flags, or use --dry-run.", file=sys.stderr)
            return 3
        dev_name = torch.cuda.get_device_name(0)

    ctx = Ctx(device=torch.device(args.device), dry=args.dry_run)
    prov = {
        "date": _dt.datetime.now().isoformat(timespec="seconds"),
        "image": args.image,
        "kernels_commit": args.kernels_commit,
        "engine_commit": args.engine_commit,
        "perf_level": read_perf_level(),
        "device_name": dev_name,
    }
    print("provenance: " + "  ".join(f"{k}={v}" for k, v in prov.items()))
    print(f"roofline    = {ROOFLINE_GBS} GB/s (gfx1201 HBM spec ceiling)")
    print(f"conditions  = {args.conditions}   reps={args.reps} rounds={args.rounds} "
          f"warmup={args.warmup}")

    rows: List[dict] = []
    for case in cases:
        run_case(case, args, ctx, prov, rows)

    if args.out and rows:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {len(rows)} rows -> {args.out}")
    elif args.out:
        print("\nno rows produced; CSV not written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
