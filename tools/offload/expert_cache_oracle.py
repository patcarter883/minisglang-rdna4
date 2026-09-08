#!/usr/bin/env python3
"""Expert-cache oracle — does a hot-expert VRAM cache beat static whole-layer placement?

MEASURE-ONLY. Nothing here runs on a GPU, imports torch, or changes what the engine serves. It
answers one go/no-go question before anyone writes a per-expert device tier:

    At our real budget (~6.7 GiB/rank of resident experts = ~21% of the expert set), what hit
    rate does a cache actually realise on a REAL qwen4_exp routing trace — and is any of it
    reachable by an ONLINE policy (LRU/LFU) rather than by static placement?

THE CONTRADICTION THIS EXISTS TO RESOLVE
----------------------------------------
Our own scoping (docs/WEIGHT_OFFLOAD_PLAN.md) closed this with "the miss curve is LINEAR
(cliff_index 0.090/0.094 vs 0.100 pure-linear), so per-expert placement gains only 1.013x/1.009x
— do NOT build the selector table". That conclusion rests on a Bernoulli(h) route model: each
reference hits with probability h independent of history. Under that model hit rate == resident
fraction by construction, and NO cache can beat static placement. It is a model assumption, not
a measurement of the route stream.

A third party reports vLLM + an LRU hot-expert VRAM cache reaching 27-30 tok/s on 2x3080-20GB at
~44% residency — far above what a memoryless route model predicts (~19-20 tok/s). llama.cpp
issue #27861 on this model family reports LRU 81.5% at 12% coverage and 90.2% at 25%.

Exactly one of those is right, and the axis they differ on is LOCALITY, which a Bernoulli(h)
model cannot represent at all. So this simulator separates three different things that all
raise the hit rate and have completely different fixes:

    SKEW      some experts are globally hotter          -> a STATIC prior wins; no cache needed
    LOCALITY  the hot SET drifts but is small at any t  -> only a DYNAMIC cache (LRU/LFU) wins
    NEITHER   memoryless uniform routing                -> nothing wins; the plan's verdict stands

`--mode hotpool` in the generator is the whole point of the design: uniform marginals (zero
exploitable skew, a static prior gains nothing) with strong recency (LRU hits hugely). It is the
case P2-prime's model cannot express and the case the third-party data says is real. T3 asserts
the simulator SEPARATES those two, which is the property the dispute turns on.

WHAT IS MEASURED VS WHAT IS ASSUMED
-----------------------------------
MEASURED (this box, today, live serve; hard-coded below and pinned by T5):
  * 0.114 ms   per device-resident MoE layer at decode
  * 0.930 ms   per fully host-streamed MoE layer at decode (10/10 experts missed)
  * 60.06 ms   today's best arm: 38 host-streamed + 10 device-resident layers = 16.65 tok/s
  * ~35 ms     of that step is the 38 streamed layers -> the dominant cost, hence this oracle
  * 13.824 MB  rank-local bytes touched per MoE layer per decode token (10 routed x 1.3824 MB)
  * 68.42 GiB  total expert-weight bytes (both ranks); ~6.7 GiB/rank is the resident budget
  * 512 experts/layer, top_k=10, 48 MoE layers, TP=2, non-EP (local id == global id)
  * 1,382,400 B rank-local per expert = w13 819200 + w2 409600 + e4m3 g16 block scales 153600
  * the ROCm realisation haircut: llama.cpp measured +3.2%/+15.3% REALISED at 81-93% NOMINAL
    hit rate (2.99x stream syncs, 9.80x H2D ops, 12.8x H2D engine time, dummy-zero-expert
    kernels still launching) -> `--derate 0.5` reports 1 + 0.5*(nominal_gain - 1) as well.

ASSUMED (stated so it can be attacked):
  * The per-miss cost is AFFINE in the miss count: S_MISS = (0.930-0.114)/10 = 0.0816 ms/miss.
    P2-prime measured t(m) as linear, so this is interpolation, not extrapolation. The residual
    curvature of that fit is the thing the plan mislabelled as a "1.013x per-expert gain"; it is
    a convexity gap at FIXED h, not a placement gain.
  * A cache hit costs the same as device residency (no partial-slab or indirection overhead).
    That makes every number here an UPPER BOUND on the realised win, which is the right
    direction for a kill gate: if the upper bound is small, no implementation can be large.
  * The byte event is one access per (layer, expert, CALL) — the grouped GEMM reads each distinct
    expert slab once per call, not once per token. Counting per (token, expert) inflates recency
    policies by 27-29% (arXiv 2608.07911) and would fabricate the answer we are testing for.

USAGE
-----
    python3 tools/offload/expert_cache_oracle.py --selftest          # T1-T7, no trace needed
    python3 tools/offload/expert_cache_oracle.py --synthetic hotpool:108,20
    python3 tools/offload/expert_cache_oracle.py --trace /home/pat/fixtures/.../route.*.bin

RUN ORDER IS A HARD RULE: `--selftest` must be green before any real trace number is admissible.
A simulator that reports an LRU win on uniform routing is broken, full stop — T1 exists to fail
it. The selftest takes ~10 s; there is no excuse for skipping it.

ONE FILE, NOT FIVE: the design called for route_trace_io.py / route_trace_gen.py /
expert_cache_sim.py plus two test modules under tests/core/. The build task then constrained
this agent to own exactly ONE new file, so all five are folded in here — the writer/reader
(TRACE IO), the generator (gen_trace), the simulator, and the T1-T7 suite behind `--selftest`.
Splitting them back out is mechanical if the capture hook wants to import the writer: nothing
below depends on anything above except through plain functions.

This file is deliberately import-free beyond the stdlib (numpy used only if present, for speed)
so it runs on the bare host with no ROCm container, and is loadable BY PATH via
importlib.util.spec_from_file_location the way tests/core/test_ghost_oracle.py:11-28 does.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import os
import random
import struct
import subprocess
import sys
import time
from array import array
from collections import deque,  OrderedDict, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

try:  # numpy is a speed-up for the stats passes only; every result is identical without it.
    import numpy as _np
except Exception:  # pragma: no cover - exercised only on a numpy-less host
    _np = None


# ---------------------------------------------------------------------------------------------
# MEASURED CONSTANTS — one place, with provenance. T5 pins the anchor; do not edit without a
# new measurement to cite.
# ---------------------------------------------------------------------------------------------

LAYERS = 48  # MoE layers in qwen4_exp (Qwen3.8-Flash-Next NVFP4)
EXPERTS = 512  # experts per layer
TOPK = 10  # routed experts per token per layer
EXPERT_BYTES = 1_382_400  # rank-local bytes/expert: 819200 (w13) + 409600 (w2) + 153600 (scales)
EXPERT_SET_GIB = LAYERS * EXPERTS * EXPERT_BYTES * 2 / (1 << 30)  # ~68.42 GiB, both ranks

T_DEV_MS = 0.114  # measured: per device-resident MoE layer, decode
T_HOST_MS = 0.930  # measured: per fully host-streamed MoE layer, decode (10/10 miss)
S_MISS_MS = (T_HOST_MS - T_DEV_MS) / TOPK  # 0.0816 ms per missed expert (affine, P2-prime t(m))
STREAMED_LAYERS_TODAY = 38  # today's best arm
DEVICE_LAYERS_TODAY = LAYERS - STREAMED_LAYERS_TODAY  # 10
STEP_MS_TODAY = 60.06  # measured wall step time of that arm
NONMOE_MS = STEP_MS_TODAY - STREAMED_LAYERS_TODAY * T_HOST_MS - DEVICE_LAYERS_TODAY * T_DEV_MS
#           = 60.06 - 35.34 - 1.14 = 23.580 ms of attention/norm/lm-head/sampler/host overhead
BASE_MS = NONMOE_MS + LAYERS * T_DEV_MS  # 29.052 ms = the h=1.0 floor
MISS_SLOPE_MS = S_MISS_MS * LAYERS * TOPK  # 39.168 ms per unit of (1 - h)
TOKS_TODAY = 1000.0 / STEP_MS_TODAY  # 16.650 tok/s
H_TODAY = DEVICE_LAYERS_TODAY / LAYERS  # 0.208333 — today's arm IS a static-layer cache

TOTAL_KEYS = LAYERS * EXPERTS  # 24576 (layer, expert) pairs per rank
GIB = 1 << 30

DEFAULT_BUDGETS_GIB = [0.0, 1.0, 2.0, 3.35, 5.0, 6.7, 8.0, 10.0, 13.4, 16.0, 31.65]
REAL_BUDGET_GIB = 6.7  # the budget every kill gate is evaluated at


def ms_step(h: float) -> float:
    """Decode step time at hit rate `h`. ms_step(10/48) == 60.06 by construction (T5)."""
    return BASE_MS + MISS_SLOPE_MS * (1.0 - h)


def tok_s(h: float) -> float:
    return 1000.0 / ms_step(h)


def misses_per_token(h: float) -> float:
    return LAYERS * TOPK * (1.0 - h)


def mb_per_token(h: float) -> float:
    """Rank-local PCIe bytes per decoded token, MB. 663.55 MB at h=0."""
    return misses_per_token(h) * EXPERT_BYTES / 1e6


def derated_tok_s(h: float, derate: float) -> float:
    """Realised tok/s after the measured ROCm haircut: realised = 1 + derate*(nominal - 1)."""
    nominal_gain = tok_s(h) / TOKS_TODAY
    return TOKS_TODAY * (1.0 + derate * (nominal_gain - 1.0))


def slots_for(cache_gib: float, expert_bytes: int = EXPERT_BYTES) -> int:
    return int(cache_gib * GIB) // expert_bytes


# ---------------------------------------------------------------------------------------------
# TRACE IO  (folded-in route_trace_io: writer + reader for the MSGLRT01 format)
# ---------------------------------------------------------------------------------------------

MAGIC = b"MSGLRT01"
VERSION = 1
HDR_FMT = "<8sIIIIIIQIIQ8x"
HDR_SIZE = struct.calcsize(HDR_FMT)  # 64
REC_FMT = "<IIHBBHH"
REC_SIZE = struct.calcsize(REC_FMT)  # 16

KIND_PREFILL = 0
KIND_DECODE = 1
KIND_OTHER = 2

FLAG_DEDUPED_UNION = 1 << 0
FLAG_BLOCKMAP_UNION = 1 << 1

_FNV_OFFSET = 0xCBF29CE484222325
_FNV_PRIME = 0x100000001B3
_U64 = (1 << 64) - 1


def fnv1a64(s: str) -> int:
    h = _FNV_OFFSET
    for b in s.encode("utf-8"):
        h = ((h ^ b) * _FNV_PRIME) & _U64
    return h


class TraceHeader:
    __slots__ = (
        "num_layers",
        "num_experts",
        "top_k",
        "tp_rank",
        "dp_rank",
        "num_records",
        "expert_bytes",
        "flags",
        "model_hash",
        "version",
    )

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k, 0))

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__slots__}


class TraceWriter:
    """Streaming writer. `num_records` is patched on close; a SIGTERM'd file keeps 0 there and
    the reader scans to EOF, so a killed run is still readable (that is the point of the field
    being patched last rather than buffered)."""

    def __init__(
        self,
        path: str,
        *,
        tp_rank: int = 0,
        dp_rank: int = 0,
        model_slug: str = "qwen4_exp",
        num_layers: int = LAYERS,
        num_experts: int = EXPERTS,
        top_k: int = TOPK,
        expert_bytes: int = EXPERT_BYTES,
        flags: int = FLAG_DEDUPED_UNION,
    ):
        self.path = path
        self.n = 0
        self._hdr = dict(
            num_layers=num_layers,
            num_experts=num_experts,
            top_k=top_k,
            tp_rank=tp_rank,
            dp_rank=dp_rank,
            expert_bytes=expert_bytes,
            flags=flags,
            model_hash=fnv1a64(model_slug),
        )
        self.f = open(path, "wb")
        self.f.write(self._pack_header(0))

    def _pack_header(self, num_records: int) -> bytes:
        h = self._hdr
        return struct.pack(
            HDR_FMT,
            MAGIC,
            VERSION,
            h["num_layers"],
            h["num_experts"],
            h["top_k"],
            h["tp_rank"],
            h["dp_rank"],
            num_records,
            h["expert_bytes"],
            h["flags"],
            h["model_hash"],
        )

    def add(
        self,
        step_id: int,
        req_uid: int,
        layer_id: int,
        kind: int,
        chunk_idx: int,
        num_tokens: int,
        ids: Sequence[int],
    ) -> None:
        ids = sorted(set(int(i) for i in ids))
        self.f.write(
            struct.pack(
                REC_FMT,
                step_id & 0xFFFFFFFF,
                req_uid & 0xFFFFFFFF,
                layer_id,
                kind,
                chunk_idx,
                num_tokens,
                len(ids),
            )
        )
        self.f.write(array("H", ids).tobytes())
        self.n += 1

    def close(self) -> None:
        self.f.flush()
        self.f.seek(0)
        self.f.write(self._pack_header(self.n))
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


class TraceReadError(RuntimeError):
    pass


class RawTrace:
    """The records of one .bin, as parallel arrays. Invariants are asserted on load."""

    def __init__(self, header: TraceHeader, path: str):
        self.header = header
        self.path = path
        self.step = array("I")
        self.uid = array("I")
        self.layer = array("H")
        self.kind = array("B")
        self.chunk = array("B")
        self.ntok = array("H")
        self.off = array("q", [0])
        self.ids = array("H")
        self.truncated = False
        # Records are per-drain-ordered, not globally monotonic (capture doc §5): the ring and the
        # host-path oversize list interleave within one drain. Counted, then sorted on load.
        self.out_of_order = 0
        self.violations: Dict[str, int] = defaultdict(int)

    @property
    def n_records(self) -> int:
        return len(self.step)

    @property
    def n_refs(self) -> int:
        return len(self.ids)


def read_trace(path: str, *, strict: bool = False) -> RawTrace:
    with open(path, "rb") as f:
        blob = f.read()
    if len(blob) < HDR_SIZE:
        raise TraceReadError(f"{path}: shorter than a header ({len(blob)} B)")
    (
        magic,
        version,
        num_layers,
        num_experts,
        top_k,
        tp_rank,
        dp_rank,
        num_records,
        expert_bytes,
        flags,
        model_hash,
    ) = struct.unpack(HDR_FMT, blob[:HDR_SIZE])
    if magic != MAGIC:
        raise TraceReadError(f"{path}: bad magic {magic!r} (expected {MAGIC!r})")
    if version != VERSION:
        raise TraceReadError(f"{path}: version {version}, this reader speaks {VERSION}")
    hdr = TraceHeader(
        version=version,
        num_layers=num_layers,
        num_experts=num_experts,
        top_k=top_k,
        tp_rank=tp_rank,
        dp_rank=dp_rank,
        num_records=num_records,
        expert_bytes=expert_bytes,
        flags=flags,
        model_hash=model_hash,
    )
    t = RawTrace(hdr, path)
    pos = HDR_SIZE
    end = len(blob)
    last_step = -1
    count = 0
    while pos + REC_SIZE <= end:
        if num_records and count >= num_records:
            break
        step, uid, layer, kind, chunk, ntok, nids = struct.unpack_from(REC_FMT, blob, pos)
        pos += REC_SIZE
        if pos + 2 * nids > end:
            t.truncated = True  # SIGTERM mid-record: drop the partial tail, keep the rest
            break
        ids = array("H")
        ids.frombytes(blob[pos : pos + 2 * nids])
        if sys.byteorder != "little":  # pragma: no cover
            ids.byteswap()
        pos += 2 * nids
        # --- invariants -------------------------------------------------------------------
        if layer >= num_layers:
            raise TraceReadError(f"{path}: layer_id {layer} >= num_layers {num_layers}")
        if nids and max(ids) >= num_experts:
            raise TraceReadError(f"{path}: expert id {max(ids)} >= num_experts {num_experts}")
        # NOT an error, and raising here was reading the format more strictly than it is written.
        # `routing_trace_capture.md` §5 states it outright: the ring (decode) and the host-path
        # `oversize` list (prefill chunks, bs>1 decode) are appended in a different order within one
        # drain, "so sort by (step_id, layer_id, chunk_idx) before analysis". Records are therefore
        # per-drain-ordered, not globally monotonic. Count how far out of order the file actually is
        # so a genuinely corrupt trace is still visible, and sort below.
        if step < last_step:
            t.out_of_order += 1
        last_step = max(last_step, step)
        if kind == KIND_DECODE:
            # A short decode route means the router emitted a DUPLICATE expert id. That is a
            # routing bug worth reporting, never worth silently accepting: it would show up
            # here as a free hit-rate bonus.
            if ntok != 1:
                t.violations["decode_num_tokens_not_1"] += 1
            if nids != top_k:
                t.violations["decode_short_route"] += 1
        t.step.append(step)
        t.uid.append(uid)
        t.layer.append(layer)
        t.kind.append(kind)
        t.chunk.append(chunk)
        t.ntok.append(ntok)
        t.ids.extend(ids)
        t.off.append(len(t.ids))
        count += 1
    if num_records and count != num_records:
        t.truncated = True
    if strict and t.violations:
        raise TraceReadError(f"{path}: invariant violations {dict(t.violations)}")
    if t.out_of_order:
        # STABLE sort by (step, layer, chunk) — the order `routing_trace_capture.md` §5 specifies.
        # Stable matters: two records that tie on the key are the ring and host-path halves of the
        # same call and must keep their emission order. Rebuilds every parallel array together;
        # `off` is a running END offset with a leading 0, so record i spans off[i]..off[i+1].
        n = len(t.step)
        order = sorted(range(n), key=lambda i: (t.step[i], t.layer[i], t.chunk[i]))
        if order != list(range(n)):
            step, uid, layer = array("I"), array("I"), array("H")
            kind, chunk, ntok = array("B"), array("B"), array("H")
            ids, off = array("H"), array("q", [0])
            for i in order:
                step.append(t.step[i]); uid.append(t.uid[i]); layer.append(t.layer[i])
                kind.append(t.kind[i]); chunk.append(t.chunk[i]); ntok.append(t.ntok[i])
                ids.extend(t.ids[t.off[i]:t.off[i + 1]])
                off.append(len(ids))
            t.step, t.uid, t.layer = step, uid, layer
            t.kind, t.chunk, t.ntok = kind, chunk, ntok
            t.ids, t.off = ids, off
    return t


# ---------------------------------------------------------------------------------------------
# SYNTHETIC GENERATION  (folded-in route_trace_gen)
# ---------------------------------------------------------------------------------------------


def _zipf_weights(n: int, alpha: float) -> List[float]:
    w = [1.0 / ((i + 1) ** alpha) for i in range(n)]
    s = sum(w)
    return [x / s for x in w]


def _alias_setup(probs: Sequence[float]):
    n = len(probs)
    q = [p * n for p in probs]
    small, large = [], []
    for i, x in enumerate(q):
        (small if x < 1.0 else large).append(i)
    alias = [0] * n
    prob = [1.0] * n
    while small and large:
        s = small.pop()
        l = large.pop()
        prob[s] = q[s]
        alias[s] = l
        q[l] = q[l] + q[s] - 1.0
        (small if q[l] < 1.0 else large).append(l)
    return prob, alias


def _alias_draw(rng: random.Random, prob, alias) -> int:
    n = len(prob)
    i = rng.randrange(n)
    return i if rng.random() < prob[i] else alias[i]


def gen_trace(
    mode: str,
    *,
    steps: int,
    layers: int = LAYERS,
    experts: int = EXPERTS,
    top_k: int = TOPK,
    seed: int = 1234,
    n_prompts: int = 16,
    prefill_every: int = 0,
    prefill_chunks: int = 2,
    prefill_touch: int = 400,
    alpha: float = 0.9,
    hot: int = 108,
    tau: int = 20,
) -> "RefStream":
    """Build a synthetic reference stream in memory.

    modes:
      uniform      top_k distinct ids i.i.d. uniform (P2-prime's "distinct" draw). No skew, no
                   locality. THE NULL: LRU must not beat static here.
      zipf:a=A     Zipf(alpha) marginals, i.i.d. draws. Skew, no locality beyond skew.
      hotpool:H,t  UNIFORM marginals with strong recency: a per-layer hot pool of size H; every
                   call draws from the pool; every `t` calls one member retires and a uniformly
                   chosen expert is admitted. Over a long run every expert is equally frequent
                   yet LRU hits hugely. This is the case a Bernoulli(h) model cannot express.
      mixed:...    zipf marginals + hotpool dynamics + injected prefill chunks touching
                   `prefill_touch` of `experts` ids, to exercise cache pollution.
    """
    rng = random.Random(seed)
    spec = mode
    base = mode.split(":", 1)[0]
    if ":" in mode:
        arg = mode.split(":", 1)[1]
        if base == "zipf":
            for part in arg.split(","):
                if part.startswith("alpha=") or part.startswith("a="):
                    alpha = float(part.split("=", 1)[1])
                else:
                    alpha = float(part)
        elif base in ("hotpool", "mixed"):
            parts = [p for p in arg.split(",") if p]
            vals = {}
            for p in parts:
                if "=" in p:
                    k, v = p.split("=", 1)
                    vals[k] = float(v)
            pos = [p for p in parts if "=" not in p]
            if pos:
                hot = int(float(pos[0]))
            if len(pos) > 1:
                tau = int(float(pos[1]))
            hot = int(vals.get("H", hot))
            tau = int(vals.get("tau", tau))
            alpha = float(vals.get("alpha", alpha))
    if base == "mixed" and prefill_every == 0:
        prefill_every = 100

    use_zipf = base in ("zipf", "mixed")
    use_pool = base in ("hotpool", "mixed")
    if use_zipf:
        w = _zipf_weights(experts, alpha)
        # A per-layer permutation so "expert 0" is not globally hot by construction — that would
        # be an artefact no real router has.
        perms = [list(range(experts)) for _ in range(layers)]
        for p in perms:
            rng.shuffle(p)
        prob, alias = _alias_setup(w)

    pools: List[List[int]] = []
    pool_sets: List[set] = []
    if use_pool:
        for _ in range(layers):
            p = rng.sample(range(experts), hot)
            pools.append(p)
            pool_sets.append(set(p))

    b = _StreamBuilder(layers=layers, experts=experts, top_k=top_k, source=f"synthetic:{spec}")
    calls = 0
    uid_pool = [10_000 + i for i in range(max(1, n_prompts))]
    step = 0
    for s in range(steps):
        if prefill_every and s and s % prefill_every == 0:
            uid = uid_pool[(s // prefill_every) % len(uid_pool)]
            for c in range(prefill_chunks):
                for lid in range(layers):
                    ids = sorted(rng.sample(range(experts), min(prefill_touch, experts)))
                    b.add(step, uid, lid, KIND_PREFILL, c, 1024, ids)
                step += 1
        uid = uid_pool[s % len(uid_pool)]
        for lid in range(layers):
            if use_pool:
                pool = pools[lid]
                if tau > 0 and calls % tau == 0 and calls:
                    j = rng.randrange(len(pool))
                    old = pool[j]
                    new = rng.randrange(experts)
                    tries = 0
                    while new in pool_sets[lid] and tries < 32:
                        new = rng.randrange(experts)
                        tries += 1
                    pool_sets[lid].discard(old)
                    pool[j] = new
                    pool_sets[lid].add(new)
                if use_zipf:
                    # zipf marginals restricted to the live pool
                    ids = set()
                    guard = 0
                    while len(ids) < top_k and guard < 200:
                        ids.add(pool[_alias_draw(rng, prob, alias) % len(pool)])
                        guard += 1
                    while len(ids) < top_k:
                        ids.add(pool[rng.randrange(len(pool))])
                    ids = sorted(ids)
                else:
                    ids = sorted(rng.sample(pool, min(top_k, len(pool))))
            elif use_zipf:
                perm = perms[lid]
                ids = set()
                guard = 0
                while len(ids) < top_k and guard < 500:
                    ids.add(perm[_alias_draw(rng, prob, alias)])
                    guard += 1
                while len(ids) < top_k:
                    ids.add(rng.randrange(experts))
                ids = sorted(ids)
            else:
                ids = sorted(rng.sample(range(experts), top_k))
            b.add(step, uid, lid, KIND_DECODE, 0, 1, ids)
        step += 1
        calls += 1
    return b.finish()


# ---------------------------------------------------------------------------------------------
# REFERENCE STREAM — the ordered (step, layer, kind, expert) list the cache actually sees.
# ---------------------------------------------------------------------------------------------

POLL_BLOCK = 64  # decode steps per pollution-curve block
POLL_BLOCKS = 8


class RefStream:
    """Flat, cache-friendly form of one or more traces concatenated.

    Record-level: layer/kind/step/uid/warm/pollution-block + an offset into `ids`.
    Reference-level: `ids` (expert id) and `keys` (layer*experts + expert).
    """

    __slots__ = (
        "layers",
        "experts",
        "top_k",
        "rec_layer",
        "rec_kind",
        "rec_step",
        "rec_uid",
        "rec_warm",
        "rec_poll",
        "off",
        "ids",
        "keys",
        "sources",
        "n_recs",
        "n_refs",
        "_ref_kind",
        "_ref_warm",
        "_ref_layer",
        "violations",
        "truncated",
        "_nu_cache",
    )

    def ref_arrays(self):
        """Per-reference kind/warm/layer, materialised once (numpy if available)."""
        if self._ref_kind is not None:
            return self._ref_kind, self._ref_warm, self._ref_layer
        n = self.n_refs
        rk = bytearray(n)
        rw = bytearray(n)
        rl = array("H", bytes(2 * n))
        off = self.off
        for r in range(self.n_recs):
            a, b = off[r], off[r + 1]
            k = self.rec_kind[r]
            w = self.rec_warm[r]
            l = self.rec_layer[r]
            for i in range(a, b):
                rk[i] = k
                rw[i] = w
                rl[i] = l
        if _np is not None:
            rk = _np.frombuffer(bytes(rk), dtype=_np.uint8)
            rw = _np.frombuffer(bytes(rw), dtype=_np.uint8).astype(bool)
            rl = _np.frombuffer(rl.tobytes(), dtype=_np.uint16)
        self._ref_kind, self._ref_warm, self._ref_layer = rk, rw, rl
        return rk, rw, rl


class _StreamBuilder:
    def __init__(self, *, layers: int, experts: int, top_k: int, source: str):
        self.layers = layers
        self.experts = experts
        self.top_k = top_k
        self.source = source
        self.rec_layer = array("H")
        self.rec_kind = array("B")
        self.rec_step = array("I")
        self.rec_uid = array("I")
        self.off = array("q", [0])
        self.ids = array("H")
        self.keys = array("i")
        self.violations: Dict[str, int] = defaultdict(int)
        self.truncated = False

    def add(self, step, uid, layer, kind, chunk, ntok, ids) -> None:
        self.rec_layer.append(layer)
        self.rec_kind.append(kind)
        self.rec_step.append(step & 0xFFFFFFFF)
        self.rec_uid.append(uid & 0xFFFFFFFF)
        base = layer * self.experts
        for e in ids:
            self.ids.append(e)
            self.keys.append(base + e)
        self.off.append(len(self.ids))

    def extend_raw(self, t: RawTrace, step_offset: int) -> int:
        experts = self.experts
        ids = t.ids
        for r in range(t.n_records):
            a, b = t.off[r], t.off[r + 1]
            self.rec_layer.append(t.layer[r])
            self.rec_kind.append(t.kind[r])
            self.rec_step.append((t.step[r] + step_offset) & 0xFFFFFFFF)
            self.rec_uid.append(t.uid[r])
            base = t.layer[r] * experts
            for i in range(a, b):
                e = ids[i]
                self.ids.append(e)
                self.keys.append(base + e)
            self.off.append(len(self.ids))
        for k, v in t.violations.items():
            self.violations[k] += v
        self.truncated = self.truncated or t.truncated
        return (t.step[-1] + step_offset + 1) if t.n_records else step_offset

    def finish(self, warmup_steps: int = 0) -> RefStream:
        s = RefStream()
        s.layers = self.layers
        s.experts = self.experts
        s.top_k = self.top_k
        s.rec_layer = self.rec_layer
        s.rec_kind = self.rec_kind
        s.rec_step = self.rec_step
        s.rec_uid = self.rec_uid
        s.off = self.off
        s.ids = self.ids
        s.keys = self.keys
        s.sources = [self.source]
        s.n_recs = len(self.rec_layer)
        s.n_refs = len(self.ids)
        s._ref_kind = s._ref_warm = s._ref_layer = None
        s._nu_cache = None
        s.violations = dict(self.violations)
        s.truncated = self.truncated
        mark_warm(s, warmup_steps)
        return s


def mark_warm(s: RefStream, warmup_steps: int) -> None:
    """Warm-up boundary in STEPS: the first `warmup_steps` distinct step_ids are simulated but
    excluded from the reported rates (cold-start compulsory misses are real misses, they just
    are not representative of steady state — both numbers are reported)."""
    warm = bytearray(s.n_recs)
    poll = array("b", bytes(s.n_recs))
    seen = 0
    last = None
    since_prefill = None
    for r in range(s.n_recs):
        st = s.rec_step[r]
        if st != last:
            last = st
            seen += 1
            if s.rec_kind[r] == KIND_DECODE and since_prefill is not None:
                since_prefill += 1
        if s.rec_kind[r] == KIND_PREFILL:
            since_prefill = 0
        warm[r] = 1 if seen > warmup_steps else 0
        if since_prefill is None or s.rec_kind[r] != KIND_DECODE:
            poll[r] = -1
        else:
            blk = (since_prefill - 1) // POLL_BLOCK
            poll[r] = blk if 0 <= blk < POLL_BLOCKS else -1
    s.rec_warm = warm
    s.rec_poll = poll


def load_streams(paths: Sequence[str], warmup_steps: int, separate: bool, strict: bool):
    """One RefStream per trace (`separate`) or one concatenated stream."""
    out = []
    if separate:
        for p in paths:
            t = read_trace(p, strict=strict)
            b = _StreamBuilder(
                layers=t.header.num_layers,
                experts=t.header.num_experts,
                top_k=t.header.top_k,
                source=os.path.basename(p),
            )
            b.extend_raw(t, 0)
            out.append(b.finish(warmup_steps))
        return out
    first = read_trace(paths[0], strict=strict)
    b = _StreamBuilder(
        layers=first.header.num_layers,
        experts=first.header.num_experts,
        top_k=first.header.top_k,
        source=",".join(os.path.basename(p) for p in paths),
    )
    off = b.extend_raw(first, 0)
    for p in paths[1:]:
        t = read_trace(p, strict=strict)
        if (t.header.num_layers, t.header.num_experts, t.header.top_k) != (
            first.header.num_layers,
            first.header.num_experts,
            first.header.top_k,
        ):
            raise TraceReadError(f"{p}: geometry differs from {paths[0]}")
        if t.header.model_hash != first.header.model_hash:
            raise TraceReadError(f"{p}: model_hash differs — traces from two checkpoints are "
                                 "not comparable")
        off = b.extend_raw(t, off)
    return [b.finish(warmup_steps)]


# ---------------------------------------------------------------------------------------------
# POLICIES
# ---------------------------------------------------------------------------------------------

PREFILL_INSERT = "insert"
PREFILL_NO_INSERT = "no-insert"
PREFILL_MRU_EVICT = "mru-evict"

_INF = 1 << 62
_EMPTY_PIN: frozenset = frozenset()


class _LRUPool:
    """Classic LRU. Insert on every access, evict least-recently-used.

    `pinned` is the in-flight call's working set. For LRU it is a no-op except under
    mru-evict (where the just-inserted key would otherwise be the victim), but it is applied
    uniformly so every dynamic policy sees the same constraint.
    """

    __slots__ = ("cap", "od")

    def __init__(self, cap: int):
        self.cap = cap
        self.od: "OrderedDict[int, None]" = OrderedDict()

    def access(self, k: int, insert: bool, mru_evict: bool, pinned, touch: bool = True) -> bool:
        od = self.od
        if k in od:
            if touch:
                od.move_to_end(k)
            return True
        if not insert or self.cap <= 0:
            return False
        if len(od) >= self.cap:
            victim = None
            it = reversed(od) if mru_evict else iter(od)
            for cand in it:
                if cand not in pinned:
                    victim = cand
                    break
            if victim is None:
                return False  # capacity < working set: nothing evictable, treat as bypass
            del od[victim]
        od[k] = None
        return False


_LAG_STEPS = 0


class _LaggedPool:
    """A pool whose MANAGER observes references L steps late — the architecture question.

    WHY THIS EXISTS. The cache is only free if the manager never forces a host sync. Reading
    `topk_ids` synchronously to decide residency costs a D2H per MoE layer per step (~0.13 ms on
    card 1 x 48 layers = ~6 ms/step, which eats a third of the win). The alternative is to observe
    routes ASYNCHRONOUSLY — drain the device-side ring the route tracer already builds — and accept
    that the manager's picture of "what is hot" is L steps stale.

    A HIT IS EVALUATED AGAINST WHAT IS RESIDENT NOW; only the INSERT is delayed. That is exactly
    the physical situation: the kernel dereferences whatever `slot_of` currently says, and the
    manager's update lands later. Evictions are likewise driven by the delayed stream, because the
    manager cannot evict on a reference it has not seen.

    If h collapses as L grows, the async design is dead and the cache needs a synchronous route —
    which changes the whole build. If h is flat to L ~= 64, the manager can be a lazy background
    drain and the hot path stays untouched.
    """

    __slots__ = ("inner", "lag", "q", "cap")

    def __init__(self, inner, lag: int):
        self.inner = inner
        self.lag = lag
        self.cap = inner.cap          # simulate() reads .cap to size the in-flight pin set
        self.q: "deque" = deque()

    def access(self, k: int, insert: bool, mru_evict: bool, pinned, touch: bool = True) -> bool:
        hit = k in self.inner.od if hasattr(self.inner, "od") else self.inner.access(
            k, False, mru_evict, pinned, touch=False)
        # Retire the delayed observations that have now become visible to the manager.
        self.q.append((k, insert))
        while len(self.q) > self.lag:
            dk, dins = self.q.popleft()
            self.inner.access(dk, dins, mru_evict, pinned)
        return hit


class _LFUPool:
    """Saturating-u8 LFU with LRU scan order as tiebreak — the shape of SGLang expert_pack.py's
    "reuse-lfu-lru-v2" `_victim_slot` (:464-486). The current call's working set is PINNED:
    upstream cannot evict a slab the in-flight grouped GEMM is about to read, and that
    constraint changes the answer at small C, so it is modelled rather than assumed away.
    """

    __slots__ = ("cap", "cnt", "buckets", "minc")

    def __init__(self, cap: int):
        self.cap = cap
        self.cnt: Dict[int, int] = {}
        self.buckets: Dict[int, "OrderedDict[int, None]"] = defaultdict(OrderedDict)
        self.minc = 1

    def access(self, k: int, insert: bool, mru_evict: bool, pinned, touch: bool = True) -> bool:
        cnt = self.cnt
        c = cnt.get(k)
        if c is not None:
            if touch:
                del self.buckets[c][k]
                nc = c if c >= 255 else c + 1
                cnt[k] = nc
                self.buckets[nc][k] = None
                if c == self.minc and not self.buckets[c]:
                    self.minc = nc if nc > c else c
            return True
        if not insert or self.cap <= 0:
            return False
        if len(cnt) >= self.cap:
            victim = None
            c = self.minc
            while victim is None and c <= 255:
                bk = self.buckets.get(c)
                if bk:
                    it = reversed(bk) if mru_evict else iter(bk)
                    for cand in it:
                        if cand not in pinned:
                            victim = cand
                            break
                c += 1
            if victim is None:
                return False
            vc = cnt.pop(victim)
            del self.buckets[vc][victim]
        cnt[k] = 1
        self.buckets[1][k] = None
        self.minc = 1
        return False


class _SLRUPool:
    """Segmented LRU: a probationary segment plus a protected segment (80% of capacity). A key
    is promoted to protected on its SECOND access, so a prefill chunk's one-shot touches churn
    only the probationary segment and cannot evict the decode hot set. Named by G2's re-gate
    list; it is the cheapest scan-resistant policy that a slot-indirection table can implement.
    """

    __slots__ = ("cap", "cap_prot", "prob", "prot")

    def __init__(self, cap: int, protected_frac: float = 0.8):
        self.cap = cap
        self.cap_prot = int(cap * protected_frac)
        self.prob: "OrderedDict[int, None]" = OrderedDict()
        self.prot: "OrderedDict[int, None]" = OrderedDict()

    def access(self, k: int, insert: bool, mru_evict: bool, pinned, touch: bool = True) -> bool:
        prob, prot = self.prob, self.prot
        if k in prot:
            if touch:
                prot.move_to_end(k)
            return True
        if k in prob:
            if touch:
                del prob[k]
                prot[k] = None
                while len(prot) > self.cap_prot:
                    dk, _ = prot.popitem(last=False)  # demote the protected LRU
                    prob[dk] = None
            return True
        if not insert or self.cap <= 0:
            return False
        while len(prob) + len(prot) >= self.cap and prob:
            victim = None
            for cand in (reversed(prob) if mru_evict else iter(prob)):
                if cand not in pinned:
                    victim = cand
                    break
            if victim is None:
                return False
            del prob[victim]
        if len(prob) + len(prot) >= self.cap:
            return False
        prob[k] = None
        return False


class _HybridPool:
    """`static-prior-pinned + LRU over the remainder` — the last arm on G2's re-gate list.

    Half the slots hold the top-frequency keys learned on the FIRST HALF of the trace (never on
    the evaluated half); the rest run LRU. It exists to answer "is the win skew, locality, or
    both?" with one number: if the hybrid beats both pure arms, the two structures are additive
    and a real implementation wants both a pinned prior and an eviction path."""

    __slots__ = ("cap", "pinned_keys", "resident", "lru")

    def __init__(self, cap: int, pinned_keys: set):
        self.cap = cap
        self.pinned_keys = pinned_keys
        self.resident = set()
        self.lru = _LRUPool(max(0, cap - len(pinned_keys)))

    def access(self, k: int, insert: bool, mru_evict: bool, pinned, touch: bool = True) -> bool:
        if k in self.pinned_keys:
            if k in self.resident:
                return True
            self.resident.add(k)  # compulsory miss, then pinned forever
            return False
        return self.lru.access(k, insert, mru_evict, pinned, touch)


class _StaticPool:
    """A fixed pinned key set. Entries still take their COMPULSORY miss on first touch, which is
    what keeps oracle-static <= belady honest (a static set that hit from t=0 could otherwise
    beat the offline optimum by |C| references)."""

    __slots__ = ("pinned_keys", "resident")

    def __init__(self, pinned_keys: set):
        self.pinned_keys = pinned_keys
        self.resident = set()

    def access(self, k: int, insert: bool, mru_evict: bool, pinned, touch: bool = True) -> bool:
        if k in self.resident:
            return True
        if k in self.pinned_keys:
            self.resident.add(k)
        return False


def _next_use_array(keys, n: int, decode_mask: Optional[bytearray] = None) -> array:
    """next_use[i] = index of the next reference to the same key after i, else INF.

    With `decode_mask`, only DECODE references count as a "use". That is deliberate and it is
    the difference between a ceiling that means something and one that does not:

      * The kill gates score DECODE misses (the cost model is a decode step). Plain Belady
        minimises TOTAL misses, so on a mixed stream it will happily sacrifice decode hits to
        serve a prefill chunk that touches ~400 of 512 experts. That is optimal for the wrong
        objective, and it is exactly how `oracle-static` came out ABOVE `belady` on h_decode in
        the first run of this file — the sim was not broken, the objective was.
      * Scored on next-DECODE-use, a prefill reference is a FREE PREFETCH in the h_decode
        metric (the bytes are charged to prefill, not to the decode step), which is precisely
        the asymmetry a real prefill-aware policy exploits.

    So: on a decode-only stream this is exactly MIN and provably optimal. On a mixed stream it
    is the strongest ceiling this file can construct in O(N log C) — minimising decode misses
    with prefill loads allowed is a WEIGHTED caching problem, which MIN does not solve exactly.
    T4 therefore checks the dominance empirically instead of assuming it, and any row that
    exceeds belady is reported rather than clamped.
    """
    nu = array("q", bytes(8 * n))
    last: Dict[int, int] = {}
    if decode_mask is None:
        for i in range(n - 1, -1, -1):
            k = keys[i]
            nxt = last.get(k)
            nu[i] = nxt if nxt is not None else _INF
            last[k] = i
    else:
        for i in range(n - 1, -1, -1):
            k = keys[i]
            nxt = last.get(k)
            nu[i] = nxt if nxt is not None else _INF
            if decode_mask[i]:
                last[k] = i
    return nu


class _BeladyPool:
    """Offline optimum (Belady/MIN) with bypass: on a full miss, evict the resident key whose
    NEXT use is furthest, or decline to insert if the incoming key's own next use is furthest.
    Bypassing is optimal and never worse than demand-paging MIN, so this is a true ceiling for
    any online policy at the same capacity. Lazily-revalidated heap => O(N log C).

    "Next use" is the next DECODE use (see `_next_use_array`), because the gates score decode
    misses. On a decode-only stream that makes this exactly MIN; on a mixed stream it is the
    strongest constructible ceiling and T4 verifies the dominance rather than assuming it.

    NOT pinned: within a call each key is referenced exactly once, so evicting an
    already-served key is legal in the reference-stream model and preserves optimality.
    """

    __slots__ = ("cap", "resident", "heap")

    def __init__(self, cap: int):
        self.cap = cap
        self.resident: Dict[int, int] = {}
        self.heap: List[Tuple[int, int]] = []

    def access(self, k: int, nu: int, insert: bool = True) -> bool:
        res = self.resident
        if k in res:
            res[k] = nu
            heapq.heappush(self.heap, (-nu, k))
            return True
        if self.cap <= 0 or not insert:
            return False
        if len(res) < self.cap:
            res[k] = nu
            heapq.heappush(self.heap, (-nu, k))
            return False
        heap = self.heap
        while heap:
            negnu, cand = heap[0]
            if res.get(cand) == -negnu:
                break
            heapq.heappop(heap)  # stale entry: this key was re-referenced or evicted
        if not heap:  # pragma: no cover - cap>0 guarantees a live resident entry
            return False
        negnu, cand = heap[0]
        if -negnu > nu:  # the furthest-future resident key is further away than the new one
            heapq.heappop(heap)
            del res[cand]
            res[k] = nu
            heapq.heappush(heap, (-nu, k))
        # else: BYPASS — the incoming key's own next use is the furthest, so admitting it
        # could only be worse. Bypassing MIN is optimal and >= demand-paging MIN, which is
        # what makes this row a true ceiling for every online policy.
        return False


# ---------------------------------------------------------------------------------------------
# SIMULATION
# ---------------------------------------------------------------------------------------------

POLICIES = (
    "static-layer",   # (a) the shipped arm — the baseline every other row is measured against
    "static-prior",   # (b) top-N learned on the FIRST half, evaluated on the SECOND half
    "lru",            # (c)
    "lfu",            # (d) saturating-u8 + LRU tiebreak, working set pinned (SGLang's shape)
    "slru",           # G2 re-gate: segmented LRU (scan-resistant)
    "prior+lru",      # G2 re-gate: pinned prior over half the slots + LRU over the remainder
    "belady",         # (f) offline optimum — THE DECIDING ROW
    "oracle-static",  # (g) top-N over the whole trace; must be <= belady or the sim is broken
)

# Policies whose hit rate is only defined on the second half of the trace (they learn on the
# first). Comparing them against a whole-stream policy without matching the window is the
# apples-to-oranges error that makes a "static wins" or "static loses" claim meaningless.
HALF_SPLIT_POLICIES = ("static-prior", "prior+lru")


class PolicyRun:
    __slots__ = ("policy", "flags", "eval_start", "note", "slots", "slots_per_layer", "seconds")

    def __init__(self, policy, flags, eval_start, note, slots, spl, seconds):
        self.policy = policy
        self.flags = flags
        self.eval_start = eval_start
        self.note = note
        self.slots = slots
        self.slots_per_layer = spl
        self.seconds = seconds


def _key_counts(s: RefStream, lo: int, hi: int, decode_only: bool) -> Dict[int, int]:
    """Access counts per (layer,expert) key over a reference window.

    `decode_only` is TRUE for every static prior on purpose: the prior exists to cut DECODE
    misses, and a prefill chunk touches ~400 of 512 experts almost uniformly, so folding
    prefill counts in would dilute the ranking with near-uniform noise and understate how much
    static skew is actually exploitable. Learning the strongest honest prior is the
    conservative choice here — it makes the static baseline HARDER to beat, which is the right
    direction for a question of the form "is a dynamic cache worth building?"."""
    c: Dict[int, int] = defaultdict(int)
    off, keys = s.off, s.keys
    for r in range(s.n_recs):
        a, b = off[r], off[r + 1]
        if a >= hi or b <= lo:
            continue
        if decode_only and s.rec_kind[r] != KIND_DECODE:
            continue
        for i in range(max(a, lo), min(b, hi)):
            c[keys[i]] += 1
    return c


def _topn_keys(counts: Dict[int, int], n: int) -> set:
    if n <= 0:
        return set()
    return set(k for k, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:n])


def _topn_per_layer(counts: Dict[int, int], n: int, layers: int, experts: int) -> set:
    per: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for k, v in counts.items():
        per[k // experts].append((k, v))
    out = set()
    for lid in range(layers):
        lst = sorted(per.get(lid, ()), key=lambda kv: (-kv[1], kv[0]))[:n]
        out.update(k for k, _ in lst)
    return out


def simulate(
    s: RefStream,
    policy: str,
    slots: int,
    *,
    pool_mode: str = "global",
    prefill_policy: str = PREFILL_INSERT,
) -> PolicyRun:
    t0 = time.time()
    n = s.n_refs
    flags = bytearray(n)
    layers, experts = s.layers, s.experts
    spl = slots // layers
    eval_start = 0
    note = ""
    per_layer = pool_mode == "per-layer"
    cap = spl if per_layer else slots
    keys, off = s.keys, s.off

    if policy == "static-layer":
        # The SHIPPED arm: pin whole layers until the byte budget is exhausted. h is exactly
        # n_pinned/layers by construction — no dynamics at all. Every other row is measured
        # against this.
        n_pinned = (slots * EXPERT_BYTES) // (experts * EXPERT_BYTES)
        n_pinned = min(n_pinned, layers)
        pinned_keys = set(range(n_pinned * experts))
        pools = [_StaticPool(pinned_keys)]
        note = f"n_pinned_layers={n_pinned}"
    elif policy in ("static-prior", "oracle-static"):
        if policy == "static-prior":
            mid_rec = s.n_recs // 2
            mid = off[mid_rec]
            counts = _key_counts(s, 0, mid, decode_only=True)
            eval_start = mid
            note = "learned on first half (decode refs), evaluated on second half"
        else:
            counts = _key_counts(s, 0, n, decode_only=True)
            note = "top-N over the WHOLE trace (cheating upper bound for static)"
        pinned_keys = (
            _topn_per_layer(counts, spl, layers, experts)
            if per_layer
            else _topn_keys(counts, slots)
        )
        pools = [_StaticPool(pinned_keys)]
    elif policy == "prior+lru":
        mid_rec = s.n_recs // 2
        mid = off[mid_rec]
        counts = _key_counts(s, 0, mid, decode_only=True)
        eval_start = mid
        n_pin = cap // 2
        pinned_keys = (
            _topn_per_layer(counts, n_pin, layers, experts)
            if per_layer
            else _topn_keys(counts, n_pin)
        )
        note = f"{n_pin} pinned (1st-half prior) + {cap - n_pin} LRU slots, evaluated on 2nd half"
        if per_layer:
            by_layer = [set() for _ in range(layers)]
            for k in pinned_keys:
                by_layer[k // experts].add(k)
            pools = [_HybridPool(cap, by_layer[l]) for l in range(layers)]
        else:
            pools = [_HybridPool(cap, pinned_keys)]
    elif policy == "lru":
        pools = [_LRUPool(cap) for _ in range(layers if per_layer else 1)]
        if _LAG_STEPS:
            pools = [_LaggedPool(x, _LAG_STEPS) for x in pools]
    elif policy == "slru":
        pools = [_SLRUPool(cap) for _ in range(layers if per_layer else 1)]
        if _LAG_STEPS:
            pools = [_LaggedPool(x, _LAG_STEPS) for x in pools]
    elif policy == "lfu":
        pools = [_LFUPool(cap) for _ in range(layers if per_layer else 1)]
    elif policy == "belady":
        pools = [_BeladyPool(cap) for _ in range(layers if per_layer else 1)]
    else:
        raise ValueError(f"unknown policy {policy!r}")

    is_static = policy in ("static-layer", "static-prior", "oracle-static")

    if policy == "belady":
        dmask = bytearray(n)
        n_dec = 0
        for r in range(s.n_recs):
            if s.rec_kind[r] == KIND_DECODE:
                a, b = off[r], off[r + 1]
                dmask[a:b] = b"\x01" * (b - a)
                n_dec += b - a
        nu = s._nu_cache
        if nu is None:
            nu = _next_use_array(keys, n, dmask if n_dec else None)
            s._nu_cache = nu  # identical for every budget; O(8N) and worth keeping
        no_ins = prefill_policy == PREFILL_NO_INSERT
        for r in range(s.n_recs):
            pool = pools[s.rec_layer[r]] if per_layer else pools[0]
            ins = not (no_ins and s.rec_kind[r] == KIND_PREFILL)
            acc = pool.access
            for i in range(off[r], off[r + 1]):
                flags[i] = 1 if acc(keys[i], nu[i], ins) else 0
        del dmask
    elif is_static:
        pool = pools[0]
        acc = pool.access
        empty: set = set()
        for i in range(n):
            flags[i] = 1 if acc(keys[i], True, False, empty) else 0
    else:
        for r in range(s.n_recs):
            a, b = off[r], off[r + 1]
            kind = s.rec_kind[r]
            insert = True
            mru = False
            touch = True
            if kind == KIND_PREFILL:
                if prefill_policy == PREFILL_NO_INSERT:
                    # the bytes are read (the miss is counted) but the cache is untouched:
                    # no insert, no evict, and no recency refresh either
                    insert = False
                    touch = False
                elif prefill_policy == PREFILL_MRU_EVICT:
                    # SGLang's preserve_oldest=is_prefill: admit, but take the NEWEST victim
                    # and leave the decode recency order alone. Refreshing recency on a prefill
                    # HIT is what silently destroys the decode hot set — the promoted decode
                    # key becomes the MRU and the very next prefill miss evicts it, which is
                    # why a naive mru-evict measures identical to plain insert.
                    mru = True
                    touch = False
            pool = pools[s.rec_layer[r]] if per_layer else pools[0]
            # Pin this call's working set — but only at DECODE. At decode the grouped GEMM
            # holds all top_k=10 slabs live at once, so none of them may be chosen as a victim
            # (this is a real constraint of the shipped upstream design and it changes the
            # answer at small C). A prefill chunk touching ~400 of 512 experts is STREAMED, not
            # held resident: pinning it would (a) misstate the hardware and (b) make
            # `mru-evict` a no-op, because a chunk's own freshly inserted slabs — exactly the
            # entries SGLang's preserve_oldest wants to evict first — would be immune. Pinning
            # is also skipped when the pool is smaller than the working set (degenerate tiny
            # budgets), where the victim search would otherwise have no legal candidate.
            pinned = (set(keys[a:b])
                      if (kind == KIND_DECODE and pool.cap >= (b - a)) else _EMPTY_PIN)
            acc = pool.access
            for i in range(a, b):
                flags[i] = 1 if acc(keys[i], insert, mru, pinned, touch) else 0
    return PolicyRun(policy, flags, eval_start, note, slots, spl, time.time() - t0)


# ---------------------------------------------------------------------------------------------
# STATS
# ---------------------------------------------------------------------------------------------


def rate(
    s: RefStream,
    run: PolicyRun,
    *,
    kind: Optional[int] = KIND_DECODE,
    warm_only: bool = True,
    lo: Optional[int] = None,
    hi: Optional[int] = None,
) -> Tuple[float, int]:
    """Hit rate over a window of the reference stream. Returns (rate, n_refs)."""
    lo = run.eval_start if lo is None else max(lo, run.eval_start)
    hi = s.n_refs if hi is None else hi
    rk, rw, _ = s.ref_arrays()
    if _np is not None:
        f = _np.frombuffer(bytes(run.flags), dtype=_np.uint8)[lo:hi]
        m = _np.ones(hi - lo, dtype=bool)
        if kind is not None:
            m &= rk[lo:hi] == kind
        if warm_only:
            m &= rw[lo:hi]
        cnt = int(m.sum())
        return (float(f[m].sum()) / cnt if cnt else float("nan")), cnt
    hits = cnt = 0
    fl = run.flags
    for i in range(lo, hi):
        if kind is not None and rk[i] != kind:
            continue
        if warm_only and not rw[i]:
            continue
        cnt += 1
        hits += fl[i]
    return (hits / cnt if cnt else float("nan")), cnt


# ---------------------------------------------------------------------------------------------
# DIAGNOSTICS — this block is what actually EXPLAINS the answer, so it is never optional.
# ---------------------------------------------------------------------------------------------


def _gini(counts: Sequence[int]) -> float:
    x = sorted(counts)
    n = len(x)
    tot = sum(x)
    if n == 0 or tot == 0:
        return 0.0
    cum = 0
    for i, v in enumerate(x, 1):
        cum += i * v
    return (2.0 * cum) / (n * tot) - (n + 1.0) / n


class _Fenwick:
    __slots__ = ("n", "t")

    def __init__(self, n: int):
        self.n = n
        self.t = array("i", bytes(4 * (n + 1)))

    def add(self, i: int, v: int) -> None:
        t, n = self.t, self.n
        i += 1
        while i <= n:
            t[i] += v
            i += i & (-i)

    def total(self, i: int) -> int:
        t = self.t
        i += 1
        s = 0
        while i > 0:
            s += t[i]
            i -= i & (-i)
        return s


def stack_distances(key_seq: Sequence[int], limit: int) -> List[int]:
    """Exact LRU stack (reuse) distance: the number of DISTINCT keys touched since the previous
    reference to this key. `d <= C` is exactly the condition for an LRU hit at capacity C, so
    the fraction of references with d <= slots is the mechanism behind every LRU number here.
    First references have distance INF (compulsory) and are excluded from the quantiles but
    counted in the <= C fraction denominator via `n_total`.
    """
    n = min(len(key_seq), limit)
    fw = _Fenwick(n + 1)
    last: Dict[int, int] = {}
    out: List[int] = []
    for i in range(n):
        k = key_seq[i]
        p = last.get(k)
        if p is not None:
            d = fw.total(i - 1) - fw.total(p)
            out.append(d)
            fw.add(p, -1)
        last[k] = i
        fw.add(i, 1)
    return out


def diagnostics(s: RefStream, budgets_slots: Sequence[Tuple[float, int]], stack_limit: int) -> dict:
    d: dict = {}
    layers, experts = s.layers, s.experts
    off, keys = s.off, s.keys

    # --- per-layer expert access counts (decode only, post-warm-up) --------------------------
    per_layer_counts = [defaultdict(int) for _ in range(layers)]
    pooled: Dict[int, int] = defaultdict(int)
    decode_refs = 0
    decode_steps = set()
    uids = set()
    uid_steps: Dict[int, set] = defaultdict(set)
    prefill_recs = 0
    for r in range(s.n_recs):
        k = s.rec_kind[r]
        if k == KIND_PREFILL:
            prefill_recs += 1
        if k != KIND_DECODE or not s.rec_warm[r]:
            continue
        lid = s.rec_layer[r]
        decode_steps.add(s.rec_step[r])
        u = s.rec_uid[r]
        if u:
            uids.add(u)
            uid_steps[u].add(s.rec_step[r])
        c = per_layer_counts[lid]
        for i in range(off[r], off[r + 1]):
            e = keys[i] - lid * experts
            c[e] += 1
            pooled[keys[i]] += 1
            decode_refs += 1

    ginis = [_gini([per_layer_counts[l].get(e, 0) for e in range(experts)]) for l in range(layers)]
    d["gini"] = {
        "mean": sum(ginis) / len(ginis) if ginis else 0.0,
        "min": min(ginis) if ginis else 0.0,
        "max": max(ginis) if ginis else 0.0,
        "per_layer": ginis,
    }

    # --- top-C mass ---------------------------------------------------------------------------
    ordered = sorted(pooled.values(), reverse=True)
    tot = sum(ordered)
    cum = []
    run = 0
    for v in ordered:
        run += v
        cum.append(run)
    topc = {}
    for gib, sl in budgets_slots:
        if sl <= 0 or tot == 0:
            topc[f"{gib:g}"] = 0.0
        else:
            idx = min(sl, len(cum)) - 1
            topc[f"{gib:g}"] = cum[idx] / tot if idx >= 0 else 0.0
    d["top_c_mass"] = topc
    d["distinct_keys_touched"] = len(pooled)

    # --- stack distance -----------------------------------------------------------------------
    dec_keys = array("i")
    dec_layer_keys = [array("i") for _ in range(layers)]
    for r in range(s.n_recs):
        if s.rec_kind[r] != KIND_DECODE or not s.rec_warm[r]:
            continue
        lid = s.rec_layer[r]
        for i in range(off[r], off[r + 1]):
            dec_keys.append(keys[i])
            dec_layer_keys[lid].append(keys[i])
    pooled_sd = stack_distances(dec_keys, stack_limit)
    real_slots = None
    for gib, sl in budgets_slots:
        if abs(gib - REAL_BUDGET_GIB) < 1e-9:
            real_slots = sl
    real_slots = real_slots if real_slots is not None else slots_for(REAL_BUDGET_GIB)
    spl = real_slots // layers

    def _q(v: Sequence[int], p: float) -> float:
        if not v:
            return float("nan")
        x = sorted(v)
        return float(x[min(len(x) - 1, int(p * len(x)))])

    n_sd_total = min(len(dec_keys), stack_limit)
    d["stack_distance_pooled"] = {
        "p50": _q(pooled_sd, 0.50),
        "p90": _q(pooled_sd, 0.90),
        "n_reuse": len(pooled_sd),
        "n_refs": n_sd_total,
        "frac_le_slots_global": (
            sum(1 for x in pooled_sd if x <= real_slots) / n_sd_total if n_sd_total else 0.0
        ),
    }
    per_layer_sd = []
    lim = max(1, stack_limit // max(1, layers))
    for lid in range(layers):
        sd = stack_distances(dec_layer_keys[lid], lim)
        ntot = min(len(dec_layer_keys[lid]), lim)
        per_layer_sd.append(
            {
                "layer": lid,
                "p50": _q(sd, 0.50),
                "p90": _q(sd, 0.90),
                "frac_le_slots_per_layer": (
                    sum(1 for x in sd if x <= spl) / ntot if ntot else 0.0
                ),
            }
        )
    fr = [x["frac_le_slots_per_layer"] for x in per_layer_sd]
    d["stack_distance_per_layer"] = {
        "slots_per_layer": spl,
        "frac_le_slots_mean": sum(fr) / len(fr) if fr else 0.0,
        "frac_le_slots_min": min(fr) if fr else 0.0,
        "frac_le_slots_max": max(fr) if fr else 0.0,
        "p50_mean": sum(x["p50"] for x in per_layer_sd) / max(1, len(per_layer_sd)),
        "p90_mean": sum(x["p90"] for x in per_layer_sd) / max(1, len(per_layer_sd)),
        "per_layer": per_layer_sd,
    }

    # --- static-prior half-split coverage (the llama.cpp falsifier, replicated) ---------------
    mid_rec = s.n_recs // 2
    first = [defaultdict(int) for _ in range(layers)]
    second = [defaultdict(int) for _ in range(layers)]
    for r in range(s.n_recs):
        if s.rec_kind[r] != KIND_DECODE:
            continue
        tgt = first if r < mid_rec else second
        c = tgt[s.rec_layer[r]]
        base = s.rec_layer[r] * experts
        for i in range(off[r], off[r + 1]):
            c[keys[i] - base] += 1
    half = {}
    for N in (32, 64, 108, 128):
        cov_num = cov_den = 0
        for lid in range(layers):
            top = set(
                k for k, _ in sorted(first[lid].items(), key=lambda kv: (-kv[1], kv[0]))[:N]
            )
            for e, v in second[lid].items():
                cov_den += v
                if e in top:
                    cov_num += v
        half[str(N)] = {
            "covered": cov_num / cov_den if cov_den else float("nan"),
            "uniform_expectation": N / experts,
        }
    d["static_prior_half_split"] = half

    # --- adjacent decode-step overlap ---------------------------------------------------------
    prev: Dict[int, set] = {}
    ov_sum = [0.0] * layers
    ov_n = [0] * layers
    for r in range(s.n_recs):
        if s.rec_kind[r] != KIND_DECODE:
            continue
        lid = s.rec_layer[r]
        cur = set(keys[off[r] : off[r + 1]])
        p = prev.get(lid)
        if p is not None and cur:
            ov_sum[lid] += len(cur & p) / len(cur)
            ov_n[lid] += 1
        prev[lid] = cur
    ov = [ov_sum[l] / ov_n[l] for l in range(layers) if ov_n[l]]
    d["adjacent_step_overlap"] = {
        "mean": sum(ov) / len(ov) if ov else float("nan"),
        "min": min(ov) if ov else float("nan"),
        "max": max(ov) if ov else float("nan"),
        "note": "llama.cpp reports ~0.442 on this model family; a wildly different number "
        "means the trace or the tap is wrong",
    }

    # --- prompt census ------------------------------------------------------------------------
    d["census"] = {
        "distinct_req_uids": len(uids),
        "decode_steps": len(decode_steps),
        "decode_refs_warm": decode_refs,
        "prefill_records": prefill_recs,
        "records": s.n_recs,
        "refs": s.n_refs,
        "decode_steps_per_uid_min": min((len(v) for v in uid_steps.values()), default=0),
        "decode_steps_per_uid_max": max((len(v) for v in uid_steps.values()), default=0),
        "violations": s.violations,
        "truncated": s.truncated,
    }
    return d


def decode_only_stream(s: RefStream, warmup_steps: int) -> RefStream:
    """The OPTIMISTIC stream: prefill records dropped entirely. G4 compares it against the
    honest mixed stream — if the gap is large, prefill handling is a first-class build
    requirement, not a later optimisation."""
    b = _StreamBuilder(layers=s.layers, experts=s.experts, top_k=s.top_k,
                       source=s.sources[0] + " [decode-only]")
    for r in range(s.n_recs):
        if s.rec_kind[r] != KIND_DECODE:
            continue
        base = s.rec_layer[r] * s.experts
        ids = [s.keys[j] - base for j in range(s.off[r], s.off[r + 1])]
        b.add(s.rec_step[r], s.rec_uid[r], s.rec_layer[r], KIND_DECODE, 0, 1, ids)
    return b.finish(warmup_steps)


def g4_analysis(s: RefStream, budget: float, pool_mode: str, warmup_steps: int,
                policies: Sequence[str] = ("lru", "lfu", "slru")) -> dict:
    """G4 — price prefill pollution instead of assuming it away.

    Reports (i) h on the honest mixed stream vs the optimistic decode-only stream, and (ii) all
    three --prefill-policy arms on the mixed stream, so the "retake G3 against the best prefill
    arm" branch is answerable from ONE run rather than from a promise to check later."""
    sl = slots_for(budget)
    out: dict = {"budget_gib": budget, "arms": {}, "decode_only": {}, "delta": {}}
    if not any(s.rec_kind[r] == KIND_PREFILL for r in range(s.n_recs)):
        out["note"] = "no prefill records in this stream — G4 is vacuous here"
        return out
    dos = decode_only_stream(s, warmup_steps)
    for p in policies:
        for pf in (PREFILL_INSERT, PREFILL_NO_INSERT, PREFILL_MRU_EVICT):
            h, _ = _h_of(s, p, sl, pool_mode=pool_mode, prefill_policy=pf)
            out["arms"][f"{p}/{pf}"] = h
        hdo, _ = _h_of(dos, p, sl, pool_mode=pool_mode)
        out["decode_only"][p] = hdo
        out["delta"][p] = hdo - out["arms"][f"{p}/{PREFILL_INSERT}"]
    best_arm = max(out["arms"].items(), key=lambda kv: kv[1])
    worst_delta = max(out["delta"].values())
    out["best_arm"] = {"name": best_arm[0], "h": best_arm[1]}
    out["max_delta"] = worst_delta
    if worst_delta > 0.15:
        out["verdict"] = (
            "RETAKE G3 — h(decode-only) - h(mixed) = {:.4f} > 0.15. Prefill pollution is "
            "material: the G3 decision must be taken against the best prefill arm "
            "({} at h={:.4f}), and any build that follows MUST carry that prefill handling as "
            "a first-class requirement.".format(worst_delta, best_arm[0], best_arm[1]))
    else:
        out["verdict"] = (
            "OK — h(decode-only) - h(mixed) = {:.4f} <= 0.15; prefill pollution does not move "
            "the decision, though the best arm is still {} at h={:.4f}.".format(
                worst_delta, best_arm[0], best_arm[1]))
    return out


def pollution_curve(s: RefStream, run: PolicyRun) -> List[dict]:
    """Decode hit rate conditioned on how recently a prefill chunk ran. A prefill chunk touches
    ~400 of 512 experts per layer; under naive insert-LRU it wipes the decode hot set, and the
    recovery shape is what says whether prefill handling is a first-class requirement (G4)."""
    hits = [0] * POLL_BLOCKS
    cnt = [0] * POLL_BLOCKS
    off = s.off
    fl = run.flags
    for r in range(s.n_recs):
        if s.rec_kind[r] != KIND_DECODE or not s.rec_warm[r]:
            continue
        blk = s.rec_poll[r]
        if blk < 0:
            continue
        a, b = off[r], off[r + 1]
        if b <= run.eval_start:
            continue
        for i in range(max(a, run.eval_start), b):
            cnt[blk] += 1
            hits[blk] += fl[i]
    return [
        {"block": i, "decode_steps_after_prefill": f"{i*POLL_BLOCK+1}-{(i+1)*POLL_BLOCK}",
         "h": (hits[i] / cnt[i] if cnt[i] else float("nan")), "n": cnt[i]}
        for i in range(POLL_BLOCKS)
    ]


# ---------------------------------------------------------------------------------------------
# REPORT
# ---------------------------------------------------------------------------------------------

_COLS = [
    ("policy", 16, "s"),
    ("cache_GiB/rank", 14, "g"),
    ("slots", 7, "d"),
    ("coverage", 9, ".4f"),
    ("h_decode", 9, ".4f"),
    ("h_all", 8, ".4f"),
    ("miss/tok", 9, ".1f"),
    ("MB/tok/rank", 12, ".1f"),
    ("ms_step", 8, ".2f"),
    ("tok/s", 7, ".2f"),
    ("speedup", 8, ".3f"),
    ("tok/s_der", 10, ".2f"),
]


def _fmt_row(vals) -> str:
    out = []
    for (name, w, f), v in zip(_COLS, vals):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            out.append("nan".rjust(w))
        elif f == "s":
            out.append(str(v).ljust(w))
        elif f == "d":
            out.append(f"{v:{w}d}")
        elif f == "g":
            out.append(f"{v:{w}g}")
        else:
            out.append(f"{v:{w}{f}}")
    return "  ".join(out)


def build_rows(
    s: RefStream,
    policies: Sequence[str],
    budgets: Sequence[float],
    *,
    pool_mode: str,
    prefill_policy: str,
    derate: float,
    keep_runs: bool = False,
) -> Tuple[List[dict], Dict[Tuple[str, float], PolicyRun]]:
    rows: List[dict] = []
    runs: Dict[Tuple[str, float], PolicyRun] = {}
    total_keys = s.layers * s.experts
    second_half = s.off[s.n_recs // 2]
    for gib in budgets:
        sl = slots_for(gib)
        for p in policies:
            run = simulate(s, p, sl, pool_mode=pool_mode, prefill_policy=prefill_policy)
            hd, nd = rate(s, run, kind=KIND_DECODE, warm_only=True)
            ha, na = rate(s, run, kind=None, warm_only=True)
            hd_all_incl, _ = rate(s, run, kind=KIND_DECODE, warm_only=False)
            # The half-split arms (static-prior, prior+lru) exist only on the second half, so
            # the ONLY window where every policy is defined is the second half. Gates rank on
            # that; the table still shows each policy's own native window.
            h2, n2 = rate(s, run, kind=KIND_DECODE, warm_only=True, lo=second_half)
            eff_slots = (sl // s.layers) * s.layers if pool_mode == "per-layer" else sl
            h = 0.0 if math.isnan(hd) else hd
            rows.append(
                {
                    "policy": p,
                    "cache_gib_per_rank": gib,
                    "slots": sl,
                    "effective_slots": eff_slots,
                    "slots_per_layer": sl // s.layers,
                    "coverage": eff_slots / total_keys,
                    "h_decode": hd,
                    "h_all": ha,
                    "h_decode_incl_warmup": hd_all_incl,
                    "h_decode_2nd_half": h2,
                    "n_decode_refs_2nd_half": n2,
                    "n_decode_refs": nd,
                    "misses_per_token": misses_per_token(h),
                    "mb_per_token_per_rank": mb_per_token(h),
                    "ms_step": ms_step(h),
                    "tok_s": tok_s(h),
                    "speedup_vs_today": tok_s(h) / TOKS_TODAY,
                    "tok_s_derated": derated_tok_s(h, derate),
                    "note": run.note,
                    "sim_seconds": round(run.seconds, 2),
                }
            )
            if keep_runs:
                runs[(p, gib)] = run
            else:
                run.flags = bytearray()  # free ~n bytes per run
                runs[(p, gib)] = run
    return rows, runs


def print_table(rows: Sequence[dict]) -> None:
    print("  ".join(n.ljust(w) if f == "s" else n.rjust(w) for n, w, f in _COLS))
    print("-" * (sum(w for _, w, _ in _COLS) + 2 * (len(_COLS) - 1)))
    for r in rows:
        print(
            _fmt_row(
                [
                    r["policy"],
                    r["cache_gib_per_rank"],
                    r["slots"],
                    r["coverage"],
                    r["h_decode"],
                    r["h_all"],
                    r["misses_per_token"],
                    r["mb_per_token_per_rank"],
                    r["ms_step"],
                    r["tok_s"],
                    r["speedup_vs_today"],
                    r["tok_s_derated"],
                ]
            )
        )


def print_diagnostics(d: dict) -> None:
    print("\n=== DIAGNOSTIC BLOCK (what explains the answer) ===")
    g = d["gini"]
    print(f"per-layer Gini of expert access counts: mean {g['mean']:.4f}  "
          f"min {g['min']:.4f}  max {g['max']:.4f}")
    print(f"distinct (layer,expert) pairs touched at decode: {d['distinct_keys_touched']}")
    print("top-C mass (fraction of decode refs covered by the C hottest pairs):")
    for k, v in d["top_c_mass"].items():
        print(f"    C = slots({k} GiB): {v:.4f}")
    sp = d["stack_distance_pooled"]
    pl = d["stack_distance_per_layer"]
    print("STACK DISTANCE (the direct predictor of LRU; d<=C is exactly the LRU-hit condition):")
    print(f"    pooled  P50 {sp['p50']:.0f}  P90 {sp['p90']:.0f}  "
          f"frac(d <= global slots) {sp['frac_le_slots_global']:.4f}  (n={sp['n_refs']})")
    print(f"    per-layer P50 {pl['p50_mean']:.1f}  P90 {pl['p90_mean']:.1f}  "
          f"frac(d <= {pl['slots_per_layer']} slots/layer) mean {pl['frac_le_slots_mean']:.4f} "
          f"[{pl['frac_le_slots_min']:.4f}, {pl['frac_le_slots_max']:.4f}]")
    print("static-prior half-split (learn 1st half, cover 2nd half) vs uniform N/512:")
    for n, v in d["static_prior_half_split"].items():
        print(f"    N={n:>4}/layer: covered {v['covered']:.4f}   uniform {v['uniform_expectation']:.4f}"
              f"   lift {v['covered'] - v['uniform_expectation']:+.4f}")
    ov = d["adjacent_step_overlap"]
    print(f"adjacent-decode-step route overlap: mean {ov['mean']:.4f} "
          f"[{ov['min']:.4f}, {ov['max']:.4f}]  (llama.cpp: ~0.442)")
    c = d["census"]
    print(f"census: {c['distinct_req_uids']} distinct req_uids, {c['decode_steps']} decode steps, "
          f"{c['decode_refs_warm']} warm decode refs, {c['prefill_records']} prefill records")
    if c["violations"]:
        print(f"  !! trace invariant violations: {c['violations']}")
    if c["truncated"]:
        print("  !! trace was truncated (SIGTERM tail dropped) — still readable, note it")


def census_admissible(d: dict, min_prompts: int = 12, min_steps: int = 20000) -> Tuple[bool, str]:
    c = d["census"]
    probs = []
    if c["distinct_req_uids"] < min_prompts:
        probs.append(f"only {c['distinct_req_uids']} distinct req_uids (need >= {min_prompts})")
    if c["decode_steps"] < min_steps:
        probs.append(f"only {c['decode_steps']} decode steps (need >= {min_steps})")
    return (not probs), "; ".join(probs)


# ---------------------------------------------------------------------------------------------
# KILL GATES
# ---------------------------------------------------------------------------------------------


def evaluate_gates(rows: Sequence[dict], budget: float = REAL_BUDGET_GIB) -> dict:
    at = {r["policy"]: r for r in rows if abs(r["cache_gib_per_rank"] - budget) < 1e-9}
    out: dict = {"budget_gib": budget, "verdicts": [], "values": {}}
    if not at:
        out["verdicts"].append(("G?", "SKIP", f"no rows at C={budget} GiB"))
        return out
    cov = next(iter(at.values()))["coverage"]
    out["values"]["coverage"] = cov
    # Rank on the 2nd-half window: it is the only one on which the half-split arms are defined,
    # and ranking policies measured on different windows is how a bogus "static wins" survives.
    h = {p: r.get("h_decode_2nd_half", r["h_decode"]) for p, r in at.items()}
    out["values"]["h_decode_2nd_half"] = h
    out["values"]["h_decode_native_window"] = {p: r["h_decode"] for p, r in at.items()}

    # G1 — does any exploitable structure exist at all
    hb = h.get("belady")
    if hb is None:
        out["verdicts"].append(("G1", "SKIP", "belady not simulated"))
    elif hb < 0.30:
        out["verdicts"].append(
            ("G1", "KILL",
             f"h_Belady={hb:.4f} < 0.30 — nothing online can exceed the offline optimum. "
             f"Ceiling is {ms_step(hb):.2f} ms/step = {tok_s(hb):.2f} tok/s "
             f"({tok_s(hb)/TOKS_TODAY:.3f}x nominal, {derated_tok_s(hb,0.5)/TOKS_TODAY:.3f}x "
             "derated). KILL THE ENTIRE HOT-EXPERT CACHE; routing is near-memoryless and "
             "P2-prime's Bernoulli(h) assumption was accidentally correct. Close unknown #5."))
    elif hb < 0.40:
        out["verdicts"].append(
            ("G1", "MARGINAL",
             f"0.30 <= h_Belady={hb:.4f} < 0.40 — survives only if a LARGER budget clears G3; "
             "report the break-even C* and stop."))
    else:
        out["verdicts"].append(("G1", "PASS", f"h_Belady={hb:.4f} >= 0.40"))

    # G2 — is LRU the right policy
    hl = h.get("lru")
    if hb is not None and hl is not None:
        if hl >= 0.75 * hb:
            out["verdicts"].append(("G2", "PASS", f"h_LRU={hl:.4f} >= 0.75*h_Belady={0.75*hb:.4f}"))
        else:
            best = max(
                ((p, v) for p, v in h.items() if p not in ("belady", "oracle-static")),
                key=lambda kv: (kv[1] if not math.isnan(kv[1]) else -1),
            )
            out["verdicts"].append(
                ("G2", "REGATE",
                 f"h_LRU={hl:.4f} < 0.75*h_Belady={0.75*hb:.4f} — do NOT kill the feature; LRU "
                 f"is leaving structure on the floor. Best online arm here is {best[0]} at "
                 f"{best[1]:.4f}; take G3 against it."))

    # G3 — the economic decision
    online = {p: v for p, v in h.items() if p not in ("belady", "oracle-static")}
    if online:
        star_p, star = max(online.items(), key=lambda kv: (kv[1] if not math.isnan(kv[1]) else -1))
        out["values"]["h_star"] = star
        out["values"]["h_star_policy"] = star_p
        if star >= 0.60:
            out["verdicts"].append(
                ("G3", "BUILD",
                 f"h*={star:.4f} ({star_p}) >= 0.60 -> {ms_step(star):.2f} ms = "
                 f"{tok_s(star):.2f} tok/s = {tok_s(star)/TOKS_TODAY:.3f}x nominal, "
                 f"{derated_tok_s(star,0.5)/TOKS_TODAY:.3f}x derated"))
        elif star < 0.45:
            out["verdicts"].append(
                ("G3", "KILL",
                 f"h*={star:.4f} ({star_p}) < 0.45 -> {tok_s(star):.2f} tok/s "
                 f"({tok_s(star)/TOKS_TODAY:.3f}x nominal, "
                 f"{derated_tok_s(star,0.5)/TOKS_TODAY:.3f}x derated). KILL outright; do not "
                 "report a break-even, do not re-open."))
        else:
            # break-even C*
            cstar = None
            for r in sorted(rows, key=lambda r: r["cache_gib_per_rank"]):
                if r["policy"] == star_p and \
                        r.get("h_decode_2nd_half", r["h_decode"]) >= 0.60:
                    cstar = r["cache_gib_per_rank"]
                    break
            msg = (f"h*={star:.4f} ({star_p}) < 0.60 at {budget} GiB — KILL THE BUILD AT THIS "
                   "BUDGET. ")
            if cstar is None:
                msg += "No swept budget reaches h=0.60; C* > the largest swept budget => KILL."
            elif cstar <= 13.4:
                msg += (f"Break-even C* = {cstar} GiB/rank <= 13.4 — re-open ONLY as a budget "
                        "question, and only if KV-pool/arena accounting can actually free it "
                        "(it currently cannot at MEM_RATIO 0.90).")
            else:
                msg += f"Break-even C* = {cstar} GiB/rank > 13.4 => KILL outright."
            out["verdicts"].append(("G3", "KILL-AT-BUDGET", msg))
    return out


def gate_g0(rows: Sequence[dict], budget: float = REAL_BUDGET_GIB) -> Tuple[bool, str]:
    """G0 — validate the SIMULATOR on the uniform null. `rows` must come from a uniform trace."""
    at = {r["policy"]: r for r in rows if abs(r["cache_gib_per_rank"] - budget) < 1e-9}
    cov = next(iter(at.values()))["coverage"]
    msgs = []
    ok = True
    for p in ("lru", "static-prior"):
        if p in at:
            d = abs(at[p]["h_decode"] - cov)
            good = d <= 0.03
            ok &= good
            msgs.append(f"|h_{p}={at[p]['h_decode']:.4f} - coverage {cov:.4f}| = {d:.4f} "
                        f"{'OK' if good else 'FAIL'} (<= 0.03)")
    if "static-layer" in at:
        r = at["static-layer"]
        n_pinned = r["slots"] // EXPERTS
        expect = n_pinned / LAYERS
        # tolerance = the compulsory-miss residue only; see T1 for the exact form
        good = abs(r["h_decode"] - expect) < 1e-3
        ok &= good
        msgs.append(
            f"h_static-layer={r['h_decode']:.6f} == n_pinned/{LAYERS} = {expect:.6f} "
            f"{'OK' if good else 'FAIL'} up to cold start "
            f"[NOTE: the spec's '== coverage exactly' is unachievable by construction — "
            f"static-layer quantises to whole 512-expert layers, so it is coverage rounded "
            f"DOWN to a multiple of 1/48; here {expect:.4f} vs coverage {cov:.4f}]")
    return ok, "\n    ".join(msgs)


# ---------------------------------------------------------------------------------------------
# SELF-TEST  (T1-T7).  Run order is a hard rule: these must be green before any real trace
# number is admissible.
# ---------------------------------------------------------------------------------------------


class _T:
    def __init__(self):
        self.results: List[Tuple[str, bool, str]] = []

    def check(self, name: str, cond: bool, detail: str) -> bool:
        self.results.append((name, bool(cond), detail))
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}: {detail}")
        return bool(cond)


def first_touch_mask(s: RefStream) -> bytearray:
    """1 where a reference is the FIRST ever to its key. Those are compulsory misses: they are
    real misses (the bytes really do cross PCIe) but they are unavoidable at ANY capacity, so
    the honest form of "the cache is big enough for everything" is `zero CAPACITY misses`, not
    `h == 1.0`. On a trace where new experts keep appearing, h == 1.0 is unreachable by
    construction and asserting it would only ever indict the trace."""
    m = bytearray(s.n_refs)
    seen = set()
    keys = s.keys
    for i in range(s.n_refs):
        k = keys[i]
        if k not in seen:
            seen.add(k)
            m[i] = 1
    return m


def capacity_misses(s: RefStream, run: PolicyRun, ft: bytearray) -> int:
    """Warm decode misses that are NOT compulsory."""
    rk, rw, _ = s.ref_arrays()
    n = 0
    fl = run.flags
    for i in range(run.eval_start, s.n_refs):
        if rk[i] == KIND_DECODE and rw[i] and not fl[i] and not ft[i]:
            n += 1
    return n


def _h_of(s: RefStream, policy: str, slots: int, **kw) -> Tuple[float, PolicyRun]:
    run = simulate(s, policy, slots, **kw)
    h, _ = rate(s, run, kind=KIND_DECODE, warm_only=True)
    return h, run


def selftest(verbose: bool = True) -> int:
    t = _T()
    t0 = time.time()
    real_slots = slots_for(REAL_BUDGET_GIB)
    coverage_real = real_slots / TOTAL_KEYS
    print(f"expert_cache_oracle selftest — real budget {REAL_BUDGET_GIB} GiB/rank = "
          f"{real_slots} slots = {real_slots/LAYERS:.1f}/layer = {coverage_real:.4f} coverage")
    print(f"(the spec quotes 5203 slots / 108.4 per layer; the exact floor of "
          f"{REAL_BUDGET_GIB}*2^30/{EXPERT_BYTES} is {real_slots} — a one-slot rounding "
          f"difference, coverage {coverage_real:.4f} either way)")

    # ---- T1 NULL: uniform routing. An LRU win here means the simulator is BROKEN. -----------
    print("\nT1 NULL (uniform routing, C=6.7 GiB): LRU must NOT beat static by more than noise")
    s1 = gen_trace("uniform", steps=3000, seed=11)
    mark_warm(s1, 300)
    h_lru, _ = _h_of(s1, "lru", real_slots)
    h_sp, _ = _h_of(s1, "static-prior", real_slots)
    h_sl, run_sl = _h_of(s1, "static-layer", real_slots)
    n_pinned = real_slots // EXPERTS
    # "h is exactly n_pinned/48 by construction" holds for CAPACITY misses only: a pinned
    # expert not yet touched during warm-up still takes its one compulsory miss, and those
    # bytes really do cross PCIe. So the exact claim is (i) ZERO capacity misses inside the
    # pinned layers, and (ii) h == n_pinned/48 minus that compulsory residue. Asserting a bare
    # `== n_pinned/48` would be asserting that compulsory misses are free, which is the very
    # accounting error this oracle exists to avoid.
    ft1 = first_touch_mask(s1)
    rk1, rw1, rl1 = s1.ref_arrays()
    cap_miss_pinned = 0
    compulsory_pinned = 0
    for i in range(s1.n_refs):
        if rk1[i] == KIND_DECODE and rw1[i] and rl1[i] < n_pinned and not run_sl.flags[i]:
            if ft1[i]:
                compulsory_pinned += 1
            else:
                cap_miss_pinned += 1
    t.check("T1.lru≈coverage", abs(h_lru - coverage_real) <= 0.03,
            f"h_lru={h_lru:.4f} coverage={coverage_real:.4f} d={abs(h_lru-coverage_real):.4f}")
    t.check("T1.static_prior≈coverage", abs(h_sp - coverage_real) <= 0.03,
            f"h_static_prior={h_sp:.4f} d={abs(h_sp-coverage_real):.4f}")
    t.check("T1.static_layer==n_pinned/48",
            cap_miss_pinned == 0 and abs(h_sl - n_pinned / LAYERS) < 1e-3,
            f"h_static_layer={h_sl:.6f} vs {n_pinned}/48 = {n_pinned/LAYERS:.6f}; "
            f"capacity misses inside pinned layers = {cap_miss_pinned} (must be 0), "
            f"compulsory = {compulsory_pinned} -> the whole {n_pinned/LAYERS - h_sl:.2e} "
            f"deficit is cold start. [NOTE: the spec's '== coverage exactly' is unachievable "
            f"by construction: static-layer quantises to whole 512-expert layers, so it is "
            f"coverage floored to a multiple of 1/48 -- {n_pinned/LAYERS:.4f} vs coverage "
            f"{coverage_real:.4f}, a gap of {coverage_real - n_pinned/LAYERS:.4f} = 84 slots "
            f"that whole-layer pinning cannot spend]")

    # ---- T2 SKEW: zipf. Learnable, so a static prior recovers most of it. -------------------
    print("\nT2 SKEW (zipf alpha=1.2): LFU >= LRU >= coverage+0.10 and static-prior >= +0.10")
    s2 = gen_trace("zipf:alpha=1.2", steps=3000, seed=22)
    mark_warm(s2, 300)
    h2_lru, _ = _h_of(s2, "lru", real_slots)
    h2_lfu, _ = _h_of(s2, "lfu", real_slots)
    h2_sp, _ = _h_of(s2, "static-prior", real_slots)
    t.check("T2.lfu>=lru", h2_lfu >= h2_lru - 1e-9, f"lfu={h2_lfu:.4f} lru={h2_lru:.4f}")
    t.check("T2.lru>=cov+0.10", h2_lru >= coverage_real + 0.10,
            f"lru={h2_lru:.4f} vs {coverage_real+0.10:.4f}")
    t.check("T2.static_prior>=cov+0.10", h2_sp >= coverage_real + 0.10,
            f"static_prior={h2_sp:.4f} vs {coverage_real+0.10:.4f} (skew IS learnable)")

    # ---- T3 LOCALITY: hotpool. THE DISCRIMINATING TEST. --------------------------------------
    # Uniform marginals + strong recency: LRU must win big while a static prior wins nothing.
    # 4 layers keeps it fast; the hot pool is sized to slots_per_layer at the real coverage.
    print("\nT3 LOCALITY (hotpool, H=slots/layer, uniform marginals): LRU wins, static prior does not")
    L3 = 4
    slots3 = int(round(coverage_real * L3 * EXPERTS))
    spl3 = slots3 // L3
    s3 = gen_trace(f"hotpool:{spl3},20", steps=20000, layers=L3, seed=33)
    mark_warm(s3, 500)
    cov3 = (slots3 // L3) * L3 / (L3 * EXPERTS)
    h3_lru, _ = _h_of(s3, "lru", slots3, pool_mode="per-layer")
    h3_sp, _ = _h_of(s3, "static-prior", slots3, pool_mode="per-layer")
    t.check("T3.lru>=cov+0.30", h3_lru >= cov3 + 0.30,
            f"h_lru={h3_lru:.4f} coverage={cov3:.4f} (+{h3_lru-cov3:.4f})")
    t.check("T3.static_prior<=cov+0.05", h3_sp <= cov3 + 0.05,
            f"h_static_prior={h3_sp:.4f} coverage={cov3:.4f} (+{h3_sp-cov3:.4f}) — locality is "
            "NOT skew, and the sim separates them")

    # ---- T4 ORDERING -------------------------------------------------------------------------
    print("\nT4 ORDERING: belady dominates, oracle-static <= belady, h monotone in C, h(full)=1")
    L4 = 8
    s4 = gen_trace("mixed:H=64,tau=15,alpha=0.9", steps=4000, layers=L4, seed=44,
                   prefill_every=400, prefill_touch=400)
    mark_warm(s4, 400)
    keys4 = L4 * EXPERTS
    budg4 = [0.0, 0.5, 1.0, 2.0]
    full_gib = (keys4 * EXPERT_BYTES + GIB) / GIB
    budg4.append(full_gib)
    ordering_ok = True
    mono_ok = True
    full_ok = True
    detail = []
    full_detail = []
    prev_h: Dict[str, float] = {}
    ft4 = first_touch_mask(s4)
    for gib in budg4:
        sl = slots_for(gib)
        hs = {}
        runs4 = {}
        for p in ("static-layer", "lru", "lfu", "slru", "belady", "oracle-static"):
            hs[p], runs4[p] = _h_of(s4, p, sl)
        for p, v in hs.items():
            if p == "belady":
                continue
            if not (hs["belady"] >= v - 1e-9):
                ordering_ok = False
                detail.append(f"belady {hs['belady']:.4f} < {p} {v:.4f} at {gib:g} GiB")
        for p, v in hs.items():
            if p in prev_h and v < prev_h[p] - 0.01:
                mono_ok = False
                detail.append(f"{p} not monotone: {prev_h[p]:.4f} -> {v:.4f} at {gib:g} GiB")
            prev_h[p] = v
        if abs(gib - full_gib) < 1e-9:
            for p in ("lru", "lfu", "belady"):
                cm = capacity_misses(s4, runs4[p], ft4)
                full_detail.append(f"{p}: h={hs[p]:.4f}, capacity misses={cm}")
                if cm != 0:
                    full_ok = False
    t.check("T4.belady_dominates", ordering_ok, "; ".join(detail) or
            "belady >= every other policy at every C")
    t.check("T4.monotone_in_C", mono_ok, "; ".join(detail) or "h non-decreasing in C for all")
    t.check("T4.full_set_no_capacity_misses", full_ok,
            "at C = the whole expert set, " + "; ".join(full_detail) +
            " (h<1 is COMPULSORY misses only — new experts first appearing after warm-up; "
            "h==1.0 exactly is unreachable on any trace that keeps touching new keys)")
    # static-prior compared on its own (second-half) window against belady on the same window
    sl = slots_for(1.0)
    r_be = simulate(s4, "belady", sl)
    half_ok = True
    half_detail = []
    for p in HALF_SPLIT_POLICIES:
        r_h = simulate(s4, p, sl)
        h_h, _ = rate(s4, r_h, kind=KIND_DECODE, warm_only=True)
        h_b, _ = rate(s4, r_be, kind=KIND_DECODE, warm_only=True, lo=r_h.eval_start)
        half_detail.append(f"belady={h_b:.4f} >= {p}={h_h:.4f}")
        if h_b < h_h - 1e-9:
            half_ok = False
    t.check("T4.belady>=half_split_arms(same window)", half_ok,
            "; ".join(half_detail) + " on the 2nd-half window")

    # ---- T5 COST ANCHOR ----------------------------------------------------------------------
    print("\nT5 COST ANCHOR: the model must reproduce today's measured 60.06 ms / 16.65 tok/s")
    t.check("T5.anchor_ms", abs(ms_step(H_TODAY) - 60.06) <= 0.05,
            f"ms_step(10/48) = {ms_step(H_TODAY):.4f} (measured 60.06)")
    t.check("T5.anchor_toks", abs(tok_s(H_TODAY) - 16.65) <= 0.02,
            f"tok_s(10/48) = {tok_s(H_TODAY):.4f} (measured 16.65)")
    t.check("T5.ceiling", abs(ms_step(1.0) - 29.05) <= 0.05,
            f"ms_step(1.0) = {ms_step(1.0):.4f} = the h=1 floor -> {tok_s(1.0):.2f} tok/s "
            f"({tok_s(1.0)/TOKS_TODAY:.3f}x hard ceiling)")

    # ---- T6 ROUND TRIP -----------------------------------------------------------------------
    print("\nT6 ROUND TRIP: writer -> reader bit-identical, and a SIGTERM'd num_records==0 file")
    import tempfile

    rng = random.Random(66)
    recs = []
    for step in range(50):
        for lid in range(LAYERS):
            ids = sorted(rng.sample(range(EXPERTS), TOPK))
            recs.append((step, 777 + step % 13, lid, KIND_DECODE, 0, 1, ids))
        if step % 10 == 0:
            for lid in range(LAYERS):
                ids = sorted(rng.sample(range(EXPERTS), 400))
                recs.append((step, 777 + step % 13, lid, KIND_PREFILL, 0, 1024, ids))
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "route.selftest.rank0.bin")
        with TraceWriter(p, tp_rank=0, model_slug="qwen4_exp") as w:
            for r in recs:
                w.add(*r)
        got = read_trace(p)
        same = got.n_records == len(recs)
        for i, r in enumerate(recs):
            if not same:
                break
            a, b = got.off[i], got.off[i + 1]
            same = (
                got.step[i] == r[0] and got.uid[i] == r[1] and got.layer[i] == r[2]
                and got.kind[i] == r[3] and got.chunk[i] == r[4] and got.ntok[i] == r[5]
                and list(got.ids[a:b]) == list(r[6])
            )
        t.check("T6.roundtrip", same, f"{got.n_records} records read back identical")
        # SIGTERM case: header never patched (num_records == 0) AND a torn final record.
        with open(p, "rb") as f:
            blob = bytearray(f.read())
        struct.pack_into("<Q", blob, struct.calcsize("<8sIIIIII"), 0)
        p2 = os.path.join(td, "route.sigterm.rank0.bin")
        with open(p2, "wb") as f:
            f.write(blob[:-7])  # tear the last record's payload
        got2 = read_trace(p2)
        t.check("T6.sigterm_tolerated",
                got2.n_records == len(recs) - 1 and got2.truncated,
                f"num_records==0 scanned to EOF, torn tail dropped: {got2.n_records} of "
                f"{len(recs)} records recovered, truncated={got2.truncated}")
        # a duplicate-id decode route must be REPORTED, not silently accepted
        p3 = os.path.join(td, "route.dup.rank0.bin")
        with TraceWriter(p3) as w:
            w.add(0, 1, 0, KIND_DECODE, 0, 1, [5, 5, 7, 8, 9, 10, 11, 12, 13, 14])
        got3 = read_trace(p3)
        t.check("T6.short_route_reported", got3.violations.get("decode_short_route") == 1,
                f"a deduped-to-9 decode route is flagged: {dict(got3.violations)}")

    # ---- T7 PREFILL --------------------------------------------------------------------------
    print("\nT7 PREFILL: --prefill-policy is wired, not cosmetic")
    L7 = 8
    s7 = gen_trace("mixed:H=96,tau=25,alpha=0.9", steps=6000, layers=L7, seed=77,
                   prefill_every=60, prefill_chunks=2, prefill_touch=400)
    mark_warm(s7, 300)
    slots7 = int(round(coverage_real * L7 * EXPERTS))
    h_ins, _ = _h_of(s7, "lru", slots7, prefill_policy=PREFILL_INSERT)
    h_noi, _ = _h_of(s7, "lru", slots7, prefill_policy=PREFILL_NO_INSERT)
    h_mru, _ = _h_of(s7, "lru", slots7, prefill_policy=PREFILL_MRU_EVICT)
    t.check("T7.no_insert>insert", h_noi > h_ins + 0.02,
            f"insert={h_ins:.4f}  no-insert={h_noi:.4f}  mru-evict={h_mru:.4f} "
            f"(+{h_noi-h_ins:.4f} for no-insert)")

    n_fail = sum(1 for _, ok, _ in t.results if not ok)
    print(f"\n{len(t.results) - n_fail}/{len(t.results)} checks passed in "
          f"{time.time()-t0:.1f}s")
    if n_fail:
        print("SELFTEST FAILED — no number from a real trace is admissible until this is green.")
    else:
        print("SELFTEST GREEN — the simulator reports no LRU win on the null, and it separates "
              "locality from skew. Real-trace numbers are admissible.")
    return 1 if n_fail else 0


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------


def _git_sha(path: str) -> Optional[str]:
    try:
        return subprocess.run(
            ["git", "-C", os.path.dirname(os.path.abspath(path)), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip() or None
    except Exception:
        return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    global EXPERT_BYTES
    ap = argparse.ArgumentParser(
        description="Expert-cache oracle: hot-expert VRAM cache vs static layer placement.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--selftest", action="store_true", help="run T1-T7 and exit non-zero on fail")
    ap.add_argument("--trace", nargs="+", default=None, help="one or more .bin route traces")
    ap.add_argument("--separate", action="store_true", help="report each trace separately")
    ap.add_argument("--synthetic", default=None,
                    help="uniform | zipf:alpha=A | hotpool:H,tau | mixed:H=..,tau=..,alpha=..")
    ap.add_argument("--synthetic-steps", type=int, default=20000)
    ap.add_argument("--synthetic-seed", type=int, default=1234)
    ap.add_argument("--emit", default=None, help="write the synthetic trace to this .bin path")
    ap.add_argument("--cache-gib", default=",".join(f"{g:g}" for g in DEFAULT_BUDGETS_GIB))
    ap.add_argument("--policy", default="all")
    ap.add_argument("--pool", choices=("global", "per-layer"), default="global")
    ap.add_argument("--prefill-policy", choices=(PREFILL_INSERT, PREFILL_NO_INSERT,
                                                 PREFILL_MRU_EVICT), default=PREFILL_INSERT)
    ap.add_argument("--warmup-steps", type=int, default=2000)
    ap.add_argument("--expert-bytes", type=int, default=EXPERT_BYTES)
    ap.add_argument("--layers", type=int, default=LAYERS)
    ap.add_argument("--experts", type=int, default=EXPERTS)
    ap.add_argument("--topk", type=int, default=TOPK)
    ap.add_argument("--derate", type=float, default=0.5)
    ap.add_argument("--lag-steps", type=int, default=0,
                    help="manager observes references N steps late (async drain). 0 = synchronous.")
    ap.add_argument("--json", default=None)
    ap.add_argument("--stack-limit", type=int, default=2_000_000,
                    help="cap on references used for the exact stack-distance pass")
    ap.add_argument("--strict", action="store_true", help="raise on trace invariant violations")
    ap.add_argument("--allow-thin", action="store_true",
                    help="print the decision table even if the prompt census fails G5")
    ap.add_argument("--no-g4", action="store_true",
                    help="skip the G4 mixed-vs-decode-only + prefill-arm sweep")
    ap.add_argument("--decode-only", action="store_true",
                    help="drop prefill records (the OPTIMISTIC stream; G4 compares the two)")
    a = ap.parse_args(argv)

    if a.selftest:
        return selftest()

    if a.expert_bytes != EXPERT_BYTES:
        print(f"# expert_bytes overridden: {EXPERT_BYTES} -> {a.expert_bytes}")
        EXPERT_BYTES = a.expert_bytes

    if not a.trace and not a.synthetic:
        ap.error("give --trace, --synthetic, or --selftest")

    budgets = [float(x) for x in a.cache_gib.split(",") if x.strip()]
    global _LAG_STEPS
    _LAG_STEPS = a.lag_steps
    policies = list(POLICIES) if a.policy == "all" else [p.strip() for p in a.policy.split(",")]

    if a.synthetic:
        streams = [gen_trace(a.synthetic, steps=a.synthetic_steps, layers=a.layers,
                             experts=a.experts, top_k=a.topk, seed=a.synthetic_seed)]
        mark_warm(streams[0], a.warmup_steps)
        if a.emit:
            s = streams[0]
            with TraceWriter(a.emit, num_layers=a.layers, num_experts=a.experts,
                             top_k=a.topk, expert_bytes=EXPERT_BYTES) as w:
                for r in range(s.n_recs):
                    ids = [s.keys[i] - s.rec_layer[r] * s.experts
                           for i in range(s.off[r], s.off[r + 1])]
                    w.add(s.rec_step[r], s.rec_uid[r], s.rec_layer[r], s.rec_kind[r], 0,
                          1 if s.rec_kind[r] == KIND_DECODE else 1024, ids)
            print(f"# wrote synthetic trace -> {a.emit}")
    else:
        streams = load_streams(a.trace, a.warmup_steps, a.separate, a.strict)

    if a.decode_only:
        for i, s in enumerate(streams):
            b = _StreamBuilder(layers=s.layers, experts=s.experts, top_k=s.top_k,
                               source=s.sources[0] + "+decode-only")
            for r in range(s.n_recs):
                if s.rec_kind[r] != KIND_DECODE:
                    continue
                ids = [s.keys[j] - s.rec_layer[r] * s.experts
                       for j in range(s.off[r], s.off[r + 1])]
                b.add(s.rec_step[r], s.rec_uid[r], s.rec_layer[r], s.rec_kind[r], 0, 1, ids)
            streams[i] = b.finish(a.warmup_steps)

    payload = {
        "constants": {
            "layers": LAYERS, "experts": EXPERTS, "top_k": TOPK,
            "expert_bytes": EXPERT_BYTES, "expert_set_gib": EXPERT_SET_GIB,
            "t_dev_ms": T_DEV_MS, "t_host_ms": T_HOST_MS, "s_miss_ms": S_MISS_MS,
            "nonmoe_ms": NONMOE_MS, "base_ms": BASE_MS, "miss_slope_ms": MISS_SLOPE_MS,
            "step_ms_today": STEP_MS_TODAY, "tok_s_today": TOKS_TODAY, "h_today": H_TODAY,
        },
        "argv": list(argv) if argv is not None else sys.argv[1:],
        "git_sha": _git_sha(__file__),
        "derate": a.derate,
        "pool": a.pool,
        "prefill_policy": a.prefill_policy,
        "warmup_steps": a.warmup_steps,
        "traces": [],
    }

    rc = 0
    for s in streams:
        print(f"\n================ {s.sources[0]} ================")
        print(f"geometry: {s.layers} layers x {s.experts} experts, top_k {s.top_k}; "
              f"{s.n_recs} records, {s.n_refs} references; pool={a.pool}, "
              f"prefill={a.prefill_policy}, warmup={a.warmup_steps} steps")
        budgets_slots = [(g, slots_for(g)) for g in budgets]
        diag = diagnostics(s, budgets_slots, a.stack_limit)
        print_diagnostics(diag)

        ok, why = census_admissible(diag)
        if not ok and not a.allow_thin and a.synthetic is None:
            print("\n!! G5 PROVENANCE FAILURE — refusing to print a decision table: " + why)
            print("   Retake the trace (>=12 distinct prompts, >=20000 decode steps, "
                  "temperature>0, rank0==rank1, meta sidecar present). Report nothing.")
            payload["traces"].append({"source": s.sources[0], "diagnostics": diag,
                                      "g5": {"admissible": False, "why": why}})
            rc = 2
            continue

        rows, runs = build_rows(s, policies, budgets, pool_mode=a.pool,
                                prefill_policy=a.prefill_policy, derate=a.derate,
                                keep_runs=True)
        print("\n=== DECISION TABLE ===")
        print_table(rows)

        real = slots_for(REAL_BUDGET_GIB)
        if ("lru", REAL_BUDGET_GIB) in runs:
            print("\n=== PREFILL POLLUTION CURVE (LRU @ 6.7 GiB, decode h by block of 64 "
                  "steps after a prefill) ===")
            for e in pollution_curve(s, runs[("lru", REAL_BUDGET_GIB)]):
                if e["n"]:
                    print(f"    steps {e['decode_steps_after_prefill']:>9}: h={e['h']:.4f} "
                          f"(n={e['n']})")

        g4 = (g4_analysis(s, REAL_BUDGET_GIB, a.pool, a.warmup_steps)
              if not a.no_g4 and not a.decode_only else {"note": "skipped"})
        if "arms" in g4 and g4["arms"]:
            print("\n=== G4 PREFILL POLLUTION (h_decode @ 6.7 GiB) ===")
            for k, v in sorted(g4["arms"].items()):
                print(f"    mixed  {k:<24} h={v:.4f}")
            for k, v in sorted(g4["decode_only"].items()):
                print(f"    decode-only {k:<19} h={v:.4f}   "
                      f"(optimistic - honest = {g4['delta'][k]:+.4f})")
            print("    " + g4["verdict"])

        gates = evaluate_gates(rows)
        print("\n=== KILL GATES @ 6.7 GiB/rank ===")
        print("  (ranked on the 2nd-half window, the only one where the half-split arms "
              "static-prior / prior+lru are defined)")
        for g, verdict, msg in gates["verdicts"]:
            print(f"  {g} {verdict}: {msg}")
        if not ok:
            reason = ("synthetic stream — the census gate is about REAL traffic and does not "
                      "apply" if a.synthetic else "--allow-thin was given")
            print(f"  G5 PROVENANCE: FAIL ({reason}; this verdict is NOT admissible for the "
                  "real decision): " + why)
        else:
            print("  G5 PROVENANCE: census OK (still requires the meta sidecar, temperature>0, "
                  "and rank0==rank1 byte-identity, which live outside this file)")

        payload["traces"].append(
            {"source": s.sources[0], "geometry": {"layers": s.layers, "experts": s.experts,
                                                  "top_k": s.top_k, "records": s.n_recs,
                                                  "refs": s.n_refs},
             "rows": rows, "diagnostics": diag, "gates": gates, "g4": g4,
             "pollution_curve": (pollution_curve(s, runs[("lru", REAL_BUDGET_GIB)])
                                 if ("lru", REAL_BUDGET_GIB) in runs else []),
             "g5": {"admissible": ok, "why": why}}
        )

    if a.json:
        with open(a.json, "w") as f:
            json.dump(payload, f, indent=2, default=float)
        print(f"\n# wrote {a.json}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
