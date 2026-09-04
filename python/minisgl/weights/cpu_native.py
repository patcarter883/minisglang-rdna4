"""The CPU-tier backend the engine actually runs: `libcpumoe.so` bound to the baked pageable tensors.

`cpu_worker.NativeBackend` declares an ABI for a `.so` that did not exist and refuses to guess.
This module is that `.so`'s binding, and it deviates from the declared ABI in two ways that are
FORCED BY THE ENGINE'S DATA LAYOUT rather than chosen — both are documented at the top of
`tools/cpu_moe/cpu_moe_so.cpp` and repeated here because a reader arriving from `cpu_worker.py`
will otherwise think the ABI drifted:

  1. NO SINGLE `table` POINTER. The declared ABI takes one packed per-expert slab. The engine holds
     each layer as two stacked tensors — `gate_up_proj._w_op` (E, 2I, H/8) int32 with gate and up
     FUSED over rows, and `down_proj._w_op` (E, H, I/8) — so the call takes four base pointers per
     layer and derives the per-expert strides from (hidden, inter). Flattening to the declared slab
     would mean a third full copy of the tier's ~27 GiB.
  2. NO `lut256`. That is policy A/B state (an e4m3->float table with the per-tensor global folded
     in). The resident scales here are already `fp16(e4m3_block_scale * weight_scale_2)` — the
     loader folds at the leaf (`models/weight.py:1343-1351`) and `post_load` `del`s the checkpoint
     copies — so the policy is `WLoadVnniFp16`, whose `post_scale()` is 1.0 and which needs no table.

WHICH POLICY, AND THE 10% THAT IS REAL AND UNCLAIMED
    `sizing._CPU_WLOAD_BY_SCHEME` maps NVFP4 to `vnni_nvfp4_e4m3_g16` — the checkpoint's own e4m3
    group scale, 0.5625 B/weight, 10% smaller AND ~1800x more accurate than the fp16 fold. That
    policy reads bytes THE ENGINE NO LONGER HAS by the time the CPU tier bakes. Getting it needs a
    repacker that re-reads the raw checkpoint, which is exactly what `resolve_weight_plan(
    cpu_repacked=)` gates and what nothing implements. So this backend serves policy E at
    `layout_fraction = 1.0` and the 10% capacity/bandwidth win stays on the table. Stated here
    because the projection arithmetic in `cpu_tier.py` quotes the e4m3 figure.

THE TILING IS DONE IN PLACE
    The VNNI core reads 16x16 tiles; the engine's tensors are row-major (codes) and group-major
    (scales). Both permutations preserve byte count exactly, so `pack_layer` permutes the baked
    tensor through a small scratch buffer and costs ZERO extra resident bytes. It runs per layer,
    immediately after that layer's bake, while the bytes are still cache-warm.
"""

from __future__ import annotations

import ctypes
import os
from typing import Any, Optional, Sequence

from .cpu_tier import CpuTierError

__all__ = ["NativeVnniBackend", "default_core_list", "find_library"]

_ENV_SO = "MINISGL_CPU_MOE_SO"
_ENV_CORES = "MINISGL_CPU_MOE_CORES"


def find_library(explicit: Optional[str] = None) -> str:
    """Locate `libcpumoe.so`. RAISES with the build line rather than degrading.

    There is deliberately no fallback to `cpu_worker.ReferenceBackend`: it is a float64 numpy
    oracle three orders of magnitude too slow to serve, so a silent downgrade turns a 0.5 ms layer
    into a multi-second one and reads as a hang rather than as a missing build.
    """
    cands = [explicit, os.environ.get(_ENV_SO)]
    here = os.path.dirname(os.path.abspath(__file__))
    # …/python/minisgl/weights -> the repo root's tools/cpu_moe
    root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    cands += [os.path.join(root, "tools", "cpu_moe", "libcpumoe.so"),
              "/opt/kernels/libcpumoe.so", "/engine/tools/cpu_moe/libcpumoe.so"]
    for c in cands:
        if c and os.path.exists(c):
            return c
    raise CpuTierError(
        "the CPU-COMPUTE tier was requested but libcpumoe.so was not found (looked at "
        f"{[c for c in cands if c]}). Build it with:\n"
        "  g++ -O3 -march=znver4 -mavx512vnni -mf16c -shared -fPIC -pthread \\\n"
        "      -o tools/cpu_moe/libcpumoe.so tools/cpu_moe/cpu_moe_so.cpp\n"
        f"or point {_ENV_SO} at one. There is no float64 fallback on purpose: it would serve, "
        "~1000x too slowly, and present as a hang instead of as a missing build."
    )


def default_core_list(rank: int, ranks: int, threads: int) -> list[int]:
    """PHYSICAL cores for this rank's CPU-MoE pool. Node-wide disjoint, and core 0 is never taken.

    Two measured facts fix this (docs/CPU_MOE_OFFLOAD.md §1.4):
      * An SMT sibling of a busy core costs ~50%, so the list is PHYSICAL core ids (0..7 on this
        box; 8..15 are the siblings) and never a hardware-thread count.
      * Only core 0 boosts (4.95 GHz vs 3.95-4.07). The VNNI kernel is clock-INSENSITIVE at this
        operating point while the engine's Python forward/dispatch thread is not, so core 0 is
        left to the engine. This is the one piece of free arbitration available.

    Ranks get DISJOINT ranges because at TP=2 both rank processes' pools are real threads on the
    same eight cores; overlapping them would put two spinning pools on one core and reproduce
    §1.5's cliff (a descheduled worker costs a whole timeslice, ~12x the layer budget).
    """
    env = os.environ.get(_ENV_CORES)
    if env:
        cores = [int(x) for x in env.replace(",", " ").split()]
        per = max(1, len(cores) // max(1, ranks))
        return cores[rank * per: rank * per + threads] or cores[:threads]
    base = 1 + rank * threads          # core 0 reserved for the engine
    return list(range(base, base + threads))


class NativeVnniBackend:
    """One process-wide AVX-512-VNNI executor over every CPU-tier layer of this rank.

    `per_layer = True` is read by `CpuMoEWorker._run`: this backend is a TABLE of layers and needs
    to be told which one, where the declared single-slab ABI encoded that in the expert ids. The
    key is the seam's `backend_expert_offset` — the running expert count `bind_plan`/`SeamLayerSink`
    derive from the PLAN order — so the registration order and the attach order cannot disagree
    without both being wrong in the same way.
    """

    name = "native-avx512-vnni-fp16"
    per_layer = True

    def __init__(self, hidden: int, inter: int, top_k: int, threads: int, cores: Sequence[int],
                 so_path: Optional[str] = None) -> None:
        self.so_path = find_library(so_path)
        lib = ctypes.CDLL(self.so_path)
        vp, ci, cll = ctypes.c_void_p, ctypes.c_int, ctypes.c_longlong
        lib.cpu_moe_pack_fp16.restype = ci
        lib.cpu_moe_pack_fp16.argtypes = [vp, vp, ci, ci, vp]
        lib.cpu_moe_open.restype = vp
        lib.cpu_moe_open.argtypes = [ci, ci, ci, ci, vp, ci]
        lib.cpu_moe_run.restype = ci
        lib.cpu_moe_run.argtypes = [vp] * 5 + [vp] * 3 + [ci, ci, vp]
        lib.cpu_moe_close.argtypes = [vp]
        lib.cpu_moe_pin_self.argtypes = [vp]
        lib.cpu_moe_calls.restype = cll
        lib.cpu_moe_calls.argtypes = [vp]
        lib.cpu_moe_tokens.restype = cll
        lib.cpu_moe_tokens.argtypes = [vp]
        self._lib = lib
        self.hidden, self.inter, self.top_k = int(hidden), int(inter), int(top_k)
        self.threads, self.cores = int(threads), list(cores)
        arr = (ctypes.c_int * len(self.cores))(*self.cores)
        self._h = lib.cpu_moe_open(self.hidden, self.inter, self.top_k, self.threads, arr,
                                   len(self.cores))
        if not self._h:
            raise CpuTierError(
                f"cpu_moe_open refused hidden={hidden} inter={inter}: the VNNI core needs both "
                f"divisible by 16 (the tile is 16x16 and the NVFP4 group is 16)."
            )
        self._layers: dict[int, tuple] = {}
        self._pinned = False
        #: PYTHON-side counters, independent of the .so's. Two counts of the same events from two
        #: sides is what distinguishes "the forward never reached the tier" from "the tier ran and
        #: returned nothing".
        self.layer_calls = 0
        self.token_calls = 0

    # -- boot ------------------------------------------------------------------------------------
    def pack_layer(self, expert_offset: int, gate_up: Any, down: Any, num_experts: int) -> None:
        """Tile ONE layer's baked tensors in place and register it under its plan-order offset."""
        import torch

        w13c, w13s = gate_up._w_op, gate_up._scales_op
        w2c, w2s = down._w_op, down._scales_op
        for t, what in ((w13c, "w13 codes"), (w13s, "w13 scales"), (w2c, "w2 codes"),
                        (w2s, "w2 scales")):
            if t.device.type != "cpu":
                raise CpuTierError(
                    f"CPU-tier layer at expert offset {expert_offset}: {what} is on "
                    f"{t.device}, not host RAM. The seam was placed CPU but its bytes never left "
                    f"the card, so this layer would be computed by the host from device memory."
                )
            if not t.is_contiguous():
                raise CpuTierError(f"{what} is not contiguous; the tiler indexes it as a flat slab")
        E, N13, _ = w13c.shape
        H = self.hidden
        I = self.inter
        if (E, N13) != (num_experts, 2 * I):
            raise CpuTierError(
                f"CPU-tier layer shape mismatch: w13 is {tuple(w13c.shape)} but the worker was "
                f"opened for {num_experts} experts x (2*{I}, {H}). One worker serves layers of ONE "
                f"shape; a second shape needs a second worker, not a reinterpretation of the strides."
            )
        scratch = torch.empty(max(2 * I * H // 2, H * I // 2, 2 * I * H // 16 * 2),
                              dtype=torch.uint8)
        sp = ctypes.c_void_p(scratch.data_ptr())
        for e in range(E):
            for t_c, t_s, n, k in ((w13c[e], w13s[e], 2 * I, H), (w2c[e], w2s[e], H, I)):
                rc = self._lib.cpu_moe_pack_fp16(ctypes.c_void_p(t_c.data_ptr()),
                                                 ctypes.c_void_p(t_s.data_ptr()), n, k, sp)
                if rc != 0:
                    raise CpuTierError(f"cpu_moe_pack_fp16(N={n}, K={k}) returned {rc}")
        # Hold REFERENCES, not just pointers: these are ordinary pageable torch tensors and the
        # only thing keeping the containers alive is the model. A raw data_ptr() cached past a
        # rebind would read freed memory and produce plausible numbers.
        self._layers[int(expert_offset)] = (w13c, w13s, w2c, w2s)

    @property
    def num_layers(self) -> int:
        return len(self._layers)

    # -- hot path --------------------------------------------------------------------------------
    def compute(self, x, ids, weights, layer: int = 0):
        """`sum_j weights[j] * E_{ids[j]}(x)` for every row of `x`, on the host cores.

        Called on `CpuMoEWorker`'s dispatcher thread, which is where the pool's slice-0 work runs —
        hence the one-time `cpu_moe_pin_self` here rather than in `__init__` (which runs on the
        boot thread, i.e. the engine's main Python thread).
        """
        import numpy as np
        import torch

        try:
            w13c, w13s, w2c, w2s = self._layers[int(layer)]
        except KeyError:
            raise CpuTierError(
                f"no CPU-tier layer registered at expert offset {layer} (have "
                f"{sorted(self._layers)}). The seam's `backend_expert_offset` and the backend's "
                f"registration order came from different walks of the plan."
            ) from None
        if not self._pinned:
            self._lib.cpu_moe_pin_self(ctypes.c_void_p(self._h))
            self._pinned = True

        xt = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
        xf = xt.reshape(-1, self.hidden).to(torch.float32).contiguous()
        it = (ids if isinstance(ids, torch.Tensor) else torch.as_tensor(ids))
        wt = (weights if isinstance(weights, torch.Tensor) else torch.as_tensor(weights))
        it = it.reshape(xf.shape[0], -1).to(torch.int32).contiguous()
        wt = wt.reshape(xf.shape[0], -1).to(torch.float32).contiguous()
        M, K = it.shape
        if K > self.top_k:
            raise CpuTierError(f"route width {K} exceeds the worker's top_k={self.top_k}")
        out = torch.empty((M, self.hidden), dtype=torch.float32)
        rc = self._lib.cpu_moe_run(
            ctypes.c_void_p(self._h),
            ctypes.c_void_p(w13c.data_ptr()), ctypes.c_void_p(w13s.data_ptr()),
            ctypes.c_void_p(w2c.data_ptr()), ctypes.c_void_p(w2s.data_ptr()),
            ctypes.c_void_p(xf.data_ptr()), ctypes.c_void_p(it.data_ptr()),
            ctypes.c_void_p(wt.data_ptr()), M, K, ctypes.c_void_p(out.data_ptr()))
        if rc != 0:
            raise CpuTierError(f"cpu_moe_run returned {rc} (layer offset {layer}, M={M}, top_k={K})")
        self.layer_calls += 1
        self.token_calls += M
        del np
        return out

    # -- evidence --------------------------------------------------------------------------------
    def counters(self) -> dict:
        """THE PROOF THE TIER RAN. `engaged()` is a SET and saturates at one, so it cannot tell a
        tier that executed one layer from one that executed twenty-one. These are monotone."""
        return {
            "so": self.so_path,
            "policy": self.name,
            "layers_registered": len(self._layers),
            "threads": self.threads,
            "cores": list(self.cores),
            "native_layer_calls": int(self._lib.cpu_moe_calls(ctypes.c_void_p(self._h))),
            "native_tokens": int(self._lib.cpu_moe_tokens(ctypes.c_void_p(self._h))),
            "python_layer_calls": self.layer_calls,
            "python_tokens": self.token_calls,
        }

    def close(self) -> None:
        if getattr(self, "_h", None):
            self._lib.cpu_moe_close(ctypes.c_void_p(self._h))
            self._h = None
