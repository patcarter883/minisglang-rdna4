"""The CPU-tier backend the engine actually runs: `libcpumoe.so` bound to the baked pageable tensors.

`cpu_worker.NativeBackend` declares an ABI for a `.so` that did not exist and refuses to guess.
This module is that `.so`'s binding, and it deviates from the declared ABI in three ways that are
FORCED BY THE ENGINE'S DATA LAYOUT rather than chosen — all are documented at the top of
`tools/cpu_moe/cpu_moe_so.cpp` and repeated here because a reader arriving from `cpu_worker.py`
will otherwise think the ABI drifted:

  1. NO SINGLE `table` POINTER. The declared ABI takes one packed per-expert slab. The engine holds
     each layer as two stacked tensors — `gate_up_proj._w_op` (E, 2I, H/8) int32 with gate and up
     FUSED over rows, and `down_proj._w_op` (E, H, I/8) — so the call takes four base pointers per
     layer and derives the per-expert strides from (hidden, inter). Flattening to the declared slab
     would mean a third full copy of the tier's ~27 GiB.
  2. NO `lut256`. That is policy A/B state (an e4m3->float table with the per-tensor global folded
     in). The VNNI policies decode e4m3 by bit surgery in the inner loop and carry the second scale
     level separately, so there is no table to pass.
  3. THE GLOBALS ARE TWO MORE BASE POINTERS, per layer, for the same reason as (1): under
     `vnni_nvfp4_e4m3_g16` the second scale level is a per-output-channel `(E, N)` f32 vector that
     lives in its own tensor (`_global_op`), not folded into the scale slab.

WHICH POLICY — DECIDED BY THE SCALE DTYPE, ASSERTED POSITIVELY
    `sizing._CPU_WLOAD_BY_SCHEME` maps NVFP4 to `vnni_nvfp4_e4m3_g16` — the checkpoint's own e4m3
    group scale, 0.5625 B/weight, 10% smaller AND ~1800x more accurate than the fp16 fold. That used
    to name bytes the engine no longer had, because the fold happened at the leaf and `post_load`
    deleted the originals. Since 2026-09-05 the NVFP4 containers keep BOTH checkpoint levels — a
    1-byte e4m3 block scale (`_scales_op`) plus a per-OUTPUT-CHANNEL f32 global (`_global_op`, (E,N))
    — and as of this change the CPU core instantiates that policy, so the tier serves it directly.
    No repacker, no re-read of the checkpoint.

    THE POLICY IS CHOSEN FROM THE RESIDENT SCALE DTYPE and then ASSERTED AGAINST WHAT THE BUILT `.so`
    ADVERTISES (`cpu_moe_policies()`), which is the direction that matters. The refusal this replaced
    was NEGATIVE — "this dtype is not fp16, die" — so it would have had to be extended by hand for
    every future scale format, and the failure of forgetting is silent. `_POLICY_BY_SCALE_DTYPE`
    plus the advertised-name check is positive: an unknown dtype has no entry and is refused, and a
    known dtype served by an `.so` too old to have the policy is refused by name.

    WHY GETTING THIS WRONG IS NOT A CRASH, WHICH IS WHY IT IS ASSERTED AND NOT INFERRED. Reading an
    e4m3 slab through the fp16 packer reads HALF A SLAB (it walks into the next expert) and drops the
    global (a uniform ~4.8e3x error per weight); reading an fp16 slab through the e4m3 packer reads
    twice as many scales as exist. Both produce finite, fluent, wrong text. The `.so` refuses the
    mismatched call shape too (`cpu_moe_run2` rc 3/4), so the check exists on both sides of the seam.

THE TILING IS DONE IN PLACE
    The VNNI core reads 16x16 tiles; the engine's tensors are row-major (codes) and group-major
    (scales). Both permutations preserve byte count exactly, so `pack_layer` permutes the baked
    tensor through a small scratch buffer and costs ZERO extra resident bytes. It runs per layer,
    immediately after that layer's bake, while the bytes are still cache-warm.
"""

from __future__ import annotations

import ctypes
import os
from typing import Any, NamedTuple, Optional, Sequence

from .cpu_tier import CpuTierError

__all__ = ["NativeVnniBackend", "default_core_list", "find_library"]

_ENV_SO = "MINISGL_CPU_MOE_SO"
_ENV_CORES = "MINISGL_CPU_MOE_CORES"


class _Policy(NamedTuple):
    """One WLoad policy of the CPU core, as the seam sees it."""

    id: int          #: the `cpu_moe_open2` enum — must match `cpu_moe_so.cpp`'s POL_*
    name: str        #: the name `sizing._CPU_WLOAD_BY_SCHEME` uses AND the `.so` advertises
    pack: str        #: the `.so` symbol that tiles a matrix of this policy's scale width
    globals: bool    #: does it need `_global_op`, the per-output-channel f32 second level?


#: RESIDENT SCALE DTYPE -> the policy that reads it. Keyed on `str(dtype)` so this module stays
#: importable with no torch (the rest of the planning layer is torch-free and this table is read by
#: the same kind of caller).
#:
#: A dtype that is NOT HERE is refused, and that is the point: the previous check asked "is this
#: fp16?" and would have had to grow a new negative case per format, where a missing row here is
#: already a refusal. Adding a format is a row plus a WLoad struct in `tools/cpu_moe/wload.hpp` —
#: the `KERNEL_CORE_POLICY` edit, never a new kernel.
_POLICY_BY_SCALE_DTYPE: "dict[str, _Policy]" = {
    "torch.float16": _Policy(0, "vnni_nvfp4_fp16_g16", "cpu_moe_pack_fp16", False),
    "torch.float8_e4m3fn": _Policy(1, "vnni_nvfp4_e4m3_g16", "cpu_moe_pack_e4m3", True),
}


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
        "  make -C tools/cpu_moe\n"
        f"or point {_ENV_SO} at one. The image builds it to /opt/kernels/libcpumoe.so, so a "
        "turnkey container always has it; this refusal means a dev tree that has never run make. "
        "There is no float64 fallback on purpose: it would serve, ~1000x too slowly, and present "
        "as a hang instead of as a missing build."
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
        picked = cores[rank * per: rank * per + threads] or cores[:threads]
        _refuse_smt_siblings(picked, source=f"{_ENV_CORES}={env!r}")
        return picked
    base = 1 + rank * threads          # core 0 reserved for the engine
    # THE DERIVED LIST RUNS OFF THE END OF THE PHYSICAL CORES AND INTO THE SIBLINGS.
    # `base + threads` is unbounded, and Linux numbers the SMT sibling of physical core c as c+P
    # (measured on this box: cpu0's siblings are "0,8"). So at ranks=2, threads=4 rank 1 gets
    # [5, 6, 7, 8] -- and 8 is the sibling of core 0, the core this function's own docstring
    # reserves for the engine. That is §1.4's ~50% penalty, silently, on the rank that draws it.
    # It has been unreachable only because `CoreBudget` capped the node-wide total at 5; raising
    # that cap to measure a wider pool makes it reachable, so it is a refusal now rather than a
    # comment. The caller is told the one thing that fixes it.
    top = base + threads
    phys = _physical_core_count()
    if top > phys:
        raise CpuTierError(
            f"the CPU MoE tier asks rank {rank} for physical cores {base}..{top - 1} but this box "
            f"has {phys} ({ranks} rank(s) x {threads} thread(s), core 0 reserved for the engine). "
            f"Core ids >= {phys} are SMT SIBLINGS, not cores -- pinning a worker onto the sibling "
            f"of a busy core measured ~50% slower (docs/CPU_MOE_OFFLOAD.md §1.4), so this refuses "
            f"rather than quietly handing one out. To use every core including core 0, set "
            f"{_ENV_CORES} explicitly, e.g. "
            f"{_ENV_CORES}='{','.join(str(c) for c in range(phys))}'."
        )
    return list(range(base, top))


def _physical_core_count(default: int = 8) -> int:
    """Physical cores = distinct SMT sibling GROUPS, never `os.cpu_count()` (which counts threads)."""
    try:
        groups = set()
        for cpu in os.listdir("/sys/devices/system/cpu"):
            path = f"/sys/devices/system/cpu/{cpu}/topology/thread_siblings_list"
            if os.path.exists(path):
                with open(path) as fh:
                    groups.add(fh.read().strip())
        return len(groups) or default
    except OSError:
        return default


def _refuse_smt_siblings(cores: Sequence[int], *, source: str) -> None:
    """Refuse a hand-written core list that pins two workers onto one physical core."""
    seen: dict[str, int] = {}
    for c in cores:
        path = f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list"
        try:
            with open(path) as fh:
                group = fh.read().strip()
        except OSError:
            continue
        if group in seen:
            raise CpuTierError(
                f"{source}: cpu{seen[group]} and cpu{c} are SMT siblings of the same physical core "
                f"({group}). Two spinning MoE workers on one core is §1.4's ~50% penalty and §1.5's "
                f"barrier cliff, so this is a refusal. List one hw thread per physical core."
            )
        seen[group] = c


def _global_slab(container: Any, expert_offset: int) -> Any:
    """`container._global_op` as an (E, N) float32 view on host RAM, or a refusal saying why not.

    The container stores it BITCAST to int32 (the GPU op's `w_zeros` pointer slot is `const int*`).
    `.view(torch.float32)` is the inverse bitcast — zero-copy, exact — and is the only correct way
    back: `.to(torch.float32)` would VALUE-convert the bit patterns.
    """
    import torch

    g = getattr(container, "_global_op", None)
    if g is None:
        raise CpuTierError(
            f"CPU-tier layer at expert offset {expert_offset}: {type(container).__name__} has an "
            f"e4m3 block scale but no `_global_op`. NVFP4's scale has TWO levels and the second one "
            f"is not optional — without it every weight of this layer is short by its "
            f"per-output-channel multiplier (~4.8e3x on this checkpoint), which is finite, "
            f"plausible and wrong rather than a crash."
        )
    if g.device.type != "cpu":
        raise CpuTierError(
            f"CPU-tier layer at expert offset {expert_offset}: `_global_op` is on {g.device}, not "
            f"host RAM. The seam moved the codes and the block scales but left the global on the "
            f"card, so the host cores would read device memory."
        )
    if g.dtype == torch.int32:
        g = g.view(torch.float32)
    elif g.dtype != torch.float32:
        raise CpuTierError(
            f"CPU-tier layer at expert offset {expert_offset}: `_global_op` is {g.dtype}; expected "
            f"float32, or int32 holding the same BYTES (post_load bitcasts it for the op's "
            f"`w_zeros` slot)."
        )
    if not g.is_contiguous():
        raise CpuTierError("`_global_op` is not contiguous; the core indexes it as a flat slab")
    return g


class NativeVnniBackend:
    """One process-wide AVX-512-VNNI executor over every CPU-tier layer of this rank.

    `per_layer = True` is read by `CpuMoEWorker._run`: this backend is a TABLE of layers and needs
    to be told which one, where the declared single-slab ABI encoded that in the expert ids. The
    key is the seam's `backend_expert_offset` — the running expert count `bind_plan`/`SeamLayerSink`
    derive from the PLAN order — so the registration order and the attach order cannot disagree
    without both being wrong in the same way.
    """

    per_layer = True

    def __init__(self, hidden: int, inter: int, top_k: int, threads: int, cores: Sequence[int],
                 so_path: Optional[str] = None, scales_dtype: Any = None) -> None:
        self.so_path = find_library(so_path)
        lib = ctypes.CDLL(self.so_path)
        vp, ci, cll, cp = ctypes.c_void_p, ctypes.c_int, ctypes.c_longlong, ctypes.c_char_p
        # THE POLICY IS FIXED HERE, at open, from the dtype of the tensors this worker will be
        # handed — not per layer and not off a pointer. One worker serves layers of ONE shape (see
        # `pack_layer`'s shape refusal) and, for the same reason, of ONE scale encoding: the core is
        # instantiated per policy at compile time, so a second encoding needs a second worker.
        self.policy = self._resolve_policy(lib, scales_dtype)
        self.name = f"native-avx512-{self.policy.name}"
        lib.cpu_moe_pack_fp16.restype = ci
        lib.cpu_moe_pack_fp16.argtypes = [vp, vp, ci, ci, vp]
        self._pack = getattr(lib, self.policy.pack)
        self._pack.restype = ci
        self._pack.argtypes = [vp, vp, ci, ci, vp]
        lib.cpu_moe_open2.restype = vp
        lib.cpu_moe_open2.argtypes = [ci, ci, ci, ci, vp, ci, ci]
        lib.cpu_moe_run2.restype = ci
        lib.cpu_moe_run2.argtypes = [vp] * 10 + [ci, ci, vp]
        lib.cpu_moe_policy_name.restype = cp
        lib.cpu_moe_policy_name.argtypes = [vp]
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
        self._h = lib.cpu_moe_open2(self.hidden, self.inter, self.top_k, self.threads, arr,
                                    len(self.cores), self.policy.id)
        if not self._h:
            raise CpuTierError(
                f"cpu_moe_open2 refused hidden={hidden} inter={inter} policy={self.policy.name}: "
                f"the VNNI core needs both divisible by 16 (the tile is 16x16 and the NVFP4 group "
                f"is 16)."
            )
        # READ BACK what the handle was actually opened for, rather than trusting the id we sent.
        # A stale `.so` whose POL_* enum drifted would otherwise serve a different policy under the
        # right name, which is the one failure this whole check exists to make impossible.
        got = (lib.cpu_moe_policy_name(ctypes.c_void_p(self._h)) or b"").decode()
        if got != self.policy.name:
            raise CpuTierError(
                f"the CPU core opened policy {got!r} when asked for {self.policy.name!r} "
                f"({self.so_path}). The `.so`'s policy enum and this module's `_POLICY_BY_SCALE_"
                f"DTYPE` ids have drifted; serving would decode every scale with the wrong policy."
            )
        self._layers: dict[int, tuple] = {}
        self._pinned = False
        #: PYTHON-side counters, independent of the .so's. Two counts of the same events from two
        #: sides is what distinguishes "the forward never reached the tier" from "the tier ran and
        #: returned nothing".
        self.layer_calls = 0
        self.token_calls = 0

    @staticmethod
    def _resolve_policy(lib: Any, scales_dtype: Any) -> _Policy:
        """Which WLoad policy serves `scales_dtype`, PROVEN available in this build.

        Three ways to fail, all loud, none of them "serve anyway":
          * a scale dtype with no policy at all (a format nobody wrote a WLoad for),
          * a policy this `.so` does not advertise (a build from before it existed),
          * a policy whose pack symbol is missing (a partial build).
        The caller has to say what it holds; there is no default, because the default would be the
        old hardcoded fp16 and that is exactly the silent-wrong-answer this replaced.
        """
        if scales_dtype is None:
            raise CpuTierError(
                "NativeVnniBackend needs `scales_dtype` — the dtype of the resident `_scales_op` "
                "slab it will be handed. It selects the WLoad policy (fp16 folded group scale vs "
                "the checkpoint's e4m3 block scale + per-channel global), and the two read "
                "different numbers of bytes per group, so a wrong guess reads half a slab or twice "
                "one. There is no safe default."
            )
        key = str(scales_dtype)
        pol = _POLICY_BY_SCALE_DTYPE.get(key)
        if pol is None:
            raise CpuTierError(
                f"no CPU-core WLoad policy reads a {key} scale slab (known: "
                f"{sorted(_POLICY_BY_SCALE_DTYPE)}). Adding one is a `struct` in "
                f"tools/cpu_moe/wload.hpp plus a row in `_POLICY_BY_SCALE_DTYPE` — the "
                f"KERNEL_CORE_POLICY edit, on the SHARED core in moe_core.hpp, never a new kernel."
            )
        try:
            lib.cpu_moe_policies.restype = ctypes.c_char_p
            advertised = (lib.cpu_moe_policies() or b"").decode().split(",")
        except AttributeError:
            advertised = []
        if pol.name not in advertised or not hasattr(lib, pol.pack):
            raise CpuTierError(
                f"this build of libcpumoe.so does not serve `{pol.name}` (it advertises "
                f"{advertised or ['<nothing: no cpu_moe_policies symbol>']}, and "
                f"{'has' if hasattr(lib, pol.pack) else 'is MISSING'} `{pol.pack}`). A "
                f"{key} scale slab needs it. Rebuild:  make -C tools/cpu_moe"
            )
        return pol

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
        # THE SCALE SLAB MUST BE THE ONE THIS WORKER WAS OPENED FOR. The policy fixes the scale
        # WIDTH the packer and the core read (1 B e4m3 vs 2 B fp16), so a mismatch is not a
        # reinterpretation, it is a different byte count per group: too few and the read walks into
        # the next expert, too many and half the groups are noise. Neither faults.
        want = {p.name: d for d, p in _POLICY_BY_SCALE_DTYPE.items()}[self.policy.name]
        for t, what in ((w13s, "w13 scales"), (w2s, "w2 scales")):
            if str(t.dtype) != want:
                raise CpuTierError(
                    f"CPU-tier layer at expert offset {expert_offset}: {what} is {t.dtype}, but "
                    f"this worker was opened for policy `{self.policy.name}`, whose packer "
                    f"`{self.policy.pack}` reads a {want} slab. One worker serves ONE scale "
                    f"encoding — the core is instantiated per policy at compile time — so a second "
                    f"encoding needs a second worker, not a reinterpretation of this one's strides."
                )
        # THE SECOND SCALE LEVEL. `_global_op` is the (E, N) f32 per-OUTPUT-CHANNEL multiplier,
        # BITCAST to int32 because the GPU op passes it through the `w_zeros` pointer slot
        # (layers/moe.py). It is a bitcast, not a conversion, so the f32 bytes are intact and the
        # CPU core reads them as f32 — the same `reinterpret_cast` the HIP kernel does. A
        # `.to(torch.float32)` here would VALUE-convert the int32 bit patterns and produce globals
        # of ~1e-38 or 0; the two spellings are one method apart, so this comment is the guard.
        w13g = w2g = None
        if self.policy.globals:
            w13g, w2g = _global_slab(gate_up, expert_offset), _global_slab(down, expert_offset)
        E, N13, _ = w13c.shape
        H = self.hidden
        I = self.inter
        if (E, N13) != (num_experts, 2 * I):
            raise CpuTierError(
                f"CPU-tier layer shape mismatch: w13 is {tuple(w13c.shape)} but the worker was "
                f"opened for {num_experts} experts x (2*{I}, {H}). One worker serves layers of ONE "
                f"shape; a second shape needs a second worker, not a reinterpretation of the strides."
            )
        for name, g, n in (("gate_up_proj", w13g, 2 * I), ("down_proj", w2g, H)):
            if g is not None and tuple(g.shape) != (E, n):
                raise CpuTierError(
                    f"CPU-tier layer at expert offset {expert_offset}: {name}._global_op is "
                    f"{tuple(g.shape)}, expected ({E}, {n}) — one f32 per expert per OUTPUT "
                    f"CHANNEL of this GEMM. A per-tensor scalar broadcast here would apply channel "
                    f"0's multiplier to every row of a matrix whose gate and up halves were merged "
                    f"from differently-scaled leaves."
                )
        scratch = torch.empty(max(2 * I * H // 2, H * I // 2, 2 * I * H // 16 * 2),
                              dtype=torch.uint8)
        sp = ctypes.c_void_p(scratch.data_ptr())
        for e in range(E):
            for t_c, t_s, n, k in ((w13c[e], w13s[e], 2 * I, H), (w2c[e], w2s[e], H, I)):
                rc = self._pack(ctypes.c_void_p(t_c.data_ptr()),
                                ctypes.c_void_p(t_s.data_ptr()), n, k, sp)
                if rc < 0:
                    # The e4m3 packer's OWN precondition, checked on the served bytes rather than
                    # inherited from the layer-0..3 census that established it: the fast decode
                    # (`e4m3x16_normpos_to_ps`, 3 ops instead of 19) is specialised to
                    # positive-normal e4m3, and a byte outside 0x08..0x7E decodes to a plausible
                    # wrong scale, not to a fault. Do not widen this to make a checkpoint load —
                    # widen the DECODER (`build_e4m3_lut` is the general one) and re-measure, since
                    # the branchy version measured 29.7% slower and inverted the layout decision.
                    raise CpuTierError(
                        f"{self.policy.pack}(N={n}, K={k}) found weight_scale byte "
                        f"0x{-rc - 1:02X} outside the positive-normal e4m3 range 0x08..0x7E that "
                        f"the CPU core's fast scale decode assumes (expert {e} of the layer at "
                        f"offset {expert_offset}). This checkpoint needs the general decode."
                    )
                if rc != 0:
                    raise CpuTierError(f"{self.policy.pack}(N={n}, K={k}) returned {rc}")
        # Hold REFERENCES, not just pointers: these are ordinary pageable torch tensors and the
        # only thing keeping the containers alive is the model. A raw data_ptr() cached past a
        # rebind would read freed memory and produce plausible numbers.
        self._layers[int(expert_offset)] = (w13c, w13s, w13g, w2c, w2s, w2g)

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
            w13c, w13s, w13g, w2c, w2s, w2g = self._layers[int(layer)]
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
        gp = (lambda t: ctypes.c_void_p(t.data_ptr() if t is not None else None))
        rc = self._lib.cpu_moe_run2(
            ctypes.c_void_p(self._h),
            ctypes.c_void_p(w13c.data_ptr()), ctypes.c_void_p(w13s.data_ptr()), gp(w13g),
            ctypes.c_void_p(w2c.data_ptr()), ctypes.c_void_p(w2s.data_ptr()), gp(w2g),
            ctypes.c_void_p(xf.data_ptr()), ctypes.c_void_p(it.data_ptr()),
            ctypes.c_void_p(wt.data_ptr()), M, K, ctypes.c_void_p(out.data_ptr()))
        if rc in (3, 4):
            # The `.so`'s own half of the policy check. rc 3 = the e4m3 core got no globals, rc 4 =
            # the fp16 core got some. Both are the silent-wrong-answer this seam is built to refuse.
            raise CpuTierError(
                f"cpu_moe_run2 refused the call shape for policy `{self.policy.name}` (rc={rc}, "
                f"layer offset {layer}): the second scale level was "
                f"{'MISSING' if rc == 3 else 'PASSED to a policy that has it folded in already'}."
            )
        if rc != 0:
            raise CpuTierError(f"cpu_moe_run2 returned {rc} (layer offset {layer}, M={M}, top_k={K})")
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
            "policy": self.policy.name,
            "backend": self.name,
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
