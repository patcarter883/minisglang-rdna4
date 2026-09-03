"""The two weight stacks, and the per-expert stack SELECTOR that the kernels would index.

BACKGROUND — WHY THERE ARE TWO STACKS AND NOT ONE MIXED-MEDIA RANGE
    Phase 0 (P1/P2/P3, ``docs/measurements/WEIGHT_OFFLOAD_2026-09-02/PHASE0_REPORT.md``) established
    that ``hipMemCreate(location.type = hipMemLocationTypeHost)`` **silently returns device VRAM**
    on this box: the property is echoed back verbatim by the query, a ``location=Device`` control is
    indistinguishable, and the pages read at HBM speed. There is therefore **no single device
    virtual address whose sub-ranges live on different media**, and the plan's mixed device/host
    stack (§2) is not constructible. The working mechanism is
    ``hipHostMalloc(...Mapped)`` + ``hipHostGetDevicePointer``, which yields a **separate** VA.

    So a mixed-media layer is necessarily TWO contiguous stacks plus a per-expert selector.

WHAT P2prime MEASURED, AND WHY THE SELECTOR IS BUILT BUT NOT WIRED
    P2prime built exactly that (device ``hipMalloc`` stack + pinned host stack + a per-expert
    ``__constant__`` pointer table) on top of the shipped decode-GEMV core, and ran it on both
    cards. Results:

      * ``cliff_index`` 0.090 (card 0) / 0.094 (card 1) against a pure-linear 0.100 → **LINEAR**.
        The ``h^10`` all-resident-layer cliff that motivated per-expert placement is **REFUTED**:
        one additional host-resident expert costs exactly one granule moved at the card's measured
        host bandwidth (29.7 vs 28.92 GB/s = 102.8% on card 0; 14.66 vs 14.47 = 101.3% on card 1).
      * But **because** the curve is linear, per-expert and layer-granular placement are provably
        equivalent at equal byte budget. ``per_expert_gain_over_layer_granular`` peaks at 1.063x and
        is **1.013x / 1.009x at h≈0.25**, the band the 16 GB VRAM budget against a 68.8 GiB model
        actually forces.

    Conclusion, and it is the decision this module encodes: **placement is LAYER-GRANULAR.** A
    per-expert device tier buys ~1% at the reachable operating point in exchange for a route change,
    a ``slot_of``, a second stack in the hot path, and a kernel change. It is not built here.

    ``ExpertStackTable`` is still a first-class object because it is the **residency ledger**, not
    a scheduling structure: it is what the permuted-mirror gate (A1.2) permutes, what telemetry
    reads, and what an EP rank's global→local remap acts on. Under the layer-granular plan every
    table is uniform (``is_uniform`` is True) and is never handed to a kernel.

EXACTLY WHAT WOULD CHANGE IN THE HIP KERNELS IF A MIXED LAYER WERE EVER BUILT
    Nothing forks. Both shared cores already funnel *every* per-expert base-pointer computation
    through three ``WLoad`` hooks, so a two-stack layout is a **loader policy on the existing
    core** (KERNEL_CORE_POLICY.md), which is precisely what P2prime demonstrated:

      core 1  ``fp8_wmma/fp8_wmma_rocm/gemv_decode.h`` — ``gemv_decode::gemv_decode_impl``,
              line ~1560: ``wq_e = WLoad::wq_expert(w_data, e, N, K);`` (+ ``ws_expert`` /
              ``wz_expert``). This is the decode path: MoE gemm1+silu, gemm2 gather-reduce and
              gemm2 scatter at M<=16, i.e. every bs=1 decode launch.
      core 2  ``fp8_wmma/fp8_wmma_rocm/moe_gemm_tiled.h`` (lines ~339 and ~511) and its
              ``moe_gemm_flag.h`` sibling (line ~65) — the WMMA grouped-GEMM prefill path, same
              three hooks.

    The change is, in full:
      1. One new templated policy, ``TwoStackWLoad<Base>``, that inherits *everything* from the
         shipped loader (``decode32``/``dot32``/``accum``/``accum_bylane``/``wscale_epi``/
         ``uses_act_scale_v``, and the WMMA tile/fragment code) and overrides only the three
         ``*_expert`` hooks to return ``c_wq_tab[e & E_MASK]`` instead of ``w + e*N*K``. The mask
         is a hardware-cheap bound so a corrupt expert id reads a valid table slot rather than
         faulting off the end of constant memory. **No inner loop, register allocation, tiling or
         numeric changes** — by construction, because the policy is a base-pointer substitution.
      2. Three ``__constant__`` pointer arrays (``c_wq_tab``/``c_ws_tab``/``c_wz_tab``) plus a
         ``set_tables(wq, ws, wz, n)`` host entry point that writes all three **together**. All
         three, always: scales and zero-points must travel with the weights or expert ``e``'s
         nibbles get dequantized against expert ``f``'s scale — plausible numbers, no crash.
      3. A ``use_table`` flag on each entry point selecting the stock loader, so the A/B baseline
         is the OLD CODE rather than an emulation of it.

    Two defects P2prime found in exactly this shape, which any future implementation must not
    repeat (both are in ``tools/offload/p2prime_kernels.hip`` / ``p2prime_two_stack_moe.py`` now):
      * the null-base guard must be ``if (!use_table && (!w || !s)) return EBADARG;`` — guarding
        ``w`` alone (or ``s`` alone) either blocks the table path outright or lets an ignored table
        read a valid stack silently;
      * ``wz_expert`` uses ``z`` as a **flag** (returns nullptr when ``z`` is null), so table mode
        must pass a never-dereferenced non-null sentinel or zero-points are silently disabled
        across the whole table path.

    None of that is landed. This module deliberately stops at the Python-side ledger.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import IntEnum
from typing import TYPE_CHECKING, Any, Callable, Sequence

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

# NO module-scope `import torch`. This module is the boundary of the torch-free planning layer:
# `chunk_plan`, `host_capacity`, `prior`, `placement` and this file must all import on a machine
# where `import torch` fails outright (as it does on this box's host, outside the serve image), so
# the whole of the placement and capacity arithmetic — the part that can be silently wrong by a
# factor of two — is unit-testable without a GPU or even a working torch. Only `TorchStackAllocator`
# and `ExpertStackTable.as_tensor` touch tensors, and they import torch on call.

__all__ = [
    "StackKind",
    "StackAllocator",
    "TorchStackAllocator",
    "HostStackUnavailable",
    "ExpertStackTable",
]


class StackKind(IntEnum):
    """Which of the two contiguous stacks a weight lives on.

    The integer values are the on-device selector encoding: a future ``c_*_tab`` upload indexes
    ``[DEVICE, HOST]``, so DEVICE must stay 0 (the "no offload" default fills a zeroed table).
    """

    DEVICE = 0
    HOST = 1


# NOTE: the measured per-card bandwidths, the pinned-host ceiling and the compute floor deliberately
# do NOT live here — they are `weights/prior.OffloadPrior`, one frozen, provenance-carrying table for
# the whole feature. Duplicating 28.93/14.48 in a second module is how the two drift and how a
# planner silently starts projecting against the FAST card on the rank that owns the slow one.


class HostStackUnavailable(RuntimeError):
    """A host-stack allocation was requested but no host arena was injected."""


class StackAllocator(ABC):
    """Hands out tensors on a named stack.

    This is the seam between the MoE interposition (this module's caller) and the arena that owns
    the pinned host pages. It is deliberately tensor-shaped rather than pointer-shaped: P5b proved
    ``hipHostMalloc(Mapped|Portable)`` + ``hipHostGetDevicePointer`` + ``_cuda_customAllocator`` /
    ``MemPool`` hands torch a usable ``data_ptr()`` under graph capture with **zero** ``hipMalloc``
    fallbacks, so the arena can and should present a torch-tensor interface.

    Two facts from P5b bind any implementation of this ABC:
      * the free callback fires **zero** times for live *and* for cached/dropped pool blocks, so a
        forward-only bump allocator is the correct and only design — torch will never hand memory
        back. Sizing must be right at construction.
      * ``hipPointerGetAttributes`` reports ``memory_type = 1`` ("Device") for the real host arena,
        the exact inverse of the Phase-0 trap where the query echoed "Host" for VRAM. **Never**
        gate a residency check on that query, in either direction.
    """

    @abstractmethod
    def alloc_like(self, kind: StackKind, t: "torch.Tensor") -> "torch.Tensor":
        """Return an *uninitialized* tensor with ``t``'s shape and dtype on ``kind``'s stack."""

    def bytes_used(self, kind: StackKind) -> int:
        return 0

    def describe(self) -> str:
        return type(self).__name__


class TorchStackAllocator(StackAllocator):
    """The one concrete allocator: plain ``torch.empty`` for the device stack, an injected callable
    for the host stack.

    The injected ``host_alloc(shape, dtype) -> Tensor`` is where the pinned-host arena plugs in
    (``arena.empty``). With it left ``None`` this allocator serves the degenerate all-device plan
    and every GPU-free unit test (pass ``device="cpu"``, and a CPU ``host_alloc`` to stand in for
    the arena). One class rather than a device/host pair, so there is exactly one place that
    decides where a tensor comes from.
    """

    def __init__(
        self,
        device: "torch.device | str" = "cpu",
        host_alloc: Callable[[tuple[int, ...], Any], "torch.Tensor"] | None = None,
    ) -> None:
        import torch

        self.device = torch.device(device)
        self._host_alloc = host_alloc
        self._bytes = {StackKind.DEVICE: 0, StackKind.HOST: 0}

    def alloc_like(self, kind: StackKind, t: "torch.Tensor") -> "torch.Tensor":
        import torch

        shape = tuple(t.shape)
        if kind is StackKind.HOST:
            if self._host_alloc is None:
                raise HostStackUnavailable(
                    "no host stack: TorchStackAllocator was built without `host_alloc`, so it can "
                    "only serve StackKind.DEVICE. Inject the pinned-host arena's allocator "
                    "(shape, dtype) -> Tensor before planning any host-resident layer."
                )
            out = self._host_alloc(shape, t.dtype)
            if tuple(out.shape) != shape or out.dtype != t.dtype:
                raise RuntimeError(
                    f"host arena returned {tuple(out.shape)}/{out.dtype}, expected {shape}/{t.dtype}"
                )
        else:
            out = torch.empty(shape, dtype=t.dtype, device=self.device)
        self._bytes[kind] += out.numel() * out.element_size()
        return out

    def bytes_used(self, kind: StackKind) -> int:
        return self._bytes[StackKind(kind)]

    def describe(self) -> str:
        host = "arena" if self._host_alloc is not None else "unavailable"
        return f"TorchStackAllocator(device={self.device}, host={host})"


# Canonical uniform tables, keyed (num_experts, kind). See `ExpertStackTable.uniform`. Bounded by
# the number of distinct expert counts in a model (one or two), not by the layer count.
_UNIFORM_CACHE: "dict[tuple[int, int], ExpertStackTable]" = {}


class ExpertStackTable:
    """Which stack each expert of one layer lives on — the residency ledger.

    Immutable by construction: every transform returns a new table. That is the point. Residency is
    a *placement* decision frozen at boot, not a *scheduling* one, so there is no in-place mutator
    to call from a forward and therefore no publish fence, no WAR hazard, and no capture hook.
    """

    __slots__ = ("_kinds", "_tensors")

    def __init__(self, kinds: Sequence[int]) -> None:
        vals = tuple(int(k) for k in kinds)
        for v in vals:
            if v not in (int(StackKind.DEVICE), int(StackKind.HOST)):
                raise ValueError(f"stack id {v} is not a StackKind")
        if not vals:
            raise ValueError("ExpertStackTable needs at least one expert")
        self._kinds = vals
        self._tensors: dict[tuple[str, str], Any] = {}

    # ── construction ────────────────────────────────────────────────────────────────────────
    @classmethod
    def uniform(cls, num_experts: int, kind: StackKind) -> "ExpertStackTable":
        """The CANONICAL uniform table for `(num_experts, kind)` — one shared, cached instance.

        Shared rather than freshly minted, because `as_tensor`'s address-stability guarantee is per
        INSTANCE and every current caller mints a new instance every time it asks
        (`LayerPlacement.table()` returns `ExpertStackTable.uniform(...)`, `MoEWeightSeam.bind`
        rebuilds its ledger). Under graph capture a per-call instance means a per-call
        `torch.tensor(...)`, i.e. a fresh device allocation whose pointer the graph bakes in and
        which is then dropped — a captured replay reading a dead address, which is exactly the
        failure mode `as_tensor` documents itself as preventing. Nothing in the shipped
        layer-granular path hands a table to a kernel today; caching here means the day one does,
        the guarantee is true instead of merely written down.

        Safe to share: the table is immutable (every transform returns a NEW table, and the only
        mutable state is `as_tensor`'s per-(device, dtype) cache, which is a pure function of the
        table's contents). Only the UNIFORM constructor is cached — `permuted`/`with_kind`/
        `local_view` mint fresh tables, so a per-layer mixed ledger is never aliased.
        """
        key = (int(num_experts), int(kind))
        got = _UNIFORM_CACHE.get(key)
        if got is None:
            got = cls((int(kind),) * num_experts)
            _UNIFORM_CACHE[key] = got
        return got

    # ── queries ─────────────────────────────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self._kinds)

    def __getitem__(self, e: int) -> StackKind:
        return StackKind(self._kinds[e])

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ExpertStackTable) and other._kinds == self._kinds

    def __repr__(self) -> str:
        if self.is_uniform:
            return f"ExpertStackTable(uniform {self.uniform_kind.name}, E={len(self)})"
        return f"ExpertStackTable(E={len(self)}, host={self.count(StackKind.HOST)})"

    @property
    def num_experts(self) -> int:
        return len(self._kinds)

    @property
    def kinds(self) -> tuple[int, ...]:
        return self._kinds

    @property
    def is_uniform(self) -> bool:
        return len(set(self._kinds)) == 1

    @property
    def uniform_kind(self) -> StackKind:
        if not self.is_uniform:
            raise ValueError(
                "table is MIXED: no single stack. A mixed table needs the device-side selector, "
                "which is not built — see this module's docstring (P2prime measured the per-expert "
                "gain at 1.01x at the reachable operating point)."
            )
        return StackKind(self._kinds[0])

    def count(self, kind: StackKind) -> int:
        return sum(1 for k in self._kinds if k == int(kind))

    # ── transforms ──────────────────────────────────────────────────────────────────────────
    def local_view(self, offset: int, count: int) -> "ExpertStackTable":
        """The EP shard's table: global expert ids ``[offset, offset+count)`` renumbered to
        ``[0, count)``.

        This mirrors ``MoELayer._ep_dispatch``'s own remap (``lo, hi = local_expert_offset,
        lo + local_num_experts``; ``local_ids = where(is_local, g_ids - lo, 0)``) — the weight
        containers under EP are already sized to the LOCAL count on dim 0, so any table that will
        ever index them must be renumbered the same way and at the same point. Composing in the
        other order (``slot_of_local[ep_i - lo]`` before the clamp) indexes out of bounds and
        device-asserts *inside* the captured graph.
        """
        if offset < 0 or count <= 0 or offset + count > len(self._kinds):
            raise ValueError(
                f"local_view({offset}, {count}) out of range for E={len(self._kinds)}"
            )
        return ExpertStackTable(self._kinds[offset : offset + count])

    def permuted(self, perm: Sequence[int]) -> "ExpertStackTable":
        """Table for a stack whose expert rows were permuted — A1.2's permuted-mirror gate.

        ``perm[new] = old``: row ``new`` of the permuted stack holds what was expert ``old``.
        """
        if sorted(perm) != list(range(len(self._kinds))):
            raise ValueError("permuted() needs a permutation of range(num_experts)")
        return ExpertStackTable(tuple(self._kinds[old] for old in perm))

    def with_kind(self, experts: Sequence[int], kind: StackKind) -> "ExpertStackTable":
        vals = list(self._kinds)
        for e in experts:
            vals[e] = int(kind)
        return ExpertStackTable(vals)

    # ── materialisation ─────────────────────────────────────────────────────────────────────
    def as_tensor(self, device: "torch.device | str" = "cpu", dtype: Any = None) -> "torch.Tensor":
        """The device-resident selector, cached per (device, dtype).

        Cached because it must be *address-stable*: a captured graph bakes the pointer, so a
        freshly allocated table on the second capture would be read from a dead address. Nothing in
        the shipped path calls this today (every layer-granular table is uniform); it exists so the
        A/B and the future kernel policy have one canonical materialisation.
        """
        import torch

        dt = torch.int32 if dtype is None else dtype
        dev = torch.device(device)
        key = (str(dev), str(dt))
        cached = self._tensors.get(key)
        if cached is None:
            cached = torch.tensor(self._kinds, dtype=dt, device=dev)
            self._tensors[key] = cached
        return cached
