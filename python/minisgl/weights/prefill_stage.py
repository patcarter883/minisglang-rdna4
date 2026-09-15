"""Explicit-copy expert staging for PREFILL — the DMA fallback the offload plan named and skipped.

WHY THIS EXISTS, and why it is not a second placement mechanism.

`WEIGHT_OFFLOAD_PLAN.md` gate P1 asked one question: can a GPU kernel read expert weights straight
out of pinned host memory fast enough that an explicit copy to VRAM is not worth its buffer? It was
measured at the grouped-GEMM's own tiled access pattern and answered **28.93 GB/s on card 0** — 2.2x
over the 13 GB/s floor below which the plan says to fall back to "explicit-copy layer-group
streaming". On that evidence the copy was correctly not built, and `MoEWeightSeam.resolve` stayed an
identity: the kernel reads the arena in place.

That measurement was taken at DECODE width and generalised to every width. It does not hold at
prefill width. Measured 2026-09-15 on the served Qwen4-Exp arm: `minisgl_prefill_seconds_total`
558.6 s for 2243 prompt tokens = **4.0 tok/s**, against 16-20 tok/s decode on the same serve — a
prefill that is slower PER TOKEN than decode, which is backwards. The mechanism is in the plan's own
words at its line 280: "a prefill chunk touches ~all 512 experts per layer". Each chunk therefore
sweeps the entire 27.9 GiB/rank host expert set through the GEMM's host reads, and the expert cache
cannot absorb it — `_admit_ok`'s second-reference rule keeps one-touch prefill sweeps OUT by design,
because admitting them measured worse (prefill pollution +0.028 LFU vs -0.0008 SLRU).

So this module invokes the fallback, for the regime that fails the gate, and ONLY that regime:

  * DECODE IS UNTOUCHED. One token reads top_k experts; staging a whole layer to read 8 of 512 would
    be a catastrophic pessimisation. The gate below is `num_tokens * top_k_local >= num_experts`,
    i.e. exactly "this launch would touch essentially every expert anyway", which is the condition
    under which a bulk sequential DMA beats scattered in-place reads. Below it, `resolve` stays the
    identity it is today and nothing in this file runs.
  * IT IS NOT VMM. The plan's dynamic-residency design (§8/M3) was killed by P6 — `hipMemUnmap` +
    `hipMemMap` on a live VA serves the STALE page on gfx1201. That defect retired the explicit-copy
    option alongside it by association, which was an error: a plain `copy_` into an ordinary device
    buffer remaps no virtual address and is untouched by P6.
  * ONE SLAB, ORDERED BY THE STREAM. Layer L's copies are issued on the COMPUTE stream, ahead of
    layer L's kernels; layer L+1 reuses the same slab, and its copies cannot start before L's
    kernels retire because they are the same stream. No event, no sync, no double buffer, and no
    way to read a half-overwritten slab. Overlap (prefetch L+1 behind L's compute) is a further
    ~15% by the plan's own estimate and is deliberately NOT done here: it needs a second slab and a
    cross-layer schedule, and it is worth nothing until the 100x is measured.

The slab is charged BEFORE the KV pool is sized (`Engine._moe_prefill_stage_bytes`), like every
other post-load reservation. A buffer this size allocated after sizing is precisely how the
2026-09-15 21:01 OOM happened.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Tuple

import torch

from minisgl.utils import init_logger

from .granule import per_expert_tensors, spec_for_container

logger = init_logger(__name__)


def _say(msg: str) -> None:
    """info_rank0 that cannot itself raise — see the note in `PrefillStager.resolve`."""
    try:
        logger.info_rank0(msg)
    except Exception:  # noqa: BLE001
        pass

_STAGER: "PrefillStager | None" = None


def container_bindings(container: Any, num_experts: int) -> List[Tuple[Tuple[str, ...], torch.Tensor]]:
    """`(every_name_bound_to_it, tensor)` for each stacked tensor a kernel reads off `container`.

    NAMES, PLURAL, AND THAT IS THE POINT. A granule spec reports an ALIASED storage as ONE component
    carrying several names (`_GroupedFP8Experts` exposes one buffer as both `weight` and `_w_op`);
    `stacked_tensors` keys by the component's primary name only. A walk that took those keys would
    rebind `weight` to the slab and leave `_w_op` pointing at the host arena — half of the layer
    staged, no error raised, and the GEMM reading whichever name its format happens to use. Mirrors
    `MoEWeightSeam._arena_bindings` (moe_interpose ~:941) exactly, including the REPLICATED entries:
    those are read by the same kernels and live in the same arena, so staging only the per-expert
    components would leave a shared scale on the host.

    Note the non-contiguity refusal in `build_twin` is defence in depth, not the first line: the
    spec derivation below raises `GranuleError` on a strided component before this returns, and it
    does so at BIND time for any real container, so a bound seam cannot carry one.
    """
    spec = spec_for_container(container, num_experts)
    stacked = per_expert_tensors(container, num_experts)
    out: List[Tuple[Tuple[str, ...], torch.Tensor]] = []
    for c in spec.components:
        out.append((tuple(c.names), stacked[c.name]))
    for r in getattr(spec, "replicated", ()):  # absent on specs that declare none
        names = (r.name,) + tuple(getattr(r, "aliases", ()) or ())
        obj: Any = container
        for part in r.name.split("."):
            obj = getattr(obj, part)
        out.append((names, obj))
    return out


class _Twin:
    """A staged stand-in for one seam's (w13, w2) pair, plus the copies that refill it."""

    __slots__ = ("w13", "w2", "copies", "nbytes")

    def __init__(self, w13: Any, w2: Any, copies: List[Tuple[torch.Tensor, torch.Tensor]],
                 nbytes: int) -> None:
        self.w13 = w13
        self.w2 = w2
        self.copies = copies
        self.nbytes = nbytes

    def refill(self) -> None:
        """Issue this layer's H2D copies on the CURRENT (compute) stream. No sync — see the module
        docstring: stream order is what makes one slab safe."""
        for dst, src in self.copies:
            dst.copy_(src, non_blocking=True)


class PrefillStager:
    """One device slab, reused by every host-resident layer, refilled per prefill launch."""

    def __init__(self, slab_bytes: int, device: torch.device) -> None:
        self._slab = torch.empty(slab_bytes, dtype=torch.uint8, device=device)
        self._bytes = slab_bytes
        self.device = device
        self.staged_launches = 0
        self.staged_bytes = 0

    @property
    def slab_bytes(self) -> int:
        return self._bytes

    # -- twin construction (once per seam, at its first staged launch) --------------------------
    def _carve(self, cursor: int, like: torch.Tensor) -> "tuple[int, torch.Tensor]":
        """A slab view with `like`'s dtype and shape. 256-B aligned, because a misaligned view of a
        packed int4/e8m0 storage would be a legal tensor that the kernel reads at the wrong offset."""
        nb = like.numel() * like.element_size()
        start = (cursor + 255) // 256 * 256
        end = start + nb
        if end > self._bytes:
            raise ValueError(f"prefill stage slab too small: need >= {end} B, have {self._bytes}")
        view = self._slab[start:end].view(like.dtype).view(like.shape)
        return end, view

    def build_twin(self, seam: Any, w13: Any, w2: Any) -> "_Twin | None":
        """Shallow-copy both containers with every tensor rebound to a slab view, or None if this
        seam cannot be staged safely.

        REFUSALS ARE DELIBERATE AND SILENT-SAFE. A component whose dotted name is nested
        (`a.b.weight`) or subscripted cannot be rebound on a SHALLOW copy without mutating the
        nested object the original container still points at — that would corrupt the real weights,
        not merely skip the optimisation. A non-contiguous source is refused for the same class of
        reason: the slab view is contiguous, so the copy would silently re-lay-out storage the
        kernel decodes positionally. Either way the seam falls back to today's in-place host read,
        which is correct and merely slow, and the refusal is logged once.
        """
        cursor = 0
        copies: List[Tuple[torch.Tensor, torch.Tensor]] = []
        twins: List[Any] = []
        for container in (w13, w2):
            bindings = container_bindings(container, seam.num_experts)
            # Alias-preserving: every name on one storage lands on ONE slab view, or the twin would
            # hold two copies where the original holds one and whichever name the format reads would
            # be the stale one.
            by_storage: Dict[int, torch.Tensor] = {}
            staged: Dict[str, torch.Tensor] = {}
            for names, t in bindings:
                if not isinstance(t, torch.Tensor):
                    continue
                name = names[0]
                if any(("." in n) or ("[" in n) for n in names):
                    _say(f"prefill stage: seam {seam.path!r} not stageable — component {name!r} "
                         f"is nested/subscripted, so rebinding it would mutate the live container. "
                         f"This layer keeps the in-place host read.")
                    return None
                if not t.is_contiguous():
                    _say(f"prefill stage: seam {seam.path!r} not stageable — component {name!r} "
                         f"is non-contiguous. This layer keeps the in-place host read.")
                    return None
                key = t.data_ptr()
                view = by_storage.get(key)
                if view is None:
                    cursor, view = self._carve(cursor, t)
                    by_storage[key] = view
                    copies.append((view, t))
                for n in names:
                    staged[n] = view
            if isinstance(container, torch.Tensor):
                # Bare stacked-tensor container: the twin IS the staged tensor, there is no object.
                twins.append(next(iter(staged.values())))
            else:
                twin = copy.copy(container)
                for name, view in staged.items():
                    setattr(twin, name, view)
                twins.append(twin)
        return _Twin(twins[0], twins[1], copies, cursor)

    def resolve(self, seam: Any, w13: Any, w2: Any) -> "tuple[Any, Any] | None":
        """Refill the slab for this seam and return its staged pair, or None to fall back."""
        twin = getattr(seam, "_prefill_twin", False)
        if twin is False:
            try:
                twin = self.build_twin(seam, w13, w2)
            except Exception as e:  # noqa: BLE001
                # The contract of this whole path is "fall back to the in-place read, correctly".
                # The log is a courtesy and must not be able to break that — `warning_rank0` needs
                # TP info, which a standalone/offline caller may not have set, and a logging failure
                # here would turn a graceful refusal into a dead forward.
                try:
                    logger.warning_rank0(
                        f"prefill stage: seam {seam.path!r} could not be staged ({e!r}); keeping "
                        f"the in-place host read for it."
                    )
                except Exception:  # noqa: BLE001
                    pass
                twin = None
            seam._prefill_twin = twin
        if twin is None:
            return None
        twin.refill()
        self.staged_launches += 1
        self.staged_bytes += twin.nbytes
        return twin.w13, twin.w2


def install(slab_bytes: int, device: torch.device) -> "PrefillStager | None":
    """Allocate the shared slab. Called once at boot, AFTER the KV pool has been sized against the
    same byte count (`Engine._moe_prefill_stage_bytes`)."""
    global _STAGER
    if slab_bytes <= 0:
        _STAGER = None
        return None
    _STAGER = PrefillStager(slab_bytes, device)
    logger.info_rank0(
        f"MoE prefill expert staging ENABLED: {slab_bytes / (1 << 30):.3f} GiB device slab, "
        f"one layer at a time, copies issued on the compute stream. Engages only when a launch "
        f"would touch ~every expert (num_tokens x top_k >= num_experts) — decode is untouched."
    )
    return _STAGER


def get() -> "PrefillStager | None":
    return _STAGER


def reset() -> None:
    """Drop the slab (tests, and any teardown that must not leak a device allocation)."""
    global _STAGER
    _STAGER = None


def per_layer_host_bytes(model: Any) -> int:
    """The largest single host-resident MoE layer, in bytes — the slab size the stager needs.

    Walked from the LIVE model rather than derived from the plan's totals, for the same reason the
    seam proof is: the plan describes what was intended, the seams describe what the forward will
    actually read.
    """
    from .moe_interpose import iter_seams
    from .stacks import StackKind

    worst = 0
    for seam in iter_seams(model):
        if not getattr(seam, "bound", False) or seam.kind is not StackKind.HOST:
            continue
        if getattr(seam, "computes_on_cpu", False):
            continue
        try:
            w13, w2 = seam._w13, seam._w2
            total = 0
            for container in (w13, w2):
                seen: set[int] = set()
                for _names, t in container_bindings(container, seam.num_experts):
                    if not isinstance(t, torch.Tensor) or t.data_ptr() in seen:
                        continue  # aliases are carved once, so they are billed once
                    seen.add(t.data_ptr())
                    total += (t.numel() * t.element_size() + 255) // 256 * 256
            worst = max(worst, total)
        except Exception:  # noqa: BLE001
            continue
    return worst
