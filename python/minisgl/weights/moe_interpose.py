"""The MoE interposition seam: substituting host-resident containers at the w13/w2 selection point.

THE SEAM
    `MoELayer.forward` selects the two expert containers in one line:

        w13, w2 = self.gate_up_proj, self.down_proj

    Seven model families share that one `MoELayer` (qwen2_moe, qwen3_5_moe, gemma4, glm4_moe_lite,
    models/utils.py, laguna, zaya) and every quant scheme's `apply()` takes the pair AS ARGUMENTS
    rather than reading it off `self`, so anything landed at this line is general by construction
    and format-agnostic by construction. That is why the seam is here and nowhere else.

WHAT THE SUBSTITUTION ACTUALLY IS, AND WHY THE FORWARD DOES ALMOST NOTHING
    Under the LAYER-GRANULAR plan (see `placement.py` for the P2prime measurement that forced it) a
    layer is entirely device-resident or entirely host-resident. So the substitution happens ONCE,
    at bind time: every component tensor of a host-resident layer is reallocated on the host stack,
    copied, and rebound in place. `resolve()` in the forward is then a two-pointer identity check.

    That is the whole point of the design. Residency is a **placement** decision, frozen before the
    first forward, not a **scheduling** one:

      * addresses are constants by the time capture runs, so graph-capture legality is *vacuous*
        rather than argued — no side stream, no publish fence, no `wait_event`, no host sync, and
        nothing that could differ between a captured replay and an eager warmup;
      * there is no miss path, no mask-to-zero, no sentinel and no fallback, so the entire
        silent-wrong-numbers class every reviewer flagged is unrepresentable;
      * a "miss" is just a correct read at PCIe speed.

    P5b confirmed the mechanism end-to-end through torch on both cards: the arena survives
    `torch.cuda.empty_cache()` (which `engine/graph.py:314` calls, and which
    `torch.cuda.graph.__enter__` calls again) with the pointer stable and contents intact, 24 graph
    replays produced one distinct sha256 matching a CPU ground truth, and 25 varying-input replays
    were byte-exact — i.e. each replay genuinely re-reads the host pages.

WHY `resolve()` STILL EXISTS IF IT IS AN IDENTITY
    Two reasons, and neither is speculative.

    1. It is the **only** place a seam/layer mismatch can be caught. Layer identity is a structural
       dotted path, but the MTP draft head builds its own `MoELayer` (`moe.py`), so a binder bug or
       a path collision would bind seam A's ledger to layer B — and because both layers have
       identically-shaped containers, the result would be plausible logits and no crash. Two `is`
       comparisons per layer per forward turn that into a loud `RuntimeError`. At bs=1 over ~48
       layers this is ~10 us against a ~45 ms step (0.02%), and under capture it runs at capture
       time only.
    2. It is the single documented extension point if a mixed layer is ever built. `resolve()` is
       where a two-stack container proxy plus a device-side selector table would be returned. It is
       NOT built: P2prime measured the per-expert gain at 1.013x at the reachable operating point.
       `stacks.ExpertStackTable` carries the exact HIP-side change that would be required.

THE THIRD TIER, AND THE ONE PLACE IT BREAKS THE PARAGRAPH ABOVE
    `StackKind.CPU` is a layer whose expert MLP is executed by AVX-512 cores on the host instead of
    by a GPU kernel. Almost all of the design above carries over unchanged — it is still a
    bind-time reallocation, still the same `_bake` with the same bitwise read-back, still frozen
    before the first forward — and the destination is simply pageable `torch.empty(device="cpu")`
    instead of an arena row, which is the whole capacity argument (no `hipHostMalloc`, no
    device-visible mapping, no pinned page, nothing charged against
    `OffloadPrior.host_arena_ceiling_bytes`).

    What does NOT carry over is the capture claim. A CPU-tier layer's forward contains a HOST CALL,
    and a host call is not a capturable node — so "graph-capture legality is vacuous" is true of the
    DEVICE and HOST tiers and false of this one. `engine/graph.py` captures the whole model forward
    into one `CUDAGraph` per batch-size bucket; a plan with K CPU layers needs K+1 separately
    captured device segments (the cut is INSIDE the layer: attention, norms and the router are all
    still GPU work) and there is no segmented-replay path today. `cpu_tier.graph_segments` counts
    them and `CpuTierMode.is_capturable` is False for every mode that has a CPU layer in it. That
    is stated here, at the seam, because this docstring is where the capture argument is made.

    `resolve()` REFUSES a CPU seam rather than returning its containers: they are pageable host
    memory with no device mapping, and handing them to the grouped kernel is the one way this tier
    could produce a fault instead of a number. `MoELayer.forward` must branch on `computes_on_cpu`
    first. The worker is attached inside `bind_plan` (before `freeze()`, which is rule R1), never
    by the caller afterwards.

WHAT THIS MODULE DOES NOT DO
    It does not allocate host pages (that is the arena behind `StackAllocator`), it does not size
    the device tier (`placement.plan_layer_granular`), it does not do VRAM accounting
    (`accounting.py`), and it does not wire itself into `Engine` (M1-B). It is the seam and the
    binder, and it is unit-testable end to end on CPU tensors with no GPU.
"""

from __future__ import annotations

import random
import weakref
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

import torch
from minisgl._hip_engage import engaged
from minisgl.utils import init_logger

from .granule import (
    GranuleError,
    GranuleSpec,
    _lookup,
    assert_granule_pair_consistent,
    offload_refusal,
    spec_for_container,
)
from .placement import LayerWeights, OffloadPlan, PlacementError, ep_local_top_k
from .stacks import ExpertStackTable, StackAllocator, StackKind

__all__ = [
    "MoEWeightSeam",
    "SeamBindReport",
    "SeamResidencyProof",
    "BindOutcome",
    "InterpositionError",
    "prove_seam_residency",
    "discover_moe_layers",
    "build_layer_weights",
    "attach_seams",
    "bind_plan",
    "ep_size_of",
    "select_selftest_indices",
    "detach_seams",
    "iter_seams",
    "seam_summary",
]

_logger = init_logger("weight-offload")

# How many times each placement arm has been RESOLVED, keyed by the same name the ledger uses.
#
# The `engaged()` ledger is a SET: it records that an arm fired once, ever. That is the right shape
# for "did the seam ever engage" and the wrong shape for a per-leg A/B, where both legs run after the
# ledger is already saturated and a set-diff is empty by construction for BOTH of them. This counter
# is what makes the per-leg question answerable, and it answers it in a direction that is easy to
# misread, so: under CUDAGRAPH REPLAY these counts DO NOT MOVE. `resolve()` is host Python inside
# `MoELayer.forward`; a replay re-executes the recorded kernel launches and never re-enters Python,
# so the captured leg reads its host-resident experts through pointers the graph baked at capture
# time. Counts frozen across a captured leg and climbing by (num MoE layers) per step across an eager
# leg is therefore the positive signal that the two legs took different paths — not evidence that the
# offload arm stopped working. (`bake.verify_after_capture()` is the separate gate that those baked
# pointers still address the arena.) One dict update per MoE layer per eager forward.
RESOLVE_COUNTS: dict[str, int] = {}

# The container attributes `MoELayer.forward` selects, in the order it selects them. Read off
# `MoELayer.expert_containers()` when available so the seam and the layer can never disagree about
# which attributes hold weights; the tuple is only the fallback for a duck-typed test double.
#
# NOTHING ELSE IN THIS MODULE MAY NAME THESE STRINGS. `freeze()` and `attach_seams` used to read
# `layer.gate_up_proj` / `specs["gate_up_proj"]` directly, which made the honest answer to "what
# must a new model or quant format implement?" be "it must call its two containers gate_up_proj and
# down_proj" — and made the freeze-time rebind check compare the wrong pair (silently, for a layer
# whose declared order differs) or raise `AttributeError` (loudly, for one that renamed them). Every
# access now goes through `_container_attrs` / `expert_containers()` / `granule_specs()`.
_DEFAULT_CONTAINER_ATTRS = ("gate_up_proj", "down_proj")


def _container_attrs(layer: Any) -> tuple[str, ...]:
    fn = getattr(layer, "expert_containers", None)
    if callable(fn):
        return tuple(fn().keys())
    return _DEFAULT_CONTAINER_ATTRS


def ep_size_of(layer: Any) -> int:
    """This layer's EFFECTIVE expert-parallel size — 1 unless it is genuinely EP-sharded.

    All THREE conjuncts `MoELayer.__init__` computes, asked of the built layer rather than of the
    config: the engine toggle, the quant method's EP veto (`supports_ep` is False for RXF and for
    unquantized experts) and `force_no_ep` (the MTP draft head). `MoELayer` collapses them into
    `enable_ep`, so reading `ep_size` WITHOUT `enable_ep` reports the process-wide EP size for a
    layer that is fully replicated — which would halve that layer's traffic figure on a
    `--enable-ep --tp 2` serve of a checkpoint whose method vetoed EP.

    Duck-typed with a default of 1 so a container-only test double, and any future layer type that
    is never EP-sharded, need declare nothing.
    """
    if not bool(getattr(layer, "enable_ep", False)):
        return 1
    return max(1, int(getattr(layer, "ep_size", 1) or 1))


# How many COMPONENTS the bake reads back and byte-compares after populating (A1.4). 64 is the
# plan's number: enough that a systematic layout error (a wrong stride, an off-by-one expert, a
# component swapped between w13 and w2) is caught with certainty, cheap enough to run on every boot
# rather than under a flag.
SELFTEST_SAMPLE = 64

# Fixed seed so EVERY TP RANK VERIFIES THE SAME COMPONENTS. Ranks run in lockstep; a rank-dependent
# sample would make one rank's boot slower for no reason and, worse, would make a real failure look
# intermittent because only whichever rank drew the bad index would report it.
SELFTEST_SEED = 0x5EED

# Bytes compared per `torch.equal` call in the read-back self-test. NOT a tuning knob — it is the
# bound on the DEVICE scratch the comparison allocates (`eq` materialises one bool per byte), taken
# at the peak-VRAM instant of the whole boot: inside `_bake`, after post_load() and before any
# original is dropped. 32 MiB is far below anything that could matter on a 16 GB card and far above
# the point where per-call launch overhead is visible against a PCIe read. See `_bitwise_equal`.
SELFTEST_COMPARE_CHUNK_BYTES = 32 << 20


def select_selftest_indices(
    n_items: int, k: int = SELFTEST_SAMPLE, seed: int = SELFTEST_SEED
) -> list[int]:
    """Deterministic, rank-invariant sample of `min(k, n_items)` item indices, SORTED.

    Sorted so the read-back walks the arena forward (items are built in placement order), which
    keeps verification a sequential read rather than 64 random PCIe round trips.
    """
    if n_items <= 0:
        return []
    if k >= n_items:
        return list(range(n_items))
    return sorted(random.Random(seed).sample(range(n_items), k))


class InterpositionError(RuntimeError):
    """The seam cannot be established or is inconsistent. Always a boot failure."""


# =====================================================================================
# Discovery — structural paths, never a construction counter
# =====================================================================================

_OP_WALK_SKIP = ("_parameters", "_buffers", "_modules", "_non_persistent_buffers_set")


def _iter_ops(obj: Any, prefix: str, seen: set[int]) -> Iterator[tuple[str, Any]]:
    """Yield `(dotted_path, node)` for every op-like object reachable from `obj`.

    THE PATH GRAMMAR IS `BaseOP.state_dict`'s, NOT the tensor walker's. A layer path is only useful
    if it is the SAME string the plan resolver produces, and `weights/plan.py::moe_layer_shapes`
    builds `model.layers.{lid}.mlp.experts` — the CHECKPOINT namespace, i.e. exactly the prefix
    `BaseOP.state_dict` / `OPList.state_dict` emit. So `OPList` numbers its members
    `<prefix>.<i>`; it must NOT go through the generic list rule, which would produce
    `model.layers.op_list[0].mlp.experts` and make `bind_plan`'s plan-vs-model comparison fail on
    every real model (the OPList is how every family in this repo holds its decoder layers).

    Everything else mirrors `granule._iter_tensors`' child rules (BaseOP `__dict__`, `nn.Module`
    children, plain list/tuple/dict members with `[i]` syntax). Those shapes never appear in a
    checkpoint key, so no plan path can name them; they exist so the walk still REACHES a layer
    parked somewhere unusual rather than silently missing it.
    """
    # local: avoid a layers <-> weights import cycle
    from minisgl.layers.base import BaseOP, OPList

    if id(obj) in seen:
        return
    seen.add(id(obj))
    if not isinstance(obj, (BaseOP, torch.nn.Module)):
        return
    yield prefix, obj

    if isinstance(obj, OPList):
        # `OPList.state_dict` does `_concat_prefix(prefix, str(i))`. Reproduce that exactly.
        for i, op in enumerate(obj.op_list):
            yield from _iter_ops(op, f"{prefix}.{i}" if prefix else str(i), seen)
        return

    def child(name: str, value: Any) -> Iterator[tuple[str, Any]]:
        path = f"{prefix}.{name}" if prefix else name
        if isinstance(value, (BaseOP, torch.nn.Module)):
            yield from _iter_ops(value, path, seen)
        elif isinstance(value, (list, tuple)):
            for i, v in enumerate(value):
                yield from child(f"{name}[{i}]", v)
        elif isinstance(value, dict):
            for k, v in value.items():
                yield from child(f"{name}[{k}]", v)

    if isinstance(obj, torch.nn.Module):
        for name, value in list(obj.__dict__.items()):
            if name in _OP_WALK_SKIP:
                continue
            yield from child(name, value)
        for name, m in obj.named_children():
            yield from _iter_ops(m, f"{prefix}.{name}" if prefix else name, seen)
        return
    for name, value in list(vars(obj).items()):
        yield from child(name, value)


def discover_moe_layers(root: Any, *, prefix: str = "") -> list[tuple[str, Any]]:
    """Every `MoELayer` under `root`, keyed by its STRUCTURAL dotted path.

    Structural, never a construction counter: the MTP draft head builds its own `MoELayer`
    (`layers/moe.py`), so a counter renumbers every layer after it and a plan keyed by that counter
    would silently place the wrong layers on the wrong stack. The path is also what two TP ranks
    compare, so it must be a function of the module tree and nothing else.
    """
    from minisgl.layers.moe import MoELayer  # local: avoid an import cycle at module load

    found = [(p, o) for p, o in _iter_ops(root, prefix, set()) if isinstance(o, MoELayer)]
    paths = [p for p, _ in found]
    if len(set(paths)) != len(paths):
        dupes = sorted({p for p in paths if paths.count(p) > 1})
        raise InterpositionError(f"structural MoE paths are not unique: {dupes}")
    return found


# =====================================================================================
# The seam
# =====================================================================================


@dataclass
class SeamBindReport:
    """One layer's bake outcome. Summed across layers by `BindOutcome`."""

    path: str
    kind: StackKind
    moved_bytes: int = 0
    components: int = 0
    verified_components: int = 0
    verified_bytes: int = 0

    def describe(self) -> str:
        return (
            f"{self.path}: {self.kind.name} moved={self.moved_bytes / (1 << 20):.1f}MiB "
            f"components={self.components} read-back-verified={self.verified_components}"
        )


class MoEWeightSeam:
    """The interposition object bound to one `MoELayer` as `layer._weight_offload`.

    Immutable after `bind()`. There is deliberately no unbind, no re-place and no per-step mutator:
    the whole safety argument of this design is that nothing about residency can change after boot.
    """

    __slots__ = (
        "path",
        "_layer",
        "num_experts",
        "top_k",
        "top_k_local",
        "ep_size",
        "w13_spec",
        "w2_spec",
        "_attrs",
        "_table",
        "_w13",
        "_w2",
        "_bound",
        "_frozen",
        "_report",
        # The `engaged()` name for this seam's placement, precomputed at bind(). Not derived in
        # `resolve()`/`cpu_forward()`: those are once per MoE layer per forward step.
        "_engage",
        # CPU-COMPUTE TIER (StackKind.CPU). None on every device/host seam, so a two-tier serve
        # stores one extra None per layer and pays one extra `is not None` per CPU forward — and
        # nothing at all per device/host forward, because `MoELayer.forward` reaches
        # `computes_on_cpu` only inside the existing `_weight_offload is not None` branch.
        "_cpu_worker",
        "_cpu_expert_offset",
    )

    def __init__(
        self,
        path: str,
        layer: Any,
        *,
        w13_spec: GranuleSpec,
        w2_spec: GranuleSpec,
    ) -> None:
        assert_granule_pair_consistent(w13_spec, w2_spec, where=path)
        self.path = path
        self._layer = layer
        self.num_experts = int(getattr(layer, "local_num_experts"))
        # GLOBAL top_k, exactly as the layer holds it — kept for reporting.
        self.top_k = int(getattr(layer, "top_k"))
        self.ep_size = ep_size_of(layer)
        # TRAFFIC top_k: the routed slots THIS rank computes per token. `num_experts` above is the
        # EP-LOCAL count, so pairing it with the GLOBAL top_k over-counts this layer's host traffic
        # by ep_size — see `placement.ep_local_top_k`, which is the one implementation and is also
        # what `plan.size_planned_layers_from_model` calls. The two used to disagree: the engine
        # path corrected for EP and this one did not, so `build_layer_weights(attach_seams(model))`
        # and the engine's own planner produced different `LayerWeights`, different projections and
        # a different `OffloadPlan.digest()` for one model on one rank.
        self.top_k_local = ep_local_top_k(self.top_k, self.ep_size, self.num_experts)
        self.w13_spec = w13_spec
        self.w2_spec = w2_spec
        # The attribute names the two specs were derived from, in `forward`'s selection order. Read
        # off the layer rather than hardcoded so the seam's identity check, the bake's spec/container
        # pairing and `MoELayer.forward` cannot disagree about which attributes hold weights.
        self._attrs = _container_attrs(layer)
        if len(self._attrs) != 2:
            raise InterpositionError(
                f"seam {path!r}: expected exactly two expert containers (w13, w2), got "
                f"{self._attrs}."
            )
        self._table = ExpertStackTable.uniform(self.num_experts, StackKind.DEVICE)
        self._w13 = getattr(layer, self._attrs[0])
        self._w2 = getattr(layer, self._attrs[1])
        self._bound = False
        self._frozen = False
        self._report = SeamBindReport(path=path, kind=StackKind.DEVICE)
        # "unbound" until bind() decides — a seam that reaches a forward without being bound is a
        # binder bug, and it has to be visible in the ledger as its own name rather than silently
        # reading as the device arm.
        self._engage = "weight_offload.moe_resolve[unbound]"
        self._cpu_worker = None
        self._cpu_expert_offset = 0

    # -- hot path ------------------------------------------------------------------------------
    def resolve(self, w13: Any, w2: Any) -> tuple[Any, Any]:
        """Return the containers this layer's kernels must read. Called once per MoE forward.

        Under layer-granular placement this is an identity plus two `is` checks — the substitution
        already happened at `bind()`. The checks are not decoration: they are the only thing that
        turns a seam-bound-to-the-wrong-layer bug (identical container shapes, plausible logits, no
        crash) into a loud failure. See the module docstring.

        REFUSES a CPU-tier seam. A `StackKind.CPU` layer's expert weights are consumed by AVX-512
        cores on the host and are never read by a GPU kernel; a `resolve()` that handed them back
        would let the grouped kernel read pageable, un-mapped host memory. The caller
        (`MoELayer.forward`) must branch on `computes_on_cpu` BEFORE resolving. The refusal lives
        HERE and not in `assert_identity` because `assert_identity` is also the boot proof's call
        (`prove_seam_residency`), and a CPU layer's containers still have to pass the
        seam-bound-to-the-wrong-layer check — refusing there would delete that check for the tier
        that holds the most bytes, which is the opposite of what either branch wanted.

        THE `engaged()` LINE. Boot-time logs prove the arena was PINNED and the bake COPIED; neither
        proves the forward ever goes through the seam. `MoELayer._weight_offload` is a class
        attribute defaulting to None, so any regression that leaves it unset — a detach, a layer
        rebuilt after bind, a model whose sparse block stopped being a `MoELayer` — makes the whole
        offload arm vanish with the boot banner unchanged and the serve merely reading device
        weights (or, at 48 layers, OOMing later for reasons that look unrelated). This repo requires
        diffing the engaged ledger per leg precisely because that class of dispatch regression is
        invisible to a bench. Fires once per (kind), from the first real forward AND from inside HIP
        graph capture; the name is precomputed at bind so the hot path is one set lookup.
        """
        if self._table.is_uniform and self._table.uniform_kind is StackKind.CPU:
            raise InterpositionError(
                f"seam {self.path!r} is CPU-COMPUTE tier: its expert weights live in PAGEABLE host "
                f"memory with no device mapping, so no GPU kernel may read them. `MoELayer.forward` "
                f"must test `seam.computes_on_cpu` before calling `resolve()`. Reaching here means "
                f"the forward has a two-tier assumption baked in and would hand the grouped kernel "
                f"a host pointer the device cannot dereference."
            )
        engaged(self._engage)
        RESOLVE_COUNTS[self._engage] = RESOLVE_COUNTS.get(self._engage, 0) + 1
        return self.assert_identity(w13, w2)

    def assert_identity(self, w13: Any, w2: Any) -> tuple[Any, Any]:
        """`resolve()` WITHOUT the ledger line. The check, on its own.

        Split out for `prove_seam_residency`, which has to make exactly the call the forward makes
        (that is the whole point of it) but MUST NOT publish `weight_offload.moe_resolve[host]` while
        doing so. That line's only value is that it distinguishes "the bake ran" from "a forward read
        the arena"; a boot-time proof that emitted it would make the two indistinguishable again and
        quietly destroy the evidence the ledger exists to provide.

        Deliberately does NOT refuse a CPU-tier seam, unlike `resolve()`. The identity check is
        exactly as load-bearing for a CPU layer as for any other — more so, since a CPU layer's
        containers are pageable host tensors that no device fault would ever catch — and this is
        the only place it runs at boot.
        """
        if w13 is not self._w13 or w2 is not self._w2:
            raise InterpositionError(
                f"weight-offload seam {self.path!r} is bound to different containers than the "
                f"layer is holding. The seam's residency ledger describes weights this forward is "
                f"not reading, so its placement, byte accounting and any host residency are all "
                f"about the wrong tensors. This is a binder bug (a colliding structural path, or a "
                f"container rebound after freeze()), never something to work around."
            )
        return w13, w2

    # -- queries -------------------------------------------------------------------------------
    @property
    def bound(self) -> bool:
        return self._bound

    @property
    def frozen(self) -> bool:
        return self._frozen

    @property
    def table(self) -> ExpertStackTable:
        """The residency ledger for this layer. Uniform under every plan this module produces."""
        return self._table

    @property
    def kind(self) -> StackKind:
        return self._table.uniform_kind

    @property
    def computes_on_cpu(self) -> bool:
        """Is this layer's expert MLP executed by host CPU cores instead of a GPU kernel?

        The ONE test `MoELayer.forward` makes before `resolve()`. A property rather than a
        `kind is StackKind.CPU` comparison at the call site so the hot path has a single
        attribute load and so the two-tier assumption cannot be re-introduced by a caller that
        compares against `StackKind.HOST` and treats "not HOST" as "device".
        """
        return self._cpu_worker is not None

    @property
    def cpu_worker(self):
        """The `cpu_worker.CpuMoEWorker` this layer submits to, or None."""
        return self._cpu_worker

    def attach_cpu_worker(self, worker: Any, *, backend_expert_offset: int = 0) -> None:
        """Wire the CPU executor for a CPU-tier seam. Boot-time only, like every other placement act.

        Separate from `bind()` because the worker is PROCESS-wide (one thread pool serves all CPU
        layers — the parallelism lives inside the native call, and a pool per layer would be 21
        competing schedulers) while `bind()` is per-layer. `backend_expert_offset` is this layer's
        base index into the worker's packed table.
        """
        if self._frozen:
            raise InterpositionError(
                f"seam {self.path!r} is frozen: the CPU worker must be attached before boot ends"
            )
        if not (self._table.is_uniform and self._table.uniform_kind is StackKind.CPU):
            raise InterpositionError(
                f"seam {self.path!r} is {self.kind.name}, not CPU: attaching a CPU worker to it "
                f"would create a layer that is computed on the host AND streamed to the device, "
                f"i.e. double-counted in the residual."
            )
        self._cpu_worker = worker
        self._cpu_expert_offset = int(backend_expert_offset)

    # -- CPU-tier hot path ----------------------------------------------------------------------
    def cpu_submit(self, hidden_states: Any, topk_weights: Any, topk_ids: Any):
        """Start this layer's expert MLP on the host. NON-BLOCKING — the GPU proceeds meanwhile.

        Copies the activation and the route to the host, waits for THAT copy (not for the compute),
        and hands the buffers to the worker. Returns a `cpu_tier.CpuHandoff`; the caller must
        `cpu_join` it before the layer's output enters the residual stream.

        WHAT CROSSES PCIe HERE: (M, 2560) bf16 down + (M, top_k) int32 and f32 down, ~5 KB at M=1.
        The 30.7 MB of expert weights a streamed layer would have moved do not move at all — that
        is the entire point of the tier.

        `.cpu()` on a CUDA tensor is a SYNCHRONISING copy, and that synchronisation is required
        rather than incidental: the worker thread reads the resulting host buffer immediately, so
        an async copy whose completion had only been *enqueued* would be read while the DMA was
        still in flight. That is a silent wrong-numbers race, not a crash. It is also the cost that
        makes this a per-layer round trip and therefore the thing a GPU run has to measure — see
        `cpu_tier.CpuTierPrior.handoff_us_bracket`, which is explicitly NOT measured.
        """
        if self._cpu_worker is None:
            raise InterpositionError(
                f"seam {self.path!r}: cpu_submit with no worker attached. Call "
                f"`attach_cpu_worker` at boot; a missing worker must not silently fall through to "
                f"a GPU path that would read pageable host memory."
            )
        x = hidden_states.detach().to("cpu", copy=True)
        w = topk_weights.detach().to("cpu", copy=True)
        i = topk_ids.detach().to("cpu", copy=True)
        return self._cpu_worker.submit(self._cpu_expert_offset, x, i, w)

    def cpu_join(self, handoff, *, like: Any = None):
        """Take the CPU partial back and return it as a device tensor shaped like `like`.

        FIFO and exactly-once — enforced by `cpu_tier.HandoffLedger`, not by this method. A failure
        in the worker is RE-RAISED here; it is never turned into a zero partial, because a MoE
        layer that silently contributed zero produces fluent, plausible, wrong text.
        """
        if self._cpu_worker is None:
            raise InterpositionError(f"seam {self.path!r}: cpu_join with no worker attached")
        out = self._cpu_worker.join(handoff)
        if like is None:
            return out
        import torch

        t = out if isinstance(out, torch.Tensor) else torch.as_tensor(out)
        return t.to(device=like.device, dtype=like.dtype).reshape(like.shape)

    def cpu_forward(self, hidden_states: Any, topk_weights: Any, topk_ids: Any):
        """BLOCK mode: submit and immediately join. The GPU idles for the duration.

        This is the SERIAL composition, and it is the default, because it is the one whose win does
        not depend on the unmeasured handoff latency: a CPU layer at the measured 47.45 GB/s (6
        threads, the knee under a live serve) is 0.583 ms against 2.12-2.49 ms for the same layer
        streamed over card 1's Gen4 x8 link. 3.6x-4.3x per layer with no concurrency required.

        The genuinely concurrent composition is `cpu_submit(...)` -> GPU work -> `cpu_join(...)`,
        which only has GPU work to hide behind in SPLIT mode (see `cpu_worker.split_route_for_cpu`
        and `cpu_tier`'s module docstring): the residual stream is sequential, so at BLOCK
        granularity there is no independent GPU work available to overlap with.

        THE `engaged()` LINE for this tier — the CPU arm's equivalent of `resolve()`'s, and it has
        to be here because `resolve()` refuses a CPU seam. Without it the third tier would be the
        one arm of three with no evidence that a forward ever reached it, which is precisely the
        dispatch regression the per-leg ledger diff is meant to expose: a plan that placed 21
        layers on the CPU and a forward that never called them look identical in the boot banner.
        """
        engaged(self._engage)
        RESOLVE_COUNTS[self._engage] = RESOLVE_COUNTS.get(self._engage, 0) + 1
        return self.cpu_join(self.cpu_submit(hidden_states, topk_weights, topk_ids),
                             like=hidden_states)

    @property
    def report(self) -> SeamBindReport:
        return self._report

    def offload_refusal(self) -> str | None:
        """Why this layer cannot be host-resident, or None. Asked of the CONTAINERS, first wins.

        Public because the PLANNER has to ask it too, not just the bake. `build_layer_weights`
        excludes a refusing seam so the plan never proposes host residency the bake would then have
        to refuse — the same filter `plan.size_planned_layers_from_model` applies. Asking it in
        exactly one place, of exactly the objects `expert_containers()` yields, is what keeps this
        format-agnostic: the format declares its own refusal
        (`_GroupedFP8Experts.offload_refusal` names MINISGL_ZAYA_OLDMOE / MINISGL_ZAYA_W8A16) and
        nothing here holds a table of formats or knob names that could go stale.
        """
        for attr in _container_attrs(self._layer):
            why = offload_refusal(getattr(self._layer, attr))
            if why is not None:
                return f"{attr}: {why}"
        return None

    def layer_weights(self, *, priority: int = 0) -> LayerWeights:
        """This layer as a placement unit for `placement.plan_layer_granular`.

        `top_k=self.top_k_local`, NOT the layer's global `top_k`: `num_experts` here is the EP-LOCAL
        shard, and `LayerWeights` pairs the two to price traffic. See `placement.ep_local_top_k`.
        """
        return LayerWeights.from_specs(
            self.path,
            num_experts=self.num_experts,
            top_k=self.top_k_local,
            w13=self.w13_spec,
            w2=self.w2_spec,
            priority=priority,
        )

    # -- boot ----------------------------------------------------------------------------------
    def bind(
        self,
        kind: StackKind,
        allocator: StackAllocator | None = None,
        *,
        selftest: int = SELFTEST_SAMPLE,
    ) -> SeamBindReport:
        """Place this layer's weights on `kind`'s stack and re-point the seam at the result.

        `StackKind.DEVICE` moves nothing — the weights are already where `load_state_dict` put
        them — so the device arm costs zero bytes and zero copies. That is a correctness
        requirement, not an optimisation: allocating a fresh device stack and copying into it would
        hold both copies live at once, and at f=0.20 on a 16 GB card that transient is several GB
        the card does not have. It also makes device layers bit-identical to a non-offloaded serve
        by construction, which removes them from every numerics gate.

        `StackKind.HOST` runs the Stage-A bake for this layer, in this order and no other:

          1. refuse everything refusable BEFORE a page is committed, so a bad plan never leaves a
             half-populated arena — which at inference is indistinguishable from a correct one;
          2. enumerate the copies (components AND replicated tensors) and validate every one before
             moving a byte;
          3. issue all the copies, then ONE `synchronize()` for the lot — at tens of GB and
             ~14.5 GB/s per rank, per-item syncs would serialise seconds of PCIe into stalls;
          4. read back a rank-invariant sample and compare BITWISE while the sources are alive;
          5. rebind every alias of a component together, which is what releases the originals;
          6. prove the originals are gone with weakrefs, naming any alias that still points at the
             device copy.

        Step 4 is not optional and it is not a return-code check. Phase 0 produced FOUR independent
        cases of this driver reporting success over wrong state: `location=Host` silently allocating
        VRAM with the property echoed back verbatim; VMM over-commit returning `hipSuccess` on
        create/map/setAccess and faulting only at first touch; `hipMemUnmap`->`hipMemMap` serving
        the stale page with no non-zero return code anywhere; and `expandable_segments` +
        `empty_cache` handing back zeroed memory in plain torch. A capability probe can PASS while
        the operation FAILS. Assert on the data.
        """
        if self._frozen:
            raise InterpositionError(
                f"seam {self.path!r} is frozen: no mapping or placement call may run after boot "
                f"(the address stability that makes graph capture vacuous depends on it)"
            )
        if self._bound:
            raise InterpositionError(f"seam {self.path!r} is already bound")
        report = SeamBindReport(path=self.path, kind=kind)
        if kind in (StackKind.HOST, StackKind.CPU):
            if allocator is None:
                raise InterpositionError(
                    f"seam {self.path!r}: {kind.name} placement needs a StackAllocator"
                    + (" with a host arena" if kind is StackKind.HOST else "")
                )
            # The CPU tier runs the SAME bake — same refusal gate, same component enumeration, same
            # bitwise read-back verification, same weakref leak proof. Only the destination differs
            # (`allocator.alloc_like(CPU, ...)` is plain pageable `torch.empty(device="cpu")`, no
            # `hipHostMalloc`, no mapping, no arena). Reusing the path rather than writing a second
            # one is deliberate: Phase 0 produced four separate cases of this driver reporting
            # success over wrong state, and the read-back-and-compare in `_bake` is what catches
            # them. A "the CPU tier is just a torch copy, it cannot fail" shortcut would drop
            # exactly that check on the tier that holds the most bytes.
            self._refuse_before_any_byte_moves()
            items = self._plan_items(allocator, kind=kind)
            # Drop the seam's own handles on the containers first. For a bare stacked tensor the
            # container IS the weight, so `_w13`/`_w2` are live references to the very sources the
            # bake is about to replace — keeping them would pin the device originals and make the
            # leaked-source check report a leak this object is itself causing. They are re-read
            # below, after the rebind.
            self._w13 = self._w2 = None
            _bake(self.path, items, selftest=selftest, report=report)
        self._table = ExpertStackTable.uniform(self.num_experts, kind)
        # Re-read after the bake: the container OBJECTS are unchanged for every quantized format
        # (the bake rebinds attributes on them), but a bare stacked tensor IS the container, so the
        # layer attribute itself moved. Re-reading keeps `resolve()`'s identity check honest either
        # way instead of depending on which of the two happened.
        self._w13 = getattr(self._layer, self._attrs[0])
        self._w2 = getattr(self._layer, self._attrs[1])
        self._bound = True
        # One name per PLACEMENT, not per layer: 48 per-layer lines would drown the boot log and,
        # worse, would make "the host arm is present" a thing you have to count rather than read.
        # Three names, so a run where every layer silently landed on the device still differs from a
        # run where the host arm engaged, and from one where the CPU tier did.
        #
        # The CPU arm gets a DIFFERENT call site in its name, because it has a different call site:
        # `resolve()` REFUSES a CPU seam, so `moe_resolve[cpu]` could never fire and would sit in
        # the ledger as a permanently-absent arm — indistinguishable from the dispatch regression
        # the ledger exists to catch. `MoELayer.forward` reaches this tier through `cpu_forward`.
        self._engage = (
            "weight_offload.moe_cpu_forward[cpu]"
            if kind is StackKind.CPU
            else f"weight_offload.moe_resolve[{kind.name.lower()}]"
        )
        self._report = report
        return report

    def _enumerate_named_tensors(self) -> list[tuple[str, Any, tuple[str, ...], Any]]:
        """`(owner_attr, container, alias_names, canonical_tensor)` for every tensor the layer's
        expert containers CURRENTLY hold — components first, then replicated, in bake order.

        ONE enumeration, two consumers: `_plan_items` (which turns each entry into a copy) and
        `live_tensors` (which turns each entry into a residency claim). They MUST see the same set:
        `live_tensors` exists to prove that what the kernels read is in the arena, and a proof that
        walked a different tensor set from the bake would be a proof about weights the forward does
        not use — vacuous in exactly the direction that reads as a pass. The tensors are re-read off
        `self._layer` on every call, never cached, because after `bind()` the containers hold the
        ARENA tensors and that is precisely the state the proof is about.
        """
        attrs = _container_attrs(self._layer)
        if len(attrs) != 2:
            # `zip` would silently truncate and leave a whole GEMM behind on the device stack while
            # the plan's capacity arithmetic said it had moved.
            raise InterpositionError(
                f"seam {self.path!r}: expected exactly two expert containers (w13, w2), got "
                f"{attrs}. A layer with a different container set needs its own placement unit — "
                f"it cannot borrow this one's granule pair."
            )
        # Pair each container with the spec that was DERIVED FROM IT, by attribute name. The specs
        # were captured at construction from `granule_specs()['gate_up_proj'/'down_proj']`, so a
        # positional `zip` here would silently swap them the day `expert_containers()` yields a
        # different order — and because `assert_granule_pair_consistent` forces the two specs to
        # share a component NAME set, a swap is undetectable downstream.
        specs = {self._attrs[0]: self.w13_spec, self._attrs[1]: self.w2_spec}
        if set(attrs) != set(specs):
            raise InterpositionError(
                f"seam {self.path!r}: the layer now presents containers {attrs} but the seam holds "
                f"granule specs for {tuple(specs)}. The specs describe different tensors than the "
                f"layer is holding."
            )
        out: list[tuple[str, Any, tuple[str, ...], Any]] = []
        for owner_attr in attrs:
            spec = specs[owner_attr]
            container = getattr(self._layer, owner_attr)
            stacked = spec.stacked_tensors(container)
            for c in spec.components:
                out.append((owner_attr, container, tuple(c.names), stacked[c.name]))
            for r in spec.replicated:
                names = (r.name,) + tuple(getattr(r, "aliases", ()) or ())
                out.append((owner_attr, container, names, _lookup(container, r.name)))
        return out

    def live_tensors(self) -> list[tuple[str, Any]]:
        """`(name, tensor)` for every tensor this layer's kernels will actually read, RIGHT NOW.

        Public because the residency PROOF needs it (`prove_seam_residency`): after `bind()` these
        are the arena rows for a HOST layer and the untouched originals for a DEVICE one, so asking
        the arena whether it owns each pointer is a direct answer to "is this layer really streaming
        from host RAM". Only the canonical tensor per alias group is yielded — the aliases are views
        of the same storage, so a second claim about them would double-count the bytes.
        """
        return [
            (f"{self.path}.{owner}.{names[0]}", t)
            for owner, _c, names, t in self._enumerate_named_tensors()
        ]

    def _plan_items(
        self, allocator: StackAllocator, *, kind: StackKind = StackKind.HOST
    ) -> list["_BakeItem"]:
        """Enumerate this layer's copies: every component AND every replicated tensor.

        COMPONENT-granular, not expert-granular. A GRANULE is one expert's slice of every tensor in
        the container — scales and zero-points travel with their weights or expert `e` gets
        dequantized against expert `f`'s scale, which is plausible numbers with no crash — but under
        layer-granular placement the whole container moves at once and each COMPONENT is one
        contiguous slab whose row stride is its own `e * row_bytes`. Copying component-wise
        therefore preserves exactly the addressing the kernels already do, in one `copy_` per
        component instead of one per expert.

        The REPLICATED tensors move too. They are excluded from `granule_bytes` because every expert
        reads the same row of them (one read per layer, not one per routed expert), but they are
        still resident bytes that `LayerWeights.resident_bytes` prices — leaving them behind would
        make the arena's occupancy disagree with the plan's capacity arithmetic by ~3% of w13.

        `kind` selects the DESTINATION stack — `StackKind.HOST` for the pinned arena,
        `StackKind.CPU` for plain pageable `torch.empty`. It is the only thing that differs
        between the two bakes; the enumeration, the refusal gate, the read-back verification and
        the leak proof are shared, which is what keeps the CPU tier under the same evidence
        regime as the host tier.
        """
        return [
            self._item(owner_attr, container, names, src, allocator, kind)
            for owner_attr, container, names, src in self._enumerate_named_tensors()
        ]

    def _item(
        self, owner_attr, container, names, src, allocator, kind: StackKind = StackKind.HOST
    ) -> "_BakeItem":
        """Build one copy item, resolving EVERY alias to its own live tensor.

        The alias's own dtype and shape are load-bearing and are NOT recoverable from the granule
        descriptor (`ExpertComponent` records alias NAMES only). `_GroupedFP8Experts.post_load`
        binds `_w_op = weight.view(uint8)` and `_scales_op = weight_scale.squeeze(-1)` — same bytes,
        different dtype and different shape — so rebinding every name to the canonical view hands
        the kernel a tensor whose dtype/shape are another name's. That is the de-aliasing bug this
        item exists to prevent, wearing a different hat.
        """
        aliases: list[_Alias] = []
        srcs: list[Any] = []
        for n in names:
            t = src if n == names[0] else _lookup(container, n)
            aliases.append(_Alias(name=n, dtype=t.dtype, shape=tuple(t.shape)))
            srcs.append(t)
        return _BakeItem(
            name=f"{self.path}.{owner_attr}.{names[0]}",
            layer=self._layer,
            owner_attr=owner_attr,
            container=container,
            attrs=tuple(aliases),
            src=src,
            alias_srcs=srcs,
            dst=allocator.alloc_like(kind, src),
            nbytes=src.numel() * src.element_size(),
        )

    def freeze(self) -> None:
        """Rule R1: after this, every placement entry point raises. Boot is over.

        This is also the LAST moment before graph capture at which a rebound container can still be
        caught cheaply, so it is checked here rather than left to the first forward. The whole
        capture argument is "addresses are constants by the time capture runs": if anything rebound
        a container attribute between `bind()` and here (a late `post_load`, a second bake, a
        checkpoint-reload path), the captured graph would bake the address of a tensor the seam's
        ledger does not describe, and `resolve()` would only report it once a forward ran — which
        under capture is *after* the graph already holds the pointer.

        The live containers are re-read through `self._attrs` — the names the layer DECLARED via
        `expert_containers()` — and not through literal `layer.gate_up_proj` / `layer.down_proj`.
        Those literals were the last place in this module that required a model to name its
        containers a particular way, and they failed in both directions: `AttributeError` on a layer
        that renamed them, and a SILENT wrong-pair comparison on one whose declared order differs
        from the literal order (`live[0] is not self._w13` would then compare down-proj against the
        gate-up ledger, pass by luck on a layer where both moved, and pass by luck again where
        neither did).
        """
        if not self._bound:
            raise InterpositionError(f"seam {self.path!r} cannot be frozen before it is bound")
        attrs = _container_attrs(self._layer)
        if tuple(attrs) != tuple(self._attrs):
            raise InterpositionError(
                f"seam {self.path!r}: the layer now declares containers {tuple(attrs)} but the seam "
                f"was built against {tuple(self._attrs)}. The ledger, the arena and the captured "
                f"graph would each be about a different set of weights."
            )
        live = tuple(getattr(self._layer, a) for a in self._attrs)
        if live[0] is not self._w13 or live[1] is not self._w2:
            raise InterpositionError(
                f"seam {self.path!r}: the layer's containers were rebound between bind() and "
                f"freeze(). Graph capture is about to bake these addresses in, and the seam's "
                f"residency ledger describes different tensors — so the arena, the byte accounting "
                f"and the captured graph would each be about a different set of weights."
            )
        self._frozen = True

    def attach(self) -> None:
        """Install this seam on the layer so `MoELayer.forward` resolves through it."""
        if self._frozen:
            raise InterpositionError(f"seam {self.path!r} is frozen: cannot re-attach after boot")
        self._layer._weight_offload = self

    def detach(self) -> None:
        """Remove the seam from the layer. Refused after `freeze()`.

        Detaching a frozen seam is how the identity check silently disappears: after capture the
        graph holds the container addresses forever, so a detach cannot un-bake them — it only
        removes the one thing that would have reported that they are the wrong ones.

        DELETES the instance attribute rather than assigning `None` over it. `MoELayer` declares
        `_weight_offload = None` as a CLASS attribute precisely so a non-offloaded layer stores
        nothing in `vars(self)` — which is what `BaseOP.state_dict` / `load_state_dict` / `post_load`
        and `granule._iter_tensors` all iterate. Assigning `None` here left a real instance entry
        behind, so a detached layer was NOT restored to the pristine shape the class attribute
        exists to guarantee, and `_weight_offload` became a permanent member of every walked
        `__dict__` — the exact difference the class-vs-instance decision was made to avoid, silently
        undone by the teardown path.
        """
        if self._frozen:
            raise InterpositionError(
                f"seam {self.path!r} is frozen: detaching after boot removes the only check that "
                f"the captured graph's baked addresses are the ones this seam placed"
            )
        self._layer.__dict__.pop("_weight_offload", None)

    # -- internals -----------------------------------------------------------------------------
    def _refuse_before_any_byte_moves(self) -> None:
        """Every reason this layer cannot be host-resident, asked BEFORE a page is committed.

        Ordering is the point: `_validate` runs before any copy, but the allocator has already been
        asked for arena rows by then, and an arena that was partly committed and then abandoned is
        the same shape of waste the capacity gate exists to avoid. This refusal is cheap, so it
        happens first and leaves the arena untouched.

        The refusal itself is asked of the CONTAINER rather than of a central env table, so a format
        that materialises the whole stack per forward declares that itself
        (`_GroupedFP8Experts.offload_refusal` names MINISGL_ZAYA_OLDMOE / MINISGL_ZAYA_W8A16) and a
        renamed knob cannot leave a stale copy here. Plan §6.1 rule 4.
        """
        why = self.offload_refusal()
        if why is not None:
            raise InterpositionError(f"seam {self.path!r}: cannot be offloaded — {why}")


# =====================================================================================
# The bake: one ordered move per host-resident layer
# =====================================================================================


@dataclass(frozen=True)
class _Alias:
    """One attribute name that reaches a component's bytes, WITH the dtype and shape it reaches
    them under.

    The dtype/shape are the whole point. `granule.ExpertComponent` records alias NAMES only, so
    without this the bake has no way to reproduce `_w_op`'s `uint8` view of an `f8_e4m3` weight, or
    `_scales_op`'s `(E, N)` view of an `(E, N, 1)` scale — and a name rebound to the wrong dtype is
    read by a kernel binding as a different number of bytes per element.
    """

    name: str
    dtype: Any
    shape: tuple[int, ...]


@dataclass
class _BakeItem:
    """One stacked tensor and the arena row it is baked into.

    `attrs` is EVERY attribute path that reaches these bytes, aliases included:
    `_GroupedFP8Experts.post_load` sets `_w_op = weight.contiguous().view(uint8)`, and under
    `MINISGL_ZAYA_OLDMOE=1` it does not delete `weight`. Rebinding only one of them de-aliases the
    pair, after which `dequant()` and the kernel read different memory — right shapes, wrong
    numbers, no crash. Rebinding all of them to the CANONICAL view is the same bug wearing a
    different hat: the names stay aliased but `_w_op` stops being uint8.

    `alias_srcs` holds the live tensor OBJECT behind every one of those names, so the leak proof
    watches all of them. Watching only the canonical object proves nothing about the storage: an
    alias is a separate `torch.Tensor` sharing the same allocation, so a surviving `_w_op` keeps the
    device original alive while the canonical `weight` object dies and the weakref reports clean.
    """

    name: str
    layer: Any
    owner_attr: str
    container: Any
    attrs: tuple[_Alias, ...]
    src: Any
    dst: Any
    alias_srcs: list = field(default_factory=list)
    nbytes: int = 0

    def rebind(self) -> None:
        """Point every alias at the arena row, then drop this item's references to the sources.

        Every rebind is READ BACK through `granule._lookup` — the walker that produced the name and
        the walker every other consumer (`stacked_tensors`, `expert_slice`, a chunked Stage-B
        repack) uses to resolve it. A `_assign` that wrote somewhere else instead of raising leaves
        the container reading the device original with the arena row orphaned: right shapes, right
        numbers, and none of the offload. The check is a dict/attribute walk per name at boot.
        """
        bare = isinstance(self.container, torch.Tensor)
        for a in self.attrs:
            view = _reinterpret(self.dst, a)
            _assign(self.layer, self.owner_attr, self.container, a.name, view)
            got = getattr(self.layer, self.owner_attr) if bare else _lookup(self.container, a.name)
            if got is not view:
                raise InterpositionError(
                    f"weight bake {self.name}: rebinding {a.name!r} did not take — the path still "
                    f"resolves to a different tensor. `_assign` and `granule._lookup` disagree "
                    f"about this path's grammar, so the container is still reading the original "
                    f"and the arena row is orphaned."
                )
        self.src = None
        self.alias_srcs = []
        if isinstance(self.container, torch.Tensor):
            # A bare stacked tensor IS its own container, so `self.container` is a second live
            # reference to the source. Leaving it set would pin the device original AND make the
            # weakref check below report a leak this item is itself causing.
            self.container = None


def _reinterpret(dst: Any, alias: _Alias) -> Any:
    """`dst`'s bytes seen with the ALIAS's OWN dtype and shape, so an alias stays an alias.

    The arena row is allocated once, in the canonical component's dtype/shape. An alias is the same
    bytes under a different view (`weight` f8_e4m3 vs `_w_op` uint8; `weight_scale` (E,N,1) vs
    `_scales_op` (E,N)), so it is rebound as a reinterpretation of the SAME storage rather than as a
    second copy — and as ITS OWN view, not the canonical one. Handing `_w_op` back as f8_e4m3 keeps
    the aliasing intact but changes what every downstream `numel()`/`element_size()`/kernel binding
    computes from it, which is a wrong-bytes bug with no crash.
    """
    if dst.dtype == alias.dtype and tuple(dst.shape) == alias.shape:
        return dst
    want = 1
    for d in alias.shape:
        want *= int(d)
    want *= torch.empty((), dtype=alias.dtype).element_size()
    have = dst.numel() * dst.element_size()
    if want != have:
        raise InterpositionError(
            f"alias {alias.name!r}: byte size {want} does not match the arena row's {have}"
        )
    out = dst.reshape(-1).view(torch.uint8).view(alias.dtype).reshape(alias.shape)
    if out.dtype != alias.dtype or tuple(out.shape) != alias.shape:
        raise InterpositionError(
            f"alias {alias.name!r}: reinterpretation produced {out.dtype}/{tuple(out.shape)}, "
            f"expected {alias.dtype}/{alias.shape}"
        )
    if out.data_ptr() != dst.data_ptr():
        raise InterpositionError(
            f"alias {alias.name!r}: reinterpretation is not the same storage as the arena row"
        )
    return out


def _assign(layer: Any, owner_attr: str, container: Any, dotted: str, value: Any) -> None:
    """Rebind the attribute `granule._lookup` would have read, so the two cannot drift.

    Mirrors that walker's path grammar exactly, including `[i]` indexing into list/dict children.
    Two cases are special:
      * a bare `torch.Tensor` container (`_UnquantizedMoEMethod.create_experts` returns a raw
        `torch.empty(E, out, in)`) has no attribute to set — the container IS the tensor, so the
        binding lives on the MoELayer. Plan §6.1 rule 6 proposed wrapping it in a new
        `_GroupedUnquantizedExperts(BaseOP)` and editing the two apply/ep_local reads; rebinding the
        owning attribute gives the same "all formats present one interface" property with zero
        changes to the unquantized forward path.
      * a tuple element cannot be replaced. Silently skipping it would leave that alias pointing at
        the dropped original — the exact silent-desync class this path exists to prevent.
    """
    if isinstance(container, torch.Tensor):
        setattr(layer, owner_attr, value)
        return
    steps = _path_steps(dotted)
    if not steps:
        raise InterpositionError(f"cannot rebind an empty path on {type(container).__name__}")
    obj: Any = container
    for kind, key in steps[:-1]:
        obj = getattr(obj, key) if kind == "attr" else obj[key]
    kind, key = steps[-1]
    if kind == "attr":
        setattr(obj, key, value)
        return
    if isinstance(obj, tuple):
        raise InterpositionError(
            f"cannot rebind {dotted!r}: it lives in a tuple, which is immutable. Leaving it "
            f"would keep an alias pointing at the dropped original."
        )
    obj[key] = value


def _path_steps(dotted: str) -> list[tuple[str, Any]]:
    """Split a walk path into resolution steps using `granule._lookup`'s EXACT grammar.

    `_lookup` parses each dot-separated part as `head` + a run of `[key]` subscripts
    (`experts[0].weight` -> getattr 'experts', index 0, getattr 'weight'). The previous splitter
    here only recognised a subscript when the WHOLE part was `[i]`, which no walk path ever is —
    `granule._iter_tensors` emits `f"{name}[{i}]"`. So `parts[0]` fell through to
    `setattr(container, "parts[0]", value)`: a brand-new junk attribute, while the real list element
    kept pointing at the device original. That is a de-alias with no exception — the container's
    live tensor stays on the device stack and the arena row is orphaned. `rebind()`'s round-trip
    check now also catches it, but the grammar must agree in the first place or the two walkers
    describe different objects.
    """
    steps: list[tuple[str, Any]] = []
    for part in dotted.split("."):
        head, _, subs = part.partition("[")
        if head:
            steps.append(("attr", head))
        while subs:
            key, _, subs = subs.partition("]")
            key = key.lstrip("[")
            try:
                steps.append(("idx", int(key)))
            except ValueError:
                steps.append(("idx", key))
    return steps


def _validate(item: _BakeItem) -> int:
    """Refuse anything that would make the copy silently lossy. Returns the item's byte count.

    Every one of these is a wrong-numbers-without-a-crash failure if allowed through:
      * a dtype mismatch would make `copy_` CAST a packed int4 / uint8 / e8m0 storage encoding as
        if it were a number (`layers/base.py::_coerce_dtype` refuses the same thing at load time);
      * a shape mismatch would broadcast or truncate;
      * a non-contiguous source means the granule derivation does not satisfy the kernels'
        "`t[e]` is a flat range at `base + e*row_bytes`" precondition (plan §6.1 rule 1);
      * a destination on a different device type than the source means the copy is not the
        device-issued write the arena's bandwidth and coherence were measured on.
    """
    src, dst = item.src, item.dst
    if src is None:
        raise InterpositionError(f"weight bake {item.name}: item already rebound")
    if src.dtype != dst.dtype:
        raise InterpositionError(
            f"weight bake {item.name}: dtype {src.dtype} -> {dst.dtype}. Packed/quantized storage "
            f"dtypes are an encoding, not a precision; a cast destroys the weight."
        )
    if tuple(src.shape) != tuple(dst.shape):
        raise InterpositionError(
            f"weight bake {item.name}: shape {tuple(src.shape)} -> {tuple(dst.shape)}"
        )
    if not src.is_contiguous():
        raise InterpositionError(
            f"weight bake {item.name}: source is not contiguous. The kernels index experts as a "
            f"flat range at base + e*row_bytes; a non-contiguous source means the granule "
            f"derivation does not match that precondition."
        )
    if not dst.is_contiguous():
        raise InterpositionError(f"weight bake {item.name}: arena row is not contiguous")
    if dst.device.type != src.device.type:
        raise InterpositionError(
            f"weight bake {item.name}: source is on {src.device} but the arena row is on "
            f"{dst.device}. The row must be addressed through hipHostGetDevicePointer so the GPU "
            f"writes the host pages; a device-type mismatch means the copy is a host-side store, "
            f"which is not the path the arena was measured on."
        )
    if dst.device.type == "cuda" and dst.device.index != src.device.index:
        # THE INDEX, not just the type. `hipHostGetDevicePointer` returns a mapping registered for
        # ONE device; a row carved from an arena pinned for `cuda:0` and written from a weight on
        # `cuda:1` is a peer access this arena never established and never measured, and on this box
        # the two cards are not even the same link generation (card 0 root port Gen5 x8, card 1 Gen4
        # x8). It is also the shape of the TP=2 rank-1 bug `ArenaMemPool.use()` guards on the other
        # side: a pool installed on the wrong device serves VRAM with zero fallbacks counted. Either
        # way the bytes end up somewhere the capacity plan did not book, so refuse rather than copy.
        raise InterpositionError(
            f"weight bake {item.name}: source is on {src.device} but the arena row is on "
            f"{dst.device}. The arena is pinned and mapped for ONE device; copying a weight from "
            f"another card's allocation into it is a peer access this arena never established, and "
            f"it means this rank's arena belongs to a different card than its weights."
        )
    return src.numel() * src.element_size()


def _bitwise_equal(dst: Any, src: Any) -> bool:
    """Compare the raw BYTES of two same-dtype, same-shape contiguous tensors.

    `torch.equal` is VALUE equality, and every weight tensor this bake moves is a bit pattern, not a
    number — which breaks the read-back gate in both directions:

      * FALSE PASS. `-0.0 == +0.0`, so an arena that stored a flipped sign bit reads as verified.
        Denormal/NaN-payload differences are the same class. The gate exists precisely because
        Phase 0 produced four cases of the driver returning success over wrong state, so it must
        assert on the bytes it wrote, not on their numeric interpretation.
      * FALSE FAIL — the worse one, because it accuses the arena. `NaN != NaN`, so a CORRECT copy of
        a stack containing a NaN fails with "the arena is wrong. Do not retry." That is reachable
        today: `quant/mxfp4.convert_mxfp4_moe` reports `e8m0_nan_groups`, i.e. a shipped
        compressed-tensors MXFP4 checkpoint can carry NaN fp16 group scales into `_scales_op`.

    `granule._bitwise_rows_equal` already takes the uint8 view for exactly these reasons; this is
    the same decision at the other end of the move.

    IT IS COMPARED IN BOUNDED SLICES, AND THAT IS A MEMORY-ACCOUNTING REQUIREMENT, NOT A STYLE
    CHOICE. `torch.equal` on CUDA is `self.eq(other).all()`: it materialises a `numel`-BYTE bool
    tensor on the DEVICE, through the ordinary caching allocator (the arena `MemPool` is only active
    inside `ArenaMemPool.use()`, which the bake has long since exited). This runs at the single
    tightest moment in the whole boot — inside `_bake`, after `post_load()` has finished and BEFORE
    the rebind drops a single device original, i.e. at the un-offloaded model's peak VRAM. A fused
    w13 weight stack is a ~1 GiB row on the target shape, so the un-sliced form asked a 16 GB card
    for a ~1 GiB scratch it does not have, up to 64 times, and the failure is a bare CUDA OOM inside
    a read-back self-test with nothing in the message connecting it to the arena. Slicing bounds the
    scratch to `chunk_bytes` and moves exactly the same bytes over PCIe (the host row is read in
    full either way), so the gate is not weakened: every byte is still compared.
    """
    if dst.dtype != src.dtype or tuple(dst.shape) != tuple(src.shape):
        return False
    d = dst.reshape(-1).view(torch.uint8)
    s = src.reshape(-1).view(torch.uint8)
    n = int(d.numel())
    step = max(1, int(SELFTEST_COMPARE_CHUNK_BYTES))
    for off in range(0, n, step):
        # Slices of a contiguous 1-D uint8 view are themselves contiguous views, so this allocates
        # no source-side copy; only the comparison scratch, which is what is being bounded.
        if not bool(torch.equal(d[off : off + step], s[off : off + step])):
            return False
    return True


def _bake(
    path: str, items: Sequence[_BakeItem], *, selftest: int, report: "SeamBindReport"
) -> None:
    """Copy, verify, rebind, prove-dropped. See `MoEWeightSeam.bind` for why the order is this one."""
    if not items:
        return
    sizes = [_validate(it) for it in items]

    for it in items:
        # Both sides are `cuda` tensors in production, so this is a device-side copy: the copy
        # engine writes the pinned host pages over PCIe, never the CPU. `non_blocking` is left at
        # its default — it would not mean what a reader assumes when both sides look like `cuda`.
        it.dst.copy_(it.src)

    # ONE sync for every copy, keyed off where the rows actually are. A CPU-only bake (the GPU-free
    # unit tests) has nothing queued, and `torch.cuda.synchronize()` there would raise, not no-op.
    #
    # SCOPED TO THE ROWS' OWN DEVICE, never the ambient current one. `torch.cuda.synchronize()` with
    # no argument synchronizes whatever device is current, and at TP=2 rank 1's arena is `cuda:1`
    # while the process-current device is global mutable state that the HIP binding, the arena's own
    # `hipSetDevice`/restore dance and torch all write. If the current device is not the one the
    # copies were queued on, the "one sync for the lot" waits on an idle card and returns
    # immediately, and the copies are still in flight. Stream ordering happens to save the read-back
    # that follows (it is queued behind them on the same stream), which is exactly what makes this
    # dangerous: the barrier this function claims to place would silently not exist, and the next
    # thing to depend on it — a rebind, a `mark_populated()`, a host-side reader in Stage B — would
    # read pages the DMA has not finished writing. Naming the device costs nothing and cannot be
    # wrong.
    if items[0].dst.is_cuda:
        torch.cuda.synchronize(items[0].dst.device)

    for idx in select_selftest_indices(len(items), selftest):
        it = items[idx]
        if not _bitwise_equal(it.dst, it.src):
            raise InterpositionError(
                f"weight bake {it.name!r}: the read-back does not match the source, and the copy "
                f"returned success. Phase 0 recorded four separate cases of this driver reporting "
                f"success over wrong state, which is why this check asserts on the DATA. Do not "
                f"retry — the arena is wrong."
            )
        report.verified_components += 1
        report.verified_bytes += sizes[idx]

    # Watch EVERY name's tensor object, not just the canonical one: an alias is a separate
    # `torch.Tensor` over the same allocation, so a surviving `_w_op` keeps the device original
    # alive (the VRAM the offload was meant to free) while `weight`'s object dies and a
    # canonical-only weakref reports clean.
    watches = [
        (f"{it.name}:{a.name}", weakref.ref(t))
        for it in items
        for a, t in zip(it.attrs, it.alias_srcs)
    ]
    for it in items:
        it.rebind()  # sets it.src = None, dropping this function's last reference
    leaked = tuple(name for name, ref in watches if ref() is not None)
    if leaked:
        # Direct evidence rather than an inference from allocator statistics: a surviving source
        # names the exact alias that still points at the old storage, which is the only diagnostic
        # that shortens the hunt for a missed `_w_op`/`weight` pair — and it means the VRAM the
        # offload was supposed to free is still held.
        raise InterpositionError(
            f"seam {path!r}: {len(leaked)} source tensor(s) survived the rebind "
            f"({', '.join(leaked[:4])}). An alias still points at the device original, so that "
            f"name and the kernel would read different memory."
        )
    report.moved_bytes += sum(sizes)
    report.components += len(items)


# =====================================================================================
# Binder — the library entry point M1-B calls from the engine
# =====================================================================================


@dataclass
class BindOutcome:
    seams: tuple[MoEWeightSeam, ...] = ()
    reports: tuple[SeamBindReport, ...] = ()
    moved_bytes: int = 0
    host_layers: int = 0
    device_layers: int = 0
    # Bytes left resident on the card, summed from the LIVE post-load granule specs of the
    # device-placed seams — not from the plan, and not from an allocator statistic. This is the only
    # independent measurement of the device tier that exists: under layer-granular placement a
    # device layer is never reallocated, so there is no allocation event for the allocator to
    # report, and `plan.total_resident_bytes - moved` is just the plan restated (it reduces
    # algebraically to `plan.host_resident_bytes - moved`). `bake.StageASession.seal()` feeds this
    # into `WeightPlanResolution.assert_device_accounting`, which is the boot assertion plan §5.3
    # requires before the KV pool is sized with no sixth subtrahend.
    device_resident_bytes: int = 0
    plan_digest: str = ""
    notes: list[str] = field(default_factory=list)
    # CPU-COMPUTE tier. Counted separately from `host_layers` throughout, because the two relieve
    # different ceilings: host layers are PINNED (charged against `OffloadPrior`'s 62 GiB
    # `hipHostMalloc` ceiling, the binding constraint) and CPU layers are PAGEABLE (charged only
    # against MemAvailable). Folding them into one counter is what would make the boot banner claim
    # a pinned arena that was never reserved.
    cpu_layers: int = 0
    cpu_resident_bytes: int = 0

    @property
    def arena_moved_bytes(self) -> int:
        """Bytes copied into the PINNED ARENA — `moved_bytes` minus the CPU tier's pageable copies.

        The number every arena-side gate means when it says "moved". `moved_bytes` is the total over
        all non-device placements, and once a third tier exists the two stop being the same thing:
        `StageAAccounting` compares the copied figure against `plan.host_resident_bytes`, caps
        `model_memory_correction()` with it, and `TorchStackPool.assert_clean` expects the pinned
        pool to have SERVED exactly that many bytes. A CPU layer's copy goes to plain
        `torch.empty(device="cpu")` — it never touches the arena, the pool or `memory_allocated()` —
        so feeding the total into any of those three states a claim about the arena that is wrong by
        the whole CPU tier (~0.6 GiB at 21 layers), in the direction that over-corrects the model
        term and over-sizes the KV pool. Equal to `moved_bytes` on every two-tier plan, which is why
        this can be derived rather than counted separately.
        """
        return self.moved_bytes - self.cpu_resident_bytes

    def describe(self) -> str:
        cpu = (
            f", {self.cpu_layers} cpu-compute "
            f"({self.cpu_resident_bytes / (1 << 30):.2f} GiB PAGEABLE, not pinned)"
            if self.cpu_layers
            else ""
        )
        return (
            f"weight-offload bound: {self.host_layers} MoE layers host-resident "
            f"({self.moved_bytes / (1 << 30):.2f} GiB moved), {self.device_layers} device-resident "
            f"({self.device_resident_bytes / (1 << 30):.2f} GiB measured){cpu}, "
            f"plan={self.plan_digest}"
        )


def build_layer_weights(
    seams: Sequence[MoEWeightSeam],
    *,
    priority: Sequence[int] | None = None,
    skipped: list[str] | None = None,
) -> list[LayerWeights]:
    """The seams that are candidates for host residency, as placement units.

    A seam whose CONTAINER refuses host residency (`granule.offload_refusal`) is EXCLUDED, not
    planned. That is not a policy choice made here, it is the format answering for itself: ZAYA's
    `_GroupedFP8Experts` refuses under `MINISGL_ZAYA_OLDMOE` / `MINISGL_ZAYA_W8A16` because those
    forwards read EVERY expert, so a host-resident layer would stream the whole (E,N,K) stack per
    step. `plan.size_planned_layers_from_model` has always filtered them; this function did not, so
    the standalone path planned a refusing layer onto HOST and then died in `bind()` — i.e. weight
    offload could not boot AT ALL on those two knobs through the entry point this module documents,
    while the engine's own planner handled them. A format that declares a refusal must be handled
    identically wherever it is enumerated, or "what must a new quant format implement?" stops being
    "nothing" and becomes "nothing, provided the caller used the other planner".

    Excluding is the correct composition and not a silent drop: `bind_plan` binds every discovered
    seam the plan does not name to `StackKind.DEVICE` (zero bytes moved) and records it in
    `BindOutcome.notes`, which is the same treatment the policy-excluded MTP draft head gets. Pass
    `skipped` to collect the reasons for a banner.
    """
    if priority is not None and len(priority) != len(seams):
        raise PlacementError("priority must have one integer per seam")
    out: list[LayerWeights] = []
    for i, s in enumerate(seams):
        why = s.offload_refusal()
        if why is not None:
            if skipped is not None:
                skipped.append(f"{s.path} (container refuses host residency: {why})")
            continue
        out.append(s.layer_weights(priority=0 if priority is None else int(priority[i])))
    return out


def attach_seams(root: Any, *, allow_meta: bool = False) -> list[MoEWeightSeam]:
    """Derive a seam for every `MoELayer` under `root` and install it on the layer.

    Call AFTER `post_load()` and on materialized tensors: `derive_granule_spec` refuses meta
    tensors by default because every meta tensor reports `data_ptr() == 0`, which makes aliasing
    and expert-invariance undetectable — the granule would then be wrong in exactly the silent way
    the walk exists to prevent. `allow_meta=True` is for a shapes-only sizing estimate before the
    load and must never be used for a bind.
    """
    return [
        attach_seam(path, layer, allow_meta=allow_meta) for path, layer in discover_moe_layers(root)
    ]


def attach_seam(path: str, layer: Any, *, allow_meta: bool = False) -> MoEWeightSeam:
    """Derive ONE layer's seam and install it. The per-layer half of `attach_seams`.

    Split out for `stage_b.SeamLayerSink`, which attaches a seam the moment the chunked load
    finalizes that layer. It cannot use `attach_seams`: the other layers' containers do not hold
    real tensors yet at that point, and `derive_granule_spec` would (correctly) refuse them.
    """
    n = int(layer.local_num_experts)
    # Which attribute holds which GEMM comes from ONE place — the layer — and the specs are
    # keyed by it here and again in `_plan_items`. Reading `specs['gate_up_proj']` while the
    # bake enumerated containers in `expert_containers()` order would pair each spec with the
    # OTHER container the day those two orders differ, and a swap is undetectable downstream
    # because `assert_granule_pair_consistent` forces the two specs to share a component set.
    attrs = _container_attrs(layer)
    try:
        # Prefer the layer's own derivation (`MoELayer.granule_specs`) — it already passes the
        # LOCAL expert count and cross-checks the pair, so the seam cannot drift from it.
        fn = getattr(layer, "granule_specs", None)
        if callable(fn):
            specs = fn(allow_meta=allow_meta)
        else:
            specs = {
                a: spec_for_container(getattr(layer, a), n, allow_meta=allow_meta) for a in attrs
            }
        missing = [a for a in attrs if a not in specs]
        if missing:
            raise InterpositionError(
                f"MoE layer {path!r}: granule_specs() is missing {missing}; it must key its "
                f"specs by the same attribute names expert_containers() yields ({attrs})."
            )
        w13, w2 = specs[attrs[0]], specs[attrs[1]]
    except GranuleError as exc:
        raise InterpositionError(f"MoE layer {path!r}: {exc}") from exc
    seam = MoEWeightSeam(path, layer, w13_spec=w13, w2_spec=w2)
    seam.attach()
    return seam


def bind_seam(
    seam: MoEWeightSeam,
    kind: StackKind,
    allocator: StackAllocator | None,
    *,
    selftest: int = SELFTEST_SAMPLE,
    out: "BindOutcome",
    count_device_bytes: bool = True,
) -> SeamBindReport:
    """Bind ONE seam and fold its result into `out`. The unit of work `bind_plan` iterates.

    `count_device_bytes=False` for a layer the PLAN does not name (the policy-excluded MTP draft
    head). Its bytes are absent from `plan.device_resident_bytes` too, and
    `WeightPlanResolution.assert_device_accounting` compares the two — so counting them here would
    make every MTP-capable checkpoint fail a boot assertion about a layer neither side planned.

    Extracted so Stage B's per-layer sink and the one-shot `bind_plan` share one implementation of
    the move AND one implementation of its accounting. The `device_resident_bytes` term in
    particular is the only INDEPENDENT measurement of the device tier that exists (see
    `BindOutcome`), and a second copy of it in the chunked path is exactly the drift that would make
    two boot modes disagree about how much VRAM the model took.
    """
    rep = seam.bind(kind, allocator, selftest=selftest)
    out.reports = out.reports + (rep,)
    out.moved_bytes += rep.moved_bytes
    if kind is StackKind.HOST:
        out.host_layers += 1
    elif kind is StackKind.CPU:
        # NOT the `else` arm. Three tiers now exist, and an `if HOST / else DEVICE` shape would bill
        # every CPU layer's ~30 MB against `device_resident_bytes` — the one INDEPENDENT measurement
        # of the device tier, which `assert_device_accounting` compares against the plan at boot. On
        # a 21-CPU-layer plan that is ~0.6 GiB of phantom VRAM in the term the KV pool is sized
        # from, and it would surface as an unrelated OOM rather than as an accounting failure.
        out.cpu_layers += 1
        out.cpu_resident_bytes += rep.moved_bytes
    else:
        out.device_layers += 1
        if count_device_bytes:
            # Measured off the live containers this seam is holding, so a plan whose byte model
            # disagrees with what post_load() actually produced (sizing.py under-counts MXFP4's
            # E8M0 -> fp16 scale widening by 2x, for one known case) is caught here rather than by
            # a KV pool sized against a fiction.
            out.device_resident_bytes += seam.w13_spec.total_bytes + seam.w2_spec.total_bytes
    return rep


def detach_seams(seams: Sequence[MoEWeightSeam]) -> None:
    for s in seams:
        s.detach()


def iter_seams(root: Any) -> Iterator[MoEWeightSeam]:
    for _, layer in discover_moe_layers(root):
        seam = getattr(layer, "_weight_offload", None)
        if seam is not None:
            yield seam


def bind_plan(
    seams: Sequence[MoEWeightSeam],
    plan: OffloadPlan,
    allocator: StackAllocator | None,
    *,
    selftest: int = SELFTEST_SAMPLE,
    freeze: bool = True,
    log: bool = True,
    cpu_worker: Any = None,
) -> BindOutcome:
    """Execute `plan` over `seams`: move the host-resident layers, then freeze.

    `cpu_worker` MUST be passed whenever the plan has CPU-tier layers, and it is attached HERE
    rather than by the caller afterwards. That is a lifetime fact, not a convenience: `freeze()`
    is rule R1 ("after this, every placement entry point raises") and `attach_cpu_worker` is a
    placement entry point, so a caller that binds-then-attaches is attaching to a frozen seam and
    gets an `InterpositionError`. Binding a CPU layer and leaving it worker-less is not a
    recoverable state either — `MoELayer.forward` would reach `cpu_forward` with nothing to submit
    to — so this function REFUSES a CPU plan with no worker instead of producing a model that
    boots and then dies on its first token.

    Order is the PLAN order, which is the seam/discovery order, which is structural — so both ranks
    populate their arenas in the identical sequence and any chunked arena lays out identically.

    THE PLAN IS A SUBSET, NOT AN EQUALITY. Every planned path must exist on the model (a planned
    layer that does not is fatal — the plan spent budget on bytes that are not there). A discovered
    layer the plan does not name is bound DEVICE, by name, in `BindOutcome.notes`, because the
    resolver excludes layers BY POLICY (`plan.OFFLOAD_MTP_HEAD`), and because the MTP head's
    checkpoint namespace is a string literal in the model rather than something the module walk can
    reconstruct. Demanding set equality made this unsatisfiable on every MTP-capable checkpoint.

    This is the standalone driver, for tests and for a caller that owns its own model walk. The
    engine's path is `bake.StageASession` / `StageARuntime`, which sequences
    plan -> attach -> load -> bind -> seal around `Engine.__init__`'s existing calls and does the
    VRAM accounting; its `bind()` calls `attach_seams` + this function, so there is exactly one
    implementation of the move and one plan behind both entry points. It passes `freeze=False`
    because `seal()` closes the seams, the arena and the process-wide `hipmem` latch together —
    rule R1 is a statement about a single point in the process's life.
    """
    by_path = {s.path: s for s in seams}
    planned = {p.path for p in plan.placements}
    missing = sorted(planned - set(by_path))
    if missing:
        # A PLANNED layer that the model does not have is always fatal: the plan spent budget on
        # bytes that are not there, so every capacity number downstream (arena size, KV pool) is
        # about a model this process is not running.
        raise InterpositionError(
            f"the plan names MoE layers this model does not have: {missing}. Discovered paths are "
            f"{sorted(by_path)}. A plan derived against a different module tree would place the "
            f"wrong layers on the wrong stack. Path grammar is `BaseOP.state_dict`'s (see "
            f"`_iter_ops`), which is what `weights/plan.py::moe_layer_shapes` mirrors."
        )

    n_cpu = sum(1 for p in plan.placements if p.kind is StackKind.CPU)
    if n_cpu and cpu_worker is None:
        raise InterpositionError(
            f"the plan places {n_cpu} layer(s) on the CPU-COMPUTE tier but no `cpu_worker` was "
            f"passed to bind_plan(). The worker has to be attached BEFORE freeze() (rule R1 makes "
            f"`attach_cpu_worker` unreachable afterwards), and a CPU seam without one is not a "
            f"degraded mode: `MoELayer.forward` would reach `cpu_forward` with nothing to submit "
            f"to and the serve would die on its first token instead of at boot."
        )

    out = BindOutcome(plan_digest=plan.digest())
    cpu_expert_offset = 0
    for placement in plan.placements:
        seam = by_path[placement.path]
        bind_seam(seam, placement.kind, allocator, selftest=selftest, out=out)
        if placement.kind is StackKind.CPU:
            # Attached inside the bind loop, in PLAN order, so `backend_expert_offset` is derived
            # from the same walk on every rank rather than from the caller's enumeration. The
            # offset counts EXPERTS, not layers: the worker holds one packed table and each CPU
            # layer occupies `num_experts` consecutive slots in it.
            seam.attach_cpu_worker(cpu_worker, backend_expert_offset=cpu_expert_offset)
            cpu_expert_offset += int(placement.num_experts)

    # A discovered layer the plan does NOT name is DEVICE-resident, explicitly and by name. It is
    # not an error, because the resolver excludes layers BY POLICY: `plan.OFFLOAD_MTP_HEAD = False`
    # skips the MTP draft head (re-read every draft step), and the draft head's checkpoint namespace
    # (`mtp.layers.0.mlp.experts`) is a string literal in the model, not something the module walk
    # can reconstruct — so requiring set EQUALITY here made the bind unsatisfiable on every
    # MTP-capable checkpoint. Binding them DEVICE (which moves zero bytes) is what keeps `resolve()`
    # guarding them: an unbound seam would still be attached and still be consulted, but with no
    # ledger and no `kind`.
    for path in sorted(set(by_path) - planned):
        bind_seam(
            by_path[path],
            StackKind.DEVICE,
            None,
            selftest=selftest,
            out=out,
            count_device_bytes=False,
        )
        out.notes.append(f"{path}: not in the plan (excluded by policy) -> DEVICE, 0 bytes moved")

    # No `torch.cuda.synchronize()` here: `_bake` already issues exactly one per layer, keyed off
    # whether that layer's arena rows are actually `cuda` (a CPU-only bake has nothing queued, and
    # calling it there would raise rather than no-op).
    if freeze:
        for seam in seams:
            seam.freeze()
    out.seams = tuple(seams)
    if log:
        # Plain `info`, not `info_rank0`: `_hip_engage`-style rank-0-only logging would hide a rank
        # whose plan diverged, which is the single most important thing to see here.
        _logger.info(f"[weight-offload] {plan.describe()}")
        _logger.info(f"[weight-offload] {out.describe()}")
        for note in out.notes:
            _logger.info(f"[weight-offload] {note}")
    return out


# NOTE — CROSS-RANK PLAN AGREEMENT LIVES IN `plan.WeightPlanResolution.assert_rank_agreement`,
# NOT HERE, and this note exists so nobody adds a second one. The check has to be a collective, and
# a collective placed on THIS side of the feature deadlocks on exactly the case it is meant to
# catch: `bind_plan` is only reached by a rank whose plan has host-resident layers, so if rank 0
# resolves an all-device plan and rank 1 does not, rank 1 enters the gather alone and blocks
# forever. The agreement therefore belongs at the one point every rank passes unconditionally —
# `bake._resolve_driver`, before its `if not resolution.enabled` early return. `bind_plan` records
# `plan.digest()` in `BindOutcome.plan_digest` and logs it on every rank (plain `info`, not
# `info_rank0`) so a divergence that somehow got past the gate is still visible in the two logs.


@dataclass(frozen=True)
class SeamResidencyProof:
    """What `prove_seam_residency` MEASURED off the live model. Every field is a count of tensors
    or bytes it actually walked; none is read from the plan."""

    moe_layers: int = 0
    host_layers: int = 0
    device_layers: int = 0
    host_tensors: int = 0
    host_bytes: int = 0
    device_tensors: int = 0
    device_bytes: int = 0
    #: CPU-COMPUTE tier (`StackKind.CPU`). Separate counters, never folded into the host ones: a
    #: host layer is PINNED and streams over PCIe, a CPU layer is PAGEABLE and never crosses the
    #: bus at all. `owns_pointer` answers OPPOSITELY for the two, so a merged counter would make
    #: the pointer check unstateable — which is how a "not HOST means DEVICE" walk would come to
    #: bill 21 pageable layers against VRAM and still print a clean banner.
    cpu_layers: int = 0
    cpu_tensors: int = 0
    cpu_bytes: int = 0
    #: Tensors that were skipped in the byte totals because another layer had already contributed
    #: the SAME allocation. Non-zero only under the stream tier, whose whole mechanism is that N
    #: layers' expert containers alias ONE buffer set (`weights/stream_tier.py`). Without the dedup
    #: this proof reported 46.9 GiB "device-resident" on a 15.9 GiB card — an obviously impossible
    #: number in a boot banner, which is worse than a missing one because it reads as authoritative.
    aliased_tensors: int = 0
    #: True when the arena was consulted, i.e. every host byte above was proven to live inside a
    #: pinned chunk. False means the walk ran without an `owns_pointer` and the host figures are
    #: only "what the seam claims", which is NOT a residency proof.
    pointer_checked: bool = False

    def describe(self) -> str:
        g = 1 << 30
        return (
            f"seam proof: {self.moe_layers} MoE layers reachable from the LIVE model, "
            f"{self.host_layers} host ({self.host_tensors} tensors, {self.host_bytes / g:.3f} GiB "
            f"{'INSIDE the pinned arena' if self.pointer_checked else 'UNVERIFIED'}), "
            f"{self.device_layers} device ({self.device_tensors} tensors, "
            f"{self.device_bytes / g:.3f} GiB)"
            + (
                f", {self.cpu_layers} cpu-compute ({self.cpu_tensors} tensors, "
                f"{self.cpu_bytes / g:.3f} GiB PAGEABLE, proven off-device)"
                if self.cpu_layers
                else ""
            )
            + (
                f", {self.aliased_tensors} tensor(s) aliased onto an allocation already counted "
                f"(stream tier)"
                if self.aliased_tensors
                else ""
            )
        )


def prove_seam_residency(
    root: Any,
    seams: Sequence[MoEWeightSeam],
    *,
    owns_pointer: Any = None,
    require_host_layers: bool = True,
) -> SeamResidencyProof:
    """Walk the LIVE model and prove the offload seam is actually in the serving path.

    WHY THIS EXISTS AND WHY IT IS NOT REDUNDANT WITH `seal()`. Every gate `seal()` runs asks the
    ARENA and the PLAN questions: did a row fall back to `hipMalloc`, does the carve match the
    reservation, do the copied bytes match the ledger. All of them can pass over a model the engine
    is not going to run. The failure this repo has already paid for once — the PLE runtime, where
    every component was green and the seam between the engine and them did not exist, so the served
    path was 0% functional — is exactly that shape. So this asks the MODEL instead, and asks it the
    two questions the arena cannot answer:

      1. **Is the seam reachable from the object the engine will call `forward()` on?**
         `discover_moe_layers(root)` is re-run against the live model (NOT against the seam list the
         binder returned), and every discovered layer must carry the corresponding seam as
         `_weight_offload`. A layer with no seam is a layer whose forward never consults the
         placement at all; a seam bound to a layer that is no longer in the tree is a ledger about
         weights nothing reads.
      2. **Do the tensors the kernels will read live where the plan says they do?** For every HOST
         layer this calls `seam.resolve(...)` with the containers read off the layer — byte for byte
         the call `MoELayer.forward` makes — and then asks the arena whether it owns each resulting
         tensor's `data_ptr()`. That is the only direct evidence that a host-placed layer streams
         from pinned host RAM rather than from a VRAM copy, and it is the one claim the whole
         feature rests on.

    `owns_pointer(ptr, nbytes) -> bool` is `PinnedWeightArena.owns_pointer`. Omitting it downgrades
    the proof to a structural one and says so in `pointer_checked`, which the caller must gate on —
    a "proof" that silently skipped its own evidence is worse than none.

    `require_host_layers` refuses a VACUOUS pass: on an enabled session, a walk that found zero
    host-resident layers means the offload arm is not in the serving path, however clean the boot
    log looked. Pass False only where an all-device bind is the expected outcome.
    """
    live = dict(discover_moe_layers(root))
    by_path = {s.path: s for s in seams}
    orphan_seams = sorted(set(by_path) - set(live))
    if orphan_seams:
        raise InterpositionError(
            f"weight-offload seams are bound to MoE layers that are NOT reachable from the model "
            f"the engine will serve: {orphan_seams}. Those seams' placement, byte accounting and "
            f"host residency are all about tensors no forward will read — the arena is populated "
            f"and the serving path is unchanged. Discovered live paths: {sorted(live)[:6]}..."
        )
    unseamed = sorted(p for p, layer in live.items() if getattr(layer, "_weight_offload", None) is None)
    if unseamed:
        raise InterpositionError(
            f"{len(unseamed)} MoE layer(s) on the live model carry no weight-offload seam, e.g. "
            f"{unseamed[:4]}. `MoELayer._weight_offload` is a CLASS attribute defaulting to None, "
            f"so an unseamed layer does not fail — it silently reads its device containers while "
            f"the plan, the arena reservation and the KV budget were all computed as though it had "
            f"been placed. Every discovered layer must be bound, even the ones the plan leaves on "
            f"the device (bind_plan binds those explicitly for this reason)."
        )

    moe = host_layers = device_layers = cpu_layers = 0
    host_tensors = host_bytes = device_tensors = device_bytes = aliased = 0
    cpu_tensors = cpu_bytes = 0
    # Bytes are attributed to an ALLOCATION, once. The stream tier points every one of its layers'
    # containers at a single buffer set, so a per-layer sum counts the same VRAM N times and the
    # totals stop being physical. Keyed on `data_ptr()` because that is what the kernel
    # dereferences and therefore what "resident" means here.
    seen_ptrs: set = set()
    for path, layer in sorted(live.items()):
        seam = layer._weight_offload
        moe += 1
        if by_path.get(path) is not seam:
            raise InterpositionError(
                f"MoE layer {path!r} carries a seam the binder did not produce. Two binders ran, or "
                f"a stale seam survived a rebuild; either way the residency ledger consulted by "
                f"`seal()` is not the one the forward consults."
            )
        if not seam.bound:
            raise InterpositionError(
                f"seam {path!r} was attached but never bound. `resolve()` would run with no "
                f"placement behind it and the layer's bytes are unaccounted on both tiers."
            )
        attrs = _container_attrs(layer)
        # EXACTLY the check `MoELayer.forward` makes — `resolve()` minus its ledger line. Not a
        # re-implementation of it: if the seam and the layer have drifted, this raises here, at boot,
        # instead of on the first request. `assert_identity` rather than `resolve` because emitting
        # `weight_offload.moe_resolve[host]` from a boot-time proof would make that ledger line stop
        # meaning "a forward read the arena", which is the only thing it is for.
        seam.assert_identity(*(getattr(layer, a) for a in attrs))
        # THREE tiers, tested by name. The two-tier shape this replaced (`is_host` / `not is_host`)
        # would classify every CPU-compute layer as DEVICE-resident and then run the DEVICE pointer
        # assertion on it. That assertion passes vacuously — a pageable host tensor is not inside
        # the pinned arena either — so a 21-layer CPU plan would have produced a green proof that
        # reported ~0.6 GiB of VRAM which does not exist, in the same banner the KV budget is read
        # from. `computes_on_cpu` is not used here: this asks about WHERE THE BYTES ARE, and an
        # unattached worker must not silently reclassify a CPU-placed layer as device.
        kind = seam.kind
        is_host = kind is StackKind.HOST
        is_cpu = kind is StackKind.CPU
        host_layers += is_host
        cpu_layers += is_cpu
        device_layers += not (is_host or is_cpu)
        if is_cpu and not seam.computes_on_cpu:
            raise InterpositionError(
                f"seam {path!r} is CPU-placed but carries no CPU worker. `MoELayer.forward` tests "
                f"`computes_on_cpu`, so this layer would fall through to `resolve()` — which "
                f"refuses — and the serve would die on its first token. bind_plan attaches the "
                f"worker in the bind loop; reaching here means the seam was bound by another path."
            )
        for name, t in seam.live_tensors():
            nbytes = t.numel() * t.element_size()
            # The residency CHECKS below still run for every tensor of every layer — an aliased
            # tensor in the wrong tier is exactly as wrong as an unaliased one. Only the byte and
            # tensor COUNTS are deduped.
            first = t.data_ptr() not in seen_ptrs
            seen_ptrs.add(t.data_ptr())
            aliased += not first
            # Counted bytes, which is 0 for an alias. `nbytes` itself stays the TRUE size, because
            # `owns_pointer(ptr, nbytes)` is a range containment test and handing it 0 would ask a
            # different, weaker question of exactly the tensors the stream tier introduced.
            billed = nbytes if first else 0
            if is_host:
                host_tensors += first
                host_bytes += billed
                if owns_pointer is not None and not owns_pointer(t.data_ptr(), nbytes):
                    raise InterpositionError(
                        f"{name}: this layer is HOST-placed, but the tensor its kernels will read "
                        f"at 0x{t.data_ptr():x} ({nbytes} B) is NOT inside the pinned arena. The "
                        f"bytes are in VRAM the capacity plan counts as free, AND "
                        f"`model_memory_correction()` will subtract them from the model term as "
                        f"though they were host RAM — so the KV pool is oversized by twice this "
                        f"tensor and the failure surfaces later as an unrelated OOM."
                    )
            elif is_cpu:
                cpu_tensors += first
                cpu_bytes += billed
                # The DIRECT claim for this tier, and it needs no arena: a CPU-compute layer's
                # weights are read by AVX-512 cores, so they must be on the CPU device. If one is
                # still a HIP tensor the bake silently did not move it, the bytes are on the card
                # the plan believes it freed, and `cpu_worker.submit` would hand the native kernel
                # a device pointer — a fault or, worse, garbage read through a stale mapping.
                if getattr(t, "device", None) is not None and t.device.type != "cpu":
                    raise InterpositionError(
                        f"{name}: this layer is CPU-COMPUTE placed but its tensor is on "
                        f"{t.device}, not host RAM. The CPU tier's entire claim is that these "
                        f"{nbytes} B never occupy VRAM and never cross PCIe; here they still do, "
                        f"and both the device budget and the pinned-arena reservation were "
                        f"computed as though they had moved."
                    )
                if owns_pointer is not None and owns_pointer(t.data_ptr(), nbytes):
                    raise InterpositionError(
                        f"{name}: this layer is CPU-COMPUTE placed but its tensor lives inside the "
                        f"PINNED arena. CPU-tier bytes are meant to be pageable and charged only "
                        f"against MemAvailable; sitting in the arena means they are also charged "
                        f"against the hipHostMalloc ceiling that the host tier's capacity is "
                        f"planned from, so the arena is over-subscribed by this tensor."
                    )
            else:
                device_tensors += first
                device_bytes += billed
                if owns_pointer is not None and owns_pointer(t.data_ptr(), nbytes):
                    raise InterpositionError(
                        f"{name}: this layer is DEVICE-placed but its tensor lives INSIDE the host "
                        f"arena. It is being streamed over PCIe on every step while the plan bills "
                        f"it against the device tier — the device tier and the KV pool are both "
                        f"sized wrong, and the layer is silently ~10x slower."
                    )
    proof = SeamResidencyProof(
        moe_layers=moe,
        host_layers=host_layers,
        device_layers=device_layers,
        host_tensors=host_tensors,
        host_bytes=host_bytes,
        device_tensors=device_tensors,
        device_bytes=device_bytes,
        aliased_tensors=aliased,
        pointer_checked=owns_pointer is not None,
        cpu_layers=cpu_layers,
        cpu_tensors=cpu_tensors,
        cpu_bytes=cpu_bytes,
    )
    # HOST **or** CPU. The guard's question is "did anything actually leave the card", and under
    # three-tier planning a CPU-only plan answers yes with `host_layers == 0` — it is the strongest
    # form of the feature, not a vacuous pass. Keeping the old `host_layers == 0` test would have
    # made an all-CPU plan unbootable while an all-DEVICE plan (the regression this exists to catch)
    # stayed just as detectable, because that one has zero on both counters.
    if require_host_layers and (host_layers + cpu_layers) == 0:
        raise InterpositionError(
            "weight offload is ENABLED but not one MoE layer on the live model is host-resident or "
            f"CPU-computed — every one of them stayed on the device: {proof.describe()}. The arena "
            "was pinned and the boot banner reported a plan, and the serving path is byte-for-byte "
            "the non-offloaded one. This is the dispatch regression an engaged() ledger exists to "
            "make visible; refusing rather than serving it."
        )
    return proof


def seam_summary(seams: Sequence[MoEWeightSeam]) -> str:
    host = [s for s in seams if s.bound and s.kind is StackKind.HOST]
    dev = [s for s in seams if s.bound and s.kind is StackKind.DEVICE]
    # Counted separately, never folded into `host`: those bytes are PINNED and these are PAGEABLE,
    # and only the first kind is charged against the hipHostMalloc ceiling. A summary that merged
    # them would report an arena that was never reserved.
    cpu = [s for s in seams if s.bound and s.kind is StackKind.CPU]
    gib = 1 << 30
    host_b = sum(s.w13_spec.total_bytes + s.w2_spec.total_bytes for s in host)
    dev_b = sum(s.w13_spec.total_bytes + s.w2_spec.total_bytes for s in dev)
    cpu_b = sum(s.w13_spec.total_bytes + s.w2_spec.total_bytes for s in cpu)
    granule = seams[0].w13_spec.granule_bytes + seams[0].w2_spec.granule_bytes if seams else 0
    cpu_txt = (
        f", {len(cpu)} cpu-compute ({cpu_b / gib:.2f} GiB pageable, "
        f"workers={sum(1 for s in cpu if s.computes_on_cpu)}/{len(cpu)})"
        if cpu
        else ""
    )
    return (
        f"MoE seams: {len(seams)} layers, {len(host)} host ({host_b / gib:.2f} GiB), "
        f"{len(dev)} device ({dev_b / gib:.2f} GiB){cpu_txt}, "
        f"granule={granule / (1 << 20):.3f} MiB, "
        f"frozen={all(s.frozen for s in seams) if seams else False}"
    )
