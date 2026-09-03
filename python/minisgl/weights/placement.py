"""LAYER-GRANULAR placement: which MoE layers live on the device stack, which on the host stack.

THE DECISION THIS FILE ENCODES, AND THE MEASUREMENT THAT FORCED IT
    P2prime (2026-09-03, both cards, `docs/measurements/WEIGHT_OFFLOAD_2026-09-02/p2prime.json`)
    built the real thing — a device `hipMalloc` stack plus a pinned `hipHostMalloc(Mapped)` stack
    with a per-expert `__constant__` pointer table, on the SHIPPED decode-GEMV core — and measured
    the per-layer time against miss count:

      * `cliff_index` **0.090** (card 0) / **0.094** (card 1) against a pure-linear 0.100 →
        **LINEAR**. One extra host-resident expert costs exactly one granule moved at that card's
        measured host bandwidth (29.7 vs 28.92 GB/s = 102.8% card 0; 14.66 vs 14.47 = 101.3%
        card 1). The `h^10` all-resident-layer cliff is **REFUTED**.
      * And precisely *because* the curve is linear, per-expert and layer-granular placement are
        provably equivalent at equal byte budget. Measured
        `per_expert_gain_over_layer_granular` peaks at **1.063x** (h=0.75) and is **1.013x /
        1.009x at h≈0.25** — the band a 16 GB card against a 68.8 GiB model actually forces.

    So a per-expert device tier buys about **1%** at the reachable operating point, in exchange for
    a route change, a `slot_of`, a second stack in the hot path and a kernel change. **It is not
    built.** Placement is per LAYER, and this module is the whole of it.

WHY THE PLANNER IS A PURE INTEGER FUNCTION
    Every TP rank must derive the identical plan. If rank 0 and rank 1 disagree about which layer is
    host-resident they hold different byte budgets, place granules at different offsets, and the
    collectives either hang or — worse — the two ranks serve different weights. So:

      * inputs are integers and a declared integer priority, never a float score, never a timing
        measurement, never a `MemAvailable` reading (`host_capacity.py` states the same rule from
        the other side: the capacity check is pass-or-raise and NEVER auto-shrinks);
      * the order is `(-priority, declaration index)`, both integers, so two ranks cannot sort
        differently;
      * `OffloadPlan.digest()` hashes the decision so a caller can prove agreement across ranks with
        one tiny collective rather than trusting the derivation.

WHY LAYERS ARE NEVER SPLIT
    Rounding down leaves at most one layer's worth of device budget unused (`unused_device_bytes`,
    always reported, never hidden). On the target shape that is <2% of the tier. Capturing it would
    need exactly the per-expert selector P2prime priced at 1%. The residual is reported so the
    trade is visible instead of assumed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Sequence

from .prior import OffloadPrior, Projection, project_step
from .stacks import ExpertStackTable, StackKind

# NO `import torch`, and deliberately no `import granule` either — `granule` pulls torch in for the
# tensor walk, and this module is the placement ARITHMETIC. Keeping it torch-free is what lets the
# decision that can be silently wrong by a factor of two be unit-tested on a machine where
# `import torch` fails, which is the state of this box's host outside the serve image.
# `LayerWeights.from_specs` is the one adapter, and it takes the two integers off a GranuleSpec.

__all__ = [
    "LayerWeights",
    "LayerPlacement",
    "OffloadPlan",
    "PlacementError",
    "distinct_experts",
    "ep_local_top_k",
    "plan_layer_granular",
    "project_plan",
    "sweep_device_fraction",
]


class PlacementError(RuntimeError):
    """The placement inputs are inconsistent. Always a boot failure, never a warning."""


def ep_local_top_k(top_k: int, ep_size: int, local_num_experts: int) -> int:
    """The routed slots ONE EP rank computes per token — the top_k `LayerWeights` must be built with.

    THE ONE IMPLEMENTATION, deliberately. `LayerWeights.num_experts` is the EP-LOCAL count
    (`MoELayer.__init__`: `local_num_experts = num_experts // ep_size`), but `MoELayer.top_k` stays
    GLOBAL — a token still routes to `k` of the `E` global experts, of which only `k/ep_size` land
    on this rank's shard. Pairing a local `E` with a global `k` is therefore not a conservative
    approximation, it is an `ep_size`-fold over-count of the layer's TRAFFIC, and it is silent:
    `distinct_experts(E/2, k, 1)` returns `k` where the rank reads `k/2` granules. Every host-byte,
    step-time and tok/s figure downstream is then wrong by that factor, in the pessimistic
    direction, and `OffloadPlan.digest()` hashes `top_k` — so two entry points that disagree about
    this also report a spurious cross-rank plan divergence.

    That is not hypothetical: `plan.size_planned_layers_from_model` computed this inline while
    `moe_interpose.MoEWeightSeam.layer_weights` passed the global `top_k` straight through, so the
    engine path and the standalone seam path produced different plans for the same model. Both now
    call this.

    CEIL, never a mean. The projection is a step-time FLOOR and the step waits for the SLOWEST rank,
    so it is set by the rank that drew more than its share of routed experts, not by the average.
    Rounding up is the conservative direction. Clamped to `[1, local_num_experts]` because
    `LayerWeights` refuses a `top_k` outside that band — an `ep_size` larger than `top_k` is legal
    (E=512, k=8, ep=16) and must yield 1, not 0.
    """
    if local_num_experts <= 0:
        raise PlacementError(f"local_num_experts must be positive, got {local_num_experts}")
    if top_k <= 0:
        raise PlacementError(f"top_k must be positive, got {top_k}")
    ep = max(1, int(ep_size))
    return min(int(local_num_experts), max(1, -(-int(top_k) // ep)))


def distinct_experts(num_experts: int, top_k: int, batch: int) -> float:
    """Expected number of DISTINCT experts a layer touches for a batch of `batch` tokens.

    `E * (1 - (1 - k/E)^M)` — the plan's §1(b) arithmetic. At M=1 it is exactly `top_k`; at M=6 on
    E=512/k=10 it is ~57, i.e. bytes *per token* fall only ~5% from M=1 to M=6. That is why this is
    a **capacity** feature and not a throughput feature, and why a caller must report the curve
    rather than quote a single batch (K0).

    Note this is the UPPER bound the planner prices against: real routing has collisions beyond the
    independence this formula assumes only in the direction of *fewer* distinct experts, so the
    projection is conservative.
    """
    if num_experts <= 0 or top_k <= 0 or batch <= 0:
        raise ValueError(f"distinct_experts({num_experts}, {top_k}, {batch}): all must be positive")
    k = min(top_k, num_experts)
    return num_experts * (1.0 - (1.0 - k / num_experts) ** batch)


def _spec_max_row_bytes(spec: Any) -> int:
    """Largest single arena row a `granule.GranuleSpec` will ask for. Duck-typed, torch-free.

    `moe_interpose._plan_items` allocates ONE row per COMPONENT (the whole stacked slab, `E` x the
    per-expert slice, because the kernels index `base + e*row_bytes` and that requires the stack to
    be contiguous) plus one row per REPLICATED tensor. So the largest row is the largest of those,
    NOT the granule and NOT the container: on the target shape it is the fused w13 weight stack.

    Falls back to `total_bytes` when a spec does not expose the component lists (a test double, a
    future descriptor). That is the whole container, which is a sound over-estimate — the packing
    bound must never be optimistic, because an optimistic one under-reserves the arena and the
    overflow rows come back as `hipMalloc` VRAM.
    """
    rows = [int(getattr(r, "nbytes", 0) or 0) for r in (getattr(spec, "replicated", ()) or ())]
    comps = getattr(spec, "components", None)
    if comps is None:
        return int(getattr(spec, "total_bytes", 0) or 0)
    n = int(getattr(spec, "num_granules", 0) or 0) or 1
    rows += [int(getattr(c, "nbytes", 0) or 0) * n for c in comps]
    return max(rows, default=int(getattr(spec, "total_bytes", 0) or 0))


@dataclass(frozen=True)
class LayerWeights:
    """One MoE layer's offloadable footprint, as an indivisible placement unit.

    `num_experts` is the LOCAL count — under EP each rank's containers are already sized to its
    expert shard on dim 0 (`MoELayer.__init__`: `local_num_experts = num_experts // ep_size`), so
    the CAPACITY arithmetic is per-rank by construction and needs no EP special case.

    `top_k` is NOT free of one. It is the routed slots THIS RANK computes per token, which under EP
    is `ceil(global_top_k / ep_size)` and not `MoELayer.top_k` — the layer keeps the global value,
    because a token still routes to `k` of the `E` global experts. Pairing a local `num_experts`
    with a global `top_k` prices `ep_size` times the granules the rank reads, silently. Build this
    through `from_specs` + `ep_local_top_k` (which is what both planners do) rather than by hand;
    the plain constructor cannot tell the two apart, exactly as it cannot for `num_experts`.

    Two byte counts, and they are NOT the same number:
      * `granule_bytes` — one expert's slice of BOTH GEMMs, excluding provably expert-invariant
        components (symmetric CT `_zeros_op` is E copies of `0x88`; every expert reads the same
        row of it, so it is read once per layer, not once per routed expert). This is the TRAFFIC
        unit and it is what a miss costs.
      * `resident_bytes` — every byte the layer occupies on whichever stack holds it, invariant
        components included. This is the CAPACITY unit and it is what the budget is spent in.
    Using either one for the other's job is a silent ~E-fold or ~3% error, so they are separate
    fields rather than one number with a comment.

    And a THIRD byte count, which is neither of those: `max_row_bytes`, the largest SINGLE
    allocation this layer will ask the arena for. It is not a capacity number and it is not a
    traffic number — it is a PACKING number, and it exists because the pinned arena is a set of
    fixed chunks that a region may never straddle (`chunk_plan.BumpAllocator`, forward-only
    next-fit: a tail the next row does not fit into is abandoned permanently). The bytes the arena
    must PIN are therefore not `ceil(payload / chunk) * chunk`; they are
    `ceil(payload / (chunk - max_row)) * chunk`, and on this feature's shape (a fused w13 weight
    stack is a ~1 GiB row in a 2 GiB chunk) the difference is tens of per cent. Charging the payload
    alone is how the reservation silently comes up short and the tail rows land in `hipMalloc`
    **VRAM** — the one outcome the whole capacity plan exists to prevent, discovered at bind time
    after the arena is pinned and the checkpoint is loaded, instead of at `reserve()` in
    milliseconds. See `chunk_plan.headroom_chunks`, whose own docstring says callers that know their
    largest row MUST pass it.

    0 means "not derived" and every consumer then falls back to `resident_bytes` (a whole layer is
    a sound upper bound on any one of its rows) via `row_bound`, so an un-populated caller gets a
    conservative answer rather than the old optimistic one.
    """

    path: str  # structural dotted path — rank-identical, never a construction counter
    num_experts: int
    top_k: int
    granule_bytes: int
    resident_bytes: int
    # Identifies WHICH weights these bytes describe. Two ranks must agree; a mismatch means one
    # repacked (`_w_rep`) and the other did not, which would give them different budgets.
    fingerprint: str = ""
    # Optional integer placement priority from a baked prior (M2). Higher = prefer device. Integer,
    # not a float score, so two ranks can never sort differently.
    priority: int = 0
    # Largest single arena row this layer allocates. See the class docstring. 0 == not derived.
    max_row_bytes: int = 0

    @property
    def row_bound(self) -> int:
        """`max_row_bytes` when known, else the whole layer — always a SOUND upper bound.

        Never 0: a caller that used 0 as "no bound" would go straight back to the `ceil(payload /
        chunk)` reservation this field exists to replace, and would do it silently."""
        return self.max_row_bytes or self.resident_bytes

    def __post_init__(self) -> None:
        if self.num_experts <= 0:
            raise PlacementError(f"{self.path}: num_experts must be positive")
        if self.top_k <= 0 or self.top_k > self.num_experts:
            raise PlacementError(
                f"{self.path}: top_k={self.top_k} is not in [1, {self.num_experts}]"
            )
        if self.granule_bytes <= 0 or self.resident_bytes <= 0:
            raise PlacementError(
                f"{self.path}: granule_bytes={self.granule_bytes} / "
                f"resident_bytes={self.resident_bytes} must both be positive"
            )
        if self.resident_bytes < self.granule_bytes * self.num_experts:
            raise PlacementError(
                f"{self.path}: resident_bytes={self.resident_bytes} is smaller than "
                f"{self.num_experts} x granule_bytes={self.granule_bytes}. The capacity number "
                f"cannot be less than the traffic number times the expert count — one of the two "
                f"was derived against the wrong expert count (global vs EP-local shard)."
            )
        if self.max_row_bytes < 0 or self.max_row_bytes > self.resident_bytes:
            raise PlacementError(
                f"{self.path}: max_row_bytes={self.max_row_bytes} is not in [0, "
                f"{self.resident_bytes}]. A single arena row cannot be larger than the layer it "
                f"belongs to; a bigger figure means it was derived against a different container "
                f"set, and the arena would be reserved against a packing bound for weights this "
                f"rank is not holding."
            )

    @classmethod
    def from_specs(
        cls,
        path: str,
        *,
        num_experts: int,
        top_k: int,
        w13: Any,
        w2: Any,
        priority: int = 0,
    ) -> "LayerWeights":
        """Adapt a `granule.GranuleSpec` pair. The only place placement touches the tensor layer."""
        if w13.num_experts != num_experts:
            raise PlacementError(
                f"{path}: granule spec says n={w13.num_experts} but the layer reports "
                f"local_num_experts={num_experts}. Under EP the container is sized to the LOCAL "
                f"shard; a mismatch means the spec was derived against the global count and every "
                f"byte figure would be off by ep_size."
            )
        h = hashlib.sha256()
        h.update(f"{path}|{w13.fingerprint()}|{w2.fingerprint()}".encode())
        return cls(
            path=path,
            num_experts=num_experts,
            top_k=top_k,
            granule_bytes=w13.granule_bytes + w2.granule_bytes,
            resident_bytes=w13.total_bytes + w2.total_bytes,
            fingerprint=h.hexdigest()[:16],
            priority=priority,
            max_row_bytes=max(_spec_max_row_bytes(w13), _spec_max_row_bytes(w2)),
        )

    def active_bytes(self, batch: int = 1) -> int:
        """Bytes this layer reads per FORWARD at batch `batch` (not per token)."""
        return int(round(distinct_experts(self.num_experts, self.top_k, batch) * self.granule_bytes))


@dataclass(frozen=True)
class LayerPlacement:
    path: str
    kind: StackKind
    resident_bytes: int
    granule_bytes: int
    num_experts: int
    top_k: int
    # Largest single arena row this layer allocates — the PACKING bound the host arena's chunk
    # reservation needs. See `LayerWeights.max_row_bytes`. 0 == not derived; `row_bound` falls back
    # to the whole layer, which is sound but loose.
    max_row_bytes: int = 0

    @property
    def row_bound(self) -> int:
        return self.max_row_bytes or self.resident_bytes

    def table(self) -> ExpertStackTable:
        """The layer's residency ledger. Uniform by construction under layer-granular placement.

        Nothing hands this to a kernel: a uniform table means the layer's containers already point
        at one stack, so the kernel's existing `w + e*row` arithmetic is correct and unchanged.
        """
        return ExpertStackTable.uniform(self.num_experts, self.kind)


@dataclass(frozen=True)
class OffloadPlan:
    """The frozen, rank-identical decision. Placement, never scheduling."""

    placements: tuple[LayerPlacement, ...]
    device_budget_bytes: int
    total_resident_bytes: int

    # -- capacity ------------------------------------------------------------------------------
    @property
    def device_resident_bytes(self) -> int:
        return sum(p.resident_bytes for p in self.placements if p.kind is StackKind.DEVICE)

    @property
    def host_resident_bytes(self) -> int:
        return sum(p.resident_bytes for p in self.placements if p.kind is StackKind.HOST)

    @property
    def max_host_row_bytes(self) -> int:
        """Largest single row the HOST arena will be asked for — the arena's packing bound.

        This is what `PinnedWeightArena.reserve(..., extra_max_region_bytes=)` needs and what
        `plan.arena_reservation_bytes` charges chunks against. Zero for an all-device plan (no arena
        is built at all), so a non-offloading serve is unchanged.

        HOST placements only: a device-resident layer never touches the arena, so charging its rows
        against the chunk bound would reserve host RAM for weights that are staying in VRAM — and at
        the operating points capacity forces, the device tier holds the LARGEST layers by
        construction (greedy fill in priority order), so including them would inflate the bound by
        exactly the layers that are not there.
        """
        return max(
            (p.row_bound for p in self.placements if p.kind is StackKind.HOST), default=0
        )

    @property
    def unused_device_bytes(self) -> int:
        """Budget left on the table because layers are indivisible.

        Reported, never hidden: this is exactly the residual a per-expert split layer would capture,
        and P2prime priced that capture at ~1% (see the module docstring). If this ever grows to a
        large fraction of the budget, the layer sizes are wildly uneven and THAT is the finding.
        """
        return max(0, self.device_budget_bytes - self.device_resident_bytes)

    @property
    def device_fraction(self) -> float:
        return (
            self.device_resident_bytes / self.total_resident_bytes
            if self.total_resident_bytes
            else 0.0
        )

    @property
    def num_device_layers(self) -> int:
        return sum(1 for p in self.placements if p.kind is StackKind.DEVICE)

    @property
    def num_host_layers(self) -> int:
        return sum(1 for p in self.placements if p.kind is StackKind.HOST)

    @property
    def is_empty(self) -> bool:
        """No layer is host-resident — the plan is a no-op and the code path costs nothing.

        The offload path is still *exercised* on every serve in this state (the seam resolves, the
        accounting runs, the plan is on the banner), which is what stops it rotting; it simply moves
        no bytes. See plan §6.2: the flag CLAMPS an automatic decision, it is never "is my feature
        enabled".
        """
        return self.num_host_layers == 0

    # -- traffic -------------------------------------------------------------------------------
    def host_active_bytes(self, batch: int = 1) -> int:
        return self._active(StackKind.HOST, batch)

    def device_active_bytes(self, batch: int = 1) -> int:
        return self._active(StackKind.DEVICE, batch)

    def _active(self, kind: StackKind, batch: int) -> int:
        total = 0
        for p in self.placements:
            if p.kind is not kind:
                continue
            total += int(
                round(distinct_experts(p.num_experts, p.top_k, batch) * p.granule_bytes)
            )
        return total

    # -- agreement -----------------------------------------------------------------------------
    def digest(self) -> str:
        """Hash of the DECISION. Two ranks must produce the same string; compare it over the CPU
        group at boot rather than trusting that the derivation was pure."""
        h = hashlib.sha256()
        h.update(f"budget={self.device_budget_bytes}|".encode())
        for p in self.placements:
            h.update(
                f"{p.path}:{int(p.kind)}:{p.resident_bytes}:{p.granule_bytes}:"
                f"{p.num_experts}:{p.top_k}|".encode()
            )
        return h.hexdigest()[:16]

    def kind_of(self, path: str) -> StackKind:
        for p in self.placements:
            if p.path == path:
                return p.kind
        raise KeyError(path)

    def describe(self) -> str:
        gib = 1 << 30
        return (
            f"weight-offload plan: {self.num_device_layers} MoE layers on DEVICE "
            f"({self.device_resident_bytes / gib:.2f} GiB), {self.num_host_layers} on HOST "
            f"({self.host_resident_bytes / gib:.2f} GiB), f={self.device_fraction:.3f}, "
            f"unused device budget {self.unused_device_bytes / gib:.2f} GiB, "
            f"digest={self.digest()}"
        )


def plan_layer_granular(
    layers: Sequence[LayerWeights],
    *,
    device_budget_bytes: int,
) -> OffloadPlan:
    """Place whole layers on the device stack until the budget is spent; the rest go host.

    Deterministic and integer-only: layers are considered in `(-priority, declaration index)` order
    and a layer is placed on the device iff its WHOLE footprint still fits. A layer that does not
    fit is skipped and the walk continues, so a run of small layers after one large one is still
    packed — the order and therefore the outcome are identical on every rank.

    `device_budget_bytes = 0` yields the pure-T1 all-host plan; a budget at or above the total
    yields the no-op plan (`is_empty`), which is the shape every model that already fits produces.
    """
    if device_budget_bytes < 0:
        raise PlacementError(f"device_budget_bytes must be >= 0, got {device_budget_bytes}")
    seen: set[str] = set()
    for lw in layers:
        if lw.path in seen:
            raise PlacementError(
                f"duplicate layer path {lw.path!r}. Paths must be STRUCTURAL and unique — a "
                f"construction counter renumbers every layer after the MTP draft head builds its "
                f"own MoELayer, which would silently place the wrong layers on the wrong stack."
            )
        seen.add(lw.path)

    order = sorted(range(len(layers)), key=lambda i: (-layers[i].priority, i))
    on_device: set[int] = set()
    used = 0
    for i in order:
        need = layers[i].resident_bytes
        if used + need <= device_budget_bytes:
            on_device.add(i)
            used += need

    placements = tuple(
        LayerPlacement(
            path=lw.path,
            kind=StackKind.DEVICE if i in on_device else StackKind.HOST,
            resident_bytes=lw.resident_bytes,
            granule_bytes=lw.granule_bytes,
            num_experts=lw.num_experts,
            top_k=lw.top_k,
            max_row_bytes=lw.max_row_bytes,
        )
        for i, lw in enumerate(layers)
    )
    return OffloadPlan(
        placements=placements,
        device_budget_bytes=device_budget_bytes,
        total_resident_bytes=sum(lw.resident_bytes for lw in layers),
    )


def project_plan(
    plan: OffloadPlan,
    prior: OffloadPrior,
    *,
    num_ranks: int,
    batch: int = 1,
    loaded: bool = False,
) -> Projection:
    """Project the decode step for this plan under P2prime's LINEAR miss-cost model.

    No cliff term, no concurrency term, no residency-probability term — P2prime refuted the first,
    P4 measured the second as free (efficiency 0.999) and layer-granular placement makes the third
    deterministic. `prior.slow_host_gbps(num_ranks)` picks the SLOW card: at TP=2 the links are
    independent so both ranks stream concurrently and the card-1 rank sets the step.

    This is a PROJECTION, not a measurement. A1.7 is measured on a served A/B with graphs on; never
    quote a number from here as achieved throughput.
    """
    if prior.miss_cost_is_linear is not True:
        raise PlacementError(
            "the prior says the measured miss-cost curve is NOT linear on this box, so this "
            "additive projection is wrong. Re-run P2prime and re-derive the model before "
            "projecting anything."
        )
    return project_step(
        host_bytes_per_rank=plan.host_active_bytes(batch),
        device_bytes_per_rank=plan.device_active_bytes(batch),
        prior=prior,
        num_ranks=num_ranks,
        loaded=loaded,
    )


@dataclass(frozen=True)
class SweepRow:
    f_requested: float
    plan: OffloadPlan
    projection: Projection

    @property
    def device_gib(self) -> float:
        return self.plan.device_resident_bytes / (1 << 30)

    @property
    def host_gib(self) -> float:
        return self.plan.host_resident_bytes / (1 << 30)


def sweep_device_fraction(
    layers: Sequence[LayerWeights],
    prior: OffloadPrior,
    *,
    num_ranks: int,
    batch: int = 1,
    grid: Sequence[float] | None = None,
) -> tuple[SweepRow, ...]:
    """The A2.4 Pareto table: tok/s against the VRAM each device fraction surrenders.

    Every GiB of device tier is ~200k KV tokens not held, so the operating point is a frontier
    choice and not a maximum. Report both axes; pick from the frontier; land the choice in
    `tools/serve.sh`'s per-model table, not only in a doc.
    """
    total = sum(lw.resident_bytes for lw in layers)
    rows = []
    for f in grid if grid is not None else prior.f_grid:
        plan = plan_layer_granular(layers, device_budget_bytes=int(total * f))
        rows.append(
            SweepRow(
                f_requested=f,
                plan=plan,
                projection=project_plan(plan, prior, num_ranks=num_ranks, batch=batch),
            )
        )
    return tuple(rows)


def format_sweep(rows: Sequence[SweepRow], prior: OffloadPrior) -> str:
    """One printable block for the `[serve]` banner and the measurement artifact."""
    out = [
        f"device-fraction sweep (prior={prior.name}, LINEAR miss cost, PROJECTED not measured):",
        "     f  dev GiB  host GiB   step ms    tok/s   vs all-host",
    ]
    base = rows[0].projection.tok_s if rows else 1.0
    for r in rows:
        out.append(
            f"  {r.f_requested:4.2f}  {r.device_gib:7.2f}  {r.host_gib:8.2f}  "
            f"{r.projection.step_ms:7.1f}  {r.projection.tok_s:7.2f}   "
            f"{r.projection.tok_s / base if base else 0.0:5.2f}x"
        )
    out.append(
        f"  gates: K4 hard-kill {prior.kill_tok_s:.3f} tok/s (llama.cpp "
        f"{prior.baseline_tok_s:.3f}/1.57); A1.7 = {prior.accept_fraction:.2f} x the ceiling. "
        f"host BW used = {prior.host_read_gbps} GB/s per card, slow rank gates."
    )
    return "\n".join(out)
