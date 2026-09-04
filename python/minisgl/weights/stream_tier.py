"""The THIRD weight tier: expert rows re-read from the checkpoint immediately before they are used.

WHY A THIRD TIER EXISTS AT ALL
------------------------------
`plan.py` decides between exactly two homes for an expert stack: VRAM, or pinned host RAM reachable
by the MoE kernel over PCIe. On the target checkpoint that choice is not sufficient and the shortfall
is arithmetic, not a tuning failure:

    RadixArk/Qwen3.8-Flash-Next-NVFP4, 48 layers -> 70.31 GiB of routed experts.
    One 15.92 GiB card already holds a 9.22 GiB non-expert body, so the device tier caps near
    5 GiB. The host tier is then ~65.9 GiB/rank against a live MemAvailable of ~59-65 GiB on this
    box, of which `host_capacity.MEM_AVAILABLE_FLOOR` reserves 12 GiB. `HostArenaCapacityError`
    refuses by ~15.5 GiB, before a page is pinned, and no chunk size closes that.

There is no assignment of {device, pinned host} that fits, so this tier is not an optimisation: it is
the only way that checkpoint boots at TP=1. It is also the tier with a genuinely different cost
model — the other two are paid ONCE at boot, this one is paid on every forward — so it is kept
separate rather than folded into the placement plan's `StackKind`.

WHAT IT DOES
------------
Every stream layer's expert containers are ALIASED onto ONE shared set of op buffers, and the rows
the layer's tokens actually route to are read from the checkpoint and written into that set
immediately before the layer's MoE kernel runs. `num_experts_per_tok` is 10 of 512, so a decode step
reads ~29 MiB of a 1.465 GiB stack — the 51x that makes this viable at all. The tier owns the
aliasing, the poisoning and the route; a model-family `ExpertRowSource` owns "read expert e of layer
L off disk and hand it back in this container's op layout".

THE FAILURE MODE THAT MATTERS IS SILENT
---------------------------------------
Every stream layer points at the same buffers, so a row this tier does not stage holds a DIFFERENT
LAYER'S expert — real weights, right shape, right magnitude. A route disagreement, an off-by-one in
the layer->shard mapping, or a kernel that touches an expert it was not routed would then produce
fluent, plausible, wrong text with no error anywhere. Three guards, all cheap enough to be
unconditional:

  * every row outside the live set is filled with NaN, so reading an unstaged expert produces NaN
    logits instead of a quotable number;
  * `staged_layer` records which layer is physically in the buffers at the copy, and the hooked
    forward re-checks it at the use;
  * the route is taken from `quant.kernels._route_align` — the op the served path routes with — and
    unioned with the aligner's own block->expert map, not from a torch `softmax().topk()`. Those two
    disagree on exact bf16 ties at the k-th boundary, which are common over 512 experts, and the
    disagreement leaves the expert the GEMM reads unstaged. Measured on this checkpoint: layer 0,
    token row 5, experts 324 and 366 both at logit -5.09375 straddling k=10.

GRAPH CAPTURE IS UNDISCHARGED. This tier issues file I/O, allocation and `index_copy_` from inside
`MoELayer.forward`; all three are illegal under HIP graph capture. It is usable today only with
`--cuda-graph-max-bs 0`, and `install_hooks` says so. Making it capturable is a real design problem
(the read is data-dependent on the route), not an oversight to be patched later in silence.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Iterable, Mapping, Optional, Protocol, Sequence, Tuple

import torch

from .moe_interpose import InterpositionError, _container_attrs

__all__ = [
    "ExpertRowSource",
    "ExpertStreamTier",
    "StreamTierError",
    "layer_index_of_path",
]


class StreamTierError(RuntimeError):
    """The stream tier cannot honour its own invariants. Never downgraded to a warning."""


def layer_index_of_path(path: str) -> int:
    """`"model.layers.31.mlp.experts"` -> 31.

    Structural, and the ONE place a seam path is turned back into a layer index. A mismatch here
    stages the wrong layer's experts, which is numerically silent, so it raises rather than
    defaulting.
    """
    parts = path.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                break
    raise StreamTierError(f"cannot recover a layer index from the MoE path {path!r}")


class ExpertRowSource(Protocol):
    """Reads named experts of one layer and returns them IN THE CONTAINER'S OP LAYOUT.

    The op layout — not the checkpoint layout — because the tier writes the result straight into the
    buffers `post_load()` produced, with `index_copy_`. Re-deriving the layout here is how a change
    to `post_load` silently stops applying to the streamed layers, so an implementation must call the
    container's own conversion rather than transcribe it.

    `gather(layer, expert_ids)` returns `{container_attr: {component_name: tensor}}`, each tensor
    stacked over `len(expert_ids)` on dim 0 in the given order. The tier asserts the key set against
    the layer's `granule_specs()`, so a source that forgets a component raises instead of leaving a
    stale row.
    """

    #: Cumulative bytes read off the checkpoint. Diagnostic; the tier only ever reads it.
    bytes_read: int

    def gather(
        self, layer: int, expert_ids: Sequence[int]
    ) -> Mapping[str, Mapping[str, torch.Tensor]]: ...

    def close(self) -> None: ...


class ExpertStreamTier:
    """Alias N MoE layers onto one shared op-buffer set and refill the routed rows per forward.

    Construction order, and it is not negotiable: `adopt()` every layer as the chunked load finalizes
    it (so the peak device cost is ONE layer's buffers, not the tier's), then `arm()` once the load
    is done, then `install_hooks()`. `arm()` is what poisons the buffers; a tier that skipped it
    would serve the donor layer's experts to every other layer for exactly one forward.
    """

    #: How many expert rows are read, converted and written per pass. NOT a tuning knob — it is the
    #: bound on the DEVICE TRANSIENT the gather allocates, and that transient is the one device cost
    #: this tier adds that `_determine_num_pages` cannot see (it happens per forward, long after the
    #: KV pool is sized). Unbounded, it scales with the number of DISTINCT experts a batch routes,
    #: which at prefill is not `top_k` but the union over every token: a 45-token chunk routed ~95 of
    #: 512 experts here and the resulting 278 MiB `torch.stack` OOM'd a card with 192 MiB free, on a
    #: boot whose decode steps had been running fine. 32 rows is ~94 MiB on this checkpoint and costs
    #: nothing measurable — the pass is dominated by the disk read either way.
    GATHER_BATCH = 32

    def __init__(self, source: ExpertRowSource, *, device: torch.device, log=None,
                 gather_batch: int = 0) -> None:
        self.source = source
        self.device = device
        self.gather_batch = int(gather_batch) if gather_batch > 0 else self.GATHER_BATCH
        self._log = log or (lambda _m: None)
        #: layer index -> the `MoELayer` op
        self.ops: "Dict[int, Any]" = {}
        #: container attr -> component name -> the ONE shared stacked tensor
        self.shared: "Dict[str, Dict[str, torch.Tensor]]" = {}
        #: container attr -> component name -> True when that component is float (poisonable)
        self._poisonable: "Dict[str, Dict[str, bool]]" = {}
        self.donor: Optional[int] = None
        self.staged_layer: int = -1
        self._live: "set[int]" = set()
        self.armed = False
        # -- counters, read by the harnesses and by `describe()` -----------------------------
        self.stages = 0
        self.experts_staged = 0
        self.seconds = 0.0
        self.calls: "list[tuple[int, int, int]]" = []  # (layer, n_experts, bytes) per staging
        #: The largest DISTINCT-expert count any single staging had to cover. At decode it is
        #: `top_k`; at prefill it is the union over the chunk's tokens, which is the number that
        #: sizes the transient. Recorded because it is not derivable from config.
        self.max_experts_staged = 0

    # -- construction -----------------------------------------------------------------------

    def adopt(self, path: str, op: Any, specs: Mapping[str, Any]) -> str:
        """Take one finalized MoE layer into the tier. Returns a one-word placement name.

        `specs` is the layer's `granule_specs()` — the same descriptors the seam captured — so the
        component list is read off the container that exists rather than transcribed per quant
        format. The first layer adopted DONATES its buffers; every later one is pointed at them and
        its own stack is dropped on the spot.
        """
        lid = layer_index_of_path(path)
        if lid in self.ops:
            raise StreamTierError(f"layer {lid} adopted twice")
        if bool(getattr(op, "enable_ep", False)):
            raise StreamTierError(
                f"layer {lid} is EP-sharded. The stream tier reads GLOBAL expert ids off the "
                f"checkpoint and writes them at the same index into a container whose dim 0 is this "
                f"rank's SHARD, so every row would be the wrong expert with no error. EP + stream "
                f"needs the id remap written first."
            )
        attrs = _container_attrs(op)
        missing = [a for a in attrs if a not in specs]
        if missing:
            raise StreamTierError(
                f"layer {lid}: granule_specs() does not describe {missing}, so the tier cannot know "
                f"which tensors travel with an expert."
            )

        donating = not self.shared
        for attr in attrs:
            cont = getattr(op, attr)
            spec = specs[attr]
            if spec.replicated:
                raise StreamTierError(
                    f"layer {lid}.{attr} has replicated (non-per-expert) tensors "
                    f"{[r.name for r in spec.replicated]}. Those are shared across the whole stack, "
                    f"so aliasing every layer onto one set would serve the donor layer's copy to all "
                    f"of them and a routed gather cannot refresh them. Refusing rather than "
                    f"streaming a layer whose non-expert state is another layer's."
                )
            if donating:
                self.shared[attr] = {}
                self._poisonable[attr] = {}
                for comp in spec.components:
                    t = getattr(cont, comp.name)
                    self.shared[attr][comp.name] = t
                    self._poisonable[attr][comp.name] = bool(t.is_floating_point())
                continue
            for comp in spec.components:
                src = self.shared[attr].get(comp.name)
                if src is None:
                    raise StreamTierError(
                        f"layer {lid}.{attr} declares component {comp.name!r} that the donor layer "
                        f"{self.donor} does not. The layers are not the same shape, so aliasing "
                        f"would read the wrong bytes."
                    )
                t = getattr(cont, comp.name)
                if t.shape != src.shape or t.dtype != src.dtype:
                    raise StreamTierError(
                        f"layer {lid}.{attr}.{comp.name}: {tuple(t.shape)}/{t.dtype} vs the donor's "
                        f"{tuple(src.shape)}/{src.dtype}. Aliasing would read the wrong bytes with "
                        f"no error."
                    )
                # Rebind every name the container knows this tensor by, not only the canonical one:
                # a container that also exposes `weight` for `_w_op` would otherwise keep a live
                # reference to its own 1.4 GiB stack and the tier would free nothing.
                for name in comp.names:
                    if hasattr(cont, name):
                        setattr(cont, name, src)

        self.ops[lid] = op
        if donating:
            self.donor = lid
            return "stream(donor)"
        return "stream"

    def arm(self) -> None:
        """Poison every row. Call once, after the load and before the first forward."""
        if not self.shared:
            raise StreamTierError("arm() with no adopted layers")
        for attr, comps in self.shared.items():
            for name, t in comps.items():
                if self._poisonable[attr][name]:
                    t.fill_(float("nan"))
        if not any(any(v.values()) for v in self._poisonable.values()):
            raise StreamTierError(
                "no floating-point component in any streamed container, so an unstaged expert "
                "cannot be poisoned and reading one would be numerically silent. Refusing."
            )
        self._live = set()
        self.staged_layer = -1
        self.armed = True

    # -- the per-forward work ---------------------------------------------------------------

    def stage(self, layer: int, expert_ids: Iterable[int]) -> None:
        """Write layer `layer`'s routed experts into the shared buffers; poison what they replace."""
        if not self.armed:
            raise StreamTierError("stage() before arm() — the buffers still hold the donor layer")
        t0 = time.time()
        bytes0 = int(self.source.bytes_read)
        want = sorted({int(i) for i in expert_ids})
        if not want:
            raise StreamTierError(f"layer {layer}: empty route, nothing to stage")

        stale = self._live - set(want)
        if stale:
            idx = torch.tensor(sorted(stale), dtype=torch.long, device=self.device)
            for attr, comps in self.shared.items():
                for name, t in comps.items():
                    if self._poisonable[attr][name]:
                        t.index_fill_(0, idx, float("nan"))
        # Cleared BEFORE the read, so a source that raises mid-gather leaves the tier claiming
        # nothing is live rather than claiming rows it never wrote.
        self._live = set()
        self.staged_layer = -1

        # In BATCHES, and the batch is the transient bound — see `GATHER_BATCH`. Each slice is read,
        # converted, written and dropped before the next is read, so the peak is one slice however
        # many experts the batch routed. Every row of `want` is written exactly once either way, so
        # the live set below is the same set a single-shot gather would produce.
        for lo in range(0, len(want), self.gather_batch):
            part = want[lo : lo + self.gather_batch]
            rows = self.source.gather(layer, part)
            idx = torch.tensor(part, dtype=torch.long, device=self.device)
            for attr, comps in self.shared.items():
                got = rows.get(attr)
                if got is None:
                    raise StreamTierError(
                        f"layer {layer}: the row source returned nothing for container {attr!r}"
                    )
                if set(got) != set(comps):
                    raise StreamTierError(
                        f"layer {layer}.{attr}: the row source returned {sorted(got)} but the "
                        f"container holds {sorted(comps)}. A missing component leaves a stale row "
                        f"from another layer under a live expert id."
                    )
                for name, dst in comps.items():
                    src = got[name]
                    if src.shape[0] != len(part) or tuple(src.shape[1:]) != tuple(dst.shape[1:]):
                        raise StreamTierError(
                            f"layer {layer}.{attr}.{name}: source gave {tuple(src.shape)}, the op "
                            f"buffer is {tuple(dst.shape)} and {len(part)} experts were asked for."
                        )
                    dst.index_copy_(0, idx, src.to(dst.dtype) if src.dtype != dst.dtype else src)
            del rows, idx

        self._live = set(want)
        self.staged_layer = layer
        self.stages += 1
        self.experts_staged += len(want)
        self.max_experts_staged = max(self.max_experts_staged, len(want))
        self.calls.append((layer, len(want), int(self.source.bytes_read) - bytes0))
        self.seconds += time.time() - t0

    def assert_staged(self, layer: int) -> None:
        if self.staged_layer != layer:
            raise StreamTierError(
                f"the shared expert buffers hold layer {self.staged_layer} but layer {layer} is "
                f"about to run. This is the silent-garbage failure mode: identical shapes, "
                f"plausible logits, another layer's weights."
            )

    def route_ids(self, op: Any, router_logits, topk_ids) -> "list[int]":
        """The GLOBAL expert ids this layer's MoE kernel will dereference for this batch.

        From `quant.kernels._route_align`, the op the served e2m1 path routes with, NOT from a torch
        `softmax().topk()` — see the module docstring for the measured tie disagreement. The block ->
        expert map the grouped GEMM iterates is unioned in rather than trusted alone, so a future
        aligner that emits a padding block under some expert id stages it instead of reading poison.
        """
        if topk_ids is not None:
            return [int(i) for i in topk_ids.flatten().tolist()]
        if router_logits is None:
            raise StreamTierError(
                "the MoE layer was called with neither router_logits nor a precomputed topk_ids, so "
                "the stream tier cannot know which experts to read."
            )
        from minisgl.quant import kernels as qk

        # `block_m` only sets how many BLOCKS the aligner emits; the SET of experts with at least one
        # token is block-size independent, so this need not track the layer's own choice.
        _, ids, _, expert_ids, ntp = qk._route_align(
            router_logits.contiguous(),
            int(op.top_k),
            bool(op.renormalize),
            int(op.num_experts),
            16,
        )
        live = int(ntp.item()) // 16
        return ids.flatten().tolist() + expert_ids[:live].tolist()

    def install_hooks(self) -> None:
        """Stage immediately before each streamed layer's MoE kernel runs.

        Hooked on `MoELayer.forward`, which is the SAME object `discover_moe_layers` returns and the
        seam binds to — so the interposition point is the one the rest of this package already
        reasons about, and the route arrives as an argument instead of being recomputed from a gate
        this tier would have to find. An instance attribute shadows the bound method, so no model
        file is edited: a streaming hack that lives in the model is a streaming hack that ships.

        NOT CAPTURABLE. The body reads files, allocates and syncs; under HIP graph capture all three
        are illegal. Serve with `--cuda-graph-max-bs 0` until that is solved.
        """
        if not self.armed:
            raise StreamTierError("install_hooks() before arm()")
        for lid, op in sorted(self.ops.items()):
            inner = op.forward

            def streamed_forward(hidden_states, router_logits=None, *args, _lid=lid, _op=op,
                                 _inner=inner, **kwargs):
                self.stage(_lid, self.route_ids(_op, router_logits, kwargs.get("topk_ids")))
                self.assert_staged(_lid)
                return _inner(hidden_states, router_logits, *args, **kwargs)

            op.forward = streamed_forward

    # -- reporting --------------------------------------------------------------------------

    def stats(self) -> "Dict[str, Any]":
        return {
            "stream_layers": sorted(self.ops),
            "stream_donor_layer": self.donor,
            "stream_stages": self.stages,
            "stream_experts_staged": self.experts_staged,
            "stream_bytes_read": int(self.source.bytes_read),
            "stream_seconds": round(self.seconds, 2),
            "stream_max_experts_per_staging": self.max_experts_staged,
            "stream_gather_batch": self.gather_batch,
            "stream_shared_bytes": sum(
                t.numel() * t.element_size()
                for comps in self.shared.values()
                for t in comps.values()
            ),
        }

    def describe(self) -> str:
        s = self.stats()
        gib = s["stream_shared_bytes"] / float(1 << 30)
        return (
            f"stream tier: {len(self.ops)} layer(s) aliased onto one {gib:.3f} GiB buffer set "
            f"(donor layer {self.donor}), {self.stages} staging(s), "
            f"{self.experts_staged} expert-rows, {s['stream_bytes_read'] / 1e9:.2f} GB read, "
            f"{self.seconds:.1f} s"
        )

    def close(self) -> None:
        try:
            self.source.close()
        except Exception:  # pragma: no cover - close must not mask a real failure
            pass
