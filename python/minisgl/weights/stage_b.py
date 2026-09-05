"""Stage B — the CHUNKED load. Stream the checkpoint into placed weights, one chunk at a time.

WHY THIS EXISTS
---------------
Stage A (`bake.py` + `moe_interpose.py`) moves weights to the host arena AFTER `load_state_dict` and
`post_load` have run. That ordering has one hard prerequisite: the whole checkpoint must fit
somewhere at once. For every model this repo has shipped it does. For
`RadixArk/Qwen3.8-Flash-Next-NVFP4` it does not — 70.31 GiB of it is routed experts and the card is
15.92 GiB — so the target checkpoint OOMs *inside* `load_state_dict`, before the bake it was written
for ever runs. That is M1-E in `docs/measurements/WEIGHT_OFFLOAD_2026-09-02/M1A_STATUS.md`.

Stage B interleaves instead of sequencing:

    for each chunk:   read its shards -> fill those keys -> post_load the ops it completed
                      -> PLACE them (device stack, or copied into the pinned arena)
                      -> drop the staging tensors

so the live set is one chunk, not one checkpoint. On the target that is 1.465 GiB (one layer's 512
NVFP4 experts) plus its repack transient, instead of 70.31 GiB.

THE THREE THINGS THAT MAKE THIS MORE THAN A LOOP
------------------------------------------------
1. **A totality ledger, not a `strict=` flag.** Chunked filling means `load_state_dict` can no longer
   prove completeness by "the dict is empty at the end" — each chunk legitimately leaves most of the
   model unfilled. So this driver snapshots the model's key set BEFORE the first chunk and refuses at
   the end if any key was never filled, or was filled by two chunks. A checkpoint whose shards do not
   cover the model is exactly as loud as before, just from here.

2. **`post_load` runs per chunk and must not run twice.** A quantized MoE container's `post_load`
   *deletes* the checkpoint buffers it read, so a second call is an `AttributeError` at best. The ops
   this driver finalizes are marked `BaseOP._post_load_done`, which the default `post_load` recursion
   skips — so the caller still ends with an ordinary `model.post_load()` for everything the chunks
   did not complete (the non-expert body), and that call does not re-enter the layers already done.

3. **Placement happens INSIDE the load window, which is the first CPU store into arena pages.**
   Under Stage A every arena write is device-issued (`_bake` copies device->arena). Stage B's copies
   are still device-issued — the chunk is staged on the card and `copy_`d into the arena from there —
   *deliberately*, so the visibility discipline is unchanged and Phase 0's unknown #6 does not have
   to be re-answered for a new store path. A future variant that reads safetensors straight into
   arena pages on the CPU would be the first genuine CPU store and would need that argument made.

WHAT IS NOT HERE
----------------
Per-EXPERT-RANGE chunking. The chunk unit is a whole MoE layer (all 512 experts), which is what the
checkpoint's file layout hands us and what keeps `quant/method.py::ct_packed_sign_convention` sound
without threading anything: that decision is sampled from the FULL stack a container holds, so a
layer-granular chunk samples exactly what the one-shot load sampled. Splitting a container across
chunks would sample a prefix of it and could decide the sign convention differently for two halves of
one stack — plausible weights, no crash. If per-expert-range chunking is ever needed, `_ct_sign` must
be decided once and threaded, and this docstring is where that starts.

Graph capture: N/A here and provably so — this is boot-time file I/O and `copy_`. The seams it binds
carry the capture argument, unchanged (`moe_interpose`, `bake.verify_after_capture`).
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, Protocol, Sequence, Tuple

__all__ = [
    "LoadChunk",
    "ChunkedLoadLedger",
    "ChunkedLoadError",
    "LayerSink",
    "DeviceLayerSink",
    "SeamLayerSink",
    "ChunkedWeightLoader",
]

_GIB = 1 << 30


class ChunkedLoadError(RuntimeError):
    """The chunked load did not cover the model, or covered part of it twice."""


@dataclass(frozen=True)
class LoadChunk:
    """One unit of the checkpoint: a set of shards, and the ops those shards COMPLETE.

    `finalize_paths` are structural dotted paths in `BaseOP.state_dict`'s grammar — the same strings
    `moe_interpose.discover_moe_layers` and `plan.moe_layer_shapes` produce, so a chunk names the
    same layer the placement plan does. An empty tuple means "this chunk completes nothing on its
    own"; its ops are finalized by the trailing whole-model `post_load()`.
    """

    name: str
    files: Tuple[str, ...]
    finalize_paths: Tuple[str, ...] = ()

    def describe(self) -> str:
        return f"{self.name} ({len(self.files)} shards, finalizes {len(self.finalize_paths)})"


@dataclass
class ChunkedLoadLedger:
    """What the load actually did. Every field is measured, none is projected."""

    chunks: int = 0
    keys_filled: int = 0
    staged_bytes: int = 0
    placed_host_layers: int = 0
    placed_device_layers: int = 0
    # A CPU-COMPUTE layer is its own tier and is counted as one. Folding it into `device` (the
    # `else` branch this used to fall into) would say the card holds bytes it does not, which is
    # the one number `assert_device_accounting` independently checks the plan against.
    placed_cpu_layers: int = 0
    seconds: float = 0.0
    #: Peak `torch.cuda.memory_allocated()` observed at a chunk boundary, and the peak torch itself
    #: reports for the whole window. The second is the one that matters for "did it fit"; the first
    #: says how much of that was the steady state rather than a transient inside one repack.
    peak_device_allocated: int = 0
    peak_device_allocated_torch: int = 0
    peak_device_reserved: int = 0
    #: Smallest free VRAM seen, straight from the driver (`torch.cuda.mem_get_info`) — includes every
    #: other tenant of the card, which `memory_allocated` cannot see.
    min_device_free: int = 0
    #: Peak process RSS high-water mark (`/proc/self/status:VmHWM`). Pinned arena pages are ordinary
    #: anonymous pages held by this process, so they are IN this number.
    peak_host_rss: int = 0
    per_chunk: list = field(default_factory=list)

    def describe(self) -> str:
        return (
            f"stage-B chunked load: {self.chunks} chunks, {self.keys_filled} keys, "
            f"{self.staged_bytes / _GIB:.2f} GiB staged, "
            f"{self.placed_host_layers} host / {self.placed_device_layers} device"
            f"{f' / {self.placed_cpu_layers} cpu-compute' if self.placed_cpu_layers else ''} "
            f"layers, "
            f"peak device alloc {self.peak_device_allocated_torch / _GIB:.2f} GiB "
            f"(reserved {self.peak_device_reserved / _GIB:.2f}), "
            f"peak host RSS {self.peak_host_rss / _GIB:.2f} GiB, {self.seconds:.1f} s"
        )


# ---------------------------------------------------------------------------------------------
# Sinks — where a finalized layer goes
# ---------------------------------------------------------------------------------------------


class LayerSink(Protocol):
    """Called with each op a chunk finalized, in chunk order, while the staging tensors are alive.

    Returns the placement it made, for the ledger. Raising aborts the load — which is the correct
    behaviour: a sink that cannot place layer 7 will not be able to place layer 8 either, and
    continuing would leave a model that is half-placed and indistinguishable from a correct one.
    """

    def place(self, path: str, op: Any) -> str: ...

    def finish(self) -> None: ...


class DeviceLayerSink:
    """Leave everything where `load_state_dict` put it. The no-offload control, and zero copies.

    Not a stub: this is what a serve with no host tier does, and it is the leg every numerics gate
    compares against, so it has to exist as a real sink rather than as `if sink is None`.
    """

    def __init__(self) -> None:
        self.placed: list[str] = []

    def place(self, path: str, op: Any) -> str:
        self.placed.append(path)
        return "device"

    def finish(self) -> None:
        return None


class SeamLayerSink:
    """Bind one MoE layer's `MoEWeightSeam` the moment the chunked load finalizes that layer.

    This is Stage A's bake, executed per layer inside the load window instead of once after it. It
    reuses `moe_interpose.bind_seam` — the SAME function `bind_plan` calls — so there is exactly one
    implementation of "move a layer's weights and prove they arrived", and the chunked path cannot
    drift from the one-shot path's validate-before-copy / bitwise-read-back / weakref-leak-proof
    order.

    The plan is consulted by path, and a discovered layer the plan does not name is bound DEVICE,
    exactly as `bind_plan` does it and for the same reason (the resolver excludes layers by policy).
    `finish()` refuses if the plan named a layer no chunk ever delivered — a planned layer that never
    arrived means the arena reservation, the device tier and the KV budget were all computed for a
    model this process is not running.

    A STREAM-TIER layer is adopted here too, and the ordering is deliberate: bind the seam DEVICE
    first (zero bytes moved, the seam's identity check still installed), THEN alias the containers
    onto the tier's shared buffers. Every discovered layer therefore carries a seam whatever tier it
    lands in, so `prove_seam_residency` and the `engaged()` ledger stay honest about how many layers
    each arm serves, and the aliasing happens INCREMENTALLY — the peak device cost of the stream tier
    is one layer's op buffers, not the whole tier's.
    """

    def __init__(
        self,
        plan: Any,
        allocator: Any,
        *,
        selftest: int | None = None,
        stream: Any = None,
        stream_layers: "Sequence[int] | None" = None,
        cpu_worker: Any = None,
        cpu_worker_factory: Any = None,
    ) -> None:
        from .moe_interpose import SELFTEST_SAMPLE, BindOutcome
        from .stacks import StackKind

        self.plan = plan
        self.allocator = allocator
        self.selftest = SELFTEST_SAMPLE if selftest is None else int(selftest)
        self._kinds = {p.path: p.kind for p in plan.placements}
        self._host = StackKind.HOST
        self._cpu = StackKind.CPU
        self.outcome = BindOutcome(plan_digest=plan.digest())
        self.seams: list[Any] = []
        self._seen: set[str] = set()
        self.stream = stream
        self.stream_layers = frozenset(int(i) for i in (stream_layers or ()))
        # THE CPU-COMPUTE TIER, on the chunked path. `bind_plan` carries the same argument and the
        # same refusal; this class is the OTHER implementation of "bind a planned layer", and the
        # tier has to exist in both or a chunked load silently binds CPU layers as DEVICE.
        self.cpu_worker = cpu_worker
        # A FACTORY, not just an instance, because the worker's shapes are not in the plan.
        # `NativeVnniBackend` is opened for ONE (hidden, inter, top_k) and the plan carries bytes
        # and expert counts, not dimensions. The first CPU layer to arrive HAS the live containers,
        # so the shapes are read off them rather than transcribed from config — the same generality
        # argument `resolve_weight_plan(model=)` makes one layer up.
        self.cpu_worker_factory = cpu_worker_factory
        #: Running expert count over CPU layers IN PLAN ORDER — the seam's `backend_expert_offset`.
        #: `bind_plan` derives it by iterating `plan.placements`; here layers arrive in LOAD order,
        #: so it is taken from the plan up front rather than from arrival, or the two paths would
        #: assign different offsets to the same layer and the backend lookup would cross layers.
        self._cpu_offset: "dict[str, int]" = {}
        off = 0
        for p in plan.placements:
            if p.kind is self._cpu:
                self._cpu_offset[p.path] = off
                off += int(p.num_experts)
        if self._cpu_offset and cpu_worker is None and cpu_worker_factory is None:
            raise ValueError(
                f"the plan places {len(self._cpu_offset)} layer(s) on the CPU-COMPUTE tier but the "
                f"chunked load's sink got no `cpu_worker`/`cpu_worker_factory`. A CPU seam without "
                f"one is not a degraded mode: `MoELayer.forward` reaches `cpu_forward` with "
                f"nothing to submit to and the "
                f"serve dies on its first token instead of at boot."
            )
        #: path -> the tier it actually landed in. Read by the harnesses; a placement that
        #: disagrees with the plan is a boot failure, never a log line.
        self.placed: "dict[str, str]" = {}
        if self.stream_layers and self.stream is None:
            raise ValueError(
                "stream_layers named without a stream tier: those layers would silently stay "
                "device-resident and the boot would OOM at the layer count the tier was added for."
            )

    def place(self, path: str, op: Any) -> str:
        from .moe_interpose import InterpositionError, attach_seam, bind_seam
        from .stacks import StackKind
        from .stream_tier import layer_index_of_path

        planned = path in self._kinds
        kind = self._kinds.get(path, StackKind.DEVICE)
        streamed = bool(self.stream_layers) and layer_index_of_path(path) in self.stream_layers
        if streamed and kind is self._host:
            raise InterpositionError(
                f"{path} is in the stream tier but the plan placed it HOST. The tier assignment and "
                f"the plan disagree, so the arena holds a reservation for a layer that will never "
                f"occupy it and the KV pool was sized against that reservation."
            )
        seam = attach_seam(path, op)
        self.seams.append(seam)
        bind_seam(
            seam,
            kind,
            # The CPU tier needs the allocator too. `TorchStackAllocator.alloc_like(CPU)` returns
            # plain PAGEABLE `torch.empty` — no arena, no `hipHostMalloc`, no HIP at all — but it
            # is still the allocator that hands it out, and passing None here would make a CPU
            # placement raise `HostStackUnavailable` on the chunked path only.
            self.allocator if kind in (self._host, self._cpu) else None,
            selftest=self.selftest,
            out=self.outcome,
            # A streamed layer's bytes are NOT the plan's device tier: all of them alias one shared
            # buffer set, so counting each would bill the tier N times over and
            # `assert_device_accounting` would fail against a plan that never named them.
            count_device_bytes=planned and not streamed,
        )
        if kind is self._cpu:
            # Attach the worker and TILE this layer's bytes here, one layer after its own bake,
            # for two reasons that are both about this being the chunked path:
            #   * `attach_cpu_worker` is a placement entry point and `seal()` freezes every seam, so
            #     a caller that waits until the load finishes is attaching to a frozen seam.
            #   * the tiling is a permutation over ~750 MiB/layer that has just been written, so
            #     doing it now is the only point at which those bytes are cache-warm.
            offset = self._cpu_offset[path]
            if self.cpu_worker is None:
                # Shapes off the LIVE containers: w13 is (E, 2I, H/8) int32 after post_load, so the
                # hidden size and the (TP-sharded) intermediate size are both readable here and
                # neither has to be transcribed from config.
                e, n13, k8 = op.gate_up_proj._w_op.shape
                self.cpu_worker = self.cpu_worker_factory(
                    hidden=int(k8) * 8, inter=int(n13) // 2, top_k=int(op.top_k)
                )
            seam.attach_cpu_worker(self.cpu_worker, backend_expert_offset=offset)
            self.cpu_worker.backend.pack_layer(
                offset, op.gate_up_proj, op.down_proj, int(op.local_num_experts)
            )
            name = "cpu(compute)"
        elif streamed:
            name = self.stream.adopt(path, op, op.granule_specs(allow_meta=False))
        elif not planned:
            self.outcome.notes.append(
                f"{path}: not in the plan (excluded by policy) -> DEVICE, 0 bytes moved"
            )
            name = "device"
        else:
            name = "host(arena)" if kind is self._host else "device"
        self._seen.add(path)
        self.placed[path] = name
        if kind is self._cpu:
            return "cpu"
        return "host" if kind is self._host else "device"

    def finish(self) -> None:
        from .moe_interpose import InterpositionError

        missing = sorted(set(self._kinds) - self._seen)
        if missing:
            raise InterpositionError(
                f"the plan names MoE layers no chunk delivered: {missing}. Every capacity number "
                f"downstream (arena reservation, device tier, KV pool) was computed for bytes that "
                f"were never loaded. Either the chunk enumeration dropped a layer or the plan was "
                f"resolved against a different module tree."
            )
        if self.stream is not None:
            arrived = {int(i) for i in self.stream.ops}
            lost = sorted(self.stream_layers - arrived)
            if lost:
                raise InterpositionError(
                    f"the stream tier was told to take layers {lost} but no chunk delivered them. "
                    f"They are still device-resident under a plan that does not bill them."
                )
            self.stream.arm()
        self.outcome.seams = tuple(self.seams)


# ---------------------------------------------------------------------------------------------
# The driver
# ---------------------------------------------------------------------------------------------


def _rss_hwm_bytes() -> int:
    """`VmHWM` — the peak RSS this process has ever held, in bytes. 0 if unreadable.

    RSS rather than `MemAvailable`: `MemAvailable` moves with every other tenant of a shared box and
    would attribute their allocations to this load. `hipHostMalloc` pins ordinary anonymous pages via
    userptr, so the arena IS in this process's RSS (and, per P5b, is NOT in the per-card
    `mem_info_gtt_used` counter — never use that one).
    """
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


class ChunkedWeightLoader:
    """Fill `model` from a sequence of `LoadChunk`s, finalizing and placing as it goes."""

    def __init__(
        self,
        model: Any,
        *,
        cast: Callable[[str, Any], Any] | None = None,
        sink: LayerSink | None = None,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.model = model
        # The dtype normalization every `load_weight` consumer must apply
        # (`models/weight.cast_checkpoint_tensor`). Passed in rather than imported so this module
        # does not depend on the models package, and so a bring-up harness can pick its own
        # activation dtype — which is exactly what that function's docstring asks for.
        self.cast = cast
        self.sink: LayerSink = DeviceLayerSink() if sink is None else sink
        self.log = log or (lambda _msg: None)
        self.ledger = ChunkedLoadLedger()
        # Snapshot BEFORE the first chunk: `post_load` deletes the checkpoint buffers of every
        # quantized container, so the model's key set is a different (smaller) set afterwards and a
        # completeness check taken at the end would silently exempt every expert tensor — the exact
        # bytes this whole feature is about.
        self._expected: set[str] = set(model.state_dict())
        self._filled: set[str] = set()

    # -- one chunk ----------------------------------------------------------------------------

    def _ops_by_path(self) -> dict:
        from .moe_interpose import discover_moe_layers

        return dict(discover_moe_layers(self.model))

    def apply_chunk(self, chunk: LoadChunk, tensors: Iterable[Tuple[str, Any]]) -> int:
        """Fill everything `tensors` carries. Returns the bytes staged.

        The whole chunk is materialized as one dict before the fill, because `load_state_dict`'s walk
        is driven by the MODEL's structure and consumes keys in model order, not stream order. That
        dict is the peak: one chunk, by construction.
        """
        from .boot_timeline import tick

        staged: dict = {}
        nbytes = 0
        # SPLIT, not one span: `tensors` is a GENERATOR that does the whole read/remap/shard/H2D/
        # stack pipeline lazily, so the time spent inside `next()` is Stage B's I/O+conversion and
        # the time outside it is this driver's own dict bookkeeping + the dtype cast. Timing the
        # `for` statement as a whole attributes the generator's seconds to the loop body and makes
        # the driver look like the cost.
        it = iter(tensors)
        while True:
            _t = time.perf_counter()
            try:
                item = next(it)
            except StopIteration:
                tick("stageb.stream_next", time.perf_counter() - _t)
                break
            tick("stageb.stream_next", time.perf_counter() - _t)
            key, value = item
            if self.cast is not None:
                _t = time.perf_counter()
                value = self.cast(key, value)
                tick("stageb.cast", time.perf_counter() - _t)
            if key in staged:
                raise ChunkedLoadError(
                    f"chunk {chunk.name!r} yielded {key!r} twice; the second value would silently "
                    f"win and the first read of {value.shape} bytes is unaccounted for"
                )
            staged[key] = value
            nbytes += value.numel() * value.element_size()
        offered = set(staged)
        already = offered & self._filled
        if already:
            raise ChunkedLoadError(
                f"chunk {chunk.name!r} re-delivers {len(already)} keys an earlier chunk already "
                f"filled, e.g. {sorted(already)[:4]}. Overlapping chunks mean one of the two reads "
                f"is dead weight and which one wins depends on chunk order."
            )
        # `_internal=True` hands the unconsumed-keys verdict to this driver rather than to
        # `load_state_dict`'s generic "Unexpected keys" — the message has to name the CHUNK, because
        # with 49 of them "somewhere in the load there was a stray key" is not a diagnosis.
        _t = time.perf_counter()
        self.model.load_state_dict(staged, missing_ok=True, _internal=True)
        tick("stageb.load_state_dict", time.perf_counter() - _t)
        if staged:
            raise ChunkedLoadError(
                f"chunk {chunk.name!r} carried {len(staged)} keys the model has no home for, e.g. "
                f"{sorted(staged)[:4]}"
            )
        self._filled |= offered
        return nbytes

    def finalize_chunk(self, chunk: LoadChunk) -> None:
        """`post_load()` and place every op this chunk completed."""
        from .boot_timeline import tick

        if not chunk.finalize_paths:
            return
        # The op index is rebuilt PER CHUNK by re-walking the whole module tree. Timed separately
        # because that is O(chunks x module tree) work that has nothing to do with the bytes.
        _t = time.perf_counter()
        by_path = self._ops_by_path()
        tick("stageb.discover_moe_layers", time.perf_counter() - _t)
        for path in chunk.finalize_paths:
            op = by_path.get(path)
            if op is None:
                raise ChunkedLoadError(
                    f"chunk {chunk.name!r} claims to finalize {path!r}, which is not a MoE layer of "
                    f"this model. Known paths: {sorted(by_path)[:4]}... "
                    f"({len(by_path)} total). The chunk enumeration and the module tree disagree; "
                    f"placing the wrong layer is silent."
                )
            if op._post_load_done:
                raise ChunkedLoadError(f"{path!r} was already finalized by an earlier chunk")
            _t = time.perf_counter()
            op.post_load()
            tick("stageb.post_load", time.perf_counter() - _t)
            op._post_load_done = True
            _t = time.perf_counter()
            where = self.sink.place(path, op)
            tick("stageb.sink_place", time.perf_counter() - _t)
            if where == "host":
                self.ledger.placed_host_layers += 1
            elif where == "cpu":
                self.ledger.placed_cpu_layers += 1
            else:
                self.ledger.placed_device_layers += 1

    # -- the loop -----------------------------------------------------------------------------

    def run(
        self,
        chunks: Sequence[LoadChunk],
        stream: Callable[[LoadChunk], Iterator[Tuple[str, Any]]],
    ) -> ChunkedLoadLedger:
        import torch

        from .boot_timeline import rss_bytes, tick, timeline

        tl = timeline()
        t0 = time.perf_counter()
        led = self.ledger
        led.min_device_free = torch.cuda.mem_get_info()[0] if torch.cuda.is_available() else 0
        for chunk in chunks:
            t_chunk = time.perf_counter()
            # Snapshot the accumulators so each chunk's row carries ITS OWN split, not the running
            # total. Chunk 0 is the dense body and every other chunk is one layer's 512 experts;
            # folding them into one average hides which of the two the seconds belong to.
            b0 = dict(tl.buckets)
            nbytes = self.apply_chunk(chunk, stream(chunk))
            self.finalize_chunk(chunk)
            # The staging tensors of this chunk are unreachable from here on (the containers hold
            # either their own repacked buffers or arena rows). Return them to the caching allocator
            # NOW rather than at the next GC: the whole point of chunking is that the live set is one
            # chunk, and a chunk still held by a cycle is a chunk that did not shrink the peak.
            _t = time.perf_counter()
            gc.collect()
            tick("stageb.gc_collect", time.perf_counter() - _t)
            led.chunks += 1
            led.staged_bytes += nbytes
            self._sample(torch)
            rss = rss_bytes()
            row = {
                "name": chunk.name,
                "bytes": nbytes,
                "seconds": round(time.perf_counter() - t_chunk, 3),
                "device_allocated": int(torch.cuda.memory_allocated()),
                "device_free": int(torch.cuda.mem_get_info()[0]),
                "host_rss": _rss_hwm_bytes(),
                # THE ANON/FILE SPLIT, per chunk. `host_rss` above is VmHWM, a high-water mark that
                # can only go up and therefore cannot show a chunk giving memory back. These two are
                # CURRENT, so a flat `rss_anon` across chunks is positive evidence that the live set
                # really is one chunk and that VmHWM's growth is page cache.
                "rss_anon": rss["RssAnon"],
                "rss_file": rss["RssFile"],
            }
            row["split"] = {
                k: round(tl.buckets[k] - b0.get(k, 0.0), 3)
                for k in tl.buckets
                if tl.buckets[k] - b0.get(k, 0.0) > 0.0005
            }
            led.per_chunk.append(row)
            tl.row(row)
            self.log(
                f"[stage-b] {chunk.describe()}: {nbytes / _GIB:.3f} GiB in "
                f"{led.per_chunk[-1]['seconds']:.2f}s, device alloc "
                f"{led.per_chunk[-1]['device_allocated'] / _GIB:.2f} GiB, free "
                f"{led.per_chunk[-1]['device_free'] / _GIB:.2f} GiB, RSS "
                f"{led.per_chunk[-1]['host_rss'] / _GIB:.2f} GiB"
            )
        self.sink.finish()
        self.assert_complete()
        led.keys_filled = len(self._filled)
        led.seconds = time.perf_counter() - t0
        self._sample(torch)
        return led

    def _sample(self, torch: Any) -> None:
        led = self.ledger
        led.peak_device_allocated = max(led.peak_device_allocated, int(torch.cuda.memory_allocated()))
        led.peak_device_allocated_torch = max(
            led.peak_device_allocated_torch, int(torch.cuda.max_memory_allocated())
        )
        led.peak_device_reserved = max(led.peak_device_reserved, int(torch.cuda.memory_reserved()))
        free = int(torch.cuda.mem_get_info()[0])
        led.min_device_free = free if led.min_device_free == 0 else min(led.min_device_free, free)
        led.peak_host_rss = max(led.peak_host_rss, _rss_hwm_bytes())

    def assert_complete(self) -> None:
        """Every key the model declared before the load was filled by exactly one chunk."""
        never = sorted(self._expected - self._filled)
        if never:
            raise ChunkedLoadError(
                f"chunked load left {len(never)} of {len(self._expected)} model tensors unfilled, "
                f"e.g. {never[:6]}. Those buffers still hold `torch.empty` garbage and nothing "
                f"downstream can tell. Either a shard is missing from the chunk enumeration or the "
                f"remap dropped a key."
            )
        extra = sorted(self._filled - self._expected)
        if extra:  # pragma: no cover - load_state_dict would have raised first
            raise ChunkedLoadError(f"chunked load filled {len(extra)} keys the model never declared: {extra[:6]}")
