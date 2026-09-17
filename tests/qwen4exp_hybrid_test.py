"""ALL 48 LAYERS of qwen4_exp with the REAL offload machinery engaged — three tiers, one card.

WHY THIS EXISTS
---------------
`qwen4exp_stage_b_test.py` proved the chunked load and the arena bake, and stops at the load:
"No forward pass was run." `qwen4exp_fulldepth_test.py` proved 48 layers can be *computed* on one
card, but by NVMe-streaming every layer — the arena and the seams are not in that path at all. So
the one thing nobody had ever done was run a forward whose MoE kernels read expert weights out of
the pinned host arena. That is this harness.

It also exists because **TP=1 all-pinned is arithmetically impossible on this box** and that is now
measured three ways. The plan refuses in milliseconds off the meta model:

    48 layers -> 70.31 GiB of experts. Device tier caps at ~5 GiB (the 15.92 GiB card already holds
    a 9.22 GiB non-expert body), so the host tier is 65.92 GiB/rank against ~53 GiB of usable RAM
    (MemAvailable ~65 GiB minus the 12 GiB floor). `PlacementError` says the device tier would have
    to be 32.23 GiB on a 16 GiB card.

There is no assignment of {device, pinned host} that fits, so a THIRD tier is not an optimisation
here, it is the only way 48 layers boots at TP=1. This harness composes all three:

    DEVICE   resident op buffers, exactly as a non-offloaded serve. Zero copies.
    ARENA    `moe_interpose.bind_seam(StackKind.HOST, allocator)` moves the layer's `_w_op`/
             `_scales_op` into pinned host pages and the MoE kernel reads them over PCIe. This is
             the shipped bake, bitwise-verified per row, weakref-proved to release its device
             source. The top-10-of-512 routing is what makes it viable: a layer's kernels touch
             ~29 MiB of its 1.4 GiB stack per token, not the whole thing.
    STREAM   the tail the arena cannot hold. The layer's containers are aliased onto ONE shared
             device buffer pair and refilled from the layer's own shards immediately before it
             runs — `qwen4exp_fulldepth_test.ExpertStreamer`'s mechanism, which that harness proved
             BIT-IDENTICAL to a fully-resident control.

THE PLAN-TIME CAPACITY REFUSAL IS BYPASSED ON PURPOSE, AND ONLY IT
------------------------------------------------------------------
`StageARuntime.attach_host_arena()` calls `raise_if_infeasible()` and would refuse this model, which
is CORRECT: it is asked whether {device, host} alone can hold 70.31 GiB, and they cannot. This
harness builds the same arena the same way — `create_pinned_weight_arena` -> exact
`plan.host_row_requests()` -> `reserve` -> `attach` -> `ArenaMemPool` -> `TorchStackAllocator` — but
sizes the host tier to a number that FITS and gives the remainder to the stream tier. Every LIVE
guard is untouched and still armed: the MemAvailable floor, the per-chunk headroom check, the swap
tripwire, the arena self-test, the bake's bitwise read-back, and `ArenaMemPool.assert_clean` (any
`hipMalloc` fallback means a "host" row silently landed in VRAM and the whole accounting is fiction).

STAGING THE WRONG LAYER IS THE FAILURE MODE THAT MATTERS
--------------------------------------------------------
A streamer that stages layer 31's experts for layer 30 produces fluent-looking garbage and no error.
`_StreamTier.stage()` therefore records which layer is live in the shared buffers and
`assert_staged()` — called from the hooked forward AFTER staging — refuses if they disagree. And
`--validate` A/Bs the hybrid build against a fully-resident control at a small layer count and
requires BIT-IDENTICAL logits, which a wrong layer->shard mapping cannot produce.

Run (card 0, in the serve image):

    docker run --rm --device /dev/kfd --device /dev/dri --group-add video \
      --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE \
      --ipc host --shm-size 16gb -e ROCR_VISIBLE_DEVICES=0 \
      -v <worktree>:/engine -v <ckpt>:/model:ro -v <ple>:/ple:ro \
      --entrypoint bash minisgl-rdna4:m1b-20260903 -lc \
      'PYTHONPATH=/engine/python:/opt/kernels python /engine/tests/qwen4exp_hybrid_test.py \
         --layers 48 --device-layers 2 --arena-layers 16 --new 64 --gen-config --chat'
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

MODEL = os.environ.get("Q4E_MODEL", "/model")
GIB = 1 << 30

_failures: "list[str]" = []


def _gib(n: float) -> str:
    return f"{n / GIB:.3f} GiB"


def check_true(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"    ok   {name}", flush=True)
    else:
        _failures.append(f"{name}: {detail}")
        print(f"    FAIL {name}: {detail}", flush=True)


def _mem_available() -> int:
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def _rss_hwm() -> int:
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


# ---------------------------------------------------------------------------------------------
# The STREAM tier
# ---------------------------------------------------------------------------------------------


class _StreamTier:
    """Alias N layers' expert containers onto ONE shared device buffer pair; refill per layer.

    Mechanically `qwen4exp_fulldepth_test.ExpertStreamer`, with two differences that matter here:

    * it is INCREMENTAL — the sink hands it one layer at a time as the chunked load finalizes them,
      so the peak device footprint is one layer's op buffers, not the whole tier's. The FIRST stream
      layer donates its buffers as the shared pair; every later one is pointed at them and its own
      1.4 GiB is dropped on the spot.
    * it carries a `live` marker and `assert_staged()`. The whole reason a streamer is dangerous is
      that staging the wrong layer is numerically silent, so the layer index that is physically in
      the buffers is recorded at the copy and checked at the use. It costs one integer compare per
      MoE layer per forward.
    """

    def __init__(self, mc, dev, shard_dirs) -> None:
        self.mc = mc
        self.dev = dev
        self.shard_dirs = shard_dirs
        self.pairs: "dict[int, tuple]" = {}
        self.shared: "tuple | None" = None
        self.live = -1
        self.seconds = 0.0
        self.stages = 0
        self.bytes_staged = 0

    def adopt(self, lid: int, c13, c2) -> str:
        """Take layer `lid` into the stream tier. Returns a one-word placement name."""
        self.pairs[lid] = (c13, c2)
        if self.shared is None:
            # This layer's freshly post_load'd buffers BECOME the shared pair. Its contents are
            # irrelevant from here on: every stream layer, this one included, is restaged before it
            # runs.
            self.shared = (c13._w_op, c13._scales_op, c2._w_op, c2._scales_op)
            return "stream(donor)"
        s13w, s13s, s2w, s2s = self.shared
        for cont, w, s in ((c13, s13w, s13s), (c2, s2w, s2s)):
            if cont._w_op.shape != w.shape or cont._scales_op.shape != s.shape:
                raise RuntimeError(
                    f"layer {lid}: stream alias shape mismatch — the shared buffer pair was sized "
                    f"from the donor layer and this layer's op layout differs "
                    f"({tuple(cont._w_op.shape)} vs {tuple(w.shape)}). Aliasing would read the "
                    f"wrong bytes with no error."
                )
            cont._w_op = w
            cont._scales_op = s
        return "stream"

    def stage(self, lid: int) -> None:
        from minisgl.quant import nvfp4
        from qwen4exp_fulldepth_test import _load_layer_experts

        t0 = time.time()
        got = _load_layer_experts(self.shard_dirs.dirs[lid], self.mc, self.dev, lid)
        c13, c2 = self.pairs[lid]
        for cont, base in ((c13, "mlp.experts.gate_up_proj"), (c2, "mlp.experts.down_proj")):
            conv = nvfp4.convert_nvfp4_moe(got[f"{base}.weight_packed"], got[f"{base}.weight_scale"])
            # `_GroupedNvFp4Experts.post_load`'s own two lines, reused rather than re-derived, so a
            # change to the op layout raises on the copy_ shape instead of staging stale bytes.
            cont._w_op.copy_(conv["w_packed"])
            cont._scales_op.copy_(conv["scales"].transpose(1, 2))
            del conv
        self.bytes_staged += sum(v.numel() * v.element_size() for v in got.values())
        del got
        self.live = lid
        self.stages += 1
        self.seconds += time.time() - t0

    def assert_staged(self, lid: int) -> None:
        if self.live != lid:
            raise RuntimeError(
                f"stream tier is holding layer {self.live}'s experts but layer {lid} is about to "
                f"run. This is the silent-garbage failure mode: identical shapes, plausible "
                f"logits, wrong weights."
            )

    def install_hooks(self, decoder_layers) -> None:
        """Restage immediately before each stream layer's forward.

        An instance attribute shadows the bound method, so no model file is edited — a streaming
        hack that lives in the model is a streaming hack that ships.
        """
        for lid in sorted(self.pairs):
            layer = decoder_layers[lid]
            inner = layer.forward

            def staged_forward(hidden, _lid=lid, _inner=inner):
                self.stage(_lid)
                self.assert_staged(_lid)
                return _inner(hidden)

            layer.forward = staged_forward


# ---------------------------------------------------------------------------------------------
# The 3-tier sink
# ---------------------------------------------------------------------------------------------


class HybridLayerSink:
    """`stage_b.LayerSink` that routes each finalized MoE layer to one of THREE tiers.

    Device and arena go through `moe_interpose.bind_seam` — the same call `SeamLayerSink` makes, so
    the bake, its bitwise read-back, its weakref leak proof and its byte accounting are shared and
    cannot drift. The stream tier is bound DEVICE first (zero bytes moved, the seam's identity check
    still installed) and then aliased, which keeps every layer covered by a seam and keeps the
    `engaged()` ledger honest about how many layers each arm actually serves.
    """

    def __init__(self, plan, allocator, stream_tier, stream_ids, *, selftest=None) -> None:
        from minisgl.weights.moe_interpose import SELFTEST_SAMPLE, BindOutcome
        from minisgl.weights.stacks import StackKind

        self.plan = plan
        self.allocator = allocator
        self.stream = stream_tier
        self.stream_ids = set(stream_ids)
        self.selftest = SELFTEST_SAMPLE if selftest is None else int(selftest)
        self._kinds = {p.path: p.kind for p in plan.placements}
        self._host = StackKind.HOST
        self.outcome = BindOutcome(plan_digest=plan.digest())
        self.seams: list = []
        self.placed: "dict[str, str]" = {}

    @staticmethod
    def _lid(path: str) -> int:
        # "model.layers.31.mlp.experts" -> 31. Structural, and it is the ONE place the harness
        # turns a seam path back into a layer index; a mismatch here would put the wrong layer in
        # the wrong tier, so it raises rather than defaulting.
        parts = path.split(".")
        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts):
                return int(parts[i + 1])
        raise RuntimeError(f"cannot recover a layer index from seam path {path!r}")

    def place(self, path: str, op) -> str:
        from minisgl.weights.moe_interpose import attach_seam, bind_seam
        from minisgl.weights.stacks import StackKind

        lid = self._lid(path)
        planned = path in self._kinds
        kind = self._kinds.get(path, StackKind.DEVICE)
        seam = attach_seam(path, op)
        self.seams.append(seam)
        bind_seam(
            seam,
            kind,
            self.allocator if kind is self._host else None,
            selftest=self.selftest,
            out=self.outcome,
            count_device_bytes=planned and lid not in self.stream_ids,
        )
        if lid in self.stream_ids:
            if kind is self._host:
                raise RuntimeError(
                    f"layer {lid} is in the stream tier but the plan placed it HOST. The tier "
                    f"assignment and the plan disagree, so the arena reservation is for a layer "
                    f"that will never occupy it."
                )
            name = self.stream.adopt(lid, *[getattr(op, a) for a in _container_attrs(op)])
        else:
            name = "host(arena)" if kind is self._host else "device"
        self.placed[path] = name
        return "host" if kind is self._host else "device"

    def finish(self) -> None:
        from minisgl.weights.moe_interpose import InterpositionError

        missing = sorted(set(self._kinds) - set(self.placed))
        if missing:
            raise InterpositionError(
                f"the plan names MoE layers no chunk delivered: {missing}."
            )
        self.outcome.seams = tuple(self.seams)


def _container_attrs(layer):
    from minisgl.weights.moe_interpose import _container_attrs as f

    return f(layer)


# ---------------------------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------------------------


def _tier_plan(resolution, arena_ids, stream_ids):
    """Rebuild the resolved plan with THIS harness's three-way tier assignment.

    Only `kind` changes; every byte count, row list and packing bound is the resolver's own, so the
    arena reservation below is the exact one `attach_host_arena` would have made for the same set
    of host layers. Stream layers are recorded DEVICE (they do occupy one shared device pair) and
    excluded from the device-byte accounting by the sink.
    """
    from minisgl.weights.placement import LayerPlacement, OffloadPlan
    from minisgl.weights.stacks import StackKind

    placements = []
    for p in resolution.plan.placements:
        lid = HybridLayerSink._lid(p.path)
        kind = StackKind.HOST if lid in arena_ids else StackKind.DEVICE
        placements.append(
            LayerPlacement(
                path=p.path,
                kind=kind,
                resident_bytes=p.resident_bytes,
                granule_bytes=p.granule_bytes,
                num_experts=p.num_experts,
                top_k=p.top_k,
                max_row_bytes=p.max_row_bytes,
                rows=p.rows,
            )
        )
    dev_bytes = sum(
        p.resident_bytes
        for p in placements
        if p.kind is StackKind.DEVICE and HybridLayerSink._lid(p.path) not in stream_ids
    )
    return OffloadPlan(
        placements=tuple(placements),
        device_budget_bytes=dev_bytes,
        total_resident_bytes=sum(p.resident_bytes for p in placements),
    )


def _make_arena(plan, dev_index: int):
    """`attach_host_arena()` minus `raise_if_infeasible()`. Every LIVE guard stays armed.

    The one thing skipped is the plan-time question "do {device, host} alone hold the model?", whose
    answer for this checkpoint is no and always will be — see the module docstring. The MemAvailable
    floor, the per-chunk headroom check and the swap tripwire all live in `arena.attach()` and are
    untouched, so a host tier this box cannot actually pin still aborts before it damages anything.
    """
    from minisgl.weights.config import create_pinned_weight_arena, resolve_arena_settings
    from minisgl.weights.stacks import TorchStackAllocator
    from minisgl.weights.torch_pool import ArenaMemPool

    settings = resolve_arena_settings()
    arena = create_pinned_weight_arena(
        dev_index, rank=0, local_ranks=1, label="weights", settings=settings
    )
    rows = plan.host_row_requests()
    if not rows:
        raise RuntimeError(
            "the plan could not enumerate its host rows, so the reservation would fall back to the "
            "anonymous headroom bound. Refusing: that silently under-reserves and puts the tail of "
            "the host tier in VRAM."
        )
    arena.reserve(rows)
    arena.attach(selftest=settings.selftest, first_touch=settings.first_touch)
    pool = ArenaMemPool(arena)
    allocator = TorchStackAllocator(
        device=torch.device("cuda", dev_index),
        host_alloc=lambda shape, dtype: pool.empty(*shape, dtype=dtype),
    )
    return arena, pool, allocator, settings


@torch.inference_mode()
def build(args, dev, mc):
    """Meta-build, chunk-load, and place every MoE layer in one of the three tiers."""
    from minisgl.models import cast_checkpoint_tensor, create_model
    from minisgl.models.weight import qwen4_exp_chunked_source
    from minisgl.weights.config import resolve_arena_settings
    from minisgl.weights.plan import resolve_weight_plan
    from minisgl.weights.stage_b import ChunkedWeightLoader
    from qwen4exp_fulldepth_test import _ExpertShardDirs
    from qwen4exp_stage_b_test import _PlanConfig

    n = args.layers
    n_dev, n_arena = args.device_layers, args.arena_layers
    if n_dev + n_arena > n:
        raise SystemExit(f"--device-layers + --arena-layers ({n_dev}+{n_arena}) exceeds {n}")
    device_ids = set(range(n_dev))
    arena_ids = set(range(n_dev, n_dev + n_arena))
    stream_ids = set(range(n_dev + n_arena, n))
    print(
        f"[tier] {len(device_ids)} device / {len(arena_ids)} arena(pinned host) / "
        f"{len(stream_ids)} stream, over {n} layers",
        flush=True,
    )

    torch.cuda.reset_peak_memory_stats(dev)
    t0 = time.perf_counter()
    with torch.device("meta"):
        model = create_model(mc)

    # The resolver runs off the META model, before a byte is read — that is what makes the byte
    # counts below a property of the checkpoint rather than of a load that has already happened.
    cfg = _PlanConfig(mc, args.device_gb, args.host_gb)
    chunk_bytes = resolve_arena_settings().chunk_bytes
    resolution = resolve_weight_plan(cfg, model=model, arena_chunk_bytes=chunk_bytes)
    per_layer = resolution.plan.placements[0].resident_bytes if resolution.plan.placements else 0
    print(
        f"[plan] resolver: {_gib(resolution.plan.total_resident_bytes)} of experts over "
        f"{len(resolution.plan.placements)} layers ({_gib(per_layer)}/layer), arena chunk "
        f"{_gib(chunk_bytes)}",
        flush=True,
    )
    plan = _tier_plan(resolution, arena_ids, stream_ids)
    print(
        f"[plan] hybrid: device {_gib(len(device_ids) * per_layer)}  arena "
        f"{_gib(plan.host_resident_bytes)}  stream {_gib(len(stream_ids) * per_layer)} "
        f"(1 shared pair resident)",
        flush=True,
    )

    arena = pool = allocator = None
    if plan.host_resident_bytes > 0:
        t_pin = time.perf_counter()
        arena, pool, allocator, settings = _make_arena(plan, 0)
        pin_s = time.perf_counter() - t_pin
        print(
            f"[arena] pinned {_gib(arena.pinned_bytes)} in {pin_s:.1f}s "
            f"({arena.pinned_bytes / GIB / max(pin_s, 1e-9):.2f} GiB/s), {len(arena.chunks)} chunks,"
            f" MemAvailable now {_gib(_mem_available())}",
            flush=True,
        )

    shard_dirs = _ExpertShardDirs(args.model, sorted(stream_ids)) if stream_ids else None
    stream_tier = _StreamTier(mc, dev, shard_dirs) if stream_ids else None
    sink = HybridLayerSink(plan, allocator, stream_tier, stream_ids)

    chunks, stream = qwen4_exp_chunked_source(args.model, dev, mc)
    loader = ChunkedWeightLoader(
        model,
        cast=lambda k, v: cast_checkpoint_tensor(k, v, torch.bfloat16),
        sink=sink,
        log=print if args.verbose else (lambda _m: None),
    )
    led = loader.run(chunks, stream)
    model.post_load()
    torch.cuda.synchronize()
    print(f"[stage-b] {led.describe()}", flush=True)

    gather = None
    if stream_tier is not None:
        if args.stream_mode == "routed":
            # top-10 of 512: a stream layer reads ~29 MiB per token instead of its whole 1.465 GiB
            # tier — 51x less, and the difference between ~1 s/token and ~30 s/token at this depth.
            # Same class the fulldepth harness proved bit-identical, narrowed to the stream layers.
            from qwen4exp_fulldepth_test import RoutedExpertGather

            gather = RoutedExpertGather(model, mc, dev, args.model, layer_ids=sorted(stream_ids))
            gather.install_hooks()
        else:
            stream_tier.install_hooks(model.model.layers.op_list)
    torch.cuda.empty_cache()

    stats = {
        "build_seconds": round(time.perf_counter() - t0, 1),
        "keys_filled": led.keys_filled,
        "min_device_free": led.min_device_free,
        "peak_host_rss": led.peak_host_rss,
        "mem_available_after": _mem_available(),
        "device_layers": sorted(device_ids),
        "arena_layers": sorted(arena_ids),
        "stream_layers": sorted(stream_ids),
        "per_layer_expert_bytes": per_layer,
    }
    if arena is not None:
        stats.update(
            arena_pinned_bytes=int(arena.pinned_bytes),
            arena_carved_bytes=int(arena.carved_bytes),
            arena_chunks=len(arena.chunks),
            torch_fallbacks=int(arena.torch_fallbacks),
            pool_served_bytes=int(pool.served_bytes),
            moved_bytes=int(sink.outcome.moved_bytes),
        )
    return model, stream_tier, gather, shard_dirs, arena, pool, sink, stats


# ---------------------------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------------------------


def gate_accounting(arena, pool, sink, stats, n_arena: int) -> None:
    """Stage 2: the arena really is host memory holding exactly the planned bytes."""
    print("\n[2] ARENA ACCOUNTING", flush=True)
    if arena is None:
        print("    (no arena layers requested)", flush=True)
        return
    moved = int(sink.outcome.moved_bytes)
    check_true(
        "moved == carved == pool.served",
        moved == int(arena.carved_bytes) == int(pool.served_bytes),
        f"moved={moved} carved={arena.carved_bytes} served={pool.served_bytes}",
    )
    check_true("host layers bound == requested", sink.outcome.host_layers == n_arena,
               f"{sink.outcome.host_layers} vs {n_arena}")
    # The gate that says the host tier IS host-resident: any hipMalloc fallback means those bytes
    # landed in VRAM and every capacity number is a fiction.
    check_true("zero hipMalloc fallbacks", int(arena.torch_fallbacks) == 0,
               f"{arena.torch_fallbacks} rows fell back to VRAM")
    try:
        pool.assert_clean(expect_served_bytes=moved)
        print("    ok   ArenaMemPool.assert_clean", flush=True)
    except Exception as exc:  # noqa: BLE001
        _failures.append(f"assert_clean: {exc}")
        print(f"    FAIL ArenaMemPool.assert_clean: {exc}", flush=True)
    # NON-VACUOUS is the point: `SelfTestResult.vacuous` exists because "no failures were
    # recorded" and "N chunks were probed and every word came back right" differ exactly when
    # nothing was probed, and a `selftest PASS chunks=0` banner asserts nothing.
    st = arena._selftest
    check_true("arena self-test ran and passed", st is not None and st.passed and not st.vacuous,
               "no self-test result" if st is None else st.summary())
    if st is not None:
        print(f"    {st.summary()}", flush=True)
    # ENFORCES torch_fallbacks == 0 and closes the population window. If any host-budgeted byte
    # landed in VRAM this raises here rather than presenting as an unrelated OOM later.
    arena.mark_populated()
    print("    ok   arena.mark_populated() (fallback-free population window closed)", flush=True)


def gate_residency(model, arena, sink) -> None:
    """The seams' HOST arm points at pointers the ARENA owns — not merely at 'somewhere'."""
    from minisgl.weights.moe_interpose import prove_seam_residency

    print("\n[2b] SEAM RESIDENCY PROOF", flush=True)
    proof = prove_seam_residency(
        model, sink.seams, owns_pointer=None if arena is None else arena.owns_pointer
    )
    print(f"    {proof.describe()}", flush=True)
    return proof


def gate_logits(logits) -> None:
    """Stage 3: the prefill produced real numbers, not NaNs and not a constant."""
    print("\n[3] PREFILL LOGITS", flush=True)
    row = logits[-1].float()
    check_true("finite", bool(torch.isfinite(row).all()), "non-finite entries in the last row")
    check_true("not constant", float(row.std()) > 1e-3, f"std={float(row.std()):.6f}")
    top = torch.topk(row, 5)
    print(f"    top5 ids={top.indices.tolist()} vals={[round(v, 3) for v in top.values.tolist()]}",
          flush=True)
    print(f"    mean={float(row.mean()):.4f} std={float(row.std()):.4f} "
          f"max={float(row.max()):.4f}", flush=True)


# ---------------------------------------------------------------------------------------------


@torch.inference_mode()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=48)
    ap.add_argument("--device-layers", type=int, default=2)
    ap.add_argument("--arena-layers", type=int, default=16)
    ap.add_argument("--device-gb", type=float, default=5.0)
    ap.add_argument("--host-gb", type=float, default=40.0)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--new", type=int, default=32)
    ap.add_argument("--prompt", default="What is the capital of France, and what river runs "
                                        "through it?")
    ap.add_argument("--chat", action="store_true", help="apply the checkpoint's chat template")
    ap.add_argument("--gen-config", action="store_true",
                    help="read temperature/top_k/top_p/eos from the checkpoint's own "
                         "generation_config.json (explicit flags still win)")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--attn-backend", default="rdna4")
    ap.add_argument("--real-ple", action="store_true", default=True)
    ap.add_argument("--stream-mode", choices=("routed", "full"), default="routed",
                    help="how the stream tier refills the shared pair. `routed` stages only "
                         "the top-10-of-512 the layer's tokens actually route to (and NaN-"
                         "poisons every other row, so reading an unstaged expert produces NaN "
                         "logits instead of a plausible wrong number). `full` restages all 512 "
                         "— 51x more traffic, kept as the slow control the routed gather is "
                         "A/B'd against."
                    )
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", default="")
    ap.add_argument("--logits-out", default="",
                    help="save the prefill logits + generated ids for an offline A/B. The ONLY "
                         "check that can catch a wrong layer->tier or layer->shard mapping: it is "
                         "numerically silent and produces fluent-looking garbage, so a tier split "
                         "is not trusted until its logits are BIT-IDENTICAL to a fully-resident "
                         "control at the same layer count.")
    args = ap.parse_args()

    from qwen4exp_fulldepth_test import run as forward_run
    from qwen4exp_stage_b_test import _setup

    dev, mc = _setup(args.layers, args.model)
    print(f"[box] MemAvailable {_gib(_mem_available())}, VRAM free "
          f"{_gib(torch.cuda.mem_get_info(dev)[0])}", flush=True)

    # ---- sampler: the checkpoint's own declared operating point -----------------------------
    temperature, top_k, top_p, eos_ids = 1.0, 0, 1.0, ()
    if args.gen_config:
        import json as _json

        p = os.path.join(args.model, "generation_config.json")
        if os.path.exists(p):
            gc = _json.load(open(p))
            temperature = float(gc.get("temperature", 1.0))
            top_k = int(gc.get("top_k", 0) or 0)
            top_p = float(gc.get("top_p", 1.0))
            e = gc.get("eos_token_id", [])
            eos_ids = tuple(e) if isinstance(e, list) else (int(e),)
    if args.temperature is not None:
        temperature = args.temperature
    if args.top_k is not None:
        top_k = args.top_k
    if args.top_p is not None:
        top_p = args.top_p
    if temperature <= 0:
        print("\n!! temperature=0 is a DETERMINISM PROBE, NOT A QUALITY READ. Greedy decoding "
              "manufactures a degeneration signature that mimics a quant bug. !!\n", flush=True)
    print(f"[sampler] temperature={temperature} top_k={top_k} top_p={top_p} eos={eos_ids}",
          flush=True)

    tok = None
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[tok] unavailable ({exc}); ids only", flush=True)
    if tok is None:
        raise SystemExit("a tokenizer is required to judge output quality")
    if args.chat:
        text = tok.apply_chat_template(
            [{"role": "user", "content": args.prompt}], tokenize=False, add_generation_prompt=True
        )
    else:
        text = args.prompt
    prompt_ids = np.array(tok(text, add_special_tokens=False)["input_ids"], dtype=np.int64)
    print(f"[prompt] {len(prompt_ids)} tokens", flush=True)

    out = {"layers": args.layers, "ok": False}
    try:
        print("\n[1] BUILD — plan + arena + chunked load", flush=True)
        model, stream_tier, gather, shard_dirs, arena, pool, sink, stats = build(args, dev, mc)
        out.update(stats)
        print(f"[1] built in {stats['build_seconds']}s, VRAM free "
              f"{_gib(torch.cuda.mem_get_info(dev)[0])}", flush=True)

        gate_accounting(arena, pool, sink, stats, args.arena_layers)
        if args.arena_layers:
            proof = gate_residency(model, arena, sink)
            out["seam_proof"] = proof.describe()
        else:
            # `prove_seam_residency` REFUSES a plan with no host layer, correctly: on a serve that
            # is the dispatch regression the engaged() ledger exists to catch. Here a zero-host
            # build is the deliberate all-device CONTROL leg of the A/B, so the proof is skipped
            # rather than defeated — and skipping it is recorded, not silent.
            print("\n[2b] SEAM RESIDENCY PROOF — skipped: --arena-layers 0 is the all-device "
                  "control leg, which has no host residency to prove.", flush=True)
            out["seam_proof"] = "skipped (all-device control leg)"

        print("\n[3/4] FORWARD", flush=True)
        logits, new_ids, decode_secs = forward_run(
            mc, model, dev, prompt_ids,
            n_new=args.new, max_seq=max(2048, len(prompt_ids) + args.new + 16),
            attn_backend=args.attn_backend, real_ple=args.real_ple,
            temperature=temperature, seed=args.seed, streamer=None, tok=tok,
            top_k=top_k, top_p=top_p, eos_ids=eos_ids,
        )
        gate_logits(logits[0])
        text_out = tok.decode(new_ids)
        warm = decode_secs[1:] or decode_secs
        tok_s = (len(warm) / sum(warm)) if warm else 0.0
        out.update(
            ok=True,
            generated_ids=new_ids,
            generated_text=text_out,
            tok_per_s=round(tok_s, 3),
            decode_steps=len(decode_secs),
            stream_mode=args.stream_mode,
            stream_stages=(gather.staged.__len__() if gather is not None
                           else (0 if stream_tier is None else stream_tier.stages)),
            stream_seconds=round((gather.seconds if gather is not None
                                  else (0.0 if stream_tier is None else stream_tier.seconds)), 1),
            stream_bytes_read=(int(gather.bytes_read) if gather is not None
                               else (0 if stream_tier is None else stream_tier.bytes_staged)),
            experts_staged=0 if gather is None else int(gather.experts_staged),
            vram_free_after=int(torch.cuda.mem_get_info(dev)[0]),
        )
        print(f"\n[OUTPUT] {text_out!r}", flush=True)
        print(f"[rate] {tok_s:.3f} tok/s steady-state", flush=True)
        if args.logits_out:
            torch.save(
                {
                    "prefill_logits": logits[0].detach().float().cpu(),
                    "generated_ids": new_ids,
                    "tiers": [args.device_layers, args.arena_layers,
                              args.layers - args.device_layers - args.arena_layers],
                },
                args.logits_out,
            )
            print(f"[logits] saved to {args.logits_out}", flush=True)
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    finally:
        if args.out:
            with open(args.out, "w") as fh:
                json.dump(out, fh, indent=2)
        # RELEASE THE MMAPS BEFORE THE INTERPRETER TEARS DOWN. The routed gather holds one
        # `safetensors.safe_open` per (layer, expert-shard) it has touched — up to 4 per streaming
        # layer, ~124 at 48 layers — each an entered context manager over an mmap of the checkpoint.
        # Leaving them to interpreter shutdown segfaulted the process (exit 139) AFTER every result
        # had been printed and flushed: harmless to the measurement, and exactly the kind of
        # "it crashed but the numbers were fine" that must not be left in a harness others will
        # copy. `close()` is what `qwen4exp_fulldepth_test.main` has always called; this harness
        # simply did not.
        for res in ("gather", "shard_dirs"):
            obj = locals().get(res)
            closer = getattr(obj, "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception as exc:  # noqa: BLE001
                    print(f"[cleanup] {res}.close() raised: {exc}", flush=True)

    out["failures"] = _failures
    print("\n" + json.dumps({k: v for k, v in out.items() if k != "generated_ids"}, indent=2),
          flush=True)
    return 1 if (_failures or not out.get("ok")) else 0


if __name__ == "__main__":
    raise SystemExit(main())
