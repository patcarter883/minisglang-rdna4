"""STAGE B — load more of `Qwen3.8-Flash-Next-NVFP4` than the card can hold, by chunking the load.

WHAT THIS MEASURES
------------------
Two legs over the SAME model config, the same shards and the same card, differing in one thing:

  --mode oneshot   the SHIPPED path: `load_weight()` -> one `state_dict` -> `load_state_dict` ->
                   `post_load`. Every tensor of every layer is on the card at once. This is the leg
                   that OOMs, and running it is how "the chunked load reached more layers" stops
                   being an assertion about arithmetic and becomes a measurement.
  --mode chunked   Stage B: `weights.stage_b.ChunkedWeightLoader` over
                   `models.weight.qwen4_exp_chunked_source`, with the pinned host arena as the sink
                   for every layer the placement plan puts on HOST.

Both print peak device bytes (torch's own high-water mark AND the driver's minimum free VRAM, which
is the only figure that sees other tenants of the card), peak host RSS, and wall time.

WHY THE CONTROL LEG IS NOT OPTIONAL
-----------------------------------
"Chunked loaded N layers" proves nothing on its own — N layers might have loaded either way. The
claim is a DIFFERENCE, so the harness measures both legs and `--sweep` walks N upward until the
one-shot leg dies, then runs the chunked leg at that N and beyond. A run that only reports the
chunked leg is reporting a number, not a result.

WHAT IT DOES NOT DO
-------------------
It does not run a forward pass, and it does not claim the loaded weights are numerically right. That
is `qwen4exp_fulldepth_test.py --validate`'s job (bit-exact streamed-vs-resident logits) and the
A1.x correctness gates'. What IS checked here is structural and is checked on every leg: the load
covers every tensor the model declares (`ChunkedWeightLoader.assert_complete`), each layer is
finalized exactly once, and — on the chunked leg — the arena's own bitwise read-back
(`moe_interpose._bake`) compares each host row against its device source while the source is alive.

Run (card 0, in the serve image):

    docker run --rm --device /dev/kfd --device /dev/dri --group-add video \
      --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE \
      --ipc host --shm-size 16gb -e ROCR_VISIBLE_DEVICES=0 \
      -v <worktree>:/engine -v <ckpt>:/model:ro -v <ple>:/ple:ro \
      --entrypoint bash minisgl-rdna4:m1b-20260903 -lc \
      'PYTHONPATH=/engine/python:/opt/kernels python /engine/tests/qwen4exp_stage_b_test.py \
         --mode chunked --layers 24 --device-gb 3 --host-gb 26'
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

MODEL = os.environ.get("Q4E_MODEL", "/model")
GIB = 1 << 30


def _gib(n: float) -> str:
    return f"{n / GIB:.3f} GiB"


def _rss_hwm() -> int:
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def _mem_available() -> int:
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def _subset_cfg_dir(src: str, n_layers: int) -> str:
    """A config-only directory declaring `n_layers` decoder layers. Weights still come from `src`."""
    import tempfile

    from qwen4exp_gpu_forward_test import _subset_config

    d = tempfile.mkdtemp(prefix="q4e-stageb-cfg-")
    _subset_config(src, d, n_layers, 512, ple_1based=2)
    return d


def _setup(n_layers: int, model_dir: str):
    from minisgl.distributed import set_tp_info, try_get_tp_info
    from minisgl.layers.rotary import set_rope_device
    from minisgl.models.config import ModelConfig
    from minisgl.utils import cached_load_hf_config

    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    if try_get_tp_info() is None:
        set_tp_info(0, 1)
    set_rope_device(dev)
    torch.set_default_dtype(torch.bfloat16)
    cfg_dir = model_dir if n_layers == 48 else _subset_cfg_dir(model_dir, n_layers)
    mc = ModelConfig.from_hf(cached_load_hf_config(cfg_dir), spec_algorithm="none")
    assert int(mc.num_layers) == n_layers, (mc.num_layers, n_layers)
    return dev, mc


class _PlanConfig:
    """The three fields `weights.plan.resolve_weight_plan` reads that do not come from the model.

    A stand-in for `EngineConfig`, not a reimplementation of it: the layer set, the EP shape and every
    byte count are read off the built model (`observed_planned_layers`), which is the whole point of
    passing `model=`. What is left is the two operator budgets and the `is_moe` signal used for the
    dense-checkpoint warning.
    """

    def __init__(self, mc, device_gb: float, host_gb: float) -> None:
        self.model_config = mc
        self.weight_offload_device_gb = float(device_gb)
        self.weight_offload_gb = float(host_gb)


def _report(leg: str, args, **extra) -> dict:
    out = {"leg": leg, "layers": args.layers, **extra}
    print(json.dumps(out, indent=2), flush=True)
    return out


def _filtered_shard_dir(model_dir: str, n_layers: int) -> str:
    """A symlink-only directory holding the body shards plus layers `0..n_layers-1`'s expert shards.

    The one-shot loader GLOBS the checkpoint directory, so on the real `/model` it materializes all
    48 layers' expert stacks whatever the config says — it OOMs at the same byte for `--layers 6` and
    `--layers 48`, which measures the absence of a layer filter, not the absence of chunking. To
    measure the CHUNKING, the control leg has to be given exactly the shards the chunked leg reads.
    This is the same trick `qwen4exp_fulldepth_test._ExpertShardDirs` uses, and for the same reason.
    """
    import tempfile

    from minisgl.models.weight import qwen4_exp_chunk_files

    body, per_layer = qwen4_exp_chunk_files(model_dir, n_layers)
    d = tempfile.mkdtemp(prefix="q4e-stageb-shards-")
    for src in body + [f for lid in range(n_layers) for f in per_layer[lid]]:
        os.symlink(src, os.path.join(d, os.path.basename(src)))
    for extra in ("config.json", "generation_config.json"):
        p = os.path.join(model_dir, extra)
        if os.path.exists(p):
            os.symlink(p, os.path.join(d, extra))
    return d


@torch.inference_mode()
def run_oneshot(args) -> dict:
    """The shipped path, unchanged. Measures where it dies."""
    from minisgl.models import cast_checkpoint_tensor, create_model
    from minisgl.models.weight import load_weight

    dev, mc = _setup(args.layers, args.model)
    weight_dir = args.model if args.all_shards else _filtered_shard_dir(args.model, args.layers)
    torch.cuda.reset_peak_memory_stats(dev)
    t0 = time.perf_counter()
    with torch.device("meta"):
        model = create_model(mc)
    expected = len(model.state_dict())
    sd = {
        k: cast_checkpoint_tensor(k, v, torch.bfloat16)
        for k, v in load_weight(weight_dir, dev, spec_algorithm="none")
    }
    # The one-shot path has no num_layers filter of its own for this family (the fulldepth harness
    # drops these by hand), so mirror that here — the OOM under test is the expert tier, not a
    # missing key filter.
    want = set(model.state_dict())
    for k in [k for k in sd if k not in want]:
        del sd[k]
    model.load_state_dict(sd)
    del sd
    model.post_load()
    torch.cuda.synchronize()
    return _report(
        "oneshot",
        args,
        ok=True,
        all_shards=bool(args.all_shards),
        keys=expected,
        seconds=round(time.perf_counter() - t0, 1),
        peak_device_allocated=torch.cuda.max_memory_allocated(dev),
        peak_device_reserved=torch.cuda.max_memory_reserved(dev),
        device_free_after=torch.cuda.mem_get_info(dev)[0],
        peak_host_rss=_rss_hwm(),
    )


@torch.inference_mode()
def run_chunked(args) -> dict:
    from minisgl.models import cast_checkpoint_tensor, create_model
    from minisgl.models.weight import qwen4_exp_chunked_source
    from minisgl.weights.bake import StageARuntime
    from minisgl.weights.plan import resolve_weight_plan
    from minisgl.weights.stage_b import ChunkedWeightLoader, DeviceLayerSink, SeamLayerSink

    dev, mc = _setup(args.layers, args.model)
    torch.cuda.reset_peak_memory_stats(dev)
    t0 = time.perf_counter()
    with torch.device("meta"):
        model = create_model(mc)

    # The plan is resolved off the META model — before a byte is read — which is the property that
    # makes an infeasible checkpoint fail in milliseconds instead of after an 84 GB load.
    cfg = _PlanConfig(mc, args.device_gb, args.host_gb)
    # `arena_chunk_bytes` is NOT optional here, for the reason `bake.py` gives at its own call site:
    # the planner charges host capacity in WHOLE chunks because `attach()` hipHostMallocs and
    # first-touches every chunk at full size. Letting it default to `DEFAULT_CHUNK_BYTES` while the
    # arena below is built from `resolve_arena_settings()` means an operator who moves
    # MINISGL_WEIGHT_ARENA_CHUNK_MIB gets a capacity answer charged against a chunk size the run
    # never uses — the plan refuses (or accepts) on arithmetic about a different arena. One
    # resolver, one chunk size.
    from minisgl.weights.config import resolve_arena_settings

    arena_chunk_bytes = resolve_arena_settings().chunk_bytes
    resolution = resolve_weight_plan(cfg, model=model, arena_chunk_bytes=arena_chunk_bytes)
    print(f"[plan] arena chunk {_gib(arena_chunk_bytes)} "
          f"(MINISGL_WEIGHT_ARENA_CHUNK_MIB)", flush=True)
    print(f"[plan] {resolution.summary_line()}", flush=True)
    for w in getattr(resolution, "warnings", ()) or ():
        print(f"[plan] WARN {w}", flush=True)
    plan = resolution.plan
    print(
        f"[plan] {plan.num_device_layers} device / "
        f"{len(plan.placements) - plan.num_device_layers} host layers, "
        f"host payload {_gib(plan.host_resident_bytes)}, device tier "
        f"{_gib(plan.device_resident_bytes)}",
        flush=True,
    )

    runtime = None
    if plan.host_resident_bytes > 0:
        runtime = StageARuntime(resolution, device_index=0, rank=0, local_ranks=1)
        t_pin = time.perf_counter()
        runtime.attach_host_arena()
        pin_s = time.perf_counter() - t_pin
        for note in runtime.attach_notes():
            print(f"[arena] {note}", flush=True)
        print(
            f"[arena] pinned {_gib(runtime.arena.pinned_bytes)} in {pin_s:.1f}s "
            f"({runtime.arena.pinned_bytes / GIB / max(pin_s, 1e-9):.2f} GiB/s), "
            f"MemAvailable now {_gib(_mem_available())}",
            flush=True,
        )
        sink = SeamLayerSink(plan, runtime.allocator)
    else:
        sink = DeviceLayerSink()

    chunks, stream = qwen4_exp_chunked_source(args.model, dev, mc)
    loader = ChunkedWeightLoader(
        model,
        cast=lambda k, v: cast_checkpoint_tensor(k, v, torch.bfloat16),
        sink=sink,
        log=print if args.verbose else (lambda _m: None),
    )
    led = loader.run(chunks, stream)
    # Everything the chunks did NOT finalize (the whole non-expert body). The expert containers are
    # skipped by `BaseOP._post_load_done`.
    model.post_load()
    torch.cuda.synchronize()

    extra = {}
    arena_bytes = 0
    if runtime is not None:
        arena_bytes = int(runtime.pool.served_bytes)
        extra = {
            "arena_pinned_bytes": int(runtime.arena.pinned_bytes),
            "arena_carved_bytes": int(runtime.arena.carved_bytes),
            "arena_chunks": len(runtime.arena.chunks),
            "torch_fallbacks": int(runtime.arena.torch_fallbacks),
            "pool_served_bytes": arena_bytes,
            "moved_bytes": int(sink.outcome.moved_bytes),
        }
        # The gate that says the host tier really is host-resident: any hipMalloc fallback means
        # those bytes silently landed in VRAM and the capacity plan is a fiction.
        runtime.pool.assert_clean(expect_served_bytes=sink.outcome.moved_bytes)
    print(f"[stage-b] {led.describe()}", flush=True)
    return _report(
        "chunked",
        args,
        ok=True,
        keys=led.keys_filled,
        chunks=led.chunks,
        host_layers=led.placed_host_layers,
        device_layers=led.placed_device_layers,
        staged_bytes=led.staged_bytes,
        seconds=round(led.seconds, 1),
        total_seconds=round(time.perf_counter() - t0, 1),
        # `memory_allocated` COUNTS THE ARENA ROWS: they are handed out through torch's allocator as
        # ordinary `device='cuda'` tensors (that is the whole P5b mechanism), so torch's high-water
        # mark includes host RAM. `..._excl_arena` removes it — the same correction
        # `bake.model_memory_correction()` applies before the KV pool is sized. `min_device_free` is
        # the driver's own answer and needs no correction; it is the figure to trust.
        peak_device_allocated=int(torch.cuda.max_memory_allocated(dev)),
        peak_device_allocated_excl_arena=int(torch.cuda.max_memory_allocated(dev)) - arena_bytes,
        peak_device_reserved=int(torch.cuda.max_memory_reserved(dev)),
        min_device_free=led.min_device_free,
        device_free_after=int(torch.cuda.mem_get_info(dev)[0]),
        peak_host_rss=led.peak_host_rss,
        mem_available_after=_mem_available(),
        per_chunk=led.per_chunk if args.verbose else led.per_chunk[:3] + led.per_chunk[-3:],
        **extra,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("oneshot", "chunked"), required=True)
    ap.add_argument("--layers", type=int, default=48)
    ap.add_argument("--device-gb", type=float, default=3.0,
                    help="VRAM the routed-expert tier may occupy, GiB (device tier)")
    ap.add_argument("--host-gb", type=float, default=26.0,
                    help="pinned host arena budget per rank, GiB")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--all-shards", action="store_true",
                    help="oneshot leg only: point the loader at the WHOLE checkpoint dir, which is "
                         "what a real serve does. Without it the leg gets exactly the shards the "
                         "chunked leg reads, which is what isolates chunking from file filtering.")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("FAIL: no HIP device visible — check device passthrough")
        return 1
    dev = torch.device("cuda:0")
    print(
        f"[gpu] {torch.cuda.get_device_name(0)} total={_gib(torch.cuda.mem_get_info(dev)[1])} "
        f"free={_gib(torch.cuda.mem_get_info(dev)[0])} MemAvailable={_gib(_mem_available())}",
        flush=True,
    )
    try:
        out = run_oneshot(args) if args.mode == "oneshot" else run_chunked(args)
        rc = 0
    except BaseException as exc:  # noqa: BLE001 — the failure IS the measurement on the control leg
        traceback.print_exc()
        out = {
            "leg": args.mode,
            "layers": args.layers,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}"[:600],
            "peak_device_allocated": int(torch.cuda.max_memory_allocated(dev)),
            "peak_device_reserved": int(torch.cuda.max_memory_reserved(dev)),
            "peak_host_rss": _rss_hwm(),
        }
        print(json.dumps(out, indent=2), flush=True)
        rc = 2
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=2)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
