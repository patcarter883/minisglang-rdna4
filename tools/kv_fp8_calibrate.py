"""Offline fp8-KV calibrator — the producer of the PER-HEAD scale sidecar.

Writes `kv_scales.safetensors`, which `minisgl.kvcache.fp8_scales` loads at engine boot. This is the
only source of per-head scales: per-head amax is a property of the model's ACTIVATIONS on real text,
and no checkpoint ships it (compressed-tensors' kv_cache_scheme is `strategy: tensor`, one scalar per
layer).

WHY A SEPARATE PROCESS. Changing a descale invalidates every cache entry already written under the
old one, so a served engine must have final scales before its first store. Calibration therefore
runs here, in a process whose cache is thrown away, and hands the serve a file.

WHY IT RUNS WITH THE DEFAULT bf16 CACHE. The quantity being measured is `amax|K|` / `amax|V|` of the
tensors handed to `store_kv`, which is what the cache would be asked to represent. Measuring it with
an fp8 cache already installed would feed fp8 error back into the activations of every later layer,
and would require booting the very fp8 path being configured. So: bf16 cache, exact activations,
`MINISGL_KV_FP8_CALIBRATE=1` to accumulate.

MLA. A latent cache (DeepSeek / GLM-4.x) has no head axis — it stores ONE compressed vector per
token, read back in both the K and V roles, which is why the mla_hip fp8 kernels want
`k_descale == v_descale == cache_descale`. So an MLA model is calibrated PER LAYER: one amax per
layer, emitted as identical `k_scale`/`v_scale` scalars. This is the only source of an MLA scale at
all — no checkpoint ships a latent-cache scale, and before this the fp8 latent cache was stored and
read with an implicit 1.0 under the compose default `MINISGL_KV_FP8=1`.

TP. `--tp N` runs the calibration forward at tensor parallelism N, ONE PROCESS PER RANK (the same
spawn the server uses). This is not a nicety: every model this box actually serves — Qwen3.6-35B
(24 GB), Laguna-XS-2.1 (20 GB), GLM-4.7-Flash (19 GB) — is bigger than one 16 GB card, so a TP=1
calibrator could only ever calibrate toys. Each rank owns `num_kv_heads/tp` KV heads and therefore
accumulates only ITS OWN amax rows; those rows are gathered into GLOBAL per-head rows on the host
before any scale is computed, so the sidecar is TP-independent. `fp8_scales._shard_row` re-slices it
per rank at serve time, so ONE sidecar serves any TP (including a TP the calibration never ran at).

Usage (inside the ROCm image, under a lease of `--tp` cards):

    PYTHONPATH=/opt/kernels:/engine/python:/engine python /engine/tools/kv_fp8_calibrate.py \
        --model cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit --tp 2 \
        --text /home/pat/fixtures/minisgl-kv-calib/kv_calib_v1.txt \
        --out /path/kv_scales.safetensors --report /path/kv_calib_report.json

`--text` is REQUIRED and must be representative: the scale it produces is a promise about the range
of everything the served model will ever store. The file's byte size and hash are echoed and written
into the sidecar metadata so a scale table can always be traced back to the data that produced it.

WHAT TO READ IN THE OUTPUT. The per-pool line reports the per-head amax SPREAD (max head / min head,
per layer). Per-head scales only earn their keep where that spread is wide — e4m3 is a floating
format, so a too-large scale mostly shifts a quiet head's exponents down at the same relative error,
and the only real loss is subnormal FLUSH. Measured: 1.6-4.4x spread on Qwen3-0.6B (per-head worth
it there: 74.4% of a wide head flushed to zero per-tensor vs 5.5% per-head), but only 1.29-1.55x on
Qwen3.5-4B, where per-head buys essentially nothing. A narrow spread is a legitimate answer of "use
the checkpoint's per-tensor scales and skip the sidecar" — `--report` writes those numbers out as
JSON so the decision is recorded rather than remembered.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

# Accumulate amax in the KV pools. Must be set BEFORE minisgl is imported (read at pool
# construction). Set at module scope so a spawned rank process — which re-imports this module before
# it runs anything — gets them on the same terms as the parent.
os.environ["MINISGL_KV_FP8_CALIBRATE"] = "1"
# Calibrate against the exact bf16 activations, never an fp8 cache (see the module docstring).
os.environ["MINISGL_KV_FP8"] = "0"

FP8_MAX = 448.0


def _chmod644(path: str) -> None:
    try:
        os.chmod(path, 0o644)
    except OSError:
        pass


def _chunks(text: str, tokenizer, ctx: int, max_chunks: int) -> list[list[int]]:
    """Split the fixture into context-sized token chunks. Chunking (rather than one giant prompt)
    keeps every chunk inside the served max_seq_len and spreads the calibration over more distinct
    positions, which is what a max-observer wants. Deterministic in (text, tokenizer, ctx): every
    rank derives the identical prompt list locally, which is what keeps a multi-rank OFFLINE run in
    lockstep without the served path's rank0->rank1 request broadcast."""
    ids = tokenizer.encode(text)
    out = [ids[i : i + ctx] for i in range(0, len(ids), ctx)]
    out = [c for c in out if len(c) >= 16]
    return out[:max_chunks]


def _gather_global_amax(local, group, rank: int, tp: int, global_heads: int):
    """Per-rank `[layers, local_heads]` amax -> GLOBAL `[layers, global_heads]`.

    Two shardings exist, both produced by `div_even(num_kv_heads, tp, allow_replicate=True)` in
    MHAKVCache:
      * SPLIT (local*tp == global): rank r owns heads [r*local, (r+1)*local) — the same contiguous,
        rank-major slice `fp8_scales._shard_row` takes at serve time, so concatenating the ranks in
        order is exactly its inverse.
      * REPLICATED (local == global, i.e. num_kv_heads < tp): every rank holds every head. The rows
        should agree; reduce with MAX anyway, so a disagreement widens the scale instead of being
        decided by whichever rank happened to be rank 0.
    The gather is a broadcast-per-source loop on the gloo CPU group — the one ProcessGroup call shape
    this codebase already relies on for its control messages.
    """
    import torch

    if tp == 1:
        return local
    rows = []
    for src in range(tp):
        buf = local.clone() if src == rank else torch.zeros_like(local)
        group.broadcast(buf, root=src).wait()
        rows.append(buf)
    local_heads = local.shape[1]
    if local_heads == global_heads:
        return torch.stack(rows).amax(dim=0)
    out = torch.cat(rows, dim=1)
    assert out.shape[1] == global_heads, (out.shape, global_heads)
    return out


def _rank_main(rank: int, args, result_q) -> None:
    """One TP rank: boot the engine, run the calibration forward, gather amax, and (rank 0) write."""
    os.environ["MINISGL_KV_FP8_CALIBRATE"] = "1"
    os.environ["MINISGL_KV_FP8"] = "0"

    import torch
    from minisgl.core import SamplingParams
    from minisgl.distributed import DistributedInfo
    from minisgl.kvcache.mha_pool import kv_amax_to_descale
    from minisgl.llm import LLM
    from minisgl.models import ModelConfig
    from minisgl.utils import cached_load_hf_config

    tp = args.tp
    raw = open(args.text, "rb").read()
    tensors = {}
    report = {"model": args.model, "tp": tp, "pools": []}
    granularity = "per_head"  # overwritten to per_layer_latent for an MLA model

    # inference_mode matches the served path (server/launch.py): no autograd graph is built for the
    # calibration forward, which on a 35B is the difference between fitting and not.
    with torch.inference_mode():
        llm = LLM(
            model_path=args.model,
            dtype=torch.bfloat16,
            tp_info=DistributedInfo(rank, tp),
            attention_backend=args.attn_backend,
            cuda_graph_max_bs=0,  # eager: capture is irrelevant here and costs boot time
            memory_ratio=args.memory_ratio,
            max_running_req=args.max_running_req,
            # The recurrent-radix snapshot store is a PREFIX-REUSE cache, and calibration chunks
            # share no prefixes — it would reserve GPU memory (0.59 GiB on Laguna) it can never use,
            # out of the same budget as the bf16 KV pool this run needs to be twice normal size.
            # Turning it off changes nothing about the K/V values being measured.
            gdn_radix=args.gdn_radix,
            # 0.0 is EngineConfig's own "no offload" default, so a model that fits is unaffected.
            weight_offload_device_gb=args.weight_offload_device_gb,
            weight_offload_gb=args.weight_offload_gb,
        )
        engine = llm.engine
        mc = ModelConfig.from_hf(cached_load_hf_config(args.model))

        pools = [(engine.kv_cache, mc.full_attn_layer_ids, "main")]
        if getattr(engine, "swa_kv_cache", None) is not None:
            pools.append((engine.swa_kv_cache, mc.swa_layer_ids, "SWA ring"))
        for pool, _, _ in pools:
            if not getattr(pool, "_calibrating", False):
                print(
                    "FAIL: KV pool is not accumulating — MINISGL_KV_FP8_CALIBRATE did not take "
                    "(it is read at POOL CONSTRUCTION, so it must be set before minisgl is imported)",
                    file=sys.stderr,
                )
                result_q.put({"rank": rank, "error": "pool not accumulating"})
                return

        prompts = _chunks(raw.decode("utf-8", "ignore"), llm.tokenizer, args.ctx, args.max_chunks)
        ntok = sum(len(p) for p in prompts)
        report["tokens"], report["chunks"] = ntok, len(prompts)
        if rank == 0:
            print(f"calibrating on {len(prompts)} chunks / {ntok} tokens (ctx={args.ctx}, tp={tp})")
        t0 = time.time()
        llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=args.max_tokens))
        if rank == 0:
            print(f"calibration forward done in {time.time() - t0:.1f}s")

        # Gather the shards into GLOBAL per-head rows BEFORE computing any scale: a scale belongs to
        # a (layer, head), so it has to come from that head's own global amax and never from a
        # rank-local reduction.
        for pool, layer_ids, name in pools:
            assert len(layer_ids) == pool.num_layers, (len(layer_ids), pool.num_layers)
            # An MLA pool accumulates ONE amax per layer (the latent is a single stored tensor with
            # no head axis) and is TP-REPLICATED, so its "global head count" is 1 and the gather
            # reduces with MAX across ranks rather than concatenating shards.
            heads = 1 if mc.is_mla else mc.num_kv_heads
            ka = _gather_global_amax(
                pool._k_amax.float().cpu(), llm.tp_cpu_group, rank, tp, heads
            )
            va = _gather_global_amax(
                pool._v_amax.float().cpu(), llm.tp_cpu_group, rank, tp, heads
            )
            if rank != 0:
                continue
            kscale, vscale = kv_amax_to_descale(ka), kv_amax_to_descale(va)
            # Per-head spread (max head / min head, per layer) — the number that decides whether a
            # per-head sidecar is worth anything on this model at all (see the module docstring).
            # Degenerates to 1.0 on MLA, where there is one column and the spread has no meaning;
            # the interesting number there is the ACROSS-LAYER range of the amax itself.
            kspread = ka.amax(dim=1) / ka.amin(dim=1).clamp(min=1e-9)
            vspread = va.amax(dim=1) / va.amin(dim=1).clamp(min=1e-9)
            if mc.is_mla:
                print(
                    f"  pool {name} (MLA latent, PER-LAYER scale): {pool.num_layers}L — "
                    f"latent amax [{ka.min():.4g}, {ka.max():.4g}] "
                    f"({float(ka.max() / ka.min().clamp(min=1e-9)):.2f}x across layers), "
                    f"descale [{kscale.min():.4g}, {kscale.max():.4g}]"
                )
            else:
                print(
                    f"  pool {name}: {pool.num_layers}L x {mc.num_kv_heads}H (global) — "
                    f"K amax [{ka.min():.4g}, {ka.max():.4g}], V amax [{va.min():.4g}, {va.max():.4g}], "
                    f"per-head spread K median {kspread.median():.2f}x max {kspread.max():.2f}x, "
                    f"V median {vspread.median():.2f}x max {vspread.max():.2f}x"
                )
            granularity = "per_layer_latent" if mc.is_mla else "per_head"
            report["pools"].append(
                {
                    "pool": name,
                    "granularity": granularity,
                    "layers": int(pool.num_layers),
                    "global_kv_heads": 1 if mc.is_mla else int(mc.num_kv_heads),
                    "k_amax_min": float(ka.min()),
                    "k_amax_max": float(ka.max()),
                    "v_amax_min": float(va.min()),
                    "v_amax_max": float(va.max()),
                    "k_spread_median": float(kspread.median()),
                    "k_spread_max": float(kspread.max()),
                    "v_spread_median": float(vspread.median()),
                    "v_spread_max": float(vspread.max()),
                    "layer_ids": [int(x) for x in layer_ids],
                }
            )
            for compact, gl in enumerate(layer_ids):
                tensors[f"model.layers.{gl}.self_attn.k_scale"] = kscale[compact].clone()
                tensors[f"model.layers.{gl}.self_attn.v_scale"] = vscale[compact].clone()

    if rank != 0:
        result_q.put({"rank": rank, "ok": True})
        return

    from safetensors.torch import save_file

    out = args.out
    if out is None:
        from minisgl.kvcache.fp8_scales import default_sidecar_path

        out = default_sidecar_path(args.model)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    meta = {
        "format": "minisgl-kv-fp8-e4m3",
        "convention": "k_scale is the DEQUANT factor: stored = k / k_scale (amax/448)",
        "granularity": granularity,
        "model": args.model,
        "calibration_tp": str(tp),
        "fixture": os.path.abspath(args.text),
        "fixture_bytes": str(len(raw)),
        "fixture_sha256_16": hashlib.sha256(raw).hexdigest()[:16],
        "tokens": str(report["tokens"]),
        "chunks": str(report["chunks"]),
        "fp8_max": str(FP8_MAX),
    }
    save_file(tensors, out, metadata=meta)
    # This process runs as ROOT inside the container, and safetensors writes 0600 — which leaves a
    # sidecar on a bind-mounted host path unreadable by the user who has to diff/serve it. Make the
    # artifact readable; a scale table is not a secret.
    _chmod644(out)
    print(f"wrote {out}: {len(tensors)} tensors")
    print(json.dumps(meta, indent=2))
    report["sidecar"], report["metadata"] = os.path.abspath(out), meta
    if args.report:
        with open(args.report, "w") as f:
            json.dump(report, f, indent=2)
        _chmod644(args.report)
        print(f"wrote report {args.report}")
    result_q.put({"rank": 0, "ok": True, "report": report})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--text", required=True, help="representative text fixture (REQUIRED)")
    ap.add_argument("--out", default=None,
                    help="sidecar path (default: the MINISGL_KV_SCALES_DIR store, else <model dir>/kv_scales.safetensors)")
    ap.add_argument("--if-missing", action="store_true",
                    help="do nothing if the engine would already find fp8-KV scales for this model")
    ap.add_argument("--report", default=None, help="write the amax/spread report as JSON here")
    ap.add_argument("--tp", type=int, default=1, help="tensor parallelism (needs a lease of --tp cards)")
    ap.add_argument("--ctx", type=int, default=2048, help="tokens per calibration chunk")
    ap.add_argument("--max-chunks", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=8, help="decode steps per chunk (also calibrates the decode store)")
    ap.add_argument("--memory-ratio", type=float, default=0.80)
    ap.add_argument("--max-running-req", type=int, default=4)
    ap.add_argument(
        "--gdn-radix", action="store_true",
        help="keep the recurrent-radix snapshot store (off by default: calibration chunks share no "
             "prefixes, so it only takes memory from the KV pool)",
    )
    ap.add_argument("--attn-backend", default="hip")
    # WEIGHT OFFLOAD. Without these a checkpoint that does not fit the card is unreachable here:
    # `bake.UnconfiguredDeviceTierError` refuses the boot from integers, because with no explicit
    # tier the resolver DERIVES it as the whole KV budget and the pool would be left nothing. That
    # is exactly the qwen4exp case, and it made the one model on this box that needs fp8-KV
    # calibration the one model that could not be calibrated -- it served with every k_scale/v_scale
    # at an implicit 1.0 instead. Mirror the SERVE's tier here: the quantity being measured is
    # amax|K| / amax|V| of real activations, and those do not depend on where the expert weights
    # live, but the boot does.
    ap.add_argument("--weight-offload-device-gb", type=float, default=0.0,
                    help="VRAM per rank for the MoE expert tier; required for an offloaded model")
    ap.add_argument("--weight-offload-gb", type=float, default=0.0,
                    help="clamp on the pinned host arena per rank (0 = no clamp)")
    args = ap.parse_args()

    if args.if_missing:
        from minisgl.kvcache.fp8_scales import resolve_kv_fp8_scales

        found = resolve_kv_fp8_scales(args.model)
        if found is not None:
            print(f"fp8-KV scales already present ({found.source}); nothing to calibrate")
            return 0
        print(f"fp8-KV: no scales for {args.model}; calibrating (first boot of this checkpoint only)")
    if not os.path.isfile(args.text):
        print(f"FAIL: --text {args.text} is not a file", file=sys.stderr)
        return 2
    raw = open(args.text, "rb").read()
    print(f"fixture {args.text}: {len(raw)} bytes, sha256[:16]={hashlib.sha256(raw).hexdigest()[:16]}")

    import multiprocessing as mp

    # One process per rank, spawned exactly as server/launch.py does (a fork would inherit a HIP
    # context). TP=1 takes the same path, so there is only one code path to keep correct.
    mp.set_start_method("spawn", force=True)
    result_q: mp.Queue = mp.Queue()
    procs = []
    for rank in range(args.tp):
        p = mp.Process(target=_rank_main, args=(rank, args, result_q), name=f"kvcalib-TP{rank}")
        p.start()
        procs.append(p)
    # Drain as the ranks finish: a rank-0 report can exceed the pipe buffer, and joining first would
    # deadlock the writer against a full pipe.
    results = []
    deadline_procs = list(procs)
    while deadline_procs:
        while not result_q.empty():
            results.append(result_q.get())
        deadline_procs = [p for p in deadline_procs if p.is_alive()]
        if deadline_procs:
            time.sleep(0.5)
    for p in procs:
        p.join()
    while not result_q.empty():
        results.append(result_q.get())
    codes = [p.exitcode for p in procs]
    if any(c != 0 for c in codes):
        print(f"FAIL: rank exit codes {codes}", file=sys.stderr)
        return 1
    if any("error" in r for r in results):
        print(f"FAIL: {results}", file=sys.stderr)
        return 1
    if len(results) != args.tp:
        print(f"FAIL: {len(results)}/{args.tp} ranks reported", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
