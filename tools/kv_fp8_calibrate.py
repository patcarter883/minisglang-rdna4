"""Offline fp8-KV calibrator — the caller of MHAKVCache.finalize_kv_calibration().

Produces `kv_scales.safetensors`, the PER-HEAD scale sidecar that `minisgl.kvcache.fp8_scales`
loads at engine boot. This is the only way to get per-head scales: per-head amax is a property of
the model's ACTIVATIONS on real text, and no checkpoint ships it (compressed-tensors' kv_cache_scheme
is `strategy: tensor`, one scalar per layer).

WHY A SEPARATE PROCESS. Changing a descale invalidates every cache entry already written under the
old one, so a served engine must have final scales before its first store. Calibration therefore
runs here, in a process whose cache is thrown away, and hands the serve a file.

WHY IT RUNS WITH THE DEFAULT bf16 CACHE. The quantity being measured is `amax|K|` / `amax|V|` of the
tensors handed to `store_kv`, which is what the cache would be asked to represent. Measuring it with
an fp8 cache already installed would feed fp8 error back into the activations of every later layer,
and would require booting the very fp8 path being configured. So: bf16 cache, exact activations,
`MINISGL_KV_FP8_CALIBRATE=1` to accumulate.

TP. This runs TP=1 (the offline LLM API is TP=1), so every KV head is local and the sidecar holds
GLOBAL per-head rows. `fp8_scales._shard_row` slices them per rank at serve time, so one sidecar
serves any TP.

Usage (inside the ROCm image, under a 1-card lease):

    PYTHONPATH=/engine/python:/engine python /engine/tools/kv_fp8_calibrate.py \
        --model Qwen/Qwen3-0.6B --text /engine/fixtures/calib.txt --out /path/kv_scales.safetensors

`--text` is REQUIRED and must be representative: the scale it produces is a promise about the range
of everything the served model will ever store. The file's byte size is echoed and written into the
sidecar metadata so a scale table can always be traced back to the data that produced it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

# Accumulate amax in the KV pools. Must be set BEFORE minisgl is imported (read at pool construction).
os.environ["MINISGL_KV_FP8_CALIBRATE"] = "1"
# Calibrate against the exact bf16 activations, never an fp8 cache (see the module docstring).
os.environ["MINISGL_KV_FP8"] = "0"

import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from minisgl.core import SamplingParams  # noqa: E402
from minisgl.llm import LLM  # noqa: E402
from minisgl.models import ModelConfig  # noqa: E402
from minisgl.utils import cached_load_hf_config  # noqa: E402

FP8_MAX = 448.0


def _chunks(text: str, tokenizer, ctx: int, max_chunks: int) -> list[list[int]]:
    """Split the fixture into context-sized token chunks. Chunking (rather than one giant prompt)
    keeps every chunk inside the served max_seq_len and spreads the calibration over more distinct
    positions, which is what a max-observer wants."""
    ids = tokenizer.encode(text)
    out = [ids[i : i + ctx] for i in range(0, len(ids), ctx)]
    out = [c for c in out if len(c) >= 16]
    return out[:max_chunks]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--text", required=True, help="representative text fixture (REQUIRED)")
    ap.add_argument("--out", default=None, help="sidecar path (default: <model dir>/kv_scales.safetensors)")
    ap.add_argument("--ctx", type=int, default=2048, help="tokens per calibration chunk")
    ap.add_argument("--max-chunks", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=8, help="decode steps per chunk (also calibrates the decode store)")
    ap.add_argument("--memory-ratio", type=float, default=0.80)
    ap.add_argument("--attn-backend", default="hip")
    args = ap.parse_args()

    if not os.path.isfile(args.text):
        print(f"FAIL: --text {args.text} is not a file", file=sys.stderr)
        return 2
    raw = open(args.text, "rb").read()
    digest = hashlib.sha256(raw).hexdigest()[:16]
    print(f"fixture {args.text}: {len(raw)} bytes, sha256[:16]={digest}")

    llm = LLM(
        model_path=args.model,
        dtype=torch.bfloat16,
        attention_backend=args.attn_backend,
        cuda_graph_max_bs=0,  # eager: capture is irrelevant here and costs boot time
        memory_ratio=args.memory_ratio,
        max_running_req=4,
    )
    engine = llm.engine
    mc = ModelConfig.from_hf(cached_load_hf_config(args.model))

    pools = [(engine.kv_cache, mc.full_attn_layer_ids)]
    if getattr(engine, "swa_kv_cache", None) is not None:
        pools.append((engine.swa_kv_cache, mc.swa_layer_ids))
    for pool, _ in pools:
        if not getattr(pool, "_calibrating", False):
            print("FAIL: KV pool is not accumulating — MINISGL_KV_FP8_CALIBRATE did not take "
                  "(is this an MHA pool? MLA has no head axis and cannot be per-head calibrated)",
                  file=sys.stderr)
            return 2

    prompts = _chunks(raw.decode("utf-8", "ignore"), llm.tokenizer, args.ctx, args.max_chunks)
    ntok = sum(len(p) for p in prompts)
    print(f"calibrating on {len(prompts)} chunks / {ntok} tokens (ctx={args.ctx})")
    t0 = time.time()
    llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=args.max_tokens))
    print(f"calibration forward done in {time.time() - t0:.1f}s")

    # Snapshot the raw amax BEFORE finalize consumes it, so the report can show the spread that
    # justifies per-head at all.
    amax = [(p._k_amax.clone(), p._v_amax.clone()) for p, _ in pools]
    for pool, _ in pools:
        pool.finalize_kv_calibration()

    tensors: dict[str, torch.Tensor] = {}
    for (pool, layer_ids), (ka, va) in zip(pools, amax):
        assert len(layer_ids) == pool.num_layers, (len(layer_ids), pool.num_layers)
        spread = (ka.amax(dim=1) / ka.amin(dim=1).clamp(min=1e-9))
        print(
            f"  pool {pool.num_layers}L x {pool.num_kv_heads}H: "
            f"K amax [{ka.min():.4g}, {ka.max():.4g}], V amax [{va.min():.4g}, {va.max():.4g}], "
            f"per-head K spread median {spread.median():.2f}x max {spread.max():.2f}x"
        )
        for compact, gl in enumerate(layer_ids):
            tensors[f"model.layers.{gl}.self_attn.k_scale"] = pool.k_descale[compact].cpu().clone()
            tensors[f"model.layers.{gl}.self_attn.v_scale"] = pool.v_descale[compact].cpu().clone()

    out = args.out
    if out is None:
        from minisgl.kvcache.fp8_scales import SIDECAR_NAME
        from minisgl.utils import download_hf_weight

        out = os.path.join(download_hf_weight(args.model), SIDECAR_NAME)
    meta = {
        "format": "minisgl-kv-fp8-e4m3",
        "convention": "k_scale is the DEQUANT factor: stored = k / k_scale (amax/448)",
        "granularity": "per_head",
        "model": args.model,
        "fixture": os.path.abspath(args.text),
        "fixture_bytes": str(len(raw)),
        "fixture_sha256_16": digest,
        "tokens": str(ntok),
        "chunks": str(len(prompts)),
        "fp8_max": str(FP8_MAX),
    }
    save_file(tensors, out, metadata=meta)
    print(f"wrote {out}: {len(tensors)} tensors")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
