"""qwen4_exp on the CARD: build a layer-subset model, run a real prefill and real decode steps.

WHY THIS EXISTS
---------------
Everything before it was static: the meta build proves the parameter SET, the loader test proves the
84 GB checkpoint streams into it, the PLE/HC parity tests pin two blocks' numerics on CPU. None of
them runs a single HIP kernel, and none of them touches the seams that only exist at forward time —
the KV pool, the attention metadata, the GDN recurrent state, the MoE dispatch, and `Context.ple`.
This is the first thing that does.

It deliberately does NOT go through `Engine`. `Engine` is what a serve needs, and a serve is blocked
on wiring the PLE runtime into it (see the report); this harness builds the same `Context` by hand so
the MODEL can be exercised now, and so that when the engine wiring lands there is a reference for
what it has to set up.

WHAT IS REAL AND WHAT IS NOT
----------------------------
  * REAL: every kernel. The HIP GDN, the paged attention backend, the NVFP4 MoE, the grouped norms,
    the hyper-connection GEMMs, the lm_head. Real KV pool, real GDN state cache, real PLE runtime
    over the real 51.2 GB NVMe n-gram table when `--real-ple` is passed.
  * SUBSET: `--layers N` (default 4) truncates the decoder to N layers. 4 is the smallest prefix that
    covers every layer kind this architecture has: 0 GDN, 1 GDN + the PLE block, 2 GDN, 3 full
    attention + the QSA indexer. `--experts E` shrinks the routed expert count.
  * RANDOM WEIGHTS by default (`--weights random`), which makes the LOGITS meaningless — they are
    checked for shape, finiteness and non-degeneracy ONLY, never for content. `--weights real` loads
    the checkpoint's own tensors for the layers in the subset.

Run (card 0, in the serve image):

    docker run --rm --device /dev/kfd --device /dev/dri --group-add video \
      --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE \
      --ipc host --shm-size 16gb -e ROCR_VISIBLE_DEVICES=0 \
      -v <worktree>:/engine -v <ckpt>:/model:ro -v <ple>:/ple:ro \
      --entrypoint bash minisgl-rdna4:m1b-20260903 -lc \
      'PYTHONPATH=/engine/python:/opt/kernels python /engine/tests/qwen4exp_gpu_forward_test.py'
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import torch

MODEL = os.environ.get("Q4E_MODEL", "/model")
PLE_DIR = os.environ.get("Q4E_PLE", "/ple")

_failures = 0


def check(name: str, got, want) -> None:
    global _failures
    ok = got == want
    _failures += not ok
    print(f"  {'ok  ' if ok else 'FAIL'} {name:52s} got={got!r:<26} want={want!r}", flush=True)


def check_true(name: str, cond, detail: str = "") -> None:
    global _failures
    _failures += not cond
    print(f"  {'ok  ' if cond else 'FAIL'} {name:52s} {detail}", flush=True)


def _subset_config(src: str, dst: str, n_layers: int, n_experts: int, ple_1based: int) -> None:
    """Write a config.json for the first `n_layers` decoder layers.

    `ple_layer_ids` is 1-BASED in this checkpoint; it is carried through unchanged when the PLE layer
    survives the truncation and dropped otherwise (a subset that excludes it must not claim it).
    """
    cfg = json.load(open(os.path.join(src, "config.json")))
    tc = cfg["text_config"]
    tc["num_hidden_layers"] = n_layers
    tc["layer_types"] = tc["layer_types"][:n_layers]
    tc["num_experts"] = n_experts
    tc["ple_layer_ids"] = [ple_1based] if ple_1based <= n_layers else []
    with open(os.path.join(dst, "config.json"), "w") as f:
        json.dump(cfg, f)


def _fill_random(model, gen: torch.Generator) -> None:
    """Give every parameter plausible values IN PLACE.

    Not `randn` everywhere: the NVFP4 leaves are a packed uint8 E2M1 blob plus a folded fp16 group
    scale, and both have to be filled in their own dtype or the MoE kernel reads NaN scales and the
    whole thing is finite-but-meaningless in a way that looks like a real failure. int64 tensors
    (`layer_multipliers`) are left alone — they are loaded from the checkpoint even in random mode.
    """
    for name, p in model.state_dict().items():
        if p.dtype == torch.uint8:  # NVFP4 packed E2M1 pairs — any byte is a legal pair
            p.random_(0, 256, generator=gen)
        elif p.dtype in (torch.int8, torch.int32, torch.int64):
            continue
        elif "scale" in name:  # group scales must be positive and O(1)
            p.uniform_(0.5, 1.5, generator=gen)
        elif name.endswith("norm.weight") or name.endswith("hc_norm.weight"):
            p.normal_(0.0, 0.02, generator=gen)
        else:
            p.normal_(0.0, 0.02, generator=gen)


@torch.inference_mode()
def main() -> int:
    """NOTE the decorator. The engine runs every forward under `torch.inference_mode()`
    (`server/launch.py:20`, `Scheduler.step`), and this model does not merely go faster without it:
    `Qwen3_5Attn` splits qkv (a multi-view op) and then q_norm/k_norm write back IN PLACE, which
    autograd forbids on a multi-view output. With grad enabled the full-attention layer raises
    "Output 0 of View is a view and is being modified inplace" — a harness artefact, not a model bug,
    but one that costs an hour if the harness omits it.
    """
    global _failures
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--experts", type=int, default=512)
    ap.add_argument("--weights", choices=("random", "real"), default="random")
    ap.add_argument("--real-ple", action="store_true", help="use the NVMe n-gram table (else random)")
    ap.add_argument("--prompt-len", type=int, default=8)
    ap.add_argument("--decode-steps", type=int, default=4)
    ap.add_argument("--attn-backend", default="rdna4")
    # With --weights real this must point at a directory holding ONLY the shards of the layers in
    # the subset (plus the four bf16 shards, which carry every layer's bf16 tensors and whose extras
    # are dropped by name). The loader globs the directory and ASSERTS every expert stack completed,
    # so pointing it at the full 196-shard checkpoint would try to stack all 48 layers.
    ap.add_argument("--model", default=MODEL)
    # Wrap every block's forward and report where a non-finite value FIRST appears. The only way to
    # turn "the logits are NaN" into a named block without guessing.
    ap.add_argument("--trace", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("FAIL: no HIP device visible (is_rocm false?) — check device passthrough")
        return 1
    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    print(f"[gpu] {torch.cuda.get_device_name(0)}  free/total="
          f"{[x >> 20 for x in torch.cuda.mem_get_info(dev)]} MiB", flush=True)

    from minisgl.distributed import set_tp_info, try_get_tp_info

    if try_get_tp_info() is None:
        set_tp_info(0, 1)
    from minisgl.layers.rotary import set_rope_device

    set_rope_device(dev)

    from minisgl import core
    from minisgl.attention import create_attention_backend
    from minisgl.gdn.metadata import build_gdn_metadata
    from minisgl.kvcache.gdn_state import GDNStateCache
    from minisgl.kvcache import create_kvcache_pool
    from minisgl.models import create_model
    from minisgl.models.config import ModelConfig
    from minisgl.moe import create_moe_backend
    from minisgl.utils import cached_load_hf_config

    tmp = tempfile.mkdtemp(prefix="q4e-gpu-")
    _subset_config(args.model, tmp, args.layers, args.experts, ple_1based=2)
    mc = ModelConfig.from_hf(cached_load_hf_config(tmp), spec_algorithm="none")
    print(f"[config] layers={mc.num_layers} gdn={mc.gdn_layer_ids} attn={mc.full_attn_layer_ids} "
          f"ple={mc.ple_layer_ids} experts={mc.num_experts} quant={getattr(mc.quant,'ct_format',None)}",
          flush=True)

    torch.set_default_dtype(torch.bfloat16)
    print("\n[1] build on the card", flush=True)
    free0 = torch.cuda.mem_get_info(dev)[0]
    if args.weights == "random":
        with torch.device(dev):
            model = create_model(mc)
        gen = torch.Generator(device=dev).manual_seed(20260903)
        _fill_random(model, gen)
        # layer_multipliers is an int64 hash constant — random values would make every n-gram row id
        # wrong, which is silent. Load the real one even in random mode.
        _load_ple_multipliers(model, mc)
    else:
        # META build, then load — exactly what `Engine` does, and it is not a style choice:
        # `BaseOP.load_state_dict` REPLACES each tensor (`setattr`) rather than copying into it, so
        # building on the card first would hold a full second copy of the weights live until the last
        # `setattr`, which does not fit.
        with torch.device("meta"):
            model = create_model(mc)
        _load_real_subset(model, mc, dev, model_dir=args.model)
    model.post_load()
    torch.cuda.synchronize()
    used = (free0 - torch.cuda.mem_get_info(dev)[0]) >> 20
    check_true("model resident on device", used > 0, f"{used} MiB")

    # ---------------- Context ----------------
    print("\n[2] context: KV pool, attention backend, MoE backend, GDN state, PLE runtime",
          flush=True)
    saved = core._GLOBAL_CTX
    core._GLOBAL_CTX = None
    # page_size 16, not 1: the native-HIP attention kernels ("hip"/"rdna4") require a KV block that
    # is a multiple of 16, and `Engine._adjust_config` overrides any smaller value for them.
    ctx = core.Context(page_size=16)
    core.set_global_ctx(ctx)

    max_running = 2
    max_seq = 64
    # `page_table` is per-TOKEN (flat slot = page*page_size + offset) — the engine's convention.
    page_size = ctx.page_size  # 16 — set at Context construction above
    ctx.page_table = page_table = torch.zeros(
        (max_running + 1, max_seq), dtype=torch.int32, device=dev
    )
    num_pages = 1 + max_running * (max_seq // page_size)  # page 0 is the dummy
    ctx.kv_cache = create_kvcache_pool(
        model_config=mc, num_pages=num_pages, page_size=page_size, dtype=torch.bfloat16, device=dev
    )
    ctx.attn_backend = create_attention_backend(args.attn_backend, mc)
    if mc.is_moe:
        ctx.moe_backend = create_moe_backend("fused")
    ctx.gdn_state = GDNStateCache(
        num_gdn_layers=mc.num_gdn_layers,
        num_slots=max_running + 2,
        conv_dim=mc.gdn_conv_dim,
        conv_kernel=mc.linear_conv_kernel_dim,
        num_v_heads=mc.linear_num_value_heads,
        head_v_dim=mc.linear_value_head_dim,
        head_k_dim=mc.linear_key_head_dim,
        dtype=torch.float32,
        ssm_dtype=torch.bfloat16,
        device=dev,
    )
    for gdn in model.iter_gdn_layers():
        gdn.warmup_conv(8)
    check("KV layers in the pool", mc.num_kv_layers, len(mc.full_attn_layer_ids))
    check("GDN state layers", mc.num_gdn_layers, len(mc.gdn_layer_ids))

    ple_rt = _make_ple_runtime(model, mc, dev, real=args.real_ple, max_seqs=max_running + 1)
    ctx.ple = ple_rt
    check_true("ctx.ple staged runtime", ple_rt is not None,
               "real NVMe table" if args.real_ple else "random in-memory table")

    # page table for the single request: table_idx 0 owns the flat token slots of pages 1..N,
    # i.e. slot ids [page_size, page_size + max_seq).
    page_table[0, :].copy_(
        torch.arange(page_size, page_size + max_seq, dtype=torch.int32, device=dev)
    )

    # ---------------- prefill ----------------
    print("\n[3] PREFILL: one request, real logits", flush=True)
    rng = np.random.default_rng(7)
    prompt = rng.integers(0, mc.vocab_size, size=args.prompt_len, dtype=np.int64)
    req = _make_req(prompt, table_idx=0)
    batch = _make_batch([req], "prefill", page_table, dev)
    ctx.attn_backend.prepare_metadata(batch)
    batch.gdn_metadata = build_gdn_metadata(
        batch, torch.tensor([1], dtype=torch.int32, device=dev), dev
    )
    ple_rt.prepare([1], [prompt])
    if args.trace:
        _install_trace(model)
    with ctx.forward_batch(batch):
        logits = model.forward()
    torch.cuda.synchronize()
    ple_rt.commit([1], [prompt])
    check("prefill logits shape", tuple(logits.shape), (1, mc.vocab_size))
    _logit_sanity("prefill", logits)

    # ---------------- decode ----------------
    print(f"\n[4] DECODE x{args.decode_steps}", flush=True)
    ids = list(prompt)
    for step in range(args.decode_steps):
        nxt = int(logits[-1].float().argmax().item())
        ids.append(nxt)
        req.append_host(torch.tensor([nxt], dtype=torch.int64))
        req.complete_one()
        batch = _make_batch([req], "decode", page_table, dev)
        ctx.attn_backend.prepare_metadata(batch)
        batch.gdn_metadata = build_gdn_metadata(
            batch, torch.tensor([1], dtype=torch.int32, device=dev), dev
        )
        tok = np.array([nxt], dtype=np.int64)
        ple_rt.prepare([1], [tok])
        with ctx.forward_batch(batch):
            logits = model.forward()
        torch.cuda.synchronize()
        ple_rt.commit([1], [tok])
        check("decode logits shape", tuple(logits.shape), (1, mc.vocab_size))
        _logit_sanity(f"decode[{step}] tok={nxt}", logits)

    core._GLOBAL_CTX = saved
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{'PASS' if not _failures else f'FAIL ({_failures} checks)'}")
    return 1 if _failures else 0


_TRACE_HITS: "list[str]" = []


def _install_trace(model) -> None:
    """Wrap every block forward so the FIRST non-finite tensor is attributed to a named block.

    Wrapping rather than editing the model: a debug hook that lives in the model is a debug hook that
    ships. Reports only the first offender per block name — a NaN propagates, so every later block is
    a consequence, not a cause.
    """
    seen = set()

    def wrap(obj, name, attr="forward"):
        fn = getattr(obj, attr)

        def inner(*a, **kw):
            out = fn(*a, **kw)
            t = out[0] if isinstance(out, tuple) else out
            if isinstance(t, torch.Tensor) and t.is_floating_point():
                bad = not bool(t.isfinite().all())
                if bad and name not in seen:
                    seen.add(name)
                    _TRACE_HITS.append(name)
                    print(f"    [trace] FIRST non-finite out of {name}  "
                          f"nan={int(t.isnan().sum())}/{t.numel()} inf={int(t.isinf().sum())}",
                          flush=True)
                elif not bad and name not in seen:
                    f = t.float()
                    print(f"    [trace] {name:44s} |x|max={f.abs().max():.4g} "
                          f"std={f.std():.4g}", flush=True)
            return out

        setattr(obj, attr, inner)

    for i, layer in enumerate(model.model.layers.op_list):
        if getattr(layer, "ple", None) is not None:
            wrap(layer.ple, f"L{i}.ple")
        wrap(layer.attn_hyper_connection, f"L{i}.attn_hc.mix", "mix")
        wrap(layer._attn_op, f"L{i}.attn")
        wrap(layer.attn_hyper_connection, f"L{i}.attn_hc.combine", "combine")
        wrap(layer.mlp_hyper_connection, f"L{i}.mlp_hc.mix", "mix")
        wrap(layer.mlp, f"L{i}.mlp")
        wrap(layer.mlp_hyper_connection, f"L{i}.mlp_hc.combine", "combine")
    wrap(model.model.hyper_connection_mixer, "mixer.mix", "mix")
    wrap(model.model.embed_tokens, "embed_tokens")


def _logit_sanity(tag: str, logits: torch.Tensor) -> None:
    """Shape/finiteness/non-degeneracy only. With random weights the VALUES mean nothing, and a test
    that asserted anything about them would be asserting noise."""
    f = logits.float()
    finite = bool(f.isfinite().all())
    spread = float(f.max() - f.min())
    check_true(f"{tag}: finite", finite, f"min={f.min():.3f} max={f.max():.3f} std={f.std():.3f}")
    check_true(f"{tag}: not constant", spread > 1e-3, f"max-min={spread:.4f}")


def _make_req(prompt: np.ndarray, table_idx: int):
    from minisgl.core import Req, SamplingParams

    class _H:  # BaseCacheHandle stand-in: nothing in the forward path reads it
        pass

    return Req(
        input_ids=torch.from_numpy(prompt.astype(np.int64)),
        table_idx=table_idx,
        cached_len=0,
        output_len=16,
        uid=0,
        sampling_params=SamplingParams(),
        cache_handle=_H(),
    )


def _make_batch(reqs, phase, page_table, dev):
    from minisgl.core import Batch

    batch = Batch(reqs=reqs, phase=phase)
    batch.padded_reqs = reqs
    pos = []
    tok = []
    rows = []
    for r in reqs:
        pos.extend(range(r.cached_len, r.device_len))
        tok.extend(int(x) for x in r.input_ids[r.cached_len : r.device_len])
        rows.extend([r.table_idx] * r.extend_len)
    batch.positions = torch.tensor(pos, dtype=torch.int32, device=dev)
    batch.input_ids = torch.tensor(tok, dtype=torch.int32, device=dev)
    batch.out_loc = page_table[
        torch.tensor(rows, dtype=torch.int64, device=dev),
        batch.positions.to(torch.int64),
    ]
    return batch


def _load_ple_multipliers(model, mc) -> None:
    """The int64 n-gram hash multipliers, from the checkpoint, even in random-weight mode."""
    import safetensors

    if not mc.ple_layer_ids:
        return
    blk = model.ple_block()
    want = "ple_embedding.layer_multipliers"
    for path in sorted(glob.glob(f"{MODEL}/model-bf16-*.safetensors")):
        with safetensors.safe_open(path, framework="pt", device="cpu") as f:
            for name in f.keys():
                if name.endswith(want) and ".ple." in name:
                    blk.ple_embedding.layer_multipliers.copy_(f.get_tensor(name))
                    return
    print(f"  WARN no {want} in the checkpoint — n-gram row ids will be wrong")


def _load_real_subset(model, mc, dev, *, model_dir: str) -> None:
    """Load the REAL checkpoint tensors for the layers in this subset, through the REAL loader.

    The filter is by NAME against the model's own declared key set, and what it drops is stated
    rather than assumed: the four bf16 shards hold every one of the 48 layers' bf16 tensors, so a
    4-layer subset legitimately sees 44 layers' worth of keys it has nowhere to put. Anything the
    model declares and the loader did NOT yield is a hard error — that is the direction that means
    a real gap.
    """
    from minisgl.models import cast_checkpoint_tensor
    from minisgl.models.weight import _load_qwen4_exp_weight

    # `cast_checkpoint_tensor` is NOT optional decoration — it is the second half of loading. This
    # checkpoint ships A_log/dt_bias as bf16 and the GDN kernels take fp32; without it the first
    # prefill dies in `gdn_prefill_wmma` with "expected scalar type Float but found BFloat16".
    want = set(model.state_dict())
    got, dropped = {}, 0
    for k, v in _load_qwen4_exp_weight(model_dir, dev, mc):
        if k in want:
            got[k] = cast_checkpoint_tensor(k, v, torch.bfloat16)
        else:
            dropped += 1
            del v
    missing = sorted(want - set(got))
    if missing:
        raise RuntimeError(
            f"{len(missing)} parameters the model declares were NOT produced by the loader "
            f"(e.g. {missing[:5]})"
        )
    print(f"  [load] {len(got)} parameters into the subset, {dropped} out-of-subset keys dropped",
          flush=True)
    model.load_state_dict(got)


def _make_ple_runtime(model, mc, dev, *, real: bool, max_seqs: int, max_tokens: int = 64):
    """`PLERuntime` over either the real NVMe table or an in-memory random one.

    The random table is NOT a stub of the gather — it is the same `ShardedRowTable` protocol with a
    different backing store, so the hash, the row ids, the staging buffer and the H2D are all the
    real ones. Only the 51.2 GB of bytes differ.

    `max_tokens` sizes the ONE static staging buffer and must cover the largest single forward the
    caller will issue — i.e. the PREFILL token count, not `max_seqs`. It was a hardcoded 64, which
    is a decode-shaped budget: any prompt past 64 tokens raised out of `stage_rows`. The caller
    knows its prefill length, so it passes it.
    """
    from minisgl.ple import PLEEmbeddingSource, PLERuntime, Qwen4ExpNGramHasher

    if not mc.ple_layer_ids:
        return None
    blk = model.ple_block()
    hasher = Qwen4ExpNGramHasher.from_checkpoint_multipliers(
        ngram_size=mc.ngram_size,
        heads_per_ngram=mc.heads_per_ngram,
        eos_token_id=mc.ngram_eos_token_id,
        checkpoint_multipliers=blk.ple_embedding.layer_multipliers.cpu().numpy(),
        vocab_size=mc.vocab_size,
        ple_layer_index=0,
        seed=mc.ngram_seed,
    )
    if real:
        src = PLEEmbeddingSource(
            ple_files=sorted(glob.glob(f"{PLE_DIR}/model-plefp8-*.safetensors")),
            meta_files=sorted(glob.glob(f"{PLE_DIR}/model-bf16-*.safetensors")),
            hasher=hasher,
            embed_dim=mc.ple_embed_dim,
            max_tokens=max_tokens,
            device=dev,
            dtype=torch.bfloat16,
        )
    else:
        src = _RandomEmbeddingSource(hasher, mc.ple_embed_dim, max_tokens, dev)
    state = blk.make_state_cache(
        num_slots=max_seqs + 1, eos_token_id=mc.ngram_eos_token_id, device=dev,
        dtype=torch.bfloat16,
    )
    return PLERuntime(source=src, state=state, max_seqs=max_seqs, device=dev)


class _RandomEmbeddingSource:
    """`PLEEmbeddingSource` with a deterministic pseudo-random row generator instead of the NVMe
    table. Same interface, same static staging buffer, same `advance` bookkeeping — see
    `minisgl/ple/source.py`. Used only when `--real-ple` is off."""

    def __init__(self, hasher, embed_dim: int, max_tokens: int, device) -> None:
        from minisgl.ple.source import PLEEmbeddingSource as _Src

        self._real = _Src
        self.hasher = hasher
        self.embed_dim = embed_dim
        self.buf = torch.zeros(max_tokens, embed_dim, dtype=torch.bfloat16, device=device)

    def stage_batch(self, state, slots, token_lists):
        total = sum(int(np.size(t)) for t in token_lists)
        g = torch.Generator(device="cpu").manual_seed(int(sum(int(t.sum()) for t in token_lists)))
        rows = torch.randn(total, self.embed_dim, generator=g, dtype=torch.float32) * 0.05
        self.buf[:total].copy_(rows.to(torch.bfloat16))
        return self.buf[:total]

    def advance(self, state, slots, token_lists):
        self._real.advance(self, state, slots, token_lists)  # type: ignore[arg-type]

    def close(self):
        pass


if __name__ == "__main__":
    sys.exit(main())
