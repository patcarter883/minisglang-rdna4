"""qwen4_exp through the REAL `Engine`/`Scheduler` WITH WEIGHT OFFLOAD ENGAGED.

WHAT THIS IS FOR
----------------
`tests/core/*` unit-test the arena, the planner, the seam and the bake on CPU tensors.
`qwen4exp_stage_b_test.py` drives the chunked loader and the arena from a bare harness. Neither of
them boots an `Engine`, and until this test NOTHING had: the whole Stage-A window
(`bake.StageASession`) was wired into `Engine.__init__` and had never run with a non-empty plan on
any model. Every gate in it was therefore green-because-inert — `enabled` was False on every serve
this repo has ever done, so `attach()`, `bind()`, `seal()` and `verify_after_capture()` all returned
at their first line.

That is exactly the failure the PLE work had just paid for: every component tested green while the
seam between the engine and them did not exist, so the served path was 0% functional. So this test's
job is not "does the arena work" — it is **is the offload arm in the serving path**.

WHAT IT ASSERTS, in increasing strength
---------------------------------------
  [1] the plan is NON-EMPTY and the session is ENABLED — an inert session passes every other check
      in this file vacuously, so this is asserted before anything else;
  [2] the boot completes: arena pinned, bake copied, `seal()`'s three gates passed, KV pool sized
      from the corrected model term;
  [3] **the seam proof**: `discover_moe_layers` re-walked on the LIVE engine model, every layer
      carries its seam, and every tensor a HOST layer's kernels will read has a `data_ptr()` inside
      a pinned arena chunk (`PinnedWeightArena.owns_pointer`). This is the load-bearing one — it is
      the only check that distinguishes "the arena holds a copy of the weights" from "the forward
      reads the arena";
  [4] `verify_after_capture()` FIRED — counted, not merely called, because the call site has existed
      since M1-A and the counter is the only way to tell a gate that ran from one that returned at
      `if not self.enabled`;
  [5] the engaged ledger carries `weight_offload.moe_resolve[host]` — i.e. a real forward went
      through the seam with a host-placed layer, which no boot-time log can prove;
  [6] the model generates: prefill, sample, decode, detokenize, with host-resident experts.

The generated TEXT is meaningless by construction (a layer prefix of a 48-layer model), so nothing
here checks quality. Shapes, counts, residency, lifecycle and finiteness only.

WHY A SUBSET
------------
48 layers at TP=1 is arithmetically impossible on this box (70.31 GiB of experts, 15.92 GiB of VRAM,
~60 GiB of pinnable RAM) and the resolver refuses it from integers. A subset is what makes offload
ENGAGE while still fitting: N layers of experts against a `--weight-offload-device-gb` below N, so
the greedy fill leaves some layers on the card and pins the rest. The mechanism under test — plan,
arena, bake, seam, accounting, capture gate — is identical at 4 layers and at 48; what a subset does
not test is capacity, which is `qwen4exp_stage_b_test.py`'s subject.

MEASURED, 2026-09-03, card 0 (RX 9070 XT), TP=1, page_size=16, memory_ratio=0.90,
`cuda_graph_max_bs=0`, `max_running_req=2`. Raw JSON in
`docs/measurements/WEIGHT_OFFLOAD_2026-09-02/offload_serve/`. Both legs are the SAME binary; the
control passes no flag at all (`--device-gb 0`), which is this repo's rule that an A/B baseline is
the old code path and never an emulation of it.

| layers | KV pages, no offload | KV pages, offload | host tier | ratio |
|--------|----------------------|-------------------|-----------|-------|
| 4      | 167,662              | **311,598**       | 4.395 GiB (3/4 layers) | 1.86x |
| 6      | 61,751               | **301,687**       | 7.324 GiB (5/6 layers) | 4.89x |
| 7      | 8,796                | **296,732**       | 8.789 GiB (6/7 layers) | 33.7x |
| 8      | OOM in `load_state_dict` | OOM in `post_load` (`convert_nvfp4_moe`) | — | — |

The last row WAS the boundary, and it was not a defect: Stage A's peak VRAM is the UN-offloaded model
by construction (the bake runs after `post_load`), so offload bought post-load VRAM — KV pool — and
bought exactly nothing at load.

STAGE B IS NOW ON THIS PATH (2026-09-03), and that ceiling moved 7 -> 32 layers. `Engine.__init__`
calls `_load_weight_chunked`, which drives `weights/stage_b.ChunkedWeightLoader` with the sink
`StageASession.chunked_sink()` hands it, so each layer is read, finalized and BAKED before the next
is read. Re-measured on the same binary, card 0, `MINISGL_WEIGHT_ARENA_CHUNK_MIB` as noted:

| layers | chunk | host tier | arena pinned | KV pages | Stage B | result |
|--------|-------|-----------|--------------|----------|---------|--------|
| 4      | 2 GiB | 4.395 GiB (3/4)   | 6.00 GiB  | **318,358** | 6.7 s   | PASS |
| 16     | 2 GiB | 20.508 GiB (14/16)| 28.00 GiB | 53,195      | 72.8 s  | PASS |
| 24     | 2 GiB | 32.227 GiB (22/24)| 44.00 GiB | 29,065      | 142.5 s | PASS |
| 32     | 3 GiB | 43.945 GiB (30/32)| 45.00 GiB | 17,001      | 224.6 s | PASS |
| 34     | 3 GiB | 48.00 GiB needed  | —         | —           | —       | capacity refusal, -890 MiB |
| 36     | 3 GiB | 51.00 GiB needed  | —         | —           | —       | capacity refusal, -3.99 GiB |
| 48     | 3 GiB | 69.00 GiB needed  | —         | —           | —       | capacity refusal, -15.52 GiB |

The 4-layer row is an A/B against the pre-Stage-B run recorded in
`offload_serve/L4_woff.json`: `plan_host_bytes`, `copied_bytes`, `seam_host_bytes` and
`seam_device_bytes` are IDENTICAL to the byte, which is what says the chunked path changes WHEN
tensors are read and not WHERE they end up. Boot went 17.9 s -> 9.2 s and the KV pool +2.2%.

What stops it at 32 is no longer VRAM and no longer the load: it is anonymous host RAM. The refusals
above are `HostArenaCapacityError` off a live `MemAvailable`, raised before a page is pinned, and the
34-layer one misses by 890 MiB — i.e. the ceiling on this box is a property of who else is resident,
not of this code. 48 layers at TP=1 needs 69.00 GiB of pinned arena against ~53 GiB of usable RAM and
no chunk size closes that; see `qwen4exp_hybrid_test.py` for the third tier that does.

TENSOR PARALLEL (`--tp`, added 2026-09-04)
-----------------------------------------
Every number in the tables above is TP=1 ON CARD 0, and that was not a choice — this file had no way
to ask for anything else, while `tools/serve.sh` DEFAULTS to TP=2 and every other model in this repo
serves there. The capacity conclusion inherited the restriction: "48 layers needs 69.00 GiB against
~53 GiB usable" is a ONE-CARD statement, and at TP=2 the ~9.2 GiB dense body shards per rank and
there are two device tiers, so both sides of that inequality move.

`--tp N` runs ONE PROCESS PER RANK, spawned by `main()` exactly as `server/launch.py` and
`tools/kv_fp8_calibrate.py` do, each building its own `LLM` with `tp_info=DistributedInfo(rank, N)`.
That is the repo's existing mechanism and this file adds no second one. `--tp 1` stays INLINE — no
process boundary it did not have before — so every TP=1 result above remains comparable.

At `--tp > 1` three TP-structure facts are asserted over the engine's own gloo group, each of which
fails SILENTLY (right shapes, plausible text) if it is wrong: hyper-connections byte-identical on
every rank (REPLICATED — the quant `ignore` list implies it, nothing else would notice a shard), the
GDN local head counts equal to `global/tp` (16 and 48 both divide 2), and the compressed-tensors sign
convention agreeing across ranks. The parent then compares the ranks' greedy token ids, which is the
one check every silent-wrong-shard failure mode surfaces in.

Run (in the serve image; `tools/run_offload_serve.sh` wraps this and exposes BOTH cards by default):

    docker run --rm --device /dev/kfd --device /dev/dri --group-add video \
      --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE \
      --ipc host --shm-size 16gb -e ROCR_VISIBLE_DEVICES=0,1 \
      -v <worktree>:/engine -v /home/pat/.cache/hf-q4e:/model:ro -v /home/pat/.cache/hf-ple:/ple:ro \
      --entrypoint bash minisgl-rdna4:m1b-20260903 -lc \
      'PYTHONPATH=/engine/python:/opt/kernels python /engine/tests/qwen4exp_offload_serve_test.py \
         --layers 4 --device-gb 2.0 --tp 2'

MEASURED AT TP=2, 2026-09-04 (cards 0+1, page_size=16, memory_ratio=0.90, `cuda_graph_max_bs=0`,
`max_running_req=2`). Raw JSON in `docs/measurements/WEIGHT_OFFLOAD_2026-09-02/offload_serve_tp2/`.

The per-rank byte counts, from the meta build (`state_dict()`, bf16 default dtype), are EXACTLY half
of TP=1's on the two tiers that shard and unchanged on the one that does not:

    routed experts   70.312 -> 35.156 GiB/rank   (1.4648 -> 0.7324 per layer)
    body              9.216 ->  5.312 GiB/rank
      of which hyper-connections 1.193 GiB, IDENTICAL at both tp — they are replicated by
      construction, so they are 12.9% of the TP=1 body and 22.5% of the TP=2 one.

| layers | chunk | device tier | host/rank | pinned/rank | KV pages | Stage B | result |
|--------|-------|-------------|-----------|-------------|----------|---------|--------|
| 4      | 3 GiB | 1.465 (2/4) | 1.465 GiB | 3.00 GiB    | 721,183  | 7.5 s   | PASS |
| 40     | 768 M | 7.32 (10/40)| 21.97 GiB | 22.50 GiB   | 12,869   | 163.5 s | PASS, **14.48 tok/s** |
| 48     | 3 GiB | 8.79 (12/48)| 26.37 GiB | 27.00 GiB   | —        | —       | capacity refusal, -6.90 GiB |

48 LAYERS STILL DOES NOT FIT, and TP=2 is not what stops it — the two-tier plan is now bounded by
VRAM on one side and by this box's other residents on the other. The refusal is exact: 27.00 GiB per
rank x 2 = 54.00 GiB against MemAvailable 59.10 GiB and a 12.00 GiB floor. That is a real
improvement (TP=1 needed 69.00 GiB and missed by 21.93) and it is still 6.90 GiB short. The device
tier cannot absorb the difference: 5.312 GiB of body + 8.79 GiB of expert tier is 14.10 GiB of a
14.33 GiB budget, i.e. 12 device layers already leaves the KV pool at ~0. Closing it needs host RAM
this box does not have free (~29 GiB is other services), the third tier
(`qwen4exp_hybrid_test.py`), or a smaller expert format — and note AWQ INT4 alone does NOT close it
either: at 0.6775 GiB/layer/rank it lands ~2.4 GiB short by the same arithmetic.

The 4-layer row is an A/B against `--device-gb 0` on the same binary (`L4_tp2_control.json`, KV
618,527): the greedy ids are IDENTICAL, so the offload path changes no number at TP=2. Both ranks
also agree with each other on every run — that cross-rank compare is what every silent-wrong-shard
failure mode surfaces in, and it is the reason the shard rules can be trusted at all.

`HIP_VISIBLE_DEVICES` must stay UNSET for TP=2: setting both it and `ROCR_VISIBLE_DEVICES` double-
filters and leaves torch with no HIP GPU (CLAUDE.md, diagnosed in Phase 3e). ROCm device 2 is the
Ryzen iGPU and is never a compute target.
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
_failures = 0


def _gib(n) -> str:
    return f"{n / GIB:.3f} GiB"


def check(name: str, got, want) -> None:
    global _failures
    ok = got == want
    _failures += not ok
    print(f"  {'ok  ' if ok else 'FAIL'} {name:52s} got={got!r:<28} want={want!r}", flush=True)


def check_true(name: str, cond, detail: str = "") -> None:
    global _failures
    _failures += not cond
    print(f"  {'ok  ' if cond else 'FAIL'} {name:52s} {detail}", flush=True)


def subset_dir(src: str, n_layers: int, n_experts: int) -> str:
    """A REAL model directory: a truncated config plus symlinks to the shards that subset needs.

    Not a config-only stand-in. `Engine` takes a `model_path` and reads a tokenizer, a generation
    config and a chat template out of it, and `load_weight` GLOBS the directory — so pointing the
    engine at `/model` with a 4-layer config still materializes all 48 layers' expert stacks and
    OOMs at the identical byte. The shard filter is what makes the subset mean anything, and it is
    the same trick `qwen4exp_stage_b_test._filtered_shard_dir` uses.
    """
    import tempfile

    from minisgl.models.weight import qwen4_exp_chunk_files
    from qwen4exp_gpu_forward_test import _subset_config

    body, per_layer = qwen4_exp_chunk_files(src, n_layers)
    d = tempfile.mkdtemp(prefix="q4e-offload-")
    for path in body + [f for lid in range(n_layers) for f in per_layer[lid]]:
        os.symlink(path, os.path.join(d, os.path.basename(path)))
    # Everything an ENGINE needs beyond weights. `config.json` is written (truncated), the rest are
    # linked verbatim: without the tokenizer files `LLM` cannot even encode a prompt, and the
    # failure is an unrelated-looking HF error three frames from here.
    for extra in (
        "generation_config.json",
        "chat_template.jinja",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
    ):
        p = os.path.join(src, extra)
        if os.path.exists(p):
            os.symlink(p, os.path.join(d, extra))
    _subset_config(src, d, n_layers, n_experts, ple_1based=2)
    return d


def _tensor_digest(t) -> str:
    """A cheap, dtype-and-shape-sensitive content hash. Used to compare a REPLICATED tensor across
    ranks, so it must be sensitive to the bytes and not merely to the shape — two ranks that hold
    different halves of a wrongly-sharded tensor agree on shape whenever the split is even."""
    import hashlib

    b = t.detach().to("cpu").contiguous().view(torch.uint8).numpy().tobytes()
    return f"{tuple(t.shape)}:{t.dtype}:{hashlib.sha256(b).hexdigest()[:16]}"


def _tp_structure(llm, tp: int, rank: int, model_dir: str) -> dict:
    """The three TP-structure facts a qwen4_exp TP=2 boot has to establish, each asserted rather
    than asserted-about.

    1. HYPER-CONNECTIONS ARE REPLICATED. The quant `ignore` list carries `*hyper_connection*`, so
       these stay bf16 — but "not quantized" is not "not sharded", and the two are easy to conflate.
       HC mixes across the `hc_count` residual streams of the WHOLE hidden state; a column-split of
       `input_mix_weight_up` would give each rank a partial mix with no all-reduce to complete it,
       which is silent (right shapes, plausible text). Checked by digesting the actual BYTES on each
       rank and comparing over the gloo group: a shape-only compare passes on any even split.
    2. GDN SHARDS. `linear_num_key_heads=16` and `linear_num_value_heads=48` both divide 2, and
       `gdn/layer.py:213` asserts it — but the assert only proves divisibility, not that the local
       count actually came out `global/tp`. Read the LOCAL `num_k_heads`/`num_v_heads` back off the
       built layer and check the product.
    3. CT SIGN AGREES ACROSS RANKS. Recorded from the engine's own post_load verification
       (`Engine.ct_sign_decisions`), so this file reports the fact rather than re-deriving it — the
       gate itself lives in the serve path, where it protects every model, not only this harness.
    """
    from minisgl.distributed import get_ep_size, is_ep_enabled, is_ep_over_tp
    from minisgl.models.config import ModelConfig
    from minisgl.utils import cached_load_hf_config

    # The repo's ONE op-tree walk. `BaseOP` is not an `nn.Module`, so `named_modules()` does not
    # exist on most of this tree — a walk written against it silently finds NOTHING, which every
    # check below would then read as a pass.
    from minisgl.weights.moe_interpose import _iter_ops

    eng = llm.engine
    ops = list(_iter_ops(eng.model, "", set()))
    # Re-read from the CHECKPOINT rather than from the engine. Neither `Engine` nor `Scheduler`
    # retains its `SchedulerConfig`, and — more to the point — the global head counts have to come
    # from a source the sharding could not have touched. Deriving "global" as `local * tp` would make
    # the check below tautological.
    mc = ModelConfig.from_hf(cached_load_hf_config(model_dir), spec_algorithm="none")
    out: dict = {}

    # -- 2. GDN -------------------------------------------------------------------------------
    gdn_local = {}
    for path, mod in ops:
        if hasattr(mod, "num_k_heads") and hasattr(mod, "num_v_heads"):
            gdn_local[path] = (int(mod.num_k_heads), int(mod.num_v_heads))
    out["gdn_layers"] = len(gdn_local)
    # A walk that finds no GDN layer is not a pass — it is a broken walk, and this model is 3/4 GDN.
    check_true("the op walk found the GDN layers at all", len(gdn_local) > 0,
               f"{len(ops)} ops walked")
    if gdn_local:
        ks = {v[0] for v in gdn_local.values()}
        vs = {v[1] for v in gdn_local.values()}
        out["gdn_local_k_heads"] = sorted(ks)
        out["gdn_local_v_heads"] = sorted(vs)
        gk = int(getattr(mc, "linear_num_key_heads", 0) or 0)
        gv = int(getattr(mc, "linear_num_value_heads", 0) or 0)
        out["gdn_global_k_heads"], out["gdn_global_v_heads"] = gk, gv
        check_true(
            f"GDN k-heads sharded {gk} -> {gk // tp} at tp={tp}",
            ks == {gk // tp} if gk else False, f"local {sorted(ks)}",
        )
        check_true(
            f"GDN v-heads sharded {gv} -> {gv // tp} at tp={tp}",
            vs == {gv // tp} if gv else False, f"local {sorted(vs)}",
        )

    # -- 3. expert parallelism ----------------------------------------------------------------
    out["ep_enabled"] = bool(is_ep_enabled())
    out["ep_over_tp"] = bool(is_ep_over_tp())
    out["ep_size"] = int(get_ep_size())

    # -- 1. hyper-connections -----------------------------------------------------------------
    # The tensors hang off the HyperConnection's CHILD ops (`input_mix_weight_up.weight`, etc.), so
    # this walks the whole subtree by path prefix rather than reading attributes off the
    # HyperConnection itself — which owns no tensor of its own and would digest to nothing.
    hc = {}
    for path, mod in ops:
        if "hyper_connection" not in path:
            continue
        for n, t in list(vars(mod).items()):
            if isinstance(t, torch.Tensor):
                hc[f"{path}.{n}"] = _tensor_digest(t)
    out["hc_tensors"] = len(hc)
    # Same reasoning: qwen4_exp is a hyper-connection architecture (hc_count=4), so zero HC tensors
    # means the walk missed them and the replication claim below would be vacuous.
    check_true("the op walk found the hyper-connection tensors at all", len(hc) > 0,
               f"{len(hc)} tensors over {len(ops)} ops")
    out["ct_sign_decisions"] = len(getattr(eng, "ct_sign_decisions", {}) or {})

    group = getattr(eng, "tp_cpu_group", None)
    if tp > 1 and group is not None:
        import torch.distributed as dist

        payload = {
            "hc": hc,
            "gdn": {k: list(v) for k, v in gdn_local.items()},
            "ct": getattr(eng, "ct_sign_decisions", {}) or {},
            "ep_size": out["ep_size"],
        }
        gathered = [None] * tp
        dist.all_gather_object(gathered, payload, group=group)
        ref = gathered[0]
        # HC: identical BYTES on every rank == replicated. This is the claim the `ignore` list only
        # implies; nothing else in the boot would notice a shard.
        hc_same = all((g or {}).get("hc") == ref["hc"] for g in gathered)
        bad = sorted(
            k for k in ref["hc"]
            if any((g or {}).get("hc", {}).get(k) != ref["hc"][k] for g in gathered)
        )
        check_true(
            "hyper-connections REPLICATED (byte-identical on every rank)",
            hc_same and len(ref["hc"]) > 0,
            f"{len(ref['hc'])} tensors, {len(bad)} differ: {bad[:4]}",
        )
        out["hc_replicated"] = bool(hc_same and ref["hc"])
        out["hc_diverging_tensors"] = bad[:8]
        # GDN: every rank must have the SAME local head counts. An asymmetric split would make one
        # rank's out_proj all-reduce contribute a differently-shaped partial.
        check_true(
            "GDN local head counts identical across ranks",
            all((g or {}).get("gdn") == ref["gdn"] for g in gathered),
        )
        # CT sign: the engine already raised on divergence, so reaching here IS the pass. Recorded
        # so a run over a checkpoint with no CT containers reads as "0 decisions", not as "checked".
        check_true(
            "CT sign decisions agree across ranks",
            all((g or {}).get("ct") == ref["ct"] for g in gathered),
            f"{len(ref['ct'])} CT container(s) — 0 means this checkpoint has none (NVFP4/MXFP4/RXF "
            f"declare no _ct_sign, so the gate is a no-op here and only the AWQ arm exercises it)",
        )
        out["ct_sign_cross_rank"] = (
            "agreed" if ref["ct"] else "not exercised: 0 CT containers in this checkpoint"
        )
    else:
        out["hc_replicated"] = None
        out["ct_sign_cross_rank"] = "tp=1: nothing to compare"
    return out


def _capture_parity(llm, args, engaged_ledger) -> dict:
    """THE GATE: does the captured decode graph compute what the eager decode computes?

    A captured graph that runs fast and computes something ELSE is the worst outcome available here
    — the text stays fluent, the tok/s improves, and nothing raises. So this is asserted before any
    throughput number is worth reading.

    THE A/B IS ON ONE BOOT, and that is the point. Both legs share this process's weights, KV pool,
    pinned arena, host tier, PLE n-gram table and sampler; the ONLY thing that differs is whether
    `GraphRunner.can_use_cuda_graph` says yes. Forcing `max_graph_bs = 0` is what turns it off:
    `pad_batch` then leaves `padded_reqs = reqs` (no dummy rows) and `Scheduler._forward` calls
    `Engine.forward_batch`'s eager path. Two separate boots would vary the KV pool sizing and the
    arena layout as well, and this repo's rule is that an A/B baseline is the OLD CODE PATH — here
    that path is literally still present in the same binary, so use it.

    WHAT "IDENTICAL" MEANS. Greedy token IDS over `--parity-steps` steps, not bitwise logits, and
    the difference is not pedantry:
      * `attn_decode`'s split-K policy is keyed on the page-table WIDTH (`max_blocks * block_size`),
        and the captured path's table is the full `aligned_max_seq_len` while the eager path's is
        this batch's own `max_seqlen_k`. Different `num_splits` = a different fp32 reduction ORDER,
        and at `num_splits == 1` literally a different kernel (single-pass vs split+reduce). That is
        pre-existing and shared with every other captured model in this repo, but it means bitwise
        equality is not the available claim.
      * at TP>1 the eager path runs collectives on a SIDE STREAM with the FFN row-split, while
        capture is `inline_collectives`-transparent — `GraphRunner.replay_canvas` documents this
        exact trap. Two things vary at once, and ids are the level at which that is still a clean
        statement.
    Ids are also the level the failure modes this is hunting actually show at: a stale n-gram
    embedding, a frozen conv state, a page table the graph reads at the wrong stride, a dispatch
    that silently changed arm — none of those are last-bit effects.

    ALSO ASSERTED HERE: the PLE prepare/commit ledger across both legs (`commit_staged` exactly once
    per forward), and that the engaged ledger has not LOST an arm — a vanished
    `weight_offload.moe_resolve[host]` would mean capture silently changed MoE dispatch, which no
    throughput bench can see.
    """
    from minisgl.core import SamplingParams

    gr = llm.engine.graph_runner
    ple_rt = getattr(llm.engine, "ple_runtime", None)
    widths = [int(w) for w in str(args.parity_batches).split(",") if w.strip()]
    o: dict = {"parity_steps": int(args.parity_steps), "parity_widths": widths}
    before_all = ple_rt.counters() if ple_rt is not None else {}
    per_width: dict = {}
    verdicts: list = []

    for w in widths:
        per_width[str(w)] = _parity_one_width(llm, gr, ple_rt, args, w)
        verdicts.append(f"bs={w}: {per_width[str(w)]['verdict']}")
    o["parity_by_width"] = per_width
    o["replay_matches_eager"] = " | ".join(verdicts)

    # Back-compat keys for the bs=1 leg, which is what earlier rounds' JSON carried.
    if "1" in per_width:
        one = per_width["1"]
        o["parity_captured_ids"] = one["captured_ids"]
        o["parity_eager_ids"] = one["eager_ids"]
        o["parity_captured_text"] = one["captured_text"]
        o["parity_eager_text"] = one["eager_text"]
        o["parity_captured_seconds"] = one["captured_seconds"]
        o["parity_eager_seconds"] = one["eager_seconds"]

    if ple_rt is not None:
        after_all = ple_rt.counters() if ple_rt is not None else {}
        o.update({f"parity_{k}": v for k, v in after_all.items()})
        # Discards are the capturer's synthetic batches and NOTHING else: one per recorded graph,
        # banked at boot, never incremented again. If a parity leg had discarded a stage, the
        # n-gram history would be one pass ahead of the conv state with no error anywhere.
        check("no PLE stage was discarded outside capture",
              after_all["ple_discards"], len(getattr(gr, "graph_map", {})))
        check("no forward ran with nothing staged", after_all["ple_commit_noops"], 0)
        check("every prepare across all parity legs was committed",
              after_all["ple_prepares"] - before_all["ple_prepares"],
              after_all["ple_commits"] - before_all["ple_commits"])

    marks = sorted(n for n in engaged_ledger if n.startswith("weight_offload."))
    o["engaged_after_capture"] = marks
    check_true("weight_offload.moe_resolve[host] SURVIVES capture",
               "weight_offload.moe_resolve[host]" in marks,
               "an arm that vanishes after capture is a silent dispatch regression benches cannot see")
    return o


# Distinct prompts, so a width-N leg is N genuinely different sequences rather than N copies of one
# (identical rows would hide any row-indexing bug — every row would be right by symmetry).
_PARITY_PROMPTS = [
    "The capital of France is",
    "Write one sentence about bicycles",
    "The largest ocean on Earth is",
    "In 1969 the first humans landed on",
]


class _Steps:
    """Trivial holder so the code below reads the same whether the trace came straight off a tap or
    was captured by an earlier leg."""

    def __init__(self, steps):
        self.steps = steps


class _LogitTap:
    """Record the per-step, per-row top-2 logits of whatever the sampler was handed.

    WHY THIS EXISTS. When a captured leg and an eager leg disagree on a greedy id, "they disagree"
    is not a diagnosis — it is the start of one, and the two candidate causes have OPPOSITE
    consequences. Either (a) the two legs' logits differ by a lot, which is a real capture bug and
    blocks everything, or (b) the two legs' logits agree to within float noise but the model's own
    top-2 MARGIN at that step is smaller than that noise, so argmax is a coin flip and the ids were
    never going to be a stable observable. This repo already knows the captured and eager decode
    attention run DIFFERENT split-K reduction orders (the captured page table is the full
    `aligned_max_seq_len`, the eager one is the batch's own `max_seqlen_k`), so a nonzero logit
    delta is EXPECTED; the question is only whether it is small next to the margin it flipped.

    Taps `sampler.sample`, not the forward, because that is the one place both the captured
    (`graph_runner.replay`) and the eager (`model.forward`) paths funnel through with the batch's
    own logits already sliced to `batch.size` — so the tap is path-agnostic by construction and
    cannot itself be what differs between the legs.

    Each call syncs (`.tolist()`), so this is scoped to the parity legs and never wraps a
    throughput measurement.
    """

    def __init__(self, sampler):
        self.sampler = sampler
        self._orig = sampler.sample
        self.steps: list = []

    def __enter__(self):
        def tap(logits, args_):
            f = logits.detach().float()
            if f.dim() == 2 and f.shape[-1] >= 2:
                top = torch.topk(f, 2, dim=-1)
                self.steps.append(
                    {"ids": top.indices.tolist(), "vals": [[round(x, 5) for x in r]
                                                          for r in top.values.tolist()]}
                )
            return self._orig(logits, args_)

        self.sampler.sample = tap
        return self

    def __exit__(self, *a):
        self.sampler.sample = self._orig
        return False


def _margin_report(cap_steps: list, eag_steps: list) -> dict:
    """Compare two top-2 traces step by step and price every top-1 disagreement.

    Returns, for the first step at which the legs' argmax differs: the model's own top-2 margin on
    each leg, and the cross-leg delta on the SAME logit. `margin_exceeds_delta = False` is the
    signature of a near-tie flipped by reduction-order noise; `True` means the legs genuinely
    disagree about the answer and capture is wrong.
    """
    rep: dict = {"steps_compared": min(len(cap_steps), len(eag_steps)),
                 "captured_steps": len(cap_steps), "eager_steps": len(eag_steps),
                 "disagreements": []}
    worst_delta = 0.0
    for s, (c, e) in enumerate(zip(cap_steps, eag_steps)):
        if len(c["ids"]) != len(e["ids"]):
            rep["disagreements"].append({"step": s, "why": "row count differs",
                                         "captured_rows": len(c["ids"]),
                                         "eager_rows": len(e["ids"])})
            continue
        for r, (ci, ei) in enumerate(zip(c["ids"], e["ids"])):
            cv, ev = c["vals"][r], e["vals"][r]
            # Cross-leg delta on the top-1 logit VALUE is the cleanest scale for "how much did the
            # two legs' arithmetic differ", and it is comparable to the margins below because all
            # three are logits of the same row at the same step.
            d = abs(cv[0] - ev[0])
            worst_delta = max(worst_delta, d)
            if ci[0] != ei[0]:
                cm, em = cv[0] - cv[1], ev[0] - ev[1]
                rep["disagreements"].append({
                    "step": s, "row": r,
                    "captured_top1": ci[0], "eager_top1": ei[0],
                    "captured_top2": ci[1], "eager_top2": ei[1],
                    "captured_margin": round(cm, 5), "eager_margin": round(em, 5),
                    "cross_leg_top1_delta": round(d, 5),
                    # The honest test: is the margin that flipped BIGGER than the arithmetic
                    # difference between the legs? If not, the flip is noise on a tie.
                    "margin_exceeds_delta": bool(min(cm, em) > max(d, 1e-6) * 4),
                    "swapped_top2": bool(ci[0] == ei[1] and ci[1] == ei[0]),
                })
    rep["worst_cross_leg_top1_delta"] = round(worst_delta, 5)
    rep["n_disagreements"] = len(rep["disagreements"])
    return rep


def _parity_one_width(llm, gr, ple_rt, args, width: int) -> dict:
    """One captured-vs-eager A/B at decode batch width `width`, inside this one boot."""
    from minisgl.core import SamplingParams

    prompts = _PARITY_PROMPTS[:width]
    assert len(prompts) == width, f"only {len(_PARITY_PROMPTS)} parity prompts, asked for {width}"
    sp = SamplingParams(temperature=0.0, max_tokens=args.parity_steps)
    d: dict = {"width": width, "prompts": prompts,
               "captured_bucket_exists": width in list(getattr(gr, "graph_bs_list", []))}

    print(f"\n[6b] capture parity bs={width}: {args.parity_steps} greedy steps, "
          f"captured vs eager (one boot)", flush=True)
    if not d["captured_bucket_exists"]:
        # Not a silent skip: a width with no bucket does not exercise capture at all, so scoring it
        # as "identical" would be scoring eager against eager.
        print(f"  NOTE: no captured bucket for bs={width} (buckets={gr.graph_bs_list}); "
              f"the 'captured' leg would fall through to eager", flush=True)

    def run_leg():
        with _LogitTap(llm.engine.sampler) as tap:
            t0 = time.perf_counter()
            res = llm.generate(prompts, sp)
        return res, tap.steps, time.perf_counter() - t0

    before = ple_rt.counters() if ple_rt is not None else {}
    # THE DETERMINISM CONTROL, and it runs FIRST because everything below is meaningless without it.
    # `captured vs eager` is only a statement about the capture mechanism if each leg is a fixed
    # function of its inputs. This repo already records that the serve is NOT bit-reproducible past
    # ~32 tokens, and the canvas-graph gate learned the same lesson the expensive way (it reported a
    # 1.575e+01 "capture delta" that was really two programs being compared). So each leg is run
    # TWICE on identical inputs; if a leg disagrees with ITSELF, the run-to-run floor is the finding
    # and no captured-vs-eager id comparison can be read as a capture verdict.
    cap, cap_steps, t_c = run_leg()
    d["captured_seconds"] = round(t_c, 3)
    cap2, cap2_steps, _ = run_leg()
    mid = ple_rt.counters() if ple_rt is not None else {}

    saved = gr.max_graph_bs
    try:
        # OFF. `can_use_cuda_graph` is `batch.is_decode and batch.size <= self.max_graph_bs`, so 0
        # forces every decode through the eager forward; `pad_batch`'s bucket lookup is guarded by
        # the same predicate, so nothing reads `graph_bs_list` while it is off.
        gr.max_graph_bs = 0
        eag, eag_steps, t_e = run_leg()
        d["eager_seconds"] = round(t_e, 3)
        eag2, eag2_steps, _ = run_leg()
    finally:
        gr.max_graph_bs = saved
    after = ple_rt.counters() if ple_rt is not None else {}

    # Self-consistency of each mechanism, on the same scale as the cross-leg number below.
    ids_c2 = [list(r["token_ids"]) for r in cap2]
    ids_e2 = [list(r["token_ids"]) for r in eag2]
    d["captured_self"] = _margin_report(cap_steps, cap2_steps)
    d["eager_self"] = _margin_report(eag_steps, eag2_steps)
    d["captured_self_ids_match"] = ([list(r["token_ids"]) for r in cap] == ids_c2)
    d["eager_self_ids_match"] = ([list(r["token_ids"]) for r in eag] == ids_e2)
    d["run_to_run_floor"] = max(d["captured_self"]["worst_cross_leg_top1_delta"],
                                d["eager_self"]["worst_cross_leg_top1_delta"])
    print(f"  determinism control: captured-vs-itself delta="
          f"{d['captured_self']['worst_cross_leg_top1_delta']} ids_match={d['captured_self_ids_match']}"
          f" | eager-vs-itself delta={d['eager_self']['worst_cross_leg_top1_delta']}"
          f" ids_match={d['eager_self_ids_match']}", flush=True)
    tap_c, tap_e = _Steps(cap_steps), _Steps(eag_steps)

    ci = [list(r["token_ids"]) for r in cap]
    ei = [list(r["token_ids"]) for r in eag]
    d["captured_ids"], d["eager_ids"] = ci, ei
    d["captured_text"] = [r["text"] for r in cap]
    d["eager_text"] = [r["text"] for r in eag]
    same = ci == ei
    if same:
        d["verdict"] = f"IDENTICAL over {sum(len(x) for x in ci)} greedy ids in {width} row(s)"
    else:
        firsts = []
        for r, (a, b) in enumerate(zip(ci, ei)):
            k = next((j for j, (x, y) in enumerate(zip(a, b)) if x != y), None)
            if k is not None:
                firsts.append(f"row{r}@step{k}")
        d["verdict"] = "DIVERGED: " + (",".join(firsts) or "length mismatch")
    d["margins"] = _margin_report(tap_c.steps, tap_e.steps)

    for r in range(width):
        print(f"  row{r} captured: {ci[r]}\n  row{r} eager:    {ei[r]}", flush=True)
    print(f"  -> {d['verdict']}", flush=True)
    print(f"  worst cross-leg top-1 logit delta: "
          f"{d['margins']['worst_cross_leg_top1_delta']}  "
          f"({d['margins']['n_disagreements']} argmax disagreement(s) over "
          f"{d['margins']['steps_compared']} steps)", flush=True)
    for dis in d["margins"]["disagreements"][:6]:
        print(f"     {dis}", flush=True)

    # THE GATE, stated against the RUN-TO-RUN FLOOR rather than against zero.
    #
    # `captured ids == eager ids` is the right gate ONLY IF each mechanism reproduces itself. When
    # it does (floor == 0 and both self-comparisons match ids), any id difference is attributable to
    # capture and must fail. When it does NOT — when the eager path already disagrees with a second
    # eager run — then the engine's own noise is larger than the effect being measured, and failing
    # capture for a difference the baseline also produces would be blaming the wrong mechanism. In
    # that case the honest gate is that the cross-leg delta is not LARGER than the floor: capture
    # must not add error beyond what the engine already has run to run.
    floor = d["run_to_run_floor"]
    deterministic = (d["captured_self_ids_match"] and d["eager_self_ids_match"] and floor == 0.0)
    d["engine_deterministic_at_this_config"] = deterministic
    cross = d["margins"]["worst_cross_leg_top1_delta"]
    real = [x for x in d["margins"]["disagreements"] if x.get("margin_exceeds_delta")]
    d["real_disagreements"] = len(real)
    if deterministic:
        check_true(f"bs={width}: captured replay == eager greedy ids (engine is deterministic here)",
                   same, d["verdict"])
    else:
        # NOT a pass-by-default. The floor is reported, the comparison is made against it, and the
        # run is explicitly marked as unable to make the strict claim.
        check_true(
            f"bs={width}: capture adds no error beyond the engine's run-to-run floor",
            cross <= max(floor * 2.0, 1e-6),
            f"cross-leg worst delta {cross} vs run-to-run floor {floor} "
            f"(captured self-match={d['captured_self_ids_match']}, "
            f"eager self-match={d['eager_self_ids_match']}) — ids: {d['verdict']}"
        )
    check_true(f"bs={width}: both legs produced tokens",
               all(len(x) > 0 for x in ci) and all(len(x) > 0 for x in ei))
    check_true(f"bs={width}: the legs ran the same number of sampler steps",
               len(tap_c.steps) == len(tap_e.steps),
               f"captured {len(tap_c.steps)} vs eager {len(tap_e.steps)}")

    if ple_rt is not None:
        cap_pre = mid["ple_prepares"] - before["ple_prepares"]
        cap_com = mid["ple_commits"] - before["ple_commits"]
        eag_pre = after["ple_prepares"] - mid["ple_prepares"]
        eag_com = after["ple_commits"] - mid["ple_commits"]
        d["ple_captured_leg"] = {"prepares": cap_pre, "commits": cap_com}
        d["ple_eager_leg"] = {"prepares": eag_pre, "commits": eag_com}
        # ONE commit per forward, on BOTH legs. The captured leg is the one that could go wrong:
        # `_stage_ple` runs in `_finish_prepare` and `commit_staged` in `_forward`, and neither moved
        # for capture — but the capturer stages batches of its own, so the arithmetic is the proof.
        check(f"bs={width} captured leg: one commit_staged per prepare",
              (cap_pre, cap_com), (cap_com, cap_com))
        check(f"bs={width} eager leg: one commit_staged per prepare",
              (eag_pre, eag_com), (eag_com, eag_com))
        # A decode step is one forward, and both legs generated `parity_steps` tokens from one
        # prefill — so each leg must have run at least `parity_steps - 1` decode forwards. Stated as
        # a lower bound rather than an equality because chunked prefill can split the prompt.
        check_true(f"bs={width}: the captured leg ran a forward per step",
                   cap_com >= args.parity_steps - 1,
                   f"{cap_com} commits for {args.parity_steps} tokens")
    return d


def _weight_digest(llm, args) -> dict:
    """A BYTE-EXACT content hash of every weight tensor the built model can reach on this rank.

    WHY THIS EXISTS. A boot-time change that only moves bytes faster has exactly one thing it must
    prove: the bytes are the same. Every other gate in this file is an accounting identity — plan
    digest, region counts, `arena_pinned_bytes`, `copied_bytes`, `stage_b_keys_filled` — and every
    one of them is a statement about the LEDGER, not about the DATA. All of them pass over an arena
    that was filled with the right NUMBER of the wrong bytes. The arena's own `selftest_light` is a
    512-probe SAMPLE of a 1.34 GiB chunk (one probe per 2.7 MiB) compared against fingerprints the
    arena itself wrote, so it proves the mapping is not aliased; it says nothing about whether the
    CHECKPOINT landed correctly. This does, and it is not a sample: every byte of every tensor.

    READ THROUGH THE POINTER THE KERNEL USES. A host-placed expert stack lives in `hipHostMalloc`
    pages addressed through `hipHostGetDevicePointer`, so its tensor reports `device='cuda'`. The
    copy below therefore goes out through the same mapping a MoE kernel dereferences, which is the
    stronger statement: Phase 0 found this driver serving STALE pages at a live VA with every HIP
    call returning success, and a digest taken from the host virtual address would not have seen it.

    TWO WALKS, because one does not reach everything:
      1. `granule._iter_tensors(model)` descends `BaseOP`/`nn.Module` children, so it reaches every
         dense weight, norm, embedding and GDN projection — but NOT a plain expert-container object
         hanging off a layer attribute, because the walker only descends into module-like nodes.
      2. `seam.live_tensors()` per discovered MoE layer, which is the canonical tensor per alias
         group for exactly the containers walk 1 misses. This is the same enumeration
         `prove_seam_residency` asks the arena about, so the digest covers precisely the tensors
         residency was proven for.
    Names are unioned, so a tensor both walks reach is hashed once and the count stays physical.
    """
    import hashlib

    from minisgl.weights import granule
    from minisgl.weights.moe_interpose import discover_moe_layers
    from minisgl.weights.stacks import StackKind

    model = llm.engine.model
    arena = getattr(getattr(getattr(llm.engine, "_woff", None), "driver", None), "arena", None)

    named: dict = {}
    for name, t in granule._iter_tensors(model, "", set()):
        named.setdefault(name, t)
    n_model_walk = len(named)
    live = dict(discover_moe_layers(model))
    seam_kind: dict = {}
    for path, layer in sorted(live.items()):
        seam = getattr(layer, "_weight_offload", None)
        if seam is None:
            continue
        for name, t in seam.live_tensors():
            named.setdefault(name, t)
            seam_kind[name] = seam.kind

    # ONE staging buffer, reused. 34 GiB of weights hashed through fresh allocations would add tens
    # of GiB of transient RSS to a process whose whole defect is that this box reclaims under
    # exactly that pressure — the digest would perturb the thing it is auditing.
    CH = 32 << 20
    stage = torch.empty(CH, dtype=torch.uint8, device="cpu")
    mv = memoryview(stage.numpy())

    def _hash(t) -> tuple:
        """(hexdigest, nbytes). Streams the tensor's BYTES; never its numbers."""
        x = t.detach()
        if not x.is_contiguous():
            x = x.contiguous()
        flat = x.reshape(-1)
        flat = flat.view(torch.uint8) if flat.numel() else flat
        n = flat.numel()
        h = hashlib.blake2b(digest_size=16)
        h.update(f"{tuple(t.shape)}|{t.dtype}|{n}|".encode())
        off = 0
        while off < n:
            k = min(CH, n - off)
            stage[:k].copy_(flat[off : off + k])
            h.update(mv[:k])
            off += k
        return h.hexdigest(), n

    o: dict = {"weight_digest_model_walk_tensors": n_model_walk}
    print(f"\n[8] weight digest: {len(named)} tensors "
          f"({n_model_walk} from the model walk, {len(seam_kind)} from the seams)", flush=True)
    t0 = time.perf_counter()
    per_tensor: dict = {}
    total_bytes = host_bytes = device_bytes = 0
    host_tensors = device_tensors = 0
    arena_owned = 0
    failures: list = []
    for name in sorted(named):
        t = named[name]
        try:
            d, n = _hash(t)
        except Exception as e:  # recorded, never swallowed — a tensor that cannot be read is a fact
            failures.append(f"{name}: {e!r}")
            continue
        per_tensor[name] = d
        total_bytes += n
        kind = seam_kind.get(name)
        if kind is StackKind.HOST:
            host_tensors += 1
            host_bytes += n
            if arena is not None and arena.owns_pointer(t.data_ptr(), n):
                arena_owned += 1
        else:
            device_tensors += 1
            device_bytes += n
    o["weight_digest_seconds"] = round(time.perf_counter() - t0, 2)

    # PER LAYER, so a mismatch says WHICH layer rather than only that one exists. The bucket key is
    # the `model.layers.<i>` prefix the plan and the seam paths both use.
    per_layer: dict = {}
    for name in sorted(per_tensor):
        parts = name.split(".")
        key = "body"
        for i, p in enumerate(parts[:-1]):
            if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                key = f"layer{int(parts[i + 1]):03d}"
                break
        per_layer.setdefault(key, hashlib.blake2b(digest_size=8))
        per_layer[key].update(f"{name}={per_tensor[name]}\n".encode())
    o["weight_digest_per_layer"] = {k: v.hexdigest() for k, v in sorted(per_layer.items())}

    agg = hashlib.blake2b(digest_size=16)
    for name in sorted(per_tensor):
        agg.update(f"{name}={per_tensor[name]}\n".encode())
    o.update(
        weight_digest=agg.hexdigest(),
        weight_digest_tensors=len(per_tensor),
        weight_digest_bytes=int(total_bytes),
        weight_digest_host_tensors=int(host_tensors),
        weight_digest_host_bytes=int(host_bytes),
        weight_digest_device_tensors=int(device_tensors),
        weight_digest_device_bytes=int(device_bytes),
        weight_digest_arena_owned_tensors=int(arena_owned),
        weight_digest_layer_buckets=len(o["weight_digest_per_layer"]),
        weight_digest_failures=failures,
    )
    print(f"  digest {o['weight_digest']}  over {o['weight_digest_tensors']} tensors / "
          f"{_gib(total_bytes)} in {o['weight_digest_seconds']} s", flush=True)
    print(f"  host {host_tensors} tensors / {_gib(host_bytes)} ({arena_owned} arena-owned)   "
          f"device {device_tensors} tensors / {_gib(device_bytes)}", flush=True)
    print(f"  layer buckets: {o['weight_digest_layer_buckets']}", flush=True)

    # NON-VACUITY, GATED rather than reported. Each of these has been a real shape of green: a
    # digest over zero tensors; a digest that never left the device tier, so the offloaded 25.7 GiB
    # — the only bytes this fix moves differently — went unhashed; and a "host" digest read from a
    # pointer the arena does not own, i.e. from a VRAM copy, which would make the whole comparison a
    # statement about the wrong memory.
    check_true("weight digest covered every layer + the body",
               o["weight_digest_layer_buckets"] == args.layers + 1,
               f"{o['weight_digest_layer_buckets']} buckets for {args.layers} layers")
    check_true("weight digest is non-vacuous (bytes hashed)", total_bytes > 0, _gib(total_bytes))
    check_true("weight digest covers the HOST tier (the bytes this change moves)",
               host_bytes > 0, f"{host_tensors} tensors / {_gib(host_bytes)}")
    check_true("every hashed HOST tensor was read through an ARENA-OWNED pointer",
               arena is not None and arena_owned == host_tensors,
               f"{arena_owned}/{host_tensors}")
    check("no tensor failed to hash", len(failures), 0)
    return o


def _repro_probe(llm, args) -> dict:
    """Is the engine a fixed function of its inputs? Asked WITHOUT capture in the picture at all.

    WHY. The captured-vs-eager gate compares two ADJACENT `generate` calls. That is only a statement
    about capture if the Nth and (N+1)th identical calls agree with each other in the first place.
    The determinism control in `_parity_one_width` showed they do not, and showed something sharper
    than plain noise: call 1 matched call 3 while disagreeing with call 2. A period-2 signature is
    not what floating-point nondeterminism looks like — it is what a per-request resource that
    ALTERNATES looks like, and with `max_running_req=2` the recurrent-state slot a single sequential
    request is assigned alternates 0,1,0,1.

    So this probe runs N identical single-request generates in ONE mode and compares EVERY pair. It
    never touches `max_graph_bs`, so whatever it finds is the engine's, not capture's. Read the
    `same_as_first` row: `[T,F,T,F]` is slot parity, `[T,F,F,F]` with no pattern is float noise, and
    all-True means the engine is reproducible and the strict capture gate is the right one.
    """
    from minisgl.core import SamplingParams

    gr = llm.engine.graph_runner
    n = int(args.repro_probe)
    prompt = _PARITY_PROMPTS[0]
    sp = SamplingParams(temperature=0.0, max_tokens=args.parity_steps or 12)
    o: dict = {"repro_probe_n": n}
    print(f"\n[6d] reproducibility probe: {n} identical generates per mode, capture NOT varied",
          flush=True)

    def series(label: str) -> dict:
        runs = [list(llm.generate([prompt], sp)[0]["token_ids"]) for _ in range(n)]
        eq = [[runs[i] == runs[j] for j in range(n)] for i in range(n)]
        same_first = [runs[0] == r for r in runs]
        # Slot parity predicts runs i and j agree exactly when i and j have the same parity.
        parity = all(eq[i][j] == ((i - j) % 2 == 0) for i in range(n) for j in range(n))
        d = {"runs": runs, "equality_matrix": eq, "same_as_first": same_first,
             "all_identical": all(same_first),
             "matches_slot_parity_prediction": parity,
             "distinct_outputs": len({tuple(r) for r in runs})}
        print(f"  {label}: same_as_first={same_first} distinct={d['distinct_outputs']} "
              f"slot_parity_pattern={parity}", flush=True)
        return d

    o["repro_captured"] = series("capture ON ")
    saved = gr.max_graph_bs
    try:
        gr.max_graph_bs = 0
        o["repro_eager"] = series("capture OFF")
    finally:
        gr.max_graph_bs = saved
    # REPORTED, not gated. This probe exists to attribute a nondeterminism that is already known to
    # be present; failing the run here would fail it for a pre-existing engine property that has
    # nothing to do with the capture work being discharged.
    o["repro_engine_is_reproducible"] = bool(
        o["repro_captured"]["all_identical"] and o["repro_eager"]["all_identical"])
    return o


# ---------------------------------------------------------------------------------------------
# HYPER-CONNECTION FUSION, AS A RUNTIME A/B
# ---------------------------------------------------------------------------------------------
# The HC fusion (commit 5c64fc8a) is four changes that are all claimed EXACT, and its own gates are
# a unit parity test plus an in-boot probe on a 40-LAYER boot. Neither is the served 48-layer model,
# and the merge left no switch to ask the question on one: `post_load` rewrites the weights in
# place, so "fusion off" is not a flag, it is a different set of tensors.
#
# It IS reversible, though, and exactly reversible, which is the whole point of the claim: the fold
# is a multiply by a power of two and the pack is a copy. So this pair of helpers reconstructs the
# PRE-FUSION module state from the post-fusion one and rebinds the pre-fusion `mix`/`combine`
# bodies (transcribed from the reference math in the class docstring, which is what the fused code
# replaced), runs the same greedy prompt, and compares token ids.
#
# ALL FOUR changes are reverted, not just the two that move weights, so a green result covers the
# whole commit:
#   1. the `(1 + w)` memo -> `_gain_vec` recomputes per call
#   2. the `/ hc` weight fold -> weights multiplied back by hc, runtime divides restored
#   3. the [324, 10240] pack -> two separate contiguous buffers, two GEMVs
#   4. `torch.add(alpha=2.0)` -> an explicit `2 * sigmoid(...)` then a plain add
#
# The legs run EAGER on both sides. A captured graph baked the fused pointers at capture time, so
# replaying it after an unfuse would read the packed buffer and measure nothing; capture-vs-eager is
# a separate gate that runs with fusion ON.


def _hc_blocks(llm) -> list:
    """Every `HyperConnection` in the live model, via the repo's ONE op-tree walk."""
    from minisgl.layers.hyperconnection import HyperConnection
    from minisgl.weights.moe_interpose import _iter_ops

    seen: set[int] = set()
    found = []
    for _, op in _iter_ops(llm.engine.model, "", set()):
        if isinstance(op, HyperConnection) and id(op) not in seen:
            seen.add(id(op))
            found.append(op)
    return found


def _hc_legacy_mix(self, hyper_input):
    """`mix` as it was BEFORE 5c64fc8a — one GEMV per projection, the `/ hc` at runtime.

    Transcribed from `git show 5c64fc8a^:python/minisgl/layers/hyperconnection.py`, not
    reconstructed from the class docstring: an A/B whose "off" leg is a paraphrase of the old code
    measures the paraphrase. The only intentional difference is the residual tuple, which carries a
    third `None` so the CURRENT `combine` signature still accepts it — and `None` is exactly the
    value that makes `combine` issue the inject GEMV itself, i.e. the pre-fusion shape."""
    import torch.nn.functional as _F

    normed = self.hc_norm.forward(hyper_input)
    if hyper_input.shape[0] == 0:
        empty_inj = (
            hyper_input.new_empty((*hyper_input.shape[:-1], self._hc))
            if self._use_combine
            else None
        )
        return hyper_input.new_empty((*hyper_input.shape[:-1], self._hs)), (
            hyper_input, normed, empty_inj)
    down = self.input_mix_weight_down.forward(normed) / self._hc
    t = _F.silu(down)
    gate = torch.sigmoid(self.input_mix_weight_up.forward(t))
    mixed = (
        gate.unflatten(-1, (self._hc, self._hs))
        * normed.unflatten(-1, (self._hc, self._hs))
    ).mean(dim=-2)
    # inject_logits=None -> `combine` issues the inject GEMV itself, which is the pre-fusion shape.
    return mixed, (hyper_input, normed, None)


def _hc_legacy_combine(self, block_output, residuals):
    """`combine` as it was BEFORE 5c64fc8a — its own inject GEMV, `/ hc`, and an explicit `2 *`."""
    hyper_input, normed, _ = residuals
    if block_output.shape[0] == 0:
        return hyper_input
    inject = self.block_inject_weight.forward(normed) / self._hc
    gate = 2.0 * torch.sigmoid(inject)
    branches = hyper_input.unflatten(-1, (self._hc, self._hs))
    return (branches + block_output.unsqueeze(-2) * gate.unsqueeze(-1)).flatten(-2)


def _hc_set_fusion(blocks: list, on: bool) -> dict:
    """Flip every HC block between the fused and the pre-fusion form. Exactly reversible, and it
    ALLOCATES NOTHING.

    THAT IS A REQUIREMENT, NOT AN OPTIMISATION, and the first version of this probe learned it the
    expensive way: it rebuilt each unfused weight as its own contiguous buffer, and at the 48-layer
    operating point the card has ~58 MB free once the KV pool, the captured graphs and a few
    generates have run. A 20 MB `torch.empty` was enough to raise OutOfMemoryError, so a gate about
    numerics failed for a reason that had nothing to do with numerics.

    It is avoidable because `post_load` already left the two checkpoint-named weights as CONTIGUOUS
    DISJOINT VIEWS of the packed buffer — `fused[:lowrank]` and `fused[lowrank:]`. Those views ARE
    the unpacked operands. So "unpack" is not a copy at all:

      * the PACK is turned off by clearing `_fused_w`, which makes `_fused_ok()` fail closed and
        sends `mix` down the two-separate-GEMV path against those same views;
      * the FOLD is undone by `fused.mul_(hc)` IN PLACE and restored by `fused.div_(hc)` — exact in
        both directions because `hc` is a power of two, so only the exponent moves;
      * `_fused_lin` is stashed and handed back, rather than reconstructed, so not even the
        `LinearReplicated` constructor's `torch.empty` runs.

    Every byte the model holds is the same byte before and after. The caller still digests the
    packed buffer either side, because "should be exact" is the claim under test.
    """
    import types

    n_touched = n_nopack = 0
    for h in blocks:
        if on:
            for attr in ("mix", "combine"):
                h.__dict__.pop(attr, None)
            h.hc_norm.__dict__.pop("_gain_vec", None)
            saved = h.__dict__.pop("_ab_saved", None)
            if saved is None:
                continue
            was_folded, fused, lin = saved
            if was_folded:
                fused.div_(h._hc)          # in place, exact: power-of-two exponent shift
            h._fused_w = fused
            h._fused_lin = lin
            h._scale_folded = was_folded
            n_touched += 1
            continue

        if h._fused_w is None:
            n_nopack += 1
            continue
        if not h._use_combine:
            # The top-level `hyper_connection_mixer` packs nothing to unpack (use_combine=False),
            # but the `/ hc` fold DOES apply to it, so it still gets the legacy `mix`.
            n_nopack += 1
        h.__dict__["_ab_saved"] = (h._scale_folded, h._fused_w, h._fused_lin)
        if h._scale_folded:
            h._fused_w.mul_(h._hc)         # in place; the .weight views see it, no copy
        h._fused_w = None                  # -> _fused_ok() fails closed -> two separate GEMVs
        h._fused_lin = None
        h._scale_folded = False
        h.__dict__["mix"] = types.MethodType(_hc_legacy_mix, h)
        if h._use_combine:
            h.__dict__["combine"] = types.MethodType(_hc_legacy_combine, h)
        # change 1: recompute `(1 + w)` every call instead of reading the memo
        h.hc_norm.__dict__["_gain_vec"] = types.MethodType(
            lambda s, capturing: s.weight + 1.0, h.hc_norm)
        n_touched += 1
    return {"blocks_touched": n_touched, "blocks_without_pack": n_nopack}


def _hc_fusion_ab(llm, args) -> dict:
    """Greedy token ids with the HC fusion ON vs reverted, in ONE boot, EAGER on both legs."""
    from minisgl.core import SamplingParams

    gr = llm.engine.graph_runner
    blocks = _hc_blocks(llm)
    packed = sum(1 for h in blocks if h._fused_w is not None)
    o: dict = {
        "hc_ab_blocks_found": len(blocks),
        "hc_ab_blocks_packed_before": packed,
    }
    print(f"\n[6e] HC fusion A/B: {len(blocks)} hyper-connection blocks, {packed} packed", flush=True)
    check_true("HC A/B found the hyper-connection blocks", len(blocks) > 0,
               "a probe that finds nothing reports 'identical' and means nothing")
    n = int(args.hc_ab_steps)
    sp = SamplingParams(temperature=0.0, max_tokens=n)
    prompts = _PARITY_PROMPTS[:2]
    # The flip itself allocates nothing, but the two generates either side of it do, and at this
    # operating point the card is down to tens of MB. Hand back whatever the allocator is merely
    # caching first.
    torch.cuda.empty_cache()
    # The fold is undone and redone IN PLACE, so the packed bytes are checked either side.
    dig_before = [_tensor_digest(h._fused_w) for h in blocks if h._fused_w is not None]
    saved_bs = gr.max_graph_bs
    flipped = False
    try:
        gr.max_graph_bs = 0  # both legs eager; the captured graph baked the fused pointers
        on1 = [list(llm.generate([p], sp)[0]["token_ids"]) for p in prompts]
        o["hc_ab_flip_census"] = _hc_set_fusion(blocks, on=False)
        flipped = True
        o["hc_ab_packed_while_off"] = sum(1 for h in blocks if h._fused_w is not None)
        off = [list(llm.generate([p], sp)[0]["token_ids"]) for p in prompts]
        _hc_set_fusion(blocks, on=True)
        flipped = False
        o["hc_ab_packed_after_restore"] = sum(1 for h in blocks if h._fused_w is not None)
        dig_after = [_tensor_digest(h._fused_w) for h in blocks if h._fused_w is not None]
        o["hc_ab_packed_bytes_restored_exactly"] = bool(dig_before == dig_after)
        on2 = [list(llm.generate([p], sp)[0]["token_ids"]) for p in prompts]
    finally:
        # The model must go back to the shipped form whatever happened here — anything measured
        # after this point would otherwise be measuring the unfused path without saying so.
        if flipped:
            _hc_set_fusion(blocks, on=True)
        gr.max_graph_bs = saved_bs
    o["hc_ab_ids_fused"] = on1
    o["hc_ab_ids_unfused"] = off
    o["hc_ab_ids_fused_again"] = on2
    o["hc_ab_tokens_per_prompt"] = n
    # The SAME-CODE FLOOR for this gate: two runs of the fused path, either side of the unfused one.
    # If they disagree, an ON-vs-OFF disagreement is not attributable to the fusion.
    o["hc_ab_same_code_floor_identical"] = bool(on1 == on2)
    o["hc_ab_ids_identical"] = bool(on1 == off)
    o["hc_ab_first_divergence"] = None
    if not o["hc_ab_ids_identical"]:
        for i, (a, b) in enumerate(zip(on1, off)):
            for j, (x, y) in enumerate(zip(a, b)):
                if x != y:
                    o["hc_ab_first_divergence"] = {"prompt": i, "step": j, "fused": x, "unfused": y}
                    break
            if o["hc_ab_first_divergence"]:
                break
    print(f"  fused==unfused: {o['hc_ab_ids_identical']}  "
          f"same-code floor (fused vs fused): {o['hc_ab_same_code_floor_identical']}  "
          f"restored packs: {o['hc_ab_packed_after_restore']}/{packed}", flush=True)
    check("HC A/B restored every pack", o["hc_ab_packed_after_restore"], packed)
    return o


def _capture_throughput_ab(llm, args) -> dict:
    """Decode tok/s with the graphs live vs with them forced off, in ONE boot.

    Same switch as the parity gate (`graph_runner.max_graph_bs = 0`), so the two legs share weights,
    KV pool, pinned arena and host tier and the only variable is the capture mechanism.

    HOW THE NUMBER IS BUILT, and why not wall time over `generate`. `generate` wall time includes the
    prefill and the tokenizer, and at these token counts the prefill is a large fraction of it — the
    published 11.85 tok/s for this operating point is a WALL figure and this repo has a standing note
    that the decode panel measures wall time and must not be quoted as a decode number. So each leg
    is run TWICE: once for `warm_tokens` (which pays the first replay, the allocator's first pass and
    any lazily-built buffer) and then once for `throughput_tokens`, and only the second is timed.
    Both legs get the identical treatment, so whatever is left in the number is in both.

    The ratio is the deliverable and it is allowed to be ~1.0. Capture removes LAUNCH overhead; if
    decode here is bound by the PCIe read of 37 layers' experts out of pinned host memory, there is
    no launch overhead left to remove and the honest result is "no win". That is reported as such.
    """
    from minisgl.core import SamplingParams

    gr = llm.engine.graph_runner
    prompt = _PARITY_PROMPTS[0]
    warm = SamplingParams(temperature=0.0, max_tokens=8)
    # `ignore_eos=True` ON THE TIMED LEG, for the same reason `_capture_ab_sampled` does it, and
    # FIXED HERE [BOOT-2026-09-05] because this function's own equal-token-count gate could not pass
    # without it. Greedy ids are NOT bit-reproducible past ~32 tokens on this engine (a per-request
    # recurrent-state slot that alternates; see `_repro_probe`), so over a 128-token run the captured
    # and eager legs drift apart and one of them reaches EOS first. Measured 2026-09-06 at 48 layers
    # TP=2: captured stopped at 68 tokens, eager ran the full 128, and `check("throughput legs
    # generated the same token count", 68, 128)` FAILED — a red run for a property of the engine that
    # has nothing to do with capture, over two legs that had by then done different amounts of work
    # and were therefore not a throughput comparison at all. Pinning the count makes the legs equal
    # work; it does not make them equal ids, and nothing here claims it does.
    run = SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=args.throughput_tokens)
    o: dict = {"throughput_tokens": int(args.throughput_tokens),
               "throughput_ignore_eos": True}
    print(f"\n[6c] throughput A/B: {args.throughput_tokens} greedy tokens, "
          f"captured vs eager (one boot)", flush=True)

    def leg(graphs_on: bool) -> tuple:
        saved = gr.max_graph_bs
        try:
            if not graphs_on:
                gr.max_graph_bs = 0
            llm.generate([prompt], warm)
            t = time.perf_counter()
            r = llm.generate([prompt], run)[0]
            dt = time.perf_counter() - t
        finally:
            gr.max_graph_bs = saved
        return len(r["token_ids"]), dt

    # INTERLEAVED AND REPEATED, and the summary statistic is the MINIMUM.
    #
    # A single captured-then-eager pair is not robust on this box: the GPU lease is waived for this
    # task, so a co-tenant can land on either card partway through and inflate whichever leg happens
    # to be running. That is not hypothetical — an earlier boot of this exact configuration timed the
    # eager leg at 503 ms/token while the captured leg in the SAME boot, and both legs of the
    # PREVIOUS boot, sat at 58-60 ms/token. Interleaving makes the two legs share whatever contention
    # exists instead of one of them absorbing all of it, and the minimum of several samples is the
    # least-contended observation of each leg rather than an average of a bimodal distribution.
    # Every sample is recorded, so the spread is visible and a bad run cannot hide inside a mean.
    reps = max(1, int(args.throughput_repeats))
    c_samples, e_samples = [], []
    n_c = n_e = 0
    for _ in range(reps):
        n_c, dt = leg(True)
        c_samples.append(dt)
        n_e, dt = leg(False)
        e_samples.append(dt)
    o["throughput_captured_samples_s"] = [round(x, 4) for x in c_samples]
    o["throughput_eager_samples_s"] = [round(x, 4) for x in e_samples]
    o["throughput_repeats"] = reps
    t_c, t_e = min(c_samples), min(e_samples)
    o["throughput_captured_spread_s"] = round(max(c_samples) - min(c_samples), 4)
    o["throughput_eager_spread_s"] = round(max(e_samples) - min(e_samples), 4)

    tps_c = n_c / t_c if t_c > 0 else 0.0
    tps_e = n_e / t_e if t_e > 0 else 0.0
    o["throughput_captured_tok_per_s"] = round(tps_c, 3)
    o["throughput_eager_tok_per_s"] = round(tps_e, 3)
    o["throughput_captured_tokens"] = n_c
    o["throughput_eager_tokens"] = n_e
    o["throughput_captured_seconds"] = round(t_c, 3)
    o["throughput_eager_seconds"] = round(t_e, 3)
    o["throughput_capture_speedup"] = round(tps_c / tps_e, 4) if tps_e > 0 else None
    # Per-token, which is the units launch overhead is actually denominated in: capture removes a
    # fixed number of launches per step, so a real win is a constant ms/token saving, not a ratio.
    o["throughput_ms_per_token_captured"] = round(1000 * t_c / max(n_c, 1), 3)
    o["throughput_ms_per_token_eager"] = round(1000 * t_e / max(n_e, 1), 3)
    o["throughput_ms_per_token_saved"] = round(
        o["throughput_ms_per_token_eager"] - o["throughput_ms_per_token_captured"], 3)
    print(f"  captured {tps_c:.3f} tok/s ({o['throughput_ms_per_token_captured']} ms/tok, "
          f"{n_c} tok in {t_c:.2f}s)", flush=True)
    print(f"  eager    {tps_e:.3f} tok/s ({o['throughput_ms_per_token_eager']} ms/tok, "
          f"{n_e} tok in {t_e:.2f}s)", flush=True)
    print(f"  -> capture x{o['throughput_capture_speedup']}  "
          f"({o['throughput_ms_per_token_saved']} ms/token saved)", flush=True)
    # NOT a gate on the speedup — "no win" is a legitimate result here and must not fail the run.
    # What IS a gate: both legs must have produced the tokens asked for, or the ratio is comparing
    # a truncated run to a full one.
    check("throughput legs generated the same token count", n_c, n_e)
    return o


def _median(xs: list) -> float:
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def _decode_steps(log) -> list:
    """The decode entries of a STEP_LOG slice, as (phase, seconds), prefill dropped.

    DROPPING STEP 0 is not a statistical nicety here. On this operating point step 0 is a PREFILL:
    a different batch shape, a different attention kernel, and a pass that reads the whole prompt
    through the same host-resident experts a decode step touches. Folding it into a per-token average
    (which `len(tokens) / wall(generate)` does) taxes every token with it, and the tax is not equal
    across the two legs — the prefill is never a graph replay, so it lands identically in both and
    dilutes whatever difference the decode steps actually have. The phase tag makes the drop exact
    rather than positional.
    """
    return [(ph, dt) for (ph, _bs, dt) in log if ph != "prefill"]


def _capture_ab_sampled(llm, args, model_dir: str) -> dict:
    """THE DELIVERABLE A/B: graphs ON vs graphs OFF, checkpoint sampler, N repeats, one boot.

    Differences from the greedy `_capture_throughput_ab` above, all deliberate:

    * THE CHECKPOINT'S OWN SAMPLER (temperature/top_k/top_p out of `generation_config.json`), because
      this repo's standing rule is that temperature 0 is not a serving configuration and fakes
      degeneration that mimics a quant bug. The sampler is a real per-step cost (a top-k/top-p over a
      248,320-token vocabulary) that greedy does not pay, and it is host-launched work of exactly the
      kind capture is supposed to help with — measuring the ratio without it measures a lane nobody
      serves.
    * `ignore_eos=True` ON THE TIMED LEGS ONLY. Sampled generation stops where it stops, and two legs
      that emitted 61 and 160 tokens are not a throughput comparison — they are two different amounts
      of work with two different amounts of prefill amortised into them. Fixing the token count makes
      the legs identical in work. The QUALITY leg below runs the same sampler WITHOUT it, so the
      stopping behaviour anyone serves is still what coherence gets judged on.
    * PROVENANCE IS GATED, not asserted in prose. Three independent counters are diffed per leg:
      `graph_runner.replays`, `engine.eager_decode_forwards`, and the per-step phase tags in
      `STEP_LOG`. A leg passes only if ALL of its decode steps took the path the leg claims. Without
      this a "capture is a wash" result is indistinguishable from a run where the graphs were never
      replayed at all — which is the classic green A/B of new-vs-itself.

    Reported as a median over repeats with the full sample list kept, never a single pair.
    """
    import json as _json

    from minisgl.core import SamplingParams
    from minisgl.scheduler.scheduler import STEP_LOG
    from minisgl.weights.moe_interpose import RESOLVE_COUNTS

    gr = llm.engine.graph_runner
    gcfg_path = os.path.join(model_dir, "generation_config.json")
    gcfg = _json.load(open(gcfg_path)) if os.path.exists(gcfg_path) else {}
    temperature = float(gcfg.get("temperature", 1.0))
    top_k = int(gcfg.get("top_k", 20))
    top_p = float(gcfg.get("top_p", 0.95))
    eos = gcfg.get("eos_token_id")

    prompt = args.ab_prompt
    try:
        tok = llm.engine.tokenizer if hasattr(llm.engine, "tokenizer") else None
        if tok is None:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(model_dir)
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": args.ab_prompt}],
            tokenize=False, add_generation_prompt=True,
        )
    except Exception as e:  # pragma: no cover - reported, never silently skipped
        print(f"  (chat template unavailable: {e!r}; using the raw prompt)", flush=True)

    n_tok = int(args.ab_tokens)
    reps = max(1, int(args.ab_repeats))
    timed = SamplingParams(temperature=temperature, top_k=top_k, top_p=top_p,
                           ignore_eos=True, max_tokens=n_tok)
    warm = SamplingParams(temperature=temperature, top_k=top_k, top_p=top_p,
                          ignore_eos=True, max_tokens=8)
    o: dict = {
        "ab_sampler": {"temperature": temperature, "top_k": top_k, "top_p": top_p,
                       "eos_token_id": eos, "ignore_eos_on_timed_legs": True},
        "ab_prompt": args.ab_prompt,
        "ab_tokens": n_tok,
        "ab_repeats": reps,
    }
    print(f"\n[6d] SAMPLED capture A/B: {n_tok} tok x {reps} reps, "
          f"temp={temperature} top_k={top_k} top_p={top_p}, graphs ON vs OFF (one boot)", flush=True)

    def leg(graphs_on: bool, sp, warm_first: bool) -> dict:
        saved = gr.max_graph_bs
        try:
            if not graphs_on:
                gr.max_graph_bs = 0
            if warm_first:
                llm.generate([prompt], warm)
            # Snapshot AFTER the warmup so the measured window holds only the timed generate.
            STEP_LOG.clear()
            r0 = gr.replays
            e0 = llm.engine.eager_decode_forwards
            rc0 = dict(RESOLVE_COUNTS)
            t = time.perf_counter()
            r = llm.generate([prompt], sp)[0]
            wall = time.perf_counter() - t
            steps = list(STEP_LOG)
        finally:
            gr.max_graph_bs = saved
        dec = _decode_steps(steps)
        phases = sorted({ph for ph, _dt in dec})
        # Step 0 dropped by phase (the prefill), and the FIRST DECODE step dropped as well: it pays
        # the first replay of this generate, the allocator's first pass over fresh KV pages and any
        # buffer built lazily on the first step. Both legs are treated identically.
        steady = [dt for _ph, dt in dec[1:]]
        return {
            "graphs_on": graphs_on,
            "tokens": len(r["token_ids"]),
            "wall_s": wall,
            "wall_tok_per_s": len(r["token_ids"]) / wall if wall > 0 else 0.0,
            "n_forward_steps": len(steps),
            "n_prefill_steps": sum(1 for ph, _b, _d in steps if ph == "prefill"),
            "prefill_s": sum(d for ph, _b, d in steps if ph == "prefill"),
            # WALL MINUS THE FORWARDS. Everything `generate` spends that is not inside
            # `Scheduler._forward`: batch assembly, `_stage_ple`'s host-side n-gram gather, the
            # sampler's D2H, detokenization and the request/response plumbing. Capture cannot touch
            # ANY of it — a graph replaces the forward, not the loop around it — so this term is the
            # ceiling on how much of a wall-clock tok/s the capture ratio can possibly move. Kept
            # per-leg because a leg where it dominates is a leg whose wall speedup will read ~1.00 no
            # matter how much faster the forward got, and that distinction is the whole analysis.
            "residual_s": wall - sum(d for _p, _b, d in steps),
            "n_decode_steps": len(dec),
            "decode_phases": phases,
            "steady_median_ms": 1000 * _median(steady) if steady else 0.0,
            "steady_min_ms": 1000 * min(steady) if steady else 0.0,
            "steady_max_ms": 1000 * max(steady) if steady else 0.0,
            "steady_n": len(steady),
            "replays_delta": gr.replays - r0,
            "eager_decode_delta": llm.engine.eager_decode_forwards - e0,
            "resolve_delta": {k: RESOLVE_COUNTS.get(k, 0) - rc0.get(k, 0)
                              for k in set(RESOLVE_COUNTS) | set(rc0)},
            "text": r["text"],
        }

    # One untimed warm leg per mode before anything is recorded, then INTERLEAVED reps. Interleaving
    # is not cosmetic: the GPU lease is waived for this task, so a co-tenant landing on either card
    # part-way through would otherwise be absorbed entirely by whichever leg was running.
    leg(True, warm, True)
    leg(False, warm, True)
    cap_legs, eag_legs = [], []
    for i in range(reps):
        c = leg(True, timed, False)
        e = leg(False, timed, False)
        cap_legs.append(c)
        eag_legs.append(e)
        print(f"  rep {i+1}/{reps}: captured {c['wall_tok_per_s']:.2f} tok/s "
              f"(steady {c['steady_median_ms']:.1f} ms/step) | eager "
              f"{e['wall_tok_per_s']:.2f} tok/s (steady {e['steady_median_ms']:.1f} ms/step)",
              flush=True)

    def summarize(legs: list, tag: str) -> dict:
        wt = [x["wall_tok_per_s"] for x in legs]
        st = [x["steady_median_ms"] for x in legs]
        return {
            f"{tag}_wall_tok_per_s_samples": [round(x, 3) for x in wt],
            f"{tag}_wall_tok_per_s_median": round(_median(wt), 3),
            f"{tag}_wall_tok_per_s_min": round(min(wt), 3),
            f"{tag}_wall_tok_per_s_max": round(max(wt), 3),
            f"{tag}_steady_ms_per_step_samples": [round(x, 3) for x in st],
            f"{tag}_steady_ms_per_step_median": round(_median(st), 3),
            f"{tag}_steady_ms_per_step_min": round(min(st), 3),
            f"{tag}_steady_ms_per_step_max": round(max(st), 3),
            f"{tag}_steady_tok_per_s_median": round(1000.0 / _median(st), 3) if _median(st) else 0.0,
            f"{tag}_prefill_s": round(_median([x["prefill_s"] for x in legs]), 4),
            f"{tag}_residual_s": round(_median([x["residual_s"] for x in legs]), 4),
            f"{tag}_residual_ms_per_token": round(
                1000 * _median([x["residual_s"] / max(x["tokens"], 1) for x in legs]), 3),
            f"{tag}_forward_share_of_wall": round(_median(
                [(x["wall_s"] - x["residual_s"]) / x["wall_s"] if x["wall_s"] else 0.0
                 for x in legs]), 4),
            f"{tag}_tokens": legs[0]["tokens"],
            f"{tag}_decode_steps": legs[0]["n_decode_steps"],
            f"{tag}_steady_n": legs[0]["steady_n"],
            # PER-REP counts, not just rep 1's. On 2026-09-04 the L48 TP=2 run failed the
            # equal-token-count gate with captured=120 / eager=119 under `ignore_eos`, where both
            # legs had recorded the SAME 8 prefill + 119 decode forwards — so the asymmetry is in the
            # OUTPUT accounting, not in the work timed, and the per-decode-step ratio is unaffected.
            # It could not be localised because only rep 1's counts were kept: "one leg is short by
            # one on every rep" and "one leg was short by one on ONE rep" are different bugs and the
            # summary could not tell them apart. Both series are kept now so the next run can.
            f"{tag}_tokens_per_rep": [x["tokens"] for x in legs],
            f"{tag}_decode_steps_per_rep": [x["n_decode_steps"] for x in legs],
            f"{tag}_prefill_steps_per_rep": [x["n_prefill_steps"] for x in legs],
        }

    o.update(summarize(cap_legs, "ab_captured"))
    o.update(summarize(eag_legs, "ab_eager"))
    mc, me = o["ab_captured_wall_tok_per_s_median"], o["ab_eager_wall_tok_per_s_median"]
    sc, se = o["ab_captured_steady_ms_per_step_median"], o["ab_eager_steady_ms_per_step_median"]
    o["ab_speedup_wall"] = round(mc / me, 4) if me > 0 else None
    o["ab_speedup_steady"] = round(se / sc, 4) if sc > 0 else None
    o["ab_ms_per_step_saved"] = round(se - sc, 3)
    # THE `steady_*` FIELDS ARE HOST WALL OF `_forward`, NOT DEVICE TIME. DO NOT QUOTE THE RATIO.
    #
    # Measured on the L48 TP=2 run of 2026-09-04: captured 0.51 ms/step vs eager 25.58 ms/step, a
    # ratio of 50.7 — while the WALL clock moved 2.5%. The 50.7 is an artifact and it is physically
    # impossible as a decode time: this step streams 568.3 MB/rank of host-resident experts across
    # PCIe, which is 39.25 ms at card 1's measured 14.48 GB/s, so no forward here completes in 0.5 ms.
    # `g.replay()` is ASYNCHRONOUS — the captured leg's `_forward` returns once the graph is enqueued
    # and the device work is absorbed at the next synchronisation point, which lands OUTSIDE the
    # timed region and inside `residual_s`. The eager leg's 48 layers of Python launches back-pressure
    # the host, so its `_forward` wall tracks the device far more closely, and comparing the two is
    # comparing an enqueue against a partial execution.
    #
    # The conservation check below is the honest statement and is emitted next to the ratio so the
    # two cannot be read apart: ms/step that LEFT `_forward` minus ms/token that reappeared in the
    # residual equals the net the wall clock actually saw (25.08 - 22.73 = 2.35, vs a 2.02 ms/step
    # wall saving). This repo already carries the rule that the decode panel measures WALL time and
    # must not be quoted as a decode number; this is the same trap one level down.
    o["ab_steady_fields_are_host_forward_wall_not_device_time"] = True
    o["ab_residual_ms_per_token_delta"] = round(
        o["ab_captured_residual_ms_per_token"] - o["ab_eager_residual_ms_per_token"], 3)
    o["ab_net_ms_per_step_saved_after_residual"] = round(
        o["ab_ms_per_step_saved"] - o["ab_residual_ms_per_token_delta"], 3)
    # The number to quote: wall seconds per DECODE STEP, which is immune to a leg emitting one more
    # or fewer token than the other (both legs run the same recorded number of decode forwards).
    nd_c, nd_e = o["ab_captured_decode_steps"], o["ab_eager_decode_steps"]
    wall_c = o["ab_captured_tokens"] / mc if mc > 0 else 0.0
    wall_e = o["ab_eager_tokens"] / me if me > 0 else 0.0
    o["ab_wall_ms_per_decode_step_captured"] = round(1000 * wall_c / nd_c, 3) if nd_c else None
    o["ab_wall_ms_per_decode_step_eager"] = round(1000 * wall_e / nd_e, 3) if nd_e else None
    o["ab_speedup_per_decode_step"] = (
        round((wall_e / nd_e) / (wall_c / nd_c), 4) if nd_c and nd_e and wall_c > 0 else None)

    # -- PROVENANCE, gated ---------------------------------------------------------------------
    prov = {
        "captured_replays_total": sum(x["replays_delta"] for x in cap_legs),
        "captured_eager_decode_total": sum(x["eager_decode_delta"] for x in cap_legs),
        "eager_replays_total": sum(x["replays_delta"] for x in eag_legs),
        "eager_eager_decode_total": sum(x["eager_decode_delta"] for x in eag_legs),
        "captured_decode_phases": sorted({p for x in cap_legs for p in x["decode_phases"]}),
        "eager_decode_phases": sorted({p for x in eag_legs for p in x["decode_phases"]}),
        "captured_resolve_delta": cap_legs[0]["resolve_delta"],
        "eager_resolve_delta": eag_legs[0]["resolve_delta"],
    }
    o["ab_provenance"] = prov
    print(f"  provenance: captured replays={prov['captured_replays_total']} "
          f"eager_forwards={prov['captured_eager_decode_total']} phases={prov['captured_decode_phases']}",
          flush=True)
    print(f"              eager    replays={prov['eager_replays_total']} "
          f"eager_forwards={prov['eager_eager_decode_total']} phases={prov['eager_decode_phases']}",
          flush=True)
    print(f"              MoE resolve/leg captured={prov['captured_resolve_delta']} "
          f"eager={prov['eager_resolve_delta']}", flush=True)
    check_true("A/B captured leg REPLAYED graphs", prov["captured_replays_total"] > 0)
    check("A/B captured leg ran NO eager decode forward", prov["captured_eager_decode_total"], 0)
    check("A/B captured leg's decode steps are all graph replays",
          prov["captured_decode_phases"], ["decode_graph"])
    check("A/B eager leg replayed NO graph", prov["eager_replays_total"], 0)
    check_true("A/B eager leg ran eager decode forwards", prov["eager_eager_decode_total"] > 0)
    check("A/B eager leg's decode steps are all eager forwards",
          prov["eager_decode_phases"], ["decode_eager"])
    # THE THIRD, INDEPENDENT PROVENANCE CHANNEL, stated as arithmetic rather than as a sign test.
    #
    # `resolve()` is host Python inside `MoELayer.forward`, so it runs once per HOST-placed MoE layer
    # per EAGER forward and never during a graph replay. Graphs here are DECODE-only, so a captured
    # leg still re-enters the seam on its PREFILL chunks — the naive "captured leg must show zero"
    # test is wrong and fails on a perfectly good run (observed: 24 host resolves from 8 prefill
    # chunks at L4). The exact invariant is that both legs resolve the same number of host layers per
    # eager forward, and that the captured leg's eager forwards are its prefill chunks ALONE:
    #
    #     captured: host_resolves == H * n_prefill                (decode contributed nothing)
    #     eager:    host_resolves == H * (n_prefill + n_decode)    (every step went through Python)
    #
    # with the SAME H recovered independently from each leg. H is the model's host-placed MoE layer
    # count, so this simultaneously proves the offload arm was live in BOTH legs (H > 0 either side —
    # a leg reading device weights would show H == 0) and that the captured leg's decode really
    # bypassed Python. A leg that captured but never replayed fails it: its decode steps would
    # contribute H each and the first equality would break.
    key_host = "weight_offload.moe_resolve[host]"
    cap_host = cap_legs[0]["resolve_delta"].get(key_host, 0)
    eag_host = eag_legs[0]["resolve_delta"].get(key_host, 0)
    cap_pf = cap_legs[0]["n_prefill_steps"]
    eag_fwd = eag_legs[0]["n_prefill_steps"] + eag_legs[0]["n_decode_steps"]
    h_cap = cap_host / cap_pf if cap_pf else 0
    h_eag = eag_host / eag_fwd if eag_fwd else 0
    prov["host_layers_per_eager_forward_captured_leg"] = h_cap
    prov["host_layers_per_eager_forward_eager_leg"] = h_eag
    prov["captured_leg_prefill_forwards"] = cap_pf
    prov["eager_leg_total_forwards"] = eag_fwd
    check_true("A/B both legs read HOST-placed experts (offload arm live in each)",
               h_cap > 0 and h_eag > 0, f"H(captured)={h_cap} H(eager)={h_eag}")
    check("A/B both legs resolve the same host-layer count per eager forward", h_cap, h_eag)
    check("A/B captured leg's ONLY seam re-entries are its prefill chunks (decode was replayed)",
          cap_host, int(h_eag * cap_pf))
    check("A/B eager leg re-entered the seam on EVERY forward", eag_host, int(h_eag * eag_fwd))
    check("A/B legs generated the same token count",
          o["ab_captured_tokens"], o["ab_eager_tokens"])

    print(f"  captured  wall {mc:.3f} tok/s   steady {sc:.2f} ms/step "
          f"({o['ab_captured_steady_tok_per_s_median']:.2f} tok/s)", flush=True)
    print(f"  eager     wall {me:.3f} tok/s   steady {se:.2f} ms/step "
          f"({o['ab_eager_steady_tok_per_s_median']:.2f} tok/s)", flush=True)
    print(f"  -> wall x{o['ab_speedup_wall']}  steady x{o['ab_speedup_steady']} "
          f"({o['ab_ms_per_step_saved']} ms/step saved)", flush=True)

    # -- QUALITY on BOTH legs, real stopping behaviour ------------------------------------------
    qual = SamplingParams(temperature=temperature, top_k=top_k, top_p=top_p,
                          max_tokens=int(args.ab_quality_tokens))
    if args.ab_quality_tokens > 0:
        for on, key in ((True, "ab_quality_captured"), (False, "ab_quality_eager")):
            saved = gr.max_graph_bs
            try:
                if not on:
                    gr.max_graph_bs = 0
                q = llm.generate([prompt], qual)[0]
            finally:
                gr.max_graph_bs = saved
            o[key] = q["text"]
            o[f"{key}_tokens"] = len(q["token_ids"])
            print(f"  [{key}] {len(q['token_ids'])} tok: {q['text']!r}", flush=True)
        check_true("captured leg produced quality tokens", o["ab_quality_captured_tokens"] > 0)
    return o


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--experts", type=int, default=512)
    # TENSOR PARALLEL. `tools/serve.sh` DEFAULTS to TP=2 and every other model in this repo serves
    # there, but every qwen4_exp run to date has been TP=1 on card 0 because this harness had no way
    # to ask for anything else — and the capacity analysis inherited that, concluding the offload
    # plan does not fit when the number it was judging against was a single card's.
    #
    # The MECHANISM is the one this repo already has, not a second one: `LLM` is TP=1 unless the
    # caller supplies its own `tp_info`, and a TP>1 offline run is ONE PROCESS PER RANK, spawned by
    # the caller exactly as `server/launch.py` and `tools/kv_fp8_calibrate.py` do. Every rank drives
    # the IDENTICAL request stream from its own `pending_requests`, so the ranks stay in lockstep
    # without the ZMQ rank0->rank1 fan-out the served path uses. A fork would inherit a HIP context,
    # so the start method is `spawn`.
    ap.add_argument("--tp", type=int, default=1, help="tensor-parallel size (1 or 2 on this box)")
    # The lever under test. Below the subset's expert total, so the greedy fill puts some layers on
    # the card and the rest in the arena — which is the only configuration in which ANY of the code
    # this file exercises runs at all.
    ap.add_argument("--device-gb", type=float, default=2.0)
    ap.add_argument("--host-gb", type=float, default=12.0)
    ap.add_argument("--max-tokens", type=int, default=6)
    # The THIRD tier. >0 hands the LAST N MoE layers to `weights/stream_tier.py` and removes them
    # from the placement plan — the only configuration in which 48 layers boots at TP=1 on this box.
    ap.add_argument("--stream-layers", type=int, default=0)
    # THE THIRD TIER. N deepest offloadable MoE layers computed by host AVX-512 cores instead of
    # streamed over PCIe. Costs physical cores and int8 activations; see weights/cpu_native.py.
    ap.add_argument("--cpu-layers", type=int, default=0)
    ap.add_argument("--cpu-threads", type=int, default=2, help="PHYSICAL cores per rank")
    # QUALITY leg. A full-depth model SHOULD produce coherent text, and a quality verdict from a
    # bare temperature multinomial over this checkpoint's 248,320-token vocabulary measures the
    # harness, not the model — it manufactures the "token noise" degeneration signature out of the
    # tail. So the sampler is the checkpoint's own: temperature 1.0, top_k 20, top_p 0.95. NEVER
    # greedy: CLAUDE.md's rule is that temperature 0 fakes degeneration that mimics a quant bug.
    ap.add_argument("--quality-prompt", default="")
    ap.add_argument("--quality-tokens", type=int, default=0)
    # ---- CUDA GRAPH CAPTURE (added 2026-09-04) ------------------------------------------------
    # `>0` captures one decode graph per bucketed batch size and replays it every decode step. It is
    # a MERGE REQUIREMENT in this repo ("eager-only is never done"), not a tuning knob, and it is
    # what this flag exists to discharge on the qwen4_exp path. Default 0 so every measurement this
    # file has already produced stays reproducible with no flag.
    ap.add_argument("--cuda-graph-max-bs", type=int, default=0)
    # ATTENTION BACKEND. Every qwen4_exp run before 2026-09-04 booted "rdna4", whose
    # `init_capture_graph` / `prepare_for_capture` / `prepare_for_replay` all RAISE
    # NotImplementedError ("rdna4 cudagraph capture lands in Phase 4; run with --cuda-graph-max-bs
    # 0"). The capture-capable implementation is its SUBCLASS `HIPAttnBackend` ("hip"), which is
    # what `docker-compose.yml` defaults to and what every production serve in this repo runs.
    # This is not a kernel change: with MINISGL_ATTN_HIP=1 `RDNA4Backend.forward` already dispatches
    # decode to the same `attn_decode.flash_decode_paged` op `HIPAttnBackend._forward_decode` calls,
    # and cold prefill to the same `attn_hip.flash_prefill`. The default moves to "hip" because a
    # bring-up harness that cannot capture cannot discharge the requirement; pass
    # `--attention-backend rdna4` to reproduce a pre-2026-09-04 run exactly.
    ap.add_argument("--attention-backend", default="hip", choices=["hip", "rdna4"])
    # EAGER-vs-REPLAY PARITY, on ONE boot. `>0` generates the same greedy prompt twice — once with
    # the decode graphs live, once with `graph_runner.max_graph_bs` forced to 0 so `pad_batch` and
    # `can_use_cuda_graph` both fall through to the eager path — and compares the token ids. Same
    # process, same weights, same KV pool, same arena, same host tier: the ONLY difference is the
    # capture mechanism, which is what makes this an A/B and not two runs of two binaries.
    ap.add_argument("--parity-steps", type=int, default=0)
    # PARITY BATCH WIDTHS. bs=1 is the cheap signal but it is NOT the whole gate: the captured graph
    # is per-bucket, so every captured width is a separate graph with its own baked page-table row
    # stride and its own MoE token count, and a bug that only bites at bs>1 (a row-major/col-major
    # page-table read, a per-row PLE slot, a MoE scatter over the wrong token count) is invisible at
    # bs=1. Comma-separated; each width runs its OWN captured-then-eager pair inside this one boot.
    ap.add_argument("--parity-batches", default="1")
    # WHAT CAPTURE ACTUALLY BUYS, as a ONE-BOOT A/B. `--parity-steps` is 12 tokens off one prefill,
    # which is far too short to be a throughput read (the prefill and the first replay's warmup are
    # most of it). This leg is the same captured-vs-eager switch held over a long enough decode run
    # for the launch overhead capture removes to be the thing being measured. Reported as tok/s on
    # BOTH legs plus the ratio; it is explicitly allowed to come out at ~1.0, because 37 of 48 layers
    # read their experts over PCIe and a PCIe-bound decode has no launch overhead to remove.
    ap.add_argument("--throughput-tokens", type=int, default=0)
    # N identical generates per mode, to separate the ENGINE's reproducibility from capture's.
    ap.add_argument("--repro-probe", type=int, default=0)
    # HC FUSION ON/OFF (`_hc_fusion_ab`). The merge left no switch — `post_load` rewrites the
    # weights — so this reconstructs the pre-fusion tensors and rebinds the pre-fusion mix/combine
    # bodies inside the live boot. `>0` is the number of greedy tokens per prompt.
    ap.add_argument("--hc-ab-steps", type=int, default=0)
    # THE WEIGHT-IDENTITY GATE for a boot-PERFORMANCE change. Hashes every byte of every reachable
    # weight tensor, host tier included and read through the device-side mapping the kernels use.
    # Off by default because it costs ~30 s and 34 GiB of reads; on for any A/B whose claim is
    # "same model, loaded faster", where the accounting identities alone are not evidence.
    ap.add_argument("--weight-digest", action="store_true")
    ap.add_argument("--throughput-repeats", type=int, default=3)
    # THE DELIVERABLE A/B (`_capture_ab_sampled`). Distinct from --throughput-tokens, which is the
    # GREEDY sanity version of the same switch: this one runs the checkpoint's own sampler, fixes the
    # token count with ignore_eos so the legs do equal work, drops the prefill AND the first decode
    # step, and GATES on three independent provenance counters. `--ab-tokens 0` leaves it off.
    ap.add_argument("--ab-tokens", type=int, default=0)
    ap.add_argument("--ab-repeats", type=int, default=5)
    # The reference operating point's prompt VERBATIM, trailing period included, so the A/B's wall
    # tok/s is comparable to the published 11.85 rather than merely similar to it.
    ap.add_argument("--ab-prompt", default="Explain in three sentences why the sky is blue.")
    ap.add_argument("--ab-quality-tokens", type=int, default=0)
    ap.add_argument("--max-running-req", type=int, default=2)
    ap.add_argument("--memory-ratio", type=float, default=0.90)
    ap.add_argument("--max-extend-tokens", type=int, default=8)
    ap.add_argument("--json", default="")
    return ap


@torch.inference_mode()
def rank_main(rank: int, tp: int, args, model_dir: str) -> dict:
    """ONE TP rank, start to finish. At `tp == 1` this is called inline and the file behaves exactly
    as it did before the flag existed; at `tp > 1` it is the `mp.Process` target."""
    global _failures

    from minisgl._hip_engage import _seen as engaged_ledger
    from minisgl.core import SamplingParams
    from minisgl.distributed import DistributedInfo
    from minisgl.llm import LLM
    from minisgl.weights.moe_interpose import discover_moe_layers
    from minisgl.weights.stacks import StackKind

    # NOTE: nothing here may touch the device before `LLM(...)`. `Engine.__init__` asserts
    # `not torch.cuda.is_initialized()` — it measures free VRAM as its baseline, so a HIP context
    # created by a harness convenience call (`get_device_name`, a probe allocation) shifts every
    # byte the KV sizing bills. The card name is recorded AFTER the boot for that reason; a timing
    # rule that only shows up as a bare `AssertionError` is worth a comment.
    out: dict = {
        "layers": args.layers,
        "device_gb": args.device_gb,
        "host_gb": args.host_gb,
        "tp": tp,
        "rank": rank,
    }

    # `--device-gb 0` is the CONTROL leg: no flag at all, i.e. byte-for-byte the serve this repo
    # ships today. It is not a disabled variant of the offload leg — it is the baseline the offload
    # leg's KV pool has to be compared against, and this repo's rule is that an A/B baseline must be
    # the OLD CODE PATH rather than an emulation of it. On the control leg every offload assertion
    # below is INVERTED: the session must be disabled and the ledger must NOT carry the host arm.
    offload = args.device_gb > 0
    print(f"\n[1] boot the engine {'WITH' if offload else 'WITHOUT'} offload", flush=True)
    kw = (
        dict(
            weight_offload_device_gb=args.device_gb,
            weight_offload_gb=args.host_gb,
            weight_offload_stream_layers=args.stream_layers,
            weight_offload_cpu_layers=args.cpu_layers,
        )
        if offload
        else {}
    )
    out["stream_layers_requested"] = int(args.stream_layers)
    out["cpu_layers_requested"] = int(args.cpu_layers)
    out["cpu_threads_per_rank"] = int(args.cpu_threads)
    if args.cpu_layers:
        # Set BEFORE the boot: `StageARuntime._make_cpu_worker` reads it when the sink opens the
        # pool, which happens inside `LLM(...)`.
        os.environ["MINISGL_CPU_MOE_THREADS"] = str(args.cpu_threads)
    t0 = time.perf_counter()
    llm = LLM(
        model_path=model_dir,
        dtype=torch.bfloat16,
        # THE TP SEAM. `LLM.__init__` does `kwargs.setdefault("tp_info", DistributedInfo(0, 1))`, so
        # passing it here is the whole mechanism — the engine derives its card from it
        # (`EngineConfig.device_index`), builds the gloo/nccl groups from it, and every layer shards
        # off `get_tp_info()`. No second path.
        tp_info=DistributedInfo(rank, tp),
        # 0 = eager decode (the pre-2026-09-04 default; the post-capture gate still fires, it just
        # has no graph to gate). >0 captures one decode graph per bucketed batch size.
        cuda_graph_max_bs=args.cuda_graph_max_bs,
        page_size=16,
        memory_ratio=args.memory_ratio,
        attention_backend=args.attention_backend,
        max_running_req=args.max_running_req,
        max_extend_tokens=args.max_extend_tokens,
        **kw,
    )
    out["boot_seconds"] = round(time.perf_counter() - t0, 1)
    # The rank's OWN card, not `cuda:0`. Card 1's root port is Gen4 x8 (14.48 GB/s vs card 0's
    # 28.93), so which physical card a rank landed on is part of every timing this file reports —
    # recording `get_device_name(0)` on both ranks would silently attribute rank 1's numbers to
    # card 0.
    dev = llm.engine.device
    out["card"] = torch.cuda.get_device_name(dev)
    out["device_index"] = int(getattr(dev, "index", 0) or 0)
    out["offload"] = offload
    woff = llm.engine._woff

    # ---------------------------------------------------------------- graph capture, both legs
    # Recorded FIRST and unconditionally, because "capture was requested" and "graphs exist" are
    # different statements: `_determine_cuda_graph_bs` can return an EMPTY list when free memory is
    # short, and `_capture_graphs` then logs "CUDA graph is disabled" and returns — a serve that
    # looks captured in its launch line and is eager in fact.
    gr = llm.engine.graph_runner
    out["attention_backend"] = args.attention_backend
    out["cuda_graph_max_bs_requested"] = int(args.cuda_graph_max_bs)
    out["cuda_graph_bs_captured"] = sorted(getattr(gr, "graph_map", {}).keys())
    out["graph_capture_engaged"] = bool(out["cuda_graph_bs_captured"])
    out["ple_graph_capturer"] = type(getattr(gr, "ple_capture", None)).__name__
    if args.cuda_graph_max_bs > 0:
        check_true("decode graphs were actually captured", out["graph_capture_engaged"],
                   f"requested max_bs={args.cuda_graph_max_bs}, captured {out['cuda_graph_bs_captured']}")
        check_true("the PLE capturer is wired into the GraphRunner",
                   out["ple_graph_capturer"] == "PLEGraphCapture",
                   "without it the capture-time warmup forward raises in Qwen4ExpPLE.forward")
    ple_rt = getattr(llm.engine, "ple_runtime", None)
    if ple_rt is not None:
        out.update({f"boot_{k}": v for k, v in ple_rt.counters().items()})
        # One discarded stage per captured batch size, and NOT ONE MORE: the capturer must discard
        # (never commit) its synthetic batch, or slot 0's n-gram history advances for a forward that
        # never happened and the prepare/commit ledger is permanently off by one.
        check("PLE stages discarded == graphs captured",
              ple_rt.discards, len(out["cuda_graph_bs_captured"]))
        check("no PLE commit ran with nothing staged (boot)", ple_rt.commit_noops, 0)

    # ---------------------------------------------------------------- TP structure, both legs
    # These run BEFORE the offload branch because they are properties of the SHARDED MODEL, not of
    # the arena: at TP=2 without offload they are the whole point of the run, and at TP=2 with it
    # they are the precondition (a wrongly-replicated GDN head would make every offload byte count
    # wrong in the same direction on both ranks, which no per-rank assertion can see).
    out.update(_tp_structure(llm, tp, rank, model_dir))

    if not offload:
        # The control leg's whole job is to be the number the offload leg is compared against, and
        # to prove the path is INERT without the flag — a Stage-A window that did something on a
        # serve nobody configured would be a regression for every other model in the repo.
        check("control: Stage-A session disabled", bool(woff.enabled), False)
        check("control: model_memory correction is zero", woff.model_memory_correction(), (0, 0))
        res = llm.generate(
            ["The capital of France is"],
            SamplingParams(temperature=0.0, max_tokens=args.max_tokens),
        )
        marks = sorted(n for n in engaged_ledger if n.startswith("weight_offload."))
        check("control: no weight_offload.* engaged", marks, [])
        out.update(
            kv_pages=int(llm.engine.num_pages),
            engaged=marks,
            session_enabled=False,
            tokens=[len(r["token_ids"]) for r in res],
            # The greedy ids themselves, not just their count. This is what a TP=1 vs TP=2 parity
            # claim is made of; a length match is satisfied by two different completions.
            token_ids=[list(r["token_ids"]) for r in res],
            text=[r["text"] for r in res],
            failures=_failures,
        )
        print(f"\n[rank {rank}] {'PASS' if not _failures else 'FAIL'}: {_failures} failure(s)",
              flush=True)
        print(json.dumps(out, indent=2), flush=True)
        return out

    # [1] THE SESSION IS ENABLED. First, and unconditionally: a disabled session accepts the whole
    # call sequence and does nothing, so every assertion below it would pass over a serve with no
    # offload at all. That is precisely the shape of green this file exists to refuse.
    check_true("Stage-A session ENABLED (plan is non-empty)", woff.enabled,
               f"host={_gib(woff.accounting.host_bytes)} device={_gib(woff.accounting.device_bytes)}")
    check("Stage-A phase", woff.phase.name, "SEALED")
    out["session_enabled"] = bool(woff.enabled)
    out["plan_host_bytes"] = int(woff.accounting.host_bytes)
    out["plan_device_bytes"] = int(woff.accounting.device_bytes)
    out["copied_bytes"] = int(woff.accounting.copied_bytes)

    # [1b] THE LOAD WAS CHUNKED. Without this the file cannot tell a boot that survived because of
    # Stage B from one that survived because the subset happened to fit one-shot — and at 4 layers
    # both are true, so the assertion has to be on the MECHANISM, not on the outcome.
    print("\n[1b] the load path", flush=True)
    led = getattr(llm.engine, "stage_b_ledger", None)
    check_true("Stage B ran (engine holds a chunked-load ledger)", led is not None,
               "one-shot load — chunked_weight_source returned None, or the session was disabled")
    check_true("weight_offload.stage_b_chunked_load engaged",
               "weight_offload.stage_b_chunked_load" in engaged_ledger)
    if led is not None:
        print(f"  {led.describe()}", flush=True)
        # One chunk for the body plus one per decoder layer. A count that is not this means the
        # enumeration and the config disagree about the depth — `assert_complete` catches the
        # missing tensors, but the count is what says in WHICH direction.
        check("chunks == 1 body + 1/layer", led.chunks, args.layers + 1)
        check("layers placed == model layers",
              led.placed_host_layers + led.placed_device_layers + led.placed_cpu_layers,
              args.layers)
        check("cpu-compute layers placed == asked for",
              led.placed_cpu_layers, int(args.cpu_layers))
        out.update(
            stage_b_chunks=int(led.chunks),
            stage_b_keys_filled=int(led.keys_filled),
            stage_b_peak_device_allocated=int(led.peak_device_allocated_torch),
            stage_b_peak_device_reserved=int(led.peak_device_reserved),
            stage_b_min_device_free=int(led.min_device_free),
            stage_b_peak_host_rss=int(led.peak_host_rss),
            stage_b_seconds=round(float(led.seconds), 1),
            stage_b_host_layers=int(led.placed_host_layers),
            stage_b_device_layers=int(led.placed_device_layers),
            stage_b_cpu_layers=int(led.placed_cpu_layers),
        )

    # [2b] THE CPU TIER'S COUNTERS. Deliberately NOT the `engaged()` ledger: that is a SET and
    # saturates at one, so it cannot tell a tier that bound 21 layers and executed ONE from one
    # that executed all 21. These are monotone, and they are read from BOTH sides of the ctypes
    # boundary — the Python wrapper's count and the `.so`'s own atomic — so "the forward never
    # reached the tier" is distinguishable from "the tier ran and returned nothing".
    if args.cpu_layers:
        cw = getattr(woff.driver, "cpu_worker", None)
        check_true("a CPU-tier worker exists", cw is not None)
        if cw is not None:
            out["cpu_tier_boot_counters"] = cw.backend.counters()
            print(f"  cpu tier: {out['cpu_tier_boot_counters']}", flush=True)
            check("cpu layers registered with the backend",
                  cw.backend.num_layers, int(args.cpu_layers))

    # [2] the bake actually moved bytes, and the arena served every one of them.
    check_true("bake copied bytes into the arena", woff.accounting.copied_bytes > 0,
               _gib(woff.accounting.copied_bytes))
    runtime = woff.driver
    arena = runtime.arena
    out["arena_pinned_bytes"] = int(arena.pinned_bytes)
    out["arena_carved_bytes"] = int(arena.carved_bytes)
    # The counter lives on the ARENA, not on the pool: `ArenaMemPool._fallback` increments
    # `arena.torch_fallbacks`, so a pool-side read is a permanent -1 that reads as a failure.
    out["arena_torch_fallbacks"] = int(arena.torch_fallbacks)
    check("arena hipMalloc fallbacks", out["arena_torch_fallbacks"], 0)
    print(f"  arena pinned={_gib(arena.pinned_bytes)} carved={_gib(arena.carved_bytes)} "
          f"served={_gib(runtime.pool.served_bytes)}", flush=True)

    # [3] THE SEAM PROOF — off the LIVE model the engine will call forward() on.
    print("\n[3] assert the seam against the live engine model", flush=True)
    proof = woff.seam_proof
    check_true("seal() produced a seam residency proof", proof is not None)
    if proof is not None:
        print(f"  {proof.describe()}", flush=True)
        check_true("proof consulted the arena (pointer-checked)", proof.pointer_checked,
                   "an unchecked proof is a structural claim, not residency")
        check_true("at least one HOST-resident MoE layer", proof.host_layers > 0,
                   f"{proof.host_layers} host / {proof.device_layers} device")
        check_true("host bytes proven inside the arena == bake's copied bytes",
                   proof.host_bytes == woff.accounting.copied_bytes,
                   f"{proof.host_bytes} vs {woff.accounting.copied_bytes}")
        out.update(
            seam_moe_layers=proof.moe_layers,
            seam_host_layers=proof.host_layers,
            seam_device_layers=proof.device_layers,
            seam_host_bytes=int(proof.host_bytes),
            seam_device_bytes=int(proof.device_bytes),
            seam_pointer_checked=bool(proof.pointer_checked),
        )
    # Independently of the proof object: walk the engine's model here, in the TEST, and confirm the
    # seams are the ones the plan named. A proof computed by the code under test and then read back
    # is a weaker statement than the same walk done from outside it.
    live = dict(discover_moe_layers(llm.engine.model))
    planned_host = {p.path for p in runtime.plan.placements if p.kind is StackKind.HOST}
    bound_host = {
        path for path, layer in live.items()
        if getattr(layer, "_weight_offload", None) is not None
        and layer._weight_offload.kind is StackKind.HOST
    }
    check("live MoE layers == model layers", len(live), args.layers)
    check_true("HOST layers on the live model == the plan's HOST set",
               bound_host == planned_host, f"{sorted(bound_host)} vs {sorted(planned_host)}")
    check_true("every live MoE layer carries a bound, frozen seam",
               all(getattr(l, "_weight_offload", None) is not None
                   and l._weight_offload.bound and l._weight_offload.frozen
                   for l in live.values()))
    # And the byte-level claim, made from outside: at least one tensor of a host layer is a pointer
    # the arena owns. If `owns_pointer` were vacuously true this would also pass, so the DEVICE
    # counter-check below is what makes it non-vacuous.
    host_owned = dev_owned = 0
    for path, layer in live.items():
        seam = layer._weight_offload
        for _n, t in seam.live_tensors():
            owned = arena.owns_pointer(t.data_ptr(), t.numel() * t.element_size())
            if seam.kind is StackKind.HOST:
                host_owned += owned
            else:
                dev_owned += owned
    check_true("host tensors are arena-owned", host_owned > 0, f"{host_owned} tensors")
    check("device tensors are NOT arena-owned (proof is non-vacuous)", dev_owned, 0)

    # [3b] THE STREAM TIER — asserted from outside, on the live model, for the same reason the seam
    # is: a tier object that reports itself healthy while no layer's containers were actually aliased
    # onto it is exactly the shape of green this file exists to refuse. The load-bearing check is the
    # DATA POINTER one: every streamed layer's op buffers must be the SAME allocation, because that
    # aliasing is what makes N layers cost one layer's VRAM.
    tier = llm.engine._woff_stream
    if args.stream_layers > 0:
        print("\n[3b] the stream tier", flush=True)
        check_true("engine built a stream tier", tier is not None)
        check_true("weight_offload.stream_tier engaged",
                   "weight_offload.stream_tier" in engaged_ledger)
        if tier is not None:
            check("stream layers adopted", len(tier.ops), args.stream_layers)
            check("stream tier is armed (rows poisoned)", tier.armed, True)
            check("streamed layers are the model's LAST N",
                  sorted(tier.ops), list(range(args.layers - args.stream_layers, args.layers)))
            # No streamed layer may be in the plan at all: if one were, the arena would hold a
            # reservation for weights that never arrive and the KV pool was sized against it.
            planned_idx = {int(p.path.split(".")[2]) for p in runtime.plan.placements}
            check_true("no streamed layer is in the placement plan",
                       not (set(tier.ops) & planned_idx),
                       f"overlap {sorted(set(tier.ops) & planned_idx)}")
            ptrs = {
                attr: {n: t.data_ptr() for n, t in comps.items()}
                for attr, comps in tier.shared.items()
            }
            aliased = all(
                getattr(getattr(live[p], attr), n).data_ptr() == ptrs[attr][n]
                for lid in tier.ops
                for p in [next(k for k in live if k.split(".")[2] == str(lid))]
                for attr in ptrs
                for n in ptrs[attr]
            )
            check_true("every streamed layer's op buffers are ONE shared allocation", aliased)
            out.update(tier.stats())
    else:
        check("no stream tier when none was asked for", tier, None)

    # [4] the post-capture gate FIRED. Counted, not called: it has been on the call path since
    # M1-A and has never once reached its body, because `enabled` was False on every serve.
    print("\n[4] the post-capture gate", flush=True)
    out["verify_after_capture_fired"] = int(woff.verify_after_capture_fired)
    check_true("verify_after_capture() reached its body", woff.verify_after_capture_fired >= 1,
               f"{woff.verify_after_capture_fired} call(s) — Engine.__init__ and Scheduler.__init__")
    check_true("post-capture seam proof is present", woff.seam_proof is not None)

    # [5]/[6] a real forward, then the engaged ledger.
    print("\n[5] generate through host-resident experts", flush=True)
    res = llm.generate(
        ["The capital of France is", "Write one sentence about bicycles"],
        SamplingParams(temperature=0.0, max_tokens=args.max_tokens),
    )
    for i, r in enumerate(res):
        print(f"  [{i}] {len(r['token_ids'])} tok: {r['text']!r}", flush=True)
    check_true("both requests produced tokens",
               all(len(r["token_ids"]) > 0 for r in res),
               f"{[len(r['token_ids']) for r in res]}")
    out["token_ids"] = [list(r["token_ids"]) for r in res]
    out["text"] = [r["text"] for r in res]

    print("\n[6] engaged ledger", flush=True)
    woff_marks = sorted(n for n in engaged_ledger if n.startswith("weight_offload."))
    for n in woff_marks:
        print(f"  [hip-engage] {n}", flush=True)
    out["engaged"] = woff_marks
    # THE LEDGER LINE THAT MATTERS. Boot logs prove the bake ran; only this proves a forward
    # actually resolved a HOST-placed layer through the seam. A dispatch regression that left
    # `_weight_offload` unset would silently drop this line and nothing else would change.
    check_true("weight_offload.moe_resolve[host] engaged",
               "weight_offload.moe_resolve[host]" in woff_marks,
               "a forward resolved a HOST-placed MoE layer through the seam")
    # THE CPU TIER'S PROOF, AND WHY IT IS NOT THE LEDGER LINE ABOVE.
    #
    # `weight_offload.moe_cpu_forward[cpu]` appears in `woff_marks` after ONE forward touches ONE
    # CPU layer, and never changes again — `engaged()` is a SET. So it cannot distinguish a tier
    # that bound 21 layers and executed 1 from one that executed all 21, which is exactly the
    # dispatch regression the per-leg diff is supposed to catch.
    #
    # The counters can, and they are read from BOTH sides of the ctypes boundary: `python_*` counts
    # calls that entered the wrapper, `native_*` counts calls the `.so` completed. Equal and nonzero
    # is the only healthy state — python > native means calls are dying inside the native call,
    # native > python is impossible and would mean a second caller.
    if args.cpu_layers:
        cw = getattr(woff.driver, "cpu_worker", None)
        if cw is not None:
            c = cw.backend.counters()
            # SPLIT THE SEAM FROM THE CORE. `compute_seconds` is wall time inside the worker
            # thread's `backend.compute` call; `native_layer_calls` divides it into a per-layer
            # figure that is directly comparable to the 0.517 ms/layer the standalone kernel bench
            # measured. Anything the STEP spends beyond that is handoff — the D2H sync, the queue
            # round trip, the GIL, the H2D of the partial — and that is the term no bench can see.
            c["worker_compute_seconds"] = round(float(cw.compute_seconds), 4)
            c["worker_layers_computed"] = int(cw.layers_computed)
            if c["native_layer_calls"]:
                c["ms_per_layer_in_backend"] = round(
                    1000.0 * cw.compute_seconds / c["native_layer_calls"], 4)
            out["cpu_tier_counters"] = c
            print(f"  cpu tier AFTER forward: {c}", flush=True)
            check_true("the CPU tier actually executed layers (COUNTER, not the engaged set)",
                       c["native_layer_calls"] > 0,
                       f"native_layer_calls={c['native_layer_calls']} "
                       f"python_layer_calls={c['python_layer_calls']}")
            check("native and python layer-call counts agree",
                  c["native_layer_calls"], c["python_layer_calls"])
            check_true("moe_cpu_forward engaged",
                       "weight_offload.moe_cpu_forward[cpu]" in woff_marks)
            # Every CPU layer must have run, not just the first one the ledger saw.
            out["cpu_tier_layer_calls_per_layer"] = (
                c["native_layer_calls"] / max(1, int(args.cpu_layers))
            )

    check_true("weight_offload.stage_a_sealed engaged",
               "weight_offload.stage_a_sealed" in woff_marks)
    check_true("weight_offload.verify_after_capture engaged",
               "weight_offload.verify_after_capture" in woff_marks)

    # [6b] CAPTURE PARITY + the once-per-forward PLE ledger.
    if args.parity_steps > 0:
        out.update(_capture_parity(llm, args, engaged_ledger))

    # [6c] THE THROUGHPUT A/B. Deliberately AFTER parity: a tok/s number is only worth reading once
    # the captured graph has been shown to compute the same thing as eager.
    if args.repro_probe > 0:
        out.update(_repro_probe(llm, args))

    if args.throughput_tokens > 0:
        out.update(_capture_throughput_ab(llm, args))

    # [6d] the SAMPLED, provenance-gated A/B — the deliverable number.
    if args.ab_tokens > 0:
        out.update(_capture_ab_sampled(llm, args, model_dir))

    # [7] QUALITY. Only at full depth: a layer PREFIX of a 48-layer model produces meaningless text
    # by construction, so asking for coherence there would be asking the harness to lie.
    if args.quality_tokens > 0 and args.quality_prompt:
        print("\n[7] sampled generation (checkpoint sampler, NEVER greedy)", flush=True)
        import json as _json

        gc_path = os.path.join(model_dir, "generation_config.json")
        gcfg = _json.load(open(gc_path)) if os.path.exists(gc_path) else {}
        sp = SamplingParams(
            temperature=float(gcfg.get("temperature", 1.0)),
            top_k=int(gcfg.get("top_k", 20)),
            top_p=float(gcfg.get("top_p", 0.95)),
            max_tokens=args.quality_tokens,
        )
        # The chat template, because this checkpoint is instruction-tuned and a raw continuation
        # prompt measures a different model than the one anyone serves.
        prompt = args.quality_prompt
        try:
            tok = llm.engine.tokenizer if hasattr(llm.engine, "tokenizer") else None
            if tok is None:
                from transformers import AutoTokenizer

                tok = AutoTokenizer.from_pretrained(model_dir)
            prompt = tok.apply_chat_template(
                [{"role": "user", "content": args.quality_prompt}],
                tokenize=False, add_generation_prompt=True,
            )
        except Exception as e:  # pragma: no cover - reported, never silently skipped
            print(f"  (chat template unavailable: {e!r}; using the raw prompt)", flush=True)
        out["quality_sampler"] = {"temperature": sp.temperature, "top_k": sp.top_k,
                                  "top_p": sp.top_p, "max_tokens": args.quality_tokens}
        out["quality_prompt"] = args.quality_prompt
        # CONTAINED, and only here. Everything above is a gate whose failure must abort; this leg is
        # a measurement, and losing the whole JSON — every gate result, every byte count from a
        # five-minute boot — because a 160-token sample OOM'd is a harness defect, not honesty. The
        # failure is RECORDED in the output and counted, never swallowed.
        try:
            tq = time.perf_counter()
            qres = llm.generate([prompt], sp)
            dt = time.perf_counter() - tq
            ntok = len(qres[0]["token_ids"])
            out["quality_text"] = qres[0]["text"]
            out["quality_tokens_out"] = ntok
            out["quality_seconds"] = round(dt, 2)
            out["tok_per_s"] = round(ntok / dt, 3) if dt > 0 else 0.0
            print(f"  {ntok} tok in {dt:.1f}s = {out['tok_per_s']} tok/s", flush=True)
            print(f"  {qres[0]['text']!r}", flush=True)
            check_true("quality leg produced tokens", ntok > 0)
        except BaseException as e:
            out["quality_error"] = f"{type(e).__name__}: {e}"
            check_true("quality leg produced tokens", False, out["quality_error"][:300])
            traceback.print_exc()

    # [8] HC FUSION ON/OFF. DELIBERATELY LAST, and contained, for a memory reason rather than a
    # logical one: reverting the pack needs its own contiguous copy of every `[lowrank, wide]`
    # weight, and although the flip frees the packed buffer as it goes, it is still ~640 MB of
    # allocator churn on a card this operating point leaves ~0.86 GiB free on. Every other
    # measurement in this run is already in `out` by the time it starts, so if it OOMs it costs
    # itself and nothing else. Its failure is RECORDED, never swallowed.
    if args.hc_ab_steps > 0:
        try:
            out.update(_hc_fusion_ab(llm, args))
        except BaseException as e:
            out["hc_ab_error"] = f"{type(e).__name__}: {e}"
            check_true("HC fusion A/B ran", False, out["hc_ab_error"][:300])
            traceback.print_exc()

    # LAST, so the counters cover every forward this file ran. Taken at [3b] they read all zeros —
    # the tier had been armed but nothing had been generated yet, which reads as "the tier never
    # engaged" and is exactly the wrong conclusion.
    if tier is not None:
        out.update(tier.stats())
        check_true("the stream tier actually staged experts during the forwards",
                   tier.stages > 0 and tier.experts_staged > 0, tier.describe())

    # [8] WEIGHT IDENTITY. DEAD LAST, after every timed leg, because it streams 34 GiB through the
    # host mapping: run earlier it would sit inside the throughput A/B's cache and PCIe state and
    # make the tok/s number a measurement of this probe. The weights are frozen at seal(), so the
    # position cannot change what it reads.
    if args.weight_digest:
        out.update(_weight_digest(llm, args))

    out["kv_pages"] = int(llm.engine.num_pages)
    out["failures"] = _failures
    print(f"\n[rank {rank}] {'PASS' if not _failures else 'FAIL'}: {_failures} failure(s)",
          flush=True)
    print(json.dumps(out, indent=2), flush=True)
    return out


def _spawn_target(rank: int, tp: int, args, model_dir: str, q) -> None:
    """`mp.Process` entry. Reports through the queue INCLUDING on failure: a rank that dies with a
    bare traceback and no record makes the parent guess which of the two ranks broke, and the answer
    is usually the one whose message got lost."""
    try:
        out = rank_main(rank, tp, args, model_dir)
    except BaseException as e:  # noqa: BLE001 - the failure IS the result
        traceback.print_exc()
        q.put({"rank": rank, "tp": tp, "failures": 1,
               "error": f"{type(e).__name__}: {e}"[:4000]})
        raise
    q.put(out)


def main() -> int:
    args = build_argparser().parse_args()

    if not torch.cuda.is_available():
        print("FAIL: no HIP device visible (is_rocm false?) — check device passthrough")
        return 1
    if args.tp > torch.cuda.device_count():
        print(f"FAIL: --tp {args.tp} but only {torch.cuda.device_count()} device(s) visible. "
              f"TP=2 needs ROCR_VISIBLE_DEVICES=0,1 with HIP_VISIBLE_DEVICES UNSET.")
        return 1

    # Built ONCE, in the parent, and passed down. Each rank building its own would give the ranks
    # different `model_path`s — harmless for the weights (the symlinks resolve to the same shards)
    # but it would make any per-rank path in the output incomparable, and it duplicates a directory
    # walk over a 38-shard checkpoint per rank.
    model_dir = subset_dir(args.model, args.layers, args.experts)
    print(f"[subset] {model_dir} ({args.layers} layers, {args.experts} experts, tp={args.tp})",
          flush=True)

    if args.tp == 1:
        # INLINE, deliberately: the TP=1 path must not acquire a process boundary it did not have,
        # or every TP=1 result this file has ever produced becomes incomparable with the next one.
        results = [rank_main(0, 1, args, model_dir)]
    else:
        import multiprocessing as mp

        # spawn, not fork: a forked child inherits the parent's HIP context and `Engine.__init__`'s
        # `not torch.cuda.is_initialized()` assert fires — or worse, does not, and the child bills
        # its KV pool against the parent's VRAM baseline. Same choice server/launch.py and
        # tools/kv_fp8_calibrate.py make, for the same reason.
        mp.set_start_method("spawn", force=True)
        q: "mp.Queue" = mp.Queue()
        procs = []
        for rank in range(args.tp):
            p = mp.Process(target=_spawn_target, args=(rank, args.tp, args, model_dir, q),
                           name=f"q4e-TP{rank}")
            p.start()
            procs.append(p)
        # Drain WHILE they run. A rank's `out` dict carries token id lists and can exceed the pipe
        # buffer, and joining first deadlocks the writer against a full pipe (the same trap
        # kv_fp8_calibrate.py documents).
        results = []
        alive = list(procs)
        while alive:
            while not q.empty():
                results.append(q.get())
            alive = [p for p in alive if p.is_alive()]
            if alive:
                time.sleep(0.5)
        for p in procs:
            p.join()
        while not q.empty():
            results.append(q.get())
        codes = [p.exitcode for p in procs]
        print(f"\n[parent] rank exit codes {codes}", flush=True)
        if any(c != 0 for c in codes):
            results.append({"rank": -1, "failures": 1, "error": f"rank exit codes {codes}"})
        if len(results) != args.tp and not any(r.get("rank") == -1 for r in results):
            results.append({"rank": -1, "failures": 1,
                            "error": f"{len(results)}/{args.tp} ranks reported"})

    results.sort(key=lambda r: r.get("rank", 0))
    failures = sum(int(r.get("failures", 0)) for r in results)

    # CROSS-RANK, in the parent: the ranks are in lockstep on an identical request stream, so they
    # must emit identical greedy ids. A divergence here is the single loudest signal that a shard is
    # wrong — every silent-wrong-weights failure mode in this file's checks (an unreplicated HC, a
    # diverged CT sign, a mis-split GDN head) surfaces as two ranks that stop agreeing, and NONE of
    # them raises anywhere.
    ranks = [r for r in results if r.get("rank", -1) >= 0 and "token_ids" in r]
    if len(ranks) > 1:
        ref = ranks[0]["token_ids"]
        same = all(r["token_ids"] == ref for r in ranks)
        print(f"\n[parent] cross-rank greedy ids identical: {same}", flush=True)
        if not same:
            for r in ranks:
                print(f"  rank {r['rank']}: {r['token_ids']}", flush=True)
            failures += 1

    out = {
        "tp": args.tp,
        "layers": args.layers,
        "device_gb": args.device_gb,
        "cards": [r.get("card") for r in results],
        "device_indices": [r.get("device_index") for r in results],
        "failures": failures,
        "ranks": results,
    }
    if ranks:
        out["token_ids"] = ranks[0]["token_ids"]
        out["text"] = ranks[0].get("text")
    print(f"\n{'PASS' if not failures else 'FAIL'}: {failures} failure(s) across "
          f"{len(results)} rank record(s)", flush=True)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"wrote {args.json}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException:
        traceback.print_exc()
        raise SystemExit(1)
