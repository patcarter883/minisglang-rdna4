#!/usr/bin/env python3
"""Attribute the qwen4_exp offloaded decode step — where do the 79.86 ms actually go?

WHY THIS FILE EXISTS
--------------------
`docs/measurements/QWEN4EXP_GRAPH_CAPTURE.md` §2.3 leaves 32.9 ms/step UNATTRIBUTED: the engine's
own banner projects 47.0 ms (7.5 ms compute floor + 39.25 ms of host-expert PCIe + 0.24 ms device
experts) against a measured 79.86. That gap is larger than every remaining lever combined. This
harness measures where the step goes, by CLASS, on the real serve.

METHOD, and its two known limits stated up front
------------------------------------------------
* **HIP events, on the EAGER leg.** The torch profiler (and any Python-level instrument) is blind
  inside a captured graph — replay dispatches no Python. So the decomposition runs on the eager leg
  and the captured/eager delta is measured separately in the same boot; the published delta is
  2.02 ms/step (81.87 -> 79.86), i.e. ~2.5%. Every per-class number below therefore describes an
  81-ms step, not a 79.9-ms one, and the difference is a fixed launch-overhead term, not a
  reweighting.
* **Deferred event reads.** `_lp_timed` in `models/qwen3_5.py` calls `Event.synchronize()` per
  region, which serialises the step and destroys the total. This one records into a preallocated
  pool and reads the whole pool ONCE per step, after a single sync — so the GPU timeline is
  undisturbed apart from ~1000 `hipEventRecord`s. The distortion is MEASURED, not assumed: leg B
  (eager, uninstrumented) and leg C (eager, instrumented) are both timed, and the difference is
  reported as `instrument_overhead_ms_per_step`.

THE NATURAL CONTROL THIS RUN GETS FOR FREE
------------------------------------------
11 of the 48 MoE layers are DEVICE-resident and 37 are HOST-resident (pinned arena, read over PCIe
by the grouped kernel itself — there is no separate copy). Same shape, same kernel, same routing
statistics; the ONLY difference is which medium the weight pointer names. So
`moe.routed[host] per layer - moe.routed[dev] per layer` is a MEASURED PCIe penalty, not a derived
one, and it is the number §2.3's 39.25 ms arithmetic has to be checked against.

TIMING HYGIENE this box has already broken once
-----------------------------------------------
* The decode panel and `STEP_LOG` measure WALL time of `Scheduler._forward`; under capture that
  returns after enqueueing an async replay. Never a decode number. Totals here come from
  wall(generate, N tokens) - wall(generate, N0 tokens), which cancels prefill and tokenizer.
* The GPU lease is waived for this task; a co-tenant landing mid-leg inflated one earlier
  measurement 8.6x. Legs are INTERLEAVED and repeated; the summary statistic is the MINIMUM.
* Card 1's root port is Gen4 x8 (14.48 GB/s) vs card 0's Gen5 x8 (28.93). Which physical card a
  rank landed on is recorded per rank; a TP=2 step runs in lockstep, so the slow rank gates.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from collections import defaultdict
from contextlib import contextmanager

import torch

sys.path.insert(0, "/engine/tests")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "tests"))

MODEL = os.environ.get("Q4E_MODEL", "/model")


# ─────────────────────────────────────────────────────────────────────────── the event profiler

class EvProf:
    """Preallocated HIP-event region timer with a DEFERRED read.

    One `hipEventRecord` per region boundary and exactly ONE `torch.cuda.synchronize()` per step.
    Regions nest (moe.total contains moe.routed); the reporting side owns the hierarchy, this side
    only accumulates. `enabled=False` makes every wrapper a straight passthrough, so the same
    wrapped model can serve an uninstrumented leg.
    """

    def __init__(self, cap: int = 16384) -> None:
        self.pool = [torch.cuda.Event(enable_timing=True) for _ in range(cap)]
        self.cap = cap
        self.i = 0
        self.pending: list = []
        self.acc: dict = defaultdict(float)
        self.cnt: dict = defaultdict(int)
        self.steps = 0
        self._steps_pending = 0
        self.enabled = False
        self.overflow = 0
        self.syncs = 0

    def _ev(self):
        if self.i >= self.cap:
            self.overflow += 1
            return None
        e = self.pool[self.i]
        self.i += 1
        return e

    @contextmanager
    def region(self, label: str):
        if not self.enabled:
            yield
            return
        s = self._ev()
        if s is None:
            yield
            return
        s.record()
        try:
            yield
        finally:
            e = self._ev()
            if e is not None:
                e.record()
                self.pending.append((label, s, e))

    def wrap(self, obj, attr: str, label):
        """Bind an instrumented shim over `obj.attr` on the INSTANCE (never the class).

        `label` may be a callable so a per-call label can depend on nothing but the site; it is
        resolved once, here, not per call.
        """
        fn = getattr(obj, attr)
        lbl = label() if callable(label) else label

        def shim(*a, **kw):
            with self.region(lbl):
                return fn(*a, **kw)

        setattr(obj, attr, shim)

    def step_end(self) -> None:
        """Called once per model forward. Flushing is LAZY — a `torch.cuda.synchronize()` per step
        would serialise host and device and make the instrumented leg's wall time meaningless. The
        pool holds ~16 steps of regions, so the sync lands roughly once per 16 steps."""
        if not self.enabled:
            self.i = 0
            return
        self._steps_pending += 1
        if self.i > (self.cap * 7) // 8:
            self.flush()

    def flush(self) -> None:
        if not self.pending:
            self.i = 0
            self.steps += self._steps_pending
            self._steps_pending = 0
            return
        torch.cuda.synchronize()
        self.syncs += 1
        for label, s, e in self.pending:
            self.acc[label] += s.elapsed_time(e)
            self.cnt[label] += 1
        self.pending.clear()
        self.i = 0
        self.steps += self._steps_pending
        self._steps_pending = 0

    def reset(self) -> None:
        self.pending.clear()
        self.acc.clear()
        self.cnt.clear()
        self.i = 0
        self.steps = 0
        self._steps_pending = 0
        self.overflow = 0
        self.syncs = 0

    def per_step(self) -> dict:
        n = max(1, self.steps)
        return {k: round(v / n, 4) for k, v in sorted(self.acc.items(), key=lambda x: -x[1])}

    def counts_per_step(self) -> dict:
        n = max(1, self.steps)
        return {k: round(v / n, 3) for k, v in sorted(self.cnt.items())}


# ─────────────────────────────────────────────────────────────────────────── instrumentation

def install(llm, prof: EvProf) -> dict:
    """Wrap the qwen4_exp forward at class boundaries. Returns the placement census."""
    from minisgl.weights.stacks import StackKind

    model = llm.engine.model          # Qwen4ExpForConditionalGeneration
    inner = model.model               # Qwen4ExpModel
    layers = inner.layers.op_list

    census = {"device": [], "host": [], "cpu": [], "unseamed": []}
    for i, L in enumerate(layers):
        seam = getattr(L.mlp.experts, "_weight_offload", None)
        if seam is None:
            census["unseamed"].append(i)
            tier = "dev"
        else:
            k = seam.kind
            if k is StackKind.HOST:
                census["host"].append(i)
                tier = "host"
            elif k is StackKind.CPU:
                census["cpu"].append(i)
                tier = "cpu"
            else:
                census["device"].append(i)
                tier = "dev"

        mixer = "gdn" if getattr(L, "linear_attn", None) is not None else "attn"
        if getattr(L, "ple", None) is not None:
            prof.wrap(L.ple, "forward", "ple")
        prof.wrap(L.attn_hyper_connection, "mix", "hc.mix")
        prof.wrap(L.attn_hyper_connection, "combine", "hc.combine")
        prof.wrap(L.mlp_hyper_connection, "mix", "hc.mix")
        prof.wrap(L.mlp_hyper_connection, "combine", "hc.combine")
        prof.wrap(L._attn_op, "forward", f"mixer.{mixer}")
        prof.wrap(L.mlp, "forward", f"moe.total.{tier}")
        prof.wrap(L.mlp.gate, "forward", "moe.router")
        prof.wrap(L.mlp.shared_expert, "forward", "moe.shared")
        prof.wrap(L.mlp.shared_expert_gate, "forward", "moe.shared_gate")
        prof.wrap(L.mlp.experts, "forward", f"moe.routed.{tier}")
        prof.wrap(L.mlp.experts._comm, "all_reduce", "moe.all_reduce")

    prof.wrap(inner.embed_tokens, "forward", "embed")
    prof.wrap(inner.hyper_connection_mixer, "mix", "hc.final_mix")
    prof.wrap(model.lm_head, "forward", "lm_head")
    prof.wrap(model, "forward", "MODEL_TOTAL")
    return census


# ─────────────────────────────────────────────────────────────────────────── isolation replay

def isolate(llm, args) -> dict:
    """Replay two component families on the LIVE weights with ~2 event boundaries per pass.

    WHY THIS EXISTS ALONGSIDE THE PER-REGION PASS. The region profiler puts an event pair around
    every hyper-connection `mix`/`combine` — 192 boundaries per step against a 10 ms total — and the
    instrument's own cost lands in exactly the small-region terms it is trying to size. This replays
    the SAME live blocks in a tight loop with one event pair per whole pass, so the per-boundary cost
    is divided by 97 blocks instead of charged to each.

    THE CACHE TRAP THIS AVOIDS: replaying ONE block in a loop would read 13.2 MB, which fits the
    64 MB MALL, and would report an HBM figure the served path never sees. Every pass here walks all
    97 blocks (1.28 GB) / all 48 expert stacks (737 MB), so the working set is 20x the MALL.

    Runs BEFORE `install()` so no wrapper Python is in the loop.
    """
    inner = llm.engine.model.model
    layers = inner.layers.op_list
    dev = llm.engine.device
    hc0 = layers[0].attn_hyper_connection
    wide, H = hc0.wide_size, hc0.hidden_size
    x = torch.randn(1, wide, dtype=torch.bfloat16, device=dev)
    y = torch.randn(1, H, dtype=torch.bfloat16, device=dev)
    R = max(1, int(args.isolate_reps))
    o: dict = {"isolate_reps": R, "hc_blocks": 2 * len(layers) + 1}

    def timed(fn, reps=R):
        for _ in range(2):
            fn()
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(reps):
            fn()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / reps

    def hc_pass():
        for L in layers:
            _, r = L.attn_hyper_connection.mix(x)
            L.attn_hyper_connection.combine(y, r)
            _, r = L.mlp_hyper_connection.mix(x)
            L.mlp_hyper_connection.combine(y, r)
        inner.hyper_connection_mixer.mix(x)

    def hc_mix_only():
        for L in layers:
            L.attn_hyper_connection.mix(x)
            L.mlp_hyper_connection.mix(x)
        inner.hyper_connection_mixer.mix(x)

    def hc_down_only():
        for L in layers:
            L.attn_hyper_connection.input_mix_weight_down.forward(x)
            L.mlp_hyper_connection.input_mix_weight_down.forward(x)
        inner.hyper_connection_mixer.input_mix_weight_down.forward(x)

    t = hc0.input_mix_weight_down.forward(x).contiguous()

    def hc_up_only():
        for L in layers:
            L.attn_hyper_connection.input_mix_weight_up.forward(t)
            L.mlp_hyper_connection.input_mix_weight_up.forward(t)
        inner.hyper_connection_mixer.input_mix_weight_up.forward(t)

    def hc_norm_only():
        for L in layers:
            L.attn_hyper_connection.hc_norm.forward(x)
            L.mlp_hyper_connection.hc_norm.forward(x)
        inner.hyper_connection_mixer.hc_norm.forward(x)

    for name, fn in (("hc_full", hc_pass), ("hc_mix_only", hc_mix_only),
                     ("hc_down_gemv", hc_down_only), ("hc_up_gemv", hc_up_only),
                     ("hc_norm", hc_norm_only)):
        try:
            o[f"{name}_ms"] = round(timed(fn), 4)
        except Exception as ex:  # reported, never silently skipped
            o[f"{name}_error"] = f"{type(ex).__name__}: {ex}"[:300]

    # routed experts, host tier vs device tier, from the same driver
    from minisgl.weights.stacks import StackKind
    h = torch.randn(1, H, dtype=torch.bfloat16, device=dev)
    host_ls, dev_ls = [], []
    for L in layers:
        seam = getattr(L.mlp.experts, "_weight_offload", None)
        (host_ls if (seam is not None and seam.kind is StackKind.HOST) else dev_ls).append(L)

    def experts_pass(ls):
        def go():
            for L in ls:
                lg = L.mlp.gate.forward(h)
                L.mlp.experts.forward(h, lg, reduce=False)
        return go

    try:
        o["experts_host_ms"] = round(timed(experts_pass(host_ls)), 4)
        o["experts_dev_ms"] = round(timed(experts_pass(dev_ls)), 4)
        o["experts_host_layers"] = len(host_ls)
        o["experts_dev_layers"] = len(dev_ls)
        if host_ls and dev_ls:
            hp = o["experts_host_ms"] / len(host_ls)
            dp = o["experts_dev_ms"] / len(dev_ls)
            o["experts_ms_per_host_layer"] = round(hp, 4)
            o["experts_ms_per_dev_layer"] = round(dp, 4)
            b = 10 * args.expert_bytes_per_rank
            o["experts_host_GBps"] = round(b / (hp * 1e-3) / 1e9, 3)
    except Exception as ex:
        o["experts_error"] = f"{type(ex).__name__}: {ex}"[:400]
    return o


# ─────────────────────────────────────────────────────────────────────────── byte model

def byte_model(cfg, tp: int) -> dict:
    """Per-rank bytes a bs=1 decode step must READ, from the config alone. Arithmetic, labelled as
    such — it is the denominator every achieved-bandwidth figure below is divided by."""
    H = cfg["hidden_size"]
    hc = cfg["hc_count"]
    lr = cfg["hc_lowrank"]
    L = cfg["num_hidden_layers"]
    wide = hc * H
    n_attn = sum(1 for t in cfg["layer_types"] if t == "full_attention")
    n_gdn = L - n_attn
    hc_blocks_full = 2 * L                      # per-layer attn + mlp hyper-connections
    hc_b = 2 * (wide + lr * wide + wide * lr + hc * wide)      # bf16, with block_inject
    hc_mixer = 2 * (wide + lr * wide + wide * lr)              # no block_inject
    # dense projections, bf16, TP-sharded on the head/intermediate axis
    qkv = H * (2 * cfg["num_attention_heads"] * cfg["head_dim"]
               + 2 * cfg["num_key_value_heads"] * cfg["head_dim"]) * 2
    o = (cfg["num_attention_heads"] * cfg["head_dim"]) * H * 2
    idx = H * ((cfg["indexer_n_heads"] + cfg["indexer_kv_heads"]) * cfg["indexer_head_dim"]) * 2
    kd, vd = cfg["linear_key_head_dim"], cfg["linear_value_head_dim"]
    nk, nv = cfg["linear_num_key_heads"], cfg["linear_num_value_heads"]
    gdn_in = H * (2 * nk * kd + 2 * nv * vd) * 2
    gdn_out = (nv * vd) * H * 2
    inter = cfg["moe_intermediate_size"]
    shared = (2 * inter * H + inter * H) * 2
    router = cfg["num_experts"] * H * 2
    lm_head = cfg["vocab_size"] * H * 2
    return {
        "hyper_connections_bf16": hc_blocks_full * hc_b + hc_mixer,      # replicated, NOT sharded
        "gdn_dense_bf16": n_gdn * (gdn_in + gdn_out) // tp,
        "attn_dense_bf16": n_attn * (qkv + o + idx) // tp,
        "moe_shared_bf16": L * shared // tp,
        "moe_router_bf16": L * router,                                   # replicated
        "lm_head_bf16": lm_head // tp,
        "_meta": {"n_attn": n_attn, "n_gdn": n_gdn, "hc_block_bytes": hc_b, "tp": tp},
    }


# ─────────────────────────────────────────────────────────────────────────── the run

def rank_main(rank: int, tp: int, args, model_dir: str) -> dict:
    from minisgl.core import SamplingParams
    from minisgl.distributed import DistributedInfo
    from minisgl.llm import LLM

    out: dict = {"rank": rank, "tp": tp}
    t0 = time.perf_counter()
    llm = LLM(
        model_path=model_dir,
        dtype=torch.bfloat16,
        tp_info=DistributedInfo(rank, tp),
        cuda_graph_max_bs=args.cuda_graph_max_bs,
        page_size=16,
        memory_ratio=args.memory_ratio,
        attention_backend=args.attention_backend,
        max_running_req=args.max_running_req,
        max_extend_tokens=args.max_extend_tokens,
        weight_offload_device_gb=args.device_gb,
        weight_offload_gb=args.host_gb,
        weight_offload_stream_layers=0,
    )
    out["boot_seconds"] = round(time.perf_counter() - t0, 1)
    dev = llm.engine.device
    out["card"] = torch.cuda.get_device_name(dev)
    out["device_index"] = int(getattr(dev, "index", 0) or 0)
    gr = llm.engine.graph_runner
    out["cuda_graph_bs_captured"] = sorted(getattr(gr, "graph_map", {}).keys())
    out["kv_pages"] = int(llm.engine.num_pages)

    # sampler out of the checkpoint — a top-k/top-p over 248,320 tokens is a real per-step cost and
    # greedy does not pay it. ignore_eos so both legs do the SAME amount of work.
    gcfg_path = os.path.join(model_dir, "generation_config.json")
    gcfg = json.load(open(gcfg_path)) if os.path.exists(gcfg_path) else {}
    temperature = float(gcfg.get("temperature", 1.0))
    top_k = int(gcfg.get("top_k", 20))
    top_p = float(gcfg.get("top_p", 0.95))
    out["sampler"] = {"temperature": temperature, "top_k": top_k, "top_p": top_p}

    prompt = args.prompt
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_dir)
        prompt = tok.apply_chat_template([{"role": "user", "content": args.prompt}],
                                         tokenize=False, add_generation_prompt=True)
    except Exception as e:  # pragma: no cover
        print(f"  (chat template unavailable: {e!r})", flush=True)

    def sp(n):
        return SamplingParams(temperature=temperature, top_k=top_k, top_p=top_p,
                              ignore_eos=True, max_tokens=n)

    N, N0 = int(args.tokens), int(args.tokens_short)

    def timed_leg(graphs_on: bool) -> tuple:
        """ms per DECODE step, prefill+tokenizer cancelled by the short-run subtraction."""
        saved = gr.max_graph_bs
        try:
            if not graphs_on:
                gr.max_graph_bs = 0
            llm.generate([prompt], sp(4))                     # warm this leg's path
            t = time.perf_counter(); r0 = llm.generate([prompt], sp(N0))[0]
            dt0 = time.perf_counter() - t
            t = time.perf_counter(); r1 = llm.generate([prompt], sp(N))[0]
            dt1 = time.perf_counter() - t
        finally:
            gr.max_graph_bs = saved
        n0, n1 = len(r0["token_ids"]), len(r1["token_ids"])
        if n1 <= n0:
            return None, (n0, n1)
        return 1000.0 * (dt1 - dt0) / (n1 - n0), (n0, n1)

    # ---- legs A/B: uninstrumented totals, interleaved, min-of-N ------------------------------
    print(f"\n[A/B] uninstrumented totals, {args.repeats} interleaved repeats", flush=True)
    cap_ms, eag_ms = [], []
    for _ in range(max(1, args.repeats)):
        m, nn = timed_leg(True)
        if m: cap_ms.append(m)
        m, nn = timed_leg(False)
        if m: eag_ms.append(m)
        print(f"   captured {cap_ms[-1]:.2f}  eager {eag_ms[-1]:.2f} ms/step", flush=True)
    out["ms_per_step_captured_samples"] = [round(x, 3) for x in cap_ms]
    out["ms_per_step_eager_samples"] = [round(x, 3) for x in eag_ms]
    out["ms_per_step_captured"] = round(min(cap_ms), 3) if cap_ms else None
    out["ms_per_step_eager"] = round(min(eag_ms), 3) if eag_ms else None
    if cap_ms and eag_ms:
        out["capture_saves_ms_per_step"] = round(min(eag_ms) - min(cap_ms), 3)

    # ---- isolation replay (BEFORE any wrapper is installed) -----------------------------------
    print("\n[I] isolation replay: HC blocks and expert stacks, live weights", flush=True)
    try:
        out["isolate"] = isolate(llm, args)
    except BaseException as ex:  # noqa: BLE001 - reported, never papered over
        out["isolate"] = {"error": f"{type(ex).__name__}: {ex}"[:1000],
                          "traceback": traceback.format_exc()[-2000:]}
    print("   " + json.dumps(out["isolate"])[:2000], flush=True)

    # ---- leg C: instrumented eager ------------------------------------------------------------
    prof = EvProf()
    census = install(llm, prof)
    out["placement"] = {k: (v if k == "unseamed" else len(v)) for k, v in census.items()}
    out["placement_device_layers"] = census["device"]

    # Per-step boundary hook AND the prefill gate. Every `generate` starts with a prefill forward
    # whose regions are a different shape and 10-100x the work of a decode step; folding those into
    # the per-step average would attribute prompt processing to decode. The gate is taken from the
    # live batch rather than positionally, because `generate` is called several times per leg.
    from minisgl.core import get_global_ctx

    step_flush = {"decode": 0, "prefill": 0}
    model = llm.engine.model
    _mf = model.forward

    def stepping_forward(*a, **kw):
        is_pf = bool(getattr(getattr(get_global_ctx(), "batch", None), "is_prefill", False))
        prev = prof.enabled
        prof.enabled = prev and not is_pf
        try:
            r = _mf(*a, **kw)
        finally:
            prof.enabled = prev
        if is_pf:
            step_flush["prefill"] += 1
        else:
            step_flush["decode"] += 1
            prof.step_end()
        return r

    model.forward = stepping_forward

    print(f"\n[C] instrumented eager leg, {args.tokens} tokens", flush=True)
    saved = gr.max_graph_bs
    inst_ms = []
    try:
        gr.max_graph_bs = 0
        llm.generate([prompt], sp(4))
        prof.reset()
        for _ in range(max(1, args.prof_repeats)):
            prof_reset_steps = prof.steps
            t = time.perf_counter(); r0 = llm.generate([prompt], sp(N0))[0]
            dt0 = time.perf_counter() - t
            t = time.perf_counter(); r1 = llm.generate([prompt], sp(N))[0]
            dt1 = time.perf_counter() - t
            n0, n1 = len(r0["token_ids"]), len(r1["token_ids"])
            if n1 > n0:
                inst_ms.append(1000.0 * (dt1 - dt0) / (n1 - n0))
    finally:
        gr.max_graph_bs = saved
        prof.enabled = False

    # The accumulation above ran with prof.enabled False (default) — enable and redo, so the
    # uninstrumented/instrumented pair is apples to apples on the SAME leg shape.
    prof.enabled = True
    prof.reset()
    inst_on_ms = []
    try:
        gr.max_graph_bs = 0
        llm.generate([prompt], sp(4))
        prof.reset()
        for _ in range(max(1, args.prof_repeats)):
            t = time.perf_counter(); r0 = llm.generate([prompt], sp(N0))[0]
            dt0 = time.perf_counter() - t
            t = time.perf_counter(); r1 = llm.generate([prompt], sp(N))[0]
            dt1 = time.perf_counter() - t
            n0, n1 = len(r0["token_ids"]), len(r1["token_ids"])
            if n1 > n0:
                inst_on_ms.append(1000.0 * (dt1 - dt0) / (n1 - n0))
    finally:
        gr.max_graph_bs = saved
        prof.flush()
        prof.enabled = False

    out["forwards"] = dict(step_flush)
    out["prof_syncs"] = prof.syncs
    out["ms_per_step_eager_wrapped_off"] = round(min(inst_ms), 3) if inst_ms else None
    out["ms_per_step_eager_wrapped_on"] = round(min(inst_on_ms), 3) if inst_on_ms else None
    if inst_ms and inst_on_ms:
        out["instrument_overhead_ms_per_step"] = round(min(inst_on_ms) - min(inst_ms), 3)
    out["prof_steps"] = prof.steps
    out["prof_event_overflow"] = prof.overflow
    out["gpu_ms_per_step"] = prof.per_step()
    out["regions_per_step"] = prof.counts_per_step()

    # ---- byte model + achieved bandwidth ------------------------------------------------------
    cfg = json.load(open(os.path.join(model_dir, "config.json")))["text_config"]
    bm = byte_model(cfg, tp)
    out["byte_model_per_rank"] = {k: v for k, v in bm.items() if not k.startswith("_")}
    out["byte_model_meta"] = bm["_meta"]
    g = out["gpu_ms_per_step"]
    n_host, n_dev = len(census["host"]), len(census["device"])
    per_expert_b = args.expert_bytes_per_rank
    if n_host and n_dev and "moe.routed.host" in g and "moe.routed.dev" in g:
        h_per = g["moe.routed.host"] / n_host
        d_per = g["moe.routed.dev"] / n_dev
        out["moe_routed_ms_per_host_layer"] = round(h_per, 4)
        out["moe_routed_ms_per_device_layer"] = round(d_per, 4)
        out["moe_routed_host_penalty_ms_per_step"] = round((h_per - d_per) * n_host, 3)
        b = cfg["num_experts_per_tok"] * per_expert_b
        out["measured_host_expert_GBps"] = round(b / (h_per * 1e-3) / 1e9, 3)
        out["measured_host_expert_GBps_penalty_only"] = round(
            b / ((h_per - d_per) * 1e-3) / 1e9, 3) if h_per > d_per else None
    hcb = bm["hyper_connections_bf16"]
    if "hc.mix" in g:
        hc_ms = g.get("hc.mix", 0) + g.get("hc.combine", 0) + g.get("hc.final_mix", 0)
        out["hc_ms_per_step"] = round(hc_ms, 3)
        out["hc_bytes_per_rank"] = hcb
        out["hc_achieved_GBps"] = round(hcb / (hc_ms * 1e-3) / 1e9, 2) if hc_ms else None
    if "lm_head" in g and g["lm_head"]:
        out["lm_head_achieved_GBps"] = round(
            bm["lm_head_bf16"] / (g["lm_head"] * 1e-3) / 1e9, 2)

    # ---- coherence: is the serve still producing sense? ---------------------------------------
    r = llm.generate([prompt], SamplingParams(temperature=temperature, top_k=top_k, top_p=top_p,
                                              max_tokens=48))[0]
    out["sample_text"] = r.get("text", "")[:600]
    return out


def _spawn_target(rank, tp, args, model_dir, q):
    try:
        q.put(rank_main(rank, tp, args, model_dir))
    except BaseException as e:  # noqa: BLE001
        traceback.print_exc()
        q.put({"rank": rank, "tp": tp, "error": f"{type(e).__name__}: {e}"[:4000]})
        raise


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--layers", type=int, default=48)
    ap.add_argument("--experts", type=int, default=512)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--device-gb", type=float, default=8.1)
    ap.add_argument("--host-gb", type=float, default=28.0)
    ap.add_argument("--cuda-graph-max-bs", type=int, default=2)
    ap.add_argument("--max-running-req", type=int, default=2)
    ap.add_argument("--memory-ratio", type=float, default=0.90)
    ap.add_argument("--max-extend-tokens", type=int, default=8)
    ap.add_argument("--attention-backend", default="hip")
    ap.add_argument("--tokens", type=int, default=48)
    ap.add_argument("--tokens-short", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--prof-repeats", type=int, default=2)
    ap.add_argument("--isolate-reps", type=int, default=20)
    ap.add_argument("--prompt", default="Explain in three sentences why the sky is blue.")
    ap.add_argument("--expert-bytes-per-rank", type=float, default=1.536e6,
                    help="bytes ONE routed expert costs this rank; 1.536 MB is the figure "
                         "QWEN4EXP_GRAPH_CAPTURE.md's 568.32 MB/token/rank is built from")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("FAIL: no HIP device visible")
        return 1
    if args.tp > torch.cuda.device_count():
        print(f"FAIL: --tp {args.tp} but {torch.cuda.device_count()} device(s) visible")
        return 1

    from qwen4exp_offload_serve_test import subset_dir
    model_dir = subset_dir(args.model, args.layers, args.experts)
    print(f"[subset] {model_dir} ({args.layers}L, {args.experts}E, tp={args.tp})", flush=True)

    if args.tp == 1:
        results = [rank_main(0, 1, args, model_dir)]
    else:
        import multiprocessing as mp
        mp.set_start_method("spawn", force=True)
        q = mp.Queue()
        procs = []
        for rank in range(args.tp):
            p = mp.Process(target=_spawn_target, args=(rank, args.tp, args, model_dir, q),
                           name=f"q4e-prof-{rank}")
            p.start()
            procs.append(p)
        results, alive = [], list(procs)
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
        print(f"\n[parent] exit codes {[p.exitcode for p in procs]}", flush=True)

    results.sort(key=lambda r: r.get("rank", 0))
    out = {"tp": args.tp, "layers": args.layers,
           "cards": [r.get("card") for r in results], "ranks": results}
    print(json.dumps(out, indent=2)[:20000], flush=True)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"wrote {args.json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
