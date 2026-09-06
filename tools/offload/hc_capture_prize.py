#!/usr/bin/env python3
"""How much of the hyper-connection launch cost SURVIVES graph capture?

WHY THIS FILE EXISTS
--------------------
`docs/measurements/QWEN4EXP_ENDGAME.md` §4.2 sizes the hyper-connections at 10.15 ms/step of which
"only 2.64 ms is bandwidth; the other 7.9 ms is ~1000 launch-bound kernels on 20 KB tensors". That
figure comes from an EAGER, event-instrumented boot. Graph capture already removes launch overhead
(it bought 2.5% overall), so some unknown fraction of the 7.9 ms is already gone in the served
configuration and optimising it would be wasted work. This harness sizes what is LEFT.

FOUR INDEPENDENT MEASUREMENTS, ONE BOOT
---------------------------------------
P1  step totals, captured vs eager, interleaved, min-of-N  -> the same-footing baseline, and a HARD
    CEILING on how much launch overhead capture can possibly have removed from ANY component.
P2  isolation replay of all 97 live HC blocks, EAGER vs a hand-captured hipGraph of the identical
    call sequence, whole-pass and per-component. In a captured graph there is no host in the loop,
    so an isolated replay is a faithful model of what the block costs inside the engine's graph;
    in EAGER it is a bound, not a model (see the caveat in the report).
P3  kernel census: torch profiler over the eager isolation pass -> the ACTUAL launch count and the
    per-kernel GPU-busy sum. The busy sum is the irreducible floor: no fusion beats it without
    changing the math. Capture does not change kernel count (a graph records the same launches as
    nodes) -- verified against the captured graph's own node count where the runtime exposes it.
P4  in-situ ABLATION: swap every HyperConnection.mix/.combine for a shape-preserving stub and
    re-time the decode step. The delta IS the hyper-connections' cost inside the real engine,
    measured rather than attributed. Run eager (cheap, no recapture) and -- if the recapture
    survives -- captured. Numerics are destroyed by the stub, so this phase is GREEDY on BOTH legs
    (the sampler cost is identical in both and cancels in the difference) and the coherence sample
    is taken BEFORE it.

TRAPS THIS BOX HAS ALREADY WALKED INTO
--------------------------------------
* Timing an async `g.replay()` enqueue measures nothing (it once produced a x50.66 "speedup").
  Every timing here brackets a LOOP with HIP events and syncs after the closing record.
* The decode panel / STEP_LOG measure WALL time of `Scheduler._forward`, which under capture returns
  after enqueueing. All totals here are wall(N tokens) - wall(N0 tokens), which cancels prefill and
  tokenizer, exactly as `prof_decode_attrib.py` does.
* Replaying ONE block in a loop reads 13.2 MB, fits the 64 MB MALL, and reports an HBM figure the
  served path never sees. Every pass walks all 97 blocks (1.28 GB).
* The torch profiler is blind inside a captured graph. P3 therefore profiles EAGER only and P2
  supplies the captured number by direct event timing of the replay.
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
import torch.nn.functional as F

sys.path.insert(0, "/engine/tests")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "tests"))

MODEL = os.environ.get("Q4E_MODEL", "/model")


# ───────────────────────────────────────────────────────────── event profiler (from prof_decode_attrib)

class EvProf:
    """Preallocated HIP-event region timer with a DEFERRED read: one `hipEventRecord` per region
    boundary, ONE `torch.cuda.synchronize()` per ~16 steps. `enabled=False` is a straight
    passthrough so the same wrapped model serves an uninstrumented control leg."""

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

    def wrap(self, obj, attr: str, label: str):
        fn = getattr(obj, attr)

        def shim(*a, **kw):
            with self.region(label):
                return fn(*a, **kw)

        setattr(obj, attr, shim)

    def step_end(self) -> None:
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


# ───────────────────────────────────────────────────────────────────────────── timing primitives

def ev_time(fn, reps: int) -> float:
    """ms per call, GPU timeline, one event pair around the WHOLE loop.

    The sync is AFTER the closing record, so an async enqueue cannot be mistaken for completion."""
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / reps


def make_graph(fn, pool=None):
    """Capture `fn` into a hipGraph. Warmup runs on a side stream, which is what the runtime
    requires before a capture; the caller keeps the graph alive for as long as it replays."""
    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(st)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, pool=pool):
        fn()
    torch.cuda.synchronize()
    return g


# ───────────────────────────────────────────────────────────────────── P2/P3: HC isolation + census

def hc_isolate(llm, args) -> dict:
    """Replay the 97 LIVE hyper-connection blocks, eager and captured, whole and by component.

    All 97 blocks per pass (1.28 GB of bf16 weight) so the working set is 20x the 64 MB MALL; a
    single-block loop would report an in-cache figure the served path never sees.
    """
    inner = llm.engine.model.model
    layers = inner.layers.op_list
    dev = llm.engine.device
    hc0 = layers[0].attn_hyper_connection
    wide, H, hc, lr = hc0.wide_size, hc0.hidden_size, hc0.hc_count, hc0.input_mix_weight_down.weight.shape[0]
    n_blocks = 2 * len(layers) + 1

    o: dict = {"hc_blocks": n_blocks, "layers": len(layers), "wide": wide, "hidden": H,
               "hc_count": hc, "hc_lowrank": lr, "reps": args.isolate_reps,
               "bs": args.isolate_bs}

    B = int(args.isolate_bs)
    x = torch.randn(B, wide, dtype=torch.bfloat16, device=dev)
    y = torch.randn(B, H, dtype=torch.bfloat16, device=dev)
    # component inputs, materialised ONCE so a component bench times the component and not its feed
    n_in = hc0.hc_norm.forward(x).contiguous()
    t_in = F.silu(hc0.input_mix_weight_down.forward(n_in) / hc)
    g_in = torch.sigmoid(hc0.input_mix_weight_up.forward(t_in)).contiguous()
    b_in = (2.0 * torch.sigmoid(hc0.block_inject_weight.forward(n_in) / hc)).contiguous()

    mixers = [L.attn_hyper_connection for L in layers] + [L.mlp_hyper_connection for L in layers]
    combiners = list(mixers)                       # every per-layer HC has a block_inject
    allmix = mixers + [inner.hyper_connection_mixer]   # the final mixer does mix only

    def hc_full():
        for L in layers:
            _, r = L.attn_hyper_connection.mix(x)
            L.attn_hyper_connection.combine(y, r)
            _, r = L.mlp_hyper_connection.mix(x)
            L.mlp_hyper_connection.combine(y, r)
        inner.hyper_connection_mixer.mix(x)

    def mix_full():
        for m in allmix:
            m.mix(x)

    def combine_full():
        for m in combiners:
            m.combine(y, (x, n_in))

    def c_norm():
        for m in allmix:
            m.hc_norm.forward(x)

    def c_norm_gain():
        # the (1 + w) gain of GroupedRMSNorm, isolated: an `add` on a 10240 CONSTANT plus a `mul`
        for m in allmix:
            n_in * (m.hc_norm.weight + 1.0)

    def c_down():
        for m in allmix:
            m.input_mix_weight_down.forward(n_in)

    def c_scale_silu():
        for m in allmix:
            F.silu(t_in / hc)

    def c_up():
        for m in allmix:
            m.input_mix_weight_up.forward(t_in)

    def c_sigmoid():
        for m in allmix:
            torch.sigmoid(g_in)

    def c_gate_mean():
        for m in allmix:
            (g_in.unflatten(-1, (hc, H)) * n_in.unflatten(-1, (hc, H))).mean(dim=-2)

    def c_inject():
        for m in combiners:
            2.0 * torch.sigmoid(m.block_inject_weight.forward(n_in) / hc)

    def c_add():
        for _ in combiners:
            (x.unflatten(-1, (hc, H)) + y.unsqueeze(-2) * b_in.unsqueeze(-1)).flatten(-2)

    benches = [
        ("hc_full", hc_full, n_blocks),
        ("mix_full", mix_full, len(allmix)),
        ("combine_full", combine_full, len(combiners)),
        ("c_hc_norm", c_norm, len(allmix)),
        ("c_norm_gain_only", c_norm_gain, len(allmix)),
        ("c_down_gemv", c_down, len(allmix)),
        ("c_scale_silu", c_scale_silu, len(allmix)),
        ("c_up_gemv", c_up, len(allmix)),
        ("c_sigmoid", c_sigmoid, len(allmix)),
        ("c_gate_mul_mean", c_gate_mean, len(allmix)),
        ("c_combine_inject", c_inject, len(combiners)),
        ("c_combine_add", c_add, len(combiners)),
    ]

    R = max(1, int(args.isolate_reps))
    o["eager_ms"] = {}
    o["captured_ms"] = {}
    o["calls_per_pass"] = {name: k for name, _, k in benches}
    graphs = {}
    pool = None
    for name, fn, _k in benches:
        try:
            o["eager_ms"][name] = round(ev_time(fn, R), 4)
        except BaseException as ex:
            o["eager_ms"][name] = f"ERROR {type(ex).__name__}: {ex}"[:200]
        try:
            g = make_graph(fn, pool=pool)
            if pool is None:
                pool = g.pool()
            graphs[name] = g
            o["captured_ms"][name] = round(ev_time(g.replay, R), 4)
        except BaseException as ex:
            o["captured_ms"][name] = f"ERROR {type(ex).__name__}: {ex}"[:300]

    # capture must reproduce eager, or the captured number is timing a different computation
    try:
        ref = [m.mix(x)[0].clone() for m in allmix]
        gout = [None] * len(allmix)

        def mix_into():
            for i, m in enumerate(allmix):
                gout[i] = m.mix(x)[0]

        gg = make_graph(mix_into)   # its OWN pool: a shared pool may alias another graph's buffers
        gg.replay()
        torch.cuda.synchronize()
        md = max(float((a - b).abs().max()) for a, b in zip(ref, gout))
        o["capture_vs_eager_max_abs_diff"] = md
        del gg
    except BaseException as ex:
        o["capture_vs_eager_error"] = f"{type(ex).__name__}: {ex}"[:300]

    # ---- P3 launch census, EAGER for every component + the CAPTURED replay of the whole pass ---
    o["census"] = {}
    cen = list(benches)
    if "hc_full" in graphs:
        cen.append(("hc_full_REPLAY", graphs["hc_full"].replay, n_blocks))
    for name, fn, k in cen:
        try:
            o["census"][name] = kernel_census(fn, reps=max(1, args.census_reps), passes=k)
        except BaseException as ex:
            o["census"][name] = {"error": f"{type(ex).__name__}: {ex}"[:400],
                                 "traceback": traceback.format_exc()[-1200:]}

    graphs.clear()
    return o


_LAUNCH_APIS = ("hipLaunchKernel", "hipExtModuleLaunchKernel", "hipModuleLaunchKernel",
                "hipGraphLaunch", "hipLaunchCooperativeKernel", "hipExtLaunchKernel",
                "hipMemcpyAsync", "hipMemsetAsync")


def kernel_census(fn, reps: int, passes: int) -> dict:
    """Count the HOST-SIDE kernel launches one pass issues, and attribute them to the aten op.

    MEASURED CONSTRAINT ON METHOD: this image's kineto build reports **no device activity** — every
    profiler event comes back `DeviceType.CPU` and `FunctionEvent.kernels` is empty, so per-kernel
    GPU durations are NOT available from the profiler here (checked directly; see the report). What
    it does record is the HIP launch API itself — `hipLaunchKernel` for every elementwise/reduction
    kernel and `hipExtModuleLaunchKernel` for every hipBLASLt GEMM — which is exactly the quantity
    the "~1000 launch-bound kernels" claim is about. GPU TIME comes from the HIP-event timings in
    `hc_isolate` instead, which is the better instrument for it anyway.

    Under graph replay the host issues ONE `hipGraphLaunch` for the whole recorded sequence; the
    same kernels still run, as graph nodes, with no per-kernel host launch. Censusing the replay is
    how "capture does not remove kernels, it removes launches" stops being an assertion."""
    from torch.profiler import ProfilerActivity, profile

    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=False, with_stack=False) as p:
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()

    evs = list(p.events())
    by_api: dict = defaultdict(int)
    by_parent: dict = defaultdict(int)
    by_aten: dict = defaultdict(int)
    total = 0
    device_events = 0
    for e in evs:
        nm = e.name
        if str(getattr(e, "device_type", "")).endswith("CUDA"):
            device_events += 1
        if nm in _LAUNCH_APIS:
            total += 1
            by_api[nm] += 1
            par = getattr(e, "cpu_parent", None)
            pname = getattr(par, "name", None) or "<toplevel>"
            # climb to the outermost aten op so `aten::mm` bills to `aten::linear`
            seen = 0
            while par is not None and seen < 8:
                nxt = getattr(par, "cpu_parent", None)
                if nxt is None or not str(getattr(nxt, "name", "")).startswith("aten::"):
                    break
                par = nxt
                seen += 1
            by_parent[getattr(par, "name", pname)] += 1
        elif nm.startswith("aten::"):
            by_aten[nm] += 1

    def per_pass(d):
        return {k: round(v / reps, 3) for k, v in sorted(d.items(), key=lambda x: -x[1])}

    return {
        "reps": reps,
        "blocks_per_pass": passes,
        "profiler_device_events": device_events,   # 0 in this image — durations come from P2
        "kernel_launches_per_pass": round(total / reps, 2),
        "kernel_launches_per_block": round(total / reps / max(1, passes), 3),
        "launches_by_hip_api_per_pass": per_pass(by_api),
        "launches_by_aten_op_per_pass": per_pass(by_parent),
        "aten_calls_per_pass": dict(list(per_pass(by_aten).items())[:20]),
    }


# ───────────────────────────────────────────────────────────────────────────────── P4: ablation

def install_stub(llm) -> int:
    """Replace every hyper-connection mix/combine with a SHAPE-PRESERVING no-op.

    `mix` must still hand the block a (T, H) tensor and a residual pair; `combine` must still hand
    the model a (T, wide) stream. The stub keeps both, costs one narrow+copy per mix and ZERO
    kernels per combine, and destroys numerics — so this runs GREEDY and after the coherence
    sample. It is the only way to price the block INSIDE the engine, including its share of the
    inter-kernel gap, without an instrument that changes what it measures."""
    inner = llm.engine.model.model
    layers = inner.layers.op_list
    targets = []
    for L in layers:
        targets += [L.attn_hyper_connection, L.mlp_hyper_connection]
    targets.append(inner.hyper_connection_mixer)
    n = 0
    for m in targets:
        H = m.hidden_size

        def _mix(x, _H=H):
            return x[..., :_H].contiguous(), (x, x)

        def _combine(y, res):
            return res[0]

        m.mix = _mix
        if getattr(m, "_use_combine", False):
            m.combine = _combine
        n += 1
    return n


def restore_hc(llm) -> None:
    for m in _hc_all(llm):
        for attr in ("mix", "combine"):
            if attr in m.__dict__:
                del m.__dict__[attr]


def _hc_all(llm):
    inner = llm.engine.model.model
    out = []
    for L in inner.layers.op_list:
        out += [L.attn_hyper_connection, L.mlp_hyper_connection]
    out.append(inner.hyper_connection_mixer)
    return out


def recapture(llm) -> dict:
    """Re-record the engine's decode graphs against whatever the model currently computes.

    Needed because a graph captured at boot replays the kernels it RECORDED: stubbing HC in Python
    after capture changes the eager path only. Frees the old graphs first — the pool is only
    reclaimed when the graph objects die."""
    gr = llm.engine.graph_runner
    model = llm.engine.model
    old = getattr(gr, "graph_map", {})
    keys = sorted(old.keys())
    gr.graph_map = {}
    del old
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    t = time.perf_counter()
    gr._capture_graphs(gr._verify_max_seq_len, gr._verify_vocab, model)
    return {"seconds": round(time.perf_counter() - t, 1),
            "bs_before": keys, "bs_after": sorted(gr.graph_map.keys())}


# ─────────────────────────────────────────────────────────────────────────────────── the run

def rank_main(rank: int, tp: int, args, model_dir: str) -> dict:
    from minisgl.core import SamplingParams, get_global_ctx
    from minisgl.distributed import DistributedInfo
    from minisgl.llm import LLM

    out: dict = {"rank": rank, "tp": tp}

    def dump():
        if args.json:
            p = f"{args.json}.rank{rank}.partial"
            try:
                with open(p, "w") as fh:
                    json.dump(out, fh, indent=2)
            except BaseException:
                pass

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
    gr = llm.engine.graph_runner
    out["card"] = torch.cuda.get_device_name(dev)
    out["device_index"] = int(getattr(dev, "index", 0) or 0)
    out["cuda_graph_bs_captured"] = sorted(getattr(gr, "graph_map", {}).keys())
    out["kv_pages"] = int(llm.engine.num_pages)
    dump()

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

    def sp(n, greedy=False):
        if greedy:
            return SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=n)
        return SamplingParams(temperature=temperature, top_k=top_k, top_p=top_p,
                              ignore_eos=True, max_tokens=n)

    N, N0 = int(args.tokens), int(args.tokens_short)

    def timed_leg(graphs_on: bool, greedy: bool = False):
        """ms per DECODE step. wall(N) - wall(N0) cancels prefill and tokenizer; the decode panel
        and STEP_LOG measure a WALL that under capture returns on enqueue and are never used."""
        saved = gr.max_graph_bs
        try:
            if not graphs_on:
                gr.max_graph_bs = 0
            llm.generate([prompt], sp(4, greedy))
            t = time.perf_counter(); r0 = llm.generate([prompt], sp(N0, greedy))[0]
            dt0 = time.perf_counter() - t
            t = time.perf_counter(); r1 = llm.generate([prompt], sp(N, greedy))[0]
            dt1 = time.perf_counter() - t
        finally:
            gr.max_graph_bs = saved
        n0, n1 = len(r0["token_ids"]), len(r1["token_ids"])
        if n1 <= n0:
            return None
        return 1000.0 * (dt1 - dt0) / (n1 - n0)

    def replays():
        return int(getattr(gr, "replays", -1))

    # ---- P1: step totals ----------------------------------------------------------------------
    print(f"\n[P1] step totals, {args.repeats} interleaved repeats", flush=True)
    cap_ms, eag_ms = [], []
    r_before = replays()
    for _ in range(max(1, args.repeats)):
        m = timed_leg(True)
        if m: cap_ms.append(m)
        m = timed_leg(False)
        if m: eag_ms.append(m)
        print(f"   captured {cap_ms[-1]:.2f}  eager {eag_ms[-1]:.2f} ms/step", flush=True)
    out["P1"] = {
        "captured_samples": [round(x, 3) for x in cap_ms],
        "eager_samples": [round(x, 3) for x in eag_ms],
        "ms_per_step_captured": round(min(cap_ms), 3) if cap_ms else None,
        "ms_per_step_eager": round(min(eag_ms), 3) if eag_ms else None,
        "capture_saves_ms_per_step": (round(min(eag_ms) - min(cap_ms), 3)
                                      if cap_ms and eag_ms else None),
        "graph_replays_during_P1": replays() - r_before,
    }
    dump()

    # ---- coherence BEFORE the ablation destroys numerics ---------------------------------------
    r = llm.generate([prompt], SamplingParams(temperature=temperature, top_k=top_k, top_p=top_p,
                                              max_tokens=48))[0]
    out["sample_text"] = r.get("text", "")[:600]
    dump()

    # ---- P2/P3 --------------------------------------------------------------------------------
    print("\n[P2/P3] HC isolation eager vs captured + kernel census", flush=True)
    try:
        out["P2"] = hc_isolate(llm, args)
    except BaseException as ex:  # noqa: BLE001
        out["P2"] = {"error": f"{type(ex).__name__}: {ex}"[:1000],
                     "traceback": traceback.format_exc()[-2500:]}
    print("   " + json.dumps(out["P2"].get("eager_ms", {}))[:1200], flush=True)
    print("   " + json.dumps(out["P2"].get("captured_ms", {}))[:1200], flush=True)
    dump()

    # ---- P4a: in-situ event regions, eager (reproduces the ENDGAME 10.15 ms figure) ------------
    print("\n[P4a] in-situ event regions, eager", flush=True)
    prof = EvProf()
    inner = llm.engine.model.model
    for L in inner.layers.op_list:
        prof.wrap(L.attn_hyper_connection, "mix", "hc.mix")
        prof.wrap(L.attn_hyper_connection, "combine", "hc.combine")
        prof.wrap(L.mlp_hyper_connection, "mix", "hc.mix")
        prof.wrap(L.mlp_hyper_connection, "combine", "hc.combine")
    prof.wrap(inner.hyper_connection_mixer, "mix", "hc.final_mix")
    model = llm.engine.model
    prof.wrap(model, "forward", "MODEL_TOTAL")
    _mf = model.forward

    def stepping_forward(*a, **kw):
        is_pf = bool(getattr(getattr(get_global_ctx(), "batch", None), "is_prefill", False))
        prev = prof.enabled
        prof.enabled = prev and not is_pf
        try:
            return _mf(*a, **kw)
        finally:
            prof.enabled = prev
            if not is_pf:
                prof.step_end()

    model.forward = stepping_forward

    inst_off, inst_on = [], []
    saved = gr.max_graph_bs
    try:
        gr.max_graph_bs = 0
        prof.enabled = False
        for _ in range(max(1, args.prof_repeats)):
            m = None
            llm.generate([prompt], sp(4))
            t = time.perf_counter(); r0 = llm.generate([prompt], sp(N0))[0]; dt0 = time.perf_counter() - t
            t = time.perf_counter(); r1 = llm.generate([prompt], sp(N))[0]; dt1 = time.perf_counter() - t
            n0, n1 = len(r0["token_ids"]), len(r1["token_ids"])
            if n1 > n0:
                inst_off.append(1000.0 * (dt1 - dt0) / (n1 - n0))
        prof.enabled = True
        prof.reset()
        llm.generate([prompt], sp(4))
        prof.flush(); prof.reset()
        for _ in range(max(1, args.prof_repeats)):
            t = time.perf_counter(); r0 = llm.generate([prompt], sp(N0))[0]; dt0 = time.perf_counter() - t
            t = time.perf_counter(); r1 = llm.generate([prompt], sp(N))[0]; dt1 = time.perf_counter() - t
            n0, n1 = len(r0["token_ids"]), len(r1["token_ids"])
            if n1 > n0:
                inst_on.append(1000.0 * (dt1 - dt0) / (n1 - n0))
    except BaseException as ex:  # noqa: BLE001 — fenced so P4b/P4c still run
        out["P4a_error"] = f"{type(ex).__name__}: {ex}"[:600]
        out["P4a_traceback"] = traceback.format_exc()[-1500:]
    finally:
        gr.max_graph_bs = saved
        try:
            prof.flush()
        except BaseException:
            pass
        prof.enabled = False

    g = prof.per_step()
    hc_region = (g.get("hc.mix", 0.0) + g.get("hc.combine", 0.0) + g.get("hc.final_mix", 0.0))
    out["P4a"] = {
        "gpu_ms_per_step": g,
        "regions_per_step": prof.counts_per_step(),
        "hc_ms_per_step_event_regions": round(hc_region, 3),
        "ms_per_step_eager_wrapped_off": round(min(inst_off), 3) if inst_off else None,
        "ms_per_step_eager_wrapped_on": round(min(inst_on), 3) if inst_on else None,
        "instrument_overhead_ms_per_step": (round(min(inst_on) - min(inst_off), 3)
                                            if inst_on and inst_off else None),
        "prof_steps": prof.steps,
        "prof_event_overflow": prof.overflow,
    }
    # full restore: one instance attr held first the region shim, then stepping_forward
    if "forward" in model.__dict__:
        del model.__dict__["forward"]
    dump()

    # unwrap the event shims before the ablation, or the ablation times the instrument too
    for m in _hc_all(llm):
        for attr in ("mix", "combine"):
            if attr in m.__dict__:
                del m.__dict__[attr]

    # ---- P4b: in-situ ABLATION, EAGER (greedy on both legs; the sampler cancels) ---------------
    print("\n[P4b] ablation A/B, eager, greedy", flush=True)
    real, stub = [], []
    try:
        for _ in range(max(1, args.ablate_repeats)):
            restore_hc(llm)
            m = timed_leg(False, greedy=True)
            if m: real.append(m)
            install_stub(llm)
            m = timed_leg(False, greedy=True)
            if m: stub.append(m)
            restore_hc(llm)
            print(f"   real {real[-1]:.2f}  stub {stub[-1]:.2f} ms/step", flush=True)
        out["P4b_eager_ablation"] = {
            "real_samples": [round(x, 3) for x in real],
            "stub_samples": [round(x, 3) for x in stub],
            "ms_per_step_real": round(min(real), 3) if real else None,
            "ms_per_step_stub": round(min(stub), 3) if stub else None,
            "hc_ms_per_step_eager": (round(min(real) - min(stub), 3) if real and stub else None),
        }
    except BaseException as ex:  # noqa: BLE001
        out["P4b_eager_ablation"] = {"error": f"{type(ex).__name__}: {ex}"[:600],
                                     "traceback": traceback.format_exc()[-1500:]}
    restore_hc(llm)
    dump()

    # ---- P4c: in-situ ABLATION, CAPTURED (needs a recapture; last, and fenced) -----------------
    print("\n[P4c] ablation A/B, captured (recapture)", flush=True)
    res: dict = {}
    try:
        restore_hc(llm)
        res["recapture_real"] = recapture(llm)
        creal = [timed_leg(True, greedy=True) for _ in range(max(1, args.ablate_repeats))]
        creal = [x for x in creal if x]
        install_stub(llm)
        res["recapture_stub"] = recapture(llm)
        cstub = [timed_leg(True, greedy=True) for _ in range(max(1, args.ablate_repeats))]
        cstub = [x for x in cstub if x]
        restore_hc(llm)
        res.update({
            "real_samples": [round(x, 3) for x in creal],
            "stub_samples": [round(x, 3) for x in cstub],
            "ms_per_step_real": round(min(creal), 3) if creal else None,
            "ms_per_step_stub": round(min(cstub), 3) if cstub else None,
            "hc_ms_per_step_captured": (round(min(creal) - min(cstub), 3)
                                        if creal and cstub else None),
        })
    except BaseException as ex:  # noqa: BLE001
        res["error"] = f"{type(ex).__name__}: {ex}"[:800]
        res["traceback"] = traceback.format_exc()[-2000:]
    out["P4c_captured_ablation"] = res
    dump()

    # ---- byte floor: HC is bf16 and REPLICATED, so this is per-rank and identical on both ------
    cfg = json.load(open(os.path.join(model_dir, "config.json")))["text_config"]
    H, hcn, lr, L = (cfg["hidden_size"], cfg["hc_count"], cfg["hc_lowrank"],
                     cfg["num_hidden_layers"])
    wide = hcn * H
    hc_b = 2 * (wide + lr * wide + wide * lr + hcn * wide)
    hc_mixer = 2 * (wide + lr * wide + wide * lr)
    total_b = 2 * L * hc_b + hc_mixer
    out["hc_bytes_per_rank"] = total_b
    for name, ms in (("event_regions", out["P4a"]["hc_ms_per_step_event_regions"]),):
        if ms:
            out[f"hc_achieved_GBps_{name}"] = round(total_b / (ms * 1e-3) / 1e9, 2)
    out["hc_bandwidth_floor_ms_at_706GBps"] = round(total_b / 706.6e9 * 1e3, 3)
    dump()
    return out


def _spawn_target(rank, tp, args, model_dir, q):
    try:
        q.put(rank_main(rank, tp, args, model_dir))
    except BaseException as e:  # noqa: BLE001
        traceback.print_exc()
        q.put({"rank": rank, "tp": tp, "error": f"{type(e).__name__}: {e}"[:4000],
               "traceback": traceback.format_exc()[-4000:]})
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
    ap.add_argument("--ablate-repeats", type=int, default=3)
    ap.add_argument("--isolate-reps", type=int, default=20)
    ap.add_argument("--isolate-bs", type=int, default=1)
    ap.add_argument("--census-reps", type=int, default=3)
    ap.add_argument("--prompt", default="Explain in three sentences why the sky is blue.")
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
                           name=f"q4e-hcprize-{rank}")
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
    print(json.dumps(out, indent=2)[:30000], flush=True)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"wrote {args.json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
