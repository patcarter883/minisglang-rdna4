#!/usr/bin/env python3
"""Decompose ONE captured decode step of the 48-layer qwen4_exp offloaded serve so the parts SUM.

WHY A THIRD ATTRIBUTION
-----------------------
Two previous ones were destroyed by the next measurement (`docs/measurements/QWEN4EXP_ENDGAME.md`
§2-§3): the "decode is PCIe-bound / 41.72 ms of host-expert reads" premise was refuted by moving 36
of 37 layers off PCIe and getting a 2.28x SLOWER step, and the "32.9 ms gap" was mostly a 7.5 ms
Phase-0 GUESS for a compute floor that measures 26.7 ms. §3 closes with the operating point's true
ms/step recorded as NOT SETTLED: 62.36 (attribution boot) vs 26.5 forward-only + 79.7 end-to-end
(serve boot) never reconciled.

THE METHOD, and why it closes where the previous two did not
------------------------------------------------------------
The previous attributions summed REGIONS INSIDE `model.forward` and compared the sum to a WALL
number produced by a different instrument on a different boot. This one measures a PARTITION of the
wall clock first and only then subdivides it, so "what is left over" is a subtraction, not a guess.

Level 1 — a partition of the wall second, valid UNDER CAPTURE:

  * `Scheduler._hp` (MINISGL_HOSTPROF, already in the repo) splits the synchronous `normal_loop`
    into `recv / sched / fwd_launch / gpu_wait / commit` by host wall. Qwen4-Exp's PLE forces that
    synchronous loop (scheduler.py: `_stage_ple` hashes host token history before the forward), so
    every decode step really is: schedule -> enqueue -> block on `copy_done` -> commit. The five
    stages are disjoint and cover the loop; `wall - sum(stages)` is reported as `loop_residual`.
  * A HIP EVENT PAIR around `engine.forward_batch`, recorded on the engine stream, gives DEVICE BUSY
    time per step. This is the one instrument that is NOT blind under capture: it brackets the
    replay rather than living inside it. It is exact here because the loop is synchronous — the
    stream is drained by the previous step's `copy_done.synchronize()`, so the start event completes
    immediately and the elapsed time is this step's device execution, sampler and D2H included.
  * Everything is taken by the DIFFERENCE METHOD: wall(N tokens) - wall(N0 tokens), and the same
    subtraction on every accumulator. Prefill, tokenizer, warmup and `generate` setup cancel
    identically in the numerator and are not divided into the decode steps. (The 79.86 ms baseline
    §3 calls a measurement error was a wall figure divided by decode steps WITHOUT this subtraction.)

  => gpu_idle_ms = wall_ms_per_step - device_busy_ms is then a MEASURED number, and the host stages
     that can only run while the GPU is idle (recv + sched + commit) are a lower bound on it.

Level 2 — what is inside `device_busy`. HIP-event regions per component, on the EAGER leg (the same
`EvProf` + `install()` used by `prof_decode_attrib.py`, imported, not copied). The captured/eager
device-time ratio is measured in the SAME boot by the level-1 event pair on both legs, and the level-
2 numbers are reported BOTH raw-eager and scaled by that measured ratio, with the ratio printed.

Level 3 — cross-checks that do not share an instrument with level 2:
  * `isolate()` replays the HC blocks and the host/device expert stacks on live weights with one
    event pair per whole pass (per-boundary instrument cost divided by 97, not charged to each).
  * a SYNTHETIC all-reduce at the exact MoE hidden shape, timed with its own event pair.
  * the loop-stage `gpu_wait` vs the event-pair `device_busy`: two independent instruments for
    (almost) the same quantity.

THE MARGINAL HOST LAYER
-----------------------
`--device-gb` is the planner knob that moves whole MoE layers between the device tier and the pinned
host arena. Run this file at two or more values in separate boots and regress `wall_ms_per_step`
against `host_layers`: that is d(step)/d(host layer) with NO bandwidth assumption anywhere. The
`--json` output carries `host_layers` / `device_layers` / `host_bytes_per_layer_per_rank` so the
regression is arithmetic on the outputs.

TRAPS THIS FILE IS WRITTEN AGAINST (all previously paid for in this repo)
------------------------------------------------------------------------
* `STEP_LOG` / the decode panel measure the host wall of `_forward`; under capture that returns after
  enqueueing an ASYNC replay and reads 0.5 ms. Never quoted here. `fwd_launch` is labelled as enqueue.
* `ab_captured_steady_ms_per_step` read a 53.7x "speedup" for exactly that reason.
* The GPU lease is WAIVED for this task and a co-tenant once inflated a leg 8.6x: legs are
  INTERLEAVED and the summary statistic is the MINIMUM over repeats.
* Card 1's root port is Gen4 x8 (14.48 GB/s) vs card 0's Gen5 x8 (28.93). Both ranks are reported
  separately; a TP=2 step runs in lockstep so the slow rank gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback

import torch

sys.path.insert(0, "/engine/tests")
sys.path.insert(0, "/engine/tools/offload")
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(_here)), "tests"))

MODEL = os.environ.get("Q4E_MODEL", "/model")

HP_KEYS = ("recv", "sched", "fwd_launch", "gpu_wait", "commit")


# ─────────────────────────────────────────────────────────── device-busy event pair (capture-safe)

class StepEvents:
    """One HIP event pair per decode step, recorded on the engine stream around `forward_batch`.

    DEFERRED READ. `Event.synchronize()` per step would serialise host and device and destroy the
    very wall time being partitioned; the pool holds `cap` pairs and is drained once, at the end of
    a leg, after a single `torch.cuda.synchronize()`. Prefill steps are skipped by asking the live
    batch (a `generate` call is 8 prefill forwards then N-1 decode forwards here).
    """

    def __init__(self, cap: int = 4096) -> None:
        self.pool = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                     for _ in range(cap)]
        self.cap = cap
        self.i = 0
        self.pending: list = []
        self.total_ms = 0.0
        self.n = 0
        self.overflow = 0
        self.enabled = False

    def reset(self) -> None:
        self.pending.clear()
        self.i = 0
        self.total_ms = 0.0
        self.n = 0
        self.overflow = 0

    def drain(self) -> None:
        if self.pending:
            torch.cuda.synchronize()
            for s, e in self.pending:
                self.total_ms += s.elapsed_time(e)
                self.n += 1
            self.pending.clear()
        self.i = 0

    def install(self, engine) -> None:
        fb = engine.forward_batch
        stream = engine.stream
        from minisgl.core import get_global_ctx

        def shim(batch, *a, **kw):
            if not self.enabled or bool(getattr(batch, "is_prefill", False)):
                return fb(batch, *a, **kw)
            if self.i >= self.cap:
                self.overflow += 1
                return fb(batch, *a, **kw)
            s, e = self.pool[self.i]
            self.i += 1
            s.record(stream)
            try:
                return fb(batch, *a, **kw)
            finally:
                e.record(stream)
                self.pending.append((s, e))

        engine.forward_batch = shim
        _ = get_global_ctx  # imported for symmetry with prof_decode_attrib's gate; unused here


# ──────────────────────────────────────────────────────────── host sub-stage profiler (no syncs)

class HostProf:
    """Wall-clock accumulator over named HOST call sites, installed on instances.

    `Scheduler._hp` partitions the loop into five stages; when one of them dominates, the next
    question is which call inside it. This subdivides `sched` and `commit` by wrapping the methods
    `_finish_prepare` actually calls. It is PURE HOST timing — no `Event`, no `synchronize` — so it
    is safe on the captured leg and cannot perturb the device timeline. Its own Python cost is
    measured, not assumed: the same leg is run with the wrappers off and the difference reported.

    Nesting is real here (`sched` contains `_finish_prepare` contains `stage_ple`), so the reporting
    side owns the hierarchy; this side only accumulates, exactly like `EvProf`.
    """

    def __init__(self) -> None:
        self.acc: dict = {}
        self.cnt: dict = {}
        self.enabled = False
        self.sites: list = []

    def wrap(self, obj, attr: str, label: str) -> bool:
        fn = getattr(obj, attr, None)
        if fn is None or not callable(fn):
            return False

        def shim(*a, **kw):
            if not self.enabled:
                return fn(*a, **kw)
            t = time.perf_counter()
            try:
                return fn(*a, **kw)
            finally:
                dt = time.perf_counter() - t
                self.acc[label] = self.acc.get(label, 0.0) + dt
                self.cnt[label] = self.cnt.get(label, 0) + 1

        try:
            setattr(obj, attr, shim)
        except Exception:
            return False
        self.sites.append(label)
        return True

    def snapshot(self) -> tuple:
        return dict(self.acc), dict(self.cnt)


def install_host_sites(llm, hp: HostProf) -> list:
    """The call sites `sched` and `commit` are actually made of. Missing ones are REPORTED, not
    silently skipped — a site that moved would otherwise read as a zero cost."""
    cand = [
        (llm, "_schedule_next_batch", "sched._schedule_next_batch"),
        (llm, "_prepare_batch", "sched._prepare_batch"),
        (llm, "_finish_prepare", "sched._finish_prepare"),
        (llm, "_stage_ple", "sched._stage_ple"),
        (llm, "_build_grammar_bitmask", "sched._grammar_bitmask"),
        (llm.cache_manager, "allocate_paged", "sched.allocate_paged"),
        (llm.engine.attn_backend, "prepare_metadata", "sched.prepare_metadata"),
        (llm.engine.sampler, "prepare", "sched.sampler_prepare"),
        (llm.decode_manager, "schedule_next_batch", "sched.decode_schedule"),
        # NESTED, and named so: `_process_last_data` is `gpu_wait` + `commit` together, so it is a
        # CHECK on those two loop stages, not a third one to add to them.
        (llm, "_process_last_data", "nested.process_last_data=gpu_wait+commit"),
        (llm, "send_result", "commit.send_result"),
        (llm, "_flush_stats", "recv._flush_stats"),
        (llm, "receive_msg", "recv.receive_msg"),
    ]
    ple = getattr(llm, "_ple", None)
    if ple is not None:
        cand.append((ple, "commit_staged", "fwd.ple_commit_staged"))
    gs = getattr(llm, "gdn_slots", None)
    if gs is not None:
        cand.append((gs, "state_indices", "sched.gdn_state_indices"))
    missing = []
    for obj, attr, label in cand:
        if not hp.wrap(obj, attr, label):
            missing.append(label)
    return missing


# ─────────────────────────────────────────────────────────────────────────────── the measured leg

def _hp_snapshot(llm) -> dict:
    d = getattr(llm, "_hp", None)
    return dict(d) if d else {}


def _hp_diff(a: dict, b: dict) -> dict:
    return {k: b.get(k, 0.0) - a.get(k, 0.0) for k in set(a) | set(b)}


def measure_leg(llm, gr, prompt, sp, N0, N, graphs_on, ev: StepEvents | None, reps: int,
                hp=None) -> dict:
    """ms per DECODE step for wall AND every accumulator, by the N-vs-N0 difference method.

    Returns the MINIMUM-wall repeat's full record: the co-tenant risk on an unleased box is a
    one-sided inflation, so min is the honest summary and the spread is reported next to it.
    """
    saved = gr.max_graph_bs
    recs = []
    try:
        if not graphs_on:
            gr.max_graph_bs = 0
        llm.generate([prompt], sp(4))                     # warm THIS leg's dispatch path
        for _ in range(max(1, reps)):
            if ev is not None:
                ev.reset(); ev.enabled = True
            h0 = _hp_snapshot(llm)
            s0 = hp.snapshot() if hp is not None else None
            t = time.perf_counter(); r0 = llm.generate([prompt], sp(N0))[0]
            dt0 = time.perf_counter() - t
            h1 = _hp_snapshot(llm)
            s1 = hp.snapshot() if hp is not None else None
            if ev is not None:
                # CUMULATIVE on purpose — do NOT reset here. The per-step figure below is a
                # DIFFERENCE of two running totals, exactly like the wall and the loop stages, so a
                # reset between the short and long generate would double-subtract the short leg.
                ev.drain()
                dev0_ms, dev0_n = ev.total_ms, ev.n
            t = time.perf_counter(); r1 = llm.generate([prompt], sp(N))[0]
            dt1 = time.perf_counter() - t
            h2 = _hp_snapshot(llm)
            s2 = hp.snapshot() if hp is not None else None
            if ev is not None:
                ev.drain()
                dev1_ms, dev1_n = ev.total_ms, ev.n
                ev.enabled = False
            n0, n1 = len(r0["token_ids"]), len(r1["token_ids"])
            if n1 <= n0:
                continue
            dn = n1 - n0
            rec = {
                "tokens_short": n0, "tokens_long": n1, "decode_steps_delta": dn,
                "wall_ms_per_step": 1000.0 * (dt1 - dt0) / dn,
                # RECORDED PER LEG, not once per run. The GPU lease is waived for this task and a
                # CPU-only co-tenant workflow shares the 16 cores; every host-side stage below
                # (`sched`, `commit`) is a Python thread competing for one of them, so a leg run
                # under load 28 is not comparable with one run under load 2 and the reader must be
                # able to see which they are holding.
                "loadavg_1m": float(open("/proc/loadavg").read().split()[0]),
            }
            d0, d1 = _hp_diff(h0, h1), _hp_diff(h1, h2)
            stages = {}
            for k in HP_KEYS:
                stages[k] = 1000.0 * (d1.get(k, 0.0) - d0.get(k, 0.0)) / dn
            rec["stage_ms_per_step"] = stages
            rec["stage_sum_ms_per_step"] = sum(stages.values())
            rec["loop_residual_ms_per_step"] = rec["wall_ms_per_step"] - rec["stage_sum_ms_per_step"]
            if hp is not None:
                # SAME difference method as the loop stages: the short generate's prefill sched cost
                # (a chunked prompt is 8 forwards whose `_finish_prepare` is a different shape and
                # cost from a decode step's) cancels instead of being averaged into a per-decode
                # figure. Call counts are differenced too, so the divisor is visible.
                a0, c0 = s0; a1, c1 = s1; a2, c2 = s2
                keys = set(a0) | set(a1) | set(a2)
                rec["site_ms_per_step"] = {
                    k: round(1000.0 * ((a2.get(k, 0.0) - a1.get(k, 0.0))
                                       - (a1.get(k, 0.0) - a0.get(k, 0.0))) / dn, 4)
                    for k in keys}
                rec["site_calls_per_step"] = {
                    k: round(((c2.get(k, 0) - c1.get(k, 0)) - (c1.get(k, 0) - c0.get(k, 0))) / dn, 3)
                    for k in keys}
            if ev is not None:
                # Decode forwards are COUNTED, not assumed. `dev1_n - dev0_n` must equal `dn` (the
                # token delta) on a bs=1 run; both are emitted so a mismatch is visible rather than
                # averaged away — an unequal-token-count leg is one of the retracted speedups this
                # repo has already paid for.
                if (dev1_n - dev0_n) > 0:
                    rec["device_busy_ms_per_step"] = (dev1_ms - dev0_ms) / (dev1_n - dev0_n)
                    rec["device_event_steps"] = [dev0_n, dev1_n]
                    rec["device_event_steps_match_token_delta"] = bool((dev1_n - dev0_n) == dn)
                if dev0_n:
                    # the SHORT leg standing alone: if the per-step device time agrees with the
                    # difference figure, no fixed per-generate device cost is hiding in either.
                    rec["device_busy_ms_per_step_shortleg"] = dev0_ms / dev0_n
            recs.append(rec)
    finally:
        gr.max_graph_bs = saved
        if ev is not None:
            ev.enabled = False
    if not recs:
        return {"error": "no usable repeat (n1 <= n0 on every one)"}
    best = min(recs, key=lambda r: r["wall_ms_per_step"])
    best["wall_samples_ms"] = [round(r["wall_ms_per_step"], 3) for r in recs]
    best["repeats"] = len(recs)
    return best


# ─────────────────────────────────────────────────────────────────────────────── cross-check: AR

def synthetic_all_reduce(llm, reps: int = 200) -> dict:
    """Time the MoE all-reduce shape on its own, with its own event pair.

    The level-2 region profiler charges an event PAIR to each of 48 per-step all-reduces; this pays
    two boundaries for `reps` calls, so the instrument cost is divided by `reps` instead of charged
    per call. Cross-checks `moe.all_reduce` without sharing its instrument.
    """
    o: dict = {}
    try:
        layers = llm.engine.model.model.layers.op_list
        comm = layers[0].mlp.experts._comm
        H = int(layers[0].attn_hyper_connection.hidden_size)
    except Exception as ex:
        return {"error": f"{type(ex).__name__}: {ex}"[:300]}
    dev = llm.engine.device
    x = torch.randn(1, H, dtype=torch.bfloat16, device=dev)
    o["shape"] = [1, int(H)]
    o["bytes"] = int(x.numel() * x.element_size())
    try:
        for _ in range(5):
            comm.all_reduce(x)
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(reps):
            comm.all_reduce(x)
        e.record()
        torch.cuda.synchronize()
        o["ms_per_call"] = round(s.elapsed_time(e) / reps, 5)
        o["reps"] = reps
        o["ms_per_step_48_calls"] = round(48 * o["ms_per_call"], 4)
    except Exception as ex:
        o["error"] = f"{type(ex).__name__}: {ex}"[:400]
    return o


# ────────────────────────────────────────────────────────────── cross-check: in-boot layer ablation

def ablation_sweep(llm, gr, prompt, sp, N0, N, census, args) -> dict:
    """d(step)/d(layer) measured WITHOUT a reboot, by removing the routed-expert call entirely.

    WHY THIS EXISTS BESIDE THE CROSS-BOOT `--device-gb` SWEEP. The planner knob is the honest lever
    — it is the thing a serve would actually change — but each point costs a ~13-minute boot and
    ~1.33 GiB/node of extra pinned arena per layer moved off the device, so on a box whose
    MemAvailable is contended the lever arm is a handful of layers against a 1.7% run-to-run spread.
    This one has an arbitrary lever arm (0..36 layers) inside ONE boot.

    WHAT IS ACTUALLY REMOVED: `MoELayer.forward` for the chosen layers, i.e. the top-k gather, the
    grouped NVFP4 expert GEMMs, and — for a HOST layer — the pinned-arena dereference that IS the
    PCIe transfer (there is no separate H2D copy; see QWEN4EXP_ENDGAME.md §2). NOT removed: the
    router GEMV (`mlp.gate`, called outside `experts.forward`), the shared expert, or the fused
    all-reduce (`rowchunked_ar_span` in the sparse block). So the slope is the routed-expert term
    ONLY, and `host_slope - device_slope` is precisely what moving one layer between tiers changes.

    THE OUTPUT IS GARBAGE AND THAT IS FINE — `ignore_eos` fixes the token count, and the difference
    method divides by the number of decode steps actually run, so a degenerate continuation costs
    nothing but its own tokens. It is a TIMING probe and is never quality-checked.

    EAGER ONLY. A captured graph replays the recorded kernels; the stub would be invisible to it.
    The captured/eager device ratio measured in level 1 is reported next to the slope rather than
    silently applied.
    """
    o: dict = {"note": "eager; routed-expert call stubbed to zeros; router/shared/all-reduce kept"}
    layers = llm.engine.model.model.layers.op_list
    orig = {i: layers[i].mlp.experts.forward for i in range(len(layers))}

    def stub(hidden_states, router_logits=None, reduce=False, **kw):
        return torch.zeros_like(hidden_states)

    def leg() -> float | None:
        r = measure_leg(llm, gr, prompt, sp, N0, N, False, None, args.ablate_repeats)
        return r.get("wall_ms_per_step")

    def run(idxs, k):
        for i in range(len(layers)):
            layers[i].mlp.experts.forward = orig[i]
        for i in idxs[:k]:
            layers[i].mlp.experts.forward = stub
        try:
            return leg()
        finally:
            for i in range(len(layers)):
                layers[i].mlp.experts.forward = orig[i]

    try:
        nh, nd = len(census["host"]), len(census["device"])
        hk = sorted({0, nh // 4, nh // 2, (3 * nh) // 4, nh})
        dk = sorted({0, max(1, nd // 2), nd})
        o["host_points"] = []
        for k in hk:
            ms = run(census["host"], k)
            o["host_points"].append({"ablated": k, "ms_per_step": round(ms, 3) if ms else None})
            print(f"    ablate {k:2d} HOST layers -> {ms:.3f} ms/step", flush=True)
        o["device_points"] = []
        for k in dk:
            ms = run(census["device"], k)
            o["device_points"].append({"ablated": k, "ms_per_step": round(ms, 3) if ms else None})
            print(f"    ablate {k:2d} DEV  layers -> {ms:.3f} ms/step", flush=True)

        def slope(pts):
            xs = [(p["ablated"], p["ms_per_step"]) for p in pts if p["ms_per_step"]]
            if len(xs) < 2:
                return None
            n = len(xs)
            mx = sum(x for x, _ in xs) / n
            my = sum(y for _, y in xs) / n
            num = sum((x - mx) * (y - my) for x, y in xs)
            den = sum((x - mx) ** 2 for x, _ in xs)
            return -num / den if den else None      # +ve = ms SAVED per layer removed

        o["ms_per_host_layer_eager"] = round(slope(o["host_points"]), 4) \
            if slope(o["host_points"]) else None
        o["ms_per_device_layer_eager"] = round(slope(o["device_points"]), 4) \
            if slope(o["device_points"]) else None
        if o["ms_per_host_layer_eager"] and o["ms_per_device_layer_eager"]:
            o["host_premium_ms_per_layer_eager"] = round(
                o["ms_per_host_layer_eager"] - o["ms_per_device_layer_eager"], 4)
    except BaseException as ex:  # noqa: BLE001
        o["error"] = f"{type(ex).__name__}: {ex}"[:600]
        o["traceback"] = traceback.format_exc()[-1200:]
        for i in range(len(layers)):
            layers[i].mlp.experts.forward = orig[i]
    return o


# ────────────────────────────────────────────────────────────────────────────────────── the run

def rank_main(rank: int, tp: int, args, model_dir: str) -> dict:
    from minisgl.core import SamplingParams
    from minisgl.distributed import DistributedInfo
    from minisgl.llm import LLM

    out: dict = {"rank": rank, "tp": tp, "mode": args.mode}

    # kernel provenance BEFORE the boot: the whole measurement is void if a stale /opt/kernels
    # fp8_wmma shadowed the freshly built e4m3 one.
    try:
        import fp8_wmma
        p = os.path.dirname(fp8_wmma.__file__)
        so = [f for f in os.listdir(p) if f.startswith("fp8_wmma_C") and f.endswith(".so")]
        out["fp8_wmma_path"] = p
        if so:
            h = hashlib.sha256(open(os.path.join(p, so[0]), "rb").read()).hexdigest()
            out["fp8_wmma_so"] = so[0]
            out["fp8_wmma_so_sha256"] = h
    except Exception as ex:
        out["fp8_wmma_error"] = f"{type(ex).__name__}: {ex}"[:300]
    out["kernels_ref"] = os.environ.get("KERNELS_REF", "")
    out["pythonpath"] = os.environ.get("PYTHONPATH", "")
    out["arena_chunk_mib"] = os.environ.get("MINISGL_WEIGHT_ARENA_CHUNK_MIB", "")
    out["hostprof_env"] = os.environ.get("MINISGL_HOSTPROF", "")

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
    out["graph_capture_engaged"] = bool(out["cuda_graph_bs_captured"])
    out["kv_pages"] = int(llm.engine.num_pages)
    out["memory_ratio"] = args.memory_ratio
    out["hostprof_active"] = getattr(llm, "_hp", None) is not None
    assert out["hostprof_active"], "MINISGL_HOSTPROF unset -> no loop-stage partition; run is void"

    # placement census, and the per-layer byte figure the bandwidth reconciliation divides by
    from minisgl.weights.stacks import StackKind
    layers = llm.engine.model.model.layers.op_list
    census = {"device": [], "host": [], "cpu": [], "unseamed": []}
    for i, L in enumerate(layers):
        seam = getattr(L.mlp.experts, "_weight_offload", None)
        if seam is None:
            census["unseamed"].append(i)
        elif seam.kind is StackKind.HOST:
            census["host"].append(i)
        elif seam.kind is StackKind.CPU:
            census["cpu"].append(i)
        else:
            census["device"].append(i)
    out["host_layers"] = len(census["host"])
    out["device_layers"] = len(census["device"])
    out["cpu_layers"] = len(census["cpu"])
    out["unseamed_layers"] = census["unseamed"]
    out["host_layer_indices"] = census["host"]
    out["device_layer_indices"] = census["device"]
    woff = llm.engine._woff
    try:
        # `plan` is a property of the DRIVER (`StageARuntime`), not of the session that owns it —
        # `StageASession` is the window/ordering object. Both spellings are tried and the one that
        # answered is recorded, because a silently-missing plan is how the byte model behind every
        # bandwidth figure below would become a blank instead of an error.
        plan = getattr(woff, "plan", None)
        out["plan_source"] = "session.plan"
        if plan is None:
            plan = woff.driver.plan
            out["plan_source"] = "session.driver.plan"
        out["plan_host_bytes"] = int(plan.host_resident_bytes)
        out["plan_device_bytes"] = int(plan.device_resident_bytes)
        if out["host_layers"]:
            out["host_bytes_per_layer_per_rank"] = out["plan_host_bytes"] // out["host_layers"]
        out["host_active_bytes_per_token_per_rank"] = int(plan.host_active_bytes(1))
        out["device_active_bytes_per_token_per_rank"] = int(plan.device_active_bytes(1))
    except Exception as ex:
        out["plan_error"] = f"{type(ex).__name__}: {ex}"[:300]

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
    ev = StepEvents()
    ev.install(llm.engine)

    # ---- LEVEL 1: the wall partition, captured and eager, INTERLEAVED ------------------------
    print(f"\n[L1] loop partition + device-busy events, {N}-vs-{N0} difference, "
          f"{args.repeats} interleaved repeats", flush=True)
    cap_l, eag_l = [], []
    for i in range(max(1, args.repeats)):
        cap_l.append(measure_leg(llm, gr, prompt, sp, N0, N, True, ev, 1))
        eag_l.append(measure_leg(llm, gr, prompt, sp, N0, N, False, ev, 1))
        print(f"   rep {i+1}: captured {cap_l[-1].get('wall_ms_per_step', float('nan')):.2f} "
              f"(dev {cap_l[-1].get('device_busy_ms_per_step', float('nan')):.2f}) | eager "
              f"{eag_l[-1].get('wall_ms_per_step', float('nan')):.2f} "
              f"(dev {eag_l[-1].get('device_busy_ms_per_step', float('nan')):.2f}) ms/step",
              flush=True)

    def pick(legs):
        ok = [x for x in legs if "wall_ms_per_step" in x]
        if not ok:
            return {"error": "every repeat failed", "raw": legs[:2]}
        b = min(ok, key=lambda r: r["wall_ms_per_step"])
        b["wall_samples_ms"] = [round(r["wall_ms_per_step"], 3) for r in ok]
        b["repeats"] = len(ok)
        return b

    out["captured"] = pick(cap_l)
    out["eager"] = pick(eag_l)
    c, e = out["captured"], out["eager"]
    if "wall_ms_per_step" in c and "wall_ms_per_step" in e:
        out["capture_factor_wall"] = round(e["wall_ms_per_step"] / c["wall_ms_per_step"], 4)
    if "device_busy_ms_per_step" in c and "device_busy_ms_per_step" in e:
        out["capture_factor_device"] = round(
            e["device_busy_ms_per_step"] / c["device_busy_ms_per_step"], 4)
    if "device_busy_ms_per_step" in c:
        c["gpu_idle_ms_per_step"] = c["wall_ms_per_step"] - c["device_busy_ms_per_step"]
        c["gpu_busy_fraction"] = c["device_busy_ms_per_step"] / c["wall_ms_per_step"]
        s = c["stage_ms_per_step"]
        # recv+sched+commit CANNOT overlap the device in the synchronous loop (the stream is drained
        # by gpu_wait before commit runs, and nothing is enqueued until fwd_launch). Lower bound.
        c["host_only_stages_ms"] = s["recv"] + s["sched"] + s["commit"]
    if "device_busy_ms_per_step" in e:
        e["gpu_idle_ms_per_step"] = e["wall_ms_per_step"] - e["device_busy_ms_per_step"]
        e["gpu_busy_fraction"] = e["device_busy_ms_per_step"] / e["wall_ms_per_step"]

    if args.mode == "quick":
        r = llm.generate([prompt], SamplingParams(temperature=temperature, top_k=top_k,
                                                  top_p=top_p, max_tokens=32))[0]
        out["sample_text"] = r.get("text", "")[:400]
        return out

    # ---- LEVEL 1b: what `sched` and `commit` are made of, CAPTURED (pure host timing) ---------
    print("\n[L1b] host sub-stages inside sched/commit, captured leg", flush=True)
    hp = HostProf()
    out["host_sites_missing"] = install_host_sites(llm, hp)
    sub = {}
    for on in (False, True):
        hp.enabled = on
        r = measure_leg(llm, gr, prompt, sp, N0, N, True, None, args.repeats, hp=hp if on else None)
        sub["wall_wrapped_on" if on else "wall_wrapped_off"] = r.get("wall_ms_per_step")
        if on:
            sites = r.get("site_ms_per_step", {})
            sub["site_ms_per_step"] = dict(sorted(sites.items(), key=lambda x: -x[1]))
            sub["site_calls_per_step"] = r.get("site_calls_per_step", {})
            sub["stage_ms_per_step"] = r.get("stage_ms_per_step")
    hp.enabled = False
    if sub.get("wall_wrapped_on") and sub.get("wall_wrapped_off"):
        sub["wrapper_overhead_ms_per_step"] = round(
            sub["wall_wrapped_on"] - sub["wall_wrapped_off"], 4)
    out["host_substages"] = sub
    print("   " + json.dumps(sub)[:2000], flush=True)

    # ---- LEVEL 3z: in-boot ablation sweep (BEFORE any wrapper Python is in the loop) ----------
    print("\n[L3z] in-boot ablation: d(step)/d(layer) with the routed-expert call removed",
          flush=True)
    out["ablation"] = ablation_sweep(llm, gr, prompt, sp, N0, N, census, args)
    print("   " + json.dumps({k: v for k, v in out["ablation"].items()
                              if k != "traceback"})[:1200], flush=True)

    # ---- LEVEL 3a: isolation replay (BEFORE any wrapper Python is in the loop) ----------------
    print("\n[L3a] isolation replay: HC blocks, expert stacks, live weights", flush=True)
    try:
        from prof_decode_attrib import isolate
        out["isolate"] = isolate(llm, args)
    except BaseException as ex:  # noqa: BLE001
        out["isolate"] = {"error": f"{type(ex).__name__}: {ex}"[:800],
                          "traceback": traceback.format_exc()[-1500:]}
    print("   " + json.dumps(out["isolate"])[:1500], flush=True)

    print("\n[L3b] synthetic all-reduce at the MoE hidden shape", flush=True)
    out["synthetic_all_reduce"] = synthetic_all_reduce(llm, args.ar_reps)
    print("   " + json.dumps(out["synthetic_all_reduce"]), flush=True)

    # ---- LEVEL 2: per-component device regions, eager -----------------------------------------
    from prof_decode_attrib import EvProf, install, byte_model
    prof = EvProf()
    install(llm, prof)
    # sampler + the whole device step, so MODEL_TOTAL's complement is visible rather than assumed
    prof.wrap(llm.engine.sampler, "sample", "sampler")

    from minisgl.core import get_global_ctx
    model = llm.engine.model
    _mf = model.forward
    step_flush = {"decode": 0, "prefill": 0}

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

    print(f"\n[L2] instrumented eager regions, {args.prof_repeats} repeats", flush=True)
    inst_off, inst_on = [], []
    for enabled, sink in ((False, inst_off), (True, inst_on)):
        prof.enabled = enabled
        prof.reset()
        saved = gr.max_graph_bs
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
                    sink.append(1000.0 * (dt1 - dt0) / (n1 - n0))
        finally:
            gr.max_graph_bs = saved
            prof.flush()
            prof.enabled = False
    out["eager_wrapped_off_ms_per_step"] = round(min(inst_off), 3) if inst_off else None
    out["eager_wrapped_on_ms_per_step"] = round(min(inst_on), 3) if inst_on else None
    if inst_off and inst_on:
        out["instrument_overhead_ms_per_step"] = round(min(inst_on) - min(inst_off), 3)
    out["prof_steps"] = prof.steps
    out["prof_event_overflow"] = prof.overflow
    out["forwards_seen"] = dict(step_flush)
    out["region_ms_per_step_eager"] = prof.per_step()
    out["regions_per_step"] = prof.counts_per_step()

    # scale the eager regions onto the captured step by the MEASURED device-time ratio
    k = out.get("capture_factor_device")
    if k:
        out["region_ms_per_step_captured_scaled"] = {
            kk: round(v / k, 4) for kk, v in out["region_ms_per_step_eager"].items()}
        out["region_scale_note"] = (
            f"eager device regions divided by the measured eager/captured device-busy ratio {k}")

    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    cfg = cfg.get("text_config", cfg)
    try:
        bm = byte_model(cfg, tp)
        out["byte_model_per_rank"] = {kk: v for kk, v in bm.items() if not kk.startswith("_")}
        out["byte_model_meta"] = bm["_meta"]
    except Exception as ex:
        out["byte_model_error"] = f"{type(ex).__name__}: {ex}"[:300]

    g = out["region_ms_per_step_eager"]
    nh, nd = out["host_layers"], out["device_layers"]
    if nh and nd and "moe.routed.host" in g and "moe.routed.dev" in g:
        hp = g["moe.routed.host"] / nh
        dp = g["moe.routed.dev"] / nd
        out["moe_routed_ms_per_host_layer_eager"] = round(hp, 4)
        out["moe_routed_ms_per_device_layer_eager"] = round(dp, 4)
        out["moe_routed_host_penalty_ms_per_step_eager"] = round((hp - dp) * nh, 3)
        b = out.get("host_active_bytes_per_token_per_rank", 0) / max(nh, 1)
        if b:
            out["in_situ_host_expert_GBps"] = round(b / (hp * 1e-3) / 1e9, 3)
            out["in_situ_host_expert_GBps_penalty_only"] = (
                round(b / ((hp - dp) * 1e-3) / 1e9, 3) if hp > dp else None)

    r = llm.generate([prompt], SamplingParams(temperature=temperature, top_k=top_k, top_p=top_p,
                                              max_tokens=48))[0]
    out["sample_text"] = r.get("text", "")[:600]
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
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--tokens-short", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--prof-repeats", type=int, default=2)
    ap.add_argument("--isolate-reps", type=int, default=20)
    ap.add_argument("--ar-reps", type=int, default=300)
    ap.add_argument("--ablate-repeats", type=int, default=2)
    ap.add_argument("--mode", default="full", choices=["full", "quick"])
    ap.add_argument("--prompt", default="Explain in three sentences why the sky is blue.")
    ap.add_argument("--expert-bytes-per-rank", type=float, default=1.3952e6)
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
    print(f"[subset] {model_dir} ({args.layers}L, {args.experts}E, tp={args.tp}, "
          f"device_gb={args.device_gb})", flush=True)

    if args.tp == 1:
        results = [rank_main(0, 1, args, model_dir)]
    else:
        import multiprocessing as mp
        mp.set_start_method("spawn", force=True)
        q = mp.Queue()
        procs = []
        for rank in range(args.tp):
            p = mp.Process(target=_spawn_target, args=(rank, args.tp, args, model_dir, q),
                           name=f"q4e-decomp-{rank}")
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
        codes = [p.exitcode for p in procs]
        print(f"\n[parent] exit codes {codes}", flush=True)
        if any(c != 0 for c in codes):
            print("[parent] WARNING: a rank exited non-zero — the run is NOT a measurement",
                  flush=True)

    results.sort(key=lambda r: r.get("rank", 0))
    out = {"tp": args.tp, "layers": args.layers, "device_gb": args.device_gb,
           "cards": [r.get("card") for r in results],
           "so_sha256": [r.get("fp8_wmma_so_sha256") for r in results],
           "ranks": results}
    print(json.dumps(out, indent=2)[:24000], flush=True)
    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"wrote {args.json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
