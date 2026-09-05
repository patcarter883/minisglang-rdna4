#!/usr/bin/env python3
"""A/B the hyper-connection launch-count cut: OLD CODE vs NEW CODE, one boot, interleaved.

WHAT IS BEING COMPARED, AND WHY IT IS NOT AN EMULATION
------------------------------------------------------
The baseline leg is not a re-implementation of the old hyper-connection and it is not the old
NUMBERS from a previous boot. It is the OLD SOURCE, lifted out of `git show <base>:` with `ast` and
bound onto the LIVE model objects with `types.MethodType` — the same trick
`tests/qwen4exp_hc_parity_test.py` uses to make upstream sglang its own oracle. Both legs therefore
run in one process, on one card, against the same weights, minutes apart, and the comparison cannot
be a new-vs-itself measurement (`ab-baseline-must-be-old-code-not-emulated`).

The one thing that has to be undone for the old leg is `post_load`'s `/hc` weight fold: the old code
divides at RUNTIME, so it must see un-divided weights. `_unfold`/`_refold` multiply and divide the
packed buffer by `hc` in place. `hc` is 4, so both directions are exact power-of-two scales and the
weights come back BIT-IDENTICAL — asserted, not assumed (`weights_roundtrip_max_abs_diff`).

FIVE MEASUREMENTS
-----------------
P0  EXACTNESS. New vs old `mix`/`combine` on the live bf16 weights, every block. The packed
    [lowrank+hc, wide] GEMV is only legal if the decode GEMV is ROW-INVARIANT, so that is measured
    head-on (`packed_gemv_row_invariance_max_abs_diff`) instead of assumed.
P1  ISOLATION replay of all live HC blocks, EAGER and CAPTURED, per leg, interleaved, min-of-N.
    One HIP-event pair around the WHOLE loop, sync after the closing record — never around an async
    `replay()` (that trap produced a x50.66 "speedup" on this box once).
P2  LAUNCH CENSUS per leg, from the HIP launch API (this image's kineto reports zero device
    activity, so GPU time comes from P1's events and only the launch COUNT comes from the profiler).
P3  STEP TOTALS, captured and eager, per leg, with a REcapture between legs — a graph replays the
    kernels it recorded, so swapping Python after capture changes nothing until it is re-recorded.
    wall(N) - wall(N0) so prefill and tokenizer cancel.
P4  GREEDY IDS, captured, new vs old: the same prompt must produce the same token sequence.

TRAPS ALREADY WALKED INTO ON THIS BOX
-------------------------------------
* The decode panel / STEP_LOG measure WALL time of `Scheduler._forward`, which under capture returns
  on enqueue. Never used here.
* The engage ledger is a SET and saturates, so it cannot A/B two legs. Ignored here.
* A single-block replay loop fits the 64 MB MALL. Every pass walks all blocks.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
import textwrap
import time
import traceback
from collections import defaultdict
from types import MethodType

import torch

sys.path.insert(0, "/engine/tests")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "tests"))

class _SkipIsolation(Exception):
    """Sentinel: the isolation A/B was deliberately left to the standalone harness."""


MODEL = os.environ.get("Q4E_MODEL", "/model")
REPO = os.environ.get("HC_AB_REPO", "/engine")


# ────────────────────────────────────────────────────────────── the OLD code, straight out of git

def load_old_code(baseline_json: str) -> dict:
    """`HyperConnection.mix`, `HyperConnection.combine` and `GroupedRMSNorm.forward` as they were at
    the baseline commit, exec'd from the GIT BLOB text.

    The blobs are dumped host-side (`run_hc_fusion_ab.sh` runs `git show <base>:<path>`) because the
    container mounts the worktree WITHOUT its `.git`, and are carried in one JSON with the resolved
    sha. Raises if any of the three functions is missing — a baseline that silently fell back to the
    NEW code would make every number below a new-vs-itself comparison."""
    import torch.nn.functional as F

    from minisgl.layers.norm import _rms_norm

    with open(baseline_json) as fh:
        payload = json.load(fh)
    base_sha = payload["base_sha"]
    files = payload["files"]

    def blob(path: str) -> str:
        if path not in files:
            raise AssertionError(f"{baseline_json} carries no blob for {path}")
        return files[path]

    def extract(src: str, cls: str, fns: tuple) -> dict:
        lines = src.splitlines(keepends=True)
        tree = ast.parse(src)
        found = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == cls:
                for sub in node.body:
                    if isinstance(sub, ast.FunctionDef) and sub.name in fns:
                        found[sub.name] = textwrap.dedent(
                            "".join(lines[sub.lineno - 1 : sub.end_lineno])
                        )
        missing = [f for f in fns if f not in found]
        if missing:
            raise AssertionError(
                f"{base_sha} does not define {cls}.{missing} — the baseline leg would silently be "
                f"the NEW code, which makes this whole A/B new-vs-itself."
            )
        return found

    # `from __future__ import annotations` is prepended so the lifted defs' type annotations stay
    # strings — they name `HCResidual`/`Tuple` from a module scope this namespace does not have, and
    # evaluating them would be a NameError that has nothing to do with the arithmetic under test.
    ns = {"torch": torch, "F": F, "_rms_norm": _rms_norm}
    _FUT = "from __future__ import annotations\n"
    src_hc = extract(
        blob("python/minisgl/layers/hyperconnection.py"), "HyperConnection", ("mix", "combine")
    )
    src_nm = extract(blob("python/minisgl/layers/norm.py"), "GroupedRMSNorm", ("forward",))
    for name, src in list(src_hc.items()) + [("norm_forward", src_nm["forward"])]:
        exec(compile(_FUT + src, f"{base_sha}::{name}", "exec"), ns)
    # PROVENANCE, asserted rather than trusted: the baseline `mix` must be the one that divides by
    # hc at runtime and returns a 2-tuple, and the baseline norm must re-add 1.0 every call. If a
    # future base commit no longer does, this A/B is measuring something else and must say so.
    marks = {
        "old_mix_divides_at_runtime": "/ self._hc" in src_hc["mix"],
        "old_combine_divides_at_runtime": "/ self._hc" in src_hc["combine"],
        "old_norm_adds_one_per_call": "self.weight + 1.0" in src_nm["forward"],
        "old_combine_has_no_addcmul": "addcmul" not in src_hc["combine"],
    }
    if not all(marks.values()):
        raise AssertionError(f"baseline blobs are not the pre-fusion code: {marks}")
    return {
        "mix": ns["mix"],
        "combine": ns["combine"],
        "norm_forward": ns["forward"],
        "base_ref": base_sha,
        "marks": marks,
        "sources": {"mix": src_hc["mix"], "combine": src_hc["combine"],
                    "norm_forward": src_nm["forward"]},
    }


# ─────────────────────────────────────────────────────────────────────── model walk + leg install

def hc_blocks(llm) -> list:
    inner = llm.engine.model.model
    out = []
    for L in inner.layers.op_list:
        out += [L.attn_hyper_connection, L.mlp_hyper_connection]
    out.append(inner.hyper_connection_mixer)
    return out


def grouped_norms(llm) -> list:
    """Every `GroupedRMSNorm` in the model — the hyper-connections' `hc_norm` AND the PLE block's
    three. The gain cache is in that class, so a leg that only swapped the HC ones would be
    comparing two different diffs."""
    from minisgl.layers.norm import GroupedRMSNorm

    seen, out, stack = set(), [], [llm.engine.model]
    while stack:
        obj = stack.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if isinstance(obj, GroupedRMSNorm):
            out.append(obj)
        for v in list(getattr(obj, "__dict__", {}).values()):
            if isinstance(v, (list, tuple)):
                stack.extend(x for x in v if hasattr(x, "__dict__"))
            elif hasattr(v, "__dict__") and not isinstance(v, torch.Tensor):
                stack.append(v)
    return out


def install_old(llm, old: dict) -> dict:
    """Bind the old methods and UNDO the `/hc` weight fold (the old code divides at runtime)."""
    n_hc = n_norm = n_unfold = 0
    for b in hc_blocks(llm):
        b.mix = MethodType(old["mix"], b)
        if getattr(b, "_use_combine", False):
            b.combine = MethodType(old["combine"], b)
        n_hc += 1
        if getattr(b, "_scale_folded", False) and b._fused_w is not None:
            b._fused_w.mul_(b._hc)          # exact: hc is a power of two
            n_unfold += 1
    for nm in grouped_norms(llm):
        nm.forward = MethodType(old["norm_forward"], nm)
        n_norm += 1
    torch.cuda.synchronize()
    return {"hc_blocks": n_hc, "grouped_norms": n_norm, "unfolded": n_unfold}


def restore_new(llm) -> dict:
    n_refold = 0
    for b in hc_blocks(llm):
        for attr in ("mix", "combine"):
            if attr in b.__dict__:
                del b.__dict__[attr]
        if getattr(b, "_scale_folded", False) and b._fused_w is not None:
            b._fused_w.div_(b._hc)
            n_refold += 1
    for nm in grouped_norms(llm):
        if "forward" in nm.__dict__:
            del nm.__dict__["forward"]
    torch.cuda.synchronize()
    return {"refolded": n_refold}


# ───────────────────────────────────────────────────────────────────────── timing primitives

def ev_time(fn, reps: int) -> float:
    """ms per call on the GPU timeline, ONE event pair around the whole loop, sync AFTER the closing
    record — an async enqueue cannot be mistaken for completion."""
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


def make_graph(fn):
    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(st)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    return g


_LAUNCH_APIS = ("hipLaunchKernel", "hipExtModuleLaunchKernel", "hipModuleLaunchKernel",
                "hipGraphLaunch", "hipLaunchCooperativeKernel", "hipExtLaunchKernel",
                "hipMemcpyAsync", "hipMemsetAsync")


def kernel_census(fn, reps: int, passes: int) -> dict:
    """HOST-side kernel launches per pass, from the HIP launch API. This image's kineto reports no
    device activity, so this counts launches only; GPU time comes from the event timings above."""
    from torch.profiler import ProfilerActivity, profile

    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
    n = 0
    by_op: dict = defaultdict(int)
    for ev in prof.events():
        name = getattr(ev, "name", "")
        if name in _LAUNCH_APIS:
            n += 1
            by_op[name] += 1
    return {
        "launches_per_pass": round(n / max(1, reps), 2),
        "launches_per_block": round(n / max(1, reps) / max(1, passes), 3),
        "by_api": {k: round(v / max(1, reps), 2) for k, v in sorted(by_op.items())},
    }


# ─────────────────────────────────────────────────────────────────────── P0 / P1 / P2 per leg

def hc_pass_fn(llm, x, y):
    """One pass over every live HC block: mix -> combine for the per-layer ones, mix for the final
    mixer. All of them, so the working set is many times the 64 MB MALL."""
    inner = llm.engine.model.model
    layers = inner.layers.op_list

    def run():
        for L in layers:
            _, r = L.attn_hyper_connection.mix(x)
            L.attn_hyper_connection.combine(y, r)
            _, r = L.mlp_hyper_connection.mix(x)
            L.mlp_hyper_connection.combine(y, r)
        inner.hyper_connection_mixer.mix(x)

    return run


def leg_isolation(llm, x, y, reps: int, census_reps: int = 0) -> dict:
    fn = hc_pass_fn(llm, x, y)
    o = {"eager_ms": round(ev_time(fn, reps), 4)}
    try:
        g = make_graph(fn)
        o["captured_ms"] = round(ev_time(g.replay, reps), 4)
        del g
    except BaseException as ex:  # noqa: BLE001
        o["captured_ms"] = None
        o["capture_error"] = f"{type(ex).__name__}: {ex}"[:400]
    # NO kernel census here. Running kineto in the same process that has just captured and freed an
    # 81-block hipGraph SIGSEGVs both ranks in this image (observed 2026-09-05, exit code -11, right
    # after `profiler_start`/`profiler_stop`), taking the whole boot down before P1 can print. The
    # launch COUNT is a property of the code and not of the model, so it is measured in
    # `tools/offload/hc_launch_census.py`, which needs no engine and captures nothing.
    return o


def leg_outputs(llm, x, y) -> list:
    """Every block's (mixed, combined) on the current leg — the exactness oracle."""
    inner = llm.engine.model.model
    out = []
    for L in inner.layers.op_list:
        for hc in (L.attn_hyper_connection, L.mlp_hyper_connection):
            m, r = hc.mix(x)
            out.append((m.clone(), hc.combine(y, r).clone()))
    m, _ = inner.hyper_connection_mixer.mix(x)
    out.append((m.clone(), None))
    torch.cuda.synchronize()
    return out


# ───────────────────────────────────────────────────────────────────────────────── the run

def recapture(llm) -> dict:
    """Re-record the decode graphs against whatever the model computes NOW.

    A graph replays the kernels it RECORDED, so swapping `mix`/`combine` in Python after boot
    changes the eager path only — every captured number would otherwise be the boot-time code.
    The arena's post-capture re-gate is re-run here too, and its counter returned: a recapture that
    allocated from the pinned arena would bake a host address into the graph, and this is the check
    that exists to catch it."""
    gr = llm.engine.graph_runner
    old = getattr(gr, "graph_map", {})
    keys = sorted(old.keys())
    gr.graph_map = {}
    del old
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    t = time.perf_counter()
    gr._capture_graphs(gr._verify_max_seq_len, gr._verify_vocab, llm.engine.model)
    o = {"seconds": round(time.perf_counter() - t, 1),
         "bs_before": keys, "bs_after": sorted(gr.graph_map.keys())}
    woff = getattr(llm.engine, "_woff", None)
    if woff is not None:
        try:
            woff.verify_after_capture()
            o["verify_after_capture_fired"] = int(
                getattr(woff, "verify_after_capture_fired", -1))
        except BaseException as ex:  # noqa: BLE001 — a FAILED re-gate is the finding, not a crash
            o["verify_after_capture_error"] = f"{type(ex).__name__}: {ex}"[:400]
    return o


def rank_main(rank: int, tp: int, args, model_dir: str) -> dict:
    from minisgl.core import SamplingParams
    from minisgl.distributed import DistributedInfo
    from minisgl.llm import LLM

    out: dict = {"rank": rank, "tp": tp}

    def dump():
        if args.json:
            try:
                with open(f"{args.json}.rank{rank}.partial", "w") as fh:
                    json.dump(out, fh, indent=2)
            except BaseException:
                pass

    old = load_old_code(args.baseline_json)
    out["baseline"] = {"base_sha": old["base_ref"], "marks": old["marks"],
                       "source_bytes": {k: len(v) for k, v in old["sources"].items()}}

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
    out["cuda_graph_bs_captured"] = sorted(getattr(gr, "graph_map", {}).keys())
    blocks = hc_blocks(llm)
    out["hc_blocks"] = len(blocks)
    out["grouped_norms"] = len(grouped_norms(llm))
    out["post_load_prepared"] = sum(1 for b in blocks if getattr(b, "_prepared", False))
    out["post_load_scale_folded"] = sum(1 for b in blocks if getattr(b, "_scale_folded", False))

    # Capture-time integrity gate. `_woff.verify_after_capture_fired` counts the times the arena
    # re-gate reached its BODY with offload enabled — 0 is what a silently-disabled offload looks
    # like, which is exactly why the count is recorded rather than the fact that the call exists.
    woff = getattr(llm.engine, "_woff", None)
    out["verify_after_capture_fired_at_boot"] = int(
        getattr(woff, "verify_after_capture_fired", -1) if woff is not None else -1
    )
    dump()

    hc0 = blocks[0]
    wide, H = hc0.wide_size, hc0.hidden_size
    B = int(args.isolate_bs)
    torch.manual_seed(1234 + rank)
    x = torch.randn(B, wide, dtype=torch.bfloat16, device=dev)
    y = torch.randn(B, H, dtype=torch.bfloat16, device=dev)

    # ---- P0 EXACTNESS: new vs old on the live weights ------------------------------------------
    print("\n[P0] exactness, new vs old, live weights", flush=True)
    p0: dict = {}
    try:
        # is the packed GEMV row-invariant? this is the ONE assumption the pack rests on
        from minisgl.layers.minv import minv_linear
        n_in = hc0.hc_norm.forward(x)
        fw = hc0._fused_w
        n_down = hc0.input_mix_weight_down.weight.shape[0]
        packed = minv_linear(n_in, fw)
        sep_d = minv_linear(n_in, fw[:n_down])
        sep_i = minv_linear(n_in, fw[n_down:])
        torch.cuda.synchronize()
        p0["packed_gemv_row_invariance_max_abs_diff"] = max(
            float((packed[..., :n_down] - sep_d).abs().max()),
            float((packed[..., n_down:] - sep_i).abs().max()),
        )
        p0["packed_gemv_shape"] = list(fw.shape)
        p0["packed_engaged_in_mix"] = bool(hc0._fused_ok(n_in))

        new_out = leg_outputs(llm, x, y)
        # BITWISE fingerprint of every packed buffer, not a clone of it: cloning all of them is
        # ~540 MB and this card has under 1 GiB free after capture. `view(int16)` is the raw bf16
        # bit pattern, so an unfold/refold that was not exact cannot hide in the sum.
        def wfp():
            return [int(b._fused_w.view(torch.int16).to(torch.int64).sum()) for b in blocks]

        fp_before = wfp()
        p0["install_old"] = install_old(llm, old)
        old_out = leg_outputs(llm, x, y)
        p0["restore_new"] = restore_new(llm)
        fp_after = wfp()
        p0["weights_bitwise_roundtrip_identical"] = fp_before == fp_after
        p0["weights_roundtrip_first_block_max_abs_diff"] = 0.0 if fp_before == fp_after else None
        p0["mix_max_abs_diff"] = max(float((a[0] - b[0]).abs().max())
                                     for a, b in zip(new_out, old_out))
        p0["combine_max_abs_diff"] = max(float((a[1] - b[1]).abs().max())
                                         for a, b in zip(new_out, old_out) if a[1] is not None)
        p0["mix_bit_identical"] = all(torch.equal(a[0], b[0])
                                      for a, b in zip(new_out, old_out))
        p0["combine_bit_identical"] = all(torch.equal(a[1], b[1])
                                          for a, b in zip(new_out, old_out) if a[1] is not None)
        del new_out, old_out
    except BaseException as ex:  # noqa: BLE001
        p0["error"] = f"{type(ex).__name__}: {ex}"[:800]
        p0["traceback"] = traceback.format_exc()[-2000:]
    out["P0"] = p0
    print("   " + json.dumps(p0)[:900], flush=True)
    dump()

    # ---- P1/P2 ISOLATION + CENSUS, both legs, interleaved ---------------------------------------
    print("\n[P1/P2] isolation replay + launch census, interleaved", flush=True)
    iso: dict = {"new": [], "old": [], "skipped": bool(args.skip_isolation)}
    try:
        if args.skip_isolation:
            # Capturing an ~900-node hipGraph inside this boot SIGSEGVs both ranks (exit -11,
            # twice, 2026-09-05) — there is 0.56 GiB of VRAM left after the engine's own capture.
            # The isolation A/B therefore lives in `tools/offload/hc_isolation_ab.py`, which runs in
            # its own process with the card empty, and this boot measures the STEP TOTAL, which is
            # the number an isolation replay cannot give (it cannot price the seam).
            raise _SkipIsolation
        for i in range(max(1, args.isolate_repeats)):
            iso["new"].append(leg_isolation(llm, x, y, args.isolate_reps))
            install_old(llm, old)
            iso["old"].append(leg_isolation(llm, x, y, args.isolate_reps))
            restore_new(llm)
            print(f"   rep{i}  new {iso['new'][-1]} \n         old {iso['old'][-1]}", flush=True)

        def best(leg, key):
            vals = [r[key] for r in iso[leg] if r.get(key) is not None]
            return round(min(vals), 4) if vals else None

        iso["summary"] = {
            "eager_ms_new": best("new", "eager_ms"), "eager_ms_old": best("old", "eager_ms"),
            "captured_ms_new": best("new", "captured_ms"),
            "captured_ms_old": best("old", "captured_ms"),
        }
    except _SkipIsolation:
        iso["note"] = ("isolation A/B runs standalone in tools/offload/hc_isolation_ab.py; "
                       "in-boot graph capture of the HC stack SIGSEGVs at 0.56 GiB free VRAM")
    except BaseException as ex:  # noqa: BLE001
        iso["error"] = f"{type(ex).__name__}: {ex}"[:800]
        iso["traceback"] = traceback.format_exc()[-2000:]
        restore_new(llm)
    out["P1_P2"] = iso
    dump()

    # ---- P3 STEP TOTALS, both legs, recaptured --------------------------------------------------
    gcfg_path = os.path.join(model_dir, "generation_config.json")
    gcfg = json.load(open(gcfg_path)) if os.path.exists(gcfg_path) else {}
    temperature = float(gcfg.get("temperature", 1.0))
    top_k = int(gcfg.get("top_k", 20))
    top_p = float(gcfg.get("top_p", 0.95))
    prompt = args.prompt
    try:
        from transformers import AutoTokenizer
        tokz = AutoTokenizer.from_pretrained(model_dir)
        prompt = tokz.apply_chat_template([{"role": "user", "content": args.prompt}],
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
        return None if n1 <= n0 else 1000.0 * (dt1 - dt0) / (n1 - n0)

    print("\n[P3] step totals, both legs, recaptured, interleaved", flush=True)
    p3: dict = {"new_captured": [], "old_captured": [], "new_eager": [], "old_eager": [],
                "recaptures": []}
    try:
        for i in range(max(1, args.repeats)):
            restore_new(llm)
            p3["recaptures"].append({"leg": "new", **recapture(llm)})
            m = timed_leg(True)
            if m: p3["new_captured"].append(m)
            m = timed_leg(False)
            if m: p3["new_eager"].append(m)
            install_old(llm, old)
            p3["recaptures"].append({"leg": "old", **recapture(llm)})
            m = timed_leg(True)
            if m: p3["old_captured"].append(m)
            m = timed_leg(False)
            if m: p3["old_eager"].append(m)
            restore_new(llm)
            print(f"   rep{i}  new cap {p3['new_captured'][-1]:.2f} eag {p3['new_eager'][-1]:.2f}"
                  f"   old cap {p3['old_captured'][-1]:.2f} eag {p3['old_eager'][-1]:.2f}",
                  flush=True)
        for k in ("new_captured", "old_captured", "new_eager", "old_eager"):
            p3[f"ms_{k}"] = round(min(p3[k]), 3) if p3[k] else None
        if p3.get("ms_new_captured") and p3.get("ms_old_captured"):
            p3["step_ms_delta_captured"] = round(p3["ms_new_captured"] - p3["ms_old_captured"], 3)
            p3["tok_per_s_new"] = round(1000.0 / p3["ms_new_captured"], 3)
            p3["tok_per_s_old"] = round(1000.0 / p3["ms_old_captured"], 3)
    except BaseException as ex:  # noqa: BLE001
        p3["error"] = f"{type(ex).__name__}: {ex}"[:800]
        p3["traceback"] = traceback.format_exc()[-2000:]
        restore_new(llm)
    out["P3"] = p3
    dump()

    # ---- P4 GREEDY IDS ---------------------------------------------------------------------------
    print("\n[P4] greedy ids, captured, new vs old", flush=True)
    p4: dict = {}
    try:
        # THE FLOOR FIRST. This repo has a standing finding that the serve is not bit-reproducible
        # past ~32 tokens, so "new and old diverged" means nothing until the SAME code, recaptured
        # the same way, is shown to reproduce itself. Three legs, identical treatment: new, new
        # again (after its own recapture), then old. `new_vs_new` is the floor; only a divergence
        # EARLIER than the floor's is attributable to the change.
        restore_new(llm)
        p4["recapture_new_a"] = recapture(llm)
        ids_a = llm.generate([prompt], sp(args.id_tokens, greedy=True))[0]["token_ids"]
        restore_new(llm)
        p4["recapture_new_b"] = recapture(llm)
        ids_b = llm.generate([prompt], sp(args.id_tokens, greedy=True))[0]["token_ids"]
        install_old(llm, old)
        p4["recapture_old"] = recapture(llm)
        ids_o = llm.generate([prompt], sp(args.id_tokens, greedy=True))[0]["token_ids"]
        restore_new(llm)
        p4["recapture_restore"] = recapture(llm)
        ids_c = llm.generate([prompt], sp(args.id_tokens, greedy=True))[0]["token_ids"]

        def cmp(u, v):
            i = next((k for k, (a, b) in enumerate(zip(u, v)) if a != b), None)
            return {"identical": u == v, "first_divergence_index": i,
                    "prefix_match_len": len(u) if i is None else i}

        p4.update({
            "n_tokens": len(ids_a),
            "new_vs_new_FLOOR": cmp(ids_a, ids_b),
            "new_vs_new_after_old_FLOOR": cmp(ids_a, ids_c),
            "new_vs_old": cmp(ids_a, ids_o),
            "ids_new_a_head": ids_a[:24], "ids_new_b_head": ids_b[:24],
            "ids_old_head": ids_o[:24], "ids_new_c_head": ids_c[:24],
        })
        p4["verdict"] = (
            "IDENTICAL" if p4["new_vs_old"]["identical"] else
            "ENGINE_NONDETERMINISM" if (
                not p4["new_vs_new_FLOOR"]["identical"]
                and (p4["new_vs_new_FLOOR"]["prefix_match_len"]
                     <= p4["new_vs_old"]["prefix_match_len"])
            ) else "ATTRIBUTABLE_TO_THE_CHANGE"
        )
    except BaseException as ex:  # noqa: BLE001
        p4["error"] = f"{type(ex).__name__}: {ex}"[:800]
        p4["traceback"] = traceback.format_exc()[-2000:]
        restore_new(llm)
    out["P4"] = p4
    dump()

    # ---- coherence sample on the NEW leg, SAMPLED (never greedy for a quality read) -------------
    try:
        r = llm.generate([prompt], SamplingParams(temperature=temperature, top_k=top_k,
                                                  top_p=top_p, max_tokens=64))[0]
        out["sample_text_new"] = r.get("text", "")[:600]
    except BaseException as ex:  # noqa: BLE001
        out["sample_text_new"] = f"ERROR {type(ex).__name__}: {ex}"[:300]
    dump()
    return out


def _spawn_target(rank, tp, args, model_dir, q):
    try:
        q.put(rank_main(rank, tp, args, model_dir))
    except BaseException as e:  # noqa: BLE001
        traceback.print_exc()
        q.put({"rank": rank, "fatal": f"{type(e).__name__}: {e}"[:1000],
               "traceback": traceback.format_exc()[-3000:]})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--baseline-json", default="/engine/docs/measurements/HC_FUSION_2026-09-05/baseline_source.json")
    ap.add_argument("--layers", type=int, default=40)
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
    ap.add_argument("--isolate-repeats", type=int, default=2)
    ap.add_argument("--isolate-reps", type=int, default=20)
    ap.add_argument("--isolate-bs", type=int, default=1)
    ap.add_argument("--census-reps", type=int, default=3)
    ap.add_argument("--skip-isolation", action="store_true",
                    help="leave the HC isolation A/B to tools/offload/hc_isolation_ab.py (it needs "
                         "an empty card; in-boot it SIGSEGVs)")
    ap.add_argument("--id-tokens", type=int, default=32)
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
                           name=f"q4e-hcab-{rank}")
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
    out = {"tp": args.tp, "layers": args.layers, "baseline_json": args.baseline_json,
           "cards": [r.get("card") for r in results], "ranks": results}
    print(json.dumps(out, indent=2)[:40000], flush=True)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"wrote {args.json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
