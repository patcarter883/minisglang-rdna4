#!/usr/bin/env python3
"""GPU time of the hyper-connection stack, OLD code vs NEW, eager and CAPTURED — synthetic replicate.

WHY SYNTHETIC, AND WHY THAT IS LEGITIMATE HERE
-----------------------------------------------
Every hyper-connection block in this checkpoint is the SAME shape, is bf16, and is TP-REPLICATED, so
a stack of `--blocks` blocks built from the real config with random weights has the identical working
set and the identical kernel sequence as the served one. The previous round measured both: the live
81-block stack inside a 40-layer boot and an independent synthetic 97-block replicate, and they
agreed to 0.1% (7.53 vs 7.55 ms/step). This file is that replicate, doubled into an A/B.

It is a SEPARATE process from the serve harness on purpose. Capturing an ~900-node hipGraph inside a
boot that has 0.56 GiB of VRAM left SIGSEGVs both ranks in this image (observed twice, 2026-09-05,
exit -11 right after P0). With no model resident there is ~15 GiB free and the capture is routine.

WHAT IT DOES NOT MEASURE: the seam. A block's cost inside the engine includes its share of the
inter-kernel gap and of whatever else is in flight; an isolated replay does not. Under CAPTURE there
is no host in the loop, so the isolated replay is a faithful model of the block's graph-node cost —
but the end-to-end verdict is the boot's step total (`hc_fusion_ab.py` P3), never this number alone.
A microbenchmark of a component cannot price a seam; that lesson cost this project 2.3x once.

TIMING DISCIPLINE: one HIP-event pair around the WHOLE rep loop, `synchronize()` AFTER the closing
record. Never around a single async `replay()` — that trap produced a x50.66 "speedup" on this box.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import textwrap
from types import MethodType

import torch
import torch.nn.functional as F


def load_old(baseline_json: str) -> dict:
    from minisgl.layers.norm import _rms_norm

    payload = json.load(open(baseline_json))

    def extract(src, cls, fns):
        lines = src.splitlines(keepends=True)
        found = {}
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.ClassDef) and node.name == cls:
                for sub in node.body:
                    if isinstance(sub, ast.FunctionDef) and sub.name in fns:
                        found[sub.name] = textwrap.dedent(
                            "".join(lines[sub.lineno - 1 : sub.end_lineno]))
        missing = [f for f in fns if f not in found]
        if missing:
            raise AssertionError(f"{payload['base_sha']} has no {cls}.{missing}")
        return found

    hc = extract(payload["files"]["python/minisgl/layers/hyperconnection.py"],
                 "HyperConnection", ("mix", "combine"))
    nm = extract(payload["files"]["python/minisgl/layers/norm.py"],
                 "GroupedRMSNorm", ("forward",))
    marks = {"old_mix_divides_at_runtime": "/ self._hc" in hc["mix"],
             "old_combine_divides_at_runtime": "/ self._hc" in hc["combine"],
             "old_norm_adds_one_per_call": "self.weight + 1.0" in nm["forward"]}
    if not all(marks.values()):
        raise AssertionError(f"baseline blobs are not the pre-fusion code: {marks}")
    ns = {"torch": torch, "F": F, "_rms_norm": _rms_norm}
    fut = "from __future__ import annotations\n"
    for src in list(hc.values()) + [nm["forward"]]:
        exec(compile(fut + src, payload["base_sha"], "exec"), ns)
    return {"mix": ns["mix"], "combine": ns["combine"], "norm_forward": ns["forward"],
            "base_sha": payload["base_sha"], "marks": marks}


def ev_time(fn, reps: int) -> float:
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="/model/config.json")
    ap.add_argument("--baseline-json",
                    default="/engine/docs/measurements/HC_FUSION_2026-09-05/baseline_source.json")
    ap.add_argument("--blocks", type=int, default=97)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    from minisgl.distributed import set_tp_info, try_get_tp_info
    if try_get_tp_info() is None:
        set_tp_info(0, 1)
    from minisgl.layers import HyperConnection

    cfg = json.load(open(args.config))["text_config"]
    H, HC, LR = cfg["hidden_size"], cfg["hc_count"], cfg["hc_lowrank"]
    eps = float(cfg.get("rms_norm_eps", 1e-6))
    W = HC * H
    dev = "cuda"
    torch.manual_seed(7)

    # `--blocks` real-shaped blocks: 2 per layer plus the final mixer. The last one is built with
    # use_combine=False, exactly as `hyper_connection_mixer` is.
    blocks = []
    torch.set_default_dtype(torch.bfloat16)
    for i in range(args.blocks):
        blocks.append(HyperConnection(hidden_size=H, hc_count=HC, hc_lowrank=LR, eps=eps,
                                      use_combine=(i < args.blocks - 1)))
    torch.set_default_dtype(torch.float32)
    for b in blocks:
        b.hc_norm.weight = (torch.randn(W) * 0.05).to(dev, torch.bfloat16)
        b.input_mix_weight_down.weight = (torch.randn(LR, W) * W ** -0.5).to(dev, torch.bfloat16)
        b.input_mix_weight_up.weight = (torch.randn(W, LR) * LR ** -0.5).to(dev, torch.bfloat16)
        if b._use_combine:
            b.block_inject_weight.weight = (torch.randn(HC, W) * W ** -0.5).to(dev, torch.bfloat16)

    bytes_per_block = 2 * (W + LR * W + W * LR + HC * W)
    x = torch.randn(args.bs, W, device=dev, dtype=torch.bfloat16)
    y = torch.randn(args.bs, H, device=dev, dtype=torch.bfloat16)
    old = load_old(args.baseline_json)

    out = {"blocks": args.blocks, "bs": args.bs, "reps": args.reps, "rounds": args.rounds,
           "base_sha": old["base_sha"],
           "working_set_MB": round(bytes_per_block * args.blocks / 1e6, 1),
           "mall_MB": 64,
           "config": {"hidden": H, "hc_count": HC, "hc_lowrank": LR, "wide": W}}

    def a_pass():
        for b in blocks:
            m, r = b.mix(x)
            if b._use_combine:
                b.combine(y, r)

    def install_old():
        for b in blocks:
            b.mix = MethodType(old["mix"], b)
            if b._use_combine:
                b.combine = MethodType(old["combine"], b)
            b.hc_norm.forward = MethodType(old["norm_forward"], b.hc_norm)
            if getattr(b, "_scale_folded", False):
                b._fused_w.mul_(b._hc)      # exact: hc is a power of two
        torch.cuda.synchronize()

    def restore_new():
        for b in blocks:
            for obj, attr in ((b, "mix"), (b, "combine"), (b.hc_norm, "forward")):
                if attr in obj.__dict__:
                    del obj.__dict__[attr]
            if getattr(b, "_scale_folded", False):
                b._fused_w.div_(b._hc)
        torch.cuda.synchronize()

    # ---- OLD leg is the un-finalized layer: no post_load, runtime divides, separate GEMVs -------
    install_old()
    old_outs = [(b.mix(x)[0].clone()) for b in blocks]
    restore_new()                      # deletes the shims; _scale_folded is still False here
    for b in blocks:
        b.post_load()
    new_outs = [(b.mix(x)[0].clone()) for b in blocks]
    torch.cuda.synchronize()
    out["mix_bit_identical"] = all(torch.equal(a, b) for a, b in zip(old_outs, new_outs))
    out["mix_max_abs_diff"] = max(float((a.float() - b.float()).abs().max())
                                  for a, b in zip(old_outs, new_outs))
    del old_outs, new_outs

    # ---- eager + captured, both legs, interleaved rounds, min-of-N -----------------------------
    legs = {"new": [], "old": []}
    graphs = {}
    pool = None
    for rnd in range(max(1, args.rounds)):
        for leg in ("new", "old"):
            if leg == "old":
                install_old()
            rec = {"eager_ms": round(ev_time(a_pass, args.reps), 4)}
            key = f"{leg}"
            if key not in graphs:
                graphs[key] = make_graph(a_pass, pool=pool)
                if pool is None:
                    pool = graphs[key].pool()
            rec["captured_ms"] = round(ev_time(graphs[key].replay, args.reps), 4)
            legs[leg].append(rec)
            if leg == "old":
                restore_new()
            print(f"  round{rnd} {leg}: {rec}", flush=True)

    # A captured graph of the NEW leg must reproduce the NEW eager result exactly, or the captured
    # number is timing a different computation.
    try:
        ref = [b.mix(x)[0].clone() for b in blocks]
        got = [None] * len(blocks)

        def mix_into():
            for i, b in enumerate(blocks):
                got[i] = b.mix(x)[0]

        gg = make_graph(mix_into)      # its OWN pool: a shared pool may alias another graph's buffers
        gg.replay()
        torch.cuda.synchronize()
        out["capture_vs_eager_max_abs_diff"] = max(
            float((a - b).abs().max()) for a, b in zip(ref, got))
        del gg, ref, got
    except BaseException as ex:  # noqa: BLE001
        out["capture_vs_eager_error"] = f"{type(ex).__name__}: {ex}"[:300]

    def best(leg, key):
        return round(min(r[key] for r in legs[leg]), 4)

    scale = 97.0 / args.blocks
    out["legs"] = legs
    out["summary"] = {
        "_units": f"ms per pass over {args.blocks} blocks; *_at_97 scales to the 48-layer serve",
        "eager_ms_old": best("old", "eager_ms"), "eager_ms_new": best("new", "eager_ms"),
        "captured_ms_old": best("old", "captured_ms"),
        "captured_ms_new": best("new", "captured_ms"),
    }
    s = out["summary"]
    s["captured_saved_ms"] = round(s["captured_ms_old"] - s["captured_ms_new"], 4)
    s["captured_saved_pct"] = round(
        100.0 * (1 - s["captured_ms_new"] / s["captured_ms_old"]), 2)
    s["eager_saved_ms"] = round(s["eager_ms_old"] - s["eager_ms_new"], 4)
    for k in ("eager_ms_old", "eager_ms_new", "captured_ms_old", "captured_ms_new",
              "captured_saved_ms"):
        s[f"{k}_at_97"] = round(s[k] * scale, 4)
    s["hc_bandwidth_floor_ms_at_706GBps_at_97"] = round(
        bytes_per_block * 97 / 706.6e9 * 1e3, 4)

    print(json.dumps(out, indent=2), flush=True)
    if args.json:
        os.makedirs(os.path.dirname(args.json), exist_ok=True)
        json.dump(out, open(args.json, "w"), indent=2)
        print(f"wrote {args.json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
