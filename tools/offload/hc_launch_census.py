#!/usr/bin/env python3
"""How many kernel launches does ONE hyper-connection block issue — old code vs new?

NO ENGINE, NO CAPTURE, ON PURPOSE. Two reasons:

  * The launch count is a property of the CODE, not of the checkpoint. A block built from the real
    config with random bf16 weights issues exactly the launches the served one issues, and the 97
    blocks of a 48-layer serve are byte-identical replicas of it, so `per_step = per_block x 97` is
    arithmetic rather than extrapolation.
  * Running kineto in the same process that has captured and freed a large hipGraph SIGSEGVs both
    ranks in this image (`minisgl-rdna4:m1b-20260903`, observed 2026-09-05: exit -11 immediately
    after `profiler_start`/`profiler_stop`). Keeping the census out of the serve harness is what
    lets the serve harness finish.

What the profiler CAN and CANNOT tell us here: this image's kineto reports ZERO device activity, so
per-kernel GPU durations are unavailable from it. It does record the HIP launch API, which is
exactly the launch question. GPU time comes from the event-timed A/B in `hc_fusion_ab.py`.

Captured, the HOST issues none of these — the whole decode step is one `hipGraphLaunch`. Every one
of them still EXECUTES as a graph node, which is why the count is the thing being cut.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import textwrap
from collections import defaultdict
from types import MethodType

import torch
import torch.nn.functional as F

_LAUNCH_APIS = ("hipLaunchKernel", "hipExtModuleLaunchKernel", "hipModuleLaunchKernel",
                "hipGraphLaunch", "hipLaunchCooperativeKernel", "hipExtLaunchKernel",
                "hipMemcpyAsync", "hipMemsetAsync")


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
    marks = {
        "old_mix_divides_at_runtime": "/ self._hc" in hc["mix"],
        "old_norm_adds_one_per_call": "self.weight + 1.0" in nm["forward"],
    }
    if not all(marks.values()):
        raise AssertionError(f"baseline blobs are not the pre-fusion code: {marks}")
    ns = {"torch": torch, "F": F, "_rms_norm": _rms_norm}
    fut = "from __future__ import annotations\n"
    for src in list(hc.values()) + [nm["forward"]]:
        exec(compile(fut + src, payload["base_sha"], "exec"), ns)
    return {"mix": ns["mix"], "combine": ns["combine"], "norm_forward": ns["forward"],
            "base_sha": payload["base_sha"], "marks": marks}


def census(fn, reps: int) -> dict:
    from torch.profiler import ProfilerActivity, profile

    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
    n = 0
    by: dict = defaultdict(int)
    for ev in prof.events():
        if getattr(ev, "name", "") in _LAUNCH_APIS:
            n += 1
            by[ev.name] += 1
    return {"launches_per_call": round(n / max(1, reps), 3),
            "by_api": {k: round(v / max(1, reps), 3) for k, v in sorted(by.items())}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="/model/config.json")
    ap.add_argument("--baseline-json",
                    default="/engine/docs/measurements/HC_FUSION_2026-09-05/baseline_source.json")
    ap.add_argument("--blocks", type=int, default=97,
                    help="HC blocks in the served model: 2 per layer + the final mixer")
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--reps", type=int, default=20)
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
    torch.manual_seed(0)

    torch.set_default_dtype(torch.bfloat16)
    blk = HyperConnection(hidden_size=H, hc_count=HC, hc_lowrank=LR, eps=eps, use_combine=True)
    torch.set_default_dtype(torch.float32)
    blk.hc_norm.weight = (torch.randn(W) * 0.05).to(dev, torch.bfloat16)
    blk.input_mix_weight_down.weight = (torch.randn(LR, W) * W ** -0.5).to(dev, torch.bfloat16)
    blk.input_mix_weight_up.weight = (torch.randn(W, LR) * LR ** -0.5).to(dev, torch.bfloat16)
    blk.block_inject_weight.weight = (torch.randn(HC, W) * W ** -0.5).to(dev, torch.bfloat16)

    x = torch.randn(args.bs, W, device=dev, dtype=torch.bfloat16)
    y = torch.randn(args.bs, H, device=dev, dtype=torch.bfloat16)

    old = load_old(args.baseline_json)
    out = {"config": {"hidden": H, "hc_count": HC, "hc_lowrank": LR, "wide": W, "bs": args.bs},
           "blocks_per_step": args.blocks, "base_sha": old["base_sha"]}

    def one():
        m, r = blk.mix(x)
        blk.combine(y, r)

    def mix_only():
        blk.mix(x)

    # OLD leg first (the layer has not been post_load'd, so its weights are un-folded, which is
    # exactly what the old code expects).
    blk.mix = MethodType(old["mix"], blk)
    blk.combine = MethodType(old["combine"], blk)
    blk.hc_norm.forward = MethodType(old["norm_forward"], blk.hc_norm)
    out["old"] = {"per_block": census(one, args.reps), "mix_only": census(mix_only, args.reps)}
    for obj, attr in ((blk, "mix"), (blk, "combine"), (blk.hc_norm, "forward")):
        del obj.__dict__[attr]

    blk.post_load()
    out["new"] = {"per_block": census(one, args.reps), "mix_only": census(mix_only, args.reps)}
    out["new"]["packed_gemv_engaged"] = bool(blk._fused_ok(blk.hc_norm.forward(x)))
    out["new"]["scale_folded"] = bool(blk._scale_folded)

    for leg in ("old", "new"):
        per = out[leg]["per_block"]["launches_per_call"]
        out[leg]["launches_per_step"] = round(per * args.blocks, 1)
    out["delta"] = {
        "launches_per_block_old": out["old"]["per_block"]["launches_per_call"],
        "launches_per_block_new": out["new"]["per_block"]["launches_per_call"],
        "launches_per_step_old": out["old"]["launches_per_step"],
        "launches_per_step_new": out["new"]["launches_per_step"],
        "cut_fraction": round(
            1.0 - out["new"]["per_block"]["launches_per_call"]
            / max(1e-9, out["old"]["per_block"]["launches_per_call"]), 4),
    }
    print(json.dumps(out, indent=2), flush=True)
    if args.json:
        os.makedirs(os.path.dirname(args.json), exist_ok=True)
        json.dump(out, open(args.json, "w"), indent=2)
        print(f"wrote {args.json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
