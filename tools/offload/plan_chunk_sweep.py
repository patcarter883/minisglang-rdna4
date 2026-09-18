"""Plan-ONLY sweep of `MINISGL_WEIGHT_ARENA_CHUNK_MIB` x device tier, at a given TP.

WHY THIS EXISTS. The 48-layer TP=2 refusal is an inequality between two integers:
`plan_arena_reservation_bytes(plan, chunk) * ranks + floor` vs live `MemAvailable`. The left side is
a PURE FUNCTION of (layer row list, chunk_bytes, device budget) -- `chunk_plan.py` says so in its
first paragraph -- so it can be evaluated in seconds on a meta build, without pinning a page, without
a card, and without the 6-minute Stage-B load that a real boot pays before it reaches the gate.

Round 1 charged 27.00 GiB/rank for a 26.37 GiB payload at 768 MiB chunks. That 0.63 GiB/rank of
next-fit tail is 1.26 GiB across the node, against a shortfall that is now sub-GiB -- i.e. the chunk
size is not a tuning knob here, it is the feasibility term. This prints the whole curve so the
operating point is chosen from the arithmetic rather than guessed.

NO GPU. Meta build only; `set_rope_device(cpu)` keeps the rotary tables off a card.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

MODEL = os.environ.get("Q4E_MODEL", "/model")
GIB = 1 << 30
MIB = 1 << 20


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=48)
    ap.add_argument("--experts", type=int, default=0, help="0 = the checkpoint's own count")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--chunk-mib", default="512,640,704,736,750,752,768,896,1024,1408,1504,1536,2048,3072")
    ap.add_argument("--device-gb", default="7.0,7.5,8.0,8.5,8.79,9.0,9.5,10.0")
    ap.add_argument("--json", default="")
    # The MTP draft head is a 49th MoE layer in the plan (`plan.py` numbers it `num_layers + n`),
    # and with `OFFLOAD_MTP_HEAD = True` it enters the arena. It matters here and not merely as one
    # more layer: its `gate_up` region is 1.5625 GiB/rank at TP=2 -- BF16 where every backbone layer
    # is NVFP4 -- which is larger than a whole chunk at the shipped 1372 MiB, so the arena's
    # never-straddle rule GROWS the chunk to fit it and the backbone's 750 MiB/layer rows then pack
    # badly into the wider chunk. A live boot measured 4.37 GiB (13.3%) abandoned that way, which is
    # what pushed the reservation over its ceiling. Sweeping with spec="none" cannot see any of this.
    ap.add_argument("--spec", default="none", help="spec algorithm; 'mtp' adds the draft head as layer num_layers")
    args = ap.parse_args()

    from minisgl.distributed import DistributedInfo, set_tp_info

    set_tp_info(0, args.tp)
    import minisgl.layers.rotary as rotary_mod

    rotary_mod.set_rope_device(torch.device("cpu"))

    from minisgl.models.config import ModelConfig
    from minisgl.models import create_model
    from minisgl.utils import cached_load_hf_config
    from minisgl.weights.plan import (
        exact_arena_reservation_bytes,
        plan_arena_reservation_bytes,
        resolve_weight_plan,
    )

    hf = cached_load_hf_config(MODEL)
    mc = ModelConfig.from_hf(hf, spec_algorithm=args.spec)
    # A truncated depth is the only knob a subset needs; experts stay at the checkpoint's count so
    # the per-layer row bytes are the REAL ones.
    import dataclasses

    over = {}
    if args.layers and args.layers != mc.num_layers:
        over["num_layers"] = args.layers
    if args.experts and args.experts != mc.num_experts:
        over["num_experts"] = args.experts
    if over:
        mc = dataclasses.replace(mc, **over)

    torch.set_default_dtype(torch.bfloat16)
    with torch.device("meta"):
        model = create_model(mc)

    body = 0
    experts = 0
    for name, p in model.state_dict().items():
        nb = p.numel() * p.element_size()
        if p.dim() >= 3 and p.shape[0] == mc.num_experts and ".experts." in name:
            experts += nb
        else:
            body += nb
    # The BODY is the term the device tier cannot escape: it is resident on every rank before a
    # single expert layer is placed, so every byte of it is a byte the host arena has to carry
    # instead. Print it by group or the "device tier is VRAM-bound" verdict is unauditable.
    groups: "dict[str, int]" = {}
    for name, p in model.state_dict().items():
        nb = p.numel() * p.element_size()
        if p.dim() >= 3 and p.shape[0] == mc.num_experts and ".experts." in name:
            continue
        key = name
        for pat in ("hyper_connection", "embed_tokens", "lm_head", "shared_expert", "ple",
                    "linear_attn", "self_attn", "mlp", "norm", "gate"):
            if pat in name:
                key = pat
                break
        else:
            key = "other:" + name
        groups[key] = groups.get(key, 0) + nb
    print("  body by group (per rank):")
    for k, v in sorted(groups.items(), key=lambda kv: -kv[1]):
        print(f"    {k:24s} {v / GIB:8.3f} GiB")

    print(f"tp={args.tp} layers={mc.num_layers} experts={mc.num_experts}")
    print(f"  per-rank body           {body / GIB:8.3f} GiB")
    print(f"  per-rank routed experts {experts / GIB:8.3f} GiB "
          f"({experts / GIB / mc.num_layers:.4f}/layer)")

    class _Cfg:
        pass

    rows_printed = False
    out = []
    for dgb in [float(x) for x in args.device_gb.split(",")]:
        for cm in [int(x) for x in args.chunk_mib.split(",")]:
            cfg = _Cfg()
            cfg.model_config = mc
            cfg.tp_info = DistributedInfo(0, args.tp)
            cfg.weight_offload_device_gb = dgb
            cfg.weight_offload_gb = 0.0
            res = resolve_weight_plan(
                cfg,
                device_budget_bytes=int(dgb * GIB),
                arena_chunk_bytes=cm * MIB,
                model=model,
            )
            plan = res.plan
            rows = plan.host_row_requests()
            if not rows_printed:
                sizes = sorted({r.nbytes if hasattr(r, "nbytes") else int(r[1]) for r in rows})
                print(f"  host rows: {len(rows)} distinct sizes {[s / MIB for s in sizes][:8]}")
                rows_printed = True
            pinned = plan_arena_reservation_bytes(plan, cm * MIB)
            out.append({
                "device_gb": dgb,
                "chunk_mib": cm,
                "device_layers": plan.num_device_layers,
                "host_layers": int(plan.num_host_layers),
                "host_payload_bytes": int(plan.host_resident_bytes),
                "device_bytes": int(plan.device_resident_bytes),
                "pinned_per_rank": int(pinned),
                "pinned_node": int(pinned) * args.tp,
                "waste_per_rank": int(pinned) - int(plan.host_resident_bytes),
            })

    out.sort(key=lambda r: (r["pinned_node"], -r["device_gb"]))
    print()
    print(f"{'dev_gb':>7} {'chunk':>6} {'devL':>5} {'payload/rk':>11} {'pinned/rk':>10} "
          f"{'waste/rk':>9} {'node':>9} {'node+12':>9}")
    for r in out:
        print(f"{r['device_gb']:7.2f} {r['chunk_mib']:6d} {r['device_layers']:5d} "
              f"{r['host_payload_bytes'] / GIB:10.3f}G {r['pinned_per_rank'] / GIB:9.3f}G "
              f"{r['waste_per_rank'] / GIB:8.3f}G {r['pinned_node'] / GIB:8.3f}G "
              f"{r['pinned_node'] / GIB + 12:8.3f}G")

    from minisgl.weights.host_capacity import mem_available_bytes

    avail = mem_available_bytes()
    print(f"\nlive MemAvailable {avail / GIB:.3f} GiB; floor 12.000 GiB "
          f"-> budget for the node arena {(avail - 12 * GIB) / GIB:.3f} GiB")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"tp": args.tp, "layers": mc.num_layers, "body_bytes": body,
                       "expert_bytes": experts, "mem_available": avail, "rows": out}, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
