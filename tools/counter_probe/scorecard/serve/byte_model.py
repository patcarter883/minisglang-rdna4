#!/usr/bin/env python3
"""Bytes read per DECODE STEP, per card, for Qwen3.6-35B-A3B-AWQ-4bit at TP=2.

WHY THIS IS NOT "35B params at 4 bits". Two properties of this checkpoint make the back-of-envelope
number wrong by a large factor, and both are read off the checkpoint here rather than assumed:

  1. Only the ROUTED EXPERTS are quantized. `quantization_config.modules_to_not_convert` lists every
     attention, linear_attn, shared_expert, router and lm_head module, so those are bf16. A "4-bit
     35B" estimate under-counts the resident bf16 backbone.
  2. It is a GDN HYBRID, not a dense-KV transformer. `layer_types` is 30 linear_attention + 10
     full_attention (full_attention_interval=4 over 40 layers). Only 10 layers hold a KV cache; the
     other 30 hold a fixed-size recurrent state that does not grow with context. A per-token KV term
     computed over 40 layers over-counts by 4x, and the context-growth term is 4x smaller than a
     dense model of the same depth.

And the term that dominates a MoE decode: at top_k=8 of num_experts=256 a single token touches 8/256
of the expert weights per layer, so the routed-expert bytes are ~3% of the resident expert bytes.
Resident footprint and per-step traffic are different questions; this computes the second.

Everything comes from the SAFETENSORS HEADERS (exact shapes and dtypes, no torch, no GPU) plus
config.json, so the table is auditable tensor by tensor. TP=2 divides sharded weights; the shared
expert is REPLICATED on both cards (it is kept bf16 and not sharded — see the shared-expert memory),
which the sharding rule below encodes.

  python3 byte_model.py --step-ms <measured ms> [--ctx 1024] [--bs 1]
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

SNAP = Path("/home/pat/.cache/huggingface/hub/models--cyankiwi--Qwen3.6-35B-A3B-AWQ-4bit/"
            "snapshots/00fcea2d3bcf5389b518d4fc082e5590e0ba4844")

# The corrected gfx1201 HBM roofline. NOT 674, NOT 644 — those are ACHIEVED bench numbers that got
# quoted as the ceiling; see the hbm-roofline memory.
HBM_GBS = 706.6

DT_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I64": 8, "I32": 4, "I16": 2, "I8": 1,
            "U8": 1, "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1}


def read_header(path: Path) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def inventory() -> dict:
    inv = {}
    for p in sorted(SNAP.glob("model-*.safetensors")):
        for name, meta in read_header(p).items():
            if name == "__metadata__":
                continue
            shape = meta["shape"]
            nb = DT_BYTES[meta["dtype"]]
            for d in shape:
                nb *= d
            inv[name] = {"shape": shape, "dtype": meta["dtype"], "bytes": nb}
    return inv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step-ms", type=float, default=None, help="measured decode step wall, ms")
    ap.add_argument("--ctx", type=int, default=1024, help="tokens of KV context per sequence")
    ap.add_argument("--bs", type=int, default=1, help="decode batch size")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    cfg = json.load(open(SNAP / "config.json"))["text_config"]
    L = cfg["num_hidden_layers"]
    types = cfg["layer_types"]
    n_full = sum(1 for t in types if t == "full_attention")
    n_lin = L - n_full
    topk, n_exp = cfg["num_experts_per_tok"], cfg["num_experts"]

    inv = inventory()
    # MTP is not served here (SPEC=none), so its head is resident but never read on a decode step.
    inv = {k: v for k, v in inv.items() if not k.startswith("mtp.")}
    # The vision tower is likewise never touched by a text-only decode.
    inv = {k: v for k, v in inv.items() if ".visual." not in k}

    # --- classify every remaining tensor ---------------------------------------------------------
    # `experts` = the routed expert stacks: read only topk/n_exp of them per token.
    # `embed`   = the input embedding: ONE row per token, not the whole table.
    # everything else is read in full every step.
    buckets: dict[str, int] = {}
    detail: dict[str, list] = {}
    for name, m in inv.items():
        if ".experts." in name and "shared_expert" not in name:
            k = "routed_experts"
        elif "shared_expert" in name:
            k = "shared_expert"
        elif "embed_tokens" in name:
            k = "embed_tokens"
        elif name.startswith("lm_head"):
            k = "lm_head"
        elif ".linear_attn." in name:
            k = "linear_attn(GDN)"
        elif ".self_attn." in name:
            k = "self_attn"
        elif ".mlp.gate" in name:
            k = "router"
        else:
            k = "norms/other"
        buckets[k] = buckets.get(k, 0) + m["bytes"]
        detail.setdefault(k, []).append((name, m["shape"], m["dtype"], m["bytes"]))

    tp = args.tp
    # Per-step READ bytes per card. Sharding: every weight bucket here is either row/column sharded
    # across TP ranks (attention, GDN, experts, lm_head) or replicated. The shared expert is
    # REPLICATED and bf16 (it is in modules_to_not_convert and is not sharded) — that is the one
    # bucket where dividing by TP would be wrong.
    per_card: dict[str, float] = {}
    per_card["self_attn"] = buckets.get("self_attn", 0) / tp
    per_card["linear_attn(GDN)"] = buckets.get("linear_attn(GDN)", 0) / tp
    per_card["router"] = buckets.get("router", 0)              # tiny, replicated
    per_card["norms/other"] = buckets.get("norms/other", 0)    # replicated
    per_card["shared_expert"] = buckets.get("shared_expert", 0)  # REPLICATED, bf16
    # Routed experts: the batch reads the UNION of the experts its tokens selected, not bs*topk —
    # a fused MoE decode loads each selected expert ONCE for the whole batch. At bs=1 the two agree
    # (8 experts), but at bs=8 they do not: bs*topk would be 64 of 256 while the expected union is
    # n_exp*(1-(1-topk/n_exp)^bs) = 57.4. Multiplying by bs is the error that makes MoE decode look
    # like it must saturate bandwidth at concurrency when it does not.
    exp_union = n_exp * (1.0 - (1.0 - topk / n_exp) ** args.bs)
    per_card["routed_experts"] = buckets.get("routed_experts", 0) * (exp_union / n_exp) / tp
    per_card["lm_head"] = buckets.get("lm_head", 0) / tp
    # Embedding: bs rows, not the table.
    emb = detail.get("embed_tokens", [])
    row_bytes = 0
    for _n, shape, dt, _b in emb:
        row_bytes += DT_BYTES[dt] * shape[1]
    per_card["embed_row"] = row_bytes * args.bs

    # --- KV cache + GDN recurrent state ----------------------------------------------------------
    # fp8 KV (MINISGL_KV_FP8=1 is the compose default) => 1 byte/element. KV heads are sharded by TP.
    kv_heads = cfg["num_key_value_heads"]
    hd = cfg["head_dim"]
    kv_bytes_per_tok_per_card = n_full * 2 * max(kv_heads // tp, 1) * hd * 1
    per_card["kv_cache_read"] = kv_bytes_per_tok_per_card * args.ctx * args.bs
    # GDN state: [num_value_heads, key_head_dim, value_head_dim] per linear layer, read AND written
    # every step, and it does NOT grow with context. mamba_ssm_dtype=float32.
    gdn_state = (cfg["linear_num_value_heads"] // tp) * cfg["linear_key_head_dim"] \
        * cfg["linear_value_head_dim"] * 4
    per_card["gdn_state_rw"] = n_lin * gdn_state * 2 * args.bs   # x2: read + write

    total = sum(per_card.values())

    print(f"model      : Qwen3.6-35B-A3B-AWQ-4bit   TP={tp}  bs={args.bs}  ctx={args.ctx}")
    print(f"layers     : {L} = {n_lin} linear_attention (GDN) + {n_full} full_attention")
    print(f"experts    : top_k={topk} of {n_exp}  -> expected UNION at bs={args.bs} is "
          f"{exp_union:.1f} experts = {100*exp_union/n_exp:.2f}% of the resident expert weights")
    print(f"quantized  : routed experts only (AWQ int4 g32); attention / GDN / shared_expert / "
          f"lm_head are bf16")
    print()
    # "checkpoint bytes" is the WEIGHT total for that bucket across both cards; it is blank for the
    # runtime buckets (KV, GDN state, embedding row), which are not checkpoint tensors at all.
    print(f"{'bucket':<22}{'checkpoint bytes':>19}{'per-card per-step':>20}{'%':>8}")
    for k in sorted(per_card, key=lambda x: -per_card[x]):
        res = buckets.get(k)
        res_s = f"{res/1e6:>16.1f} MB" if res else f"{'(runtime)':>19}"
        print(f"{k:<22}{res_s}{per_card[k]/1e6:>17.2f} MB{100*per_card[k]/total:>8.1f}")
    print(f"{'TOTAL':<22}{'':>18}{total/1e6:>17.2f} MB")

    out = {"per_card_step_bytes": per_card, "total_bytes": total, "hbm_gbs": HBM_GBS,
           "bs": args.bs, "ctx": args.ctx, "tp": tp}
    if args.step_ms:
        gbs = total / (args.step_ms * 1e-3) / 1e9
        print()
        print(f"measured step wall : {args.step_ms:.3f} ms")
        print(f"achieved bandwidth : {gbs:.1f} GB/s")
        print(f"vs HBM roofline    : {100*gbs/HBM_GBS:.1f} %  (ceiling {HBM_GBS} GB/s)")
        out.update({"step_ms": args.step_ms, "achieved_gbs": gbs,
                    "pct_of_roofline": 100 * gbs / HBM_GBS})
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(out, indent=1))
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
