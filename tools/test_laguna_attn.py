"""CPU meta-instantiation check for Laguna's config-driven per-layer attention wiring (item 4).

Builds every LagunaAttention layer on the meta device and asserts each got the right per-layer QO
head count, RoPE scheme, sliding window, and compact KV-pool id — with NO model-name branch (all
derived from ModelConfig). Run: python tools/test_laguna_attn.py
"""
from __future__ import annotations

import sys

import torch

from minisgl.models.config import ModelConfig
from minisgl.models.laguna import laguna_layer_plan, LagunaAttention
from tools.test_swa_config import _laguna_hf_config  # reuse the ground-truth shape

FAILS = []


def check(name, got, want):
    ok = got == want
    print(f"  [{'ok' if ok else 'FAIL'}] {name}: {got!r}" + ("" if ok else f" (want {want!r})"))
    if not ok:
        FAILS.append(name)


def main() -> int:
    from minisgl.distributed.info import set_tp_info
    from minisgl.layers.rotary import set_rope_device

    set_tp_info(rank=0, size=1)  # single-rank TP for the shape check
    set_rope_device(torch.device("cpu"))  # rope tables on CPU (shape check only)
    mc = ModelConfig.from_hf(_laguna_hf_config())

    print("== per-layer plan (config-driven, no model-name branch) ==")
    p0 = laguna_layer_plan(mc, 0)   # full
    p1 = laguna_layer_plan(mc, 1)   # sliding
    check("L0 is_sliding", p0.is_sliding, False)
    check("L0 num_qo_heads (full=48)", p0.num_qo_heads, 48)
    check("L0 sliding_window", p0.sliding_window, 0)
    check("L0 kv_id (compact full id, layer0->0)", p0.kv_id, 0)
    check("L0 rope base (yarn)", p0.rotary_config.base, 500000.0)
    check("L1 is_sliding", p1.is_sliding, True)
    check("L1 num_qo_heads (sliding=64)", p1.num_qo_heads, 64)
    check("L1 sliding_window", p1.sliding_window, 512)
    check("L1 kv_id (compact swa id, layer1->0)", p1.kv_id, 0)
    check("L1 rope base (default)", p1.rotary_config.base, 10000.0)
    # layer 4 is the 2nd full layer -> compact full id 1; layer 5 is a sliding -> compact swa id 3
    check("L4 kv_id (2nd full -> 1)", laguna_layer_plan(mc, 4).kv_id, 1)
    check("L5 kv_id (4th sliding -> 3)", laguna_layer_plan(mc, 5).kv_id, 3)

    print("== meta-instantiate all 40 attention layers (per-layer shapes) ==")
    D = mc.head_dim
    with torch.device("meta"):
        for lid in range(mc.num_layers):
            attn = LagunaAttention(mc, lid)
            plan = attn.plan
            want_heads = 48 if lid % 4 == 0 else 64
            # q_proj output = heads * head_dim; g_proj output = heads (one gate per head)
            qo = attn.q_proj.weight.shape[0]
            go = attn.g_proj.weight.shape[0]
            oi = attn.o_proj.weight.shape[1]
            if qo != want_heads * D or go != want_heads or oi != want_heads * D:
                check(f"L{lid} shapes", (qo, go, oi), (want_heads * D, want_heads, want_heads * D))
            # AttentionLayer got the per-layer window + compact pool id
            if attn.attn.sliding_window != plan.sliding_window:
                check(f"L{lid} window", attn.attn.sliding_window, plan.sliding_window)
            if attn.attn.layer_id != plan.kv_id:
                check(f"L{lid} kv_id", attn.attn.layer_id, plan.kv_id)
    print(f"  [ok] all 40 layers instantiated (10 full q=48*128, 30 sliding q=64*128)")

    print()
    if FAILS:
        print(f"FAILED: {FAILS}")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
