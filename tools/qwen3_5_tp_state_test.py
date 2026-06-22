"""Phase 4-2 — CPU check that the GDN recurrent-state cache + MHA KV cache size to per-rank-local
heads under TP, matching the head-parallel GDN layer (Phase 4-1).

A GDN-hybrid engine keeps a fixed per-sequence recurrent state (conv + ssm) per GDN layer alongside
the paged MHA KV cache. Under TP each rank's `linear_attn` owns conv_dim/tp conv channels and
num_v_heads/tp value heads, so its state slot must match — otherwise the conv/ssm buffers mismatch
the layer's projections at runtime. This test, for TP=1 and TP=2, asserts:

  * the engine's sizing formula (`div_even(gdn_conv_dim, tp)`, `div_even(linear_num_value_heads, tp)`)
    equals the GDN layer's ACTUAL local geometry (`QwenGatedDeltaNet.conv_dim` / `.num_v_heads`)
    built at that tp — so engine state alloc can't drift from the layer;
  * a concretely-allocated `GDNStateCache` with those local dims yields conv/ssm slot shapes that
    match what `forward_prefill`/`forward_decode` read (conv (slots, conv_dim, k-1); ssm (slots,
    num_v_heads, head_v_dim, head_k_dim));
  * the full-attention KV head count shards as `div_even(num_kv_heads, tp, allow_replicate=True)`.

CPU-only (GDNStateCache allocs a tiny CPU buffer; the layer is meta); no GPU lease:
    PYTHONPATH=/engine/python python /engine/tools/qwen3_5_tp_state_test.py
"""
from __future__ import annotations

import torch
from minisgl.gdn.layer import QwenGatedDeltaNet
from minisgl.kvcache.gdn_state import GDNStateCache
from minisgl.models.config import ModelConfig
from minisgl.utils import cached_load_hf_config, div_even

MODELS = ["Qwen/Qwen3.5-4B", "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit"]
NUM_SLOTS = 4  # tiny: 1 NULL + a few real, enough to check shapes


def run(model_path: str) -> None:
    print(f"\n=== {model_path} ===")
    mc = ModelConfig.from_hf(cached_load_hf_config(model_path))
    assert mc.is_gdn_hybrid
    full_conv = mc.gdn_conv_dim
    full_v = mc.linear_num_value_heads
    print(f"full: conv_dim={full_conv}, num_v_heads={full_v}, head_v_dim={mc.linear_value_head_dim}, "
          f"head_k_dim={mc.linear_key_head_dim}, num_kv_heads={mc.num_kv_heads}")

    for tp in (1, 2):
        # --- layer's actual local geometry at this tp (built on meta) ---
        gdn = QwenGatedDeltaNet(
            hidden_size=mc.hidden_size,
            num_k_heads=mc.linear_num_key_heads,
            num_v_heads=mc.linear_num_value_heads,
            head_k_dim=mc.linear_key_head_dim,
            head_v_dim=mc.linear_value_head_dim,
            conv_kernel_size=mc.linear_conv_kernel_dim,
            tp_size=tp,
            device="meta",  # no real memory / no GPU
        )
        # --- the engine's GDNStateCache sizing formula (engine.py) ---
        eng_conv_dim = div_even(full_conv, tp)
        eng_v_heads = div_even(full_v, tp)
        assert eng_conv_dim == gdn.conv_dim, (
            f"tp={tp}: engine conv_dim {eng_conv_dim} != layer local {gdn.conv_dim}"
        )
        assert eng_v_heads == gdn.num_v_heads, (
            f"tp={tp}: engine num_v_heads {eng_v_heads} != layer local {gdn.num_v_heads}"
        )
        # conv1d_weight + in_proj geometry must agree with the state conv_dim
        assert gdn.conv1d_weight.shape[0] == gdn.conv_dim
        assert gdn.A_log.shape == (gdn.num_v_heads,)

        # --- concrete state cache with the engine's (tp-divided) dims ---
        cache = GDNStateCache(
            num_gdn_layers=mc.num_gdn_layers,
            num_slots=NUM_SLOTS,
            conv_dim=eng_conv_dim,
            conv_kernel=mc.linear_conv_kernel_dim,
            num_v_heads=eng_v_heads,
            head_v_dim=mc.linear_value_head_dim,
            head_k_dim=mc.linear_key_head_dim,
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )
        conv0, ssm0 = cache.conv(0), cache.ssm(0)
        want_conv = (NUM_SLOTS, gdn.conv_dim, mc.linear_conv_kernel_dim - 1)
        want_ssm = (NUM_SLOTS, gdn.num_v_heads, mc.linear_value_head_dim, mc.linear_key_head_dim)
        assert tuple(conv0.shape) == want_conv, (tuple(conv0.shape), want_conv)
        assert tuple(ssm0.shape) == want_ssm, (tuple(ssm0.shape), want_ssm)

        # --- full-attn KV head sharding (engine.py:_determine_num_pages) ---
        kv_local = div_even(mc.num_kv_heads, tp, allow_replicate=True)

        print(f"  TP={tp}: conv_dim {full_conv}->{gdn.conv_dim}, v_heads {full_v}->{gdn.num_v_heads} "
              f"| conv_state{tuple(conv0.shape)} ssm_state{tuple(ssm0.shape)} | kv_heads/rank={kv_local}")


def main() -> None:
    for m in MODELS:
        run(m)
    print("\nPASS — GDN state cache + KV cache size to per-rank-local heads; engine formula == layer geometry.")


if __name__ == "__main__":
    main()
