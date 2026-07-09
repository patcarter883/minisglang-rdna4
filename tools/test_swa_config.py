"""CPU-only config-shape checks for Sliding-Window-Attention (SWA / Laguna) support.

Validates that ModelConfig.from_hf parses the Laguna hybrid schedule WITHOUT tripping the
GDN-hybrid path, that the SWA helper properties + KV-layer accounting are correct, and reports
the KV-footprint before/after the window-capped sizing. Run: python tools/test_swa_config.py
"""
from __future__ import annotations

import sys

from transformers import PretrainedConfig

from minisgl.models.config import ModelConfig


def _laguna_hf_config() -> PretrainedConfig:
    # Ground-truth shape from poolside/Laguna-XS.2-INT4 config.json (2026-07).
    layer_types = []
    heads = []
    for i in range(40):
        if i % 4 == 0:
            layer_types.append("full_attention")
            heads.append(48)
        else:
            layer_types.append("sliding_attention")
            heads.append(64)
    return PretrainedConfig(
        model_type="laguna",
        architectures=["LagunaForCausalLM"],
        num_hidden_layers=40,
        hidden_size=2048,
        vocab_size=100352,
        head_dim=128,
        num_attention_heads=48,
        num_key_value_heads=8,
        sliding_window=512,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        max_position_embeddings=131072,
        num_experts=256,
        num_experts_per_tok=8,
        moe_intermediate_size=512,
        shared_expert_intermediate_size=512,
        moe_routed_scaling_factor=2.5,
        layer_types=layer_types,
        num_attention_heads_per_layer=heads,
        rope_parameters={
            "full_attention": {
                "rope_type": "yarn",
                "rope_theta": 500000.0,
                "factor": 32.0,
                "beta_fast": 64.0,
                "beta_slow": 1.0,
                "original_max_position_embeddings": 4096,
                "partial_rotary_factor": 0.5,
                "attention_factor": 1.0,
            },
            "sliding_attention": {
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 1.0,
            },
            "original_max_position_embeddings": 4096,
        },
    )


def main() -> int:
    mc = ModelConfig.from_hf(_laguna_hf_config())
    fails = []

    def check(name, got, want):
        ok = got == want
        print(f"  [{'ok' if ok else 'FAIL'}] {name}: {got!r}" + ("" if ok else f" (want {want!r})"))
        if not ok:
            fails.append(name)

    print("== SWA schedule / trap-avoidance ==")
    check("sliding_window", mc.sliding_window, 512)
    check("is_swa_hybrid", mc.is_swa_hybrid, True)
    # THE TRAP: a SWA schedule must NOT read as a GDN hybrid.
    check("is_gdn_hybrid (must be False)", mc.is_gdn_hybrid, False)
    check("num_gdn_layers", mc.num_gdn_layers, 0)
    check("num_swa_layers", mc.num_swa_layers, 30)
    check("num full-attn layers", len(mc.full_attn_layer_ids), 10)
    check("swa_layer_ids[:4]", mc.swa_layer_ids[:4], [1, 2, 3, 5])
    check("full_attn_layer_ids[:3]", mc.full_attn_layer_ids[:3], [0, 4, 8])

    print("== KV-layer accounting (main pool = full-attn only) ==")
    check("num_kv_layers (main pool)", mc.num_kv_layers, 10)

    print("== per-layer QO heads ==")
    check("attn_head_counts[0] (full)", mc.attn_head_counts[0], 48)
    check("attn_head_counts[1] (sliding)", mc.attn_head_counts[1], 64)

    print("== dual RoPE ==")
    check("full rope base (yarn theta)", mc.rotary_config.base, 500000.0)
    check("full rope rotary_dim (partial 0.5)", mc.rotary_config.rotary_dim, 64)
    check("full rope is yarn-scaled", mc.rotary_config.scaling is not None, True)
    check("sliding rope base", mc.sliding_rotary_config.base, 10000.0)
    check("sliding rope rotary_dim (partial 1.0)", mc.sliding_rotary_config.rotary_dim, 128)
    check("sliding rope is default (no scaling)", mc.sliding_rotary_config.scaling is None, True)

    print("== router scaling ==")
    check("routed_scaling_factor", mc.routed_scaling_factor, 2.5)

    print("== KV footprint (bf16, TP=1, 32k context, per active sequence) ==")
    ctx = 32768
    win = mc.sliding_window
    per_tok_layer = 2 * mc.head_dim * mc.num_kv_heads * 2  # K+V * head_dim * kv_heads * 2 bytes
    naive = 40 * ctx * per_tok_layer  # every attn layer full-context
    capped = (10 * ctx + 30 * win) * per_tok_layer  # full full-ctx + sliding window-capped
    print(f"  naive  (40 layers x 32k)          : {naive/2**30:.3f} GiB/seq")
    print(f"  capped (10 x 32k + 30 x {win})     : {capped/2**30:.3f} GiB/seq")
    print(f"  reduction                          : {naive/capped:.2f}x")

    # Regression guard: a real GDN hybrid must still be a GDN hybrid, not SWA.
    print("== regression: GDN hybrid still routes GDN ==")
    gdn = ModelConfig.from_hf(PretrainedConfig(
        model_type="qwen3_next", architectures=["Qwen3NextForCausalLM"],
        num_hidden_layers=8, hidden_size=512, vocab_size=1000, head_dim=64,
        num_attention_heads=8, num_key_value_heads=2, rms_norm_eps=1e-6,
        max_position_embeddings=4096, rope_theta=10000.0,
        linear_num_key_heads=4, linear_num_value_heads=8, linear_key_head_dim=64,
        linear_value_head_dim=64, linear_conv_kernel_dim=4, full_attention_interval=4,
    ))
    check("gdn is_gdn_hybrid", gdn.is_gdn_hybrid, True)
    check("gdn is_swa_hybrid (must be False)", gdn.is_swa_hybrid, False)
    check("gdn num_kv_layers (2 full of 8)", gdn.num_kv_layers, 2)

    print()
    if fails:
        print(f"FAILED: {fails}")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
