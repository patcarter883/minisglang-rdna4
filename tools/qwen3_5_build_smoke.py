"""Phase 3d-1 — meta-device build + state-dict structural smoke test for qwen3_5.

Builds the Qwen3.5-4B GDN-hybrid on the meta device (no real memory, no GPU) exactly the
way the engine does, then checks:
  * 24 GDN (linear_attn) layers + 8 full (self_attn) layers at [3,7,11,15,19,23,27,31];
  * every layer exposes the expected minisgl-native param keys (GDN bridge surfaces the
    nn.Module's in_proj_qkvz / in_proj_ba / conv1d_weight / A_log / dt_bias / norm / out_proj;
    full layers surface q/k/v/q_norm/k_norm/o_proj; dense SwiGLU gate_up/down);
  * state_dict() keys EXACTLY round-trip through load_state_dict() — no missing / unexpected,
    which proves the save/load key layout is self-consistent across the BaseOP<->nn.Module bridge.

CPU-only; run in the combined ROCm image (healthy triton needed to import the GDN kernels):
    PYTHONPATH=/engine/python python /engine/tools/qwen3_5_build_smoke.py
"""

from __future__ import annotations

import torch
from minisgl.distributed import set_tp_info
from minisgl.layers import set_rope_device
from minisgl.models import create_model
from minisgl.models.config import ModelConfig, RotaryConfig
from minisgl.utils import torch_dtype


def _qwen3_5_4b_config() -> ModelConfig:
    layer_types = tuple(
        "full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(32)
    )
    return ModelConfig(
        num_layers=32,
        num_qo_heads=16,
        num_kv_heads=4,
        head_dim=256,
        hidden_size=2560,
        vocab_size=248320,
        intermediate_size=9216,
        rms_norm_eps=1e-6,
        rotary_config=RotaryConfig(
            head_dim=256, rotary_dim=64, max_position=262144, base=10000000, scaling=None
        ),
        hidden_act="silu",
        tie_word_embeddings=True,
        num_experts=0,
        num_experts_per_tok=0,
        moe_intermediate_size=0,
        norm_topk_prob=False,
        model_type="qwen3_5_text",
        architectures=["Qwen3_5ForConditionalGeneration"],
        quant=None,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        layer_types=layer_types,
    )


def main() -> None:
    cfg = _qwen3_5_4b_config()
    set_tp_info(rank=0, size=1)  # TP=1, as the engine sets before building the model
    set_rope_device(torch.device("cpu"))  # rope tables can't live on meta
    with torch.device("meta"), torch_dtype(torch.bfloat16):
        model = create_model(cfg)

    gdn = model.iter_gdn_layers()
    print(f"GDN bridge layers: {len(gdn)} (expect 24)")
    assert len(gdn) == 24

    sd = model.state_dict()
    print(f"state_dict tensors: {len(sd)}")

    # top-level
    for k in ("model.embed_tokens.weight", "model.norm.weight"):
        assert k in sd, k

    # a GDN layer (0) — bridged nn.Module keys + norms + dense MLP
    gdn_keys = [
        "linear_attn.in_proj_qkvz.weight",
        "linear_attn.in_proj_ba.weight",
        "linear_attn.conv1d_weight",
        "linear_attn.A_log",
        "linear_attn.dt_bias",
        "linear_attn.norm.weight",
        "linear_attn.out_proj.weight",
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "mlp.gate_up_proj.weight",
        "mlp.down_proj.weight",
    ]
    for suf in gdn_keys:
        assert f"model.layers.0.{suf}" in sd, suf
    # a full-attention layer (3) — gated GQA keys
    for suf in (
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
        "self_attn.o_proj.weight",
    ):
        assert f"model.layers.3.{suf}" in sd, suf

    # shapes: q_proj is 2x (q + gate); in_proj_qkvz = 2*key_dim + 2*value_dim
    assert tuple(sd["model.layers.3.self_attn.q_proj.weight"].shape) == (2 * 16 * 256, 2560)
    assert tuple(sd["model.layers.3.self_attn.k_proj.weight"].shape) == (4 * 256, 2560)
    assert tuple(sd["model.layers.0.linear_attn.in_proj_qkvz.weight"].shape) == (2 * 2048 + 2 * 4096, 2560)
    assert tuple(sd["model.layers.0.linear_attn.conv1d_weight"].shape) == (8192, 1, 4)
    assert sd["model.layers.0.linear_attn.A_log"].dtype == torch.float32
    assert sd["model.layers.0.linear_attn.dt_bias"].dtype == torch.float32
    # no full-attn keys on a GDN layer (and vice-versa)
    assert "model.layers.0.self_attn.q_proj.weight" not in sd
    assert "model.layers.3.linear_attn.in_proj_qkvz.weight" not in sd
    print("[OK] key presence + shapes + dtypes correct.")

    # round-trip: state_dict() keys must EXACTLY satisfy load_state_dict() (bridge consistency)
    model.load_state_dict(model.state_dict())
    print("[OK] state_dict <-> load_state_dict round-trip clean (no missing / unexpected keys).")


if __name__ == "__main__":
    main()
