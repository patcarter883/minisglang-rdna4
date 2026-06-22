"""Phase 3M-1 — meta-device build + structural smoke for qwen3_5_moe (Qwen3.6-35B-A3B int4).

Builds the 35B GDN-hybrid MoE on the meta device (no real memory, no GPU) exactly as the engine
does, from the REAL HF config, then checks:
  * 30 GDN (linear_attn) layers + 10 full (self_attn) at [3,7,...,39];
  * each layer's MLP is the MoE sparse block: bf16 router `gate` (E,H), AWQ int4 grouped experts
    (gate_up/down qweight/scales/qzeros, stacked over E), bf16 shared expert (merged gate_up +
    down) + bf16 sigmoid `shared_expert_gate`;
  * GDN bridge keys (in_proj_qkvz/ba, conv1d_weight, A_log/dt_bias fp32) + full-attn gated GQA;
  * untied lm_head; and state_dict() <-> load_state_dict() round-trips with no missing/unexpected.

CPU-only; run in the combined ROCm image (healthy triton for the GDN kernels):
    PYTHONPATH=/engine/python python /engine/tools/qwen3_5_moe_build_smoke.py
"""
from __future__ import annotations

import torch
from transformers import AutoConfig

from minisgl.distributed import set_tp_info
from minisgl.layers import set_rope_device
from minisgl.models import create_model
from minisgl.models.config import ModelConfig
from minisgl.utils import torch_dtype

MODEL = "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit"


def main() -> None:
    cfg = ModelConfig.from_hf(AutoConfig.from_pretrained(MODEL))
    assert cfg.is_gdn_hybrid and cfg.is_moe and cfg.quant.is_awq, cfg
    H = cfg.hidden_size                  # 2048
    E = cfg.num_experts                  # 256
    inter = cfg.moe_intermediate_size    # 512
    sinter = cfg.shared_expert_intermediate_size  # 512
    pf, g = 8, cfg.quant.group_size      # 8, 32

    set_tp_info(rank=0, size=1)
    set_rope_device(torch.device("cpu"))
    with torch.device("meta"), torch_dtype(torch.bfloat16):
        model = create_model(cfg)

    gdn = model.iter_gdn_layers()
    print(f"GDN bridge layers: {len(gdn)} (expect 30)")
    assert len(gdn) == cfg.num_gdn_layers == 30

    sd = model.state_dict()
    print(f"state_dict tensors: {len(sd)}")
    for k in ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"):
        assert k in sd, k  # untied lm_head
    assert sd["model.embed_tokens.weight"].shape[0] == cfg.vocab_size

    # --- a GDN layer (0): bridged GDN keys (bf16) + MoE block ---
    L0 = "model.layers.0."
    for suf in ("linear_attn.in_proj_qkvz.weight", "linear_attn.in_proj_ba.weight",
                "linear_attn.conv1d_weight", "linear_attn.A_log", "linear_attn.dt_bias",
                "linear_attn.norm.weight", "linear_attn.out_proj.weight"):
        assert L0 + suf in sd, suf
    assert sd[L0 + "linear_attn.A_log"].dtype == torch.float32
    assert sd[L0 + "linear_attn.dt_bias"].dtype == torch.float32
    assert tuple(sd[L0 + "linear_attn.conv1d_weight"].shape) == (cfg.gdn_conv_dim, 1, 4)

    # --- a full-attn layer (3): gated GQA (bf16, no bias), q/k norm ---
    L3 = "model.layers.3."
    for suf in ("self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
                "self_attn.q_norm.weight", "self_attn.k_norm.weight", "self_attn.o_proj.weight"):
        assert L3 + suf in sd, suf
    assert tuple(sd[L3 + "self_attn.q_proj.weight"].shape) == (2 * 16 * 256, H)  # q + per-head gate
    assert tuple(sd[L3 + "self_attn.k_proj.weight"].shape) == (2 * 256, H)       # 2 kv heads
    assert L3 + "self_attn.q_proj.bias" not in sd  # attention_bias=False

    # --- MoE sparse block (present on BOTH layer 0 and 3; mlp_only_layers=[]) ---
    for L in (L0, L3):
        assert tuple(sd[L + "mlp.gate.weight"].shape) == (E, H)               # bf16 router
        assert tuple(sd[L + "mlp.shared_expert_gate.weight"].shape) == (1, H)  # bf16 sigmoid gate
        # bf16 shared expert (merged gate_up + down), NOT quantized
        assert tuple(sd[L + "mlp.shared_expert.gate_up_proj.weight"].shape) == (2 * sinter, H)
        assert tuple(sd[L + "mlp.shared_expert.down_proj.weight"].shape) == (H, sinter)
        assert sd[L + "mlp.shared_expert.gate_up_proj.weight"].dtype == torch.bfloat16
        assert "qweight" not in repr([k for k in sd if k.startswith(L + "mlp.shared_expert.")])
        # AWQ int4 grouped experts (K-major qweight (E,K,N//pf), scales (E,K//g,N), qzeros), stacked
        assert tuple(sd[L + "mlp.experts.gate_up_proj.qweight"].shape) == (E, H, 2 * inter // pf)
        assert tuple(sd[L + "mlp.experts.gate_up_proj.scales"].shape) == (E, H // g, 2 * inter)
        assert tuple(sd[L + "mlp.experts.gate_up_proj.qzeros"].shape) == (E, H // g, 2 * inter // pf)
        assert tuple(sd[L + "mlp.experts.down_proj.qweight"].shape) == (E, inter, H // pf)
        assert tuple(sd[L + "mlp.experts.down_proj.scales"].shape) == (E, inter // g, H)
        assert tuple(sd[L + "mlp.experts.down_proj.qzeros"].shape) == (E, inter // g, H // pf)
        assert sd[L + "mlp.experts.gate_up_proj.qweight"].dtype == torch.int32
        assert sd[L + "mlp.experts.gate_up_proj.scales"].dtype == torch.float16

    # no full-attn keys on a GDN layer (and vice-versa)
    assert L0 + "self_attn.q_proj.weight" not in sd
    assert L3 + "linear_attn.in_proj_qkvz.weight" not in sd

    nlayers = sum(1 for k in sd if k.endswith("input_layernorm.weight"))
    assert nlayers == cfg.num_layers == 40, nlayers
    print(f"[OK] {nlayers} layers ({len(gdn)} GDN / {nlayers - len(gdn)} full); MoE sparse blocks "
          f"(bf16 router/shared/gates + AWQ int4 grouped experts) — all keys/shapes/dtypes correct.")

    model.load_state_dict(model.state_dict())
    print("[OK] state_dict <-> load_state_dict round-trip clean (no missing / unexpected keys).")


if __name__ == "__main__":
    main()
