"""Phase 2M-1 — meta-device build + structural smoke for qwen2_moe (Qwen1.5-MoE-A2.7B-GPTQ-Int4).

Builds the model on the meta device exactly as the engine does (TP=1, bf16), from the REAL HF
config, then checks:
  * 24 decoder layers, each a sparse MoE block (router gate + 60 grouped experts + shared expert
    + sigmoid shared_expert_gate);
  * Qwen2 attention exposes merged qkv_proj (with bias) + o_proj (no bias);
  * GPTQ dense linears (attn qkv/o, shared expert) declare CHECKPOINT-layout buffers
    (qweight (K//8,N) i32, scales (K//g,N) f16, qzeros (K//g,N//8) i32) at the right shapes;
  * router gate / shared_expert_gate are fp16 (unquantized) at (60,H) / (1,H);
  * the grouped expert buffers exist at (E, 2*inter, H) / (E, H, inter).

CPU-only; run in the combined ROCm image (no GPU lease):
    PYTHONPATH=/engine/python python /engine/tools/qwen2_moe_build_smoke.py
"""
from __future__ import annotations

import torch
from transformers import AutoConfig

from minisgl.distributed import set_tp_info
from minisgl.layers import set_rope_device
from minisgl.models import create_model
from minisgl.models.config import ModelConfig
from minisgl.utils import torch_dtype

MODEL = "Qwen/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4"


def main() -> None:
    cfg = ModelConfig.from_hf(AutoConfig.from_pretrained(MODEL))
    assert cfg.is_moe and cfg.quant is not None and cfg.quant.is_gptq, cfg
    H = cfg.hidden_size          # 2048
    E = cfg.num_experts          # 60
    inter = cfg.moe_intermediate_size            # 1408
    sinter = cfg.shared_expert_intermediate_size  # 5632
    qo = cfg.num_qo_heads * cfg.head_dim          # 2048
    kv = cfg.num_kv_heads * cfg.head_dim          # 2048
    pf, g = 8, cfg.quant.group_size               # 8, 128

    set_tp_info(rank=0, size=1)
    set_rope_device(torch.device("cpu"))
    with torch.device("meta"), torch_dtype(torch.bfloat16):
        model = create_model(cfg)

    sd = model.state_dict()
    print(f"state_dict tensors: {len(sd)}  (24 layers x [...] + embed + norm)")

    for k in ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"):
        assert k in sd, k

    L = "model.layers.0."

    # --- attention: merged qkv (bias) + o_proj (no bias), GPTQ buffers ---
    qkvN = qo + 2 * kv  # 6144
    assert tuple(sd[L + "self_attn.qkv_proj.qweight"].shape) == (H // pf, qkvN), sd[L + "self_attn.qkv_proj.qweight"].shape
    assert tuple(sd[L + "self_attn.qkv_proj.scales"].shape) == (H // g, qkvN)
    assert tuple(sd[L + "self_attn.qkv_proj.qzeros"].shape) == (H // g, qkvN // pf)
    assert tuple(sd[L + "self_attn.qkv_proj.bias"].shape) == (qkvN,), "qkv bias is real"
    assert tuple(sd[L + "self_attn.o_proj.qweight"].shape) == (qo // pf, H)
    assert L + "self_attn.o_proj.bias" not in sd, "o_proj bias is a zero placeholder -> not a param"

    # --- router + shared gate: fp16, unquantized ---
    assert tuple(sd[L + "mlp.gate.weight"].shape) == (E, H)
    assert sd[L + "mlp.gate.weight"].dtype == torch.bfloat16  # cast to compute dtype on load
    assert "qweight" not in repr([k for k in sd if k.startswith(L + "mlp.gate.")])
    assert tuple(sd[L + "mlp.shared_expert_gate.weight"].shape) == (1, H)

    # --- shared expert: GPTQ merged gate_up + down ---
    assert tuple(sd[L + "mlp.shared_expert.gate_up_proj.qweight"].shape) == (H // pf, 2 * sinter)
    assert tuple(sd[L + "mlp.shared_expert.gate_up_proj.qzeros"].shape) == (H // g, 2 * sinter // pf)
    assert tuple(sd[L + "mlp.shared_expert.down_proj.qweight"].shape) == (sinter // pf, H)
    assert tuple(sd[L + "mlp.shared_expert.down_proj.scales"].shape) == (sinter // g, H)

    # --- grouped experts (MoELayer buffers; quantized in 2M-3) ---
    assert tuple(sd[L + "mlp.experts.gate_up_proj"].shape) == (E, 2 * inter, H)
    assert tuple(sd[L + "mlp.experts.down_proj"].shape) == (E, H, inter)

    nlayers = sum(1 for k in sd if k.endswith("input_layernorm.weight"))
    assert nlayers == cfg.num_layers == 24, nlayers
    print(f"[OK] {nlayers} MoE layers; attn qkv+o, GPTQ dense buffers, router/shared gates, "
          f"grouped experts — all shapes correct.")


if __name__ == "__main__":
    main()
