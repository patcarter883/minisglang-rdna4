"""Phase 3d-4 — compare minisgl's LOADED weight VALUES to HF for layer 0 (GDN) + a full-attn layer.

3d-3 verified keys/shapes/dtypes from headers only. This loads the real tensors via minisgl's
`load_weight` and via HF, and diffs them element-wise — catching a wrong concat order / transpose
/ misrouted tensor that header checks cannot see. GPU via the lease.
"""
from __future__ import annotations

import torch
from minisgl.distributed import set_tp_info
from minisgl.models.weight import load_weight
from transformers import AutoModelForCausalLM

MODEL = "Qwen/Qwen3.5-4B"
DEV = torch.device("cpu")


def _cmp(tag, a, b):
    a, b = a.float(), b.float()
    if a.shape != b.shape:
        print(f"  {tag}: SHAPE MISMATCH ms{tuple(a.shape)} hf{tuple(b.shape)}")
        return
    rel = ((a - b).norm() / (b.norm() + 1e-9)).item()
    cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
    flag = "  <-- MISMATCH" if rel > 1e-2 else ""
    print(f"  {tag}: rel={rel:.3e} cos={cos:.6f}{flag}")


def main() -> None:
    set_tp_info(rank=0, size=1)
    ms = {k: v for k, v in load_weight(MODEL, DEV)}
    print(f"minisgl loaded {len(ms)} tensors")

    hf = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cpu")
    h = hf.model.language_model if hasattr(hf.model, "language_model") else hf.model
    L0 = h.layers[0].linear_attn
    p = "model.layers.0."
    print("== layer 0 GDN ==")
    _cmp("in_proj_qkvz", ms[p + "linear_attn.in_proj_qkvz.weight"],
         torch.cat([L0.in_proj_qkv.weight, L0.in_proj_z.weight], dim=0))
    _cmp("in_proj_ba", ms[p + "linear_attn.in_proj_ba.weight"],
         torch.cat([L0.in_proj_b.weight, L0.in_proj_a.weight], dim=0))
    _cmp("conv1d_weight", ms[p + "linear_attn.conv1d_weight"], L0.conv1d.weight)
    _cmp("A_log", ms[p + "linear_attn.A_log"], L0.A_log)
    _cmp("dt_bias", ms[p + "linear_attn.dt_bias"], L0.dt_bias)
    _cmp("norm.weight", ms[p + "linear_attn.norm.weight"], L0.norm.weight)
    _cmp("out_proj", ms[p + "linear_attn.out_proj.weight"], L0.out_proj.weight)
    _cmp("input_layernorm", ms[p + "input_layernorm.weight"], h.layers[0].input_layernorm.weight)
    _cmp("post_attn_ln", ms[p + "post_attention_layernorm.weight"], h.layers[0].post_attention_layernorm.weight)
    _cmp("mlp.gate_up", ms[p + "mlp.gate_up_proj.weight"],
         torch.cat([h.layers[0].mlp.gate_proj.weight, h.layers[0].mlp.up_proj.weight], dim=0))
    _cmp("mlp.down", ms[p + "mlp.down_proj.weight"], h.layers[0].mlp.down_proj.weight)

    print("== layer 3 full-attn ==")
    A = h.layers[3].self_attn
    q = "model.layers.3.self_attn."
    _cmp("q_proj", ms[q + "q_proj.weight"], A.q_proj.weight)
    _cmp("k_proj", ms[q + "k_proj.weight"], A.k_proj.weight)
    _cmp("v_proj", ms[q + "v_proj.weight"], A.v_proj.weight)
    _cmp("o_proj", ms[q + "o_proj.weight"], A.o_proj.weight)
    _cmp("q_norm", ms[q + "q_norm.weight"], A.q_norm.weight)
    _cmp("k_norm", ms[q + "k_norm.weight"], A.k_norm.weight)

    print("== embed / final norm ==")
    _cmp("embed_tokens", ms["model.embed_tokens.weight"], h.embed_tokens.weight)
    _cmp("model.norm", ms["model.norm.weight"], h.norm.weight)


if __name__ == "__main__":
    main()
