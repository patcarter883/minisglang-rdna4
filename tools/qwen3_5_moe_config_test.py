#!/usr/bin/env python
"""Phase 3M-0 — CPU config test for Qwen3.6-35B-A3B (qwen3_5_moe + compressed-tensors int4).

The 35B GDN-hybrid MoE = the 4B qwen3_5 GDN-hybrid attention pattern + a Qwen2-MoE-style sparse
block (256 routed experts top-8 + an always-on shared expert), with ONLY the routed experts
quantized (compressed-tensors pack-quantized, g32, symmetric). Asserts ModelConfig.from_hf +
QuantConfig.from_hf extract the GDN dims, the layer interleave, the MoE dims, partial rotary, and
the compressed-tensors quant; plus the all-MoE `intermediate_size=0` (no dense MLP) path.

  * In the combined image: `python tools/qwen3_5_moe_config_test.py` (transformers AutoConfig).
  * On a host with broken torch/transformers: `--standalone` loads models/config.py + the REAL
    quant/config.py (both torch-free) with transformers stubbed, reading config.json directly.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

MODEL = "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit"
_HUB = Path.home() / ".cache/huggingface/hub"


def _model_config_cls(standalone: bool):
    if not standalone:
        from minisgl.models.config import ModelConfig  # type: ignore

        return ModelConfig
    import importlib.util
    import types

    tf = types.ModuleType("transformers")
    tf.PretrainedConfig = type("PretrainedConfig", (), {})
    sys.modules["transformers"] = tf
    root = Path(__file__).resolve().parent.parent / "python/minisgl"
    for mod_name in ("minisgl", "minisgl.quant"):
        sys.modules.setdefault(mod_name, types.ModuleType(mod_name))
    spec_q = importlib.util.spec_from_file_location("minisgl.quant.config", root / "quant/config.py")
    qmod = importlib.util.module_from_spec(spec_q)
    sys.modules["minisgl.quant.config"] = qmod
    spec_q.loader.exec_module(qmod)
    spec = importlib.util.spec_from_file_location("mc_under_test", root / "models/config.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mc_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod.ModelConfig


def _hf_config(standalone: bool):
    if not standalone:
        from transformers import AutoConfig  # type: ignore

        return AutoConfig.from_pretrained(MODEL, trust_remote_code=False)

    cfg_path = glob.glob(str(_HUB / f"models--{MODEL.replace('/', '--')}/snapshots/*/config.json"))[0]
    raw = json.load(open(cfg_path))

    class Stub:
        def __init__(self, d):
            self.__dict__.update(d)

    top = Stub(raw)
    top.text_config = Stub(raw["text_config"])
    top.text_config.architectures = None  # promoted from top by from_hf
    # quantization_config stays a dict on top; QuantConfig._as_dict handles dict directly.
    return top


def main() -> None:
    standalone = "--standalone" in sys.argv
    ModelConfig = _model_config_cls(standalone)
    m = ModelConfig.from_hf(_hf_config(standalone))

    full_attn = [i for i in range(m.num_layers) if i not in set(m.gdn_layer_ids)]
    q = m.quant
    print(f"qwen3_5_moe: type={m.model_type} layers={m.num_layers} gdn={m.num_gdn_layers} "
          f"experts={m.num_experts}x top{m.num_experts_per_tok} conv_dim={m.gdn_conv_dim} "
          f"rotary_dim={m.rotary_config.rotary_dim} "
          f"quant={q.method}/{q.bits}b/g{q.group_size}/sym={q.sym}/ignore={len(q.ignore)}")

    # --- architecture / GDN-hybrid ---
    # model_type comes from text_config ("qwen3_5_moe_text", same _text suffix as the 4B);
    # registration keys off `architectures`, not model_type.
    assert m.model_type.startswith("qwen3_5_moe") and m.is_moe and m.is_gdn_hybrid, m.model_type
    assert m.architectures == ["Qwen3_5MoeForConditionalGeneration"], m.architectures
    assert (m.num_layers, m.num_gdn_layers) == (40, 30), (m.num_layers, m.num_gdn_layers)
    assert full_attn == [3, 7, 11, 15, 19, 23, 27, 31, 35, 39], full_attn
    assert (m.linear_num_key_heads, m.linear_num_value_heads) == (16, 32)
    assert (m.linear_key_head_dim, m.linear_value_head_dim, m.linear_conv_kernel_dim) == (128, 128, 4)
    assert m.gdn_conv_dim == 8192, m.gdn_conv_dim  # 2*16*128 + 32*128
    assert (m.hidden_size, m.num_qo_heads, m.num_kv_heads, m.head_dim) == (2048, 16, 2, 256)
    assert (m.rotary_config.rotary_dim, m.rotary_config.head_dim) == (64, 256)  # partial 0.25
    assert int(m.rotary_config.base) == 10000000, m.rotary_config.base
    assert m.tie_word_embeddings is False and m.vocab_size == 248320

    # --- MoE dims; no dense MLP (all layers MoE -> intermediate_size 0) ---
    assert (m.num_experts, m.num_experts_per_tok) == (256, 8)
    assert m.moe_intermediate_size == 512 and m.shared_expert_intermediate_size == 512
    assert m.intermediate_size == 0, m.intermediate_size

    # --- compressed-tensors int4 g32 symmetric, with an ignore list (bf16 attn/shared/gates) ---
    assert q is not None and q.is_compressed_tensors, q
    assert q.method == "compressed-tensors" and q.bits == 4 and q.group_size == 32, q
    assert q.sym is True, q.sym
    assert len(q.ignore) > 0, "compressed-tensors must carry an ignore list"
    print(f"[OK] qwen3_5_moe 35B GDN-hybrid MoE parse + compressed-tensors quant "
          f"(ignore={len(q.ignore)} entries) all asserts pass.")


if __name__ == "__main__":
    main()
