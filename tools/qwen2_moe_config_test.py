#!/usr/bin/env python
"""Phase 2M-0 — CPU config test for Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4 (qwen2_moe + GPTQ-Int4).

Parses the real HF config via ModelConfig.from_hf and QuantConfig.from_hf and asserts the MoE
dims, the shared-expert dim, and the GPTQ quant fields. Also regression-checks that a dense
(non-MoE, non-GPTQ) config still parses with the new field defaulted.

  * In the combined image: `python tools/qwen2_moe_config_test.py` (uses transformers AutoConfig).
  * On a host with broken torch/transformers: `--standalone` loads models/config.py + the REAL
    quant/config.py (both torch-free) with transformers stubbed, and reads config.json directly.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

MOE = "Qwen/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4"
DENSE = "Qwen/Qwen3-0.6B"
_HUB = Path.home() / ".cache/huggingface/hub"


def _model_config_cls(standalone: bool):
    if not standalone:
        sys.path.insert(0, "python")
        from minisgl.models.config import ModelConfig  # type: ignore

        return ModelConfig
    import importlib.util
    import types

    tf = types.ModuleType("transformers")
    tf.PretrainedConfig = type("PretrainedConfig", (), {})
    sys.modules["transformers"] = tf
    root = Path(__file__).resolve().parent.parent / "python/minisgl"
    # Load the REAL QuantConfig (torch-free) so GPTQ recognition is genuinely exercised.
    for mod_name in ("minisgl", "minisgl.quant"):
        sys.modules.setdefault(mod_name, types.ModuleType(mod_name))
    spec_q = importlib.util.spec_from_file_location(
        "minisgl.quant.config", root / "quant/config.py"
    )
    qmod = importlib.util.module_from_spec(spec_q)
    sys.modules["minisgl.quant.config"] = qmod
    spec_q.loader.exec_module(qmod)
    spec = importlib.util.spec_from_file_location("mc_under_test", root / "models/config.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mc_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod.ModelConfig


def _hf_config(repo: str, standalone: bool):
    if not standalone:
        from transformers import AutoConfig  # type: ignore

        return AutoConfig.from_pretrained(repo)

    cfg_path = glob.glob(str(_HUB / f"models--{repo.replace('/', '--')}/snapshots/*/config.json"))[0]
    raw = json.load(open(cfg_path))

    class Stub:
        def __init__(self, d):
            self.__dict__.update(d)

    return Stub(raw)


def check_moe(ModelConfig, standalone) -> None:
    mc = ModelConfig.from_hf(_hf_config(MOE, standalone))
    print(f"[moe] model_type={mc.model_type} is_moe={mc.is_moe} arch={mc.architectures}")
    assert mc.is_moe, "qwen2_moe must be detected as MoE"
    assert mc.model_type == "qwen2_moe"
    assert mc.num_layers == 24, mc.num_layers
    assert mc.hidden_size == 2048
    assert mc.num_experts == 60, mc.num_experts
    assert mc.num_experts_per_tok == 4, mc.num_experts_per_tok
    assert mc.moe_intermediate_size == 1408, mc.moe_intermediate_size
    assert mc.shared_expert_intermediate_size == 5632, mc.shared_expert_intermediate_size
    assert mc.norm_topk_prob is False, mc.norm_topk_prob
    assert mc.num_qo_heads == 16 and mc.num_kv_heads == 16, (mc.num_qo_heads, mc.num_kv_heads)
    assert mc.head_dim == 128, mc.head_dim
    # quant
    q = mc.quant
    assert q is not None and q.is_gptq, q
    assert q.method == "gptq" and q.bits == 4 and q.group_size == 128, q
    assert q.sym is True, q.sym
    assert q.desc_act is False, q.desc_act
    print(f"[moe] quant={q}")
    print("[moe] PASS")


def check_dense(ModelConfig, standalone) -> None:
    mc = ModelConfig.from_hf(_hf_config(DENSE, standalone))
    assert not mc.is_moe, "dense Qwen3 must not be MoE"
    assert mc.shared_expert_intermediate_size == 0, mc.shared_expert_intermediate_size
    assert mc.quant is None, mc.quant
    assert not mc.is_gdn_hybrid
    print("[dense] PASS (regression: shared_expert=0, quant=None)")


if __name__ == "__main__":
    standalone = "--standalone" in sys.argv
    ModelConfig = _model_config_cls(standalone)
    check_moe(ModelConfig, standalone)
    check_dense(ModelConfig, standalone)
    print("\nALL PASS")
