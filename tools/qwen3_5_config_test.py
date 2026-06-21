"""Phase 3d-0 — CPU parity check for the Qwen3.5 GDN-hybrid config parse.

Asserts that ``ModelConfig.from_hf`` extracts the linear-attention (GDN) dims, the
layer interleave (``layer_types``), and the partial-rotary geometry from the real
Qwen3.5-4B config, AND that a dense model is left untouched (no GDN fields, full
rotary). No GPU, no weights — pure config parsing.

Two run modes:
  * In a healthy env (the combined ROCm image): ``PYTHONPATH=python python
    tools/qwen3_5_config_test.py`` — imports minisgl + transformers normally and
    parses the cached Qwen/Qwen3.5-4B config via AutoConfig.
  * On a host whose torch/transformers/triton is broken: pass ``--standalone`` to
    load ONLY models/config.py with stubbed deps and a hand-built config object
    (still exercises the real from_hf logic). Used during 3d-0 bring-up.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_CFG = (
    "/home/pat/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/"
    "snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/config.json"
)


def _load_model_config_cls(standalone: bool):
    if not standalone:
        from minisgl.models.config import ModelConfig  # type: ignore

        return ModelConfig

    # --- standalone: load models/config.py with the heavy imports stubbed out ---
    import importlib.util
    import types

    tf = types.ModuleType("transformers")
    tf.PretrainedConfig = type("PretrainedConfig", (), {})
    sys.modules["transformers"] = tf
    for name in ("minisgl", "minisgl.quant", "minisgl.quant.config"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["minisgl.quant.config"].QuantConfig = type(
        "QuantConfig", (), {"from_hf": staticmethod(lambda config: None)}
    )
    path = Path(__file__).resolve().parent.parent / "python/minisgl/models/config.py"
    spec = importlib.util.spec_from_file_location("mc_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mc_under_test"] = mod  # dataclass needs the module registered
    spec.loader.exec_module(mod)
    return mod.ModelConfig


def _hybrid_config(standalone: bool):
    if not standalone:
        from transformers import AutoConfig  # type: ignore

        return AutoConfig.from_pretrained("Qwen/Qwen3.5-4B", trust_remote_code=False)

    class Stub:
        def __init__(self, d):
            self.__dict__.update(d)

    raw = json.load(open(_CFG))
    top = Stub(raw)
    top.text_config = Stub(raw["text_config"])
    top.text_config.architectures = None  # promoted from top by from_hf
    return top


def _dense_config(standalone: bool):
    fields = dict(
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=8,
        hidden_size=2048,
        vocab_size=151936,
        intermediate_size=6144,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        max_position_embeddings=40960,
        model_type="qwen3",
        architectures=["Qwen3ForCausalLM"],
        rope_theta=1000000.0,
        head_dim=128,
        tie_word_embeddings=True,
    )
    if not standalone:
        from transformers import Qwen3Config  # type: ignore

        return Qwen3Config(**fields)

    class Stub:
        def __init__(self, d):
            self.__dict__.update(d)

    return Stub(fields)


def main() -> None:
    standalone = "--standalone" in sys.argv
    ModelConfig = _load_model_config_cls(standalone)

    m = ModelConfig.from_hf(_hybrid_config(standalone))
    full_attn = [i for i in range(m.num_layers) if i not in set(m.gdn_layer_ids)]
    print(
        f"Qwen3.5-4B: layers={m.num_layers} gdn={m.num_gdn_layers} full={full_attn} "
        f"conv_dim={m.gdn_conv_dim} rotary_dim={m.rotary_config.rotary_dim}"
    )
    assert m.is_gdn_hybrid
    assert (m.num_layers, m.num_gdn_layers) == (32, 24)
    assert (m.linear_num_key_heads, m.linear_num_value_heads) == (16, 32)
    assert (m.linear_key_head_dim, m.linear_value_head_dim, m.linear_conv_kernel_dim) == (128, 128, 4)
    assert m.gdn_conv_dim == 8192
    assert (m.rotary_config.rotary_dim, m.rotary_config.head_dim) == (64, 256)
    assert int(m.rotary_config.base) == 10000000
    assert full_attn == [3, 7, 11, 15, 19, 23, 27, 31]
    assert m.tie_word_embeddings is True
    assert m.intermediate_size == 9216 and m.hidden_act == "silu" and m.vocab_size == 248320
    assert m.architectures == ["Qwen3_5ForConditionalGeneration"]
    print("[OK] Qwen3.5-4B GDN-hybrid parse asserts pass.")

    d = ModelConfig.from_hf(_dense_config(standalone))
    assert not d.is_gdn_hybrid and d.layer_types is None
    assert d.rotary_config.rotary_dim == d.head_dim
    assert d.linear_num_key_heads is None and d.num_gdn_layers == 0
    print("[OK] dense Qwen3 regression: non-GDN, full rotary, GDN fields None.")


if __name__ == "__main__":
    main()
