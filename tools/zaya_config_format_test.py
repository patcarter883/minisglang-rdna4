"""Regression test for the ZAYA config-format detector (Megatron export vs HF release).

WHY THIS EXISTS. minisgl models a ZAYA block as TWO alternating layers (even = CCA attention,
odd = MoE), so `num_layers` must come out at 2x the HF release's block count and 1x the Megatron
export's already-split count. The original detector asked "does the config carry `layer_types`?",
documented as "the Megatron config lacks it". transformers >= 5.13 SYNTHESIZES `layer_types` for
any config carrying `sliding_window` (the Megatron export ships `sliding_window: null`), so the
detector fired on the Megatron export, doubled 80 -> 160, and the loader died on
`model.layers.80.input_norm.weight` before a single forward ran. ZAYA simply did not boot.

The fix detects on MEGATRON-ONLY spellings, because the synthesis only runs one way: the HF-format
config has no `moe_router_topk`/`ffn_hidden_size`/`zaya_mlp_expansion`/`num_query_groups`/
`norm_epsilon` even after transformers fills in class defaults, whereas the Megatron export's loaded
config DOES report `num_experts_per_tok`, `moe_intermediate_size`, `router_hidden_size` and
`rms_norm_eps` that its config.json never contained. This test asserts BOTH halves of that claim
against the real checkpoints, so a future transformers bump that starts synthesizing a Megatron
spelling fails here loudly instead of at weight-load time.

CPU ONLY — no GPU, no lease. Run inside the serve image (host python cannot import torch):

    docker run --rm -v /home/pat/models:/models:ro -v <tree>:/engine -w /engine \
      -e PYTHONPATH=/engine/python --entrypoint bash minisgl-rdna4:<tag> \
      -lc 'python3 tools/zaya_config_format_test.py'
"""

from __future__ import annotations

import json
import os
import sys

from transformers import AutoConfig, PretrainedConfig

from minisgl.models.config import ModelConfig

MEGATRON = os.environ.get("ZAYA_MEGATRON", "/models/ZAYA1-8B-fp8")
HF = os.environ.get("ZAYA_HF", "/models/ZAYA1-8B-MXFP4")

# The keys the detector relies on being un-synthesizable. Kept in sync with config.py by assertion,
# not by comment: if config.py's tuple changes, this test tests the wrong thing silently.
MEGATRON_ONLY = ("moe_router_topk", "ffn_hidden_size", "zaya_mlp_expansion",
                 "num_query_groups", "norm_epsilon")

_fails: list[str] = []


def check(cond: bool, what: str, detail: str = "") -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {what}{(' — ' + detail) if detail else ''}")
    if not cond:
        _fails.append(what)


def main() -> int:
    print("== detector key set matches config.py ==")
    src = open(os.path.join(os.path.dirname(__file__), "..", "python", "minisgl", "models",
                            "config.py")).read()
    for k in MEGATRON_ONLY:
        check(f'"{k}"' in src, f"config.py still keys on {k}")

    for label, path, blocks in ((f"MEGATRON export {MEGATRON}", MEGATRON, None),
                                (f"HF release   {HF}", HF, None)):
        if not os.path.isdir(path):
            print(f"== SKIP (not on disk): {path} ==")
            continue
        print(f"== {label} ==")
        raw = json.load(open(os.path.join(path, "config.json")))
        cfg = AutoConfig.from_pretrained(path)
        n_file = int(raw["num_hidden_layers"])
        is_megatron = any(k in raw for k in MEGATRON_ONLY)
        expect = n_file if is_megatron else n_file * 2
        mc = ModelConfig.from_hf(cfg)
        check(mc.num_layers == expect, f"num_layers == {expect}",
              f"file num_hidden_layers={n_file}, format={'megatron' if is_megatron else 'hf'}, "
              f"got {mc.num_layers}")
        # The mechanism itself, stated as an assertion so a transformers bump breaks HERE.
        for k in MEGATRON_ONLY:
            in_file = k in raw
            seen = getattr(cfg, k, None) is not None
            if is_megatron:
                continue  # presence on the Megatron side is what we detect on; nothing to prove
            check(not seen, f"HF config does not gain `{k}` from class defaults",
                  f"in file={in_file}, getattr sees={seen}")
        if is_megatron:
            # The counter-example that killed the old detector: HF-side names ARE synthesized here.
            synth = [k for k in ("layer_types", "num_experts_per_tok", "moe_intermediate_size",
                                 "router_hidden_size", "rms_norm_eps")
                     if k not in raw and getattr(cfg, k, None) is not None]
            check("layer_types" in synth,
                  "the old `layer_types` detector is genuinely unusable here",
                  f"synthesized-but-absent-from-file: {synth}")
        # Downstream ZAYA fields must survive the branch (a mis-detected format also mis-reads these).
        check(mc.num_experts == 16, "num_experts == 16", str(mc.num_experts))
        check(mc.num_experts_per_tok == 1, "top-k == 1", str(mc.num_experts_per_tok))
        check(mc.zaya_mlp_expansion == 256, "zaya_mlp_expansion == 256",
              str(mc.zaya_mlp_expansion))
        check(mc.num_layers % 2 == 0, "layer count is even (attn/MoE pairs)", str(mc.num_layers))

    # Synthetic guards: a non-ZAYA config must be untouched by the ZAYA branch either way.
    print("== non-ZAYA configs are unaffected ==")
    plain = PretrainedConfig(model_type="llama", num_hidden_layers=32, hidden_size=4096,
                             num_attention_heads=32, num_key_value_heads=8, vocab_size=32000,
                             max_position_embeddings=4096)
    # `norm_epsilon` is a Megatron marker but must not drag a non-CCA config anywhere.
    plain.norm_epsilon = 1e-5
    check(ModelConfig.from_hf(plain).num_layers == 32, "llama + norm_epsilon stays 32 layers")

    print()
    if _fails:
        print(f"FAILED: {len(_fails)} check(s): {_fails}")
        return 1
    print("ZAYA CONFIG-FORMAT DETECTOR GREEN")
    return 0


if __name__ == "__main__":
    sys.exit(main())
