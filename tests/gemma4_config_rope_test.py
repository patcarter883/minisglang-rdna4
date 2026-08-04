"""Gemma4 config-parse + RoPE parity against the HuggingFace reference.

CPU-only: needs no GPU lease and cannot disturb a running serve. Run it inside the serve image
(the host torch install is broken, and the image ships the transformers Gemma4 reference):

    docker run --rm --entrypoint bash \
      -v <worktree>:/wt -v /home/pat/.cache/huggingface:/root/.cache/huggingface \
      -e HF_HUB_OFFLINE=1 minisgl-rdna4:lean \
      -lc 'cd /wt && PYTHONPATH=/wt/python:/opt/kernels python tests/gemma4_config_rope_test.py'

What it is really guarding: Gemma4's full-attention RoPE is `rope_type="proportional"`, which is
NOT the ordinary "rotate a contiguous prefix" partial rope. The reference keeps
`partial*head_dim/2` frequencies, takes their exponent over the FULL head_dim, zero-pads back to
head_dim/2 and rotates full-width — so with head_dim 512 / partial 0.25 the rotated channel pairs
are (i, i+256) for i<64, i.e. the set {0..63} u {256..319}. A prefix-partial rope instead pairs
(i, i+64) over {0..127} and divides the exponent by 128. Both the frequencies and the pairing
differ, and the failure is silent: relative position is scrambled, output stays grammatical but
loses precise reasoning. This test pins the exact tensor.
"""

from __future__ import annotations

import glob
import sys

import torch
from transformers import AutoConfig

from minisgl.layers import rotary as rotary_mod
from minisgl.layers.rotary import get_rope
from minisgl.models.config import ModelConfig

MODEL_GLOB = (
    "/root/.cache/huggingface/hub/"
    "models--cyankiwi--gemma-4-26B-A4B-it-qat-AWQ-INT4/snapshots/*/"
)

# Ground truth read off the checkpoint's own config.json + safetensors header.
EXPECTED = {
    "num_layers": 30,
    "num_qo_heads": 16,
    "head_dim": 512,  # FULL-attention layers (they size the main paged pool)
    "num_kv_heads": 2,
    "swa_head_dim": 256,  # SLIDING layers (their own ring pool)
    "swa_num_kv_heads": 8,
    "num_kv_layers": 5,
    "num_swa_layers": 25,
    "full_attn_layer_ids": [5, 11, 17, 23, 29],
    "num_experts": 128,
    "num_experts_per_tok": 8,
    "intermediate_size": 2112,
    "moe_intermediate_size": 704,
    "attn_softmax_scale": 1.0,
    "attention_k_eq_v": True,
    "final_logit_softcapping": 30.0,
    "sliding_window": 1024,
}


def check(name: str, got, want) -> bool:
    ok = got == want
    print(f"  {'ok ' if ok else 'FAIL'} {name:24s} got={got!r:<28} want={want!r}")
    return ok


def main() -> int:
    matches = glob.glob(MODEL_GLOB)
    if not matches:
        print(f"SKIP: checkpoint not cached under {MODEL_GLOB}")
        return 0
    path = matches[0]
    hf = AutoConfig.from_pretrained(path)
    mc = ModelConfig.from_hf(hf, spec_algorithm="none")

    failures = 0

    print("[1] ModelConfig fields")
    for key, want in EXPECTED.items():
        got = getattr(mc, key)
        if isinstance(want, list):
            got = list(got)
        failures += not check(key, got, want)

    print("\n[2] derived predicates")
    for key, want in (
        ("is_gemma4", True),
        ("is_swa_hybrid", True),
        ("is_moe", True),
        ("has_split_head_dim", True),
        ("is_gdn_hybrid", False),  # must NOT be mistaken for a GDN hybrid
        ("is_mla", False),
        ("is_cca_hybrid", False),
    ):
        failures += not check(key, getattr(mc, key), want)

    print("\n[3] quantization")
    failures += not check("quant.is_int4", mc.quant.is_int4, True)
    failures += not check("quant.group_size", mc.quant.group_size, 32)

    print("\n[4] RoPE parity vs transformers Gemma4TextRotaryEmbedding")
    from transformers.models.gemma4.modeling_gemma4 import (
        Gemma4TextRotaryEmbedding,
        apply_rotary_pos_emb,
    )

    rotary_mod.set_rope_device(torch.device("cpu"))
    text_cfg = hf.text_config
    ref = Gemma4TextRotaryEmbedding(text_cfg, device=torch.device("cpu"))

    torch.manual_seed(0)
    n_tok = 37
    pos = torch.arange(n_tok).unsqueeze(0)

    for layer_type, rc in (
        ("full_attention", mc.rotary_config),
        ("sliding_attention", mc.sliding_rotary_config),
    ):
        head_dim = rc.head_dim
        cos, sin = ref(torch.zeros(1, n_tok, text_cfg.hidden_size), pos, layer_type)
        q = torch.randn(1, n_tok, EXPECTED["num_qo_heads"], head_dim, dtype=torch.float32)
        want = apply_rotary_pos_emb(q, cos, sin, unsqueeze_dim=2)

        scaling = tuple(sorted(rc.scaling.items())) if rc.scaling else None
        rope = get_rope(rc.head_dim, rc.rotary_dim, rc.max_position, rc.base, scaling)
        got = rope.forward_one(pos[0], q[0].reshape(n_tok, -1)).reshape(want.shape)

        err = (got - want).abs().max().item()
        ok = err < 1e-4
        failures += not ok
        print(
            f"  {'ok ' if ok else 'FAIL'} {layer_type:18s} head_dim={head_dim:<4d} "
            f"rotary_dim={rc.rotary_dim:<4d} theta={rc.base:<10.0f} max|err|={err:.3e}"
        )

    print("\n[5] proportional inv_freq shape (the silent-failure guard)")
    # 64 live frequencies then 192 exact zeros; inv_freq[0] == 1.0 and the exponent denominator is
    # the full 512, so inv_freq[63] == 1e6 ** (-126/512).
    full_rope = get_rope(
        mc.rotary_config.head_dim,
        mc.rotary_config.rotary_dim,
        mc.rotary_config.max_position,
        mc.rotary_config.base,
        tuple(sorted(mc.rotary_config.scaling.items())),
    )
    cos_row = full_rope._cos_sin_cache[1][: mc.rotary_config.head_dim // 2]
    inv_freq = torch.acos(cos_row.clamp(-1, 1))  # position 1 -> angle == inv_freq
    n_live = int((inv_freq > 1e-9).sum())
    failures += not check("live frequencies", n_live, 64)
    failures += not check("zero-padded tail", int((inv_freq[64:] == 0).all()), 1)
    want_last = 1e6 ** (-126 / 512)
    got_last = inv_freq[63].item()
    ok = abs(got_last - want_last) < 1e-6
    failures += not ok
    print(
        f"  {'ok ' if ok else 'FAIL'} {'inv_freq[63]':24s} got={got_last:.9f} "
        f"want={want_last:.9f}  (denominator must be head_dim 512, not rotary width 128)"
    )

    print(f"\n{'PASS' if failures == 0 else f'FAIL ({failures} checks)'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
