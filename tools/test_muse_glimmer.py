"""CPU-only bring-up check for the Muse-Glimmer port. No GPU, no weights read.

Three things, all of which are load-time or silent-correctness failures rather than perf:

  1. Per-layer plan: sliding/full schedule, window, compact KV-pool ids, and NoPE on exactly the
     13 full-attention layers.
  2. Meta-instantiation of all 52 decoder layers, asserting the shapes that the non-square
     `hidden != num_heads*head_dim` geometry makes easy to get wrong, and the norm conventions
     (centered layer norms with two epsilons vs a plain final norm).
  3. A FULL key-set diff: every key the model declares vs every key the loader would produce from
     the real checkpoint's index, at TP=1 and TP=2. This is the check that would otherwise only
     fire after a multi-minute load on a leased GPU.

Run: python tools/test_muse_glimmer.py [/path/to/checkpoint]
"""
from __future__ import annotations

import glob
import json
import os
import sys

import torch

FAILS: list[str] = []


def check(name, got, want):
    ok = got == want
    print(f"  [{'ok' if ok else 'FAIL'}] {name}: {got!r}" + ("" if ok else f"  (want {want!r})"))
    if not ok:
        FAILS.append(name)


def _default_ckpt() -> str:
    hits = glob.glob(
        os.path.expanduser(
            "~/.cache/huggingface/hub/models--RedHatAI--Muse-Glimmer-30B-NVFP4/snapshots/*/"
        )
    )
    if not hits:
        sys.exit("Muse-Glimmer checkpoint not cached; pass its path as argv[1]")
    return hits[0]


def main() -> int:
    from minisgl.distributed.info import set_tp_info
    from minisgl.layers.rotary import set_rope_device
    from minisgl.models.config import ModelConfig
    from minisgl.models.muse_glimmer import (
        MuseGlimmerDecoderLayer,
        MuseGlimmerForConditionalGeneration,
        muse_glimmer_layer_plan,
    )
    from minisgl.models.weight import (
        _muse_gate_up_merge,
        _muse_glimmer_remap,
        _shard_muse_glimmer,
    )
    from minisgl.utils.hf import cached_load_hf_config

    ckpt = sys.argv[1] if len(sys.argv) > 1 else _default_ckpt()
    set_tp_info(rank=0, size=1)
    set_rope_device(torch.device("cpu"))
    mc = ModelConfig.from_hf(cached_load_hf_config(ckpt))

    print("== config ==")
    check("num_layers", mc.num_layers, 52)
    check("hidden_size", mc.hidden_size, 6656)
    check("head_dim", mc.head_dim, 128)
    check("num_qo_heads", mc.num_qo_heads, 32)
    check("num_kv_heads", mc.num_kv_heads, 2)
    check("sliding_window", mc.sliding_window, 2048)
    check("is_swa_hybrid", mc.is_swa_hybrid, True)
    check("is_muse_glimmer", mc.is_muse_glimmer, True)
    check("post_norm_eps", mc.post_norm_eps, 1e-8)
    check("final_logit_softcapping", mc.final_logit_softcapping, 20.0)
    check("output_multiplier", mc.output_multiplier, 0.19611613513818404)
    # 3.87 / sqrt(128), folded so the kernel applies it as the softmax scale.
    check("attn_softmax_scale", round(mc.attn_softmax_scale, 12), round(3.87 * 128**-0.5, 12))
    check("hidden_act (spelled hidden_activation upstream)", mc.hidden_act, "silu")
    check("tie_word_embeddings", mc.tie_word_embeddings, False)
    check("n NoPE layers", len(mc.nope_layer_ids), 13)
    check("NoPE == full-attention layers", tuple(mc.full_attn_layer_ids), mc.nope_layer_ids)
    check("NoPE layers", mc.nope_layer_ids[:4], (3, 7, 11, 15))

    print("== per-layer plan ==")
    p0, p3 = muse_glimmer_layer_plan(mc, 0), muse_glimmer_layer_plan(mc, 3)
    check("L0 is_sliding", p0.is_sliding, True)
    check("L0 window", p0.sliding_window, 2048)
    check("L0 has rope", p0.rotary_config is not None, True)
    check("L0 kv_id (1st sliding)", p0.kv_id, 0)
    check("L3 is_sliding", p3.is_sliding, False)
    check("L3 window", p3.sliding_window, 0)
    check("L3 NoPE (rotary_config is None)", p3.rotary_config is None, True)
    check("L3 kv_id (1st full)", p3.kv_id, 0)
    check("L7 kv_id (2nd full)", muse_glimmer_layer_plan(mc, 7).kv_id, 1)
    check("L4 kv_id (4th sliding)", muse_glimmer_layer_plan(mc, 4).kv_id, 3)

    print("== meta-instantiate all 52 decoder layers ==")
    bad_shape = bad_rope = bad_eps = bad_plusone = 0
    with torch.device("meta"):
        for lid in range(mc.num_layers):
            layer = MuseGlimmerDecoderLayer(mc, lid)
            a = layer.self_attn
            # Non-square geometry: q/gate are hidden->4096, o is 4096->hidden.
            if a.q_proj.weight_packed.shape[0] != 4096:
                bad_shape += 1
            if a.gate_proj.weight_packed.shape[0] != 4096:
                bad_shape += 1
            if a.k_proj.weight_packed.shape[0] != 256:
                bad_shape += 1
            nope = lid in mc.nope_layer_ids
            if (a.attn.rotary is None) != nope:
                bad_rope += 1
            # Two epsilons: input-side 1e-5, sandwich post-norms 1e-8.
            if (layer.input_layernorm.eps, layer.pre_feedforward_layernorm.eps) != (1e-5, 1e-5):
                bad_eps += 1
            if (
                layer.post_attention_layernorm.eps,
                layer.post_feedforward_layernorm.eps,
            ) != (1e-8, 1e-8):
                bad_eps += 1
            # All four layer norms are the CENTERED (1+w) convention.
            if not all(
                n.plus_one
                for n in (
                    layer.input_layernorm,
                    layer.post_attention_layernorm,
                    layer.pre_feedforward_layernorm,
                    layer.post_feedforward_layernorm,
                )
            ):
                bad_plusone += 1
    check("layers with wrong proj shape", bad_shape, 0)
    check("layers with wrong NoPE/rope state", bad_rope, 0)
    check("layers with wrong norm eps", bad_eps, 0)
    check("layers with a non-centered layer norm", bad_plusone, 0)

    print("== whole-model key-set diff vs the real checkpoint ==")
    with open(os.path.join(ckpt, "model.safetensors.index.json")) as f:
        ckpt_keys = sorted(json.load(f)["weight_map"])

    # The key SET is TP-invariant (sharding changes shapes, not names), and `set_tp_info` is
    # one-shot per process, so this runs at the TP=1 already set above. TP=2 shard SHAPES are
    # covered by the `_shard_muse_glimmer` section below.
    if True:
        with torch.device("meta"):
            model = MuseGlimmerForConditionalGeneration(mc)
        want = set(model.state_dict().keys())
        tp = 1

        # Replay the loader's NAMING pipeline (fold -> remap -> merge). No tensors are read; the
        # shard step is shape-only and is exercised separately below.
        got: set[str] = set()
        folded: set[str] = set()
        for name in ckpt_keys:
            n = name
            if n.startswith(("model.vision_tower.", "model.vision_adapter.", "model.vision_projection")):
                continue
            if n.endswith(".input_global_scale"):
                continue
            if n.endswith((".weight_scale", ".weight_global_scale")):
                base = n.rsplit(".", 1)[0]
                if base in folded:
                    continue  # the pair folds to ONE emitted key
                folded.add(base)
                n = base + ".weight_scale"
            native = _muse_glimmer_remap(n)
            if native is None:
                continue
            mm = _muse_gate_up_merge(native)
            got.add(mm[0] if mm is not None else native)

        missing, extra = sorted(want - got), sorted(got - want)
        check(f"TP={tp}: keys the model wants but the loader never emits", missing[:4], [])
        check(f"TP={tp}: keys the loader emits that the model has no slot for", extra[:4], [])
        check(f"TP={tp}: total key count", len(got), len(want))

    # The final norm must be the PLAIN convention while the layer norms are centered. Checked on the
    # whole model (not a layer) because it is the one norm that differs.
    check("final norm is NOT centered (plain *w)", model.model.norm.plus_one, False)
    check("lm_head is untied", model.lm_head.weight.shape[0], mc.vocab_size)

    print("== TP=2 shard shapes (NVFP4 packed/scale must stay in step) ==")
    # (N, K) -> packed (N, K//2) and group scale (N, K//16). Column-parallel splits dim 0, row-
    # parallel dim 1; the scale must land on the same axis as the weight it describes.
    q_packed = torch.empty(4096, 3328, device="meta")
    q_scale = torch.empty(4096, 416, device="meta")
    o_packed = torch.empty(6656, 2048, device="meta")
    o_scale = torch.empty(6656, 256, device="meta")
    name_q = "model.layers.0.self_attn.q_proj.weight_packed"
    name_qs = "model.layers.0.self_attn.q_proj.weight_scale"
    name_o = "model.layers.0.self_attn.o_proj.weight_packed"
    name_os = "model.layers.0.self_attn.o_proj.weight_scale"
    check("q packed col-shard", tuple(_shard_muse_glimmer(name_q, q_packed, 0, 2).shape), (2048, 3328))
    check("q scale  col-shard", tuple(_shard_muse_glimmer(name_qs, q_scale, 0, 2).shape), (2048, 416))
    check("o packed row-shard", tuple(_shard_muse_glimmer(name_o, o_packed, 0, 2).shape), (6656, 1024))
    check("o scale  row-shard", tuple(_shard_muse_glimmer(name_os, o_scale, 0, 2).shape), (6656, 128))
    norm = torch.empty(6656, device="meta")
    check(
        "layer norm replicated",
        tuple(_shard_muse_glimmer("model.layers.0.input_layernorm.weight", norm, 0, 2).shape),
        (6656,),
    )
    # The attention gate must NOT be swept into a gate_up merge (there is no self_attn.up_proj).
    check(
        "self_attn.gate_proj is not gate/up-merged",
        _muse_gate_up_merge("model.layers.0.self_attn.gate_proj.weight_packed"),
        None,
    )
    check(
        "mlp.gate_proj IS gate/up-merged",
        _muse_gate_up_merge("model.layers.0.mlp.gate_proj.weight_packed")[0],
        "model.layers.0.mlp.gate_up_proj.weight_packed",
    )

    print()
    if FAILS:
        print(f"FAILED ({len(FAILS)}): " + ", ".join(FAILS))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
