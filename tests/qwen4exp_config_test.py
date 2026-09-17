"""qwen4_exp (Qwen3.8-Flash-Next) config plumbing + registration. Bring-up plan T0.4 / T2.3.

Reads the REAL checked-in `config.json` of `RadixArk/Qwen3.8-Flash-Next-NVFP4` (and a second,
differently-quantized qwen4_exp config) through the same `cached_load_hf_config` ->
`ModelConfig.from_hf` path the engine uses, and asserts the architecture facts that fail SILENTLY if
they are wrong:

  * `num_experts_per_tok` — a 0 here routes nothing and is the classic silent-zero spelling trap;
  * `norm_topk_prob` — absent from the file, so the generic `False` default would leave the top-10
    router weights summing to < 1 (mis-scaled experts, degenerate text, no error);
  * the layer schedule — 36 linear + 12 full, and `num_kv_layers == 12` sizes the paged KV pool;
  * `ple_layer_ids` — 1-BASED in the file; landing the PLE block on index 2 instead of 1 loads
    cleanly and only degrades quality;
  * `gdn_output_gate` — "sigmoid" here, not the silu default, and it gates all 36 GDN layers;
  * `mtp_num_hidden_layers` — the checkpoint ships an MTP head this engine does not implement, so
    asking for it must RAISE rather than silently serve without speculation.

CPU only — no GPU, no weights. Run in the serve image (host torch does not import):

    docker run --rm -v <worktree>:/engine --entrypoint bash minisgl-rdna4:m1b-20260903 -lc \
      'PYTHONPATH=/engine/python:/opt/kernels python /engine/tests/qwen4exp_config_test.py'
"""

from __future__ import annotations

import os
import sys

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "qwen4exp")

_failures = 0


def check(name: str, got, want) -> None:
    global _failures
    ok = got == want
    _failures += not ok
    print(f"  {'ok  ' if ok else 'FAIL'} {name:42s} got={got!r:<34} want={want!r}")


def load(cfg_dir: str, **kw):
    from minisgl.models.config import ModelConfig
    from minisgl.utils import cached_load_hf_config

    return ModelConfig.from_hf(cached_load_hf_config(cfg_dir), **kw)


def main() -> int:
    global _failures
    import shutil
    import tempfile

    tmp = tempfile.mkdtemp(prefix="qwen4exp-cfg-")
    nvfp4_dir = os.path.join(tmp, "nvfp4")
    ct_dir = os.path.join(tmp, "ct")
    for d, src in ((nvfp4_dir, "config.json"), (ct_dir, "config_ct_variant.json")):
        os.makedirs(d, exist_ok=True)
        shutil.copyfile(os.path.join(FIXTURES, src), os.path.join(d, "config.json"))

    print("[1] RadixArk/Qwen3.8-Flash-Next-NVFP4 config.json -> ModelConfig")
    mc = load(nvfp4_dir, spec_algorithm="none")
    check("architectures[0]", mc.architectures[0], "Qwen4ExpForConditionalGeneration")
    check("model_type", mc.model_type, "qwen4_exp_text")
    check("is_qwen4_exp", mc.is_qwen4_exp, True)
    check("is_gdn_hybrid", mc.is_gdn_hybrid, True)
    check("is_moe", mc.is_moe, True)
    check("is_mla / is_swa / is_cca", (mc.is_mla, mc.is_swa_hybrid, mc.is_cca_hybrid), (False,) * 3)
    check("num_layers", mc.num_layers, 48)
    check("hidden_size", mc.hidden_size, 2560)
    check("vocab_size", mc.vocab_size, 248320)
    check("tie_word_embeddings", mc.tie_word_embeddings, False)
    check("rms_norm_eps", mc.rms_norm_eps, 1e-6)

    print("\n[2] MoE routing (silent-zero / silent-unnormalized traps)")
    check("num_experts", mc.num_experts, 512)
    check("num_experts_per_tok", mc.num_experts_per_tok, 10)
    check("moe_intermediate_size", mc.moe_intermediate_size, 640)
    check("shared_expert_intermediate_size", mc.shared_expert_intermediate_size, 640)
    # NOT in config.json: inherited from Qwen3NextConfig via Qwen4ExpTextConfig. False would
    # silently mis-scale every routed expert.
    check("norm_topk_prob (default True)", mc.norm_topk_prob, True)
    check("first_k_dense_replace", mc.first_k_dense_replace, 0)

    print("\n[3] hybrid layer schedule (idx % 4 == 3 is full attention)")
    check("num_gdn_layers", mc.num_gdn_layers, 36)
    check("gdn_layer_ids", mc.gdn_layer_ids, [i for i in range(48) if i % 4 != 3])
    check("full_attn_layer_ids", mc.full_attn_layer_ids, list(range(3, 48, 4)))
    check("num_kv_layers (paged pool depth)", mc.num_kv_layers, 12)
    check("gdn dims", (mc.linear_num_key_heads, mc.linear_num_value_heads), (16, 48))
    check("gdn head dims", (mc.linear_key_head_dim, mc.linear_value_head_dim), (128, 128))
    check("gdn conv kernel", mc.linear_conv_kernel_dim, 4)
    check("gdn conv_dim", mc.gdn_conv_dim, 2 * 16 * 128 + 48 * 128)
    check("gdn_output_gate", mc.gdn_output_gate, "sigmoid")

    print("\n[4] attention geometry")
    check("num_qo_heads", mc.num_qo_heads, 24)
    check("num_kv_heads", mc.num_kv_heads, 2)
    check("head_dim", mc.head_dim, 256)
    check("rotary_dim (partial 0.25)", mc.rotary_config.rotary_dim, 64)
    check("rope base", mc.rotary_config.base, 10000000)
    check("rope scaling (mrope must NOT leak)", mc.rotary_config.scaling, None)
    check("max_position", mc.rotary_config.max_position, 262144)

    print("\n[5] hyper-connections / PLE / QSA indexer")
    check("hc_count", mc.hc_count, 4)
    check("hc_lowrank", mc.hc_lowrank, 320)
    check("hc_hidden_size (wide stream)", mc.hc_hidden_size, 10240)
    # config.json says [2]; 1-based -> decoder index 1, where `layers.1.ple.*` actually lives.
    check("ple_layer_ids (0-based)", mc.ple_layer_ids, (1,))
    check("ple_embed_dim", mc.ple_embed_dim, 2560)
    check("ple_conv_kernel_size", mc.ple_conv_kernel_size, 4)
    check("ngram_size", mc.ngram_size, 3)
    check("heads_per_ngram", mc.heads_per_ngram, 8)
    check("n-gram hash heads", (mc.ngram_size - 1) * mc.heads_per_ngram, 16)
    check("ple_embed_dim / heads == row width", mc.ple_embed_dim // 16, 160)
    check("split_ngram_parts", mc.split_ngram_parts, 128)
    check("indexer_budget", mc.indexer_budget, 2048)
    check("indexer dims", (mc.indexer_n_heads, mc.indexer_kv_heads, mc.indexer_head_dim), (4, 1, 128))

    print("\n[6] quantization: modelopt/NVFP4 normalizes to compressed-tensors (T1.1 LANDED)")
    # `quant_method: "modelopt"` is rewritten into the compressed-tensors shape at parse time, so
    # every downstream consumer sees one code path. `unparsed_quant_method` must now be None — a
    # non-None value here means the arm stopped matching and the whole body would build FULL
    # PRECISION with no error. The format/ignore semantics are asserted in qwen4exp_quant_test.py.
    check("quant parsed", mc.quant is not None, True)
    check("unparsed_quant_method", mc.unparsed_quant_method, None)
    check("quant.is_nvfp4", mc.quant.is_nvfp4, True)
    check("quant.group_size", mc.quant.group_size, 16)

    print("\n[7] MTP head: shipped by the checkpoint, NOT implemented -> must refuse")
    check("mtp_num_hidden_layers (spec=none)", mc.mtp_num_hidden_layers, 0)
    check("num_nextn_predict_layers (spec=none)", mc.num_nextn_predict_layers, 0)
    raised = None
    try:
        load(nvfp4_dir, spec_algorithm="mtp")
    except NotImplementedError as e:
        raised = str(e)
    check("--spec-algorithm mtp raises", raised is not None, True)
    if raised:
        print(f"       -> {raised.splitlines()[0][:110]}")

    print("\n[8] registration resolves")
    from minisgl.models.register import _MODEL_REGISTRY

    check(
        "registry entry",
        _MODEL_REGISTRY.get("Qwen4ExpForConditionalGeneration"),
        (".qwen4exp", "Qwen4ExpForConditionalGeneration"),
    )
    import importlib

    mod = importlib.import_module("minisgl.models.qwen4exp")
    check("class importable", hasattr(mod, "Qwen4ExpForConditionalGeneration"), True)

    print("\n[9] a SECOND qwen4_exp config (compressed-tensors mxfp4 variant) parses identically")
    mc2 = load(ct_dir, spec_algorithm="none")
    same = (
        "num_layers", "hidden_size", "vocab_size", "num_experts", "num_experts_per_tok",
        "moe_intermediate_size", "shared_expert_intermediate_size", "hc_count", "hc_lowrank",
        "ple_layer_ids", "ngram_size", "indexer_budget", "gdn_output_gate", "norm_topk_prob",
        "num_qo_heads", "num_kv_heads", "head_dim",
    )
    for field in same:
        check(f"{field} (variant)", getattr(mc2, field), getattr(mc, field))
    # This one DOES parse (compressed-tensors), which is exactly why the two are both kept.
    check("variant quant parsed", mc2.quant is not None, True)
    check("variant unparsed_quant_method", mc2.unparsed_quant_method, None)

    shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{'PASS' if not _failures else f'FAIL ({_failures} checks)'}")
    return 1 if _failures else 0


if __name__ == "__main__":
    sys.exit(main())
