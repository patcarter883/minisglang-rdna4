"""`quant_method: "modelopt"` (NVFP4) -> QuantConfig. Bring-up plan T1.1.

Reads the REAL checked-in `config.json` of `RadixArk/Qwen3.8-Flash-Next-NVFP4` through the same
`cached_load_hf_config` -> `QuantConfig.from_hf` path the engine uses, and asserts the three things
that fail SILENTLY if the modelopt header is mis-normalized:

  * `is_nvfp4` — the format lives in `quant_algo: "NVFP4"`, not in a `format` key. Miss it and
    `ct_format` is None, `weight_is_e2m1` becomes True instead, and every routed expert routes into
    the MXFP4 W4A8 kernel: group-32 strides over group-16 scales, plus an orphaned `weight_scale_2`.
  * `group_size == 16` — with no `config_groups` translation the compressed-tensors branch defaults
    to 32, which mis-strides every block scale (a real number, from the wrong 16 elements).
  * the `ignore` list — modelopt writes fnmatch GLOBS (`*.self_attn.*`, `*hyper_connection*`).
    Handed to the historical substring matcher they match NOTHING, so the ignore list evaporates and
    the bf16 attention / GDN / hyper-connection / shared-expert / PLE / gate modules all build
    QUANTIZED against tensors the checkpoint ships unpacked. This is the whole reason the list is
    checked per-module below rather than just counted.

Also asserts the normalization is CONFINED to modelopt: an AWQ header and a compressed-tensors
header parse byte-identically to before.

CPU only — no GPU, no weights, no lease. Run in the serve image (host torch does not import):

    docker run --rm -v <worktree>:/engine --entrypoint bash minisgl-rdna4:m1b-20260903 -lc \
      'PYTHONPATH=/engine/python:/opt/kernels python /engine/tests/qwen4exp_quant_test.py'
"""

from __future__ import annotations

import copy
import json
import os
import sys
from types import SimpleNamespace

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "qwen4exp")
#: The live checkpoint, when it is on this box. The fixture is the durable copy; if both exist they
#: must agree, or the fixture has drifted from what we actually serve.
LIVE_CONFIG = "/home/pat/.cache/hf-q4e/config.json"

_failures = 0


def check(name: str, got, want) -> None:
    global _failures
    ok = got == want
    _failures += not ok
    print(f"  {'ok  ' if ok else 'FAIL'} {name:52s} got={got!r:<26} want={want!r}")


def raw_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def from_dict(cfg: dict):
    """`QuantConfig.from_hf` takes an object with attributes (an HF PretrainedConfig). A namespace
    over the raw JSON is the same surface for the two keys it reads, and keeps this test free of a
    transformers version's opinion about an architecture it does not know."""
    from minisgl.quant.config import QuantConfig

    return QuantConfig.from_hf(SimpleNamespace(**cfg))


def main() -> int:
    global _failures
    from minisgl.quant.config import QuantConfig

    cfg_path = os.path.join(FIXTURES, "config.json")
    cfg = raw_config(cfg_path)

    print(f"[0] fixture vs the live checkpoint ({LIVE_CONFIG})")
    if os.path.exists(LIVE_CONFIG):
        check("fixture config.json == live config.json", raw_config(LIVE_CONFIG), cfg)
    else:
        print("       (live checkpoint not on this box — fixture only)")

    print("\n[1] the modelopt header parses at all")
    q = from_dict(cfg)
    check("from_hf(...) is not None", q is not None, True)
    if q is None:
        print("\nFAIL (modelopt arm missing)")
        return 1
    check("method (normalized)", q.method, "compressed-tensors")
    check("is_compressed_tensors", q.is_compressed_tensors, True)
    check("ct_format", q.ct_format, "nvfp4-pack-quantized")
    check("is_nvfp4", q.is_nvfp4, True)
    check("bits", q.bits, 4)
    check("group_size", q.group_size, 16)
    check("weight_type", q.weight_type, "float")
    check("sym", q.sym, True)
    # NVFP4 is the *only* float-4bit property that may be True: weight_is_e2m1 (MXFP4) excludes it,
    # and is_int4 is an integer-family predicate. Both flipping the wrong way picks a wrong kernel.
    check("weight_is_e2m1 (MXFP4) must be False", q.weight_is_e2m1, False)
    check("is_int4 must be False", q.is_int4, False)
    check("is_fp8_w8a8 must be False", q.is_fp8_w8a8, False)
    check("is_awq / is_gptq / is_rxf", (q.is_awq, q.is_gptq, q.is_rxf), (False, False, False))
    # One config_group -> single-format; the scalars above ARE the whole story and `for_module`
    # degenerates to "self, unless ignored".
    check("ct_groups (single group -> empty)", q.ct_groups, ())

    print("\n[2] the ignore list round-trips (one entry in, one matcher out, same order)")
    raw_ignore = tuple(cfg["quantization_config"]["ignore"])
    check("entry count preserved", len(q.ignore), len(raw_ignore))
    print("       raw glob                       ->  stored entry")
    for raw, stored in zip(raw_ignore, q.ignore):
        print(f"       {raw:30s} ->  {stored}")
    # A glob must NOT survive as a plain substring: `'*.self_attn.*' in name` is False for every
    # module name there is, which is the silent evaporation this whole arm exists to prevent.
    globbed = [p for p in raw_ignore if any(c in p for c in "*?[")]
    check("globs present in the checkpoint", len(globbed), 10)
    check(
        "every glob stored as a regex, not a substring",
        all(s.startswith("re:") for r, s in zip(raw_ignore, q.ignore) if r in globbed),
        True,
    )
    check(
        "non-glob entries stored verbatim (minus the language_model de-wrap)",
        [s for r, s in zip(raw_ignore, q.ignore) if r not in globbed],
        ["model.embed_tokens", "model.embed_tokens", "lm_head"],
    )

    print("\n[3] per-module verdict on REAL checkpoint module names")
    # The routed experts are the ONLY quantized modules in this checkpoint (73,728 projections);
    # everything else is bf16 and named by an ignore glob. Names are in the loader's de-wrapped
    # space (`model.language_model.` -> `model.`), which is what `is_module_quantized` is asked.
    quantized = [
        "model.layers.0.mlp.experts.0.gate_proj",
        "model.layers.0.mlp.experts.511.up_proj",
        "model.layers.47.mlp.experts.310.down_proj",
    ]
    unquantized = [
        # full attention (12 layers) + its QSA indexer
        "model.layers.3.self_attn.q_proj",
        "model.layers.47.self_attn.o_proj",
        "model.layers.3.self_attn.indexer.index_qk_proj",
        # GDN (36 layers)
        "model.layers.0.linear_attn.in_proj_qkv",
        "model.layers.0.linear_attn.out_proj",
        # hyper-connections: per layer and the top-level mixer that replaces the final norm
        "model.layers.0.attn_hyper_connection.input_mix_weight_down",
        "model.layers.0.mlp_hyper_connection.block_inject_weight",
        "model.hyper_connection_mixer.input_mix_weight_up",
        # router gate, shared expert, shared-expert gate
        "model.layers.0.mlp.gate",
        "model.layers.0.mlp.shared_expert.down_proj",
        "model.layers.0.mlp.shared_expert_gate",
        # PLE block
        "model.layers.1.ple.key_proj",
        "model.layers.1.ple.value_proj",
        # embeddings / head / the unimplemented MTP head
        "model.embed_tokens",
        "lm_head",
        "mtp.layers.0.self_attn.q_proj",
        "model.mtp.layers.0.mlp.experts.gate_up_proj",
    ]
    for name in quantized:
        check(f"QUANT   {name}", q.for_module(name) is q, True)
    for name in unquantized:
        check(f"bf16    {name}", q.for_module(name), None)

    print("\n[4] the normalization is confined to modelopt (no regression elsewhere)")
    ct = from_dict(raw_config(os.path.join(FIXTURES, "config_ct_variant.json")))
    check("compressed-tensors variant still parses", ct is not None and ct.is_compressed_tensors, True)
    awq = from_dict(
        {
            "quantization_config": {
                "quant_method": "awq",
                "bits": 4,
                "group_size": 128,
                "zero_point": True,
                "modules_to_not_convert": ["model.language_model.lm_head"],
            }
        }
    )
    check("awq method", awq.method, "awq")
    check("awq group_size", awq.group_size, 128)
    check("awq sym", awq.sym, False)
    check("awq ignore de-wrapped, NOT regex-ified", awq.ignore, ("model.lm_head",))
    check("no quantization_config -> None", from_dict({"model_type": "llama"}), None)

    print("\n[5] an unreadable modelopt algo must be REFUSED, not guessed")
    bad = copy.deepcopy(cfg)
    bad["quantization_config"]["quant_algo"] = "W4A8_AWQ_BETA"
    check("unknown quant_algo -> None", from_dict(bad), None)
    # ...and that None is what ModelConfig turns into `unparsed_quant_method`, i.e. a named
    # condition rather than a checkpoint that looks unquantized.

    print("\n[6] a modelopt header with quant_algo ONLY (no config_groups) keeps the NVFP4 block")
    bare = {
        "quantization_config": {
            "quant_method": "modelopt",
            "quant_algo": "NVFP4",
            "ignore": ["lm_head"],
        }
    }
    qb = from_dict(bare)
    check("bare is_nvfp4", qb.is_nvfp4, True)
    check("bare group_size (16, not the CT default 32)", qb.group_size, 16)
    check("bare bits", qb.bits, 4)
    check("bare weight_type", qb.weight_type, "float")
    check("bare quantizes a normal module", qb.for_module("model.layers.0.mlp.gate_proj") is qb, True)
    check("bare honours its ignore", qb.for_module("lm_head"), None)

    print("\n[7] `ignore` stays ADVISORY when the checkpoint's own tensors are known")
    # `ckpt_quantized` (from the shipped tensor index) overrides the list entirely — the container
    # -entry ambiguity that rule exists for is unchanged by this arm.
    import dataclasses

    q2 = dataclasses.replace(q, ckpt_quantized=frozenset({"model.layers.0.mlp.experts.0.gate_proj"}))
    check("in ckpt_quantized -> quantized", q2.for_module("model.layers.0.mlp.experts.0.gate_proj") is q2, True)
    check("not in ckpt_quantized -> bf16", q2.for_module("model.layers.0.mlp.experts.1.gate_proj"), None)

    assert QuantConfig  # imported for the dataclasses.replace above / keeps the import honest
    print(f"\n{'PASS' if not _failures else f'FAIL ({_failures} checks)'}")
    return 1 if _failures else 0


if __name__ == "__main__":
    sys.exit(main())
