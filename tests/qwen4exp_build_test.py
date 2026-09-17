"""qwen4_exp meta-device build: the parameter set is EXACTLY the checkpoint's, and every
unimplemented block refuses. Bring-up plan T2.2/T2.3 (structure) + T1.4 at the name level.

Two things are proved here, and they are the whole point of bring-up tranche 1a:

  1. **Key-set parity.** The model built on `meta` declares exactly the keys the weight-name mapper
     produces from the real 296,475-tensor index — same set, no more, no less. `BaseOP.load_state_dict`
     pops every key it declares and raises on leftovers, so this set IS the loader's contract. In
     particular it pins the two structural facts that are easy to get wrong and impossible to notice:
       * there is **no** `model.norm.weight` (the checkpoint ships none; `hyper_connection_mixer`
         plays that role), and
       * there are **no** per-layer `input_layernorm` / `post_attention_layernorm` (the
         hyper-connection's own `hc_norm` is the pre-block norm).
     Adding either would fail here rather than at load time on a 135 GB checkpoint.

  2. **Refusal.** The hyper-connection mix/combine and the QSA indexer are declared but compute
     nothing; each raises `NotImplementedError` naming what is missing, and so does the model's
     `forward`. A model that refuses is far better than one that runs and is subtly wrong.
     As of tranche 1b `GroupedRMSNorm` and `Qwen4ExpPLE` have moved out of that set — they compute,
     and their numerics are pinned against the reference implementation in
     `tests/qwen4exp_ple_test.py` (which needs a real device) and `tests/qwen4exp_ple_hash_test.py`.
     Section [5b] here only checks that they no longer refuse, and that `PLE.forward` with no
     host-staged batch is a loud structural error rather than a silent skip.

Shapes are NOT checked against the checkpoint here — the 84 GB body is not downloaded, so a shape
table would be a claim, not a measurement. Shape/dtype parity is plan T1.4 with the weights present.

CPU/meta only — no GPU, no weights. Run in the serve image:

    docker run --rm -v <worktree>:/engine --entrypoint bash minisgl-rdna4:m1b-20260903 -lc \
      'PYTHONPATH=/engine/python:/opt/kernels python /engine/tests/qwen4exp_build_test.py [tp_size]'
"""

from __future__ import annotations

import collections
import dataclasses
import os
import re
import shutil
import sys
import tempfile

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "fixtures", "qwen4exp")
sys.path.insert(0, HERE)

_failures = 0


def check(name: str, got, want) -> None:
    global _failures
    ok = got == want
    _failures += not ok
    print(f"  {'ok  ' if ok else 'FAIL'} {name:50s} got={got!r:<26} want={want!r}")


def main() -> int:
    global _failures
    tp_size = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    if tp_size != 1:
        # Not a limitation of the test: qwen4_exp itself is TP=1 in this bring-up, and refuses
        # tp>1 at construction. Section [9] below asserts that refusal.
        print(f"qwen4_exp is TP=1 only in this bring-up; refusing to run the build at tp={tp_size}")
        return 1

    from minisgl.distributed import set_tp_info

    set_tp_info(0, tp_size)

    import minisgl.layers.rotary as rotary_mod

    rotary_mod.set_rope_device(torch.device("cpu"))

    from qwen4exp_remap_test import load_ckpt_names, native_key_plan

    from minisgl.models import create_model
    from minisgl.models.config import ModelConfig
    from minisgl.models.weight import qwen4_exp_nvfp4_modules
    from minisgl.quant.config import QuantConfig
    from minisgl.utils import cached_load_hf_config

    tmp = tempfile.mkdtemp(prefix="qwen4exp-build-")
    shutil.copyfile(os.path.join(FIXTURES, "config.json"), os.path.join(tmp, "config.json"))
    mc = ModelConfig.from_hf(cached_load_hf_config(tmp), spec_algorithm="none")

    names = load_ckpt_names()
    nvfp4_modules = qwen4_exp_nvfp4_modules(names)

    # The modelopt arm has LANDED (plan T1.1): `QuantConfig.from_hf` rewrites the modelopt header
    # into the compressed-tensors shape, so `mc.quant` is a real NVFP4 config straight from
    # config.json. The explicitly-constructed `nvfp4` below stays as the ORACLE the build is
    # compared against, so a regression in from_hf (a wrong group size, a lost `ignore` glob) shows
    # up here structurally instead of as fluent garbage at serve time.
    check("QuantConfig.from_hf parses modelopt", mc.quant is not None, True)
    check("nothing recorded as unparsed", mc.unparsed_quant_method, None)
    if mc.quant is not None:
        check("method", mc.quant.method, "compressed-tensors")
        check("ct_format", mc.quant.ct_format, "nvfp4-pack-quantized")
        check("group_size (16, not MXFP4's 32)", mc.quant.group_size, 16)
        check("is_nvfp4", mc.quant.is_nvfp4, True)
        # The `ignore` GLOBS are the whole reason the bf16 half of this model stays bf16. Fed to a
        # substring matcher (the pre-T1.1 behaviour) every one of them misses, and the attention,
        # GDN, hyper-connections, shared expert and PLE all build quantized against unpacked tensors.
        for mod, want_q in (
            ("model.layers.0.mlp.experts.0.gate_proj", True),
            ("model.layers.0.self_attn.q_proj", False),
            ("model.layers.0.linear_attn.in_proj_qkv", False),
            ("model.layers.0.mlp.shared_expert.gate_proj", False),
            ("model.layers.0.attn_hyper_connection.input_mix_weight_down", False),
            ("model.layers.1.ple.key_proj", False),
            ("model.layers.0.mlp.gate", False),
        ):
            check(f"quantized? {mod.split('.', 2)[-1]:44s}", mc.quant.is_module_quantized(mod), want_q)
    nvfp4 = QuantConfig(
        method="compressed-tensors",
        bits=4,
        group_size=16,
        sym=True,
        weight_type="float",
        ct_format="nvfp4-pack-quantized",
        ckpt_quantized=nvfp4_modules,
    )
    mc_q = dataclasses.replace(mc, quant=nvfp4)

    torch.set_default_dtype(torch.bfloat16)
    with torch.device("meta"):
        model = create_model(mc_q)
    sd = model.state_dict()

    print(f"\n[build] tp_size={tp_size}  parameters={len(sd)}")
    groups = collections.Counter(re.sub(r"\.\d+\.", ".{N}.", k) for k in sd)
    for group, count in sorted(groups.items()):
        print(f"  x{count:<4} {group}")

    print("\n[1] key-set parity: meta build vs the 296,475-tensor checkpoint index")
    native, _skips, _stats = native_key_plan(names)
    missing = sorted(native - set(sd))          # loader would yield these; model has nowhere to put
    extra = sorted(set(sd) - native)            # model declares these; loader never yields them
    check("loader keys the model cannot accept", missing[:8], [])
    check("model keys the loader never yields", extra[:8], [])
    check("exact set equality", set(sd) == native, True)
    check("parameter count", len(sd), 1140)

    print("\n[2] structure: what must NOT exist")
    check("no final model.norm", [k for k in sd if re.fullmatch(r"model\.norm\..*", k)], [])
    check("no input_layernorm", [k for k in sd if "input_layernorm" in k][:3], [])
    check("no post_attention_layernorm", [k for k in sd if "post_attention" in k][:3], [])
    check("no mtp.* head", [k for k in sd if k.startswith("mtp")][:3], [])
    check("no visual tower", [k for k in sd if "visual" in k][:3], [])
    check("no n-gram table shards", [k for k in sd if "ngram_embedding" in k][:3], [])
    check("lm_head untied (own tensor)", "lm_head.weight" in sd, True)

    print("\n[3] structure: what must exist")
    check("hyper_connection_mixer tensors", sum("hyper_connection_mixer" in k for k in sd), 3)
    check("per-layer HC blocks", sum(k.endswith("_hyper_connection.hc_norm.weight") for k in sd), 96)
    check("block_inject only on layer HCs", sum("block_inject_weight" in k for k in sd), 96)
    check("GDN layers", sum(k.endswith(".linear_attn.conv1d_weight") for k in sd), 36)
    check("full-attn layers", sum(k.endswith(".self_attn.q_norm.weight") for k in sd), 12)
    check("QSA indexers", sum(k.endswith(".indexer.index_qk_proj.weight") for k in sd), 12)
    check("MoE gates", sum(k.endswith(".mlp.gate.weight") for k in sd), 48)
    check("shared experts", sum(k.endswith(".mlp.shared_expert_gate.weight") for k in sd), 48)
    ple = sorted(k for k in sd if ".ple." in k)
    check("PLE tensors", len(ple), 7)
    check("PLE is on decoder index 1 ONLY", sorted({k.split(".")[2] for k in ple}), ["1"])

    print("\n[4] shapes of the hyper-connection stream (the 4x-wide residual)")
    wide = mc.hc_hidden_size
    check("wide stream", wide, 10240)
    check(
        "hc_norm",
        tuple(sd["model.layers.0.attn_hyper_connection.hc_norm.weight"].shape),
        (wide,),
    )
    check(
        "input_mix_weight_down",
        tuple(sd["model.layers.0.attn_hyper_connection.input_mix_weight_down.weight"].shape),
        (320, wide),
    )
    check(
        "input_mix_weight_up",
        tuple(sd["model.layers.0.attn_hyper_connection.input_mix_weight_up.weight"].shape),
        (wide, 320),
    )
    check(
        "block_inject_weight",
        tuple(sd["model.layers.0.attn_hyper_connection.block_inject_weight.weight"].shape),
        (4, wide),
    )
    check("ple.conv1d_weight", tuple(sd["model.layers.1.ple.conv1d_weight"].shape), (wide, 1, 4))
    check("ple.key_proj", tuple(sd["model.layers.1.ple.key_proj.weight"].shape), (wide, 2560))
    check("ple.value_proj", tuple(sd["model.layers.1.ple.value_proj.weight"].shape), (2560, 2560))
    check(
        "ple layer_multipliers dtype",
        sd["model.layers.1.ple.ple_embedding.layer_multipliers"].dtype,
        torch.int64,
    )
    check(
        "indexer.index_qk_proj",
        tuple(sd["model.layers.3.self_attn.indexer.index_qk_proj.weight"].shape),
        ((4 + 1) * 128, 2560),
    )

    print("\n[5] the wide-residual plumbing COMPUTES (it used to refuse — tranche T0.3)")
    layer0 = model.model.layers.op_list[0]
    layer1 = model.model.layers.op_list[1]
    layer3 = model.model.layers.op_list[3]
    x_wide = torch.zeros(2, wide, device="meta")
    x = torch.zeros(2, mc.hidden_size, device="meta")

    def refuses(label, exc, fn, *a):
        """`fn(*a)` must raise `exc`. The message's first line is printed, because the whole value
        of a refusal is that it NAMES what is missing at the point of use."""
        global _failures
        try:
            fn(*a)
        except exc as e:
            print(f"  ok   {label:50s} -> {str(e).splitlines()[0][:78]}")
            return
        _failures += 1
        names = getattr(exc, "__name__", None) or "/".join(t.__name__ for t in exc)
        print(f"  FAIL {label:50s} -> did NOT raise {names}")

    # Shape-only on meta; the NUMERICS are pinned bit-for-bit against the sglang reference source in
    # tests/qwen4exp_hc_parity_test.py (fp32 exact, bf16 within 2 ULP on real layer-10 tensors).
    mixed, (res_x, res_n) = layer0.attn_hyper_connection.mix(x_wide)
    check("mix: wide -> hidden", tuple(mixed.shape), (2, mc.hidden_size))
    check("mix returns the UNNORMED stream", tuple(res_x.shape), (2, wide))
    check("mix returns the NORMED stream", tuple(res_n.shape), (2, wide))
    combined = layer0.attn_hyper_connection.combine(mixed, (res_x, res_n))
    check("combine: hidden -> wide", tuple(combined.shape), (2, wide))
    folded, _ = model.model.hyper_connection_mixer.mix(x_wide)
    check("mixer.mix folds for lm_head", tuple(folded.shape), (2, mc.hidden_size))

    print("\n[5b] what is still NOT implemented refuses, at the point of use")
    refuses("QSAIndexer.forward", NotImplementedError, layer3.self_attn.indexer.forward, x)
    # The budget is the ONLY thing separating "dense is bit-equivalent to the sparse selection" from
    # "we are serving a different model". At or below it the check must pass silently; one token past
    # it, it must raise — clamping or truncating instead would be undetectable from the output.
    layer3.self_attn.indexer.assert_dense_is_exact(mc.indexer_budget)
    print(f"  ok   {'dense is exact at seq_len == budget':50s} -> {mc.indexer_budget} passes silently")
    refuses(
        "one token past the indexer budget",
        NotImplementedError,
        layer3.self_attn.indexer.assert_dense_is_exact,
        mc.indexer_budget + 1,
    )
    refuses(
        "aux-hidden capture (no drafter can read a 4x stream)",
        NotImplementedError,
        model.set_capture_layers,
        [10],
    )
    # `hyper_connection_mixer` ships no block_inject_weight, so combining through it is a structural
    # error (RuntimeError), not merely unimplemented.
    refuses(
        "mixer.combine is a structural error",
        RuntimeError,
        model.model.hyper_connection_mixer.combine,
        x,
        (x_wide, x_wide),
    )
    check("PLE conv state width (dilated by ngram_size)", layer1.ple.conv_state_len, (4 - 1) * mc.ngram_size)
    # PLE.forward needs a batch staged by PLERuntime on the host first. With none, it is a
    # STRUCTURAL error — never a silent skip, which would drop the n-gram features from every token
    # at a cost in quality and no error anywhere. AssertionError is the shape it takes with no global
    # context at all (this test builds no engine); RuntimeError is the shape inside a real forward
    # where the context exists but `ctx.ple` was never wired.
    refuses(
        "PLE.forward without a staged batch",
        (RuntimeError, AssertionError),
        layer1.ple.forward,
        x_wide,
    )

    print("\n[6] engine seams")
    check("iter_gdn_layers", len(model.iter_gdn_layers()), 36)
    check("ple_block() finds the single PLE block", model.ple_block() is layer1.ple, True)

    print("\n[7] the model also builds unquantized (the quant=None fallback)")
    # NOT `mc` any more: since T1.1 landed, `ModelConfig.from_hf` parses the modelopt header, so `mc`
    # already carries the NVFP4 config and building from it would just repeat section [1]. The
    # fallback worth pinning is what a checkpoint with NO parsed quant produces — a DIFFERENT key
    # set, which is why routing a quantized checkpoint into it is loud rather than silent.
    with torch.device("meta"):
        model_bf16 = create_model(dataclasses.replace(mc, quant=None))
    sd_bf16 = model_bf16.state_dict()
    # 96 fewer keys than the NVFP4 build: an unquantized expert stack is ONE tensor per GEMM
    # (gate_up_proj, down_proj) instead of weight_packed + weight_scale, over 48 layers.
    check("bf16 build parameter count", len(sd_bf16), 1140 - 48 * 2)
    check(
        "bf16 experts want a bare .weight the NVFP4 ckpt does not ship as bf16",
        "model.layers.0.mlp.experts.gate_up_proj" in sd_bf16,
        True,
    )

    print("\n[8] load_weight routes to the qwen4_exp loader, NOT the Qwen3.5 one")
    # qwen4_exp IS a GDN hybrid (36 of 48 layers), so without an explicit `is_qwen4_exp` branch
    # AHEAD of `is_gdn_hybrid`, `load_weight` falls into `_load_qwen3_5_weight` — whose remap passes
    # `hyper_connection_mixer.*` / `ple.*` / `indexer.*` through as "direct" and then dies in
    # load_state_dict with an unexpected-keys dump that names no cause. `tmp` holds only config.json,
    # so the right loader is identified by WHICH error it raises: the qwen4_exp one complains there
    # are no shards; the Qwen3.5 one would run and yield nothing.
    from minisgl.models import load_weight

    raised = None
    try:
        next(iter(load_weight(tmp, torch.device("cpu"), spec_algorithm="none")))
    except (FileNotFoundError, StopIteration) as e:
        raised = f"{type(e).__name__}: {e}"
    check("qwen4_exp loader ran (no shards under tmp)", "FileNotFoundError" in (raised or ""), True)
    if raised:
        print(f"       -> {raised.splitlines()[0][:100]}")

    print("\n[9] TP > 1 is refused at BUILD, not silently under-sharded")
    # Refused in two independent places on purpose: the model (so the failure is at construction,
    # naming the cause) and the loader (so nobody bypasses the model check by calling load_weight
    # directly). A missing shard rule falls through to "replicate", which loads all 84 GB on every
    # rank and then fails somewhere unrelated.
    #
    # `set_tp_info` is deliberately set-once per process (a second call raises), so the module global
    # is swapped directly here. Test-only, and restored in the `finally` — a real engine sets TP once
    # at boot and never changes it.
    import minisgl.distributed.info as _info

    _saved_tp = _info._TP_INFO
    _info._TP_INFO = _info.DistributedInfo(0, 2)
    try:
        refuses("create_model(tp=2)", NotImplementedError, create_model, mc_q)
        refuses(
            "load_weight(tp=2)",
            NotImplementedError,
            lambda: next(iter(load_weight(tmp, torch.device("cpu"), spec_algorithm="none"))),
        )
    finally:
        _info._TP_INFO = _saved_tp

    shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{'PASS' if not _failures else f'FAIL ({_failures} checks)'}")
    return 1 if _failures else 0


if __name__ == "__main__":
    sys.exit(main())
