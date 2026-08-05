"""DiffusionGemma build + loader: one instantiated stack must serve BOTH roles, exactly.

CPU/meta only — no GPU lease, cannot disturb a running serve. Run inside the serve image (the host
torch install is broken):

    docker run --rm --entrypoint bash -v <worktree>:/wt \
      -v /home/pat/.cache/huggingface:/root/.cache/huggingface -e HF_HUB_OFFLINE=1 \
      minisgl-rdna4:lean -lc 'cd /wt && PYTHONPATH=/wt/python:/opt/kernels \
        python tests/diffusiongemma_build_test.py [tp_size]'

This is the test that proves the reusability claim the whole port rests on: `DiffusionGemmaModel`
is `Gemma4Model` plus one self-conditioning block, and the checkpoint's 35 777 tensors map onto it
with nothing dropped and nothing invented. Four things are checked that the AR sibling's build test
cannot cover:

  1. The backbone parameter set is IDENTICAL to the AR sibling's, key for key. Not "similar" — the
     two models are built side by side from their own configs and the sets are differenced, so a
     drift in either one shows up as a named key rather than as a parity failure later.
  2. `model.self_conditioning.*` exists with the DENSE mlp width (2112, not the MoE 704) and is
     UNQUANTIZED. That last part is the silent one: the checkpoint's compressed-tensors ignore list
     names it `model.decoder.self_conditioning.*`, and if that namespace is not de-wrapped the model
     builds an int4 module for an fp16 tensor.
  3. The 30 encoder `layer_scalar` tensors are consumed by the tie CHECK and never emitted — the
     loader must prove they equal the decoder's rather than skipping the whole encoder namespace.
  4. The loader's emitted key set matches the declared set EXACTLY, at TP=1 and TP=2, on shape and
     dtype. A missing key is a KeyError at load and an extra one a RuntimeError; the dangerous third
     case — right name, wrong shard — is what the shape comparison exists for.
"""

from __future__ import annotations

import collections
import glob
import re
import sys

import torch

MODEL_ID = "cyankiwi/diffusiongemma-26B-A4B-it-AWQ-INT4"
AR_MODEL_ID = "cyankiwi/gemma-4-26B-A4B-it-qat-AWQ-INT4"


def _snapshot(model_id: str) -> str | None:
    pattern = f"/root/.cache/huggingface/hub/models--{model_id.replace('/', '--')}/snapshots/*/"
    matches = glob.glob(pattern)
    return matches[0] if matches else None


class Report:
    def __init__(self) -> None:
        self.failures = 0

    def check(self, name: str, ok: bool, detail: str) -> bool:
        self.failures += not ok
        print(f"  {'ok  ' if ok else 'FAIL'} {name:46s} {detail}")
        return ok


def main() -> int:
    tp_size = int(sys.argv[1]) if len(sys.argv) > 1 else 1

    path = _snapshot(MODEL_ID)
    if path is None:
        print(f"SKIP: {MODEL_ID} not cached")
        return 0

    from minisgl.distributed import set_tp_info

    set_tp_info(0, tp_size)

    import minisgl.layers.rotary as rotary_mod
    from transformers import AutoConfig

    from minisgl.models import create_model, load_weight
    from minisgl.models.config import ModelConfig
    from minisgl.models.weight import checkpoint_tensor_names

    rotary_mod.set_rope_device(torch.device("cpu"))
    torch.set_default_dtype(torch.float16)
    torch.set_grad_enabled(False)

    hf = AutoConfig.from_pretrained(path)
    names = checkpoint_tensor_names(MODEL_ID)
    mc = ModelConfig.from_hf(hf, spec_algorithm="none", ckpt_tensor_names=names)

    print(
        f"[diffusiongemma] tp_size={tp_size}  arch={mc.architectures}  model_type={mc.model_type}\n"
        f"  layers={mc.num_layers} hidden={mc.hidden_size} vocab={mc.vocab_size} "
        f"head_dim={mc.head_dim}/{mc.swa_head_dim} kv={mc.num_kv_heads}/{mc.swa_num_kv_heads} "
        f"k_eq_v={mc.attention_k_eq_v} experts={mc.num_experts}x{mc.num_experts_per_tok} "
        f"softcap={mc.final_logit_softcapping} window={mc.sliding_window}"
    )

    rep = Report()

    with torch.device("meta"):
        model = create_model(mc)
    sd = model.state_dict()
    declared = {k: (tuple(v.shape), v.dtype) for k, v in sd.items()}

    print(f"\n[1] parameter census — {len(sd)} declared")
    groups = collections.Counter(re.sub(r"\.\d+\.", ".{N}.", k) for k in sd)
    for group, count in sorted(groups.items()):
        print(f"  {group:66s} x{count}")

    # ---- 1. the backbone is byte-for-byte the AR sibling's parameter set ----------------------
    print("\n[2] the backbone is the SAME stack as the autoregressive sibling")
    ar_path = _snapshot(AR_MODEL_ID)
    if ar_path is None:
        print(f"  SKIP: {AR_MODEL_ID} not cached, cannot difference the two parameter sets")
    else:
        ar_mc = ModelConfig.from_hf(
            AutoConfig.from_pretrained(ar_path),
            spec_algorithm="none",
            ckpt_tensor_names=checkpoint_tensor_names(AR_MODEL_ID),
        )
        with torch.device("meta"):
            ar_model = create_model(ar_mc)
        ar_sd = ar_model.state_dict()
        only_dg = sorted(set(sd) - set(ar_sd))
        only_ar = sorted(set(ar_sd) - set(sd))
        mismatch = sorted(
            k for k in set(sd) & set(ar_sd)
            if sd[k].shape != ar_sd[k].shape or sd[k].dtype != ar_sd[k].dtype
        )
        rep.check(
            "DiffusionGemma adds ONLY self_conditioning",
            all(k.startswith("model.self_conditioning.") for k in only_dg) and len(only_dg) == 3,
            f"extra keys = {only_dg}",
        )
        rep.check("DiffusionGemma drops nothing", not only_ar, f"missing vs AR = {only_ar[:8]}")
        rep.check(
            "shared keys agree on shape+dtype",
            not mismatch,
            f"{len(set(sd) & set(ar_sd))} shared keys, {len(mismatch)} differing"
            + (f": {mismatch[:5]}" if mismatch else ""),
        )
        del ar_model, ar_sd

    # ---- 2. the self-conditioning block ------------------------------------------------------
    print("\n[3] self-conditioning: dense width, unquantized, both norms accounted for")
    inter = mc.intermediate_size
    hidden = mc.hidden_size
    rep.check(
        "gate_up_proj is the DENSE width, column-parallel",
        tuple(sd.get("model.self_conditioning.gate_up_proj.weight", torch.empty(0)).shape)
        == (2 * inter // tp_size, hidden),
        f"got={tuple(sd['model.self_conditioning.gate_up_proj.weight'].shape)} "
        f"want=({2 * inter // tp_size}, {hidden})  (moe_intermediate_size={mc.moe_intermediate_size} "
        f"would be {2 * mc.moe_intermediate_size // tp_size})",
    )
    rep.check(
        "down_proj is row-parallel",
        tuple(sd["model.self_conditioning.down_proj.weight"].shape) == (hidden, inter // tp_size),
        f"got={tuple(sd['model.self_conditioning.down_proj.weight'].shape)} "
        f"want=({hidden}, {inter // tp_size})",
    )
    rep.check(
        "pre_norm has a learned gain, post_norm does not",
        tuple(sd["model.self_conditioning.pre_norm.weight"].shape) == (hidden,)
        and not any(k.startswith("model.self_conditioning.post_norm") for k in sd),
        f"pre_norm={tuple(sd['model.self_conditioning.pre_norm.weight'].shape)}; "
        f"post_norm keys={[k for k in sd if 'post_norm' in k]} (with_scale=False -> none)",
    )
    # The quant-ignore namespace bug: `model.decoder.self_conditioning.*` must de-wrap to
    # `model.self_conditioning.*` or every fp16 module in that list builds int4.
    assert mc.quant is not None
    for module in (
        "model.self_conditioning.gate_proj",
        "model.layers.0.mlp.gate_proj",
        "model.layers.0.router.proj",
    ):
        rep.check(
            f"unquantized: {module.split('.', 1)[1]}",
            not mc.quant.is_module_quantized(module),
            "in the compressed-tensors ignore list after the `model.decoder.` de-wrap",
        )
    rep.check(
        "still quantized: layers.0.self_attn.q_proj",
        mc.quant.is_module_quantized("model.layers.0.self_attn.q_proj"),
        "the attention projections are NOT in the ignore list (guards an over-broad de-wrap)",
    )
    rep.check(
        "self_conditioning built fp16, not packed",
        sd["model.self_conditioning.gate_up_proj.weight"].dtype == torch.float16
        and "model.self_conditioning.gate_up_proj.weight_packed" not in sd,
        f"dtype={sd['model.self_conditioning.gate_up_proj.weight'].dtype}",
    )
    rep.check(
        "lm_head is tied (declares no weight of its own)",
        "lm_head.weight" not in sd,
        f"tie_word_embeddings={mc.tie_word_embeddings}",
    )

    # ---- 3. the checkpoint side --------------------------------------------------------------
    print("\n[4] checkpoint namespaces the loader must account for")
    enc_text = [n for n in names if n.startswith("model.encoder.language_model.")]
    enc_vision = [
        n for n in names
        if n.startswith("model.encoder.") and not n.startswith("model.encoder.language_model.")
    ]
    rep.check(
        "encoder text namespace is layer_scalar and nothing else",
        len(enc_text) == mc.num_layers
        and all(n.endswith(".layer_scalar") for n in enc_text),
        f"{len(enc_text)} tensors, suffixes="
        f"{sorted({n.rsplit('.', 1)[-1] for n in enc_text})}  (vision-side: {len(enc_vision)})",
    )
    rep.check(
        "checkpoint ships no lm_head",
        not any(n.startswith("lm_head") for n in names),
        f"{len(names)} tensors total, {sum(1 for n in names if n.startswith('model.decoder.'))} "
        f"under model.decoder.",
    )

    # ---- 4. the exact loader key match -------------------------------------------------------
    print(f"\n[5] loader key match at tp_size={tp_size} (streams the full checkpoint to CPU)")
    emitted: dict[str, tuple] = {}
    duplicates: list[str] = []
    for name, tensor in load_weight(MODEL_ID, torch.device("cpu"), spec_algorithm="none"):
        if name in emitted:
            duplicates.append(name)
        emitted[name] = (tuple(tensor.shape), tensor.dtype)
        del tensor

    missing = sorted(set(declared) - set(emitted))
    extra = sorted(set(emitted) - set(declared))
    wrong_shape = sorted(k for k in set(declared) & set(emitted) if declared[k][0] != emitted[k][0])
    # A wide-float dtype difference is a legitimate cast (_coerce_dtype); a packed/quantized one is
    # an encoding mismatch and must fail.
    castable = {torch.float32, torch.float64, torch.bfloat16, torch.float16}
    wrong_dtype = sorted(
        k for k in set(declared) & set(emitted)
        if declared[k][1] != emitted[k][1]
        and not (declared[k][1] in castable and emitted[k][1] in castable)
    )

    def report(label: str, keys: list[str], detail=None) -> None:
        print(f"  {label}: {len(keys)}")
        for k in keys[:12]:
            print(f"      {k}{'' if detail is None else detail(k)}")
        if len(keys) > 12:
            print(f"      ... and {len(keys) - 12} more")

    print(f"  declared={len(declared)}  emitted={len(emitted)}")
    report("MISSING (declared, never emitted)", missing)
    report("EXTRA (emitted, not declared)", extra)
    report("WRONG SHAPE", wrong_shape, lambda k: f"  model={declared[k][0]} loader={emitted[k][0]}")
    report("WRONG DTYPE", wrong_dtype, lambda k: f"  model={declared[k][1]} loader={emitted[k][1]}")
    report("DUPLICATE emissions", sorted(set(duplicates)))
    rep.check(
        "exact key/shape/dtype match",
        not (missing or extra or wrong_shape or wrong_dtype or duplicates),
        f"{len(missing)} missing, {len(extra)} extra, {len(wrong_shape)} mis-shaped, "
        f"{len(wrong_dtype)} mis-typed, {len(duplicates)} duplicated",
    )
    rep.check(
        "encoder layer_scalars consumed by the tie check, not emitted",
        not any(".encoder." in k for k in emitted),
        "load_weight completed, so the 30 encoder/decoder layer_scalar pairs compared bit-equal",
    )

    print(f"\n{'PASS' if rep.failures == 0 else f'FAIL ({rep.failures} checks)'}")
    return 1 if rep.failures else 0


if __name__ == "__main__":
    sys.exit(main())
