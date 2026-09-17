"""qwen4_exp weight loader, run on the REAL `RadixArk/Qwen3.8-Flash-Next-NVFP4` bytes.

`qwen4exp_remap_test.py` proves the NAME pipeline over the checkpoint index; this proves the
LOADER — the thing that actually opens the shards, folds the modelopt two-level NVFP4 scale, concats
the GDN in_proj pairs, merges gate/up and stacks 512 experts — by comparing every tensor it emits
against what the model built on `meta` declares, on NAME, SHAPE and DTYPE.

Why all three: `BaseOP.load_state_dict` pops each declared key and raises on leftovers, so a MISSING
key is a KeyError and an EXTRA key is a RuntimeError — both loud. The case that is NOT loud, and the
reason this test exists, is a key with the right name and the wrong shape or dtype: a folded NVFP4
scale that came out (N, K//32) because the group size defaulted to MXFP4's, an expert stack built in
the wrong expert order, a gate/up merge concatenated on the packing axis. All of those load cleanly
and produce fluent garbage.

SUBSET BY DEFAULT. The full body is 84 GB across 206 shards; the default run takes the shards for
decoder layers 0 (GDN), 1 (GDN + the single PLE block) and 3 (full attention + QSA indexer) plus
every `model-bf16-*` shard present, symlinks them into a temp dir with the real config.json and
index.json, and loads that. Every emitted key must be a declared key with the same shape — a subset
of the parameters, loaded from real bytes, is a measurement; a projection is not. `--full` streams
all 206 shards (bounded CPU memory: one expert stack at a time, ~4 GB peak) and additionally asserts
that the emitted set EQUALS the declared set.

CPU only. No GPU — the loader is handed `torch.device("cpu")`, so this cannot disturb a
serve. Run in the serve image:

    docker run --rm -v <worktree>:/engine -v <ckpt>:/model:ro \
      --entrypoint bash minisgl-rdna4:m1b-20260903 -lc \
      'PYTHONPATH=/engine/python:/opt/kernels python /engine/tests/qwen4exp_loader_test.py /model'
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys
import tempfile

import torch

DEFAULT_MODEL = "/model"
#: Decoder layers whose expert shards are pulled into the subset. 0 = plain GDN layer, 1 = the ONE
#: layer carrying the PLE block, 3 = the first full-attention layer (self_attn + QSA indexer). Every
#: structural family in the checkpoint is represented by that trio.
SUBSET_LAYERS = (0, 1, 3)

_failures = 0


def check(name: str, got, want) -> None:
    global _failures
    ok = got == want
    _failures += not ok
    print(f"  {'ok  ' if ok else 'FAIL'} {name:56s} got={got!r:<20} want={want!r}")


def check_true(name: str, cond, detail: str = "") -> None:
    """For a check whose evidence is a MEASUREMENT rather than an equality — the detail string
    carries the numbers, so a pass is auditable and not just a green tick."""
    global _failures
    _failures += not cond
    print(f"  {'ok  ' if cond else 'FAIL'} {name:56s} {detail}")


def _subset_dir(model_path: str, tmp: str) -> str:
    """Symlink a per-layer subset of the shards (plus all bf16 shards, config and index) into `tmp`.

    Symlinks, not copies: the point is to exercise the real loader over real bytes without a 84 GB
    read or a 84 GB duplicate.
    """
    wanted = {f"layer-{lid:05d}-experts-" for lid in SUBSET_LAYERS}
    n = 0
    for src in glob.glob(f"{model_path}/*"):
        base = os.path.basename(src)
        if base.endswith(".safetensors") and not (
            base.startswith("model-bf16-") or any(base.startswith(w) for w in wanted)
        ):
            continue
        if "model-plefp8-" in base:
            continue  # the n-gram table is not a parameter; the loader skips it by name anyway
        os.symlink(src, os.path.join(tmp, base))
        n += base.endswith(".safetensors")
    print(f"[subset] {n} shards symlinked into {tmp} (layers {list(SUBSET_LAYERS)} + all bf16)")
    return tmp


def main() -> int:
    model_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    full = "--full" in sys.argv[1:]
    if not os.path.isfile(os.path.join(model_path, "config.json")):
        print(f"SKIP: no config.json under {model_path}")
        return 0

    from minisgl.distributed import set_tp_info

    set_tp_info(0, 1)

    import minisgl.layers.rotary as rotary_mod

    from minisgl.models import create_model, load_weight
    from minisgl.models.config import ModelConfig
    from minisgl.models.weight import checkpoint_tensor_names
    from minisgl.utils import cached_load_hf_config

    # `cached_load_hf_config`, NOT `AutoConfig.from_pretrained`: transformers in this image has no
    # `qwen4_exp` entry and AutoConfig hard-raises on an unknown model_type. minisgl's loader falls
    # back to the generic PretrainedConfig and restores `model_type` by hand — which is the ONLY
    # reason `is_qwen4_exp` works at all, and is what the engine itself calls. Testing through
    # AutoConfig would be testing a path the serve never takes.
    mc_cfg = cached_load_hf_config(model_path)

    # The rotary tables are built eagerly at model construction; on a meta build there is no device,
    # so point them at the CPU. Same thing gemma4_loader_test does, same reason.
    rotary_mod.set_rope_device(torch.device("cpu"))

    torch.set_default_dtype(torch.bfloat16)
    mc = ModelConfig.from_hf(
        mc_cfg,
        spec_algorithm="none",
        ckpt_tensor_names=checkpoint_tensor_names(model_path),
    )
    print(f"[config] {mc.model_type} layers={mc.num_layers} experts={mc.num_experts} "
          f"hc={mc.hc_count}x{mc.hidden_size} quant={mc.quant and mc.quant.method}")
    with torch.device("meta"):
        model = create_model(mc)
    declared = {k: (tuple(v.shape), v.dtype) for k, v in model.state_dict().items()}
    print(f"[model] declares {len(declared)} parameters")

    tmp_ctx = None
    if full:
        load_from = model_path
    else:
        tmp_ctx = tempfile.TemporaryDirectory(prefix="q4e-subset-")
        load_from = _subset_dir(model_path, tmp_ctx.name)

    emitted: "dict[str, tuple]" = {}
    duplicates: "list[str]" = []
    n_bytes = 0
    # Two tensors are kept by VALUE, for section [3b]. Everything else is reduced to (shape, dtype)
    # immediately, because holding 84 GB is the thing this test is careful not to do.
    _VALUE_KEYS = (
        "model.layers.0.mlp.experts.gate_up_proj.weight_scale",
        "model.layers.0.mlp.experts.gate_up_proj.weight_packed",
        # The NVFP4 global is the SECOND half of a two-level scale, so [3b] cannot check magnitude
        # without it — dequantizing from the block scale alone is exactly the ~4.8e3x error the
        # `w_zeros`-slot plumbing exists to prevent, and would silently pass a looser check.
        "model.layers.0.mlp.experts.gate_up_proj.weight_global",
        "model.layers.0.mlp.shared_expert.gate_up_proj.weight",
        "model.layers.0.linear_attn.A_log",
        "model.layers.0.linear_attn.dt_bias",
    )
    _STACKED_VALUE_KEYS = ("weight_scale", "weight_packed", "weight_global")
    kept: "dict[str, torch.Tensor]" = {}
    for name, tensor in load_weight(load_from, torch.device("cpu"), spec_algorithm="none"):
        if name in emitted:
            duplicates.append(name)
        emitted[name] = (tuple(tensor.shape), tensor.dtype)
        n_bytes += tensor.numel() * tensor.element_size()
        if name in _VALUE_KEYS:
            # Expert 0 only for the stacked tensors — one 640x2560 matrix, not 512 of them.
            kept[name] = tensor[0].clone() if name.endswith(_STACKED_VALUE_KEYS) \
                else tensor.clone()
        del tensor
    print(f"[loader] emitted {len(emitted)} parameters, {n_bytes / 2**30:.2f} GiB")

    extra = sorted(set(emitted) - set(declared))
    common = set(emitted) & set(declared)
    wrong_shape = sorted(k for k in common if declared[k][0] != emitted[k][0])
    # dtype: the engine legitimately casts between wide float types after the loader (`_coerce_dtype`
    # — a checkpoint may ship fp32 where a layer declares bf16). A PACKED/quantized dtype mismatch is
    # never legitimate: the dtype IS the encoding contract (uint8 E2M1 nibbles vs a bf16 matrix).
    castable = {torch.float32, torch.float64, torch.bfloat16, torch.float16}
    wrong_dtype = sorted(
        k
        for k in common
        if declared[k][1] != emitted[k][1]
        and not (declared[k][1] in castable and emitted[k][1] in castable)
    )

    print("\n[1] every emitted tensor is a parameter the model declared, with the right shape")
    check("duplicate emissions", duplicates[:3], [])
    check("emitted-but-not-declared", extra[:3], [])
    check("shape mismatches", wrong_shape[:3], [])
    check("dtype mismatches (packed dtypes)", wrong_dtype[:3], [])

    print("\n[2] the subset covers every structural family (real bytes, not the index)")
    fams = {
        "hyper-connection (wide 4x2560)": "model.layers.0.attn_hyper_connection.hc_norm.weight",
        "final mixer (replaces model.norm)": "model.hyper_connection_mixer.input_mix_weight_up.weight",
        "GDN fused in_proj_qkvz": "model.layers.0.linear_attn.in_proj_qkvz.weight",
        "GDN fused in_proj_ba": "model.layers.0.linear_attn.in_proj_ba.weight",
        "GDN flat conv1d": "model.layers.0.linear_attn.conv1d_weight",
        "PLE conv1d (layer 1 only)": "model.layers.1.ple.conv1d_weight",
        "PLE key_proj": "model.layers.1.ple.key_proj.weight",
        "full-attn gated q_proj": "model.layers.3.self_attn.q_proj.weight",
        "QSA indexer qk_proj": "model.layers.3.self_attn.indexer.index_qk_proj.weight",
        "NVFP4 expert stack (packed)": "model.layers.0.mlp.experts.gate_up_proj.weight_packed",
        "NVFP4 expert stack (folded scale)": "model.layers.0.mlp.experts.gate_up_proj.weight_scale",
        "bf16 shared expert (merged)": "model.layers.0.mlp.shared_expert.gate_up_proj.weight",
    }
    for label, key in fams.items():
        check(label, emitted.get(key, "MISSING"), declared.get(key, "NOT-DECLARED"))

    print("\n[3] the modelopt NVFP4 scale reached the repo-native TWO-LEVEL form")
    wp = emitted.get("model.layers.0.mlp.experts.gate_up_proj.weight_packed")
    ws = emitted.get("model.layers.0.mlp.experts.gate_up_proj.weight_scale")
    wg = emitted.get("model.layers.0.mlp.experts.gate_up_proj.weight_global")
    if wp and ws:
        e, n, k_half = wp[0]
        check("packed dtype is uint8 E2M1", str(wp[1]), "torch.uint8")
        check("expert count on dim 0", e, mc.num_experts)
        check("merged gate_up rows = 2 * moe_inter", n, 2 * mc.moe_intermediate_size)
        check("packed cols = hidden/2 (2 nibbles/byte)", k_half, mc.hidden_size // 2)
        check("scale shape (E, N, K/16) — group 16, NOT 32", ws[0], (e, n, mc.hidden_size // 16))
        # The block scale is now the checkpoint's OWN e4m3 tensor, byte-verbatim: no fold, no
        # rounding, nothing to get wrong except the direction of the global, which is checked by
        # MAGNITUDE in [3b]. The shape being K/16 rather than K/32 is what pins the NVFP4 (not
        # MXFP4) interpretation. The dtype being float8_e4m3fn rather than fp16 is what pins that
        # the LOSSY fold (measured 4.37e-04 max rel-err) is no longer on this path.
        check("block-scale dtype is e4m3 (NOT the lossy fp16 fold)", str(ws[1]),
              "torch.float8_e4m3fn")
        check(
            "no raw per-tensor globals survive",
            [k for k in emitted if k.endswith(".weight_global_scale")][:2],
            [],
        )
    else:
        check("expert stack present", False, True)

    print("\n[3a] the per-output-channel global SURVIVED the gate|up merge and the expert stack")
    # This is the shape claim the whole design rests on. The global is emitted at the LEAF as an (N,)
    # vector per projection; `emit` concatenates gate|up on dim 0 and `_ExpertStacker` stacks over E.
    # If the vector shape were wrong — a scalar, or an (N,1) — the merge would produce something that
    # still loads (E, something) and the kernel would stride it wrong. So the assertion is that the
    # merged, stacked global is EXACTLY (E, 2*moe_inter): one f32 per expert per OUTPUT CHANNEL of
    # the merged matrix, which is the `w_zeros`-slot layout `E4m3GroupScaleGlobal::wz_base` indexes.
    if wg:
        check("global shape (E, 2*moe_inter) after merge+stack", wg[0],
              (mc.num_experts, 2 * mc.moe_intermediate_size))
        check("global dtype is f32", str(wg[1]), "torch.float32")
    else:
        check("gate_up global emitted", False, True)
    wg2 = emitted.get("model.layers.0.mlp.experts.down_proj.weight_global")
    if wg2:
        # down_proj is NOT merged, so its global is the plain (E, hidden) vector — a different N
        # from gate_up's, which is exactly why the contract is a VECTOR and not one number.
        check("down_proj global shape (E, hidden)", wg2[0], (mc.num_experts, mc.hidden_size))
    else:
        check("down_proj global emitted", False, True)

    print("\n[3b] the two-level NVFP4 scale has the right MAGNITUDE, not just the right shape")
    # Section [3] checks shape and dtype. Both were already correct on the day this loader produced
    # inf: modelopt's `weight_scale_2` is the RECIPROCAL of compressed-tensors' `weight_global_scale`
    # (2.078e-4 vs ~4812 here), the code divided where it should have multiplied, and the result
    # overflowed to inf -> NaN logits from the first MoE block. Nothing structural was wrong.
    #
    # The anchor is the SAME LAYER's shared expert, which this checkpoint ships UNQUANTIZED in bf16.
    # It is the same kind of matrix, trained in the same model, so its coefficient magnitude is the
    # measurement that settles the direction — no formula is taken on faith.
    ws_v = kept.get("model.layers.0.mlp.experts.gate_up_proj.weight_scale")
    wp_v = kept.get("model.layers.0.mlp.experts.gate_up_proj.weight_packed")
    wg_v = kept.get("model.layers.0.mlp.experts.gate_up_proj.weight_global")
    anchor_v = kept.get("model.layers.0.mlp.shared_expert.gate_up_proj.weight")
    if ws_v is not None and wp_v is not None and wg_v is not None and anchor_v is not None:
        from minisgl.quant.mxfp4 import FP4_E2M1_LUT
        from minisgl.quant.nvfp4 import NVFP4_GROUP_SIZE, unpack_e2m1_nibbles

        # `isfinite` has no float8 kernel, so finiteness is asserted on the f32 the dequant uses —
        # which is the value that actually reaches the kernel anyway.
        ws_f32 = ws_v.to(torch.float32)
        check("block scale is finite", bool(ws_f32.isfinite().all()), True)
        check("global is finite and strictly positive",
              bool(wg_v.isfinite().all()) and bool((wg_v > 0).all()), True)
        lut = torch.tensor(FP4_E2M1_LUT, dtype=torch.float32)
        codes = unpack_e2m1_nibbles(wp_v).to(torch.int64)
        # Kernel order: group fold first, per-output-channel epilogue second.
        w = lut[codes] * ws_f32.repeat_interleave(NVFP4_GROUP_SIZE, dim=-1)
        w = w * wg_v.to(torch.float32).unsqueeze(-1)
        got_mean = float(w.abs().mean())
        want_mean = float(anchor_v.to(torch.float32).abs().mean())
        ratio = got_mean / want_mean if want_mean else float("inf")
        # An order of magnitude is a deliberately loose band: expert and shared-expert magnitudes are
        # not required to match, only to be the same KIND of number. The failure this guards against
        # is off by 10^7, so a factor-of-10 window separates it from every legitimate difference.
        check_true(
            "dequantized expert |w| matches the bf16 shared expert to within 10x",
            0.1 < ratio < 10.0,
            f"expert |w|mean={got_mean:.6f}  bf16 anchor |w|mean={want_mean:.6f}  ratio={ratio:.3f}",
        )
        check_true(
            "dequantized expert |w| is bounded (the global direction is the MULTIPLIER)",
            bool(w.isfinite().all()) and float(w.abs().max()) < 10.0,
            f"|w|max={float(w.abs().max()):.6f}",
        )
        # THE ACCURACY CLAIM, MEASURED ON THESE BYTES rather than quoted. The fp16 fold this
        # replaced is recomputed here from the SAME two levels and compared against the same fp64
        # golden, so the two arms differ only in where the rounding happens.
        gold = (
            lut[codes].to(torch.float64)
            * ws_f32.to(torch.float64).repeat_interleave(NVFP4_GROUP_SIZE, dim=-1)
            * wg_v.to(torch.float64).unsqueeze(-1)
        )
        folded = (ws_f32 * wg_v.to(torch.float32).unsqueeze(-1)).to(torch.float16)
        w_fold = lut[codes] * folded.to(torch.float32).repeat_interleave(NVFP4_GROUP_SIZE, dim=-1)
        nz = gold != 0
        rel_split = ((w.to(torch.float64) - gold).abs() / gold.abs().clamp_min(1e-30))[nz]
        rel_fold = ((w_fold.to(torch.float64) - gold).abs() / gold.abs().clamp_min(1e-30))[nz]
        split_max, fold_max = float(rel_split.max()), float(rel_fold.max())
        check_true(
            "the fp16 fold this replaced IS lossy (the docstring used to claim it was exact)",
            fold_max > 1e-4,
            f"fp16-fold rel-err max={fold_max:.3e} mean={float(rel_fold.mean()):.3e}",
        )
        check_true(
            "the two-level split is exact to f32 round-off, and beats the fold by >1000x",
            split_max < 1e-6 and split_max * 1000.0 < fold_max,
            f"split rel-err max={split_max:.3e}  vs  fold max={fold_max:.3e}  "
            f"({(fold_max / split_max) if split_max else float('inf'):.0f}x)",
        )
    else:
        check("layer-0 expert stack + bf16 shared-expert anchor present", False, True)

    print("\n[3c] the GDN gating params reach the kernels as fp32")
    # `cast_checkpoint_tensor` (models/weight.py) is the second half of loading and lives OUTSIDE
    # `load_weight`, so the raw loader legitimately yields these as bf16 — this checkpoint stores
    # both that way. What is asserted is that the shared cast, the one the Engine applies, upcasts
    # them: `GDNLinearAttn` loads its module with assign=True, so a bf16 A_log survives into the
    # parameter and `gdn_prefill_wmma` raises "expected scalar type Float but found BFloat16".
    from minisgl.models import cast_checkpoint_tensor

    for key in ("model.layers.0.linear_attn.A_log", "model.layers.0.linear_attn.dt_bias"):
        raw = kept.get(key)
        if raw is None:
            check(f"{key.rsplit('.', 1)[-1]} emitted", False, True)
            continue
        check(
            f"{key.rsplit('.', 1)[-1]}: cast_checkpoint_tensor -> fp32",
            cast_checkpoint_tensor(key, raw, torch.bfloat16).dtype,
            torch.float32,
        )

    print("\n[4] the two structural absences the checkpoint enforces")
    check("no model.norm.weight declared", "model.norm.weight" in declared, False)
    check(
        "no per-layer input_layernorm declared",
        any(".input_layernorm." in k for k in declared),
        False,
    )
    check(
        "no per-layer post_attention_layernorm",
        any(".post_attention_layernorm." in k for k in declared),
        False,
    )

    if full:
        print("\n[5] --full: the emitted set EQUALS the declared set")
        missing = sorted(set(declared) - set(emitted))
        check("declared-but-not-emitted", missing[:5], [])
        check("parameter count", len(emitted), len(declared))
    else:
        # A subset run cannot prove coverage, but it CAN prove that the layers it did load are
        # complete — i.e. nothing in a loaded layer was silently dropped.
        print("\n[5] subset: the loaded layers are COMPLETE (nothing silently dropped)")
        pat = re.compile(r"^model\.layers\.(\d+)\.")
        for lid in SUBSET_LAYERS:
            want = {
                k
                for k in declared
                if (m := pat.match(k)) and int(m.group(1)) == lid
            }
            got = want & set(emitted)
            check(f"layer {lid}: declared keys emitted", len(got), len(want))
        if missing_from_subset := sorted(
            k for k in set(declared) - set(emitted)
            if (m := pat.match(k)) and int(m.group(1)) in SUBSET_LAYERS
        ):
            print(f"       missing: {missing_from_subset[:5]}")

    if tmp_ctx is not None:
        tmp_ctx.cleanup()
    print("\n" + ("PASS" if not _failures else f"FAIL ({_failures} checks)"))
    return 1 if _failures else 0


if __name__ == "__main__":
    sys.exit(main())
