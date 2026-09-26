from __future__ import annotations

import dataclasses
import fnmatch
import re
from dataclasses import dataclass
from typing import Any


def _norm_ignore(patterns: tuple[str, ...]) -> tuple[str, ...]:
    """Strip the multimodal wrapper infix so ignore entries — stored in the checkpoint's native
    key space (e.g. 'model.language_model.layers.0.linear_attn.in_proj_b') — match the loader's
    de-wrapped module names ('model.layers.0.linear_attn.in_proj_b'; the loader strips
    'language_model.'). `re:`-prefixed regexes are left untouched (the author controls them).

    DiffusionGemma nests the text stack under `model.decoder.` and the loader de-wraps that the same
    way, so the same strip must apply — and its failure mode is nastier than a missed ignore usually
    is. The dense MLP, the router and the self-conditioning block are ALL fp16 in that checkpoint via
    entries like 'model.decoder.layers.0.mlp.gate_proj'; if they do not match, every one of those
    modules is built QUANTIZED against tensors the checkpoint ships unpacked, and the model declares
    `weight_packed` where the loader emits `weight` — a wall of missing/extra keys at boot with no
    hint that a namespace mismatch caused it. Anchored at the start, unlike the `language_model.`
    infix strip, because `decoder.` is a plausible substring of a genuine module path."""
    out = []
    for p in patterns:
        if p and not p.startswith("re:"):
            p = p.replace("language_model.", "")
            if p.startswith("model.decoder."):
                p = "model." + p.removeprefix("model.decoder.")
        out.append(p)
    return tuple(out)


# ---- modelopt (NVIDIA TensorRT-ModelOpt) -> compressed-tensors normalization ------------------
#
# `quant_method: "modelopt"` describes the SAME on-disk formats compressed-tensors does; it only
# spells the header differently. Rather than grow a second parallel arm through every consumer
# (`is_nvfp4`, `weight_is_e2m1`, `for_module`, the linear/MoE method factories), a modelopt header is
# rewritten INTO the compressed-tensors shape at parse time and falls through the existing branch.
# Downstream sees `method="compressed-tensors"` and is untouched.
#
# Three things differ and each is silent-wrong if left alone:
#   1. the format lives in `quant_algo` ("NVFP4"), not in `format` — without the translation
#      `ct_format` is None, `is_nvfp4` is False, and the NVFP4 experts route into the MXFP4 W4A8
#      kernel (group-32 vs the real group-16, and an orphaned `weight_scale_2`);
#   2. `ignore` entries are fnmatch GLOBS ("*.self_attn.*", "*hyper_connection*"), not the plain
#      substrings compressed-tensors uses. Left as substrings, `"*.self_attn.*" in name` is False for
#      EVERY name (no literal asterisk in a module path), so the whole ignore list evaporates and the
#      bf16 attention / GDN / hyper-connection / shared-expert / PLE modules all build quantized
#      against tensors the checkpoint ships unpacked;
#   3. `targets` names a torch CLASS ("Linear"), not a module-path selector.
_MODELOPT_METHODS = ("modelopt", "modelopt_fp8", "modelopt_fp4")

# `quant_algo` -> the compressed-tensors `format` that denotes the identical on-disk layout.
# Deliberately a closed table: an algo that is not here is a packing this repo has no reader for, and
# returning None (-> `ModelConfig.unparsed_quant_method`) says so instead of guessing a format.
_MODELOPT_ALGO_FORMAT = {
    "NVFP4": "nvfp4-pack-quantized",  # E2M1 + per-16 e4m3 block scale + per-tensor fp32 global
    "NVFP4_AWQ": "nvfp4-pack-quantized",
    "FP8": "float-quantized",  # e4m3 weights (W8A8 when input_activations are declared)
    "FP8_PER_TENSOR": "float-quantized",
    "FP8_PER_CHANNEL_PER_TOKEN": "float-quantized",
}

# The weights/input_activations block `config_groups` would carry, per algo, for the headers that
# ship `quant_algo` ALONE (modelopt does this for whole-model schemes). Only used when there is no
# `config_groups` at all — otherwise the file's own numbers win. Without it the compressed-tensors
# branch would fall back to its group_size=32 default, which for NVFP4 is the WRONG block size and
# mis-strides every scale.
_MODELOPT_ALGO_SCHEME = {
    "nvfp4-pack-quantized": {
        "weights": {"num_bits": 4, "group_size": 16, "type": "float", "symmetric": True},
        "input_activations": {"num_bits": 4, "type": "float"},
    },
    "float-quantized": {
        "weights": {"num_bits": 8, "type": "float", "symmetric": True},
        "input_activations": {"num_bits": 8, "type": "float"},
    },
}

_GLOB_META = ("*", "?", "[")


def _glob_to_ignore(pattern: str) -> str:
    """One modelopt `ignore` glob -> an entry `is_module_quantized` matches correctly.

    A pattern with no glob metacharacter ("lm_head", "model.embed_tokens") is left as a plain entry
    so it keeps the historical substring semantics every other checkpoint relies on. A pattern that
    DOES glob is translated to a fully anchored `re:` regex — anchored because a glob is a whole-name
    match, and an unanchored `mtp\\..*` would also swallow a hypothetical `model.mtp_proj.*`.
    """
    if pattern.startswith("re:") or not any(c in pattern for c in _GLOB_META):
        return pattern
    # fnmatch.translate already appends \Z; \A closes the front.
    return "re:" + r"\A" + fnmatch.translate(pattern)


def _modelopt_targets(targets: tuple) -> tuple:
    """modelopt `targets` are torch CLASS names ("Linear"), compressed-tensors' are module-path
    selectors. A class name reaches `for_module` as a substring test that matches nothing, so a
    MULTI-group modelopt header would resolve every module to "unquantized" — the whole model built
    full precision with no error. Map a bare class-like token to the catch-all it means; leave
    anything that looks like a path (has a dot or a glob) to the normal selector logic."""
    out = []
    for t in targets:
        t = str(t)
        if t and "." not in t and not any(c in t for c in _GLOB_META):
            out.append("re:.*")  # a class-name target selects every Linear, i.e. everything
        else:
            out.append(_glob_to_ignore(t))
    return tuple(out)


def _modelopt_to_compressed_tensors(d: dict) -> "dict | None":
    """Rewrite a modelopt `quantization_config` dict into the compressed-tensors shape, or None if
    its `quant_algo` names a packing this repo cannot read."""
    algo = str(d.get("quant_algo") or "").upper()
    fmt = _MODELOPT_ALGO_FORMAT.get(algo)
    if fmt is None:
        return None
    groups = d.get("config_groups") or {}
    if not any((g or {}).get("weights") for g in groups.values()):
        groups = {"group_0": dict(_MODELOPT_ALGO_SCHEME[fmt], targets=("re:.*",))}
    else:
        groups = {
            k: dict(g, targets=_modelopt_targets(tuple(g.get("targets") or ())))
            for k, g in groups.items()
        }
    return {
        "quant_method": "compressed-tensors",
        "format": fmt,
        "config_groups": groups,
        # ORDER MATTERS: `_norm_ignore` (the `language_model.` / `model.decoder.` de-wrap, so an
        # entry written in the checkpoint's wrapped key space matches the loader's de-wrapped module
        # names) skips anything already `re:`-prefixed. Run it on the raw GLOB text first, then
        # translate. Doing it the other way round leaves `model.language_model.embed_tokens`
        # un-de-wrapped and it silently matches nothing. `_norm_ignore` runs again inside the
        # compressed-tensors branch; it is a no-op on the `re:` entries this produces.
        "ignore": tuple(_glob_to_ignore(p) for p in _norm_ignore(tuple(d.get("ignore") or ()))),
        # Carried through untouched for any consumer that reads them off the raw dict.
        "kv_cache_scheme": d.get("kv_cache_quant_algo"),
        "modelopt_quant_algo": algo,
    }


# ---- Quark (AMD) -> compressed-tensors normalization -------------------------------------------
#
# Same trick as modelopt above, for the same reason: `quant_method: "quark"` describes on-disk
# layouts compressed-tensors already names, so it is rewritten at parse time and every downstream
# consumer (`weight_is_e2m1`, `is_nvfp4`, the method factories) keeps ONE code path.
#
# Quark spells four things differently, and three of them are silent-wrong if ignored:
#   1. the format is a (dtype, scale_format) PAIR under `global_quant_config.weight` -- "fp4"+"e8m0"
#      is MXFP4, "fp4"+"e4m3" is NVFP4. There is no `format` string to read.
#   2. `exclude` is Quark's `ignore`. These are plain module paths (not globs), so they feed
#      `_norm_ignore` directly.
#   3. per-layer overrides live in `layer_quant_config` / `layer_type_quant_config`. This repo has no
#      reader for a mixed-precision-per-layer scheme, so a NON-EMPTY override table REFUSES rather
#      than quietly quantizing those layers with the global spec.
#   4. `export.pack_method` describes the on-disk nibble order. "reorder" is the layout the
#      compressed-tensors MXFP4/NVFP4 readers already expect; anything else is a packing this repo
#      has no reader for and must refuse, because a wrong nibble order does not fail -- it returns
#      plausible, finite, wrong numbers.
#
# ACTIVATIONS ARE DELIBERATELY DROPPED. Quark/ModelOpt checkpoints declare `input_tensors`, but
# gfx1201 has no FP4 arithmetic and this engine never consumes a checkpoint's activation scheme for
# the e2m1 formats -- it picks W4A8 (per-token fp8) or W4A16 (unquantized) at the METHOD, exactly as
# it does for a compressed-tensors MXFP4 file, which declares `input_activations: null`. Emitting
# null here keeps those two identical rather than inventing an activation scheme no kernel
# implements.
#
# CORRECTION 2026-09-22: this comment used to say the checkpoint "asks for dynamic per-group fp4".
# It does not. Qwen3.8-Flash-Next's config declares
#     input_activations: {dynamic: False, group_size: 16, num_bits: 4, type: float}
# i.e. STATIC per-group fp4, and it ships the scales to match -- 1536 `*.input_scale` /
# `*.input_global_scale` tensors that the loader counts in its ignore ledger. So what is dropped is
# not an unusable runtime scheme, it is REAL CALIBRATION DATA, replaced by dynamic per-token fp8.
# That substitution is still the only thing gfx1201 can execute (no fp4 math, and a per-group fp4
# scale is neither the right format nor the right granularity for a per-token fp8 quantizer), and
# per-token amax is arguably more adaptive than a static table. But it has never been MEASURED for
# accuracy against the scheme the checkpoint was calibrated for, and it is worth being accurate about
# which of those two things is true: the substitution is forced, not free.
def _ct_targets_to_patterns(targets: tuple) -> tuple:
    """compressed-tensors `targets` -> module-name patterns this engine can match.

    A `targets` entry is EITHER a module selector (`re:.*\\.self_attn\\.q_proj$`, or a dotted path)
    OR the NAME OF A TORCH MODULE CLASS — `["Linear"]` is the canonical spelling for "every
    nn.Linear in the model", and it is what a single-group compressed-tensors checkpoint almost
    always ships. Matched as a name pattern it selects NOTHING, because no module is called
    "Linear": on Qwen3.8-Flash-Next-MXFP4-FP8-GPTQ that made `for_module` return None for every
    routed expert, i.e. served the MXFP4 bulk of the model as though it were unquantized while the
    checkpoint ships `weight_packed`. So a class-name target becomes the catch-all `re:.*`, and the
    `ignore` list plus `ckpt_quantized` do the narrowing they already do.

    Detection is deliberately narrow — a bare identifier with no dot, no regex prefix and an
    upper-case initial. Module paths in the wild are lower-case and dotted (`model.layers.0...`), and
    a genuine regex carries the `re:` prefix, so neither is mistaken for a class.

    Returns `(patterns, is_catchall)`. The flag matters for ORDERING: see the ct_groups construction.
    """
    out, catchall = [], False
    for t in targets:
        t = str(t)
        if (not t.startswith("re:") and "." not in t and "/" not in t
                and t[:1].isupper() and t.isidentifier()):
            out.append("re:.*")
            catchall = True
        else:
            out.append(t)
    return tuple(out), catchall


def _group_own_format(w: dict) -> "str | None":
    """A compressed-tensors group's weight FORMAT from its own dtype, when it declares none.

    Only decides the cases the headline format cannot be trusted for. 4-bit float is ambiguous on
    (num_bits, type) alone — MXFP4 and NVFP4 differ by group size and scale structure — so it is left
    to the declared/inherited format, exactly as before. 8-bit float is not ambiguous: it is
    `float-quantized` (fp8 e4m3), whatever a multi-group checkpoint says at the top level.
    """
    if not w:
        return None
    if str(w.get("type", "")).lower() == "float" and int(w.get("num_bits") or 0) == 8:
        return "float-quantized"
    return None


_QUARK_METHODS = ("quark",)

# (weight dtype, scale_format) -> the compressed-tensors `format` naming the identical layout.
# Closed table on purpose: an unlisted pair is a packing with no reader here, and returning None
# (-> `ModelConfig.unparsed_quant_method`) names it instead of guessing.
_QUARK_WEIGHT_FORMAT = {
    ("fp4", "e8m0"): "mxfp4-pack-quantized",   # E2M1 codes + per-group E8M0 exponent
    ("fp4", "e4m3"): "nvfp4-pack-quantized",   # E2M1 codes + per-group e4m3 + per-tensor global
    ("fp8_e4m3", "float"): "float-quantized",
    ("fp8_e4m3", ""): "float-quantized",
}
_QUARK_PACK_METHODS = ("reorder",)


def _quark_to_compressed_tensors(d: dict) -> "dict | None":
    """Rewrite a Quark `quantization_config` into the compressed-tensors shape, or None if it names
    a scheme this repo has no reader for."""
    gq = d.get("global_quant_config") or {}
    w = gq.get("weight") or gq.get("weights") or {}
    dtype = str(w.get("dtype") or "").lower()
    sfmt = str(w.get("scale_format") or "").lower()
    fmt = _QUARK_WEIGHT_FORMAT.get((dtype, sfmt))
    if fmt is None:
        return None
    if w.get("is_dynamic"):          # weights are static by construction; dynamic means we misread it
        return None
    if d.get("layer_quant_config") or d.get("layer_type_quant_config"):
        return None                  # per-layer overrides: no reader (see note 3)
    pack = str((d.get("export") or {}).get("pack_method") or "reorder").lower()
    if pack not in _QUARK_PACK_METHODS:
        return None                  # unknown nibble order: wrong is SILENT here (see note 4)
    gs = int(w.get("group_size") or 32)
    return {
        "quant_method": "compressed-tensors",
        "format": fmt,
        "config_groups": {
            "group_0": {
                "weights": {
                    "num_bits": 4 if dtype == "fp4" else 8,
                    "type": "float",
                    "strategy": "group",
                    "group_size": gs,
                    "symmetric": True,
                },
                # See the module note: dropped on purpose, matching a compressed-tensors MXFP4 file.
                "input_activations": None,
                "targets": ["Linear"],
            }
        },
        # STRIP THE PARAMETER SUFFIX. Quark excludes name TENSORS ("mtp.layers.0.mlp.gate_proj.weight"),
        # compressed-tensors `ignore` names MODULES ("mtp.layers.0.mlp.gate_proj"), and the match is a
        # substring test against the module name. An entry that is LONGER than the module name matches
        # nothing, so every suffixed exclude evaporates — measured: `mtp.layers.0.mlp.gate_proj` came
        # back is_module_quantized=True against a checkpoint that ships that tensor unquantized, which
        # builds a quantized layer over full-precision weights. `lm_head` hid this because it is the
        # one entry Quark writes without a suffix.
        "ignore": _norm_ignore(tuple(
            e[: -len(sfx)] if (sfx := next((x for x in (".weight", ".bias", ".weight_scale")
                                            if e.endswith(x)), "")) else e
            for e in (d.get("exclude") or ())
        )),
        "quark_pack_method": pack,
    }


@dataclass(frozen=True)
class QuantConfig:
    """Parsed weight-quantization config (W4A8 family). Phase 2 targets AWQ (dense,
    uniform — all proj linears quantized) and compressed-tensors (Phase 3, has an
    ignore list). group_size is the checkpoint's; the kernel provider converts to its
    native layout (op group_size=32)."""

    method: str  # "awq" | "compressed-tensors" | "gptq"
    bits: int  # 4 (int4 W4A8/W4A16 family) | 8 (fp8 W8A8, compressed-tensors float-quantized)
    group_size: int  # 128 (AWQ/GPTQ) / 32 (compressed-tensors)
    sym: bool  # symmetric (no zero-point) vs asymmetric (AWQ zero_point=True -> False)
    ignore: tuple[str, ...] = ()  # module-name suffixes left unquantized (CT); () for AWQ
    # GPTQ act-order: when True the checkpoint reorders input channels by activation magnitude
    # (g_idx is a non-trivial permutation). False (the common case, e.g. Qwen1.5-MoE) -> g_idx is
    # the identity i//group_size and can be ignored on repack.
    desc_act: bool = False
    # WEIGHT storage element type (compressed-tensors `config_groups[*].weights.type`):
    #   "int"   -> integer-quantized (int4 W4A8/W4A16, int8) — the AWQ/GPTQ/CT-int4 path.
    #   "float" -> float-quantized (fp8 e4m3 weights, e.g. ZAYA's W8A8; MXFP4 e2m1 in future).
    # AWQ/GPTQ are always integer, so this defaults "int"; only compressed-tensors reads it.
    weight_type: str = "int"
    # ACTIVATION scheme the checkpoint DECLARES for its quantized GEMMs (compressed-tensors
    # `input_activations`): "fp8" -> per-token dynamic fp8 acts (W8A8 — MUST be honored, the acts
    # are calibrated for it); None -> weight-only (activations stay in the compute dtype, W4A16/W8A16).
    # An env var must NEVER substitute a different activation scheme than the checkpoint declares.
    act_type: str | None = None
    # WEIGHT quantization GRANULARITY (compressed-tensors `config_groups[*].weights.strategy`):
    # "group" (a scale per `group_size` along K), "channel" (per output row), "tensor" (one scalar),
    # or "block" (a scale per `block_structure` tile — 2-D, varying along BOTH N and K).
    #
    # Parsed because dropping it is not neutral. `strategy: "block"` with `group_size: null` used to
    # fall through to the `else 32` default below and be described as group-32 — a scheme the
    # checkpoint does not contain, silently. Qwen3.8-Flash-Next-MXFP4-FP8-GPTQ ships exactly that for
    # its attention and GDN projections (fp8, block [128,128]), so it would have been served as if
    # its scale varied along K in 32-element groups when it actually tiles 128x128.
    weight_strategy: str | None = None
    # The `block_structure` tile, e.g. (128, 128). None for every non-block strategy.
    block_structure: tuple[int, ...] | None = None
    # compressed-tensors `format` string (e.g. "mxfp4-pack-quantized", "nvfp4-pack-quantized",
    # "pack-quantized", "float-quantized"). It disambiguates the two float-4bit packings that
    # num_bits+type ALONE conflate: MXFP4 (group-32, E8M0 exponent scale, no global scale) vs NVFP4
    # (group-16, FP8-E4M3 block scale + per-tensor FP32 global scale, W4A4). Only compressed-tensors
    # sets it; None for every other method.
    ct_format: str | None = None
    # MIXED-PRECISION (compressed-tensors `format: mixed-precision`): one entry per `config_groups`
    # group, as (targets, scheme) — `targets` is the group's regex/substring module selector and
    # `scheme` is that group's own fully-parsed QuantConfig. EMPTY for every single-format checkpoint
    # (AWQ/GPTQ and single-group compressed-tensors), where the scalar fields above ARE the whole
    # story and `for_module` degenerates to "self, unless ignored" — i.e. behaviour is unchanged.
    # Populated only when a checkpoint genuinely mixes schemes across modules, e.g. Qwen3.8-27B-NVFP4:
    # NVFP4 for the bulk MLP, fp8 W8A8 for attention / GDN in_proj / lm_head / the last 8 MLP layers.
    ct_groups: tuple[tuple[tuple[str, ...], "QuantConfig"], ...] = ()
    # Modules the CHECKPOINT actually ships in quantized form (native/de-wrapped names), derived from
    # its tensor index. None = unknown, fall back to matching the `ignore` list.
    #
    # This exists because a container entry in `ignore` is ambiguous, and the two conventions in the
    # wild contradict each other. GLM-4.7-Flash-AWQ ignores `model.layers.0` meaning "all of dense
    # layer 0 stays bf16" (prefix semantics — 341 modules depend on it), while Qwen3.8-NVFP4 ignores
    # `...layers.N.linear_attn` meaning ONLY the container, with its children in_proj_qkv/z/out_proj
    # quantized. No string rule satisfies both; the checkpoint's own tensors do, unambiguously — so
    # the ignore list becomes advisory and the shipped tensors decide.
    ckpt_quantized: frozenset[str] | None = None

    @property
    def is_fp8_block(self) -> bool:
        """DeepSeek-style BLOCKWISE fp8: e4m3 weights with a 2-D scale per `block_structure` tile.

        Distinct from `is_fp8_w8a8`, which is per-OUTPUT-CHANNEL (one scale per row, `(N,1)`) and
        folds into the GEMM epilogue. A block scale varies along K as well, so it cannot be an
        epilogue factor and no kernel here consumes it — the method dequantizes to bf16 at load
        instead (see Fp8BlockLinearMethod). Checked BEFORE is_fp8_w8a8 at the dispatch,
        because a block checkpoint satisfies both and the channel method would try to load a
        `(N,1)` scale where the file has `(N/128, K/128)`.
        """
        return (self.is_compressed_tensors and self.weight_type == "float" and self.bits == 8
                and self.weight_strategy == "block" and bool(self.block_structure))

    @property
    def is_awq(self) -> bool:
        return self.method == "awq"

    @property
    def is_gptq(self) -> bool:
        return self.method == "gptq"

    @property
    def is_compressed_tensors(self) -> bool:
        return self.method == "compressed-tensors"

    @property
    def is_fp8_w8a8(self) -> bool:
        """True for the fp8 W8A8 scheme: compressed-tensors *float-quantized* 8-bit weights
        (F8_E4M3) with per-token fp8 activations (e.g. ZAYA). Routed to the native w8a8_moe /
        w8a8 fp8-WMMA kernels — NOT the int4 W4A8 path. `act_type == 'fp8'` marks the calibrated
        per-token activation quant that MUST be honored (an env may pick W8A16 as a perf opt-in but
        never silently swap the declared activation scheme)."""
        return self.is_compressed_tensors and self.weight_type == "float" and self.bits == 8

    @property
    def is_nvfp4(self) -> bool:
        """NVFP4 (compressed-tensors `nvfp4-pack-quantized`): E2M1 4-bit weights with a per-16-group
        FP8-E4M3 block scale AND a per-tensor FP32 global scale, plus FP4 activations (a W4A4 scheme).
        This is a DIFFERENT format from MXFP4 (group-32 E8M0 exponent scale, no global scale) that
        num_bits+weight_type alone cannot tell apart — hence the `format` gate. gfx1201 has no FP4
        WMMA / FP4-activation path, so NVFP4 is DETECTED here and rejected explicitly at method
        creation rather than silently misrouted into the MXFP4 W4A8 kernel (which would crash deep in
        the loader on the FP8 scale dtype / group-16 mismatch / orphaned global-scale tensors)."""
        return self.is_compressed_tensors and self.ct_format == "nvfp4-pack-quantized"

    @property
    def weight_is_e2m1(self) -> bool:
        """MXFP4: compressed-tensors float-quantized 4-bit (OCP E2M1 weights + E8M0 per-32-block
        scale, `format: mxfp4-pack-quantized`). Served through the SAME W4A8 fp8-WMMA kernel as int4
        with `weight_is_e2m1=True` — a different 4-bit decode table + E8M0->fp16 group scale, not a
        new kernel. Routes MxFp4LinearMethod (dense) / _MxFp4MoEMethod (experts). EXCLUDES NVFP4
        (also float-4bit) — that format has an incompatible scale layout and is handled by is_nvfp4."""
        return (self.is_compressed_tensors and self.weight_type == "float"
                and self.bits == 4 and not self.is_nvfp4)

    @property
    def is_int4(self) -> bool:
        """Integer 4-bit weight family (AWQ / GPTQ / compressed-tensors int4) — the shared W4A8
        expert/linear kernel (int4 weight x per-token fp8 act, or true W4A16 where flagged)."""
        return self.bits == 4 and self.weight_type == "int"

    def is_module_quantized(self, name: str, *, exact: bool = False) -> bool:
        """Is the weight module `name` (e.g. 'model.layers.47.mlp.experts.0.gate_proj') quantized
        under this config? False if `name` matches any `ignore` entry — a `re:`-prefixed regex
        (compressed-tensors) or a plain substring (AWQ/GPTQ `modules_to_not_convert`). This lets
        a checkpoint keep specific modules at full precision (bf16/fp16) on an otherwise-quantized
        backbone — an MTP / draft head, the router gate, dense early layers — and the model build it
        unquantized accordingly (universal: not tied to any one model or quant method).

        `exact` selects how a PLAIN (non-`re:`) entry matches. The historical default is a bare
        substring test, which is right for the suffix-shaped entries these lists usually carry
        ('lm_head', 'mlp.gate_proj') but WRONG for an entry naming a container module: the ignore
        entry 'model.layers.0.linear_attn' is the GDN module itself, yet as a substring it also
        swallows its children 'model.layers.0.linear_attn.in_proj_qkv'/'.in_proj_z'/'.out_proj' —
        which a mixed-precision config_group explicitly TARGETS as fp8. `exact=True` requires the
        pattern to align to a dotted path boundary and run to the END of the name, so a container
        entry matches only itself. Used by `for_module` for mixed-precision checkpoints (where the
        positive `targets` selector makes the distinction load-bearing); the substring default is
        kept everywhere else so no existing checkpoint changes behaviour."""
        for pat in self.ignore:
            if not pat:
                continue
            if pat.startswith("re:"):
                if re.search(pat[3:], name):
                    return False
            elif (name == pat or name.endswith("." + pat)) if exact else (pat in name):
                return False
        return True

    def for_module(self, name: str) -> "QuantConfig | None":
        """The EFFECTIVE scheme for weight module `name`, or None if it is unquantized.

        This is the single place a caller should ask "how is this module quantized?" — it folds the
        two independent reasons a module may differ from the checkpoint's headline scheme:

          1. the `ignore` list (module kept at full precision)          -> None
          2. MIXED-PRECISION `config_groups` (module in a different     -> that group's QuantConfig
             group than its neighbours, selected by the group `targets`)

        Single-format checkpoints have no `ct_groups`, so this returns `self` for every non-ignored
        module — byte-identical to the previous `is_module_quantized(name) -> create_linear_method(q)`
        pattern it replaces. Under mixed-precision a module matching NO group is unquantized (None):
        compressed-tensors only quantizes what a group's `targets` selects.

        Groups are tried in declaration order and the FIRST match wins, which is what the specific-
        before-general layout these checkpoints ship requires: Qwen3.8-27B lists the 8 fp8 MLP layers
        (`layers.(56|...|63).mlp.(gate|up|down)_proj`) in group_0 and the catch-all NVFP4 MLP
        (`.*mlp.(gate|up|down)_proj`) in group_1, so the narrow rule must be consulted first."""
        if self.ckpt_quantized is not None:
            # STRUCTURAL and authoritative: the checkpoint either ships this module quantized or it
            # does not. Immune to the container-entry ambiguity the ignore list cannot express.
            if name not in self.ckpt_quantized:
                return None
        elif not self.is_module_quantized(name, exact=bool(self.ct_groups)):
            return None
        if not self.ct_groups:
            return self
        for targets, scheme in self.ct_groups:
            for pat in targets:
                if not pat:
                    continue
                if pat.startswith("re:"):
                    if re.search(pat[3:], name):
                        return scheme
                elif pat in name:
                    return scheme
        return None

    @staticmethod
    def _as_dict(qc: Any) -> dict:
        if isinstance(qc, dict):
            return qc
        if hasattr(qc, "to_dict"):
            return qc.to_dict()
        return dict(getattr(qc, "__dict__", {}))

    @classmethod
    def from_hf(cls, hf_config: Any) -> "QuantConfig | None":
        # quantization_config sits on the top-level config (also check text_config).
        qc = getattr(hf_config, "quantization_config", None)
        if qc is None and getattr(hf_config, "text_config", None) is not None:
            qc = getattr(hf_config.text_config, "quantization_config", None)
        if qc is None:
            return None
        d = cls._as_dict(qc)
        method = str(d.get("quant_method", "")).lower()

        # modelopt is compressed-tensors wearing a different header (see the module notes above):
        # normalize and fall through, so `is_nvfp4`/`weight_is_e2m1`/`for_module` and every method
        # factory keep exactly one code path. An algo we cannot read normalizes to None and the
        # function returns None, which `ModelConfig.unparsed_quant_method` reports by name.
        if method in _MODELOPT_METHODS:
            normalized = _modelopt_to_compressed_tensors(d)
            if normalized is None:
                return None
            d, method = normalized, "compressed-tensors"

        # Quark (AMD) is the same story — normalize and fall through (see the module notes above).
        if method in _QUARK_METHODS:
            normalized = _quark_to_compressed_tensors(d)
            if normalized is None:
                return None
            d, method = normalized, "compressed-tensors"

        # modules_to_not_convert (AWQ/GPTQ): plain module-name substrings kept at full precision
        # (attn, router gate, an unquantized MTP/draft head). Folded into `ignore` so is_module_quantized
        # is uniform across methods.
        not_convert = tuple(d.get("modules_to_not_convert") or ())
        if method == "awq":
            return cls(
                method="awq",
                bits=int(d.get("bits", 4)),
                group_size=int(d.get("group_size", 128)),
                sym=not bool(d.get("zero_point", True)),  # AWQ is asymmetric by default
                ignore=_norm_ignore(not_convert),
            )
        if method == "gptq":
            # GPTQ int4: qweight int32 packed along INPUT (K//pf, N), per-group scales (K//g, N),
            # qzeros (K//g, N//pf). `sym` true -> symmetric (the op's zeros=None path). desc_act
            # true would need an activation-order permutation (g_idx); we assert it off where used.
            return cls(
                method="gptq",
                bits=int(d.get("bits", 4)),
                group_size=int(d.get("group_size", 128)),
                sym=bool(d.get("sym", True)),
                desc_act=bool(d.get("desc_act", False)),
                ignore=_norm_ignore(not_convert),
            )
        if method in ("compressed-tensors", "compressed_tensors"):
            # Read group_size / num_bits / symmetric off the FIRST weights group (uniform across
            # groups for these checkpoints). ASYMMETRIC (symmetric:false) ships a per-group
            # weight_zero_point tensor; the linear method loads and uses it (vs the symmetric
            # constant zero-point 8). Config-driven, not model-specific.
            ignore = tuple(d.get("ignore", ()) or ())
            fmt = str(d.get("format", "")).lower() or None
            groups = d.get("config_groups") or {}
            norm_ignore = _norm_ignore(ignore)

            def _scheme(g: dict, group_fmt: str | None) -> "QuantConfig":
                """One config_groups entry -> a fully-parsed scheme. `group_fmt` is the group's own
                `format` when it declares one (mixed-precision), else the top-level format."""
                w = (g or {}).get("weights") or {}
                # ACTIVATION scheme the checkpoint declares (per-token fp8 -> W8A8). Honor it: an env
                # var never substitutes a different act scheme than declared (W8A16 is only an OPT-IN
                # perf override, never the default). Present + float type -> "fp8"; else weight-only.
                ia = (g or {}).get("input_activations") or {}
                return cls(
                    method="compressed-tensors",
                    # WEIGHT element type: "float" (fp8 e4m3 W8A8, e.g. ZAYA — `float-quantized`) vs
                    # "int" (int4/int8). Drives the fp8-vs-int4 kernel selection downstream.
                    bits=int(w["num_bits"]) if w.get("num_bits") else 4,
                    group_size=int(w["group_size"]) if w.get("group_size") else 32,
                    sym=bool(w["symmetric"]) if w.get("symmetric") is not None else True,
                    ignore=norm_ignore,
                    weight_type=str(w["type"]).lower() if w.get("type") else "int",
                    act_type="fp8" if ia and str(ia.get("type", "")).lower() == "float" else None,
                    weight_strategy=(str(w["strategy"]).lower() if w.get("strategy") else None),
                    block_structure=(tuple(int(b) for b in w["block_structure"])
                                     if w.get("block_structure") else None),
                    # A mixed-precision group carries its OWN format ("float-quantized" /
                    # "nvfp4-pack-quantized"); single-format checkpoints declare it top-level only.
                    #
                    # A group must NOT inherit a format its own dtype contradicts. The top-level
                    # format names a WEIGHT PACKING, and in a multi-group checkpoint the groups need
                    # not share one: Qwen3.8-Flash-Next-MXFP4-FP8-GPTQ declares
                    # `format: mxfp4-pack-quantized` at the top while its group_1 is 8-bit float,
                    # i.e. fp8. Inheriting there tagged an fp8 group as MXFP4, which routes it to the
                    # e2m1 nibble decode — plausible, finite, wrong numbers. 8-bit float is
                    # `float-quantized` no matter what the headline says.
                    ct_format=(str((g or {}).get("format", "")).lower() or None)
                    or _group_own_format(w)
                    or group_fmt,
                )

            # Groups in DECLARATION order (dict order == the JSON's, which these checkpoints author
            # specific-before-general; see for_module). Skip entries with no `weights` block.
            ordered = [(k, g) for k, g in groups.items() if (g or {}).get("weights")]
            # MIXED-PRECISION: modules are split across groups by regex `targets`, so no single scalar
            # scheme describes the checkpoint — record every group and let `for_module` resolve. A
            # single-group checkpoint keeps ct_groups empty (the scalars below are the whole story).
            ct_groups: tuple[tuple[tuple[str, ...], "QuantConfig"], ...] = ()
            if len(ordered) > 1 or fmt == "mixed-precision":
                # ORDER: explicit targets first, CLASS-LEVEL catch-alls last — regardless of
                # declaration order. `for_module` is first-match-wins, and the two layouts in the
                # wild disagree about what that should mean. Qwen3.8-27B-NVFP4 ships
                # specific-before-general with both groups explicit, so declaration order is already
                # right for it and partitioning leaves it untouched. Qwen3.8-Flash-Next-MXFP4-FP8-GPTQ
                # ships the opposite: group_0 is `["Linear"]` (every Linear) and the fp8 attention/GDN
                # regexes come after — so honouring declaration order handed every attention and GDN
                # projection to the MXFP4 e2m1 decode when the file ships them as blockwise fp8.
                # A class-level target is compressed-tensors' FALLBACK ("any Linear not otherwise
                # specified"), so it belongs last; that makes both layouts resolve correctly without
                # either checkpoint having to declare an order it does not control.
                _parsed = [(_norm_ignore(pats), _scheme(g, fmt), ca)
                           for _, g in ordered
                           for pats, ca in (_ct_targets_to_patterns(tuple(g.get("targets") or ())),)]
                ct_groups = tuple((p, sc) for p, sc, ca in _parsed if not ca) + tuple(
                    (p, sc) for p, sc, ca in _parsed if ca
                )
            # Headline scalars stay the FIRST group's, so `quant.bits`/`group_size`/... keep meaning
            # for the single-format checkpoints (and for callers that only want a rough descriptor).
            head = _scheme(ordered[0][1], fmt) if ordered else cls(
                method="compressed-tensors", bits=4, group_size=32, sym=True, ignore=norm_ignore
            )
            return dataclasses.replace(head, ct_format=fmt, ct_groups=ct_groups)
        return None  # unsupported scheme -> treat as unquantized (will likely fail to load)
