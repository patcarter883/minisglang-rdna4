from __future__ import annotations

import dataclasses
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


@dataclass(frozen=True)
class QuantConfig:
    """Parsed weight-quantization config (W4A8 family). Phase 2 targets AWQ (dense,
    uniform — all proj linears quantized) and compressed-tensors (Phase 3, has an
    ignore list). group_size is the checkpoint's; the kernel provider converts to its
    native layout (op group_size=32)."""

    method: str  # "awq" | "compressed-tensors" | "gptq" | "rxf"
    bits: int  # 4 (int4 W4A8/W4A16 family) | 8 (fp8 W8A8, compressed-tensors float-quantized)
    group_size: int  # 128 (AWQ/GPTQ) / 32 (compressed-tensors, rxf)
    sym: bool  # symmetric (no zero-point) vs asymmetric (AWQ zero_point=True -> False)
    ignore: tuple[str, ...] = ()  # module-name suffixes left unquantized (CT); () for AWQ
    # GPTQ act-order: when True the checkpoint reorders input channels by activation magnitude
    # (g_idx is a non-trivial permutation). False (the common case, e.g. Qwen1.5-MoE) -> g_idx is
    # the identity i//group_size and can be ignored on repack.
    desc_act: bool = False
    # RXF only: block-diagonal Hadamard rotation span (offline weights + runtime activations are
    # rotated by the same orthonormal FWHT-span; it cancels in the dot). 32 is the shipped default.
    rotation_span: int = 32
    # WEIGHT storage element type (compressed-tensors `config_groups[*].weights.type`):
    #   "int"   -> integer-quantized (int4 W4A8/W4A16, int8) — the AWQ/GPTQ/CT-int4 path.
    #   "float" -> float-quantized (fp8 e4m3 weights, e.g. ZAYA's W8A8; MXFP4 e2m1 in future).
    # AWQ/GPTQ/RXF are always integer, so this defaults "int"; only compressed-tensors reads it.
    weight_type: str = "int"
    # ACTIVATION scheme the checkpoint DECLARES for its quantized GEMMs (compressed-tensors
    # `input_activations`): "fp8" -> per-token dynamic fp8 acts (W8A8 — MUST be honored, the acts
    # are calibrated for it); None -> weight-only (activations stay in the compute dtype, W4A16/W8A16).
    # An env var must NEVER substitute a different activation scheme than the checkpoint declares.
    act_type: str | None = None
    # compressed-tensors `format` string (e.g. "mxfp4-pack-quantized", "nvfp4-pack-quantized",
    # "pack-quantized", "float-quantized"). It disambiguates the two float-4bit packings that
    # num_bits+type ALONE conflate: MXFP4 (group-32, E8M0 exponent scale, no global scale) vs NVFP4
    # (group-16, FP8-E4M3 block scale + per-tensor FP32 global scale, W4A4). Only compressed-tensors
    # sets it; None for every other method.
    ct_format: str | None = None
    # MIXED-PRECISION (compressed-tensors `format: mixed-precision`): one entry per `config_groups`
    # group, as (targets, scheme) — `targets` is the group's regex/substring module selector and
    # `scheme` is that group's own fully-parsed QuantConfig. EMPTY for every single-format checkpoint
    # (AWQ/GPTQ/RXF and single-group compressed-tensors), where the scalar fields above ARE the whole
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
    def is_awq(self) -> bool:
        return self.method == "awq"

    @property
    def is_rxf(self) -> bool:
        return self.method == "rxf"

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
        return self.bits == 4 and self.weight_type == "int" and not self.is_rxf

    def is_module_quantized(self, name: str, *, exact: bool = False) -> bool:
        """Is the weight module `name` (e.g. 'model.layers.47.mlp.experts.0.gate_proj') quantized
        under this config? False if `name` matches any `ignore` entry — a `re:`-prefixed regex
        (compressed-tensors / RXF) or a plain substring (AWQ/GPTQ `modules_to_not_convert`). This lets
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
        if method == "rxf":
            # RXF ("Rotated eXtra Fast") W4(NL codebook)-A8(int8) with a fixed Hadamard rotation.
            # Op layout already (weight_packed uint8 [N,K/2], weight_scale fp16 [N,K/32]); group=32,
            # symmetric NL codebook (no zero-points). Served by the native rxf_hip kernels.
            return cls(
                method="rxf",
                bits=4,
                group_size=32,
                sym=True,
                rotation_span=int(d.get("rotation_span", 32)),
                ignore=_norm_ignore(tuple(d.get("ignore", ()) or ())),  # e.g. a bf16 MTP head
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
                    # A mixed-precision group carries its OWN format ("float-quantized" /
                    # "nvfp4-pack-quantized"); single-format checkpoints declare it top-level only.
                    ct_format=(str((g or {}).get("format", "")).lower() or None) or group_fmt,
                )

            # Groups in DECLARATION order (dict order == the JSON's, which these checkpoints author
            # specific-before-general; see for_module). Skip entries with no `weights` block.
            ordered = [(k, g) for k, g in groups.items() if (g or {}).get("weights")]
            # MIXED-PRECISION: modules are split across groups by regex `targets`, so no single scalar
            # scheme describes the checkpoint — record every group and let `for_module` resolve. A
            # single-group checkpoint keeps ct_groups empty (the scalars below are the whole story).
            ct_groups: tuple[tuple[tuple[str, ...], "QuantConfig"], ...] = ()
            if len(ordered) > 1 or fmt == "mixed-precision":
                ct_groups = tuple(
                    (_norm_ignore(tuple(g.get("targets") or ())), _scheme(g, fmt))
                    for _, g in ordered
                )
            # Headline scalars stay the FIRST group's, so `quant.bits`/`group_size`/... keep meaning
            # for the single-format checkpoints (and for callers that only want a rough descriptor).
            head = _scheme(ordered[0][1], fmt) if ordered else cls(
                method="compressed-tensors", bits=4, group_size=32, sym=True, ignore=norm_ignore
            )
            return dataclasses.replace(head, ct_format=fmt, ct_groups=ct_groups)
        return None  # unsupported scheme -> treat as unquantized (will likely fail to load)
