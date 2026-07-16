from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


def _norm_ignore(patterns: tuple[str, ...]) -> tuple[str, ...]:
    """Strip the multimodal wrapper infix so ignore entries — stored in the checkpoint's native
    key space (e.g. 'model.language_model.layers.0.linear_attn.in_proj_b') — match the loader's
    de-wrapped module names ('model.layers.0.linear_attn.in_proj_b'; the loader strips
    'language_model.'). `re:`-prefixed regexes are left untouched (the author controls them)."""
    out = []
    for p in patterns:
        if p and not p.startswith("re:"):
            p = p.replace("language_model.", "")
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

    def is_module_quantized(self, name: str) -> bool:
        """Is the weight module `name` (e.g. 'model.layers.47.mlp.experts.0.gate_proj') quantized
        under this config? False if `name` matches any `ignore` entry — a `re:`-prefixed regex
        (compressed-tensors / RXF) or a plain substring (AWQ/GPTQ `modules_to_not_convert`). This lets
        a checkpoint keep specific modules at full precision (bf16/fp16) on an otherwise-quantized
        backbone — an MTP / draft head, the router gate, dense early layers — and the model build it
        unquantized accordingly (universal: not tied to any one model or quant method)."""
        for pat in self.ignore:
            if not pat:
                continue
            if pat.startswith("re:"):
                if re.search(pat[3:], name):
                    return False
            elif pat in name:
                return False
        return True

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
            gs, bits, sym = 32, 4, True
            wtype, atype = "int", None
            groups = d.get("config_groups") or {}
            for g in groups.values():
                w = (g or {}).get("weights") or {}
                if not w:
                    continue
                if w.get("group_size"):
                    gs = int(w["group_size"])
                if w.get("num_bits"):
                    bits = int(w["num_bits"])
                if "symmetric" in w and w["symmetric"] is not None:
                    sym = bool(w["symmetric"])
                # WEIGHT element type: "float" (fp8 e4m3 W8A8, e.g. ZAYA — `format:float-quantized`)
                # vs "int" (int4/int8). Drives the fp8-vs-int4 kernel selection downstream.
                if w.get("type"):
                    wtype = str(w["type"]).lower()
                # ACTIVATION scheme the checkpoint declares (per-token fp8 -> W8A8). Honor it: an env
                # var never substitutes a different act scheme than declared (W8A16 is only an OPT-IN
                # perf override, never the default). Present + float type -> "fp8"; else weight-only.
                ia = (g or {}).get("input_activations") or {}
                if ia and str(ia.get("type", "")).lower() == "float":
                    atype = "fp8"
                break
            return cls(
                method="compressed-tensors", bits=bits, group_size=gs, sym=sym,
                ignore=_norm_ignore(ignore), weight_type=wtype, act_type=atype, ct_format=fmt,
            )
        return None  # unsupported scheme -> treat as unquantized (will likely fail to load)
