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
    bits: int  # 4
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
            gs, bits, sym = 32, 4, True
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
                break
            return cls(
                method="compressed-tensors", bits=bits, group_size=gs, sym=sym,
                ignore=_norm_ignore(ignore),
            )
        return None  # unsupported scheme -> treat as unquantized (will likely fail to load)
