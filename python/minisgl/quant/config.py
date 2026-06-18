from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class QuantConfig:
    """Parsed weight-quantization config (W4A8 family). Phase 2 targets AWQ (dense,
    uniform — all proj linears quantized) and compressed-tensors (Phase 3, has an
    ignore list). group_size is the checkpoint's; the kernel provider converts to its
    native layout (op group_size=32)."""

    method: str  # "awq" | "compressed-tensors"
    bits: int  # 4
    group_size: int  # 128 (AWQ) / 32 (compressed-tensors)
    sym: bool  # symmetric (no zero-point) vs asymmetric (AWQ zero_point=True -> False)
    ignore: tuple[str, ...] = ()  # module-name suffixes left unquantized (CT); () for AWQ

    @property
    def is_awq(self) -> bool:
        return self.method == "awq"

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

        if method == "awq":
            return cls(
                method="awq",
                bits=int(d.get("bits", 4)),
                group_size=int(d.get("group_size", 128)),
                sym=not bool(d.get("zero_point", True)),  # AWQ is asymmetric by default
            )
        if method in ("compressed-tensors", "compressed_tensors"):
            # Minimal parse; full per-group/ignore handling is Phase 3 (the 35B).
            ignore = tuple(d.get("ignore", ()) or ())
            gs = 32
            groups = d.get("config_groups") or {}
            for g in groups.values():
                w = (g or {}).get("weights") or {}
                if w.get("group_size"):
                    gs = int(w["group_size"])
                    break
            return cls(
                method="compressed-tensors", bits=4, group_size=gs, sym=True, ignore=ignore
            )
        return None  # unsupported scheme -> treat as unquantized (will likely fail to load)
