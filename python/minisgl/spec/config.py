from __future__ import annotations

from dataclasses import dataclass

__all__ = ["SpecConfig", "SPEC_ALGORITHMS"]

# Supported proposers. "ngram" = prompt-lookup (zero-model) — the MVP. "mtp" = the model's own
# appended next-token-prediction head (GLM-4.x / Qwen3.5), run autoregressively as a draft model.
# EAGLE/DFlash land later (see SPEC_DECODE.md §4); all reuse the same verify cycle, only the
# proposer changes.
SPEC_ALGORITHMS = ("ngram", "mtp")


@dataclass(frozen=True)
class SpecConfig:
    """Speculative-decoding configuration. Present (non-None on the engine) only when spec-decode
    is enabled; every spec code path is gated on it, so a default serve is byte-for-byte unchanged."""

    algorithm: str  # one of SPEC_ALGORITHMS
    num_draft: int  # K: draft tokens proposed per step; verify runs K+1 query positions/seq
    ngram_max: int  # largest trailing n-gram window the proposer matches on
    ngram_min: int = 1  # smallest window to fall back to

    def __post_init__(self) -> None:
        if self.algorithm not in SPEC_ALGORITHMS:
            raise ValueError(
                f"spec algorithm {self.algorithm!r} not in {SPEC_ALGORITHMS}"
            )
        if self.num_draft < 1:
            raise ValueError(f"spec_num_draft must be >= 1, got {self.num_draft}")
        if not (1 <= self.ngram_min <= self.ngram_max):
            raise ValueError(
                f"require 1 <= ngram_min <= ngram_max, got "
                f"min={self.ngram_min} max={self.ngram_max}"
            )
