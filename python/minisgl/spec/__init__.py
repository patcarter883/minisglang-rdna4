"""Speculative decoding for minisgl (Triton-free, native-HIP).

MVP: n-gram / prompt-lookup proposal + greedy verify on MHA models, reusing the existing
extend-prefill paged HIP kernel for the verify forward (no new kernel). See ``SPEC_DECODE.md``
for the full architecture, the overlap-scheduling decision, and the phased plan toward
MTP/EAGLE draft heads and an MLA multi-query verify kernel.
"""

from __future__ import annotations

from .accept import AcceptResult, verify_greedy
from .config import SPEC_ALGORITHMS, SpecConfig
from .proposer import propose_ngram

__all__ = ["AcceptResult", "verify_greedy", "propose_ngram", "SpecConfig", "SPEC_ALGORITHMS"]
