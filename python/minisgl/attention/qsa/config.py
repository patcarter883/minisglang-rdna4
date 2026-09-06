"""QSA (query-sparse-attention) profile — the ONE place that reads the indexer fields.

Ported from `sglang/srt/layers/attention/qsa/config.py`, kept to the `compressed` (Qwen4-Exp)
variant because that is the only schema this engine has a checkpoint for. The tokenwise
(qsa_0511 / Qwen3.5-DSA) variant is deliberately NOT stubbed: a half-parsed profile that
selects the wrong indexer is worse than a missing one.

WHAT THE FIELDS MEAN (values from `RadixArk/Qwen3.8-Flash-Next-NVFP4/config.json`):

    indexer_n_heads        4     index QUERY heads  (Hi)
    indexer_kv_heads       1     index KEY heads    — MQA; the scoring op requires exactly 1
    indexer_head_dim     128     per-head index width (di)
    indexer_budget      2048     TOKENS each query row may attend  (T)
    indexer_compress_ratio 4     tokens averaged into one compressed key  (r)

    block_topk  = T // r = 512   compressed BLOCKS selected per query row
    index_width = T + r - 1 = 2051
        the width of the expanded token-index row: `block_topk*r` selected tokens plus the
        query's own still-incomplete trailing group (at most r-1 more tokens).

THE PAGE-ALIGNMENT REQUIREMENT, and why it is a hard error rather than a fallback.
A compressed key is addressed by `physical_kv_slot // r` — there is no compressed allocator and
no ownership state, which is the whole economy of this design (see `cache.QSAIndexCache`). That
identity holds iff the r members of a group land in r CONSECUTIVE physical slots, i.e. iff the KV
page size is a multiple of r. At `page_size = 1` the allocator hands out arbitrary slots and the
identity silently addresses another request's compressed key — right shapes, plausible text,
wrong model. So `require_page_size` raises; it does not fall back to a per-group allocator,
because that would be a second, untested addressing scheme reachable only in a configuration
nobody benchmarks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from minisgl.models import ModelConfig


@dataclass(frozen=True)
class QSAProfile:
    n_heads: int
    kv_heads: int
    head_dim: int
    budget: int
    compress_ratio: int

    @property
    def block_topk(self) -> int:
        """Compressed blocks selected per query row (`budget / compress_ratio`)."""
        return self.budget // self.compress_ratio

    @property
    def index_width(self) -> int:
        """Width of an expanded token-index row: `budget` selected + up to `r-1` trailing."""
        return self.budget + self.compress_ratio - 1

    def require_page_size(self, page_size: int) -> None:
        if page_size % self.compress_ratio != 0:
            raise ValueError(
                f"QSA needs a KV page_size that is a multiple of indexer_compress_ratio "
                f"({self.compress_ratio}); got page_size={page_size}. The compressed-key cache is "
                f"addressed by `physical_slot // {self.compress_ratio}` with no allocator of its "
                f"own, and that identity only holds when a group's {self.compress_ratio} members "
                f"are consecutive physical slots — which the paged allocator guarantees only when a "
                f"group cannot straddle a page. Serve this model with --page-size 16."
            )


def parse_qsa_profile(config: "ModelConfig") -> Optional[QSAProfile]:
    """Normalized QSA profile for `config`, or None when the model carries no indexer.

    Raises when a profile is PRESENT but malformed — a partially-specified indexer is a config
    bug and there is no defensible default for any of these five numbers.
    """
    names = (
        "indexer_n_heads",
        "indexer_kv_heads",
        "indexer_head_dim",
        "indexer_budget",
        "indexer_compress_ratio",
    )
    present = [n for n in names if getattr(config, n, None) is not None]
    if not present:
        return None
    missing = [n for n in names if getattr(config, n, None) is None]
    if missing:
        raise ValueError(f"QSA config is missing required fields: {missing}")
    v = {n: int(getattr(config, n)) for n in names}
    if any(x <= 0 for x in v.values()):
        raise ValueError(f"QSA config values must be positive: {v}")
    if v["indexer_kv_heads"] != 1:
        raise ValueError(
            f"the QSA MQA scoring op requires indexer_kv_heads=1, got {v['indexer_kv_heads']}"
        )
    if v["indexer_compress_ratio"] < 2:
        # A padding/dummy row carries logical length 1 and must never reach a compression
        # boundary; ratio >= 2 is what guarantees that.
        raise ValueError(
            f"QSA requires indexer_compress_ratio >= 2, got {v['indexer_compress_ratio']}"
        )
    if v["indexer_budget"] % v["indexer_compress_ratio"] != 0:
        raise ValueError(
            f"indexer_budget ({v['indexer_budget']}) must be divisible by "
            f"indexer_compress_ratio ({v['indexer_compress_ratio']})"
        )
    return QSAProfile(
        n_heads=v["indexer_n_heads"],
        kv_heads=v["indexer_kv_heads"],
        head_dim=v["indexer_head_dim"],
        budget=v["indexer_budget"],
        compress_ratio=v["indexer_compress_ratio"],
    )


__all__ = ["QSAProfile", "parse_qsa_profile"]
