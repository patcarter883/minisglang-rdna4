from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Literal

import torch

if TYPE_CHECKING:
    from minisgl.attention import BaseAttnBackend, BaseAttnMetadata
    from minisgl.distributed import EPCommunicator as EPContext
    from minisgl.kvcache import BaseCacheHandle, BaseKVCachePool
    from minisgl.kvcache.cca_state import CCAStateCache
    from minisgl.kvcache.gdn_state import GDNStateCache
    from minisgl.moe import BaseMoeBackend


@dataclass
class SamplingParams:
    temperature: float = 0.0
    top_k: int = -1
    top_p: float = 1.0
    ignore_eos: bool = False
    max_tokens: int = 1024
    # Stop strings: generation finishes (and the output is truncated) at the first occurrence of any
    # of these in the decoded text. Matched on the detokenized string, scheduler-agnostic.
    stop: List[str] = field(default_factory=list)
    # Structured-output spec: None (free), "json" (any valid JSON object), or a JSON-schema string.
    # A constrained request is masked per-token by a grammar matcher (drafts propose unconstrained and
    # the grammar is enforced at the spec verify argmax).
    grammar: str | None = None
    # Reasoning + structured output: when a grammar is combined with an active thinking phase, the
    # model opens `<think>…</think>` reasoning BEFORE the answer, and masking JSON from token 0 would
    # suppress that reasoning (truncated / CoT-leaked output). This carries the reasoning parser's
    # close delimiter (e.g. "</think>"); the scheduler resolves it to a token id and does NOT
    # advance/mask the grammar matcher until that token is emitted — reasoning is free, the schema is
    # enforced only on the post-`</think>` answer. None => grammar applies from token 0 (non-reasoning
    # models, thinking-off requests, or unconstrained reqs — all unchanged).
    think_close_delim: str | None = None
    # Reasoning BUDGET (backstop for the gate above): a reasoning model often rambles in long/loose
    # prose and never emits a clean `</think>`, so the gate never opens and no JSON is produced. When
    # set (or via the scheduler's MINISGL_THINK_BUDGET default), after this many reasoning tokens the
    # scheduler FORCE-emits the think-close token, opening the gate so the schema engages. None => the
    # scheduler's env default applies; <=0 also falls back to the default. Only meaningful together with
    # `think_close_delim` (constrained + thinking); ignored otherwise.
    think_budget: int | None = None

    @property
    def is_greedy(self) -> bool:
        return (self.temperature <= 0.0 or self.top_k == 1) and self.top_p == 1.0

    @property
    def is_constrained(self) -> bool:
        return self.grammar is not None


@dataclass(eq=False)
class Req:
    input_ids: torch.Tensor  # cpu tensor
    table_idx: int
    cached_len: int
    output_len: int
    uid: int
    sampling_params: SamplingParams
    cache_handle: BaseCacheHandle

    def __post_init__(self) -> None:
        assert self.input_ids.is_cpu
        self.device_len = len(self.input_ids)
        self.max_device_len = len(self.input_ids) + self.output_len
        assert 0 <= self.cached_len < self.device_len <= self.max_device_len

    @property
    def remain_len(self) -> int:
        return self.max_device_len - self.device_len

    @property
    def extend_len(self) -> int:
        return self.device_len - self.cached_len

    def complete_one(self) -> None:
        self.cached_len = self.device_len
        self.device_len += 1

    def append_host(self, next_token: torch.Tensor) -> None:
        self.input_ids = torch.cat([self.input_ids, next_token])

    @property
    def can_decode(self) -> bool:
        return self.remain_len > 0

    def __repr__(self) -> str:
        return (
            f"{type(self)}(table_idx={self.table_idx}, "
            f"cached_len={self.cached_len}, device_len={self.device_len}, "
            f"max_device_len={self.max_device_len})"
        )


@dataclass
class Batch:
    reqs: List[Req]
    phase: Literal["prefill", "decode"]
    # these fields should be set by scheduler
    input_ids: torch.Tensor = field(init=False)
    positions: torch.Tensor = field(init=False)
    out_loc: torch.Tensor = field(init=False)
    padded_reqs: List[Req] = field(init=False)
    # this field should be set by attention backend
    attn_metadata: BaseAttnMetadata = field(init=False)
    # GDN (linear-attention) per-batch metadata — set by the scheduler ONLY for GDN-hybrid
    # models (None otherwise, so the dense path is unaffected). See gdn/metadata.py.
    gdn_metadata: object | None = field(default=None, init=False)
    # ZAYA CCA per-batch metadata — set by the scheduler ONLY for CCA-hybrid (Zaya) models
    # (None otherwise, so dense/GDN paths are unaffected). See cca/metadata.py.
    cca_metadata: object | None = field(default=None, init=False)
    # Speculative-decode VERIFY batch: phase is "decode" (so the LM head returns all-token logits,
    # no last-token reduction) BUT each req carries extend_len = K+1 query tokens. Multi-token paths
    # that key on `is_prefill` (GDN layer dispatch + gdn_metadata) must treat a verify batch like a
    # prefill; attention/MLA key on extend_len/max_seqlen_q and need no flag. False for every
    # normal batch. See scheduler._spec_decode_step and SPEC_DECODE.md.
    spec_verify: bool = field(default=False, init=False)

    @property
    def is_prefill(self) -> bool:
        return self.phase == "prefill"

    @property
    def is_decode(self) -> bool:
        return self.phase == "decode"

    @property
    def size(self) -> int:
        return len(self.reqs)

    @property
    def padded_size(self) -> int:
        return len(self.padded_reqs)


@dataclass
class Context:
    page_size: int
    # NOTE: this table always treat page_size = 1
    page_table: torch.Tensor = field(init=False)
    attn_backend: BaseAttnBackend = field(init=False)
    moe_backend: BaseMoeBackend = field(init=False)
    kv_cache: BaseKVCachePool = field(init=False)
    # Window-bounded SWA KV pool (Laguna) — set by the Engine ONLY for SWA-hybrid models. Holds the
    # sliding layers' paged KV as a per-sequence ring of `sliding_window` slots (indexed table_idx*W
    # + pos%W), so long-context serving does not allocate a full-context slot per sliding layer. The
    # attention backend reaches it via get_global_ctx().kv_cache's sibling; None for every non-SWA model.
    swa_kv_cache: "BaseKVCachePool | None" = field(default=None, init=False)
    # GDN recurrent-state cache — set by the Engine ONLY for GDN-hybrid models, so a layer
    # forward reaches it via `get_global_ctx().gdn_state`. Stays None for every dense model.
    gdn_state: "GDNStateCache | None" = field(default=None, init=False)
    # ZAYA CCA recurrent-state cache (conv_states + prev_hs) — set by the Engine ONLY for CCA-hybrid
    # models, reached via `get_global_ctx().cca_state`. Stays None for every non-Zaya model.
    cca_state: "CCAStateCache | None" = field(default=None, init=False)
    # Expert-parallel (EP) state — set by the Engine ONLY when --enable-ep (dp_size>1). MoELayer.forward
    # reaches it via get_global_ctx().ep so the EP dispatch/combine (all_gather token rows over the EP
    # group, masked local-expert compute, all_reduce(SUM)) runs INSIDE the captured decode graph. None
    # for every non-EP run (single replica or DP-only), where MoE stays purely replica-local.
    ep: "EPContext | None" = field(default=None, init=False)
    _batch: Batch | None = field(default=None, init=False)

    @property
    def batch(self) -> Batch:
        assert self._batch is not None, "No active batch in context"
        return self._batch

    @contextmanager
    def forward_batch(self, batch: Batch):
        assert self._batch is None, "Nested forward_batch is not allowed"
        try:
            self._batch = batch
            yield
        finally:
            self._batch = None


_GLOBAL_CTX: Context | None = None


def set_global_ctx(ctx: Context):
    global _GLOBAL_CTX
    assert _GLOBAL_CTX is None, "Global context is already set"
    _GLOBAL_CTX = ctx


def get_global_ctx() -> Context:
    assert _GLOBAL_CTX is not None, "Global context is not set"
    return _GLOBAL_CTX
