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
    # OpenAI repetition controls, applied to the tokens THIS REQUEST HAS GENERATED (not the prompt —
    # penalising the caller's own vocabulary makes long prompts steer the answer away from their own
    # subject matter). logit -= presence*(count>0) + frequency*count, exactly the OpenAI formula.
    # 0.0 (both) is the neutral default and skips the whole path, so unpenalised traffic is unchanged.
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    stop: List[str] = field(default_factory=list)
    # INCLUSIVE stop strings: generation finishes at the first occurrence, but the matched string is
    # KEPT in the output (unlike `stop`, which trims it). For tool-call closers (`</tool_call>`) on
    # models that don't emit EOS after a call: stop the over-generation yet leave the block parseable.
    stop_keep: List[str] = field(default_factory=list)
    # Structured-output spec: None (free), "json" (any valid JSON object), or a JSON-schema string.
    # A constrained request is masked per-token by a grammar matcher (drafts propose unconstrained and
    # the grammar is enforced at the spec verify argmax).
    grammar: str | None = None
    # Reasoning + structured output: when a grammar is combined with an active thinking phase, the
    # model opens `<think>…</think>` reasoning BEFORE the answer, and masking JSON from token 0 would
    # suppress that reasoning (truncated / CoT-leaked output). This carries the reasoning parser's
    # close delimiter (e.g. "</think>"); the scheduler resolves it to a token SEQUENCE and does NOT
    # advance/mask the grammar matcher until that sequence is emitted — reasoning is free, the schema
    # is enforced only on the post-`</think>` answer. None => grammar applies from token 0
    # (non-reasoning models, thinking-off requests, or unconstrained reqs — all unchanged).
    # This is also the backstop's FORCE target, so it must stay one concrete, emittable literal.
    think_close_delim: str | None = None
    # An ADDITIONAL literal whose emission also ends the reasoning phase, but which is NEVER
    # force-emitted at the budget: the model's own ANSWER-TURN header. It exists for a template whose
    # generation prompt stops mid-header — Muse-Glimmer's ends `<|start|>assistant`, so a reply with
    # no reasoning at all begins ` to=user<|message|>` and never opens a reasoning turn. Without it
    # such a reply would never release the gate and the β backstop would splice a turn header into
    # the middle of a perfectly good answer. None for every family whose generation prompt ends at a
    # turn boundary (`<|im_start|>assistant\n`) — Qwen3, GLM, Laguna, Gemma-4, DeepSeek — i.e. a pure
    # no-op on the common path.
    think_answer_delim: str | None = None
    # The close delimiter with a WILDCARD recipient, as (prefix, suffix). A channel-routed template
    # names the recipient INSIDE the delimiter, so `think_close_delim` above is only the concrete
    # `to=user` form: when the model ends reasoning by routing to a TOOL it emits
    # `<|eom|><|start|>assistant to=<toolname><|message|>`, which the literal cannot match. These two
    # bracket it — the scheduler matches "prefix, then a bounded run of any tokens, then suffix".
    # Both None unless the checkpoint's template actually varies the delimiter by recipient.
    think_close_prefix: str | None = None
    think_close_suffix: str | None = None
    # Reasoning BUDGET (backstop for the gate above): a reasoning model often rambles in long/loose
    # prose and never emits a clean `</think>`, so the gate never opens and no JSON is produced. When
    # set (or via the scheduler's MINISGL_THINK_BUDGET default), after this many reasoning tokens the
    # scheduler FORCE-emits the think-close token, opening the gate so the schema engages. None => the
    # scheduler's env default applies; <=0 also falls back to the default. Only meaningful together with
    # `think_close_delim` (constrained + thinking); ignored otherwise.
    think_budget: int | None = None
    # CAM editable-memory (Option B): the explicit subject to read from the standing store. When set
    # AND the engine has CAM built, the scheduler computes this request's tap bank at prefill and injects
    # it at the L24 tap (seed-once). None -> a plain request (tap no-op). Rides UserMsg -> Req like grammar.
    mem_subject: str | None = None
    # CAM #100 pointer (multi-process model-share): with mem_subject set, mem_remember=object_token_ids
    # WRITES subject->object into the backend's engine.cam (the store lives in the scheduler process, so
    # the write must ride the request); None -> a READ/deliver request (the scheduler forces the exact
    # stored object tokens, then the base continues). Rides UserMsg -> Req like mem_subject.
    mem_remember: List[int] | None = None
    # CAM #100 control op riding a generate: "facts" | "forget" | "stats". The scheduler computes the
    # result from the backend engine.cam and FORCE-EMITS it (tokenised) as the reply text + EOS, so the
    # data-returning ops need no new message type. mem_subject supplies the subject for "forget".
    mem_op: str | None = None
    # Prefix-cache policy: whether this request's GENERATED output is inserted into the radix prefix
    # cache when it finishes. True (default) = normal caching. False = ephemeral output: the already-
    # cached shared PROMPT prefix is kept (evictable, for cross-request reuse) but the unique generated
    # tail is freed directly instead of being inserted. Used by RSA rollouts, whose per-rollout
    # generations are never reused — inserting them just pollutes the cache and inflates KV occupancy
    # (finished rollouts wouldn't "drop" on the KV graph). Rides UserMsg -> Req like the fields above.
    cache_output: bool = True
    # CAM write mode for a mem_remember write: "force" = explicit ingest (always writes, bypasses gates);
    # "auto"/None = ambient auto-write (subject to the store's freeze + no-clobber policy). Lets a curated
    # store be protected from conversational overwrite while explicit /cam/remember still curates.
    mem_write_mode: str | None = None
    # CAM namespace (#6 multi-tenant isolation): which per-tenant/session store this op reads/writes.
    # None -> "default" (single-store back-compat). Every CAM op (deliver/write/facts/forget/stats/retrieve)
    # is scoped to this namespace so one conversation cannot read or overwrite another's memory.
    mem_namespace: str | None = None
    # Per-request RNG seed. None (default) = draw from the process RNG, i.e. today's behaviour on
    # every path. It exists because BLOCK DIFFUSION has no greedy mode at all: its canvas starts as
    # uniform noise over the whole vocabulary and every denoising step draws a multinomial, so
    # `temperature 0 / top_p 1 / top_k 1` — the autoregressive path's reproducibility switch — buys
    # nothing there, and two identical requests to the same serve return different text. With a seed
    # a block's whole denoising trajectory is a deterministic function of (prompt, seed), which is
    # what lets a cold-vs-prefix-hit byte-identity gate exist for the canvas path at all (and lets a
    # caller reproduce a generation). Read by the diffusion loop; the AR sampler is unaffected.
    seed: int | None = None

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
        # Set by Scheduler._free_req_resources the first time this request's resources are released.
        # Overlap scheduling reaches that function from several independent paths (normal finish, the
        # spec and tidar decode steps, abort), and the de-dup used to be the one-step-deep
        # `finished_reqs` set that each path REPLACED rather than accumulated — so a request freed by
        # the spec path was forgotten by the next step and freed a second time, unlocking its radix
        # handle twice and driving node.ref_count negative. The flag lives on the REQUEST, so every
        # path is safe with no cross-step bookkeeping.
        self._resources_freed = False
        n = len(self.input_ids)
        self.device_len = n
        self.max_device_len = n + self.output_len
        assert 0 <= self.cached_len < self.device_len <= self.max_device_len
        # Host token buffer, preallocated to the max length this sequence can ever reach (prompt +
        # output_len committed tokens). `input_ids` is ALWAYS a length-prefix VIEW into this buffer,
        # so committing a token is an O(1) amortized in-place write instead of the O(seq_len) full
        # realloc+copy a growing `torch.cat` incurred (which made one generation O(seq_len^2)). Every
        # reader sees a byte-identical tensor (same values, dtype, cpu device); only the backing
        # store changed. Appends past the preallocated max (a speculative accept near the end) grow
        # the buffer by doubling — see _append_host_ids.
        # Prompt boundary, so the penalty path can count GENERATED tokens only.
        self._prompt_len = n
        self._ids_buf = torch.empty(self.max_device_len, dtype=self.input_ids.dtype)
        self._ids_buf[:n] = self.input_ids
        self._ids_len = n
        self.input_ids = self._ids_buf[:n]
        # CAM editable-memory (Option B): the per-request tap bank+conf, computed once at prefill from
        # the request's explicit subject and reused across decode. None for every non-memory request, so
        # the L24 tap stays a byte-exact no-op (see models/qwen3_5.py stage_cam/clear_cam).
        self.mem_bank: "torch.Tensor | None" = None
        self.mem_conf: "torch.Tensor | None" = None
        self._mem_seed: "int | None" = None       # the object's first (store-preferred) token
        self._mem_placed: bool = False            # seed-once: True once _mem_seed has been emitted

    @property
    def has_penalty(self) -> bool:
        sp = self.sampling_params
        return sp.presence_penalty != 0.0 or sp.frequency_penalty != 0.0

    @property
    def generated_ids(self) -> "torch.Tensor":
        """The tokens this request has produced so far (excludes the prompt)."""
        return self._ids_buf[self._prompt_len : self._ids_len]

    @property
    def remain_len(self) -> int:
        return self.max_device_len - self.device_len

    @property
    def extend_len(self) -> int:
        return self.device_len - self.cached_len

    def complete_one(self) -> None:
        self.cached_len = self.device_len
        self.device_len += 1

    def _append_host_ids(self, tokens: torch.Tensor) -> None:
        # Append one or more committed tokens into the preallocated host buffer in O(1) amortized
        # time. `tokens` is a 1-D cpu tensor whose dtype already matches input_ids (callers build it
        # that way — exactly as the old torch.cat required). Grows (doubling) only in the rare case a
        # path commits past max_device_len, keeping the result identical to a torch.cat either way.
        n = self._ids_len
        end = n + tokens.numel()
        if end > self._ids_buf.numel():
            new_cap = max(end, self._ids_buf.numel() * 2)
            grown = torch.empty(new_cap, dtype=self._ids_buf.dtype)
            grown[:n] = self._ids_buf[:n]
            self._ids_buf = grown
        self._ids_buf[n:end] = tokens
        self._ids_len = end
        self.input_ids = self._ids_buf[:end]

    def append_host(self, next_token: torch.Tensor) -> None:
        self._append_host_ids(next_token)

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
    # Block-diffusion CANVAS batch (DiffusionGemma). Like a verify batch it is phase="decode" with
    # extend_len = canvas_length query tokens per request, but it differs in two ways that no other
    # path in the engine has:
    #   * attention is BIDIRECTIONAL over [encoder KV | canvas KV] — the backend keys `causal=0`
    #     off this flag, on both layer geometries;
    #   * the canvas K/V is SCRATCH. Every denoising step overwrites the same slots, so
    #     cached_len/device_len do NOT advance across the <=48 steps of a block, nothing is
    #     sampled per step, and `complete_one` is never called.
    # False for every autoregressive batch, so no existing path changes shape.
    canvas: bool = field(default=False, init=False)

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
    # CAM editable-memory (Option B) — the built CAMMemory (store+tap+router), set by the Engine ONLY
    # when MINISGL_CAM=1 + a checkpoint is given. The model's L24 tap reaches it via stage_cam; stays
    # None for every non-memory run, where the tap is a byte-exact no-op.
    cam_state: "object | None" = field(default=None, init=False)
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
