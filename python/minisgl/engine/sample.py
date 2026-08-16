from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List

import os

import torch
from minisgl.utils import nvtx_annotate

from . import _sampler_hip

if TYPE_CHECKING:
    from minisgl.core import Batch


@dataclass
class BatchSamplingArgs:
    temperatures: torch.Tensor | None
    top_k: torch.Tensor | None = None
    top_p: torch.Tensor | None = None
    # Structured output: packed xgrammar token bitmask [bs, ceil(vocab/32)] (constrained rows carry
    # the grammar's allowed set; unconstrained rows are all-ones). Applied to logits before sampling.
    grammar_bitmask: torch.Tensor | None = None
    # Reasoning gate: bool [bs] marking rows still inside a `<think>…</think>` span. Their end-of-turn
    # (EOS) logits are set to -inf before sampling so a thinking model cannot terminate the turn
    # mid-reasoning (→ blank/truncated answer). Bounded by the reasoning-budget backstop, which
    # force-emits </think> and clears the gate, after which EOS is allowed again. None = no suppression.
    eos_suppress: torch.Tensor | None = None
    # Repetition penalties (OpenAI presence/frequency). Only the rows that asked for them are
    # carried: `pen_rows` indexes into the batch, `pen_counts` is a [n_pen, vocab] per-token count of
    # what those rows have generated, and the two coefficient vectors are [n_pen]. None when no row in
    # the batch has a non-zero penalty, which is the overwhelmingly common case and costs nothing.
    pen_rows: torch.Tensor | None = None
    pen_counts: torch.Tensor | None = None
    pen_presence: torch.Tensor | None = None
    pen_frequency: torch.Tensor | None = None
    pen_uids: list | None = None


def make_device_tensor(data: List, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.tensor(data, dtype=dtype, pin_memory=True).to(device, non_blocking=True)


def _apply_top_k(probs: torch.Tensor, top_k: torch.Tensor) -> torch.Tensor:
    # per-row top-k mask (top_k: [bs] int); keep the k highest probs, zero the rest.
    sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
    ranks = torch.arange(probs.shape[-1], device=probs.device).unsqueeze(0)
    sorted_probs = sorted_probs.masked_fill(ranks >= top_k.unsqueeze(-1), 0.0)
    return torch.zeros_like(probs).scatter_(-1, sorted_idx, sorted_probs)


def _apply_top_p(probs: torch.Tensor, top_p: torch.Tensor) -> torch.Tensor:
    # per-row nucleus mask (top_p: [bs] float); keep the smallest prefix whose mass > top_p.
    sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
    cumsum = sorted_probs.cumsum(dim=-1)
    sorted_probs = sorted_probs.masked_fill((cumsum - sorted_probs) > top_p.unsqueeze(-1), 0.0)
    return torch.zeros_like(probs).scatter_(-1, sorted_idx, sorted_probs)


def sample_impl(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: torch.Tensor | int | None,
    top_p: torch.Tensor | float | None,
) -> torch.Tensor:
    # Fused native HIP sampler (temperature+softmax+top-k+top-p+multinomial in one kernel, no sort)
    # when available; otherwise the torch reference below. Only tensor top_k/top_p route to the op
    # (the int/float scalar forms are unused by the engine's Sampler.prepare).
    if (
        _sampler_hip.available()
        and logits.is_cuda
        and logits.dtype == torch.float32
        and not isinstance(top_k, int)
        and not isinstance(top_p, float)
    ):
        return _sampler_hip.sample(logits, temperatures, top_k, top_p)
    # torch port of the former flashinfer.sampling path (greedy goes through argmax in Sampler).
    probs = torch.softmax(logits / temperatures.unsqueeze(-1).clamp_min(1e-6), dim=-1)
    if top_k is not None:
        probs = _apply_top_k(probs, top_k)
    if top_p is not None:
        probs = _apply_top_p(probs, top_p)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return torch.multinomial(probs, num_samples=1).squeeze(-1).to(torch.int32)


@dataclass
class Sampler:
    device: torch.device
    vocab_size: int
    # The TOKENIZER's vocab length, when it is smaller than the model's (padded) vocab_size. The
    # trailing `vocab_size - real_vocab_size` lm_head rows are UNTRAINED padding (Qwen3.5-family:
    # 248320 config vs 248077 tokenizer = 243 pad ids); their logits are checkpoint noise — or live
    # dequant artifacts on the fp8 lm_head path — and are fully eligible for top-k selection unless
    # masked. Both reference engines make them unreachable (vLLM slices logits to org_vocab_size,
    # logits_processor.py:103; SGLang slices to vocab_size, logits_processor.py:865). None/equal ->
    # no mask (a tokenizer that could not be loaded must not silently disable sampling).
    real_vocab_size: int | None = None
    # uid -> [vocab] float32 count of tokens that request has generated. Created lazily for penalised
    # requests only (~1 MB each at a 248k vocab) and updated incrementally by one index_add per step,
    # so the cost does not grow with output length. Released by `free_penalty_state` on finish.
    _pen_counts: dict = field(default_factory=dict)
    # End-of-generation token ids ([E] int on device), set once by the scheduler after it resolves the
    # model's full EOS set. Used to suppress EOS for reasoning-phase rows (see BatchSamplingArgs.eos_suppress).
    eos_token_ids: torch.Tensor | None = None

    def _penalty_plan(self, batch: Batch) -> dict:
        """Per-row presence/frequency state for the rows that asked for it. Empty dict when none did."""
        rows = [i for i, r in enumerate(batch.reqs) if r.has_penalty]
        if not rows:
            return {}
        counts, uids = [], []
        for i in rows:
            req = batch.reqs[i]
            buf = self._pen_counts.get(req.uid)
            if buf is None:
                buf = torch.zeros(self.vocab_size, dtype=torch.float32, device=self.device)
                # Seed from whatever this request already generated, so a penalty is correct even if
                # the buffer is created mid-stream (preemption, or a first decode after prefill).
                gen = req.generated_ids
                if gen.numel():
                    buf.index_add_(
                        0, gen.to(self.device, torch.long),
                        torch.ones(gen.numel(), dtype=torch.float32, device=self.device))
                self._pen_counts[req.uid] = buf
            counts.append(buf)
            uids.append(req.uid)
        sp = [batch.reqs[i].sampling_params for i in rows]
        return dict(
            pen_rows=torch.tensor(rows, dtype=torch.long, device=self.device),
            pen_counts=torch.stack(counts),
            # Plain tensors, not make_device_tensor: that pins host memory (CUDA-only) for an async
            # H2D copy, which is pointless for an [n_pen] vector and makes this path untestable off-GPU.
            pen_presence=torch.tensor([p.presence_penalty for p in sp],
                                      dtype=torch.float32, device=self.device),
            pen_frequency=torch.tensor([p.frequency_penalty for p in sp],
                                       dtype=torch.float32, device=self.device),
            pen_uids=uids,
        )

    def _commit_penalty(self, tokens: torch.Tensor, args: BatchSamplingArgs) -> torch.Tensor:
        """Fold the tokens just drawn into the penalised rows' running counts — one index_add per
        penalised row, so per-step cost is O(1) in output length. No-op when nothing is penalised."""
        if args.pen_rows is None:
            return tokens
        picked = tokens[args.pen_rows].to(torch.long)
        ones = torch.ones(1, dtype=args.pen_counts.dtype, device=args.pen_counts.device)
        for i in range(picked.numel()):
            args.pen_counts[i].index_add_(0, picked[i : i + 1], ones)
        return tokens

    def free_penalty_state(self, uid: int) -> None:
        """Release a finished request's count buffer (idempotent)."""
        self._pen_counts.pop(uid, None)

    def prepare(self, batch: Batch) -> BatchSamplingArgs:
        params = [r.sampling_params for r in batch.reqs]
        pen = self._penalty_plan(batch)
        if all(p.is_greedy for p in params):
            # Greedy still needs penalties applied — they change which token is the argmax, which is
            # the entire point of asking for them at temperature 0.
            return BatchSamplingArgs(temperatures=None, **pen)

        MIN_P = MIN_T = 1e-6
        ts = [max(0.0 if p.is_greedy else p.temperature, MIN_T) for p in params]
        top_ks = [p.top_k if p.top_k >= 1 else self.vocab_size for p in params]
        top_ps = [min(max(p.top_p, MIN_P), 1.0) for p in params]
        temperatures = make_device_tensor(ts, torch.float32, self.device)
        top_k, top_p = None, None
        if any(k != self.vocab_size for k in top_ks):
            top_k = make_device_tensor(top_ks, torch.int32, self.device)
        if any(p < 1.0 for p in top_ps):
            top_p = make_device_tensor(top_ps, torch.float32, self.device)
        return BatchSamplingArgs(temperatures, top_k=top_k, top_p=top_p, **pen)

    @nvtx_annotate("Sampler")
    def sample(self, logits: torch.Tensor, args: BatchSamplingArgs) -> torch.Tensor:
        with torch.cuda.nvtx.range("Sampler"):
            # NaN/Inf scrub BEFORE anything reads the row. A transient NaN from any upstream kernel
            # makes sampling undefined — SGLang's sanitizer documents the exact consequence ("can
            # come back as out-of-vocab token ids", srt/utils/async_probe.py) and runs on every
            # sample; vLLM leans on masking alone. Unconditional nan_to_num_ is one elementwise op
            # (no host sync — an isnan().any() check would cost more than the scrub); NaN -> -1e30
            # removes the token from contention rather than crowning it argmax-of-garbage.
            # In-place is safe even on a captured graph's output buffer: replay overwrites it fully.
            if os.environ.get("MINISGL_SANITIZE_LOGITS", "1") != "0":
                torch.nan_to_num_(logits, nan=-1e30, posinf=1e30, neginf=-1e30)
            # Padded-vocab fence: the untrained tail can never be sampled (see real_vocab_size).
            if self.real_vocab_size is not None and self.real_vocab_size < logits.shape[-1]:
                logits[:, self.real_vocab_size:] = float("-inf")
            if args.grammar_bitmask is not None:  # structured output: mask disallowed tokens to -inf
                from .grammar import apply_token_bitmask

                logits = apply_token_bitmask(logits.float(), args.grammar_bitmask)
            if args.eos_suppress is not None and self.eos_token_ids is not None:
                # Reasoning phase: forbid end-of-turn tokens for rows still inside <think> so the model
                # can't stop mid-reasoning. apply_token_bitmask above already returned a fresh fp32
                # tensor; otherwise copy so we never mutate the forward's (possibly captured) buffer.
                rows = args.eos_suppress.nonzero(as_tuple=True)[0]
                if rows.numel():
                    logits = logits.float() if args.grammar_bitmask is not None else logits.float().clone()
                    logits[rows.unsqueeze(1), self.eos_token_ids.to(logits.device).unsqueeze(0)] = float("-inf")
            if args.pen_rows is not None:
                # OpenAI penalties: logit -= presence*(count>0) + frequency*count, on the rows that
                # asked for them. Clone before writing — the forward's logits can be the captured
                # graph's static output buffer, and mutating it in place would corrupt the next replay.
                logits = logits.float().clone()
                c = args.pen_counts[:, : logits.shape[-1]]
                logits[args.pen_rows] -= (
                    args.pen_presence.unsqueeze(1) * (c > 0).to(c.dtype)
                    + args.pen_frequency.unsqueeze(1) * c
                )
            if args.temperatures is None:  # greedy sampling
                # Penalties apply to greedy too: they change which token is the argmax, which is the
                # entire point of asking for them at temperature 0.
                return self._commit_penalty(torch.argmax(logits, dim=-1), args)
            return self._commit_penalty(
                sample_impl(logits.float(), args.temperatures, args.top_k, args.top_p), args)
