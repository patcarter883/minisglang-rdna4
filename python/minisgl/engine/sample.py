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
    # min-p relative probability floor [bs] fp32, or None when no row asked (the overwhelmingly
    # common case — None keeps the fused HIP sampler eligible).
    min_p: torch.Tensor | None = None
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
    # Rows (batch indices, host list) whose request asked for per-token logprobs, and the largest N
    # asked. None/0 on the overwhelmingly common no-logprobs path — zero cost there. The capture
    # itself happens inside Sampler.sample (see _capture_logprobs) so it reads the SAME processed
    # logits (post-softcap, post-vocab-fence) that sampling reads.
    logprob_rows: list | None = None
    logprob_topn: int = 0


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


def _apply_top_k_top_p(probs: torch.Tensor, top_k: torch.Tensor,
                       top_p: torch.Tensor | None) -> torch.Tensor:
    """top-k (and the top-p that follows it) done in a [bs, kmax] slice instead of the full vocab.

    WHY. `_apply_top_k` + `_apply_top_p` run `torch.sort` over the WHOLE vocab twice per decode
    step — 248,320 elements per row on this family — to answer a question bounded by k, which the
    checkpoints here set to 20. Two full sorts, two full `zeros_like` allocations and two full
    scatters, to keep twenty values. `torch.topk(kmax)` answers exactly the same question over a
    slice 4 orders of magnitude smaller.

    WHY IT IS THE SAME ANSWER, not an approximation. top-k zeroes everything below rank k, so after
    it the only survivors are inside the top kmax = max(top_k) of the row. A descending sort of that
    masked row is the topk slice followed by zeros, and zeros add nothing to the top-p cumsum — so
    the nucleus decision on every surviving element is bit-identical. (`torch.topk(sorted=True)`
    returns descending order, matching `sort(descending=True)`.) Ties may be ORDERED differently
    between topk and sort, exactly as they already may be between two sort implementations; which of
    two equal-probability tokens survives is not defined by the reference either.

    Falls back to the pair above when `kmax >= vocab` (top-k disabled for some row), where a topk is
    a full sort anyway and the slice would buy nothing.

    COSTS ONE SYNC. `kmax` is a device scalar, so `.item()` blocks. That is a fixed ~tens of µs per
    step against two 248k-element sorts, and this is the reference path — the fused HIP kernel above
    is what avoids the sync entirely."""
    vocab = probs.shape[-1]
    kmax = int(top_k.max().item())
    if kmax >= vocab:
        probs = _apply_top_k(probs, top_k)
        return _apply_top_p(probs, top_p) if top_p is not None else probs
    vals, idx = torch.topk(probs, kmax, dim=-1)
    ranks = torch.arange(kmax, device=probs.device).unsqueeze(0)
    vals = vals.masked_fill(ranks >= top_k.unsqueeze(-1), 0.0)
    if top_p is not None:
        cumsum = vals.cumsum(dim=-1)
        vals = vals.masked_fill((cumsum - vals) > top_p.unsqueeze(-1), 0.0)
    return torch.zeros_like(probs).scatter_(-1, idx, vals)


def sample_impl(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: torch.Tensor | int | None,
    top_p: torch.Tensor | float | None,
    min_p: torch.Tensor | None = None,
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
        and min_p is None
    ):
        return _sampler_hip.sample(logits, temperatures, top_k, top_p)
    # torch port of the former flashinfer.sampling path (greedy goes through argmax in Sampler).
    probs = torch.softmax(logits / temperatures.unsqueeze(-1).clamp_min(1e-6), dim=-1)
    if min_p is not None:
        # min-p: drop tokens below min_p * max_prob for the row — a RELATIVE floor that adapts to
        # how peaked the distribution is (a guard top_p cannot express). Before top-k/top-p,
        # matching vLLM's processor order. Any row asking for it routes the batch here (the fused
        # HIP kernel has no min_p in its contract) — correctness over the fused win.
        floor = probs.max(dim=-1, keepdim=True).values * min_p.unsqueeze(-1)
        probs = probs.masked_fill(probs < floor, 0.0)
    if top_k is not None:
        # Bounded by k -> one topk over a [bs, kmax] slice instead of two full-vocab sorts.
        probs = _apply_top_k_top_p(probs, top_k, top_p)
    elif top_p is not None:
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
    # Gemma2-style final-logit soft cap (config `final_logit_softcapping`): logits are squashed to
    # (-cap, cap) via cap*tanh(l/cap) before any masking/sampling — an outlier-logit clamp both
    # reference engines apply at the head. None (every current checkpoint) = no-op.
    logit_softcap: float | None = None
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
            counts.append(self.penalty_counts(req))
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

    def penalty_counts(self, req) -> torch.Tensor:
        """This request's PERSISTENT [vocab] token-count buffer, created on first use.

        Seeded from whatever the request has already generated, so a penalty is correct even when the
        buffer is created mid-stream (preemption, or the first decode after prefill). Shared by the
        plain lane (`_penalty_plan`) and the spec verify lane, which must penalise against the same
        running count or the two lanes disagree about what has been said."""
        buf = self._pen_counts.get(req.uid)
        if buf is None:
            buf = torch.zeros(self.vocab_size, dtype=torch.float32, device=self.device)
            gen = req.generated_ids
            if gen.numel():
                buf.index_add_(
                    0, gen.to(self.device, torch.long),
                    torch.ones(gen.numel(), dtype=torch.float32, device=self.device))
            self._pen_counts[req.uid] = buf
        return buf

    def commit_penalty_tokens(self, uid: int, ids) -> None:
        """Fold COMMITTED tokens into a request's running count. The spec lane's entry point: it
        commits an accepted prefix plus one emitted token per step, not a single token."""
        buf = self._pen_counts.get(uid)
        if buf is None or not len(ids):
            return
        idx = torch.as_tensor(list(ids), dtype=torch.long, device=buf.device)
        buf.index_add_(0, idx, torch.ones(idx.numel(), dtype=buf.dtype, device=buf.device))

    def penalise_block(self, block: torch.Tensor, counts: torch.Tensor,
                       presence: float, frequency: float, drafts) -> None:
        """Apply OpenAI presence/frequency penalties to ONE request's [q, V] verify block, in place.

        The plain lane penalises one row against one count vector. A verify block is q = K+1 rows of
        the SAME request at consecutive positions, so row i must be penalised against the history plus
        whatever occupies positions before it — i.e. `drafts[:i]`. That prefix is known before the
        verify forward (the drafter proposes and the drafts are TP-broadcast first), so no rollback is
        needed: if verify rejects at n, rows > n are discarded and their assumed prefix never reaches
        the output, while row n assumed `drafts[:n]`, which IS the accepted prefix.

        Row i differs from row i-1 in exactly one column, so this is one broadcast subtract plus <=K
        single-column corrections rather than a dense [K+1, V] penalty tensor."""
        V = block.shape[-1]
        c = counts[:V]
        block -= (presence * (c > 0).to(block.dtype) + frequency * c).unsqueeze(0)
        seen = c.clone()
        for i, tok in enumerate(drafts):
            if i + 1 >= block.shape[0]:
                break                       # a padded staged row past the real block; never accepted
            t = int(tok)
            if t >= V:
                continue                    # fenced pad id; it can never be committed anyway
            before = presence * (seen[t] > 0).to(block.dtype) + frequency * seen[t]
            seen[t] += 1
            after = presence * (seen[t] > 0).to(block.dtype) + frequency * seen[t]
            block[i + 1:, t] -= (after - before)

    def _commit_penalty(self, tokens: torch.Tensor, args: BatchSamplingArgs) -> torch.Tensor:
        """Fold the tokens just drawn into the penalised rows' running counts — one index_add per
        penalised row, so per-step cost is O(1) in output length. No-op when nothing is penalised."""
        if args.pen_rows is None:
            return tokens
        picked = tokens[args.pen_rows].to(torch.long)
        ones = torch.ones(1, dtype=args.pen_counts.dtype, device=args.pen_counts.device)
        for i in range(picked.numel()):
            # Write the PERSISTENT per-uid buffer, not `args.pen_counts`. That field is a
            # `torch.stack` COPY built fresh by `_penalty_plan` every step, so an index_add_ into it
            # landed in a temporary that was dropped when the step ended: the running count never
            # advanced past its creation-time seed (empty for a request that starts fresh), every
            # `c` read at the penalty site was all-zero, and presence/frequency resolved to exactly
            # 0.0 for the whole life of the request. MEASURED before this line changed: with
            # presence_penalty=20.0, temperature 0 and top_k 1, the model emitted
            # "banana banana banana banana banana banana banana banana banana banana" —
            # byte-identical to presence_penalty=0.01, reproducibly. The penalty was inert.
            self._pen_counts[args.pen_uids[i]].index_add_(0, picked[i : i + 1], ones)
            args.pen_counts[i].index_add_(0, picked[i : i + 1], ones)  # keep the step copy coherent
        return tokens

    def free_penalty_state(self, uid: int) -> None:
        """Release a finished request's count buffer (idempotent)."""
        self._pen_counts.pop(uid, None)

    def prepare(self, batch: Batch) -> BatchSamplingArgs:
        params = [r.sampling_params for r in batch.reqs]
        pen = self._penalty_plan(batch)
        lp_rows = [i for i, p in enumerate(params) if getattr(p, "logprobs", 0) > 0]
        if lp_rows:
            pen = dict(pen, logprob_rows=lp_rows,
                       logprob_topn=max(params[i].logprobs for i in lp_rows))
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
        min_p = None
        min_ps = [max(getattr(p, "min_p", 0.0) or 0.0, 0.0) for p in params]
        if any(mp > 0.0 for mp in min_ps):
            min_p = make_device_tensor(min_ps, torch.float32, self.device)
        return BatchSamplingArgs(temperatures, top_k=top_k, top_p=top_p, min_p=min_p, **pen)

    def condition_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Head-side logit conditioning that EVERY lane turning a logit row into a committed token
        must apply — the plain sampler below, and every speculative verify forward alike.

        Three steps, in this order (matching what both reference engines do at the head):

        1. NaN/Inf scrub BEFORE anything reads the row. A transient NaN from any upstream kernel
           makes sampling undefined — SGLang's sanitizer documents the exact consequence ("can come
           back as out-of-vocab token ids", srt/utils/async_probe.py) and runs on every sample; vLLM
           leans on masking alone. Unconditional nan_to_num_ is one elementwise op (no host sync — an
           isnan().any() check would cost more than the scrub); NaN -> -1e30 removes the token from
           contention rather than crowning it argmax-of-garbage.
        2. Final-logit softcap, when the checkpoint asks for one (see logit_softcap).
        3. The padded-vocab fence: the untrained lm_head tail can never be sampled (see
           real_vocab_size). MEASURED on cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit, which logs at boot
           `masking padded vocab tail [248077, 248320) — 243 untrained ids fenced off`; those 243
           rows carry AWQ dequant noise, and an unfenced row is fully eligible for a greedy argmax
           or a top-k nucleus.

        This is ONE function, not three inline copies, because the copy already cost us: 2fba9ae
        landed exactly this block inline in Scheduler._spec_decode_step and left the three sibling
        verify forwards (_tidar_block_predict, _ddtree_tree_verify, _spec_decode_step_tidar_fused)
        reading raw logits — the fused-TiDAR one COMMITS its argmax directly. Anything that grows a
        fourth processor here reaches every lane at once.

        In-place where it can be (scrub + fence): safe even on a captured graph's static output
        buffer, because a replay overwrites it fully. The softcap has to allocate, so the returned
        tensor is NOT always the argument — callers must rebind, never assume in-place.
        """
        if os.environ.get("MINISGL_SANITIZE_LOGITS", "1") != "0":
            torch.nan_to_num_(logits, nan=-1e30, posinf=1e30, neginf=-1e30)
        if self.logit_softcap:
            logits = torch.tanh(logits.float() / self.logit_softcap) * self.logit_softcap
        if self.real_vocab_size is not None and self.real_vocab_size < logits.shape[-1]:
            logits[:, self.real_vocab_size:] = float("-inf")
        return logits

    @nvtx_annotate("Sampler")
    def sample(self, logits: torch.Tensor, args: BatchSamplingArgs) -> torch.Tensor:
        with torch.cuda.nvtx.range("Sampler"):
            # Scrub / softcap / padded-vocab fence. Shared verbatim with the spec verify lane — see
            # condition_logits for why each step is here and why it is not inlined.
            logits = self.condition_logits(logits)
            # Per-token logprob capture, for the rows that asked (SamplingParams.logprobs > 0).
            # Taken HERE — after the scrub / softcap / padded-vocab fence, before grammar masks,
            # EOS suppression and penalties — so it reports the raw model distribution rather than
            # the served one (the difference matters to an engine-numerics probe; probe requests
            # carry none of those masks anyway). The log_softmax stays on-device; the sampled
            # token's own logprob is gathered at return time (_finish_logprob_capture).
            _lp_full = None
            if args.logprob_rows:
                _lp_full = torch.log_softmax(
                    logits[torch.tensor(args.logprob_rows, device=logits.device)].float(), dim=-1)
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
                tokens = self._commit_penalty(torch.argmax(logits, dim=-1), args)
            else:
                tokens = self._commit_penalty(
                    sample_impl(logits.float(), args.temperatures, args.top_k, args.top_p,
                                args.min_p), args)
            if _lp_full is not None:
                self._finish_logprob_capture(_lp_full, tokens, args)
            return tokens

    def _finish_logprob_capture(self, lp_full: torch.Tensor, tokens: torch.Tensor,
                                args: BatchSamplingArgs) -> None:
        """Stash top-N (ids, logprobs) + the sampled token's logprob for the flagged rows. One small
        D2H copy per step, probe traffic only. The engine pops this right after sample() and rides
        it back on ForwardOutput.logprobs; keying is by BATCH ROW index (the scheduler maps rows to
        reqs by position)."""
        n = min(args.logprob_topn, lp_full.shape[-1])
        topv, topi = lp_full.topk(n, dim=-1)
        rows_t = torch.tensor(args.logprob_rows, device=tokens.device)
        tok_lp = lp_full.gather(-1, tokens[rows_t].long().unsqueeze(-1)).squeeze(-1)
        self._captured_logprobs = {
            int(r): (topi[j].tolist(), topv[j].tolist(), float(tok_lp[j]))
            for j, r in enumerate(args.logprob_rows)
        }

    def pop_captured_logprobs(self) -> dict | None:
        cap = getattr(self, "_captured_logprobs", None)
        self._captured_logprobs = None
        return cap
