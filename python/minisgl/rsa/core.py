"""Markovian RSA orchestrator: backend client, tail truncation, round loop.

The Markovian property is enforced in :func:`run_markovian_rsa`: round ``t``
aggregates ONLY candidates drawn from round ``t-1``'s population. No earlier
round (and no accumulated cross-round history) is ever sampled. The original
query string is re-stated as the problem statement each round, but the
aggregation *context* is purely the previous round's traces -- a Markov chain
over rounds, matching the ZAYA1-8B report's Markovian RSA.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import time
from dataclasses import dataclass, field
from typing import List, Optional

# httpx + openai are needed ONLY by the HTTP BackendClient (the standalone shim path). The
# in-process client (inproc.py) and the orchestrator below depend on neither, so import them
# optionally — an in-engine serve that never instantiates the HTTP client must not require them.
try:
    import httpx
    import openai
except ImportError:  # pragma: no cover - only the HTTP shim path needs these
    httpx = None  # type: ignore[assignment]
    openai = None  # type: ignore[assignment]

from . import extract, prompts
from .config import RSAParams

# Per-request RSA timing breakdown (default ON; set MINISGL_RSA_TIMING=0 to silence). Distinguishes
# the structural cost (T sequential generation rounds) from the recoverable cost (straggler idle
# inside each round's barrier, tail re-tokenization, and inter-round prompt-build on the event loop).
_RSA_TIMING = os.environ.get("MINISGL_RSA_TIMING", "1") != "0"


def _pct(sorted_vals: List[float], q: float) -> float:
    """q-percentile of an already-sorted list (nearest-rank), 0.0 if empty."""
    if not sorted_vals:
        return 0.0
    i = min(len(sorted_vals) - 1, int(q * (len(sorted_vals) - 1) + 0.5))
    return sorted_vals[i]

from minisgl.utils import init_logger

# Use minisgl's init_logger (StreamHandler->stdout at INFO, propagate=False) like every other module.
# A plain logging.getLogger("minisgl.rsa") has NO handler and propagates to the root logger, whose
# default level is WARNING -> every RSA info() line (round/selection AND [rsa-timing]) was dropped.
logger = init_logger(__name__)


def advance_to_boundary(tail: str, max_skip_fraction: float = 0.1) -> str:
    """Advance a sliced tail's start to the next semantic boundary.

    A fixed token cut can land mid-sentence; look for a paragraph (then line)
    break within the leading fraction of the tail and start there instead.
    """
    window = max(int(len(tail) * max_skip_fraction), 1)
    cut = tail.find("\n\n", 0, window)
    if cut != -1:
        return tail[cut + 2 :].lstrip("\n")
    cut = tail.find("\n", 0, window)
    if cut != -1:
        return tail[cut + 1 :]
    return tail


class RSAError(Exception):
    """A whole RSA round failed; maps to HTTP 502 in the shim server."""


@dataclass
class Candidate:
    text: str  # answer-bearing text used as the aggregation input
    finish_reason: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class UsageTotals:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    n_requests: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def add(self, candidate: Candidate) -> None:
        self.prompt_tokens += candidate.prompt_tokens
        self.completion_tokens += candidate.completion_tokens
        self.n_requests += 1


@dataclass
class RSAResult:
    final_text: str
    population: List[Candidate]
    rounds: List[List[Candidate]]
    usage: UsageTotals
    selection_method: str  # "majority_vote" | "final_aggregation" | "sample"
    vote_detail: Optional[dict] = None


class BackendClient:
    """Thin async client for the minisglang OpenAI-compatible server.

    minisglang's frontend exposes ``/v1/chat/completions`` (it does not
    implement the native ``n`` fan-in parameter, ``/tokenize``, or a usage
    report), so every rollout is a separate chat request and tails fall back
    to a local HF tokenizer or a character approximation.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str = "EMPTY",
        timeout: float = 1800.0,
        tokenizer: Optional[str] = None,
    ):
        assert openai is not None and httpx is not None, (
            "the HTTP BackendClient (standalone RSA shim) needs `openai` and `httpx` installed; "
            "the in-engine RSA path uses InProcessBackendClient and needs neither."
        )
        self.base_url = base_url
        root = base_url.rstrip("/").removesuffix("/v1")
        self.openai = openai.AsyncOpenAI(
            base_url=base_url, api_key=api_key, timeout=timeout, max_retries=0
        )
        self.http = httpx.AsyncClient(base_url=root, timeout=60.0)
        self._tokenizer_name = tokenizer
        self._tokenizer = None  # None = not tried, False = unavailable
        self._tokenizer_lock = asyncio.Lock()

    async def close(self) -> None:
        await self.openai.close()
        await self.http.aclose()

    async def default_model(self) -> str:
        models = await self.openai.models.list()
        return models.data[0].id

    async def complete(
        self,
        messages: List[dict],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        top_p: float = 1.0,
        top_k: int = -1,
        ignore_eos: bool = False,
        stop: Optional[List[str]] = None,
        max_retries: int = 1,
        grammar: Optional[str] = None,
        tools: Optional[List[dict]] = None,
        chat_template_kwargs: Optional[dict] = None,
        think_close_delim: Optional[str] = None,
        think_budget: Optional[int] = None,
    ) -> Optional[Candidate]:
        """One chat completion; returns None on permanent failure. The structured knobs
        (grammar/tools/chat_template_kwargs/think_*) mirror the in-process client; over HTTP they ride
        the same minisgl extra_body extension fields the api_server reads (the in-engine path is the
        primary one — this keeps the legacy shim from silently dropping structured requests)."""
        attempt = 0
        while True:
            try:
                extra = {"top_k": top_k, "ignore_eos": ignore_eos}
                # minisgl request extensions the api_server understands off the body.
                if grammar is not None:
                    extra["grammar"] = grammar
                if chat_template_kwargs:
                    extra["chat_template_kwargs"] = chat_template_kwargs
                if think_close_delim is not None:
                    extra["think_close_delim"] = think_close_delim
                if think_budget is not None:
                    extra["reasoning_max_tokens"] = think_budget
                resp = await self.openai.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    stop=list(stop) if stop else None,
                    tools=tools or None,
                    # top_k / ignore_eos + the structured extensions are minisgl fields, not standard
                    # OpenAI, so they ride in extra_body (the api_server reads them off the body).
                    extra_body=extra,
                )
                msg = resp.choices[0].message
                content = msg.content or ""
                usage = resp.usage
                return Candidate(
                    text=content,
                    finish_reason=resp.choices[0].finish_reason,
                    prompt_tokens=usage.prompt_tokens if usage else 0,
                    completion_tokens=usage.completion_tokens if usage else 0,
                )
            except (openai.APIConnectionError, openai.APITimeoutError) as e:
                if attempt < max_retries:
                    attempt += 1
                    logger.warning("rollout transport error, retry %d: %s", attempt, e)
                    continue
                logger.error("rollout failed after %d retries: %s", attempt, e)
                return None
            except openai.APIStatusError as e:
                logger.error("rollout failed with status %s: %s", e.status_code, e)
                return None

    async def _get_tokenizer(self):
        """Lazily load a local HF tokenizer; False when unavailable."""
        if self._tokenizer is not None:
            return self._tokenizer
        async with self._tokenizer_lock:
            if self._tokenizer is not None:
                return self._tokenizer
            name = self._tokenizer_name
            try:
                if name is None:
                    r = await self.http.get("/v1/models")
                    r.raise_for_status()
                    entry = r.json()["data"][0]
                    name = entry.get("root") or entry["id"]

                def load():
                    from transformers import AutoTokenizer

                    return AutoTokenizer.from_pretrained(name)

                self._tokenizer = await asyncio.to_thread(load)
                logger.info("loaded local tokenizer %r", name)
            except Exception as e:
                self._tokenizer = False
                logger.warning(
                    "local tokenizer unavailable (%s); tails use char approximation",
                    e,
                )
        return self._tokenizer

    async def tail(self, text: str, tail_tokens: int) -> str:
        """Truncate *text* to its final *tail_tokens* tokens.

        Token-exact via a local HF tokenizer when available, else a character
        approximation. Both advance the cut to a semantic boundary so
        aggregation prompts never start mid-thought.
        """
        if tail_tokens <= 0:
            return text
        # NOTE: no `len(text) <= tail_tokens` early-out — that compares characters to a TOKEN
        # budget. Token count can exceed char count (byte-level BPE on CJK/code: one code point ->
        # several tokens), so a short-looking string can still blow the tail budget. Let the
        # tokenizer (or the char approximation below) make the decision.

        cut = None
        tokenizer = await self._get_tokenizer()
        if tokenizer:

            def token_slice():
                ids = tokenizer.encode(text, add_special_tokens=False)
                if len(ids) <= tail_tokens:
                    return None
                return tokenizer.decode(ids[-tail_tokens:], skip_special_tokens=False)

            cut = await asyncio.to_thread(token_slice)
            if cut is None:
                return text
        if cut is None:
            approx = tail_tokens * 4  # ~4 chars/token
            if len(text) <= approx:
                return text
            cut = text[-approx:]
        return prompts.TRUNCATION_MARKER + advance_to_boundary(cut)


async def _run_round(
    client: BackendClient,
    message_sets: List[List[dict]],
    *,
    model: str,
    params: RSAParams,
    max_tokens: int,
    semaphore: asyncio.Semaphore,
    usage: UsageTotals,
    round_idx: int,
    chat_template_kwargs: Optional[dict] = None,
) -> List[Candidate]:
    """Fan out one rollout per message set, bounded by *semaphore*."""

    async def one(messages: List[dict]) -> tuple:
        async with semaphore:
            c = await client.complete(
                messages,
                model=model,
                temperature=params.temperature,
                max_tokens=max_tokens,
                top_p=params.top_p,
                top_k=params.top_k,
                ignore_eos=params.ignore_eos,
                stop=params.stop,
                max_retries=params.max_retries,
                # thinking-mode is threaded to every rollout for consistency; the exploration rounds
                # stay grammar/tools-FREE (structured output is applied only to the final answer).
                chat_template_kwargs=chat_template_kwargs,
            )
        return c, time.monotonic()  # (candidate, absolute finish time) for straggler analysis

    start = time.monotonic()
    results = await asyncio.gather(*(one(m) for m in message_sets))
    population = [c for (c, _) in results if c is not None]
    for c in population:
        usage.add(c)
    if not population:
        raise RSAError(f"round {round_idx}: all {len(message_sets)} rollouts failed")
    round_wall = time.monotonic() - start
    if _RSA_TIMING:
        # per-rollout completion offsets from round start: the barrier can't release until the LAST
        # one finishes, so the p50->p100 gap ("straggler tail") is GPU-underutilized time (most
        # rollouts done, a few long ones running) -- the recoverable idle inside the structural barrier.
        done = sorted(fin - start for (c, fin) in results if c is not None)
        logger.info(
            "[rsa-timing] round %d: %d/%d cand, %.2fs wall | rollout-done p0/p50/p100=%.2f/%.2f/%.2fs"
            " straggler-tail=%.2fs | %d compl tok (max %d)",
            round_idx, len(population), len(message_sets), round_wall,
            done[0], _pct(done, 0.5), done[-1], done[-1] - _pct(done, 0.5),
            sum(c.completion_tokens for c in population),
            max(c.completion_tokens for c in population),
        )
    else:
        logger.info(
            "round %d: %d/%d candidates, %d prompt + %d completion tokens, %.1fs",
            round_idx, len(population), len(message_sets),
            sum(c.prompt_tokens for c in population),
            sum(c.completion_tokens for c in population), round_wall,
        )
    return population


async def _tails_for(
    client: BackendClient,
    population: List[Candidate],
    params: RSAParams,
) -> dict:
    """Compute each candidate's tail once per round (keyed by identity)."""
    tails = await asyncio.gather(
        *(client.tail(c.text, params.tail_tokens) for c in population)
    )
    return {id(c): t for c, t in zip(population, tails)}


async def run_markovian_rsa(
    client: BackendClient,
    params: RSAParams,
    messages: List[dict],
    model: str,
    rng: Optional[random.Random] = None,
    *,
    chat_template_kwargs: Optional[dict] = None,
    grammar: Optional[str] = None,
    tools: Optional[List[dict]] = None,
    think_close_delim: Optional[str] = None,
    think_budget: Optional[int] = None,
) -> RSAResult:
    """Run the full Markovian RSA loop and return the aggregated result.

    Round 0 expands the original prompt into N rollouts. Each aggregation
    round t in 1..T-1 builds N new prompts, each from a random K-subset of
    **round t-1's population only** (the Markov step), and samples one new
    candidate per prompt. The final answer is selected from the last round.
    """
    rng = rng or random.Random()
    usage = UsageTotals()
    semaphore = asyncio.Semaphore(params.max_concurrency)
    query = prompts.render_query(messages)
    request_system = prompts.extract_request_system(messages)
    # phase timers (s): gen_s = GPU generation (the structural T-rounds cost); tail_s/build_s =
    # recoverable event-loop orchestration (tail re-tokenization + aggregation-prompt building).
    _wall0 = time.monotonic()
    gen_s = tail_s = build_s = 0.0

    # Round 0 (expansion): N independent rollouts of the original prompt.
    _t = time.monotonic()
    population = await _run_round(
        client,
        [messages] * params.n,
        model=model,
        params=params,
        max_tokens=params.max_tokens,
        semaphore=semaphore,
        usage=usage,
        round_idx=0,
        chat_template_kwargs=chat_template_kwargs,
    )
    gen_s += time.monotonic() - _t
    rounds = [population]

    agg_budget = params.agg_max_tokens or params.max_tokens
    t = 1
    while t < params.t:
        # Markov step: sample ONLY from the previous round's population.
        prev = rounds[-1]
        _t = time.monotonic()
        tails = await _tails_for(client, prev, params)
        tail_s += time.monotonic() - _t
        _t = time.monotonic()
        message_sets = []
        for _ in range(params.n):
            chosen = rng.sample(prev, k=min(params.k, len(prev)))
            message_sets.append(
                prompts.build_aggregation_messages(
                    query, [tails[id(c)] for c in chosen], request_system
                )
            )
        build_s += time.monotonic() - _t
        _t = time.monotonic()
        population = await _run_round(
            client,
            message_sets,
            model=model,
            params=params,
            max_tokens=agg_budget,
            semaphore=semaphore,
            usage=usage,
            round_idx=t,
            chat_template_kwargs=chat_template_kwargs,
        )
        gen_s += time.monotonic() - _t
        rounds.append(population)
        t += 1

    _t = time.monotonic()
    final_text, method, vote_detail = await _select(
        client, params, rounds[-1], query, request_system, model, rng, usage,
        max_tokens=agg_budget,
        grammar=grammar, tools=tools, chat_template_kwargs=chat_template_kwargs,
        think_close_delim=think_close_delim, think_budget=think_budget,
    )
    sel_s = time.monotonic() - _t
    logger.info(
        "selection=%s, rounds=%d/%d, population=%d, total: %d requests, "
        "%d prompt + %d completion tokens",
        method,
        len(rounds),
        params.t,
        len(rounds[-1]),
        usage.n_requests,
        usage.prompt_tokens,
        usage.completion_tokens,
    )
    if _RSA_TIMING:
        # The split: gen_s is the structural cost (T sequential generation rounds, each barrier-gated);
        # tail_s+build_s is the recoverable inter-round orchestration on the event loop; sel_s is the
        # final selection (near-zero for majority_vote, ~one generation for final_aggregation). If
        # gen_s dominates and gen_s ~ sum of straggler-tails (see per-round lines), the win is in the
        # straggler barrier; if tail_s/build_s are large, it's the re-tokenization / event-loop stalls.
        total = time.monotonic() - _wall0
        orch = tail_s + build_s
        logger.info(
            "[rsa-timing] TOTAL %.2fs | gen(GPU rounds)=%.2fs (%.0f%%) | orch=%.2fs (tail=%.2fs "
            "build=%.2fs, %.0f%%) | select(%s)=%.2fs | N=%d K=%d T=%d",
            total, gen_s, 100 * gen_s / total if total else 0.0, orch, tail_s, build_s,
            100 * orch / total if total else 0.0, method, sel_s,
            params.n, params.k, params.t,
        )
    return RSAResult(
        final_text=final_text,
        population=rounds[-1],
        rounds=rounds,
        usage=usage,
        selection_method=method,
        vote_detail=vote_detail,
    )


async def _select(
    client: BackendClient,
    params: RSAParams,
    population: List[Candidate],
    query: str,
    request_system: Optional[str],
    model: str,
    rng: random.Random,
    usage: UsageTotals,
    max_tokens: int,
    grammar: Optional[str] = None,
    tools: Optional[List[dict]] = None,
    chat_template_kwargs: Optional[dict] = None,
    think_close_delim: Optional[str] = None,
    think_budget: Optional[int] = None,
) -> tuple:
    """Pick the final answer text from the final-round population."""
    # A structured request (response_format / json_schema / tools) CANNOT be satisfied by the
    # text-selection shortcuts (sample / majority_vote just return an unconstrained prose rollout).
    # Force the final aggregation call below, which regenerates the answer under the grammar/tools.
    structured = grammar is not None or tools is not None
    if not structured:
        if params.selection == "sample":
            return rng.choice(population).text, "sample", None

        answers = [extract.extract_boxed(c.text) for c in population]
        normalized = [
            extract.normalize_answer(a) if a is not None else None for a in answers
        ]
        extractable = sum(1 for a in normalized if a)
        want_vote = params.selection == "majority" or (
            params.selection == "auto" and extractable >= 2
        )
        if want_vote:
            vote = extract.majority_vote(answers)
            if vote is not None:
                winner, tally = vote
                matching = [c for c, a in zip(population, normalized) if a == winner]
                best = next(
                    (c for c in matching if c.finish_reason == "stop"), matching[0]
                )
                return best.text, "majority_vote", {"winner": winner, "tally": dict(tally)}
            if params.selection == "majority":
                return rng.choice(population).text, "sample", None

    # Fallback (selection == "final_agg", OR any structured request): one final aggregation call,
    # applying the grammar/tools/thinking constraints so the final answer honors response_format /
    # json_schema / tool-calling exactly like the plain lane.
    chosen = rng.sample(population, k=min(params.k, len(population)))
    tails = await _tails_for(client, chosen, params)
    msgs = prompts.build_final_selection_messages(
        query, [tails[id(c)] for c in chosen], request_system,
        for_tools=tools is not None,
    )
    final = await client.complete(
        msgs,
        model=model,
        # deliberately cooler than rollouts for a decisive final answer, but reuse
        # the same truncation knobs so the selection call can't run away either.
        temperature=0.3,
        max_tokens=max_tokens,
        top_p=params.top_p,
        top_k=params.top_k,
        ignore_eos=params.ignore_eos,
        stop=params.stop,
        max_retries=params.max_retries,
        grammar=grammar,
        tools=tools,
        chat_template_kwargs=chat_template_kwargs,
        think_close_delim=think_close_delim,
        think_budget=think_budget,
    )
    if final is None:
        return rng.choice(population).text, "sample", None
    usage.add(final)
    logger.info(
        "[rsa-timing] final-aggregation gen: %d prompt + %d completion tok, finish=%s, structured=%s",
        final.prompt_tokens, final.completion_tokens, final.finish_reason, structured,
    )
    return final.text, ("structured_aggregation" if structured else "final_aggregation"), None
