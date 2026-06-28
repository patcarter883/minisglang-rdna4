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
import random
import time
from dataclasses import dataclass, field
from typing import List, Optional

import httpx
import openai

from . import extract, prompts
from .config import RSAParams

logger = logging.getLogger("minisgl.rsa")


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
        max_retries: int = 1,
    ) -> Optional[Candidate]:
        """One chat completion; returns None on permanent failure."""
        attempt = 0
        while True:
            try:
                resp = await self.openai.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
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
) -> List[Candidate]:
    """Fan out one rollout per message set, bounded by *semaphore*."""

    async def one(messages: List[dict]) -> Optional[Candidate]:
        async with semaphore:
            return await client.complete(
                messages,
                model=model,
                temperature=params.temperature,
                max_tokens=max_tokens,
                max_retries=params.max_retries,
            )

    start = time.monotonic()
    results = await asyncio.gather(*(one(m) for m in message_sets))
    population = [c for c in results if c is not None]
    for c in population:
        usage.add(c)
    if not population:
        raise RSAError(f"round {round_idx}: all {len(message_sets)} rollouts failed")
    logger.info(
        "round %d: %d/%d candidates, %d prompt + %d completion tokens, %.1fs",
        round_idx,
        len(population),
        len(message_sets),
        sum(c.prompt_tokens for c in population),
        sum(c.completion_tokens for c in population),
        time.monotonic() - start,
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

    # Round 0 (expansion): N independent rollouts of the original prompt.
    population = await _run_round(
        client,
        [messages] * params.n,
        model=model,
        params=params,
        max_tokens=params.max_tokens,
        semaphore=semaphore,
        usage=usage,
        round_idx=0,
    )
    rounds = [population]

    agg_budget = params.agg_max_tokens or params.max_tokens
    t = 1
    while t < params.t:
        # Markov step: sample ONLY from the previous round's population.
        prev = rounds[-1]
        tails = await _tails_for(client, prev, params)
        message_sets = []
        for _ in range(params.n):
            chosen = rng.sample(prev, k=min(params.k, len(prev)))
            message_sets.append(
                prompts.build_aggregation_messages(
                    query, [tails[id(c)] for c in chosen], request_system
                )
            )
        population = await _run_round(
            client,
            message_sets,
            model=model,
            params=params,
            max_tokens=agg_budget,
            semaphore=semaphore,
            usage=usage,
            round_idx=t,
        )
        rounds.append(population)
        t += 1

    final_text, method, vote_detail = await _select(
        client, params, rounds[-1], query, request_system, model, rng, usage,
        max_tokens=agg_budget,
    )
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
) -> tuple:
    """Pick the final answer text from the final-round population."""
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

    # Fallback (and selection == "final_agg"): one final aggregation call.
    chosen = rng.sample(population, k=min(params.k, len(population)))
    tails = await _tails_for(client, chosen, params)
    msgs = prompts.build_final_selection_messages(
        query, [tails[id(c)] for c in chosen], request_system
    )
    final = await client.complete(
        msgs,
        model=model,
        temperature=0.3,
        max_tokens=max_tokens,
        max_retries=params.max_retries,
    )
    if final is None:
        return rng.choice(population).text, "sample", None
    usage.add(final)
    return final.text, "final_aggregation", None
