"""In-process RSA backend client.

Drives the generation engine DIRECTLY through the FastAPI front-end
(``FrontendManager``) instead of an HTTP round-trip to the server's own OpenAI
port. This is what lets the entire Markovian-RSA loop run *inside* the
minisglang server process and be served on the normal port: a single
``/v1/chat/completions`` request with an ``rsa`` parameter fans out N rollouts as
N internal generations, aggregates over T rounds, and returns the final answer.

``complete()`` reuses the front-end's new_user / send_one / wait_for_ack
primitive; ``tail()`` / ``_get_tokenizer()`` are inherited from
:class:`~minisgl.rsa.core.BackendClient` (a local HF tokenizer loaded from the
served model path, so no ``/v1/models`` HTTP probe is needed). N rollouts run
concurrently on the same event loop and fan out across DP/EP replicas through the
existing per-replica request routing.
"""

from __future__ import annotations

import asyncio
import logging
from typing import List, Optional

from minisgl.core import SamplingParams
from minisgl.message import TokenizeMsg

from .core import BackendClient, Candidate

logger = logging.getLogger("minisgl.rsa.inproc")


class InProcessBackendClient(BackendClient):
    def __init__(self, state, model_path: str):
        # Deliberately do NOT call super().__init__ — that builds the openai/httpx
        # HTTP clients we are replacing. We only need the tokenizer machinery, set
        # up by hand below. _tokenizer_name is non-None so the inherited
        # _get_tokenizer never falls back to the HTTP /v1/models probe.
        self._state = state  # server.api_server.FrontendManager
        self._model_path = model_path
        self._tokenizer_name = model_path
        self._tokenizer = None
        self._tokenizer_lock = asyncio.Lock()

    async def default_model(self) -> str:
        return self._model_path

    async def close(self) -> None:
        return None

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
    ) -> Optional[Candidate]:
        """One chat generation through the in-process engine. Returns None on
        permanent failure (RSA tolerates dropped rollouts)."""
        state = self._state
        attempt = 0
        while True:
            uid = state.new_user()
            try:
                await state.send_one(
                    TokenizeMsg(
                        uid=uid,
                        # list of {role, content}; the tokenizer applies the chat template
                        text=messages,
                        sampling_params=SamplingParams(
                            temperature=temperature,
                            top_p=top_p,
                            top_k=top_k,
                            ignore_eos=ignore_eos,
                            max_tokens=max_tokens,
                            stop=list(stop) if stop else [],
                        ),
                    )
                )
                text = ""
                # Do NOT break on `finished`: letting wait_for_ack run to its natural
                # StopAsyncIteration is what triggers its del ack_map/event_map[uid]
                # cleanup. A manual early break would leak one entry per rollout across
                # the N*T+ generations a single RSA call issues.
                async for ack in state.wait_for_ack(uid):
                    text += ack.incremental_output
                return Candidate(
                    text=text,
                    # the front-end UserReply carries no stop/length split; default to
                    # "stop" (selection only prefers, never requires, a "stop" finish).
                    finish_reason="stop",
                    prompt_tokens=0,
                    completion_tokens=0,
                )
            except Exception as e:  # noqa: BLE001 - mirror the HTTP client's resilience
                state.ack_map.pop(uid, None)
                state.event_map.pop(uid, None)
                if attempt < max_retries:
                    attempt += 1
                    logger.warning("in-proc rollout error, retry %d: %r", attempt, e)
                    continue
                logger.error("in-proc rollout failed after %d retries: %r", attempt, e)
                return None
