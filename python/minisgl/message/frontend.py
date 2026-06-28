from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .utils import deserialize_type, serialize_type


@dataclass
class BaseFrontendMsg:
    @staticmethod
    def encoder(msg: BaseFrontendMsg) -> Dict:
        return serialize_type(msg)

    @staticmethod
    def decoder(json: Dict) -> BaseFrontendMsg:
        return deserialize_type(globals(), json)


@dataclass
class BatchFrontendMsg(BaseFrontendMsg):
    data: List[BaseFrontendMsg]


@dataclass
class UserReply(BaseFrontendMsg):
    uid: int
    incremental_output: str
    finished: bool
    # Token accounting for the OpenAI `usage` block. completion_tokens is CUMULATIVE for this uid at
    # this reply (monotonic; the final reply carries the total). prompt_tokens is the input length,
    # filled by the owning tokenizer process (0 until known). finish_reason is "stop" (EOS / stop
    # string) or "length" (max_tokens); None until the reply is finished.
    completion_tokens: int = 0
    prompt_tokens: int = 0
    finish_reason: str | None = None
