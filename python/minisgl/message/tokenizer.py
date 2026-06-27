from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from minisgl.core import SamplingParams

from .utils import deserialize_type, serialize_type


@dataclass
class BaseTokenizerMsg:
    @staticmethod
    def encoder(msg: BaseTokenizerMsg) -> Dict:
        return serialize_type(msg)

    @staticmethod
    def decoder(json: Dict) -> BaseTokenizerMsg:
        return deserialize_type(globals(), json)


@dataclass
class BatchTokenizerMsg(BaseTokenizerMsg):
    data: List[BaseTokenizerMsg]


@dataclass
class DetokenizeMsg(BaseTokenizerMsg):
    uid: int
    next_token: int
    finished: bool
    # Speculative decoding commits several tokens for one req in a single step. They travel in ONE
    # message (next_token + extra_tokens, in order) so the incremental detokenizer — which keys
    # streaming offsets by uid and assumes one message per uid per batch — stays correct. Empty for
    # the normal one-token-per-step path. See DetokenizeManager.detokenize and SPEC_DECODE.md.
    extra_tokens: List[int] = field(default_factory=list)


@dataclass
class TokenizeMsg(BaseTokenizerMsg):
    uid: int
    text: str | List[Dict[str, str]]
    sampling_params: SamplingParams


@dataclass
class AbortMsg(BaseTokenizerMsg):
    uid: int
