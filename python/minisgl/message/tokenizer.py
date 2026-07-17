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
    # OpenAI tool specs to inject into the chat template (tool-trained models emit <tool_call> blocks
    # the api_server parses back). None = no tools offered. Defaulted so existing senders are unchanged.
    tools: List[Dict] | None = None
    # Extra kwargs forwarded verbatim to `apply_chat_template` (e.g. {"enable_thinking": False} to
    # turn a reasoning model's thinking mode off). None = template defaults (thinking ON for Qwen3).
    chat_template_kwargs: Dict | None = None


@dataclass
class AbortMsg(BaseTokenizerMsg):
    uid: int


@dataclass
class StatsMsg(BaseTokenizerMsg):
    """Periodic scheduler-side metrics snapshot, piggybacked on the scheduler -> detokenizer ZMQ path
    (no new socket). The detokenizer forwards it to the frontend as a StatsFrontendMsg, where it feeds
    the /metrics endpoint. One per DP replica per flush; the frontend sums across replicas. Counters
    are cumulative (monotonic); the running/waiting/kv/gdn fields are instantaneous gauges."""

    dp_rank: int
    spec_draft_tokens: int
    spec_accepted_tokens: int
    spec_emitted_tokens: int
    spec_steps: int
    running_requests: int
    waiting_requests: int
    kv_tokens_total: int
    kv_tokens_used: int
    gdn_slots_total: int
    gdn_slots_used: int
    # CAM editable-memory store stats (0 when CAM is off), aggregated across namespaces.
    cam_facts: int = 0
    cam_namespaces: int = 0
    cam_evicted: int = 0
    cam_max_bank_load: int = 0
    cam_crowded_banks: int = 0
    cam_recovered_from_backup: int = 0    # 1 if boot restored from .bak (primary store was lost)
    cam_index_nn_cos_max: float = 0.0     # worst cosine-index crowding (vs deliver_tau = interference wall)
    cam_last_save_age_s: float = 0.0      # seconds since last successful store save (durability risk)
