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


@dataclass
class StatsFrontendMsg(BaseFrontendMsg):
    """Scheduler metrics snapshot forwarded by the detokenizer to the frontend (one per DP replica per
    flush). Mirrors message.tokenizer.StatsMsg; consumed by FrontendManager.listen() to update the
    Prometheus /metrics backend snapshot. See server/metrics.py."""

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
