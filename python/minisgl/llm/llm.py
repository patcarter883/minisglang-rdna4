from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from minisgl.core import SamplingParams
from minisgl.distributed import DistributedInfo
from minisgl.message import (
    BaseBackendMsg,
    DetokenizeMsg,
    UserMsg,
)
from minisgl.scheduler import Scheduler, SchedulerConfig


class RequestAllFinished(Exception):
    pass


@dataclass
class RequestStatus:
    uid: int
    input_ids: List[int]
    output_ids: List[int]
    # TERMINAL REJECTION, carried through to `generate`'s result dict. See `offline_send_result`.
    error: str | None = None


class LLM(Scheduler):
    def __init__(self, model_path: str, dtype: torch.dtype = torch.bfloat16, **kwargs):
        # TP=1 unless the caller supplies its own `tp_info`. A TP>1 offline run is ONE PROCESS PER
        # RANK (the caller spawns them, exactly as server/launch.py does), each building its own LLM
        # with its own rank. Every rank then drives the IDENTICAL request stream from its own
        # `pending_requests`, so the ranks stay in lockstep without the ZMQ rank0->rank1 fan-out that
        # the served path uses. That is what lets tools/kv_fp8_calibrate.py calibrate a model whose
        # weights do not fit on one card.
        kwargs.setdefault("tp_info", DistributedInfo(0, 1))
        config = SchedulerConfig(
            model_path=model_path,
            dtype=dtype,
            offline_mode=True,
            **kwargs,
        )
        super().__init__(config)
        self.pending_requests: List[Tuple[List[int] | str, SamplingParams]] = []
        self.status_map: Dict[int, RequestStatus] = {}
        self.counter = 0

    def _tokenize_one(self, prompt: List[int] | str) -> torch.Tensor:
        if isinstance(prompt, str):
            return self.tokenizer.encode(prompt, return_tensors="pt").view(-1).to(torch.int32)
        else:
            return torch.tensor(prompt, dtype=torch.int32, device="cpu")

    def offline_receive_msg(self, blocking: bool = False) -> List[BaseBackendMsg]:
        if blocking and len(self.pending_requests) == 0:
            raise RequestAllFinished()
        results: List[BaseBackendMsg] = []
        added, sum_input_len = 0, 0
        for tokens_or_prompt, sampling_params in self.pending_requests:
            if sum_input_len >= self.prefill_budget:
                break
            input_ids = self._tokenize_one(tokens_or_prompt)
            sum_input_len += len(input_ids)
            uid, added = self.counter + added, added + 1
            results.append(UserMsg(uid=uid, input_ids=input_ids, sampling_params=sampling_params))
            self.status_map[uid] = RequestStatus(
                uid=uid,
                input_ids=(
                    input_ids.tolist() if isinstance(tokens_or_prompt, str) else tokens_or_prompt
                ),
                output_ids=[],
            )
        self.counter += added
        self.pending_requests = self.pending_requests[added:]
        return results

    def offline_send_result(self, reply: List[DetokenizeMsg]) -> None:
        for msg in reply:
            status = self.status_map[msg.uid]
            # A TERMINAL REJECTION IS NOT A TOKEN. `Scheduler._handle_msg` refuses a request whose
            # input exceeds `max_seq_len` by replying `DetokenizeMsg(next_token=0, finished=True,
            # error=...)`, and that field's own comment says the filler token "must NOT be decoded".
            # The HTTP detokenizer honours it; this path did not — it appended token 0 like any other
            # token and dropped `error` on the floor, so an over-length request came back as a
            # successful ONE-TOKEN completion ('!' for this checkpoint) with no signal anywhere the
            # caller could see. MEASURED 2026-09-06: the QSA long-context ladder scored a rejected
            # 131,056-token request as a passing run (docs/measurements/QSA_INDEXER.md §4b). Every
            # offline consumer — the benches in tools/, kv_fp8_calibrate.py, every harness in tests/
            # — reads this path, so the failure was silent-wrong for all of them, not just one.
            if msg.error is not None:
                status.error = msg.error
                continue
            if not (msg.finished and msg.next_token in self.eos_token_ids):
                status.output_ids.append(msg.next_token)

    def generate(
        self,
        prompts: List[str] | List[List[int]],
        sampling_params: List[SamplingParams] | SamplingParams,
    ) -> List[Dict[str, str | List[int]]]:
        self.pending_requests = []
        self.status_map = {}
        self.counter = 0
        if isinstance(sampling_params, SamplingParams):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.pending_requests.append((prompt, sp))
        try:
            self.run_forever()
        except RequestAllFinished:
            pass
        results: List[Dict[str, str | List[int]]] = []
        for i in range(len(prompts)):
            status = self.status_map[i]
            output_text = self.tokenizer.decode(status.output_ids)
            # `error` is always present (None on the normal path) so a caller can test it without a
            # `.get` and cannot mistake "key absent" for "no error".
            results.append({"text": output_text, "token_ids": status.output_ids,
                            "error": status.error})
        return results

    def base_logits(self, token_ids: List[int]) -> torch.Tensor:
        """Last-position logits [vocab] for a single prompt from the SERVED model (no CAM staged).

        The CAM write gate (`/cam/remember`) needs base p(object) with memory OFF. Rather than a second
        HF copy, capture the served model's own logits by hooking the sampler for a 1-token prefill of
        the prompt (the logit-oracle pattern, tools/oracle_ours.py). CAM is cleared so the L24 tap is a
        byte-exact no-op. Returns the last-position logits on the engine device."""
        from minisgl.core import SamplingParams

        if self.engine.cam is not None:
            self.engine.model.model.clear_cam()
        cap: Dict[str, torch.Tensor] = {}
        orig = self.engine.sampler.sample

        def _hook(logits, args):
            cap["l"] = logits[0].detach().float().clone()  # [vocab] last position, req 0
            return orig(logits, args)

        self.engine.sampler.sample = _hook  # type: ignore[method-assign]
        try:
            self.generate([list(token_ids)], SamplingParams(temperature=0.0, max_tokens=1))
        finally:
            self.engine.sampler.sample = orig  # type: ignore[method-assign]
        return cap["l"]
