from __future__ import annotations

import multiprocessing as mp
import os
from typing import Dict, List

import torch
from minisgl.message import (
    AbortBackendMsg,
    AbortMsg,
    BaseBackendMsg,
    BaseFrontendMsg,
    BaseTokenizerMsg,
    BatchBackendMsg,
    BatchFrontendMsg,
    BatchTokenizerMsg,
    DetokenizeMsg,
    StatsFrontendMsg,
    StatsMsg,
    TokenizeMsg,
    UserMsg,
    UserReply,
)
from minisgl.utils import (
    ZmqPullQueue,
    ZmqPushQueue,
    init_logger,
    load_tokenizer,
    resolve_stop_token_ids,
)


def _unwrap_msg(msg: BaseTokenizerMsg) -> List[BaseTokenizerMsg]:
    if isinstance(msg, BatchTokenizerMsg):
        return msg.data
    return [msg]


def _is_cam_msg(sampling_params) -> bool:
    """A request that touches the per-replica CAM store: it carries a subject to deliver/write, an
    object to write, or a control op (facts/forget/stats/retrieve). These must all be pinned to ONE
    DP replica (below) because `engine.cam` is per-scheduler-replica — round-robining them would split
    a user's writes and reads across independent stores."""
    return bool(getattr(sampling_params, "mem_subject", None)
                or getattr(sampling_params, "mem_remember", None)
                or getattr(sampling_params, "mem_op", None))


@torch.inference_mode()
def tokenize_worker(
    *,
    tokenizer_path: str,
    addr: str,
    create: bool,
    backend_addrs: List[str],
    frontend_addr: str,
    local_bs: int,
    tokenizer_id: int = -1,
    model_source: str = "huggingface",
    ack_queue: mp.Queue[str] | None = None,
) -> None:
    # One PUSH socket per DP replica's rank-0 ingress. With dp_size=1 this is a single socket and the
    # routing below collapses to the historical single-backend behaviour. Each tokenizer worker holds
    # its OWN round-robin cursor; aborts are BROADCAST to every replica (abort_req is idempotent — a
    # replica that does not own the uid no-ops), so no shared uid->replica map is needed across workers.
    assert len(backend_addrs) >= 1
    send_backends = [
        ZmqPushQueue(a, create=False, encoder=BaseBackendMsg.encoder) for a in backend_addrs
    ]
    dp_size = len(send_backends)
    rr_cursor = 0
    send_frontend = ZmqPushQueue(frontend_addr, create=False, encoder=BaseFrontendMsg.encoder)
    recv_listener = ZmqPullQueue(addr, create=create, decoder=BatchTokenizerMsg.decoder)
    assert local_bs > 0
    tokenizer = load_tokenizer(tokenizer_path)
    logger = init_logger(__name__, f"tokenizer_{tokenizer_id}")

    from .detokenize import DetokenizeManager
    from .tokenize import TokenizeManager

    tokenize_manager = TokenizeManager(tokenizer)
    detokenize_manager = DetokenizeManager(
        tokenizer, resolve_stop_token_ids(tokenizer_path, tokenizer)
    )

    # Per-uid request metadata for the OpenAI usage block + stop strings, set when a prompt is
    # tokenized and read/cleared when its replies stream back (same process owns both directions).
    prompt_tokens_map: Dict[int, int] = {}
    stop_map: Dict[int, List[str]] = {}
    keep_map: Dict[int, List[str]] = {}  # INCLUSIVE stops (matched-string kept), e.g. tool closers

    if ack_queue is not None:
        ack_queue.put(f"Tokenize server {tokenizer_id} is ready")

    try:
        while True:
            pending_msg = _unwrap_msg(recv_listener.get())
            while len(pending_msg) < local_bs and not recv_listener.empty():
                pending_msg.extend(_unwrap_msg(recv_listener.get()))

            logger.debug(f"Received {len(pending_msg)} messages")

            detokenize_msg = [m for m in pending_msg if isinstance(m, DetokenizeMsg)]
            tokenize_msg = [m for m in pending_msg if isinstance(m, TokenizeMsg)]
            abort_msg = [m for m in pending_msg if isinstance(m, AbortMsg)]
            # Scheduler metrics snapshots (only reach the detokenizer worker, which owns the frontend
            # link): forward each to the frontend as a StatsFrontendMsg for the /metrics endpoint.
            stats_msg = [m for m in pending_msg if isinstance(m, StatsMsg)]
            for sm in stats_msg:
                send_frontend.put(
                    StatsFrontendMsg(
                        dp_rank=sm.dp_rank,
                        spec_draft_tokens=sm.spec_draft_tokens,
                        spec_accepted_tokens=sm.spec_accepted_tokens,
                        spec_emitted_tokens=sm.spec_emitted_tokens,
                        spec_steps=sm.spec_steps,
                        running_requests=sm.running_requests,
                        waiting_requests=sm.waiting_requests,
                        kv_tokens_total=sm.kv_tokens_total,
                        kv_tokens_used=sm.kv_tokens_used,
                        gdn_slots_total=sm.gdn_slots_total,
                        gdn_slots_used=sm.gdn_slots_used,
                        cam_facts=sm.cam_facts,
                        cam_namespaces=sm.cam_namespaces,
                        cam_evicted=sm.cam_evicted,
                        cam_max_bank_load=sm.cam_max_bank_load,
                        cam_crowded_banks=sm.cam_crowded_banks,
                        cam_recovered_from_backup=sm.cam_recovered_from_backup,
                        cam_index_nn_cos_max=sm.cam_index_nn_cos_max,
                        cam_last_save_age_s=sm.cam_last_save_age_s,
                    )
                )
            assert len(detokenize_msg) + len(tokenize_msg) + len(abort_msg) + len(stats_msg) == len(
                pending_msg
            )
            if len(detokenize_msg) > 0:
                results = detokenize_manager.detokenize(detokenize_msg, stop_map, keep_map)
                replies: List[UserReply] = []
                stop_abort_uids: List[int] = []
                for msg, res in zip(detokenize_msg, results, strict=True):
                    finished = msg.finished or res.stop_hit
                    # A stop string finished the request before the engine did -> tell the backend to
                    # abort it (broadcast; idempotent on replicas that don't own the uid).
                    if res.stop_hit and not msg.finished:
                        stop_abort_uids.append(msg.uid)
                    # finish_reason: a stop-STRING match is always "stop"; otherwise carry the engine's
                    # reason ("length" for max_tokens/KV-budget, "stop" for EOS), defaulting to "stop"
                    # for finishes the scheduler didn't tag (e.g. CAM result-string, older paths).
                    if not finished:
                        finish_reason = None
                    elif res.stop_hit:
                        finish_reason = "stop"
                    else:
                        finish_reason = msg.finish_reason or "stop"
                    replies.append(
                        UserReply(
                            uid=msg.uid,
                            incremental_output=res.incremental,
                            finished=finished,
                            completion_tokens=res.completion_tokens,
                            prompt_tokens=prompt_tokens_map.get(msg.uid, 0),
                            finish_reason=finish_reason,
                        )
                    )
                    if finished:
                        prompt_tokens_map.pop(msg.uid, None)
                        stop_map.pop(msg.uid, None)
                        keep_map.pop(msg.uid, None)
                batch_output: BaseFrontendMsg = (
                    replies[0] if len(replies) == 1 else BatchFrontendMsg(data=list(replies))
                )
                send_frontend.put(batch_output)
                if stop_abort_uids:
                    abort_out: BaseBackendMsg = (
                        AbortBackendMsg(uid=stop_abort_uids[0])
                        if len(stop_abort_uids) == 1
                        else BatchBackendMsg(data=[AbortBackendMsg(uid=u) for u in stop_abort_uids])
                    )
                    for sb in send_backends:
                        sb.put(abort_out)

            if len(tokenize_msg) > 0:
                tensors = tokenize_manager.tokenize(tokenize_msg)
                # Record prompt length (usage) + stop strings for each new request; consumed when its
                # replies stream back through the detokenize branch above.
                for msg, t in zip(tokenize_msg, tensors, strict=True):
                    prompt_tokens_map[msg.uid] = int(t.numel())
                    if msg.sampling_params.stop:
                        stop_map[msg.uid] = list(msg.sampling_params.stop)
                    if msg.sampling_params.stop_keep:
                        keep_map[msg.uid] = list(msg.sampling_params.stop_keep)
                user_msgs = [
                    UserMsg(uid=msg.uid, input_ids=t, sampling_params=msg.sampling_params)
                    for msg, t in zip(tokenize_msg, tensors, strict=True)
                ]
                # Per-replica routing: bucket each UserMsg to ONE replica (round-robin), then flush one
                # batch per replica. Each request is delivered to exactly one DP replica. EXCEPTION: CAM
                # store ops are PINNED to a single replica (MINISGL_CAM_DP_RANK, default 0) so every
                # remember/ask/retrieve sees the same per-replica engine.cam — round-robin would scatter a
                # user's writes and reads across independent stores. (dp_size==1 -> cam_rank 0 == no-op.)
                cam_rank = int(os.environ.get("MINISGL_CAM_DP_RANK", "0")) % dp_size
                per_replica: List[List[UserMsg]] = [[] for _ in range(dp_size)]
                for um in user_msgs:
                    if _is_cam_msg(um.sampling_params):
                        per_replica[cam_rank].append(um)
                    else:
                        per_replica[rr_cursor].append(um)
                        rr_cursor = (rr_cursor + 1) % dp_size
                for dp_rank, msgs in enumerate(per_replica):
                    if not msgs:
                        continue
                    out: BaseBackendMsg = (
                        msgs[0] if len(msgs) == 1 else BatchBackendMsg(data=list(msgs))
                    )
                    send_backends[dp_rank].put(out)
            if len(abort_msg) > 0:
                # Broadcast aborts to ALL replicas (idempotent on the unowning replicas) — the owning
                # replica frees the request; the rest no-op. Avoids a cross-worker uid->replica map.
                abort_batch: BaseBackendMsg = (
                    AbortBackendMsg(uid=abort_msg[0].uid)
                    if len(abort_msg) == 1
                    else BatchBackendMsg(data=[AbortBackendMsg(uid=m.uid) for m in abort_msg])
                )
                for sb in send_backends:
                    sb.put(abort_batch)
    except KeyboardInterrupt:
        pass
