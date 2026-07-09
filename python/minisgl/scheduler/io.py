from __future__ import annotations

from typing import TYPE_CHECKING, Final, List

import torch
from minisgl.message import (
    BaseBackendMsg,
    BaseTokenizerMsg,
    BatchTokenizerMsg,
    DetokenizeMsg,
    StatsMsg,
)
from minisgl.utils import ZmqPubQueue, ZmqPullQueue, ZmqPushQueue, ZmqSubQueue, init_logger

if TYPE_CHECKING:
    from .config import SchedulerConfig

logger = init_logger(__name__)


class SchedulerIOMixin:
    """
    Mixin class for Scheduler I/O operations.

    This class handles the communication between the scheduler and the tokenizer.

    Public Utilities:
        receive_msg: Function to receive messages from the tokenizer.
        send_result: Function to send results back to the tokenizer.
        sync_all_ranks: Function to synchronize all ranks on CPU side.
    """

    def __init__(self, config: SchedulerConfig, tp_cpu_group: torch.distributed.ProcessGroup):
        tp_info = config.tp_info
        self.tp_cpu_group: Final = tp_cpu_group
        self._tp_size: Final = tp_info.size
        self._tp_is_primary: Final = tp_info.is_primary()
        if config.offline_mode:
            self.receive_msg = self.offline_receive_msg
            self.send_result = self.offline_send_result
            return  # early exit

        if tp_info.is_primary():
            self._recv_from_tokenizer: Final = ZmqPullQueue(
                config.zmq_backend_addr,
                create=True,
                decoder=BaseBackendMsg.decoder,
            )
            self._send_into_tokenizer: Final = ZmqPushQueue(
                config.zmq_detokenizer_addr,
                create=config.backend_create_detokenizer_link,
                encoder=BaseTokenizerMsg.encoder,
            )

        recv = self._recv_msg_single_rank
        send = self._reply_tokenizer_rank0
        if tp_info.size > 1:
            if tp_info.is_primary():
                recv = self._recv_msg_multi_rank0
                self._send_into_ranks: Final = ZmqPubQueue(
                    config.zmq_scheduler_broadcast_addr, create=True, encoder=BaseBackendMsg.encoder
                )
            else:
                recv = self._recv_msg_multi_rank1
                send = self._reply_tokenizer_rank1
                self._recv_from_rank0: Final = ZmqSubQueue(
                    config.zmq_scheduler_broadcast_addr,
                    create=False,
                    decoder=BaseBackendMsg.decoder,
                )

        self.receive_msg = recv
        self.send_result = send

    def run_when_idle(self):
        raise NotImplementedError("should be implemented")

    def offline_receive_msg(self, blocking: bool = False) -> List[BaseBackendMsg]:
        raise NotImplementedError("should be implemented")

    def offline_send_result(self, reply: List[DetokenizeMsg]) -> None:
        raise NotImplementedError("should be implemented")

    def sync_all_ranks(self) -> None:
        self.tp_cpu_group.barrier().wait()

    def _flush_stats(self) -> None:
        """Metrics snapshot hook. Overridden with the real sampler on the Scheduler; a bare mixin (or
        a scheduler built before metrics wiring) no-ops."""
        return

    def _emit_stats(self, msg: StatsMsg) -> None:
        """Push a scheduler metrics snapshot down the detokenizer link (tp-primary only). Rides the
        existing scheduler -> detokenizer PUSH socket; the detokenizer forwards it to the frontend."""
        sock = getattr(self, "_send_into_tokenizer", None)
        if sock is not None:
            sock.put(msg)

    def _recv_msg_single_rank(self, blocking: bool = False) -> List[BaseBackendMsg]:
        self._flush_stats()
        pending_msgs: List[BaseBackendMsg] = []
        if blocking:
            self.run_when_idle()
            pending_msgs.append(self._recv_from_tokenizer.get())
        while not self._recv_from_tokenizer.empty():
            pending_msgs.append(self._recv_from_tokenizer.get())
        return pending_msgs

    def _recv_msg_multi_rank0(self, blocking: bool = False) -> List[BaseBackendMsg]:
        # Collect ALL pending raw messages (block for >=1 when requested), THEN broadcast the total
        # count and fan them out. The broadcast count is the single source of truth for how many
        # messages rank1 pulls, so a per-rank `blocking` disagreement can no longer desync the
        # PUB/SUB stream. (The old code special-cased the blocking first message — sending it
        # out-of-band before the count broadcast — which hard-deadlocked whenever rank0's and rank1's
        # `blocking` differed: rank0 stuck in broadcast() while rank1 waited forever in get().)
        self._flush_stats()
        raw_msgs: List[bytes] = []
        if blocking:
            self.run_when_idle()
            raw_msgs.append(self._recv_from_tokenizer.get_raw())
        while not self._recv_from_tokenizer.empty():
            raw_msgs.append(self._recv_from_tokenizer.get_raw())

        # broadcast the TOTAL number of raw messages to all ranks
        src_tensor = torch.tensor(len(raw_msgs))
        self.tp_cpu_group.broadcast(src_tensor, root=0).wait()

        pending_msgs: List[BaseBackendMsg] = []
        for raw in raw_msgs:
            self._send_into_ranks.put_raw(raw)
            pending_msgs.append(self._recv_from_tokenizer.decode(raw))
        return pending_msgs

    def _recv_msg_multi_rank1(self, blocking: bool = False) -> List[BaseBackendMsg]:
        # rank1 NEVER independently block-receives a message: it pulls EXACTLY the count rank0
        # broadcasts. This is what makes the protocol immune to a per-rank `blocking` disagreement
        # (the old `if blocking: self._recv_from_rank0.get()` here is precisely what deadlocked
        # against rank0's count broadcast). `blocking` now only drives the idle bookkeeping.
        if blocking:
            self.run_when_idle()

        dst_tensor = torch.tensor(-1)
        self.tp_cpu_group.broadcast(dst_tensor, root=0).wait()
        dst_length = int(dst_tensor.item())

        pending_msgs: List[BaseBackendMsg] = []
        for _ in range(dst_length):
            pending_msgs.append(self._recv_from_rank0.get())
        return pending_msgs

    # Raw byte markers for the one-time PUB/SUB subscription handshake. Distinct from each other; they
    # never collide with real traffic because the handshake completes before the server accepts reqs.
    _PUBSUB_SYNC: Final = b"\x00minisgl-pubsub-sync"
    _PUBSUB_SYNC_END: Final = b"\x00minisgl-pubsub-sync-end"

    def establish_inter_rank_link(self) -> None:
        """Guarantee the rank0->rank{1..} PUB/SUB fan-out is live before any real request flows.

        ZMQ PUB/SUB has no connection handshake: a message PUBlished before a SUBscriber's
        subscription has propagated is silently DROPPED (the "slow joiner"). With the scheduler's
        broadcast socket that means rank0's very first fan-out message can vanish, leaving rank1
        blocked forever in get() and rank0 in the count broadcast (the first-request deadlock).

        Run once, before the main loop: rank0 PUBlishes SYNC markers until rank1 confirms (reliably,
        over the GLOO group) that it has received one, then a final SYNC_END marker; rank1 drains
        in-order through SYNC_END so its SUB buffer is empty before real traffic. ZMQ preserves order
        on a connection, so once a SYNC arrives the link is live and no real message is lost."""
        if self._tp_size <= 1:
            return
        if self._tp_is_primary:
            while True:
                self._send_into_ranks.put_raw(self._PUBSUB_SYNC)
                ack = torch.tensor(0)
                self.tp_cpu_group.broadcast(ack, root=1).wait()
                if int(ack.item()) == 1:
                    break
            self._send_into_ranks.put_raw(self._PUBSUB_SYNC_END)
        else:
            seen = False
            while True:
                while not self._recv_from_rank0.empty():
                    if self._recv_from_rank0.get_raw() == self._PUBSUB_SYNC:
                        seen = True
                ack = torch.tensor(1 if seen else 0)
                self.tp_cpu_group.broadcast(ack, root=1).wait()
                if seen:
                    break
            # drain (in order) through the END marker so no SYNC bytes leak into the real stream
            while self._recv_from_rank0.get_raw() != self._PUBSUB_SYNC_END:
                pass

    def _reply_tokenizer_rank0(self, reply: List[DetokenizeMsg]) -> None:
        num_reply = len(reply)
        logger.debug_rank0(f"Replying to tokenizer: {num_reply} messages")
        if num_reply == 1:
            self._send_into_tokenizer.put(reply[0])
        elif num_reply > 1:
            self._send_into_tokenizer.put(BatchTokenizerMsg(data=reply))  # type: ignore

    def _reply_tokenizer_rank1(self, reply: List[DetokenizeMsg]) -> None:
        _ = reply  # do nothing for non-primary ranks
