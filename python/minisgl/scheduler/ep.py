from __future__ import annotations

from typing import TYPE_CHECKING, List

import torch
import torch.distributed as dist

from minisgl.core import Batch
from minisgl.utils import init_logger

if TYPE_CHECKING:
    from minisgl.core import Req

logger = init_logger(__name__)


class SchedulerEPMixin:
    """Expert-parallel (EP) synchronous lockstep loop.

    EP shards the MoE experts across the DP replicas, so every MoE forward runs an all_gather +
    masked-local-expert GEMM + all_reduce over the DP/EP group (see MoELayer.forward / EPCommunicator).
    Those collectives REQUIRE every replica to issue the SAME-shape collective every step — otherwise
    a replica that skipped a step (it had no work) leaves the others' all_reduce hanging (deadlock).

    The lockstep is made implicit and GRAPH-CAPTURABLE by agreeing, OUTSIDE the captured graph, a
    single per-step decision via ONE gloo all_reduce(MAX) on a 2-int CPU tensor:

        [ max prefill-token-count across replicas , max decode batch-size across replicas ]

    Priority is decode-batches-via-graph when no replica wants prefill (the headline path):
      * DECODE step: common bs = the smallest captured graph bs >= the agreed max real decode bs.
        Every replica pads its decode batch to that bs (the existing dummy_req/graph mechanism) and
        replays the SAME captured graph — the in-graph EP collectives line up at fixed shapes. A
        replica with no decode work runs the all-dummy graph (its outputs are discarded).
      * PREFILL step (any replica has a ready prefill): each replica runs its local prefill EAGERLY;
        the MoE EP path zero-pads its token rows up to the agreed common prefill-token count
        (ep.pad_tokens) so the all_gather sees equal N on every rank. A replica with no prefill runs
        a 1-token dummy prefill to participate in the collectives.

    The MAX all_reduce is the ONLY per-step host sync and it is OUTSIDE the graph (it just SELECTS
    which graph to replay / which pad size to use), so CUDA-graph capture of the decode path — incl.
    the in-graph EP all_gather/all_reduce — is preserved.
    """

    # ---- provided by Scheduler / its other mixins -----------------------------------------------
    engine: object
    prefill_manager: object
    decode_manager: object
    prefill_budget: int

    def _ep_agree(self, local_prefill_tokens: int, local_decode_bs: int) -> tuple[int, int]:
        """One gloo all_reduce(MAX) over the DP/EP group; returns (max_prefill_tokens, max_decode_bs)."""
        t = torch.tensor([local_prefill_tokens, local_decode_bs], dtype=torch.int64)
        dist.all_reduce(t, op=dist.ReduceOp.MAX, group=self.engine.dp_cpu_group)
        return int(t[0].item()), int(t[1].item())

    def _ep_prepare_dummy_prefill(self):
        """A 1-token dummy prefill so an idle replica still issues the 40 MoE EP collectives in
        lockstep — without leaking a KV page or mutating the shared dummy_req.

        Unlike the decode dummy (which can use reqs=[]), the ZAYA CCA prefill metadata
        (query_start_loc / state_indices / num_seqs) is built from batch.reqs and MUST see one real
        seq, else `cu_seqlens end != num_tokens` (zaya.py). So reqs=[dummy_req], but with
        skip_alloc=True so NO KV page is allocated (the dummy_req's page_table already points at the
        reserved null page; allocating one leaked it every step -> CacheManager integrity crash under
        concurrency). The caller snapshots/restores dummy_req's length state so forward_batch's
        complete_one doesn't advance the shared dummy. The MoE EP path zero-pads the 1 row up to
        ep.pad_tokens before the all_gather, lining up with the busy replica's real prefill.
        """
        dummy = self.engine.dummy_req
        batch = Batch(reqs=[dummy], phase="prefill")
        batch.padded_reqs = [dummy]
        return self._finish_prepare(batch, skip_alloc=True)

    def ep_loop(self) -> None:
        """One EP lockstep iteration. Mirrors normal_loop but coordinates phase+size across replicas
        every step so the MoE collectives never deadlock."""
        engine = self.engine
        ep = engine.ctx.ep

        # Always drain our own ingress (non-blocking): under lockstep a replica must keep stepping
        # even with an empty local queue, so it can participate in another replica's collectives.
        for msg in self.receive_msg(blocking=False):
            self._process_one_msg(msg)

        # Local intent for THIS step (prefill takes priority within a replica, matching the DP path).
        prefill_batch = self.prefill_manager.schedule_next_batch(self.prefill_budget)
        # The agreed common size MUST be the TOKEN count (sum of extend_len), NOT batch.size (the
        # request count): the MoE all_gather pads to hidden_states.shape[0] = total query tokens, so
        # agreeing on req-count let one replica all_gather N=31 (real tokens) while another padded its
        # dummy to N=1 (req count) -> mismatched RCCL shapes -> the collective wedges. Sum extend_len
        # so every replica pads its rows to the SAME token count before the all_gather.
        local_prefill_tokens = (
            sum(r.extend_len for r in prefill_batch.reqs) if prefill_batch is not None else 0
        )
        # Decode bs is the running-req count (decode batch is built below only if we take a decode step).
        local_decode_bs = len(self.decode_manager.running_reqs) if self.decode_manager.runnable else 0

        max_prefill, max_decode = self._ep_agree(local_prefill_tokens, local_decode_bs)

        if max_prefill == 0 and max_decode == 0:
            # No replica has any work — every replica blocks on its own ingress (a coordinated idle).
            # The next iteration re-agrees once anyone wakes up; no collective is issued meanwhile.
            self.run_when_idle()
            return

        if max_prefill > 0:
            # ---- PREFILL lockstep step (eager) -----------------------------------------------------
            # If THIS replica didn't schedule a prefill, put back nothing and run a dummy 1-token
            # prefill so the MoE collectives line up. (Our real prefill_batch, if any, already popped
            # its reqs off the pending list in schedule_next_batch.)
            is_real = prefill_batch is not None
            # Common token count = max real prefill tokens across replicas; the MoE EP path zero-pads
            # each replica's rows up to it before the all_gather (then slices its real rows back).
            ep.pad_tokens = max_prefill
            # The dummy prefill runs reqs=[dummy_req] (CCA prefill needs a real seq), so
            # forward_batch's complete_one would advance the SHARED dummy_req's lengths every step
            # (unbounded position drift). Snapshot + restore them around the dummy forward.
            dr = engine.dummy_req
            saved_lens = None if is_real else (dr.cached_len, dr.device_len)
            try:
                forward_input = (
                    self._prepare_batch(prefill_batch)
                    if is_real
                    else self._ep_prepare_dummy_prefill()
                )
                data = (forward_input, self._forward(forward_input, track_reqs=is_real))
            finally:
                ep.pad_tokens = None
                if saved_lens is not None:
                    dr.cached_len, dr.device_len = saved_lens
            # A dummy replica discards its outputs (no real req -> no detokenize, no decode promotion).
            if is_real:
                self._process_last_data(data)
            return

        # ---- DECODE lockstep step (graph) ----------------------------------------------------------
        # Common bs = smallest captured graph bs >= the agreed max real decode bs (fall back to the
        # raw max when it exceeds the largest graph -> eager, still equal N on every replica).
        graph_bs_list: List[int] = self.engine.graph_runner.graph_bs_list
        common_bs = max_decode
        if graph_bs_list:
            bigger = [b for b in graph_bs_list if b >= max_decode]
            if bigger:
                common_bs = min(bigger)
        # Build this replica's decode batch (may be empty) and pad to common_bs with dummy reqs so the
        # in-graph EP collectives match. pad_tokens stays None (graph shapes already equal N=common_bs).
        ep.pad_tokens = None
        local_reqs: List[Req] = (
            self.decode_manager.ordered_reqs if self.decode_manager.runnable else []
        )
        batch = Batch(reqs=local_reqs, phase="decode") if local_reqs else None
        forward_input = self._ep_prepare_decode(batch, common_bs)
        # track_reqs only matters for the real-decode case (it re-filters can_decode); for an all-dummy
        # batch local_reqs is empty so filter_reqs over the (empty) reqs is a no-op either way.
        data = (forward_input, self._forward(forward_input, track_reqs=bool(local_reqs)))
        if local_reqs:
            self._process_last_data(data)

    def _ep_prepare_decode(self, batch: "Batch | None", common_bs: int):
        """Prepare a decode batch padded to the agreed common bs. An empty batch (no local work)
        carries ZERO real reqs and all-dummy padded_reqs, so this replica still replays the graph +
        collectives (its logits[:0] slice is empty -> no real outputs, no KV writes)."""
        engine = self.engine
        if batch is None:
            batch = Batch(reqs=[], phase="decode")  # no real reqs -> no allocate_paged / KV writes
        # Force the graph-selection padding to the AGREED common bs (overrides pad_batch's own choice
        # so every replica replays the IDENTICAL graph). padded_reqs = real reqs + dummies up to bs;
        # _make_positions/_make_input_tuple key on padded_reqs, allocate_paged/_make_write_tuple on
        # batch.reqs (real only), so an all-dummy batch allocates nothing and writes no real KV.
        batch.padded_reqs = batch.reqs + [engine.dummy_req] * (common_bs - len(batch.reqs))
        return self._finish_prepare(batch)
