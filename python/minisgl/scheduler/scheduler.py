from __future__ import annotations

import os
from typing import TYPE_CHECKING, List, NamedTuple, NoReturn, Set, Tuple, TypeAlias

import torch
from minisgl.core import Batch, Req
from minisgl.env import ENV
from minisgl.message import (
    AbortBackendMsg,
    BaseBackendMsg,
    BatchBackendMsg,
    DetokenizeMsg,
    ExitMsg,
    UserMsg,
)
from minisgl.spec import propose_ngram, verify_greedy
from minisgl.utils import init_logger, load_tokenizer

from .cache import CacheManager
from .config import SchedulerConfig
from .decode import DecodeManager
from .gdn_slots import GDNSlotManager
from .io import SchedulerIOMixin
from .prefill import ChunkedReq, PrefillManager
from .table import TableManager

if TYPE_CHECKING:
    from minisgl.engine import BatchSamplingArgs, ForwardOutput


logger = init_logger(__name__)

Indice2D: TypeAlias = Tuple[torch.Tensor, torch.Tensor]


# For overlap scheduling, we also need to cache some other data to avoid IMA
class ForwardInput(NamedTuple):
    batch: Batch
    sample_args: BatchSamplingArgs
    input_tuple: Indice2D  # (token_mapping, positions)
    write_tuple: Indice2D  # (req_mapping, seq_lens or -1)


ForwardData: TypeAlias = "Tuple[ForwardInput, ForwardOutput]"


class Scheduler(SchedulerIOMixin):
    def __init__(self, config: SchedulerConfig):
        from minisgl.engine import Engine

        self.engine = Engine(config)

        # use another stream to overlap metadata processing with computation
        self.device = self.engine.device
        self.stream = torch.cuda.Stream(device=self.device)
        self.engine_stream_ctx = torch.cuda.stream(self.engine.stream)
        torch.cuda.set_stream(self.stream)

        # initialize other managers
        self.table_manager = TableManager(config.max_running_req, self.engine.page_table)
        # GDN-hybrid models MUST use the non-radix ("naive") prefix cache: GDN recurrent state
        # is not prefix-cacheable, and a radix hit would report cached_len>0 with no state behind
        # it (silent garbage). Force it here; dense models keep config.cache_type.
        cache_type = config.cache_type
        if self.engine.gdn_state is not None and cache_type != "naive":
            logger.warning_rank0(
                f"GDN-hybrid model: forcing prefix cache 'naive' (was {cache_type!r}); "
                "GDN state is not prefix-cacheable"
            )
            cache_type = "naive"
        self.cache_manager = CacheManager(
            self.engine.num_pages, config.page_size, self.engine.page_table, cache_type
        )
        self.decode_manager = DecodeManager(config.page_size)
        self.prefill_manager = PrefillManager(
            self.cache_manager, self.table_manager, self.decode_manager
        )
        # GDN recurrent-state slot lifecycle — active ONLY for GDN-hybrid models (engine
        # constructs the state cache in 3d). None (inert) for every dense model today, so
        # the dense scheduling path below is unchanged.
        self.gdn_slots = (
            GDNSlotManager(self.engine.gdn_state)
            if self.engine.gdn_state is not None
            else None
        )

        # some alias for easy access
        self.finished_reqs: Set[Req] = set()
        self.tokenizer = load_tokenizer(config.model_path)
        self.eos_token_id = self.tokenizer.eos_token_id
        self.token_pool = self.table_manager.token_pool
        self.prefill_budget = config.max_extend_tokens
        # self.config = config

        # Initialize the I/O mixin
        super().__init__(config, self.engine.tp_cpu_group)

    def run_when_idle(self) -> None:
        """Called when the scheduler is idle to perform background tasks."""
        logger.info_rank0("Scheduler is idle, waiting for new reqs...")
        self.cache_manager.check_integrity()

    def overlap_loop(self, last_data: ForwardData | None) -> ForwardData | None:
        """
        The main loop of overlapping scheduling and execution.

        It will overlap the execution of current batch and processing of last batch's results,
        which can effectively hide CPU latency and improve GPU utilization.
        """
        blocking = not (
            last_data is not None  # don't block if we have a batch to be processed
            or self.prefill_manager.runnable
            or self.decode_manager.runnable
        )
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            with self.engine_stream_ctx:  # run the batch in the engine's stream
                self.engine.stream.wait_stream(self.stream)
                ongoing_data = (forward_input, self._forward(forward_input))

        self._process_last_data(last_data)
        return ongoing_data

    def normal_loop(self) -> None:
        blocking = not (self.prefill_manager.runnable or self.decode_manager.runnable)
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            ongoing_data = (forward_input, self._forward(forward_input))

        self._process_last_data(ongoing_data)

    @torch.inference_mode()
    def run_forever(self) -> NoReturn:
        # Speculative decoding runs in a dedicated synchronous loop: acceptance is a
        # data-dependent host-sync that fundamentally conflicts with the zero-sync overlap path
        # (see SPEC_DECODE.md §1). All GPU work runs on the engine stream, like the eager path.
        if self.engine.spec_config is not None:
            with self.engine_stream_ctx:
                self.engine.stream.wait_stream(self.stream)
                while True:
                    self._spec_loop()
        if ENV.DISABLE_OVERLAP_SCHEDULING:
            with self.engine_stream_ctx:
                self.engine.stream.wait_stream(self.stream)
                while True:
                    self.normal_loop()
        else:
            assert torch.cuda.current_stream() == self.stream
            data = None
            while True:
                data = self.overlap_loop(data)

    def shutdown(self) -> None:
        torch.cuda.synchronize(self.device)
        self.sync_all_ranks()
        self.engine.shutdown()

    def _process_last_data(self, last_data: ForwardData | None) -> None:
        if last_data is None:
            return

        batch, (_, next_tokens_cpu, copy_done) = last_data[0].batch, last_data[1]
        copy_done.synchronize()
        reply: List[DetokenizeMsg] = []
        new_finished_reqs: Set[Req] = set()
        with self.cache_manager.lazy_free_region():
            for i, req in enumerate(batch.reqs):
                if isinstance(req, ChunkedReq):
                    continue
                next_token = next_tokens_cpu[i]
                req.append_host(next_token.unsqueeze(0))
                next_token = int(next_token.item())
                finished = not req.can_decode
                if not req.sampling_params.ignore_eos:
                    finished |= next_token == self.eos_token_id
                reply.append(DetokenizeMsg(uid=req.uid, next_token=next_token, finished=finished))

                # NOTE: overlap scheduling may make the request freed twice, skip second free
                if finished and req not in self.finished_reqs:
                    self.decode_manager.remove_req(req)
                    self._free_req_resources(req)
                    new_finished_reqs.add(req)
                elif batch.is_prefill:  # for prefill, non-chunk req, cache the prefix
                    self.cache_manager.cache_req(req, finished=False)

        self.finished_reqs = new_finished_reqs
        self.send_result(reply)

    def _process_one_msg(self, msg: BaseBackendMsg) -> None:
        if isinstance(msg, BatchBackendMsg):
            for msg in msg.data:
                self._process_one_msg(msg)
        elif isinstance(msg, ExitMsg):
            raise KeyboardInterrupt
        elif isinstance(msg, UserMsg):
            logger.debug_rank0("Received user msg: %s", msg)
            input_len, max_seq_len = len(msg.input_ids), self.engine.max_seq_len
            max_output_len = max_seq_len - input_len
            if max_output_len <= 0:
                return logger.warning_rank0(
                    f"Input sequence length {input_len} exceeds {max_seq_len}, "
                    f"request {msg.uid} is dropped."
                )
            if msg.sampling_params.max_tokens > max_output_len:
                msg.sampling_params.max_tokens = max_output_len
                logger.warning_rank0(
                    f"Adjust max_tokens to {max_output_len} for request {msg.uid}."
                )
            self.prefill_manager.add_one_req(msg)
        elif isinstance(msg, AbortBackendMsg):
            logger.debug_rank0("Aborting request %d", msg.uid)
            req_to_free = self.prefill_manager.abort_req(msg.uid)
            req_to_free = req_to_free or self.decode_manager.abort_req(msg.uid)
            if req_to_free is not None:
                self._free_req_resources(req_to_free)
        else:
            logger.error(f"Unknown message type: {type(msg)}")
            raise NotImplementedError

    def _free_req_resources(self, req: Req) -> None:
        self.table_manager.free(req.table_idx)
        self.cache_manager.cache_req(req, finished=True)
        # Release the GDN state slot (idempotent — overlap scheduling can free a req twice).
        # This single site covers both normal finish (via _process_last_data) and abort.
        if self.gdn_slots is not None:
            self.gdn_slots.free(req.uid)

    def _prepare_batch(self, batch: Batch) -> ForwardInput:
        self.engine.graph_runner.pad_batch(batch)
        self.cache_manager.allocate_paged(batch.reqs)
        batch.positions = _make_positions(batch, self.device)
        input_mapping = _make_input_tuple(batch, self.device)
        write_mapping = _make_write_tuple(batch, self.device)
        batch.out_loc = self.engine.page_table[input_mapping]
        self.engine.attn_backend.prepare_metadata(batch)
        # GDN-hybrid: allocate/reuse a recurrent-state slot per sequence and build the
        # per-batch GDN metadata (cu_seqlens / state_indices / has_initial_state). Inert for
        # dense models (gdn_slots is None). The GDN layers read batch.gdn_metadata in 3d.
        if self.gdn_slots is not None:
            from minisgl.gdn.metadata import build_gdn_metadata

            state_indices = self.gdn_slots.state_indices(batch)
            batch.gdn_metadata = build_gdn_metadata(batch, state_indices, self.device)
        return ForwardInput(
            batch=batch,
            sample_args=self.engine.sampler.prepare(batch),
            input_tuple=input_mapping,
            write_tuple=write_mapping,
        )

    def _schedule_next_batch(self) -> ForwardInput | None:
        # TODO: support other policies: e.g. DECODE first
        batch = (
            self.prefill_manager.schedule_next_batch(self.prefill_budget)
            or self.decode_manager.schedule_next_batch()
        )
        return self._prepare_batch(batch) if batch else None

    def _forward(self, forward_input: ForwardInput) -> ForwardOutput:
        batch, sample_args, input_mapping, output_mapping = forward_input
        batch.input_ids = self.token_pool[input_mapping]
        forward_output = self.engine.forward_batch(batch, sample_args)
        self.token_pool[output_mapping] = forward_output.next_tokens_gpu
        self.decode_manager.filter_reqs(forward_input.batch.reqs)
        return forward_output

    # ===================================================================================
    # Speculative decoding (synchronous loop). See SPEC_DECODE.md for the full design.
    # ===================================================================================

    def _spec_loop(self) -> None:
        """One synchronous spec-decode iteration. Prefill still goes through the normal path;
        only an all-greedy decode batch is replaced by a propose→verify→accept step."""
        blocking = not (self.prefill_manager.runnable or self.decode_manager.runnable)
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        # Prefill takes priority and uses the normal (non-spec) synchronous path.
        batch = self.prefill_manager.schedule_next_batch(self.prefill_budget)
        if batch is not None:
            forward_input = self._prepare_batch(batch)
            self._process_last_data((forward_input, self._forward(forward_input)))
            return

        if not self.decode_manager.runnable:
            return

        reqs = sorted(self.decode_manager.running_reqs, key=lambda req: req.uid)
        if all(req.sampling_params.is_greedy for req in reqs):
            self._spec_decode_step(reqs)
        else:
            # Spec-decode is greedy-only for now (lossless accept). A non-greedy req in the
            # running set falls the whole batch back to a plain synchronous decode step.
            batch = self.decode_manager.schedule_next_batch()
            forward_input = self._prepare_batch(batch)
            self._process_last_data((forward_input, self._forward(forward_input)))

    def _spec_decode_step(self, reqs: List[Req]) -> None:
        spec = self.engine.spec_config
        assert spec is not None
        device = self.device
        page_table = self.engine.page_table

        # --- 1. propose drafts (host n-gram lookup over each req's full token sequence) --------
        # Clamp to remain_len-1 so even a full accept (K_i+1 emitted) stays within the req budget.
        drafts: List[List[int]] = []
        for req in reqs:
            k_i = max(0, min(spec.num_draft, req.remain_len - 1))
            d = (
                propose_ngram(
                    req.input_ids,
                    num_draft=k_i,
                    max_ngram=spec.ngram_max,
                    min_ngram=spec.ngram_min,
                )
                if k_i > 0
                else []
            )
            drafts.append(d)

        # --- 2. stage: extend each req to K_i+1 query tokens; write drafts into the token pool --
        # Confirmed token sits at position c0 (= cached_len); drafts go at c0+1 .. c0+K_i.
        d_rows: List[int] = []
        d_cols: List[int] = []
        d_vals: List[int] = []
        for req, d in zip(reqs, drafts):
            c0 = req.cached_len
            req.device_len = c0 + len(d) + 1  # extend_len = K_i+1
            for j, tok in enumerate(d):
                d_rows.append(req.table_idx)
                d_cols.append(c0 + 1 + j)
                d_vals.append(tok)
        if d_vals:
            self.token_pool[
                torch.tensor(d_rows, dtype=torch.int64, device=device),
                torch.tensor(d_cols, dtype=torch.int64, device=device),
            ] = torch.tensor(d_vals, dtype=self.token_pool.dtype, device=device)

        # --- 3. build the verify batch (phase='decode' -> full per-position logits + extend) ---
        batch = Batch(reqs=reqs, phase="decode")
        batch.padded_reqs = reqs  # eager: no graph padding
        self.cache_manager.allocate_paged(reqs)
        batch.positions = _make_positions(batch, device)
        input_mapping = _make_input_tuple(batch, device)
        batch.out_loc = page_table[input_mapping]
        self.engine.attn_backend.prepare_metadata(batch)
        batch.input_ids = self.token_pool[input_mapping]

        # --- 4. verify forward -> per-position argmax (greedy == sampling here) ----------------
        logits = self.engine.forward_verify(batch)
        preds = logits.argmax(dim=-1).to(torch.int32).cpu()  # [sum(K_i+1)]; this syncs

        # --- 5. accept + commit + rollback per req --------------------------------------------
        offset = 0
        total_emitted = 0
        reply: List[DetokenizeMsg] = []
        new_finished_reqs: Set[Req] = set()
        c_rows: List[int] = []
        c_cols: List[int] = []
        c_vals: List[int] = []
        free_chunks: List[torch.Tensor] = []
        for req, d in zip(reqs, drafts):
            q_len = len(d) + 1
            target = preds[offset : offset + q_len].tolist()
            offset += q_len
            result = verify_greedy(d, target)
            if os.environ.get("MINISGL_SPEC_FORCE_N0") == "1":
                # Diagnostic: stage+verify drafts but accept none (emit only the bonus). Should be
                # byte-identical to plain decode through the multi-query kernel — isolates whether
                # the bug is in the verify forward vs. the accept/commit path.
                result = result._replace(emitted=result.emitted[:1], num_accepted=0)
            c0 = req.cached_len
            old_device_len = c0 + len(d) + 1

            if os.environ.get("MINISGL_SPEC_DEBUG") in ("2", "3"):
                logger.info_rank0(
                    f"[spec-dbg] uid={req.uid} c0={c0} dev={req.device_len} "
                    f"conf={int(req.input_ids[c0])} k={len(d)} n={result.num_accepted} "
                    f"emit={result.emitted}"
                )

            # Decide which emitted tokens to keep, truncating at EOS.
            keep: List[int] = []
            eos = False
            for tok in result.emitted:
                keep.append(tok)
                if (not req.sampling_params.ignore_eos) and tok == self.eos_token_id:
                    eos = True
                    break

            # Commit kept tokens to the host sequence + GPU token pool (positions c0+1 .. c0+len).
            for j, tok in enumerate(keep):
                c_rows.append(req.table_idx)
                c_cols.append(c0 + 1 + j)
                c_vals.append(tok)
            req.input_ids = torch.cat(
                [req.input_ids, torch.tensor(keep, dtype=req.input_ids.dtype)]
            )
            req.cached_len = c0 + len(keep)  # KV valid through cached_len-1
            req.device_len = req.cached_len + 1
            total_emitted += len(keep)
            finished = eos or (not req.can_decode)

            # One message carries all of this step's committed tokens (the detokenizer keys
            # streaming state by uid and assumes one message per uid per batch).
            if keep:
                reply.append(
                    DetokenizeMsg(
                        uid=req.uid,
                        next_token=keep[0],
                        finished=finished,
                        extra_tokens=keep[1:],
                    )
                )

            # Rollback: free the KV slots of staged positions beyond the kept run. Position
            # `cached_len` (the bonus token) is freed too and reallocated next step (its KV was
            # computed for a now-rejected draft). page_size==1 -> slot index == page index.
            if old_device_len > req.cached_len:
                free_chunks.append(page_table[req.table_idx, req.cached_len : old_device_len])

            if finished:
                new_finished_reqs.add(req)

        if c_vals:
            self.token_pool[
                torch.tensor(c_rows, dtype=torch.int64, device=device),
                torch.tensor(c_cols, dtype=torch.int64, device=device),
            ] = torch.tensor(c_vals, dtype=self.token_pool.dtype, device=device)
        if free_chunks:
            self.cache_manager._free(torch.cat(free_chunks))

        for req in new_finished_reqs:
            self.decode_manager.remove_req(req)
            self._free_req_resources(req)
        self.finished_reqs = new_finished_reqs
        self.send_result(reply)
        self._spec_debug(reqs, drafts, total_emitted)

    def _spec_debug(self, reqs: List[Req], drafts: List[List[int]], emitted: int) -> None:
        # MINISGL_SPEC_DEBUG=1: accumulate acceptance stats and log the running mean every 50
        # steps on the primary rank. Inert otherwise (one dict lookup). proposed = sum K_i;
        # accepted drafts = emitted - num_reqs (each req emits 1 bonus + its accepted drafts).
        import os

        if os.environ.get("MINISGL_SPEC_DEBUG") != "1":
            return
        st = getattr(self, "_spec_stats", None)
        if st is None:
            st = self._spec_stats = {"steps": 0, "proposed": 0, "accepted": 0, "emitted": 0, "reqs": 0}
        st["steps"] += 1
        st["proposed"] += sum(len(d) for d in drafts)
        st["accepted"] += emitted - len(reqs)  # accepted drafts (excludes the per-req bonus)
        st["emitted"] += emitted
        st["reqs"] += len(reqs)
        if st["steps"] % 50 == 0:
            acc_rate = st["accepted"] / max(1, st["proposed"])
            toks_per_step = st["emitted"] / max(1, st["steps"])
            logger.info_rank0(
                f"[spec] step={st['steps']} accept_rate={acc_rate:.2f} "
                f"draft_accepted={st['accepted']}/{st['proposed']} "
                f"emitted/step={toks_per_step:.2f} (reqs/step={st['reqs']/st['steps']:.1f})"
            )


def _make_positions(batch: Batch, device: torch.device) -> torch.Tensor:
    needed_size = sum(r.extend_len for r in batch.padded_reqs)
    indices_host = torch.empty(needed_size, dtype=torch.int32, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        torch.arange(
            req.cached_len,
            req.device_len,
            dtype=torch.int32,
            out=indices_host[offset : offset + length],
        )
        offset += length
    return indices_host.to(device, non_blocking=True)


def _make_input_tuple(batch: Batch, device: torch.device) -> Indice2D:
    mapping_host = torch.empty(len(batch.positions), dtype=torch.int64, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        mapping_host[offset : offset + length].fill_(req.table_idx)
        offset += length
    return mapping_host.to(device, non_blocking=True), batch.positions.to(torch.int64)


def _make_write_tuple(batch: Batch, device: torch.device) -> Indice2D:
    mapping_list = [req.table_idx for req in batch.reqs]
    mapping_host = torch.tensor(mapping_list, dtype=torch.int64, pin_memory=True)
    write_list = [(req.device_len if req.can_decode else -1) for req in batch.reqs]
    write_host = torch.tensor(write_list, dtype=torch.int64, pin_memory=True)
    return mapping_host.to(device, non_blocking=True), write_host.to(device, non_blocking=True)
