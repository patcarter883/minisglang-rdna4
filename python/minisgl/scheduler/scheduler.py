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
from minisgl.spec import AcceptResult, ProposeContext, make_proposer, verify_greedy
from minisgl.utils import div_ceil, init_logger, load_tokenizer

from .cache import CacheManager
from .cca_slots import CCASlotManager
from .config import SchedulerConfig
from .decode import DecodeManager
from .ep import SchedulerEPMixin
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


class Scheduler(SchedulerEPMixin, SchedulerIOMixin):
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
        # GDN AND CCA recurrent state are both non-prefix-cacheable: a radix hit would report
        # cached_len>0 with no recurrent state behind it (silent garbage). Force naive for either.
        has_recurrent_state = (
            self.engine.gdn_state is not None or self.engine.cca_state is not None
        )
        if has_recurrent_state and cache_type != "naive":
            logger.warning_rank0(
                f"recurrent-state hybrid model: forcing prefix cache 'naive' (was {cache_type!r}); "
                "GDN/CCA recurrent state is not prefix-cacheable"
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
        # ZAYA CCA recurrent-state slot lifecycle — active ONLY for CCA-hybrid (Zaya) models
        # (engine builds the state cache when is_cca_hybrid). None (inert) for every other model,
        # so the scheduling path below is unchanged.
        self.cca_slots = (
            CCASlotManager(self.engine.cca_state)
            if self.engine.cca_state is not None
            else None
        )

        # some alias for easy access
        self.finished_reqs: Set[Req] = set()
        self.tokenizer = load_tokenizer(config.model_path)
        self.eos_token_id = self.tokenizer.eos_token_id
        self.token_pool = self.table_manager.token_pool
        self.prefill_budget = config.max_extend_tokens
        # self.config = config

        # Speculative-decode proposer (n-gram / MTP / draft-model). None unless spec is enabled.
        self._proposer = (
            make_proposer(self.engine.spec_config, self.engine)
            if self.engine.spec_config is not None
            else None
        )
        # Target hidden-state capture wiring. A draft-head proposer (MTP / EAGLE3 / DFlash) declares
        # `needs_last_hidden` and/or `capture_layer_ids`; the engine then asks forward_verify to
        # return the target's hidden states and feeds them back into ProposeContext for the next
        # propose. A pure n-gram proposer declares neither, so capture stays OFF and a normal serve
        # pays nothing. Aux-layer capture is programmed into the target model ONCE here.
        self._spec_needs_last_hidden = False
        self._spec_capture_layer_ids: List[int] | None = None
        # Prompt-prefill draft-KV seed (MINISGL_SPEC_PREFILL_SEED=1): run a hidden-capturing prefill
        # and seed the proposer's persistent draft KV over the prompt, so the FIRST draft already sees
        # full prompt context (lifts early-token acceptance). Off by default — it adds prefill work and
        # grows the per-step draft-KV stack by the prompt length; lossless either way (the verify
        # corrects every draft, so the seed can only change acceptance, never output). Only engages for
        # a proposer that owns a seedable per-req draft KV (MTP / EAGLE3 -> supports_prefill_seed).
        self._spec_seed_enabled = False
        if self._proposer is not None:
            self._spec_needs_last_hidden = bool(self._proposer.needs_last_hidden)
            self._spec_capture_layer_ids = self._proposer.capture_layer_ids
            if self._spec_capture_layer_ids:
                self.engine.model.set_capture_layers(self._spec_capture_layer_ids)
            self._spec_seed_enabled = (
                os.environ.get("MINISGL_SPEC_PREFILL_SEED") == "1"
                and bool(self._proposer.supports_prefill_seed)
                and (self._spec_needs_last_hidden or bool(self._spec_capture_layer_ids))
            )
            if self._spec_seed_enabled:
                logger.info_rank0("spec-decode: prompt-prefill draft-KV seed ENABLED")
            # Capture the MLA verify CUDA graphs NOW — after the aux-capture layers are programmed
            # above, so the captured forward stashes the hidden/aux the draft head consumes. No-op for
            # non-MLA spec (graph disabled in _adjust_config) or pure n-gram with graphs off. Verify
            # batches are all-running-reqs, so cap the captured sizes at max_running_req.
            needs_hidden = self._spec_needs_last_hidden or bool(self._spec_capture_layer_ids)
            num_aux = len(self._spec_capture_layer_ids) if self._spec_capture_layer_ids else 0
            verify_bs = [b for b in self.engine.graph_runner.graph_bs_list
                         if b <= config.max_running_req]
            self.engine.capture_spec_verify_graphs(needs_hidden, num_aux, verify_bs)
        # uid -> last_hidden / aux_hidden of the verified position carried to the NEXT propose. Empty
        # unless a draft-head proposer requested capture (so n-gram serve allocates nothing).
        self._spec_last_hidden: dict[int, torch.Tensor] = {}
        self._spec_aux_hidden: dict[int, torch.Tensor] = {}

        # Structured-output (constrained decoding) state. Built lazily on the first constrained
        # request, so a plain serve never imports xgrammar. uid -> live GrammarMatcher.
        self._grammar_backend = None
        self._grammar_matchers: dict[int, object] = {}

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
        # Expert parallelism runs a dedicated synchronous lockstep loop: every step issues the MoE
        # all_gather/all_reduce over the DP/EP group, so all replicas must agree the per-step
        # phase+size (one gloo all_reduce(MAX), OUTSIDE the graph) or the collectives deadlock. This
        # conflicts with the zero-sync overlap path, like spec decode. See SchedulerEPMixin.
        if self.engine.enable_ep:
            with self.engine_stream_ctx:
                self.engine.stream.wait_stream(self.stream)
                while True:
                    self.ep_loop()
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
                # Structured output: advance this req's grammar matcher with the committed token so the
                # next step's bitmask reflects the new state. Skip on finish (req is done). A terminated
                # grammar (complete JSON) is allowed to emit EOS, which the matcher won't accept — guard.
                if not finished:
                    m = self._grammar_matchers.get(req.uid)
                    if m is not None and not m.is_terminated():
                        m.accept_token(next_token)
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
        # Release the CCA conv-state slot (idempotent — overlap scheduling can free a req twice).
        if self.cca_slots is not None:
            self.cca_slots.free(req.uid)
        # Release any spec-decode proposer draft state (MTP persistent per-uid KV; n-gram no-op).
        if self._proposer is not None:
            self._proposer.free(req.uid)
        # Release the structured-output grammar matcher (idempotent).
        self._grammar_matchers.pop(req.uid, None)

    def _prepare_batch(self, batch: Batch) -> ForwardInput:
        self.engine.graph_runner.pad_batch(batch)
        return self._finish_prepare(batch)

    def _finish_prepare(self, batch: Batch, skip_alloc: bool = False) -> ForwardInput:
        # The padding decision (batch.padded_reqs) is set by the caller — pad_batch for the DP/normal
        # path, or the EP lockstep mixin (which forces a common-bs / dummy-prefill padding so every
        # replica issues identical-shape MoE collectives). Everything below is padding-agnostic.
        # skip_alloc=True (EP idle-replica dummy prefill ONLY): the batch's lone dummy_req already
        # points its page_table at the reserved null page, so allocating a real KV page for it would
        # leak one every dummy step -> CacheManager integrity-check crash. Skip the allocation and let
        # the dummy write into the reserved page.
        if not skip_alloc:
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
        # CCA-hybrid (Zaya): allocate/reuse a conv-state slot per sequence and build the per-batch
        # CCA metadata (query_start_loc / state_indices / has_initial_state). Inert for non-CCA
        # models (cca_slots is None). The CCA layers read batch.cca_metadata in the model forward.
        if self.cca_slots is not None:
            from minisgl.cca.metadata import build_cca_metadata

            cca_state_indices = self.cca_slots.state_indices(batch)
            batch.cca_metadata = build_cca_metadata(batch, cca_state_indices, self.device)
        sample_args = self.engine.sampler.prepare(batch)
        # Structured output: attach the per-row grammar bitmask (None unless a constrained req is in
        # the batch). The sampler masks disallowed tokens before argmax/sampling. Built over
        # padded_reqs so the row order matches logits[:batch.size]. No-op for a plain serve.
        sample_args.grammar_bitmask = self._build_grammar_bitmask(batch)
        return ForwardInput(
            batch=batch,
            sample_args=sample_args,
            input_tuple=input_mapping,
            write_tuple=write_mapping,
        )

    def _build_grammar_bitmask(self, batch: Batch) -> torch.Tensor | None:
        """Packed xgrammar token bitmask [batch.size, ceil(vocab/32)] on device, or None if no req in
        the batch is constrained. Constrained rows carry the matcher's currently-allowed token set;
        every other row (unconstrained reqs + CUDA-graph dummy padding) is left all-ones (no mask).
        Matchers are created lazily here so a constrained req's FIRST sample (the prefill bonus token)
        is already masked."""
        reqs = batch.padded_reqs
        if not any(r.sampling_params.is_constrained for r in reqs):
            return None
        if self._grammar_backend is None:
            from minisgl.engine.grammar import GrammarBackend

            self._grammar_backend = GrammarBackend(self.tokenizer, self.engine.sampler.vocab_size)
        backend = self._grammar_backend
        bitmask = backend.allocate_bitmask(len(reqs))
        bitmask.fill_(-1)  # all-ones default => unconstrained / dummy rows allow every token
        for i, r in enumerate(reqs):
            if not r.sampling_params.is_constrained:
                continue
            m = self._grammar_matchers.get(r.uid)
            if m is None:
                m = self._grammar_matchers[r.uid] = backend.make_matcher(r.sampling_params.grammar)
            if not m.is_terminated():
                m.fill_next_token_bitmask(bitmask, i)
            # a terminated matcher leaves its row all-ones: nothing left to emit but EOS, allow it.
        return bitmask.to(self.device)

    def _verify_greedy_constrained(
        self, matcher, draft: List[int], logits_block: torch.Tensor, ignore_eos: bool
    ) -> AcceptResult:
        """Grammar-constrained speculative acceptance for ONE request (structured output + spec decode).

        The proposer drafts UNCONSTRAINED; the grammar is enforced here, at the verify argmax. Walk the
        K+1 verify positions: at each, mask the target logits with the matcher's currently-allowed set
        (its state after the committed prefix), take the masked argmax as the grammar-valid target token
        ``t_i``, and accept ``draft[i]`` iff it equals ``t_i``. The committed run is exactly the masked
        target argmaxes (== what plain constrained decode would emit -> LOSSLESS), so a draft that
        violates the grammar simply mismatches ``t_i`` and is rejected. The matcher is advanced by every
        committed token (never by EOS, which isn't a grammar token); the walk stops at the bonus /
        first mismatch / EOS, so it never over-advances past what the caller keeps. ``logits_block`` is
        the req's [K+1, vocab] verify logits on CPU."""
        from minisgl.engine.grammar import apply_token_bitmask

        backend = self._grammar_backend
        emitted: List[int] = []
        n_acc = 0
        K = len(draft)
        for i in range(K + 1):
            bitmask = backend.allocate_bitmask(1)
            bitmask.fill_(-1)
            terminated = matcher.is_terminated()
            if not terminated:
                matcher.fill_next_token_bitmask(bitmask, 0)
            masked = apply_token_bitmask(logits_block[i : i + 1].float(), bitmask)
            t_i = int(masked[0].argmax().item())
            emitted.append(t_i)
            is_eos = (not ignore_eos) and t_i == self.eos_token_id
            if not is_eos and not terminated:
                matcher.accept_token(t_i)
            if is_eos:
                break  # EOS ends the sequence; not fed to the matcher
            if i < K and draft[i] == t_i:
                n_acc += 1
                continue
            break  # bonus token (t_i != draft[i], or i == K)
        return AcceptResult(emitted=emitted, num_accepted=n_acc)

    def _schedule_next_batch(self) -> ForwardInput | None:
        # TODO: support other policies: e.g. DECODE first
        batch = (
            self.prefill_manager.schedule_next_batch(self.prefill_budget)
            or self.decode_manager.schedule_next_batch()
        )
        return self._prepare_batch(batch) if batch else None

    def _forward(self, forward_input: ForwardInput, track_reqs: bool = True) -> ForwardOutput:
        batch, sample_args, input_mapping, output_mapping = forward_input
        batch.input_ids = self.token_pool[input_mapping]
        forward_output = self.engine.forward_batch(batch, sample_args)
        self.token_pool[output_mapping] = forward_output.next_tokens_gpu
        # track_reqs=False for an EP lockstep DUMMY batch (no real reqs): its dummy_req must NOT be
        # promoted into the decode running set (it would pollute every subsequent decode step).
        if track_reqs:
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

        # Prefill takes priority and uses the normal (non-spec) synchronous path — except when the
        # prompt-prefill draft-KV seed is enabled, where it runs a hidden-capturing prefill instead.
        batch = self.prefill_manager.schedule_next_batch(self.prefill_budget)
        if batch is not None:
            # A constrained (structured-output) req must take the plain prefill path so its first token
            # (the prefill bonus) is grammar-masked — the hidden-capturing seed forward bypasses the
            # sampler's bitmask. It also isn't a spec req, so seeding its draft KV is pointless.
            constrained = any(r.sampling_params.is_constrained for r in batch.reqs)
            if self._spec_seed_enabled and not constrained:
                self._spec_prefill_seeded(batch)
            else:
                forward_input = self._prepare_batch(batch)
                self._process_last_data((forward_input, self._forward(forward_input)))
            return

        if not self.decode_manager.runnable:
            return

        reqs = sorted(self.decode_manager.running_reqs, key=lambda req: req.uid)
        # Spec-decode runs for an all-greedy decode set (lossless accept is greedy-only). Constrained
        # (structured-output) reqs now spec-decode too: their drafts are proposed UNCONSTRAINED and the
        # grammar is enforced at the verify argmax (_verify_greedy_constrained), so a grammar-violating
        # draft is simply rejected. A non-greedy req still falls the whole batch back to a plain decode.
        spec_ok = all(req.sampling_params.is_greedy for req in reqs)
        if spec_ok:
            self._spec_decode_step(reqs)
        else:
            batch = self.decode_manager.schedule_next_batch()
            forward_input = self._prepare_batch(batch)
            self._process_last_data((forward_input, self._forward(forward_input)))

    def _spec_prefill_seeded(self, batch: Batch) -> None:
        """Prefill forward that ALSO captures the per-token target hidden over the prompt and seeds the
        proposer's persistent draft KV from it (the prompt-prefill draft-KV seed lever). Mirrors the
        normal prefill (`_forward` + `_process_last_data`) but runs the one forward with
        `return_hidden=True`, so the bonus token and the prompt hidden come from a SINGLE pass. Only
        reqs whose WHOLE prompt is in this forward (cached_len==0, not chunked) are seeded — a chunked
        or prefix-cache-hit prompt's earlier hidden isn't available here, so it falls back to the cold
        cache (still lossless, just no early-token lift)."""
        forward_input = self._prepare_batch(batch)
        # Plan the seed-eligible reqs + their hidden-row slices BEFORE the forward advances cached_len
        # (forward_batch calls complete_one). Prefill rows follow padded_reqs order, which equals
        # batch.reqs for a prefill (no CUDA-graph padding — see GraphRunner.pad_batch).
        real = {id(r) for r in batch.reqs}
        plan: List[Tuple[Req, int, int]] = []  # (req, hidden-row offset, prompt_len)
        offset = 0
        for req in batch.padded_reqs:
            ext = req.extend_len
            if (not isinstance(req, ChunkedReq)) and req.cached_len == 0 and id(req) in real:
                plan.append((req, offset, ext))
            offset += ext

        fi_batch, sample_args, input_mapping, output_mapping = forward_input
        fi_batch.input_ids = self.token_pool[input_mapping]
        out, last_hidden, aux_hidden = self.engine.forward_batch(
            fi_batch, sample_args, return_hidden=True
        )
        self.token_pool[output_mapping] = out.next_tokens_gpu
        self.decode_manager.filter_reqs(fi_batch.reqs)

        # Seed each eligible req's draft KV over its prompt slice, and carry the first-step seed hidden
        # (the prompt's LAST position produced the bonus token, so its hidden seeds the first propose —
        # exactly the row the decode-time carry would have used). Cloned off the big verify tensors so
        # they can be released. last_hidden/aux_hidden are still the prompt's hidden here (input_ids
        # has not yet had the bonus appended — that happens in _process_last_data below).
        for req, off, plen in plan:
            lh = last_hidden[off : off + plen] if last_hidden is not None else None
            ax = aux_hidden[:, off : off + plen] if aux_hidden is not None else None
            self._proposer.seed_prefill(req, lh, ax)
            if self._spec_needs_last_hidden and lh is not None:
                self._spec_last_hidden[req.uid] = lh[plen - 1].clone()
            if self._spec_capture_layer_ids and ax is not None:
                self._spec_aux_hidden[req.uid] = ax[:, plen - 1].clone()

        self._process_last_data((forward_input, out))

    def _spec_decode_step(self, reqs: List[Req]) -> None:
        spec = self.engine.spec_config
        assert spec is not None and self._proposer is not None
        device = self.device
        page_table = self.engine.page_table
        # MINISGL_SPEC_TIMING=1: per-phase wall-clock (propose / verify-forward / accept) to attribute
        # the synchronous spec step. Adds cuda.synchronize barriers, so DIAGNOSTIC-only.
        _timing = os.environ.get("MINISGL_SPEC_TIMING") == "1"
        if _timing:
            import time as _time
            torch.cuda.synchronize(device); _t0 = _time.perf_counter()

        # --- 1. propose drafts (proposer-specific: n-gram lookup / MTP head / draft model). The
        # proposer clamps per-req to the remaining budget; an empty list ⇒ plain decode for that req.
        # Draft-head proposers read the target hidden states captured at the PREVIOUS verify (keyed
        # by uid); n-gram declared neither, so these dicts are empty and ProposeContext is bare.
        capture = self._spec_needs_last_hidden or bool(self._spec_capture_layer_ids)
        ctx = ProposeContext(
            device,
            last_hidden={r.uid: self._spec_last_hidden[r.uid]
                         for r in reqs if r.uid in self._spec_last_hidden}
            if self._spec_needs_last_hidden else None,
            aux_hidden={r.uid: self._spec_aux_hidden[r.uid]
                        for r in reqs if r.uid in self._spec_aux_hidden}
            if self._spec_capture_layer_ids else None,
        )
        drafts = self._proposer.propose(reqs, spec.num_draft, ctx)
        if _timing:
            torch.cuda.synchronize(device); _t1 = _time.perf_counter()

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
        batch.spec_verify = True  # multi-token; GDN/lm-head treat it like a prefill (see core.py)
        self.cache_manager.allocate_paged(reqs)
        # MLA verify CUDA graph: when every req drafted exactly num_draft (uniform K+1 query tokens)
        # and the batch fits a captured size, PAD with dummy verify reqs FIRST so positions/out_loc
        # cover the padded rows (dummy rows point at the dummy page — never stale real KV), then replay
        # the graph. Otherwise (partial-K step / graphs off) stay eager with dynamic metadata.
        use_vgraph = self.engine.graph_runner.can_use_verify_graph(batch)
        if use_vgraph:
            self.engine.graph_runner.pad_verify(batch)
        else:
            batch.padded_reqs = reqs
        batch.positions = _make_positions(batch, device)
        input_mapping = _make_input_tuple(batch, device)
        batch.out_loc = page_table[input_mapping]
        if not use_vgraph:
            self.engine.attn_backend.prepare_metadata(batch)
        batch.input_ids = self.token_pool[input_mapping]

        # GDN-hybrid: build the per-batch recurrent metadata (varlen, like a prefill — see
        # build_gdn_metadata + the spec_verify dispatch). Instead of snapshotting + re-advancing, the
        # verify forward CAPTURES the conv+ssm state AFTER each of the K+1 tokens into per-layer
        # scratch (gdn_prefill_verify / causal_conv1d_fwd_verify). After acceptance the scheduler
        # installs the state after the accepted prefix (index = committed-1) directly into the slot —
        # no 2x re-advance, and bit-exact vs 1-token decode. See SPEC_DECODE.md.
        gdn_state_indices = None
        if self.gdn_slots is not None:
            from minisgl.gdn.metadata import build_gdn_metadata

            gdn_state_indices = self.gdn_slots.state_indices(batch)
            batch.gdn_metadata = build_gdn_metadata(batch, gdn_state_indices, device)
            batch.gdn_metadata.capture_verify_state = True
            batch.gdn_metadata.verify_max_qlen = max(len(d) + 1 for d in drafts)

        # --- 4. verify forward -> per-position argmax (greedy == sampling here) ----------------
        # Draft-head proposers also need the target's hidden states at the verified positions; the
        # engine returns them from the SAME forward (no extra pass). last_hidden [T, hidden], aux
        # [num_capture_layers, T, hidden] or None; T = sum(K_i+1). Stays on-device until we slice the
        # per-uid seed rows after acceptance (then drop the full tensors).
        last_hidden = aux_hidden = None
        if _timing:
            _t_stage = _time.perf_counter()  # CPU-side staging (steps 2-3) done; forward next
        if capture:
            logits, last_hidden, aux_hidden = self.engine.forward_verify(batch, return_hidden=True)
        else:
            logits = self.engine.forward_verify(batch)
        preds = logits.argmax(dim=-1).to(torch.int32).cpu()  # [sum(K_i+1)]; this syncs
        if _timing:
            _t2 = _time.perf_counter()  # forward already synced by .cpu()

        # --- 5. accept + commit + rollback per req --------------------------------------------
        offset = 0
        total_emitted = 0
        reply: List[DetokenizeMsg] = []
        new_finished_reqs: Set[Req] = set()
        c_rows: List[int] = []
        c_cols: List[int] = []
        c_vals: List[int] = []
        free_chunks: List[torch.Tensor] = []
        accepted_counts: List[int] = []  # drafts accepted per req (drives proposer draft-state rollback)
        # GDN per-token-state install bookkeeping. For each still-running seq, record its BATCH index
        # (to gather the captured scratch) and the per-token index to install (= committed-1, the
        # state AFTER the last committed token). A finished seq frees its slot, so its state is moot.
        gdn_install_batch_idx: List[int] = []
        gdn_install_t_index: List[int] = []
        # Fresh per-uid target hidden seeds for the NEXT step's propose (draft-head proposers only).
        new_last_hidden: dict[int, torch.Tensor] = {}
        new_aux_hidden: dict[int, torch.Tensor] = {}
        for i, (req, d) in enumerate(zip(reqs, drafts)):
            q_len = len(d) + 1
            block_start = offset  # this req's first query row in the [sum(K_i+1)] verify output
            offset += q_len
            matcher = (
                self._grammar_matchers.get(req.uid)
                if req.sampling_params.is_constrained
                else None
            )
            if matcher is not None:
                # Structured output + spec: grammar-mask the verify argmax per position and advance the
                # matcher through the accepted chain (lossless; see _verify_greedy_constrained). Needs
                # the raw logit rows on host, not the precomputed unmasked argmax.
                block = logits[block_start : block_start + q_len].float().cpu()
                result = self._verify_greedy_constrained(
                    matcher, d, block, req.sampling_params.ignore_eos
                )
            else:
                target = preds[block_start : block_start + q_len].tolist()
                result = verify_greedy(d, target)
            if os.environ.get("MINISGL_SPEC_FORCE_N0") == "1":
                # Diagnostic: stage+verify drafts but accept none (emit only the bonus). Should be
                # byte-identical to plain decode through the multi-query kernel — isolates whether
                # the bug is in the verify forward vs. the accept/commit path.
                result = result._replace(emitted=result.emitted[:1], num_accepted=0)
            accepted_counts.append(result.num_accepted)
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
            # GDN: a still-running req's recurrent state must end after the accepted prefix. The verify
            # captured the state after each token; install scratch index committed-1 (state after the
            # last committed token). committed == len(keep) >= 1 (always >= the 1 bonus token) for a
            # non-EOS-truncated, still-running seq. Finished reqs free their slot, so state is moot.
            if gdn_state_indices is not None and not finished:
                gdn_install_batch_idx.append(i)
                gdn_install_t_index.append(len(keep) - 1)

            # Draft-head seed: the target hidden at the row that PRODUCED the last committed token
            # (block_start + len(keep)-1 — same index the GDN install uses). The draft for the next
            # step is seeded from the accepted token's hidden state. Cloned off the big verify tensor
            # so the per-step [T, hidden] / [L, T, hidden] outputs can be released. Finished reqs are
            # gone next step, so we skip them.
            if capture and not finished and keep:
                row = block_start + len(keep) - 1
                if last_hidden is not None:
                    new_last_hidden[req.uid] = last_hidden[row].clone()
                if aux_hidden is not None:
                    new_aux_hidden[req.uid] = aux_hidden[:, row].clone()

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

            # Rollback: free WHOLE pages allocated for this step that fall entirely beyond the kept
            # run (page-size-aware: page_size 1 for MHA frees per token; 16 for MLA frees per page).
            # A partial page straddling cached_len stays (it holds valid KV); its rejected-draft tail
            # is overwritten as the seq grows back. The bonus token's slot is recomputed next step.
            ps = self.cache_manager.page_size
            free_start = div_ceil(req.cached_len, ps) * ps
            free_end = div_ceil(old_device_len, ps) * ps
            if free_end > free_start:
                # contiguous page-aligned token range; cache_manager._free strides by page_size.
                free_chunks.append(page_table[req.table_idx, free_start:free_end])

            if finished:
                new_finished_reqs.add(req)

        if c_vals:
            self.token_pool[
                torch.tensor(c_rows, dtype=torch.int64, device=device),
                torch.tensor(c_cols, dtype=torch.int64, device=device),
            ] = torch.tensor(c_vals, dtype=self.token_pool.dtype, device=device)
        if free_chunks:
            self.cache_manager._free(torch.cat(free_chunks))

        # GDN: install the captured accepted-prefix recurrent state into each still-running seq's slot.
        # The verify forward (gdn_prefill_verify) already computed the EXACT per-token conv+ssm state,
        # so this is a pure gather — no re-advance, no second forward. Bit-exact vs 1-token decode.
        if gdn_state_indices is not None and gdn_install_batch_idx:
            md = batch.gdn_metadata
            sel = torch.tensor(gdn_install_batch_idx, dtype=torch.long, device=device)
            slots = gdn_state_indices.to(torch.long)[sel]
            t_index = torch.tensor(gdn_install_t_index, dtype=torch.long, device=device)
            self.engine.gdn_state.install_verify_state(
                md.conv_scratch, md.ssm_scratch, slots, t_index
            )

        # Carry the fresh target hidden seeds to the next propose (replaces the consumed step's seeds
        # for these uids; finished uids drop out because they aren't in the fresh dict). No-op for
        # n-gram (capture False ⇒ both dicts stay empty).
        if capture:
            for uid in [r.uid for r in reqs]:
                self._spec_last_hidden.pop(uid, None)
                self._spec_aux_hidden.pop(uid, None)
            self._spec_last_hidden.update(new_last_hidden)
            self._spec_aux_hidden.update(new_aux_hidden)

        # Roll back any draft-owned state (draft KV / recurrent) to the accepted prefix. No-op for
        # n-gram; MTP/DFlash/EAGLE truncate their draft KV. (GDN backbone-state rollback is handled
        # separately in the verify forward path, not here — it is the target's state, not the draft's.)
        self._proposer.on_accept(reqs, accepted_counts)

        for req in new_finished_reqs:
            self.decode_manager.remove_req(req)
            self._free_req_resources(req)
        self.finished_reqs = new_finished_reqs
        self.send_result(reply)
        if _timing:
            torch.cuda.synchronize(device); _t3 = _time.perf_counter()
            ph = getattr(self, "_spec_phase", None)
            if ph is None:
                ph = self._spec_phase = {
                    "propose": 0.0, "stage": 0.0, "forward": 0.0, "accept": 0.0, "n": 0
                }
            ph["propose"] += _t1 - _t0
            ph["stage"] += _t_stage - _t1  # CPU-side verify-batch staging (steps 2-3)
            ph["forward"] += _t2 - _t_stage  # verify forward + preds.cpu() sync
            ph["accept"] += _t3 - _t2
            ph["n"] += 1
            if ph["n"] % 50 == 0:
                n = ph["n"]
                logger.info_rank0(
                    f"[spec-timing] step={n} propose={ph['propose']/n*1e3:.1f}ms "
                    f"stage={ph['stage']/n*1e3:.1f}ms forward={ph['forward']/n*1e3:.1f}ms "
                    f"accept={ph['accept']/n*1e3:.1f}ms "
                    f"total={(ph['propose']+ph['stage']+ph['forward']+ph['accept'])/n*1e3:.1f}ms"
                )
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
            vinfo = getattr(self.engine.graph_runner, "_verify", None)
            vreplays = vinfo.get("replays", 0) if vinfo else 0
            logger.info_rank0(
                f"[spec] step={st['steps']} accept_rate={acc_rate:.2f} "
                f"draft_accepted={st['accepted']}/{st['proposed']} "
                f"emitted/step={toks_per_step:.2f} (reqs/step={st['reqs']/st['steps']:.1f}) "
                f"verify_graph_replays={vreplays}"
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
