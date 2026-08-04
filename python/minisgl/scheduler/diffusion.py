"""Block-diffusion execution mode: the canvas slot manager and the scheduler loop.

A block-diffusion request does not emit one token per forward. It cycles

    ENCODE (1 causal forward, writes KV)
      -> DENOISE (<=max_denoising_steps NON-CAUSAL forwards over a fixed canvas, emits nothing)
      -> COMMIT (the whole argmax canvas at once)
      -> ENCODE the committed block ... -> next block

which is a genuinely different shape from `PrefillManager -> DecodeManager -> finished`, so it lives
BESIDE the autoregressive loop rather than inside it: `_diffusion_loop` is a peer of `_spec_loop`,
selected once at boot from `ModelConfig.is_block_diffusion`. Nothing in the normal / overlap / spec /
EP loops changes, and a model without a `canvas_length` never constructs a `CanvasManager`.

THE ONE IDEA THAT MAKES THIS CHEAP: the canvas slots and the committed block's slots are the SAME
slots. A block allocates `canvas_length` KV slots once, the decoder overwrites them bidirectionally
on every denoising step, and at commit the encoder overwrites them ONE more time with the causal K/V
the next block will read. No scratch pool, no per-step allocation, and the request's page-table row
stays an ordinary monotone prefix that the radix cache can hold.

Four traps, all of which produce fluent-but-wrong output rather than an error:

  1. THE ENCODER PASS MUST NOT SAMPLE. It is a plain causal forward whose logits are discarded — the
     reference never reads them (`generate` keeps only `encoder_outputs.past_key_values`). Running it
     through `forward_batch` would sample a token, append it to the request and emit it.
  2. `cached_len` / `device_len` MUST NOT MOVE ACROSS DENOISING STEPS. The canvas is scratch; the
     same slots are rewritten every step. Advancing them would allocate a fresh canvas per step and
     leave the block attending its own denoising history.
  3. THE EMITTED TOKENS ARE THE ARGMAX CANVAS, NOT THE SAMPLED ONE that goes into the next forward.
  4. THE COMMITTED BLOCK MUST BE RE-ENCODED CAUSALLY before the next block reads it. The KV sitting
     in those slots is the decoder's BIDIRECTIONAL K/V; the model never serves from that.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Dict, List, Tuple

import torch
from minisgl.core import Batch, Req
from minisgl.diffusion import CanvasState, DiffusionSamplerConfig
from minisgl.message import DetokenizeMsg
from minisgl.utils import div_ceil, init_logger

from .prefill import ChunkedReq

if TYPE_CHECKING:
    pass

logger = init_logger(__name__)


class CanvasManager:
    """Per-request denoising state, keyed by uid — the only stable identity a request has.

    Modelled on `RecurrentSlotManager` and deliberately the same discipline: allocate lazily at the
    start of a block, hold for the block's whole life, free IDEMPOTENTLY (a request can be released
    from several paths — normal finish, abort, canvas commit — and a second free must be a no-op,
    not a KeyError).

    It owns only the DENOISING state. The KV slots are ordinary page-table slots owned by the
    CacheManager, because that is what lets a committed block become a normal radix-cacheable prefix
    instead of a special region."""

    def __init__(self, config: DiffusionSamplerConfig, device: torch.device) -> None:
        self.config = config
        self.device = device
        self._state: Dict[int, CanvasState] = {}
        # Per-request RNG, allocated only for a seeded request and held across that request's blocks
        # (a per-BLOCK generator would replay one block's noise in the next). Freed with the request.
        self._gen: Dict[int, torch.Generator] = {}
        # Denoising steps actually spent per committed block. This is `k` — the number the whole
        # cost argument for block diffusion turns on, and the one an isolated kernel bench cannot
        # produce. Kept here rather than in a metric so it survives a serve with metrics off.
        self.steps_per_block: List[int] = []

    def get(self, uid: int) -> "CanvasState | None":
        return self._state.get(uid)

    def begin(self, uid: int, seed: "int | None" = None) -> CanvasState:
        """Open a fresh block: a canvas of uniform-random ids over the WHOLE vocabulary (this
        architecture has no mask token), a null self-conditioning signal, and the step counter at
        max_denoising_steps counting down.

        `seed` (SamplingParams.seed, None by default) installs a per-REQUEST generator, held across
        the request's blocks so block 2 does not replay block 1's noise. It is the only way to
        reproduce a block-diffusion generation: this path has no greedy mode — the canvas is drawn
        from noise and every step draws a multinomial — so `temperature 0` pins nothing, and without
        a seed two identical requests to one serve return different text. Unseeded requests keep the
        process RNG, i.e. exactly the previous behaviour."""
        gen = self._gen.get(uid)
        if gen is None and seed is not None:
            gen = torch.Generator(device=self.device)
            gen.manual_seed(int(seed))
            self._gen[uid] = gen
        state = CanvasState(self.config, self.device, generator=gen)
        self._state[uid] = state
        return state

    def end(self, uid: int) -> None:
        """Close a block, recording how many steps it took. Not the same as `free`: a request that
        continues to another block ends this block and begins the next."""
        state = self._state.pop(uid, None)
        if state is not None:
            self.steps_per_block.append(self.config.max_denoising_steps - state.step_index)

    def free(self, uid: int) -> None:
        """Release a request entirely (finish/abort). Idempotent by construction."""
        self._state.pop(uid, None)
        self._gen.pop(uid, None)

    @property
    def num_active(self) -> int:
        return len(self._state)


class SchedulerDiffusionMixin:
    """`_diffusion_loop` and the canvas step. Mixed into Scheduler; every attribute it touches
    (`engine`, `cache_manager`, `token_pool`, the managers) is set up by `Scheduler.__init__`."""

    # ---------------------------------------------------------------------------------------
    def _diffusion_loop(self) -> None:
        """One synchronous block-diffusion iteration.

        Synchronous by necessity, exactly like `_spec_loop`: whether a block stops early is a
        data-dependent host sync (the stability + confidence criteria read the entropies back), which
        is fundamentally incompatible with the zero-sync overlap path."""
        blocking = not (self.prefill_manager.runnable or self.decode_manager.runnable)
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        # Prompt prefill takes priority, and it is an ENCODER pass — see `_canvas_encode`.
        batch = self.prefill_manager.schedule_next_batch(self._prefill_budget_now())
        if batch is not None:
            self._canvas_encode(batch)
            return

        if not self.decode_manager.runnable:
            return
        self._canvas_step(self.decode_manager.ordered_reqs)

    # ---------------------------------------------------------------------------------------
    def _canvas_forward(self, batch: Batch, *, allocate: bool) -> torch.Tensor:
        """Build the device-side batch state and run a NON-SAMPLING forward, returning raw logits.

        Shared by the encoder pass and the block re-encode. `allocate=False` is for a forward over
        slots this request already owns (the re-encode writes over the canvas's own slots), where a
        second `allocate_paged` would hand out a fresh page range and leak the old one — the manager
        is deliberately not idempotent."""
        device = self.device
        if allocate:
            self.cache_manager.allocate_paged(batch.reqs)
        batch.padded_reqs = batch.reqs
        batch.positions = _make_positions(batch, device)
        input_mapping = _make_input_tuple(batch, device)
        batch.out_loc = self.engine.page_table[input_mapping]
        self.engine.attn_backend.prepare_metadata(batch)
        # SWA-radix RESTORE, the canvas path's equivalent of the `_finish_prepare` gate the AR loop
        # runs. A prompt encoder pass is an ordinary prefill and CAN hit a page-aligned window
        # snapshot; without this the sliding layers would extend across a boundary whose window is
        # whatever the recycled ring block last held — a silent stale window, and the reason the
        # canvas serve shipped with SWA-radix switched off. Restricted to `is_prefill`, so the
        # re-encode (phase="decode") is skipped: its ring already holds its own window.
        #
        # MEASURED (docs/…BLOCK_DIFFUSION.md §D5): a FULL prefix hit is byte-identical to a cold
        # prefill, 4/4 cells, at 192 and 3184 reused tokens. A PARTIAL hit (shared prefix, differing
        # tail) DIVERGED from its cold reference on a 3168-token prefix and the cause is NOT isolated
        # — a canvas amplifies a 1-ULP prefill-shape difference far more than an AR decode does,
        # because the entropy bound sorts 256 entropies and thresholds a cumulative sum, so one
        # near-tie rewrites the whole block. Do not read the full-hit result as covering partial hits.
        if self._swa_radix and batch.is_prefill:
            self._restore_swa_states(batch)
        batch.input_ids = self.token_pool[input_mapping]
        # forward_verify is the engine's "no sampling, no complete_one" entry point. That contract is
        # exactly what an encoder pass needs: it writes the KV cache and its logits are discarded
        # (the reference's `generate` keeps only `encoder_outputs.past_key_values`).
        return self.engine.forward_verify(batch)

    def _canvas_encode(self, batch: Batch) -> None:
        """The prompt encoder pass: an ordinary causal prefill that writes the KV cache and emits
        NOTHING. The request then enters the decode manager with `cached_len == device_len`, which is
        the state a block starts from."""
        reqs = batch.reqs
        for req in reqs:
            if req.sampling_params.is_constrained:
                raise NotImplementedError(
                    "structured output is not supported for block diffusion: a grammar constrains "
                    "one token at a time against a causal prefix, and a canvas resamples all "
                    f"{self._canvas_cfg.canvas_length} positions every step. Refusing rather than "
                    "silently ignoring the grammar."
                )
        self._canvas_forward(batch, allocate=True)
        for req in reqs:
            # A CHUNK of a multi-forward encoder pass. This is not the "prompt longer than
            # --max-extend-tokens" case it was originally refused as, and refusing it is what made
            # SWA-radix unusable here: the snapshot-capable radix splits EVERY prefill at the last
            # page boundary (PrefillAdder._add_one_req, `is_recurrent_radix`) so a window snapshot
            # lands page-aligned — so a prompt whose length is not a multiple of page_size, i.e. 15
            # out of 16 prompts, arrives as [aligned body][sub-page tail] and the body is a
            # ChunkedReq. The body writes KV and nothing else: advance it to its own chunk end (the
            # AR path gets this from forward_batch's complete_one, which forward_verify deliberately
            # does not do), stash the page-aligned window so the tail's commit can attach it, and
            # leave it out of the cache and the decode manager. `can_decode` is False for a chunk, so
            # filter_reqs below already excludes it. The canvas still starts only from the COMPLETE
            # prefix, which is what the original refusal was protecting.
            if isinstance(req, ChunkedReq):
                req.cached_len = req.device_len
                if self._swa_radix:
                    self._stash_swa_state(req)
                continue
            # The whole prompt is now valid KV. No `complete_one()`: that would advance device_len by
            # the one autoregressive token this model does not produce.
            req.cached_len = req.device_len
            inserted = self.cache_manager.cache_req(req, finished=False)
            # SWA-radix CAPTURE (the AR loop does this in `_process_last_data`, which the canvas loop
            # does not go through). The prompt's sliding window is snapshotted onto the node just
            # inserted, so the NEXT request sharing this prompt can restore it instead of
            # re-encoding. Without it the only snapshots a canvas serve ever produced were the
            # finish-commit ones, i.e. whole completed conversations.
            if self._swa_radix:
                self._maybe_capture_swa_state(req, inserted)
        self.decode_manager.filter_reqs(reqs)

    # ---------------------------------------------------------------------------------------
    def _canvas_step(self, reqs: List[Req]) -> None:
        """One denoising step for every in-flight block, then commit whichever blocks finished."""
        device = self.device
        cfg = self._canvas_cfg
        L = cfg.canvas_length

        states: List[CanvasState] = []
        for req in reqs:
            state = self.canvas_slots.get(req.uid)
            if state is None:
                state = self.canvas_slots.begin(req.uid, req.sampling_params.seed)
                # ONE allocation per block. `device_len` is set BEFORE allocate_paged because that is
                # what it reads, and it then stays put for every step of the block (trap 2).
                req.device_len = req.cached_len + L
                self.cache_manager.allocate_paged([req])
            else:
                assert req.device_len == req.cached_len + L, (
                    f"canvas request {req.uid} lost its block extent "
                    f"({req.cached_len} -> {req.device_len}, expected +{L})"
                )
            states.append(state)

        # --- stage the current canvases into the token pool ---------------------------------
        # This is the spec path's staging scatter with a whole canvas instead of a draft block: the
        # ids have to be in the pool because `batch.input_ids` (and hence the KV scatter target) is
        # gathered from it by the shared `_make_input_tuple` mapping.
        rows = torch.cat(
            [torch.full((L,), r.table_idx, dtype=torch.int64, device=device) for r in reqs]
        )
        cols = torch.cat(
            [torch.arange(r.cached_len, r.cached_len + L, device=device) for r in reqs]
        )
        self.token_pool[rows, cols] = torch.cat([s.canvas for s in states]).to(
            self.token_pool.dtype
        )

        # --- the canvas batch ----------------------------------------------------------------
        batch = Batch(reqs=reqs, phase="decode")
        batch.canvas = True  # -> causal=0 on both geometries + the [window | canvas] ring rows
        batch.padded_reqs = reqs
        batch.positions = _make_positions(batch, device)
        input_mapping = _make_input_tuple(batch, device)
        batch.out_loc = self.engine.page_table[input_mapping]
        self.engine.attn_backend.prepare_metadata(batch)
        batch.input_ids = self.token_pool[input_mapping]

        # Self-conditioning: the PREVIOUS step's soft embedding, [L, hidden] per request. A request
        # on the first step of its block contributes zeros, which is EXACTLY what the reference does
        # (`soft_embeddings = torch.zeros_like(inputs_embeds)`) — so a batch mixing fresh and
        # mid-block requests is handled by concatenation, not by a per-request branch. All-fresh
        # skips the block entirely (see DiffusionGemmaSelfConditioning.forward).
        if all(s.soft_conditioning is None for s in states):
            self_conditioning = None
        else:
            hidden = self.engine.model.model.embed_tokens.weight.shape[1]
            self_conditioning = torch.cat([
                s.soft_conditioning
                if s.soft_conditioning is not None
                else batch.input_ids.new_zeros(
                    (L, hidden), dtype=self.engine.model.model.embed_tokens.weight.dtype
                )
                for s in states
            ])

        logits = self.engine.forward_canvas(batch, batch.input_ids, self_conditioning)

        # --- advance each request's denoising state ------------------------------------------
        reply: List[DetokenizeMsg] = []
        finished_now = set()
        for i, (req, state) in enumerate(zip(reqs, states)):
            out = state.step(logits[i * L : (i + 1) * L])
            # The soft embedding is built HERE, from the temperature-scaled logits the sampler just
            # consumed, so the [L, 262144] fp32 tensor dies with this iteration instead of being
            # carried across the step boundary (see DiffusionGemmaForBlockDiffusion.soft_embedding).
            state.soft_conditioning = self.engine.model.soft_embedding(out.scaled)
            out.scaled = None
            if state.finished:
                self._canvas_commit(req, state, reply, finished_now)

        if finished_now:
            for req in finished_now:
                self.decode_manager.remove_req(req)
                self._free_req_resources(req)
            self.finished_reqs = finished_now
        if reply:
            self.send_result(reply)
        self._flush_stats()

    # ---------------------------------------------------------------------------------------
    def _canvas_commit(
        self, req: Req, state: CanvasState, reply: List[DetokenizeMsg], finished_now: set
    ) -> None:
        """Commit a finished block: emit the ARGMAX canvas (trap 3), truncated at the first EOS, and
        either finish the request or re-encode the block and open the next one."""
        device = self.device
        cfg = self._canvas_cfg
        L = cfg.canvas_length
        c0 = req.cached_len

        tokens = state.argmax.tolist()
        # Everything strictly after the first EOS becomes nothing at all. The reference pads it to
        # pad_token_id and keeps the row length fixed; a serve truncates, which is the same text.
        eos = False
        for j, tok in enumerate(tokens):
            if tok in self.eos_token_ids:
                tokens = tokens[: j + 1]
                eos = True
                break
        # A block is a fixed 256 tokens but max_tokens is arbitrary, so the tail past the request's
        # own budget is dropped. The canvas itself is always full-length — shrinking it would change
        # the model's input shape and therefore its output.
        budget = max(req.max_device_len - c0, 0)
        if len(tokens) > budget:
            tokens = tokens[:budget]
        n = len(tokens)

        if n:
            self.token_pool[
                torch.full((n,), req.table_idx, dtype=torch.int64, device=device),
                torch.arange(c0, c0 + n, device=device),
            ] = torch.tensor(tokens, dtype=self.token_pool.dtype, device=device)
            req.append_host(torch.tensor(tokens, dtype=req.input_ids.dtype))

        steps = cfg.max_denoising_steps - state.step_index
        self.canvas_slots.end(req.uid)
        finished = eos or n < L or req.max_device_len - (c0 + n) <= 0
        # `k` — the realised denoising steps for this block — is the number the entire cost case for
        # block diffusion turns on, and it is invisible in tok/s (a block emits all its tokens at
        # once). Logged per block so it can be read off a real serve rather than modelled.
        logger.info_rank0(
            f"[canvas] uid={req.uid} block@{c0} steps={steps}/{cfg.max_denoising_steps} "
            f"emitted={n} mean_entropy={state.last_mean_entropy:.4f} "
            f"eos={eos} finished={finished}"
        )

        # Release the slots past the kept run. The canvas always allocated L; a truncated block keeps
        # only its prefix, and the rest goes straight back — the same page-aligned rollback the spec
        # path does, so a partial page straddling the boundary is retained.
        ps = self.cache_manager.page_size
        free_start = div_ceil(c0 + n, ps) * ps
        free_end = div_ceil(c0 + L, ps) * ps
        if free_end > free_start:
            self.cache_manager._free(self.engine.page_table[req.table_idx, free_start:free_end])

        req.cached_len = c0 + n
        req.device_len = req.cached_len

        if n:
            reply.append(
                DetokenizeMsg(
                    uid=req.uid,
                    next_token=tokens[0],
                    finished=finished,
                    extra_tokens=tokens[1:],
                    finish_reason=("stop" if eos else "length") if finished else None,
                )
            )
        # The KV now in [c0, c0+n) is the decoder's BIDIRECTIONAL K/V, which this model never serves
        # from (trap 4). Overwrite it with the causal encoder K/V. No allocation — these are the
        # canvas's own slots.
        #
        # ALSO ON THE FINISHING BLOCK, which it did not used to be. `_free_req_resources` inserts the
        # whole finished sequence into the prefix cache (and attaches its window snapshot), so a last
        # block left un-re-encoded publishes the decoder's bidirectional K/V as a reusable prefix —
        # in the main pool AND in the SWA ring. That is a prefix-cache correctness bug independent of
        # SWA-radix: it poisons plain radix reuse of a completed conversation, which is exactly the
        # multi-turn case prefix caching exists for. The cost is one causal forward per request, at
        # the end, once.
        if n:
            self._canvas_reencode(req, c0, n)
        if finished:
            finished_now.add(req)
            return

    def _canvas_reencode(self, req: Req, c0: int, n: int) -> None:
        """Causal encoder pass over the just-committed block, in place over its own slots.

        `phase="decode"` with `extend_len = n > 1` is the ordinary paged-extend prefill shape: the
        full layers read [0, c0+n) from the main pool with the prefix-offset causal mask, and the
        sliding layers gather their window from the ring and attend [pad | window | new]. Both read
        the window at ring slots disjoint from the canvas's, which is only true because the ring
        stride was widened by `canvas_length`."""
        req.cached_len, req.device_len = c0, c0 + n
        batch = Batch(reqs=[req], phase="decode")
        self._canvas_forward(batch, allocate=False)
        req.cached_len = req.device_len = c0 + n


# `_make_positions` / `_make_input_tuple` live in scheduler.py; importing them at module scope would
# be circular (scheduler.py imports this mixin), so they are bound lazily on first use.
def _make_positions(batch, device):
    from .scheduler import _make_positions as f

    return f(batch, device)


def _make_input_tuple(batch, device):
    from .scheduler import _make_input_tuple as f

    return f(batch, device)


__all__ = ["CanvasManager", "SchedulerDiffusionMixin"]
