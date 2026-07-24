from __future__ import annotations

import os
from typing import TYPE_CHECKING, Dict, List

import torch

from .base import Proposer, ProposeContext

if TYPE_CHECKING:
    from minisgl.core import Req


__all__ = ["MTPProposer"]


class MTPProposer(Proposer):
    """Native MTP (multi-token-prediction) self-speculation: the target's OWN appended next-token
    head (GLM-4.x ``model.layers.<num_layers>`` / Qwen3.5 ``mtp.*``) run autoregressively as the
    draft model. The one MTP decoder layer fuses the embedded token with the previous hidden state
    and predicts the next token:

        h     = layer( fuse( embed(tok), prev_hidden ) )      ; next_draft = argmax(head(h))

    where for the FIRST step ``tok`` is the just-confirmed token and ``prev_hidden`` is the target's
    last hidden at that token (captured during the previous verify, fed back via
    ``ctx.last_hidden``); for later steps ``tok`` is the previous draft and ``prev_hidden`` is the
    MTP layer's own output hidden.

    **Persistent draft KV.** The MTP decoder layer is a real transformer layer, so its self-attention
    needs the full causal context, not just the K draft tokens. This proposer keeps a PERSISTENT
    per-request MTP KV cache (one MTP layer): every confirmed token is run through the MTP layer and
    its K/V appended (growing the context across decode steps); the K draft tokens append temporary
    K/V that ``on_accept`` truncates back to the accepted prefix. Restricting the attention to only
    the K draft tokens (no prior context) collapses the MTP head to ~40% single-token accuracy; the
    persistent cache restores the head's trained accuracy.

    The cache starts empty at the first decode step (the prompt prefill is not replayed through the
    MTP layer), so the very first drafts have little context, but it fills in within a few steps.
    ``needs_last_hidden=True`` makes the scheduler capture the target's last_hidden seed.
    """

    needs_last_hidden = True
    supports_prefill_seed = True

    def __init__(self, engine, num_draft: int) -> None:
        self._engine = engine
        self._num_draft = num_draft
        self._head = engine.model.mtp
        assert self._head is not None, (
            "MTP proposer requires a model with an MTP head (GLM-4.x num_nextn_predict_layers>0 / "
            "Qwen3.5 mtp_num_hidden_layers>0); none was loaded"
        )
        self._device = engine.device
        # Persistent per-uid MTP KV: list of (k_full, v) one entry per processed position. The first
        # `_committed[uid]` entries are confirmed (permanent context); any beyond are this step's
        # draft tail (truncated by on_accept). _last_hidden carries the MTP layer's own output hidden
        # at the last confirmed position into the next step's chain.
        self._cache: Dict[int, list] = {}
        self._committed: Dict[int, int] = {}
        self._pos_shift = int(os.environ.get("MINISGL_MTP_POS_SHIFT", "0"))

        # --- CUDA-graph-capturable BUFFERED propose (MINISGL_SPEC_PROPOSE_GRAPH=1) ------------------
        # Replaces the per-uid Python-list KV + per-req eager loop with a GLOBAL fixed-shape draft-KV
        # buffer keyed by req.table_idx (the same stable slot the page_table / GDN-state use) + a per-
        # slot cursor, and a single BATCHED K-step chain over step_masked (fixed shapes → capturable).
        # Byte-exact vs the eager list path (spec is lossless: output == base greedy regardless). The
        # graph wrapper is layered on top of this batched path once it is validated eager.
        # Default ON (the captured path is the "complete" one; eager is unfinished). Opt out with
        # MINISGL_SPEC_PROPOSE_GRAPH=0. Requires the masked draft attention (Qwen3.5 MTP head) — any
        # other head (e.g. GLM MTP) gracefully keeps the eager list path.
        _want = os.environ.get("MINISGL_SPEC_PROPOSE_GRAPH", "1") != "0"
        attn = self._head.self_attn
        self._buffered = _want and hasattr(attn, "forward_draft_masked")
        if _want and not self._buffered:
            from minisgl.utils import init_logger
            init_logger(__name__).info_rank0(
                "spec-decode: buffered/captured MTP propose unavailable (head has no masked draft "
                "attention) — using the eager list path")
        if self._buffered:
            # Draft-KV buffer geometry is head-specific: Qwen packs GQA (nkv heads, symmetric hd for
            # K and V); GLM MLA materializes full multi-head with an ASYMMETRIC k-dim (qk) vs v-dim.
            # Each MTP attention declares its own (n_k_heads, k_dim, n_v_heads, v_dim).
            _nkh, _kdim, _nvh, _vdim = attn.draft_buffer_dims()
            # page_table is [max_running_req + 1, aligned_max_seq_len]; its row count is exactly the
            # slot space of req.table_idx (0..max_running_req), so key the draft-KV buffer off it.
            self._max_slots = int(engine.page_table.shape[0])
            # Cap the draft-KV window to bound VRAM (buffer = slots * max_ctx * nkv * hd * 2kv * dt).
            # TODO: sequences whose MTP context exceeds max_ctx are not yet handled (would overflow the
            # cursor); the write_col is clamped as a backstop. Raise MINISGL_MTP_MAX_CTX for long ctx.
            self._max_ctx = min(int(engine.max_seq_len),
                                int(os.environ.get("MINISGL_MTP_MAX_CTX") or "8192"))
            dt = engine.dtype
            dev = self._device
            self._k_buf = torch.zeros(self._max_slots, self._max_ctx, _nkh, _kdim, device=dev, dtype=dt)
            self._v_buf = torch.zeros(self._max_slots, self._max_ctx, _nvh, _vdim, device=dev, dtype=dt)
            self._cur = torch.zeros(self._max_slots, dtype=torch.int64, device=dev)   # committed len/slot
            self._col_idx = torch.arange(self._max_ctx, device=dev)                    # [max_ctx]
            self._slot_uid: Dict[int, int] = {}   # which uid currently owns each slot (reset cursor on reuse)
            self._drafted_slots: List[int] = []   # slots that drafted last step (for on_accept advance)
            # Static I/O buffers for the CAPTURED K-step chain (one graph per exact batch size, captured
            # on demand). The chain reads _g_seed/_g_tok/_g_slots/_g_base/_g_curb[:B] and writes drafts to
            # _g_out[:B]; k_buf/v_buf are persistent externals the graph writes in place.
            hidden = int(getattr(self._head, "hidden_size", 0)) \
                or int(self._head.pre_fc_norm_hidden.weight.shape[-1])
            G = self._max_slots
            self._g_seed = torch.zeros(G, hidden, device=dev, dtype=dt)
            self._g_tok = torch.zeros(G, dtype=torch.int64, device=dev)
            self._g_slots = torch.zeros(G, dtype=torch.int64, device=dev)
            self._g_base = torch.zeros(G, dtype=torch.int64, device=dev)
            self._g_curb = torch.zeros(G, dtype=torch.int64, device=dev)
            self._g_out = torch.zeros(G, self._num_draft, dtype=torch.int64, device=dev)
            self._graphs: Dict[int, "torch.cuda.CUDAGraph"] = {}   # one captured graph per exact bs
            self._pool = None
            # MINISGL_SPEC_PROPOSE_NOCAPTURE=1 keeps the batched buffer path but runs it EAGER (A/B the
            # capture win vs the buffering alone). DP+EP keeps propose eager (the captured MoE all_gather
            # would pin a fixed N vs an idle replica's self-agreed N). EP-OVER-TP is fine: its MTP draft
            # head is built REPLICATED (force_no_ep) so propose issues NO EP collective — just a plain-TP
            # all_reduce — and the TP ranks capture in lockstep. So capture is allowed under EP-over-TP.
            from minisgl.distributed import is_ep_over_tp
            self._capture_ok = (
                os.environ.get("MINISGL_SPEC_PROPOSE_NOCAPTURE") != "1"
                and (not getattr(engine, "enable_ep", False) or is_ep_over_tp()))
            from minisgl.utils import init_logger
            init_logger(__name__).info_rank0(
                f"spec-decode: MTP BUFFERED propose ENABLED (slots={self._max_slots}, "
                f"max_ctx={self._max_ctx}, draft-KV {2 * self._k_buf.numel() * self._k_buf.element_size() / 1e6:.0f} MB)")

    @torch.inference_mode()
    def propose(self, reqs: List["Req"], num_draft: int, ctx: ProposeContext) -> List[List[int]]:
        if self._buffered:
            return self._propose_buffered(reqs, num_draft, ctx)
        # Per-req draft budget: clamp to remain_len-1 so a full accept stays within budget. A req with
        # no captured last_hidden yet (the very first decode step after prefill) drafts nothing and
        # falls back to a plain decode (which then produces a seed for the next step).
        out: List[List[int]] = [[] for _ in reqs]
        head = self._head
        device = self._device
        dbg = os.environ.get("MINISGL_MTP_DBG") == "1"
        for i, req in enumerate(reqs):
            k_i = max(0, min(num_draft, req.remain_len - 1))
            seed = ctx.last_hidden.get(req.uid)
            if k_i <= 0 or seed is None:
                continue
            # Drop the previous step's rejected-draft tail (keep only confirmed context).
            cache = self._cache.setdefault(req.uid, [])
            committed = self._committed.get(req.uid, 0)
            del cache[committed:]

            base_pos = req.cached_len  # position of the confirmed token
            conf_tok = int(req.input_ids[base_pos])
            cur_tok = torch.tensor([conf_tok], dtype=torch.int64, device=device)
            cur_hidden = seed.view(1, -1)  # [1, hidden]
            # On-device draft chain: keep the per-step argmax and the next-token id ON DEVICE across
            # the K steps, so the autoregressive chain runs without a CPU<->GPU round-trip per step
            # (the chain is sequential, but no host sync lets the GPU chain the steps back-to-back; the
            # CPU just enqueues and pulls the whole chain to host ONCE at the end). Per-step positions
            # are sliced from one precomputed tensor (vs a torch.tensor(...) alloc per step).
            positions_all = torch.arange(
                base_pos + self._pos_shift,
                base_pos + self._pos_shift + k_i,
                dtype=torch.int32,
                device=device,
            )
            draft_ids: List[torch.Tensor] = []
            # Step 0 processes the confirmed token (appends its K/V to the persistent cache and
            # predicts d0); steps 1.. process each draft. The attention sees the full cache (confirmed
            # prefix built over prior steps + this chain), so the MTP head runs with real context.
            for step in range(k_i):
                fused = head.fuse(head.embed(cur_tok), cur_hidden)
                logits, cur_hidden = head.step(
                    fused, positions_all[step : step + 1], cache, len(cache)
                )
                nxt = logits.argmax(dim=-1)  # [1] target vocab (on device)
                draft_ids.append(nxt)
                cur_tok = nxt  # next step embeds this id; no host sync
            drafts = torch.cat(draft_ids).cpu().tolist() if draft_ids else []
            out[i] = drafts
            if dbg:
                print(f"[mtp-dbg] uid={req.uid} conf={conf_tok} base_pos={base_pos} "
                      f"ctx={committed} draft={drafts}", flush=True)
        return out

    @torch.inference_mode()
    def _propose_buffered(
        self, reqs: List["Req"], num_draft: int, ctx: ProposeContext
    ) -> List[List[int]]:
        """Batched, capture-shaped MTP propose over the GLOBAL draft-KV buffer (see __init__). One
        BATCHED chain of K step_masked calls replaces the per-req Python loop + growing-list stack.

        Cursor/slot bookkeeping mirrors the eager list EXACTLY (so output stays byte-identical to base):
          * slot = req.table_idx; `_cur[slot]` = committed KV length (like the eager `_committed[uid]`).
          * a slot reused by a NEW uid resets `_cur=0` (stale buffer rows are masked off, not read).
          * step j writes the processed token's K/V at column `_cur+j` (step 0 = confirmed token, step j =
            draft d_{j-1}); K columns written = [confirmed, d0 .. d_{K-2}], d_{K-1}'s K/V never stored —
            identical to eager. `on_accept` advances `_cur += min(1+n, K)` (the same full-accept cap).
          * reqs with no seed (cold first step) or k_i<=0 draft [] and DON'T advance their cursor."""
        device = self._device
        K = num_draft
        out: List[List[int]] = [[] for _ in reqs]
        active_i: List[int] = []
        seeds: List[torch.Tensor] = []
        for i, req in enumerate(reqs):
            k_i = max(0, min(K, req.remain_len - 1))
            seed = ctx.last_hidden.get(req.uid) if ctx.last_hidden else None
            if k_i <= 0 or seed is None:
                continue
            s = int(req.table_idx)
            if self._slot_uid.get(s) != req.uid:   # fresh req on this slot → cold cache
                self._slot_uid[s] = req.uid
                self._cur[s] = 0
            active_i.append(i)
            seeds.append(seed.view(-1))
        self._drafted_slots = [int(reqs[i].table_idx) for i in active_i]
        if not active_i:
            return out
        B = len(active_i)
        # Load the batch into the STATIC input buffers the (capturable) chain reads from.
        self._g_slots[:B].copy_(torch.tensor([int(reqs[i].table_idx) for i in active_i],
                                             dtype=torch.int64, device=device))
        self._g_base[:B].copy_(torch.tensor([int(reqs[i].cached_len) for i in active_i],
                                            dtype=torch.int64, device=device))
        self._g_tok[:B].copy_(torch.tensor(
            [int(reqs[i].input_ids[reqs[i].cached_len]) for i in active_i],
            dtype=torch.int64, device=device))
        self._g_seed[:B].copy_(torch.stack(seeds).to(self._engine.dtype))
        self._g_curb[:B].copy_(self._cur[self._g_slots[:B]])   # committed length per row
        # Capture one graph per EXACT batch size B (no padding). Padding a small batch up to maxrun did
        # 6x the MoE work at bs=1 and regressed low-batch mixed; per-B is affordable now that the grouped
        # GQA einsum shrank the per-graph pool ~8x (the OOM that first forced padding is gone).
        self._run_chain(B)
        drafts = self._g_out[:B].cpu().tolist()               # [B, K] (real rows only)
        for i, d in zip(active_i, drafts):
            k_i = max(0, min(K, reqs[i].remain_len - 1))
            out[i] = d[:k_i]     # stage only the budgeted prefix (extra K-k_i drafts overwritten next step)
        return out

    def _chain(self, B: int) -> None:
        """The K-step draft chain over the STATIC buffers [:B] — the body captured into a CUDA graph
        (and run eagerly when capture is off). Reads _g_seed/_g_tok/_g_slots/_g_base/_g_curb, writes each
        step's argmax to _g_out and the token's K/V into the persistent k_buf/v_buf. No host sync."""
        head = self._head
        slots = self._g_slots[:B]
        base = self._g_base[:B]
        cur = self._g_curb[:B]
        cur_tok = self._g_tok[:B]
        cur_hidden = self._g_seed[:B]
        col = self._col_idx.unsqueeze(0)
        for j in range(self._num_draft):
            fused = head.fuse(head.embed(cur_tok), cur_hidden)
            write_col = (cur + j).clamp(max=self._max_ctx - 1)               # OOB backstop
            positions = (base + self._pos_shift + j).to(torch.int32)
            mask_bias = torch.where(col <= write_col.unsqueeze(1),
                                    0.0, float("-inf")).to(torch.float32)
            logits, cur_hidden = head.step_masked(
                fused, positions, self._k_buf, self._v_buf, slots, write_col, mask_bias)
            nxt = logits.argmax(dim=-1)
            self._g_out[:B, j] = nxt
            cur_tok = nxt

    def _run_chain(self, B: int) -> None:
        """Run the chain for batch size B: replay the captured graph (capturing it on first sight of B),
        or run eager when capture is disabled. The index tensors (_g_slots, cursor) are read from static
        buffers at replay time, so one graph per B serves every step regardless of which slots/cursors."""
        if not self._capture_ok:
            self._chain(B)
            return
        g = self._graphs.get(B)
        if g is None:
            # Canonical capture dance: warm up on a side stream (allocates workspaces), then capture.
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                self._chain(B)
                self._chain(B)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self._pool):
                self._chain(B)
            if self._pool is None:
                self._pool = g.pool()
            self._graphs[B] = g
        g.replay()

    @torch.inference_mode()
    def seed_prefill(self, req: "Req", last_hidden, aux_hidden=None) -> None:
        # Seed the persistent MTP KV from the prompt prefill so the FIRST draft already sees full
        # prompt context (the cache otherwise starts empty at the first decode -> cold early tokens).
        # We mirror the decode-time convention exactly: propose processes the pair
        # (embed(token_p), h_{p-1}) at RoPE position p (h = the target hidden that produced token_p),
        # so we run the MTP layer's k/v over the prompt pairs p=1..P-1 and mark them all committed.
        # The first decode step's propose then appends position P (the bonus token), attending to the
        # whole prompt. (Position 0 has no h_{-1}, so it is the one prompt token left out of the cache
        # — one key among P, negligible.)
        if last_hidden is None:
            return
        P = last_hidden.shape[0]
        if P < 2:
            return  # nothing to seed (P==1: only the bonus, handled by the first propose)
        device = self._device
        tokens = req.input_ids[1:P].to(device=device, dtype=torch.int64)  # token_p, p=1..P-1
        prev_hidden = last_hidden[0 : P - 1].to(self._engine.dtype)  # h_{p-1}
        positions = torch.arange(1, P, dtype=torch.int32, device=device)
        if self._buffered:
            # BUFFERED path (default: head has forward_draft_masked): seed the GLOBAL draft-KV buffer
            # directly instead of the per-uid list (which _propose_buffered never reads). Slot/cursor
            # convention mirrors _propose_buffered EXACTLY so a seeded prefix is byte-identical to the
            # eager list path: slot = req.table_idx; columns 0..S-1 hold RoPE positions 1..P-1 (S=P-1);
            # `_cur[slot] = S` so the first decode propose writes the bonus token at column S and its
            # mask exposes the whole seeded prefix. Register `_slot_uid[slot]` so that first propose
            # does NOT treat the slot as cold and reset the cursor to 0.
            S = P - 1
            if S > self._max_ctx:
                # Prompt longer than the draft-KV window: fall back to the cold cache (still lossless,
                # just no early-token lift). Seeding the tail would misalign the column<->position map
                # the decode chain assumes (write_col = cur+j at RoPE position base+j).
                return
            slot = int(req.table_idx)
            self._head.seed_buffered(
                tokens, prev_hidden, positions, self._k_buf, self._v_buf, slot, 0)
            self._slot_uid[slot] = req.uid
            self._cur[slot] = S
            return
        entries = self._head.seed_kv(tokens, prev_hidden, positions)
        self._cache[req.uid] = entries
        self._committed[req.uid] = len(entries)

    def on_accept(self, reqs: List["Req"], num_accepted: List[int]) -> None:
        if self._buffered:
            # Advance each drafted slot's committed cursor by min(1+n, K) — the SAME full-accept cap as
            # the eager path (d_{K-1}'s K/V was never written). Reqs that drafted [] this step (cold /
            # k_i<=0, not in _drafted_slots) do NOT advance: their confirmed token wasn't written.
            drafted = set(self._drafted_slots)
            for req, n in zip(reqs, num_accepted):
                s = int(req.table_idx)
                if s in drafted:
                    self._cur[s] = self._cur[s] + min(1 + n, self._num_draft)
            return
        # The confirmed token (always committed) plus the n accepted drafts become permanent MTP
        # context; the K-n rejected drafts' K/V are dropped. The cache entry layout per step is
        # [confirmed, d0, d1, ...]; committing 1 + n keeps confirmed + the accepted run.
        #
        # FULL-ACCEPT CAP: propose stores k_i cache entries [confirmed, d0 .. d_{k-2}] — the LAST
        # draft d_{k-1} is predicted but never processed, so its K/V is never appended. On a full
        # accept (n == K) `1 + n` would run one past len(cache), leaving a 1-key hole that misaligns
        # the next confirmed token. d_{k-1}'s K/V genuinely does not exist, so cap `committed` at what
        # was actually stored: it is simply absent from the head's context (positions stay aligned, no
        # overwrite). Only fires on full accepts; a no-op otherwise (1+n <= len(cache) for n < K).
        for req, n in zip(reqs, num_accepted):
            if req.uid in self._cache:
                self._committed[req.uid] = min(
                    self._committed.get(req.uid, 0) + 1 + n, len(self._cache[req.uid]))
                # Eagerly trim so a finished/aborted-then-reused uid never inherits stale tail.
                del self._cache[req.uid][self._committed[req.uid]:]

    def free(self, uid: int) -> None:
        if self._buffered:
            # Release any slot this uid owned so a reused slot is treated as cold (cursor reset). Stale
            # buffer rows need not be zeroed — the next owner's mask never exposes columns past its cursor.
            for s, u in list(self._slot_uid.items()):
                if u == uid:
                    del self._slot_uid[s]
                    self._cur[s] = 0
            return
        self._cache.pop(uid, None)
        self._committed.pop(uid, None)
