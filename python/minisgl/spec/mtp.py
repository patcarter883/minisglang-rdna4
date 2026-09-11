from __future__ import annotations

import os
from typing import TYPE_CHECKING, Dict, List, Optional

import torch

from minisgl.utils import init_logger

from .base import ProposeContext
from .capture import CapturableProposer, StagedPropose

if TYPE_CHECKING:
    from minisgl.core import Req


__all__ = ["MTPProposer"]

logger = init_logger(__name__)


class MTPProposer(CapturableProposer):
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
    needs the full causal context, not just the K draft tokens (restricting it to the chain collapses
    the head to ~40% single-token accuracy). The KV lives in ONE GLOBAL fixed-shape buffer
    ``[max_slots, ring, nkv, hd]`` keyed by ``req.table_idx`` — the same stable slot index the
    paged ``page_table`` and the GDN/CCA recurrent state use — plus a per-slot cursor. A growing
    per-uid Python list would be both a dynamic shape and a host-side mutation, i.e. uncapturable;
    the fixed window + additive ``-inf`` mask beyond the cursor is byte-exact against it
    (``softmax(-inf) == 0``; validated by ``tools/mtp_forward_draft_parity.py``).

    Capture/replay itself is NOT here — it is the shared ``CapturableProposer`` machinery, which
    DFlash and EAGLE3 ride too. This class supplies only the four hooks.
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
        self._pos_shift = int(os.environ.get("MINISGL_MTP_POS_SHIFT", "0"))
        # Draft-chain trace, on the SAME switch DFlash and EAGLE3 use. It used to be a private
        # MINISGL_MTP_DBG that docker-compose does not forward — so MTP's drafted chains were
        # unreachable through the only way this repo serves, and the replay-vs-eager identity gate
        # (which diffs exactly these lines) silently had nothing to compare.
        self._dbg = os.environ.get("MINISGL_SPEC_DEBUG") in ("2", "3")
        self.init_propose_capture(engine)

    # ------------------------------------------------------------------ hook: buffer allocation
    def init_propose_capture(self, engine) -> None:
        """Allocate the global draft-KV buffer, the per-slot cursors and the static propose I/O.

        Every tensor a captured body touches must be allocated exactly ONCE here: a graph records
        kernel argument POINTERS, so a per-uid dict of freshly allocated tensors can never be baked
        into one. The masked draft attention is a hard requirement (there is no second, eager
        implementation of propose to fall back to — carrying two was the debt this removes); both
        shipped MTP heads (Qwen3.5 ``qwen3_5.py`` and GLM ``glm4_moe_lite.py``) implement it."""
        attn = self._head.self_attn
        assert hasattr(attn, "forward_draft_masked"), (
            f"{type(attn).__name__} has no forward_draft_masked/draft_buffer_dims — an MTP head must "
            "provide the fixed-shape masked draft attention (see Qwen3_5MTPAttn) for propose to be "
            "capturable; there is no eager fallback path any more")
        # Draft-KV geometry is head-specific: Qwen packs GQA (nkv heads, symmetric hd for K and V);
        # GLM MLA materializes full multi-head with an ASYMMETRIC k-dim (qk) vs v-dim.
        _nkh, _kdim, _nvh, _vdim = attn.draft_buffer_dims()
        # page_table is [max_running_req + 1, aligned_max_seq_len]; its row count is exactly the slot
        # space of req.table_idx. One EXTRA row on top is the reserved NULL slot that bucket-padding
        # rows write into (a real buffer row no live sequence ever owns, so its garbage is discarded).
        self._live_slots = int(engine.page_table.shape[0])
        self._null_slot = self._live_slots
        self._max_slots = self._live_slots + 1
        dt = engine.dtype
        dev = self._device
        # Long-context spec gate: past this committed length a request skips propose and decodes
        # plain. Above the window the seed fell back to cold, so the drafts are context-BLIND
        # (near-zero accept) while still paying propose + verify(K+1) — net-negative.
        # RING, not a linear absolute-indexed buffer. Column WAS the absolute position, which capped
        # context at the allocation and made _ctx_gate skip propose entirely past it — spec silently
        # became a no-op for long prompts (measured: 3.8 drafts/step at 6952 prompt tokens, 0.00 at
        # 15900, with nothing logged). A ring stores the most recent W positions at col = pos % W and
        # masks from ABSOLUTE POSITIONS, so context is unbounded at fixed VRAM. Same design DFlash's
        # prefix ring already uses here (spec/dflash.py: col = pos % C, keep = (pa <= qa) & (qa - pa
        # < window)). The kernel needs no change: step_masked already takes an arbitrary write_col
        # and an explicit mask_bias, and never assumes column == position.
        #
        # W=512, MEASURED. The buffer is max_slots*W, so this is 3 MB where the old full-context
        # 8192 was 50 MB. A provenance-asserted sweep (35B-MXFP4 TP=2, MTP K=4) found NO detectable
        # cost — every difference sits inside the run-to-run noise floor (+-0.06 accept,
        # +-0.15 emitted/step), accept / emitted-per-step / tok-s:
        #     W=512   3 MB   0.54/2.81/51.1   0.49/2.76/18.9   0.54/2.58/9.3
        #     W=2048 13 MB   0.49/2.62/48.5   0.48/2.58/18.5   0.53/2.62/9.3
        #     W=8192 50 MB   0.53/2.81/50.4   0.49/2.76/18.9   0.54/2.67/9.3
        #                    (ctx 6955)       (ctx 25651)      (ctx 45548)
        # i.e. the drafter does not need long context: 512 tokens of recent history match 8192 at
        # every depth tested. The freed 47 MB is post-KV-pool SLACK, not KV pool — the draft KV is
        # allocated from what is free AFTER the pool is sized — so it buys OOM headroom for the
        # capture/transient path, which is exactly where this stack runs out (see _spec_seed_fits).
        # Raise it if a model's drafter turns out to be more context-sensitive than MTP's.
        # 512 IS A MEASURED DEFAULT, BUT IT WAS MEASURED ON A DENSE-ATTENTION DRAFTER. For a
        # QSA model the draft window and the target's attention are not the same thing: upstream's
        # MTP decode "reuses the draft-extend's target-aligned selection" (sglang qwen4_exp.py,
        # `_compute_qsa_topk_indices`), i.e. the drafter sees the up-to-`indexer_budget` keys the
        # TARGET selected out of the whole context. This ring instead holds the most RECENT tokens.
        # Below the budget the two coincide — the selection takes every visible key, which is why
        # the short-context numbers are unaffected — but past it a 512-token recent window is a
        # strictly different (and much narrower) view than the 2048 keys the target attended over,
        # and the drafter was trained on the latter. So on a QSA model the floor is the selection
        # width, not 512. Still an approximation of upstream, and deliberately labelled as one.
        _idx_width = getattr(self._head, "qsa_index_width", None)
        _floor = max(512, int(_idx_width)) if _idx_width else 512
        self._ring = max(64, int(os.environ.get("MINISGL_MTP_KV_WINDOW") or _floor))
        # Kept as an ESCAPE HATCH only (unset = unbounded). It used to default to the full-context
        # buffer length and was the silent cliff; the ring makes any length drafts-capable, so there
        # is nothing to gate.
        _gate_env = os.environ.get("MINISGL_SPEC_MAX_CONTEXT")
        self._ctx_gate = int(_gate_env) if _gate_env else (1 << 62)
        self._k_buf = torch.zeros(self._max_slots, self._ring, _nkh, _kdim, device=dev, dtype=dt)
        self._v_buf = torch.zeros(self._max_slots, self._ring, _nvh, _vdim, device=dev, dtype=dt)
        self._cur = torch.zeros(self._max_slots, dtype=torch.int64, device=dev)  # ABSOLUTE len/slot
        # Absolute position held by each ring column; -1 = empty. This is what the mask is built from.
        self._pos_buf = torch.full((self._max_slots, self._ring), -1,
                                   dtype=torch.int64, device=dev)
        self._col_idx = torch.arange(self._ring, device=dev)                     # [ring]
        self._slot_uid: Dict[int, int] = {}   # which uid owns each slot (reset the cursor on reuse)
        self._drafted_slots: List[int] = []   # slots that drafted last step (for on_accept advance)
        hidden = int(getattr(self._head, "hidden_size", 0)) \
            or int(self._head.pre_fc_norm_hidden.weight.shape[-1])
        G = self._max_slots
        self._g_seed = torch.zeros(G, hidden, device=dev, dtype=dt)
        self._g_curb = torch.zeros(G, dtype=torch.int64, device=dev)
        self._g_out = torch.zeros(G, self._num_draft, dtype=torch.int64, device=dev)
        # The three int64 row inputs live as ROWS OF ONE tensor, staged through ONE pinned host
        # buffer: a single H2D copy per step instead of three `torch.tensor(...)` allocations plus
        # three copies. `_g_slots`/`_g_base`/`_g_tok` are VIEWS, so their device pointers are stable
        # (which is what capture requires) and refreshing `_g_idx` refreshes all three.
        self._h_idx = torch.zeros(3, G, dtype=torch.int64, device="cpu", pin_memory=True)
        self._g_idx = torch.zeros(3, G, dtype=torch.int64, device=dev)
        self._g_slots, self._g_base, self._g_tok = self._g_idx[0], self._g_idx[1], self._g_idx[2]
        self.init_propose_capture_state(engine, tag="MTP")
        logger.info_rank0(
            f"spec-decode: MTP propose buffers (slots={self._live_slots}+NULL, "
            f"ring={self._ring} (unbounded context), draft-KV "
            f"{(self._k_buf.numel() + self._v_buf.numel()) * dt.itemsize / 1e6:.0f} MB)")

    # ----------------------------------------------------------------------- hook: HOST staging
    def stage_propose(
        self, reqs: List["Req"], num_draft: int, ctx: ProposeContext, **kw
    ) -> Optional[StagedPropose]:
        """Pick the rows that will draft and refresh the static inputs in place.

        Cursor/slot bookkeeping: slot = ``req.table_idx``; ``_cur[slot]`` = committed KV length; a
        slot reused by a NEW uid resets ``_cur = 0`` (stale rows are masked off, never zeroed); step
        j writes the processed token's K/V at column ``_cur + j`` (step 0 = the confirmed token, step
        j = draft d_{j-1}), so d_{K-1}'s K/V is never stored and ``on_accept`` advances the cursor by
        ``min(1+n, K)``. Reqs that draft nothing do NOT advance."""
        rows: List[int] = []
        budget: List[int] = []
        seeds: List[torch.Tensor] = []
        h = self._h_idx
        for i, req in enumerate(reqs):
            k_i = max(0, min(num_draft, req.remain_len - 1))
            seed = ctx.last_hidden.get(req.uid) if ctx.last_hidden else None
            if k_i <= 0 or seed is None or req.cached_len > self._ctx_gate:
                continue
            s = int(req.table_idx)
            if self._slot_uid.get(s) != req.uid:   # fresh req on this slot -> cold cache
                self._slot_uid[s] = req.uid
                self._cur[s] = 0
                # The ring mask keys off ABSOLUTE positions, so a cursor reset alone is not enough:
                # the previous owner's positions would still satisfy `pa <= qa`. Invalidate the ring.
                self._pos_buf[s].fill_(-1)
            j = len(rows)
            h[0, j] = s
            h[1, j] = int(req.cached_len)
            h[2, j] = int(req.input_ids[req.cached_len])
            rows.append(i)
            budget.append(k_i)
            seeds.append(seed.view(-1))
        self._drafted_slots = [int(reqs[i].table_idx) for i in rows]
        if not rows:
            return None
        B = len(rows)
        self._g_idx[:, :B].copy_(h[:, :B], non_blocking=True)
        self._g_seed[:B].copy_(torch.stack(seeds).to(self._engine.dtype))
        self._g_curb[:B].copy_(self._cur[self._g_slots[:B]])   # committed length per row
        return StagedPropose(B, rows, budget)

    def pad_propose_rows(self, bs: int, bucket: int) -> None:
        """Point rows [bs, bucket) at the reserved NULL slot with an empty cursor and a zero seed."""
        self._g_slots[bs:bucket].fill_(self._null_slot)
        self._g_base[bs:bucket].zero_()
        self._g_tok[bs:bucket].zero_()
        self._g_curb[bs:bucket].zero_()
        self._g_seed[bs:bucket].zero_()

    # ------------------------------------------------------------------------ hook: the BODY
    def propose_body(self, bs: int) -> None:
        """The K-step draft chain over the STATIC buffers [:bs] — captured into a CUDA graph, and run
        eagerly (same callable) when capture is unavailable. Reads _g_seed/_g_tok/_g_slots/_g_base/
        _g_curb, writes each step's argmax to _g_out and the token's K/V into k_buf/v_buf. No sync."""
        head = self._head
        slots = self._g_slots[:bs]
        base = self._g_base[:bs]
        cur = self._g_curb[:bs]
        cur_tok = self._g_tok[:bs]
        cur_hidden = self._g_seed[:bs]
        col = self._col_idx.unsqueeze(0)
        for j in range(self._num_draft):
            fused = head.fuse(head.embed(cur_tok), cur_hidden)
            q_abs = cur + j                                   # ABSOLUTE position being written
            write_col = torch.remainder(q_abs, self._ring)     # ring slot
            positions = (base + self._pos_shift + j).to(torch.int32)
            # Publish this token's position BEFORE masking so the row it is about to write is
            # visible to its own attention (the old `col <= write_col` mask included write_col).
            self._pos_buf[slots, write_col] = q_abs
            pa = self._pos_buf[slots]                          # [bs, ring] absolute pos per column
            qa = q_abs.unsqueeze(1)
            keep = (pa >= 0) & (pa <= qa) & ((qa - pa) < self._ring)
            mask_bias = torch.where(keep, 0.0, float("-inf")).to(torch.float32)
            logits, cur_hidden = head.step_masked(
                fused, positions, self._k_buf, self._v_buf, slots, write_col, mask_bias)
            nxt = logits.argmax(dim=-1)
            self._g_out[:bs, j] = nxt
            cur_tok = nxt

    # ------------------------------------------------------------------- hook: the ONE host sync
    def read_drafts(self, reqs: List["Req"], staged: StagedPropose, **kw) -> List[List[int]]:
        out: List[List[int]] = [[] for _ in reqs]
        drafts = self._g_out[: staged.bs].cpu().tolist()     # ONE D2H for the whole step
        for i, k_i, d in zip(staged.rows, staged.budget, drafts):
            out[i] = d[:k_i]   # stage only the budgeted prefix (the extra K-k_i are overwritten)
            if self._dbg:
                print(f"[mtp-dbg] uid={reqs[i].uid} k={k_i} draft={out[i]}", flush=True)
        return out

    def reset_propose_state(self) -> None:
        """The warmup/capture dummy batch ran the chain on the NULL slot only, so nothing a live
        request can observe was touched. Zero the NULL slot's cursor for tidiness."""
        self._cur[self._null_slot] = 0
        self._pos_buf[self._null_slot].fill_(-1)
        self._drafted_slots = []

    # ------------------------------------------------------------------------------ seeding etc.
    @torch.inference_mode()
    def seed_prefill(self, req: "Req", last_hidden, aux_hidden=None) -> None:
        """Seed the persistent MTP KV from the prompt prefill so the FIRST draft already sees the
        prompt context. Mirrors the decode-time convention exactly: propose processes the pair
        (embed(token_p), h_{p-1}) at RoPE position p, writing at ring column p % ring with absolute
        label p. ``last_hidden`` holds the rows for absolute positions [cached_len-P, cached_len) —
        on a radix prefix-cache hit that origin is > 0 and the rows are the prompt SUFFIX, not its
        start (complete_one has already advanced cached_len to the full prompt length; same
        ``ctx_start = cached_len - P`` convention as DFlash). ``_cur[slot]`` is the ABSOLUTE index
        of the next pair to write, so it is set to cached_len and the first decode propose APPENDS
        the bonus pair (token_{cached_len}, RoPE cached_len) at column cached_len % ring instead of
        clobbering the last seeded pair. ``_slot_uid`` is registered so that first propose does not
        treat the slot as cold and reset the cursor."""
        if last_hidden is None:
            return
        P = last_hidden.shape[0]
        if P < 2:
            return  # nothing to seed (P==1: only the bonus, handled by the first propose)
        device = self._device
        slot = int(req.table_idx)
        end = int(req.cached_len)                        # one past the last seeded pair
        origin = end - P                                 # absolute position of last_hidden[0]
        assert origin >= 0, f"seed_prefill: {P} hidden rows exceed cached_len {end}"
        # Seed the ring TAIL. The old code REFUSED when the prompt exceeded the buffer and left the
        # cache cold, which is what made long prompts draft blind. With col = pos % ring the tail is
        # always representable; it just may WRAP, so write it as up to two contiguous runs. Pair
        # `origin` itself is unseedable on a cache hit (h_{origin-1} was never recomputed).
        p_lo = max(origin + 1, end - self._ring)         # first prompt position kept in the ring
        tokens = req.input_ids[p_lo:end].to(device=device, dtype=torch.int64)
        prev_hidden = last_hidden[p_lo - 1 - origin : end - 1 - origin].to(self._engine.dtype)
        positions = torch.arange(p_lo, end, dtype=torch.int32, device=device)
        n = int(positions.numel())
        if n > 0:
            c0 = p_lo % self._ring
            first = min(n, self._ring - c0)              # contiguous run before the wrap
            self._head.seed_buffered(tokens[:first], prev_hidden[:first], positions[:first],
                                     self._k_buf, self._v_buf, slot, c0)
            if first < n:
                self._head.seed_buffered(tokens[first:], prev_hidden[first:], positions[first:],
                                         self._k_buf, self._v_buf, slot, 0)
            abs_pos = torch.arange(p_lo, end, dtype=torch.int64, device=device)
            self._pos_buf[slot, torch.remainder(abs_pos, self._ring)] = abs_pos
        self._slot_uid[slot] = req.uid
        self._cur[slot] = end

    def on_accept(self, reqs: List["Req"], num_accepted: List[int]) -> None:
        # Advance each drafted slot's committed cursor by min(1+n, K) — the FULL-ACCEPT CAP: propose
        # stores [confirmed, d0 .. d_{K-2}] and d_{K-1}'s K/V is never written, so 1+n at n==K would
        # leave a 1-key hole that misaligns the next confirmed token. Reqs that drafted [] this step
        # do NOT advance: their confirmed token was never written. Rejected drafts need no free —
        # their columns are simply re-masked next step.
        drafted = set(self._drafted_slots)
        for req, n in zip(reqs, num_accepted):
            s = int(req.table_idx)
            if s in drafted:
                self._cur[s] = self._cur[s] + min(1 + n, self._num_draft)

    def free(self, uid: int) -> None:
        # Release any slot this uid owned so a reused slot is treated as cold (cursor reset). Stale
        # buffer rows need not be zeroed — the next owner's mask never exposes columns past its cursor.
        for s, u in list(self._slot_uid.items()):
            if u == uid:
                del self._slot_uid[s]
                self._cur[s] = 0
                # Ring columns are position-masked, not cursor-masked, so a stale position from the
                # previous owner WOULD pass `pa <= qa` for the next one. Invalidate explicitly.
                self._pos_buf[s].fill_(-1)
