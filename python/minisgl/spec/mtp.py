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
    ``[max_slots, max_ctx, nkv, hd]`` keyed by ``req.table_idx`` — the same stable slot index the
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
        self._dbg = os.environ.get("MINISGL_MTP_DBG") == "1"
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
        # Cap the draft-KV window to bound VRAM: the buffer is slots*max_ctx*(nkh*kdim+nvh*vdim)*dt,
        # AND every propose gathers k_buf[slot_rows]/v_buf[slot_rows] over the WHOLE window as a
        # transient of comparable size — a naive 8192 window OOMs a heavy head (GLM MLA: 96
        # materialized heads => ~1.4 GB buffer + ~0.3 GB/step transient). Size it from the memory
        # actually free at build time and budget for both.
        from minisgl.engine.graph import get_free_memory

        per_col = self._max_slots * (_nkh * _kdim + _nvh * _vdim) * dt.itemsize
        _budget = int(get_free_memory(dev) * float(os.environ.get("MINISGL_MTP_KV_FRAC", "0.33")))
        _mem_cap = max(512, _budget // max(per_col * 2, 1))
        self._max_ctx = min(int(engine.max_seq_len),
                            int(os.environ.get("MINISGL_MTP_MAX_CTX") or "8192"),
                            int(_mem_cap))
        # Long-context spec gate: past this committed length a request skips propose and decodes
        # plain. Above the window the seed fell back to cold, so the drafts are context-BLIND
        # (near-zero accept) while still paying propose + verify(K+1) — net-negative.
        self._ctx_gate = int(os.environ.get("MINISGL_SPEC_MAX_CONTEXT") or self._max_ctx)
        self._k_buf = torch.zeros(self._max_slots, self._max_ctx, _nkh, _kdim, device=dev, dtype=dt)
        self._v_buf = torch.zeros(self._max_slots, self._max_ctx, _nvh, _vdim, device=dev, dtype=dt)
        self._cur = torch.zeros(self._max_slots, dtype=torch.int64, device=dev)  # committed len/slot
        self._col_idx = torch.arange(self._max_ctx, device=dev)                  # [max_ctx]
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
            f"max_ctx={self._max_ctx}, draft-KV "
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
            write_col = (cur + j).clamp(max=self._max_ctx - 1)               # OOB backstop
            positions = (base + self._pos_shift + j).to(torch.int32)
            mask_bias = torch.where(col <= write_col.unsqueeze(1),
                                    0.0, float("-inf")).to(torch.float32)
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
        self._drafted_slots = []

    # ------------------------------------------------------------------------------ seeding etc.
    @torch.inference_mode()
    def seed_prefill(self, req: "Req", last_hidden, aux_hidden=None) -> None:
        """Seed the persistent MTP KV from the prompt prefill so the FIRST draft already sees full
        prompt context. Mirrors the decode-time convention exactly: propose processes the pair
        (embed(token_p), h_{p-1}) at RoPE position p, so we run the MTP layer's k/v over the prompt
        pairs p=1..P-1 and mark them committed; the first decode propose then appends position P.
        Slot/cursor convention matches ``stage_propose`` exactly (columns 0..S-1 hold positions
        1..P-1, ``_cur[slot] = S``), and ``_slot_uid`` is registered so that first propose does not
        treat the slot as cold and reset the cursor."""
        if last_hidden is None:
            return
        P = last_hidden.shape[0]
        if P < 2:
            return  # nothing to seed (P==1: only the bonus, handled by the first propose)
        S = P - 1
        if S > self._max_ctx:
            # Prompt longer than the draft-KV window: fall back to the cold cache (still lossless,
            # just no early-token lift). Seeding the tail would misalign the column<->position map
            # the decode chain assumes (write_col = cur+j at RoPE position base+j).
            return
        device = self._device
        tokens = req.input_ids[1:P].to(device=device, dtype=torch.int64)  # token_p, p=1..P-1
        prev_hidden = last_hidden[0 : P - 1].to(self._engine.dtype)       # h_{p-1}
        positions = torch.arange(1, P, dtype=torch.int32, device=device)
        slot = int(req.table_idx)
        self._head.seed_buffered(
            tokens, prev_hidden, positions, self._k_buf, self._v_buf, slot, 0)
        self._slot_uid[slot] = req.uid
        self._cur[slot] = S

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
