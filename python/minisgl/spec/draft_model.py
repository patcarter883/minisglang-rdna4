from __future__ import annotations

import os
from typing import TYPE_CHECKING, Dict, List, Optional

import torch

from minisgl.utils import init_logger

from .base import ProposeContext
from .capture import CapturableProposer, StagedPropose

if TYPE_CHECKING:
    from minisgl.core import Req


__all__ = ["DraftModelProposer"]

logger = init_logger(__name__)


class DraftModelProposer(CapturableProposer):
    """EAGLE3 draft-model speculative proposer.

    Owns a SEPARATE small draft checkpoint (thoughtworks/GLM-4.7-Flash-Eagle3) loaded directly from
    its safetensors — NOT through the engine's target-weight machinery, because it is a *different*
    architecture (standard Llama GQA, not the target's MLA) with its OWN compressed draft vocab. The
    draft borrows the target's token embedding (the checkpoint ships none; draft hidden == target
    hidden), runs a linear chain of K draft tokens autoregressively each step, and maps each draft id
    from the compressed draft vocab to the target vocab via `d2t` before the scheduler stages it into
    the target verify.

    `capture_layer_ids` programs the target model (glm4_moe_lite) to stash the 3 EAGLE3 aux layers'
    residual-stream hidden states during the verify forward; the scheduler feeds them back per-uid via
    `ctx.aux_hidden`. The draft fuses them (fc) into the seed feature for each step's chain.

    **Persistent draft KV.** The single draft layer's self-attention needs the full causal context,
    not just the K-token chain — restricting attention to the chain collapses the head to ~2% accept
    (near random). So this proposer keeps a PERSISTENT per-request draft KV (one layer, like the GLM
    MTP head). It lives in ONE GLOBAL fixed-shape buffer `[max_slots, ring, Hkv, hd]` keyed by
    `req.table_idx` plus a per-slot cursor, NOT a per-uid Python list: a growing list is both a
    dynamic contraction dim and a host-side mutation, i.e. two capture blockers, and it made propose
    O(prompt) in interpreted Python per draft step (the prompt seed pushed P-1 one-row tuples into
    it, and every step re-stacked the lot). Rejected drafts are not truncated — the cursor simply
    does not advance past them and next step's mask re-hides their columns.

    **This is a WINDOWED buffer over an UNWINDOWED drafter, and that is a real semantic.** Unlike
    DFlash, the EAGLE3 draft layer attends every cached key with no sliding window, so capping the
    buffer at the ring window is not a pure traffic reduction: past it the drafts genuinely differ.
    It stays end-to-end LOSSLESS because verify gates every emitted token — but an EAGLE3 A/B must be
    read as accept-len at a stated window, never as byte-equality of drafts.
    """

    needs_last_hidden = False
    supports_prefill_seed = True
    # EAGLE3's per-step seed is fc(aux) at the LAST confirmed position — one column, not a history.
    # The scheduler's aux accumulator is a global (`MINISGL_DFLASH_FULLCTX`, default on) that is NOT
    # gated on the proposer, so without this cap it grows a [num_aux, P, hidden] buffer with a
    # torch.cat per step per request for a consumer that reads exactly one column of it.
    aux_ctx_cap = 1
    # Same reasoning for the seeded-prefill store: `seed_prefill` receives the raw prompt-suffix
    # slice directly, so the per-uid buffer only ever feeds stage_propose's aux[:, -1] — keep one
    # column instead of cloning up to [num_aux, 512, hidden] per seeded request.
    prefill_aux_tail = 1

    def __init__(self, engine, num_draft: int, draft_model_path: str) -> None:
        from minisgl.models.glm_eagle3 import GLMEagle3DraftModel
        from minisgl.utils import cached_load_hf_config, download_hf_weight

        self._engine = engine
        self._num_draft = num_draft
        self._device = engine.device
        self._dtype = engine.dtype

        # Pull hidden/vocab from the loaded target — the draft borrows the target embed table.
        target_hidden = engine.model.model.embed_tokens.weight.shape[1]
        target_vocab = engine.model.model.embed_tokens.num_embeddings

        folder = download_hf_weight(draft_model_path)
        hf = cached_load_hf_config(draft_model_path)
        hidden = int(hf.hidden_size)
        assert hidden == target_hidden, (
            f"EAGLE3 draft hidden {hidden} != target hidden {target_hidden}; the draft borrows the "
            "target embed table, so the dims must match."
        )
        num_heads = int(hf.num_attention_heads)
        num_kv_heads = int(hf.num_key_value_heads)
        head_dim = int(getattr(hf, "head_dim", hidden // num_heads))
        inter = int(hf.intermediate_size)
        draft_vocab = int(hf.draft_vocab_size)
        eps = float(hf.rms_norm_eps)
        rp = getattr(hf, "rope_parameters", None) or getattr(hf, "rope_scaling", None) or {}
        rope_theta = float(rp.get("rope_theta", getattr(hf, "rope_theta", 1e6)))
        max_pos = int(hf.max_position_embeddings)

        # Target decoder layers to capture for the EAGLE3 aux fusion. Checkpoint/SGLang ids count
        # the residual stream at the INPUT of layer i (`aux.append(hidden_states + residual)` BEFORE
        # running layer i), whereas minisgl's model.forward grabs `residual` AFTER layer `lid` runs
        # (= the input to layer lid+1) — so SGLang's "input to layer i" == minisgl's "output of
        # layer i-1" and every declared id maps to a minisgl capture id MINUS 1. Prefer the ids the
        # draft checkpoint was trained against; fall back to SGLang's target-side default
        # set_eagle3_layers_to_capture() = [2, N//2, N-3] for N target layers (GLM-4.7-Flash N=47
        # -> minisgl ids [1, 22, 43]). MINISGL_EAGLE3_CAPTURE_LAYERS overrides with minisgl-
        # convention ids for GPU iteration.
        target_layers = int(cached_load_hf_config(engine.model_path).num_hidden_layers)
        ckpt_ids = (getattr(hf, "eagle_aux_hidden_state_layer_ids", None)
                    or getattr(hf, "aux_hidden_state_layer_ids", None))
        if ckpt_ids:
            ids = [int(x) - 1 for x in ckpt_ids]
        else:
            ids = [i - 1 for i in (2, target_layers // 2, target_layers - 3)]
        env_ids = os.environ.get("MINISGL_EAGLE3_CAPTURE_LAYERS")
        if env_ids:
            ids = [int(x) for x in env_ids.split(",") if x.strip()]
        assert all(0 <= i < target_layers for i in ids), (
            f"EAGLE3 capture layer ids {ids} out of range for the {target_layers}-layer target — "
            "a never-captured id fails only at the first return_hidden forward, and a wrong-depth "
            "id silently collapses accept."
        )
        self.capture_layer_ids = ids
        num_aux = len(self.capture_layer_ids)

        # Build the draft on the engine device (materialized, not meta — it is tiny).
        with torch.device(self._device):
            self._draft = GLMEagle3DraftModel(
                hidden_size=hidden,
                intermediate_size=inter,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                num_aux_layers=num_aux,
                draft_vocab_size=draft_vocab,
                target_vocab_size=target_vocab,
                rms_norm_eps=eps,
                rope_theta=rope_theta,
                max_position=max_pos,
            )
        self._load_draft_weights(folder)
        self._draft.bind_embed(engine.model.model.embed_tokens)
        # d2t (delta) on device for the draft->target id map.
        self._d2t = self._draft.d2t.to(self._device)
        self._dbg = os.environ.get("MINISGL_SPEC_DEBUG") in ("2", "3")
        # Diagnostic offsets for the step-0 embedded-token index and RoPE base position (default 0).
        self._tok_off = int(os.environ.get("MINISGL_EAGLE3_TOK_OFF", "0"))
        self._pos_off = int(os.environ.get("MINISGL_EAGLE3_POS_OFF", "0"))
        self.init_propose_capture(engine)

    # ------------------------------------------------------------------ hook: buffer allocation
    def init_propose_capture(self, engine) -> None:
        """Allocate the global draft-KV buffer, the per-slot cursors and the static propose I/O —
        once, because a captured graph records POINTERS. Mirrors MTPProposer.init_propose_capture;
        the only EAGLE3 differences are that the chain feeds BOTH an embedding and a hidden into each
        step, and that the seed comes from `ctx.aux_hidden` (fused by `fc`) rather than last_hidden."""
        d = self._draft
        _nkh, _kdim, _nvh, _vdim = d.draft_buffer_dims()
        # page_table row count == the slot space of req.table_idx; one EXTRA row is the reserved NULL
        # slot bucket-padding rows write into (a real row no live sequence owns).
        self._live_slots = int(engine.page_table.shape[0])
        self._null_slot = self._live_slots
        self._max_slots = self._live_slots + 1
        dev, dt = self._device, self._dtype
        # RING, not a linear absolute-indexed buffer — the same fix MTP already carries (spec/mtp.py).
        # Column WAS the absolute position, so the buffer capped context at its allocation and
        # `_ctx_gate` skipped propose entirely past it: EAGLE3 became a SILENT no-op on long prompts.
        # Measured on GLM-4.7-Flash under a real agent prompt (~25k tokens of system+tools before the
        # user says anything): accept-len 0.00 over 300 reqs, verify-width 0 for 100% of steps, every
        # verify eager because width 0 is not a captured rung — with nothing logged to say why.
        # A ring stores the most recent W positions at col = pos % W and masks from ABSOLUTE
        # positions, so context is unbounded at fixed VRAM. The kernel needs no change: step_masked
        # already takes an arbitrary write_col and an explicit mask_bias and never assumes
        # column == position. Same design as MTP's ring and DFlash's prefix ring.
        #
        # W=512 mirrors MTP's measured default, where a provenance-asserted sweep found W=512 matches
        # W=8192 at every context depth tested (accept / emitted-per-step / tok-s all inside the noise
        # floor) for 3 MB instead of 50 MB. EAGLE3's drafter is a different model, so if it turns out
        # to be more context-sensitive than MTP's, raise MINISGL_EAGLE3_KV_WINDOW.
        self._ring = max(64, int(os.environ.get("MINISGL_EAGLE3_KV_WINDOW") or 512))
        # Kept as an ESCAPE HATCH only (unset = unbounded). It used to default to the full-context
        # buffer length and was the silent cliff; the ring makes any length drafts-capable, so there
        # is nothing to gate.
        _gate_env = os.environ.get("MINISGL_SPEC_MAX_CONTEXT")
        self._ctx_gate = int(_gate_env) if _gate_env else (1 << 62)
        self._k_buf = torch.zeros(self._max_slots, self._ring, _nkh, _kdim, device=dev, dtype=dt)
        self._v_buf = torch.zeros(self._max_slots, self._ring, _nvh, _vdim, device=dev, dtype=dt)
        self._cur = torch.zeros(self._max_slots, dtype=torch.int64, device=dev)
        # Absolute position resident in each ring column (-1 = empty). The mask reads this, never the
        # column index, so a wrapped ring still hides the future and anything older than the window.
        self._pos_buf = torch.full((self._max_slots, self._ring), -1,
                                   dtype=torch.int64, device=dev)
        self._col_idx = torch.arange(self._ring, device=dev)
        self._slot_uid: Dict[int, int] = {}
        self._drafted_slots: List[int] = []
        hidden = int(self._draft.hidden_size)
        G = self._max_slots
        self._g_seed = torch.zeros(G, hidden, device=dev, dtype=dt)
        self._g_curb = torch.zeros(G, dtype=torch.int64, device=dev)
        self._g_out = torch.zeros(G, self._num_draft, dtype=torch.int64, device=dev)
        self._h_idx = torch.zeros(3, G, dtype=torch.int64, device="cpu", pin_memory=True)
        self._g_idx = torch.zeros(3, G, dtype=torch.int64, device=dev)
        self._g_slots, self._g_base, self._g_tok = self._g_idx[0], self._g_idx[1], self._g_idx[2]
        self.init_propose_capture_state(engine, tag="EAGLE3")
        logger.info_rank0(
            f"spec-decode: EAGLE3 propose buffers (slots={self._live_slots}+NULL, "
            f"ring={self._ring} (unbounded context), draft-KV "
            f"{(self._k_buf.numel() + self._v_buf.numel()) * dt.itemsize / 1e6:.0f} MB)")

    def _load_draft_weights(self, folder: str) -> None:
        """Load the 15-tensor EAGLE3 checkpoint directly. The checkpoint key layout (midlayer.* /
        fc / norm / lm_head / d2t / t2d) maps onto the GLMEagle3DraftModel attributes below. The
        whole draft is replicated on every TP rank (no sharding)."""
        import safetensors.torch as st

        path = None
        for f in os.listdir(folder):
            if f.endswith(".safetensors"):
                path = os.path.join(folder, f)
                break
        assert path is not None, f"no .safetensors in EAGLE3 draft folder {folder}"
        sd = st.load_file(path, device=str(self._device))

        # checkpoint key -> draft attribute path (.weight for the linears/norms).
        kmap = {
            "fc.weight": "fc.weight",
            "midlayer.input_layernorm.weight": "input_layernorm.weight",
            "midlayer.hidden_norm.weight": "hidden_norm.weight",
            "midlayer.self_attn.q_proj.weight": "q_proj.weight",
            "midlayer.self_attn.k_proj.weight": "k_proj.weight",
            "midlayer.self_attn.v_proj.weight": "v_proj.weight",
            "midlayer.self_attn.o_proj.weight": "o_proj.weight",
            "midlayer.post_attention_layernorm.weight": "post_attention_layernorm.weight",
            "midlayer.mlp.gate_proj.weight": "gate_proj.weight",
            "midlayer.mlp.up_proj.weight": "up_proj.weight",
            "midlayer.mlp.down_proj.weight": "down_proj.weight",
            "norm.weight": "norm.weight",
            "lm_head.weight": "lm_head.weight",
        }
        d = self._draft
        for ckpt_key, attr_path in kmap.items():
            assert ckpt_key in sd, f"EAGLE3 ckpt missing {ckpt_key}"
            obj_name, leaf = attr_path.split(".")
            mod = getattr(d, obj_name)
            tensor = sd[ckpt_key].to(self._dtype)
            cur = getattr(mod, leaf)
            assert cur.shape == tensor.shape, (
                f"shape mismatch for {ckpt_key}: model {tuple(cur.shape)} vs ckpt {tuple(tensor.shape)}"
            )
            setattr(mod, leaf, tensor.contiguous())
        # d2t / t2d: integer maps, not cast to model dtype.
        assert "d2t" in sd and "t2d" in sd, "EAGLE3 ckpt missing d2t/t2d"
        d.d2t = sd["d2t"].to(torch.int64)
        d.t2d = sd["t2d"].to(torch.bool)

    # ----------------------------------------------------------------------- hook: HOST staging
    def stage_propose(
        self, reqs: List["Req"], num_draft: int, ctx: ProposeContext, **kw
    ) -> Optional[StagedPropose]:
        """Pick the rows that will draft and refresh the static inputs in place. The step-0 seed is
        `fc(aux)` — the fused captured target aux at the last confirmed token — which is where EAGLE3
        differs from MTP (whose seed is the target's raw last_hidden)."""
        rows: List[int] = []
        budget: List[int] = []
        seeds: List[torch.Tensor] = []
        h = self._h_idx
        for i, req in enumerate(reqs):
            k_i = max(0, min(num_draft, req.remain_len - 1))
            aux = ctx.aux_hidden.get(req.uid)  # [num_aux, hidden] at the last confirmed token
            if k_i <= 0 or aux is None or req.cached_len > self._ctx_gate:
                continue
            if aux.dim() == 3:
                # The scheduler's full-context accumulator hands out [num_aux, P, hidden] for EVERY
                # proposer (`MINISGL_DFLASH_FULLCTX` is not gated on the proposer being DFlash), but
                # EAGLE3 conditions on the LAST confirmed position only. The old code fed the whole
                # 3-D buffer into `fc` via `unsqueeze(0)`, which is a hard shape error — EAGLE3 could
                # not serve a single token on this build. Take the last column, which is exactly the
                # legacy 2-D value (`aux_hidden[:, row]`).
                aux = aux[:, -1]
            tok_idx = max(0, min(req.cached_len + self._tok_off, req.input_ids.shape[0] - 1))
            s = int(req.table_idx)
            if self._slot_uid.get(s) != req.uid:   # fresh req on this slot -> cold cache
                self._slot_uid[s] = req.uid
                self._cur[s] = 0
                # The ring mask keys off ABSOLUTE positions, so a cursor reset alone is not enough:
                # the previous owner's positions would still satisfy `pa <= qa`. Invalidate the ring.
                self._pos_buf[s].fill_(-1)
            j = len(rows)
            h[0, j] = s
            h[1, j] = int(req.cached_len) + self._pos_off
            h[2, j] = int(req.input_ids[tok_idx])
            rows.append(i)
            budget.append(k_i)
            seeds.append(aux.reshape(-1))          # [num_aux*hidden], fc'd on device in the body
        self._drafted_slots = [int(reqs[i].table_idx) for i in rows]
        if not rows:
            return None
        B = len(rows)
        self._g_idx[:, :B].copy_(h[:, :B], non_blocking=True)
        self._g_seed[:B].copy_(
            self._draft.fc.forward(torch.stack(seeds).to(self._dtype)))
        self._g_curb[:B].copy_(self._cur[self._g_slots[:B]])
        return StagedPropose(B, rows, budget)

    def pad_propose_rows(self, bs: int, bucket: int) -> None:
        self._g_slots[bs:bucket].fill_(self._null_slot)
        self._g_base[bs:bucket].zero_()
        self._g_tok[bs:bucket].zero_()
        self._g_curb[bs:bucket].zero_()
        self._g_seed[bs:bucket].zero_()

    # ------------------------------------------------------------------------ hook: the BODY
    def propose_body(self, bs: int) -> None:
        """The K-step autoregressive EAGLE3 chain over the STATIC buffers [:bs]. Step 0 processes the
        confirmed token paired with fc(aux); step j the previous draft paired with the draft layer's
        OWN output hidden. The d2t (draft->target) remap stays ON DEVICE, so the whole chain runs
        with no host round-trip — which is both a latency property and a capture requirement."""
        d = self._draft
        slots = self._g_slots[:bs]
        base = self._g_base[:bs]
        cur = self._g_curb[:bs]
        cur_tok = self._g_tok[:bs]
        cur_hidden = self._g_seed[:bs]
        for j in range(self._num_draft):
            q_abs = cur + j                                        # ABSOLUTE position being written
            write_col = torch.remainder(q_abs, self._ring)         # ring slot
            positions = (base + j).to(torch.int32)
            # Publish this token's position BEFORE masking so the row it is about to write is visible
            # to its own attention (the old `col <= write_col` mask included write_col).
            self._pos_buf[slots, write_col] = q_abs
            pa = self._pos_buf[slots]                              # [bs, ring] abs pos per column
            qa = q_abs.unsqueeze(1)
            # Causal + in-window, keyed on ABSOLUTE positions: hides empty columns (-1), the future,
            # and anything the ring has already overwritten. Column order is meaningless once wrapped.
            keep = (pa >= 0) & (pa <= qa) & ((qa - pa) < self._ring)
            mask_bias = torch.where(keep, 0.0, float("-inf")).to(torch.float32)
            logits, cur_hidden = d.step_masked(
                d.embed(cur_tok), cur_hidden, positions,
                self._k_buf, self._v_buf, slots, write_col, mask_bias)
            draft_id = logits.argmax(dim=-1)                        # compressed draft vocab
            target_id = draft_id + self._d2t[draft_id]              # -> target vocab, on device
            self._g_out[:bs, j] = target_id
            cur_tok = target_id

    # ------------------------------------------------------------------- hook: the ONE host sync
    def read_drafts(self, reqs: List["Req"], staged: StagedPropose, **kw) -> List[List[int]]:
        out: List[List[int]] = [[] for _ in reqs]
        drafts = self._g_out[: staged.bs].cpu().tolist()   # ONE D2H for the whole step
        for i, k_i, dd in zip(staged.rows, staged.budget, drafts):
            out[i] = dd[:k_i]
            if self._dbg:
                print(f"[eagle3-dbg] uid={reqs[i].uid} k={k_i} draft={out[i]}", flush=True)
        return out

    def reset_propose_state(self) -> None:
        self._cur[self._null_slot] = 0
        self._pos_buf[self._null_slot].fill_(-1)
        self._drafted_slots = []

    @torch.inference_mode()
    def seed_prefill(self, req: "Req", last_hidden=None, aux_hidden=None) -> None:
        """Seed the persistent draft KV from the prompt prefill so the FIRST draft already sees the
        prompt context (otherwise the cache starts empty -> cold early tokens, the ~2% floor's milder
        cousin). Same convention as the decode-time chain: the pair (embed(token_p), fc(aux_{p-1}))
        lives at RoPE position p, writing at ring column p % ring with absolute label p.
        ``aux_hidden`` holds the rows for absolute positions [cached_len-P, cached_len) — on a radix
        prefix-cache hit that origin is > 0 and the rows are the prompt SUFFIX, not its start
        (complete_one has already advanced cached_len to the full prompt length; same
        ``ctx_start = cached_len - P`` convention as DFlash). ``_cur[slot]`` is the ABSOLUTE index
        of the next pair to write, so it is set to cached_len and the first decode propose APPENDS
        the bonus pair (token_{cached_len}, RoPE cached_len) at column cached_len % ring instead of
        clobbering the last seeded pair."""
        if aux_hidden is None:
            return
        P = aux_hidden.shape[1]  # aux_hidden: [num_aux, P, hidden]
        if P < 2:
            return
        device = self._device
        slot = int(req.table_idx)
        end = int(req.cached_len)                        # one past the last seeded pair
        origin = end - P                                 # absolute position of aux_hidden[:, 0]
        assert origin >= 0, f"seed_prefill: {P} aux rows exceed cached_len {end}"
        # Seed the ring TAIL. The old code REFUSED when the prompt exceeded the buffer and left the
        # cache cold, which — together with the _ctx_gate that has now gone — is what made long
        # prompts draft blind. With col = pos % ring the tail is always representable; it just may
        # WRAP, so write it as up to two contiguous runs. Pair `origin` itself is unseedable on a
        # cache hit (aux_{origin-1} was never recomputed).
        p_lo = max(origin + 1, end - self._ring)         # first prompt position kept in the ring
        tokens = req.input_ids[p_lo:end].to(device=device, dtype=torch.int64)  # token_p
        aux_prev = (aux_hidden[:, p_lo - 1 - origin : end - 1 - origin]
                    .to(self._dtype).permute(1, 0, 2).contiguous())
        fused = self._draft.fuse_aux(aux_prev)           # [end-p_lo, hidden]
        positions = torch.arange(p_lo, end, dtype=torch.int32, device=device)
        n = int(positions.numel())
        if n > 0:
            c0 = p_lo % self._ring
            first = min(n, self._ring - c0)              # contiguous run before the wrap
            self._draft.seed_buffered(
                self._draft.embed(tokens[:first]), fused[:first], positions[:first],
                self._k_buf, self._v_buf, slot, c0)
            if first < n:
                self._draft.seed_buffered(
                    self._draft.embed(tokens[first:]), fused[first:], positions[first:],
                    self._k_buf, self._v_buf, slot, 0)
            abs_pos = torch.arange(p_lo, end, dtype=torch.int64, device=device)
            self._pos_buf[slot, torch.remainder(abs_pos, self._ring)] = abs_pos
        self._slot_uid[slot] = req.uid
        self._cur[slot] = end

    def on_accept(self, reqs: List["Req"], num_accepted: List[int]) -> None:
        # The confirmed token (always committed) plus the n accepted drafts become permanent draft
        # context. Columns written per step are [confirmed, d0 .. d_{K-2}] — d_{K-1}'s K/V is never
        # written — so cap the advance at K, exactly as the eager list path capped `committed` at
        # len(cache). The K-n rejected drafts need no truncation: the cursor does not reach them and
        # next step's mask re-hides their columns.
        drafted = set(self._drafted_slots)
        for req, n in zip(reqs, num_accepted):
            s = int(req.table_idx)
            if s in drafted:
                self._cur[s] = self._cur[s] + min(1 + n, self._num_draft)

    def free(self, uid: int) -> None:
        for s, u in list(self._slot_uid.items()):
            if u == uid:
                del self._slot_uid[s]
                self._cur[s] = 0
                self._pos_buf[s].fill_(-1)   # stale absolute positions must not survive the slot
