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


# Target GLM-4.7-Flash decoder layers to capture for EAGLE3 aux fusion.
#
# SGLang's target-side default set_eagle3_layers_to_capture() = [2, N//2, N-3] for
# N=num_hidden_layers; GLM-4.7-Flash N=47 -> SGLang ids [2, 23, 44]. BUT SGLang captures the residual
# stream at the INPUT of layer i (`aux.append(hidden_states + residual)` BEFORE running layer i),
# whereas minisgl's GLMModel.forward grabs `residual` AFTER layer `lid` runs (= the input to layer
# lid+1). So SGLang's "input to layer i" == minisgl's "output of layer i-1" -> the equivalent minisgl
# capture ids are [1, 22, 43]. Override via env for GPU iteration.
_GLM47_CAPTURE_LAYER_IDS = [
    int(x) for x in os.environ.get("MINISGL_EAGLE3_CAPTURE_LAYERS", "1,22,43").split(",")
]


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
    MTP head). It lives in ONE GLOBAL fixed-shape buffer `[max_slots, max_ctx, Hkv, hd]` keyed by
    `req.table_idx` plus a per-slot cursor, NOT a per-uid Python list: a growing list is both a
    dynamic contraction dim and a host-side mutation, i.e. two capture blockers, and it made propose
    O(prompt) in interpreted Python per draft step (the prompt seed pushed P-1 one-row tuples into
    it, and every step re-stacked the lot). Rejected drafts are not truncated — the cursor simply
    does not advance past them and next step's mask re-hides their columns.

    **This is a WINDOWED buffer over an UNWINDOWED drafter, and that is a real semantic.** Unlike
    DFlash, the EAGLE3 draft layer attends every cached key with no sliding window, so capping the
    buffer at `max_ctx` is not a pure traffic reduction: past the window the drafts genuinely differ.
    It stays end-to-end LOSSLESS because verify gates every emitted token — but an EAGLE3 A/B must be
    read as accept-len at a stated window, never as byte-equality of drafts.
    """

    needs_last_hidden = False
    capture_layer_ids = _GLM47_CAPTURE_LAYER_IDS
    supports_prefill_seed = True

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
        from minisgl.engine.graph import get_free_memory

        per_col = self._max_slots * (_nkh * _kdim + _nvh * _vdim) * dt.itemsize
        # Budget the persistent buffer AND the per-step k_buf[slot_rows] gather (a second transient
        # of comparable size at full batch) — the same sizing MTP arrived at after a bs>=4 OOM.
        _budget = int(get_free_memory(dev) * float(os.environ.get("MINISGL_EAGLE3_KV_FRAC", "0.33")))
        _mem_cap = max(512, _budget // max(per_col * 2, 1))
        self._max_ctx = min(int(engine.max_seq_len),
                            int(os.environ.get("MINISGL_EAGLE3_MAX_CTX") or "8192"),
                            int(_mem_cap))
        # Past this committed length a request skips propose and decodes plain: above the window the
        # seed fell back to cold, so drafts are context-blind while still paying propose + verify.
        self._ctx_gate = int(os.environ.get("MINISGL_SPEC_MAX_CONTEXT") or self._max_ctx)
        self._k_buf = torch.zeros(self._max_slots, self._max_ctx, _nkh, _kdim, device=dev, dtype=dt)
        self._v_buf = torch.zeros(self._max_slots, self._max_ctx, _nvh, _vdim, device=dev, dtype=dt)
        self._cur = torch.zeros(self._max_slots, dtype=torch.int64, device=dev)
        self._col_idx = torch.arange(self._max_ctx, device=dev)
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
            f"max_ctx={self._max_ctx}, draft-KV "
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
            tok_idx = max(0, min(req.cached_len + self._tok_off, req.input_ids.shape[0] - 1))
            s = int(req.table_idx)
            if self._slot_uid.get(s) != req.uid:   # fresh req on this slot -> cold cache
                self._slot_uid[s] = req.uid
                self._cur[s] = 0
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
        col = self._col_idx.unsqueeze(0)
        for j in range(self._num_draft):
            write_col = (cur + j).clamp(max=self._max_ctx - 1)      # OOB backstop
            positions = (base + j).to(torch.int32)
            mask_bias = torch.where(col <= write_col.unsqueeze(1),
                                    0.0, float("-inf")).to(torch.float32)
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
        self._drafted_slots = []

    @torch.inference_mode()
    def seed_prefill(self, req: "Req", last_hidden=None, aux_hidden=None) -> None:
        """Seed the persistent draft KV from the prompt prefill so the FIRST draft already sees full
        prompt context (otherwise the cache starts empty -> cold early tokens, the ~2% floor's milder
        cousin). Same convention as the decode-time chain: the pair (embed(token_p), fc(aux_{p-1}))
        lives at RoPE position p, so we write the draft layer's k/v for prompt pairs p=1..P-1 into
        columns 0..S-1 of this req's slot and set the cursor to S; the first decode propose then
        appends position P (the bonus) at column S."""
        if aux_hidden is None:
            return
        P = aux_hidden.shape[1]  # aux_hidden: [num_aux, P, hidden]
        if P < 2:
            return
        S = P - 1
        if S > self._max_ctx:
            # Prompt longer than the draft-KV window: fall back to the cold cache (lossless, just no
            # early-token lift). Seeding the tail would misalign the column<->position map the chain
            # assumes (write_col = cur+j at RoPE position base+j).
            return
        device = self._device
        tokens = req.input_ids[1:P].to(device=device, dtype=torch.int64)  # token_p, p=1..P-1
        aux_prev = aux_hidden[:, 0 : P - 1].to(self._dtype).permute(1, 0, 2).contiguous()
        fused = self._draft.fuse_aux(aux_prev)  # [P-1, hidden]
        positions = torch.arange(1, P, dtype=torch.int32, device=device)
        slot = int(req.table_idx)
        self._draft.seed_buffered(
            self._draft.embed(tokens), fused, positions, self._k_buf, self._v_buf, slot, 0)
        self._slot_uid[slot] = req.uid
        self._cur[slot] = S

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
