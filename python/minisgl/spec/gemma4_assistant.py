"""Gemma-4 assistant (MTP) proposer: drafts with a `gemma4_assistant` checkpoint that reads the
target's KV cache instead of keeping its own (models/gemma4_assistant.py).

Per request, step 0 feeds (the confirmed token, the target's hidden at the position before it); step
j feeds (draft j-1, the drafter's projected hidden). Every step queries from position L-1 over the
target's keys [0, L), L = req.cached_len, so there is no draft-side state to seed, roll back or free.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, List, Optional

import torch

from minisgl.utils import init_logger

from .base import ProposeContext
from .capture import CapturableProposer, StagedPropose

if TYPE_CHECKING:
    from minisgl.core import Req

__all__ = ["Gemma4AssistantProposer", "is_gemma4_assistant"]

logger = init_logger(__name__)


def is_gemma4_assistant(draft_model_path: Optional[str]) -> bool:
    if not draft_model_path:
        return False
    from minisgl.utils import cached_load_hf_config

    return getattr(cached_load_hf_config(draft_model_path), "model_type", "") == "gemma4_assistant"


class Gemma4AssistantProposer(CapturableProposer):
    needs_last_hidden = True
    # No draft KV to seed; this makes the scheduler keep the prompt's last hidden so the first
    # decode step already drafts.
    supports_prefill_seed = True

    def __init__(self, engine, num_draft: int, draft_model_path: str) -> None:
        import attn_decode
        import safetensors.torch as st

        from minisgl.models.gemma4_assistant import Gemma4AssistantDraft, _AttnTarget
        from minisgl.utils import cached_load_hf_config, download_hf_weight

        self._engine = engine
        self._num_draft = num_draft
        self._device = engine.device
        target = engine.model
        backend = engine.attn_backend
        hf = cached_load_hf_config(draft_model_path)

        # Each drafter layer reads the target's LAST layer of the same attention type.
        layers = target.model.layers.op_list
        last = {}
        for layer in layers:
            plan = layer.self_attn.plan
            last["sliding_attention" if plan.is_sliding else "full_attention"] = layer
        types = list(hf.text_config.layer_types)
        missing = {t for t in types if t not in last}
        if missing:
            raise ValueError(f"gemma4_assistant: target has no {sorted(missing)} layer to read KV from")

        def attn_target(layer_type):
            attn = last[layer_type].self_attn
            plan = attn.plan
            pool = engine.swa_kv_cache if plan.is_sliding else engine.kv_cache
            is_fp8 = pool.dtype == torch.float8_e4m3fn
            return _AttnTarget(pool, plan.kv_id, attn.attn.rotary, plan.head_dim, is_fp8)

        self._sliding = attn_target("sliding_attention") if "sliding_attention" in types else None
        self._full = attn_target("full_attention") if "full_attention" in types else None
        probe = lambda hd: torch.empty(1, 1, hd)   # noqa: E731 - scale depends on head_dim only
        scales = {backend._softmax_scale(probe(t.head_dim)) for t in (self._sliding, self._full) if t}
        assert len(scales) == 1, f"gemma4_assistant: per-type softmax scales differ {scales}"
        self._scale = scales.pop()
        self._decode = (attn_decode.flash_decode_paged, attn_decode.flash_decode_paged_fp8)
        self._backend = backend
        self._embed = target.model._embed_scaled

        with torch.device(self._device):
            self._draft = Gemma4AssistantDraft(hf, types)
        folder = download_hf_weight(draft_model_path)
        sd = {}
        for f in sorted(os.listdir(folder)):
            if f.endswith(".safetensors"):
                sd.update(st.load_file(os.path.join(folder, f), device="cpu"))
        self._draft.load(sd, self._device, engine.dtype)
        if self._draft.backbone != target.model.embed_tokens.weight.shape[1]:
            raise ValueError(
                f"gemma4_assistant backbone_hidden_size {self._draft.backbone} != target hidden "
                f"{target.model.embed_tokens.weight.shape[1]}")
        self.verify_hidden_size = self._draft.backbone
        self._dbg = os.environ.get("MINISGL_SPEC_DEBUG") in ("2", "3")
        self.init_propose_capture(engine)

    # ------------------------------------------------------------------ hook: buffer allocation
    def init_propose_capture(self, engine) -> None:
        dev = self._device
        self._null_slot = int(engine.page_table.shape[0])
        G = self._null_slot + 1
        self._ps = int(self._backend.page_size)
        self._max_pages = (int(engine.page_table.shape[1]) + self._ps - 1) // self._ps
        self._W = int(self._backend.swa_window) if self._sliding is not None else 1
        self._g_seed = torch.zeros(G, self._draft.backbone, device=dev, dtype=engine.dtype)
        self._g_tok = torch.zeros(G, dtype=torch.int64, device=dev)
        self._g_pos = torch.zeros(G, dtype=torch.int64, device=dev)
        self._g_out = torch.zeros(G, self._num_draft, dtype=torch.int64, device=dev)
        self._g_full_bt = torch.zeros(G, self._max_pages, dtype=torch.int32, device=dev)
        self._g_full_len = torch.ones(G, dtype=torch.int32, device=dev)
        self._g_swa_bt = torch.zeros(G, self._W, dtype=torch.int32, device=dev)
        self._g_swa_len = torch.ones(G, dtype=torch.int32, device=dev)
        self._cols = torch.arange(self._W, dtype=torch.int64, device=dev)
        self.init_propose_capture_state(engine, tag="Gemma4-assistant")
        logger.info_rank0(
            f"spec-decode: Gemma-4 assistant drafter ({len(self._draft.layers)} layers, hidden "
            f"{self._draft.hidden}, K={self._num_draft}) reads target KV "
            f"(sliding window {self._W}, page {self._ps})")

    # ----------------------------------------------------------------------- hook: HOST staging
    def stage_propose(
        self, reqs: List["Req"], num_draft: int, ctx: ProposeContext, **kw
    ) -> Optional[StagedPropose]:
        rows, budget, seeds, tbl, lens, toks = [], [], [], [], [], []
        for i, req in enumerate(reqs):
            k_i = max(0, min(num_draft, req.remain_len - 1))
            seed = ctx.last_hidden.get(req.uid) if ctx.last_hidden else None
            if k_i <= 0 or seed is None or req.cached_len < 1:
                continue
            rows.append(i)
            budget.append(k_i)
            seeds.append(seed.view(-1))
            tbl.append(int(req.table_idx))
            lens.append(int(req.cached_len))
            toks.append(int(req.input_ids[req.cached_len]))
        if not rows:
            return None
        B = len(rows)
        dev = self._device
        host = torch.tensor([tbl, lens, toks], dtype=torch.int64, pin_memory=True)
        idx = host.to(dev, non_blocking=True)
        t, L = idx[0], idx[1]
        self._g_tok[:B].copy_(idx[2])
        self._g_pos[:B].copy_(L - 1)
        self._g_seed[:B].copy_(torch.stack(seeds).to(self._g_seed.dtype))
        if self._full is not None:
            gpt = self._engine.page_table
            pages = gpt[t, : self._max_pages * self._ps : self._ps]
            if self._ps > 1:
                pages = torch.div(pages, self._ps, rounding_mode="floor")
            self._g_full_bt[:B, : pages.shape[1]].copy_(pages.to(torch.int32))
            self._g_full_len[:B].copy_(L.to(torch.int32))
        if self._sliding is not None:
            R = int(self._backend.swa_ring_stride)
            cnt = torch.clamp(L, max=self._W)
            slots = t[:, None] * R + torch.remainder((L - cnt)[:, None] + self._cols[None, :], R)
            self._g_swa_bt[:B].copy_(slots.to(torch.int32))
            self._g_swa_len[:B].copy_(cnt.to(torch.int32))
        return StagedPropose(B, rows, budget)

    def pad_propose_rows(self, bs: int, bucket: int) -> None:
        self._g_tok[bs:bucket].zero_()
        self._g_pos[bs:bucket].zero_()
        self._g_seed[bs:bucket].zero_()
        self._g_full_bt[bs:bucket].zero_()
        self._g_full_len[bs:bucket].fill_(1)
        self._g_swa_bt[bs:bucket].zero_()
        self._g_swa_len[bs:bucket].fill_(1)

    # ------------------------------------------------------------------------ hook: the BODY
    def propose_body(self, bs: int) -> None:
        attn = (
            (self._sliding, self._g_swa_bt[:bs], self._g_swa_len[:bs]),
            (self._full, self._g_full_bt[:bs], self._g_full_len[:bs]),
        )
        pos = self._g_pos[:bs]
        tok = self._g_tok[:bs]
        seed = self._g_seed[:bs]
        for j in range(self._num_draft):
            tok, seed = self._draft.step(self._embed(tok), seed, pos, attn, self._decode, self._scale)
            self._g_out[:bs, j] = tok

    # ------------------------------------------------------------------- hook: the ONE host sync
    def read_drafts(self, reqs: List["Req"], staged: StagedPropose, **kw) -> List[List[int]]:
        out: List[List[int]] = [[] for _ in reqs]
        drafts = self._g_out[: staged.bs].cpu().tolist()
        for i, k_i, d in zip(staged.rows, staged.budget, drafts):
            out[i] = d[:k_i]
            if self._dbg:
                print(f"[g4a-dbg] uid={reqs[i].uid} k={k_i} draft={out[i]}", flush=True)
        return out
