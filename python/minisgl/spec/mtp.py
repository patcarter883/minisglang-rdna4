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

    @torch.inference_mode()
    def propose(self, reqs: List["Req"], num_draft: int, ctx: ProposeContext) -> List[List[int]]:
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
            drafts: List[int] = []
            # Step 0 processes the confirmed token (appends its K/V to the persistent cache and
            # predicts d0); steps 1.. process each draft. The attention sees the full cache (confirmed
            # prefix built over prior steps + this chain), so the MTP head runs with real context.
            for step in range(k_i):
                fused = head.fuse(head.embed(cur_tok), cur_hidden)
                positions = torch.tensor(
                    [base_pos + step + self._pos_shift], dtype=torch.int32, device=device
                )
                logits, cur_hidden = head.step(fused, positions, cache, len(cache))
                nxt = int(logits.argmax(dim=-1).item())
                drafts.append(nxt)
                cur_tok = torch.tensor([nxt], dtype=torch.int64, device=device)
            out[i] = drafts
            if dbg:
                print(f"[mtp-dbg] uid={req.uid} conf={conf_tok} base_pos={base_pos} "
                      f"ctx={committed} draft={drafts}", flush=True)
        return out

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
        entries = self._head.seed_kv(tokens, prev_hidden, positions)
        self._cache[req.uid] = entries
        self._committed[req.uid] = len(entries)

    def on_accept(self, reqs: List["Req"], num_accepted: List[int]) -> None:
        # The confirmed token (always committed) plus the n accepted drafts become permanent MTP
        # context; the K-n rejected drafts' K/V are dropped. The cache entry layout per step is
        # [confirmed, d0, d1, ...]; committing 1 + n keeps confirmed + the accepted run.
        for req, n in zip(reqs, num_accepted):
            if req.uid in self._cache:
                self._committed[req.uid] = self._committed.get(req.uid, 0) + 1 + n
                # Eagerly trim so a finished/aborted-then-reused uid never inherits stale tail.
                del self._cache[req.uid][self._committed[req.uid]:]

    def free(self, uid: int) -> None:
        self._cache.pop(uid, None)
        self._committed.pop(uid, None)
