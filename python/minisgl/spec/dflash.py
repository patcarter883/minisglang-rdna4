from __future__ import annotations

import os
from typing import TYPE_CHECKING, List, Optional

import torch

from minisgl.utils import init_logger

from .base import Proposer, ProposeContext

logger = init_logger(__name__)

if TYPE_CHECKING:
    from minisgl.core import Req


__all__ = ["DFlashProposer"]


class DFlashProposer(Proposer):
    """DFlash block-diffusion draft proposer.

    Unlike MTP/EAGLE3 (autoregressive K-token chains), DFlash drafts a whole BLOCK in ONE bidirectional
    "denoising" forward (block-diffusion). Per request, each step:

      1. fuse the captured target aux concat -> target_hidden = hidden_norm(fc(concat))  (KV prefix);
      2. build block_output_ids = [anchor, mask, mask, ...] of length B = block_size, where anchor is
         the just-confirmed token and mask is mask_token_id; noise_embed = embed(block_output_ids);
      3. ONE forward (DFlashDraftModel.denoise) -> [B, hidden]; head -> [B, vocab];
      4. drafts = argmax(logits[1:])  -> B-1 candidate tokens (position 0 is the known anchor).
         Compressed-vocab (Checkpoint B): argmax over draft vocab j -> target id j + d2t[j].

    The captured target hidden states ARE the cross-block context, injected as the per-layer KV prefix.
    FULL-CONTEXT conditioning (the z-lab fix, MINISGL_DFLASH_FULLCTX=1, default): the scheduler
    accumulates the target aux over ALL committed positions and feeds the whole prefix ([num_aux, P,
    hidden], P = committed length) at its true absolute RoPE positions, so the drafter attends over the
    entire context it was trained on. Feeding only the last-token vector (legacy, FULLCTX=0) ran the
    drafter out-of-distribution → ~0.33 accept-len. There is still NO persistent draft KV — the full
    prefix is recomputed each step from the accumulated aux — so `on_accept` is a no-op and verification
    stays the existing linear verify_greedy (DFlash is a linear block, not a tree). Perf follow-up: a
    persistent draft KV (z-lab's crop-per-step) would make this O(new) instead of O(P) per block.

    `capture_layer_ids` programs the target model to stash the configured decoder layers' residual-
    stream hidden during the verify forward; the scheduler feeds them back per-uid via `ctx.aux_hidden`.

    Env knobs (GPU iteration): MINISGL_DFLASH_CAPTURE_LAYERS overrides the captured target-layer ids;
    MINISGL_DFLASH_BLOCK caps the block size (drafts emitted = min(block-1, num_draft, budget));
    MINISGL_DFLASH_POS_OFF shifts the anchor RoPE base; MINISGL_DFLASH_CTX_POS sets the prefix position.
    """

    needs_last_hidden = False
    capture_layer_ids: Optional[List[int]] = None  # set in __init__ from the ckpt config

    def __init__(self, engine, num_draft: int, draft_model_path: str) -> None:
        from minisgl.models.dflash import DFlashDraftModel
        from minisgl.utils import cached_load_hf_config, download_hf_weight

        self._engine = engine
        self._num_draft = num_draft
        self._device = engine.device
        self._dtype = engine.dtype

        folder = download_hf_weight(draft_model_path)
        hf = cached_load_hf_config(draft_model_path)

        # Two config dialects: z-lab (dflash_config) and speculators (top-level keys). Normalize.
        dfc = getattr(hf, "dflash_config", None) or {}
        speculators_type = getattr(hf, "speculators_model_type", None)
        tlc = getattr(hf, "transformer_layer_config", None) or {}

        def cfg(*names, default=None):
            for src in (dfc, tlc, hf.__dict__ if hasattr(hf, "__dict__") else {}):
                for n in names:
                    if isinstance(src, dict) and n in src:
                        return src[n]
                    if not isinstance(src, dict) and hasattr(src, n):
                        return getattr(src, n)
            return default

        hidden = int(cfg("hidden_size"))
        num_layers = int(cfg("num_hidden_layers"))
        num_heads = int(cfg("num_attention_heads"))
        num_kv_heads = int(cfg("num_key_value_heads"))
        head_dim = int(cfg("head_dim", default=hidden // num_heads))
        inter = int(cfg("intermediate_size"))
        eps = float(cfg("rms_norm_eps", default=1e-6))
        rp = getattr(hf, "rope_parameters", None) or getattr(hf, "rope_scaling", None) or {}
        rope_theta = (rp.get("rope_theta") if isinstance(rp, dict) else None) or cfg(
            "rope_theta", default=None
        )
        if rope_theta is None:
            # Fallback: some config dialects nest rope_theta under rope_parameters and transformers
            # does not surface it as a flat attr — read the raw config.json.
            import json as _json

            raw = _json.load(open(os.path.join(folder, "config.json"))) if os.path.exists(
                os.path.join(folder, "config.json")
            ) else {}
            rope_theta = (raw.get("rope_parameters") or {}).get("rope_theta") or raw.get(
                "rope_theta"
            ) or 1e6
        rope_theta = float(rope_theta)
        max_pos = int(cfg("max_position_embeddings", default=262144))

        self._block_size = int(dfc.get("block_size") or getattr(hf, "block_size", 0) or 0)
        assert self._block_size >= 2, f"DFlash block_size must be >= 2, got {self._block_size}"
        self._mask_token_id = int(
            dfc.get("mask_token_id") if "mask_token_id" in dfc else getattr(hf, "mask_token_id")
        )
        block_cap = int(os.environ.get("MINISGL_DFLASH_BLOCK", "0") or 0)
        if block_cap:
            self._block_size = min(self._block_size, block_cap)

        # Captured target-layer ids (z-lab target_layer_ids / speculators aux_hidden_state_layer_ids).
        ids = dfc.get("target_layer_ids") or getattr(hf, "aux_hidden_state_layer_ids", None)
        env_ids = os.environ.get("MINISGL_DFLASH_CAPTURE_LAYERS")
        if env_ids:
            ids = [int(x) for x in env_ids.split(",") if x.strip() != ""]
        assert ids, "DFlash ckpt has no target_layer_ids / aux_hidden_state_layer_ids"
        self.capture_layer_ids = [int(x) for x in ids]
        num_aux = len(self.capture_layer_ids)

        # Pruned-vocab variant (Checkpoint B) ships its own embed/lm_head over a compressed draft vocab.
        draft_vocab = int(getattr(hf, "draft_vocab_size", 0) or 0)
        tied = bool(cfg("tie_word_embeddings", default=False)) or draft_vocab == 0
        self._compressed = draft_vocab > 0 and not tied

        target_hidden = engine.model.model.embed_tokens.weight.shape[1]
        assert hidden == target_hidden, (
            f"DFlash draft hidden {hidden} != target hidden {target_hidden}; the tied-vocab variant "
            "borrows the target embed/head, so dims must match."
        )

        with torch.device(self._device):
            self._draft = DFlashDraftModel(
                hidden_size=hidden,
                intermediate_size=inter,
                num_layers=num_layers,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                num_aux_layers=num_aux,
                rms_norm_eps=eps,
                rope_theta=rope_theta,
                max_position=max_pos,
                draft_vocab_size=draft_vocab if self._compressed else None,
                own_embed=self._compressed,
            )
        self._load_draft_weights(folder)

        if self._compressed:
            self._d2t = self._draft.d2t.to(self._device)
        else:
            # Tied-vocab: borrow the target embed table + lm_head.
            self._draft.bind_embed(engine.model.model.embed_tokens)
            self._draft.bind_lm_head(engine.model.lm_head)
            self._d2t = None

        self._dbg = os.environ.get("MINISGL_SPEC_DEBUG") in ("2", "3")
        self._pos_off = int(os.environ.get("MINISGL_DFLASH_POS_OFF", "0"))
        # Position of the target-context KV prefix. Default = the anchor's own position (so the prefix
        # carries the right RoPE phase); override for diagnostics.
        self._ctx_pos_env = os.environ.get("MINISGL_DFLASH_CTX_POS")

    def _load_draft_weights(self, folder: str) -> None:
        """Load the DFlash checkpoint directly. fc/hidden_norm/norm + per-layer Qwen3 decoder tensors
        map onto the DFlashDraftModel attributes; the whole draft is replicated on every TP rank."""
        import safetensors.torch as st

        path = None
        for f in sorted(os.listdir(folder)):
            if f.endswith(".safetensors"):
                path = os.path.join(folder, f)
                break
        assert path is not None, f"no .safetensors in DFlash draft folder {folder}"
        # MINISGL_DFLASH_QUANT=fp8|int8: weight-only quantize the draft LINEARS to ~half memory (fits
        # a big drafter replicated on every 16 GB card). Load to CPU so the fp16 transient stays in
        # host RAM and only the 1-byte packed weight lands on GPU (the load-time OOM peak is the issue).
        quant_mode = (os.environ.get("MINISGL_DFLASH_QUANT", "") or "").strip().lower() or None
        if quant_mode not in (None, "fp8", "int8"):
            raise ValueError(f"MINISGL_DFLASH_QUANT must be fp8|int8, got {quant_mode!r}")
        sd = st.load_file(path, device="cpu" if quant_mode else str(self._device))
        d = self._draft
        if quant_mode:
            logger.info_rank0(f"DFlash drafter: weight-only {quant_mode} quant of the draft linears")

        def assign(mod, leaf, key, quant=False):
            assert key in sd, f"DFlash ckpt missing {key}"
            t = sd[key].to(self._dtype).contiguous()
            cur = getattr(mod, leaf)
            assert cur.shape == t.shape, (
                f"shape mismatch {key}: model {tuple(cur.shape)} vs ckpt {tuple(t.shape)}"
            )
            if quant and quant_mode:
                mod.load_quant(t, quant_mode, self._dtype, self._device)  # t on CPU -> packed on GPU
            else:
                setattr(mod, leaf, t.to(self._device))

        assign(d.fc, "weight", "fc.weight", quant=True)
        assign(d.hidden_norm, "weight", "hidden_norm.weight")
        assign(d.norm, "weight", "norm.weight")
        for i, layer in enumerate(d.layers):
            p = f"layers.{i}."
            assign(layer.input_layernorm, "weight", p + "input_layernorm.weight")
            assign(layer.post_attention_layernorm, "weight", p + "post_attention_layernorm.weight")
            assign(layer.q_proj, "weight", p + "self_attn.q_proj.weight", quant=True)
            assign(layer.k_proj, "weight", p + "self_attn.k_proj.weight", quant=True)
            assign(layer.v_proj, "weight", p + "self_attn.v_proj.weight", quant=True)
            assign(layer.o_proj, "weight", p + "self_attn.o_proj.weight", quant=True)
            assign(layer.q_norm, "weight", p + "self_attn.q_norm.weight")
            assign(layer.k_norm, "weight", p + "self_attn.k_norm.weight")
            assign(layer.gate_proj, "weight", p + "mlp.gate_proj.weight", quant=True)
            assign(layer.up_proj, "weight", p + "mlp.up_proj.weight", quant=True)
            assign(layer.down_proj, "weight", p + "mlp.down_proj.weight", quant=True)

        if self._compressed:
            assert "embed_tokens.weight" in sd, "compressed DFlash ckpt missing embed_tokens"
            d.set_own_embed(sd["embed_tokens.weight"].to(self._dtype).contiguous().to(self._device))
            assign(d._own_lm_head, "weight", "lm_head.weight", quant=True)
            assert "d2t" in sd, "compressed DFlash ckpt missing d2t"
            d.d2t = sd["d2t"].to(torch.int64)
            if "t2d" in sd:
                d.t2d = sd["t2d"].to(torch.bool)

    @torch.inference_mode()
    def propose(
        self, reqs: List["Req"], num_draft: int, ctx: ProposeContext, topk: int = 0
    ) -> List[List[int]]:
        out: List[List[int]] = [[] for _ in reqs]
        draft = self._draft
        device = self._device
        mask_id = self._mask_token_id
        B = self._block_size
        # DDTree (topk>0): stash the per-position top-K MARGINALS (target-vocab ids + log-probs) of the
        # k_i drafted positions per req, keyed by id(req), for build_draft_tree. Mirrors the scheduler's
        # _tidar_block_predict topk path. Cleared each call.
        self._ddtree_topk: dict[int, tuple] = {}
        for i, req in enumerate(reqs):
            # Block emits up to B-1 drafts; clamp to the per-step draft budget and the req budget.
            k_i = max(0, min(num_draft, B - 1, req.remain_len - 1))
            # aux: 3D [num_aux, P, hidden] over all committed positions (full-context, the z-lab fix),
            # or legacy 2D [num_aux, hidden] at the last confirmed token (MINISGL_DFLASH_FULLCTX=0).
            aux = ctx.aux_hidden.get(req.uid)
            if k_i <= 0 or aux is None:
                continue

            anchor_tok = int(req.input_ids[req.cached_len])
            base_pos = req.cached_len + self._pos_off

            # target_hidden = hidden_norm(fc(concat)) — the per-layer KV prefix.
            if aux.dim() == 3:
                # Full context: prefix = aux of committed positions [cached_len-P .. cached_len-1], at
                # their TRUE absolute RoPE positions; the block [anchor, mask...] follows at [base_pos..].
                P = aux.shape[1]
                aux_t = aux.permute(1, 0, 2).contiguous().to(self._dtype)  # [P, num_aux, hidden]
                target_hidden = draft.fuse_aux(aux_t)  # [P, hidden]
                ctx_start = req.cached_len - P
                ctx_pos = torch.arange(
                    ctx_start, ctx_start + P, dtype=torch.int32, device=device
                )
            else:
                # Legacy single-position prefix (P=1) at the anchor's own position.
                target_hidden = draft.fuse_aux(aux.unsqueeze(0).to(self._dtype))  # [1, hidden]
                ctx_pos_val = (
                    int(self._ctx_pos_env) if self._ctx_pos_env is not None else base_pos
                )
                ctx_pos = torch.tensor([ctx_pos_val], dtype=torch.int32, device=device)

            # noise block = [anchor, mask*(B-1)]; one denoising forward emits all B hidden vectors.
            block_ids = torch.full((B,), mask_id, dtype=torch.int64, device=device)
            block_ids[0] = anchor_tok
            noise_embed = draft.embed(block_ids).to(self._dtype)  # [B, hidden]
            block_pos = torch.arange(base_pos, base_pos + B, dtype=torch.int32, device=device)

            hidden = draft.denoise(noise_embed, target_hidden, block_pos, ctx_pos)  # [B, hidden]
            logits = draft.head(hidden)  # [B, vocab]
            # Positions 1..B-1 are the speculation (position 0 is the known anchor).
            block_logits = logits[1 : 1 + k_i]  # [k_i, vocab]
            ids = block_logits.argmax(dim=-1)  # [k_i] draft-vocab ids
            if self._compressed:
                ids = ids + self._d2t[ids]  # draft id -> target id (delta map)
            drafts = [int(x) for x in ids.tolist()]
            out[i] = drafts
            if topk > 0 and k_i > 0:
                lp = torch.log_softmax(block_logits.float(), dim=-1)  # [k_i, vocab]
                tv, ti = lp.topk(topk, dim=-1)  # [k_i, topk], descending
                if self._compressed:
                    ti = ti + self._d2t[ti]  # draft ids -> target ids (delta map)
                self._ddtree_topk[id(req)] = (ti.cpu().tolist(), tv.cpu().tolist())
            if self._dbg:
                print(f"[dflash-dbg] uid={req.uid} anchor={anchor_tok} base_pos={base_pos} "
                      f"B={B} k={k_i} draft={drafts}", flush=True)
        return out

    def on_accept(self, reqs: List["Req"], num_accepted: List[int]) -> None:
        # No persistent draft KV: the captured target aux (refreshed each step by the scheduler) is
        # the cross-block context, so there is nothing to roll back.
        return

    def free(self, uid: int) -> None:
        return
