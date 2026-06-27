from __future__ import annotations

import os
from typing import TYPE_CHECKING, Dict, List

import torch

from .base import Proposer, ProposeContext

if TYPE_CHECKING:
    from minisgl.core import Req


__all__ = ["DraftModelProposer"]


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


class DraftModelProposer(Proposer):
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
    MTP head): every confirmed token is run through the draft layer (growing the context across decode
    steps), the K drafts append temporary K/V, and `on_accept` truncates back to the accepted prefix;
    `free(uid)` drops it on finish. It never touches the engine's paged KV. (`MINISGL_EAGLE3_NO_CTX=1`
    rebuilds the cache empty each step — the diagnostic that exposed this ~2% floor.)
    """

    needs_last_hidden = False
    capture_layer_ids = _GLM47_CAPTURE_LAYER_IDS

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
        # Persistent per-uid draft KV (one draft layer). Like the MTP head, the EAGLE3 draft's
        # self-attention needs the full causal context, not just the K draft tokens — restricting
        # attention to the chain collapses the 1-layer head to near-random. Each entry is the KV for
        # one processed position; the first `_committed[uid]` are confirmed (permanent), any beyond
        # are this step's draft tail (truncated by on_accept). MINISGL_EAGLE3_NO_CTX=1 disables it
        # (rebuild-empty each step) for diagnostics.
        self._cache: Dict[int, list] = {}
        self._committed: Dict[int, int] = {}
        self._no_ctx = os.environ.get("MINISGL_EAGLE3_NO_CTX") == "1"

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

    @torch.inference_mode()
    def propose(self, reqs: List["Req"], num_draft: int, ctx: ProposeContext) -> List[List[int]]:
        # Per-req, like the MTP head: the 1-layer draft attention needs the full causal context, so
        # each request keeps a PERSISTENT KV (one draft layer) that grows across decode steps. Step 0
        # processes the confirmed token (fused with the captured target aux) and appends its K/V to
        # the persistent cache; later steps process each draft. The attention therefore sees the full
        # confirmed prefix, not just the K-token chain. on_accept truncates the cache to the accepted
        # prefix; free(uid) drops it on finish.
        out: List[List[int]] = [[] for _ in reqs]
        draft = self._draft
        device = self._device
        for i, req in enumerate(reqs):
            k_i = max(0, min(num_draft, req.remain_len - 1))
            aux = ctx.aux_hidden.get(req.uid)  # [num_aux, hidden] at the last confirmed token
            if k_i <= 0 or aux is None:
                continue

            tok_idx = req.cached_len + self._tok_off
            tok_idx = max(0, min(tok_idx, req.input_ids.shape[0] - 1))
            conf_tok = int(req.input_ids[tok_idx])
            base_pos = req.cached_len + self._pos_off

            # Drop the previous step's rejected-draft tail (keep only confirmed context).
            if self._no_ctx:
                cache: list = []
            else:
                cache = self._cache.setdefault(req.uid, [])
                committed = self._committed.get(req.uid, 0)
                del cache[committed:]

            fused = draft.fuse_aux(aux.unsqueeze(0).to(self._dtype))  # [1, hidden] step-0 hidden
            cur_tok = torch.tensor([conf_tok], dtype=torch.int64, device=device)
            cur_hidden = fused  # [1, hidden]
            drafts: List[int] = []
            for step in range(k_i):
                embed_e = draft.embed(cur_tok)  # [1, hidden]
                positions = torch.tensor([base_pos + step], dtype=torch.int32, device=device)
                logits, cur_hidden = draft.step(embed_e, cur_hidden, positions, cache)
                draft_id = int(logits.argmax(dim=-1).item())  # COMPRESSED draft vocab
                target_id = draft_id + int(self._d2t[draft_id].item())  # -> target vocab
                drafts.append(target_id)
                cur_tok = torch.tensor([target_id], dtype=torch.int64, device=device)
            out[i] = drafts
            if self._dbg:
                print(f"[eagle3-dbg] uid={req.uid} conf={conf_tok} base_pos={base_pos} "
                      f"k={k_i} ctx={len(cache) - k_i} draft={drafts}", flush=True)
        return out

    def on_accept(self, reqs: List["Req"], num_accepted: List[int]) -> None:
        # The confirmed token (always committed) plus the n accepted drafts become permanent draft
        # context; the K-n rejected drafts' K/V are dropped. The cache layout per step is
        # [confirmed, d0, d1, ...]; committing 1 + n keeps confirmed + the accepted run.
        if self._no_ctx:
            return
        for req, n in zip(reqs, num_accepted):
            if req.uid in self._cache:
                self._committed[req.uid] = self._committed.get(req.uid, 0) + 1 + n
                del self._cache[req.uid][self._committed[req.uid]:]

    def free(self, uid: int) -> None:
        self._cache.pop(uid, None)
        self._committed.pop(uid, None)
