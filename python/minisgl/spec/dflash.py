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
        # CCA-recurrent drafter (the trained ZAYA DFlashCCADraftModel) has no attention/rope/block_size
        # — a fundamentally different arch from the z-lab Qwen3-GQA drafter. Build it on its own path.
        architectures = getattr(hf, "architectures", []) or []
        self._is_cca = "DFlashCCADraftModel" in architectures or bool(getattr(hf, "cca_config", None))
        if self._is_cca:
            self._build_cca(engine, hf, cfg, dfc, folder, hidden, num_layers)
            return
        self._is_cca = False
        # Laguna DFlash drafter (poolside/Laguna-XS-2.1-DFlash): a Laguna-XS gated GQA trunk (fused
        # qkv, per-head softplus g_proj gate, per-aux hidden norms, causal + sliding-window block
        # attention). Same fc->KV-prefix mechanism as the z-lab Qwen path, different decoder layer.
        model_type = str(getattr(hf, "model_type", "") or "").lower()
        self._is_laguna = model_type == "laguna" or any("Laguna" in a for a in architectures)
        if self._is_laguna:
            self._build_laguna(engine, hf, cfg, dfc, folder, hidden, num_layers)
            return
        self._is_laguna = False
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
        if self._block_size < 2:
            # The ZAYA DFlash checkpoints don't record block_size in dflash_config; the trained block is
            # num_spec masks + 1 anchor. --spec-num-draft carries the trained num_spec, and the propose
            # step emits min(block-1, num_draft) drafts, so block = num_draft+1 makes --spec-num-draft
            # set the width exactly and matches training (m4dss num_spec=4 -> block 5; ns15=15 -> 16).
            self._block_size = self._num_draft + 1
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

        # --- Persistent per-request draft KV (the z-lab crop-per-step fix) -------------------------
        # The fc+k/v-proj+rotary of a committed position's captured aux is FIXED once committed (the
        # scheduler only ever APPENDS accepted positions to the aux buffer — never rolls back), so it
        # is cacheable. Instead of re-projecting the whole [num_aux, P, hidden] prefix every propose
        # (O(P) per generated token — the ~47 GFLOP/step re-feed that made DFlash a net LOSS), we keep
        # a per-uid, per-layer prefix K/V and project ONLY the newly-accepted tail each step (O(new)).
        # _kv[uid] = list over layers of [k_ctx, v_ctx]; _kv_plen[uid] = #positions already projected.
        # Only active on the full-context (aux.dim()==3) path with no ctx window; disable via
        # MINISGL_DFLASH_PERSIST_KV=0 to fall back to the recompute path (diagnostic).
        self._persist = os.environ.get("MINISGL_DFLASH_PERSIST_KV", "1") not in ("0", "false", "no")
        self._ctx_window = int(os.environ.get("MINISGL_DFLASH_CTX_WINDOW", "0") or 0)
        self._kv: dict[int, list] = {}
        self._kv_plen: dict[int, int] = {}

    def _build_cca(self, engine, hf, cfg, dfc, folder, hidden, num_layers) -> None:
        """Build + load the CCA-recurrent DFlash drafter (ZAYA DFlashCCADraftModel). B = 1 + num_draft
        (no fixed block_size in the ckpt); the seed = fc(single committed-position aux) is the drafter's
        WHOLE context — it was trained on one seed, so NO full-context prefix (unlike the Qwen path)."""
        from minisgl.models.dflash_cca import DFlashCCADraftModel

        cca = dict(getattr(hf, "cca_config", None) or {})
        inter = int(cfg("intermediate_size"))
        eps = float(cfg("rms_norm_eps", default=1e-6))
        head_dim = int(cca.get("head_dim") or cfg("head_dim"))
        num_q_heads = int(cca.get("num_q_heads") or (hidden // head_dim))
        num_k_heads = int(cca.get("num_k_heads", 2))

        ids = dfc.get("target_layer_ids")
        if (env_ids := os.environ.get("MINISGL_DFLASH_CAPTURE_LAYERS")):
            ids = [int(x) for x in env_ids.split(",") if x.strip() != ""]
        assert ids, "CCA DFlash ckpt has no target_layer_ids"
        self.capture_layer_ids = [int(x) for x in ids]
        target_hidden = engine.model.model.embed_tokens.weight.shape[1]

        self._block_size = 1 + self._num_draft  # block = [anchor, mask*num_draft]
        self._mask_token_id = int(dfc.get("mask_token_id", 0))
        self._compressed = False
        self._d2t = None

        with torch.device(self._device):
            self._draft = DFlashCCADraftModel(
                hidden_size=hidden, intermediate_size=inter, num_layers=num_layers,
                num_q_heads=num_q_heads, num_k_heads=num_k_heads, head_dim=head_dim,
                num_aux_layers=len(self.capture_layer_ids), target_hidden=target_hidden,
                conv_kernel=int(cca.get("block_conv_kernel", 3)),
                rms_norm_eps=eps, clamp_temp=bool(cca.get("clamp_temp", True)),
                block_mixer=str(cca.get("block_mixer", "attn")),
            )
        self._load_cca_weights(folder)
        self._draft.bind_embed(engine.model.model.embed_tokens)
        self._draft.bind_lm_head(engine.model.lm_head)

        self._dbg = os.environ.get("MINISGL_SPEC_DEBUG") in ("2", "3")
        self._pos_off = 0
        self._ctx_pos_env = None
        # rdna4's on_accept/free reference the persistent-KV dicts (Qwen path); the CCA drafter has no
        # persistent KV, so init them empty so free()/on_accept are safe no-ops for CCA reqs.
        self._kv = {}
        self._kv_plen = {}

    def _load_cca_weights(self, folder: str) -> None:
        """Load the CCA drafter checkpoint (keys: fc, norm, layers.i.{linear_q,linear_k,val_proj,o_proj,
        input_layernorm,post_attention_layernorm,gate_proj,up_proj,down_proj}.weight + conv_qk.weight/bias
        + temp). Replicated on every TP rank."""
        import safetensors.torch as st

        path = next((os.path.join(folder, f) for f in sorted(os.listdir(folder))
                     if f.endswith(".safetensors")), None)
        assert path is not None, f"no .safetensors in CCA DFlash folder {folder}"
        sd = st.load_file(path, device=str(self._device))
        d = self._draft

        def put(obj, leaf, key):
            assert key in sd, f"CCA DFlash ckpt missing {key}"
            t = sd[key].to(self._dtype).contiguous()
            cur = getattr(obj, leaf)
            assert cur.shape == t.shape, (
                f"shape mismatch {key}: model {tuple(cur.shape)} vs ckpt {tuple(t.shape)}"
            )
            setattr(obj, leaf, t.to(self._device))

        put(d.fc, "weight", "fc.weight")
        put(d.norm, "weight", "norm.weight")
        for i, layer in enumerate(d.layers):
            p = f"layers.{i}."
            put(layer.input_layernorm, "weight", p + "input_layernorm.weight")
            put(layer.post_attention_layernorm, "weight", p + "post_attention_layernorm.weight")
            put(layer.linear_q, "weight", p + "linear_q.weight")
            put(layer.linear_k, "weight", p + "linear_k.weight")
            put(layer.val_proj, "weight", p + "val_proj.weight")
            put(layer.o_proj, "weight", p + "o_proj.weight")
            put(layer.gate_proj, "weight", p + "gate_proj.weight")
            put(layer.up_proj, "weight", p + "up_proj.weight")
            put(layer.down_proj, "weight", p + "down_proj.weight")
            put(layer, "conv_qk_weight", p + "conv_qk.weight")
            put(layer, "conv_qk_bias", p + "conv_qk.bias")
            put(layer, "temp", p + "temp")

    def _build_laguna(self, engine, hf, cfg, dfc, folder, hidden, num_layers) -> None:
        """Build + load the Laguna-XS DFlash drafter (poolside/Laguna-XS-2.1-DFlash). Same fc->per-layer
        KV-prefix block-diffusion as the z-lab Qwen path, but the trunk is a Laguna-XS gated GQA layer
        (fused qkv_proj, per-head softplus g_proj gate, per-aux hidden norms) with CAUSAL + sliding-
        window block attention. Full draft vocab == target vocab and NO own embed/head in the ckpt, so
        it borrows the target embed_tokens + lm_head (no d2t remap). Replicated on every TP rank."""
        from minisgl.models.dflash import DFlashDraftModel

        num_heads = int(cfg("num_attention_heads"))
        num_kv_heads = int(cfg("num_key_value_heads"))
        head_dim = int(cfg("head_dim", default=hidden // num_heads))
        inter = int(cfg("intermediate_size"))
        eps = float(cfg("rms_norm_eps", default=1e-6))
        rope_theta = float(cfg("rope_theta", default=500000.0))
        max_pos = int(cfg("max_position_embeddings", default=262144))
        sliding_window = int(getattr(hf, "sliding_window", 0) or 0)
        causal = bool(dfc.get("causal", True))

        self._block_size = int(dfc.get("block_size") or getattr(hf, "block_size", 0) or 16)
        block_cap = int(os.environ.get("MINISGL_DFLASH_BLOCK", "0") or 0)
        if block_cap:
            self._block_size = min(self._block_size, block_cap)
        assert self._block_size >= 2, f"DFlash block_size must be >= 2, got {self._block_size}"
        self._mask_token_id = int(dfc.get("mask_token_id", 12))

        ids = dfc.get("target_layer_ids") or getattr(hf, "aux_hidden_state_layer_ids", None)
        if (env_ids := os.environ.get("MINISGL_DFLASH_CAPTURE_LAYERS")):
            ids = [int(x) for x in env_ids.split(",") if x.strip() != ""]
        assert ids, "Laguna DFlash ckpt has no target_layer_ids"
        self.capture_layer_ids = [int(x) for x in ids]
        num_aux = len(self.capture_layer_ids)

        # Full-vocab drafter with no own head -> tied to the target (borrow embed_tokens + lm_head).
        self._compressed = False
        self._d2t = None
        target_hidden = engine.model.model.embed_tokens.weight.shape[1]
        assert hidden == target_hidden, (
            f"Laguna DFlash draft hidden {hidden} != target hidden {target_hidden}"
        )

        # Build the empty drafter in the COMPUTE DTYPE (bf16), not torch's fp32 default. _PlainLinear /
        # RMSNorm allocate `torch.empty(...)` with no explicit dtype, so under the default they land as
        # fp32 — 2x the resident bytes of the bf16 checkpoint they're about to be filled with. On a 16 GB
        # TP=2 pair that fp32 scaffold (~2 GiB) is then freed when _load_laguna_weights replaces each
        # weight with its bf16 tensor, but expandable_segments keeps ~0.9 GiB of it RESERVED — dead
        # headroom that (with the on-card staging, now fixed) was the spec-decode boot OOM. Constructing
        # bf16 up front halves the scaffold so the freed blocks are reused by the incoming bf16 weights.
        # get_rope keeps its cos/sin cache fp32 (explicit dtype) and every norm weight is overwritten by
        # the bf16 checkpoint, so this only right-sizes the linear scaffolds — no numerics change.
        _prev_default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(self._dtype)
        try:
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
                    draft_vocab_size=None,
                    own_embed=False,
                    decoder_layer_type="laguna_xs",
                    sliding_window=sliding_window,
                    causal=causal,
                    per_aux_norm=True,
                )
        finally:
            torch.set_default_dtype(_prev_default_dtype)
        self._load_laguna_weights(folder)
        self._draft.bind_embed(engine.model.model.embed_tokens)
        self._draft.bind_lm_head(engine.model.lm_head)

        self._dbg = os.environ.get("MINISGL_SPEC_DEBUG") in ("2", "3")
        self._pos_off = int(os.environ.get("MINISGL_DFLASH_POS_OFF", "0"))
        self._ctx_pos_env = os.environ.get("MINISGL_DFLASH_CTX_POS")
        self._persist = os.environ.get("MINISGL_DFLASH_PERSIST_KV", "1") not in ("0", "false", "no")
        self._ctx_window = int(os.environ.get("MINISGL_DFLASH_CTX_WINDOW", "0") or 0)
        self._kv: dict[int, list] = {}
        self._kv_plen: dict[int, int] = {}
        logger.info_rank0(
            f"DFlash Laguna drafter: {num_layers}L h={hidden} heads={num_heads}/{num_kv_heads} "
            f"block={self._block_size} mask_id={self._mask_token_id} window={sliding_window} "
            f"causal={causal} aux_layers={self.capture_layer_ids} (bf16, borrow target embed/head)"
        )

    def _load_laguna_weights(self, folder: str) -> None:
        """Load the Laguna DFlash checkpoint (bf16, ~0.92 GB). fc / hidden_norm / aux_hidden_norms.i /
        norm + per-layer {input_layernorm, post_attention_layernorm, self_attn.qkv_proj (fused ->
        split q|k|v), self_attn.o_proj, self_attn.g_proj, self_attn.q_norm, self_attn.k_norm,
        mlp.{gate,up,down}_proj}. Replicated on every TP rank."""
        import safetensors.torch as st

        path = next((os.path.join(folder, f) for f in sorted(os.listdir(folder))
                     if f.endswith(".safetensors")), None)
        assert path is not None, f"no .safetensors in Laguna DFlash folder {folder}"
        # Load the checkpoint to HOST RAM, not GPU: the checkpoint staging + the model's initial
        # on-device empty weights (built under `with torch.device(device)`) must not coexist on the card.
        # Under expandable_segments the freed staging stays RESERVED (never returned), so loading the whole
        # checkpoint straight to the card (the old device=cuda path) left ~1.8 GiB of idle-but-reserved
        # memory for a ~1 GiB drafter — a big part of the spec-decode boot OOM (only 0.19 GiB free after
        # capture -> any runtime alloc OOMs). Stage in HOST RAM and move each final tensor to the card one
        # at a time; a trailing empty_cache (below) returns the transient.
        # NOTE: the reassignment below (setattr, not copy_) is load-bearing — the model was
        # built with fp32 `torch.empty` weights (_PlainLinear), so we must REPLACE them with the bf16
        # checkpoint tensors, not copy_ into the fp32 buffers (which would leave the drafter fp32 and trip
        # `dense_gemm_rd: only bf16/fp16 activations` on the first propose).
        sd = st.load_file(path, device="cpu")
        d = self._draft

        def put(mod, leaf, key):
            assert key in sd, f"Laguna DFlash ckpt missing {key}"
            t = sd[key].to(self._dtype).contiguous()
            cur = getattr(mod, leaf)
            assert cur.shape == t.shape, (
                f"shape mismatch {key}: model {tuple(cur.shape)} vs ckpt {tuple(t.shape)}"
            )
            setattr(mod, leaf, t.to(self._device))  # bf16 host tensor -> card (replaces the fp32 empty)

        put(d.fc, "weight", "fc.weight")
        put(d.hidden_norm, "weight", "hidden_norm.weight")
        put(d.norm, "weight", "norm.weight")
        assert d.aux_hidden_norms is not None
        for i, an in enumerate(d.aux_hidden_norms):
            put(an, "weight", f"aux_hidden_norms.{i}.weight")

        q_dim = d.layers[0].q_dim
        kv_dim = d.layers[0].kv_dim
        for i, layer in enumerate(d.layers):
            p = f"layers.{i}."
            put(layer.input_layernorm, "weight", p + "input_layernorm.weight")
            put(layer.post_attention_layernorm, "weight", p + "post_attention_layernorm.weight")
            # Fused qkv_proj [q_dim+2*kv_dim, hidden] -> split into q/k/v _PlainLinear weights.
            qkv_key = p + "self_attn.qkv_proj.weight"
            assert qkv_key in sd, f"Laguna DFlash ckpt missing {qkv_key}"
            qkv = sd[qkv_key].to(self._dtype)
            assert qkv.shape[0] == q_dim + 2 * kv_dim, (
                f"qkv rows {qkv.shape[0]} != q{q_dim}+2*kv{kv_dim}"
            )
            layer.q_proj.weight = qkv[:q_dim].contiguous().to(self._device)
            layer.k_proj.weight = qkv[q_dim:q_dim + kv_dim].contiguous().to(self._device)
            layer.v_proj.weight = qkv[q_dim + kv_dim:].contiguous().to(self._device)
            put(layer.o_proj, "weight", p + "self_attn.o_proj.weight")
            put(layer.g_proj, "weight", p + "self_attn.g_proj.weight")
            put(layer.q_norm, "weight", p + "self_attn.q_norm.weight")
            put(layer.k_norm, "weight", p + "self_attn.k_norm.weight")
            put(layer.gate_proj, "weight", p + "mlp.gate_proj.weight")
            put(layer.up_proj, "weight", p + "mlp.up_proj.weight")
            put(layer.down_proj, "weight", p + "mlp.down_proj.weight")
        # Release the host staging dict + return the GPU segments freed when the fp32 empty weights were
        # replaced above. Without this, expandable_segments keeps that ~1 GiB reserved-but-idle on the
        # card — dead runtime headroom the spec-verify path then OOMs against.
        del sd
        torch.cuda.empty_cache()

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
        if self._is_cca:
            return self._propose_cca(reqs, num_draft, ctx, topk, out)
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
            prefix_kv = None
            if aux.dim() == 3:
                # Full context: prefix = aux of committed positions [cached_len-P .. cached_len-1], at
                # their TRUE absolute RoPE positions; the block [anchor, mask...] follows at [base_pos..].
                P = aux.shape[1]
                ctx_start = req.cached_len - P
                if self._persist and self._ctx_window == 0:
                    # FAST PATH: project only the newly-accepted tail [cached .. P-1] and append it to
                    # the per-uid persistent K/V; reuse the cached prefix for the rest. The scheduler
                    # only appends accepted positions to `aux`, so aux[:, :cached] is unchanged from the
                    # previous step and needs no re-projection. free(uid) drops the cache on finish.
                    uid = req.uid
                    cached = self._kv_plen.get(uid, 0)
                    if P < cached:  # uid reuse without free (defensive): rebuild from scratch
                        cached = 0
                        self._kv.pop(uid, None)
                    if P > cached:
                        new_aux = aux[:, cached:P].permute(1, 0, 2).contiguous().to(self._dtype)
                        new_pos = torch.arange(
                            ctx_start + cached, ctx_start + P, dtype=torch.int32, device=device
                        )
                        new_kv = draft.project_prefix(new_aux, new_pos)  # per-layer (k_ctx, v_ctx)
                        if cached == 0:
                            self._kv[uid] = [[k, v] for (k, v) in new_kv]
                        else:
                            cache = self._kv[uid]
                            for l, (k, v) in enumerate(new_kv):
                                cache[l][0] = torch.cat([cache[l][0], k], dim=0)
                                cache[l][1] = torch.cat([cache[l][1], v], dim=0)
                        self._kv_plen[uid] = P
                    prefix_kv = self._kv[uid]
                else:
                    # Recompute path (persist disabled or ctx-window active): project the whole prefix.
                    aux_t = aux.permute(1, 0, 2).contiguous().to(self._dtype)  # [P, num_aux, hidden]
                    target_hidden = draft.fuse_aux(aux_t)  # [P, hidden]
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

            if prefix_kv is not None:
                hidden = draft.denoise_cached(noise_embed, prefix_kv, block_pos)  # [B, hidden]
            else:
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

    def _propose_cca(self, reqs, num_draft, ctx, topk, out):
        """CCA-recurrent single-seed propose. seed = fc(aux at the committed position) is the drafter's
        whole context; one bidirectional block forward over [anchor, mask*num_draft] emits B hidden
        vectors; argmax(logits[1:1+k]) are the drafts (+ top-K marginals for DDTree)."""
        draft = self._draft
        device = self._device
        mask_id = self._mask_token_id
        B = self._block_size
        for i, req in enumerate(reqs):
            k_i = max(0, min(num_draft, B - 1, req.remain_len - 1))
            aux = ctx.aux_hidden.get(req.uid)
            if k_i <= 0 or aux is None:
                continue
            if aux.dim() == 3:
                aux = aux[:, -1]  # CCA wants a SINGLE committed-position seed (last, if accumulated)
            anchor_tok = int(req.input_ids[req.cached_len])
            # Diagnostic: MINISGL_CCA_NO_SEED=1 runs the drafter UNCONTEXTUALIZED (seed=None). If
            # accept-len barely changes vs the seeded run, the aux->seed conditioning isn't helping.
            seed = (
                None if os.environ.get("MINISGL_CCA_NO_SEED") == "1"
                else draft.fuse_aux(aux.unsqueeze(0).to(self._dtype))  # [1, hidden]
            )
            block_ids = torch.full((B,), mask_id, dtype=torch.int64, device=device)
            block_ids[0] = anchor_tok
            noise_embed = draft.embed(block_ids).to(self._dtype).unsqueeze(0)  # [1, B, hidden]
            hidden = draft.denoise(noise_embed, seed)  # [1, B, hidden]
            logits = draft.head(hidden[0])  # [B, vocab]
            block_logits = logits[1 : 1 + k_i]  # [k_i, vocab]
            ids = block_logits.argmax(dim=-1)
            out[i] = [int(x) for x in ids.tolist()]
            if topk > 0 and k_i > 0:
                lp = torch.log_softmax(block_logits.float(), dim=-1)
                tvals, ti = lp.topk(topk, dim=-1)
                self._ddtree_topk[id(req)] = (ti.cpu().tolist(), tvals.cpu().tolist())
            if self._dbg:
                a = aux.float(); l0 = block_logits[0].float()
                top = l0.topk(5)
                sst = "None" if seed is None else (
                    f"mean={seed.float().mean():.3f} std={seed.float().std():.3f} amax={seed.float().abs().max():.1f}")
                print(f"[dflash-cca-dbg] uid={req.uid} anchor={anchor_tok} k={k_i} "
                      f"aux{tuple(aux.shape)} mean={a.mean():.3f} std={a.std():.3f} amax={a.abs().max():.1f} "
                      f"seed {sst} "
                      f"d0_top5={top.indices.tolist()} lp={[round(float(x),2) for x in top.values.tolist()]} "
                      f"drafts={out[i]}", flush=True)
        return out

    def on_accept(self, reqs: List["Req"], num_accepted: List[int]) -> None:
        # Nothing to roll back: the persistent draft K/V holds only COMMITTED (accepted) positions.
        # The scheduler appends this step's accepted-tail aux to `ctx.aux_hidden`, and the NEXT propose
        # lazily projects that delta into the cache (see the fast path in `propose`). Rejected drafts
        # never enter the prefix, so there is no draft tail to truncate.
        return

    def free(self, uid: int) -> None:
        # Drop the finished/aborted request's persistent draft K/V (and its projected-length counter).
        self._kv.pop(uid, None)
        self._kv_plen.pop(uid, None)
