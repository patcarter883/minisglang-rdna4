from __future__ import annotations

import os
from typing import TYPE_CHECKING, List, Optional

import torch

from minisgl.utils import init_logger

from .base import ProposeContext
from .capture import CapturableProposer, StagedPropose

logger = init_logger(__name__)

# Sentinel absolute position for a ring column that holds nothing (or holds a previous owner's key).
# Any real query position minus this is astronomically larger than any sliding window, so the mask
# discards it — which is why the ring needs no zeroing of stale K/V, only of this position vector.
_NO_POS = -(1 << 40)

if TYPE_CHECKING:
    from minisgl.core import Req


__all__ = ["DFlashProposer"]


# Prompt-prefill seed length, in trailing aux positions. MEASURED, not provisional: the {0,32,64,128,
# 256,528} x {3571-tok code @1600, 95-tok instruction @384} sweep, two full replicates, is in
# CONTINUANCE §11.8 / tools/spec_seed_tail_sweep_results.txt. 64 is the ONLY seeded tail whose four
# long-prompt legs all beat their own boot's matched plain leg, and it is joint-best on the combined
# score (86.2 tok/s geo-mean vs 79.7 seeding-off, 72.1 plain).
#
# But the number itself barely matters, and saying so is the point of this comment: on a 95-token
# prompt tails 32..528 are BIT-IDENTICAL (same completion md5, same 5.394 accept-len, same first
# draft chain), and on a 3.5k-token prompt the seed produces NO lift in the P<64 region where it is
# physically able to act (2.46/2.39 seeding-off vs 2.03-2.83 seeded, unordered) — the long-prompt
# tail-vs-tail spread is the greedy content lottery, which is 19% wide at a FIXED tail. What the seed
# is really a function of is PROMPT LENGTH: +23.9% accept-len at 95 tokens, +16.1% at 223, +10.7% at
# 351, then 0 to -3% from 607 out to 3495 — crossing zero at the drafter's own 512-key window. So
# 341c4df0's eviction story is falsified (a 32-token seed cannot evict 512 keys yet behaves like the
# 528 one), and the real follow-up is to gate seeding on prompt_len <= sliding_window, not to retune
# this constant. Override with MINISGL_DFLASH_SEED_TAIL (0 = seeding off).
_SEED_TAIL_DEFAULT = 64

# Compacting-buffer slack, in rows, on top of the drafter's attention window (see _init_prefix_kv).
# Amortisation only, never correctness: the newest `window` rows are resident at every step for any
# slack >= the per-step append (<= block_size). 128 makes the memmove fire roughly every 128/accepted
# steps (~30 at a 4-token accept) and costs 128 extra rows/layer = ~0.26 MB/uid over the window.
_KV_SLACK = 128

_SLIDING_ATTENTION = "sliding_attention"

# Rows projected per call on the cold ring-rebuild path. 2048 is under the largest capacity measured
# working (6160) with margin, and well under the one measured faulting (8208). Override with
# MINISGL_DFLASH_REBUILD_CHUNK to re-characterise the boundary rather than to tune throughput — this
# path runs once per request, not per step.
_REBUILD_CHUNK = int(os.environ.get("MINISGL_DFLASH_REBUILD_CHUNK") or 2048)


def dflash_layer_masks(cfg, num_layers: int):
    """Per-layer (causal, window) for a DFlash drafter, from DECLARED config fields only.

    Transcribed from the two merged upstream implementations, which were written independently and
    agree exactly:

      vllm/model_executor/models/qwen3_dflash.py
        `_dflash_layer_causal`: "``dflash_config.causal`` overrides all layers; else only SWA layers
        causal."; `use_swa` "forces SWA on every layer, even an all-full ``layer_types``";
        window = `dflash_config.swa_window_size` else top-level `sliding_window`, and a sliding layer
        with no window is an ERROR rather than a silent full-attention layer.
      sglang/srt/models/dflash.py
        full_attention -> AttentionType.ENCODER_ONLY (non-causal), window -1
        sliding_attention -> AttentionType.DECODER (causal), window `sliding_window - 1`

    (The `-1` is a representation difference, not a semantic one: SGLang stores `window_left` while
    this engine's mask tests `(qpos - kpos) < window`, which keeps exactly the same distances.)

    Reading these rather than sniffing `model_type`/`architectures` is the whole point. The previous
    dispatch gated the windowed+causal build on the checkpoint being NAMED laguna, so four of the
    seven DFlash drafters on this box — every one that declares a window but is not called Laguna —
    silently got `causal=False, sliding_window=0`: the wrong mask, and (because an unbounded prefix
    has no fixed-capacity ring) a permanently eager propose. That outcome was then written up as a
    property of those checkpoints. It is a property of the dispatch.
    """
    dfc = getattr(cfg, "dflash_config", None) or {}
    if not isinstance(dfc, dict):
        dfc = dict(dfc)
    tlc = getattr(cfg, "transformer_layer_config", None) or {}

    def declared(key, default=None):
        for src in (dfc, tlc, cfg.__dict__ if hasattr(cfg, "__dict__") else {}):
            if isinstance(src, dict):
                if key in src:
                    return src[key]
            elif hasattr(src, key):
                return getattr(src, key)
        return default

    layer_types = declared("layer_types")
    use_swa = bool(dfc.get("use_swa", False))
    override = dfc.get("causal")  # declared global override; None = derive per layer
    if override is None and "sliding_window_non_causal" in dfc:
        # The upstream `speculators` DFlashSpeculatorConfig spells the same axis inverted:
        # `sliding_window_non_causal: bool = False` ("Use non-causal synthetic block attention for
        # sliding-window layers"). Read it where it is declared rather than only its vLLM alias.
        override = not bool(dfc["sliding_window_non_causal"])
    # MEASUREMENT/OPERATOR override for a checkpoint that declares NEITHER. The per-layer rule below
    # is vLLM/SGLang's, and it was written for the z-lab Qwen DFlash drafters; it is NOT established
    # for every block-diffusion drafter. Muse-Glimmer's card describes its drafter as predicting
    # "entire blocks of 16 tokens in a single forward pass" — block diffusion, i.e. bidirectional
    # WITHIN the block — and z-lab's own HF reference sets is_causal=False on every layer, while
    # vLLM/SGLang make sliding layers causal. Those genuinely disagree, and no Muse config field
    # settles it, so expose the axis and let acceptance decide instead of guessing in code.
    _env = (os.environ.get("MINISGL_DFLASH_CAUSAL") or "").strip().lower()
    if _env in ("0", "false", "no"):
        override = False
    elif _env in ("1", "true", "yes"):
        override = True

    any_sliding = False
    if layer_types is not None:
        any_sliding = any(lt == _SLIDING_ATTENTION for lt in layer_types)

    window_decl = dfc.get("swa_window_size", declared("sliding_window"))
    causal, window = [], []
    for i in range(num_layers):
        if layer_types is None or (use_swa and not any_sliding):
            is_sliding = use_swa
        else:
            is_sliding = i < len(layer_types) and layer_types[i] == _SLIDING_ATTENTION
        if is_sliding and not window_decl:
            raise ValueError(
                f"DFlash layer {i} is `{_SLIDING_ATTENTION}` but no window is declared "
                "(dflash_config.swa_window_size or a top-level sliding_window)")
        causal.append(bool(override) if override is not None else bool(is_sliding))
        window.append(int(window_decl) if is_sliding else 0)
    return causal, window


def dflash_trunk_probe(names):
    """What KIND of trunk this drafter is, from the CHECKPOINT'S OWN TENSORS — never its name.

    `names` is `models.weight.checkpoint_tensor_names(folder)`, which reads only safetensors headers.
    Each signal is the presence of a weight that only that variant has, so a checkpoint that ships
    the structure gets the code path whether or not anybody has taught minisgl its vendor string:

      conv_qk            -> the CCA-recurrent drafter (a different architecture, its own builder)
      self_attn.g_proj   -> gated attention (the `laguna_xs` decoder layer's softplus output gate)
      aux_hidden_norms.* -> a per-captured-layer RMSNorm on each aux BEFORE fc
      self_attn.qkv_proj -> fused QKV, rather than separate q/k/v projections
    """
    return {
        "cca": any(".conv_qk." in n or n.startswith("conv_qk.") for n in names),
        "gated": any(".g_proj." in n for n in names),
        "per_aux_norm": any(n.startswith("aux_hidden_norms.") for n in names),
        "fused_qkv": any(".qkv_proj." in n for n in names),
    }


class DFlashProposer(CapturableProposer):
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
    # Set True in _build_laguna when the drafter is CAUSAL + SLIDING-WINDOW. It is a property of the
    # checkpoint, not a switch: the z-lab DFlash drafters are constructed non-causal with
    # sliding_window=0, so `_block_mask` returns None and the block attends the ENTIRE prefix
    # bidirectionally. There is no fixed-capacity ring that can hold an unbounded prefix, so those
    # drafters keep the eager per-uid path. Same for the CCA-recurrent drafter (a different
    # architecture with no prefix at all). See `propose` for the dispatch.
    propose_capturable = False

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

        # WHICH BUILDER — decided by the checkpoint's own TENSORS, not its model_type/architectures.
        #
        # The dispatch this replaces asked `model_type == "laguna" or "Laguna" in architectures`, so
        # the windowed+causal build (and with it the prefix ring, and with it a capturable propose)
        # was reachable only by a checkpoint carrying that vendor string. Measured consequence on
        # this box: of seven DFlash drafters, ONE was captured — the one literally named laguna —
        # while Muse-Glimmer (`sliding_window: 2048`, 5x sliding_attention) and all three windowed
        # z-lab drafters ran eager, on a mask they were never trained with, and the engine reported
        # the reason as "the z-lab DFlash drafter is non-causal and unwindowed", which is a statement
        # about this dispatch and not about those checkpoints.
        #
        # `conv_qk` is the CCA-recurrent drafter's convolution and `self_attn.g_proj` is the gated
        # (laguna_xs) decoder layer's per-head softplus gate. Both are weights only that variant has,
        # so a checkpoint that ships the structure gets the right builder whether or not anyone has
        # taught minisgl its name. `cca_config` stays as an OR because the CCA trunk is a genuinely
        # different architecture and a declared marker for it is legitimate.
        from minisgl.models.weight import checkpoint_tensor_names

        self._ckpt_names = checkpoint_tensor_names(draft_model_path)
        self._probe = dflash_trunk_probe(self._ckpt_names)
        # Per-layer (causal, window), from the declared fields, upstream's rule. Both builders below
        # consume these — so the mask a drafter gets is a function of what it DECLARES, in one place.
        self._layer_causal, self._layer_window = dflash_layer_masks(hf, num_layers)

        self._is_cca = self._probe["cca"] or bool(getattr(hf, "cca_config", None))
        if self._is_cca:
            self._build_cca(engine, hf, cfg, dfc, folder, hidden, num_layers)
            return
        self._is_cca = False
        # Gated (laguna_xs) GQA trunk: fused qkv, per-head softplus g_proj gate, per-aux hidden
        # norms. Same fc->KV-prefix mechanism as the ungated Qwen3 trunk, different decoder layer —
        # which is why this is a LOADER/layer-type difference, not a separate mask policy.
        self._is_laguna = self._probe["gated"]
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
        # The drafter's rope SCHEME, not just its theta. Only a real scheme becomes scaling (mirrors
        # models/config.py); "default"/absent stays None so every existing drafter is unchanged.
        # Passed as a hashable tuple because the rope builder is cached. Non-scalar entries (e.g. an
        # mrope_section list) are dropped — they are multimodal-only and would break the cache key.
        rope_scaling = None
        if isinstance(rp, dict) and str(rp.get("rope_type", "default")) not in ("default", "None"):
            rope_scaling = tuple(
                sorted((k, v) for k, v in rp.items() if isinstance(v, (str, int, float, bool)))
            )
            logger.info_rank0(
                f"spec-decode: DFlash drafter rope scheme={rp.get('rope_type')} "
                f"({dict(rope_scaling)}) — applied, not defaulted")
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
        # Read through `cfg`, not `dfc.get`: Muse-Glimmer's assistant checkpoint ships NO
        # `dflash_config` sub-dict and states `target_layer_ids` at the TOP LEVEL, which both the
        # sub-dict lookup and the differently-named speculators fallback miss. The assert below then
        # fires, which is the good outcome — the bad one is capturing the WRONG layers, where
        # nothing errors and acceptance just quietly degrades.
        ids = cfg("target_layer_ids", "aux_hidden_state_layer_ids")
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
                rope_scaling=rope_scaling,
                max_position=max_pos,
                draft_vocab_size=draft_vocab if self._compressed else None,
                own_embed=self._compressed,
                # The mask this checkpoint DECLARES, per layer. Previously omitted entirely, so the
                # model took its defaults (causal=False, sliding_window=0) and every windowed
                # drafter that reached this path silently attended its whole prefix bidirectionally.
                layer_causal=self._layer_causal,
                layer_window=self._layer_window,
                per_aux_norm=self._probe["per_aux_norm"],
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
        # _kv[uid] = per-layer [k_buf, v_buf]; _kv_fill[uid] = valid rows; _kv_end[uid] = the ABSOLUTE
        # position that prefix ends at (see _init_prefix_kv). Only active on the full-context
        # (aux.dim()==3) path; disable via MINISGL_DFLASH_PERSIST_KV=0 for the recompute path
        # (diagnostic). MINISGL_DFLASH_CTX_WINDOW no longer disables it — the delta is tracked in
        # absolute positions, so a capped aux buffer is fine; that knob is now purely a scheduler-side
        # cap on the accumulated aux (and it used to silently revert this whole fast path).
        self._persist = os.environ.get("MINISGL_DFLASH_PERSIST_KV", "1") not in ("0", "false", "no")
        self._ctx_window = int(os.environ.get("MINISGL_DFLASH_CTX_WINDOW", "0") or 0)
        # Size the ring from what the drafter DECLARES, not from a family assumption. A drafter whose
        # every layer is windowed has a bounded prefix and therefore a fixed-capacity ring and a
        # capturable propose; anything else records the real reason.
        self._finish_prefix_kv(engine)

    def _finish_prefix_kv(self, engine) -> None:
        """Size the prefix K/V and decide capture from the DECLARED per-layer mask. One policy, used
        by every attention-trunk drafter, so "can this be captured?" is asked of the checkpoint's
        structure instead of its name.

        A group is one distinct (causal, window) pair — the same partition vLLM makes when it splits
        a mixed drafter into multiple KV cache groups (`_group_causal: dict[gid, bool]`, baked into
        the captured metadata). One group whose window is bounded is the case the existing single
        ring already expresses, so it captures today. Mixed/unbounded groups need one ring PER GROUP
        (an unbounded group sized by a max-context cap, as vLLM's FullAttentionSpec is bounded by
        max_model_len); until those are allocated the reason recorded below states the actual
        structural fact and its cost, and never claims the checkpoint is something it is not."""
        bounded = all(w > 0 for w in self._layer_window)
        # EVERY attention-trunk drafter is capturable, mixed included: the ring is sized per layer
        # (window+slack for a windowed layer, a max-context cap for a `full_attention` one) and the
        # captured body builds one mask per distinct (causal, window, capacity). The only thing that
        # can stop it now is memory or an explicit opt-out, and both say so.
        if self._persist:
            self._init_prefix_kv(int(max(self._layer_window)) if bounded else 0)
            self.init_propose_capture(engine)
            return
        self._init_prefix_kv(int(max(self._layer_window)) if bounded else 0)
        self.propose_uncapturable_reason = (
            "MINISGL_DFLASH_PERSIST_KV=0 — the persistent prefix ring is off")

    # ---- persistent prefix K/V (ring) ------------------------------------------------------------
    def _init_prefix_kv(self, window: int) -> None:
        """Size the per-request persistent prefix K/V from the drafter's own attention window.

        `window == 0` (z-lab: causal=False, sliding_window=0 -> `_block_mask` returns None) means the
        drafter attends the WHOLE prefix bidirectionally, so nothing may be dropped and the cache
        stays an unbounded append. Laguna (causal + window 512) can only ever read the newest
        `window` rows, so the cache is a fixed-capacity COMPACTING buffer:

          * capacity C = window + _KV_SLACK. The per-step append is at most block_size rows, so the
            newest `window` rows are always resident; the slack is amortisation headroom, not
            correctness. On overflow the newest `window` rows are memmoved to the front ONCE
            (~1 MB/layer) every ~C-window/append steps, instead of `torch.cat` reallocating and
            copying the WHOLE prefix on every step of every request (10 full-prefix copies/req/step,
            O(P) forever, 614 MB/uid resident at a 30k prefix).
          * COMPACTING, not modulo. A modulo ring permutes the key order, which reorders the probs·V
            reduction and would force `_block_mask` onto explicit absolute positions. Keeping the
            rows contiguous keeps the sliced form numerically equivalent to the unsliced one.
        """
        self._kv_window = window
        self._kv_cap = (window + _KV_SLACK) if window > 0 else 0
        # How many trailing aux rows the SCHEDULER needs to keep for us (0 = all). The drafter reads
        # at most `window` prefix rows and we re-project at most one accepted block per step, so
        # window + block_size is exactly sufficient — everything older is masked to -inf anyway.
        self.aux_ctx_cap = (window + self._block_size) if window > 0 else 0
        self._kv: dict[int, list] = {}       # uid -> per-layer [k_buf, v_buf]
        self._kv_fill: dict[int, int] = {}   # uid -> valid rows in those buffers
        # uid -> ABSOLUTE end position of the prefix already projected (== req.cached_len at the time
        # of projection). Deliberately absolute rather than "rows projected": once the scheduler caps
        # the aux buffer at `aux_ctx_cap`, its length P STOPS GROWING, so a `P > cached_rows` delta
        # test silently stops projecting anything. cached_len never stops growing.
        self._kv_end: dict[int, int] = {}

    def _append_prefix_kv(self, uid: int, new_kv: list, rebuild: bool) -> list:
        """Append the freshly-projected tail rows to uid's persistent prefix K/V; return the live
        per-layer (k, v) views. `rebuild` forces a cold start (uid reuse / inconsistent delta)."""
        n = new_kv[0][0].shape[0]
        C, W = self._kv_cap, self._kv_window
        cache = None if rebuild else self._kv.get(uid)
        if cache is None:
            self._kv.pop(uid, None)
            if C == 0:  # unbounded (bidirectional drafter): plain append, as before
                self._kv[uid] = [[k, v] for (k, v) in new_kv]
                self._kv_fill[uid] = n
                return self._kv[uid]
            cache = [
                [torch.empty((C,) + k.shape[1:], dtype=k.dtype, device=k.device),
                 torch.empty((C,) + v.shape[1:], dtype=v.dtype, device=v.device)]
                for (k, v) in new_kv
            ]
            self._kv[uid] = cache
            fill = 0
        else:
            fill = self._kv_fill.get(uid, 0)

        if C == 0:
            for l, (k, v) in enumerate(new_kv):
                cache[l][0] = torch.cat([cache[l][0], k], dim=0)
                cache[l][1] = torch.cat([cache[l][1], v], dim=0)
            self._kv_fill[uid] = fill + n
            return cache

        if n >= C:  # a single append bigger than the ring: only its newest C rows can ever be read
            for l, (k, v) in enumerate(new_kv):
                cache[l][0].copy_(k[-C:])
                cache[l][1].copy_(v[-C:])
            self._kv_fill[uid] = C
            return cache
        if fill + n > C:
            # Compact: keep the newest rows the window can still reach, memmoved to the front. The
            # .clone() is required — copy_ between OVERLAPPING slices of the same tensor is UB.
            keep = min(fill, W, C - n)
            for l in range(len(cache)):
                for t in (0, 1):
                    buf = cache[l][t]
                    if keep > 0:
                        buf[:keep].copy_(buf[fill - keep : fill].clone())
            fill = keep
        for l, (k, v) in enumerate(new_kv):
            cache[l][0][fill : fill + n].copy_(k)
            cache[l][1][fill : fill + n].copy_(v)
        self._kv_fill[uid] = fill + n
        return cache

    # ---- CAPTURED propose: the fixed-shape ring + the four hooks -------------------------------
    def init_propose_capture(self, engine) -> None:
        """Allocate the persistent per-slot prefix-K/V RING and the static propose I/O.

        Why a ring pool and not the per-uid compacting buffers this replaces: a captured graph records
        kernel argument POINTERS, so a dict of freshly allocated per-uid tensors can never be baked
        into one. The pool is ONE tensor per layer, `[slots, C, Hkv, hd]`, indexed by the same stable
        `req.table_idx` the page_table and the GDN/CCA recurrent state use, plus ONE extra NULL row
        that bucket-padding and discarded projections write into.

        MODULO ring, not the compacting buffer. Compaction is a conditional memmove — host control
        flow, which a graph cannot contain. The cost of going modulo is that the ring's column order
        is no longer position order, so the mask can no longer be built from concatenation indices:
        `_ppos[slot, col]` carries each column's ABSOLUTE position and the mask is
        `pos <= qpos and (qpos - pos) < window`. That is strictly more general, and it incidentally
        removes the latent bug where a nonzero MINISGL_DFLASH_POS_OFF made RoPE positions
        non-contiguous while `_block_mask` still assumed they were.

        Capacity C = window + block: a query at the END of the block reaches back `window` positions,
        and the ring must additionally absorb the block's worth of newly committed positions before
        the oldest needed key is overwritten."""
        d = self._draft
        dev, dt = self._device, self._dtype
        L = len(d.layers)
        Hkv, hd = d.layers[0].num_kv_heads, d.layers[0].head_dim
        A = self._block_size                  # positions committed per step is at most one block
        self._pc_A = A
        # PER-LAYER ring capacity — this is what lets a MIXED drafter be captured at all.
        # A `sliding_attention` layer only ever reads `window` rows back, so window + slack is
        # sufficient. A `full_attention` layer reads the WHOLE prefix, which has no natural bound —
        # so it gets a max-context CAP, exactly as vLLM bounds its FullAttentionSpec by
        # max_model_len. Sizing every layer at the largest capacity would work and is what a single
        # ring forces, but it is wasteful precisely where it hurts: on a 6-layer drafter with one
        # full layer that is 6x the max-context ring instead of 1x plus five small ones.
        full_cap = int(os.environ.get("MINISGL_DFLASH_FULL_CAP") or 0) or min(
            int(getattr(engine, "max_seq_len", 0) or 8192), 8192)
        self._layer_cap = [(w + _KV_SLACK) if w > 0 else (full_cap + A)
                           for w in self._layer_window]
        self._layer_win = list(self._layer_window)
        self._layer_cau = list(self._layer_causal)
        self._pc_C = max(self._layer_cap)
        self._window = self._kv_window
        # How much aux the SCHEDULER must retain for us. `_init_prefix_kv` derives this from a single
        # window and answers "all of it" (0) when unbounded — which, now that an unbounded layer has
        # a max-context-capped ring, would have the scheduler grow a buffer larger than anything the
        # ring can hold. Bound it by the widest ring plus one block.
        self.aux_ctx_cap = max(self._layer_cap) + A
        self._num_aux = len(self.capture_layer_ids or [])
        self._hidden = int(d.hidden_size)
        vocab = int(engine.model.model.embed_tokens.num_embeddings)

        # Slot space. `req.table_idx` runs 0..max_running_req, so the pool would like that many rows;
        # a row costs L*C*Hkv*hd*2 (K and V) bytes, which at a 512-window Laguna drafter is ~10.8 MB.
        # Cap it against the memory actually free at build time (the KV pool is already allocated by
        # now) rather than trusting max_running_req: a request whose slot falls outside the pool
        # simply skips propose and decodes plain, which is lossless.
        from minisgl.engine.graph import get_free_memory

        per_slot = sum(self._layer_cap) * Hkv * hd * dt.itemsize * 2
        want = int(engine.page_table.shape[0])
        # `or "0.30"` not a dict default: docker-compose's `VAR: "${VAR:-}"` sets the variable to the
        # EMPTY STRING, so the key IS present and `os.environ.get(k, default)` returns "" — which
        # float() raises on and which kills the serve at boot, not at the call site. Every other env
        # reader in this file already uses the `or` form; these three KV_FRAC readers did not, so
        # forwarding any of them through compose (the only way this repo serves) was fatal.
        # RESERVE RUNTIME HEADROOM FIRST. `KV_FRAC * free` alone is not a budget: it spends a share of
        # what is free at BUILD time and leaves the rest to be consumed by the KV pool's own growth,
        # activations and graph pools. Measured failure — qwen35b-awq, MEM_RATIO=0.86, a 360 MB ring
        # for a 6-layer drafter with one full_attention layer: free went 1.31 -> 0.32 GiB at boot and
        # the serve then died mid-generation on a 146 MiB allocation. It did NOT die under a short
        # benchmark, only under long generations, so the cost is invisible to a probe that stops at a
        # few hundred tokens. Subtract the headroom before taking the fraction.
        # The floor is a RUNTIME floor, and KV_FRAC is deliberately NOT applied on top of it: taking a
        # fraction of "free minus floor" double-discounts and still left 360 MB spent here on the pair
        # that could least afford it. What must hold is simply `free - ring >= floor`, because the
        # thing that OOMs is a long generation's transient workspace, which scales with context and
        # is invisible to any boot-time measurement. 1.0 GiB is not a guess: the measured failure
        # allocated 146 MiB with 0 bytes free after the ring had taken free from ~1.3 to ~0.3 GiB.
        _head = float(os.environ.get("MINISGL_DFLASH_RING_HEADROOM_GB") or "1.0") * (1 << 30)
        _free = get_free_memory(dev)
        budget = int(max(0.0, _free - _head))
        fits = budget // max(per_slot, 1)
        self._ring_budget_note = (
            f"free={_free / 1e9:.2f}GB floor={_head / 1e9:.2f}GB "
            f"budget={budget / 1e6:.0f}MB per_slot={per_slot / 1e6:.0f}MB fits={fits}")
        if fits < 1:
            # Honest refusal: this is a MEMORY fact about this box and this operating point, measured
            # here, not a claim about the checkpoint. Say what it would have cost so the operator can
            # trade KV pool for it deliberately (lower --memory-ratio, or raise KV_FRAC).
            self.propose_capturable = False
            self.propose_uncapturable_reason = (
                f"the prefix ring needs {per_slot / 1e6:.0f} MB/slot but only "
                f"{budget / 1e6:.0f} MB is budgetable ({_free / 1e9:.2f} GiB free minus "
                f"{_head / 1e9:.2f} GiB runtime headroom, x KV_FRAC) — lower --memory-ratio to buy "
                f"room, or raise MINISGL_DFLASH_KV_FRAC / lower MINISGL_DFLASH_RING_HEADROOM_GB")
            logger.warning_rank0(f"spec-decode: DFlash propose stays EAGER — "
                                 f"{self.propose_uncapturable_reason}")
            return
        # An explicit CAP, because the automatic budget cannot see the future. Everything allocated
        # after this point (spec-verify graphs, and then a long generation's context-scaled
        # workspace) competes for the same memory, so "free right now" over-states what the ring can
        # afford: measured on qwen35b-awq, free was 2.82 GB here and 0.32 GB by the time serving
        # started. Slots are the only lever that does not change numerics — a request whose
        # table_idx falls outside the pool simply decodes plain, which is lossless — so expose them
        # and let the operator trade concurrency-with-drafting against KV pool.
        _cap = int(os.environ.get("MINISGL_DFLASH_RING_SLOTS") or 0)
        self._pool_slots = max(1, min(want, fits, _cap or want))
        self._null_slot = self._pool_slots
        S = self._pool_slots + 1
        self._pk = [torch.zeros(S, c, Hkv, hd, device=dev, dtype=dt) for c in self._layer_cap]
        self._pv = [torch.zeros(S, c, Hkv, hd, device=dev, dtype=dt) for c in self._layer_cap]
        # One position map PER DISTINCT CAPACITY (usually one; two for a mixed drafter). The ring is
        # modulo, so a column means nothing without the absolute position it currently holds, and
        # rings of different capacity map the same position to different columns.
        self._ppos = {c: torch.full((S, c), _NO_POS, dtype=torch.int64, device=dev)
                      for c in sorted(set(self._layer_cap))}

        G = S  # static I/O rows: one per pool slot (+NULL), which bounds the captured bucket too
        Q = self._block_size
        self._g_slots = torch.zeros(G, dtype=torch.int64, device=dev)
        # [slot, ring column, absolute position, rope position] for the A projected prefix rows/req
        self._g_pre = torch.zeros(4, G, A, dtype=torch.int64, device=dev)
        # [block token ids, block RoPE positions, block ABSOLUTE positions]
        self._g_blk = torch.zeros(3, G, Q, dtype=torch.int64, device=dev)
        self._aux_stage = torch.zeros(G, A, self._num_aux, self._hidden, device=dev, dtype=dt)
        self._g_out = torch.zeros(G, Q - 1, dtype=torch.int64, device=dev)
        # Block logits live in a static buffer because DDTree reads the per-position top-K marginals
        # of the SAME forward; a graph's internal tensors are not addressable from Python afterwards.
        # fp32 so it is lossless whatever kernel family the LM head dispatched to (the bf16 decode
        # GEMV below M=16, minv above it) — DDTree reads log-probs off this, not off the argmax.
        self._g_logits = torch.zeros(G * (Q - 1), vocab, device=dev, dtype=torch.float32)
        # Causal mask WITHIN the block. Position-independent and window-independent (the block is
        # `block_size` wide and the window is far larger), so it is a constant, not per-step data.
        tri = torch.arange(Q, device=dev).view(Q, 1) >= torch.arange(Q, device=dev).view(1, Q)
        self._blk_tri = torch.where(tri, 0.0, float("-inf")).to(torch.float32)
        # ...and its BIDIRECTIONAL twin, for a `full_attention` layer: within the block it sees every
        # position, not just the ones behind it. Both are constants, so a mixed drafter costs one
        # extra [Q, Q] tensor and no per-step work.
        self._blk_open = torch.zeros(Q, Q, device=dev, dtype=torch.float32)
        # Pinned host staging -> three H2D copies per step for the whole batch.
        self._h_slots = torch.zeros(G, dtype=torch.int64, pin_memory=True)
        self._h_pre = torch.zeros(4, G, A, dtype=torch.int64, pin_memory=True)
        self._h_blk = torch.zeros(3, G, Q, dtype=torch.int64, pin_memory=True)
        self._ar_A = torch.arange(A, dtype=torch.int64)     # host, for the staging arithmetic
        self._ar_Q = torch.arange(Q, dtype=torch.int64)
        self._slot_uid: dict[int, int] = {}   # pool slot -> owning uid (reset the ring on reuse)
        self._ring_end: dict[int, int] = {}   # pool slot -> ABSOLUTE end position already projected
        # uid -> ring slot, plus the free list. The ring is its OWN slot space, sized by memory;
        # `free(uid)` returns the slot so a finished request's ring row is reused rather than lost.
        self._slot_of_uid: dict[int, int] = {}
        self._free_slots: list = list(range(self._pool_slots))
        self._rebuilds = 0                    # cold/gap eager rebuilds (evidence, not a knob)
        self.propose_capturable = True
        self.init_propose_capture_state(engine, tag="DFlash")
        # Report the per-layer geometry, not one number: on a mixed drafter the interesting fact is
        # that the windowed layers are small and only the full_attention one pays a max-context ring.
        _bytes = 2 * S * sum(self._layer_cap) * Hkv * hd * dt.itemsize
        _grp = sorted(set(zip(self._layer_cau, self._layer_win, self._layer_cap)))
        logger.info_rank0(
            f"spec-decode: DFlash propose ring (slots={self._pool_slots}+NULL of {want}, "
            f"block={Q}, {L} layers in {len(_grp)} group(s) "
            f"(causal,window,cap)={_grp}, prefix-KV {_bytes / 1e6:.0f} MB, "
            f"logits buf {self._g_logits.numel() * 4 / 1e6:.0f} MB) [{self._ring_budget_note}]")

    @torch.inference_mode()
    def _rebuild_ring(self, slot: int, aux: torch.Tensor, end: int, m: int) -> None:
        """EAGER cold/gap rebuild of one slot's resident window: project the newest `m` committed aux
        positions and scatter them into the ring. Runs once per request (its first propose, or after
        a gap the fixed block-sized tail cannot bridge), never in the steady state — the counter is
        reported so "captured" can't quietly mean "rebuilding every step"."""
        dev = self._device
        P = int(aux.shape[1])
        # ONE PASS PER RING CAPACITY, each fed only as many rows as that ring can hold. Feeding the
        # same `m` to every layer scatters m rows into a c-column ring whenever c < m — duplicate
        # columns, and `_pk`/`_ppos` are separate scatters that can then resolve the duplicates
        # DIFFERENTLY, leaving the mask certain a column holds position p while the K/V there is
        # p'. `m` comes in as min(P, max capacity), so this only ever narrows it.
        # CHUNKED. The projection is one GEMM per layer with M = the row count, and this path is the
        # only place M is unbounded — it grows with the ring capacity, i.e. with context. Measured:
        # a full-attention ring of 8208 rows faults the queue (HSA_STATUS_ERROR_EXCEPTION 0x1016)
        # while 6160 and 4112 are clean, and the fault survives every mask/capture variation, so the
        # one-shot large-M projection through the drafter's quantised linears is what changed across
        # that boundary. Chunking also stops a cold rebuild from becoming a single multi-thousand-row
        # stall on the decode path. The result is identical: each row's K/V depends only on its own
        # aux and position, so the split is arithmetically inert.
        for c in sorted(set(self._layer_cap)):
            ids = [l for l, cc in enumerate(self._layer_cap) if cc == c]
            mc = min(m, c)
            ppos = self._ppos[c]
            ppos[slot].fill_(_NO_POS)
            for lo in range(0, mc, _REBUILD_CHUNK):
                hi = min(lo + _REBUILD_CHUNK, mc)
                # rows [P-mc+lo, P-mc+hi) carry absolute positions [end-mc+lo, end-mc+hi)
                rows = aux[:, P - mc + lo : P - mc + hi].permute(1, 0, 2).contiguous().to(self._dtype)
                pos = torch.arange(end - mc + lo, end - mc + hi, dtype=torch.int64, device=dev)
                col = pos % c                  # unique for mc <= c, which the min above guarantees
                ws = torch.full((hi - lo,), slot, dtype=torch.int64, device=dev)
                self._draft.project_prefix_into(
                    rows, pos.to(torch.int32), self._pk, self._pv, ws, col, layer_ids=ids)
                ppos[slot, col] = pos
        self._rebuilds += 1

    def stage_propose(
        self, reqs: List["Req"], num_draft: int, ctx: ProposeContext, topk: int = 0, **kw
    ) -> Optional[StagedPropose]:
        A, C, Q = self._pc_A, self._pc_C, self._block_size
        rows: List[int] = []
        budget: List[int] = []
        hs, hpre, hblk, ar = self._h_slots, self._h_pre, self._h_blk, self._ar_A
        for i, req in enumerate(reqs):
            # The block emits up to Q-1 drafts; clamp to the per-step and per-request budgets.
            k_i = max(0, min(num_draft, Q - 1, req.remain_len - 1))
            aux = ctx.aux_hidden.get(req.uid)
            if k_i <= 0 or aux is None or aux.dim() != 3 or aux.shape[1] < 1:
                continue
            # RING SLOT, ALLOCATED — not `req.table_idx` reused as one. `table_idx` indexes the
            # page_table, whose row count is max_running_req+1 and is INDEPENDENT of how many ring
            # slots memory allowed. Using it directly means a pool smaller than the page table drops
            # every request whose row happens to sit above the cut, and it is not "the excess few":
            # measured with a 2-slot ring against a 5-row page table, 100% of steps came back
            # width 0 (`accept-len 0.00, replay=0`) — spec silently OFF, at half the eager
            # throughput. A pool of N must serve N requests, whichever rows they landed on.
            slot = self._slot_of_uid.get(req.uid)
            if slot is None:
                if not self._free_slots:
                    self._pc_warn_once(
                        "slot", f"all {self._pool_slots} prefix-ring slots are in use (memory-capped)"
                                " — further concurrent requests decode plain, which is lossless")
                    continue
                slot = self._free_slots.pop()
                self._slot_of_uid[req.uid] = slot
                self._slot_uid[slot] = req.uid
                self._ring_end.pop(slot, None)
            end = int(req.cached_len)
            P = int(aux.shape[1])
            covered = self._ring_end.get(slot)
            if covered is None or end < covered or end - covered > A:
                self._rebuild_ring(slot, aux, end, min(P, C))
            self._ring_end[slot] = end

            j = len(rows)
            m = min(P, A)
            # Stage the FIXED block-sized aux tail. The leading A-m rows (only before the request has
            # committed a full block) are zero and are routed to the NULL slot below.
            self._aux_stage[j, A - m :].copy_(aux[:, P - m : P].permute(1, 0, 2))
            if m < A:
                self._aux_stage[j, : A - m].zero_()
            p = end - A + ar                                # absolute position of each staged row
            live = ar >= (A - m)
            hs[j] = slot
            hpre[0, j] = torch.where(live, torch.full_like(p, slot),
                                     torch.full_like(p, self._null_slot))
            hpre[1, j] = torch.where(live, p % C, torch.zeros_like(p))
            hpre[2, j] = torch.where(live, p, torch.full_like(p, _NO_POS))
            hpre[3, j] = p.clamp(min=0)                     # RoPE position (never negative)
            hblk[0, j, 0] = int(req.input_ids[end])         # anchor
            hblk[0, j, 1:] = self._mask_token_id
            hblk[1, j] = end + self._pos_off + self._ar_Q   # RoPE positions of the block
            hblk[2, j] = end + self._ar_Q                   # ABSOLUTE positions (mask)
            rows.append(i)
            budget.append(k_i)
        if not rows:
            return None
        B = len(rows)
        self._g_slots[:B].copy_(hs[:B], non_blocking=True)
        self._g_pre[:, :B].copy_(hpre[:, :B], non_blocking=True)
        self._g_blk[:, :B].copy_(hblk[:, :B], non_blocking=True)
        return StagedPropose(B, rows, budget)

    def pad_propose_rows(self, bs: int, bucket: int) -> None:
        """Route rows [bs, bucket) entirely at the NULL slot: their projections land in a row nobody
        reads, their prefix mask is all -inf (every NULL column carries _NO_POS) and only their own
        block's causal diagonal survives — so the softmax still has a live key and cannot produce a
        NaN that a later kernel would propagate out of the graph."""
        self._g_slots[bs:bucket].fill_(self._null_slot)
        self._g_pre[0, bs:bucket].fill_(self._null_slot)
        self._g_pre[1, bs:bucket].zero_()
        self._g_pre[2, bs:bucket].fill_(_NO_POS)
        self._g_pre[3, bs:bucket].zero_()
        self._g_blk[:, bs:bucket].zero_()
        self._aux_stage[bs:bucket].zero_()

    def propose_body(self, bs: int) -> None:
        """THE CAPTURED BODY: fixed block-sized prefix projection -> absolute-position mask -> one
        batched denoising forward -> head. Reads only the static buffers' first `bs` rows; writes the
        ring, `_g_out` and `_g_logits`. No host sync, no data-dependent shape, one fixed trip count."""
        d = self._draft
        A, Q = self._pc_A, self._block_size
        m = bs * A
        # 1) Re-project the newest A committed positions into the ring. Idempotent for the ones
        #    already there (the aux of a committed position never changes), so a fixed row count can
        #    stand in for the step's variable number of newly accepted positions.
        #    Columns are derived ON DEVICE from the staged ABSOLUTE position (`pos % capacity`) rather
        #    than staged from the host: with per-layer capacities there is no single column to stage,
        #    and a modulo is cheaper than the extra H2D anyway. Pure tensor ops, so still capturable.
        pre = self._g_pre[:, :bs]
        ws, wp = pre[0].reshape(m), pre[2].reshape(m)
        cols = [torch.remainder(wp, c) for c in self._layer_cap]
        d.project_prefix_into(
            self._aux_stage[:bs].reshape(m, self._num_aux, self._hidden),
            pre[3].reshape(m).to(torch.int32), self._pk, self._pv, ws, cols)
        for c, ppos in self._ppos.items():
            ppos[ws, torch.remainder(wp, c)] = wp
        # 2) Additive mask PER LAYER from ABSOLUTE positions (see init_propose_capture for why not
        #    concat idx). Built once per distinct (causal, window, capacity), so a uniform drafter
        #    still materialises exactly one mask and hands the same tensor to every layer.
        slots = self._g_slots[:bs]
        qa = self._g_blk[2, :bs].unsqueeze(2)            # [bs, Q, 1]
        cache, masks = {}, []
        for l in range(len(d.layers)):
            key = (self._layer_cau[l], self._layer_win[l], self._layer_cap[l])
            if key not in cache:
                causal_l, win_l, cap_l = key
                pa = self._ppos[cap_l][slots].unsqueeze(1)   # [bs, 1, C]
                # A column that has never been written (or belongs to a previous owner) carries
                # _NO_POS. The window test used to kill those implicitly — _NO_POS is so negative
                # that `qa - pa` exceeds any window — but an UNBOUNDED layer applies no window test,
                # so liveness has to be explicit or a full_attention layer would attend garbage.
                # EXPAND to the query axis up front. `pa` is [bs, 1, C] and only the causal / window
                # terms carry a Q axis, so a layer with NEITHER (a full_attention layer: bidirectional
                # AND unbounded) would leave `keep` at [bs, 1, C] and the concat with the [bs, Q, Q]
                # block mask fails. Broadcasting here makes the mask well-formed for every group.
                keep = (pa != _NO_POS).expand(bs, Q, cap_l)
                if causal_l:
                    keep = keep & (pa <= qa)
                if win_l > 0:
                    keep = keep & ((qa - pa) < win_l)
                blk = self._blk_tri if causal_l else self._blk_open
                cache[key] = torch.cat(
                    [torch.where(keep, 0.0, float("-inf")).to(torch.float32),
                     blk.expand(bs, Q, Q)], dim=2)          # [bs, Q, C+Q]
            masks.append(cache[key])
        # 3) One batched denoising forward over [anchor, mask, mask, ...].
        noise = d.embed(self._g_blk[0, :bs].reshape(-1)).to(self._dtype).view(bs, Q, -1)
        hidden = d.denoise_batched(
            noise, self._g_blk[1, :bs].to(torch.int32), self._pk, self._pv, slots, masks)
        # 4) Head over block rows 1..Q-1 only (row 0 is the known anchor and is never read).
        n = bs * (Q - 1)
        logits = d.head(hidden[:, 1:].reshape(n, -1))
        self._g_logits[:n] = logits
        if d.has_markov:
            # DSpark, batched: same semi-autoregressive walk, anchors are block column 0.
            ids = d.markov_block_argmax(
                logits.view(bs, Q - 1, -1), self._g_blk[0, :bs, 0].to(torch.int64)
            ).reshape(-1)
        else:
            ids = logits.argmax(dim=-1)
        if self._compressed:
            ids = ids + self._d2t[ids]                   # draft vocab -> target vocab (on device)
        self._g_out[:bs] = ids.view(bs, Q - 1)

    def read_drafts(
        self, reqs: List["Req"], staged: StagedPropose, topk: int = 0, **kw
    ) -> List[List[int]]:
        out: List[List[int]] = [[] for _ in reqs]
        Q = self._block_size
        drafts = self._g_out[: staged.bs].cpu().tolist()      # ONE D2H for the whole step
        for i, k_i, row in zip(staged.rows, staged.budget, drafts):
            out[i] = row[:k_i]
            if self._dbg:
                print(f"[dflash-dbg] uid={reqs[i].uid} k={k_i} draft={out[i]}", flush=True)
        if topk > 0:
            # DDTree: per-position top-K marginals of the SAME forward, read off the static logits
            # buffer. Two batched D2H copies for the step, not two per request.
            n = staged.bs * (Q - 1)
            lp = torch.log_softmax(self._g_logits[:n].float(), dim=-1)
            tv, ti = lp.topk(topk, dim=-1)
            if self._compressed:
                ti = ti + self._d2t[ti]
            ti_all, tv_all = ti.cpu().tolist(), tv.cpu().tolist()
            for j, (i, k_i) in enumerate(zip(staged.rows, staged.budget)):
                b = j * (Q - 1)
                self._ddtree_topk[id(reqs[i])] = (ti_all[b : b + k_i], tv_all[b : b + k_i])
        return out

    def reset_propose_state(self) -> None:
        """Undo what the warmup/capture dummy batch wrote. It ran entirely on the NULL slot, so only
        that row's position vector needs clearing; no live request can have observed anything."""
        for _ppos in self._ppos.values():
            _ppos[self._null_slot].fill_(_NO_POS)
        self._slot_uid.clear()
        self._ring_end.clear()
        self._slot_of_uid.clear()
        self._free_slots = list(range(self._pool_slots))
        self._rebuilds = 0

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
        self._persist = False
        self._ctx_window = 0
        self._init_prefix_kv(0)
        self.propose_uncapturable_reason = (
            "the CCA-recurrent drafter fuses its context into a single position — it owns no prefix "
            "KV, so there is no fixed-shape propose body to capture")

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
        # Derived, not re-read. The old form was `causal = bool(dfc.get("causal", True))` — a key the
        # upstream `speculators` schema never writes (it declares `sliding_window_non_causal`, and
        # vLLM treats `dflash_config.causal` as an OVERRIDE, not the source). It landed on the right
        # answer for this checkpoint only because its default happened to match. `dflash_layer_masks`
        # applies the real rule, override included, for every drafter.
        sliding_window = max(self._layer_window)
        causal = all(self._layer_causal)

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
                    layer_causal=self._layer_causal,
                    layer_window=self._layer_window,
                    per_aux_norm=self._probe["per_aux_norm"],
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
        # SAME policy as every other attention-trunk drafter — the ring and the capture decision are
        # a function of the declared per-layer mask, not of which builder ran. This is the line that
        # used to be reachable only for a checkpoint named laguna.
        self._finish_prefix_kv(engine)

        # PROMPT-PREFILL SEED (full-context path only). Without it the drafter's aux prefix is built
        # append-only from ACCEPTED GENERATED positions (scheduler.py:_spec_aux_hidden), so its context
        # at generated position P is min(P, window) tokens OF ITS OWN OUTPUT and it never sees the
        # prompt at all. Measured consequence (docs/CONTINUANCE §11.5): accept-len is a monotone
        # function of P that saturates exactly at this window — real code 3.5 at P<32 rising to 7.4 at
        # P=512-1024 — so every request starts starved and stays starved for ~512 tokens. Capping the
        # drafter prefix at 8 positions collapses real code 6.150 -> 3.667, which is the causal control.
        #
        # Seeding is the fix, and it is bounded by the window rather than the prompt: the drafter's
        # own mask drops every key older than `sliding_window`, so seeding more than window+block is
        # numerically inert AND would charge O(prompt) on every later propose (attend_block re-reads
        # the whole prefix each step). window+block covers the oldest key the last block query can
        # still see. CCA does NOT get this: its seed is a single fused position by construction, so it
        # has no prefix to seed (see _build_cca).
        # HOW MUCH prompt to seed is a real trade-off, and the obvious answer is WRONG. Seeding
        # `sliding_window + block_size` (528) was measured NET-NEGATIVE on long prompts: accept-len
        # 4.635 -> 3.910 (-15.6%) on a 3.5k-token code prompt, giving back essentially the whole spec
        # win — while being +22.5% on a 95-token prompt. The window is a FIXED 512 keys, so a
        # window-sized prompt seed EVICTS the model's own recent output for ~500 generated tokens,
        # and that recent output is what actually predicts the next token. So `P` in §11.5's table
        # never indexed "window occupancy"; it indexed "how much of the window is my own output" —
        # which is why the starvation reading of that table did not survive contact with the fix.
        # The default below is therefore a small context anchor, not a window-full.
        # MINISGL_DFLASH_SEED_TAIL overrides it (0 disables seeding entirely), mirroring how
        # `_moe_block_m` exposes MINISGL_MOE_BLOCK_M for autotuning around a derived default.
        self.supports_prefill_seed = True
        _tail_env = os.environ.get("MINISGL_DFLASH_SEED_TAIL")
        if _tail_env not in (None, ""):
            self.prefill_aux_tail = max(0, int(_tail_env))
            self.supports_prefill_seed = self.prefill_aux_tail > 0
        elif sliding_window > 0:
            self.prefill_aux_tail = min(_SEED_TAIL_DEFAULT, sliding_window + self._block_size)
        else:
            self.prefill_aux_tail = _SEED_TAIL_DEFAULT

        logger.info_rank0(
            f"DFlash Laguna drafter: {num_layers}L h={hidden} heads={num_heads}/{num_kv_heads} "
            f"block={self._block_size} mask_id={self._mask_token_id} window={sliding_window} "
            f"causal={causal} aux_layers={self.capture_layer_ids} "
            f"prefill_seed=on(tail={self.prefill_aux_tail or 'all'}) "
            f"prefix_kv={'ring cap=' + str(self._kv_cap) if self._kv_cap else 'unbounded'} "
            f"aux_cap={self.aux_ctx_cap or 'unbounded'} (bf16, borrow target embed/head)"
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
            # A DraftLinear may be TP-SHARDED, so it must slice the FULL checkpoint tensor itself;
            # a bare setattr would install the whole matrix on every rank and silently un-shard it.
            if hasattr(mod, "full_shape") and leaf == "weight":
                assert tuple(mod.full_shape) == tuple(t.shape), (
                    f"shape mismatch {key}: model {tuple(mod.full_shape)} vs ckpt {tuple(t.shape)}")
                mod.load(t, self._device)
                return
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

        # FULL dims: the fused checkpoint tensor is whole, and each DraftLinear shards its own slice.
        q_dim = d.layers[0].full_q_dim
        kv_dim = d.layers[0].full_kv_dim
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
            layer.q_proj.load(qkv[:q_dim].contiguous(), self._device)
            layer.k_proj.load(qkv[q_dim:q_dim + kv_dim].contiguous(), self._device)
            layer.v_proj.load(qkv[q_dim + kv_dim:].contiguous(), self._device)
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
        # EXPLICIT "do not quantize" sentinel. The empty string cannot express it: compose forwards
        # an unset variable as "", and tools/serve.sh then does `: "${MINISGL_DFLASH_QUANT:=nvfp4}"`
        # — and `:=` substitutes when the variable is unset OR EMPTY. So through the normal launch
        # path an unquantized drafter was UNREACHABLE, and asking for one silently served nvfp4
        # (caught by a reserve of 1.37 GiB where bf16 needs ~3.15, and legs byte-identical to the
        # nvfp4 arm). A non-empty sentinel survives `:=`, so bf16 becomes expressible.
        if quant_mode in ("none", "bf16", "off"):
            quant_mode = None
        if quant_mode not in (None, "fp8", "int8", "nvfp4"):
            raise ValueError(
                f"MINISGL_DFLASH_QUANT must be fp8|int8|nvfp4|none, got {quant_mode!r}")
        # MIXED precision under nvfp4. Meta's own GGUF build of the Muse-Glimmer drafter is not
        # uniform: Q4_K everywhere except `ffn_down`, which they hold at Q6_K (and every norm at
        # F32). Mirror that carve-out.
        #
        # Note honestly that a weight-error probe does NOT reproduce their choice —
        # tools/dflash_quant_probe.py measures every leaf of this drafter at the same relative L2
        # (~0.095 nvfp4 vs ~0.027 fp8, a flat 3.6x), with down_proj marginally the BEST. That is a
        # limitation of the proxy, not evidence against the carve-out: down_proj's sensitivity is an
        # ACTIVATION-outlier property (its input is post-SwiGLU), which a weight-norm probe cannot
        # see. So this follows the vendor's recipe plus that mechanism, and costs +0.27 GiB.
        _FP8_LEAVES = ("down_proj",)

        def mode_for(key: str) -> str:
            if quant_mode != "nvfp4":
                return quant_mode
            return "fp8" if any(f".{leaf}." in key for leaf in _FP8_LEAVES) else "nvfp4"
        # ALWAYS stage in HOST RAM, never straight to the card. The old form
        # (`device="cpu" if quant_mode else str(self._device)`) put the WHOLE checkpoint on EVERY
        # rank whenever the drafter was unquantized, so:
        #   * TP sharding bought nothing at load time — each rank still materialised the full
        #     checkpoint before slicing it (this OOM'd Qwen3.6-27B's 3.45 GB bf16 drafter at 14.28
        #     of 15.92 GiB, trying to allocate 170 MiB);
        #   * and under expandable_segments the freed staging stays RESERVED, which is the same
        #     failure `_load_laguna_weights` already documents and avoids.
        # Staging on CPU and moving each FINAL (already-sliced, already-quantized) tensor to the card
        # keeps the peak at one tensor, not one checkpoint.
        sd = st.load_file(path, device="cpu")
        d = self._draft
        if quant_mode:
            logger.info_rank0(f"DFlash drafter: weight-only {quant_mode} quant of the draft linears")

        # Tensor-name DIALECTS for the same trunk. Muse-Glimmer's assistant checkpoint namespaces the
        # hidden-fusion pair under `encoder.` and spells the norm `output_norm_enc`; the tensors,
        # their shapes and the math are identical to z-lab's. Aliasing at the key level keeps ONE
        # loader instead of forking it per vendor — the fork is what the CCA drafter needed, and only
        # because its LAYERS differed, not its names.
        aliases = {
            "fc.weight": ("encoder.fc.weight",),
            "hidden_norm.weight": ("encoder.output_norm_enc.weight",),
        }

        def assign(mod, leaf, key, quant=False):
            if key not in sd:
                for alt in aliases.get(key, ()):
                    if alt in sd:
                        key = alt
                        break
            assert key in sd, f"DFlash ckpt missing {key}"
            t = sd[key].to(self._dtype).contiguous()
            want = mod.full_shape if (hasattr(mod, "full_shape") and leaf == "weight") \
                else getattr(mod, leaf).shape
            assert tuple(want) == tuple(t.shape), (
                f"shape mismatch {key}: model {tuple(want)} vs ckpt {tuple(t.shape)}"
            )
            if quant and quant_mode:
                # t on CPU -> packed on GPU; mode is per-LEAF (see `mode_for`). load_quant slices to
                # this rank BEFORE quantizing, so a shard's per-output-channel scales are its own.
                mod.load_quant(t, mode_for(key), self._dtype, self._device)
            elif hasattr(mod, "full_shape") and leaf == "weight":
                mod.load(t, self._device)   # may be TP-sharded; it slices the full tensor itself
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

        # DSpark: the Markov bigram logit-bias head, when the checkpoint ships one. Detected by
        # TENSOR PRESENCE, matching this file's builder dispatch — a plain DFlash checkpoint has
        # neither key and the drafter behaves exactly as before. `confidence_head.*` is loaded by
        # nothing yet: it drives DSpark's VARIABLE verify length, which needs a dynamic draft budget
        # and so cannot ride the fixed-shape captured propose. Deliberately deferred, not forgotten.
        # MINISGL_DSPARK_MARKOV=0 loads the checkpoint but SKIPS the head, so the same drafter runs
        # as plain DFlash. Diagnostic: it separates "the Markov application is wrong" from "the
        # drafter/backbone is mismatched", which acceptance alone cannot.
        if "markov_head.markov_w1.weight" in sd and os.environ.get("MINISGL_DSPARK_MARKOV") != "0":
            d.set_markov(
                sd["markov_head.markov_w1.weight"].to(self._dtype).contiguous().to(self._device),
                sd["markov_head.markov_w2.weight"].to(self._dtype).contiguous().to(self._device),
            )
            print(f"[dflash] DSpark Markov head loaded (rank="
                  f"{sd['markov_head.markov_w1.weight'].shape[1]}); block positions decode "
                  f"semi-autoregressively", flush=True)

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
        """Dispatch on the DRAFTER, not on a flag.

        A causal + sliding-window drafter (Laguna) has a bounded prefix, so it rides the shared
        captured path (`CapturableProposer.propose` -> stage/body/read). A non-causal, unwindowed
        z-lab drafter attends its ENTIRE prefix bidirectionally and a CCA-recurrent drafter has no
        prefix at all; neither has a fixed-capacity shape to capture, so both keep the eager
        per-request path below. That is a checkpoint property, not an env switch."""
        self._ddtree_topk: dict[int, tuple] = {}
        if self.propose_capturable:
            return super().propose(reqs, num_draft, ctx, topk=topk)
        return self._propose_eager(reqs, num_draft, ctx, topk)

    @torch.inference_mode()
    def _propose_eager(
        self, reqs: List["Req"], num_draft: int, ctx: ProposeContext, topk: int = 0
    ) -> List[List[int]]:
        out: List[List[int]] = [[] for _ in reqs]
        draft = self._draft
        device = self._device
        mask_id = self._mask_token_id
        B = self._block_size
        # DDTree (topk>0): the per-position top-K MARGINALS (target-vocab ids + log-probs) of the k_i
        # drafted positions per req are stashed in `self._ddtree_topk`, keyed by id(req), for
        # build_draft_tree. Mirrors the scheduler's _tidar_block_predict topk path; cleared by
        # `propose` each call (both dispatch arms fill the same dict).
        if self._is_cca:
            return self._propose_cca(reqs, num_draft, ctx, topk, out)
        # ONE host sync per STEP, not per request. The drafted ids stay on device through the whole
        # per-request loop and are read back once at the end; the old `ids.tolist()` inside the loop
        # stalled the queue N times per step, serialising every request's ~150 kernel launches behind
        # the previous request's completion.
        pend: List[tuple] = []      # (out_index, req, anchor_tok, base_pos, k_i, ids[k_i])
        pend_topk: List[tuple] = []  # (id(req), ti[k_i,topk], tv[k_i,topk])
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
                if self._persist:
                    # FAST PATH: project only the newly-committed tail and append it to the per-uid
                    # persistent K/V; reuse the cached prefix for the rest. The scheduler only appends
                    # ACCEPTED positions to `aux`, so everything before the tail is unchanged from the
                    # previous step and needs no re-projection. free(uid) drops the cache on finish.
                    #
                    # The delta is measured in ABSOLUTE positions (`req.cached_len`), not in aux rows:
                    # once the scheduler caps the aux buffer at `aux_ctx_cap` its row count P stops
                    # growing, and the old `P > rows_already_projected` test would then silently stop
                    # projecting anything at all.
                    uid = req.uid
                    prev_end = self._kv_end.get(uid, 0)
                    n = req.cached_len - prev_end
                    rebuild = uid not in self._kv or not (0 <= n <= P)
                    if rebuild:
                        n = P
                    if n > 0:
                        new_aux = aux[:, P - n : P].permute(1, 0, 2).contiguous().to(self._dtype)
                        new_pos = torch.arange(
                            req.cached_len - n, req.cached_len, dtype=torch.int32, device=device
                        )
                        new_kv = draft.project_prefix(new_aux, new_pos)  # per-layer (k_ctx, v_ctx)
                        cache = self._append_prefix_kv(uid, new_kv, rebuild)
                        self._kv_end[uid] = req.cached_len
                    else:
                        cache = self._kv.get(uid)
                    if cache is None:
                        # No prefix at all (an empty aux buffer on a cold uid). Nothing to condition
                        # on, so skip this req -> plain decode. Lossless; verify gates every token.
                        continue
                    fill = self._kv_fill[uid]
                    prefix_kv = [(k[:fill], v[:fill]) for (k, v) in cache]
                else:
                    # Recompute path (MINISGL_DFLASH_PERSIST_KV=0, diagnostic): re-project the prefix
                    # every step. Window-slice the AUX first, so `fuse_aux` (the 10240->2048 fc, the
                    # single biggest FLOP here) also runs O(window) instead of O(context).
                    if 0 < self._kv_window < P:
                        aux = aux[:, P - self._kv_window :]
                        P = self._kv_window
                        ctx_start = req.cached_len - P
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
            # SLICE THE ROWS BEFORE THE HEAD. Positions 1..k_i are the speculation (position 0 is the
            # known anchor) and rows k_i+1..B-1 were never read, yet the borrowed target head scored
            # all B of them over the full 100352-wide vocab, TP-all_gathered them, and paid a
            # .permute().contiguous() copy on the result. Bit-identical, unlike (a)-(c): the head is
            # M-INVARIANT BY CONSTRUCTION (layers/embedding.py:16-21 — per-(row,col) independent fp32
            # dot in a fixed K-order), and B=16 -> k_i=15 rows stays inside the same
            # `dense_bf16_gemv` band (_LMHEAD_GEMV_MMAX = 16), so no kernel-family switch either.
            block_logits = draft.head(hidden[1 : 1 + k_i])  # [k_i, vocab]
            if draft.has_markov:
                # DSpark: bias each position by the token chosen at the previous one (anchor first).
                anchor_dev = torch.as_tensor(anchor_tok, dtype=torch.int64, device=device)
                ids = draft.markov_block_argmax(block_logits, anchor_dev)  # [k_i]
            else:
                ids = block_logits.argmax(dim=-1)  # [k_i] draft-vocab ids
            if self._compressed:
                ids = ids + self._d2t[ids]  # draft id -> target id (delta map)
            pend.append((i, req, anchor_tok, base_pos, k_i, ids))
            if topk > 0 and k_i > 0:
                lp = torch.log_softmax(block_logits.float(), dim=-1)  # [k_i, vocab]
                tv, ti = lp.topk(topk, dim=-1)  # [k_i, topk], descending
                if self._compressed:
                    ti = ti + self._d2t[ti]  # draft ids -> target ids (delta map)
                pend_topk.append((id(req), ti, tv))

        # --- the ONE host sync of the step -------------------------------------------------------
        if pend:
            flat = torch.cat([p[5] for p in pend]).tolist()  # single D2H for the whole batch
            off = 0
            for (i, req, anchor_tok, base_pos, k_i, _ids) in pend:
                out[i] = flat[off : off + k_i]
                off += k_i
                if self._dbg:
                    print(f"[dflash-dbg] uid={req.uid} anchor={anchor_tok} base_pos={base_pos} "
                          f"B={B} k={k_i} draft={out[i]}", flush=True)
        if pend_topk:
            # DDTree only. Two batched D2H copies for the whole step instead of two PER REQUEST.
            ti_all = torch.cat([p[1] for p in pend_topk]).cpu().tolist()
            tv_all = torch.cat([p[2] for p in pend_topk]).cpu().tolist()
            off = 0
            for (rid, ti, _tv) in pend_topk:
                n = ti.shape[0]
                self._ddtree_topk[rid] = (ti_all[off : off + n], tv_all[off : off + n])
                off += n
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
        # Drop the finished/aborted request's persistent draft K/V (and its cursors).
        self._kv.pop(uid, None)
        self._kv_fill.pop(uid, None)
        self._kv_end.pop(uid, None)
        if self.propose_capturable:
            # Release the ring slot. Stale K/V need not be zeroed — clearing the slot's ownership is
            # enough, because the next owner's first propose rebuilds the ring and resets every
            # column's ABSOLUTE POSITION to _NO_POS, which is what the mask actually consults.
            # Return it to the FREE LIST too, or a capped ring leaks a slot per finished request and
            # degrades to "no drafting at all" after `_pool_slots` completions.
            s = self._slot_of_uid.pop(uid, None)
            if s is not None and s not in self._free_slots:
                self._free_slots.append(s)
            for s, u in list(self._slot_uid.items()):
                if u == uid:
                    del self._slot_uid[s]
                    self._ring_end.pop(s, None)
