from __future__ import annotations
import dataclasses
import os
import re
from dataclasses import dataclass
from typing import Any, Collection, Dict, Optional, Tuple
from transformers import PretrainedConfig

from minisgl.quant.config import QuantConfig


def _full_layer_override(config, attr: str, layer_types) -> Optional[int]:
    """`attr` on the first full-attention layer, when transformers holds it as a PER-LAYER override.

    transformers 5.17 folds Gemma-4's `global_head_dim` / `num_global_key_value_heads` into
    per-layer overrides of `head_dim` / `num_key_value_heads` and DROPS the original fields, so
    `getattr(config, "global_head_dim")` is None there and the full layers would silently be built at
    the sliding geometry (256/8 instead of 512/2). None when the config carries no per-layer view
    (older transformers, which keep the global_* fields) or no full-attention layer."""
    view = config.__dict__.get("_heterogeneity_spec") and getattr(config, "per_layer_config", None)
    if view is None or not layer_types:
        return None
    for i, t in enumerate(layer_types):
        if t == "full_attention":
            return getattr(view[i], attr, None)
    return None

@dataclass(frozen=True)
class RotaryConfig:
    head_dim: int
    rotary_dim: int
    max_position: int
    base: float
    scaling: Dict[str, Any] | None
    # Interleaved RoPE pairing (GLM-4.x `rope_interleave=True`) vs NeoX rotate-half. Applying the
    # wrong pairing scrambles relative position -> grammatical-but-degenerate generation.
    interleave: bool = False


def _rotary_from_subdict(
    sub: Dict[str, Any], head_dim: int, max_position: int
) -> RotaryConfig:
    """Build a RotaryConfig from one rope sub-dict (Laguna nests two: full_attention → yarn,
    sliding_attention → default). Handles per-scheme partial rotary + scaling independently."""
    rope_theta = sub["rope_theta"]
    partial = sub.get("partial_rotary_factor")
    rotary_dim = int(head_dim * partial) if partial is not None else head_dim
    rope_type = sub.get("rope_type", "default")
    if rope_type == "proportional":
        # Gemma4's partial rotary is NOT the usual "rotate a contiguous prefix". The reference builds
        # a HALF-WIDTH inv_freq of `partial*head_dim/2` live frequencies whose exponent denominator is
        # the FULL head_dim, then ZERO-PADS it back out to head_dim/2 and rotates full-width. With
        # head_dim 512 / partial 0.25 that rotates channel pairs (i, i+256) for i<64 — the set
        # {0..63} u {256..319} — whereas a prefix-partial rope pairs (i, i+64) over {0..127} and
        # divides the exponent by 128. Different frequencies AND different pairing: silently wrong
        # relative position, grammatical-but-degenerate output. Carrying it as FULL rotary with a
        # zero-padded inv_freq (built in layers/rotary.py `case "proportional"`) reproduces it
        # exactly, and keeps the NeoX tail_hip kernel usable since rotary_dim == head_dim.
        rotary_dim = head_dim
    scaling = sub if rope_type not in (None, "default") else None
    return RotaryConfig(
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        max_position=max_position,
        base=rope_theta,
        scaling=scaling,
    )


def _norm_output_gate(value) -> str:
    """Normalize the checkpoint's GDN `output_gate_type` to the two activations the kernels
    implement. swish IS silu (upstream maps it identically); absent means the HF default (silu).
    Anything else raises HERE, at config load: the gate multiplies every GDN layer's output, and
    running the wrong one produces degenerate-but-grammatical text with no error anywhere else."""
    if value is None:
        return "silu"
    v = str(value).strip().lower()
    if v in ("silu", "swish"):
        return "silu"
    if v == "sigmoid":
        return "sigmoid"
    raise ValueError(
        f"unsupported GDN output_gate_type {value!r}: the gated-norm kernels implement "
        "silu/swish and sigmoid"
    )


# Layer-type names denoting a FULL-CONTEXT attention layer: one that keeps a paged KV entry for
# every token. A SET, not the bare string "full_attention", because the SAME checkpoint reports
# DIFFERENT names depending on the installed transformers:
#
#   transformers 5.14.1  AutoConfig does not register `qwen4_exp_text` and RAISES, so
#                        utils/hf.py:257 falls back to PretrainedConfig.from_dict(config.json),
#                        which preserves the on-disk name -> "full_attention".
#   transformers 5.17.0  AutoConfig DOES register it, and the registered class renames the layer
#                        type -> "qwen_sparse_attention".
#
# So Qwen3.8-Flash-Next broke by being UNDERSTOOD: it booted for months only because transformers
# could not parse its config, and the upgrade that taught transformers to read it is what broke the
# engine. Worth stating because the instinct on seeing this is "the checkpoint changed" -- it did
# not; `config.json` still says full_attention on disk.
#
# The failure was silent and total: `full_attn_layer_ids` returned [] while `gdn_layer_ids` claimed
# 36 of 48 layers, leaving 12 in NEITHER partition, each reaching Qwen4ExpDecoderLayer with
# attn_kv_id=None to die on a bare assert naming neither the layer type nor the version.
#
# "sparse" describes how the layer SELECTS keys (the QSA indexer picks a subset), not how much
# context it RETAINS: the whole context stays resident, so it is full-context for every sizing and
# indexing purpose here. Sliding-window layers are deliberately NOT in this set -- they keep a
# window-bounded ring pool and are counted separately.
_FULL_CONTEXT_ATTENTION = frozenset({"full_attention", "qwen_sparse_attention"})
# Every layer-type name this file can partition. An unknown one must fail LOUDLY at config time.
_KNOWN_LAYER_TYPES = _FULL_CONTEXT_ATTENTION | {"linear_attention", "sliding_attention"}


@dataclass(frozen=True)
class ModelConfig:
    num_layers: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int
    vocab_size: int
    intermediate_size: int
    rms_norm_eps: float
    rotary_config: RotaryConfig
    hidden_act: str
    tie_word_embeddings: bool
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    norm_topk_prob: bool
    # Qwen2-MoE-style shared expert (always-on, runs alongside the top-k routed experts) +
    # its sigmoid gate. 0 for models without a shared expert (Qwen3-MoE, Mixtral).
    shared_expert_intermediate_size: int
    model_type: str
    architectures: list[str]
    quant: QuantConfig | None = None
    # ---- MLA (multi-head latent attention; DeepSeek / GLM-4.x MoE). None for non-MLA models. ----
    # Set ONLY when the config carries kv_lora_rank, so every other model keeps is_mla=False. The
    # absorbed-decode latent dim is kv_lora_rank + qk_rope_head_dim; the per-head qk dim is
    # qk_nope_head_dim + qk_rope_head_dim (== head_dim, overwritten in from_hf for MLA).
    kv_lora_rank: int | None = None
    q_lora_rank: int | None = None
    qk_nope_head_dim: int | None = None
    qk_rope_head_dim: int | None = None
    v_head_dim: int | None = None
    # ---- fine-grained MoE routing (GLM-4.x / DeepSeek "noaux_tc": sigmoid score + correction
    # bias + group-limited top-k, normalize, scale). Defaults are the no-op / plain-top-k case. ----
    n_group: int = 1
    topk_group: int = 1
    routed_scaling_factor: float = 1.0
    first_k_dense_replace: int = 0  # first K decoder layers use a dense MLP, not MoE
    n_shared_experts: int = 0  # always-on shared experts (added, not gated — GLM/DeepSeek style)
    num_nextn_predict_layers: int = 0  # GLM/DeepSeek MTP heads appended after the decoder
    mtp_num_hidden_layers: int = 0  # Qwen3.5 MTP head (mtp.* namespace); 0 = no MTP head
    # ---- GDN / linear-attention (Qwen3-Next / Qwen3.5 hybrid). None for dense models. ----
    # `layer_types[i]` is "linear_attention" (GDN) or "full_attention". Populated by from_hf
    # ONLY when the config carries linear-attention dims, so the dense path stays untouched.
    linear_num_key_heads: int | None = None
    linear_num_value_heads: int | None = None
    linear_key_head_dim: int | None = None
    linear_value_head_dim: int | None = None
    linear_conv_kernel_dim: int | None = None
    layer_types: tuple[str, ...] | None = None
    # GDN output-gate activation, NORMALIZED: "silu" (covers the checkpoint spellings silu/swish —
    # upstream maps swish->silu) or "sigmoid". Read from `output_gate_type`; absent -> "silu" (the
    # HF default). An unknown value raises in from_hf: better a loud load failure than 48 layers of
    # grammatical-but-degenerate output through the wrong gate. Was hardcoded SiLU end to end.
    gdn_output_gate: str = "silu"
    # Gemma2-style final-logit soft cap; None for models that don't declare it.
    final_logit_softcapping: float | None = None
    # ---- Sliding-window attention (SWA) hybrid (Laguna: repeating [full, sliding×3]). None for a
    # non-SWA model. `layer_types[i]` is "sliding_attention" (windowed) or "full_attention" (global);
    # a SWA layer keeps paged KV but capped at `sliding_window` tokens (its own ring pool), NOT full
    # context. Distinct from the GDN `layer_types` vocabulary ("linear_attention"/"full_attention") —
    # is_gdn_hybrid now REQUIRES a linear layer, so a SWA schedule never routes through the GDN path.
    sliding_window: int | None = None
    # Per-layer query/output head count (Laguna: 48 on full layers, 64 on sliding). None -> uniform
    # num_qo_heads for every layer. The model builder sizes each layer's q_proj from this.
    attn_head_counts: tuple[int, ...] | None = None
    # Second RoPE scheme for the SLIDING layers (Laguna: default θ=1e4, full-rotary). `rotary_config`
    # carries the FULL-attention rope (Laguna: yarn θ=5e5, partial-0.5). None -> every layer shares
    # `rotary_config`. The model builder picks per-layer by the attn schedule.
    sliding_rotary_config: RotaryConfig | None = None
    # ---- Gemma4 (`gemma4` / `diffusion_gemma`): a SWA hybrid whose two layer types differ in
    # head_dim AND kv-head count, not just in QO head count the way Laguna does. `head_dim` /
    # `num_kv_heads` above carry the FULL-attention geometry (512 / 2) because they size the MAIN
    # paged pool, which for a SWA hybrid holds exactly the full-attention layers; these two carry
    # the SLIDING geometry (256 / 8) for the separate ring pool. None -> the two types share one
    # geometry (every other SWA model), so nothing downstream has to branch.
    swa_head_dim: int | None = None
    swa_num_kv_heads: int | None = None
    # Softmax scale override. Gemma4 uses 1.0 — the usual 1/sqrt(d) temperature is folded into its
    # LEARNED k_norm (a near-constant 0.1260 on sliding / 0.0623 on full layers), so applying
    # head_dim**-0.5 on top would roughly double (sliding) or halve (full) the logit temperature
    # with no error anywhere. None -> the usual head_dim**-0.5.
    attn_softmax_scale: float | None = None
    # Gemma4 `attention_k_eq_v`: the FULL-attention layers ship no v_proj at all. V is not an alias
    # of the cached K — it is the PRE-norm, PRE-RoPE k_proj output passed through an unweighted
    # RMSNorm, so both tensors must still be materialised and cached independently.
    attention_k_eq_v: bool = False
    # Gemma: logits = c * tanh(logits / c), applied after lm_head. None -> no cap.
    final_logit_softcapping: float | None = None
    # Gemma: embeddings are scaled by sqrt(hidden_size), CAST TO THE WEIGHT DTYPE before the
    # multiply (fp16 -> 53.0625, not 53.0660). None -> no scaling.
    embed_scale: float | None = None
    # Muse-Glimmer: the logits are pre-scaled by this BEFORE the tanh softcap, giving
    # `T * tanh(lm_head(h) * m / T)`. `m = 1/sqrt(hidden/head_dim... ) = 1/sqrt(26)` in the shipping
    # checkpoint. It lands on the RETURNED logits, so it changes sampling (not just a training-time
    # loss scale) and any spec-decode verify path has to reproduce it. None -> no pre-scale.
    output_multiplier: float | None = None
    # Muse-Glimmer: the two SANDWICH post-norms (post_attention / post_feedforward, which sit on the
    # sublayer OUTPUT before the residual add) use their own, much tighter epsilon (1e-8) than the
    # two input-side norms (rms_norm_eps, 1e-5). Two epsilons in one layer is unusual enough that
    # collapsing them to one is a silent-quality bug. None -> reuse rms_norm_eps everywhere.
    post_norm_eps: float | None = None
    # Muse-Glimmer: per-layer RoPE base, where 0 means NoPE — that layer applies NO positional
    # encoding. Stored as the RAW per-layer list so `nope_layer_ids` can key on it; the nonzero
    # entries are all the global theta in the shipping checkpoint (the reference implementation only
    # reads this list as a boolean, so a per-layer NONZERO theta would be silently ignored upstream
    # too). None -> every layer ropes, which is every other model.
    layer_rope_theta: Tuple[float, ...] | None = None
    # ---- Block diffusion (DiffusionGemma). The decoder denoises a FIXED-length canvas of this many
    # tokens per block instead of emitting one token per step, so this is not a tuning knob: it sizes
    # the per-request scratch slots, widens the SWA ring stride, and fixes the query count of every
    # canvas forward. Read from the TOP-LEVEL config (`canvas_length`) — the text config knows
    # nothing about it. None for every autoregressive model, which is what `is_block_diffusion` keys
    # on, so no path anywhere branches on a model name.
    canvas_length: int | None = None
    # ---- Nemotron-H hybrid (Mamba-2 + MoE + a few global-attention layers). Populated by from_hf
    # ONLY when model_type == "nemotron_h".
    #
    # This family does not fit the `layer_types` vocabulary and is deliberately NOT squeezed into it.
    # Two reasons, and both are structural rather than cosmetic:
    #
    #  1. Every other model here has layer = mixer + MLP. Nemotron-H has ONE mixer per layer, and it
    #     is a mamba mixer, an attention mixer, or the MoE itself — 52 layers, 52 norms, one sublayer
    #     each. "moe" is a peer of "attention" in this schedule, not something that follows it.
    #  2. Mapping "mamba" onto "linear_attention" would make `is_gdn_hybrid` True and route this
    #     model into the GDN state cache, the GDN slot manager and the GDN kernels — a DIFFERENT
    #     recurrence. It would build, run, and be wrong.
    #
    # So the schedule gets its own field with its own vocabulary ("mamba"/"moe"/"attention") and
    # `layer_types` stays None, which keeps every GDN and SWA branch dead for this family.
    block_types: tuple[str, ...] | None = None
    # Mamba-2 geometry. inner = mamba_num_heads * mamba_head_dim (4096); the conv1d runs over
    # inner + 2*n_groups*ssm_state (6144) and in_proj emits 2*inner + 2*n_groups*ssm_state +
    # num_heads (10304) as [z, x, B, C, dt].
    mamba_num_heads: int | None = None
    mamba_head_dim: int | None = None
    mamba_ssm_state: int | None = None
    mamba_n_groups: int | None = None
    mamba_conv_kernel: int | None = None
    mamba_chunk_size: int | None = None
    mamba_dt_min: float | None = None
    mamba_dt_max: float | None = None
    mamba_conv_bias: bool = True
    mamba_proj_bias: bool = False
    # Nemotron-H's experts are NOT SwiGLU: `mlp_hidden_act: "relu2"`, a single up_proj into a squared
    # ReLU into down_proj, with no gate half. Every MoE in this repo before it was gate+up fused with
    # SiLU, so this is load-bearing — a gated path applied here silently halves the intermediate and
    # multiplies by the wrong thing. Read from the checkpoint, never defaulted.
    moe_act: str | None = None
    moe_shared_intermediate: int | None = None
    # ---- ZAYA CCA hybrid (cross-channel attention conv front-end + EDA/MOD MoE). None for non-Zaya.
    # Populated by from_hf ONLY when model_type == "zaya", so every other model keeps is_cca_hybrid
    # False. The schedule is implicit (even layer -> CCA attention, odd -> MoE), so there is no
    # layer_types list; cca_layer_ids derives it from num_layers. State dtype is fp32.
    is_cca: bool = False
    cca_time0: int | None = None  # conv kernel of conv_qk.0 (depthwise); padding TP0 = cca_time0-1
    cca_time1: int | None = None  # conv kernel of conv_qk.1 (grouped);  padding TP1 = cca_time1-1
    cca_num_k_heads: int | None = None  # num_query_groups (k/v heads) = 2
    cca_num_q_heads: int | None = None  # num_attention_heads (q heads)  = 8
    cca_head_dim: int | None = None  # 128
    cca_clamp_temp: bool = False  # if True key temp is exp(clamp(temp,1e-7,2.0)); else raw temp
    zaya_mlp_expansion: int | None = None  # router down_proj width (e.g. 256)
    zaya_use_eda: bool = False  # expert-decision-aggregation: thread prev router hidden across MoE
    zaya_use_mod: bool = False  # mixture-of-depths: extra "skip" expert at index num_experts
    scale_residual_merge: bool = False  # affine on the fp32 residual stream before each input_norm
    residual_in_fp32: bool = False  # carry the residual stream in fp32 across all layers
    # ---- Qwen4-Exp (Qwen3.8-Flash-Next) — hyper-connections, PLE n-gram block, QSA indexer ----
    # Populated ONLY when model_type is qwen4_exp / qwen4_exp_text, so every other family keeps the
    # None/() defaults and `is_qwen4_exp` stays False.
    #
    # HYPER-CONNECTIONS. The residual stream is `hc_count` copies wide (4 x 2560 = 10240) for the
    # WHOLE decoder: each block reads a mixed 2560-wide view and writes back into the wide stream.
    # `hc_lowrank` (320) is the rank of the mix gate's down/up pair. There is NO final `norm` tensor
    # in this checkpoint — the top-level `hyper_connection_mixer` plays that role.
    hc_count: int | None = None
    hc_lowrank: int | None = None
    # PLE (per-layer n-gram embedding). `ple_layer_ids` is stored 0-BASED here; the checkpoint's
    # config.json field is 1-BASED ([2] -> decoder index 1, which is where `layers.1.ple.*` lives).
    # Converting on the way in is load-bearing: attaching the block one layer late loads cleanly and
    # only degrades quality (see the bring-up plan, T4.3).
    ple_layer_ids: Tuple[int, ...] = ()
    ple_embed_dim: int | None = None      # 2560 = 16 n-gram heads x 160 (the row-table row width)
    ple_conv_kernel_size: int | None = None  # 4 (depthwise short conv over the WIDE stream)
    ngram_size: int | None = None            # 3
    heads_per_ngram: int | None = None       # 8 -> (ngram_size-1)*heads_per_ngram = 16 hash heads
    ngram_vocab_size_base: int | None = None            # 20_000_000
    make_ngram_vocab_size_divisible_by: int | None = None  # 128
    split_ngram_parts: int | None = None                # 128 table shards on disk
    # Two inputs to the n-gram HASH that the shipping config.json does not spell out, and that are
    # silent-wrong if guessed (a wrong value still yields a valid row id in the right head's band —
    # a real embedding from the wrong row, for every token, with no error anywhere):
    #   `ngram_seed`          `Qwen4ExpTextConfig.seed`, default 1234. It seeds `layer_multipliers`.
    #                         The checkpoint SHIPS those multipliers, and ple/hashing.py asserts the
    #                         derivation from this seed reproduces them exactly — the two check each
    #                         other rather than either being trusted.
    #   `ngram_eos_token_id`  the reference's `_shift_right_ignore_eos` fill and the initial 2-token
    #                         context. `config.eos_token_id` (first entry if it is a list).
    ngram_seed: int | None = None
    ngram_eos_token_id: int | None = None
    # QSA sparse-attention indexer (the 12 full-attention layers). Unimplemented (bring-up T5); the
    # dims are carried so the indexer's checkpoint tensors have somewhere to land and so the
    # <= indexer_budget "dense is bit-equivalent" argument can be asserted rather than assumed.
    indexer_budget: int | None = None
    indexer_compress_ratio: int | None = None
    indexer_head_dim: int | None = None
    indexer_kv_heads: int | None = None
    indexer_n_heads: int | None = None
    # The checkpoint declared a `quantization_config` that `QuantConfig.from_hf` did NOT parse, so
    # `quant` is None and every module will build FULL PRECISION. That is a silent-wrong condition
    # for any family (the loader then hands packed tensors to bf16 buffers, or the shapes simply
    # miss), so record the method name rather than let "unquantized" be indistinguishable from
    # "quantization we failed to read". None = nothing was declared, or it parsed fine.
    unparsed_quant_method: str | None = None

    @property
    def is_moe(self) -> bool:
        # A model with routed experts IS a MoE model (num_experts>0) — the principled signal, not a
        # model-name substring: Laguna (model_type=="laguna") and Zaya (=="zaya") carry no "moe" in
        # their type yet both route experts and need the engine's moe_backend + the loader's expert
        # stacking. The legacy substring/CCA checks are kept as a belt-and-suspenders for any family
        # that sets num_experts oddly.
        return self.num_experts > 0 or "moe" in self.model_type or self.is_cca_hybrid

    @property
    def is_mla(self) -> bool:
        """True for a multi-head latent-attention model (DeepSeek / GLM-4.x MoE)."""
        return self.kv_lora_rank is not None

    @property
    def is_mamba_hybrid(self) -> bool:
        """True for Nemotron-H. Keyed on an actual mamba layer, mirroring `is_gdn_hybrid`'s rule —
        presence of the layer kind, never merely "the list exists"."""
        return self.block_types is not None and any(t == "mamba" for t in self.block_types)

    @property
    def mamba_layer_ids(self) -> list[int]:
        """Decoder indices whose mixer is Mamba-2. Position in THIS list is the compact recurrent-slot
        id, mirroring gdn_layer_ids — the state cache is sized by len(), not by num_layers."""
        return [] if self.block_types is None else [
            i for i, t in enumerate(self.block_types) if t == "mamba"]

    @property
    def moe_block_layer_ids(self) -> list[int]:
        """Decoder indices whose mixer IS the MoE. Named `moe_block_` rather than `moe_` because a
        conventional MoE model has an MoE inside most layers; here it replaces the mixer."""
        return [] if self.block_types is None else [
            i for i, t in enumerate(self.block_types) if t == "moe"]

    @property
    def mamba_inner_dim(self) -> int | None:
        if self.mamba_num_heads is None or self.mamba_head_dim is None:
            return None
        return self.mamba_num_heads * self.mamba_head_dim

    @property
    def mamba_conv_dim(self) -> int | None:
        """Width the causal conv1d runs over: x plus B and C, NOT z and NOT dt."""
        inner = self.mamba_inner_dim
        if inner is None or self.mamba_n_groups is None or self.mamba_ssm_state is None:
            return None
        return inner + 2 * self.mamba_n_groups * self.mamba_ssm_state

    @property
    def mamba_in_proj_dim(self) -> int | None:
        """Width in_proj emits: [z, x, B, C, dt]. Checked against the checkpoint at load; a mismatch
        means the split order or the group count is wrong and every number downstream is garbage."""
        inner, conv = self.mamba_inner_dim, self.mamba_conv_dim
        if inner is None or conv is None or self.mamba_num_heads is None:
            return None
        return inner + conv + self.mamba_num_heads

    @property
    def is_gdn_hybrid(self) -> bool:
        """True for a GDN/linear-attention hybrid (interleaved linear + full layers). Keyed on the
        presence of an actual "linear_attention" layer, NOT merely `layer_types is not None` — a SWA
        model (Laguna) also carries a `layer_types` list ("sliding_attention"/"full_attention"), and
        must NOT be mistaken for a GDN hybrid (that would strand its 30 sliding layers out of the KV
        pool → OOB crash). The two schedules are disjoint by their layer-type vocabulary."""
        return self.layer_types is not None and any(
            t == "linear_attention" for t in self.layer_types
        )

    @property
    def is_swa_hybrid(self) -> bool:
        """True for a sliding-window-attention hybrid (Laguna). Its full layers keep a full-context
        paged KV cache; its sliding layers keep only a `sliding_window`-token ring. Disjoint from the
        GDN path (which has no `sliding_window` and uses "linear_attention" layers)."""
        return (
            self.sliding_window is not None
            and self.layer_types is not None
            and any(t == "sliding_attention" for t in self.layer_types)
        )

    @property
    def is_qwen4_exp(self) -> bool:
        """True for the Qwen3.8-Flash-Next backbone (`qwen4_exp` multimodal wrapper or its unwrapped
        `qwen4_exp_text` config). Text-only here: the vision tower is skipped by the loader, as it is
        for every other multimodal checkpoint this engine serves.

        NOTE this model is ALSO a GDN hybrid (`is_gdn_hybrid` is True: 36 of its 48 layers are
        linear-attention). Every dispatch that branches on `is_gdn_hybrid` must therefore test
        `is_qwen4_exp` FIRST, or a qwen4_exp checkpoint silently routes into the Qwen3.5 path — which
        has no hyper-connections, no PLE and a different key namespace."""
        return self.model_type in ("qwen4_exp", "qwen4_exp_text")

    @property
    def hc_hidden_size(self) -> int:
        """Width of the hyper-connection residual stream = hc_count * hidden_size (10240 for
        Qwen3.8-Flash-Next). Equals `hidden_size` for every model without hyper-connections, so a
        buffer sized by this is correct everywhere."""
        return self.hidden_size * (self.hc_count or 1)

    @property
    def is_muse_glimmer(self) -> bool:
        """True for the Muse-Glimmer backbone (multimodal wrapper `muse_glimmer` or its unwrapped
        text config). Text-only here: the vision tower is skipped by the loader, as it is for every
        other multimodal checkpoint this engine serves."""
        return self.model_type in ("muse_glimmer", "muse_glimmer_text")

    @property
    def nope_layer_ids(self) -> Tuple[int, ...]:
        """Layers that apply NO positional encoding — `layer_rope_theta[i] == 0`. Empty for every
        model without a per-layer theta list, so the ordinary all-rope path is unchanged. In
        Muse-Glimmer these coincide exactly with the FULL-attention layers (3, 7, …, 51): the global
        layers mix context with no positional prior while the sliding layers carry RoPE. They are
        derived independently rather than aliased to `full_attn_layer_ids`, because the coincidence
        is a property of this checkpoint's config, not a structural invariant."""
        if self.layer_rope_theta is None:
            return ()
        return tuple(i for i, t in enumerate(self.layer_rope_theta) if not t)

    @property
    def is_gemma4(self) -> bool:
        """True for the Gemma4 backbone (`gemma4` autoregressive, `diffusion_gemma` block-diffusion).
        Both share one 30-layer stack, so every structural branch keys on this, not on the head."""
        return self.model_type in ("gemma4", "gemma4_text", "diffusion_gemma", "diffusion_gemma_text")

    @property
    def is_block_diffusion(self) -> bool:
        """True for a block-diffusion decoder (DiffusionGemma): the model emits a whole
        `canvas_length` block per commit, refined over up to `max_denoising_steps` NON-CAUSAL
        forwards, instead of one token per step.

        Keyed on `canvas_length` — a field only a block-diffusion checkpoint carries — and NOT on
        model_type, because the backbone is shared with the autoregressive `gemma4` sibling and
        every structural branch below this one must key on the STACK, not the head."""
        return self.canvas_length is not None and self.canvas_length > 0

    @property
    def has_split_head_dim(self) -> bool:
        """True when the sliding and full layers do NOT share a head_dim, so the two KV pools need
        different geometry and the attention backend cannot cache one softmax scale / one reshape
        width for the whole model."""
        return self.swa_head_dim is not None and self.swa_head_dim != self.head_dim

    @property
    def swa_layer_ids(self) -> list[int]:
        """Global indices of the SLIDING-window layers, in order. The SWA ring KV pool is indexed by
        position in THIS list (a compact swa id), mirroring gdn_layer_ids / full_attn_layer_ids."""
        if self.layer_types is None:
            return []
        return [i for i, t in enumerate(self.layer_types) if t == "sliding_attention"]

    @property
    def num_swa_layers(self) -> int:
        return len(self.swa_layer_ids)

    @property
    def gdn_layer_ids(self) -> list[int]:
        """Global indices of the linear-attention (GDN) layers, in order. The GDN state
        cache is indexed by position in THIS list (gdn_layer_id), not the global layer_id."""
        if self.layer_types is None:
            return []
        return [i for i, t in enumerate(self.layer_types) if t == "linear_attention"]

    @property
    def num_gdn_layers(self) -> int:
        return len(self.gdn_layer_ids)

    @property
    def unknown_layer_types(self) -> tuple[str, ...]:
        """Layer-type names in `layer_types` this file cannot partition, in first-seen order.

        Exists because the failure mode of not checking is terrible: an unrecognised name falls out
        of BOTH `gdn_layer_ids` and `full_attn_layer_ids`, and the first symptom is a bare
        `assert attn_kv_id is not None` in a decoder layer — which names neither the offending layer
        type nor the transformers version that produced it. That cost a boot to diagnose once."""
        if self.layer_types is None:
            return ()
        seen: list[str] = []
        for t in self.layer_types:
            if t not in _KNOWN_LAYER_TYPES and t not in seen:
                seen.append(t)
        return tuple(seen)

    @property
    def full_attn_layer_ids(self) -> list[int]:
        """Global indices of the FULL-attention layers, in order. The paged KV pool is indexed by
        position in THIS list (a compact kv id) — the GDN/linear layers keep no paged KV, so
        indexing the pool by the global layer_id would allocate (and strand) a KV slot for every
        linear layer. For a non-hybrid model every layer is full-attention, so this is the identity
        [0..num_layers).

        Nemotron-H answers from `block_types`, and it MUST: only 6 of its 52 layers are attention, so
        falling through to the non-hybrid identity would size the paged pool for 52 layers and strand
        8.7x the KV it needs — a pure capacity loss with no error anywhere to attribute it to."""
        if self.block_types is not None:
            return [i for i, t in enumerate(self.block_types) if t == "attention"]
        if self.layer_types is None:
            return list(range(self.num_layers))
        return [i for i, t in enumerate(self.layer_types) if t in _FULL_CONTEXT_ATTENTION]

    @property
    def num_kv_layers(self) -> int:
        """Number of layers that keep a FULL-CONTEXT paged KV cache = the main pool's layer dimension.
        For a GDN hybrid only the full-attention layers do (the 3-in-4 linear layers keep fixed
        recurrent state instead). For a SWA hybrid (Laguna) only the full-attention layers keep a
        full-context pool; the sliding layers keep a SEPARATE window-bounded ring pool (num_swa_layers,
        sized by `sliding_window`, not full context) — so sizing the main pool by num_layers would
        over-allocate full-context KV for all 30 sliding layers (~3.8x waste at 32k). Every other
        family keeps num_layers unchanged (dense/MLA are all-attention; CCA uses its own compact id)."""
        if self.is_gdn_hybrid or self.is_swa_hybrid:
            return len(self.full_attn_layer_ids)
        return self.num_layers

    @property
    def gdn_conv_dim(self) -> int:
        """conv_dim = 2*key_dim + value_dim — the causal-conv channel count."""
        assert self.linear_key_head_dim is not None and self.linear_num_key_heads is not None
        key_dim = self.linear_key_head_dim * self.linear_num_key_heads
        value_dim = self.linear_value_head_dim * self.linear_num_value_heads
        return key_dim * 2 + value_dim

    @property
    def is_cca_hybrid(self) -> bool:
        """True for a ZAYA CCA hybrid (even layers = CCA attention, odd layers = MoE)."""
        return self.is_cca

    @property
    def cca_layer_ids(self) -> list[int]:
        """Global indices of the CCA (attention-bearing) layers, in order. The CCA conv-state
        cache AND the paged KV pool are indexed by position in THIS list (cca_layer_id), which is
        contiguous over the attention-bearing layers."""
        if not self.is_cca:
            return []
        return [i for i in range(self.num_layers) if i % 2 == 0]

    @property
    def num_cca_layers(self) -> int:
        return len(self.cca_layer_ids)

    @property
    def cca_conv_dim(self) -> int:
        """conv channel count C = (num_q_heads + num_k_heads) * head_dim (= 1280 for Zaya)."""
        assert self.cca_num_q_heads is not None and self.cca_num_k_heads is not None
        assert self.cca_head_dim is not None
        return (self.cca_num_q_heads + self.cca_num_k_heads) * self.cca_head_dim

    @property
    def cca_conv_width(self) -> int:
        """conv_states width TP = (cca_time0-1) + (cca_time1-1) (= 2 for Zaya)."""
        assert self.cca_time0 is not None and self.cca_time1 is not None
        return (self.cca_time0 - 1) + (self.cca_time1 - 1)

    @classmethod
    def from_hf(
        cls,
        config: PretrainedConfig,
        spec_algorithm: str = "mtp",
        ckpt_tensor_names: "Collection[str] | None" = None,
    ) -> ModelConfig:
        quant = QuantConfig.from_hf(config)  # quantization_config is top-level
        # A declared-but-unparsed quantization_config is the worst kind of silent failure: `quant`
        # comes back None, every module builds full precision, and the only symptom is a shape/key
        # mismatch deep in the loader (or, worse, a checkpoint whose packed tensors happen to fit).
        # Record WHICH method we could not read so a model builder can say so at boot.
        # (Qwen3.8-Flash-Next-NVFP4 hits this today: `quant_method: "modelopt"` has no arm in
        # QuantConfig.from_hf — bring-up plan T1.1.)
        # Same lookup QuantConfig.from_hf does (top-level, then text_config) — asking only the
        # top level would miss a wrapper that nests it and report "nothing declared".
        _decl_quant = getattr(config, "quantization_config", None)
        if _decl_quant is None and getattr(config, "text_config", None) is not None:
            _decl_quant = getattr(config.text_config, "quantization_config", None)
        unparsed_quant_method = None
        if quant is None and _decl_quant:
            _m = _decl_quant.get("quant_method") if isinstance(_decl_quant, dict) else None
            unparsed_quant_method = str(_m or "unknown")
        if quant is not None and ckpt_tensor_names:
            # Which modules does the checkpoint ACTUALLY ship quantized? A quantized module carries a
            # packing/scale tensor (`weight_packed`/`qweight`/`weight_scale`/`scales`/...); a
            # full-precision one carries only a bare `.weight`. Names are de-wrapped to the loader's
            # native space (the `language_model.` infix stripped) so they match what the model asks
            # with. See QuantConfig.ckpt_quantized for why the ignore list alone cannot decide this.
            _QSUFFIX = (".weight_packed", ".qweight", ".weight_scale", ".scales",
                        ".weight_global_scale", ".weight_scale_2", ".qzeros", ".weight_zero_point",
                        # DeepSeek-style BLOCKWISE fp8 names its scale `weight_scale_inv`, and
                        # `.weight_scale` is not a suffix of it — so without this entry a blockwise
                        # module ships fp8 bytes while being classified UNQUANTIZED, and the loader
                        # asks for a bf16 `.weight` it will never get. Qwen3.8-Flash-Next-MXFP4-FP8
                        # ships exactly this for its attention and GDN projections.
                        ".weight_scale_inv")

            def _native(n: str) -> str:
                # Mirror the loader's de-wrapping EXACTLY (same rewrites as quant _norm_ignore):
                # strip the multimodal `language_model.` infix, and collapse DiffusionGemma's
                # `model.decoder.` nesting to `model.`. Miss either and this set is keyed in a
                # different namespace than the names the model asks with — every lookup misses and
                # a fully-quantized model silently builds bf16.
                n = n.replace("language_model.", "")
                return "model." + n.removeprefix("model.decoder.") if n.startswith("model.decoder.") else n

            quant = dataclasses.replace(quant, ckpt_quantized=frozenset(
                _native(name[: -len(sfx)])
                for name in ckpt_tensor_names for sfx in _QSUFFIX if name.endswith(sfx)
            ))
        top = config
        if hasattr(config, "text_config") and config.text_config is not None:
            config = config.text_config
            # A multimodal wrapper whose `model_type` the installed transformers does NOT register
            # comes back from the generic `PretrainedConfig.from_dict` fallback with `text_config`
            # still a plain DICT — the class that would have promoted it does not exist. Promote it
            # here so the rest of `from_hf` can keep reading fields with `getattr`. Without this the
            # unwrap below dies on `'dict' object has no attribute 'architectures'`, which is what a
            # brand-new architecture (Muse-Glimmer on transformers < 5.15) hits FIRST — before any
            # model code runs, so it reads as "unsupported" rather than "your transformers is old".
            if isinstance(config, dict):
                _sub_type = config.get("model_type")
                config = PretrainedConfig.from_dict(dict(config))
                # `model_type` is a CLASS attribute, not an __init__ kwarg, so from_dict drops it and
                # the sub-config would report "" — the same wart cached_load_hf_config works around
                # for the top-level config. Restore it, or every `model_type`-gated branch below
                # (is_muse_glimmer, is_cca, …) silently reads the wrong architecture.
                if _sub_type:
                    config.model_type = _sub_type
            for attr in ("architectures", "rope_theta", "rope_scaling"):
                if not getattr(config, attr, None) and getattr(top, attr, None):
                    setattr(config, attr, getattr(top, attr))

        model_type = getattr(config, "model_type", "llama")
        # Qwen3.8-Flash-Next. Detected on model_type alone: `qwen4_exp` (the multimodal wrapper) and
        # `qwen4_exp_text` (what the text_config unwrap above leaves behind).
        _is_qwen4_exp = model_type in ("qwen4_exp", "qwen4_exp_text")
        # PORT_PLAN §detection: a Zaya CCA hybrid is identified by model_type=="zaya" OR an explicit
        # top-level `cca` flag — NOT the conjunction. AND silently disabled the entire CCA port for a
        # `zaya` checkpoint that omits `cca` (no error, garbage output); OR matches the spec and the
        # shipping checkpoints (which carry both).
        is_cca = (model_type == "zaya") or bool(getattr(config, "cca", False))
        # HF-format ZAYA fuses attn+MoE into ONE decoder layer (num_hidden_layers = #blocks); minisgl
        # models them as TWO alternating layers (even=CCA-attn, odd=MoE), so double the count. The
        # Megatron-export config already ships the split (80) count. See docs/zaya-port/HF_FORMAT_LOADER.md.
        #
        # DETECT ON A MEGATRON KEY, NEVER ON AN HF ONE. The old detector was `layer_types is not
        # None` ("the Megatron config lacks it"). That premise died: transformers >=5.13 SYNTHESIZES
        # `layer_types` (80x'hybrid') for any config carrying `sliding_window`, which the Megatron
        # export does — so the detector fired on the Megatron export, doubled 80 -> 160, and the
        # loader died with KeyError: 'model.layers.80.input_norm.weight' before any forward ran.
        # Every HF-side name is equally unusable for the same reason: measured under transformers
        # 5.13.0, the Megatron export's LOADED config also reports `num_experts_per_tok`=1,
        # `moe_intermediate_size`=2048, `router_hidden_size`=256 and `rms_norm_eps` purely from
        # ZayaConfig class defaults, though its config.json has none of them. Only the reverse
        # direction is safe: these Megatron-export spellings have NO class default, so getattr sees
        # them iff the file shipped them (measured absent on the HF-format ZAYA1-8B-MXFP4 config).
        # Tie it to `is_cca` so a non-ZAYA config that happens to carry one of these names (e.g.
        # `norm_epsilon`) cannot be dragged into this branch.
        _MEGATRON_ZAYA_KEYS = ("moe_router_topk", "ffn_hidden_size", "zaya_mlp_expansion",
                               "num_query_groups", "norm_epsilon")
        _megatron_zaya = is_cca and any(
            getattr(config, k, None) is not None for k in _MEGATRON_ZAYA_KEYS
        )
        num_layers = config.num_hidden_layers
        _hf_zaya = is_cca and not _megatron_zaya
        if _hf_zaya:
            num_layers = num_layers * 2
        # Zaya names kv heads `num_query_groups` (Megatron) / `num_key_value_heads` (HF); top-k
        # `moe_router_topk` (Megatron) / `num_experts_per_tok` (HF).
        num_kv_heads = (
            (getattr(config, "num_query_groups", None)
             or getattr(config, "num_key_value_heads", None))
            if is_cca
            else getattr(config, "num_key_value_heads", None)
        ) or config.num_attention_heads
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        tie_word_embeddings = getattr(config, "tie_word_embeddings", False)
        if is_cca:
            # ZAYA ties the lm_head to embed_tokens (no separate lm_head.weight in the checkpoint),
            # but the HF config omits tie_word_embeddings; force it on.
            tie_word_embeddings = True
        # All-MoE models (e.g. qwen3_5_moe, mlp_only_layers=[]) carry no dense `intermediate_size`.
        # ZAYA stores its (merged gate+up) FFN width as `ffn_hidden_size`; the per-gate width is half.
        intermediate_size = getattr(config, "intermediate_size", 0)
        # Routed-expert count: Mixtral/Qwen use num_local_experts/num_experts; GLM-4.x & DeepSeek
        # MoE use n_routed_experts.
        num_experts = (
            getattr(config, "num_local_experts", None)
            or getattr(config, "num_experts", None)
            or getattr(config, "n_routed_experts", 0)
        )
        # Gemma4 spells the routed top-k `top_k_experts`; everyone else uses `num_experts_per_tok`.
        # Defaulting to 0 here is silent death (the MoE routes nothing), so read both names.
        num_experts_per_tok = (
            getattr(config, "num_experts_per_tok", 0)
            or getattr(config, "top_k_experts", 0)
        )
        moe_intermediate_size = getattr(config, "moe_intermediate_size", 0)
        if is_cca:
            # ZAYA top-k: moe_router_topk (Megatron) / num_experts_per_tok (HF, read just above).
            num_experts_per_tok = (getattr(config, "moe_router_topk", None)
                                   or num_experts_per_tok or 1)
            # Per-expert FFN width: ffn_hidden_size//2 (Megatron ships the MERGED gate+up width) or,
            # for HF format (no ffn_hidden_size), the moe_intermediate_size read directly above.
            ffn_hidden_size = getattr(config, "ffn_hidden_size", 0) or 0
            if ffn_hidden_size:
                moe_intermediate_size = ffn_hidden_size // 2
        norm_topk_prob = getattr(config, "norm_topk_prob", False)
        if _is_qwen4_exp and not hasattr(config, "norm_topk_prob"):
            # The shipping config.json omits `norm_topk_prob`, so the False default above would
            # silently leave the top-10 router weights summing to < 1 — mis-scaled experts, i.e.
            # degenerate text with no error anywhere. The architecture's default is True, read out of
            # the reference rather than guessed: `Qwen4ExpTextConfig(Qwen3NextConfig)`
            # (sglang/srt/configs/qwen4_exp.py:15) inherits `norm_topk_prob=True`
            # (sglang/srt/configs/qwen3_next.py:214), and the MoE block consumes it as
            # `renormalize=config.norm_topk_prob` (sglang/srt/models/qwen2_moe.py:322).
            norm_topk_prob = True
        shared_expert_intermediate_size = getattr(config, "shared_expert_intermediate_size", 0)
        architectures = getattr(config, "architectures", ["LlamaForCausalLM"])

        # MLA (DeepSeek / GLM-4.x MoE): present only when kv_lora_rank is set. When MLA, head_dim
        # has no meaningful config value (hidden/heads is non-integral), so derive it from the
        # per-head qk dim (qk_nope + qk_rope) — the value the absorbed/materialized kernels use.
        kv_lora_rank = getattr(config, "kv_lora_rank", None)
        q_lora_rank = getattr(config, "q_lora_rank", None)
        qk_nope_head_dim = getattr(config, "qk_nope_head_dim", None)
        qk_rope_head_dim = getattr(config, "qk_rope_head_dim", None)
        v_head_dim = getattr(config, "v_head_dim", None)
        if kv_lora_rank is not None:
            head_dim = qk_nope_head_dim + qk_rope_head_dim
        # noaux_tc / fine-grained MoE knobs (defaults = plain top-k, no shared expert).
        n_group = getattr(config, "n_group", 1) or 1
        topk_group = getattr(config, "topk_group", 1) or 1
        # Laguna names it `moe_routed_scaling_factor` (2.5); GLM/DeepSeek use `routed_scaling_factor`.
        # Read the moe-prefixed key first so Laguna's 2.5 isn't silently defaulted to 1.0.
        routed_scaling_factor = (
            getattr(config, "moe_routed_scaling_factor", None)
            or getattr(config, "routed_scaling_factor", 1.0)
            or 1.0
        )
        first_k_dense_replace = getattr(config, "first_k_dense_replace", 0) or 0
        # Laguna has no `first_k_dense_replace`; it declares dense layer-0 via `mlp_layer_types`
        # ([ "dense", "sparse", ... ]) / `mlp_only_layers` ([0]). Derive the count of LEADING dense
        # layers so the decoder's `layer_id < first_k_dense_replace` dense/MoE dispatch is correct.
        if not first_k_dense_replace:
            _mlp_types = getattr(config, "mlp_layer_types", None)
            if _mlp_types is not None:
                fk = 0
                for _t in _mlp_types:
                    if _t == "dense":
                        fk += 1
                    else:
                        break
                first_k_dense_replace = fk
            else:
                _mol = getattr(config, "mlp_only_layers", None) or []
                if list(_mol) == list(range(len(_mol))):
                    first_k_dense_replace = len(_mol)
        n_shared_experts = getattr(config, "n_shared_experts", 0) or 0
        num_nextn_predict_layers = getattr(config, "num_nextn_predict_layers", 0) or 0
        mtp_num_hidden_layers = getattr(config, "mtp_num_hidden_layers", 0) or 0
        # Some checkpoints ship the MTP-head TENSORS but leave the count at 0 in config (e.g. a quant
        # tool that dropped the field). MINISGL_NUM_NEXTN / MINISGL_MTP_LAYERS
        # force the head on so spec-decode can use it. 0/unset -> trust the config.
        _mtp_forced = False
        if (_nn := os.environ.get("MINISGL_NUM_NEXTN")):
            num_nextn_predict_layers = int(_nn)
            _mtp_forced = True
        if (_ml := os.environ.get("MINISGL_MTP_LAYERS")):
            mtp_num_hidden_layers = int(_ml)
            _mtp_forced = True
        # The MTP / next-n head is a SELF-speculation head — only useful under --spec-algorithm mtp.
        # For any other serve (plain decode, dflash, eagle3, ngram, tidar) it wastes memory and, on a
        # quantized checkpoint that ships a bf16 MTP head, mismatches the quantized-backbone loader
        # (KeyError on mtp.self_attn.q_proj.weight_packed). So build it ONLY when MTP spec is active,
        # unless an explicit env override forced it on. Applied inside from_hf so the model builder
        # AND the streaming weight loader (both go through from_hf) agree on load_mtp.
        if spec_algorithm != "mtp" and not _mtp_forced:
            mtp_num_hidden_layers = 0
            num_nextn_predict_layers = 0
        # qwen4_exp's MTP head IS implemented (models/qwen4exp.py::Qwen4ExpMTPHead, bring-up plan
        # T8.1). The refusal that stood here was correct while it did not exist — the head is not
        # the Qwen3.5 one with a different prefix, and half-building it would have drafted badly
        # rather than raised. Its three stated blockers resolved as: the fused expert tensors are
        # the STACKED layout MoELayer already wants (the easy case, not the hard one); the head's
        # own hyper-connections are built by the same `_make_hc` the backbone uses; and the
        # hc_count-wide seed is exactly what `Qwen4ExpModel.forward(return_hidden=True)` already
        # returns, consumed PER BRANCH rather than folded (see the head's docstring).

        # A config field is a CLAIM; the tensors are the fact. `cyankiwi/Agents-A1-AWQ-INT4` declares
        # `mtp_num_hidden_layers: 1` in its own config.json and ships ZERO mtp.* tensors — the
        # quantizer dropped the head and left the field. Building from the claim produced a 22-buffer
        # MTP head and then died in the weight loader with a bare `KeyError:
        # 'mtp.pre_fc_norm_embedding.weight'` (layers/base.py `state_dict.pop`), which names a symptom
        # and not a cause. So when we are about to build a head, verify the checkpoint actually ships
        # one and say precisely what is wrong if it does not. `ckpt_tensor_names=None` (unit tests,
        # any caller without the weights on disk) skips the cross-check and trusts the config.
        # An EMPTY name set means we learned nothing (no safetensors on disk yet, a GGUF/other
        # format, an unreadable index) — not "this checkpoint has no MTP". Treat it as unknown and
        # trust the config, so a probe failure can never invent a startup error.
        if (
            (mtp_num_hidden_layers > 0 or num_nextn_predict_layers > 0)
            and ckpt_tensor_names
        ):
            from .weight import checkpoint_ships_mtp

            if not checkpoint_ships_mtp(ckpt_tensor_names, num_layers):
                _claim = (
                    f"MINISGL_MTP_LAYERS/MINISGL_NUM_NEXTN forced a head on"
                    if _mtp_forced
                    else f"its config declares mtp_num_hidden_layers="
                         f"{mtp_num_hidden_layers} / num_nextn_predict_layers="
                         f"{num_nextn_predict_layers}"
                )
                raise ValueError(
                    f"speculative decoding was requested (--spec-algorithm mtp) and {_claim}, but "
                    f"this checkpoint ships NO MTP tensors — no `mtp.*` keys and no appended "
                    f"`layers.>={num_layers}` head among its {len(ckpt_tensor_names)} tensors. The "
                    f"config claims a head the weights do not back (a quantizer that drops the MTP "
                    f"head but leaves the field does exactly this). Serve it without self-speculation "
                    f"(--spec-algorithm none, or ngram/eagle3/dflash, which need no MTP head), or use "
                    f"a checkpoint revision that ships the head."
                )

        # RMSNorm eps: Llama/Qwen use `rms_norm_eps`; ZAYA names it `norm_epsilon`.
        rms_norm_eps = getattr(config, "rms_norm_eps", None)
        if rms_norm_eps is None:
            rms_norm_eps = getattr(config, "norm_epsilon", 1e-5)

        # Llama/Qwen: rope_theta is a direct attr; Mistral: it's inside rope_scaling dict;
        # Qwen3.5: a single `rope_parameters` dict (rope_theta + partial_rotary_factor + mrope).
        # Normalize falsy values to None: some configs (e.g. ZAYA) ship `rope_scaling: false`,
        # which must be treated as "no scaling" — not a dict to .get() on.
        rope_scaling = getattr(config, "rope_scaling", None) or None
        rope_params = getattr(config, "rope_parameters", None) or None
        rope_theta = getattr(config, "rope_theta", None)
        for _rope_d in (rope_params, rope_scaling):
            if rope_theta is not None or _rope_d is None:
                continue
            rope_theta = _rope_d.get("rope_theta")
            if rope_theta is None:
                # ZAYA nests rope_theta PER layer_type: rope_parameters =
                # {'hybrid': {rope_theta 5e6, ...}, 'hybrid_sliding': {rope_theta 1e4, ...},
                # 'rope_type': 'default'}. Pick the theta of the layer type the model actually uses
                # (ZAYA1-8B is all 'hybrid'); fall back to the first nested rope_theta. minisgl builds
                # ONE rope, so a genuinely mixed-theta model would need per-layer rope — not the case
                # for shipped ZAYA, but assert-worthy if a future config carries >1 distinct theta.
                lts = getattr(config, "layer_types", None) or []
                cand = _rope_d.get(lts[0]) if lts else None
                if not (isinstance(cand, dict) and "rope_theta" in cand):
                    cand = next(
                        (v for v in _rope_d.values() if isinstance(v, dict) and "rope_theta" in v),
                        None,
                    )
                rope_theta = cand.get("rope_theta") if isinstance(cand, dict) else None
        # Partial rotary (Qwen3.5: rotary_dim = head_dim * partial_rotary_factor, e.g. 0.25 -> 64).
        # Plain models leave it None -> full rotary (rotary_dim == head_dim).
        partial = getattr(config, "partial_rotary_factor", None)
        if partial is None and rope_params is not None:
            partial = rope_params.get("partial_rotary_factor")
        rotary_dim = int(head_dim * partial) if partial is not None else head_dim

        # Rope scaling: only a real scheme (llama3/yarn/...) becomes RotaryConfig.scaling.
        # A "default" rope_type carries no scaling -> None (so _get_rope takes the plain-rotary
        # path, identical to the default branch). Qwen3.5's rope_parameters is rope_type
        # "default" + an mrope_section LIST; mrope is multimodal-only (text serving uses plain
        # partial rotary, already set via rotary_dim above) and the list is unhashable in the
        # cached rope builder, so it must NOT be folded into scaling.
        rope_dict = rope_scaling if rope_scaling is not None else rope_params
        scaling = (
            rope_dict
            if rope_dict is not None and rope_dict.get("rope_type") not in (None, "default")
            else None
        )

        # GDN / linear-attention hybrid (Qwen3-Next / Qwen3.5). Only populated when the config
        # actually carries linear-attention dims, so dense models keep layer_types=None even if
        # they define a (sliding/full) layer_types list of their own.
        linear_num_key_heads = getattr(config, "linear_num_key_heads", None)
        layer_types = None
        if linear_num_key_heads is not None:
            layer_types = getattr(config, "layer_types", None)
            if layer_types is None:
                # Implicit pattern: a full-attention layer every `full_attention_interval`.
                interval = getattr(config, "full_attention_interval", 4)
                layer_types = [
                    "full_attention" if (i + 1) % interval == 0 else "linear_attention"
                    for i in range(config.num_hidden_layers)
                ]
            layer_types = tuple(layer_types)

        # Nemotron-H (Mamba-2 + MoE + global attention). Its schedule is `layers_block_type`, a per
        # layer choice of the ONE mixer that layer runs. Deliberately kept out of `layer_types` — see
        # the `block_types` field comment for why mapping "mamba" onto "linear_attention" would build
        # a working, wrong model.
        _is_nemotron_h = getattr(config, "model_type", None) == "nemotron_h"
        block_types = None
        if _is_nemotron_h:
            _bt = getattr(config, "layers_block_type", None)
            if _bt is None:
                raise ValueError(
                    "nemotron_h config has no `layers_block_type`; the mixer schedule is not "
                    "derivable from anything else in the config and must not be guessed")
            _known = {"mamba", "moe", "attention"}
            _bad = sorted(set(_bt) - _known)
            if _bad:
                raise ValueError(f"nemotron_h `layers_block_type` has unknown kinds {_bad}")
            if len(_bt) != config.num_hidden_layers:
                raise ValueError(
                    f"nemotron_h `layers_block_type` has {len(_bt)} entries but "
                    f"num_hidden_layers is {config.num_hidden_layers}")
            block_types = tuple(_bt)

        # Sliding-window-attention hybrid (Laguna): an UN-gated `layer_types` of "full_attention" /
        # "sliding_attention" plus a top-level `sliding_window`. Kept separate from the GDN branch
        # above (which is gated on linear_num_key_heads) so the two schedules never collide; a SWA
        # config has no linear dims, so this is the branch that populates layer_types for it.
        sliding_window = getattr(config, "sliding_window", None)
        attn_head_counts = None
        sliding_rotary_config = None
        _full_rope_override: RotaryConfig | None = None
        if layer_types is None and sliding_window is not None:
            _lts = getattr(config, "layer_types", None)
            if _lts is not None and any(t == "sliding_attention" for t in _lts):
                layer_types = tuple(_lts)
        # Gemma4's two layer types differ in head_dim (256 sliding / 512 full) and kv-head count
        # (8 / 2), not just in QO head count. `head_dim`/`num_kv_heads` are rebound to the FULL
        # geometry because they size the main paged pool (= the full-attention layers for a SWA
        # hybrid); the sliding geometry is carried separately for the ring pool.
        swa_head_dim = None
        swa_num_kv_heads = None
        _is_gemma4 = model_type in (
            "gemma4", "gemma4_text", "diffusion_gemma", "diffusion_gemma_text"
        )
        _is_muse = model_type in ("muse_glimmer", "muse_glimmer_text")
        # Muse-Glimmer scales Q by `qk_scale_factor` (3.87) AFTER a weightless QK-norm and BEFORE
        # attention, ON TOP OF the ordinary head_dim**-0.5 — it replaces neither. Because both Q and
        # K are RMS-normalised to unit RMS, that factor is what actually sets the softmax
        # temperature. Folding it into the scale is exact (scaling Q by c then dotting is identical
        # to scaling the logits by c) and saves a full-tensor multiply per layer.
        _muse_qk_scale = float(getattr(config, "qk_scale_factor", 3.87)) if _is_muse else None
        # `attention_k_eq_v` decides whether the FULL-attention layers ship a v_proj at all, so
        # getting it wrong is not a tuning error — it changes the parameter set. Gemma4 declares it;
        # DiffusionGemma DELETES it (its @strict config subclass rebinds the field to an
        # AttributeError sentinel) while keeping the behaviour hard-coded in its attention class.
        # A plain getattr therefore reads False for DiffusionGemma and the model builds 30 v_projs
        # for a checkpoint that ships 25 — a KeyError at load. So when the config is silent, decide
        # from the TENSORS, exactly as the MTP head is decided: a checkpoint whose v_proj count is
        # short of num_layers is a k_eq_v checkpoint.
        attention_k_eq_v = getattr(config, "attention_k_eq_v", None)
        if _is_gemma4 and not isinstance(attention_k_eq_v, bool):
            if ckpt_tensor_names:
                _v_layers = {
                    m.group(1)
                    for n in ckpt_tensor_names
                    if ".self_attn.v_proj." in n
                    and (m := re.search(r"(?:^|\.)layers\.(\d+)\.", n)) is not None
                }
                attention_k_eq_v = 0 < len(_v_layers) < num_layers
            else:
                # No tensor list (unit tests, config-only callers): every shipping Gemma4-family
                # checkpoint to date is k_eq_v, and guessing False would silently mis-shape the
                # model, whereas guessing True is caught loudly by an unexpected v_proj key.
                attention_k_eq_v = True
        attention_k_eq_v = bool(attention_k_eq_v)
        if _is_gemma4:
            _global_head_dim = getattr(config, "global_head_dim", None) or _full_layer_override(
                config, "head_dim", layer_types
            )
            if _global_head_dim:
                swa_head_dim, head_dim = head_dim, _global_head_dim
                swa_num_kv_heads, num_kv_heads = (
                    num_kv_heads,
                    getattr(config, "num_global_key_value_heads", None)
                    or _full_layer_override(config, "num_key_value_heads", layer_types)
                    or num_kv_heads,
                )
        if layer_types is not None and any(t == "sliding_attention" for t in layer_types):
            # Per-layer QO head counts (Laguna: 48 full / 64 sliding).
            _heads = getattr(config, "num_attention_heads_per_layer", None)
            if _heads is not None:
                attn_head_counts = tuple(int(h) for h in _heads)
            # Two RoPE schemes: rope_parameters nests one sub-dict per attention type. Build the
            # FULL rope into rotary_config below (override the generic single-rope parse) and the
            # SLIDING rope into sliding_rotary_config. Each is built over ITS OWN head_dim — for
            # Gemma4 those differ (512 full / 256 sliding), and feeding the full head_dim to the
            # sliding rope would mis-size its cos/sin cache.
            if isinstance(rope_params, dict) and "full_attention" in rope_params:
                _maxpos = config.max_position_embeddings
                _full_rope_override = _rotary_from_subdict(
                    rope_params["full_attention"], head_dim, _maxpos
                )
                if "sliding_attention" in rope_params:
                    sliding_rotary_config = _rotary_from_subdict(
                        rope_params["sliding_attention"],
                        swa_head_dim if swa_head_dim is not None else head_dim,
                        _maxpos,
                    )

        # ---- Qwen4-Exp: hyper-connection / PLE / QSA-indexer dims ----
        # `ple_layer_ids` in config.json is 1-BASED (the shipping value [2] names decoder index 1,
        # which is exactly where the checkpoint's `layers.1.ple.*` tensors live). Convert here, once,
        # and range-check: attaching the PLE block to index 2 instead of 1 loads without a single
        # error and only degrades quality (bring-up plan T4.3).
        ple_layer_ids: Tuple[int, ...] = ()
        if _is_qwen4_exp:
            _raw_ple = list(getattr(config, "ple_layer_ids", None) or [])
            ple_layer_ids = tuple(sorted({int(i) - 1 for i in _raw_ple}))
            for _p in ple_layer_ids:
                if not 0 <= _p < num_layers:
                    raise ValueError(
                        f"qwen4_exp ple_layer_ids={_raw_ple} (1-based) maps to decoder index {_p}, "
                        f"outside [0, {num_layers}). Either the field is already 0-based in this "
                        f"checkpoint or num_hidden_layers disagrees with it."
                    )
            if not getattr(config, "hc_count", None):
                raise ValueError(
                    "qwen4_exp config is missing `hc_count`: the whole decoder carries an "
                    "hc_count-times-wide residual stream, so there is no safe default."
                )

        # The two hash inputs config.json leaves implicit. `seed` is absent from the shipping file,
        # so the architecture default (1234) is what every row id depends on — carried explicitly so
        # ple/hashing.py can assert it reproduces the checkpoint's own `layer_multipliers` instead of
        # assuming it. `eos_token_id` may be an int or a list; the reference takes `[0]` of a list
        # (`Qwen4ExpTextNGramEmbedding.__init__`), and it is the fill for positions whose n-gram
        # context would cross a segment boundary.
        ngram_seed = None
        ngram_eos_token_id = None
        if _is_qwen4_exp:
            from minisgl.ple.hashing import DEFAULT_NGRAM_SEED

            ngram_seed = int(getattr(config, "seed", None) or DEFAULT_NGRAM_SEED)
            _eos = getattr(config, "eos_token_id", None)
            if isinstance(_eos, (list, tuple)):
                _eos = _eos[0] if _eos else None
            if _eos is None:
                raise ValueError(
                    "qwen4_exp config has no `eos_token_id`, which the n-gram hash needs: it is the "
                    "fill for positions whose n-gram context would reach across a segment boundary, "
                    "and the seed of every sequence's initial 2-token context. Defaulting it would "
                    "hash the first tokens of every request against the wrong n-gram, silently."
                )
            ngram_eos_token_id = int(_eos)

        return cls(
            num_layers=num_layers,
            num_qo_heads=config.num_attention_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            intermediate_size=intermediate_size,
            # The Gemma lineage (and Muse-Glimmer, which inherits its config shape) spells this
            # `hidden_activation`; everyone else spells it `hidden_act`. Read both rather than let
            # the "silu" default silently paper over a checkpoint that asked for gelu.
            hidden_act=getattr(config, "hidden_act", None)
            or getattr(config, "hidden_activation", None)
            or "silu",
            rms_norm_eps=rms_norm_eps,
            tie_word_embeddings=tie_word_embeddings,
            # SWA models (Laguna) override this with the FULL-attention rope (yarn); the sliding
            # layers use `sliding_rotary_config`. Every other model builds the single generic rope.
            rotary_config=_full_rope_override
            if _full_rope_override is not None
            else RotaryConfig(
                head_dim=head_dim,
                rotary_dim=rotary_dim,
                max_position=config.max_position_embeddings,
                base=rope_theta,
                scaling=scaling,
                interleave=bool(getattr(config, "rope_interleave", False))
                and not os.environ.get("MINISGL_DISABLE_ROPE_INTERLEAVE"),
            ),
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            moe_intermediate_size=moe_intermediate_size,
            norm_topk_prob=norm_topk_prob,
            shared_expert_intermediate_size=shared_expert_intermediate_size,
            model_type=model_type,
            architectures=architectures,
            quant=quant,
            # Block diffusion: `canvas_length` sits on the TOP-LEVEL config, not the text config, so
            # it is read off `top` (which is `config` itself for a flat checkpoint). Absent -> None
            # -> is_block_diffusion False, and every canvas branch downstream stays dead.
            canvas_length=getattr(top, "canvas_length", None) or None,
            linear_num_key_heads=linear_num_key_heads,
            gdn_output_gate=_norm_output_gate(getattr(config, "output_gate_type", None)),
            final_logit_softcapping=getattr(config, "final_logit_softcapping", None) or None,
            linear_num_value_heads=getattr(config, "linear_num_value_heads", None),
            linear_key_head_dim=getattr(config, "linear_key_head_dim", None),
            linear_value_head_dim=getattr(config, "linear_value_head_dim", None),
            linear_conv_kernel_dim=getattr(config, "linear_conv_kernel_dim", None),
            layer_types=layer_types,
            block_types=block_types,
            mamba_num_heads=getattr(config, "mamba_num_heads", None) if _is_nemotron_h else None,
            mamba_head_dim=getattr(config, "mamba_head_dim", None) if _is_nemotron_h else None,
            mamba_ssm_state=getattr(config, "ssm_state_size", None) if _is_nemotron_h else None,
            mamba_n_groups=getattr(config, "n_groups", None) if _is_nemotron_h else None,
            mamba_conv_kernel=getattr(config, "conv_kernel", None) if _is_nemotron_h else None,
            mamba_chunk_size=getattr(config, "chunk_size", None) if _is_nemotron_h else None,
            # The shipped spelling is time_step_min/time_step_max, not the `time_step_limit` tuple
            # other Mamba-2 configs use. Defaulted only because a config that omits both is using
            # the upstream defaults, which are these.
            mamba_dt_min=float(getattr(config, "time_step_min", 0.001)) if _is_nemotron_h else None,
            mamba_dt_max=float(getattr(config, "time_step_max", 0.1)) if _is_nemotron_h else None,
            mamba_conv_bias=bool(getattr(config, "use_conv_bias", True)),
            mamba_proj_bias=bool(getattr(config, "mamba_proj_bias", False)),
            moe_act=getattr(config, "mlp_hidden_act", None) if _is_nemotron_h else None,
            moe_shared_intermediate=(
                getattr(config, "moe_shared_expert_intermediate_size", None)
                if _is_nemotron_h else None),
            sliding_window=(
                sliding_window
                if layer_types is not None
                and any(t == "sliding_attention" for t in layer_types)
                else None
            ),
            attn_head_counts=attn_head_counts,
            sliding_rotary_config=sliding_rotary_config,
            swa_head_dim=swa_head_dim,
            swa_num_kv_heads=swa_num_kv_heads,
            # Gemma4 hard-codes scaling=1.0 in its attention (modeling_gemma4.py:1153); the
            # temperature lives in the learned k_norm instead. Every other family leaves this None
            # and keeps the standard head_dim**-0.5.
            attn_softmax_scale=(
                1.0
                if _is_gemma4
                else (_muse_qk_scale * head_dim**-0.5 if _muse_qk_scale is not None else None)
            ),
            attention_k_eq_v=attention_k_eq_v,
            embed_scale=(config.hidden_size**0.5) if _is_gemma4 else None,
            output_multiplier=getattr(config, "output_multiplier", None) if _is_muse else None,
            post_norm_eps=getattr(config, "post_norm_eps", None) if _is_muse else None,
            layer_rope_theta=(
                tuple(_lrt)
                if _is_muse and isinstance((_lrt := getattr(config, "layer_rope_theta", None)), list)
                else None
            ),
            kv_lora_rank=kv_lora_rank,
            q_lora_rank=q_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            n_group=n_group,
            topk_group=topk_group,
            routed_scaling_factor=routed_scaling_factor,
            first_k_dense_replace=first_k_dense_replace,
            n_shared_experts=n_shared_experts,
            num_nextn_predict_layers=num_nextn_predict_layers,
            mtp_num_hidden_layers=mtp_num_hidden_layers,
            is_cca=is_cca,
            cca_time0=getattr(config, "cca_time0", 2) if is_cca else None,
            cca_time1=getattr(config, "cca_time1", 2) if is_cca else None,
            # ZAYA's HF config names the k/v head count `num_key_value_heads` (the Megatron export
            # used `num_query_groups`); read the HF name first, fall back to the Megatron one.
            cca_num_k_heads=(
                (getattr(config, "num_key_value_heads", None) or getattr(config, "num_query_groups", None))
                if is_cca
                else None
            ),
            cca_num_q_heads=(config.num_attention_heads if is_cca else None),
            cca_head_dim=(head_dim if is_cca else None),
            cca_clamp_temp=bool(getattr(config, "clamp_temp", False)) if is_cca else False,
            # ZAYA router MLP width is `router_hidden_size` (the HF name); keep the old key as fallback.
            zaya_mlp_expansion=(
                getattr(config, "router_hidden_size", None) or getattr(config, "zaya_mlp_expansion", 256)
                if is_cca
                else None
            ),
            # EDA + MOD are ARCHITECTURE constants in transformers ZAYA (ZayaRouter always builds
            # num_experts+1 classes for the MOD skip; use_eda = layer_idx!=0) — there is NO use_eda/
            # use_mod config flag, so a CCA model defaults them ON (else routing loses the skip expert
            # and the EDA state thread -> degenerate). Still overridable by an explicit config key.
            zaya_use_eda=bool(getattr(config, "zaya_use_eda", True)) if is_cca else False,
            zaya_use_mod=bool(getattr(config, "zaya_use_mod", True)) if is_cca else False,
            scale_residual_merge=bool(getattr(config, "scale_residual_merge", is_cca)),
            residual_in_fp32=bool(getattr(config, "residual_in_fp32", False)),
            unparsed_quant_method=unparsed_quant_method,
            # Qwen4-Exp. All gated on _is_qwen4_exp so no other family can pick these up from a
            # coincidentally-named config key.
            hc_count=getattr(config, "hc_count", None) if _is_qwen4_exp else None,
            hc_lowrank=getattr(config, "hc_lowrank", None) if _is_qwen4_exp else None,
            ple_layer_ids=ple_layer_ids,
            ple_embed_dim=(
                (getattr(config, "ple_embed_dim", None) or config.hidden_size)
                if _is_qwen4_exp
                else None
            ),
            ple_conv_kernel_size=(
                getattr(config, "ple_conv_kernel_size", None) if _is_qwen4_exp else None
            ),
            ngram_size=getattr(config, "ngram_size", None) if _is_qwen4_exp else None,
            heads_per_ngram=getattr(config, "heads_per_ngram", None) if _is_qwen4_exp else None,
            ngram_vocab_size_base=(
                getattr(config, "ngram_vocab_size_base", None) if _is_qwen4_exp else None
            ),
            make_ngram_vocab_size_divisible_by=(
                getattr(config, "make_ngram_vocab_size_divisible_by", None)
                if _is_qwen4_exp
                else None
            ),
            split_ngram_parts=(
                getattr(config, "split_ngram_parts", None) if _is_qwen4_exp else None
            ),
            ngram_seed=ngram_seed,
            ngram_eos_token_id=ngram_eos_token_id,
            indexer_budget=getattr(config, "indexer_budget", None) if _is_qwen4_exp else None,
            indexer_compress_ratio=(
                getattr(config, "indexer_compress_ratio", None) if _is_qwen4_exp else None
            ),
            indexer_head_dim=(
                getattr(config, "indexer_head_dim", None) if _is_qwen4_exp else None
            ),
            indexer_kv_heads=(
                getattr(config, "indexer_kv_heads", None) if _is_qwen4_exp else None
            ),
            indexer_n_heads=(
                getattr(config, "indexer_n_heads", None) if _is_qwen4_exp else None
            ),
        )