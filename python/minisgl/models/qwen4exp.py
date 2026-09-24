"""Qwen3.8-Flash-Next (`qwen4_exp`) — text-only decoder SKELETON. Bring-up tranche 1a.

WHAT THIS FILE IS
-----------------
The parameter/structure definition of the 48-layer text decoder of
`Qwen4ExpForConditionalGeneration`, verified against the real checkpoint headers of
`RadixArk/Qwen3.8-Flash-Next-NVFP4` (not against docs):

  * 48 layers, schedule `layer_types[i]`: `idx % 4 == 3` (3, 7, ..., 47) is FULL attention with
    `self_attn.*`; the other 36 are linear attention with `linear_attn.*`. A layer never has both.
  * EVERY layer has the MoE `mlp.*` block (512 routed experts, top-10, inter 640, plus an always-on
    bf16 shared expert + sigmoid gate) AND two hyper-connection blocks
    (`attn_hyper_connection`, `mlp_hyper_connection`).
  * ONLY decoder index 1 carries the PLE n-gram block (`ple.*`). config.json's `ple_layer_ids: [2]`
    is 1-BASED; `ModelConfig` converts it.
  * `lm_head` is UNTIED, and there is **no** `model.language_model.norm.weight` — the top-level
    `hyper_connection_mixer` (3 tensors, no `block_inject_weight`) plays the final-norm role and is
    what feeds `lm_head`.
  * The residual stream is `hc_count * hidden_size` = 4 * 2560 = **10240** wide for the whole
    decoder. Each block reads a mixed 2560-wide view and writes back into the wide stream.

WHAT IS IMPLEMENTED
-------------------
The whole `Qwen4ExpPLE` n-gram block (tranche 1b), a transcription of
`transformers/models/qwen4_exp/modeling_qwen4_exp.py` (`main`, fetched 2026-09-03) — the reference
implementation of this architecture — checked against it numerically in `tests/qwen4exp_ple_test.py`,
not merely read. The host half of the PLE block (hash -> NVMe row gather -> one H2D into a static
buffer) lives in `minisgl/ple/`; the table itself was already done in `minisgl/weights/row_table.py`
and is NOT re-implemented.

The hyper-connections (tranche T0.3 / GATE-2) are `minisgl.layers.HyperConnection` — they are
architecture-level plumbing, not a qwen4_exp detail, so they live under `layers/` next to the norms
and linears they are made of, and `_make_hc` below is the only place this config's field names are
read. Their `hc_norm` is `minisgl.layers.GroupedRMSNorm`, the same grouped (1+w) norm the PLE block
uses. `tests/qwen4exp_hc_parity_test.py` pins `mix` and `combine` SEPARATELY against the sglang
reference source itself (fp32 bit-exact, bf16 within 2 ULP on the real layer-10 tensors).

THE FORWARD RUNS. WHAT IS BOUNDED, AND HOW IT REFUSES
-----------------------------------------------------
`Qwen4ExpForConditionalGeneration.forward()` computes logits. Three things are deliberately NOT
implemented, and each raises rather than approximating:

  * **QSA indexer** (T5) — the sparse-attention selection on the 12 full-attention layers.
    `QSAIndexer` is declared (its checkpoint tensors need somewhere to land and the loader key set
    must stay exact) but computes nothing. What runs instead is DENSE causal attention, which is not
    an approximation: below `indexer_budget` (2048) the selection picks `min(topk, visible)` keys —
    every visible one — so dense IS the sparse result, bit for bit. `Qwen4ExpAttn.forward` asserts
    that inequality against the live batch on every call and raises the moment any request's context
    crosses 2048. It does not clamp, truncate or fall back.
  * **Aux-hidden capture / draft heads** — refused: the per-layer residual here is the 4x-wide
    hyper-connection stream, not the hidden-size feature a drafter's fc was trained on.

`UNIMPLEMENTED` below is the machine-readable version of that list; it is logged once per build.

TENSOR PARALLELISM (added 2026-09-04; was a blanket refusal until then)
----------------------------------------------------------------------
TP>1 is legal wherever the counts divide — `_assert_tp_divides` checks them at build and names the
one that failed. `num_key_value_heads = 2`, so TP=2 is the last legal degree on the GQA side, which
is also all this box has. The shard rules are `models/weight._shard_qwen4_exp`, applied at READ on
the checkpoint name so the GDN in_proj concat, the gate/up merge and the per-expert stack compose
rank-local parts; most of them delegate to `_shard_qwen3_5` because these are the Qwen3.5 shapes.

WHAT DOES NOT SHARD, and it is not an oversight: the hyper-connections. `HyperConnection` mixes
across the hc_count residual streams of the WHOLE hidden state, so a column split would leave each
rank with a partial mix and no all-reduce to complete it — right shapes, plausible text, wrong
model. They are `LinearReplicated`, the quant `ignore` list keeps them bf16, and
`qwen4exp_offload_serve_test.py` digests their BYTES on every rank and compares over the gloo group
rather than trusting either fact. The cost is real and worth stating: at 48 layers they are
1.193 GiB, identical at TP=1 and TP=2, i.e. 22.5% of the 5.312 GiB per-rank TP=2 body (measured).
The QSA indexer and the PLE block are replicated for the same construction reason.

MEASURED at TP=2, 2026-09-04 (cards 0+1, meta build for the byte counts): routed experts
35.156 GiB/rank (0.7324/layer, exactly half of TP=1's 70.312), body 5.312 GiB/rank (from 9.216).

REUSED VERBATIM (do not fork)
-----------------------------
`QwenGatedDeltaNet` + the `GDNLinearAttn` bridge (`gdn/layer.py`, `models/qwen3_5.py`), the gated
partial-rotary attention `Qwen3_5Attn`, and the sparse block `Qwen3_5MoeSparseBlock` — the Qwen3.5
shapes are identical here. The differences are the hyper-connections, the PLE block, the QSA
indexer, and the absence of the per-layer input/post norms and the final norm.
"""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING, List, Tuple

import torch
import torch.nn.functional as F
from minisgl.core import get_global_ctx
from minisgl.distributed import get_tp_info
from minisgl.gdn.layer import QwenGatedDeltaNet
from minisgl.layers import (
    BaseOP,
    GroupedRMSNorm,
    HyperConnection,
    LinearReplicated,
    OPList,
    ParallelLMHead,
    RMSNorm,
    VocabParallelEmbedding,
)
from minisgl.ple import PLEBatch
from minisgl.quant import create_linear_method
from minisgl.utils import init_logger, nvtx_annotate

from .base import BaseLLMModel
from .qwen3_5 import GDNLinearAttn, Qwen3_5Attn, Qwen3_5MTPAttn
from .qwen3_5_moe import Qwen3_5MoeSparseBlock

if TYPE_CHECKING:
    from .config import ModelConfig

logger = init_logger(__name__)

# Every piece of this architecture that is NOT implemented yet, with the bring-up-plan step that
# owns it and the EXACT condition under which the gap becomes observable. Logged once per build by
# `Qwen4ExpForConditionalGeneration.__init__`. Deleting an entry here is the last step of landing
# that piece — keep it honest. Every one of these also raises at the point of use; the list is the
# summary, never the enforcement.
UNIMPLEMENTED: Tuple[Tuple[str, str], ...] = (
    (
        "vision tower (model.visual.*)",
        "text-only serve, as for every other multimodal checkpoint here. The loader counts the 333 "
        "skipped vision tensors in its ignore ledger; an image token in a prompt is a tokenizer/"
        "front-end concern and never reaches this model",
    ),
    (
        "cudagraph capture of the PREFILL / spec-VERIFY forwards",
        "DECODE capture is implemented and exercised (2026-09-04): boot with the `hip` attention "
        "backend and --cuda-graph-max-bs > 0, and `PLEGraphCapture` stages the n-gram batch the "
        "capture-time warmup forward needs. Prefill stays eager everywhere in this engine, and "
        "spec-verify capture is moot while --spec-algorithm mtp is refused for this architecture. "
        "The `rdna4` attention backend still raises on capture (its Phase-4 gap, not this model's); "
        "the capture-capable subclass is `hip`, which is what every production serve here uses",
    ),
)


def _make_hc(config: "ModelConfig", *, use_combine: bool) -> HyperConnection:
    """`minisgl.layers.HyperConnection` from this config. The layer takes plain scalars (it is
    architecture-agnostic and lives under `layers/`); this is the one place the qwen4_exp config
    field names are read."""
    return HyperConnection(
        hidden_size=config.hidden_size,
        hc_count=config.hc_count,
        hc_lowrank=config.hc_lowrank,
        eps=config.rms_norm_eps,
        use_combine=use_combine,
    )


class QSAIndexer(BaseOP):
    """Query-sparse-attention indexer for a full-attention layer. Parameters only (T5).

    Checkpoint tensors, verified from the shard header:
        index_qk_proj.weight  [n_heads*head_dim + kv_heads*head_dim, hidden]  = [640, 2560]
        q_layernorm.weight    [head_dim] = [128]
        k_layernorm.weight    [head_dim] = [128]

    The two layernorms are Gemma-style `(1 + w)` (`sglang/srt/layers/attention/qsa/qsa_indexer.py`
    imports `GemmaRMSNorm`), which is why they are built with `plus_one=True`.

    Not called by anything. At `seq_len <= indexer_budget` (2048) the upstream selection kernel
    clamps `row_topk = min(topk, visible)` and therefore selects every visible token, i.e. it is
    bit-equivalent to dense causal attention — which is why a short-context forward can legitimately
    run the DENSE path and still be exact. `Qwen4ExpAttn.forward` checks that inequality against the
    live batch on EVERY call (`assert_dense_is_exact`) and raises the moment a request crosses the
    budget, so nothing can quietly run a wrong attention believing it is sparse.
    """

    def __init__(self, config: "ModelConfig", index_layer_id: int = 0) -> None:
        d = config.indexer_head_dim
        assert d and config.indexer_n_heads and config.indexer_kv_heads, (
            "qwen4_exp full-attention layers need indexer_head_dim / _n_heads / _kv_heads"
        )
        qk_out = (config.indexer_n_heads + config.indexer_kv_heads) * d
        self.index_qk_proj = LinearReplicated(config.hidden_size, qk_out, has_bias=False)
        self.q_layernorm = RMSNorm(d, eps=config.rms_norm_eps, plus_one=True)
        self.k_layernorm = RMSNorm(d, eps=config.rms_norm_eps, plus_one=True)
        self._budget = int(config.indexer_budget or 0)
        assert self._budget > 0, "qwen4_exp needs a positive indexer_budget"
        # Compact 0..11 index into the index-key cache's layer dimension. The global layer_id is
        # 3,7,...,47, so it would index a 48-deep cache of which 36 rows are never written — the
        # same compaction `attn_kv_id` already does for the paged KV pool.
        self._index_layer_id = int(index_layer_id)
        self._n_heads = int(config.indexer_n_heads)
        self._head_dim = int(d)

    @property
    def budget(self) -> int:
        return self._budget

    @property
    def index_layer_id(self) -> int:
        return self._index_layer_id

    def forward(self, hidden_states: torch.Tensor, qsa) -> "object":
        """Run the four selection stages for this layer; return its `QSASelection`.

        Stage 1 (here): project, then build the QUERY — `RoPE(pos, GemmaRMSNorm(q_raw))`, per
        128-dim index head. The KEY half `k_tok` is emitted RAW: it is neither normed nor roped per
        token, because normalisation and rotation happen once per GROUP, after the fp32 mean. Doing
        either per token instead is a different (and quietly worse) selector that no output text
        distinguishes.

        Stages 2-4 (`QSARuntime.select`): ring-store the raw key, compress the groups that complete
        at this forward, score, top-k, expand, and resolve to physical KV slots.
        """
        qk = self.index_qk_proj.forward(hidden_states)
        d, hi = self._head_dim, self._n_heads
        q_raw = qk[:, : hi * d]
        token_k = qk[:, hi * d :].reshape(-1, d)
        q = self.q_layernorm.forward(q_raw.reshape(-1, d)).reshape(-1, hi * d)
        q = qsa.rotary.forward_one(qsa.plan.rope_pos, q).reshape(-1, hi, d)
        return qsa.select(self._index_layer_id, q, token_k, self.k_layernorm)

    def assert_dense_is_exact(self, max_ctx_len: int) -> None:
        """Raise unless dense causal attention is BIT-EQUIVALENT to this layer's sparse selection.

        The equivalence is not an approximation argument: at `seq_len <= budget` the selection picks
        `min(topk, visible) == visible` keys, i.e. all of them, so the sparse and dense attentions
        are the same computation. Above the budget they diverge and there is no defensible fallback
        — running dense anyway would be a DIFFERENT model (denser, so not obviously worse output, and
        therefore not detectable from the text), and truncating context would be worse. So: refuse.
        """
        if max_ctx_len > self._budget:
            raise NotImplementedError(
                f"qwen4_exp: a request reached context length {max_ctx_len}, past the QSA indexer "
                f"budget of {self._budget} WITH THE QSA RUNTIME DISABLED. Below the budget the "
                f"indexer selects every visible token, so this engine's dense causal attention is "
                f"bit-equivalent and is what ran; beyond it the checkpoint's attention is genuinely "
                f"sparse. The sparse path EXISTS (minisgl/attention/qsa) — this build simply is not "
                f"running it. Re-enable it (do not set MINISGL_QSA=0), make the `qsa_index` kernel "
                f"package importable, and serve with a KV page size that is a multiple of "
                f"indexer_compress_ratio (--page-size 16). Do NOT raise this bound instead: running "
                f"dense above the budget attends MORE, so the text stays fluent and the substitution "
                f"is undetectable from the output."
            )


class Qwen4ExpAttn(Qwen3_5Attn):
    """`Qwen3_5Attn` (gated q_proj, partial rotary, per-head q/k norm) plus the QSA indexer.

    Every shape was confirmed against the checkpoint: q_proj [12288, 2560] = 2 * 24 heads * 256
    (q interleaved with its per-head sigmoid output gate), k/v_proj [512, 2560] = 2 kv heads * 256,
    o_proj [2560, 6144], q_norm/k_norm [256]. So the Qwen3.5 class is reused unchanged and only the
    `indexer` submodule is added.

    TWO FORWARDS, AND WHICH ONE RUNS. When `ctx.qsa` is set (a `QSARuntime` — built by
    `Qwen4ExpForConditionalGeneration._ensure_qsa` on the first forward of a QSA-capable build) the
    indexer selects and the attention is genuinely sparse, at any context length. When it is NOT set
    — no `qsa_index` kernels, a KV page size the DSV4 addressing cannot use, or `MINISGL_QSA=0` —
    the layer runs DENSE and `assert_dense_is_exact` re-arms the old refusal, because dense is
    bit-equivalent to the selection only below `indexer_budget`. The refusal was never about
    approximation quality: attending MORE keeps the text fluent, so a silent fallback above the
    budget is undetectable from the output. It is therefore kept as the no-QSA path's guard rather
    than deleted with the feature that replaced it.
    """

    def __init__(self, config: "ModelConfig", layer_id: int, *, attn_kv_id: int) -> None:
        super().__init__(config, layer_id, attn_kv_id=attn_kv_id)
        self.indexer = QSAIndexer(config, index_layer_id=attn_kv_id or 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        qsa = getattr(ctx, "qsa", None)
        if qsa is not None:
            return super().forward(x, self.indexer.forward(x, qsa))
        # No QSA runtime: dense, and only legal below the budget. `device_len` is prompt + committed
        # tokens, i.e. the KV length this forward attends over — the quantity the budget bounds.
        # Checked per FORWARD rather than per request admission because a request crosses the budget
        # mid-generation: admitted at 2000 tokens it is fine, at 2049 it is not, and only the forward
        # sees that moment. `default=0` covers the empty-batch cases the engine legitimately issues
        # (an idle EP replica's dummy step), which attend over nothing.
        batch = ctx.batch
        self.indexer.assert_dense_is_exact(max((r.device_len for r in batch.reqs), default=0))
        return super().forward(x)


class Qwen4ExpNGramEmbedding(BaseOP):
    """The learned scalars of the n-gram embedding. The 51.2 GB TABLE itself is NOT a model
    parameter: it stays NVMe-resident and is served by `minisgl/weights/row_table.py`
    (`open_qwen4exp_ngram_table`), which mmaps `model-plefp8-*.safetensors` and gathers rows. The
    weight loader must therefore skip `ple_embedding.ngram_embedding.*`, and it does — see
    `qwen4_exp_remap`'s `ple-ngram-table` skip reason.

    `ngram_heads_offsets` / `ngram_heads_vocab_sizes` are likewise skipped: `row_table.NgramHeads`
    reads them straight out of the checkpoint files, and duplicating them here would create two
    sources of truth for an arithmetic that fails silently (an off-by-one reads another head's
    embeddings).

    `layer_multipliers` (int64 [ngram_size]) IS kept: it is the multiplier set the n-gram key is
    mixed with, and `minisgl/ple/hashing.py` is its only reader. That module also DERIVES the same
    three values from (vocab_size, ngram_size, ple_layer_index, seed=1234) and refuses to run if the
    two disagree — the derivation and the checkpoint tensor check each other, because a wrong
    multiplier reads a real embedding from the wrong row and errors nowhere.
    """

    def __init__(self, config: "ModelConfig") -> None:
        assert config.ngram_size, "qwen4_exp PLE needs ngram_size"
        self.layer_multipliers = torch.empty(config.ngram_size, dtype=torch.int64)


class Qwen4ExpPLE(BaseOP):
    """The per-layer n-gram (PLE) block. Present on ONE decoder layer only. IMPLEMENTED (tranche 1b).

    Shapes verified from the checkpoint shard headers:
        conv1d_weight        [hc*H, 1, ple_conv_kernel_size] = [10240, 1, 4]  (depthwise, dilated
                             by ngram_size; stored FLAT like the GDN conv1d, not as an nn.Conv1d)
        key_proj.weight      [hc*H, ple_embed_dim]  = [10240, 2560]
        value_proj.weight    [H, ple_embed_dim]     = [ 2560, 2560]
        norm_conv/key/query  [hc*H]                 = [10240]  grouped (1+w), group = hidden_size
        ple_embedding.layer_multipliers [ngram_size] int64

    `ple_embed_dim` (2560) is the width of one gathered n-gram embedding = 16 hash heads x 160,
    which is exactly the row width `row_table.py` already validated on the real table.

    THE MATH, transcribed from `modeling_qwen4_exp.py::Qwen4ExpTextPLELayer.forward`:

        key   = norm_key(key_proj(e)).unflatten(-1, (hc, H))       # e = n-gram embedding (T, 2560)
        value = value_proj(e)                                      # (T, H)
        query = norm_query(hidden_wide).unflatten(-1, (hc, H))
        gate  = (key * query).sum(-1, keepdim=True) / sqrt(H)      # (T, hc, 1)
        gate  = sign(gate) * sqrt(clamp_min(|gate|, 1e-6))         # signed sqrt "soft" gate
        gv    = sigmoid(gate) * value.unsqueeze(-2)                # (T, hc, H) -> flatten to wide
        out   = gv + silu(dilated_depthwise_conv(norm_conv(gv)))

    Note the asymmetry in the last line, and that it is silent if got wrong: the conv is fed the
    NORMED gated value while the residual adds the UNNORMED one. The conv is depthwise over all
    10240 channels with kernel 4 and dilation `ngram_size` = 3, i.e. a receptive field of 10 tokens
    reading positions t, t-3, t-6, t-9 — hence a 9-column state per sequence.

    The block returns a wide (10240) delta the decoder layer ADDS to the stream; it does not
    consume or produce the 2560-wide mixed view.

    GRAPH CAPTURE. The decode path is shape-static: an `index_select` from the persistent
    `PLEStateCache.conv_state`, one concat of the single new column, four scaled adds (the four
    dilated taps), and an `index_copy_` back. No host sync, no allocation keyed on the batch, and
    the n-gram embeddings arrive in a static staging buffer filled before replay (`ple/runtime.py`).
    The prefill path loops over sequences with `F.conv1d`; prefill is not captured in this engine.
    """

    def __init__(self, config: "ModelConfig") -> None:
        hs = config.hidden_size
        wide = config.hc_hidden_size
        embed = config.ple_embed_dim
        k = config.ple_conv_kernel_size
        assert embed and k, "qwen4_exp PLE needs ple_embed_dim and ple_conv_kernel_size"
        self.ple_embedding = Qwen4ExpNGramEmbedding(config)
        self.conv1d_weight = torch.empty(wide, 1, k)
        self.key_proj = LinearReplicated(embed, wide, has_bias=False)
        self.value_proj = LinearReplicated(embed, hs, has_bias=False)
        self.norm_key = GroupedRMSNorm(wide, group_size=hs, eps=config.rms_norm_eps)
        self.norm_query = GroupedRMSNorm(wide, group_size=hs, eps=config.rms_norm_eps)
        self.norm_conv = GroupedRMSNorm(wide, group_size=hs, eps=config.rms_norm_eps)
        self._hc = config.hc_count
        self._hs = hs
        self._wide = wide
        # Dilation is `ngram_size`, NOT the conv kernel size — the conv strides the same n-gram
        # spacing the hash does. `state_len = (k - 1) * dilation` = 9 for (4, 3); a state sized
        # `k - 1` = 3 (the GDN convention) would silently truncate the receptive field to 4 tokens.
        self._dilation = int(config.ngram_size)
        self._state_len = (int(k) - 1) * self._dilation

    # -- state geometry, for whoever allocates the cache -------------------

    @property
    def conv_state_len(self) -> int:
        return self._state_len

    def make_state_cache(self, *, num_slots: int, eos_token_id: int, device, dtype):
        """Allocate the `PLEStateCache` this block's geometry implies. Keeping the constructor here
        is what stops the 9 from being re-derived (or mis-derived) at the allocation site."""
        from minisgl.ple import PLEStateCache

        return PLEStateCache(
            num_slots=num_slots,
            wide=self._wide,
            state_len=self._state_len,
            context_len=self._dilation - 1,
            eos_token_id=eos_token_id,
            dtype=dtype,
            device=device,
        )

    # -- compute -----------------------------------------------------------

    def _short_conv(self, x: torch.Tensor, batch: "PLEBatch", conv_state: torch.Tensor):
        """`x` is the NORMED gated value, flat (T, wide). Returns silu(conv(x)) as (T, wide) and
        updates `conv_state` in place for every sequence in the batch."""
        w = self.conv1d_weight  # (wide, 1, k)
        d, s = self._dilation, self._state_len
        idx = batch.state_indices

        if batch.is_decode:
            # One token per sequence -> the conv degenerates to k scaled adds over a 10-wide window
            # (9 cached + 1 new). Fully static: no F.conv1d, no per-sequence Python, capturable.
            # Padding rows all carry slot 0, so the `index_copy_` writes several times to the NULL
            # slot; the duplicate-index order is unspecified but every writer targets the slot no
            # sequence owns, so nothing real is disturbed.
            st = conv_state.index_select(0, idx)  # (N, wide, s)
            full = torch.cat([st, x.unsqueeze(-1)], dim=-1)  # (N, wide, s+1)
            taps = w.shape[-1]
            out = full[..., 0] * w[:, 0, 0]
            for j in range(1, taps):
                out = out + full[..., j * d] * w[:, 0, j]
            conv_state.index_copy_(0, idx, full[..., 1:])
            return F.silu(out)

        # Prefill / chunked prefill: varlen. One F.conv1d per sequence over its own
        # [state | chunk] window; the last `s` columns of that window become the new state. The slot
        # ids come from the HOST list, not `idx.tolist()` — the latter is a device->host sync in the
        # middle of a forward.
        outs: List[torch.Tensor] = []
        start = 0
        if not batch.seq_lens:
            return x[:0]
        for slot, n in zip(batch.slots, batch.seq_lens):
            xs = x[start : start + n].t().unsqueeze(0)  # (1, wide, n)
            full = torch.cat([conv_state[slot].unsqueeze(0), xs], dim=-1)  # (1, wide, s+n)
            y = F.conv1d(full, w, groups=self._wide, dilation=d)  # (1, wide, n)
            # Spec VERIFY: stash the `[state | chunk]` window BEFORE collapsing it to the last `s`
            # columns. The accepted-prefix state is `full[0][:, keep : keep+s]`, which is already in
            # hand here and unrecoverable afterwards — the write below keeps only the t == n case,
            # i.e. the state after every draft including the rejected ones. `commit_verified`
            # installs the right slice once the accept loop knows `keep`. Costs one reference to a
            # tensor this forward allocated anyway; nothing is copied.
            if batch.defer_commit and slot:
                batch.conv_windows[slot] = full[0]
            conv_state[slot].copy_(full[0, :, -s:])
            outs.append(y[0].t())
            start += n
        return F.silu(torch.cat(outs, dim=0) if len(outs) > 1 else outs[0])

    def compute(
        self,
        hidden: torch.Tensor,
        embeddings: torch.Tensor,
        batch: "PLEBatch",
        conv_state: torch.Tensor,
    ) -> torch.Tensor:
        """The block, with every runtime input passed explicitly. `forward` is the thin wrapper that
        reads them out of the global context; tests drive THIS, so the numerics can be checked
        without an engine."""
        hc, hs = self._hc, self._hs
        t = hidden.shape[0]
        key = self.norm_key.forward(self.key_proj.forward(embeddings)).view(t, hc, hs)
        value = self.value_proj.forward(embeddings)  # (T, hs)
        query = self.norm_query.forward(hidden).view(t, hc, hs)
        gate = (key * query).sum(dim=-1, keepdim=True) / math.sqrt(hs)
        # Signed sqrt. `clamp_min` BEFORE the sqrt (on |gate|, not on gate) — it is a gradient guard
        # upstream, and moving it changes the value for |gate| < 1e-6.
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gated = torch.sigmoid(gate) * value.unsqueeze(-2)  # (T, hc, hs)
        gated = gated.reshape(t, hc * hs)
        # The conv sees the NORMED gated value; the residual adds the UNNORMED one.
        return gated + self._short_conv(self.norm_conv.forward(gated), batch, conv_state)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """`hidden` is the wide (hc*H) stream. Returns the wide delta the layer adds to it.

        Everything host-side — the n-gram hash, the NVMe row gather, the H2D — happened in
        `PLERuntime.prepare` before this forward; see `minisgl/ple/runtime.py` for why that split is
        what makes the decode step capturable.
        """
        rt = getattr(get_global_ctx(), "ple", None)
        if rt is None or rt.batch is None:
            raise RuntimeError(
                "qwen4_exp PLE: no staged batch. `PLERuntime.prepare(slots, token_lists)` must run "
                "on the host BEFORE the model forward — it hashes the recent tokens, gathers the "
                "16 rows/token from the NVMe-resident n-gram table and copies them into the static "
                "device buffer this layer reads. There is no fallback: skipping the block would "
                "drop the n-gram features silently, which costs quality and errors nowhere."
            )
        batch = rt.batch
        if batch.embeddings.shape[0] != hidden.shape[0]:
            raise ValueError(
                f"PLE staged {batch.embeddings.shape[0]} token embeddings but the forward carries "
                f"{hidden.shape[0]} tokens — the staging ran against a different batch."
            )
        return self.compute(hidden, batch.embeddings, batch, rt.state.conv_state)


class Qwen4ExpDecoderLayer(BaseOP):
    """One decoder block. NOTE what is ABSENT: there is no `input_layernorm` and no
    `post_attention_layernorm`. The hyper-connection's own `hc_norm` is the pre-block norm, and the
    checkpoint ships no such tensors (upstream literally `delattr`s them —
    `sglang/srt/models/qwen4_exp.py::_init_qwen4_exp_layer_extensions`). Adding either would be an
    unfillable key at load.
    """

    def __init__(
        self,
        config: "ModelConfig",
        layer_id: int,
        *,
        is_gdn: bool,
        gdn_layer_id: int | None,
        attn_kv_id: int | None,
        expert_quant,
    ) -> None:
        if is_gdn:
            assert gdn_layer_id is not None
            q = config.quant

            def _gdn_method(module: str):
                name = f"model.layers.{layer_id}.linear_attn.{module}"
                return create_linear_method(q.for_module(name) if q is not None else None)

            gdn = QwenGatedDeltaNet(
                hidden_size=config.hidden_size,
                num_k_heads=config.linear_num_key_heads,
                num_v_heads=config.linear_num_value_heads,
                head_k_dim=config.linear_key_head_dim,
                head_v_dim=config.linear_value_head_dim,
                conv_kernel_size=config.linear_conv_kernel_dim,
                tp_size=get_tp_info().size,
                eps=config.rms_norm_eps,
                # "sigmoid" for this checkpoint (`output_gate_type`), NOT the silu default. The gate
                # multiplies every one of the 36 GDN layers' outputs; the wrong one is degenerate
                # text with no error (GATE-5 asks the .so to confirm it consumed the argument).
                activation=config.gdn_output_gate,
                dtype=torch.get_default_dtype(),
                device=torch.device("meta"),
                qkvz_method=_gdn_method("in_proj_qkv"),
                ba_method=_gdn_method("in_proj_b"),
                out_proj_method=_gdn_method("out_proj"),
            )
            self.linear_attn = GDNLinearAttn(gdn, gdn_layer_id)
            self._attn_op: BaseOP = self.linear_attn
        else:
            assert attn_kv_id is not None
            self.self_attn = Qwen4ExpAttn(config, layer_id, attn_kv_id=attn_kv_id)
            self._attn_op = self.self_attn
        # PLE sits on exactly one layer, as a SIBLING of the mixer (not inside it).
        if layer_id in config.ple_layer_ids:
            self.ple = Qwen4ExpPLE(config)
        self.mlp = Qwen3_5MoeSparseBlock(config, expert_quant)
        self.attn_hyper_connection = _make_hc(config, use_combine=True)
        self.mlp_hyper_connection = _make_hc(config, use_combine=True)
        self._layer_id = layer_id
        self._has_ple = layer_id in config.ple_layer_ids

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """`hidden` is the hc_count-WIDE stream in and out. Dataflow transcribed from
        `sglang/srt/models/qwen4_exp.py` (`_prepare_qwen4_exp_attn` / `_prepare_qwen4_exp_mlp` /
        `_postprocess_qwen4_exp_layer`).

        There is no separate `residual` carried between layers the way the Qwen3.5 fused-norm path
        does — the WIDE stream IS the residual, and `hc_norm` inside `mix` is the pre-block norm this
        checkpoint ships instead of an input_layernorm. Two orderings here are load-bearing and
        silent if swapped: the PLE delta is added to the wide stream BEFORE the attention mix (it is
        a sibling of the mixer, not a term inside it), and each `combine` consumes the residual pair
        from ITS OWN `mix` — the mlp block's gate must read the post-attention stream, not the
        pre-attention one."""
        if self._has_ple:
            hidden = hidden + self.ple.forward(hidden)
        x, res = self.attn_hyper_connection.mix(hidden)
        x = self._attn_op.forward(x)
        hidden = self.attn_hyper_connection.combine(x, res)
        x, res = self.mlp_hyper_connection.mix(hidden)
        x = self.mlp.forward(x)
        return self.mlp_hyper_connection.combine(x, res)


class Qwen4ExpModel(BaseOP):
    def __init__(self, config: "ModelConfig") -> None:
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size, embedding_dim=config.hidden_size
        )
        gdn_pos = {gid: pos for pos, gid in enumerate(config.gdn_layer_ids)}
        attn_pos = {aid: pos for pos, aid in enumerate(config.full_attn_layer_ids)}
        expert_quant = config.quant
        self.layers = OPList(
            [
                Qwen4ExpDecoderLayer(
                    config,
                    lid,
                    is_gdn=lid in gdn_pos,
                    gdn_layer_id=gdn_pos.get(lid),
                    attn_kv_id=attn_pos.get(lid),
                    expert_quant=expert_quant,
                )
                for lid in range(config.num_layers)
            ]
        )
        # THE FINAL NORM. There is deliberately no `self.norm`: this checkpoint ships no
        # `model.language_model.norm.weight`, and the top-level `hyper_connection_mixer` — the same
        # hyper-connection block with use_combine=False, hence 3 tensors and no block_inject_weight —
        # is what folds the 10240-wide stream down to the 2560 hidden `lm_head` consumes.
        self.hyper_connection_mixer = _make_hc(config, use_combine=False)
        self._hc_count = config.hc_count
        self._capture_layer_ids: List[int] | None = None

    def set_capture_layers(self, ids: List[int] | None) -> None:
        if ids:
            # Aux capture exists to feed a draft head (EAGLE3/DFlash/MTP) the exact feature it was
            # trained on. Here the per-layer stream is the hc_count-WIDE residual (10240), not the
            # 2560 hidden every drafter in this engine consumes, and there is no defensible way to
            # pick one of the 4 branches or to fold them (the mixer's fold is trained for the LM
            # head, not for a drafter). Handing back the wide tensor would be a silent shape error
            # in the drafter's fc, and handing back a folded one would be silently OOD. Refuse.
            raise NotImplementedError(
                "qwen4_exp does not support aux-hidden capture: its per-layer residual stream is "
                f"the {self._hc_count}x-wide hyper-connection stream, not the hidden-size feature a "
                "draft head expects. Serve with --spec-algorithm none/ngram (the checkpoint's own "
                "mtp.* head is bring-up plan T8.1)."
            )
        self._capture_layer_ids = None

    def forward(
        self, input_ids: torch.Tensor, return_hidden: bool = False
    ) -> "torch.Tensor | Tuple[torch.Tensor, torch.Tensor, None]":
        # Build the QSA plan ONCE per forward, before any layer runs. Everything in it is
        # batch-derived and layer-independent (positions, the visible-block window, the compressed
        # page table, the compression plan), so the 12 index layers share one index-arithmetic bill.
        qsa = getattr(get_global_ctx(), "qsa", None)
        if qsa is not None:
            qsa.prepare(get_global_ctx().batch)
        x = self.embed_tokens.forward(input_ids)
        # The stream starts as hc_count copies of the embedding (upstream `_prepare_qwen4_exp_attn`
        # does this lazily on the first layer; doing it once here is identical and cheaper).
        hidden = torch.cat([x] * self._hc_count, dim=-1)
        for layer in self.layers.op_list:
            hidden = layer.forward(hidden)
        # `hyper_connection_mixer.mix` REPLACES the final norm: this checkpoint ships no
        # `model.norm.weight`. Its `.mix` folds hc_count*hidden -> hidden for the lm_head; its second
        # return value (the residual pair) is the mixer's own business and is discarded, exactly as
        # upstream does (`hidden_states, _ = self.hyper_connection_mixer.mix(hidden_states)`).
        mixed = self.hyper_connection_mixer.mix(hidden)[0]
        if return_hidden:
            # The WIDE pre-mixer stream is what upstream hands its MTP head (`hc_hidden_states`), so
            # it is the honest second return here — but nothing in this engine consumes a wide seed
            # yet, and `aux_hidden` is None because capture is refused above.
            return mixed, hidden, None
        return mixed


def _assert_tp_divides(config: "ModelConfig", tp: int) -> None:
    """Refuse a TP degree this checkpoint's head/width counts do not divide, naming the count.

    This REPLACED a blanket `tp > 1` refusal (2026-09-04). That refusal was correct while there was
    no `_shard_qwen4_exp` — the generic sharders fall through to "replicate", which would have
    loaded the whole 84 GB on every rank and failed somewhere unrelated. Now the shard rules exist,
    and what is left is the real constraint: TP is legal exactly where every partitioned count
    divides. Checked HERE, at build, so an indivisible degree is a named refusal at second zero
    rather than a shape mismatch several minutes into a 38-shard load.

    Note `num_kv_heads = 2` for this checkpoint: TP=2 is the LAST legal degree on the GQA side, and
    TP=4 is refused by this function rather than by an unhelpful assert inside AttentionLayer. The
    box has two cards, so that bound is not currently reachable in practice.
    """
    if tp <= 1:
        return
    counts = {
        "num_attention_heads": config.num_qo_heads,
        "num_key_value_heads": config.num_kv_heads,
        "linear_num_key_heads": config.linear_num_key_heads,
        "linear_num_value_heads": config.linear_num_value_heads,
        "moe_intermediate_size": config.moe_intermediate_size,
        "shared_expert_intermediate_size": config.shared_expert_intermediate_size,
    }
    bad = {k: v for k, v in counts.items() if v and int(v) % tp}
    if bad:
        raise NotImplementedError(
            f"qwen4_exp cannot serve at tp_size={tp}: "
            + ", ".join(f"{k}={v} is not divisible by {tp}" for k, v in sorted(bad.items()))
            + ". Every one of these is a per-rank partition (attention heads, GDN key/value heads, "
            "the routed- and shared-expert intermediate width); an uneven split would give one rank "
            "a differently-shaped partial for its all-reduce. Serve at a tp that divides them."
        )
    # NVFP4 packs the routed experts along the INPUT K in 2-per-byte units with a group scale every
    # 16 elements, so the ROW-parallel down_proj needs its per-rank K to stay a whole number of both.
    # `_shard_qwen4_exp` raises on the scale axis at load; saying it here as well means the operator
    # learns it before paying for a load, and the two checks are the same arithmetic stated twice on
    # purpose — this one is reachable with no checkpoint on disk.
    inter = int(config.moe_intermediate_size or 0)
    gs = int(getattr(config.quant, "group_size", 0) or 16) if config.quant is not None else 16
    if inter and (inter // tp) % gs:
        raise NotImplementedError(
            f"qwen4_exp at tp_size={tp}: moe_intermediate_size/{tp} = {inter // tp} is not a "
            f"multiple of the NVFP4 group size {gs}. The routed down_proj is row-parallel over K, "
            f"so its packed bytes (K/2) and its group scales (K/{gs}) would round to different rank "
            f"boundaries and each rank would dequantize its columns with another rank's scales."
        )


class Qwen4ExpMTPAttn(Qwen3_5MTPAttn):
    """The MTP layer's self-attention: `Qwen3_5MTPAttn`'s draft entry points, plus the QSA indexer
    submodule the checkpoint ships for this layer.

    THE INDEXER IS BUILT AND NEVER CALLED, exactly as it is on the backbone's full-attention layers
    (`QSAIndexer`: "Not called by anything"). It exists here so `mtp.layers.0.self_attn.indexer.*`
    has somewhere to load; the draft chain then runs DENSE, which is the same regime the backbone
    already serves in below `indexer_budget` and not a new approximation. It is also the only
    regime available: the draft attention keeps its OWN [max_slots, max_ctx, nkv, hd] buffer and
    never touches the paged KV pool, so there is no compressed page table for a selection to index.

    A dense draft cannot make the serve wrong. Speculative decoding is lossless by construction —
    every draft is verified against the target — so an imperfect draft head costs ACCEPTANCE RATE
    and nothing else. That is why this is a defensible first implementation rather than a guess.

    WHERE IT STOPS BEING FREE (audit [11], threshold checked against the code 2026-09-18). Below
    `indexer_budget` visible tokens the target's selection takes EVERY visible key, so the two key
    sets are identical and the dense draft is not an approximation at all — it is the same
    computation. Above it they genuinely diverge: the target attends `indexer_budget /
    indexer_compress_ratio` compressed blocks selected from the WHOLE context (2048 tokens' worth,
    chosen from anywhere), while this ring attends the most recent `_idx_width` positions it has
    written. Same COUNT of keys, different SET, and the drafter's is strictly the recency one.

    So acceptance is expected to fall on long contexts, and it falls silently — a spec measurement
    taken on short prompts does not characterise the same drafter on a 32k one. The adaptive verify
    width (spec/width.py) reacts to the acceptance drop and narrows, so this shows up as reduced
    speculative GAIN rather than as a stall. Closing it is not a bug fix: it needs the draft head to
    run its own QSA selection over its own ring, which needs a compressed index for a buffer that
    deliberately has none (see the paragraph above). Measure before building that — on this box spec
    is a measured loss on the offload arm for reasons that have nothing to do with this.
    """

    def __init__(self, config: "ModelConfig", layer_id: int) -> None:
        super().__init__(config, layer_id)
        # index_layer_id 0: this is the head's only attention layer. The submodule is parameter
        # storage; nothing reads the id on a path that runs.
        self.indexer = QSAIndexer(config, index_layer_id=0)


class Qwen4ExpMTPHead(BaseOP):
    """Qwen3.8-Flash-Next MTP self-speculation head (`mtp.*`), bring-up plan T8.1.

    NOT the Qwen3.5 head with a different prefix. Three things differ, and each one is silent
    rather than loud if it is transcribed from the tensor shapes instead of from the reference
    (`sglang/srt/models/qwen4_exp_mtp.py::_fuse_residual_linear_shared`):

    1. **The seed stays hc_count-WIDE.** `fc_hidden` is [2560, 2560] while `pre_fc_norm_hidden` is
       [10240], which reads like something must fold 10240 -> 2560 between them — and the head even
       ships a `hyper_connection_mixer` that performs exactly that fold for the backbone. It does
       not do it here. `fc_hidden` is applied PER HYPER-CONNECTION BRANCH (the wide stream is
       viewed as [.., hc, H]), the embedding term is BROADCAST-added to all four branches, and the
       result stays wide and feeds the layer directly. The mixer is for the layer OUTPUT, where it
       replaces a final norm before the lm_head, exactly as in the backbone. Folding the seed and
       re-widening it type-checks end to end, produces fluent-looking drafts, and would show up
       only as an acceptance rate near zero.

    2. **The layer is a hyper-connection block, not a pre/post-norm block.** `mtp.layers.0` ships
       `attn_hyper_connection` / `mlp_hyper_connection` at [4, 10240] and NO input_layernorm or
       post_attention_layernorm — `hc_norm` inside `mix` is the pre-block norm. Same structure as
       `Qwen4ExpDecoderLayer`, so the dataflow here is that layer's, not Qwen3.5's.

    3. **`pre_fc_norm_hidden` is GROUPED, and this DIVERGES from upstream sglang.** Both norms use
       the `(1 + w)` gain; `pre_fc_norm_embedding` is a plain 2560 norm over the embedding. But
       `pre_fc_norm_hidden` is [10240] — it touches the WIDE stream, and every other wide norm in
       this checkpoint normalizes each hc branch on its own (`hc_norm`, and the PLE's
       `norm_key`/`norm_query`/`norm_conv`; grouped in HF transformers AND in sglang). sglang's MTP
       head is the lone exception, building it as a full-width `GemmaRMSNorm(hc_count*hidden_size)`;
       HF ships no MTP head at all, so it cannot arbitrate. A/B on this checkpoint, same 4 prompts,
       same protocol, sampled: GROUPED 0.605 accepted-drafts/verify vs FULL-WIDTH 0.444 (+36%, ~4.4
       sigma on 451 vs 368 accepted). The two are different functions — one variance over 10240 vs
       four over 2560 — and the difference is SILENT: both draft fluently, they just get rejected.

    Reuses the target's embed_tokens and (untied) lm_head, like every other MTP head here.
    """

    def __init__(self, config: "ModelConfig", embed: VocabParallelEmbedding,
                 lm_head: ParallelLMHead, expert_quant) -> None:
        eps = config.rms_norm_eps
        hs, hc = config.hidden_size, config.hc_count
        self.pre_fc_norm_embedding = RMSNorm(hs, eps=eps, plus_one=True)
        # GROUPED (group = hidden_size), which DIVERGES from upstream sglang — measured, see below.
        #
        # `pre_fc_norm_hidden.weight` is [hc*H] = [10240], i.e. it touches the WIDE residual stream,
        # and every other wide norm in this checkpoint is grouped per branch: `hc_norm` and the PLE's
        # `norm_key`/`norm_query`/`norm_conv` are all `Qwen4ExpTextRMSNorm(..., group_size=hidden_size)`
        # in HF transformers, and sglang uses its own `GroupedGemmaRMSNorm(group_size=hidden_size)` for
        # `hc_norm`. The ONE exception is sglang's MTP head, which builds this norm as a plain
        # full-width `GemmaRMSNorm(hc_count * hidden_size)` — and HF ships no MTP head at all
        # (`_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]`), so it cannot arbitrate.
        #
        # The two are different functions (one variance over 10240 vs four over 2560 each), and the
        # difference is silent: both produce fluent drafts, they just get REJECTED. Measured on this
        # checkpoint, full-width scored mean accept-len 0.28-0.33 against a trained head's expected
        # 0.55-0.8. See tests/qwen4exp_mtp_parity_test.py, whose falsification arm now pins the
        # grouped-vs-full-width distinction (the earlier version transcribed the full-width reading
        # into BOTH the implementation and its reference, so it could not see this).
        self.pre_fc_norm_hidden = GroupedRMSNorm(hc * hs, group_size=hs, eps=eps)
        # REPLICATED for the same reason the Qwen3.5 fc is: these produce the seed that feeds the
        # (head-sharded) MTP layer, so a column-parallel output would hand the layer a truncated
        # hidden under TP>1 — and this arm is TP=2 ONLY (min_tp=2), so that path is not theoretical.
        self.fc_embedding = LinearReplicated(hs, hs, has_bias=False)
        self.fc_hidden = LinearReplicated(hs, hs, has_bias=False)
        # THE LAYER IS THE REPO'S OWN `Qwen4ExpDecoderLayer`, not a transcription of it. The first
        # version of this head hand-copied that layer's dataflow into a private `_block` — the same
        # copy-a-similar-thing mistake KERNEL_CORE_POLICY forbids for kernels. The backbone layer is
        # PROVEN (it serves this model correctly) and a divergence in a copy is silent: the head
        # drafts plausible tokens that are simply never the target's, which is indistinguishable
        # from a weak drafter. Building the real layer and swapping ONLY its attention keeps the
        # hyper-connection order, the MoE routing and the residual handling as the shipped code.
        #
        # `layers` (an OPList of one) also makes the state_dict path `mtp.layers.0.*`, which is
        # EXACTLY the checkpoint's spelling. layer_id = num_layers: past the decoder, so it cannot
        # be in `ple_layer_ids` (the head carries no PLE — upstream sets `ple_layer_ids = []` for
        # the MTP model) and its quant namespace cannot collide with a real decoder layer's.
        # expert_quant=None: the head is BF16 END TO END here (no *_scale / *_packed under `mtp.`),
        # so passing the backbone's quant would hunt for packs that do not exist.
        self.layers = OPList([
            Qwen4ExpDecoderLayer(
                config, config.num_layers,
                is_gdn=False, gdn_layer_id=None, attn_kv_id=0, expert_quant=None,
            )
        ])
        # The draft chain cannot use the paged-KV attention: it runs K steps ahead of the target over
        # its own ring. Swap in the draft-capable attention — the SAME projections, gate and norms
        # (it subclasses the same hierarchy) plus the three draft entry points.
        _layer = self.layers.op_list[0]
        _layer.self_attn = Qwen4ExpMTPAttn(config, config.num_layers)
        _layer._attn_op = _layer.self_attn
        # Folds the layer's wide output to hidden for the lm_head; this checkpoint ships no final
        # norm for the head, exactly as the backbone ships none for itself.
        self.hyper_connection_mixer = _make_hc(config, use_combine=False)
        self._hc_count = hc
        self._hidden_size = hs
        # The width of the key set this head's attention was TRAINED to see: QSA selects up to
        # `indexer_budget` compressed groups, expanded to `budget + ratio - 1` token columns
        # (QSAConfig.index_width). Published so the proposer can size its draft window to it — a
        # recent-history window narrower than this shows the drafter a strictly different context
        # from the one the target attended over. None on a non-QSA build, where the proposer keeps
        # its own measured default.
        self.qsa_index_width = (
            int(config.indexer_budget) + int(config.indexer_compress_ratio) - 1
            if config.indexer_budget and config.indexer_compress_ratio
            else None
        )
        self._embed = embed
        self._lm_head = lm_head

    # -- the MTPProposer contract ---------------------------------------------------------------

    @property
    def self_attn(self):
        """The draft attention, where `MTPProposer` expects to find it (`spec/mtp.py` reads
        `head.self_attn.draft_buffer_dims()` to size the draft-KV ring).

        A PROPERTY, not an attribute: `BaseOP.state_dict` / `load_state_dict` / `post_load` all walk
        `vars(self)`, and a property is not in `vars`. So this exposes the contract without creating
        a second weight path — an attribute alias would make the same tensors reachable as both
        `mtp.self_attn.*` and `mtp.layers.0.self_attn.*`, and the loader would fill one and leave
        the other holding torch.empty garbage."""
        return self.layers.op_list[0].self_attn

    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        return self._embed.forward(tokens)

    def fuse(self, embed_e: torch.Tensor, last_hidden: torch.Tensor) -> torch.Tensor:
        """`last_hidden` is the hc_count-WIDE pre-mixer stream (what
        `Qwen4ExpModel.forward(return_hidden=True)` returns as its second value), NOT a post-norm
        hidden. Returns a wide seed. See point 1 of the class docstring for why this is per-branch."""
        if last_hidden.shape[-1] != self._hc_count * self._hidden_size:
            raise ValueError(
                f"qwen4_exp MTP seeds from the {self._hc_count}x{self._hidden_size}-wide pre-mixer "
                f"stream; got a {last_hidden.shape[-1]}-wide hidden. A folded (hidden-size) seed is "
                f"the one mistake this head cannot detect at runtime — it would run and draft badly."
            )
        e = self.fc_embedding.forward(self.pre_fc_norm_embedding.forward(embed_e))   # [.., H]
        h = self.pre_fc_norm_hidden.forward(last_hidden)                             # [.., hc*H]
        branches = h.view(*h.shape[:-1], self._hc_count, self._hidden_size)          # [.., hc, H]
        branches = self.fc_hidden.forward(branches)                                  # per-branch
        return (e.unsqueeze(-2) + branches).reshape(*h.shape)                        # [.., hc*H]

    def _block(self, wide: torch.Tensor, attn_fn) -> Tuple[torch.Tensor, torch.Tensor]:
        """The layer body, shared by step()/step_masked(): identical to `Qwen4ExpDecoderLayer.forward`
        (no PLE — the head carries none), with the attention call injected so the two draft paths
        differ ONLY in their attention core, as they do on the Qwen3.5 head."""
        layer = self.layers.op_list[0]
        # Exactly `Qwen4ExpDecoderLayer.forward`, with only the attention call substituted (the
        # draft attention takes ring buffers instead of the paged context). Every other statement
        # reads the LAYER's own attributes, so the order cannot drift from the shipped one.
        x, res = layer.attn_hyper_connection.mix(wide)
        x = attn_fn(x)
        wide = layer.attn_hyper_connection.combine(x, res)
        x, res = layer.mlp_hyper_connection.mix(wide)
        x = layer.mlp.forward(x)
        wide = layer.mlp_hyper_connection.combine(x, res)
        # `.mix` replaces the final norm, as it does for the backbone's lm_head.
        mixed = self.hyper_connection_mixer.mix(wide)[0]
        return self._lm_head.logits_all_rows(mixed), wide

    def step_masked(
        self, fused: torch.Tensor, positions: torch.Tensor,
        k_buf: torch.Tensor, v_buf: torch.Tensor, slot_rows: torch.Tensor,
        write_col: torch.Tensor, meta: "DraftAttnMeta",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._block(
            fused,
            lambda x: self.layers.op_list[0].self_attn.forward_draft_masked(
                x, positions, k_buf, v_buf, slot_rows, write_col, meta),
        )

    @torch.inference_mode()
    def seed_buffered(
        self, tokens: torch.Tensor, prev_hidden: torch.Tensor, positions: torch.Tensor,
        k_buf: torch.Tensor, v_buf: torch.Tensor, slot: int, start_col: int = 0,
    ) -> None:
        """Seed the draft-KV buffer from the prompt prefill. Mirrors `step_masked`'s fuse + pre-block
        norm before the attention, which HERE is `attn_hyper_connection.mix` rather than an
        input_layernorm — the residual pair it returns is discarded because seeding writes k/v and
        runs no attention."""
        fused = self.fuse(self.embed(tokens), prev_hidden)
        _layer = self.layers.op_list[0]
        x = _layer.attn_hyper_connection.mix(fused)[0]
        _layer.self_attn.seed_kv_masked(x, positions, k_buf, v_buf, slot, start_col)


class Qwen4ExpForConditionalGeneration(BaseLLMModel):
    def __init__(self, config: "ModelConfig") -> None:
        if not config.is_qwen4_exp:
            raise ValueError(
                f"Qwen4ExpForConditionalGeneration built from a {config.model_type!r} config"
            )
        tp = get_tp_info().size
        _assert_tp_divides(config, tp)
        if config.quant is None and config.unparsed_quant_method:
            # Do not let "we could not read the quantization_config" look like "this checkpoint is
            # bf16". The NVFP4 body is ~63 GB packed; built unquantized it is neither loadable nor
            # servable, and the first symptom would be a key/shape miss deep in the loader.
            logger.warning_rank0(
                f"qwen4_exp: config declares quantization_config quant_method="
                f"{config.unparsed_quant_method!r}, which QuantConfig.from_hf does NOT parse — every "
                f"module is being built FULL PRECISION. For RadixArk/Qwen3.8-Flash-Next-NVFP4 that "
                f"is wrong (routed experts are NVFP4 E2M1 + a two-level scale). Bring-up plan T1.1 "
                f"adds the modelopt arm; until then this model builds but cannot load that "
                f"checkpoint's experts."
            )
        self.model = Qwen4ExpModel(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            # UNTIED in this checkpoint (`lm_head.weight` is a top-level tensor of its own).
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )
        # The MTP self-speculation head (bring-up plan T8.1). Built only when the config says the
        # checkpoint ships one AND spec is actually mtp — `from_hf` zeroes mtp_num_hidden_layers
        # otherwise, so a non-spec serve pays nothing and loads no mtp.* tensor.
        self.mtp = (
            Qwen4ExpMTPHead(config, embed=self.model.embed_tokens, lm_head=self.lm_head,
                            expert_quant=None)
            if config.mtp_num_hidden_layers > 0
            else None
        )
        super().__init__()
        # Say what this build does NOT do, once, at boot — where an operator reads it — rather than
        # only at the moment something raises deep in a forward.
        # Counts come from the CONFIG, not from the shipping checkpoint's 48/36/12: a layer-subset
        # bring-up config (or any future variant) built the numbers into this line as a literal and
        # the boot log then stated a depth the engine was not running.
        logger.info_rank0(
            f"qwen4_exp built: {config.num_layers} layers ({len(config.gdn_layer_ids)} GDN + "
            f"{len(config.full_attn_layer_ids)} full-attn), PLE on decoder index "
            f"{sorted(config.ple_layer_ids)}, {config.hc_count}x{config.hidden_size}-wide "
            f"hyper-connection residual. NOT implemented in this build:"
        )
        for what, why in UNIMPLEMENTED:
            logger.info_rank0(f"  * {what}\n      ({why})")
        self._config = config
        self._qsa_built = False

    # ---- QSA runtime construction -------------------------------------------------------------
    def build_qsa_runtime(self, *, force: bool = False):
        """Build `ctx.qsa` — the index-key cache + the index RoPE — or explain why it is not built.

        Called once, from `prepare_qsa` on the first forward (which is a WARMUP forward, so the
        allocations land before graph capture). Returns the runtime or None; when it returns None
        the full-attention layers run dense and `assert_dense_is_exact` re-arms the >budget refusal,
        which is the honest behaviour — this must never silently serve dense above the budget.

        `force=True` turns each of the three "not available" cases into a raise, so a test that
        MEANS to exercise the sparse path cannot accidentally measure the dense one. (The gate
        harness uses it; that is the `ab-harness-must-assert-provenance` rule applied to a feature
        whose failure mode is running the OTHER implementation and looking fine.)
        """
        from minisgl.attention.qsa import QSAIndexCache, QSARuntime, parse_qsa_profile
        from minisgl.attention.qsa import ops as qsa_ops
        from minisgl.layers.rotary import get_rope

        ctx = get_global_ctx()
        if os.environ.get("MINISGL_QSA", "1") == "0":
            if force:
                raise RuntimeError("MINISGL_QSA=0 disables the sparse path this run requires")
            logger.warning_rank0("qwen4_exp: MINISGL_QSA=0 — full-attn layers run DENSE (<=budget)")
            return None
        profile = parse_qsa_profile(self._config)
        if profile is None:
            if force:
                raise RuntimeError("config carries no QSA indexer fields")
            return None
        indexers = self.indexers()
        if not indexers:
            if force:
                raise RuntimeError("this build has no full-attention layer, so no indexer")
            return None
        try:
            profile.require_page_size(ctx.page_size)
            qsa_ops.backend()  # resolves + imports qsa_index unless MINISGL_QSA_OPS=torch
        except Exception as exc:
            if force:
                raise
            logger.warning_rank0(
                f"qwen4_exp: QSA sparse path unavailable ({exc}) — full-attn layers run DENSE, "
                f"and any request past indexer_budget will be REFUSED rather than served densely."
            )
            return None
        rot = self._config.rotary_config
        # The index RoPE is the LAYER'S OWN attention rope — same rotary_dim/base/scaling — read
        # over the 128-wide index head instead of the 256-wide attention head. Only the head width
        # differs, so cos/sin are literally the same table; building it here rather than reaching
        # into `self.attn.rotary` keeps the 128-vs-256 view explicit.
        page_table = ctx.page_table
        with torch.device(page_table.device):
            rotary = get_rope(
                head_dim=profile.head_dim,
                rotary_dim=rot.rotary_dim,
                max_position=rot.max_position,
                base=rot.base,
                rope_scaling=tuple(rot.scaling.items()) if rot.scaling else None,
                interleave=rot.interleave,
            )
        kv = ctx.kv_cache
        n_slots = kv.k_cache(0).shape[0] * kv.k_cache(0).shape[1]
        cache = QSAIndexCache(
            profile=profile,
            num_index_layers=len(indexers),
            num_req_rows=page_table.shape[0],
            num_kv_slots=n_slots,
            dtype=torch.bfloat16,
            device=page_table.device,
        )
        logger.info_rank0(
            f"qwen4_exp QSA ACTIVE: budget={profile.budget} ratio={profile.compress_ratio} "
            f"block_topk={profile.block_topk} index_width={profile.index_width} "
            f"ops={qsa_ops.backend()} {cache}"
        )
        return QSARuntime(profile, cache, rotary, page_table)

    def prepare_qsa(self, *, force: bool = False) -> None:
        """Idempotently install `ctx.qsa`. Must run OUTSIDE cudagraph capture (it allocates)."""
        ctx = get_global_ctx()
        if self._qsa_built:
            return
        assert not torch.cuda.is_current_stream_capturing(), (
            "QSA runtime built during graph capture — it allocates the index-key cache. Build it "
            "in warmup (the engine calls prepare_qsa before capture)."
        )
        ctx.qsa = self.build_qsa_runtime(force=force)
        self._qsa_built = True

    def forward(self, return_hidden: bool = False):
        if not self._qsa_built:
            # First forward is a warmup (eager, pre-capture), which is exactly where the index-key
            # cache must be allocated. Guarded by the flag so the captured decode path never
            # re-enters this — `prepare_qsa` asserts it is not capturing.
            self.prepare_qsa()
        input_ids = get_global_ctx().batch.input_ids
        if return_hidden:
            # (post-mixer for lm_head, the WIDE pre-mixer stream, aux=None). Nothing consumes the
            # wide seed yet — the only caller of return_hidden is a draft head, and `set_capture_
            # layers` refuses — but returning the right tensor is cheaper than returning a lie.
            mixed, wide, aux = self.model.forward(input_ids, return_hidden=True)
            return self.lm_head.forward(mixed), wide, aux
        return self.lm_head.forward(self.model.forward(input_ids))

    def iter_gdn_layers(self) -> List[GDNLinearAttn]:
        """GDN bridge ops in gdn_layer_id order (the engine sizes gdn_state + warms up from this)."""
        return [
            layer.linear_attn
            for layer in self.model.layers.op_list
            if isinstance(getattr(layer, "linear_attn", None), GDNLinearAttn)
        ]

    def set_capture_layers(self, ids: List[int] | None) -> None:
        self.model.set_capture_layers(ids)  # refuses a non-empty set — see Qwen4ExpModel

    def indexers(self) -> List[QSAIndexer]:
        """Every full-attention layer's QSA indexer, in layer order. Empty on a GDN-only subset."""
        cached = getattr(self, "_indexer_cache", None)
        if cached is None:
            cached = self._indexer_cache = [
                layer.self_attn.indexer
                for layer in self.model.layers.op_list
                if getattr(layer, "self_attn", None) is not None
            ]
        return cached

    def prepare_for_replay(self, batch) -> None:
        """Restate the QSA indexer-budget refusal on the CAPTURED decode path.

        `Qwen4ExpAttn.forward` asks `assert_dense_is_exact(max device_len)` once per forward, and
        that check is the only thing standing between this engine and serving a DIFFERENT model:
        below `indexer_budget` the checkpoint's sparse selection picks every visible key, so this
        engine's dense causal attention is bit-equivalent; above it the two genuinely diverge, and
        dense is not "slightly wrong" — it attends MORE, so the output stays fluent and the
        substitution is undetectable from the text.

        That check is Python inside `forward()`, so under cudagraph capture it runs exactly once —
        at capture, over `dummy_req` rows whose `device_len` is 1 — and never again for the life of
        the process. Restating it in this `GraphRunner.replay` pre-hook is what keeps a request that
        crosses the budget mid-generation raising, instead of quietly switching models at token
        2049. Deliberately NOT removed from `Qwen4ExpAttn.forward`: prefill and any uncaptured
        decode never reach this hook, and that path is where the check has always lived.

        No-op when the build has no full-attention layer at all (a GDN-only layer subset, which the
        bring-up harness routinely runs): nothing dense executes, so there is nothing to bound. The
        bound is a property of the config's `indexer_budget` and is identical on every indexer, so
        the first one answers for all of them.
        """
        qsa = getattr(get_global_ctx(), "qsa", None)
        if qsa is not None:
            # The sparse path is live: there is no budget to enforce — but the SELECTION's static
            # plan has to be refreshed here, because the captured `model.forward()` never re-enters
            # `QSARuntime.prepare`. This hook is the only place per replay that runs on the host.
            qsa.prepare_for_replay(batch)
            return
        idx = self.indexers()
        if not idx:
            return
        idx[0].assert_dense_is_exact(max((r.device_len for r in batch.reqs), default=0))

    def ple_block(self) -> "Qwen4ExpPLE":
        """The single PLE block, for whoever allocates its state cache and stages its batches.

        Returned rather than re-derived at the allocation site: the block owns its conv geometry
        (`conv_state_len` = 9, not the kernel-size-minus-one an ordinary short conv would imply) and
        `make_state_cache` builds the matching `PLEStateCache`. Raises if the model was built with no
        PLE layer at all, which for this checkpoint means `ple_layer_ids` was mis-parsed."""
        blocks = [
            layer.ple for layer in self.model.layers.op_list if getattr(layer, "ple", None) is not None
        ]
        if len(blocks) != 1:
            raise RuntimeError(
                f"expected exactly one qwen4_exp PLE block, found {len(blocks)}. The checkpoint "
                f"ships `layers.1.ple.*` only; `ple_layer_ids` in config.json is 1-BASED and "
                f"ModelConfig converts it."
            )
        return blocks[0]


__all__ = [
    "GroupedRMSNorm",
    "HyperConnection",
    "QSAIndexer",
    "Qwen4ExpAttn",
    "Qwen4ExpPLE",
    "Qwen4ExpDecoderLayer",
    "Qwen4ExpModel",
    "Qwen4ExpForConditionalGeneration",
    "UNIMPLEMENTED",
]
