"""DiffusionGemma — the block-diffusion head on the shared Gemma4 backbone.

`DiffusionGemmaForBlockDiffusion` and `Gemma4ForConditionalGeneration` are the SAME 30-layer stack
(see `models/gemma4.py` for the five traps it carries); this file adds only what block diffusion
needs on top. Three structural facts decide the shape of everything here, and each one is a place a
plausible-looking implementation goes silently wrong:

  1. ENCODER AND DECODER ARE ONE STACK. The reference declares `model.encoder` and `model.decoder`
     as two sub-models, but every text parameter is TIED — the checkpoint ships the decoder copy
     only, plus 30 loose `layer_scalar` buffers on the encoder side (HF ties Parameters, not
     buffers). So one instantiated backbone serves both roles and the loader asserts the 30 stray
     buffers really are equal (`weight.py`), rather than assuming it.
  2. THE TWO ROLES DIFFER ONLY IN CAUSALITY AND SELF-CONDITIONING. The encoder pass is an ordinary
     causal Gemma4 prefill that WRITES the KV cache — `forward()`, inherited semantics, no changes.
     The decoder (canvas) pass is bidirectional over `[encoder KV] ++ [canvas KV]`, writes nothing
     back, and prepends the self-conditioning block — `forward_canvas()`. There is no
     cross-attention anywhere: the decoder's only channel from the prompt is the read-only KV cache.
  3. THE SELF-CONDITIONING SIGNAL IS A SOFT EMBEDDING, NOT LOGITS. The reference carries the
     previous step's `[canvas, 262144]` logits across the step boundary and re-multiplies them by
     the embedding table each step. `soft_embedding()` does that multiply ONCE, immediately, and the
     engine carries the `[canvas, 2816]` result. That is a claim about the CARRIED STATE and only
     about it: ~100x less to hold across the step boundary, and the 268 MiB fp32 softmax transient
     dies inside the step that made it. It is NOT a claim about arithmetic — the matmul is the same
     second LM head vLLM pays, 378 GFLOP per canvas step, which TP splits into 189 GFLOP/rank.
     Sharding is not a reduction. Two agents have now read the old "~100x smaller" wording as a cost
     claim and gone looking for a saving that was never there, so it says which quantity it means.

The head itself is `lm_head` tied to `embed_tokens` (the checkpoint ships no `lm_head.*`), with the
same 30*tanh(x/30) softcap as the AR sibling — except the reference casts to fp32 BEFORE the cap
here, which matters because tanh saturates and an fp16 intermediate would quantize the logits the
entropy bound is about to sort by.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from minisgl.core import get_global_ctx
from minisgl.layers import ParallelLMHead, RMSNorm
from minisgl.layers.norm import RMSNormNoScale
from minisgl.utils import nvtx_annotate

from .base import BaseLLMModel
from .gemma4 import Gemma4DenseMLP, Gemma4Model, _build_vision

if TYPE_CHECKING:
    from minisgl.quant.config import QuantConfig

    from .config import ModelConfig

# Vocabulary columns contracted per soft-embedding matmul chunk, or 0 for "one matmul, whole shard".
#
# THE AXIS IS THE WHOLE POINT. This used to chunk over canvas ROWS (32 at a time) to bound the fp32
# softmax transient, and that is a correct thing to want and the wrong axis to get it on: the softmax
# is the cheap operand and the EMBEDDING SHARD is the expensive one, so 8 row-chunks re-streamed all
# 738 MB of it eight times — 5.9 GB/rank/step against a 0.74 GB floor, measured at 15.3 ms/step by
# both the step timer and rocprofv3. Chunking over VOCAB re-reads nothing: each chunk owns a disjoint
# slice of the shard and of the probability row, and the partial products sum.
#
# The transient it was protecting no longer exists at all — `soft_embedding` now consumes the
# sampler's own `probs`, which that step already computed and is about to drop (see the docstring), so
# there is no second softmax to bound and no reason to chunk for memory. 0 keeps it as one GEMM; a
# positive value is here because a K=131072 contraction is a split-K shape and rocBLAS's choice of
# split is not ours to assume — if the single GEMM ever measures worse than hand-blocking it, this is
# the knob, and blocking K changes only the fp32 summation order.
_SOFT_EMBED_VOCAB_CHUNK = 0


class DiffusionGemmaSelfConditioning(Gemma4DenseMLP):
    """The previous denoising step's soft embedding, folded into this step's input embedding.

    It IS the dense-MLP primitive — same intermediate_size (2112, the DENSE width, not the MoE
    704), same tanh-gelu, same gate/up merge, same row-parallel down — wrapped in two norms, so it
    subclasses rather than re-declaring the projections. `post_norm` is `with_scale=False` (the
    checkpoint ships no tensor for it) and is applied to the SUM UNCONDITIONALLY, including on the
    first denoising step when the signal is zero: it renormalizes the embedding every step, so
    treating "no signal yet" as "skip the block" would feed layer 0 a differently-scaled input on
    step 1 than on every other step.
    """

    def __init__(self, config: "ModelConfig"):
        # `layer_id=None`: this MLP is not layer-scoped, so its quantization is decided by its own
        # module name. It sits in the checkpoint's compressed-tensors ignore list (fp16), but the
        # name is passed rather than hard-coding `quant=None` so a future quantized export builds.
        super().__init__(config, None, quant_module="model.self_conditioning.gate_proj")
        self.pre_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self._post_norm = RMSNormNoScale(eps=config.rms_norm_eps)

    @nvtx_annotate("SelfCond")
    def forward(self, inputs_embeds: torch.Tensor, signal: torch.Tensor | None) -> torch.Tensor:
        """`signal=None` is the first denoising step of a block (the reference passes an explicit
        `zeros_like`). Short-circuiting the MLP for it is EXACT, not an approximation:
        RMSNorm(0) = rsqrt(eps)*0 = 0, gelu(0)*0 = 0, and down_proj carries no bias, so the branch
        contributes exactly zero. The decision is made from the step index, which is identical on
        every TP rank, so the skipped row-parallel all-reduce cannot desync the ranks."""
        if signal is not None:
            inputs_embeds = inputs_embeds + super().forward(self.pre_norm.forward(signal))
        return self._post_norm.forward(inputs_embeds)


class DiffusionGemmaModel(Gemma4Model):
    """The shared backbone plus the decoder-only self-conditioning block."""

    def __init__(self, config: "ModelConfig", expert_quant: "QuantConfig | None"):
        super().__init__(config, expert_quant)
        self.self_conditioning = DiffusionGemmaSelfConditioning(config)

    def forward_canvas(
        self, input_ids: torch.Tensor, self_conditioning: torch.Tensor | None
    ) -> torch.Tensor:
        """One denoising step over the canvas. Identical to the inherited encoder `forward` except
        for the self-conditioning block on the input embedding — the non-causal attention is a
        property of the BATCH (the attention backend reads it off the canvas metadata), not of the
        layer stack, which is why the stack itself needs no diffusion-specific branch."""
        h = self._embed_scaled(input_ids)
        h = self.self_conditioning.forward(h, self_conditioning)
        for layer in self.layers.op_list:
            h = layer.forward(h)
        return self.norm.forward(h)


class DiffusionGemmaForBlockDiffusion(BaseLLMModel):
    def __init__(self, config: "ModelConfig"):
        self.model = DiffusionGemmaModel(config, expert_quant=config.quant)
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )
        self._softcap = config.final_logit_softcapping
        self.vision = _build_vision(config)
        super().__init__()

    def _softcapped(self, logits: torch.Tensor) -> torch.Tensor:
        if self._softcap is None:
            return logits
        # fp32 BEFORE the cap, matching the reference: tanh saturates, so capping in fp16 would
        # quantize exactly the region the diffusion sampler's entropy ordering discriminates in.
        logits = logits.float()
        return torch.tanh(logits / self._softcap) * self._softcap

    def forward(self, return_hidden: bool = False):
        """The ENCODER pass: an ordinary causal Gemma4 forward that writes the KV cache.

        This is the whole of what the engine's prefill path needs, and it is what runs for the
        prompt and for every committed 256-token block. The canvas pass is NOT this method — it
        needs a self-conditioning signal and a non-causal batch — see `forward_canvas`."""
        hidden = self.model.forward(get_global_ctx().batch.input_ids)
        logits = self._softcapped(self.lm_head.forward(hidden))
        if return_hidden:
            return logits, hidden, None
        return logits

    def forward_canvas_hidden(
        self, input_ids: torch.Tensor, self_conditioning: torch.Tensor | None = None
    ) -> torch.Tensor:
        """The CAPTURED half of a denoising step: the 30-layer backbone, ending at the final norm.

        Split out of `forward_canvas` so the canvas cudagraph can capture the backbone WITHOUT the
        LM head, which is a memory decision and not a stylistic one. The head is ~4 kernel launches
        of the ~1000 a canvas step issues, so capturing it buys nothing measurable; but its output is
        `[canvas, 262144]`, and every intermediate a captured region produces is pinned for the life
        of the graph in its private pool — 134 MiB of fp16 logits plus three 268 MiB fp32 stages of
        the softcap, ~1 GiB PER CAPTURED BATCH SIZE, on a 16 GB card that is already holding the
        weights and the KV pool. The hidden state is `[canvas, 2816]` = 1.4 MiB. That is the trade.
        """
        return self.model.forward_canvas(input_ids, self_conditioning)

    @property
    def canvas_softcap(self) -> float | None:
        """The final-logit softcap `canvas_logits` applies, for a caller that takes the RAW logits
        (`softcap=False`) and applies it itself — the fused canvas tail recomputes it per element
        instead of materialising three fp32 [rows, vocab/tp] passes of it."""
        return self._softcap

    def canvas_logits(self, hidden: torch.Tensor, softcap: bool = True) -> torch.Tensor:
        """The EAGER tail of a denoising step: fp32 softcapped logits for EVERY canvas position, over
        THIS RANK'S vocabulary columns — `[rows, vocab/tp]`, NOT `[rows, vocab]`.

        `logits_local_shard` rather than `lm_head.forward`, deliberately, for two reasons. The
        obvious one: `forward` reduces to the last token on a prefill batch, and every canvas
        position is scored. The load-bearing one: nothing downstream INDEXES the vocabulary
        dimension, it only REDUCES over it — logsumexp, softmax, entropy, argmax, multinomial, and
        `soft_embedding`'s `probs @ E` — and every one of those decomposes over disjoint column
        blocks into a per-rank partial plus a `[canvas]`-sized message. So the all_gather that used
        to sit here materialised a 134 MiB tensor on both ranks (plus a `permute().contiguous()` over
        all of it, to undo the gather's rank-major interleave) that neither rank ever read a column
        of. It cost 23.2 ms of a 190 ms step, and the softcap that follows it another 3.1 ms for
        capping columns this rank does not own.

        THE RETURN SHAPE IS TP-DEPENDENT, which is unusual in this file and is why it says so twice.
        `CanvasState.step` reads the width and reduces accordingly (it refuses a width that is
        neither the whole vocabulary nor this rank's shard), and `soft_embedding` takes the shard
        whole instead of slicing it. At tp_size == 1 the shard IS the vocabulary and every caller —
        including the parity fixtures, which compare full-vocab canvas logits against HF — sees
        exactly what it saw before."""
        raw = self.lm_head.logits_local_shard(hidden)
        return self._softcapped(raw) if softcap else raw

    def forward_canvas(
        self, input_ids: torch.Tensor, self_conditioning: torch.Tensor | None = None
    ) -> torch.Tensor:
        """One denoising step, backbone + head, for callers that want the whole thing eagerly (the
        parity fixtures, and any path with no captured graph). The served step runs the two halves
        separately — see `forward_canvas_hidden`."""
        return self.canvas_logits(self.forward_canvas_hidden(input_ids, self_conditioning))

    def soft_embedding(self, probs: torch.Tensor) -> torch.Tensor:
        """The self-conditioning state to carry into the NEXT denoising step: `probs @ E` scaled by
        the embedding's own sqrt(hidden), i.e. the probability-weighted average embedding.

        `probs` is the sampler's OWN full-vocab fp32 softmax of the temperature-scaled logits
        (`DiffusionStep.probs`), not the logits. That is a deliberate coupling and it is what makes
        this cheap: the sampler has to build that exact tensor anyway — the entropy bound, the
        stopping criterion and the multinomial all consume it — and this used to build a second,
        numerically-equivalent copy of it. Two full-vocab softmaxes per step for one distribution.

        The one it takes is `softmax(logits - logsumexp(logits))`, the shift `torch.distributions.
        Categorical` applies, where the reference's soft embedding uses `softmax(logits)` directly.
        Those are the same function of the same input in exact arithmetic and differ in fp32 only by
        rounding (the shift moves every logit by one common constant per row); measured max|delta| on
        the carried state is at the fp32 epsilon of the embedding magnitude. That is the same
        substitution `CanvasState.step` already makes for the multinomial, for the same reason.

        Under TP the embedding table is vocab-sharded, so each rank contracts its own vocab slice and
        the partial sums are all-reduced — the same decomposition the vocab-parallel embedding gather
        uses. When `probs` is already ONE RANK'S SHARD (`probs.shape[1] == count`, which is what the
        vocab-parallel canvas tail hands over) there is nothing to slice: the row is the shard."""
        emb = self.model.embed_tokens
        start, count = emb.vocab_range
        weight = emb.weight[:count]
        # A full-vocab row gets sliced to this rank's columns; an already-sharded row is taken whole.
        # Keying on the width rather than on a flag means a caller cannot pass the wrong one silently:
        # any other width is neither, and is a shape error at the matmul instead of a wrong answer.
        local = probs if probs.shape[1] == count else probs[:, start : start + count]
        chunk = _SOFT_EMBED_VOCAB_CHUNK or count
        out = None
        for lo in range(0, count, chunk):
            hi = min(lo + chunk, count)
            part = local[:, lo:hi].to(weight.dtype) @ weight[lo:hi]
            out = part if out is None else out + part
        if emb.tp_size > 1:
            out = emb._comm.all_reduce(out)
        return out * self.model._scale_tensor(out)


__all__ = ["DiffusionGemmaForBlockDiffusion", "DiffusionGemmaModel", "DiffusionGemmaSelfConditioning"]
