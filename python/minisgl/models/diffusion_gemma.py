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
     engine carries the `[canvas, 2816]` result — mathematically identical, ~100x smaller, and it
     keeps the 268 MiB fp32 softmax transient off the step boundary.

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
from .gemma4 import Gemma4DenseMLP, Gemma4Model

if TYPE_CHECKING:
    from minisgl.quant.config import QuantConfig

    from .config import ModelConfig

# Rows of the canvas processed per softmax chunk when building the soft embedding. The full
# [canvas, vocab] fp32 softmax is 268 MiB for a 256-token canvas at vocab 262144 — as big as the
# logits themselves — and it is a pure transient. Chunking bounds it without changing the result
# (each row's softmax is independent); 32 keeps it at ~33 MiB.
_SOFT_EMBED_CHUNK = 32


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

    def forward_canvas(
        self, input_ids: torch.Tensor, self_conditioning: torch.Tensor | None = None
    ) -> torch.Tensor:
        """One denoising step: full-vocab fp32 softcapped logits for EVERY canvas position.

        `logits_all_rows` rather than `lm_head.forward`, deliberately: the latter reduces to the
        last token on a prefill batch, and every canvas position is scored."""
        hidden = self.model.forward_canvas(input_ids, self_conditioning)
        return self._softcapped(self.lm_head.logits_all_rows(hidden))

    def soft_embedding(self, logits: torch.Tensor) -> torch.Tensor:
        """The self-conditioning state to carry into the NEXT denoising step: `softmax(logits) @ E`
        scaled by the embedding's own sqrt(hidden), i.e. the probability-weighted average embedding.

        `logits` are the TEMPERATURE-SCALED logits the sampler consumed, not the raw ones. The
        reference carries the [canvas, vocab] logits across the step boundary and does this matmul
        at the START of the next step; doing it here is the same arithmetic against the same
        embedding table, and it is what makes the carried state [canvas, hidden] instead of
        [canvas, 262144]. Under TP the embedding table is vocab-sharded, so each rank contracts its
        own vocab slice and the partial sums are all-reduced — the same decomposition the vocab-
        parallel embedding gather uses."""
        emb = self.model.embed_tokens
        start, count = emb.vocab_range
        weight = emb.weight[:count]
        out = logits.new_empty((logits.shape[0], weight.shape[1]), dtype=weight.dtype)
        for lo in range(0, logits.shape[0], _SOFT_EMBED_CHUNK):
            hi = min(lo + _SOFT_EMBED_CHUNK, logits.shape[0])
            # fp32 softmax over the FULL vocab (the normalizer is global), then the local slice.
            probs = logits[lo:hi].softmax(dim=-1, dtype=torch.float32)
            out[lo:hi] = probs[:, start : start + count].to(weight.dtype) @ weight
        if emb.tp_size > 1:
            out = emb._comm.all_reduce(out)
        return out * self.model._scale_tensor(out)


__all__ = ["DiffusionGemmaForBlockDiffusion", "DiffusionGemmaModel", "DiffusionGemmaSelfConditioning"]
