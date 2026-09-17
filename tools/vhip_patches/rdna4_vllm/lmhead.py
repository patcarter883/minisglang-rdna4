# SPDX-License-Identifier: Apache-2.0
"""Route the vLLM LM-head / logits vocab projection to native RDNA4 HIP kernels.

WHY
    At bs=1 decode the vocab projection is one of the largest single memory reads in the step:
    Qwen3.6-35B-A3B has vocab 248320 x hidden 2048 -> 1.02 GB of LM-head weights, 0.5 GB per rank at
    TP=2, streamed EVERY step. For a 3B-active MoE that is comparable to the entire expert working
    set. vLLM's incumbent on gfx12x is `ops.wvSplitK` (skinny GEMM), selected in
    model_executor/layers/utils.py:179-184 because n<=5 — NOT rocBLAS.

MECHANISM (supported hook, no monkeypatching)
    `LogitsProcessor` is `@PluggableLayer.register("logits_processor")`
    (model_executor/layers/logits_processor.py:18). `PluggableLayer.__new__` swaps in a subclass
    registered under the same __name__, so a `vllm.general_plugins` entry point that imports this
    module is enough. `load_general_plugins()` runs at v1/worker/worker_base.py:247, before the model
    is constructed.

    We deliberately do NOT hook `torch.ops.vllm.rocm_unquantized_gemm`: `direct_register_custom_op`
    captures the impl by reference, so a module-attribute patch is inert and a second registration
    raises. Overriding it needs a source patch, which is out of scope for a plugin.

NO TRITON/rocBLAS FALLBACK
    Every path here lands on a HIP kernel. Unsupported dtype/shape RAISES — it never silently
    defers to F.linear or wvSplitK. That is deliberate: three separate wirings in this stack were
    found INERT in 2026-07-28 testing precisely because they degraded silently.

NO CASTS, NO COPIES
    * the weight is consumed exactly as vLLM stores it -- `[vocab_per_tp, hidden]` row-major
      contiguous. No repack, no pre-permute, no transpose.
    * `hidden_states` is passed straight through; contiguity is ASSERTED, never forced with
      `.contiguous()` (that would be a hidden copy on the hot path).
    * no dtype conversion: the gemv requires `x.dtype == w.dtype`, and we raise rather than cast.
    * the only allocation is the output, plus (M>16, ragged M only) a zero-pad to the tile — the
      full-tile kernels cannot mask a ragged M edge, and the padded rows are sliced off. M<=16
      (every decode and every prefill at max_num_seqs<=16) takes the gemv and pads nothing.
"""

from __future__ import annotations

import os

import torch

from vllm.logger import init_logger
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding

logger = init_logger(__name__)

# dense_bf16_gemv is a decode-shaped kernel: it holds the M rows of x in registers, so M is bounded.
# (fp8_wmma/fp8_wmma_rocm/gemv_decode.h). Above it we use the dense_gemm family.
_GEMV_MAX_M = int(os.environ.get("VLLM_RDNA4_LMHEAD_GEMV_MAX_M", "16"))
# dense_gemm tile geometry, mirroring minisgl/layers/minv.py (the validated production dispatch).
_BLOCK_M = int(os.environ.get("VLLM_RDNA4_LMHEAD_BLOCK_M", "64"))
_BN = int(os.environ.get("VLLM_RDNA4_LMHEAD_BN", "64"))
_PIPE_M = int(os.environ.get("VLLM_RDNA4_LMHEAD_PIPE_M", "512"))
_PIPE_MI = int(os.environ.get("VLLM_RDNA4_LMHEAD_PIPE_MI", "4"))
_PIPE_PBK = int(os.environ.get("VLLM_RDNA4_LMHEAD_PIPE_PBK", "64"))


_KERNELS = None
_ENGAGED = False


def _kernels():
    """Resolve the merged HIP kernel package ONCE.

    In vllm24-hip:combined the merged fp8_wmma build is installed under the name
    `w4a8_fp8_wmma` (/opt/kernels/w4a8_fp8_wmma). Importing the standalone `fp8_wmma` as well makes
    torch.library re-register the same fakes and raises
    "the operator fp8_wmma_C::moe_bf16_gemm already has an fake impl registered".
    So bind the already-loaded package and never import the other name. This is a PACKAGING
    resolution, not a kernel fallback — both names are the same .so.
    """
    global _KERNELS
    if _KERNELS is None:
        try:
            import w4a8_fp8_wmma as k
        except ImportError:
            import fp8_wmma as k
        _KERNELS = k
    return _KERNELS


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(
            f"rdna4 lmhead: {msg}. Refusing rather than falling back to wvSplitK/rocBLAS "
            "(set VLLM_RDNA4_LMHEAD=0 to disable this plugin entirely)."
        )


def hip_vocab_projection(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """C[M, OUT] = x[M, IN] @ w[OUT, IN]^T on native HIP kernels. No fallback, no cast, no copy.

    All dense_gemm variants run the identical fixed 16-wide full-K reduction with no split-K, so
    they are bit-identical to one another per output row — the M-dependent dispatch below does not
    change results, only speed.
    """
    _require(x.dim() == 2, f"x must be 2-D, got {tuple(x.shape)}")
    _require(w.dim() == 2, f"weight must be 2-D, got {tuple(w.shape)}")
    _require(x.size(1) == w.size(1), f"K mismatch: x{tuple(x.shape)} vs w{tuple(w.shape)}")
    _require(
        w.dtype in (torch.bfloat16, torch.float16),
        f"weight dtype {w.dtype} unsupported (need bf16/fp16)",
    )
    _require(
        x.dtype == w.dtype,
        f"x dtype {x.dtype} != weight dtype {w.dtype}; a cast here would be a hidden copy on the "
        "hot path — serve with --dtype matching the checkpoint instead",
    )
    _require(x.is_contiguous(), "x is not contiguous")
    _require(w.is_contiguous(), "weight is not contiguous")

    M, IN = x.shape
    OUT = w.size(0)

    global _ENGAGED
    if not _ENGAGED:
        _ENGAGED = True
        print(f"[rdna4_vllm] lmhead ENGAGED: x{tuple(x.shape)} {x.dtype} @ w{tuple(w.shape)} "
              f"-> {'dense_bf16_gemv' if M <= _GEMV_MAX_M else 'dense_gemm'}", flush=True)

    if M <= _GEMV_MAX_M:
        # Decode + normal prefill (M == number of scheduled sequences, not tokens). Streams the
        # weight once at ~672 GB/s; nothing is padded or copied.
        return _kernels().dense_bf16_gemv(x, w)

    import dense_gemm as _dg

    _require(IN % 16 == 0, f"IN={IN} must be a multiple of 16 for the 128-bit vector loads")
    if OUT % _BN != 0:
        # Ragged OUT: only the LDS kernel can mask a partial N tile.
        return _dg.dense_gemm(x, w, _BLOCK_M, _BN)
    if M >= _PIPE_M and IN % _PIPE_PBK == 0:
        pbm = 256 if M >= 1024 else 128
        pbn = 128 if OUT % 128 == 0 else _BN
        Mp = ((M + pbm - 1) // pbm) * pbm
        xp = x if Mp == M else torch.nn.functional.pad(x, (0, 0, 0, Mp - M))
        return _dg.dense_gemm_pipe(xp, w, pbm, pbn, _PIPE_MI, _PIPE_PBK)[:M]
    Mp = ((M + _BLOCK_M - 1) // _BLOCK_M) * _BLOCK_M
    xp = x if Mp == M else torch.nn.functional.pad(x, (0, 0, 0, Mp - M))
    return _dg.dense_gemm_rd(xp, w, _BLOCK_M, _BN)[:M]


@PluggableLayer.register_oot(name="LogitsProcessor")
class Rdna4LogitsProcessor(LogitsProcessor):
    """LogitsProcessor whose vocab projection runs on HIP instead of wvSplitK.

    Only the GEMM is replaced. The TP gather (`_gather_logits`) and the vocab-padding slice are
    vLLM's, verbatim — those are correctness-critical and not a kernel concern.
    """

    def _get_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: VocabParallelEmbedding,
        embedding_bias: torch.Tensor | None,
    ) -> torch.Tensor | None:
        # tie_word_embeddings just swaps WHICH module supplies .weight; the layout is identical, so
        # this path is correct either way.
        logits = hip_vocab_projection(hidden_states, lm_head.weight)
        if embedding_bias is not None:
            logits = logits + embedding_bias
        bias = getattr(lm_head, "bias", None)
        if bias is not None:
            logits = logits + bias

        logits = self._gather_logits(logits)
        if logits is not None:
            logits = logits[..., : self.org_vocab_size]
        return logits

    def get_top_tokens(
        self,
        lm_head: VocabParallelEmbedding,
        hidden_states: torch.Tensor,
        embedding_bias: torch.Tensor | None = None,
    ):
        # Second vocab-projection site (HasTopTokens / speculator). Overridden too so a spec-decode
        # config cannot silently drop back to wvSplitK.
        return super().get_top_tokens(lm_head, hidden_states, embedding_bias)
