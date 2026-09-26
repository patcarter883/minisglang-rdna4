from __future__ import annotations

import os
from typing import Dict

import torch
import torch.nn.functional as F
from minisgl.core import get_global_ctx
from minisgl.distributed import DistributedCommunicator, get_tp_info
from minisgl.utils import div_ceil, nvtx_annotate

from .base import BaseOP

# LM-head logits path. The vocab GEMV [M,K] x [vocab_tp,K] is a pure memory-bound streaming shape; the
# WMMA-tiled `minv` GEMM (built for large M) stalls at M=1 (~124 GB/s on the 35B LM head). The shared
# gemv_decode_core<Bf16GemvLoader> streams the weight fully coalesced at ~HBM (~673 GB/s measured) AND
# is M-invariant BY CONSTRUCTION (per-(row,col) independent fp32 dot in a fixed K-order → logits for a
# position are bit-identical regardless of batch M). That is exactly minv's guarantee, so it replaces
# minv on the LM head for EVERY case — decode (M=1) AND spec-verify (M>1) — with no losslessness gate.
# Only M beyond the core's MMAX cap (16) or non-bf16/fp16 weights fall back to minv.
_LMHEAD_GEMV_MMAX = 16  # gemv_decode_core MMAX cap; LM-head verify M (spec K+1) sits well under this
_lmhead_gemv_fn = None
_lmhead_gemv_probed = False


def _get_lmhead_gemv():
    """Lazily resolve fp8_wmma.dense_bf16_gemv (None if the kernel package is unavailable)."""
    global _lmhead_gemv_fn, _lmhead_gemv_probed
    if not _lmhead_gemv_probed:
        _lmhead_gemv_probed = True
        try:
            from fp8_wmma import dense_bf16_gemv

            _lmhead_gemv_fn = dense_bf16_gemv
        except Exception:
            _lmhead_gemv_fn = None
    return _lmhead_gemv_fn


def _lm_head_linear(x: torch.Tensor, weight: torch.Tensor,
                    bias: torch.Tensor | None) -> torch.Tensor:
    """Full-vocab logits x @ weight^T (+bias), stored fp32.

    The GEMV core always accumulated each dot in fp32; storing the result in bf16 quantized the
    finished logits to the bf16 ULP — an exact 0.125 grid at |logit| 16..32, with bit-identical
    TIES between unrelated tokens (measured on the served LM-head top-20; llama.cpp on the same
    weights is continuous). Ties collapse <0.125-nat distinctions at sampling time, so the store
    is now fp32 end-to-end: the M-invariant decode GEMV takes fp32_out (a store POLICY on the
    shared core), and rows > MMAX take one half-precision GEMM that accumulates AND stores in fp32
    (`out_dtype`). minv stays out of this path — its bf16 store is the exact grid this removes.

    Rows > MMAX are NOT rare: every block-diffusion canvas step scores 256 rows, and every spec
    verify with bs x qlen > 16 lands here too. They used to cast each 8192-row weight chunk to fp32
    and run an fp32 GEMM — 53.5 ms per canvas step against 1.71 ms for this (RX 9070 XT, M=256,
    K=2816, N=131072), which was a third of the whole DiffusionGemma step. Same products (a half x
    half product is exact in fp32), same fp32 accumulation; only summation order differs."""
    gemv = _get_lmhead_gemv()
    if (gemv is not None
            and weight.dtype in (torch.bfloat16, torch.float16)
            and x.dtype == weight.dtype
            and x.dim() == 2 and x.shape[0] <= _LMHEAD_GEMV_MMAX):
        try:
            out = gemv(x.contiguous(), weight, True)
        except TypeError:
            # Older kernel package without the fp32_out policy: keep serving (bf16 store) rather
            # than dying — the grid is a fidelity deficit, not an outage.
            out = gemv(x.contiguous(), weight)
        if bias is not None:
            out = out + bias
        return out
    if weight.dtype in (torch.bfloat16, torch.float16) and x.dtype == weight.dtype:
        out = torch.mm(x, weight.t(), out_dtype=torch.float32)
    else:  # mixed/fp32 inputs: cast the weight a chunk at a time, never whole (1.5 GB at 262k vocab)
        xf = x.float()
        out = torch.empty(x.shape[0], weight.shape[0], dtype=torch.float32, device=x.device)
        for i in range(0, weight.shape[0], 8192):
            out[:, i : i + 8192] = xf @ weight[i : i + 8192].float().t()
    if bias is not None:
        out += bias
    return out


class VocabParallelEmbedding(BaseOP):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        tp_info = get_tp_info()
        tp_rank = tp_info.rank
        self.tp_size = tp_info.size
        self.num_embeddings = num_embeddings
        self.num_embeddings_tp = div_ceil(num_embeddings, self.tp_size)
        start_idx = self.num_embeddings_tp * tp_rank
        finish_idx = min(start_idx + self.num_embeddings_tp, num_embeddings)
        self.vocab_range = (start_idx, finish_idx - start_idx)
        self.weight = torch.empty(self.num_embeddings_tp, embedding_dim)
        self._comm = DistributedCommunicator()

    @nvtx_annotate("Embedding")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # torch port of the former `indexing` .cu op: vocab-parallel gather with
        # out-of-range masking, then all-reduce across TP ranks.
        if self.tp_size > 1:
            start, count = self.vocab_range
            mask = (x >= start) & (x < start + count)
            local_idx = (x - start).clamp_(0, count - 1)
            y = self.weight[local_idx]  # fresh gather copy; zero out-of-range rows in place
            # equivalent to torch.where(mask, y, 0) but without the full zeros_like temporary.
            y.mul_(mask.unsqueeze(-1))
            return self._comm.all_reduce(y)
        return self.weight[x]


class ParallelLMHead(VocabParallelEmbedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
        tie_word_embeddings: bool = False,
        tied_embedding: VocabParallelEmbedding | None = None,
    ):
        super().__init__(num_embeddings, embedding_dim)
        self.bias = torch.empty(self.num_embeddings_tp) if bias else None
        self.tied_embedding = tied_embedding
        assert (tied_embedding is not None) == tie_word_embeddings

    def load_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        *,
        prefix: str = "",
        _internal: bool = False,
        missing_ok: bool = False,
    ) -> None:
        if not self.tied_embedding:
            return super().load_state_dict(
                state_dict, prefix=prefix, _internal=_internal, missing_ok=missing_ok
            )
        else:
            # pop the lm_head.weights and lm_head.bias if they exist
            possible_weight = f"{prefix}.weight"
            possible_bias = f"{prefix}.bias"
            if possible_weight in state_dict:
                state_dict.pop(possible_weight)
            if possible_bias in state_dict:
                state_dict.pop(possible_bias)

    def state_dict(
        self,
        *,
        prefix: str = "",
        result: Dict[str, torch.Tensor] | None = None,
    ) -> Dict[str, torch.Tensor]:
        if not self.tied_embedding:
            return super().state_dict(prefix=prefix, result=result)
        return {} if result is None else result

    def logits_all_rows(self, x: torch.Tensor) -> torch.Tensor:
        """Full-vocab logits over ALL input rows (no prefill last-token reduction, no batch context).

        Used by the MTP draft head, which runs OUTSIDE the normal batch forward (every row scored)
        and MUST produce identical full-vocab logits on every TP rank — calling F.linear over the
        local vocab shard would leave each rank with a different half, so the per-rank argmax drafts
        would diverge and desync the verify batch (collective deadlock). Mirrors the all_gather in
        ``forward`` but keeps every row."""
        module = self.tied_embedding or self
        # M-invariant so verify logits match decode: the bf16 decode GEMV when it applies, else minv.
        logits = _lm_head_linear(x, module.weight, self.bias)  # [rows, vocab//tp]
        if self.tp_size == 1:
            return logits
        input_shape = logits.shape
        output_tensor = self._comm.all_gather(logits)
        output_tensor = output_tensor.view((self.tp_size,) + input_shape)
        output_tensor = output_tensor.permute(1, 0, 2).contiguous()
        output_tensor = output_tensor.reshape(input_shape[:1] + (self.tp_size * input_shape[1],))
        return output_tensor[:, : self.num_embeddings]

    def logits_local_shard(self, x: torch.Tensor) -> torch.Tensor:
        """Logits over THIS RANK'S vocab columns only — `logits_all_rows` without the all_gather.

        For a caller that reduces over the vocabulary rather than indexing into it, gathering is pure
        cost. The block-diffusion canvas is the case: it scores 256 rows x 262144 columns EVERY
        denoising step and every consumer (logsumexp, softmax, entropy, argmax, multinomial, and the
        soft embedding's `probs @ E`) is a reduction that decomposes over disjoint column blocks. The
        gather moved 67 MB/rank and was followed by a `permute().contiguous()` over the whole
        [256, 262144] result to undo the rank-major interleave — 23.2 ms of a 190 ms step, to
        materialise on both ranks a tensor neither of them indexes.

        Returns `[rows, count]` where `count` is the VALID width from `vocab_range`, not the padded
        shard width. Those differ on the last rank whenever tp_size does not divide the vocabulary,
        and the padded columns are uninitialised weight rows — `logits_all_rows` drops them after the
        gather (`[:, :num_embeddings]`), so a shard-consuming caller must drop them here or it
        silently samples tokens that do not exist."""
        module = self.tied_embedding or self
        _, count = self.vocab_range
        logits = _lm_head_linear(x, module.weight, self.bias)
        return logits if logits.shape[1] == count else logits[:, :count]

    def argmax_all_rows(self, x: torch.Tensor) -> torch.Tensor:
        """Greedy token per row, identical on every TP rank, without gathering the logits: each rank
        reduces its own vocab shard and only (max, index) pairs are exchanged. Ties resolve to the
        lowest vocab id, as a full-row argmax would (shards are in ascending vocab order)."""
        local = self.logits_local_shard(x)                        # [rows, count] fp32
        val, idx = local.max(dim=-1)
        if self.tp_size == 1:
            return idx
        start, _ = self.vocab_range
        pair = torch.stack([val, (idx + start).to(torch.float32)], dim=-1)  # ids < 2^24: exact
        allp = self._comm.all_gather(pair.contiguous()).view(self.tp_size, -1, 2)
        best = allp[..., 0].argmax(dim=0)                         # first max = lowest rank
        return allp[..., 1].gather(0, best[None]).squeeze(0).to(torch.int64)

    @nvtx_annotate("LMHead")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        batch = ctx.batch
        bs = batch.size
        if batch.is_prefill:
            indices = batch.attn_metadata.get_last_indices(bs)
            x = x[indices].contiguous()
            del indices

        module = self.tied_embedding or self
        # M-invariant: verify(M=K+1) logits == decode(M=1). bf16 decode GEMV when it applies, else minv.
        logits = _lm_head_linear(x, module.weight, self.bias)
        if self.tp_size == 1:
            return logits
        input_shape = logits.shape
        output_tensor = self._comm.all_gather(logits)

        # Fast path keyed on the number of scored ROWS, not the request count. For plain decode
        # rows == bs == 1; for a speculative-decode VERIFY batch one request contributes K+1 rows,
        # so gating on bs==1 would collapse them to a single logit row (target-length mismatch).
        if input_shape[0] == 1:
            return output_tensor.view(1, -1)[:, : self.num_embeddings]

        output_tensor = output_tensor.view((self.tp_size,) + input_shape)
        output_tensor = output_tensor.permute(1, 0, 2).contiguous()
        output_tensor = output_tensor.reshape(input_shape[:1] + (self.tp_size * input_shape[1],))
        return output_tensor[:, : self.num_embeddings]
