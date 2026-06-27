from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch
from minisgl.core import Batch, get_global_ctx

from .base import BaseAttnBackend, BaseAttnMetadata
from ._triton_unified import KVQuantMode, unified_attention

if TYPE_CHECKING:
    from minisgl.models import ModelConfig


@dataclass
class RDNA4Metadata(BaseAttnMetadata):
    cache_seqlens: torch.Tensor  # per-seq total KV length (= seqused_k)
    cu_seqlens_q: torch.Tensor  # [bs+1] cumulative query lengths
    max_seqlen_q: int
    max_seqlen_k: int
    page_table: torch.Tensor  # [bs, max_pages] page-indexed block table
    cold_prefill: bool  # prefill with no prefix-cache hit (every seq's KV == its new tokens)

    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.cu_seqlens_q[1 : 1 + bs] - 1


class TritonRDNA4Backend(BaseAttnBackend):
    """Tuned RDNA4 (gfx1201) unified prefill+decode attention, lifted from vLLM's
    ``triton_attn``. Phase 1a: bf16 KV via the engine's torch store, 2D grid, Triton-heuristic
    tuning (no 3D flash-decode / autotuner yet — those land in Phase 1b/4). cudagraph capture
    is not yet supported: run with ``--cuda-graph-max-bs 0``."""

    # Number of parallel tiled-softmax segments for the 3D flash-decode path
    # (matches vLLM's NUM_PAR_SOFTMAX_SEGMENTS default; the autotuner refines it later).
    # Env-overridable for tuning/validation (e.g. =1 collapses 3D to a single pass ~= 2D).
    NUM_PAR_SOFTMAX_SEGMENTS = int(os.environ.get("MINISGL_ATTN_SEGMENTS", "64"))

    def __init__(self, config: ModelConfig):
        ctx = get_global_ctx()
        self.config = config
        self.kvcache = ctx.kv_cache
        self.page_size = ctx.page_size
        self.scale = config.head_dim**-0.5
        # fp8 (e4m3fn) KV path: detected from the actual KV buffer dtype. Per-tensor
        # scale 1.0 (direct e4m3 cast on store; the kernel folds the descale into the
        # score/accumulator). The store cast lives in MHAKVCache.store_kv (.to(cache.dtype)).
        self.kv_is_fp8 = self.kvcache.dtype == torch.float8_e4m3fn
        if self.kv_is_fp8:
            self._kv_quant_mode = KVQuantMode.FP8_PER_TENSOR
            ones = torch.ones(1, dtype=torch.float32, device=self.kvcache.device)
            self._k_descale, self._v_descale = ones, ones
        else:
            self._kv_quant_mode = KVQuantMode.NONE
            self._k_descale = self._v_descale = None
        # 3D flash-decode segment scratch (f32), lazily sized on first forward.
        self._seq_threshold_3D = 0
        self._segm_output: torch.Tensor | None = None
        self._segm_max: torch.Tensor | None = None
        self._segm_expsum: torch.Tensor | None = None
        # --- Native HIP attention (on by default; MINISGL_ATTN_HIP=0 forces pure Triton) ---
        # Per-op hybrid, all of head_dim 64/128/256: decode -> attn_decode.flash_decode_paged (fp8
        # variant when KV is fp8); cold prefill -> attn_hip.flash_prefill; extend/paged-prefix prefill
        # -> attn_prefill_paged.flash_prefill_paged (fp8 variant when KV is fp8). 256 (Qwen3.5/3.6
        # full-attn) uses BR/BC=16 tiling. No head_dim falls back to Triton on the HIP path.
        # Hard-require: when enabled the .so must import or boot fails (the
        # user opted in to default-on, so a missing build is a hard error, not a silent fallback).
        self._attn_hip = os.environ.get("MINISGL_ATTN_HIP", "1") != "0"
        if self._attn_hip:
            import attn_decode  # noqa: F401  registers torch.ops.attn_decode.*
            import attn_hip  # noqa: F401  registers torch.ops.attn_hip.*
            import attn_prefill_paged  # noqa: F401  registers torch.ops.attn_prefill_paged.*

            self._hip_decode_op = torch.ops.attn_decode.flash_decode_paged
            self._hip_decode_fp8_op = torch.ops.attn_decode.flash_decode_paged_fp8
            self._hip_prefill_op = torch.ops.attn_hip.flash_prefill  # dense cold prefill
            self._hip_prefill_paged_op = (
                torch.ops.attn_prefill_paged.flash_prefill_paged  # paged/chunked extend prefill
            )
            self._hip_prefill_paged_fp8_op = (
                torch.ops.attn_prefill_paged.flash_prefill_paged_fp8  # fp8-KV paged extend prefill
            )
            # All three attention kernels now cover head_dim 64/128/256: decode (attn_decode), cold
            # prefill (attn_hip) and extend/paged prefill (attn_prefill_paged) — 256 (Qwen3.5/3.6
            # full-attn) uses head_dim-dependent BR/BC=16 tiling to fit the 64 KB gfx1201 LDS. So no
            # head_dim falls back to Triton on the HIP path.
            self._hip_prefill_ok = config.head_dim in (64, 128, 256)

    def _ensure_segm_scratch(self, q: torch.Tensor) -> None:
        """Allocate the persistent f32 segment scratch for the 3D flash-decode path.
        Sized once to the max decode batch (page-table rows) so it covers every batch;
        the kernel's capacity gate falls back to 2D for anything larger."""
        if self._segm_output is not None:
            return
        rows = int(get_global_ctx().page_table.shape[0])  # max_running_req + 1
        num_heads_q = q.shape[1]
        head_dim = q.shape[2]
        headdim_padded = 1 << (head_dim - 1).bit_length()
        seg = self.NUM_PAR_SOFTMAX_SEGMENTS
        dev = q.device
        self._seq_threshold_3D = rows
        self._segm_output = torch.empty(
            (rows, num_heads_q, seg, headdim_padded), dtype=torch.float32, device=dev
        )
        self._segm_max = torch.empty((rows, num_heads_q, seg), dtype=torch.float32, device=dev)
        self._segm_expsum = torch.empty((rows, num_heads_q, seg), dtype=torch.float32, device=dev)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor:
        metadata = batch.attn_metadata
        assert isinstance(metadata, RDNA4Metadata)
        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)
        # Native-HIP per-op dispatch (default on). store_kv above already persisted the new
        # tokens' K/V into the paged cache, so the decode kernel reads them back; the cold-prefill
        # kernel computes attention over the contiguous new-token K/V directly.
        if self._attn_hip:
            if metadata.max_seqlen_q == 1:
                return self._hip_decode(q, layer_id, metadata)
            if self._hip_prefill_ok:  # head_dim 64/128/256
                if metadata.cold_prefill:
                    # dense prefill over the contiguous new-token K/V (works for fp8 KV too,
                    # since it reads inline k/v, not the cache).
                    return self._hip_prefill(q, k, v, metadata)
                # extend / radix-hit prefill: paged K/V prefix + new tokens, prefix-offset causal
                # mask. fp8 variant folds the per-tensor descale (bf16 + fp8 KV both covered).
                return self._hip_prefill_paged(q, layer_id, metadata)
            # unsupported head_dim -> fall through to Triton.
        out = torch.empty_like(q)
        self._ensure_segm_scratch(q)
        # Always pass the 3D scratch + segments; the kernel's gate routes prefill
        # (max_seqlen_q>1) to the 2D grid and decode (max_seqlen_q==1) to 3D flash-decode.
        unified_attention(
            q=q,
            k=self.kvcache.k_cache(layer_id),  # (num_pages, page_size, kv_heads, head_dim)
            v=self.kvcache.v_cache(layer_id),
            out=out,
            cu_seqlens_q=metadata.cu_seqlens_q,
            max_seqlen_q=metadata.max_seqlen_q,
            seqused_k=metadata.cache_seqlens,
            max_seqlen_k=metadata.max_seqlen_k,
            softmax_scale=self.scale,
            causal=True,
            window_size=(-1, -1),  # no sliding window
            block_table=metadata.page_table,
            softcap=0.0,
            q_descale=None,
            k_descale=self._k_descale,
            v_descale=self._v_descale,
            kv_quant_mode=self._kv_quant_mode,
            seq_threshold_3D=self._seq_threshold_3D,
            num_par_softmax_segments=self.NUM_PAR_SOFTMAX_SEGMENTS,
            softmax_segm_output=self._segm_output,
            softmax_segm_max=self._segm_max,
            softmax_segm_expsum=self._segm_expsum,
        )
        return out

    def _hip_decode(
        self, q: torch.Tensor, layer_id: int, metadata: RDNA4Metadata
    ) -> torch.Tensor:
        # Paged flash-decode over the full KV cache. q is [B, Hq, D] (one token per seq).
        k_cache = self.kvcache.k_cache(layer_id)  # [num_pages, page_size, kv_heads, head_dim]
        v_cache = self.kvcache.v_cache(layer_id)
        block_table = metadata.page_table.to(torch.int32)
        ctx_lens = metadata.cache_seqlens.to(torch.int32)
        if self.kv_is_fp8:
            # fp8 (e4m3) paged KV: per-tensor descale 1.0 (store cast uses scale 1.0).
            return self._hip_decode_fp8_op(
                q, k_cache, v_cache, block_table, ctx_lens, self.scale, 1.0, 1.0, 0
            )
        return self._hip_decode_op(q, k_cache, v_cache, block_table, ctx_lens, self.scale, 0)

    def _hip_prefill(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, metadata: RDNA4Metadata
    ) -> torch.Tensor:
        # Dense causal prefill, no prefix cache (cold_prefill guarantees each seq's KV == its new
        # tokens). q is [tokens, Hq, D]; k/v arrive flat [tokens, Hk*D] -> reshape to [tokens, Hk, D]
        # (store_kv above already consumed the flat k/v). flash_prefill is single-sequence, so slice
        # the varlen batch by cu_seqlens_q and run each independently.
        D = self.config.head_dim
        k = k.view(-1, k.shape[-1] // D, D)
        v = v.view(-1, v.shape[-1] // D, D)
        cu = metadata.cu_seqlens_q.tolist()
        out = torch.empty_like(q)
        for i in range(len(cu) - 1):
            s, e = cu[i], cu[i + 1]
            if e - s <= 0:
                continue
            out[s:e] = self._hip_prefill_op(
                q[s:e].contiguous(), k[s:e].contiguous(), v[s:e].contiguous(),
                self.scale, 1, 0,  # causal=1, sliding_window=0 (matches the Triton path)
            )
        return out

    def _hip_prefill_paged(
        self, q: torch.Tensor, layer_id: int, metadata: RDNA4Metadata
    ) -> torch.Tensor:
        # Chunked / radix-hit prefill: Q = the packed varlen new tokens [total_q, Hq, D];
        # K/V read from the paged cache (prefix + new, already stored above) with a prefix-offset
        # causal mask. context_lens = full per-seq KV length (cache_seqlens); cu_seqlens_q = new
        # tokens. kv_block_stride=0 (minisgl's cache is contiguous, not vLLM's interleaved view).
        k_cache = self.kvcache.k_cache(layer_id)  # [num_pages, page_size, kv_heads, head_dim]
        v_cache = self.kvcache.v_cache(layer_id)
        block_table = metadata.page_table.to(torch.int32)
        cu_q = metadata.cu_seqlens_q.to(torch.int32)
        ctx_lens = metadata.cache_seqlens.to(torch.int32)
        q = q.contiguous()
        if self.kv_is_fp8:
            # fp8 (e4m3) paged KV: per-tensor descale 1.0 (store cast uses scale 1.0), folded
            # in the kernel. Descales sit between scale and causal in this op's signature.
            return self._hip_prefill_paged_fp8_op(
                q, k_cache, v_cache, block_table, cu_q, ctx_lens,
                self.scale, 1.0, 1.0, 1, 0, metadata.max_seqlen_q, 0,  # k/v_descale, causal, sw, kv_block_stride
            )
        return self._hip_prefill_paged_op(
            q, k_cache, v_cache, block_table, cu_q, ctx_lens,
            self.scale, 1, 0, metadata.max_seqlen_q, 0,  # causal=1, sw=0, kv_block_stride=0
        )

    def prepare_metadata(self, batch: Batch) -> None:
        # Lifted from the FlashAttention backend: the page-table slicing + cu_seqlens
        # construction is backend-agnostic (the global page table is page_size=1).
        reqs = batch.padded_reqs
        seqlens_q = [req.extend_len for req in reqs]
        seqlens_k = [req.device_len for req in reqs]
        cached_lens = [req.cached_len for req in reqs]
        max_seqlen_k = max(seqlens_k)
        max_seqlen_q = max(seqlens_q)
        CPU_KWARGS = {"device": "cpu", "dtype": torch.int32, "pin_memory": True}
        device = self.kvcache.device

        cache_seqlens = torch.tensor(seqlens_k, **CPU_KWARGS).to(device, non_blocking=True)

        cold_prefill = False
        if max_seqlen_q == 1:
            cu_seqlens_q = torch.arange(0, len(reqs) + 1, device=device, dtype=torch.int32)
        elif all(l == 0 for l in cached_lens):  # prefill, no cache hit
            cold_prefill = True
            cu_seqlens_q = torch.tensor([0] + seqlens_k, **CPU_KWARGS).cumsum_(0)
            cu_seqlens_q = cu_seqlens_q.to(device, non_blocking=True)
        else:  # extend prefill with partial cache hit
            cu_seqlens_q = torch.tensor([0] + seqlens_q, **CPU_KWARGS).cumsum_(0)
            cu_seqlens_q = cu_seqlens_q.to(device, non_blocking=True)

        page_table = get_global_ctx().page_table
        # global page table treats page_size=1; slice + rescale to page indices.
        new_page_table = torch.stack(
            [page_table[req.table_idx, : max_seqlen_k : self.page_size] for req in reqs]
        )
        if self.page_size > 1:
            new_page_table.div_(self.page_size, rounding_mode="floor")

        batch.attn_metadata = RDNA4Metadata(
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            page_table=new_page_table,
            cold_prefill=cold_prefill,
        )

    # --- cudagraph capture: not yet supported (Phase 4). Boot with --cuda-graph-max-bs 0. ---
    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        raise NotImplementedError(
            "triton_rdna4 cudagraph capture lands in Phase 4; run with --cuda-graph-max-bs 0"
        )

    def prepare_for_capture(self, batch: Batch) -> None:
        raise NotImplementedError

    def prepare_for_replay(self, batch: Batch) -> None:
        raise NotImplementedError
