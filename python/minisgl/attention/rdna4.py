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
    # Fused-TiDAR structured attention mask (C.1/C.4): [total_q, max_kv] fp32 additive bias
    # (0 allowed / -inf denied), indexed [packed_q_row, key_pos]. When set, the paged-extend kernel
    # runs with causal=0 and lets this carry the whole block structure. None for a normal serve.
    custom_mask: torch.Tensor | None = None

    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.cu_seqlens_q[1 : 1 + bs] - 1


class RDNA4Backend(BaseAttnBackend):
    """RDNA4 (gfx1201) prefill+decode attention. By DEFAULT (``MINISGL_ATTN_HIP=1``) it dispatches to
    the native HIP flash kernels (``attn_decode`` / ``attn_hip`` / ``attn_prefill_paged``) — no Triton
    kernel runs. The tuned Triton ``unified_attention`` path (lifted from vLLM's ``triton_attn``) is
    the ``MINISGL_ATTN_HIP=0`` opt-out only; hence the rename off the old ``triton_rdna4`` name. A
    head_dim the HIP prefill kernels don't cover (not 64/128/256) raises rather than silently using
    Triton. cudagraph capture is not yet supported here: run with ``--cuda-graph-max-bs 0`` (the
    ``hip`` subclass adds decode capture)."""

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
        # Persistent attention-output buffer, reused across forwards (eager paths only). See
        # _get_out_buf; grown to the largest token count seen so no per-forward torch.empty_like.
        self._out_buf: torch.Tensor | None = None
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
            # Canonical rdna4-hip-kernels packages: ops exposed as module-level callables.
            import attn_decode
            import attn_hip
            import attn_prefill_paged

            self._hip_decode_op = attn_decode.flash_decode_paged
            self._hip_decode_fp8_op = attn_decode.flash_decode_paged_fp8
            self._hip_prefill_op = attn_hip.flash_prefill  # dense cold prefill
            self._hip_prefill_paged_op = (
                attn_prefill_paged.flash_prefill_paged  # paged/chunked extend prefill
            )
            self._hip_prefill_paged_fp8_op = (
                attn_prefill_paged.flash_prefill_paged_fp8  # fp8-KV paged extend prefill
            )
            # All three attention kernels now cover head_dim 64/128/256: decode (attn_decode), cold
            # prefill (attn_hip) and extend/paged prefill (attn_prefill_paged) — 256 (Qwen3.5/3.6
            # full-attn) uses head_dim-dependent BR/BC=16 tiling to fit the 64 KB gfx1201 LDS. So no
            # head_dim falls back to Triton on the HIP path.
            self._hip_prefill_ok = config.head_dim in (64, 128, 256)

    def _get_out_buf(self, q: torch.Tensor) -> torch.Tensor:
        """Persistent [tokens, Hq, D] attention-output buffer, reused across forwards instead of a
        per-call ``torch.empty_like(q)``. Safe because each layer's attention output is consumed by
        o_proj before the next attention call (nothing retains it across two attention calls), and
        the eager paths that use it are never cudagraph-captured (decode-capture in the hip subclass
        returns the kernel-owned output directly, so it never routes here). Grows to the largest
        token count seen; returns a contiguous ``[:tokens]`` slice."""
        n = q.shape[0]
        buf = self._out_buf
        if buf is None or buf.shape[0] < n or buf.shape[1:] != q.shape[1:] or buf.dtype != q.dtype:
            self._out_buf = torch.empty((n, *q.shape[1:]), dtype=q.dtype, device=q.device)
            return self._out_buf
        return buf[:n]

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
            # Native-HIP is on but there is no HIP prefill kernel for this head_dim. Do NOT silently
            # fall through to the Triton kernel (a different code path with different numerics) — that
            # silent fallback was the misleading behaviour behind this backend's old "triton_rdna4"
            # name. Fail loud instead.
            raise RuntimeError(
                f"native-HIP attention (MINISGL_ATTN_HIP=1) has no prefill kernel for head_dim="
                f"{self.config.head_dim} (supported: 64/128/256). Set MINISGL_ATTN_HIP=0 to use the "
                f"Triton unified_attention fallback instead."
            )
        # Deliberate Triton path: reached ONLY when MINISGL_ATTN_HIP=0 (an explicit opt-out for
        # A/B / debugging), never as a silent fallback from the native-HIP path above.
        from minisgl._hip_engage import engaged
        engaged(f"attn:TRITON_unified_FALLBACK(attn_hip={self._attn_hip},hd={q.shape[-1]})")
        out = self._get_out_buf(q)  # A2 persistent buffer
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
            # fp8 (e4m3) paged KV: per-tensor descale = the calibrated store scale (1.0 if
            # MINISGL_KV_FP8_CALIBRATE=0), folded into the score/accumulator by the kernel.
            ks, vs = self.kvcache.k_scale[layer_id], self.kvcache.v_scale[layer_id]
            return self._hip_decode_fp8_op(
                q, k_cache, v_cache, block_table, ctx_lens, self.scale, ks, vs, 0
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
        out = self._get_out_buf(q)
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
        # Fused-TiDAR: a custom_mask carries the whole block structure -> run causal=0 and let the
        # kernel's mask_bias arg apply it. None on a normal serve (plain prefix-offset causal).
        custom_mask = getattr(metadata, "custom_mask", None)
        if self.kv_is_fp8:
            # fp8 (e4m3) paged KV: per-tensor descale = the calibrated store scale (A5; 1.0 if
            # calibration off), folded in the kernel. Descales sit between scale and causal.
            # DDTree / fused-TiDAR: the fp8 kernel applies the SAME additive mask_bias as the bf16
            # path (attn_prefill_paged_kernels.hip fp8 branch, "same contract as the bf16 kernel"),
            # so pass custom_mask straight through as mask_bias and run causal=0 when it is present
            # (the mask carries the whole ancestor/block structure). The per-tensor descale composes
            # cleanly: it is folded into the scores BEFORE the additive mask, so the -inf/0 mask
            # entries mask the already-descaled scores exactly as on the bf16 path.
            ks, vs = self.kvcache.k_scale[layer_id], self.kvcache.v_scale[layer_id]
            fp8_causal = 0 if custom_mask is not None else 1
            from minisgl._hip_engage import engaged
            engaged("attn_prefill_paged.flash_prefill_paged_fp8"
                    + ("(masked)" if custom_mask is not None else ""))
            return self._hip_prefill_paged_fp8_op(
                q, k_cache, v_cache, block_table, cu_q, ctx_lens,
                self.scale, ks, vs, fp8_causal, 0, metadata.max_seqlen_q, 0,  # k/v_descale, causal, sw, kv_block_stride
                custom_mask,  # mask_bias (None on a normal serve; the DDTree/TiDAR ancestor mask otherwise)
            )
        causal = 0 if custom_mask is not None else 1
        from minisgl._hip_engage import engaged
        engaged("attn_prefill_paged.flash_prefill_paged" + ("(masked)" if custom_mask is not None else ""))
        return self._hip_prefill_paged_op(
            q, k_cache, v_cache, block_table, cu_q, ctx_lens,
            self.scale, causal, 0, metadata.max_seqlen_q, 0, custom_mask,  # ..., kv_block_stride, mask_bias
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
        # global page table treats page_size=1; gather every req's row in ONE vectorized
        # advanced-index (was a per-req Python list-comp + torch.stack on the decode hot path),
        # then stride + rescale to page indices. Equivalent to
        # torch.stack([page_table[r.table_idx, :max_seqlen_k:page_size] for r in reqs]).
        table_idx = torch.tensor(
            [req.table_idx for req in reqs], dtype=torch.long, pin_memory=True
        ).to(page_table.device, non_blocking=True)
        new_page_table = page_table[table_idx, : max_seqlen_k : self.page_size]  # [bs, cols], fresh
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
            "rdna4 cudagraph capture lands in Phase 4; run with --cuda-graph-max-bs 0"
        )

    def prepare_for_capture(self, batch: Batch) -> None:
        raise NotImplementedError

    def prepare_for_replay(self, batch: Batch) -> None:
        raise NotImplementedError
