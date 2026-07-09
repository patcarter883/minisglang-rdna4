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
    # ---- SWA (sliding-window) ring-pool metadata — populated ONLY for a SWA-hybrid model. The
    # sliding layers store/read from the separate window-bounded ring pool (ctx.swa_kv_cache), not
    # the full-context main pool, so they need their own out_loc / page_table / cache_seqlens:
    #   swa_out_loc      [total_new_tokens]  ring slot per new token = table_idx*W + pos%W
    #   swa_page_table   [bs, max_win]       per-seq ring block = arange(table_idx*W, +min(seqlen,W))
    #   swa_cache_seqlens[bs]                per-seq valid key count = min(seqlen, W)
    # None for every non-SWA layer/model (the full layers + all other models use the main-pool fields).
    swa_out_loc: torch.Tensor | None = None
    swa_page_table: torch.Tensor | None = None
    swa_cache_seqlens: torch.Tensor | None = None

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
        # SWA (Laguna) sliding-window ring KV pool — set by the Engine only for a SWA-hybrid model.
        # The sliding layers route their paged KV here (window-bounded), keyed by the layer's compact
        # swa id. None for every non-SWA model (the full-context main pool serves every layer).
        self.swa_kv = getattr(ctx, "swa_kv_cache", None)
        self.swa_window = config.sliding_window or 0
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
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch,
        sliding_window: int = 0,
    ) -> torch.Tensor:
        metadata = batch.attn_metadata
        assert isinstance(metadata, RDNA4Metadata)
        # SWA (sliding-window) layer: store/read the window-bounded ring pool, not the main pool.
        if sliding_window > 0 and self.swa_kv is not None:
            return self._swa_forward(q, k, v, layer_id, metadata, sliding_window)
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
            # fused-TiDAR custom_mask is not wired on the fp8-KV path yet.
            assert custom_mask is None, "fused-TiDAR custom_mask not wired on the fp8-KV path yet"
            ks, vs = self.kvcache.k_scale[layer_id], self.kvcache.v_scale[layer_id]
            return self._hip_prefill_paged_fp8_op(
                q, k_cache, v_cache, block_table, cu_q, ctx_lens,
                self.scale, ks, vs, 1, 0, metadata.max_seqlen_q, 0,  # k/v_descale, causal, sw, kv_block_stride
            )
        causal = 0 if custom_mask is not None else 1
        from minisgl._hip_engage import engaged
        engaged("attn_prefill_paged.flash_prefill_paged" + ("(masked)" if custom_mask is not None else ""))
        return self._hip_prefill_paged_op(
            q, k_cache, v_cache, block_table, cu_q, ctx_lens,
            self.scale, causal, 0, metadata.max_seqlen_q, 0, custom_mask,  # ..., kv_block_stride, mask_bias
        )

    # ---- SWA (sliding-window) ring-pool attention (Laguna sliding layers) ----------------------
    # A sliding layer stores/reads its paged KV in the SEPARATE window-bounded ring pool (self.swa_kv)
    # instead of the full-context main pool. `layer_id` here is the layer's COMPACT swa id (position
    # among the sliding layers). Reached from forward() when sliding_window > 0. Requires the native
    # HIP ops (self._attn_hip); a SWA model must run --attention-backend hip.
    def _swa_forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int,
        metadata: RDNA4Metadata, sliding_window: int,
    ) -> torch.Tensor:
        assert self._attn_hip, (
            "SWA (sliding-window) attention requires the native-HIP backend (MINISGL_ATTN_HIP=1). "
            "The Triton unified path does not wire the SWA ring pool."
        )
        assert metadata.swa_out_loc is not None, "SWA metadata missing (is_swa_hybrid not wired?)"
        # Persist the new tokens' K/V into the ring pool at the ring slots (table_idx*W + pos%W).
        self.swa_kv.store_kv(k, v, metadata.swa_out_loc, layer_id)
        if metadata.max_seqlen_q == 1:
            return self._swa_decode(q, layer_id, metadata)
        if metadata.cold_prefill:
            return self._swa_prefill_cold(q, k, v, metadata, sliding_window)
        raise NotImplementedError(
            "SWA extend/chunked prefill is not supported: a SWA-hybrid model must run the naive "
            "prefix cache (whole-prompt cold prefill). Radix reuse across the window boundary is "
            "unsound, and the ring pool holds only the last `window` tokens."
        )

    def _swa_decode(
        self, q: torch.Tensor, layer_id: int, metadata: RDNA4Metadata
    ) -> torch.Tensor:
        k_cache = self.swa_kv.k_cache(layer_id)  # [num_swa_slots, 1, kv_heads, head_dim]
        v_cache = self.swa_kv.v_cache(layer_id)
        block_table = metadata.swa_page_table.to(torch.int32)
        ctx_lens = metadata.swa_cache_seqlens.to(torch.int32)
        # The ring block IS the window (<= W recent keys, all causal-valid for the newest query), so
        # no extra window mask is needed — sliding_window=0. Byte-identical to a full decode over a
        # <=W-length cache.
        if self.swa_kv.dtype == torch.float8_e4m3fn:
            ks, vs = self.swa_kv.k_scale[layer_id], self.swa_kv.v_scale[layer_id]
            return self._hip_decode_fp8_op(
                q, k_cache, v_cache, block_table, ctx_lens, self.scale, ks, vs, 0
            )
        return self._hip_decode_op(q, k_cache, v_cache, block_table, ctx_lens, self.scale, 0)

    def _swa_prefill_cold(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        metadata: RDNA4Metadata, window: int,
    ) -> torch.Tensor:
        # Dense cold prefill over the inline prompt K/V with a CAUSAL + WINDOW mask (the kernel's
        # native sliding_window arg). The ring store above kept the last `window` tokens for decode.
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
                self.scale, 1, window,  # causal=1, sliding_window=window
            )
        return out

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

        swa_out_loc = swa_page_table = swa_cache_seqlens = None
        if self.swa_kv is not None and self.swa_window > 0:
            swa_out_loc, swa_page_table, swa_cache_seqlens = self._build_swa_metadata(
                reqs, seqlens_q, cached_lens, device
            )

        batch.attn_metadata = RDNA4Metadata(
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            page_table=new_page_table,
            cold_prefill=cold_prefill,
            swa_out_loc=swa_out_loc,
            swa_page_table=swa_page_table,
            swa_cache_seqlens=swa_cache_seqlens,
        )

    def _build_swa_metadata(self, reqs, seqlens_q, cached_lens, device):
        """Ring-pool metadata for the sliding layers (SWA-hybrid only). Each request owns a fixed
        `window`-slot block [table_idx*W, table_idx*W + W); token at absolute position p writes slot
        table_idx*W + (p % W) (the ring). The read block for a request of length S is the first
        min(S, W) slots (when S >= W every slot holds one of the last W positions; when S < W slots
        0..S-1 hold positions 0..S-1) with cache_seqlen = min(S, W). Order within the block is
        irrelevant — each stored key already carries its RoPE at absolute position, and a decode
        query's softmax over keys is permutation-invariant."""
        W = self.swa_window
        out_slots: list[int] = []
        table_rows: list[list[int]] = []
        seqlens_win: list[int] = []
        max_win = 0
        for req, qlen, c0 in zip(reqs, seqlens_q, cached_lens):
            t = req.table_idx
            base = t * W
            # new tokens this batch: absolute positions [c0, c0+qlen) -> ring slots
            out_slots.extend(base + (p % W) for p in range(c0, c0 + qlen))
            S = c0 + qlen  # device_len
            cnt = min(S, W)
            seqlens_win.append(cnt)
            table_rows.append([base + s for s in range(cnt)])
            max_win = max(max_win, cnt)
        CPU = {"device": "cpu", "dtype": torch.int32, "pin_memory": True}
        swa_out_loc = torch.tensor(out_slots, **CPU).to(device, non_blocking=True)
        # rectangular [bs, max_win] page table (short rows padded with 0 = the NULL slot; the kernel
        # bounds reads by swa_cache_seqlens so the pad is never attended).
        padded = [row + [0] * (max_win - len(row)) for row in table_rows]
        swa_page_table = torch.tensor(padded, **CPU).to(device, non_blocking=True)
        swa_cache_seqlens = torch.tensor(seqlens_win, **CPU).to(device, non_blocking=True)
        return swa_out_loc, swa_page_table, swa_cache_seqlens

    # --- cudagraph capture: not yet supported (Phase 4). Boot with --cuda-graph-max-bs 0. ---
    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        raise NotImplementedError(
            "rdna4 cudagraph capture lands in Phase 4; run with --cuda-graph-max-bs 0"
        )

    def prepare_for_capture(self, batch: Batch) -> None:
        raise NotImplementedError

    def prepare_for_replay(self, batch: Batch) -> None:
        raise NotImplementedError
