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

# Flash BC tile for head_dim 64/128 (attn_kernels_hip.hip kBC): the online-softmax reduces in BC-wide
# blocks. Front-padding the SWA extend buffer by (cached_len - Wp) % BC lands the first new-token row
# at the same residue mod BC as a cold prefill => identical block grouping => bit-identical.
_SWA_BC_ALIGN = 32

# head_dims the native-HIP attention kernels cover. All three (attn_decode / attn_hip /
# attn_prefill_paged) switch on exactly this set and TORCH_CHECK anything else, so one tuple gates
# every op. It is tested PER CALL, not once per model: a split-head_dim model runs two geometries
# through ONE backend instance (Gemma4: 256-wide sliding layers and 512-wide full-attention layers).
# 512 is Gemma4's full-attention geometry; the three packages carry it via the same templated cores
# (the prefill pair needs the D-split warp ladder to stay spill-free at that width). Keep this tuple
# in step with the kernels' own switch arms — it exists to turn an unsupported width into a message
# that names the geometry, not to be a second, quieter source of truth about what is built.
_HIP_HEAD_DIMS = (64, 128, 256, 512)


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
    # Block-diffusion CANVAS batch: attend every query over every key inside `cache_seqlens`, with no
    # causal mask and no window, on BOTH layer geometries. Distinct from `custom_mask`, which also
    # forces causal=0 but carries an additive per-cell bias: a canvas needs no bias at all, and
    # materializing an all-zero one would cost a [canvas, cur_len+canvas] fp32 tensor per layer per
    # step to say nothing (measured bit-identical to passing None —
    # tools/canvas_attention_probe.py). False on every autoregressive batch.
    bidirectional: bool = False
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
    # SWA CAPTURED spec-verify (paged-from-ring): the K+1 verify shape is made cudagraph-capturable by
    # attending the ring pool through the SAME paged-extend kernel as the full layers (not the eager
    # dense _gather+_swa_prefill_extend, whose per-seq .clone()/torch.cat allocate fresh + host-sync the
    # window length -> not capture-safe). These two fields carry the STATIC per-seq ring block table and
    # context length the captured verify reads; both are populated ONLY by the verify capture/replay
    # prep (HIPAttnBackend._fill_swa_verify_static). None on every eager path (which keeps swa_table_idx
    # + the dense extend, byte-unchanged):
    #   swa_verify_page_table   [bs, W+qlen]  cache-index j -> ring slot; indices [0,Wp)=window (ascending
    #                                         absolute pos), [Wp,Wp+qlen)=the K+1 new-token slots.
    #   swa_verify_cache_seqlens[bs]          context_len = Wp+qlen (bounds the kernel's key reads; the
    #                                         padded tail past it is ignored, same trick as page_table).
    # The kernel's window test (qpos-kpos)>=W uses cache index j as kpos and prefix_len+r as qpos; the
    # constant (cached_len-Wp) offset cancels in the difference, so the paged read reproduces the EXACT
    # sliding-window mask of the dense extend (proven greedy-identical). swa_out_loc (above) carries the
    # K+1 new-token store slots.
    swa_verify_page_table: torch.Tensor | None = None
    swa_verify_cache_seqlens: torch.Tensor | None = None
    # SWA EXTEND (chunked-continuation OR cross-request radix reuse): per-seq (batch order) table_idx
    # of the sliding-window ring block, so the extend can gather each seq's window [cached_len-W,
    # cached_len) directly from ITS ring (a prior chunk wrote it; a cross-request reuse is seeded by
    # the scheduler's _restore_swa_states before the forward). None => no SWA extend supported for this
    # forward (naive single-shot cold only). See rdna4.py::_swa_prefill_extend.
    swa_table_idx: List[int] | None = None
    # Lazy per-forward host-sync caches. A metadata object is built ONCE per forward and shared
    # across every attention layer, so the cold-prefill kernels' `.tolist()` slicing would re-sync
    # the same tensor ~40 times (once per layer). Memoize the first sync here; all later layers of
    # the SAME forward reuse it. (A fresh RDNA4Metadata is built each prepare_metadata, so the cache
    # never leaks across forwards; cu_seqlens_q / cache_seqlens are never mutated between layers.)
    _cu_seqlens_q_list: List[int] | None = None
    _cache_seqlens_list: List[int] | None = None

    def cu_seqlens_q_list(self) -> List[int]:
        """`cu_seqlens_q.tolist()`, computed once per forward and cached (host sync)."""
        lst = self._cu_seqlens_q_list
        if lst is None:
            lst = self._cu_seqlens_q_list = self.cu_seqlens_q.tolist()
        return lst

    def cache_seqlens_list(self) -> List[int]:
        """`cache_seqlens.tolist()`, computed once per forward and cached (host sync)."""
        lst = self._cache_seqlens_list
        if lst is None:
            lst = self._cache_seqlens_list = self.cache_seqlens.tolist()
        return lst

    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.cu_seqlens_q[1 : 1 + bs] - 1


class RDNA4Backend(BaseAttnBackend):
    """RDNA4 (gfx1201) prefill+decode attention. By DEFAULT (``MINISGL_ATTN_HIP=1``) it dispatches to
    the native HIP flash kernels (``attn_decode`` / ``attn_hip`` / ``attn_prefill_paged``) — no Triton
    kernel runs. The tuned Triton ``unified_attention`` path (lifted from vLLM's ``triton_attn``) is
    the ``MINISGL_ATTN_HIP=0`` opt-out only; hence the rename off the old ``triton_rdna4`` name. A
    head_dim the HIP kernels don't cover (not ``_HIP_HEAD_DIMS``) raises rather than silently using
    Triton — checked per call, since a split-head_dim model routes two geometries through one
    instance. cudagraph capture is not yet supported here: run with ``--cuda-graph-max-bs 0`` (the
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
        # Per-seq ring STRIDE (>= window). = window + spec block when spec is enabled on a SWA model,
        # so a K+1 spec-verify's speculative block occupies slots disjoint from the live window (see
        # engine.py). Without spec it equals the window (Track A's ring, byte-identical). The window
        # SIZE stays self.swa_window; only slot addressing (store/gather/decode-read) strides by this.
        self.swa_ring_stride = getattr(ctx, "swa_ring_stride", self.swa_window)
        self.page_size = ctx.page_size
        # Softmax temperature is resolved PER CALL from the query's own head_dim (see
        # _softmax_scale), never cached once per model: a split-head_dim model runs both geometries
        # through this one backend instance, so a single config.head_dim**-0.5 would silently apply
        # the full layers' temperature to the sliding ones — wrong logits on 25 of 30 layers with no
        # error anywhere. `attn_softmax_scale` overrides the 1/sqrt(d) form outright when the config
        # sets one (Gemma4: 1.0 — the temperature lives in its learned k_norm instead). A uniform
        # model resolves exactly the value this line used to cache.
        self._scale_override = config.attn_softmax_scale
        self._scale_by_head_dim: dict[int, float] = {}
        # fp8 (e4m3fn) KV path: detected from the actual KV buffer dtype. The native-HIP ops read
        # the pool's [num_kv_heads] descale ROW per layer (per-head capable) and fold it into the
        # score/accumulator. The Triton unified FALLBACK below can only take ONE scalar, so its
        # descale is resolved per layer in _triton_descale() and it refuses a non-uniform table
        # rather than silently dequantizing every head with head 0's scale.
        # This backend is constructed AFTER Engine installs the scales (engine.py orders KV pool ->
        # scale install -> attention backend), so the table it reads here is already final.
        self.kv_is_fp8 = self.kvcache.dtype == torch.float8_e4m3fn
        if self.kv_is_fp8:
            self._kv_quant_mode = KVQuantMode.FP8_PER_TENSOR
            kd, vd = self.kvcache.k_descale, self.kvcache.v_descale
            self._descale_uniform = bool(
                (kd == kd[:, :1]).all().item() and (vd == vd[:, :1]).all().item()
            )
        else:
            self._kv_quant_mode = KVQuantMode.NONE
            self._descale_uniform = True
        # Persistent attention-output buffer, reused across forwards (eager paths only). See
        # _get_out_buf; grown to the largest token count seen so no per-forward torch.empty_like.
        self._out_buf: torch.Tensor | None = None
        # attn_prefill_paged's split-K policy input; see the `split_ctx` property. Resolved lazily
        # because the global page table is allocated after the backend is constructed.
        self._split_ctx: int | None = None
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
            # All three attention kernels cover head_dim 64/128/256: decode (attn_decode), cold
            # prefill (attn_hip) and extend/paged prefill (attn_prefill_paged) — 256 (Qwen3.5/3.6
            # full-attn) uses head_dim-dependent BR/BC=16 tiling to fit the 64 KB gfx1201 LDS. The
            # coverage check is _HIP_HEAD_DIMS, applied per call in forward() (a split-head_dim model
            # has no single answer). No covered head_dim falls back to Triton on the HIP path.

    # ---- the split-K policy input, and why the ENGINE owns it ---------------------------------
    # `attn_prefill_paged` chooses between its single-pass kernel and split-K + reduce from a context
    # bound. It used to READ that bound off the block table's row width, and the row width is not the
    # same number on the two paths that must agree: the eager metadata allocates the row at the
    # batch's true context (`page_table[idx, :max_seqlen_k:page_size]`), while every capture family
    # allocates a STATIC row sized from max_seq_len. So a captured graph ran split-K + reduce where
    # the eager forward it was captured from ran the single-pass kernel — two different kernels on
    # the same inputs, measured on the DiffusionGemma canvas at max|delta| = 8.324e+00 against a
    # hidden state whose own max is 41.44, and reproduced in isolation for the K+1 spec-verify and
    # chunked-prefill shapes (attn_prefill_paged/tests/test_split_width_invariance.py).
    #
    # The kernel no longer infers it. `split_ctx` is a required op argument with NO default, and it
    # comes from the CAPACITY OF THE POOL BEING READ — a serve-lifetime constant that is therefore
    # trivially the same at capture, at replay, and on every eager forward:
    #
    #   main paged pool -> `self.split_ctx`, the global page table's token width (aligned_max_seq_len)
    #   SWA ring pool   -> `window + max_seqlen_q`, the ring row's own capacity
    #
    # THE CONSEQUENCE, STATED BECAUSE IT IS A REAL TRADE. A captured graph bakes its launch
    # configuration, so the split decision CANNOT depend on the running context length — only on
    # numbers fixed for the serve. Context length therefore no longer informs it, and the only
    # discriminator left is SHAPE (`base_grid` vs MINISGL_ATTN_PREFILL_FILL_CTAS). A thin-grid call
    # over a short context now splits where the old eager path would have gone single-pass. That is
    # the honest cost of making the two paths agree; the alternative — letting each path pick for
    # itself — is the bug.
    @property
    def split_ctx(self) -> int:
        """Context bound for the MAIN paged pool: the global page table's token width. Identical on
        the eager and captured paths by construction (there is only one global page table)."""
        sc = self._split_ctx
        if sc is None:
            sc = self._split_ctx = int(get_global_ctx().page_table.shape[1])
            # Say it once, in the serve log. There is no gate for the spec-verify capture families
            # (only the canvas has one), so this line is what makes the split-K policy input
            # auditable on a model whose graphs nobody has diffed.
            from minisgl.utils import init_logger
            init_logger(__name__).info_rank0(
                f"[attn] attn_prefill_paged split_ctx={sc} (global page-table token width); the "
                "SAME value is passed on the eager and captured paths, and the kernel no longer "
                "infers it from the block-table row width"
            )
        return sc

    def _softmax_scale(self, q: torch.Tensor) -> float:
        """Softmax temperature for THIS call, taken from the query's actual head_dim (q is
        ``[tokens | bs, Hq, D]`` on every path here). Per call because one backend instance serves
        both geometries of a split-head_dim model."""
        override = self._scale_override
        if override is not None:
            return override
        head_dim = q.shape[-1]
        scale = self._scale_by_head_dim.get(head_dim)
        if scale is None:
            scale = self._scale_by_head_dim[head_dim] = float(head_dim) ** -0.5
        return scale

    def _no_hip_kernel(self, head_dim: int) -> None:
        """Raise for a head_dim the native-HIP kernels do not cover. Called only off the failing
        branch, so the gate itself stays a bare tuple test on the hot path.

        Deliberately does NOT fall through to the Triton unified kernel: that is a different code
        path with different numerics, and silently taking it when a kernel is missing is exactly the
        behaviour this backend's old "triton_rdna4" name papered over. Opting into Triton has to be
        an explicit MINISGL_ATTN_HIP=0."""
        note = ""
        if head_dim in _HIP_HEAD_DIMS:
            # The width IS in the supported set, so the kernels this process actually loaded are
            # older than the engine — the classic symptom of an image whose /opt/kernels predates
            # the source, which otherwise surfaces as a bare TORCH_CHECK from inside the .so.
            note = (
                f" head_dim={head_dim} IS in this engine's supported set, so the LOADED kernel "
                "package is older than the engine — rebuild /opt/kernels (or mount a package built "
                "in this same image; a .so from another image will not load at all)."
            )
        raise RuntimeError(
            f"native-HIP attention (MINISGL_ATTN_HIP=1) has no kernel for head_dim={head_dim} "
            f"(attn_decode / attn_hip / attn_prefill_paged all cover "
            f"{'/'.join(str(d) for d in _HIP_HEAD_DIMS)})." + note + " Set MINISGL_ATTN_HIP=0 to "
            "opt into the Triton unified_attention fallback instead."
        )

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
            # Gate on THIS query's head_dim, not the model's: a split-head_dim model reaches here
            # with 256 on its sliding layers and 512 on its full ones, and only one of those has a
            # kernel. Covers decode as well as prefill — all three ops share _HIP_HEAD_DIMS.
            if q.shape[-1] not in _HIP_HEAD_DIMS:
                self._no_hip_kernel(q.shape[-1])
            if metadata.max_seqlen_q == 1:
                return self._hip_decode(q, layer_id, metadata)
            if metadata.cold_prefill:
                # dense prefill over the contiguous new-token K/V (works for fp8 KV too,
                # since it reads inline k/v, not the cache).
                return self._hip_prefill(q, k, v, metadata)
            # extend / radix-hit prefill: paged K/V prefix + new tokens, prefix-offset causal
            # mask. fp8 variant folds the per-tensor descale (bf16 + fp8 KV both covered).
            return self._hip_prefill_paged(q, layer_id, metadata)
        # Deliberate Triton path: reached ONLY when MINISGL_ATTN_HIP=0 (an explicit opt-out for
        # A/B / debugging), never as a silent fallback from the native-HIP path above.
        from minisgl._hip_engage import engaged
        engaged(f"attn:TRITON_unified_FALLBACK(attn_hip={self._attn_hip},hd={q.shape[-1]})")
        kdsc, vdsc = self._triton_descale(layer_id)
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
            softmax_scale=self._softmax_scale(q),
            causal=True,
            window_size=(-1, -1),  # no sliding window
            block_table=metadata.page_table,
            softcap=0.0,
            q_descale=None,
            k_descale=kdsc,
            v_descale=vdsc,
            kv_quant_mode=self._kv_quant_mode,
            seq_threshold_3D=self._seq_threshold_3D,
            num_par_softmax_segments=self.NUM_PAR_SOFTMAX_SEGMENTS,
            softmax_segm_output=self._segm_output,
            softmax_segm_max=self._segm_max,
            softmax_segm_expsum=self._segm_expsum,
        )
        return out

    def _triton_descale(self, layer_id: int):
        """(k_descale, v_descale) for the Triton unified fallback, or (None, None) on a bf16 cache.

        unified_attention's FP8_PER_TENSOR mode does a single `tl.load(k_scale)`, so it can only take
        ONE scalar for the whole layer. Hand it element 0 of the layer's row — correct for a
        per-tensor scale (checkpoint kv_cache_scheme is `strategy: tensor`, so every head shares it)
        and REFUSE a genuinely per-head table rather than dequantize heads 1..H-1 with head 0's
        scale. Per-head fp8 is a native-HIP-backend feature; this is the MINISGL_ATTN_HIP=0 path."""
        if not self.kv_is_fp8:
            return None, None
        if not self._descale_uniform:
            raise NotImplementedError(
                "PER-HEAD fp8-KV descale on the Triton unified attention fallback. Triton's "
                "FP8_PER_TENSOR mode takes one scalar per layer, so a per-head table cannot be "
                "applied there. Serve with the native-HIP attention backend (the default; "
                "MINISGL_ATTN_HIP=1), or use a per-tensor scale source."
            )
        return self.kvcache.k_descale[layer_id][:1], self.kvcache.v_descale[layer_id][:1]

    def _hip_decode(
        self, q: torch.Tensor, layer_id: int, metadata: RDNA4Metadata
    ) -> torch.Tensor:
        # Paged flash-decode over the full KV cache. q is [B, Hq, D] (one token per seq).
        k_cache = self.kvcache.k_cache(layer_id)  # [num_pages, page_size, kv_heads, head_dim]
        v_cache = self.kvcache.v_cache(layer_id)
        block_table = metadata.page_table.to(torch.int32)
        ctx_lens = metadata.cache_seqlens.to(torch.int32)
        scale = self._softmax_scale(q)
        if self.kv_is_fp8:
            # fp8 (e4m3) paged KV: per-tensor descale = the calibrated store scale (1.0 if
            # MINISGL_KV_FP8_CALIBRATE=0), folded into the score/accumulator by the kernel.
            ks, vs = self.kvcache.k_descale[layer_id], self.kvcache.v_descale[layer_id]  # persistent device tensors (graph-safe)
            return self._hip_decode_fp8_op(
                q, k_cache, v_cache, block_table, ctx_lens, scale, ks, vs, 0
            )
        return self._hip_decode_op(q, k_cache, v_cache, block_table, ctx_lens, scale, 0)

    def _hip_prefill(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, metadata: RDNA4Metadata
    ) -> torch.Tensor:
        # Dense causal prefill, no prefix cache (cold_prefill guarantees each seq's KV == its new
        # tokens). q is [tokens, Hq, D]; k/v arrive flat [tokens, Hk*D] -> reshape to [tokens, Hk, D]
        # (store_kv above already consumed the flat k/v). flash_prefill is single-sequence, so slice
        # the varlen batch by cu_seqlens_q and run each independently.
        # D comes from THIS call's q, not config.head_dim: GQA varies the head COUNT, never the head
        # width, so k/v share q's D — while a split-head_dim model's two layer types do not share a
        # model-wide D, and reshaping 256-wide k/v by 512 is a wrong-shape view, not an error.
        D = q.shape[-1]
        k = k.view(-1, k.shape[-1] // D, D)
        v = v.view(-1, v.shape[-1] // D, D)
        cu = metadata.cu_seqlens_q_list()  # memoized once per forward (was per-layer .tolist())
        scale = self._softmax_scale(q)
        out = self._get_out_buf(q)
        for i in range(len(cu) - 1):
            s, e = cu[i], cu[i + 1]
            if e - s <= 0:
                continue
            out[s:e] = self._hip_prefill_op(
                q[s:e].contiguous(), k[s:e].contiguous(), v[s:e].contiguous(),
                scale, 1, 0,  # causal=1, sliding_window=0 (matches the Triton path)
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
        scale = self._softmax_scale(q)
        # Two independent reasons to drop the prefix-offset causal mask, and they must not be
        # conflated. Fused-TiDAR sets a `custom_mask` that CARRIES the block structure, so causal=0
        # plus an additive bias. A block-diffusion CANVAS wants no mask at all: every one of its
        # `canvas_length` queries attends every key in cache_seqlens. Passing an all-zero bias would
        # be arithmetically identical (measured: bit-identical, tools/canvas_attention_probe.py) but
        # would allocate a [canvas, cur_len+canvas] fp32 tensor per layer per denoising step to
        # express nothing.
        custom_mask = getattr(metadata, "custom_mask", None)
        non_causal = custom_mask is not None or metadata.bidirectional
        if self.kv_is_fp8:
            # fp8 (e4m3) paged KV: per-tensor descale = the calibrated store scale (A5; 1.0 if
            # calibration off), folded in the kernel. Descales sit between scale and causal.
            # DDTree / fused-TiDAR: the fp8 kernel applies the SAME additive mask_bias as the bf16
            # path (attn_prefill_paged_kernels.hip fp8 branch, "same contract as the bf16 kernel"),
            # so pass custom_mask straight through as mask_bias and run causal=0 when it is present
            # (the mask carries the whole ancestor/block structure). The per-tensor descale composes
            # cleanly: it is folded into the scores BEFORE the additive mask, so the -inf/0 mask
            # entries mask the already-descaled scores exactly as on the bf16 path.
            ks, vs = self.kvcache.k_descale[layer_id], self.kvcache.v_descale[layer_id]  # persistent device tensors (graph-safe)
            fp8_causal = 0 if non_causal else 1
            from minisgl._hip_engage import engaged
            engaged("attn_prefill_paged.flash_prefill_paged_fp8"
                    + ("(masked)" if custom_mask is not None
                       else "(canvas)" if metadata.bidirectional else ""))
            return self._hip_prefill_paged_fp8_op(
                q, k_cache, v_cache, block_table, cu_q, ctx_lens,
                scale, ks, vs, fp8_causal, 0, metadata.max_seqlen_q,
                self.split_ctx, 0,  # split-K policy input (see `split_ctx`), kv_block_stride
                custom_mask,  # mask_bias (None on a normal serve; the DDTree/TiDAR ancestor mask otherwise)
            )
        causal = 0 if non_causal else 1
        from minisgl._hip_engage import engaged
        engaged("attn_prefill_paged.flash_prefill_paged"
                + ("(masked)" if custom_mask is not None
                   else "(canvas)" if metadata.bidirectional else ""))
        return self._hip_prefill_paged_op(
            q, k_cache, v_cache, block_table, cu_q, ctx_lens,
            scale, causal, 0, metadata.max_seqlen_q,
            self.split_ctx, 0, custom_mask,  # split_ctx, kv_block_stride, mask_bias
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
        # Same per-call kernel-coverage gate as forward(); a sliding layer's head_dim is its own.
        if q.shape[-1] not in _HIP_HEAD_DIMS:
            self._no_hip_kernel(q.shape[-1])
        if metadata.max_seqlen_q == 1:
            self.swa_kv.store_kv(k, v, metadata.swa_out_loc, layer_id)
            return self._swa_decode(q, layer_id, metadata)
        if metadata.cold_prefill:
            self.swa_kv.store_kv(k, v, metadata.swa_out_loc, layer_id)
            return self._swa_prefill_cold(q, k, v, metadata, sliding_window)
        # SWA CAPTURED spec-verify (paged-from-ring). When the verify capture prep populated the static
        # ring block table, store the K+1 new tokens into the ring (disjoint speculative slots — ring
        # stride = W+K+1) then attend [window | new] straight from the ring through the SAME paged-extend
        # kernel the full layers use. Fully static-buffer / capture-safe (no per-seq clone/cat/host-sync);
        # greedy-identical to the eager dense extend below (the paged sliding-window mask is exact — see
        # RDNA4Metadata.swa_verify_page_table). None on every eager path, which keeps the dense extend.
        if metadata.swa_verify_page_table is not None:
            self.swa_kv.store_kv(k, v, metadata.swa_out_loc, layer_id)
            return self._swa_prefill_paged(q, layer_id, metadata, sliding_window)
        # SWA EXTEND — a chunked-continuation OR a cross-request radix reuse. Each seq's boundary window
        # [cached_len-W, cached_len) is already in ITS OWN ring block: a prior prefill chunk wrote it, or
        # (cross-request reuse) the scheduler's _restore_swa_states seeded it from a page-aligned
        # snapshot before this forward. GATHER that window from the ring FIRST (before store_kv overwrites
        # it with the new chunk), then attend [pad | window | new] under the SAME causal+SWA kernel as a
        # cold prefill — PROVEN bit-identical with a BC front-pad (tools/swa_prefix_extend_validate.py).
        if metadata.swa_table_idx is None:
            raise NotImplementedError(
                "SWA extend/chunked prefill reached without ring metadata (swa_table_idx is None). "
                "A SWA-hybrid model needs the SWA extend metadata wired (is_swa_hybrid path)."
            )
        windows = self._gather_swa_windows(layer_id, metadata, sliding_window, k.dtype)
        self.swa_kv.store_kv(k, v, metadata.swa_out_loc, layer_id)  # persist new tokens for later decode
        return self._swa_prefill_extend(q, k, v, metadata, sliding_window, windows)

    def _gather_swa_windows(
        self, layer_id: int, metadata: RDNA4Metadata, window: int, out_dtype: torch.dtype
    ):
        """Per-seq boundary window (batch order), gathered from THIS layer's ring BEFORE store_kv. Entry
        is None for a cold seq (cached_len==0) or (pad, k_win, v_win) with k/v_win [Wp, Hk, D] in
        ascending absolute position (Wp=min(cached_len,W)). Reads the ring at slots table_idx*R + p%R
        (R = ring stride >= W; disjoint from a prior verify's speculative block so the gather is clean)."""
        W = window
        R = self.swa_ring_stride  # per-seq ring stride (== W without spec; W + spec block under spec)
        cu = metadata.cu_seqlens_q_list()
        dev_lens = metadata.cache_seqlens_list()  # device_len per seq
        tidx = metadata.swa_table_idx
        k_ring = self.swa_kv.k_cache(layer_id).view(-1, *self.swa_kv.k_cache(layer_id).shape[2:])  # [slots,Hk,D]
        v_ring = self.swa_kv.v_cache(layer_id).view(-1, *self.swa_kv.v_cache(layer_id).shape[2:])
        dev = k_ring.device
        out = []
        for i in range(len(cu) - 1):
            qlen = cu[i + 1] - cu[i]
            cached_len = dev_lens[i] - qlen
            if qlen <= 0 or cached_len <= 0:
                out.append(None)
                continue
            Wp = min(cached_len, W)
            pos = torch.arange(cached_len - Wp, cached_len, device=dev, dtype=torch.long)
            slots = tidx[i] * R + (pos % R)
            pad = (cached_len - Wp) % _SWA_BC_ALIGN  # BC front-pad -> flash block grouping == cold
            k_win, v_win = k_ring[slots].clone(), v_ring[slots].clone()
            if k_win.dtype == torch.float8_e4m3fn:
                # DEQUANTIZE here, where the descale row for THIS layer is in scope. This is the one
                # fp8 cache read in the engine that is not folded into a kernel's descale argument —
                # _swa_prefill_extend cats the window with inline bf16 K/V and runs the BF16 flash
                # prefill over it, so the scale has to be applied in python. It used to be a bare
                # `.to(bf16)` justified by "the descale is 1.0", which held only while nothing ever
                # calibrated one; with real scales installed a bare cast yields k/descale (~448x too
                # large) and the model emits UNK spam. Per-head row [Hkv] broadcasts over [Wp,Hkv,D].
                kd = self.swa_kv.k_descale[layer_id].view(1, -1, 1)
                vd = self.swa_kv.v_descale[layer_id].view(1, -1, 1)
                k_win = (k_win.float() * kd).to(out_dtype)
                v_win = (v_win.float() * vd).to(out_dtype)
            out.append((pad, k_win, v_win))
        return out

    def _swa_prefill_extend(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        metadata: RDNA4Metadata, window: int, windows: list,
    ) -> torch.Tensor:
        """Attend each seq's new chunk over [pad zeros | ring window | new chunk] with the kernel's
        causal+sliding_window mask, dropping the front (pad+Wp) rows. Contiguous ascending absolute
        position => same non-masked key SET+values as a cold prefill; the `pad` zero rows (all window-
        masked, value-irrelevant) align the flash block grouping to cold => BIT-IDENTICAL. `windows`
        (from _gather_swa_windows) is per-seq None|(pad, k_win, v_win); k/v are the inline new bf16 K/V."""
        D = q.shape[-1]  # the SLIDING layers' head_dim, which need not be the model-wide one
        k = k.view(-1, k.shape[-1] // D, D)
        v = v.view(-1, v.shape[-1] // D, D)
        cu = metadata.cu_seqlens_q_list()
        scale = self._softmax_scale(q)
        out = self._get_out_buf(q)
        for i in range(len(cu) - 1):
            s, e = cu[i], cu[i + 1]
            if e - s <= 0:
                continue
            win = windows[i]
            if win is None:  # cold seq in a mixed batch: plain windowed prefill over its own tokens
                out[s:e] = self._hip_prefill_op(
                    q[s:e].contiguous(), k[s:e].contiguous(), v[s:e].contiguous(),
                    scale, 1, window,
                )
                continue
            pad, k_win, v_win = win  # [Wp, Hk, D]
            # NOTE: an fp8 ring window is already DEQUANTIZED to the compute dtype by
            # _gather_swa_windows (it owns layer_id, hence the descale row), so k_win/v_win are
            # always the same dtype as the inline new K/V by the time they get here.
            Wp, Hk = k_win.shape[0], k_win.shape[1]
            front = pad + Wp
            parts_k = ([k_win.new_zeros((pad, Hk, D))] if pad else []) + [k_win, k[s:e]]
            parts_v = ([v_win.new_zeros((pad, Hk, D))] if pad else []) + [v_win, v[s:e]]
            k_ext = torch.cat(parts_k, dim=0).contiguous()
            v_ext = torch.cat(parts_v, dim=0).contiguous()
            q_ext = torch.cat([q.new_zeros((front, q.shape[1], D)), q[s:e]], dim=0).contiguous()
            out_ext = self._hip_prefill_op(q_ext, k_ext, v_ext, scale, 1, window)
            out[s:e] = out_ext[front:]
        return out

    def _swa_prefill_paged(
        self, q: torch.Tensor, layer_id: int, metadata: RDNA4Metadata, window: int
    ) -> torch.Tensor:
        """Multi-query SWA attention read straight from the ring pool through the paged-extend kernel.

        Serves two shapes off the same rows. (1) CAPTURED spec-verify: each seq's K+1 new tokens over
        its ring window + those new tokens, causal=1 + sliding_window=W. (2) BLOCK-DIFFUSION CANVAS:
        each seq's canvas_length tokens over the same [window | new] rows, causal=0 + sliding_window=0
        — see the `canvas` branch below. The ring block table (metadata.swa_verify_page_table) lays out
        cache index j -> ring slot as [window(Wp) | new(qlen)]; context_len (swa_verify_cache_seqlens) =
        Wp+qlen bounds the read. This is the capture-safe analogue of _swa_prefill_extend: no per-seq
        clone/cat and no host-sync of the window length — the same STATIC-buffer paged path the full
        layers' verify uses (_hip_prefill_paged), just on the ring pool and with the window mask. The new
        tokens were persisted to the ring by store_kv (disjoint speculative slots) before this call, so the
        paged read sees them. bf16 ring reads inline; an fp8 ring folds the per-tensor descale (#40)."""
        k_cache = self.swa_kv.k_cache(layer_id)  # [num_swa_slots, 1, kv_heads, head_dim] (page_size=1)
        v_cache = self.swa_kv.v_cache(layer_id)
        block_table = metadata.swa_verify_page_table.to(torch.int32)  # int32 static buf -> no-op cast
        cu_q = metadata.cu_seqlens_q.to(torch.int32)
        ctx_lens = metadata.swa_verify_cache_seqlens.to(torch.int32)
        q = q.contiguous()
        scale = self._softmax_scale(q)
        # A CANVAS reads the same [Wp | new] ring rows but with NO mask: bidirectional over the
        # window AND over the whole canvas. `sliding_window` must go to 0 as well as `causal`,
        # because the kernel's window test is independent of the causal one — leaving it at W would
        # re-impose a symmetric +/-W band across the canvas, which is a DIFFERENT model (the window
        # in this architecture is a property of what the encoder cache RETAINED, never a mask over
        # the canvas; measured rel_fro 0.60 apart in tools/canvas_attention_probe.py).
        canvas = metadata.bidirectional
        causal, sw = (0, 0) if canvas else (1, window)
        # Split-K policy input for the RING pool (see `split_ctx`): a ring row is [<=window kept
        # slots | max_seqlen_q new ones], so that sum is its capacity. `window` is the layer's
        # config and `max_seqlen_q` is the graph's fixed width, so both paths compute the same int.
        ring_split_ctx = int(window) + int(metadata.max_seqlen_q)
        tag = "(swa-canvas)" if canvas else "(swa-verify)"
        from minisgl._hip_engage import engaged
        if self.swa_kv.dtype == torch.float8_e4m3fn:
            ks, vs = self.swa_kv.k_descale[layer_id], self.swa_kv.v_descale[layer_id]  # persistent device tensors (graph-safe)
            engaged("attn_prefill_paged.flash_prefill_paged_fp8" + tag)
            return self._hip_prefill_paged_fp8_op(
                q, k_cache, v_cache, block_table, cu_q, ctx_lens,
                scale, ks, vs, causal, sw, metadata.max_seqlen_q, ring_split_ctx, 0, None,
            )
        engaged("attn_prefill_paged.flash_prefill_paged" + tag)
        return self._hip_prefill_paged_op(
            q, k_cache, v_cache, block_table, cu_q, ctx_lens,
            scale, causal, sw, metadata.max_seqlen_q, ring_split_ctx, 0, None,
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
        scale = self._softmax_scale(q)
        if self.swa_kv.dtype == torch.float8_e4m3fn:
            ks, vs = self.swa_kv.k_descale[layer_id], self.swa_kv.v_descale[layer_id]  # persistent device tensors (graph-safe)
            return self._hip_decode_fp8_op(
                q, k_cache, v_cache, block_table, ctx_lens, scale, ks, vs, 0
            )
        return self._hip_decode_op(q, k_cache, v_cache, block_table, ctx_lens, scale, 0)

    def _swa_prefill_cold(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        metadata: RDNA4Metadata, window: int,
    ) -> torch.Tensor:
        # Dense cold prefill over the inline prompt K/V with a CAUSAL + WINDOW mask (the kernel's
        # native sliding_window arg). The ring store above kept the last `window` tokens for decode.
        # D is the SLIDING layer's own head_dim. It used to be config.head_dim, which is the FULL
        # layers' geometry on a split-head_dim model — this path would then have reshaped 256-wide
        # k/v as 512-wide and attended garbage.
        D = q.shape[-1]
        k = k.view(-1, k.shape[-1] // D, D)
        v = v.view(-1, v.shape[-1] // D, D)
        cu = metadata.cu_seqlens_q_list()  # memoized once per forward (was per-layer .tolist())
        scale = self._softmax_scale(q)
        out = self._get_out_buf(q)
        for i in range(len(cu) - 1):
            s, e = cu[i], cu[i + 1]
            if e - s <= 0:
                continue
            out[s:e] = self._hip_prefill_op(
                q[s:e].contiguous(), k[s:e].contiguous(), v[s:e].contiguous(),
                scale, 1, window,  # causal=1, sliding_window=window
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

        swa_out_loc = swa_page_table = swa_cache_seqlens = swa_table_idx = None
        swa_canvas_page_table = swa_canvas_cache_seqlens = None
        if self.swa_kv is not None and self.swa_window > 0:
            if batch.canvas:
                # A canvas reads [window | canvas] in ONE row, which `_build_swa_metadata`'s
                # cnt = min(S, W) cannot express (it caps the row at W and would drop the canvas
                # keys the queries must see). Same ring, different row shape.
                swa_out_loc, swa_canvas_page_table, swa_canvas_cache_seqlens = (
                    self._build_swa_canvas_metadata(reqs, seqlens_q, cached_lens, device)
                )
            else:
                swa_out_loc, swa_page_table, swa_cache_seqlens = self._build_swa_metadata(
                    reqs, seqlens_q, cached_lens, device
                )
            swa_table_idx = [req.table_idx for req in reqs]  # ring block per seq (SWA extend gather)

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
            swa_table_idx=swa_table_idx,
            # A canvas rides the SAME ring fields the captured spec-verify uses, so _swa_forward's
            # existing `swa_verify_page_table is not None` dispatch routes it with no new branch —
            # the rows differ, the read path does not.
            swa_verify_page_table=swa_canvas_page_table,
            swa_verify_cache_seqlens=swa_canvas_cache_seqlens,
            bidirectional=batch.canvas,
        )

    def _build_swa_canvas_metadata(self, reqs, seqlens_q, cached_lens, device):
        """Ring rows for a BLOCK-DIFFUSION canvas step: `[Wp window keys | qlen canvas keys]` per
        request, `context_len = Wp + qlen`, read with causal=0 / sliding_window=0.

        Two things separate this from `_build_swa_metadata`, and both are load-bearing.

        1. THE ROW IS NOT CAPPED AT W. The read count there is `cnt = min(S, W)`, which for a canvas
           at absolute positions [cur_len, cur_len+L) would return the last W positions and silently
           drop either window keys or canvas keys depending on cur_len. A canvas query must see BOTH
           the retained window and every one of its L siblings, so the row is the concatenation and
           the length is Wp + L.
        2. THE WINDOW IS THE PREFIX'S, NOT THE QUERY'S. Wp = min(cur_len, W) is computed from the
           CACHED length, not from cur_len + L: every canvas position sees the SAME window, the one
           the encoder left behind. A window that slid per canvas position would be a different
           model, and a plausible-looking one.

        This is also why the ring stride has to exceed W (see `_swa_ring_block` in engine.py): at
        stride W the canvas position cur_len+j lands on the slot holding prefix position
        cur_len+j-W, i.e. inside the window it is about to read — measured at 256 of 256 canvas
        slots colliding (tools/canvas_attention_probe.py)."""
        W = self.swa_window
        R = self.swa_ring_stride
        assert R > W, (
            f"block-diffusion canvas needs a widened SWA ring (stride {R} <= window {W}); at "
            f"stride == window every canvas slot aliases a live window slot and the decoder "
            f"overwrites the prefix it must attend to"
        )
        out_slots: list[int] = []
        rows: list[list[int]] = []
        ctx_lens: list[int] = []
        max_row = 0
        for req, qlen, c0 in zip(reqs, seqlens_q, cached_lens):
            base = req.table_idx * R
            canvas_slots = [base + (p % R) for p in range(c0, c0 + qlen)]
            out_slots.extend(canvas_slots)
            Wp = min(c0, W)
            row = [base + (p % R) for p in range(c0 - Wp, c0)] + canvas_slots
            rows.append(row)
            ctx_lens.append(Wp + qlen)
            max_row = max(max_row, len(row))
        CPU = {"device": "cpu", "dtype": torch.int32, "pin_memory": True}
        swa_out_loc = torch.tensor(out_slots, **CPU).to(device, non_blocking=True)
        # Rectangular [bs, max_row]; short rows pad with slot 0 (the NULL slot). The kernel bounds
        # its reads by context_len, so the pad is never attended — same trick as the main page table.
        padded = [row + [0] * (max_row - len(row)) for row in rows]
        page_table = torch.tensor(padded, **CPU).to(device, non_blocking=True)
        cache_seqlens = torch.tensor(ctx_lens, **CPU).to(device, non_blocking=True)
        return swa_out_loc, page_table, cache_seqlens

    def _build_swa_metadata(self, reqs, seqlens_q, cached_lens, device):
        """Ring-pool metadata for the sliding layers (SWA-hybrid only). Each request owns a fixed
        `stride`-slot block [table_idx*R, table_idx*R + R) (R = ring stride >= window W); token at
        absolute position p writes slot table_idx*R + (p % R) (the ring). The read block for a request
        of length S is the last min(S, W) positions [S-cnt, S), addressed POSITION-BASED (slot p%R) so a
        widened ring (R>W, spec) reads only the live window and never a stale speculative-block slot;
        cache_seqlen = min(S, W). Order within the block is irrelevant — each stored key already carries
        its RoPE at absolute position, and a decode query's softmax over keys is permutation-invariant.
        When R==W (no spec) the position-based read is the SAME slot SET as the old first-cnt read."""
        W = self.swa_window
        R = self.swa_ring_stride
        out_slots: list[int] = []
        table_rows: list[list[int]] = []
        seqlens_win: list[int] = []
        max_win = 0
        for req, qlen, c0 in zip(reqs, seqlens_q, cached_lens):
            t = req.table_idx
            base = t * R
            # new tokens this batch: absolute positions [c0, c0+qlen) -> ring slots
            out_slots.extend(base + (p % R) for p in range(c0, c0 + qlen))
            S = c0 + qlen  # device_len
            cnt = min(S, W)
            seqlens_win.append(cnt)
            if R == W:
                # Track A ring (no spec): first cnt slots — byte-identical to the validated path.
                table_rows.append([base + s for s in range(cnt)])
            else:
                # Widened ring (spec): the last cnt positions [S-cnt, S) at their ring slots p%R, so
                # rejected speculative tokens (in disjoint slots) are never read.
                table_rows.append([base + (p % R) for p in range(S - cnt, S)])
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
