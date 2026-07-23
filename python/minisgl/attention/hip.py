"""Fully Triton-free attention for gfx1201 (RDNA4): native HIP rocwmma flash-PREFILL
(``torch.ops.attn_hip.flash_prefill``) + native HIP paged flash-DECODE
(``torch.ops.attn_decode.flash_decode_paged``). No Triton kernel is invoked on either path.

Subclasses ``RDNA4Backend`` ONLY to reuse its ``__init__`` (kvcache / scale / page_size /
fp8 detection) and ``prepare_metadata`` (which builds the ``RDNA4Metadata`` page-table +
cu_seqlens_q + cache_seqlens that both kernels consume). ``forward`` is fully overridden.

The two kernel packages are framework-agnostic ``torch.ops`` extensions shared with vllm-gfx1201
(prefill: the ``attn_hip`` worktree; paged decode: the ``attn_decode`` worktree). They must be
importable (on PYTHONPATH) when this backend is selected.

Constraints (v0 — eager, validated on dense head_dim 64/128):
  * PREFILL has two native-HIP paths, dispatched on metadata.cold_prefill:
      - COLD (no prefix-cache hit): attn_hip.flash_prefill, single-sequence + contiguous, so the
        varlen batch is sliced per sequence by cu_seqlens_q (each seq's keys are its own tokens).
      - EXTEND (radix-hit / chunked prefill, cached_len > 0): attn_prefill_paged.flash_prefill_paged
        reads the paged K/V prefix + new tokens with a prefix-offset causal mask (fp8 variant folds
        the per-tensor descale). Both are Triton-free; metadata is built by the inherited
        prepare_metadata. head_dim 256 (Qwen3.5/3.6 full-attn) uses BR/BC=16 tiling.
  * DECODE cudagraph capture is wired; PREFILL (both paths) runs eager.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, List

import torch
from minisgl.core import get_global_ctx

from .rdna4 import RDNA4Backend, RDNA4Metadata

if TYPE_CHECKING:
    from minisgl.core import Batch
    from minisgl.models import ModelConfig


class HIPAttnBackend(RDNA4Backend):
    def __init__(self, config: "ModelConfig") -> None:
        super().__init__(config)
        # Import here (not at module top) so the kernel packages are only required when the
        # "hip" backend is actually selected. Canonical rdna4-hip-kernels packages expose their ops
        # as module-level callables (the ops register under a build-unique torch.ops.<pkg>_C ns).
        import attn_decode
        import attn_hip

        self._prefill = attn_hip.flash_prefill
        self._decode = attn_decode.flash_decode_paged
        self._decode_fp8 = attn_decode.flash_decode_paged_fp8

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: "Batch",
        sliding_window: int = 0,
    ) -> torch.Tensor:
        metadata = batch.attn_metadata
        assert isinstance(metadata, RDNA4Metadata)
        # SWA (sliding-window) layer: store/read the window-bounded ring pool (ctx.swa_kv_cache) with
        # a windowed mask, instead of the full-context main pool. layer_id is the compact swa id.
        if sliding_window > 0 and self.swa_kv is not None:
            return self._swa_forward(q, k, v, layer_id, metadata, sliding_window)
        # Persist the current tokens' K/V into the paged cache (decode + extend prefill read it back).
        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)
        if batch.is_prefill:
            if metadata.cold_prefill:
                # No prefix-cache hit: each seq's KV == its own new tokens.
                # fp8 KV: attend over the fp8-quantized K/V we JUST stored (via the paged fp8 kernel,
                # cached_len=0 -> full causal), NOT the inline bf16. Otherwise a cold/naive prefill
                # attends full-precision bf16 while a prefix-cached request attends the fp8 the cache
                # holds -> the two diverge (observed: GSM8K 6/8 answers differ under fp8 KV). Routing
                # cold through the same paged kernel makes cached == naive bit-for-bit under fp8. bf16
                # KV keeps the dense contiguous flash_prefill (bf16 store is lossless, cold == extend
                # already: GSM8K 0/8), and it is faster for the cold shape.
                if self.kv_is_fp8:
                    return self._hip_prefill_paged(q, layer_id, metadata)
                # dense contiguous prefill over the inline K/V (attn_hip.flash_prefill).
                return self._forward_prefill(q, k, v, metadata)
            # Radix-hit / chunked extend: Q = new tokens, K/V = paged prefix + new (just stored),
            # prefix-offset causal. Native HIP attn_prefill_paged kernel (Triton-free; fp8 variant
            # folds the per-tensor descale). The helper is inherited from RDNA4Backend but
            # calls ONLY attn_prefill_paged.* — no Triton kernel runs on this path.
            return self._hip_prefill_paged(q, layer_id, metadata)
        # Decode phase: a spec-decode VERIFY batch carries max_seqlen_q = K+1 > 1 (multi-query
        # against the paged prefix), so it takes the extend kernel — exactly like a radix-hit
        # prefill. Plain decode (one token/seq, max_seqlen_q == 1) uses the single-token decode
        # kernel. (Dispatch on max_seqlen_q, not is_prefill, so verify routes correctly.)
        if metadata.max_seqlen_q > 1:
            return self._hip_prefill_paged(q, layer_id, metadata)
        return self._forward_decode(q, layer_id, metadata)

    def _forward_prefill(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, metadata: RDNA4Metadata
    ) -> torch.Tensor:
        # At the backend boundary q is [tokens, Hq, D] (the attention layer view's it) but k/v are
        # still flat [tokens, Hk*D] -> reshape to [tokens, Hk, D] for the kernel. (store_kv above
        # consumed the flat k/v, same as the Triton path.)
        D = self.config.head_dim
        k = k.view(-1, k.shape[-1] // D, D)
        v = v.view(-1, v.shape[-1] // D, D)
        # flash_prefill is single-sequence; minisgl batches varlen sequences -> slice by
        # cu_seqlens_q and run each independently. This is the COLD path only (forward() routes
        # prefix-cache hits to _hip_prefill_paged), so each seq's keys are exactly its current
        # tokens. Assert that invariant: cache_seqlens == query lengths (a violation means a hit
        # leaked past the cold_prefill dispatch).
        cu = metadata.cu_seqlens_q_list()  # memoized once per forward (was per-layer .tolist())
        klen = metadata.cache_seqlens_list()  # memoized once per forward (was per-layer .tolist())
        from minisgl._hip_engage import engaged
        engaged("attn_hip.flash_prefill")
        out = self._get_out_buf(q)  # persistent buffer (A2, inherited); eager prefill, never captured

        for i in range(len(cu) - 1):
            s, e = cu[i], cu[i + 1]
            qlen = e - s
            if qlen <= 0:
                continue
            assert klen[i] == qlen, (
                f"cold HIP prefill got a prefix-cache hit (seq {i}: kv_len={klen[i]} != "
                f"q_len={qlen}); cold_prefill dispatch in forward() should have routed this to "
                "the paged extend kernel."
            )
            out[s:e] = self._prefill(
                q[s:e].contiguous(), k[s:e].contiguous(), v[s:e].contiguous(),
                self.scale, 1, 0,  # causal=1, sliding_window=0 (matches the Triton path)
            )
        return out

    def _forward_decode(
        self, q: torch.Tensor, layer_id: int, metadata: RDNA4Metadata
    ) -> torch.Tensor:
        k_cache = self.kvcache.k_cache(layer_id)  # [num_pages, page_size, kv_heads, head_dim]
        v_cache = self.kvcache.v_cache(layer_id)
        block_table = metadata.page_table.to(torch.int32)
        ctx_lens = metadata.cache_seqlens.to(torch.int32)
        from minisgl._hip_engage import engaged
        if self.kv_is_fp8:
            # fp8 (e4m3) paged KV: per-tensor descale = the calibrated store scale (A5; 1.0 if
            # calibration off), folded in the kernel. Pass the PERSISTENT device descale tensors
            # (0-dim views) the canonical op now requires — stable address, graph-safe.
            ks, vs = self.kvcache.k_descale[layer_id], self.kvcache.v_descale[layer_id]
            engaged("attn_decode.flash_decode_paged_fp8")
            return self._decode_fp8(q, k_cache, v_cache, block_table, ctx_lens, self.scale, ks, vs, 0)
        engaged("attn_decode.flash_decode_paged")
        return self._decode(q, k_cache, v_cache, block_table, ctx_lens, self.scale, 0)

    # ---- cudagraph capture (DECODE only) -----------------------------------------------------
    # Decode is one token/seq, so the only per-step varying metadata the kernel reads is
    # cache_seqlens (KV length, +1 each step) and the page table (a row can gain a page). We hold
    # both in STATIC buffers the captured graph reads; prepare_for_replay refreshes them in place
    # before g.replay(). cu_seqlens_q is a fixed arange (all q-lengths are 1). The decode kernel
    # bounds its reads by ctx_lens, so a fixed max-width page table is fine (stale tail ignored).
    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        dev = self.kvcache.device
        self._cap_max_bs = max(bs_list)
        self._cap_max_pages = (max_seq_len + self.page_size - 1) // self.page_size
        self._cap_cache_seqlens = torch.ones(self._cap_max_bs, dtype=torch.int32, device=dev)
        self._cap_page_table = torch.zeros(
            self._cap_max_bs, self._cap_max_pages, dtype=torch.int32, device=dev
        )
        self._cap_cu_q = torch.arange(self._cap_max_bs + 1, dtype=torch.int32, device=dev)
        # ---- SWA (Laguna sliding layers) ring-pool decode metadata (persistent, refreshed in place) --
        # A SWA-hybrid model's 30 sliding layers route decode through _swa_forward -> _swa_decode, which
        # reads swa_out_loc (ring slot the new token WROTE), swa_page_table (the <=W-slot read window),
        # and swa_cache_seqlens (min(len,W)) via the shared RDNA4Metadata. Those change every step (the
        # ring slot = table_idx*R + pos%R advances, the window slides), so — exactly like GDN's
        # state_indices — they live in PERSISTENT buffers whose CONTENTS _fill_swa_decode_static refreshes
        # before each g.replay(); the captured store_kv/decode kernels read them through fixed pointers.
        # Width is the window W (the read window is capped at W; the decode kernel bounds reads by
        # swa_cache_seqlens, so stale columns past each row's cnt are ignored — same trick as page_table).
        if self.swa_kv is not None and self.swa_window > 0:
            W = self.swa_window
            self._cap_swa_out_loc = torch.zeros(self._cap_max_bs, dtype=torch.int32, device=dev)
            self._cap_swa_page_table = torch.zeros(self._cap_max_bs, W, dtype=torch.int32, device=dev)
            self._cap_swa_cache_seqlens = torch.ones(self._cap_max_bs, dtype=torch.int32, device=dev)
            # Cached device column index [0..W) for the VECTORIZED ring page-table build (avoids a
            # per-step arange). Default-on; MINISGL_SWA_METADATA_VEC=0 falls back to the eager
            # _build_swa_metadata python path (kept for the A/B timing that proved the win).
            self._swa_cols = torch.arange(W, dtype=torch.int64, device=dev)
            self._swa_vec = os.environ.get("MINISGL_SWA_METADATA_VEC", "1") != "0"

    def _decode_metadata_static(self, bs: int) -> RDNA4Metadata:
        md = RDNA4Metadata(
            cache_seqlens=self._cap_cache_seqlens[:bs],
            cu_seqlens_q=self._cap_cu_q[: bs + 1],
            max_seqlen_q=1,
            max_seqlen_k=self._cap_max_pages * self.page_size,
            page_table=self._cap_page_table[:bs],
            cold_prefill=False,
        )
        # SWA-hybrid: attach the persistent ring-pool decode metadata the sliding layers read.
        if self.swa_kv is not None and self.swa_window > 0:
            md.swa_out_loc = self._cap_swa_out_loc[:bs]
            md.swa_page_table = self._cap_swa_page_table[:bs]
            md.swa_cache_seqlens = self._cap_swa_cache_seqlens[:bs]
        return md

    def _fill_swa_decode_static(self, batch: "Batch") -> None:
        """Refresh the persistent SWA ring-pool decode buffers from `batch.padded_reqs` (eager, OUTSIDE
        the graph). VECTORIZED by default (MINISGL_SWA_METADATA_VEC=1): the ring math is a closed form,
        so build it with O(bs) host reads + on-device arithmetic instead of a per-step O(bs*W) python
        loop. For decode (qlen=1) the new token sits at absolute position c0=device_len-1:
            out_loc      = table_idx*R + c0 % R                          (ring slot it writes)
            cache_seqlen = min(device_len, W)                            (= cnt, the live window)
            page_table[:, j] = table_idx*R + ( (R==W) ? j : (S-cnt+j) % R )
        Byte-identical to the eager `_build_swa_metadata`: for every column j < cnt the slot matches
        (R==W: base+j == the first-cnt read; R>W: base+(S-cnt+j)%R == the last-cnt position read); the
        columns j>=cnt differ (formula vs 0-pad) but are NEVER attended — the decode kernel bounds reads
        by cache_seqlens=cnt. MINISGL_SWA_METADATA_VEC=0 restores the eager python path (kept for the
        A/B host-cost comparison). All device tensors here are built OUTSIDE the captured graph; only the
        copy_ into the persistent buffers matters for pointer stability."""
        reqs = batch.padded_reqs
        bs = len(reqs)
        dev = self.kvcache.device
        if not self._swa_vec:
            # Eager reference path (per-step python list-build + H2D) — the pre-vectorization behaviour.
            seqlens_q = [r.extend_len for r in reqs]
            cached_lens = [r.cached_len for r in reqs]
            out_loc, page_table, cache_seqlens = self._build_swa_metadata(
                reqs, seqlens_q, cached_lens, dev
            )
            self._cap_swa_out_loc[:bs].copy_(out_loc)
            self._cap_swa_cache_seqlens[:bs].copy_(cache_seqlens)
            self._cap_swa_page_table[:bs, : page_table.shape[1]].copy_(page_table)
            return
        W = self.swa_window
        R = self.swa_ring_stride
        # O(bs) host reads -> one pinned H2D each (bs is tiny: <= max_graph_bs).
        tbl = torch.tensor([r.table_idx for r in reqs], dtype=torch.int64, pin_memory=True).to(
            dev, non_blocking=True)
        S = torch.tensor([r.device_len for r in reqs], dtype=torch.int64, pin_memory=True).to(
            dev, non_blocking=True)
        base = tbl * R                                   # [bs] ring block start
        c0 = S - 1                                       # [bs] new-token absolute position (qlen=1)
        cnt = torch.clamp(S, max=W)                      # [bs] live-window length = min(S, W)
        cols = self._swa_cols                            # [W] cached device arange
        if R == W:
            slots = base[:, None] + cols[None, :]                                    # [bs, W]
        else:
            slots = base[:, None] + torch.remainder((S - cnt)[:, None] + cols[None, :], R)
        self._cap_swa_out_loc[:bs].copy_((base + torch.remainder(c0, R)).to(torch.int32))
        self._cap_swa_cache_seqlens[:bs].copy_(cnt.to(torch.int32))
        self._cap_swa_page_table[:bs, :W].copy_(slots.to(torch.int32))

    def _fill_decode_static(self, batch: "Batch") -> None:
        """Refresh the static decode buffers from `batch.padded_reqs` (real rows + dummy padding).
        Runs eager, OUTSIDE the graph; it writes the exact tensors the captured kernel reads."""
        reqs = batch.padded_reqs
        bs = len(reqs)
        dev = self.kvcache.device
        seqlens_k = [req.device_len for req in reqs]
        self._cap_cache_seqlens[:bs].copy_(
            torch.tensor(seqlens_k, dtype=torch.int32, device=dev)
        )
        gpt = get_global_ctx().page_table  # global page_size=1 table
        # Vectorized gather (was a per-req Python loop): pull all rows' full max-width strided page
        # ids at once. The captured decode kernel bounds its reads by ctx_lens, so writing the whole
        # width (incl. the per-seq stale tail beyond npages) is equivalent to the old per-row
        # [:npages] copy — the tail is ignored either way.
        table_idx = torch.tensor([req.table_idx for req in reqs], dtype=torch.long, device=gpt.device)
        rows = gpt[table_idx, : self._cap_max_pages * self.page_size : self.page_size]  # [bs, ncols]
        if self.page_size > 1:
            rows = torch.div(rows, self.page_size, rounding_mode="floor")
        ncols = rows.shape[1]
        self._cap_page_table[:bs, :ncols].copy_(rows.to(torch.int32))

    def prepare_for_capture(self, batch: "Batch") -> None:
        self._fill_decode_static(batch)
        if self.swa_kv is not None and self.swa_window > 0:
            self._fill_swa_decode_static(batch)
        batch.attn_metadata = self._decode_metadata_static(batch.padded_size)

    def prepare_for_replay(self, batch: "Batch") -> None:
        self._fill_decode_static(batch)
        if self.swa_kv is not None and self.swa_window > 0:
            self._fill_swa_decode_static(batch)
        batch.attn_metadata = self._decode_metadata_static(batch.padded_size)

    # ---- spec-verify cudagraph capture (v2 S1: STANDARD K+1 causal verify, no custom mask) --------
    # The two-forward CCA verify forward stages qlen=K+1 query tokens/seq against the paged prefix with
    # a prefix-offset causal mask -> the inherited `_hip_prefill_paged` kernel (max_seqlen_q>1 branch).
    # For capture, the only per-step-varying metadata the kernel reads is cache_seqlens (device_len) and
    # the page table (a row can gain a page); both live in static buffers refreshed by
    # `prepare_verify_for_replay` before g.replay(). cu_seqlens_q is a static arange*qlen (all q-lengths
    # are qlen). max_seqlen_q is the fixed qlen. The kernel bounds reads by cache_seqlens, so a fixed
    # max-width page table (stale tail ignored) is fine — same trick as decode/MLA-verify. Mirrors
    # MLABackend.init_verify_capture / _fill_verify_static. NOTE (v2 S4): the FUSED custom-mask forward
    # needs an additional static mask_bias + §7.6 positions buffer — see docs/V2_CCA_VERIFY_CAPTURE.md.
    def init_verify_capture(self, max_seq_len: int, bs_list: List[int], num_draft: int) -> None:
        dev = self.kvcache.device
        self._vcap_max_bs = max(bs_list)
        self._vcap_qlen = num_draft + 1
        self._vcap_max_pages = (max_seq_len + self.page_size - 1) // self.page_size
        self._vcap_cache_seqlens = torch.ones(self._vcap_max_bs, dtype=torch.int32, device=dev)
        self._vcap_page_table = torch.zeros(
            self._vcap_max_bs, self._vcap_max_pages, dtype=torch.int32, device=dev
        )
        self._vcap_cu_q = (
            torch.arange(self._vcap_max_bs + 1, dtype=torch.int32, device=dev) * self._vcap_qlen
        )

    def _verify_metadata_static(self, bs: int) -> RDNA4Metadata:
        return RDNA4Metadata(
            cache_seqlens=self._vcap_cache_seqlens[:bs],
            cu_seqlens_q=self._vcap_cu_q[: bs + 1],
            max_seqlen_q=self._vcap_qlen,
            max_seqlen_k=self._vcap_max_pages * self.page_size,
            page_table=self._vcap_page_table[:bs],
            cold_prefill=False,
        )

    def _fill_verify_static(self, batch: "Batch") -> None:
        """Refresh the static verify buffers from `batch.padded_reqs` (eager, OUTSIDE the graph).
        cache_seqlens = device_len; page_table = each seq's page row (stale tail beyond cache_seqlens
        is ignored by the paged-extend kernel's causal bound)."""
        reqs = batch.padded_reqs
        bs = len(reqs)
        dev = self.kvcache.device
        dls = torch.tensor([req.device_len for req in reqs], dtype=torch.int32, device=dev)
        self._vcap_cache_seqlens[:bs].copy_(dls)
        gpt = get_global_ctx().page_table  # global page_size=1 table
        # Vectorized gather (was a per-req Python loop): same trick as _fill_decode_static — pull all
        # rows' full max-width strided page ids in one advanced-index op. The paged-extend verify kernel
        # bounds its key reads by cache_seqlens, so writing the whole width (incl. the per-seq stale tail
        # beyond npages) is equivalent to the old per-row [:npages] copy. This runs every spec-decode
        # verify step, so the per-req loop was pure overhead on the hot path.
        table_idx = torch.tensor([req.table_idx for req in reqs], dtype=torch.long, device=gpt.device)
        rows = gpt[table_idx, : self._vcap_max_pages * self.page_size : self.page_size]  # [bs, ncols]
        if self.page_size > 1:
            rows = torch.div(rows, self.page_size, rounding_mode="floor")
        ncols = rows.shape[1]
        self._vcap_page_table[:bs, :ncols].copy_(rows.to(torch.int32))

    def prepare_verify_for_capture(self, batch: "Batch") -> None:
        self._fill_verify_static(batch)
        batch.attn_metadata = self._verify_metadata_static(batch.padded_size)

    def prepare_verify_for_replay(self, batch: "Batch") -> None:
        self._fill_verify_static(batch)
        batch.attn_metadata = self._verify_metadata_static(batch.padded_size)

    # ---- FUSED spec-verify cudagraph capture (v2 S4: the custom-mask single-forward) --------------
    # The fused-TiDAR forward stages `fused_qlen = 1+B+B²` (flat) or `1+B+B·(tp+B)` (segmented) query
    # tokens/seq and runs the paged-extend kernel with `causal=0` + a DENSE `custom_mask` [total_q,
    # max_kv] carrying the whole block-diffusion structure. Two axes differ from the K+1 verify above:
    #   * qlen is `fused_qlen` (fixed per (B, layout)), not `num_draft+1`;
    #   * the kernel reads `metadata.custom_mask`, whose eager `max_kv = c0 + fused_qlen` GROWS each
    #     step. For capture the mask must live in a STATIC max-width buffer `[max_bs*fused_qlen,
    #     max_pages*ps]` (contiguous). Its row stride (= max_pages*ps) is then CONSTANT across capture
    #     and every replay — the graph bakes `mask_kv_stride` once (bindings.cpp reads `mb.size(1)`),
    #     so we ALWAYS pass the full-width slice `static[:total_q, :]`. The kernel bounds key reads by
    #     `cache_seqlens` (= context_len ≤ max_kv), so the stale columns past context_len are ignored
    #     — the same trick as the page table. On replay the scheduler-built `[total_q, max_kv]` mask is
    #     copied into the static buffer's leading rows/cols; dummy-padded rows keep all-allowed (0.0)
    #     from init (their output is discarded). See docs/V2_CCA_VERIFY_CAPTURE.md §S4.
    def init_fused_verify_capture(self, max_seq_len: int, bs_list: List[int], fused_qlen: int) -> None:
        dev = self.kvcache.device
        self._fcap_max_bs = max(bs_list)
        self._fcap_qlen = fused_qlen
        self._fcap_max_pages = (max_seq_len + self.page_size - 1) // self.page_size
        self._fcap_max_kv = self._fcap_max_pages * self.page_size
        self._fcap_cache_seqlens = torch.ones(self._fcap_max_bs, dtype=torch.int32, device=dev)
        self._fcap_page_table = torch.zeros(
            self._fcap_max_bs, self._fcap_max_pages, dtype=torch.int32, device=dev
        )
        self._fcap_cu_q = (
            torch.arange(self._fcap_max_bs + 1, dtype=torch.int32, device=dev) * fused_qlen
        )
        # Static max-width mask buffer (contiguous). 0.0 = all-allowed; filled per replay. Full-width
        # slices keep a constant row stride (max_kv) across capture/replay so the graph is stable.
        self._fcap_custom_mask = torch.zeros(
            self._fcap_max_bs * fused_qlen, self._fcap_max_kv, dtype=torch.float32, device=dev
        )

    def _fused_verify_metadata_static(self, bs: int) -> RDNA4Metadata:
        tq = bs * self._fcap_qlen
        return RDNA4Metadata(
            cache_seqlens=self._fcap_cache_seqlens[:bs],
            cu_seqlens_q=self._fcap_cu_q[: bs + 1],
            max_seqlen_q=self._fcap_qlen,
            max_seqlen_k=self._fcap_max_kv,
            page_table=self._fcap_page_table[:bs],
            cold_prefill=False,
            custom_mask=self._fcap_custom_mask[:tq, :],  # full-width -> constant stride
        )

    def _fill_fused_verify_static(self, batch: "Batch") -> None:
        """Refresh cache_seqlens + page_table from `batch.padded_reqs` (eager, OUTSIDE the graph)."""
        reqs = batch.padded_reqs
        bs = len(reqs)
        dev = self.kvcache.device
        dls = torch.tensor([req.device_len for req in reqs], dtype=torch.int32, device=dev)
        self._fcap_cache_seqlens[:bs].copy_(dls)
        gpt = get_global_ctx().page_table  # global page_size=1 table
        # Vectorized gather (was a per-req Python loop) — identical trick to _fill_decode_static /
        # _fill_verify_static: one advanced-index op over all rows; the kernel bounds key reads by
        # cache_seqlens so the stale per-seq tail beyond npages is ignored.
        table_idx = torch.tensor([req.table_idx for req in reqs], dtype=torch.long, device=gpt.device)
        rows = gpt[table_idx, : self._fcap_max_pages * self.page_size : self.page_size]  # [bs, ncols]
        if self.page_size > 1:
            rows = torch.div(rows, self.page_size, rounding_mode="floor")
        ncols = rows.shape[1]
        self._fcap_page_table[:bs, :ncols].copy_(rows.to(torch.int32))

    def prepare_fused_verify_for_capture(self, batch: "Batch") -> None:
        # Dummy capture batch: cache_seqlens = fused_qlen (cached_len 0 dummy), page table -> dummy page.
        # Mask stays all-allowed (0.0) from init so the warmup attention is well-defined (no all-(-inf)
        # row -> no NaN). Capture records pointers/strides only; replay overwrites the values.
        self._fill_fused_verify_static(batch)
        batch.attn_metadata = self._fused_verify_metadata_static(batch.padded_size)

    def prepare_fused_verify_for_replay(self, batch: "Batch") -> None:
        # The scheduler built `batch.attn_metadata.custom_mask` = [total_q_real, max_kv_real]; grab it
        # BEFORE swapping in the static metadata, then copy into the static buffer's leading block.
        src_mask = batch.attn_metadata.custom_mask
        self._fill_fused_verify_static(batch)
        tq_pad = batch.padded_size * self._fcap_qlen
        if src_mask is not None:
            tq, kv = src_mask.shape
            self._fcap_custom_mask[:tq, :kv].copy_(src_mask)
            # dummy-padded rows (if bs was rounded up): reset their read window to all-allowed so a
            # prior replay's stale mask never denies every key (cache_seqlens bounds reads to fused_qlen
            # for a cached_len-0 dummy). Real rows past `kv` are bounded away by cache_seqlens.
            if tq < tq_pad:
                self._fcap_custom_mask[tq:tq_pad, : self._fcap_qlen].zero_()
        batch.attn_metadata = self._fused_verify_metadata_static(batch.padded_size)

    # ---- DDTREE spec-verify cudagraph capture (draft-TREE ancestor-mask single-forward) -----------
    # The DDTree tree-verify stages `tree_qlen = budget+1` query tokens/seq (each req's real tree nodes
    # PADDED up to that fixed count) and runs the paged-extend kernel with `causal=0` + an ancestor-only
    # `custom_mask`. Identical static-mask machinery to the FUSED path (init_fused_verify_capture) with
    # two differences:
    #   * qlen is `tree_qlen` (budget+1), not a TiDAR-block-derived fused_qlen; and
    #   * the mask width is CAPPED at `max_ctx` (MINISGL_DDTREE_MAXCTX). The custom_mask carries one
    #     column per key over the WHOLE per-seq context, so an uncapped model-max width (e.g. 40960)
    #     would blow the static buffer (max_bs*qlen*max_kv*4 B) on a 16 GB card. Sequences whose context
    #     exceeds the cap fall back to EAGER (still lossless) — see GraphRunner.can_use_ddtree_verify.
    def init_ddtree_verify_capture(
        self, max_seq_len: int, bs_list: List[int], tree_qlen: int, max_ctx: int
    ) -> None:
        dev = self.kvcache.device
        self._dcap_max_bs = max(bs_list)
        self._dcap_qlen = tree_qlen
        full = ((max_seq_len + self.page_size - 1) // self.page_size) * self.page_size
        self._dcap_max_kv = min(full, max_ctx)
        self._dcap_max_pages = (self._dcap_max_kv + self.page_size - 1) // self.page_size
        self._dcap_cache_seqlens = torch.ones(self._dcap_max_bs, dtype=torch.int32, device=dev)
        self._dcap_page_table = torch.zeros(
            self._dcap_max_bs, self._dcap_max_pages, dtype=torch.int32, device=dev
        )
        self._dcap_cu_q = (
            torch.arange(self._dcap_max_bs + 1, dtype=torch.int32, device=dev) * tree_qlen
        )
        self._dcap_custom_mask = torch.zeros(
            self._dcap_max_bs * tree_qlen, self._dcap_max_kv, dtype=torch.float32, device=dev
        )

    def _ddtree_verify_metadata_static(self, bs: int) -> RDNA4Metadata:
        tq = bs * self._dcap_qlen
        return RDNA4Metadata(
            cache_seqlens=self._dcap_cache_seqlens[:bs],
            cu_seqlens_q=self._dcap_cu_q[: bs + 1],
            max_seqlen_q=self._dcap_qlen,
            max_seqlen_k=self._dcap_max_kv,
            page_table=self._dcap_page_table[:bs],
            cold_prefill=False,
            custom_mask=self._dcap_custom_mask[:tq, :],  # full-width -> constant stride
        )

    def _fill_ddtree_verify_static(self, batch: "Batch") -> None:
        """Refresh cache_seqlens + page_table from `batch.padded_reqs` (eager, OUTSIDE the graph)."""
        reqs = batch.padded_reqs
        bs = len(reqs)
        dev = self.kvcache.device
        dls = torch.tensor([req.device_len for req in reqs], dtype=torch.int32, device=dev)
        self._dcap_cache_seqlens[:bs].copy_(dls)
        gpt = get_global_ctx().page_table  # global page_size=1 table
        for i, req in enumerate(reqs):
            npages = (req.device_len + self.page_size - 1) // self.page_size
            row = gpt[req.table_idx, : npages * self.page_size : self.page_size]
            if self.page_size > 1:
                row = torch.div(row, self.page_size, rounding_mode="floor")
            self._dcap_page_table[i, :npages].copy_(row.to(torch.int32))

    def prepare_ddtree_verify_for_capture(self, batch: "Batch") -> None:
        # Dummy capture batch: cache_seqlens = tree_qlen (cached_len 0 dummy), page table -> dummy page.
        # Mask stays all-allowed (0.0) from init so the warmup attention is well-defined.
        self._fill_ddtree_verify_static(batch)
        batch.attn_metadata = self._ddtree_verify_metadata_static(batch.padded_size)

    def prepare_ddtree_verify_for_replay(self, batch: "Batch") -> None:
        # The scheduler built `batch.attn_metadata.custom_mask` = [total_q_real, ctx_real]; grab it
        # BEFORE swapping in the static metadata, then copy into the static buffer's leading block.
        src_mask = batch.attn_metadata.custom_mask
        self._fill_ddtree_verify_static(batch)
        tq_pad = batch.padded_size * self._dcap_qlen
        if src_mask is not None:
            tq, kv = src_mask.shape
            self._dcap_custom_mask[:tq, :kv].copy_(src_mask)
            # dummy-padded rows (if bs rounded up): reset their window to all-allowed. Real rows past
            # `kv` are bounded away by cache_seqlens (= ctx_real = kv), so no stale-column zeroing needed.
            if tq < tq_pad:
                self._dcap_custom_mask[tq:tq_pad, : self._dcap_qlen].zero_()
        batch.attn_metadata = self._ddtree_verify_metadata_static(batch.padded_size)
