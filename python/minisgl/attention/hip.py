"""Fully Triton-free attention for gfx1201 (RDNA4): native HIP rocwmma flash-PREFILL
(``torch.ops.attn_hip.flash_prefill``) + native HIP paged flash-DECODE
(``torch.ops.attn_decode.flash_decode_paged``). No Triton kernel is invoked on either path.

Subclasses ``RDNA4Backend`` ONLY to reuse its ``__init__`` (kvcache / softmax-scale policy /
page_size / fp8 detection) and ``prepare_metadata`` (which builds the ``RDNA4Metadata`` page-table +
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
from minisgl.utils import init_logger

from .rdna4 import _HIP_HEAD_DIMS, RDNA4Backend, RDNA4Metadata

if TYPE_CHECKING:
    from minisgl.core import Batch
    from minisgl.models import ModelConfig



logger = init_logger(__name__)

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
        # Kernel-coverage gate on THIS query's head_dim (see RDNA4Backend.forward). This backend is
        # Triton-free, so there is nothing to fall back to — but the kernels' own TORCH_CHECK fires
        # three layers down and names neither the layer type nor the fact that a split-head_dim
        # model's OTHER geometry is fine. Raise here, where that can be said.
        if q.shape[-1] not in _HIP_HEAD_DIMS:
            self._no_hip_kernel(q.shape[-1])
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
        # consumed the flat k/v, same as the Triton path.) D is THIS call's query width — GQA varies
        # the head count, never the head width — because a split-head_dim model has no model-wide D.
        D = q.shape[-1]
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
        scale = self._softmax_scale(q)
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
                scale, 1, 0,  # causal=1, sliding_window=0 (matches the Triton path)
            )
        return out

    def _forward_decode(
        self, q: torch.Tensor, layer_id: int, metadata: RDNA4Metadata
    ) -> torch.Tensor:
        k_cache = self.kvcache.k_cache(layer_id)  # [num_pages, page_size, kv_heads, head_dim]
        v_cache = self.kvcache.v_cache(layer_id)
        block_table = metadata.page_table.to(torch.int32)
        ctx_lens = metadata.cache_seqlens.to(torch.int32)
        scale = self._softmax_scale(q)
        from minisgl._hip_engage import engaged
        if self.kv_is_fp8:
            # fp8 (e4m3) paged KV: per-tensor descale = the calibrated store scale (A5; 1.0 if
            # calibration off), folded in the kernel. Pass the PERSISTENT device descale tensors
            # (0-dim views) the canonical op now requires — stable address, graph-safe.
            ks, vs = self.kvcache.k_descale[layer_id], self.kvcache.v_descale[layer_id]
            engaged("attn_decode.flash_decode_paged_fp8")
            ret = self._decode_fp8(q, k_cache, v_cache, block_table, ctx_lens, scale, ks, vs, 0)
        else:
            engaged("attn_decode.flash_decode_paged")
            ret = self._decode(q, k_cache, v_cache, block_table, ctx_lens, scale, 0)
        # ATTENTION PARITY TAP (MINISGL_ATTN_PARITY_N=N, debug; eager decode only — a python tap
        # cannot run inside a captured graph). Every Nth decode-attention call, recompute this
        # layer's decode attention in torch fp32 from the SAME paged KV (incl. the fp8 descale) and
        # log the worst per-row rel error. There is no second attention backend on RDNA4 to A/B
        # against, so the reference IS torch — this is how the HIP decode-attention kernel gets
        # checked in real serving conditions (real page tables, real lengths, real KV contents).
        n = getattr(self, "_parity_n", None)
        if n is None:
            n = self._parity_n = int(os.environ.get("MINISGL_ATTN_PARITY_N", "0") or 0)
            self._parity_calls = 0
        if n and not torch.cuda.is_current_stream_capturing():
            self._parity_calls += 1
            if self._parity_calls % n == 0:
                self._attn_parity_check(q, ret, layer_id, metadata, scale)
        return ret

    def _attn_parity_check(self, q, out, layer_id, metadata, scale) -> None:
        try:
            k_cache = self.kvcache.k_cache(layer_id)
            v_cache = self.kvcache.v_cache(layer_id)
            ps = k_cache.shape[1]
            worst, worst_row, worst_len = 0.0, -1, 0
            for i in range(q.shape[0]):
                L = int(metadata.cache_seqlens[i])
                if L <= 0:
                    continue
                pages = metadata.page_table[i, : (L + ps - 1) // ps].long()
                k = k_cache[pages].reshape(-1, k_cache.shape[-2], k_cache.shape[-1])[:L]
                v = v_cache[pages].reshape(-1, v_cache.shape[-2], v_cache.shape[-1])[:L]
                if self.kv_is_fp8:
                    k = k.float() * float(self.kvcache.k_descale[layer_id])
                    v = v.float() * float(self.kvcache.v_descale[layer_id])
                else:
                    k, v = k.float(), v.float()
                qi = q[i].float()
                Hq, D = qi.shape
                G = Hq // k.shape[1]
                k = k.repeat_interleave(G, dim=1)   # kv head h//G serves query head h
                v = v.repeat_interleave(G, dim=1)
                p = torch.softmax(torch.einsum("hd,lhd->hl", qi, k) * scale, dim=-1)
                ref = torch.einsum("hl,lhd->hd", p, v)
                r = ((out[i].float().reshape(Hq, D) - ref).norm()
                     / ref.norm().clamp_min(1e-9)).item()
                if r > worst:
                    worst, worst_row, worst_len = r, i, L
            lvl = logger.warning_rank0 if worst > 2e-2 else logger.info_rank0
            lvl(f"[attn-parity] layer={layer_id} call={self._parity_calls} "
                f"worst_rel={worst:.3e} row={worst_row} kv_len={worst_len}"
                + (" — DIVERGENT" if worst > 2e-2 else ""))
        except Exception as e:  # a broken tap must never take down the serve
            logger.warning_rank0(f"[attn-parity] check failed: {e}")

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

    def _fill_paged_static(
        self, reqs: list, cache_seqlens: torch.Tensor, page_table: torch.Tensor, max_pages: int
    ) -> None:
        """Refresh a captured family's MAIN-POOL statics (cache_seqlens + page table) from `reqs`.

        One body, shared by every fixed-shape capture family (decode, K+1 spec verify, fused-TiDAR
        verify, block-diffusion canvas) because it was already four byte-identical copies and a
        divergence between them is invisible: each one would still run, just against a stale or
        mis-strided table. Runs eager, OUTSIDE the graph; it writes the exact tensors the captured
        kernels read through their baked pointers.

        Vectorized (was a per-req Python loop on the decode hot path): pull all rows' full max-width
        strided page ids in one advanced-index op. Every kernel here bounds its key reads by
        cache_seqlens, so writing the whole width — including the per-seq stale tail past that
        sequence's own page count — is equivalent to a per-row `[:npages]` copy."""
        bs = len(reqs)
        dev = cache_seqlens.device
        cache_seqlens[:bs].copy_(
            torch.tensor([req.device_len for req in reqs], dtype=torch.int32, device=dev)
        )
        gpt = get_global_ctx().page_table  # global page_size=1 table
        table_idx = torch.tensor([req.table_idx for req in reqs], dtype=torch.long, device=gpt.device)
        rows = gpt[table_idx, : max_pages * self.page_size : self.page_size]  # [bs, ncols]
        if self.page_size > 1:
            rows = torch.div(rows, self.page_size, rounding_mode="floor")
        page_table[:bs, : rows.shape[1]].copy_(rows.to(torch.int32))

    def _fill_decode_static(self, batch: "Batch") -> None:
        """Refresh the static decode buffers from `batch.padded_reqs` (real rows + dummy padding)."""
        self._fill_paged_static(
            batch.padded_reqs, self._cap_cache_seqlens, self._cap_page_table, self._cap_max_pages
        )

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
    def init_verify_capture(
        self, max_seq_len: int, bs_list: List[int], num_draft: "int | List[int]"
    ) -> None:
        # `num_draft` may be a LIST of widths (adaptive verify width, spec/width.py): every captured
        # width gets its own qlen-shaped statics, and `set_verify_width` repoints the live ones before
        # each capture/replay. The qlen-INDEPENDENT buffers (cache_seqlens, page_table, and the SWA
        # ring cache_seqlens) are allocated ONCE and shared by every width.
        dev = self.kvcache.device
        widths = [num_draft] if isinstance(num_draft, int) else sorted(set(int(w) for w in num_draft))
        self._vcap_max_bs = max(bs_list)
        self._vcap_max_pages = (max_seq_len + self.page_size - 1) // self.page_size
        self._vcap_cache_seqlens = torch.ones(self._vcap_max_bs, dtype=torch.int32, device=dev)
        self._vcap_page_table = torch.zeros(
            self._vcap_max_bs, self._vcap_max_pages, dtype=torch.int32, device=dev
        )
        # ---- SWA-hybrid (Laguna): STATIC ring-pool VERIFY metadata (persistent, refreshed in place) ----
        # The 30 sliding layers verify the K+1 tokens through the ring pool. Made capturable by the
        # paged-from-ring path (rdna4._swa_prefill_paged) — mirrors the decode SWA static buffers (Track
        # D) at the verify shape. Widths are FIXED per captured qlen: swa_out_loc holds bs*qlen new-token
        # store slots; the ring block table is [bs, W+qlen] (window Wp<=W ++ the qlen new slots);
        # cache_seqlens = Wp+qlen bounds the read (padded tail ignored). Contents are rebuilt by
        # _fill_swa_verify_static before capture and every replay; the captured store_kv/paged-extend
        # read them through fixed pointers. NOTE these are allocated PER WIDTH rather than sliced out of
        # a max-width buffer: the page table's ROW STRIDE is W+qlen, so a `[:, :W+ql]` view of a wider
        # allocation would be non-contiguous and the kernel reads it as a dense [bs, W+ql] block. Each
        # one is a few KB, so the duplication is free.
        swa = self.swa_kv is not None and self.swa_window > 0
        if swa:
            self._vcap_swa_cache_seqlens = torch.ones(
                self._vcap_max_bs, dtype=torch.int32, device=dev
            )
        self._vcap_by_qlen: "dict[int, dict]" = {}
        for w in widths:
            ql = w + 1
            ent = {
                "cu_q": torch.arange(self._vcap_max_bs + 1, dtype=torch.int32, device=dev) * ql,
            }
            if swa:
                ent["swa_out_loc"] = torch.zeros(
                    self._vcap_max_bs * ql, dtype=torch.int32, device=dev
                )
                ent["swa_page_table"] = torch.zeros(
                    self._vcap_max_bs, self.swa_window + ql, dtype=torch.int32, device=dev
                )
            self._vcap_by_qlen[ql] = ent
        self.set_verify_width(max(widths) + 1)

    def set_verify_width(self, qlen: int) -> None:
        """Point the live verify statics at the buffers captured for `qlen` query rows/seq. Called by
        GraphRunner immediately before prepare_verify_for_capture / _for_replay, so the fill helpers
        and the metadata builder below need no width argument."""
        ent = self._vcap_by_qlen[qlen]
        self._vcap_qlen = qlen
        self._vcap_cu_q = ent["cu_q"]
        if self.swa_kv is not None and self.swa_window > 0:
            self._vcap_swa_out_loc = ent["swa_out_loc"]
            self._vcap_swa_page_table = ent["swa_page_table"]

    def _verify_metadata_static(self, bs: int) -> RDNA4Metadata:
        md = RDNA4Metadata(
            cache_seqlens=self._vcap_cache_seqlens[:bs],
            cu_seqlens_q=self._vcap_cu_q[: bs + 1],
            max_seqlen_q=self._vcap_qlen,
            max_seqlen_k=self._vcap_max_pages * self.page_size,
            page_table=self._vcap_page_table[:bs],
            cold_prefill=False,
        )
        # SWA-hybrid: attach the persistent ring-pool VERIFY metadata the sliding layers read (paged
        # verify). swa_out_loc = bs*qlen new-token store slots (store_kv), swa_verify_page_table /
        # swa_verify_cache_seqlens = the [window|new] ring block table + context length.
        if self.swa_kv is not None and self.swa_window > 0:
            md.swa_out_loc = self._vcap_swa_out_loc[: bs * self._vcap_qlen]
            md.swa_verify_page_table = self._vcap_swa_page_table[:bs]
            md.swa_verify_cache_seqlens = self._vcap_swa_cache_seqlens[:bs]
        return md

    def _fill_verify_static(self, batch: "Batch") -> None:
        """Refresh the static verify buffers from `batch.padded_reqs` (eager, OUTSIDE the graph)."""
        self._fill_paged_static(
            batch.padded_reqs, self._vcap_cache_seqlens, self._vcap_page_table,
            self._vcap_max_pages,
        )

    def _fill_swa_multiquery_static(
        self, reqs: list, qlen: int, out_loc: torch.Tensor, page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
    ) -> None:
        """Refresh a fixed-qlen multi-query SWA ring-pool static set from `reqs` (eager, OUTSIDE the
        graph). Shared by the K+1 spec verify and the block-diffusion canvas: the ROWS are identical
        arithmetic, only the query count and the causality differ (the latter is a metadata flag, not
        a row). For each seq (device_len S, qlen new tokens): cached_len c0 = S - qlen, window
        Wp = min(c0, W). Builds, per seq:
            out_loc (bs*qlen)   ring slot per NEW token = table_idx*R + (c0+j) % R
            page_table[i]       [ window slots  base+(p%R) for p in [c0-Wp, c0)  |  the qlen new slots ]
                                (ascending absolute position; tail past Wp+qlen padded with 0)
            cache_seqlens[i]    Wp + qlen         (bounds the paged read; padded tail ignored)
        These are the SAME ring slots the eager `_build_swa_metadata` / `_gather_swa_windows` (verify)
        and `_build_swa_canvas_metadata` (canvas) address — store slots identical, window read = the
        last Wp positions at their ring slots p%R — so a captured replay is bit-identical to the eager
        forward it replaces. `page_table`'s ROW STRIDE is W+qlen and the kernel reads it as a dense
        block, so it must be an allocation of exactly that width, never a narrow view of a wider one.
        bs is tiny (<= the captured max), so the O(bs*(W+qlen)) python build is negligible against the
        forward it feeds; one pinned H2D per buffer."""
        bs = len(reqs)
        dev = out_loc.device
        W = self.swa_window
        R = self.swa_ring_stride
        row_w = page_table.shape[1]
        out_slots: list[int] = []
        pt_rows: list[list[int]] = []
        ctx: list[int] = []
        for req in reqs:
            base = req.table_idx * R
            c0 = req.device_len - qlen  # cached prefix before the qlen new tokens
            Wp = min(c0, W) if c0 > 0 else 0
            new_slots = [base + ((c0 + j) % R) for j in range(qlen)]
            out_slots.extend(new_slots)
            win_slots = [base + (p % R) for p in range(c0 - Wp, c0)]  # ascending absolute pos
            row = win_slots + new_slots
            row += [0] * (row_w - len(row))  # pad tail (never read; ctx bounds it)
            pt_rows.append(row)
            ctx.append(Wp + qlen)
        CPU = {"device": "cpu", "dtype": torch.int32, "pin_memory": True}
        out_loc[: bs * qlen].copy_(torch.tensor(out_slots, **CPU).to(dev, non_blocking=True))
        page_table[:bs].copy_(torch.tensor(pt_rows, **CPU).to(dev, non_blocking=True))
        cache_seqlens[:bs].copy_(torch.tensor(ctx, **CPU).to(dev, non_blocking=True))

    def _fill_swa_verify_static(self, batch: "Batch") -> None:
        self._fill_swa_multiquery_static(
            batch.padded_reqs, self._vcap_qlen, self._vcap_swa_out_loc,
            self._vcap_swa_page_table, self._vcap_swa_cache_seqlens,
        )

    def prepare_verify_for_capture(self, batch: "Batch") -> None:
        self._fill_verify_static(batch)
        if self.swa_kv is not None and self.swa_window > 0:
            self._fill_swa_verify_static(batch)
        batch.attn_metadata = self._verify_metadata_static(batch.padded_size)

    def prepare_verify_for_replay(self, batch: "Batch") -> None:
        self._fill_verify_static(batch)
        if self.swa_kv is not None and self.swa_window > 0:
            self._fill_swa_verify_static(batch)
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
        self._fill_paged_static(
            batch.padded_reqs, self._fcap_cache_seqlens, self._fcap_page_table,
            self._fcap_max_pages,
        )

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

    # ---- BLOCK-DIFFUSION CANVAS cudagraph capture (bidirectional multi-query) ---------------------
    # A canvas denoising step is the SAME static shape as the K+1 spec verify above — a fixed number
    # of query tokens per sequence run through the paged-extend kernel against the main pool, plus a
    # `[window | new]` ring row for the sliding layers — with exactly two differences:
    #   * qlen is `canvas_length` (256), not num_draft+1; and
    #   * `bidirectional=True`, which is what makes `_hip_prefill_paged` / `_swa_prefill_paged` run
    #     causal=0 AND sliding_window=0. A canvas has no mask at all: every query attends every key
    #     inside cache_seqlens, on BOTH layer geometries (measured, tools/canvas_attention_probe.py).
    # It gets its OWN buffers rather than borrowing the verify family's for the reason
    # init_verify_capture already documents about widths: a canvas ring row is `W + 256` wide where a
    # verify row is `W + K+1`, and the kernel reads that table as a DENSE [bs, row_w] block, so a
    # narrow view of a wider allocation would be silently mis-strided rather than an error. The fill
    # helpers are shared (`_fill_paged_static` / `_fill_swa_multiquery_static`) — the rows are the
    # same arithmetic, only the query count differs.
    #
    # WHY THIS IS WORTH CAPTURING AT ALL, given the step is 256 tokens wide rather than 1: a block runs
    # the IDENTICAL forward k times (k = 12-19 measured), and everything that varies across those k
    # steps is the CONTENTS of these buffers plus the input ids — the shapes and the slot addressing
    # are fixed for the whole block. WHAT IT IS NOT is a speedup: measured, capture collapses the
    # per-step host launch loop from 30.8 ms to 0.8 ms and moves the step by -0.3%, because that
    # launch time was entirely overlapped with GPU work (gfx activity 100% median). It is carried for
    # correctness and for the residual it will expose once the backbone shrinks, not for tok/s. See
    # GraphRunner.capture_canvas_graphs and docs/DIFFUSIONGEMMA_BLOCK_DIFFUSION.md D6.
    def init_canvas_capture(self, max_seq_len: int, bs_list: List[int], canvas_len: int) -> None:
        dev = self.kvcache.device
        self._ccap_max_bs = max(bs_list)
        self._ccap_qlen = canvas_len
        self._ccap_max_pages = (max_seq_len + self.page_size - 1) // self.page_size
        self._ccap_cache_seqlens = torch.ones(self._ccap_max_bs, dtype=torch.int32, device=dev)
        self._ccap_page_table = torch.zeros(
            self._ccap_max_bs, self._ccap_max_pages, dtype=torch.int32, device=dev
        )
        self._ccap_cu_q = (
            torch.arange(self._ccap_max_bs + 1, dtype=torch.int32, device=dev) * canvas_len
        )
        if self.swa_kv is not None and self.swa_window > 0:
            self._ccap_swa_out_loc = torch.zeros(
                self._ccap_max_bs * canvas_len, dtype=torch.int32, device=dev
            )
            self._ccap_swa_page_table = torch.zeros(
                self._ccap_max_bs, self.swa_window + canvas_len, dtype=torch.int32, device=dev
            )
            self._ccap_swa_cache_seqlens = torch.ones(
                self._ccap_max_bs, dtype=torch.int32, device=dev
            )

    def _canvas_metadata_static(self, bs: int) -> RDNA4Metadata:
        md = RDNA4Metadata(
            cache_seqlens=self._ccap_cache_seqlens[:bs],
            cu_seqlens_q=self._ccap_cu_q[: bs + 1],
            max_seqlen_q=self._ccap_qlen,
            max_seqlen_k=self._ccap_max_pages * self.page_size,
            page_table=self._ccap_page_table[:bs],
            cold_prefill=False,
            # THE flag that separates this from a verify of the same shape. Dropping it would replay
            # a perfectly healthy graph that applies a prefix-offset CAUSAL mask to the canvas — a
            # different model, and one that still emits fluent text (rel_fro 0.60 apart, §C3).
            bidirectional=True,
        )
        if self.swa_kv is not None and self.swa_window > 0:
            md.swa_out_loc = self._ccap_swa_out_loc[: bs * self._ccap_qlen]
            md.swa_verify_page_table = self._ccap_swa_page_table[:bs]
            md.swa_verify_cache_seqlens = self._ccap_swa_cache_seqlens[:bs]
        return md

    def _prepare_canvas_static(self, batch: "Batch") -> None:
        self._fill_paged_static(
            batch.padded_reqs, self._ccap_cache_seqlens, self._ccap_page_table,
            self._ccap_max_pages,
        )
        if self.swa_kv is not None and self.swa_window > 0:
            self._fill_swa_multiquery_static(
                batch.padded_reqs, self._ccap_qlen, self._ccap_swa_out_loc,
                self._ccap_swa_page_table, self._ccap_swa_cache_seqlens,
            )
        batch.attn_metadata = self._canvas_metadata_static(batch.padded_size)

    # Capture and replay prep are the same work — unlike the decode/verify families there is no mask
    # to seed and no width to repoint — but both names exist so the GraphRunner call sites read the
    # same as every other family.
    def prepare_canvas_for_capture(self, batch: "Batch") -> None:
        self._prepare_canvas_static(batch)

    def prepare_canvas_for_replay(self, batch: "Batch") -> None:
        self._prepare_canvas_static(batch)
