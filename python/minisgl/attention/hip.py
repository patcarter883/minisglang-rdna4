"""Fully Triton-free attention for gfx1201 (RDNA4): native HIP rocwmma flash-PREFILL
(``torch.ops.attn_hip.flash_prefill``) + native HIP paged flash-DECODE
(``torch.ops.attn_decode.flash_decode_paged``). No Triton kernel is invoked on either path.

Subclasses ``TritonRDNA4Backend`` ONLY to reuse its ``__init__`` (kvcache / scale / page_size /
fp8 detection) and ``prepare_metadata`` (which builds the ``RDNA4Metadata`` page-table +
cu_seqlens_q + cache_seqlens that both kernels consume). ``forward`` is fully overridden.

The two kernel packages are framework-agnostic ``torch.ops`` extensions shared with vllm-gfx1201
(prefill: the ``attn_hip`` worktree; paged decode: the ``attn_decode`` worktree). They must be
importable (on PYTHONPATH) when this backend is selected.

Constraints (v0 — eager, validated on dense head_dim 64/128):
  * PREFILL: the kernel is single-sequence + contiguous, so a varlen batch is sliced per sequence
    by cu_seqlens_q. Correct only when each sequence's keys are its own current tokens (NO
    prefix-cache hit) — asserted. head_dim 256 (Qwen3.5/3.6 full-attn) is not yet enabled in the
    prefill kernel (needs the smem-reduction pass); DECODE already supports 256.
  * cudagraph capture is not wired (run with cuda_graph_max_bs=0).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List

import torch
from minisgl.core import get_global_ctx

from .triton_rdna4 import RDNA4Metadata, TritonRDNA4Backend

if TYPE_CHECKING:
    from minisgl.core import Batch
    from minisgl.models import ModelConfig


class HIPAttnBackend(TritonRDNA4Backend):
    def __init__(self, config: "ModelConfig") -> None:
        super().__init__(config)
        # Import here (not at module top) so the kernel packages are only required when the
        # "hip" backend is actually selected. These register torch.ops.attn_hip.* / attn_decode.*.
        import attn_decode  # noqa: F401
        import attn_hip  # noqa: F401

        self._prefill = torch.ops.attn_hip.flash_prefill
        self._decode = torch.ops.attn_decode.flash_decode_paged
        self._decode_fp8 = torch.ops.attn_decode.flash_decode_paged_fp8

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: "Batch"
    ) -> torch.Tensor:
        metadata = batch.attn_metadata
        assert isinstance(metadata, RDNA4Metadata)
        # Persist the current tokens' K/V into the paged cache (decode reads it back).
        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)
        if batch.is_prefill:
            return self._forward_prefill(q, k, v, metadata)
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
        # cu_seqlens_q and run each independently. Valid only with no prefix-cache hit (each
        # seq's keys are exactly its current tokens). Assert that: cache_seqlens == query lengths.
        cu = metadata.cu_seqlens_q.tolist()
        klen = metadata.cache_seqlens.tolist()
        out = torch.empty_like(q)
        for i in range(len(cu) - 1):
            s, e = cu[i], cu[i + 1]
            qlen = e - s
            if qlen <= 0:
                continue
            assert klen[i] == qlen, (
                "HIP prefill does not support a prefix-cache hit "
                f"(seq {i}: kv_len={klen[i]} != q_len={qlen}); disable prefix caching "
                "(naive cache) or extend the kernel to gather the paged prefix."
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
        if self.kv_is_fp8:
            # fp8 (e4m3) paged KV: per-tensor descale folded in the kernel (store uses scale 1.0).
            return self._decode_fp8(q, k_cache, v_cache, block_table, ctx_lens, self.scale, 1.0, 1.0, 0)
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

    def _decode_metadata_static(self, bs: int) -> RDNA4Metadata:
        return RDNA4Metadata(
            cache_seqlens=self._cap_cache_seqlens[:bs],
            cu_seqlens_q=self._cap_cu_q[: bs + 1],
            max_seqlen_q=1,
            max_seqlen_k=self._cap_max_pages * self.page_size,
            page_table=self._cap_page_table[:bs],
            cold_prefill=False,
        )

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
        for i, req in enumerate(reqs):
            npages = (seqlens_k[i] + self.page_size - 1) // self.page_size
            row = gpt[req.table_idx, : npages * self.page_size : self.page_size]
            if self.page_size > 1:
                row = torch.div(row, self.page_size, rounding_mode="floor")
            self._cap_page_table[i, :npages].copy_(row.to(torch.int32))

    def prepare_for_capture(self, batch: "Batch") -> None:
        self._fill_decode_static(batch)
        batch.attn_metadata = self._decode_metadata_static(batch.padded_size)

    def prepare_for_replay(self, batch: "Batch") -> None:
        self._fill_decode_static(batch)
        batch.attn_metadata = self._decode_metadata_static(batch.padded_size)
