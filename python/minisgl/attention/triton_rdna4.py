from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch
from minisgl.core import Batch, get_global_ctx

from .base import BaseAttnBackend, BaseAttnMetadata
from ._triton_unified import unified_attention

if TYPE_CHECKING:
    from minisgl.models import ModelConfig


@dataclass
class RDNA4Metadata(BaseAttnMetadata):
    cache_seqlens: torch.Tensor  # per-seq total KV length (= seqused_k)
    cu_seqlens_q: torch.Tensor  # [bs+1] cumulative query lengths
    max_seqlen_q: int
    max_seqlen_k: int
    page_table: torch.Tensor  # [bs, max_pages] page-indexed block table

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
        # 3D flash-decode segment scratch (f32), lazily sized on first forward.
        self._seq_threshold_3D = 0
        self._segm_output: torch.Tensor | None = None
        self._segm_max: torch.Tensor | None = None
        self._segm_expsum: torch.Tensor | None = None

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
            k_descale=None,
            v_descale=None,
            seq_threshold_3D=self._seq_threshold_3D,
            num_par_softmax_segments=self.NUM_PAR_SOFTMAX_SEGMENTS,
            softmax_segm_output=self._segm_output,
            softmax_segm_max=self._segm_max,
            softmax_segm_expsum=self._segm_expsum,
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

        if max_seqlen_q == 1:
            cu_seqlens_q = torch.arange(0, len(reqs) + 1, device=device, dtype=torch.int32)
        elif all(l == 0 for l in cached_lens):  # prefill, no cache hit
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
