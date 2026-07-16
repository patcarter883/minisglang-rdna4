from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch
from minisgl._hip_engage import engaged
from minisgl.core import Batch, get_global_ctx

from .base import BaseAttnBackend, BaseAttnMetadata

if TYPE_CHECKING:
    from minisgl.models import ModelConfig


@dataclass
class MLAMetadata(BaseAttnMetadata):
    cache_seqlens: torch.Tensor  # [bs] per-seq total KV length (device_len) — int32
    cu_seqlens_q: torch.Tensor  # [bs+1] cumulative NEW-token (query) lengths — int32
    cu_seqlens_k: torch.Tensor  # [bs+1] cumulative total KV lengths — int32 (prefill materialize)
    max_seqlen_q: int
    page_table: torch.Tensor  # [bs, max_pages] page-indexed block table (for the decode kernel)
    # Spec-verify graph capture: the per-query (seq index, causal kbound) the mla_verify kernel reads.
    # Normally `verify()` derives these via `_verify_indices` (data-dependent repeat_interleave — NOT
    # cudagraph-capturable). For a captured verify graph the scheduler/backend precompute them into
    # STATIC buffers (q_lens are uniformly num_draft+1, so seq_idx is a fixed pattern and kbound is a
    # cheap per-step fill); when present, `verify()` reads them directly. None on the eager path.
    verify_seq_idx: "torch.Tensor | None" = None
    verify_kbound: "torch.Tensor | None" = None

    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.cu_seqlens_q[1 : 1 + bs] - 1


class MLABackend(BaseAttnBackend):
    """Multi-head latent attention (DeepSeek / GLM-4.x MoE) over the paged latent cache.

    The W_UK/W_UV absorption + the kv_a/kv_b/q projections live in the model's MLA layer; this
    backend is the thin kernel + cache + metadata shell:
      - ``store_latent`` scatters the new tokens' latent (c_KV-norm ‖ k_rope) into MLAKVCache.
      - ``decode`` runs the ABSORBED ``mla_hip.mla_decode`` (q[B,H,kv_lora+rope] over the latent).
      - ``prefill`` runs the MATERIALIZED ``mla_hip.mla_prefill`` (full per-head q/k/v assembled by
        the model from the latent — handles cold + extend via cu_seqlens_q/k + prefix-offset causal).

    The generic ``forward(q,k,v,...)`` is unused (the MLA layer calls the methods above directly).
    DECODE cudagraph capture is supported (static cache_seqlens + page_table buffers, see below);
    MoE-decode must run the graph-safe gather_reduce path (MINISGL_MOE_SCATTER=0). Prefill is eager.
    """

    def __init__(self, config: "ModelConfig"):
        ctx = get_global_ctx()
        self.config = config
        self.kvcache = ctx.kv_cache
        self.page_size = ctx.page_size
        # Softmax temperature = 1/sqrt(qk_head_dim); the absorbed dot over kv_lora+rope reconstructs
        # the same per-head qk_nope+qk_rope score, so prefill and decode share this scale.
        self.scale = float(config.qk_nope_head_dim + config.qk_rope_head_dim) ** -0.5
        import mla_hip  # canonical package: ops exposed as module-level callables

        self._decode_op = mla_hip.mla_decode
        self._decode_fp8_op = mla_hip.mla_decode_fp8
        self._prefill_op = mla_hip.mla_prefill
        self._verify_op = mla_hip.mla_verify
        self._verify_fp8_op = mla_hip.mla_verify_fp8
        # fp8 (e4m3) latent KV cache — opt-in via MINISGL_KV_FP8=1 (the engine allocates the latent
        # pool as float8_e4m3fn). Store is a plain bf16->e4m3 cast (scale 1.0), so decode dequant uses
        # descale 1.0, matching the HIP MHA fp8 path. The prefill rebuild dequants in the model layer.
        self.kv_is_fp8 = self.kvcache.dtype == torch.float8_e4m3fn
        # canonical mla fp8 ops now take k/v_descale as DEVICE tensors (read [0]). MLA's descale is a
        # static 1.0 (scale-1.0 store cast), so one persistent 1-elem tensor suffices — stable address,
        # graph-safe (a fresh torch.tensor() per forward would break cuda-graph replay).
        self._fp8_descale = torch.ones(1, dtype=torch.float32, device=self.kvcache.device)

    # ---- cache + kernels (called by the model's MLA layer) ----
    def store_latent(self, latent: torch.Tensor, out_loc: torch.Tensor, layer_id: int) -> None:
        # latent: [tokens, kv_lora_rank + qk_rope_head_dim]
        self.kvcache.store_kv(latent, None, out_loc, layer_id)

    def decode(self, q: torch.Tensor, layer_id: int, metadata: MLAMetadata) -> torch.Tensor:
        # q: [B, H, kv_lora_rank + qk_rope] -> out [B, H, kv_lora_rank]
        latent_cache = self.kvcache.latent_cache(layer_id)  # [num_pages, page_size, latent_dim]
        block_table = metadata.page_table.to(torch.int32)
        ctx_lens = metadata.cache_seqlens.to(torch.int32)
        if self.kv_is_fp8:
            # e4m3 latent cache: k_descale=v_descale=1.0 (store was a scale-1.0 cast).
            engaged("mla_hip.mla_decode_fp8")
            return self._decode_fp8_op(q, latent_cache, block_table, ctx_lens, self.scale,
                                       self._fp8_descale, self._fp8_descale, 0, 0)
        engaged("mla_hip.mla_decode")
        return self._decode_op(q, latent_cache, block_table, ctx_lens, self.scale, 0, 0)

    def verify(self, q: torch.Tensor, layer_id: int, metadata: MLAMetadata) -> torch.Tensor:
        """Speculative-decode VERIFY: absorbed MULTI-QUERY decode over the paged latent (no prefix
        re-materialization). q: [total_q, H, kv_lora_rank + qk_rope] (confirmed + drafts, packed by
        cu_seqlens_q) -> out [total_q, H, kv_lora_rank]. Each query attends latent [0, cached_len+qi]
        (causal); see mla_hip.mla_verify."""
        latent_cache = self.kvcache.latent_cache(layer_id)
        block_table = metadata.page_table.to(torch.int32)
        if metadata.verify_seq_idx is not None:
            # Captured-graph path: precomputed static indices (see MLAMetadata / prepare_verify_*).
            q_seq_idx, q_kbound = metadata.verify_seq_idx, metadata.verify_kbound
        else:
            q_seq_idx, q_kbound = self._verify_indices(metadata)
        if self.kv_is_fp8:
            engaged("mla_hip.mla_verify_fp8")
            return self._verify_fp8_op(
                q, latent_cache, block_table, q_seq_idx, q_kbound, self.scale,
                self._fp8_descale, self._fp8_descale, 0, 0
            )
        engaged("mla_hip.mla_verify")
        return self._verify_op(q, latent_cache, block_table, q_seq_idx, q_kbound, self.scale, 0, 0)

    def _verify_indices(self, metadata: MLAMetadata):
        """Build per-query-row (seq index, causal context length) from the verify metadata.
        q_kbound[r] = cached_len[seq] + within_seq_query_index + 1 (causal over prefix + this query
        + earlier drafts of the same seq). cached_len = device_len - q_len."""
        cu = metadata.cu_seqlens_q.to(torch.int64)  # [B+1]
        dev = cu.device
        q_lens = cu[1:] - cu[:-1]  # [B]
        cached = metadata.cache_seqlens.to(torch.int64) - q_lens  # prefix per seq
        B = q_lens.numel()
        total_q = int(cu[-1].item())
        seq_idx = torch.repeat_interleave(torch.arange(B, device=dev), q_lens)  # [total_q]
        qi = torch.arange(total_q, device=dev) - torch.repeat_interleave(cu[:-1], q_lens)
        q_kbound = torch.repeat_interleave(cached, q_lens) + qi + 1
        return seq_idx.to(torch.int32), q_kbound.to(torch.int32)

    def prefill(
        self,
        q: torch.Tensor,  # [total_q, H, qk_head_dim]
        k: torch.Tensor,  # [total_k, H, qk_head_dim]
        v: torch.Tensor,  # [total_k, H, v_head_dim]
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
    ) -> torch.Tensor:
        # causal=1, sliding_window=0; the prefix offset (k_len - q_len) is carried by cu_seqlens.
        engaged("mla_hip.mla_prefill")
        return self._prefill_op(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            cu_seqlens_q.to(torch.int32),
            cu_seqlens_k.to(torch.int32),
            self.scale,
            1,
            0,
            int(max_seqlen_q),
        )

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch,
        sliding_window: int = 0,
    ) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError(
            "MLA attention uses MLABackend.{store_latent, prefill, decode} directly, not forward()"
        )

    def prepare_metadata(self, batch: Batch) -> None:
        # Lifted from the RDNA4 backend (page-table slicing + cu_seqlens). Adds cu_seqlens_k (full
        # per-seq KV length) which the materialized MLA prefill needs alongside cu_seqlens_q.
        reqs = batch.padded_reqs
        seqlens_q = [req.extend_len for req in reqs]
        seqlens_k = [req.device_len for req in reqs]
        max_seqlen_q = max(seqlens_q)
        max_seqlen_k = max(seqlens_k)
        CPU_KWARGS = {"device": "cpu", "dtype": torch.int32, "pin_memory": True}
        device = self.kvcache.device

        cache_seqlens = torch.tensor(seqlens_k, **CPU_KWARGS).to(device, non_blocking=True)
        if max_seqlen_q == 1:
            cu_seqlens_q = torch.arange(0, len(reqs) + 1, device=device, dtype=torch.int32)
        else:
            cu_seqlens_q = torch.tensor([0] + seqlens_q, **CPU_KWARGS).cumsum_(0)
            cu_seqlens_q = cu_seqlens_q.to(device, non_blocking=True)
        cu_seqlens_k = torch.tensor([0] + seqlens_k, **CPU_KWARGS).cumsum_(0)
        cu_seqlens_k = cu_seqlens_k.to(device, non_blocking=True)

        page_table = get_global_ctx().page_table
        # global page table treats page_size=1; slice + rescale to page indices for the decode kernel.
        new_page_table = torch.stack(
            [page_table[req.table_idx, : max_seqlen_k : self.page_size] for req in reqs]
        )
        if self.page_size > 1:
            new_page_table.div_(self.page_size, rounding_mode="floor")

        batch.attn_metadata = MLAMetadata(
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            page_table=new_page_table,
        )

    # ---- cudagraph capture (DECODE only) -----------------------------------------------------
    # Decode is one token/seq, so the only per-step-varying metadata mla_decode reads is
    # cache_seqlens (latent KV length, +1 each step) and the page table (a row can gain a page).
    # Both live in STATIC int32 buffers the captured graph reads; prepare_for_replay refreshes them
    # in place before g.replay(). The latent STORE (store_latent at batch.out_loc, from the model-
    # level capture buffer) and the decode kernel both run inside the graph; the decode kernel bounds
    # its reads by cache_seqlens, so a fixed max-width page table is fine (stale tail ignored). The
    # latent pool is a fixed tensor (captured by reference). cu_seqlens_* are unused by decode (q-len
    # is always 1) — a static arange placeholder. Mirrors HIPAttnBackend's capture. (MoE-decode must
    # use the graph-safe gather_reduce path, MINISGL_MOE_SCATTER=0 — the scatter atomicAdd is not
    # graph-capturable; see [[splitk-gemm2-modest-win]] / production-serve-config-and-bench.)
    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        dev = self.kvcache.device
        self._cap_max_bs = max(bs_list)
        self._cap_max_pages = (max_seq_len + self.page_size - 1) // self.page_size
        self._cap_cache_seqlens = torch.ones(self._cap_max_bs, dtype=torch.int32, device=dev)
        self._cap_page_table = torch.zeros(
            self._cap_max_bs, self._cap_max_pages, dtype=torch.int32, device=dev
        )
        self._cap_cu_q = torch.arange(self._cap_max_bs + 1, dtype=torch.int32, device=dev)

    def _decode_metadata_static(self, bs: int) -> MLAMetadata:
        return MLAMetadata(
            cache_seqlens=self._cap_cache_seqlens[:bs],
            cu_seqlens_q=self._cap_cu_q[: bs + 1],
            cu_seqlens_k=self._cap_cu_q[: bs + 1],  # unused by decode (placeholder)
            max_seqlen_q=1,
            page_table=self._cap_page_table[:bs],
        )

    def _fill_decode_static(self, batch: Batch) -> None:
        """Refresh the static decode buffers from `batch.padded_reqs` (real rows + dummy padding).
        Runs eager, OUTSIDE the graph; writes the exact int32 tensors the captured kernel reads."""
        reqs = batch.padded_reqs
        bs = len(reqs)
        dev = self.kvcache.device
        seqlens_k = [req.device_len for req in reqs]
        self._cap_cache_seqlens[:bs].copy_(torch.tensor(seqlens_k, dtype=torch.int32, device=dev))
        gpt = get_global_ctx().page_table  # global page_size=1 table
        for i, req in enumerate(reqs):
            npages = (seqlens_k[i] + self.page_size - 1) // self.page_size
            row = gpt[req.table_idx, : npages * self.page_size : self.page_size]
            if self.page_size > 1:
                row = torch.div(row, self.page_size, rounding_mode="floor")
            self._cap_page_table[i, :npages].copy_(row.to(torch.int32))

    def prepare_for_capture(self, batch: Batch) -> None:
        self._fill_decode_static(batch)
        batch.attn_metadata = self._decode_metadata_static(batch.padded_size)

    def prepare_for_replay(self, batch: Batch) -> None:
        self._fill_decode_static(batch)
        batch.attn_metadata = self._decode_metadata_static(batch.padded_size)

    # ---- cudagraph capture (SPEC-DECODE VERIFY) ----------------------------------------------
    # The verify forward stages num_draft+1 query tokens PER req (confirmed + K drafts), so the
    # token count is bs*(K+1) — fixed once K=num_draft is fixed. We capture one graph per bs (number
    # of reqs). The mla_verify kernel reads, per query row, a (seq index, causal kbound); since every
    # req has exactly K+1 query rows, `verify_seq_idx` is a STATIC pattern ([0]*(K+1),[1]*(K+1),...)
    # and `verify_kbound[i*(K+1)+j] = device_len[i] - (K+1) + j + 1` is a cheap per-step fill. Both
    # live in static buffers the captured kernel reads; `prepare_verify_for_replay` refreshes them
    # (+ cache_seqlens + page_table) in place before g.replay(). cu_seqlens_q is unused on this path
    # (verify reads the precomputed indices), kept as an arange*(K+1) placeholder.
    def init_verify_capture(self, max_seq_len: int, bs_list: List[int], num_draft: int) -> None:
        dev = self.kvcache.device
        self._vcap_max_bs = max(bs_list)
        self._vcap_qlen = num_draft + 1
        self._vcap_max_pages = (max_seq_len + self.page_size - 1) // self.page_size
        T = self._vcap_max_bs * self._vcap_qlen
        self._vcap_cache_seqlens = torch.ones(self._vcap_max_bs, dtype=torch.int32, device=dev)
        self._vcap_page_table = torch.zeros(
            self._vcap_max_bs, self._vcap_max_pages, dtype=torch.int32, device=dev
        )
        self._vcap_cu_q = (
            torch.arange(self._vcap_max_bs + 1, dtype=torch.int32, device=dev) * self._vcap_qlen
        )
        self._vcap_kbound = torch.zeros(T, dtype=torch.int32, device=dev)
        # static seq-index pattern: row r belongs to seq r // (K+1).
        self._vcap_seq_idx = (
            torch.arange(T, dtype=torch.int32, device=dev) // self._vcap_qlen
        )

    def _verify_metadata_static(self, bs: int) -> MLAMetadata:
        ql = self._vcap_qlen
        return MLAMetadata(
            cache_seqlens=self._vcap_cache_seqlens[:bs],
            cu_seqlens_q=self._vcap_cu_q[: bs + 1],
            cu_seqlens_k=self._vcap_cu_q[: bs + 1],  # unused by verify (placeholder)
            max_seqlen_q=ql,
            page_table=self._vcap_page_table[:bs],
            verify_seq_idx=self._vcap_seq_idx[: bs * ql],
            verify_kbound=self._vcap_kbound[: bs * ql],
        )

    def _fill_verify_static(self, batch: Batch) -> None:
        """Refresh the static verify buffers from `batch.padded_reqs` (real rows + dummy padding).
        Eager, OUTSIDE the graph. cache_seqlens = device_len; kbound[i,j] = device_len_i-(K+1)+1+j;
        page_table = each seq's page row (stale tail beyond cache_seqlens is ignored by the kernel)."""
        reqs = batch.padded_reqs
        bs = len(reqs)
        ql = self._vcap_qlen
        dev = self.kvcache.device
        dls = torch.tensor([req.device_len for req in reqs], dtype=torch.int32, device=dev)
        self._vcap_cache_seqlens[:bs].copy_(dls)
        # kbound[i, j] = device_len_i - (K+1) + 1 + j  (causal bound for query j of seq i)
        kb = (dls - ql + 1).unsqueeze(1) + torch.arange(ql, dtype=torch.int32, device=dev).unsqueeze(0)
        self._vcap_kbound[: bs * ql].copy_(kb.reshape(-1))
        gpt = get_global_ctx().page_table  # global page_size=1 table
        for i, req in enumerate(reqs):
            npages = (req.device_len + self.page_size - 1) // self.page_size
            row = gpt[req.table_idx, : npages * self.page_size : self.page_size]
            if self.page_size > 1:
                row = torch.div(row, self.page_size, rounding_mode="floor")
            self._vcap_page_table[i, :npages].copy_(row.to(torch.int32))

    def prepare_verify_for_capture(self, batch: Batch) -> None:
        self._fill_verify_static(batch)
        batch.attn_metadata = self._verify_metadata_static(batch.padded_size)

    def prepare_verify_for_replay(self, batch: Batch) -> None:
        self._fill_verify_static(batch)
        batch.attn_metadata = self._verify_metadata_static(batch.padded_size)
