# SPDX-License-Identifier: Apache-2.0
"""Native HIP MLA DECODE backend for vLLM on gfx1201 (RDNA4).

WHY THIS EXISTS
    On ROCm the only MLA decode backend vLLM can pick is TRITON_MLA, and TritonMLAMetadataBuilder
    never declares `query_len_support` — so it inherits SINGLE_ONLY, i.e. one query token per
    sequence. Speculative decoding verifies 1+K tokens per sequence in a single decode step, so any
    MLA + spec config (GLM-4.7-Flash MTP or EAGLE3) dies during cudagraph capture on

        assert m.max_query_len <= self.reorder_batch_threshold   # decode only

    That is not a tuning problem: the Triton MLA decode kernel takes one seq_len per *request* and
    has nowhere to put a per-draft-token causal bound, so MLA spec decode is simply unavailable.

    Our mla_hip package already ships the kernel that closes this: `mla_verify` is absorbed
    MULTI-QUERY decode over the paged latent, taking a per-query sequence index and a per-query
    causal bound — exactly the missing piece. This backend declares QueryLenSupport.UNIFORM and
    routes single-query steps to `mla_decode` and spec-verify steps to `mla_verify`.

SCOPE
    bf16 latent cache (mla_hip is bf16-typed; the fp8 variants exist but are not wired here yet, so
    `supported_kv_cache_dtypes` deliberately omits fp8 rather than silently mis-reading the cache).
    Set VLLM_MLA_HIP_DECODE=0 to leave decode on TRITON_MLA and keep only the prefill backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch

from vllm.config.cache import CacheDType
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonBackend,
    MLACommonDecodeMetadata,
    MLACommonImpl,
    MLACommonMetadata,
    MLACommonMetadataBuilder,
    QueryLenSupport,
)
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionLayer,
    AttentionType,
    MultipleOf,
)


@dataclass
class MlaHipDecodeMetadata(MLACommonDecodeMetadata):
    """Adds the per-query indices `mla_verify` needs for a spec-verify step.

    Both are None on an ordinary 1-token-per-request decode, which selects `mla_decode`.
    """

    q_seq_idx: torch.Tensor | None = None  # [num_decode_tokens] int32: row -> block_table row
    q_kbound: torch.Tensor | None = None  # [num_decode_tokens] int32: row -> causal context length


class MlaHipMetadataBuilder(MLACommonMetadataBuilder[MLACommonMetadata]):
    # UNIFORM (not VARLEN): vLLM hands us equal query lengths per request, which is what the
    # arange/repeat_interleave index construction below assumes.
    query_len_support: ClassVar[QueryLenSupport] = QueryLenSupport.UNIFORM
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def _build_decode(
        self,
        block_table_tensor: torch.Tensor,
        seq_lens_device: torch.Tensor,
        max_seq_len: int,
        query_start_loc_cpu: torch.Tensor,
        query_start_loc_device: torch.Tensor,
        num_decode_tokens: int,
        dcp_tot_seq_lens_device: torch.Tensor | None,
    ) -> MlaHipDecodeMetadata:
        num_reqs = seq_lens_device.shape[0]
        q_seq_idx = q_kbound = None

        if num_reqs > 0 and num_decode_tokens > num_reqs:
            # Spec verify: 1+K query rows per request. Derive the row->request mapping from
            # query_start_loc rather than from num_decode_tokens // num_reqs — under cudagraph
            # replay num_decode_tokens is PADDED up to a captured size, so that division silently
            # yields the wrong query length and every index after it is garbage (which is exactly
            # how this first showed up: drafts were accepted but the text degenerated).
            dev = seq_lens_device.device
            qsl = query_start_loc_device
            rows = torch.arange(num_decode_tokens, device=dev, dtype=torch.int32)
            # right=True on the exclusive ends gives the request each row falls in; padding rows
            # past the last real token clamp onto the final request and are discarded downstream.
            seq = torch.searchsorted(qsl[1:].contiguous(), rows, right=True)
            seq = seq.clamp_(max=num_reqs - 1)
            starts = qsl[seq]
            q_lens = qsl[seq + 1] - starts
            within = rows - starts
            # seq_lens already counts this step's new tokens, so query j of a request may attend to
            # (seq_len - q_len + j + 1) keys; the draft tokens after it stay masked.
            q_seq_idx = seq.to(torch.int32)
            q_kbound = (
                seq_lens_device.to(torch.int32)[seq] - q_lens.to(torch.int32) + within + 1
            ).to(torch.int32)

        return MlaHipDecodeMetadata(
            block_table=block_table_tensor,
            seq_lens=seq_lens_device,
            dcp_tot_seq_lens=dcp_tot_seq_lens_device,
            q_seq_idx=q_seq_idx,
            q_kbound=q_kbound,
        )


class MlaHipBackend(MLACommonBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["auto", "bfloat16"]

    @staticmethod
    def get_name() -> str:
        return "MLA_HIP"

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        # kv_lora_rank + qk_rope_head_dim. mla_hip hardcodes a 64-wide rope tail.
        return [576]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(16)]

    @classmethod
    def supports_block_size(cls, block_size: int | None) -> bool:
        return block_size is None or block_size % 16 == 0

    @staticmethod
    def get_kv_cache_stride_order(include_num_layers_dimension: bool = False) -> tuple[int, ...]:
        if include_num_layers_dimension:
            return (1, 0, 2, 3)
        return (0, 1, 2)

    @staticmethod
    def get_impl_cls() -> type["MlaHipImpl"]:
        return MlaHipImpl

    @staticmethod
    def get_builder_cls() -> type["MlaHipMetadataBuilder"]:
        return MlaHipMetadataBuilder

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return True


class MlaHipImpl(MLACommonImpl[MLACommonMetadata]):
    # We do not produce a decode LSE; that is only needed for decode context parallelism.
    can_return_lse_for_decode: bool = False

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        **mla_args,
    ) -> None:
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            **mla_args,
        )
        if any([alibi_slopes, sliding_window, logits_soft_cap]):
            raise NotImplementedError(
                "MlaHipImpl does not support alibi_slopes, sliding_window or logits_soft_cap"
            )
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("MlaHipImpl supports decoder self-attention only")

        import mla_hip

        self._mla_decode = mla_hip.mla_decode
        self._mla_verify = mla_hip.mla_verify

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert kv_c_and_k_pe_cache.numel() > 0
        decode = attn_metadata.decode
        assert decode is not None

        if type(q) is tuple:
            q = torch.cat(q, dim=-1)
        assert isinstance(q, torch.Tensor)
        q = q.contiguous()

        # [num_blocks, page_size, kv_lora_rank + qk_rope_head_dim] — already the layout the kernel
        # indexes, so no head-dim unsqueeze/reshape (and no copy) is needed here.
        cache = kv_c_and_k_pe_cache
        block_table = decode.block_table.to(torch.int32)

        if getattr(decode, "q_seq_idx", None) is None:
            out = self._mla_decode(
                q, cache, block_table, decode.seq_lens.to(torch.int32), float(self.scale), 0
            )
        else:
            # Spec verify: 1+K query rows per request, each with its own causal bound.
            out = self._mla_verify(
                q, cache, block_table, decode.q_seq_idx, decode.q_kbound, float(self.scale), 0
            )
        return out, None
