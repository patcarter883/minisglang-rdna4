# SPDX-License-Identifier: Apache-2.0
"""Native HIP MLA prefill backend for vLLM on gfx1201 (RDNA4).

WHY THIS EXISTS
    vLLM's ROCm MLA *prefill* selector offers exactly two backends — ROCM_AITER_FA and FLASH_ATTN —
    and both bottom out in the CK-tile FMHA (`flash_attn.mha_varlen_fwd` -> `fmha_fwd`). That CK
    kernel is built for CDNA tile shapes and SEGFAULTS on gfx1201 at GLM-4.7-Flash's MLA dims
    (qk_head_dim=256, v_head_dim=256): the worker dies mid-prefill on the first request, taking the
    engine with it. The image already bakes our own `mla_hip` kernels at /opt/kernels and even
    exports VLLM_MLA_HIP=1, but shipped no glue to reach them — so vLLM never used them.

    This module is that glue: an MLAPrefillBackend whose two entry points call
    `mla_hip.mla_prefill` / `mla_hip.mla_prefill_lse`, the validated raw-WMMA RDNA4 kernel.

SCOPE
    bf16 only — the kernel is bf16-typed end to end, so `supported_dtypes` excludes fp16 and the
    selector will (correctly) skip this backend on a fp16 serve rather than silently mis-cast.
    Instantiated for the two MLA shapes the kernel templates cover: DeepSeek (128/64/128) and
    GLM-4.7-Flash (192/64/256).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vllm.v1.attention.backends.mla.prefill.base import MLADimensions, MLAPrefillBackend

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey


class MlaHipPrefillBackend(MLAPrefillBackend):
    """MLA prefill on the native `mla_hip` RDNA4 WMMA kernel."""

    # The kernel reinterprets q/k/v as __hip_bfloat16; fp16 would be a silent bit-reinterpret.
    supported_dtypes = [torch.bfloat16]

    # Must match the template instantiations in mla_rocm/mla_prefill_kernels.hip. Adding a shape
    # here without adding the matching LAUNCH() there turns a clean selector skip into a
    # TORCH_CHECK at first prefill.
    supported_mla_dimensions = [
        # DeepSeek-V2/V3: qk 128+64=192, v 128
        MLADimensions(qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128),
        # GLM-4.7-Flash: qk 192+64=256, v 256
        MLADimensions(qk_nope_head_dim=192, qk_rope_head_dim=64, v_head_dim=256),
    ]

    @staticmethod
    def get_name() -> str:
        return "MLA_HIP"

    @classmethod
    def is_available(cls) -> bool:
        try:
            import mla_hip  # noqa: F401
        except ImportError:
            return False
        # mla_prefill_lse is the newer of the two ops; a stale baked mla_hip has only mla_prefill,
        # which cannot serve the chunked-context path. Require both rather than fail at runtime.
        return hasattr(mla_hip, "mla_prefill") and hasattr(mla_hip, "mla_prefill_lse")

    @classmethod
    def supports_compute_capability(cls, device_capability) -> bool:
        from vllm.platforms import current_platform

        return current_platform.is_rocm()

    def supports_quant_output(self, quant_key: "QuantKey") -> bool:
        # No fused quantized-output epilogue in the kernel; let vLLM run its post-quant pass.
        return False

    # ---- internals -------------------------------------------------------------------------

    @staticmethod
    def _cu(t: torch.Tensor) -> torch.Tensor:
        # The kernel indexes cu_seqlens as int32; vLLM builds these as int32 already, but a
        # metadata change upstream would otherwise surface as garbage offsets, not an error.
        return t if t.dtype == torch.int32 else t.to(torch.int32)

    def _run(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        causal: bool,
        return_softmax_lse: bool,
    ):
        import mla_hip

        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        args = (
            q,
            k,
            v,
            self._cu(cu_seqlens_q),
            self._cu(cu_seqlens_k),
            float(self.scale),
            int(causal),
            0,  # sliding_window: MLA is full-context
            int(max_seqlen_q),
        )
        if return_softmax_lse:
            out, lse = mla_hip.mla_prefill_lse(*args)
            return out, lse
        return mla_hip.mla_prefill(*args)

    # ---- MLAPrefillBackend interface -------------------------------------------------------

    def run_prefill_new_tokens(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        return_softmax_lse: bool,
        out: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        assert output_scale is None, (
            "MLA_HIP prefill has no fused quantized-output epilogue; "
            "supports_quant_output() returns False so vLLM should not pass output_scale."
        )
        md = self._prefill_metadata
        # New tokens attend causally over themselves: cu_seqlens_k == cu_seqlens_q, so the kernel's
        # prefix_len (= k_len - q_len) is 0 and the causal mask is the plain lower triangle.
        res = self._run(
            q,
            k,
            v,
            cu_seqlens_q=md.query_start_loc,
            cu_seqlens_k=md.query_start_loc,
            max_seqlen_q=md.max_query_len,
            causal=True,
            return_softmax_lse=return_softmax_lse,
        )
        if out is None:
            return res
        attn_out = res[0] if return_softmax_lse else res
        out.copy_(attn_out)
        return (out, res[1]) if return_softmax_lse else out

    def run_prefill_context_chunk(
        self,
        chunk_idx: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        md = self._prefill_metadata
        assert md.chunked_context is not None
        # Context is already-computed KV, so it is UNMASKED (causal=False) and every new token sees
        # all of it. vLLM folds this partial into the causal one via merge_attn_states, which needs
        # the LSE — hence mla_prefill_lse is mandatory on this path.
        return self._run(
            q,
            k,
            v,
            cu_seqlens_q=md.query_start_loc,
            cu_seqlens_k=md.chunked_context.cu_seq_lens[chunk_idx],
            max_seqlen_q=md.max_query_len,
            causal=False,
            return_softmax_lse=True,
        )
