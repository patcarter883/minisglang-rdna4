"""Out-of-tree ``PluggableLayer`` subclass that routes the Qwen Gated-Delta-Net linear-attention
compute (conv1d + gated-delta-rule + gated RMSNorm) through the native gdn_hip HIP ops on gfx1201
— with ZERO edits to vLLM source (qwen_gdn_linear_attn.py / mamba_utils.py).

Mechanism (verified): GDN compute is a MODEL layer (``QwenGatedDeltaNetAttention._forward_core``),
NOT an AttentionImpl (mamba backends are metadata-only). ``QwenGatedDeltaNetAttention`` is a
``PluggableLayer``; ``PluggableLayer.__new__`` swaps the in-tree class for a subclass registered
under its ``__name__``. So we subclass + ``@PluggableLayer.register_oot`` and override methods only.

IMPORTANT: ``__new__`` returns the subclass instance but Python runs the BASE ``__init__`` — this
subclass may ONLY override methods, NEVER add ``__init__`` state. The overrides read existing attrs
only, so this is safe.

Spec-decode (multi-query-per-seq) and warmup (no attn_metadata) keep the stock fla-Triton path;
gdn_hip covers the non-spec single-token decode + varlen-recurrent / WMMA-chunked prefill only.

``get_state_dtype`` forces the GDN conv+ssm state to fp32 (the gdn_hip kernels read/write the state
in place as fp32) — scoped to Qwen GDN only; this replaces the old mamba_utils.py global patch.
"""
from __future__ import annotations

import os

import torch

from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

import gdn_hip  # noqa: F401  canonical rdna4-hip-kernels pkg: loads gdn_hip_C + registers fakes

logger = init_logger(__name__)

# WMMA matrix-core chunked prefill is the default GDN prefill when gdn_hip is on (4.8-5.9x faster
# than the scalar recurrent prefill); set VLLM_GDN_HIP_RECURRENT_ONLY=1 to force the recurrent path.
VLLM_GDN_HIP_RECURRENT_ONLY = os.environ.get("VLLM_GDN_HIP_RECURRENT_ONLY", "0") == "1"


def _gdn_select_prefill(layer):
    """Pick the GDN prefill op: WMMA matrix-core chunked prefill (4.8-5.9x faster) for 128-dim
    Qwen3.5/3.6 GDN heads, else the scalar recurrent prefill (also forced by
    VLLM_GDN_HIP_RECURRENT_ONLY=1). Both share the fp32 op boundary and the same state cache."""
    if (not VLLM_GDN_HIP_RECURRENT_ONLY
            and getattr(layer, "head_k_dim", 0) == 128
            and getattr(layer, "head_v_dim", 0) == 128):
        return gdn_hip.gdn_prefill_wmma
    return gdn_hip.gdn_prefill


@PluggableLayer.register_oot(name="QwenGatedDeltaNetAttention")
class QwenGdnHipAttention(QwenGatedDeltaNetAttention):
    def _forward_core(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor,
    ):
        # gdn_hip native path: route the non-spec prefill/decode GDN compute through the AOT HIP
        # ops (no Triton JIT). Spec-decode keeps the fla-Triton path — gdn_hip is single-token-per-seq
        # decode / varlen-recurrent prefill only.
        raw = get_forward_context().attn_metadata
        am = raw[self.prefix] if isinstance(raw, dict) else None
        if am is None:
            # WARMUP / memory-profiling pass (no attn_metadata): the output is discarded, so do NOT
            # run the fla-Triton GDN prefill just to feed the profiler — on ROCm the native GDN prefill
            # is ALWAYS Triton (flashinfer/cutedsl are CUDA-only), and its JIT/autotune is the exact
            # ~30-min cost gdn_hip exists to eliminate (worse: a fresh GDN shape misses the warm cache).
            # Return the pre-allocated core_attn_out buffer; the REAL forward runs gdn_hip, whose
            # workspace is smaller than Triton's, so the profiled peak (hence KV sizing) stays safe.
            return core_attn_out
        if am.spec_sequence_masks is None:
            return self._forward_core_gdn_hip(
                mixed_qkv=mixed_qkv,
                b=b,
                a=a,
                core_attn_out=core_attn_out,
                attn_metadata=am,
            )
        return super()._forward_core(mixed_qkv, b, a, core_attn_out)

    # NOTE: fp32 recurrent state is NOT forced here. A per-layer get_state_dtype override only
    # affects the KV-cache-spec path and NOT the attention-block-size planner
    # (get_mamba_state_dtype_from_config), which desyncs them and trips the MambaSpec page_size
    # assert. It is forced at the shared calculator in register.py::_force_fp32_gdn_state instead.

    def _forward_core_gdn_hip(
        self,
        *,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor,
        attn_metadata: GDNAttentionMetadata,
    ):
        """Native-HIP GDN core (no Triton JIT). Mirrors the fla-Triton ``_forward_core`` non-spec
        path exactly — conv1d + gated-delta-rule + the gated RMSNorm output — but every op is an
        AOT-compiled ``torch.ops.gdn_hip.*`` kernel. fp32 at the op boundary: the conv/ssm state
        are fp32 (forced in get_state_dtype when gdn_hip is on) and the bf16 activations are cast
        ``.float()`` here; ``core_attn_out`` is written back in the layer dtype by the caller's
        output projection.

        Drops ``fused_post_conv_prep`` + ``l2norm_fwd``: the gdn_hip recurrence folds the q/k
        L2-norm and the g/beta (softplus/sigmoid) computation in, consuming raw ``a, b``.
        """
        non_spec_query_start_loc = attn_metadata.non_spec_query_start_loc
        non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor  # noqa: E501
        has_initial_state = attn_metadata.has_initial_state
        self_kv_cache = self.kv_cache
        # conv_state must be (..., dim, width-1) for the conv kernels (DS layout); SD needs a
        # transpose. gdn_hip's conv kernels take that [slots, C, W-1] layout directly.
        conv_state = (
            self_kv_cache[0]
            if is_conv_state_dim_first()
            else self_kv_cache[0].transpose(-1, -2)
        )
        ssm_state = self_kv_cache[1]

        # gdn_hip's kernels index state with contiguous [slots,...] strides, but vLLM's paged mamba
        # cache view is non-contiguous. The old path .contiguous()-ed the WHOLE cache every step (all
        # ~262 slots) — a 2.75GB fp32 ssm copy measured at ~64% of decode. Instead materialize ONLY
        # the slots this batch touches. PREFILL keeps the full-shadow path (state_indices is per-seq
        # and tokens segment via cu_seqlens, so slot-gather would alias duplicate token->slot maps).
        # DECODE (single-token-per-seq, unique slot each) gathers just the active slots and scatters
        # back — the ~64% decode win. (fla-Triton reads the strided view directly; this matches it.)

        num_actual_tokens = attn_metadata.num_actual_tokens

        mixed_qkv = mixed_qkv[:num_actual_tokens]
        b = b[:num_actual_tokens]
        a = a[:num_actual_tokens]

        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        ).float()
        state_indices = non_spec_state_indices_tensor[:num_actual_tokens].long()  # type: ignore[index]
        scale = self.head_k_dim**-0.5
        # A_log is fp32 by construction; dt_bias defaults fp32 but the loader may cast it to the
        # model dtype — float() both so the fp32 gdn_hip kernels never see a bf16 input.
        A_log = self.A_log.float()
        dt_bias = self.dt_bias.float()

        n_k = self.num_k_heads // self.tp_size
        n_v = self.num_v_heads // self.tp_size

        def _split_conv_qkv(conv_out: torch.Tensor):
            q, k, v = conv_out.split(
                [self.key_dim // self.tp_size,
                 self.key_dim // self.tp_size,
                 self.value_dim // self.tp_size],
                dim=-1,
            )
            q = q.reshape(-1, n_k, self.head_k_dim).contiguous()
            k = k.reshape(-1, n_k, self.head_k_dim).contiguous()
            v = v.reshape(-1, n_v, self.head_v_dim).contiguous()
            return q, k, v

        if attn_metadata.num_prefills > 0:
            # Varlen recurrent path (covers any decode-len-1 seqs in the same batch via cu_seqlens).
            # Full-cache shadow (see note above): prefill is not the decode bottleneck.
            conv_state_k = conv_state if conv_state.is_contiguous() else conv_state.contiguous()
            ssm_state_k = ssm_state if ssm_state.is_contiguous() else ssm_state.contiguous()
            assert has_initial_state is not None
            x = mixed_qkv.float().contiguous()  # prefill keeps fp32 acts (not the decode hot path)
            conv_out = gdn_hip.causal_conv1d_fwd(
                x,
                conv_weights,
                None,
                non_spec_query_start_loc,  # cu_seqlens int32
                state_indices,
                has_initial_state.to(torch.uint8),
                conv_state_k,
                1,  # SiLU
            )
            q, k, v = _split_conv_qkv(conv_out)
            core = _gdn_select_prefill(self)(
                q, k, v, a.float(), b.float(), A_log, dt_bias,
                non_spec_query_start_loc, state_indices,
                has_initial_state.to(torch.uint8), ssm_state_k, scale, 1,
            )
            if conv_state_k.data_ptr() != conv_state.data_ptr():
                conv_state.copy_(conv_state_k)
            if ssm_state_k.data_ptr() != ssm_state.data_ptr():
                ssm_state.copy_(ssm_state_k)
        else:
            # Pure single-token-per-seq decode: the STRIDE-AWARE gdn_hip kernels index the (possibly
            # non-contiguous) paged mamba cache IN PLACE via state_indices + the tensor's real strides —
            # no gather, no scatter, no contiguous shadow (exactly what fla-Triton does). state_indices
            # carries the real slot ids (slot 0 == NULL_BLOCK_ID, handled in-kernel).
            # FUSED cast-free decode: ONE launch (conv + l2norm + recurrence, stride-aware paged cache,
            # raw core out) replaces causal_conv1d_update + 3 splits + gdn_decode — the launch-storm
            # cut that closes the residual gap to fla. fp16 acts (scalar_t), state_t from ssm_state;
            # conv_weights stays fp32 (`const float*`). Returns the RAW core; caller applies self.norm.
            core = gdn_hip.ops.gdn_decode_conv(
                mixed_qkv.contiguous(), conv_weights, None, conv_state,
                a, b, A_log, dt_bias, ssm_state, state_indices, 1, scale, 1,
            )

        core_attn_out[:num_actual_tokens] = core.to(core_attn_out.dtype)
        return
