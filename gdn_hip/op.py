"""Python entry point for the native GDN HIP ops (gfx1201).

Loads the AOT-compiled gdn_hip_C extension and registers fake/meta impls so torch.compile/Inductor
treat the ops as opaque (no graph-break) — the same pattern as zaya_cca. Exposes thin wrappers that
enforce the v1 fp32 boundary (kernels are fp32; callers may pass bf16 and get fp32 back).

These ops are framework-agnostic: minisgl's GDN layer and vLLM's qwen_gdn_linear_attn can both call
torch.ops.gdn_hip.* after swapping their fla-Triton calls.
"""
import glob
import os

import torch

_so = glob.glob(os.path.join(os.path.dirname(__file__), "gdn_hip_C*.so"))
if not _so:
    raise ImportError(
        "gdn_hip_C extension not built; run `GPU_ARCHS=gfx1201 python setup.py build_ext --inplace`"
    )
torch.ops.load_library(_so[0])


@torch.library.register_fake("gdn_hip::gdn_decode")
def _gdn_decode_fake(q, k, v, a, b, A_log, dt_bias, ssm_state, state_indices, scale, use_l2norm):
    return v.new_empty((v.shape[0], v.shape[1], v.shape[2]))


@torch.library.register_fake("gdn_hip::gdn_prefill")
def _gdn_prefill_fake(q, k, v, a, b, A_log, dt_bias, cu_seqlens, state_indices,
                      has_initial_state, ssm_state, scale, use_l2norm):
    return v.new_empty((v.shape[0], v.shape[1], v.shape[2]))


@torch.library.register_fake("gdn_hip::gdn_prefill_verify")
def _gdn_prefill_verify_fake(q, k, v, a, b, A_log, dt_bias, cu_seqlens, state_indices,
                             has_initial_state, ssm_state, max_qlen, scale, use_l2norm):
    N = state_indices.shape[0]
    out = v.new_empty((v.shape[0], v.shape[1], v.shape[2]))
    scratch = ssm_state.new_empty((max_qlen, N, v.shape[1], v.shape[2], q.shape[2]))
    return out, scratch


@torch.library.register_fake("gdn_hip::causal_conv1d_fwd_verify")
def _conv_fwd_verify_fake(x, weight, bias, cu_seqlens, state_indices, has_initial_state, conv_state,
                          max_qlen, activation):
    N = state_indices.shape[0]
    out = torch.empty_like(x)
    scratch = conv_state.new_empty((max_qlen, N, x.shape[1], weight.shape[1] - 1))
    return out, scratch


@torch.library.register_fake("gdn_hip::gdn_prefill_chunked")
def _gdn_prefill_chunked_fake(q, k, v, a, b, A_log, dt_bias, cu_seqlens, state_indices,
                              has_initial_state, ssm_state, scale, use_l2norm):
    return v.new_empty((v.shape[0], v.shape[1], v.shape[2]))


@torch.library.register_fake("gdn_hip::gdn_prefill_wmma")
def _gdn_prefill_wmma_fake(q, k, v, a, b, A_log, dt_bias, cu_seqlens, state_indices,
                           has_initial_state, ssm_state, scale, use_l2norm):
    return v.new_empty((v.shape[0], v.shape[1], v.shape[2]))


@torch.library.register_fake("gdn_hip::causal_conv1d_update")
def _conv_update_fake(x, weight, bias, conv_state, state_indices, activation):
    return torch.empty_like(x)


@torch.library.register_fake("gdn_hip::causal_conv1d_fwd")
def _conv_fwd_fake(x, weight, bias, cu_seqlens, state_indices, has_initial_state, conv_state,
                   activation):
    return torch.empty_like(x)


@torch.library.register_fake("gdn_hip::rmsnorm_gated")
def _rmsnorm_gated_fake(x, z, weight, eps):
    return torch.empty_like(x)


@torch.library.register_fake("gdn_hip::rmsnorm_gated_bwd")
def _rmsnorm_gated_bwd_fake(go, x, z, weight, eps):
    return (torch.empty_like(x), torch.empty_like(z),
            torch.empty(x.shape[-1], dtype=torch.float32, device=x.device))


@torch.library.register_fake("gdn_hip::causal_conv1d_bwd")
def _causal_conv1d_bwd_fake(go, x, weight, bias, activation):
    dbias_n = weight.shape[0] if bias is not None else 0
    return (torch.empty_like(x),
            torch.empty(weight.shape[0], weight.shape[1], dtype=torch.float32, device=x.device),
            torch.empty(dbias_n, dtype=torch.float32, device=x.device))


# ---- raw ops ----
gdn_decode = torch.ops.gdn_hip.gdn_decode
gdn_prefill = torch.ops.gdn_hip.gdn_prefill
gdn_prefill_verify = torch.ops.gdn_hip.gdn_prefill_verify
causal_conv1d_fwd_verify = torch.ops.gdn_hip.causal_conv1d_fwd_verify
gdn_prefill_chunked = torch.ops.gdn_hip.gdn_prefill_chunked
gdn_prefill_wmma = torch.ops.gdn_hip.gdn_prefill_wmma
causal_conv1d_update = torch.ops.gdn_hip.causal_conv1d_update
causal_conv1d_fwd = torch.ops.gdn_hip.causal_conv1d_fwd
rmsnorm_gated = torch.ops.gdn_hip.rmsnorm_gated
rmsnorm_gated_bwd = torch.ops.gdn_hip.rmsnorm_gated_bwd
causal_conv1d_bwd = torch.ops.gdn_hip.causal_conv1d_bwd

__all__ = [
    "gdn_decode", "gdn_prefill", "gdn_prefill_verify", "gdn_prefill_chunked", "gdn_prefill_wmma",
    "causal_conv1d_update", "causal_conv1d_fwd", "causal_conv1d_fwd_verify", "rmsnorm_gated",
    "rmsnorm_gated_bwd", "causal_conv1d_bwd",
]
