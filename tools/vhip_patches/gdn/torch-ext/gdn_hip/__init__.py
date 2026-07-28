"""gdn_hip — native HIP kernels for the Gated Delta Net linear-attention path on gfx1201 (RDNA4).

A framework-agnostic Torch extension (shared by minisgl-rdna4 and vllm-gfx1201), AOT-compiled once
(no Triton JIT/autotune) to replace the fla-Triton GDN kernels (chunk_gated_delta_rule + the decode
SSM kernel + causal_conv1d + rmsnorm_gated) that hang on RDNA4.

This is the kernel-builder / `kernels`-hub packaging of the kernel formerly vendored at
minisgl-rdna4/gdn_hip. Load it with `kernels.get_kernel("<repo-id>")` or import the built package
directly. Ops are exposed both as callables here and (for torch.compile) as opaque custom ops with
registered fake/meta impls.

Forward ops: gdn_decode, gdn_prefill[_verify|_chunked|_wmma], causal_conv1d_[fwd|update|fwd_verify],
rmsnorm_gated. Native backward ops: rmsnorm_gated_bwd, causal_conv1d_bwd. The differentiable training
entry points (recompute-backward) live in `.autograd` and are re-exported below.
"""
import torch

from ._ops import add_op_namespace_prefix, ops

# ---- fake/meta impls: keep torch.compile/Inductor from graph-breaking on the opaque custom ops ----


@torch.library.register_fake(add_op_namespace_prefix("gdn_decode"))
def _gdn_decode_fake(q, k, v, a, b, A_log, dt_bias, ssm_state, state_indices, scale, use_l2norm):
    return v.new_empty((v.shape[0], v.shape[1], v.shape[2]))


@torch.library.register_fake(add_op_namespace_prefix("gdn_decode_gated"))
def _gdn_decode_gated_fake(q, k, v, a, b, A_log, dt_bias, ssm_state, state_indices, z, norm_weight,
                           eps, scale, use_l2norm):
    return v.new_empty((v.shape[0], v.shape[1], v.shape[2]))


@torch.library.register_fake(add_op_namespace_prefix("gdn_decode_conv_gated"))
def _gdn_decode_conv_gated_fake(mixed_qkv, conv_weight, conv_bias, conv_state, a, b, A_log, dt_bias,
                                ssm_state, state_indices, z, norm_weight, eps, activation, scale,
                                use_l2norm):
    B, HV, V = a.shape[0], a.shape[1], ssm_state.shape[2]
    return mixed_qkv.new_empty((B, HV, V))


@torch.library.register_fake(add_op_namespace_prefix("gdn_prefill"))
def _gdn_prefill_fake(q, k, v, a, b, A_log, dt_bias, cu_seqlens, state_indices,
                      has_initial_state, ssm_state, scale, use_l2norm):
    return v.new_empty((v.shape[0], v.shape[1], v.shape[2]))


@torch.library.register_fake(add_op_namespace_prefix("gdn_prefill_verify"))
def _gdn_prefill_verify_fake(q, k, v, a, b, A_log, dt_bias, cu_seqlens, state_indices,
                             has_initial_state, ssm_state, max_qlen, scale, use_l2norm,
                             slot_table=None, num_accepted=None):
    out = v.new_empty((v.shape[0], v.shape[1], v.shape[2]))
    if slot_table is not None:   # fused publish: state goes straight to the cache, no scratch
        return out, ssm_state.new_empty((0,))
    N = state_indices.shape[0]
    scratch = ssm_state.new_empty((max_qlen, N, v.shape[1], v.shape[2], q.shape[2]))
    return out, scratch


@torch.library.register_fake(add_op_namespace_prefix("causal_conv1d_fwd_verify"))
def _conv_fwd_verify_fake(x, weight, bias, cu_seqlens, state_indices, has_initial_state, conv_state,
                          max_qlen, activation, slot_table=None, num_accepted=None):
    out = torch.empty_like(x)
    if slot_table is not None:
        return out, conv_state.new_empty((0,))
    N = state_indices.shape[0]
    scratch = conv_state.new_empty((max_qlen, N, x.shape[1], weight.shape[1] - 1))
    return out, scratch


@torch.library.register_fake(add_op_namespace_prefix("gdn_prefill_chunked"))
def _gdn_prefill_chunked_fake(q, k, v, a, b, A_log, dt_bias, cu_seqlens, state_indices,
                              has_initial_state, ssm_state, scale, use_l2norm):
    return v.new_empty((v.shape[0], v.shape[1], v.shape[2]))


@torch.library.register_fake(add_op_namespace_prefix("gdn_prefill_wmma"))
def _gdn_prefill_wmma_fake(q, k, v, a, b, A_log, dt_bias, cu_seqlens, state_indices,
                           has_initial_state, ssm_state, scale, use_l2norm):
    return v.new_empty((v.shape[0], v.shape[1], v.shape[2]))


@torch.library.register_fake(add_op_namespace_prefix("causal_conv1d_update"))
def _conv_update_fake(x, weight, bias, conv_state, state_indices, activation):
    return torch.empty_like(x)


@torch.library.register_fake(add_op_namespace_prefix("causal_conv1d_fwd"))
def _conv_fwd_fake(x, weight, bias, cu_seqlens, state_indices, has_initial_state, conv_state,
                   activation):
    return torch.empty_like(x)


@torch.library.register_fake(add_op_namespace_prefix("rmsnorm_gated"))
def _rmsnorm_gated_fake(x, z, weight, eps):
    return torch.empty_like(x)


@torch.library.register_fake(add_op_namespace_prefix("rmsnorm_gated_bwd"))
def _rmsnorm_gated_bwd_fake(go, x, z, weight, eps):
    return (torch.empty_like(x), torch.empty_like(z),
            torch.empty(x.shape[-1], dtype=torch.float32, device=x.device))


@torch.library.register_fake(add_op_namespace_prefix("causal_conv1d_bwd"))
def _causal_conv1d_bwd_fake(go, x, weight, bias, activation):
    dbias_n = weight.shape[0] if bias is not None else 0
    return (torch.empty_like(x),
            torch.empty(weight.shape[0], weight.shape[1], dtype=torch.float32, device=x.device),
            torch.empty(dbias_n, dtype=torch.float32, device=x.device))


# ---- raw forward + native-backward ops ----
gdn_decode = ops.gdn_decode
gdn_decode_gated = ops.gdn_decode_gated
gdn_decode_conv_gated = ops.gdn_decode_conv_gated
gdn_prefill = ops.gdn_prefill
gdn_prefill_verify = ops.gdn_prefill_verify
causal_conv1d_fwd_verify = ops.causal_conv1d_fwd_verify
gdn_prefill_chunked = ops.gdn_prefill_chunked
gdn_prefill_wmma = ops.gdn_prefill_wmma
causal_conv1d_update = ops.causal_conv1d_update
causal_conv1d_fwd = ops.causal_conv1d_fwd
rmsnorm_gated = ops.rmsnorm_gated
rmsnorm_gated_bwd = ops.rmsnorm_gated_bwd
causal_conv1d_bwd = ops.causal_conv1d_bwd

# ---- differentiable training entry points (recompute-backward; opt-in, inference path untouched) ----
from .autograd import (  # noqa: E402,F401
    causal_conv1d_batch_train,
    causal_conv1d_fwd_train,
    enable,
    gdn_prefill_batch_train,
    gdn_prefill_train,
    gdn_prefill_wmma_batch_train,
    gdn_prefill_wmma_train,
    ref_causal_conv1d_fwd,
    ref_gdn_prefill_core,
    ref_rmsnorm_gated,
    rmsnorm_gated_train,
)

__all__ = [
    # forward ops
    "gdn_decode", "gdn_decode_gated", "gdn_decode_conv_gated", "gdn_prefill", "gdn_prefill_verify", "gdn_prefill_chunked", "gdn_prefill_wmma",
    "causal_conv1d_update", "causal_conv1d_fwd", "causal_conv1d_fwd_verify", "rmsnorm_gated",
    # native backward ops
    "rmsnorm_gated_bwd", "causal_conv1d_bwd",
    # differentiable training API
    "gdn_prefill_train", "gdn_prefill_wmma_train", "causal_conv1d_fwd_train", "rmsnorm_gated_train",
    "gdn_prefill_batch_train", "gdn_prefill_wmma_batch_train", "causal_conv1d_batch_train", "enable",
    # pure-torch references
    "ref_gdn_prefill_core", "ref_causal_conv1d_fwd", "ref_rmsnorm_gated",
]
