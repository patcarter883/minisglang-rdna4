"""Swap HuggingFace Qwen3.5 gated-delta-net (fla) layers for minisgl's native Triton-free `gdn_hip`
path, so training backprops through `gdn_hip` instead of `fla` — which HANGS on RDNA4 as the gate
opens, and which isn't even installed in the serve image.

Usage (after loading a frozen HF Qwen3.5 model, before training the tap):

    from minisgl.gdn.hf_patch import patch_qwen3_5_gdn
    n = patch_qwen3_5_gdn(model)      # replaces every Qwen3_5GatedDeltaNet in place; returns the count

The shim implements HF's linear-attn forward signature — `(hidden_states, cache_params=None,
attention_mask=None)` — and runs minisgl's differentiable per-sequence native prefill. It is
PREFILL-ONLY: CAM / tap-training runs full-sequence teacher-forced with `use_cache=False`, so
`cache_params` is always None; if a cached (incremental-decode) call arrives it raises rather than
silently mis-computing. Numerics: forward matches fla to ~1e-3 (bf16 + wmma); dL/d_hidden matches fla
autograd at cos>0.9999 (validated in tools/gdn_backward_validate.py Level 2, both recurrent and wmma).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .layer import QwenGatedDeltaNet


def _build_native_from_hf(hf) -> QwenGatedDeltaNet:
    """Construct a minisgl QwenGatedDeltaNet matching the HF layer's geometry and copy its (frozen)
    weights. Mirrors the weight map validated in tools/gdn_backward_validate.py Level 2."""
    dtype = hf.in_proj_qkv.weight.dtype
    device = hf.in_proj_qkv.weight.device
    ms = QwenGatedDeltaNet(
        hidden_size=hf.hidden_size,
        num_k_heads=hf.num_k_heads,
        num_v_heads=hf.num_v_heads,
        head_k_dim=hf.head_k_dim,
        head_v_dim=hf.head_v_dim,
        conv_kernel_size=hf.conv_kernel_size,
        eps=hf.layer_norm_epsilon,
        dtype=dtype,
        device=device,
    )
    with torch.no_grad():
        # HF splits the input projection into (qkv, z) and (b, a); minisgl fuses them as (qkvz) and (ba).
        ms.in_proj_qkvz.weight.copy_(torch.cat([hf.in_proj_qkv.weight, hf.in_proj_z.weight], dim=0))
        ms.in_proj_ba.weight.copy_(torch.cat([hf.in_proj_b.weight, hf.in_proj_a.weight], dim=0))
        ms.conv1d_weight.copy_(hf.conv1d.weight)
        ms.A_log.copy_(hf.A_log.float())
        ms.dt_bias.copy_(hf.dt_bias.float())
        ms.norm.weight.copy_(hf.norm.weight)
        ms.out_proj.weight.copy_(hf.out_proj.weight)
    # inherit the base's frozen/trainable flags (CAM freezes the base; only the tap trains). The native
    # forward still builds a differentiable graph so grad flows THROUGH to hidden_states regardless.
    for p in ms.parameters():
        p.requires_grad_(False)
    return ms


class NativeGDNShim(nn.Module):
    """Drop-in replacement for HF Qwen3_5GatedDeltaNet: same forward signature, native gdn_hip compute.
    Prefill-only (full-sequence, zero initial state) — the CAM/training regime."""

    def __init__(self, hf) -> None:
        super().__init__()
        self.ms = _build_native_from_hf(hf)
        self.layer_idx = getattr(hf, "layer_idx", None)

    def forward(self, hidden_states: torch.Tensor, cache_params=None, attention_mask=None,
                use_cache=False, **kwargs):
        # transformers >=5.13 passes use_cache / cache_position / position_ids etc. to the mixer; this
        # shim is PREFILL-ONLY (teacher-forced, no KV cache), so absorb + ignore them. A truthy cache is
        # still an error (incremental decode is unsupported).
        if cache_params is not None or use_cache:
            raise NotImplementedError(
                "NativeGDNShim is prefill-only (use_cache=False); incremental decode with a cache is "
                "not supported — CAM/tap-training runs full-sequence teacher-forced.")
        # transformers >=5.13 hands the linear-attn mixer fp32 hidden states even on a bf16 base; the native
        # layer's projections keep the base dtype (bf16). Align input to the native weight dtype and return
        # the native (base-dtype) output — the surrounding decoder layer's residual + MLP are base-dtype, so
        # upcasting the output back to fp32 would poison the next linear (float vs bf16 weight).
        _wdt = self.ms.in_proj_qkvz.weight.dtype
        if hidden_states.dtype != _wdt:
            hidden_states = hidden_states.to(_wdt)
        # HF passes [B, T, hidden] (batched, all sequences length T). Run the whole batch through the
        # BATCHED differentiable native prefill — one varlen op call for all B sequences (kills the
        # per-sequence Python loop that made native ~3x slower than the fla-torch fallback). Set
        # GDN_HIP_BATCH_TRAIN=0 to fall back to the per-seq loop (parity/debug).
        if hidden_states.dim() == 3:
            import os
            if os.environ.get("GDN_HIP_BATCH_TRAIN", "1") != "0":
                return self.ms._prefill_train_batch(hidden_states)
            return torch.stack(
                [self.ms._prefill_train_one_seq(hidden_states[i]) for i in range(hidden_states.shape[0])],
                dim=0)
        return self.ms._prefill_train_one_seq(hidden_states)


def patch_qwen3_5_gdn(model) -> int:
    """Replace every Qwen3_5GatedDeltaNet in `model` with a NativeGDNShim (weights copied). Returns the
    number of layers patched. Detection is by class name so it works without importing the HF module."""
    patched = 0
    for parent in model.modules():
        for name, child in list(parent.named_children()):
            if type(child).__name__ == "Qwen3_5GatedDeltaNet":
                setattr(parent, name, NativeGDNShim(child))
                patched += 1
    return patched
