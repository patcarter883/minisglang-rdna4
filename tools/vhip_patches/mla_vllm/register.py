# SPDX-License-Identifier: Apache-2.0
"""vLLM general-plugin entry point for the native HIP MLA prefill backend.

Registered as `mla_hip` in the `vllm.general_plugins` group, so it activates via
VLLM_PLUGINS=...,mla_hip. Honors VLLM_MLA_HIP (default "1"; the vllm24-hip image already exports
it) — set VLLM_MLA_HIP=0 to fall back to vLLM's own selector.
"""

from __future__ import annotations

import os

_REGISTERED = False


def register(verbose: bool = True) -> bool:
    """Put MlaHipPrefillBackend at the front of the ROCm MLA-prefill priority list.

    Returns True if registered (or already registered), False if disabled/unavailable.
    """
    global _REGISTERED
    if _REGISTERED:
        return True
    if os.environ.get("VLLM_MLA_HIP", "1") != "1":
        if verbose:
            print("[mla_hip] disabled via VLLM_MLA_HIP=0")
        return False

    from vllm.platforms import current_platform

    if not current_platform.is_rocm():
        if verbose:
            print("[mla_hip] not ROCm — skipping")
        return False

    from vllm.v1.attention.backends.mla.prefill import selector
    from vllm.v1.attention.backends.mla.prefill.registry import (
        MLAPrefillBackendEnum,
        register_mla_prefill_backend,
    )

    from mla_vllm.prefill import MlaHipPrefillBackend

    if not MlaHipPrefillBackend.is_available():
        if verbose:
            print(
                "[mla_hip] mla_hip kernel package not importable (or too old — needs "
                "mla_prefill_lse); leaving vLLM's MLA prefill selection alone"
            )
        return False

    register_mla_prefill_backend(
        MLAPrefillBackendEnum.CUSTOM, "mla_vllm.prefill.MlaHipPrefillBackend"
    )

    # Prepend rather than replace: validate_configuration() still gates us on dtype/MLA dims, so an
    # unsupported shape (or a fp16 serve) cleanly falls through to vLLM's own list instead of
    # hard-failing. NOTE that fallback lands back on the CK FMHA that segfaults on gfx1201 — the
    # boot log line below is what tells you which backend actually won.
    _orig_priorities = selector._get_mla_prefill_backend_priorities

    def _priorities_with_mla_hip(device_capability):
        return [MLAPrefillBackendEnum.CUSTOM, *_orig_priorities(device_capability)]

    selector._get_mla_prefill_backend_priorities = _priorities_with_mla_hip
    _REGISTERED = True

    if verbose:
        print("[mla_hip] registered MlaHipPrefillBackend as MLA prefill backend CUSTOM (bf16; "
              "DeepSeek 128/64/128 + GLM-4.7-Flash 192/64/256)")

    _register_decode(verbose)
    return True


def _register_decode(verbose: bool = True) -> bool:
    """Put MlaHipBackend ahead of TRITON_MLA in the ROCm MLA *decode* priority list.

    This is what makes MLA + speculative decoding possible at all here: TRITON_MLA is
    QueryLenSupport.SINGLE_ONLY, so GLM MTP/EAGLE3 assert out during cudagraph capture. Ours is
    UNIFORM, backed by mla_hip.mla_verify. VLLM_MLA_HIP_DECODE=0 opts back out.
    """
    if os.environ.get("VLLM_MLA_HIP_DECODE", "1") != "1":
        if verbose:
            print("[mla_hip] decode backend disabled via VLLM_MLA_HIP_DECODE=0 (TRITON_MLA)")
        return False

    from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
    from vllm.platforms import rocm as rocm_platform

    register_backend(AttentionBackendEnum.CUSTOM, "mla_vllm.decode.MlaHipBackend")

    _orig = rocm_platform._get_backend_priorities

    def _priorities(use_mla, use_sparse, use_kv_connector=False):
        backends = _orig(use_mla, use_sparse, use_kv_connector)
        if use_mla and not use_sparse:
            return [AttentionBackendEnum.CUSTOM, *backends]
        return backends

    rocm_platform._get_backend_priorities = _priorities
    if verbose:
        print("[mla_hip] registered MlaHipBackend as MLA decode backend CUSTOM "
              "(QueryLenSupport.UNIFORM — unblocks MLA spec decode)")
    return True
