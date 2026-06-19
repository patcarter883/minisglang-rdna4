"""Compatibility shim for the vendored FLA / mamba GDN Triton kernels.

The kernel sources under ``minisgl/gdn/{fla,mamba}/ops`` are vendored VERBATIM from
vLLM 0.22.69 (flash-linear-attention origin). Their only non-torch/triton imports
were a handful of ``vllm.*`` symbols; those import lines are mechanically rewritten
to pull from this module instead, so minisgl carries the GDN kernels as a
dependency-free copy (the same principle as consuming the W4A8 csrc via
``torch.ops`` rather than importing vLLM).

Keep this surface matched to exactly what the vendored files import. If the kernels
are re-synced from a newer vLLM, re-run the import rewrite and reconcile here.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as tldevice  # == vllm.triton_utils.tldevice

__all__ = [
    "triton",
    "tl",
    "tldevice",
    "current_platform",
    "cdiv",
    "next_power_of_2",
    "num_compute_units",
    "NULL_BLOCK_ID",
    "PAD_SLOT_ID",
]

# --- vllm.v1.attention.backends.utils constants (mamba/ops/causal_conv1d.py) ---
NULL_BLOCK_ID = 0
PAD_SLOT_ID = -1


# --- vllm.utils.math_utils (fla/ops/layernorm_guard.py) ---
def cdiv(a: int, b: int) -> int:
    """Ceiling division — matches vllm.utils.math_utils.cdiv (cdiv(7,3)==3)."""
    return -(-a // b)


def next_power_of_2(n: int) -> int:
    """Smallest power of two >= n; n<=0 -> 1 (matches vllm.utils.math_utils)."""
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


# --- vllm.utils.platform_utils.num_compute_units (fla/ops/layernorm_guard.py) ---
def num_compute_units(device_id: int = 0) -> int:
    """Compute-unit count of the device. On RDNA4/ROCm torch reports the CU count
    as ``multi_processor_count`` — queried at runtime, never hardcoded."""
    return torch.cuda.get_device_properties(device_id).multi_processor_count


# --- vllm.platforms.current_platform (fla/ops/utils.py) ---
class _CurrentPlatform:
    """Minimal stand-in exposing only the predicate the vendored kernels touch.

    fla/ops/utils.py uses this once, at import time:
        device = "cuda" if current_platform.is_cuda_alike() else get_available_device()
    On a ROCm torch build we want "cuda" so ``device_torch_lib`` binds to
    ``torch.cuda`` (the valid namespace on ROCm); the actual vendor (amd/nvidia)
    is detected separately from the live Triton backend.
    """

    @staticmethod
    def is_cuda_alike() -> bool:
        # True for both NVIDIA CUDA and AMD ROCm torch builds (build-time check,
        # so it holds even when imported CPU-only with no GPU visible).
        return torch.version.cuda is not None or torch.version.hip is not None


current_platform = _CurrentPlatform()
