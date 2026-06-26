"""Gate for the native tail_hip elementwise ops (rms_norm / rms_norm_add / silu_and_mul / rope).

On by default; ``MINISGL_TAIL_HIP=0`` forces the torch references (debugging / A-B parity).

Hard-require: when enabled, ``tail_hip`` must import here or the layer modules fail to load — the
user opted in to default-on, so a missing/unbuilt .so is a hard error, not a silent torch fallback.
The per-call ``.bfloat16()`` guard below is NOT an opt-out: it only routes the (rare, never on the
bf16 serve hot path) off-dtype tensors to the torch ref, since the kernels are bf16-only. On the
real serve path activations + weights are bf16 (the model is built under ``torch_dtype(config.dtype)``),
so the HIP op is always the one taken.
"""
from __future__ import annotations

import os

import torch

ENABLED = os.environ.get("MINISGL_TAIL_HIP", "1") != "0"

if ENABLED:
    import tail_hip  # noqa: F401  loads tail_hip_C + registers torch.ops.tail_hip.*


def active(*tensors: torch.Tensor) -> bool:
    """True when the HIP path should run: enabled AND every tensor is bf16 (the kernels' only
    supported dtype). Contiguity is handled at the call site via ``.contiguous()`` (a no-op when
    already contiguous), so it is not part of the gate."""
    return ENABLED and all(t.dtype == torch.bfloat16 for t in tensors)
