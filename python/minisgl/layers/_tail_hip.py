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

# The canonical rdna4-hip-kernels `tail_hip` package exposes its ops as module-level callables
# (rms_norm / rms_norm_add / silu_and_mul / rope); the ops themselves register under the
# build-unique torch.ops.tail_hip_C namespace. Re-export the callables here so the layer modules
# have a single import site for the native tail ops (and the MINISGL_TAIL_HIP gate lives in one place).
if ENABLED:
    import tail_hip

    from minisgl._hip_engage import engaged as _engaged

    # Wrap each native op in a thin per-op shim that fires `engaged("tail_hip.<op>")` on first call, so
    # the [hip-engage] manifest distinguishes WHICH tail kernel fired (rms_norm vs rope vs silu_and_mul
    # ...) instead of a single generic `tail_hip` line. The shim just tags + delegates (no dtype/shape
    # handling — the callers already gate via active() and pass .contiguous() bf16 tensors).
    _silu_and_mul = tail_hip.silu_and_mul
    # gelu_and_mul is OPTIONAL: the canonical rdna4-hip-kernels `tail` kernel is silu-only, so a
    # hard `tail_hip.gelu_and_mul` crashes the import on that image even for models that never use
    # gelu (e.g. ZAYA: silu experts + a plain torch F.gelu router). Bind it if present; activation.py
    # falls back to torch F.gelu when this is None, so a gelu-tail model still runs (just not native).
    _gelu_and_mul = getattr(tail_hip, "gelu_and_mul", None)
    _rms_norm = tail_hip.rms_norm
    _rms_norm_add = tail_hip.rms_norm_add
    _rope = tail_hip.rope

    def silu_and_mul(*args, **kwargs):
        _engaged("tail_hip.silu_and_mul")
        return _silu_and_mul(*args, **kwargs)

    def rms_norm(*args, **kwargs):
        _engaged("tail_hip.rms_norm")
        return _rms_norm(*args, **kwargs)

    def rms_norm_add(*args, **kwargs):
        _engaged("tail_hip.rms_norm_add")
        return _rms_norm_add(*args, **kwargs)

    def rope(*args, **kwargs):
        _engaged("tail_hip.rope")
        return _rope(*args, **kwargs)

    if _gelu_and_mul is not None:
        def gelu_and_mul(*args, **kwargs):
            _engaged("tail_hip.gelu_and_mul")
            return _gelu_and_mul(*args, **kwargs)
    else:
        gelu_and_mul = None


def active(*tensors: torch.Tensor) -> bool:
    """True when the HIP path should run: enabled AND every tensor is bf16 (the kernels' only
    supported dtype). Contiguity is handled at the call site via ``.contiguous()`` (a no-op when
    already contiguous), so it is not part of the gate. The per-op engaged() now lives in the op
    shims above, so this gate no longer emits a generic `tail_hip` engage line."""
    return ENABLED and all(t.dtype == torch.bfloat16 for t in tensors)
