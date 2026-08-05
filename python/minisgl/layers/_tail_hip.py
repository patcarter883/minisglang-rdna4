"""Gate for the native tail_hip elementwise ops (rms_norm / rms_norm_add / silu_and_mul / rope).

On by default; ``MINISGL_TAIL_HIP=0`` forces the torch references (debugging / A-B parity).

Hard-require: when enabled, ``tail_hip`` must import here or the layer modules fail to load — the
user opted in to default-on, so a missing/unbuilt .so is a hard error, not a silent torch fallback.

The kernels are dtype-GENERIC (templated on the scalar type, bf16/fp16/fp32), so ``active()`` no
longer names a dtype. It used to demand bf16 because the kernels were hard-typed ``__hip_bfloat16``,
and that had a cost worth remembering: an fp16 checkpoint (Gemma4) failed the gate on EVERY call and
ran the multi-launch torch decomposition of every RMSNorm and RoPE for the whole serve, with no
error and no log line — the only visible tell was the absence of ``[hip-engage] tail_hip.*``. The
gate now checks what actually matters: that the tensors AGREE on a float dtype the kernels can
instantiate, because they are reinterpret_cast to one scalar type inside.
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
    # Same optionality for the TANH gelu (HF `gelu_pytorch_tanh`), for the same reason: an older
    # baked .so has neither op, and a hard attribute lookup would crash the import for every model.
    _gelu_tanh_and_mul = getattr(tail_hip, "gelu_tanh_and_mul", None)
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

    if _gelu_tanh_and_mul is not None:
        def gelu_tanh_and_mul(*args, **kwargs):
            _engaged("tail_hip.gelu_tanh_and_mul")
            return _gelu_tanh_and_mul(*args, **kwargs)
    else:
        gelu_tanh_and_mul = None


# What the templated kernels can instantiate (TAIL_DISPATCH_FLOAT in tail_kernels.hip). Stated once,
# here, so widening the kernel set is a one-line change on this side rather than a hunt through gates.
_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


def active(*tensors: torch.Tensor | None) -> bool:
    """True when the HIP path should run: enabled, every (non-None) tensor is on the GPU, and they
    share ONE dtype the kernels can instantiate. ``None`` entries are skipped so an optional weight —
    the unweighted RMSNorm has no weight tensor at all — does not disable the gate.

    The device test is not decoration: the ops are registered at ``kCUDA``, so a CPU tensor cannot
    reach them, and the old gate happened to exclude CPU only because CPU tensors in the test suite
    were fp32 and the gate demanded bf16. Widening the dtype set removed that accident.

    Contiguity is handled at the call site via ``.contiguous()`` (a no-op when already contiguous),
    so it is not part of the gate. The per-op engaged() lives in the op shims above, so this gate
    emits no engage line itself."""
    if not ENABLED:
        return False
    live = [t for t in tensors if t is not None]
    if not live or not all(t.is_cuda for t in live):
        return False  # a CPU tensor can never reach a HIP kernel — the ops are registered at kCUDA
    dtypes = {t.dtype for t in live}
    if len(dtypes) != 1:
        return False  # a mixed-dtype call the shared-scalar-type kernels cannot serve
    return dtypes.pop() in _SUPPORTED_DTYPES
