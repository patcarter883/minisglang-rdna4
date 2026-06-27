"""Python entry point for the native fused SwiGLU HIP op (gfx1201, Task B #18).

Loads swiglu_hip_C and registers a fake so torch.compile / cudagraph capture step over it.
Framework-agnostic: torch.ops.swiglu_hip.fused_swiglu(x, w_gate_up, w_down) -> out, matching
F.linear(silu_and_mul(F.linear(x, w_gate_up)), w_down) for unquantized bf16/fp16 weights.
"""
import glob
import os

import torch

_so = glob.glob(os.path.join(os.path.dirname(__file__), "swiglu_hip_C*.so"))
if not _so:
    raise ImportError(
        "swiglu_hip_C not built; run `GPU_ARCHS=gfx1201 python setup.py build_ext --inplace`"
    )
torch.ops.load_library(_so[0])


@torch.library.register_fake("swiglu_hip::fused_swiglu")
def _fused_swiglu_fake(x, w_gate_up, w_down):
    return torch.empty_like(x)


fused_swiglu = torch.ops.swiglu_hip.fused_swiglu

__all__ = ["fused_swiglu"]
