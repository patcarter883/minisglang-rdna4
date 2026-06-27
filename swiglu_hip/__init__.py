"""Native fused SwiGLU HIP op for gfx1201 (Task B #18): torch.ops.swiglu_hip.fused_swiglu."""
from .op import fused_swiglu

__all__ = ["fused_swiglu"]
