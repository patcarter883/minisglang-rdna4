"""Minisgl-local split-K W4A8 gemm2 SCATTER kernel for gfx1201 (Task A #17):
torch.ops.moe_splitk_hip.moe_gemm_splitk_scatter."""
from .op import moe_gemm_splitk_scatter

__all__ = ["moe_gemm_splitk_scatter"]
