"""moe_w8a16_wmma — fp8-weight × bf16-act grouped MoE WMMA GEMM for gfx1201 (RDNA4).

Keeps OLDMOE=1 bf16-activation correctness at native grouped-WMMA speed by dequanting the fp8 weight
tile to bf16 in-register (no full-stack materialization) and touching only routed experts.
"""
from .op import moe_w8a16_gemm, moe_w8a16_gemm_scatter  # noqa: F401
from .fused import fused_moe_w8a16  # noqa: F401

__all__ = ["moe_w8a16_gemm", "moe_w8a16_gemm_scatter", "fused_moe_w8a16"]
