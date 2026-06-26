"""attn_hip — native HIP/rocwmma flash-attention for gfx1201 (RDNA4), Triton-free.

A standalone, framework-agnostic torch.ops extension shared by minisgl-rdna4 and vllm-gfx1201:
AOT-compiled once (no Triton JIT/autotune), it replaces the Triton attention kernel on the serve
path. Import `op` to load the .so and register the ops as torch.ops.attn_hip.*.
"""
from .op import flash_prefill  # noqa: F401
