"""Quantization support for minisgl-rdna4.

Framework-agnostic engine-side plumbing (quant-config detection, the LinearMethod
protocol, weight routing) lives here. ALL quantized-kernel calls + weight-layout
conversion go through `kernels.py` (the swappable provider) so the parallel custom
kernel framework can drop in without touching layers/models/loader.
"""

from .config import QuantConfig
from .method import LinearMethod, UnquantizedLinearMethod, W4A8LinearMethod, create_linear_method

__all__ = [
    "QuantConfig",
    "LinearMethod",
    "UnquantizedLinearMethod",
    "W4A8LinearMethod",
    "create_linear_method",
]
