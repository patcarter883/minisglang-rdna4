"""Build the native flash-attention HIP custom op for gfx1201 (RDNA4).

Builds torch.ops.attn_hip.flash_prefill as ONE AOT-compiled .so — no per-shape Triton JIT/autotune.
Mirrors the proven gdn_hip / w4a8_fp8_wmma cpp_extension recipe. Uses rocwmma (header-only, ships in
the ROCm include path) for the gfx12 WMMA fragment layout.

Usage (inside the combined ROCm image; CPU-only, no GPU needed to compile):
    GPU_ARCHS=gfx1201 python setup.py build_ext --inplace
"""
import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

TARGET_ARCH = os.environ.get("GPU_ARCHS", "gfx1201").split(";")[0]
# Optional extra preprocessor defines for A/B experiments, e.g. ATTN_DEFINES="SCALAR_PV=1".
_extra_defs = [f"-D{d}" for d in os.environ.get("ATTN_DEFINES", "").split() if d]

setup(
    name="attn_hip",
    version="0.1.0",
    ext_modules=[
        CUDAExtension(
            name="attn_hip_C",
            sources=["bindings.cpp", "attn_kernels.hip"],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17", "-fPIC"],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    f"--offload-arch={TARGET_ARCH}",
                    "-Wno-unused-result",
                    "-Wno-unused-variable",
                ] + _extra_defs,
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
