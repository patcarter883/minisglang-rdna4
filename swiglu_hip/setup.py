"""Build the native fused SwiGLU HIP op for gfx1201 (RDNA4) — Task B (#18).

torch.ops.swiglu_hip.fused_swiglu as one AOT .so. Output-parallel bf16/fp16 GEMV (no WMMA, no
Triton) for the occupancy-starved M<=2 dense shared expert. Mirrors the tail_hip / gdn_hip recipe.

    GPU_ARCHS=gfx1201 python setup.py build_ext --inplace
"""
import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

TARGET_ARCH = os.environ.get("GPU_ARCHS", "gfx1201").split(";")[0]

setup(
    name="swiglu_hip",
    version="0.1.0",
    ext_modules=[
        CUDAExtension(
            name="swiglu_hip_C",
            sources=["bindings.cpp", "swiglu_kernels.hip"],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17", "-fPIC"],
                "nvcc": ["-O3", "-std=c++17", f"--offload-arch={TARGET_ARCH}",
                         "-Wno-unused-result", "-Wno-unused-variable"],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
