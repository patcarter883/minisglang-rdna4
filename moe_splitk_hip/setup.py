"""Build the minisgl-local split-K W4A8 gemm2 SCATTER op for gfx1201 (RDNA4) — Task A (#17).

torch.ops.moe_splitk_hip.moe_gemm_splitk_scatter as one AOT .so. Forked from vllm-gfx1201
w4a8_fp8_wmma (v5 tiled WMMA + act-quant) with a split_k grid.z for decode occupancy. Mirrors the
tail_hip / gdn_hip recipe.

    GPU_ARCHS=gfx1201 python setup.py build_ext --inplace
"""
import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

TARGET_ARCH = os.environ.get("GPU_ARCHS", "gfx1201").split(";")[0]

setup(
    name="moe_splitk_hip",
    version="0.1.0",
    ext_modules=[
        CUDAExtension(
            name="moe_splitk_hip_C",
            sources=["bindings.cpp", "moe_splitk_kernels.hip"],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17", "-fPIC"],
                "nvcc": ["-O3", "-std=c++17", f"--offload-arch={TARGET_ARCH}",
                         "-Wno-unused-result", "-Wno-unused-variable"],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
