"""Build the native W8A8-fp8 grouped-MoE HIP ops for gfx1201 (RDNA4).

torch.ops.w8a8_fp8_wmma.{mmq_w8a8_moe_gemm, mmq_w8a8_moe_gemm1_silu, mmq_w8a8_moe_gemm_scatter,
mmq_w8a8_moe_gather_reduce} as one AOT .so. W8A8 = fp8 (e4m3) weights with a per-output-channel
fp32 scale + fp8 activations. Full feature parity with w4a8_fp8_wmma (same WMMA core,
__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12 intrinsics — NO rocwmma headers needed).
Mirrors the rxf_hip / gdn_hip AOT recipe.

    GPU_ARCHS=gfx1201 python setup.py build_ext --inplace
"""
import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

TARGET_ARCH = os.environ.get("GPU_ARCHS", "gfx1201").split(";")[0]

setup(
    name="w8a8_fp8_wmma",
    version="0.1.0",
    ext_modules=[
        CUDAExtension(
            name="w8a8_fp8_wmma_C",
            sources=["bindings.cpp", "w8a8_moe_kernel.hip"],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17", "-fPIC"],
                "nvcc": ["-O3", "-std=c++17", f"--offload-arch={TARGET_ARCH}",
                         "-Wno-unused-result", "-Wno-unused-variable"],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
