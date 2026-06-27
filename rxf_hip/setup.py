"""Build the native RXF W4A8 HIP ops for gfx1201 (RDNA4).

torch.ops.rxf_hip.{rotate_quant_int8, linear, moe_gemm} as one AOT .so. Uses rocwmma
int8 fragments (gfx12 i32_16x16x16_iu8 WMMA) for the dense GEMM. Mirrors the attn_hip /
gdn_hip recipe (rocwmma headers ship in the ROCm include path).

    GPU_ARCHS=gfx1201 python setup.py build_ext --inplace
"""
import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

TARGET_ARCH = os.environ.get("GPU_ARCHS", "gfx1201").split(";")[0]

setup(
    name="rxf_hip",
    version="0.1.0",
    ext_modules=[
        CUDAExtension(
            name="rxf_hip_C",
            sources=["bindings.cpp", "rxf_kernels.hip"],
            include_dirs=["/opt/rocm-7.2.1/include"],  # rocwmma headers
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17", "-fPIC"],
                "nvcc": ["-O3", "-std=c++17", f"--offload-arch={TARGET_ARCH}",
                         "-Wno-unused-result", "-Wno-unused-variable"],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
