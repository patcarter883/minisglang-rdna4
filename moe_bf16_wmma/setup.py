# Build: GPU_ARCHS=gfx1201 python setup.py build_ext --inplace  (inside the combined ROCm image)
import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

archs = os.environ.get("GPU_ARCHS", "gfx1201")
here = os.path.dirname(os.path.abspath(__file__))

setup(
    name="moe_bf16_wmma",
    ext_modules=[
        CUDAExtension(
            name="moe_bf16_C",
            sources=["bindings.cpp", "moe_bf16_kernels.hip"],
            include_dirs=[here, "/opt/rocm-7.2.1/include"],  # rocwmma headers (flash_wmma.h)
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": ["-O3", "-std=c++17", f"--offload-arch={archs}",
                         "-Wno-unused-result", "-Wno-unused-variable"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
