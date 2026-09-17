"""LOCAL build of mla_hip_C for greening WITHOUT the Nix/kernel-builder toolchain.

Compiles the SAME torch-ext/torch_binding.cpp + mla_rocm/*.hip that kernel-builder builds, using
torch's cpp_extension + hipcc (the proven minisgl recipe). It vendors local/registration.h so the
TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ...) binding compiles here exactly as it does under
kernel-builder (which generates its own registration.h). Defines ROCM_KERNEL so the ops.impl(...)
block registers.

Run from the package root (mla/):
    GPU_ARCHS=gfx1201 python local/setup.py build_ext --inplace
then move the produced mla_hip_C*.so into torch-ext/mla_hip/ (local/build_local.sh does this).

The Hub build path is `nix build` via flake.nix — this file is only for local validation.
"""
import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

TARGET_ARCH = os.environ.get("GPU_ARCHS", "gfx1201").split(";")[0]
ROCM_INCLUDE = os.environ.get("ROCM_INCLUDE", "/opt/rocm/include")  # rocwmma headers (mla_prefill)
_extra_defs = [f"-D{d}" for d in os.environ.get("MLA_DEFINES", "").split() if d]

setup(
    name="mla_hip_C",
    version="0.1.0",
    ext_modules=[
        CUDAExtension(
            name="mla_hip_C",
            sources=[
                "torch-ext/torch_binding.cpp",
                "mla_rocm/mla_kernels.hip",
                "mla_rocm/mla_prefill_kernels.hip",
            ],
            include_dirs=[os.path.abspath("torch-ext"), os.path.abspath("local"), ROCM_INCLUDE],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17", "-fPIC", "-DROCM_KERNEL"],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    f"--offload-arch={TARGET_ARCH}",
                    "-DROCM_KERNEL",
                    "-Wno-unused-result",
                    "-Wno-unused-variable",
                ] + _extra_defs,
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
