"""Python entry point for the native "tail" elementwise HIP ops (gfx1201).

Loads tail_hip_C and registers fakes so torch.compile steps over them. Framework-agnostic:
torch.ops.tail_hip.{rms_norm, rms_norm_add, silu_and_mul, rope}. Conventions match minisgl-rdna4
(plus_one RMSNorm gain, gated silu over [...,2D], NeoX partial RoPE with cat(cos,sin) cache).
"""
import glob
import os

import torch

_so = glob.glob(os.path.join(os.path.dirname(__file__), "tail_hip_C*.so"))
if not _so:
    raise ImportError(
        "tail_hip_C not built; run `GPU_ARCHS=gfx1201 python setup.py build_ext --inplace`"
    )
torch.ops.load_library(_so[0])


@torch.library.register_fake("tail_hip::rms_norm")
def _rms_norm_fake(x, w, eps, plus_one):
    return torch.empty_like(x)


@torch.library.register_fake("tail_hip::rms_norm_add")
def _rms_norm_add_fake(x, residual, w, eps, plus_one):
    return torch.empty_like(x)


@torch.library.register_fake("tail_hip::silu_and_mul")
def _silu_and_mul_fake(x):
    shape = list(x.shape)
    shape[-1] //= 2
    return x.new_empty(shape)


@torch.library.register_fake("tail_hip::rope")
def _rope_fake(x, pos, cache, head_size, rotary_dim):
    return torch.empty_like(x)


rms_norm = torch.ops.tail_hip.rms_norm
rms_norm_add = torch.ops.tail_hip.rms_norm_add
silu_and_mul = torch.ops.tail_hip.silu_and_mul
rope = torch.ops.tail_hip.rope

__all__ = ["rms_norm", "rms_norm_add", "silu_and_mul", "rope"]
