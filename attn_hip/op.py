"""Python entry point for the native flash-attention HIP op (gfx1201).

Loads the AOT-compiled attn_hip_C extension and registers a fake/meta impl so torch.compile/Inductor
treat the op as opaque (no graph-break) — same pattern as gdn_hip / zaya_cca. Framework-agnostic:
minisgl's attention layer and vLLM's backend can both call torch.ops.attn_hip.flash_prefill after
swapping their Triton attention call.

v0: bf16 dense (non-paged) prefill, causal + optional sliding window, GQA. q/k/v are
[seq, heads, head_dim], contiguous.
"""
import glob
import os

import torch

_so = glob.glob(os.path.join(os.path.dirname(__file__), "attn_hip_C*.so"))
if not _so:
    raise ImportError(
        "attn_hip_C extension not built; run `GPU_ARCHS=gfx1201 python setup.py build_ext --inplace`"
    )
torch.ops.load_library(_so[0])


@torch.library.register_fake("attn_hip::flash_prefill")
def _flash_prefill_fake(q, k, v, scale, causal, sliding_window, mask_bias=None):
    return torch.empty_like(q)


flash_prefill = torch.ops.attn_hip.flash_prefill

__all__ = ["flash_prefill"]
