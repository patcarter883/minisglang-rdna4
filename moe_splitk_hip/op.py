"""Python entry point for the minisgl-local split-K W4A8 gemm2 SCATTER op (gfx1201, Task A #17).

Loads moe_splitk_hip_C and registers a fake. torch.ops.moe_splitk_hip.moe_gemm_splitk_scatter
writes the pre-zeroed fp32 (M,N) accumulator in place (atomic scatter over experts AND the split_k
K-slices), matching w4a8_fp8_wmma.mmq_fp8_moe_gemm_scatter at split_k=1.
"""
import glob
import os

import torch

_so = glob.glob(os.path.join(os.path.dirname(__file__), "moe_splitk_hip_C*.so"))
if not _so:
    raise ImportError(
        "moe_splitk_hip_C not built; run `GPU_ARCHS=gfx1201 python setup.py build_ext --inplace`"
    )
torch.ops.load_library(_so[0])


@torch.library.register_fake("moe_splitk_hip::moe_gemm_splitk_scatter")
def _splitk_scatter_fake(x, w_packed, scales, w_zeros, sorted_token_ids, expert_ids,
                         num_tokens_post_padded, topk_weights, output, top_k, block_m, split_k):
    return None


moe_gemm_splitk_scatter = torch.ops.moe_splitk_hip.moe_gemm_splitk_scatter

__all__ = ["moe_gemm_splitk_scatter"]
