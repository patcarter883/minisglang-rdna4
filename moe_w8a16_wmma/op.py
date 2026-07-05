"""Python entry point for the W8A16 (fp8-weight × bf16-act) grouped MoE WMMA GEMM (gfx1201).

  moe_w8a16_gemm         -> C[P,OUT] bf16   (gemm1-style, sorted-padded rows; src = offs//top_k)
  moe_w8a16_gemm_scatter -> C[M,OUT] fp32   (gemm2-style, topk-weighted scatter-accumulate)
Weights are e4m3 bytes (uint8, layout (E,OUT,IN)) + per-output-channel f32 scale (E,OUT).
"""
import glob
import os

import torch

_so = glob.glob(os.path.join(os.path.dirname(__file__), "moe_w8a16_C*.so"))
if not _so:
    raise ImportError(
        "moe_w8a16_C not built; run `GPU_ARCHS=gfx1201 python setup.py build_ext --inplace`"
    )
torch.ops.load_library(_so[0])


@torch.library.register_fake("moe_w8a16::moe_w8a16_gemm")
def _gemm_fake(A, w_fp8, w_scales, sorted_token_ids, expert_ids, num_tokens_post_padded,
               topk_weights, top_k, block_m, num_valid_tokens, BN, mul_weight):
    return A.new_empty((sorted_token_ids.shape[0], w_fp8.shape[1]))


@torch.library.register_fake("moe_w8a16::moe_w8a16_gemm_scatter")
def _scatter_fake(A, w_fp8, w_scales, sorted_token_ids, expert_ids, num_tokens_post_padded,
                  topk_weights, M, top_k, block_m, num_valid_tokens, BN, out_top_k):
    return torch.empty((M, w_fp8.shape[1]), dtype=torch.float32, device=A.device)


moe_w8a16_gemm = torch.ops.moe_w8a16.moe_w8a16_gemm
moe_w8a16_gemm_scatter = torch.ops.moe_w8a16.moe_w8a16_gemm_scatter

__all__ = ["moe_w8a16_gemm", "moe_w8a16_gemm_scatter"]
