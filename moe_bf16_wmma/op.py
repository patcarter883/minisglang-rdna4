"""Python entry point for the unquantized (bf16/fp16) grouped MoE WMMA GEMM (gfx1201).

Loads moe_bf16_C and registers fake impls so torch.compile treats the ops as opaque. Ops:
  moe_bf16_gemm              -> C[P,OUT]   (gemm1-style, allocating; NOT graph-capturable)
  moe_bf16_gemm_scatter      -> C[M,OUT]   (gemm2-style, allocating; NOT graph-capturable)
  moe_bf16_gemm_out          -> writes caller's C[P,OUT]   (graph-capturable)
  moe_bf16_gemm_scatter_out  -> atomic-adds into caller's C[M,OUT] fp32 (graph-capturable; zero first)
"""
import glob
import os

import torch

_so = glob.glob(os.path.join(os.path.dirname(__file__), "moe_bf16_C*.so"))
if not _so:
    raise ImportError(
        "moe_bf16_C not built; run `GPU_ARCHS=gfx1201 python setup.py build_ext --inplace`"
    )
torch.ops.load_library(_so[0])


@torch.library.register_fake("moe_bf16::moe_bf16_gemm")
def _gemm_fake(A, w, sorted_token_ids, expert_ids, num_tokens_post_padded, topk_weights,
               top_k, block_m, num_valid_tokens, BN, mul_weight):
    return A.new_empty((sorted_token_ids.shape[0], w.shape[1]))


@torch.library.register_fake("moe_bf16::moe_bf16_gemm_scatter")
def _scatter_fake(A, w, sorted_token_ids, expert_ids, num_tokens_post_padded, topk_weights,
                  M, top_k, block_m, num_valid_tokens, BN, out_top_k):
    return torch.empty((M, w.shape[1]), dtype=torch.float32, device=A.device)


@torch.library.register_fake("moe_bf16::moe_bf16_gemm_out")
def _gemm_out_fake(A, w, sorted_token_ids, expert_ids, num_tokens_post_padded, topk_weights, out,
                   top_k, block_m, num_valid_tokens, BN, mul_weight):
    return None


@torch.library.register_fake("moe_bf16::moe_bf16_gemm_scatter_out")
def _scatter_out_fake(A, w, sorted_token_ids, expert_ids, num_tokens_post_padded, topk_weights, out,
                      top_k, block_m, num_valid_tokens, BN, out_top_k):
    return None


moe_bf16_gemm = torch.ops.moe_bf16.moe_bf16_gemm
moe_bf16_gemm_scatter = torch.ops.moe_bf16.moe_bf16_gemm_scatter
moe_bf16_gemm_out = torch.ops.moe_bf16.moe_bf16_gemm_out
moe_bf16_gemm_scatter_out = torch.ops.moe_bf16.moe_bf16_gemm_scatter_out

__all__ = [
    "moe_bf16_gemm",
    "moe_bf16_gemm_scatter",
    "moe_bf16_gemm_out",
    "moe_bf16_gemm_scatter_out",
]
