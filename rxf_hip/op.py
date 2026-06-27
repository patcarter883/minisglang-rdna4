"""Python entry point for the native RXF W4A8 HIP ops (gfx1201).

Loads rxf_hip_C and registers fakes so torch.compile steps over them. Framework-agnostic:
torch.ops.rxf_hip.{rotate_quant_int8, linear, moe_gemm}. RXF = W4(NL codebook)-A8(int8)
with a fixed block-diagonal Hadamard rotation (cancels in the dot, spreads activation
outliers). Numerics match vllm-gfx1201/paroquant_rotation/rxf_kernels.py (the parity oracle).
"""
import glob
import os

import torch

_so = glob.glob(os.path.join(os.path.dirname(__file__), "rxf_hip_C*.so"))
if not _so:
    raise ImportError(
        "rxf_hip_C not built; run `GPU_ARCHS=gfx1201 python setup.py build_ext --inplace`"
    )
torch.ops.load_library(_so[0])


@torch.library.register_fake("rxf_hip::rotate_quant_int8")
def _rotate_quant_fake(x, span):
    q = torch.empty_like(x, dtype=torch.int8)
    scale = x.new_empty(x.shape[:-1], dtype=torch.float32)
    return q, scale


@torch.library.register_fake("rxf_hip::linear")
def _linear_fake(q, a_scale, w_packed, w_scale, nl, bias):
    return q.new_empty((q.shape[0], w_packed.shape[0]), dtype=torch.bfloat16)


@torch.library.register_fake("rxf_hip::moe_gemm")
def _moe_gemm_fake(q, a_scale, w_packed, w_scale, nl, sorted_ids, expert_ids,
                   num_tokens_post_padded, top_k, block_m, num_valid_tokens):
    return q.new_empty((sorted_ids.shape[0], w_packed.shape[1]), dtype=torch.bfloat16)


@torch.library.register_fake("rxf_hip::moe_gemv")
def _moe_gemv_fake(q, a_scale, w_packed, w_scale, nl, sorted_ids, expert_ids,
                   num_tokens_post_padded, top_k, block_m, num_valid_tokens):
    return q.new_empty((sorted_ids.shape[0], w_packed.shape[1]), dtype=torch.bfloat16)


@torch.library.register_fake("rxf_hip::moe_gemm_scatter")
def _moe_gemm_scatter_fake(q, a_scale, w_packed, w_scale, nl, sorted_ids, expert_ids,
                           num_tokens_post_padded, topk_weights, out_scatter,
                           top_k, block_m, num_valid_tokens):
    return None


@torch.library.register_fake("rxf_hip::moe_gather_reduce")
def _moe_gather_reduce_fake(out2, sorted_ids, topk_weights, num_tokens_post_padded,
                            num_tokens, top_k, num_valid_tokens):
    return out2.new_empty((num_tokens, out2.shape[1]), dtype=torch.float32)


rotate_quant_int8 = torch.ops.rxf_hip.rotate_quant_int8
linear = torch.ops.rxf_hip.linear
moe_gemm = torch.ops.rxf_hip.moe_gemm
moe_gemv = torch.ops.rxf_hip.moe_gemv
moe_gemm_scatter = torch.ops.rxf_hip.moe_gemm_scatter
moe_gather_reduce = torch.ops.rxf_hip.moe_gather_reduce

# Default IQ4-NL integer codebook (matches rxf_kernels.py _NL_DEFAULT). int8 [16].
NL_DEFAULT = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]

__all__ = ["rotate_quant_int8", "linear", "moe_gemm", "moe_gemv", "moe_gemm_scatter",
           "moe_gather_reduce", "NL_DEFAULT"]
