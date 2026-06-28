"""Python entry point for the native W8A8-fp8 grouped-MoE HIP ops (gfx1201).

Loads w8a8_fp8_wmma_C and registers fakes so torch.compile / cudagraph capture steps over
them. Framework-agnostic: torch.ops.w8a8_fp8_wmma.{mmq_w8a8_moe_gemm, mmq_w8a8_moe_gemm1_silu,
mmq_w8a8_moe_gemm_scatter, mmq_w8a8_moe_gather_reduce}.

W8A8 = fp8 (e4m3) weights with a PER-OUTPUT-CHANNEL fp32 scale + fp8 activations. Strict
simplification of W4A8 (w4a8_fp8_wmma): identical WMMA core / activation quant / scatter /
gather-reduce plumbing; only the B load (contiguous fp8 byte copy, no nibble unpack) and the
scale fold (per-N channel in the epilogue, not per-K-group in the loop) differ. No zeros, no
group scale, no weight_is_e2m1.

ABI (spec/W8A8_KERNEL_SPEC.md): x fp16 (M,K); w_fp8 (E,N,K) e4m3 (uint8); scales (E,N) f32;
routing int32; scatter output fp32 (M,N) pre-zeroed. kernel ids: 6=wmma, 7=gemv.

The thin wrappers below resolve a descriptive kernel name ("wmma"/"gemv") to the opaque
int the ABI carries (kernel_names.h MoeKernel: Wmma=6, Gemv=7) — no caller writes the int.
"""
import glob
import os
from typing import Union

import torch

_so = glob.glob(os.path.join(os.path.dirname(__file__), "w8a8_fp8_wmma_C*.so"))
if not _so:
    raise ImportError(
        "w8a8_fp8_wmma_C not built; run `GPU_ARCHS=gfx1201 python setup.py build_ext --inplace`"
    )
torch.ops.load_library(_so[0])


@torch.library.register_fake("w8a8_fp8_wmma::mmq_w8a8_moe_gemm")
def _moe_gemm_fake(x, w_fp8, scales, sorted_token_ids, expert_ids,
                   num_tokens_post_padded, top_k, block_m, kernel):
    # (P, N) where P = padded sorted-token rows, N = w_fp8.size(1) output channels.
    return x.new_empty((sorted_token_ids.shape[0], w_fp8.shape[1]), dtype=torch.float16)


@torch.library.register_fake("w8a8_fp8_wmma::mmq_w8a8_moe_gemm1_silu")
def _moe_gemm1_silu_fake(x, w_fp8, scales, sorted_token_ids, expert_ids,
                         num_tokens_post_padded, top_k, block_m, kernel):
    # Fused gate|up two-slab GEMM + silu_and_mul -> (P, inter); w_fp8.size(1) == 2*inter.
    inter = w_fp8.shape[1] // 2
    return x.new_empty((sorted_token_ids.shape[0], inter), dtype=torch.float16)


@torch.library.register_fake("w8a8_fp8_wmma::mmq_w8a8_moe_gemm_scatter")
def _moe_gemm_scatter_fake(x, w_fp8, scales, sorted_token_ids, expert_ids,
                           num_tokens_post_padded, topk_weights, output,
                           top_k, block_m, kernel):
    # In-place atomicAdd into the pre-zeroed fp32 (M, N) output; returns nothing.
    return None


@torch.library.register_fake("w8a8_fp8_wmma::mmq_w8a8_moe_gather_reduce")
def _moe_gather_reduce_fake(out2, sorted_token_ids, topk_weights,
                            num_tokens_post_padded, top_k):
    # Verbatim from w4a8: gather padded expert rows and topk-reduce -> (num_tokens, N).
    num_tokens = out2.shape[0] // top_k
    return out2.new_empty((num_tokens, out2.shape[1]), dtype=torch.float32)


# Descriptive kernel name -> opaque ABI int (kernel_names.h MoeKernel). Callers pass a
# name; the int never appears above the torch boundary. wmma = prefill/gemm2, gemv = decode.
_KERNEL_IDS = {"wmma": 6, "gemv": 7}


def _resolve_kernel(kernel: Union[str, int]) -> int:
    if isinstance(kernel, str):
        try:
            return _KERNEL_IDS[kernel]
        except KeyError:
            raise ValueError(
                f"unknown w8a8 moe kernel {kernel!r}; expected one of {sorted(_KERNEL_IDS)}"
            )
    return int(kernel)


def mmq_w8a8_moe_gemm(x, w_fp8, scales, sorted_token_ids, expert_ids,
                      num_tokens_post_padded, top_k, block_m, kernel):
    """Grouped W8A8 GEMM -> (P, N) fp16. kernel: "wmma" (prefill/gemm2) or "gemv" (decode)."""
    return torch.ops.w8a8_fp8_wmma.mmq_w8a8_moe_gemm(
        x, w_fp8, scales, sorted_token_ids, expert_ids, num_tokens_post_padded,
        top_k, block_m, _resolve_kernel(kernel))


def mmq_w8a8_moe_gemm1_silu(x, w_fp8, scales, sorted_token_ids, expert_ids,
                            num_tokens_post_padded, top_k, block_m, kernel="wmma"):
    """Fused gemm1 ([gate|up]) + silu_and_mul -> (P, inter) fp16. wmma kernel only."""
    return torch.ops.w8a8_fp8_wmma.mmq_w8a8_moe_gemm1_silu(
        x, w_fp8, scales, sorted_token_ids, expert_ids, num_tokens_post_padded,
        top_k, block_m, _resolve_kernel(kernel))


def mmq_w8a8_moe_gemm_scatter(x, w_fp8, scales, sorted_token_ids, expert_ids,
                              num_tokens_post_padded, topk_weights, output,
                              top_k, block_m, kernel):
    """Fused gemm2 + topk-weight + atomic scatter into pre-zeroed (M, N) fp32 `output`."""
    torch.ops.w8a8_fp8_wmma.mmq_w8a8_moe_gemm_scatter(
        x, w_fp8, scales, sorted_token_ids, expert_ids, num_tokens_post_padded,
        topk_weights, output, top_k, block_m, _resolve_kernel(kernel))


def mmq_w8a8_moe_gather_reduce(out2, sorted_token_ids, topk_weights,
                               num_tokens_post_padded, top_k):
    """Contention-free reduce: gemm2 (P,N) fp16 -> topk-weighted (M, N) fp32."""
    return torch.ops.w8a8_fp8_wmma.mmq_w8a8_moe_gather_reduce(
        out2, sorted_token_ids, topk_weights, num_tokens_post_padded, top_k)


__all__ = [
    "mmq_w8a8_moe_gemm", "mmq_w8a8_moe_gemm1_silu", "mmq_w8a8_moe_gemm_scatter",
    "mmq_w8a8_moe_gather_reduce",
]
