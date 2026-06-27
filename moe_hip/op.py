"""Python entry for the native HIP moe_align op (gfx1201).

torch.ops.moe_hip.moe_align: drop-in for vLLM moe_align_block_size(topk_ids, block_size, num_experts,
None, pad_sorted_ids=True). Groups routed tokens per expert, pads each expert's run to a multiple of
block_size with the sentinel = topk_ids.numel(). Returns (sorted_ids, expert_ids, num_tokens_post_pad).
    topk_ids:[M, top_k] int32  ->  sorted_ids:[P] int32, expert_ids:[cdiv(P,bs)] int32, ntp:[1] int32
"""
import glob
import os

import torch

_so = glob.glob(os.path.join(os.path.dirname(__file__), "moe_hip_C*.so"))
if not _so:
    raise ImportError("moe_hip_C not built; run `GPU_ARCHS=gfx1201 python setup.py build_ext --inplace`")
torch.ops.load_library(_so[0])


def _padded_len(numel: int, num_experts: int, block_size: int) -> int:
    P = numel + num_experts * (block_size - 1)
    P = ((P + block_size - 1) // block_size) * block_size
    if numel < num_experts:
        P = min(numel * block_size, P)
    return P


@torch.library.register_fake("moe_hip::moe_align")
def _fake(topk_ids, num_experts, block_size):
    numel = topk_ids.numel()
    P = _padded_len(numel, num_experts, block_size)
    nblk = (P + block_size - 1) // block_size
    opt = dict(dtype=torch.int32, device=topk_ids.device)
    return (torch.empty(P, **opt), torch.empty(nblk, **opt), torch.empty(1, **opt))


def moe_align(topk_ids: torch.Tensor, num_experts: int, block_size: int):
    return torch.ops.moe_hip.moe_align(topk_ids, num_experts, block_size)


__all__ = ["moe_align"]
