"""Full bf16 unquantized fused-MoE on the HIP grouped WMMA GEMM (gfx1201) — Triton-free.

Replaces the fused_moe_kernel_triton (gemm1+SiLU, gemm2) + moe_sum_reduce_triton fallback used for
UNQUANTIZED (bf16/fp16) MoE experts. Orchestration mirrors the standard grouped-MoE, but the
intermediate stays in the moe_align sorted-padded layout [P, .] throughout (the layout the HIP
kernels consume), so both GEMMs and the topk-weighted combine are native HIP.

    y[m] = sum_{k<top_k} topk_w[m,k] * ( SiLU(x[m] @ w1[e][:N]^T) * (x[m] @ w1[e][N:]^T) ) @ w2[e]^T
    where e = topk_ids[m,k],  w1: [E, 2N, K],  w2: [E, K, N].
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .op import moe_bf16_gemm, moe_bf16_gemm_scatter


def moe_align_block_size(topk_ids: torch.Tensor, block_m: int, num_experts: int):
    """Sorted-padded expert grouping. Prefers the HIP/compiled moe_align if importable, else a torch
    fallback (cheap; the GEMMs dominate). Returns (sorted_ids[P], expert_ids[P/block_m], num_pad[1])."""
    try:
        from sgl_kernel import moe_align_block_size as _sgl  # compiled (HIP) when present
        M, top_k = topk_ids.shape
        max_pad = topk_ids.numel() + num_experts * (block_m - 1)
        max_blocks = (max_pad + block_m - 1) // block_m
        sorted_ids = topk_ids.new_full((max_pad,), M * top_k, dtype=torch.int32)
        expert_ids = topk_ids.new_zeros((max_blocks,), dtype=torch.int32)
        num_pad = topk_ids.new_zeros((1,), dtype=torch.int32)
        cumsum = topk_ids.new_zeros((num_experts + 2,), dtype=torch.int32)
        _sgl(topk_ids, num_experts + 1, block_m, sorted_ids, expert_ids, num_pad, cumsum, True)
        return sorted_ids, expert_ids, num_pad
    except Exception:
        pass
    # torch fallback
    M, top_k = topk_ids.shape
    dev = topk_ids.device
    flat = topk_ids.reshape(-1).cpu()
    expanded = torch.arange(M * top_k)
    sorted_ids, expert_ids = [], []
    for e in range(num_experts):
        ids_e = expanded[flat == e].tolist()
        if not ids_e:
            continue
        pad = (-len(ids_e)) % block_m
        run = ids_e + [M * top_k] * pad
        sorted_ids.extend(run)
        expert_ids.extend([e] * (len(run) // block_m))
    return (torch.tensor(sorted_ids, dtype=torch.int32, device=dev),
            torch.tensor(expert_ids, dtype=torch.int32, device=dev),
            torch.tensor([len(sorted_ids)], dtype=torch.int32, device=dev))


def fused_moe_bf16(hidden_states: torch.Tensor,      # [M, K] bf16
                   w1: torch.Tensor,                  # [E, 2N, K] bf16
                   w2: torch.Tensor,                  # [E, K, N] bf16
                   topk_weights: torch.Tensor,        # [M, top_k] f32
                   topk_ids: torch.Tensor,            # [M, top_k] int32
                   block_m: int = 64, BN: int = 128) -> torch.Tensor:
    M, K = hidden_states.shape
    E, twoN, _ = w1.shape
    N = twoN // 2
    top_k = topk_ids.shape[1]
    num_valid = M * top_k

    sorted_ids, expert_ids, num_pad = moe_align_block_size(topk_ids.to(torch.int32), block_m, E)
    tw = topk_weights.to(torch.float32).reshape(-1).contiguous()

    # gemm1: [P, 2N] = hidden[offs//top_k] @ w1[e]^T  (sorted-padded rows)
    inter1 = moe_bf16_gemm(hidden_states, w1, sorted_ids, expert_ids, num_pad, None,
                           top_k, block_m, num_valid, BN, 0)
    # SiLU gate on the sorted-padded intermediate -> [P, N]
    inter2 = (F.silu(inter1[:, :N].float()) * inter1[:, N:].float()).to(hidden_states.dtype).contiguous()
    # gemm2 + topk-weighted scatter-combine: [M, K] fp32
    out = moe_bf16_gemm_scatter(inter2, w2, sorted_ids, expert_ids, num_pad, tw,
                                M, top_k, block_m, num_valid, BN, top_k)
    return out.to(hidden_states.dtype)
