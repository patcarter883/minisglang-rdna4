"""Full W8A16 (fp8-weight × bf16-act) fused-MoE on the HIP grouped WMMA GEMM (gfx1201).

Drop-in for the OLDMOE=1 dequant→Triton bf16 MoE path in minisgl/layers/moe.py, but WITHOUT the
per-forward full-stack dequant: the fp8 weight tile is dequanted to bf16 in-register, activations stay
bf16 (no fp8-act precision loss), and only the routed experts are touched. Same math as fused_moe_bf16:

    y[m] = sum_{k<top_k} topk_w[m,k] * ( SiLU(x[m] @ W1[e]:N^T) * (x[m] @ W1[e]N:^T) ) @ W2[e]^T
    where W1[e] = dequant(w13_fp8[e], w13_scales[e]),  W2[e] = dequant(w2_fp8[e], w2_scales[e]).
    w13_fp8: [E, 2N, K] e4m3 + w13_scales [E, 2N];  w2_fp8: [E, K, N] e4m3 + w2_scales [E, K].
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F

from .op import moe_w8a16_gemm, moe_w8a16_gemm_scatter

# Autotuned tile defaults (moe_w8a16_autotune.py, ZAYA E16/K2048/inter4096/top1, M=29):
# block_m=16 BN=32 = 2.15ms/layer vs 64/128's 3.65ms = 1.70× faster (minimal moe_align padding at the
# tiny-M top-1 decode). Overridable via env for the serve path.
_DEF_BLOCK_M = int(os.environ.get("MINISGL_W8A16_BLOCK_M", "16"))
_DEF_BN = int(os.environ.get("MINISGL_W8A16_BN", "32"))


def moe_align_block_size(topk_ids: torch.Tensor, block_m: int, num_experts: int):
    """Sorted-padded expert grouping. Prefers the compiled moe_hip/sgl align, else a torch fallback."""
    try:
        import moe_hip
        return moe_hip.moe_align(topk_ids.to(torch.int32), num_experts, block_m)
    except Exception:
        pass
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


def fused_moe_w8a16(hidden_states: torch.Tensor,   # [M, K] bf16
                    w13_fp8: torch.Tensor,          # [E, 2N, K] uint8 (e4m3)
                    w13_scales: torch.Tensor,       # [E, 2N] f32
                    w2_fp8: torch.Tensor,           # [E, K, N] uint8 (e4m3)
                    w2_scales: torch.Tensor,        # [E, K] f32
                    topk_weights: torch.Tensor,     # [M, top_k] f32
                    topk_ids: torch.Tensor,         # [M, top_k] int32
                    block_m: int | None = None, BN: int | None = None) -> torch.Tensor:
    if block_m is None:
        block_m = _DEF_BLOCK_M
    if BN is None:
        BN = _DEF_BN
    try:
        from minisgl._hip_engage import engaged
        engaged("moe_w8a16")
    except Exception:
        pass
    M, K = hidden_states.shape
    E, twoN, _ = w13_fp8.shape
    N = twoN // 2
    top_k = topk_ids.shape[1]
    num_valid = M * top_k
    x = hidden_states.to(torch.bfloat16).contiguous()

    sorted_ids, expert_ids, num_pad = moe_align_block_size(topk_ids.to(torch.int32), block_m, E)
    tw = topk_weights.to(torch.float32).reshape(-1).contiguous()

    # gemm1: [P, 2N] bf16 = x[offs//top_k] @ dequant(w13[e])^T
    inter1 = moe_w8a16_gemm(x, w13_fp8, w13_scales, sorted_ids, expert_ids, num_pad, None,
                            top_k, block_m, num_valid, BN, 0)
    # SiLU gate on the sorted-padded intermediate -> [P, N]
    inter2 = (F.silu(inter1[:, :N].float()) * inter1[:, N:].float()).to(torch.bfloat16).contiguous()
    # gemm2 + topk-weighted scatter-combine: [M, K] fp32 -> hidden dtype
    out = moe_w8a16_gemm_scatter(inter2, w2_fp8, w2_scales, sorted_ids, expert_ids, num_pad, tw,
                                 M, top_k, block_m, num_valid, BN, top_k)
    return out.to(hidden_states.dtype)
