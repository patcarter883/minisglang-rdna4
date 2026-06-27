#!/usr/bin/env python
"""W4A8 MoE parity check: run the engine's grouped-MoE forward (quant/kernels.w4a8_moe)
on synthetic int4 experts (op layout, symmetric uint4b8) and compare to a bf16-dequant
reference MoE. Validates the MoE compute integration (topk -> moe_align -> 2 grouped
GEMMs -> silu -> gather-reduce) numerically, independent of checkpoint loading."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from minisgl.quant import kernels


def pack_int4(q: torch.Tensor) -> torch.Tensor:
    # q (E,N,K) in [0,15] -> (E,N,K//8) int32, nibble j = input k8*8+j (natural order).
    E, N, K = q.shape
    packed = torch.zeros((E, N, K // 8), dtype=torch.int32, device=q.device)
    for j in range(8):
        packed |= (q[:, :, j::8].to(torch.int32) & 0xF) << (j * 4)
    return packed


def dequant(q: torch.Tensor, scales: torch.Tensor, g: int) -> torch.Tensor:
    # symmetric uint4b8: value = (q - 8) * per-group scale.
    se = scales.repeat_interleave(g, dim=2).float()  # (E,N,K)
    return ((q.float() - 8.0) * se).to(torch.bfloat16)


def ref_moe(x, w13_bf, w2_bf, gating, top_k, renorm):
    M, K = x.shape
    probs = torch.softmax(gating.float(), dim=-1)
    tw, ti = torch.topk(probs, top_k, dim=-1)
    if renorm:
        tw = tw / (tw.sum(dim=-1, keepdim=True) + 1e-8)
    out = torch.zeros((M, K), dtype=torch.float32, device=x.device)
    for m in range(M):
        for k in range(top_k):
            e = int(ti[m, k]); w = float(tw[m, k])
            h = x[m : m + 1].float() @ w13_bf[e].float().t()  # (1, 2*inter)
            d = h.shape[1] // 2
            act = F.silu(h[:, :d]) * h[:, d:]
            out[m] += w * (act @ w2_bf[e].float().t())[0]
    return out


def main() -> None:
    torch.manual_seed(0)
    dev = "cuda"
    E, K, inter, top_k, g = 32, 2048, 512, 4, 32
    q13 = torch.randint(0, 16, (E, 2 * inter, K), device=dev)
    q2 = torch.randint(0, 16, (E, K, inter), device=dev)
    s13 = (torch.rand(E, 2 * inter, K // g, device=dev) * 0.02 + 0.005).to(torch.float16)
    s2 = (torch.rand(E, K, inter // g, device=dev) * 0.02 + 0.005).to(torch.float16)
    w13, w2 = pack_int4(q13), pack_int4(q2)
    w13_bf, w2_bf = dequant(q13, s13, g), dequant(q2, s2, g)

    # M=1,2 exercise the DECODE scatter-fusion path (mmq_fp8_moe_gemm_scatter); M=8 the unfused
    # gemm2 + gather_reduce (prefill) path. Both must match the bf16-dequant reference.
    ok = True
    for M in (1, 2, 8):
        torch.manual_seed(100 + M)
        x = (torch.randn(M, K, device=dev) * 0.1).to(torch.bfloat16)
        gating = torch.randn(M, E, device=dev)
        out = kernels.w4a8_moe(x, w13, s13, None, w2, s2, None, gating, top_k, renormalize=True)
        ref = ref_moe(x, w13_bf, w2_bf, gating, top_k, renorm=True)
        cos = F.cosine_similarity(out.float().flatten()[None], ref.flatten()[None]).item()
        rel = ((out.float() - ref).norm() / (ref.norm() + 1e-8)).item()
        path = "scatter" if M <= 2 else "gather "
        good = cos > 0.99
        ok &= good
        print(f"  [{'PASS' if good else 'FAIL'}] M={M} ({path}) cos-sim={cos:.5f} rel-err={rel:.4f}")
    print(f"(E={E} K={K} inter={inter} top_k={top_k} g={g})")
    print("VERDICT:", "MoE PARITY" if ok else "INVESTIGATE")


if __name__ == "__main__":
    main()
