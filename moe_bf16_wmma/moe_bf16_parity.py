"""Numeric parity for the bf16 grouped MoE WMMA GEMM (gfx1201).

Validates both epilogues of the single kernel against an fp32 torch reference that uses the SAME
bf16-rounded operands (so a faithful kernel matches to bf16 accumulation noise; a real
indexing/WMMA-layout/expert-map bug shows as a large error):
  moe_bf16_gemm          -> C[P,OUT] bf16  (gemm1: sorted-padded rows, src = offs//top_k)
  moe_bf16_gemm_scatter  -> C[M,OUT] fp32  (gemm2: topk-weighted scatter, out[token] += w * A@weᵀ)

Run inside the combined ROCm image UNDER a 1-card lease:
  cd /engine && PYTHONPATH=/engine python moe_bf16_wmma/moe_bf16_parity.py
(build first: cd moe_bf16_wmma && GPU_ARCHS=gfx1201 python setup.py build_ext --inplace)
"""
from __future__ import annotations

import torch

import moe_bf16_wmma as M  # loads .so + registers torch.ops.moe_bf16.*

DEV = "cuda"
torch.manual_seed(0)


def moe_align(topk_ids: torch.Tensor, E: int, block_m: int):
    """Reproduce moe_align_block_size: group expanded (token,slot) ids by expert, pad each expert
    run to a multiple of block_m with the sentinel (= num_valid). Returns the kernel inputs."""
    Mt, top_k = topk_ids.shape
    num_valid = Mt * top_k
    flat = topk_ids.reshape(-1).cpu()          # python-loop align runs on CPU
    expanded = torch.arange(Mt * top_k)
    sorted_ids, expert_ids = [], []
    for e in range(E):
        ids_e = expanded[flat == e].tolist()
        if not ids_e:
            continue
        pad = (-len(ids_e)) % block_m
        run = ids_e + [num_valid] * pad
        sorted_ids.extend(run)
        expert_ids.extend([e] * (len(run) // block_m))
    P = len(sorted_ids)
    return (torch.tensor(sorted_ids, dtype=torch.int32, device=DEV),
            torch.tensor(expert_ids, dtype=torch.int32, device=DEV),
            torch.tensor([P], dtype=torch.int32, device=DEV), num_valid, P)


def _report(name, got, ref, thr=2e-2) -> bool:
    got, ref = got.float().flatten(), ref.float().flatten()
    cos = torch.nn.functional.cosine_similarity(got, ref, dim=0).item()
    rel = ((got - ref).norm() / (ref.norm() + 1e-8)).item()
    ok = (cos > 0.999) and (rel < thr) and torch.isfinite(got).all().item()
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:34s} cos={cos:.6f} rel={rel:.3e} (thr={thr:.0e})")
    return ok


def check_gemm1(Mt, E, IN, OUT, top_k, block_m, BN) -> bool:
    A = torch.randn(Mt, IN, device=DEV, dtype=torch.bfloat16)
    w = torch.randn(E, OUT, IN, device=DEV, dtype=torch.bfloat16) * (IN ** -0.5)
    gate = torch.randn(Mt, E, device=DEV)
    topk_ids = gate.topk(top_k, dim=-1).indices.to(torch.int32)
    sti, eid, ntpp, nvalid, P = moe_align(topk_ids, E, block_m)

    out = torch.ops.moe_bf16.moe_bf16_gemm(A, w, sti, eid, ntpp, None,
                                           top_k, block_m, nvalid, BN, 0)
    # reference: only valid sorted rows
    ref = torch.zeros(P, OUT, device=DEV)
    valid = torch.zeros(P, dtype=torch.bool, device=DEV)
    Af, wf = A.float(), w.float()
    for r in range(P):
        offs = int(sti[r])
        if offs >= nvalid:
            continue
        token, e = offs // top_k, int(eid[r // block_m])
        ref[r] = Af[token] @ wf[e].T
        valid[r] = True
    return _report(f"gemm1 M{Mt} IN{IN} OUT{OUT} tk{top_k} bm{block_m} BN{BN}",
                   out[valid], ref[valid])


def check_gemm2(Mt, E, IN, OUT, top_k, block_m, BN) -> bool:
    gate = torch.randn(Mt, E, device=DEV)
    topk = gate.softmax(-1).topk(top_k, dim=-1)
    topk_ids = topk.indices.to(torch.int32)
    topk_w = topk.values.to(torch.float32).reshape(-1).contiguous()  # [M*top_k], indexed by offs
    sti, eid, ntpp, nvalid, P = moe_align(topk_ids, E, block_m)
    A = torch.randn(P, IN, device=DEV, dtype=torch.bfloat16)          # sorted-padded intermediate
    w = torch.randn(E, OUT, IN, device=DEV, dtype=torch.bfloat16) * (IN ** -0.5)

    out = torch.ops.moe_bf16.moe_bf16_gemm_scatter(A, w, sti, eid, ntpp, topk_w, Mt,
                                                   top_k, block_m, nvalid, BN, top_k)
    ref = torch.zeros(Mt, OUT, device=DEV)
    Af, wf = A.float(), w.float()
    for r in range(P):
        offs = int(sti[r])
        if offs >= nvalid:
            continue
        token, e = offs // top_k, int(eid[r // block_m])
        ref[token] += float(topk_w[offs]) * (Af[r] @ wf[e].T)
    return _report(f"gemm2 M{Mt} IN{IN} OUT{OUT} tk{top_k} bm{block_m} BN{BN}", out, ref)


def check_fused(Mt, E, K, N, top_k, block_m, BN) -> bool:
    """Full bf16 MoE (align -> gemm1 -> SiLU -> gemm2 -> topk combine) vs a torch reference."""
    from moe_bf16_wmma import fused_moe_bf16
    x = torch.randn(Mt, K, device=DEV, dtype=torch.bfloat16)
    w1 = torch.randn(E, 2 * N, K, device=DEV, dtype=torch.bfloat16) * (K ** -0.5)
    w2 = torch.randn(E, K, N, device=DEV, dtype=torch.bfloat16) * (N ** -0.5)
    gate = torch.randn(Mt, E, device=DEV)
    tk = gate.softmax(-1).topk(top_k, dim=-1)
    topk_ids = tk.indices.to(torch.int32)
    topk_w = tk.values.to(torch.float32)

    out = fused_moe_bf16(x, w1, w2, topk_w, topk_ids, block_m=block_m, BN=BN)

    xf, w1f, w2f = x.float(), w1.float(), w2.float()
    ref = torch.zeros(Mt, K, device=DEV)
    for m in range(Mt):
        for k in range(top_k):
            e = int(topk_ids[m, k])
            h = xf[m] @ w1f[e].T                       # [2N]
            g = torch.nn.functional.silu(h[:N]) * h[N:]  # [N]
            ref[m] += float(topk_w[m, k]) * (g @ w2f[e].T)
    return _report(f"fused_moe M{Mt} K{K} N{N} E{E} tk{top_k} bm{block_m} BN{BN}", out, ref, thr=3e-2)


def main():
    ok = True
    print("=== moe_bf16 gemm1 (non-scatter, sorted-padded) ===")
    ok &= check_gemm1(16, 8, 256, 128, 2, 32, 64)
    ok &= check_gemm1(8, 8, 512, 256, 2, 16, 64)
    ok &= check_gemm1(64, 16, 2048, 512, 4, 64, 128)   # Qwen3.5-MoE-ish geometry
    print("=== moe_bf16 gemm2 (scatter, topk-weighted) ===")
    ok &= check_gemm2(16, 8, 256, 128, 2, 32, 64)
    ok &= check_gemm2(2, 8, 512, 2048, 2, 16, 64)       # decode M=2
    ok &= check_gemm2(64, 16, 512, 2048, 4, 64, 128)
    print("=== full fused bf16 MoE (align -> gemm1 -> SiLU -> gemm2 -> combine) ===")
    ok &= check_fused(16, 8, 256, 512, 2, 32, 64)
    ok &= check_fused(4, 8, 512, 1024, 2, 16, 64)
    ok &= check_fused(64, 16, 512, 1408, 4, 64, 128)     # Qwen3.5-MoE-ish (K=512, N=1408/expert)
    print("=" * 60)
    print("RESULT:", "ALL PASS" if ok else "FAIL (see above)")


if __name__ == "__main__":
    main()
