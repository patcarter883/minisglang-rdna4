"""Numeric parity for the W8A16 (fp8-weight × bf16-act) grouped MoE WMMA GEMM (gfx1201).

Reference dequants the SAME fp8 weights the kernel consumes (e4m3 byte -> f32 * per-channel scale),
then does an fp32 matmul with bf16-rounded operands — so a faithful kernel matches to bf16 accumulation
noise; an indexing / WMMA-layout / scale / expert-map bug shows as a large error.

Run inside the combined ROCm image UNDER a 1-card lease:
  cd /engine/moe_w8a16_wmma && GPU_ARCHS=gfx1201 python setup.py build_ext --inplace
  cd /engine && PYTHONPATH=/engine python moe_w8a16_wmma/moe_w8a16_parity.py
"""
from __future__ import annotations

import torch

import moe_w8a16_wmma as M  # noqa: F401  loads .so + registers torch.ops.moe_w8a16.*

DEV = "cuda"
torch.manual_seed(0)


def make_fp8(E, OUT, IN):
    """Random e4m3 weights + a per-output-channel f32 scale. Returns (w_bytes uint8 (E,OUT,IN),
    scales f32 (E,OUT), w_deq bf16 (E,OUT,IN) = e4m3_to_f32 * scale) — the kernel and ref share w_deq."""
    w_f8 = (torch.randn(E, OUT, IN, device=DEV) * (IN ** -0.5)).to(torch.float8_e4m3fn)
    scales = (torch.rand(E, OUT, device=DEV) * 0.5 + 0.5).to(torch.float32)  # (0.5, 1.0)
    w_bytes = w_f8.view(torch.uint8).contiguous()
    w_deq = (w_f8.float() * scales.unsqueeze(-1)).to(torch.bfloat16)          # exact e4m3 widen * scale
    return w_bytes, scales, w_deq


def moe_align(topk_ids, E, block_m):
    Mt, top_k = topk_ids.shape
    num_valid = Mt * top_k
    flat = topk_ids.reshape(-1).cpu()
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


def _report(name, got, ref, thr=2e-2):
    got, ref = got.float().flatten(), ref.float().flatten()
    cos = torch.nn.functional.cosine_similarity(got, ref, dim=0).item()
    rel = ((got - ref).norm() / (ref.norm() + 1e-8)).item()
    ok = (cos > 0.999) and (rel < thr) and torch.isfinite(got).all().item()
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:30s} cos={cos:.6f} rel={rel:.3e}")
    return ok


def check_gemm1(Mt, E, IN, OUT, top_k, block_m, BN):
    A = torch.randn(Mt, IN, device=DEV, dtype=torch.bfloat16)
    w_bytes, scales, w_deq = make_fp8(E, OUT, IN)
    gate = torch.randn(Mt, E, device=DEV)
    topk_ids = gate.topk(top_k, dim=-1).indices.to(torch.int32)
    sti, eid, ntpp, nvalid, P = moe_align(topk_ids, E, block_m)

    out = torch.ops.moe_w8a16.moe_w8a16_gemm(A, w_bytes, scales, sti, eid, ntpp, None,
                                             top_k, block_m, nvalid, BN, 0)
    ref = torch.zeros(P, OUT, device=DEV)
    valid = torch.zeros(P, dtype=torch.bool, device=DEV)
    Af, wf = A.float(), w_deq.float()
    for r in range(P):
        offs = int(sti[r])
        if offs >= nvalid:
            continue  # padded row: kernel leaves out[r] UNINITIALIZED (unused; gemm2 skips it too)
        valid[r] = True
        e = int(eid[r // block_m])
        ref[r] = Af[offs // top_k] @ wf[e].T
    # compare ONLY valid rows — padded rows are uninitialized garbage by design, never consumed.
    return _report(f"gemm1 M{Mt} E{E} {IN}x{OUT} tk{top_k}", out[:P][valid], ref[valid])


def check_gemm2_scatter(Mt, E, IN, OUT, top_k, block_m, BN):
    A = torch.randn(Mt, IN, device=DEV, dtype=torch.bfloat16)  # sorted-padded rows conceptually
    w_bytes, scales, w_deq = make_fp8(E, OUT, IN)
    gate = torch.randn(Mt, E, device=DEV)
    topk_ids = gate.topk(top_k, dim=-1).indices.to(torch.int32)
    topk_w = torch.rand(Mt, top_k, device=DEV, dtype=torch.float32)
    sti, eid, ntpp, nvalid, P = moe_align(topk_ids, E, block_m)
    # A here is [Mt, IN] as the "token" rows; scatter uses src = row_pad, token = offs//top_k
    Ap = torch.randn(P, IN, device=DEV, dtype=torch.bfloat16)
    tw = topk_w.reshape(-1).contiguous()

    out = torch.ops.moe_w8a16.moe_w8a16_gemm_scatter(Ap, w_bytes, scales, sti, eid, ntpp, tw,
                                                     Mt, top_k, block_m, nvalid, BN, top_k)
    ref = torch.zeros(Mt, OUT, device=DEV)
    Af, wf = Ap.float(), w_deq.float()
    for r in range(P):
        offs = int(sti[r])
        if offs >= nvalid:
            continue
        e = int(eid[r // block_m])
        token = offs // top_k
        ref[token] += float(tw[offs]) * (Af[r] @ wf[e].T)
    return _report(f"gemm2scat M{Mt} E{E} {IN}x{OUT} tk{top_k}", out, ref)


if __name__ == "__main__":
    ok = True
    # ZAYA-ish MoE shapes: hidden K=2048, moe_inter N=512 (gemm1 OUT=2N=1024), E=256, top_k up to 8.
    for (Mt, E, tk) in [(4, 8, 2), (16, 16, 4), (2, 32, 1), (29, 64, 8)]:
        ok &= check_gemm1(Mt, E, 2048, 1024, tk, 64, 128)   # gemm1: K=2048 -> 2*inter=1024
        ok &= check_gemm2_scatter(Mt, E, 512, 2048, tk, 64, 128)  # gemm2: inter=512 -> K=2048
    print("\nW8A16 MoE PARITY:", "PASS" if ok else "FAIL")
