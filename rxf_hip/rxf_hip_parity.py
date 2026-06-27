#!/usr/bin/env python
"""Parity check for the native RXF W4A8 HIP ops (torch.ops.rxf_hip.*) on gfx1201.

Validates each op against a torch reference that mirrors the Triton RXF numerics
(vllm-gfx1201/paroquant_rotation/rxf_kernels.py):
  1. rotate_quant_int8  : FWHT-32 + per-token int8 quant (exact int8 + scale match)
  2. linear (GEMV, M<=2) and linear (WMMA, M>2): NL-coded int4 . int8 -> bf16
  3. end-to-end dense   : x -> rotate_quant -> linear
  4. moe_gemm           : grouped per-expert int8 GEMM vs per-token reference

Run under a 1-card lease in the shared ROCm image (see repo README / CLAUDE.md):
  PYTHONPATH=/engine python /engine/rxf_hip/rxf_hip_parity.py
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

import rxf_hip

NL = torch.tensor(rxf_hip.NL_DEFAULT, dtype=torch.int8)
SPAN = 32
NORM = 0.1767766953  # == 1/sqrt(32), matched to the kernel + Triton literal


def ref_rotate_quant(x: torch.Tensor, span: int = SPAN):
    """FWHT-span over each size-span group of a row, then per-token symmetric int8 quant."""
    M, K = x.shape
    xr = x.float().reshape(M, K // span, span)
    h = 1
    while h < span:
        xr = xr.reshape(M, K // span, span // (2 * h), 2, h)
        a, b = xr[..., 0, :], xr[..., 1, :]
        xr = torch.stack([a + b, a - b], dim=-2).reshape(M, K // span, span)
        h *= 2
    xr = xr.reshape(M, K) * NORM
    absmax = xr.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    scale = (absmax / 127.0).squeeze(1)
    q = torch.round(xr * (127.0 / absmax)).clamp(-127, 127).to(torch.int8)
    return q, scale


def unpack_codes(w_packed: torch.Tensor, nl: torch.Tensor) -> torch.Tensor:
    """uint8 [N, K/2] -> fp32 NL code matrix [N, K] (low nibble = even channel)."""
    N, Kp = w_packed.shape
    lo = (w_packed & 0x0F).to(torch.long)
    hi = (w_packed >> 4).to(torch.long)
    idx = torch.stack([lo, hi], dim=-1).reshape(N, Kp * 2)  # even=lo, odd=hi
    return nl.to(idx.device)[idx].float()


def ref_linear(q, a_scale, w_packed, w_scale, nl, bias=None):
    codes = unpack_codes(w_packed, nl)  # [N, K]
    ws = w_scale.float().repeat_interleave(SPAN, dim=1)  # [N, K]
    wdq = codes * ws
    acc = q.float() @ wdq.t()  # [M, N]
    acc = acc * a_scale[:, None]
    if bias is not None:
        acc = acc + bias.float()
    return acc


def make_weights(N, K, dev, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    idx = torch.randint(0, 16, (N, K), generator=g, device=dev, dtype=torch.uint8)
    w_packed = (idx[:, 0::2] | (idx[:, 1::2] << 4)).contiguous()  # [N, K/2]
    w_scale = ((torch.rand(N, K // SPAN, generator=g, device=dev) - 0.5) * 0.04).to(torch.float16)
    return w_packed, w_scale


def report(tag, out, ref, thresh=0.99):
    out, ref = out.float().flatten(), ref.float().flatten()
    cos = F.cosine_similarity(out[None], ref[None]).item()
    rel = ((out - ref).norm() / (ref.norm() + 1e-8)).item()
    good = cos > thresh
    print(f"  [{'PASS' if good else 'FAIL'}] {tag:28s} cos-sim={cos:.5f} rel-err={rel:.4f}")
    return good


def main():
    dev = "cuda"
    nl = NL.to(dev)
    ok = True

    # 1. rotate_quant_int8 — exact int8 + scale match
    print("rotate_quant_int8:")
    for M, K in ((1, 2048), (8, 4096)):
        x = (torch.randn(M, K, device=dev) * 0.3).to(torch.bfloat16)
        q, scale = torch.ops.rxf_hip.rotate_quant_int8(x.contiguous(), SPAN)
        qref, sref = ref_rotate_quant(x)
        # int8 may differ by +-1 at round ties; require >=99.5% exact and scale ~match
        exact = (q == qref).float().mean().item()
        sgood = (scale - sref).abs().max().item() < 1e-4
        good = exact > 0.995 and sgood
        ok &= good
        print(f"  [{'PASS' if good else 'FAIL'}] M={M} K={K}  int8-exact={exact:.4f} "
              f"scale-maxabs={ (scale - sref).abs().max().item():.2e}")

    # 2/3. dense linear — GEMV (M<=2), WMMA (M>2), and end-to-end
    print("linear (q given) + end-to-end:")
    N, K = 1024, 2048
    w_packed, w_scale = make_weights(N, K, dev, seed=1)
    bias = (torch.randn(N, device=dev) * 0.1)
    for M in (1, 2, 8, 64):
        # op-direct: feed reference-quantized q so we isolate the GEMM
        x = (torch.randn(M, K, device=dev) * 0.3).to(torch.bfloat16)
        qref, sref = ref_rotate_quant(x)
        out = torch.ops.rxf_hip.linear(qref.to(dev), sref.to(dev), w_packed, w_scale, nl, bias)
        ref = ref_linear(qref.to(dev), sref.to(dev), w_packed, w_scale, nl, bias)
        ok &= report(f"linear M={M} ({'gemv' if M <= 2 else 'wmma'})", out, ref)
        # end-to-end: rotate_quant on device then linear
        q, scale = torch.ops.rxf_hip.rotate_quant_int8(x.contiguous(), SPAN)
        out_e2e = torch.ops.rxf_hip.linear(q, scale, w_packed, w_scale, nl, bias)
        ref_e2e = ref_linear(qref.to(dev), sref.to(dev), w_packed, w_scale, nl, bias)
        ok &= report(f"e2e    M={M}", out_e2e, ref_e2e)

    # 4. moe_gemm — grouped per-expert int8 GEMM
    print("moe_gemm (grouped):")
    from vllm import _custom_ops as vops
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size

    E, Kk, Ninter, top_k, block_m = 8, 1024, 512, 2, 16
    wp = torch.stack([make_weights(Ninter, Kk, dev, seed=10 + e)[0] for e in range(E)])
    wsc = torch.stack([make_weights(Ninter, Kk, dev, seed=10 + e)[1] for e in range(E)])
    for M in (4, 16):
        x = (torch.randn(M, Kk, device=dev) * 0.3).to(torch.bfloat16)
        gating = torch.randn(M, E, device=dev)
        q, ascale = torch.ops.rxf_hip.rotate_quant_int8(x.contiguous(), SPAN)
        tw = torch.empty(M, top_k, dtype=torch.float32, device=dev)
        ti = torch.empty(M, top_k, dtype=torch.int32, device=dev)
        tei = torch.empty(M, top_k, dtype=torch.int32, device=dev)
        vops.topk_softmax(tw, ti, tei, gating.float(), True)
        sorted_ids, expert_ids, ntp = moe_align_block_size(
            ti, block_m, E, None, pad_sorted_ids=True)
        out = torch.ops.rxf_hip.moe_gemm(
            q, ascale, wp, wsc, nl, sorted_ids, expert_ids, ntp, top_k, block_m, M * top_k)
        # reference: per padded row, dequant expert weight . quantized rotated activation
        qref, sref = ref_rotate_quant(x)
        P = sorted_ids.shape[0]
        ref = torch.zeros(P, Ninter, device=dev)
        nvalid = int(ntp.item())
        for r in range(min(P, nvalid)):
            offs = int(sorted_ids[r].item())
            if offs >= M * top_k:
                continue
            e = int(expert_ids[r // block_m].item())
            src = offs // top_k
            ref[r] = ref_linear(qref[src:src+1].to(dev), sref[src:src+1].to(dev),
                                wp[e], wsc[e], nl)[0]
        # compare only valid (non-padding) rows
        valid = sorted_ids[:nvalid] < (M * top_k)
        ok &= report(f"moe M={M}", out[:nvalid][valid], ref[:nvalid][valid])

    print("VERDICT:", "RXF PARITY" if ok else "INVESTIGATE")


if __name__ == "__main__":
    main()
