#!/usr/bin/env python
"""RXF W4A8 Python-dispatch parity: exercise the engine seams (quant.kernels.rxf_linear,
quant.kernels.rxf_moe, and RXFLinearMethod) on synthetic RXF-format weights and compare to a
bf16-dequant reference. Complements rxf_hip/rxf_hip_parity.py (which tests the raw torch.ops).

  PYTHONPATH=/engine/python:/engine python /engine/tools/rxf_parity.py
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

import rxf_hip
from minisgl.quant import kernels

NL = torch.tensor(rxf_hip.NL_DEFAULT, dtype=torch.int8)
SPAN = 32
NORM = 0.1767766953


def ref_rotate_quant(x, span=SPAN):
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
    q = torch.round(xr * (127.0 / absmax)).clamp(-127, 127)
    return q * (absmax / 127.0)  # dequantized rotated activation [M,K] fp32


def make_w(N, K, dev, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    idx = torch.randint(0, 16, (N, K), generator=g, device=dev, dtype=torch.uint8)
    w_packed = (idx[:, 0::2] | (idx[:, 1::2] << 4)).contiguous()
    w_scale = ((torch.rand(N, K // SPAN, generator=g, device=dev) - 0.5) * 0.04).to(torch.float16)
    codes = NL.to(dev)[idx.long()].float()
    w_dq = codes * w_scale.float().repeat_interleave(SPAN, dim=1)  # [N,K] fp32 dequant weight
    return w_packed, w_scale, w_dq


def cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return F.cosine_similarity(a[None], b[None]).item()


def main():
    dev = "cuda"
    ok = True

    print("dense (kernels.rxf_linear):")
    N, K = 1024, 2048
    wp, ws, w_dq = make_w(N, K, dev, seed=1)
    bias = torch.randn(N, device=dev) * 0.1
    for M in (1, 2, 8, 64):
        x = (torch.randn(M, K, device=dev) * 0.3).to(torch.bfloat16)
        out = kernels.rxf_linear(x, wp, ws, bias, SPAN)
        ref = ref_rotate_quant(x) @ w_dq.t() + bias.float()
        c = cos(out, ref)
        good = c > 0.99
        ok &= good
        print(f"  [{'PASS' if good else 'FAIL'}] M={M:3d} cos-sim={c:.5f}")

    print("MoE (kernels.rxf_moe):")
    E, Kk, inter, top_k = 8, 1024, 512, 2
    w13p, w13s, w13dq = zip(*[make_w(2 * inter, Kk, dev, 10 + e) for e in range(E)])
    w2p, w2s, w2dq = zip(*[make_w(Kk, inter, dev, 50 + e) for e in range(E)])
    w13p, w13s = torch.stack(w13p), torch.stack(w13s)
    w2p, w2s = torch.stack(w2p), torch.stack(w2s)
    w13dq, w2dq = torch.stack(w13dq), torch.stack(w2dq)

    def ref_moe(x, gating):
        M = x.shape[0]
        probs = torch.softmax(gating.float(), -1)
        tw, ti = torch.topk(probs, top_k, -1)
        tw = tw / (tw.sum(-1, keepdim=True) + 1e-8)
        out = torch.zeros(M, Kk, device=dev)
        for m in range(M):
            for k in range(top_k):
                e, w = int(ti[m, k]), float(tw[m, k])
                h = ref_rotate_quant(x[m:m+1]) @ w13dq[e].t()  # (1, 2*inter)
                act = (F.silu(h[:, :inter]) * h[:, inter:]).to(torch.bfloat16)
                out[m] += w * (ref_rotate_quant(act) @ w2dq[e].t())[0]
        return out

    # M=1,2 exercise the fused scatter decode path; M=4,16 the gemm2+gather_reduce prefill path.
    for M in (1, 2, 4, 16):
        x = (torch.randn(M, Kk, device=dev) * 0.3).to(torch.bfloat16)
        gating = torch.randn(M, E, device=dev)
        out = kernels.rxf_moe(x, w13p, w13s, w2p, w2s, gating, top_k, True, span=SPAN)
        ref = ref_moe(x, gating)
        c = cos(out, ref)
        good = c > 0.99
        ok &= good
        path = "scatter" if M <= 2 else "gather "
        print(f"  [{'PASS' if good else 'FAIL'}] M={M:3d} ({path}) cos-sim={c:.5f}")

    print("VERDICT:", "RXF DISPATCH PARITY" if ok else "INVESTIGATE")


if __name__ == "__main__":
    main()
