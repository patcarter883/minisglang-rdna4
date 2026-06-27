"""Numeric parity for the native tail_hip elementwise ops (GPU).

Each op is checked against minisgl-rdna4's EXACT torch reference (the impls these replace):
  rms_norm / rms_norm_add  : layers/norm.py  _rms_norm (fp32-internal, plus_one gain)
  silu_and_mul             : layers/activation.py  _gated(F.silu)
  rope                     : layers/rotary.py  _apply (NeoX rotate-half, partial rd, cat cache)
Kernels are fp32-internal / bf16-out, so we compare vs the fp32 reference ROUNDED TO bf16 (the best
a bf16 output can represent) — a real bug (wrong reduction, mask, RoPE pairing) shows as a large Δ.

Run inside vllm22-w4a8:combined UNDER a 1-card lease (see attn_decode_parity.py for the docker line).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

import op  # noqa: F401  loads tail_hip_C + registers torch.ops.tail_hip.*

DEV = "cuda"
torch.manual_seed(0)
DELTA = 1.5e-2   # vs bf16-rounded reference (kernel error within bf16 representability)


def _ok(name, got, ref):
    ref_b = ref.bfloat16().float()
    d = (got.float() - ref_b).abs().max().item()
    cos = F.cosine_similarity(got.float().flatten(), ref.flatten(), dim=0).item()
    ok = (d <= DELTA) and (cos >= 0.999)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:30s} max|Δ|bf16={d:.3e}  cos={cos:.6f}")
    return ok


# ---- references (verbatim from minisgl) ----
def ref_rms(x, w, eps, plus_one):
    xf = x.float()
    var = xf.pow(2).mean(-1, keepdim=True)
    normed = xf * torch.rsqrt(var + eps)
    g = (w.float() + 1.0) if plus_one else w.float()
    return normed * g


def ref_silu(x):
    d = x.shape[-1] // 2
    return F.silu(x[..., :d].float()) * x[..., d:].float()


def ref_rope(x, pos, base, hs, rd, max_pos):
    inv = 1.0 / (base ** (torch.arange(0, rd, 2, dtype=torch.float, device=x.device) / rd))
    t = torch.arange(max_pos, dtype=torch.float, device=x.device)
    freqs = torch.einsum("i,j->ij", t, inv)
    cache = torch.cat((freqs.cos(), freqs.sin()), -1)          # [max_pos, rd]
    cos_h, sin_h = cache[pos].chunk(2, -1)
    cos = torch.cat((cos_h, cos_h), -1); sin = torch.cat((sin_h, sin_h), -1)
    n = x.shape[0]
    xf = x.view(n, -1, hs).float()
    d = rd // 2
    xr = xf[..., :rd]
    x1, x2 = xr[..., :d], xr[..., d:]
    rot = torch.cat((-x2, x1), -1)
    out_rot = xr * cos.view(n, 1, rd) + rot * sin.view(n, 1, rd)
    out = torch.cat((out_rot, xf[..., rd:]), -1) if rd < hs else out_rot
    return out.view(n, -1), cache


def main():
    print("=== tail_hip parity (vs minisgl torch refs, bf16-rounded) ===")
    ok = True
    # RMSNorm (Qwen3.5 hidden sizes; plus_one on/off)
    for (N, D, po) in [(64, 2048, 1), (37, 4096, 0), (128, 5120, 1), (200, 896, 1)]:
        x = torch.randn(N, D, device=DEV, dtype=torch.bfloat16)
        w = torch.randn(D, device=DEV, dtype=torch.bfloat16) * 0.1
        got = torch.ops.tail_hip.rms_norm(x, w, 1e-6, po)
        ok &= _ok(f"rms_norm N{N} D{D} po{po}", got, ref_rms(x, w, 1e-6, po))

    # fused residual-add RMSNorm (residual mutated to x+residual; out = rmsnorm(sum))
    for (N, D, po) in [(64, 2048, 1), (96, 4096, 0)]:
        x = torch.randn(N, D, device=DEV, dtype=torch.bfloat16)
        res = torch.randn(N, D, device=DEV, dtype=torch.bfloat16)
        res_ref = (x.float() + res.float())
        got = torch.ops.tail_hip.rms_norm_add(x, res, w := torch.randn(D, device=DEV, dtype=torch.bfloat16) * 0.1, 1e-6, po)
        ok &= _ok(f"rms_norm_add N{N} D{D} po{po}", got, ref_rms(res_ref.bfloat16(), w, 1e-6, po))
        ok &= _ok(f"  residual-updated N{N}", res.float(), res_ref)

    # SiLU-mul (dtype-generic: bf16 attn/norm path + fp16 W4A8-MoE intermediates)
    for (N, D) in [(64, 4096), (128, 11008), (37, 1536)]:
        x = torch.randn(N, 2 * D, device=DEV, dtype=torch.bfloat16)
        got = torch.ops.tail_hip.silu_and_mul(x)
        ok &= _ok(f"silu_and_mul bf16 N{N} D{D}", got, ref_silu(x))
    for (N, D) in [(2, 768), (128, 768), (33, 1536)]:   # fp16, MoE-shaped (P, 2*inter)
        x = torch.randn(N, 2 * D, device=DEV, dtype=torch.float16)
        got = torch.ops.tail_hip.silu_and_mul(x)
        ref = ref_silu(x)
        # fp16 output -> compare against an fp16-rounded reference (fp16 mantissa is finer than bf16)
        d = (got.float() - ref.half().float()).abs().max().item()
        cos = F.cosine_similarity(got.float().flatten(), ref.flatten(), dim=0).item()
        sok = (d <= 5e-3) and (cos >= 0.999)
        print(f"  [{'PASS' if sok else 'FAIL'}] {f'silu_and_mul fp16 N{N} D{D}':30s} max|Δ|fp16={d:.3e}  cos={cos:.6f}")
        ok &= sok

    # RoPE (full + partial rotary; NeoX)
    for (N, H, hs, rd) in [(64, 16, 128, 128), (100, 16, 128, 64), (37, 8, 256, 128)]:
        x = torch.randn(N, H * hs, device=DEV, dtype=torch.bfloat16)
        pos = torch.randint(0, 4096, (N,), device=DEV, dtype=torch.int32)
        ref, cache = ref_rope(x, pos.long(), 1e6, hs, rd, 4096)
        got = torch.ops.tail_hip.rope(x, pos, cache.contiguous(), hs, rd)
        ok &= _ok(f"rope N{N} H{H} hs{hs} rd{rd}", got, ref)
    print("RESULT:", "ALL PASS" if ok else "FAILURES PRESENT")


if __name__ == "__main__":
    main()
