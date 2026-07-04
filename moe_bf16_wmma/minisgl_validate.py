"""Validate the vendored full-HIP unquantized MoE path inside minisglang (gfx1201).

Three checks, on a synthetic MoE (no model download needed):
  (a) parity vs an fp32 torch reference (absolute correctness),
  (b) parity vs minisgl's Triton `fused_experts_impl` (the path it replaces), cos > 0.99,
  (c) graph-capture safety: capture `_fused_experts_bf16_hip` under torch.cuda.graph, replay, compare.

Run inside the combined ROCm image UNDER a 1-card lease (build the kernel first):
  cd /engine/moe_bf16_wmma && GPU_ARCHS=gfx1201 python setup.py build_ext --inplace
  cd /engine && PYTHONPATH=/engine:/engine/python python moe_bf16_wmma/minisgl_validate.py
"""
from __future__ import annotations

import torch

from minisgl.moe import fused as MF

DEV = "cuda"
torch.manual_seed(0)


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float().flatten(), b.float().flatten()
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float().flatten(), b.float().flatten()
    return ((a - b).norm() / (b.norm() + 1e-8)).item()


def _ref_fp32(x, w1, w2, tw, tid, top_k, N):
    xf, w1f, w2f = x.float(), w1.float(), w2.float()
    M, K = x.shape
    ref = torch.zeros(M, K, device=DEV)
    for m in range(M):
        for k in range(top_k):
            e = int(tid[m, k])
            h = xf[m] @ w1f[e].T
            g = torch.nn.functional.silu(h[:N]) * h[N:]
            ref[m] += float(tw[m, k]) * (g @ w2f[e].T)
    return ref


def _make(M, E, K, N, top_k, dtype):
    x = torch.randn(M, K, device=DEV, dtype=dtype) * 0.5
    w1 = torch.randn(E, 2 * N, K, device=DEV, dtype=dtype) * (K ** -0.5)
    w2 = torch.randn(E, K, N, device=DEV, dtype=dtype) * (N ** -0.5)
    gate = torch.randn(M, E, device=DEV)
    tk = gate.softmax(-1).topk(top_k, dim=-1)
    tid = tk.indices.to(torch.int32)
    tw = tk.values.to(torch.float32)
    return x, w1, w2, tw, tid


def check(M, E, K, N, top_k, dtype):
    tag = f"M{M} E{E} K{K} N{N} tk{top_k} {str(dtype).split('.')[-1]}"
    x, w1, w2, tw, tid = _make(M, E, K, N, top_k, dtype)
    config = MF.try_get_optimal_moe_config(w1.shape, (E, K, N), top_k, M)

    # (b0) Triton reference (force the Triton branch). fused_experts_impl mutates hidden in place.
    saved = MF._MOE_BF16_OK
    MF._MOE_BF16_OK = False
    out_triton = MF.fused_experts_impl(x.clone(), w1, w2, tw, tid, activation="silu").clone()
    MF._MOE_BF16_OK = saved

    # (a/b) HIP path (eager)
    out_hip = MF._fused_experts_bf16_hip(x.clone(), w1, w2, tw, tid, config)

    ref = _ref_fp32(x, w1, w2, tw, tid, top_k, N)
    cos_ref, cos_tri = _cos(out_hip, ref), _cos(out_hip, out_triton)
    cos_tri_ref = _cos(out_triton, ref)
    ok = (cos_ref > 0.99) and (cos_tri > 0.99) and torch.isfinite(out_hip).all().item()
    print(f"  [{'PASS' if ok else 'FAIL'}] {tag:34s} "
          f"cos(hip,fp32)={cos_ref:.5f} cos(hip,triton)={cos_tri:.5f} "
          f"cos(triton,fp32)={cos_tri_ref:.5f} rel(hip,triton)={_rel(out_hip, out_triton):.2e}")
    return ok


def check_capture(M, E, K, N, top_k, dtype):
    tag = f"capture M{M} E{E} K{K} N{N} tk{top_k} {str(dtype).split('.')[-1]}"
    x, w1, w2, tw, tid = _make(M, E, K, N, top_k, dtype)
    config = MF.try_get_optimal_moe_config(w1.shape, (E, K, N), top_k, M)

    def run():
        return MF._fused_experts_bf16_hip(x, w1, w2, tw, tid, config)

    # Warmup on a side stream (allocate buffers + lazy init) BEFORE capture.
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            eager = run()
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    static_out = None
    try:
        with torch.cuda.graph(g):
            static_out = run()
    except Exception as ex:
        print(f"  [FAIL] {tag:40s} capture raised: {type(ex).__name__}: {ex}")
        return False

    # Mutate inputs, replay, and confirm the captured graph recomputes correctly.
    x2, w1b, w2b, tw2, tid2 = _make(M, E, K, N, top_k, dtype)
    x.copy_(x2); w1.copy_(w1b); w2.copy_(w2b); tw.copy_(tw2); tid.copy_(tid2)
    g.replay()
    torch.cuda.synchronize()
    ref = _ref_fp32(x, w1, w2, tw, tid, top_k, N)
    cos = _cos(static_out, ref)
    ok = (cos > 0.99) and torch.isfinite(static_out).all().item()
    print(f"  [{'PASS' if ok else 'FAIL'}] {tag:40s} replay cos(fp32)={cos:.5f}")
    return ok


def main():
    ok = True
    print("=== (a/b) HIP fused MoE vs fp32 ref AND vs Triton fused_experts_impl ===")
    ok &= check(16, 8, 256, 512, 2, torch.bfloat16)
    ok &= check(4, 8, 512, 1024, 2, torch.bfloat16)
    ok &= check(64, 16, 512, 1408, 4, torch.bfloat16)   # Qwen3.5-MoE-ish
    ok &= check(2, 8, 512, 2048, 2, torch.bfloat16)      # decode M=2
    ok &= check(16, 8, 256, 512, 2, torch.float16)       # fp16
    ok &= check(64, 16, 512, 1408, 4, torch.float16)
    print("=== (c) graph-capture safety (torch.cuda.graph capture + replay) ===")
    ok &= check_capture(2, 8, 512, 2048, 2, torch.bfloat16)
    ok &= check_capture(16, 16, 512, 1408, 4, torch.bfloat16)
    ok &= check_capture(16, 8, 256, 512, 2, torch.float16)
    print("=" * 70)
    print("RESULT:", "ALL PASS" if ok else "FAIL (see above)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
