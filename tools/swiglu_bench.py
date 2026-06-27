#!/usr/bin/env python
"""bf16 SwiGLU MLP latency at the shapes Task B (#18) targets — the dense shared expert
(qwen3_5_moe / qwen2_moe `*MoeSharedExpert`) and the dense FFN (`utils.GatedMLP`).

Measure-first: at M=1 (decode) the unquantized path is two `F.linear` GEMVs + `silu_and_mul`,
and the small-inter shared expert is occupancy-starved (few waves, far below peak weight BW).
This bench times the baseline and, once `torch.ops.*.fused_swiglu` exists, the fused kernel +
a cos-sim parity check, at matched shapes.

  PYTHONPATH=/engine/python:/engine python /engine/tools/swiglu_bench.py

Single card (-n 1). Shapes are per-TP-rank (the work one card actually does): shared expert
runs tp=2 on the 35B (inter_local = inter/2); GatedMLP runs tp=1 on the single-card 4B.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

PEAK_GBS = 644.0  # RX 9070 XT HBM peak


def _make_silu():
    # Match the model's silu_and_mul without importing the heavy minisgl.layers chain.
    try:
        import tail_hip  # noqa: F401  (registers torch.ops.tail_hip)
        op = torch.ops.tail_hip.silu_and_mul
        op(torch.zeros(1, 4, device="cuda", dtype=torch.bfloat16))  # probe
        print("silu_and_mul: native tail_hip")
        return lambda x: op(x.contiguous())
    except Exception as ex:
        print("silu_and_mul: torch fallback (", ex, ")")

        def f(x):
            d = x.shape[-1] // 2
            return (F.silu(x[..., :d].float()) * x[..., d:].float()).to(x.dtype)

        return f


silu_and_mul = _make_silu()


def t(fn, iters=200, warmup=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    e.synchronize()
    return s.elapsed_time(e) / iters * 1000  # us/call


def baseline(x, w_gate_up, w_down):
    return F.linear(silu_and_mul(F.linear(x, w_gate_up)), w_down)


def fused_op():
    try:
        import swiglu_hip  # noqa: F401  (registers torch.ops.swiglu_hip)
        return torch.ops.swiglu_hip.fused_swiglu
    except Exception as ex:
        print("swiglu_hip.fused_swiglu unavailable:", ex)
        return None


def cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return (a @ b / (a.norm() * b.norm() + 1e-12)).item()


def main():
    dev = "cuda"
    # (name, hidden, inter_local, layers, note) — inter_local is the per-card intermediate.
    SHAPES = [
        ("shared-expert 35B (tp2)", 2048, 256, 40, "inter=512/2"),
        ("GatedMLP 4B (tp1)", 2560, 9216, 32, "dense FFN"),
    ]
    fused = fused_op()
    print(f"fused op available: {fused is not None}")
    for name, hidden, inter, layers, note in SHAPES:
        w_gate_up = (torch.randn(2 * inter, hidden, device=dev) * 0.02).to(torch.bfloat16)
        w_down = (torch.randn(hidden, inter, device=dev) * 0.02).to(torch.bfloat16)
        wbytes = (2 * inter * hidden + hidden * inter) * 2  # bf16 weight read / call
        print(f"\n=== {name}  hidden={hidden} inter_local={inter} ({note}); "
              f"{wbytes/1e6:.2f} MB/call, {layers} layers ===")
        print(f"{'M':>4} {'base us':>9} {'base GB/s':>10} {'%peak':>6} "
              f"{'fused us':>9} {'speedup':>8} {'cos':>8}  est ms/step (base->fused)")
        for M in (1, 8):
            x = (torch.randn(M, hidden, device=dev) * 0.3).to(torch.bfloat16)
            tb = t(lambda: baseline(x, w_gate_up, w_down))
            gbs = wbytes / (tb * 1e-6) / 1e9
            tf = cosv = sp = float("nan")
            if fused is not None:
                ref = baseline(x, w_gate_up, w_down)
                got = fused(x, w_gate_up, w_down)
                cosv = cos(ref, got)
                tf = t(lambda: fused(x, w_gate_up, w_down))
                sp = tb / tf
            step_b = tb * layers / 1000.0
            step_f = (tf * layers / 1000.0) if fused is not None else float("nan")
            print(f"{M:>4} {tb:>9.1f} {gbs:>10.1f} {100*gbs/PEAK_GBS:>5.0f}% "
                  f"{tf:>9.1f} {sp:>8.2f} {cosv:>8.5f}  {step_b:>6.2f} -> {step_f:.2f}")


if __name__ == "__main__":
    main()
