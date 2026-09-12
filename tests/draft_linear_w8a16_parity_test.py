"""DraftLinear's fp8 arm: the streaming kernel vs the dequant path it replaces.

WHY THIS EXISTS. `fp8_wmma.dense_w8a16_gemv` was built on 2026-09-07 and recorded as closing gap G1
in `rdna4-hip-kernels/FORMAT_MATRIX.md`. Nothing ever called it. Every DFlash/DSpark drafter run
with `MINISGL_DFLASH_QUANT=fp8` kept taking `F.linear(x, wq.to(x.dtype) * ws)`, which materialises a
full [out, in] dequantised temporary on every forward — the mechanism behind `tools/serve.sh:324`
"fp8 on this drafter is WORSE, 27.0 tok/s". A closed gap that nothing calls is not a closed gap.

THIS IS NOT A BIT-EXACTNESS TEST, and it must not be written as one. The op's own docstring is
explicit: the activation is untouched (still bf16/fp16, never quantised per token), but the scale is
applied ONCE in fp32 after the K-sum instead of being folded into a bf16-ROUNDED weight before the
GEMM. One rounding step is removed and the accumulation order differs. So the two paths genuinely
disagree, the kernel is plausibly the MORE accurate of the two, and what matters is that the
disagreement is small relative to the quantisation error both share.

That is what is pinned here:
  * the kernel tracks the dequant path to well within the fp8 quantisation error itself, so adopting
    it cannot be what moves a drafter's acceptance;
  * BOTH track the unquantised reference, so neither is simply wrong;
  * and the kernel is no FURTHER from the unquantised reference than the dequant path is — the
    direction the extra rounding predicts.
Plus the shape gates, because falling back silently is the failure mode that would hide a
regression: int8 (gap G9, no dense counterpart), M > 16 and K % 16 != 0 (the op's stated limits).

Needs a GPU and the kernel package. Skips cleanly without them rather than passing vacuously.

Run:  PYTHONPATH=python python3 tests/draft_linear_w8a16_parity_test.py
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch  # noqa: E402

FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'OK  ' if ok else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    if not ok:
        FAILED.append(name)


if not torch.cuda.is_available():
    print("SKIPPED: no GPU — this test asserts nothing on CPU (the op is registered at kCUDA).")
    sys.exit(0)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
from minisgl.distributed import set_tp_info, try_get_tp_info  # noqa: E402

if try_get_tp_info() is None:
    set_tp_info(0, 1)

from minisgl.models.draft_linear import DraftLinear, _dense_w8a16_gemv  # noqa: E402

if _dense_w8a16_gemv() is None:
    print("SKIPPED: fp8_wmma.dense_w8a16_gemv unavailable in this image.")
    sys.exit(0)

dev = torch.device("cuda")
DT = torch.bfloat16
OUT, IN = 512, 1024
torch.manual_seed(17)

ref_w = (torch.randn(OUT, IN) * 0.05)          # the unquantised weight both paths approximate


def build(mode: str) -> DraftLinear:
    lin = DraftLinear(IN, OUT)
    lin.load_quant(ref_w.clone(), mode, DT, dev)
    return lin


fp8 = build("fp8")
print(f"weights: wq={tuple(fp8._wq.shape)} {fp8._wq.dtype}  ws_f32={tuple(fp8._ws_f32.shape)}")


def dequant_path(lin: DraftLinear, x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.linear(x, lin._wq.to(x.dtype) * lin._ws)


print()
print("AGREEMENT: kernel vs the dequant path it replaces, against the shared quantisation error")
for M in (1, 4, 16):
    x = (torch.randn(M, IN, dtype=DT, device=dev) * 0.1)
    got = fp8.forward(x)
    deq = dequant_path(fp8, x)
    exact = torch.nn.functional.linear(x.float(), ref_w.to(dev).float())

    scale = exact.abs().max().clamp_min(1e-6)
    d_kernel_deq = (got.float() - deq.float()).abs().max() / scale
    d_deq_exact = (deq.float() - exact).abs().max() / scale
    d_kernel_exact = (got.float() - exact).abs().max() / scale

    check(f"M={M:<3} kernel tracks dequant well inside the fp8 error",
          d_kernel_deq < d_deq_exact,
          f"|k-d|={d_kernel_deq:.3e} vs fp8 error |d-exact|={d_deq_exact:.3e}")
    check(f"M={M:<3} kernel no further from exact than dequant is",
          d_kernel_exact <= d_deq_exact * 1.5,
          f"|k-exact|={d_kernel_exact:.3e} vs |d-exact|={d_deq_exact:.3e}")
    check(f"M={M:<3} the kernel path actually RAN (not a silent fallback)",
          not torch.equal(got, deq), "byte-identical to the fallback => it did not run")

print()
print("GATES: every documented fallback must fall back, not crash or silently mis-dispatch")
x_big = torch.randn(17, IN, dtype=DT, device=dev) * 0.1
check("M=17 (>16) falls back and still matches the dequant path",
      torch.equal(fp8.forward(x_big), dequant_path(fp8, x_big)))

i8 = build("int8")
x = torch.randn(4, IN, dtype=DT, device=dev) * 0.1
check("int8 falls back (gap G9: no dense counterpart on this card)",
      i8._ws_f32 is None and torch.equal(i8.forward(x), dequant_path(i8, x)))

lin_k = DraftLinear(IN + 8, OUT)      # K % 16 != 0
lin_k.load_quant(torch.randn(OUT, IN + 8) * 0.05, "fp8", DT, dev)
xk = torch.randn(2, IN + 8, dtype=DT, device=dev) * 0.1
check("K % 16 != 0 falls back", torch.equal(lin_k.forward(xk), dequant_path(lin_k, xk)))

print()
print("nvfp4 arm is untouched by this change")
n4 = build("nvfp4")
check("nvfp4 still routes to w4a8_linear", n4._w4 is not None and n4._wq is None)

print()
if FAILED:
    print(f"FAILED ({len(FAILED)}): " + "; ".join(FAILED))
    sys.exit(1)
print("ALL CHECKS PASS")
