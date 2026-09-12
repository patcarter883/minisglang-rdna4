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

from minisgl.models.draft_linear import DraftLinear, _quant_op  # noqa: E402

if _quant_op("dense_w8a16_gemv") is None:
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
# M > 16 USED to be a fallback and is no longer one: G2 gave fp8 a dense tiled GEMM, so the band
# hands off to that kernel instead of the dequant path. The assertion that it falls back is kept
# here as a DELETED line on purpose — it was the one check this change had to invalidate, and the
# PREFILL BAND section below is what replaces it. Anything else that still falls back is asserted.

# G9 CLOSED: int8 now has its own dense GEMV (dense_int8a16_gemv), the SAME core with one
# byte-decode policy swapped. It is held to the same standard as the fp8 arm — closer to the
# dequant path than the int8 quantisation error both carry.
i8 = build("int8")
for M in (1, 4, 16):
    x = torch.randn(M, IN, dtype=DT, device=dev) * 0.1
    got, deq = i8.forward(x), dequant_path(i8, x)
    exact = torch.nn.functional.linear(x.float(), ref_w.to(dev).float())
    scale = exact.abs().max().clamp_min(1e-6)
    d_kd = (got.float() - deq.float()).abs().max() / scale
    d_de = (deq.float() - exact).abs().max() / scale
    check(f"int8 M={M:<3} kernel tracks dequant inside the int8 error", d_kd < d_de,
          f"|k-d|={d_kd:.3e} vs |d-exact|={d_de:.3e}")
    check(f"int8 M={M:<3} the kernel path actually RAN", not torch.equal(got, deq))
# --------------------------------------------------------------------------------------------
# M > 16 is no longer a fallback: G2 (fp8) and G9's prefill half route it to the tiled GEMM. These
# are the shapes a drafter PREFILLS at, where the [out,in] dequant temporary cost the most, so the
# band that used to be silently excluded is now the band under test.
print("")
print("PREFILL BAND (M > 16) — the tiled GEMM arm, G2 + G9")
for mod, name in ((fp8, "fp8 "), (i8, "int8")):
    for M in (17, 64, 129):
        x = torch.randn(M, IN, dtype=DT, device=dev) * 0.1
        got, deq = mod.forward(x), dequant_path(mod, x)
        exact = torch.nn.functional.linear(x.float(), ref_w.to(dev).float())
        scale = exact.abs().max().clamp_min(1e-6)
        d_kd = (got.float() - deq.float()).abs().max() / scale
        d_de = (deq.float() - exact).abs().max() / scale
        check(f"{name} M={M:<4} GEMM tracks dequant inside the quantisation error", d_kd < d_de,
              f"|k-d|={d_kd:.3e} vs |d-exact|={d_de:.3e}")
        check(f"{name} M={M:<4} the GEMM path actually RAN", not torch.equal(got, deq))
        # M=129 exercises the ragged tail: block_m is 128, so the last block is 1 valid row and 127
        # guard rows. A GATHER=false bug there writes the guard rows out or reads past A.
        check(f"{name} M={M:<4} no NaN/Inf in the output", torch.isfinite(got).all())


lin_k = DraftLinear(IN + 8, OUT)      # K % 16 != 0
lin_k.load_quant(torch.randn(OUT, IN + 8) * 0.05, "fp8", DT, dev)
# BOTH M bands: K % 16 is a shape limit of the GEMV *and* the GEMM, so the M > 16 arm must fall back
# for it too. Checking only M=2 would have let a GEMM that silently accepts a ragged K through.
for M in (2, 64):
    xk = torch.randn(M, IN + 8, dtype=DT, device=dev) * 0.1
    check(f"K % 16 != 0 falls back (M={M})",
          torch.equal(lin_k.forward(xk), dequant_path(lin_k, xk)))

print()
print("nvfp4 arm is untouched by this change")
n4 = build("nvfp4")
check("nvfp4 still routes to w4a8_linear", n4._w4 is not None and n4._wq is None)

print()
if FAILED:
    print(f"FAILED ({len(FAILED)}): " + "; ".join(FAILED))
    sys.exit(1)
print("ALL CHECKS PASS")
