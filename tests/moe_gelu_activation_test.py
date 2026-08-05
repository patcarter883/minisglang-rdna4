"""Gated-gelu activation policy for the W4A8 MoE path — CPU only, no GPU lease, no weights.

Gemma4's 128 routed experts use HF `gelu_pytorch_tanh`, which is the TANH APPROXIMATION. The trap
this guards is that substituting the exact erf gelu (`layers.activation.gelu_and_mul`) or silu is
never an error — same shapes, same dtype, finite values — so it can only be caught by comparing
numbers. Checks:

  1. `gelu_tanh_and_mul` == F.gelu(gate, approximate="tanh") * up, in fp32 and bf16.
  2. it is NOT the erf gelu (i.e. `gelu_and_mul` is a real, measurable substitution, not a synonym).
  3. `kernels.w4a8_moe` still defaults to "silu" and rejects an unknown activation.

    docker run --rm --entrypoint bash -v <worktree>:/wt \
      -v /home/pat/.cache/huggingface:/root/.cache/huggingface -e HF_HUB_OFFLINE=1 \
      minisgl-rdna4:lean -lc 'cd /wt && PYTHONPATH=/wt/python:/opt/kernels \
        python tests/moe_gelu_activation_test.py'
"""

from __future__ import annotations

import inspect
import sys

import torch
import torch.nn.functional as F


def main() -> int:
    from minisgl.layers.activation import gelu_and_mul, gelu_tanh_and_mul

    torch.manual_seed(0)
    fails = []

    # (1) fp32 reference: the helper is fp32-internal, so this is the tightest the comparison gets.
    x = torch.randn(4096, 2 * 512, dtype=torch.float32) * 3.0
    gate, up = x[:, :512], x[:, 512:]
    ref = F.gelu(gate, approximate="tanh") * up
    got = gelu_tanh_and_mul(x)
    err32 = (got - ref).abs().max().item()
    rel32 = ((got - ref).abs() / ref.abs().clamp_min(1e-6)).max().item()
    print(f"fp32  max|Δ| = {err32:.3e}   max rel = {rel32:.3e}")
    if err32 != 0.0:
        fails.append(f"fp32 gelu_tanh_and_mul is not bit-exact vs F.gelu(tanh)*up (max|Δ|={err32:.3e})")

    # (2) bf16: gemm1's real output dtype. The helper widens to fp32 and rounds once on the way out,
    # so the only admissible error is that single bf16 rounding (2^-8 relative).
    xb = x.to(torch.bfloat16)
    gb, ub = xb[:, :512].float(), xb[:, 512:].float()
    refb = (F.gelu(gb, approximate="tanh") * ub).to(torch.bfloat16)
    gotb = gelu_tanh_and_mul(xb)
    errb = (gotb.float() - refb.float()).abs().max().item()
    print(f"bf16  max|Δ| vs fp32-then-round = {errb:.3e}  (dtype {gotb.dtype})")
    if gotb.dtype != torch.bfloat16 or errb != 0.0:
        fails.append(f"bf16 gelu_tanh_and_mul mismatch (max|Δ|={errb:.3e}, dtype={gotb.dtype})")

    # (3) the erf gelu is a MEASURABLY different function — proves the tanh/erf distinction is not
    # cosmetic and that a silent swap would shift every expert output.
    erf_delta = (gelu_and_mul(x) - ref).abs().max().item()
    silu_delta = ((F.silu(gate) * up) - ref).abs().max().item()
    print(f"erf-gelu substitution would cost max|Δ| = {erf_delta:.3e}")
    print(f"silu     substitution would cost max|Δ| = {silu_delta:.3e}")
    if erf_delta == 0.0:
        fails.append("gelu_and_mul is indistinguishable from the tanh gelu — check the tolerance setup")

    # (4) silu stays the default on the kernel entry point, and an unknown activation is refused
    # rather than coerced.
    from minisgl.quant import kernels

    default = inspect.signature(kernels.w4a8_moe).parameters["activation"].default
    print(f"kernels.w4a8_moe activation default = {default!r}")
    if default != "silu":
        fails.append(f"w4a8_moe activation default changed to {default!r} — silu must stay default")

    from minisgl.layers.moe import _check_activation

    for scheme, act, ok in (("W4A8", "silu", True), ("W4A8", "gelu", True), ("W4A8", "swiglu", False)):
        try:
            _check_activation(scheme, act)
            rejected = False
        except NotImplementedError:
            rejected = True
        if rejected == ok:
            fails.append(f"_check_activation({scheme}, {act}) -> rejected={rejected}, expected {not ok}")

    for f in fails:
        print(f"FAIL: {f}")
    print("PASS" if not fails else f"{len(fails)} FAILURE(S)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
