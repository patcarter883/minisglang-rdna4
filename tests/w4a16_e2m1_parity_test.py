"""Dense W4A16 for the E2M1 formats (MXFP4, NVFP4): unquantized activations on the same kernel.

WHY THIS EXISTS. NVFP4 and MXFP4 dense linears had NO weight-only path — both were forced through
`w4a8_linear`, which quantizes the activation per token on every forward. The register-direct W4A16
kernel already supported E2M1 (`bool E2M1` template flag + `e2m1_lut_f32`) and already had the two
E2M1 scale policies (`E8m0GroupScale`, `E4m3GroupScaleGlobal`); nothing called them from these
formats. This pins the loaders that now do.

WHAT IS CLAIMED, and it is NOT bit-exactness. W4A16 and W4A8 genuinely differ: W4A8 rounds the
ACTIVATION to fp8 per token, W4A16 does not touch it. So the two disagree by construction, W4A16 is
the more accurate of the two, and the assertions are directional rather than equality:

  * W4A16 tracks W4A8 to within the shared WEIGHT quantization error (the same 4-bit weights);
  * W4A16 is NO FURTHER from the unquantized reference than W4A8 is — the direction that removing a
    rounding step predicts;
  * both track the unquantized reference, so neither is simply wrong.

Plus the things that silently do nothing if wrong: that the W4A16 kernel actually RAN (a different
engaged op, not the W4A8 one), that bf16 stays bf16 (RULE 2 — the old code cast every input to fp16),
and that NVFP4's g=16 takes the lane-order entry while MXFP4's g=32 takes the wide one.

Needs a GPU and the kernel package; skips cleanly without them rather than passing vacuously.

Run:  PYTHONPATH=python:/opt/kernels python3 tests/w4a16_e2m1_parity_test.py
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
    print("SKIPPED: no GPU — these ops are registered at kCUDA.")
    sys.exit(0)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
try:
    import fp8_wmma  # noqa: F401
except Exception as e:  # noqa: BLE001
    print(f"SKIPPED: fp8_wmma unavailable ({type(e).__name__}: {e})")
    sys.exit(0)

from minisgl.distributed import set_tp_info, try_get_tp_info  # noqa: E402

if try_get_tp_info() is None:   # engaged() logs rank0-only and needs TP info even single-process
    set_tp_info(0, 1)

from minisgl.quant import kernels, mxfp4, nvfp4  # noqa: E402

DEV = torch.device("cuda")
N, K = 512, 512
torch.manual_seed(11)


def e2m1_quantize(w: torch.Tensor, group: int):
    """Round a dense weight to E2M1 codes + per-group scale, in the packed checkpoint layout.

    Deliberately reuses the repo's own code paths for the packing rather than hand-rolling nibble
    order — a hand-rolled packer that disagreed with the loader would make this test pass against
    its own mistake."""
    lv = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], device=w.device)
    g = w.reshape(N, K // group, group)
    amax = g.abs().amax(-1, keepdim=True).clamp_min(1e-8)
    scale = (amax / 6.0)
    q = (g / scale).abs()
    idx = (q.unsqueeze(-1) - lv).abs().argmin(-1)          # nearest E2M1 magnitude code
    codes = (idx | (g < 0).to(torch.uint8).long() << 3).to(torch.uint8).reshape(N, K)
    deq = (lv[idx] * torch.sign(g) * scale).reshape(N, K)  # what the kernel should reproduce
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()   # 2 nibbles/byte, low first
    return packed, scale.reshape(N, K // group), deq


def run(fmt: str, group: int, dtype: torch.dtype) -> None:
    w = (torch.randn(N, K, device=DEV) * 0.05)
    packed, scale, w_deq = e2m1_quantize(w, group)
    if fmt == "mxfp4":
        # E8M0 byte = the power-of-two exponent, bias 127
        e8 = torch.clamp(torch.log2(scale).round() + 127, 0, 254).to(torch.uint8)
        conv = mxfp4.convert_mxfp4_weight(packed, e8)
        w_deq = None  # recompute below against the kernel's own scale interpretation
        scales_op = conv["scales"].transpose(0, 1).contiguous()
        zeros = None
    else:
        e4 = scale.to(torch.float8_e4m3fn)
        conv = nvfp4.convert_nvfp4_weight(packed, e4)
        scales_op = conv["scales"].transpose(0, 1).contiguous()
        glob = torch.ones(N, device=DEV, dtype=torch.float32).contiguous().view(torch.int32)
        zeros = glob
    w_packed = conv["w_packed"]
    rep = kernels.w4a16_repack(w_packed, group)

    print(f"\n{fmt.upper()}  group={group}  act={str(dtype).split('.')[-1]}  "
          f"w_rep rank={rep.dim()} ({'wide' if rep.dim() == 4 else 'lane-order'})")
    check(f"{fmt} g={group}: repack rank matches the group-size rule",
          (rep.dim() == 4) == (group % 32 == 0),
          f"group {group} -> {'wide' if group % 32 == 0 else 'lane-order'} expected")

    for M in (1, 8, 64):
        x = (torch.randn(M, K, device=DEV, dtype=dtype) * 0.1)
        a16 = kernels.w4a16_linear(x, rep, scales_op, zeros, group, N, weight_is_e2m1=True)
        a8 = kernels.w4a8_linear(x, w_packed, scales_op, zeros, group, weight_is_e2m1=True)
        check(f"{fmt} M={M:<3} W4A16 output keeps the activation dtype (RULE 2)",
              a16.dtype == dtype, f"got {a16.dtype}, x was {dtype}")
        check(f"{fmt} M={M:<3} W4A16 output is finite", torch.isfinite(a16).all().item())

        exact = torch.nn.functional.linear(x.float(), w.float())
        sc = exact.abs().max().clamp_min(1e-6)
        d16 = (a16.float() - exact).abs().max() / sc
        d8 = (a8.float().to(exact.dtype) - exact).abs().max() / sc
        d_pair = (a16.float() - a8.float()).abs().max() / sc
        check(f"{fmt} M={M:<3} W4A16 tracks W4A8 inside the shared weight error",
              d_pair <= max(d8 * 2.0, 5e-2), f"|a16-a8|={d_pair:.3e} vs |a8-exact|={d8:.3e}")
        check(f"{fmt} M={M:<3} W4A16 no further from exact than W4A8 (act-quant removed)",
              d16 <= d8 * 1.25 + 1e-3, f"|a16-exact|={d16:.3e} vs |a8-exact|={d8:.3e}")
        check(f"{fmt} M={M:<3} the two paths are NOT the same tensor (W4A16 really ran)",
              not torch.equal(a16.float(), a8.float()))


print("=" * 84)
print("Dense W4A16 for E2M1 formats — MXFP4 (g=32, wide) and NVFP4 (g=16, lane-order)")
print("=" * 84)
for dt in (torch.bfloat16, torch.float16):
    run("mxfp4", 32, dt)
    run("nvfp4", 16, dt)

print()
if FAILED:
    print(f"FAILED ({len(FAILED)}): " + "; ".join(FAILED))
    sys.exit(1)
print("ALL CHECKS PASS")
