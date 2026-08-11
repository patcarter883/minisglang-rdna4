"""GPU probe: `_PlainLinear` in nvfp4 mode vs the bf16 reference, on real drafter weights.

Why this exists. A wrong E2M1 code layout or a mismatched scale convention does NOT raise — it
produces a drafter that runs fine and proposes badly, which is indistinguishable from "4-bit simply
hurts acceptance". That confound is expensive to unpick from a serve, so pin the numerics here.

Checks, in order of how badly each would mislead:
  1. quantize/dequantize round-trip is self-consistent (encoder vs the golden decoder).
  2. the KERNEL agrees with that dequant — i.e. the packed layout this encoder writes is the layout
     `w4a8_linear(weight_is_e2m1=True)` reads. A silent mismatch here is the dangerous one.
  3. end-to-end error vs the bf16 linear is in the band the weight-error probe predicted (~0.095
     relative L2), not merely "small".

Needs ONE card: gpu-lease -n 1 -- docker run ... python /engine/tools/dflash_nvfp4_linear_probe.py
"""
from __future__ import annotations

import glob
import os
import sys

import torch

FAILS: list[str] = []


def check(name: str, got, want, tol: float | None = None) -> None:
    ok = (abs(got - want) <= tol) if tol is not None else (got == want)
    FAILS.append(name) if not ok else None
    extra = "" if tol is None else f" (tol {tol:g})"
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {got!r} vs {want!r}{extra}")


def main() -> int:
    from minisgl.distributed.info import set_tp_info
    from minisgl.models.dflash import _PlainLinear
    from minisgl.quant.nvfp4 import dequantize_nvfp4_folded, quantize_nvfp4_rtn

    set_tp_info(rank=0, size=1)
    dev = torch.device("cuda")
    hits = glob.glob(
        os.path.expanduser(
            "~/.cache/huggingface/hub/models--meta-models--Muse-Glimmer-30B-assistant/snapshots/*/"
        )
    )
    if not hits:
        sys.exit("drafter not cached")
    from safetensors import safe_open

    path = sorted(glob.glob(os.path.join(hits[0], "*.safetensors")))[0]
    with safe_open(path, framework="pt") as h:
        W = h.get_tensor("layers.0.self_attn.q_proj.weight").float()  # [4096, 6656]
    print(f"== weight {tuple(W.shape)} from the real drafter ==")

    # 1. encoder vs golden decoder
    packed, scales = quantize_nvfp4_rtn(W)
    deq = dequantize_nvfp4_folded(packed, scales)
    rel = float((deq - W).norm() / W.norm())
    print(f"  round-trip relative L2 = {rel:.4f}")
    check("round-trip in the expected 4-bit band", round(rel, 2), 0.10, tol=0.03)

    # 2 + 3. the kernel must read what the encoder wrote.
    lin = _PlainLinear(W.shape[1], W.shape[0])
    lin.load_quant(W, "nvfp4", torch.bfloat16, dev)
    x = (torch.randn(64, W.shape[1], device=dev, dtype=torch.bfloat16) * 0.05).contiguous()

    got = lin.forward(x).float()
    ref_bf16 = torch.nn.functional.linear(x, W.to(dev).to(torch.bfloat16)).float()
    ref_deq = torch.nn.functional.linear(x, deq.to(dev).to(torch.bfloat16)).float()

    for nm, t in (("deq", deq), ("got", got), ("ref_bf16", ref_bf16), ("ref_deq", ref_deq)):
        bad = int(torch.isnan(t).sum()) + int(torch.isinf(t).sum())
        if bad:
            print(f"  !! {nm}: {bad} non-finite of {t.numel()}  (amax={t[torch.isfinite(t)].abs().max():.4g})")
    r_kernel_vs_deq = float((got - ref_deq).norm() / ref_deq.norm())
    r_kernel_vs_bf16 = float((got - ref_bf16).norm() / ref_bf16.norm())
    print(f"  kernel vs dequantized-weight linear : {r_kernel_vs_deq:.4f}")
    print(f"  kernel vs bf16 linear               : {r_kernel_vs_bf16:.4f}")
    # The kernel also quantizes ACTIVATIONS to fp8, so it cannot match the dequant reference exactly;
    # but it must be far closer to it than the 4-bit weight error, or the layouts disagree.
    check("kernel agrees with the encoder's layout", r_kernel_vs_deq < 0.04, True)
    check("end-to-end error is 4-bit-shaped, not garbage", r_kernel_vs_bf16 < 0.15, True)
    check("output shape", tuple(got.shape), (64, W.shape[0]))

    # 3-D input: the captured batched propose path calls this with [N, Q, H].
    x3 = (torch.randn(2, 8, W.shape[1], device=dev, dtype=torch.bfloat16) * 0.05).contiguous()
    check("3-D input shape preserved", tuple(lin.forward(x3).shape), (2, 8, W.shape[0]))

    # down_proj stays fp8 under the mixed policy — make sure that arm still works from the same class.
    lin8 = _PlainLinear(W.shape[1], W.shape[0])
    lin8.load_quant(W, "fp8", torch.bfloat16, dev)
    r8 = float((lin8.forward(x).float() - ref_bf16).norm() / ref_bf16.norm())
    print(f"  fp8 arm end-to-end                  : {r8:.4f}")
    check("fp8 arm still sane", r8 < 0.06, True)

    print()
    if FAILS:
        print(f"FAILED ({len(FAILS)}): " + ", ".join(FAILS))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
