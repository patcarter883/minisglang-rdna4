"""QKNormRope — the attention front end every AttentionLayer and drafter shares — fused vs chain.

    gpu-lease -n 1 -- docker run ... minisgl-rdna4:lean -lc \
      'cd /wt && PYTHONPATH=/opt/kernels:/wt/python python tests/qk_norm_rope_layer_test.py'

The kernel test (rdna4-hip-kernels tail/tests/test_qk_norm_rope.py) proves the op equals the norm ->
rope chain. This one proves the ENGINE hands it the right arguments: the same QKNormRope, once on
the fused path and once forced onto its own op-chain fallback, must give BIT-IDENTICAL q/k/v for each
model family's configuration — and the fused path must actually have been TAKEN (a gate that quietly
refuses makes "identical" trivially true).
"""
import sys

import torch

from minisgl.layers import RMSNorm, get_rope
from minisgl.layers.attention import QKNormRope
from minisgl.layers.norm import RMSNormNoScale

DEV = "cuda"
FAILS = []


def ok(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        FAILS.append(name)


def norm(kind, hd, dt, g):
    if kind is None:
        return None
    if kind == "noscale":
        return RMSNormNoScale(eps=1e-6)
    n = RMSNorm(hd, eps=1e-6, plus_one=(kind == "w+1"))
    n.weight = (torch.randn(hd, generator=g) * 0.3 + (0 if kind == "w+1" else 1)).to(dt).to(DEV)
    return n


# name, hq, hk, hd, rd (None = NoPE), norm kind, v_norm, v_is_k, q head-strided (Qwen3.5 q|gate)
CASES = [
    ("qwen3.6", 8, 1, 256, 64, "w+1", False, False, True),
    ("gemma4-sliding", 8, 4, 256, 256, "w", True, False, False),
    ("gemma4-full", 8, 1, 512, 512, "w", True, True, False),
    ("laguna", 16, 2, 128, 64, "w", False, False, False),
    ("muse-nope", 8, 2, 128, None, "noscale", False, False, False),
    ("zaya-rope-only", 8, 2, 128, 128, None, False, False, False),
    ("dflash", 16, 4, 128, 128, "w", False, False, False),
]


def main():
    from minisgl.distributed import set_tp_info
    set_tp_info(0, 1)   # the engage ledger logs on rank 0
    g = torch.Generator().manual_seed(0)
    for dt in (torch.float16, torch.bfloat16):
        for name, hq, hk, hd, rd, kind, v_norm, v_is_k, strided in CASES:
            with torch.device(DEV):   # the engine builds its rope caches on the device
                rot = None if rd is None else get_rope(hd, rd, 8192, 10000.0)
            qn, kn = norm(kind, hd, dt, g), norm(kind, hd, dt, g)
            if kind == "noscale":
                kn = qn
            op = QKNormRope(hq, hk, hd, qn, kn, rot)
            ref_op = QKNormRope(hq, hk, hd, qn, kn, rot)
            v_eps = 1e-6 if v_norm else None
            ref_op._prep = ((dt, v_eps), False)   # forced onto the op-chain fallback
            for n in (1, 7, 300):
                def inputs():
                    # the fallback norms in place: fresh, identical inputs per side
                    gg = torch.Generator().manual_seed(n)
                    if strided:
                        qg = (torch.randn(n, hq, 2 * hd, generator=gg) * 2).to(dt).to(DEV)
                        q = qg[..., :hd]
                    else:
                        q = (torch.randn(n, hq * hd, generator=gg) * 2).to(dt).to(DEV)
                    kv = (torch.randn(n, 2 * hk * hd, generator=gg) * 2).to(dt).to(DEV)
                    k, v = kv.split([hk * hd, hk * hd], dim=-1)   # strided views, as from a split
                    return q, k, (k if v_is_k else v)
                pos = torch.randint(0, 8000, (n,), generator=torch.Generator().manual_seed(n + 1)).to(DEV)
                got = op.forward(*inputs(), pos, v_norm_eps=v_eps)
                ref = ref_op.forward(*inputs(), pos, v_norm_eps=v_eps)
                took = bool(op._prep and op._prep[1])
                eq = [torch.equal(a.reshape(n, -1), b.reshape(n, -1)) for a, b in zip(got, ref)]
                ok(f"{name} {str(dt)[6:]} n={n}", took and all(eq) and got[0].is_contiguous()
                   and got[1].is_contiguous(), f"fused taken {took}  q/k/v equal {eq}")
    print("ALL GREEN" if not FAILS else f"FAILED: {FAILS}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    with torch.inference_mode():
        sys.exit(main())
