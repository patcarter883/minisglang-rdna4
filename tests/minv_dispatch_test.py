"""minv_linear dispatch gate: re-routing a shape must be BIT-IDENTICAL, and must stay M-invariant.

The reason layers/minv.py is allowed to switch kernels on shape at all is the family invariant — rd,
pipe and lds run the same fixed 16-wide K-reduction with no split-K, so they agree bit-for-bit. This
test refuses to take that on trust for the shapes the dispatch was re-routed on:

  1. minv_linear vs the LDS kernel (the family's reference reduction order) on every real Gemma4 /
     DiffusionGemma shape that reaches it, at the M values a serve issues — decode, spec-verify /
     short extend, the 256-token canvas, chunked prefill. Checks the DISPATCH, not just the kernels.
  2. M-invariance ACROSS a kernel crossover: rows computed at an M that lands on rd must equal the
     same rows at an M that lands on pipe. This is the property the file exists to provide and the
     only new hazard a re-routed dispatch can introduce.
  3. That the two shapes this change touches (ragged OUT, and IN % PBK != 0) still come from
     dense_gemm rather than silently falling back to the M-VARIANT F.linear.

Needs a GPU and the dense_gemm package. Run inside the serve image:

    gpu-lease -n 1 -- docker run --rm --device /dev/kfd --device /dev/dri --group-add video \
      -v <worktree>:/engine -e PYTHONPATH=/opt/kernels:/engine/python \
      --entrypoint bash minisgl-rdna4:<tag> -lc 'python /engine/tests/minv_dispatch_test.py'
"""
from __future__ import annotations

import sys

import torch
import torch.nn.functional as F

# (name, IN, OUT) — per-rank at TP=2, i.e. what a served rank actually calls.
SHAPES = [
    ("mlp.gate_up", 2816, 2112),
    ("mlp.down", 1056, 2816),      # IN=1056: NOT a multiple of the pipe K-chunk -> exercises the K-tail
    ("router", 2816, 128),         # narrow OUT -> must stay on rd at every M
    ("lm_head", 2816, 8192),       # LM-head class, narrowed so the test stays cheap
    ("ragged_out", 2816, 2000),    # OUT % BN != 0 -> the LDS masking path
]
MS = [1, 8, 16, 17, 32, 64, 128, 129, 192, 256, 384, 512]
GEMV_MAXM = 16   # at or below this, minv_linear uses the shared bf16 GEMV (a different, also
                 # M-invariant, reduction order by design) — gate on closeness, not bit-equality.


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: no HIP/CUDA device")
        return 0
    import dense_gemm as dg

    from minisgl.layers.minv import minv_linear, minv_supported

    print(f"PROVENANCE dense_gemm={dg.__file__}")
    fails = 0

    def ref(x, w):
        return dg.dense_gemm(x.contiguous(), w, 64, 64).float()

    # --- 1. the dispatch is bit-identical to the family reference -------------------------------
    print("\n== minv_linear vs the LDS reference ==")
    for name, IN, OUT in SHAPES:
        torch.manual_seed(0)
        w = (torch.randn(OUT, IN, device="cuda") * 0.02).to(torch.bfloat16)
        worst = 0.0
        for M in MS:
            x = torch.randn(M, IN, device="cuda").to(torch.bfloat16)
            got = minv_linear(x, w)
            assert got.shape == (M, OUT), f"{name} M={M}: shape {tuple(got.shape)}"
            d = (got.float() - ref(x, w)).abs().max().item()
            if M <= GEMV_MAXM:
                if d > 5e-2:
                    print(f"  FAIL {name} M={M}: GEMV path off reference by {d:.3e}")
                    fails += 1
                continue
            worst = max(worst, d)
            if d != 0.0:
                print(f"  FAIL {name} M={M}: max|minv - lds| = {d:.3e}")
                fails += 1
        print(f"  {name:<12} IN={IN:<5} OUT={OUT:<6} M>{GEMV_MAXM}: max|minv - lds| = {worst:.3e}"
              f" {'EXACT' if worst == 0.0 else 'DIFFERS'}")
        del w

    # --- 2. M-invariance across the rd -> pipe crossover -----------------------------------------
    print("\n== M-invariance across the kernel crossover (rows[16:32]) ==")
    for name, IN, OUT in SHAPES:
        torch.manual_seed(1)
        w = (torch.randn(OUT, IN, device="cuda") * 0.02).to(torch.bfloat16)
        xfull = torch.randn(512, IN, device="cuda").to(torch.bfloat16)
        base, worst = None, 0.0
        ms = [m for m in MS if m >= 32]   # rows[16:32] needs 32 rows to exist at every M compared
        for M in ms:
            got = minv_linear(xfull[:M].contiguous(), w)[16:32].float()
            assert got.shape[0] == 16, f"{name} M={M}: sliced {got.shape[0]} rows"
            if base is None:
                base = got
                continue
            worst = max(worst, (got - base).abs().max().item())
        if worst != 0.0:
            print(f"  FAIL {name}: rows[16:32] drift {worst:.3e} across M")
            fails += 1
        print(f"  {name:<12} M in {ms}: max|d vs M={ms[0]}| = {worst:.3e} "
              f"{'OK' if worst == 0.0 else 'FAIL'}")
        del w, xfull

    # --- 3. the touched shapes do NOT fall back to F.linear ---------------------------------------
    print("\n== the re-routed shapes come from dense_gemm, not F.linear ==")
    for name, IN, OUT, M in [("odd-K down_proj", 1056, 2816, 256), ("ragged OUT", 2816, 2000, 256)]:
        torch.manual_seed(2)
        w = (torch.randn(OUT, IN, device="cuda") * 0.02).to(torch.bfloat16)
        x = torch.randn(M, IN, device="cuda").to(torch.bfloat16)
        ok = minv_supported(x, w)
        got = minv_linear(x, w)
        d_ref = (got.float() - ref(x, w)).abs().max().item()
        d_roc = (got.float() - F.linear(x, w).float()).abs().max().item()
        good = ok and d_ref == 0.0 and d_roc > 0.0
        fails += 0 if good else 1
        print(f"  {name:<16} supported={ok} |minv-lds|={d_ref:.3e} |minv-F.linear|={d_roc:.3e} "
              f"{'OK' if good else 'FAIL'}")
        del w

    print("\n" + ("MINV DISPATCH GREEN" if fails == 0 else f"{fails} FAILURES"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
