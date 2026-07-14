"""Does a matrix-core GEMM give bit-identical rows when M (row count) changes?

If rocBLAS/WMMA selects a different kernel or K-reduction order by problem shape, then
  linear(x[:m], W)  !=  linear(x[:M], W)[:m]
even though row i only depends on x[i]·W. That M-dependence is the prime suspect for the
CCA chunked-prefill divergence: each chunk runs the q/k projection at a different M than the
single pass, so rounding-sensitive rows flip by ~bf16 ULP and compound through the network.

Tests bf16 and fp32, over the ZAYA q/k projection shapes, for a range of M splits.
"""
import torch, torch.nn.functional as F

def test(dtype, K, Nout, Mfull, splits, dev="cuda"):
    torch.manual_seed(0)
    x = torch.randn(Mfull, K, device=dev, dtype=dtype)
    W = torch.randn(Nout, K, device=dev, dtype=dtype)
    full = F.linear(x, W)  # [Mfull, Nout]
    print(f"\n=== dtype={dtype} K={K} Nout={Nout} Mfull={Mfull} ===")
    for m in splits:
        part = F.linear(x[:m], W)          # recompute first m rows at M=m
        d = (part - full[:m]).abs()
        ndiff = (d.max(dim=1).values > 0).sum().item()
        print(f"  M={m:4d}: max|Δ|={d.max().item():.3e}  rows_differing={ndiff}/{m}")

def test_pad(dtype, K, Nout, Mfull, splits, pad_to, dev="cuda"):
    """Mitigation test: if we PAD every GEMM's M up to a fixed tile `pad_to` (with zeros), does the
    real region become M-invariant? i.e. does linear(pad(x[:m],pad_to),W)[:m] == linear(pad(x,pad_to),W)[:m]?"""
    torch.manual_seed(0)
    x = torch.randn(Mfull, K, device=dev, dtype=dtype)
    W = torch.randn(Nout, K, device=dev, dtype=dtype)
    def padded_linear(xx):
        m = xx.shape[0]
        P = ((m + pad_to - 1) // pad_to) * pad_to
        xp = torch.zeros(P, K, device=dev, dtype=dtype); xp[:m] = xx
        return F.linear(xp, W)[:m]
    full = padded_linear(x)
    print(f"\n=== PAD-to-{pad_to}  dtype={dtype} K={K} Nout={Nout} Mfull={Mfull} ===")
    for m in splits:
        part = padded_linear(x[:m])
        d = (part - full[:m]).abs()
        ndiff = (d.max(dim=1).values > 0).sum().item()
        print(f"  M={m:4d}: max|Δ|={d.max().item():.3e}  rows_differing={ndiff}/{m}")

def test_int8(K, Nout, Mfull, splits, dev="cuda"):
    """INT8 matmul: is it M-invariant? Integer accumulation has no rounding, so it SHOULD be exact
    regardless of kernel/reduction order. Confirms the RXF (W4A8-int8) path is not a divergence source."""
    torch.manual_seed(0)
    x = torch.randint(-127, 127, (Mfull, K), device=dev, dtype=torch.int8)
    W = torch.randint(-127, 127, (Nout, K), device=dev, dtype=torch.int8)
    try:
        full = torch._int_mm(x, W.t())
    except Exception as e:
        print(f"\n=== INT8 torch._int_mm unavailable: {e} -> reason: integer accumulation is exact/associative -> M-invariant by construction ===")
        return
    print(f"\n=== INT8 _int_mm K={K} Nout={Nout} Mfull={Mfull} ===")
    for m in splits:
        part = torch._int_mm(x[:m].contiguous(), W.t())
        d = (part - full[:m]).abs()
        ndiff = (d.max(dim=1).values > 0).sum().item()
        print(f"  M={m:4d}: max|Δ|={d.max().item()}  rows_differing={ndiff}/{m}")

def main():
    dev = "cuda"
    # ZAYA CCA q/k projection: hidden=2048 -> latent ~ (nq+nk)*head_dim. Use representative shapes.
    for dt in (torch.bfloat16, torch.float32):
        test(dt, K=2048, Nout=1024, Mfull=131, splits=[33, 48, 60, 96, 97, 124])
    # also a big MoE-ish shape
    test(torch.bfloat16, K=2048, Nout=2816, Mfull=160, splits=[33, 48, 64, 96, 128])
    # ACTUAL ZAYA CCA projection shapes: q Nout=1024, k Nout=256, v Nout=128. Find the pad tile that
    # makes each M-invariant (small-N GEMMs may need a larger tile than the N=1024 q proj).
    for Nout in (1024, 256, 128):
        print(f"\n##### CCA proj shape Nout={Nout} #####")
        test(torch.bfloat16, K=2048, Nout=Nout, Mfull=520, splits=[33, 48, 60, 96, 131, 256])
        for pad in (128, 256, 512):
            test_pad(torch.bfloat16, K=2048, Nout=Nout, Mfull=520, splits=[33, 48, 60, 131, 384, 511], pad_to=pad)
    # ROBUST: pad N up to a common stable width (1024) AND M to 128 -> do the small k/v projections
    # run the SAME M-invariant kernel as q? (real cols sliced back). Simulate a real-Nout weight
    # zero-padded to Npad, check the first real-Nout cols are M-invariant across M.
    def test_npad(Nreal, Npad, Mfull=520, splits=(33, 48, 60, 131, 384, 511), tileM=128, dev="cuda"):
        torch.manual_seed(0)
        x = torch.randn(Mfull, 2048, device=dev, dtype=torch.bfloat16)
        W = torch.randn(Nreal, 2048, device=dev, dtype=torch.bfloat16)
        Wp = torch.zeros(Npad, 2048, device=dev, dtype=torch.bfloat16); Wp[:Nreal] = W
        def pl(xx):
            m = xx.shape[0]; P = ((m + tileM - 1)//tileM)*tileM
            xp = torch.zeros(P, 2048, device=dev, dtype=torch.bfloat16); xp[:m] = xx
            return F.linear(xp, Wp)[:m, :Nreal]
        full = pl(x)
        print(f"\n=== N-PAD Nreal={Nreal}->Npad={Npad}  padM={tileM} ===")
        for m in splits:
            d = (pl(x[:m]) - full[:m]).abs(); nd = (d.max(dim=1).values > 0).sum().item()
            print(f"  M={m:4d}: max|Δ|={d.max().item():.3e}  rows_differing={nd}/{m}")
    for Nreal in (256, 128):
        for Npad in (Nreal, 512, 1024):
            test_npad(Nreal, Npad)
    # INT8 (RXF W4A8) M-invariance — expected exact.
    test_int8(K=2048, Nout=1024, Mfull=520, splits=[33, 48, 60, 131, 256, 384])
    # MITIGATION: pad M to a fixed tile -> is the real region M-invariant? Larger Mfull to check that
    # ALL multiples of 128 agree with each other (not just 128 vs 256).
    for pad in (128,):
        test_pad(torch.bfloat16, K=2048, Nout=1024, Mfull=520, splits=[33, 48, 60, 131, 200, 384, 511], pad_to=pad)

if __name__ == "__main__":
    main()
