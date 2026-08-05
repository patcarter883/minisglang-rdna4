"""The w4a8 dense mid-band dispatch is a measured REGION, not a constant — pin it.

`quant/kernels.py::_pick_dense_kernel` used to send the whole mid-band (gemv_max < M < 64) to
`prefill_wmma` on a comment that had never been measured. It cost up to 3.09x on a single shape.
The replacement fences a real, measured corner with three terms — a width floor, an M ceiling, and
the DEVICE's CU count, because the corner genuinely moves between this box's two cards (the RX 9070
XT has 64 CUs, the RX 9070 has 56, and at N=8192 they want OPPOSITE arms).

This test is CPU-only and needs no GPU: it drives the selector directly and fakes the CU count, so
it pins BOTH cards' behaviour from either machine. What it must catch:

  1. the old "always prefill_wmma in the mid-band" rule never comes back;
  2. the decode and prefill bands are untouched by the mid-band change;
  3. the corner is device-dependent in the direction measured (the whole point of the CU term);
  4. `w4a8_linear` actually PASSES N — a selector that can see N but is called without it silently
     degrades to "never prefill_wmma", which would look like a clean pass everywhere else.

    python tests/w4a8_dense_dispatch_test.py
"""
from __future__ import annotations

import sys
from contextlib import contextmanager

import minisgl.quant.kernels as KK

TILED, PREFILL, GEMV = "wmma_tiled_tuned", "prefill_wmma", "decode_gemv"
ASH = "prefill_wmma_ashuffle"


@contextmanager
def cus(n: int):
    """Force the CU count the selector sees (2 x multi_processor_count on a real device)."""
    orig = KK._device_cu_count
    KK._device_cu_count = lambda: n
    try:
        yield
    finally:
        KK._device_cu_count = orig


def main() -> int:
    fails = []

    def check(got, want, what):
        if got != want:
            fails.append(f"{what}: got {got}, want {want}")
            print(f"  FAIL {what}: {got} != {want}")
        else:
            print(f"  ok   {what}: {got}")

    pick = KK._pick_dense_kernel

    # --- 1. decode band and prefill band are unchanged -----------------------------------------
    print("\n== decode band (M <= gemv cap) is decode_gemv regardless of N ==")
    for m in (1, 8, 16):
        for n in (2048, 34816):
            check(pick(m, False, 32, k=2816, n=n), GEMV, f"M={m} N={n}")
    print("\n== M=17 with a K the GEMV cannot consume still leaves the decode band ==")
    check(pick(16, False, 32, k=2820, n=2048), TILED, "M=16 K=2820 (K%32!=0)")

    print("\n== prefill band (M >= 64) is wmma_tiled_tuned at every N, both cards ==")
    for cu in (56, 64):
        with cus(cu):
            for m in (64, 128, 2048):
                for n in (1024, 8192, 9216, 34816):
                    check(pick(m, False, 32, k=2816, n=n), TILED, f"cu={cu} M={m} N={n}")

    # --- 2. the mid-band: narrow N is tiled on BOTH cards ---------------------------------------
    print("\n== mid-band, narrow N (< the width floor): wmma_tiled_tuned, both cards ==")
    for cu in (56, 64):
        with cus(cu):
            for n in (1024, 2048, 2816, 4096, 5120, 6144):
                for m in (17, 32, 40, 63):
                    check(pick(m, False, 32, k=2816, n=n), TILED, f"cu={cu} M={m} N={n}")

    # --- 3. the corner, and the fact that it MOVES between the two cards ------------------------
    # Measured (tools/w4a8_dense_midband_crosscard.py, 2026-08-05, M<=48):
    #   GPU 0, RX 9070 XT, 64 CUs: prefill wins 9216/10240/11264/17408; TILED wins 8192 and 16384
    #   GPU 1, RX 9070,    56 CUs: prefill wins ALL of 8192/9216/10240/11264/16384/17408
    print("\n== the wide-N corner on the RX 9070 XT (64 CUs) ==")
    with cus(64):
        for n in (9216, 10240, 11264, 17408, 34816):
            for m in (17, 32, 40, 48):
                check(pick(m, False, 32, k=2816, n=n), PREFILL, f"64CU M={m} N={n}")
        for n in (8192, 16384):  # ceil(N/128) = 64 and 128 -> whole 64-CU waves
            for m in (17, 32, 40, 48):
                check(pick(m, False, 32, k=2816, n=n), TILED, f"64CU M={m} N={n} (full wave)")

    print("\n== the SAME shapes on the RX 9070 (56 CUs) -- the corner moves ==")
    with cus(56):
        for n in (8192, 9216, 10240, 11264, 16384, 17408):
            for m in (17, 32, 40, 48):
                check(pick(m, False, 32, k=2816, n=n), PREFILL, f"56CU M={m} N={n}")

    print("\n== M ceiling: above it the corner closes on both cards ==")
    for cu in (56, 64):
        with cus(cu):
            for n in (9216, 17408):
                for m in (49, 56, 63):
                    check(pick(m, False, 32, k=2816, n=n), TILED, f"cu={cu} M={m} N={n}")

    # --- 3b. the THIRD arm: the tall-K / mid-N box, confirmed on both cards ---------------------
    print("\n== the ashuffle box (K>=5120, 5120<=N<=6144) -- same on both cards ==")
    for cu in (56, 64):
        with cus(cu):
            for k_, n in ((5120, 6144), (8704, 5120), (5120, 5120)):
                for m in (17, 32, 48, 63):
                    check(pick(m, False, 32, k=k_, n=n), ASH, f"cu={cu} K={k_} N={n} M={m}")
    print("\n== just outside the box it is wmma_tiled_tuned (ashuffle loses 1.3-1.8x there) ==")
    with cus(64):
        for k_, n in ((5120, 4096), (5120, 1024), (2048, 6144), (4096, 6144)):
            for m in (17, 32, 48):
                check(pick(m, False, 32, k=k_, n=n), TILED, f"K={k_} N={n} M={m}")
    print("\n== the box does not leak into the decode or prefill bands ==")
    with cus(64):
        check(pick(16, False, 32, k=5120, n=6144), GEMV, "K=5120 N=6144 M=16")
        check(pick(64, False, 32, k=5120, n=6144), TILED, "K=5120 N=6144 M=64")
    print("\n== K unknown cannot reach the ashuffle box ==")
    with cus(64):
        check(pick(32, False, 32, n=6144), TILED, "K=None N=6144 M=32")

    # --- 4. no N -> the safe arm; and group-16 e2m1 still forced to tiled ------------------------
    print("\n== N unknown falls back to wmma_tiled_tuned (never to the old always-prefill) ==")
    with cus(64):
        for m in (17, 32, 40, 63):
            check(pick(m, False, 32, k=2816), TILED, f"M={m} n=None")
    print("\n== group%32 != 0 pins BOTH prefill arms off -- e2m1 AND plain int4 ==")
    with cus(64):
        for m in (17, 40, 64):
            check(pick(m, True, 16, k=2816, n=17408), TILED, f"e2m1 g16 M={m}")
            # int4 g=16 exists in the wild (CohereLabs North-Mini-Code w4a16) and used to fall
            # straight through to prefill_wmma, which hard-requires group_size % 32 == 0.
            check(pick(m, False, 16, k=2816, n=17408), TILED, f"int4 g16 wide-N M={m}")
            check(pick(m, False, 16, k=5120, n=6144), TILED, f"int4 g16 ash-box M={m}")

    # --- 5. w4a8_linear must actually pass N ----------------------------------------------------
    print("\n== w4a8_linear passes N to the selector (a silent None would fake a pass above) ==")
    import inspect

    src = inspect.getsource(KK.w4a8_linear)
    ok = "_pick_dense_kernel(" in src and "n=w_packed.shape[0]" in src
    check(ok, True, "w4a8_linear -> _pick_dense_kernel(..., n=w_packed.shape[0])")

    print(f"\n{'FAILED: ' + str(len(fails)) if fails else 'PASS'}")
    for f in fails:
        print("   " + f)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
