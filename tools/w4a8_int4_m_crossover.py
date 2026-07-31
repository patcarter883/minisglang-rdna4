"""Is `_W4A8_GEMV_MAX_INT4 = 8` a CLIFF (cap the verify width there) or a CROSSOVER (don't)?

`spec/width.py` caps the adaptive verify ladder at MAX_VERIFY_ROWS-1 so the flat verify M stays
<= 16 at bs=1. A review asked whether the cap must be 8 instead on an int4 W4A8 checkpoint, since
`quant/kernels.py:_pick_dense_kernel` leaves `decode_gemv` for `prefill_wmma` above M=8 there
(vs 16 for e2m1). Those are different claims:

  CLIFF     the kernel the dispatcher picks ABOVE the threshold is SLOWER than the one below it,
            so crossing costs throughput for nothing -> the ladder must not cross it.
  CROSSOVER the dispatcher switches BECAUSE the other kernel is faster from there up -> capping the
            width at 8 would give up rows for no reason, i.e. a pessimization.

So: time BOTH kernels at the same M, min-of-N (a mean read a 1.02x effect as 1.21x once), on int4
weights, for the M range the ladder can reach. Verdict is printed, not asserted.

Run (single card):
  gpu-lease -n 1 -- docker run --rm --device /dev/kfd --device /dev/dri --group-add video \
    --security-opt seccomp=unconfined --security-opt label=disable --ipc host --shm-size 16gb \
    -e HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES -e ROCR_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES \
    -v <worktree>:/engine minisgl-rdna4:lean \
    python /engine/tools/w4a8_int4_m_crossover.py
"""
import sys
import time

import torch

sys.path.insert(0, "/engine/python")
sys.path.insert(0, "/opt/kernels")

import fp8_wmma  # noqa: E402


def w4a8_linear(x, packed, scales, zeros, group, kernel):
    """The op `minisgl.quant.kernels.w4a8_linear` calls, minus its `engaged()` logging (which needs
    a TP context this standalone probe has no reason to build). Same kernel, same arguments."""
    return fp8_wmma.mmq_fp8_gemm(
        x, packed, scales, kernel=kernel, w_zeros=zeros, weight_is_e2m1=False)


def w4a8_dispatch(x, packed, scales, zeros, group):
    """What the SERVE path would pick for this M (quant/kernels._pick_dense_kernel)."""
    from minisgl.quant.kernels import _pick_dense_kernel

    return w4a8_linear(x, packed, scales, zeros, group,
                       _pick_dense_kernel(x.shape[0], False, group))

DEV = "cuda"
GROUP = 128
REPEAT = 5          # min-of-N, never a mean
ITERS = 50


def make_w(n: int, k: int):
    packed = torch.randint(-(2**31), 2**31 - 1, (n, k // 8), dtype=torch.int32, device=DEV)
    scales = (torch.rand(n, k // GROUP, device=DEV, dtype=torch.float16) * 0.01 + 0.001)
    return packed, scales


def bench(fn) -> float:
    best = float("inf")
    for _ in range(REPEAT):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(ITERS):
            fn()
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - t0) / ITERS * 1e6)  # us
    return best


def main() -> None:
    print(f"torch {torch.__version__}  dev {torch.cuda.get_device_name(0)}")
    # K must be a multiple of 512 for the int4 decode GEMV ("v11 needs K % 512 == 0"),
    # so the K=11008 MLP-down shape simply cannot take that kernel and is excluded.
    shapes = [(4096, 4096), (11008, 4096), (6144, 2048), (2048, 6144)]
    for n, k in shapes:
        packed, scales = make_w(n, k)
        print(f"\n=== int4 W4A8 dense  N={n} K={k} group={GROUP}  (us, min of {REPEAT}x{ITERS}) ===")
        print(f"{'M':>4} {'decode_gemv':>12} {'prefill_wmma':>13} {'gemv/wmma':>10} "
              f"{'us/row gemv':>12} {'us/row wmma':>12}")
        for m in (1, 4, 8, 9, 12, 16):
            x = torch.randn(m, k, device=DEV, dtype=torch.bfloat16)
            try:
                g = bench(lambda: w4a8_linear(x, packed, scales, None, GROUP, kernel="decode_gemv"))
            except Exception as e:                                  # noqa: BLE001
                g = float("nan")
                print(f"  decode_gemv M={m} unavailable: {type(e).__name__}: {e}")
            w = bench(lambda: w4a8_linear(x, packed, scales, None, GROUP, kernel="prefill_wmma"))
            ratio = g / w if w else float("nan")
            print(f"{m:>4} {g:>12.1f} {w:>13.1f} {ratio:>10.2f} {g / m:>12.1f} {w / m:>12.1f}")
        # The decisive comparison for the width cap: total cost of a width-7 step (M=8, gemv) vs a
        # width-15 step (M=16, whatever the dispatcher picks). Capping at 8 is only right if the
        # latter is more than 2x the former (i.e. worse PER ROW past the boundary).
        x8 = torch.randn(8, k, device=DEV, dtype=torch.bfloat16)
        x16 = torch.randn(16, k, device=DEV, dtype=torch.bfloat16)
        c8 = bench(lambda: w4a8_dispatch(x8, packed, scales, None, GROUP))
        c16 = bench(lambda: w4a8_dispatch(x16, packed, scales, None, GROUP))
        print(f"  dispatcher: M=8 {c8:.1f}us  M=16 {c16:.1f}us  -> M=16 costs {c16 / c8:.2f}x M=8 "
              f"({'CLIFF: cap at 8' if c16 / c8 > 2.0 else 'NO CLIFF at 8: crossover only'})")


if __name__ == "__main__":
    main()
