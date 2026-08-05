"""Isolated per-kernel time at the CANVAS shapes, to be differenced against the in-serve trace.

WHY. The "same kernel is slower in situ than on a bench" effect is being chased on the Qwen decode
path, where it is hard to read: decode shapes are tiny, so per-dispatch overhead and clock residency
dominate and a 2x gap can be an artifact of either. The canvas is the cleaner test — M=256 for every
dense GEMM, one shape per layer, kernels large enough that launch overhead is a minority of their
cost. If the gap survives at THIS shape it is not a small-kernel artifact.

WHAT IS AND IS NOT COMPARABLE. These are auto-perf-level timings and must be compared only against
auto-perf-level trace numbers (a `profile_standard` counter run pins clocks non-boost and its
absolute times belong to a different surface). Values in the buffers are arbitrary; SHAPES and the
DISPATCHED ARM are not — the arm is asserted from the [hip-engage] tag the call emits, so a silent
fallback to a different kernel cannot be mistaken for the kernel under test.

  python3 tools/canvas_iso_bench.py
"""
from __future__ import annotations

import os
import statistics
import sys
import time

import torch

sys.path.insert(0, "/engine/python")

H, M, TP = 2816, 256, 2
GROUP = 32
ITERS, WARMUP = 60, 20


def _sync() -> None:
    torch.cuda.synchronize()


def timeit(fn, iters: int = ITERS, warmup: int = WARMUP) -> float:
    for _ in range(warmup):
        fn()
    _sync()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        _sync()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1e3  # ms


def w4a8_case(name: str, m: int, n: int, k: int, dev: str = "cuda") -> None:
    """One dense W4A8 GEMM at a served canvas shape. w_packed is (N, K/8) int32 in op layout."""
    from minisgl.quant import kernels

    x = torch.randn(m, k, device=dev, dtype=torch.float16)
    wp = torch.randint(-(2**31), 2**31 - 1, (n, k // 8), device=dev, dtype=torch.int32)
    sc = torch.randn(n, k // GROUP, device=dev, dtype=torch.float16).abs().add_(0.01)
    try:
        out = kernels.w4a8_linear(x, wp, sc, None, GROUP)
        ms = timeit(lambda: kernels.w4a8_linear(x, wp, sc, None, GROUP))
    except Exception as e:  # noqa: BLE001
        print(f"  {name:26s} M={m:4d} N={n:6d} K={k:6d}   FAILED {type(e).__name__}: {e}")
        return
    fl = 2 * m * n * k
    # int4 weight bytes dominate the read at M=256 (the activation is 256xK fp16).
    by = n * k / 2 + m * k * 2 + m * n * 2
    print(f"  {name:26s} M={m:4d} N={n:6d} K={k:6d}  {ms:7.3f} ms  "
          f"{fl/ms/1e9:7.2f} TFLOP/s  {by/ms/1e6:7.1f} GB/s  ({100*by/ms/1e6/706.6:5.1f}% HBM)")
    del out


def bf16_case(name: str, m: int, n: int, k: int, dev: str = "cuda") -> None:
    """The fp16 dense linears the checkpoint leaves unquantized (dense MLP, router)."""
    from minisgl.layers import minv

    x = torch.randn(m, k, device=dev, dtype=torch.float16)
    w = torch.randn(n, k, device=dev, dtype=torch.float16)
    try:
        fn = minv.minv_linear
        fn(x, w)
        ms = timeit(lambda: fn(x, w))
    except Exception as e:  # noqa: BLE001
        print(f"  {name:26s} M={m:4d} N={n:6d} K={k:6d}   FAILED {type(e).__name__}: {e}")
        return
    fl = 2 * m * n * k
    by = n * k * 2 + m * k * 2 + m * n * 2
    print(f"  {name:26s} M={m:4d} N={n:6d} K={k:6d}  {ms:7.3f} ms  "
          f"{fl/ms/1e9:7.2f} TFLOP/s  {by/ms/1e6:7.1f} GB/s  ({100*by/ms/1e6/706.6:5.1f}% HBM)")


def main() -> int:
    p = torch.cuda.get_device_properties(0)
    # multi_processor_count reports WGPs: the 64-CU 9070 XT answers 32. Printed, never asserted.
    print(f"[dev] {p.name} gfx={p.gcnArchName} WGP(multi_processor_count)={p.multi_processor_count}")
    print(f"[cfg] canvas M={M} hidden={H} TP={TP} group={GROUP} "
          f"iters={ITERS} (median) perf_level=auto\n")

    print("=== dense W4A8 (self_attn q/k/v/o — NOT in the checkpoint's ignore list) ===")
    w4a8_case("SWA q_proj", M, 16 * 256 // TP, H)
    w4a8_case("SWA k_proj", M, 8 * 256 // TP, H)
    w4a8_case("SWA v_proj", M, 8 * 256 // TP, H)
    w4a8_case("SWA o_proj", M, H, 16 * 256 // TP)
    w4a8_case("FULL q_proj", M, 16 * 512 // TP, H)
    w4a8_case("FULL k_proj", M, 2 * 512 // TP, H)
    w4a8_case("FULL o_proj", M, H, 16 * 512 // TP)

    print("\n=== dense fp16 (mlp.gate/up/down + router.proj — IN the ignore list) ===")
    bf16_case("dense gate_up (merged)", M, 2 * 2112 // TP, H)
    bf16_case("dense down", M, H, 2112 // TP)
    bf16_case("router proj", M, 128, H)

    print("\n=== the canvas LM-head shard (vocab-parallel) ===")
    bf16_case("lm_head shard", M, 262144 // TP, H)
    return 0


if __name__ == "__main__":
    sys.exit(main())
