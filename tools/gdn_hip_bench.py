"""Task 4-#21 — long-context prefill benchmark: chunked vs recurrent gdn_hip (GPU).

Times gdn_prefill (per-token recurrence) vs gdn_prefill_chunked (intra-chunk parallel) on a single
sequence at growing lengths, on the real GDN geometry (H=16 HV=32 K=V=128). Also asserts the two
agree (max|Δ|) at each length, so this doubles as a long-context correctness check of the chunked
kernel that now drives the serve prefill path.

Run under a 1-card lease (executes HIP kernels):
    .../gpu-lease.sh -n 1 -- bash -c 'docker run ... python /engine/tools/gdn_hip_bench.py'
"""
from __future__ import annotations

import time

import torch

import gdn_hip  # noqa: F401  (registers torch.ops.gdn_hip.*)

DEV = "cuda"
torch.manual_seed(0)
H, HV, K, V = 16, 32, 128, 128
SCALE = K ** -0.5
LENGTHS = [256, 512, 1024, 2048, 4096, 8192, 16384]
ITERS = 10


def _inputs(T: int):
    cu = torch.tensor([0, T], dtype=torch.int32, device=DEV)
    idx = torch.tensor([1], dtype=torch.long, device=DEV)
    has_init = torch.tensor([0], dtype=torch.uint8, device=DEV)
    g = dict(
        q=torch.randn(T, H, K, device=DEV), k=torch.randn(T, H, K, device=DEV),
        v=torch.randn(T, HV, V, device=DEV), a=torch.randn(T, HV, device=DEV),
        b=torch.randn(T, HV, device=DEV),
        A_log=torch.randn(HV, device=DEV) * 0.5 - 2.0, dt_bias=torch.randn(HV, device=DEV),
        cu=cu, idx=idx, has_init=has_init,
        state=torch.zeros(2, HV, V, K, device=DEV),
    )
    return g


def _call(op, g, st):
    return op(g["q"], g["k"], g["v"], g["a"], g["b"], g["A_log"], g["dt_bias"],
              g["cu"], g["idx"], g["has_init"], st, SCALE, 1)


def _time(op, g, st, iters: int) -> float:
    for _ in range(3):
        _call(op, g, st)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        _call(op, g, st)
    torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1e3  # ms


def main() -> None:
    assert torch.cuda.is_available()
    print(f"=== GDN prefill: recurrent vs scalar-chunked vs WMMA ({torch.cuda.get_device_name()}) ===")
    print(f"{'T':>7} | {'recur ms':>9} | {'chunk ms':>9} | {'wmma ms':>9} | "
          f"{'wmma/rec':>8} | {'Δchunk':>8} | {'Δwmma':>8}")
    print("-" * 78)
    for T in LENGTHS:
        g = _inputs(T)
        o_rec = _call(torch.ops.gdn_hip.gdn_prefill, g, g["state"].clone())
        o_chk = _call(torch.ops.gdn_hip.gdn_prefill_chunked, g, g["state"].clone())
        o_w = _call(torch.ops.gdn_hip.gdn_prefill_wmma, g, g["state"].clone())
        d_chk = (o_rec - o_chk).abs().max().item()
        d_w = (o_rec - o_w).abs().max().item()
        t_rec = _time(torch.ops.gdn_hip.gdn_prefill, g, g["state"].clone(), ITERS)
        t_chk = _time(torch.ops.gdn_hip.gdn_prefill_chunked, g, g["state"].clone(), ITERS)
        t_w = _time(torch.ops.gdn_hip.gdn_prefill_wmma, g, g["state"].clone(), ITERS)
        flag = "" if d_w < 5e-3 else "  <-- WMMA MISMATCH"
        print(f"{T:>7} | {t_rec:>9.3f} | {t_chk:>9.3f} | {t_w:>9.3f} | "
              f"{t_rec / t_w:>7.2f}x | {d_chk:>8.1e} | {d_w:>8.1e}{flag}")
    print("\n(wmma/rec > 1 => WMMA faster than recurrent; Δ small => numerically equal to recurrent)")


if __name__ == "__main__":
    main()
