"""STEP-1, part 3: the BYTE-SLOPE. us per MB of scale traffic removed, at constant instruction count.

For the two decode GEMV kernels (gemm1_silu kernel="gemv", gemm2_gather_reduce) the scale fold lives
in gemv_decode.h's `group_size >= 32` branch: EXACTLY ONE `ws[g]` load and ONE multiply per 32-k
chunk, for group_size 32, 64 AND 128 alike (there is no GSc compile-time specialisation on the GEMV
path -- that only exists in the tiled WMMA moe_gemm). So sweeping 32/64/128 changes NOTHING but the
size of the scale array, i.e. the DRAM traffic behind an unchanged instruction stream.

Fit time vs scale bytes over that sweep -> slope (us per MB). The proposal (fp16 -> e4m3 uint8 at
group 16) removes 0.0625 B/param of scale traffic; slope x that footprint = the proposal's ceiling,
with the fold-count confound removed.

gemm_scatter (kernel="wmma", the M<=2 path) IS GSc-specialised at 128, so it is swept 32/64 only.
"""
import time

import torch

import fp8_wmma
import moe_hip

H, I_TP, E, TOP_K, BLOCK_M, LAYERS = 2048, 256, 256, 8, 16, 39
HBM_CEIL = 706.6e9


def bench(fn, iters=300, warmup=60):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    best = 1e9
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - t0) * 1e6 / iters)
    return best


def mk_w(n, k, g):
    torch.manual_seed(0)
    w = torch.randint(-(2**31), 2**31 - 1, (E, n, k // 8), dtype=torch.int32, device="cuda")
    s = (torch.rand(E, n, k // g, device="cuda") * 0.02 + 0.005).to(torch.float16)
    return w, s


def linfit(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den, my - (num / den) * mx


print(f"{'kernel':<32}{'M':>3}{'dist':>6}{'g32 us':>9}{'g64 us':>9}{'g128 us':>9}"
      f"{'slope us/MB':>13}{'ceil us/MB':>12}{'proposal us':>12}{'x39 ms/step':>13}")
print("-" * 118)
tot = {}

for M in (1, 8, 16):
    torch.manual_seed(1)
    ids = torch.randint(0, E, (M, TOP_K), dtype=torch.int32, device="cuda")
    sid, eid, ntp = moe_hip.moe_align(ids, E, BLOCK_M)
    tw = torch.rand(M, TOP_K, device="cuda", dtype=torch.float32)
    P, dist = sid.shape[0], int(torch.unique(ids).numel())
    twf = tw.reshape(-1).contiguous()
    x1 = (torch.randn(M, H, device="cuda") * 0.3).to(torch.bfloat16)
    x2 = (torch.randn(P, I_TP, device="cuda") * 0.3).to(torch.bfloat16)

    def g1(w, s):
        return fp8_wmma.mmq_fp8_moe_gemm1_silu(
            x1, w, s, sid, eid, ntp, TOP_K, BLOCK_M, kernel="gemv", weight_is_e2m1=True)

    def g2_scatter(w, s):
        acc = torch.zeros((M, H), dtype=torch.float32, device="cuda")
        fp8_wmma.mmq_fp8_moe_gemm_scatter(
            x2, w, s, sid, eid, ntp, twf, acc, TOP_K, BLOCK_M, kernel="wmma",
            weight_is_e2m1=True, split_k=4)

    def g2_gather(w, s):
        return fp8_wmma.mmq_fp8_moe_gemm2_gather_reduce(
            x2, w, s, sid, eid, ntp, twf, TOP_K, BLOCK_M, weight_is_e2m1=True)

    cases = [("gemm1_silu(gemv) w13", 2 * I_TP, H, g1, (32, 64, 128))]
    cases.append(("gemm_scatter(wmma,splitk4) w2", H, I_TP, g2_scatter, (32, 64))
                 if M <= 2 else ("gemm2_gather_reduce w2", H, I_TP, g2_gather, (32, 64, 128)))

    for name, n, k, fn, grps in cases:
        t, mb = {}, {}
        for g in grps:
            w, s = mk_w(n, k, g)
            f = (lambda w=w, s=s: fn(w, s))
            f()
            t[g] = bench(f)
            mb[g] = dist * n * (k // g) * 2 / 1e6      # scale MB streamed per launch
            w = s = None
            torch.cuda.empty_cache()
        slope, _ = linfit([mb[g] for g in grps], [t[g] for g in grps])   # us per MB
        prop_mb = dist * n * k * 0.0625 / 1e6          # MB the proposal removes (fp16->u8 @ g16)
        prop_us = slope * prop_mb
        cells = "".join(f"{t.get(g, float('nan')):>9.1f}" for g in (32, 64, 128))
        print(f"{name:<32}{M:>3}{dist:>6}{cells}{slope:>13.2f}"
              f"{1e6/HBM_CEIL*1e6:>12.2f}{prop_us:>12.2f}{prop_us*LAYERS/1000:>13.4f}")
        tot[M] = tot.get(M, 0.0) + prop_us * LAYERS / 1000

print()
print("Proposal ceiling per decode step (39 sparse layers), fold-confound removed:")
STEP = {1: 13.51, 8: 31.9, 16: 31.9}
for M, ms in sorted(tot.items()):
    print(f"  M={M:<3} saved {ms:.4f} ms/step   = {100*ms/STEP[M]:.3f}% of the {STEP[M]} ms step")

# Run (no engine, kernels only; CPU cannot import torch on this host):
#   gpu-lease -n 1 -- bash -c 'docker run --rm --device /dev/kfd --device /dev/dri \
#     --group-add video --security-opt seccomp=unconfined --security-opt label=disable \
#     --cap-add SYS_PTRACE --ipc host --shm-size 16gb \
#     -e HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES -e ROCR_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES \
#     -v <dir>:/sp --entrypoint bash minisgl-rdna4:bl16 \
#     -lc "PYTHONPATH=/opt/kernels python3 /sp/nvfp4_scale_byte_slope.py"'
