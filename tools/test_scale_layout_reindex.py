#!/usr/bin/env python3
"""CPU proof that the GROUP-MAJOR weight-scale/zero layout is a pure reindex of the old one.

Context: the int4 kernels used to store per-group scales as (N, G) and packed zeros as (N/8, G),
so a WMMA fragment's 16 lanes — which differ only in the output channel — read 16 addresses
`num_groups * 2` bytes apart: 16 separate requests per fragment, once per k-group. At the served
AWQ `group_size=32` that is paid 4x more often than at g=128. The fix stores them GROUP-MAJOR,
(G, N) and (G, N/8), so the channel is the contiguous axis and the read coalesces into one request.

This test pins the loader half of that change. It asserts the new converter output is a bit-exact
TRANSPOSE of the old one, and — the part that actually matters — that the value each KERNEL decodes
for every (channel, group) pair is unchanged:

    old kernel:  wz[(n / 8) * G + g]        >> ((n % 8) * 4)
    new kernel:  wz[g * (N / 8) + (n / 8)]  >> ((n % 8) * 4)

It is deliberately dependency-free (no torch, no GPU) so it runs anywhere, including a host whose
torch install is broken. The GPU-side bit-identity gate is fp8_wmma/tests/test_w4a16_regdirect.py.

Run: python3 tools/test_scale_layout_reindex.py
"""

import random
import sys

PF, BITS, MASK = 8, 4, 0xF
# AWQ stores nibbles in an interleaved lane order; both layouts share this permutation, so it must
# be applied identically or the comparison would be vacuous.
REVERSE_AWQ_PACK_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]


def unpack_zeros(qzeros, G, N):
    """(G, N//PF) packed int32 -> uz[g][n], undoing the AWQ interleave. Shared by both layouts."""
    uz = [[0] * N for _ in range(G)]
    for g in range(G):
        for npk in range(N // PF):
            w = qzeros[g][npk]
            lane = [(w >> (b * BITS)) & MASK for b in range(PF)]
            for j in range(PF):
                uz[g][npk * PF + j] = lane[REVERSE_AWQ_PACK_ORDER[j]]
    return uz


def pack_old(uz, G, N):
    """Channel-major: transpose to (N, G), then pack 8 CHANNELS per int32 down the N axis."""
    uz_t = [[uz[g][n] for g in range(G)] for n in range(N)]
    zeros = [[0] * G for _ in range(N // PF)]
    for j in range(PF):
        for i, row in enumerate(range(j, N, PF)):
            for g in range(G):
                zeros[i][g] |= (uz_t[row][g] & MASK) << (j * BITS)
    return zeros


def pack_new(uz, G, N):
    """Group-major: keep (G, N), pack 8 CHANNELS per int32 along the N axis."""
    zeros = [[0] * (N // PF) for _ in range(G)]
    for j in range(PF):
        for g in range(G):
            for i, col in enumerate(range(j, N, PF)):
                zeros[g][i] |= (uz[g][col] & MASK) << (j * BITS)
    return zeros


def check(N, K, group, seed):
    G = K // group
    rng = random.Random(seed)
    scales = [[rng.random() for _ in range(N)] for _ in range(G)]
    qzeros = [[rng.getrandbits(30) for _ in range(N // PF)] for _ in range(G)]

    uz = unpack_zeros(qzeros, G, N)
    old_zeros, new_zeros = pack_old(uz, G, N), pack_new(uz, G, N)
    old_scales = [[scales[g][n] for g in range(G)] for n in range(N)]  # (N, G)

    fails = []
    if any(scales[g][n] != old_scales[n][g] for g in range(G) for n in range(N)):
        fails.append("scales are not an exact transpose")
    if any(new_zeros[g][i] != old_zeros[i][g] for g in range(G) for i in range(N // PF)):
        fails.append("packed zeros are not an exact transpose")
    # The real gate: identical decoded zero-point per (channel, group) under each kernel's indexing.
    for n in range(N):
        for g in range(G):
            got_old = (old_zeros[n // PF][g] >> ((n % PF) * 4)) & MASK
            got_new = (new_zeros[g][n // PF] >> ((n % PF) * 4)) & MASK
            if got_old != got_new:
                fails.append(f"decode mismatch at n={n} g={g}: {got_old} != {got_new}")
                break
            if got_new != uz[g][n]:
                fails.append(f"decode lost the source nibble at n={n} g={g}")
                break
    return fails


def main():
    # Shapes: served Qwen3.6-35B-A3B-AWQ gemm1/gemm2 (g=32), GLM-4.7-Flash-AWQ (g=128), NVFP4 (g=16),
    # plus a non-power-of-two channel count to catch stride assumptions.
    cases = [
        ("qwen gemm1  N=1024 K=2048 g=32", 1024, 2048, 32),
        ("qwen gemm2  N=2048 K=768  g=32", 2048, 768, 32),
        ("glm         N=3072 K=2048 g=128", 3072, 2048, 128),
        ("nvfp4       N=512  K=1024 g=16", 512, 1024, 16),
        ("odd-ish     N=24   K=192  g=32", 24, 192, 32),
    ]
    bad = 0
    for seed, (name, N, K, group) in enumerate(cases):
        fails = check(N, K, group, seed)
        print(f"[{'ok ' if not fails else 'FAIL'}] {name}")
        for f in fails:
            print(f"        {f}")
        bad += bool(fails)
    print(f"\n{len(cases) - bad}/{len(cases)} shapes bit-exact")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
