#!/usr/bin/env python3
"""Is `ckpt.h2d` slow because of the LINK, the CALL COUNT, or because the source is PAGEABLE?

THE NUMBER. The 48-layer TP=2 boot charges `ckpt.h2d` 116.7 s on rank 0 and 155.1 s on rank 1 for
39.99 GB in 222,252 calls: **342 MiB/s, 176 KiB per call, 525 us per call**, on a link measured at
25,700 MiB/s. It is the largest cost inside Stage B and the second largest in the boot.

WHAT IS ALREADY RULED OUT. The source is NOT a lazily-faulted mmap view: `_shard_qwen4_exp` ends in
`.clone()` on both routed-expert branches, so the file pages are faulted in during `ckpt.shard` —
and `ckpt.shard` prices out at 77.84 GB / 44.1 s = 1769 MiB/s, which is the same rate a cold-shard
mmap read measures standalone (`ckpt_read_rate.py`: 1680 MiB/s). The read path is accounted for.
So `.to(device)` receives a CONTIGUOUS, RESIDENT, ordinary heap tensor and still runs at 342 MiB/s.

THE THREE CANDIDATES, which call for different fixes:
  * PAGEABLE SOURCE — the driver cannot DMA from unpinned memory, so every call stages through an
    internal pinned bounce buffer, chunked and synchronous. Fix: one reusable pinned staging tensor.
  * PER-CALL OVERHEAD — 222k launches at a fixed cost. Fix: batch, which is a real refactor.
  * SIZE — 176 KiB is simply below the link's efficient point. Fix: nothing local helps.

This separates them: the same total bytes, at the boot's actual per-call size, from a pageable
source, from a pinned staging buffer, and in one big transfer. If pinned staging is close to the
big-transfer rate then the fix is a staging buffer and it is a few lines; if pinned staging is no
better than pageable then the cost is per-call and a staging buffer would be dead code.
"""
import os
import sys
import time

sys.path.insert(0, "/engine/python")
import torch  # noqa: E402

MIB = 1 << 20
# The boot's real shape: 222,252 calls, 39.99 GB, 176 KiB mean.
CALL_KIB = int(os.environ.get("H2D_CALL_KIB", "176"))
CALLS = int(os.environ.get("H2D_CALLS", "4000"))
NB = CALL_KIB * 1024

dev = torch.device("cuda:0")
torch.cuda.init()
print(f"device: {torch.cuda.get_device_name(0)}   {CALLS} calls x {CALL_KIB} KiB = "
      f"{CALLS * NB / MIB:.0f} MiB", flush=True)

src_pageable = [torch.empty(NB, dtype=torch.uint8) for _ in range(64)]
for t in src_pageable:
    t.random_(0, 255)
stage = torch.empty(NB, dtype=torch.uint8).pin_memory()
big = torch.empty(CALLS * NB, dtype=torch.uint8)


def bench(label, fn):
    fn(8)  # warm
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn(CALLS)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    tot = CALLS * NB
    print(f"  {label:<44} {dt:7.3f} s  {tot / MIB / dt:9.1f} MiB/s  "
          f"{dt / CALLS * 1e6:8.1f} us/call", flush=True)
    return tot / MIB / dt


def pageable(n):
    for i in range(n):
        src_pageable[i % 64].to(dev)


def pinned_stage(n):
    for i in range(n):
        stage.copy_(src_pageable[i % 64])
        stage.to(dev, non_blocking=False)


def pinned_stage_nb(n):
    for i in range(n):
        stage.copy_(src_pageable[i % 64])
        stage.to(dev, non_blocking=True)
        torch.cuda.synchronize()


def one_big(n):
    big[:n * NB].to(dev)


def one_big_pinned(n):
    p = big[:n * NB].pin_memory()
    p.to(dev)


print("\n=== per-call, the boot's shape ===")
a = bench("A pageable .to(device)   [what ships]", pageable)
b = bench("B pinned staging, blocking", pinned_stage)
c = bench("C pinned staging, non_blocking+sync", pinned_stage_nb)
print("\n=== reference: the same bytes in ONE transfer ===")
d = bench("D one pageable transfer", one_big)
print(f"\nB/A = {b / a:.2f}x   C/A = {c / a:.2f}x   ceiling(D)/A = {d / a:.2f}x", flush=True)
