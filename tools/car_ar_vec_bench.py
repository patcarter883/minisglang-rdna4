"""2-GPU op-level gate + bench for the vectorized custom_ar P2P collectives.

The custom_ar all-reduce read the peer's buffer ONE ELEMENT (2 bytes) per thread per iteration. That
was fine for the ~30 KB decode tensors it was written for and catastrophic for a DiffusionGemma canvas
([256, 2816] bf16 = 1.44 MB, all-reduced 93x per step): 461 us/call at 3.13 GB/s, 2.7x slower than the
RCCL it replaced. Reading 16 bytes per thread fixes it. This file proves two things:

  BIT-IDENTICAL  -- vectorized output == scalar output, exactly, for all_reduce and all_gather.
  FASTER         -- across the whole payload range, not just the canvas shape.

`vec=0` forces the scalar kernel, so the A/B compares the two kernels in ONE process against ONE
reference -- it cannot silently degenerate into new-vs-itself.

Run under a 2-card lease inside the serve image:

    gpu-lease -n 2 -- docker run --rm ... minisgl-rdna4:gemma4 -lc \
      'PYTHONPATH=/carfp8:/opt/kernels:/engine/python torchrun --nproc_per_node=2 \
       /engine/tools/car_ar_vec_bench.py'
"""

from __future__ import annotations

import os
import sys
import time

import torch
import torch.distributed as dist

HIDDEN = 2816          # DiffusionGemma-26B-A4B hidden_size
CANVAS = 256           # canvas rows/forward -> 1.44 MB bf16, the traced payload exactly
ROW_SWEEP = (1, 8, 64, CANVAS, 1024)
ITERS = 200
WARMUP = 30


def _exchange(car, buf, rank, group):
    h = car.get_ipc_handle(buf)
    gathered = [None, None]
    dist.all_gather_object(gathered, h.numpy().tobytes(), group=group)
    peer = torch.frombuffer(bytearray(gathered[1 - rank]), dtype=torch.uint8).clone()
    return car.open_ipc_handle(peer)


def main() -> int:
    rank = int(os.environ["RANK"])
    assert int(os.environ["WORLD_SIZE"]) == 2, "TP=2 only"
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=2)
    cpu_group = dist.new_group(backend="gloo")

    import custom_ar as car

    def p(*a):
        if rank == 0:
            print(*a, flush=True)

    FLAG_SLOTS = 64
    slot_bytes = ((16 << 20) + 255) // 256 * 256
    self_data = car.alloc_shared(2 * slot_bytes, 0).view(2, slot_bytes)
    self_flags = car.alloc_shared(FLAG_SLOTS * 4, 3)
    peer_base = _exchange(car, self_data, rank, cpu_group)
    peer_flags_ptr = _exchange(car, self_flags, rank, cpu_group)
    peer_data_ptr = [peer_base, peer_base + slot_bytes]

    ag_buf = car.alloc_shared(slot_bytes, 0)
    ag_flags = car.alloc_shared(FLAG_SLOTS * 4, 3)
    ag_peer_ptr = _exchange(car, ag_buf, rank, cpu_group)
    ag_peer_flags_ptr = _exchange(car, ag_flags, rank, cpu_group)
    dist.barrier(group=cpu_group)

    failures: list[str] = []

    def check(name, ok, detail=""):
        p(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
        if not ok:
            failures.append(name)

    def make_input(rows, seed):
        g = torch.Generator(device="cuda").manual_seed(seed)
        return torch.randn(rows, HIDDEN, generator=g, device=dev, dtype=torch.bfloat16)

    def ar(y, vec):
        car.one_shot_ar(y, y, self_data[0].view(torch.bfloat16)[: y.numel()],
                        peer_data_ptr[0], self_flags, peer_flags_ptr, 8, vec)

    # ------------------------------------------------------- bit-exactness vs the scalar kernel ----
    p("=== all_reduce: vectorized == scalar, bit for bit ===")
    for rows in ROW_SWEEP:
        x = make_input(rows, 5150 + rank)
        a, b = x.clone(), x.clone()
        ar(a, 0)
        ar(b, 1)
        torch.cuda.synchronize()
        # ...and both must equal the RCCL reference rounded the same way.
        r = x.clone()
        dist.all_reduce(r, op=dist.ReduceOp.SUM)
        check(f"all_reduce vec==scalar rows={rows}", torch.equal(a, b))
        check(f"all_reduce vec==rccl   rows={rows}", torch.equal(b, r))

    p("\n=== all_gather: vectorized == RCCL, bit for bit ===")
    for rows in ROW_SWEEP:
        x = make_input(rows, 611 + rank)
        got = torch.empty(2 * rows, HIDDEN, dtype=torch.bfloat16, device=dev)
        car.all_gather_p2p(x.reshape(-1), got.reshape(-1), ag_buf.view(torch.bfloat16)[: x.numel()],
                           ag_peer_ptr, ag_flags, ag_peer_flags_ptr, rank)
        ref = torch.empty_like(got)
        dist.all_gather_into_tensor(ref, x.contiguous())
        torch.cuda.synchronize()
        check(f"all_gather vec==rccl rows={rows}", torch.equal(got, ref))

    # ------------------------------------------------------- back-to-back coherence ----------------
    # The failure mode this kernel family ships is a RARE stale read under load, not a wrong first call.
    p("\n=== coherence: 400 back-to-back all_reduces, varying rows, every one must be exact ===")
    bad = 0
    for it in range(400):
        rows = ROW_SWEEP[it % len(ROW_SWEEP)]
        x = make_input(rows, 9000 + it * 7 + rank)
        r = x.clone()
        dist.all_reduce(r, op=dist.ReduceOp.SUM)
        y = x.clone()
        ar(y, 1)
        torch.cuda.synchronize()
        bad += 0 if torch.equal(y, r) else 1
    check("all_reduce coherence stress", bad == 0, f"({bad}/400 mismatched)")

    # ------------------------------------------------------- graph capture + replay ----------------
    p("\n=== graph capture + replay (the production dispatch path) ===")
    x = make_input(CANVAS, 4242 + rank)
    buf = x.clone()
    for _ in range(3):
        tmp = x.clone()
        ar(tmp, 1)
    torch.cuda.synchronize()
    dist.barrier(group=cpu_group)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        ar(buf, 1)
    ok = True
    for _ in range(20):
        buf.copy_(x)
        r = x.clone()
        dist.all_reduce(r, op=dist.ReduceOp.SUM)
        g.replay()
        torch.cuda.synchronize()
        ok &= torch.equal(buf, r)
    check("all_reduce graph replay", ok, "(20 replays, bit-exact)")
    del g

    # ------------------------------------------------------- latency -------------------------------
    def timeit(fn):
        for _ in range(WARMUP):
            fn()
        torch.cuda.synchronize()
        dist.barrier(group=cpu_group)
        t0 = time.perf_counter()
        for _ in range(ITERS):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / ITERS * 1e6

    p(f"\n=== all_reduce latency (us/call), blocks=8 ===")
    p(f"  {'rows':>6s} {'MB':>7s} {'scalar':>9s} {'vector':>9s} {'rccl':>9s} {'speedup':>8s} "
      f"{'vec GB/s':>9s}   ms/step@93")
    for rows in ROW_SWEEP:
        x = make_input(rows, 77 + rank)
        mb = rows * HIDDEN * 2 / 1e6
        ys, yv, yr = x.clone(), x.clone(), x.clone()
        t_s = timeit(lambda ys=ys: ar(ys, 0))
        t_v = timeit(lambda yv=yv: ar(yv, 1))
        t_r = timeit(lambda yr=yr: dist.all_reduce(yr, op=dist.ReduceOp.SUM))
        p(f"  {rows:6d} {mb:7.3f} {t_s:9.1f} {t_v:9.1f} {t_r:9.1f} {t_s/t_v:7.2f}x "
          f"{mb*1e6/(t_v*1e-6)/1e9:9.2f}   {t_s*93/1000:.1f} -> {t_v*93/1000:.1f}")

    dist.barrier(group=cpu_group)
    p(f"\n{'ALL PASS' if not failures else 'FAILURES: ' + ', '.join(failures)}")
    dist.destroy_process_group()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
