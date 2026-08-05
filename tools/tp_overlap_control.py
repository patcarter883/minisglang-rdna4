"""CONTROL for the comms/compute overlap result: does ANYTHING overlap on a side stream on this box?

tp_collective_regime_sweep.py measured a side-stream all-reduce as NEGATIVE -- overlapped was slower
than serial at every payload. That has two very different explanations and the fix (or the decision to
stop) depends on which:

  (A) COLLECTIVE-SPECIFIC. `one_shot_ar` is a spin-wait P2P kernel: it busy-polls a peer flag and
      streams the peer's buffer over PCIe. If it hogs CUs or serializes against other work, only
      collectives fail to overlap and some other formulation might still win.

  (B) BOX-WIDE. Two streams simply do not run concurrently here, so NO side-stream mechanism can ever
      hide anything and the entire overlap program is dead on arrival regardless of formulation.

So: run the SAME side-stream harness over four different "background" jobs, from a pure-compute GEMM
(which must overlap if streams work at all) through to the collective. Whatever the answer, it is the
same harness, same events, same process -- the arms cannot drift.

  gemm/gemm    -- two independent GEMMs. THE CONTROL. If this does not overlap, the box is (B).
  copy/gemm    -- a big D2D copy under a GEMM. Tests the copy path specifically.
  rccl/gemm    -- RCCL all_reduce under a GEMM. Different kernel family to custom_ar.
  car/gemm     -- custom_ar one_shot_ar under a GEMM. The production collective.

Timed by CUDA events on the main stream (bracketing the whole region), not wall clock, so the ~40 us
per-call synchronize floor on this box does not contaminate the small arms.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist

HIDDEN = 2816
DTYPE = torch.bfloat16
ITERS = 50
WARMUP = 10


def _exchange(car, buf, rank, group):
    h = car.get_ipc_handle(buf)
    gathered = [None, None]
    dist.all_gather_object(gathered, h.numpy().tobytes(), group=group)
    peer = torch.frombuffer(bytearray(gathered[1 - rank]), dtype=torch.uint8).clone()
    return car.open_ipc_handle(peer)


def main() -> int:
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=2)
    cpu = dist.new_group(backend="gloo")
    import custom_ar as car

    def p(*a):
        if rank == 0:
            print(*a, flush=True)

    p(f"# GPU_MAX_HW_QUEUES={os.environ.get('GPU_MAX_HW_QUEUES', '(unset -> ROCm default 4)')}")
    p(f"# HSA_ENABLE_SDMA={os.environ.get('HSA_ENABLE_SDMA', '(unset)')}\n")

    slot = ((48 << 20) + 255) // 256 * 256
    self_data = car.alloc_shared(2 * slot, 0).view(2, slot)
    self_flags = car.alloc_shared(64 * 4, 3)
    peer_base = _exchange(car, self_data, rank, cpu)
    peer_flags = _exchange(car, self_flags, rank, cpu)
    peer_data = [peer_base, peer_base + slot]
    dist.barrier(group=cpu)
    ctr = [0]

    main_s = torch.cuda.current_stream()
    side_s = torch.cuda.Stream()

    def ev_time(fn, iters=ITERS):
        """Device time for `fn`, bracketed by events on the MAIN stream. One sync for the whole loop."""
        for _ in range(WARMUP):
            fn()
        torch.cuda.synchronize()
        dist.barrier(group=cpu)
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(main_s)
        for _ in range(iters):
            fn()
        b.record(main_s)
        torch.cuda.synchronize()
        return a.elapsed_time(b) / iters * 1000.0  # us

    rows = 2048
    x = torch.randn(rows, HIDDEN, device=dev, dtype=DTYPE)
    # Foreground compute: an independent GEMM, sized to be comparable to the collective.
    fw = torch.randn(HIDDEN, 8192, device=dev, dtype=DTYPE)
    fa = torch.randn(rows, HIDDEN, device=dev, dtype=DTYPE)
    # Background GEMM (the control), independent of everything.
    bw = torch.randn(HIDDEN, 8192, device=dev, dtype=DTYPE)
    ba = torch.randn(rows, HIDDEN, device=dev, dtype=DTYPE)
    src = torch.empty(rows * HIDDEN, device=dev, dtype=DTYPE)
    dst = torch.empty_like(src)
    y = x.clone()

    def fg():
        torch.mm(fa, fw)

    def bg_gemm():
        torch.mm(ba, bw)

    def bg_copy():
        dst.copy_(src)

    def bg_rccl():
        dist.all_reduce(y, op=dist.ReduceOp.SUM)

    def bg_car():
        n = y.numel()
        s = ctr[0] & 1
        ctr[0] += 1
        torch.ops.custom_ar_C.one_shot_ar(
            y, y, self_data[s].view(DTYPE)[:n], peer_data[s], self_flags, peer_flags, 8, 1)

    p(f"=== side-stream overlap, foreground = mm([{rows},{HIDDEN}] x [{HIDDEN},8192]) ===")
    p(f"  {'background':>12s} {'bg alone':>9s} {'fg alone':>9s} {'serial':>9s} {'overlap':>9s} "
      f"{'ideal':>9s} {'hidden':>8s} {'of ideal':>9s}")
    for name, bg in (("gemm", bg_gemm), ("copy(D2D)", bg_copy), ("rccl", bg_rccl), ("car", bg_car)):
        def serial():
            bg()
            fg()

        def overlapped():
            ev = torch.cuda.Event()
            ev.record(main_s)
            side_s.wait_event(ev)
            with torch.cuda.stream(side_s):
                y.record_stream(side_s)
                dst.record_stream(side_s)
                bg()
                done = torch.cuda.Event()
                done.record(side_s)
            fg()
            main_s.wait_event(done)

        tb, tf = ev_time(bg), ev_time(fg)
        ts, to = ev_time(serial), ev_time(overlapped)
        ideal = max(tb, tf)
        headroom = ts - ideal
        p(f"  {name:>12s} {tb:9.1f} {tf:9.1f} {ts:9.1f} {to:9.1f} {ideal:9.1f} {ts-to:8.1f} "
          f"{(ts-to)/max(headroom,1e-9)*100:8.0f}%")

    dist.barrier(group=cpu)
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
