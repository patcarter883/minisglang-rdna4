"""Where does the TP all-reduce actually cost anything, NOW that one_shot_ar is vectorized?

This exists to kill a stale number before anyone optimizes against it. The "collectives are 33% of the
backbone" figure was measured with the SCALAR custom_ar (461.9 us for a 1.44 MB canvas all-reduce, 3.13
GB/s -- 2.7x slower than the RCCL it replaced). The vectorized kernel landed; the number moved a lot. So
re-measure, then decide.

It answers four questions the overlap design depends on, in one process so the arms cannot drift:

  1. LATENCY vs BANDWIDTH. Sweep rows 1 -> 8192 and print effective GB/s. Overlap is only worth
     building where the collective is BANDWIDTH-bound (time grows with payload); in the latency-bound
     regime the collective is a fixed ~tens of us that no amount of chunking hides.

  2. THE 8 MB CLIFF. The engine caps the custom_ar IPC slot at 8 MB (engine.py: car_max_bytes), and
     CustomARDistributedImpl.all_reduce falls back to RCCL for anything larger. At hidden=2816 bf16
     that cliff is at 1419 rows. Every real chunked prefill chunk (2048 tokens) is ON THE RCCL SIDE OF
     IT. Measure both sides so the fallback is a measured fact, not a code-reading claim.

  3. DOES CHUNKING ALONE PAY? Above the cliff, 2 half-size custom_ar calls each FIT the slot where 1
     full call does not. That is a win available with NO overlap and NO side stream -- pure dispatch
     policy. Measured here as `car x2`.

  4. WHAT IS THE OVERLAP HEADROOM? A collective can only be hidden behind compute that is (a) issued
     after it and (b) independent of it. `serial` vs `overlapped` here brackets the best case: an
     empty-dependency GEMM of a realistic per-chunk size, with and without the collective riding a side
     stream underneath it. If overlapped ~= max(compute, comms) the mechanism works at this shape; if
     overlapped ~= compute + comms the collective is not actually overlapping and no seam will help.

Every all_reduce arm is checked BIT-EXACT against RCCL before it is timed, so a fast wrong answer
cannot be reported as a fast right one.

Run under a 2-card lease inside the serve image:

    gpu-lease -n 2 -- docker run --rm ... -lc \
      'PYTHONPATH=/opt/kernels:/engine/python torchrun --nproc_per_node=2 \
       /engine/tools/tp_collective_regime_sweep.py'
"""

from __future__ import annotations

import os
import sys
import time

import torch
import torch.distributed as dist

HIDDEN = int(os.environ.get("SWEEP_HIDDEN", "2816"))   # DiffusionGemma / gemma-4-26B-A4B hidden_size
DTYPE = torch.bfloat16
ITEM = 2
ENGINE_SLOT_BYTES = 8 * 1024 * 1024                    # engine.py caps car_max_bytes at 8 MB
SLOT_BYTES = 48 << 20                                  # this bench's slot: big enough to probe PAST it
ROWS = (1, 8, 64, 256, 512, 1024, 1419, 1420, 2048, 3200, 4096, 8192)
ITERS = 60
WARMUP = 10


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
    cpu = dist.new_group(backend="gloo")

    import custom_ar as car

    has_vec = "vec" in str(torch.ops.custom_ar_C.one_shot_ar.default._schema)

    def p(*a):
        if rank == 0:
            print(*a, flush=True)

    p(f"# custom_ar schema has `vec` arg: {has_vec}  "
      f"({'VECTORIZED kernel' if has_vec else 'SCALAR kernel -- STALE, results will be pessimistic'})")
    p(f"# hidden={HIDDEN} dtype=bf16   engine slot cap = {ENGINE_SLOT_BYTES/1e6:.1f} MB "
      f"-> custom_ar gives up above {ENGINE_SLOT_BYTES//(HIDDEN*ITEM)} rows\n")

    slot = ((SLOT_BYTES + 255) // 256) * 256
    self_data = car.alloc_shared(2 * slot, 0).view(2, slot)
    self_flags = car.alloc_shared(64 * 4, 3)
    peer_base = _exchange(car, self_data, rank, cpu)
    peer_flags = _exchange(car, self_flags, rank, cpu)
    peer_data = [peer_base, peer_base + slot]
    dist.barrier(group=cpu)

    ctr = [0]

    def car_ar(y):
        """One custom_ar all-reduce, double-buffered exactly as CustomARDistributedImpl does."""
        s = ctr[0] & 1
        ctr[0] += 1
        buf = self_data[s].view(DTYPE)[: y.numel()]
        if has_vec:
            torch.ops.custom_ar_C.one_shot_ar(y, y, buf, peer_data[s], self_flags, peer_flags, 8, 1)
        else:
            torch.ops.custom_ar_C.one_shot_ar(y, y, buf, peer_data[s], self_flags, peer_flags)

    def car_ar_chunked(y, nchunk):
        """`nchunk` DISJOINT row chunks, each its own all_reduce. Bit-exact by construction: disjoint
        rows make each row's reduction an independent 2-rank elementwise SUM, so the split cannot
        change any row's value. Both ranks split at the same deterministic boundary and submit in the
        same order, so the collectives match pairwise."""
        n = y.shape[0]
        bounds = [(n * i) // nchunk for i in range(nchunk + 1)]
        for i in range(nchunk):
            lo, hi = bounds[i], bounds[i + 1]
            if hi > lo:
                car_ar(y[lo:hi])

    def timeit(fn, iters=ITERS):
        for _ in range(WARMUP):
            fn()
        torch.cuda.synchronize()
        dist.barrier(group=cpu)
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1e6

    def make(rows, seed):
        g = torch.Generator(device="cuda").manual_seed(seed)
        return torch.randn(rows, HIDDEN, generator=g, device=dev, dtype=DTYPE)

    # ============================================================ 1+2+3. the regime sweep ============
    p("=== all_reduce cost vs payload: where is the bandwidth regime, and where is the 8 MB cliff? ===")
    p(f"  {'rows':>6s} {'MB':>7s} {'fits':>5s} | {'car x1':>9s} {'car x2':>9s} {'car x4':>9s} "
      f"{'rccl':>9s} | {'best':>9s} {'GB/s':>7s} {'vs rccl':>8s}  {'engine today':>13s}")
    failures: list[str] = []
    rows_data = []
    for rows in ROWS:
        mb = rows * HIDDEN * ITEM / 1e6
        fits = rows * HIDDEN * ITEM <= ENGINE_SLOT_BYTES

        # --- bit-exactness gate, every arm, before any timing -------------------------------------
        x = make(rows, 31337 + rank)
        ref = x.clone()
        dist.all_reduce(ref, op=dist.ReduceOp.SUM)
        for name, fn in (("x1", lambda y: car_ar(y)),
                         ("x2", lambda y: car_ar_chunked(y, 2)),
                         ("x4", lambda y: car_ar_chunked(y, 4))):
            y = x.clone()
            fn(y)
            torch.cuda.synchronize()
            if not torch.equal(y, ref):
                failures.append(f"car {name} rows={rows} NOT bit-exact vs RCCL")

        y1, y2, y4, yr = x.clone(), x.clone(), x.clone(), x.clone()
        t1 = timeit(lambda: car_ar(y1))
        t2 = timeit(lambda: car_ar_chunked(y2, 2))
        t4 = timeit(lambda: car_ar_chunked(y4, 4))
        tr = timeit(lambda: dist.all_reduce(yr, op=dist.ReduceOp.SUM))

        best = min(t1, t2, t4, tr)
        # What the ENGINE does today: custom_ar if the whole tensor fits the 8 MB slot, else RCCL.
        today = t1 if fits else tr
        rows_data.append((rows, mb, fits, t1, t2, t4, tr, today, best))
        p(f"  {rows:6d} {mb:7.3f} {'yes' if fits else 'NO':>5s} | {t1:9.1f} {t2:9.1f} {t4:9.1f} "
          f"{tr:9.1f} | {best:9.1f} {mb*1e6/(best*1e-6)/1e9:7.2f} {tr/best:7.2f}x  "
          f"{today:11.1f}us")

    # ============================================================ 4. overlap headroom ================
    # Can a collective actually hide behind independent compute on this box? Bracket it: the SAME
    # collective and the SAME GEMM, issued serially vs with the collective on a side stream.
    p("\n=== overlap headroom: does a side-stream collective actually hide behind independent compute? ===")
    p("  (GEMM is independent of the all-reduced tensor -- this is the BEST case, an upper bound)")
    side = torch.cuda.Stream()
    main = torch.cuda.current_stream()
    p(f"  {'rows':>6s} {'MB':>7s} {'comms':>8s} {'compute':>8s} {'serial':>8s} {'overlap':>8s} "
      f"{'hidden':>8s} {'of ideal':>9s}")
    for rows in (512, 1024, 2048, 4096):
        x = make(rows, 909 + rank)
        # A GEMM sized so its cost is the same order as the collective: [rows, HIDDEN] x [HIDDEN, K].
        w = torch.randn(HIDDEN, 2048, device=dev, dtype=DTYPE)
        a = torch.randn(rows, HIDDEN, device=dev, dtype=DTYPE)
        y = x.clone()

        def comms_only():
            car_ar_chunked(y, 2) if rows * HIDDEN * ITEM > ENGINE_SLOT_BYTES else car_ar(y)

        def compute_only():
            torch.mm(a, w)

        def serial():
            comms_only()
            compute_only()

        def overlapped():
            ev = torch.cuda.Event()
            ev.record(main)
            side.wait_event(ev)
            with torch.cuda.stream(side):
                y.record_stream(side)
                comms_only()
                done = torch.cuda.Event()
                done.record(side)
            compute_only()
            main.wait_event(done)

        tc = timeit(comms_only)
        tk = timeit(compute_only)
        ts = timeit(serial)
        to = timeit(overlapped)
        ideal = max(tc, tk)
        mb = rows * HIDDEN * ITEM / 1e6
        p(f"  {rows:6d} {mb:7.3f} {tc:8.1f} {tk:8.1f} {ts:8.1f} {to:8.1f} {ts-to:8.1f} "
          f"{(ts-to)/max(ts-ideal,1e-9)*100:8.0f}%")

    dist.barrier(group=cpu)
    p("\n" + ("ALL BIT-EXACT" if not failures else "FAILURES: " + "; ".join(failures)))
    dist.destroy_process_group()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
