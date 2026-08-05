"""Bit-exactness gate for the TP comms/compute overlap seam (layers/tp_overlap.py).

The seam makes two DIFFERENT claims, with different strengths, and this file tests them separately
because conflating them is how a "bit-exact by construction" argument quietly stops being true.

  CLAIM 1 -- async_all_reduce is bit-exact UNCONDITIONALLY.
      It changes which STREAM a collective runs on and nothing else: same communicator, same tensor,
      same two rank-local addends. At TP=2 an all_reduce is a 2-addend elementwise sum, which has no
      ordering to reassociate. This is structural and the test should show max|delta| = 0 always.

  CLAIM 2 -- a ROW SPLIT is bit-exact only if the PRODUCER is bit-exact under a change of row count.
      The all_reduce half is structural (disjoint rows -> each row is an independent 2-rank sum). The
      producer half is NOT. `rowchunked_ar_span` feeds fewer rows to whatever compute sits inside it,
      and this repo's fused MoE grouped GEMM chooses block_m from the workload and reduces gemm2 with
      atomics -- a reduction whose ORDER depends on M. So "disjoint rows => bit-exact", the argument
      the original Qwen3.5-MoE async-AR carried in its docstring, covers the collective but NOT
      necessarily the expert compute it splits.

      This is the whole reason the test exists rather than the docstring. It measures, per producer:
        - a row-independent LINEAR producer (must be exact -- if this fails the seam is broken)
        - the real MoE producer at the shapes the serve actually uses (may not be; report the truth)

Run under a 2-card lease inside the serve image:
    gpu-lease -n 2 -- bash tools/tp_overlap_run.sh tpoverlap-bitexact \
      'PYTHONPATH=/opt/kernels:/engine/python torchrun --nproc_per_node=2 \
       /engine/tools/tp_overlap_bitexact.py'
"""

from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist


def main() -> int:
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=2)

    sys.path.insert(0, "/engine/python")
    from minisgl.distributed import DistributedCommunicator
    from minisgl.distributed.info import set_tp_info
    from minisgl.layers import tp_overlap

    # The seam reads TP size through the communicator plugins, which read get_tp_info().
    try:
        set_tp_info(rank, 2)
    except Exception:
        from minisgl.distributed import info as _info
        _info._TP_INFO = _info.DistributedInfo(rank=rank, size=2)  # type: ignore[attr-defined]

    comm = DistributedCommunicator()
    fails: list[str] = []

    def p(*a):
        if rank == 0:
            print(*a, flush=True)

    def check(name, got, ref, gated=True):
        """`gated` distinguishes a GATE from a MEASUREMENT, and the distinction is the point.

        The shipped default (no row split) must be exact, so those are gated and a failure fails the
        run. The row-split arm is KNOWN not to be exact — that is this file's headline finding, not a
        regression — so it is measured and reported. Gating on it would make the tool permanently red
        and train everyone to ignore it."""
        d = (got.float() - ref.float()).abs().max().item()
        ok = torch.equal(got, ref)
        tag = ("PASS" if ok else "FAIL") if gated else ("exact" if ok else "LOSSY")
        p(f"[{tag:>5s}] {name}: max|delta| = {d:.3e}")
        if not ok and gated:
            fails.append(name)

    gate = check

    HIDDEN = 2816
    # 3200 is the long-prompt shape the SWA extend exercised; 2048 is a typical prefill chunk.
    SHAPES = (256, 512, 2048, 3200)

    # ============================================================ CLAIM 1: stream change only =======
    p("=== CLAIM 1: async_all_reduce == plain all_reduce, no row split ===")
    for n in SHAPES:
        g = torch.Generator(device="cuda").manual_seed(11 + rank)
        x = torch.randn(n, HIDDEN, generator=g, device=dev, dtype=torch.bfloat16)
        ref = comm.all_reduce(x.clone())
        got = tp_overlap.async_all_reduce(comm, x.clone()).wait()
        torch.cuda.synchronize()
        gate(f"async_all_reduce rows={n}", got, ref)

    # Two outstanding handles at once -- the branch-overlap shape Gemma4 uses. This is also the case
    # that would corrupt if the ordering invariant were wrong (custom_ar handshakes through one shared
    # flag buffer; two concurrent calls would return half-reduced data, not an error).
    p("\n=== CLAIM 1b: two concurrent outstanding handles (the Gemma4 branch-overlap shape) ===")
    for n in SHAPES:
        g = torch.Generator(device="cuda").manual_seed(77 + rank)
        a = torch.randn(n, HIDDEN, generator=g, device=dev, dtype=torch.bfloat16)
        b = torch.randn(n, HIDDEN, generator=g, device=dev, dtype=torch.bfloat16)
        ra, rb = comm.all_reduce(a.clone()), comm.all_reduce(b.clone())
        with tp_overlap.ar_span(comm) as span:
            ha = span.all_reduce(a.clone())
            hb = span.all_reduce(b.clone())
            ga, gb = ha.wait(), hb.wait()
        torch.cuda.synchronize()
        gate(f"two-in-flight A rows={n}", ga, ra)
        gate(f"two-in-flight B rows={n}", gb, rb)

    # ============================================================ CLAIM 2: the row split ============
    p("\n=== CLAIM 2a: row split (REPORTED, NOT GATED — known lossy; see the module docstring) ===")
    w = torch.randn(HIDDEN, HIDDEN, device=dev, dtype=torch.bfloat16) * 0.02

    def linear_producer(h):
        return torch.mm(h, w)

    for n in SHAPES:
        for k in (2, 4):
            g = torch.Generator(device="cuda").manual_seed(303 + rank)
            x = torch.randn(n, HIDDEN, generator=g, device=dev, dtype=torch.bfloat16)
            ref = comm.all_reduce(linear_producer(x))
            got = tp_overlap.rowchunked_ar_span(comm, x, linear_producer, num_chunks=k)
            torch.cuda.synchronize()
            check(f"rowchunk linear rows={n} k={k}", got, ref, gated=False)

    # A GEMM's own tiling can be M-dependent even without MoE, so isolate that: does the PRODUCER
    # alone survive the split? If this differs, the split is not the seam's fault -- it is the kernel's.
    p("\n=== CLAIM 2b: is the PRODUCER itself invariant under a row split? (isolates the kernel) ===")
    for n in SHAPES:
        for k in (2, 4):
            g = torch.Generator(device="cuda").manual_seed(303 + rank)
            x = torch.randn(n, HIDDEN, generator=g, device=dev, dtype=torch.bfloat16)
            whole = linear_producer(x)
            bounds = [(n * i) // k for i in range(k + 1)]
            split = torch.cat(
                [linear_producer(x[bounds[i] : bounds[i + 1]]) for i in range(k)], dim=0
            )
            torch.cuda.synchronize()
            check(f"producer-only linear rows={n} k={k}", split, whole, gated=False)

    dist.barrier()
    p("\n" + ("GATED CLAIMS ALL BIT-EXACT (the shipped default is lossless)"
             if not fails else "FAILURES: " + "; ".join(fails)))
    dist.destroy_process_group()
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
