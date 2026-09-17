"""The TP one-shot P2P all_gather vs the RCCL gather it replaces — bit-exact, with provenance.

WHY THIS EXISTS. `144353c` added the custom one-shot P2P all_gather kernel (`all_gather_p2p`) and
wired it into ONE of its two siblings: `EPCommunicator.all_gather`, the EP dispatch gather. The TP
sibling, `CustomARDistributedImpl.all_gather`, kept the line "custom_ar has no all_gather; defer to
RCCL". Since custom_ar had already taken every all_reduce, that left `LMHead.forward`'s vocab-sharded
logits gather as the ONLY RCCL collective in a TP=2 serve — on every forward, decode included. This
test is the gate for wiring the second sibling.

WHAT IT PINS, and why each part is load-bearing:

  * BIT-EXACT vs `dist.all_gather_into_tensor`. The gather is pure data movement, so unlike a reduce
    there is no reordering excuse: anything but byte-identical is a bug. Callers index the result
    positionally (`LMHead` does `.view((tp,)+shape).permute(1,0,2)`), so the RANK-MAJOR LAYOUT is as
    much a part of the contract as the values — a correct-bytes/wrong-order gather would sample the
    wrong vocabulary shard and is checked by comparing the full tensor, not a norm.

  * PROVENANCE. `dist.all_gather_into_tensor` is monkey-patched with a counter. An eligible shape must
    NOT increment it (else the "parity" is RCCL-vs-RCCL and passes vacuously — the exact failure mode
    `ab-harness-must-assert-provenance` and `tests-must-measure-where-the-effect-is-visible` warn
    about); an ineligible shape MUST increment it.

  * THE FALLBACK BANDS, because widening this path made every EXISTING caller new. Three callers
    besides LMHead reach it: `logits_all_rows` (full-prefill logprobs), `engine.py:632` (a weight
    gather at load), `diffusion/sampler.py:118`. The first two are far larger than the slot and MUST
    self-fall-back to RCCL — on BOTH ranks together, since a one-sided fallback deadlocks on the peer
    flag. Non-contiguous and odd-byte-count inputs likewise.

  * THE REAL SHAPES. Decode bs=1 and spec-verify row counts at a true vocab shard width, not a toy
    tensor: the wire-type bitcast is chosen by BYTE COUNT, so element width and row count are exactly
    what could break it.

Needs TWO GPUs and the custom_ar package; skips cleanly otherwise rather than passing vacuously.

Run (BOTH cards must be visible to the container — this is a real TP=2 collective):
      <docker run ...> python /engine/tests/tp_custom_all_gather_parity_test.py
"""
from __future__ import annotations

import os
import sys
import traceback

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

VOCAB = 151936           # Qwen-class vocabulary; shard width is what LMHead actually gathers
TP = 2


def _run(rank: int, world: int, addr: str, q) -> None:
    fails: list[str] = []
    log: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        log.append(f"  [rank{rank}] {'OK  ' if ok else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
        if not ok:
            fails.append(f"rank{rank}: {name} {detail}")

    try:
        torch.cuda.set_device(rank)
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

        # Mirror the engine's real init: nccl default PG, gloo group for the IPC handshake.
        dist.init_process_group(backend="nccl", rank=rank, world_size=world, init_method=addr)
        cpu_group = dist.new_group(backend="gloo")

        from minisgl.distributed import DistributedCommunicator, set_tp_info, get_tp_info
        from minisgl.distributed import enable_custom_ar_distributed
        set_tp_info(rank, world)

        shard = -(-VOCAB // TP)
        # Cover 8 rows of a real vocab shard: decode (1) and spec-verify (5) both land inside.
        ag_bytes = 8 * shard * 2
        enable_custom_ar_distributed(get_tp_info(), cpu_group, 4 << 20, ag_max_bytes=ag_bytes)

        impl = DistributedCommunicator.plugins[-1]
        if type(impl).__name__ != "CustomARDistributedImpl":
            q.put((rank, True, [f"  [rank{rank}] SKIPPED: custom_ar did not install ({type(impl).__name__})"]))
            return
        if impl.ag_data is None:
            q.put((rank, True, [f"  [rank{rank}] SKIPPED: image custom_ar has no all_gather_p2p"]))
            return
        log.append(f"  [rank{rank}] gather slot = {impl.ag_slot_bytes} B ({impl.ag_slot_bytes/2//shard} vocab rows)")

        comm = DistributedCommunicator()

        # ---- provenance counter: every RCCL gather that still happens ----
        real_ag = dist.all_gather_into_tensor
        calls = {"n": 0}

        def counting_ag(out, inp, *a, **k):
            calls["n"] += 1
            return real_ag(out, inp, *a, **k)

        dist.all_gather_into_tensor = counting_ag

        def reference(x: torch.Tensor) -> torch.Tensor:
            shape = list(x.shape)
            shape[0] *= world
            out = torch.empty(shape, dtype=x.dtype, device=x.device)
            real_ag(out, x.contiguous())          # the UNcounted, unquestioned RCCL path
            return out

        def case(name: str, x: torch.Tensor, expect_custom: bool) -> None:
            ref = reference(x)
            before = calls["n"]
            got = comm.all_gather(x)
            used_rccl = calls["n"] > before
            check(f"{name}: took the {'custom' if expect_custom else 'RCCL'} path",
                  used_rccl != expect_custom,
                  f"rccl_calls={calls['n']-before} (expected {'0' if expect_custom else '>=1'})")
            check(f"{name}: shape matches RCCL", tuple(got.shape) == tuple(ref.shape),
                  f"{tuple(got.shape)} vs {tuple(ref.shape)}")
            if tuple(got.shape) == tuple(ref.shape):
                check(f"{name}: BIT-EXACT vs RCCL", torch.equal(got, ref),
                      "" if torch.equal(got, ref) else
                      f"max|d|={(got.float()-ref.float()).abs().max().item():.3e}, "
                      f"{(got != ref).sum().item()} differing elements")

        torch.manual_seed(1000 + rank)
        dev = torch.device("cuda")

        # --- the shapes the serve actually issues, through the custom path ---
        case("LMHead decode  [1, vocab/tp] bf16",
             (torch.randn(1, shard, dtype=torch.bfloat16, device=dev) * 4), True)
        case("LMHead verify  [5, vocab/tp] bf16",
             (torch.randn(5, shard, dtype=torch.bfloat16, device=dev) * 4), True)
        case("hidden-state   [128, 2560] bf16",
             torch.randn(128, 2560, dtype=torch.bfloat16, device=dev), True)
        case("f32 payload    [7, 1024] f32",
             torch.randn(7, 1024, dtype=torch.float32, device=dev), True)
        case("int32 payload  [3, 512] i32",
             torch.randint(-2**30, 2**30, (3, 512), dtype=torch.int32, device=dev), True)
        # Ragged row count: the wire bitcast is by byte count, so a row width that is not a clean
        # multiple of the wire element must still round-trip exactly.
        case("ragged width   [3, 1023] bf16",
             torch.randn(3, 1023, dtype=torch.bfloat16, device=dev), True)

        # --- the bands that MUST fall back, on both ranks together ---
        case("OVERSIZED (logits_all_rows / weight gather) [64, vocab/tp] bf16",
             torch.randn(64, shard, dtype=torch.bfloat16, device=dev), False)
        case("NON-CONTIGUOUS [16, 512] bf16 transposed",
             torch.randn(512, 16, dtype=torch.bfloat16, device=dev).t(), False)
        case("ODD BYTE COUNT [3, 5] int8",
             torch.randint(-128, 127, (3, 5), dtype=torch.int8, device=dev), False)

        # --- the layout contract LMHead depends on, exercised exactly as LMHead does it ---
        x = (torch.randn(4, shard, dtype=torch.bfloat16, device=dev) * 4)
        ref = reference(x)
        got = comm.all_gather(x)
        ins = tuple(x.shape)
        lay_ref = ref.view((world,) + ins).permute(1, 0, 2).contiguous() \
                     .reshape(ins[:1] + (world * ins[1],))[:, :VOCAB]
        lay_got = got.view((world,) + ins).permute(1, 0, 2).contiguous() \
                     .reshape(ins[:1] + (world * ins[1],))[:, :VOCAB]
        check("LMHead rank-major de-interleave is bit-identical", torch.equal(lay_got, lay_ref))

        # --- back-to-back gathers must not race on one slot (double-buffering) ---
        a = torch.randn(2, shard, dtype=torch.bfloat16, device=dev) * 4
        b = torch.randn(2, shard, dtype=torch.bfloat16, device=dev) * 4
        ra, rb = reference(a), reference(b)
        ga = comm.all_gather(a)
        gb = comm.all_gather(b)
        torch.cuda.synchronize()
        check("back-to-back gathers both correct (double-buffer)",
              torch.equal(ga, ra) and torch.equal(gb, rb))

        dist.all_gather_into_tensor = real_ag
        dist.barrier(group=cpu_group)
        q.put((rank, not fails, log + ([f"  [rank{rank}] FAILURES: {fails}"] if fails else [])))
    except Exception:
        q.put((rank, False, log + [f"  [rank{rank}] EXCEPTION:\n{traceback.format_exc()}"]))


def main() -> int:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        print(f"SKIPPED: needs 2 GPUs (saw {torch.cuda.device_count() if torch.cuda.is_available() else 0}).")
        return 0
    try:
        import custom_ar  # noqa: F401
    except Exception as e:
        print(f"SKIPPED: custom_ar unavailable ({e!r}).")
        return 0

    addr = "tcp://127.0.0.1:29591"
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_run, args=(r, TP, addr, q)) for r in range(TP)]
    for p in procs:
        p.start()
    results = []
    for _ in range(TP):
        try:
            results.append(q.get(timeout=300))
        except Exception:
            results.append((-1, False, ["  TIMEOUT waiting for a rank (deadlock in the gather?)"]))
    for p in procs:
        p.join(timeout=30)
        if p.is_alive():
            p.terminate()

    ok = True
    for _, rank_ok, log in sorted(results):
        print("\n".join(log))
        ok = ok and rank_ok
    print()
    if not ok:
        print("FAILED")
        return 1
    print("ALL CHECKS PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
