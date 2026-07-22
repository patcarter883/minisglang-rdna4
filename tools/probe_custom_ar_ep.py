"""GO/NO-GO feasibility probe: cross-process one-shot P2P all-reduce (custom_ar) for the EP topology.

Faithfully mirrors the zaya DP+EP runtime topology:
  * TWO separate OS processes (torch.multiprocessing spawn), NOT 2 GPUs in one process.
  * Both processes inherit the full lease-visible device set (both cards visible), each set_device's
    its OWN card (rank 0 -> cuda:0, rank 1 -> cuda:1), exactly like the DP launcher.
  * IPC handle exchange over a gloo CPU group spanning the 2 processes (the analogue of dp_cpu_group).
  * A single one_shot_ar P2P all-reduce of a small bf16 tensor; verify the SUM is bit-correct on BOTH
    ranks vs a reference (rank0_input + rank1_input).

If open_ipc_handle / the flag handshake fails across the two discrete RDNA4 cards, this prints NO-GO
with the exact error and exits non-zero. Bounded: a watchdog aborts the handshake so a hang can't wedge
the box (the flag_probe has its own iteration cap; the all-reduce is guarded by a host-side timeout).

Run under a 2-card lease with a bounded timeout, e.g.:
  gpu-lease -n 2 --timeout 300 -- <docker run ...> python /engine/tools/probe_custom_ar_ep.py
"""
from __future__ import annotations

import os
import sys
import traceback

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _run(rank: int, world: int, addr: str, result_q) -> None:
    ok = False
    detail = ""
    try:
        torch.cuda.set_device(rank)
        dev = torch.cuda.current_device()
        peer = 1 - dev
        ndev = torch.cuda.device_count()
        p2p = (0 <= peer < ndev) and torch.cuda.can_device_access_peer(dev, peer)
        detail += f"[rank{rank}] pid={os.getpid()} cur_dev={dev} peer={peer} ndev={ndev} p2p={p2p}\n"
        if not p2p:
            raise RuntimeError(f"no P2P access dev{dev}<->dev{peer} (device_count={ndev})")

        dist.init_process_group(backend="gloo", rank=rank, world_size=world, init_method=addr)

        import custom_ar as car

        # --- flag-handshake diagnostic FIRST (isolates cross-GPU atomic visibility) ----------------
        # Each rank allocs a fine-grained IPC flag, exchanges handles, then flag_probe does a pure
        # system-scope store + spin-until-peer>=token with an internal iteration cap (returns -1 on
        # cap = P2P atomics NOT visible). This is the single cheapest GO/NO-GO signal.
        self_flag = car.alloc_shared(4, 3)  # int32[1] fine-grained
        h = car.get_ipc_handle(self_flag)
        gathered = [None] * world
        dist.all_gather_object(gathered, h.numpy().tobytes())
        peer_bytes = torch.frombuffer(bytearray(gathered[1 - rank]), dtype=torch.uint8).clone()
        peer_flag_ptr = car.open_ipc_handle(peer_bytes)
        detail += f"[rank{rank}] open_ipc_handle(flag) OK ptr={peer_flag_ptr:#x}\n"
        dist.barrier()

        res = torch.zeros(1, dtype=torch.int32, device=f"cuda:{dev}")
        car.flag_probe(self_flag, peer_flag_ptr, 1, res)
        torch.cuda.synchronize()
        spins = int(res.item())
        detail += f"[rank{rank}] flag_probe spins={spins} ({'saw peer' if spins >= 0 else 'CAP HIT -> P2P atomics DEAD'})\n"
        if spins < 0:
            raise RuntimeError("flag_probe hit iteration cap -> cross-GPU system-scope atomics NOT visible")
        dist.barrier()

        # --- full one_shot_ar all-reduce -----------------------------------------------------------
        N = 4096  # a decode-sized tensor (2*bs*H order of magnitude)
        slot_bytes = ((N * 2 + 255) // 256) * 256
        self_data = car.alloc_shared(2 * slot_bytes, 0).view(2, slot_bytes)  # uint8[2, slot_bytes]
        self_flags = car.alloc_shared(64 * 4, 3)  # int32[64]

        def _exchange(buf):
            hh = car.get_ipc_handle(buf)
            g = [None] * world
            dist.all_gather_object(g, hh.numpy().tobytes())
            pb = torch.frombuffer(bytearray(g[1 - rank]), dtype=torch.uint8).clone()
            return car.open_ipc_handle(pb)

        peer_data_base = _exchange(self_data)
        peer_flags_ptr = _exchange(self_flags)
        peer_data_ptr = [peer_data_base, peer_data_base + slot_bytes]
        dist.barrier()

        # deterministic per-rank inputs; reference sum is rank0_val + rank1_val elementwise
        val = float(rank + 1)  # rank0 -> 1.0, rank1 -> 2.0
        x = torch.full((N,), val, dtype=torch.bfloat16, device=f"cuda:{dev}")
        expected = torch.full((N,), 1.0 + 2.0, dtype=torch.bfloat16, device=f"cuda:{dev}")

        slot = 0
        sb = self_data[slot].view(torch.bfloat16)[:N]
        car.one_shot_ar(x, x, sb, peer_data_ptr[slot], self_flags, peer_flags_ptr)
        torch.cuda.synchronize()

        max_err = (x.float() - expected.float()).abs().max().item()
        match = torch.equal(x, expected)
        detail += f"[rank{rank}] one_shot_ar done: match={match} max_err={max_err} sample={x[:4].tolist()}\n"

        # second call (exercise the OTHER double-buffer slot + flag advance under back-to-back use)
        x2 = torch.full((N,), val * 10, dtype=torch.bfloat16, device=f"cuda:{dev}")
        expected2 = torch.full((N,), 10.0 + 20.0, dtype=torch.bfloat16, device=f"cuda:{dev}")
        slot2 = 1
        sb2 = self_data[slot2].view(torch.bfloat16)[:N]
        car.one_shot_ar(x2, x2, sb2, peer_data_ptr[slot2], self_flags, peer_flags_ptr)
        torch.cuda.synchronize()
        match2 = torch.equal(x2, expected2)
        detail += f"[rank{rank}] 2nd one_shot_ar (slot1) match={match2} sample={x2[:4].tolist()}\n"

        ok = bool(match and match2)
        dist.barrier()
    except Exception as e:  # noqa: BLE001
        detail += f"[rank{rank}] EXCEPTION {e!r}\n{traceback.format_exc()}\n"
    finally:
        try:
            result_q.put((rank, ok, detail))
        except Exception:
            pass
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass


def main() -> int:
    addr = "tcp://127.0.0.1:29591"
    world = 2
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_run, args=(r, world, addr, q)) for r in range(world)]
    for p in procs:
        p.start()
    results = {}
    for _ in range(world):
        try:
            rank, ok, detail = q.get(timeout=240)
            results[rank] = (ok, detail)
        except Exception as e:  # noqa: BLE001
            print(f"[main] TIMEOUT/err waiting for a rank result: {e!r} -> NO-GO (possible hang)")
            break
    for p in procs:
        p.join(timeout=20)
        if p.is_alive():
            print(f"[main] terminating still-alive pid={p.pid}")
            p.terminate()
    print("=" * 78)
    for r in sorted(results):
        print(results[r][1], end="")
    print("=" * 78)
    all_ok = len(results) == world and all(v[0] for v in results.values())
    print(f"VERDICT: {'GO ✅  cross-process P2P one-shot all-reduce WORKS on both cards' if all_ok else 'NO-GO ❌  see errors above'}")
    return 0 if all_ok else 2


if __name__ == "__main__":
    sys.exit(main())
