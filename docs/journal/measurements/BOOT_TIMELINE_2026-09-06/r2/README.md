# 48-layer TP=2 boot A/B — O_DIRECT checkpoint read, 2026-09-06

Both legs: 48 layers, TP=2, `--device-gb 8.1 --host-gb 28.0 --cuda-graph-max-bs 2`, chunk 1372 MiB,
sequential, `gpu-lease -n 2`, one job on the box at a time, host box sampled at 1 Hz throughout.

`MINISGL_WEIGHT_ARENA_FLOOR_GIB=4` in **both** legs, and that is the only deviation from the shipped
default (12). It is a pre-flight GATE, not a measurement parameter: `MemAvailable` does not count the
ZFS ARC, which sits at 14-16 GiB on this box and IS reclaimable, so 48.23 GiB + a 12 GiB floor needs
60.23 GiB against a 56.6 GiB reading and aborts in `reserve()` before pinning a page. That abort
blocked round 1 entirely, twice. A lower floor also arms the swap tripwire LATER, so it cannot abort
a leg the default would have completed.

## Result — the fix LOSES, 3.5x

| | before (mmap, HEAD) | after (O_DIRECT) |
|---|---|---|
| boot | **520.5 s** | **1809.5 s** |
| stage_b | 334.1 s | 1498.4 s |
| `ckpt.h2d` | 174.7 s | **1382.1 s** |
| `ckpt.safe_open` + `get_tensor` + `shard` | ~297 s | **59.2 s** |
| `arena.host_alloc` | 104.2 s | 73.2 s |
| `graph_capture` | 53.6 s | 19.7 s |
| swap-out pages | 6,179,287 | **44,581,555** |
| major faults | 46,946,380 | **132,845,622** |
| peak host RSS | 31.28 GB | 31.33 GB |
| MemAvailable at launch | 59.34 GiB | 59.55 GiB |
| load avg at launch | 2.47 | 3.04 |

Bit-identical on both legs: plan digest `a5e49a2e2b307ff3`, 156 forecast / 156 actual regions,
`arena_pinned_bytes` 25,895,632,896 x 2, `arena_carved_bytes` 25,769,803,776, `torch_fallbacks` 0,
`seam_pointer_checked` true, 1236 keys filled, 36 host / 12 device layers, and the same six greedy
token ids from both prompts on both ranks (`" Paris. Paris is a city"`).

## Why it lost

The read got 5x FASTER and the boot got 3.5x slower. The regression is entirely `ckpt.h2d`, whose
source tensors are the reader's buffers.

`safetensors.safe_open`'s pages are FILE-BACKED: reclaiming one is free, the kernel drops a clean
page. The O_DIRECT reader's destination is an ANONYMOUS buffer, and reclaiming anonymous memory on
this box means COMPRESSING IT INTO ZRAM (swap here is a 45.9 GiB zram device, not a disk). Streaming
72.6 GiB of checkpoint through anonymous buffers, on a box whose two 24.12 GiB pinned arenas have
already taken half of RAM, turns every reclaimed checkpoint page from a free drop into a compression:
6.2M -> 44.6M swap-out pages, 46.9M -> 132.8M major faults.

**The page-cache/ARC pressure the change set out to remove was the CHEAP kind of pressure.**

## What the before-leg established, and keeps

The 1 Hz box sample (`tools/offload/box_sampler.py`) joined to the per-chunk rows on the new
`t_end_wall` (`tools/offload/boot_join.py`, output in `join_before.txt`) shows what `ckpt.h2d` is:
not a transfer cost. Every spike row sits where the ARC is at its 16 GiB cap with MemAvailable at
9 GiB and millions of direct-reclaim scans — layer-00002 costs 17.2 s with 4.63M direct scans,
layer-00010 costs 22.9 s with 3.54M major faults — while layers 31-45, ARC falling and no scans,
cost 0.45-0.6 s for the identical bytes. In the 2026-09-06 baseline the same 33 s stall landed in
`stageb.sink_place` on rank 0 and in `ckpt.h2d` on rank 1 at the same wall second. A stall that lands
in a different bucket on each rank is the box stopping both processes.

So `ckpt.h2d` remains the largest attributed cost and it is still a HOST MEMORY cost. What this A/B
rules out is the specific remedy: you cannot fix it by moving the checkpoint stream off the page
cache and into anonymous memory.

## Aborted first attempt — per-tensor O_DIRECT

`L48_r2_after_ABORTED_pertensor.*` is a leg that was reaped by hand rather than left to poison the
box. The first cut of the reader served shards over the whole-file cap with one aligned pread per
tensor. It is byte-identical and only 0.84x mmap on an idle box, but inside the boot it was still in
the BODY chunk after 13 minutes (it costs 15.6 s on the mmap path), reading 9.6 MiB per `preadv` at
53 MiB/s with both ranks at 100% of a core in userspace and the NVMe idle: every pread allocated a
fresh anonymous buffer, so `get_user_pages` faulted ~12 MiB of new anon INSIDE the read syscall.
Same disease as the leg that ran to completion, further along.

## Files

* `L48_r2_{before,after}.boot.rank{0,1}.json` — per-rank boot timelines (phases, buckets, per-chunk rows)
* `L48_r2_{before,after}.test.json` — harness verdict + every arena/seam invariant
* `L48_r2_{before,after}.run.log` — full engine log
* `L48_r2_{before,after}.box_sampler.jsonl.gz` — 1 Hz /proc/vmstat + /proc/meminfo + ARC trace
* `L48_r2_{before,after}.box.txt` — box state either side of the leg
* `join_{before,after}.txt` — the rows joined against the box trace
