# 48-layer TP=2 boot A/B — O_DIRECT into ONE REUSED buffer, 2026-09-06 (round 3)

Both legs: 48 layers, TP=2, `--device-gb 8.1 --host-gb 28.0 --cuda-graph-max-bs 2`, chunk 1372 MiB,
`gpu-lease -n 2`, ONE job on the box at a time, sequential and back to back, host sampled at 1 Hz
throughout, container named and reaped on timeout.

`MINISGL_WEIGHT_ARENA_FLOOR_GIB=4` in **both** legs, the only deviation from the shipped default
(12) and the same one round 2 used. It is a pre-flight GATE, not a measurement parameter:
`MemAvailable` does not count the ZFS ARC, which sits at 13-16 GiB on this box and IS reclaimable,
so the 48.23 GiB plan plus a 12 GiB floor needs 60.23 GiB against a ~59 GiB reading and aborts in
`reserve()` before pinning a page. A lower floor also arms the swap tripwire LATER, so it cannot
abort a leg the default would have completed.

## Result — 2.26x

| | before (mmap, `02957062`) | after (O_DIRECT, reused buffer, `25d61787`) |
|---|---|---|
| **boot** | **574.2 s** | **254.6 s** (−55.7 %) |
| `stage_b_run` | 376.6 / 371.5 s | **111.1 / 121.8 s** (−70.5 %) |
| `ckpt.h2d` | 194.0 s | **51.6 s** (−73.4 %) |
| `stageb.post_load` | 106.1 s | **7.0 s** (−93.4 %) |
| `arena_pin_attach` | 103.0 s | 80.2 s (−22.1 %) |
| `graph_capture` | 84.1 s | **33.6 s** (−60.0 %) |
| `ckpt.shard` | 44.3 s | 9.0 s (−79.8 %) |
| `ckpt.safe_open` | 0.7 s | 19.9 s (the read moved INTO this bucket) |
| `ckpt.nvfp4_prepass` | 0.3 s | 14.2 s (see "what got worse") |
| swap-out pages | 11,676,813 | **3,048,992** |
| major faults | 61,229,442 | **19,643,069** |
| `disk_read_bytes_total` | 73.65 GB | 81.68 GB (higher, and expected) |
| peak host RSS | 31.35 GB | 31.41 GB |
| load avg at launch | 1.17 | 2.13 |
| MemAvailable at launch | 59.63 GiB | 73.48 GiB |
| ARC at launch | 14.52 GiB | 1.54 GiB |

**Bit-identical.** Plan digest `a5e49a2e2b307ff3`, 156 forecast / 156 actual regions,
`arena_pinned_bytes` 25,895,632,896 ×2, `arena_carved_bytes` 25,769,803,776, `torch_fallbacks` 0,
`seam_pointer_checked` true, 1236 keys filled, 49 chunks, 36 host / 12 device layers, `kv_pages`
3053, the same `engaged` ledger on both ranks, and the same greedy token ids from both prompts on
both ranks (`" Paris. Paris is a city"` / `".\n\n<think>\nThe user"`). `failures: 0 -> 0`. The full
field-by-field comparison is `ab_report.txt`; every identity field matches.

## Why it wins — it is not the rate, it is the footprint

`zfs_readpath_r3.txt`, four distinct cold shards, one per leg, CPU only, quiet box:

    mmap, 4 KiB walk           627.0 MiB/s   dCached +0.33 GiB   dARC +0.33 GiB
    read() into reused buf    4947.8 MiB/s   dCached +0.00 GiB   dARC +0.31 GiB
    O_DIRECT into reused buf  4967.0 MiB/s   dCached +0.00 GiB   dARC +0.00 GiB
    posix_fadvise(DONTNEED)   returns 0 and frees NOTHING on this ZFS 2.4.3 mount

mmap costs TWICE the bytes — a page-cache page AND an ARC buffer per 4 KiB. Once the two 24.12 GiB
pinned arenas have taken half of RAM, every one of those allocations must reclaim, and on this box
reclaim of anonymous memory is a zram COMPRESSION. That is the whole 320 s.

The proof that it is a BOX cost and not a transfer cost is in the buckets that are not reads at all:
`stageb.post_load` (device-side NVFP4 conversion) fell 93.4 % and `graph_capture` fell 60 %. Neither
touches the checkpoint. They were slow because the box was stopping both processes, and they got
fast because it stopped doing that. The 1 Hz trace says the same thing directly: through the whole
of the after leg's Stage B the ARC and `Cached` sit FLAT at 8.2 / 11.4 GiB with MemAvailable
*rising*, where the before leg's ARC ran up to its 16 GiB cap with MemAvailable at 11 GiB.

`disk_read_bytes_total` going UP by 8 GB is the change working as designed: with mmap, rank 1's read
was served from the page cache rank 0 had just filled, so the ranks shared. O_DIRECT caches nothing,
so both ranks read every byte from the drive — 8 GB more physical I/O, and 320 s less boot.

## What got worse, and what is a confound

* `ckpt.nvfp4_prepass` 0.3 → 14.2 s and `stage_b_enumerate_prepass` 7.0 → 19.5 s. The prepass is a
  HEADER-ONLY mmap pass over all 196 shards and is deliberately left on `safetensors.safe_open`.
  It was cheap in the before leg because the ARC was warm; it is cold now, permanently, because
  nothing else warms the ARC any more. Real cost of the change, ~14 s, named rather than netted out.
* `ckpt.safe_open` 0.7 → 19.9 s is not a regression — it is the read, which used to be charged to
  `ckpt.h2d` and `ckpt.shard` as deferred page faults. 64.8 GiB per rank in 19.9 s = 3.3 GiB/s.
* **The confound:** the after leg started with 73.48 GiB MemAvailable against the before leg's
  59.63 GiB. That is not a quiet-box failure, it is where the same memory was sitting: the before
  leg's ARC held 14.52 GiB (warmed by the parity gate) and the after leg's held 1.54 GiB, so the
  reclaimable totals were 74.15 GiB and 75.02 GiB — matched to 0.9 GiB. The one bucket this can
  still flatter is `arena_pin_attach` (−22.8 s), which runs BEFORE any checkpoint byte is read and
  therefore cannot be credited to this change. Discount it: the fix is worth ~297 s of the 320 s.
* One leg each. The budget allowed two 48-layer TP=2 boots; the historical run-to-run swing on this
  box is ±40 % and this is −55.8 % with the mechanism visible in four independent counters
  (swap-out, major faults, ARC trace, and two non-read buckets), so it is not a coin flip — but it
  is one sample per leg and is reported as such.

## Provenance — old CODE, not emulated

Two separate git worktrees, so neither leg can read the other's source and neither was edited mid-run:

| leg | tree | sha | `ckpt_read.safe_open` in `models/weight.py` |
|---|---|---|---|
| before | `/home/pat/code/minisgl-rdna4-ramperf-base` | `02957062` | 0 occurrences (mmap) |
| after  | `/home/pat/code/minisgl-rdna4-ramperf`      | `25d61787` | 2 occurrences |

Kernels: `rdna4-hip-kernels-e4m3` `38ec1572`, the same prebuilt `fp8_wmma` torch-ext for both legs,
mounted at `/kbuild` ahead of `/opt/kernels`, with the in-container provenance assert on
`fp8_wmma.__file__`.

## Files

* `L48_r3_{before,after}.boot.rank{0,1}.json` — per-rank timelines (phases, buckets, per-chunk rows)
* `L48_r3_{before,after}.test.json` — harness verdict and every arena/seam invariant
* `L48_r3_{before,after}.run.log` — full engine log
* `L48_r3_{before,after}.box_sampler.jsonl` — 1 Hz /proc/vmstat + /proc/meminfo + ARC trace
* `L48_r3_{before,after}.box.txt` — box state either side of the leg
* `ab_report.txt` — `tools/offload/boot_ab_report.py`: timings, counters and the bit-identity block
* `zfs_readpath_r3.txt` — `tools/offload/zfs_readpath_probe.py`, the measurement the fix rests on
* `PROVENANCE.txt`
