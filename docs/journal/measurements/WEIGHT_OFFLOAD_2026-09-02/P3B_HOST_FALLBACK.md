# P3b — supplementary: what the *surviving* host-memory routes actually do

**This is not a P3 fork and must not be quoted as one.** P3 is invalidated (see
`P3_HOST_CAPACITY.md`): `hipMemCreate(location=Host)` returns device VRAM on this box, so
P3 has no host-capacity number to report. P3b measures the routes that do work, so the
design question P3's fork existed to answer is still answered.

**Date:** 2026-09-03 · **Box:** `blue`, ROCm 7.2.4 · **Cards:** rank 0 → `0000:03:00.0`
RX 9070 XT (card1), rank 1 → `0000:07:00.0` RX 9070 (card2).

---

## Results

Per-chunk allocation, a **device-issued** first touch through the device pointer from
`hipHostGetDevicePointer`, and a full resweep of every chunk against its own
`(device, index)` fingerprint at head / middle / tail.

| arm | ranks | chunk | committed | per rank | alloc GB/s (med) | device touch GB/s (med) | VRAM growth | resweep |
|---|---|---|---|---|---|---|---|---|
| `hipHostMalloc` | 1 | 2 GiB | **34.0 GiB** | 34.0 | 4.88 | **28.33** | 267 MiB | 0 / 17 |
| `hipHostMalloc` | 2 | 2 GiB | **62.0 GiB** | 34.0 + 28.0 | 4.69 | 21.28 | 291 MiB | 0 / 31 |
| `mmap(MAP_SHARED)` + `hipHostRegister` | 1 | 1 GiB | 4.0 GiB | 4.0 | 1.46 | **28.18** | 207 MiB | 0 / 4 |
| `mmap(MAP_SHARED)` + `hipHostRegister` | 2 | 1 GiB | 8.0 GiB | 4.0 + 4.0 | 1.43 | 16.02 | 215 MiB | 0 / 8 |

**Zero read-back failures anywhere.** VRAM growth is ambient (200–300 MiB regardless of a
4 GiB or 62 GiB arena — it does not scale with the arena, which is the point).

### These really are host memory — the same three discriminators P3's gate uses

- **Accounting:** `hipHostMalloc` drops `MemAvailable` by **0.998 of the arena** — eager,
  1:1, unreclaimable. The file-backed arm moves `Cached` by **1.000 / 0.991** of the arena
  instead, and barely moves `MemAvailable`, which is correct: page cache is reclaimable.
  *(Scoring the mmap arm on `MemAvailable` initially produced a false "not host memory";
  the meter has to match the page type.)*
- **Bandwidth:** device touch lands at **28.3 GB/s** on both routes single-rank —
  indistinguishable from the recorded 28.1 GB/s pinned-host H2D, and 10× below the
  ~303 GB/s that P3's gate saw from the VRAM impostor.
- **Device cost:** VRAM growth does not scale with the arena.

### Capacity — the question the fork existed to ask

- **One rank reaches 34 GiB cleanly.** Target met, not a floor stop.
- **Two ranks reached 62 GiB of the 68 GiB target** (34.0 + 28.0). Rank 0 hit its full
  34 GiB; rank 1 stopped on the **`mem_available_floor` guard at 12.5 GiB remaining** — a
  RAM-headroom stop on a box with 19.8 GiB already in zram swap, **not** an API limit.
  Baseline `MemAvailable` was 67.5–71 GiB against a 68 GiB target plus a 12 GiB floor, so
  62 GiB is about what the arithmetic predicts. Note the two-rank arm drove
  **114,813 pages out to swap**; the single-rank arm caused 460. That is the headroom
  boundary being felt, and is a reason to size the real arena below the maximum.

### Concurrency

Device-touch median falls **28.3 → 21.3 GB/s** (pinned) and **28.2 → 16.0 GB/s**
(file-backed) when both ranks stream at once. Suggestive of host-side contention rather
than two independent x8 links, but **P4 owns that question** — this is an incidental
observation from a capacity probe with no CPU-load control arm, not a concurrency result.

---

## Caveats — read these before using the numbers

1. **The mmap route is NOT characterised.** A first attempt at **8 GiB/rank with 2 GiB
   chunks** did not complete: after writing all 8 GiB (confirmed via
   `/proc/PID/io write_bytes`) the worker spun on **user** CPU for **>711 s** with the full
   8 GiB resident and no further progress, and was killed. At 1 GiB chunks and 4 GiB it
   completes in seconds. This is an unexplained pathology in
   `mmap(MAP_SHARED)`-on-ZFS + `hipHostRegister` and it is **not** diagnosed here.
   **The plan's stated fallback needs its own probe before it is adopted.** Its capacity
   ceiling at scale is unmeasured; only 4 GiB/rank is demonstrated.
2. Scratch files are on **ZFS** (`/home`, 387 GB free), deliberately not tmpfs — a
   tmpfs-backed "file" arm would secretly be a RAM arm.
3. **No engine was loaded.** All numbers are optimistic relative to a served box.
4. The two-rank `host_accounting_frac` of 0.669 is a snapshot artifact (the ranks peaked at
   different times and ARC shrank under pressure), not a sign the commit went elsewhere —
   the per-arm VRAM and bandwidth checks both confirm host memory.
5. `hipHostMalloc` alloc rate **4.69–4.88 GB/s** median reproduces the recorded 4.9–5.6
   GB/s. A 34 GiB arena therefore costs **~7 s** to allocate — the low end of the plan's
   7–20 s boot estimate. The file-backed route is ~3.3× slower to set up (1.4 GB/s).

---

## What this means for the design

The mixed device/host single stack via VMM is dead on this box (P3). But the underlying
capability the plan wanted — **34 GiB per rank of host memory the GPU reads at PCIe speed**
— is available and verified via `hipHostMalloc`, at full capacity for one rank and at
62/68 GiB for two, limited by this box's RAM rather than by any API.

That is a materially better position than `FILE_BACKED_MMAP_FALLBACK` implies, and it is
worth deciding explicitly whether the pinned-host route can carry T1/T2 in place of
host-located VMM pages before conceding the device tier and dragging plan §4.4's
`route_E`/`align_E` prerequisite back into scope. **That decision is not P3's to make** and
this probe does not make it — pinned host memory is not VMM-mappable into a contiguous
device VA, which is the property T1/T2 were designed around.

**Artifacts:** `p3b.json` (all four arms, box state per arm), `p3b.run.log`,
`p3b.mmap.run.log`, `tools/offload/p3b_host_fallback_capacity.py`.
