# P3 — Host-located VMM capacity

**Date:** 2026-09-03 · **Box:** `blue`, kernel 7.0.10-1-cachyos-custom, ROCm **7.2.4**
**Cards:** HIP dev 0 = `0000:03:00.0` RX 9070 XT (`/sys/class/drm/card1`, 15.92 GiB VRAM),
HIP dev 1 = `0000:07:00.0` RX 9070 (`/sys/class/drm/card2`). Every timing below is from
**HIP dev 0 / card1 / RX 9070 XT** unless stated.
**Lease:** waived for this development work by explicit user instruction. Both compute
cards confirmed idle (0% util, 57 MiB VRAM each) before starting; no other GPU probe ran
concurrently.

**Plan ref:** `docs/WEIGHT_OFFLOAD_PLAN.md` §3 (P3), §9, §11 item 4.

---

## Headline

> `hipMemCreate(location.type = hipMemLocationTypeHost)` **does not return host memory on
> this box. It returns device VRAM.** The location field is accepted, echoed back verbatim
> by `hipMemGetAllocationPropertiesFromHandle`, and ignored.

P3's question — "does host-located VMM deliver 34 GiB for one rank and 68 GiB for two?" —
therefore has no capacity answer, because the thing being sized is not host memory. The
probe's own hard invalidator `pages_are_not_host_located` fires, and per the probe contract
**no design fork may be read from the run**. `verdict.fork` is `null`.

That is not an inconclusive result. It is a stronger result than any of the four forks:
the mixed device/host single stack (T1/T2) rests on host-located VMM pages, and on this
box's ROCm those do not exist. No amount of free RAM, a quiet box, or a larger arena
changes it.

---

## The evidence — three independent discriminators

A return code proves nothing here: **every call in the sequence returns `hipSuccess`
either way.** One 2 GiB `hipMemCreate(location=Host)` on HIP dev 0, mapped to a device VA,
first-touched through that VA with `hipMemsetD32` (never a CPU store — plan §5.2):

| discriminator | measured | reading |
|---|---|---|
| discrete VRAM absorbed | **+109.7%** of the arena | **device memory** (+9.7% page-table/alignment overhead) |
| host `MemAvailable` absorbed | **−5.2%** of the arena | **not host RAM** — host memory did not drop at all |
| GTT absorbed | +0.1% of the arena | not GTT either |
| first touch via device VA | **303.1 GB/s** | PCIe H2D here is 26.8–28.7 GB/s. 303 GB/s is HBM. It never crossed the bus. |
| driver's `hipMemGetAllocationPropertiesFromHandle` | `location_type=2` (`Host`) | **the query passes while the operation does the opposite** |
| read-back after first touch | clean | correctness was never the problem |

Accounting and bandwidth **agree**, independently, that this is device memory.

### Corroboration from the capacity climb

Growing the arena 512 MiB at a time (`_p3_diag_alias.py`, scenario `ceiling`):

| after | MemAvailable | MemFree | `mem_info_vram_used` | `mem_info_gtt_used` |
|---|---|---|---|---|
| baseline | 67.629 GiB | 47.014 | 0.056 GiB | 0.022 |
| 1 chunk | 67.616 | 47.002 | **0.750** | 0.024 |
| 9 chunks | 67.616 | 47.002 | **4.750** | 0.024 |
| 17 chunks | 67.616 | 47.001 | **8.750** | 0.024 |
| 25 chunks | 67.616 | 47.001 | **12.750** | 0.024 |
| 31 chunks | 67.616 | 47.001 | **15.750** | 0.024 |

VRAM tracks the arena 1:1. Host memory never moves. `hipMemCreate` then returns
`hipErrorOutOfMemory` (rc=2) at **15.5 GiB — the card's VRAM, not the box's RAM.**

A control run against `location.type` = `Device(1)` is **indistinguishable**: 1.097 of the
arena into VRAM, 303.6 GB/s first touch. `HostNuma(3)` and `HostNumaCurrent(4)` are
rejected outright with `hipErrorInvalidValue`. There is no location value on this stack
that yields host memory.

Aliasing was ruled out separately: 4096 distinct words written one per MiB across a 4 GiB
range all read back correctly, 0 mismatches, 4096/4096 distinct values. The allocation is
real, single-mapped memory — it is simply in the wrong place.

---

## Two driver defects worth recording

**1. `hipMemRelease` does not return the resource; `hipMemAddressFree` does.**
A churn loop that created / mapped / touched / **unmapped / released** the same 512 MiB
handle repeatedly still walked VRAM monotonically to 15.75 GiB and then OOM'd at the same
31st iteration as a loop that held everything live. The resource only came back when the
enclosing `hipMemAddressReserve` range was freed.

**2. Over-commit is not reported — it is an unrecoverable GPU page fault.**
`hipMemCreate`, `hipMemMap` and `hipMemSetAccess` all return `hipSuccess` past the point
where the pages can be backed. The failure surfaces only on **first touch**, as:

```
Memory access fault by GPU node-1 (Agent handle: 0x2d40e6d0) on address 0x7fd700002000.
Reason: Page not present or supervisor privilege.
```

which is a `SIGABRT`, not an error return. **This is how the first attempt at P3 died.** A
capacity climb built on this API cannot fail gracefully, so any probe touching it must run
in a disposable subprocess. This is a fresh instance of the known trap: *a capability probe
can pass while the operation fails* — here the probe passed for three whole API calls.

---

## Kill criterion / fork

P3 does not kill the project; it forks the design. **This run emits no fork**, because a
hard invalidator fired (`pages_are_not_host_located`) and the probe contract forbids
reading a design decision off a run whose own data checks failed.

The practical consequence is nevertheless unambiguous, and it is the fork's
`FILE_BACKED_MMAP_FALLBACK` branch reached by a different road:

- **T1/T2's host-located VMM tier does not exist on this box.** Do not build on
  `hipMemCreate(location=Host)`.
- The **device tier is forfeited** for that route, so plan §4.4's `route_E`/`align_E`
  prerequisite (separate `route_num_experts` / `align_num_experts` args) **returns to
  scope**.
- The NVMe path (§9) opens.

Because the fork's stated fallback is now the *only* route, its viability was measured
separately — see **`P3B_HOST_FALLBACK.md`** / `p3b.json`. Summary: both surviving routes
work, reach capacity, and run at the expected PCIe speed.

**Not established by this run** (the climb never ran, by design): allocation rate for a
host-located arena, the 34 GiB-vs-34e9 B threshold comparison, and the two-rank
concurrency question for host-located pages. All three are moot for this API.

---

## Box state

Recorded at baseline, per the "a capacity probe on a busy box is a misleading probe" rule:

- MemTotal **91.84 GiB**, MemAvailable at baseline **67.52 GiB** (notably *higher* than the
  51.7 GiB the task brief anticipated), MemFree 47.0 GiB
- Swap: 53.9 GiB total (zram0 48 GiB @ prio 100 + an 8 GiB file), **19.8 GiB in use**;
  resting `pswpout` rate measured at **0.00 MB/s**
- ZFS ARC **8.01 GiB** (cap 16.00 GiB)
- Per-card VRAM used at baseline: 57 MiB on each discrete card
- **No engine was loaded** (`--no-require-engine`). The plan asks for P3 on a loaded box;
  this result is optimistic relative to the served configuration. It does not affect the
  finding, which is about the API, not the headroom.

The quiet-box projection in `p3.json` is retained but is now moot for this route: it sizes
a host arena that this API cannot produce.

---

## Artifacts

| file | what |
|---|---|
| `p3.json` / `p3.md` | the probe's own result, schema 1, `status=ok`, `verdict.fork=null`, `verdict.invalidated=true` |
| `p3.run.log` | full stdout of the successful (gated) run |
| `p3_diagnostics_bigchunk.json` | chunk-size sweep, 512 MiB → 8 GiB, each in its own process |
| `p3_diagnostics_va_reuse.json` | the original crash reproduced; commit attribution vs `hipHostMalloc` |
| `p3_diagnostics_alias.json` | aliasing ruled out; the 15.5 GiB VRAM ceiling; release/free behaviour |
| `p3_diagnostics_location.json` | all five `location` values, with a `Device` control |
| `P3B_HOST_FALLBACK.md` / `p3b.json` | supplementary: what the surviving host routes do |

### Changes made to the probe during this run

`tools/offload/p3_host_capacity.py` gained a **host-locatedness gate** that runs before any
climb, in a fault-fenced subprocess, and stops the run with the invalidator recorded if the
pages are not host memory. Without it the probe burns the card's VRAM and dies on a page
fault with no artifact — which is exactly what happened on attempt 1.
`read_gpu_vram_sysfs()` also now reads `mem_info_gtt_used`/`_total`; without GTT, "landed in
host RAM" and "landed nowhere" both read as "VRAM did not grow", which is the precise
ambiguity this probe has to resolve. `--skip-locatedness-gate` restores the old behaviour.
