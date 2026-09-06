# Phase 0 Gate Report — Weight Offload

**Date:** 2026-09-02/03 · **Worktree:** `/home/pat/code/minisgl-rdna4-offload` (`feat/weight-offload`, off `8bcc7035`)
**Probes:** P0–P6, all run. **Reader:** the engineer who will build M1.
**Plan under test:** [`docs/WEIGHT_OFFLOAD_PLAN.md`](../../WEIGHT_OFFLOAD_PLAN.md) §3 exit criterion.

---

## 0. Verdict in one paragraph

**Phase 0 is a PARTIAL PASS.** The *bandwidth* is there (28.93 GB/s kernel-read on card 0, 2.2× the 13 GB/s kill floor) and the *torch plumbing* is there (P5 green in all four legs). But **the plan's core mechanism does not exist on this box**: `hipMemCreate(location.type=hipMemLocationTypeHost)` silently returns **device VRAM**, confirmed independently by P1, P2 and P3. There is therefore **no single device VA whose sub-ranges live on different media**, so §2's mixed device/host stack — the architectural idea the whole plan is built on — is **not constructible**. P2 could not run at all as a consequence, so **the miss-count curve shape, which §11 calls the second-highest-leverage unknown, is still unmeasured**. Separately, P0 found llama.cpp at **4.99 tok/s**, not the derived 33.8, which collapses the break-even gate to zero and makes T1-only a **~3.7× win** rather than a marginal one. **Start M1 (T1, all-host, pinned). Do not start M2 until P2′ runs.**

---

## 1. PASS / FAIL per probe

| Probe | Exit criterion (plan §3) | Result | Verdict |
|---|---|---|---|
| **P0** | informational — sets the bar | llama.cpp **4.990 tok/s** bs=1 decode (pooled 5.162); CONC=6 aggregate 6.551 tok/s wall | **PASS** (and it moved the bar 6.8×) |
| **P1** | ≥ 13 GB/s kernel-read **and** ≥ 0.5× copy engine | **28.93 GB/s** card 0 tiled, **14.48 GB/s** card 1; copy engine 28.70; both thresholds cleared on both cards | **PASS — QUALIFIED.** Mechanism substituted (`hipHostMalloc(Mapped)` + `hipHostGetDevicePointer`, not `hipMemCreate(location=Host)`). Card 1 clears the absolute floor by only **1.11×**. |
| **P2** | correct, with sublinear-or-linear miss cost | **NOT MEASURED.** Two attempts, two precondition aborts. Media separation ratio came back **1.00** (device 117.9 vs "host" 118.2 GB/s) — the host pages are VRAM. `cliff_index` never computed. | **INDETERMINATE / FAIL-TO-RUN.** The criterion is *unmet*, not cleared. `kill_triggered=false` is a statement about an unevaluated criterion. |
| **P3** | answers the RAM fork | Answered **decisively, by refuting its own premise**: host-located VMM does not exist (+109.7 % into VRAM, −5.2 % into MemAvailable, first touch 303 GB/s ≫ 28.7 GB/s PCIe ceiling). Fork resolved → the non-VMM branch. Supplementary P3b: pinned host **34.0 GiB × 1 rank OK**, **62 of 68 GiB × 2 ranks** (MemAvailable floor). | **PASS as a question-answering probe.** The fork is resolved. The *capacity* answer is a **near-miss, not a fit** — see §3.4. |
| **P4** | fixes the denominator | Ran clean, attempt 1. Concurrency is **free** (`efficiency = 0.999`). But card 1's root port `0000:00:01.3` is trained **Gen4 x8** while card 0's is **Gen5 x8** — 78/78 mid-DMA samples. 28.70 / 14.34 GB/s idle; **12.38 / 12.36 GB/s under host DDR load**. | **PASS** (and it changed every TP=2 number) |
| **P5** | green | Green in **all four legs**. `data_ptr() == reserved_base`, 0 `hipMalloc` fallbacks, arena survives `empty_cache` 12/12, free callback fires 0 times, **24/24 exact graph replays**. VMM granularity **4096 B** confirmed. | **PASS — with one gap**, see §3.5. |
| **P6** | expected to fail | `status: BROKEN_STALE`, `dynamic_residency_available: false`. **10/40 correct = handles⁻¹** — totally broken. All calls returned `hipSuccess`. Writes land on the **wrong physical page** (`distinct_landing_sets = [[0]]`), not dropped. Survives a 256 MiB cache flush → page-table, not cache. | **PASS as a tripwire** (expected failure confirmed, harder than the prior observation) |

### Overall

> **PARTIAL PASS.** Enough to build **M1 (T1, 100 % host)**. **Not** enough to build **M2 (T2)** — the specified T2 mechanism is unbuildable *and* the measurement that justifies any device tier was never taken.

The plan's literal exit criterion — "P1 ≥ 13 GB/s, P2 correct with sublinear-or-linear miss cost, P3 answers the RAM fork, P5 green" — is **3 of 4**. P2 is the missing one, and §3 says "otherwise go to §8 fallback". That instruction is now stale: §8 is M3/dynamic residency, which P6 has independently killed. The correct action is neither "proceed as written" nor "§8" — it is **proceed to M1 and re-scope M2**, per §5 below.

---

## 2. Which design branch is selected

**The mixed device/host single stack is dead. Take the separate-host-stack branch — but via *pinned anonymous* host memory, not the file-backed mmap the plan named.**

Three independent probes agree on the blocker:

| Signal | P1 | P2 | P3 |
|---|---|---|---|
| VRAM absorbed by a "host" allocation | 1.0× region | 1.75–1.78× region | **1.097×** region |
| Host `MemAvailable` moved | no | 0.04× | **−5.2 %** |
| GTT moved | no | 0.0× | **+0.1 %** |
| Kernel read from "host" pages | **690 GB/s** (= HBM) | **118.2 GB/s** = device arm's 117.9 (ratio 1.00) | **303 GB/s** (first touch, vs a 28.7 GB/s bus) |
| VA CPU-accessible? | **no** (`---s` on renderD128, `os.write` → EFAULT) | — | — |
| Return codes | all `hipSuccess` | all `hipSuccess` | all `hipSuccess` |

`hipMemGetAllocationPropertiesFromHandle` echoes `location.type=Host` back verbatim. The field is **accepted, echoed, and ignored**. `HostNuma(3)` / `HostNumaCurrent(4)` are rejected outright with `hipErrorInvalidValue`. A `location=Device(1)` control is **indistinguishable** (1.097 into VRAM, 303.6 GB/s).

**Plan §0 fact #2 is REFUTED.** "`hipMemCreate` with `location.type = hipMemLocationTypeHost` … a kernel reads and writes those host-backed pages correctly" — those pages were never host-backed. `vmm_probe2.py` passed because correctness and return codes are both clean; only *placement* was wrong, and placement was never checked.

### Full mechanism inventory (measured, this box, ROCm 7.2.4 / gfx1201)

| Mechanism | Real host pages? | Device-VA placeable? | Kernel-read BW (card 0) | Verdict |
|---|---|---|---|---|
| VMM `location=Device` | no (device) | yes | 692 GB/s | the device tier, as ordinary VRAM |
| VMM `location=Host` | **NO — silently VRAM** | yes | 690 GB/s | **does not exist** (D2) |
| VMM `location=HostNuma` / `HostNumaCurrent` | — | — | — | `hipErrorInvalidValue` |
| `hipHostMalloc(Mapped)` + `hipHostGetDevicePointer` | **yes** | **no** (separate VA) | **28.93 GB/s** | ✅ **SELECTED for T1** |
| `mmap(MAP_SHARED)` + `hipHostRegister` | **yes** (page cache) | no | 28.18 GB/s | fallback for capacity overflow; **uncharacterised > 4 GiB/rank** |
| `hipMallocManaged` + advise/prefetch | yes, **all-host and immovable** | n/a | 28.8 GB/s | advise/prefetch all no-ops (D3) |
| `hipMemUnmap` → `hipMemMap` at a used VA | — | — | — | **BROKEN** (P6) — T3 unreachable |

**Selected branch: pinned anonymous host arena (`hipHostMalloc(Mapped)`), one contiguous device pointer per component, no device tier inside it.**

Why pinned rather than the plan's named `mmap(MAP_SHARED)` + `hipHostRegister` fallback:
- pinned is demonstrated at **34.0 GiB/rank**; file-backed is demonstrated only at **4 GiB/rank**, and an 8 GiB / 2 GiB-chunk attempt wrote all 8 GiB then spun **> 711 s of user CPU** with no progress and had to be killed;
- setup is **4.88 GB/s vs 1.46 GB/s** (3.3× faster to boot);
- device-touch bandwidth is within noise (28.33 vs 28.18 GB/s).

File-backed stays on the shelf as the **capacity-overflow tier** (it costs **zero anonymous RAM** — P3b measured `MemAvailable` delta ≈ 0 and `Cached` delta = 1.000× committed) and needs its own probe before adoption.

### What this branch forfeits, and what replaces it

§9's constraint — *"mixed-media placement and NVMe overflow are mutually exclusive"* — is **moot**: mixed-media placement is impossible full stop, so the NVMe/page-cache path is unconstrained.

T2 as specified (top-`D` **expert rows** device-backed inside the same VA) would now require **two stacks + `slot_of`**, which reopens the entire `route_E`/`align_E` prerequisite in §4.4/§8 — the exact fatal class the plan exists to avoid. **Do not do that.**

The replacement is **layer-granular placement**, which §7 A2.3 already names as the fallback:
> a layer is all-device or all-host.

This needs **no VMM, no single-VA trick, no `slot_of`, no route change, and no kernel change**. A device-resident layer is an ordinary `hipMalloc` tensor stack; a host-resident layer is a pinned stack; the MoE kernel takes whichever base pointer that layer registered. It also makes the `h¹⁰` all-resident-layer question **irrelevant by construction** — which is fortunate, because P2 never answered it.

---

## 3. Corrected ceilings

### 3.1 Corrections to §1's physics table

| Quantity | Plan value | Measured | Δ |
|---|---|---|---|
| PCIe H2D @ 2.8 MiB granule | 26.8 GB/s | **card 0: 26.70 · card 1: 13.79** | card 1 is **half** |
| PCIe H2D @ 256 MiB | 28.7 GB/s | **card 0: 28.70 · card 1: 14.34** | card 1 is **half** |
| Kernel-read, tiled GEMM pattern, host pages | *unknown (127.2 withdrawn)* | **card 0: 28.93 · card 1: 14.48** | ≈ copy engine; **tiling costs nothing** (28.93 vs 28.94 linear); random-128 B costs 49 % |
| Host DDR5 read | ~45 GB/s (*brief, unmeasured*) | **36.2–54.3 GB/s bracket** (18.11 GB/s memcpy'd) | assumption confirmed, now bracketed |
| llama.cpp CPU-expert ceiling | **33.8 tok/s (derived)** | **4.990 tok/s** | **6.8× wrong — strike it** |
| Active expert bytes/token | 1.33 GB (48×10×2.8 MB) | GGUF packing measures **1.1625 GB** | different packing; see caveat below |
| VMM granularity | 4096 B | **4096 B** (P5 + P6, min == recommended) | confirmed |
| Zero-cache DMA-only ceiling | 20.2 tok/s | see §3.2 | restated |

### 3.2 The honest tok/s ceiling at TP=2

**Read this first — there are two wrong numbers in circulation.** P4's script emits `restated_zero_cache_dma_tok_s_tp2 = 20.18`; that field is efficiency-only. P4's *prose* then says ~10.3 tok/s, obtained as `20.2 × (14.34/28.1)`; that treats the plan's 20.2 as if each rank moved the *whole* 1.33 GB. **Neither is right.** At TP=2 each rank holds half of every expert, so each rank streams **0.665 GB/token**, and P4 proved the two links are **independent** (`efficiency = 0.999`), so the ranks stream concurrently and **the slower rank sets the step**.

Step time = `0.665 GB / BW_slow_rank + t_compute`. Using P1's tiled kernel-read figures and §1(c)'s 5–10 ms compute:

| Condition | Slow-rank BW | DMA-only | + 5 ms | + 10 ms | **quote this** |
|---|---|---|---|---|---|
| **TP=2, box idle** (card 1 gates) | 14.48 GB/s | 21.8 | 19.6 | 17.9 | **≈ 18.7 tok/s** |
| **TP=2, under host DDR load** (P4 proxy) | 12.38 GB/s | 18.6 | 17.0 | 15.7 | **≈ 16.3 tok/s** |
| *TP=2 if card 1's slot is fixed to Gen5* | 28.93 GB/s | 43.5 | 35.7 | 30.3 | *≈ 32.8 tok/s* |
| *TP=1, whole model on card 0 (does not fit)* | 28.93 GB/s | 21.8 | — | — | *reference only* |

**Sharp consequence:** because card 1 is *exactly* half of card 0, **TP=2 today buys nothing in DMA terms over a hypothetical TP=1 on card 0** (21.8 vs 21.8 tok/s). The second card is contributing capacity, not bandwidth. Fixing that slot is the single highest-leverage action in this whole report — see §6.

### 3.3 Against the real llama.cpp baseline

The brief quoted a prior of 4.58 tok/s. **P0 measured it fresh: 4.990 tok/s** bs=1 decode (canonical run 3; across-run medians 4.941 / 5.252 / 4.990, 6.2 % spread; pooled 15-rep median 5.162). Use **4.990**. The prior's companion figure of 11.1 tok/s prompt is **not reproduced** — P0 measures ~49.7 tok/s prefill — so do not cite it.

| | tok/s | vs llama.cpp 4.990 |
|---|---|---|
| llama.cpp, bs=1 | 4.990 | 1.00× |
| **K4 hard-kill threshold** (`1/1.57 ×`) | **3.178** | 0.64× |
| **T1 all-host, TP=2, idle** | **≈ 18.7** | **3.75×** |
| T1 all-host, TP=2, host-loaded | ≈ 16.3 | 3.27× |
| incumbent 35B M=1 (different, smaller model) | 49 | 9.8× |

**T1 clears K4 by 5.9×.** The charter's "all compute on GPU" constraint is *cheap* here, not expensive — the opposite of what §5.6's `alternatives_rejected` note anticipated.

**§1(a)'s break-even gate is STRUCK.** It required `h > 1 − 26.8/45 = 40 %`. That derivation compared GPU streaming against a **45 GB/s DDR abstraction llama.cpp never reaches** — P0 shows llama.cpp is **CPU-compute-bound** (i-quant expert GEMV on 8 threads), drawing only ~3.8 GB/s of host DDR (~8.5 % of the bus) and taking only **0.25–0.43 %** of its per-token expert bytes from storage. Recomputed against the measured baseline, the required byte hit rate is **h = 0 % at 0, 5 and 10 ms compute floors**. A **zero-cache, zero-device-tier, all-host** tier already beats llama.cpp by ~3.7×.

**Therefore A0.4 binds only on its 50 % arm, and K1 (`h < 40 %` → the device tier is worthless) can no longer fire on break-even grounds.** The device tier now has to justify itself on **speed and capacity**, not on beating a hybrid runtime.

### 3.4 Capacity is now the binding constraint, not bandwidth

P3b, on an **idle box with no engine loaded**:

| Arm | Target | Achieved | Stopped by |
|---|---|---|---|
| pinned, 1 rank | 34 GiB | **34.0 GiB** ✅ | target met |
| pinned, 2 ranks | 68 GiB | **62.0 GiB** (34.0 + 28.0) ❌ | `MemAvailable` floor; **114,813 pages swapped** |
| file-backed, 1 rank | 4 GiB | 4.0 GiB ✅ | target met (not probed higher) |

Box: `MemTotal` 91.84 GiB. The 68 GiB the plan wants is a **near-miss on an empty box** and will not fit once the engine, KV pool, activations and page cache are resident. **§11 unknown #4 is answered: NO.**

This makes the device tier a **capacity enabler, not just a perf lever** — every GB placed on device is a GB the host arena does not need:

| device tier `f` | dev GB/rank | host GB/rank | host GB total | step (ms) | tok/s | vs T1 |
|---|---|---|---|---|---|---|
| 0.00 (T1) | 0.0 | 34.4 | **68.8** ❌ *does not fit* | 53.4 | 18.7 | 1.00× |
| 0.10 | 3.4 | 31.0 | 61.9 ⚠️ *at P3b's idle ceiling* | 48.9 | 20.4 | 1.09× |
| 0.20 | 6.9 | 27.5 | 55.0 ✅ | 44.4 | 22.5 | 1.20× |
| 0.25 | 8.6 | 25.8 | 51.6 ✅ | 42.2 | 23.7 | 1.27× |
| 0.30 | 10.3 | 24.1 | 48.2 ✅ | 39.9 | 25.0 | 1.34× |

*(layer-granular, TP=2, card-1-gated 14.48 GB/s host + 692 GB/s device, 7.5 ms compute, 34.4 GB expert bytes/rank)*

Two things follow. **(1) Pure T1 at 68.8 GiB probably does not boot on this box** — M1 must either ship with a layer-granular device tier from day one, or use the file-backed tier for the tail, or run on a smaller checkpoint. **(2) The layer-granular tier is cheap and predictable**: ~1.2–1.3× at 7–9 GB/rank, cliff-free, `h¹⁰`-immune.

### 3.5 Confidence flags on the ceilings

- **28.93 / 14.48 GB/s are card-specific and idle.** Kernel-read under host DDR load was **not measured**; the 12.38 GB/s figure is a **copy-engine proxy** from P4, a different engine and a different path. Bracket accordingly.
- **1.33 GB/token is the plan's arithmetic, not a measurement of the minisgl packing.** P0 measured 1.1625 GB/token for the GGUF i-quant packing (exact GGUF tensor-table walk, self-validated to 99.99 % of on-disk bytes). If minisgl's w4a8 packing is closer to 1.16, every tok/s above rises ~14 %. **Measure `granule_bytes` for real in M1** — §4.4's walker already computes it.
- **Card 1's 14.48 GB/s is conditional on a BIOS/platform state**, not on the design.
- **P4's DDR figures are a bracket** derived from bytes copied by glibc `memmove` (2× if non-temporal, 3× if read-for-ownership), not an uncore counter.

---

## 4. Is T2 worth building?

**As specified: NO — it is unbuildable.** Per-expert placement inside one VA requires host-located VMM pages, which do not exist (§2).

**Reworked as layer-granular: YES, and it is now on the M1 critical path, not M2's — because it is a capacity prerequisite (§3.4), not just a speedup.**

**Is per-expert placement worth resurrecting as a two-stack design? Unknown, and the prior is bad.** P2 was supposed to answer this and did not. Here is what the surrounding measurements imply:

- Host kernel-read is **28.93 GB/s = 4.2 % of HBM (692 GB/s)** — a **23.9× ratio**.
- If a MoE layer's time is gated by its slowest workgroup (the `P = h¹⁰` effect §3 warns about), **one** host-resident expert out of top-10 costs nearly what **ten** cost. Then per-expert placement needs `P(all 10 resident) = h¹⁰` to be near 1, i.e. **h ≥ 0.9895** for 90 % of layers to be fully resident.
- A realistic tier — say 8 GB/rank device out of 34.4 GB of expert bytes = 23 % of bytes — cannot plausibly reach that. Even a generous LRU byte hit rate of 50 % gives `0.5¹⁰ = 0.1 %` of layers fully resident.
- **If the curve is linear**, the same tier saves ~23–50 % of bytes → up to 2×.
- **If the curve is cliffed**, per-expert placement buys **approximately zero**.

That is a 2× vs 0× spread on a single unmeasured question. **Do not spend a day of M2 on per-expert placement until P2′ measures the shape.** Layer-granular placement sidesteps the question entirely and delivers the (smaller, predictable) 1.2–1.3× shown above.

**§4.5's K2 is unaffected and still expected to fire** — P6 has independently killed T3/M3 regardless of what the oracle says, so `LRU − static_prior` now only chooses *which static prior*, not whether to build dynamic residency. **Do not build M3.**

---

## 5. What changes in the plan

Each edit below has been applied to `docs/WEIGHT_OFFLOAD_PLAN.md` in place and marked `> **[P0-2026-09-02]**`.

| § | Claim | Status | Correction |
|---|---|---|---|
| **§0 fact 2** | "`hipMemCreate` with `location.type = hipMemLocationTypeHost` … a kernel reads and writes those host-backed pages correctly (`vmm_probe2.py`)" | **REFUTED** | Those pages are **VRAM**. P1 + P2 + P3, three independent methods. `vmm_probe2.py` never checked placement. |
| **§1 table** | PCIe H2D 26.8 / 28.7 GB/s | **INCOMPLETE** | Per-card. Card 1 is exactly half (Gen4 x8 root port). |
| **§1 table** | Host DDR ~45 GB/s (brief) | **now measured** | 36.2–54.3 GB/s bracket. |
| **§1 table** | llama.cpp ceiling **33.8 tok/s** | **REFUTED** | **4.990 tok/s.** 6.8× wrong. Every gate derived from 33.8 is void. |
| **§1 table** | Zero-cache ceiling 20.2 tok/s | **restated** | TP=2: **18.7 idle / 16.3 loaded / 32.8 if card 1 is fixed**. |
| **§1(a)** | break-even `h > 40 %` | **STRUCK** | Measured break-even `h = 0 %`. llama.cpp is CPU-compute-bound, not DDR-bound. |
| **§1(b)(c)** | batch scaling; overlap is a 15 % lever | **unchanged** | Not contradicted. |
| **§2** | one VA, device- **and** host-located `hipMemCreate` handles | **NOT CONSTRUCTIBLE** | Replaced by: device stacks (`hipMalloc`) + host stacks (`hipHostMalloc(Mapped)`), **layer-granular**. |
| **§2** | "Device/host boundary may straddle a page **[ADJ]**" | **MOOT** | No mixed VA exists. |
| **§2 tier table** | T2 = per-expert top-`D` rows | **REPLACED** | T2 = **layer-granular**: a layer is all-device or all-host. |
| **§2 tier table** | T3 dynamic residency | **DEAD** | P6: remap serves the stale page, writes land on the wrong page, all `hipSuccess`. |
| **§3 P1 row** | — | **annotated** | PASS, mechanism substituted; card 1 marginal at 1.11× the floor. |
| **§3 P2 row** | — | **annotated** | Not measured. Criterion unevaluated. |
| **§3 P3 row** | "If 2 × 34 GB fits: proceed with host-located VMM pages" | **both halves wrong** | Host-located VMM does not exist, **and** 2×34 GiB does not fit (62 GiB on an idle box). |
| **§3 exit** | "Otherwise go to §8 fallback" | **stale** | §8 is M3, which P6 killed. Correct action: M1 + re-scoped M2. |
| **§5.1** | `hipvmm.py` binds `hipMemAddressReserve/Create/Map/SetAccess` | **rewritten** | Bind `hipHostMalloc` / `hipHostGetDevicePointer` / `hipHostFree` (+ `hipHostRegister` for the file-backed tier). VMM reserve/map is only needed if a future device-tier design wants address contiguity. |
| **§5.1** | 512 MiB backing chunks, ~136 handles for 68 GB ≈ 1.3 ms | **restated** | Pinning is the cost, not handle creation: **4.88 GB/s → ~7 s per 34 GiB rank**. Matches §0 fact 3. |
| **§5.3** | "Never allocate or map after `:245`" | **reinforced** | Still true, and now also because pinned arenas are invisible to `memory_reserved()`. |
| **§5.4 A1.4** | populate self-test / `_selftest_light` mandatory | **strongly reinforced** | P3 and P6 both show the driver returning `hipSuccess` while the placement or page table is wrong. **An out-of-band data check is the only defence.** |
| **§5.6 K4** | `< 0.64 × llama.cpp` | **instantiated** | **K4 = 3.178 tok/s** at bs=1. Projected T1 clears it by 5.9×. |
| **§5.6 A1.7** | `≥ 0.75 × 1/(t_compute + 1.33 GB/BW_P1)` | **instantiated** | **A1.7 = 14.0 tok/s** at bs=1, TP=2 (from 18.7 tok/s mechanism ceiling). |
| **§7** | M2 = static-prior **per-expert** device tier, conditional on A0.4 + P2 | **REPLACED** | M2 = **layer-granular** placement, conditional on **P2′**. Also promoted: part of it is a **capacity prerequisite** for M1 on this box. |
| **§8 M3** | "Do not start" | **upgraded to DEAD** | P6 is conclusive. Keep `tools/vmm_conformance.py` wired with `--expect broken` so a future AMD fix exits 3. |
| **§9** | "host-located `hipMemCreate` pages are anonymous, so a VMM-mapped mixed stack cannot be file-backed … mutually exclusive" | **MOOT, and inverted** | There is no VMM-mapped host stack at all, so nothing is mutually exclusive. And P3b's 62-GiB ceiling makes the file-backed/page-cache tier **more** relevant, sooner. |
| **§10** | critical path P1 → P3 → M1-A … | **re-ordered** | Insert **P2′** before M2. M1-A loses the VMM binding (−~1 d) and gains the layer-granular split (+~1 d). |
| **§11 #1** | kernel-read BW unknown | **ANSWERED** | 28.93 / 14.48 GB/s; tiled ≈ linear; random-128 B −49 %. |
| **§11 #2** | miss-count curve shape | **STILL OPEN** | Now the top unknown. |
| **§11 #3** | PCIe topology / TP=2 concurrency | **ANSWERED** | Independent x8 links, concurrency free; **card 1 trained Gen4**. |
| **§11 #4** | does 2 × 34 GB fit | **ANSWERED: NO** | 62 GiB on an *idle* box, with swapping. |
| **§11 #6** | CPU/GPU coherence of host pages | **ANSWERED (for pinned)** | Pinned zero-copy passed **every** coherence test both directions, on both cards, **with no explicit flush**, after fence, after `clflush`, at sub-64 B line granularity. |
| **§11 #7** | `expandable_segments` × capture × `MemPool` | **ANSWERED green — plus a landmine**, see §6 | |
| **§11 #8, #9** | ZFS tuning; `w8a8_moe_regdirect` rank bug | **untouched** | Still open. |

### New risks the plan does not carry

1. **[SERVE-WIDE, COLLATERAL] `expandable_segments:True` + `empty_cache()` returns stale/zeroed device memory.** In **plain torch** — no `MemPool`, no custom allocator — a 16 MiB float32 tensor whose `fill_` completed and was synchronised read back **all zeros in 7 of 12 reps**, at a VA torch had just unmapped and re-mapped (VA reuse 1.00). With `expandable_segments` off: **0 of 12**. This is the box's `hipMemUnmap`→`hipMemMap` defect (P6) surfacing inside torch's own allocator. **`engine/graph.py:314` calls `empty_cache()` and `docker-compose.yml:68` sets `expandable_segments:True` — both ingredients are in production today.** This is a demonstrated broken *primitive*, not a demonstrated wrong serve output. **It needs its own investigation and it is not a weight-offload problem.**
2. **[BOX-OPS] Half the machine's H2D bandwidth is unavailable.** Root port `0000:00:01.3` trains to **Gen4 x8** despite `max_link_speed = 32 GT/s`. Not ASPM (78/78 mid-DMA samples), not a measurement artifact (card 1 lands at exactly 91 % of the *Gen4* ceiling, and 91 % of Gen5 is impossible at 14.34 GB/s). Plausibly a BIOS PCIe-gen / bifurcation / riser signal-integrity issue.
3. **[DRIVER, D1] Per-chunk `hipMemSetAccess` fails non-deterministically** when two adjacent mappings in one reservation have **different sizes** — 33–34 % of chunks; a minimal (1.5 MiB, 7.5 MiB) pair fails 50–75 % of 100 fresh reservations. Control (equal sizes): **0/3072**. Workaround (one `SetAccess` over the whole hole-free span): **0/4096**. Only matters if VMM is used at all; noted so nobody rediscovers it.
4. **[PLANNING] `hipMemRelease` does not return the resource** — only `hipMemAddressFree` of the enclosing reservation does. And **over-commit is not reported**: `hipMemCreate`/`Map`/`SetAccess` all return `hipSuccess` past the point pages can be backed; the failure surfaces at **first touch as an unrecoverable SIGABRT GPU page fault**. Any code on this API must be fault-fenced in a disposable subprocess.

---

## 6. Still unknown, and what to run next

Ranked by leverage.

| # | Unknown | Probe | Cost | Why it matters |
|---|---|---|---|---|
| **1** | **Is per-layer time LINEAR or CLIFFED in miss count?** P2's actual question, still unanswered. | **P2′** — mixed-media grouped MoE GEMM built on **explicit two-stack pointers** (device `hipMalloc` stack + pinned host stack), not VMM. Correctness first, then per-layer wall time at 0/1/2/5/10 misses of top-10. **Constructible today** — P1 proved kernel reads from pinned host pointers are correct on both cards, both directions. | 0.5 d | **2× vs 0×** on whether per-expert placement is worth anything. Gates M2's shape. |
| **2** | **Can card 1's root port be trained to Gen5?** | BIOS sweep (PCIe gen / bifurcation / riser), then re-run P4's single-card arm. | 1 h | **The single highest-leverage action in this report.** Doubles rank 1 → TP=2 ceiling **18.7 → 32.8 tok/s (1.75×)**, and makes rule (b)'s fairness flag disappear. Costs a reboot. |
| **3** | **Does torch accept a `hipHostGetDevicePointer` address?** P5's four legs all ran over a **VMM reservation**, and its "host" legs were — we now know — device memory. So P5 validated foreign-pointer plumbing over *device* memory only. **The selected mechanism is untested through torch.** | **P5b** — re-run P5's six arms with the reservation replaced by `hipHostMalloc(Mapped)` + `hipHostGetDevicePointer`, in-image, with capture. | 0.25 d | Blocks M1-A. If it fails, the C++/`from_blob` route returns (+3 d). |
| **4** | **Kernel-read bandwidth from host pages under realistic host DDR load.** Only the copy engine was loaded (P4); the shader path was measured idle only (P1). | Merge P4's synthetic-load arm into P1's kernel-read arm. | 0.25 d | Every ceiling in §3.2's "loaded" row rests on a cross-engine proxy. |
| **5** | **Real `granule_bytes` for the minisgl packing.** 1.33 GB is arithmetic; 1.1625 GB is the *GGUF* packing. | §6.1's walker, run once after `post_load()`. | free (part of M1-A) | ±14 % on every tok/s number here. |
| **6** | **File-backed `mmap(MAP_SHARED)` + `hipHostRegister` at scale.** Demonstrated at 4 GiB/rank; **8 GiB/2 GiB-chunk spins > 711 s of user CPU**. | **P3c** — sweep chunk size and rank count to 34 GiB/rank; root-cause the spin. | 0.5 d | This is the *only* route past the 62 GiB pinned ceiling, and §3.4 says we need it (or a device tier). |
| **7** | **Does a live weight/state tensor land on a poisoned `expandable_segments` re-map in the served path?** | Serve-side repro on the current image. | 0.5 d | **Not an offload question.** Potentially a live correctness bug in production today. Own it separately. |
| **8** | **Expert reuse at E=512 / top-k=10 on real traffic** (§11 #5). | M0 router-only replay + E=256 cross-validation. | 4 d, parallel | Unchanged by Phase 0. Note A0.4's break-even arm is now vacuous (§3.3), so it binds only on its 50 % arm. |
| **9** | Card 1 was never exercised by P2, P5 or P6. | Fold a card-1 leg into P2′ and P5b. | free | This box has burned people on cross-card assumptions before. |
| **10** | ZFS `recordsize=1M` + `direct=always` (§11 #8, sudo-blocked); `w8a8_moe_regdirect` rank bug (§11 #9). | unchanged | 5 min | Untouched by Phase 0. |

### Recommended sequence

```
NOW      P2′ (0.5 d)  +  P5b (0.25 d)   ── both cheap, both gate M1/M2 shape
         BIOS: card 1 Gen5 attempt (1 h) ── independent, huge if it lands
THEN     M1-A  pinned host arena + layer-granular split (VMM binding dropped)
         M1-B..D  as written, with A1.7 = 14.0 tok/s and K4 = 3.178 tok/s
         P3c in parallel if the host arena does not fit
LATER    M2 = layer-granular tier, shape informed by P2′
NEVER    M3 / T3 — P6 is conclusive
SEPARATE expandable_segments × empty_cache corruption — not this feature's bug
```

---

## 7. Provenance

All probes ran in `/home/pat/code/minisgl-rdna4-offload` on branch `feat/weight-offload` off `8bcc7035`. The shared tree `/home/pat/code/minisgl-rdna4` was never touched. **The GPU lease was waived by explicit user instruction**; cards were verified idle before and after every run and no two GPU probes ran concurrently, but no arbiter enforced it. `ROCR_VISIBLE_DEVICES=0,1` fenced the Ryzen iGPU out of enumeration throughout. Box state (`free`, `rocm-smi`, `/proc/vmstat pswpout`, `MemAvailable`, physical card) is recorded per leg in every artifact.

Card identities, used consistently above: **card 0 = RX 9070 XT, `0000:03:00.0`, drm `card1`, root port `0000:00:01.1` (Gen5 x8)** · **card 1 = RX 9070, `0000:07:00.0`, drm `card2`, root port `0000:00:01.3` (Gen4 x8)**.

Stack: ROCm 7.2.4, `libamdhip64.so.7.2.53211-3d9ef42`, kernel `7.0.10-1-cachyos-custom`, torch `2.14.0.dev20260803+rocm7.2` (P5, in `minisgl-rdna4:lean`). **Every driver finding in this report is specific to that stack** and should be re-checked after any ROCm bump — that is what P6 exists for.

**Superseded / do-not-cite artifacts:** `p1.ACCIDENTAL-PRERUN-INVALID.{json,md}` (pre-fix build); `p5_run1_slot8MiB/` and `p5_run2_slot16MiB_devicecompare/` (unsound device-vs-device checker); `p0_run1_schema4/` (archived repeat, null I/O denominator). `p2.aborted.json` is attempt **2**; attempt 1 survives only as `p2_diagnostics/p2_attempt1_setaccess_abort.log`.

| Probe | Canonical artifact |
|---|---|
| P0 | `p0.json` · `P0_LLAMACPP_BASELINE.md` · `p0_cross_run.json` |
| P1 | `p1.json` · `p1.md` |
| P2 | `P2_MIXED_MEDIA_MOE.md` · `p2_driver_defects.json` · `p2.aborted.json` |
| P3 | `p3.json` · `P3_HOST_CAPACITY.md` · `p3b.json` · `P3B_HOST_FALLBACK.md` |
| P4 | `p4.json` · `p4.md` |
| P5 | `p5.json` · `P5_TORCH_FOREIGN_PTR.md` · `p5_diagnostics/` |
| P6 | `p6.json` · `p6.md` |
