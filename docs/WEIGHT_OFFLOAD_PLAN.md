# Weight Offload for minisgl — Implementation Plan

**Status:** decision-ready. Supersedes the four subsystem designs where they conflict; adjudications are marked **[ADJ]**.
**Scope:** system-RAM (and later NVMe) resident model weights, all compute on GPU, general across models and quant formats, dense and MoE.

> **[P0-2026-09-02] PHASE 0 HAS RUN. READ [`docs/measurements/WEIGHT_OFFLOAD_2026-09-02/PHASE0_REPORT.md`](measurements/WEIGHT_OFFLOAD_2026-09-02/PHASE0_REPORT.md) BEFORE THIS DOCUMENT.**
> Phase 0 is a **PARTIAL PASS**. Two structural facts invalidate large parts of what follows, and every
> edit made in consequence is marked inline with a `> **[P0-2026-09-02]**` block:
> 1. **`hipMemCreate(location.type=hipMemLocationTypeHost)` silently returns device VRAM on this box**
>    (ROCm 7.2.4 / gfx1201), confirmed by three independent probes. **§2's mixed device/host single VA
>    is NOT CONSTRUCTIBLE.** The replacement is separate stacks with **layer-granular** placement.
> 2. **llama.cpp measures 4.990 tok/s, not the derived 33.8.** Every gate derived from 33.8 is void;
>    §1(a)'s 40 % break-even is **struck** (measured break-even `h = 0 %`).
>
> Still true and unchanged: the kernel-read bandwidth is there (28.93 GB/s card 0), torch takes a
> foreign device pointer under capture (P5 green), and dynamic residency is **dead** (P6 confirms the
> remap defect harder than before). Still **unmeasured**: P2's miss-count curve shape — the question
> that decides whether any per-expert device tier is worth building.

> **[M1A-2026-09-03] M1-A HAS LANDED (code only — no GPU work has run). READ
> [`docs/measurements/WEIGHT_OFFLOAD_2026-09-02/M1A_STATUS.md`](measurements/WEIGHT_OFFLOAD_2026-09-02/M1A_STATUS.md).**
> Both remaining gate probes came back, and both went the cheap way. Every consequent edit below is
> marked with a `> **[M1A-2026-09-03]**` block:
> 1. **P2′: the miss curve is LINEAR** (`cliff_index` 0.090 / 0.094 vs 0.100 pure-linear, on both
>    cards; marginal miss cost = exactly one granule at the card's measured host bandwidth, 102.8 % /
>    101.3 %). §11 unknown #2's "2× vs 0×" collapses to **≈ 0×**: per-expert placement buys
>    **1.013× / 1.009×** at the reachable operating point, 1.063× at an unreachable peak.
>    **Layer-granular is FINAL, not a fallback.**
> 2. **P5b: torch takes a `hipHostGetDevicePointer` address** under `MemPool` and under capture, on
>    both cards, 0 `hipMalloc` fallbacks. The C++ `from_blob` route (+3 d) is **not** incurred.
>    §11 unknown #10 is closed. ⚠️ `hipPointerGetAttributes` reports **"Device" for the real host
>    arena** — the inverse of the Phase-0 trap. That query is unreliable in **both** directions.
> 3. **NEW, and it costs 7.2 GiB/rank of VRAM:** the arena pins **whole 2 GiB chunks** over an
>    allocator that may never let a region straddle one, so reserving the host tier as *anonymous
>    headroom* over-reserves by **+28 %** (80.00 GiB pinned for a 62.44 GiB payload). That drives the
>    device tier the target model needs from **≈ 3.90 GiB/rank to 11.06 GiB/rank on a 16 GiB card** —
>    the difference between booting and not. **`PHASE0_REPORT.md` §3.4's "✅ fits" column is wrong for
>    this reason** (it charges the payload, not the pin). Fix = enumerate the rows as NAMED regions
>    from `sizing.meta_gemm_spec`, which already builds the real container on `meta` pre-load.
> 4. Byte model refined off the built model: **31.22 GiB/rank** of expert bytes (not 34.4 GB),
>    **granule = 1,327,104 B**, **1.27 GB/token** node-wide (−4 % on the plan's 1.33 GB arithmetic,
>    not the −14 % the GGUF figure suggested — §11 unknown #12). All-host T1 projects
>    **19.42 tok/s** (was 18.7), so **A1.7 re-instantiates at 14.57** and K4 (3.178) is cleared by
>    **6.1×** — on a projection, not a measurement.
> 5. **Still not discharged: graph capture.** No GPU work of any kind has run. Under the repo's
>    mandatory rule this feature is **unfinished**, and dense linears are still not planned (A1.5).

---

## 0. Verdict up front

Three of the four subsystem designs are built on machinery that the adversarial review dismantled: a per-step residency cache with a device-side `slot_of` gather, a prefetch predictor, an eviction policy, and a fetch schedule. Between them the reviews found **16 fatal defects**, and nearly all of them trace to one root cause: *the expert set for layer L is computed inside the captured graph, so any design that must decide-and-fetch per step has no legal place to stand.* Every fix bolted onto that shape (lagged predictors, two-generation eviction windows, miss sentinels, publish fences) adds a new silent-wrong-numbers path.

**This plan removes the mechanism instead of hardening it.**

The weight stack becomes a **single fixed device-addressable virtual address range whose backing is decided once, at boot, and never changes**. Hot expert rows sit on device-located physical pages; the rest sit on host-located physical pages that a kernel reads directly over PCIe. `w13.shape[0]` stays `E_global`, row `e` stays at `base + e·stride`, the fused `_route_align` keeps working, `moe_align` sees the same expert count it always did, and there is no `slot_of`, no eviction, no miss, no prefetch, no per-step host window, and no TP-divergence surface. Residency becomes a *placement* decision, not a *scheduling* one.

That is only possible because of three measured facts:

1. **HIP VMM remap is broken on this box** (unmap→map at a used VA silently serves the stale physical page, every call returning `hipSuccess`; reads *and* writes follow the stale page). So residency **cannot** be dynamic at page granularity anyway. **[ADJ]** — the fixed-VA-remap premise from the VMM design is dead; do not build it, do not gate on it.
2. ~~**VMM *mapping* works**: `hipMemAddressReserve` (256 GiB, free), `hipMemCreate` with `location.type = hipMemLocationTypeHost`, `hipMemMap` onto a device VA, `hipMemSetAccess` — all succeed, and a kernel reads and writes those host-backed pages correctly (`vmm_probe2.py`).~~

   > **[P0-2026-09-02] REFUTED.** Those pages are **VRAM**, not host memory. `location.type` is accepted,
   > echoed back verbatim by `hipMemGetAllocationPropertiesFromHandle`, and **ignored**. Three independent
   > methods agree: discrete VRAM absorbs +109.7 % of the arena (P3), host `MemAvailable` moves −5.2 %,
   > GTT +0.1 %, the VA is **not CPU-accessible** (`---s` on renderD128, `os.write` → EFAULT, P1), and a
   > kernel reads it at **690–303 GB/s = HBM speed** against a 28.7 GB/s PCIe ceiling. A `location=Device`
   > control is indistinguishable. `HostNuma`/`HostNumaCurrent` are rejected with `hipErrorInvalidValue`.
   > `vmm_probe2.py` passed because correctness and return codes are both clean — it never checked *placement*.
   > **What actually works:** `hipHostMalloc(Mapped)` + `hipHostGetDevicePointer` (real host pages,
   > **28.93 GB/s** kernel-read, fully coherent both directions with no explicit flush), and
   > `mmap(MAP_SHARED)` + `hipHostRegister` (28.18 GB/s, zero anonymous RAM). **Neither can be placed at a
   > chosen offset inside a reservation**, so there is no route to a mixed-media VA. See PHASE0_REPORT §2.

3. **The pin-rate scare was an artifact.** `torch.empty(pin_memory=True)` measured 200 MB/s on its *first* call (lazy HIP init + `CachingHostAllocator` bring-up). Raw `hipHostMalloc` in 4 GiB chunks is **4.9–5.6 GB/s**; `hipHostRegister` over a file-backed ZFS `MAP_SHARED` mmap is **1.8 GB/s** and delivers **identical H2D bandwidth**. A 34 GB per-rank arena is ~7–20 s of boot, not 5.6 minutes.

If the placement-only design measures acceptably, the project is roughly **one third** the code of any of the four proposals, with none of their fatal classes.

---

## 1. The corrected physics — every gate below is derived from these

> **[P0-2026-09-02] THIS TABLE IS SUPERSEDED BY THE MEASURED COLUMN.** The rows below are the original
> derivation, kept for provenance; the **Measured** column is authoritative. **The two cards are NOT
> symmetric**: card 1's root port `0000:00:01.3` is trained **Gen4 x8** while card 0's `0000:00:01.1` is
> **Gen5 x8** (78/78 samples taken *during* active DMA). Every single-card figure below is card 0.

| Quantity | Value | Source | Was claimed | **[P0] Measured** |
|---|---|---|---|---|
| PCIe H2D, 2.8 MiB granule | **26.8 GB/s** | measured | — | card 0 **26.70** · card 1 **13.79** (P4) |
| PCIe H2D, 1 GiB contiguous | 28.7 GB/s | measured | — | card 0 **28.70** · card 1 **14.34** @256 MiB (P4) |
| **Kernel-read, tiled grouped-GEMM pattern, host pages** | *unknown — 127.2 withdrawn* | — | — | card 0 **28.93** · card 1 **14.48** (P1). Tiling costs **nothing** vs linear (28.93 vs 28.94); random-128 B costs 49 %. |
| Host DDR5 read (aggregate) | ~45 GB/s | brief | — | **36.2–54.3 GB/s bracket** (18.11 GB/s memcpy'd, P4) |
| Active expert bytes/token (48L × 10 × 2.8 MB) | 1.33 GB | arithmetic | — | **still arithmetic.** The *GGUF* packing measures **1.1625 GB/token** (P0, exact tensor-table walk). Measure the minisgl packing in M1 — ±14 % on everything below. |
| **Zero-cache DMA-only ceiling** | **20.2 tok/s** | derived | 36 tok/s | **TP=2, slow rank gates: 21.8 idle / 18.6 host-loaded / 43.5 if card 1 is fixed** |
| Realistic zero-cache step (DMA + compute) | **14–17 tok/s** | derived | — | **18.7 idle / 16.3 loaded / 32.8 if card 1 is fixed** (TP=2, +5–10 ms compute) |
| llama.cpp CPU-expert ceiling (45 GB/s ÷ 1.33 GB) | ~~**33.8 tok/s**~~ | ~~derived~~ | — | **REFUTED — 4.990 tok/s** (P0, bs=1 decode; pooled 15-rep median 5.162). **6.8× wrong.** llama.cpp is **CPU-compute-bound**, not DDR-bound: it draws ~3.8 GB/s of host DDR (~8.5 % of the bus) and takes 0.25–0.43 % of its per-token expert bytes from storage. |
| Incumbent 35B M=1 served | 49 tok/s | recorded | — | unchanged |
| Pin rate, `hipHostMalloc` 4 GiB chunks | 4.9–5.6 GB/s | measured | 0.2 GB/s | **4.88 GB/s** at 2 GiB chunks (P3b) → ~7 s per 34 GiB rank |
| NVMe O_DIRECT @ granule, QD≥4 | 6.9–9.0 GB/s | measured | 28 GB/s striped | not re-measured |
| NVMe cold row latency (4 K–128 K, constant) | 157 µs p50 / 200 µs p99 | measured | — | not re-measured |
| VMM allocation granularity, gfx1201 | 4096 B | measured | — | **4096 B confirmed** (min == recommended, P5 + P6) |
| **Host pinned arena capacity** | *assumed 2 × 34 GiB* | — | — | **1 rank 34.0 GiB ✅ · 2 ranks 62.0 of 68 GiB ❌** on an *idle* box, with 114 k pages swapped (P3b). **§11 #4 answered: NO.** |
| **Concurrency scaling, two cards** | *unknown* | — | — | **0.999** — the x8 links are genuinely independent; host DDR does **not** bind at idle (P4) |

**Three consequences that reframe the project:**

**(a)** ~~**The break-even is against llama.cpp, not against zero.** GPU-side expert streaming beats computing the experts on the CPU in place only when `H/(1−h) > DDR`, i.e. **h > 1 − 26.8/45 = 40 %**. Below a 40 % byte hit rate this entire feature is strictly worse than a hybrid runtime. That number, not "does the cache hit", is the go/no-go.~~

> **[P0-2026-09-02] STRUCK.** The 40 % figure compared GPU streaming against a **45 GB/s DDR abstraction
> llama.cpp never reaches**. Measured, llama.cpp is CPU-compute-bound at **4.990 tok/s**. Recomputed
> against the real baseline, the required byte hit rate is **h = 0 % at 0, 5 and 10 ms compute floors**:
> a **zero-cache, zero-device-tier, all-host** tier already beats llama.cpp by **~3.7×**. Consequences:
> **A0.4 binds only on its 50 % arm**; **K1 can no longer fire on break-even grounds**; **K4 instantiates
> to 3.178 tok/s**, which the projected T1 clears by 5.9×. The device tier must now justify itself on
> **speed and capacity**, not on beating a hybrid runtime.

**(b) Aggregate throughput barely scales with batch.** Distinct experts per layer at batch M is `E(1−(1−k/E)^M)`: M=1→9.9, M=6→57, M=16→139, M=32→238. Bytes **per token** are 1.33 / 1.27 / 1.19 / 1.00 GB. Batching 1→6 buys 5 %. So this is a **capacity** feature — "serve a model that otherwise cannot be loaded" — not a throughput feature, and it must be sold and gated as one.

**(c) Overlap is a ~15 % lever, not the design's centre.** Expert-weight read from VRAM at 640 GB/s is ~0.04 ms/layer against ~0.92 ms/layer of DMA at bs=1. Total decode compute for the target shape is ~5–10 ms against ~44 ms of DMA. Maximum hideable fraction is 10–25 %. **[ADJ]** — the prefetch/overlap machinery in the scheduling subsystem (predictor, fork/join, two-generation window, ~1.2 ms/step of torch resolve) targets the smallest lever at the highest correctness cost. Cut it. The placement design gets whatever overlap the memory system gives for free, since host reads are issued by the GEMM itself and overlap with its own compute.

---

## 2. Architecture — one mechanism, three tiers of ambition

> **[P0-2026-09-02] THE MECHANISM IN THIS SECTION IS NOT CONSTRUCTIBLE ON THIS BOX.** There is no route
> to a single contiguous device VA whose sub-ranges live on different media (P1 + P2 + P3; full mechanism
> inventory in PHASE0_REPORT §2). **Replacement, which preserves every property this section was chosen for:**
>
> ```
> TWO stacks per (rank, component), each ordinary and contiguous:
>   device stack : hipMalloc                                    (692 GB/s)
>   host   stack : hipHostMalloc(Mapped)+hipHostGetDevicePointer (28.93 GB/s card 0 / 14.48 card 1)
> Placement is LAYER-GRANULAR: a layer is entirely device-resident or entirely host-resident,
> decided once at boot, frozen for the process lifetime.
> ```
>
> Row `e` of component `c` is still at `base(c, layer) + e·row_bytes(c)`; `E = w13.shape[0]` is still
> `E_global`; `_route_align` still fuses; the kernel takes whichever base pointer that layer registered.
> **No `slot_of`, no eviction, no miss path, no route-width change, no capture hook, no kernel change** —
> every "unrepresentable by construction" claim below survives intact.
>
> Layer-granular also **deletes the `h¹⁰` all-resident-layer risk entirely**, which is fortunate: P2 never
> measured the miss-count curve, so the per-expert variant rests on an unmeasured 2×-vs-0× question.
> **Do not build per-expert placement (which would need two stacks + `slot_of`, reopening §4.4/§8's
> `route_E`/`align_E` prerequisite) until P2′ measures the shape.**
>
> The original text follows for provenance.

```
ONE reserved VA range per (rank, component)          <- hipMemAddressReserve, once
  region(c) = [ rows 0..E-1 of component c, expert-major, natural stride ]
              backed page-by-page, at BOOT, by either
                 device-located hipMemCreate handles   (hot)
                 host-located   hipMemCreate handles   (cold, read over PCIe by the kernel)
              hipMemSetAccess once, RW, device 0 -- then FROZEN for the process lifetime
```

* Row `e` of component `c` is always at `region(c) + e·row_bytes(c)`. **The kernels are untouched.** `E = w13.shape[0]` is still `E_global`; `_route_align` (`quant/kernels.py:401-425`) still fuses; `moe_align`'s unguarded `atomicAdd` still sees ids in `[0, E)`; `_moe_block_m` still sees the tuned `E`.
* A "miss" is a correct read at PCIe speed. **There is no miss path, no mask-to-zero, no assertion, no fallback.** The entire silent-wrong-numbers class every reviewer flagged is *unrepresentable*.
* Under TP/EP each rank owns its own reservation and its own shard. Placement is a pure function of config + a baked prior file, identical procedure on both ranks, so `host_arena.py:15-22` invariant 1 is satisfied trivially — no eviction, no event query, nothing timing-dependent reaches a collective.
* Under graph capture: addresses are constants, access flags are frozen, no mapping call ever runs after boot, no side stream is forked, no host sync. Capture legality is *vacuous*, not argued.

**Component-major, not frame-major. [ADJ]** The MoE kernels do implicit-contiguous per-component pointer arithmetic (`w_base = w_rep + e·(N/16)·ktiles·32`, `ws_e = w_scales + e·G·N`, verified in `w4a8_fp8_wmma_kernel.hip:1676-1680`). `FrameLayout` is reused as the **granule descriptor** — the authoritative, enforced enumeration of what must travel together — but never as the physical packing. A frame-major slab would give `_w_rep` a row stride of `frame_bytes` and read the wrong bytes with no crash.

**Device/host boundary may straddle a page. [ADJ]** Back `floor(D·row_bytes(c)/4096)` pages of each component region from the device pool and the remainder from host; at most one row per component per layer is split across media. This deletes the `m = lcm(4096/gcd(4096,row_bytes))` alignment computation, the `Epad` padding waste, and the "refuse if m > 64" boot failure from the VMM design. A split row is a bandwidth rounding error, not a correctness event.

> **[P0-2026-09-02] MOOT.** No mixed-media VA exists, so no boundary straddles anything. Under
> layer-granular placement the two stacks are separately allocated and independently aligned.

**Three tiers, landed in order:**

| Tier | Placement | New machinery | Capture risk | **[P0] Status** |
|---|---|---|---|---|
| **T1** | 100 % host-backed | none beyond the VA + torch-tensor plumbing | none | **BUILD.** Mechanism = pinned host stack. Projected **18.7 tok/s** TP=2 idle (3.75× llama.cpp). ⚠️ **at 68.8 GiB it probably does not fit** — see the capacity row in §1. |
| **T2** | ~~static prior: top-`D` rows/layer device-backed, rest host~~ **layer-granular: top-`D` LAYERS device-resident** | a baked prior file, boot-time placement | none | **RESCOPED.** Per-expert is unbuildable *and* unjustified (P2 unmeasured). Layer-granular gives ~**1.2–1.3× at 7–9 GB/rank**, cliff-free — **and is a capacity prerequisite for M1 on this box** (it is what brings the host arena under P3b's 62 GiB ceiling). |
| **T3** *(conditional)* | dynamic residency | everything the four designs proposed | all of it | **DEAD.** P6: `hipMemUnmap`→`hipMemMap` at a used VA scores **10/40 = handles⁻¹**, every call returns `hipSuccess`, and **writes land on the wrong physical page** (`distinct_landing_sets=[[0]]`, `writes_lost_entirely=0`) — silent cross-page corruption. Survives a 256 MiB cache flush → page-table, not cache. |

T3 is only reachable if (i) the oracle shows dynamic beats the static prior by a wide margin **and** (ii) T2's measured throughput leaves room worth chasing. The review's **K2** finding stands: *if `LRU_hit − static_prior_hit < 5 points`, kill dynamic residency and ship the prior.* That check costs one simulator run and must happen before a single line of residency code.

---

## 3. Phase 0 — probes. ~3 days. Nothing else starts until these pass.

> **[P0-2026-09-02] ALL SEVEN RAN. RESULT: PARTIAL PASS (3 of 4 exit conditions).**
>
> | Probe | Verdict | Headline |
> |---|---|---|
> | **P0** | **PASS** | llama.cpp **4.990 tok/s** bs=1 (not 33.8). CONC=6 aggregate 6.551 tok/s. **K4 = 3.178 tok/s.** |
> | **P1** | **PASS — QUALIFIED** | **28.93 GB/s** card 0 tiled (2.2× the 13 GB/s floor) · **14.48** card 1 (**1.11×** — marginal). **Mechanism substituted**: `hipHostMalloc(Mapped)`, not `hipMemCreate(location=Host)`. |
> | **P2** | **INDETERMINATE — FAIL TO RUN** | Media separation ratio **1.00** (dev 117.9 vs "host" 118.2 GB/s). `cliff_index` never computed. **The criterion is unevaluated, not cleared.** |
> | **P3** | **PASS** (by refuting its own premise) | Host-located VMM **does not exist**. Fork → non-VMM branch. Capacity: **34 GiB × 1 ✅, 62 of 68 GiB × 2 ❌**. |
> | **P4** | **PASS** | Concurrency **free** (eff. 0.999). **Card 1's root port is Gen4 x8** → exactly half of card 0. Under host DDR load both converge to **12.4 GB/s**. |
> | **P5** | **PASS — one gap** | Green in all 4 legs; `data_ptr()==reserved_base`, 24/24 exact replays, free callback never fires. **Gap: all legs ran over a VMM reservation (= device memory), so the *pinned host pointer* is untested through torch → run P5b.** |
> | **P6** | **PASS as a tripwire** | `BROKEN_STALE`, expected failure confirmed harder than the prior observation. |
>
> **Two defect classes were found in the probes themselves and are worth internalising:** P1's
> `cpu_readable()` proved CPU-accessibility by `os.write()`-ing to `/dev/null`, which discards the payload
> *without copying from user space* and therefore reported **every** pointer as readable; and P3/P2 both
> show the driver returning `hipSuccess` while placement or the page table is wrong. **Assert on the
> actual operation and on out-of-band data, never on a return code or a query.** This is why §5.4's A1.4
> populate self-test is non-negotiable.

Each is standalone, ctypes or a small script, `gpu-lease -n 1` where noted, run in a dedicated `git worktree` (never `$PWD`). All outputs land in `docs/measurements/WEIGHT_OFFLOAD_<date>/`.

| # | Probe | Cost | Answers / kills |
|---|---|---|---|
| **P0** | **llama.cpp baseline.** Same target checkpoint, same box, tok/s at bs=1 and at CONC=6 aggregate. No lease needed if CPU-only. | 0.5 d | Sets the real bar. If llama.cpp ≥ 25 tok/s the required `h` is > 30 % *before* accounting for compute, and the project's value proposition is quality/context, not speed. **State it now or the whole plan is measured against zero.** |
| **P1** | **Host-backed pages behind a device VA — bandwidth and correctness, on a virgin VA.** ≥ 256 MB working set (4× MALL) so nothing is cache-resident; three access patterns: linear stream, the grouped-GEMM's tiled/strided pattern, random 128 B; sweep wave occupancy. Compare against `hipMemcpyAsync` from an ordinary pinned buffer on the identical buffer. **The 127.2 GB/s in `vmm_probe2.py` is a 2 MiB L2-resident artifact and is withdrawn.** | 0.5 d, lease | **KILL P1**: if kernel-read from host-located pages is < 0.5× the copy-engine figure (< ~13 GB/s), the placement design's ceiling drops below llama.cpp and T1/T2 are dead. Fallback: explicit-copy layer-group streaming (Phase 5 fallback below). |
| **P2** | **Mixed-media GEMM.** One `w4a8_moe` / `w8a8_moe_regdirect` launch over a stack whose rows straddle device- and host-backed pages. Correctness first; then per-layer wall time as a function of *miss count* (0, 1, 2, 5, 10 of top-10). | 0.5 d, lease | **KILL P2**: incorrect results, or a cliff worse than linear in miss count (the workgroup-completion effect: a layer runs at HBM speed only if all ~10 routed experts are resident, `P = h^10`). If the cliff is superlinear, T2's cache buys almost nothing and only T1 or explicit copy survive. |
| **P3** | **Host-located capacity.** `hipMemCreate(location=Host)` for 34 GB (one rank) and 68 GB (two ranks concurrently), timed, with the engine loaded and ARC warm, watching `MemAvailable` and `/proc/vmstat pswpout`. Stop at an 8 GiB floor. | 0.5 d, lease ×2 | **Forks the design.** If 2×34 GB fits: proceed with host-located VMM pages. If not: fall back to **file-backed** `mmap(MAP_SHARED)` + `hipHostRegister` + `hipHostGetDevicePointer` (measured at full H2D bandwidth), which costs zero anonymous RAM, lets page cache and NVMe absorb the tail — **but forfeits the mixed device/host single-stack**, so T2 becomes "two stacks + `slot_of`" and the whole route_E/align_E prerequisite (§4.4) comes back. |
| **P4** | **PCIe topology and concurrency.** `max_link_width`/`LnkSta` at the *root ports* (`0000:00:01.1`, `0000:00:01.3` — one reviewer reads x8, another x16 at the downstream endpoints). Then concurrent two-card pinned H2D, per-card and aggregate, with and without a synthetic CPU memory load. | 0.5 d, lease ×2 | Every ceiling in §1 is a single-card idle number. At TP=2 two ranks want 57 GB/s from a ~45 GB/s DDR bus. Fixes the denominator of every gate. |
| **P5** | **torch over a foreign device pointer, in-image.** `torch._C._cuda_customAllocator` + `torch.cuda.MemPool` + `use_mem_pool` yielding `t.data_ptr() == reserved_base` (verified on host); confirm in the serve image, **concurrently with `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` (the compose default, `docker-compose.yml:68`) and with graph capture active**. Also: does `torch.cuda.empty_cache()` invoke the custom free callback for a live user-MemPool block? (`graph.py:314` calls it after the slab would exist.) | 0.5 d, lease | **KILL P5** would force a C++/`from_blob` extension instead of ~30 lines of ctypes. Not fatal, but +3 days and a build-system change. Guard the known abort: a tensor outliving the pool raises `c10::Error: invalid device pointer` at `HIPCachingAllocator.cpp:4658` and **aborts the process** — hold pool and both `CFUNCTYPE` trampolines as module-level singletons, no-op free, one-shot alloc. |
| **P6** | **`hipMemUnmap`+`hipMemMap` conformance** — a 20-line torch-free port of `vmm_probe6.py` test D. Expected to FAIL. Kept as `tools/vmm_conformance.py`: an upstream ROCm/HIP repro and a regression tripwire, never a shipped code path. | 0.25 d | If AMD ever fixes it, dynamic page-level residency becomes available with no redesign. |

~~**Phase 0 exit criterion:** P1 ≥ 13 GB/s kernel-read, P2 correct with sublinear-or-linear miss cost, P3 answers the RAM fork, P5 green. Otherwise go to §8 fallback.~~

> **[P0-2026-09-02] RESULT: 3 of 4. P2 is the missing one.** The "otherwise go to §8" instruction is now
> **stale** — §8 is M3/dynamic residency, which P6 independently killed. The correct action is neither
> "proceed as written" nor "§8":
>
> **→ START M1 (T1, pinned host arena, layer-granular split). DO NOT START M2 UNTIL P2′ RUNS.**
>
> **Run next, in this order (PHASE0_REPORT §6):**
> 1. **P2′** (0.5 d) — the miss-count curve, rebuilt on **explicit two-stack pointers** instead of VMM.
>    Constructible today; P1 proved pinned-host kernel reads are correct on both cards, both directions.
>    Answers a **2×-vs-0×** question about any per-expert tier.
> 2. **BIOS: train card 1's root port to Gen5** (1 h) — **the highest-leverage action in this report.**
>    Doubles rank 1 → TP=2 ceiling **18.7 → 32.8 tok/s (1.75×)** and dissolves P4's fairness flag.
> 3. **P5b** (0.25 d) — repeat P5's six arms over `hipHostMalloc(Mapped)`+`hipHostGetDevicePointer`.
>    **Blocks M1-A**; P5 only validated foreign pointers over *device* memory.
> 4. **P3c** (0.5 d) — file-backed `mmap(MAP_SHARED)`+`hipHostRegister` at scale. Only needed if the
>    host arena does not fit; note the demonstrated **> 711 s CPU spin** at 8 GiB / 2 GiB chunks.
> 5. Kernel-read under synthetic host DDR load — the "loaded" ceilings currently rest on a copy-engine proxy.

---

## 4. Phase 1 — M0: the routing oracle. ~4 days. Runs in parallel with Phase 0.

The go/no-go instrument. Adopted essentially as the measurement subsystem specified it, with the review's corrections applied.

### 4.1 What is built

**New:** `python/minisgl/weights/route_oracle.py`, `tools/expert_route_replay.py`, `tools/expert_cache_sim.py`, `tests/core/test_route_oracle.py`, `tests/core/test_expert_cache_sim.py`.
**Modified:** `python/minisgl/layers/moe.py` (tap), `python/minisgl/scheduler/scheduler.py` (drain + idle-tick report), `python/minisgl/engine/engine.py` (materialize after `post_load()`), `docker-compose.yml`, `docs/COMPOSE_ENV_AUDIT.md`.

### 4.2 The tap — and the traps it must not fall into

* **Tap the route, not the weights.** EP branch: on `ep_i` **before** `_ep_dispatch` (`moe.py:1092`) so ids are *global* — `moe.py:997-998` / `:1038-1039` remap non-local experts to **local id 0** with weight 0, and a downstream tap attributes `(ep−1)/ep` of all traffic to expert 0. Non-EP branch: immediately before `method.apply` (`moe.py:1100`).
* **`topk_ids` is `None` on most served configs** (the kernels fuse softmax+top-k: `kernels.py:548/794/925`, `moe/fused.py:29-49`). **[ADJ]** — do not recompute with `torch.topk`: two reviewers showed the fused HIP top-k's tie-break and `__expf` reduction differ, so a recomputed route can record a *different expert* than the one executed, with the conservation check blind to it. Instead thread an optional preallocated `observe_ids` out-buffer through `method.apply` into `_route_align`, which already produces `topk_ids`. One `copy_` instead of a recomputed top-k, exact by construction, and it covers `rxf_moe`'s own torch route and the fused backend identically. Keep `torch.topk` only as a `src="topk-logits-APPROX"` fallback the simulator flags.
* **Allocation site.** The model is built on the **meta device** (`engine.py:218`). `MoELayer.__init__` may only *register metadata*. Device counters are allocated in `RouteOracle.materialize(device)` called from `Engine.__init__` after `post_load()` (`:241`) and before `_determine_num_pages` (`:245`), with an `assert not t.is_meta`.
* **Layer identity** is a structural dotted path from an `OPList`/`__dict__`/`nn.Module` walk, **not** a construction counter — the MTP draft head builds its own `MoELayer` (`moe.py:909-910`) and a counter renumbers silently.
* **Negative ids.** `moe/fused.py:46-48` writes a real `-1` sentinel. Mask unconditionally (`torch.where((f<0)|(f>=E), sentinel, f)`) before `index_add_` — a negative index there is an OOB device write.
* **Drain per FORWARD, not per scheduler step. [ADJ]** A spec step is K draft forwards + 1 verify forward, and `_spec_ep_loop` makes an idle EP replica issue dummy forwards. Hook `Scheduler._forward()`; emit one record per forward with `(forward_id, kind, layer_bitmap, M_real, accepted_tokens)`.
* **Conservation check, per layer. [ADJ]** The global `sum(delta) == M·top_k·L` form fires spuriously on GLM (`first_k_dense_replace`), ZAYA (MoE on odd layers, plus MOD `keep`-masked rows), Gemma4 (chunked `experts.forward`) and any spec step. Assert `delta[l].sum() == M_real(l)·top_k(l)` per registered layer. Zero the counters after `_capture_graphs` — the eager warmup at `graph.py:351` runs the tap on `dummy_req` batches.
* **WAR fence on `_counts`. [ADJ]** The drain as designed had no compute-stream fence, so step k+1's `index_add_` races the D2H and tears the frame. Snapshot into a double-buffered device staging tensor **on the compute stream**, then D2H that from the side stream (the `gdn_state.py:274-294` shape, both halves). Use a **dedicated third stream**, not `get_snapshot_side_stream()` — that stream carries the landed GDN snapshot tier and lock-stepping it to compute every step regresses prefix-hit latency.

### 4.3 The primary experiment

**Router-only replay is the primary artifact.** The box cannot load the 512-expert target, so an E=256 curve extrapolated to E=512 is an assumption. `tools/expert_route_replay.py` loads only the gate/router weights + embeddings (tiny `LinearReplicated` per layer) and runs a corpus of **real logged prompts** through the routers on CPU. True E=512 route sequence, zero VRAM.

**Cross-validation is mandatory.** The same replay methodology, run on Qwen3.6-35B (E=256) against that model's actual live token stream, must reproduce the live in-serve oracle's byte hit-rate curve within **5 points**. If it does not, neither number gates anything.

### 4.4 Simulator output

Granule = **co-demanded (w13 + w2, all components, one expert)**. `granule_bytes` computed after `post_load()` by the walker in §6.1 — deduped by storage pointer, excluding provably expert-invariant tensors (symmetric CT `_zeros_op` is E identical copies of `0x88`, ~3 % of w13), handling the bare-tensor unquantized container.

Report per (cache size ∈ {4, 8, 12, 16.6, 24 GB} × policy ∈ {LRU, LFU, static-prior, prior-pinned hybrid, **no-insert-on-prefill**, **segmented-LRU**, Belady}):
byte hit rate; **`P(all top-k resident) = ` the layer-level miss probability**, not just per-granule hit rate (P2 tells us which one predicts time); bytes per **accepted** token split by step kind; implied tok/s from the P1-measured mechanism bandwidth, not a hardcoded constant; and `1/max(t_compute_floor, bytes/BW)` so compute-bound points are flagged rather than reported as achievable.
Policy-independent: cold-set size, per-layer Gini / top-C mass, inter-reference recency distribution, **decode hit rate conditioned on a preceding prefill** (a prefill chunk touches ~all 512 experts per layer and evicts everything under LRU).

### 4.5 M0 acceptance / kill

* **A0.1** every registered layer emits a per-rank coverage line at plain `info` (not `info_rank0` — `_hip_engage.py:22-24` would hide rank 1), with equal layer bitmaps across ranks on an ep-over-tp config.
* **A0.2** per-layer conservation holds on every drain; a violation marks the trace `INVALID`.
* **A0.3** cross-validation within 5 points.
* **A0.4 GO:** byte hit rate at the affordable device tier ≥ **max(50 %, break-even vs P0's llama.cpp number)**.
* **K1 (redirect):** hit rate < 40 % → the device tier is worthless; ship **T1 only** (all-host), or nothing.
* **K2 (redirect, and the best outcome):** `LRU − static_prior < 5 points` → **never build T3**. Ship the static prior. This is the check that saves ~3 weeks and every fatal finding.
* **K0 (new, from the batch analysis):** if the amortization crossover batch exceeds the KV-feasible concurrency at the chosen device-tier size, aggregate throughput is flat and the feature must be sold purely on capacity. Report it; do not let it be discovered post-ship.

**Graph-capture status:** oracle collection runs **eager** (`MINISGL_EXPERT_ORACLE_MODE=trace`) — the route is a pure function of gate weights and tokens, so an eager run yields the identical sequence with no padding. The `hist` mode (graphs ON, `n_real` masking) is deferred to M4's production health signal. Say so in the report: *measurement ran eager; the feature must run captured.* Never quote tok/s from an oracle-on leg.

---

## 5. Phase 2 — M1: the host-resident weight tier. ~6 days. **This is the landable milestone.**

100 % host-backed. No cache, no residency, no policy. Correct, slow, and it serves a model that does not fit.

### 5.1 What is built

**New:**
* `python/minisgl/weights/hipvmm.py` — ctypes binding (`hipMemAddressReserve/Create/Map/SetAccess/Release/GetAllocationGranularity/hipHostRegister/hipHostGetDevicePointer`), styled on `utils/roctx.py:28-57` but **without** graceful degradation: a missing symbol raises. Module-level `freeze()` flag makes every mapping entry point raise after boot (**rule R1**).

  > **[P0-2026-09-02] RENAME AND RESCOPE → `hipmem.py`.** The VMM half is no longer the mechanism.
  > **Bind:** `hipHostMalloc` / `hipHostGetDevicePointer` / `hipHostFree` (the host stack), `hipMalloc`
  > (the device stack), and `hipHostRegister`/`hipHostUnregister` (the file-backed overflow tier).
  > **Do not bind** `hipMemAddressReserve/Create/Map/SetAccess` unless a later design needs address
  > contiguity — and if it does, know that (a) `location=Host` is silently ignored, (b) per-chunk
  > `hipMemSetAccess` fails **33–34 %** of the time when adjacent mappings differ in size (workaround: one
  > `SetAccess` over the whole hole-free span — 0/4096 failures), (c) `hipMemRelease` does **not** return
  > the resource, only `hipMemAddressFree` does, and (d) **over-commit is not reported** — every call
  > returns `hipSuccess` and the failure surfaces at first touch as an unrecoverable **SIGABRT GPU page
  > fault**, so any VMM sizing code must be fault-fenced in a disposable subprocess.
  > `freeze()` and rule R1 carry over unchanged and now cover the pinning entry points.
* `python/minisgl/weights/layout.py` — pure integer layout: per-component `row_bytes`, region offsets, the device/host page split per region, chunk interval lists. GPU-free unit tests.
* `python/minisgl/weights/arena.py` — reservation, 512 MiB backing chunks (`hipMemCreate` 9.5 µs, `hipMemRelease` 7.2 µs, so ~136 handles for 68 GB ≈ 1.3 ms), one-shot `hipMemSetAccess`, the one-shot pluggable-allocator `MemPool` giving torch a `cuda` tensor over the reservation, region/row views, `populate_from(model)`, `freeze()`, `stats()`.

  > **[P0-2026-09-02]** The handle-creation cost is not the cost — **pinning is**: 4.88 GB/s measured, so
  > **~7 s per 34 GiB rank** (consistent with §0 fact 3). Use **2 GiB** chunks (P3b's demonstrated size).
  > The `MemPool` plumbing is **validated** (P5: `data_ptr()==reserved_base`, 0 `hipMalloc` fallbacks,
  > arena survives `empty_cache` 12/12, the custom **FREE callback never fires** for a live *or* dropped
  > pool block, 24/24 numerically exact graph replays) — **but only over a device pointer.** P5b must
  > confirm the same over a `hipHostGetDevicePointer` address before this file is written.
  > Also from P5: keep the pool and both `CFUNCTYPE` trampolines as module-level singletons (already in
  > §3's P5 row), and note `empty_cache()` costs 41.7–45.9 µs with `expandable_segments` vs 142 µs without.

  > **[M1A-2026-09-03] LANDED as `pinned_arena.py` + `chunk_plan.py` + `torch_pool.py`, and this bullet
  > has a hole that costs 7.2 GiB/rank of VRAM.** It says nothing about **reservation granularity**.
  > The arena pins whole chunks, and `BumpAllocator` is forward-only next-fit that may never let a
  > region straddle a chunk — so it abandons a tail per chunk and the sound reservation is
  > `headroom_chunks(payload, chunk, max_row)`, **not** `ceil(payload / chunk)`. Measured on the
  > target shape (48 layers, E=512, top_k=10, CT-int4 g32, TP=2): payload 31.22 GiB/rank, largest row
  > 444.0 MiB, so **20 chunks = 40.00 GiB/rank pinned, not 16 = 32.00** (+28 %). Node-wide that is
  > **80.00 GiB pinned against a 55.80 GiB usable ceiling → INFEASIBLE**, and the required device tier
  > becomes **11.06 GiB/rank** where exact packing needs **≈ 3.90**. On a 16 GiB card with the tier
  > billed inside `model_memory` (§5.3, correctly), 11.06 leaves ~3.3 GiB for dense weights + KV +
  > state + draft + graph — very likely unbootable.
  > **THE FIX, and it is the first M1-B item:** stop reserving anonymous headroom. `sizing.meta_gemm_spec`
  > already builds the real container under `torch.device("meta")` *before* load, so every per-component
  > row size is knowable at `reserve()` time — emit them as named `RegionRequest`s. The reservation
  > becomes exact, `verify_matches_plan()` stops being vacuous, and `carve_digest()` gains real coverage.
  > Secondary lever: nothing above **2 GiB** has ever been pinned on this box, and a 4 GiB chunk would
  > save 0.65 GiB/rank of tier — worth a 0.5 d probe (**P3d**), not a default change.

  > **[M1B-2026-09-03] DONE, and the ≈ 3.90 GiB/rank estimate above was OPTIMISTIC — the computed
  > answer is 5.85.** `OffloadPlan.host_row_requests()` emits the per-component rows as
  > `RegionRequest(forecast=True)` and `StageARuntime.attach_host_arena` reserves those, so
  > `plan_regions` runs the real next-fit allocator over the real list. Target shape, computed by the
  > landed code: rows are **384/48/12 MiB (w13) + 192/24/6 MiB (w2) = 666 MiB/layer**; three whole
  > layers fit one 2 GiB chunk (1998 MiB, 50 MiB abandoned), so all-host is **16 chunks =
  > 32.00 GiB/rank · 64.00 GiB/node**, not 20 / 40.00 / 80.00. The required device tier falls from
  > **11.06 → 5.85 GiB/rank** (17 → 9 of 48 layers); at that tier the 39 host layers pin 13 chunks =
  > **52.00 GiB/node ≤ 55.80 → FEASIBLE**. The 3.90 figure assumed the *payload* (54.63 GiB/node) was
  > the charge; 42 host layers actually pin 14 chunks = 56.00 GiB/node, 0.20 GiB over the ceiling.
  > **5.2 GiB/rank of VRAM recovered, not 7.2.**
  > Three things the fix had to carry that this bullet did not anticipate: (a) torch's caching
  > allocator asks for a rounded SEGMENT, not for the tensor, so rows are reserved at
  > `chunk_plan.torch_allocation_bytes` (`round_up(max(n, 2 MiB), 2 MiB)`) — **modelled, never
  > measured**, and the first thing a GPU run must check; (b) the `MemPool` C ABI carries a size and
  > no identity, so a forecast region is reconciled by *envelope + attribution*, never by name —
  > `verify_matches_plan()` now refuses a VACUOUS pass and is called from `seal()`, its first
  > production caller; (c) the observed (meta-model) path was not applying `post_load_delta_bytes` at
  > all, which the old +28 % slop had been hiding — `placement.PostLoadCorrection` closes it.
  > P5b confirmed the `MemPool` plumbing over a **host** pointer, so this bullet's precondition is met:
  > `data_ptr() == hipHostGetDevicePointer(...)`, 0 fallbacks, free callback fires zero times, arena
  > survives `empty_cache` and `graph.__enter__`, 24/24 replays one sha256.
  > ⚠️ **Do not gate residency on `hipPointerGetAttributes`** — it reports "Device" for the real host
  > arena. `PinnedWeightArena.owns_pointer()` (arithmetic on the returned pointers) is the only
  > trustworthy test. ⚠️ **`amdgpu mem_info_gtt_used` stays at 0** for this mechanism, so host-tier
  > telemetry must use `MemAvailable` or its own accounting.
* `python/minisgl/weights/granule.py` — the walker + `GranuleSpec` (§6.1).
* `python/minisgl/weights/registry.py` — the structural op walk.
* `python/minisgl/weights/plan.py` — `resolve_weight_plan(config)`: the **single** resolver, next to `resolve_prefix_cache` (`engine/config.py:75-117`).
* `tools/vmm_conformance.py`, `tests/core/test_weight_layout.py`, `tests/core/test_granule_spec.py`, `tests/core/test_granule_desync.py`, `tests/core/test_new_format_fail_closed.py`, `tools/weight_offload_serve_ab.sh`, `tools/weight_offload_correctness.py`.

**Modified:** `engine/engine.py`, `engine/config.py`, `server/args.py`, `layers/moe.py` (one container normalization + one registration hook), `layers/linear.py` (registration hook), `tools/serve.sh`, `docker-compose.yml`, `docs/COMPOSE_ENV_AUDIT.md`, plus guards in `tools/qwen2_moe_weight_map_test.py:100`, `tools/glm4_moe_lite_build_smoke.py:78,130`, `tools/test_muse_glimmer.py:150`.

**Explicitly NOT built:** `slot_of`, `prepare_for_replay` hooks, `WeightStore.stage`, eviction, prefetch, the residency proxy, `route_only`, `moe_residency_*` kernels, `PinnedFrameArena` for weights.

### 5.2 Load-time: the bake, and why it cannot be post-hoc for the target model

**[ADJ]** — the layer-group design's "bake after `load_state_dict`" is unimplementable for the motivating checkpoint: `weight.py:906` etc. open shards with `device=str(device)` and `post_load` repacks the whole stack on GPU, so 68 GB OOMs long before the bake runs. **Two stages, and Stage A alone is not sufficient for the target model — say so.**

* **Stage A (M1):** resolve the plan from `ModelConfig` before the meta build (`engine.py:218`); reserve the VA and map all chunks *before* `load_state_dict` (`:239`) so a capacity failure aborts in seconds; run the normal load + `post_load()`; then copy each granule into its VA row **through the device pointer** (never CPU stores into host-located pages — the coherence granularity is unstated and a CPU-written page can be stale in L2; P1 must include a CPU-write/kernel-read arm), `torch.cuda.synchronize()`, drop device originals, `freeze()`. Peak VRAM = the un-offloaded model. Works for any model that fits at load.
* **Stage B (prerequisite for the target model, +3 days):** per-container chunked load. `BaseOP.post_load(granule_range=None)`; per-key device selection in the five loaders (`weight.py:906/1106/1245/1359/1454`) — `_load_qwen3_5_weight` already reads `device="cpu"` and `.to(device)` post-shard (`:546/:622`) and is the precedent; an expert-range argument on the four MoE repack ops **and** the dense `repack_w_rep_wide` in the same change.
* **`_ct_packed_is_uint4b8` (`quant/method.py:31-40`) samples only the first 65536 int32 of the whole stack.** Compute it **once**, over a fixed deterministic sample of the full stack, before any chunking; store it in the plan and pass it into every chunk. Re-deriving per chunk XOR-corrupts a contiguous block of experts — plausible text, no crash. Raise if `|counts[8]−counts[0]|/total < ε` rather than coin-flipping.

  > **[M1A-2026-09-03] DONE, but this note under-stated the hazard twice, and one arm is still open.**
  > Landed as `ct_packed_sign_convention` + `apply_ct_sign` (`quant/method.py`), decision stored as
  > `container._ct_sign` for Stage B to thread, ε-refusal on an ambiguous histogram, and the *same*
  > `conv` applied to the weight and its zero-point so they can never land in different domains.
  > Two things the note missed: (a) the obvious **strided** replacement is worse than a prefix on every
  > shipped MoE shape — the stride is an exact multiple of the packed row length, so it samples
  > **one packed column** (1 of 512 on w13) and decides the whole checkpoint from 8 input channels;
  > the landed sampler is 512 block-spread runs of 128 contiguous packed words covering `[0, n)`.
  > (b) it was **int32-pinned** (`.view(int32)`, `for p in range(8)`), so a uint8-packed int4 stack —
  > which is how MXFP4/NVFP4/RXF already ship — read 6/8 nibbles as zero and answered confidently
  > backwards; the sampler is now element-width-derived and dtype-preserving.
  > **STILL OPEN:** the decision is per-container, and under TP the two ranks hold *different shards*,
  > so they sample independently and can decide oppositely with neither ever seeing the tie the ε-gate
  > would refuse. The intra-layer w13-vs-w2 case is now a hard `post_load` failure on every serve;
  > the **cross-rank** case is pinned by test (`TestCrossRankHazard`) and **not closed**. It needs a
  > CPU-group compare of the `CtSignConvention` at `post_load`, or a checkpoint-level decision read
  > from `quantization_config` and threaded into every container — which is what Stage B needs anyway.
  > Dense CT linears have no pair to cross-check at all: ~150 independent detections per model.

### 5.3 VRAM accounting

Reserve and map **between `post_load()` (`engine.py:241`) and `_determine_num_pages` (`:245`)**. Device-located chunks are then inside `device_used = old_free − new_free` and, being invisible to `memory_reserved()`, land in the non-torch term of `model_memory` (`engine.py:1173-1176`) — billed correctly with **no sixth subtrahend**.

> **[M1A-2026-09-03] LANDED, and the "no sixth subtrahend" rule has a consequence this section did not
> anticipate: `--weight-offload-device-gb` is effectively MANDATORY whenever offload engages.**
> The device tier is billed *only* inside `model_memory`. With a **derived** budget of
> `total_memory × memory_ratio` (14.4 GiB of a 16 GiB card), the greedy fill takes the tier to the
> budget whenever the stack does not fit, so `model ≥ tier ≥ memory_ratio × old_free` and
> `available_memory` is **negative before** state/draft/graph/snap are counted. `assert num_pages > 1`
> then fires — after ~7 s of pinning and a full checkpoint load — naming four causes that do not
> include the tier, and pointing the operator at `--memory-ratio`, which shrinks the same budget the
> tier already consumed. `bake.UnconfiguredDeviceTierError` now refuses this from integers **before a
> page is pinned**, naming the one lever that moves. This is not a re-introduced on/off switch: the
> plan is still derived on every serve and a fitting model still resolves to an empty plan.
> **Also landed here:** the correction is the *inverse* mistake — a pool-served host arena is an
> ordinary `cuda` tensor to `memory_allocated()`, so ~50 GB of HOST RAM would be billed as device
> memory. `accounting.corrected_model_memory` removes it, **clamped to `[0, live_reading]`** because
> `_determine_num_pages` calls `empty_cache()` between the sample and the read. `num_pages` itself is
> now **MIN all-reduced** over the TP CPU group (`Engine._tp_min_num_pages`) — it never was, and this
> feature adds a new per-rank measured term to it.
> **Open:** the boot assertion is present (`assert_device_accounting` against a *measured*
> `observed_device_bytes`, not a re-derivation), but no post-`post_load()` reconciliation caller
> re-plans from `LayerWeights.from_specs` and diffs the two plans, so the register-direct repack arms
> (`_w_rep`, `_scales_rd`, MOE_W4A16 / MXFP4 / RXF / W8A8 regdirect) remain unchecked against sizing.

**[ADJ]** — the residency and NVMe designs both demand a sixth term at `:1190-1197`. That is a *double* subtraction given eager pre-sizing allocation, and at a 16 GB nominal tier it drives `available_memory` negative and trips the `Not enough memory for KV cache` assert at `:1260`. Resolution: **no new subtrahend**, plus a boot assertion `|Δmemory_allocated − plan.device_bytes| < tol` so the two can never silently disagree, plus an annotation on the single `KV sizing:` line (`:1201-1216`): `weight-arena=<GB> (dev tier, inside model) host=<GB>×tp<N>; step floor <ms> → ≤ <n> forwards/s`. **Never allocate or map after `:245`** — a VMM mapping is invisible to `_prefill_budget_now`'s `reserved − allocated` correction (`scheduler.py:2289`) and silently collapses the prefill budget while the warning at `:2302` misdirects the operator to lower `--memory-ratio`.

### 5.4 Correctness gates — the strongest tests available

**[ADJ]** — `max|Δlogits| == 0` is **unachievable at bs ≤ 2** for reasons unrelated to offload: `kernels.py:665-679` takes an unconditional `M<=2` branch into `mmq_fp8_moe_gemm_scatter`, whose own comment says "NOT bit-exact vs gather_reduce: the atomic reduction order varies", and `MINISGL_MOE_G2FUSE=0` does not guard it. Two runs of the same binary differ. Gates below use a measured same-binary noise floor.

* **A1.0 `eps0`** — offload-OFF, identical fixed forward, N ≥ 5 repeats, record `eps0 = max|Δlogits|`. Also extend `MINISGL_MOE_G2FUSE=0` to bypass the `M<=2` scatter (2-line change) so a deterministic reference exists at all; on that arm assert exactly `0`.
* **A1.1 Identity** — offload-ON (all host) vs offload-OFF, same fixed forward: `max|Δ| ≤ eps0`, and exactly `0` on the deterministic arm. This is a *numerics identity* test on a single forward, **not a generation diff** — the serve is not bit-reproducible past ~32 tokens.
* **A1.2 Permuted-mirror equivalence** *(the strongest test, adopted from the VMM review)* — populate the arena twice, once identity and once under a fixed random permutation of expert rows with the route remapped to match. Logits must agree to `eps0` on a fixed prompt/seed. Any weight↔scale, w13↔w2, or component-offset desync makes this fail catastrophically rather than subtly. Run at TP=1 and TP=2, across every format the checkpoint zoo exercises. **Merge gate.**
* **A1.3 Desync matrix (must-fail)** — desync **each** per-expert component independently (w13 weight, w13 scales, w13 zeros, w2 ×3) and require each to move `max|Δ|` above `eps0`. A no-change result must be *explained* by a proven expert-invariance (`t[0]` bitwise-equal to `t[1:]`) and that component excluded from the granule — never accepted. Run on an **asymmetric** checkpoint so the zeros arm is non-vacuous.
* **A1.4 Populate self-test** — after `populate_from`, read back a fixed pseudo-random sample of 64 granules through the arena and `torch.equal` against the source slices before they are dropped. Plus `_selftest_light`: a fingerprint word per chunk pre-populate, read back, all match — the driver returns `hipSuccess` even when the page table is wrong (`hipMemRetainAllocationHandle` reported a remap as done while the device disagreed), so an out-of-band data check is mandatory, and `_selftest_full` must be phase-gated so it can never run post-populate and overwrite weights.
  > **[P0-2026-09-02] STRONGLY REINFORCED — this is now the most load-bearing gate in §5.4.** Phase 0
  > produced **four** independent cases of the driver reporting success over wrong state: `location=Host`
  > silently allocating VRAM with the property echoed back verbatim (P1/P2/P3); VMM over-commit returning
  > `hipSuccess` on create/map/setAccess and failing only at first touch with a SIGABRT page fault (P3);
  > `hipMemUnmap`→`hipMemMap` serving the stale page with `nonzero_hip_return_codes = []` (P6); and
  > `expandable_segments` + `empty_cache` returning zeroed memory in plain torch (P5). **A capability probe
  > can PASS while the operation FAILS. Assert on the data, never on a return code or a query.**
  > P1's own `cpu_readable()` helper is the cautionary tale: it proved CPU-accessibility by `os.write()`-ing
  > to `/dev/null`, which discards the payload *without copying from user space*, so it reported **every**
  > pointer as readable and the probe faulted instead of recording a result.
* **A1.5 Dense parity** — the same A1.1/A1.2 gates on a dense-only model (llama/qwen3) with its linears host-resident.

### 5.5 Served-path measurement

`tools/weight_offload_serve_ab.sh`, shaped on `tools/moe_g2_split_serve_ab.sh`. One image, one worktree, one kernels build; legs differ by **one knob**. `--num-pages` **pinned** across legs (an auto-sized pool differs because `model_memory` shrinks, so an unpinned A/B measures admission policy). `SPEC=none` for primary arms. Graphs **ON** for every acceptance number — a leg that fell back to eager is VOID, detected by the capture log line *and* `MINISGL_GRAPH_TIMING=1` host deltas. Fresh `os.urandom(16).hex()` nonce per prompt so no prefill is a radix-cache hit. Per-leg artifacts (all four or VOID): `.ledger`, `.banner`, `.json` (raw repeats), `.trace`.

**Provenance. [ADJ]** `engaged()` latches once per name *process-wide* and the eager warmup at `graph.py:348` fires it before capture, so a bare `engaged()` cannot prove an arm ran in the graph. Use `engaged_cap(name)` = `name + (':cap' if torch.cuda.is_current_stream_capturing() else ':eager')` and require the `:cap` variant. `LAYOUT_ID` must be computed **programmatically** — hash every resolved uppercase module-level knob in `minisgl.quant.kernels` plus every `MINISGL_*` key/value in the environment plus the engine and kernels git shas — not from a hand-picked six-knob list, which already misses `MOE_SPLITK_SCATTER`, `MOE_FUSED_SILU`, `MOE_FLAG`, `MOE_G2FUSE`, `MOE_BLOCK_M`, `DISPATCH_CU`, `FORCE_TILED_LINEARS`.

### 5.6 M1 acceptance / kill

* **A1.6** target model boots at TP=2, graphs captured across **all five families** (decode, verify, fused, ddtree, canvas — each runs a warmup forward that reads the arena; each must see a populated, frozen arena), and answers coherently on a **sampled** quality run (temp 0.8 / top_p 0.95 / top_k 50, ≥ 20 prompts × 512 tokens, scored against the degeneration triage table, plus the verbatim-echo probe for a countable out-of-support signal). **Never temp=0.**
* **A1.7** bs=1 decode ≥ `0.75 × (1/(t_compute + 1.33 GB / BW_P1))`, i.e. within 25 % of the pure mechanism ceiling.
  > **[P0-2026-09-02] INSTANTIATED: A1.7 = 14.0 tok/s** at bs=1, TP=2, box idle. Derivation: at TP=2 each
  > rank streams **0.665 GB/token** and the links are independent (P4, eff. 0.999), so **the slower rank
  > sets the step** — `0.665 / 14.48 GB/s = 45.9 ms` (card 1) `+ 7.5 ms` compute = **18.7 tok/s** ceiling,
  > × 0.75 = **14.0**. Under host DDR load the ceiling drops to ~16.3 tok/s (A1.7 → 12.2). If card 1's slot
  > is fixed to Gen5, the ceiling is **32.8 tok/s** (A1.7 → 24.6) — **re-derive A1.7 after any BIOS change.**
  > ⚠️ Two wrong numbers are in circulation and must not be quoted: P4's
  > `restated_zero_cache_dma_tok_s_tp2 = 20.18` (efficiency-only) and P4's prose ~10.3 tok/s (which charges
  > each rank the *whole* 1.33 GB instead of its shard).
  > **[M1A-2026-09-03] RE-INSTANTIATED: A1.7 = 14.57 tok/s.** The byte model is no longer arithmetic:
  > derived off the built model it is **31.22 GiB/rank** of expert bytes, **granule 1,327,104 B**, i.e.
  > **0.637 GB/rank/token**. Step floor `0.637 / 14.48 + 7.5 ms = 51.5 ms` → **19.42 tok/s** ceiling,
  > × 0.75 = **14.57**. §11 unknown #12's ±14 % bracket closes to **−4 %**. This is still card-1-gated
  > and still **PROJECTED** — re-derive after any BIOS change to card 1's slot (K7), and note the
  > resolver prints this threshold itself in the `[weight-offload] gates:` line.
* **A1.8** aggregate tok/s at CONC ∈ {1, 2, 6} reported alongside per-user tok/s and TTFT at 2 k and 8 k prompts. Not a threshold — a *published* curve, because it is the number that decides whether anyone runs this.
* **K3 (hard kill):** A1.1/A1.2/A1.3 cannot be made green. An unmeasurable weight path is disqualifying.
* **K4 (hard kill):** A1.7 measured tok/s < `1/1.57 ×` P0's llama.cpp number. GPU-side streaming is then strictly worse than a hybrid CPU runtime, and the honest answer is to say so. **[ADJ]** — the layer-group review's point stands and belongs in `alternatives_rejected` with the arithmetic: *CPU-side expert GEMV is 1.57× bandwidth-superior at every batch this box reaches; rejected only because the charter requires all compute on GPU.*
  > **[P0-2026-09-02] INSTANTIATED: K4 = 3.178 tok/s** (`4.990 / 1.57`; 3.288 on P0's pooled median — no
  > gate turns on which is cited). **The projected T1 clears it by 5.9×.** The `alternatives_rejected`
  > framing above is now **wrong in tone**: the 1.57× bandwidth-superiority argument assumed llama.cpp
  > reaches the DDR bus. It does not — it is **CPU-compute-bound** on i-quant expert GEMV across 8 threads
  > (P0: ~3.8 GB/s of DDR = ~8.5 % of the bus; storage supplies 0.25–0.43 % of per-token expert bytes).
  > **The charter's all-compute-on-GPU constraint is cheap here, not expensive.** Say *that* instead.

**Graph-capture status: fully captured.** No eager fallback anywhere; the arena is immutable and address-stable, so capture is unaffected by construction.

---

## 6. Cross-cutting contracts (land with M1, not after)

### 6.1 Generality — what a new model or quant format must implement: **nothing**

One free function, derived from the live container after `post_load()`, never a per-format table (which buffers survive is a function of *format × six env knobs*):

```python
def granule_tensors(container, n) -> GranuleSpec:
    # Walk raw __dict__ INCLUDING '_'-prefixed names: BaseOP.state_dict/load_state_dict/
    # post_load all skip them (base.py:59,:76,:101) and every quantized post_load DELETES the
    # public checkpoint names, so a state_dict-driven walk silently finds ZERO expert weights.
```

Rules, each closing a specific reviewed defect:

1. **Classify by `t.dim() >= 1 and t.shape[0] == n and t.is_contiguous()`.** That predicate *is* the kernels' precondition (`t[e]` is a flat range at `base + e·(numel//n)·itemsize`), so derivation and correctness cannot drift.
2. **Fail closed.** Any tensor neither axis-`n` nor declared `_residency_shared` **raises at boot**, with the offending name and the one-line fix in the message. A new format adding a buffer gets a loud failure, never an omitted scale.
3. **Dedupe by storage.** Key on `(untyped_storage().data_ptr(), storage_offset, shape, stride)`. `_GroupedFP8Experts.post_load` sets `_w_op = weight.contiguous().view(uint8)` — an alias — and under `MINISGL_ZAYA_OLDMOE=1` does not delete `weight`, so a naive walk doubles the granule and, after rebinding, *de-aliases* it so `dequant()` and the kernel read different memory. Aliases are one component with N names; partial overlaps raise.
4. **Refuse whole-stack materialisers.** `MINISGL_ZAYA_OLDMOE=1` / `MINISGL_ZAYA_W8A16=1` are **hard-refused** together with offload — `_GroupedFP8Experts.dequant()` materialises the entire `(E,N,K)` bf16 stack per forward.
5. **Detect expert-invariance.** `t[0]` bitwise-equal to `t[1:]` (symmetric CT `_zeros_op` = E copies of `0x88`) → one replicated row, excluded from the granule.
6. **Normalize the bare-tensor container.** `_UnquantizedMoEMethod.create_experts` (`moe.py:508-509`) returns a raw `torch.empty(E, out, in)` — no `__dict__`, so the walk returns nothing and `copy.copy` on it *duplicates storage*. Wrap it in a 2-line `_GroupedUnquantizedExperts(BaseOP)` with `self.weight = t`, and update the two `apply`/`ep_local` reads. **All nine formats then present one interface.** This is a loader policy on the existing `MoEQuantMethod` dispatch, not a kernel fork — KERNEL_CORE_POLICY-clean.
7. **`nn.Module` subtrees are walked too.** `GDNLinearAttn._gdn`, ZAYA's `CCAConv`/`ZayaRouter`, CAM's routers, and the quantized GDN projections in `_MethodLinear` are invisible to a `BaseOP.__dict__` walk — and on Qwen3.5 those are 3 of every 4 layers' dense weights. Descend into `nn.Module` via `named_parameters/named_buffers(recurse=False)` plus that node's own underscore tensors.
8. **Totality assertion.** Sum bytes over every tensor reachable by a dtype-agnostic torch-object walk from the model root; assert it equals the sum over derived specs plus a declared-exempt list. A non-zero residual **aborts** with the missing paths printed. Without this, "fail closed" is only closed over whatever the walk happened to reach.

**Dense is the same mechanism.** `GranuleSpec.granule_axis = None` means whole-container-is-one-granule; `_LinearTPImpl.post_load` (`linear.py:73-76`) registers with the same walker; `quant/method.py`'s `layer._w_rep_wide` / `_w_packed_op` / `_scales_op` / `_zeros_op` are the same repack-and-delete shape. `lm_head` and embeddings are eligible under the same registry. **A1.5 makes dense a merge gate, not a follow-on.** The eight dense-only model families get offload from day one.

> **[M1A-2026-09-03] LANDED as `granule.py`. A new MODEL family implements nothing. A new QUANT FORMAT
> implements exactly ONE line, and forgetting it is a loud boot failure, not silence.**
> ```python
> self._num_experts = num_experts   # in __init__, before post_load
> ```
> `declared_granule_count` uses an `UNSET` sentinel — deliberately **not** `None`, which legitimately
> means "dense" — so an undeclared granule axis raises `GranuleError` naming the fix, instead of
> silently reading a 512-expert stack as one granule (it did, before this was found: `granule_bytes`
> 4× too large and `expert_slice(c, 999)` accepted). Two coverage tests walk all eight quantized
> containers at **construction** time *and* through `create_moe_quant_method`, so a format added to the
> selector but not to a test list is still caught. Everything else is derived; `sizing.py`'s analytic
> table is a fallback, and an unrecognised scheme is sized by a **meta-device build of the real
> container** rather than refused. No kernel forked, no new package.
> **Rule-by-rule status:** 1 ✅ (contiguity is checked *before* the byte-range dedupe — checked after,
> `t` and `t.transpose(1,2)` key identically and the strided view is silently absorbed). 2 ✅.
> 3 ✅ (byte-range key + aliases rebound with **each alias's own** dtype/shape). 4 ✅ but moved: the
> refusal is a **snapshot taken inside `post_load`**, not a live `os.environ` read at placement time.
> 5 ⚠️ **CHANGED — content-based invariance was a TP DESYNC.** Two ranks hold different shards, so a
> byte-derived exclusion could differ per rank with no collective to notice. It is now a
> `_residency_shared` **declaration** (config-derived, hence rank-identical) that is **verified
> bitwise against every live row**, so a stale declaration raises instead of dropping a real
> per-expert buffer. 6 ❌ **NOT landed** — `models/weight.py::_get_expert_stack_info` strips the
> trailing `.weight`, so wrapping the bare tensor breaks every unquantized MoE checkpoint's state_dict
> keys; handled at the descriptor level instead (a bare `torch.Tensor` yields one synthetic component
> named `weight`, byte-stable across that future change). 7 ✅. 8 ❌ **NOT landed, and it is the one
> real generality hole:** without the totality assertion the walk is fail-closed only over what it
> *reaches* (`BaseOP` / `nn.Module` / list / tuple / dict), and a new **non-tensor decode decision** —
> the next `_ct_sign` — is fail-**open** by construction. `_granule_policy` records such decisions and
> the fingerprint hashes them, but nothing forces a new format to declare one.
> **Dense: mechanism ✅, enumeration ❌.** `_LinearTPImpl` is an `ExpertContainer` with
> `_granule_dense = True` and presents the *identical* four-method surface (same function objects —
> a test asserts identity, so a look-alike fails). But `plan.py` and `attach_seams` enumerate
> `MoELayer` only, so **A1.5 is NOT satisfied**; a dense checkpoint currently gets an explicit warning
> saying dense is not planned, rather than a silent "nothing to do".

### 6.2 Config: no env gate at merge

**[ADJ]** — every subsystem proposed an on/off env knob; the standing rule forbids it, and a control leg from the same binary is exactly the emulated baseline that is not permitted.

* `EngineConfig.weight_offload_gb: float = 0.0` + `--weight-offload-gb`, `default=ServerArgs.weight_offload_gb`, resolved in **one** function `resolve_weight_plan(config)` (defensive `getattr(config, ..., default)` per `config.py:87`). The plan is **derived**: compute offloadable bytes vs measured VRAM minus dense-resident minus a KV floor; if everything fits, the plan is empty and the code path is a no-op with zero device cost — so it is still exercised on every serve and cannot rot.
* The flag only **clamps** an automatic decision. It is never "is my feature enabled".
* No `MINISGL_WOFF_CAPTURE_STREAM`, no `MINISGL_WEIGHT_SLAB_REMAP`, no `MINISGL_MOE_OFFLOAD`, no `_MODE` enum. `MINISGL_WEIGHT_ARENA_DIR` and the oracle knobs survive as durable-fixture paths, forwarded as `"${VAR:-}"` and read through `kvcache/_envutil.env_int/env_float`.
* Validated operating point lands in `tools/serve.sh`'s per-model table (`:44-55`, measurement in the comment, composed into `cmd=(...)` at `:629-643`) **and on the `[serve]` banner** (`:646-673`) with the resolved plan, `LAYOUT_ID`, and the step floor.

### 6.3 Metrics

**Probe phase (M0–M2):** `logger.info_rank0(summary())` on the idle tick **plus a step-counted cadence** (a busy serve may not idle for hours and SIGTERM does not run `atexit`). No Prometheus.

> **[M1A-2026-09-03] PROBE-PHASE REPORTING LANDED; METRICS DID NOT.** `seam_summary()`,
> `WeightPlanResolution.render_lines()` / `summary_line()`, the `[weight-offload]` and `[weight-arena]`
> banner lines, the `KV sizing:` annotation (which now also prints the **applied**, post-clamp
> corrections) and `accounting.report().render()` all exist. **No Prometheus gauge exists at all** —
> that is M4, and the five-hop/`−1`-sentinel/TP-reduce discipline below applies verbatim or they
> export a flat zero rather than erroring. Note the P5b finding when wiring them: **`amdgpu
> mem_info_gtt_used` is NOT a usable host-tier residency signal** (it stays at exactly 0 for
> `hipHostMalloc` userptr pinning) — use `MemAvailable` or the arena's own accounting.
> Use plain `info`, not `info_rank0`: a rank whose plan diverged is exactly the rank you need to hear.

**Ship (M4):** `weight_arena_host_bytes`, `weight_arena_device_bytes`, `expert_layer_miss_rate`. Ten hops, four of which fail **silently** — both `_FAST_SCALAR_FIELDS` tuples (`message/utils.py:35-60`, `:61-86`), `tokenizer/server.py:109-137`, `api_server.py:2537-2563`, `metrics.py:185-200`. **Initialise each gauge to sentinel `−1` and require observing `−1` on `/metrics` before the first request** — otherwise "the cache never hits" and "the wire is broken" are the same observation. TP-reduce on the existing CPU group at the idle tick before emitting (`io.py:96-101` is tp-primary only and the frontend sums DP but never TP, so under EP a naive gauge reports rank 0's half of the distribution as the whole).

---

## 7. Phase 3 — M2: the static-prior device tier. ~5 days. **Conditional on A0.4 and P2.**

> **[P0-2026-09-02] RESCOPED, AND PARTLY PROMOTED INTO M1.**
> **Per-expert placement is unbuildable** (no mixed-media VA) **and unjustified** (P2 never measured the
> miss-count curve). Rebuilding it as two stacks + `slot_of` reopens §4.4/§8's `route_E`/`align_E`
> prerequisite — **do not.** **M2 becomes layer-granular placement**, which A2.3 below already names as
> the fallback: a layer is all-device or all-host, both are ordinary contiguous stacks, no `slot_of`, no
> route change, no kernel change, and the `h¹⁰` effect is irrelevant by construction.
>
> **Part of this is no longer optional and no longer M2's:** P3b measured the pinned host ceiling at
> **62 of 68 GiB on an idle box**, so a 100 % host arena for the target model **probably does not boot**.
> Every GB on device is a GB the host arena does not need. Projected (TP=2, card-1-gated 14.48 GB/s host,
> 692 GB/s device, 7.5 ms compute, 34.4 GB expert bytes/rank):
>
> | device `f` | dev GB/rank | host GB total | step ms | tok/s | vs T1 |
> |---|---|---|---|---|---|
> | 0.00 | 0.0 | **68.8 ❌ does not fit** | 53.4 | 18.7 | 1.00× |
> | 0.10 | 3.4 | 61.9 ⚠️ at the measured ceiling | 48.9 | 20.4 | 1.09× |
> | 0.20 | 6.9 | 55.0 ✅ | 44.4 | 22.5 | 1.20× |
> | 0.25 | 8.6 | 51.6 ✅ | 42.2 | 23.7 | 1.27× |
> | 0.30 | 10.3 | 48.2 ✅ | 39.9 | 25.0 | 1.34× |
>
> **Conditional on P2′, not P2.** A0.4's break-even arm is vacuous (§1(a) struck); it binds only on 50 %.

> **[M1A-2026-09-03] P2′ RAN. M2 COLLAPSES TO ~2 DAYS, and per-expert placement is CLOSED, not deferred.**
> The curve is **LINEAR on both cards** (`cliff_index` 0.085–0.095 across all six measurements; effective
> miss concurrency `W` = 0.72–0.82, i.e. misses are serialised, the opposite of a max-gated layer). So on
> a linear curve per-expert and layer-granular are **provably equal at equal byte budget**, and the
> measured `per_expert_gain_over_layer_granular` is **1.013× (card 0) / 1.009× (card 1)** at the h≈0.25
> band capacity forces, peaking at 1.063× / 1.050× somewhere 16 GiB of VRAM cannot reach. **Do not build
> a per-expert device tier.** (`stacks.ExpertStackTable`'s docstring carries the exact HIP change — one
> `TwoStackWLoad<Base>` WLoad policy overriding `wq/ws/wz_expert` on the two existing shared cores, plus
> both P2′ driver defects — should that ever be revisited. Nothing is forked.)
> The layer-granular tier itself is **already in M1-A** (`placement.plan_layer_granular`), so what is
> left of M2 is: the static prior file (checkpoint sha + `LAYOUT_ID` + tp/ep shape), and A2.4's Pareto
> sweep **measured** instead of projected — including its second axis (max context / `max_running`
> surrendered per GiB, from the real `cache_per_page`, not the prose "~200 k KV tokens/GiB").
> **A2.3 and K5 are effectively unfireable**: with the curve linear there is no per-expert increment to
> fall short of. **The table below is superseded** — its "fits" column charges the payload, while the
> arena pins whole chunks (see the §5.1 block). Corrected: all-host = **62.44 GiB/node payload →
> 80.00 GiB pinned vs a 55.80 GiB ceiling → INFEASIBLE**; required tier **11.06 GiB/rank** as landed,
> **≈ 3.90** once the rows are named regions; projected 19.42 → 23.03 (f=0.20) → 24.56 (f=0.25) tok/s.

Top-`D` expert rows per layer backed by device pages, decided at boot from the baked prior; the rest host-backed. **No runtime residency at all.**

**What changes vs M1:** `layout.py` gains the per-region device/host page split; `plan.py` gains `--weight-prior <file>` (produced by `tools/expert_cache_sim.py`, keyed by checkpoint sha + `LAYOUT_ID` + tp/ep shape); `arena.py` maps two handle populations. **Nothing else.** No kernel change, no `slot_of`, no route change, no capture hook, no fences, no TP coordination.

**Acceptance:**
* **A2.1** all M1 correctness gates re-run at `D > 0` — including A1.2 permuted-mirror with the permutation crossing the device/host boundary.
* **A2.2** measured layer-level miss rate matches the simulator's prediction for the same prior within 5 points. A divergence means the shipped placement is not the simulated one.
* **A2.3** bs=1 tok/s improves over M1 by at least `0.7 × (predicted from P2's miss-count curve)`. If the realised gain is far below the P2-predicted gain, the `h^10` all-resident-layer effect dominates and the tier is not worth its VRAM — report and consider **layer-granular placement** instead (a layer is all-device or all-host; makes the streamed case one contiguous 1.42 GB read at 93 % of link and makes `h^10` irrelevant).
* **A2.4 Pareto** — sweep `D` and report **both** axes: tok/s *and* max context / max_running (each GB of device tier is ~200 k KV tokens surrendered). Pick from the frontier, land it in the serve table.

**Graph-capture status: fully captured**, identically to M1.

**K5:** A2.3 fails and layer-granular placement also fails → ship M1 only.

---

## 8. Phase 4 — M3: dynamic residency. **Do not start.** Conditional on K2 failing decisively.

> **[P0-2026-09-02] UPGRADED FROM "DO NOT START" TO DEAD.** P6 is conclusive and does not depend on K2:
> `hipMemUnmap`→`hipMemMap` at a used VA scores **10/40 = handles⁻¹** — the totally-broken signature, not
> "works a quarter of the time" — on both media, both the copy engine and the shader. **Every call returns
> `hipSuccess`.** Writes are **not dropped**; they land on the wrong physical page
> (`distinct_landing_sets=[[0]]`, `writes_lost_entirely=0`) — silent cross-page corruption. It survives a
> 256 MiB (4× MALL) flush, so it is the page table, not a cache. Controls held (`control_ok`,
> `park_integrity_after`, `shader_arm_trustworthy` all true), so the verdict is attributable to the remap.
> **K2 now only chooses *which* static prior, never whether to build dynamic residency.**
> **Action:** keep `tools/vmm_conformance.py` and wire it into CI with `--expect broken` so a future AMD
> fix exits **3** instead of silently passing. Re-run it after any ROCm bump.
> *(Timing note: the full remap cycle measured **47.5 µs device / 46.6 µs host**, below the recorded
> 71.7–99.1 µs band; "unmap dominates" is confirmed at 31.3 of 47.5 µs. The definitions differ — P6 excludes
> `hipMemCreate`/`Release` — reconcile before quoting either. Only one size (8 MiB) was tested, so
> size-independence is neither confirmed nor refuted.)*

Only if the oracle shows `LRU − static_prior ≥ 15 points` *and* M2's measured throughput leaves ≥ 30 % on the table. In that case, and only then, the full machinery returns and every fatal finding must be resolved explicitly:

* **Route width:** add `route_num_experts` / `align_num_experts` as **separate explicit arguments** to `moe_hip.moe_route_align`, `w4a8_moe`, `w4a16_moe`, `w8a8_moe`, `rxf_moe` — stop deriving `num_experts` from `w13.shape[0]` (`kernels.py:508/780/907/1032/1162/1258`), and assert `gating_output.shape[-1] == route_num_experts`. Land the 4-line `bi < num_experts` bounds guard in `align_body` that `moe.py:1130` already asks for. **Prerequisite, not a perf item** — a C-slot stack today silently routes over the first C logits.
* **Fence at the choke point:** `slot_of` publish + `wait_event` in `Engine.forward_batch` / `forward_verify` / `forward_canvas`, **not** in `GraphRunner.replay` — there are six replay entry points plus the eager path, and `replay_canvas` runs three extra reference forwards under a bit-exactness gate that raises.
* **WAR:** per-slot `lastuse` event recorded on the compute stream after every replay; the fetch stream waits it before overwriting. Window depth derived from `1 + max_unsynchronized_replays_per_step`, not hardcoded 2.
* **Publish ordering:** `slot_of` written on the *same stream as the fetch, after it*; double-buffered pinned staging; delta-only `index_copy_`, never a whole-table H2D.
* **EP composition:** gather **after** the clamp (`local = where(is_local, ep_i − lo, 0); slot = slot_of[local]`), never before — `slot_of_local[ep_i − lo]` indexes out of bounds and device-asserts inside the graph.
* Capacity: `C ≥ max over ALL captured shapes of (bs·qlen·top_k) + P_pinned`, registered by every capture family before any capture. At bs=6/K=4 that is ~300 slots/layer, 3× the naive estimate.

Estimated **+3 weeks** and it reintroduces every risk this plan exists to avoid. Treat K2 as the expected outcome.

---

## 9. Phase 5 — NVMe. Deferred, and here is why it is a different feature

NVMe cannot serve the decode path: measured 6.9–9.0 GB/s at the granule with QD ≥ 4 gives 5–7 tok/s at 1.33 GB/token, and mmap random-granule reads are 0.64–0.87 GB/s (10× worse than O_DIRECT — mmap is disqualified). **NVMe's job is capacity beyond RAM and restart time, nothing else.**

~~**Critical constraint the designs missed:** host-located `hipMemCreate` pages are **anonymous**, so a VMM-mapped mixed stack cannot be file-backed; conversely `hipHostRegister` over a file mapping gives a device pointer but no control over its VA, so it cannot be interleaved with device pages. **Mixed-media placement and NVMe overflow are mutually exclusive with current APIs.** If P3 shows 2 × 34 GB of host-located pages fits, take the mixed stack and defer NVMe. If it does not, the fallback is file-backed `mmap(MAP_SHARED)` + `hipHostRegister` on a prefix — which gives a free three-tier gradient (registered / page-cache / NVMe) at zero design cost, and forfeits the device tier.~~

> **[P0-2026-09-02] MOOT, AND INVERTED.** There is no VMM-mapped host stack **at all**, so nothing is
> mutually exclusive — the file-backed path is unconstrained by the (nonexistent) mixed stack, and the
> layer-granular device tier coexists with it freely.
> **And the fork resolved the wrong way for capacity:** P3b measured **62 of 68 GiB** for two pinned ranks
> on an *idle* box (34.0 + 28.0; rank 1 stopped on the `MemAvailable` floor with 114,813 pages swapped).
> With the engine, KV pool and page cache resident there is less. **So the page-cache/NVMe tier is more
> relevant, sooner, than "deferred" implies.**
> **Before adopting it, run P3c.** The file-backed route is demonstrated only at **4 GiB/rank**: it works
> (28.18 GB/s device touch, 0 resweep failures, `Cached` delta = 1.000× committed, `MemAvailable` delta ≈ 0
> — i.e. genuinely zero anonymous RAM), but setup is **3.3× slower** (1.46 vs 4.88 GB/s) and an 8 GiB /
> 2 GiB-chunk attempt wrote all 8 GiB then **spun > 711 s of user CPU** with no progress and was killed.
> That pathology must be root-caused before this is a plan and not a hope.

When it lands: dedicated ZFS dataset `recordsize=1M compression=off direct=always primarycache=metadata` (blobs are incompressible — measured `du == apparent` on a 5.0 GB AWQ shard; `direct=standard` is already set on both pools); page-aligned granule stride; a persistent `preadv` worker pool (never per-batch thread spawn — measured ~700 µs/iteration and entirely thread-bound); striping opt-in only (nvme0 is the OS root pool and two-pool concurrent measured *slower* than nvme1 solo at the granule). `liburing` is absent from the serve image; `os.preadv` + `O_DIRECT` are present.

**Note:** creating the tuned datasets requires a sudo action that was denied this session, so the recordsize/`direct=always` improvement is a **specified experiment, not a measured claim**; all quoted NVMe bandwidth is from the default 128 K datasets and is a conservative floor.

---

## 10. Effort, critical path, kill summary

| Phase | Work | Days | Blocks |
|---|---|---|---|
| P0 | Probes P0–P6 | 3 | everything |
| M0 | Oracle + replay + simulator + tests | 4 | (parallel with P0) |
| M1-A | `hipvmm` + `layout` + `arena` + `granule` + `registry` + `plan` | 4 | P1, P3, P5 |
| M1-B | Engine/config/args wiring, accounting, serve table, compose | 1 | M1-A |
| M1-C | Correctness gates (eps0, identity, permuted-mirror, desync matrix, dense parity) | 2 | M1-B |
| M1-D | Served A/B harness + capture across all five families + TP=2 | 2 | M1-C |
| M1-E | *Stage B chunked load* (required for the target model only) | 3 | M1-D |
| M2 | Static-prior device tier + Pareto sweep | 5 | M0 GO, P2, M1 |
| M4 | Metrics chain + sentinel verification + operating point | 2 | M2 |
| M3 | Dynamic residency *(expected: not built)* | +15 | K2 fails |
| NVMe | *(deferred)* | +10 | P3 fork |

~~**Critical path:** P1 → P3 → M1-A → M1-C → M1-D → M1-E → M2 → M4 ≈ **26 working days** (~5.5 weeks) to a validated, captured, TP=2 operating point serving a model that does not fit VRAM, with dense models covered by the same landing. M0 runs concurrently and gates only M2.~~

> **[P0-2026-09-02] REVISED CRITICAL PATH.** P1/P3/P5 are done; **P2′ and P5b are inserted**, M1-A loses
> the VMM binding (−~1 d) and gains the layer-granular split (+~1 d), and part of M2 moves into M1 as a
> capacity prerequisite (K6):
>
> ```
> NOW      P2′ (0.5 d)  +  P5b (0.25 d)          both cheap, both gate shape / block M1-A
>          BIOS: card 1 → Gen5 (1 h)             independent, free 1.75× if it lands
> THEN     M1-A  pinned host arena + layer-granular split   (VMM binding dropped)
>          M1-B..D  as written, with A1.7 = 14.0 tok/s and K4 = 3.178 tok/s
>          P3c (0.5 d) in parallel IF the host arena does not fit
> LATER    M2 = layer-granular tier, sized from P2′
> NEVER    M3 / T3 — P6 is conclusive (§8)
> SEPARATE expandable_segments × empty_cache corruption — not this feature's bug (§11 #7)
> ```
>
> M0 still runs concurrently and still gates only M2 — but note A0.4's break-even arm is now vacuous
> (§1(a) struck) and K2 is subsumed by P6, so M0's remaining job is sizing the prior, not go/no-go.

> **[M1A-2026-09-03] CRITICAL PATH AGAIN, after P2′/P5b and the M1-A landing.** P2′ and P5b are DONE
> and both went the cheap way (linear curve → no per-expert tier; torch takes the host pointer → no
> +3 d C++ extension). M1-A's code is in — 10.1 k new lines under `python/minisgl/weights/`, 722 tests,
> **zero GPU runs**. Remaining, in order:
>
> ```
> M1-B  2.5 d  (1) rebuild the serve image — `tail_hip has no attribute silu_and_mul` blocks
>                  every import-level check of engine.py today
>              (2) NAMED arena regions instead of anonymous headroom  <-- worth 7.2 GiB/rank of VRAM;
>                  do it BEFORE any measurement, it changes the feasibility answer
>              (3) first GPU run: `pytest -m gpu`, serially, on BOTH cards
>              (4) pin one 30 GiB arena for real, timed, tripwire live
>              (5) serve-table entry + [serve] banner + compose env
> M1-C  2.5 d  A1.0 eps0 -> A1.4 populate self-test -> A1.1 identity -> A1.2 permuted-mirror
>              (merge gate, TP=1 and TP=2) -> A1.3 desync must-fail on an ASYMMETRIC checkpoint
>              -> A1.5 dense parity, which needs the dense enumeration + dense seam that DO NOT EXIST.
>              Budget dense inside this milestone: MoE and dense land together.
> M1-D  2 d    served A/B, graphs ON, --num-pages pinned, engaged_cap() not engaged(),
>              programmatic LAYOUT_ID, diff the engaged ledgers per leg.
>              Discharges A1.6 (five capture families) and measures A1.7 / K4.
> M1-E  3 d    Stage B chunked load — the target checkpoint OOMs during load_state_dict today.
>              Thread container._ct_sign into every chunk. First CPU stores into arena pages: the
>              visibility discipline is currently UNREACHABLE, not proven — write it down then.
> M2    2 d    (down from 5) static prior file + MEASURED A2.4 Pareto with its second axis.
> M4    2 d    metrics chain (five hops or it exports a flat zero), -1 sentinel, TP-reduce.
> K7    1 h    BIOS: card 1 root port Gen4 x8 -> Gen5. Free 1.75x. Sample link speed MID-DMA.
> P3d   0.5 d  can this box pin a 4 GiB chunk? worth 0.65 GiB/rank of device tier.
> P3c   0.5 d  file-backed overflow tier — ONLY if the device tier cannot be afforded.
> ```

**Kill / redirect summary:**

| Gate | Kill condition | Fallback | **[P0] Status** |
|---|---|---|---|
| P1 | kernel-read from host pages < 13 GB/s | explicit-copy layer-group streaming; T2 dead | **NOT MET** — 28.93 card 0. ⚠️ card 1 at **14.48 clears by only 1.11×**; treat card 1 as marginal and re-derive this floor against **14.48**, not 28.1. |
| P2 | mixed-media GEMM wrong, or superlinear miss cliff | layer-granular placement, or T1 only | **NOT EVALUATED.** Mixed-media is unbuildable, so the probe aborted on its precondition. **The fallback (layer-granular) is taken anyway** — reached by capability failure rather than a timing cliff. **Do not read this as "T2 survived".** |
| P3 | 2 × 34 GB host-located does not fit | file-backed mmap tier; no device tier; NVMe path opens | **FIRES, twice over.** Host-located VMM does not exist **and** 2 × 34 GiB does not fit (62 GiB idle). → **fallback taken**, but via *pinned* host memory (P3b: 34 GiB/rank, 4.88 GB/s), with file-backed as the overflow tier pending P3c. |
| P5 | torch cannot take a foreign device pointer in-image | C++ `from_blob` extension, +3 d | **NOT MET** — green in 4/4 legs. **But all legs ran over device memory → P5b must re-test over a pinned host pointer before this gate is closed.** |
| A0.4 / K1 | byte hit rate < 40 % (break-even vs CPU) | ship T1 only, or nothing | **CANNOT FIRE on break-even grounds** — measured break-even is `h = 0 %` (§1(a) struck). A0.4 binds only on its 50 % arm. |
| K2 | LRU − static_prior < 5 pts | **never build M3** — the expected and best outcome | **SUBSUMED.** P6 killed M3 outright; K2 now only picks *which* static prior. |
| K3 | permuted-mirror / desync gates cannot go green | hard kill; unmeasurable weights are disqualifying | unchanged — and **reinforced**: P3/P6 both show `hipSuccess` returned over a wrong page table, so out-of-band data checks are the only defence. |
| K4 | M1 tok/s < 0.64 × llama.cpp | hard kill; the charter constraint costs more than it buys — say so | **K4 = 3.178 tok/s.** Projected T1 (18.7) clears by **5.9×**. |
| K5 | M2 gain far below P2 prediction | ship M1 only | **unevaluable until P2′.** For layer-granular the prediction is analytic (§7 table), not P2-derived. |

> **[P0-2026-09-02] Two new gates the plan did not carry:**
> * **K6 (capacity, hard):** the resolved host arena must fit in `MemAvailable` with headroom for KV, page
>   cache and the engine. P3b measured the pinned ceiling at **62 of 68 GiB on an idle box**. A 100 % host
>   plan for the target model **fails this today** → M1 must ship with a layer-granular device tier, or the
>   file-backed overflow tier (P3c), or a smaller checkpoint.
> * **K7 (box-ops, not a design gate):** card 1's root port `0000:00:01.3` trains **Gen4 x8**. Every TP=2
>   ceiling in this document is gated by it. **Attempt a BIOS fix before costing the design around
>   14.48 GB/s** — success is a free **1.75×**.

> **[M1A-2026-09-03] KILL-TABLE UPDATE.**
> * **P2 / K5 — RESOLVED by P2′, and effectively unfireable.** The curve is LINEAR, so there is no
>   per-expert increment for the realised gain to fall short of; layer-granular *is* the design, and it
>   already landed in M1-A. Read "worth building" nowhere in `p2prime.md` — the probe's `decide()`
>   prints that label from the `cliff_index` band alone and ignores the gain number, which is 1.01×.
> * **P5 / P5b — CLOSED GREEN.** The +3 d C++ `from_blob` fallback is not incurred.
> * **K6 (capacity) — FIRES, and is now quantified.** All-host for the target shape is
>   **62.44 GiB/node of payload → 80.00 GiB PINNED at 2 GiB chunks vs a 55.80 GiB usable ceiling**.
>   The resolver refuses it at config time in milliseconds, before a page is pinned, naming the
>   required tier. **11.06 GiB/rank as landed; ≈ 3.90 once the rows are named regions** — so K6 is
>   currently firing ~2.8× harder than the physics requires, and the fix is a sizing change, not RAM.
> * **NEW, K8 (boot arithmetic, hard, already enforced):** with the device tier billed inside
>   `model_memory` and the budget *derived* as `total × memory_ratio`, `available_memory` goes negative
>   on every offloading serve. `bake.UnconfiguredDeviceTierError` refuses this pre-pin. Consequence:
>   **`--weight-offload-device-gb` is effectively mandatory whenever offload engages** (§5.3 block).
> * **K3 / A1.6 / A1.7 / K4 — ALL STILL UNMEASURED.** Nothing has touched a GPU. The projections clear
>   K4 by 6.1×, and a projection is not a gate result.

**Non-kills, to state explicitly so they are not misread:** a low hit rate measured at E=256 on an existing checkpoint (methodology limitation, §4.3); a bs=1 regression against the 35B (different model, and this is a capacity feature).

---

## 11. Genuinely unknown — and the experiment that resolves each

> **[P0-2026-09-02] SCOREBOARD:** #1 ✅ answered · #2 ❌ **still open, now the top unknown** · #3 ✅ answered
> · #4 ✅ answered (**NO**) · #5 unchanged · #6 ✅ answered for pinned · #7 ✅ green **+ a landmine** ·
> #8, #9 untouched. **Five new unknowns added at the end.**

1. ✅ **Kernel-read bandwidth from host-located pages, at the grouped-GEMM's tiled access pattern, on gfx1201.** Every ceiling in this plan multiplies through it and the only extant number (127.2 GB/s) is an L2-resident artifact. → **P1**, half a day. Highest-leverage unknown in the document.
   > **[P0] ANSWERED: 28.93 GB/s (card 0) / 14.48 (card 1)**, ≥ 256 MB working set. **The tiled pattern
   > costs nothing** vs a linear stream (28.93 vs 28.94); random-128 B costs 49 %. Measured via
   > `hipHostMalloc(Mapped)` zero-copy, **not** the plan's nominal mechanism.
2. ❌ **Mixed-media GEMM: correctness, and whether per-layer time is linear or cliffed in miss count.** Decides whether the device tier is worth VRAM at all, and whether placement should be per-expert or per-layer. → **P2**.
   > **[P0] STILL OPEN — and now the highest-leverage unknown in the document.** P2 aborted on its
   > precondition (media separation ratio 1.00) and never timed anything. The stakes: host read is
   > **4.2 % of HBM (28.93 vs 692 GB/s, a 23.9× ratio)**. If a layer is gated by its slowest workgroup,
   > one miss of top-10 costs nearly what ten do, `P = h¹⁰` governs, and per-expert placement needs
   > `h ≥ 0.9895` — unreachable (a generous 50 % byte hit rate gives `0.5¹⁰ = 0.1 %` of layers fully
   > resident). If it is linear, the same tier is worth up to **2×**. **2× vs 0× on one unmeasured curve.**
   > → **P2′**: rebuild on **explicit two-stack pointers** (device `hipMalloc` + pinned host), no VMM.
   > Constructible today. Layer-granular placement sidesteps the question entirely.
   > **[M1A-2026-09-03] ANSWERED: LINEAR. The spread collapses to ≈ 0×, not 2×.** P2′ ran on both cards
   > over explicit two-stack pointers with a per-expert `__constant__` pointer table.
   > `cliff_index` = **0.090 / 0.094** against a pure-linear 0.100 (band: LINEAR ≤ 0.25), 0.085–0.095
   > across all six independent measurements. Effective miss concurrency `W` = 0.72–0.82 — misses are
   > **serialised**, the opposite of a max-gated layer, so **the `h¹⁰` fear is REFUTED**. Decisive
   > internal check: the marginal cost of one host-resident expert equals **exactly one 2.338 MiB
   > granule at the card's independently measured host bandwidth** (102.8 % card 0 / 101.3 % card 1),
   > and card 1's slope is exactly 2× card 0's, tracking the Gen4-vs-Gen5 root port.
   > **But the decision number is the gain, not the band:** `per_expert_gain_over_layer_granular` is
   > **1.013× / 1.009×** at h≈0.25 (the band capacity forces) and peaks at 1.063× / 1.050× at an
   > unreachable h=0.75–0.90. **Layer-granular captures 95–99 % of the value** with no route change, no
   > `slot_of`, no second stack in the hot path. **Per-expert placement is closed.** Caveats: synthetic
   > shape, bs=1 only, distinct-expert routing, two GEMV arms in isolation, no capture.
3. ✅ **PCIe root-port width and TP=2 concurrent H2D.** Two reviewers read the topology differently (x8 at `00:01.x` vs x16 at `03:00.0`/`07:00.0`, downstream of a board switch). At TP=2 the DDR bus, not the links, may bind. → **P4**.
   > **[P0] ANSWERED.** Bifurcated x16 direct to the root complex, **independent x8 per card**; the x16
   > readings at `03:00.0`/`07:00.0` are the cards' own on-package bridges. **Concurrency is free**
   > (efficiency 0.999; aggregate 43.0 GB/s = 91 % of the 47.25 GB/s *trained* ceiling). DDR does not bind
   > at idle — but **under load it does**: aggregate collapses to 24.7 GB/s (penalty 0.575) and both cards
   > converge to ~12.4 GB/s, with the *fast* rank absorbing the whole loss. **The real finding was
   > asymmetry, not concurrency:** card 1's root port is trained **Gen4 x8** (16 GT/s) vs card 0's Gen5 x8,
   > in 78/78 mid-DMA samples; both hit 91 % of their *own* link, which is what proves it is the slot.
4. ✅ **Whether 2 × 34 GB of host-located `hipMemCreate` pages fit on a box with ~16–28 GB in use, a 16 GiB ARC cap, and 10 GB of zram already committed** — and, if they do, whether concurrent two-card streaming still achieves the single-card figure with that much memory locked. → **P3**, run with the engine loaded and ARC warm, not on an idle box.
   > **[P0] ANSWERED: NO, twice.** Host-located `hipMemCreate` pages **do not exist** (they are VRAM), and
   > the surviving pinned route reached **62 of 68 GiB** — on an *idle* box with **no engine loaded**, with
   > 114,813 pages swapped, rank 1 stopping on the `MemAvailable` floor. **See K6.** *(The plan asked for
   > this with the engine loaded; it was run without, so the 62 GiB is optimistic.)*
5. **Expert reuse at E=512 / top_k=10 on real traffic.** Cannot be measured directly (the model does not load). Mitigated by router-only replay + the E=256 cross-validation (A0.3); if A0.3 fails, there is no trustworthy go/no-go number and that must be reported as the finding.
6. ✅ **CPU/GPU coherence granularity of host-located VMM pages under a CPU write followed by a kernel read.** `vmm_probe2.py` proved the *device*-write direction; `populate_from` writes from the host side unless forced through the device pointer. → an arm of **P1**; mitigation is already in the plan (write through the device VA, then `synchronize()`).
   > **[P0] ANSWERED for the selected mechanism.** Pinned zero-copy passed **every** coherence test on
   > **both cards, both directions**: dev-write→CPU-read, and CPU-write→kernel-read **with no explicit
   > flush**, after a fence, after `clflush`, and at **sub-64 B line granularity**. So `populate_from` may
   > write from the host side directly. **Keep the write-through-the-device-pointer mitigation anyway** —
   > it costs nothing, and this box has repeatedly returned `hipSuccess` over a wrong page table.
7. ✅ **Whether `expandable_segments:True` (the compose default) coexists with graph capture *and* a user `MemPool` over a user reservation**, and whether `torch.cuda.empty_cache()` (called at `graph.py:314`, after the arena exists) invokes the custom free callback for a live pool block. → **P5**, in-image.
   > **[P0] ANSWERED GREEN — AND IT SURFACED A SERVE-WIDE LANDMINE.** They coexist: 4/4 legs green, the
   > arena survives `empty_cache` 12/12 with the pointer stable, the **custom free callback fires 0 times
   > ever** (live block, dropped block, capture-enter), and 24/24 replays are numerically exact.
   >
   > **⚠️ COLLATERAL — NOT AN OFFLOAD BUG, POSSIBLY LIVE IN PRODUCTION TODAY.** In **plain torch** — no
   > `MemPool`, no custom allocator, no reservation — a 16 MiB float32 tensor whose `fill_` completed and
   > was synchronised read back **all zeros in 7 of 12 reps**, at a VA torch had just unmapped and
   > re-mapped (VA reuse 1.00). With `expandable_segments` **off: 0 of 12.** This is the P6 remap defect
   > surfacing inside torch's own expandable-segment allocator. **`engine/graph.py:314` calls
   > `empty_cache()` and `docker-compose.yml:68` sets `expandable_segments:True` — both ingredients are in
   > production.** This is a demonstrated broken **primitive**, not a demonstrated wrong serve output;
   > nobody has yet shown a live weight/state tensor landing on a poisoned re-map. **It needs its own
   > investigation and must not be reported as a known serve corruption until that step is done.**
8. **Whether ZFS `recordsize=1M` + `direct=always` actually beats the measured 6.9–9.0 GB/s.** Blocked on a denied sudo action; all NVMe numbers are the conservative 128 K-recordsize floor. → create the datasets and re-run `tools/nvme_tier_probe.py`.
9. **`w8a8_moe_regdirect` reads `w13_scales.shape[2]` (`kernels.py:1034`) while its only producer yields a 2-D `(E, N)` tensor (`moe.py:413`).** With `MINISGL_MOE_W8A8_REGDIRECT` defaulting to `1`, the ZAYA fp8 path looks like it raises `IndexError` on the first decode step. Unverified — no lease taken. → 5-minute run; it does not affect the granule walk (which makes no rank assumption) but it changes which fp8 arm the format matrix should treat as live.

> **[P0-2026-09-02] NEW UNKNOWNS, added by Phase 0.**
>
> 10. **Does torch accept a `hipHostGetDevicePointer` address?** P5's four legs all ran over a VMM
>     reservation, and its "host" legs were — we now know — **device memory**. So P5 validated foreign-pointer
>     plumbing over device memory **only**, and the *selected* mechanism is untested through torch.
>     → **P5b**, 0.25 d, in-image, with capture. **Blocks M1-A.** Failure returns the C++/`from_blob`
>     route (+3 d).
>     > **[M1A-2026-09-03] ANSWERED GREEN — 4/4 legs, both physical cards, all six gate conditions.**
>     > `t.data_ptr()` == the `hipHostGetDevicePointer` address exactly under `_cuda_customAllocator` +
>     > `MemPool` + `use_mem_pool`; **0 `hipMalloc` fallbacks in every leg**; free callback fires **zero**
>     > times for live *and* cached blocks; the arena survives `empty_cache()` and
>     > `torch.cuda.graph.__enter__` with the pointer stable; 24 replays → **one** sha256 matching CPU
>     > ground truth, plus 25/25 varying-input replays byte-exact. `expandable_segments` verified LIVE
>     > via `memory_snapshot().is_expandable` with the two plain legs as a differential control.
>     > Placement proven three ways with **no** reliance on a return code or a location field: VRAM delta
>     > **0.000× size**, `MemAvailable` delta 0.991–1.020×, pages CPU-read **and write** accessible by a
>     > syscall probe against a real file. Torch-level host-arena read **26.61 GB/s card 0 / 13.868 card
>     > 1** (ratio 1.92× = Gen5 x8 vs Gen4 x8), i.e. **92 % / 96 % of P1's hand-written kernel**.
>     > **The +3 d C++ route is not incurred.** Two traps this closed:
>     > `hipPointerGetAttributes` says **"Device" for the real host arena** (inverse of the Phase-0
>     > trap — unreliable in both directions, never a residency check), and `mem_info_gtt_used` stays
>     > at **0** (userptr pinning is not GTT-accounted — telemetry must use `MemAvailable`).
>     > Caveat: tested at **688 MiB**, and capture was one elementwise op over a 4 MiB view.
> 11. **Kernel-read bandwidth from host pages under realistic host DDR load.** P1 measured the shader path
>     idle only; P4 loaded the **copy engine**. Every "loaded" ceiling therefore rests on a cross-engine
>     proxy. → merge P4's synthetic-load arm into P1's kernel-read arm, 0.25 d.
> 12. **The real `granule_bytes` for the minisgl packing.** 1.33 GB is arithmetic; 1.1625 GB is the *GGUF*
>     i-quant packing. **±14 % on every tok/s figure in this document.** → §6.1's walker, once, after
>     `post_load()`. Free, part of M1-A.
>     > **[M1A-2026-09-03] MOSTLY ANSWERED — the bracket closes to −4 %, not −14 %.** Derived through the
>     > walker on a meta build of the real containers for the target shape (48 layers, E=512, top_k=10,
>     > H=2048, I=768, CT-int4 g32 sym, TP=2): **granule = 1,327,104 B** (1.266 MiB, one expert across
>     > both GEMMs, per rank), **666.0 MiB per MoE layer per rank**, **31.22 GiB/rank** total, and
>     > **1.27 GB/token node-wide** at bs=1 against the plan's 1.33 GB arithmetic. All-host T1 therefore
>     > projects **19.42 tok/s**, not 18.7. Two post-`post_load` deltas the arithmetic missed are now
>     > charged: compressed-tensors **symmetric** zeros are *synthesised* as a real `(E, G, N/pf)` int32
>     > buffer (+18 MiB/layer — resident but **not** in the granule, since every expert reads the same
>     > row), and MXFP4 widens its E8M0 scale to fp16 (+6.25 % at g=32). **Still analytic/meta-derived,
>     > not measured on the real checkpoint** — and the register-direct repack arms go through kernels
>     > `sizing.py` cannot see, so a post-`post_load()` reconciliation caller is still owed.
> 13. **Can card 1's root port be trained to Gen5?** → BIOS PCIe-gen / bifurcation / riser sweep, then
>     re-run P4's single-card arm. 1 h for a **1.75×**. See K7.
> 14. **File-backed `mmap(MAP_SHARED)` + `hipHostRegister` at scale.** Demonstrated at 4 GiB/rank; an
>     8 GiB / 2 GiB-chunk attempt spun **> 711 s of user CPU**. → **P3c**, 0.5 d. This is the only route
>     past the 62 GiB pinned ceiling other than a device tier. See §9.
>
> **Coverage gap to close opportunistically:** **card 1 was never exercised by P2, P5 or P6.** Same arch and
> driver, so identical behaviour is expected — but it is not measured, and this box has burned people on
> cross-card assumptions before. Fold a card-1 leg into P2′ and P5b; it is free.