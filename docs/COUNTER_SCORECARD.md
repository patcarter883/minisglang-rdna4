# Counter scorecard — re-measuring the perf verdicts this repo reached by inference

**Date** 2026-08-05 · **Engine** `minisgl-rdna4` @ `40f84d33` (rdna4 HEAD; supersedes `76c3eeae`, which
is its parent — HEAD is where the counter tooling actually lives) · **Kernels**
`rdna4-hip-kernels` @ `2ade7c1` · **Card** RX 9070 XT, `multi_processor_count` = **32 WGP** (64 CU;
never assert 64 — the 9070 answers 28) · **Stack** ROCm 7.14.0.

Every regime call in this repo was reached by INFERENCE — ISA instruction counts, ledgers of failed
experiments, and percentages taken against a roofline that was later corrected. Hardware counters now
work on gfx1201, so this measures them.

Reproduce: `GPU_LEASE_WEDGE_WATCH=0 gpu-lease -n 2 -- bash tools/counter_probe/scorecard/run_sweep.sh phase0_timing phase1_waitsplit phase2_kernels phase3_g2fuse`
then `python3 tools/counter_probe/scorecard/scorecard.py`.

---

## Method, and the three things that had to be right

**Counters at `profile_standard`, TIMES at `auto`, never mixed.** `profile_standard` ungates the
perfmon clock but PINS core and memory clocks to a fixed non-boost state. The int4 16384² GEMV
measures **949,906 ns pinned vs 222,643 ns at auto** — a bandwidth computed from the pinned timestamp
understates the kernel by 4.3× and reads exactly like a catastrophic regression. So bytes come from
counters (a clock-independent property of the algorithm) and time comes from a separate
`--kernel-trace` pass at auto.

**`GL2C_MISS` calibrated to bytes, not assumed.** Two streaming shapes whose working sets far exceed
the 64 MB last-level cache — and which therefore must be read from HBM exactly once — both give
**256 B per miss** (fp8 268.5 MB / 1,049,047 = 256.0 B; int4 139.0 MB / 543,041 = 254.9 B). So
HBM bytes = `GL2C_MISS × 256`.

**`FetchSize` is unusable on gfx1201.** It returns a hard **ZERO** even with the perfmon clock
ungated, despite being listed as working. So does `SQ_INSTS_SMEM`. Do not build a roofline on it.

**Roofline denominator = 706.6 GB/s** (1380 MHz mem OC), not 674 and not 644, and never a previously
*achieved* bench figure — quoting an achieved throughput as the ceiling makes every later kernel look
more "done" than it is, which is exactly how a percentage-of-roofline claim closes off real work.

**Derived percentage metrics are partly broken.** `VALUBusy` returns **115–278%** on the MoE kernels
— impossible, so its normalisation (probably a CU-count assumption) is wrong. It is usable as a
*relative* signal between kernels and NOT as an absolute. `MemUnitBusy` and `OccupancyPercent` stayed
in range everywhere and are treated as sound. `WAVE_DEP_WAIT` / `WAVE_ISSUE_WAIT` are percentages,
not cycle counts.

**Fixture gate.** Phase 1 reproduced the recorded 2026-08-05 fixture before anything downstream was
trusted: fp8 `SQ_BUSY_CYCLES` 3,460,536 vs recorded 3,460,996; int4 2,030,756 vs 2,042,555;
`SQ_WAVES` 4096 / 2048 exact.

---

## The measurement

All rows are the **DECODE band, M ≤ 32**. Counters @ `profile_standard`; times @ `auto`, **mean over
the same 4 dispatches the counters cover** — see the byte/time matching note under claim 3.

| shape | kernel | band | waves | Occ% | MemUnit% | IssueWait% | L2 hit% | HBM MB | mean ns | GB/s | **% of 706.6** | VALU/wave |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense GEMV 4096² | fp8 | cache-adjacent | 4,096 | 61.9 | 66.7 | 2.6 | 54.2 | 16.81 | 34,480 | 487.4 | **69.0** | 854 |
| dense GEMV 4096² | int4 | cache-adjacent | 2,048 | 52.1 | 56.4 | 12.3 | 56.7 | 8.66 | 16,490 | 525.4 | **74.4** | 1,391 |
| dense GEMV 16384² | fp8 | HBM-streaming | 16,384 | 78.8 | 85.7 | 0.4 | 58.8 | 268.56 | 442,134 | 607.4 | **86.0** | 3,086 |
| dense GEMV 16384² | int4 | HBM-streaming | 8,192 | 77.4 | **97.2** | 2.1 | 62.5 | 139.02 | 223,812 | 621.1 | **87.9** | 5,039 |
| bf16 shared.gate N=1 K=2048 | bf16 | launch-floor | 16 | **0.1** | 27.4 | 0.2 | 49.3 | 0.01 | 10,060 | 0.9 | **0.1** | 52 |
| bf16 shared.down N=2048 K=256 | bf16 | small-N | 64 | **1.1** | 34.1 | 0.1 | 63.0 | 1.05 | 11,480 | 91.8 | **13.0** | 253 |
| bf16 in_proj_qkvz N=6144 K=2048 | bf16 | large-N | 6,144 | 71.5 | 80.1 | 0.3 | 54.1 | 25.18 | 112,971 | 222.9 | **31.5** | 162 |
| bf16 LM head N=32768 K=2048 | bf16 | large-N | 32,768 | **87.8** | **96.9** | 0.1 | 55.6 | 134.29 | 377,274 | 355.9 | **50.4** | 162 |
| MoE gemm1 M=1 | int4 | decode | 4,096 | 35.7 | 63.1 | 11.2 | 56.7 | 8.66 | 28,200 | 307.2 | **43.5** | 862 |
| MoE gemm1 M=5 | int4 | decode | 16,384 | 44.8 | 80.4 | 16.2 | 57.8 | 34.66 | 78,351 | 442.3 | **62.6** | 938 |
| MoE gemm1 M=6 | int4 | decode | 16,384 | 45.5 | 81.2 | 16.8 | 58.9 | 34.67 | 99,521 | 348.3 | **49.3** | 1,015 |
| MoE gemm1 M=30 | int4 | decode | 16,384 | 42.9 | 81.3 | 27.6 | 75.6 | 34.91 | 233,682 | 149.4 | **21.1** | 2,918 |
| MoE gemm2 (unfused) M=1 | int4 | decode | 8,192 | 63.4 | 80.6 | 28.2 | **95.9** | 4.92 | 127,832 | 38.5 | **5.4** | 3,193 |
| MoE gemm2 (unfused) M=5/6/30 | int4 | decode | 32,768 | 69.7 | 88.2 | 28.4 | **96.2** | 19.67 | ~470,000 | ~40 | **~5.6** | 3,193 |

**Production fused MoE gemm2 (`grid.y = M`) — the served path — is tabulated under claim 2.**

---

## Priority 1 — the open one: what is `SQ_WAIT_ANY` actually waiting on?

**Prior state.** Both decode GEMVs measured stall-dominated: `SQ_WAIT_ANY` 57× `SQ_BUSY_CYCLES` (fp8)
and 49× (int4). "Stalled" established, "stalled on what" not.

**First: the 49–57× ratio is a UNIT MISMATCH and should not be quoted again.** `SQ_WAIT_ANY` is
summed **per wave**; `SQ_BUSY_CYCLES` is a **per-SQ** busy count. With many waves resident their
ratio is large by construction and says nothing about saturation. The commensurable normalisation is
`SQ_WAIT_ANY / SQ_WAVE_CYCLES`, which is **95.6% (fp8)** and **83.3% (int4)** — i.e. each wave spends
most of its resident time waiting, which is the *normal* many-wave latency-hiding regime, not evidence
of a pathology.

### The counter route does NOT exist — `WAVE_DEP_WAIT` is misnamed, and I verified it

The task assumed a `SQ_WAIT_*` family that splits memory from dependency. **It is not there.** The two
promising names are both aliases of the aggregate, which I confirmed arithmetically against my own
raw counters:

```
WAVE_DEP_WAIT   == 100 * SQ_WAIT_ANY      / SQ_WAVE_CYCLES
    fp8   computed 95.63  vs reported 95.2      int4  computed 83.29  vs reported 84.5
WAVE_ISSUE_WAIT == 100 * SQ_WAIT_INST_ANY / SQ_WAVE_CYCLES
    fp8   computed  2.78  vs reported  2.6      int4  computed 11.99  vs reported 12.4
```

So `WAVE_DEP_WAIT` is **not dependency-specific** — it is `SQ_WAIT_ANY` renormalised, the very
aggregate we were trying to break down. There is no `SQ_WAIT_CNT_VM` in gfx1201's 143-counter list.
**This counter set cannot answer the question. Do not spend a window trying.**

**What the counters DO establish** (4096², decode M=1):

| | wait/wave-cycles | instruction-fetch wait | Occ% | MemUnit% | VALU/wave |
|---|---:|---:|---:|---:|---:|
| fp8 `Fp8DenseGemvLoader` | 95.6% | 2.8% | 61.9 | 66.7 | 854 |
| int4 `Int4Fp8GemvLoader` | 83.3% | 12.0% | 52.1 | 56.4 | **1,391 (1.63×)** |
| `fillBufferAligned` (control) | 97.5% | — | 2.6 | 17.3 | — |

The 49–57× "stall ratio" is a **unit mismatch** and should not be quoted again: `SQ_WAIT_ANY` is
summed per wave, `SQ_BUSY_CYCLES` is per-SQ, so their ratio grows with occupancy by construction.
Normalised properly it is 95.6% / 83.3% — each wave spends most of its resident time waiting, which
is the *normal* many-wave latency-hiding regime, not a pathology. Only ~3% / 12% of that is
instruction-fetch, so the remainder is memory or operand waits and the counters cannot say which.

### Answer to the live design question — settled CAUSALLY, not by counters

A separate agent answered it by experiment rather than by counter naming: **varying prefetch depth at
FIXED wave count cuts per-wave stall 33% (fp8) / 41% (int4)**. A stall that shrinks when you issue
loads earlier, with occupancy held constant, is **coverable memory latency** — not a dependency
chain. **Occupancy is the cover, and the low-occupancy deep-buffer corner lost 0 of 64 cells**
(int4 median 2.48×, fp8 3.61×, bf16 1.59× slower).

My counters are consistent with that and add the boundary condition: per-wave MLP tops out near ~12
outstanding loads while occupancy delivers far more, and `WAVE_ISSUE_WAIT` is only 2.6% / 12.3% — so
there is no issue-slot contention that more waves would aggravate. **Deep-buffering loses; occupancy
wins for this kernel family.**

> **Correction to an earlier draft of this document.** I originally read `WAVE_DEP_WAIT` as
> dependency-specific and concluded "the pipes are already busy, so the deficit is instructions, not
> latency". The *prediction* (deep-buffering loses) happened to be right; the *reasoning* was wrong,
> because it rested on a misnamed counter. The int4 1.63× VALU-per-wave excess is real and still
> argues for removing dequant instructions (the RXF int8-dot path) — but it is not what binds the
> kernel, and it is not a reason to trade away occupancy.

---

## Priority 2 — verdicts that closed off work

### 1. "Dense GEMV is at its floor; the launch-count/fusion lever is dead" — **REFINED** (band-split)

The claim silently pools two regimes.

- **Large-N (the shapes that carry the time): CONFIRMED.** LM head Occ **87.8%**, MemUnit **96.9%**;
  in_proj_qkvz Occ 71.5%, MemUnit 80.1%. These are memory-unit-saturated. Nothing in tiling.
- **Small-N: NOT at a bandwidth floor — at a LAUNCH floor, with occupancy of 0.1–1.1%.**
  `shared.gate` (N=1) runs **16 waves** on a 64-CU part and reaches 0.1% of roofline; `shared.down`
  runs **64 waves** at 13.0%. Calling this "at its floor" is true only in the sense that the fixed
  per-dispatch cost dominates — it is not a bandwidth statement.
- The counters also **explain** why KSPLIT (adding waves to exactly these shapes) regressed: at
  ~4.5 µs with 16 waves the kernel is dominated by fixed cost, and extra waves add a barrier without
  removing it. And the e2e launch-count falsifications stand independently — under CUDA-graph capture
  there is no per-dispatch cost to recover.

**One real gap the counters expose:** the LM head sits at **MemUnit 96.9% but only 50.4% of roofline**.
The memory path is saturated with requests that are not achieving peak bandwidth — unlike int4
streaming, which converts MemUnit 97.2% into 87.9% of peak. That is a **~1.7× efficiency gap on the
single largest decode GEMV**, and it is an access-pattern/request-efficiency problem, not an
occupancy or launch one. See claim 7.

### 2. "MoE decode is reduction-floor bound" — **REFUTED as stated** (for the kernels measured)

The claim is a fixed ~17 µs floor plus a per-output-column cost, with padding free and the working
set cache-resident.

- **The M-scaling does not fit a floor-plus-column model.** gemm1: 28.2 / 78.4 / 99.5 / 233.7 µs at
  M = 1 / 5 / 6 / 30 while HBM bytes go 8.66 / 34.66 / 34.67 / 34.91 MB. From M=5→30 the bytes are
  **flat** and the time grows **3.0×** — the incremental cost there is neither a floor nor columns,
  it is **compute on real rows** (VALU/wave 938 → 2,918).
- **gemm1 is not floor-bound at M=1 either**: MemUnit 63.1%, reaching 43.5% of roofline, rising to
  **62.6% at M=5**. It is weight-streaming-limited in M=1→5 and compute-limited above.
- **The unfused gemm2 is L2-bandwidth bound, not reduction-floor bound**: **L2 hit 96%**,
  MemUnit 88%, **5.6–6.0% of HBM roofline**, and the **highest `WAVE_ISSUE_WAIT` anywhere (28%)**.
  It re-reads weights out of L2 roughly 25× rather than streaming them. Its cost is also
  **invariant to M across 5/6/30** (identical counters: 32,768 waves, 104,640,512 VALU insts,
  ~467 µs) because it is driven by the **padded row count P**, which saturates at `E × block_m = 512`
  once `M × top_k ≥ E`. At M=5 it computes 512 rows for 40 real ones — **92% wasted work**. Padding
  is free in gemm1; in the unfused gemm2 it is the dominant cost.

#### The production fused gemm2 — and this is the headline

The rows above are the **unfused** path. Production decode calls a separate kernel,
`moe_gemm2_gather_reduce_core` with `grid.y = M`. I copied its template out of the torch TU verbatim
(`diff`-verified empty against `2ade7c1`) and measured it. **It is not reduction-floor bound; at
served decode it is occupancy-starved by more than an order of magnitude.**

| config | grid | waves | Occ% | MemUnit% | L2 hit% | HBM MB | min ns | GB/s | % of 706.6 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **K=256 M=1 (the served decode case)** | **8 blocks** | **64** | **2.2** | 31.2 | 51.2 | 2.17 | 20,280 | 107.2 | **15.2** |
| K=256 M=5 | 40 | 320 | 12.3 | 54.2 | 55.0 | 8.71 | 35,401 | 245.9 | 34.8 |
| K=256 M=6 | 48 | 384 | 14.8 | 53.3 | 58.5 | 8.72 | 27,640 | 315.3 | 44.6 |
| K=256 M=30 | 240 | 1,920 | 54.9 | 70.8 | 97.9 | 12.03 | 78,200 | 153.9 | 21.8 |
| K=512 M=1 | 64 | 512 | 18.1 | 65.8 | 57.0 | 4.34 | 24,240 | 179.0 | 25.3 |
| K=512 M=30 | 1,920 | 15,360 | 66.0 | **95.1** | 56.6 | 130.14 | 343,403 | 379.0 | 53.6 |

`grid = (ceil(N/256), M)` with `M = 1` at bs=1 decode gives **8 workgroups on a 64-CU part — 2.2%
occupancy and 31% memory-unit busy**. The kernel is idle-by-construction at the batch size the
engine spends most of its life in. It reaches MemUnit 95.1% only at M=30 with K=512, i.e. only when
`grid.y` finally supplies enough parallelism.

Fusion itself was a large real win — 20,280 ns versus the unfused path's 123,282 ns at K=256 M=1,
about **6×** — so this is not an argument against it. But it landed the kernel in a regime the
"reduction floor" framing then declared closed. **The binding constraint at served decode is
parallelism, and `grid.y = M` is the specific thing that caps it.**

### 3. "Decode GEMV is at 81% of HBM" — **REFUTED** (the percentage is a synthetic-shape artifact)

Independently refuted by another agent the same day (fixture
`tools/_fixtures/gemv_deep_buffer_gate_diag.json`, merged `8a8bca6`): the 81% reproduces *exactly*,
but only under the original methodology — a **hardcoded 640 GB/s** `PEAK_GBs` in `perf_gemv.py`, with
back-to-back launches and **no MALL rotation**. Two compounding errors: a stale denominator and a hot
cache. Measured honestly (weights rotated past the 64 MB MALL, graph-replay timed) fp8 reads
**55–63%**; the recorded int4 "38%" is the pre-unification kernel and today's shared core reads 53.6%.

**My counters independently reproduce the same shape-dependence, which is the mechanism.** Because I
compute HBM bytes from `GL2C_MISS` rather than from an assumed working set, the numbers are not
inflated by cache hits. Bytes and time are **matched per dispatch** (both means over the same 4
dispatches) — mixing mean bytes with the *minimum* time produced an impossible 127.6% on one
cache-resident shape, which is the tell for exactly this mistake:

| shape | weights | HBM MB/dispatch | mean ns | GB/s | **% of 706.6** | Occ% | MemUnit% |
|---|---:|---:|---:|---:|---:|---:|---:|
| **N=K=2048 int4** (production) | 2.1 MB | 2.17 | 8,600 | 252.3 | **35.7** | 24.8 | 44.5 |
| **N=K=2048 fp8** (production) | 4.2 MB | 4.21 | 13,980 | 301.1 | **42.6** | 54.2 | 49.6 |
| **N=6144 K=2048 int4** (production) | 6.3 MB | 6.50 | 14,390 | 451.9 | **64.0** | 52.7 | 64.4 |
| **N=6144 K=2048 fp8** (production) | 12.6 MB | 12.62 | 25,720 | 490.8 | **69.5** | 64.8 | 60.8 |
| N=K=4096 int4 | 8.4 MB | 8.66 | 16,490 | 525.4 | 74.4 | 52.1 | 56.4 |
| N=K=16384 int4 (synthetic) | 134.2 MB | 139.02 | 223,812 | 621.1 | **87.9** | 77.4 | 97.2 |
| N=K=16384 fp8 (synthetic) | 268.4 MB | 268.56 | 442,134 | 607.4 | **86.0** | 78.8 | 85.7 |

and the bf16 loader at real serve shapes: LM head **50.4%**, in_proj_qkvz **31.5%**,
shared.down **13.0%**; MoE gemm1 **43.5 → 62.6%**.

**A GEMV only approaches the roofline at shapes far larger than anything the engine launches.** The
81% was taken in the synthetic regime and then read as a property of "the decode GEMV". At the shapes
actually served, the same kernels read **13.0–69.5%** of 706.6 GB/s — against 86–88% at 16384². That
is real, unclosed headroom, and this verdict was wrongly holding the door shut on it.

Note also that every production shape here has a weight footprint of **2–13 MB, far inside the 64 MB
MALL**, yet `GL2C_MISS × 256` still accounts for essentially the whole weight on every dispatch — the
weights are not being retained between launches. So "% of HBM roofline" is not even the right frame
for these shapes; they are dominated by per-launch streaming and occupancy, not by the HBM ceiling.

**Generalisable:** never quote a percentage-of-roofline without the shape and the cache state it was
taken at, and never let a synthetic sizing sweep stand in for the served geometry.

### 4. "Serving is overhead-bound, memory controller ≤27% at every batch size" — **REFINED** (directionally right, figure slightly low, and the inference is wrong)

**Now measured on a live serve** (Qwen3.6-35B-A3B-AWQ TP=2, `SPEC=none`, perf level **auto**,
rocprofv3 `--kernel-trace`, no `--pmc`):

| | bs=1 | bs=4 |
|---|---:|---:|
| achieved bandwidth / 706.6 GB/s | **31.5%** | **31.1%** |
| bandwidth counting **busy time only** | 44.3% | 40.6% |

So the ceiling is **~31%, not ≤27%** — directionally right, modestly understated. The byte table is
not "35B at 4 bits": only the routed experts are quantized (everything else bf16), it is a GDN hybrid
with KV in 10 of 40 layers, and experts are charged as the expected **union** over the batch rather
than `bs × top_k`. Top buckets at bs=1: GDN weights 41%, lm_head 21%, routed experts 12%.

**The inference remains refuted, and now for a sharper reason.** The step is *not* idle-dominated
(28.8% idle, below) — yet bandwidth is still only 31%, and **even counting only kernel-execution time
the GPU reaches just 44.3%**. So the deficit is **inside the kernels, not between them**. That is
exactly where the per-kernel sweep localises it: the production MoE gemm2 at **2.2% occupancy**, the
served GEMV shapes at **13.0–69.5%** of roofline. Claims 4 and 5 have been used together to argue
"overhead-bound ⇒ fuse and amortise"; measured, they do not compose that way.

### 5. "bs=1 decode is inter-kernel-gap bound, ~68% idle" — **REFUTED**, close to inverted

**Directly measured: the bs=1 decode step is 71% GPU-BUSY.**

| | bs=1 | bs=4 |
|---|---:|---:|
| wall/step (unprofiled control, median) | **10.970 ms** (p10–p90 10.8–12.0), 89.6 tok/s | 15.767 ms |
| GPU-busy/step, **union of kernel intervals** | **7.809 ms** | 12.057 ms |
| GPU-busy/step, naive sum | 9.197 ms | 14.036 ms |
| kernels/step | 1,144 | 1,436 |
| **idle** | **28.8%** | **23.5%** |

Union, not sum: kernels on different streams overlap, so summing double-counts and can exceed wall.

**Why the wall is trustworthy, which was the whole difficulty.** The marker-derived wall inside the
profiling window was 61.4 ms/step — that is the *instrument*, not the workload: rocprofv3 costs
~50 ms/step while its window is open and it lands entirely in the inter-kernel gaps. The profiled run
shows this in its own data (median gap 11.3 ms vs mean 28.0 ms; 200 in-window steps at ~61 ms against
400 out-of-window at ~11 ms). The control leg, same image, settles it at **10.970 ms** with a tight
p10–p90. Any residual error is safe-direction — inflated kernel durations would make idle look
*lower*, not higher, so 28.8% is if anything an overestimate.

**Where 68% came from.** It is a **Laguna-XS** number — a much smaller model whose tiny kernels make
gaps comparable to kernel time — and it was never a Qwen 35B measurement. The torch-profiler route
that appears to confirm it is an artifact: the repo's own analyser over
`tools/loads/qwen35b_mtp_decode.pt.trace.json.gz` reports 78.2% idle, but also **161.9 ms/step wall
and 35.3 ms/step GPU-busy** against a real 10.97 ms step — GPU-busy alone exceeds the entire real
step by 3×. A profiler-derived idle fraction on this stack manufactures its own answer and would keep
re-confirming itself on every re-profile.

**Two things established on the way that are worth as much as the number:** the plain-decode loops
carried **no ROCTx markers at all** (only `_run_spec_step` did), so a `SPEC=none` serve was simply
unprofilable; and `minisgl-rdna4:lean-prof` had gone stale and can no longer boot today's engine (a
`tail_hip_C::store_kv()` ABI change), so it is now derived from whatever `lean` currently is.

**Not established:** bs=8 — the engine will not boot at `max-running-requests=8` on 16 GB cards for
this hybrid (per-sequence recurrent state is reserved up front), so "at every batch size" rests on
two points, not a sweep. Only rank0/card0 was traced, and only `SPEC=none`.

### 6. "Occupancy is THE lever" — **CONFIRMED where waves are available; the useful output is the REGIME BOUNDARY**

Not a blanket verdict either way. Three distinct regimes, and the claim is right in one of them:

1. **Occupancy-starved → occupancy IS the lever.** The production fused MoE gemm2 at served decode
   sits at **2.2% occupancy / 15.2% of roofline**; MoE gemm1 at 35–45% occupancy with MemUnit 63–81%.
   Independently, the low-occupancy deep-buffer corner **lost 0 of 64 cells** (int4 median 2.48×,
   fp8 3.61×, bf16 1.59× slower), and prefetch depth at fixed wave count cuts stall 33–41% — i.e. the
   stalls are coverable memory latency and waves are what covers them. Per-wave MLP saturates near
   ~12 outstanding loads while occupancy delivers far more.
2. **Already saturated → nothing to buy.** LM head Occ 87.8% / MemUnit 96.9%; int4 streaming
   Occ 77.4% / MemUnit 97.2%. Raising occupancy here is unrecoverable effort.
3. **Structurally capped → occupancy is not available at all.** `shared.gate` has only 16 waves to
   have because N=1. The direct fix was measured and **regressed** (KSPLIT).

And the opposite failure mode exists alongside it: **39 dense tiles silently spill up to 632 B/lane
while `tile_select.h` prices them as free** — there, registers rather than waves are the mispriced
resource. Both are true; the error was ever treating "occupancy" as a single global lever rather than
asking, per kernel, *which resource is actually scarce.*

The gfx1201 occupancy law (established twice independently from 105 VGPR counts, fixture
`tools/_fixtures/gfx1201_occupancy_law_from_gemv.json`) is
`waves(v) = min(16, 1536 // (ceil(v/24)*24))` — register file [1536,1560), granule 24, cap 16.

---

## Priority 3

### 7. "LM-head GEMV is the serve lever (5.4× over rocBLAS)" — **CONFIRMED as a win, and it is NOT finished**

It is the largest single decode GEMV measured (134.3 MB/step, 32,768 waves) and it is well-optimised
— Occ 87.8%, MemUnit 96.9%. But it converts that into only **50.4% of roofline**, where int4
streaming converts a comparable MemUnit 97.2% into **87.9%**. That is a **~1.7× residual gap on the
biggest decode GEMV**, and the counters localise it: not occupancy (87.8%), not issue contention
(0.1%), but **per-request memory efficiency**. This is live headroom the "it's the serve lever, we
won it" framing has been treating as spent.

### 10. "Fused MoE decode occupancy collapse — 33× slower at bs=1" — **mechanism CONFIRMED by analogy**

I did not instrument that kernel, but the scorecard contains a direct quantitative analogue.
Collapsing a decode GEMV's grid to a handful of blocks is measured here at
**`shared.gate`: 16 waves → 0.1% occupancy → 0.1% of roofline (0.9 GB/s)**, versus 87.8% / 50.4% for
the same loader family at large N. A one-block-per-token fusion produces exactly that grid collapse,
and a 2–3 order-of-magnitude loss is entirely consistent with a 33× slowdown. Occupancy really is the
explanation *in this specific case* — which is not in tension with claim 6, because here the fusion
**destroyed** parallelism that existed, rather than failing to add parallelism that never existed.

### 8 / 9 — **NOT MEASURED**


"CCA prefill is attention-kernel-bound" and "DFlash propose is weight-streaming bound" were not
reached. Both are also **prefill-band** claims (M ≥ 64) and nothing in this scorecard bears on them —
the sweep is decode-only by construction.

---

## Ranked: refuted verdicts by optimisation headroom wrongly closed off

**1. "MoE decode is reduction-floor bound" (claim 2) — by far the largest.**
It declared the served model's dominant block structurally stuck. The production fused gemm2 at
served decode is measured at **2.2% occupancy, 31% memory-unit busy, 15.2% of roofline** — it runs
**8 workgroups on a 64-CU part** because `grid = (ceil(N/256), M)` and `M=1`. That is not a floor,
it is an idle GPU, and the cap is one identified term. "Reduction floor" pointed every subsequent
experiment at the warp-reduce, which is why the one lever tried (lane-owns-column) failed — it
attacked the wrong term. Headroom implied: the same kernel reaches MemUnit 95.1% once `grid.y`
supplies parallelism, so this is a multiple, not a few percent, on the MoE decode block.

**2. "Decode GEMV is at 81% of HBM" (claim 3).**
Reproduces only against a hardcoded 640 GB/s constant with a hot cache. Honestly measured, fp8 reads
**55–63%**; my counters put the *served* shapes at **13.0–69.5%** of 706.6 GB/s, with only the
synthetic 16384² reaching 86–88%. A whole class of decode GEMV work was closed off on a number taken
at a geometry the engine never launches.

**3. "Occupancy is THE lever" (claim 6) — mis-scoped rather than wrong.**
Right in the occupancy-starved regime (which includes the MoE gemm2 above), wasted effort in the
already-saturated one (LM head MemUnit 96.9%), and inapplicable where N caps waves structurally. The
cost was spending it as a global rule instead of asking per kernel which resource is scarce — and the
mirror-image failure (39 dense tiles spilling 632 B/lane, priced as free) went unlooked-for.

**4. "bs=1 decode is ~68% idle" (claim 5) — refuted, close to inverted.**
Measured on a live serve: the step is **71% GPU-BUSY** (7.809 ms of a 10.970 ms wall; **28.8% idle**,
23.5% at bs=4). The 68% is a **Laguna-XS** number that was never a Qwen 35B measurement, and the
torch-profiler route that appears to confirm it is an artifact that manufactures its own answer. This
sent gap-closing and megakernel work after a bubble roughly **2.4× larger than the one that exists** —
and, worse, pointed *between* the kernels when the deficit is inside them.

**5. "Serving is overhead-bound / umc ≤27%" (claim 4) — directionally right, and the inference is the problem.**
Measured 31.5% (bs=1) / 31.1% (bs=4), so the figure is modestly understated rather than wrong. The
damage is in how it was used: paired with claim 5 to argue "overhead-bound ⇒ fuse and amortise". The
serve says otherwise — the step is not idle-dominated, and **even counting only kernel-execution time
the GPU reaches just 44.3%**. The slack is *inside* the kernels (MoE gemm2 at 2.2% occupancy; served
GEMV shapes at 13.0–69.5% of roofline), which is precisely where fusion does not help.

**6. "LM-head GEMV is the serve lever" (claim 7) — the win is real, the closure is not.**
MemUnit 96.9% converted into only 50.4% of roofline, where int4 streaming converts a comparable 97.2%
into 87.9%. ~1.7× of per-request memory efficiency remains on the largest single decode GEMV.

**7. "Dense GEMV is at its floor" (claim 1) — least headroom lost.**
Correct for large-N; only the small-N framing was loose, and the launch-count lever is independently
dead under graph capture.

### The one concrete next move this implies

The #1 finding names its own fix. `moe_gemm2_gather_reduce_core` launches
`grid = (ceil(N/per_block), M)`, so at bs=1 the second grid dimension contributes **1** and the
kernel gets 8 workgroups. Nothing about the maths requires that: the reduction over `K` is a
per-output-row sum, so it admits **exactly the split-K treatment that already worked on attention
decode** — add a `k_split` grid dimension, have each split accumulate a partial, and reduce. That
change took `flash_decode_paged` from 16 CTAs to 1024 at B=1 and bought 12.9×. The MoE gemm2 is in
the same shape of hole (8 CTAs, 2.2% occupancy) and the same tool fits.

Two cautions carried over from that work: the split must be **shape-derived** so the grid stays
capture-constant under CUDA graphs, and `num_splits == 1` must fall back to the single-pass path so
short/large-M cases stay byte-identical. Measure it **sampled and end-to-end on the serve**, not as
an isolated microbench — this scorecard is full of isolated numbers that did not transfer.

### Confirmed, and worth as much as the refutations
- **Fusion of the MoE gemm2 was a genuine ~6× win** (20,280 ns vs 123,282 ns at K=256 M=1).
- **Padding really is free in gemm1** — but it is the dominant cost in the *unfused* gemm2 (92% wasted
  rows), which is precisely why the fused kernel exists.
- **Deep-buffering loses and occupancy wins for the GEMV family** — 0 of 64 cells, settled causally.
- **The int4 dequant excess is real**: 1,391 VALU instructions per wave vs fp8's 854 (1.63×). It is
  not what binds the kernel, but it is a true instruction-count cost and the RXF int8-dot path is
  still the way to remove it.

---

## Open / not done

- **Hardware COUNTERS on a live serve** (as opposed to the `--kernel-trace` timing now done for
  claims 4 and 5). Blocked by a real constraint, not skipped: the serve image is ROCm 7.2.1 where
  `--pmc` hangs, counters need 7.14, and a `.so` built in one image will not load in another —
  failing *silently as a 0%-GPU hang*. Doing it properly means rebuilding the whole kernel set
  against 7.14 torch. This would let the in-kernel deficit found at claim 4 be attributed per kernel
  on the served path rather than inferred from the isolated sweep.
- **bs ≥ 8** — the engine will not boot at `max-running-requests=8` on 16 GB cards for this hybrid
  (per-sequence recurrent state is reserved up front), so "at every batch size" rests on bs=1 and
  bs=4. Only rank0/card0 was traced, and only `SPEC=none`.
- **Prefill band (M ≥ 64)** entirely — claims 8 and 9 live there.
