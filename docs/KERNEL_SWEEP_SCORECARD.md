# Per-kernel sweep of the served path — what actually runs, and where the deficits are

**Date** 2026-08-06 · **Engine** `minisgl-rdna4` @ `4d719ad6` · **Kernels** `rdna4-hip-kernels` @ `8a8bca6`
· **Image** `minisgl-rdna4:ksweep-prof` (built from BOTH clean worktrees; see "the image trap")
· **Cards** RX 9070 XT (64 CU) + RX 9070 (56 CU) · **Roofline** 706.6 GB/s · **MALL** 64 MB

Successor to `docs/COUNTER_SCORECARD.md`, which measured isolated shapes. This one profiles the
**live serve**: Qwen3.6-35B-A3B-AWQ TP=2 `SPEC=none` at M = 1 / 5 / 6 and at prefill.

## Which route each number came from

| route | what it gives | conditions |
|---|---|---|
| **TRACE** | per-kernel share of step, dispatch count, **grid/block/VGPR/scratch/LDS per dispatch** | live serve, rocprofv3 `--kernel-trace --marker-trace`, perf level **auto**, no counters |
| **BASE** | the TRUE wall/step | same image, **no profiler** — rocprofv3 costs ~50 ms/step inside a window |
| **STATIC** | VGPR/scratch/LDS for all 5,569 shipped kernels | code-object `NT_AMDGPU_METADATA`, no GPU |
| **ISO** | isolated kernel time under 3 cache states | same image + toolchain, auto clocks |

`--pmc` still cannot run against the serve (7.2.1 hangs on it, counters need 7.14, and a `.so` built
in one image will not load in the other). **It was not needed for occupancy**: the kernel trace
carries `Grid_Size`, `Workgroup_Size`, `VGPR_Count` and `Scratch_Size` per dispatch, so workgroup
count, wave count, spills and register-limited occupancy are *measured on the served path*.

Occupancy model (gfx1201: 32 WGP × 4 SIMD = 128 SIMDs × 16 slots = 2048; VGPR file 1536, granule 24):

```
waves_per_simd(v) = min(16, 1536 // (ceil(v/24)*24))
occ_upper%        = 100 * min(waves_launched, 128*waves_per_simd) / 2048
```

Calibrated against the five hardware `OccupancyPercent` readings in `COUNTER_SCORECARD.md`: it is a
consistent **upper bound**, read 0.71–0.88× by the counter, which time-averages ramp-up and tail.

---

## Three measurement faults found first — every one would have produced a confident wrong answer

**1. The image matched neither commit.** `minisgl-rdna4:post-tile` (and `:lean`) predate the engine's
producer act-quant wiring; the serve dies at boot with
`mmq_fp8_moe_gemm1_silu() got an unexpected keyword argument 'x_fp8'`. The image had to be rebuilt
from both clean worktrees. An independent static diff confirmed `fp8_wmma` in the image differed from
`8a8bca6` by 725 changed lines across three headers.

**2. Under TP=2, one rank's trace silently truncates — and WHICH rank is nondeterministic.**

| arm | Agent 1 (9070 XT) | Agent 2 (9070) |
|---|---|---|
| normal_loop | 998,740 dispatches / **98.58 s** | 59,048 / **3.28 s** ← dead |
| overlap_loop | 313,205 / **26.49 s** ← dead | 681,865 / **95.67 s** |

Both rank processes inherit the rocprofv3 `LD_PRELOAD` and write the same output. Reading per-kernel
shares off the truncated agent understates its busy time and manufactures a "this config does less
GPU work" result — which is exactly what the overlap arm looked like before this was caught. The
parser now prints a coverage line per agent and marks the truncated one **DO NOT USE**.
**Consequence: the AR rank-skew analysis is not possible from these traces** (pairing the Nth
collective across ranks needs both ranks); it computed a nonsense 3.2 s "skew" from truncated data.

**3. `gap_after` measures the instrument, not the workload.** Summed per step it reaches 12.8 ms
inside a step whose true wall is 17.1 ms — because rocprofv3's ~50 ms/step lands entirely in the
inter-kernel gaps. **The starvation-by-downstream-gap signal is unusable from a profiled trace.**

Minor: 6 of ~2.9M CSV rows are torn by concurrent rank writes (skipped and counted); and
`(anonymous namespace)::` is part of a kernel's *name*, so naive signature-trimming rendered
`moe_align`, `moe_topk_softmax`, `rms_norm_add` and the paged flash kernels as **blank rows**.

---

## Coverage — what the served path actually dispatches

**The engage ledger has blind spots and must not be used alone.** `dense_gemm`, `custom_ar` and
`swiglu_hip` contain **no `engaged()` call sites at all**, so they can never appear in it. The trace
is the ground truth.

Qwen ledger (13 entries): `attn_decode.flash_decode_paged_fp8`, `fp8_wmma.dense_bf16_gemv[gdn_proj]`,
`fp8_wmma.mmq_fp8_moe_gemm1_silu(gemv+prequant)`, `.mmq_fp8_moe_gemm2_gather_reduce`,
`.mmq_fp8_moe_gemm_scatter`, `gdn_hip.gdn_decode_conv_gated_replay`, `moe_hip.moe_route_align`,
`tail_hip.{rms_norm, rms_norm_add, rms_norm_add_quant, rope, silu_and_mul, store_kv}`.

Plus, from the trace only: `custom_ar::one_shot_ar_vec_kernel`, `dense_gemm::dense_gemm_pipe_kernel`
and `dense_gemm::dense_gemm_rd_kernel` (**prefill only**), `ncclDevKernel_Generic_4`, and rocBLAS
Tensile `Cijk_…MT128x128x32` (prefill only).

**Dead on the served path:**
- **`swiglu_hip` — entirely dead.** Its one op `fused_swiglu` has zero references in
  `python/minisgl`; only `tools/` benches import it. It ships in the image and never runs.
- **`mla_hip`, `zaya_cca`, `attn_hip`** — absent for Qwen (they belong to GLM/ZAYA/non-paged paths).
- **`dense_gemm` is decode-dead** — it appears only in prefill, at 3.2% of prefill busy.
- Registered-but-uncalled op families: `fp8_wmma` bf16-MoE (`moe_bf16_gemm*`, 6 ops),
  `mmq_regdirect_w4a16_moe_gemv*`, `dense_gemm.{dense_gemm_out, *_sk, split_k_*}`,
  `attn_decode.flash_decode` (non-paged), `zaya_cca.{cca_decode_qk_b128x2, cca_decode_qk_fp4}`.

---

## Step budget — busy vs the TRUE wall

Rank 0 (9070 XT), Qwen, normal_loop. Busy is the **union** of kernel intervals (not the sum — kernels
on different streams overlap and summing can exceed the wall).

| batch | true wall/step (BASE) | busy/step (TRACE) | **busy %** | idle % | kernels/step |
|---|---:|---:|---:|---:|---:|
| bs=1 | 10.775 ms | 5.441 ms | **50.5%** | 49.5% | 807 |
| bs=5 | 17.119 ms | 13.989 ms | **81.7%** | 18.3% | 1,398 |
| bs=6 | 17.069 ms | 13.946 ms | **81.7%** | 18.3% | 1,397 |

Throughput: bs=1 **91.2 tok/s**, bs=5 **283.5**, bs=6 **344.6**. Prefill TTFT median **0.500 s**
(~2.5k-token unique prompts, so no radix reuse).

This refines `COUNTER_SCORECARD.md`'s "bs=1 is 71% busy" (taken at CONC=4 on the older kernel
package): at CONC=6 with the producer act-quant fusion live, **bs=1 is only 50.5% busy, but bs=5/6
are 82%**. Batch, not the loop, is what fills the machine.

---

## THE HEADLINE — the isolated-vs-in-serve gap is a CACHE-RESIDENCY ARTEFACT, and the elementwise-eviction hypothesis is refuted at decode

Byte accounting per step, rank 0, from the trace (work-items × 2 B × read+write; the assumption is
stated, and a 4 B accounting doubles both sides and changes no ratio):

| window | GEMM MB/step | elementwise+other MB/step | elem as % of GEMM | **bytes between two GEMMs** | GEMM stream vs 64 MB MALL |
|---|---:|---:|---:|---:|---:|
| bs=1 | 72.0 | 1.0 | **1.4%** | **2.9 KB** | **1.12×** |
| bs=5 | 164.9 | 8.1 | 4.9% | 14.9 KB | 2.58× |
| bs=6 | 164.9 | 8.2 | 5.0% | 15.1 KB | 2.58× |
| prefill | 560.9 | 364.0 | **64.9%** | **747 KB** | 8.76× |

**At decode the elementwise/norm chain moves 2.9–15.1 KB between two GEMM dispatches, against a
64 MB last-level cache. It cannot evict a weight working set.** The hypothesis that ~2,800 trivial
elementwise dispatches flush the weights between GEMMs is refuted by three orders of magnitude.

**The real mechanism is simpler and it is not an eviction at all: the GEMM weight stream is already
1.12–2.58× the MALL every single step.** The weights were never going to be resident, whatever ran
between them. So the *isolated* number is the artefact — a microbench that re-reads one small weight
tensor back-to-back measures a cache-hit rate the serve can never have. This is consistent with
`COUNTER_SCORECARD.md` claim 3 (the 81% was a hot-cache, 640 GB/s-denominator artefact) and supplies
the missing mechanism for it.

**This also settles the fusion question the hypothesis would have reopened.** Fusing elementwise ops
to save *cache traffic* is not worth doing at decode: the traffic being saved is 1.4–5.0% of the
GEMM traffic, and it is not displacing anything.

**At PREFILL the picture inverts and the hypothesis is live**: elementwise+other is **64.9%** of GEMM
traffic and 747 KB flows between consecutive GEMMs. If eviction-driven fusion is ever worth doing on
this engine, prefill is where to look — not decode.

---

## Suspect `moe_align` — REFUTED as an induced cost, with the grid that proves it

The hypothesis was that `moe_align`'s padded row count `P` sets the MoE gemm2 grid. **It does not.**

| batch | `moe_align` | `moe_gemm2_gather_reduce_core` grid | workgroups | occ | real rows (M×top_k) |
|---|---|---|---:|---:|---:|
| bs=1 | route_align, **1 WG, 0.4%** | **not dispatched at all** | — | — | 8 |
| bs=5 | 1 WG, 0.4% | `(2048, 6, 1)` blk 256 | 48 | 18.8% | 40 |
| bs=6 | 1 WG, 0.4% | `(2048, 6, 1)` blk 256 | 48 | 18.8% | 48 |

`grid.y` is **6 = the graph-capture batch bucket**, i.e. `grid = (ceil(N/256), M)` exactly as
documented — **M, not `P`**. `moe_align` is itself maximally starved (1 workgroup, 0.4% occupancy)
but it is 0.78% of the step and it does not choose anyone else's geometry. Padding waste is not the
mechanism here.

**And a correction to the finding that motivated this whole sweep:** the fused gemm2 measured at
*8 workgroups / 2.2% occupancy at M=1* was measured **in isolation**. The served path **does not
dispatch it at M=1 at all** — at bs=1 the MoE down-projection runs `w4a8_tile::moe_gemm_tiled_kernel`
at **1024 workgroups / 100% occupancy**. The fused gemm2 appears only at M ≥ 5, at **48 workgroups /
18.8% occupancy**. It is still launch-starved and still worth the split-K fix in flight — but the
severity is 18.8%, not 2.2%, and it is absent from the batch size "the engine spends most of its life
in". (Kernel untouched here, as instructed.)

---

## `normal_loop` vs `overlap_loop` — REFUTED as a lever; the synchronous loop is FASTER

`--gdn-radix` (default on for this GDN hybrid) forces the synchronous `normal_loop`; `--no-gdn-radix`
lets the zero-sync `overlap_loop` run. Self-verifying: the ROCTx range name recorded 800/800
`normal_step` in one arm and 800/800 `overlap_step` in the other. `--num-pages 3072` pinned so the KV
pool is a constant of the experiment. **Unprofiled BASE legs — no instrument in the loop:**

| phase | normal_loop | overlap_loop | overlap vs normal |
|---|---|---|---:|
| decode bs=1 | 10.775 ms · 91.2 tok/s | 10.969 ms · 89.2 tok/s | **−2.2%** |
| decode bs=5 | 17.119 ms · 283.5 tok/s | 18.021 ms · 271.6 tok/s | **−4.2%** |
| decode bs=6 | 17.069 ms · 344.6 tok/s | 17.915 ms · 326.1 tok/s | **−5.4%** |
| prefill TTFT | 0.4997 s | 0.4842 s | +3.1% (faster) |

**Overlap scheduling is a 2–5% decode LOSS on this model**, and buys ~3% cold TTFT. The synchronous
loop that `gdn_radix` forces is not costing throughput — so "the served loop is `normal_loop`" is not
an explanation for anything, and recurrent-radix's prefix reuse is not being paid for in decode
speed. Do not spend a window on the loop.

---

## Ranked deficits — `share_of_busy × deficit`, bs=6 (rank 0)

`deficit` = worst normalised shortfall among the flags that fired (launch starvation against 64 CUs,
occupancy against 25%, spill against the 632 B/lane figure in the standing lead). Geometry is the
**time-dominant** grid, not the worst one seen — several kernels run 5–53 distinct grids per step.

| # | score | %busy | ms/step | n/step | WG | occ% | VGPR | scratch | flag | kernel |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 0.220 | 25.19 | 3.514 | 81 | **8** | **3.1** | 32 | 0 | occ | `custom_ar::one_shot_ar_vec_kernel` |
| 2 | 0.033 | 3.80 | 0.530 | 40 | **8** | **3.1** | 32 | 0 | occ | `gemv_decode_core<Bf16GemvLoader>` (small-N arm) |
| 3 | 0.023 | 2.51 | 0.350 | 211 | 12 | 2.3 | 32 | 0 | occ | `at::native::elementwise_kernel_manual_unroll<128,8>` |
| 4 | 0.022 | 8.85 | 1.234 | 40 | **48** | **18.8** | 80 | 0 | occ | `moe_gemm2_gather_reduce_core` ← split-K in flight |
| 5 | 0.014 | 5.75 | 0.802 | 30 | **48** | **18.8** | 176 | 0 | occ | `gdn_decode_conv_gated_kernel` |
| 6 | 0.013 | 1.27 | 0.177 | 1 | 2 | 0.8 | 256 | **776 B** | spill | `ncclDevKernel_Generic_4` (177 µs/dispatch) |
| 7 | 0.012 | 1.33 | 0.186 | 40 | 6 | 2.3 | 32 | 0 | occ | `anon::moe_topk_softmax_kernel` |
| 8 | 0.008 | 0.78 | 0.108 | 40 | **1** | **0.4** | 16 | 0 | occ | `anon::moe_align_kernel` |
| 9 | 0.007 | 0.80 | 0.112 | 40 | 6 | 2.3 | 40 | 0 | occ | `anon::rms_norm_add_kernel<bf16,true>` |

`custom_ar` is **#1 by a factor of 6.6×** — 25% of busy at 8 workgroups. Per instruction it is **not
an optimisation target**: the vectorised AR is already the fastest TP=2 path on this hardware and
every alternative goes through RCCL. Recorded as an observation, not a backlog item.

### Prefill (the band `COUNTER_SCORECARD.md` never reached)

| %busy | ms/step | WG | occ% | VGPR | kernel |
|---:|---:|---:|---:|---:|---|
| **48.48** | 48.740 | 2624 | **50.0** | **184** | `moe_gemm1_silu_ashuffle_kernel` — register-capped, not starved |
| 25.47 | 25.600 | 8 | 3.1 | 32 | `custom_ar::one_shot_ar_vec_kernel` |
| 8.59 | 8.633 | 20992 | 100.0 | 40 | `moe_gemm_tiled_ashuffle_kernel` — clean |
| 5.52 | 5.550 | **16** | **6.2** | 184 | `gdn_prefill_wmma_kernel` — starved AND register-capped |
| 3.01 | 3.027 | **16** | **6.2** | 40 | `causal_conv1d_fwd_kernel` — starved |
| 2.78 | 2.790 | 624 | 31.2 | 256 | rocBLAS Tensile `Cijk_…MT128x128x32` |
| 2.53 | 2.547 | 832 | 31.2 | 248 | `anon::flash_prefill_paged_fp8_kernel` |
| 2.11 | 2.125 | 32 | 12.5 | 160 | `dense_gemm_pipe_kernel<64,2,64,1,false,bf16>` |
| 1.04 | 1.042 | 52 | 20.3 | 72 | `dense_gemm_rd_kernel<64,false,bf16>` |

**The single biggest prefill kernel is capped at 50% occupancy by its 184 VGPRs** (184 → 8 waves/SIMD
by the granule law), not by launch size. That is a register-pressure lever, and it is the largest
untouched item in this document.

---

## The spill lead, rescoped

The standing lead — *"39 dense tiles silently spill up to 632 B/lane and `tile_select.h` prices them
as free"* — is **confirmed in kind, understated in magnitude, and largely irrelevant on the served
path.** Static sweep of all 5,569 shipped kernels
(`tools/_fixtures/kernel_static_resources.csv`):

- **717 kernels carry scratch; only 343 have true `vgpr_spill_count > 0`.** The other 374 (the flat
  528 B / 272 B cohort in `gdn_hip`, `sampler_hip`, `gemv_decode_core`) spill **zero registers** at
  25–84 VGPRs — that is a non-promoted private array, not register pressure. Do not "fix" those.
- The **632 B is exactly reproduced** and named: `w4a8_tile::mmq_fp8_gemm_wmma_tiled_tuned_kernel<T,
  BM=512, BN=256, NWARPS=32, g=32, WN=1>`, 192 VGPR, 189 spilled. That family has **134** spilling
  instantiations, not 39.
- **The worst spiller is not in `fp8_wmma` at all**: `dense_gemm::dense_gemm_pipe_kernel` spills in
  **64 of 106** instantiations, max **2676 B/lane with 1376 VGPRs spilled** — 4.2× the quoted figure.
- `tile_select.h` genuinely has **no register or scratch term** in its scoring lattice. Structurally
  confirmed.
- **But**: the only `dense_gemm_pipe_kernel` instantiation the serve dispatches is
  `<64,2,64,1,false,bf16>` at **160 VGPR and scratch 0**, in prefill, at 2.1% of prefill busy. **The
  catastrophic spillers are not on the served path.** The only spilling kernel that reaches the
  decode step is `ncclDevKernel_Generic_4` (776 B, 1.27% of busy).

405 kernels are pinned at 5 waves/SIMD by the 256-VGPR ceiling; 601 sit above 192 VGPR (< 8 waves).

---

## Confirmed CLEAN — do not re-profile these

Material share, no flag fired, on the served path:

| kernel | band | WG | occ | note |
|---|---|---:|---:|---|
| `gemv_decode_core<Bf16GemvLoader>` (main arm) | decode, 41.5% of busy | 384 | **100%** | 8 grids/step; only 35% of its time launch-starved, and that is the structurally-capped small-N tail |
| `w4a8_tile::moe_gemm_tiled_kernel` | decode bs=1, 10.7% | 1024 | **100%** | this is what runs at M=1, not the fused gemm2 |
| `gemv_decode_core<Int4Fp8GemvLoader>` | decode, 10.8% | 384 | 75% | |
| `w4a8_fp8_moe::moe_compute_act_fp8_kernel` | decode, 4.4% | 768 | **100%** | the newly-wired producer act-quant |
| `moe_gemm_tiled_ashuffle_kernel` | prefill, 8.6% | 20992 | **100%** | |
| `w4a8_fp8_moe::moe_gather_reduce_kernel` | prefill, 0.9% | 6656 | **100%** | |

---

## Reproduce

```
GPU_LEASE_WEDGE_WATCH=0 gpu-lease -n 2 -- bash tools/counter_probe/ksweep/run_all.sh
python3 tools/counter_probe/ksweep/rank_report.py --results-dir <results> --tag qwen-normal
```

Fixtures: `tools/_fixtures/kernel_static_resources.csv` (5,569 kernels),
`tools/counter_probe/results/ksweep/*.kernels.csv`, `*.analysis.json`, `*.phases.json`.
