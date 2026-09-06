# CPU MoE offload — can the hybrid be additive?

Worktree `minisgl-rdna4-cpumoe`, branch `feat/cpu-moe-offload`, base `b1f52659`.
Box: Ryzen 7 7800X3D (8 physical cores / 16 hw threads, 96 MB V-cache), DDR5 dual channel,
91 GiB RAM. Model: Qwen3.8-Flash-Next-NVFP4 — 48 layers, 512 experts, top-10, hidden 2560,
moe_intermediate 640, 4,915,200 weights/expert.

Every number below is tagged **[M]** measured, **[D]** derived by arithmetic from measured
numbers, or **[P]** projected through a model. Nothing untagged is a number.

---

## VERDICT

**YES.** An int8-activation AVX-512-VNNI expert core reaches **27.38 GB/s on ONE physical core**
and **44.64 GB/s on two** [M], against the fp32 core's 8.78 / 17.02 [M] in the same paired window.
That is **3.12x per core**, and it means the CPU MoE tier now fits inside the ~5 physical cores a
live TP=2 serve leaves free — where the fp32 core needed 6-16 threads and therefore substituted
for the engine instead of supplementing it.

The win is **core occupancy, not bandwidth.** Both cores end at the same ~56 GB/s DDR wall [M];
VNNI simply gets there at 3 threads where fp32 needs 8-16. DDR and PCIe remain **zero-sum** — a
co-load taking ~93% of the bus halved both kernels [M]. CPU compute never adds bandwidth; it
removes the 12.4-14.5 GB/s PCIe cap on the share it takes.

Two caveats stated up front, neither of which changes the sign:

1. **The 57.2 GB/s @ 1T probe figure that motivated this work was an artifact.** It is the
   *16 MB cache-resident* line of `cpu_moe_ceiling_bench.c mode=gemv`; the same probe's
   DDR-resident 1T figure is 32.2-34.1 GB/s [M], and a real full layer reaches 27.38 [M].
   The honest single-core number is 27.38, not 57.2. It is still enough.
2. **int8 activations cost 8.3e-03 rel_rms** against the fp32 core's 2.4e-07 [M] — ~34,000x.
   That is irreducible for int8, not a bug. It is also **4.4-4.9x more accurate than the
   per-token fp8 activations the GPU path serves today** [M]. See §3.

---

## 1. Per-core throughput — fp32 vs int8-VNNI

### 1.1 The headline table

**[M]** Paired and interleaved (the two policies alternate *within* each rep, so the comparison
survives box drift even where the absolute level does not). 4 reps. Real checkpoint bytes from
`/model`, full layer = 10 experts x (gate, up, SiLU-mul, down), real routing weights, top-10.
Table 5.27 GiB DDR-resident (2048 experts, **56x the 96 MB V-cache**), route redrawn every
iteration. Threads pinned to **physical** cores 0..T-1. GB/s = weight bytes actually consumed
(0.5625 B/weight, e4m3 layout, 2,764,800 B/expert).

**Box: QUIET** — 0.43-2.25 of 16 hw threads busy, loadavg 4.3-5.9 (loadavg is D-state-inflated;
the CPU-busy figure is the real one). A concurrent graph-capture workflow ran through parts of the
session; every table says which window it came from.

| threads | fp32 GB/s (med / p5) | VNNI GB/s (med / p5) | VNNI GB/s **per core** | VNNI ms/layer (med / p5) | per-core speedup |
|--------:|---------------------:|---------------------:|-----------------------:|-------------------------:|-----------------:|
|  1 |  8.78 /  9.09 | **27.38** / 32.04 | 27.38 | 1.0096 / 0.8628 | **3.12x** |
|  2 | 17.02 / 18.02 | **44.64** / 51.19 | 22.32 | 0.6211 / 0.5401 | **2.62x** |
|  3 | 24.96 / 26.77 |   54.00 / 60.04 | 18.00 | 0.5121 / 0.4605 | 2.16x |
|  4 | 33.64 / 35.51 |   53.50 / 63.23 | 13.38 | 0.5174 / 0.4373 | 1.59x |
|  6 | 46.39 / 51.94 |   56.16 / 64.55 |  9.36 | 0.4929 / 0.4283 | 1.21x |
|  8 | 42.71 / 58.94 |   56.72 / 64.20 |  7.09 | 0.4880 / 0.4307 | 1.33x |
| 16 | 55.29 / 60.09 |   53.32 / 63.99 |  3.33 | 0.5196 / 0.4321 | 0.96x |

**VNNI saturates DDR at 3 threads. fp32 needs 8-16 to reach the same place.** That is the whole
finding.

### 1.2 The fp32 baseline was NOT contaminated

The original 8.47 GB/s @1T from `RESULTS_KERNEL` was re-measured on a quiet box and **confirmed**
[M]: 8.95 med / 10.76 p5 unpinned, 10.32 med / 10.46 p5 pinned to core 0. The whole 16-thread
curve reproduces (56.33 vs the recorded 55.46 at T=16). Per-core efficiency degrades monotonically
with thread count — 8.95 / 8.52 / 8.34 / 8.60 / 7.87 / 3.52 GB/s-per-thread at T = 1/2/3/4/6/16 [M]
— which is the shape of a compute-bound kernel hitting a bus, not a bandwidth-bound one scaling.

### 1.3 Why fp32 cannot be fixed — the architectural ceiling

**[M]** `decode_microbench`, L1-resident, 1 thread, core 0 @4.95 GHz, 3 reps. GB/s restated at the
checkpoint-native 0.5625 B/weight:

| variant | Gw/s | weights/cycle | GB/s @5 GHz |
|---|---:|---:|---:|
| A  `cvtepu8_epi32 + vpermps` (the shipped fp32 decode) | 24.8-25.1 | 5.0 | 14.1 |
| C  `vpshufb -> int8 -> vpdpbusd` (VNNI) | 148-152 | 30.3 | 85.0 |
| E  fp32 `vfmadd` roofline, **decode deleted entirely** | 35.9 | 7.3 | **20.4** |

**E is a hard architectural ceiling.** 20.4 GB/s per core at 5 GHz (16.6 at 4.07) is the most any
*fp32-activation* formulation can do on this machine even if dequantization were free. The full
fp32 layer reaches 10.46 = 51% of it [D], so at most ~1.35x remains in that formulation — nowhere
near the 3x needed. C beats E by 4.1x for an arithmetic reason, not an implementation one:
**VPDPBUSD retires 64 weights per instruction** (4 int8 MACs x 16 lanes) where VFMADD retires 16,
**and int8 weights need no int→float conversion at all**.

Two fp32 optimizations were tried and measured as **NO-WIN**, and are recorded so they are not
re-proposed: row blocking RB=1/2/4 (8.5-9.0 GB/s for all three widths [M]) and an
E2M1→fp16-via-high-byte-table decode (variant B, measured *slower* than A [M]).

### 1.4 `taskset` matters, in three separate ways — all [M]

1. **Physical-core pinning collapses the spread** rather than raising the mean: fp32 T1 p5 =
   10.30 / 10.33 / 10.36 / 10.46 across reps pinned, vs 8.20-10.41 unpinned. It removes the median
   blow-ups entirely.
2. **WHICH core matters more than pinning per se.** Measured with an inline-asm dependent-add
   chain: **only core 0 boosts** (4.95-5.01 GHz); cores 1-7 sit at 3.95-4.07. fp32 is 21% faster on
   core 0 — *exactly the clock ratio*, because it is clock-bound. VNNI is 34.72 on core 4 vs 33.90
   on core 0, i.e. **completely clock-insensitive**, because at 1T DDR-resident it is memory-bound.
   So every "1 core" fp32 number pinned to core 0 is a ~22% best case; every VNNI one is unbiased.
3. **An SMT sibling of a busy core costs ~50%** (fp32 T1 4.06 med / 6.2 p5 on cpu9, sibling of a
   loaded cpu1, vs 8.4 / 8.6 on a free physical core). **Any core budget must be stated in PHYSICAL
   cores.**

### 1.5 The fp32 pool does not degrade gracefully — it falls off a cliff

**[M]** Unpinned and oversubscribed under load, T=16 gave a **fixed** 6.0013 and 6.0018 ms/layer
(4.61 GB/s) in two independent runs, and T=12 a 3.12 ms median against a 0.647 ms p5. That is the
sense-reversing spin barrier losing a whole scheduler timeslice whenever a worker is descheduled —
not a bandwidth result. **The 55 GB/s @16T headline only exists on an otherwise idle box.** This is
why the core budget in §4 is a *refusal* rather than a clamp.

### 1.6 Additivity — measured on both channels

**[M]** MoE threads pinned to cores 0-1, co-load pinned to cores 6-7:

| co-load on cores 6-7 | fp32 T2 GB/s | VNNI T2 GB/s | co-load achieved |
|---|---:|---:|---|
| none | 17.61 | 41.98 | — |
| 2x AVX-512 FMA (pure compute) | 17.23 (−2%) | 41.33 (−2%) | 505.6 GFLOP/s |
| 2x sequential DDR stream | 10.36 (−41%) | 20.60 (−51%) | 52.7 GB/s |

**Core isolation is free. DDR isolation does not exist.** Half a TFLOP of AVX-512 on the
neighbouring cores costs 2%; a stream that takes ~93% of the bus halves everything. The stream
co-load is a deliberate worst case — the real PCIe path is capped at 12.4-14.5 GB/s.

So the hybrid is **additive in the resource that was blocking it (cores)** and **remains zero-sum
in the one that always was (DDR)**. That was the known constraint, not a new one.

---

## 2. oneDNN / ZenDNN — REJECTED, on format, not on performance

**The user's constraint, recorded: no checkpoint mixing, no requantization of the weights.**
It was **honoured** — and it is precisely what rejects both libraries. Nothing below required a
benchmark, because both fail at the API/descriptor level before a kernel is ever selected.

### 2.1 The blocking reason, common to both

Our weights are **E2M1 4-bit FLOATS**: the value set {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}.
Both libraries' 4-bit integer types are **s4 [−8,7]** / **u4 [0,15]**, *uniformly spaced*.
Verified arithmetically: E2M1 scaled by its LCM denominator 2 spans **[−12, 12]** — it **does not
fit s4**. There is therefore **no numerics-preserving repack** into either library's int4 format;
using them would require exactly the requantization the user ruled out.

### 2.2 oneDNN specifically

- **Availability was never the obstacle.** Three routes exist: (a) already on disk — libdnnl is
  vendored in torch at `.../torch/include/oneapi/dnnl`, DNNL_VERSION 3.7.1, zero build;
  (b) `/home/pat/vcpkg/ports/onednn` pins v3.12; (c) a CPU-only source build (~15-30 min) from the
  clone in the session scratchpad. **No build was done on purpose** — the dispatch gates reject our
  shape at descriptor-creation time, so a build could not change the answer.
- **Activations cannot be int8** on the weight-decompression path in any *released* version.
  `src/cpu/x64/matmul/brgemm_matmul_utils.cpp:355-378` gates `weights_decompression_support` on
  `fpmath_mode ∈ {bf16, f16, strict, any}` with the matching `src_dt`; its only consumers are
  `bf16_with_int_wei_dt` / `f16_with_int_wei_dt` / `f32_with_int_wei_dt`.
- An int8-activation x int4-weight grouped path (`int8_grouped_quantization_dt`, same file line 381)
  **does** exist, uses the VNNI 4b-packed B layout (`BA16a64b4a`, line 848) and carries no AMX gate
  — but it is **main-only**: symbol count is 0 in tags v3.9 / v3.10 / v3.11 / v3.12. The shape we
  want is unreleased.
- oneDNN does have a native `dnnl_f4_e2m1` dtype, but not on a CPU path that pairs it with int8
  activations and group-16 scales.
- **No grouped/MoE primitive** that batches 10 independent experts.
- **Not benchmarked; no number is quoted.** The path that *would* dispatch on Zen 4 (bf16
  activations, s4 weights decompressed to bf16) is the same decompress-bound, per-core-compute-bound
  shape as our fp32 core — no structural reason to expect 3-4x. Worse for M=1: brgemm copies and
  unpacks B into a scratchpad per call (`apply_scales_in_buffer_b`, `brgemm_matmul_utils.cpp:2011`)
  and at M=1 there is **zero reuse to amortise it**.

### 2.3 ZenDNN specifically

ZenDNN's W4A8 path is the closest thing on offer and still does not fit:

- Weights are **`data_type_t::s4`, symmetric, range [−8,7]**, "2 values packed per byte", with
  dequant `dequant_value = s4_weight * scale`. Same integer-lattice blocker as oneDNN.
- Activations must be handed in as **bf16** with `dynamic_quant = true`; the library quantizes to
  s8 internally. We would be handing it f32/int8, and f32 src/dst is explicitly rejected at
  validation on this path.
- Group-16 scales are structurally fine (`(K/G) % 4 == 0` holds), and `group_matmul_direct` does
  have a genuine fused-MoE surface (Op1 gate+up → `silu_and_mul` → Op2 down, weighted-reduce
  post-op) — architecturally the right shape. But the fused-MoE *dynamic source* path additionally
  requires **`M[i] >= 16` per expert**, and decode is **M = 1**.
- **ZenDNN is not installed on this box** (nothing outside the scratchpad docs), so nothing was
  benchmarked and no number is quoted.

**Verdict: REJECT both.** Not "slower than ours" — *cannot express our weights without breaking a
stated constraint*. The consequence is that the hand-written AVX-512 core in `tools/cpu_moe/` is not
a preference, it is the only thing that reads this checkpoint's bytes as they are.

---

## 3. The accuracy cost of int8 activations

### 3.1 The measurement

**[M]** Reference: a float64 dequant-and-GEMM over the **raw safetensors bytes**
(`make_fixture.py`), full layer, 10 experts x (gate, up, SiLU-mul, down), real routing weights.

| layer | fp32 core | VNNI int8 core | ratio |
|---|---:|---:|---:|
| L0 | 2.416e-07 | 8.279e-03 | 3.4e4 |
| L1 | 2.367e-07 | 9.077e-03 | 3.8e4 |
| L2 | 2.188e-07 | 7.637e-03 | 3.5e4 |
| L3 | 2.557e-07 | 8.645e-03 | 3.4e4 |

**It is irreducible, not a bug.** 16 roughly-Gaussian activations give amax ≈ 2σ, so the step is
2σ/127 and the relative RMS error is ≈ 1/220 = 4.5e-03 per GEMV; a layer quantizes twice (x, then
h) → predicted √2 x 4.5e-03 = 6.4e-03 before SiLU amplification, measured 8.3e-03 [D vs M].
Cross-validated by an **independent numpy implementation** with float64 weights and only the
activation format varied, reproducing the kernel to five digits (8.2786e-03 vs 8.279e-03) [M].

### 3.2 The comparison that actually decides it

fp32 is the wrong baseline. The GPU path this offloads *from* already serves these experts with
**per-token FP8 E4M3 activations** (`python/minisgl/quant/method.py:440`). Measured on the same
fixture with weights held **exact** [M]:

| activation format | L0 rel_rms | L1 rel_rms |
|---|---:|---:|
| fp32 (the CPU fp32 core) | 0.0 | 0.0 |
| **int8, group of 16 — THIS KERNEL** | **8.279e-03** | **9.077e-03** |
| int8, per token | 2.009e-02 | 2.243e-02 |
| fp8 e4m3, group of 16 | 3.253e-02 | 3.201e-02 |
| **fp8 e4m3, per token — WHAT THE GPU SERVES TODAY** | **4.094e-02** | **4.026e-02** |

**Moving a layer to this kernel makes it 4.4-4.9x MORE accurate than the layer served today.**
The outlier in the system is the fp32 CPU core, which is more accurate than anything else anywhere
in the serving path.

### 3.3 Is it acceptable for a reasoning model?

**It is a real decision, and it points the safe way — but it has not been quality-tested.**

- The honest framing is *not* "8e-03 vs 2e-07". It is "8e-03 vs the 4e-02 already being served",
  and by that framing the change is strictly an improvement on the layers it touches.
- This is a reasoning model, where quality loss shows up late and in ways an rms number does not
  see. Per the repo's own rule, **quality must never be tested at temp=0** — greedy fakes
  degeneration that mimics a quant bug. No sampled quality run has been done. That is the gate.
- **The fp32 core stays** as (a) the correctness **oracle** every VNNI change is validated against —
  it is bit-unchanged by this work, 2.416e-07, identical in the pre-existing and new binaries [M] —
  and (b) the higher-fidelity option whenever spare cores exist.
- `CpuActPolicy` carries the speed **and** the error together, by construction, so a plan cannot
  quote one without the other.

**Flagged, not papered over:** `RESULTS_KERNEL` records 1.988e-07 for L0 with the same fixture and
the same on-disk binary; today it reproduces at **2.416e-07** (~20% higher) across all four layers.
Both are pure fp32 accumulation rounding and nothing downstream depends on the difference, but the
recorded table did not reproduce and the discrepancy is **unaccounted for**.

---

## 4. The seam, and the core-arbitration answer

### 4.1 A third placement tier

`StackKind` gains **`CPU = 2`** and the seam / planner / bake path carries it end to end:

- `plan_three_tier()` assigns the CPU block **FIRST** — deterministic and budget-independent, so it
  is **rank-identical**; the existing greedy `(-priority, index)` device fill then runs over what is
  left.
- `TorchStackAllocator.alloc_like(CPU)` returns plain **pageable** `torch.empty(device='cpu')` —
  no `hipHostMalloc`, no device mapping, no arena — and the **same `_bake`** runs with the same
  bitwise read-back self-test.
- `MoEWeightSeam.resolve()` **REFUSES** a CPU seam: a pageable, unmapped pointer must never reach
  the grouped kernel. `MoELayer.forward` branches on `seam.computes_on_cpu` *before* resolving.
  `ExpertStackTable.as_tensor` refuses to materialise a device selector containing CPU.
- **A lifetime bug from the prior phase was fixed:** `attach_cpu_worker` is a placement entry point,
  and `freeze()` (rule R1) made every caller-side attach raise — so all 7 seam tests had **never
  run**. The worker is now attached *inside* `bind_plan`, in plan order, with a packed per-layer
  expert offset. `bind_plan` raises if a CPU plan arrives with `cpu_worker=None`: a worker-less CPU
  seam is not a degraded mode, it dies on the first token.
- `sizing.cpu_wload_policy()` maps a quant scheme to a `wload.hpp` policy or `None`;
  `plan.cpu_tier_gate()` turns that into **four refusals** — no CPU core for the format / EP active /
  core budget exceeded / nothing to place — and `resolve_weight_plan(num_cpu_layers=, cpu_repacked=)`
  **RAISES** on a refused request instead of silently downgrading to 0.
- The EP refusal is load-bearing: `MoELayer._ep_dispatch` all-gathers and re-orders rows across
  ranks, so a CPU partial computed from **pre-gather** rows lands on the **wrong tokens** — fluent,
  plausible, and wrong.

**Tests: 954 passed** [M]; the single failure is the pre-existing `TestRealHip` case that needs a
card. Box during the run: 0.56 of 16 hw threads busy, load 2.09. Nothing in that run is a timing
measurement — it is arithmetic over the kernel phase's recorded numbers.

### 4.2 Overlap design: **BLOCK** — whole layers on the CPU, serial with the GPU, no overlap

The premise that a synchronous per-layer CPU call must be slower than streaming is **refuted by
this box's own numbers** [D from M]:

| per layer, 10 experts | cost |
|---|---:|
| streamed over card 1's Gen4 x8, quiet (14.48 GB/s), 3.072 MB/expert | 2.12 ms |
| streamed, loaded (12.36 GB/s) | 2.49 ms |
| **computed on 4 pinned physical cores** (53.50 GB/s, 2.7648 MB/expert) | **0.517 ms** |

**4.1-4.8x cheaper with zero concurrency required.** The pessimistic end of the *unmeasured* handoff
(40 µs x 37 layers = 1.48 ms) does not change the sign.

> **`[ENDGAME-2026-09-04]` The two "streamed" rows are ~2x too high and this table's conclusion is
> REVERSED by measurement.** GPU-event timed on the live serve, same shape, same kernel, only the
> medium differing: a host-resident routed-expert layer costs **1.135 ms on card 1** (the 14.48 GB/s
> link) and **0.589 ms on card 0**; a device-resident one costs 0.114 ms
> (`attrib/decode_attrib_run2.json`, `isolate`). The streamed rows above were computed from bytes ÷
> link rate and never measured against the engine, which reads the pinned arena *inside* the grouped
> NVFP4 kernel — there is no separate H2D copy to pay for, so the compute is absorbed into the read.
> The CPU row survives as a kernel figure but not as a *layer* figure: on the live serve the same
> tier measures 1.64-1.83 ms/layer in-backend at 2 threads and the end-to-end serve goes 12.61 →
> 11.19 tok/s at 36 CPU layers. The handoff was the thing that was never measured, and it is what
> decided it. See §6.2's correction and `QWEN4EXP_ENDGAME.md` §5.

**REJECTED, with the arithmetic:**

- **SPLIT** (partition each layer's top-k between GPU and CPU) is the only genuine bs=1 concurrency
  there is, and `split_speedup()` prices it at
  `min(pcie + cpu, ddr_wall) / cpu = min(12.36 + 53.50, 57.0) / 53.50 =` **1.065 — a 6.5% gain**,
  ~1.2 ms of a ~40 ms step. Against it: an extra host fence *inside* every layer, the unmeasured
  handoff moved onto the critical path, and the loss of even the possibility of segmented capture.
  It exists, gated and opt-in.
- **Cross-microbatch pipelining** (CPU runs batch A's layer L while the GPU runs batch B's) yields
  **exactly zero at bs=1**, which is the operating point this feature exists for. It needs the
  scheduler to split a decode batch into offset halves — every attention kernel then runs at half
  the batch width, and this engine's decode is already 71% busy in-kernel. `CpuTierMode` has no
  member for it, on purpose.
- **Cross-layer prefetch/overlap does not exist here at all**: the residual stream is sequential,
  and layer L+1's *route* is a function of layer L's *output*.

**Capture cost, corrected:** an earlier version counted graph segments *between* layers. The cut is
**inside** a layer — attention, the norms and the router of a CPU-MoE layer are still GPU work — so
K CPU layers cost **K+1** device segments whether or not they are contiguous. Contiguity buys
nothing for capture; the stated reason `assign_cpu_block` took a contiguous block was wrong.

### 4.3 Core arbitration — the answer

**The budget is a REFUSAL, denominated in PHYSICAL cores.**

```
8 physical − 1.88 (TP=2 engine, MEASURED: 0.962 + 0.922 over a 20 s /proc/PID/stat window)
           − 0.50 (OS, kworkers, the offload driver's copy/dispatch threads — ESTIMATE)
           = 5.62  →  max_threads = 5  node-wide, across all ranks
```

- **`assert_fits` raises; it does not clamp.** Justification is §1.5: the pool does not degrade in
  proportion when starved, it falls off a cliff (a fixed 6.0 ms/layer, ~12x its 0.49 ms) and takes
  the scheduler's cores with it. An over-budget configuration is a boot error.
- **Physical, not hardware threads** — §1.4(3): an SMT sibling of a busy core measured ~50% slower,
  so a budget in hw threads double-counts every core and produces a plan that measures at half its
  projection.
- **WHICH cores: the tier takes from the TOP and leaves core 0 to the engine.** Core 0 is the only
  one that boosts (4.95-5.01 vs 3.95-4.07 GHz [M]); the VNNI kernel is clock-*insensitive* at this
  operating point (34.72 on core 4 vs 33.90 on core 0 [M]) while the engine's Python
  forward/dispatch thread is not. This costs the CPU tier nothing measurable and is the one piece of
  free arbitration available.
- `engine_cores = 1.88` is **one observation, not a distribution** — the concurrent workflow exited
  before it could be re-measured under a known decode load. It is flagged as such in the
  `provenance` string.

At the intended operating point — **2 physical cores per rank x 2 ranks = 4** — the tier fits with
1.6 cores to spare. **3 per rank is refused** (6 > 5), and the refusal message names the measured
reason.

### 4.4 Capacity — the second, independent win

A CPU-tier layer consumes **zero device bytes** (like a HOST layer) **and zero PINNED-arena bytes**
(unlike one): its weights are read by CPU cores with ordinary loads, so pageable memory suffices,
and pageable memory is not subject to `OffloadPrior.host_arena_ceiling_bytes` (P3b: 62 GiB
node-wide, derated x0.90 to 55.8 GiB).

The shipped plan is at **54.2 GiB node-wide against 55.8** [D] — i.e. the pinned ceiling, not VRAM,
is the binding constraint today. Moving those 37 layers to the CPU tier **frees all 54.2 GiB of
pinned arena** and is the only lever that relieves the ceiling without surrendering KV pool.

It is also **10% fewer resident bytes** (2,764,800 vs 3,072,000 B/expert), for the reason in §5.
`cpu_layout_fraction = 0.9` is exact for *this* checkpoint and defaults to 1.0 for every other
format — a plan may only claim it when a repacker is actually wired.

---

## 5. The format finding — `weight_scale_2` is a MULTIPLIER

### 5.1 The convention is inverted relative to the documented one

```
W[n,k] = E2M1_LUT[code] * e4m3(weight_scale[n, k/16]) * weight_scale_2
                                                        ^^^^^^^^^^^^^^ MULTIPLY
```

`python/minisgl/quant/nvfp4.py` documents and implements the llm-compressor convention —
`weight_global_scale = 448*6/amax`, a **DIVISOR** — and its loader keys on the *name*
`.weight_global_scale`. This checkpoint spells it **`.weight_scale_2`** and stores the
**reciprocal**. Using the documented sign yields |W| ≈ 4.2e6 instead of rms 0.0135.

**Pinned four independent ways** [M]:

1. Dividing → |W| 4.2e6 (absurd); multiplying → rms 0.0135 (a textbook weight matrix).
2. `down_proj`'s largest block-scale byte is **126 = e4m3 448 = FP8_E4M3_MAX for EVERY expert** —
   only `scale = block_amax / FP4_MAX / weight_scale_2` can produce that saturation.
3. `gate_proj` and `up_proj` share **ONE** `weight_scale_2` across all **512** experts (they were
   quantized as one stacked/fused tensor) — which is exactly why *their* per-expert block maxima sit
   below 448.
4. `input_scale * 2688 = 5.3`, a plausible activation amax; the reciprocal would be 1.4e6.

**Not currently a live bug:** `config.py:385` lists `.weight_scale_2` among the names the loader
ignores. But the consequence is important and easy to miss — **these experts are not served through
the NVFP4 fold path at all.** Any new kernel must use the multiplier convention, and any future
attempt to route this checkpoint through `nvfp4.py` will silently produce 4.2e6-magnitude weights.

### 5.2 The checkpoint's native e4m3 scale is both smaller and far more accurate

**[M]** Same tensors, same float64 reference, only the scale layout varied:

| scale layout | bytes/expert | B/weight | rel_rms |
|---|---:|---:|---:|
| **e4m3 (checkpoint-native, what the CPU core holds)** | **2,764,800** | 0.5625 | **1.988e-07** |
| fp16-folded (what the GPU path holds in VRAM) | 3,072,000 | 0.625 | 3.577e-04 |

**10% smaller AND ~1800x more accurate.** The 3.58e-04 is not a kernel error — it is the fp16
group-scale fold's own quantization loss (fp16 has 11 mantissa bits → ~2.4e-04 relative; measured
matches). Note e4m3 → fp16 is *lossless* (3 mantissa bits into 10); the loss comes from folding the
per-tensor global in.

**This independently corroborates the e4m3 scale-policy work on the GPU side.** That change is
legal as a **WLoad policy** under `KERNEL_CORE_POLICY.md` — a weight format is a loader policy on
the existing shared core, never a new kernel — and is scoped at ~4 days. It is **blocked on
templating two hand-written non-WLoad kernels**:

- `rdna4-hip-kernels/fp8_wmma/fp8_wmma_rocm/moe_kernel.hip:299` — `moe_gemm1_silu_alds_kernel`
- `rdna4-hip-kernels/fp8_wmma/fp8_wmma_rocm/moe_kernel.hip:450` — `moe_gemm1_silu_ashuffle_kernel`
  (the served default)

Both take `const __half* __restrict__ w_scales` directly in their signatures and dequantize inline,
so the scale width is welded into the kernel body rather than expressed as a policy. Templating
those two on a `WLoad` is the prerequisite; it also buys the GPU tier the same 10% VRAM reduction.

### 5.3 One more measured layout lesson, recorded so it is not re-litigated

The e4m3 layout **initially LOST**, and the reason is instructive: the first e4m3 decoder was a
fully general e4m3→fp32 bit-surgery with a subnormal blend and a NaN mask, **19 vector ops per
tile**. At 1-3 threads the loop is ALU-bound, not DDR-bound, so those ops cost more than the 11%
byte saving was worth — e4m3 lost by **29.7% at 1 thread** and 5.6% at the 2-thread operating point
[M].

The fix was a **census over the real checkpoint**: 629,145,600 `weight_scale` bytes across layers
0-3 — exponent field spans 8..15, byte range 64..126, **zero subnormals, zero NaN slots, zero
negatives, zero zeros** [M]. A block scale is an amax over a positive constant, so none of those can
occur in a well-formed NVFP4 tensor. Specialising to positive-normal e4m3 makes the decode
`cvtepu8 → shift → add` — **three ops** — and hoisting the per-tensor global out of the tile loop
removes one more zmm multiply. **1-core throughput 21.3 → 27.4 GB/s (+29%)**, and the 2-thread point
flipped from −5.6% to +8.7% [M]. `Emit<WLoadVnniE4m3>` re-checks the precondition on every scale
byte at **load** time and dies loudly if a checkpoint ever violates it.

Prefetch (`CPU_MOE_VNNI_PF` = 0/2/4/8 tiles ahead) is a measured **NO-WIN** — the HW streamer
already covers the sequential ~900 KB per-projection runs. Default stays 0.

### 5.4 Why the VNNI core is a second core and not a fourth WLoad

`KERNEL_CORE_POLICY` allows exactly one exemption — "a genuinely different algorithm or tiling" —
and this is it, for an arithmetic reason rather than a convenience one:

- **fp32 core:** lane = a *k* index, one row at a time, row-major bytes, vector activation.
- **VNNI core:** lane = a **ROW**. `VPDPBUSD` reduces 4 *k* values into each int32 lane, so *k*
  cannot be the lane index; rows must be, and the resident bytes must be tiled 16 rows x 16 k.
  Rows-in-lanes is also what makes the group scale a plain 16-lane multiply and removes every
  horizontal reduction.

What stays **shared**: the E2M1 codebook (`kE2M1I8` is derived from and checked against `kE2M1`),
the WLoad scale semantics, `ExpertSlab`, `silu_mul`, the row-range partition contract, the
accumulate/store convention, the plan reader, the table builder, the thread pool, and the
verify/bench harness. And the int8 core is itself **format-parameterised the same way**: two
policies (D e4m3, E fp16) share one body whose only policy-dependent line is `WL::tile_scale()`.

**E2M1 → int8 is EXACT and the weights are never requantized:** the magnitudes
{0, .5, 1, 1.5, 2, 3, 4, 6} doubled are the integers {0, 1, 2, 3, 4, 6, 8, 12}, so the codebook maps
losslessly and the implicit /2 folds into the group scale. `+16` makes them unsigned for VPDPBUSD's
(u8 x s8) form; the bias comes off with a per-group `16*sum(xq[g])` correction shared by all 16 rows
of the tile.

---

## 6. Honest end-to-end expectation vs the target

**Target: 23.3 tok/s, without MTP and without speculative decoding** (the user ruled both out as
unfair). **Today: 11.85 tok/s** [M], offloaded TP=2, 37 host / 11 device layers.

### 6.1 The budget arithmetic [D]

Per token, 37 offloaded layers, top-10, 2,764,800 B/expert = **1.023 GB/token**.

- 11.85 tok/s (today) = 12.1 GB/s
- 23.3 tok/s (target) = **23.8 GB/s**

| cores | VNNI GB/s [M] | MoE ms/token (37 L) [D] | share of the 42.9 ms token budget |
|---:|---:|---:|---|
| 1 | 27.38 | 37.4 | 87% — meets 23.8 GB/s, but far too tight |
| **2** | **44.64** | **23.0** | **54% — fits, 19.9 ms left for everything else** |
| 3 | 54.00 | 18.9 | 44% — DDR-saturated, diminishing |
| fp32 @ 6 | 46.39 | 22.1 | 51% — same place, 3x the cores |

At 2 cores/rank VNNI takes 44.6 of the ~57 GB/s DDR wall, leaving ~12 GB/s — almost exactly the
12.4-14.5 GB/s PCIe cap for streaming the device-resident layers. The two halves land balanced,
which is a coincidence worth not disturbing.

### 6.2 The projection [P]

Run through `project_cpu_tier` (BLOCK, vnni_int8, 37 CPU layers, TP=2, 48 layers total), quoting
both ends of the *unmeasured* 10-40 µs/layer handoff bracket:

| cores/rank (total) | handoff | raw tok/s [P] | **calibrated tok/s [P]** |
|---|---|---:|---:|
| 1 (2) | 10 µs | 32.23 | 20.53 |
| 1 (2) | 40 µs | 31.11 | 19.82 |
| **2 (4)** | **10 µs** | **36.72** | **23.39** |
| **2 (4)** | **40 µs** | **35.28** | **22.47** |
| 3 (6) | — | **REFUSED** — 6 physical cores requested, 5 free | |

"Calibrated" applies **x0.637**, the single measured-vs-projected fidelity ratio this model has ever
been checked against (11.85 measured / 18.6 projected on the shipped two-tier plan). **One point.**

> ### `[ENDGAME-2026-09-04]` §6.2 IS WRONG. MEASURED: the tier is a REGRESSION, not a 2x.
>
> The tier is now wired end to end and has run on the live 48-layer TP=2 serve (commit `e9dd1d7d`;
> raw `docs/measurements/WEIGHT_OFFLOAD_2026-09-02/r3_cpu{0,12,36}t2.json`). One harness, one boot
> procedure, the checkpoint's own sampler, graphs OFF on every leg because **the tier cannot be
> captured at all**, differing only in `--cpu-moe-layers`:
>
> | CPU layers | tier split | measured tok/s (rank 0 / rank 1) |
> |---:|---|---:|
> | 0 | 11 dev / 37 host | **12.61 / 12.59** |
> | 12 | 11 dev / 25 host / 12 cpu | 12.39 / 12.36 |
> | 36 | 11 dev / 1 host / 36 cpu | **11.19 / 11.18** |
>
> **11.19 tok/s measured against 22.47-23.39 projected — the projection over-promised by 2.0x**, and
> it was wrong in *sign*, not merely in magnitude: every layer moved to the CPU costs decode. The
> `x0.637` calibration did not save it; one point was never enough, and it was the wrong shape.
>
> The refuted premise is §4.2's table. Its 2.12-2.49 ms/layer "streamed" prior is what the 4.1-4.8x
> advantage was computed from, and it is **~2x too high**: a host-resident routed-expert layer
> measures **1.135 ms** on card 1 (the slow link) and **0.589 ms** on card 0, GPU-event timed on the
> live serve (`attrib/decode_attrib_run2.json`, `isolate` block). The CPU core's own 0.517 ms is
> *not* the error — the kernel is as fast and as accurate as claimed (1.64-1.83 ms/layer in-backend
> at 2 threads with the seam included; 8.379e-03 rel_rms vs a float64-activation oracle). The error
> is that it was scored against a stream cost nobody had measured, and that the seam around it
> (D2H/H2D of the hidden state, a Python per-layer handoff, and the loss of every captured graph)
> costs more than the layer does. §6.4's item 1 — "nothing here has run against a live serve" — is
> precisely what went wrong.
>
> **What survives, uncontested:** the capacity win. The pinned arena drops 27.10 → 0.732 GiB/rank,
> **52.73 GiB freed node-wide**, proven off-device by the seam proof. The tier is a capacity feature
> that costs 11% decode. Full write-up: `docs/measurements/QWEN4EXP_ENDGAME.md` §5.

### 6.3 The honest reading

**The target is reachable but not comfortably cleared.** At the intended 2 cores/rank operating
point the calibrated projection lands at **22.5-23.4 tok/s against a 23.3 target** — i.e. **on the
line**, straddling it depending on which end of the unmeasured handoff bracket is true. Raw
(uncalibrated) it is 35-37 tok/s, but that model has demonstrably over-promised by 1.57x once, and
quoting the raw number would be exactly the over-promise the calibration exists to prevent.

**A ~2x improvement over today's 11.85 is well-supported by the measured per-layer numbers**
(4.1-4.8x cheaper per offloaded layer, §4.2). Clearing 23.3 with margin is not.

### 6.4 What is NOT measured — and it gates the tok/s claim

1. **Nothing here has run against a live serve.** Every figure is the MoE expert path in isolation.
   The 23.3 target depends on attention, the dense layers, the router, and the host↔device handoff,
   none of which this harness touches.
2. **The handoff is unmeasured.** `handoff_us_bracket = (10, 40)` µs/layer is an estimate;
   `handoff_measured = False`; every caller must state which end it is quoting, and the projection
   carries the flag so a report cannot launder it.
3. **`projection_fidelity = 0.637` is ONE point.**
4. **`engine_cores = 1.88` is one observation**, taken before the concurrent workflow exited, never
   re-measured under a known decode load.
5. **`ddr_contention_derate` is fit from ONE point**, and that point is a deliberate worst case
   (a co-load taking ~93% of the bus, far beyond the 12.4-14.5 GB/s the real PCIe path reaches).
6. **VNNI has never been measured under a real co-running serve** — the serve exited mid-session.
   The only co-load data is the synthetic generator in §1.6.
7. **No sampled quality run** on the int8-activation path (§3.3). This is the gate before it ships.
8. **The load-time repack** into the 16x16 tile layout is real work not yet costed in a boot budget.
9. The 2.416e-07 vs 1.988e-07 fp32-oracle discrepancy (§3.3) is **unexplained**.

---

## Files

| path | what |
|---|---|
| `tools/cpu_moe/wload.hpp` | WLoad weight-format policies incl. D (vnni e4m3) and E (vnni fp16), the specialised positive-normal e4m3 decode + its load-time precondition, the int8 E2M1 codebook + its check |
| `tools/cpu_moe/moe_core.hpp` | the fp32 AVX-512 E2M1 core, `gemv_e2m1_vnni` (int8 core), scalar-double twins, `quantize_act_g16{,_range}`, `QAct` |
| `tools/cpu_moe/cpu_moe_layer.cpp` | full-layer decode driver, `Emit<>` loaders incl. the 16x16 tile repack, threaded table fill, `run_policy<>` dispatch, `--policy vnni\|vnni16`, `selftest_vnni` |
| `tools/cpu_moe/make_fixture.py` | reads real safetensors, computes the float64 dequant-and-GEMM reference |
| `tools/cpu_moe/act_format_error.py` | activation-format error with weights held exact (§3.2) |
| `tools/cpu_moe/coload.c` | compute / stream co-load generator (§1.6) |
| `tools/cpu_moe/run_percore_ab.sh` | paired interleaved bench driver with load capture |
| `tools/cpu_moe/decode_microbench.cpp` | the per-core arithmetic ceiling (§1.3) |
| `tools/cpu_moe/RESULTS_2026-09-04.txt` | ceiling probe (DDR gather, interference, the 57.2 figure's real origin) |
| `tools/cpu_moe/RESULTS_KERNEL_2026-09-04.txt` | fp32 core: correctness, throughput, format finding |
| `tools/cpu_moe/RESULTS_PERCORE_2026-09-04.txt` | per-core budget, taskset effects, clock census |
| `tools/cpu_moe/RESULTS_VNNI_2026-09-04.txt` | the int8 core: headline, accuracy, scale width, additivity |
| `python/minisgl/weights/cpu_tier.py` | the third tier: `CpuActPolicy`, `CoreBudget`, `CpuTierPrior`, `ddr_share`, `split_speedup`, `project_cpu_tier` |
| `python/minisgl/weights/cpu_worker.py` | the native worker binding |
| `python/minisgl/weights/{placement,plan,sizing,stacks}.py` | `StackKind.CPU`, `plan_three_tier`, `cpu_tier_gate`, `cpu_wload_policy` |
| `python/minisgl/{layers/moe.py,weights/moe_interpose.py}` | the forward-side seam + its CPU refusal |
| `tests/core/test_cpu_tier_{placement,seam,handoff,planner_wiring}.py`, `test_cpu_moe_reference.py` | 954 passing |

Build (host, **no GPU, no container**):

```
g++ -O3 -march=znver4 -std=c++17 -o cpu_moe_layer_vnni tools/cpu_moe/cpu_moe_layer.cpp -lpthread
```

---

## Recommended next steps, in order

1. **Measure the handoff.** It is the largest unmeasured term in the only projection that decides
   whether the target is met, and it is cheap to measure.
2. **Run it against a live serve at 2 cores/rank** and get a real tok/s. Everything above is the
   expert path in isolation.
3. **Sampled quality run** on the int8-activation path (never at temp=0), against the fp8-per-token
   baseline the GPU already serves — not against fp32.
4. **Re-measure `engine_cores`** under a known decode load; the whole core budget hangs off one
   observation.
5. **Template `moe_kernel.hip:299` and `:450` on a WLoad** — it unblocks the GPU-side e4m3 scale
   policy, which §5.2 shows is 10% smaller and ~1800x more accurate than what is served now.
