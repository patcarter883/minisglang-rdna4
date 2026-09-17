# CONTINUE — bf16/fp16 kernel parity with the quantised path

**Goal (user's words):** make the unquantised kernels *work* the same way as the quantised ones —
same structure, everything templated and tiled, the cost model, all the tricks — and exceed Triton
and rocBLAS.

**Session ended:** 2026-08-07. Serve is DOWN, both GPUs FREE, nothing committed, nothing deployed.

---

## RESOLVED 2026-08-07 — the A/B was re-run cold, and split-K landed

Everything in the old "START HERE" is settled. Short version: **−9.6% survives of the claimed −50%**
(1.227× → 1.105× vs rocBLAS), the **router was never a regression** (it is now the biggest win in the
set), and the "robust" `lm_head M=512` result was the artifact. Details below; the harness is
`minisgl-rdna4/tools/minv_tile_ab.py` and the fixture is
`tools/_fixtures/minv_tile_ab_cold_card0.csv`.

### The rotation bug was real but SECONDARY. There were two defects.

1. **It ran hot** — the rotation was built and never used. Note the doc's own suggested fix (a
   counter inside the timed lambda) does **not** work either: `torch.cuda.CUDAGraph` bakes the weight
   pointer in at capture, so every replay re-reads one buffer. The rotation must be unrolled INTO the
   graph — `reps` distinct captured calls over `ncopy` buffers — which is what `sweep_policy.py`'s
   `gbench` always did.
2. **The baseline leg was not the old code.** It emulated "pinned" by setting the NEW
   `_BLOCK_M_OVERRIDE/_BN_OVERRIDE` to 64/64. The old code never fed `_BLOCK_M` to the pipe arm at
   all — pipe ran its own staircase — so the override pinned pipe at `pbm=64/mi=1`, a config the old
   engine would never launch above M=256. **That is where the "3.02× lm_head" came from.**

The harness now loads the real pre-change source (`git show 356bfdaa:…/minv.py`) as its own module
and calls both legs, refuses to run if the two sources hash equal, and records the ACTUAL dispatch
per leg. 13 of 30 cells dispatch identically in both legs and all land within ±1.5% — a free
per-run noise floor, and the thing that exposed defect #2.

### Corrected results (cold, card 0, 129 MB rotation)

| | old | new |
|---|---|---|
| vs rocBLAS | 1.227× | **1.105×** |
| cells beating rocBLAS | 7/30 | **12/30** |
| total | | **−9.9%** |

* **`lm_head M=512`: −0.4%, not −70%.** Real old dispatch is `pipe bm256/bn64/mi2` at 3428 µs vs
  3418 µs derived. The only genuine lm_head win is **M=128 (−45.8%, 1.77× → 0.96×)**.
* **The router is now the biggest win: −47% to −51% at M=64..256** (split-K, below), 0.66–0.76×
  rocBLAS.
* Before split-K, the tile change alone gave the router a consistent −10.5% to −14.1%.

### A FIFTH methodology error, same family as the three below

Router M=192 initially regressed **+19.1%**. `sweep_policy.py` hoists the M-pad out of the timed
region (`xp = {bm: pad(x, bm)}` built once, before `gbench`), but `minv_linear` pads INSIDE the call.
The surface priced `rd bm128/bn32` at 16.07 µs; end-to-end it is 25.3 µs — ~9 µs of unpriced pad, and
the fitted rule duly picked it. **Fixed** by selecting the rd tile on minimum padded rows
(`bm = min(lattice, key=lambda b: (ceil(M/b)*b, -b))`), which picks bm96 at M=192 and turns it into
−11.1%. If you extend the lattice, time it through the same wrapper the engine calls.

---

## Branches (nothing committed)

| path | branch | contents |
|---|---|---|
| `/home/pat/code/rdna4-hip-kernels-bf16tune` | `feat/bf16-structural-parity` | 4 files +202/−57, 5 new tools/fixtures; **plus the `SLICED` reduction-order policy** in `dense_gemm_kernels.hip` + `grid_split` through both bindings, `local/splitk_router_ladder.py`, `tools/_fixtures/splitk_router_ladder_card0.txt` |
| `/home/pat/code/minisgl-rdna4-minv` | `feat/minv-unpin-tiles` | `minv.py` tile unpinning + pad-aware rd tile + the split-K router arm; `tools/minv_tile_ab.py`; `tools/_fixtures/minv_tile_ab_cold_card0.csv` |
| `/home/pat/code/minisgl-rdna4-desync` | `feat/canvas-desync` | **FALSIFIED**, keep as the record — see Part D13 |

Also uncommitted in the canonical `minisgl-rdna4`: `docs/DIFFUSIONGEMMA_BLOCK_DIFFUSION.md` Part D13,
and `control-panel/config.json` (new console knobs). `/home/pat/code/minisgl-rdna4-hosttier` is a
removed worktree whose directory survives with root-owned `__pycache__` — needs `sudo rm -rf`.

---

## What landed, and how well it is verified

### 1. `tile_select.h` is dtype-agnostic — GATED, HIGH CONFIDENCE
Five int4-specific literals became an `OperandFormat` policy (`bk` free, `b_it` from `b_bits`, LDS
bytes/elem + scale array, VGPR model, and the **scale-line term made structurally absent**). Same
shape as the existing `AStage` policy.

**Gate: 124,080 cells, 0 mismatches** — `fp8_wmma/local/tile_select_bitident.cpp` compiles the old and
new headers into one TU under separate namespaces and diffs every `choose_tile`/`choose_moe_bn` pick.
Build it with plain `g++ -O2 -std=c++17` (hoist `<cstdio> <cmath> <cstdlib> <initializer_list>` ABOVE
the namespaces or the std lib lands inside them; alias `oldm::w4a8_tile_select`).

Validation that fell out rather than being designed: `bf16_smem_bytes` in `moe_bf16_ops.hip` is
`(block_m + BN)*(BF16_BK + LDS_PAD)*2`, which is *exactly* `moe_lds(..., FMT_BF16)`. The model prices
the launcher's real LDS request.

**Open:** `FMT_BF16`'s `vgpr_fixed/vgpr_per_frag` are carried over from W4A8 and are **unfitted**. The
bf16 probe pins `bk` to the lattice minimum (16) on most shapes — the classic sign of an unmodelled
per-k-step cost (loop + `__syncthreads` overhead that doesn't scale with `bk`). First thing a tiled
surface must check.

### 2. bf16 decode GEMV selector — MEASURED, HIGH CONFIDENCE
`select_gemv_tiling` gained `GemvProfile` + `n_cols`; the bf16 launcher passes N and opts into
`UnquantisedBf16`. The other six callers don't mention `GemvProfile` and are byte-identical.

Fitted rule: **never bylane, nw=4, `cols = N >= 2048 ? 2 : 1`**.

| | all 92 cells | served | ms/step |
|---|---|---|---|
| shipped (`K≤1024→bylane`) | 1.2075 | 1.0607 | 3.873 |
| fitted | **1.0476** | **1.0070** | 3.625 |

**Measured on the compiled build: 4.197 → 3.814 ms/step (−9.1%), 0.4% spread over 3 runs.**

Facts behind it: bylane is the per-cell oracle on **1 of 90 cells**; 30.6% of configs sit inside the
oracle's own spread (flat optimum → two branches is the honest complexity); `bylane iff N>=16384`
scores 1.4186, the worst rule tried.

**Caveat:** the surface predicted 3.625, the build measured 3.814. The gap is `attn.qkv_proj`, a
16.2%-spread shape whose oracle sample was lucky. Surfaces are trustworthy where cells are quiet.

**Deployment gap worth chasing:** the baked `/opt/kernels` measures 4.197, and the source comment
records the *pre-table* tiling at 4.199 — the deployed image predates the `select_gemv_tiling`
adoption entirely. Some of the scorecard's "deficit" is an unbuilt fix.

### 3. `moe_bf16_ops.hip` wired to the model — STRUCTURAL ONLY
`BN <= 0` → `choose_moe_bn(..., FMT_BF16)`; any positive BN honoured verbatim. **This path is
registered but uncalled on the served path** — parity, not throughput. Do not claim a serve win.

### 4. `minv.py` tiles unpinned — SURFACE-SCORED, E2E UNVERIFIED
`_BLOCK_M`/`_BN` were `64` for every shape and every M. Now derived; env vars became overrides
(`0` = derive), which is what makes a same-build A/B possible. Arm rule unchanged (right on 34/36).

Against the surface: **1.0812 → 1.0204 vs oracle, 1.1580 → 1.0928 vs rocBLAS, 9 → 13 of 36 cells
beating rocBLAS, worst 1.89× → 1.33×, zero regressions.** E2E verification is the hot-cache run above.

---

## Fixtures created (these did not exist before)

* `rdna4-hip-kernels/tools/_fixtures/bf16_gemv_surface_card0.csv` — 1,656 cells, 46 shapes × 2 M × 18
  configs × 3 repeats, cold, per-cell spread.
* `rdna4-hip-kernels/tools/_fixtures/dense_gemm_surface_card0.csv` — 2,844 cells, 6 served shapes ×
  M{16,64,128,192,256,512} × full lds/rd/pipe lattice **including MI=1**, cold, graph-replay,
  **rocBLAS in every cell**.

Generators/scorers: `fp8_wmma/local/{sweep_bf16_gemv_surface,fit_bf16_gemv_selector}.py`,
`dense_gemm/local/sweep_policy.py` (now `--csv`, MI=1 added to `PIPE_INST`).

---

## FIVE errors, one root cause — read before trusting any sweep

(#4 the emulated baseline and #5 the unpriced M-pad are written up at the top; #6 below is new.
Every one of them is **the harness and the thing it measures silently disagreeing**.)

6. `splitk_router_ladder.py` originally used `sys.path.append` for the local `torch-ext` build. The
   serve image ships `PYTHONPATH=/opt/kernels:…`, so append puts the local build LAST and the probe
   measures the BAKED kernels — i.e. a freshly compiled kernel change reads as "no effect". Use
   `sys.path.insert(0, …)` **and assert** `dense_gemm.__file__` is under the local tree.
   `sweep_policy.py` and `splitk_router_flips.py` both still `append` — fix them before reuse.
   `minv_tile_ab.py` has `--require-local-kernels` for the same reason.



1. GEMV surface over-predicted 5% — argmax over 18 configs read noise on a 16%-spread shape.
2. Dense scoring graded the shipped policy against **MI=2 configs it never launches**, because
   `PIPE_INST` had drifted from `minv.py`'s `mi = _PIPE_MI if pbm >= 256 else 1`.
3. `minv_ab.py` measured hot because the rotation was built and not used.

All three: **the harness and the thing it measures silently disagreeing.** Concrete fix worth landing —
have the sweep read the instantiated lattice from the kernel and FAIL if its config list doesn't
match, the same provenance assertion `tools/rec_radix_ab.sh` already makes for serve legs. Without
it, every future sweep can repeat #2.

Also: my first unpinned rule regressed `lm_head M=256` by 43% because I dropped the
`OUT>=65536 && M>=192 → pbm=256` clause. It is load-bearing. Keep it.

---

## Split-K — LANDED 2026-08-07, both routes taken (route 2 was cheap after all)

**The doc below said route 1 (select by shape) and dismissed route 2 (make the non-split arm reduce
in split-K's order) as "far more invasive". Route 2 turned out to be ~20 lines, and route 1 alone
would have been a bad trade.** Both are now in.

**Why route 1 alone fails:** the premise "the router never sees large M" is false, and the fixture
that supported it stops at M=512. The router runs once per layer per token, so prefill chunks put M
in the thousands — and since `minv_linear` peels M≤16 to the decode GEMV, the router's dense_gemm
traffic is *mostly* the large-M side. Measured crossover is **M≈448**: split-K is 0.57–0.83×
rocBLAS below it and **2.0–2.8×** above it. Shape-only selection commits the shape at every M, so it
would have bought decode by selling prefill.

**Route 2, as landed** — `SLICED`, a reduction-ORDER policy on the shared `dense_gemm_rd_kernel`
(a template parameter, not a new kernel; see KERNEL_CORE_POLICY.md). It runs the same `kslice`
partition split-K uses, but inside one block, keeping a running fp32 total advanced once per slice —
bit-for-bit what `dense_gemm_reduce_kernel` does to the partials. Cost: `NFRAG*8` extra VGPRs (16 at
the router's BN=32), measured at ±1% vs plain rd at every M.

That splits one conflated decision into two:

1. **WHICH SHAPES use the split-K reduction order** — a per-shape commitment (it IS a bit-move vs
   plain rd), never a function of M. In `minv.py`: `OUT <= 256 and IN >= 1024`. The `IN` guard is
   load-bearing: below it `split_k_slices()` returns ≤1 and the op falls back to plain-rd order
   internally, silently re-mixing two orders on the same weight.
2. **HOW it is scheduled** — `dense_gemm_rd_sk(..., grid_split=...)`. `True` fans slices across
   `blockIdx.z` + reduce pass; `False` is the SLICED single-block walk. **Bit-identical**, so this
   one is free to read M, and does: `grid_split = M < 448`.

**Result on the router: 0.66–0.76× rocBLAS below the crossover, no regression above it.** The
non-split tile lattice's own per-cell oracle never got below 1.19× on this shape.

### Gates (all green, card 0, fixture `tools/_fixtures/splitk_router_ladder_card0.txt`)

* **Schedule bit-identity** — `torch.equal(grid, slice)` True at every M on 16…8192.
* **M-invariance across the schedule boundary** — M=1024 on `slice` vs chunks of 16/32/64/128/192/256
  on `grid`: max|d| **0.000e+00**.
* **M-invariance of the whole shipped path** — all 6 served shapes, M=512 vs chunks of
  32/64/128/192/256: **PASS**, and 0 unexpected bit differences among the non-split arms.
* **Expert-flip** (`splitk_router_flips.py` vs the real captured `gemma4_router.pt`, 5 layers /
  2800 real rows): **0/22400** top-k index mismatches. This is the criterion that actually decides an
  MoE router — top-k is a step function and one ULP can reroute a token, so a "max rel delta 3e-7"
  report would have said nothing. The fixture already existed and had never been run.
* `local/splitk_check.py` still **GREEN** (the original grid-split path is unregressed; bit-move
  is ~0.9 ULP).

### Still open on split-K

* `dense_gemm_pipe_sk` did NOT get the SLICED policy — only `rd` did. Pipe's split arm is still
  all-or-nothing, so no pipe-class shape can take the split-K order without the same prefill penalty.
* The band is deliberately narrow (`OUT<=256`). **Any new shape entering it changes numerics and
  needs `splitk_router_flips.py` re-run against a real capture of ITS inputs**, not a synthetic one.

<details>
<summary>Original design note (superseded, kept for the reasoning)</summary>

## Split-K bit-exactness — designed, not started

**The constraint is not what it looks like.** `dense_gemm_reduce_kernel` already sums fp32 partials in
ascending slice order, so split-K *is* deterministic. The problem is that split-K reassociates the K
reduction, so it is **not bit-identical to the non-split arms** — and `minv` must never mix them for
the same weight across different M, or a token at M=32 stops matching itself at M=512. That is
exactly what prefix caching, chunked prefill and spec-verify depend on.

Two honest routes:

1. **Select split-K by SHAPE, never by M.** All M share one reduction order → M-invariance holds.
   Split-K loses badly at large M (26.2 → 61.1 µs at M=2048), so only shapes that never see large M
   qualify. **The router is exactly that** — OUT=128, flat 15.6–16.3 µs from M=16 to 512, currently
   1.19–1.27× rocBLAS, and split-K is already measured at **0.63× rocBLAS at M=32** (18.1 → 9.1).
   Gate: prove one shape's output is bit-identical across the whole M ladder.
2. Make the non-split arms reduce in split-K's order too. Far more invasive.

Recommend (1), scoped to the router. Re-measure the router cold first (see START HERE).

</details>

---

## Why the dense cost model was NOT ported (do not retry without reading this)

`choose_tile` cannot be applied to `dense_gemm`, and not with a new `AStage` either:

* `C_BSTAGE = 16.0` — the dominant fitted constant — prices int4 unpack + dequant. bf16 B is a straight
  `uint4` copy; that work does not exist.
* The scale-line term is the model's **only** cache-retention term and is identically zero for bf16 —
  yet cache retention is the physics behind the measured `M*OUT` inversion.
* **The model has no B-traffic/B-reuse term at all.** `rd` re-reads B once per 16 rows of M
  (`~ceil(M/16)·OUT·IN`). That is the single most important fact about this family, and `minv.py`'s
  `_RD_MAX_MN = 512K` is a hand-fitted stand-in for it. **Adding that term is the real work.**
* `threads = 32*nwarps` is false — `blockDim` is pinned at 256 with only `block_m/16` warps active.
* pipe's VGPR needs `MI` and `PBK/ADIV` axes, and **ADIV's direction is empirically inverted**
  (shrinking the buffer raised VGPR and added spills).

What *does* transfer: the grid/occupancy skeleton (`WGS`, `ROUNDS`, `blocks_per_cu`, `occ`,
`tile_better`, the `z_blocks` split-K hook). `minv.py:92-95` is already a hand-derived `ROUNDS`
argument — the skeleton is in use, just in prose.

---

## Remaining kernel gaps a selector cannot reach

From the dense surface (perfect selection still leaves ~9.7% of rocBLAS):

* ~~**Router (OUT=128): 1.19–1.27× at every M.**~~ **CLOSED** — split-K now runs it at 0.66–0.76×
  rocBLAS. "Split-K is the only lever" was right; see the split-K section above.
* **Mid-M (256–512), wide OUT: 1.05–1.21×** across gate_up/down/qkv/o_proj/lm_head. All three arms
  lose together — the lattice has no competitive config. That is Tensile's tiles being better.
  **PARTLY FALSIFIED for `o_proj` (2026-08-07): it was never the tiles, it was a 4 KB row-stride L2
  set conflict** — see the next section. The claim still stands for gate_up/down/qkv/lm_head, whose
  IN is not near a 4 KB multiple.
* `64 of 106` `dense_gemm_pipe` instantiations **spill** (max 2676 B/lane, 1376 VGPR).

---

## The 4 KB row-stride conflict — measured, bit-exact fix found, NOT integrated

`dense_gemm_rd`'s warp fragment load reads 16 consecutive ROWS at one k-offset, i.e. 16 addresses
exactly `2*IN` bytes apart. When `2*IN` is a multiple of **4096 B** all 16 fall in one L2 set — a
16-way conflict on every fragment load. (IN=1024, a 2 KB stride, is clean, so the critical stride is
4 KB, not "any power of two".)

Cost per K-element, M=128 OUT=2816, `rd bm128/bn32`:

| IN | 2016 | 2032 | **2048** | 2064 | 2080 | 2112 | | 4080 | **4096** | 4112 |
|---|---|---|---|---|---|---|---|---|---|---|
| ns/K | 14.12 | 17.67 | **29.57** | 17.63 | 14.01 | 13.37 | | 16.99 | **25.99** | 16.97 |

**The fix needs no kernel change:** widen K by `PAD` with the extra **weight** columns zero. The
extra k-steps add exact `0.0` to an fp32 accumulator, so the result is **bit-identical**
(`torch.equal` True on all 24 cells measured), while the stride moves off the boundary. `PAD=16` on
IN=2048 gives 4128 B — still inside the shoulder — and *hurts*; **PAD=64 is the safe pick**. Rule:
smallest `PAD` (multiple of 16) with `2*(IN+PAD) mod 4096 >= 128`.

`o_proj` (IN=2048, OUT=2816) vs rocBLAS, M=16/64/128/192/256/512:
**1.02 / 1.02 / 2.20 / 1.53 / 1.20 / 1.14 → 0.82 / 0.85 / 0.94 / 1.12 / 1.00 / 1.02.**
Pipe benefits too (1.12–1.36×), less than rd (up to 2.11×), because it stages B through LDS.

**CU mode is not the answer — measured, do not re-propose.** `-mcumode` confines a workgroup to one
CU instead of spreading its waves across the WGP's two, which halves the concurrent requestors on the
conflicting sets. It does relieve the spike (**−24%** at IN=2048, −16% at 4096) but does not remove
it (2048 still 22.42 ns/K against ~14.7–18.8 for its neighbours), and it **costs +5% to +14% on every
non-conflicting stride** by cutting memory-level parallelism where there was no conflict to relieve.
Every other served shape is non-conflicting, so a global `-mcumode` is a net loss. Verified at the
artifact — all 136 kernel descriptors, `COMPUTE_PGM_RSRC1` bit 29 — with rocBLAS as an unaffected
control matching within 0.5% across both runs. Build hook: `EXTRA_HIPCC=-mcumode` in
`dense_gemm/local/setup.py`; fixture `tools/_fixtures/cumode_vs_wgp_card0.txt`. Hazard if revisited:
CU mode caps workgroup LDS at 64 KB and `dense_gemm_kernel` requests exactly 65536 B.

**Why the zero-pad is not integrated.** The weight must be *stored* padded — padding per call would copy the
whole weight. Avoiding a VRAM duplicate means allocating `[OUT, IN+PAD]`, zeroing the tail, and
exposing the logical weight as a non-contiguous view `[:, :IN]` for the `F.linear` fallback — which
collides with the `is_contiguous()` check in `minv_supported()`. Duplicating instead costs ~346 MB
for o_proj across 30 layers, which is real money against the KV pool. **That trade is the open
decision.** Only `o_proj` is affected in this checkpoint, but IN=2048/4096/8192 is a very common
hidden dim, so a new model or TP degree can land on it at any time.

Fixture: `rdna4-hip-kernels/tools/_fixtures/stride_aliasing_card0.txt`. Probes:
`dense_gemm/local/probe_stride_aliasing.py`, `local/probe_zeropad_stride_fix.py`.

---

## Two documentation errors to fix

1. **"5.4× over rocBLAS" on the LM head is misattributed.** `bench_bf16_lmhead_gemv.py:90-91` reads
   **5.43× vs our own `minv`** and only **1.16× vs rocBLAS**. Propagated in
   `docs/SPEC_PROPOSE_GRAPH_CAPTURE.md:187` and `docs/COUNTER_SCORECARD.md:339`.
2. **`COUNTER_SCORECARD.md:64` Claim 7 is dead.** It builds a "~1.7× residual gap on the biggest
   decode GEMV" on 355.9 GB/s / 50.4% of roofline. `KERNEL_SWEEP_SCORECARD.md` measured 95.7% on the
   same shape/kernel/compiler, and this session measured **690 GB/s = 106% of the read+write copy
   ceiling with 0.0% spread** (lm_head is read-dominated). Three independent measurements against it.

---

## Unrelated but live (from earlier in the same session)

* **Host-RAM snapshot tier MERGED** to `rdna4` as `356bfdaa`: KV pool **141,408 → 205,248 (+45.1%)**.
  Ladder later doubled to 8 and the LRU cap to **245** with the pool unchanged — proving ladder depth
  is now free in VRAM.
* **Ghost oracle is collecting** (`MINISGL_GHOST_ORACLE=1`, state in
  `/home/pat/fixtures/minisgl-ghost-oracle`, keyed by model, survives restarts). Last reading over
  21,083 real requests: `gap_rec 3.7%` / **`gap_evict 6.3%`** — above the ~5% bar, so **Stage 2 (host
  KV tier) is justified**. The pre-change state is parked as `BEFORE-ladder4-cap41.bin`; the current
  file is a clean post-change measurement.
* **Stage 2 has two prerequisites** or it is useless/fatal: hook `_enforce_rec_cap` (not just
  `evict()`), and `is_leaf` → `_is_device_leaf` at `radix_cache.py:317,335,174,290,400` or the first
  spilled node wedges the eviction heap into a serve-killing assert.
* **fp8-KV is running with identity scales** (uncalibrated) — user chose to track, not fix.

## Environment notes

* `gpu-lease -n 1 -- …` for everything; the serve holds both cards when up, so free it first.
* Kernel builds: `cd <worktree>/fp8_wmma && bash local/build_local.sh` inside `minisgl-rdna4:lean`.
  Then the benches' `sys.path.insert(0, ../torch-ext)` picks up YOUR build instead of `/opt/kernels`.
* Bare probes must call `set_tp_info(0, 1)` or the engage ledger's rank-0 logger raises.
* `minv_linear` peels `M <= 16` to the decode GEMV, so a dense-arm A/B must start at M=64.
