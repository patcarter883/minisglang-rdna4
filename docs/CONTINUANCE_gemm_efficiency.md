# CONTINUANCE — W4A8 / W8A8 GEMM efficiency to 50% / 75% of the FP8 matrix ceiling

Session of 2026-08-12. Started as "why is Muse-Glimmer prefill slow", became a kernel-efficiency
campaign. Three changes landed in a worktree, bit-parity verified, ~3x on the NVFP4 path. Read the
CEILINGS section first — I got the roofline wrong twice and every conclusion built on it inverted.

---

## THE TARGET

| kernel | measured now | target | gap |
|---|---|---|---|
| w8a8 fp8 (no 4-bit decode) | 164.1 TFLOPS (42.2%) | **75% = 292** | 1.78x |
| w4a8 e2m1 g=32 | 77.0 (19.8%) | 50% = 195 | 2.5x |
| w4a8 e2m1 g=16 (NVFP4, Muse) | 65.8 (16.9%) | 50% = 195 | 3.0x |

75% for w8a8 because it has no quantisation overhead — no unpack, no group scales, it is a straight
fp8 GEMM and should sit near the ceiling.

## CEILINGS — get this right or every conclusion inverts

RX 9070 XT (gfx1201), 64 CU @ 2970 MHz boost. **MATRIX** path, dense:

| path | FLOP/CU/clk | boost |
|---|---|---|
| MATRIX FP16/BF16 | 1024 | 194.6 TFLOPS |
| **MATRIX FP8/INT8** | **2048** | **389.3 TFLOPS** |
| MATRIX INT4 | 4096 | 778.5 TOPS |
| SHADER FP32 | — | 48.66 |
| SHADER FP16 (packed a*b+c*d+e) | — | 97.32 |

**97.32 is the SHADER figure, not a matrix ceiling.** I scored a WMMA fp8 GEMM against it, got "70% of
peak, the GEMM is fast", and spent hours looking away from the GEMM. Real answer was ~18%. FP8 and
INT8 matrix ops run at the same rate (Wikipedia RDNA 4 table: INT8 314 base / 389 boost).

**There is no power excuse.** Board is rated 370 W, the driver cap is set to 320 W, and AMD's 389.3
figure is at the factory 300 W limit — i.e. we have MORE power than the spec number assumes and
deliver a fraction of it. "It draws 350 W therefore it is at the limit" is circular: it assumes our
J/FLOP is the silicon's. Silicon does ~1.30 TFLOPS/W; our best path does 0.51.

## WHAT LANDED (worktree, NOT merged)

`/home/pat/code/rdna4-hip-kernels-perf`, branch `perf/gemm-efficiency`, off `b6a0ea1`.
All in `fp8_wmma/fp8_wmma_rocm/w4a8_fp8_wmma_kernel.hip`.

1. **Group-size instantiation lattice.** `TILED_TUNED_BKT` had compile-time instantiations for 128
   and 32 ONLY; every other accepted size fell to `BKT=0` (runtime divide, not folded to shifts).
   The kernel accepts `group_size % 16 == 0 && <= 128`, i.e. eight sizes — six were silently ~2x
   slow. Now all eight, loud default. **g=16 2.28x, g=64 2.01x.**
   The same file's TILE lattice already stated the rule: *"Instantiation count is not a constraint
   here; silent misrouting is."* Same author, same file, opposite rule on the group axis.
2. **`ASHUFFLE_BKT` correctness bug.** Was `if (group_size==128) ...128; else ...32;` — and `BKT` is
   SEMANTIC (`const int BK = BKT ? BKT : group_size`), so every non-128 checkpoint was computed as
   g=32: wrong scale stride, silently WRONG numbers. Undispatched arm (needs `VLLM_W4A8_V10_CFG`),
   so it never hit the served path — but `tools/w4a8_dense_tile_arms.py` DOES set that env and time
   it. It escaped only because every recorded sweep used g in {32,128}, the two values the broken
   branch handled correctly. Fixed, but the arm should be DELETED instead (see TODO 1).
3. **`VLLM_W7_DIAG` removed.** Development-only stage ablation, but read INSIDE the kernel: five
   bools live kernel-wide gating branches in the staging loop AND the innermost WMMA/LDS loop.
   Deleting it was worth **+11.0% on g=32**, +3.6% g=128, +1.4% g=16 — a debug knob taxing the
   served path.
4. **Staging depth decoupled from group size.** Was `BK = group_size`, so the LDS tile was exactly
   one quant group deep — a MEMORY decision tied to a QUANTISATION policy. At K=4096 g=128 ran 32
   staging rounds of 48 KB; g=16 ran 256 rounds of 6 KB — same bytes, 8x the barriers, each round
   too small to saturate. Now `GPS = TARGET_STAGE_K(128)/GS` groups per stage, scales staged for all
   GPS groups, applied per group inside the stage. **g=16 +28.0%, g=64 +10.8%, g=32 +1.9%,
   g=128 unchanged (GPS=1, the control).**

**Parity: bit-identical across all 8 group sizes** vs the git-HEAD kernel, both compiled into one
binary from real sources, 1,048,576 elements, zero diffs. (`local/efficiency/parity.hip`.)

### Cumulative, NVFP4 g=16, 4096^3, tile 256x128, boost clocks

| stage | TFLOPS | % of 389.3 |
|---|---|---|
| as shipped | 22.1 | 5.7% |
| + group lattice | 50.7 | 13.0% |
| + V7_DIAG deleted | 51.4 | 13.2% |
| + staging decoupled | **65.8** | **16.9%** |

**2.98x.** Also lifted the "fine" sizes: g=32 68.0 -> 77.0, g=64 36.8 -> 82.0.

## COUNTER PROFILE (the map for what remains)

Per WMMA-floor instruction (floor = FLOP / 16384; 4096^3 -> 8.39 M inst/dispatch):

| kernel | VALU | SALU | LDS | WMMA% of issue | wait% |
|---|---|---|---|---|---|
| w8a8 | 4.6 | 1.0 | 0.7 | 21.7% | 70.0% |
| w4a8 g=32 (pre-item-4) | 24.1 | 13.7 | 2.9 | 4.1% | 83.7% |
| w4a8 g=16 (as shipped) | 55.6 | 19.9 | 5.1 | 1.8% | 86.6% |

Muse production shape (2048x19968x6656) reproduces the square within 2% on every metric — tuning on
4096^3 transfers. `diag` ablation (before it was deleted) at g=16: baseline 2700 us, skip-WMMA 2509
(-7%), skip-global 2198 (-19%), skip-LDS 2276 (-16%), skip-both 1702 (-37%). **The matrix math is 7%
of the runtime; staging is ~37%.** This is a data-movement-bound kernel.

## TODO, in order

1. **DELETE the dead arms** `mmq_fp8_gemm_prefill_wmma_kernel` and `..._ashuffle_kernel`, with
   `VLLM_W4A8_V10_CFG`, `V10_DB`, `V17_CFG`, `V17_SPLITK`. `_pick_dense_kernel`
   (minisgl `quant/kernels.py`) only ever returns `wmma_tiled_tuned` or `decode_gemv` — these are
   unreachable from the engine, measured 0 wins over 210 cells, and are where the correctness bug
   lived. Fold ashuffle's ONE real advantage (B-only LDS, double-buffered staging; beat tiled 1.64x
   at the 256x256 tile) into the shared core as a policy — the tile-surface doc already concluded
   exactly this and nobody acted on it. They are also active decoys: three of my `.replace(...,1)`
   edits silently landed in `prefill_wmma` because its loop body matches the tiled kernel's.
2. **LDS budget mismatch — BLOCKING the package build.** Stage depth is now `GS*GPS` toward 128, so
   LDS per tile is `(BM+BN)*(BK+8)` plus `GPS*BN*4` for staged scales. The launcher's
   `TORCH_CHECK(shmem <= 65536)` and `tile_select.h`'s legality test BOTH still compute the old
   `(bm+bn)*(group_size+8)`. Some tiles legal at g=16 no longer are. The standalone harness sizes it
   correctly (which is why the measurements are clean); the package will fail at launch until both
   are updated. **Fix this before the next `build_local.sh`.**
3. **Item 2 — SALU flood.** Was 13.7/WMMA at g=32. Item 4 deleted 7/8 of the loop iterations that
   arithmetic lived in — RE-MEASURE the counters before touching it; the remaining term may be much
   smaller than the original profile implies.
4. **Item 3 — w8a8 to 75%.** Untouched. 4.6x floor, 1.0 SALU, but **70% wait**: starved, not
   crowded. Prefetch distance and `s_waitcnt` placement in the double-buffer, not instruction count.
   Independent of the W4A8 work.
5. **Finish the env triage.** All of these are development scaffolding that outlived its purpose:
   `V7_CFG` -> make it an explicit argument (the op already takes `kernel=`), not an ambient global
   that re-tiles every kernel in the process; `V7_SWIZ` -> unconditional if it always wins, else a
   cost-model input; `TILE_CU` -> derive it (it exists because torch reports WGPs not CUs — fix at
   source); `TILE_EXPLAIN` -> delete, redundant with the exported `dense_tile_explain` op;
   `MAGIC`/`MAGIC_COLS` -> promote or delete; `DENSE_SMALLM_OFF`/`DENSE_TILED_OFF` -> bake-off
   toggles whose bake-off concluded; `DECODE_GEMV_BK`/`NW` -> argument or cost model.
   Also: **no version numbers in kernel names or knobs** — V7/V10/V17 all violate that rule.
6. **Refit the tile chooser at g=16.** Before item 4 the chooser mispicked by 1.61x at g=16 (it was
   fitted on g=32/128 data only, and never on E2M1 at all — `w4a8_dense_tile_surface.py` hardcoded
   `e2m1=False`). Item 4 changed the LDS/tile trade, so re-sweep before refitting.
7. **Serve-level validation** on Muse-Glimmer once the package builds: the isolated g=16 number
   (24.3 TFLOPS pre-change) independently reproduced the figure derived from live serve prefill
   timings, so the microbench is measuring the real thing — expect prefill ~680 -> ~2000 tok/s.

## TOOLING (preserved at `rdna4-hip-kernels-perf/fp8_wmma/local/efficiency/`)

- `w4a8_counters.hip` — standalone W4A8 harness. BKT lattice 0/16/32/64/128, explicit group size.
  Compiles the REAL kernel template (extracted verbatim), so a win here is a win in the package.
- `gemm_counters.hip` — same for the dense w8a8 kernel.
- `parity.hip` — orig (git HEAD, namespace `w4a8_orig`) vs reworked, byte compare, all 8 groups.
- `iterate.sh` — re-extract kernel core + rebuild harness in SECONDS (vs ~9 min for the package).
  This is the loop to work in.
- `profile_all.sh`, `run_all_counters.sh` — full counter sweeps with perf-level set/restore.
- `kernel_roofline.py` — roofline surface with power soft-start and continuous power monitoring.
- `rl2.txt`, `muse_e2m1.{txt,csv}` — the recorded surfaces.

Also built: docker image **`rdna4-prof714:latest`** (ROCm 7.14 runtime + lean `/opt/venv`).

## GOTCHAS THAT COST TIME

- **Edits land in the wrong kernel.** `prefill_wmma` appears EARLIER in the file and its loop body
  matches the tiled kernel's; `str.replace(x, y, 1)` hits it first. Always verify line numbers
  against `grep -n "^__global__"` after editing.
- **torch hipifies**: edit `w4a8_fp8_wmma_kernel.hip`; `*_hip.hip` is a generated build artifact
  (root-owned). Verify your change appears in the generated file.
- **gpu-lease + docker**: wrap in `bash -c '...'` with SINGLE quotes so `$HIP_VISIBLE_DEVICES`
  expands INSIDE the lease shell. Bare `-e VAR=$VAR` expands in your shell to empty -> "No HIP GPUs".
- **Counters**: need ROCm 7.14 rocprofv3 + `power_dpm_force_performance_level=profile_standard`
  (RDNA4 `auto` gates the perfmon clock). Exclusive `-n 2` lease, resolve cards by PCI SLOT
  (card3 is the iGPU), restore via trap with verified read-back. `rdna4-rocm7.14` has no `python3`
  — rocprofv3 is a python launcher, so use `rdna4-prof714`. CSV trap: `Counter_Value` is `$(NF-2)`.
- **torch 2.14+rocm7.2 SEGFAULTS on the 7.14 runtime** at first kernel launch, silently, no
  traceback. That is why the harnesses are torch-free — do not try to profile through torch.
- **Power measurement**: sample CONTINUOUSLY on a thread; sampling after a duty step lands in the
  idle trough (a 70% step read 37 W while a 50% step read 121 W). Ramp = a few ms of rising load to
  let DPM engage (a SOFT START), not a duty-limited envelope — holding duty levels for seconds with
  idle gaps makes every burst its own cold start and measured 362 W peak / 312 W mean.
- torch reports **WGPs not CUs** (a 64-CU card answers 32).

## RULES CONFIRMED THE HARD WAY

- Name by function, never `vN`. Seven version-numbered knobs exist and I walked past all of them.
- Develop for everything the kernel SUPPORTS, not what you can see in use today. Two of eight group
  sizes were fast because two were what someone had in front of them.
- An env switch onto a never-optimal path is not a knob, it is a trap — it exists so a losing arm can
  be kept without deciding to delete it. It cost a correctness bug and 11% of throughput here.
- Power draw is not evidence of being at a performance limit; it is evidence of drawing current.
- State which PIPELINE (matrix vs shader) and which PRECISION before any "% of peak" claim.
