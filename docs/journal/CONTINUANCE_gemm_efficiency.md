# CONTINUANCE — W4A8 / W8A8 GEMM efficiency to 50% / 75% of the FP8 matrix ceiling

Sessions of 2026-08-12. Started as "why is Muse-Glimmer prefill slow", became a kernel-efficiency
campaign. Read the CEILINGS section first — the roofline was got wrong twice and every conclusion
built on it inverted.

**Session 2 corrected several things this document previously asserted (marked CORRECTED), found a
silent wrong-answer bug in what session 1 landed, and settled the chooser question by measurement.**

---

## THE TARGET

| kernel | ROCm 7.2.1 | ROCm 7.14 | target | gap from best |
|---|---|---|---|---|
| w8a8 fp8 (no 4-bit decode) | 189.9 (48.8%) | **208.9 (53.7%)** | 75% = 292 | 1.40x |
| w8a8, Muse shape | 193.3 (49.7%) | **222.5 (57.2%)** | 75% = 292 | 1.31x |
| w4a8 e2m1 g=32 | 77.1 (19.8%) | 75.2 (19.3%) | 50% = 195 | 2.5x |
| w4a8 e2m1 g=16 (NVFP4, Muse) | **67.2 (17.3%)** | 67.7 (17.4%) | 50% = 195 | 2.9x |

All on card 0 at boost clocks, 4096³ unless noted, tile 256x128. And on top of any of these, the
tile chooser is leaving a further **1.226x** on the table — see THE CHOOSER.

75% for w8a8 because it has no quantisation overhead — no unpack, no group scales, it is a straight
fp8 GEMM and should sit near the ceiling.

> **CORRECTED:** w8a8 was recorded as 164.1 (42.2%). Re-measured it is 189.9 (48.8%). The old figure
> came from a different session's run; nothing changed in that kernel. This is the reason
> `ab_vs_ref.sh` now exists — see MEASUREMENT DISCIPLINE.

## CEILINGS — get this right or every conclusion inverts

RX 9070 XT (gfx1201), 64 CU @ 2970 MHz boost. **MATRIX** path, dense:

| path | FLOP/CU/clk | boost |
|---|---|---|
| MATRIX FP16/BF16 | 1024 | 194.6 TFLOPS |
| **MATRIX FP8/INT8** | **2048** | **389.3 TFLOPS** |
| MATRIX INT4 | 4096 | 778.5 TOPS |
| SHADER FP32 | — | 48.66 |
| SHADER FP16 (packed a*b+c*d+e) | — | 97.32 |

**97.32 is the SHADER figure, not a matrix ceiling.** A WMMA fp8 GEMM scored against it read as "70%
of peak, the GEMM is fast" and hours went into looking away from the GEMM. Real answer was ~18%.

**There is no power excuse.** Board is rated 370 W, the driver cap is 320 W, and AMD's 389.3 figure
is at the factory 300 W limit. Silicon does ~1.30 TFLOPS/W; our best path does 0.51.

## WHAT LANDED

`/home/pat/code/rdna4-hip-kernels-perf`, branch `perf/gemm-efficiency`, off `b6a0ea1`.

**Session 1 — `fc59982`** (all in `fp8_wmma/fp8_wmma_rocm/w4a8_fp8_wmma_kernel.hip`):

1. **Group-size instantiation lattice.** `TILED_TUNED_BKT` had compile-time instantiations for 128
   and 32 ONLY; the other six accepted sizes fell to `BKT=0` (runtime divide). **g=16 2.28x,
   g=64 2.01x.** The same file's TILE lattice already stated the rule: *"Instantiation count is not
   a constraint here; silent misrouting is."*
2. **`ASHUFFLE_BKT` correctness bug.** `if (group_size==128) ...128; else ...32;` — and `BKT` is
   SEMANTIC, so every non-128 checkpoint was computed as g=32: wrong scale stride, silently WRONG.
   It escaped because every recorded sweep used g in {32,128}.
3. **`VLLM_W7_DIAG` removed.** Five bools gating branches in the staging loop AND the innermost
   WMMA/LDS loop. Deleting it: **+11.0% g=32**, +3.6% g=128, +1.4% g=16.
4. **Staging depth decoupled from group size.** `GPS = TARGET_STAGE_K/GS` groups per stage.
   **g=16 +28.0%, g=64 +10.8%, g=32 +1.9%, g=128 unchanged (GPS=1, the control).**

**Session 2 — `86fed3a`:**

5. **CORRECTNESS: item 4 silently truncated K.** `num_stages = K / BK` was safe while BK *was* the
   group size, because the launcher checks `K % group_size == 0`. Decoupling made the real
   requirement `K % (GS*GPS) == 0` — at g=16 that is `K % 128`, **eight times stricter** — and
   nothing enforced it. The integer divide dropped the tail of K and the kernel returned a truncated
   dot product **with no error**. At K=1248 and K=832 (Muse `down_proj` / `o_proj` at TP=8),
   **99.6% of output elements were wrong.** The last stage is now partial: A is zero-padded past K
   so every WMMA in the pad contributes exactly 0, and pad groups' scales are zeroed so 0 × garbage
   cannot become 0 × Inf.
   Cost, interleaved best-of-3 vs `fc59982`: **1.011 / 0.977 / 0.998 / 1.011** at g=16/32/64/128.
   g=32 pays 2.3%; the NVFP4 g=16 case *gains* 1.1%.
6. **The LDS budget mismatch is fixed and cannot recur.** The staging depth is defined ONCE, in
   `tile_select.h` (`groups_per_stage` / `stage_depth` / `tile_lds`), and the kernel, the launcher,
   the chooser, the standalone harness and the Python sweep all call it. There were **six** copies
   of `(BM+BN)*(group_size+8)`; none counted the static scale array at all.
7. **`TARGET_STAGE_K` was an untested constant.** Swept 32/64/128/256 at 4096³: **128 wins at every
   group size** (g=16: 64.1 / 66.0 / 67.0; g=128 flat at ~82, the GPS=1 control), and 256 does not
   fit LDS at this tile. It stays 128 — now with a measurement behind it and a `-D` override so the
   question can be re-asked. My occupancy hypothesis (deeper stage → 1 block/CU → worse) was
   **refuted**: deeper is monotonically better up to the LDS wall.

**Parity: bit-identical to the pre-campaign kernel (`b6a0ea1`) across all 8 group sizes at
K = 832 / 1248 / 2048 / 4096 / 6656.** Plus an fp32-reference end-to-end check through the real
launcher and chooser at the same K values (`rel_err` ≈ 0.026 uniformly = the fp8 activation
quantisation floor, not truncation).

**The package builds** (`fp8_wmma/local/build_local.sh` inside `minisgl-rdna4:lean`). It could not
before session 2.

## THE CHOOSER — RESOLVED BY MEASUREMENT. It is not mis-fitted; it is FENCED.

Fixing the LDS/rounds model changed **49% of dense picks** (9,848 of 20,064 grid cells; MoE picks
bit-identical over 114,048 — the shared core was not disturbed). That looked like the campaign's
biggest open risk: the model's constants were fitted against a surface measured with
`BK = group_size`, which no longer describes the kernel.

**A full re-sweep says the corrected model is essentially perfect inside its own search space.**
7,290 cells — 27 shapes x 15 M x 36 tiles + 3 arms + the chooser's own pick, E2M1 decode,
CUDA-graph-replay timed on card 0. Fixture: `tools/_fixtures/toolchain_r72_vs_r714/`.

| group size | cells | chooser vs best **WN=1** tile | chooser vs best tile that **exists** | cost of the fence |
|---|---|---|---|---|
| g=16 (NVFP4) | 60 | **0.986** | 1.192 | 1.208x |
| g=32 | 135 | **1.010** | 1.220 | 1.208x |
| g=128 | 15 | **1.000** | 1.490 | 1.490x |
| **all** | 210 | **1.003** | 1.229 | **1.226x** |

The model ranks the tiles it is ALLOWED to pick to within 0.3% of the measured oracle — at g=16
too, which it was never fitted on. **The entire 1.229x gap is `WN_SET = {1}`.** The oracle tile uses
**WARPS_N > 1 in 179 of 210 cells (85%)**.

So the remaining dense prize is not a refit. It is admitting WARPS_N to the search — and solving the
regressions that widening caused when it was last tried (35 of 300 cells, worst 2.35x, on the OLD
surface with the OLD kernel and no E2M1 column; that evidence is now stale in all three respects).

Worst cells today, all of them WN>1 oracles: `g128 lm_head` M=63 **2.10x**, `grid N=11264` M=64
2.01x, `lm_head tp2` M=63 1.98x, `muse.gate_up g32` M=63 1.96x.

**Both prefill arms are the best candidate in 0 of 210 cells** (`prefill_wmma`) and **0 of 150**
(`prefill_wmma_ashuffle`). They never win anything on this surface — see TODO 2.

## COUNTER PROFILE — re-measured after item 4

Per WMMA-floor instruction (floor = FLOP / 16384). Card 0, `profile_standard`:

| kernel | VALU | SALU | LDS | WMMA% of issue | wait% | VGPR | SGPR |
|---|---|---|---|---|---|---|---|
| w8a8 4096³ | 4.6 | 1.0 | 0.7 | 13.7% | 69.0% | 248 | 128 |
| w4a8 g=16 | 34.6 | 11.0 | 2.9 | 2.0% | 79.0% | 152 | 128 |
| w4a8 g=32 | 26.9 | 10.8 | 2.4 | 2.4% | 82.6% | 152 | 128 |
| w4a8 g=64 | 23.8 | 10.7 | 2.1 | 2.7% | 85.1% | 152 | 128 |
| w4a8 g=128 | 22.0 | 10.8 | 2.0 | 2.8% | 87.8% | 160 | 128 |
| w8a8 Muse | 4.1 | 0.9 | 0.7 | 14.8% | 73.9% | 248 | 128 |
| w4a8 g=16 Muse | 34.2 | 11.0 | 2.9 | 2.0% | 73.6% | 152 | 128 |

Item 4 moved g=16 a long way: VALU 55.6 → 34.6, SALU 19.9 → 11.0, LDS 5.1 → 2.9, wait 86.6% → 79.0%.

> **CORRECTED — the SALU story.** The old TODO said item 4 "deleted 7/8 of the loop iterations that
> arithmetic lived in" so the SALU term should collapse. It did not: 13.7 → 10.8 at g=32, and it is
> now **FLAT at ~11 across every group size**. Flat means it is *not* staging-loop address
> arithmetic (that scales with stage count) — it is ~11 scalar ops per WMMA issued, structural.
> Chasing it as a staging artefact will not find it.

**VALU is the dominant term and it is the 4-bit decode.** The g=128 floor of 22.0 is the
int4→e4m3 unpack; the g=16 excess (34.6, +12.6) is the extra per-group scale staging and epilogue.
w8a8, which does no decode, sits at 4.6 and reaches 48.8% of ceiling.

Note `LDS_Block_Size` in the rocprof CSV reports **static LDS only** (4096 B = the wsc array); the
~52 KB dynamic request is not in it. Do not read occupancy off that column.

## THE TOOLCHAIN — ROCm 7.14 / clang 23 is NOT a free upgrade

Same commit `86fed3a` built by both images, timed on the same card. Full detail and the raw surfaces
in `tools/_fixtures/toolchain_r72_vs_r714/README.md`.

| measurement | 7.14 vs 7.2.1 |
|---|---|
| W4A8 tile surface, all 7,290 cells | 1.016 |
| W4A8 **as production dispatches it** (the chooser's pick) | **0.9997** |
| w8a8 dense GEMM 4096³ | **1.105** |
| w8a8 dense GEMM, Muse shape | **1.151** |

**w8a8 is the story: 193.3 -> 222.5 TFLOPS on the Muse shape, 49.7% -> 57.2% of ceiling, for free.**
The static table predicted it before any card was leased —
`w8a8_dense::dense_gemm_tiled_kernel` VGPR 247 -> 231, crossing a granule into 6 waves/SIMD from 5.
It was register-starved (TODO 3 said exactly that) and clang 23 unstarves it. W4A8 gains nothing
because it is not register-bound.

**MEASURED ON THE PRODUCTION SERVE: 7.14 costs ~20% throughput.** Qwen3.6-35B-A3B-AWQ, GDN MoE,
TP=2, conc=6, provenance-asserted per leg (`tools/toolchain_serve_ab.sh`): throughput bs=6 **0.804**,
bs=2 0.773, decode at ctx 17711 **0.886**, prefill at 21671 tok **0.873**. Only bs=1 is unaffected
(0.992) — a single-stream smoke test passes it. On Qwen3.5-4B TP=1 the same test shows prefill
-11% and GDN decode FLAT, so the small model does not reach the regime where it bites.

**And it is not bandwidth.** Both legs sit at gfx 96% / UMC ~30% on both cards and deliver 20%
different throughput, so the lost time is ISSUE SLOTS. The ISA diff says why: package-wide in
`gdn_hip`, clang 23 emits **1,582 fewer `v_dual_*` (VOPD, two ops per slot, -9.3%)** and **+39%
`scratch_*` spill traffic** for +0.4% total instructions. Same arithmetic, more slots, more spill,
and on `gdn_decode_kernel` occupancy 12 -> 8 so there are fewer waves left to hide it.

**Two static regressions, now explained:**
- **GDN decode loses a third of its occupancy.** `gdn_decode_kernel` VGPR 112-114 -> 169-179,
  occupancy 12 -> 8, on all 24 instantiations; the package is 23 kernels up, **123 down, spill worse
  in 54 and better in none**. A previous campaign bought +9.58% killing scratch spill in this exact
  path. Measure GDN decode on 7.14 before serving on it.
- `attn_prefill_paged`: 6 up, 46 down; `flash_prefill_paged_kernel` VGPR 165 -> 197, occ 9 -> 7.

Clean wins besides w8a8: `attn_decode` 14/0, `mla` 4/0, `dense_gemm::dense_gemm_pipe_kernel`
VGPR 187 -> 95 (occ 8 -> 16). All 13 packages build clean under clang 23.

`tools/kernel_static_resources.py` is the instrument — CPU-only, reads what the compiler decided out
of a built `.so`, and it called the w8a8 result correctly in advance. Use it before spending a lease.

## TODO, in order

1. **Widen `WN_SET` — the chooser is fenced, not wrong.** 85% of oracle cells want WARPS_N>1 and the
   fence costs 1.226x geomean / 2.10x worst. Do NOT refit the WN=1 constants; they measure 1.003.
   The last widening attempt regressed 35/300 cells (worst 2.35x) — re-derive that on the CURRENT
   surface, which differs in all three ways that mattered (post-stage-decoupling kernel, an E2M1
   column, and a g=16 column that never existed). The two regressing families to beat were
   `lag.gate_up tp1` K=2048 N=16384 g=32 M=17..64 and the lm_head at M<=32.
   While sweeping, also vary **TARGET_STAGE_K jointly with the tile**: depth 256 was excluded only
   because 256x128 does not fit — a narrower tile at depth 256 does (128x64 needs 54,784 B).
2. **Retire the two dead standalone prefill kernel BODIES** —
   `mmq_fp8_gemm_prefill_wmma_kernel` (:323) and `..._ashuffle_kernel` (:682) — together with
   `VLLM_W4A8_DENSE_TILED_OFF`. They are reachable *only* behind that opt-out; both arms default to
   the `gemm_tiled*` re-expressions, which are bit-exact (`test_dense_tiled_bitexact.py`, 32/32
   max|diff|=0) and perf-neutral. They are the decoys that silently absorbed three edits last
   session because their loop bodies match the tiled kernel's.
   > **CORRECTED — do NOT delete the DK arms themselves.** The old TODO said `DK::PrefillWmma` and
   > `DK::PrefillWmmaAshuffle` were unreachable because minisgl's `_pick_dense_kernel` returns only
   > `wmma_tiled_tuned`/`decode_gemv`. That is true of minisgl and **false of the box**: they are
   > live in vllm-gfx1201 — `w4a8_vllm/vllm_adapter.py:335-338` (the per-M ladder),
   > `:201-202` (the load-time autotune probe), and `mxfp4_linear.py:38`. Deleting the arms breaks
   > vLLM's W4A8 path. Delete the bodies, keep the arms.
   Also stale and worth fixing while there: `gemm_tiled.h:17` and `.hip:995` still say the tiled
   re-expression is enabled *by* `VLLM_W4A8_DENSE_TILED` and is "NOT the served default" — inverted
   relative to the shipped `_OFF` polarity. `tests/w4a8_dense_dispatch_test.py` asserts arms
   `_pick_dense_kernel` can no longer return, and `tools/_midband_serve_inner.sh:80-84` patches an
   anchor that no longer matches.
3. **w8a8 to 75%.** The closest target and now partly collected: **57.2% on ROCm 7.14** (was
   49.7%), needs 1.31x more. The mechanism is confirmed — it is REGISTER-BOUND, and the whole 7.14
   win is one granule crossing (VGPR 247 -> 231, 5 -> 6 waves/SIMD). So keep pulling that lever
   deliberately rather than waiting for compilers: 231 VGPR still buys only 6 of 16 waves/SIMD, and
   the next boundary is 219 (7 waves) then 192 (8). With 61,440 B of static LDS it is also 1
   block/CU, so LDS is the other half. Then prefetch distance and `s_waitcnt` placement in the
   double-buffer. Independent of the W4A8 work, and the cheapest remaining win in the campaign.
4. **The 4-bit decode VALU term.** 22 VALU per WMMA-floor instruction is the structural cost of
   unpacking int4/e2m1 into e4m3 in the staging loop. This is where the remaining W4A8 gap lives —
   not in SALU (flat, structural) and not in the matrix math (2% of issue). Ask whether the decode
   can move off the critical path (wider loads, `v_perm`-style byte assembly, or decoding straight
   into the WMMA operand layout rather than through LDS).
5. **Finish the env triage.** `V7_CFG` → explicit argument (the op already takes `kernel=`);
   `V7_SWIZ` → unconditional if it always wins, else a cost-model input; `TILE_CU` → derive it
   (it exists because torch reports WGPs not CUs); `TILE_EXPLAIN` → delete, redundant with the
   exported `dense_tile_explain`; `MAGIC`/`MAGIC_COLS` → promote or delete;
   `DENSE_SMALLM_OFF`/`DENSE_TILED_OFF` → bake-off toggles whose bake-off concluded;
   `DECODE_GEMV_BK`/`NW` → argument or cost model. **No version numbers in kernel names or knobs** —
   V7/V10/V17 all violate that rule. (`V17_*` is on the regdirect W4A16 launchers, a *different*
   live family — rename, do not delete.)
6. **Serve-level validation** on Muse-Glimmer: the isolated g=16 number independently reproduced the
   figure derived from live serve prefill timings, so the microbench measures the real thing.
   Expect prefill ~680 → ~2000 tok/s. Do this only after TODO 1, since the chooser now picks
   differently on every Muse shape.
7. **Decide on the 7.14 image per-package, not wholesale.** It is a clear win for w8a8 dense,
   attn_decode and mla; a wash for W4A8; and a measured regression for GDN decode and
   attn_prefill_paged. The build is proven (13/13 packages compile under clang 23 from `86fed3a`),
   so the open question is only whether the GDN decode occupancy loss shows up in serve latency.

## MEASUREMENT DISCIPLINE — earned twice this campaign

- **Never compare against a number from a previous session.** The "w8a8 is 164.1" and "the guards
  cost 4.5%" claims were both artefacts of doing exactly that; interleaved, the real numbers are
  189.9 and 2.3%. `local/efficiency/ab_vs_ref.sh` builds the reference kernel from a git ref into a
  second binary and alternates them under one lease. Use it.
- **Parity proves "same as before", not "correct".** The K-truncation bug passed every parity run
  for a session because both sides were tested only at K=2048, where the stage happened to divide K.
  Parity needs shapes chosen to *break* the invariant, not shapes that are convenient.
- **A tool that cannot express a case cannot catch it.** The harness covered group sizes
  {0,16,32,64,128} — the same "two of eight sizes are what someone had in front of them" gap the
  kernel was being fixed for, reproduced in the tool meant to catch it. It now covers all eight.
- `iterate.sh`'s build gate was "does the binary exist", with the compile `|| true`-d behind an
  error grep — so a failed build printed `HARNESS OK` over yesterday's binary. It now removes the
  binary first.

## TOOLING (`rdna4-hip-kernels-perf/fp8_wmma/local/efficiency/`)

- `iterate.sh <scratch> [TARGET_STAGE_K]` — re-extract kernel core + rebuild harness in SECONDS
  (vs ~9 min for the package). **This is the loop to work in.**
- `extract_parity.sh <scratch> [ref]` — builds both sides of `parity.hip` from real checkouts.
- `ab_vs_ref.sh <scratch> [ref]` — interleaved best-of-N A/B against a git ref.
- `sweep_stage_target.sh <scratch>` — TARGET_STAGE_K × group size.
- `measure_after_restage.sh <scratch>` — parity + timing lattice + counters, one lease, perf level
  pinned only around the counter pass.
- `counter_table.py <scratch>` — rocprof CSVs → the issue-budget table above. Host-side, no GPU.
- `w4a8_counters.hip` / `gemm_counters.hip` — standalone W4A8 / W8A8 harnesses. They compile the
  REAL kernel template (extracted verbatim), so a win here is a win in the package.
- `parity.hip` — base vs worktree, byte compare, all 8 groups. Takes `M N K`; **vary K**.

In `minisgl-rdna4/tools/`:
- `kernel_static_resources.py` — per-kernel VGPR/SGPR/spill/scratch/LDS out of a built `.so`.
  **CPU-only**, and it predicted the w8a8 7.14 win before a card was leased. Two `.so` files diffs
  them by kernel family. Reach for this BEFORE spending a lease on a toolchain or codegen question.
- `compare_tile_surfaces.py` — two surface CSVs -> per-candidate / per-shape geomean, whether the
  ORACLE TILE MOVES, and what keeping the old oracle costs.
- `_w4a8_tile_policy.py` — the one Python statement of the tile LDS footprint, cross-checked against
  the built package by `assert_matches_package()`. The four CSV-analysis tools import it.

Also: docker image `rdna4-prof714:latest` (ROCm 7.14 runtime + lean `/opt/venv`).

## GOTCHAS THAT COST TIME

- **On the ROCm 7.14 image `torch.cuda.device_count()` returns 0 until CUDA is initialised.**
  `is_available()` is True, `hipGetDeviceCount` reports 3, and `torch.zeros(4, device="cuda")` works
  — the count alone is 0. On 7.2.1 it is 2 either way. Every tool that gates on `device_count()` at
  startup therefore reports "no GPU" on the new image, which reads as a lease or permissions fault
  and is neither. Call `torch.cuda.init()` before enumerating.
- **`GROUPS` is a bash built-in** (the user's supplementary group IDs). `GROUPS=(16 32 64 128)` is
  silently ignored and your loop iterates over group *IDs*. Cost one wasted lease.
- **Edits land in the wrong kernel.** `prefill_wmma` appears EARLIER in the file and its loop body
  matches the tiled kernel's; `str.replace(x, y, 1)` hits it first. Verify against
  `grep -n "^__global__"` after editing. (TODO 2 removes the decoy.)
- **torch hipifies**: edit `w4a8_fp8_wmma_kernel.hip`; `*_hip.hip` is a generated build artifact
  (root-owned). Verify your change appears in the generated file.
- **gpu-lease + docker**: wrap in `bash -c '...'` with SINGLE quotes so `$HIP_VISIBLE_DEVICES`
  expands INSIDE the lease shell. Bare `-e VAR=$VAR` expands in your shell to empty → "No HIP GPUs".
- **`python /path/script.py` puts the SCRIPT's dir on `sys.path`, not the cwd** — so a script run
  from `torch-ext/` still imports the *installed* `fp8_wmma`, not the one just built. Set
  `PYTHONPATH`, or the freshly built op is silently absent.
- **Counters**: need ROCm 7.14 rocprofv3 + `power_dpm_force_performance_level=profile_standard`
  (RDNA4 `auto` gates the perfmon clock). Exclusive `-n 2` lease, resolve cards by PCI SLOT
  (card3 is the iGPU), restore via trap with verified read-back. `rdna4-rocm7.14` has no `python3` —
  rocprofv3 is a python launcher, so use `rdna4-prof714`. rocprofv3 also profiles the runtime's own
  `fillBufferAligned` memsets; filter them out or SALU is inflated by a constant that looks real.
- **torch 2.14+rocm7.2 SEGFAULTS on the 7.14 runtime** at first kernel launch, silently, no
  traceback. That is why the harnesses are torch-free.
- **Power measurement**: sample CONTINUOUSLY on a thread; sampling after a duty step lands in the
  idle trough. Ramp = a SOFT START, not a duty-limited envelope.
- torch reports **WGPs not CUs** (a 64-CU card answers 32).

## RULES CONFIRMED THE HARD WAY

- Name by function, never `vN`. Seven version-numbered knobs exist.
- Develop for everything the kernel SUPPORTS, not what you can see in use today. Two of eight group
  sizes were fast because two were what someone had in front of them — and then the *harness*
  repeated the same mistake.
- An env switch onto a never-optimal path is not a knob, it is a trap.
- Power draw is not evidence of being at a performance limit; it is evidence of drawing current.
- State which PIPELINE (matrix vs shader) and which PRECISION before any "% of peak" claim.
- **A number that is duplicated will diverge.** The staging depth was written in four places and had
  drifted in three of them within one session; the LDS formula in six. If two files must agree about
  a number, one of them has to *call* the other.
- **Check who else consumes the thing you are deleting.** "Unreachable" was established from one
  repo's dispatch table; the arms were live in another repo on the same box.
