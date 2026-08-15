# ROCm 7.2.1 / clang 22  vs  ROCm 7.14 / clang 23 — the toolchain A/B

Collected 2026-08-12. Both sides built from kernels commit `86fed3a`, arch gfx1201, by each image's
own `local/build_local.sh`, so the ONLY variable is the toolchain.

| | image | ROCm | clang | torch |
|---|---|---|---|---|
| A | `minisgl-rdna4:lean` | 7.2.1 | 22.0.0git | 2.14.0.dev+rocm7.2 |
| B | `minisgl-rdna4:lean714` | 7.14 | 23.0.0git | 2.12.0+rocm7.14.0 |

All timing on **card 0 (RX 9070 XT, 64 CU, PCI 0000:03:00.0)** at boost clocks. The two cards are a
mismatched pair, so nothing here may be compared against a card-1 number.

## Files

- `static_resources.txt` — per-kernel VGPR/SGPR/spill/scratch/LDS for all 13 packages, both
  toolchains, from `tools/kernel_static_resources.py`. **No GPU needed** — this is what the compiler
  decided, and it predicted the timing result below before any card was leased.
- `tile_surface_rocm72.{csv,txt}` / `tile_surface_rocm714.{csv,txt}` — the full W4A8 dense tile
  surface: 27 shapes x 15 M x 36 tiles + 3 arms + the chooser's own pick, E2M1 decode,
  CUDA-graph-replay timed. 7,290 cells each. From `tools/w4a8_dense_tile_surface.py --e2m1 --auto`.
- Compare with `python tools/compare_tile_surfaces.py A.csv B.csv --label-a ... --label-b ...`.

## Result

**The upgrade is not uniform, and for the W4A8 served path it is worth nothing.**

| measurement | 7.14 vs 7.2.1 |
|---|---|
| W4A8 tile surface, all 7,290 cells | **1.016** |
| W4A8 as PRODUCTION dispatches it (the chooser's own pick) | **0.9997** |
| w8a8 dense GEMM, 4096³ | **1.105** |
| w8a8 dense GEMM, Muse 2048x19968x6656 | **1.151** |

w8a8 goes 193.3 -> 222.5 TFLOPS on the Muse shape: **49.7% -> 57.2% of the 389.3 TFLOPS FP8 matrix
ceiling**, from a compiler upgrade alone. The static table predicted exactly this and says why:
`w8a8_dense::dense_gemm_tiled_kernel` VGPR 247 -> 231, crossing a granule boundary into 6 waves/SIMD
from 5. The kernel was register-starved; clang 23 unstarved it.

W4A8 gains nothing because it is not register-bound — it is decode/staging bound at ~22-35 VALU per
WMMA-floor instruction and 79-88% wait. Per group size: g=16 1.007, g=32 0.975, g=64 0.964,
g=128 1.020.

## Regressions — read before adopting the image

- **GDN decode loses a third of its occupancy.** `gdn_decode_kernel` VGPR 112-114 -> 169-179,
  occupancy **12 -> 8 waves/SIMD**, on all 24 instantiations. Same for `gdn_decode_conv_kernel`
  (10->8), `gdn_verify_replay_kernel` (10->8, spill in 20), `gdn_decode_replay_kernel` (10->8).
  Package total: 23 kernels gain a step, **123 lose one, spill worse in 54 and better in none.**
  A previous campaign bought +9.58% by killing scratch spill in exactly this path; this hands it
  back. **Measure GDN decode on 7.14 before serving on it.**
- **attn_prefill_paged**: 6 gain, 46 lose. `flash_prefill_paged_kernel` VGPR 165 -> 197, occ 9 -> 7.
- **The high-WARPS_N tiles — the ones that WIN — get slower**: `128x128x4` 0.940, `64x128x4` 0.974,
  while tiles nobody picks get faster (`16x64` 1.317). That is why the surface mean is +1.6% and
  production is +0.0%.

Clean wins besides w8a8: `attn_decode` (14 gain, 0 lose), `mla` (4/0),
`dense_gemm::dense_gemm_pipe_kernel` (VGPR 187 -> 95, occ 8 -> 16, spill better in 48).

## Gotcha this cost an hour

On the 7.14 image **`torch.cuda.device_count()` returns 0 until CUDA is initialised.**
`is_available()` is True, `hipGetDeviceCount` reports 3, `torch.zeros(4, device="cuda")` works — but
the count is 0 until something forces init. On 7.2.1 it returns 2 either way. Any tool that gates on
`device_count()` at startup silently reports "no GPU" on the new image, which reads as a lease or
permissions fault and is neither. Call `torch.cuda.init()` first.

---

# SERVE A/B — the regression is REAL and it is ~20% on the production model

`tools/toolchain_serve_ab.sh`. Same engine worktree, same model, same config, same cards; the only
difference is the image and the kernel packages, both built from `86fed3a` by their own compiler.
Provenance asserted per leg by sha256 of every loaded `.so` (recorded in the logs).

## Qwen3.6-35B-A3B-AWQ, GDN MoE, TP=2, conc=6, 2 reps — THE PRODUCTION SERVE

| section | r72 (7.2.1) | r714 (7.14) | ratio |
|---|---|---|---|
| decode bs=1 | 93.3 tok/s | 92.6 | 0.992 |
| **throughput bs=2** | 142.4 tok/s | 110.1 | **0.773** |
| **throughput bs=6** | 339.0 tok/s | 272.5 | **0.804** |
| **decode @ ctx 6975** | 81.5 tok/s | 72.5 | **0.889** |
| **decode @ ctx 17711** | 80.7 tok/s | 71.5 | **0.886** |
| **prefill @ 10533 tok** | 143012 tok/s | 123030 | **0.860** |
| **prefill @ 21671 tok** | 218513 tok/s | 190734 | **0.873** |

**Do not move the production serve to the 7.14 image.** It costs ~20% throughput at batch, ~11%
decode at depth and ~13% prefill. Only bs=1 is unaffected — which is why a single-stream smoke test
would have passed it.

## Qwen3.5-4B, dense GDN, TP=1, conc=4 — a much weaker signal

| section | r72 | r714 | ratio |
|---|---|---|---|
| decode @ ctx 37 / 6975 / 17711 | 64.3 / 55.1 / 53.0 | 64.4 / 54.9 / 52.4 | 1.00 / 1.00 / 0.99 |
| prefill @ 10533 tok | 7314 tok/s | 6722 | 0.919 |
| prefill @ 21671 tok | 5775 tok/s | 5115 | 0.886 |

GDN decode is FLAT here and regresses 11% on the 35B — the small dense model does not reach the
regime where it bites. **A 4B smoke test is not a substitute for the served model.**

## Why — and it is NOT bandwidth

Device telemetry over each leg (Prometheus, `amdgpu_*_activity_percent`, 10 s samples):

| leg | gfx0 | gfx1 | umc0 | umc1 |
|---|---|---|---|---|
| r72 | 96% | 96% | 30% | 30% |
| r714 | 96% | 96% | 28% | 30% |

**Identical occupancy of the shader engine and identical memory-controller activity, 20% less work
delivered.** The lost time is issue slots, which is exactly what the ISA diff shows. Package-wide
for `gdn_hip`, same source, two compilers:

| | clang 22 | clang 23 | delta |
|---|---|---|---|
| `v_dual_*` (VOPD packed, 2 ops/slot) | 17,051 | 15,469 | **-1,582 (-9.3%)** |
| `v_fmac_f32_e32` (single) | 21,805 | 23,237 | +1,432 (+6.6%) |
| `scratch_*` (real spill traffic) | 1,710 | 2,374 | **+664 (+38.8%)** |
| total instructions | 1,004,319 | 1,008,314 | +0.4% |

clang 23 **un-packs dual-issue pairs** — the same arithmetic, ~1,582 more issue slots — and spills
39% more. On `gdn_decode_kernel` that lands as VGPR 112-114 -> 169-179 and occupancy 12 -> 8
waves/SIMD: more stalls, and fewer waves left to hide them with.

`tools/kernel_isa_diff.py` is the instrument (CPU-only).

---

# RECONCILING WITH THE 2026-07-29 ISA STUDY ("+124 improved, 0 regressed")

An earlier study compiled `fp8_wmma/fp8_wmma_rocm/moe_kernel.hip` under both toolchains and found
clang 23 raised occupancy on 124 of 513 kernels with **zero regressions**, spill-free. Today's
result looks like its opposite. It is not a contradiction — three separate things, all verified:

## 1. SCOPE. The study compiled one translation unit. Its result still holds there.

Filtering today's 13-package comparison to the MoE families that live in `moe_kernel.hip`:
**319 gain / 12 lose**, headlined by `w4a8_tile::moe_gemm_tiled_ashuffle_kernel` VGPR 165 -> 96,
**occ 9 -> 16** — the very kernel and the very step the study recorded. The prediction reproduces.
Every regression is in code the study never compiled: `gdn`, `attn_prefill_paged`, `rxf_fold`, and
the dense `mmq_fp8_gemm_wmma_tiled_tuned_kernel`.

## 2. THE SOURCE MOVED — into exactly clang 23's blind spot.

Ten `gdn/` commits landed after 2026-07-29. One of them, `ef5024a` (**2026-07-31, two days after
the study**), is *"de-scratch the decode recurrent state — Qwen35B 82.5 -> 90.4 tok/s (+9.58% e2e)"*.

`gdn_decode_kernel`, built four ways (same flags, only the source state and the compiler vary):

| source state | compiler | VGPR | scratch (total) | min occ |
|---|---|---|---|---|
| `ef5024a~1` (what the study saw) | clang 22 | 19–22 | 8,448 B | 16 |
| `ef5024a~1` | clang 23 | 25–31 | 8,448 B | 16 |
| `86fed3a` (today) | clang 22 | 111–133 | **0 B** | 10 |
| `86fed3a` (today) | clang 23 | **168–192** | **764 B** | **8** |

Before the de-scratch the recurrent state lived in scratch, both compilers agreed, and there was
**nothing for clang 23 to lose** — the whole gdn package is 8 gain / 0 lose / 0 spill at that
commit, exactly the study's pattern. `ef5024a` then hand-moved that state into registers (VGPR
19 -> 111–133, scratch 8,448 -> 0). **clang 23 cannot hold it**: registers balloon, scratch comes
back, occupancy falls 10 -> 8. The compiler regresses an optimisation that did not exist when the
study ran.

## 3. THE STUDY ALREADY FALSIFIED ITS OWN PREDICTION AT RUNTIME.

Its own later section is titled *"occupancy prediction FALSIFIED, spill elimination is the real
win"*: the `tiled(wmma)` kernel gained occupancy 9 -> 16 and measured **0.97–1.01x — no gain**, while
the real 1.35–1.83x landed on a kernel whose occupancy did **not** move but whose scratch went
132 -> 0. Its conclusion was *"Occupancy was a bad predictor here; ScratchSize was the good one.
**Read scratch first when triaging a toolchain change.**"* It even recorded that
`attn_prefill_paged` had once got **~30% slower when occupancy rose**.

So "+124 improved, 0 regressed" was never a speed prediction, and the metric it warned about is
exactly the one that moved this time: package-wide `scratch_*` traffic **+38.8%** in `gdn_hip`.

## 4. Minor: not literally the same clang 23.

The study measured vanilla `46fcb339`. `minisgl-rdna4:lean714` reports
`46fcb339fb61119b337f973c7ca9e710a319fdd0+PATCHED:440716f8b87be9d8e20ed910e10e5b6d14d57cf6` — the
same base commit plus AMD patches. Not needed to explain the result, but it means "clang 23" is not
one fixed thing across these two measurements.

## The lesson

A toolchain A/B on ONE translation unit, read through ONE metric, at ONE point in the source's
history, generalises to none of the other three axes. What made today's answer different was
measuring **all 13 packages**, reading **scratch and dual-issue packing** rather than occupancy,
against **current** source, and finishing on a **served model**.

---

# WHY clang 23 DID IT — and two hypotheses the data killed

## REFUTED: register pressure does NOT explain the lost dual-issue

The obvious story is that clang 23 allocates more registers, which makes fewer VOPD pairs legal
(`v_dual_*` pairs are formed after RA and are constrained by source-bank collisions). Testable:
kernels whose VGPR rose should be the kernels that lost pairs. Over the 339 `gdn_hip` kernels that
use VOPD at all (`tools/vopd_pressure_correlation.py`):

| register change | kernels | net `v_dual_*` delta | % of their own pairs |
|---|---|---|---|
| vgpr UP | 232 | −794 | **−6.2%** |
| vgpr same | 54 | −3 | −0.2% |
| vgpr DOWN | 53 | −785 | **−26.8%** |

**Pearson r(ΔVGPR, Δv_dual) = −0.049.** No correlation — and the kernels that *shed* registers lost
four times the fraction of their pairs. So these are **two independent regressions** in clang 23,
not one causing the other: it forms fewer VOPD pairs, AND it allocates differently. Do not repeat
the pressure story; it was mine and the data killed it.

## REFUTED as a fix: CU mode

Both shipped builds are 100% WGP (`tools/dump_wgp_mode.sh`: `{'WGP': 339}` of 339 descriptors on
each) — clang 23 did not change the mode, so CU-vs-WGP is unrelated to the regression. Tested
anyway, since higher register pressure is where CU mode might pay. `gdn_decode_kernel`:

| build | VGPR | scratch | min occ |
|---|---|---|---|
| clang 22, WGP (shipped) | 111–133 | **0** | 10 |
| clang 23, WGP (shipped) | 168–192 | 764 B | 8 |
| clang 23, **CU** | 96 | **7,240 B** | **16** |

CU mode buys maximum occupancy by clamping registers and spilling the remainder — package-wide
**183 kernels gain an occupancy step, 0 lose, and 138 spill worse**. For GDN that is the wrong side
of the trade: the +9.58% e2e win this package already banked came from *removing* 8,448 B of
scratch, and CU mode hands 7,240 B of it straight back. CU mode is not the fix here. It remains an
open question for a spill-free, compute-bound kernel — a different experiment.
(The `EXTRA_HIPCC` hook needed to run this now exists in `gdn/local/setup.py`, kernels `d274a7a`.)

## What survives as the mechanism

Two independent clang 23 behaviours, both measured, neither explained by the other:
1. **~9% fewer VOPD pairs** package-wide (17,051 → 15,469), with matching growth in single-issue
   `v_fmac_f32_e32` (+1,432). Same arithmetic, more issue slots.
2. **+38.8% scratch traffic**, concentrated in kernels that were *hand-tuned to hold state in
   registers*. At `ef5024a~1`, before that hand-tuning existed, clang 23 regressed nothing in this
   package.

# WHY OUR RESULT DIFFERS FROM "7.14 IS FASTER"

The public reports are overwhelmingly about **libraries** (hipBLASLt / rocBLAS / CK / attention) and
overwhelmingly about **CDNA**. Neither applies here: this repo ships its **own hand-written RDNA4
HIP kernels**, and the serve overlays them over the image's `/opt/kernels`, so a library improvement
in the image is largely bypassed on the paths we measured. Where the compiler DOES help our code it
helps a lot — **w8a8 dense +15% (193.3 → 222.5 TFLOPS, 49.7% → 57.2% of ceiling)** — and the static
table predicts which kernels those are.

**A confound to state plainly:** the serve A/B swapped the image AND the kernels together, so
`torch 2.14.dev+rocm7.2 → 2.12+rocm7.14`, the ROCm runtime, and `TORCH_BLAS_PREFER_HIPBLASLT` 0 → 1
all moved with the compiler. The kernel-only evidence (torch-free harnesses, ISA, static tables) is
clean and points the same way, but the exact −20% is an image-level number, not a
compiler-only one. Isolating it needs a kernel-only swap, which the `.so`/torch ABI currently blocks.
