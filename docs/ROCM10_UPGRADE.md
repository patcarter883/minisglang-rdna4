# ROCm 10.0 upgrade — evaluation

Measured 2026-09-16. Supersedes the "clang 23 costs 20%" conclusion in
[[clang23-regresses-hand-descratched-kernels]] **for ROCm 10.0 specifically** — 10.0's clang 23 is
not 7.14's clang 23.

## The stack is ready and the Dockerfile change is small

| | |
|---|---|
| ROCm 10.0.0 | released 2026-08-26; Radeon RX 9000 / gfx1201 fully supported |
| Actual toolchain | **HIP 7.15.26333, clang 23.0.0** — "10.0" is a marketing number |
| Base image | `rocm/dev-ubuntu-26.04:10.0.0-full` (and `-24.04` if staying on 24.04) |
| torch | pytorch.org `nightly/rocm10.0`, cp312 **and** cp314: `torch-2.15.0.dev20260915+rocm10.0` |
| py3.14 deps | all 19 fine — 11 pure-python, 6 cp314 wheels, `tokenizers`/`safetensors` via **abi3** |
| Breaking changes | rocSPARSE index type; **AMD SMI ABI** — check `gpu-lease` / monitoring |

`rocm/pytorch:rocm10.0_*` images are runtime-only: **no rocThrust** (`thrust/complex.h` absent
system-wide), so they cannot build our kernels. Use a `dev-ubuntu-*` base + pip torch, which is what
the Dockerfile already does.

## The 20% regression does NOT reproduce on 10.0

Package-wide ISA, `gdn_hip`, one source commit built by each compiler (md5-asserted distinct
binaries):

| metric | clang22 | clang23 (10.0) | ratio | 7.14 ratio |
|---|---|---|---|---|
| `v_dual_*` (VOPD) | 14,564 | 17,412 | **1.196** | 0.907 |
| `scratch_*` | 1,278 | 1,917 | **1.500** | 1.388 |
| `v_fmac_f32` | 20,308 | 18,024 | 0.888 | (+1,432) |
| total instructions | 841,241 | 843,376 | 1.003 | 1.004 |

**The VOPD un-packing is fixed and reversed.** 7.14 packed 9% FEWER dual-issue ops; 10.0 packs 20%
MORE than clang 22. That was the dominant mechanism behind the 0.804 throughput ("same arithmetic,
more issue slots"), and it is gone.

**The spill is worse**, 1.50x vs 7.14's 1.388x, and it is NARROW — 49 kernels gained scratch from
zero and they are dominated by one templated body:

    +40  gdn_decode_conv_kernel<BFloat16,Half,128,128>
    +40  gdn_decode_conv_kernel<Half,Half,128,128>
    +28  gdn_decode_conv_kernel<*,0,0>   (x4 instantiations)
    +26  gdn_decode_kernel<Half,Half,128,128>
    kernels with any scratch: 108 -> 157

`gdn_decode_kernel<double,float,128,128>` itself has ZERO scratch in both legs; its deltas are
scheduling (waitcnt +22%, total +8%), not spill.

## MEASURED: the serve A/B (2026-09-16, conc=6, Qwen3.6-35B-A3B-AWQ, TP=2)

Two images, each serving kernels built by ITS OWN compiler, provenance asserted by comparing the
loaded `.so` hashes across legs (they differ). 3 reps per leg.

**The 20% regression is gone.**

| metric | clang22 | ROCm 10.0 | ratio | 7.14 was | verdict |
|---|---|---|---|---|---|
| throughput bs=1 | 94.3 | 95.0 | 1.007 | — | inside noise |
| throughput bs=2 | 145.3 | 142.0 | **0.977** | 0.773 | OUTSIDE noise |
| throughput bs=6 | 334.1 | 328.4 | **0.983** | 0.804 | OUTSIDE noise |
| decode @ctx17711 | 78.3 | 80.2 | 1.024 | 0.886 | inside noise |
| TPOT ms/tok | 10.58 | 10.50 | 1.008 | — | — |
| prefill @3375 (cold) | 3463 | 3654 | **1.055** | — | — |
| prefill @10533 (cold) | 5376 | 5625 | **1.046** | — | — |
| prefill @21671 (cold) | 5423 | 5673 | **1.046** | 0.873 | — |

Run-to-run spread on the throughput rows is ~1.0-1.5%, so bs=2/bs=6 at -2.3%/-1.7% are small but
REAL; bs=1 and deep decode are genuinely neutral (the ctx17711 row's own spread is 8.7%).

**Net: prefill ~+5%, deep decode neutral, concurrent throughput ~-2%.** A wash-to-slightly-positive
trade, against 7.14's uniform 11-23% loss.

### The remaining -2% is the spill, and it is the fix target

The ISA said VOPD packing +19.6% and scratch +50.0%. The serve says the VOPD gain shows up as the
prefill and deep-decode wins, and the spill shows up as the concurrency loss — concurrency is where
occupancy matters most. That points the fix at the same place the ISA did: `gdn_decode_conv_kernel`
and the other 48 kernels that gained scratch from zero.

### MEASUREMENT TRAP: prefill tok/s is only valid on the COLD rep

`serve_perf.py` reuses its prompts across reps, so reps 2+ hit the radix prefix cache, TTFT
collapses, and "prefill tok/s" reads as 250,000-450,000:

    @21671 clang22 reps = [5423, 411434, 383613]
    @21671 ROCm10  reps = [5673, 437851, 445321]

A median over reps is meaningless for that metric. Use rep 1 only, or give every prompt a fresh
nonce. Taking the median here would have reported prefill as 1.141x and 0.904x on adjacent rows.

## What is NOT yet known

* **Net throughput.** VOPD up is good, spill up is bad; which wins is not derivable from the ISA.
  Needs `tools/toolchain_serve_ab.sh` at **conc=6** — bs=1 measured 0.992 on the 7.14 regression and
  would have passed it.
* **Only `gdn` was built.** `fp8_wmma`, `moe`, `attn_*`, `qsa_index` need the same package-wide diff.
  gdn is where the 7.14 regression was localised, so it is the right first probe, not the whole answer.
* **Whether 10.0 fixes the ROCr idle spin** (`docs/ROCR_IDLE_SPIN.md`) — untested; needs a GPU run.

## Fix target for the remaining spill

One kernel body, `gdn_decode_conv_kernel`, across its instantiations. Cheapest routes first:
`__launch_bounds__` / `amdgpu-waves-per-eu` to give the allocator room, then live-range restructuring.
This is the work [[clang23-regresses-hand-descratched-kernels]] describes as "hand de-scratching",
now needing a redo against clang 23's allocator — but scoped to one kernel family, not the surface.

## Method notes (both cost real time)

* `build_ext` **re-copies `build/lib.../*.so` without compiling**. Two legs came out byte-identical
  (same md5) twice before this was caught. `git clean -xfd` did not clear it, and the artefacts are
  root-owned by the container. Clean inside the container, and **always md5 the two legs before
  believing a compiler A/B**.
* zsh aborts a whole command on a failed glob (`rm -rf dir *.so` with no `.so` match runs nothing).
