# NVFP4's two-level scale as a kernel POLICY — e4m3 block scale + per-output-channel f32 global

**Date:** 2026-09-05 · **Kernels:** `/home/pat/code/rdna4-hip-kernels-e4m3` branch `feat/e4m3-scale-policy` (base `169be57`) ·
**Engine:** `/home/pat/code/minisgl-rdna4-e4m3` branch `feat/e4m3-scale` (base `5e88111e`) ·
**Hardware used: NONE.** Every number below is CPU-side — a compile, an fp64 golden over raw
checkpoint bytes, host arithmetic, or a disassembly diff. A hyper-connection fusion workflow owned
both gfx1201 cards for the duration; **all on-device validation is deferred and enumerated in §4.**

---

## Executive summary

**Is the e4m3 policy implemented? Yes.** `w4a8_tile::E4m3GroupScaleGlobal` is a new WScale policy in
`tile_config.h` alongside `Fp16GroupScale` (the shipped behaviour restated as a policy). The two 4-bit
loaders became `Int4Fp8LoaderT<WSP>` / `Int4Fp8GemvLoaderT<WSP>`; three hand-written non-WLoad
`__global__` bodies were templated instead of copied; the engine emits, merges, stacks and routes the
two levels end to end.

**Does it compile? Yes — MEASURED.** A from-scratch build of the kernels worktree inside
`minisgl-rdna4:m1b-20260903` (`local/build_local.sh`, `MAX_JOBS=8`, `--offload-arch=gfx1201`, 7 TUs):
**exit 0, 405 s wall, zero compiler warnings or errors**, `fp8_wmma_C…so` = 101,972,400 B, and a
second independent clean build reproduced it **byte for byte** (sha256 `65ed58b3…`, 321 s). The
CPU-only policy-contract probe (`local/wscale_policy_contract.hip`, 147 lines — a `static_assert`
lattice plus an exhaustive host sweep) compiles and **PASSes**.

**Is it measurably more accurate than the fp16 fold? Yes — MEASURED, on real bytes, against an fp64
golden.** Across 27 real expert tensors (layers 0/23/47 × experts 0/7/300 × gate/up/down of
`Qwen3.8-Flash-Next-NVFP4`): the fp16 fold carries **2.671e-04 … 4.370e-04 max** relative error
(mean 1.425e-04 … 2.616e-04); the two-level split carries **3.625e-08 … 5.517e-08 max**. That is a
**5,618× … 12,003×** reduction elementwise and **682× … 1,185×** through a GEMV on per-row relative
L2. All 27 tensors, no exceptions.

**And it is smaller — MEASURED through `weights/sizing.py`, not asserted:** the 48-layer expert stack
goes **70.3125 GiB → 63.6328 GiB, a 6.6797 GiB net saving (9.500%)**, exactly the briefed 6.68 GiB.

**But it cannot yet run on a GPU**, and not only because no card was available: **two kernel-binding
predicates still reject the new layout** (§4.1). Both are small, both are named, neither was papered
over. Nothing in this branch has executed on gfx1201.

---

## 1. The consolidation debt, paid

### 1.1 What was templated

The brief's item (a) was the highest-risk piece: two **hand-written, non-WLoad** `__global__` bodies
that hardcoded `const __half* __restrict__ w_scales` and carried their own inlined nibble decode.
They are the NVFP4 **prefill** gemm1 — the flag arm is gated to group ∈ {32,64,128}, so group-16
NVFP4 *always* lands here. Copying them is precisely what `KERNEL_CORE_POLICY.md` forbids.

| File / kernel | Before | After |
|---|---|---|
| `moe_kernel.hip:348` `moe_gemm1_silu_alds_kernel` | `template<AT,BN>`, hardcoded `const int* w_packed` + `const __half* w_scales`, own inlined int4/e2m1 decode | `template<AT,WLoad,BN>` over `WLoad::WT/WScaleT/ws_expert/wz_expert`, staging through the **shared** `WLoad::stage_b`, folding through `WLoad::group_wscale`, `WLoad::wscale_epi` in the epilogue |
| `moe_kernel.hip:500` `moe_gemm1_silu_ashuffle_kernel` | same | same (this is the served default) |
| `moe_kernel.hip:238` `moe_gemm_scalar_kernel` | golden reference, also hardcoded `__half` | `template<AT,SCATTER,WSP>` — the reference can now check the format the fast paths added |
| `moe_gemm_tiled.h` `Int4Fp8Loader` | one loader | `Int4Fp8LoaderT<WSP>`; aliases `Int4Fp8Loader = <Fp16GroupScale>`, `Nvfp4E4m3Loader = <E4m3GroupScaleGlobal>` |
| `gemv_decode.h` `Int4Fp8GemvLoader` | one loader | `Int4Fp8GemvLoaderT<WSP>`; aliases `Int4Fp8GemvLoader`, `Nvfp4E4m3GemvLoader` |

`stage_b`/`group_wscale` gained a defaulted `n_bound`, decoupling the **channel bound** from the
`(G,N)` row **stride** — that is what makes the fused gate|up kernels callable with any `WLoad`.

**Item (b) — the SILU epilogue hook that did not exist.** `gemv_decode.h:1588` never called
`wscale_epi` at all; it assumed the identity, which held only because every loader instantiated with
`SILU=true` happened to return `1.0f`. It now calls it on **both** halves — gate at `j`, up at
`j+inter` — hoisted out of the row loop and applied **before** `silu_and_mul`, because silu is
non-linear (`silu(s·g)·(s·u) ≠ s²·silu(g)·u`) and gate/up are differently-scaled matrices. Both fused
epilogues in `moe_kernel.hip` took the same two-sided hook before `moe_silu_and_mul_h`.
The signature widened to `wscale_epi(ws_e, wz_e, abs_n)` across **all 8 loader definitions** and
**9 call sites** (`moe_gemm_tiled.h` ×5, `gemv_decode.h` ×2, `moe_kernel.hip` ×2, `moe_gemm_flag.h` ×1)
— the epilogue needs the `w_zeros` slot to reach the global.

**Item (c) — four accum bodies became one text.** `accum()`, `accum_bylane()`, `accum_pf<PF>()` and
`accum_bylane_pf<PF>()` are now four **loop shapes** over one `consume_chunk()`. Three of them
carried their own inlined `__half2float(ws[...])` + nibble decode + group fold (~90 duplicated lines
each). `gemv_decode.h` is **+149/−214 lines** — the file got *smaller* while gaining a format.

### 1.2 The policy itself

`w4a8_tile::E4m3GroupScaleGlobal` in `tile_config.h` (+64/−0). Contract: `using ScaleT`,
`decode(p,off)`, `wz_base(z,e,N,ng)`, `epi(wz_e,nc)`, and two `constexpr` flags `uses_zeros` /
`has_global`. For e4m3: `ScaleT = unsigned char` (the 1-byte block scale, same `(E,G,N)` group-major
layout), `decode = e4m3_to_f32`, `epi` = the per-output-channel f32 global read out of the `w_zeros`
pointer slot. **No op-schema change** — NVFP4 is symmetric so that slot is null; precedent is
`NlInt8Loader`, which already threads the RXF NL codebook through it. `uses_zeros=false` is
`constexpr`, so the zero-point read is dead-code-eliminated under this policy.

### 1.3 Instantiation delta — MEASURED, and larger than the brief projected

Device-symbol census of the changed offload bundle (`.so.1.hipv4-amdgcn-amd-amdhsa--gfx1201`),
baseline `.so` vs policy `.so`, both built in the same image:

| | symbols | device instructions |
|---|---:|---:|
| baseline (fp16 only) | 797 | 2,789,118 |
| after | 1,301 | 4,962,339 |
| — of which **new e4m3** | **504** | 2,191,480 |
| — of which **fp16** | **797** | **2,770,859** |

**+504 symbols = +63.2%**, not the briefed ~+33%. Breakdown of the 504: `gemv_decode_core` 336,
`moe_gemm_tiled_kernel` 68, `moe_gemm_tiled_ashuffle_kernel` 68, `moe_gemm1_silu_ashuffle_kernel` 14,
`moe_gemm1_silu_alds_kernel` 14, `moe_gemm_scalar_kernel` 4. The overshoot is the decode-GEMV tiling
lattice (`MMAX × cols × PF × bylane`), which multiplies 336 ways where the brief costed the tiled arm
only. Binary cost: 87,214,520 → 101,924,000 B (+16.9%); 4 of the 5 other device bundles are
byte-identical in size, i.e. the growth is confined to the one TU.

### 1.4 Is the fp16 path behaviourally unchanged? — a qualified YES, and the qualification matters

**MEASURED:** the fp16 arm retains **exactly its 797 symbols** — nothing was lost, renamed away, or
silently dropped — and its total device instruction count moves by **−18,259 (−0.655%)**.

**MEASURED, and it refutes a stronger claim:** the fp16 ISA is **not** byte-identical. Comparing one
matched pair symbol-for-symbol (`gemv_decode_core<__half, Int4Fp8GemvLoader…,4,8,1>` vs
`…Int4Fp8GemvLoaderT<Fp16GroupScale>…`, addresses and encodings stripped): 3,005 → 2,992
instructions, differing throughout in **register allocation and scheduling** (`v32→v1`, reordered
`v_dual_mov` pairs), not in opcode mix. Across the whole bundle only 7/797 bodies hash identically.

So "the fp16 path is unchanged" is a **source-equivalence argument plus a compile-time contract, not
an ISA-identity proof**. `wscale_epi` returns a literal `1.0f` under `Fp16GroupScale` and
`uses_zeros` is `constexpr`, so the added hooks fold away — but the compiler re-allocated around
them. **Only an on-device bit-exactness A/B of the fp16 MXFP4/W4A8 MoE path can close this**, and it
is listed in §4.

### 1.5 The CPU-only contract probe — a real check, not a smoke test

`local/wscale_policy_contract.hip` (147 lines, new, needs no GPU) does two things and **PASSes**:

1. **Compile-time contract.** `static_assert`s that every WLoad riding a WScale policy agrees on
   element types, on the zeros-vs-global meaning of the `w_zeros` slot, and that the two 4-bit
   loaders differ *only* in that policy. Re-forking either loader stops compiling — which is the
   point.
2. **Exhaustive host sweep** over all 256 e4m3 byte patterns × 16 E2M1 codes × a global sweep:

| check | result |
|---|---|
| E2M1 × e4m3 exact in f32 | **4064/4064 pairs, 0 inexact** |
| e4m3 values that fail to round-trip fp16 | **0/254** — the fold's loss is *not* a storage-width problem |
| fp16(block×global) vs two levels, **fp16-normal** products | **636,624 / 846,968 lossy**, max **4.880e-04**, mean 1.563e-04 (the 2⁻¹¹ bound is 4.883e-04) |
| products below fp16's normal minimum (6.104e-05) | **185,224**, max rel-err **5.882e-02** — the fold *denormalises* the scale; two levels cannot |

That last row is new information the brief did not have: on a checkpoint with smaller globals the
fold's error is not 4e-04 but up to **5.9e-02**.

---

## 2. Accuracy against the fp64 golden

`tests/qwen4exp_nvfp4_golden_test.py` (577 lines, new) builds an fp64 dequant
`E2M1_LUT[nibble] × f64(e4m3_byte) × f64(global)` straight off the **raw safetensors bytes** at
`/home/pat/.cache/hf-q4e`, with **no dependence on the code under test**: both decode tables are
constructed in the test from the OCP spec by bit arithmetic, then cross-checked against the repo LUT
(16/16 codes), against torch's own float8 bitcast (254/256 finite bytes + identical NaN masks), and
against `mxfp4.unpack_e2m1_nibbles` on 1,638,400 real layer-0 codes. **Result: ALL CHECKS PASSED.**

### 2.1 Leaf tensors (27 of them, MEASURED)

| metric | fp16 fold | two-level split | ratio |
|---|---|---|---|
| elementwise rel-err **max** | 2.671e-04 … **4.370e-04** | 3.625e-08 … 5.517e-08 | 5,618× … 12,003× |
| elementwise rel-err **mean** | 1.425e-04 … 2.616e-04 | — | — |
| 4-row GEMV, per-row relative L2 | 1.688e-04 … 3.016e-04 | 2.405e-07 … 2.634e-07 | 682× … 1,185× |

Worst case is layer-0 gate/up at fold max 4.370e-04 vs split 4.202e-08 — **10,402×**.

**Two corrections to the brief, both MEASURED:**

- The brief's mean range "1.4–2.4e-04" is **slightly understated**: layer-0 gate/up sit at
  **2.61e-04**, above the quoted ceiling. True range 1.425e-04 … 2.616e-04.
- The brief says the split is "**EXACT — 0.000e+00**". **It is not literally exact.** Worst observed
  is **5.517e-08 ≈ 2⁻²⁴·⁵**: the f32 product of a 6-bit (E2M1 × e4m3) significand and a 24-bit f32
  global needs ~30 bits, so **one** rounding survives. The accuracy argument is untouched — 5.5e-08
  vs 4.4e-04 is four orders of magnitude — but the word "exact" should read "exact to f32 round-off".
- The brief's GEMV figure "1.2e-03–3.2e-03" **reproduces only under its own metric and is not a
  stable number.** It is the *pointwise* relative error of a literal 4-row GEMV, dominated by outputs
  that land near zero by cancellation: over 32 seeds on one tensor it spans **3.6e-04 … 1.7e-01**
  (median 1.26e-03, p90 4.5e-03). The test therefore **gates on the norm metric** and prints the
  pointwise figure for continuity only. Worth understanding *why*: the per-group fp16 rounding is
  **systematic** — all 16 weights of a group share one rounded scale — so it carries through a
  reduction essentially undiluted instead of averaging away.

### 2.2 Absolute anchors — a relative metric cannot catch a sign error

`amax == FP4_E2M1_MAX × max_block_scale × global` holds **exactly** on all 27 tensors, and **9 of
them saturate e4m3** (ratio == 2688 to the bit) — only possible if all three scale levels decode
correctly. Dequantised expert |w|mean = 0.010417 against the bf16 shared expert's 0.007061
(ratio 1.475), i.e. the right order of magnitude, not 4.2e6.

### 2.3 Negative controls — every one of them red

| mutation | rel-err (correct split = 4.202e-08, fp16 fold = 4.370e-04) |
|---|---|
| direction inverted (divide, not multiply) | **2.316e+07** — the *documented* sign is the wrong one for this producer |
| global dropped entirely (unwired `w_zeros`) | **4.811e+03** |
| block scale value-converted (`.to(f8)`) instead of bitcast (`.view(f8)`) | **1.043e+01** |
| global collapsed to a container-wide scalar | **5.000e-01** |
| per-expert global offset dropped (expert-0's global for all 8) | **1.409e+00** vs correct 5.847e-08 |

### 2.4 The merge constraint — VERIFIED, not assumed, two ways

**(a) On real bytes.** `tests/qwen4exp_loader_test.py` loads layers 0/1/3 and asserts the
merged+stacked global is exactly `(512, 1280)` f32 for gate_up and `(512, 2560)` f32 for down_proj,
with **zero** emitted-but-not-declared, **zero** shape and **zero** dtype mismatches across all
**966 emitted parameters (13.19 GiB)**. Block scale lands as `(512, 1280, 160)` **float8_e4m3fn** —
group-16, not 32 — and no raw per-tensor global survives.

**(b) In unit form, on a case the real bytes cannot prove.** Census over all 512 experts of layers
0/23/47: `gate_proj` and `up_proj` share the **same** `weight_scale_2` in **1536/1536** cases
(4 distinct values per layer); `down_proj` has **281 / 132 / 261** distinct per-expert values. So a
merge that *dropped* up's global would be invisible to a value check on these bytes.
`tests/core/test_nvfp4_two_level_scale.py` therefore drives the loader's **own** `_gate_up_merge` /
`_get_expert_stack_info` / `_ExpertStacker` (imported, not re-implemented) with **different** globals
on gate and up, and asserts the merged vector is constant on each contiguous output-channel range.
The N-vector needed **no downstream special case**: `torch.cat(dim=0)` and the expert stack carry it.

### 2.5 Host test results — MEASURED

| suite | result |
|---|---|
| `tests/core/test_nvfp4_two_level_scale.py` + `tests/core/test_weight_sizing.py` | **59 passed, 0 failed** |
| `tests/qwen4exp_nvfp4_golden_test.py /model` | **ALL CHECKS PASSED** (exit 0) |
| `tests/qwen4exp_loader_test.py /model` | **exit 0** |
| `local/wscale_policy_contract.hip` | **PASS** |
| `tests/qwen4exp_fulldepth_test.py` | **NOT RUN — needs a GPU** (see §4) |

*Runner note:* the container image ships **no pytest**, so the two `tests/core/` files were executed
through a 60-line stand-in providing `approx` / `raises` / `mark.parametrize` / `monkeypatch`
(scratchpad only, not committed). That is a harness substitution and is disclosed as one; the test
bodies are unmodified.

---

## 3. Resident bytes — computed from `weights/sizing.py`, not projected

Recomputed independently for this report by calling `analytic_gemm_bytes` / `analytic_gemm_rows` on
the real 48-layer shape (E=512, H=2560, I=640, group 16):

| | bytes |
|---|---:|
| per-expert, **fold** (fp16 group scale) | 3,072,000 |
| per-expert, **split** (e4m3 + global share) | **2,780,160** |
| 48-layer expert stack, fold | **70.3125 GiB** |
| 48-layer expert stack, split | **63.6328 GiB** |
| **net saving** | **6.6797 GiB (9.500%)** |
| gross scale-byte saving (2 B/group → 1 B/group) | 7.0312 GiB |
| global-vector cost (512·(1280+2560)·4·48) | 0.3516 GiB |

**Confirms the briefed 6.68 GiB to four decimals.** One correction: the brief's per-expert figure
**2,764,800 B is weight + block-scale only**; the true resident per-expert is **2,780,160 B**, the
extra 15,360 B being that expert's share of the global. The 3,072,000 B fold figure is exact.

**The granule row is the part that fails silently.** `GemmBytes` gained a `scale2` field rather than
rolling the global into `scale`, because the arena reserves by **enumerating rows** and rows never
straddle a chunk — "one 25 MiB row + one 8 MiB row" and "one 33 MiB row" reserve *differently*.
`analytic_gemm_rows` now emits `w13.scale2 = 2,621,440` and `w2.scale2 = 5,242,880` per layer.
Under-counting here does not error: the plan under-reserves by E·N·4 per container, the bump
allocator runs dry, and weights budgeted host-resident land in VRAM with a KV pool sized off the same
under-count and a boot that looks fine.

---

## 4. Everything that still requires a GPU

This workflow was CPU-only **by design** — a hyper-connection fusion workflow held both cards. No
`/dev/kfd`, no `torch.cuda`, no serve, no `rocm-smi`. Nothing here has executed on gfx1201.

### 4.1 Two blocking gaps that must be closed *before* the first GPU run

These are not "validation TODOs" — they are code paths that will hard-fail on the first call, and
they were found by reading the branch for this report:

1. **`torch_binding.cpp` still rejects the `(E,N)` global.** Ten grouped-MoE entry points validate
   the zeros slot as `w_zeros.dim() == 3 && w_zeros.size(2)*8 == N && w_zeros.size(1) == scales.size(1)`
   (lines 260, 324, 374, 425, 489, 592, 630, 705, 763, 852). The kernel *bodies* accept the global —
   the policy owns the slot — but the binding's shape predicate does not. The engine
   (`layers/moe.py::_NvFp4MoEMethod`) passes `w13._global_op` / `w2._global_op` into exactly that
   argument, so the very first call raises `w_zeros must be (E, K/group, N/8) int32`.
   Fix: widen the predicate to "AWQ zeros `(E,G,N/8)` i32 **XOR** NVFP4 global `(E,N)` i32, selected
   by the scales dtype". `quant/kernels.py::_check_moe_scale_pair` already documents this gap in a
   dated NOTE and fires first with a message that names the reason.
2. **The fused DECODE gemm2 is still fp16-only, and NVFP4 routes to it.**
   `mmq_fp8_moe_gemm2_gather_reduce_forward` keeps `TORCH_CHECK(scales.scalar_type() == at::kHalf)`
   (`torch_binding.cpp:918`), its launcher still does
   `reinterpret_cast<const __half*>(scales.data_ptr<at::Half>())` (`moe_kernel.hip:1922`), and it
   instantiates `moe_gemm2_gather_reduce_core<…, Int4Fp8GemvLoader, …>` — the fp16 alias. The symbol
   census confirms **zero** `moe_gemm2_gather_reduce_core` e4m3 instantiations among the 504 added.
   Meanwhile `kernels.py:760` routes NVFP4 decode straight there (`_gemv_ok` is true at group-16
   under `MINISGL_NVFP4_GEMV=1`). So **NVFP4 decode currently has no e4m3 arm at all.**
   Interim mitigation available today: `MINISGL_MOE_G2FUSE=0` reverts to the bit-exact WMMA gemm2 +
   `gather_reduce`, whose op *is* converted.

Three ops **were** converted and their bindings widened to accept `fp16 | float8_e4m3fn | uint8`:
`mmq_fp8_moe_gemm_forward` (incl. its gemv arm), `mmq_fp8_moe_gemm1_silu_forward`,
`mmq_fp8_moe_gemm_scatter_forward`. Two further fp16-only ops are **provably unreachable** for NVFP4
and were deliberately left: the two `*_flag_*` ops `TORCH_CHECK(group_size ∈ {32,64,128})` and NVFP4
is group-16; the two `mmq_regdirect_w4a16_moe_gemv_*` ops need `group_size % 32`, which is why
`_NvFp4MoEMethod` is documented as always taking the LDS path.

### 4.2 Deferred on-device validation, in order

1. **Op-level parity on gfx1201** — `Nvfp4E4m3Loader` / `Nvfp4E4m3GemvLoader` vs the fp64 golden and
   vs `moe_gemm_scalar_kernel<…,E4m3GroupScaleGlobal>` (the templated reference exists precisely for
   this). Must cover the two-sided `wscale_epi` in the SILU epilogue, which no host test can reach.
2. **fp16 non-regression, bit-exact.** §1.4 shows the fp16 ISA changed (register allocation). A
   before/after MXFP4 + W4A8 MoE A/B on the *same* card must be bit-identical, or the "policy is
   behaviour-preserving" claim is unproven.
3. **A 4-layer serve** of `Qwen3.8-Flash-Next-NVFP4` — first end-to-end proof the two levels reach
   the kernel with the right strides.
4. **Graph capture** — eager-only is never done (`RULE: graph capture required`). The scale dtype is
   now part of the dispatch, so the captured width must be re-checked
   (`CORRECTNESS: capture width picked a DIFFERENT kernel`).
5. **The `engaged()` ledger diff, per leg.** A vanished arm is a silent dispatch regression that
   benches cannot see. Expect `fp8_wmma.mmq_fp8_moe_gemm1_silu…` and the gemv arm to change identity.
6. **Resident-byte reconciliation against a real boot** — the analytic 63.6328 GiB against
   `post_load()` reality, including the new `scale2` granule rows.
7. **PCIe traffic** on host-resident layers: scales are ~20% of expert bytes, halving them is a ~10%
   traffic cut, and card 1's root port is Gen4 x8 (14.48 GB/s), so this is a measurable claim — and
   an unmeasured one today.

### 4.3 Also deferred: the CPU expert tier

`weights/cpu_native.py` now **refuses** an e4m3 scale slab rather than reading half a slab through
the fp16 `WLoadVnniFp16` policy and dropping the global (which would be a uniform ~4.8e3× error,
finite and fluent). The gap *moved* rather than closing: `sizing._CPU_WLOAD_BY_SCHEME` already maps
NVFP4 to `vnni_nvfp4_e4m3_g16`, which is now **exactly** the layout the containers hold — so the
previously-required checkpoint repacker is no longer needed; what is missing is that policy
instantiated in the CPU core with a `global` pointer threaded to its `post_scale()`. The ~10%
capacity/bandwidth win (and the ~1800× accuracy win measured there independently) is now one
kernel-side policy away instead of one repacker away.

---

## 5. Dense NVFP4 — DEFERRED and NAMED (RULE 5)

**Dense NVFP4 was not converted.** `quant/method.py::NvFp4LinearMethod` stays on `fold_nvfp4_scale`,
and this is stated in its docstring with a date and a reason, not omitted:

> the DENSE cores (`w4a8_fp8_wmma_kernel.hip`, `gemm_tiled.h`) still hardcode `const __half* w_scales`
> and have no policy seam at all. Handing them e4m3 bytes would reinterpret them as halves and return
> finite, plausible, wrong numbers.

The separation is **fenced in code, not by convention**: `nvfp4.nvfp4_leaf_splits` splits `.experts.`
modules and folds everything else, so a dense NVFP4 linear cannot start receiving e4m3 by accident.
The follow-up is to template those two dense cores exactly as the MoE cores were templated, after
which the predicate becomes `return True` and `fold_nvfp4_scale` is deleted.

Scope note (**PROJECTED**, from the checkpoint's own key set): on the served target
`Qwen3.8-Flash-Next-NVFP4` **only the routed experts are quantized**, so the dense NVFP4 surface is
other checkpoints' (Laguna / Muse-Glimmer) dense linears. The accuracy and byte wins land where the
bytes are. **This must appear in the commit message.**

---

## 6. `quant/nvfp4.py`'s exactness claim — CORRECTED

The module docstring said:

> This fold is exact (fp16 easily holds e4m3/global; E2M1→e4m3 decode is lossless)

It is not, and **that claim is why nobody looked**. It now reads, with the measurement inline:

> **THIS FOLD IS LOSSY. THE DOCSTRING HERE USED TO CLAIM IT WAS EXACT; IT IS NOT.** fp16 carries an
> 11-bit significand, and `e4m3_block × global` is a 4-bit significand times an arbitrary f32 one —
> the product does not land on an fp16 grid point, so every single group scale is rounded.

`fold_nvfp4_scale`'s own docstring is likewise relabelled **LEGACY, LOSSY, AND STILL LIVE FOR DENSE
LINEARS**. Per §2.1 the replacement claim should say **"exact to f32 round-off (≤5.5e-08)"**, not
"exact"; the split's docstring wording should be tightened to match before merge.

---

## 7. For posterity: this checkpoint's `weight_scale_2` is a MULTIPLIER

The module documents (and `fold_nvfp4_scale` implements) `weight_global_scale` as a **divisor**.
This checkpoint's `weight_scale_2` is its **reciprocal — a multiplier**. Using the documented sign
yields |W| ~ 4.2e6 against a true rms of 0.0135: **finite, non-NaN in f32, and therefore capable of
loading "successfully" and serving garbage.** Pinned four independent ways:

1. `down_proj` block scales saturate at e4m3's 448 (byte 126) for every expert;
2. gate/up share one global across all 512 experts (quantized as one stacked tensor);
3. `input_scale × 2688 = 5.3`;
4. the golden test's mutation control: inverting the direction gives **2.316e+07** rel-err (§2.3).

**These experts were therefore never served through the NVFP4 fold path at all** — `config.py:385`
already ignores that name, so it was never a live bug, but it means the fold path had no production
exposure on this checkpoint. Direction normalisation is now a **host-side loader job** done in exactly
one place (`nvfp4_global_multiplier`): the kernel's `E4m3GroupScaleGlobal::epi` only ever
**multiplies** and has no divide, deliberately — a policy that could divide would carry the convention
into the kernel, where it is unobservable.

---

## Appendix — what changed, and how to reproduce

**Kernels** (`/home/pat/code/rdna4-hip-kernels-e4m3`, +525/−380 across 6 files, +1 new probe):
`fp8_wmma_rocm/tile_config.h` (+64/−0) · `moe_gemm_tiled.h` (+59/−25) · `gemv_decode.h` (+149/−214) ·
`moe_kernel.hip` (+211/−137) · `moe_gemm_flag.h` (+3/−1) · `torch-ext/torch_binding.cpp` (+39/−3) ·
`local/wscale_policy_contract.hip` (147 lines, new).

**Engine** (`/home/pat/code/minisgl-rdna4-e4m3`, +962/−308 across 11 files, +2 new tests):
`quant/nvfp4.py` (+289/−35) · `models/weight.py` (+287/−190) · `quant/kernels.py` (+71/−6) ·
`layers/moe.py` (+58/−23) · `quant/method.py` (+18/−1) · `weights/sizing.py` (+35/−10) ·
`weights/cpu_native.py` (+36/−10) · `weights/stream_tier.py` (+28/−2) · three test files updated ·
`tests/core/test_nvfp4_two_level_scale.py` (394 lines, new) ·
`tests/qwen4exp_nvfp4_golden_test.py` (577 lines, new).

Reproduce, CPU-only, no lease:

```bash
IMG=minisgl-rdna4:m1b-20260903
# kernels: clean build (405 s, 7 TUs, 0 warnings) + the policy contract probe
docker run --rm -v /home/pat/code/rdna4-hip-kernels-e4m3:/kernels --entrypoint bash $IMG -lc '
  cd /kernels/fp8_wmma && rm -rf build && MAX_JOBS=8 GPU_ARCHS=gfx1201 bash local/build_local.sh &&
  /opt/rocm/bin/hipcc -O2 -std=c++20 -I fp8_wmma_rocm -o /tmp/wsp local/wscale_policy_contract.hip && /tmp/wsp'
# engine: the fp64 golden and the loader, on the real checkpoint
docker run --rm -v /home/pat/code/minisgl-rdna4-e4m3:/engine -v /home/pat/.cache/hf-q4e:/model:ro \
  --entrypoint bash $IMG -lc 'PYTHONPATH=/engine/python:/opt/kernels python /engine/tests/qwen4exp_nvfp4_golden_test.py /model'
```

**Build reproducibility — MEASURED, with a caveat about *which* build.** Two independent
from-scratch builds (`rm -rf build`, 405 s and 321 s) produced the **byte-identical** artifact,
101,972,400 B, sha256 `65ed58b3306ca0a6e658012a19a9fddd6ad2dd3edb6ad77b14f857fba9216cea` — so the
clean build **is** bit-reproducible in this image. The `.so` that the earlier work left sitting in
the worktree is **not** that artifact (101,924,000 B, sha `ffdecccd…`): it came from a chain of
*incremental* builds and differs from a clean build of the same source. Always ship the clean-build
output; a `.so` produced incrementally across header edits is not provably the source you have.
