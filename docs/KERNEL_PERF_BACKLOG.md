# Kernel-perf campaign — backlog

Tracks the gfx1201 (RDNA4) HIP-kernel perf work. The **current cycle** (in progress) is the
routing consolidation + dead-kernel/env cleanup; the **deferred backlog** below is explicitly
scheduled to be attended to **after** that cycle lands (validated + merged).

## CURRENT CYCLE (in progress) — consolidation + cleanup
1. **Landed:** w4a8 packed-store + e2m1 fix (`90ed548`, +1.9x int4 to 88 TF/s, MXFP4 21→65 + correctness, bit-exact both dtypes).
2. **To fold in:** D's w8a8 MoE occupancy clamp (branch `occ-audit`, bit-exact +1.15–1.71x).
3. **Build the consolidated `pick_dense_config(M,N,K,group,e2m1,captured)` selector** in the kernel
   library (owns BM/BN/GTILE + tiled-vs-decode + the small-M/wide-N `prefill_wmma` config point),
   replacing the engine's hardcoded strings + per-scheme duplication. MoE gemm1 gemv→32. dense_gemm
   served-tile cap.
4. **Remove dead research kernels + env:** `prefill_wmma_b128`(6), `wmma_dbuf`(8), `wmma_dbuf2`(9),
   `splitk_smallm`(12), `regdirect_shuffle`(13), `nsplit_smallm`(14), dense `mmq_regdirect_fp8/_wide/_f16`
   (engine-unused); + the `VLLM_W4A8_V6/V8/V9/V12/V13/V14_*` env family, `DENSE_TILED_OFF`/`SMALLM_OFF`,
   and testing-only `MINISGL_*` toggles (`MINV_PIPE_*`, `MOE_PROF`, per-kernel tuners).
5. Rebuild → exhaustive parity → serve smoke (int4 + MXFP4, TP=2 graph capture) → merge.

## ACTIVE — structural: weight-loader policy dedup → merge into ONE `fp8_wmma` package
**In progress** (agent on branch `core-unify`), then the package merge (me), BEFORE the register-blocking
rewrites (so those + every future opt land ONCE). Problem: the tiled cores are COPY-PASTED — two physical
`moe_gemm_tiled.h` (w4a8 int4-decode vs w8a8 fp8-direct) differ only in the ~10-line weight-staging; entire
tiling/LDS/double-buffer/WMMA/occupancy core duplicated (that's why C's packed-store and B/D's occupancy
clamp did NOT transfer across).
STEP 1 (DONE — branch `core-unify` @ `ee844c1`, worktree /home/pat/code/rdna4-hip-kernels-unify): `WLoad`
policy owns ALL per-format seams (staging + scale/pointer types + per-group fold + epilogue scale — the split
was deeper than staging alone). Both `moe_gemm_tiled_kernel` + `_ashuffle_kernel` now `template<AT, MMA, WLoad,
…>`. BYTE-IDENTICAL: W4A8 int4 sym+asym+e2m1 20/20 max|Δ|=0, W8A8 fp8 16/16 max|Δ|=0. W4A8 got ~6-7.6% FASTER
(byte-identical, better codegen); loaders self-contained in moe_gemm_tiled.h (easy merge lift). CORRECTION:
C's packed-2-store is in the DENSE prefill kernel, NOT the MoE tiled core — the MoE int4 loader still uses the
byte-loop → packed-storing it is now a once-for-both future opt.

### RETURN-TO — claw back W8A8 MoE tiled +3.5–3.9% (accepted regression, 2026-07-18)
Unifying the leaner hand-written W8A8 kernel onto the (W4A8-shaped) unified template costs W8A8 MoE gemm
+3.5–3.9% (marginally over ±3%; ≈1% end-to-end on ZAYA, the main W8A8 user). NOT a spill cliff — same VGPR
range/worst case, just higher register pressure at WARPS_N=4 from the broad reshape. ACCEPTED as a one-time
cost (offset by W4A8's ~6% gain on the shared path + the dedup buys once-for-both future opts). FIX to try:
`if constexpr(WLoad::is_fp8)` specialization so the W8A8 instantiation compiles to its lean form (drop the
W4A8 group-fold/zeros/AT machinery on that path). Quick follow-up, not a blocker. Agent's one attempt
(scalar group-fold) held max|Δ|=0 but didn't move perf → the shift is the reshape, needs the if-constexpr split.
STEP 2 (DONE — kernels branch `fp8-wmma-merge` @ `d176d72`, engine `feat/kernel-fusion` @ `7f05b45`, on
REVIEW branches not yet main/rdna4): ONE package `fp8_wmma` (one physical shared core, 3 coexisting TUs; 7
duplicate `__global__` kernels made static; repack_w_rep_wide int4/fp8 rank-dispatcher; torch.ops.fp8_wmma_C).
Engine rewired 67 sites (kernels.py+method.py+moe.py). BYTE-IDENTICAL max|Δ|=0 int4+e2m1+fp8+dense; 35B AWQ
TP2 graph-capture serve smoke COHERENT with ops firing from fp8_wmma. DEPLOY DEPENDENCY: engine imports
fp8_wmma → CANNOT land to rdna4 until fp8_wmma is built into /opt/kernels (baked image still has old pkgs).
LANDING = build fp8_wmma into the kernel deploy → merge both branches → remove old w4a8/w8a8 dirs → restart serves.
Original packaging spec below (all implemented):
**FOLD w4a8_fp8_wmma + w8a8_fp8_wmma → one package `fp8_wmma`** (name locked). One
build.toml + all .hip + the SINGLE physical `moe_gemm_tiled.h` (no copy/sync). Merge torch-ext
(torch_binding.cpp/.h both op sets: mmq_fp8_* + mmq_w8a8_*), __init__.py (both wrapper sets), one _ops.py
namespace (torch.ops.fp8_wmma_C.*). Rewire engine imports `w4a8_fp8_wmma`/`w8a8_fp8_wmma` → `fp8_wmma`
(~30 call sites in kernels.py + method.py). Update /opt/kernels bake, lean image, KERNELS.md, _kernels
symlink. Validate int4+fp8 resolve from the one package + serve smoke both.
Payoff: every core opt written once for all formats; **w8a8 gets a real DENSE tiled kernel** (Fp8DirectLoader)
retiring the grouped-over-E=1 hack in `w8a8_dense_linear`; launchers + `make_moe_tile_config` + GTILE clamp
unify; RXF W4-NL folds in as a third loader; the flagship fp8 GEMM slots in as the Fp8DirectLoader core.

## DEFERRED BACKLOG — flagship fp8 GEMM: INTEGRATE (don't chase the vendor)
Round 1+2 established (worktree `rdna4-hip-kernels-fp8gemm`, branch `fp8-gemm-flagship`, `b23dfae`+`242156a`):
fp8 WMMA theoretical peak ≈ 389 TF/s but **UNREACHABLE** — gfx1201 lacks the global→LDS DMA
(`vmem-to-lds-load-insts`), so ~40 VGPR mandatory global staging caps the spill-free macro-tile at 64×64;
any wider tile spills → collapse. So we CANNOT exceed hipBLASLt via tile size; only hand-scheduled ISA
(Round 3) could close the last 7–19%, and only to ~match on the compute-dense shape. **Kernel achieves
90–93% of hipBLASLt (vs the old serving path's 104 TF/s = 46%), exact parity.** ACTION: **integrate the
flagship into the w8a8 dense + MoE fp8-compute paths** (they cap ~104 → ~180, ≈1.75x) — pairs naturally
with the weight-loader templating (the flagship IS the Fp8DirectLoader core done right). Round 3 (ISA
scheduling) = diminishing returns, DEFER/skip unless a fp8-dense-heavy model needs it.

## DEFERRED BACKLOG — register-blocking rewrites (attend to AFTER the current cycle)
These are the second-tier occupancy items from D's audit: kernels capped by **register pressure /
scratch spills**, NOT by the LDS-clamp pattern (so no one-line bit-exact fix — they need a
register-blocking rewrite: shrink live state, reuse operand registers, tighten prefetch depth, and
where the C++ compiler spills a large in-register tile, manual VGPR budgeting / inline-asm scheduling).

Priority order (impact × tractability):

1. **`attn_prefill_paged` (flash prefill paged)** — 25.4 KB static LDS, occ 2, **spilling 22 VGPR /
   scratch 80**. The spill is the direct throughput cap. Register-blocking rewrite of the prefill
   attention inner loop. HIGH — prefill attention is on every request; see
   [[cca-prefill-attention-kernel-bound]], [[hip-kernel-occupancy-audit]].
2. **`attn_hip` prefill** — 251 VGPR / occ 2, register-limited (no spill yet, but 1 wave from it).
   Reduce live state to lift to occ 3. HIGH — same prefill-attention hot path.
3. **`mla_hip` decode/verify** — occ **1** (VGPR 225 + 28.4 KB static `s_o[NWARPS][LATENT]` fp32).
   Shrink the fp32 output-accumulator (math-touch, not a clamp). MED — GLM/DeepSeek MLA, but bs=1
   decode is bandwidth-bound so upside is situational; measure before investing.
4. **flagship fp8 GEMM — spill-free 64×128 macro-tile** (register budgeting / inline-asm) — the SAME
   register-blocking class; the compiler spills the 128-wide fp32 accumulator to scratch → collapse.
   This is the lever to push the flagship kernel past hipBLASLt (>244 → toward the 355–389 TF/s
   ceiling). Then **integrate the flagship into the w8a8 dense + MoE fp8 paths** (currently they cap
   ~104 TF/s; the flagship hit 181–188 = 92% of hipBLASLt in round 1). HIGH — biggest raw win.
5. **`rxf_hip` linear gemv/gemm** (236 VGPR) and **`moe_splitk`** (224 VGPR, decode-only) — mild
   register pressure; LOW priority, register reduction only.
6. **`dense_gemm` bf16 wide-tile occupancy** — 36–46 KB LDS → 1 block/WGP, but the tile is
   caller-chosen; fix belongs in the minisgl dispatch heuristic (cap served `block_m+BN` ≤ ~227 →
   2 blocks). LOW/MED — could also be swept into the consolidation selector.

NOT to touch: `gdn_prefill_wmma`'s 60.8 KB LDS is deliberate (fp32 state fills the WGP,
[[gdn-wmma-lds-budget]]); served MoE (`moe_w8a16`, `moe_bf16`) already at 3 blocks/WGP — healthy;
elementwise (tail/swiglu/sampler) at HBM peak — fine.

CAVEAT (D): the packed-store 2x is a compiler codegen/occupancy effect — re-verify VGPR/occupancy
(`--save-temps` / rocprof) if the ROCm/hipcc toolchain in the image changes; the same source can land
on either side of the spill cliff.
