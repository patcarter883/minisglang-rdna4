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

## DEFERRED BACKLOG — structural: weight-loader policy (dedup W4A8/W8A8/RXF core)
**Do this BEFORE the register-blocking rewrites** (so those, and every future opt, land ONCE). The
tiled cores are COPY-PASTED: two physical `moe_gemm_tiled.h` (w4a8 int4-decode vs w8a8 fp8-direct),
differing only in the ~10-line weight-staging block — entire tiling/LDS/double-buffer/WMMA/occupancy
core duplicated. That's why C's packed-store (w4a8 copy) and B/D's occupancy clamp (w8a8 launcher) did
NOT transfer. FIX: the kernel is already templated on an `MMA` policy — add a `WLoad` (weight-loader)
policy the same way: `template<AT, MMA, WLoad, …>` with `WLoad::stage_b(...)` the only per-format code.
Loaders: `Int4Fp8Loader` (nibble unpack + decode_w4_to_e4m3 + C's packed 2-store), `Fp8DirectLoader`
(4-byte copy), `E2M1Loader`, later `Int8`/RXF W4-NL. ONE `moe_gemm_tiled.h` + ONE dense `gemm_tiled.h`,
format-generic. Payoff: every core opt written once for all formats; **w8a8 gets a real DENSE tiled
kernel** (Fp8DirectLoader) retiring the grouped-over-E=1 hack in `w8a8_dense_linear`; launchers +
`make_moe_tile_config` + GTILE clamp unify too. Validate byte-identical per format (max|Δ|=0 vs current).

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
