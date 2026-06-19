# Handoff → minisgl-rdna4: W4A8 grouped-MoE kernel changes (tile-framework + SWMMAC)

**From:** the W4A8 tile-framework effort in `vllm-gfx1201` (the `w4a8_fp8_wmma/` dependency you
consume via `torch.ops.w4a8_fp8_wmma.*`).
**Date:** 2026-06-18. **Landed on `vllm-gfx1201` `main`** (commits `784a672`, `7e28757`, merge
`c092c8e`). Full design: `vllm-gfx1201/w4a8_fp8_wmma/TILE_FRAMEWORK_DESIGN.md`.

This concerns your **Phase-2-MoE path** (done, parity 0.99894) and the **Phase-3d 35B routed
experts** that reuse it.

---

## TL;DR — nothing you rely on breaks
The **torch op interface is unchanged.** `mmq_fp8_moe_gemm`, `mmq_fp8_moe_gemm1_silu`,
`mmq_fp8_moe_gemm_scatter`, `mmq_fp8_moe_gather_reduce` have identical signatures and **identical
default behaviour**. Your `moe_impl.py` keeps working as-is after you rebuild the W4A8 dependency.
Two things were *added* internally, both opt-in:

1. **Consolidated grouped-MoE WMMA kernels** behind a clean `TileConfig`/policy abstraction —
   bit-exact + perf-neutral vs the old `moe_gemm_v5/v6`, **off by default**.
2. **A new grouped-MoE 2:4-sparse SWMMAC sibling kernel** — a *future* perf path, **not yet wired
   to a torch op** (research; see below).

If you do nothing, you get the exact same numerics + perf you have now. Rebuild the dependency to
pick up the new files; your parity harness (`tools/moe_parity.py`) should stay green.

---

## 1. What landed (in `w4a8_fp8_wmma/`)
- `tile_config.h` — shared device primitives + a zero-cost `WmmaFp8` MMA policy + `TileConfig`
  (the scattered `VLLM_W4A8_MOE_{BN,GTILE,BLOCK_M,...}` env knobs collected into one struct).
- `moe_gemm_tiled.h` — grouped **v5** (A-in-LDS) and **v6** (A-shuffle + gtile) re-expressed over
  the policy. Bit-exact to the originals (register-identical; microbench `mean_rel=0`).
- `moe_kernel.hip` — now `#include`s the headers; **`VLLM_W4A8_MOE_TILED=1`** routes v5/v6 through
  the consolidated kernels (default unset → originals, unchanged). Use it only for an A/B.
- `autotune_grouped.hip` + `tile_crossover_cache.jsonl` — a TileConfig sweep → per-shape winner
  cache (v6 wins all 35B MoE shapes; BN=64; gtile 8→4 as P grows). Reference for tile selection.
- `bench_tiled_vs_v5.hip` — standalone HIP A/B harness (no torch) if you want to re-verify.
- `docker-compose.yml` — added `moetest` (1-GPU MoE correctness through torch) + `benchtiled`
  (TP=2 offline A/B) profiles on a `:tiled` image, all via `scripts/gpu-lease.sh`.

**Why you might care:** if you ever need to read/extend the grouped-MoE GEMM, the consolidated
`moe_gemm_tiled.h` is far cleaner than the v5/v6/v7 sprawl in `moe_kernel.hip` — one templated
kernel, the knobs as a struct, the inner matrix-core op as a swappable policy.

---

## 2. The SWMMAC sibling — the future W4A8-MoE perf lever (research, not Phase-3-ready)
`moe_gemm_swmmac.h` — a grouped-MoE **2:4-sparse fp8 SWMMAC** GEMM. It reuses the same spine
(per-expert slabs via `expert_ids`, routed gather via `sorted_token_ids`, act-scale +
fp16/scatter epilogue) but the GEMM loop is the RDNA4 **SWMMAC** sparse matrix-core path (weight =
sparse-A 2:4-compressed fp8 + index read straight from DRAM; activation = dense-B). Validated
**bit-exact by self-consistency** (`bench_swmmac_grouped.hip`, 1-card lease) at the production
`block_m=64` for both 35B MoE shapes.

**Status / what it would take to use in minisgl:**
- It is a `.hip` kernel + a standalone self-consistency bench. **No torch op binding yet** — to
  call it from `moe_impl.py` you'd add a `mmq_fp8_moe_swmmac` op in `bindings.cpp` (mirror the
  existing MoE ops) + the 2:4→compressed `(Wc, idx)` weight repack at load.
- **It needs a 2:4-PRUNED + recovered MoE checkpoint, which does not exist.** The 35B AWQ experts
  are structureless under 4:2 (Tier-3 screen, `vllm-gfx1201` `RESEARCH_swmmac.md`), so you cannot
  prune them on the fly without accuracy loss. **No served grouped-MoE-SWMMAC result is possible
  today** — only self-consistency correctness + a microbench speedup number.
- This fp8/per-channel kernel is the **first rung**. The variant that actually wins for *your*
  W4A8 MoE is **int4-sparse + per-group-scale** (~3 bit/wt) — `bench_swmmac_int4.hip` in
  `feat/swmmac-microbench` shows **1.17–1.55× over dense-int4 W4A8 with less memory**. That is the
  real lever for the **35B gemm2 decode-bandwidth wall** (gemm2 reads weights at ~126–151 GB/s vs
  gemm1's 273 — the documented decode bottleneck). If/when a pruned MoE model exists, the
  int4-sparse grouped SWMMAC op is the Phase-3d 35B MoE-decode optimization.

The complete SWMMAC story (dense kernel, vLLM plugin, PPL gates, the cracked index recipe) is in
`vllm-gfx1201/w4a8_fp8_wmma/SWMMAC_HANDOFF.md` + `RESEARCH_swmmac.md` (committed on
`feat/swmmac-microbench`, `e3ac002`). **Do not edit that worktree's `swmmac_gemm_k`** — it's a
separate concern's committed kernel.

### 2b. UPDATE 2026-06-19 — grouped SWMMAC fully characterized; verdict: NOT a minisgl deliverable yet
The grouped SWMMAC rungs above are now **built and measured to their ceiling** (on
`feat/w4a8-tile-autotune`; full data in `TILE_FRAMEWORK_DESIGN.md §2.7c/§2.7d`). Three results that
**supersede the optimistic numbers in §2 / §4** — read these before treating SWMMAC as a drop-in:

- **int4-sparse + per-group-scale grouped kernel** (`moe_gemm_swmmac_int4.h`) — the production-shaped
  rung, self-consistency bit-exact (1 fp16 ULP) at g32/g64, bm32/64.
- **tiled-LDS variant** (`moe_gemm_swmmac_int4_tiled_kernel`) — stages the weight tile once in LDS.
  Bit-exact (non-scatter only; **SCATTER path NOT yet validated**). Helps **only weight-BW-bound
  shapes**: gemm2 prefill **1.13→1.25× vs WMMA-dense**; large-prefill gemm1 stays at **parity**.
- **honest decode A/B vs the v7-GEMV you actually dispatch** (`bench_swmmac_int4_vs_v7_decode.hip`):
  the old "12–18× / 1.17–1.55×" was **baseline-inflated** (vs WMMA-tiled, a kernel nobody runs at
  decode). Against v7: at **single-stream decode T=1, v7 WINS gemm1** (SWMMAC 0.39×); SWMMAC ~ties
  gemm2 (1.28×). Crossover is **2D**: gemm2 → SWMMAC almost always; gemm1 → v7 at T≤~4, SWMMAC at T≥8.

**Verdict — do NOT consume SWMMAC in minisgl yet** (it is *not* ready to send, for structural reasons,
not packaging):
1. **No torch op** — still raw `.hip` kernels, not exposed via `torch.ops.w4a8_fp8_wmma.*`; your
   provider has nothing to call. (The op binding was scoped, then **deliberately deferred** —
   `run_moe_gemm` receives dense `(E,N,K/8)` weights, not compressed `(Wc_i4, idx)`, so a dispatch
   branch can't even receive its inputs without new bindings/adapter plumbing the project declined.)
2. **No weight source** — the HW instruction is 2:4, which is PPL-hopeless on the 35B; the PPL-viable
   ratios (6:8/4:6) don't map to it. There is no served path and no research path to one.
3. **Self-consistency only** — never validated against a served reference.

So the **WMMA-dense + grouped path you already consume (parity 0.99894) remains the only deployable
W4A8-MoE kernel.** SWMMAC stays a parked future lever: revisit only if a 2:4-pruned + finetuned MoE
checkpoint ever exists, at which point it needs the torch op + served validation first.

---

## 3. Practical notes
- **Rebuild the dependency** to get the new headers (they're `#include`d by `moe_kernel.hip`, which
  `setup.py` compiles): `pip install .` in `w4a8_fp8_wmma/`, or rebuild the image you base on.
- **Production config:** the 35B MoE uses `block_m=64` — that's the config validated bit-exact.
  v6 is the served default and wins the autotune sweep at all 35B shapes.
- **The 35B quantizes only routed MoE experts** (attn/shared/dense are unquantized), so the grouped
  MoE GEMM is the *only* W4A8 path that runs on it — exactly your Phase-3d hot path. Dense W4A8
  kernels never execute on the 35B.
- **All GPU work goes through `scripts/gpu-lease.sh`** (now on `main`): `-n 1`/`-n 2`, it blocks
  until cards free and frees on exit. The box is heavily shared (you, moe-tune, swmmac all run here).
- Validation method for a sparse/quant kernel with no reference model: **self-consistency** — synth
  a weight, apply the structure, compute two ways on-device, assert rel≈0 (see
  `bench_swmmac_grouped.hip`). Reuse it for the int4-sparse op.

---

## 4. Suggested next step for minisgl (when you reach 35B MoE perf)
The 35B MoE decode wall is gemm2 weight bandwidth. The ordered levers (most are active
worktrees in `vllm-gfx1201`): **int4-sparse grouped SWMMAC** (needs a pruned MoE model — biggest
win, ~1.2–1.55×), N-interleave burst-repack (`feat/w4a8-burst-repack-research`), apply-level fusion
(`feat/w4a8-moe-apply-fusion`), nt/streaming load hints (`feat/decode-bw-levers`). For now the
consolidated WMMA path is your stable, validated baseline and the torch op contract is unchanged.
