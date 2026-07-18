# Kernel-fusion overnight loop — charter + journal

> **STATUS: COMPLETE (2026-07-18 ~00:55).** Safely-unattended high-ROI shortlist exhausted. Shipped 1
> validated fusion (GDN gated-norm, bit-exact + serve-clean). Highest remaining target (MoE gemm1+silu)
> SHELVED as attended-recommended — a major quantized-gemv restructure with tol-only (not bit-exact)
> validation; warrants human review, not unattended commit. See the final LOG entry + backlog below.


Autonomous, unattended. Goal: reduce **bs=1 decode latency** on gfx1201 by fusing per-layer op chains.
Proven this session by elimination: bs=1 decode (~40 tok/s = 25ms/tok) is **~90% overhead** (kernel
dispatch + tiny memory-bound gemvs + GDN/MoE glue), NOT bandwidth (mem-OC flat) nor comms (fp8/gather
flat). Fusion is the lever. Low-level HIP only — intrinsics, WMMA, inline asm if needed. No new abstractions.

## HARD GUARDRAILS (never violate)
1. **Worktrees only**: kernels `/home/pat/code/rdna4-hip-kernels-fusion` (branch feat/kernel-fusion),
   engine `/home/pat/code/minisgl-rdna4-fusion`. NEVER edit the shared trees or other agents' worktrees.
2. **Build without a lease** (hipcc codegen needs no GPU): `docker run --rm -v <kern-wt>/<pkg>:/kernels
   --entrypoint bash minisgl-rdna4:lean -lc 'export PATH=/opt/venv/bin:$PATH; cd /kernels && rm -rf build
   && GPU_ARCHS=gfx1201 bash local/build_local.sh'`. ALWAYS `rm -rf build` first (stale-object ABI trap).
   Build in **minisgl-rdna4:lean** (torch 2.14), NOT vllm22-w4a8:combined (torch 2.10, wrong image).
3. **Every GPU run through the lease, bounded**: `gpu-lease -n 1 --timeout 900 -- ...` (one card is enough
   for a single-kernel parity/perf test). Forward `HIP_VISIBLE_DEVICES`/`ROCR_VISIBLE_DEVICES`. Never
   hand-set devices. Be a good citizen — short leases, `--timeout`, no hogging (other agents run too).
4. **Parity BEFORE commit, always**. A fused kernel must match its reference: bit-exact where the fusion
   is algebraically identical; tight tol (bf16 rtol≤1e-2 / fp16≤3e-3, justify) where reordering changes
   rounding. NO commit without a passing parity test. NO silent-corruption risk — this box has already
   spent a whole session chasing silent corruption; treat it as the cardinal sin.
5. **HANG = STOP**. Timeout every probe (`timeout 300 docker …`). If a probe times out AND `rocm-smi`
   then shows the card N/A/unknown → GPU is wedged; you CANNOT reboot unattended. Write `BLOCKED: GPU
   wedged, needs reboot` at the top of this file, restore any config, and END the loop. Do not keep
   hammering a dead card.
6. **Commit only validated wins**: parity-pass AND a measured micro-perf win. Name kernels by FUNCTION,
   no version numbers (`_v2`, etc.) — grep-verify zero `_v[0-9]`. Commit to the worktree branch only.
7. Don't restart/stop other agents' serves or `lactd`. LACT is tuned (1380/-75mV) — leave it.

## PER-ITERATION PROTOCOL
1. Read this file top-to-bottom (state = the log below). If `BLOCKED`, stop.
2. If no active target: **survey** — statically map which per-layer chains are SEPARATE kernels vs already
   fused (grep the kernel pkgs + `python/minisgl/layers/*`), cross-ref existing rocprof traces
   (`.rocprofv3/*.dat`, `tools/analyze_trace.py`) to rank by decode-time overhead. Pick the highest-ROI
   **un-fused** chain (a lot is already done — do NOT redo). Write a 5-line mini-spec here, then proceed.
3. Implement the fused kernel in the kernels worktree (torch-ext binding + .hip). Small, correct, one chain.
4. Build (guardrail 2). Fix compile errors (LSP "ATen not found" is host-only noise; trust the container).
5. **Parity** (guardrail 4): write/extend a tiny harness that runs the fused kernel and the reference on
   the same inputs and asserts closeness. Lease one card, timeout-guarded.
6. **Perf**: micro-bench fused vs reference (isolated, warmup + timed), same lease. Record GB/s or µs.
7. If parity-pass + win → commit (worktree) + log. If parity-fail → debug ≤2 tries then revert + log why.
   If no win → revert kernel, log the finding (negative results are results).
8. Append a dated entry to the LOG below: target, what you did, parity result, perf delta, next step.

## TARGET ROI SHORTLIST (survey first — several may already be done)
- **MoE dispatch**: route→gather→grouped-gemm→scatter as one kernel (biggest 35B overhead; complex —
  index bugs fault easily, so heavy parity). `python/minisgl/layers/moe.py`, `moe_hip/`, `moe_splitk_hip/`.
- **dequant+gemv**: fold W4 dequant into the decode gemv (no dequantized-weight HBM round-trip).
  `w8a8_fp8_wmma/`, `w4a8*`, check what the W4A8 gemv already fuses (memory: int4 gemv≤8 / e2m1 gemv≤16).
- **RMSNorm → proj prologue**: fuse the norm into the next projection's load. `swiglu_hip/`, `tail_hip/`.
- **GDN conv+ssm+gate**: `gdn_hip/` — sequential at bs=1, keep state resident (mind the 64KB LDS budget).
- **attention epilogue → o_proj → residual**: `attn_decode/`, `mla_hip/`.

## STOP CONDITIONS
- GPU wedged (guardrail 5) → BLOCKED, end.
- No un-fused ROI target left → write "COMPLETE: shortlist exhausted", end.
- Same target failing parity 3 iterations → shelve it (log why), move to next; if none left, end.

## CURRENT TARGET — #1 GDN mixer-glue fusion (survey-selected)
Collapse the GDN linear-attention decode core on the 3-in-4 GDN layers: **~8 launches → 1**.
Chain (`python/minisgl/gdn/layer.py:424-457,219-231`): `causal_conv1d_update`(+SiLU) → split+**3×
.contiguous()** → `gdn_decode` → **2× .contiguous()** → `rmsnorm_gated`. Three separate `gdn_hip` ops +
~5 copy launches, all on B=1 cache-resident tensors (pure dispatch overhead). Deliberately unfused
(KERNELS.md:157 "clean boundary"), NOT tried-and-reverted → open ROI. Keep `in_proj_qkvz`/`in_proj_ba`/
`out_proj` gemvs separate (real weight reads).
Kernels worktree: `rdna4-hip-kernels-fusion/gdn/` (gdn_kernels.hip @123/@772, torch_binding.cpp, __init__.py).
PARITY: can be **BIT-EXACT** (recurrent path + fp32-internal norm are bit-stable). Refs already tooled:
`tools/gdn_hip_parity.py`, `tools/gdn_layer_parity.py`, `tools/qwen3_5_gdn_isolate.py`. PERF: `tools/gdn_hip_bench.py`.
RISK: 64KB LDS budget (gdn already tight — see memory gdn-wmma-lds-budget); recurrent conv/ssm state correctness.
**STAGED (each stage: build→parity(bit-exact)→perf→commit before next; de-risks the megakernel):**
- Stage A: kill the ~5 `.contiguous()`/reshape copies — make `gdn_decode`/`rmsnorm_gated` accept the
  strided split views directly (or split inside the kernel). Lowest risk, removes copy launches.
- Stage B: fuse `rmsnorm_gated` into `gdn_decode`'s epilogue (gdn already emits `core`; norm consumes core+z).
- Stage C: fuse `causal_conv1d_update` into the prologue (conv+ssm state both resident → mind LDS).
Do NOT attempt the full megakernel first — stage it.

## LOG (append newest last)
- 2026-07-18 (setup): worktrees created (minisgl d5e1bb6, kernels 673b43b). Journal written.
- 2026-07-18 (iter 1 — SURVEY): mapped the GDN-hybrid+MoE decode op-chain with launch evidence. Found the
  projection gemvs ALREADY dequant-fused (int4≤8/e2m1≤16, in-register — not the target). Ranked un-fused
  chains: **#1 GDN mixer glue (~8 launches, 75% of layers)** ← selected; #2 MoE gemm1→silu_and_mul (decode
  gemv can't use the wmma-only fused-silu epilogue); #3 dense/shared SwiGLU HBM round-trip; #4 full-attn
  unmerged qkv + standalone rotary. rocprof .dat traces are binary buffer dumps (analyze_trace.py can't
  parse) → used in-source measured numbers + KERNELS.md ledger. Spec above. NO code changed yet.
  Next iter: Stage A — remove the GDN `.contiguous()` copy launches; build; bit-exact parity via gdn_layer_parity.
- 2026-07-18 (iter 2 — Stage A VOID + Stage B scoped; cards busy=other agent's CAM serve, no GPU used):
  **CORRECTION: Stage A is a no-op at bs=1.** Verified in torch (container, no GPU): a `[1,K]` slice has a
  size-1 leading dim → `is_contiguous()=True`, `.contiguous() is self` (NO copy kernel). Only n>1 (batched
  decode) actually copies. So the "~5 copy launches" don't exist at the bs=1 target — SKIP Stage A (it'd
  only help batched throughput). Real bs=1 GDN overhead = the **3 compute kernels**.
  **STAGE B fully scoped (implement next iter):** fuse `rmsnorm_gated` INTO `gdn_decode`'s epilogue.
  - `gdn_decode_kernel` (gdn_kernels.hip:123): grid=(B*HV), each block = one (bi,hv), thread r computes
    core element r for r<V=head_v_dim, writes `out[(bi*HV+hv)*V + r]`. `rmsnorm_gated_kernel` (:890): row
    m=(bi,hv), D=V, `out = x*rsqrt(mean(x^2)+eps)*weight[j]*silu(z)`. **Block↔row match is exact.**
  - Fusion: after `gdn_step` gives `o` (fp32) per thread r: round through scalar_t (`float o_r=(float)(scalar_t)o;`
    → BIT-EXACT vs ref which stores core as bf16 then re-reads), block-reduce Σo_r^2 via existing `red`
    scratch, `inv=rsqrtf(sum/V+eps)`, write `(scalar_t)(o_r*inv*weight[r]*silu(z[(bi*HV+hv)*V+r]))`.
  - GOTCHA: the `slot<=0` early-`return` (:157) must become non-returning (set o_r=0, still join the
    reduction `__syncthreads`) or the block reduction deadlocks/half-reduces. All threads must reach it.
  - New op (name by FUNCTION, no vN): `gdn_decode_gated` taking extra `z, norm_weight(fp32), eps`; add
    binding + gdn_hip/__init__ export. Engine: `gdn/layer.py` forward_decode calls it and SKIPS
    `_output_projection`'s separate `rmsnorm_gated` (still does out_proj on the normed result).
  - PARITY: `tools/gdn_layer_parity.py` / `gdn_hip_parity.py`, expect BIT-EXACT. PERF: `gdn_hip_bench.py`,
    expect -1 launch/GDN-layer + core round-trip gone. Commit only if bit-exact + measured win.
- 2026-07-18 (iter 3 — Stage B IMPLEMENTED + VALIDATED + COMMITTED; cards free):
  Added `gdn_decode_gated` op (kernel `gdn_decode_gated_kernel` cloned from gdn_decode + rmsnorm_gated
  epilogue; launcher; binding wrapper/def/impl; __init__ fake+export). Built clean in lean image.
  **PARITY (tools/gdn_decode_gated_parity.py): BIT-EXACT** — output AND ssm_state mutation both max|Δ|=0
  vs gdn_decode+rmsnorm_gated. **PERF (tools/gdn_decode_gated_bench.py): 1.16x** — 25.30 vs 29.42 us/call,
  4.12us saved/GDN-layer/token (bs=1 eager; ~30 GDN layers → ~124us/tok eager, less under graph but the
  core HBM round-trip removal persists). **COMMITTED kernels fa83125** (existing gdn_decode untouched).
  NEXT iter: wire into engine — `gdn/layer.py` forward_decode calls `gdn_decode_gated` (pass z_flat,
  _norm_weight_fp32(), eps) and SKIP `_output_projection`'s separate `rmsnorm_gated` (still out_proj on the
  normed result); whole-layer parity via gdn_layer_parity.py + a short serve coherence smoke; commit engine
  change if clean. Then Stage C (fuse causal_conv1d_update prologue → 1 launch/GDN-layer total).
- 2026-07-18 (iter 4 — Stage B ENGINE WIRING done + validated + committed; STAGE B COMPLETE):
  `forward_decode` now calls `gdn_decode_gated` behind `_GDN_FUSED_NORM` (env MINISGL_GDN_FUSED_NORM, default
  1) + `hasattr(gdn,"gdn_decode_gated")` guard → old .so falls back to the 2-kernel path safely. z_flat =
  z.reshape(-1,V).contiguous() (== what _output_projection fed rmsnorm_gated). **VALIDATED end-to-end**
  (tools/car_gdn_fused_smoke.sh): Qwen3.5-4B GDN-dense TP=1 serve under graph capture — 5/5 keyword canaries
  COHERENT (paris/au/oxygen/jupiter/4), `has gdn_decode_gated: True`, and log `[hip-engage]
  gdn_hip.gdn_decode_gated` CONFIRMS the fused kernel is executed (not the fallback). **COMMITTED ef618f0.**
  Stage B (gated-norm fusion) DONE: gdn_decode+rmsnorm_gated -> 1 kernel, bit-exact, 1.16x op, serve-clean.
  NEXT iter: **Stage C** — fuse `causal_conv1d_update` (conv-state roll + SiLU) into `gdn_decode_gated`'s
  PROLOGUE → 1 launch/GDN-layer total (was 3). HIGHER RISK: two persistent states (conv_state + ssm_state)
  resident in one kernel + the conv depthwise reads conv_state[slot] — mind the 64KB LDS budget (see memory
  gdn-wmma-lds-budget). Read `causal_conv1d_update` kernel (gdn_kernels.hip:772) + its state layout; scope
  whether it fits the (bi,hv) block structure (conv is per conv-channel, not per v-head — may NOT map cleanly;
  if it doesn't, Stage C may be lower-ROI than expected — evaluate before implementing).
- 2026-07-18 (iter 5 — Stage C SHELVED (infeasible) + next target selected; no code changed, CPU only):
  **Stage C infeasible, confirmed by reading the kernels.** `causal_conv1d_update_kernel` (gdn_kernels.hip:845):
  grid=(B), ONE block/batch iterating ALL C=conv_dim channels, each channel rolls its own `conv_state[slot,c]`.
  `gdn_decode_gated`: grid=(B*HV), per-(batch,v-head). Fusing conv into those blocks → multiple v-head blocks
  re-conv the SHARED q/k channels (GQA: HV v-heads share H k-heads) AND race on the same conv_state roll-update
  → redundant compute + STATE CORRUPTION. Would need a 2-phase grid-sync megakernel (out of scope, risky). GDN
  chain optimally fused at 3→2 launches (Stage B done). NOT redoing.
  **NEXT TARGET = #2 MoE gemm1 → silu_and_mul** (highest remaining ROI: every 35B layer). Dense GatedMLP + MoE
  both do gemv→`tail_hip.silu_and_mul`→gemv; silu_and_mul (activation.py:25, python/minisgl/quant/kernels.py:338)
  is a SEPARATE elementwise kernel with a full gate_up HBM round-trip. Fuse silu(gate)*up into the gemm1 gemv
  epilogue (write [M,d] directly). PRECEDENT: gemm2 already fuses scatter+reduce epilogue (kernels.py:350-379),
  so the moe-gemm family supports epilogue fusion. **RISK: HIGH** — quantized (fp8 act × W4) grouped gemv; the
  survey noted the existing fused-silu epilogue is "wmma-only, unusable at decode gemv" (kernels.py:337), so this
  ADDS a silu epilogue to the DECODE gemv path. PROTOCOL (extra rigor for a quant kernel, per no-silent-corruption
  guardrail): op parity vs (gemm1 + tail_hip.silu_and_mul) across MULTIPLE shapes (M=1,2; several inter/expert
  dims) + scales, bit-exact-or-tight-tol, THEN serve smoke; if the gemv can't cleanly pair gate[j]/up[j] in the
  epilogue, fall back to #3 dense-bf16 SwiGLU (shared expert, lower risk) or shelve. Files: quant/kernels.py
  :325-342 (gemm1+silu), moe_hip/ or the mmq_fp8_moe_gemm kernel. Next iter: read gemm1 kernel + gemm2 epilogue
  precedent; assess feasibility/safety before writing.
- 2026-07-18 (iter 6 — MoE gemm1+silu FEASIBILITY assessed; FEASIBLE but HIGH-risk quant; CPU only):
  Read w4a8_fp8_wmma/w4a8_fp8_wmma_rocm/moe_kernel.hip. Decode gemm1 = `moe_gemv_decode_kernel` (:572),
  output [P, N=2*inter] (gate|up), tiled BY COLUMN (col=blockIdx.x*BN+threadIdx.x) → gate col j and up col
  j+inter are in DIFFERENT blocks, so a plain epilogue CANNOT pair them. Separate `tail_hip.silu_and_mul`
  follows (quant/kernels.py:338) = extra launch + [P,2*inter] HBM round-trip. A FUSED reference already
  exists for the WMMA/large-M path: `moe_gemm1_silu_alds_kernel` (:255) / `_ashuffle_` (:404) decode BOTH
  dg+du per output j and fuse silu — but "wmma-only, unusable at decode gemv" (the M<=2 path is the plain
  gemv). **FEASIBLE:** add `moe_gemv_decode_silu` — model on moe_gemv_decode but each thread gemvs gate row
  j AND up row j+inter (pairing, as the WMMA fused does), silu(gate)*up in fp32, write [P, inter]. Removes
  the silu launch + gate_up round-trip EVERY MoE layer (highest remaining ROI). **RISK HIGH** (fp8 act ×
  W4/e2m1, per-row scales/zeros for gate vs up, e2m1-vs-int4 decode) — a subtle scale/zero/shape bug is
  exactly the silent-corruption the guardrail forbids. **MANDATORY VALIDATION before commit:** op parity vs
  (moe_gemv_decode + tail_hip.silu_and_mul) across the FULL matrix — M∈{1,2} × int4 AND e2m1 × zeros/no-zeros
  × a few inter/E/group sizes; likely NOT bit-exact (silu fp32 order differs from tail_hip) so tight rtol
  (justify); cross-check vs the WMMA moe_gemm1_silu too; THEN a serve smoke on a MoE model. If the quant
  details can't be validated confidently in ≤3 tries → SHELVE as ATTENDED-RECOMMENDED (do not gamble a quant
  kernel unattended). Next iter: read moe_gemv_decode_kernel + moe_gemm1_silu_alds_kernel in FULL (scale/zero/
  group/e2m1 handling) before writing a line.
- 2026-07-18 (iter 7 — MoE gemm1+silu SHELVED (attended-recommended); loop COMPLETE; CPU only):
  Finished reading moe_gemv_decode (output: `out[row*N+col]=(AT)(acc*a_scale)`, :207/scatter :205) + the
  WMMA fused silu (`moe_gemm1_silu_alds`: separate accg/accu, gate scale `ws_e[abs_n]` vs up scale
  `ws_e[inter+abs_n]`, run_g/run_u then silu*mul). VERDICT: the decode-gemv silu fusion is a MAJOR
  RESTRUCTURE of moe_gemv_decode (per-warp COLS cols → for each out col j<inter compute gate row j AND up
  row j+inter: dual acc_gate/acc_up, 2x weight+scale+zp loads, e2m1 AND int4 paths, the gather) and CANNOT
  be bit-exact (out1 fp8/bf16 store + silu-order differ from tail_hip.silu_and_mul) → TOL-ONLY parity.
  Major quant restructure + tol-only validation = the exact silent-corruption profile the guardrail says
  NOT to commit unattended. **SHELVED — ATTENDED-RECOMMENDED** (do it with human review of the quant kernel).
  Remaining shortlist has no clean bit-exact high-ROI target: #3 SwiGLU = same gemv-restructure class
  (tol-only); #4 qkv-merge = bit-exact-able (reuse LinearColParallelMerged) but quant-weight-concat + only
  1/4 (full-attn) layers. Safely-unattended shortlist EXHAUSTED after GDN Stage B. **LOOP COMPLETE.**

## OVERNIGHT RESULT (for morning review)
- **SHIPPED (validated, on branch feat/kernel-fusion):** GDN gated-norm fusion `gdn_decode_gated` —
  gdn_decode + rmsnorm_gated → 1 kernel (−1 launch + core HBM round-trip), on 75% of 35B layers + all dense
  4B GDN layers. BIT-EXACT (output+ssm_state, max|Δ|=0), 1.16x op-level, 4B TP=1 serve smoke COHERENT with
  [hip-engage] confirming the fused kernel runs. Commits: kernels `fa83125`, engine `ef618f0` (defensive:
  MINISGL_GDN_FUSED_NORM default-1 + hasattr fallback). Harness: tools/gdn_decode_gated_{parity,bench}.py,
  tools/car_gdn_fused_smoke.sh.
- **CORRECTLY AVOIDED (negative results, read not trial):** Stage-A GDN `.contiguous()` copies (no-op at
  bs=1); Stage-C conv-prologue fusion (per-channel/GQA state-race); confirms GDN optimally fused at 3→2.
- **BACKLOG (attended-recommended, scoped):** (1) MoE gemm1+silu → `moe_gemv_decode_silu` (restructure
  moe_gemv_decode to pair gate row j + up row j+inter, silu(gate)*up, write [P,inter]; model on the WMMA
  `moe_gemm1_silu_alds`; parity vs moe_gemv_decode+tail_hip.silu_and_mul across M{1,2}×int4/e2m1×zeros×sizes,
  TOL-based; then serve smoke; HIGHEST remaining ROI, every MoE layer). (2) qkv-merge for full-attn layers
  (bit-exact, reuse merged-linear; lower ROI). (3) dense/shared SwiGLU (same restructure as #1).
- Next attended session: pick up the MoE fusion (#1) with human review of the quant kernel — it's the real
  remaining win but needs the careful validation a quant restructure warrants.

## ATTENDED FOLLOW-UP — backlog #1 (MoE gemm1+silu) SHIPPED (2026-07-18, user-directed)
User directed "start MoE gemm1→silu fusion" (attended, so the quant-kernel review the shelving required is
now in the loop). Built `moe_gemv_decode_silu_kernel` exactly as scoped: each warp owns one FUSED output col
j<inter, computes gate (weight col j) + up (col j+inter) sharing the gathered fp8 activation, writes
silu(gate)*up → (P,inter). Wired via mmq_fp8_moe_gemm1_silu(kernel="gemv") (run_moe_gemm1_silu Gemv branch +
relaxed wmma-only asserts); engine routes w4a8_moe's decode gemm1+silu through it (MINISGL_MOE_FUSED_SILU,
default on).
- **KEY SURPRISE vs the shelving verdict:** reusing the WMMA path's exact epilogue helper `moe_silu_and_mul_h`
  (gate/up rounded through AT, silu fp32→AT, AT*AT) makes the fused output **BIT-EXACT** to unfused-gemv +
  that same canonical silu — NOT tol-only as predicted. Parity max=0.000e+00 across int4 sym/asym + e2m1,
  fp16/bf16, T∈{1,2}, up to the 35B gemm1 shape. The gemv accumulation is byte-identical to the unfused gemv
  (same code path), and the only epilogue difference vs the OLD tail_hip.silu_and_mul is a 1–2 ULP silu-order
  choice — but the fused matches the WMMA fused path (already bit-exact to torch _C.silu_and_mul) exactly.
  So the silent-corruption risk that made this attended-only is GONE (bit-exact, not tolerance-gated).
- **Perf:** op-level 1.24x (35B T=1, 50.9→41.0us, −9.8us/MoE-layer/token) .. 1.47x (E=64 inter=512); +at T=2.
- **Serve:** Qwen3.6-35B-A3B-AWQ TP=2 under GRAPH CAPTURE — COHERENT (5/5 greedy), and both
  `[hip-engage] mmq_fp8_moe_gemm1_silu(gemv)` (decode) AND `(wmma)` (prefill) fire DURING capture.
- Commits: kernels `ff5a611`, engine `68e1c74`. Harness: tools/moe_gemm1_silu_gemv_{parity,bench}.py,
  moe_gemm1_silu_parity_run.sh, moe_fused_silu_smoke.sh.
- **Remaining backlog:** (2) qkv-merge full-attn (bit-exact-able, 1/4 layers); (3) dense/shared SwiGLU
  (same restructure class as #1, now proven bit-exact-able via moe_silu_and_mul_h — de-risked).

## ATTENDED FOLLOW-UP — backlog #3 (dense/shared SwiGLU) SHIPPED (2026-07-18, user-directed)
"Continue clearing backlog." Built the DENSE analog of #1: `mmq_fp8_gemv_decode_silu_kernel` — same K-tiled
decode GEMV as mmq_fp8_gemv_decode, each warp owns one FUSED col j<inter computing gate (weight col j) + up
(col j+inter) sharing the staged fp8 activation, writes silu(gate)*up → (M,inter). New op
`mmq_fp8_gemm_silu` (launch_mmq_fp8_gemm_silu_gfx1201: act-quant + gemv-silu dispatch). Reused
moe_silu_and_mul_h verbatim (replicated as w4a8_silu_and_mul_h) → **BIT-EXACT** to (mmq_fp8_gemm gate|up) +
silu (parity max=0.000e+00: int4 sym/asym + e2m1, fp16/bf16, M∈{1,2,4}, incl K=1536/group=32).
- **Plumbing (safe-fallback design):** Linear.forward_swiglu prefers the method's apply_swiglu (fused decode)
  and falls back to the BIT-IDENTICAL silu_and_mul(forward(x)) → swapping a call site is always safe.
  apply_swiglu added to W4A8LinearMethod + MxFp4LinearMethod; kernels.w4a8_linear_silu; env
  MINISGL_DENSE_FUSED_SILU (default on). Call sites swapped: GLM-4.7-Flash shared expert (the validated
  target — quantized merged gate_up, K=1536, every layer) + Qwen2-MoE shared expert.
- **Perf:** op-level 1.19x-1.30x decode (2.5-7.5us/MLP/token).
- **Serve:** GLM-4.7-Flash-AWQ TP=2 under GRAPH CAPTURE — COHERENT 5/5 (thinking off), `[hip-engage]
  mmq_fp8_gemm_silu(int4)` fires DURING capture. (First pass showed thinking-mode CoT preambles at 40 tok →
  probe artifact, NOT a fusion bug; confirmed by disabling thinking.)
- Commits: kernels `be85bfe`, engine `f6a8bb4`. Harness: tools/w4a8_gemm_silu_{parity,bench}.py,
  glm_fused_silu_smoke.sh.
- **Remaining backlog:** (2) qkv-merge for full-attn layers — bit-exact-by-construction (weight concat, 3
  gemms→1), pure plumbing (reuse LinearQKVMerged + a 3-way weight-loader merge), but conditional ROI (full-
  attn layers only; GDN-hybrid models are ~1/4 attn) and the qwen3_5 q_proj gate-interleave adds fiddliness.
