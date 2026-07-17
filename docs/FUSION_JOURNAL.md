# Kernel-fusion overnight loop — charter + journal

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
