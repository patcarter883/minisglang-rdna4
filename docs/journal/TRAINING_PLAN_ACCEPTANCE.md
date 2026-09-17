# Training plan — lift TiDAR spec acceptance (round 2, dump-informed)

> **Revised 2026-07-03 after the fused-replica DUMP** (see `CONTINUE_FUSED_TIDAR.md` §"DUMP RESULT").
> Decisions locked: (1) **commit to the fused / bidirectional path** as the train+serve target;
> (2) **regenerate a reasoning on-policy corpus**; (3) **local de-risk (Step 0) gates the cloud burst.**

## Where we are (round 1 done)

- Round-1 OPD burst ran (4× A100-80 RunPod, 8k generic-rollout windows, ~1000 DP steps, batch 8,
  warm-start `pat883/zaya1-tidar-megatron` → `pat883/zaya1-tidar-opd`). Served fp8 = the current
  `/home/pat/code/_big/zaya1-tidar-opd-fp8`.
- Measured: **two-forward (causal)** accept **0.18–0.19**, emitted/step 1.72–1.76, **16.9 tok/s =
  0.67× AR** (25.2 tok/s), MI300X A/B. **fused (bidirectional)** accept **0.07–0.09**, emitted/step 1.27
  (local gfx1201). Acceptance is HW-independent (same weights), so the 0.18 vs 0.08 gap is real.
- Round-1 per-position lift: pos1 +4.7, pos2 +4.7, pos3 +3.2, **pos0 −1.6** (regressed).

## What the DUMP changed (read this before touching the recipe)

1. **Train==serve is ALREADY aligned for fused/bidir.** `tidar_mcore.py::tidar_block_loss` optimizes the
   *bidirectional* diffusion block (`diff_acc`, `diff_acc_by_pos`); the FUSED path serves exactly that
   block. The *causal* two-forward path is the mismatched one — and it's architecturally slower-than-AR
   (2 forwards) no matter how high acceptance goes. **→ Fused/bidir is the target; two-forward is only a
   lossless fallback.** (Decision 1.)
2. **The fused low-acceptance is STRUCTURAL, not a bug.** At k=0 (7/8 steps today) the committed tail
   ends in the *bonus*, which is never an input embedding to any replica (`R_k[0]` is a mask token that
   only *predicts* it). Two-forward re-drafts on `committed+bonus` with the real token → fused < two-fwd
   **by construction**, worst at low acceptance. The cap SHRINKS as `diff_acc` rises (k=0 stops
   dominating, `R_k` then conditions on correctly-accepted drafts). **→ raising `diff_acc` is the lever
   that lets fused overtake two-forward.** No `r_sel`/RoPE/conv fix recovers it — do NOT chase those.
3. **pos0 is load-bearing for FUSED and round-1 regressed it (−1.6).** `R_k[0]` = the replica's pos0 =
   the bonus prediction; the dump measured `R_k[0]==bonus` only **4/8**. Two-forward doesn't care (it
   gets the bonus from the causal AR row), but fused critically depends on pos0. **→ round-2 must PROTECT
   `diff_acc_by_pos[0]`, not trade it for later positions.** This is the single most dump-connected knob.

## Step 0 — LOCAL DE-RISK (free, no cloud; gates the burst) — Decision 3

Before spending cloud budget, confirm the two premises the whole plan rests on. Both are local
(gfx1201, `gpu-lease -n 1`), reusing the dump harness — no training.

- **0a. Is the cap purely a k=0 effect?** Bucket *this step's* acceptance by the k that PRODUCED the
  carried drafts (the replica-selection k of the prior step). Implemented as `_fused_stats["by_prev_k"]`
  in `scheduler.py::_spec_decode_step_tidar_fused` (logged under `MINISGL_SPEC_DEBUG=1`).
  **Green-light criterion:** acceptance on drafts produced after a k≥1 step ≫ acceptance after a k=0
  step → the cap is k=0-specific and *will* dissolve as `diff_acc` rises. If fused stays low even after
  k≥1 steps, the fused approach has a deeper problem → reconsider before spending.
- **0b. Does fused beat AR TODAY?** Run the fused window and compare tok/s to plain AR on the same card
  (`tools/run_tidar_window.sh` baseline-vs-fused). If fused < AR even at emitted/step 1.27, the
  TiDAR-for-speed thesis needs the accept lift just to reach parity — this sizes the bar training must
  clear.

Record 0a/0b in `CONTINUE_FUSED_TIDAR.md`. Only on a green 0a do we launch the round-2 burst.

**0a RESULT (2026-07-03) — GREEN, decisive.** Fused SEG run to step 500, `accept-by-prior-k`:
`prevk=0: 0.03   prevk=1: 0.10   prevk=2: 0.39`. Acceptance jumps 3–13× once the prior step accepts
≥1 draft — and at k=2 (0.39) it EXCEEDS the two-forward 0.18. This is the exact signature the plan
predicted: the fused cap is entirely the k=0 bonus-conditioning effect; escape k=0 and fused drafts
are excellent. ⇒ raising base `diff_acc` (training) will cascade fused acceptance upward — the burst
is justified. (Buckets need ≳step-400 to populate; k≥1 events are rare at 0.07 base accept, so run
long. Instrumented as `_fused_stats["byk"]`, logged `[spec-fused-0a]` under `MINISGL_SPEC_DEBUG=1`.)

**0b RESULT (2026-07-03) — RED. STOP: acceptance is NOT the binding constraint; per-step COST is.**
Measured fused single-forward vs AR (local gfx1201, GENTOK=256, median of 3, `tools/run_ab_window.sh`
FUSED=1 SEG=1 OLDMOE=1): **AR 25.0 tok/s ; fused 3.9 tok/s = 0.16× AR** (~287 ms/step vs AR's ~40 ms).
The fused forward is ~7× an AR decode: it drafts all `1+B+B²` (≈21, or 29 segmented) masked query
tokens EVERY step, eager (`--graph 0`), + a Python-built custom mask + per-step CCA verify-state
capture. That per-step cost is FIXED (independent of acceptance), and max emitted/step = B+1. So even
at 100% acceptance, B=4 fused = 5 tok / 0.287 s = **19 tok/s < 25 AR** — training acceptance CANNOT
make B=4 fused beat AR. **The earlier "fused is per-forward ahead" reframe was WRONG** (the fused
forward is NOT bandwidth-bound-equal to a 1-token AR decode). Two-forward is 0.67× AR; fused is worse.

**STEP 0.5 RESULT (2026-07-04) — the cost is ONE thing: the custom-mask attention kernel.**
Instrumented `MINISGL_TIDAR_TIME=1` step breakdown + swept {flat,seg}×B∈{2,4} (`tools/run_fused_cost_sweep.sh`):
```
[spec-fused-time] ms/step   stage=2.2   forward=284.2   commit=1.6   total=287.9
segB4 3.9 · flatB2 4.2 · segB2 3.9 tok/s   (ALL ~0.15× AR; AR 25.2)
```
- **Forward = 98.6% of the step (284 ms).** Mask-build (2.2 ms) and CCA capture/install (1.6 ms) are
  noise. The CPU-mask fix (built the seg allow-matrix on CPU vs ~n² per-cell GPU writes) landed and is
  correct (mask tests pass) but was NEVER the bottleneck.
- **B and flat/seg don't matter** — flatB2 (7 query toks) ≈ segB4 (29) ≈ 284 ms. The forward is
  **fixed-cost, independent of query-token count** ⇒ the B² replicas are NOT the cost; B-sweep can't help.
- **Root cause = the custom-mask attention path.** Fused runs `_hip_prefill_paged_op` with `causal=0`
  + a dense `mask_bias` (triton_rdna4.py:226); the two-forward *verify* forward uses the same kernel
  with `causal=1` (triangular skip) and runs ~51 ms. The ~233 ms delta (~5.8 ms/layer × 40) is the
  custom-mask kernel path being ~5–6× the causal one. NOT an SDPA fallback — the HIP kernel itself is
  slow for `causal=0 + mask_bias` (bad tiling/occupancy for the small-q shape and/or no fully-masked
  key-block skip). cudagraph will NOT fix this (284 ms is compute, not launch overhead).

**PROFILE RESULT (2026-07-04) — CORRECTION: the fused step is CPU-DISPATCH-BOUND, not compute-bound.**
`torch.profiler` over 8 fused steps (`MINISGL_TIDAR_PROFILE=1`, table + trace `tools/fused_prof.pt.trace.json`):
```
hipLaunchKernel   70.5%  1.663s  74752 calls  → ~9,300 kernel LAUNCHES / step
aten::to/_to_copy  45%   1.06s   ~38k calls   → ~4,800 dtype CONVERSIONS / step
aten::copy_        45%   1.05s   34256 calls  → ~4,300 copies / step   (2.36s/8 ≈ 295ms/step)
```
The ~284 ms is NOT a slow compute kernel — it's eager-mode CPU dispatch: ~9,300 launches + ~4,800
conversions + ~4,300 copies PER STEP. This overturns the "compute-bound, cudagraph won't help" call
above — launch-bound is EXACTLY what cudagraph fixes, and it also explains B-independence (launch count
~constant in B) and the 5–6× vs the causal verify (fused issues ~5× more ops/step). Leading suspect for
the op flood: `OLDMOE=1` (fp8→bf16 weight dequant PER FORWARD, per-expert-per-layer = thousands of tiny
kernels + conversions). Config-only test in flight: fused WITHOUT OLDMOE — if the step collapses, the
dequant is the cost. Two fixes now on the table, both promising:
  1. **Cut the op flood** (cheapest, no graph risk): if OLDMOE dequant is it, route the fused MoE through
     the native fp8 WMMA GEMM (no per-forward dequant) — see [[w8a8-fp8-moe-uses-native-wmma]] — or cache
     the dequant. Kill the ~4,800 conversions/step.
  2. **cudagraph-capture the fused step**: eliminates dispatch overhead wholesale; blocker is graph-safe
     CCA state capture (serve is --graph 0 for CCA today).

**PROBE RESULT (2026-07-04) — the cost is the OLDMOE dequant, and native fp8 fused is VIABLE.**
Fused TIME breakdown WITHOUT OLDMOE (native fp8 WMMA MoE) vs with:
```
OLDMOE=1 (fp8→bf16 dequant/forward):  forward = 284 ms   (the ~9,300-launch op-flood)
OLDMOE=0 (native fp8 WMMA):           forward =  55 ms   → ~1.4× an AR decode (40 ms)
```
So the per-forward fp8→bf16 weight dequant is ~230 ms (~83%) of the step; native fp8 is 55 ms. At
55 ms/forward, fused hits **~1.35× AR at a trained accept≈1, ~2× AR at accept≈2** — viable. The
original 0b "0.16× AR" was apples-to-oranges (fused ran OLDMOE=1 dequant-flood; AR baseline ran native
fp8). Apples-to-apples both-native: fused 55 ms vs AR 40 ms — a net win once training lifts accept past
~0.4. **⇒ The fused path is fundamentally viable; the sole cost blocker is the OLDMOE dequant path.**

### ⇒ FIX MENU for the MoE-precision path (pick per the OLDMOE-correctness question)
OLDMOE=1 exists for CORRECTNESS (fp8 weights + bf16 acts; OLDMOE=0 uses fp8 ACTS which hurt quality).
So the goal is bf16-act correctness WITHOUT the per-forward dequant flood:
  1. ~~**Batched dequant**~~ — TRIED (2026-07-04), DID NOT HELP. Vectorized `_GroupedFP8Experts.dequant`
     (per-expert Python loop → one whole-tensor op) — forward stayed 284→323 ms. So the ~9,300
     launches were a CPU-side artifact that OVERLAPS the GPU compute; the wall-time bottleneck is
     GPU-COMPUTE-bound: OLDMOE=1 materializes the full (E=256,N,K) bf16 stack per layer AND runs the
     bf16 Triton MoE over ALL experts, every forward. Native fp8 (OLDMOE=0) avoids both (grouped fp8
     kernel, activated experts only) → 55 ms. (The profiler table I got was CPU-only/no-CUDA-column, so
     I over-weighted the launch story — should've pulled the GPU-side breakdown via TraceLens.) The
     vectorized dequant is kept (bit-identical, cleaner) but is NOT the fix.
  2. **Fused fp8-weight × bf16-act GEMM** (best): a W8A16-style MoE kernel that dequants in-register and
     GEMMs — no separate dequant, no act-precision loss. **BUILT 2026-07-04** (`moe_w8a16_wmma/`): adapted
     the validated bf16 grouped-MoE WMMA kernel (`moe_bf16_wmma`, rocwmma) to take fp8 e4m3 weights +
     per-output-channel scale, dequant fp8→bf16 IN-REGISTER during B-staging (exact widen, unscaled),
     bf16 acts, bf16 WMMA, weight-scale in the epilogue; routed experts only. Compiles clean
     (`moe_w8a16_C.so`). Wired into `moe.py` behind `MINISGL_ZAYA_W8A16=1` (uses the always-present
     `_w_op`/`_scales_op`). **VALIDATED 2026-07-04 ✅**: parity PASS (gemm1 cos 0.999997, gemm2 cos
     0.999999 vs dequant ref); fused step **87 ms** (3.1× faster than OLDMOE=1's 284 ms) with **bf16-act
     correctness** — 3/4 prompts exact, divergence only at char 271 (late ULP, matches/beats OLDMOE=1's
     char-135; far better than native-fp8's char-49). At 87 ms/step fused ≈ 1.3× AR @trained accept≈2,
     ~2.2× at perfect accept. **AUTOTUNED 2026-07-04**: swept (block_m,BN)
     at the real ZAYA MoE shapes (E16/K2048/inter4096/top1, M=29; `moe_w8a16_autotune.py`) → winner
     **block_m=16 BN=32 = 2.15ms/layer vs 64/128's 3.65ms (1.70×)**; minimal moe_align padding at the
     tiny-M top-1 decode. Locked as the default (env `MINISGL_W8A16_BLOCK_M`/`_BN`). Re-measured fused:
     **forward 56 ms = native-fp8's 55 ms**, losslessness UNCHANGED (3/4 exact, char-271 ULP). So W8A16
     now delivers **native-fp8 speed WITH bf16-act correctness** — 5× faster than OLDMOE=1's 284 ms.
     At 60 ms/step fused ≈ 1.3× AR @trained accept≈1, ~2× @accept≈2, ~3.3× at ceiling.
     **⇒ COST PIVOT FULLY CLOSED: fused has native-fp8 speed + correctness; training is the lever.**
  3. **Accept OLDMOE=0** if the native-fp8 correctness hit is tolerable — cheapest of all; needs a
     coherence/quality check on the TiDAR model first (NEXT probe).
  4. **cudagraph-capture** — still a valid orthogonal win (removes remaining dispatch), but no longer
     required now that native-fp8 fused is already 55 ms.
Since #1 (batched dequant) is out, the real options are #2 (fp8w×bf16a fused MoE kernel — substantial,
but the native grouped fp8 kernel already exists at 55 ms; making it W8A16 mixed-precision is the crux
work) or #3 (accept native fp8 OLDMOE=0, 55 ms, weaker correctness). **NEXT (decision-maker): quantify
OLDMOE=0's correctness hit on the TiDAR model** (baseline OLDMOE=1 vs OLDMOE=0 text diff + coherence).
If tolerable → ship native-fp8 fused, viable NOW → training becomes the lever again. If not → fund the
W8A16 MoE kernel. Early signal from the OLDMOE=0 fused run: output coherent (Paris ✓, 17+26=43 ✓) but
the fused-vs-greedy lossless guarantee weakened (diverged char 49/54 on 2/4 prompts vs OLDMOE=1's later
single divergence) — fp8-act noise. So OLDMOE=0 trades some spec-losslessness for 5× speed.

### ⇒ (earlier, now CORRECTED×2) guesses: attention-kernel-slow → dispatch-bound → actually the OLDMOE dequant flood
The fused block structure REQUIRES a masked attention, so this kernel is on the critical path and is
the ONLY thing between fused and viability. Ceiling math: at the current 284 ms/forward, B=4 fused caps
at 5/0.284 = 17.6 tok/s < 25 AR even at 100% accept — DEAD. If the masked kernel reaches causal-kernel
speed (~51 ms), fused at a trained accept≈1 → 2/0.051 ≈ 39 tok/s ≈ **1.5× AR** — attractive. So fused
viability hinges on a HIP-kernel optimization (est: profile w/ torch.profiler — rocprof is dead on
gfx1201; then give the `mask_bias` path proper tiling + skip fully-masked key blocks, or write a
decode-shaped masked kernel). THIS gates training — acceptance is worthless until the kernel is fixed.
**Decision needed (user): fund the masked-attention kernel work, or ship base AR (25 tok/s) + shelve
fused.** The pre-existing cost pivots below are SUPERSEDED by this finding (B/mask-build/graph ≠ lever).

### ⇒ (superseded) earlier COST guesses — kept for the record; the breakdown ruled them out
Do NOT launch the round-2 burst — acceptance is the wrong lever until the fused forward is competitive.
Levers on COST (measure `tools/run_ab_window.sh` FUSED=1 tok/s after each):
- **Small B (B=2).** Fused cost ∝ B² replicas; B=2 → 4 replicas (vs 16). At B=2, max emitted=3; if the
  step drops to ~AR-cost, 3/step could clear AR. This is the highest-leverage single knob.
- **cudagraph-capture the fused forward.** Serve runs eager (`--graph 0`) for CCA; eager launch
  overhead across ~29-token tensors × all layers is a big chunk. Capturing the fused step (if the CCA
  state capture can be made graph-safe) should cut step time materially.
- **Kill the Python mask build.** `fused_paged_layout_segmented` builds an O(n_query²) bool matrix in a
  Python double loop, called twice/step — precompute the shape-invariant structure once (only ctx
  abs_pos varies with c0) and reuse.
- **Cheaper CCA verify-state capture** (per-step unfold reconstruction) if it shows up in a profile.
**Gate 0.5:** fused step time low enough that `(achievable_accept+1)/step_time > AR`. Only then does the
acceptance-training round below become the lever. If cost can't be brought under AR-competitive even at
B=2 + graph, TiDAR-fused is a dead end on this stack → serve base AR (25 tok/s) and shelve fused.

## Round-2 recipe (what to change vs round 1) — GATED on Step 0.5 clearing

Priority order, all in ONE burst after Step 0 is green:

### L-A (top) — PROTECT pos0 while lifting later positions
Round-1 loss shaping regressed pos0. Re-weight the per-position diffusion CE so pos0 is not sacrificed
(e.g. a floor weight on pos0, or a uniform-then-late-tilt schedule instead of pure late-tilt). Add
`diff_acc_by_pos[0]` as a hard gate (must be ≥ round-1 baseline). This is the fused bonus-prediction
quality — the dump's headline lever.

### L-B — Reasoning on-policy corpus — Decision 2
Regenerate the greedy-rollout corpus from **reasoning prompts** (math / code / multi-step QA producing
long `<think>` CoT), **on-policy from the fp8 served model** (`/big/zaya1-tidar-opd-fp8`, the serving
target). Rollouts run LOCAL under `gpu-lease` (2× gfx1201). Scale windows 8k → **16–32k**, longer
windows (reasoning traces are long — exactly where spec pays). Replaces `opd_rollouts.jsonl`.

### L-C — Scale
Steps 1k → **2–4k**; batch 8 → **16** (round-1 only used ~44% of a100-80 VRAM). Cheap, near-monotone
until plateau.

### L-D — Block-size sweep
Sweep B ∈ {2,4,6}. Smaller B shrinks the k=0 replica waste in fused (fewer wasted mask replicas per
step) and the B² fused cost; bigger B = more drafts but lower per-pos accept. Pick B at the fused
throughput knee (measure with the fused path, not two-forward).

### L-E (last, diminishing) — loss shaping / distillation temperature
Only after L-A..C land and if there's headroom.

## Gates (every round — reuse existing harnesses)
- **pos0 not regressed:** `diff_acc_by_pos[0]` ≥ round-1 (NEW hard gate, dump-driven).
- **Acceptance win, measured on FUSED:** fused accept up vs round-1 AND trending to cross the
  two-forward 0.18 (the signal the structural cap is being overcome). Use `MINISGL_SPEC_DEBUG` +
  the Step-0 k-bucket, not just the offline `diff_acc`.
- **Reasoning preserved:** `eval_regression.py` + a `<think>`/CoT coherence probe + math/code accuracy —
  an OPD that dents reasoning is a FAIL even if acceptance rises (ZAYA1-8B is the reasoning model).
- **Live throughput:** fused tok/s vs AR (`tools/run_tidar_window.sh` fused mode) — must exceed 1.0× AR
  to justify TiDAR over just serving base AR. This is the bottom line.

## Sequencing
**Step 0 (local de-risk) → [green] →** one 4× A100-80 burst: **L-A pos0-protect + L-B reasoning corpus +
L-C scale(2–4k / batch16)** → gates → **L-D B-sweep** → re-measure fused A/B. Warm-start from
`pat883/zaya1-tidar-opd` (round-1 output), push to a NEW repo (never clobber the warm-start source).
**Cloud spend is USER-GATED — no burst without an explicit go.**

## Infra (unchanged, already hardened — see [[opd-cloud-pipeline-hardened]])
One-command 4× A100 RunPod burst via `cloud-lease` (prestage-then-lease); regression-eval +
fp8-acceptance eval already wired. Corpus regen is local+leased (fp8 rollouts on the 2× gfx1201 box).
No cloud has gfx1201 → serving/capture stays local; only the arch-agnostic Megatron training is cloud
(MI300X/A100). Cost sketch: ~$5–15/burst (batch16, 2–4k steps), off-meter prestage.
