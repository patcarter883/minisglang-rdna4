# Continuation — fused single-forward TiDAR on minisgl-rdna4 (Phase C)

**Read this first to resume.** Worktree `/home/pat/code/minisgl-rdna4-tidar-spec`, branch
`feat/tidar-spec` (off `rdna4`). **Everything below is UNCOMMITTED** — commit on the feature branch
when ready (never `main`). Sister design docs: `docs/PHASE_C_FUSED_FORWARD.md` (algorithm + seams),
`docs/TRAINING_PLAN_ACCEPTANCE.md` (post-fused OPD round).

## Goal
Serve the OPD TiDAR diffusion ZAYA (a diffusion version of the **reasoning** ZAYA1-8B) on minisgl with
a **fused single-forward** spec-decode = one forward per step that BOTH verifies the previous block's
drafts AND pre-drafts the next block from B replicas → ~2× AR throughput. The two-forward path is
built + lossless but only ~parity throughput (2 forwards). Fused halves the forwards.

## What is DONE and validated
- **Phase B two-forward TiDAR spec: lossless, works.** accept 0.17–0.27, emitted/step 1.74–2.08,
  ~parity throughput. `--spec-algorithm tidar` (default, i.e. `MINISGL_TIDAR_FUSED` unset). This is
  the SHIPPABLE TiDAR spec path today.
- **B.0 CCA verify-state (no HIP kernel):** conv_state is raw qk_new rolling window → reconstructed in
  torch (`cca/metadata.py::capture_cca_verify_state`), `cca_state.py::install_verify_state`, wired in
  `scheduler.py::_spec_decode_step`. CPU test `tests/cca_verify_state_test.py` (bit-exact).
- **C.1 kernel — `mask_bias` in `attn_prefill_paged`** (bf16 path): kernel.hip + bindings.cpp + op.py.
  A fused call passes `causal=0` and the mask carries the block structure. GPU-validated **0 ULP** vs
  SDPA (`attn_prefill_paged/attn_prefill_paged_mask_test.py`). ⚠ The `.so` is rebuilt IN THE WORKTREE
  (`GPU_ARCHS=gfx1201 python setup.py build_ext --inplace` in the combined image); a fresh checkout
  must rebuild it. fp8-KV path NOT wired (asserts no mask) — serve uses bf16 KV, fine.
- **C.2/C.5:** `RDNA4Metadata.custom_mask` + threaded through `HIPAttnBackend._hip_prefill_paged`
  (`attention/triton_rdna4.py`).
- **C.3 mask math:** `spec/tidar_mask.py` (`fused_paged_layout`, `fused_paged_layout_segmented`,
  positions §7.6, logit split). CPU test `tests/tidar_mask_test.py` (all pass).
- **C.4 fused step BUILT + RUNS:** `scheduler.py::_spec_decode_step_tidar_fused` behind
  `MINISGL_TIDAR_FUSED=1`. Lossless-ish (3/4 prompts; residual is ULP — see below). ~1.2× AR at flat.
- **Serving-config decision (validated):** ZAYA fp8 = OUR W8A8 (fp8 weights + fp8 per-token dynamic
  acts, MoE experts only; Zyphra ships bf16-only, no official fp8). Activation quant was perf-only,
  buys ~zero memory (weights dominate), and hurt correctness → **serve with fp8 weights + bf16
  activations** via `MINISGL_ZAYA_OLDMOE=1` (fp8 stays in HBM 9 GB, dequant-per-forward to bf16).

## THE WALL (where to resume) — fused replica drafting gives near-zero acceptance
The fused path's next-block drafts come from the B mask-replicas `R_0..R_{B-1}` (`R_k` = drafts if k
accepted). Measured acceptance is **~0.03** (fused) vs **0.20** (two-forward) — the replicas produce
near-useless drafts, so the fused speedup isn't realized (emitted/step ~1.1). FOUR fix attempts each
moved it only ~0.01–0.02:
  flat conv 0.05 → self-attn ctx 0.01 → causal ctx 0.01 → `r_sel=k+1` selection 0.03.
This plateau ⇒ MULTIPLE stacked bookkeeping bugs in the replica mechanism (conv context + replica
position/selection alignment + likely RoPE-position/mask details). **Stop blind fixing.**

### DUMP RESULT (2026-07-03) — the wall is STRUCTURAL, not a bookkeeping bug
Built `MINISGL_TIDAR_DUMP=1` (a position-by-position draft-vs-reference dump in
`_spec_decode_step_tidar_fused` + `_tidar_fused_dump`; the reference `--check-drafts` gate). Run:
`FUSED=1 SEG=1 OLDMOE=1 DUMP=1 MAXTOK=64 gpu-lease -n 1 -- bash tools/run_tidar_window.sh`. Findings
from an 8-step SEG dump (accept 0.07, emitted/step 1.27):
- **7/8 steps are k=0.** This OPD checkpoint is undertrained — even two-forward only accepts 0.20, so
  most steps commit ONLY the bonus. The replicas' relative acceptance-length signal is barely exercised.
- **No selection/shift rule reproduces the two-forward `gt`.** The best-matching replica jumps around
  step to step (R_2, R_0, R_0, R_2/3…); neither `r_sel=k`, `r_sel=k+1`, nor a uniform 1-position shift
  dominates. So the plateau is NOT one more mis-indexed `r_sel`/RoPE bug to find.
- **STRUCTURAL CAP (root cause):** replicas are all-MASK blocks. `R_r` conditions on the real token
  embeddings of `[committed|confirmed|drafts[:r]]`. At a rejection boundary (k<B, i.e. almost always
  here) the true committed tail ends in the **bonus**, and the bonus is **never an input embedding to
  any replica** — `R_k[0]` is a mask token that merely *predicts* the bonus (its K/V are the mask
  embedding), so `R_k[1:]` cannot actually see it. The two-forward path avoids this by feeding the real
  bonus token into a fresh `block_predict(committed+bonus)`. Hence fused (0.07) < two-forward (0.20) by
  construction, and the gap is WIDEST exactly in the low-acceptance regime this model sits in.
- **Throughput reframe:** fused already delivers 1.27 emitted/step in **1 forward** vs two-forward's
  ~2.0 in **2 forwards** (=1.0/forward). On a bandwidth-bound 8B decode (weights load once/forward) the
  fused mechanism is per-forward AHEAD and hits its theoretical `accept+1`. **The lever is ACCEPTANCE
  (model-limited), not replica bookkeeping.**

**Fidelity gate result (R_0 vs bp0) — CONFOUNDED, no clean port bug.** Ran `R_0` (fused replica r=0)
vs `bp0 = _tidar_block_predict(pre-step committed)`: match 1/8 EXACT, mostly 2–4/4 (correlated, not
identical). This is EXPECTED and does NOT prove a port bug, because the two use different intra-block
attention: `_tidar_block_predict` is CAUSAL (each mask attends confirmed + earlier masks — the
"causal target forward"), while the fused `R_r` is BIDIRECTIONAL within its block (reference design,
`tidar_mask.py` predicate `k_r==q_r`). Same prefix, different block attention → different drafts. So
the gate can't isolate a paged-port bug; it instead surfaces a genuine (secondary) fact: the two-forward
path drafts CAUSALLY and the fused path drafts BIDIRECTIONALLY, and on this undertrained model the causal
draft happens to accept better (0.20 vs 0.07). That is a minor lever, NOT the fix — the structural cap
below dominates: at k=0 (7/8 steps) `next_drafts=R_{k+1}=R_1` conditions on the just-REJECTED `draft[0]`
(≠bonus), poisoning the block regardless of causal/bidir; and the reference's `r_sel=k` fallback (lean on
`R_k[0]≈bonus` via the bidir block) also fails — `R_k[0]==bonus` held only 4/8. No fused selection
recovers the two-forward conditioning.

**Conclusion:** the fused single-forward MECHANISM is validated; the wall is (a) an undertrained model
and (b) an inherent all-mask-replica cap (the bonus is never an input embedding to any replica), NOT
stacked indexing bugs. Do NOT keep chasing `r_sel`/RoPE/conv — the dump shows there is no such fix.
The real path to ~2× is the acceptance-training round (`docs/TRAINING_PLAN_ACCEPTANCE.md` — raise base
`accept` so fused's `accept+1`-in-one-forward scaling pays off); ship the two-forward path (lossless,
0.20) meanwhile. A distant secondary experiment for the fused path (do only if training stalls): try
CAUSAL replica blocks to match the two-forward's better-accepting draft — expected small bump, not a
close of the 0.07→0.20 gap.

### Do this next: a draft-vs-reference DUMP (systematic)
Instrument `_spec_decode_step_tidar_fused` (behind a new env, e.g. `MINISGL_TIDAR_DUMP=1`) to, on a
few steps, compare — POSITION BY POSITION — the fused replica drafts `R_k` against:
  (a) a fresh `self._tidar_block_predict([req], B, mask_id)` on `[committed+drafts[:k]]` (the
      ground-truth causal block draft), and
  (b) the verify targets `p_ar.argmax`.
This is exactly the reference gate `--check-drafts` in
`/home/pat/code/vllm-gfx1201-tidar-fused/zaya/tidar/single_forward_ours.py` ("R_k == a fresh
block_predict token-for-token"). Read `single_forward_ours.py::build_segmented` + `run_loop` + the
`§7.6 fused_forward_position_ids` docstring in `tidar_mask.py` CAREFULLY — the reference's replica
positions/selection are the source of truth; my minisgl mapping is where the bug(s) live. Likely
suspects to check with the dump: (1) does `R_k[0]` predict the bonus position or the post-bonus
position? (offset/selection); (2) are the replica RoPE positions §7.6-correct on the paged layout;
(3) does the segmented ctx actually reach `R_r`'s conv (packed-neighbour `qk_new[t-tp..t]`).

## Also open (lower priority)
- **ULP losslessness residual:** even bf16 (OLDMOE), the fused path diverges on 1/4 prompts at a deep
  close-decision point (prompt "village", char ~135, reproducible). Ruled out mask/attn/conv/routing;
  it's batch-composition-dependent grouped-MoE reduction-order ULP (see `docs/PHASE_C_FUSED_FORWARD.md`
  + the batch-invariant-kernel discussion). Likely accept as "lossless modulo bf16 ties" OR needs a
  batch-invariant MoE GEMM. Chase only after acceptance is fixed (acceptance matters far more).

## How to run (from the worktree)
```
# fused acceptance/lossless gate (fp8):   FUSED=1 [SEG=1] [OLDMOE=1] [NOREP=1] MAXTOK=128
gpu-lease -n 1 -- bash tools/run_tidar_window.sh          # baseline vs tidar + lossless diff
# envs threaded: FUSED (MINISGL_TIDAR_FUSED), SEG, NOREP (FUSED_NOREP), OLDMOE (ZAYA_OLDMOE)
# [spec-fused] accept lines -> tools/spec_tidar.server.log ; diff verdict -> tools/tidar_window.out
```
Models: `/home/pat/code/_big/zaya1-tidar-opd-fp8` (diffusion, has tidar_config.json), original
reasoning `/home/pat/models/ZAYA1-8B-fp8`, pre-OPD `/home/pat/code/_big/zaya1-tidar-megatron-fork`.
⚠ A fresh worktree lacks the compiled HIP `.so`s — copy from `/home/pat/code/minisgl-rdna4/*/*_C.*.so`
into the matching dirs, THEN rebuild `attn_prefill_paged` (it has the mask_bias source change).
CPU tests: `PYTHONPATH=python python tests/tidar_mask_test.py` / `tests/cca_verify_state_test.py`
inside the combined image (`pip install msgpack` first).

## Fallback if fused stays hard
Ship the two-forward path (lossless, 0.20 accept) + the validated kernel infra. Fused ~2× becomes a
funded follow-on. Then the acceptance-training round (`docs/TRAINING_PLAN_ACCEPTANCE.md`).
```
GPU protocol: every GPU job via `gpu-lease -n 1 --`; stop the CONTAINER not the wrapper
(TaskStop-orphan). Serve safe on this box (ZAYA fp8, one card). Cloud spend USER-GATED.
```
