# Continuance — ZAYA serving throughput (v2 cudagraph port + W8A16) — READ FIRST

Worktree `/home/pat/code/minisgl-rdna4-tidar-spec`, branch `feat/tidar-spec` (published to
`patcarter883/minisglang-rdna4`, private; default branch `rdna4`). Everything below is COMMITTED
(`1ba037d` + `54a68a0`). **Read `docs/ZAYA_SERVING_NORTH_STAR.md` (the overall strategy) and
`docs/V2_CCA_VERIFY_CAPTURE.md` (the S4 plan) first.** Recall memory `zaya-serving-north-star-cudagraph-
lever` + `tidar-spec-live-in-minisgl`.

## One-line context
ZAYA serving is dispatch+BW-bound; the two levers are cudagraph (dispatch) and spec-decode (BW). This
run shipped the cudagraph lever (v0 1.59× + serve default) and W8A16 (the MoE cost fix), and built +
validated v2 S1–S3 (graph-capture the CCA spec-VERIFY forward). **NEXT = v2 S4: the FUSED custom-mask
forward = the ~4× crown jewel.**

## What is DONE + validated (don't redo)
- **W8A16 MoE kernel** (`moe_w8a16_wmma/`): fp8-weight × bf16-act, in-register dequant, routed-experts
  only. Parity cos 0.99999; autotuned block_m=16/BN=32 (1.7×); fused step 56ms (native-fp8 speed) WITH
  bf16-act correctness (fixed the OLDMOE=1 284ms dequant flood). **DEFAULT** ZAYA fp8 MoE
  (`MINISGL_ZAYA_W8A16`, fail-safe to native). Launchers: `moe_w8a16_wmma/run_{parity,autotune}.sh`.
- **Fused-TiDAR acceptance is STRUCTURAL/model-limited** (dump proved it): all-mask replicas can't see
  the bonus; 0.07–0.20 accept until training. Do NOT chase `r_sel`/RoPE/replica bookkeeping. The lever
  is the acceptance-training round (`docs/TRAINING_PLAN_ACCEPTANCE.md`; round-2 = pos0-protect loss +
  reasoning on-policy corpus — `zaya/megatron` `tidar_mcore.py` has the pos0-protect hook already).
- **v0 GREEN**: cudagraph on the CCA AR decode = **1.59×** (23→37 tok/s); all 5 custom HIP kernels
  verified engaged + survive capture (`_hip_engage.py`, `MINISGL_HIP_ENGAGE_LOG`). `tools/run_cca_graph_v0.sh`.
- **Graph capture is the serve DEFAULT** now: `docker-compose.yml serve` = `--cuda-graph-max-bs 16` +
  `MINISGL_MOE_SCATTER=0` (graph-safe). ⚠ SMOKE-TEST the 35B GDN serve under `--graph 16` (v0 proved
  CCA, GLM proves MLA-MoE, GDN capture is implemented — but the 35B specifically wasn't booted under it).
- **v2 S1–S3 DONE + validated byte-identical**: graph-capture the CCA spec-VERIFY forward.
  - S1 `HIPAttnBackend` verify-capture (hip.py: `init_verify_capture`/`_fill_verify_static`/
    `prepare_verify_for_{capture,replay}`, K+1 causal paged-extend).
  - S2 `capture_cca_verify_state` writes conv/prev scratch IN-PLACE (metadata.py) + `CCAVerifyGraphCapture`
    (cca/graph_capture.py: static state_indices/query_start_loc/per-layer scratch/has_initial_state/seg_lens).
  - S3 wired `GraphRunner` (graph.py: build+drive `cca_verify` in `capture_verify_graphs`/`replay_verify`)
    + lifted `engine.py` `is_cca_hybrid` gate (keep page_size=1). **GATE PASSED**: `FUSED=0 W8A16=1 GRAPH=8
    tools/run_tidar_window.sh` → verify graph replays, graph-on output BYTE-IDENTICAL to eager on all 4
    prompts. Two bugs fixed en route: CCAVerifyGraphCapture needs `is_prefill=True`+static
    `has_initial_state` (verify = multi-query varlen, NOT 1-tok decode); guarded zaya.py:307 `.item()`
    host-sync assert with `torch.cuda.is_current_stream_capturing()`.

## v2 S4 DONE + validated byte-identical (2026-07-04) — the fused custom-mask forward is captured
The crown-jewel dispatch-free fused forward is now cudagraph-captured. Implemented additively
(hip.py static max-width mask buffer + full-width-slice constant stride; graph.py
`capture_fused_verify_graphs`/`can_use_fused_verify`/`replay_fused_verify`; engine.py route +
`capture_spec_fused_verify_graphs`; scheduler.py `batch.fused_verify` + `__init__` capture trigger).
GATE PASSED: `FUSED=1 SEG=1 W8A16=1`, GRAPH=8 fused output BYTE-IDENTICAL to GRAPH=0 eager fused on
all 4 prompts, with `fused-verify GRAPH REPLAY engaged` confirming the captured graph actually ran
(no false-PASS from eager fallback). Full write-up in `docs/V2_CCA_VERIFY_CAPTURE.md` §"S4 RESULT".
NOT yet committed — code + docs sit in the working tree. Open items below.

## NEXT after S4 (pick up here)
- **ITL measurement — DONE (2026-07-04).** `TIME=1`, byte-identical output → pure-dispatch delta:
  fused forward 60.8→45.7 ms (**1.33×**), step total 64.3→48.4 ms (**1.33× end-to-end**, tok/s
  20.7→27.6 @ emitted/step 1.33). Forward is ~95% of the step (stage+commit ≈3 ms), so no dilution.
  Compounds with acceptance-training (same 48 ms step @ trained accept ~0.5 ≈ 70 tok/s). Details in
  `docs/V2_CCA_VERIFY_CAPTURE.md` §"S4 RESULT". ⇒ the remaining lever is emitted/step:
- **Acceptance-training round** (the emitted/step lever) — fused accept 0.04-0.09 is model-limited,
  not a kernel issue. Round-2 = pos0-protect loss + reasoning on-policy corpus
  (`docs/TRAINING_PLAN_ACCEPTANCE.md`; `zaya/megatron/tidar_mcore.py` has the pos0-protect hook). This
  is what makes fused overtake AR (today fused graph-on 27.6 < AR graph-on 36.7 purely on emit≈1.33).
- **Multi-req batch (bs>1) fused replay** — S4 replays only at EXACT captured bs (the fused scheduler
  builds input_ids/positions/out_loc/custom_mask over `reqs`, not `padded_reqs`, so a padded batch
  under-fills the static buffers). Non-exact bs falls back to eager (still lossless). To capture the
  padded case, extend the fused staging loop to build the dummy-row tensors (custom_mask/positions/
  out_loc) — the "dummy-row mask semantics" noted as the S4 open risk. Only needed for concurrent
  serving; the serial validation path is bs=1.
- **35B GDN serve smoke-test under `--graph 16`** (still pending — see below).

## (superseded) v2 S4 spec — the fused custom-mask forward, ~4× crown jewel — DONE, see above
Full plan in `docs/V2_CCA_VERIFY_CAPTURE.md` §S4. It is a SEPARATE capture path from the K+1 verify
(different qlen `1+B+B²`, a dense `custom_mask`, `causal=0`). Concrete pieces (all additive; validate at
step 4):
1. `HIPAttnBackend.init_fused_verify_capture(max_seq_len, bs_list, fused_qlen)` — static page_table/
   cache_seqlens/cu_q(=arange*fused_qlen) + a **static custom_mask buffer** `[max_bs*fused_qlen,
   max_pages*ps]` (contiguous). `prepare_fused_verify_for_{capture,replay}` fill them + copy the
   scheduler-built `batch.attn_metadata.custom_mask` into `static[:total_q,:context_len]` + set
   `metadata.custom_mask = static[:total_q,:max_kv]`. Kernel bounds reads by cache_seqlens (stale tail ok).
2. `CCAVerifyGraphCapture` at `Q=fused_qlen` (constructor already parameterised — pass `fused_qlen-1`).
3. `GraphRunner.capture_fused_verify_graphs` (mirror `capture_verify_graphs`) + `can_use_fused_verify`
   (`spec_verify` AND all `extend_len==fused_qlen`) + `replay_fused_verify`. Scheduler calls the capture
   after building the TiDAR proposer (B → fused_qlen via `fused_paged_layout`).
4. Route `_spec_decode_step_tidar_fused`'s `forward_verify` through `can_use_fused_verify`/`replay_fused`.
   Mask-build + verify-state install stay eager.
**GATE:** `FUSED=1 SEG=1 W8A16=1 GRAPH=8 tools/run_tidar_window.sh` — fused graph-on BYTE-IDENTICAL to
eager fused (save the GRAPH=8 tidar.spec.json, run GRAPH=0, diff — the exact method that validated S3).
Open risk to resolve on GPU: the custom_mask stride/shape under a static contiguous buffer + the dummy-
capture mask semantics. NOTE the throughput payoff also needs the acceptance-training round (fused accept
is model-limited today) — S4 removes the dispatch tax, training fills emitted/step.

## Run commands + GPU protocol (MANDATORY)
- Every GPU job via `gpu-lease -n 1 --`; stop the CONTAINER not the wrapper (TaskStop-orphan).
- v0/graph tests: `GRAPH=8 tools/run_cca_graph_v0.sh` (AR) ; `FUSED=0/1 W8A16=1 GRAPH=8 MAXTOK=128
  tools/run_tidar_window.sh` (spec). Env threaded: FUSED/SEG/W8A16/GRAPH/TIME/DUMP/OLDMOE.
- ⚠ Fresh worktree must copy the compiled `*_C.*.so` from `/home/pat/code/minisgl-rdna4/*/` + rebuild
  `attn_prefill_paged` + build `moe_w8a16_wmma` (`GPU_ARCHS=gfx1201 python setup.py build_ext --inplace`).
- Losslessness = graph-on vs graph-off token-diff (the real gate), NOT the harness spec-vs-nospec diff
  (confounded by the pre-existing W8A16 grouped-MoE ULP — a char-271 divergence hit by any W8A16 path).

## What NOT to do
- Don't chase fused replica bookkeeping / `r_sel` (structural cap, dump-proven).
- Don't grind MoE kernel micro-opt or quant-format churn (≤10% of wall); don't touch custom all-reduce
  (dead on RDNA4, no XGMI).
- Don't spend cloud on acceptance training before the fused path is graph-captured AND you've decided
  fused-vs-two-forward (both now graph-capturable; two-forward is lossless, fused needs S4 + training).
