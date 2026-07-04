# v2 — CUDA-graph capture of the CCA (fused-TiDAR) spec-verify forward

North-star v2 (docs/ZAYA_SERVING_NORTH_STAR.md): the crown jewel. Compounds the two orthogonal levers
— a graph-captured *fused* forward is **dispatch-free** (v0's ~1.6×) AND commits **accept+1 tokens per
forward** (spec-decode) → ~4× the eager-no-spec baseline. This is a PORT of the already-working MLA
spec-verify capture to the CCA path, not a research problem.

## What already exists (the template + the enablers)
- **Plain-decode CCA capture — DONE** (commit 3699388, `CCAGraphCapture`): threads conv/prev_hs state
  through static buffers; `prepare_for_capture` → `torch.cuda.graph` → `prepare_for_replay`. v0 proved
  it (1.59×). This is the state-threading pattern the verify path reuses.
- **MLA spec-verify capture — DONE** (commit 4986cea, `GraphRunner.capture_verify_graphs` graph.py:244;
  `MLABackend.init_verify_capture` / `prepare_verify_for_capture` / `prepare_verify_for_replay`). Works
  because MLA verify is UNIFORM K+1-query causal: precompute static `q_seq_idx`/`q_kbound` (row r → seq
  r//(K+1); `q_kbound[r]=cached_len[seq]+qi+1`), refresh only `cached_len`-dependent bits per replay.
- **The generic driver** `capture_verify_graphs` is backend-agnostic: it needs the backend to expose
  `init_verify_capture` + `prepare_verify_for_{capture,replay}` and to read STATIC verify metadata. It
  also builds `VerifyCaptureBuffer` for input_ids/positions/out_loc/logits(/hidden/aux).

## The gate to lift
`engine.py` (spec-decode constraints): `if config.spec_config is not None and not is_mla:` forces
`page_size→1` (keep — per-token rollback) AND `cuda_graph_max_bs→0` (DISABLE capture). v2 removes the
`cuda_graph_max_bs→0` for CCA (keep page_size=1), and routes the fused step through a captured graph
when the batch is graphable.

## Why CCA-fused is harder than MLA (the real work)
The fused forward (`scheduler._spec_decode_step_tidar_fused` → `engine.forward_verify`) is NOT a
uniform K+1 causal verify. Per req it stages `n_query = 1 + B + B²` (flat) or `1 + B + B·(tp+B)`
(segmented) query tokens with:
1. **A custom dense `mask_bias`** `[total_q, max_kv]` (the block-diffusion structure via
   `fused_paged_layout`/`_segmented`), fed to `attn_prefill_paged` with `causal=0`. The mask's block
   structure is FIXED per (B, layout), but its `max_kv = c0 + n_query` GROWS each step (prefix causal
   part). → capture needs a STATIC max-width mask buffer, refreshed per replay (values change, shape
   fixed); the kernel already bounds reads by `cache_seqlens`, so a stale tail is fine (same trick as
   plain decode / MLA).
2. **§7.6 RoPE positions** — static per (c0, B): `[c0] + fused_forward_position_ids`. Only the `c0`
   offset changes per step → refresh a static positions buffer (add the running `c0`).
3. **CCA recurrent state read + verify-state CAPTURE** — `build_cca_metadata(..., capture_verify_state=
   True)` + `install_verify_state` after accept. The conv/prev_hs slots + the capture scratch
   (`conv_scratch`/`prev_scratch`) must be static buffers the captured forward writes and the eager
   post-step install reads (extend `CCAGraphCapture` to the verify shapes).
4. **Fixed query count** — `n_query` is fixed per B, so `total_q = bs·n_query` is a fixed capture shape
   (like MLA's `bs·(K+1)`). Partial/finished reqs pad to a dummy (like `pad_verify`).

## Incremental plan (cheapest-validated-first; each step has a gate)
**S1 — HIPAttnBackend verify-capture for the STANDARD (non-fused, two-forward) CCA verify.** The
two-forward path's verify forward (forward#2) is K+1 causal (NO custom mask) — the closest analog to
MLA. Add `init_verify_capture`/`prepare_verify_for_{capture,replay}` to `HIPAttnBackend` threading the
paged page_table + cache_seqlens static buffers (reuse the plain-decode capture's `_fill_decode_static`
generalized to qlen=K+1). Gate: capture the two-forward verify, token-diff vs eager (lossless) + ITL.
**S2 — CCA state through static verify buffers.** Extend `CCAGraphCapture` (or a verify sibling) so the
verify forward's conv/prev_hs read + `capture_verify_state` land in static buffers `install_verify_state`
reads eagerly after replay. Gate: two-forward CCA spec, graph on, token-identical to eager.
**S3 — lift the engine gate for CCA** (keep page_size=1) + wire `can_use_verify_graph`/`replay_verify`
into the CCA spec scheduler path. Gate: end-to-end two-forward CCA spec captured, ITL win measured.
**S4 — the FUSED custom-mask forward.** Add the static max-width `mask_bias` + §7.6 positions buffers;
`prepare_verify_for_replay` recomputes the mask/positions for the current `c0` (eager, outside the
graph) into the static buffer. Route `_spec_decode_step_tidar_fused` through the captured graph when
graphable. Gate: fused CCA spec, graph on, token-identical to eager fused (the losslessness gate) +
the ~4× ITL. This is the crown jewel; S1–S3 de-risk the state/threading before the mask complexity.

## Non-negotiables
- **Losslessness gate every step** — capture must be token-identical to the eager fused path (the
  existing `tools/run_tidar_window.sh` prefix-exact diff). A graph that drifts is a fail.
- **page_size stays 1** for CCA spec (per-token rollback); only the graph disable is lifted.
- **MoE decode must be graph-safe** (`MINISGL_MOE_SCATTER=0`, already the serve default now).
- Fixed capture shapes: one graph per (bs) at fixed `n_query`; partial-K/finished steps fall back to
  eager (like `can_use_verify_graph` / `pad_verify`).

## Progress
- **S1 DONE (2026-07-04):** `HIPAttnBackend.init_verify_capture` / `_verify_metadata_static` /
  `_fill_verify_static` / `prepare_verify_for_{capture,replay}` (hip.py) — mirrors `MLABackend`, K+1
  causal paged-extend, no custom mask. Additive, inert until S3.
- **S2 DONE (2026-07-04):** (a) `capture_cca_verify_state` (metadata.py) now writes conv/prev scratch
  IN-PLACE into a pre-bound persistent buffer (falls back to fresh-alloc when absent = unchanged eager
  behaviour) — the fix that makes it capturable. (b) `CCAVerifyGraphCapture` (cca/graph_capture.py) —
  static state_indices + query_start_loc(=arange*Q) + per-CCA-layer conv/prev scratch [Q,max_bs,C,TP]/
  [Q,max_bs,hidden] + fixed seg_lens=[Q]*bs; `prepare_verify_for_{capture,replay}` bind them into the
  metadata. Additive, inert until S3.

## S3 — wire it end-to-end + FIRST validation (the next increment)
Now that S1 (attn) + S2 (CCA state) provide the static buffers, wire the driver + gate:
1. **GraphRunner.capture_verify_graphs** (graph.py:244): after `attn_backend.prepare_verify_for_capture`,
   ALSO call the CCA verify capturer's `prepare_verify_for_capture` (and GDN's, symmetrically) so the
   captured verify forward's recurrent-state + scratch land in static buffers. Construct
   `CCAVerifyGraphCapture(device, max_bs, num_draft, cca_layer_ids, conv_dim, conv_width, hidden)` from
   `cca_state` (conv_dim=1280, conv_width=2, hidden=2048, 40 layers for ZAYA) alongside the existing
   `cca_capture`. Mirror in `replay_verify` (graph.py:337): call `cca_verify.prepare_verify_for_replay`
   before `g.replay()`.
2. **Lift the gate** (engine.py spec-decode constraints): drop the `cuda_graph_max_bs → 0` override for
   CCA (and GDN) — keep `page_size → 1`. Guard so only recurrent backends with a verify capturer engage.
3. **Scheduler**: route the CCA spec verify forward through `can_use_verify_graph`/`replay_verify` when
   graphable (uniform K drafts, fits bs) — first on the TWO-FORWARD path (`_spec_decode_step`), which is
   standard K+1 causal (no custom mask) = exactly what S1 handles.
4. **VALIDATE (the gate):** two-forward CCA spec, `--graph N`, token-identical to eager
   (`tools/run_tidar_window.sh` prefix-exact diff) + ITL win. This is the first end-to-end proof the
   whole verify-capture chain (S1 attn + S2 CCA state + install) is correct.

## S3 RESULT (2026-07-04) — GREEN, proven bit-identical.
Wired `GraphRunner` to build + drive `CCAVerifyGraphCapture` in `capture_verify_graphs`/`replay_verify`,
and lifted the `engine.py` gate for `is_cca_hybrid` (keep page_size=1). Two bugs found + fixed on the
first run: (1) `CCAVerifyGraphCapture` must set `is_prefill=True` + a static `has_initial_state` buffer
(the verify is the multi-query varlen path per `build_cca_metadata`, NOT the 1-token decode path — the
`[bs*Q, hidden]` vs `[bs, hidden]` crash); (2) guarded the prefill path's `.item()` host-sync assert
(zaya.py:307, forbidden mid-capture) with `torch.cuda.is_current_stream_capturing()`. Result: verify
graph captures clean (qlen=5, sizes [1,2,4]); `verify_graph_replays` climbs (both block-predict AND
verify replay); accept 0.22 == eager; and the DECISIVE gate — graph-on two-forward output is BYTE-
IDENTICAL to eager on all 4 prompts (`GRAPH VERIFY == EAGER VERIFY: PASS`). The recurrent-state-through-
a-captured-verify-graph machinery (S1 attn + S2 CCA-state static buffers + S3 wiring) is correct.
Run: `FUSED=0 W8A16=1 GRAPH=8 tools/run_tidar_window.sh`.

## S4 — the FUSED custom-mask forward (crown jewel, after S3 is green)
Add static max-width `mask_bias` + §7.6 positions buffers to `HIPAttnBackend`
(`prepare_verify_for_replay` recomputes the mask/positions for the current `c0` eagerly into the static
buffer); route `_spec_decode_step_tidar_fused` through the captured graph. Gate: fused CCA spec, graph
on, token-identical to eager fused + the ~4× ITL.
