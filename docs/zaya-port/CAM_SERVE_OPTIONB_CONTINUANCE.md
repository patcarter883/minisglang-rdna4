# CAM serving — Option B (first-class residual-tap integration) CONTINUANCE

You are building CAM into the minisgl backend scheduler as a **first-class residual tap** (the box
owner chose Option B over the smaller logit-bridge). Read, in order: this file, then
`CAM_SERVE_TASK2_SCOPING.md` (the A-vs-B analysis + exact seams), then the design specs in
memory-organ: `docs/serving/integration_design.md` (the data-plane, authoritative) and
`online_api.md` (the edit-plane REST surface). Task tracking: 5 tasks (#1 done, #2 in_progress).

## Where the work lives
- **Worktree `/home/pat/code/minisgl-rdna4-camB`, branch `cam-serve-optionB`** (off `cam-serve-integration`
  HEAD 13c0707). Mount THIS worktree into GPU runs, never the shared tree (agents edit it concurrently).
- The lean image bakes kernels at `/opt/kernels`, so the worktree does NOT need the vendored `.so`.

## DONE this session (all GPU-validated)
- **Task 1 (HTTP server)** — `python/minisgl/cam/serve_app.py` (standalone uvicorn, mounts only
  `cam_router`, warms `get_cam_runtime()` at boot). All four `/cam/*` endpoints pass over curl; fixed a
  `/cam/facts` 500 (`cam_api.py::list_facts` now decodes token-ids via the runtime tokenizer). Driver:
  `scratchpad/run_http.sh` (in the session scratchpad).
- **Task-1.5 de-risk — the TAP delivers 3/3 (GREEN).** `python/minisgl/cam/tap_check.py` hooks
  `CAMMemory.apply_tap` into the HF base's decoder layer[24] and greedily decodes with NO router. Result:
  tap delivers Dutch/English/Russian; base-OFF baseline gets them wrong. **CRITICAL FINDING:** an
  always-on tap degenerates into repetition ("English English English…"); it needs the SAME **seed-once**
  discipline as the router — clear the bank once the object's first token lands. Seed-once tap output is
  byte-identical quality to the router path. The served model already has `clear_cam()` for this.
  (Memory: `cam-tap-needs-seed-once`.)

## Non-negotiable gotchas (do not rediscover)
- **GPU var-expansion trap:** run `docker run` INSIDE a `bash <script>` under `gpu-lease`, so
  `$HIP_VISIBLE_DEVICES`/`$ROCR_VISIBLE_DEVICES` expand in the LEASE shell. Pasting the bare recipe →
  empty vars → CPU fallback → `gdn_hip_C::causal_conv1d_fwd … 'CPU' backend` (looks like a kernel bug;
  it is NOT). See `scratchpad/run_tap.sh` / `run_http.sh`.
- Lean image; `PYTHONPATH=/opt/kernels:/engine/python:/engine`; `/engine` is `:ro` (write to a `:rw` mount).
- Space-prefix subjects AND objects (`_encode_sp`) for the store. CAMMemory bakes knobs from `meta.json`.
- **Tap + seed-once, not always-on** (finding above).

## Exact seams (verified file:line, worktree paths)
- `engine/engine.py:82` `Engine.__init__`; model built `:132`, `post_load` `:134`; GDN-state block
  `:157-191` (the pattern to mirror for a CAM-state build). Embed handle `self.model.model.embed_tokens.weight`
  (used at `:596`); lm_head `self.model.lm_head.weight`. Forward dispatch `Engine.forward_batch:513-552`,
  eager `model.forward()` at `:530`, graph replay `:528`.
- `core.py:42` `class Req` (add `mem_bank`/`mem_conf` fields, default None, like the gdn/cca pattern);
  `core.py:126` `class Context`, `gdn_state`/`cca_state` fields at `:135`/`:138` (add `cam_state` beside).
- `models/qwen3_5.py:306-316` `stage_cam`/`clear_cam` (already present); tap inject `:337-341`; per-forward
  fields `:298-301`. **The tap hook is already wired — it just needs staging + a built CAMMemory.**
- `scheduler/scheduler.py:107` tokenizer already loaded here; `_process_one_msg:317-346` (add CAM control
  msg branches); prefill add `add_one_req` (~:337); forward call site `_forward:486-495` (stage bank
  before, seed-once clear after).
- ZMQ: new `BaseBackendMsg`/`BaseFrontendMsg` dataclasses auto-register via `globals()`
  (`message/backend.py`, `message/frontend.py`), BUT the tokenizer worker hard-filters to 3 msg types and
  asserts (`tokenizer/server.py:81-84`) — widen passthrough for CAM control msgs.

## Plan (design-doc §7 phasing; task #s)
- **Phase 0 (#2) — plumbing, no behavior change when off — CODE WRITTEN, boot-validation pending:**
  1. ✅ `core.py`: `Req.mem_bank/mem_conf` set in `__post_init__` (default None); `Context.cam_state` field.
  2. ✅ `engine/engine.py::__init__` (after the CCA block): guarded build — if `MINISGL_CAM=1` +
     `MINISGL_CAM_CHECKPOINT` is a dir + the model has `stage_cam`, build
     `CAMMemory(ckpt, inner.embed_tokens, self.model.lm_head.weight)` (reuse SERVED weights, NO HF copy),
     stash on `self.cam` + `ctx.cam_state`, `inner.stage_cam(cam, None, None)` to register the tap layer.
     Off/failure → None → tap byte-exact no-op (try/except so CAM can NEVER break serving). All 5 files
     `py_compile`-clean.
  3. **PENDING (do first next session):** boot the full scheduler with `MINISGL_CAM=1
     MINISGL_CAM_CHECKPOINT=/ckpt` and confirm (a) the `CAM: backend memory built…` log fires, (b) a
     normal `/generate` with NO memory field returns coherent text (serving unperturbed). Watch: booting
     Qwen3.5-4B (GDN-hybrid) may autotune — mount the warm Triton cache; native gdn_hip should avoid the
     ~22-min grind. The build is low-risk (same weights CAMMemory already consumes from HF in runtime.py);
     the REAL unknown is Phase 1 parity (below).
- **Phase 1 (#3) — eager MVP `--graph 0`, single request. THE KEY UNKNOWN: does minisgl's served
  forward deliver through the HF-trained tap?** (tap_check validated the tap against the HF base; Phase 1
  is the first test through minisgl's own kernels.) explicit-subject `memory` field on the
  request; at prefill compute `(bank,conf)=cam.read(subject_ids)` → `req.mem_bank/mem_conf`; scheduler
  `stage_cam(cam, bank, conf)` before `_forward`; **seed-once: `clear_cam()` when the emitted token ==
  `cam.seed_token(bank,conf)`**; `/cam/*` (or `/v1/memory/*`) as ZMQ control msgs; `/cam/remember` write
  gate still needs a base-logits read (a synthetic no-tap forward, or reuse the prompt prefill logits).
  Acceptance: reproduce the tap_check 3/3 delivery THROUGH the running scheduler over HTTP.
- **Phase 2 (#4) — graph capture:** `CAMGraphCapture` static bank/conf buffers (mirror
  `gdn/graph_capture.py`), `--graph N`, byte-identical eager-vs-graph, ~20% TPOT.
- **Phase 3 (#5) — concurrency/per-row banks/full `/v1/memory/*`/TP** (online_api.md §3 COW swap).

## First action next session
`gpu-status`; re-run `scratchpad/run_tap.sh` to reconfirm tap 3/3; then implement Phase 0 (items 1–3
above) and boot the full scheduler (`--attn hip --graph 0 --model Qwen/Qwen3.5-4B` + `MINISGL_CAM=1`
`MINISGL_CAM_CHECKPOINT=/ckpt`) to confirm memory-off serving is unperturbed, then wire Phase 1.
</content>
</invoke>
