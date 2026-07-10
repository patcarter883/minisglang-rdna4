# CAM serving — Task 2 (backend model-share) SCOPING

Task 2 = stop loading a co-located ~8 GB HF base in the FastAPI frontend; instead reuse the
**already-served** minisgl Qwen3.5-4B in the backend scheduler process, with `/cam/*` becoming a
thin ZMQ control-plane. This doc scopes it against the real seams (mapped 2026-07-07) so the box
owner can pick the scope before any multi-day build.

## Ground truth (verified file:line)
- **Backend scheduler is a single long-lived model process.** `Engine.forward_batch` → `model.forward()`
  (`engine/engine.py:513-552`, forward at `:530`); one `Scheduler`/`Engine` per process for the run.
- **Everything CAMMemory needs is already in the scheduler:**
  - tokenizer: `self.tokenizer = load_tokenizer(...)` (`scheduler/scheduler.py:107`).
  - input embed: `model.model.embed_tokens.weight` (`qwen3_5.py:278`); lm_head: `model.lm_head.weight`
    (`qwen3_5.py:452`). These are exactly `CAMMemory.__init__(ckpt, base_embed, lm_head_weight)`
    (`cam/memory.py:383`).
- **The residual tap is ALREADY in the served model, unused.** `stage_cam`/`clear_cam` (`qwen3_5.py:306-316`)
  + inject after `tap_layer` (`qwen3_5.py:337-341`, byte-exact no-op when no bank staged). Nothing calls
  `stage_cam`; the call site would be `scheduler.py:486-489` (between batch schedule and `_forward`).
- **ZMQ path:** frontend → tokenizer worker → scheduler; replies scheduler → detokenizer → frontend.
  New `BaseBackendMsg`/`BaseFrontendMsg` dataclasses auto-register via `globals()` (no registry edit),
  but the **tokenizer worker hard-filters to 3 msg types and asserts** (`tokenizer/server.py:81-84`) —
  passthrough must be widened for CAM control msgs. Handler seam: `Scheduler._process_one_msg`
  (`scheduler.py:317-346`).
- **The existing `/cam/*` frontend plane never touches the backend** — it loads its OWN HF base
  (`cam/runtime.py:46-87`); that IS the 8 GB duplicate this task removes.

## The mechanism fork (this is what makes A vs B different)
The checkpoint ships BOTH a residual **tap** and a logit **router**. Only the **router (logit) path**
has delivered live (3/3, Task 1). They integrate with the backend very differently:
- **Router/logit path (PROVEN):** needs the full last-position logit vector pulled BEFORE sampling, to
  add `router_delta` then argmax + seed-once. The scheduler's normal forward returns *sampled tokens*,
  not raw logits → needs a new **synthetic read-only "base_logits(token_ids)→[vocab]" forward mode**.
- **Tap/residual path (UNVALIDATED in-serve):** composes with the *existing* generate pipeline — stage
  bank → normal decode → tap injects at L24 → object falls out of argmax naturally. No raw-logit pull
  for `/cam/ask`. BUT `/cam/remember`'s base-uncertainty write gate STILL needs `p_base(object)`, a
  raw-logit read — so the synthetic base_logits mode is required regardless of path.

## Options
### Option A — "logit bridge" (keep the proven mechanism; smallest faithful model-share)
Keep `CAMMemory` + `router_delta` + seed-once EXACTLY as proven. Move CAMMemory into the scheduler
(embed/lm_head handles are there). Add ONE backend capability: `cam_base_logits(token_ids)→[vocab]`
(a synthetic read-only prefill forward that returns last-position logits). `/cam/remember|ask|facts|
delete` become ZMQ control msgs handled in `_process_one_msg`; the frontend `cam_router` repoints its
`_get_runtime()` seam from the co-located HF model to a ZMQ round-trip. **8 GB duplicate gone.**
- **Cost:** ~2–4 days. The one non-trivial piece is the synthetic base_logits forward (build a one-off
  prefill `Batch`, run forward, read logits before sampling, tear down KV) — real but bounded engine work.
- **Risk (gating, cheap to test FIRST):** the served minisgl Qwen3.5 forward must produce logits close
  enough to HF Qwen3.5 (which the router was trained against) that delivery survives. **De-risk with a
  ~1 hr probe** (compare minisgl vs HF last-logits + router delivery on the 3 probe prompts) BEFORE
  committing to the build.
- **Perf note:** `/cam/ask` recomputes the full prefix each step (no KV reuse) → N synthetic forwards
  of growing length per answer. Fine for short edits; not a hot path. KV-reuse is a later optimization.

### Option B — first-class residual-tap integration (the `integration_design.md` design; weeks)
Port `GatedMemoryTap` as a `BaseOP`, `CAMState` engine object, `ctx.cam_state`, per-request `memory`
field, `/v1/memory/*` REST surface (per `online_api.md`), and `CAMGraphCapture` static bank buffers for
the ~20% TPOT graph-capture win. 4 phases (Phase 0 plumbing → 1 eager MVP → 2 graph capture → 3
concurrency/TP). This is the production end-state AND the only path that unlocks the capture TPOT win.
- **Cost:** multi-week.
- **Extra risk over A:** reopens the residual-tap mechanism, whose in-serve delivery is UNVALIDATED
  (router, not tap, is what delivered 3/3). Must re-validate tap delivery in-engine first.

### Option A→B (recommended sequencing)
Ship **A** first (proven mechanism, kills the duplicate, days), gated by the cheap parity probe. Then,
if the capture-TPOT win and the full `/v1/memory/*` surface are wanted, do **B** as a separate track on
top — with A's model-share already in place, B's Phase 0/1 reuse the same in-scheduler CAMMemory.

## Recommendation
**Option A, de-risked by the parity probe first.** It is faithful to what already works, directly
achieves the stated Task-2 goal (no 8 GB duplicate), and is days not weeks. B is the eventual
production end-state but should follow A, not replace it, and only after tap-delivery is re-validated
in-engine.

## First concrete step either way
A ~1 hr GPU probe: stand up the served minisgl Qwen3.5-4B, extract last-position logits for the 3 probe
prompts, and compare to the HF base + check `router_delta` still delivers. This answers the one
question that gates the whole task (is the served model a drop-in logit source for the trained router)
and is worth doing before writing integration code.
</content>
</invoke>
