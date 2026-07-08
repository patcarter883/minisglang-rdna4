# CAM serving — model-share: architecture, status, and the remaining gap

## TL;DR
Engine-level model-share is **already built and committed on `rdna4`** — the served minisgl engine builds
`engine.cam` (a `CAMMemory`, i.e. the cosine-NN pointer store) from its OWN weights, so there is no second
base copy. The #100 pointer delivery therefore works through it with the existing `cam_api` in the
**single-process** deployment. The remaining gap is **multi-process** pointer delivery (frontend ↔ backend
engine.cam). The earlier "FrontendManager bridge + embed-only loader" idea (commit e5680cd, now reverted)
was redundant with `engine.cam` and is dropped.

## What already exists (committed on rdna4)
- `engine.py`: when `MINISGL_CAM=1` + `MINISGL_CAM_CHECKPOINT` set, the engine builds
  `self.cam = CAMMemory(ckpt, inner.embed_tokens, lm_head_weight)` **from the served model's own weights**
  (`engine.cam`) — no co-located HF copy.
- `llm/llm.py`: `LLM.base_logits(token_ids)` (served-model logits) + `LLM.generate(...)`.
- `scheduler.py::_prepare_cam` / `_stage_cam`: residual-tap delivery keyed on
  `sampling_params.mem_subject` (space-prefixed subject → `engine.cam.read` → tap at prefill, seed-once).
- `runtime.py::BackendCAMRuntime` (`MINISGL_CAM_BACKEND=1`): exposes `engine.cam` as `.memory`,
  `LLM.base_logits` as `.base_logits`, `.tokenizer`, `.ask_tap` (mem_subject tap delivery).
- `cam_api.py`: uses ONLY `runtime.{tokenizer, memory, base_logits}` + `memory.deliver_object_ids`, so it
  is `BackendCAMRuntime`-compatible **as-is**.

## Two deployment topologies
1. **Single-process (CAM-dedicated serve) — model-share DONE, needs validation.**
   `serve_app.py` (or a thin `LLM` driver) with `MINISGL_CAM=1 MINISGL_CAM_BACKEND=1`
   `MINISGL_CAM_CHECKPOINT=/ckpt`. `BackendCAMRuntime` holds the ONE served `LLM`; `engine.cam` is the
   cosine-NN store. `/cam/ask` = `deliver_object_ids` (pointer, exact ids) + `llm.base_logits`
   continuation. **One model copy.** No new code — validation only.

2. **Multi-process (full `api_server`: `/generate` + `/v1/*` + `/cam/*`) — the real GAP.**
   The backend scheduler process holds `engine.cam`; the FastAPI frontend runs `cam_api`. The
   **residual-tap** path already crosses the boundary via `sampling_params.mem_subject` (a normal generate
   request; the scheduler reads `engine.cam` and taps in the backend). But the #100 **pointer**
   (`deliver_object_ids`) is a frontend-side lookup that needs the backend's `engine.cam` — the frontend's
   `get_cam_runtime()` today would load a SECOND model (BackendCAMRuntime's own `LLM`, or the standalone
   HF base). So multi-process pointer delivery is not yet reachable without a second copy.

   **Fix (the actual next implementation):** a small **control-plane message** frontend→backend —
   `CamDeliverMsg(subject_ids) -> object_ids` — answered by the scheduler calling
   `engine.cam.deliver_object_ids(subject_ids)`; then `/cam/ask` emits those ids and lets the base
   continue via a normal `/generate`. This reuses `engine.cam` (no second copy) and mirrors how
   `mem_subject` already rides the request path.

## Validation plan
- **Single-process (now):** run `BackendCAMRuntime` (one `LLM`) with `MINISGL_CAM=1 MINISGL_CAM_BACKEND=1`;
  remember Klingon/Sindarin, `/cam/ask` → assert pointer delivery == the standalone path, and assert the
  process holds ONE model (no co-located HF base). Needs a free GPU.
- **Multi-process (after the control-plane message):** full `api_server`; assert `/generate` + `/cam/ask`
  share one backend model, and pointer delivery matches.

## Superseded / reverted
- The FrontendManager text-gen bridge + `load_input_embed` (embed-only) — `engine.cam` already uses the
  served model's own embed + logits, so both are unnecessary. Reverted from this branch.
