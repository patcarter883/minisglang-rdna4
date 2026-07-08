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

2. **Multi-process (full `api_server`: `/generate` + `/v1/*` + `/cam/*`) — IMPLEMENTED (ready-to-validate).**
   The backend scheduler holds `engine.cam`; the FastAPI frontend runs `cam_api`. Rather than a new
   control-plane message, both #100 CAM ops **ride the existing generate request** through
   `sampling_params` (the same seam `mem_subject` already uses for the tap), so there is **no new message
   type and no second model copy**:

   - **Deliver** (`/cam/ask`): a generate with `mem_subject=subject`. `scheduler._prepare_cam` calls
     `engine.cam.deliver_object_ids(subject_ids)`; `_process_last_data` **forces** those exact object tokens
     as the first N emitted tokens (committed to the KV history), then the served base continues — exactly
     `_gen_ptr`. Falls back to the residual tap when the subject addresses no stored object.
   - **Remember** (`/cam/remember`): a `max_tokens=1` generate with `mem_subject=subject` +
     `mem_remember=object_token_ids`. `_prepare_cam` writes `subject->object` into `engine.cam` at prefill
     (write-only stub generation). The base-uncertainty gate is skipped for now (storing a base-known fact
     is harmless — the pointer delivers the same object the base would).
   - **Frontend:** `FrontendCAMRuntime` (`MINISGL_CAM_FRONTEND=1`) holds only a tokenizer (no model, no local
     store) and sends these generates via the FrontendManager primitive; `cam_api` routes remember/ask to
     it (`is_frontend_share`).

   Changed: `core.py` (`SamplingParams.mem_remember`), `scheduler.py` (`_prepare_cam` write/deliver +
   `_process_last_data` forced tokens), `runtime.py` (`FrontendCAMRuntime`), `cam_api.py` (frontend branches).

   **Still to do:** `/cam/facts` `/forget` `/stats` return data that does not fit a generate, so they need a
   small control-plane message (follow-up). And full-serve validation (below).

## Validation plan
- **Single-process (now):** run `BackendCAMRuntime` (one `LLM`) with `MINISGL_CAM=1 MINISGL_CAM_BACKEND=1`;
  remember Klingon/Sindarin, `/cam/ask` → assert pointer delivery == the standalone path, and assert the
  process holds ONE model (no co-located HF base). Needs a free GPU.
- **Multi-process (after the control-plane message):** full `api_server`; assert `/generate` + `/cam/ask`
  share one backend model, and pointer delivery matches.

## Superseded / reverted
- `load_input_embed` (embed-only loader) — `engine.cam` already builds its store from the served model's
  own embed + logits, so a separate embed load is unnecessary. Reverted.
- The original "generic FrontendManager text-gen bridge + greedy-token gate" is subsumed: the frontend
  still rides the FrontendManager primitive (`FrontendCAMRuntime`), but the CAM logic (deliver/write)
  lives in the backend `engine.cam` via `mem_subject`/`mem_remember`, not a generic bridge. No new message
  type, no embed copy.
