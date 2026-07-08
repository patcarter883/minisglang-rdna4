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

   - **Data ops** (`/cam/facts`, `/forget`, `/stats`): a `mem_op` sampling param ("facts"|"forget"|"stats").
     `_prepare_cam` computes the result from `engine.cam`, tokenises the JSON, and force-emits it as the
     reply text + EOS (reusing the same forced-token path) — so even the data-returning ops need NO new
     message type or reply channel. `FrontendCAMRuntime.facts/forget/stats` send the op and JSON-parse the
     reply.

   Changed: `core.py` (`SamplingParams.mem_remember`, `mem_op`), `scheduler.py` (`_prepare_cam`
   write/deliver/control + `_cam_ctrl_result` + `_process_last_data` forced tokens), `runtime.py`
   (`FrontendCAMRuntime` remember/ask/facts/forget/stats), `cam_api.py` (frontend branches on all endpoints).

   **Still to do:** full-serve validation (below). Note: CAM state is per-scheduler-replica, so a multi-
   replica (DP) deployment needs CAM requests pinned to one replica (or a replicated store) for a
   consistent view — single-replica (dp_size=1, the eager CAM contract) is unaffected.

## Validation — DONE (multi-process, gfx1201)
`MINISGL_CAM=1 MINISGL_CAM_CHECKPOINT=/ckpt MINISGL_CAM_FRONTEND=1 python -m minisgl --model Qwen/Qwen3.5-4B
--attention-backend hip` (full api_server: backend scheduler + tokenizer/detokenizer + frontend). curl:
- **VRAM: 14.3 GB on ONE card, GPU 1 free** — a single model copy (no co-located base). Model-share confirmed.
- `remember` Klingon/Sindarin (mem_remember write) → `{"stored":true}` (writes to the backend engine.cam).
- `ask` (mem_subject forced tokens) → `"Klingon and she is a member of the Quillsworth family…"` — the
  exact object delivered from memory + coherent base continuation.
- `ask` PARAPHRASE (reordered subject "Quillsworth Zephyrina") → `"Klingon…"` — paraphrase delivery works.
- `facts` (mem_op) → `[{Zephyrina:Klingon},{Cornelius:Sindarin}]`; `stats` (mem_op) →
  `{"B":32,"total_edits":2,…}`; `forget` Cornelius (mem_op) → `{"deleted":true}`; `facts` after → only
  Zephyrina remains. All data ops correctly reflect/mutate the backend engine.cam.

Every op crosses the frontend↔backend boundary via the sampling-param ride — no new message type, no
second model copy.

## Transparent CAM — ambient reads + writes on `/v1/chat` (VALIDATED)
CAM becomes ambient on the NORMAL endpoints (no `/cam/*`, no `mem_subject`), gated by env flags:
- **Read** (`MINISGL_CAM_AUTO=1`): `api_server._cam_auto_augment` runs before generation — the backend
  `retrieve` op (`mem_op="retrieve"`) decodes the prompt, pulls capitalised proper-noun spans, cosine-
  matches each against the store (tau-gated), and matched facts are prepended as a system note. The base
  then answers using the in-context fact (it can *copy* the novel object — works for any query shape).
- **Write** (`MINISGL_CAM_AUTO_WRITE=1`): `_cam_auto_write` extracts durable `(subject,object)` facts from
  the latest user turn (`FrontendCAMRuntime.extract_facts`: LLM extraction, with a capitalised-"X is Y"
  regex fallback for small thinking models that don't emit clean JSON) and `remember`s them.

Both additive, off by default; each costs one extra generation per turn (retrieve / extract).
**Validated end-to-end** (Qwen3.5-4B, one card): baseline "what is X's language?" → model doesn't know;
after a plain chat that STATES "the mother tongue of Bartholomew Fizzwick is Dothraki", `/cam/facts` shows
the auto-learned fact and a later plain chat "what language does he speak?" answers **Dothraki** — with
no CAM params on either turn. Same for an explicitly-remembered fact (Klingon).

Follow-ups: proper thinking-disable for cleaner LLM extraction; the base-uncertainty write gate; a
fact-statement heuristic to skip the extraction generation on chit-chat turns (latency).

## Superseded / reverted
- `load_input_embed` (embed-only loader) — `engine.cam` already builds its store from the served model's
  own embed + logits, so a separate embed load is unnecessary. Reverted.
- The original "generic FrontendManager text-gen bridge + greedy-token gate" is subsumed: the frontend
  still rides the FrontendManager primitive (`FrontendCAMRuntime`), but the CAM logic (deliver/write)
  lives in the backend `engine.cam` via `mem_subject`/`mem_remember`, not a generic bridge. No new message
  type, no embed copy.
