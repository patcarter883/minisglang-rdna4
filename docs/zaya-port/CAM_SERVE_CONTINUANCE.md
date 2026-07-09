# CAM serving integration — CONTINUANCE (for a fresh session)

You are picking up the integration of memory-organ's **CAM** editable-memory (base-uncertainty WRITE GATE
+ product-key store + trained tap at layer 24 + learned per-token gate ROUTER) into the **minisgl-rdna4**
serve engine (Qwen3.5-4B). The core path is **built and PROVEN LIVE**; three finishing tasks remain. Read
this whole file before touching anything, then read `docs/zaya-port/CAM_SERVE_CONTRACT.md` (the interfaces).

## TL;DR of state
- **PROVEN LIVE** on the `minisgl-rdna4:lean` serving image: `/cam/remember` (write gate stores the
  base-unknowable) + `/cam/ask` (router-gated seed-once decode) delivered **3/3 fluent** through the real
  `CAMRuntime` + `cam_api` logic. e.g. `"The mother tongue of Oleg Kotov is"` → `"English. The first step
  is to find a good place to live"`.
- What is NOT done: (1) boot the **FastAPI HTTP server** with `MINISGL_CAM=1` and hit it over `curl`
  (only the endpoint *logic* was driven, not the running uvicorn server); (2) **backend model-share** — the
  co-located base is a 2nd ~8 GB model, tight on one 16 GB card; sharing the *served* model via the ZMQ
  scheduler is the production path; (3) **#100 sequential latent delivery** for genuine multi-token objects.

## Where everything lives
- **minisgl branch `cam-serve-integration`** (off `rdna4`). Files:
  - `python/minisgl/cam/memory.py` — `CAMMemory` (store/tap/router port; write gate, read, apply_tap,
    router_delta, forget=rebuild-from-survivors, list_facts/seed_token/delete aliases).
  - `python/minisgl/cam/runtime.py` — `CAMRuntime` + `get_cam_runtime()`: co-located frozen HF base +
    tokenizer + CAMMemory; `base_logits(token_ids)` under `autocast(bf16)`; native-GDN patch via
    `minisgl.gdn.hf_patch.patch_qwen3_5_gdn`. Enabled by env `MINISGL_CAM=1` + `MINISGL_CAM_CHECKPOINT=<dir>`.
  - `python/minisgl/server/cam_api.py` — `/cam/remember|ask|facts` (behind `MINISGL_CAM=1`); SPACE-PREFIX
    subject/object encoding (`_encode_sp`) — REQUIRED to match training (without it delivery was 1/3 garbage).
  - `python/minisgl/models/qwen3_5.py` — guarded data-plane hook after `tap_layer` (no-op unless a bank is
    staged) + `stage_cam`/`clear_cam`. Used only by the (future) residual-tap serving path; `/cam/ask` uses
    the LOGIT-only `router_delta`, so the hook is not on the current live path.
  - `python/minisgl/gdn/hf_patch.py` — `NativeGDNShim` now absorbs `use_cache`/`**kwargs` and bf16-aligns the
    fp32-upcast hidden (transformers>=5.13 compat). This also helps memory-organ on newer transformers.
  - `python/minisgl/cam/e2e_check.py` — the live driver (replicates cam_api /remember + /ask via CAMRuntime).
  - `python/minisgl/cam/roundtrip_check.py` — CPU loader<->export round-trip.
- **memory-organ worktree `/home/pat/code/memory-organ-softsteer`, branch `soft-steering`** (merged to main):
  - `cam/export_serving.py` (+ `--export-serving DIR` in `cam/recall_mag.py`) — produces the checkpoint.
  - `cam/recall_mag.py::eval_serve` / `--serve` — the offline reference loop (rank write gate + eviction).
  - `tools/serve.sh` — warm-up + `--export-serving`.
- **Real checkpoint (already produced):** `/home/pat/code/memory-organ-softsteer/cam_ckpt/`
  (`meta.json`, `tap.pt` 63 MB, `adapter.pt` 29 MB, `router.pt`). Regenerate via `tools/serve.sh` with
  `--export-serving /ckpt` and a writable `-v .../cam_ckpt:/ckpt` mount (NOT `/engine`, which is `:ro`).

## How to RUN (the substrate that has all kernels — use it, per the box owner)
Everything runs on the **`minisgl-rdna4:lean` image** (canonical kernels baked at `/opt/kernels`,
transformers 5.13, minisgl source mounted at `/engine`). This is the compose-service convention — don't
hand-mount a checkout for `.so` files. Book the GPU with `gpu-lease -n 1 -- ...` (see repo CLAUDE.md). The
proven live-e2e recipe (adapt for the HTTP server task):
```
gpu-lease -n 1 -- docker run --rm --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE --ipc host --shm-size 16gb \
  -e HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES -e ROCR_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES \
  -e HF_HUB_OFFLINE=1 -e PYTHONPATH=/opt/kernels:/engine/python:/engine \
  -e MINISGL_CAM_CHECKPOINT=/ckpt -e CAM_NATIVE_GDN=1 \
  -e CAM_POOLED_SUBJ_KEY=1 -e CAM_LEARNED_KEY_POOL=1 -e CAM_WRITE_AT_READ=1 \
  -v /home/pat/code/minisgl-rdna4:/engine:ro \
  -v /home/pat/code/memory-organ-softsteer/cam_ckpt:/ckpt:ro \
  -v /home/pat/.cache/huggingface:/root/.cache/huggingface \
  --entrypoint bash minisgl-rdna4:lean -lc 'python /engine/python/minisgl/cam/e2e_check.py'
```
Run this FIRST to confirm the live path still passes (3/3) before starting the remaining tasks.

> **GOTCHA (cost ~15 min, 2026-07-07):** the `-e HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES` above only
> works if `$HIP_VISIBLE_DEVICES` expands **inside the `gpu-lease` shell** (where the arbiter injects
> it), NOT your outer shell (where it's empty). If you paste the `gpu-lease -n 1 -- docker run …`
> line directly, your shell expands the vars to empty FIRST → the container sees no GPU → torch CPU
> fallback → `NotImplementedError: gdn_hip_C::causal_conv1d_fwd … 'CPU' backend` (a device error that
> masquerades as a kernel regression — it is NOT). Fix: put the `docker run` in a `bash <script>` (or
> `bash -c '…'`) run under the lease so the vars expand in the lease shell. See the working
> `scratchpad/run_e2e.sh` / `run_http.sh` pattern. (Matches the `gpu-lease-visible-devices-recipe`
> memory.)

### Task 1 — DONE (2026-07-07): standalone HTTP server + curl proven
`python/minisgl/cam/serve_app.py` (new) is a standalone uvicorn app mounting ONLY `cam_router`,
driven by `get_cam_runtime()`, warmed at boot (no ZMQ/backend). All four `/cam/*` endpoints pass
over real curl in the lean image: `/health`→`cam_loaded:true` (9s warm); `/cam/remember`→stored w/
base_p; `/cam/ask`→delivers "English."/"Dutch." 2/2; `/cam/facts`→lists decoded subject/object;
`DELETE /cam/facts/{subject}`→removes it. Also fixed a `/cam/facts` 500 in `cam_api.py`
(`list_facts()` returns token-ids; the endpoint now decodes them via the runtime tokenizer). Run it
with `scratchpad/run_http.sh` (or the recipe above, swapping the `-lc` for
`python /engine/python/minisgl/cam/serve_app.py` + a `-p 1919:1919`).

## REMAINING TASKS (priority order)

### 1. Boot the FastAPI HTTP server + curl the /cam/* endpoints
Goal: `MINISGL_CAM=1` + `MINISGL_CAM_CHECKPOINT=/ckpt` starts a server that answers real `curl` POSTs to
`/cam/remember` and `/cam/ask`. The endpoint *logic* is proven; this is the server shell.
- `python/minisgl/server/cam_api.py` mounts `cam_router` into `api_server.py` behind `MINISGL_CAM=1`, but
  its `_get_runtime()` expects `minisgl.cam.get_cam_runtime()` (built — CAMRuntime). The issue: `api_server`'s
  `lifespan` wires to the **backend scheduler over ZMQ** and expects the full serve stack. Two options:
  (a) EASIEST — a standalone `uvicorn` app that mounts ONLY `cam_router` (no backend/ZMQ), driven by
      `get_cam_runtime()`. Add e.g. `python/minisgl/cam/serve_app.py` (FastAPI() + include_router(cam_router)
      + `if __name__: uvicorn.run(...)`), run it in the lean image on the lease, `curl` from another shell
      **inside the same container/network**. This proves the HTTP `/cam/*` path with zero backend coupling.
  (b) Full `api_server` with `MINISGL_CAM=1` — needs a served model too (2 models; see task 2). Defer.
- Watch: TestClient needs `httpx`; uvicorn is present (serving image). Prefer real uvicorn + curl.

### 2. Backend model-share (resource-efficient, production path)
The co-located HF base (CAMRuntime) is a 2nd ~8 GB model — coexists with minisgl's backend model only on a
2-card box. The right design: CAM lives in the **backend scheduler process** (where the served Qwen3.5 model
already is), staged into the forward via the `qwen3_5.py` hook (already added), with `/cam/*` as a thin ZMQ
control-plane from the frontend. This is the multi-day piece the design doc describes. Scope it: (a) put the
CAMMemory + write-gate + router state in the backend; (b) `base_logits`/decode reuse the backend's forward
(no HF copy); (c) ZMQ messages for remember/ask. `docs/serving/integration_design.md` + `online_api.md` (in
memory-organ) are the design refs.

### 3. #100 sequential latent delivery (multi-token objects)
Tracked at github.com/patcarter883/memory-organ#100. The pooled-latent object shortcut is a DEAD END (a
mean-of-phrase latent decodes to a wrong token through the single-token readout — verified). Genuine
multi-token objects need per-step value emission via the `--readout {perpos|decoder}` path (already
prototyped in memory-organ) wired into the decode loop, with the router gating the delivery sequence.
Prereq: facts bound OUTSIDE the base-known filter (the write-policy work) so multi-token objects exist.

## Gotchas learned this session (do not rediscover)
- **Use the `minisgl-rdna4:lean` image** — kernels at `/opt/kernels`, transformers 5.13, source at `/engine`.
  `import minisgl` needs `PYTHONPATH=/opt/kernels:/engine/python:/engine`.
- **transformers 5.13 is dtype-fragile with HF Qwen3.5 in bf16** — the GDN mixer gets fp32 hidden; fixed via
  the shim's bf16-align + `base_logits` under `autocast(bf16)`. If you touch the base forward, keep autocast.
- **SPACE-PREFIX subjects AND objects** (`" "+text`) for the store — matches memory-organ `_sp_tokens`.
  Prompts stay plain. This is THE difference between 1/3 garbage and 3/3.
- **`/engine` mounts `:ro`** — write checkpoints/outputs to a separate `:rw` mount.
- **CAMMemory bakes behaviour knobs from `meta.json`** (pooled_subj_key, learned_key_pool, write_at_read,
  etc.), NOT os.environ — the exporter (`cam/export_serving.py`) must emit them (it does).
- **`/cam/ask` uses LOGIT-only router_delta (tap off)**, seed-once (stop injecting once the object's first
  token lands, then base fluency). The residual-tap hook in qwen3_5.py is for a future path, not this one.
- The offline memory-organ serve (`--serve`) is the quality reference: rank gate skips base-known cleanly,
  eviction (#17) rebuilds banks from survivors.

## First actions for the new session
1. `gpu-status`; run the RUN recipe above → confirm `e2e_check.py` still prints `CAM-SERVE E2E OK` 3/3.
2. Start Task 1 (standalone `cam_router` uvicorn app + curl) — smallest, highest-signal completion.
3. Report, then decide Task 2 (backend share) vs Task 3 (#100) with the box owner.
