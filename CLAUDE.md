# CLAUDE.md — instructions for all agents working in this repo

## GPU booking & profiling — follow the vllm-gfx1201 protocol (MANDATORY)

This box's two discrete gfx1201 cards are **shared across every repo on the machine**, not just
this one. Agents working in `~/code/vllm-gfx1201` (and its worktrees) are leasing the *same two
physical cards* you are. There is exactly **one** booking arbiter for the whole box, and it lives
in the vllm-gfx1201 checkout. So for anything GPU-related, **follow the protocol documented in
[`/home/pat/code/vllm-gfx1201/CLAUDE.md`](/home/pat/code/vllm-gfx1201/CLAUDE.md)** — specifically
its **"GPU sharing protocol"** and **"Trace analysis: TraceLens"** sections. The essentials:

### Booking a GPU
- The box has **two** gfx1201 compute cards: **GPU 0** (RX 9070 XT, 16 GB) and **GPU 1**
  (RX 9070, 16 GB). ROCm device **2** is the Ryzen iGPU — **never** a compute target.
- **EVERY GPU workload** — a bench, a raw `python`/torch probe, a container, anything that touches a
  card — **MUST be launched through the shared arbiter, by its absolute path:**
  ```
  /home/pat/code/vllm-gfx1201/scripts/gpu-lease.sh -n 1 -- <your command>
  ```
  Call it **by that absolute path** — do NOT copy it into this repo. Its flock lives at the
  hardcoded `/home/pat/code/vllm-gfx1201/.gpu-locks`, so only the original script coordinates with
  the other agents on the box. A local copy would book against a private lock and collide with
  everyone else.
- **`-n` = HOW MANY cards, NOT which card.** `-n 1` (the default) = one card; `-n 2` = both.
  Single-card model/probe/bench → `-n 1`. Only a genuine TP=2 job → `-n 2`. There is **no**
  "pin a specific card" flag and you never need one — the arbiter auto-assigns the lowest free card
  and injects `ROCR/HIP_VISIBLE_DEVICES` for you. Never pass `-n 2` thinking it selects "card #2":
  that leases BOTH cards and starves every other agent.
- **Let it block (the default).** Waiting *is* the coordination — do NOT poll `rocm-smi`, do NOT
  hand-set `HIP_VISIBLE_DEVICES`, and do NOT ask a human for a GPU window. There is no human in the
  loop. Use `--nowait` only to skip if the box is busy, `--timeout S` for a bounded wait.
- **`--detach` for any long-lived `up -d`/server** (binds the lease to container lifetime).
  Foreground jobs (`run --rm`, a script that blocks) need no flag — the lease frees when they exit,
  including on crash/Ctrl-C (flock auto-releases; no stale "reserved" state, no janitor).
- See who holds what:  `/home/pat/code/vllm-gfx1201/scripts/gpu-status.sh`
- CPU-only work (builds, static analysis, editing, trace post-processing) needs **no** lease.

### Profiling — run TraceLens after collecting traces
- After any profiling run that produces `*.pt.trace.json.gz`, **run TraceLens before drawing
  conclusions** — it adds Python→GPU linkage, per-kernel TFLOPS/TB·s roofline, and (TP≥2) straggler
  analysis that flat kernel bucketing cannot:
  ```
  /home/pat/code/vllm-gfx1201/profiling/run_tracelens.sh <trace_dir>
  ```
- This is **host-side post-processing — no GPU required. Do NOT wrap it in gpu-lease.sh.**
- One-time host install if missing: `uv tool install "git+https://github.com/AMD-AGI/TraceLens.git"`.

> Any vllm-gfx1201 checkout's `gpu-lease.sh` works (main or a worktree, e.g. the README's
> `…-gpu-lease/scripts/gpu-lease.sh`) — all anchor the same hardcoded `…/vllm-gfx1201/.gpu-locks`,
> so they coordinate the same two cards. Pick one; don't copy it.

## Container-run conventions for *this* repo (MANDATORY)

GPU runs here do **not** use this repo's `Dockerfile` (that's the inherited CUDA/NVIDIA upstream
artifact, kept for the future native image — it is NOT the run path). Instead we mount this repo's
source into the **shared ROCm image `vllm22-w4a8:combined`** (same image, toolchain, and W4A8
kernel as vllm-gfx1201) via hand-rolled `docker run`. The canonical recipe is in
[`README.md`](README.md) ("Running"); the conventions every run must keep:

- **ROCm device passthrough is mandatory or `is_rocm()` → False** (torch sees no HIP GPU, Triton
  disables, boot crashes). Always pass the full set:
  `--device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined
  --security-opt label=disable --cap-add SYS_PTRACE --ipc host --shm-size 16gb`.
- **Devices come from the lease, never by hand.** Inside the `gpu-lease.sh -- bash -c '…'` wrapper,
  pass `-e HIP_VISIBLE_DEVICES=$LEASE_ROCR_DEVICES -e ROCR_VISIBLE_DEVICES=$LEASE_ROCR_DEVICES`
  (the arbiter injects `LEASE_ROCR_DEVICES`). Do NOT hardcode `0`/`1`.
- **Mount the source, don't bake it (yet):** `-v "$PWD":/engine` and run with
  `PYTHONPATH=/engine/python python /engine/tools/…`. Activate the image venv first
  (`--entrypoint bash … -lc 'source /app/.venv/bin/activate && …'`) so Triton's JIT has PATH.
- **Reuse vllm-gfx1201's warm Triton cache** — same image/toolchain, so it hits:
  `-v /home/pat/code/vllm-gfx1201/.triton-cache-combined:/root/.triton`. Never pay the cold
  GDN/attention autotune. **Caveats carry over from vllm-gfx1201's CLAUDE.md §2–3:** for a throwaway
  or concurrently-compiling run, mount an **isolated root-owned copy** (don't corrupt the shared
  production cache); start a **fresh** dir only if the image's vLLM/Triton/ROCm/torch/arch changed
  or you suspect corruption.
- **HF cache:** `-v /home/pat/.cache/huggingface:/root/.cache/huggingface -e HF_HUB_OFFLINE=1`.
  Drop `HF_HUB_OFFLINE` only for a deliberate `snapshot_download` of an un-cached model.
- **No docker-compose here, so the compose-only conventions do NOT apply** — there is no
  `COMPOSE_PROJECT_NAME`, no per-card `8000+card` port injection, no compose Triton/HF env. Those
  are vllm-gfx1201 compose specifics. This repo's serve port is `1919` (`EXPOSE`); a hand-rolled
  serve maps it with an explicit `-p`.

When in doubt, read the canonical vllm-gfx1201 CLAUDE.md — it is the source of truth for the GPU,
profiling, and cache protocols; the sections above mirror the rules that bind GPU work done from
*this* repo and note where this repo's container path differs.
