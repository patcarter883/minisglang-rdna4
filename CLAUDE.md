# CLAUDE.md — instructions for all agents working in this repo

## GPU booking & profiling — follow the vllm-gfx1201 protocol (MANDATORY)

This box's two discrete gfx1201 cards are **shared across every repo on the machine**, not just
this one. Agents working in `~/code/vllm-gfx1201` (and its worktrees) are leasing the *same two
physical cards* you are. There is exactly **one** booking arbiter for the whole box. It now lives in
its **own canonical repo, [`/home/pat/code/gpu-lease`](/home/pat/code/gpu-lease)** (bare `gpu-lease` /
`gpu-status` on `$PATH` via `lease install`; unifies local **and** cloud leasing + a monitoring
console — its `README.md` is authoritative for leasing). For the profiling (TraceLens), container-run,
and cache conventions, [`/home/pat/code/vllm-gfx1201/CLAUDE.md`](/home/pat/code/vllm-gfx1201/CLAUDE.md)
remains the source. The essentials:

### Booking a GPU
- The box has **two** gfx1201 compute cards: **GPU 0** (RX 9070 XT, 16 GB) and **GPU 1**
  (RX 9070, 16 GB). ROCm device **2** is the Ryzen iGPU — **never** a compute target.
- **EVERY GPU workload** — a bench, a raw `python`/torch probe, a container, anything that touches a
  card — **MUST be launched through the shared arbiter, now a bare command on `$PATH`:**
  ```
  gpu-lease -n 1 -- <your command>
  ```
  The lease tool moved to its **own canonical repo `/home/pat/code/gpu-lease`** and is installed on
  `$PATH` via `lease install`, so call it **bare as `gpu-lease`** — the old
  `/home/pat/code/vllm-gfx1201/scripts/gpu-lease.sh` path is **gone**. If `gpu-lease` isn't found in
  your shell, run `lease install` (or open a fresh shell) first; do **not** copy the tool into this
  repo. Its flock still lives at the hardcoded `/home/pat/code/vllm-gfx1201/.gpu-locks`, so every
  agent in every worktree coordinates on the same locks — a local copy would book against a private
  lock and collide with everyone else. Flags, semantics, and env-injection are unchanged from the old
  script.
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
- See who holds what:  `gpu-status` (bare command, same repo/PATH as `gpu-lease`)
- CPU-only work (builds, static analysis, editing, trace post-processing) needs **no** lease.

### Profiling — run TraceLens after collecting traces
- After any profiling run that produces `*.pt.trace.json.gz`, **run TraceLens before drawing
  conclusions** — it adds Python→GPU linkage, per-kernel TFLOPS/TB·s roofline, and (TP≥2) straggler
  analysis that flat kernel bucketing cannot:
  ```
  /home/pat/code/vllm-gfx1201/profiling/run_tracelens.sh <trace_dir>
  ```
- This is **host-side post-processing — no GPU required. Do NOT wrap it in `gpu-lease`.**
- One-time host install if missing: `uv tool install "git+https://github.com/AMD-AGI/TraceLens.git"`.

> The lease interface is now the bare `gpu-lease` / `gpu-status` commands from the canonical
> **`/home/pat/code/gpu-lease`** repo (installed on `$PATH` via `lease install`; it unifies local
> **and** cloud leasing plus a monitoring console — its `README.md` is authoritative). Every
> invocation anchors the same hardcoded `…/vllm-gfx1201/.gpu-locks`, so all agents/worktrees
> coordinate the same two cards. Never use a `scripts/…gpu-lease.sh` path — it no longer exists.

## Source isolation — per-task git worktree, NEVER mount the shared `$PWD` (MANDATORY)

The working tree at `/home/pat/code/minisgl-rdna4` is **shared mutable state with no lock**, exactly
like the two GPUs. Multiple agents edit it concurrently. A container started with `-v "$PWD":/engine`
reads whatever (possibly torn, mid-edit) state is on disk **at boot AND lazily across the whole run**
(Python imports, AOT `.so` rebuilds, Triton JIT all happen long after container start). The failure
modes are both a loud crash (a half-written `moe.py` fails to import) and — far worse — a silent one:
the run "validates" garbage and you commit/report plausible-but-meaningless numbers. **The worktree
is the source-side equivalent of the GPU lease: it is how you take an exclusive, consistent snapshot
of the code for the duration of a job.**

So for ANY container/GPU run, or any multi-step task that edits code while something reads it:

1. **Make an isolated worktree first** (off the commit you intend to build on):
   ```
   git worktree add -b <task-branch> /home/pat/code/minisgl-rdna4-<task> <base-commit>
   ```
   Bring in ONLY your hunks (when the shared tree mixes your edits with another agent's, reconstruct
   your changeset there — `git diff <base> -- <your-files>` filtered to your hunks, applied in the
   worktree — rather than `git commit -A`'ing the mixture).
2. **Mount the WORKTREE, never `$PWD`:** every `docker run -v <worktree>:/engine`, every
   `PYTHONPATH=<worktree>/python`, every AOT build path points at the worktree. Validation harnesses
   in `tools/` must set `REPO=<worktree>` (they default-mounted `$PWD` historically — fix that).
3. **The isolation window spans the entire job**, not just boot — keep editing in the worktree (or
   not at all) until the run is fully done, because the container re-reads source lazily.
4. **Clean up when done:** `git worktree remove <path>` after you've committed/merged.

This rule got skipped once (an EP validation read a tree another agent was mid-edit on) **because the
GPU-lease rule was written down and this one was not.** Both are now codified; treat them as equally
mandatory.

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
- **Devices come from the lease, never by hand.** Inside the `gpu-lease -- bash -c '…'` wrapper,
  **forward the arbiter's already-composed pair**:
  `-e HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES -e ROCR_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES`.
  Do NOT hardcode `0`/`1`, and do NOT set BOTH to `$LEASE_ROCR_DEVICES` — that is the *physical*
  card index, so it double-filters and breaks whenever the lease assigns **card 1**:
  `ROCR_VISIBLE_DEVICES=1` selects physical card 1 and re-indexes it to 0, then
  `HIP_VISIBLE_DEVICES=1` selects nothing → torch `RuntimeError: No HIP GPUs are available` (it only
  ever worked on card 0). The arbiter exports the correct composition in the lease shell already
  (`ROCR_VISIBLE_DEVICES=<physical>`, `HIP_VISIBLE_DEVICES=0`) — pass those through verbatim.
  (Diagnosed in Phase 3e, 2026-06-22; the canonical vllm-gfx1201 doc is unaffected — its compose path
  auto-injects the pair.)
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

## Kernel core policy — ONE core per shape; weight/act formats are WLoad POLICIES (MANDATORY)

Before writing or "optimizing" ANY GEMM / GEMV / MoE kernel in `rdna4-hip-kernels`, read
[`/home/pat/code/rdna4-hip-kernels/KERNEL_CORE_POLICY.md`](/home/pat/code/rdna4-hip-kernels/KERNEL_CORE_POLICY.md).
The rule: **a new weight format (int4/e2m1/fp8/NL-codebook/W8A16/bf16) or activation dtype is a loader
policy on the EXISTING shared core — never a new kernel, never a new package.** Two kernels that compute
the same shape and differ only in weight-unpack/dequant/scale MUST be one `template<class WLoad, …>`
body. Copy-pasting a `*_gemv`/`*_gemm`/`*_moe` kernel and editing the decode lines is the exact debt this
forbids — occupancy/coalescing/tiling wins live in the core, so a copy silently strands every future win
(a real, measured example: the fp8 decode GEMV reached 81% of HBM while the copy-pasted int4 GEMV sat at
38% on the same card — the served path's dominant bandwidth, lost to a fork). Only a genuinely different
*algorithm or tiling* justifies a new kernel. The doc carries the live consolidation backlog (decode
GEMV ×5, `rxf`, `moe_bf16`/`moe_w8a16`, `dense_gemm` dead variants) — extend the shared core, do not fork.
