# ZAYA / DP+EP — continuance prompt (paste into a new session)

You are continuing work on the **ZAYA1-8B-fp8** bring-up in minisglang (`/home/pat/code/minisgl-rdna4`,
branch `rdna4`) on a shared 2× gfx1201 box. Read the project memory (auto-injected `MEMORY.md`) and the
docs under `docs/zaya-port/` before acting. Most relevant docs:
`ZAYA_REFERENCE.md, PORT_PLAN.md, W8A8_KERNEL_SPEC.md, SERVING_MATRIX.md, DP_EP_SPEC.md, RSA_SHIM.md`.

## ⛔ RULES THAT MUST BE FOLLOWED (learned the hard way this session)
1. **Source isolation — work in your own git worktree; NEVER mount the shared `$PWD` into a container.**
   The working tree is shared mutable state with no lock; another agent edits it concurrently. A
   container started with `-v "$PWD":/engine` reads whatever (possibly torn, mid-edit) state is on disk
   at boot AND lazily across the run → crashes, or worse, silently "validates" garbage. The worktree is
   the source-side equivalent of the GPU lease. FIRST ACTION below sets this up.
2. **GPU work goes through the arbiter:** `/home/pat/code/vllm-gfx1201/scripts/gpu-lease.sh` (absolute
   path). Container recipe per `CLAUDE.md` (image `vllm22-w4a8:combined`, full device passthrough,
   forward the lease HIP/ROCR pair, `PYTHONPATH=/engine/python:/engine`, mount `/home/pat/models` ro,
   warm triton cache). `pip install msgpack` before any minisgl import (missing from the image venv).
   (The MoE decode scatter is unconditional and graph-capturable now — no flag.)
3. **GPU validation FAILS FAST + is MONITORED.** ONE bounded attempt + short `timeout`, never a blind
   retry loop (a crash-looping serve held the shared 2-card lease ~30 min). During any long GPU job,
   periodically `docker ps` + `gpu-status.sh`; verify a container is yours before `docker stop` (use
   `docker stop`, `docker rm -f` is sandbox-denied). A `--rm` container shows `RestartCount=0` even when
   crash-looping — judge by what it runs + how long the lease is held.
4. **"Not done until it's graph-capturable."** Production ZAYA = `--attn hip` + CUDA graphs. EP
   collectives must run INSIDE the captured decode graph at fixed shapes; lockstep via a per-step
   common-bs agreement OUTSIDE the graph (no in-graph host handshake, no variable `all_to_all`). Eager
   is necessary but NOT sufficient.

## STATE — what's done
- **Committed (`fccc840` on rdna4):** ZAYA port (`models/zaya.py`, CCA cache `kvcache/cca_state.py`,
  CCA graph capture `cca/graph_capture.py`, scheduler slots, registry), native **W8A8-fp8 MoE kernel**
  (`w8a8_fp8_wmma/`), Markovian **RSA shim** (`minisgl.rsa`). GPU-validated (numbers below — but taken
  against the shared tree, so RE-CONFIRM in isolation before relying on them).
- **W8A8 kernel:** replaces ZAYA's fp8→bf16-dequant→Triton MoE. Parity bit-close, coherent eager+graph.
  Decode TPOT M=1: **graph 23.9 ms (41.9 tok/s) / eager 43.5 ms vs old 270 ms** (~11× graph). Legacy
  dequant kept behind `MINISGL_ZAYA_OLDMOE=1` (A/B only).
- **Serving matrix (`SERVING_MATRIX.md`):** KV pool **65,615 tok** @0.90 util; frontier `B×L ≤ 65,615`;
  decode batching ~free; context dominates TPOT; **chunked prefill is mandatory** (`max_extend_tokens`
  small) or single-shot prefill OOMs on activations. RSA (4000-tail): **N=8 is the 1-card sweet spot**;
  N=16 needs DP=2. DP=2 measured ~1.98× (1066 tok/s peak), ~1.91× at B=8/L=4096 (454 tok/s).
- **DP launcher: WORKS** (uncommitted). `--data-parallel-size N` spawns full replicas + per-replica
  request routing in `scheduler/io.py` (replaces rank-0 broadcast-to-all) → one endpoint that routes →
  **removes the need for a multi-backend RSA shim.** Validated: dp=2 EP-off served concurrent chat
  completions OK (`tools/dp_serve_dp2_DP2.log`). NOTE: validated with `-v $PWD` → re-confirm in worktree.
- **EP: implemented, one bug fixed, NOT yet re-validated.** `scheduler/ep.py` has the graph-capturable
  common-bs lockstep (pad to agreed bs, idle replica runs all-dummy graph, EP collectives in-graph via
  all_gather + masked-local `w8a8_moe` + all_reduce — no new `all_to_all`/C++; experts shard 8/card).
  It crash-looped because the idle-replica dummy batch used `dummy_req` with `sampling_params=None`
  (`AttributeError: 'NoneType'.is_greedy` in `engine/sample.py:63`). **FIXED:** `engine/engine.py`
  `dummy_req` now uses `SamplingParams()` (greedy). Compiles; NOT yet GPU-re-validated.

## STATE — uncommitted DP/EP code (in the shared tree, mixed with another agent's work)
Mine: `distributed/{__init__,impl,info}.py`, `server/{args,launch}.py`, `scheduler/ep.py`,
`scheduler/scheduler.py` (ep_loop), `layers/moe.py` (EP dispatch), `engine/engine.py` (dummy_req fix),
plus `tools/dp_validate*.{sh,out}`, `tools/dp_serve_*.log`, `docs/zaya-port/{DP_EP_SPEC,dp_ep_workflow}`.
⚠️ The shared tree ALSO contains another agent's qwen35b / eagle3 / spec-sweep work + a
`scheduler/recurrent_slots.py` refactor (GDN+CCA share `RecurrentSlotManager`). **Do NOT `git commit -A`.**

## IMMEDIATE NEXT ACTIONS (in order)
1. **`EnterWorktree`** — make an isolated worktree off `fccc840`, bring in ONLY my intended DP/EP hunks
   (exclude the concurrent agent's qwen35b/eagle3/spec/recurrent_slots changes). All container mounts +
   AOT builds + validation happen against the WORKTREE path, never `$PWD`.
2. **Re-validate EP** in the worktree: `--data-parallel-size 2 --enable-ep`, **graph capture ON**
   (`--graph 16`), 2-card lease, ONE monitored fail-fast run (short timeout). Confirm: graphs captured
   incl. in-graph EP collectives (no eager fallback), coherent, and **greedy-parity vs DP-only**, and
   experts sharded 8/card (bigger KV pool than DP-only).
3. **Re-confirm load-bearing numbers in isolation** (W8A8 perf, DP/EP throughput) — current ones were
   measured against the unlocked tree.
4. **A/B with vs without EP** → write `docs/zaya-port/DP_EP_RESULTS.md`: KV-pool gain from EP (frees
   ~4 GB experts/card → more KV → more RSA concurrency), throughput, top-1 expert-skew (16 experts / 2
   ranks). The headline payoff of EP for ZAYA is KV headroom, not throughput.
5. **Commit** the DP/EP work (worktree → clean diff, no hunk-surgery needed).
6. **Codify the rules:** add a "Source isolation — per-task git worktree, never mount `$PWD`" section to
   `CLAUDE.md` next to the GPU-lease protocol (it's why this got skipped — the lease rule is written
   down, this one wasn't).

## Do NOT
- Do NOT resume `dp_ep_workflow.js` as-is (it uses retry loops + `$PWD`). Prefer manual, monitored,
  worktree-based validation. If you do use a workflow, mount the worktree and make GPU phases fail-fast.
- Do NOT load ZAYA in vLLM — the user wants it in minisglang.
