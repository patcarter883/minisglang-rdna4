# CONTINUANCE — minisgl serves garbled output (EOS bug fixed; GLM coherence regression bisect in progress)

Paste this into a fresh session. Repo: `/home/pat/code/minisgl-rdna4`, default branch `rdna4` (tip `687db94`).
Both gfx1201 cards were free at handoff. Follow this repo's CLAUDE.md for all GPU work (gpu-lease,
worktree isolation, lean serve compose profile). **Lesson that triggered all this: validate REAL free
generation — clean EOS stop AND coherent content — not narrow paths. Premature "it works" claims were
the problem.**

## THE HEADLINE
The user proved minisgl garbles output across models while the SAME checkpoints run fine on vLLM
(:8000) and Lemonade GGUF. It is the server. Two distinct server bugs:

### Bug 1 — EOS / stop-tokens — **FIXED + GPU-VALIDATED, NOT yet committed/merged**
Root cause: server stopped only on `tokenizer.eos_token_id` (a SINGLE id). Chat models emit an
end-of-turn token that is NOT the base eos, and their real stop set is a LIST in
`generation_config.json` (GLM `[154820,154827,154829]` vs tokenizer `<|endoftext|>`=154820; Qwen
`[248046,248044]`). Server never read generation_config → generation blew through turn boundaries and
leaked `<|user|>`/`</think>` into content. Also detok didn't `skip_special_tokens`.
Fix (3 files, applied to BOTH the main checkout AND worktree `/home/pat/code/minisgl-rdna4-eos` on
branch `fix/eos-stop-tokens` off rdna4):
- `python/minisgl/utils/hf.py`: `_eos_token_ids(model_path, tokenizer)` unions tokenizer eos +
  `GenerationConfig.from_pretrained(...).eos_token_id`; attaches `tokenizer.minisgl_stop_set` (NOTE:
  attr must NOT end in `_id/_ids` — the tokenizer `__setattr__` intercepts those as special-token
  setters and raises "Cannot set a non-string value as the eos_token"; hence `minisgl_stop_set`).
- `python/minisgl/scheduler/scheduler.py`: `self.eos_token_ids = self.tokenizer.minisgl_stop_set`;
  4 stop checks now `in self.eos_token_ids` (lines ~296/467/898/1213).
- `python/minisgl/tokenizer/detokenize.py`: uses the set for the trailing-EOS trim; `batch_decode(...,
  skip_special_tokens=True)` on both decode calls.
Validated on GLM-4.7-Flash-AWQ: `SPECIAL-TOKEN LEAKS: NONE`, `finish_reason: stop` (no more cap-blow).
**TODO: land it** — commit worktree `fix/eos-stop-tokens` and merge to `rdna4` (clean fast-forward
worktree cherry-pick like prior merges; do NOT drag other branches' commits).

### Bug 2 — COHERENCE REGRESSION — **BISECT IN PROGRESS, this is the active task**
Even after Bug-1 fix, GLM-4.7-Flash-AWQ produces degenerate repetition ("The user is asking… 17 + 25.
17 + 25…", "* * *"), never reaching the answer, at BOTH greedy AND temp 0.7/top_p 0.9. This is the real
"garbling." It is GLM/MLA-specific: TODAY on the SAME lean stack, base-Zaya (GDN) produced coherent
code, Qwen3-0.6B (dense) produced valid JSON, Qwen3.6-35B (GDN-AWQ) produced a coherent tool call —
only GLM (MLA + QuantTrio-AWQ) degenerates. Memory says GLM-4.7-Flash was previously "GPU-validated
coherent (…is Paris)" → it REGRESSED.

Bisect facts:
- Good baseline `1baf6e5` "glm: TP=2 MLA sharding — GLM-4.7-Flash serves (GPU-validated, coherent)"
  — PRE lean migration.
- Bad `687db94` (rdna4 tip) — confirmed degenerate.
- Prime suspect `3ff7598` "Lean vllm-free serving image + de-vendor kernels" — its validation covered
  0.6B + 35B AWQ but NOT GLM/MLA. Lean era `3ff7598..687db94` = **32 commits**.
- **CONFOUND (critical):** the lean migration also changed the KERNEL IMPORT INTERFACE
  (`import tail_hip` → `torch.ops.tail_hip_C`). `3ff7598` crash-loops on the CURRENT lean image with
  `ModuleNotFoundError: No module named 'tail_hip'`. Kernels are BAKED in the image (minisgl-rdna4:lean),
  NOT in this repo's git. So a naive source bisect across the interface boundary is INVALID, and a
  kernel-level regression would not be git-bisectable here at all.

Coherence test used (no EOS fix needed — good=says Paris/42, bad=degenerates):
`curl :1919/v1/chat/completions` greedy, prompts "What is the capital of France? One word." (expect
Paris) and "What is 17 + 25? Answer briefly." (expect 42).

## NEXT STEPS (in order)
1. Find the earliest lean-era commit that BOOTS on the current lean image (kernel interface =
   `torch.ops.<pkg>_C`, not `import tail_hip`). I was mid-search: `git log --oneline 3ff7598..687db94
   -- python/minisgl/layers/_tail_hip.py python/minisgl/layers/activation.py` (candidates seen:
   `232a015`, `ae16731`, `753ad89`, `1ba037d`, and de-vendor commits `fd9580b/0973e31/d776ceb`). Grep
   each candidate's `_tail_hip.py` for `torch.ops`/`tail_hip_C` vs `import tail_hip` to find the boot
   boundary.
2. Serve GLM at that earliest-bootable commit (clean worktree, lean image, `MINISGL_MODEL=
   QuantTrio/GLM-4.7-Flash-AWQ MINISGL_TP=2 MINISGL_CUDA_GRAPH_MAX_BS=0` — GLM is MLA→eager). Test
   coherence.
   - COHERENT → source-bisect [that commit .. 687db94] on the current image (~5 steps), each = worktree
     checkout + serve + Paris/42 test.
   - DEGENERATE → regression is in the lean image / canonical kernels / MLA integration, NOT later
     source. Then: (a) inspect `python/minisgl/attention/mla.py` + the mla_hip kernel usage for a
     lean-migration change; (b) optionally confirm old=good by serving a PRE-lean commit (e.g.
     `1baf6e5`) on the OLD image `vllm22-w4a8:combined` with the vendored `.so` (gitignored — must
     build or copy the 12 kernel .so into that worktree; see memory "Worktree validation needs .so
     copied").
3. Whatever the bisect yields, VALIDATE the fix with real free generation (Paris + 42 + clean stop),
   then land Bug-1 EOS fix + the Bug-2 fix on rdna4.

## SESSION CONTEXT (already landed on rdna4 = 687db94)
- RSA sampling knobs (top_p/top_k/ignore_eos/stop) — `96af03a`.
- `auto`→`hip` attention default on ROCm + base-Zaya DP2+EP compose — `fb68c2e`.
- Grammar overlap-scheduling doubling fix + OpenAI tool calling — `687db94`.
- These are validated for what they cover, but tool-calling/RSA validation NEVER checked clean free-gen
  EOS — so they coexist with Bug 1/2.

## OTHER OPEN THREADS (lower priority)
- **Zaya finding may be mis-attributed**: this morning I filed "TiDAR-OPD checkpoint degenerately
  collapses" to the `zaya` repo (`/home/pat/code/zaya`, pushed) — but the `= = =`-to-cap was PLAUSIBLY
  Bug 1/2, not the checkpoint. REVISIT after the coherence fix (re-serve the TiDAR ckpt with fixes).
- **EP deadlock** filed as `minisglang-rdna4` issue #1 (DP=2+EP ALLREDUCE deadlock under RSA fan-out).
- **Laguna port plan** at `docs/laguna-port/LAGUNA_SERVING_PLAN.md`. User corrections to fold in:
  (a) W8A8 MoE kernel EXISTS (`quant/kernels.py:316 w8a8_moe`) → INT8 experts are wiring not a new
  kernel (verify fp8-vs-symmetric-int8 format); (b) the Hadamard `transform_config` is the RXF quant,
  handled by `rxf_hip` (`quant/kernels.py:432`, `rotate_quant_int8`) — not an unhandled unknown. Both
  collapse Phase-2's "big risks" to wiring. Laguna serving is deferred behind the coherence bug.

## HOUSEKEEPING
- Worktrees to remove when done: `/home/pat/code/minisgl-rdna4-tools` (root-owned leftover — needs
  elevated rm), `/home/pat/code/minisgl-rdna4-eos` (branch fix/eos-stop-tokens — keep until Bug-1
  landed), `/home/pat/code/minisgl-rdna4-bisect` (detached — reuse for the bisect).
- Main checkout is on branch `cam-serve-integration` (another agent). It ALSO has the uncommitted
  Bug-1 EOS fix + another agent's zaya-rope-sanitize change in `utils/hf.py` — don't clobber.
- `_eos-fix.patch` of the 3-file Bug-1 fix is at `/tmp/eos-fix.patch` (pre-rename; the live version
  uses `minisgl_stop_set`).
