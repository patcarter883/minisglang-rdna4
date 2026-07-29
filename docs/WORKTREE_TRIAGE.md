# Worktree triage — 2026-07-29

41 worktrees across `minisgl-rdna4` and `rdna4-hip-kernels` held **uncommitted** work. That is the
pattern that has already cost this project real work twice (the GDN fused-publish delta, and the
vhipspec patches — both recovered only by accident). This pass preserved all of it and classified it.

## Everything is now recoverable

| what | where |
|---|---|
| tracked modifications, 22 worktrees | git tags `wip/<worktree-name>` (made with `git stash create`, so **no worktree was modified**) |
| untracked source, 30 worktrees, 200 files | `~/wip-untracked-2026-07-29.tar.gz` (96 MB) |

Inspect a preserved tree with `git show wip/<name>` / `git diff <branch> wip/<name>`.
Nothing below has been deleted; the worktrees are exactly as their owners left them.

## TIER 1 — uncommitted CORE-SOURCE changes

Ranked by number of core files touched (`python/minisgl/`, `*_rocm/*.hip`, `torch-ext/*`).
**Two of these are levers this repo's own perf plan lists as pending, fully written, never landed:**

| worktree | branch | core files | what it is |
|---|---|---:|---|
| `*-fusestorm` (BOTH repos) | `perf/fuse-storm-fp8norm` | 8 + 7 | **PREQUANT fusion** — the RMSNorm epilogue emits the per-row e4m3 activation + fp32 scale, so the MoE gemm1+silu and the W4A8 decode GEMV skip their pre-quant kernel entirely. 477 insertions, kernel side AND engine side. This is "Phase 2" in the decode plan, recorded there as *written, preserved, never measured*. Bit-exact by construction; does not change any kernel's grid. **Highest-value item in this table.** |
| `*-repcca` | `feat/replicate-linear-attn` | 8 | Replicates the linear-attention mixer so the GDN `o_proj` all-reduce disappears (~30 of 81 `custom_ar` calls, est. 0.4 ms/step). Plan lists it as *written, preserved, unvalidated*; costs VRAM on 16 GB cards, so it needs a VRAM check + A/B. |
| `*-redkill` | `perf/gemv-reduction-kill` | 5 | GEMV cross-lane reduction removal + a `lanecol` parity/perf harness. |
| `*-cumode` | `perf/cu-mode-test` | 5 | CU-mode experiment; touches `gemv_decode.h`. |
| `*-minvmoe` | `task/minv-moe` | 5 | M-invariant MoE + a `swa_window.py` that exists nowhere else. |
| `*-arep` / `*-agep` / `*-agepf` | `perf/custom-a{r,g}-ep{,-fused}` | 5/5/4 | custom all-reduce / all-gather over EP. |
| `*-vhipspec` | `task/vhip-glm-spec` | 4 | vLLM-0.24 GLM spec harness + `tools/vhip_patches/`. The GDN half was recovered and is now in `rdna4-hip-kernels` main; the MLA patches here are still only on disk. |
| `*-epfuse`, `*-camfix`, `*-moealign`, `*-gdnar`, `*-dsplitk`, `*-carcomms`, `*-tapmetric`, `*-eos`, `*-asyncar2`, `*-rxf`, `*-zcbundle`, `*-gdnsplit`, `*-dsplit`, `*-specdiag`, `*-moepad`, `*-hostloop` | various | 1-4 | smaller deltas; see the `wip/*` tag for each. |

`*-dsplitk` additionally has an **unresolved merge** (`UU` on two files) — a merge left half-finished.

## TIER 2 — tooling/scratch only, branch already merged (droppable)

`*-requant`, `*-camtap`, `*-f16gemv`, `*-fusion`, `*-gemv`, `*-localize`, `*-overlap`, `*-specfix`,
`*-swaverify`, `*-vcap`, `agent-a3dc3cb79fec6f58f` — benches, probes, drivers, A/B shell scripts.
All preserved in the tarball; the worktrees can go once someone confirms they want nothing back.

## TIER 3 — tooling/scratch only, branch unmerged

`*-alignbase`, `*-cartridge`, `*-glmperf`, `*-rxf`, `*-specgrammar`. Branch still carries unique
commits, so keep the branch; the working-tree dirt itself is scratch.

## The rule this pass suggests

A worktree is a *lease on the source*, like a GPU lease — it is supposed to be released. The failure
mode here is not that people leave worktrees, it is that they leave them **dirty**, and dirty state
is invisible to every `git log` anyone will ever run. `git stash create` + a tag costs nothing and
makes the state greppable; consider doing it at the END of a task rather than months later.
