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
| `*-fusestorm` (BOTH repos) | `perf/fuse-storm-fp8norm` | 8 + 7 | **PREQUANT fusion** — the RMSNorm epilogue emits the per-row e4m3 activation + fp32 scale, so the MoE gemm1+silu and the W4A8 decode GEMV skip their pre-quant kernel entirely. 477 insertions, kernel side AND engine side. This is "Phase 2" in the decode plan, recorded there as *written, preserved, never measured*. Bit-exact by construction; does not change any kernel's grid. ~~Highest-value item in this table.~~ **A/B'd 2026-07-29 — NEUTRAL, see the verdict below. Not landed.** |
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


## VERDICT: fusestorm (PREQUANT norm->fp8 fusion) — NEUTRAL, do not land

A/B'd 2026-07-29 on Qwen3.6-35B-A3B-AWQ TP=2, sampled. Both deltas (477 kernel + 193 engine lines)
rebased onto current HEAD with zero conflicts, built inside the serve image, one build with the
`MINISGL_FUSE_NORM_FP8` gate flipped. Both legs verified: the mount was asserted via
`docker inspect .Mounts` and the flag read back with `printenv` INSIDE the container.

|          | off   | on    | delta |
|----------|------:|------:|------:|
| bs=1     | 80.29 | 79.96 | -0.4% |
| conc=4   | 232.5 | 233.9 | +0.6% |
| conc=8   | 364.5 | 366.4 | +0.5% |

All inside run-to-run noise (+-4% observed across runs the same day). The fusion **did** engage —
`prequant`/`rms_norm_fp8` log hits went 0 -> 2 — and output stayed coherent. So it is correct, and
it buys nothing on the served path.

**WHY, and it is the third time:** the fusion removes ~80 pre-quant kernel launches/step. The
served decode is **CUDA-graph captured**, where per-kernel dispatch is already free. It also removes
an activation write+read, but that is ~5 MB/step against a ~1.6 GB/step budget.

Launch count has now failed as a lever three times in a row, each time sized from an eager
microbench or a raw launch tally:
* KSPLIT tiling (more waves for small-N GEMV) — made `gate_up` *slower*, 296 -> 154 GB/s
* column-concatenated projections (110 fewer launches/step) — 84.07 vs 84.04, exactly 0.0%
* fusestorm PREQUANT (~80 fewer launches/step) — this table

**Rule: on the captured decode path, do not size a lever by launch count.** The eager 4.02 us/call
dispatch cost does not exist there; what remains is the GPU-side inter-kernel gap (~1.25 us) and it
is largely absorbed by capture. Size levers by BYTES or by a measured kernel-time budget instead.

The work is preserved (tags `wip/rdna4-hip-kernels-fusestorm`, `wip/minisgl-rdna4-fusestorm`) and is
worth revisiting **only** for an eager / non-captured configuration, where those launches do cost.
