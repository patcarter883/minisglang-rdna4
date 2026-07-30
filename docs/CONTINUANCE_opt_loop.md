# CONTINUANCE — opt_loop: CCA serving loop + next kernel loop

Written 2026-07-30. Pick this up in a fresh session to kick off two runs. Everything below was
established by measurement in the previous session; the "do not repeat" list cost most of a day.

---

## 0. READ THIS FIRST — how to pick a target

**Do NOT pick targets from `rdna4-hip-kernels/KERNELS.md`.** It was wrong on every target decision
made against it in one session:

| ledger claim | reality (measured) |
|---|---|
| moe_splitk RD-b128 = "highest-value single gap in the family" | op sits behind `MOE_SCATTER=0`; ledger never mentions it is unreachable in the served config |
| attn_decode b128 K/V "untried, not reverted" | tried: it is a **numerics** change (18/46 tensors differ), not a load-width tweak |
| cca "b128x2 measured-best ~1.12×, b64 still default" | already wired behind `MINISGL_CCA_DECODE_B128`, measured **~1.05× in situ**, off for a documented reason |
| tail b128 = "biggest lever" | `PERFORMANCE.md` measures tail at **100% of HBM** — nothing to win |
| lists `moe_bf16`, `moe_w8a16`, `rxf` + 2 more | **not on disk** |
| Part 2/3 rocwmma rows | contradicted by its own Part 0 |

**Target selection procedure that replaces it:**

1. **Profile a real serve step** (`rocprofv3 --kernel-trace` on a live decode loop), rank kernels by
   summed `End-Start`, Amdahl-filter. Not a microbench, not a ledger.
2. **Check REACHABILITY before spending anything.** Grep the engine call site and confirm the op
   actually executes under the production config (`--graph N`, `MOE_SCATTER=0`). An op behind a
   default-off flag is not a target until you know *why* the flag exists.
3. **If there is a flag, read its justification and test it.** Two flags were checked this session:
   one justification was false (see §3), one was a real measurement (see §2). You cannot tell which
   without looking.
4. `KERNELS.md` is fine as a *reference for how a lever works* once the profile has chosen the
   kernel. It must never choose the kernel.

---

## 1. STATE — what is on main

`rdna4-hip-kernels` @ `202dd65`, `minisgl-rdna4` @ `dacc5b85`.

| commit | what | status |
|---|---|---|
| `3f8cb2c` + `25cc782` | moe_splitk routed to the register-direct core, **now default-ON** | 1.767× isolated; **see §3 — not reachable under graph capture yet** |
| `f442054` | attn_decode split-reduce re-grid + (SKU × KV-dtype) FILL table | 1.243× bit-exact; composed **1.357×** on 32 WGP, 1.03× on 28 WGP |
| `74e28c4` | opt_loop fixes (default-ON rule, gate-aware verify, control labels, profiler + serving harnesses) | |
| `dacc5b85` / `202dd65` | **one image**: every script targets `minisgl-rdna4:lean` (the compose serving image) | 24 stale tags + `vllm22-w4a8:combined` deleted |

**Measured serve baseline** (before today's kernel work was in the image):
`Qwen3.6-35B-A3B-AWQ-4bit`, TP=2, `GRAPH=16`, `MOE_SCATTER=0`, M=1 decode → **69.1 tok/s**.

### Prerequisites before either loop

- [ ] **`minisgl-rdna4:lean` rebuild** was in flight at handover (`KERNELS_REF=202dd65`, clean
      worktrees at `/home/pat/code/_build-kernels` and `/home/pat/code/_build-engine`). Verify it
      finished and is newer than 2026-07-30, else rebuild:
      ```
      cd /home/pat/code/_build-engine
      docker build -t minisgl-rdna4:lean --build-context kernels=/home/pat/code/_build-kernels \
        --build-arg KERNELS_REF=$(git -C /home/pat/code/_build-kernels rev-parse --short HEAD) .
      ```
      Then `git worktree remove` both build worktrees.
- [ ] **`opt_loop/bin/serve_ab.sh` has NEVER booted a serve.** Validate it once on `zaya_decode`
      before any loop depends on it.
- [ ] **Re-measure the 69.1 tok/s baseline on the new image** — today's kernel work is only in the
      image after the rebuild, so the old number cannot show it.

---

## 2. TASK A — CCA **serving** loop

**Target: grid occupancy at batch-1. NOT b128 load width.**

`KERNELS.md` records `H≈10 workgroups live at batch-1` on a 32-WGP card, and both it and the engine
agree the ceiling is *"grid occupancy (output-partition / spec-decode, not load tuning)"*. The
engine adds the diagnosis at `python/minisgl/models/zaya.py:216-218`:

> *"Bit-exact vs `cca_decode_qk` (pure permutation) but only ~1.05× — **CCA decode is
> launch-overhead-bound, not w1-bytes-bound** — so it's off by default."*

So the b128/b128x2 twin is already implemented, already wired, already measured, and correctly
disabled. **Do not re-target it.** The lever is filling the grid: output-partitioning, or more work
in flight via spec-decode.

**Why this needs the SERVING loop, not the kernel loop:** launch overhead only exists in the real
decode loop. This is also the standing counter-example in the memory — *CCA CU +25% isolated gave
**0% e2e***. An isolated microbench will lie to you about this kernel specifically.

Launch:
```
Workflow({scriptPath: '/home/pat/code/rdna4-hip-kernels/opt_loop/kernel_loop.workflow.js',
          args: {kernel:'cca', pkgDir:'/home/pat/code/rdna4-hip-kernels/cca',
                 gate:'bitexact', budget:4, perRound:2}})
```
…**but** the loop script is still kernel-shaped. For CCA it must use
`opt_loop/COMMANDMENT_serving_template.md` and gate through `opt_loop/bin/serve_ab.sh`
(`CONFIG=zaya_decode`), where accept = **non-overlap** (`min(cand) > max(base)`) across interleaved
reps in ONE lease, not a mean ratio. Wiring that mode switch into the workflow script is the first
piece of work.

---

## 3. TASK B — next kernel loop, and the biggest single open win

### 3a. The open win: make the fused MoE scatter path reachable under graph capture

This is worth more than a new loop and should probably go first.

`python/minisgl/quant/kernels.py:33` claims the atomic scatter *"is NOT graph-capture-safe"*, which
is why `MINISGL_MOE_SCATTER` defaults to 0 and the whole fused decode-gemm2 path is unreachable in
production. **That claim is false — tested and disproved:**

```
CAPTURE: OK
replay 1: finite=True max|delta_vs_eager|=1.43e-06   (NaN-poisoned before each replay)
replay 2: finite=True max|delta_vs_eager|=1.43e-06
replay 3: finite=True max|delta_vs_eager|=1.19e-06
```
(probe: `scratchpad/capture_probe.py`, reproduced below in spirit — persistent `acc`, `acc.zero_()`
*inside* the captured region, then the scatter.)

And the gated path is **faster than what production runs today**, head-to-head at the same shape,
amortized + cache-busted:

```
moe_splitk atomic-scatter (gated)     21.93 us
g2fuse gather_reduce (PRODUCTION)     38.42 us
scatter / g2fuse = 1.7519x
```

So: a **1.75× on the decode gemm2** is sitting behind a flag whose justification is wrong. The
engine change is small — hoist `acc` from a per-call `torch.zeros((M,K))` to a persistent buffer,
zero it inside the captured region, and flip the `M <= 2 and _MOE_SCATTER` gate. Note the split-K
kernel additionally needs `_MOE_SPLITK >= 2` (also defaults to 0).

Caveats to respect:
- The probe used a **synthetic shape**; the engine captures a whole decode step, not one op.
- Atomics make the result **batch-order dependent** — use the tolerance gate, not bit-exact.
- Verify at the real serve shape: `KERNEL_CORE_POLICY` quotes in-serve gemm2 at 19.2 µs where this
  measured g2fuse at 38.42, so one of those is shape-specific or stale.

### 3b. The kernel loop itself

Pick the target from a **fresh serve profile** (§0), not the ledger. Run it with:
```
Workflow({scriptPath: '/home/pat/code/rdna4-hip-kernels/opt_loop/kernel_loop.workflow.js',
          args: {kernel:'<pkg>', pkgDir:'/home/pat/code/rdna4-hip-kernels/<pkg>',
                 gate:'bitexact', budget:4, perRound:2}})
```

---

## 4. RUNNING TWO LOOPS IN PARALLEL

Three blockers, all real, two already half-fixed:

1. **`rocprofv3` cannot run twice concurrently** (`scripts/rocprofv3_profile.sh:23` — rocprofiler-sdk
   registration race, SIGABRT). `opt_loop/bin/profile_kernel.sh` needs a **global flock** around the
   metrix call. NOT YET DONE.
2. **No card pinning.** `gpu-lease` has no "pick card X" flag, and the two cards differ by ~1.24×
   (GPU0 64 CU @2400 / 32 WGP; GPU1 56 CU @2210 / 28 WGP), both reporting `gfx1201`. Fix: a
   **timing** leg should lease `-n 2` for exclusivity even though it uses one card. Builds and ISA
   work stay `-n 1` / no lease. To pin a specific card deterministically, lease both and select
   inside with `ROCR_VISIBLE_DEVICES=<card> HIP_VISIBLE_DEVICES=0`.
3. **Contamination.** Noise bands (4.2% moe_splitk, 6.9% attn_decode) were measured on a quiet box.
   Never run two serving legs at once — shared PCIe/host-BW/power drags the baseline leg down and
   *inflates* the ratio into a false win.

CPU is **not** a constraint: 16 cores, idle load ~2.

---

## 5. HARD RULES — each of these cost real time this session

- **If it merges, it is ON.** No env var, no opt-in kwarg, no `default=None → old path`. The
  worktree is the isolation. moe_splitk's 1.767× sat inert on main because it merged behind
  `w_rep=None`. Repo precedent: `KERNELS.md:194` (Qwen MXFP4 decode at ~½ speed because
  `MINISGL_MOE_MXFP4_REGDIRECT` was off), `accum_bylane` shipped DEFAULT OFF at a measured 1.11×.
- **Run the CONTROL before trusting any gate.** Record the baseline **twice** and compare. A first
  parity run reported *18/18 FAIL* on a **correct** fold; three harness bugs caused it —
  `hash()` seeding (per-process randomised), unmasked **uninitialised padded output rows** (22065
  NaN, absmax 6.55e4 = fp16 max), and gating an **atomic** op bit-exactly (impossible even against
  itself).
- **Controls are not results.** Label `per_case` entries `control:` / `measured:`. Six control cases
  at ~1.000 averaged with one real 1.767× produced a phantom "1.083×".
- **A profiler that exits 0 with all-zero counters is a FAILURE.** Six rocprofv3 attempts / ~95 min
  ended `rc=0` with every counter `0.0000`, because metrix **derived** names
  (`OccupancyPercent`, `LDSBankConflict`, …) were passed to `rocprofv3 --pmc`, which wants **raw**
  counters. Use `opt_loop/bin/profile_kernel.sh` — it validates names (exit 4), fails closed if it
  cannot validate (exit 5), and treats all-zero as failure (exit 3).
- **Never hand-roll a workspace copy, never `rm`.** `rm` raises an approval prompt and **stalled an
  unattended run for 9 hours**. Use `opt_loop/bin/fresh_workspace.sh`.
- **Diagnose before killing.** Check CPU + process tree + log growth and state the verdict
  (working / crashed / wedged) *before* acting. I killed a healthy profiling container without
  looking and destroyed the evidence.
- **Build/bench only inside the container.** Host torch is 2.10 vs the image's 2.14 — a host-built
  `.so` fails with a silent `libc10` import error presenting as `gpu_use=0%`. Host torch is
  deliberately left broken as a guardrail; do not "fix" it.
- **One image.** `minisgl-rdna4:lean`, the compose serving image. A per-experiment tag is
  worktree-local, never a default in a committed script.
- **The two cards are not equivalent** and both report `gfx1201`. Tag every number with its card;
  an unpinned A/B is a ~20% phantom.

---

## 6. THE HONEST GAP

None of the three wins on main has been measured **end-to-end in tok/s**. The `attn_decode` 1.357×
is isolated; `moe_splitk`'s 1.767× is on a path production cannot currently reach. Given this
repo's own history — *CCA +25% isolated → 0% e2e* — treat all three as unproven for serving until
a serve A/B says otherwise. Closing that gap is worth more than starting anything new.
