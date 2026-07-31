# CONTINUANCE — Laguna spec-decode cost + Laguna-vs-Qwen plain-decode gap

Session date: 2026-07-31. Branch `spec-ondevice-accept`, worktree `/home/pat/code/minisgl-rdna4-specod`
(off `4a031cc3`). Trigger: a lucebox blog post claiming 296 tok/s for Laguna-XS-2.1 + the official
DFlash drafter on a single RTX 3090, vs our numbers.

---

## 1. Measured baselines (reproduce before trusting anything below)

All `poolside/Laguna-XS-2.1-NVFP4`, TP=2, graph-captured, greedy, 384-token generation,
`MINISGL_SPEC_MHA_PAGED=1 MINISGL_SWA_RADIX=1`, `CONC=2 GRAPH_BS=8 MEM_RATIO=0.96`.

| Config | True tok/s | Accept-len | Step |
|---|---|---|---|
| **plain (no spec)** | **74.02** | — | 13.51 ms |
| spec DFlash K=7 | 68.45 | 2.553 | 37.40 ms |
| spec DFlash K=11 | 55.55 | 2.570 | 46.45 ms |
| spec DFlash K=15 | 47.50 | 2.582 | 54.36 ms |
| spec DFlash K=16 (**shipped default**) | 39.68 | 2.582 | 66.27 ms |

**Spec is net-negative at every K.** Plain decode wins.

> **SUPERSEDED 2026-07-31 — THIS IS NO LONGER TRUE, AND IT IS THE HEADLINE OF THIS DOCUMENT.**
> On the current build, measured against a `--spec-algorithm none` PLAIN leg **in the same boot, same
> config** (§11.8, K=15, TP=2, bs=1, graph 8, fp8 KV, radix, two full replicates):
>
> | prompt | plain tok/s | DFlash spec tok/s | |
> |---|---:|---:|---:|
> | 3571-token real code @1600 | 65.59 | **73.55** | **+12.1%** |
> | 95-token instruction @384 | 79.32 | **101.11** | **+27.5%** |
>
> DFlash spec on Laguna is a WIN, not a loss. Nothing about the drafter changed. What changed is
> everything this document blamed it for:
>   * **K=16 -> 15** — the shipped default sat exactly one row past the `M<=16` decode-kernel cliff
>     (§2 finding 1, +20% for a one-character change), so every number in the table above was
>     measured on the prefill/WMMA kernel family;
>   * **verify-side MoE gemm2 is 3.02x cheaper at qlen 16** (§9) — group-16 by-lane;
>   * the fused sigmoid+bias route (§ router work) lifts BOTH legs, so it is not the source of the
>     ratio, but it is in the build;
>   * and the table above is a PROSE benchmark, the worst of the three prompt classes (§11.2).
>
> The row that matters for the original lucebox comparison is the SHORT one: **101 tok/s single-stream
> greedy at 95 tokens**, versus 74.02 for plain in this table.
**SCOPED 2026-07-31 by §11: this whole table is a PROSE benchmark** (a 384-token B-tree essay). The
accept-lens are correct for that prompt and wrong as a property of the model — the same build, same
K, same day gives accept-len 4.5 on real code and 9.5 on repetitive output. Re-read §7 and §11
before quoting any row here.

GPU-side, from `MINISGL_PROFILE` traces (`tools/spec_profile_split.sh`):

| | Laguna plain | Laguna spec K=16 | Qwen3.6-35B plain |
|---|---|---|---|
| GPU/step | 9.79 ms | 64.5 ms | 8.83 ms |
| Wall/step | 13.51 ms | 66.27 ms | 11.12 ms |
| Non-GPU/step | 3.72 ms (27.5%) | ~1.8 ms | 2.30 ms (20.7%) |
| Dispatches/step | 1974 | 2922 | 1224 |

TTFT: `page_size=16` + SWA-radix took TTFT **215 ms -> 18.7 ms (11x)**. Keep both flags on.

### MEASUREMENT TRAP — cost me half the session
Under spec, **one SSE stream chunk carries a whole accepted block**, so counting stream chunks
undercounts tok/s by the accept-len (I read 14.9 tok/s when the truth was 49.8). Always take
`usage.completion_tokens` from a non-streaming request, or `minisgl_generation_tokens_total`.
`tools/spec_k_sweep.sh` has the correct harness.

---

## 2. CONFIRMED findings

1. **`M<=16` decode-kernel dispatch cliff.** Verify runs at `qlen = K+1`. The decode GEMV/MoE kernels
   assert `M<=16` (`quant/kernels.py:1117`, `quant/method.py:28`), so the shipped `spec-num-draft 16`
   -> qlen=17 falls off the decode path onto the prefill/WMMA family. K=16 -> K=15 is **+20%
   (39.68 -> 47.50) at identical accept-len**. `SPEC_K=16` is the single worst possible value.
2. **Accept-len is flat at ~2.55-2.68 for K=7..16.** Drafting 16 tokens buys nothing over 7. Acceptance
   is NOT the problem and a better drafter is NOT the lever.
   **WRONG AS STATED — see §11.** The flatness is an artifact of sweeping K on a single prose prompt.
   Measured across prompt classes on one build, accept-len spans 2.6-9.5 (and 7.4 on real code once
   the drafter's 512-token window is full). Acceptance IS the lever; the drafter is never conditioned
   on the prompt (§11.5). Note also `spec/dflash.py:523` caps drafts at `block_size-1 = 15`, so
   K=16 cannot draft more than K=15 regardless.
3. **Expert divergence is why verify doesn't amortize.** 256 experts, top-8: K tokens route
   independently and pull up to 8*K distinct experts. Measured at M=17, dense projections cost 2.3x
   for 17x rows (amortizing correctly) while MoE costs **14x** (32.8 ms of the 64.5 ms step).
   Predicted MoE doubling from qlen 8->16 is +15 ms; measured step delta +16.96 ms.
   **RESOLVED — see §7. The open question is answered: the alignment is optimal (no kernel bug), but
   the "disjoint routing" premise stated here is WRONG. Overlap is substantial; the correct growth
   law is `distinct(qlen)`, which is 6.94x at qlen 16, not 8x.**
4. **Drafter propose costs ~9 ms/step.** Fitting the sweep gives `step ~= 22.6 ms + 2.12 ms * qlen`;
   the 22.6 ms intercept vs plain's 13.51 ms is drafter overhead. lucebox's is 2.4 ms. py-spy shows
   the drafter running eager with live per-layer `models/dflash.py` `attend_block` frames.
5. **By-lane GEMV is disabled for group-16.** `rdna4-hip-kernels/fp8_wmma/fp8_wmma_rocm/gemv_decode.h:157-159`:
   `const bool bylane_ok = bylane_available && group_size >= 32;` — excluded only because
   `accum_bylane` folds one scale group per 32-k chunk and group-16 spans two. Laguna's MoE down-proj
   is `K = moe_intermediate/TP = 512/2 = 256`; the kernel's own table (`gemv_decode.h:99-100`) at
   K=256 is **K-on-lanes 43.6 us vs BYLANE 9.4 us (4.6x)**. Worth ~0.64 ms/step. The serve warns about
   this at every boot and nobody had read it.
   **DONE — see §9, but the numbers here are wrong.** The 4.6x/0.64 ms comes from a `us/MB` table for
   a DIFFERENT kernel; the fused gemm2 loops over `top_k` outside the sweep, so at the served M=1 it
   measures **0.85x (a loss)**. It is +20.1% at NREQ=8. The lever is real but it is a CONCURRENCY
   lever, not a bs=1 one.
6. **NVFP4 group-16 scale tax.** fp16 scale per 16 weights = 0.125 B/param = **25% on top of the 4-bit
   weights** (effective 5 bits vs MXFP4 group-32's 4.25), ~3.93 GB of scale traffic across the expert
   stack. `nvfp4.py` folds the per-tensor global into the block scale, and the quotient needs fp16.
   Keeping the global separate would let scales stay e4m3 **uint8** and halve that traffic.
7. **Laguna-vs-Qwen plain gap (74 vs ~90) decomposes as** +0.96 ms GPU (essentially all MoE: 2.74 ms
   vs 1.86 ms) + 1.42 ms host (**1974 vs 1224 dispatches/step**). Everything else is at parity or
   better — Laguna's dense GEMV is 3.249 ms vs Qwen's 3.89, SWA attention 0.303 ms beats GDN's ~0.37.
   Caveat: not a matched A/B (Qwen numbers from `33efd96a`/`ffe17f00` at GRAPH_BS=16/max-running 16/AWQ).

---

## 3. FALSIFIED — do NOT re-investigate

| Hypothesis | Verdict | Evidence |
|---|---|---|
| `page_size=1` causes the spec decode cost | **NO** | 14.92 -> 14.87 tok/s. (It IS worth 11x on TTFT.) |
| Reasoning think-gate is the bottleneck | **NO** | gate off/on = 50.0 / 48.8 tok/s (+1.7%) |
| gloo TP lockstep (`_bcast_drafts_tp`/`_bcast_accept_tp`) | **NO** | step is 64.5 ms **GPU** of 66.3 ms wall; gloo is sub-ms |
| Verify runs K sequential forwards | **NO** | `spec/dflash.py:580-590` one denoise emits all K; verify is one batched `[qlen, vocab]` |
| Accept/commit path cost | **NO** | `MINISGL_SPEC_FORCE_N0=1`: accept-len 3.34 -> 2.04, step **identical** (67.1 vs 67.2 ms) |
| MoE gemm2 routed to the wrong kernel | **NO** | `MINISGL_MOE_G2FUSE=0` (forces WMMA grouped gemm2) identical: 54.14 vs 54.33 ms |
| NVFP4 upconverts weights to fp8 | **NO** | VRAM math: measured 10.40 GiB/card matches 4-bit+scales (21.3 GiB); fp8 would be 31.4 GB of experts alone |
| ~~Low acceptance / need a better drafter~~ | **RETRACTED by §11** | the "flat across all K" evidence was one prose prompt; accept-len is 2.6-9.5 by prompt class and rises to 7.4 on code as the drafter's window fills |
| NVFP4 scale traffic is worth halving (e4m3 uint8) | **NO** | 34.5 MB of a 2258 MB step; step is at 23.7% of HBM peak -> 0.30% bs=1 / 0.14% NREQ=8 (§10) |

`_gate_mask_spec_logits` (`scheduler.py:1271`) showed as **63% of py-spy samples** — that is
**sync absorption**, not work. Every leaf frame was in `libhsa-runtime64.so`; the list-index H2D there
is the first blocking op after the verify forward. Don't chase it.

---

## 4. Work in progress on this branch (uncommitted)

- **`python/minisgl/spec/accept_gpu.py`** — added `truncate_at_stop_ondevice`, generalizing the
  single-id `truncate_at_eos_ondevice` to (a) a bool vocab LUT for **multi-EOS** and (b) a per-req
  `gate_tid` regime for **`</think>` truncation**, since gate truncation is the same "stop at first
  occurrence" operation. `truncate_at_eos_ondevice` kept as a delegating shim. **Validated: 1500 cases
  exact-match vs the host loop** — `tools/validate_stop_trunc.py` (CPU, no lease needed; run it inside
  the image, host torch is broken: `libmpi_cxx.so.40` missing).
- **NOT yet wired into the scheduler.** Three gates at `scheduler.py:3424-3430` each independently
  block the on-device path for a production Laguna request: `not any_sampled`, `not any_gated`,
  `len(self.eos_token_ids) <= 1`. Laguna's `generation_config.json` has `"eos_token_id": [2, 24]`.
- **`docker-compose.yml`** — added `MINISGL_SPEC_FORCE_N0` passthrough.
- **New tools:** `validate_stop_trunc.py`, `spec_cost_isolation.sh`, `spec_gate_isolation.sh`,
  `spec_k_sweep.sh`, `moe_g2fuse_ab.sh`, `spec_profile_split.sh`. All the `.sh` ones must be invoked
  **under** the arbiter (`gpu-lease -n 2 -- bash tools/<x>.sh`) — they deliberately do not lease
  themselves. Traces `tools/{spec,plain}_trace.json` are on disk (14M / 7.2M).

---

## 5. Next actions, ranked

1. **Measure distinct experts per verify batch** (finding 3's open question). One instrumented run.
   Decides whether MoE verify cost is inherent or a kernel bug — and it also decides whether
   lucebox's 296 tok/s is even reachable. **Do this first; it gates everything about spec.**
2. **By-lane GEMV for group-16** (finding 5). Extend `accum_bylane` to fold two half-chunk scales
   (the K-on-lanes path already does a per-16-K-half fold), then relax `group_size >= 32` to `>= 16`.
   **CORRECTION: this is Laguna-ONLY, not "helps Qwen more".** Qwen3.6-35B-AWQ is `group_size: 32`
   (checked in its `config.json`), so it already cleared the old gate — its +7.7% e2e IS by-lane,
   already banked. Only group-16 NVFP4 was excluded, i.e. Laguna. Canonical-kernels change: own worktree in
   `/home/pat/code/rdna4-hip-kernels`, extend the shared core (`BYLANE` is already a template param —
   do not fork), parity gate, and **build the `.so` inside the serve image** or you get a 0%-GPU wedge
   rather than a build error.
3. **Cut the 750 extra dispatches/step** (finding 7). **Reframed by §8 — the 1974 is a GPU-side
   dispatch count, not a host launch count (1920 of it replays from ONE captured graph), and the
   biggest group is none of the three guessed here: it is 468 router/top-k glue dispatches/step.**
4. **e4m3 uint8 scales instead of folded fp16** (finding 6) — halves NVFP4 scale traffic.
   **NO-GO — see §10.** The byte arithmetic is right (34.5 MB/step, 1.53% of step traffic) but the
   step runs at 23.7% of the HBM ceiling, so removing those bytes is worth **0.30% at bs=1 / 0.14% at
   NREQ=8** — a ceiling, measured with the fold-vs-byte confound isolated. Do not re-open for tok/s.
   The one real payoff is **0.918 GiB/card of VRAM**; revisit only if the KV pool is binding.
5. **Spec defaults:** ship `SPEC_K=7` (+72% over the shipped 16) or default Laguna to no spec until
   verify amortizes. Never ship `SPEC_K=16` — it is exactly one row past the cliff.
   **REVISED by §11: do NOT default Laguna to no spec.** Ship **K=15** graph-captured with
   `MINISGL_SPEC_SAMPLED=1` (2.70x repetitive / 1.35x real code vs plain eager bs=1; loss only on
   free-form prose). `SPEC_K=16` stays banned. K=7 was never re-measured off the prose prompt.
6. **Land the accept-path hygiene** (multi-EOS + gated on-device). It will NOT move tok/s — say so in
   the commit message — but it is the only thing that makes the fast path reachable for a multi-EOS
   reasoning model, which every production Laguna request is. **Drop the gloo->device lockstep idea**;
   measured as noise.

---

## 6. Protocol reminders that bit me

- `docker exec <c> env` shows the **exec shell's** environment, not PID 1's. Read
  `/proc/1/environ` (or the boot log) to check what the engine actually got. I wrongly reported that
  two env vars hadn't taken.
- Host `python3` cannot import torch (`libmpi_cxx.so.40`). Run anything importing torch inside
  `minisgl-rdna4:lean`.
- `sleep` is required in readiness loops — `read -t N </dev/null` returns instantly on EOF and silently
  turns a 5-minute wait into milliseconds.
- lucebox's long-context claims do not survive a bandwidth check (flat 38.6 tok/s from 64K->256K with
  ~1.8K tokens of full-attention KV resident is selective attention, not lossless paging), but their
  **short-context 296 tok/s and 13.7 ms verify are the real target** and are not explained away.

---

## 7. MEASURED: distinct experts per verify batch (2026-07-31) — §5 item 1, RESOLVED

Probe: `quant/kernels.py:_route_stats`, armed by `MINISGL_MOE_ROUTE_STATS`, recording every small-M
MoE call right after `moe_hip.moe_align`. Harness `tools/moe_route_stats.sh` (three legs: plain,
`SPEC_K=7`, `SPEC_K=15`), analysis `tools/moe_route_stats_analyze.py`. Laguna TP=2, EP=0, `CONC=1`,
1200 MoE calls per leg per rank; both ranks and all three legs agree, and the `M=64` prompt-prefill
row is byte-identical across legs — the probe is deterministic.

Run **eager** (`GRAPH_BS=0`) on purpose: routing is a deterministic function of the hidden states, so
it is the same captured or not, and the probe's `.item()` is illegal mid-capture. **This is a routing
statistic, not a timing run.** `top_k=8`, `E=256`, `block_m=16`.

| qlen (M) | pairs = M*8 | distinct experts | `moe_align` blocks | blocks/distinct | vs M=1 |
|---|---|---|---|---|---|
| 1 (plain decode) | 8 | **8.0** | 8.0 | **1.00** | 1.00x |
| 8 (`SPEC_K=7`) | 64 | **36.6** (23-57) | 36.6 | **1.00** | **4.58x** |
| 16 (`SPEC_K=15`) | 128 | **55.5** (33-89) | 55.5 | **1.00** | **6.94x** |
| 64 (prompt prefill) | 512 | 122.3 (99-179) | 129.6 | 1.06 | 15.3x |

Union growth at qlen 16 — experts touched after the first *i* draft rows:

```
i:     1     2     3     4     5     6     7     8     9    10    11    12    13    14    15    16
|U|:  8.0  13.2  18.1  22.1  25.9  29.4  33.0  36.3  39.4  42.3  45.0  47.4  49.6  51.8  53.7  55.5
new:  8.00  5.15  4.99  3.99  3.75  3.50  3.61  3.31  3.11  2.88  2.72  2.34  2.26  2.21  1.86  1.86
```

### Two conclusions, and they point opposite ways

**(a) There is NO kernel bug. `blocks/distinct = 1.00` at every decode and verify M.** `moe_align`
launches exactly one `block_m=16` tile per active expert, and no verify batch ever puts more than 16
rows on one expert, so the grouped GEMM streams the *minimum possible* number of expert slabs. It is
already amortizing 100% of the overlap that exists. Nothing to fix, and the "if overlap is high there
IS a kernel bug" branch of finding 3 is closed.

**(b) Finding 3's PREMISE was wrong, but its verdict survives.**

The analytic bound in `docs/PERF_PUSH_CONTINUANCE.md:112-117`, `E[distinct] = E*(1-(1-top_k/E)^M)`,
assumes INDEPENDENT UNIFORM routing and **overestimates real divergence by ~1.9x**:

| qlen | binomial model | MEASURED |
|---|---:|---:|
| 8 | 57.4 | **36.6** |
| 16 | 102.0 | **55.5** |

Consecutive draft tokens are semantically correlated and route far more coherently than the model
assumes. Any cap on useful K derived from that formula is therefore too tight.

Routing is *not* disjoint: at qlen 16
the 128 (token,expert) pairs land on 55.5 distinct experts, 43% of the disjoint worst case. Even so,
the last draft row still brings **1.86 new experts** — 23% of `top_k` — and the curve is still rising
at row 16. So MoE verify cost grows as `distinct(qlen)`, sublinearly but steeply, and no amount of
overlap saves it. The doc's "14x at M=17" was `6.94x` of inherent divergence times the `M<=16` cliff
(finding 1) putting that batch on the WMMA prefill family — the two effects compose.

### REVISED 2026-07-31 (§11): the conditional resolved — accept-len is 2.6-9.5, NOT flat at 2.55

The caveat that gated this section is settled by the Phase 1a audit (**§11**). The short version:

* **Accept-len is the same quantity in both builds** (`emitted/steps`; all three definitions agree
  to three decimals at `reqs/step=1.0`, verified in all 28 cells). No accounting artifact, and
  **no regression** — `e6ddb502`'s number reproduces on the current build.
* **Accept-len is NOT flat.** It is dominated by the prompt class and by generation length:
  measured **within one boot of one build**, K=15, bs=1, eager, greedy — repetitive **9.492**,
  real code **4.508** (720 tok) / **6.150** (1600 tok), prose **2.912**. §1's `2.58` is the
  *prose-at-384-tokens* cell; it is correct, it is just not the model's acceptance.
* Cause: the DFlash drafter is never conditioned on the prompt, so its 512-token window fills only
  with its own output; accept-len rises with `P` and saturates at the window (**7.366** for real
  code at `P>=512`). See §11.5.

**Recomputed MoE cost per emitted token,** `distinct(qlen) / (8 * accept_len)`, at K=15 (qlen 16,
`distinct(16)/8 = 6.94x`), using measured accept-lens instead of the assumed 2.55:

| workload / regime | accept-len | MoE cost per emitted token |
|---|---|---|
| prose, 384-512 tok (§1's cell) | 2.58-2.91 | **2.4-2.7x worse** |
| real code, 720 tok | 4.32-4.75 | **1.46-1.61x worse** |
| real code, 1600 tok | 6.150 | **1.13x worse** |
| real code, drafter window full (`P>=512`) | 7.366 | **0.94x — spec WINS** |
| repetitive/counting, 720 tok | 9.34-9.52 | **0.73x — spec WINS** |

So the old heading ("spec loses at EVERY K") was a statement about **one prompt class**, not about
the model or about expert divergence. **The divergence measurement stands unchanged** — routing is
still 6.94x at qlen 16, `blocks/distinct = 1.00`, no kernel bug. What changes is the denominator:
spec crosses over on the MoE stack at accept-len ~6.94, which real code reaches once the drafter's
window is full and repetitive workloads reach immediately.

End-to-end this matches: best measured config (graph-captured, `SPEC_SAMPLED=1`, K=15, bs=1) is
**164.83 tok/s repetitive / 82.39 tok/s real code / 48.31 tok/s prose**, against a plain **eager**
bs=1 baseline of **61.0 tok/s** — i.e. **2.70x / 1.35x / 0.79x**. (§1's `74.02` plain was
graph-captured at CONC=2, so the code comparison is indicative, not matched; the matched plain leg
is the first item of §11.6's next measurement.)

`K=7` (qlen 8, `distinct/8 = 4.58x`) was **not** re-measured under the corrected prompt set — the
old K-sweep's flat `~2.55` is now known to be a prose-only artifact, so no K recommendation should
be inherited from it. `SPEC_K=16` remains the one value never to ship (§2 finding 1: qlen 17 falls
off the decode-kernel cliff, and `spec/dflash.py:523` caps drafts at `B-1 = 15` anyway, so K=16
buys zero extra drafts).

**Revised guidance for §5 item 5:** do **not** "default Laguna to no spec". Ship K=15 with graph
capture and `MINISGL_SPEC_SAMPLED=1`; it is a large win on repetitive/structured output, a modest
win on real code, and a loss only on free-form prose. The real lever is the drafter-conditioning
hole in §11.5, not K.

### What it says about lucebox's 296 tok/s

Their target is the same checkpoint family, so the same 256-expert top-8 router and the same union
curve. A 13.7 ms verify at their K would have to pay the same 4.6-6.9x MoE weight traffic. So either
(a) their number is plain decode, not the spec win it is presented as, or (b) a 3090 is far enough
from bandwidth-bound on this expert stack that streaming 6.9x the slabs is nearly free — which is a
claim about their card's headroom, not about a technique we are missing. Nothing here is a kernel
gap on our side. **Do not spend more time chasing spec on Laguna;** the remaining headroom is in
plain decode (§5 items 2-4), where finding 7's 74-vs-90 gap lives.

**Postscript (§9): we now serve Laguna at 301.66 tok/s aggregate at 8 concurrent requests.** That
brackets lucebox's 296 without any spec decode at all, which makes (a) the more likely reading of
their claim — a batched-throughput figure presented as a single-stream one. Worth checking what
concurrency their 296 was measured at before treating it as a target we are missing.

---

## 8. MEASURED: where the 1974 dispatches/step actually go (2026-07-31) — §5 item 3, reframed

Source: `tools/plain_trace.json` (77 MB chrome-trace, rocprofiler-sdk via torch profiler, rank 0 of
the TP=2 plain-decode run). Host-side post-processing only — no GPU, no lease.

**Denominator, established two independent ways.** `hipGraphLaunch` appears exactly **50** times (one
per decode step), and essentially every kernel name's total count is an exact multiple of 50
(12100, 5950, 4050, 4000, 3900, 1950…). 98,715 kernel events / 50 = **1974.3 dispatches/step**,
reproducing finding 7's number. No prefill kernels are present. Laguna has **40 decoder layers**
(10 full-attention + 30 sliding, layer 0 dense + layers 1-39 sparse MoE).

### Correction 1 — the 1974 is a GPU-side count, and it is NOT host launch cost

**1920 of the 1974 replay from a single captured HIP graph.** Only **54.3/step are host-launched**
(26.2 `hipLaunchKernel` + 27.1 memcpy + 1 memset). So finding 7's "1.42 ms host cost from 1974 vs
1224 dispatches" does not follow: what 1920 in-graph dispatches buy you is GPU **inter-kernel gap**,
not CPU time. Any real host-side gap has to come from the ~54 eager dispatches plus Python.

**And this trace cannot price anything.** Median graph-launch-to-graph-launch is **103.7 ms** vs the
13.51 ms real step — a 7.7x inflation, because rocprofiler intercepts per dispatch
(1974 x ~50 us ~= 99 ms, which accounts for essentially the whole gap). Counts and relative GPU-busy
shares from this trace are sound; absolute times and any "ms saved" derived from it are not.

### Correction 2 — the three things §5 item 3 named are small or already done

| §5 item 3 said | measured | verdict |
|---|---|---|
| 107 `copyBuffer`/step | 107.1 — but **80 are in-graph D2D**, 27.1 are eager | real, and #3 below |
| 161 norm launches/step | 161 exactly (81 `rms_norm` + 80 `rms_norm_add`) | real math, not glue |
| SWA ring metadata | **14 dispatches/step = 0.7%** | **dead end — already vectorized** |

The 80 in-graph copies are `RMSNorm.forward_inplace` (`layers/norm.py:33`) blitting the out-of-place
`rms_norm` result back into the q/k slice, twice per layer (`layers/attention.py:58`, `:60`).

### The actual bucket table (per step; %GPU is share of 9792 us GPU-busy)

| #/step | %disp | %GPU | bucket |
|---:|---:|---:|---|
| **432** | 21.9 | 8.6 | **MoE router / top-k glue (torch ops)** |
| 273 | 13.8 | 21.9 | MoE routed compute (act-quant, align, gemm1/2, fill, cast, add) |
| 242 | 12.3 | 33.2 | backbone GEMV bf16 (q,k,v,g,o x40; router x39; dense MLP; lm_head) |
| 198 | 10.0 | 2.3 | dtype-cast elementwise (bf16<->f32) |
| 196 | 9.9 | 8.5 | shared expert |
| 161 | 8.2 | 3.3 | norms |
| 90 | 4.6 | 5.1 | attention core |
| 82 | 4.2 | 11.4 | TP comms (81 `custom_ar`) |
| 80 | 4.1 | 0.9 | `copyBuffer` QK-norm write-back (in graph) |
| 80 | 4.1 | 1.4 | rope (2 launches/layer) |
| 54.3 | 2.8 | 1.8 | host-eager metadata + sampler |
| 40 | 2.0 | 1.1 | `torch.cat([q,k,v])` |
| 40 | 2.0 | 0.5 | attention output-gate `softplus` |

### THE lever: 468 dispatches/step routing a [1, 256] float row

`LagunaSparseBlock._route` (`models/laguna.py:229-237`) is **12 torch kernels per sparse layer** —
`logits.float()`, `bias.float()`, `sigmoid`, `add`, `warpMergeSortTopK`, `bitonicSortKVInPlace`,
`gather`, `sum`, `+1e-20`, `div`, `*2.5`, `.int().contiguous()` — x39 layers = 468/step, averaging
**1.9 us each**, for arithmetic on **256 floats**. That is 21.9% of all dispatches and 8.6% of GPU
busy spent almost entirely on launch latency. One fused `laguna_route` HIP kernel (bf16 logits +
`e_score_correction_bias` -> `topk_weights` fp32 + `topk_ids` int32, one workgroup per token)
collapses 468 -> 39, and `moe_align` can fold into it for 39 more.

### Ranked removable groups (each a fusion of ops that already run back-to-back on the same data)

| saves/step | % | group | mechanism |
|---:|---:|---|---|
| **429** | 21.7 | router/top-k glue | one fused `laguna_route` kernel (`models/laguna.py:229`) |
| 160 | 8.1 | split q/k/v/g proj + the `cat` that rejoins them | one `LinearColParallelMerged` with a 4-way output list — the class already takes one (`layers/linear.py:91-104`), and `LinearQKVMerged` is the precedent |
| 120 | 6.1 | attention output-gate chain | `softplus(gate.float()).to(dtype)` + mul = 4 dispatches/layer; fold into the `flash_decode` epilogue (`models/laguna.py:162`) |
| 80 | 4.1 | QK-norm write-back | give `tail_hip.rms_norm` an `out=`/in-place entry so it writes the q/k slice directly |
| 78 | 4.0 | act-quantize pre-pass + its GEMV | fold act-quant into the `Int4Fp8GemvLoader` prologue — a WLoad policy change, not a new kernel |
| 78 | 4.0 | MoE epilogue (fill + cast + add) | accumulate into a bf16 buffer pre-seeded with the shared-expert output |
| 40 | 2.0 | rope as two launches | after the qkv merge, q and k are adjacent slices with the same `positions` |

**Total 985/step removable -> ~989/step, a 50% cut, with no numerics change.** Not removable: the 82
all-reduces (TP=2, and the two per layer are sequentially dependent), and the 242 backbone GEMV /
90 attention-core / 273 MoE-compute dispatches, which are the actual math.

Also grounded but small: the sampler clones the full-vocab fp32 logits (**401,408 B**) every step for
the EOS-suppress path (`engine/sample.py:113-116`) before the `rows.numel()` guard that would skip
it. Worth fixing; ~0.5% of dispatches.

---

## 9. MEASURED: group-16 by-lane GEMV (2026-07-31) — §5 item 2, LANDED but NOT the win claimed

Kernel change: `rdna4-hip-kernels` `perf/bylane-group16` (`785f6a5` + `668d36b`). `accum_bylane` now
splits its 32-k chunk at the group boundary exactly as `accum` already did, and the minimum scale
group became a **WLoad trait** (`bylane_min_group_v`, default 32; `Int4Fp8GemvLoader` declares 16)
rather than one constant in `select_gemv_tiling` holding every loader to the strictest case.

Also fixed there: `env_int` treated a present-but-EMPTY value as `atoi("")==0`. docker-compose's
`VAR: "${VAR:-}"` passthrough sets the variable to `""`, so plumbing any `*_BYLANE` var through
compose would have silently forced by-lane **off** on every composed serve — including the group-32
shapes where it was already the default and winning.

### Parity — PASS, with a control that gives the gate teeth

`fp8_wmma/local/parity_g2fuse_bylane16.py` (the existing `parity_g2fuse.py` fixes GROUP=32/K=512 and
the fused gemm2 only takes by-lane at K<=320, so its by-lane branch never launches at all).

- **K-on-lanes forced: `max|d| = 0` vs the engine 2-kernel flow**, every dtype and T. The pre-change
  path is byte-identical, so the A/B baseline really is the old behaviour.
- By-lane differs on **1-59 elements of 2048-32768**, relmean ~1e-8. Bound is 2 ulp of the OT partial
  round, NOT fp32 epsilon: the kernel rounds each per-expert partial through `cvt_out<OT>` before the
  fp32 top_k fold, so a 1-ulp fp32 shift can cross an OT boundary. bf16's 8-bit mantissa is why one
  case reads 3.5e-4 while fp16 stays under 1e-4. (My first gate was a flat 1e-4 and "failed" on that
  case — the bound was wrong, not the kernel.)
- **Negative control:** flattening the scales pairwise (`ws[2i] := ws[2i+1]`) makes a whole-chunk fold
  and a half-chunk fold coincide; the two references then differ by **rel 0.375**, so a mis-fold has
  four orders of room to show and does not. The half-fold is provably applied — a tolerance alone
  could not have told a reordering from the exact bug this change exists to avoid.

### The isolated win is REAL but is NOT at the served bs=1 shape

`fp8_wmma/local/bench_g2fuse_bylane.py` (extended to carry group/e2m1 in the shape table). Note the
earlier version of that bench compared by-lane **against itself** — its K-on-lanes leg *unset* the
env, and the default at K<=1024 is by-lane; the tell was a uniform `rel=0.00e+00`, impossible for two
tilings with different fp32 summation orders.

| shape | K-on-lanes | BYLANE | speedup |
|---|---:|---:|---:|
| Laguna g16 e2m1 **M=1 (served bs=1)** | 59.0 us | 69.1 us | **0.85x — a LOSS** |
| Laguna g16 e2m1 M=8 | 207.7 us | 95.1 us | **2.19x** |
| Laguna g16 e2m1 M=16 | 363.7 us | 120.5 us | **3.02x** |
| Qwen g32 int4 M=1 | 41.2 us | 38.0 us | 1.08x |
| Qwen g128 int4 M=8 | 142.4 us | 67.6 us | 2.11x |
| long-K control K=1024 M=1 | 36.9 us | 37.4 us | 0.99x |

**Finding 5's "4.6x at K=256, worth ~0.64 ms/step" is FALSIFIED at the served shape.** That figure
came from the `gemv_decode.h` us/MB table, which is not this kernel: the fused gemm2 has an OUTER
loop over `top_k`, so by-lane does `top_k*K` serially per lane and cannot fill the machine at M=1 —
the crossover note at `moe_kernel.hip:1450` already said so and gives 1.15x at M=1 for group-32.

### e2e: NEUTRAL at bs=1, **+20% at NREQ=8** — the isolated M-crossover reproduces in the serve

Matched A/B, same image, one env var (`tools/bylane16_ab.sh`; graph-captured, true tok/s). Both
provenance witnesses agree on every leg: PID 1's env, and the kernel-side `K=256` under-occupancy
warning (PRESENT on the K-on-lanes legs, absent on the by-lane legs).

| leg | NREQ=1 tok/s | NREQ=8 AGGREGATE tok/s |
|---|---:|---:|
| Laguna BYLANE=0 (pre-change K-on-lanes) | 74.44 | 251.12 |
| Laguna BYLANE on (new default) | 74.37 | **301.66 (+20.1%)** |
| Qwen BYLANE=0 | 95.14 | — |
| Qwen BYLANE on (already the default pre-change) | 95.13 | — |

**A CONC=1 A/B samples the one regime where the tiling does not pay, and would have thrown this
away.** The isolated crossover (0.85x at M=1, 2.19x at M=8) reproduces almost exactly in the serve:
neutral single-stream, +20.1% at 8 concurrent. The fused gemm2's M *is* the decode batch height, so
concurrency is what moves it.

The bs=1 Qwen legs remain the useful control for the neutrality claim: forcing K-on-lanes on the
shape where by-lane was **already enabled** also costs 0.0% at bs=1, so the single-stream flatness is
a property of the serving path (the step is not gemm2-latency-bound there), not of group-16.

Two harness bugs caught by the provenance checks, both worth keeping in mind:
`docker exec <c> tr ... < /proc/1/environ` reads the **host's** PID 1 (the redirect is evaluated by
the outer shell) — use `docker exec <c> sh -c "... < /proc/1/environ"`. And the better witness is
kernel-side: `select_gemv_tiling` warns once per process when it takes K-on-lanes at an under-filled
K, so the `K=256` warning present/absent proves which tiling actually ran.

### SPEC-RELEVANT SIDE EFFECT: verify-side gemm2 is now 3.02x cheaper at qlen 16

The M-scaling above is not just a concurrency story — **M is also the spec verify batch height**, so
this change directly cuts verify cost:

| | M=1 | M=8 | M=16 | M=16 / M=1 |
|---|---:|---:|---:|---:|
| K-on-lanes (before) | 59.0 us | 207.7 | 363.7 | **6.16x** |
| BYLANE (now) | 69.1 us | 95.1 | 120.5 | **1.74x** |

Any spec step-cost number measured before this landed is stale. Re-baseline on this build.

**This is also the correct scoping of §7's `blocks/distinct = 1.00`.** That result is about the
ALIGNED grouped GEMM (gemm1, tiled gemm2), where `moe_align` dedups perfectly. The FUSED
`moe_gemm2_gather_reduce_core` does NOT go through that dedup: its grid is `(N/pb, M)` with an outer
`top_k` loop, so it issues `M * top_k` expert-slab loads — **128 at qlen 16 against only 55.5 distinct
experts, i.e. 2.31x of cross-token dedup left on the floor** (1.75x at qlen 8). Both statements are
true of different kernels. Raising the `M<=2` gate at `quant/kernels.py:452` so verify also takes the
aligned tiled-WMMA scatter path is the change that would claim that 2.31x; BYLANE has already taken a
3.02x from the orthogonal direction (lane occupancy), so measure the residual before assuming it adds.

**Related correction.** The "MoE gemm2 costs 17.76x (24.32 -> 431.83 us)" figure quoted elsewhere is
measured at **M=17**, i.e. OVER the `M<=16` cliff, so it is the fall onto `wmma_tiled_tuned`
(`bm=256`, 93% padding at 17 rows), NOT `gather_reduce`'s M-scaling. Under the cliff at M=16,
`gather_reduce` was 6.16x M=1 and is now 1.74x. Attributing that jump to expert fanout double-counts
the cliff.

### Verdict

**Keep it on, and it is a real win — just not the one finding 5 described, and not at bs=1.**
Correct, bit-exact on the path it replaces, structurally right (a loader trait rather than one
constant holding every loader to the strictest case), and it silences a boot warning that had been
firing unread. It is e2e-neutral single-stream and a 15% isolated regression at M=1, but **+20.1%
aggregate at 8 concurrent requests**, which is the regime a served box actually runs in.

Because the fused gemm2 must stay M-invariant (`layers/minv.py` — prefix-caching, chunked-prefill
and spec-verify all depend on the per-token dot not varying with batch composition), the tiling
cannot be chosen per-M. One choice for all M: the concurrent win dominates.

**Method note worth carrying forward.** The first A/B ran at CONC=1 and read "flat, 0.0%" — a correct
measurement of the wrong regime, and it would have discarded a +20% lever. When a kernel's isolated
speedup is steep in M, the serve A/B has to sweep the concurrency that sets M, not just the default.

---

## 10. MEASURED: NVFP4 e4m3-uint8 scales (§5 item 4) — **NO-GO on performance**, ~0.3% at bs=1

Finding 6's premise ("halve the scale traffic") is arithmetically correct and the byte saving is
real. It does **not** convert to time, because nothing in the Laguna decode step is bandwidth-bound.
Establish this before touching any loader or kernel — the change is invasive (blast radius below)
and the ceiling is a third of a percent.

### The arithmetic, verified against the checkpoint

`poolside/Laguna-XS-2.1-NVFP4` safetensors, summed by tensor class:

| class | on disk | vs weights |
|---|---:|---:|
| `weight_packed` (E2M1, 4-bit) | 15.703 GB | — |
| `weight_scale` (**e4m3 uint8**, group 16) | 1.963 GB | **12.5%** |
| after `fold_nvfp4_scale` -> **fp16** | 3.926 GB | **25.0%** |
| dense bf16 (attn proj + layer-0 MLP + norms) | 3.004 GB | — |

So finding 6 is right: fp16 at group 16 is 0.125 B/param = 25% on top of 0.5 B/param of weights, and
the fold costs ~1.96 GB of extra scale bytes (0.985 GB, **0.918 GiB, per card at TP=2**).

**Per decode step per card at bs=1** (8 of 256 experts x 39 sparse layers, + the shared expert,
+ TP-sharded bf16 backbone, + the replicated bf16 `lm_head`):

| | MB/step |
|---|---:|
| routed experts (weights + fp16 scales) | 306.7 (of which **61.3** is scale) |
| shared expert | 38.3 (7.7 scale) |
| dense bf16 backbone | 1501.9 |
| `lm_head` | 411.0 |
| **total** | **2258.0** |

That is **167 GB/s** against the 706.6 GB/s ceiling (1380 MHz mem OC) = **23.7% of peak**, which is
the engine-wide "umc <= 27% at every batch size" result reproduced from first principles on this model.

fp16 -> e4m3-uint8 removes **34.5 MB/step**, i.e. **1.53% of step traffic**. Even if those bytes
were being fetched at the full DRAM ceiling and their removal were pure time, that is **48.8 us of a
13510 us step = 0.36%**.

### Measured, on the kernels the engine actually dispatches

`group_size 16 vs 32` at identical weight bytes looks like the obvious A/B (g32 has exactly half the
scale bytes) and it reports 1.02-1.08x. **It is a confounded A/B and must not be used.**
`gemv_decode.h accum()/accum_bylane()` take a *different branch* below group 32: the 32-k chunk
straddles two groups, so g16 does TWO `__half2float` loads and splits the dot into `plo`/`phi`.
e4m3-uint8-at-16 keeps that branch verbatim (same two loads, same split) and *adds* a convert. The
g16->g32 delta is mostly fold structure, not bytes.

The clean byte-only probe is **g32 vs g64 vs g128 on the GEMV path**: in the `group_size >= 32`
branch there is exactly ONE `ws[g]` load and ONE fold per 32-k chunk for 32, 64 and 128 alike, and
the decode GEMV has no `GSc` compile-time specialisation (that exists only in the tiled WMMA
`moe_gemm`). So the instruction stream is *identical* and only the scale array size changes. Fit
time vs scale bytes -> us/MB, then price the proposal's 0.0625 B/param.
Served TP=2 shapes: `w13` N=512 K=2048, `w2` N=2048 K=256, E=256, top_k=8, block_m=16.

| kernel | M | g32 us | g64 us | g128 us | slope us/MB | ceiling us/MB | proposal |
|---|---:|---:|---:|---:|---:|---:|---:|
| `gemm1_silu(gemv)` w13 | 1 | 23.5 | 22.8 | 23.1 | **1.01** | 1.42 | 0.53 us -> 0.021 ms/step |
| `gemm_scatter(wmma,splitk4)` w2 | 1 | 26.0 | 25.0 | n/a | 7.18 (INVALID) | 1.42 | <= 0.37 us -> 0.015 ms/step |
| `gemm1_silu(gemv)` w13 | 8 | 108.9 | 109.0 | 108.8 | **0.00** | 1.42 | ~0 |
| `gemm2_gather_reduce` w2 | 8 | 94.0 | 93.3 | 93.1 | **0.56** | 1.42 | 1.13 us -> 0.044 ms/step |
| `gemm1_silu(gemv)` w13 | 16 | 195.2 | 194.1 | 193.6 | **0.30** | 1.42 | 0.080 ms/step |
| `gemm2_gather_reduce` w2 | 16 | 113.3 | 113.0 | 113.1 | **0.09** | 1.42 | 0.013 ms/step |

The `gemm_scatter` row is **excluded**: 7.18 us/MB is 5x the physical DRAM cost of a MB, so that
delta is not bytes — the tiled kernel's `k_sub = group_size/16` inner loop is runtime at `GSc=0`, so
g32/g64 differ in instruction count there too. It is bounded by physics instead (0.262 MB x 1.42).
Every valid slope is **at or below** the ceiling slope, as it must be.

**Per-step ceiling for the proposal, fold-confound removed:**

| regime | saved ms/step | step | **e2e ceiling** |
|---|---:|---:|---:|
| bs=1 (M=1) | 0.021 + <=0.015 + 0.005 (shared) = **0.041** | 13.51 ms | **0.30%** |
| NREQ=8 (M=8) | **0.044** | ~31.9 ms | **0.14%** |
| M=16 | 0.093 | ~31.9 ms | 0.29% |

Independent cross-check: the whole-step byte accounting above gives 48.8 us = 0.36% at bs=1. The two
methods agree.

### Why the NREQ=8 regime does not rescue it (it did for §9)

§9's by-lane lever was neutral at bs=1 and +20.1% at NREQ=8 because its isolated delta was **steep in
M** (0.85x at M=1, 2.19x at M=8 -> 4.39 ms/step isolated, which converted ~1:1 to the 5.4 ms/step the
serve actually gained). This lever is the opposite: the byte slope *falls* with M (1.01 -> 0.00 -> 0.30
us/MB on gemm1), because at larger M each expert slab's scales are amortised over more rows while the
weights are not. 0.044 ms on a ~31.9 ms step is 0.14%. There is no concurrency at which this pays.

### **GO/NO-GO: NO-GO for throughput.**

0.3% at bs=1 and 0.14% at NREQ=8 are *ceilings* that additionally ignore the e4m3->f32 convert the
change adds and the unchanged group-16 double-fold. The measured e2e conversion factor for isolated
MoE-kernel savings at bs=1 in this engine is ~0 (§9: a 0.4 ms/step isolated delta moved tok/s 0.09%).
Do not spend the blast radius below on it.

### What IS real, and is not a bandwidth argument: **0.918 GiB/card of VRAM**

Not folding to fp16 keeps 1.963 GB of scales as uint8 -> **0.985 GB = 0.918 GiB freed per card at
TP=2**. On a 16 GiB card serving Laguna at `MEM_RATIO=0.96` with ~10.4 GiB of weights that is roughly
a fifth of the free pool, ~96k more tokens of full-attention fp8 KV (10 full layers x 4 kv-heads/card
x 128 x 2 x 1 B = 10240 B/token). If KV pool is ever the binding constraint on Laguna concurrency or
context length, **that** is the reason to do this — not tok/s. Measure the pool first.

### Numerics of the current fold (checked, and it is fine)

52.4M folded scale elements sampled from 800 matrices: folded |value| spans 1.94e-3 .. 1.38e-1, so
**zero** subnormals, **zero** flush-to-zero, **zero** overflow in fp16 — the fold is safe.
It is however **not exact**, contrary to `nvfp4.py`'s docstring: representing `e4m3/global` in fp16
costs mean **1.63e-4** / max **4.49e-4** relative (fp16 ulp/2 = 4.88e-4). That is ~1000x below the
E2M1 weight-quantisation noise floor (1-2 mantissa bits), so it is harmless — but the docstring's
"This fold is exact" should read "exact to fp16 rounding, ~1.6e-4 rel". No quality reason to change.

### Blast radius, if someone revives this for the VRAM

- **Kernels** (`fp8_wmma` only — no other package consumes an fp16 group scale): the decode GEMV core
  is already `WScaleT`-templated and already has a non-`__half` sibling (`WScaleT = float`, RXF), so
  a `WScaleT = uint8_t` loader with an `e4m3_to_f32` on read is a genuine WLoad policy there. Every
  *other* path hardcodes it: ~128 `const __half* w_scales` signature / `data_ptr<at::Half>()` sites
  across 12 files (`moe_kernel.hip` 11, `w4a8_fp8_wmma_kernel.hip` 35, `w8a8_moe_kernel.hip` 11,
  `gemm_tiled.h`, `moe_gemm_flag.h`, `moe_gemm1_silu_flag.h`, + the `*_hip` twins). Prefill (M>32)
  and the dense NVFP4 path all go through those, so a decode-only change would leave two encodings
  of the same weight resident — i.e. no VRAM saving at all. It is all-or-nothing.
- **Op schemas**: `scales` dtype is part of every `torch.ops` signature and every `_register_fake`.
- **Engine**: `nvfp4.py` stops folding and must carry the per-tensor global through the leaf->merge
  path that `fold_nvfp4_scale` exists to avoid. Checked: in this checkpoint `gate_proj` and
  `up_proj` share a global per expert but experts differ from each other, so post-stack the global is
  a per-expert (in general per-output-channel) fp32 vector, not a scalar. That is tractable via the
  existing `wscale_epi()` per-channel hook at 4 B x N per matrix (2 KB vs 128 KB of scales), but it
  touches the gate-up merge, the expert stack and the GDN in_proj concat.

### Method note

Two things nearly produced a wrong answer here. (1) A first pass at 200 iters/no min-of-N read the
g16->g32 delta as up to 1.21x; min-of-5 collapsed it to 1.08x — small-kernel A/Bs on a shared box need
min-of-N. (2) The obvious g16-vs-g32 A/B is confounded by a *branch*, not just bytes, and it
over-reports the lever by ~3x. When a knob changes two things, find the pair of settings that changes
only one — here g32/g64/g128, where the instruction stream is provably identical.

---

## 11. Phase 1a — the acceptance audit (MEASURED)

Gating task from §7's caveat: is the "accept-len 8.1 (`e6ddb502`, Jul 23) -> 2.58 (§1, Jul 31)"
collapse real? **Answer: there is no regression to bisect.** The two numbers are the same quantity
and both reproduce today, on the current build, within noise. They were measured on **different
prompt classes**, and accept-len on this drafter spans **2.6 -> 9.5 with the build held constant**.

### 11.1 Definition verdict — COMMENSURABLE (one basis correction to `e6ddb502`)

The `/metrics` gauge formula *did* change between the two builds, and it does **not** matter:

| build | `minisgl_spec_mean_accept_len` |
|---|---|
| `44dfb97f` (`e6ddb502`'s descendant) | `metrics.py:253-255` — `1.0 + accepted/steps` |
| `4a031cc3` (current) | `metrics.py:262-264` — `emitted/(emitted-accepted)` (changed by `63aac736`, Jul 24) |

**Neither headline number came from that gauge.** `2.58` is `spec_k_sweep.sh:47-48`,
`accept-len = em/st` on raw counter deltas. `8.1` is the `[spec]` debug line, archived verbatim at
`minisgl-rdna4-swaverify/tools/swa_dflash.spec.log:52-53`:

```
[spec] mean accept-len=8.09 over 300 reqs
[spec] step=300 accept_rate=0.55 draft_accepted=2427/4403 emitted/step=9.09 (reqs/step=1.0)
```

`2427/300 = 8.09` exactly, so **`8.1` is the accepted-drafts-only basis** and its `emitted/steps`
counterpart *in the same log line* is **9.09**. The counter-increment code
(`scheduler.py:3792-3796` current == `:3729-3733` swaverify) is byte-identical in both trees.

At `reqs/step == 1.0` all three definitions collapse to the same value
(`emitted = accepted + steps`). The harness prints all three per cell and **they agree to three
decimals in every one of the 28 cells measured**; `reqs/step` printed `1.0000` everywhere (single
exception 0.9971, EOS truncation). No cell is batch-inflated. **The accounting-artifact hypothesis
is dead** — but so is the premise it was protecting: the honest like-for-like statement was always
`9.09 -> 2.58`, and 9.09 is reproducible today.

### 11.2 The matrix — 3 builds/configs x 3 prompt classes, one harness

Conditions, identical in every cell: K=15, bs=1 (`--max-running-requests 1`), TP=2, **eager**
(`--cuda-graph-max-bs 0`), greedy `temperature 0.0`, `--cache-type naive`, bf16 KV
(`MINISGL_KV_FP8=0`), `--memory-ratio 0.90`, short context (64-99 prompt tokens), non-streaming.
Prompt A = `e6ddb502`'s own four counting/repetition strings **verbatim**
(`tools/swa_dflash_lossless.sh:56-63`, 720 tok). Prompt A2 = prose (includes `spec_k_sweep.sh`'s
B-tree essay, 512 tok). Prompt B = real code (write a templated C++ B-tree, 720 tok).

| leg | build | page_size | prompt | accept-len (all 3 defs) | TRUE tok/s | steps | emitted | accepted | drafted | ms/step |
|---|---|---|---|---|---|---|---|---|---|---|
| A | `4a031cc3` current | 1 | A repetitive | **9.492** | 146.96 | 303 | 2876 | 2573 | 4448 | 64.67 |
| A | current | 1 | A2 prose | **2.912** | 47.12 | 351 | 1022 | 671 | 5196 | 61.91 |
| A | current | 1 | B real code | **4.508** | 70.09 | 319 | 1438 | 1119 | 4719 | 64.40 |
| B | `44dfb97f` swaverify | 1 (forced) | A repetitive | 9.186 | 105.11 | 307 | 2820 | 2513 | 4500 | 87.49 |
| B | swaverify | 1 (forced) | A2 prose | 3.329 | 37.25 | 307 | 1022 | 715 | 4506 | 89.53 |
| B | swaverify | 1 (forced) | B real code | 4.746 | 52.98 | 303 | 1438 | 1135 | 4491 | 89.70 |
| C | current, `MINISGL_SPEC_MHA_PAGED=1` | 16 | A repetitive | 9.399 | 143.12 | 306 | 2876 | 2570 | 4490 | 65.76 |
| C | current, paged | 16 | A2 prose | 2.879 | 45.90 | 355 | 1022 | 667 | 5256 | 62.84 |
| C | current, paged | 16 | B real code | 4.318 | 66.22 | 333 | 1438 | 1105 | 4928 | 65.30 |

Read it two ways:

* **Build vs build (A vs B, page_size matched at 1): +3.3% / -12.5% / -5.0%.** All inside the
  same-build boot-to-boot band measured below (§11.3). **No build effect is detectable**; any build
  effect is bounded near +/-15%, two orders of magnitude below the claimed 3x.
* **Prompt class, inside ONE boot of ONE build (leg A, minutes apart, same process, same config):
  9.492 -> 4.508 -> 2.912.** A **3.26x spread from the prompt alone.** That is the whole of the
  claimed "8.1 -> 2.58" collapse, reproduced with the code held constant.
* `e6ddb502`'s 9.09 emitted/steps (8.09 accepted-only) is **exceeded on the current build**: 9.492
  (8.492 accepted-only), reproduced 4/4 across independent boots with `emitted = 2876` every time.
* §1's `2.58` is reproduced as a **prose-at-384-tokens** number (prose cells 2.621-3.125).

**Provenance, asserted from inside each container per leg** (this is not new-vs-itself):
distinct `/engine/python` digests `b5368ebc...` (current) vs `3febf440...` (swaverify); distinct
`scheduler.py` md5 `c1c99c9b` vs `c9b4bb8d`; `git rev-parse HEAD` = `4a031cc3` vs `44dfb97f` with
`git status --porcelain` showing no source edits; build discriminator
`grep -c MINISGL_SPEC_MHA_PAGED engine.py` = **4 vs 0**; the gauge source line printed per leg.
Image is a **proven constant** across all legs (`/opt/minisgl` `7a465c14`, `/opt/kernels`
`886f4202`, torch 2.14.0.dev+rocm7.2). The real argv from `/proc/<pid>/cmdline` is
**character-identical** across legs A and B, cross-checked against each engine's own parsed
`ServerArgs` echo. Both A and B logged `spec-decode (MHA): overriding page_size -> 1`; C logged
`keeping page_size=16 (page-aware rollback)`.

**Neutralisation was mandatory and is why this is comparable at all:** swaverify has *no*
`tools/serve.sh` and its `laguna-dflash` compose service uses a different knob namespace
(`MINISGL_SPEC_K`, `MINISGL_CUDA_GRAPH_MAX_BS`, ...), so passing `SPEC_K=15` there would have
silently served K=16. `tools/spec_truth.sh` hand-writes the argv and runs under the `run` compose
profile, binding `127.0.0.1:21955` **inside** the container — port 1919 is never touched, so two
legs cannot collide.

### 11.3 Knob sweep (current build, five more serve legs)

Same cell as leg A (repetitive/counting is the only class tight enough to adjudicate a knob).

| knob | verdict | evidence |
|---|---|---|
| `MINISGL_SPEC_PREFILL_SEED=1` | **INERT BY CONSTRUCTION — 0.00%** | `spec/base.py:65 supports_prefill_seed = False`; only `spec/draft_model.py:56` and `spec/mtp.py:44` override it; `scheduler.py:347` ANDs on it, so `_spec_prefill_seeded` never runs for DFlash. Runtime: flag present in the engine's own environ, `"prompt-prefill draft-KV seed ENABLED"` logged **zero** times, and the counting cell reproduced leg A **exactly** (steps 303, emitted 2876, accepted 2573). |
| `MINISGL_SPEC_MHA_PAGED` (ps 16 vs 1) | **acceptance-neutral, slightly negative** | 9.399 vs 9.492, 2.879 vs 2.912, 4.318 vs 4.508 — 1-4%, consistently negative, never a collapse. Exonerated as the suspect; safe as a default on acceptance grounds. |
| `MINISGL_SPEC_SAMPLED` @ temp 0 | **inert** | `_req_spec_ok` (`scheduler.py:3143-3150`) lets greedy reqs speculate either way; `any_sampled` (`:3413-3415`) needs a non-greedy req. Measured 9.368 (=1) vs 9.523 (=0), both inside the eager band. |
| `MINISGL_SPEC_SAMPLED` @ temp 0.7 | **it is the on/off switch for spec existing at all** | With `=0`: `spec steps = 0, emitted = 0, drafted = 0` — every non-greedy request falls through to plain decode (61.13 / 60.87 tok/s). With `=1`: counting 9.523 @ 164.41, real code 3.951 @ 70.06. Must stay default-ON (`4c504316`). |
| `--cuda-graph-max-bs 8` | **large throughput win** | counting 56.6-56.7 vs 64.7-65.1 ms/step (-13%), 164.8-165.2 vs 144.5-147.0 tok/s (+12.5%); real code 82.4-83.5 vs 70.1-74.0 tok/s (+14%). Accept-len 9.338 vs eager 9.469 mean (-1.4%). |

**Noise floor, measured not assumed.** Legs S2/S3/S4 toggle knobs that are *provably inert* on the
greedy cells, so their spread against leg A **is** the four-boot run-to-run band:

```
counting  9.492 / 9.492 / 9.523 / 9.368   (range 1.7%;  emitted = 2876 in all four)
prose     2.912 / 3.125 / 2.912 / 2.621   (range 19%)
code      4.508 / 4.624 / 4.746 / 4.318   (range 9.9%)
```

Nothing smaller than ~10-20% is resolvable on prose or code at this sample size.

**Best measured config:** `--cuda-graph-max-bs 8` + `MINISGL_SPEC_SAMPLED=1` + `PREFILL_SEED` unset
+ `MINISGL_SPEC_MHA_PAGED=0`, K=15, bs=1, TP=2 — counting 9.338 @ **164.83 tok/s**, real code 4.624
@ **82.39 tok/s**, prose 2.682 @ 48.31 tok/s. Against the plain **eager bs=1** baseline of
**61.0 tok/s** (a free by-product of the `SAMPLED=0` temp-0.7 cells, and the like-for-like plain
number §1 never had): **2.70x repetitive, 1.35x real code, 0.79x prose.**

### 11.4 Is verify-graph capture lossless for acceptance? — YES on acceptance; output determinism differs

* **Acceptance: within noise.** Graph 9.338 (two independent boots, raw counters *bit-identical*)
  vs an eager minimum of 9.368 — a 0.3% margin the 1.7% eager band cannot adjudicate. Graph buys
  +12-14% true tok/s. **Do not turn graph capture off on acceptance grounds.**
* **Output: reproducibly different on one knife-edge prompt.** `e6ddb502` prompt #2
  (multiplication table for 7) emits 663 tokens / `finish=stop` under graph (2/2 legs) vs 720 /
  `finish=length` under eager (4/4 legs), at temperature 0.
* **But this is NOT a spec-verify defect.** The swaverify build produced the *same* 663/stop while
  running **eager**. It is a knife-edge logit tie flipped by any change of numeric path (static
  buffers, tile/kernel selection under capture), which the two builds already resolve differently
  from each other. Reported as a "spec correctness bug" it would send the next agent to the wrong
  file. Not isolated further; the next step if anyone cares is a logits-level eager-vs-replay diff
  on one verify step, not another e2e leg.

### 11.5 What actually drives the 3.26x prompt spread: the drafter never sees the prompt

One extra instrumented leg (`tools/spec_dflash_divergence.sh`, 1600-token real-code generation,
current build, eager, K=15, bs=1, `MINISGL_SPEC_DEBUG=2`):

* **Divergence position is a smooth decay, not a spike.** `n` = drafts accepted before the first
  mismatch, over 260 verify steps: `n=0` **11.5%**, `n=15` (full accept) **7.7%**, monotone decay
  between. That rules out a conditioning-*wiring* fault (would pin `n=0` near 100%) and a
  position/mask off-by-one (would spike at a fixed `n`). It is genuine per-token disagreement at a
  per-position conditional acceptance of ~0.86.
* **Accept-len is a monotone function of `P` = target hidden states already in the drafter's aux
  prefix, and it saturates exactly at the drafter's 512 sliding window:**

  | `P` bucket | real code | repetitive |
  |---|---|---|
  | <32 | 3.556 | 5.833 |
  | 32-64 | 3.545 | 7.250 |
  | 64-128 | 4.062 | 6.600 |
  | 128-256 | 3.529 | 7.333 |
  | 256-512 | 6.143 | 8.258 |
  | **512-1024** | **7.366** | **9.660** |
  | >1024 | 7.299 | 7.948 |

* **Causal control:** capping the drafter prefix at 8 positions (`MINISGL_DFLASH_CTX_WINDOW=8`)
  collapses real code **6.150 -> 3.667** accept-len and **92.1 -> 55.2 tok/s**, and flattens the
  `P`-dependence to 2.4-4.1 across every bucket. The `P<32` bucket is identical in both legs
  (3.556) — a clean internal control, since below `P=8` the two configurations are the same thing.
* **Mechanism, read-only and confirmed:** `_spec_aux_hidden` is built append-only from **accepted
  generated positions** (`scheduler.py:3682-3701`) and consumed at `spec/dflash.py:535-564`
  (`P = aux.shape[1]`, `ctx_start = req.cached_len - P`). The only site that could seed it from the
  prompt (`scheduler.py:2270-2276`) is dead code for DFlash (see the `PREFILL_SEED` row above).
  So the drafter's context at generated position `P` is exactly `min(P, 512)` tokens **of its own
  output** — every request starts blind and stays starved for ~512 tokens. High acceptance on
  prompts that carry no information (counting), poor acceptance where the prompt carries everything
  (code).
* **Basis reconciliation with the reference:** their "81% acceptance" is a per-position conditional
  rate -> accept-len 5.08 at K=15. Our windowed 7.366 implies ~0.865; our whole-720-token-request
  4.5 implies ~0.78. The gap was never 3x — it is ~3 points of conditional acceptance, entirely
  explained by window occupancy.

**The fix is code, not env:** implement `seed_prefill` on `DFlashProposer` with
`supports_prefill_seed = True` (the receiving code at `scheduler.py:2270-2276` already writes the
whole prompt aux), and relax the `req.cached_len == 0` filter at `scheduler.py:2247` so
prefix-cache hits are still seeded. Payoff scales with prompt length: negligible for a 95-token
prompt, decisive for the 1-4k-token prompts of real agentic-coding traffic, where the drafter would
start at ~7.4 instead of ~3.5.

### 11.6 Bottom line (post adversarial audit)

**NOT A REGRESSION. NOT A METRIC ARTIFACT. It is a comparison artifact**: two measurements of the
same quantity taken on different workloads under different serve configs — `e6ddb502`'s counting
prompts at 720 tok / K=16 / naive / eager / bf16 KV, versus §1's prose essay at 384 tok / graph /
radix / fp8 KV / `mem 0.96`. Both reproduce today on the current build. **Do not open a commit
bisect** — the spec accept/verify path in `scheduler.py` is byte-identical between the two builds
(the whole `44dfb97f -> ea2669f7` scheduler diff is host-profiler + prefix-cache counters).

Claims deliberately **downgraded** by the audit, so nobody inherits them as facts:

* "Current build is +3.3%/-12.5%/-5.0% vs swaverify" -> **not resolvable**; state it as a
  +/-15% bound.
* "Verify-graph capture is a fidelity bug" -> **overstated**; see §11.4.
* "Real code 6.150 is past the 5.1 target" -> **ill-posed as stated.** Accept-len on this drafter
  is a function of *generation length* (code: 4.3-4.7 at 720 tok, 6.150 at 1600 tok). **No bare
  accept-len number in this document is interpretable without its `max_tokens`.**

**What is genuinely UNDETERMINED — the shipped configuration was never measured for acceptance.**
Every acceptance number here and in §1's spec rows was taken at `MINISGL_KV_FP8=0`,
`--cache-type naive`, `--max-running-requests 1`, page_size=1 (except leg C). Production compose
defaults are `MINISGL_KV_FP8=1` (`docker-compose.yml:56`), radix, `MINISGL_SPEC_MHA_PAGED=1`,
`CONC=4`, graph-captured. fp8-quantised target K/V shifts both the target's argmax and the captured
aux the drafter is conditioned on. bf16 KV was *mandatory* to keep the cross-build comparison valid
(swaverify has no fp8 SWA descale), so this axis was correctly held constant — and consequently
**served acceptance is unknown and could be below every number above.**

**The one next measurement** (settles the coverage hole and tests the root cause's one falsifiable
prediction in the same leg): current build, production config (`MINISGL_KV_FP8=1`, radix,
`MINISGL_SPEC_MHA_PAGED=1`, `--cuda-graph-max-bs 8`, K=15, bs=1, TP=2, greedy), a **2-4k-token real
code prompt** at `max_tokens=1600`, run twice in one boot (the second exercises the radix
prefix-cache-hit path), plus the identical prompt with `--spec-algorithm none` for the matched plain
baseline. Report accept-len bucketed by `P` exactly as `spec_dflash_divergence.sh` already does.
*Prediction:* if the drafter is genuinely blind to the prompt, the `P<64` bucket stays ~3.5 despite
a 4k-token prompt; if it jumps, the starvation diagnosis is dead.

### 11.8 MEASURED: the prompt-prefill seed TAIL SWEEP — the tail is not the lever, the PROMPT LENGTH is

Closes the sweep `341c4df0` deferred (`_SEED_TAIL_DEFAULT = 64` was provisional). `tail ∈ {0, 32, 64,
128, 256, 528}` x {LONG 3571-token code prompt @ 1600 tok, SHORT 95-token instruction @ 384 tok},
plus a matched `--spec-algorithm none` PLAIN leg, **two full independent replicates** (14 boots).
`MINISGL_DFLASH_SEED_TAIL=0` disables seeding on the same binary, so unlike `5bd8d5a7` this is one
tree and one image; verified equivalent to the pre-change build (warm md5 `282b24cb` byte-identical
to that bank's baseline, LONG accept-len 4.675 vs its 4.635). Config, image and prompt file are
identical to `5bd8d5a7`. Full numbers, provenance and the void leg:
`tools/spec_seed_tail_sweep_results.txt`.

**The table** (tok/s = `usage.completion_tokens`/wall, non-streaming; 4 samples per cell):

| tail | LONG accept-len | LONG tok/s | LONG vs PLAIN | SHORT accept-len | SHORT tok/s | SHORT vs PLAIN | combined |
|---|---|---|---|---|---|---|---|
| PLAIN | 1.000 | 65.59 | +0.0% | 1.000 | 79.32 | +0.0% | 72.13 |
| **0 (off)** | 4.479 | **75.39** | **+14.9%** | 4.442 | 84.32 | +6.3% | 79.73 |
| 32 | 4.019 | 67.50 | +2.9% | 5.394 | 101.07 | +27.4% | 82.60 |
| **64** | 4.384 | 73.55 | +12.1% | **5.394** | **101.11** | **+27.5%** | **86.23** |
| 128 | 4.396 | 73.74 | +12.4% | 5.394 | 101.13 | +27.5% | 86.36 |
| 256 | 4.056 | 68.23 | +4.0% | 5.394 | 101.18 | +27.6% | 83.08 |
| 528 | 4.101 | 69.00 | +5.2% | 5.394 | 101.16 | +27.5% | 83.54 |

`ms/step` is flat at 60.6-61.6 (LONG) / 52.0-54.6 (SHORT) for **every** tail including 0 — seeding
costs nothing per step, it only changes draft quality. "combined" = geometric mean of the two regime
means.

**The tail size is not the lever, and there are two independent proofs.**

1. **SHORT is bit-identical for every tail in {32,…,528}** — same completion md5 `f6a4727c`, same 71
   steps, same 5.394, same first draft chain, in both replicates. Above 32 trailing positions the
   knob does nothing on a 95-token prompt. It is *not* an inert code path: at 3571 tokens the first
   draft chain differs between every tail from draft position 4 onward.
2. **The LONG differences are the greedy content lottery.** Restrict accept-len to `P<64`, the only
   region where the seed is still inside the drafter's fixed 512-key window:

   | | LONG `P<64` (rep1 / rep2) | SHORT `P<64` (both reps) |
   |---|---|---|
   | tail 0 (off) | 2.462 / 2.393 | 3.000 |
   | 32 | 2.241 / 2.031 | 4.176 |
   | 64 | 2.407 / 2.345 | 4.176 |
   | 128 | 2.615 / 2.241 | 4.176 |
   | 256 | 2.826 / 2.276 | 4.176 |
   | 528 | 2.241 / 2.481 | 4.176 |

   On the LONG prompt the seed produces **no lift at all**, in either replicate, in the only place it
   can act. On the SHORT prompt the same statistic moves **3.000 → 4.176 (+39.2%)**, identically in 4
   boots x 5 tails. Meanwhile whole-request LONG accept-len spans **3.919-4.675 at tail=0 alone**
   across 3 independent generations — a 19% band, wider than any tail-vs-tail difference in the
   table. Everything past `P=512` is the drafter conditioned on its own output, which is where the
   LONG numbers actually move.

**The crossover is real and it sits at the drafter's window.** Truncating the code prompt to
intermediate lengths is **VOID** as a length measurement (each truncation asks for a different
continuation; accept-len swung 3.0-8.0 *within one leg*). `tools/spec_seed_padlen.sh` holds the task
fixed instead — the same 95-token instruction is always last, preceded by N tokens of filler:

| prompt_tok | PLAIN tok/s | tail 0 acc / tok/s | tail 64 acc / tok/s | seed Δ accept-len |
|---|---|---|---|---|
| 95 | 78.29 | 4.352 / 81.62 | 5.394 / 99.30 | **+23.9%** |
| 223 | 75.53 | 3.792 / 69.43 | 4.402 / 78.91 | +16.1% |
| 351 | 73.46 | 4.118 / 73.15 | 4.560 / 78.89 | +10.7% |
| 607 | 72.57 | 4.256 / 73.74 | 3.648 / 62.89 | -14.3% |
| 1119 | 69.38 | 4.163 / 69.03 | 4.163 / 68.17 | +0.0% |
| 2143 | 65.10 | 4.256 / 67.88 | 4.209 / 66.85 | -1.1% |
| 3495 | 57.78 | 4.506 / 65.47 | 4.352 / 63.42 | -3.4% |

The advantage decays monotonically 95 → 223 → 351 and is gone by ~600 tokens; 1119-3495 sit at 0 to
-3%. **Crossover: between 351 and 607 prompt tokens** — the drafter's own 512-key sliding window, the
only length scale in the system that could put it there. `n=1` per cell, so the 607 cell's -14.3% is
scatter around zero, not a dip.

**This falsifies `341c4df0`'s own explanation.** That commit blamed EVICTION ("a window-sized seed
evicts the model's recent output"). A 32-token seed can evict at most 32 of 512 keys, yet tail=32 is
*not* better than tail=528 on LONG (4.019 vs 4.101, both under tail=0's 4.479) and is bit-identical
to it on SHORT. Size is not the mechanism. And `5bd8d5a7`'s headline -15.6% at tail=528 does not
survive replication as a *seed* effect: re-measured it is -8.4% against a tail=0 that itself moves 19%.

**Chosen default: `_SEED_TAIL_DEFAULT = 64`, unchanged — now measured, no longer provisional.**
Decision rule: must not regress LONG below plain, then maximise across both.

* Gate — every LONG leg above its **own boot's** matched plain leg, 4/4: passed by tail 0 and tail 64
  **only**. 32 fails rep1 (62.54 vs 64.19); 128 fails rep2 (62.41 vs 64.16); 256 fails rep2 (64.02 vs
  64.16); 528 fails rep2 (64.12 vs 64.16).
* Of the two survivors: LONG 75.39 (off) vs 73.55 (64) — off by 2.5%, **inside** the 19% lottery
  band. SHORT 84.32 vs 101.11 — 64 by 19.9%, reproducible to <1%. Combined 79.73 vs 86.23.
* 128 ties 64 on the combined score (86.36) but fails the gate and has 4x the spread (62.41-86.11 vs
  70.02-77.26).

**No single tail wins both regimes; the trade-off is stated, not hidden.** Choosing 64 over
seeding-off costs the long-prompt regime ~2.5% tok/s (unresolvable against its own noise) to buy the
short-prompt regime 19.9% (reproducible). The opposite reading — "seeding off is simply best at long
prompts, ship it off" — is **not** what the data says: off is 0-3% better there, under the noise
floor. The correct statement is that **the tail number is irrelevant** and the feature earns its keep
for prompts up to ~500 tokens and is free-but-pointless above that.

**Next, and it is the measured follow-up, not a guess:** gate the seed on prompt length — seed when
`prompt_len <= sliding_window` (512), skip above. The crossover table is the whole justification; it
keeps +24%/+16%/+11% at 95/223/351 and gives back the 0-3% the long regime pays. That is a scheduler
predicate, not a new tail value, so this sweep cannot pick it — it needs its own A/B.

### 11.7 Artifacts

- `tools/spec_seed_tail_sweep.sh` / `_driver.sh` — the §11.8 tail sweep (one boot per tail, seed
  provenance asserted per leg from the scheduler's ENABLED line, warm/CODE1/CODE2/SHORT1/SHORT2).
- `tools/spec_seed_padlen.sh` / `_driver.sh` — the held-task prompt-length crossover (§11.8).
- `tools/spec_seed_crossover.sh` — the **VOID** truncation-based length sweep; kept so nobody re-runs
  it (each truncation changes the requested continuation and the lottery swamps the effect).
- `tools/seedtail_parse.py` / `tools/seedtail_window_analysis.py` — the tail x regime table and the
  `P<512`/`P<128`/`P<64` restricted accept-len that separates the seed from the content lottery.
- `tools/spec_seed_tail_sweep_results.txt` — every raw cell, provenance, and the void leg.
- `tools/spec_truth.sh` — the 3-leg matrix harness (hand-written argv, three accept-len definitions
  per cell, `reqs/step` batch-inflation guard with an explicit `*** VOID ***` branch, provenance
  block printed from inside the container).
- `tools/spec_sweep.sh` — the knob sweep (adds `PREFILL_SEED` / `SAMPLED` / `GRAPHBS` legs and
  temperature-0.7 cells).
- `tools/spec_dflash_divergence.sh` — divergence-position histogram + accept-len bucketed by `P`.
- `tools/spec_accept_audit_results.txt` — every cell's raw counters, as printed by the harnesses.
- All three `.sh` must be run **inside** the image under `gpu-lease -n 2` via the `run` compose
  profile; they bind `127.0.0.1:21955` in-container and deliberately do not lease themselves.
- The isolation worktree `/home/pat/code/minisgl-rdna4-spectruth` (branch `task/spec-laguna-truth`
  @ `4a031cc3`) is left in place; remove with `git worktree remove` when the follow-up lands.
