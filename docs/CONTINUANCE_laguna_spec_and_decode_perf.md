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
| Low acceptance / need a better drafter | **NO** | accept-len flat across all K; cost is per-verify-row |

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
5. **Spec defaults:** ship `SPEC_K=7` (+72% over the shipped 16) or default Laguna to no spec until
   verify amortizes. Never ship `SPEC_K=16` — it is exactly one row past the cliff.
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

**(b) Finding 3's PREMISE was wrong, but its verdict survives.** Routing is *not* disjoint: at qlen 16
the 128 (token,expert) pairs land on 55.5 distinct experts, 43% of the disjoint worst case. Even so,
the last draft row still brings **1.86 new experts** — 23% of `top_k` — and the curve is still rising
at row 16. So MoE verify cost grows as `distinct(qlen)`, sublinearly but steeply, and no amount of
overlap saves it. The doc's "14x at M=17" was `6.94x` of inherent divergence times the `M<=16` cliff
(finding 1) putting that batch on the WMMA prefill family — the two effects compose.

### This kills spec decode on Laguna at EVERY K, not just K=16

MoE cost per *emitted* token, relative to plain decode, is `distinct(qlen) / (8 * accept_len)`, and
accept-len is flat at ~2.55 (finding 2):

| K | qlen | distinct/8 | accept-len | MoE cost per emitted token |
|---|---|---|---|---|
| 15 | 16 | 6.94x | 2.58 | **2.69x worse** |
| 7 | 8 | 4.58x | 2.55 | **1.80x worse** |
| 3 | 4 | 2.76x | <=2.55 | **>=1.08x worse** |
| 2 | 3 | 2.26x | <=2.4 (bounded by K+1) | ~0.94x — break-even at best |

Spec only stops losing on the MoE stack alone at K≈2, where acceptance is bounded below the cost —
**and that ignores the ~9 ms/step drafter (finding 4), which is by itself 66% of a 13.51 ms plain
step.** There is no K at which DFlash spec wins on this model on this box. §5 item 5 should read
"default Laguna to no spec", not "ship K=7".

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
