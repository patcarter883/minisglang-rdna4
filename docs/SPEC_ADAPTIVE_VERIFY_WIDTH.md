# Adaptive spec-decode VERIFY width — design, the M cliff, and what "lossless" does and does not mean

Companion to `docs/SPEC_PROPOSE_GRAPH_CAPTURE.md`. Code: `python/minisgl/spec/width.py` (pure, no
torch), the width-keyed verify capture in `python/minisgl/engine/graph.py`, and the call sites in
`Scheduler._spec_decode_step`. Host-side gate: `tools/verify_width_unit.py` (no GPU).

---

## 1. What it does

A speculative step verifies `width + 1` query rows per sequence: the confirmed token plus `width`
drafted tokens. Before this, `width` was static at `--spec-num-draft`. Now:

* `verify_width_ladder(K, quant)` defines a **small ladder** of widths (K=15 → `[3, 7, 15]`,
  K=6 → `[3, 4, 6]`, K=4 → `[2, 3, 4]`), and **every rung is CAPTURED** —
  `GraphRunner._verify` is keyed by qlen, so narrowing never falls off the graph.
* `AdaptiveVerifyWidth` keeps a per-uid EMA of accepted drafts and returns `mean + 1` rounded UP to a
  rung. The `+1` is load-bearing: the statistic is **censored** (a step that accepts all `W` rows
  says acceptance was *at least* `W`), so without one exploratory row the controller could narrow
  once and never climb back. `tools/verify_width_unit.py` gates exactly that: a request accepting ~2
  settles at 3, a saturating one stays at 15, and a narrowed one that starts accepting more climbs
  back to 15.
* `pad_to_captured_width` then makes the step uniform. It pads to the **smallest captured width ≥ the
  longest draft list**, *not* to `spec.num_draft` — padding back up to `num_draft` would have undone
  the narrowing on every step while the controller happily reported a narrow width.

---

## 2. The M cliff, and why the cap is now a function of the checkpoint

The verify forward is flat over tokens: the model sees one `[M, hidden]` activation with
`M = padded_bs * (width + 1)`. Several kernels change family at an M threshold. Those thresholds are
**imported** from the modules that own them (`live_m_thresholds`), with the documented values kept
alongside only as a torch-free fallback and **cross-checked at boot** — a drift logs a warning
instead of leaving this file silently stale.

| family | threshold | source |
|---|---|---|
| generic (bf16 dense, LM head, fused SwiGLU) | 16 | `layers/minv.py:_DECODE_GEMV_MAXM`, `layers/embedding.py:_LMHEAD_GEMV_MMAX` |
| e2m1 / MXFP4 / NVFP4 quantized dense | 16 | `quant/kernels.py:_W4A8_GEMV_MAX_E2M1` |
| **int4 W4A8 quantized dense** | **8** | `quant/kernels.py:_W4A8_GEMV_MAX_INT4` |

The int4 row was a review finding, and it was **measured before it was honoured**
(`tools/w4a8_int4_m_crossover.py`, min-of-5×50, gfx1201). Timing *both* kernels at the *same* M on
four dense shapes:

```
int4 W4A8, N=4096 K=4096   M:  1     4     8     9    12    16
  decode_gemv  (us)          26.2  39.0  53.1  55.4  66.0  83.2
  prefill_wmma (us)         104.2 103.2 104.9 117.9 118.3 120.5
  gemv/wmma                  0.25  0.38  0.51  0.47  0.56  0.69
dispatcher M=8 -> M=16 cost:  2.03x (N=K=4096) / 1.55x (11008x4096) / 1.50x (6144x2048) / 3.68x (2048x6144)
```

`decode_gemv` is faster at **every** M up to 16 on three of four shapes, so the switch at 8 is a
genuine **cliff**, not a crossover: past M=8 an int4 model pays *more per verify row*. Hence
`max_verify_rows(quant) == 8` there and a ladder max of 7.

Two honest riders:

* **This is a no-op on every configuration this repo ships.** Qwen3.6-35B-AWQ runs MTP at K=4 and
  GLM-4.7-Flash-AWQ runs EAGLE3 at K=6 — both already under 7 — and Laguna is NVFP4 (e2m1, ceiling
  16, ladder unchanged at `[3, 7, 15]`). It only bites a hand-set `--spec-num-draft > 7` on an int4
  checkpoint. `tools/verify_width_unit.py` asserts that non-effect rather than trusting it, and the
  serve confirms it out loud: Qwen3.6-35B-AWQ now boots with
  `ADAPTIVE verify width ON widths=[2, 3, 4] (K=4, capped at 7 for the M<=8 decode-kernel cliff)` —
  the int4 ceiling detected and applied, the ladder unchanged.
* The measurement suggests `_W4A8_GEMV_MAX_INT4 = 8` is itself too low for these dense shapes —
  `decode_gemv` wins to M=16. **Raising it is the better fix and is NOT done here**: it changes the
  kernel, and therefore the numerics, for ordinary int4 decode at M=9..16 as well. Recorded as
  follow-up.

Scope of the clamp, stated plainly: because `M = padded_bs * (width+1)`, the cliff is only
*reachable* at `padded_bs == 1`. At bs=8 any width ≥ 1 is already past it. Clamping by
`cap // padded_bs` would drive the width to 1 at concurrency for a boundary you cannot get back
under, so the cap is a hard `width ≤ cap - 1` and the width at bs>1 is chosen by acceptance alone.

---

## 3. The staging invariant (a real defect, found in review)

`pad_to_captured_width` returns **both** the (possibly truncated) real drafts and the staged rows,
and it guarantees

```
len(staged[i]) >= len(drafts[i])   for every i
```

This is not decoration. The accept loop walks the flat verify output with `offset += staged_q_len`
but **slices `q_len = len(drafts)+1` rows**. A request staged narrower than it drafted reads into its
*neighbour's* rows, and the last request in the batch trips `verify_greedy`'s
`assert len(target) == K + 1` and takes the scheduler down.

The adaptive path could not violate it (the controller truncates `drafts` to a captured width first),
but the **DP+EP** path pinned `w_pad = widths[-1]` *without* truncating — and `widths[-1]` is capped
for the M cliff, so `--spec-num-draft 16` under DP+EP staged 15 rows against 16 real drafts. The
width choice, the truncation and the padding are now **one pure function**, which is also what makes
the invariant testable: DP+EP is otherwise not reachable without standing up a two-replica MoE serve.
`tools/verify_width_unit.py` covers it, including a witness that reproduces the pre-fix form.

---

## 4. What "lossless" means here — and the one thing it does NOT mean

**The guarantee.** Acceptance statistics choose the WIDTH; they never choose the OUTPUT. Every
emitted token is the target's own output at its own query row — `verify_greedy` emits
`target[0..n]`, i.e. the target's argmax chain. In **exact arithmetic** the emitted sequence is the
target's greedy continuation *regardless of the width, and regardless of the drafter*: drafts only
decide how many of those tokens arrive per step. Truncating a draft list from K to W deletes trailing
speculative rows and nothing else.

**What it does not mean: byte-identical text against a different fixed width.** It is not, on this
engine, and that is a property the change *inherits* rather than introduces.

**First, the thing that has to be established before ANY text comparison on this stack: on
Qwen3.6-35B-A3B-AWQ + MTP the noise floor is NOT zero, even with both known nondeterminism sources
disabled.** `tools/fixgate_neutral.txt`, greedy, seed 1234, one prompt, `MINISGL_KV_FP8=0` **and**
`MINISGL_MOE_G2FUSE=0`, each request issued twice in the same boot:

```
                       MT=32      MT=64      MT=128                 MT=256
753f08d2 (pre-fix)   aba5d080   e36f8819   729464.. / bd68f4b6..   c51e18fb (x2)
FIXED                aba5d080   e36f8819   bd68f4b6.. / 729464..   cf0598fa / f2ab13c9
```

At MT=128 the pre-fix tree returned **two different completions for the same request in one boot**,
and the fixed tree returned **the same two, in the other order**. At MT=256 the fixed tree returned
two more. So on this model a byte-identity gate is comparing against a coin flip: the reviewer's
observation that "the new tree differs from the pre-change run at 128 and 256 tokens" is reproduced
here *between a tree and itself*. Whatever else is true, that comparison cannot attribute anything to
the adaptive width. (Laguna + DFlash under the identical harness is `repeat=SAME` at all four
lengths, so this is specific to the Qwen3.6-35B-AWQ stack, upstream of anything here, and it
predates this work. It is worth its own investigation and is **not chased here**.)

**Second, the control that settles the question.** On the **UNTOUCHED phase parent `d276137c`** — no
`width.py`, no `capture.py`, propose eager, width fixed — Qwen3.6-35B-AWQ + MTP, greedy, both noise
sources off, each request issued twice (`tools/fixgate_mtp_width_control.txt`):

```
  --spec-num-draft      MT=32      MT=64      MT=128     MT=256
  K=4                 aba5d080   e36f8819   8b07b0d7   508690.. (repeat SAME)
  K=3                 6e13b1f1   3a1f7e9a   056da2d7   d28be93d (repeat DIFFERS)
  K=2                 0f8c8d5f   e8f7bc34   b1ab67bc   c4f212bb (repeat SAME)
```

**Three fixed widths, three different completions, at every length — with `repeat=SAME` at 32, 64
and 128 on all three, i.e. a zero noise floor exactly where the comparison is being made.** Nothing
from this phase is in that tree. So `width → text` is a pre-existing property of this engine, and
"no fixed pre-change width reproduces the new default" is a *restatement of that property*, not
independent evidence of a defect in the controller: a width-VARYING scheme cannot, even in
principle, reproduce any single fixed width on an engine that is not M-invariant.

**Third, the mechanism** — so this is an explanation, not an excuse:

* **A MoE verify forward cannot be M-invariant, structurally.** Routing partitions the `M`
  tokens across experts and `moe_align` pads each expert's rows up to `_moe_block_m`
  (`quant/kernels.py`, itself a function of `num_tokens`); change `M` and the per-expert row counts
  and tile height change, so the grouped-GEMM reduction regroups. On top of that the fused gemm2 is
  documented **in place** as varying its atomic reduction order run to run — observed live here:
  Laguna+DDTree repeated the *same* greedy request within one boot and the two completions differed
  at 256 tokens (`tools/fixgate_ddtree.txt`), which is why the determinism gates force
  `MINISGL_MOE_G2FUSE=0`.
* **Both** models in play are MoE (Laguna-XS.2 is 256 experts / top-8; Qwen3.6-35B-A3B likewise), so
  the difference between them is NOT structural. It is the argmax MARGIN distribution: an M-driven
  last-bit change only becomes visible text when it crosses a near-tie. Empirically, on the review's
  prompt, Laguna's DFlash leg crossed one only at the kernel-family boundary (K=16→15) while
  Qwen3.6-35B + MTP crosses one at essentially any width change. Do not read that as "dense is safe";
  read it as "how often you see it is model- and prompt-dependent, and the underlying arithmetic
  dependence is always there."
* What would have falsified the position: if K=4 / K=3 / K=2 had produced the **same** text on the
  parent, the engine would be M-invariant and the adaptive width really would have introduced a
  difference. It did not.

**Fourth, the adaptive tree's output is drawn from exactly that set of fixed-width outputs** — which
is the tightest confirmation available. The last leg of the same run is the shipped tree at its
default, ladder `[2, 3, 4]`, and at MT=64 its two identical requests returned `e36f8819…` and
`3a1f7e9a…` — which are, respectively, **the parent's K=4 output and the parent's K=3 output**. The
controller picked a different rung for the second request (its EMA carries over), and the text
followed the rung. Not a new failure mode: the same tokens the pre-change engine emits at whichever
width is in force.

**What IS gated, and passes** (`tools/fixgate_neutral.txt`): on **Laguna + DFlash**, the shipped
configuration, the review fixes are **byte-identical to `753f08d2` at all four generation lengths**,
with `repeat=SAME` on every request (zero noise floor) and different `width.py`/`scheduler.py` md5s
inside the two containers proving the legs really are different code. On **Qwen3.6-35B-AWQ + MTP**
they are identical at 32 and 64 tokens and land inside the model's own nondeterminism above that.

**The one change that IS visible to a user upgrading:** Laguna's shipped default was
`--spec-num-draft 16` (qlen 17) and the ladder caps it at 15 (qlen 16). That is the deliberate M≤16
clamp, it is worth +41.7% at NREQ=1, and a pre-change run pinned at K=15 reproduces the new output
byte-identically — so the propose rewrite and the capture contribute *zero* text change and the
delta is entirely the qlen 17→16 kernel-family flip. Somebody upgrading this branch should know that
deterministic greedy completions for the default Laguna config change, once, for that reason.

---

## 5. Known limits

* **DP+EP is pinned, not adaptive** (`_adaptive_width_ok` returns False) and remains **untested on
  hardware**: the reasoning follows the existing EP comments, and the pure-function invariant above
  is unit-tested, but no DP=2 `--enable-ep` serve was run.
* **The controller uses the batch MEAN**, so at concurrency a request that would accept many drafts
  can be truncated by its neighbours. Lossless, but it forgoes speculation for that request. A
  per-request width needs a ragged verify, which the captured-graph contract forbids. The EMA α
  (0.25), the exploration term (+1) and the ladder shape are unmeasured choices.
* **`Engine._graph_capture_bytes` does not model multi-width capture.** It still reserves at
  `qlen = num_draft + 1`, which now *over*-reserves (all widths share one `VerifyCaptureBuffer`), so
  it errs safe; the per-width graph-pool cost (measured: 0.23 GiB for 4→12 graphs) is unmodelled.
* **DDTree's tree-verify runs at `tree_qlen = budget + 1` = 33**, far past the M boundary. Out of
  scope here (this is the linear verify), but it means the cliff constraint is still violated on that
  path — `budget <= 15` is the only cliff-comparable DDTree configuration.
* **ngram / TiDAR paths were not exercised** and are unaffected by construction.
* **`Qwen3.6-35B-A3B-AWQ` does not reproduce itself** at ≥128 tokens even with `MINISGL_KV_FP8=0` and
  `MINISGL_MOE_G2FUSE=0` (§4). That is a pre-existing engine finding surfaced by this work, not
  fixed by it, and it means no byte-identity gate on that model is trustworthy past ~64 tokens.
* **`_W4A8_GEMV_MAX_INT4 = 8` is probably too low for these dense shapes** (§2). Raising it would be
  the better fix and would let an int4 model use the full 15-wide ladder; not done here because it
  changes ordinary int4 decode numerics at M=9..16.

---

## 7. MEASURED 2026-08-01 — §1's objective is wrong: the optimal rung depends on BATCH SIZE

`tools/verify_width_rung_sweep.sh`, Laguna TP=2, K=16 held fixed (so propose cost is identical and
verify width is the only variable), short ~95-token prompt, TRUE tok/s from `usage.completion_tokens`
over wall, each leg's rung asserted from the engine's own `PINNED to rung N` log line rather than
inferred from how it was invoked:

| rung | M at bs=1 | bs=1 tok/s | M at bs=8 | bs=8 tok/s |
|---|---|---|---|---|
| 3  | 4  | 93.41  | 32  | **254.31** |
| 7  | 8  | **103.22** | 64  | 145.94 |
| 15 | 16 | 90.63  | 128 | 125.92 |
| free-running controller | — | ~103–109 | — | 202.1 |

**Both optima are INTERIOR, and they are on opposite ends of the ladder.** The mechanism is
`M = bs * (width + 1)` against the M<=16 decode-kernel boundary this document already documents in
§2: at bs=1 every rung stays inside the decode-GEMV family, so step cost is nearly flat and more
accepted tokens per step wins (peak at 7, with 15 falling off at the cliff edge); at bs=8 every rung
is already past the boundary into WMMA + MoE expert fanout, where cost scales with verify rows, so
the NARROWEST rung wins by a wide margin.

**Pinning rung 3 at bs=8 beats the live adaptive controller by +26% (254.31 vs 202.31).**

### What this means for §1

* The controller's objective — accepted tokens per STEP — is not the serving objective, which is
  tokens per MILLISECOND. `choose()` (`width.py:260`) has **no batch-size term**: it averages per-uid
  acceptance EMAs and reads a rung off the ladder, while the cost of that rung is set by
  `bs * (width+1)`. At bs=8 it therefore climbs into rungs that cost more time than they return.
* §1's claim that the `+1` exploratory row lets a narrowed request "climb back" is arithmetically
  false for any realistic acceptance: escaping rung W requires `mean_ema > W-1`, i.e. >66.7% of
  offered rows at rung 3, >85.7% at rung 7, >93.3% at rung 15, because `record`
  (`width.py:285`) stores `accepted` at face value and never uses the `width` argument it is handed
  at line 277 to detect censoring. Every rung is a stable fixed point.
* **Do NOT "fix" the censoring alone.** Imputing the next rung up on a saturated observation drives
  the controller toward rung 15, which the table above shows is the WORST rung at BOTH batch sizes.
  The censoring bug and the missing cost model happen to cancel at bs=8 and compound at bs=1.
* `tools/verify_width_unit.py:113-114` cannot catch any of this: `record([1], [min(15, w)], w)` is
  100% acceptance at every width — the one regime where a fixed `+1` clears `W-1` — and it drives a
  single uid with no notion of step cost. §1 cites it as proof of a property the code does not have.

### The shape of a correct fix (NOT yet implemented)

Choose the rung that maximizes **expected accepted tokens per millisecond** at the CURRENT batch
size, not accepted tokens per step. That requires (a) a censoring-corrected acceptance estimate and
(b) a per-rung step-cost model keyed on `M = bs*(width+1)` — the cheap version being a cap that keeps
M at or under the same decode-kernel boundary §2 already imports from the kernel modules. Both halves
are needed: (a) alone overshoots, (b) alone cannot tell a good drafter from a bad one.

## 8. The fix, and what it measures at

`choose()` now has BOTH terms §7 said it needed, and neither alone is sufficient:

* **Censoring-corrected acceptance.** `record` folds each step into per-depth survival counters
  (`_st`/`_sh`): a step at width W observes "did the run reach j" for j = 1..W and nothing beyond,
  so `survival(j)` is unbiased for j <= W and extrapolates past the deepest rung ever offered with
  the deepest measured rate. `expected_run()` = sum of survivals is then a run-length estimate that
  does NOT collapse to whatever rung the controller happens to be sitting on. Decayed (`_DECAY`,
  ~200-step memory) so it tracks a drafter whose acceptance shifts mid-request.
* **A batch-size cost cap.** `width_cap(bs)` returns the widest rung with `bs*(width+1) <= 32`.
  Deterministic in `bs` alone — deliberately NOT a measured step time, because each TP rank would
  time a different cost, choose a different width, and desync the verify batch into an illegal
  address. 32 rows is the unique power-of-two reproducing both measured optima.
* **Cold start from the MIDDLE rung.** With no observations the survival estimate must assume
  something; assuming the best opens at the widest rung, which is the most expensive place to be
  wrong. Measured: opening at the top cost 97.65 tok/s at bs=1 vs 102.77 opening at the middle.

### Measured (Laguna TP=2, K=16, TRUE tok/s, rung asserted from the engine log)

| leg | bs=1 short | bs=8 short | bs=8 long (~3.1k prompt) |
|---|---|---|---|
| pinned rung 3  | 93.41  | **254.31** | **125.75** |
| pinned rung 7  | **103.22** | 145.94 | 88.99 |
| pinned rung 15 | 90.63  | 125.92 | 75.73 |
| OLD controller | —      | 202.1  | — |
| **NEW controller** | **102.77** | **253.97** | **122.09** |

The new controller lands on the optimal rung unaided at every point measured: **99.6%** of the best
pinned rung at bs=1 short, **99.9%** at bs=8 short (**+25.6% over the old controller**), and **97.1%**
at bs=8 long.

### Known residual, NOT fixed here

The 2.9% gap at bs=8 long is real. Long context is monotone-decreasing in width (125.8 / 89.0 / 75.7),
i.e. narrower is better there than the row budget alone implies, because KV traffic per verify row
grows with sequence length while the budget is a pure row count. As requests drain and `bs` falls,
the cap admits wider rungs that long context does not actually want. A context-length term in
`width_cap` would close it; it is deliberately not guessed at here, because the only honest form is
another sweep (rungs x bs x context), and the residual is ~3%.

`tools/verify_width_unit.py` §5 now gates the regime that broke: partial acceptance at bs=1 and bs=8
against the SAME drafter, the extrapolation past the offered rung, that a bad drafter still narrows,
and TP-rank determinism. The pre-existing §4 cases still pass unchanged.
