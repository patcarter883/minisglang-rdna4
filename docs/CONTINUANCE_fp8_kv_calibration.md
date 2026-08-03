# CONTINUANCE — finish the fp8-KV calibration work

**Written** 2026-08-04. **Repos/SHAs this describes:** `minisgl-rdna4` @ `dfd3e68a` (branch
`rdna4`), `rdna4-hip-kernels` @ `02714ba` (branch `main`). Both merged and serve-validated.

Read this whole file before touching anything. The single most important fact is in §1: **the
calibrator that shipped cannot calibrate any model this box actually serves.** Everything else is
downstream of fixing that.

---

## 0. What already landed — do NOT redo it

`MINISGL_KV_FP8=1` is the **docker-compose default**, so the default serve has always run an fp8 KV
cache. Until this work it did so uncalibrated and unsafe. Three things were fixed:

1. **`finalize_kv_calibration()` had no caller anywhere in the repo.** Every `k_scale`/`v_scale`
   stayed 1.0 — K/V cast straight to e4m3 with no range fitting, flushing everything below 2^-9 to
   zero. Now `python/minisgl/kvcache/fp8_scales.py` resolves scales in `Engine.__init__` between
   pool construction and graph capture.
2. **The e4m3 store returned NaN above 448**, where torch saturates. Unreachable at scale 1.0 (raw
   K/V amax is O(10–100)) but **guaranteed to fire the moment a max-calibrated scale puts the
   largest observed value exactly at 448** — so fixing (1) without (2) would have shipped the bug.
   One NaN key kills an entire softmax row. This, not the SWA path, is why Laguna and Qwen3.5-4B
   emitted a single repeated token.
3. Two engine reads hardcoded "the descale is 1.0": the SWA ring-window gather (a bare `.to(bf16)`,
   18–40× off against Laguna's own checkpoint scales) and the Triton fallback.

Also: **stochastic rounding was implemented, measured, and deleted.** Do not re-propose it for the
KV cache. It was better on 1 of 84 (layer, context) cells, median 1.44–1.48× *worse*, 2.13× kernel
time. SR removes a *coherent accumulating* bias; a KV cache is write-once storage so there is
nothing to accumulate, and sign-symmetric activations already cancel RNE's bias to 1.6e-5 of RMS.
It paid the textbook √2 variance penalty (rel-RMSE 0.0267 → 0.0382 = 1.43×) for nothing. The same
technique **is** an 8× win on the GDN recurrent state, which is a recurrence — that contrast is the
whole lesson and it is recorded in-source at both rounding sites.

Scale resolution order, already implemented: **sidecar → checkpoint
`quantization_config.kv_cache_scheme` + `self_attn.{k,v}_scale` broadcast per-head → loud warning +
defined identity.** Int8 schemes are refused (an int8 scale is `amax/127` and would be misapplied).
Global→compact layer mapping goes through `full_attn_layer_ids`/`swa_layer_ids` so SWA/GDN hybrids
land in the right pool.

---

## 1. THE BLOCKER — the calibrator cannot run on any production model

`tools/kv_fp8_calibrate.py` runs **TP=1** ("the offline LLM API is TP=1", its docstring). Measured
on this box:

| model | on-disk size | fits one 16 GB card? |
|---|---|---|
| `cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit` | 24 GB | **no** |
| `poolside/Laguna-XS-2.1-NVFP4` | 20 GB | **no** |
| `QuantTrio/GLM-4.7-Flash-AWQ` | 19 GB | **no** |

Both cards are 16,304 MB. So the per-head sidecar path — *the only source of per-head scales, since
no checkpoint ships them* — currently works only for small models (it was validated on Qwen3.5-4B
TP=1). **Nothing that this box serves in production can be calibrated today.** Any plan that starts
"run the calibrator on the 35B" is dead on arrival; check this first.

### Fixing it — the shape of the answer
Prefer **teaching the calibrator TP=2**. The sidecar format is already right for it: each rank owns
`num_kv_heads/tp` heads, the sidecar holds **global** per-head rows, and `fp8_scales._shard_row`
already slices global rows per rank at serve time (one sidecar serves any TP). So the work is to run
the calibration forward at TP=2 and gather per-head amax across ranks into global rows — not to
change the file format or the serve side.

Alternatives, both worse, listed so they are not re-discovered: calibrating through GTT spill
(fits, but PCIe-bound at ~25–30 GB/s vs 707 GB/s HBM — see the `amdgpu_gtt_used_bytes` gauge, and
note a spilled run gives *correct* amax, just slowly, so this is a fallback not an error); or
CPU-offload calibration (slower still, and risks dtype drift from the served path).

Whatever you choose, the invariant that must not break: **calibration runs against the bf16 cache**
(`MINISGL_KV_FP8=0`, `MINISGL_KV_FP8_CALIBRATE=1`, set before `minisgl` is imported — they are read
at pool construction). Measuring amax with an fp8 cache installed feeds fp8 error back into every
later layer's activations and would require booting the very path being configured.

---

## 2. There is no calibration fixture — make one, and make it durable

`--text` is **required** and no fixture exists in the repo (`fixtures/` does not exist; nothing
matches `calib*`). The scale it produces is a promise about the range of everything the served model
will ever store, so the text must be representative of real serving traffic — for this box that
means agent/tool-call transcripts and long-context material, not a generic wikitext dump.

Repo rule that applies directly here: **fixtures must be durable and recorded** — never write one to
session tmpfs or the scratchpad, and echo its byte size. The tool already helps: it hashes the text
and writes size + hash into the sidecar metadata so a scale table can be traced back to the data
that produced it. Put the fixture somewhere committed or in a stable path and record which one
produced which sidecar.

---

## 3. The open question you must answer with a measurement, not a preference

**Should `MINISGL_KV_FP8=1` remain the compose default?**

Known cost, measured with a real per-head sidecar on Qwen3.5-4B TP=1, graph capture on:
**26/29 vs bf16's 27/29**, at 65.3 vs 65.1 tok/s (bs=1) and 222.9 vs 222.9 (bs=4). The single extra
failure was a **mid-context fact retrieval at 7.7k tokens**. Laguna TP=2 on checkpoint scales was
25/29 vs bf16 24/29 at 79.6 vs 78.6 tok/s.

So on the evidence so far fp8 KV is roughly **throughput-neutral** and costs a little
long-context accuracy. Its actual value is VRAM: it halves KV bytes, which buys context or
concurrency. That trade has never been measured end-to-end on this box — *that* is the experiment
worth running, not another accuracy microbench. Frame it as: at equal VRAM, does fp8-KV-plus-more-
context beat bf16-KV-with-less?

Note the interaction with ReplaySSM, which also just landed: the GDN ring is bought out of the same
KV pool (L=8, +12.5% of the checkpoint). Both features spend the same budget.

---

## 4. Per-head is the small part — calibrate the expectation

Do not oversell per-head. Re-measured on real Qwen3-0.6B K/V over real text with the merged kernel:
attention-output rel-RMSE **0.0775 → 0.0647 (−16.6%)**, better on 9 of 15 (layer, ctx) cells but
*worse* on L0 (3/3) and L7 (2/3). Storage rel-RMSE is a **wash** (0.026797 → 0.026720).

The mechanism is **only subnormal flush** — e4m3 is a floating format whose per-element exponent
already tracks the value, so a too-large scale mostly just shifts a quiet head's exponents down at
the same relative error. This is *not* the int8 situation where the scale **is** the resolution.
Demonstrated through the real kernel: a wide-dynamic-range head is **74.4% flushed to zero**
per-tensor vs **5.5%** per-head.

Therefore per-head only pays where head amax spread is wide. Measured spread: **1.6–4.4× on
Qwen3-0.6B** but only **1.29–1.55× on Qwen3.5-4B**, where per-head buys essentially nothing.
**Measure the spread on the target model before investing in a sidecar for it** — it is cheap and it
predicts the payoff.

---

## 5. Rules and gates that bind this work

- **Never calibrate mid-serve.** One descale must undo every store ever written under it, so
  changing it later invalidates the whole cache. A captured graph provably picks up a late in-place
  descale write (`max|Δ| = 4.85e-01`), and restoring the old value restores the output bit-exactly.
  Scales must be final before any readable KV is written — hence the `Engine.__init__` placement,
  which is a correctness requirement and not a convenience.
- **RULE 4** (`rdna4-hip-kernels/KERNEL_CORE_POLICY.md`): not done until it boots at the served
  TP/graph-capture config and answers. A microbench win that does not move the serve is not a win.
- **No env-gating on merge.** If it merges it is ON; the worktree is the isolation, not a flag.
- **Source isolation**: per-task git worktree, never mount the shared `$PWD`. Image builds need
  **clean** worktrees for both contexts, a bumped `KERNELS_REF` (it is only a cache-buster label —
  forget it and you ship stale kernels believing otherwise), and
  `--build-context uprof=/home/pat/pkgs` or the build fails with a confusing "pull access denied".
  **Preflight by importing the kernel packages in the built image**: an ABI mismatch presents as a
  0%-GPU wedge until the readiness timeout, *not* a build error.
- **GPU**: every workload through `gpu-lease -n 1 -- <cmd>` (bare command, let it block). `-n` is
  how many cards, not which.
- **Cross-card reproducibility**: GPU 0 is an RX 9070 XT, GPU 1 an RX 9070, and the lease assigns
  whichever is free. The same parity case produces **different error digits** on the two cards. Use
  the pass/fail margin against tolerance as the signal; a changed error figure between runs is not
  by itself a regression.
- Two gotchas that cost time last session: do **not** pass `-p <project>` to compose under
  `gpu-lease` (the arbiter sets `COMPOSE_PROJECT_NAME=lease-<name>` and looks the container up by
  it; overriding makes it think the launch failed and **release the lease** while your container
  keeps running unleased). And a TTFT probe must match **any** delta chunk — GLM streams
  `reasoning_content` before `content`, so filtering on `"content"` never fires.

---

## 6. Suggested order of work

1. Measure per-head amax spread on the target model(s) (§4). If it is ~1.3× the sidecar is not worth
   building for that model, and the answer is checkpoint/per-tensor scales — say so and stop.
2. Build the representative, durable calibration fixture (§2).
3. Teach the calibrator TP=2 (§1). Gate it by reproducing the existing TP=1 Qwen3.5-4B sidecar
   through the TP=2 path — same model, same text, scales should agree.
4. Generate sidecars for the production models; serve-validate each under RULE 4 with graph capture.
5. Answer §3 with an equal-VRAM context/concurrency experiment, and change or keep the compose
   default on that evidence.

## 7. Loose end, unrelated to fp8 but worth knowing

`docker-compose.yml` defaults to `minisgl-rdna4:lean`, which is **38 hours older than the merged
work described here**. `minisgl-rdna4:comb` is the validated image built from `dfd3e68a` +
`02714ba` (ReplaySSM active at L=8, fp8 scale resolution live). Nothing is served from the merged
code until `lean` is repointed or `MINISGL_IMAGE` is set.
