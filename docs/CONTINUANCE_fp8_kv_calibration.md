# CONTINUANCE — fp8-KV calibration

**Updated** 2026-08-04 (second pass). **Repos/SHAs:** `minisgl-rdna4` branch `rdna4` (this work
merged on top of `b7ee6e68`), `rdna4-hip-kernels` @ `02714ba` — unchanged, no kernel edit was needed.

The blocker the first pass ended on ("the calibrator cannot run on any model this box serves") is
**resolved**: the calibrator runs at TP=N, and Qwen3.6-35B and Laguna-XS-2.1 are calibrated and
serve-validated. §1–§3 are what is now known; §5 is what is left.

---

## 0. What landed in pass 2

1. **`tools/kv_fp8_calibrate.py --tp N`** — one process per rank (`mp.spawn`, the shape
   `server/launch.py` uses). Each rank owns `num_kv_heads/tp` heads and accumulates only its own
   amax rows; the rows are gathered into GLOBAL per-head rows on the host (broadcast-per-source on
   the gloo group) *before* any scale is computed, so the sidecar is TP-independent and
   `fp8_scales._shard_row` re-slices it for whatever TP the serve runs at. Supporting changes:
   * `LLM(...)` now takes `tp_info` (was hard-coded TP=1). A multi-rank OFFLINE run needs no ZMQ:
     every rank derives the identical prompt list from the fixture and drives it locally.
   * `SchedulerIOMixin.establish_inter_rank_link()` is a no-op in offline mode (it used to
     AttributeError on `_send_into_ranks` the moment a multi-rank offline run entered `run_forever`).
   * `kv_amax_to_descale()` in `mha_pool.py` — ONE formula, shared by the in-pool
     `finalize_kv_calibration` and the host-side TP gather.
   * Calibration runs with the recurrent-radix snapshot store OFF (`--gdn-radix` keeps it):
     calibration chunks share no prefixes, so it reserved 0.59 GiB that Laguna's bf16 KV pool needed
     — that reservation was the boot failure on Laguna.
2. **`tools/make_kv_calib_fixture.py`** — deterministic builder for the calibration fixture,
   INTERLEAVED by source so any prefix is still a mixture (the calibrator reads only the first
   `--max-chunks`).
3. **`tools/kv_scale_compare.py`** — numpy-only sidecar differ; the gate for any calibrator change.
4. **`tools/kv_fp8_serve_ab.sh`** + **`tools/kv_ctx_capacity_probe.py`** — the RULE-4 serve
   validation and the equal-VRAM experiment §3 of the first pass asked for.
5. **fp8 MLA is now calibrated too** — per LAYER, the only granularity a latent admits.
   `MLAKVCache` gained an amax accumulator, a persistent per-layer descale/inv-scale table and a
   scaled store; `attention/mla.py` passes a 1-element VIEW of that table to the fp8 decode/verify
   ops instead of the old hard-coded `torch.ones(1)`; and the prefill MATERIALIZE path in
   `glm4_moe_lite.py` multiplies the dequant by the same descale (a bare `.to(bf16)` there was the
   third read site, the same shape of bug as the SWA ring gather on the MHA side).
   `install_kv_fp8_scales` now drives MLA pools as well — they were excluded, which is how GLM ended
   up serving an e4m3 latent cache with an implicit scale of 1.0 under the compose default.

## 1. The fixture (durable, recorded)

`/home/pat/fixtures/minisgl-kv-calib/kv_calib_v1.txt` — **2,009,224 bytes, sha256[:16]
`c6b3888faa71d4b0`**, manifest beside it. Measured mixture: engine+kernel source 31.2%, babilong
long-context needle samples 25.6%, repo markdown 21.1%, prose (wikitext-2) 14.9%, real tool/MCP JSON
from this box's agent caches 9.9%. Rebuild byte-identically with
`python3 tools/make_kv_calib_fixture.py`. The calibrator stamps size+hash into the sidecar metadata,
so any scale table traces back to the data that produced it.

## 2. Calibration results (TP=2, 131k tokens, ctx 4096 × 32 chunks)

Sidecars + JSON reports in `/home/pat/fixtures/minisgl-kv-calib/sidecars/`.

| model | pools | per-head K spread (median / max) | V spread | wall |
|---|---|---|---|---|
| `cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit` | 10L × 2H | **1.09× / 1.28×** | 1.11× / 1.89× | 34s |
| `poolside/Laguna-XS-2.1-NVFP4` | 10L × 8H main + 30L × 8H SWA ring | 1.42× / 1.79× (main), 1.28× / 1.73× (ring) | 1.50× / 2.16×, 1.45× / 2.53× | 50s |
| `QuantTrio/GLM-4.7-Flash-AWQ` (MLA) | 47L latent | n/a — **per-LAYER**: latent amax 4.06–11.75 (2.89× across layers), descale 0.00907–0.02623 | same scalar | 42s |
| `Qwen/Qwen3-0.6B` (reference) | 28L × 8H | 1.84× / 4.61× | 1.95× / 6.28× | 1s |

**Per-head is not the story on the production models.** The first pass predicted this and it held:
the 35B's spread is 1.09×, and at TP=2 it has ONE KV head per rank, so per-head is literally
per-tensor there. What the sidecar buys on the 35B is *any* calibration at all — that checkpoint
ships no `kv_cache_scheme`, so without it every scale is 1.0 (the loud boot warning).

**Two independent checks that the numbers are right:**
* **TP-invariance.** Qwen3-0.6B calibrated at TP=1 and TP=2 on the same fixture: median relative
  difference **0**, 96.2% of entries within 1%, max 4.93e-2. TP=2 is run-to-run bit-identical; TP=1
  drifts 4.8e-3 against itself. The residual TP delta is forward numerics (column/row sharding
  changes reduction order), not the gather — a head-ordering bug would move whole heads, not 4% of
  entries by <5%. It is not divergent decode tokens either: `--max-tokens 1` reproduces it exactly.
  `tools/kv_scale_compare.py --tol 0.05` is the gate; a 2% tolerance would fail on TP noise alone.
* **Against a vendor calibration.** Laguna ships its own per-tensor `k_scale`/`v_scale`
  (`kv_cache_scheme`, minmax observer). Our fixture-derived per-head max vs their per-tensor scale,
  40 layers: K ratio median **1.034** (range 0.907–1.270), V median **1.128** (0.920–1.585). An
  independent calibration on different data lands within ~10% — the fixture is range-representative.

## 3. Should `MINISGL_KV_FP8=1` stay the compose default? YES — measured

Qwen3.6-35B TP=2, graph capture on, `CONC=4`; three legs, each asserting its own boot-log provenance
(`tools/kv_fp8_serve_ab.sh`; artifacts under `/home/pat/fixtures/minisgl-kv-calib/serve_ab*`):

| leg | KV pool | acceptance (3 reps) | decode tok/s M=1 / M=4 |
|---|---|---|---|
| bf16 (`MINISGL_KV_FP8=0`) | **26,720 tokens** | 27 / 26 / 27 of 29 | 90.8 / 251.1 |
| fp8, no sidecar (today's default) | **53,440 tokens** | 27 of 29 | 90.4 / 252.9 |
| fp8 + per-head sidecar | **53,440 tokens** | 27 / 28 / 27 of 29 | 90.6 / 251.6 |

Throughput is a wash (<1%). Quality is a wash too — and the 26-vs-27 gap that a single run would
have called a regression is **noise**: "generates syntactically plausible code" is a keyword
assertion on free text and failed in 2 of 3 bf16 reps and 1 of 3 fp8 reps. Always run `ACC_REPS=3`;
the serve is not bit-reproducible past ~32 tokens. The two failures common to every leg
(`temperature>0 with different seeds diverges`, `natural stop -> finish_reason=stop`) are unrelated
to the KV cache and are open serve bugs.

**The decisive result is capacity** (`tools/kv_ctx_capacity_probe.py`, mid-context needle, same VRAM):

| context | bf16 | fp8 + sidecar |
|---|---|---|
| 7.2k prompt tokens | served, needle FOUND | served, FOUND |
| 14.7k | served, FOUND | served, FOUND |
| 30.7k | **rejected — "exceeds the servable maximum 26720 (KV pool)"** | served, **FOUND** |
| 46.7k | **rejected** | served, **FOUND** |
| 62.7k | rejected | rejected (max 53,440) |

At equal VRAM fp8-KV does not trade accuracy for context on this model — it *adds* context bf16
cannot serve at all, and retrieves correctly inside it. Keep the default ON.

**GLM-4.7-Flash (MLA), same three legs, TP=2 + graph capture:** KV pool **61,232 → 122,480
tokens** (2× again), decode 56.6 (bf16) / 55.2 (fp8 uncal) / 55.2 (fp8 + per-layer sidecar) tok/s,
acceptance 27/29 on every leg (2 reps each; the movers are the same leg-independent checks). The
calibrated leg answering at all is the round-trip proof: the store now divides by 0.009–0.026, so a
read that ignored the descale would be ~40–110× out and the suite would collapse, not score 27/29.

**How to serve with a sidecar** — explicit path, deliberately NOT installed into the shared HF model
dir (that would silently change every other agent's serve of the same checkpoint):

```
cp /home/pat/fixtures/minisgl-kv-calib/sidecars/qwen35b-awq_tp2.safetensors ./kv_scales_qwen35b_tp2.safetensors
MINISGL_KV_FP8=1 MINISGL_KV_FP8_SCALES=/engine/kv_scales_qwen35b_tp2.safetensors \
  MINISGL_IMAGE=minisgl-rdna4:comb MODEL=qwen35b-awq TP=2 \
  gpu-lease -n 2 --detach -- docker compose --profile serve up -d
```
The 35B sidecar is committed at the repo root for exactly this (2.3 KB; the serve mounts the repo at
`/engine`). Boot must log `installed PER-HEAD scales from sidecar …` — that line IS the provenance.

**Laguna keeps its checkpoint scales.** Ours agree within ~10% but are up to 9% *smaller* on some
layers (very slightly more clipping), and the checkpoint path is already serve-validated (25/29).
No measurement says ours is better; do not switch without one.

## 4. How to run a calibration (recipe that works)

```
MINISGL_IMAGE=minisgl-rdna4:comb \
MINISGL_CMD='python /engine/tools/kv_fp8_calibrate.py --model <hf id> --tp 2 \
  --text /fixtures/minisgl-kv-calib/kv_calib_v1.txt --ctx 4096 --max-chunks 32 \
  --out /fixtures/minisgl-kv-calib/sidecars/<name>.safetensors --report <name>.report.json' \
gpu-lease -n 2 -- docker compose --profile run run --rm -v /home/pat/fixtures:/fixtures run
```
Gotchas that cost time: run compose **from the worktree** (the `run` service mounts `.` as
`/engine`); `minisgl-rdna4:lean` is stale — use `:comb`; the container writes as root (the tool
chmods its own outputs 0644, but anything written before that needs a `chmod` run); a model with a
large recurrent state (Laguna) needs `--max-running-req 2`.

## 5. What is left, in order

1. **MTP does not boot at the default memory ratio on the 35B — the recurrent-radix snapshot
   store is why.** `MODEL=qwen35b-awq SPEC=mtp TP=2` at `MEM_RATIO=0.80` dies in
   `_determine_num_pages` ("Not enough memory for KV cache after reserving recurrent state / draft
   model / CUDA-graph buffers"). Measured cause, by elimination: the snapshot store reserves
   **0.38 GiB** (cap=23 × 16.4 MiB, sized by ladder depth × max_running) against a post-weights
   budget of ~1 GiB, and `--no-gdn-radix` at the SAME 0.80 boots fine (74,272-token pool). At
   `MEM_RATIO=0.86` MTP boots with the store and runs **102.3 tok/s vs 90.4 no-spec at M=1**
   (+13%, accept 2.78 of 5). `serve.sh` already special-cases the memory ratio for `dflash` and
   explicitly says "MTP needs none of this" — that is now false. Fix: extend the case to `mtp`, or
   size the snapshot store out of what is actually left.
2. **Decide the headroom policy with a measurement.** Every scale here is a pure max (`amax/448`),
   so traffic wider than the fixture clips — gracefully (the kernel saturates; it does not NaN), but
   it clips. The Laguna cross-check shows two honest calibrations of the same model disagreeing by
   up to 1.59× on V, which is an argument for a margin (`amax * k / 448`); the cost of a too-large
   scale is only subnormal flush. Measure `k ∈ {1.0, 1.25, 1.5}` against the acceptance suite and
   the ctx probe before changing the formula. Do not just add a fudge factor.
3. **Re-calibrate when the traffic mix changes.** The sidecar records the fixture's size+sha256; if
   the box's workload shifts (new agent, new model family), rebuild the fixture and re-run rather
   than trusting a scale fitted to old traffic.
4. **Dead kernel fork to delete** (`rdna4-hip-kernels`): `tail/tail_rocm/tail_kernels_hip.hip` is an
   OLD copy of the store path — per-tensor scale, int64 `out_loc`, and an `f32_to_e4m3` **without**
   the 448 saturation the live file has. `build.toml` compiles only `tail_kernels.hip`, so it is
   inert today, but it is exactly the copy-paste fork KERNEL_CORE_POLICY forbids and it reads as if
   the NaN bug were still live.
5. **`docker-compose.yml` still defaults to `minisgl-rdna4:lean`**, which predates the fp8 work.
   `minisgl-rdna4:comb` is the validated image; nothing is served from the merged code until `lean`
   is repointed or `MINISGL_IMAGE` is set (every command here sets it).

## 5b. ReplaySSM under speculative decode — what is and is not known

Asked and answered while chasing the MTP boot failure above, because both features spend the same
budget:

* **It was never A/B'd under spec.** `tools/replay_serve_ab.sh` — the driver that produced the
  "+2.1% at M=4, wash at M=1–2" result — boots both legs with `SPEC=none` (line 52). Re-running it
  under spec needs a kernel package built WITHOUT `gdn_decode_conv_gated_replay`, and the two images
  it used (`minisgl-rdna4:replayssm` / `:replayctl`) no longer exist on this box.
* **By construction it cannot help a spec step.** The replay rung lives only in
  `GDNLayer.forward_decode`. A spec step runs `forward_verify`, which FLUSHES the ring before its
  varlen kernels and INVALIDATES it after (`gdn/layer.py:504,529`) — the ring's whole benefit is
  deferring the `ssm_state` read-modify-write across consecutive decode steps, and a verify ends
  that window every time. Confirmed engaged-at-capture only: under `SPEC=mtp` the boot log shows
  `gdn_decode_conv_gated_replay` at plain-decode graph capture (01:39:14) and
  `causal_conv1d_fwd_verify`/`gdn_prefill_verify` at spec-verify capture (01:39:25), and
  `[hip-engage]` fires once per op, so it cannot distinguish per-step use afterwards.
* **Its VRAM cost is NOT what breaks spec.** The ring is `L*(K+V)/(V*K)` of `ssm_state` — ~12.5% at
  L=8, i.e. ~0.014 GiB of the 0.11 GiB "GDN/CCA recurrent state" reservation. The 0.38 GiB
  recurrent-radix snapshot store is 27× bigger and is the thing that tips MTP over (§5.1).

So: no measured effect, a mechanism that says "inert on spec steps", and a cost too small to be the
spec blocker. If it matters enough to settle, the honest experiment is a control kernel package
without the replay op, driven at `SPEC=mtp MEM_RATIO=0.86`.

## 6. Rules that bind this work (unchanged, still true)

* **Never calibrate mid-serve.** One descale must undo every store ever written under it, and a
  captured graph provably picks up a late in-place descale write (`max|Δ| = 4.85e-01`). Scales are
  resolved in `Engine.__init__`, between pool construction and capture, and never touched again.
* **Calibrate against the bf16 cache** (`MINISGL_KV_FP8=0` + `MINISGL_KV_FP8_CALIBRATE=1`, set before
  `minisgl` is imported). Measuring amax through an fp8 cache feeds fp8 error into every later
  layer's activations.
* **Stochastic rounding stays deleted** on this store: write-once storage has no bias to accumulate,
  and SR paid the √2 variance penalty for nothing (median 1.44–1.48× worse, 2.13× kernel time). The
  same technique is an 8× win on the GDN recurrent state — recurrence vs storage is the distinction.
* RULE 4: not done until it boots at the served TP with graph capture and answers.
* Source isolation: per-task worktree, never mount the shared `$PWD`. GPU: `gpu-lease -n <cards>`,
  let it block. Cross-card: GPU 0 is a 9070 XT and GPU 1 a 9070 — the same case yields different
  error digits, so use the margin against tolerance as the signal.
* Do **not** pass `-p <project>` to compose under `gpu-lease` (the arbiter looks the container up by
  `COMPOSE_PROJECT_NAME=lease-<name>`; overriding it makes it release the lease under a live
  container). A TTFT probe must match **any** delta chunk — GLM streams `reasoning_content` first.
