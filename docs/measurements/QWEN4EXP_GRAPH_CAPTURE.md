# Graph capture on the offloaded 48-layer TP=2 qwen4_exp serve

`[CAPTURE-2026-09-04]` — cards 0 (RX 9070 XT) + 1 (RX 9070), image `minisgl-rdna4:m1b-20260903`,
worktree `/home/pat/code/minisgl-rdna4-offload` (branch `feat/weight-offload`). GPU lease waived for
this task; both cards held by one process per run.

---

## 0. Verdict up front

**Graph capture is DISCHARGED on the full 48-layer TP=2 offloaded serve.** The repo's merge
requirement ("eager-only is never done") is met on the qwen4_exp path: decode graphs are captured
per bucketed batch size, they replay, and the replayed greedy token ids are IDENTICAL to eager's at
both captured widths (bs=1 and bs=2) at full depth, measured as a one-boot A/B inside the same
process against the same weights, KV pool and pinned arena.

**What it buys: +2.5%, and that is the honest number.** 81.87 → 79.86 ms per decode step
(×1.0253), five interleaved repeats per leg, non-overlapping sample ranges, identical to four digits
on both ranks. It buys little because **this decode is PCIe-bound, not launch-bound**: 37 of 48
layers read their experts out of pinned host memory, 568.32 MB/token/rank, and at card 1's measured
Gen4 x8 rate of 14.48 GB/s that read alone is 39.25 ms — 49% of the step. Capture removes launch
overhead; there was ~2 ms/step of it to remove. A second, independent run at 44 layers with the same
interleaved method measured 2.293 ms/token saved, so the *magnitude* of the saving replicates.

> **`[ENDGAME-2026-09-04]` CORRECTION — the ABSOLUTES above are retired; the DELTA stands.**
> Two later boots of this exact operating point, instrumented with GPU events instead of a
> wall-derived divide, measure the captured decode step at **62.36 / 62.57 ms**, not 79.86, and the
> capture saving at **2.80 / 3.78 ms/step** rather than 2.02. The 79.86 figure divides a run that
> contains 8 prefill forwards by 119 decode steps; this document's own §2.4 records the same
> instrument producing **41 / 59 / 80 ms** for one config across three boots, which is why the
> conclusion here was correctly stated as a *delta* and only the absolute is wrong.
> The 39.25 ms PCIe arithmetic is also superseded by measurement: the host-expert read is
> **41.72 ms** on rank 1 (card 1, 13.53-13.62 GB/s achieved = 94% of its 14.48 GB/s ceiling) and
> only **21.78 ms** on rank 0 (card 0, 26.0 GB/s) — the step is gated by the slower link, exactly as
> predicted, but the two ranks are 1.92x apart and rank 0 pays the difference as **20.6 ms/step of
> idle inside the MoE all-reduce**, which nobody had costed. Full attribution:
> `docs/measurements/QWEN4EXP_ENDGAME.md` §3. Raw: `attrib/decode_attrib_run{1,2}.json`.

Three earlier speedup figures produced by this harness are **retracted** — see §2.4. Nothing in this
document quotes them.

| Question | Answer | Evidence |
|---|---|---|
| Capture engaged at 48/48 layers, TP=2, arena live? | **YES** | `capture/L48_tp2_capture_r3_chunk750_dev11_floor8.json`, `offload_serve_tp2/L48_tp2_capture_ab.json` — `graph_capture_engaged=true`, `cuda_graph_bs_captured=[1,2]` on both ranks |
| Replay computes what eager computes? | **YES**, identical greedy ids at bs=1 AND bs=2 at full depth | `replay_matches_eager = "bs=1: IDENTICAL over 12 greedy ids in 1 row(s) \| bs=2: IDENTICAL over 24 greedy ids in 2 row(s)"` |
| Arena pointers survive capture? | **YES** | `verify_after_capture_fired=2`, `arena_torch_fallbacks=0`, `seam_pointer_checked=true` |
| Throughput delta | **×1.0253 per decode step** (MEASURED) | `L48_tp2_capture_ab.json` |
| Output still coherent at full depth, capture ON | **YES**, checkpoint sampler | `ab_quality_captured` — clean `</think>`, correct Rayleigh answer |
| Failures left open | **1 per rank** (a 1-token output-accounting asymmetry) | §2.5 — left RAISING, not papered over |

---

## 1. The measured operating point WITH capture

Landed in `tools/serve.sh` as the `qwen4exp` arm (added `[CAPTURE-2026-09-04]`). A validated
operating point that lives only in a doc reads as a regression to the next person.

```
MODEL=qwen4exp TP=2 CONC=2 tools/serve.sh
  → --attention-backend hip            (NOT rdna4 — see §3.1; rdna4 raises on capture)
    --cuda-graph-max-bs 2              (GRAPH_BS follows CONC; captures buckets [1, 2])
    --memory-ratio 0.96
    --page-size 16
    --weight-offload-device-gb 8.1     (11/48 layers device-side, 8.06 GiB/rank)
    --weight-offload-gb 28             (37/48 layers host-pinned, 27.10 GiB/rank = 54.20 node)
    MINISGL_WEIGHT_ARENA_CHUNK_MIB=750 (exact multiple of the per-rank row set → ZERO straddle waste)
    MINISGL_WEIGHT_ARENA_FLOOR_GIB=9   (BOX property, not a model property — see §1.2)
    MINISGL_PLE_FILES / MINISGL_PLE_META_FILES  (the n-gram sidecar; the model does not boot without them)
```

### 1.1 What was actually measured, at what

| Term | Reference (EAGER, pre-capture) | Capture A/B boot | Full-depth capture boot |
|---|---|---|---|
| artifact | `offload_serve_tp2/L48_tp2_chunk750_dev11.json` | `offload_serve_tp2/L48_tp2_capture_ab.json` | `capture/L48_tp2_capture_r3_chunk750_dev11_floor8.json` |
| layers / TP | 48 / 2 | 48 / 2 | 48 / 2 |
| attention backend | `rdna4` | `hip` | `hip` |
| `cuda_graph_max_bs` | 0 (eager) | 2 → captured `[1,2]` | 2 → captured `[1,2]` |
| `memory_ratio` | 0.96 | 0.90 (harness default) | 0.90 (harness default) |
| host / device layers | 37 / 11 | 37 / 11 | 37 / 11 |
| arena pinned per rank | 27.10 GiB | 27.10 GiB (29,097,984,000 B) | 27.10 GiB |
| `arena_torch_fallbacks` | 0 | 0 | 0 |
| Stage B | 215.7 s, 49 chunks, 1140 keys | 214.0 s | 191.7 s |
| boot | 545.4 s | 613.8 s | 558.4 s |
| `kv_pages` | 8010 | 2850 | 2850 |
| `verify_after_capture_fired` | 2 | 2 | 2 |
| wall tok/s (not a decode number) | 11.85 | 11.693 | 11.456 |
| failures | 0 | 1 (§2.5) | 1 (§2.4) |

**`kv_pages` 8010 → 2850 is the MEMORY RATIO, not the cost of capture.** The reference boot ran at
`--memory-ratio 0.96`; both capture boots ran at the harness default 0.90, and `tools/serve.sh`
already documents that 0.96 is what yields 8,010 pages @ 196,608 B on this card. Attributing that
5,160-page drop to the graphs would be wrong. What capture's *own* VRAM cost is at 0.96 is
**NOT MEASURED** — the graph buffers here are small (`[max_bs, vocab]` fp32 logits at bs≤2 with
vocab ≈ 248k is ≈ 2 MB, plus one shared graph pool), but "small" is arithmetic, not a measurement.
**The `--memory-ratio 0.96` + capture combination has never been booted.** It is what the serve.sh
arm composes, and it is the one term in that arm carrying a projection rather than a measurement;
the serve.sh comment says so, and the first boot of that arm is the confirmation.

### 1.2 The four load-bearing terms are unchanged by capture

`device-gb 8.1` (12 device layers OOMs in Stage B, not at rest), `chunk 750 MiB` (rows are
400/200/100/50 MiB = exactly 750 MiB/layer/rank, so any rounding wastes the remainder of every
chunk), `memory-ratio 0.96`, and `FLOOR_GIB=9` (54.20 GiB pinned + the 12 GiB default floor = 66.20
does not fit in the ~62–65 GiB of `MemAvailable` the rank processes see; 54.20 + 9 = 63.20 does).
The full-depth capture retake ran at floor 8 — the artifact name records it. **The floor is a box
property, not a model property. Do not carry 8 or 9 to a box with more RAM**, and note that four
48-layer boot attempts during this work died in `HostArenaCapacityError` / `HostArenaSwapThrashError`
purely because ~10.4 GiB of tmpfs was squatting under `/tmp/claude-1000`. That failure aborts during
host pinning, long before any graph is recorded; it never touched capture.

---

## 2. The A/B: what capture buys, and what decode is bound by

Artifact: `docs/measurements/WEIGHT_OFFLOAD_2026-09-02/offload_serve_tp2/L48_tp2_capture_ab.json`.
ONE boot. Checkpoint sampler (temp 1.0, top_k 20, top_p 0.95, eos `[248046, 248044]`), `ignore_eos`
on the timed legs so both legs do equal work, 5 interleaved repeats per leg plus one untimed warm leg
per mode. The switch is `graph_runner.max_graph_bs = 0`, which makes `can_use_cuda_graph` and
`pad_batch` both fall through to the eager forward — same process, same weights, same KV pool, same
pinned arena, so the only variable is the capture mechanism. This satisfies the repo rule that an
A/B baseline must be the old *code path*, not an emulation of it.

### 2.1 The number

| | eager | captured | ratio |
|---|---|---|---|
| **ms per decode step (THE NUMBER)** | **81.87** | **79.86** | **×1.0253** |
| wall tok/s samples | 12.109, 12.214, 12.214, 12.245, 12.300 | 12.569, 12.624, 12.628, 12.639, 12.678 | — |
| wall tok/s median | 12.214 | 12.628 | ×1.0339 |
| decode steps / prefill forwards per rep | 119 / 8 | 119 / 8 | equal |

The two sample ranges do **not** overlap, so ~2.5% is outside run-to-run noise on this box. Both
ranks agree to four digits (rank 0 ×1.0253, rank 1 ×1.0252).

**Why the headline is normalized per decode step and not per emitted token.** Both legs recorded
exactly the same 8 prefill + 119 decode forwards, but the captured leg reported 120 emitted tokens
and the eager leg 119 under `ignore_eos` with `max_tokens=120`. That asymmetry is in output
accounting, not in the work timed (§2.5), and the per-step ratio is immune to it. **1.025 is the
number I stand behind; 1.034 is its upper bound.**

### 2.2 Provenance — four independent channels, all gated

The repo rule is that an A/B must assert provenance or "green" means new-vs-itself. Everything below
is in the artifact under `ab_provenance`:

1. `GraphRunner.replays` — captured **595**, eager **0**.
2. `Engine.eager_decode_forwards` — captured **0**, eager **595**.
3. Per-step phase tags recorded at the step that ran — captured leg `["decode_graph"]` exclusively,
   eager leg `["decode_eager"]` exclusively.
4. **The offload arm was live in BOTH legs**, recovered independently per leg. The MoE seam's
   resolve counter is host Python, so it does *not* run under replay: the captured leg re-entered it
   only on its 8 prefill chunks (296 = 8 × 37), the eager leg on all 127 forwards (4699 = 127 × 37).
   H = 37 host-placed MoE layers falls out of each leg separately and matches. A
   captured-but-never-replayed run fails this arithmetic.

Counters 1 and 2 are **always on and unconditional**, not env-gated — "graphs were captured at boot"
does not prove the decode steps replayed one (a batch wider than `max_graph_bs`, a prefill-shaped
step, or a scheduler that never reaches the decode lane all produce a run that benches exactly like
eager). One int add per decode step.

Engaged ledger identical and complete on both legs: `weight_offload.moe_resolve[device]`,
`moe_resolve[host]`, `stage_a_sealed`, `stage_b_chunked_load`, `verify_after_capture`.
**NOTE for whoever diffs ledgers next:** the ledger is a SET and saturates before leg 1, so a
per-leg set-diff is empty by construction on both legs and proves nothing. The counters are what
make the legs distinguishable. The repo's "diff the engaged ledgers per leg" rule needs a counter
channel here, not a set-diff.

**Instrumentation bias, stated rather than waved:** the resolve counter adds one dict update per MoE
layer per **eager** forward only (48/step, measured 1.63 µs/step = 0.081% of the 2020 µs/step net
saving). Immaterial but not zero, and it favours capture.

### 2.3 What decode is bound by — PCIe, not launch overhead

MEASURED inputs, arithmetic conclusion:

```
37 host layers x 10 routed experts x 1.536 MB   = 568.32 MB / token / rank
card 1 root port, MEASURED Gen4 x8              = 14.48 GB/s   (card 0: 28.93)
                                                => 39.25 ms/token, gating a lockstep TP=2 step
39.25 / 79.86                                   = 49% of the measured decode step
capture removes                                 =  2.02 ms/step = 2.5%
```

So the PCIe read of host-resident experts is ~20× the entire launch-overhead budget capture could
ever recover. This is the repo's existing finding restated on a new model: "bs=1 decode is
inter-kernel-gap bound" was already REFUTED for Qwen35B, and it is refuted here too — for a
different reason (there the deficit was in-kernel; here it is off-card).

**Honest gap.** The engine's own banner projects 7.5 ms compute floor + 39.25 ms host + 0.24 ms
device = 47.0 ms/step → 21.3 tok/s. Measured is 79.86 ms/step. The 32.9 ms difference is
**UNATTRIBUTED** — it is not measured to any term. The obvious suspects (the host gather is
synchronous inside the layer with no prefetch overlap; DMA below link peak; the per-expert Python
route on the critical path) are named in `WEIGHT_OFFLOAD_PLAN.md` §T8.3(ii) and none of them has a
trace behind it. **That gap, not launch overhead, is where the next throughput work is.**

> **`[ENDGAME-2026-09-04]` CLOSED. The 32.9 ms was never one thing, and most of it was the banner,
> not the engine.** Attributed in `QWEN4EXP_ENDGAME.md` §3, two boots, GPU events:
> * **~19 ms of it is `prior.compute_floor_ms = 7.5`**, a documented Phase-0 GUESS (the midpoint of
>   an unmeasured 5-10 ms bracket) standing in for a MEASURED **26.7 ms** of non-expert on-device
>   work. The projection was wrong, not the serve.
> * **~17 ms of it never existed**: the step is 62.36 ms, not 79.86. See the correction at the head
>   of this document.
> * The rest is real and newly named: **20.6 ms/step of rank-0 all-reduce idle** from the 1.92x link
>   asymmetry, and **10.15 ms/step of hyper-connection traffic** (of which only 2.64 ms is
>   bandwidth; the other 7.9 ms is ~1000 launch-bound kernels on 20 KB tensors).
> * Only **3.56 ms/step** of the 62.36 remains unattributed — MoE block arithmetic plus inter-kernel
>   gaps. Every named suspect above (no prefetch overlap; DMA below peak; Python route) is either
>   REFUTED or reclassified in ENDGAME §3.2: within a layer the transfer *is* the kernel (there is no
>   separate H2D copy at all) and card 1 runs the read at 94% of its link ceiling.

### 2.4 RETRACTED speedup figures from this harness

Three numbers this harness printed during the session are wrong and must not be quoted:

| Retracted | Where | Why it is invalid |
|---|---|---|
| **×8.61** ("444.8 ms/token saved") | `capture/L44_tp2_final_r2.json`, `throughput_capture_speedup` | Single-shot, non-interleaved. The eager leg timed 503.2 ms/token while the captured leg in the SAME boot and both legs of the previous boot sat at 58–61 ms/token — a co-tenant landed on a card mid-leg. The lease is waived for this task, so this is a live hazard, not a hypothetical. Superseded by the interleaved retake below. |
| **×1.6395** | `capture/L48_tp2_capture_r3_...floor8.json` | The legs generated **different token counts** (captured 128, eager 68 — the greedy eager leg hit EOS). The harness's own `check("throughput legs generated the same token count")` FAILED, which is the `failures=1` on that run. Comparing a 128-token run to a 68-token one is not a ratio. |
| **×50.66** (`ab_speedup_steady`, "1978.86 tok/s") | `L48_tp2_capture_ab.json` | An artifact of *where* the timer sits. `steady_*` is the host wall of `Scheduler._forward`; `g.replay()` is asynchronous, so the captured leg's `_forward` returns after enqueueing (0.51 ms) and the device work is absorbed at the next sync, outside the timed region and inside `residual_s`. 0.51 ms/step is physically impossible for a step that streams 568.3 MB over PCIe. Conservation checks out: 25.08 ms/step left `_forward`, 22.73 ms/token reappeared in the residual, net 2.02 ms/step is what the wall clock saw. |

**The corroborating retake** — `capture/L44_tp2_throughput_r2.json`, 44 layers, 3 interleaved
repeats, min-of-N: captured 58.886 ms/token vs eager 61.179 ms/token = **2.293 ms/token saved**
(×1.0389). Independent boot, different depth, same order of magnitude as the 48-layer A/B's
2.02 ms/step. A constant ms/step saving is exactly the shape a fixed launch-overhead removal should
have; a ratio is not.

**Absolute per-token times are NOT comparable across boots on this box.** The same configuration
produced 41, 59 and 80 ms/token across three runs. That is why the deliverable is a one-boot
interleaved A/B and why the saving is quoted as a constant ms/step.

### 2.5 The failure left open

`failures=1` per rank on the A/B run. Under `ignore_eos` with `max_tokens=120`, the captured leg
emitted 120 tokens and the eager leg 119, while both recorded the identical 8 prefill + 119 decode
forwards. The asymmetry is therefore in output accounting, not in the work timed. **UNEXPLAINED. It
is left RAISING rather than papered over.** Only rep 1's counts were retained, so it could not be
localised; the harness now records `*_tokens_per_rep` so the next run can.

### 2.6 Quality with capture ON, at full depth

Checkpoint sampler, never temperature 0 (the repo rule: greedy fakes degeneration that mimics a
quant bug). Captured leg on "Explain in three sentences why the sky is blue.": reasoning preamble,
clean `</think>`, then *"Sunlight contains all colors of the visible spectrum. Molecules in Earth's
atmosphere scatter shorter blue wavelengths more strongly than longer red wavelengths…"* (truncated
by `max_tokens`). The eager control in the same boot is equally coherent. The EOS-honoured quality
leg with graphs ON produced a complete, correct three-sentence Rayleigh answer in 90 tokens.

---

## 3. What capture required that was not obvious

### 3.1 Hard blocker 1 — the attention backend. `rdna4` cannot capture; `hip` can.

Every qwen4_exp run before 2026-09-04 booted `attention_backend="rdna4"`, whose
`init_capture_graph` / `prepare_for_capture` / `prepare_for_replay` all raise
`NotImplementedError("rdna4 cudagraph capture lands in Phase 4; run with --cuda-graph-max-bs 0")`
(`python/minisgl/attention/rdna4.py:885-894`). The capture-capable implementation is its **subclass**
`HIPAttnBackend` ("hip"), which is what `docker-compose.yml` defaults to and what every production
serve in this repo runs.

**This is not a kernel change.** With `MINISGL_ATTN_HIP=1`, `RDNA4Backend.forward` already dispatches
decode to the same `attn_decode.flash_decode_paged` op `HIPAttnBackend._forward_decode` calls, and
cold prefill to the same `attn_hip.flash_prefill`. The harness default moved to `hip`
(`--attention-backend rdna4` still reproduces a pre-2026-09-04 run exactly), and the serve.sh arm
carries `attn=hip`.

### 3.2 Hard blocker 2 — the PLE seam had no capture hook at all

`Qwen4ExpPLE.forward` reads `get_global_ctx().ple.batch` and **raises** when it is None — deliberately,
because silently skipping the block would drop the n-gram features. `GraphRunner._capture_graphs`
stages attn metadata and, for a hybrid, GDN/CCA/CAM state, and **nothing else**; there was no PLE
hook. So the capture-time warmup forward died before a single graph was recorded, and cudagraph
capture was **structurally impossible** for this model.

New file: `python/minisgl/ple/graph_capture.py` (`PLEGraphCapture`), wired into `GraphRunner` via a
new `ple=` constructor argument. **Nothing about the PLE block moved into the graph that was not
already there** — the device arithmetic was always shape-static (index_select / cat / four scaled
adds / index_copy_). The gap was purely that capture had no way to STAGE a batch.

* **capture** → `prepare_for_capture` stages `bs` rows on the reserved NULL slot 0 with a one-token
  EOS context. That is exactly the convention `GDNGraphCapture` uses for its state slots and exactly
  what `Scheduler._stage_ple` writes for a cudagraph PADDING row. The gathered embeddings land in
  `PLEEmbeddingSource.embeddings[:bs]` and the slot ids in `PLERuntime._slot_idx[:bs]` — both static
  buffers, both at offset 0, so the pointers the graph bakes are the ones every later replay
  refreshes in place.
* **after capture** → `after_capture` **DISCARDS** the staged batch. It must not be committed.
* **replay** → `prepare_for_replay` does NO staging. The scheduler already staged this batch in
  `_finish_prepare`, over `padded_reqs` (i.e. including the padding rows). What the hook does is
  assert the two facts a replay silently depends on: that a batch was staged with exactly
  `padded_size` rows, and that it landed at the SAME ADDRESSES the capture baked. The address check
  is an equality on `data_ptr()` for embeddings, slot index and conv state — deliberately not a
  shape/dtype check, because a refactor that allocates a fresh staging tensor would still match on
  shape and dtype and would still produce finite, plausible, WRONG text.

**`Scheduler._stage_ple` and `PLERuntime.commit_staged` did NOT move.** Host-side staging already sat
outside the captured region (`_stage_ple` in `_finish_prepare`, before the forward; `commit_staged`
in `_forward`, after it). The only gap was that capture had no way to stage.

**commit-exactly-once was PROVEN arithmetically, not assumed.** `PLERuntime` gained
prepare/commit/discard/commit_noop counters; the invariant is `prepares == commits + discards` AND
`commit_noops == 0`, and it is a gate in the harness:

* boot with capture: `prepares = len(graph_bs_list)`, `commits = 0`, `discards = len(graph_bs_list)`,
  `commit_noops = 0` — measured 3/0/3/0 at L4 (buckets `[1,2,4]`) and 2/0/2/0 at L48 (buckets
  `[1,2]`). The capturer's synthetic batches are DISCARDED, never committed, so they cannot put the
  n-gram history one pass ahead of the conv state.
* per-leg across the parity A/B: captured leg 12 prepares / 12 commits, eager leg 12 / 12,
  `commit_noops = 0` throughout.

`resolve_prefix_cache` still returns `'naive'` for any model with `ple_layer_ids`. Untouched.

### 3.3 A silent correctness bug that capture INTRODUCED, and the general seam that fixes it

`Qwen4ExpAttn.forward` asks `assert_dense_is_exact(max device_len over batch.reqs)` — the QSA
indexer-budget refusal. Below `indexer_budget` (2048) the checkpoint's sparse selection picks every
visible key, so this engine's dense causal attention is bit-equivalent; above it the two genuinely
diverge, and dense is not "slightly wrong" — it attends MORE, so the output stays fluent and the
substitution is undetectable from the text.

That check is **host Python inside `forward()`**, so under capture it runs exactly once — at capture,
over `dummy_req` rows whose `device_len` is 1 — and never again for the life of the process. A
request crossing the budget mid-generation would then quietly switch to a different, denser model at
token 2049 with no error anywhere. This is the same class as the recorded page-table-width bug and
the GDN `ReplaySSM` gate: **a shape-dependent decision asked once, at the wrong width.**

Fix: `GraphRunner` now holds `self.model` and calls `model.prepare_for_replay(batch)` immediately
before `g.replay()`, and `Qwen4ExpForConditionalGeneration.prepare_for_replay` restates the refusal
over the live batch. This is a **general seam on `BaseLLMModel`**, not a qwen4_exp special case: any
model with a host-side per-step guard inside `forward()` has the same silent failure the moment
capture is enabled. The check is deliberately NOT removed from `Qwen4ExpAttn.forward` — prefill and
any uncaptured decode never reach the replay hook, and that path is where the check has always lived.
No-op when the build has no full-attention layer (the GDN-only layer subsets the bring-up harness
runs).

### 3.4 The offload seam required NOTHING to move — and that is why it is a provenance channel

The pinned arena is capture-safe and was already measured so (P5b): it survives
`torch.cuda.graph.__enter__` and `empty_cache()`, pointers are stable, 24 replays gave one sha256
matching CPU ground truth. `moe_interpose.py` is explicit that "addresses are constants by the time
capture runs" — residency is zero-copy placement decided at boot, not caching.

Confirmed on the real serve: `verify_after_capture_fired = 2` (the gate at `weights/bake.py:651`
that proves the arena's pointers did not move across capture), `arena_torch_fallbacks = 0`,
`seam_pointer_checked = true`, and the engaged ledger after capture still carries all five
`weight_offload.*` marks.

The consequence used in §2.2: the MoE seam's *resolve* step is host Python that a graph replay does
not execute, so its counter falls to zero on captured decode steps. That is what let the A/B recover
H = 37 host-placed layers independently from each leg.

### 3.5 Parity is NOT bitwise, by construction — and this is pre-existing

`attn_decode`'s split-K policy is keyed on the page-table **WIDTH** (`max_blocks * block_size`). The
captured table is the full `aligned_max_seq_len`; the eager one is the batch's own `max_seqlen_k`.
Different width → different `num_splits` → a different fp32 reduction order, and at `num_splits == 1`
literally a different kernel. **This is the exact hazard the repo recorded and fixed at
61d96cf / 0972e387**, and it is shared with every captured model in this engine — it is not a
qwen4_exp regression, and it was not reintroduced here.

So the gate is *identical greedy token ids*, not bit-equality, and that is what was claimed and
measured:

| Configuration | bs=1 | bs=2 |
|---|---|---|
| TP=1, L4 | 12/12 identical | 12/12 identical |
| TP=2, L4 (both ranks) | 12/12 identical | 12/12 identical |
| TP=2, L44 offloaded (both ranks) | 12/12 identical | **DIVERGED at greedy token 5 of 6** |
| **TP=2, L48 offloaded, FULL DEPTH (both ranks)** | **12/12 identical** | **24/24 identical** |

**The L44 bs=2 divergence RESOLVED at full depth.** At 44/48 layers the model is a layer *prefix* and
is degenerate (request 0 loops one token, so logits are near-tied); the eager control at the same
depth was equally degenerate, so the divergence was a property of the truncated model, not of
capture. It was flagged OPEN at the time and it is now CLOSED by the 48-layer measurement — the same
gate, same widths, same prompts, identical ids.

Two other divergences, both correctly attributed away from capture:

* TP=1 vs TP=2 ids diverge (token 10 of 12 at L4) — but they diverge **identically on the eager
  leg**, so that is TP reduction order.
* Beyond ~12 steps, greedy legs drift apart (in one 48-layer run the captured greedy leg ran to 128
  tokens while eager hit EOS at 68). The repo already records that this serve is not bit-reproducible
  past ~32 tokens. **Do not gate capture parity on long greedy runs**; the harness's short parity
  gate and the sampled A/B are the right instruments.

Engine reproducibility was separated from capture's with a `--repro-probe` leg: N identical generates
per mode, `repro_engine_is_reproducible = true` on both captured and eager legs (4/4 and 6/6
identical). So "captured and eager differ" can never be confused with "the engine is nondeterministic".

### 3.6 The harness pipes to `tail`

`tools/run_offload_serve.sh` pipes; a pipeline's exit status is the last command's. That already hid
a SIGSEGV for this repo's entire history. Rank exit codes are checked explicitly in the harness's
own accounting (`failures` per rank, aggregated at the top level) — read `failures`, never the shell.

---

## 4. Still undischarged or deferred

**Capture-adjacent, and the honest edges of this result:**

| Item | Status |
|---|---|
| **`--memory-ratio 0.96` WITH capture** | **NOT MEASURED.** Both capture boots ran at 0.90. It is what the serve.sh arm composes, and its first boot is the confirmation. See §1.1. |
| **Capture above bs=2** | Buckets `[1,2]` only, because CONC=2 is the measured admission point. `GRAPH_BS` follows `CONC` in serve.sh, so raising concurrency raises capture coverage — but neither has been measured at this operating point, and a batch above the captured max runs FULLY EAGER. |
| **Prefill / spec-VERIFY capture** | Not done. Prefill is eager everywhere in this engine; spec-verify capture is moot while `--spec-algorithm mtp` is refused for this architecture. |
| **The 1-token accounting asymmetry** | Open, `failures=1`, left raising (§2.5). |
| **The 32.9 ms/step unattributed gap** | ~~Open (§2.3)~~ **`[ENDGAME-2026-09-04]` CLOSED** — attributed in `QWEN4EXP_ENDGAME.md` §3. Two-thirds of it was the banner's own `compute_floor_ms = 7.5` guess against 26.7 ms of measured non-expert work; the step is 62.36 ms, not 79.86. Residual after attribution: **3.56 ms/step**. Both suspects named here are refuted: within a layer the transfer *is* the kernel (no separate H2D copy exists) and card 1 reads at 94% of its link ceiling. |

**Model features, unchanged by this work** (all four still in the `UNIMPLEMENTED` banner the build
logs once, and all four still raise at the point of use):

| Item | Status |
|---|---|
| **Vision tower** (`model.visual.*`) | Not planned. Text-only serve, as for every other multimodal checkpoint here; the loader counts 333 skipped vision tensors. |
| **MTP speculative head** (`mtp.*`) | Deferred (plan T8.1). Fused expert tensors, its own hyper-connections, seeds from the `hc_count`-wide PRE-mixer stream. `ModelConfig.from_hf` REFUSES `--spec-algorithm mtp` for this architecture rather than half-building the Qwen3.5 head. |
| **QSA sparse indexer above `indexer_budget` = 2048** | Deferred (plan T5). Below the budget the selection is the identity and dense attention is bit-equivalent; above it the engine **raises** — now on the captured path too (§3.3). Caps usable context at 2048 of 262,144. |
| **Dense-linear offload** | Not planned (`WEIGHT_OFFLOAD_PLAN.md` A1.5). Only the MoE expert tier is placed. |

**The e4m3 scale policy** — MEASURED **4.37e-04 max weight rel-err** on the NVFP4 path. It is legal
under the kernel core policy as a **WLoad policy on the existing shared core**, not a new kernel, and
it is estimated at **~4 days**. It is **blocked on templating two hand-written non-WLoad kernels**:
`moe_gemm1_silu_alds_kernel` at `rdna4-hip-kernels/fp8_wmma/fp8_wmma_rocm/moe_kernel.hip:299` and
`moe_gemm1_silu_ashuffle_kernel` at `:450` — both take `w_packed`/`w_scales`/`w_zeros` directly, i.e.
weight decode is hardcoded into the body rather than expressed as a `WLoad`. This is exactly the
consolidation backlog `KERNEL_CORE_POLICY.md` already carries; the fix is to extend the shared core,
not to fork a third kernel.

**`ct_sign_cross_rank` — UNTESTED, NOT CLEARED.** Every run reports
`"not exercised: 0 CT containers in this checkpoint"` (and `ct_sign_decisions = 0`). The cross-rank
sign-convention check has therefore never executed against real data on this path. That is an absence
of evidence, not a pass, and it must not be read as one. `tests/core/test_ct_sign_convention.py`
covers the logic synthetically; a checkpoint that actually ships CT containers would be the first
real exercise.

---

## 5. Corrections to earlier documents `[CAPTURE-2026-09-04]`

Applied in place, each tagged in the source file:

1. `docs/measurements/QWEN4EXP_FIRST_RUN.md` §5 — the **Graph capture** row said "UNDISCHARGED …
   never attempted for this model … serve eager until it has been". **Now DISCHARGED**; the row
   points here.
2. `docs/QWEN4EXP_BRINGUP_PLAN.md` §4 (Deferred table) and the T6 tranche — same claim, plus
   "T6.2 (PLE staging) … unverified". **T6.1 and T6.2 are DONE and verified**; T6.3 (selected-page
   set in static buffers) stays open with T5, since there is no indexer to capture yet.
3. `docs/QWEN4EXP_BRINGUP_PLAN.md` §Key findings item 6 — "Deliberately deferred … graph capture,
   the offload arena, TP=2". All three of those have since landed; only vision, MTP and the QSA
   indexer remain.
4. `docs/WEIGHT_OFFLOAD_PLAN.md` §Verdict item 5 — "**Still not discharged: graph capture.** No GPU
   work of any kind has run." Both halves are now false.
5. `python/minisgl/models/qwen4exp.py` `UNIMPLEMENTED` — the banner declared cudagraph capture "has
   not been exercised for this model. Serve eager until it has been." Corrected in the same change
   that made capture work: the entry now scopes to **prefill / spec-verify** capture and records
   that DECODE capture is implemented and exercised (2026-09-04, `hip` backend).
6. `tools/serve.sh` — the 48-layer row of the chunk table and the operating-point block described an
   **eager** serve. A `qwen4exp` arm now carries the whole point, with the capture terms
   (`attn=hip`, `CONC`/`GRAPH_BS` cap 2) and a `[CAPTURE-2026-09-04]` dated comment.

---

## 6. For the record — the AWQ INT4 checkpoint is reclaimable disk

`/home/pat/.cache/hf-awq` is **176 GiB** and turned out **NOT to be needed**: NVFP4 fits at TP=2, and
every measurement in this document is on the NVFP4 checkpoint (`/home/pat/.cache/hf-q4e`, 78 GiB,
plus the PLE sidecar `/home/pat/.cache/hf-ple`, 49 GiB). `/home` is at **97% (69 GiB free of 1.9 T)**,
so this is the single largest recoverable item on the box.

**Caveat before deleting:** the e4m3 scale-policy work above targets the NVFP4 path, not AWQ, so it
does not need this checkpoint either — but if that work turns into a precision A/B that wants an INT4
reference, the calculus changes. Nothing currently planned needs it.
