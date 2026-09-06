# M1-A Status Report — Weight Offload

**Date:** 2026-09-03 · **Worktree:** `/home/pat/code/minisgl-rdna4-offload` · **Branch:** `feat/weight-offload`
**Supersedes, where they conflict:** `docs/WEIGHT_OFFLOAD_PLAN.md` §2, §5.1, §5.3, §5.6, §7, §10, §11
and `PHASE0_REPORT.md` §3.4. Plan edits applied in place, tagged `[M1A-2026-09-03]`.

**One paragraph.** Both gate probes came back decisive and both went the cheap way: the miss curve is
**LINEAR**, so per-expert placement is dead for good and layer-granular is final; and the pinned-host
mechanism **works through torch**, so the C++ `from_blob` extension and its +3 days are not incurred.
M1-A landed as ~10.1 k lines of new `python/minisgl/weights/` plus ~790 lines of edits to seven existing
files, with 722 CPU-only tests green. It is **not finished**: nothing has ever touched a GPU, graph
capture is undischarged, dense linears are not planned, Stage B does not exist, and one sizing defect —
the arena reserves anonymous headroom over a never-straddle allocator — inflates the device tier the
target model needs from **~3.9 GiB/rank to 11.06 GiB/rank on a 16 GiB card**, which is the difference
between the target checkpoint booting and not. Fix that first in M1-B.

---

## 1. What the gate probes decided

### P2′ — mixed-media grouped MoE decode-GEMV, explicit two-stack pointers, both cards

**Verdict: LINEAR. Per-expert placement is not worth resurrecting. Layer-granular is FINAL.**

| Measurement | Card 0 (RX 9070 XT, Gen5 x8) | Card 1 (RX 9070, Gen4 x8) | Pure-linear reference |
|---|---|---|---|
| `cliff_index`, layer total | 0.090 | 0.094 | 0.100 (band: LINEAR ≤ 0.25) |
| `cliff_index`, all 6 measurements | 0.085 – 0.095 | | |
| effective miss concurrency `W` | 0.72 – 0.82 | | ~1 = fully serialised |
| marginal cost of 1 host expert | 0.0825 ms | 0.1672 ms | = one 2.338 MiB granule at 29.7 / 14.66 GB/s |
| that as % of measured host BW | 102.8 % | 101.3 % | |
| `per_expert_gain_over_layer_granular` **peak** | 1.063× (h=0.75) | 1.050× (h=0.90) | |
| **same, at the reachable h≈0.25** | **1.013×** | **1.009×** | |

The `h¹⁰` all-resident-layer cliff that motivated abandoning per-expert placement is **REFUTED** — a
layer is not gated by its slowest workgroup, misses cost additively at full PCIe rate. But the
conclusion is unchanged and now rests on a *stronger* argument: on a linear curve, per-expert and
layer-granular are **provably equal at equal byte budget**, so per-expert placement has nothing to win
rather than being blocked by a cliff. It buys ~1 % at the operating point 16 GiB of VRAM actually
permits, and at most 5–6 % anywhere on the curve — in exchange for a route change, a `slot_of`, a
second stack in the hot path, and a change to two shared HIP cores.

**Plan §11 unknown #2 — "2× vs 0×, the top unknown in the document" — resolves to ≈ 0×.** No M2 day
should be spent on per-expert placement. `stacks.ExpertStackTable`'s docstring carries the exact HIP
change (one `TwoStackWLoad<Base>` WLoad policy, two files, both P2′ driver defects recorded) should
anyone ever need it; nothing was forked.

Consequence for the M1 model: layer time is `t_device + (1−f) × bytes/expert ÷ host_GB_s`. No cliff
term, no concurrency term, no residency-probability term. **Card 1's slope is exactly 2× card 0's** —
every TP=2 host ceiling is card-1-gated at 14.48 GB/s, never 28.93, never the average.

*Caveat:* synthetic shape (E=512, top_k=10, tokens=1, hidden=2048, inter=768, int4+zp, 2.338 MiB
granule), bs=1 only, distinct-expert routing, two GEMV arms in isolation, no graph capture. The
linearity argument is bandwidth-structural and should carry; the absolute ms will not.

*Artifact defect to know about:* the probe's own `decide()` prints "per-expert placement is worth
building" from the `cliff_index` band alone, ignoring the gain number. `p2prime.md` still carries that
wording. **Read the gain, not the label.**

### P5b — torch over `hipHostMalloc(Mapped|Portable)` + `hipHostGetDevicePointer`, in-image, under capture, both cards

**Verdict: GREEN, 4/4 legs, all six gate conditions on both physical cards. The ~30-line ctypes route
is sufficient. The C++ `torch::from_blob` extension and its +3 days are NOT incurred.**

- Placement proven three independent ways with no reliance on any location field or return code: VRAM
  delta **0.000× size** on all four legs (both `hipMemGetInfo` and per-BDF `amdgpu mem_info_vram_used`);
  `MemAvailable` delta 0.991–1.020× size; pages CPU-readable **and writable** via a syscall probe
  against a real file (the Phase-0 fake-host VMM pages gave `EFAULT` here).
- `t.data_ptr()` == the `hipHostGetDevicePointer` address exactly, under `_cuda_customAllocator` +
  `MemPool` + `use_mem_pool`. **0 `hipMalloc` fallbacks in every leg.**
- Arena survives `torch.cuda.empty_cache()` (pointer stable, contents intact, 6 reps) and
  `torch.cuda.graph.__enter__` — i.e. `engine/graph.py:314` needs no special-casing.
- Free callback fires **zero** times for a live pool block and zero for a cached/dropped one.
- 24 replays → **one** sha256, matching CPU ground truth; a separate 25-replay varying-input loop was
  host-verified byte-exact every rep, proving each replay genuinely re-reads host pages.
- `expandable_segments` verified LIVE via `memory_snapshot().is_expandable`, with the two plain legs as
  a differential control.

| | Card 0 | Card 1 |
|---|---|---|
| torch-level host-arena read | **26.61 GB/s** | **13.868 GB/s** (first ever taken on card 1) |
| vs its own copy-engine reference | 0.93× (28.69) | 0.97× (14.33) |
| vs P1's hand-written tiled kernel | 92 % of 28.93 | 96 % of 14.48 |
| ratio card0/card1 | **1.92×** | matches Gen5 x8 vs Gen4 x8 exactly |

**Two traps this closed, both must be respected in code:**
1. `hipPointerGetAttributes` reports `memory_type = 1 ("Device")` for the **real host arena** — the
   exact inverse of the Phase-0 trap where the query echoed "Host" for VRAM. **That query is unreliable
   in both directions and must never be a residency check.** `hipmem.py` deliberately does not bind it;
   `PinnedWeightArena.owns_pointer()` (pure arithmetic against the returned pointers) is the only
   trustworthy test.
2. `amdgpu mem_info_gtt_used` stayed at exactly 0 across all four legs (`hipHostMalloc` pins ordinary
   anonymous pages via userptr, which is not GTT-accounted). **Any host-tier telemetry must use
   `MemAvailable` or its own accounting, never the per-card GTT counter.**

*Caveats:* tested at 688 MiB, not 30–55 GiB — allocation time and failure behaviour at scale remain
P3b's numbers, not this probe's. Capture was one elementwise op over a 4 MiB view, **not** a
full-forward serve capture.

---

## 2. What actually landed in M1-A

`python/minisgl/weights/` — **10,147 lines, all new**, plus 788 inserted / 34 deleted across seven
existing files. 722 weight-offload tests, all green; **717 of them require neither a GPU nor torch**.

### New modules — status per file

| File | Lines | State | What is real |
|---|---:|---|---|
| `chunk_plan.py` | 511 | **DONE** | Pure-integer chunk/bump arithmetic. Forward-only next-fit, never straddles a chunk. `headroom_chunks()` with the sound packing bound. `digest()`. Default chunk 2 GiB (P3b's demonstrated size), hard-refuses past 4 GiB. |
| `host_capacity.py` | 429 | **DONE** | `/proc/meminfo` + `/proc/vmstat` + **cgroup v2/v1** (the serve image is a container; `/proc/meminfo` is not namespaced). 12 GiB floor, charge every local rank, fail closed when unreadable, never auto-shrink. `SwapTripwire` on `pswpout`. |
| `hipmem.py` | 333 | **DONE, UNEXECUTED** | ctypes binding, missing symbol raises. Rule-R1 `freeze()` with a depth-counted teardown window. Deliberately binds neither the VMM entry points nor `hipPointerGetAttributes`, reasons recorded inline. **Never run against a real card.** |
| `pinned_arena.py` | 1114 | **DONE, UNEXECUTED** | `NEW→RESERVED→ATTACHED→POPULATED→CLOSED`. `reserve()` is pure (zero HIP calls) so an infeasible plan aborts in ms. Per-chunk `MemAvailable` + swap check, device-issued first touch, rollback on every failure path incl. the swap abort. Structural chunk check (base alignment + pairwise disjointness) that runs even with the data self-test off. Fingerprint resweep. `owns_pointer`, `carve_digest`. |
| `torch_pool.py` | 326 | **WRITTEN, NEVER EXECUTED** | The P5b-validated `_cuda_customAllocator` + `MemPool` + `use_mem_pool(device=...)` plumbing, permanent anti-GC anchoring, no-op free callback, counted fallback that **refuses to `hipMalloc` mid-capture**, `assert_clean()` merge gate. Faithful transcription of a validated probe — but torch is not importable on this host, so it is unproven. |
| `granule.py` | 1248 | **DONE** | The one format-agnostic walk. Answers "what is one expert" for all nine formats + dense. Contiguity-gated byte-range dedupe, declared-shared verification, decode-**policy** recording (`_ct_sign`), `binding_fingerprint`, capture guard. |
| `placement.py` | 571 | **DONE**, torch-free | `LayerWeights` / `plan_layer_granular` / `OffloadPlan.digest()` / `distinct_experts` / `project_plan` / `sweep_device_fraction`. `ep_local_top_k` is the one implementation of the EP top-k rule. |
| `prior.py` | 197 | **DONE** | Every Phase-0 constant with per-field provenance. `slow_host_gbps(n)` returns the **slowest** of n cards. `project_step` refuses to run on a non-linear prior. |
| `sizing.py` | 649 | **DONE** | Config→bytes. Dispatch-order-faithful scheme resolution, closed-form per container, **meta-device build** as the primary path, `post_load_delta_bytes` (CT symmetric zeros are resident-but-not-granule; MXFP4 E8M0→fp16 widening). An unrecognised format degrades to the meta model instead of raising. |
| `plan.py` | 1354 | **DONE** | `resolve_weight_plan(config, model=...)`. Observed path (walk the built model) preferred; config transcription is the pre-build fallback. Capacity charged in **whole pinned chunks**. `required_device_bytes` walks the planner's own order. `agreement_digest` + `assert_rank_agreement` over the TP CPU group. |
| `moe_interpose.py` | 1239 | **DONE** | The seam. `discover_moe_layers` uses the **state_dict** path grammar (`model.layers.0.mlp.experts`), matching the plan. `bind_plan` enforces planned ⊆ discovered. Ordered bake: refuse-before-any-byte, validate-all-then-copy, one device-scoped sync, chunked **bitwise** read-back, alias-preserving rebind with read-back through `granule._lookup`, weakref leak proof over every alias. |
| `bake.py` | 1096 | **DONE, UNEXECUTED** | `StageASession`: `begin → attach → note_loaded → bind → seal → verify_after_capture`. Every transition asserts its predecessor; a disabled session accepts the identical sequence and does nothing. Rank-symmetric failure barrier (`_run_staged`) so one rank's abort cannot hang its peer for the 7-day gloo timeout. |
| `accounting.py` | 520 | **DONE** | Torch-free four-check ledger: host arena costs 0 device bytes (the Phase-0 trap detector), bake moved what the plan priced, device tier == measured granules, torch released the originals. Clamped `corrected_model_memory`. |
| `config.py` | 105 | **DONE** | The single place arena env is read. Deliberately provides **no on/off knob** (§6.2). |

### Edits to existing files

| File | What |
|---|---|
| `layers/moe.py` | `MoELayer._weight_offload = None` as a **class** attribute (stays out of `vars(self)`, so `state_dict`/`post_load`/the granule walk are untouched) + two lines in `forward` right after `w13, w2 = self.gate_up_proj, self.down_proj`, before the EP branch. All seven quantized containers declare `(ExpertContainer, BaseOP)` + `self._num_experts`. `post_load` now cross-checks the w13/w2 decode policy unconditionally. |
| `layers/linear.py` | `_LinearTPImpl(ExpertContainer, BaseOP)` with `_granule_dense = True` — dense presents the **identical** four-method surface, same function objects. |
| `quant/method.py` | `_ct_packed_is_uint4b8` (first 65536 int32 — a *prefix* of one column) replaced by `ct_packed_sign_convention` + `apply_ct_sign`: element-width-derived byte view (works on uint8-packed int4, not just int32), 512 block-spread runs over the whole stack, raises on an ambiguous histogram, decision stored as `_ct_sign` for Stage B to thread. |
| `engine/engine.py` | `StageASession.begin(..., model=self.model)` after the meta build; `attach()` before `load_state_dict`; `note_loaded/bind/seal` strictly between `post_load()` and `_determine_num_pages`; `verify_weight_arena_after_capture()`; `_weight_offload_device_budget`; `_tp_min_num_pages` (MIN all-reduce — `num_pages` was never cross-rank reduced); the clamped model-memory correction and the `KV sizing:` annotation. |
| `engine/config.py`, `server/args.py` | `weight_offload_device_gb` / `weight_offload_gb` + `--weight-offload-device-gb` / `--weight-offload-gb`, both in **GiB**. |
| `scheduler/scheduler.py` | `_prefill_budget_now` subtracts `weight_arena_torch_slack_bytes()` (arena segments are reserved-not-allocated and **not** reusable); post-capture arena gate after the four spec captures. |
| `pyproject.toml` | `gpu` pytest marker + `-m "not gpu"` in default addopts, so a bare `pytest` can never lease one of the two shared cards. |

### Stubbed, missing, or not real — be clear about these

- **Nothing has run on a GPU.** Not one line of the HIP path — `hipHostMalloc` through this binding, the
  fingerprint self-test on real pages, `hipMemsetD32` over a whole chunk, the MemPool wiring, the bake,
  any of the four accounting checks. 5 tests are GPU-marked and **none has ever executed**.
- **Never pinned at scale.** Largest arena exercised is ~6 MiB of fake ctypes memory. 30–34 GiB
  allocation time and failure behaviour are P3b's numbers, not this code's.
- **`torch_pool.py` is unexecuted** — torch does not import on this host (`libmpi_cxx.so.40`).
- **Engine import is unverified.** Both `minisgl-rdna4:lean` and `:lean-pytest` fail at
  `layers/_tail_hip.py:37` (`tail_hip has no attribute silu_and_mul` — the mounted tree is newer than
  the baked kernels). The `engine.py` / `scheduler.py` hunks are ruff-clean, AST-tested and reviewed,
  **not import-tested.** Fix the image/source skew before anything else in M1-B.
- **Dense linears are not planned.** `plan.py` and `attach_seams` enumerate `MoELayer` only. The
  mechanism below is already general (`GranuleSpec(num_experts=None)`, `_LinearTPImpl` is an
  `ExpertContainer`); what is missing is enumeration + a dense seam. Today a dense checkpoint gets an
  explicit warning saying so instead of a silent "nothing to do". **This is a merge gate (A1.5) and it
  is not satisfied.**
- **No Stage B.** The target checkpoint still OOMs during `load_state_dict`, before the bake runs.
- **No metrics.** No `weight_arena_host_bytes` / `_device_bytes` / `expert_layer_miss_rate`. The repo's
  five-hop plumbing rule applies or they export a flat zero.
- **No `tools/serve.sh` table entry, no `[serve]` banner line, no compose env.** There is no measured
  operating point to land yet.
- **No correctness harness.** A1.0–A1.5 (eps0, identity, permuted-mirror, desync must-fail matrix,
  dense parity) are not written. `ExpertStackTable.permuted()` exists as the ledger half of A1.2; the
  harness does not.
- **No cross-rank digest comparison beyond the plan.** `assert_rank_agreement` covers the TP group
  only; a DP-replica divergence is undetected, and `carve_digest()` is computed and logged but never
  diffed between ranks.
- **`ArenaMemPool.assert_clean()`'s byte arm is on** (wired into `seal()`), but the post-capture
  re-gate has never had a real capture to gate.

---

## 3. Graph-capture status — per repo rule, this is unfinished work

**THE MANDATORY GRAPH-CAPTURE GATE IS NOT DISCHARGED FOR THIS FEATURE.** No capture of any kind was
exercised in M1-A; no GPU work was run at all.

| Piece | Capture status |
|---|---|
| `chunk_plan`, `host_capacity`, `prior`, `sizing`, `plan`, `placement` | **N/A and provably so.** Boot-time integer arithmetic. No tensor, no stream, no kernel, no host sync, nothing reachable from a captured region. Cannot break capture; proves nothing about it. |
| `granule` | **N/A.** Boot-time descriptor. Has an active `_assert_not_capturing()` guard on `derive_granule_spec` / `expert_slice` / `stacked_tensors` (it launches `torch.equal` and host-syncs, both illegal mid-capture) — the guard is unit-tested by monkeypatching the two `torch.cuda` predicates, **never against a real capture**. |
| `moe_interpose` seam (`resolve()`) | **Capture-legal by construction, UNPROVEN.** Two `is` comparisons and a tuple return; no allocation, no launch, no side stream, no host sync, no fence. Runs at capture time only. `freeze()` re-reads the containers and refuses a rebind after bind, so a late repack cannot silently bake a stale pointer. |
| `pinned_arena` | **Address-stable by construction, UNPROVEN.** Immutable and frozen after `seal()`; `close()` refuses once frozen without `force=True` (a captured graph bakes the pointer and never re-resolves it). |
| `torch_pool` | **The riskiest piece, and entirely unexecuted.** `_capturing()` probe, `use()` refuses entry during capture, `_fallback` refuses `hipMalloc` mid-capture and returns NULL with a named stderr line. All guards written against real torch semantics, none run. |
| `bake.verify_after_capture()` | **Written, never fired.** Re-reads `(alloc_events, served_bytes)` and re-runs `assert_clean()` after capture. Called from the end of `Engine.__init__` **and** from `Scheduler.__init__` after the four spec captures — three of the five families are captured there, so a gate at the end of `Engine.__init__` alone would cover 2 of 5. |

What P5b proved is only that the *substrate* is capture-compatible: one elementwise op over a 4 MiB
view, 24 replays → one sha256, `data_ptr` stable, `empty_cache` on `graph.__enter__` freed nothing.
**A1.6 still requires a full-forward capture at the served TP across all five families (decode, verify,
fused, ddtree, canvas) over a populated frozen arena.** That is M1-D and it needs a card.

---

## 4. What a new model or quant format must implement

**Target: nothing. Current answer: a new MODEL family needs nothing; a new QUANT FORMAT needs one line.**

**A new model family: nothing.** The plan is resolved off the **built model** (`observed_planned_layers`
→ `discover_moe_layers` → `MoELayer.granule_specs()`), so the layer set, the EP sharding (all three
conjuncts, including the quant-method veto and `force_no_ep`) and the byte counts are read, not
transcribed. The seam is one edit in the shared `MoELayer.forward`, so all seven families that share it
are covered by construction. Paths use the `state_dict` grammar, so no builder needs to declare
anything. *(The config transcription in `plan.py` survives only as a pre-build fallback and is known to
be wrong for `models/utils.py::MoEMLP`, which passes no `quant=` at all — the observed path is what
makes that harmless.)*

**A new quant format: exactly one declaration, and forgetting it is a loud boot failure, not silence.**

```python
self._num_experts = num_experts    # in __init__, before post_load
```

`declared_granule_count` uses an `UNSET` sentinel (distinct from `None`, which legitimately means
"dense"), so an undeclared axis raises `GranuleError` naming both the fix and the escape hatch, instead
of silently reading the whole stack as one granule. Two coverage tests walk all eight quantized
container classes at **construction** time and again through `create_moe_quant_method`, so a format
added to the selector but not to a test list is still caught.

Everything else is derived or optional:
- Component set, dtypes, shapes, aliases, per-expert vs replicated: derived by the one walk. A format
  that grows a buffer gets a loud failure; one that drops a buffer needs no edit.
- `_residency_shared = (...)`: optional, and **verified bitwise against every live row** — a stale
  declaration raises instead of dropping a real per-expert buffer.
- `offload_refusal()`: optional. A format whose forward reads the whole stack (ZAYA's fp8 under
  `MINISGL_ZAYA_OLDMOE=1` / `_W8A16=1`) declares it; the planner then skips the layer and binds it
  DEVICE at zero bytes rather than dying at bind.
- `sizing.py`'s analytic table: **not** required. An unrecognised format resolves to `SCHEME_UNKNOWN`
  and is sized by the meta-device build of the real container. No kernel is forked; nothing is a new
  package (KERNEL_CORE_POLICY-clean).

**Two residual gaps, both honest:**
1. **`_granule_policy` is fail-OPEN.** Tensors are fail-closed, but a new *non-tensor decode decision*
   (the next `_ct_sign`) that nobody declares is exactly as invisible as `_ct_sign` was before it was
   found. Closing that needs the plan's §6.1 rule-8 **totality assertion** at the model root
   (sum of derived specs + declared-exempt == sum over every reachable tensor). Not landed.
2. **The walk is fail-closed only over what it reaches** (`BaseOP` / `nn.Module` / list / tuple / dict).
   A tensor cached on a `LinearMethod` rather than on the layer would be silently omitted. I verified
   no shipped container does this; nothing enforces it. Same fix: rule 8.

---

## 5. The corrected ceiling

All figures below are the resolver's own output on the target-shaped config (48 layers, E=512,
top_k=10, hidden 2048, inter 768, compressed-tensors int4 g32 symmetric, TP=2, no EP), computed with
the landed code, `prior=gfx1201-dual-2026-09-02`, host BW `(28.93, 14.48)` GB/s with the **slow rank
gating**, device 692 GB/s, 7.5 ms compute floor.

### Bytes — what actually changed vs Phase 0

| Quantity | Phase 0 §3.4 | **M1-A measured-by-model** |
|---|---|---|
| Expert bytes, per rank | 34.4 GB | **31.22 GiB** (= 33.5 GB) |
| per MoE layer, per rank | — | **666.0 MiB** (698,351,616 B) |
| granule (one expert, both GEMMs), per rank | 2.8 MB (arithmetic) | **1,327,104 B = 1.266 MiB** |
| bytes/token node-wide, bs=1 | 1.33 GB (arithmetic) · 1.1625 GB (GGUF) | **1.27 GB** (10 × 48 × 1.327 MB × 2 ranks) |
| largest single arena row | not modelled | **444.0 MiB** (a w13 component slab) |

Phase 0's ±14 % `granule_bytes` bracket (unknown #12) closes to **−4 %** against the plan's arithmetic.
Two post-`post_load` deltas Phase 0 did not carry are now charged: compressed-tensors **symmetric**
zeros are synthesised as a real `(E, G, N/pf)` int32 buffer (+18 MiB/layer, resident but *not* in the
granule — every expert reads the same row), and MXFP4 widens its E8M0 scale to fp16 (+6.25 % at g=32).

### tok/s, against the measured baselines

| | tok/s | vs llama.cpp 4.990 | vs K4 3.178 |
|---|---:|---:|---:|
| llama.cpp, bs=1, measured | 4.990 | 1.00× | 1.57× |
| **K4 hard-kill** | **3.178** | 0.64× | — |
| **T1 all-host, TP=2, idle — projected** | **19.42** (51.5 ms step) | **3.89×** | **6.11×** |
| at f=0.20 device tier | 23.03 | 4.62× | 7.25× |
| at f=0.25 | 24.56 | 4.92× | 7.73× |
| at the *required* tier, 11.06 GiB/rank | 27.60 | 5.53× | 8.68× |
| *if card 1's slot trains Gen5 (K7)* | *≈ 34* | *≈ 6.8×* | — |

Phase 0's 18.7 tok/s becomes **19.42** (fewer bytes). **A1.7 re-instantiates at 14.57 tok/s**
(0.75 × 19.42), up from the plan's 14.0. K4 is cleared by 6.1× on the projection — but every number in
this table is **PROJECTED, not measured**, and A1.7/K4 are measured gates on a served A/B with graphs
on. Nothing here discharges them.

### Capacity — the binding constraint, and a new defect in it

Phase 0 §3.4 charged capacity against the **payload**. The arena pins **whole 2 GiB chunks**, and its
bump allocator may never let a region straddle a chunk, so it abandons a tail per chunk. Reserving the
host tier as anonymous headroom therefore needs `headroom_chunks(payload, chunk, max_row)`, not
`ceil(payload/chunk)`:

| | value |
|---|---|
| host payload, all-host | 31.22 GiB/rank · **62.44 GiB/node** |
| **pinned reservation** at 2 GiB chunks | **40.00 GiB/rank · 80.00 GiB/node** (20 chunks, not 16 — **+28 %**) |
| usable pinned budget (P3b 62.0 GiB × 0.90 headroom) | **55.80 GiB/node** |
| verdict | **INFEASIBLE** — resolver aborts at config time, in ms |
| **required device tier, as landed** | **11.06 GiB/rank** (→ 52.00 GiB pinned, 27.60 tok/s) |
| same at a 4 GiB chunk | 10.41 GiB/rank (nothing > 2 GiB has ever been pinned on this box) |
| **required device tier if rows were NAMED regions (exact packing)** | **≈ 3.90 GiB/rank** (6 layers on device, 54.63 GiB/node host) |

> **[M1B-2026-09-03] FIXED. The computed answer is 5.85 GiB/rank, not the ≈ 3.90 estimated below.**
> The rows are now emitted as `RegionRequest(forecast=True)` by `OffloadPlan.host_row_requests()` and
> reserved by `StageARuntime.attach_host_arena`, so the reservation is the real next-fit packing:
>
> | | as landed in M1-A | **M1-B, named regions** |
> |---|---|---|
> | pinned reservation, all-host | 20 chunks · 40.00 GiB/rank · 80.00 GiB/node | **16 chunks · 32.00 GiB/rank · 64.00 GiB/node** |
> | required device tier | 11.06 GiB/rank (17 of 48 layers) | **5.85 GiB/rank (9 of 48 layers)** |
> | at that tier | 26 chunks · 52.00 GiB/node — but the tier is unaffordable | **13 chunks · 52.00 GiB/node ≤ 55.80 → FEASIBLE** |
> | VRAM recovered | — | **5.20 GiB/rank** (~1.0 M KV tokens) |
>
> The ≈ 3.90 estimate charged the *payload* (54.63 GiB/node) rather than the reservation: 42 host
> layers pin 14 chunks = 56.00 GiB/node, 0.20 GiB over the ceiling, so 39 host layers / 9 device is
> the real minimum. Per-layer rows are 384/48/12 MiB (w13) + 192/24/6 MiB (w2) = 666 MiB, and three
> whole layers fit a 2 GiB chunk (1998 MiB used, 50 MiB abandoned).
>
> **The one assumption a GPU run must replace:** torch's caching allocator asks the arena for a
> rounded SEGMENT, not for the tensor, so rows are reserved at `chunk_plan.torch_allocation_bytes`
> = `round_up(max(n, 2 MiB), 2 MiB)`. That mirrors `kRoundLarge`/`kSmallBuffer`, but the
> `kLargeBuffer` bucket (a 6 MiB row costing a 20 MiB segment) is deliberately NOT modelled because
> torch splits such blocks and charging every one 20 MiB over-reserves ~3× on a small-row shape. On
> this shape the exposure is 48 × 14 MiB and is absorbed by the 50 MiB tail each chunk already
> abandons; on another shape it would fail loudly at the carve, not silently in VRAM. Measure it:
> compare `ArenaMemPool.stats()`'s served sizes against `OffloadPlan.host_row_requests()`.
>
> Two adjacent defects the fix had to close. `verify_matches_plan()` now REFUSES a vacuous pass and
> is called from `seal()` — it had no production caller, and on the shipping shape it had nothing to
> compare. And `observed_planned_layers` (the path `engine.py` actually uses, on the meta model) was
> not applying `sizing.post_load_delta_bytes` at all, so the CT-symmetric `_zeros_op` and the MXFP4
> scale widening were uncounted; the old +28 % slop was hiding it. `placement.PostLoadCorrection`
> closes that, gated on the spec being meta-derived so a post-`post_load()` re-plan is not
> double-charged.

**This is the headline number of the report.** 11.06 GiB of a 16 GiB card, with the device tier billed
*inside* `model_memory` (the plan's correct "no sixth subtrahend" rule), leaves ~3.3 GiB at
`--memory-ratio 0.9` for the dense weights, the KV pool, recurrent state, the draft model and the graph
buffers. That is very likely unbootable. At ~3.90 GiB/rank it is comfortable. **The 7.2 GiB/rank
difference is entirely an artefact of reserving anonymous headroom instead of enumerating the rows**,
and it is the single highest-value fix in M1-B (≈ 1.4 M KV tokens of VRAM recovered per rank).

A related consequence already landed as a boot refusal: with a *derived* device budget
(`total_memory × memory_ratio`) and the tier billed inside `model_memory`, `available_memory` is
negative before state/draft/graph/snap are counted whenever the plan is non-empty — i.e. **every**
offloading serve was arithmetically unbootable, failing after ~7 s of pinning and a full checkpoint
load with an assert naming four causes that do not include the tier. `bake.UnconfiguredDeviceTierError`
now refuses from integers before a page is pinned, naming `--weight-offload-device-gb`.
**Practical effect: that flag is effectively mandatory whenever offload actually engages.**

---

## 6. What remains — M1-B..E, in order

Estimates assume one engineer, a free card when needed, and the GPU-lease waiver still in force
(serialise: never two GPU jobs at once).

### M1-B — engine/config wiring + the sizing fix · **2.5 d** (was 1 d)

1. **Fix the image/source skew first** (0.25 d). `tail_hip has no attribute silu_and_mul` blocks every
   import-level verification of `engine.py`. Rebuild the serve image against the current kernels tree
   (per CLAUDE.md: clean worktrees for both contexts, bump `KERNELS_REF`).
2. ~~**Enumerate the arena rows as NAMED regions instead of anonymous headroom** (1 d).~~
   **DONE [M1B-2026-09-03].** Required device tier **11.06 → 5.85 GiB/rank**, all-host reservation
   **80.00 → 64.00 GiB/node**, and the target shape is FEASIBLE at 52.00 GiB/node. See the correction
   block in §5. Residual for the first GPU run: confirm torch's real `MemPool` request sizes against
   `chunk_plan.torch_allocation_bytes`, which is modelled and unmeasured.
3. **First GPU run** (0.5 d): `pytest -m gpu tests/core/test_pinned_weight_arena.py` and
   `test_granule_regdirect_gpu.py`, **serially on BOTH cards**. This box has burned people twice on
   cross-card assumptions and P2′/P5/P6 only ever ran on card 0. Confirm real `hipHostMalloc` bases are
   ≥ 512 B aligned (`_verify_chunk_structure` asserts it; if ROCm returns 256 B it becomes a boot
   failure — right direction, wrong time to discover it).
4. **Pin at scale** (0.25 d): one 30 GiB arena on one rank, timed, with the swap tripwire live. P3b's
   4.88 GB/s → ~7 s/34 GiB is the prediction.
5. Serve-table entry, `[serve]` banner line, compose env + `docs/COMPOSE_ENV_AUDIT.md` (0.5 d) — land
   the operating point in the launch table, not only in docs.

### M1-C — correctness gates · **2.5 d** (was 2 d; +0.5 for dense)

Order matters: **A1.0 eps0 first** (the noise floor), then **A1.4** (populate self-test on real pages,
which is the Phase-0 trap detector and already implemented — just needs to run), then **A1.1**
identity, then **A1.2 permuted-mirror** (the merge gate, TP=1 and TP=2, every format the zoo has), then
**A1.3 desync must-fail matrix** on an **asymmetric** checkpoint so the zeros arm is non-vacuous. Then
**A1.5 dense parity**, which requires the dense enumeration + dense seam that do not exist — budget
that inside this milestone, not after it. **MoE and dense land together; a dense backlog is not a
resting state.**

### M1-D — served A/B under capture at TP=2 · **2 d**

`tools/weight_offload_serve_ab.sh` on the `moe_g2_split_serve_ab.sh` shape. One image, one worktree,
one kernels build, legs differ by one knob, `--num-pages` **pinned** across legs (an auto-sized pool
differs because `model_memory` shrinks). Graphs **ON** — a leg that fell back to eager is VOID.
`engaged_cap(name)` (not bare `engaged()`, which latches during the eager warmup) and a programmatic
`LAYOUT_ID`. **Diff the engaged() ledgers per leg.** Discharges A1.6 (five capture families) and
measures A1.7 (≥ 14.57 tok/s) and K4 (> 3.178). Sampled quality run only — **never temp=0.**

### M1-E — Stage B chunked load · **3 d**

Only the target checkpoint needs it, and it needs it absolutely: 68 GB OOMs during `load_state_dict`
long before Stage A's bake runs. `BaseOP.post_load(granule_range=None)`; per-key device selection in
the five loaders (`weight.py:906/1106/1245/1359/1454` — `_load_qwen3_5_weight` at `:546/:622` is the
precedent); an expert-range argument on the four MoE repack ops **and** the dense `repack_w_rep_wide`
in the same change. **Thread `container._ct_sign`** into every chunk — it is published for exactly this
and re-deriving per chunk XOR-corrupts a contiguous block of experts with plausible text and no crash.
Stage B also introduces the first **CPU store** into arena pages; today every write is device-issued,
so the visibility discipline is currently unreachable rather than proven — write it down then.

### After M1

- **M2 collapses.** P2′ made the layer-granular tier M1's, and priced per-expert placement at ~1 %. M2
  is now only: the static prior file (checkpoint sha + `LAYOUT_ID` + tp/ep shape), the A2.4 Pareto
  sweep **measured** rather than projected, and its second axis (max context / `max_running`
  surrendered per GiB of tier, from the real `cache_per_page`). **≈ 2 d, down from 5.**
- **M4 metrics · 2 d.** Five-hop plumbing, `-1` sentinel init, TP-reduce on the CPU group, and the
  validated operating point in `tools/serve.sh`'s table.
- **K7 · 1 h, free 1.75×.** Card 1's root port trains Gen4 x8; every TP=2 ceiling in this document is
  gated by it. Try the BIOS before costing anything else around 14.48 GB/s. Sample link speed
  **mid-DMA** — idle sysfs reads 2.5 GT/s on both cards (ASPM).
- **P3c · 0.5 d**, only if the device tier cannot be afforded: file-backed `mmap(MAP_SHARED)` +
  `hipHostRegister` is the only route past the 62 GiB pinned ceiling other than device VRAM, and it is
  uncharacterised above 4 GiB/rank.
- **P3d · 0.5 d** (new): can this box pin a **4 GiB** chunk? Worth 0.65 GiB/rank of device tier, and it
  is the cheapest lever on the capacity problem after the named-region fix.

---

## 7. What invalidates the plan or the Phase 0 report

All applied in place to `docs/WEIGHT_OFFLOAD_PLAN.md`, tagged `> **[M1A-2026-09-03]**`.

1. **`PHASE0_REPORT.md` §3.4's device-fraction table is WRONG in its "fits" column.** It charges the
   **payload** against the ceiling; the arena pins **whole chunks over a never-straddle allocator**, so
   the real charge at 2 GiB chunks is +28 %. Its `f=0.20 → 55.0 GB ✅` row does **not** fit (it pins
   72 GiB/node). The corrected table is §5 above. *(Phase 0 is a dated gate record; the correction is
   recorded here and in the plan, not by rewriting it.)*
2. **Plan §11 unknown #2 — ANSWERED.** LINEAR, not cliffed. The "2× vs 0×" spread collapses to ≈ 0×.
   Layer-granular is final; per-expert placement is closed, not deferred.
3. **Plan §11 unknown #10 — ANSWERED GREEN.** torch takes a `hipHostGetDevicePointer` address under
   `MemPool` and under capture, on both cards. The C++ `from_blob` route (+3 d) is **not** needed.
4. **Plan §11 unknown #12 — mostly answered.** `granule_bytes` = 1,327,104 B/expert/layer/rank →
   1.27 GB/token node-wide, −4 % on the plan's 1.33 GB arithmetic, not the −14 % the GGUF figure
   suggested. Still analytic + meta-derived; **not** measured on the real checkpoint.
5. **Plan §5.1's arena bullet is incomplete and the omission costs ~7.2 GiB/rank of VRAM.** It says
   nothing about the reservation granularity. Anonymous headroom must be replaced by named regions.
6. **Plan §5.3 + §6.2 interact badly.** "No sixth subtrahend" (correct) + a *derived* device budget of
   `total × memory_ratio` makes `available_memory` negative on every offloading serve. Fixed by a
   pre-pin refusal; the consequence is that `--weight-offload-device-gb` is effectively **mandatory**
   whenever offload engages, which the plan's "no env gate" section did not anticipate.
7. **Plan §5.6's A1.7 = 14.0 tok/s is superseded by 14.57** (0.75 × 19.42, on the refined byte model).
   Re-derive again after any BIOS change to card 1's slot.
8. **Plan §10's K5 is now evaluable and effectively cannot fire.** With the curve linear, M2's
   layer-granular gain *is* M1's, and the per-expert increment P2 was supposed to size is ~1 %.
9. **Plan §5.2's `_ct_packed_is_uint4b8` note was right about the prefix and wrong about the scale of
   the hazard.** On every shipped MoE shape the obvious strided replacement aliases to *one packed
   column* (stride is an exact multiple of the packed row length), and the detector was int32-pinned so
   it read uint8-packed int4 as 6/8 zeros and answered confidently backwards. Both fixed; the
   *cross-rank* divergence (two TP ranks sampling different shards and deciding oppositely) is **pinned
   by test, not closed** — it needs a CPU-group compare at `post_load`, or a checkpoint-level decision
   threaded from `quantization_config`, which is also what Stage B needs.

**Unchanged and still true:** M3/T3 is dead (P6). The mixed device/host VA is not constructible.
`hipPointerGetAttributes` lies in both directions. Card 1 gates every TP=2 ceiling. The 62 GiB two-rank
pinned ceiling was measured on an **idle box with no engine loaded**, and the 0.90 headroom derate that
turns it into 55.80 GiB is a **judgement, not a measurement** — it is the single knob deciding whether
the target model boots, and nobody has measured the real number with a live serve resident.
