# The 48-layer TP=2 boot defect — what it was, what fixed it, and what the fix did NOT touch

`[BOOT-2026-09-05]` — measured 2026-09-06, 48 layers, TP=2, the committed operating point. Every
number here is **n=2 per arm**, four full 48-layer TP=2 boots, strictly sequential, one job on the
box at a time, each under its own `gpu-lease -n 2`.

Raw artifacts: `docs/measurements/BOOT_TIMELINE_2026-09-06/identity/`.

---

## 1. Verdict

| | before (`02957062`, mmap) | after (`9bf388db`, O_DIRECT into one reused buffer) |
|---|---|---|
| **boot** | 503.5 / 509.7 s — **mean 506.6** | 303.8 / 296.4 s — **mean 300.1** |
| **speedup** | | **1.69x, −206.5 s** |
| `stage_b_run` (rank0) | 312.6 / 365.1 s | 124.1 / 87.3 s |
| decode throughput (sampled, 5 reps) | 14.782 tok/s | 14.703 / 14.788 tok/s |
| peak host RSS | 31.33 / 31.31 GB | 31.30 / 31.34 GB |
| CPU unit suite | 4 failed, 1045 passed, 14 skipped | **identical**, same 4 tests |

**Weight identity holds, byte for byte.** Every one of the four boots produced the same weight
digest on the same rank — `19f67bc6695b16db8b37f76b90407961` (rank 0) and
`ef7229c5d763e56a7c7764b68ff86051` (rank 1) — over **1990 tensors / 70.44 GiB per rank**, hashed
with blake2b through the device-side mapping the MoE kernels dereference. 216 host tensors
(23.95 GiB), all 216 read through arena-owned pointers; 49 layer buckets for 48 layers + body.

**Behavioural identity holds at the width where it is evidence.** The 12-step bs=1 parity probe
returned `[11751, 13, 561, 6511, 314, 9564, 369, 19241, 13, 561, 6511, 314]` in
**4 legs × 2 ranks × 2 modes (captured and eager) = 192 greedy ids, all identical**, across both code
versions.

---

## 2. The defect was NOT what the opening brief said it was

The brief framed it as *"Stage B is 28-47x slower than the drive; over 99% of boot is not I/O."*
The first half is right and the second is **wrong, and it was wrong because nobody had run the
attribution**. Depth-0 phases sum to the total, so the boot was never dark:

| depth-0 phase (rank 0) | before mean | after mean | delta |
|---|---|---|---|
| `weight_load` | **355.53** | **140.64** | **−214.9** |
|   └ `stage_b_run` | 338.88 | 105.72 | −233.2 |
|   └ `stage_b_enumerate_prepass` | 16.59 | 34.87 | +18.3 |
| `arena_pin_attach` | **86.60** | **78.34** | −8.3 (noise; see §5) |
| `graph_capture` | **61.21** | **69.55** | +8.3 (noise; see §5) |
| `ct_sign_verify` | 0.08 | 8.42 | **+8.3 (real)** |
| everything else (10 phases) | ~1.0 | ~1.0 | — |
| **TOTAL** | **505.4** | **298.9** | **−206.5** |

The "previously-dark 352-593 s" resolves into exactly three things: **`arena_pin_attach` (~87 s),
`graph_capture` (~61 s), and the part of `weight_load` outside Stage B (~17 s)**. There is no
unaccounted remainder. `model_construct_meta`, `kv_sizing`, `init_communication`, `backends_init`
and the other nine phases together are **under one second** and were never candidates.

And boot is overwhelmingly I/O-shaped after all — see §3.

---

## 3. What the win actually is: the read

Bucket-level, rank 0, all four legs:

| bucket | before | before2 | after1 | after2 | before mean | after mean |
|---|---|---|---|---|---|---|
| `stageb.stream_next` | 276.66 | 331.92 | 78.07 | 66.06 | **304.29** | **72.06** |
| `ckpt.h2d` | 215.97 | 271.06 | 39.99 | 26.46 | **243.51** | **33.23** |
| `arena.host_alloc` | 94.52 | 69.92 | 73.17 | 79.55 | 82.22 | 76.36 |
| `ckpt.shard` | 42.05 | 42.27 | 7.94 | 8.20 | 42.16 | 8.07 |
| `stageb.sink_place` | 22.52 | 19.49 | 4.37 | 7.89 | 21.00 | 6.13 |
| `ckpt.get_tensor` | 7.11 | 7.32 | 1.35 | 1.37 | 7.21 | 1.36 |
| `stageb.post_load` | 6.67 | 6.72 | 35.18 | 6.97 | 6.69 | 21.07 |
| `ckpt.nvfp4_prepass` | 4.18 | 0.31 | 22.06 | 20.20 | 2.24 | **+18.9** |
| `ckpt.safe_open` | 0.68 | 0.41 | 18.44 | 19.56 | 0.54 | **+18.5** |

`ckpt.h2d` is a host-to-device *copy*, and it fell 210 s. It was never 210 s of copying: under
`safetensors`' mmap the copy's SOURCE pages were faulted in lazily **during** the copy, so the read
was billed to `ckpt.h2d` as deferred page faults. With O_DIRECT the bytes are already in the buffer,
`ckpt.h2d` becomes a pure copy, and the read appears — correctly — in `ckpt.safe_open`. That is why
`safe_open` going 0.5 → 19.0 s is not a regression: it is the same work, finally in its own bucket.
`major_faults` corroborates it mechanically: **56.1 / 60.2 M → 24.1 / 23.6 M**.

Two costs are real and are not netted out:

* `ckpt.nvfp4_prepass` **+18.9 s** and `stage_b_enumerate_prepass` **+18.3 s**. The header-only
  prepass is deliberately still on mmap and used to be cheap only because something else had left
  the ARC warm. Nothing warms it now.
* `ct_sign_verify` **+8.3 s** (0.08 → 8.42, consistent across both after legs, and r3 saw the same
  0.24 → 6.66). Same cause.

Net of those, the read fix is worth ~250 s and the boot keeps ~206 s of it.

---

## 4. Corrections to earlier documents `[BOOT-2026-09-05]`

### 4.1 "THE TWO LARGEST WINS ARE NOT READS" — **REFUTED**

`docs/measurements/BOOT_TIMELINE_2026-09-06/r3/README.md`, the `9bf388db` commit message, the
`weights/ckpt_read.py` module docstring and the `models/weight.py` comment all assert that
`stageb.post_load` fell 106.1 → 7.0 s and `graph_capture` fell 84.1 → 33.6 s, that "neither touches
the checkpoint", and therefore that the win is the box no longer stalling both processes rather than
the read. **At n=2 per arm neither figure survives:**

| | r3 before | r3 after | id before | id before2 | id after1 | id after2 |
|---|---|---|---|---|---|---|
| `stageb.post_load` | 106.08 | 7.02 | **6.67** | **6.72** | 35.18 | 6.97 |
| `graph_capture` | 84.11 | 33.64 | **68.79** | **53.62** | **68.65** | **70.45** |

`post_load` is ~6.7 s in **both** of my before legs, so r3's 106.1 s before-leg was the outlier and
the claimed 99 s saving does not exist. `graph_capture` does not improve at all — 61.2 s before
vs 69.6 s after, i.e. slightly *worse*, and r3's 84.1 → 33.6 s is within the spread of a phase that
ranges 53.6-70.5 s across four boots of two code versions. The same applies to `arena_pin_attach`,
which r3 credited with −22.8 s: it spans 71.6-101.6 s on the before arm alone.

The mechanism claim should be replaced by the one in §3, which is supported: **the win is the read,
and it is large because mmap was charging it to `ckpt.h2d` as page faults.** The box-pressure story
is not needed to explain 206 s and is not established by these legs.

### 4.2 The identity evidence in r2/r3 was taken at a width that cannot discriminate — **CORRECTED**

r2 and r3 both certify "the same greedy token ids from both prompts on both ranks". Those are
`out["token_ids"]`, produced by test step `[5]`, which submits **both prompts in one
`llm.generate([...])` with `max_running_req=2`** — so they decode together at **bs=2**, where MoE
gemm2 is `mmq_fp8_moe_gemm_scatter`, an atomic scatter with no fixed accumulation order. Measured:

* `L48_r2_before`, `L48_r2_after`, `L48_r3_before`, `L48_r3_after` → `[11751, 13, 11751, 369, 264, 3177]`
* `L48_id_before`, `L48_id_before2` → `[11751, 13, 561, 6511, 314, 9564]`

`L48_id_before` and `L48_r3_before` are **the same `02957062` checkout** and disagree. Both
completions are coherent (" Paris. Paris is a city" / " Paris. The capital of Germany") — a near-tie,
not corruption. So that field is not a function of the weights and must never gate an A/B: it
produces a green that is a coin landing the same way twice, and a red that indicts a correct change.
`tools/offload/boot_ab_report.py` now prints it as **report-only** and gates on `parity_*_ids`
(bs=1) instead.

**The same-code floor was established first, and it passes:** before == before2 and after1 == after2
on every gated field, and within each boot `_repro_probe` reports
`repro_engine_is_reproducible: true` (4 identical generates in each of capture-on and capture-off).

### 4.3 "`posix_fadvise(POSIX_FADV_DONTNEED)` returns 0 and frees NOTHING on this mount" — **REFUTED**

This was the recorded reason for rejecting the smaller, in-place alternative (bound mmap's footprint
rather than replace the reader). It was measured on leg 1 of `tools/offload/zfs_readpath_probe.py` —
the **16 MiB-stride control**, which populates `dCached +0.00 GiB`. It was asking an empty cache to
drop something. Re-measured with the fadvise leg moved onto the 4 KiB walk, which had just added
+0.33 GiB to `Cached` and +0.34 GiB to the ARC (`identity/zfs_readpath_fadvise_retest.txt`):

```
mmap 4K walk                    621.7 MiB/s   dCached +0.33 GiB   dARC +0.34 GiB
  fadvise(DONTNEED) while mapped   rc=0   dCached +0.00 GiB   dARC -0.11 GiB
  fadvise(DONTNEED) after munmap   rc=0   dCached -0.33 GiB   dARC +0.00 GiB
```

`fadvise` **works**. It drops 0.11 GiB of ARC while mapped and the full 0.33 GiB of page cache once
unmapped — ordinary POSIX semantics (it cannot evict pages that are still mapped), not a ZFS defect.
The landed O_DIRECT fix is measured and stands; but an mmap + munmap + `fadvise` variant was never
actually ruled out, and the stated reason for not trying it was an artifact. Open, not urgent.

The rate table reproduces: O_DIRECT 5122.9 MiB/s (was 4967.0), 4 KiB mmap 621.7 (was 627.0).

### 4.4 The 2.26x headline was one sample per arm

`574.2 → 254.6 s` is a single leg each. At n=2 the same comparison is **506.6 → 300.1 s, 1.69x**.
Same direction, same sign on every large bucket; the magnitude was inflated by an unusually slow
before leg. Quote 1.69x.

---

## 5. RSS verdict — there was never a leak

`stage_b_peak_host_rss` is **31.33 / 31.31 GB before and 31.30 / 31.34 GB after: unchanged by the
fix**, so it is not what the fix addressed and not what made the boot slow.

The "22x the design's 1.465 GiB live set" signal is **the pinned arena itself, correctly counted**.
At end of boot `RssAnon` is 24.84 / 25.23 / 25.24 / 24.80 GiB against an `arena_pinned_bytes` of
**24.12 GiB** — i.e. the arena plus ~1 GiB of ordinary process anon. `VmHWM` 35.4-36.2 GiB is the
arena plus Stage B's transient working set. Independently, `tools/offload/heap_retention_probe.py`
put 222,252 alloc/free cycles at Stage B's size mix through glibc and measured **0.8 MiB** retained,
with or without `mallopt(M_MMAP_THRESHOLD)`. Nothing is retained that should not be.

---

## 6. Identity evidence in full

| claim | evidence | result |
|---|---|---|
| **Plan identity** | digest `a5e49a2e2b307ff3`, `156(forecast=156)` regions, 18 chunks, 24.12 GiB reserved/rank, 49 Stage-B chunks, 1236 keys, 36 host / 12 device layers, `arena_torch_fallbacks` 0, `seam_pointer_checked` true, `kv_pages` 3053 | SAME on all 4 legs, both ranks |
| **Weight identity** | blake2b over 1990 tensors / 70.44 GiB per rank, read through the device pointer; two walks unioned (`granule._iter_tensors` for the module tree + `seam.live_tensors()` for the expert containers it cannot reach) | `19f67bc6…` r0 / `ef7229c5…` r1 on all 4 legs; all 49 per-layer buckets match |
| — non-vacuity | 216/216 host tensors read through **arena-owned** pointers; 49 buckets == 48 layers + body; 25,716,326,400 host bytes hashed == `plan_host_bytes`; 0 hash failures | gated, passed |
| **Behavioural (bs=1)** | 12 greedy ids, captured and eager, both ranks, 4 legs | 192/192 identical |
| — floor, within boot | `_repro_probe`, 4 identical generates per mode | `repro_engine_is_reproducible: true`, all legs |
| — floor, across boots | before vs before2; after1 vs after2 | identical on every gated field |
| **Behavioural (bs=2)** | `[5]` two-prompt batch | **varies on unchanged code — not a gate** (§4.2) |
| **Throughput** | sampled A/B, checkpoint sampler, `ignore_eos`, 128 tok × 5 reps | 14.782 → 14.703 / 14.788 tok/s |
| **Test suite** | `tests/core tests/misc tests/kernel`, both trees, same container | 4 failed / 1045 passed / 14 skipped, **identical failure sets** |

Throughput per-rep samples overlap completely — before `[14.674, 14.632, 14.782, 14.827, 14.785]`,
after1 `[14.515, 14.662, 14.729, 14.703, 14.856]` — and both bracket the published reference
**14.546 tok/s** (`QWEN4EXP_L48_E4M3_HCFUSE_2026-09-05.json`). Eager legs likewise: 14.101 →
14.145 / 14.137. **The boot fix costs no steady-state.**

### Pre-existing test failures, baselined (identical on `02957062` and `9bf388db`)

```
FAILED tests/core/test_weight_boot_budget.py::test_kv_budget_keeps_exactly_five_subtrahends
FAILED tests/kernel/test_index.py::test_indexing            - RuntimeError: ninja exited...
FAILED tests/kernel/test_index.py::test_indexing_with_mask  - RuntimeError: ninja exited...
FAILED tests/kernel/test_store.py::test_store_cache         - RuntimeError: ninja exited...
```

Plus one collection error, excluded and identical on both trees:
`tests/kernel/test_tensor.py` — `Device value [rocm[1]] not in the allowed options: [cuda[1]]`.

Two harness defects found and fixed while running this, both pre-existing:

* `_capture_throughput_ab` timed its greedy legs **without `ignore_eos`**, so its own
  equal-token-count gate could not pass: greedy ids are not reproducible past ~32 tokens, the legs
  drifted, captured stopped at 68 tokens and eager ran 128, and the run went red for an engine
  property unrelated to capture — over two legs that by then were not a throughput comparison.
* `_capture_ab_sampled`'s equal-token gate fires stochastically on one rep in five (127 vs 128)
  while `decode_steps` is 127 on **every** rep in **both** legs — an output-accounting off-by-one,
  not a work asymmetry. The published 09-05 reference has the identical artifact
  (`ab_captured_tokens_per_rep = [128, 128, 128, 127, 128]`). Known since 2026-09-04; not a
  regression.

---

## 7. What is NOT fixed — the next blocker

**`arena_pin_attach` is now the largest single phase that the read fix did not touch: 78.3 s after
(76.4 s of it `arena.host_alloc`), against 86.6 s before.** That is one `hipHostMalloc` of 24.12 GiB
per rank at **~325 MiB/s**. It is not the first-touch fill — that was retired in `db6f8eea` after
`tools/offload/attach_touch_probe.py` showed `hipHostMalloc` commits eagerly here and pre-faulting
made things 33% worse. It is the allocation itself, and it is 26% of the remaining boot.

Also unaddressed, and now proportionally larger: `ckpt.nvfp4_prepass` + `stage_b_enumerate_prepass`
(+37 s combined) and `ct_sign_verify` (+8 s), all three of which the fix made *worse* by leaving the
ARC cold. A prepass that read its headers through the same O_DIRECT path would likely recover most
of it.

---

## 8. Process notes for this round

* **GPU leasing was resumed after a waiver.** All four boots and both test-suite runs ran under
  `gpu-lease -n 2`, one job at a time, never two concurrent single-card boots.
* **Two orphaned containers were found earlier** — one blocked 90 min at 0% CPU, one **spinning at
  197% CPU for ten hours**. The second was running under every measurement taken after 10:57, so
  **any figure from that window is suspect**, including the 519.5 / 726 / 871.1 s boot times in the
  opening brief. Every leg here names its container and reaps it on a `trap`.
* **A reported claim did not match the tree.** An earlier round reported
  `"torch_checks_updated": 3` for `rdna4-hip-kernels-e4m3/fp8_wmma/torch-ext/torch_binding.cpp`; the
  count of `must be fp16` checks there is **24, identical to the baseline tree**, and two boots then
  died of `scales must be fp16`. Grep before reporting. (The e4m3 path was fixed differently, at
  kernels `38ec157`; the built `.so` used by all four legs here carries the
  `E4m3GroupScaleGlobal` symbols and the in-container `fp8_wmma.__file__` provenance assert fires
  on every run.)
* Box at launch of the three-leg sequence: load 2.07, MemAvailable 71.2 GiB, ARC 3.75 GiB, both
  cards FREE, no non-infra containers. Per-leg `MemAvailable` at start: 65.6 / 69.1 / 69.7 /
  70.3 GiB — the before arm started **lower**, i.e. the comparison is conservative.
* `MINISGL_WEIGHT_ARENA_FLOOR_GIB=4` on every leg (shipped default is 12). It is a pre-flight gate,
  not a measurement parameter: `MemAvailable` does not count the ZFS ARC, which is reclaimable, so
  the 48.23 GiB plan + a 12 GiB floor cannot be satisfied on this box and `reserve()` aborts before
  pinning a page. Same value on both arms.
* Source isolation: the two arms are **separate git worktrees**
  (`minisgl-rdna4-ramperf-base` @ `02957062`, `minisgl-rdna4-ramperf` @ `9bf388db`), so neither leg
  could read the other's source. The harness file is byte-identical in both; only
  `models/weight.py` and `weights/ckpt_read.py` differ. `9bf388db`'s executable AST is identical to
  the measured `25d61787` (the delta is comments and docstrings only), verified mechanically.

---

## 9. Files

```
docs/measurements/BOOT_TIMELINE_2026-09-06/identity/
  L48_id_{before,before2,after1,after2}.boot.rank{0,1}.json   per-rank phase/bucket/counter timeline
  L48_id_{...}.test.json                                      harness verdict, weight digest, ids, tok/s
  L48_id_{...}.run.log                                        full engine log
  L48_id_{...}.box.txt, .box_sampler.jsonl.gz                 box state around, and 1 Hz through, each leg
  ab_report_before2_vs_after1.txt                             the A/B
  ab_report_floor_after1_vs_after2.txt                        the SAME-CODE FLOOR
  pytest_{before_02957062,after_9bf388db}.txt                 CPU suite, both trees
  zfs_readpath_fadvise_retest.txt                             §4.3
```

New/changed tooling, all host-side and lease-free:
`tests/qwen4exp_offload_serve_test.py::_weight_digest` (`--weight-digest`),
`tools/offload/boot_ab_report.py` (weight-digest + per-layer diff, bs=1 parity gated, bs=2 ids
demoted to report-only), `tools/offload/zfs_readpath_probe.py` (fadvise leg moved onto a populated
cache).
