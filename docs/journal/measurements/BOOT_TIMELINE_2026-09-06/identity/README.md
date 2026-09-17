# Identity round — does the faster boot load the SAME model? `[BOOT-2026-09-05]`

Four full 48-layer TP=2 boots, **n=2 per arm**, strictly sequential, one job on the box at a time,
each under its own `gpu-lease -n 2`. The narrative, the corrections to r2/r3, and the full phase
attribution are in **`docs/measurements/BOOT_DEFECT.md`** — this page is just the artifact index and
the one-screen answer.

## Answer

**Yes, byte for byte.** All four boots produced the same weight digest per rank over **1990 tensors
/ 70.44 GiB**, hashed with blake2b **through the device-side mapping the MoE kernels dereference**:

| | rank 0 | rank 1 |
|---|---|---|
| `L48_id_before`  (02957062) | `19f67bc6695b16db8b37f76b90407961` | `ef7229c5d763e56a7c7764b68ff86051` |
| `L48_id_before2` (02957062) | `19f67bc6695b16db8b37f76b90407961` | `ef7229c5d763e56a7c7764b68ff86051` |
| `L48_id_after1`  (9bf388db) | `19f67bc6695b16db8b37f76b90407961` | `ef7229c5d763e56a7c7764b68ff86051` |
| `L48_id_after2`  (9bf388db) | `19f67bc6695b16db8b37f76b90407961` | `ef7229c5d763e56a7c7764b68ff86051` |

All 49 per-layer buckets match on every pairing. The two ranks differ from each other, as they must
— they hold different TP shards.

Boot **506.6 -> 300.1 s (1.69x)**. Throughput **14.782 -> 14.703 / 14.788 tok/s** (sampled, 5 reps,
`ignore_eos`), against a published reference of 14.546.

## Why the digest, and why it is not vacuous

Every other gate in this harness is an **accounting** identity — plan digest, region forecast,
`arena_pinned_bytes`, `copied_bytes`, `stage_b_keys_filled`. All of them pass over an arena holding
the right NUMBER of the wrong bytes. The arena's own `selftest_light` is a 512-probe sample per
1.34 GiB chunk compared against fingerprints **the arena itself wrote**, so it proves the mapping is
not aliased and says nothing about whether the checkpoint landed. Four checks are therefore gated:

* `weight_digest_layer_buckets == layers + 1` — 49 for 48 layers + body
* `weight_digest_bytes > 0`
* `weight_digest_host_bytes > 0` — a digest that never left the device tier would skip the 23.95 GiB
  the fix actually moves differently
* **`arena_owned_tensors == host_tensors`** — 216/216. Without it a "host" digest could be read from
  a VRAM copy, making the whole comparison a statement about the wrong memory.

## The behavioural gate is bs=1, and that is not a detail

`out["token_ids"]` (test step `[5]`) submits **both prompts in one `generate()` with
`max_running_req=2`**, so they decode at **bs=2**, where MoE gemm2 is `mmq_fp8_moe_gemm_scatter` — an
atomic scatter with no fixed accumulation order. It varies across boots **on unchanged code**:
`L48_id_before` and `L48_r3_before` are the same `02957062` checkout and disagree. r2 and r3 gated on
it. It is now **report-only** in `tools/offload/boot_ab_report.py`.

The gated field is the 12-step **bs=1** parity probe, captured and eager:
`[11751, 13, 561, 6511, 314, 9564, 369, 19241, 13, 561, 6511, 314]` — **192/192 identical** across
4 legs x 2 ranks x 2 modes.

**The same-code floor was established first and passes**: `before == before2`, `after1 == after2` on
every gated field, and within each boot `repro_engine_is_reproducible: true`.

## Files

| file | what |
|---|---|
| `L48_id_{before,before2,after1,after2}.boot.rank{0,1}.json` | per-rank phases (with `depth`), buckets, counters, per-chunk rows |
| `L48_id_*.test.json` | harness verdict, weight digest + per-layer map, parity ids, tok/s |
| `L48_id_*.run.log` | full engine log |
| `L48_id_*.box.txt` / `.box_sampler.jsonl.gz` | box state around each leg / 1 Hz `/proc` + ARC through it |
| `ab_report_before2_vs_after1.txt` | **the A/B** |
| `ab_report_floor_after1_vs_after2.txt` | **the same-code floor** |
| `pytest_before_02957062.txt` / `pytest_after_9bf388db.txt` | CPU suite, both trees: 4 failed / 1045 passed / 14 skipped, identical failure sets |
| `zfs_readpath_fadvise_retest.txt` | `fadvise(DONTNEED)` re-measured on a POPULATED cache — it works; the earlier "frees nothing" was measured on the empty-cache control leg |
| `PROVENANCE.txt` | shas, harness hash (byte-identical both worktrees), kernel `.so` hash, args |

`L48_id_before` used an earlier probe set (`--throughput-tokens` instead of `--ab-tokens`) and is
kept as a second before-arm sample for boot time and the weight digest; its one failure is the
pre-existing `_capture_throughput_ab` `ignore_eos` defect, fixed in this round and described in
`BOOT_DEFECT.md` §6.
