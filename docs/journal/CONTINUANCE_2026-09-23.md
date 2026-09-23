# Continuance — Flash-Next MXFP4 on minisgl, 2026-09-22/23

Handover for the q4e (Qwen3.8-Flash-Next MXFP4) work. Written after a reboot wiped `/tmp`, which
took every bench JSON and arm log with it — so the numbers below are the surviving record. Repo at
`5e0aca77`; the serving worktree `minisgl-rdna4-regress` is in sync with it.

## THE MEASUREMENT PROBLEM — READ THIS BEFORE QUOTING ANY tok/s FROM YESTERDAY

**GPU0's SMU firmware hung at 2026-09-22 20:45.** Prometheus is authoritative:
`amdgpu_power_watts{gpu="0"}` and `amdgpu_gfx_clock_mhz{gpu="0"}` stop dead at that timestamp while
`gpu="1"` keeps reporting. `rocm-smi` returned NOTHING for GPU[0] — no temperature, no clocks, no
power — and dmesg filled with `SMU is in hanged state, failed to send smu message!` for
`0000:03:00.0`.

**dmesg's first timestamp (10:23:06 on 09-23) is a LIE about when it started** — that is only when
the kernel resumed logging. Reading dmesg alone produced a confident "this began 40 seconds ago",
which was wrong by 13.5 hours and would have exonerated three contaminated arms. **For "when did a
card stop being healthy", query Prometheus, not dmesg.**

The card still computed (serves produced tokens) but its clocks were unmanaged, so throughput
numbers crossing 20:45 are not comparable:

| arm | finished | verdict |
|---|---|---|
| `bench-ours-q4e-mxfp4` (cache off) | 09-22 19:18 | CLEAN |
| `bench-ours-q4e-mxfp4-cache` (cache on) | 09-22 19:37 | CLEAN |
| `bench-ours-q4e-mxfp4-fixed` (PLE 4K) | 09-22 20:51 | **STRADDLES the line** |
| `bench-ours-q4e-mxfp4-ecfix` (cache fix) | 09-22 23:01 | CONTAMINATED |
| `bench-ours-q4e-capture` (GRAPH_BS=2) | 09-23 10:24 | CONTAMINATED |

RETRACTED as a result: the PLE 4K result of "+11.5% median, 7/8 rows, p=0.035", the cache fix's
"+3.9% median", and capture's "+2.1%". The reboot at 11:27 cleared the SMU (hang count back to 0),
and a peer session is re-measuring the `GRAPH_BS=0` control and `GRAPH_BS=2` arms back-to-back.

**What is NOT affected**, because none of it is a wall-clock measurement:

* Every correctness result (CPU replay, no GPU).
* The PLE device-traffic measurement (`/proc/diskstats`).
* The expert-cache A/B wash — both legs ran before 20:45.
* All cache *counters* (`observed_h`, promotions/tick, refs/tick). These count installs, not time.

## WHAT LANDED, AND WHAT IT IS WORTH

### 1. The expert cache was silently serving WRONG EXPERT WEIGHTS (the real result)

Independently reproduced via `tools/offload/expert_cache_replay.py` on the real 22,001-step route
trace, CPU only:

```
BEFORE (a8b6c95e, what production ran)   served_h 0.2306   WRONG-BYTES 54,199 = 1.88% of expert reads
                                         slot_of published 3376 experts over 2056 slots
                                         DANGLING 1345, violations 2665
AFTER  (on rdna4)                        served_h 0.4774-0.4802   WRONG-BYTES 0, violations 0
```

More experts were marked resident than slots exist, so the kernel read another expert's weights on
~1 read in 53 — the exact "plausible wrong numbers with no error anywhere" failure
`expert_cache.py`'s own docstring forbids. **This is a live candidate for the q4e degeneration**
(see `q4e-serve-degrades-within-a-boot` in memory) and is NOT yet confirmed as its cause — that
needs a Hermes session on the fixed build, not a throughput bench.

### 2. Root cause of the freeze: the observe BURST, not `low_water`

`d2c3ac43`. References arrive in one burst per route-trace drain (64 steps) but a slot takes a
scheduler TICK to come back, and every placement path lived inside the manager's queue-drain loop —
which breaks on an empty queue. So the manager placed exactly `_low_water` experts per burst,
discarded the other ~4,000 admitted references, and slept while the slots it had just asked for
arrived and sat unused (`free=25` on 11,000 of 11,000 ticks). Rate law, exact at five points:
`promotions/tick == _low_water / ticks-between-bursts`.

Hardware result (counters, unaffected by the SMU): **h 0.3256 → 0.4200**, install rate
**0.39 → ~5/tick**, `free` no longer pinned at 0, `stale_pub=0`.

**Why three diagnoses failed first:** the `free=0 / to_retract=25 / inflight=25` signature was a
SAMPLING ARTIFACT. The summary printed on a *manager-batch* cadence, which only ever happens
mid-burst; the same frozen run sampled at a tick boundary reads `free=25, to_retract=0`. Reporting
is now tick-cadenced. Two of my own attempts were reverted as measured no-ops (`a8b6c95e`) and both
dead ends are recorded in `expert_cache.py` so nobody spends a fourth run on them.

### 3. PLE must come from a 4K-recordsize dataset

`7ee7cfe2`. A PLE row is 120 B and decode reads 16/token. The checkpoint ships its own PLE shards,
so `ours.yml` pointed `/ple` at `q4e-mxfp4` on `zpcachyoshome/models/hf` at **recordsize=1M**.
Measured: **418–444 KiB of device traffic per row vs 17–24 KiB on 4K (18–24x)** — ~240 MiB/s of
random reads on the same ZFS pool that serves `/home`. It froze the desktop for seconds at a time
(PSI `io some=8.36`) and the user abandoned a Hermes session over it.

`zpcachyoshome/models/ple-mxfp4` (4K, 44G) now holds a rewritten copy; `Q4E_PLE_DIR` points there.

**THE TRAP:** `zfs_bclone_enabled=1`, so a plain `cp` **block-clones** and silently inherits the
source's 1M geometry. 47 GB "copied" in under 30 s, pool `avail` never moved, `zfs get recordsize`
said 4K the whole time, and a probe showed zero improvement. Use `cp --reflink=never` and verify
with `zdb -ddddd <dataset> <inode>` reading `dblk`. Never trust the property.

Note this is a LOCAL I/O win, not an engine-gap explanation: their fork reads its row file once into
a bounded mlock'd RAM arena, so recordsize barely touches it. We `pread` per row against a shared,
record-granular, 8 GiB-capped ARC.

### 4. Capture-safe route trace (`5e0aca77`) — GRAPH_BS still 0

Three defects, one independently live:

* **(A) `ring_rows` defaulted to 1 without spec**, and `record()` takes the sync-free device ring
  only when `M <= ring_rows`. Host-path records land in `oversize`, which `drain()` packs to file
  and **never forwards to the observer** — so on this arm (`--max-running-requests 2`) every
  concurrent decode step was invisible to the expert cache. Leading candidate for h plateauing at
  0.4067 against a 0.5425 offline ceiling.
* **(B)** the ring slot was host Python, so capture baked it and every replay wrote the same slot.
  Now a per-step STAGE buffer harvested at the step boundary.
* **(C)** a captured bucket's padded rows carry garbage routing; masked at harvest, where the real
  row count is known.

**NOT VALIDATED ON HARDWARE.** No real `torch.cuda.CUDAGraph` was ever exercised (the test image
has no GPU), so HIP's acceptance of the recorded slice-assign into a tensor outside the graph's
private pool is untested. The one boot that ran did complete `graph_capture 22.00 s` with the arena
clean afterwards and the cache fed at full rate — but its throughput numbers are SMU-contaminated.

Two residual gaps, not applicable at `EP=0/DP=1`, now visible via `rows_dropped`: `ring_rows` does
not scale with `dp_size` under EP, and `record()`'s `rows_mismatch` comment overclaims for EP.

### 5. `tools/missing_target_probe.py` (`1fed726d`) — a detector the panel was blind to

Every detector in `toolcall_degen_probe` is SHAPE-based, so an invented but plausible document name
reads as clean. A Flash-Next session hallucinated document names on its **second turn** with
`junk_args`/`id_noise`/`think_zero` all at 0.0%. This reads the tool result's STATUS FIELD instead.
Base rate is NOT zero (Laguna-XS 23.3%, deepseek-v4-pro 13.0%, most models 0.0%) because it also
counts benign path-guessing, so it is only meaningful as a **cross-arm comparison on the same task**.

Also: **degeneration is not gated on accumulated context.** Turn 2 is a live window.

## WHERE THE ENGINE GAP STANDS

Theirs 24–26 tok/s, ours ~16–20 (pre-SMU-failure figures). A 13-agent analysis attributed ~63% and
was explicit that the largest term rests on an unmeasured number:

* ~12 ms/token (±3) expert-residency deficit — ours h measured, theirs **deduced**. Their
  `R4D_LRU_TELEMETRY=1` defaults off, so their true hit rate has never been measured. **One env var
  on their next baseline settles whether this is a residency or a compute problem — do this first.**
* 3.45–3.66 ms/step graph capture, MEASURED, but on a ~60 ms step in a different config — treat as
  an upper bound.
* **~10 ms (37%) UNACCOUNTED**: a non-expert on-device compute differential nobody has measured on
  their engine at all. Our side closes to within 3–9 ms; theirs does not close.

REFUTED and not to be re-proposed: VRAM budget (we already hold ~2× their expert residency), PLE as
an engine difference (they pay the same 1M recordsize), launch count, GEMM tiling, `--expert-direct-load`.
A routing-profile pinned residency plan was ABLATED on our own trace and is **worse** than our LRU
(their scheme 0.6235 vs our global pool 0.7184).

## OPEN, IN ROUGH PRIORITY ORDER

1. **Re-measure everything post-reboot.** A peer session holds both GPUs for the `GRAPH_BS=0`
   control vs `GRAPH_BS=2` arms plus a ~120k prefill gate. Do not boot a serve until it reports.
2. **The 120k prefill gate with capture on** at the shipped `MEM_RATIO 0.85`. The historical failure
   is `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION` ~1 s into the first chunk, killing both ranks,
   and it was LENGTH-gated at 16,382 — so the gate must be ~120k, not 16k. `serve.sh` says
   `GRAPH_BS` stays 0 until QSA stage 4b is row-tiled; **that landed** (`_ATTN_ROW_TILE = 256`,
   `qsa/runtime.py:86`) and `GRAPH_BS` was never re-derived. The graph reserve itself is ~2 MiB at
   `GRAPH_BS=2` (0.14% of the KV pool) — the old "78% of context reach" was the `MEM_RATIO` cut, not
   the graphs.
3. **Run the Hermes prompt against the fixed build.** The decisive functional cell, and the only way
   to test whether the wrong-bytes defect was the degeneration. Functional probes pass on the
   degrading arm when fresh and prove nothing.
4. **The unlocalised remainder of the cache gap:** h 0.42 against a 0.5425 offline ceiling.
5. `_W4A8MoEMethod`, `_FP8MoEMethod`, `_UnquantizedMoEMethod` still declare no `cache_plane_attrs`,
   so the expert cache is silently OFF for those formats.
6. The 52.3 GiB mlocked host arena (26.15 GiB × 2 ranks of 93 GiB) pushed 17.9 GiB of the user's own
   applications into swap. Not the cause of the desktop freezes (the recordsize was), but a real
   risk on any TP>1 offload serve — check `VmLck` × ranks against RAM first.

## OPERATIONAL

* `~/.config/systemd/user/claude-remote-control.service` resumes this conversation with Remote
  Control after a reboot. **It worked** — fired at 11:27:50 into tmux `claude-rc`. It must be wrapped
  in tmux: without a pty `claude` silently falls back to `--print` and dies with "Input must be
  provided either through stdin or as a prompt argument", which reads like a config error.
  A resumed session does NOT restore background tasks, monitors or workflows.
* `claude-code` updated 2.1.251 → 2.1.280.
* Harness scripts lived in `/tmp/claude-1000` and are GONE. Rebuild in a durable location — this
  is the second time non-durable storage has cost measurements
  (`measurement-fixtures-must-be-durable-and-recorded`).

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01SNE3yUKbzwTaAZVrPEDJHc
