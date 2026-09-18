# Continue here after the reboot — 2026-09-18

The box was rebooted to (a) recover a hung SMU on GPU 0 and (b) pick up kernel parameters that
make the GTT host-memory path usable. Nothing below is speculative: every number is measured and
the commit that carries it is named.

## Why we rebooted

1. **SMU hang on GPU 0.** `amdgpu 0000:03:00.0: SMU is in hanged state, failed to send smu message`
   (card1 = the RX 9070 XT = compute GPU 0). Compute was unaffected; telemetry was dead. Consequence
   worth remembering: `lease wedge` read `util=0% mem=-1% power=-1W` and returned `ok`, so it did
   **not** false-positive — but it was **blind** on that card, because a real wedge can never
   satisfy its `util >= 97%` condition when the SMU is not answering.
2. **GTT needs boot parameters.** `amdgpu.gttsize` is fixed at driver probe, which is why raising
   `/sys/module/ttm/parameters/pages_limit` at runtime changed nothing (`mem_info_gtt_total` stayed
   at 45.92 GiB and the serve still failed).

## Verify these FIRST, before anything else

```
cat /proc/cmdline | tr ' ' '\n' | grep -E 'gttsize|pages_limit'   # expect both
cat /sys/class/drm/card1/device/mem_info_gtt_total                # expect ~64 GiB, was 45.92
rocm-smi --showpower                                              # GPU 0 should report again
lease wedge                                                       # card 0 should give real numbers
```

If `mem_info_gtt_total` is still 45.92 GiB the parameters did not take — check
`/etc/default/limine` (backup of the pre-change file is in the session scratchpad) and re-run
`sudo limine-update`.

## THE ROOT CAUSE WE FIXED (read this before touching the offload path)

**The "pinned host arena" was never pinned.** `hipHostMalloc` takes ROCR's userptr path
(`KFD_IOC_ALLOC_MEM_FLAGS_USERPTR`), which is HMM-managed — KFD registers an MMU notifier and
re-validates on invalidation instead of taking a page pin. `/proc/<pid>/status` showed `VmPin` and
`VmLck` both **0.00** for a rank holding a 29.91 GiB resident arena. 2 x 27.94 GiB of ordinary
swappable anon on a 91.8 GiB box.

Measured symptom (qwen4exp TP=2, one queued request):

* `minisgl_prefill_computed_tokens_total` **0 for 12+ minutes** — prefill never started
* 50-100 MB/s of swap **in both directions**, continuously; zswap disabled, so all real disk I/O
* swap occupancy frozen at 55.0 GiB, **37.0 GiB of it `SwapCached`** (faulted back in, slot retained
  — which is why occupancy cannot fall while swap-in runs, and why "swap used" looked stuck)
* ranks' resident anon oscillating **46.5 <-> 62.7 GiB**

Fixed by `mlock`ing at allocation (`0d1a1771`). Verified: `VmLck` 0.00 -> 5.36 GiB for 4 x 1.34 GiB,
released on free, 0.9 s. `5bd01e1d` skips the mlock on the GTT path, where the memory is
driver-allocated and already unswappable.

**The swap tripwire cannot catch this** — it watches ALLOCATION, and this happens on first use.

## The A/B to run first

Two routes to the same trade (56 GiB genuinely unavailable instead of nominally available and
thrashing). Compare them, warm-to-warm:

```
# A: userptr + mlock   (the current default)
repo=minisgl profile=serve cards=2  MODEL=q4e SPEC=none CONC=2 TP=2
# B: GTT               (needs the new cmdline)
... plus  HSA_USERPTR_FOR_PAGED_MEM=0
```

What to record for each: `[boot-timeline] box:` line (swap in/out), time to first prefill token on a
long prompt, and `tok/s` from `/home/pat/fixtures/minisgl-ring-proper/measure.py`.

**MEASURE WARM-TO-WARM.** The first run after any boot reads ~13.5-13.8 tok/s with the PLE table
cold in ARC; the second on the same serve reads ~17.6. A cold first run against a warm baseline
reads as a 24% regression that does not exist. This has bitten twice.

## Known-good reference numbers (all warm, SPEC=none, CONC=2, TP=2, expert cache 2.5 GiB)

| build | tok/s | range |
|---|---|---|
| pre-session baseline | 17.76 | 17.55-18.63 |
| threaded decode gather | 17.57 | 17.1-18.7 |
| + begin/finish overlap | 17.74 | 17.4-19.0 |

Boot swap (`[boot-timeline] box:`), same arena: MTP leg 15.54 in / 19.19 out; `defrag=defer` leg
44.68 / 50.24; the io_uring leg 4.62 / 8.96. **None of these had the mlock fix** — expect all of
them to change, and do not compare across them without re-baselining.

## Open, in priority order

1. **Does mlock actually kill the pre-prefill wait?** That is the whole point of `0d1a1771` and it
   has NOT been tested end to end — the box came down before a serve ran with it.
2. **GTT vs mlock**, per the A/B above.
3. **MTP on qwen4exp is still unmeasured.** It OOMs on VRAM at the validated operating point: the
   MTP head is ~4.86 GiB and `plan.OFFLOAD_MTP_HEAD = False` makes it device-resident by policy,
   against ~11.97 GiB already resident on a 15.92 GiB card. Shrinking `EXPERT_CACHE_GB` to fit it is
   what deadlocked the cache (that specific wedge is now guarded — `expert_cache.py` clamps
   in-flight to `slots//3`, and `serve.sh` prints any deviation from the table — but the VRAM
   shortfall is untouched). The table's own precondition for re-measuring spec ("only after a
   device-resident expert cache") is now satisfied for the first time, because the route-trace ring
   never recorded verify steps until this session.
4. **`MAX_PREFILL_LENGTH` is 2048** (`serve.sh:1082`) while the same file argues for 8192 and the
   blocker it cites (`b7f9ae39` row-tiled stage 4b, `_ATTN_ROW_TILE = 256`) is gone. Worth doing —
   but it is a mitigation for the per-chunk expert sweep, NOT for the swap problem above. Do not
   conflate them.
5. **io_uring and registration are ON by default** for gathers >= 256 rows (`434fa256`), by
   decision, not by a settled measurement — three probe runs disagreed because the probe's own
   iowait swamped the signal. Decode (16 rows) never reaches the threshold. To settle: a quiet box,
   cache state equalised per arm, windows long enough that p90 stops being 3x the median.
6. **`tests/fixtures/qwen4exp/ngram_meta.json` is excluded by `.gitignore:11` (`*.json`)**, so
   `qwen4exp_ple_test.py` and `_ple_hash_test.py` cannot run in a fresh checkout. Pre-existing.

## Tools added this session

* `tools/offload/ple_readpath_probe.py` — mechanism comparison on the real table. **Arms are
  interleaved on purpose**: running each arm's block back-to-back warms the ARC monotonically and
  hands the win to whoever went last. Doing it wrong made one run report serial pread 8x faster than
  the same probe had minutes earlier. Keep the interleave.
* `tests/ple_row_table_pread_parity_test.py` (44 checks) — all four gather mechanisms byte-identical,
  plus the async split and the shipped default path.
* `tests/uring_backend_test.py` (15 checks) — every io_uring mode, asserting the **opcode and flags
  actually used**, not just the bytes: a mode that silently fell back to a plain READ would
  otherwise pass green.
