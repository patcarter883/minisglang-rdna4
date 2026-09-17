# CONTINUANCE — GPU tuning + custom_ar comms (reboot required)

_Written 2026-07-17 before a reboot to recover a hung GPU0._

## ⚠️ REBOOT REQUIRED — GPU0 is hard-hung

- **GPU0 = RX 9070 XT (PCI 03:00.0)** hard-locked during a **memory-OC sweep** when `max_memory_clock`
  hit **1450 MHz** (past the GDDR6 EDC cliff). The amdgpu self-reset **FAILED**:
  `GPU reset end with ret = -19 (ENODEV)`, `failed to reset legacy queue`. No `reset` sysfs node.
  → **Only a reboot recovers it.** GPU1 (RX 9070, 07:00.0) stayed healthy.
- **LACT config was RESTORED to a safe state** and applies cleanly on boot:
  `performance_level: auto`, `max_memory_clock: 1354`, `voltage_offset: -10` (both cards).
  Backup of the pre-sweep config: `/tmp/lact_config.orig.*` and `/etc/lact/config.yaml.bak.*`.

### After reboot — verify first
```
gpu-status                       # both cards should be FREE / responsive
rocm-smi | grep -E '^[01] '      # GPU0 no longer N/A/unknown
```

## Memory OC — ANSWERED (do NOT re-sweep aggressively)

Effective **read bandwidth** vs `max_memory_clock` (−10 mV held, integrity clean unless noted):

| mclk | BW_READ | note |
|------|---------|------|
| 1258 (stock) | 583 GB/s | |
| 1300 | 605 | |
| 1350 | 627 | (≈ current 1354 setting) |
| **1400** | **~645** | **peak, integrity clean** |
| 1450 | 150 | EDC cliff: quintillions of integrity errors + **hung GPU0** |

- **Reliable memory OC = `max_memory_clock: 1380`** (peak −1 margin step, ~+10% BW over stock). Current
  1354 is already safe. **Never exceed 1400.**
- Tooling (in worktree `minisgl-rdna4-carcomms/tools/`): `gpu_tune_probe.py` (bw/integrity/stab),
  `gpu_tune_sweep.sh` (LACT-driven: edits `config.yaml` + restarts lactd per step, perf stays `auto`,
  restores original on exit). **BUG in the sweep:** steps were too coarse (1400→1450 jumped the cliff
  into a hang). Fix: ≤15 MHz steps near the top and **STOP at the first BW drop** — 1400 was already the
  peak, there was no reason to try 1450.

## Undervolt sweep — NOT DONE (deferred, risky)

- Defer until after reboot. It is **more** hang-prone than the memory sweep (core-voltage instability →
  another failed-reset reboot). Expect ≥1 hang/recovery cycle.
- Method: hold `max_memory_clock` safe (1320), step `voltage_offset` −10 → −30 → −50 … via LACT, run the
  `stab` probe (boost-clock golden-GEMM), STOP at first `STAB_MISMATCH>0` / hang. Back off one step.
- Current −10 mV is stable. The pre-existing config ran −15 mV. Only push lower if the risk is acceptable.

## custom_ar comms work — validated, UNCOMMITTED, in worktrees

Worktrees: `minisgl-rdna4-carcomms` (branch `feat/car-comms`, off rdna4 04691a8) +
`rdna4-hip-kernels-carcomms` (branch `feat/car-push-fp8-gather`). See memory
`custom-ar-comms-push-fp8-gather.md`.

- **push-flip: REVERTED to the old PULL kernel** (write-self / read-peer). RDNA4 can't keep the
  push-then-read-local pattern coherent for bf16 under serve load (silent truncation). Default
  `one_shot_ar` = pull = proven coherent.
- **fp8** (`MINISGL_CAR_FP8=1`): WORKS. 35B TP2 serve coherent (lossy 1.87%, differs 3/8 prompts).
  **+2.0% tok/s** on 35B. Per-token e4m3, fixed 8-block grid, system-scope coherent loads. Opt-in.
- **custom all_gather** (`MINISGL_CAR_ALLGATHER=1`): rewritten to PULL, **token-identical** to RCCL,
  **+1.2%** on 35B. Costs vocab-sized IPC VRAM (row-cap `MINISGL_CAR_AG_MAX_ROWS`, default 32; use 8 on 35B).
- **bf16-gather** (`MINISGL_CAR_LOGITS_BF16=1`): token-identical, throughput flat.
- **dist-argmax**: `ParallelLMHead.argmax_tp` primitive, op-exact, **NOT serve-integrated** (needs a
  token-output capture graph). No serve number.
- **Build**: `.so` MUST be built in `minisgl-rdna4:lean` (torch 2.14), NOT `vllm22-w4a8:combined`
  (torch 2.10, ABI-incompatible + no msgpack — that's the vLLM image, wrong for serving). Build:
  `docker run --rm -v <kernels-wt>/custom_ar:/kernels --entrypoint bash minisgl-rdna4:lean -lc
  'export PATH=/opt/venv/bin:$PATH; cd /kernels && rm -rf build && GPU_ARCHS=gfx1201 bash local/build_local.sh'`
  (clear `build/` first — stale objects cause torch-ABI undefined-symbol at load).
- **Bench verdict (35B TP2, full clocks)**: comms is a MINOR lever — fp8 +2%, all_gather +1.2%, rest
  noise. 35B decode is compute/memory-bound, not comms-bound.

### Serve/bench recipe (35B, lean image, TP2, graph)
`tools/car_bench.sh` (5 modes: rccl/pull/bf16gather/allgather/fp8 — coherence + tok/s).
`tools/car_full_validate.sh` = op-verify + serve gate. `tools/car_comms_verify.py` = 2-GPU op tests
(all PASS: push-flip pull, fp8 per-token, custom all_gather pull, dist-argmax, stress).

## Next steps after reboot
1. Verify both GPUs recovered.
2. (Optional) set `max_memory_clock: 1380` in LACT (both cards, keep −10 mV / auto).
3. (Optional, risky) run the undervolt sweep with ≤ tiny steps, or accept −10 mV.
4. **Decide whether to COMMIT the custom_ar comms work** — fp8 (+2%) + pull all_gather (+1.2%) are the
   ship-worthy pieces (opt-in flags, default off); push-flip stays reverted; dist-argmax needs the
   token-output-graph integration to be usable.
