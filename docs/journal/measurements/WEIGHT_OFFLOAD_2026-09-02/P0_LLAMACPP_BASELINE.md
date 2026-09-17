# P0 — llama.cpp baseline on the target checkpoint

**Probe:** P0 (WEIGHT_OFFLOAD_PLAN.md §3). **Status:** `ok`. **Run:** 2026-09-02T15:26:16.547486+00:00 on `blue`.

> This is the denominator of the entire weight-offload project. Gates **K4** (`M1 tok/s < 0.64 × llama.cpp` → hard kill) and **A0.4** (`byte hit rate ≥ max(50 %, break-even vs P0)`) both read the number below.

## Verdict

P0 baseline: llama.cpp bs=1 decode = 4.990 tok/s on the target checkpoint. The plan's DERIVED 33.8 tok/s CPU-expert ceiling is NOT reachable in practice: measured is 6.77x below it. Break-even math must use the measured number, not 33.8. Against this baseline, a pure zero-cache host-streaming tier already wins: the required byte hit rate h is 0. The 40% break-even in Section 1(a) is an artifact of comparing to a 45 GB/s DDR abstraction llama.cpp does not achieve here. K4 hard-kill threshold for M1: 3.178 tok/s. CONC=6 aggregate = 6.551 tok/s wall (1.411x the bs=1 WALL rate, 24% of linear; matched bases).

## Measured

| Leg | Metric | Median | Min | Max | Spread % | Valid reps |
|---|---|---|---|---|---|---|
| bs=1 | decode tok/s | 4.9896 | 4.6786 | 5.333 | 13.12 | 5/5 |
| bs=1 | prompt tok/s | 49.6828 | 26.8547 | 54.6051 | 55.86 | 5/5 |
| bs=1 | wall tok/s (prefill incl.) | 4.6436 | 4.2637 | 4.9315 | 14.38 | 5/5 |
| CONC=6 | aggregate tok/s (wall) | 6.5505 | 6.4434 | 6.7056 | 4.0 | 5/5 |
| CONC=6 | aggregate tok/s (Σ server decode rates) | 7.3233 | 7.1906 | 7.4907 | 4.1 | 5/5 |
| CONC=6 | per-stream decode tok/s | 1.2168 | 1.179 | 1.2572 | 6.43 | 30 streams |

One warm-up repetition was discarded in every leg. `ignore_eos` forces exactly `n_predict` tokens per rep so reps are comparable; a fresh 16-byte nonce at token ~8 of every prompt defeats prefix reuse, `cache_prompt:false` is set per request, and the server-side prompt cache is disabled with `--cache-ram 0`. Whether the reps are actually steady state is checked, not assumed — see below.

## Does the plan's derived 33.8 tok/s ceiling hold?

- Plan §1 lists **33.8 tok/s** as the *llama.cpp CPU-expert ceiling*, explicitly marked **"derived — must be measured"** (45 GB/s DDR ÷ 1.33 GB/token).
- Measured here: **4.9896 tok/s**, i.e. **0.1476×** the derived ceiling (6.77× short).
- Reachable in practice: **False**.

The derived ceiling assumes the CPU runtime is DDR-bandwidth-bound at 45 GB/s. Whether it is, on this box, is settled by the measured I/O attribution below — not asserted here. What is certain either way: any break-even or K4 arithmetic that uses 33.8 is wrong by the shortfall factor above.

### Measured I/O attribution — what actually bound the baseline

| Leg | tokens generated | server read GB / token (per-PID) | nvme read GB / token (box-wide) | fraction of THIS checkpoint's expert bytes | major faults / token |
|---|---|---|---|---|---|
| bs1 | 600 | 0.005 | 0.0064 | 0.004288 | 1015.58 |
| conc6 | 3600 | 0.0069 | 0.0043 | 0.005895 | 573.61 |

Read this as the discriminator between the two candidate explanations of a low number: **CPU-compute bound** (i-quant expert GEMV on 8 cores) vs **storage bound** (a 93.683 GB checkpoint mmap'd on a box with less RAM than that, through a capped ZFS ARC). A near-zero read-per-token says compute; a fraction approaching 1 says the expert bytes are coming off disk every token.

The denominator is this checkpoint's OWN active expert bytes per token, **not** the plan's 1.33 GB — that constant is arithmetic on the minisgl target shape (48L × 10 × 2.8 MB) and a different packing from UD-IQ4_XS, so a ratio against it would compare across quantizations. The plan-constant ratio is kept in the JSON as `fraction_of_PLAN_constant_1p33GB_NOT_THIS_PACKING` for contrast only. `server read` is process-attributed (`/proc/<pid>/io`); `nvme read` is box-wide and includes any other job's traffic in the same window.

**Denominator provenance (gguf_tensor_table_exact).** This llama.cpp build emits no `print_info:`/`load_tensors:` lines, so the geometry is read from the checkpoint itself: an exact walk of the GGUF tensor table summing every `*_exps` tensor at its own ggml type size — **59.5198 GB** of expert weights (63.5% of the checkpoint) over 48 MoE layers, 512 experts, top-10 routed → **1.1625 GB per decoded token**. The walk is validated on the operation, not the query: it reconstructs 99.99% of the on-disk shard bytes (93.6826 GB) and is rejected outside ±1 %. Expert byte breakdown by quant type (GB): {'IQ4_NL': 20.2899, 'IQ3_S': 33.8821, 'Q8_0': 4.4564, 'IQ4_XS': 0.8913}. The cruder pro-rata bound (`total × k/E`) would have said 1.8295 GB/token — 1.57× too high, because it charges attention/embedding/PLE bytes to the expert term.

## Attribution — was this actually the 2-card `--fit` configuration?

**Verdict:** `confirmed`. [bs1] weights on device: None GB in the log's ROCm buffers; VRAM grew on ['card0', 'card1'] across model load.; [conc6] weights on device: None GB in the log's ROCm buffers; VRAM grew on ['card0', 'card1'] across model load.

| Leg | log ROCm weight GB | log CPU weight GB | layers offloaded | VRAM delta across load (GB) | devices enumerated |
|---|---|---|---|---|---|
| bs1 | None | None | None/None | {'card0': 15.991, 'card1': 15.691} | None |
| conc6 | None | None | None/None | {'card0': 15.71, 'card1': 15.722} | None |

A plausible tok/s number does not distinguish the claimed configuration from a run where the HIP backend failed to register, where `--fit` offloaded nothing, or where the Ryzen iGPU (47 GB of GTT) leaked into enumeration and poisoned `--fit` auto-sizing. This section asserts on the **operation** — weight bytes on device and VRAM that actually moved — not on the flags having been passed.

## Steady state — is the median a rate or a point on a ramp?

| Leg | steady state | 1st-half mean | 2nd-half mean | drift % | strictly monotone | reps in order |
|---|---|---|---|---|---|---|
| bs1 (decode_tok_s) | True | 5.1613 | 4.7703 | -7.57 | False | [4.9896, 5.333, 5.1013, 4.6786, 4.862] |
| conc6 (aggregate_tok_s_wall) | True | 6.628 | 6.5644 | -0.96 | False | [6.7056, 6.5505, 6.4434, 6.4734, 6.6554] |

One 100-token warm-up cannot warm a 93.7 GB working set on a box with ~50 GB available, so this table is load-bearing: a trending series means the median is a point on a curve and every gate that reads it inherits the error.

## Break-even hit rate, recomputed against the measured baseline

| Compute floor | Required byte hit rate `h` to beat llama.cpp |
|---|---|
| 0 ms | **0 % — beaten with no cache at all** |
| 5 ms | **0 % — beaten with no cache at all** |
| 10 ms | **0 % — beaten with no cache at all** |

For contrast, the plan's §1(a) figure — computed against a 45 GB/s DDR abstraction rather than a measured runtime — is **40.44 %**.

**K4 hard-kill threshold for M1:** 3.178 tok/s at bs=1.

## Concurrency scaling — matched bases only

| Comparison | bs=1 | CONC | ×bs=1 | % of linear |
|---|---|---|---|---|
| wall vs wall (prefill included) | 4.6436 | 6.5505 | 1.411 | 24 % |
| decode vs Σ decode rates (prefill excluded) | 4.9896 | 7.3233 | 1.468 | 24 % |

The naive `CONC wall ÷ bs=1 decode` ratio would have been **1.313** — it divides a prefill-inclusive aggregate by a prefill-exclusive rate and is recorded only so it is not mistaken for a result.

## Box state — recorded with every leg (the box was NOT idle)

| Window | Δ major faults | Δ pswpout | Δ pswpin | Δ nvme read (GB) | avg nvme read GB/s | Δ MemAvailable (GB) |
|---|---|---|---|---|---|---|
| bs1 | 609347 | 127939 | 118356 | 3.817 | 0.029 | -2.67 |
| conc6 | 2065000 | 34400 | 228734 | 15.363 | 0.028 | -4.58 |

At probe start: MemTotal 91.84 GB, MemAvailable 68.02 GB, SwapFree 33.55 GB. Checkpoint on disk: 93.683 GB across 3 shards (**larger than installed RAM**), on a ZFS pool with a 16.0 GiB ARC cap.

## Cards

- card0: AMD Radeon RX 9070 XT (vram_total 17.096 GB, 0.06 GB in use at probe start)
- card1: AMD Radeon RX 9070 (vram_total 17.096 GB, 0.06 GB in use at probe start)
- ROCR_VISIBLE_DEVICES=0,1 -- card2 (Ryzen 7800X3D iGPU) fenced out of enumeration

## Prior figure carried as a citation only

- Cited: **4.58 tok/s** decode, **11.1 tok/s** prompt.
- Source: session note recorded in the weight-offload Phase-0 brief, 2026-09-02: 'llama.cpp on the target checkpoint on THIS box is 4.58 tok/s at bs=1 (decode), prompt 11.1 t/s, measured this session via lemonade + llama.cpp rocm-nightly, 2 cards, auto-fit offload.'
- Provenance quality: NOT a durable fixture: no log, no JSON, no recorded box state, no rep count, no spread, no record of which cards or which llama-server argv. This probe exists to replace it with one.
- Measured by this script: `False`.
- Reproduced by this run: **True** (fresh median 4.9896 tok/s, 8.94 % from the cited value).

## Reproduce

```
python3 tools/offload/p0_llamacpp_baseline.py 
```

llama-server launch lines (one per leg):

```
# bs1
/home/pat/.cache/lemonade/bin/llamacpp/rocm-nightly/llama-server -m /home/pat/.cache/huggingface/hub/models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots/5d16c055a7c5cb276e721ee154f9c22420dde2a1/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf --host 127.0.0.1 --port 18099 -np 1 -c 8192 -fit on -fa auto -t 8 --no-webui --metrics --cache-ram 0 -sm layer
# conc6
/home/pat/.cache/lemonade/bin/llamacpp/rocm-nightly/llama-server -m /home/pat/.cache/huggingface/hub/models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots/5d16c055a7c5cb276e721ee154f9c22420dde2a1/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf --host 127.0.0.1 --port 18099 -np 6 -c 8192 -fit on -fa auto -t 8 --no-webui --metrics --cache-ram 0 -sm layer
```

Raw data: `p0.json` (schema 5). Server logs: `p0_server_*.log`.

