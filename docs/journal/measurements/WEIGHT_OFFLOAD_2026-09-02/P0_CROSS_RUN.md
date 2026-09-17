# P0 — cross-run reproducibility

`p0.json` reports spread **within** one run. This file reports spread **across** the independent runs actually taken, which is the honest uncertainty on the number gates **K4** and **A0.4** read. Nothing here is a new measurement.

| Run | timestamp | schema | status | GPU | valid reps |
|---|---|---|---|---|---|
| `docs/measurements/WEIGHT_OFFLOAD_2026-09-02/p0_run1_schema4/p0.json` | 2026-09-02T14:52:44.595156+00:00 | 4 | ok | confirmed | {'bs1': 5, 'conc6': 5} |
| `docs/measurements/WEIGHT_OFFLOAD_2026-09-02/p0_run2_schema5/p0.json` | 2026-09-02T15:11:43.749764+00:00 | 5 | ok | confirmed | {'bs1': 5, 'conc6': 5} |
| `docs/measurements/WEIGHT_OFFLOAD_2026-09-02/p0.json` | 2026-09-02T15:26:16.547486+00:00 | 5 | ok | confirmed | {'bs1': 5, 'conc6': 5} |

## Median per run

| Metric | run1 (schema 4) | run2 (schema 5) | run3 (schema 5, canonical) | across-run median | spread % |
|---|---|---|---|---|---|
| bs=1 decode tok/s | 4.9406 | 5.2518 | 4.9896 | **4.9896** | 6.24 |
| bs=1 prompt tok/s | 50.3924 | 53.3524 | 49.6828 | **50.3924** | 7.28 |
| bs=1 wall tok/s (prefill incl.) | 4.5825 | 4.9027 | 4.6436 | **4.6436** | 6.9 |
| CONC=6 aggregate tok/s (wall) | 6.8361 | 6.8941 | 6.5505 | **6.8361** | 5.03 |
| CONC=6 aggregate tok/s (sum of decode rates) | 7.5197 | 7.5496 | 7.3233 | **7.5197** | 3.01 |
| CONC=6 per-stream decode tok/s | 1.2588 | 1.2635 | 1.2168 | **1.2588** | 3.71 |

## Pooled repetitions (all runs, warm-ups already discarded by the probe)

| Metric | n | median | min | max | stdev | spread % |
|---|---|---|---|---|---|---|
| bs=1 decode tok/s | 15 | **5.162** | 4.6786 | 5.3534 | 0.2273 | 13.07 |
| bs=1 prompt tok/s | 15 | **52.9761** | 26.8547 | 55.3379 | 6.9351 | 53.77 |
| bs=1 wall tok/s (prefill incl.) | 15 | **4.8135** | 4.2637 | 4.9727 | 0.2326 | 14.73 |
| CONC=6 aggregate tok/s (wall) | 15 | **6.8361** | 6.4434 | 7.0142 | 0.1776 | 8.35 |
| CONC=6 aggregate tok/s (sum of decode rates) | 15 | **7.5155** | 7.1906 | 7.7577 | 0.165 | 7.55 |
| CONC=6 per-stream decode tok/s | 90 | **1.2567** | 1.179 | 1.2988 | 0.0292 | 9.53 |

**Canonical run** (`run3 (schema 5, canonical)`, the one `p0.json` points at): bs=1 decode **4.9896 tok/s**, K4 threshold **3.178 tok/s**, 0.1476× the plan's derived 33.8 tok/s ceiling.

Recomputed on the pooled median instead, K4 would be **3.288 tok/s** (+3.46% from the canonical run). The two agree well inside the across-run spread, so no gate turns on which run is cited.

Regenerate: `python3 tools/offload/p0_cross_run.py`
