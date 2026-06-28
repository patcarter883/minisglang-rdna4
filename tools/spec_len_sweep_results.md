# Spec-decode length sweep — wall-clock tokens/sec @ batch=1 (2026-06-28)

Measured with `tools/run_spec_len_sweep.sh` (one container per model, re-boots the server per K,
times 3×256-token batch=1 greedy generations after a warmup). Baseline = **no-spec with CUDA graph
ON** (the real production decode path); spec is **eager** (the engine disables graph capture when
spec is set), so spec must beat the graphed baseline on *net* tokens/sec — emit/step alone is not
enough. `accept` = cumulative draft accept_rate; `emit/step` = tokens committed per decode step.

## GLM-4.7-Flash-AWQ — TP=2, MLA+MoE
| algo | K | seed | tok/s | emit/step | accept | vs baseline |
|------|---|------|-------|-----------|--------|-------------|
| none (graph) | – | – | **35.70** | – | – | 1.00× |
| MTP | 1 | – | 20.61 | 1.43 | 0.43 | 0.58× |
| MTP | 2 | – | 25.25 | 1.70 | 0.35 | 0.71× ← MTP peak |
| MTP | 3 | – | 23.07 | 1.70 | 0.23 | 0.65× |
| MTP | 4 | – | 21.54 | 1.70 | 0.18 | 0.60× |
| MTP | 6 | – | 20.28 | 1.70 | 0.12 | 0.57× |
| MTP | 8 | – | (hung) | – | – | – |
| EAGLE3 | 1 | on | 22.51 | 1.55 | 0.56 | 0.63× |
| EAGLE3 | 2 | on | 31.88 | 2.10 | 0.55 | 0.89× |
| EAGLE3 | 3 | on | 35.22 | 2.48 | 0.50 | 0.99× |
| EAGLE3 | 4 | on | 34.98 | 2.53 | 0.39 | 0.98× |
| EAGLE3 | 6 | on | **37.23** | 2.81 | 0.31 | **1.04× ← EAGLE3 peak** |
| EAGLE3 | 8 | on | 35.07 | 2.90 | 0.24 | 0.98× |

- **EAGLE3 (seed ON) is the only GLM spec config that beats the graphed baseline: optimal K≈6, +4.3%.**
- **MTP is a net LOSS on GLM-AWQ** (peak 0.71× at K=2): the AWQ-quantized MTP MoE drafts too weakly
  (emit/step saturates at 1.70), and the eager penalty vs the graphed baseline is never recovered.
- K=8 MTP hung the verify (qlen=9) — a real edge, but far past the optimum.

## Qwen3.5-4B — TP=1, GDN+MoE (bf16 MTP head)
| algo | K | tok/s | emit/step | accept | vs baseline |
|------|---|-------|-----------|--------|-------------|
| none (graph) | – | **23.76** | – | – | 1.00× |
| MTP | 1 | 31.43 | 1.76 | 0.77 | 1.32× |
| MTP | 2 | 35.90 | 2.32 | 0.67 | 1.51× |
| MTP | 3 | **36.66** | 2.61 | 0.55 | **1.54× ← peak** |
| MTP | 4 | 34.55 | 2.68 | 0.43 | 1.45× |
| MTP | 5 | 32.37 | 2.73 | 0.35 | 1.36× |
| MTP | 6 | 31.59 | 2.86 | 0.32 | 1.33× |

- **MTP on Qwen3.5-4B is a big win: optimal K≈3, +54%** (23.76→36.66 tok/s). The bf16 MTP head
  drafts strongly (accept 0.55–0.77), unlike GLM's AWQ MTP.

## Qwen3.6-35B-A3B-AWQ — TP=2, GDN+MoE (compressed-tensors W4A16 MoE, brought up 2026-06-28)
Required a bring-up (see QWEN35B_BRINGUP_SCOPE.md): the checkpoint is compressed-tensors int4
weight-only (not AWQ despite the name); added `_GroupedCompressedTensorsExperts` + the loader/shard +
a TP=2 MTP `fc` fix. GPU-validated coherent.
| algo | K | tok/s | emit/step | accept | vs baseline |
|------|---|-------|-----------|--------|-------------|
| none (graph) | – | **25.45** | – | – | 1.00× |
| MTP | 1 | 30.03 | 1.93 | 0.94 | 1.18× |
| MTP | 2 | 37.69 | 2.66 | 0.84 | 1.48× |
| MTP | 3 | 40.57 | 3.09 | 0.71 | 1.59× |
| MTP | 4 | **41.89** | 3.35 | 0.60 | **1.65× ← peak** |
| MTP | 5 | 41.83 | 3.58 | 0.53 | 1.64× |
| MTP | 6 | 36.87 | 3.42 | 0.41 | 1.45× |

- **MTP on the 35B-A3B is the best result: optimal K≈4, +65%** (25.45→41.89 tok/s). The A3B MoE MTP
  head drafts strongly (accept 0.60–0.94) even at int4 — the high active-expert capacity makes a much
  better draft than GLM's AWQ MTP.

## Takeaways — optimal spec length
- **The optimum is small and where emit/step's rise stops outpacing the growing per-step (eager,
  qlen=K+1) verify cost.** Emit/step keeps climbing with K, but tok/s peaks then falls. Use tok/s,
  not accept-rate, to pick K.
- **Optimal K: Qwen3.6-35B-A3B MTP ≈ 4; Qwen3.5-4B MTP ≈ 3; GLM EAGLE3 ≈ 6; GLM MTP ≈ 2 (net loss).**
- **Uplift vs the graphed baseline: Qwen3.6-35B-A3B MTP +65%; Qwen3.5-4B MTP +54%; GLM EAGLE3+seed
  +4.3%; GLM MTP negative.** The MoE-MTP models (4B, 35B-A3B) win big; the AWQ dense-ish MTP (GLM) loses.
- Quantization of the draft is decisive: bf16 draft (Qwen MTP) wins big; AWQ draft (GLM MTP) loses.
  The prefill-seed lever (`MINISGL_SPEC_PREFILL_SEED=1`, used for the EAGLE3 rows) adds ~8.5% tok/step
  on EAGLE3 and is lossless (see SPEC_DECODE.md §"Prompt-prefill draft-KV seed").
