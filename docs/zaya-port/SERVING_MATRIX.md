# ZAYA1-8B-fp8 serving matrix (concurrency × context) — for RSA serving limits

Measured on one gfx1201 (RX 9070 XT, 16 GB), native Triton-free path: `--attn hip` + W8A8-fp8 MoE
kernel + CUDA graphs (bs 1–16), `MINISGL_MOE_SCATTER=0`, chunked prefill (`max_extend_tokens=2048`),
`memory_ratio=0.90`. Tool: `tools/zaya_serving_matrix.py`. Decode TPOT isolated as
t(1+STEPS)−t(1) over STEPS=16. **KV pool = 65,615 tokens (5.0 GiB).** No Triton compiled (verified).

Feasibility frontier (decode): `B × L_context ≤ 65,615`. (Chunked prefill keeps prefill-activation
memory bounded, so the KV pool — not prefill — is the real concurrency limit.)

## Decode TPOT (ms)  [rows = context L, cols = concurrency B]
| L \ B | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| 1024  | 25.5 | 25.4 | 28.8 | 29.3 | 29.7 |
| 4096  | 32.6 | 32.0 | 33.2 | 33.7 | — (over-KV) |
| 8192  | 40.9 | 40.4 | 42.6 | — | — |
| 16384 | 61.7 | 59.5 | — | — | — |
| 32768 | 92.3 | — | — | — | — |

## Decode throughput (tok/s, aggregate over the batch)
| L \ B | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| 1024  | 39 | 79 | 139 | 273 | **539** |
| 4096  | 31 | 63 | 121 | 237 | — |
| 8192  | 24 | 49 | 94 | — | — |
| 16384 | 16 | 34 | — | — | — |
| 32768 | 11 | — | — | — | — |

## Prefill latency (ms, for B×L tokens, chunked at 2048)
| L \ B | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| 1024  | 244 | 416 | 831 | 1661 | 3324 |
| 4096  | 1082 | 2165 | 4331 | 8664 | — |
| 8192  | 2870 | 5749 | 11455 | — | — |
| 16384 | 8641 | 17326 | — | — | — |
| 32768 | 30253 | — | — | — | — |

## Reading it
- **Decode batching is almost free.** TPOT rises only 25.5→29.7 ms from B=1→16 at L=1024, so aggregate
  decode throughput scales ~linearly with concurrency (→539 tok/s at B=16). Decode is memory-/latency-
  bound, not compute-bound.
- **Context dominates decode TPOT.** 25→33→41→62→92 ms at 1k→4k→8k→16k→32k (attention over the KV).
- **Prefill is the heavy, super-linear cost.** ~0.24 ms/tok at L=1k rising to ~0.92 ms/tok at L=32k
  (intra-prompt attention is O(L²)). E.g. N=8 × 4k-token prefill = 8.7 s; a single 32k prefill = 30 s.

## RSA serving limits (one 16 GB card), with the Markovian 4000-token tail
Each rollout runs at `L ≈ 4000 (carried tail) + generation`; context resets each round (Markov), so it
does not accumulate. Concurrency = N rollouts. Frontier `N × L ≤ 65,615`:

| N (rollouts) | max context/rollout | fits 4k tail + gen? | aggregate decode |
|---|---|---|---|
| 16 | ~4,100 | tail only, ~0 gen headroom | ~539 tok/s (~34/rollout) |
| 8  | ~8,200 | 4k tail + ~4k gen ✓ | 237 tok/s @8k-ctx (~30/rollout) |
| 4  | ~16,400 | 4k tail + ~12k gen ✓ | 94 tok/s @8k → 34 @16k |
| 2  | ~32,800 | 4k tail + long gen ✓ | up to 50 tok/s |

- **N=16 is impractical on one card**: 16×4k = 65,536 ≈ the whole KV pool, leaving no room for
  generation past the tail — rollouts would serialize once they generate. Use τ≤2000, cap generation,
  or DP=2.
- **N=8 is the one-card sweet spot**: 4k tail + ~4k generation per rollout fits (~64k), ~237 tok/s
  aggregate decode. Per-round aggregation prefill ≈ 8.7 s (N=8 × 4k) — RSA wall-time is prefill-bound
  at round boundaries, not decode-bound.
- **N=4** gives comfortable long-trace room (~16k/rollout) for deep reasoning at lower concurrency.
- **DP=2 (both cards)** ~doubles the KV pool (~131k aggregate; per-sequence still capped at one card's
  65.6k) → N=16 at the 4k tail + generation becomes viable, or N=8 at 16k.

## DP=2 — measured (two independent replicas, one per card, run CONCURRENTLY)
minisgl is TP-only and CCA can't TP, so ZAYA multi-card = two full replicas + a router (NOT a sharded
engine). Experts replicate (8 GB < 16 GB) → pure DP, no EP, no MoE all-to-all. Decode TPOT under
concurrent dual-card load ≈ solo (no meaningful interference); throughput is ~1.9–2.0× linear:

| cell | replica A (card0) | replica B (card1) | solo | aggregate |
|---|---|---|---|---|
| B=16 L=1024 | 29.7ms / 539 | 30.3ms / 527 | 29.7ms / 539 | **1066 tok/s (1.98×)** |
| B=8  L=1024 | 29.4ms / 272 | 28.8ms / 278 | 29.3ms / 273 | 550 tok/s |
| B=8  L=4096 | 36.1ms / 221 | 34.4ms / 233 | 33.7ms / 237 | **454 tok/s (1.91×)** |

Prefill contends more (~18% slower under concurrent load — overlapping compute-heavy prefill bursts),
decode is near-linear. **RSA N=16 on DP=2** = split 8 rollouts/card at the N=8 sweet spot → ~454 tok/s
aggregate at 4k context (4k tail + ~4k gen). Gaps to use it: (1) no DP launcher (run 2 single-card
serves on 1919/1920); (2) the RSA shim is single-backend — needs multi-backend round-robin to fan
rollouts across both replicas.
